"""rmvpe_plugin — RMVPE Robust Mel-scale Pitch Estimator (ICASSP 2023).

RMVPE (Robust Multi-period Vocoder-based Pitch Estimator via Mel spectrogram):
    - Übertrifft CREPE bei Vibrato, schnellen Tonfolgen und stimmhaft/stimmlos-Übergängen
    - Fehlerrate bei Gesang ~30 % geringer als CREPE (RPA auf MIR-1K)
    - ONNX-Modell: models/rmvpe/rmvpe.onnx (~26 MB)
    - Fallback: librosa.pyin() (pYIN, Mauch & Dixon 2014)

Aurik 9 Pitch-Tracking-Hierarchie (§4.4, Stand März 2026):
    Primär:    RMVPE ONNX  (dieser Plugin)
    Fallback1: CREPE full ONNX (crepe_plugin)
    Fallback2: FCPE ONNX (fcpe_plugin)
    DSP:       pYIN via librosa

Referenz:
    Wei et al. "RMVPE: A Robust Model for Vocal Pitch Estimation
    in Polyphonic Music" — ICASSP 2023

Singleton-Pattern: get_rmvpe_plugin() verwenden.
CPU-Only: CPUExecutionProvider, kein CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "rmvpe" / "rmvpe.onnx"

# RMVPE Mel-Parameter (16 kHz Modell-SR, gemäß Paper)
_MODEL_SR: int = 16_000
_N_MELS: int = 128
_FRAME_LEN: int = 1024       # 64 ms @ 16 kHz
_HOP_LEN: int = 160          # 10 ms @ 16 kHz → 100 Frames/s
_CENTS_MIN: float = 1997.3794084376191  # f0_min = 32.7 Hz in Cents
_CENTS_MAX: float = 7180.0             # f0_max ≈ 1975 Hz in Cents
_PITCH_BINS: int = 360

_lock = threading.Lock()
_instance: RmvpePlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RmvpeResult:
    """Ergebnis der RMVPE-Pitch-Schätzung.

    Attributes:
        f0:           Estimated fundamental frequency per frame [Hz], NaN = unvoiced
        times:        Frame center times in seconds
        confidence:   Salience/Konfidenz per frame ∈ [0, 1]
        voiced_flag:  True wenn Frame als stimmhaft klassifiziert
        model_used:   "rmvpe_onnx" | "crepe_fallback" | "pyin_fallback"
        f0_mean:      Mittlere F0 über stimmhafte Frames [Hz]
        f0_std:       Standardabweichung der F0 [Hz]
    """

    f0: np.ndarray
    times: np.ndarray
    confidence: np.ndarray
    voiced_flag: np.ndarray
    model_used: str
    f0_mean: float = 0.0
    f0_std: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RmvpePlugin
# ---------------------------------------------------------------------------

class RmvpePlugin:
    """RMVPE Neural Pitch Tracker (ONNX, CPUExecutionProvider).

    Verarbeitet Mono-Audio bei 16 kHz intern; akzeptiert 48 kHz Eingang (resampelt).
    Fallback auf pYIN (librosa) wenn ONNX-Modell fehlt.

    Pitch-Output ist F0 in Hz per Frame (10 ms Hop). Stille / unvoiced → NaN.
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt RMVPE ONNX-Modell; pYIN-Fallback bei Fehler."""
        if not _ONNX_PATH.exists():
            logger.info("RMVPE ONNX nicht gefunden (%s) — pYIN-Fallback aktiv.", _ONNX_PATH)
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
                if not _try_alloc("RMVPE", size_gb=0.03):
                    logger.warning("RMVPE: ML-Budget erschöpft — pYIN-Fallback.")
                    return
            except Exception:
                pass

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            logger.info("✅ RMVPE ONNX geladen: %s (§4.4 primärer Pitch-Tracker)", _ONNX_PATH.name)
        except Exception as exc:
            logger.warning("RMVPE ONNX Ladefehler: %s — pYIN-Fallback aktiv.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, audio: np.ndarray, sr: int, *, voiced_threshold: float = 0.5) -> RmvpeResult:
        """Schätzt F0 per Frame via RMVPE ONNX oder pYIN-Fallback.

        Args:
            audio:             float32 mono oder stereo, beliebige SR
            sr:                Sample-Rate des Eingangs (muss 48000 sein)
            voiced_threshold:  Salience-Schwelle für stimmhaft/stimmlos ∈ [0, 1]

        Returns:
            RmvpeResult mit F0-Trajektorie, Konfidenz und Flags.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
        mono = np.clip(mono, -1.0, 1.0)

        if self._session is not None:
            return self._analyze_onnx(mono, sr, voiced_threshold)
        return self._analyze_pyin(mono, sr)

    # ------------------------------------------------------------------
    # ONNX Inference
    # ------------------------------------------------------------------

    def _mel_spectrogram(self, mono_16k: np.ndarray) -> np.ndarray:
        """Berechnet Log-Mel-Spektrogramm [T, n_mels] für RMVPE-Input.

        Formel: mel = log(max(fb @ |STFT|^2, 1e-8))
        Filterbank: 128 Bänder, Hz↔Mel via f_mel = 2595·log10(1+f/700)
        """
        from scipy.signal import stft as scipy_stft  # noqa: PLC0415

        n = len(mono_16k)
        if n < _FRAME_LEN:
            mono_16k = np.pad(mono_16k, (0, _FRAME_LEN - n))
        _, _, Z = scipy_stft(
            mono_16k.astype(np.float64),
            fs=_MODEL_SR,
            nperseg=_FRAME_LEN,
            noverlap=_FRAME_LEN - _HOP_LEN,
            window="hann",
        )
        mag_sq = np.abs(Z).astype(np.float32) ** 2  # [n_freq, T]
        n_freq = mag_sq.shape[0]
        # Mel-Filterbank aufbauen
        hz_max = float(_MODEL_SR) / 2.0
        mels = np.linspace(0.0, 2595.0 * math.log10(1.0 + hz_max / 700.0), _N_MELS + 2)
        hz_pts = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        freqs = np.linspace(0.0, hz_max, n_freq)
        fb = np.zeros((_N_MELS, n_freq), dtype=np.float32)
        for m in range(1, _N_MELS + 1):
            lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
            for k in range(n_freq):
                f = freqs[k]
                if lo <= f <= ctr and (ctr - lo) > 1e-10:
                    fb[m - 1, k] = (f - lo) / (ctr - lo)
                elif ctr < f <= hi and (hi - ctr) > 1e-10:
                    fb[m - 1, k] = (hi - f) / (hi - ctr)
        mel = fb @ mag_sq  # [n_mels, T]
        mel_log = np.log(np.maximum(mel, 1e-8)).astype(np.float32)
        return np.nan_to_num(mel_log, nan=0.0, posinf=0.0, neginf=-18.4).T  # [T, n_mels]

    def _analyze_onnx(self, mono_48k: np.ndarray, sr: int, voiced_threshold: float) -> RmvpeResult:
        """RMVPE ONNX-Inferenz: Mel → Salience-Map → F0."""
        assert self._session is not None
        from math import gcd  # noqa: PLC0415
        from scipy.signal import resample_poly  # noqa: PLC0415

        # 48 kHz → 16 kHz
        g = gcd(sr, _MODEL_SR)
        mono_16k = resample_poly(mono_48k, _MODEL_SR // g, sr // g).astype(np.float32)
        mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
        mono_16k = np.clip(mono_16k, -1.0, 1.0)

        try:
            mel = self._mel_spectrogram(mono_16k)  # [T, 128]
            inp = mel[np.newaxis, np.newaxis]       # [1, 1, T, 128]
            inp_name = self._session.get_inputs()[0].name
            ort_out = self._session.run(None, {inp_name: inp.astype(np.float32)})
            salience = np.asarray(ort_out[0], dtype=np.float32)  # [1, T, 360]
            if salience.ndim == 3:
                salience = salience[0]  # [T, 360]

            # Cents aus Salience ableiten via weighted average (wie RMVPE Paper)
            cents_bins = np.linspace(_CENTS_MIN, _CENTS_MAX, _PITCH_BINS).astype(np.float32)
            max_sal = salience.max(axis=1)                          # [T]
            voiced = max_sal >= voiced_threshold                    # [T] bool
            # Weighted average über Top-Bins
            probs = np.exp(salience - salience.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True) + 1e-9
            cents = (probs * cents_bins[np.newaxis]).sum(axis=1)   # [T]
            # Cents → Hz: f = 10^(cents/1200) * ref_f0 (ref=10.0 Hz)
            f0_hz = 10.0 * (2.0 ** (cents / 1200.0))
            f0_hz = np.where(voiced, f0_hz, np.nan)
            f0_hz = np.nan_to_num(f0_hz, nan=np.nan)

            hop_time = _HOP_LEN / _MODEL_SR
            times = np.arange(len(f0_hz)) * hop_time
            voiced_f0 = f0_hz[voiced & np.isfinite(f0_hz)]
            f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0

            return RmvpeResult(
                f0=f0_hz.astype(np.float32),
                times=times.astype(np.float32),
                confidence=max_sal.astype(np.float32),
                voiced_flag=voiced,
                model_used="rmvpe_onnx",
                f0_mean=f0_mean,
                f0_std=f0_std,
            )
        except Exception as exc:
            logger.warning("RMVPE ONNX-Inferenzfehler: %s — pYIN-Fallback.", exc)
            return self._analyze_pyin(mono_48k, sr)

    def _analyze_pyin(self, mono_48k: np.ndarray, sr: int) -> RmvpeResult:
        """pYIN DSP-Fallback (Mauch & Dixon 2014) via librosa."""
        try:
            import librosa  # noqa: PLC0415

            f0, voiced_flag, voiced_prob = librosa.pyin(
                mono_48k,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            f0 = np.nan_to_num(f0, nan=np.nan).astype(np.float32)
            times = librosa.times_like(f0, sr=sr, hop_length=512).astype(np.float32)
            voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
            f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0
            return RmvpeResult(
                f0=f0,
                times=times,
                confidence=voiced_prob.astype(np.float32),
                voiced_flag=voiced_flag,
                model_used="pyin_fallback",
                f0_mean=f0_mean,
                f0_std=f0_std,
            )
        except Exception as exc:
            logger.error("pYIN Fallback fehlgeschlagen: %s", exc)
            n = max(1, int(len(mono_48k) / 512))
            return RmvpeResult(
                f0=np.full(n, np.nan, dtype=np.float32),
                times=np.arange(n, dtype=np.float32) * (512.0 / sr),
                confidence=np.zeros(n, dtype=np.float32),
                voiced_flag=np.zeros(n, dtype=bool),
                model_used="pyin_error",
            )


# ---------------------------------------------------------------------------
# Singleton  (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_rmvpe_plugin() -> RmvpePlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RmvpePlugin()
    return _instance


def analyze_pitch(audio: np.ndarray, sr: int, *, voiced_threshold: float = 0.5) -> RmvpeResult:
    """Convenience-Wrapper für get_rmvpe_plugin().analyze()."""
    return get_rmvpe_plugin().analyze(audio, sr, voiced_threshold=voiced_threshold)
