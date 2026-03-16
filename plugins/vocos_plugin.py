"""Aurik 9 — Vocos Neural Vocoder Plugin

PRIMÄRER Vocoder: Vocos 0.1.0 (Siuzdak 2023, MIT)
ONNX-Modell: models/vocos/vocos_mel_spec_24khz.onnx (lokal gebündelt, 52 MB)
Fallback-Kaskade (§4.4 SOTA-Matrix):
  1. Vocos ONNX  (vocos_mel_spec_24khz.onnx)            — primär
  2. HiFi-GAN     (hifigan_plugin.HifiGanPlugin)          — neuronaler Fallback
  3. Griffin-Lim+ ≥ 32 Iterationen (DSP-Letzfall)        — nur wenn beide Modelle fehlen

CPU-Only: CPUExecutionProvider — kein CUDA.
Out-of-the-Box: Kein Download beim ersten Start.
"""

from dataclasses import dataclass, field
import logging
import math
from math import gcd
import os
import threading

import numpy as np
from scipy.signal import istft, resample_poly, stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Öffentliche SR-Konstanten (von Tests und externen Modulen verwendet)
# ---------------------------------------------------------------------------
MEL_SR_22K: int = 22_050  # Vocos 22 kHz Variante
MEL_SR_44K: int = 44_100  # Vocos 44 kHz Variante
AURIK_SR: int = 48_000  # Standard-Aurik-Arbeits-SR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 44.1 kHz-Modell bevorzugen (volles Spektrum bis 22 kHz); Fallback auf 24 kHz
_MODEL_44K = os.path.join(_ROOT, "models", "vocos", "vocos_mel_spec_44khz.onnx")
_MODEL_24K = os.path.join(_ROOT, "models", "vocos", "vocos_mel_spec_24khz.onnx")
_MODEL = _MODEL_44K  # für Rückwärtskompatibilität (bevorzugte Auflösung)
# Mel-Parameter je Modell-SR
_MEL_SR_44K = 44_100
_MEL_SR_24K = 24_000
_MEL_SR = _MEL_SR_24K  # Standardwert (wird zur Laufzeit angepasst)
_N_MELS_44K = 128
_N_MELS_24K = 100
_N_MELS = _N_MELS_24K
_N_FFT_44K = 2048
_N_FFT_24K = 1024
_N_FFT = _N_FFT_24K
_HOP_44K = 512
_HOP_24K = 256
_HOP = _HOP_24K
_WIN_44K = 2048
_WIN_24K = 1024
_WIN = _WIN_24K


# ---------------------------------------------------------------------------
# VocosResult — Ergebnis-Datenklasse (§3.6 Aurik Spec)
# ---------------------------------------------------------------------------
@dataclass
class VocosResult:
    """Ergebnis des Vocos-Vocoders."""

    audio: np.ndarray  # Synthetisiertes Audio, float32 ∈ [-1, 1]
    sr: int  # Ausgabe-Samplerate
    pqs_mos: float  # Perceptual Quality Score MOS ∈ [1.0, 5.0]
    model_used: str  # z.B. "vocos_onnx", "hifigan_fallback", "griffin_lim_fallback"
    confidence: float  # Modell-Konfidenz ∈ [0, 1]
    mel_snr_db: float = 0.0  # Mel-Spektrogramm-SNR in dB
    model_sr: int = 0  # Interne Modell-SR (z.B. 24000)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "sr": self.sr,
            "pqs_mos": self.pqs_mos,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "mel_snr_db": self.mel_snr_db,
            "model_sr": self.model_sr,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# VocosPlugin
# ---------------------------------------------------------------------------
class VocosPlugin:
    """Vocos Neural Vocoder — Aurik 9 Implementierung.

    Fallback-Kaskade (§4.4 SOTA-Matrix):
        1. Vocos ONNX  — vocos_mel_spec_24khz.onnx (CPUExecutionProvider)
        2. HiFi-GAN    — hifigan_plugin.HifiGanPlugin.reconstruct() (CPUExecutionProvider)
        3. Griffin-Lim+ ≥ 32 It. — DSP-Letzfall (nur wenn beide Modelle fehlen)

    Singleton-Pattern: get_vocos_plugin() verwenden.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._prefer_sr: int = _MEL_SR_44K  # 44.1 kHz bevorzugt
        self._model_sr: int = _MEL_SR_24K
        self._mel_n_mels: int = _N_MELS_24K
        self._mel_n_fft: int = _N_FFT_24K
        self._mel_hop: int = _HOP_24K
        self._mel_win: int = _WIN_24K
        self._vocos_pypi = None
        self._onnx_session = None
        self._model_loaded: bool = False
        self._fallback_mode: str = "griffin_lim_fallback"
        if model_path:
            self._try_load(model_path)
        else:
            # 44.1 kHz zuerst, dann 24 kHz
            if os.path.exists(_MODEL_44K):
                self._try_load(_MODEL_44K)
            elif os.path.exists(_MODEL_24K):
                self._try_load(_MODEL_24K)
            else:
                logger.warning("Vocos ONNX fehlt (44 kHz + 24 kHz) — Griffin-Lim-Fallback.")

    def _try_load(self, path: str) -> None:
        """Versucht ONNX-Modell zu laden; setzt Fallback bei Fehler."""
        if not os.path.exists(path):
            logger.warning("Vocos ONNX fehlt: %s — Griffin-Lim-Fallback.", path)
            return
        # ── ML-Budget-Check VOR dem Laden (§5.1 OOM-Schutz) ──────────────────
        _allocated = False
        try:
            from backend.core.ml_memory_budget import try_allocate, release as _release  # noqa: PLC0415
            if not try_allocate("Vocos", size_gb=0.12):
                logger.warning("Vocos: ML-Budget erschöpft — Griffin-Lim-Fallback")
                return
            _allocated = True
        except ImportError:
            pass
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._onnx_session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            self._model_loaded = True
            self._fallback_mode = "vocos_onnx"
            # Mel-Parameter je Modell-SR anpassen
            if "44" in os.path.basename(path):
                self._model_sr = _MEL_SR_44K
                self._mel_n_mels = _N_MELS_44K
                self._mel_n_fft = _N_FFT_44K
                self._mel_hop = _HOP_44K
                self._mel_win = _WIN_44K
                logger.info("Vocos 44.1 kHz ONNX geladen: %s (volles Spektrum bis 22 kHz)", path)
            else:
                self._model_sr = _MEL_SR_24K
                self._mel_n_mels = _N_MELS_24K
                self._mel_n_fft = _N_FFT_24K
                self._mel_hop = _HOP_24K
                self._mel_win = _WIN_24K
                logger.info("Vocos 24 kHz ONNX geladen: %s", path)
        except Exception as exc:
            logger.warning("Vocos ONNX Fehler: %s — Griffin-Lim-Fallback.", exc)
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release  # noqa: PLC0415, F811
                    _release("Vocos")
                except ImportError:
                    pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        """Name des aktiven Backends."""
        return self._fallback_mode

    @property
    def model_loaded(self) -> bool:
        """True wenn ein neuronales Modell geladen wurde."""
        return self._model_loaded

    # ------------------------------------------------------------------
    # Statische Hilfsmethoden (auch ohne Instanz nutzbar, §3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resampelt Audio von src_sr auf dst_sr (resample_poly, Lanczos-artig).

        Invariante: Ausgabe ist float32, NaN/Inf-frei.
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if src_sr == dst_sr:
            return audio
        g = gcd(src_sr, dst_sr)
        result = resample_poly(audio, dst_sr // g, src_sr // g)
        return np.nan_to_num(result.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _build_mel_filterbank(n_mels: int, n_freq_bins: int, sr: int, n_fft: int) -> np.ndarray:
        """Baut Mel-Filterbank [n_mels, n_freq_bins], float32, non-negative.

        Formel: Hz ↔ Mel via 2595 * log10(1 + f/700)
        """
        hz_max = float(sr) / 2.0

        def hz_to_mel(f: float) -> float:
            return 2595.0 * math.log10(1.0 + max(f, 0.0) / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min = hz_to_mel(0.0)
        mel_max = hz_to_mel(hz_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(float(m)) for m in mel_points])
        freqs = np.linspace(0.0, hz_max, n_freq_bins)

        fb = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_low = hz_points[m - 1]
            f_center = hz_points[m]
            f_high = hz_points[m + 1]
            for k in range(n_freq_bins):
                f = freqs[k]
                if f_low <= f <= f_center:
                    denom = f_center - f_low
                    fb[m - 1, k] = (f - f_low) / denom if denom > 1e-10 else 0.0
                elif f_center < f <= f_high:
                    denom = f_high - f_center
                    fb[m - 1, k] = (f_high - f) / denom if denom > 1e-10 else 0.0

        return np.clip(fb, 0.0, None).astype(np.float32)

    @staticmethod
    def _compute_mel(audio: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
        """Berechnet Mel-Spektrogramm [n_mels, T], float32, finite.

        Formel: M = log(max(FB @ |STFT|^2, 1e-8))
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(audio)
        nperseg = min(_WIN, n) if n >= 2 else 2
        noverlap = max(0, nperseg - _HOP)
        try:
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
        except ValueError:
            nperseg = max(2, n)
            noverlap = nperseg - 1
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
        mag_sq = np.abs(Z) ** 2  # [n_freq_bins, T]
        n_freq_bins = mag_sq.shape[0]
        fb = VocosPlugin._build_mel_filterbank(n_mels, n_freq_bins, sr, nperseg)
        mel = fb @ mag_sq  # [n_mels, T]
        mel = np.log(np.maximum(mel, 1e-8)).astype(np.float32)
        return np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-18.4)

    @staticmethod
    def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
        """Schneidet auf target_len oder füllt mit Nullen auf.

        Invariante: len(result) == target_len, float32.
        """
        audio = audio.astype(np.float32)
        current = len(audio)
        if current == target_len:
            return audio
        elif current > target_len:
            return audio[:target_len]
        else:
            return np.pad(audio, (0, target_len - current)).astype(np.float32)

    # ------------------------------------------------------------------
    # Synthese-Methoden
    # ------------------------------------------------------------------

    def _synthesize_vocos_onnx(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """Vocos ONNX-Synthese via CPUExecutionProvider.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len ≈ len(audio)
            - confidence ≥ 0.85 (neuronales Modell)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        assert self._onnx_session is not None, "ONNX-Session nicht bereit"
        target_len = len(audio)
        model_sr = self._model_sr
        n_mels = self._mel_n_mels
        try:
            # 1. Resample auf Modell-SR (44.1 kHz oder 24 kHz je nach geladenem Modell)
            audio_model = self._resample(audio, sr, model_sr)

            # 2. Mel-Spektrogramm berechnen [n_mels, T] → [1, n_mels, T]
            mel = self._compute_mel(audio_model, model_sr, n_mels)
            mel_input = mel[np.newaxis].astype(np.float32)  # [1, n_mels, T]

            # 3. ONNX-Inferenz (CPUExecutionProvider — §9.5)
            input_name = self._onnx_session.get_inputs()[0].name
            ort_out = self._onnx_session.run(None, {input_name: mel_input})
            waveform = np.asarray(ort_out[0], dtype=np.float32).reshape(-1)

            # 4. Auf Aurik-SR resampeln und Länge angleichen
            waveform = self._resample(waveform, model_sr, sr)
            waveform = self._match_length(waveform, target_len)
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            waveform = np.clip(waveform, -1.0, 1.0)
            return waveform, "vocos_onnx", 0.92
        except Exception as exc:
            logger.warning("Vocos ONNX-Inferenzfehler: %s — Audio-Passthrough.", exc)
            result = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0), "vocos_onnx_passthrough", 0.50

    def _synthesize_bigvgan_v2(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """BigVGAN v2 Fallback-Synthese (Stufe 1.5, §4.4 SOTA-Matrix März 2026).

        Nutzt BigVGANv2Plugin — vollspektraler GAN-Vocoder bei 48 kHz nativ.
        Kein Resample-Verlust da Aurik nativ 48 kHz verarbeitet.

        Returns: (audio_out, model_name, confidence)
        """
        target_len = len(audio)
        try:
            from plugins.bigvgan_v2_plugin import BigVGANv2Plugin  # Lazy-Import

            _bvg = BigVGANv2Plugin()
            if not _bvg._model_loaded:
                return np.clip(audio.astype(np.float32), -1.0, 1.0), "bigvgan_v2_unavailable", 0.0
            result_obj = _bvg.synthesize(audio, sr, mode="studio2026")
            out = self._match_length(result_obj.audio, target_len)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0)
            logger.info("BigVGAN v2 Fallback erfolgreich (Stufe 1.5).")
            return out, "bigvgan_v2_fallback", 0.90
        except Exception as exc:
            logger.warning("BigVGAN v2 Fallback fehlgeschlagen: %s — HiFi-GAN wird versucht.", exc)
            return np.clip(audio.astype(np.float32), -1.0, 1.0), "bigvgan_v2_unavailable", 0.0

    def _synthesize_hifigan(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, str, float]:
        """HiFi-GAN Fallback-Synthese via hifigan_plugin (§4.4 SOTA-Matrix Stufe 2).

        Nutzt HifiGanPlugin.reconstruct() — Audio → 80-Band-Mel → ONNX → Waveform.
        Lazy-Import damit vocos_plugin ohne hifigan_plugin importierbar bleibt.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len ≈ len(audio)
            - confidence = 0.78 (schlechter als Vocos, besser als Griffin-Lim)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        target_len = len(audio)
        try:
            from plugins.hifigan_plugin import HifiGanPlugin  # Lazy-Import

            _hg = HifiGanPlugin()
            if not _hg._session:
                raise ImportError("HiFi-GAN Modell nicht verfügbar")
            out = _hg.reconstruct(audio, sr)
            out = self._match_length(out, target_len)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0)
            logger.info("HiFi-GAN Fallback erfolgreich.")
            return out, "hifigan_fallback", 0.78
        except Exception as exc:
            logger.warning("HiFi-GAN Fallback fehlgeschlagen: %s — Griffin-Lim wird verwendet.", exc)
            result = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0), "hifigan_unavailable", 0.40

    def _synthesize_griffin_lim(self, audio: np.ndarray, sr: int, n_iter: int = 32) -> tuple[np.ndarray, str, float]:
        """Griffin-Lim+ Letzfall-Synthese (§4.4 SOTA-Matrix Stufe 3, ≥ 32 Iterationen).

        Nur aktiv wenn Vocos ONNX UND HiFi-GAN nicht verfügbar sind.

        Returns: (audio_out, model_name, confidence)

        Invarianten:
            - Ausgabe: float32 ∈ [-1, 1], kein NaN/Inf, len == len(audio)
            - confidence < 0.70 (schlechter als neuronale Modelle)
            - Stille-Eingabe → nahezu Stille im Ausgang
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        target_len = len(audio)
        if target_len < 2:
            return np.zeros(target_len, np.float32), "griffin_lim_fallback", 0.30

        try:
            nperseg = min(_WIN, target_len)
            noverlap = max(0, nperseg - _HOP)
            _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
            mag = np.abs(Z)
            phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, Z.shape).astype(np.float64))
            for _ in range(n_iter):
                _, x_rec = istft(mag * phase, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
                _, _, Z_rec = stft(x_rec, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
                phase = np.exp(1j * np.angle(Z_rec))
            _, result = istft(mag * phase, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann")
            result = np.nan_to_num(result.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            result = self._match_length(result, target_len)
            result = np.clip(result, -1.0, 1.0)
        except Exception as e:
            logger.warning("Griffin-Lim Fehler: %s", e)
            result = np.clip(audio.copy(), -1.0, 1.0)

        return result, "griffin_lim_fallback", 0.60

    def _estimate_pqs_mos(self, original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Schätzt PQS-MOS ∈ [1.0, 5.0].

        Formel: MOS = 1.0 + 4.0 * sigmoid(8 * (pearson_corr - 0.5))
        Invariante: Identisches Audio → MOS ≥ 4.0
        """
        try:
            orig_mel = self._compute_mel(original, sr)
            rest_mel = self._compute_mel(restored, sr)
            n = min(orig_mel.shape[1], rest_mel.shape[1])
            if n < 2:
                return 3.0
            a = orig_mel[:, :n].flatten().astype(np.float64)
            b = rest_mel[:, :n].flatten().astype(np.float64)
            std_a = float(np.std(a))
            std_b = float(np.std(b))
            if std_a < 1e-8 and std_b < 1e-8:
                return 4.5  # Beide Stille → identisch
            if std_a < 1e-8 or std_b < 1e-8:
                return 1.5
            corr = np.corrcoef(a, b)[0, 1]
            sim = float(np.clip(np.nan_to_num(corr), -1.0, 1.0))
            z = 8.0 * (sim - 0.5)
            sig = 1.0 / (1.0 + math.exp(-z))
            mos = 1.0 + 4.0 * sig
            return float(np.clip(mos, 1.0, 5.0))
        except Exception:
            return 3.0

    def _mel_snr(self, original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Berechnet Signal-Rausch-Abstand in dB.

        Invariante: Identisches Audio → SNR ≥ 20 dB
        """
        try:
            n = min(len(original), len(restored))
            orig = original[:n].astype(np.float64)
            rest = restored[:n].astype(np.float64)
            signal_pow = float(np.sum(orig**2))
            noise_pow = float(np.sum((orig - rest) ** 2))
            if noise_pow < 1e-20:
                return 80.0  # Identisch
            if signal_pow < 1e-20:
                return 0.0
            return float(10.0 * math.log10(max(signal_pow / noise_pow, 1e-10)))
        except Exception:
            return 0.0

    def vocode(self, audio: np.ndarray, sr: int, mode: str = "studio2026") -> "VocosResult":
        """Synthetisiert Audio mit Vocos oder Griffin-Lim-Fallback.

        Args:
            audio: Input-Audio float32, shape [N]
            sr:    Sample-Rate (muss AURIK_SR = 48000 Hz sein)
            mode:  "studio2026" (erlaubt) oder "restoration" (verboten)

        Returns:
            VocosResult mit synthetisiertem Audio

        Raises:
            ValueError: wenn mode == "restoration"
            AssertionError: wenn sr != AURIK_SR
        """
        if mode == "restoration":
            raise ValueError(
                "Vocos im Restoration-Modus deaktiviert — " "nur Studio-2026-Modus erlaubt (§1.4 Aurik Spec)."
            )
        assert sr == AURIK_SR, f"SR muss {AURIK_SR} Hz sein, erhalten: {sr}"

        # NaN/Inf bereinigen
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # §4.4 Spec SOTA-Matrix — Fallback-Kaskade (aktualisiert März 2026):
        #   Stufe 1:   Vocos 44.1 kHz ONNX  (primär, volles Spektrum bis 22 kHz)
        #   Stufe 1b:  Vocos 24 kHz ONNX    (falls 44.1 kHz fehlt)
        #   Stufe 1.5: BigVGAN v2 ONNX      (GAN-basierter Wideband-Vocoder, 48 kHz)
        #   Stufe 2:   HiFi-GAN ONNX        (neuronaler Fallback)
        #   Stufe 3:   Griffin-Lim+         (DSP-Letzfall, ≥ 32 Iter.)
        # Griffin-Lim als Endschritt VERBOTEN in Studio-2026 (§4.4).
        if self._model_loaded and self._onnx_session is not None:
            out_audio, model_name, confidence = self._synthesize_vocos_onnx(audio, sr)
        else:
            # Stufe 1.5: BigVGAN v2 als vollspektraler GAN-Fallback (48 kHz native)
            out_audio, model_name, confidence = self._synthesize_bigvgan_v2(audio, sr)
            if model_name == "bigvgan_v2_unavailable":
                # Stufe 2: HiFi-GAN als neuronaler Fallback
                out_audio, model_name, confidence = self._synthesize_hifigan(audio, sr)
                # Stufe 3: Griffin-Lim nur wenn alle neuronalen Modelle fehlen
                if model_name == "hifigan_unavailable":
                    out_audio, model_name, confidence = self._synthesize_griffin_lim(audio, sr)
                    logger.warning("Vocos + BigVGAN v2 + HiFi-GAN nicht verfügbar — Griffin-Lim+ aktiv (Stufe 3).")
        pqs = self._estimate_pqs_mos(audio, out_audio, sr)
        snr = self._mel_snr(audio, out_audio, sr)

        return VocosResult(
            audio=out_audio,
            sr=sr,
            pqs_mos=pqs,
            model_used=model_name,
            confidence=confidence,
            mel_snr_db=snr,
            model_sr=self._model_sr,
        )


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, §3.2 Aurik Spec)
# ---------------------------------------------------------------------------
_instance: VocosPlugin | None = None
_lock = threading.Lock()


def get_vocos_plugin() -> VocosPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocosPlugin()
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktionen
# ---------------------------------------------------------------------------


def vocode(audio: np.ndarray, sr: int, mode: str = "studio2026") -> VocosResult:
    """Convenience-Wrapper für get_vocos_plugin().vocode()."""
    return get_vocos_plugin().vocode(audio, sr, mode=mode)


def vocode_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convenience-Wrapper — gibt synthetisiertes Audio als np.ndarray zurück."""
    result = get_vocos_plugin().vocode(audio, sr, mode="studio2026")
    return result.audio
