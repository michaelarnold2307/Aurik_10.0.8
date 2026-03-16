"""sgmse_plugin — SGMSE+: Score-Based Generative Model for Speech Enhancement.

SGMSE+ (Richter et al. 2022) verwendet stochastische Differentialgleichungen (SDE)
zur Aufhebung von Rausch- und Hallprozessen. Überlegener Nachfolger von WPE für
kombinierte Enhancement + Dereverberation.

Verbesserung gegenüber WPE (2010):
    - SGMSE+ löst das inverse Problem via Score-Matching (p(clean|noisy))
    - Unterstützung breitbandiger 48 kHz Musikrestaurierung
    - Hallunterdrückung UND Rauschreduzierung in einem Schritt

Modell:
    models/sgmse_plus/sgmse_plus.onnx (~120 MB)
    Input:  [1, 2, n_fft//2+1, T] float32 (Real + Imag getrennt)
    Output: [1, 2, n_fft//2+1, T] float32 (denoised Real + Imag)
    Sigma:  Rauschpegel ∈ [0.01, 1.0] als skalarer Input

Fallback-Kaskade (§4.4):
    1. SGMSE+ ONNX (dieser Plugin)
    2. WPE DSP (Nara-WPE, wpe_plugin.py)

Backward-Kompatibilität:
    Alle früheren Exporte (WpePlugin, SgmsePlugin, get_wpe_plugin, …)
    bleiben erhalten und zeigen auf wpe_plugin zur Rückwärtskompatibilität.

Referenz:
    Richter et al. "Speech Enhancement and Dereverberation with Diffusion-Based
    Generative Models" — IEEE/ACM TASLP 2022.
    https://github.com/sp-uhh/sgmse

Singleton: get_sgmse_plus_plugin() verwenden.
CPU-Only: CPUExecutionProvider.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "sgmse_plus" / "sgmse_plus.onnx"

# Verarbeitungs-Konstanten (48 kHz)
_SR: int = 48_000
_N_FFT: int = 512       # 10.7 ms @ 48 kHz (typisch für SGMSE+)
_HOP: int = 128         # 2.7 ms
_WIN: int = 512

_lock_plus = threading.Lock()
_instance_plus: SGMSEPlusPlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SgmseResult:
    """Ergebnis der SGMSE+ Enhancement-Inferenz.

    Attributes:
        audio:      Bereinigtes / Dereverb-Audio, float32 ∈ [-1, 1]
        sr:         Sample-Rate (48000)
        model_used: "sgmse_plus_onnx" | "wpe_dsp_fallback"
        snr_improvement_db: Geschätzter SNR-Gewinn in dB
    """

    audio: np.ndarray
    sr: int
    model_used: str
    snr_improvement_db: float = 0.0

    def __post_init__(self) -> None:
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.clip(self.audio, -1.0, 1.0)


# ---------------------------------------------------------------------------
# SGMSEPlusPlugin
# ---------------------------------------------------------------------------

class SGMSEPlusPlugin:
    """SGMSE+ Score-Based Speech/Music Enhancement (ONNX + WPE-Fallback).

    Verarbeitet kombinierte Rausch- und Hallunterdrückung via score-basierter
    generativer Inferenz oder fällt auf WPE DSP zurück (§4.4 Spec).
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt SGMSE+ ONNX; WPE-Fallback wenn nicht verfügbar."""
        if not _ONNX_PATH.exists():
            logger.info(
                "SGMSE+ ONNX nicht gefunden (%s) — WPE-DSP-Fallback aktiv. "
                "Modell: https://github.com/sp-uhh/sgmse",
                _ONNX_PATH,
            )
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
                if not _try_alloc("SGMSE+", size_gb=0.12):
                    logger.warning("SGMSE+: ML-Budget erschöpft — WPE-DSP-Fallback.")
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
            logger.info("✅ SGMSE+ ONNX geladen (%s, §4.4 — WPE-Nachfolger)", _ONNX_PATH.name)
        except Exception as exc:
            logger.warning("SGMSE+ ONNX nicht ladbar: %s — WPE-DSP-Fallback aktiv.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, audio: np.ndarray, sr: int, sigma: float = 0.5) -> SgmseResult:
        """Kombinierte Rausch-/Hallunterdrückung via SGMSE+ oder WPE-Fallback.

        Algorithm (ONNX-Pfad):
            1. STFT → Real/Imag [1, 2, F, T]
            2. SGMSE+ forward: score-basiertes Denoising bei Sigma σ
               (Ornstein–Uhlenbeck SDE: dx = -½βx dt + √β dW, t ∈ [0,1])
            3. ISTFT aus Enhanced Complex Spektrum

        Args:
            audio: float32, 48000 Hz, mono oder stereo
            sr:    Sample-Rate (muss 48000 sein)
            sigma: Rauschpegel-Schätzung ∈ [0.01, 1.0]. Standard 0.5 (adaptiv).

        Returns:
            SgmseResult mit bereinigtem Audio.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stereo = audio.ndim == 2 and audio.shape[1] == 2

        def process_channel(ch: np.ndarray) -> np.ndarray:
            if self._session is not None:
                return self._enhance_onnx(ch, sigma)
            return self._wpe_fallback(ch, sr)

        if stereo:
            left = process_channel(audio[:, 0])
            right = process_channel(audio[:, 1])
            n = min(len(left), len(right), len(audio))
            out = np.stack([left[:n], right[:n]], axis=1)
        else:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            out = process_channel(mono)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        rms_in = float(np.sqrt(np.mean(audio ** 2))) + 1e-10
        rms_diff = float(np.sqrt(np.mean((out - audio) ** 2))) + 1e-10
        snr_imp = 20.0 * math.log10(rms_in / rms_diff) if rms_diff < rms_in else 0.0

        return SgmseResult(
            audio=out.astype(np.float32),
            sr=sr,
            model_used="sgmse_plus_onnx" if self._session is not None else "wpe_dsp_fallback",
            snr_improvement_db=float(np.clip(snr_imp, 0.0, 30.0)),
        )

    # ------------------------------------------------------------------
    # ONNX Inference (SGMSE+ deterministic forward pass @ optimal sigma)
    # ------------------------------------------------------------------

    def _stft(self, mono: np.ndarray) -> tuple[np.ndarray, int]:
        """STFT → Complex Spectrogram."""
        from scipy.signal import stft as scipy_stft  # noqa: PLC0415

        n_orig = len(mono)
        _, _, Z = scipy_stft(
            mono.astype(np.float64),
            fs=_SR,
            nperseg=_WIN,
            noverlap=_WIN - _HOP,
            window="hann",
        )
        return Z.astype(np.complex64), n_orig

    def _istft(self, Z: np.ndarray, n_orig: int) -> np.ndarray:
        """Inverse STFT."""
        from scipy.signal import istft as scipy_istft  # noqa: PLC0415

        _, x = scipy_istft(
            Z.astype(np.complex128),
            fs=_SR,
            nperseg=_WIN,
            noverlap=_WIN - _HOP,
            window="hann",
        )
        x = x.astype(np.float32)
        if len(x) > n_orig:
            x = x[:n_orig]
        elif len(x) < n_orig:
            x = np.pad(x, (0, n_orig - len(x)))
        return x

    def _enhance_onnx(self, mono: np.ndarray, sigma: float) -> np.ndarray:
        """SGMSE+ ONNX-Inferenz: Score-Based Enhancement."""
        assert self._session is not None
        try:
            Z, n_orig = self._stft(mono)
            # SGMSE+ input: [1, 2, F, T] — Real und Imag als separate Kanäle
            real_c = Z.real[np.newaxis, np.newaxis].astype(np.float32)
            imag_c = Z.imag[np.newaxis, np.newaxis].astype(np.float32)
            inp = np.concatenate([real_c, imag_c], axis=1)  # [1, 2, F, T]

            # Sigma als skalarer Input (falls Modell diesen Eingang erwartet)
            input_names = [i.name for i in self._session.get_inputs()]
            feed: dict[str, np.ndarray] = {input_names[0]: inp}
            if len(input_names) > 1:
                feed[input_names[1]] = np.array([[[[sigma]]]], dtype=np.float32)

            ort_out = self._session.run(None, feed)
            out_arr = np.asarray(ort_out[0], dtype=np.float32)  # [1, 2, F, T]

            out_real = out_arr[0, 0] if out_arr.shape[1] >= 2 else out_arr[0, 0]
            out_imag = out_arr[0, 1] if out_arr.shape[1] >= 2 else np.zeros_like(out_real)

            Z_enhanced = (out_real + 1j * out_imag).astype(np.complex64)
            Z_enhanced = np.nan_to_num(Z_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

            result = self._istft(Z_enhanced, n_orig)
            return np.clip(np.nan_to_num(result, nan=0.0), -1.0, 1.0)
        except Exception as exc:
            logger.warning("SGMSE+ ONNX-Inferenzfehler: %s — WPE-Fallback.", exc)
            return self._wpe_fallback(mono, _SR)

    # ------------------------------------------------------------------
    # WPE DSP Fallback
    # ------------------------------------------------------------------

    def _wpe_fallback(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """WPE-Dereverberation als Fallback (wpe_plugin, §4.4)."""
        try:
            from plugins.wpe_plugin import get_wpe_plugin  # noqa: PLC0415

            plugin = get_wpe_plugin()
            result = plugin.enhance(mono, sr)
            if hasattr(result, "audio"):
                return np.clip(np.nan_to_num(result.audio.flatten(), nan=0.0), -1.0, 1.0)
            # Legacy: result ist ndarray
            arr = np.asarray(result, dtype=np.float32).flatten()
            return np.clip(np.nan_to_num(arr, nan=0.0), -1.0, 1.0)
        except Exception as exc:
            logger.error("WPE-Fallback fehlgeschlagen: %s — Audio unverändert.", exc)
            return np.clip(np.nan_to_num(mono.copy(), nan=0.0), -1.0, 1.0)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_sgmse_plus_plugin() -> SGMSEPlusPlugin:
    """Thread-sicherer Singleton-Accessor für SGMSE+."""
    global _instance_plus
    if _instance_plus is None:
        with _lock_plus:
            if _instance_plus is None:
                _instance_plus = SGMSEPlusPlugin()
    return _instance_plus


def enhance_sgmse(audio: np.ndarray, sr: int, sigma: float = 0.5) -> SgmseResult:
    """Convenience-Wrapper für get_sgmse_plus_plugin().enhance()."""
    return get_sgmse_plus_plugin().enhance(audio, sr, sigma)


# ---------------------------------------------------------------------------
# Backward-Kompatibilität: WPE-Exporte aus wpe_plugin re-exportieren
# ---------------------------------------------------------------------------
# ruff: noqa: F401
from plugins.wpe_plugin import (  # noqa: F401, E402
    SGMSEPlugin,
    SgmsePlugin,
    WpePlugin,
    _omlsa_fallback,
    _wpe_nara,
    _wpe_numpy,
    _wpe_stft,
    enhance,
    get_sgmse_plugin,
    get_wpe_plugin,
)

__all__ = [
    # Neue SGMSE+-Implementierung
    "SGMSEPlusPlugin",
    "SgmseResult",
    "get_sgmse_plus_plugin",
    "enhance_sgmse",
    # Backward-Kompatibilität (WPE-Basis)
    "WpePlugin",
    "SgmsePlugin",
    "SGMSEPlugin",
    "get_wpe_plugin",
    "get_sgmse_plugin",
    "enhance",
]
