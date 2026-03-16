"""WaveUNetPlugin — Quellentrennung via DSP-Stub (kein Docker/HF).
Kein ONNX-Modell verfuegbar – HPSS-DSP-Fallback.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: WaveUNetPlugin | None = None


class WaveUNetPlugin:
    def __init__(self) -> None:
        logger.info("WaveUNetPlugin: HPSS-DSP-Stub (kein ONNX vorhanden).")

    def separate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        mono = mono.astype(np.float32)
        try:
            import librosa

            harm, perc = librosa.effects.hpss(mono)
            harm = np.nan_to_num(harm, nan=0.0, posinf=0.0, neginf=0.0)
            perc = np.nan_to_num(perc, nan=0.0, posinf=0.0, neginf=0.0)
        except ImportError:
            harm = mono.copy()
            perc = np.zeros_like(mono)
        return np.clip(harm, -1.0, 1.0), np.clip(perc, -1.0, 1.0)


def get_waveunet_plugin() -> WaveUNetPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = WaveUNetPlugin()
    return _inst


def separate(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    return get_waveunet_plugin().separate(audio, sr)
