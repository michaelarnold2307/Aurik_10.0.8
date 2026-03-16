"""ConvTasNetPlugin — Quellentrennung via HPSS-DSP (kein Docker/HF).
Kein ONNX-Modell verfuegbar – reiner DSP-Fallback (librosa HPSS).
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: ConvTasNetPlugin | None = None


class ConvTasNetPlugin:
    def __init__(self) -> None:
        logger.info("ConvTasNetPlugin: HPSS-DSP-Modus (kein ONNX vorhanden).")

    def separate(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Gibt (harmonic, percussive) als float32-Arrays zurueck."""
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
        return (np.clip(harm, -1.0, 1.0), np.clip(perc, -1.0, 1.0))

    def separate_vocals(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        harm, perc = self.separate(audio, sr)
        return harm, perc  # harm ~ Vokal-Proxy


def get_convtasnet_plugin() -> ConvTasNetPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = ConvTasNetPlugin()
    return _inst


def separate(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    return get_convtasnet_plugin().separate(audio, sr)
