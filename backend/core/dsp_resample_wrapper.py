from __future__ import annotations

from aurik6.core.resampling_utils import resample_to_48k
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DSPResampleWrapper:
    """
    Wrapper für DSP-Module/Modelle: Stellt sicher, dass Ein- und Ausgang immer 48 kHz sind.
    Das eigentliche Modul muss eine process(audio, sr) Methode besitzen.
    """

    def __init__(self, dsp_module: Any) -> None:
        self.dsp_module = dsp_module

    @property
    def name(self) -> str:
        return self.dsp_module.__class__.__name__

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Vorverarbeitung: Resample auf 48 kHz, falls nötig
        if sr != 48000:
            logger.debug(f"[Resample-Wrapper] Eingangssignal: {sr} Hz → 48000 Hz ({self.dsp_module.__class__.__name__})")
        audio_48k, sr_48k = resample_to_48k(audio, sr)
        # Modul-Verarbeitung (immer mit 48 kHz)
        out = self.dsp_module.process(audio_48k, sr_48k)
        # Nachverarbeitung: Sicherstellen, dass Output auch 48 kHz ist
        if sr_48k != 48000:
            logger.debug(f"[Resample-Wrapper] Ausgangssignal: {sr_48k} Hz → 48000 Hz ({self.dsp_module.__class__.__name__})")
            out, _ = resample_to_48k(out, sr_48k)
        out = np.asarray(out)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0)
