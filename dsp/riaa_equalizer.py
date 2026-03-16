"""
riaa_equalizer.py - RIAA-Entzerrer für Aurik 6.0

Dieses Modul entzerrt oder simuliert die RIAA-Kennlinie für Vinyl (Stub).
"""

import logging
import warnings

import numpy as np

try:
    import onnxruntime as ort
    import torch
except ImportError:
    torch = None
    ort = None
from scipy.signal import bilinear, lfilter

logger = logging.getLogger("aurik.dsp.riaa_equalizer")
logger.setLevel(logging.INFO)


class RIAAEqualizer:
    """
    SOTA-RIAA-Entzerrer:
    - Digitale RIAA-Entzerrungskurve (apply/invert)
    - Deep-Learning-Inferenz (ONNX/Torch) als Option
    """

    def __init__(self, mode: str = "apply", model_path: str = None):
        self.mode = mode  # "apply" oder "invert"
        self.model_path = model_path
        self.model = None
        self.backend = None
        if model_path:
            if ort is not None:
                try:
                    self.model = ort.InferenceSession(model_path)
                    self.backend = "onnx"
                except Exception as e:
                    warnings.warn(f"ONNX-Modell konnte nicht geladen werden: {e}")
            elif torch is not None:
                try:
                    self.model = torch.jit.load(model_path)
                    self.backend = "torch"
                except Exception as e:
                    warnings.warn(f"Torch-Modell konnte nicht geladen werden: {e}")
            else:
                warnings.warn("Weder ONNX noch Torch verfügbar. Nur klassischer Filter nutzbar.")

    def process(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Wendet RIAA-Entzerrungskurve an (apply/invert).
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, optionale DL-Inferenz, Rückfallstrategie
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Entzerrtes Audio (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        audio_out = None
        fallback_used = False
        try:
            # Deep-Learning-Inferenz
            if self.model is not None and self.backend == "onnx":
                inp = audio.astype(np.float32)[None, None, :]
                try:
                    out = self.model.run(None, {self.model.get_inputs()[0].name: inp})[0]
                    audio_out = out.squeeze().astype(audio.dtype)
                except Exception as e:
                    logger.warning(f"ONNX-Inferenz fehlgeschlagen: {e}")
                    fallback_used = True
            elif self.model is not None and self.backend == "torch":
                try:
                    inp = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    out = self.model(inp).detach().cpu().numpy().squeeze()
                    audio_out = out.astype(audio.dtype)
                except Exception as e:
                    logger.warning(f"Torch-Inferenz fehlgeschlagen: {e}")
                    fallback_used = True
            if audio_out is None:
                # Fallback: Digitale RIAA-Entzerrung
                audio_out = self._riaa_filter(audio, sr, invert=(self.mode == "invert"))
                fallback_used = True
        except Exception as e:
            logger.error(f"Fehler bei RIAA-Entzerrung: {e}")
            audio_out = audio.copy()
            fallback_used = True

        if audit_log:
            logger.info(f"RIAAEqualizer: mode={self.mode}, fallback_used={fallback_used}")
        return audio_out.astype(audio.dtype)

    def _riaa_filter(self, audio: np.ndarray, sr: int, invert: bool = False) -> np.ndarray:
        # RIAA-Entzerrungskurven-Parameter (IEC98)
        # Zeitkonstanten: 3180us, 318us, 75us
        t1, t2, t3 = 3180e-6, 318e-6, 75e-6
        if invert:
            # Inverse RIAA (für Testzwecke oder Simulation)
            b = [1, 0, 0]
            a = [
                1,
                (1 / (2 * np.pi * t1)) + (1 / (2 * np.pi * t2)) + (1 / (2 * np.pi * t3)),
                (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t2))
                + (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t3))
                + (1 / (2 * np.pi * t2)) * (1 / (2 * np.pi * t3)),
                (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t2)) * (1 / (2 * np.pi * t3)),
            ]
        else:
            # RIAA-Entzerrung (Standard)
            b = [
                1,
                (1 / (2 * np.pi * t1)) + (1 / (2 * np.pi * t2)) + (1 / (2 * np.pi * t3)),
                (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t2))
                + (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t3))
                + (1 / (2 * np.pi * t2)) * (1 / (2 * np.pi * t3)),
                (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t2)) * (1 / (2 * np.pi * t3)),
            ]
            a = [1, (1 / (2 * np.pi * t1)) + (1 / (2 * np.pi * t2)), (1 / (2 * np.pi * t1)) * (1 / (2 * np.pi * t2))]
        # Bilinear-Transformation
        bz, az = bilinear(b, a, sr)
        filtered = lfilter(bz, az, audio)
        return filtered.astype(audio.dtype)
