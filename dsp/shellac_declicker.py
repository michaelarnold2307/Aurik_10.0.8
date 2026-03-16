"""
shellac_declicker.py - Schellack-Declicker für Aurik 6.0

Dieses Modul entfernt grobe Klicks/Knackser von Schellackplatten (Stub).
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

logger = logging.getLogger("aurik.dsp.shellac_declicker")
logger.setLevel(logging.INFO)


class ShellacDeclicker:
    """
    SOTA-Declicker für Schellackplatten:
    - Deep-Learning-Inferenz (ONNX/Torch) für Declicking
    - Klassische Pulsdetektion als Fallback
    """

    def __init__(self, sensitivity: float = 1.0, model_path: str = None):
        self.sensitivity = sensitivity
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
                warnings.warn("Weder ONNX noch Torch verfügbar. Nur klassische Pulsdetektion nutzbar.")

    def process(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt Klicks/Knackser von Schellackplatten.
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung, optionale DL-Inferenz, Rückfallstrategie
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Deklicktes Audio (np.ndarray)
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
                # Fallback: Klassische Pulsdetektion
                from scipy.signal import medfilt

                diff = np.abs(audio - medfilt(audio, kernel_size=7))
                mask = diff > self.sensitivity * np.max(diff)
                audio_out = audio.copy()
                if np.any(mask):
                    idx = np.where(mask)[0]
                    for i in idx:
                        left = max(0, i - 3)
                        right = min(len(audio) - 1, i + 3)
                        audio_out[i] = np.median(audio_out[left : right + 1])
                fallback_used = True
        except Exception as e:
            logger.error(f"Fehler beim Declicking: {e}")
            audio_out = audio.copy()
            fallback_used = True

        if audit_log:
            declicking_error = float(np.mean(np.abs(audio - audio_out)))
            logger.info(f"ShellacDeclicker: declicking_error={declicking_error:.4f}, fallback_used={fallback_used}")
        return audio_out.astype(audio.dtype)
