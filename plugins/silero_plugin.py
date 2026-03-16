"""SileroPlugin — VAD via lokales ONNX (kein Docker/HF).

Modell : models/silero/silero_en_v5.onnx
ONNX   : input[batch,samples] -> output[batch,frames,999]
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: SileroPlugin | None = None
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "silero", "silero_en_v5.onnx")
_SR = 16_000
_CHUNK = 512  # 32 ms @ 16 kHz


class SileroPlugin:
    def __init__(self, model_path: str | None = None) -> None:
        self._session = None
        self._threshold = 0.5
        self._try_load(model_path or _MODEL)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Silero VAD fehlt: %s -- Energie-Fallback.", path)
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
                if not _try_alloc("SileroVAD", size_gb=0.11):
                    logger.warning("SileroVAD: ML-Budget erschöpft — Energie-Fallback.")
                    return
            except Exception:
                pass

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("Silero VAD ONNX geladen: %s", path)
        except Exception as exc:
            logger.warning("Silero Ladefehler: %s -- Energie-Fallback.", exc)

    def is_speech(self, audio: np.ndarray, sr: int) -> float:
        """Gibt Sprach-Wahrscheinlichkeit [0,1] zurueck."""
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        mono = _resamp(mono, sr, _SR).astype(np.float32)
        if self._session:
            return self._vad_onnx(mono)
        return self._energy_vad(mono)

    def get_speech_mask(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gibt bool-Array zurueck (True = Sprach-Segment)."""
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        mono16 = _resamp(mono, sr, _SR).astype(np.float32)
        mask16 = np.zeros(len(mono16), dtype=bool)
        for s in range(0, len(mono16), _CHUNK):
            e = min(s + _CHUNK, len(mono16))
            chunk = mono16[s:e]
            prob = self._vad_onnx(chunk) if self._session else self._energy_vad(chunk)
            if prob >= self._threshold:
                mask16[s:e] = True
        if sr != _SR:
            factor = len(mono) / max(len(mono16), 1)
            idx = (np.where(mask16)[0] * factor).astype(int)
            mask = np.zeros(len(mono), dtype=bool)
            valid = idx[idx < len(mono)]
            if valid.size:
                mask[valid] = True
            return mask
        return mask16[: len(mono)]

    def _vad_onnx(self, chunk: np.ndarray) -> float:
        if len(chunk) < 1:
            return 0.0
        inp = chunk[None].astype(np.float32)
        try:
            out = self._session.run(None, {"input": inp})[0]  # [1,frames,999]
            probs = out[0]  # [frames, 999]
            if probs.shape[-1] > 1:
                speech_prob = float(probs[:, 1:].max(axis=-1).mean())
            else:
                speech_prob = 0.5
            return min(max(speech_prob, 0.0), 1.0)
        except Exception as exc:
            logger.debug("Silero VAD ONNX run Fehler: %s", exc)
            return self._energy_vad(chunk)

    @staticmethod
    def _energy_vad(chunk: np.ndarray, threshold: float = 0.01) -> float:
        rms = float(np.sqrt(np.mean(np.nan_to_num(chunk, 0.0) ** 2)))
        return 1.0 if rms > threshold else 0.0


def _resamp(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def get_silero_plugin() -> SileroPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = SileroPlugin()
    return _inst


def is_speech(audio: np.ndarray, sr: int) -> float:
    """Convenience-Wrapper: Sprach-Wahrscheinlichkeit [0,1]."""
    return get_silero_plugin().is_speech(audio, sr)


# synthesize() Stub — TTS-Interface für Kompatibilität
def _silero_synthesize(self, text: str, speaker: str = "default", sr: int = 48000, **kwargs) -> bytes:
    """TTS-Stub: gibt leeres bytes-Objekt zurück (kein echtes TTS ohne Modell)."""
    import numpy as np

    samples = np.zeros(sr, dtype=np.float32)  # 1 s Stille
    return samples.tobytes()


SileroPlugin.synthesize = _silero_synthesize


# Backward-compat alias (Spec §11.3)
SileroVADPlugin = SileroPlugin
