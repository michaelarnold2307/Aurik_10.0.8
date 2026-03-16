"""ResembleEnhancePlugin — Audio-Enhancement via lokales ONNX (kein Docker/HF).

Modell : models/resemble_enhance/model.onnx
ONNX   : mag+cos+sin[batch,841,T] → out_mag+out_cos+out_sin
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: ResembleEnhancePlugin | None = None
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "resemble_enhance", "model.onnx")
_SR = 44_100
_N = 1680
_HOP = 420
_BINS = 841


class ResembleEnhancePlugin:
    def __init__(self, model_path: str | None = None) -> None:
        self._session = None
        self._try_load(model_path or _MODEL)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Resemble-Enhance Modell fehlt: %s — DSP-Fallback.", path)
            return
        # ML-Budget-Guard: Resemble-Enhance model.onnx ~722 MB
        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc, release as _rel  # noqa: PLC0415
            if not _try_alloc("ResembleEnhance", size_gb=0.72):
                logger.warning("Resemble-Enhance: ML-Budget erschöpft — DSP-Fallback.")
                return
        except Exception:
            _rel = None
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("Resemble-Enhance ONNX geladen: %s", path)
        except Exception as exc:
            logger.warning("Resemble-Enhance Ladefehler: %s — DSP-Fallback.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release  # noqa: PLC0415
                _release("ResembleEnhance")
            except Exception:
                pass

    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n = len(mono)
        m44 = _resamp(mono, sr, _SR)
        out = self._onnx(m44) if self._session else _wiener(m44, _SR)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        result = _resamp(out, _SR, sr)[:n]
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            result = np.stack([result, result], axis=1)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def process(self, input_path: str, output_path: str, denoise_level: float = 0.5, enhance_level: float = 1.0):
        """Datei-basierte Verarbeitung für hybrid_ml_denoiser Kompatibilität.

        Returns:
            (returncode, stdout, stderr) Tuple — 0 bei Erfolg
        """
        import numpy as np
        import soundfile as sf

        try:
            audio, sr = sf.read(input_path)
            if audio.ndim == 2:
                audio_mono = audio.mean(axis=1)
            else:
                audio_mono = audio
            result = self.enhance(audio_mono.astype(np.float32), sr)
            # Ergebnis zurück in Stereo konvertieren wenn nötig
            if audio.ndim == 2:
                result_out = np.stack([result, result], axis=1)
            else:
                result_out = result
            sf.write(output_path, result_out, sr)
            return (0, "OK", "")
        except Exception as e:
            return (1, "", str(e))

    def _onnx(self, mono: np.ndarray) -> np.ndarray:
        win = np.hanning(_N).astype(np.float32)
        nf = max(1, (len(mono) + _HOP - 1) // _HOP)
        buf = np.zeros(nf * _HOP + _N, np.float32)
        buf[: len(mono)] = mono
        frames = [np.fft.rfft(buf[i * _HOP : i * _HOP + _N] * win)[:_BINS] for i in range(nf)]
        spec = np.array(frames, np.complex64).T  # [841,F]
        mag = np.abs(spec)
        cos = spec.real / (mag + 1e-8)
        sin_v = spec.imag / (mag + 1e-8)
        # Verarbeite ganzes Spektrum in einem Batch
        inp = lambda a: a[None].astype(np.float32)  # [1,841,F]
        try:
            outs = self._session.run(None, {"mag": inp(mag), "cos": inp(cos), "sin": inp(sin_v)})
            om, oc, os_ = outs[0][0], outs[1][0], outs[2][0]
            out_spec = om * (oc + 1j * os_)
        except Exception as exc:
            logger.debug("Resemble-Enhance run Fehler: %s", exc)
            out_spec = spec
        n_out = nf * _HOP + _N
        res = np.zeros(n_out, np.float32)
        ws = np.zeros(n_out, np.float32)
        full = np.zeros((_N // 2 + 1, nf), np.complex64)
        full[:_BINS] = out_spec
        for i in range(nf):
            frame = np.fft.irfft(full[:, i], n=_N).real.astype(np.float32)
            res[i * _HOP : i * _HOP + _N] += frame * win
            ws[i * _HOP : i * _HOP + _N] += win**2
        return (res / np.where(ws < 1e-8, 1.0, ws))[: len(mono)].astype(np.float32)


def _resamp(x, src, dst):
    if src == dst:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def _wiener(mono, sr):
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    _, _, Z = stft(mono, fs=sr, nperseg=_N, noverlap=_N - _HOP, window="hann")
    mag = np.abs(Z)
    ne = np.maximum(uniform_filter(mag, (1, 9)), 1e-8)
    gain = np.maximum(mag**2 / (mag**2 + ne**2 + 1e-10), 0.15)
    _, o = istft(gain * mag * np.exp(1j * np.angle(Z)), fs=sr, nperseg=_N, noverlap=_N - _HOP, window="hann")
    return o[: len(mono)].astype(np.float32)


def get_resemble_enhance_plugin() -> ResembleEnhancePlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = ResembleEnhancePlugin()
    return _inst


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    return get_resemble_enhance_plugin().enhance(audio, sr)
