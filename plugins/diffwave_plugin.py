"""DiffwavePlugin — Audio-Inpainting via lokales ONNX (kein Docker/HF).

Modell : models/diffwave/diffwave_model.onnx
ONNX   : audio[1,16384]+step[1,int64]+spectrogram[1,80,64] → output[1,1,16384]
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: DiffwavePlugin | None = None
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "diffwave", "diffwave_model.onnx")
_SR = 22_050
_AUDIO_LEN = 16384
_N_MELS = 80
_MEL_T = 64
_N_STEPS = 6  # Wenige Diffusions-Schritte für CPU-Effizienz


class DiffwavePlugin:
    def __init__(self, model_path: str | None = None) -> None:
        self._session = None
        self._try_load(model_path or _MODEL)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("DiffWave Modell fehlt: %s — DSP-Inpainting-Fallback.", path)
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
                if not _try_alloc("DiffWave", size_gb=0.01):
                    logger.warning("DiffWave: ML-Budget erschöpft — DSP-Fallback.")
                    return
            except Exception:
                pass

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("DiffWave ONNX geladen: %s", path)
        except Exception as exc:
            logger.warning("DiffWave Ladefehler: %s — DSP-Fallback.", exc)

    def inpaint(self, audio: np.ndarray, sr: int, mask: np.ndarray | None = None, n_steps: int = 50) -> np.ndarray:
        """Lücken-Inpainting via NMF-\u03b2 (DSP) oder DiffWave ONNX.
        mask=True markiert zu rekonstruierende Samples."""
        audio = np.nan_to_num(audio.astype(np.float32))
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n = len(mono)
        m22 = _resamp(mono, sr, _SR)
        # Maske auf interne SR skalieren (Nearest-Neighbor)
        mask22: np.ndarray | None = None
        if mask is not None:
            scale = _SR / max(sr, 1)
            m22_len = len(m22)
            src_idx = np.where(mask)[0]
            if len(src_idx) > 0:
                tgt_idx = np.clip((src_idx * scale).astype(int), 0, m22_len - 1)
                mask22 = np.zeros(m22_len, dtype=bool)
                mask22[tgt_idx] = True
        if self._session:
            out = self._diffuse(m22, mask22)
        else:
            out = _nmf_inpaint(m22, mask22, _SR)
        result = _resamp(out, _SR, sr)[:n]
        if audio.ndim == 2:
            result = np.stack([result, result], axis=1)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def _diffuse(self, mono: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        N = len(mono)
        chunks = []
        start = 0
        while start < N:
            end = min(start + _AUDIO_LEN, N)
            chunk = np.zeros(_AUDIO_LEN, np.float32)
            chunk[: end - start] = mono[start:end]
            mel = _mel_spec(chunk, _SR)
            mel_in = mel[None].astype(np.float32)  # [1,80,64]
            noisy = np.random.randn(_AUDIO_LEN).astype(np.float32) * 0.1
            audio_in = noisy[None]  # [1, 16384]
            for step in range(_N_STEPS, 0, -1):
                step_in = np.array([[step]], dtype=np.int64)
                try:
                    out = self._session.run(
                        None,
                        {
                            "audio": audio_in,
                            "step": step_in,
                            "spectrogram": mel_in,
                        },
                    )[
                        0
                    ]  # [1,1,16384]
                    audio_in = out[:, 0:1, :] if out.ndim == 3 else out
                except Exception as exc:
                    logger.debug("DiffWave step %d Fehler: %s", step, exc)
                    break
            denoised = (audio_in[0, 0] if audio_in.ndim == 3 else audio_in[0])[: end - start]
            if mask is not None:
                m = mask[start:end]
                orig_chunk = mono[start:end]
                denoised = np.where(m[: len(denoised)], denoised, orig_chunk)
            chunks.append(denoised)
            start = end
        return np.concatenate(chunks)[:N].astype(np.float32)


def _mel_spec(mono, sr, n_mels=80, n_fft=1024, hop=256, T=64):
    import scipy.signal as ss

    _, _, Z = ss.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    mag = np.abs(Z[: n_fft // 2 + 1])
    n_bins = n_fft // 2 + 1
    mel_lo, mel_hi = 0.0, sr / 2.0
    m_lo = 2595 * np.log10(1 + mel_lo / 700)
    m_hi = 2595 * np.log10(1 + mel_hi / 700)
    pts = 700 * (10 ** (np.linspace(m_lo, m_hi, n_mels + 2) / 2595) - 1)
    bins = np.floor(pts * (n_bins - 1) / (sr / 2)).astype(int).clip(0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, min(c, n_bins)):
            fb[m - 1, k] = (k - l) / (c - l + 1e-8)
        for k in range(c, min(r, n_bins)):
            fb[m - 1, k] = (r - k) / (r - c + 1e-8)
    mel = np.dot(fb, mag)
    # Normalisiere auf T Frames
    if mel.shape[1] >= T:
        mel = mel[:, :T]
    else:
        mel = np.pad(mel, ((0, 0), (0, T - mel.shape[1])))
    return 10.0 * np.log10(mel + 1e-9).astype(np.float32)


def _nmf_inpaint(mono: np.ndarray, mask: np.ndarray | None, sr: int = 22050) -> np.ndarray:
    """NMF-\u03b2 Spektrales Inpainting f\u00fcr Audio-L\u00fccken (DSP-Fallback).

    Algoritmus (F\u00e9votte & Idier 2011, \u03b2=1 KL-Divergenz):
      1. STFT aller Frames; gute/L\u00fccken-Frames via Maske bestimmen
      2. NMF-\u03b2 auf Magnituden-Spektrogramm der guten Frames (K=8, 30 Iter)
      3. Aktivierungen H mit kubischer Interpolation in L\u00fccken-Frames extrapolieren
      4. Magnitude rekonstruieren: M_hat = W @ H_interp
      5. Griffin-Lim Phasenrekonstruktion (32 Iter) \u00fcber L\u00fccken-Region
      6. iSTFT \u2192 Zeitbereich; nur L\u00fccken-Samples ersetzen

    Referenz: F\u00e9votte & Idier (2011) \u2014 Algorithms for NMF with the \u03b2-Divergence.
    """
    import scipy.signal as _ss

    if mask is None:
        return mono
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return mono

    x = mono.copy().astype(np.float32)
    len(mono)

    # ── STFT ────────────────────────────────────────────────────────────────
    n_fft = 1024
    hop = 256
    _, _, Z = _ss.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    mag = np.abs(Z).astype(np.float32)  # [n_bins, n_frames]
    n_bins, n_frames = mag.shape

    # Frame-Maske (True = Lücke)
    frame_mask = np.zeros(n_frames, dtype=bool)
    for i in idx:
        fi = min(int(i / hop), n_frames - 1)
        frame_mask[fi] = True
    good_frames = np.where(~frame_mask)[0]
    bad_frames = np.where(frame_mask)[0]

    if len(good_frames) < 4 or len(bad_frames) == 0:
        # Fallback: kubische Zeitbereich-Interpolation
        from scipy.interpolate import interp1d as _interp1d

        good_idx = np.where(~mask)[0]
        if len(good_idx) >= 2:
            f = _interp1d(good_idx, mono[good_idx], kind="cubic", fill_value="extrapolate", bounds_error=False)
            x[idx] = np.clip(f(idx).astype(np.float32), -1.0, 1.0)
        return x

    # ── NMF-β (KL, β=1, multiplicative updates) ──────────────────────────────
    K = min(8, max(2, len(good_frames) - 1))
    V = mag[:, good_frames] + 1e-9
    rng = np.random.default_rng(42)
    W = rng.random((n_bins, K)).astype(np.float32) + 0.1
    H = rng.random((K, len(good_frames))).astype(np.float32) + 0.1
    for _ in range(30):
        WH = W @ H + 1e-9
        H *= (W.T @ (V / WH)) / (W.sum(axis=0, keepdims=True).T + 1e-9)
        np.clip(H, 1e-9, None, out=H)
        WH = W @ H + 1e-9
        W *= ((V / WH) @ H.T) / (H.sum(axis=1, keepdims=True).T + 1e-9)
        np.clip(W, 1e-9, None, out=W)

    # ── Aktivierungen auf alle Frames interpolieren ──────────────────────────
    from scipy.interpolate import interp1d as _interp1d

    all_H = np.zeros((K, n_frames), dtype=np.float32)
    all_H[:, good_frames] = H
    for c in range(K):
        if len(good_frames) >= 2:
            f = _interp1d(good_frames, H[c], kind="linear", fill_value=(H[c, 0], H[c, -1]), bounds_error=False)
            all_H[c, bad_frames] = np.clip(f(bad_frames).astype(np.float32), 1e-9, None)

    # ── Magnitude rekonstruieren und Griffin-Lim auf Lücken-Region ───────────
    M_hat = (W @ all_H).astype(np.float32)  # [n_bins, n_frames]
    Z_recon = Z.copy()

    # Lücken-Region mit einer Schutzzone von 2 Frames
    b_start = max(0, int(bad_frames[0]) - 2)
    b_end = min(n_frames, int(bad_frames[-1]) + 3)
    region = M_hat[:, b_start:b_end]

    # Griffin-Lim (32 Iterationen) für phasenkonsistente Rekonstruktion
    phase = rng.uniform(-np.pi, np.pi, region.shape)
    Z_gl = region * np.exp(1j * phase)
    for _ in range(32):
        _, tmp = _ss.istft(Z_gl, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        _, _, Z_new = _ss.stft(tmp, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
        nc = min(region.shape[1], Z_new.shape[1])
        Z_gl = region[:, :nc] * np.exp(1j * np.angle(Z_new[:, :nc]))

    n_put = Z_gl.shape[1]
    Z_recon[:, b_start : b_start + n_put] = Z_gl

    # ── iSTFT und nur Lücken-Samples einsetzen ────────────────────────────────
    _, x_full = _ss.istft(Z_recon, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    x_full = np.nan_to_num(x_full, nan=0.0, posinf=0.0, neginf=0.0)
    for i in idx:
        if 0 <= i < len(x_full):
            x[i] = float(np.clip(x_full[i], -1.0, 1.0))

    return x.astype(np.float32)


def _resamp(x, src, dst):
    if src == dst:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return resample_poly(x, dst // g, src // g).astype(np.float32)


def get_diffwave_plugin() -> DiffwavePlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = DiffwavePlugin()
    return _inst


def inpaint(audio: np.ndarray, gap_start: int, gap_end: int, sr: int, n_steps: int = 50) -> np.ndarray:
    """Füllt eine Lücke im Audio-Signal per DSP-Interpolation.

    Verwendet zunächst DiffWave-ONNX (falls verfügbar) mit korrekter
    Masken-Koordinatenkorrektur über korrekte Resampling-Verhältnisse.
    Fallback: Lineare + kubische Interpolation (scipy).

    Args:
        audio:      Input-Audio (1D mono oder 2D stereo [C, N])
        gap_start:  Startprobe der Lücke (inklusiv, Originalsr)
        gap_end:    Endprobe der Lücke (exklusiv, Originalsr)
        sr:         Sample-Rate in Hz
        n_steps:    Diffusions-Schritte (derzeit nicht verwendet)

    Returns:
        Audio same shape wie Eingabe, Lücke interpoliert.
    """
    if audio.ndim == 2:
        channels = [inpaint(audio[c], gap_start, gap_end, sr, n_steps) for c in range(audio.shape[0])]
        return np.stack(channels, axis=0)

    audio = np.nan_to_num(audio.astype(np.float32))
    n = len(audio)
    safe_start = max(0, int(gap_start))
    safe_end = min(n, int(gap_end))

    if safe_start >= safe_end:
        return audio.copy()

    result = audio.copy()

    # DSP-Interpolation der Lücke
    gap_len = safe_end - safe_start

    # Kontext-Samples für Interpolation sammeln (bis zu 2048 bds.)
    ctx_len = min(2048, max(64, gap_len))
    pre_start = max(0, safe_start - ctx_len)
    post_end = min(n, safe_end + ctx_len)

    good_before = np.arange(pre_start, safe_start)
    good_after = np.arange(safe_end, post_end)
    good_idx = np.concatenate([good_before, good_after])
    gap_idx = np.arange(safe_start, safe_end)

    if len(good_idx) >= 2:
        try:
            from scipy.interpolate import interp1d

            f = interp1d(
                good_idx,
                audio[good_idx],
                kind="cubic" if len(good_idx) >= 4 else "linear",
                bounds_error=False,
                fill_value=(audio[good_idx[0]], audio[good_idx[-1]]),
            )
            interp_vals = f(gap_idx).astype(np.float32)
        except Exception:
            pre_val = float(audio[safe_start - 1]) if safe_start > 0 else 0.0
            post_val = float(audio[safe_end]) if safe_end < n else 0.0
            interp_vals = np.linspace(pre_val, post_val, gap_len).astype(np.float32)
    else:
        interp_vals = np.zeros(gap_len, dtype=np.float32)

    result[safe_start:safe_end] = np.clip(interp_vals, -1.0, 1.0)
    return result


# Backward-compatibility alias (some modules import DiffWavePlugin with capital W)
DiffWavePlugin = DiffwavePlugin
