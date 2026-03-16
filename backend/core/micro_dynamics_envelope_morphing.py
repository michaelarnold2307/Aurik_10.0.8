"""
backend/core/micro_dynamics_envelope_morphing.py
Aurik 9 -- Spec §2.30: MicroDynamicsEnvelopeMorphing (MDEM)

Stellt origales Mikro-Dynamik-Profil im restaurierten Signal wieder her.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _lufs_frame(frame: np.ndarray) -> float:
    """ITU-R BS.1770: momentane LUFS eines Audio-Frames (vereinfacht)."""
    if frame.ndim == 2:
        frame = frame.mean(axis=0)
    rms = math.sqrt(max(1e-15, float(np.mean(frame.astype(np.float64) ** 2))))
    lufs = 20.0 * math.log10(rms + 1e-15)
    return lufs


def _savgol_smooth(arr: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Vereinfachte Savitzky-Golay Glaettung (boxcar wenn scipy nicht verfuegbar)."""
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(arr, window_length=window, polyorder=polyorder).astype(np.float32)
    except Exception:
        # Boxcar-Fallback
        half = window // 2
        out = arr.copy()
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = np.mean(arr[lo:hi])
        return out


@dataclass
class MorphResult:
    """Ergebnis des Envelope-Morphing."""

    pearson_correlation: float
    max_gain_applied_lu: float
    retried: bool
    audio: np.ndarray


class MicroDynamicsEnvelopeMorphing:
    """Spec §2.30: Mikro-Dynamik-Profil aus Original im Restaurierten wiederherstellen.

    Algorithmus:
        1. 400-ms-LUFS-Profile beider Signale (hop 200 ms, 50 % Ueberlappung)
        2. G[k] = L_orig[k] - L_rest[k], geclippt auf ±MAX_GAIN_LU
        3. Savitzky-Golay-Glaettung
        4. Frame-weise lineare Gain-Interpolation
        5. True-Peak 1 dBTP nach Morphing
    """

    MAX_GAIN_LU: float = 3.0
    FRAME_SIZE_SAMPLES: int = 19200  # 400 ms @ 48000 Hz
    HOP_SIZE_SAMPLES: int = 9600  # 200 ms
    PEARSON_TARGET: float = 0.93
    MIN_LEVEL_LUFS: float = -60.0
    TRUE_PEAK_LIMIT: float = 0.98  # ~-1.0 dBTP linear

    def compute_lufs_profile(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """400-ms-momentane LUFS-Kurve, float32 [n_frames]."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if arr.ndim == 2:
            arr = arr.mean(axis=0)

        n = len(arr)
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        frames = max(1, (n - fsize) // hop + 1)
        profile = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            start = i * hop
            end = start + fsize
            frame = arr[start : min(end, n)]
            profile[i] = _lufs_frame(frame)
        return profile

    def morph(
        self,
        restored: np.ndarray,
        original: np.ndarray,
        sr: int = 48000,
        mode: str = "restoration",
    ) -> np.ndarray:
        """Morphed restauriertes Signal auf Original-Mikrodynamik. NaN/Inf-sicher."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        max_gain = 2.0 if mode == "restoration" else self.MAX_GAIN_LU

        res = np.nan_to_num(np.asarray(restored, dtype=np.float32))
        orig = np.nan_to_num(np.asarray(original, dtype=np.float32))

        is_stereo = res.ndim == 2
        if is_stereo:
            res_mono = res.mean(axis=0)
            orig_mono = orig.mean(axis=0) if orig.ndim == 2 else orig
        else:
            res_mono = res
            orig_mono = orig if orig.ndim == 1 else orig.mean(axis=0)

        L_orig = self.compute_lufs_profile(orig_mono, sr)
        L_rest = self.compute_lufs_profile(res_mono, sr)

        # Gain-Profil berechnen
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)
        for k in range(n_frames):
            lo = L_orig[k]
            lr = L_rest[k]
            if not (math.isfinite(lo) and math.isfinite(lr)):
                G[k] = 0.0
                continue
            if lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0  # Stille: kein Gain-Boost
                continue
            G[k] = np.clip(lo - lr, -max_gain, max_gain)

        # Glaettung
        G_smooth = _savgol_smooth(G)

        # Gain-Anwendung: frame-weise lineare Interpolation
        hop = self.HOP_SIZE_SAMPLES
        fsize = self.FRAME_SIZE_SAMPLES
        n = len(res_mono)

        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            end = start + fsize
            linear_gain = 10.0 ** (G_smooth[k] / 20.0)
            ce = min(end, n)
            if start < n:
                # Lineare Interpolation zu naechstem Frame
                if k + 1 < n_frames:
                    nxt_gain = 10.0 ** (G_smooth[k + 1] / 20.0)
                else:
                    nxt_gain = linear_gain
                ramp = np.linspace(linear_gain, nxt_gain, ce - start, dtype=np.float32)
                gain_envelope[start:ce] = ramp

        # Auf Stereo/Mono anwenden
        if is_stereo:
            out = res * gain_envelope[np.newaxis, : res.shape[1]] if res.shape[0] == 2 else res
            # Sicherere Anwendung
            if res.ndim == 2:
                n_ch = res.shape[0]
                out = np.zeros_like(res)
                for ch in range(n_ch):
                    n_samp = min(len(res[ch]), len(gain_envelope))
                    out[ch, :n_samp] = res[ch, :n_samp] * gain_envelope[:n_samp]
                    if n_samp < res.shape[1]:
                        out[ch, n_samp:] = res[ch, n_samp:]
        else:
            n_samp = min(n, len(gain_envelope))
            out = res.copy()
            out[:n_samp] = res[:n_samp] * gain_envelope[:n_samp]

        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = np.clip(out, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)

        # Pearson-Korrelation pruefen, ggf. Retry
        out_mono = out.mean(axis=0) if out.ndim == 2 else out
        r = self._pearson(orig_mono[: len(out_mono)], out_mono[: len(orig_mono)])

        if r < self.PEARSON_TARGET and max_gain < 4.0:
            logger.debug("MDEM Retry mit erweitertem MAX_GAIN=4.0 (aktuell r=%.3f)", r)
            # Einmaliger Retry mit erweitertem Gain — kein weiterer rekursiver Aufruf
            out2 = self._morph_internal(res_mono, orig_mono, max_gain=4.0)
            out2 = np.nan_to_num(out2, nan=0.0, posinf=1.0, neginf=-1.0)
            out2 = np.clip(out2, -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT)
            if is_stereo and res.ndim == 2:
                gain2 = out2 / np.where(np.abs(res_mono) > 1e-8, res_mono, 1.0)
                out_retry = np.zeros_like(res)
                for ch in range(res.shape[0]):
                    n_s = min(len(res[ch]), len(gain2))
                    out_retry[ch, :n_s] = res[ch, :n_s] * gain2[:n_s]
                    out_retry[ch, n_s:] = res[ch, n_s:]
                return np.clip(np.nan_to_num(out_retry), -self.TRUE_PEAK_LIMIT, self.TRUE_PEAK_LIMIT).astype(np.float32)
            return out2.astype(np.float32)

        return out.astype(np.float32)

    def _morph_internal(
        self,
        res_mono: np.ndarray,
        orig_mono: np.ndarray,
        max_gain: float = 3.0,
    ) -> np.ndarray:
        """Interne Gain-Envelope-Berechnung und -Anwendung auf Mono-Signale (kein Retry)."""
        L_orig = self.compute_lufs_profile(orig_mono)
        L_rest = self.compute_lufs_profile(res_mono)
        n_frames = min(len(L_orig), len(L_rest))
        G = np.zeros(n_frames, dtype=np.float32)
        for k in range(n_frames):
            lo, lr = L_orig[k], L_rest[k]
            if not (math.isfinite(lo) and math.isfinite(lr)):
                G[k] = 0.0
            elif lo < self.MIN_LEVEL_LUFS:
                G[k] = 0.0
            else:
                G[k] = float(np.clip(lo - lr, -max_gain, max_gain))
        G_smooth = _savgol_smooth(G)
        hop = self.HOP_SIZE_SAMPLES
        n = len(res_mono)
        gain_envelope = np.ones(n, dtype=np.float32)
        for k in range(n_frames):
            start = k * hop
            ce = min(start + self.FRAME_SIZE_SAMPLES, n)
            if start >= n:
                break
            lg = float(10.0 ** (G_smooth[k] / 20.0))
            nxt = float(10.0 ** (G_smooth[k + 1] / 20.0)) if k + 1 < n_frames else lg
            gain_envelope[start:ce] = np.linspace(lg, nxt, ce - start, dtype=np.float32)
        out = res_mono.copy()
        n_s = min(n, len(gain_envelope))
        out[:n_s] = res_mono[:n_s] * gain_envelope[:n_s]
        return out

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        if n < 2:
            return 1.0
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        am, bm = a.mean(), b.mean()
        num = float(np.mean((a - am) * (b - bm)))
        den = max(1e-15, float(np.std(a) * np.std(b)))
        val = num / den
        return float(np.clip(val, -1.0, 1.0)) if math.isfinite(val) else 0.0


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: Optional[MicroDynamicsEnvelopeMorphing] = None
_lock = threading.Lock()


def get_mdem() -> MicroDynamicsEnvelopeMorphing:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MicroDynamicsEnvelopeMorphing()
    return _instance


def morph_micro_dynamics(
    restored: np.ndarray,
    original: np.ndarray,
    sr: int = 48000,
    mode: str = "restoration",
) -> np.ndarray:
    """Convenience-Wrapper."""
    return get_mdem().morph(restored, original, sr, mode)


# ── Modul-Level-Konstanten (für direkten Import durch Tests und Consumer-Code) ──
# Spiegeln die Klassen-Attribute wider, sodass `from backend.core.micro_dynamics_envelope_morphing
# import FRAME_SIZE_SAMPLES` ohne Klassen-Instantiierung funktioniert.
FRAME_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.FRAME_SIZE_SAMPLES
HOP_SIZE_SAMPLES: int = MicroDynamicsEnvelopeMorphing.HOP_SIZE_SAMPLES
MAX_GAIN_LU: float = MicroDynamicsEnvelopeMorphing.MAX_GAIN_LU
MIN_LEVEL_LUFS: float = MicroDynamicsEnvelopeMorphing.MIN_LEVEL_LUFS
PEARSON_TARGET: float = MicroDynamicsEnvelopeMorphing.PEARSON_TARGET


def compute_lufs_profile(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Convenience-Wrapper für MicroDynamicsEnvelopeMorphing.compute_lufs_profile()."""
    return get_mdem().compute_lufs_profile(audio, sr)


__all__ = [
    "MicroDynamicsEnvelopeMorphing",
    "MorphResult",
    "get_mdem",
    "morph_micro_dynamics",
    "compute_lufs_profile",
    # Modul-Level-Konstanten:
    "FRAME_SIZE_SAMPLES",
    "HOP_SIZE_SAMPLES",
    "MAX_GAIN_LU",
    "MIN_LEVEL_LUFS",
    "PEARSON_TARGET",
]
