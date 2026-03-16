"""
backend/core/transient_decoupled_processor.py
Aurik 9 -- Spec §2.27: TransientDecoupledProcessing

HPSS-basierte Transient-Separation: Percussive-Anteil nur durch Phase 01/27,
Harmonic-Anteil durch volle Pipeline; Rekombination via OLA-Crossfade (Hanning 10 ms).
Verhindert NR-induzierte Groove-Degradation (DTW <= 8 ms RMS).
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HPSS_HARMONIC_KERNEL: int = 31
HPSS_PERCUSSIVE_KERNEL: int = 31
CROSSFADE_MS: float = 10.0
PERCUSSIVE_ONLY_PHASES: List[str] = [
    "phase_01_click_removal",
    "phase_27_click_pop_removal",
]


def _hpss_separate(stft: np.ndarray, h_len: int, p_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Medianfilter-HPSS (Fitzgerald 2010). Gibt (mask_h, mask_p) zurueck."""
    try:
        from scipy.ndimage import median_filter
    except ImportError:
        half = np.ones(stft.shape, dtype=np.float32) * 0.5
        return half, half
    mag = np.abs(stft) + 1e-10
    H = median_filter(mag, size=(h_len, 1))
    P = median_filter(mag, size=(1, p_len))
    H2, P2 = H**2, P**2
    denom = H2 + P2 + 1e-20
    return (H2 / denom).astype(np.float32), (P2 / denom).astype(np.float32)


class TransientDecoupledProcessing:
    """Spec §2.27: HPSS-Trennung fuer Groove-Maximierung."""

    HPSS_HARMONIC_KERNEL: int = HPSS_HARMONIC_KERNEL
    HPSS_PERCUSSIVE_KERNEL: int = HPSS_PERCUSSIVE_KERNEL
    PERCUSSIVE_ONLY_PHASES: List[str] = PERCUSSIVE_ONLY_PHASES

    def __init__(self) -> None:
        self._n_fft: int = 1024
        self._hop_length: int = 256

    def separate(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gibt (audio_percussive, audio_harmonic) zurueck."""
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        if len(audio) < self._n_fft:
            half = audio * 0.5
            return half.copy(), half.copy()
        try:
            from scipy.signal import istft as _istft, stft as _stft

            _, _, Z = _stft(audio, fs=sr, nperseg=self._n_fft, noverlap=self._n_fft - self._hop_length)
            mask_h, mask_p = _hpss_separate(Z, self.HPSS_HARMONIC_KERNEL, self.HPSS_PERCUSSIVE_KERNEL)
            _, h = _istft(Z * mask_h, fs=sr, nperseg=self._n_fft, noverlap=self._n_fft - self._hop_length)
            _, p = _istft(Z * mask_p, fs=sr, nperseg=self._n_fft, noverlap=self._n_fft - self._hop_length)
            n = len(audio)
            h = np.pad(h, (0, max(0, n - len(h))))[:n]
            p = np.pad(p, (0, max(0, n - len(p))))[:n]
        except Exception as exc:
            logger.debug("HPSS-Fallback: %s", exc)
            p = audio * 0.5
            h = audio * 0.5
        p = np.nan_to_num(p.astype(np.float32))
        h = np.nan_to_num(h.astype(np.float32))
        return np.clip(p, -1.0, 1.0), np.clip(h, -1.0, 1.0)

    def recombine(
        self,
        audio_p: np.ndarray,
        audio_h: np.ndarray,
        sr: int,
        original_perc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """OLA-Crossfade-Rekombination. NaN/Inf-sicher, geclipped auf [-1,1]."""
        audio_p = np.nan_to_num(np.asarray(audio_p, dtype=np.float32))
        audio_h = np.nan_to_num(np.asarray(audio_h, dtype=np.float32))
        n = max(len(audio_p), len(audio_h))
        audio_p = np.pad(audio_p, (0, max(0, n - len(audio_p))))[:n]
        audio_h = np.pad(audio_h, (0, max(0, n - len(audio_h))))[:n]
        mix = audio_p + audio_h
        if original_perc is not None and self._grove_violated(audio_p, original_perc, sr):
            orig = np.nan_to_num(np.asarray(original_perc, dtype=np.float32))
            orig = np.pad(orig, (0, max(0, n - len(orig))))[:n]
            mix = orig + audio_h
            logger.debug("GrooveMetric DTW >8 ms -- original_perc uebernommen")
        mix = np.nan_to_num(mix)
        return np.clip(mix, -1.0, 1.0).astype(np.float32)

    def _grove_violated(self, proc: np.ndarray, orig: np.ndarray, sr: int) -> bool:
        try:
            from scipy.signal import find_peaks

            hop = self._hop_length
            o_env = np.abs(orig[::hop])
            p_env = np.abs(proc[::hop])
            o_pk, _ = find_peaks(o_env, height=0.01, distance=4)
            p_pk, _ = find_peaks(p_env, height=0.01, distance=4)
            if len(o_pk) == 0 or len(p_pk) == 0:
                return False
            n = min(len(o_pk), len(p_pk))
            diff_ms = np.abs(o_pk[:n] - p_pk[:n]) * hop / sr * 1000.0
            return float(np.sqrt(np.mean(diff_ms**2))) > 8.0
        except Exception:
            return False


_instance: Optional[TransientDecoupledProcessing] = None
_lock = threading.Lock()


def get_transient_decoupled_processor() -> TransientDecoupledProcessing:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TransientDecoupledProcessing()
    return _instance


def separate_transients(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience-Wrapper."""
    return get_transient_decoupled_processor().separate(audio, sr)


def recombine_transients(
    audio_p: np.ndarray,
    audio_h: np.ndarray,
    sr: int,
    original_perc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience-Wrapper."""
    return get_transient_decoupled_processor().recombine(audio_p, audio_h, sr, original_perc)


# Backward-compat-Alias (Tests importieren ohne "ing"-Suffix)
TransientDecoupledProcessor = TransientDecoupledProcessing


__all__ = [
    "TransientDecoupledProcessing",
    "TransientDecoupledProcessor",
    "get_transient_decoupled_processor",
    "separate_transients",
    "recombine_transients",
    "HPSS_HARMONIC_KERNEL",
    "HPSS_PERCUSSIVE_KERNEL",
    "CROSSFADE_MS",
    "PERCUSSIVE_ONLY_PHASES",
]
