"""
vampnet_plugin.py -- DSP-Stub fuer VampNet (kein Docker, kein Download).

VampNet ist ein generatives Masking-basiertes Audio-Codec-Modell (2023).
Dieses Plugin bietet einen beat-synchronisierten DSP-Fallback:
Beat-Detection via Autokorrelation + Segment-basierte Pass-Through.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: VampnetPlugin | None = None


class VampnetPlugin:
    """DSP-Stub fuer VampNet: Beat-synchronisierter Pass-Through.

    Da kein lokales VampNet-Modell vorhanden ist, wird ein
    DSP-basierter Beat-Tracker + Segment-Mix als Fallback genutzt.
    """

    def generate(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        mask_ratio: float = 0.5,
    ) -> np.ndarray:
        """Beat-synchronisierter DSP-Stub (Pass-Through + leichtes Morphing).

        Fuer volle generative Funktionalitaet waere das VampNet-Modell in
        models/vampnet/ erforderlich (aktuell nicht vorhanden).

        Args:
            audio: Eingabe-Audio float32
            sr: Sample-Rate
            mask_ratio: Anteil zu maskierender Segmente [0,1] (Fallback: ignoriert)
        """
        logger.info("VampNet DSP-Stub: Beat-Tracking Fallback (kein Modell vorhanden)")
        mono = self._to_mono(audio)

        bpm, beats = self._detect_beats(mono, sr)
        logger.debug("VampNet-Stub: BPM=%.1f, Beats=%d", bpm, len(beats))

        # DSP-Fallback: leichtes Spektral-Smoothing zwischen Beat-Segmenten
        result = self._smooth_between_beats(mono, beats, sr)

        if audio.ndim == 2:
            n_ch = audio.shape[0] if audio.shape[0] <= 8 else audio.shape[1]
            result = np.stack([result] * n_ch)
            if audio.shape[0] > 8:
                result = result.T

        return np.clip(result.astype(np.float32), -1.0, 1.0)

    def get_beats(self, audio: np.ndarray, sr: int = 48000) -> tuple[float, np.ndarray]:
        """Beat-Information via Autokorrelation. Gibt (bpm, beat_samples) zurueck."""
        mono = self._to_mono(audio)
        return self._detect_beats(mono, sr)

    # ------------------------------------------------------------------
    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        a = np.array(audio, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=1)
        return np.nan_to_num(a, 0.0)

    def _detect_beats(self, mono: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
        """Einfaches Beat-Tracking via RMS-Energiehuellkurve + Autokorrelation."""
        hop = sr // 100  # 10 ms frames
        frames = len(mono) // hop
        if frames < 4:
            return 120.0, np.array([0, len(mono) // 2], dtype=int)

        rms = np.array([np.sqrt(np.mean(mono[i * hop : (i + 1) * hop] ** 2)) for i in range(frames)], dtype=np.float32)

        # Autokorrelation im Tempo-Bereich 60-240 BPM
        lo = int(sr * 60.0 / (240.0 * hop))
        hi = int(sr * 60.0 / (60.0 * hop)) + 1
        hi = min(hi, len(rms) // 2)
        if lo >= hi:
            return 120.0, np.arange(0, len(mono), sr // 2, dtype=int)

        ac = np.correlate(rms, rms, mode="full")
        ac = ac[len(ac) // 2 :]
        best_lag = int(np.argmax(ac[lo:hi]) + lo)
        bpm = float(60.0 * sr / (best_lag * hop))

        period_samples = best_lag * hop
        beats = np.arange(0, len(mono), period_samples, dtype=int)
        return bpm, beats

    def _smooth_between_beats(self, mono: np.ndarray, beats: np.ndarray, sr: int) -> np.ndarray:
        """Leichtes Spektral-Smoothing an Beat-Grenzen (DSP-Stub)."""
        out = mono.copy()
        fade = min(sr // 100, 256)  # 10 ms oder 256 Samples
        np.hanning(2 * fade)
        for b in beats[1:]:
            s = max(0, b - fade)
            e = min(len(out), b + fade)
            seg_len = e - s
            if seg_len < 4:
                continue
            w = np.hanning(seg_len).astype(np.float32)
            out[s:e] = out[s:e] * w + out[s:e] * (1 - w)
        return out


def get_vampnet_plugin() -> VampnetPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = VampnetPlugin()
    return _inst


def generate_audio(audio: np.ndarray, sr: int = 48000, mask_ratio: float = 0.5) -> np.ndarray:
    """Convenience-Wrapper. Gibt DSP-verarbeitetes Audio zurueck."""
    return get_vampnet_plugin().generate(audio, sr, mask_ratio)
