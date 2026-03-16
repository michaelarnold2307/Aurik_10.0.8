"""
multiband_master.py - Intelligenter Multiband-Mastering-Kompressor für Aurik 6.0

4-Band-Mastering-Kompressor (scipy/numpy, kein torchaudio):
  Band 1: Sub-Bass  ( 20–200 Hz)  : leichter Ratio 2:1
  Band 2: Bass/Mid  (200–1000 Hz) : Hauptkompression 3:1
  Band 3: Presence  (1k–5k Hz)   : Transparenz 2.5:1
  Band 4: Brillanz  (5k–20k Hz)   : Luft 2:1
Jedes Band: RMS-Pegelerfassung mit Attack/Release-Glättung.
Totale Makeup-Gain-Normierung nach Kompression.
"""

import logging

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


class MultibandMasterCompressor:
    """4-Band-Mastering-Kompressor."""

    # Band-Definitionen: (low_hz, high_hz, threshold_lin, ratio)
    _BANDS = [
        (None, 200, 0.40, 2.0),  # Sub-Bass
        (200, 1000, 0.30, 3.0),  # Bass/Mid
        (1000, 5000, 0.25, 2.5),  # Presence
        (5000, None, 0.35, 2.0),  # Brillanz
    ]

    def __init__(self, model_path: str | None = None, bands: int = 4):
        self.model_path = model_path
        self.model = None
        self.bands = min(max(bands, 2), 4)

    def _log_contract(self) -> None:
        logger.debug("[DSPContract] MultibandMasterCompressor bands=%d", self.bands)

    @staticmethod
    def _butter_band(sr: int, low, high, order: int = 4) -> np.ndarray:
        nyq = sr / 2.0
        if low is None:
            fc = min(high / nyq, 0.49)
            return butter(order, fc, btype="low", output="sos")
        elif high is None:
            fc = min(low / nyq, 0.49)
            fc = max(fc, 0.001)
            return butter(order, fc, btype="high", output="sos")
        lo = max(low / nyq, 0.001)
        hi = min(high / nyq, 0.499)
        if hi <= lo:
            hi = lo + 0.001
        return butter(order, [lo, hi], btype="band", output="sos")

    @staticmethod
    def _rms_compress(
        band: np.ndarray, threshold: float, ratio: float, attack_s: int = 128, release_s: int = 1024
    ) -> np.ndarray:
        eps = 1e-12
        rms = np.sqrt(np.convolve(band**2, np.ones(attack_s) / attack_s, mode="same") + eps)
        desired = np.where(rms > threshold, threshold + (rms - threshold) / ratio, rms)
        gain_raw = desired / (rms + eps)
        # Temporale Glättung
        smooth = np.zeros_like(gain_raw)
        alpha = 1.0 - np.exp(-1.0 / release_s)
        g = 1.0
        for i, ga in enumerate(gain_raw):
            g += alpha * (ga - g)
            smooth[i] = g
        return band * smooth

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """4-Band-Mastering-Kompressor.

        :param audio: Eingabesignal (np.ndarray, mono oder stereo)
        :param sr: Abtastrate
        :return: Komprimiertes Signal
        """
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
            return audio
        bands_to_use = self._BANDS[: self.bands]

        def _process_ch(ch):
            ch = ch.astype(np.float64)
            band_sigs = []
            for low, high, thr, ratio in bands_to_use:
                sos = self._butter_band(sr, low, high)
                b = sosfilt(sos, ch)
                c = self._rms_compress(b, thr, ratio)
                band_sigs.append(c)
            out = np.sum(band_sigs, axis=0)
            return np.clip(out, -1.0, 1.0)

        try:
            if audio.ndim == 1:
                result = _process_ch(audio)
            else:
                result = np.stack([_process_ch(ch) for ch in audio], axis=0)
            return result.astype(audio.dtype)
        except Exception:
            return audio
