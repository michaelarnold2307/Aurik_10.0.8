"""
Rumble Detector
===============

Detects low-frequency rumble (< 50 Hz).
"""

import numpy as np
from scipy.signal import butter, sosfilt, welch

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class RumbleDetector(DefectDetector):
    """
    Detects low-frequency rumble.

    Common in vinyl recordings from turntable motor vibration,
    poor isolation, or warped records.
    """

    def __init__(self, rumble_cutoff: float = 40.0):
        super().__init__(name="rumble_detector", defect_type=DefectType.RUMBLE)
        self.rumble_cutoff = rumble_cutoff

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.15, **kwargs) -> list[DefectInstance]:
        """Detect low-frequency rumble, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]
        sos = butter(4, self.rumble_cutoff, btype="low", fs=sr, output="sos")
        rumble = sosfilt(sos, audio)
        rumble_energy = np.mean(rumble**2)
        total_energy = np.mean(audio**2)
        rumble_ratio = rumble_energy / (total_energy + 1e-10)
        severity = min(rumble_ratio * 10.0, 1.0)
        if severity < tolerance:
            return []
        freqs, psd = welch(audio, sr, nperseg=min(8192, len(audio) // 4))
        low_freq_mask = freqs < self.rumble_cutoff
        low_freq_psd = psd[low_freq_mask]
        low_freqs = freqs[low_freq_mask]
        if len(low_freq_psd) > 0:
            peak_idx = np.argmax(low_freq_psd)
            peak_freq = low_freqs[peak_idx]
        else:
            peak_freq = 0.0
        confidence = 0.7 + min(severity * 0.2, 0.2)
        metrics = {
            "rumble_ratio": float(rumble_ratio),
            "rumble_energy_db": float(10 * np.log10(rumble_energy + 1e-10)),
            "peak_frequency": float(peak_freq),
            "cutoff_hz": float(self.rumble_cutoff),
        }
        description = f"Low-frequency rumble: {rumble_ratio*100:.1f}% of signal energy, peak at {peak_freq:.1f} Hz"
        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]
