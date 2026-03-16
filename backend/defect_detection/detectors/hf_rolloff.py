"""
High-Frequency Rolloff Detector
================================

Detects premature high-frequency attenuation.
"""

import numpy as np
from scipy.signal import welch

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class HFRolloffDetector(DefectDetector):
    """
    Detects high-frequency roll-off.

    Identifies premature attenuation of high frequencies,
    common in MP3 compression, cassette tape, or old recordings.
    """

    def __init__(self, expected_rolloff: float = 18000.0):
        super().__init__(name="hf_rolloff_detector", defect_type=DefectType.HF_ROLLOFF)
        self.expected_rolloff = expected_rolloff

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.15, **kwargs) -> list[DefectInstance]:
        """Detect high-frequency roll-off, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]
        freqs, psd = welch(audio, sr, nperseg=min(4096, len(audio) // 4))
        psd_db = 10 * np.log10(psd + 1e-10)
        mid_freq_mask = (freqs >= 1000) & (freqs <= 4000)
        ref_level = np.mean(psd_db[mid_freq_mask])
        threshold = ref_level - 3.0
        hf_mask = freqs >= 5000
        hf_freqs = freqs[hf_mask]
        hf_psd_db = psd_db[hf_mask]
        if len(hf_freqs) == 0:
            return []
        below_threshold = hf_psd_db < threshold
        if np.any(below_threshold):
            rolloff_idx = np.argmax(below_threshold)
            rolloff_freq = hf_freqs[rolloff_idx]
        else:
            rolloff_freq = hf_freqs[-1]
        severity = max(0.0, (self.expected_rolloff - rolloff_freq) / (self.expected_rolloff - 8000))
        severity = min(severity, 1.0)
        if severity < tolerance:
            return []
        hf_energy = np.mean(psd[hf_mask])
        mid_energy = np.mean(psd[mid_freq_mask])
        hf_attenuation_db = 10 * np.log10(hf_energy / (mid_energy + 1e-10))
        confidence = 0.6 + min(abs(hf_attenuation_db) / 20.0, 0.3)
        metrics = {
            "rolloff_frequency": float(rolloff_freq),
            "expected_rolloff": float(self.expected_rolloff),
            "hf_attenuation_db": float(hf_attenuation_db),
            "reference_level_db": float(ref_level),
        }
        description = f"High-frequency roll-off at {rolloff_freq:.0f} Hz ({hf_attenuation_db:.1f} dB below mid-range)"
        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]
