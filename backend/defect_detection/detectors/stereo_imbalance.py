"""
Stereo Imbalance Detector
=========================

Detects level imbalance between stereo channels.
"""

import numpy as np

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class StereoImbalanceDetector(DefectDetector):
    """
    Detects stereo channel imbalance.

    Identifies level differences between left and right channels
    that may indicate recording/playback issues.
    """

    def __init__(self, threshold_db: float = 1.0):
        super().__init__(name="stereo_imbalance_detector", defect_type=DefectType.STEREO_IMBALANCE)
        self.threshold_db = threshold_db

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 1.0, **kwargs) -> list[DefectInstance]:
        """Detect stereo imbalance, kontextbewusst."""
        if audio.ndim != 2 or audio.shape[1] != 2:
            return []
        left = audio[:, 0]
        right = audio[:, 1]
        left_rms = np.sqrt(np.mean(left**2))
        right_rms = np.sqrt(np.mean(right**2))
        if left_rms < 1e-10 or right_rms < 1e-10:
            return []
        imbalance_db = 20 * np.log10(left_rms / right_rms)
        imbalance_abs = abs(imbalance_db)
        severity = min(imbalance_abs / 6.0, 1.0)
        if severity < 0.1 or imbalance_abs < tolerance:
            return []
        louder_channel = 0 if imbalance_db > 0 else 1
        quieter_channel = 1 - louder_channel
        confidence = min(0.7 + (imbalance_abs / 10.0), 0.95)
        metrics = {
            "imbalance_db": float(imbalance_db),
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
            "louder_channel": int(louder_channel),
        }
        channel_names = ["Left", "Right"]
        description = f"Stereo imbalance: {channel_names[louder_channel]} channel {imbalance_abs:.1f} dB louder than {channel_names[quieter_channel]}"
        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
                affected_channels=[louder_channel, quieter_channel],
            )
        ]
