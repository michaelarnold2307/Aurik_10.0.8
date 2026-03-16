"""
DC Offset Detector
==================

Detects DC offset (non-zero mean).
"""

import numpy as np

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class DCOffsetDetector(DefectDetector):
    """
    Detects DC offset.

    DC offset is a constant shift away from zero in the waveform,
    caused by hardware issues or poor digitization.
    """

    def __init__(self, threshold: float = 0.001):
        super().__init__(name="dc_offset_detector", defect_type=DefectType.DC_OFFSET)
        self.threshold = threshold

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.02, **kwargs) -> list[DefectInstance]:
        """Detect DC offset, kontextbewusst."""
        if audio.ndim == 2:
            return self._detect_multichannel(audio, sr, tolerance)
        return self._detect_mono(audio, sr, tolerance)

    def _detect_mono(self, audio: np.ndarray, sr: int, tolerance: float) -> list[DefectInstance]:
        """Detect DC offset in mono audio."""

        # Calculate mean (DC component)
        dc_offset = np.mean(audio)
        dc_offset_abs = abs(dc_offset)

        # Severity: 0.001 = 0.1, 0.1 = 1.0
        severity = min(dc_offset_abs / 0.1, 1.0)
        if severity < tolerance or dc_offset_abs < self.threshold:
            return []

        # Confidence: very high for DC offset (easy to measure accurately)
        confidence = 0.95

        metrics = {
            "dc_offset": float(dc_offset),
            "dc_offset_abs": float(dc_offset_abs),
        }

        description = f"DC offset detected: {dc_offset:+.4f}"

        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]

    def _detect_multichannel(self, audio: np.ndarray, sr: int, tolerance: float) -> list[DefectInstance]:
        """Detect DC offset in multi-channel audio."""

        defects = []

        for ch_idx in range(audio.shape[1]):
            ch_audio = audio[:, ch_idx]
            ch_defects = self._detect_mono(ch_audio, sr, tolerance)
            for defect in ch_defects:
                defect.affected_channels = [ch_idx]
                defect.description += f" (Channel {ch_idx})"
            defects.extend(ch_defects)
        return defects
