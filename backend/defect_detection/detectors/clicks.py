"""
Clicks & Pops Detector
======================

Detects transient clicks and pops (vinyl defects, digital errors).
"""

import numpy as np

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class ClicksDetector(DefectDetector):
    """
    Detects clicks and pops in audio.

    Clicks are short-duration transient disturbances (< 5ms).
    Common in vinyl recordings, digital errors, electrical interference.
    """

    def __init__(self, threshold_percentile: float = 99.95):
        super().__init__(name="clicks_detector", defect_type=DefectType.CLICKS)
        self.threshold_percentile = threshold_percentile

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.15, **kwargs) -> list[DefectInstance]:
        """Detect clicks/pops in audio, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Analyze first channel
        diff = np.abs(np.diff(audio))
        threshold = np.percentile(diff, self.threshold_percentile)
        potential_clicks = diff > threshold
        click_indices = np.where(potential_clicks)[0]
        if len(click_indices) == 0:
            return []
        click_groups = self._group_clicks(click_indices, max_gap=sr // 1000)  # 1ms gap
        duration_seconds = len(audio) / sr
        clicks_per_second = len(click_groups) / duration_seconds
        severity = min(clicks_per_second / 100.0, 1.0)
        if severity < tolerance:
            return []
        click_amplitudes = diff[click_indices]
        mean_excess = np.mean(click_amplitudes / threshold)
        confidence = min(0.5 + (mean_excess - 1.0) * 0.2, 0.95)
        metrics = {
            "num_clicks": len(click_groups),
            "clicks_per_second": float(clicks_per_second),
            "max_amplitude": float(np.max(click_amplitudes)),
            "mean_amplitude": float(np.mean(click_amplitudes)),
            "threshold": float(threshold),
        }
        description = f"Detected {len(click_groups)} clicks ({clicks_per_second:.1f} clicks/sec)"
        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]

    def _group_clicks(self, indices: np.ndarray, max_gap: int) -> list[list[int]]:
        """Group nearby click indices into events."""
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for idx in indices[1:]:
            if idx - current_group[-1] <= max_gap:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]

        groups.append(current_group)
        return groups
