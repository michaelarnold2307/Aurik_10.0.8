"""
Clipping Detector
=================

Detects audio clipping (hard and soft).
"""

import numpy as np

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class ClippingDetector(DefectDetector):
    """
    Detects clipping in audio.

    Clipping occurs when signal amplitude exceeds the maximum representable value,
    causing waveform distortion (flat tops/bottoms).
    """

    def __init__(self, hard_threshold: float = 0.99, soft_threshold: float = 0.95):
        super().__init__(name="clipping_detector", defect_type=DefectType.CLIPPING)
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.0001, **kwargs) -> list[DefectInstance]:
        """Detect clipping in audio, kontextbewusst."""
        # Handle stereo
        if audio.ndim == 2:
            return self._detect_multichannel(audio, sr, tolerance)
        # Mono detection
        return self._detect_mono(audio, sr, tolerance)

    def _detect_mono(self, audio: np.ndarray, sr: int, tolerance: float) -> list[DefectInstance]:
        """Detect clipping in mono audio."""
        audio_abs = np.abs(audio)

        # Hard clipping: samples at or above threshold
        hard_clips = audio_abs >= self.hard_threshold
        hard_clip_ratio = np.mean(hard_clips)

        # Soft clipping: samples close to maximum
        soft_clips = (audio_abs >= self.soft_threshold) & (audio_abs < self.hard_threshold)
        soft_clip_ratio = np.mean(soft_clips)

        total_clip_ratio = hard_clip_ratio + soft_clip_ratio

        # No significant clipping
        if total_clip_ratio < tolerance:
            return []

        # Calculate severity (0.0 - 1.0)
        # 0.1% clipping = 0.1 severity, 10% = 1.0 severity
        severity = min(total_clip_ratio * 1000, 1.0)

        # Confidence (high confidence for clear clipping)
        confidence = 0.95 if hard_clip_ratio > 0.001 else 0.75

        # Metrics
        metrics = {
            "hard_clip_ratio": float(hard_clip_ratio),
            "soft_clip_ratio": float(soft_clip_ratio),
            "total_clip_ratio": float(total_clip_ratio),
            "max_amplitude": float(np.max(audio_abs)),
            "num_clipped_samples": int(np.sum(hard_clips)),
        }

        # Localize clipping events
        clipped_samples = np.where(hard_clips)[0]
        if len(clipped_samples) > 0:
            start_time = clipped_samples[0] / sr
            end_time = clipped_samples[-1] / sr
        else:
            start_time = None
            end_time = None

        description = f"Clipping detected: {total_clip_ratio*100:.2f}% of samples"
        if hard_clip_ratio > 0:
            description += f" ({hard_clip_ratio*100:.2f}% hard clipping)"

        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
                start_time=start_time,
                end_time=end_time,
            )
        ]

    def _detect_multichannel(self, audio: np.ndarray, sr: int, tolerance: float) -> list[DefectInstance]:
        """Detect clipping in multi-channel audio."""
        defects = []

        for ch_idx in range(audio.shape[1]):
            ch_audio = audio[:, ch_idx]
            ch_defects = self._detect_mono(ch_audio, sr, tolerance)
            # Add channel info
            for defect in ch_defects:
                defect.affected_channels = [ch_idx]
                defect.description += f" (Channel {ch_idx})"
            defects.extend(ch_defects)
        return defects
