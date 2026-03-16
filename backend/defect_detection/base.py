"""
Base classes for AURIK Defect Detection System
===============================================

Abstract base class and data models for audio defect detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DefectType(Enum):
    """Audio defect types that can be detected."""

    CLIPPING = "clipping"
    CLICKS = "clicks"
    CRACKLE = "crackle"
    BROADBAND_NOISE = "broadband_noise"
    HUM = "hum"
    BUZZ = "buzz"
    DISTORTION = "distortion"
    SPECTRAL_ARTIFACTS = "spectral_artifacts"
    RUMBLE = "rumble"
    HF_ROLLOFF = "hf_rolloff"
    STEREO_IMBALANCE = "stereo_imbalance"
    DROPOUTS = "dropouts"
    PHASE_ISSUES = "phase_issues"
    DC_OFFSET = "dc_offset"
    ALIASING = "aliasing"


class SeverityLevel(Enum):
    """Severity classification for defects."""

    NONE = 0  # 0.0 - 0.1
    MINOR = 1  # 0.1 - 0.3
    MODERATE = 2  # 0.3 - 0.6
    SEVERE = 3  # 0.6 - 0.9
    CRITICAL = 4  # 0.9 - 1.0

    @classmethod
    def from_score(cls, score: float) -> "SeverityLevel":
        """Convert severity score (0.0-1.0) to level."""
        if score < 0.1:
            return cls.NONE
        elif score < 0.3:
            return cls.MINOR
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.9:
            return cls.SEVERE
        else:
            return cls.CRITICAL


@dataclass
class TreatmentRecommendation:
    """Treatment recommendation for a detected defect."""

    method: str  # e.g., "declip", "denoise", "declick"
    module_path: str  # e.g., "dsp.automatic_declipper"
    params: dict[str, Any]  # Recommended parameters
    priority: int  # 1 (highest) to 5 (lowest)
    expected_improvement: float  # 0.0 - 1.0
    side_effects: list[str]  # Potential issues
    requires_manual_check: bool = False  # Human verification needed?


@dataclass
class DefectInstance:
    """Single instance of a detected defect."""

    type: DefectType
    severity: float  # 0.0 (none) to 1.0 (critical)
    confidence: float  # 0.0 to 1.0
    severity_level: SeverityLevel

    # Spatial/temporal localization
    start_time: float | None = None  # seconds
    end_time: float | None = None  # seconds
    affected_channels: list[int] | None = None

    # Quantitative metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Treatment
    treatment: TreatmentRecommendation | None = None

    # Metadata
    description: str = ""
    detected_by: str = ""  # Detector name


@dataclass
class DefectReport:
    """Complete defect analysis report for audio."""

    defects: list[DefectInstance]

    # Summary statistics
    total_defects: int
    critical_count: int
    severe_count: int
    moderate_count: int
    minor_count: int

    # Overall assessment
    overall_quality: float  # 0.0 (terrible) to 1.0 (perfect)
    needs_restoration: bool

    # Processing recommendations (ordered by priority)
    recommended_treatments: list[TreatmentRecommendation]

    # Metadata
    audio_duration: float  # seconds
    sample_rate: int
    num_channels: int
    analysis_time: float  # seconds

    def get_defects_by_type(self, defect_type: DefectType) -> list[DefectInstance]:
        """Get all defects of a specific type."""
        return [d for d in self.defects if d.type == defect_type]

    def get_critical_defects(self) -> list[DefectInstance]:
        """Get all critical/severe defects."""
        return [d for d in self.defects if d.severity_level in (SeverityLevel.CRITICAL, SeverityLevel.SEVERE)]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "defects": [
                {
                    "type": d.type.value,
                    "severity": d.severity,
                    "confidence": d.confidence,
                    "severity_level": d.severity_level.name,
                    "start_time": d.start_time,
                    "end_time": d.end_time,
                    "affected_channels": d.affected_channels,
                    "metrics": d.metrics,
                    "treatment": (
                        {
                            "method": d.treatment.method,
                            "module_path": d.treatment.module_path,
                            "params": d.treatment.params,
                            "priority": d.treatment.priority,
                            "expected_improvement": d.treatment.expected_improvement,
                            "side_effects": d.treatment.side_effects,
                            "requires_manual_check": d.treatment.requires_manual_check,
                        }
                        if d.treatment
                        else None
                    ),
                    "description": d.description,
                    "detected_by": d.detected_by,
                }
                for d in self.defects
            ],
            "summary": {
                "total_defects": self.total_defects,
                "critical_count": self.critical_count,
                "severe_count": self.severe_count,
                "moderate_count": self.moderate_count,
                "minor_count": self.minor_count,
                "overall_quality": self.overall_quality,
                "needs_restoration": self.needs_restoration,
            },
            "recommended_treatments": [
                {
                    "method": t.method,
                    "module_path": t.module_path,
                    "params": t.params,
                    "priority": t.priority,
                    "expected_improvement": t.expected_improvement,
                    "side_effects": t.side_effects,
                    "requires_manual_check": t.requires_manual_check,
                }
                for t in self.recommended_treatments
            ],
            "metadata": {
                "audio_duration": self.audio_duration,
                "sample_rate": self.sample_rate,
                "num_channels": self.num_channels,
                "analysis_time": self.analysis_time,
            },
        }


class DefectDetector(ABC):
    """
    Abstract base class for audio defect detectors.

    Each detector is responsible for:
    1. Detecting a specific type of audio defect
    2. Quantifying severity (0.0 - 1.0)
    3. Providing confidence score (0.0 - 1.0)
    4. Optionally localizing defects in time/space
    """

    def __init__(self, name: str, defect_type: DefectType):
        self.name = name
        self.defect_type = defect_type

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int, **kwargs) -> list[DefectInstance]:
        """
        Detect defects in audio.

        Args:
            audio: Audio array (n_samples,) or (n_samples, n_channels)
            sr: Sample rate
            **kwargs: Detector-specific parameters

        Returns:
            List of detected defect instances
        """

    def _create_instance(
        self,
        severity: float,
        confidence: float,
        metrics: dict[str, float],
        description: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        affected_channels: list[int] | None = None,
    ) -> DefectInstance:
        """Helper to create DefectInstance with common fields."""
        return DefectInstance(
            type=self.defect_type,
            severity=np.clip(severity, 0.0, 1.0),
            confidence=np.clip(confidence, 0.0, 1.0),
            severity_level=SeverityLevel.from_score(severity),
            start_time=start_time,
            end_time=end_time,
            affected_channels=affected_channels,
            metrics=metrics,
            description=description,
            detected_by=self.name,
        )
