import logging
"""
Unified Defect Detector
=======================

Main orchestrator for audio defect detection.
Runs all registered detectors and generates comprehensive report.
"""

import time

import numpy as np

from backend.defect_detection.base import (
    DefectInstance,
    DefectReport,
    SeverityLevel,
)

# Import and register all detectors
from backend.defect_detection.detectors import (
    BroadbandNoiseDetector,
    ClicksDetector,
    ClippingDetector,
    DCOffsetDetector,
    DistortionDetector,
    HFRolloffDetector,
    HumDetector,
    RumbleDetector,
    StereoImbalanceDetector,
)
from backend.defect_detection.registry import DefectDetectorRegistry, get_global_registry
from backend.defect_detection.treatment_recommender import TreatmentRecommender

logger = logging.getLogger(__name__)


class UnifiedDefectDetector:
    """
    Unified audio defect detection system.

    Orchestrates multiple defect detectors to provide:
    - Comprehensive defect analysis
    - Severity scoring for all defect types
    - Treatment recommendations with priorities
    - Overall quality assessment

    Similar to iZotope RX10's "Repair Assistant".

    Usage:
        detector = UnifiedDefectDetector()
        report = detector.analyze(audio, sr)

        # Review defects
        for defect in report.get_critical_defects():
            logger.info(f"{defect.type.value}: {defect.description}")

        # Apply treatments
        for treatment in report.recommended_treatments:
            logger.info(f"Priority {treatment.priority}: {treatment.method}")
    """

    def __init__(
        self,
        registry: DefectDetectorRegistry | None = None,
        enable_treatments: bool = True,
        user_policy: dict | None = None,
        reference_profile: dict | None = None,
        tontraeger_chain: list | None = None,
        audit_context: dict | None = None,
        custom_tolerances: dict | None = None,
    ):
        """
        Initialisiert kontextbewusste Defekterkennung.
        Schwellenwerte und Toleranzen werden aus User-Policy, Referenz, Tonträgerkette und Audit-Kontext gesetzt.
        """
        self.registry = registry or get_global_registry()
        self.enable_treatments = enable_treatments
        self.treatment_recommender = TreatmentRecommender() if enable_treatments else None
        self.user_policy = user_policy or {}
        self.reference_profile = reference_profile or {}
        self.tontraeger_chain = tontraeger_chain or []
        self.audit_context = audit_context or {}
        self.custom_tolerances = custom_tolerances or {}
        self.detector_tolerances = self._build_tolerances()
        # Register all default detectors if registry is empty
        if len(self.registry.list_names()) == 0:
            self._register_default_detectors()

    def _build_tolerances(self) -> dict:
        """
        Erzeugt kontextbewusste Toleranzen für alle Detektoren.
        SOTA-Weltspitze-Niveau: Maximal robust gegen Fehlalarme bei professionellem Material.
        """
        tolerances = {
            "clipping": self.custom_tolerances.get("clipping", 0.01),
            "broadband_noise": self.custom_tolerances.get("broadband_noise", 0.5),
            "hum": self.custom_tolerances.get("hum", 0.5),
            "stereo_imbalance": self.custom_tolerances.get("stereo_imbalance", 2.0),
            "dc_offset": self.custom_tolerances.get("dc_offset", 0.05),
            "clicks": self.custom_tolerances.get("clicks", 0.3),
            "rumble": self.custom_tolerances.get("rumble", 0.3),
            "distortion": self.custom_tolerances.get("distortion", 0.3),
            "hf_rolloff": self.custom_tolerances.get("hf_rolloff", 0.3),
        }
        # Policy/Referenz/Audit können Toleranzen überschreiben
        for key in tolerances:
            if key in self.user_policy:
                tolerances[key] = self.user_policy[key]
            if key in self.reference_profile:
                tolerances[key] = self.reference_profile[key]
            if key in self.audit_context:
                tolerances[key] = self.audit_context[key]
        return tolerances

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        detector_names: list[str] | None = None,
        context: dict | None = None,
    ) -> DefectReport:
        """
        Analyze audio for all defects.

        Args:
            audio: Audio array (n_samples,) or (n_samples, n_channels)
            sr: Sample rate
            detector_names: Optional list of specific detectors to run

        Returns:
            Comprehensive defect report with treatments
        """
        start_time = time.time()

        # Kontext zusammenführen
        ctx = context or {}
        tolerances = self.detector_tolerances.copy()
        for key in tolerances:
            if key in ctx:
                tolerances[key] = ctx[key]
        # Get detectors to run
        if detector_names:
            detectors = [self.registry.get(name) for name in detector_names]
            detectors = [d for d in detectors if d is not None]
        else:
            detectors = self.registry.get_all()
        # Run all detectors mit kontextbewussten Toleranzen
        all_defects: list[DefectInstance] = []
        for detector in detectors:
            try:
                # Toleranz für Detektor bestimmen
                tol = tolerances.get(detector.defect_type.value, None)
                # Fallback: Default-Toleranz, falls None (SOTA-Weltspitze)
                if tol is None:
                    if detector.defect_type.value == "clipping":
                        tol = 0.01
                    elif detector.defect_type.value == "broadband_noise" or detector.defect_type.value == "hum":
                        tol = 0.5
                    elif detector.defect_type.value == "stereo_imbalance":
                        tol = 2.0
                    elif detector.defect_type.value == "dc_offset":
                        tol = 0.05
                    elif (
                        detector.defect_type.value == "clicks"
                        or detector.defect_type.value == "rumble"
                        or detector.defect_type.value == "distortion"
                        or detector.defect_type.value == "hf_rolloff"
                    ):
                        tol = 0.3
                    else:
                        tol = 0.3
                defects = detector.detect(audio, sr, tolerance=tol)
                # NaN/Inf-Guard für Toleranzwerte
                if tol is not None and not np.isfinite(tol):
                    tol = 0.3
                all_defects.extend(defects)
            except Exception as e:
                logger.error(f"Warning: Detector {detector.name} failed: {e}")
                continue

        # Generate treatment recommendations
        recommended_treatments = []
        if self.enable_treatments and self.treatment_recommender:
            for defect in all_defects:
                treatment = self.treatment_recommender.recommend(defect)
                defect.treatment = treatment

            # Get unique treatments sorted by priority
            recommended_treatments = self.treatment_recommender.recommend_batch(all_defects)

        # Calculate summary statistics
        severity_counts = dict.fromkeys(SeverityLevel, 0)
        for defect in all_defects:
            severity_counts[defect.severity_level] += 1

        # Overall quality assessment
        overall_quality = self._calculate_overall_quality(all_defects)

        # Determine if restoration needed
        needs_restoration = (
            severity_counts[SeverityLevel.CRITICAL] > 0
            or severity_counts[SeverityLevel.SEVERE] > 0
            or overall_quality < 0.7
        )

        # Audio metadata
        duration = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
        num_channels = 1 if audio.ndim == 1 else audio.shape[1]

        analysis_time = time.time() - start_time

        return DefectReport(
            defects=all_defects,
            total_defects=len(all_defects),
            critical_count=severity_counts[SeverityLevel.CRITICAL],
            severe_count=severity_counts[SeverityLevel.SEVERE],
            moderate_count=severity_counts[SeverityLevel.MODERATE],
            minor_count=severity_counts[SeverityLevel.MINOR],
            overall_quality=overall_quality,
            needs_restoration=needs_restoration,
            recommended_treatments=recommended_treatments,
            audio_duration=duration,
            sample_rate=sr,
            num_channels=num_channels,
            analysis_time=analysis_time,
        )

    def _calculate_overall_quality(self, defects: list[DefectInstance]) -> float:
        """
        Calculate overall audio quality score (0.0 - 1.0).

        Algorithm:
        - Start with perfect quality (1.0)
        - Subtract weighted severity scores
        - Weights: Critical=0.3, Severe=0.2, Moderate=0.1, Minor=0.05
        """
        if not defects:
            return 1.0

        quality = 1.0

        for defect in defects:
            # Weight by severity level
            if defect.severity_level == SeverityLevel.CRITICAL:
                quality -= defect.severity * 0.3
            elif defect.severity_level == SeverityLevel.SEVERE:
                quality -= defect.severity * 0.2
            elif defect.severity_level == SeverityLevel.MODERATE:
                quality -= defect.severity * 0.1
            elif defect.severity_level == SeverityLevel.MINOR:
                quality -= defect.severity * 0.05

        return max(quality, 0.0)

    def _register_default_detectors(self):
        """Register all default detectors."""
        default_detectors = [
            ClippingDetector(),
            ClicksDetector(),
            BroadbandNoiseDetector(),
            HumDetector(),
            DistortionDetector(),
            RumbleDetector(),
            HFRolloffDetector(),
            StereoImbalanceDetector(),
            DCOffsetDetector(),
        ]

        for detector in default_detectors:
            self.registry.register(detector)

    def list_detectors(self) -> list[str]:
        """List all registered detector names."""
        return self.registry.list_names()

    def quick_scan(self, audio: np.ndarray, sr: int) -> dict:
        """
        Quick scan returning simple summary (faster than full analyze).

        Returns:
            {
                'has_defects': bool,
                'critical_count': int,
                'needs_restoration': bool,
                'quality_score': float,
            }
        """
        # Run only fast detectors
        fast_detectors = ["clipping_detector", "dc_offset_detector", "stereo_imbalance_detector"]
        report = self.analyze(audio, sr, detector_names=fast_detectors)

        return {
            "has_defects": report.total_defects > 0,
            "critical_count": report.critical_count,
            "needs_restoration": report.needs_restoration,
            "quality_score": report.overall_quality,
        }
