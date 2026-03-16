import logging
"""
logger = logging.getLogger(__name__)
AURIK Unified Defect Detection System v8.2
==========================================

Systematische Audio-Defekt-Erkennung mit Severity Scoring & Treatment Recommendations.
Konkurriert mit iZotope RX10's "Repair Assistant".

Unterstützte Defekte:
- Clipping (Übersteuerung)
- Clicks & Pops (Vinyl, digitale Fehler)
- Crackle (Vinyl-Rauschen)
- Broadband Noise (Rauschen)
- Hum & Buzz (50/60 Hz, harmonics)
- Distortion (THD, IMD)
- Spectral Artifacts (Musical Noise, Pre-Echo)
- Low Frequency Rumble
- High Frequency Roll-off
- Stereo Imbalance

Usage:
    from backend.defect_detection import UnifiedDefectDetector

    detector = UnifiedDefectDetector()
    report = detector.analyze(audio, sr)

    for defect in report.defects:
        logger.info(f"{defect.type}: Severity={defect.severity:.2f}, Confidence={defect.confidence:.2f}")
        logger.info(f"Recommended: {defect.treatment.method} with {defect.treatment.params}")
"""

from backend.defect_detection.base import (
    DefectDetector,
    DefectInstance,
    DefectReport,
    DefectType,
    SeverityLevel,
    TreatmentRecommendation,
)
from backend.defect_detection.registry import DefectDetectorRegistry
from backend.defect_detection.treatment_recommender import TreatmentRecommender
from backend.defect_detection.unified_detector import UnifiedDefectDetector

__version__ = "8.2.0"

__all__ = [
    "DefectDetector",
    "DefectType",
    "DefectReport",
    "DefectInstance",
    "TreatmentRecommendation",
    "SeverityLevel",
    "DefectDetectorRegistry",
    "UnifiedDefectDetector",
    "TreatmentRecommender",
]
