"""
Aurik 9 — forensics-Paket
==========================
Forensische Analyse von Tonträgerketten und Medientypen.
"""

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    TransferChain,
    detect_medium_chain,
    get_medium_detector,
)

__all__ = [
    "MediumDetector",
    "MediumDetectionResult",
    "TransferChain",
    "get_medium_detector",
    "detect_medium_chain",
]
