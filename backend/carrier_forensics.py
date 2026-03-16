"""
backend/carrier_forensics.py — Kompatibilitäts-Shim (Aurik 6.0 → 9.x)
=============================================

Dieses Modul ist ein reiner Re-Export-Shim für
``backend.core.medium_classifier``.

Migrationsanleitung::

    # Alt (Aurik 6.0):
    from backend.carrier_forensics import CarrierForensics
    # Neu (Aurik 9.x):
    from backend.core.medium_classifier import MediumClassifier, classify_medium

Referenz: §2.1 Aurik-9-Spec, MediumClassifier (§6.1 MaterialType)
"""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "backend.carrier_forensics ist veraltet (Aurik 6.0). "
    "Verwende 'from backend.core.medium_classifier import MediumClassifier, classify_medium'.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.core.medium_classifier import (  # noqa: F401, E402
    ClassificationResult,
    MediumClassifier,
    classify_medium,
    get_medium_classifier,
)

# Aurik-6.0-kompatibler Alias
CarrierForensics = MediumClassifier

__all__ = [
    "CarrierForensics",
    "MediumClassifier",
    "ClassificationResult",
    "classify_medium",
    "get_medium_classifier",
]

# --- Aurik-6.0-Original-Code entfernt (2026-03-11, §9.4 Anti-Parallelwelten) ---
# Originaldatei war: carrier_forensics.py für Aurik 6.0
# Nachfolger: backend.core.medium_classifier (MediumClassifier)
