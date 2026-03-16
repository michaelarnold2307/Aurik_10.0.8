"""Shim – leitet weiter an backend.core.epistemic_gate.ethics_engine (kanonisch).

Gemäß §9.4 Anti-Parallelwelten: kein eigener Code, nur Re-Export.
"""

from backend.core.epistemic_gate.ethics_engine import (  # noqa: F401
    AuthenticityConstraints,
    EpistemicDecision,
    EthicsEngine,
    EthicsReport,
    ProcessingMode,
)

__all__ = [
    "EpistemicDecision",
    "ProcessingMode",
    "AuthenticityConstraints",
    "EthicsReport",
    "EthicsEngine",
]
