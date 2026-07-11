import pytest

"""tests/unit/test_defect_phase_mapper_confidence.py

Regression-Tests fuer confidence-aware Priorisierung im DefectPhaseMapper.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.defect_phase_mapper import DefectPhaseMapper
from backend.core.defect_scanner import DefectType


@dataclass
class _DummyDefect:
    defect_type: DefectType
    severity: float
    confidence: float


@pytest.mark.unit
def test_low_confidence_downweights_secondary_phases_in_restoration() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.10)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=8)

    # Primary muss weiterhin vorne bleiben.
    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    # Bei niedriger Confidence koennen sekundäre Phasen konservativ ausfallen.
    if "phase_18_noise_gate" in phases:
        idx_primary = phases.index("phase_03_denoise")
        idx_secondary = phases.index("phase_18_noise_gate")
        assert idx_primary < idx_secondary


def test_high_confidence_keeps_secondary_recovery_relevant() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.95)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=6)

    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    assert "phase_18_noise_gate" in phases


def test_missing_confidence_falls_back_to_neutral_weighting() -> None:
    mapper = DefectPhaseMapper()

    class _NoConfidence:
        def __init__(self) -> None:
            self.defect_type = DefectType.CLICKS
            self.severity = 0.9

    phases = mapper.phases_for_defect_profile([_NoConfidence()], mode="restoration", max_phases=5)

    assert "phase_01_click_removal" in phases


def test_very_low_confidence_suppresses_secondary_in_restoration() -> None:
    mapper = DefectPhaseMapper()
    defect = _DummyDefect(defect_type=DefectType.HIGH_FREQ_NOISE, severity=1.0, confidence=0.05)

    phases = mapper.phases_for_defect_profile([defect], mode="restoration", max_phases=8)

    assert "phase_03_denoise" in phases
    assert "phase_29_tape_hiss_reduction" in phases
    assert "phase_18_noise_gate" not in phases
