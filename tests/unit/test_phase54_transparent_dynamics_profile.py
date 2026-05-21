"""Unit-Tests: TransparentDynamics._compute_transparent_dynamics_profile() (§2.56)

Verifiziert dass der adaptive mix_delta-Parameter korrekt aus
Quality-Mode und Restorability berechnet wird.
"""

import numpy as np
import pytest

from backend.core.phases.phase_54_transparent_dynamics import (
    TransparentDynamicsV1 as TransparentDynamicsPhase,
)
from backend.core.phases.phase_54_transparent_dynamics import _extract_compression_pressure

# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _profile(material: str = "vinyl", qm: str = "balanced", rest: float = 50.0) -> dict:
    return TransparentDynamicsPhase._compute_transparent_dynamics_profile(material, qm, rest)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransparentDynamicsProfileQualityMode:
    """Quality-Mode-Anpassungen auf mix_delta."""

    def test_quality_positive_delta(self):
        delta = _profile(qm="quality")["mix_delta"]
        assert delta > 0.0, f"quality should increase mix, got {delta}"

    def test_maximum_positive_delta(self):
        delta = _profile(qm="maximum")["mix_delta"]
        assert delta > 0.0

    def test_fast_negative_delta(self):
        delta = _profile(qm="fast")["mix_delta"]
        assert delta < 0.0, f"fast should decrease mix, got {delta}"

    def test_balanced_zero_delta(self):
        delta = _profile(qm="balanced")["mix_delta"]
        assert delta == pytest.approx(0.0), f"balanced should have 0 delta, got {delta}"

    def test_none_quality_mode_zero_delta(self):
        delta = _profile(qm=None)["mix_delta"]
        assert delta == pytest.approx(0.0)


class TestTransparentDynamicsProfileRestorability:
    """Restorability-Einfluss auf mix_delta."""

    def test_low_restorability_negative_delta(self):
        """Low-Rest (<40) → mehr dry (weniger Kompression auf fragiles Material)."""
        delta = _profile(rest=25.0)["mix_delta"]
        assert delta < 0.0, f"low restorability should decrease mix, got {delta}"

    def test_high_restorability_no_change(self):
        delta = _profile(rest=80.0)["mix_delta"]
        assert delta == pytest.approx(0.0)

    def test_combined_quality_maximum_low_rest(self):
        """quality + low_rest: Net-delta = +0.05 - 0.10 = -0.05."""
        delta = _profile(qm="quality", rest=20.0)["mix_delta"]
        assert delta == pytest.approx(-0.05)

    def test_combined_fast_low_rest(self):
        """fast + low_rest: -0.05 - 0.10 = -0.15"""
        delta = _profile(qm="fast", rest=20.0)["mix_delta"]
        assert delta == pytest.approx(-0.15)


class TestTransparentDynamicsProfileBounds:
    """Grenzwerte."""

    def test_delta_within_limits(self):
        """mix_delta muss in [-0.20, +0.20] bleiben."""
        for material in ["shellac", "vinyl", "cd_digital", "unknown"]:
            for qm in ["quality", "maximum", "fast", "balanced", None]:
                for rest in [5.0, 50.0, 90.0]:
                    delta = _profile(material, qm, rest)["mix_delta"]
                    assert -0.20 <= delta <= 0.20, f"mix_delta out of bounds: {delta} ({material}, {qm}, {rest})"

    def test_unknown_material_no_error(self):
        delta = _profile("unknown_xyz")["mix_delta"]
        assert isinstance(delta, float)


class TestTransparentDynamicsProfileIntegration:
    """End-to-End: Profil fließt in PhaseResult metadata."""

    def test_profile_in_metadata(self):
        from backend.core.phases.phase_54_transparent_dynamics import TransparentDynamicsV1 as TransparentDynamicsPhase

        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.3, 0.3, (48000, 2)).astype(np.float32)

        # Use quality mode kwarg
        result = phase.process(
            audio,
            quality_mode="quality",
            restorability_score=60.0,
            strength=0.5,
        )
        assert result.success
        assert "transparent_dynamics_profile" in result.metadata
        assert "mix_delta" in result.metadata
        # quality mode → positive mix_delta
        assert result.metadata["mix_delta"] >= 0.0


class TestTransparentDynamicsCompressionPressure:
    def test_extract_compression_pressure_from_numeric_scores(self):
        pressure = _extract_compression_pressure(
            {
                "compression_artifacts": 0.80,
                "dynamic_compression_excess": 0.30,
            }
        )
        assert pressure == pytest.approx(0.80)

    def test_extract_compression_pressure_from_mixed_objects(self):
        class _Score:
            def __init__(self, severity: float):
                self.severity = severity

        pressure = _extract_compression_pressure(
            {
                "DefectType.COMPRESSION_ARTIFACTS": _Score(0.62),
                "DefectType.DIGITAL_ARTIFACTS": _Score(0.90),
            }
        )
        assert pressure == pytest.approx(0.62)

    def test_high_compression_pressure_enables_hard_intervention(self):
        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.2, 0.2, (48000, 2)).astype(np.float32)

        result = phase.process(
            audio,
            strength=0.05,
            defect_scores={"compression_artifacts": 0.90},
            quality_mode="balanced",
            restorability_score=45.0,
        )

        assert result.success
        assert result.metadata["hard_intervention_active"] is True
        assert result.metadata["control_strength"] >= 0.75
        assert result.metadata["mix"] >= 0.70

    def test_low_compression_pressure_keeps_hard_intervention_off(self):
        phase = TransparentDynamicsPhase(sample_rate=48000)
        audio = np.random.uniform(-0.2, 0.2, (48000, 2)).astype(np.float32)

        result = phase.process(
            audio,
            strength=0.20,
            defect_scores={"compression_artifacts": 0.20},
            quality_mode="balanced",
            restorability_score=45.0,
        )

        assert result.success
        assert result.metadata["hard_intervention_active"] is False
        assert result.metadata["control_floor"] == pytest.approx(0.0)
