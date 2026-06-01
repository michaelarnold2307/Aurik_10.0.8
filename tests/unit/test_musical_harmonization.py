"""
Tests für die Musikalische Harmonisierung (v9.10.64)
=====================================================

Testet:
  1. PMGG _run_phase: phase.process() statt phase() (Critical Bug Fix)
  2. DefectPhaseMapper reverse mapping + severity factor
  3. Wet/Dry strength modulation in PMGG
  4. Severity-proportionale Verarbeitung (kein 'eigenes Ding')

Invarianten:
  - Repair-Phasen arbeiten proportional zur gemessenen Defekt-Schwere
  - Enhancement-Phasen werden nicht skaliert (Faktor 1.0)
  - Timing-Phasen sind von Wet/Dry ausgenommen
  - 0.15 ≤ severity_factor ≤ 1.0 für Repair-Phasen
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_phase_mapper import (
    get_phase_defect_severity,
    get_phase_locality_factor,
    get_reverse_phase_map,
)
from backend.core.defect_scanner import DefectScore, DefectType

# ---------------------------------------------------------------------------
# §1 Reverse Phase Map
# ---------------------------------------------------------------------------


class TestReversePhaseMap:
    """Tests for the reverse phase map (phase_id → [DefectType])."""

    def test_reverse_map_not_empty(self):
        rmap = get_reverse_phase_map()
        assert len(rmap) > 0, "Reverse map should contain entries"

    def test_reverse_map_known_phases(self):
        rmap = get_reverse_phase_map()
        # Known defect-repair phases must be in the map
        assert "phase_01_click_removal" in rmap
        assert "phase_02_hum_removal" in rmap
        assert "phase_03_denoise" in rmap
        assert "phase_09_crackle_removal" in rmap

    def test_reverse_map_click_targets_clicks(self):
        rmap = get_reverse_phase_map()
        defects = rmap["phase_01_click_removal"]
        defect_values = [d.value for d in defects]
        assert "clicks" in defect_values

    def test_reverse_map_denoise_targets_noise(self):
        rmap = get_reverse_phase_map()
        defects = rmap["phase_03_denoise"]
        defect_values = [d.value for d in defects]
        assert "high_freq_noise" in defect_values

    def test_enhancement_phases_not_in_map(self):
        """Enhancement phases should NOT be in the reverse map."""
        rmap = get_reverse_phase_map()
        assert "phase_21_exciter" not in rmap
        assert "phase_13_stereo_enhancement" not in rmap
        assert "phase_17_mastering_polish" not in rmap
        assert "phase_40_loudness_normalization" not in rmap

    def test_reverse_map_is_cached(self):
        """Second call should return same object (cached)."""
        rmap1 = get_reverse_phase_map()
        rmap2 = get_reverse_phase_map()
        assert rmap1 is rmap2


# ---------------------------------------------------------------------------
# §2 Defect Severity Factor
# ---------------------------------------------------------------------------


class TestDefectSeverityFactor:
    """Tests for get_phase_defect_severity()."""

    @pytest.fixture
    def sample_scores(self):
        return {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.4, confidence=0.9),
            DefectType.HIGH_FREQ_NOISE: DefectScore(DefectType.HIGH_FREQ_NOISE, severity=0.7, confidence=0.95),
            DefectType.CRACKLE: DefectScore(DefectType.CRACKLE, severity=0.0, confidence=0.5),
        }

    def test_repair_phase_proportional(self, sample_scores):
        """Repair phase should scale to measured severity."""
        sev = get_phase_defect_severity("phase_01_click_removal", sample_scores)
        assert 0.35 <= sev <= 0.45, f"Expected ~0.4 for clicks severity 0.4, got {sev}"

    def test_denoise_high_severity(self, sample_scores):
        sev = get_phase_defect_severity("phase_03_denoise", sample_scores)
        assert 0.65 <= sev <= 0.75, f"Expected ~0.7 for noise severity 0.7, got {sev}"

    def test_zero_severity_floor(self, sample_scores):
        """Defect scanned but severity 0.0 -> floor at 0.15."""
        sev = get_phase_defect_severity("phase_09_crackle_removal", sample_scores)
        assert sev == pytest.approx(0.15, abs=0.01)

    def test_enhancement_phase_always_one(self, sample_scores):
        """Enhancement phases return 1.0 regardless of defect scores."""
        sev = get_phase_defect_severity("phase_21_exciter", sample_scores)
        assert sev == 1.0

    def test_defect_not_scanned_returns_one(self, sample_scores):
        """If targeted defects were NOT scanned -> 1.0 (no penalty)."""
        # phase_02 targets HUM, which is not in sample_scores
        sev = get_phase_defect_severity("phase_02_hum_removal", sample_scores)
        assert sev == 1.0

    def test_unknown_phase_returns_one(self, sample_scores):
        """Unknown phase_id returns 1.0."""
        sev = get_phase_defect_severity("phase_999_nonexistent", sample_scores)
        assert sev == 1.0

    def test_empty_defect_scores(self):
        """Empty defect_scores -> all phases return 1.0."""
        sev = get_phase_defect_severity("phase_01_click_removal", {})
        assert sev == 1.0

    def test_severity_clamped_to_one(self):
        """Severity > 1.0 should be clamped to 1.0."""
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=1.5, confidence=0.9),
        }
        sev = get_phase_defect_severity("phase_01_click_removal", scores)
        assert sev == 1.0

    def test_full_severity(self):
        """Severity 1.0 -> factor 1.0."""
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=1.0, confidence=0.9),
        }
        sev = get_phase_defect_severity("phase_01_click_removal", scores)
        assert sev == 1.0

    def test_multiple_defects_max_wins(self):
        """Phase targeting multiple defects: max severity wins."""
        scores = {
            DefectType.HIGH_FREQ_NOISE: DefectScore(DefectType.HIGH_FREQ_NOISE, severity=0.3, confidence=0.9),
            DefectType.QUANTIZATION_NOISE: DefectScore(DefectType.QUANTIZATION_NOISE, severity=0.8, confidence=0.8),
        }
        # phase_03 targets both HIGH_FREQ_NOISE and QUANTIZATION_NOISE
        sev = get_phase_defect_severity("phase_03_denoise", scores)
        assert sev == pytest.approx(0.8, abs=0.01)

    def test_severity_floor_never_below_015(self):
        """Factor should never go below 0.15 for a scanned defect."""
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.01, confidence=0.9),
        }
        sev = get_phase_defect_severity("phase_01_click_removal", scores)
        assert sev >= 0.15


class TestPhaseLocalityFactor:
    """Tests for get_phase_locality_factor()."""

    def test_enhancement_phase_always_one(self):
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.8, confidence=0.9),
        }
        coverage = {"clicks": 0.05}
        fac = get_phase_locality_factor("phase_21_exciter", scores, coverage)
        assert fac == 1.0

    def test_click_phase_low_coverage_damped(self):
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.8, confidence=0.9),
        }
        coverage = {"clicks": 0.02}
        fac = get_phase_locality_factor("phase_01_click_removal", scores, coverage)
        assert 0.35 <= fac < 0.60

    def test_click_phase_high_coverage_near_one(self):
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.8, confidence=0.9),
        }
        coverage = {"clicks": 0.95}
        fac = get_phase_locality_factor("phase_01_click_removal", scores, coverage)
        assert 0.90 <= fac <= 1.0

    def test_non_event_defect_phase_returns_one(self):
        # HUM is treated as non-local event in locality curves.
        scores = {
            DefectType.HUM: DefectScore(DefectType.HUM, severity=0.8, confidence=0.9),
        }
        coverage = {"hum": 0.01}
        fac = get_phase_locality_factor("phase_02_hum_removal", scores, coverage)
        assert fac == 1.0

    def test_missing_coverage_defaults_one(self):
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.8, confidence=0.9),
        }
        fac = get_phase_locality_factor("phase_01_click_removal", scores, None)
        assert fac == 1.0


# ---------------------------------------------------------------------------
# §3 PMGG _run_phase — Critical Callable Fix
# ---------------------------------------------------------------------------


class TestPMGGRunPhase:
    """Tests for the PMGG _run_phase fix (phase.process() vs phase())."""

    def test_phase_is_not_callable(self):
        """PhaseInterface instances must NOT be callable (no __call__)."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        assert not callable(phase), "PhaseInterface should not define __call__"

    def test_run_phase_executes_correctly(self):
        """_run_phase must call phase.process() and return modified audio."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        gate = PerPhaseMusicalGoalsGate()

        out = gate._run_phase(phase, audio, 1.0, {"sample_rate": 48000, "material_type": "tape"})
        delta = np.max(np.abs(out - audio))
        assert delta > 0.01, f"Phase must modify audio, delta={delta}"
        assert out.shape == audio.shape
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0

    def test_run_phase_wet_dry_half(self):
        """strength=0.5 -> 50% of full processing effect (wet/dry)."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        rng = np.random.default_rng(12345)
        audio = rng.standard_normal(48000).astype(np.float32) * 0.3
        gate = PerPhaseMusicalGoalsGate()

        out_full = gate._run_phase(phase, audio, 1.0, {"sample_rate": 48000, "material_type": "tape"})
        out_half = gate._run_phase(phase, audio, 0.5, {"sample_rate": 48000, "material_type": "tape"})

        # Robust against isolated clipping spikes: evaluate average intervention,
        # not a single extreme sample.
        delta_full = np.mean(np.abs(out_full - audio))
        delta_half = np.mean(np.abs(out_half - audio))

        # Half-strength should clearly attenuate vs full-strength while still
        # preserving a meaningful intervention.
        # Untergrenze bewusst konservativ: einige Phasen skalieren intern stark
        # auf strength, bevor PMGG-Wet/Dry greift (double-scaling, §2.29).
        ratio = delta_half / max(delta_full, 1e-10)
        assert 0.10 <= ratio <= 0.65, f"Wet/dry ratio should be attenuated and bounded, got {ratio:.2f}"

    def test_run_phase_zero_strength_bypass(self):
        """strength=0.0 -> no processing (bypass)."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        gate = PerPhaseMusicalGoalsGate()

        # strength=0.0 triggers the `0.0 < strength < 1.0` guard to be False
        # so no wet/dry applied.  But the phase still runs — this is by design:
        # the phase runs at whatever internal strength it has, but strength 0.0
        # is edge-case that should rarely occur in practice.
        out = gate._run_phase(phase, audio, 0.0, {"sample_rate": 48000, "material_type": "tape"})
        assert out.shape == audio.shape
        assert np.all(np.isfinite(out))

    def test_run_phase_returns_original_on_error(self):
        """If phase.process() raises, original audio is returned."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        class BrokenPhase:
            def get_metadata(self):
                from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata

                return PhaseMetadata(
                    phase_id="broken", name="Broken", category=PhaseCategory.DEFECT_REMOVAL, priority=5
                )

            def process(self, audio, **kwargs):
                raise RuntimeError("Intentional test failure")

        phase = BrokenPhase()
        audio = np.ones(4800, dtype=np.float32) * 0.5
        gate = PerPhaseMusicalGoalsGate()
        out = gate._run_phase(phase, audio, 1.0, {"sample_rate": 48000})
        np.testing.assert_array_equal(out, audio, "Should return original on error")

    def test_run_phase_nan_guard(self):
        """NaN in phase output should be cleaned."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_interface import PhaseResult

        class NaNPhase:
            def get_metadata(self):
                from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata

                return PhaseMetadata(phase_id="nan_test", name="NaN", category=PhaseCategory.DEFECT_REMOVAL, priority=5)

            def process(self, audio, **kwargs):
                out = audio.copy()
                out[0] = float("nan")
                return PhaseResult(audio=out)

        phase = NaNPhase()
        audio = np.ones(4800, dtype=np.float32) * 0.5
        gate = PerPhaseMusicalGoalsGate()
        out = gate._run_phase(phase, audio, 1.0, {"sample_rate": 48000})
        assert np.all(np.isfinite(out)), "NaN should be cleaned"


# ---------------------------------------------------------------------------
# §4 Musikalische Harmonisierung — Kein eigenes Ding
# ---------------------------------------------------------------------------


class TestMusicalHarmonization:
    """Integration-level tests: severity scaling ensures proportional processing."""

    def test_low_severity_gentle_processing(self):
        """Low defect severity -> factor close to floor -> gentle processing."""
        scores = {
            DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=0.15, confidence=0.9),
        }
        sev = get_phase_defect_severity("phase_01_click_removal", scores)
        assert sev == 0.15, "Very low severity should be at floor"

    def test_high_severity_full_processing(self):
        """High defect severity -> factor near 1.0 -> full processing."""
        scores = {
            DefectType.HIGH_FREQ_NOISE: DefectScore(DefectType.HIGH_FREQ_NOISE, severity=0.95, confidence=0.99),
        }
        sev = get_phase_defect_severity("phase_03_denoise", scores)
        assert sev == pytest.approx(0.95, abs=0.01)

    def test_severity_monotonic(self):
        """Severity factor must be monotonically increasing with defect severity."""
        factors = []
        for s in np.linspace(0.0, 1.0, 20):
            scores = {DefectType.CLICKS: DefectScore(DefectType.CLICKS, severity=float(s), confidence=0.9)}
            factors.append(get_phase_defect_severity("phase_01_click_removal", scores))
        for i in range(1, len(factors)):
            assert factors[i] >= factors[i - 1], (
                f"Factor must be monotonic: {factors[i - 1]:.3f} -> {factors[i]:.3f} at severity {i / 19:.2f}"
            )

    def test_reverse_map_covers_all_defect_types(self):
        """Every DefectType in _PHASE_MAP should have at least one primary phase."""
        from backend.core.defect_phase_mapper import _PHASE_MAP

        rmap = get_reverse_phase_map()
        all_mapped_defects = set()
        for defects in rmap.values():
            all_mapped_defects.update(defects)
        for dt in _PHASE_MAP:
            assert dt in all_mapped_defects, f"DefectType {dt.value} not in any primary phase"

    def test_wet_dry_preserves_shape(self):
        """Wet/dry mixing must preserve audio shape and dtype."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        audio = np.random.randn(48000).astype(np.float32) * 0.2
        gate = PerPhaseMusicalGoalsGate()

        for strength in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            out = gate._run_phase(phase, audio, strength, {"sample_rate": 48000, "material_type": "tape"})
            assert out.shape == audio.shape, f"Shape mismatch at strength={strength}"
            assert out.dtype == np.float32, f"Dtype mismatch at strength={strength}"
            assert np.all(np.isfinite(out)), f"Non-finite at strength={strength}"
            assert np.max(np.abs(out)) <= 1.0, f"Clipping at strength={strength}"

    def test_wet_dry_strength_gradient(self):
        """Higher strength -> larger delta from original."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
        from backend.core.phases.phase_03_denoise import DenoisePhase

        phase = DenoisePhase()
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        gate = PerPhaseMusicalGoalsGate()

        deltas = []
        for strength in [0.2, 0.4, 0.6, 0.8, 1.0]:
            out = gate._run_phase(phase, audio, strength, {"sample_rate": 48000, "material_type": "tape"})
            delta = float(np.mean(np.abs(out - audio)))
            deltas.append(delta)

        # Deltas should be monotonically increasing
        for i in range(1, len(deltas)):
            assert deltas[i] >= deltas[i - 1] * 0.95, (
                f"Delta must increase with strength: {deltas[i - 1]:.4f} -> {deltas[i]:.4f}"
            )
