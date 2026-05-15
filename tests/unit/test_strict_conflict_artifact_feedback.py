"""Unit tests for strict conflict adaptation from artifact-related phase events."""

from __future__ import annotations

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


def _profile() -> dict:
    return {
        "family_scalars": {
            "general": 1.0,
            "denoise": 1.0,
            "reconstruction": 1.0,
        },
        "strict_conflict_policy": {
            "rollback_decay_per_family": {
                "general": 0.96,
                "denoise": 0.90,
                "reconstruction": 0.93,
            },
            "rollback_decay_floor": 0.55,
            "phase_strength_caps": {
                "phase_03_denoise": 0.80,
            },
        },
    }


class TestStrictConflictArtifactFeedback:
    """Artifact-related conflicts must tighten future phase intervention online."""

    def test_artifact_rollback_decays_matching_family_scalar(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.83},
        )

        family_scalars = restorer._song_calibration_profile["family_scalars"]
        assert family_scalars["denoise"] == pytest.approx(0.90, abs=1e-9)
        runtime = restorer._phase_goal_conflict_runtime
        assert runtime["by_family"]["denoise"] == 1
        assert runtime["by_phase"]["phase_03_denoise"] == 1
        assert runtime["events"][-1]["reason"] == "artifact_freedom_rollback"

    def test_repeated_artifact_events_tighten_phase_cap(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "artifact_freedom_rollback",
            {"artifact_freedom": 0.84},
        )
        restorer._register_phase_goal_conflict_event(
            "phase_03_denoise",
            "noise_texture_rollback",
            {"noise_texture_deviation_db_oct": 16.0},
        )

        profile = restorer._song_calibration_profile
        assert profile["family_scalars"]["denoise"] == pytest.approx(0.81, abs=1e-9)
        assert profile["strict_conflict_policy"]["phase_strength_caps"]["phase_03_denoise"] == pytest.approx(
            0.736,
            abs=1e-9,
        )

    def test_unknown_phase_uses_general_decay_without_crashing(self):
        restorer = UnifiedRestorerV3()
        restorer._song_calibration_profile = _profile()

        restorer._register_phase_goal_conflict_event(
            "phase_27_click_pop_removal",
            "hf_hallucination_rescue",
            {"hf_delta_ratio": 0.21},
        )

        assert restorer._song_calibration_profile["family_scalars"]["general"] == pytest.approx(0.96, abs=1e-9)
