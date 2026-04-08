"""Stability guard tests for bounded-but-non-edge behavior.

Ensures calibration scalars and implicit phase strengths are pulled away from
hard bounds while explicit caller strengths remain untouched.
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata, create_phase_result
from backend.core.quality_mode import QualityMode
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


class _DummyPhase:
    def __init__(self, phase_id: str = "phase_03_denoise"):
        self._meta = PhaseMetadata(
            phase_id=phase_id,
            name="Dummy",
            category=PhaseCategory.RESTORATION,
            priority=5,
            version="1.0",
            dependencies=[],
            estimated_time_factor=0.0,
            memory_requirement_mb=1,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.0,
            description="dummy",
        )
        self.last_strength = None

    def get_metadata(self):
        return self._meta

    def process(self, audio: np.ndarray, sample_rate: int = 48000, **kwargs):
        self.last_strength = kwargs.get("strength")
        return create_phase_result(audio=np.asarray(audio, dtype=np.float32))


def test_song_calibration_pullback_avoids_hard_edges():
    profile = UnifiedRestorerV3._build_song_calibration_profile(
        material_type=MaterialType.SHELLAC,
        mode=QualityMode.BALANCED,
        restorability_score=0.0,
        input_snr_db=-10.0,
        max_defect_severity=1.0,
        pipeline_confidence=1.0,
    )

    g = float(profile["global_scalar"])
    assert 0.50 < g < 1.50
    fam = profile["family_scalars"]
    assert all(0.30 < float(v) < 1.80 for v in fam.values())


def test_implicit_strength_gets_stability_corridor():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_03_denoise")
    uv3._conductor_strength_hints = {"phase_03_denoise": 0.99}

    audio = np.zeros(1024, dtype=np.float32)
    uv3._profiled_phase_call(phase, audio, sample_rate=48000)

    assert isinstance(phase.last_strength, float)
    assert 0.03 <= phase.last_strength <= 0.97


def test_explicit_strength_not_overridden_by_corridor():
    uv3 = UnifiedRestorerV3()
    phase = _DummyPhase("phase_03_denoise")

    audio = np.zeros(1024, dtype=np.float32)
    uv3._profiled_phase_call(phase, audio, sample_rate=48000, strength=0.99)

    assert isinstance(phase.last_strength, float)
    assert abs(phase.last_strength - 0.99) < 1e-9


def test_phase_calibration_scalar_has_interior_pullback():
    profile_low = {"global_scalar": 0.10, "family_scalars": {"denoise": 0.10, "general": 0.10}}
    s_low = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_low)
    assert s_low > 0.30

    profile_high = {"global_scalar": 2.50, "family_scalars": {"denoise": 2.50, "general": 2.50}}
    s_high = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_high)
    assert s_high < 1.80


def test_mid_calibration_production_nachbesserung_deboosts_on_low_artifact_floor():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.89,
        "brillanz": 0.90,
        "waerme": 0.70,
    }
    out = UnifiedRestorerV3._mid_pipeline_calibration_step(
        scores,
        profile,
        "33pct",
        5,
        10,
        artifact_floor=0.94,
    )

    assert out is not None
    fam = out["family_scalars"]
    assert float(fam["transient"]) < 1.0
    assert float(fam["reconstruction"]) < 1.0
    assert float(fam["dynamics_eq"]) < 1.0


def test_continuous_joy_refinement_writes_events_and_adjusts():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.89,
        "brillanz": 0.90,
        "waerme": 0.70,
        "micro_dynamics": 0.80,
        "artikulation": 0.78,
    }
    out = UnifiedRestorerV3._continuous_joy_refinement_step(
        scores,
        profile,
        "phase_23_spectral_repair",
        7,
        20,
        artifact_floor=0.94,
    )

    assert out is not None
    fam = out["family_scalars"]
    assert float(fam["reconstruction"]) < 1.0
    assert float(fam["transient"]) < 1.0
    events = out.get("_joy_closed_loop_events", [])
    assert isinstance(events, list)
    assert len(events) == 1
    assert events[0]["trigger_phase"] == "phase_23_spectral_repair"


def test_continuous_joy_refinement_returns_none_when_no_signal():
    profile = {
        "global_scalar": 1.0,
        "family_scalars": {
            "denoise": 1.0,
            "reverb": 1.0,
            "reconstruction": 1.0,
            "dynamics_eq": 1.0,
            "transient": 1.0,
            "vocal": 1.0,
            "instrument": 1.0,
            "general": 1.0,
        },
    }
    scores = {
        "natuerlichkeit": 0.93,
        "brillanz": 0.80,
        "waerme": 0.80,
        "micro_dynamics": 0.90,
        "artikulation": 0.90,
    }
    out = UnifiedRestorerV3._continuous_joy_refinement_step(
        scores,
        profile,
        "phase_10_declip",
        4,
        20,
        artifact_floor=1.0,
    )
    assert out is None


def test_song_cluster_policy_is_deterministic_and_bounded():
    pol = UnifiedRestorerV3._derive_song_cluster_policy(
        material="mp3_low",
        era_decade=1995,
        genre_label="Rock",
        restorability_tier="poor",
        is_schlager=False,
    )
    assert isinstance(pol.get("cluster_key"), str)
    assert 0.90 <= float(pol["artifact_sensitivity"]) <= 1.25
    assert 0.90 <= float(pol["fatigue_sensitivity"]) <= 1.20
    assert 0.85 <= float(pol["recovery_bias"]) <= 1.20


def test_joy_fatigue_runtime_index_increases_with_better_scores():
    low = UnifiedRestorerV3._compute_joy_fatigue_runtime_index(
        {
            "natuerlichkeit": 0.86,
            "waerme": 0.68,
            "micro_dynamics": 0.76,
            "emotionalitaet": 0.74,
            "brillanz": 0.90,
        },
        {"risk_level": "high"},
        artifact_freedom=0.93,
        emotional_arc_score=0.80,
    )
    high = UnifiedRestorerV3._compute_joy_fatigue_runtime_index(
        {
            "natuerlichkeit": 0.93,
            "waerme": 0.80,
            "micro_dynamics": 0.90,
            "emotionalitaet": 0.89,
            "brillanz": 0.82,
        },
        {"risk_level": "low"},
        artifact_freedom=0.99,
        emotional_arc_score=0.92,
    )
    assert float(high["joy_index"]) > float(low["joy_index"])
    assert float(high["fatigue_index"]) < float(low["fatigue_index"])


def test_auto_improvement_recommendations_contains_actionable_items():
    rec = UnifiedRestorerV3._derive_auto_improvement_recommendations(
        musical_goal_scores={
            "natuerlichkeit": 0.88,
            "waerme": 0.70,
            "brillanz": 0.90,
            "micro_dynamics": 0.79,
        },
        musical_goals_passed={"natuerlichkeit": False},
        artifact_freedom=0.94,
        phase_regression_log={"phase_03_denoise": -2.1},
        top_defects=[{"type": "noise", "severity": 0.72}],
    )
    assert isinstance(rec, dict)
    assert int(rec.get("count", 0)) >= 3
    assert isinstance(rec.get("recommendations", []), list)
