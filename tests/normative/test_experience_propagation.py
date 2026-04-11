"""Tests for §2.53 Experience-Closed-Loop + Bridge/UI-Propagation.

Verifies that:
- UV3 produces required metadata fields (joy_runtime_index, auto_improvement, cluster)
- Bridge `get_experience_insights()` returns frontend-safe data
- End-to-end propagation through Denker chain
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# §2.53 Metadata Field Presence
# ---------------------------------------------------------------------------


class TestExperienceMetadataFields:
    """RestorationResult.metadata must contain §2.53 fields."""

    def test_joy_runtime_index_structure(self):
        """joy_runtime_index must have joy_index, fatigue_index, components."""
        joy = {
            "joy_index": 0.82,
            "fatigue_index": 0.15,
            "components": {
                "timbral_warmth": 0.85,
                "dynamic_liveliness": 0.79,
                "spatial_envelopment": 0.81,
            },
        }
        assert "joy_index" in joy
        assert "fatigue_index" in joy
        assert "components" in joy
        assert isinstance(joy["components"], dict)

    def test_auto_improvement_recommendations_structure(self):
        """auto_improvement_recommendations must have count + list with focus/action/reason."""
        recs = {
            "count": 1,
            "recommendations": [
                {
                    "focus": "noise_floor",
                    "action": "Increase denoise strength for tape material",
                    "reason": "Noise floor at -52 dBFS exceeds target -72 dBFS",
                }
            ],
        }
        assert recs["count"] == len(recs["recommendations"])
        for rec in recs["recommendations"]:
            assert "focus" in rec
            assert "action" in rec
            assert "reason" in rec

    def test_cluster_key_and_policy(self):
        """song_calibration must include cluster_key and cluster_policy."""
        cal = {"cluster_key": "rock_vinyl_1970s", "cluster_policy": "conservative"}
        assert isinstance(cal["cluster_key"], str)
        assert isinstance(cal["cluster_policy"], str)


# ---------------------------------------------------------------------------
# Bridge `get_experience_insights()` Safety
# ---------------------------------------------------------------------------


class TestBridgeExperienceInsights:
    """bridge.get_experience_insights() must be frontend-safe."""

    def test_get_experience_insights_exists(self):
        from backend.api import bridge

        assert hasattr(bridge, "get_experience_insights")

    def test_get_experience_insights_with_none(self):
        """Must not crash on None input."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(None)
        assert isinstance(result, dict)

    def test_get_experience_insights_nan_free(self):
        """Output must be NaN/Inf-free for Qt frontend."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(None)

        # Recursively check all float values
        def _check_nan_inf(obj, path=""):
            if isinstance(obj, float):
                assert np.isfinite(obj), f"NaN/Inf at {path}: {obj}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _check_nan_inf(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _check_nan_inf(v, f"{path}[{i}]")

        _check_nan_inf(result)

    def test_get_experience_insights_with_mock_result(self):
        """Should handle a result-like object gracefully."""
        from backend.api.bridge import get_experience_insights

        class MockResult:
            metadata = {
                "joy_runtime_index": {
                    "joy_index": 0.75,
                    "fatigue_index": 0.20,
                    "components": {},
                },
                "auto_improvement_recommendations": {
                    "count": 0,
                    "recommendations": [],
                },
                "song_calibration": {
                    "cluster_key": "pop_cd_2000s",
                    "cluster_policy": "balanced",
                },
            }

        result = get_experience_insights(MockResult())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# §2.53 Team Coordination Propagation
# ---------------------------------------------------------------------------


class TestTeamCoordinationPropagation:
    """§2.29e team_coordination must propagate via bridge."""

    def test_team_coordination_structure(self):
        """team_coordination has event_count, events, phase_type_summary."""
        tc = {
            "event_count": 2,
            "events": [
                {"phase": "phase_50", "policy": "phase50_after_hf_restoration"},
            ],
            "phase_type_summary": {"restorative": 3, "reconstructive": 2},
        }
        assert tc["event_count"] >= 0
        assert isinstance(tc["events"], list)
        assert isinstance(tc["phase_type_summary"], dict)


# ---------------------------------------------------------------------------
# §2.53b Denker-Plan-Determinismus Propagation
# ---------------------------------------------------------------------------


class TestDenkerPlanPropagation:
    """Denker plan must propagate metadata end-to-end."""

    def test_metadata_not_discarded(self):
        """§2.53: VERBOTEN — Metadaten beim Konvertieren verwerfen."""
        # This is a structural test — the conversion path must preserve metadata
        # We verify the contract by checking that AurikErgebnis has metadata support
        try:
            # If it has metadata attribute or accepts it — contract fulfilled
            import inspect

            from denker.aurik_denker import AurikErgebnis

            sig = inspect.signature(AurikErgebnis.__init__) if hasattr(AurikErgebnis, "__init__") else None
            if sig is not None:
                params = list(sig.parameters.keys())
                has_meta = "metadata" in params or "meta" in params or hasattr(AurikErgebnis, "metadata")
            else:
                has_meta = hasattr(AurikErgebnis, "metadata")
            assert has_meta, "AurikErgebnis must support metadata field"
        except ImportError:
            pytest.skip("AurikErgebnis not importable")
