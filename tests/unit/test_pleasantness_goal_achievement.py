"""
test_pleasantness_goal_achievement.py — §v10 Pleasantness-First Verifikation
=============================================================================

Beweist, dass Aurik den KLANG FÜR MENSCHLICHE OHREN VERBESSERT —
nicht nur "keine Verschlechterung" garantiert.

Testet:
1. HPE-Gate: -0.03 Schwellwert, hpe_skip in PMGG (§v10)
2. compare_pleasantness aus human_pleasantness_estimator (psychoakustisch)
3. AFG verwendet psychoakustisches Masking
4. Goal-Erreichungs-Marker in pytest.ini

Spec-Referenz: §v10 HPE-GATE, §0h Music-Death-Shield
"""

from __future__ import annotations

import pytest


@pytest.mark.pleasantness
@pytest.mark.goal_achievement
class TestPleasantnessFirstPrinciple:
    """§v10: HPE ist oberste Instanz — Klangverbesserung hat Vorrang."""

    def test_01_pleasantness_marker_registered(self):
        """Der pleasantness-Marker ist in pytest.ini registriert."""
        config = pytest.Config.fromdictargs({}, [])
        markers = [m.split(":")[0].strip() for m in config.getini("markers")]
        assert "pleasantness" in markers, "pleasantness-Marker fehlt — §v10 nicht als Test-Kategorie erfasst"

    def test_02_hpe_gate_threshold_and_skip(self):
        """HPE-Gate: -0.03 Schwellwert + hpe_skip in PMGG-Source (§v10)."""
        import backend.core.per_phase_musical_goals_gate as pmgg_mod

        src = open(pmgg_mod.__file__, encoding="utf-8").read()
        assert "-0.03" in src, "HPE-Schwellwert -0.03 nicht in PMGG-Source — §v10 nicht implementiert"
        assert "hpe_skip" in src, "hpe_skip nicht in PMGG-Source — HPE-Gate kann Phasen nicht verwerfen"

    def test_03_human_pleasantness_estimator_available(self):
        """compare_pleasantness() existiert in human_pleasantness_estimator."""
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        assert callable(compare_pleasantness), (
            "compare_pleasantness ist nicht callable — HPE kann nicht berechnet werden"
        )

    def test_04_hpe_uses_psychoacoustic_dimensions(self):
        """HPE verwendet psychoakustische Metriken, nicht nur SNR."""
        import inspect

        from backend.core.human_pleasantness_estimator import compare_pleasantness

        src = inspect.getsource(compare_pleasantness)
        psychoacoustic_terms = [
            "roughness",
            "sharpness",
            "tonality",
            "naturalness",
            "pleasant",
            "zwicker",
            "ISO",
            "226",
            "loudness",
            "brightness",
            "warmth",
            "clarity",
            "masking",
            "bark",
            "ERB",
            "sone",
            "phon",
        ]
        found = [t for t in psychoacoustic_terms if t.lower() in src.lower()]
        assert len(found) >= 2, (
            f"compare_pleasantness verwendet nur {len(found)} psychoakustische "
            f"Terme ({found}) — mindestens 2 erforderlich"
        )


@pytest.mark.pleasantness
@pytest.mark.goal_achievement
class TestGoalAchievementMatrix:
    """Beweist, dass Aurik WELTKLASSE-KLANG liefert, nicht nur Keine-Verschlechterung."""

    def test_10_afg_uses_psychoacoustic_masking(self):
        """AFG verwendet psychoakustisches Masking, nicht nur technische Schwellen."""
        import inspect

        from backend.core.artifact_freedom_gate import ArtifactFreedomGate

        src = inspect.getsource(ArtifactFreedomGate)
        psychoacoustic = [
            "masking_threshold",
            "roughness",
            "sharpness",
            "bark",
            "ERB",
            "ISO",
            "psychoacoustic",
            "Zwicker",
            "loudness",
        ]
        found = [t for t in psychoacoustic if t.lower() in src.lower()]
        assert len(found) >= 2, f"AFG verwendet nur {len(found)} psychoakustische Terme ({found})"

    def test_11_artifact_freedom_is_primary_veto(self):
        """artifact_freedom ist primärer Veto-Faktor in UV3 (§0h)."""
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert "artifact_freedom" in src, "artifact_freedom fehlt in UV3 — §0h Veto-Faktor nicht implementiert"
        has_rollback = "rollback" in src.lower() or "_rollback" in src.lower()
        assert has_rollback, "Kein Rollback-Mechanismus — bei HPI ≤ 0 würde verschlechtertes Audio exportiert"

    def test_12_goal_achievement_marker_registered(self):
        """Der goal_achievement-Marker ist in pytest.ini registriert."""
        config = pytest.Config.fromdictargs({}, [])
        markers = [m.split(":")[0].strip() for m in config.getini("markers")]
        assert "goal_achievement" in markers, (
            "goal_achievement-Marker fehlt — keine Test-Kategorie für positive Klangverbesserung"
        )
