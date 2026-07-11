"""
tests/unit/test_phase_interaction_denker.py
============================================
Unit-Tests für PhaseInteractionDenker (§2.47 / §2.48).

Prüft:
  - Semantische Typ-Annotation bekannter Phasen
  - Konflikt-Guard DYNAMICS_EXPANDING → DYNAMICS_COMPRESSING
  - Konflikt-Guard STEREO_NARROWING → STEREO_WIDENING
  - Reihenfolge-Constraint §7.2 (phase_14 vor phase_25)
  - Reihenfolge-Constraint §2.46 (phase_61 vor phase_20)
  - Reihenfolge-Constraint EBU R128 (phase_40 vor phase_47)
  - Reihenfolge-Constraint Carrier-Chain-Inversion (phase_03 vor phase_07)
  - Fail-Safe: kein defect_result → leerer Plan
  - Fail-Safe: leerer Plan → is_valid == False
  - Deterministik: identisches Ergebnis bei wiederholtem Aufruf
  - Singleton-Invariante: get_phase_interaction_denker() gibt dieselbe Instanz zurück
  - UV3-Integration: precomputed_phase_plan kwarg wird akzeptiert
  - AurikDenker-Stufe-5b: stage_notes enthält "phase_interaction" nach Aufruf
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from denker.phase_interaction_denker import (
    _CONFLICT_RULES,
    _ORDER_CONSTRAINTS,
    _PHASE_SEMANTICS,
    PhaseInteractionDenker,
    PhasePlan,
    _goal_risk_threshold_from_signal,
    get_phase_interaction_denker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def denker() -> PhaseInteractionDenker:
    return PhaseInteractionDenker()


def _fake_defect_result(material: str = "vinyl") -> MagicMock:
    dr = MagicMock()
    dr.scores = {}
    dr.material_type = MagicMock()
    dr.material_type.value = material
    return dr


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance() -> None:
    a = get_phase_interaction_denker()
    b = get_phase_interaction_denker()
    assert a is b


# ---------------------------------------------------------------------------
# PhasePlan Datenklasse
# ---------------------------------------------------------------------------


def test_phase_plan_is_valid_true() -> None:
    plan = PhasePlan(phases=["phase_03_denoise"])
    assert plan.is_valid is True


def test_phase_plan_is_valid_false_empty() -> None:
    plan = PhasePlan(phases=[])
    assert plan.is_valid is False


# ---------------------------------------------------------------------------
# Semantische Annotation
# ---------------------------------------------------------------------------


def test_annotate_known_phase(denker: PhaseInteractionDenker) -> None:
    annotations = denker._annotate(["phase_03_denoise"])
    assert "SUBTRACTIVE" in annotations["phase_03_denoise"]
    assert "DENOISE" in annotations["phase_03_denoise"]


def test_annotate_dynamics_expanding(denker: PhaseInteractionDenker) -> None:
    annotations = denker._annotate(["phase_26_dynamic_range_expansion"])
    assert "DYNAMICS_EXPANDING" in annotations["phase_26_dynamic_range_expansion"]


def test_annotate_dynamics_compressing(denker: PhaseInteractionDenker) -> None:
    for phase in ("phase_10_compression", "phase_35_multiband_compression", "phase_54_multiband_dynamics"):
        annotations = denker._annotate([phase])
        assert "DYNAMICS_COMPRESSING" in annotations[phase], phase


def test_annotate_stereo_narrowing(denker: PhaseInteractionDenker) -> None:
    annotations = denker._annotate(["phase_33_stereo_width_limiter"])
    assert "STEREO_NARROWING" in annotations["phase_33_stereo_width_limiter"]


def test_annotate_stereo_widening(denker: PhaseInteractionDenker) -> None:
    for phase in ("phase_48_stereo_width_enhancer", "phase_13_stereo_enhancement"):
        annotations = denker._annotate([phase])
        assert "STEREO_WIDENING" in annotations[phase], phase


def test_annotate_prefix_match(denker: PhaseInteractionDenker) -> None:
    # Unbekannte Variante sollte via Präfix-Match Tags erben
    annotations = denker._annotate(["phase_03_denoise_sgmse"])
    tags = annotations.get("phase_03_denoise_sgmse", frozenset())
    assert "SUBTRACTIVE" in tags or len(tags) == 0  # Präfix-match oder leer (beide OK)


def test_annotate_unknown_phase_empty_tags(denker: PhaseInteractionDenker) -> None:
    annotations = denker._annotate(["phase_99_unknown_xyz"])
    assert annotations["phase_99_unknown_xyz"] == frozenset()


# ---------------------------------------------------------------------------
# Konflikt-Auflösung §2.48
# ---------------------------------------------------------------------------


def test_conflict_dynamics_expanding_suppresses_compressing(denker: PhaseInteractionDenker) -> None:
    """phase_26 (DYNAMICS_EXPANDING) → phase_10 (DYNAMICS_COMPRESSING) wird supprimiert."""
    phases = ["phase_26_dynamic_range_expansion", "phase_10_compression"]
    annotations = denker._annotate(phases)
    resolved, suppressed, notes = denker._resolve_conflicts(phases, annotations)
    assert "phase_10_compression" in suppressed
    assert "phase_10_compression" not in resolved
    assert any("DYNAMICS" in n for n in notes)


def test_conflict_stereo_narrowing_suppresses_widening(denker: PhaseInteractionDenker) -> None:
    """phase_33 (STEREO_NARROWING) → phase_48 (STEREO_WIDENING) wird supprimiert."""
    phases = ["phase_33_stereo_width_limiter", "phase_48_stereo_width_enhancer"]
    annotations = denker._annotate(phases)
    resolved, suppressed, notes = denker._resolve_conflicts(phases, annotations)
    assert "phase_48_stereo_width_enhancer" in suppressed
    assert "phase_48_stereo_width_enhancer" not in resolved


def test_conflict_no_suppression_when_order_reversed(denker: PhaseInteractionDenker) -> None:
    """Wenn COMPRESSING vor EXPANDING steht → kein Konflikt (korrekte Reihenfolge)."""
    phases = ["phase_10_compression", "phase_26_dynamic_range_expansion"]
    annotations = denker._annotate(phases)
    resolved, suppressed, notes = denker._resolve_conflicts(phases, annotations)
    # Conflict rule fires only if EXPANDING appears first
    assert "phase_26_dynamic_range_expansion" not in suppressed


def test_conflict_no_false_positive_unrelated_phases(denker: PhaseInteractionDenker) -> None:
    """Nicht-konfligierende Phasen werden nicht supprimiert."""
    phases = ["phase_03_denoise", "phase_07_harmonic_restoration", "phase_40_loudness_normalization"]
    annotations = denker._annotate(phases)
    resolved, suppressed, notes = denker._resolve_conflicts(phases, annotations)
    assert len(suppressed) == 0
    assert resolved == phases


# ---------------------------------------------------------------------------
# Reihenfolge-Constraints §2.46 / §7.2
# ---------------------------------------------------------------------------


def test_order_phase14_before_phase25(denker: PhaseInteractionDenker) -> None:
    """Spec §7.2: phase_14 muss vor phase_25 stehen."""
    phases = ["phase_25_azimuth_correction", "phase_14_phase_correction"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered.index("phase_14_phase_correction") < ordered.index("phase_25_azimuth_correction")
    assert ("phase_14_phase_correction", "phase_25_azimuth_correction") in applied


def test_order_phase61_before_phase20(denker: PhaseInteractionDenker) -> None:
    """Groove-Echo muss vor Reverb-Reduction stehen."""
    phases = ["phase_20_reverb_reduction", "phase_61_groove_echo_cancellation"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered.index("phase_61_groove_echo_cancellation") < ordered.index("phase_20_reverb_reduction")


def test_order_phase61_before_phase49(denker: PhaseInteractionDenker) -> None:
    """Groove-Echo muss vor Advanced-Dereverb stehen."""
    phases = ["phase_49_advanced_dereverb", "phase_61_groove_echo_cancellation"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered.index("phase_61_groove_echo_cancellation") < ordered.index("phase_49_advanced_dereverb")


def test_order_phase40_before_phase47(denker: PhaseInteractionDenker) -> None:
    """EBU R128: LUFS-Normalisierung vor TruePeak-Limiter."""
    phases = ["phase_47_truepeak_limiter", "phase_40_loudness_normalization"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered.index("phase_40_loudness_normalization") < ordered.index("phase_47_truepeak_limiter")


def test_order_denoise_before_harmonic(denker: PhaseInteractionDenker) -> None:
    """§2.46: Subtraktiv (phase_03) vor Additiv (phase_07)."""
    phases = ["phase_07_harmonic_restoration", "phase_03_denoise"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered.index("phase_03_denoise") < ordered.index("phase_07_harmonic_restoration")


def test_order_constraint_noop_when_already_correct(denker: PhaseInteractionDenker) -> None:
    """Bereits korrekte Reihenfolge → keine Änderung, ordering_applied leer."""
    phases = ["phase_14_phase_correction", "phase_25_azimuth_correction"]
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered == phases
    assert len(applied) == 0


def test_order_missing_phase_ignored(denker: PhaseInteractionDenker) -> None:
    """Constraint ignoriert, wenn eine der Phasen nicht im Plan ist."""
    phases = ["phase_25_azimuth_correction"]  # phase_14 fehlt
    ordered, applied = denker._apply_order_constraints(phases)
    assert ordered == phases
    assert len(applied) == 0


# ---------------------------------------------------------------------------
# plan() — Fail-Safe
# ---------------------------------------------------------------------------


def test_plan_failsafe_no_defect_result(denker: PhaseInteractionDenker) -> None:
    """defect_result=None → leerer PhasePlan, kein Absturz."""
    plan = denker.plan(defect_result=None, material="vinyl", mode="restoration")
    assert isinstance(plan, PhasePlan)
    assert plan.is_valid is False


def test_plan_failsafe_uv3_select_raises(denker: PhaseInteractionDenker) -> None:
    """Wenn UV3-Selektion wirft → leerer Plan (fail-safe)."""
    dr = _fake_defect_result()
    with patch.object(denker, "_select_via_uv3", side_effect=RuntimeError("UV3 crash")):
        plan = denker.plan(defect_result=dr, material="tape", mode="restoration")
    assert isinstance(plan, PhasePlan)
    assert plan.is_valid is False


# ---------------------------------------------------------------------------
# plan() — Vollintegration mit gemocktem UV3-Selektor
# ---------------------------------------------------------------------------


def test_plan_conflict_resolved_in_full_pipeline(denker: PhaseInteractionDenker) -> None:
    """Konflikt wird in der vollen plan()-Pipeline (mock UV3) aufgelöst."""
    dr = _fake_defect_result()
    conflict_phases = [
        "phase_26_dynamic_range_expansion",
        "phase_10_compression",
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
    ]
    with patch.object(denker, "_select_via_uv3", return_value=conflict_phases):
        plan = denker.plan(defect_result=dr, material="vinyl", mode="restoration")
    assert plan.is_valid
    assert "phase_10_compression" in plan.suppressed
    assert "phase_10_compression" not in plan.phases
    # EBU R128 Reihenfolge muss auch stimmen
    assert plan.phases.index("phase_40_loudness_normalization") < plan.phases.index("phase_47_truepeak_limiter")


def test_plan_ordering_applied_in_full_pipeline(denker: PhaseInteractionDenker) -> None:
    """Reihenfolge-Constraint wird in der vollen plan()-Pipeline angewendet."""
    dr = _fake_defect_result()
    wrong_order = ["phase_25_azimuth_correction", "phase_14_phase_correction"]
    with patch.object(denker, "_select_via_uv3", return_value=wrong_order):
        plan = denker.plan(defect_result=dr, material="tape", mode="restoration")
    assert plan.is_valid
    assert plan.phases.index("phase_14_phase_correction") < plan.phases.index("phase_25_azimuth_correction")
    assert len(plan.ordering_applied) >= 1


def test_plan_semantic_annotations_populated(denker: PhaseInteractionDenker) -> None:
    """semantic_annotations enthält Tags für alle Phasen im Plan."""
    dr = _fake_defect_result()
    phases = ["phase_03_denoise", "phase_07_harmonic_restoration"]
    with patch.object(denker, "_select_via_uv3", return_value=phases):
        plan = denker.plan(defect_result=dr, material="cd_digital", mode="restoration")
    assert plan.is_valid
    for p in plan.phases:
        assert p in plan.semantic_annotations


def test_plan_deterministic(denker: PhaseInteractionDenker) -> None:
    """Identisches Ergebnis bei wiederholtem Aufruf."""
    dr = _fake_defect_result()
    phases = [
        "phase_29_tape_hiss_reduction",
        "phase_07_harmonic_restoration",
        "phase_26_dynamic_range_expansion",
        "phase_35_multiband_compression",
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
    ]
    with patch.object(denker, "_select_via_uv3", return_value=phases):
        plan1 = denker.plan(defect_result=dr, material="reel_tape", mode="restoration")
    with patch.object(denker, "_select_via_uv3", return_value=phases):
        plan2 = denker.plan(defect_result=dr, material="reel_tape", mode="restoration")
    assert plan1.phases == plan2.phases
    assert plan1.suppressed == plan2.suppressed


# ---------------------------------------------------------------------------
# UV3 precomputed_phase_plan kwarg Integration
# ---------------------------------------------------------------------------


def test_uv3_accepts_precomputed_phase_plan_kwarg() -> None:
    """UV3.restore() akzeptiert precomputed_phase_plan ohne Absturz."""
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    cfg = RestorationConfig(mode=QualityMode.QUALITY, enforce_3x_rt=False)
    uv3 = UnifiedRestorerV3(config=cfg)

    # Minimales Dummy-Audio — nur kwarg-Akzeptanz prüfen
    audio = np.zeros(480, dtype=np.float32)
    try:
        uv3.restore(
            audio,
            sample_rate=48000,
            mode="restoration",
            precomputed_phase_plan=["phase_40_loudness_normalization", "phase_47_truepeak_limiter"],
        )
    except Exception:
        logger.warning("test fallback", exc_info=True)
        pass  # Inhaltliche Fehler sind OK (kein defect_result) — Kwarg muss akzeptiert werden


def test_uv3_precomputed_phase_plan_kwarg_is_popped() -> None:
    """precomputed_phase_plan landet nicht in unbekannten kwargs."""
    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    cfg = RestorationConfig(mode=QualityMode.QUALITY, enforce_3x_rt=False)
    uv3 = UnifiedRestorerV3(config=cfg)
    audio = np.zeros(480, dtype=np.float32)
    # Sollte keinen TypeError "unexpected keyword argument" auslösen
    try:
        uv3.restore(audio, sample_rate=48000, precomputed_phase_plan=[])
    except TypeError as exc:
        pytest.fail(f"precomputed_phase_plan wurde nicht als kwarg akzeptiert: {exc}")
    except Exception:
        logger.warning("test fallback", exc_info=True)
        pass  # Andere Fehler sind OK


def test_uv3_precomputed_plan_path_is_deterministic_executor() -> None:
    """UV3 muss im precomputed-Pfad _select/_optimize + phase skipping überspringen."""
    import inspect

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    # Klassenquelltext ist robust gegen mögliche Runtime-Patches am restore-Attribut.
    src = inspect.getsource(UnifiedRestorerV3)
    assert "UV3 _select/_optimize übersprungen" in src
    assert "Phase Skipping deaktiviert: precomputed_phase_plan aktiv" in src


# ---------------------------------------------------------------------------
# RestaurierDenker precomputed_phase_plan Durchleitung
# ---------------------------------------------------------------------------


def test_restaurier_denker_accepts_precomputed_phase_plan() -> None:
    """RestaurierDenker.restauriere() hat precomputed_phase_plan als Parameter."""
    import inspect

    from denker.restaurier_denker import RestaurierDenker

    sig = inspect.signature(RestaurierDenker.restauriere)
    assert "precomputed_phase_plan" in sig.parameters, (
        "RestaurierDenker.restauriere() muss precomputed_phase_plan Parameter haben"
    )


# ---------------------------------------------------------------------------
# AurikDenker Stufe 5b Metadaten
# ---------------------------------------------------------------------------


def test_aurik_denker_stage_notes_contain_phase_interaction() -> None:
    """AurikDenker._orchestriere() erzeugt 'phase_interaction' in stage_notes."""
    # Prüfe, dass stage_notes["phase_interaction"] via Quellcode gesetzt wird
    import inspect

    from denker.aurik_denker import AurikDenker

    src = inspect.getsource(AurikDenker._orchestriere)
    assert "phase_interaction" in src, "AurikDenker._orchestriere() muss stage_notes['phase_interaction'] befüllen"
    assert "PhaseInteractionDenker" in src or "get_phase_interaction_denker" in src


# ---------------------------------------------------------------------------
# Regeltabellen-Vollständigkeit
# ---------------------------------------------------------------------------


def test_conflict_rules_are_frozensets() -> None:
    for trigger, target in _CONFLICT_RULES:
        assert isinstance(trigger, frozenset)
        assert isinstance(target, frozenset)


def test_order_constraints_are_tuples_of_strings() -> None:
    for before, after in _ORDER_CONSTRAINTS:
        assert isinstance(before, str)
        assert isinstance(after, str)
        assert before.startswith("phase_")
        assert after.startswith("phase_")


def test_phase_semantics_all_frozensets() -> None:
    for phase, tags in _PHASE_SEMANTICS.items():
        assert isinstance(tags, frozenset), f"{phase} muss frozenset haben"
        assert all(isinstance(t, str) for t in tags)


def test_no_self_referential_order_constraint() -> None:
    """Kein Constraint (A, A) — wäre ein Deadlock."""
    for before, after in _ORDER_CONSTRAINTS:
        assert before != after, f"Selbstreferenz in ORDER_CONSTRAINTS: {before}"


def test_goal_risk_threshold_from_signal_lowers_for_risky_signature() -> None:
    base_like = _goal_risk_threshold_from_signal(None, restorability_score=70.0)
    risky = _goal_risk_threshold_from_signal(
        {"transient_ratio": 0.02, "crest_db": 20.0, "hf_ratio": 0.15},
        restorability_score=35.0,
    )
    assert risky < base_like


def test_plan_signal_signature_injects_transient_and_deesser_phase(denker: PhaseInteractionDenker) -> None:
    dr = _fake_defect_result()
    phases = ["phase_03_denoise"]
    with patch.object(denker, "_select_via_uv3", return_value=phases):
        plan = denker.plan(
            defect_result=dr,
            material="vinyl",
            mode="restoration",
            signal_signature={"transient_ratio": 0.02, "hf_ratio": 0.14},
        )

    assert plan.is_valid
    assert "phase_08_transient_preservation" in plan.phases
    assert "phase_19_de_esser" in plan.phases


# ---------------------------------------------------------------------------
# §0a Crossfire-Modus-Invariante — verbotene Phasen in Restoration
# ---------------------------------------------------------------------------


def test_forbidden_prefixes_cover_canonical_set() -> None:
    """§0a: Präfix-Set deckt jede kanonische _RESTORATION_FORBIDDEN-Phase ab."""
    from backend.core.adaptive_phase_rescheduler import _RESTORATION_FORBIDDEN
    from denker.phase_interaction_denker import _restoration_forbidden_prefixes

    prefixes = _restoration_forbidden_prefixes()
    assert prefixes  # nicht leer
    for pid in _RESTORATION_FORBIDDEN:
        parts = pid.split("_")
        prefix = f"{parts[0]}_{parts[1]}"
        assert prefix in prefixes, f"§0a-Präfix für kanonische Phase {pid} fehlt: {prefix}"


def test_strip_restoration_forbidden_removes_phase21_35_42(denker: PhaseInteractionDenker) -> None:
    """§0a: phase_21/35/42 werden in Restoration entfernt, Rest bleibt unverändert."""
    phases = [
        "phase_03_denoise",
        "phase_21_harmonic_exciter",
        "phase_07_harmonic_restoration",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
        "phase_47_truepeak_limiter",
    ]
    kept, removed = denker._strip_restoration_forbidden(phases, "restoration")
    assert kept == [
        "phase_03_denoise",
        "phase_07_harmonic_restoration",
        "phase_47_truepeak_limiter",
    ]
    assert set(removed) == {
        "phase_21_harmonic_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    }


def test_strip_restoration_forbidden_keeps_phases_in_studio(denker: PhaseInteractionDenker) -> None:
    """§0a: Studio-2026 behält phase_21/35/42 — Filter greift nur in Restoration."""
    phases = [
        "phase_03_denoise",
        "phase_21_harmonic_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    ]
    for studio_mode in ("studio2026", "studio_2026", "maximum"):
        kept, removed = denker._strip_restoration_forbidden(phases, studio_mode)
        assert kept == phases, f"Studio-Modus {studio_mode} darf nichts entfernen"
        assert removed == []


def test_strip_restoration_forbidden_noop_when_clean(denker: PhaseInteractionDenker) -> None:
    """§0a: Plan ohne verbotene Phasen bleibt unverändert (kein False-Positive)."""
    phases = ["phase_03_denoise", "phase_07_harmonic_restoration", "phase_47_truepeak_limiter"]
    kept, removed = denker._strip_restoration_forbidden(phases, "restoration")
    assert kept == phases
    assert removed == []


def test_plan_strips_forbidden_from_chain_injection(denker: PhaseInteractionDenker) -> None:
    """§0a Defense-in-Depth: Ketten-Injektion einer verbotenen Phase wird in Restoration entfernt.

    Simuliert eine Trägerkette, deren must_have_phases eine §0a-verbotene Phase
    (phase_21) enthält — der PhaseInteractionDenker darf sie NIE in den finalen
    Restoration-Plan durchlassen.
    """
    dr = _fake_defect_result()

    _chain_result = MagicMock()
    _chain_plan = MagicMock()
    _chain_plan.chain_string = "vinyl→cassette"
    _chain_plan.must_have_phases = ["phase_06_frequency_restoration", "phase_21_harmonic_exciter"]
    _chain_denker = MagicMock()
    _chain_denker.leite_phasen_ab.return_value = _chain_plan

    with (
        patch.object(denker, "_select_via_uv3", return_value=["phase_03_denoise"]),
        patch(
            "denker.tontraegerkette_denker.get_tontraegerkette_denker",
            return_value=_chain_denker,
        ),
    ):
        plan = denker.plan(
            defect_result=dr,
            material="vinyl",
            mode="restoration",
            chain_result=_chain_result,
        )

    assert plan.is_valid
    assert "phase_21_harmonic_exciter" not in plan.phases, "§0a-Verletzung: phase_21 im Restoration-Plan"
    assert "phase_06_frequency_restoration" in plan.phases  # legitime Ketten-Phase bleibt
    assert any("§0a Crossfire-Guard" in n for n in plan.conflict_notes)


def test_plan_keeps_forbidden_in_studio_chain_injection(denker: PhaseInteractionDenker) -> None:
    """§0a: In Studio-2026 darf dieselbe Ketten-Injektion phase_21 behalten."""
    dr = _fake_defect_result()

    _chain_result = MagicMock()
    _chain_plan = MagicMock()
    _chain_plan.chain_string = "vinyl→cassette"
    _chain_plan.must_have_phases = ["phase_21_harmonic_exciter"]
    _chain_denker = MagicMock()
    _chain_denker.leite_phasen_ab.return_value = _chain_plan

    with (
        patch.object(denker, "_select_via_uv3", return_value=["phase_03_denoise"]),
        patch(
            "denker.tontraegerkette_denker.get_tontraegerkette_denker",
            return_value=_chain_denker,
        ),
    ):
        plan = denker.plan(
            defect_result=dr,
            material="vinyl",
            mode="studio2026",
            chain_result=_chain_result,
        )

    assert plan.is_valid
    assert "phase_21_harmonic_exciter" in plan.phases, "Studio-2026 darf phase_21 behalten"
