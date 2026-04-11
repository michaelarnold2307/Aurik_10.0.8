"""
denker/phase_interaction_denker.py — PhaseInteractionDenker
=============================================================
Steuernde Intelligenz für Phase-Orchestrierung in Aurik 9.

Übernimmt von UnifiedRestorerV3:
  - semantische Typ-Annotation jeder Phase
  - typ-paar-basierte Konflikterkennung (§2.48)
  - constraint-getriebene Reihenfolge (Spec §2.46 Carrier-Chain-Inversion)

Bislang war diese Logik in UV3._select_phases() (1400+ Zeilen) und
UV3._optimize_phase_plan_intelligence() als hartcodierte if/elif-Kaskaden
und Einzelregel-Guards eingebettet. Neue Phasen erforderten neue Guards.

Dieser Denker ersetzt das durch semantische Regeln:
  1. UV3._select_phases() läuft als Werkzeug (Defekt-zu-Phasen-Mapping bleibt dort)
  2. UV3._optimize_phase_plan_intelligence() läuft als Werkzeug (kausal-bewusstes Ordering)
  3. PhaseInteractionDenker verfeinert das Ergebnis: Typ-basierte Konflikte + Constraints
  4. PhasePlan.phases wird via precomputed_phase_plan-kwarg an UV3.restore() übergeben

UV3 ist dann reiner Executor — keine Orchestrierungs-Entscheidungen mehr in restore().

Aufruf (AurikDenker Stufe 5b, nach StrategieDenker, vor _run_rest):
    plan = get_phase_interaction_denker().plan(
        defect_result=cached_defect_result,
        material=material,
        mode=effective_mode,
        chain_info=chain_info,
        defekt_hint=_defekt_hint,
        audio=aktuelles_audio,
        sr=sr,
        restorability_score=float(getattr(cached_restorability_result, "restorability_score", 70.0)),
    )
    # plan.phases → precomputed_phase_plan → RestaurierDenker → UV3.restore()

Spec §2.46 Carrier-Chain-Inversion + §2.47 Adaptive-Intelligence + §2.48 Kumulative-Guard
v9.11.1 — PhaseInteractionDenker
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: PhaseInteractionDenker | None = None
_lock = threading.Lock()


def get_phase_interaction_denker() -> PhaseInteractionDenker:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PhaseInteractionDenker()
    return _instance


# ---------------------------------------------------------------------------
# Semantische Typ-Taxonomie (§2.48)
# ---------------------------------------------------------------------------
# Jede Phase erhält eine Menge semantischer Tags.
# Neue Phasen: einfach eintragen — Konfliktregeln greifen automatisch.
# Präfix-Matching: "phase_03_denoise_ml" erbt Tags von "phase_03_denoise".

_PHASE_SEMANTICS: dict[str, frozenset[str]] = {
    # ── Subtraktiv: Rauschen / Artefakte entfernen ──────────────────────────
    "phase_01_click_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_02_hum_removal": frozenset({"SUBTRACTIVE", "SPECTRAL_NOTCH"}),
    "phase_03_denoise": frozenset({"SUBTRACTIVE", "DENOISE", "BROADBAND"}),
    "phase_05_rumble_filter": frozenset({"SUBTRACTIVE", "LF_SUPPRESS"}),
    "phase_09_crackle_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_18_noise_gate": frozenset({"SUBTRACTIVE", "GATING"}),
    "phase_20_reverb_reduction": frozenset({"SUBTRACTIVE", "REVERB"}),
    "phase_27_click_pop_removal": frozenset({"SUBTRACTIVE", "TRANSIENT_SUPPRESS"}),
    "phase_28_dc_offset_normalizer": frozenset({"SUBTRACTIVE", "DC"}),
    "phase_29_tape_hiss_reduction": frozenset({"SUBTRACTIVE", "DENOISE", "SPECTRAL"}),
    "phase_30_dc_offset_removal": frozenset({"SUBTRACTIVE", "DC"}),
    "phase_43_mp_senet_enhancement": frozenset({"SUBTRACTIVE", "DENOISE", "ML"}),
    "phase_49_advanced_dereverb": frozenset({"SUBTRACTIVE", "REVERB"}),
    # ── Additiv: Spektrum / Energie ergänzen ────────────────────────────────
    "phase_06_frequency_restoration": frozenset({"ADDITIVE", "FREQUENCY_EXT"}),
    "phase_07_harmonic_restoration": frozenset({"ADDITIVE", "HARMONIC"}),
    "phase_11_transient_shaper": frozenset({"ADDITIVE", "TRANSIENT_ADD"}),
    "phase_21_harmonic_exciter": frozenset({"ADDITIVE", "HARMONIC"}),
    "phase_38_presence_boost": frozenset({"ADDITIVE", "FREQUENCY_EXT", "HF"}),
    "phase_55_diffusion_inpainting": frozenset({"ADDITIVE", "INPAINTING"}),
    # ── Dynamik-Expansion ───────────────────────────────────────────────────
    "phase_26_dynamic_range_expansion": frozenset({"DYNAMICS_EXPANDING"}),
    # ── Dynamik-Kompression ─────────────────────────────────────────────────
    "phase_10_compression": frozenset({"DYNAMICS_COMPRESSING"}),
    "phase_35_multiband_compression": frozenset({"DYNAMICS_COMPRESSING"}),
    "phase_54_multiband_dynamics": frozenset({"DYNAMICS_COMPRESSING"}),
    # ── Stereo: Korrigierend (muss vor Azimuth stehen — §7.2) ───────────────
    "phase_14_phase_correction": frozenset({"STEREO_CORRECTIVE", "PRE_AZIMUTH_REQUIRED"}),
    "phase_15_stereo_balance": frozenset({"STEREO_CORRECTIVE"}),
    # ── Stereo: Einengend ───────────────────────────────────────────────────
    "phase_33_stereo_width_limiter": frozenset({"STEREO_NARROWING"}),
    # ── Stereo: Erweiternd ──────────────────────────────────────────────────
    "phase_13_stereo_enhancement": frozenset({"STEREO_WIDENING"}),
    "phase_48_stereo_width_enhancer": frozenset({"STEREO_WIDENING"}),
    # ── Azimuth (benötigt PRE_AZIMUTH_REQUIRED davor — Spec §7.2) ───────────
    "phase_25_azimuth_correction": frozenset({"AZIMUTH", "NEEDS_PRE_AZIMUTH"}),
    # ── Groove-Echo (muss Reverb-Reduktion vorangehen) ──────────────────────
    "phase_61_groove_echo_cancellation": frozenset({"PRE_REVERB_REQUIRED"}),
    # ── Mastering / Output (EBU R128-Kette) ─────────────────────────────────
    "phase_16_final_eq": frozenset({"MASTERING", "EQ"}),
    "phase_17_mastering_polish": frozenset({"MASTERING", "POLISH"}),
    "phase_40_loudness_normalization": frozenset({"MASTERING", "LOUDNESS_NORMALIZATION"}),
    "phase_47_truepeak_limiter": frozenset({"MASTERING", "LIMITER", "NEEDS_LOUDNESS"}),
    "phase_41_output_format_optimization": frozenset({"OUTPUT"}),
}

# ---------------------------------------------------------------------------
# Konfliktregeln — semantisch, nicht phasenname-gebunden (§2.48)
# ---------------------------------------------------------------------------
# Tupel (Auslöser-Tags, Ziel-Tags):
#   Wenn Phase A Auslöser-Tags hat UND Phase B Ziel-Tags hat UND B nach A kommt
#   → B wird supprimiert (§2.48 "suppress_later").
# Neue Konflikte: einfach eintragen — kein Phasen-Code ändern.

_CONFLICT_RULES: list[tuple[frozenset[str], frozenset[str]]] = [
    # Dynamik-Expansion → kein anschließendes Komprimieren (hebt Expansion auf)
    (frozenset({"DYNAMICS_EXPANDING"}), frozenset({"DYNAMICS_COMPRESSING"})),
    # Stereo-Einengung → kein anschließendes Erweitern (hebt Einengung auf)
    (frozenset({"STEREO_NARROWING"}), frozenset({"STEREO_WIDENING"})),
]

# ---------------------------------------------------------------------------
# Reihenfolge-Constraints — deklarative Tabelle (§2.46 / §7.2)
# ---------------------------------------------------------------------------
# Tupel (Phase A, Phase B): A muss im Plan vor B stehen.
# Gilt nur wenn beide Phasen selektiert sind.

_ORDER_CONSTRAINTS: list[tuple[str, str]] = [
    # Spec §7.2 CAUSE_TO_PHASES azimuth_error: Phase-Korrektur vor Azimuth
    ("phase_14_phase_correction", "phase_25_azimuth_correction"),
    # Groove-Echo muss vor Reverb-Reduktion stehen (sonst Fehlidentifikation als Raumhall)
    ("phase_61_groove_echo_cancellation", "phase_20_reverb_reduction"),
    ("phase_61_groove_echo_cancellation", "phase_49_advanced_dereverb"),
    # EBU R128: LUFS-Normalisierung vor TruePeak-Limiter (§6.1)
    ("phase_40_loudness_normalization", "phase_47_truepeak_limiter"),
    # Spec §2.46 Carrier-Chain-Inversion: subtraktiv vor additiv
    ("phase_03_denoise", "phase_07_harmonic_restoration"),
    ("phase_03_denoise", "phase_06_frequency_restoration"),
    ("phase_03_denoise", "phase_21_harmonic_exciter"),
    ("phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"),
    ("phase_29_tape_hiss_reduction", "phase_06_frequency_restoration"),
    ("phase_29_tape_hiss_reduction", "phase_21_harmonic_exciter"),
]

# ---------------------------------------------------------------------------
# Goal-Risiko-Schwelle + schützende Phasen (§GoalRisk, ExzellenzDenker-Integration)
# ---------------------------------------------------------------------------
# Prophylaktische Phasen-Injektion wenn ExzellenzDenker.prognostiziere() Risiko meldet.
_GOAL_RISK_THRESHOLD: float = 0.60
"""Risikoschwelle ∈ [0, 1] ab der eine schützende Phase injiziert wird."""

_GOAL_RISK_PROTECTIVE_PHASES: dict[str, str] = {
    "natuerlichkeit": "phase_03_denoise",  # Rauschen → Natürlichkeit
    "authentizitaet": "phase_03_denoise",  # Rauschen → Authentizität
    "brillanz": "phase_06_frequency_restoration",  # HF-Verlust → Brillanz
    "timbre": "phase_07_harmonic_restoration",  # HF-Verlust → Timbre
    "groove": "phase_09_crackle_removal",  # Transient-Armut → Groove
    "micro_dynamics": "phase_29_tape_hiss_reduction",  # Rausch-Maskierung → MikroDynamik
    "artikulation": "phase_24_dropout_repair",  # Dropout → Artikulation
}

# ---------------------------------------------------------------------------
# Goal-Risiko-Schwelle + schützende Phasen (§GoalRisk, ExzellenzDenker-Integration)
# ---------------------------------------------------------------------------
# Prophylaktische Phasen-Injektion wenn ExzellenzDenker.prognostiziere() Risiko meldet.
_GOAL_RISK_THRESHOLD: float = 0.60
"""Risikoschwelle ∈ [0, 1] ab der eine schützende Phase injiziert wird."""

_GOAL_RISK_PROTECTIVE_PHASES: dict[str, str] = {
    "natuerlichkeit": "phase_03_denoise",  # Rauschen → Natürlichkeit
    "authentizitaet": "phase_03_denoise",  # Rauschen → Authentizität
    "brillanz": "phase_06_frequency_restoration",  # HF-Verlust → Brillanz
    "timbre": "phase_07_harmonic_restoration",  # HF-Verlust → Timbre
    "groove": "phase_09_crackle_removal",  # Transient-Armut → Groove
    "micro_dynamics": "phase_29_tape_hiss_reduction",  # Rausch-Maskierung → MikroDynamik
    "artikulation": "phase_24_dropout_repair",  # Dropout → Artikulation
}


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class PhasePlan:
    """Semantisch aufgelöster, konfliktfreier Phase-Ausführungsplan.

    Wird von AurikDenker als precomputed_phase_plan an UV3.restore() übergeben.
    UV3 verwendet dann diesen Plan direkt — keine eigene Orchestrierung mehr.
    """

    phases: list[str]
    """Geordnete, deduplizierte, konfliktfreie Phasenliste für UV3."""

    suppressed: dict[str, str] = field(default_factory=dict)
    """phase_id → Grund der Supprimierung (§2.48 Konflikt-Guard)."""

    ordering_applied: list[tuple[str, str]] = field(default_factory=list)
    """Angewandte Reihenfolge-Constraints (before, after)."""

    conflict_notes: list[str] = field(default_factory=list)
    """Menschenlesbare Konflikt-Protokoll-Einträge."""

    semantic_annotations: dict[str, list[str]] = field(default_factory=dict)
    """phase_id → Semantik-Tags (für Diagnose/Logging)."""

    material: str = ""
    mode: str = ""

    phase_quality_tiers: dict[str, str] = field(default_factory=dict)
    """Qualitäts-Tier pro Phase: 'maximum' | 'quality' | 'fast' (aus StrategieDenker).
    Advisory — wird von UV3 für zukünftige per-Phase-Algorithmus-Selektion genutzt."""

    @property
    def is_valid(self) -> bool:
        """True wenn der Plan mindestens eine Phase enthält."""
        return bool(self.phases)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PhaseInteractionDenker:
    """Steuernde Intelligenz für Phasen-Orchestrierung.

    Workflow:
      1. _select_via_uv3(): UV3 selektiert Phasen (Werkzeug, nicht Orchestrator)
      2. _annotate(): semantische Typ-Annotation
      3. _resolve_conflicts(): Typ-Paar-basierte Supprimierung (§2.48)
      4. _apply_order_constraints(): deklarative Reihenfolge-Constraints (§2.46)
      5. PhasePlan zurückgeben → AurikDenker → UV3.restore(precomputed_phase_plan=...)
    """

    def plan(
        self,
        defect_result: Any,
        material: str = "unknown",
        mode: str = "quality",
        *,
        chain_info: Any | None = None,
        chain_result: Any | None = None,
        defekt_hint: Any | None = None,
        audio: np.ndarray | None = None,
        sr: int = 48000,
        causal_plan: Any | None = None,
        restorability_score: float = 70.0,
        pipeline_confidence: Any | None = None,
        goal_risk_map: dict[str, float] | None = None,
        strategie_plan: Any | None = None,
    ) -> PhasePlan:
        """Erstellt einen konfliktfreien, semantisch geordneten Phasenplan.

        Bei Fehlern wird ein leerer PhasePlan zurückgegeben — UV3 übernimmt
        dann die Selektion autonom (fail-safe).

        Args:
            defect_result:        DefectAnalysisResult vom DefektDenker.
            material:             Erkanntes Trägermedium (z. B. "vinyl").
            mode:                 Qualitätsmodus ("restoration", "studio2026").
            chain_info:           Tonträgerketten-Dict (optional).
            defekt_hint:          Heuristik-Phasenliste vom DefektDenker (optional).
            audio:                Eingabe-Audio für interne UV3-Analysen (optional).
            sr:                   Samplerate in Hz.
            causal_plan:          CausalDefectReasoner-Ergebnis (optional).
            restorability_score:  Restorability 0–100 für UV3-Optimizer.
            pipeline_confidence:  UQ-Konfidenz für UV3-Optimizer (optional).

        Returns:
            PhasePlan mit geordneter, konfliktfreier Phasenliste.
            Bei Fehler: leerer PhasePlan (UV3 übernimmt Selektion).
        """
        if defect_result is None:
            logger.debug("PhaseInteractionDenker: kein defect_result — UV3 übernimmt Selektion.")
            return PhasePlan(
                phases=[], material=material, mode=mode, conflict_notes=["Kein defect_result — UV3-Fallback."]
            )
        try:
            return self._plan_internal(
                defect_result=defect_result,
                material=material,
                mode=mode,
                chain_info=chain_info,
                chain_result=chain_result,
                defekt_hint=defekt_hint,
                audio=audio,
                sr=sr,
                causal_plan=causal_plan,
                restorability_score=restorability_score,
                pipeline_confidence=pipeline_confidence,
                goal_risk_map=goal_risk_map,
                strategie_plan=strategie_plan,
            )
        except Exception as exc:
            logger.warning(
                "PhaseInteractionDenker.plan() fehlgeschlagen: %s — UV3 übernimmt Selektion.",
                exc,
            )
            return PhasePlan(
                phases=[],
                material=material,
                mode=mode,
                conflict_notes=[f"PID fehlgeschlagen (fail-safe): {exc}"],
            )

    # ── Interne Pipeline ─────────────────────────────────────────────────────

    def _plan_internal(
        self,
        defect_result: Any,
        material: str,
        mode: str,
        chain_info: Any | None,
        chain_result: Any | None,
        defekt_hint: Any | None,
        audio: np.ndarray | None,
        sr: int,
        causal_plan: Any | None,
        restorability_score: float,
        pipeline_confidence: Any | None,
        goal_risk_map: dict[str, float] | None,
        strategie_plan: Any | None,
    ) -> PhasePlan:
        # 1. Phase-Selektion via UV3 (UV3 = Werkzeug, nicht Orchestrator)
        uv3_phases = self._select_via_uv3(
            defect_result=defect_result,
            mode=mode,
            chain_info=chain_info,
            defekt_hint=defekt_hint,
            audio=audio,
            sr=sr,
            causal_plan=causal_plan,
            restorability_score=restorability_score,
            pipeline_confidence=pipeline_confidence,
        )

        if not uv3_phases:
            return PhasePlan(
                phases=[],
                material=material,
                mode=mode,
                conflict_notes=["UV3-Selektion lieferte keine Phasen — UV3-Fallback."],
            )

        merged_phases = list(uv3_phases)
        injected_notes: list[str] = []

        # 2. Ketten-Pflicht-Phasen (§2.46 Feature 1: TontraegerketteDenker)
        # Injiziert must_have_phases aus der erkannten Trägerkette,
        # unabhängig vom DefectScanner-Score (§6.2a Komplement).
        if chain_result is not None:
            try:
                from denker.tontraegerkette_denker import get_tontraegerkette_denker

                _chain_plan = get_tontraegerkette_denker().leite_phasen_ab(chain_result)
                for must in _chain_plan.must_have_phases:
                    if must not in merged_phases:
                        merged_phases.append(must)
                        note = f"§6.2a Ketten-Injektion [{_chain_plan.chain_string}]: {must}"
                        injected_notes.append(note)
                        logger.info("PhaseInteractionDenker %s", note)
            except Exception as _ce:
                logger.debug("PhaseInteractionDenker: leite_phasen_ab() fehlgeschlagen: %s", _ce)

        # 3. Goal-Risk-Injektion (§GoalRisk Feature 2: ExzellenzDenker.prognostiziere())
        # Injiziert schützende Phasen wenn ein Musical Goal mit Risiko >= Schwelle bedroht ist.
        if goal_risk_map:
            for goal, risk in goal_risk_map.items():
                if risk >= _GOAL_RISK_THRESHOLD:
                    protective = _GOAL_RISK_PROTECTIVE_PHASES.get(goal)
                    if protective and protective not in merged_phases:
                        merged_phases.append(protective)
                        note = f"§GoalRisk-Injektion [{goal}={risk:.2f}]: {protective}"
                        injected_notes.append(note)
                        logger.info("PhaseInteractionDenker %s", note)

        # 4. Semantische Annotation
        annotations = self._annotate(merged_phases)

        # 5. Semantische Konflikterkennung + Auflösung (§2.48)
        resolved, suppressed, conflict_notes = self._resolve_conflicts(merged_phases, annotations)

        # 6. Reihenfolge-Constraints erzwingen (§2.46 / §7.2)
        ordered, ordering_applied = self._apply_order_constraints(resolved)

        # 7. Qualitäts-Tier je Phase (Feature 3: StrategieDenker.schaetze_phasen_tier())
        # Advisory-only — UV3 kann diese Infos für per-Phase-Algorithmus-Selektion nutzen.
        phase_quality_tiers: dict[str, str] = {}
        if strategie_plan is not None:
            try:
                from denker.strategie_denker import get_strategie_denker

                phase_quality_tiers = get_strategie_denker().schaetze_phasen_tier(
                    strategie_plan,
                    ordered,
                    restorability_score=restorability_score,
                )
                if phase_quality_tiers:
                    logger.debug(
                        "PhaseInteractionDenker: %d Tier-Empfehlungen (%d maximum, %d quality, %d fast)",
                        len(phase_quality_tiers),
                        sum(1 for t in phase_quality_tiers.values() if t == "maximum"),
                        sum(1 for t in phase_quality_tiers.values() if t == "quality"),
                        sum(1 for t in phase_quality_tiers.values() if t == "fast"),
                    )
            except Exception as _te:
                logger.debug("PhaseInteractionDenker: schaetze_phasen_tier() fehlgeschlagen: %s", _te)

        logger.info(
            "PhaseInteractionDenker: %d→%d Phasen | "
            "+%d injiziert | %d supprimiert | "
            "%d Ordnungsänderungen | material=%s mode=%s",
            len(uv3_phases),
            len(ordered),
            len(injected_notes),
            len(suppressed),
            len(ordering_applied),
            material,
            mode,
        )
        for note in conflict_notes:
            logger.info("§2.48 Konflikt-Guard: %s", note)

        return PhasePlan(
            phases=ordered,
            suppressed=suppressed,
            ordering_applied=ordering_applied,
            conflict_notes=conflict_notes + injected_notes,
            semantic_annotations={p: sorted(annotations.get(p, frozenset())) for p in ordered},
            phase_quality_tiers=phase_quality_tiers,
            material=material,
            mode=mode,
        )

    def _select_via_uv3(
        self,
        defect_result: Any,
        mode: str,
        chain_info: Any | None,
        defekt_hint: Any | None,
        audio: np.ndarray | None,
        sr: int,
        causal_plan: Any | None,
        restorability_score: float,
        pipeline_confidence: Any | None,
    ) -> list[str]:
        """Ruft UV3._select_phases() + _optimize_phase_plan_intelligence() ab.

        UV3 ist hier ein *Werkzeug* zur Defekt→Phasen-Übersetzung — kein Orchestrator.
        Der PhaseInteractionDenker verfeinert das Ergebnis anschließend semantisch.
        """
        try:
            from backend.core.unified_restorer_v3 import (
                QualityMode,
                RestorationConfig,
                UnifiedRestorerV3,
            )

            # Leicht-Instanz nur für Selektion (kein ML, kein Lazy-Load)
            _is_studio = mode in ("studio2026", "studio_2026", "maximum")
            _qmode = QualityMode.MAXIMUM if _is_studio else QualityMode.QUALITY
            _cfg = RestorationConfig(
                mode=_qmode,
                studio_2026=_is_studio,
                enforce_3x_rt=False,
                enable_performance_guard=False,
                enable_adaptive_skipping=False,
                enable_phase_gate=False,
            )
            _uv3 = UnifiedRestorerV3(config=_cfg)

            raw = _uv3._select_phases(
                defect_result,
                causal_plan=causal_plan,
                chain_info=chain_info,
                defekt_hint=defekt_hint,
                audio=audio,
                sr=sr,
            )
            optimized = _uv3._optimize_phase_plan_intelligence(
                raw,
                causal_plan=causal_plan,
                pipeline_confidence=pipeline_confidence,
                restorability_score=restorability_score,
            )
            logger.debug(
                "PhaseInteractionDenker: UV3-Selektion → %d Phasen (vor Semantic-Refine)",
                len(optimized),
            )
            return optimized

        except Exception as exc:
            logger.warning(
                "PhaseInteractionDenker: UV3-Phasenselektion fehlgeschlagen: %s",
                exc,
            )
            return []

    def _annotate(self, phases: list[str]) -> dict[str, frozenset[str]]:
        """Weist jeder Phase ihre semantischen Tags zu.

        Präfix-Matching: "phase_03_denoise_ml" erbt Tags von "phase_03_denoise".
        Unbekannte Phasen erhalten leere Tag-Menge.
        """
        result: dict[str, frozenset[str]] = {}
        for phase in phases:
            tags = _PHASE_SEMANTICS.get(phase)
            if tags is None:
                # Präfixsuche
                for known, known_tags in _PHASE_SEMANTICS.items():
                    if phase.startswith(known):
                        tags = known_tags
                        break
            result[phase] = tags if tags is not None else frozenset()
        return result

    def _resolve_conflicts(
        self,
        phases: list[str],
        annotations: dict[str, frozenset[str]],
    ) -> tuple[list[str], dict[str, str], list[str]]:
        """Erkennt und löst semantische Phasen-Konflikte (§2.48).

        Für jede Konflikt-Regel: wenn Phase A Auslöser-Tags hat, wird die
        erste nachfolgende Phase mit Ziel-Tags supprimiert.

        Invariante: Supprimierung ist deterministisch (First-Wins-Prinzip).
        """
        suppressed: dict[str, str] = {}
        conflict_notes: list[str] = []

        for trigger_tags, target_tags in _CONFLICT_RULES:
            trigger_phase: str | None = None
            for phase in phases:
                if phase in suppressed:
                    continue
                tags = annotations.get(phase, frozenset())
                if trigger_phase is None:
                    if tags & trigger_tags == trigger_tags:
                        trigger_phase = phase
                        continue
                # Auslöser wurde bereits gefunden — suche Konflikt-Kandidaten
                if trigger_phase is not None and tags & target_tags == target_tags:
                    reason = (
                        f"§2.48 [{', '.join(sorted(trigger_tags))}] "
                        f"→ [{', '.join(sorted(target_tags))}]: "
                        f"{trigger_phase} vor {phase} — {phase} supprimiert"
                    )
                    suppressed[phase] = reason
                    conflict_notes.append(reason)
                    logger.info("PhaseInteractionDenker: %s", reason)
                    # Only first target per trigger suppressed; reset for next occurrence
                    trigger_phase = None

        resolved = [p for p in phases if p not in suppressed]
        return resolved, suppressed, conflict_notes

    def _apply_order_constraints(
        self,
        phases: list[str],
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Erzwingt deklarative Reihenfolge-Constraints (§2.46 / §7.2).

        Für jedes Tupel (A, B): wenn A nach B steht, wird A vor B verschoben.
        Konvergiert in O(n × constraints) — keine Endlosschleifen möglich.
        """
        result = list(phases)
        applied: list[tuple[str, str]] = []

        for before, after in _ORDER_CONSTRAINTS:
            if before not in result or after not in result:
                continue
            idx_before = result.index(before)
            idx_after = result.index(after)
            if idx_before > idx_after:
                result.remove(before)
                idx_after_new = result.index(after)
                result.insert(idx_after_new, before)
                applied.append((before, after))
                logger.debug(
                    "PhaseInteractionDenker: §ORDER %s → vor → %s",
                    before,
                    after,
                )

        return result, applied
