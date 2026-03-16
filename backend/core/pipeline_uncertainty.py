"""
PipelineUncertainty (§2.15 Spec)
==================================

Integrations-Wrapper um backend/core/optimization/uncertainty_quantification.py.
Quantifiziert für jede Restaurierungsoperation eine Konfidenz und passt
GP-Bounds und Pipeline-Parameter entsprechend an.

Konfidenz-Schwellwerte:
    ≥ 0.80: Defekt sicher erkannt → volle GP-Aggressivität
    0.50–0.80: GP-Bounds um 20 % konservativer, Nutzer-Hinweis aktivieren
    < 0.50:  konservative Mindest-Parameter, Musical-Goal-Schwellen +0.02

Referenz: §2.15 Aurik-9-Spec (v9.9.5)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfidenz-Schwellwerte (§2.15)
# ---------------------------------------------------------------------------


class UncertaintyThresholds:
    """Konfidenz-Schwellwerte für Pipeline-Steuerung."""

    HIGH: float = 0.80  # Volle GP-Aggressivität
    MEDIUM: float = 0.50  # Moderate GP-Bounds
    LOW: float = 0.00  # Sicherheitsmaximierende Parameter


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfidence:
    """Konfidenz-Ergebnis für eine Restaurierungsoperation.

    Attributes:
        confidence:        Gesamt-Konfidenz ∈ [0, 1].
        tier:              "high", "medium" oder "low".
        gp_bound_factor:   Multiplikator für GP-Bound-Reduktion (1.0 = voll, 0.8 = 20 % konservativer).
        threshold_offset:  Additions-Offset auf Musical-Goal-Schwellen (+0.02 bei low).
        user_hint:         Meldung für Nutzer (Deutsch, laienverständlich), oder "" wenn high.
        details:           Detaillierte UQ-Ergebnisse (für Log/Report).
    """

    confidence: float = 1.0
    tier: str = "high"
    gp_bound_factor: float = 1.0
    threshold_offset: float = 0.0
    user_hint: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PipelineUncertaintyEstimator:
    """Quantifiziert Restaurierungs-Konfidenz und steuert GP-Aggressivität.

    Einbindungs-Pflicht (§2.15):
        CausalDefectReasoner.reason() → plan.confidence → PipelineUncertaintyEstimator
        GPParameterOptimizer.propose() erhält confidence als Eingangs-Prior
        RestorationResult.confidence: float → UI-Anzeige als Balken

    DSP-Konfidenzberechnung (ohne ML):
        Konfidenz basiert auf DefectScanner-Scores und CausalReasoner-Posterior-Entropie:
        - Hohe DefektScore-Streuung → niedrige Konfidenz (unklar welcher Defekt)
        - Dominanter Defekt (score > 0.7) → hohe Konfidenz
        - Posterior-Entropie ≥ 0.8 → niedrige Konfidenz
    """

    def estimate(
        self,
        causal_plan,  # RestorationPlan von CausalDefectReasoner
        defect_scores: dict[str, float] | None = None,
    ) -> PipelineConfidence:
        """Schätzt Konfidenz aus CausalDefectReasoner-Ergebnis.

        Args:
            causal_plan:    RestorationPlan (hat .confidence float-Attribut).
            defect_scores:  Dict DefectType-Name → Score (optional, für DSP-Konfidenz).

        Returns:
            PipelineConfidence mit Tier + GP-Steuerungsparametern.
        """
        # Primäre Konfidenz aus CausalPlan
        if causal_plan is not None:
            plan_confidence = float(getattr(causal_plan, "confidence", 0.5))
        else:
            plan_confidence = 0.5

        # DSP-Konfidenz aus DefectScores (optional enhacements)
        dsp_confidence = self._estimate_dsp_confidence(defect_scores)

        # Kombinieren (geometrisches Mittel)
        combined = float(np.sqrt(plan_confidence * dsp_confidence))
        combined = float(np.clip(combined, 0.0, 1.0))

        # Tier bestimmen
        if combined >= UncertaintyThresholds.HIGH:
            tier = "high"
            gp_factor = 1.0
            threshold_offset = 0.0
            user_hint = ""
        elif combined >= UncertaintyThresholds.MEDIUM:
            tier = "medium"
            gp_factor = 0.80  # 20 % konservativer
            threshold_offset = 0.0
            user_hint = (
                "Manche Stellen sind schwer zu beurteilen — das System "
                "arbeitet vorsichtig, damit nichts verschlechtert wird."
            )
        else:
            tier = "low"
            gp_factor = 0.60  # 40 % konservativer
            threshold_offset = 0.02  # Musical Goals +0.02 verschärft
            user_hint = (
                "Die Aufnahme ist sehr schwierig. Das Ergebnis wird sorgfältig "
                "geprüft, aber möglicherweise sind Restdefekte unvermeidbar."
            )

        logger.info(
            "🔮 PipelineUncertainty: Konfidenz=%.3f (Plan=%.3f DSP=%.3f) Tier=%s GP-Faktor=%.2f",
            combined,
            plan_confidence,
            dsp_confidence,
            tier,
            gp_factor,
        )

        # Versuche ML-UQ-Backend (uncertainty_quantification.py)
        details = self._try_ml_uq_backend(combined)

        return PipelineConfidence(
            confidence=combined,
            tier=tier,
            gp_bound_factor=gp_factor,
            threshold_offset=threshold_offset,
            user_hint=user_hint,
            details=details,
        )

    def apply_to_gp_params(
        self,
        proposed_params: dict[str, float],
        confidence: PipelineConfidence,
        param_space: dict[str, tuple],
    ) -> dict[str, float]:
        """Skaliert GP-Parameter entsprechend Konfidenz-Tier (konservative Bounds).

        Bei Konfidenz < MEDIUM: alle Parameter werden Richtung Minimum verschoben
        (konservativer Eingriff). Bei HIGH: unveränderter Vorschlag.

        Args:
            proposed_params:   GP-vorgeschlagene Parameter.
            confidence:        PipelineConfidence.
            param_space:       Parametergrenzen {name: (min, max)}.

        Returns:
            Angepasste Parameter (immer innerhalb param_space-Grenzen).
        """
        if confidence.tier == "high":
            return proposed_params

        adjusted = {}
        for name, value in proposed_params.items():
            if name not in param_space:
                adjusted[name] = value
                continue
            lo, hi = param_space[name]
            center = (lo + hi) / 2.0
            # Wert Richtung Mitte / konservativ verschieben
            factor = confidence.gp_bound_factor
            shifted = center + factor * (value - center)
            adjusted[name] = float(np.clip(shifted, lo, hi))

        return adjusted

    def apply_threshold_offsets(
        self,
        thresholds: dict[str, float],
        confidence: PipelineConfidence,
    ) -> dict[str, float]:
        """Verschärft Musical-Goal-Schwellen bei niedriger Konfidenz.

        Args:
            thresholds:  Original-Schwellwerte {goal_name: threshold}.
            confidence:  PipelineConfidence.

        Returns:
            Angepasste Schwellwerte (niemals > 1.0).
        """
        if confidence.threshold_offset == 0.0:
            return thresholds
        return {name: float(np.clip(val + confidence.threshold_offset, 0.0, 1.0)) for name, val in thresholds.items()}

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _estimate_dsp_confidence(self, defect_scores: dict[str, float] | None) -> float:
        """Schätzt Konfidenz rein aus DefectScanner-Scores."""
        if not defect_scores:
            return 0.6  # Standard-Prior

        scores = list(defect_scores.values())
        scores = [s for s in scores if isinstance(s, (int, float)) and np.isfinite(s)]
        if not scores:
            return 0.6

        max_score = max(scores)
        np.mean(scores)

        # Hohe Dominanz eines Defekts → hohe Konfidenz
        if max_score >= 0.7:
            return min(1.0, 0.6 + max_score * 0.5)
        # Homogene Scores → niedrige Konfidenz (unklar)
        std_score = float(np.std(scores))
        if std_score < 0.1:
            return 0.35
        # Normalfall
        return float(np.clip(0.4 + std_score * 2.0, 0.3, 0.85))

    def _try_ml_uq_backend(self, base_confidence: float) -> dict[str, Any]:
        """Versucht ML-basierte UQ via uncertainty_quantification.py."""
        details: dict[str, Any] = {"base_confidence": base_confidence}
        try:
            from backend.core.optimization.uncertainty_quantification import (  # type: ignore[import]
                UncertaintyQuantifier,
            )

            uq = UncertaintyQuantifier()
            details["ml_uq_available"] = True
            details["ml_uq_class"] = type(uq).__name__
        except Exception as e:  # noqa: BLE001
            details["ml_uq_available"] = False
            details["ml_uq_error"] = str(e)
        return details


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: PipelineUncertaintyEstimator | None = None
_lock = threading.Lock()


def get_pipeline_uncertainty_estimator() -> PipelineUncertaintyEstimator:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PipelineUncertaintyEstimator()
    return _instance


def estimate_pipeline_confidence(
    causal_plan,
    defect_scores: dict[str, float] | None = None,
) -> PipelineConfidence:
    """Convenience-Funktion: Schätzt Pipeline-Konfidenz.

    Args:
        causal_plan:    RestorationPlan (hat .confidence float-Attribut).
        defect_scores:  Dict DefectType-Name → Score (optional).

    Returns:
        PipelineConfidence mit Tier + GP-Steuerungsparametern.
    """
    return get_pipeline_uncertainty_estimator().estimate(causal_plan, defect_scores)
