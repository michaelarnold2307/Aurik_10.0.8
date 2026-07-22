"""
denker/perceptual_council.py — PerceptualQualityCouncil
========================================================

Zentrale perzeptive Qualitätbewertungsinstanz für Aurik 10.0.0.

Bisher war die finale Qualitätsschätzung auf 5+ unabhängige Module verteilt:
  - quality_estimate (Spec §8.1) in aurik_denker.py (Formel)
  - HPE ± Delta im restaurier_denker.py (versteckt, nie propagiert)
  - Inviting Sound im restaurier_denker.py (versteckt)
  - Musical Goals + VERSA MOS im exzellenz_denker.py
  - SweetSpot-Optimizer im restaurier_denker.py (versteckt)

Dieser Council vereint alle perzeptiven Metriken in EINER gewichteten
Bewertung und gibt eine klare Handlungsempfehlung.

Spec: §8.1 (erweitert), §v10.3 — v10.0.0
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class PerceptualQualityVerdict:
    """Gewichtete, multimodale Qualitätsbewertung eines restaurierten Signals.

    Ersetzt schrittweise die bisherige quality_estimate-Formel (§8.1) und
    bündelt alle perzeptiven Metriken an einer Stelle.
    """

    # ── Gewichteter Gesamtscore (∈ [0, 1]) ──────────────────────────────
    holistic_score: float
    """Gewichteter Gesamt-Qualitätsscore aus allen Teilmetriken."""

    # ── Einzelmetriken (alle ∈ [0, 1] bzw. MOS-1/4 normiert) ────────────
    defect_severity: float = 0.0
    """Defektschwere ∈ [0, 1]; 0 = sauber, 1 = vollständig defekt."""

    versa_mos_norm: float = 0.0
    """VERSA MOS (nicht-referenzielle Qualität), normiert auf [0, 1] via (MOS-1)/4."""

    excellence_score: float = 0.0
    """ExzellenzScore ∈ [0, 1] aus 15 Musical Goals."""

    goals_passed: int = 0
    """Anzahl bestandener Musical Goals."""

    goals_total: int = 0
    """Anzahl geprüfter Musical Goals."""

    # ── Herkunft des Scores ──────────────────────────────────────────────
    scoring_method: str = "perceptual_council_v1"
    """Welcher Algorithmus den Score berechnet hat (für Debugging/Telemetrie)."""

    # ── Handlungsempfehlung ──────────────────────────────────────────────
    recommendation: str = "accept"
    """Empfehlung: 'accept' | 'review' | 'retry_quality' | 'retry_studio2026' | 'rollback'.

    - accept:              Qualität gut genug für das Material.
    - review:              Grenzwertig – menschliche Prüfung empfohlen.
    - retry_quality:       Qualität unzureichend – erneuter Versuch in QUALITY.
    - retry_studio2026:    Schlechte Goals – erneuter Versuch in STUDIO 2026.
    - rollback:            Verschlechterung erkannt – unverändertes Original bevorzugen.
    """

    recommendation_reason: str = ""
    """Kurze, deutschsprachige Begründung der Empfehlung."""

    # ── Metadaten für Debugging ──────────────────────────────────────────
    action_log: list[str] = field(default_factory=list)
    """Einzelfaktoren und ihre Gewichtung (Deutsch, für Logging/UI)."""

    improvement_hints: dict[str, float] = field(default_factory=dict)
    """Goal → Lücke: welche Musical Goals verbesserungswürdig sind (Score < Threshold)."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Telemetrie und Logging."""
        return {
            "holistic_score": float(self.holistic_score),
            "defect_severity": float(self.defect_severity),
            "versa_mos_norm": float(self.versa_mos_norm),
            "excellence_score": float(self.excellence_score),
            "goals_passed": self.goals_passed,
            "goals_total": self.goals_total,
            "scoring_method": self.scoring_method,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "action_log": list(self.action_log),
            "improvement_hints": {k: float(v) for k, v in self.improvement_hints.items()},
        }


# ---------------------------------------------------------------------------
# Material-adaptive Schwellen und Gewichte
# ---------------------------------------------------------------------------

_MATERIAL_MOS_TARGETS: dict[str, float] = {
    "wax_cylinder": 3.5,
    "shellac": 3.8,
    "lacquer_disc": 3.7,
    "wire_recording": 3.6,
    "vinyl": 4.0,
    "tape": 4.2,
    "reel_tape": 4.3,
    "cassette": 4.0,
    "cassette_dolby_b": 4.2,
    "cassette_dolby_c": 4.2,
    "cassette_dolby_s": 4.3,
    "dat": 4.5,
    "cd_digital": 4.5,
    "mp3_low": 3.9,
    "mp3_high": 4.5,
    "aac": 4.5,
    "minidisc": 4.0,
    "streaming": 4.0,
    "unknown": 3.8,
}

_GOAL_FALLBACK_THRESHOLD: float = 0.75
"""Default-Grenzwert für ein bestandenes Musical Goal."""


# ---------------------------------------------------------------------------
# Quality Council
# ---------------------------------------------------------------------------


class PerceptualQualityCouncil:
    """Zentrale perzeptive Qualitätsbewertung.

    Singleton-Pattern (thread-safe), analog zu den Domänen-Denkern (§3.2).
    """

    def assess(
        self,
        *,
        defect_severity: float = 0.0,
        versa_mos: float = 0.0,
        excellence_score: float = 0.0,
        musical_goals: dict[str, float] | None = None,
        goals_passed: int = 0,
        goals_total: int = 0,
        material: str = "unknown",
        chain_info: dict[str, Any] | None = None,
        # Optional: perzeptiver Kontext (zukünftig aus RestaurierDenker)
        hpe_score: float | None = None,
        hpe_delta: float | None = None,
        inviting_score: float | None = None,
        sweet_spot_score: float | None = None,
    ) -> PerceptualQualityVerdict:
        """Bewertet die perzeptive Qualität des restaurierten Signals.

        Args:
            defect_severity:  Defektschwere ∈ [0, 1].
            versa_mos:        VERSA MOS ∈ [1, 5] (0 = nicht gemessen).
            excellence_score: ExzellenzScore ∈ [0, 1].
            musical_goals:    15 Musical-Goals-Scores.
            goals_passed:     Bestandene Goals.
            goals_total:      Geprüfte Goals gesamt.
            material:         Trägermedium (z. B. 'vinyl', 'tape').
            chain_info:       Tonträgerketten-Dict (optional).
            hpe_score:        Human Pleasantness (optional, zukünftig).
            hpe_delta:        HPE-Veränderung durch Pipeline (optional).
            inviting_score:   Inviting-Sound-Score (optional).
            sweet_spot_score: SweetSpot-Gesamtscore (optional).

        Returns:
            PerceptualQualityVerdict mit gewichtetem Score und Empfehlung.
        """
        _log: list[str] = []
        _goals = dict(musical_goals or {})

        # ── 1. Defekt-Komponente ────────────────────────────────────────
        _defect_component = 1.0 - float(np.clip(defect_severity, 0.0, 1.0))
        _log.append(f"Defekt-Komponente: {_defect_component:.3f} (Schwere={defect_severity:.2f})")

        # ── 2. VERSA MOS-Komponente (nicht-referenziell) ────────────────
        _versa_norm: float = 0.55  # neutraler Default (≈ MOS 3.2)
        if versa_mos > 0.0:
            _versa_norm = float(np.clip((versa_mos - 1.0) / 4.0, 0.0, 1.0))
            _log.append(f"VERSA MOS: {versa_mos:.2f} → normiert {_versa_norm:.3f}")

        # ── 3. Musical-Goals-Komponente (Exzellenz) ─────────────────────
        _goals_mean: float = excellence_score
        if _goals:
            _finite = [v for v in _goals.values() if isinstance(v, (int, float)) and math.isfinite(v)]
            if _finite:
                _goals_mean = float(np.mean(_finite))
        _goals_ratio = goals_passed / max(goals_total, 1)
        _goals_component = 0.60 * _goals_mean + 0.40 * _goals_ratio
        _log.append(f"Goals-Komponente: {_goals_component:.3f} (Mittel={_goals_mean:.3f}, Ratio={_goals_ratio:.3f})")

        # ── 4. Material-adaptive Gewichtung ─────────────────────────────
        _material = str(material or "unknown").strip().lower()
        _is_historical = _material in {
            "wax_cylinder",
            "shellac",
            "lacquer_disc",
            "wire_recording",
            "vinyl",
            "tape",
            "reel_tape",
            "cassette",
            "cassette_dolby_b",
            "cassette_dolby_c",
            "cassette_dolby_s",
        }
        _is_digital = _material in {"cd_digital", "dat", "mp3_high", "aac", "streaming"}

        # Historische Materialien: Defekt-Komponente wichtiger (Vergebung)
        # Digitale Materialien: Goals-Komponente wichtiger (Präzision)
        if _is_historical:
            _w_defect = 0.35
            _w_versa = 0.25
            _w_goals = 0.40
            _log.append(f"Gewichtung (historisch): defect={_w_defect}, versa={_w_versa}, goals={_w_goals}")
        elif _is_digital:
            _w_defect = 0.20
            _w_versa = 0.30
            _w_goals = 0.50
            _log.append(f"Gewichtung (digital): defect={_w_defect}, versa={_w_versa}, goals={_w_goals}")
        else:
            _w_defect = 0.30
            _w_versa = 0.25
            _w_goals = 0.45
            _log.append(f"Gewichtung (standard): defect={_w_defect}, versa={_w_versa}, goals={_w_goals}")

        # ── 5. Holistischer Score ───────────────────────────────────────
        if goals_total == 0:
            # Keine Goals gemessen → 2-Komponenten-Formel analog §8.1
            _holistic = 0.40 * _defect_component + 0.60 * _versa_norm
            _log.append(f"Holistischer Score (2-Komp., keine Goals): {_holistic:.3f}")
        else:
            _holistic = _w_defect * _defect_component + _w_versa * _versa_norm + _w_goals * _goals_component
            _log.append(f"Holistischer Score (3-Komp.): {_holistic:.3f}")
        _holistic = float(np.clip(_holistic, 0.0, 1.0))
        _log.append(f"Holistischer Score: {_holistic:.3f}")

        # ── 6. Verbesserungshinweise (Goals mit Lücken) ─────────────────
        _improvements: dict[str, float] = {}
        _material_target = _MATERIAL_MOS_TARGETS.get(_material, 3.8)
        # Nur wenn keine VERSA-Verbesserung
        for _goal, _score in _goals.items():
            _threshold = _GOAL_FALLBACK_THRESHOLD
            if isinstance(_score, (int, float)) and math.isfinite(_score) and _score < _threshold:
                _improvements[_goal] = float(_threshold - _score)

        # ── 7. Empfehlung ───────────────────────────────────────────────
        _recommendation, _reason = self._recommend(
            holistic_score=_holistic,
            versa_mos=versa_mos,
            material_target=_material_target,
            goals_passed=goals_passed,
            goals_total=goals_total,
            has_improvements=bool(_improvements),
            improvement_count=len(_improvements),
            is_historical=_is_historical,
        )
        _log.append(f"Empfehlung: {_recommendation} — {_reason}")

        return PerceptualQualityVerdict(
            holistic_score=_holistic,
            defect_severity=defect_severity,
            versa_mos_norm=_versa_norm,
            excellence_score=excellence_score,
            goals_passed=goals_passed,
            goals_total=goals_total,
            scoring_method="perceptual_council_v1",
            recommendation=_recommendation,
            recommendation_reason=_reason,
            action_log=_log,
            improvement_hints=_improvements,
        )

    @staticmethod
    def _recommend(
        *,
        holistic_score: float,
        versa_mos: float,
        material_target: float,
        goals_passed: int,
        goals_total: int,
        has_improvements: bool,
        improvement_count: int = 0,
        is_historical: bool,
    ) -> tuple[str, str]:
        """Leitet eine Handlungsempfehlung aus dem holistischen Score und Kontext ab."""
        # Verschlechterung: holistic_score sehr niedrig trotz niedriger Defektschwere
        # → mögliches Overprocessing
        if holistic_score < 0.30:
            return (
                "rollback",
                "Qualität unter 0.30 — Rollback empfohlen. "
                "Die Restaurierung hat das Signal wahrscheinlich verschlechtert.",
            )

        # Exzellent
        if holistic_score >= 0.80:
            return (
                "accept",
                f"Hervorragende Qualität (Score={holistic_score:.2f}). Alle perzeptiven Metriken im Zielbereich.",
            )

        # Gut — aber mit Verbesserungspotential
        if holistic_score >= 0.65:
            if goals_passed < goals_total * 0.6:
                return (
                    "review",
                    f"Gute Gesamtqualität (Score={holistic_score:.2f}), "
                    f"aber nur {goals_passed}/{goals_total} Goals bestanden. "
                    "Eine manuelle Prüfung wird empfohlen.",
                )
            if has_improvements and not is_historical:
                return (
                    "retry_studio2026",
                    f"Qualität ausreichend (Score={holistic_score:.2f}), "
                    f"aber {improvement_count} Goals unter Schwelle. "
                    "Erneuter Versuch in STUDIO 2026 könnte Verbesserung bringen.",
                )
            return (
                "accept",
                f"Solide Qualität (Score={holistic_score:.2f}). Für historisches Material im erwarteten Bereich.",
            )

        # Grenzwertig
        if holistic_score >= 0.45:
            if versa_mos > 0.0 and versa_mos < material_target:
                return (
                    "retry_quality",
                    f"Qualität grenzwertig (Score={holistic_score:.2f}, "
                    f"VERSA MOS={versa_mos:.1f} < Ziel={material_target:.1f}). "
                    "Erneuter Versuch in QUALITY-Modus empfohlen.",
                )
            return (
                "review",
                f"Qualität grenzwertig (Score={holistic_score:.2f}). "
                "Menschliche Prüfung vor endgültiger Freigabe empfohlen.",
            )

        # Unzureichend
        return (
            "retry_quality",
            f"Qualität unzureichend (Score={holistic_score:.2f}). Erneuter Versuch mit QUALITY-Modus empfohlen.",
        )


# ---------------------------------------------------------------------------
# Singleton-Pattern (§3.2)
# ---------------------------------------------------------------------------

_instance: PerceptualQualityCouncil | None = None
_lock = threading.Lock()


def get_perceptual_council() -> PerceptualQualityCouncil:
    """Thread-sicherer Singleton (§3.2 Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerceptualQualityCouncil()
    return _instance


def assess_quality(
    *,
    defect_severity: float = 0.0,
    versa_mos: float = 0.0,
    excellence_score: float = 0.0,
    musical_goals: dict[str, float] | None = None,
    goals_passed: int = 0,
    goals_total: int = 0,
    material: str = "unknown",
) -> PerceptualQualityVerdict:
    """Convenience-Funktion: Einzeilige Qualitätsbewertung über Singleton.

    Beispiel:
        verdict = assess_quality(
            defect_severity=0.3, versa_mos=4.2,
            excellence_score=0.85, goals_passed=12, goals_total=15,
            material='vinyl',
        )
        print(verdict.recommendation)  # 'accept'
    """
    return get_perceptual_council().assess(
        defect_severity=defect_severity,
        versa_mos=versa_mos,
        excellence_score=excellence_score,
        musical_goals=musical_goals,
        goals_passed=goals_passed,
        goals_total=goals_total,
        material=material,
    )
