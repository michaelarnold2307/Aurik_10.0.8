"""
core/self_learning_optimizer.py
Self-Learning Optimizer (SLO)
================================

Lernt vollautomatisch aus jedem Verarbeitungsergebnis, welche
Processing-Variante für welches Material und welchen Defekt-Typ am besten
funktioniert.  Kein Nutzereingriff nötig.

Algorithmus: UCB1 Multi-Armed Bandit pro (Material × Variante)-Kombination.
  - Jede Kombination (Material, Variante) ist ein "Arm".
  - Bei jeder neuen Restaurierung wird der Arm mit dem höchsten UCB1-Score gewählt.
  - Nach Abschluss wird das Ergebnis (quality_delta) zurückgemeldet und die
    Statistik aktualisiert.
  - Persistenz: Zustand wird in `logs/self_learning_state.json` gespeichert
    und beim nächsten Start wiederhergestellt.

Öffentliche API (verwendet von AutonomousRestorationEngine):
  - record_result(material, variant, defect_profile, quality_delta)
  - recommend_variant(material, defect_profile) → variant_name | None

Legacy-API (rückwärtskompatibel, deprecated):
  - update_from_feedback(features, feedback)
  - predict(features)
  - optimize(features)

Author: Aurik Development Team
Version: 2.0.0 "UCB1 Autonomy Edition"
Date: 2026-02-17
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
import threading
import time
from typing import Any

from backend.core.processing_modes import ProcessingMode

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = "logs/self_learning_state.json"


# ---------------------------------------------------------------------------
# Arm-Statistik (UCB1)
# ---------------------------------------------------------------------------


@dataclass
class ArmStats:
    """Statistik für einen (Material, Variante)-Arm."""

    count: int = 0
    """Anzahl der bisherigen Pulls."""
    mean_delta: float = 0.0
    """Gleitender Mittelwert der quality_delta-Werte."""
    sum_sq: float = 0.0
    """Summe der Quadrate (für Varianz-Berechnung)."""
    last_updated: float = field(default_factory=time.time)

    def update(self, delta: float) -> None:
        """Inkrementelles Update mit exponentiell gerichtetem Glättungsfaktor."""
        self.count += 1
        alpha = 1.0 / self.count  # Klassisches inkrementelles Mittel
        self.mean_delta += alpha * (delta - self.mean_delta)
        self.sum_sq += delta**2
        self.last_updated = time.time()

    def ucb1_score(self, total_pulls: int, exploration_factor: float = 1.4) -> float:
        """
        UCB1-Score: mean + C * sqrt(ln(N) / n)
        Nicht-gezogene Arme haben Score = +∞ (werden zuerst ausprobiert).
        """
        if self.count == 0:
            return float("inf")
        if total_pulls <= 1:
            return self.mean_delta
        exploitation = self.mean_delta
        exploration = exploration_factor * math.sqrt(math.log(total_pulls) / self.count)
        return exploitation + exploration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArmStats:
        return cls(**d)


# ---------------------------------------------------------------------------
# Self-Learning Optimizer
# ---------------------------------------------------------------------------


class SelfLearningOptimizer:
    """
    UCB1-basierter Self-Learning Optimizer für Aurik.

    Lernt vollautomatisch, welche (Material, Variante)-Kombination die
    beste Qualitätsverbesserung erzielt.  Session-übergreifende Persistenz
    via JSON.
    """

    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        state_path: str = _DEFAULT_STATE_PATH,
        exploration_factor: float = 1.4,
    ):
        self.mode = mode
        self.state_path = state_path
        self.exploration_factor = exploration_factor
        self._lock = threading.RLock()

        # Kern-Datenstruktur: {(material_str, variant_str): ArmStats}
        self._arms: dict[tuple[str, str], ArmStats] = {}
        self._total_pulls: int = 0

        # Legacy-API Kompatibilität
        self.learning_rate: float = 0.05
        self.model_params: dict[str, float] = {}
        self._legacy_history: list[dict[str, Any]] = []

        # Zustand laden
        self._load_state()
        logger.info(
            "SelfLearningOptimizer geladen | Modus=%s | Arme=%d | Gesamt-Pulls=%d",
            mode.value,
            len(self._arms),
            self._total_pulls,
        )

    # ------------------------------------------------------------------
    # Öffentliche API (neu)
    # ------------------------------------------------------------------

    def record_result(
        self,
        material,  # MaterialType oder str
        variant: str,
        defect_profile: Any,
        quality_delta: float,
    ) -> None:
        """
        Registriert das Ergebnis eines Processing-Durchlaufs.

        Args:
            material:       Erkanntes Material (MaterialType oder str).
            variant:        Name der verwendeten Variante (z. B. 'balanced').
            defect_profile: DefectAnalysisResult (für zukünftige Feature-Extraktion).
            quality_delta:  Qualitätsverbesserung (positiv = besser, negativ = schlechter).
        """
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)

        with self._lock:
            if key not in self._arms:
                self._arms[key] = ArmStats()
            self._arms[key].update(quality_delta)
            self._total_pulls += 1

        logger.debug(
            "SLO Record | %s × %s | Δ=%.3f | Count=%d | Mean=%.3f",
            mat_str,
            variant,
            quality_delta,
            self._arms[key].count,
            self._arms[key].mean_delta,
        )
        self._save_state()

    def recommend_variant(
        self,
        material,  # MaterialType oder str
        defect_profile: Any,
    ) -> str | None:
        """
        Empfiehlt die Variante mit dem höchsten UCB1-Score für das gegebene Material.

        Returns:
            Varianten-Name oder None wenn noch keine Erfahrungswerte vorliegen.
        """
        mat_str = material.value if hasattr(material, "value") else str(material)

        with self._lock:
            # Alle Arme für dieses Material
            material_arms = {variant: stats for (m, variant), stats in self._arms.items() if m == mat_str}

            if not material_arms:
                logger.debug("SLO: Noch keine Daten für Material %s.", mat_str)
                return None

            # Besten Arm nach UCB1-Score
            best_variant = max(
                material_arms,
                key=lambda v: material_arms[v].ucb1_score(self._total_pulls, self.exploration_factor),
            )
            best_score = material_arms[best_variant].ucb1_score(self._total_pulls, self.exploration_factor)
            logger.debug(
                "SLO Empfehlung: %s × %s (UCB1=%.3f)",
                mat_str,
                best_variant,
                best_score,
            )
            return best_variant

    def get_statistics(self) -> dict[str, Any]:
        """Gibt vollständige Lernstatistiken zurück (für Audit/Debugging)."""
        with self._lock:
            return {
                "mode": self.mode.value,
                "total_pulls": self._total_pulls,
                "arms": {f"{m}×{v}": stats.to_dict() for (m, v), stats in self._arms.items()},
            }

    def total_pulls(self) -> int:
        """Öffentlicher Accessor: Gesamtzahl der bisherigen Processing-Durchläufe."""
        return self._total_pulls

    def arm_mean_delta(self, material: Any, variant: str) -> float | None:
        """
        Öffentlicher Accessor: Mittleres quality_delta für (material, variant).

        Returns:
            mean_delta oder None wenn keine Daten vorliegen.
        """
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)
        with self._lock:
            if key not in self._arms:
                return None
            return self._arms[key].mean_delta

    def has_data_for(self, material: Any, variant: str) -> bool:
        """True wenn mindestens ein Ergebnis für (material, variant) vorliegt."""
        mat_str = material.value if hasattr(material, "value") else str(material)
        key = (mat_str, variant)
        with self._lock:
            return key in self._arms and self._arms[key].count > 0

    # ------------------------------------------------------------------
    # Persistenz
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Speichert den aktuellen Zustand in JSON."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        try:
            with self._lock:
                state = {
                    "mode": self.mode.value,
                    "total_pulls": self._total_pulls,
                    "arms": {f"{m}|{v}": stats.to_dict() for (m, v), stats in self._arms.items()},
                }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except (OSError, TypeError) as exc:
            logger.warning("SLO: Zustand konnte nicht gespeichert werden: %s", exc)

    def _load_state(self) -> None:
        """Lädt gespeicherten Zustand aus JSON (falls vorhanden)."""
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            with self._lock:
                self._total_pulls = int(state.get("total_pulls", 0))
                for key_str, arm_dict in state.get("arms", {}).items():
                    parts = key_str.split("|", 1)
                    if len(parts) == 2:
                        self._arms[(parts[0], parts[1])] = ArmStats.from_dict(arm_dict)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("SLO: Zustand konnte nicht geladen werden: %s", exc)

    # ------------------------------------------------------------------
    # Legacy-API (rückwärtskompatibel, deprecated)
    # ------------------------------------------------------------------

    def update_from_feedback(self, features: dict[str, float], feedback: float) -> None:
        """[Deprecated] Nutze record_result() stattdessen."""
        for k, v in features.items():
            if k not in self.model_params:
                self.model_params[k] = 1.0
            grad = (feedback - self.predict(features)) * v
            self.model_params[k] += self.learning_rate * grad
        self._legacy_history.append({"features": features, "feedback": feedback})

    def predict(self, features: dict[str, float]) -> float:
        """[Deprecated] Lineare Vorhersage basierend auf gelernten Gewichten."""
        return sum(self.model_params.get(k, 1.0) * v for k, v in features.items())

    def optimize(self, features: dict[str, float]) -> dict[str, float]:
        """[Deprecated] Gibt gewichtete Features zurück."""
        return {k: v * self.model_params.get(k, 1.0) for k, v in features.items()}

    def get_history(self) -> list[dict[str, Any]]:
        """[Deprecated] Legacy-History."""
        return self._legacy_history.copy()
