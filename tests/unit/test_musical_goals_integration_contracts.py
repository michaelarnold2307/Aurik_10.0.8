"""Integration-Contracts für Musical-Goals-Verdrahtung (Low-Effort, High-Signal).

Diese Tests sichern gezielt die reparierten Integrationspunkte ab:
- FeedbackChain Legacy-Kompatibilität (max_retries)
- FeedbackChain GoalPriority-Callback-Pfad
- UnifiedRestorerV3-FeedbackChain-Wiring (max_iterations statt max_retries)
- UnifiedRestorerV3 GoalApplicability/AdaptiveThresholds-Verdrahtung
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.core.feedback_chain import FeedbackChain
from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3


SR = 48_000


def _silence(secs: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


class TestFeedbackChainIntegrationContracts:
    def test_01_legacy_max_retries_supported(self) -> None:
        """Legacy-API aus UnifiedRestorer darf keinen TypeError erzeugen."""
        fc = FeedbackChain(max_retries=2)
        res = fc.run(_silence(), lambda a, sr: a)
        assert res.iterations >= 1
        assert res.total_retries == res.iterations

    def test_02_goal_priority_callback_can_abort(self) -> None:
        """Externer GoalPriority-Callback muss Iteration abbrechen koennen."""
        fc = FeedbackChain(max_iterations=5)

        def _abort_cb(_before: np.ndarray, _after: np.ndarray) -> tuple[bool, str]:
            return True, "contract-abort"

        fc.goal_priority_callback = _abort_cb
        res = fc.run(_silence(), lambda a, sr: a)
        assert res.iterations == 1
        assert any("contract-abort" in s for s in res.metadata.get("goal_priority_log", []))


class TestUnifiedRestorerWiringContracts:
    def test_03_feedback_chain_wiring_uses_max_iterations(self) -> None:
        """UnifiedRestorer muss kompatibel mit FeedbackChain-Signatur bleiben."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "max_iterations=5" in src
        assert "max_retries=5" not in src

    def test_04_goal_wiring_contains_applicability_and_adaptive_thresholds(self) -> None:
        """Applicability + adaptive thresholds muessen im Restore-Result verdrahtet sein."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "_goal_applicability_result = _goal_applicability" in src
        assert "_effective_goal_thresholds" in src
        assert "adaptive_thresholds=locals().get(\"_effective_goal_thresholds\", {})" in src
        assert "goal_applicability=(" in src


class TestUnifiedRestorerRuntimeIntegration:
    @pytest.mark.timeout(120)
    def test_05_restore_populates_goal_fields_minimal(self) -> None:
        """Echter Minimal-Restore: Goal-Felder muessen im Ergebnis gesetzt sein."""
        cfg = RestorationConfig(
            mode=QualityMode.FAST,
            enable_performance_guard=False,
            enable_phase_gate=False,
        )
        restorer = UnifiedRestorerV3(config=cfg)

        # 1s Stille minimiert Laufzeit und vermeidet datenabhaengige Streuung.
        audio = _silence(1.0)
        result = restorer.restore(audio, sample_rate=SR)

        assert isinstance(result.goal_applicability, dict)
        assert isinstance(result.adaptive_thresholds, dict)
        assert isinstance(result.physical_ceiling, dict)

        # Applicability sollte mindestens den Goal-Raum abbilden.
        assert len(result.goal_applicability) >= 6
        assert "natuerlichkeit" in result.goal_applicability

        # Effektive Thresholds sollen aus dem Checker befuellt werden (nicht leer).
        assert len(result.adaptive_thresholds) >= 7
        assert "natuerlichkeit" in result.adaptive_thresholds
