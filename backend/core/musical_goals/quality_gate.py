"""
AURIK v8 Enhanced Quality Gates with Multi-Metric Validation
=============================================================

Pre/Post Validation mit Musical Goals + Perceptual Quality Metrics (NISQA, DNSMOS, ViSQOL, CDPAM).

Component 4.5: Quality Gates Pre/Post Validation (ENHANCED)
Impact: +1.5 Punkte - Garantiert Musical Goals + Objective Quality + Auto-Reprocessing

HIPS Compliance:
- Requirement 1: Explizite Verantwortung (Quality Gate als Gatekeeper)
- Requirement 4: Reversibilität (Automatic Rollback bei Violations + Auto-Reprocessing)
- Requirement 6: Auditierbarkeit (Vollständiges Logging aller Entscheidungen)
- Requirement 8: Normative Einkapselung (Unter ConductEnforcer-Kontrolle)

New in v8.1 (Excellence Strategy #2):
- Multi-Metric Validation: NISQA + DNSMOS + ViSQOL + CDPAM
- Automatic Reprocessing: Intelligente Fallback-Strategien bei Failures
- Perceptual Quality Scores: Objective MOS prediction ohne Human Listening Tests
- Configurable Weights: Flexible metric importance per mode
- False Accept Rate: Target <2% (from 15%)

Solution Architecture:
1. Pre-Check: Musical Goals + Perceptual Metrics Baseline
2. Post-Check: Weighted Multi-Metric Validation
3. Auto-Reprocessing: On failure → Parameter tuning → Alternative chains → Hybrid blend
4. Final Decision: Best of {original, processed, reprocessed, hybrid}

Metrics:
- NISQA: Speech Quality MOS (1-5), trained on ITU-T P.808
- DNSMOS: P.835 Speech Enhancement (SIG, BAK, OVRL scores)
- ViSQOL: Virtual Speech Quality Objective Listener (MOS-LQO 1-5)
- CDPAM: Cross-Domain Perceptual Audio Metrics
- Musical Goals: 7 perceptual dimensions (Brillanz, Wärme, etc.)

Quelle: Excellence Roadmap Strategy #2 - Perceptual Quality Gates
Autor: AI Team
Datum: 8. Februar 2026
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..conduct_enforcer.conduct_enforcer import ConductEnforcer
from .musical_goals_metrics import MusicalGoalsChecker
from .processing_modes import PROCESSING_MODE_CONFIGS, ProcessingMode

logger = logging.getLogger(__name__)


class QualityGateDecision(Enum):
    """Quality Gate decision outcomes"""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ROLLBACK_REQUIRED = "rollback_required"


@dataclass
class PreCheckResult:
    """Result of pre-processing quality gate check"""

    passed: bool
    measurable: bool
    baseline_scores: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    edge_cases_detected: list[str] = field(default_factory=list)
    recommendation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PostCheckResult:
    """Result of post-processing quality gate check"""

    passed: bool
    decision: QualityGateDecision
    baseline_scores: dict[str, float]
    achieved_scores: dict[str, float]
    violations: dict[str, dict[str, float]] = field(default_factory=dict)
    improvements: dict[str, float] = field(default_factory=dict)
    degradations: dict[str, float] = field(default_factory=dict)
    action: str | None = None
    recommendation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityGateReport:
    """Comprehensive quality gate report for auditing"""

    session_id: str
    mode: ProcessingMode
    pre_check: PreCheckResult
    post_check: PostCheckResult | None
    processing_steps: list[str] = field(default_factory=list)
    total_violations: int = 0
    critical_violations: int = 0
    rollback_occurred: bool = False
    final_decision: QualityGateDecision = QualityGateDecision.PASSED
    timestamp_start: str = field(default_factory=lambda: datetime.now().isoformat())
    timestamp_end: str | None = None


class MusicalGoalsQualityGate:
    """
    Pre/Post Musical Goals Quality Gate with automatic rollback.

    Ensures Musical Goals are:
    1. Measurable before processing (Pre-Check)
    2. Continuously monitored during processing
    3. Validated after processing (Post-Check)
    4. Automatically rolled back if critical violations occur

    HIPS Compliance:
    - Explizite Verantwortung: Quality Gate entscheidet über Processing-Erfolg
    - Reversibilität: Automatic Rollback bei Violations
    - Auditierbarkeit: Vollständiges Logging via ProcessingLogger
    - Normative Einkapselung: Nutzt ConductEnforcer für Validierung

    Attributes:
        checker: Musical goals measurement system
        conduct_enforcer: Normative validation engine
        strict_mode: If True, any violation causes rollback
        critical_threshold: Threshold below which goals are critical (default 0.70)
    """

    def __init__(
        self,
        strict_mode: bool = False,
        critical_threshold: float = 0.70,
        conduct_enforcer: ConductEnforcer | None = None,
    ) -> None:
        """
        Initialize Quality Gate.

        Args:
            strict_mode: If True, any violation triggers rollback
            critical_threshold: Below this value, violations are critical
            conduct_enforcer: Optional ConductEnforcer instance
        """
        self.checker = MusicalGoalsChecker()
        self.conduct_enforcer = conduct_enforcer or ConductEnforcer()
        self.strict_mode = strict_mode
        self.critical_threshold = critical_threshold

        # Report history for auditing
        self.reports: list[QualityGateReport] = []

        logger.info(
            f"MusicalGoalsQualityGate initialized "
            f"(strict_mode={strict_mode}, critical_threshold={critical_threshold})"
        )

    def pre_check(
        self,
        audio: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        context: dict[str, Any] | None = None,
    ) -> PreCheckResult:
        """
        Pre-processing check: Validate Goals sind messbar, establish baseline.

        Checks:
        1. Sind alle 7 Musical Goals messbar?
        2. Baseline-Scores für Post-Comparison
        3. Edge Cases Detection (extreme degradation, spectrum conflicts, etc.)

        Args:
            audio: Input audio signal
            sr: Sample rate
            mode: Processing mode (determines thresholds)
            context: Optional context (medium_type, genre, etc.)

        Returns:
            PreCheckResult with measurability check and baseline scores

        HIPS Compliance:
        - Requirement 2: Kontextbewusstsein (prüft Edge Cases)
        - Requirement 6: Auditierbarkeit (loggt alle Entscheidungen)
        """
        context = context or {}
        warnings = []
        edge_cases = []

        mode_str = mode.value if hasattr(mode, "value") else str(mode)
        logger.info(f"Pre-Check started (mode={mode_str})")

        # Measure baseline Musical Goals
        try:
            baseline = self.checker.measure_all(audio, sr)
        except Exception as e:
            logger.error(f"Pre-Check failed: {e}")
            return PreCheckResult(
                passed=False,
                measurable=False,
                baseline_scores={},
                warnings=[f"Measurement failed: {str(e)}"],
                recommendation="Check audio format and sample rate",
            )

        # Check if all goals are measurable (not None/NaN)
        measurable = all(score is not None and not np.isnan(score) for score in baseline.values())

        if not measurable:
            unmeasurable_goals = [goal for goal, score in baseline.items() if score is None or np.isnan(score)]
            warnings.append(f"Unmeasurable goals: {', '.join(unmeasurable_goals)}")

        # Edge Case Detection
        edge_cases_detected = self._detect_edge_cases(audio, sr, baseline, context)
        if edge_cases_detected:
            edge_cases.extend(edge_cases_detected)
            logger.warning(f"Edge cases detected: {edge_cases_detected}")

        # Check for extreme degradation
        if self._is_extremely_degraded(audio, sr, baseline):
            warnings.append("Extreme degradation detected - Musical Goals may not be fully achievable")
            edge_cases.append("extreme_degradation")

        # Spectrum-Goals conflict detection
        spectrum_conflicts = self._check_spectrum_conflicts(audio, sr, baseline, mode)
        if spectrum_conflicts:
            warnings.extend(spectrum_conflicts)
            edge_cases.append("spectrum_conflict")

        # Determine pass/fail
        passed = measurable and len(edge_cases) == 0

        recommendation = None
        if not passed:
            if not measurable:
                recommendation = "Fix unmeasurable goals before processing"
            elif "extreme_degradation" in edge_cases:
                recommendation = "Consider lowering Musical Goals thresholds or using FORENSIC mode"
            elif "spectrum_conflict" in edge_cases:
                recommendation = "Adjust Musical Goals or choose different processing mode"

        result = PreCheckResult(
            passed=passed,
            measurable=measurable,
            baseline_scores=baseline,
            warnings=warnings,
            edge_cases_detected=edge_cases,
            recommendation=recommendation,
        )

        logger.info(f"Pre-Check complete: passed={passed}, " f"warnings={len(warnings)}, edge_cases={len(edge_cases)}")

        return result

    def post_check(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        baseline_scores: dict[str, float] | None = None,
        context: dict[str, Any] | None = None,
        adaptive_thresholds: dict[str, float] | None = None,
    ) -> PostCheckResult:
        """
        Post-processing check: Validate Musical Goals wurden erreicht.

        Checks:
        1. Wurden alle Musical Goals erreicht (>= Thresholds)?
        2. Wo sind Verbesserungen/Verschlechterungen?
        3. Sind Violations kritisch (< critical_threshold)?
        4. Automatic Rollback erforderlich?

        Args:
            original: Original audio
            processed: np.ndarray
            sr: Sample rate
            mode: Processing mode (determines default thresholds)
            baseline_scores: Pre-computed baseline (if available)
            context: Optional context dictionary
            adaptive_thresholds: Optional adaptive thresholds (material-specific).
                               If provided, overrides mode-specific defaults.
                               Allows validation against realistic goals for degraded material.
            context: Optional context

        Returns:
            PostCheckResult with achievement validation and rollback decision

        HIPS Compliance:
        - Requirement 1: Explizite Verantwortung (entscheidet über Rollback)
        - Requirement 4: Reversibilität (fordert Rollback an)
        - Requirement 6: Auditierbarkeit (detaillierter Report)
        """
        context = context or {}

        mode_str = mode.value if hasattr(mode, "value") else str(mode)
        logger.info(f"Post-Check started (mode={mode_str})")

        # Measure baseline if not provided
        if baseline_scores is None:
            try:
                baseline_scores = self.checker.measure_all(original, sr)
            except Exception as e:
                logger.error(f"Baseline measurement failed: {e}")
                baseline_scores = {}

        # Measure achieved goals
        try:
            achieved_scores = self.checker.measure_all(processed, sr)
        except Exception as e:
            logger.error(f"Post-Check measurement failed: {e}")
            return PostCheckResult(
                passed=False,
                decision=QualityGateDecision.FAILED,
                baseline_scores=baseline_scores,
                achieved_scores={},
                action="rollback",
                recommendation="Processing failed - rollback to original",
            )

        # Get thresholds (adaptive if available, otherwise mode-specific defaults)
        if adaptive_thresholds:
            thresholds = adaptive_thresholds
            logger.info("Using ADAPTIVE thresholds (material-specific, relaxed for degraded material)")
        elif context and "adaptive_thresholds" in context:
            thresholds = context["adaptive_thresholds"]
            logger.info("Using ADAPTIVE thresholds from context")
        else:
            mode_config = PROCESSING_MODE_CONFIGS[mode]
            thresholds = mode_config.musical_goals
            mode_str = mode.value if hasattr(mode, "value") else str(mode)
            logger.info(f"Using DEFAULT thresholds (mode: {mode_str})")

        # Check for violations
        violations = {}
        improvements = {}
        degradations = {}
        critical_violations = []

        for goal_name, threshold in thresholds.items():
            achieved = float(np.nan_to_num(achieved_scores.get(goal_name, 0.0), nan=0.0))
            baseline = float(np.nan_to_num(baseline_scores.get(goal_name, 0.0), nan=0.0))
            delta = achieved - baseline

            # Check if goal achieved
            if achieved < threshold:
                violations[goal_name] = {
                    "expected": threshold,
                    "achieved": achieved,
                    "delta": delta,
                    "baseline": baseline,
                }

                # Critical violation?
                if achieved < self.critical_threshold:
                    critical_violations.append(goal_name)
                    logger.error(
                        f"CRITICAL VIOLATION: {goal_name} = {achieved:.3f} "
                        f"< {self.critical_threshold} (threshold: {threshold:.3f})"
                    )
                else:
                    logger.warning(
                        f"Violation: {goal_name} = {achieved:.3f} " f"< {threshold:.3f} (baseline: {baseline:.3f})"
                    )

            # Track improvements/degradations
            if delta > 0.01:  # Improved by >1%
                improvements[goal_name] = delta
            elif delta < -0.01:  # Degraded by >1%
                degradations[goal_name] = delta

        # Determine decision
        if violations:
            if critical_violations:
                decision = QualityGateDecision.ROLLBACK_REQUIRED
                action = "rollback"
                recommendation = (
                    f"Critical violations detected in {len(critical_violations)} goals. "
                    f"Rollback required: {', '.join(critical_violations)}"
                )
                passed = False
            elif self.strict_mode:
                decision = QualityGateDecision.ROLLBACK_REQUIRED
                action = "rollback"
                recommendation = (
                    f"Strict mode: {len(violations)} violations detected. "
                    f"Rollback required: {', '.join(violations.keys())}"
                )
                passed = False
            else:
                decision = QualityGateDecision.WARNING
                action = "warn"
                recommendation = (
                    f"Non-critical violations in {len(violations)} goals. " f"Consider adjusting processing parameters."
                )
                passed = False
        else:
            decision = QualityGateDecision.PASSED
            action = None
            recommendation = f"All Musical Goals achieved! Improvements: {len(improvements)}"
            passed = True

        result = PostCheckResult(
            passed=passed,
            decision=decision,
            baseline_scores=baseline_scores,
            achieved_scores=achieved_scores,
            violations=violations,
            improvements=improvements,
            degradations=degradations,
            action=action,
            recommendation=recommendation,
        )

        decision_str = decision.value if hasattr(decision, "value") else str(decision)
        logger.info(
            f"Post-Check complete: passed={passed}, "
            f"violations={len(violations)}, critical={len(critical_violations)}, "
            f"improvements={len(improvements)}, decision={decision_str}"
        )

        return result

    def validate_processing(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        session_id: str | None = None,
        processing_steps: list[str] | None = None,
        adaptive_thresholds: dict[str, float] | None = None,
    ) -> QualityGateReport:
        """
        Full validation: Pre-Check + Post-Check + Report generation.

        Complete workflow:
        1. Pre-Check on original
        2. (Processing happens externally)
        3. Post-Check on processed
        4. Generate comprehensive report
        5. Return rollback decision

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate
            mode: Processing mode
            session_id: Optional session identifier
            processing_steps: List of processing step names
            adaptive_thresholds: Optional adaptive thresholds from Adaptive Goals System.
                               If provided, post-check validates against material-specific goals.

        Returns:
            QualityGateReport with full analysis and decisions

        Example:
            >>> gate = MusicalGoalsQualityGate()
            >>>
            >>> # With adaptive thresholds (RECOMMENDED for degraded material)
            >>> report = gate.validate_processing(
            ...     original, processed, sr,
            ...     mode=ProcessingMode.RESTORATION,
            ...     adaptive_thresholds=restorer._adaptive_thresholds  # From Adaptive Goals System
            ... )
            >>>
            >>> # Without adaptive thresholds (uses mode defaults)
            >>> report = gate.validate_processing(
            ...     original, processed, sr, mode=ProcessingMode.STUDIO_2026
            ... )
            >>>
            >>> if report.final_decision == QualityGateDecision.ROLLBACK_REQUIRED:
            ...     audio = original  # Rollback
            >>> else:
            ...     audio = processed  # Accept
        """
        session_id = session_id or f"qg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        processing_steps = processing_steps or []

        mode_str = mode.value if hasattr(mode, "value") else str(mode)
        logger.info(f"Full validation started (session={session_id}, mode={mode_str})")

        # Pre-Check
        pre_check = self.pre_check(original, sr, mode)

        # Post-Check with adaptive thresholds (if available)
        post_check = self.post_check(
            original,
            processed,
            sr,
            mode,
            baseline_scores=pre_check.baseline_scores,
            adaptive_thresholds=adaptive_thresholds,
        )

        # Generate report
        report = QualityGateReport(
            session_id=session_id,
            mode=mode,
            pre_check=pre_check,
            post_check=post_check,
            processing_steps=processing_steps,
            total_violations=len(post_check.violations),
            critical_violations=sum(
                1 for v in post_check.violations.values() if v["achieved"] < self.critical_threshold
            ),
            rollback_occurred=(post_check.decision == QualityGateDecision.ROLLBACK_REQUIRED),
            final_decision=post_check.decision,
            timestamp_end=datetime.now().isoformat(),
        )

        # Store report
        self.reports.append(report)

        final_decision_str = (
            report.final_decision.value if hasattr(report.final_decision, "value") else str(report.final_decision)
        )
        logger.info(
            f"Full validation complete: "
            f"pre_passed={pre_check.passed}, "
            f"post_passed={post_check.passed}, "
            f"final_decision={final_decision_str}"
        )

        return report

    # =========================================================================
    # Edge Case Detection (Private Methods)
    # =========================================================================

    def _detect_edge_cases(
        self, audio: np.ndarray, sr: int, baseline: dict[str, float], context: dict[str, Any]
    ) -> list[str]:
        """
        Detect edge cases that may impact Musical Goals achievability.

        Edge Cases:
        - Extreme degradation (SNR < 30 dB, >80% defects)
        - Spectrum conflicts (missing bands vs. goal requirements)
        - Unknown medium type
        - Mixed medium (vinyl+tape)
        """
        edge_cases = []

        # Check for extreme degradation
        if self._is_extremely_degraded(audio, sr, baseline):
            edge_cases.append("extreme_degradation")

        # Check for spectrum conflicts
        if self._has_spectrum_conflict(audio, sr, baseline):
            edge_cases.append("spectrum_conflict")

        # Check for mixed/unknown medium
        medium_type = context.get("medium_type", "unknown")
        if medium_type == "unknown":
            edge_cases.append("unknown_medium")
        elif "+" in medium_type or "," in medium_type:
            edge_cases.append("mixed_medium")

        return edge_cases

    def _is_extremely_degraded(self, audio: np.ndarray, sr: int, baseline: dict[str, float]) -> bool:
        """Check if audio is extremely degraded (SNR < 30 dB or Goals < 50%)."""
        # Simple heuristic: If multiple goals are < 0.50, likely extreme degradation
        low_goals = sum(1 for score in baseline.values() if score < 0.50)
        return low_goals >= 3  # 3+ goals < 50%

    def _has_spectrum_conflict(self, audio: np.ndarray, sr: int, baseline: dict[str, float]) -> bool:
        """Check for spectrum-goals conflicts (e.g., no HF but brillanz required)."""
        # Check if bass-kraft is low but required
        if baseline.get("bass-kraft", 0) < 0.30 and baseline.get("bass-kraft", 0) < 1.0:
            return True

        # Check if brillanz is low but required
        if baseline.get("brillanz", 0) < 0.30 and baseline.get("brillanz", 0) < 1.0:
            return True

        return False

    def _check_spectrum_conflicts(
        self, audio: np.ndarray, sr: int, baseline: dict[str, float], mode: ProcessingMode
    ) -> list[str]:
        """
        Detect spectrum-goals conflicts and generate warnings.

        Examples:
        - No HF content but mode requires brillanz=0.95
        - No bass content but mode requires bass-kraft=0.90
        """
        warnings = []
        mode_config = PROCESSING_MODE_CONFIGS[mode]
        thresholds = mode_config.musical_goals

        mode_str = mode.value if hasattr(mode, "value") else str(mode)
        # Check brillanz conflict
        if baseline.get("brillanz", 0) < 0.30 and thresholds.get("brillanz", 0) > 0.85:
            warnings.append(
                f"Spectrum conflict: No HF content but {mode_str} requires brillanz={thresholds['brillanz']}"
            )

        # Check bass-kraft conflict
        if baseline.get("bass-kraft", 0) < 0.30 and thresholds.get("bass-kraft", 0) > 0.85:
            warnings.append(
                f"Spectrum conflict: No bass content but {mode_str} requires bass-kraft={thresholds['bass-kraft']}"
            )

        return warnings

    # =========================================================================
    # Report Export (for Auditing)
    # =========================================================================

    def export_report(self, report: QualityGateReport, output_path: Path) -> None:
        """
        Export Quality Gate Report to JSON for auditing.

        HIPS Requirement 6: Auditierbarkeit

        Args:
            report: QualityGateReport to export
            output_path: Path to JSON file
        """
        import json

        report_dict = {
            "session_id": report.session_id,
            "mode": report.mode.value if hasattr(report.mode, "value") else str(report.mode),
            "timestamp_start": report.timestamp_start,
            "timestamp_end": report.timestamp_end,
            "pre_check": {
                "passed": report.pre_check.passed,
                "measurable": report.pre_check.measurable,
                "baseline_scores": report.pre_check.baseline_scores,
                "warnings": report.pre_check.warnings,
                "edge_cases": report.pre_check.edge_cases_detected,
                "recommendation": report.pre_check.recommendation,
            },
            "post_check": {
                "passed": report.post_check.passed if report.post_check else None,
                "decision": (
                    report.post_check.decision.value
                    if report.post_check and hasattr(report.post_check.decision, "value")
                    else (str(report.post_check.decision) if report.post_check else None)
                ),
                "violations": report.post_check.violations if report.post_check else {},
                "improvements": report.post_check.improvements if report.post_check else {},
                "degradations": report.post_check.degradations if report.post_check else {},
                "action": report.post_check.action if report.post_check else None,
                "recommendation": report.post_check.recommendation if report.post_check else None,
            },
            "processing_steps": report.processing_steps,
            "summary": {
                "total_violations": report.total_violations,
                "critical_violations": report.critical_violations,
                "rollback_occurred": report.rollback_occurred,
                "final_decision": (
                    report.final_decision.value
                    if hasattr(report.final_decision, "value")
                    else str(report.final_decision)
                ),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report exported to {output_path}")

    def get_recent_reports(self, n: int = 10) -> list[QualityGateReport]:
        """Get n most recent reports."""
        return self.reports[-n:]

    def clear_reports(self) -> None:
        """Clear report history."""
        self.reports.clear()
        logger.info("Report history cleared")


# =============================================================================
# Enhanced Quality Gate with Multi-Metric Validation (Excellence Strategy #2)
# =============================================================================


@dataclass
class PerceptualMetrics:
    """Perceptual quality metrics: ViSQOL v3 (--audio mode) + CDPAM.

    Hinweis §4.4/§10.2: NISQA und DNSMOS sind für Musik-Qualitätsbewertung
    VERBOTEN (Sprach-Modelle). nisqa_mos/dnsmos_* sind als deaktivierte Compat-
    Felder (Wert 0.0) beibehalten, damit bestehende Aufrufer nicht brechen.
    Aktive Metriken: visqol_mos_lqo (--audio mode), cdpam_score.
    """

    nisqa_mos: float = 0.0  # DEAKTIVIERT §10.2 — Sprach-Modell, nicht für Musik
    dnsmos_ovrl: float = 0.0  # DEAKTIVIERT §10.2 — Sprach-Modell, nicht für Musik
    dnsmos_sig: float = 0.0  # DEAKTIVIERT §10.2 — Sprach-Modell, nicht für Musik
    dnsmos_bak: float = 0.0  # DEAKTIVIERT §10.2 — Sprach-Modell, nicht für Musik
    visqol_mos_lqo: float = 0.0  # 1-5 scale — ViSQOL v3 --audio mode (§4.4 erlaubt)
    cdpam_score: float = 0.0  # 0-100 scale — Perceptual similarity Musik (§4.4 erlaubt)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EnhancedPreCheckResult:
    """Pre-check result with Musical Goals + Perceptual Metrics."""

    passed: bool
    measurable: bool
    baseline_musical_goals: dict[str, float]
    baseline_perceptual: PerceptualMetrics | None
    warnings: list[str] = field(default_factory=list)
    edge_cases_detected: list[str] = field(default_factory=list)
    recommendation: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EnhancedPostCheckResult:
    """Post-check result with multi-metric decision logic."""

    passed: bool
    decision: QualityGateDecision
    baseline_musical_goals: dict[str, float]
    achieved_musical_goals: dict[str, float]
    baseline_perceptual: PerceptualMetrics | None
    achieved_perceptual: PerceptualMetrics | None
    violations: dict[str, dict[str, float]] = field(default_factory=dict)
    perceptual_improvements: dict[str, float] = field(default_factory=dict)
    perceptual_degradations: dict[str, float] = field(default_factory=dict)
    weighted_quality_score: float = 0.0  # Combined score 0-1
    action: str | None = None
    recommendation: str | None = None
    reprocessing_performed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EnhancedQualityGate:
    """
    Enhanced Quality Gate mit Multi-Metric Validation + Auto-Reprocessing.

    Excellence Strategy #2: Perceptual Quality Gates
    - Combines Musical Goals (14 Dimensionen §1.2) + Perceptual Metrics (ViSQOL v3, CDPAM)
    - Weighted decision logic: configurable metric importance
    - Automatic reprocessing on failure with intelligent fallback strategies
    - Target: False Accept Rate <2% (from 15%)

    §4.4/§10.2: NISQA und DNSMOS sind VERBOTEN (Sprach-Modelle).
    Erlaubte Metriken: PEAQ, FAD, CDPAM, PQS-MOS, ViSQOL v3 (--audio), Musical Goals.

    Attributes:
        musical_gate: Base Musical Goals quality gate
        enable_perceptual_metrics: If True, use ViSQOL v3/CDPAM
        enable_auto_reprocessing: If True, attempt automatic reprocessing on failure
        metric_weights: Importance weights for each metric category
        nisqa_threshold: DEAKTIVIERT §10.2 (Compat-Feld, wird ignoriert)
        dnsmos_threshold: DEAKTIVIERT §10.2 (Compat-Feld, wird ignoriert)
        visqol_threshold: Minimum ViSQOL v3 MOS-LQO --audio mode (default: 3.0/5.0)
        cdpam_threshold: Minimum CDPAM score — Musik-Wahrnehmungsähnlichkeit (default: 80/100)
    """

    def __init__(
        self,
        strict_mode: bool = False,
        critical_threshold: float = 0.70,
        enable_perceptual_metrics: bool = True,
        enable_auto_reprocessing: bool = True,
        metric_weights: dict[str, float] | None = None,
        nisqa_threshold: float = 3.5,
        dnsmos_threshold: float = 3.2,
        visqol_threshold: float = 3.0,
        cdpam_threshold: float = 80.0,
        conduct_enforcer: ConductEnforcer | None = None,
    ) -> None:
        """
        Initialize Enhanced Quality Gate.

        Args:
            strict_mode: If True, any violation triggers rollback
            critical_threshold: Below this, violations are critical
            enable_perceptual_metrics: Use NISQA/DNSMOS/ViSQOL/CDPAM
            enable_auto_reprocessing: Attempt reprocessing on failure
            metric_weights: Importance weights (default: musical=0.50, perceptual=0.50)
            nisqa_threshold: NISQA MOS threshold (1-5)
            dnsmos_threshold: DNSMOS Overall threshold (1-5)
            visqol_threshold: ViSQOL MOS-LQO threshold (1-5)
            cdpam_threshold: CDPAM score threshold (0-100)
            conduct_enforcer: Optional ConductEnforcer instance
        """
        # Base Musical Goals gate
        self.musical_gate = MusicalGoalsQualityGate(
            strict_mode=strict_mode, critical_threshold=critical_threshold, conduct_enforcer=conduct_enforcer
        )

        # Enhanced features
        self.enable_perceptual_metrics = enable_perceptual_metrics
        self.enable_auto_reprocessing = enable_auto_reprocessing

        # Metric weights (musical vs perceptual importance)
        if metric_weights is None:
            self.metric_weights = {"musical_goals": 0.50, "perceptual_quality": 0.50}
        else:
            self.metric_weights = metric_weights

        # Perceptual thresholds
        # nisqa_threshold/dnsmos_threshold: DEAKTIVIERT §10.2 — Compat-Parameter,
        # werden intern ignoriert (NISQA/DNSMOS sind Sprach-Metriken, verboten für Musik)
        self.nisqa_threshold = nisqa_threshold  # Compat-only, nicht aktiv
        self.dnsmos_threshold = dnsmos_threshold  # Compat-only, nicht aktiv
        self.visqol_threshold = visqol_threshold  # ViSQOL v3 --audio mode (§4.4 erlaubt)
        self.cdpam_threshold = cdpam_threshold  # §4.4: VERSA ersetzt CDPAM, Schwellwert bleibt 0-100-kompatibel

        # Quality plugins (lazy loading)
        # _nisqa_plugin/_dnsmos_plugin: deaktiviert §10.2 (Sprach-Modelle)
        self._nisqa_plugin = None  # deaktiviert
        self._dnsmos_plugin = None  # deaktiviert
        self._visqol_plugin = None
        self._versa_plugin = None  # VERSA 2024 non-reference MOS (§4.4)

        # Auto-reprocessing engine (lazy loading)
        self._reprocessing_engine = None

        logger.info(
            f"EnhancedQualityGate initialized: "
            f"perceptual={enable_perceptual_metrics}, "
            f"auto_reprocessing={enable_auto_reprocessing}, "
            f"weights={metric_weights}"
        )

    def _get_nisqa_plugin(self):
        """DEAKTIVIERT §10.2 — NISQA ist ein Sprach-Modell, verboten für Musik.
        Gibt immer None zurück; Compat-Stub damit alte Aufrufer nicht brechen.
        """
        return None  # §10.2: NISQA verboten für Musik-Qualitätsbewertung

    def _get_dnsmos_plugin(self):
        """DEAKTIVIERT §10.2 — DNSMOS P.835 ist ein Sprach-Modell, verboten für Musik.
        Gibt immer None zurück; Compat-Stub damit alte Aufrufer nicht brechen.
        """
        return None  # §10.2: DNSMOS verboten für Musik-Qualitätsbewertung

    def _get_visqol_plugin(self):
        """Lazy load ViSQOL plugin."""
        if self._visqol_plugin is None:
            try:
                from plugins.visqol_plugin import ViSQOLPlugin

                self._visqol_plugin = ViSQOLPlugin()
                logger.info("ViSQOL plugin loaded")
            except Exception as e:
                logger.warning(f"ViSQOL plugin unavailable: {e}")
        return self._visqol_plugin

    def _get_versa_plugin(self):
        """Lazy load VERSA plugin (§4.4 CDPAM-Nachfolger)."""
        if self._versa_plugin is None:
            try:
                from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

                self._versa_plugin = get_versa_plugin()
                logger.info("VERSA plugin loaded (§4.4)")
            except Exception as e:
                logger.warning(f"VERSA plugin unavailable: {e}")
        return self._versa_plugin

    def _get_reprocessing_engine(self):
        """Lazy load auto-reprocessing engine."""
        if self._reprocessing_engine is None:
            from .auto_reprocessing import AutoReprocessingEngine

            self._reprocessing_engine = AutoReprocessingEngine(
                max_attempts=5, min_improvement=0.02, enable_hybrid_fallback=True, enable_forensic_guidance=True
            )
            logger.info("Auto-reprocessing engine loaded")
        return self._reprocessing_engine

    def _measure_perceptual_metrics(
        self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None
    ) -> PerceptualMetrics | None:
        """
        Measure perceptual quality metrics (NISQA, DNSMOS, ViSQOL, CDPAM).

        Args:
            audio: Audio to measure
            sr: Sample rate
            reference: Reference audio (for ViSQOL, CDPAM - full-reference metrics)

        Returns:
            PerceptualMetrics or None if plugins unavailable
        """
        if not self.enable_perceptual_metrics:
            return None

        import tempfile

        import soundfile as sf

        # Save audio to temp files for plugins
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            audio_path = Path(tmp.name)

        try:
            # NISQA und DNSMOS sind VERBOTEN als Musik-Metriken (§10.2, §4.4).
            # NISQA: Deep-CNN für Sprach-Qualitäts-Prediction — keine Musik-Trainingsdaten.
            # DNSMOS P.835: Trainiert auf 16 kHz DNS-Challenge-Sprachkorpus.
            # Erlaubte Metriken: PEAQ, FAD, CDPAM, PQS-MOS, ViSQOL v3 (--audio), Musical Goals.
            nisqa_mos = 0.0  # deaktiviert §10.2
            dnsmos_ovrl = 0.0  # deaktiviert §10.2
            dnsmos_sig = 0.0  # deaktiviert §10.2
            dnsmos_bak = 0.0  # deaktiviert §10.2

            # ViSQOL (Full-Reference - requires reference audio)
            visqol_mos = 0.0
            if reference is not None:
                visqol = self._get_visqol_plugin()
                if visqol is not None:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp:
                            sf.write(ref_tmp.name, reference, sr)
                            ref_path = Path(ref_tmp.name)

                        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
                            scores = visqol.calculate(str(ref_path), str(audio_path), out.name)
                            visqol_mos = float(scores.get("MOS_LQO", 0.0))
                            Path(out.name).unlink(missing_ok=True)

                        ref_path.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"ViSQOL failed: {e}")

            # VERSA (non-reference MOS, §4.4 CDPAM-Nachfolger)
            cdpam_score = 0.0
            versa = self._get_versa_plugin()
            if versa is not None:
                try:
                    import numpy as _np  # noqa: PLC0415
                    import soundfile as sf  # noqa: PLC0415

                    audio_arr, sr_v = sf.read(str(audio_path), always_2d=False)
                    audio_arr = _np.asarray(audio_arr, dtype=_np.float32)
                    if audio_arr.ndim == 2:
                        audio_arr = audio_arr.mean(axis=1)
                    audio_arr = _np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    versa_result = versa.score(audio_arr, sr_v)
                    # MOS [1,5] → [0,100] für PerceptualMetrics.cdpam_score
                    cdpam_score = float(_np.clip((versa_result.mos - 1.0) / 4.0 * 100.0, 0.0, 100.0))
                except Exception as e:
                    logger.warning(f"VERSA failed: {e}")

            return PerceptualMetrics(
                nisqa_mos=nisqa_mos,
                dnsmos_ovrl=dnsmos_ovrl,
                dnsmos_sig=dnsmos_sig,
                dnsmos_bak=dnsmos_bak,
                visqol_mos_lqo=visqol_mos,
                cdpam_score=cdpam_score,
            )

        finally:
            # Cleanup
            audio_path.unlink(missing_ok=True)

    def enhanced_pre_check(
        self,
        audio: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        context: dict[str, Any] | None = None,
    ) -> EnhancedPreCheckResult:
        """
        Enhanced pre-check: Musical Goals + Perceptual Metrics baseline.

        Args:
            audio: Input audio
            sr: Sample rate
            mode: Processing mode
            context: Optional context

        Returns:
            EnhancedPreCheckResult with baselines
        """
        # Musical Goals baseline
        musical_pre = self.musical_gate.pre_check(audio, sr, mode, context)

        # Perceptual metrics baseline (No-Reference metrics only in pre-check)
        perceptual_baseline = None
        if self.enable_perceptual_metrics:
            logger.info("Measuring perceptual metrics baseline...")
            perceptual_baseline = self._measure_perceptual_metrics(audio, sr)

        return EnhancedPreCheckResult(
            passed=musical_pre.passed,
            measurable=musical_pre.measurable,
            baseline_musical_goals=musical_pre.baseline_scores,
            baseline_perceptual=perceptual_baseline,
            warnings=musical_pre.warnings,
            edge_cases_detected=musical_pre.edge_cases_detected,
            recommendation=musical_pre.recommendation,
        )

    def enhanced_post_check(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        baseline_musical: dict[str, float] | None = None,
        baseline_perceptual: PerceptualMetrics | None = None,
        context: dict[str, Any] | None = None,
    ) -> EnhancedPostCheckResult:
        """
        Enhanced post-check: Musical Goals + Perceptual Metrics + Weighted Decision.

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate
            mode: Processing mode
            baseline_musical: Musical Goals baseline
            baseline_perceptual: Perceptual metrics baseline
            context: Optional context

        Returns:
            EnhancedPostCheckResult with multi-metric decision
        """
        # Musical Goals validation
        musical_post = self.musical_gate.post_check(
            original, processed, sr, mode, baseline_scores=baseline_musical, context=context
        )

        # Perceptual metrics (Full-Reference with original as reference)
        perceptual_achieved = None
        perceptual_improvements = {}
        perceptual_degradations = {}

        if self.enable_perceptual_metrics:
            logger.info("Measuring perceptual metrics on processed...")
            perceptual_achieved = self._measure_perceptual_metrics(processed, sr, reference=original)

            # Calculate improvements/degradations
            if perceptual_achieved and baseline_perceptual:
                # "nisqa" / "dnsmos" entfernt — verboten §4.4+§10.2 (Sprach-Metriken, Werte immer 0.0)
                perceptual_improvements = {
                    "visqol": perceptual_achieved.visqol_mos_lqo - baseline_perceptual.visqol_mos_lqo,
                    "cdpam": perceptual_achieved.cdpam_score - baseline_perceptual.cdpam_score,
                }

                # Identify degradations (negative improvements)
                perceptual_degradations = {k: v for k, v in perceptual_improvements.items() if v < -0.1}

        # Weighted quality score calculation
        weighted_score = self._calculate_weighted_quality_score(musical_post.achieved_scores, perceptual_achieved, mode)

        # Multi-metric decision logic
        decision, action, recommendation = self._make_multi_metric_decision(
            musical_post, perceptual_achieved, perceptual_degradations, weighted_score
        )

        return EnhancedPostCheckResult(
            passed=(decision == QualityGateDecision.PASSED),
            decision=decision,
            baseline_musical_goals=baseline_musical or {},
            achieved_musical_goals=musical_post.achieved_scores,
            baseline_perceptual=baseline_perceptual,
            achieved_perceptual=perceptual_achieved,
            violations=musical_post.violations,
            perceptual_improvements=perceptual_improvements,
            perceptual_degradations=perceptual_degradations,
            weighted_quality_score=weighted_score,
            action=action,
            recommendation=recommendation,
            reprocessing_performed=False,
        )

    def _calculate_weighted_quality_score(
        self, musical_scores: dict[str, float], perceptual: PerceptualMetrics | None, mode: ProcessingMode
    ) -> float:
        """
        Calculate weighted quality score combining Musical Goals + Perceptual Metrics.

        Returns:
            Combined score 0-1 (higher is better)
        """
        # Musical Goals average (already 0-1 normalized)
        musical_avg = np.mean(list(musical_scores.values())) if musical_scores else 0.0

        # Perceptual metrics average (normalize MOS 1-5 → 0-1, CDPAM 0-100 → 0-1)
        if perceptual:
            # nisqa_mos / dnsmos_ovrl entfernt — verboten §4.4+§10.2 (Sprach-Metriken, immer 0.0 → würden Mittelwert auf -0.25 ziehen)
            perceptual_avg = np.mean(
                [
                    (perceptual.visqol_mos_lqo - 1.0) / 4.0,
                    perceptual.cdpam_score / 100.0,  # 0-100 → 0-1
                ]
            )
        else:
            perceptual_avg = 0.0

        # Weighted combination
        weighted = (
            self.metric_weights["musical_goals"] * musical_avg
            + self.metric_weights["perceptual_quality"] * perceptual_avg
        )

        return float(np.clip(weighted, 0.0, 1.0))

    def _make_multi_metric_decision(
        self,
        musical_result: PostCheckResult,
        perceptual: PerceptualMetrics | None,
        perceptual_degradations: dict[str, float],
        weighted_score: float,
    ) -> tuple[QualityGateDecision, str | None, str | None]:
        """
        Make decision based on Musical Goals + Perceptual Metrics.

        Returns:
            (decision, action, recommendation)
        """
        # Check Musical Goals
        musical_passed = musical_result.passed
        musical_critical = musical_result.decision == QualityGateDecision.ROLLBACK_REQUIRED

        # Check Perceptual Metrics
        perceptual_passed = True
        perceptual_critical = False

        if perceptual:
            # Check thresholds
            # nisqa_mos / dnsmos_ovrl Schwellwert-Prüfungen entfernt — verboten §4.4+§10.2
            # (Werte immer 0.0, würden immer perceptual_critical=True erzwingen → ROLLBACK_REQUIRED bei jedem Gate)
            if perceptual.visqol_mos_lqo < self.visqol_threshold:
                perceptual_passed = False

            if perceptual.cdpam_score < self.cdpam_threshold:
                perceptual_passed = False

        # Combined decision
        if musical_passed and perceptual_passed:
            return (QualityGateDecision.PASSED, None, f"All gates passed! Weighted score: {weighted_score:.3f}")

        elif musical_critical or perceptual_critical:
            return (
                QualityGateDecision.ROLLBACK_REQUIRED,
                "reprocess_or_rollback",
                f"Critical violations detected. Weighted score: {weighted_score:.3f}",
            )

        else:
            return (
                QualityGateDecision.WARNING,
                "warn",
                f"Non-critical violations. Consider reprocessing. Weighted score: {weighted_score:.3f}",
            )

    def validate_with_auto_reprocessing(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        sr: int,
        mode: ProcessingMode,
        processing_function: Callable[[np.ndarray, int, dict[str, Any]], np.ndarray],
        context: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> tuple[np.ndarray, EnhancedPostCheckResult]:
        """
        Full validation with automatic reprocessing on failure.

        Excellence Strategy #2: Perceptual Quality Gates

        Complete Workflow:
        1. Pre-check on original (baseline)
        2. Post-check on processed
        3. If failed and auto-reprocessing enabled:
           a. Trigger AutoReprocessingEngine
           b. Try fallback strategies (parameter reduction, alternative chains, hybrid blend)
           c. Validate each attempt
           d. Return best result
        4. Else: return processed or rollback to original

        Args:
            original: Original audio
            processed: Processed audio
            sr: Sample rate
            mode: Processing mode
            processing_function: Function to reprocess audio
                Signature: (audio, sr, params) -> processed_audio
            context: Optional context (forensics, medium_type, etc.)
            session_id: Session identifier for auditing

        Returns:
            (final_audio, enhanced_post_check_result)
            final_audio: Best audio (processed, reprocessed, or original)
            enhanced_post_check_result: Validation results

        Example:
            >>> gate = EnhancedQualityGate(enable_auto_reprocessing=True)
            >>> final_audio, result = gate.validate_with_auto_reprocessing(
            ...     original, processed, sr,
            ...     mode=ProcessingMode.STUDIO_2026,
            ...     processing_function=my_processor
            ... )
            >>> if result.passed:
            ...     save_audio(final_audio, "output.wav")
        """
        context = context or {}
        session_id = session_id or f"eqg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            f"Enhanced validation started (session={session_id}, "
            f"mode={mode.value}, auto_reprocessing={self.enable_auto_reprocessing})"
        )

        # Pre-check
        pre_check = self.enhanced_pre_check(original, sr, mode, context)

        # Post-check on initial processed
        post_check = self.enhanced_post_check(
            original,
            processed,
            sr,
            mode,
            baseline_musical=pre_check.baseline_musical_goals,
            baseline_perceptual=pre_check.baseline_perceptual,
            context=context,
        )

        # Check if reprocessing needed
        if (
            not post_check.passed
            and self.enable_auto_reprocessing
            and post_check.decision in [QualityGateDecision.ROLLBACK_REQUIRED, QualityGateDecision.WARNING]
        ):

            logger.info(f"Quality gates failed ({post_check.decision.value}), " f"triggering automatic reprocessing...")

            # Define quality validator for reprocessing engine
            def quality_validator(orig, proc, sample_rate):
                """Validator for AutoReprocessingEngine."""
                post = self.enhanced_post_check(
                    orig,
                    proc,
                    sample_rate,
                    mode,
                    baseline_musical=pre_check.baseline_musical_goals,
                    baseline_perceptual=pre_check.baseline_perceptual,
                    context=context,
                )

                passed = post.passed
                scores = {**post.achieved_musical_goals}
                if post.achieved_perceptual:
                    scores.update(
                        {
                            # "nisqa" / "dnsmos" entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
                            "visqol": post.achieved_perceptual.visqol_mos_lqo,
                            "cdpam": post.achieved_perceptual.cdpam_score,
                        }
                    )

                violations = post.violations

                return passed, scores, violations

            # Trigger auto-reprocessing
            engine = self._get_reprocessing_engine()
            reprocessing_result = engine.reprocess_on_failure(
                original=original,
                failed_processed=processed,
                sr=sr,
                processing_function=processing_function,
                quality_validator=quality_validator,
                baseline_scores=pre_check.baseline_musical_goals,
                initial_violations=post_check.violations,
                context=context,
            )

            logger.info(
                f"Auto-reprocessing complete: "
                f"success={reprocessing_result.success}, "
                f"attempts={reprocessing_result.total_attempts}, "
                f"strategy={reprocessing_result.strategy_used.value}"
            )

            # Validate best result from reprocessing
            final_post_check = self.enhanced_post_check(
                original,
                reprocessing_result.best_audio,
                sr,
                mode,
                baseline_musical=pre_check.baseline_musical_goals,
                baseline_perceptual=pre_check.baseline_perceptual,
                context=context,
            )

            # Mark reprocessing flag
            final_post_check.reprocessing_performed = True
            final_post_check.recommendation = (
                f"{final_post_check.recommendation} "
                f"(After {reprocessing_result.total_attempts} reprocessing attempts)"
            )

            return reprocessing_result.best_audio, final_post_check

        else:
            # No reprocessing needed or disabled
            if post_check.passed:
                logger.info("✓ Quality gates passed without reprocessing")
                return processed, post_check
            else:
                logger.warning(
                    f"Quality gates failed but auto-reprocessing disabled. " f"Decision: {post_check.decision.value}"
                )

                # Rollback to original if critical
                if post_check.decision == QualityGateDecision.ROLLBACK_REQUIRED:
                    logger.warning("Rolling back to original audio")
                    return original, post_check
                else:
                    return processed, post_check
