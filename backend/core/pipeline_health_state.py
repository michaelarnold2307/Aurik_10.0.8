"""Shared pipeline health state helpers for Denker/UV3/UI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# §1.4a Structured FailReason (v9.10.130)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailReason:
    """Structured failure reason for critical quality modules.

    §1.4a mandates that PQS, HolisticPerceptualGate, ArtifactFreedomGate,
    and MusicalGoalsChecker produce structured fail_reason entries on failures.

    Attributes
    ----------
    component : str
        Name of the failing module (e.g. "HolisticPerceptualGate", "ArtifactFreedomGate").
    error_code : str
        Machine-readable code (e.g. "HPI_BELOW_ZERO", "ARTIFACT_VETO", "P1_REGRESSION").
    severity : str
        One of "failed", "degraded", "warning".
    action : str
        What the system did in response (e.g. "rollback", "safe_mode", "bypass").
    details : str
        Human-readable explanation.
    phase_id : str
        Phase that triggered the failure, if applicable.
    """

    component: str
    error_code: str
    severity: str = "failed"
    action: str = ""
    details: str = ""
    phase_id: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert to legacy dict format for backward compatibility."""
        return {
            "component": self.component,
            "error_code": self.error_code,
            "severity": self.severity,
            "action": self.action,
            "details": self.details,
            "phase_id": self.phase_id,
        }


def make_fail_reason(
    component: str,
    error_code: str,
    *,
    severity: str = "failed",
    action: str = "",
    details: str = "",
    phase_id: str = "",
) -> FailReason:
    """Factory for FailReason — convenience wrapper."""
    return FailReason(
        component=component,
        error_code=error_code,
        severity=severity,
        action=action,
        details=details,
        phase_id=phase_id,
    )


# ---------------------------------------------------------------------------
# PipelineHealthState
# ---------------------------------------------------------------------------


class PipelineHealthState(str, Enum):
    """Canonical health state values across pipeline layers."""

    OK = "ok"
    DEGRADED = "degraded"
    CRITICAL_DEGRADED = "critical_degraded"
    BLOCKED = "blocked"


def normalize_pipeline_health_state(raw: Any) -> PipelineHealthState:
    """Normalize external/raw state values to canonical enum values."""
    value = str(raw or "").strip().lower()
    for state in PipelineHealthState:
        if value == state.value:
            return state
    return PipelineHealthState.OK


def pipeline_health_from_fail_reasons(fail_reasons: list[dict[str, Any]] | None) -> PipelineHealthState:
    """Derive canonical health state from structured fail_reasons metadata."""
    reasons = list(fail_reasons or [])
    if not reasons:
        return PipelineHealthState.OK

    def _norm(value: Any) -> str:
        return str(value or "").strip().lower()

    severities = {_norm(entry.get("severity")) for entry in reasons if isinstance(entry, dict)}
    if "blocked" in severities:
        return PipelineHealthState.BLOCKED
    if severities & {"critical", "critical_degraded"}:
        return PipelineHealthState.CRITICAL_DEGRADED
    if severities & {"degraded", "warning"}:
        return PipelineHealthState.DEGRADED

    blocked_codes = {"PIPELINE_BLOCKED", "NO_RUNTIME_PATH"}
    critical_codes = {
        "ARC_REGRESSION_ROLLBACK",
        "QUALITY_GATE_ABORT",
        "GOAL_PRIORITY_ABORT",
    }

    normalized_codes = {
        str(entry.get("error_code", "")).strip().upper() for entry in reasons if isinstance(entry, dict)
    }
    if normalized_codes & blocked_codes:
        return PipelineHealthState.BLOCKED
    if normalized_codes & critical_codes:
        return PipelineHealthState.CRITICAL_DEGRADED
    return PipelineHealthState.DEGRADED


def primary_fail_reason_from_fail_reasons(
    fail_reasons: list[dict[str, Any]] | None,
    *,
    default: str = "",
) -> str:
    """Resolve a deterministic primary fail reason from structured fail_reasons.

    Priority per entry: error_code -> exc_msg -> message.
    The first non-empty candidate across entries is used.
    """
    reasons = list(fail_reasons or [])
    for entry in reasons:
        if not isinstance(entry, dict):
            continue
        for key in ("error_code", "exc_msg", "message"):
            value = str(entry.get(key, "") or "").strip()
            if value and value not in {"none", "None"}:
                return value
    fallback = str(default or "").strip()
    if fallback in {"none", "None"}:
        return ""
    return fallback


def resolve_fail_reason(
    *,
    typed_fail_reason: Any = None,
    metadata: dict[str, Any] | None = None,
    stage_notes: dict[str, Any] | None = None,
    fail_reasons: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve fail_reason from typed field, metadata, and stage notes with clear precedence."""
    meta = metadata or {}
    notes = stage_notes or {}
    reason = typed_fail_reason or meta.get("fail_reason", "") or notes.get("fail_reason", "")
    if not reason:
        reason = primary_fail_reason_from_fail_reasons(fail_reasons or meta.get("fail_reasons") or [])
    reason_str = str(reason or "").strip()
    if reason_str in {"", "none", "None"}:
        return ""
    return reason_str
