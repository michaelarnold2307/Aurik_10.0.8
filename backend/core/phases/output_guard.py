"""Shared output guard helpers for high-quality phase acceptance checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutputGuardDecision:
    """Decision payload used by high-quality output guards."""

    fallback: bool
    reason: str
    rms_delta_db: float
    stereo_side_ratio: float


def rms(audio: np.ndarray) -> float:
    """Return RMS for mono/stereo audio."""
    x = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def side_rms(audio: np.ndarray) -> float:
    """Return side-channel RMS for stereo audio (handles both (N,2) and (2,N))."""
    if audio.ndim != 2:
        return 0.0
    # (N, 2) samples-first
    if audio.shape[1] == 2 and audio.shape[0] > 2:
        side = 0.5 * (audio[:, 0].astype(np.float32) - audio[:, 1].astype(np.float32))
    # (2, N) channels-first
    elif audio.shape[0] == 2 and audio.shape[1] > 2:
        side = 0.5 * (audio[0].astype(np.float32) - audio[1].astype(np.float32))
    else:
        return 0.0
    return float(np.sqrt(np.mean(side**2) + 1e-12))


def evaluate_output_guard(
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    enabled: bool,
    max_abs_rms_delta_db: float,
    stereo_side_ratio_min: float,
    stereo_side_ratio_max: float,
) -> OutputGuardDecision:
    """Evaluate conservative output guard constraints for phase outputs."""
    rms_delta_db = float(20.0 * np.log10((rms(candidate) + 1e-12) / (rms(original) + 1e-12)))
    side_ratio = 1.0
    # Detect stereo for both (N, 2) samples-first and (2, N) channels-first layouts

    def _is_stereo_2d(arr: np.ndarray) -> bool:
        if arr.ndim != 2:
            return False
        return (arr.shape[1] == 2 and arr.shape[0] > 2) or (arr.shape[0] == 2 and arr.shape[1] > 2)

    is_stereo = _is_stereo_2d(original) and _is_stereo_2d(candidate)
    if is_stereo:
        side_ratio = float((side_rms(candidate) + 1e-12) / (side_rms(original) + 1e-12))

    if not enabled:
        return OutputGuardDecision(False, "disabled", rms_delta_db, side_ratio)

    if abs(rms_delta_db) > float(max_abs_rms_delta_db):
        return OutputGuardDecision(True, "rms_shift", rms_delta_db, side_ratio)

    if is_stereo and not (float(stereo_side_ratio_min) <= side_ratio <= float(stereo_side_ratio_max)):
        return OutputGuardDecision(True, "stereo_side_ratio", rms_delta_db, side_ratio)

    return OutputGuardDecision(False, "ok", rms_delta_db, side_ratio)
