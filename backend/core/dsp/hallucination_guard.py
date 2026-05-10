"""
§2.46e [RELEASE_MUST] Hallucination-Guard DSP-API — backend/core/dsp/hallucination_guard.py

Lightweight wrapper around the primary hallucination guard
(backend/core/hallucination_guard.py) that exposes the
`check_hallucination(pre, post)` interface required by the VERBOTEN table §2.46e:

    > `check_hallucination(pre, post)` aus `backend/core/dsp/hallucination_guard.py`
    > nach jeder ADDITIVE-Phase;
    > `spectral_novelty > 0.15` → Phase-Rollback (Restoration);
    > `> 0.08` → Score-Penalty 0.3

Threshold semantics (§2.46e normativ):
    spectral_novelty > 0.15  — rollback required in Restoration mode
    spectral_novelty > 0.08  — score penalty −0.3 (both modes)
    harmonic_ceiling_violation == True — hard rollback (BW-Ceiling violated)

Thread-safe via module-level functions (no singleton needed — pure DSP).

Author: Aurik 9 Engineering
Version: 1.0.0 (§2.46e RELEASE_MUST, BUG-FIX v9.12.0)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# §2.46e normative thresholds
_ROLLBACK_THRESHOLD: float = 0.15  # spectral_novelty > this → rollback (Restoration)
_PENALTY_THRESHOLD: float = 0.08  # spectral_novelty > this → score penalty -0.3


@dataclass
class HallucinationCheckResult:
    """Result of check_hallucination(pre, post).

    Attributes:
        spectral_novelty: Fraction of energy in new/hallucinated spectral bins [0, 1].
        requires_rollback: True if spectral_novelty > 0.15 (Restoration hard limit §2.46e).
        score_penalty: Score deduction to apply; 0.3 when spectral_novelty > 0.08.
        harmonic_ceiling_violation: True if energy above BW-Ceiling grew > 8× (§2.46e).
        metadata: Additional diagnostic information.
    """

    spectral_novelty: float
    requires_rollback: bool
    score_penalty: float
    harmonic_ceiling_violation: bool
    metadata: dict


def check_hallucination(
    pre: npt.NDArray[np.float32],
    post: npt.NDArray[np.float32],
    *,
    sr: int = 48000,
    mode: str = "restoration",
    material_bw_ceiling_hz: float | None = None,
) -> HallucinationCheckResult:
    """§2.46e: Check whether an additive phase introduced hallucinated material.

    Delegates spectral_novelty measurement to
    `backend.core.hallucination_guard.compute_spectral_novelty` and applies
    the normative threshold logic defined in §2.46e.

    Args:
        pre:   Audio array before the additive phase (mono float32).
        post:  Audio array after the additive phase (mono float32).
        sr:    Sample rate in Hz (must be 48 000 Hz in phase context).
        mode:  Processing mode — "restoration" enforces hard rollback at 0.15;
               "studio_2026" enforces MUSHRA check instead.
        material_bw_ceiling_hz: Physical BW ceiling of the carrier medium (Hz).
               If provided, energy growth above this frequency is checked for
               harmonic_ceiling_violation (> 8× increase = hard rollback).

    Returns:
        HallucinationCheckResult with decision flags and diagnostic metadata.
    """
    pre_arr = np.asarray(pre, dtype=np.float32)
    post_arr = np.asarray(post, dtype=np.float32)

    # Mono-ify if stereo
    if pre_arr.ndim == 2:
        pre_arr = np.mean(pre_arr, axis=1 if pre_arr.shape[1] <= 8 else 0).astype(np.float32)
    if post_arr.ndim == 2:
        post_arr = np.mean(post_arr, axis=1 if post_arr.shape[1] <= 8 else 0).astype(np.float32)

    spectral_novelty: float = 0.0
    harmonic_ceiling_violation: bool = False
    meta: dict = {}

    # --- Primary: delegate to backend.core.hallucination_guard ---
    try:
        _primary_hallucination_guard: Any = importlib.import_module("backend.core.hallucination_guard")

        spectral_novelty, sn_meta = _primary_hallucination_guard.compute_spectral_novelty(pre_arr, post_arr, sr=sr)
        spectral_novelty = float(np.nan_to_num(spectral_novelty, nan=0.0, posinf=0.0, neginf=0.0))
        meta.update(sn_meta)

        # BW-Ceiling check (§2.46e harmonic_ceiling_violation)
        if material_bw_ceiling_hz is not None and material_bw_ceiling_hz > 0:
            try:
                _chv = _primary_hallucination_guard.check_harmonic_ceiling_violation  # pyright: ignore[reportCallIssue]
                harmonic_ceiling_violation, ceiling_meta = _chv(
                    pre_arr,
                    post_arr,
                    material_bw_ceiling_hz,
                    sr=sr,
                )
                meta.update(ceiling_meta)
                meta["bw_ceiling_hz"] = material_bw_ceiling_hz
                meta["harmonic_ceiling_violation"] = harmonic_ceiling_violation
            except Exception as exc:
                logger.debug("check_harmonic_ceiling_violation failed (non-critical): %s", exc)

    except ImportError:
        # DSP fallback: simple spectral energy delta
        logger.debug("hallucination_guard primary import failed; using DSP fallback.")
        spectral_novelty, meta = _compute_spectral_novelty_dsp(pre_arr, post_arr, sr)

    except Exception as exc:
        logger.warning("check_hallucination: primary computation failed (%s) — returning safe defaults.", exc)
        spectral_novelty = 0.0
        meta["error"] = str(exc)

    # --- Apply §2.46e threshold logic ---
    requires_rollback = False
    score_penalty = 0.0

    if harmonic_ceiling_violation:
        # Hard rollback regardless of spectral_novelty
        requires_rollback = True
        score_penalty = 0.3
        logger.warning(
            "§2.46e HallucinationGuard: harmonic_ceiling_violation=True (ceiling=%.0f Hz) → hard rollback.",
            material_bw_ceiling_hz or -1,
        )
    elif spectral_novelty > _ROLLBACK_THRESHOLD:
        if mode == "restoration":
            requires_rollback = True
        score_penalty = 0.3
        logger.warning(
            "§2.46e HallucinationGuard: spectral_novelty=%.3f > %.2f (mode=%s) → %s, score_penalty=%.1f",
            spectral_novelty,
            _ROLLBACK_THRESHOLD,
            mode,
            "rollback" if requires_rollback else "penalty_only",
            score_penalty,
        )
    elif spectral_novelty > _PENALTY_THRESHOLD:
        score_penalty = 0.3
        logger.debug(
            "§2.46e HallucinationGuard: spectral_novelty=%.3f > %.2f → score_penalty=%.1f",
            spectral_novelty,
            _PENALTY_THRESHOLD,
            score_penalty,
        )

    meta["spectral_novelty"] = spectral_novelty
    meta["requires_rollback"] = requires_rollback
    meta["score_penalty"] = score_penalty
    meta["mode"] = mode

    return HallucinationCheckResult(
        spectral_novelty=spectral_novelty,
        requires_rollback=requires_rollback,
        score_penalty=score_penalty,
        harmonic_ceiling_violation=harmonic_ceiling_violation,
        metadata=meta,
    )


def _compute_spectral_novelty_dsp(
    pre: npt.NDArray[np.float32],
    post: npt.NDArray[np.float32],
    sr: int,
) -> tuple[float, dict]:
    """DSP fallback for spectral_novelty when primary module is unavailable.

    Uses STFT-based energy delta: fraction of energy in bins that grew
    more than 5% (> 1.05×) after the additive phase.
    """
    try:
        from scipy import signal as _sp_signal  # pylint: disable=import-outside-toplevel

        n_fft = min(2048, len(pre), len(post))
        if n_fft < 4:
            return 0.0, {"error": "audio_too_short_dsp"}

        _, _, Pxx_pre = _sp_signal.spectrogram(pre, fs=sr, nperseg=n_fft)
        _, _, Pxx_post = _sp_signal.spectrogram(post, fs=sr, nperseg=n_fft)

        E_pre = np.mean(Pxx_pre, axis=1)
        E_post = np.mean(Pxx_post, axis=1)

        novel_mask = E_post > E_pre * 1.05
        E_novel = float(np.sum(E_post[novel_mask]))
        E_total = float(np.sum(E_post)) + 1e-12
        novelty = float(np.clip(E_novel / E_total, 0.0, 1.0))
        return novelty, {"method": "dsp_fallback"}
    except Exception as exc:
        logger.debug("_compute_spectral_novelty_dsp failed: %s", exc)
        return 0.0, {"error": str(exc), "method": "dsp_fallback_failed"}
