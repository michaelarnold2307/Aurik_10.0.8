"""
Phase 62 — Crosstalk Cancellation.

Channel crosstalk in early stereo recordings where channel separation was
limited (< 20 dB).  Implements the analytically exact inverse of the Vinyl
groove constant-α crosstalk mixing model (Blauert 1997; IEC 60098):

    L_played = L_cut + α(f) · R_cut
    R_played = R_cut + α(f) · L_cut

Inverse (assuming α(f) uniform per frame):
    det = 1 − α²
    L_clean = (L_played − α · R_played) / det
    R_clean = (R_played − α · L_played) / det

α(f) is frequency-dependent: typically −25 to −30 dB at low frequencies,
increasing toward HF (worse channel separation above 5 kHz).

Algorithm:
1. Compute long-term average auto/cross-spectra from full signal
2. Estimate α(f): |S_LR(f)| / sqrt(S_LL(f) · S_RR(f)), capped at 0.70
3. Apply per-frame analytical inverse matrix in STFT domain
4. Wet/dry blend controlled by strength

Scientific basis: Blauert (1997) "Spatial Hearing", §3.1 Channel Separation;
IEC 60098:1987 Groove dimensions; Avendano & Jot (2002) "Frequency-Domain
Crosstalk Separation".
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)

_MIN_CROSSTALK_SCORE: float = 0.10
# Maximum α to prevent over-cancellation (α > 0.70 → |det| < 0.51 → unstable)
_ALPHA_MAX: float = 0.70
# Estimation window: 4096 for frequency resolution at 48 kHz (≈11.7 Hz/bin)
_ESTIM_NFFT: int = 4096
# Processing STFT
_PROC_NFFT: int = 4096
_PROC_HOP: int = _PROC_NFFT // 4


def _estimate_alpha_f(
    left: np.ndarray,
    right: np.ndarray,
    n_fft: int,
    hop: int,
) -> np.ndarray:
    """Estimate frequency-dependent crosstalk coefficient α(f) via long-term spectra.

    Uses the coherence between L and R as the magnitude estimate for α, which is
    the correct estimator for the model  L = S + αR, R = S' + αL (Avendano & Jot 2002).

    Returns α(f) array of length n_fft//2 + 1, dtype float64, clipped to _ALPHA_MAX.
    """
    window = sps.windows.hann(n_fft, sym=False)
    n = len(left)
    n_frames = max(1, (n - n_fft) // hop + 1)

    sum_ll = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    sum_rr = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    sum_lr_r = np.zeros(n_fft // 2 + 1, dtype=np.float64)  # Real part of S_LR
    sum_lr_i = np.zeros(n_fft // 2 + 1, dtype=np.float64)  # Imag part of S_LR

    for i in range(n_frames):
        s = i * hop
        e = s + n_fft
        if e > n:
            break
        fl = np.fft.rfft(left[s:e] * window)
        fr = np.fft.rfft(right[s:e] * window)
        sum_ll += np.abs(fl) ** 2
        sum_rr += np.abs(fr) ** 2
        cross = fl * np.conj(fr)
        sum_lr_r += cross.real
        sum_lr_i += cross.imag

    # α(f) = |S_LR(f)| / sqrt(S_LL(f) · S_RR(f))  — coherence magnitude
    denom = np.sqrt(np.maximum(sum_ll * sum_rr, 1e-30))
    alpha_f = np.sqrt(sum_lr_r**2 + sum_lr_i**2) / denom
    # Hard cap to guarantee invertibility and prevent over-correction
    return np.clip(alpha_f, 0.0, _ALPHA_MAX)


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    defect_scores: dict | None = None,
) -> np.ndarray:
    """Main entry point for Phase 62."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        xt_score = float(defect_scores.get("crosstalk", 0.0))
        if xt_score < _MIN_CROSSTALK_SCORE:
            logger.debug("Phase 62: crosstalk score %.3f < %.3f — skipped", xt_score, _MIN_CROSSTALK_SCORE)
            return np.clip(audio, -1.0, 1.0)

    # Crosstalk cancellation only applies to stereo
    if audio.ndim != 2:
        logger.debug("Phase 62: mono input — skipped (no crosstalk possible)")
        return np.clip(audio, -1.0, 1.0)

    # Normalise to [channels, samples] = (2, N)
    if audio.shape[0] == 2 and audio.shape[1] != 2:
        left = audio[0].astype(np.float64)
        right = audio[1].astype(np.float64)
        channels_first = True
    else:
        # samples-first (N, 2)
        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)
        channels_first = False

    n = len(left)
    window = sps.windows.hann(_PROC_NFFT, sym=False)

    # ── Step 1: estimate α(f) from full signal ─────────────────────────────
    alpha_f = _estimate_alpha_f(left, right, _ESTIM_NFFT, _ESTIM_NFFT // 4)
    # Interpolate to processing NFFT bins (ESTIM and PROC have same n_fft here)
    # Scale alpha by strength so we only invert the fraction the user requests
    alpha_applied = alpha_f * float(np.clip(strength, 0.0, 1.0))

    # Precompute per-bin denominator 1/(1 - α²)
    det_inv = 1.0 / np.maximum(1.0 - alpha_applied**2, 1e-6)

    # ── Step 2: apply per-frame analytical inverse ─────────────────────────
    n_frames = max(1, (n - _PROC_NFFT) // _PROC_HOP + 1)
    left_out = np.zeros(n, dtype=np.float64)
    right_out = np.zeros(n, dtype=np.float64)
    win_sum = np.zeros(n, dtype=np.float64)

    for i in range(n_frames):
        s = i * _PROC_HOP
        e = s + _PROC_NFFT
        if e > n:
            break

        fl = np.fft.rfft(left[s:e] * window)
        fr = np.fft.rfft(right[s:e] * window)

        # Analytical inverse of the crosstalk mixing matrix
        #   L_clean = (L_played − α · R_played) / det
        #   R_clean = (R_played − α · L_played) / det
        fl_clean = (fl - alpha_applied * fr) * det_inv
        fr_clean = (fr - alpha_applied * fl) * det_inv

        frame_l = np.fft.irfft(fl_clean, n=_PROC_NFFT) * window
        frame_r = np.fft.irfft(fr_clean, n=_PROC_NFFT) * window

        left_out[s:e] += frame_l
        right_out[s:e] += frame_r
        win_sum[s:e] += window**2

    win_sum = np.maximum(win_sum, 1e-8)
    left_out /= win_sum
    right_out /= win_sum

    left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
    right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)

    if channels_first:
        result = np.stack([left_out, right_out], axis=0)
    else:
        result = np.stack([left_out, right_out], axis=1)

    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class CrosstalkCancellationPhase(PhaseInterface):
    """Phase 62: Analytical crosstalk cancellation for early stereo recordings.

    Implements the exact inverse of the vinyl groove α-mixing model rather than
    coherence-thresholded BSS, yielding predictable separation without coherence
    artefacts.
    """

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_62_crosstalk_cancellation",
            name="Crosstalk Cancellation",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_14"],
            estimated_time_factor=0.05,
            version="2.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.65,
            description=(
                "Analytical inverse of the vinyl-groove constant-α crosstalk "
                "mixing model. Estimates α(f) from long-term L/R coherence, "
                "then applies the exact 2×2 matrix inverse per frequency bin."
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.5,
        defect_scores: dict | None = None,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "crosstalk_score": float((_defect_scores or {}).get("crosstalk", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Crosstalk cancellation skipped due to zero effective strength"],
            )
        _rms_in = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))
        result_audio = apply(audio, sample_rate, strength=_effective_strength, defect_scores=_defect_scores)
        elapsed = _time.perf_counter() - t0

        _rms_out = float(np.sqrt(np.mean(np.asarray(result_audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_drop = 20.0 * np.log10(max(_rms_out / _rms_in, 1e-30)) if _rms_in > 1e-8 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "crosstalk_score": float((_defect_scores or {}).get("crosstalk", 0.0)),
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )


# EOF
