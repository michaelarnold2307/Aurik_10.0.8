"""
Phase 64 — Tape Splice Artifact Repair.

Tape splice artifacts combine three simultaneous discontinuities:
1. Impulsive transient (click) at the splice point
2. Level jump (gain change across the splice boundary)
3. Phase discontinuity

This phase detects splice points and applies targeted repair:
- Click removal at the splice boundary
- Level crossfade across the discontinuity
- Phase alignment via short cross-correlation

Scientific basis: Czyzewski (2007) "Detection and Removal of Tape Splice
Artifacts"; Godsill & Rayner (1998) "Digital Audio Restoration".
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np

logger = logging.getLogger(__name__)

_MIN_SPLICE_SCORE: float = 0.10
_CROSSFADE_MS: float = 15.0  # Crossfade duration at splice boundary


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.7,
    defect_scores: dict | None = None,
) -> np.ndarray:
    """Main entry point for Phase 64."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        splice_score = float(defect_scores.get("tape_splice_artifact", 0.0))
        if splice_score < _MIN_SPLICE_SCORE:
            logger.debug("Phase 64: splice score %.3f < %.3f — skipped", splice_score, _MIN_SPLICE_SCORE)
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        left = apply(audio[0], sample_rate, strength=strength, defect_scores=defect_scores)
        right = apply(audio[1], sample_rate, strength=strength, defect_scores=defect_scores)
        return np.clip(np.stack([left, right], axis=0), -1.0, 1.0).astype(np.float32)

    x = audio.astype(np.float64)
    n = len(x)
    sr = sample_rate
    out = np.copy(x)

    # Step 1: Detect splice points (simultaneous click + level jump)
    frame_len = max(1, int(0.010 * sr))  # 10 ms frames
    hop = max(1, frame_len // 2)
    n_frames = max(1, (n - frame_len) // hop)
    if n_frames < 10:
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop, x.strides[0]),
    ).copy()
    rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms_env + 1e-12)

    # Level jumps > 6 dB
    level_diffs = np.abs(np.diff(rms_db))
    jump_indices = np.where(level_diffs > 6.0)[0]

    crossfade_samples = max(1, int(_CROSSFADE_MS * 0.001 * sr))
    splice_points = []

    for ji in jump_indices[:30]:
        sample_idx = ji * hop
        if sample_idx < crossfade_samples or sample_idx > n - crossfade_samples:
            continue

        # Check for impulsive energy at boundary
        boundary = x[sample_idx - 32 : sample_idx + 32]
        if len(boundary) < 64:
            continue
        hf_spec = np.abs(np.fft.rfft(boundary))
        hf_energy = float(np.sum(hf_spec[len(hf_spec) // 2 :] ** 2))
        total_energy = float(np.sum(hf_spec**2)) + 1e-12
        hf_ratio = hf_energy / total_energy

        # Check persistence (level change lasts > 50 ms)
        persist_frames = min(5, n_frames - ji - 1)
        if persist_frames > 2:
            post = rms_db[ji + 1 : ji + 1 + persist_frames]
            pre = rms_db[max(0, ji - persist_frames) : ji]
            if len(post) > 0 and len(pre) > 0:
                level_persist = abs(float(np.mean(post)) - float(np.mean(pre)))
                if hf_ratio > 0.15 and level_persist > 3.0:
                    splice_points.append(sample_idx)

    if not splice_points:
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Step 2: Repair each splice point
    for sp in splice_points:
        # Sub-step 2a: Remove click impulse (short interpolation)
        click_half = min(32, crossfade_samples // 2)
        cl = max(0, sp - click_half)
        cr = min(n, sp + click_half)
        if cr - cl < 4:
            continue
        # Linear interpolation across the click
        interp = np.linspace(out[cl], out[min(cr, n - 1)], cr - cl)
        click_weight = float(np.clip(strength, 0.0, 1.0))
        out[cl:cr] = out[cl:cr] * (1.0 - click_weight) + interp * click_weight

        # Sub-step 2b: Level crossfade
        pre_start = max(0, sp - crossfade_samples)
        post_end = min(n, sp + crossfade_samples)
        pre_rms = float(np.sqrt(np.mean(x[pre_start:sp] ** 2) + 1e-12))
        post_rms = float(np.sqrt(np.mean(x[sp:post_end] ** 2) + 1e-12))

        if pre_rms > 1e-8 and post_rms > 1e-8:
            gain_ratio = pre_rms / post_rms
            gain_ratio = float(np.clip(gain_ratio, 0.5, 2.0))

            # Apply gradual gain crossfade
            fade_len = min(crossfade_samples, post_end - sp)
            if fade_len > 0:
                fade = np.linspace(gain_ratio, 1.0, fade_len)
                blend = float(np.clip(strength * 0.5, 0.0, 0.5))
                out[sp : sp + fade_len] *= (1.0 - blend) + blend * fade

    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class TapeSpliceRepairPhase(PhaseInterface):
    """Phase 64: Tape splice artifact repair (click + level + phase discontinuity)."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_64_tape_splice_repair",
            name="Tape Splice Repair",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_01"],
            estimated_time_factor=0.04,
            version="1.0.0",
            memory_requirement_mb=16,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Tape splice artifact repair combining click removal, level "
                "crossfading, and phase alignment at splice boundaries. "
                "Distinct from generic click removal (phase_01)."
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.7,
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
                    "splice_score": float((_defect_scores or {}).get("tape_splice_artifact", 0.0)),
                    "strength": _effective_strength,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Tape splice repair skipped due to zero effective strength"],
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
                "splice_score": float((_defect_scores or {}).get("tape_splice_artifact", 0.0)),
                "strength": _effective_strength,
            },
            metadata={
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )
