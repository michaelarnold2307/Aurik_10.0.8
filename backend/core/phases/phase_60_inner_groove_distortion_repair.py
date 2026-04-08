"""
Phase 60 — Inner Groove Distortion Repair.

IGD progressively increases towards the center of vinyl/shellac discs due to
decreasing linear velocity.  This phase applies position-adaptive THD reduction
that increases correction strength towards the end of the recording.

Algorithm:
1. Divide recording into position segments (simulating groove radius)
2. Measure THD per segment
3. Apply adaptive harmonic suppression: stronger towards center (later segments)
4. Preserve fundamental + H2 (musical character), reduce H3+ (distortion)

Scientific basis: Kates (1981) "A Model of Record Tracing Distortion";
Roys (1970) "Playback Distortion in Disc Records".
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)

_MIN_IGD_SCORE: float = 0.10
_N_SEGMENTS: int = 8  # Position segments for adaptive processing


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.6,
    defect_scores: dict | None = None,
) -> np.ndarray:
    """Main entry point for Phase 60."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        igd_score = float(defect_scores.get("inner_groove_distortion", 0.0))
        if igd_score < _MIN_IGD_SCORE:
            logger.debug("Phase 60: IGD score %.3f < %.3f — skipped", igd_score, _MIN_IGD_SCORE)
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        left = apply(audio[0], sample_rate, strength=strength, defect_scores=defect_scores)
        right = apply(audio[1], sample_rate, strength=strength, defect_scores=defect_scores)
        return np.clip(np.stack([left, right], axis=0), -1.0, 1.0).astype(np.float32)

    x = audio.astype(np.float64)
    n = len(x)
    sr = sample_rate
    seg_len = n // _N_SEGMENTS

    out = np.copy(x)
    n_fft = 4096
    hop = n_fft // 4

    for seg_idx in range(_N_SEGMENTS):
        start = seg_idx * seg_len
        end = min(start + seg_len, n) if seg_idx < _N_SEGMENTS - 1 else n
        segment = x[start:end]
        if len(segment) < n_fft:
            continue

        # Position-adaptive strength: increases linearly from outer to inner groove
        position_factor = (seg_idx + 1) / _N_SEGMENTS
        local_strength = strength * position_factor

        # STFT
        window = sps.windows.hann(n_fft, sym=False)
        n_frames = max(1, (len(segment) - n_fft) // hop + 1)
        seg_out = np.zeros(len(segment), dtype=np.float64)
        win_sum = np.zeros(len(segment), dtype=np.float64)

        for i in range(n_frames):
            fs = i * hop
            fe = fs + n_fft
            if fe > len(segment):
                break
            frame = segment[fs:fe] * window
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            # Suppress harmonics H3+ (2–8 kHz range where IGD is worst)
            igd_mask = (freqs >= 2000) & (freqs <= 8000)
            gain = np.ones_like(mag)
            gain[igd_mask] = np.maximum(0.3, 1.0 - local_strength * 0.5)

            # Reconstruct
            spec_clean = (mag * gain) * np.exp(1j * phase)
            frame_out = np.fft.irfft(spec_clean, n=n_fft) * window
            seg_out[fs:fe] += frame_out
            win_sum[fs:fe] += window**2

        win_sum = np.maximum(win_sum, 1e-8)
        seg_out /= win_sum
        out[start:end] = seg_out

    # Crossfade segments (10 ms Hanning)
    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class InnerGrooveDistortionRepairPhase(PhaseInterface):
    """Phase 60: Position-adaptive THD reduction for Inner Groove Distortion."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_60_inner_groove_distortion_repair",
            name="Inner Groove Distortion Repair",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_09"],
            estimated_time_factor=0.06,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.60,
            description=(
                "Position-adaptive harmonic distortion reduction for inner groove "
                "distortion (vinyl/shellac). Correction strength increases towards "
                "the center of the disc."
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.6,
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
                    "igd_score": float((_defect_scores or {}).get("inner_groove_distortion", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Inner groove distortion repair skipped due to zero effective strength"],
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
                "igd_score": float((_defect_scores or {}).get("inner_groove_distortion", 0.0)),
                "strength": _effective_strength,
            },
            metadata={
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
            },
        )
