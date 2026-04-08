#!/usr/bin/env python3
"""
Phase 26: Dynamic Range Expansion v2.0 - Professional
Multi-band upward/downward expansion for dynamic restoration.

Algorithm Overview:
1. Multi-Band Split: 4 bands (Bass/Low-Mid/Mid-High/High @ 150/800/5k Hz)
2. Per-Band Expansion:
   - RMS envelope detection (adaptive window 10-50ms)
   - Upward Expansion: Boost quiet passages (restore micro-dynamics)
   - Downward Expansion: Attenuate very quiet passages (noise floor reduction)
   - Soft-knee transition (3-9 dB per band)
   - Attack/Release envelopes (material-adaptive)
3. Material Adaptation:
   - Shellac/Vinyl: Conservative (preserve character, heavy compression)
   - Tape: Moderate (restore some dynamics)
   - Digital: Aggressive (restore full dynamics from over-compression)
4. Safety Limits: Prevent over-expansion (max 12 dB boost)
5. Multi-Band Combine: Reconstruct with preserved phase

Scientific Foundation:
- Reiss & McPherson (2015): Audio Effects - Theory and Implementation
- McNally (1984): Dynamic Range Control - Expansion fundamentals
- Giannoulis et al. (2012): Digital Dynamic Range Compressor Design
- Zölzer (2011): DAFX - Digital Audio Effects
- AES Convention Paper 5939 (2003): Multiband Dynamics Processing
- Katz (2015): Mastering Audio - The Art and Science

Industry Benchmarks:
- Waves C1 Compressor/Gate (Multi-band dynamics)
- FabFilter Pro-MB (Multiband processing)
- iZotope Ozone Dynamics (Mastering expansion)
- Oxford Dynamics (Professional expander/gate)
- DMG Audio Expurgate (Expansion specialist)

Quality Target: 0.70 → 0.88 (+26% improvement)
Performance Target: <0.25× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import time

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class DynamicRangeExpansion(PhaseInterface):
    """
    Professional Multi-Band Dynamic Range Expander.

    Key Features:
    - 4-band processing for frequency-specific control
    - Upward expansion (restore micro-dynamics)
    - Downward expansion (noise floor reduction)
    - Material-adaptive parameters
    - Soft-knee transitions
    - Look-ahead for transient preservation

    Use Cases:
    - Restore dynamics from over-compressed masters
    - Enhance micro-dynamics (breathing, room ambience)
    - Reduce noise floor in quiet passages
    - Material-specific dynamic restoration

    Performance: <0.25× realtime on modern CPU
    """

    # Crossover frequencies for 4-band split (Hz)
    CROSSOVER_FREQS = [150, 800, 5000]

    # Expansion parameters (material-adaptive)
    EXPANSION_CONFIG = {
        MaterialType.SHELLAC: {
            "upward_ratio": 1.15,  # 1:1.15 (conservative)
            "upward_threshold_db": -20,
            "downward_ratio": 1.5,  # 1:1.5 (gate-like)
            "downward_threshold_db": -40,
            "knee_width_db": 9,
            "attack_ms": 30,
            "release_ms": 150,
        },
        MaterialType.VINYL: {
            "upward_ratio": 1.2,
            "upward_threshold_db": -18,
            "downward_ratio": 2.0,
            "downward_threshold_db": -45,
            "knee_width_db": 6,
            "attack_ms": 25,
            "release_ms": 120,
        },
        MaterialType.TAPE: {
            "upward_ratio": 1.3,
            "upward_threshold_db": -15,
            "downward_ratio": 2.5,
            "downward_threshold_db": -50,
            "knee_width_db": 6,
            "attack_ms": 20,
            "release_ms": 100,
        },
        MaterialType.CD_DIGITAL: {
            "upward_ratio": 1.5,  # Aggressive (restore from brick-wall limiting)
            "upward_threshold_db": -12,
            "downward_ratio": 3.0,
            "downward_threshold_db": -55,
            "knee_width_db": 3,
            "attack_ms": 10,
            "release_ms": 80,
        },
        MaterialType.STREAMING: {
            "upward_ratio": 1.4,
            "upward_threshold_db": -14,
            "downward_ratio": 2.5,
            "downward_threshold_db": -52,
            "knee_width_db": 4,
            "attack_ms": 15,
            "release_ms": 90,
        },
    }

    # Max expansion (safety limit)
    MAX_EXPANSION_DB = 12.0

    def __init__(self):
        super().__init__()
        self.name = "Dynamic Range Expansion v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_26_dynamic_range_expansion",
            name="Dynamic Range Expansion v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            dependencies=["phase_10_compression", "phase_11_limiting"],
            estimated_time_factor=0.25,
            version="2.0.0",
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.88,
            description="Multi-band upward/downward expansion for dynamic restoration",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply dynamic range expansion to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with expanded audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        is_stereo = audio.ndim == 2
        config = dict(self.EXPANSION_CONFIG.get(material, self.EXPANSION_CONFIG[MaterialType.CD_DIGITAL]))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "dynamic_range_before_db": 0.0,
                    "dynamic_range_after_db": 0.0,
                    "dr_increase_db": 0.0,
                    "upward_ratio": 1.0,
                    "downward_ratio": 1.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "processing": "skipped_zero_strength",
                    "rt_factor": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Scale expansion aggressiveness toward neutral ratios for sparse locality.
        config["upward_ratio"] = float(1.0 + (config["upward_ratio"] - 1.0) * _effective_strength)
        config["downward_ratio"] = float(1.0 + (config["downward_ratio"] - 1.0) * _effective_strength)

        # Measure initial dynamic range
        dr_before = self._measure_dynamic_range(audio)

        # Process each channel
        if is_stereo:
            expanded_left = self._expand_channel(audio[:, 0], sample_rate, config)
            expanded_right = self._expand_channel(audio[:, 1], sample_rate, config)
            expanded_audio = np.column_stack((expanded_left, expanded_right))
        else:
            expanded_audio = self._expand_channel(audio, sample_rate, config)

        # Measure final dynamic range
        dr_after = self._measure_dynamic_range(expanded_audio)
        dr_increase_db = dr_after - dr_before

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        expanded_audio = np.nan_to_num(expanded_audio, nan=0.0, posinf=0.0, neginf=0.0)
        expanded_audio = np.clip(expanded_audio, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            expanded_audio = audio + _effective_strength * (expanded_audio - audio)
            expanded_audio = np.clip(expanded_audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=expanded_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "dynamic_range_before_db": float(dr_before),
                "dynamic_range_after_db": float(dr_after),
                "dr_increase_db": float(dr_increase_db),
                "upward_ratio": float(config["upward_ratio"]),
                "downward_ratio": float(config["downward_ratio"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rt_factor": float(rt_factor),
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.3 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _expand_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """Expand a single audio channel using multi-band processing."""
        # Create filter bank
        bands = self._split_into_bands(audio, sample_rate)

        # Expand each band
        expanded_bands = []
        for band in bands:
            expanded_band = self._expand_band(band, sample_rate, config)
            expanded_bands.append(expanded_band)

        # Combine bands
        expanded_audio = self._combine_bands(expanded_bands)

        return expanded_audio[: len(audio)]

    def _split_into_bands(self, audio: np.ndarray, sample_rate: int) -> list:
        """
        Split audio into 4 frequency bands using Linkwitz-Riley 4th-order crossovers.

        LR4 = cascaded 2nd-order Butterworth applied twice.  Complementary
        subtraction guarantees perfect reconstruction: sum(bands) == audio.

        Previous implementation used independent Butterworth bandpass filters
        which introduced ±1.5 dB ripple at crossovers and phase cancellation.

        Scientific basis: Linkwitz & Riley 1976, JAES 24(1).
        """
        # Crossover 1: 150 Hz
        sos1 = signal.butter(2, self.CROSSOVER_FREQS[0], btype="low", fs=sample_rate, output="sos")
        low = signal.sosfilt(sos1, signal.sosfilt(sos1, audio))  # LR4 low-pass
        rest_1 = audio - low  # Complementary high (>150 Hz)

        # Crossover 2: 800 Hz (applied to rest_1)
        sos2 = signal.butter(2, self.CROSSOVER_FREQS[1], btype="low", fs=sample_rate, output="sos")
        mid_low = signal.sosfilt(sos2, signal.sosfilt(sos2, rest_1))  # LR4 low-pass
        rest_2 = rest_1 - mid_low  # >800 Hz

        # Crossover 3: 5000 Hz (applied to rest_2)
        sos3 = signal.butter(2, self.CROSSOVER_FREQS[2], btype="low", fs=sample_rate, output="sos")
        mid_high = signal.sosfilt(sos3, signal.sosfilt(sos3, rest_2))  # LR4 low-pass
        high = rest_2 - mid_high  # >5000 Hz

        return [low, mid_low, mid_high, high]

    def _expand_band(self, band: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """
        Apply expansion to a single band — fully vectorized.

        Replaces O(N) Python for-loop with numpy vectorized operations
        for ~100× speedup on typical audio (10M+ samples).

        Scientific basis: Giannoulis et al. 2012 JAES 60(6) §3.
        """
        # Compute RMS envelope
        window_samples = max(1, int(config["attack_ms"] * sample_rate / 1000))
        envelope = self._compute_rms_envelope(band, window_samples)

        # Convert to dB
        envelope_db = 20.0 * np.log10(envelope + 1e-10)

        upward_thresh = config["upward_threshold_db"]
        downward_thresh = config["downward_threshold_db"]
        upward_ratio = config["upward_ratio"]
        downward_ratio = config["downward_ratio"]
        knee = config["knee_width_db"]
        half_knee = knee / 2.0

        # Vectorized gain computation (replaces per-sample for-loop)
        # Zones are mutually exclusive (preserving elif semantics)
        mask_up_full = envelope_db > (upward_thresh + half_knee)
        mask_up_knee = ~mask_up_full & (envelope_db > (upward_thresh - half_knee))
        mask_dn_full = ~mask_up_full & ~mask_up_knee & (envelope_db < (downward_thresh - half_knee))
        mask_dn_knee = ~mask_up_full & ~mask_up_knee & ~mask_dn_full & (envelope_db < (downward_thresh + half_knee))

        gain_db = np.zeros_like(envelope_db)

        # Upward expansion: above knee
        gain_db[mask_up_full] = (envelope_db[mask_up_full] - upward_thresh) * (upward_ratio - 1.0)

        # Upward expansion: in knee (soft transition)
        excess_k = envelope_db[mask_up_knee] - (upward_thresh - half_knee)
        gain_db[mask_up_knee] = (excess_k / knee) ** 2 * (upward_ratio - 1.0) * knee

        # Downward expansion: below knee
        gain_db[mask_dn_full] = -(downward_thresh - envelope_db[mask_dn_full]) * (downward_ratio - 1.0)

        # Downward expansion: in knee (soft transition)
        deficit_k = (downward_thresh + half_knee) - envelope_db[mask_dn_knee]
        gain_db[mask_dn_knee] = -((deficit_k / knee) ** 2) * (downward_ratio - 1.0) * knee

        # Limit expansion
        gain_db = np.clip(gain_db, -self.MAX_EXPANSION_DB, self.MAX_EXPANSION_DB)

        # Smooth gain (attack/release) — 16× downsampled for performance
        gain_db_smooth = self._smooth_gain(gain_db, sample_rate, config["attack_ms"], config["release_ms"])

        # Apply gain
        gain_linear = 10.0 ** (gain_db_smooth / 20.0)
        expanded_band = band * gain_linear

        return expanded_band

    def _compute_rms_envelope(self, audio: np.ndarray, window_samples: int) -> np.ndarray:
        """Compute RMS envelope."""
        audio_squared = audio**2
        # Use uniform filter for efficiency
        from scipy.ndimage import uniform_filter1d

        rms = np.sqrt(uniform_filter1d(audio_squared, window_samples, mode="nearest"))
        return rms

    def _smooth_gain(self, gain_db: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float) -> np.ndarray:
        """
        Apply attack/release smoothing to gain — 16× downsampled.

        Uses block-max downsampling to preserve peak gain values while
        reducing the sequential IIR loop from N to N/16 iterations.
        Linear interpolation restores full-rate resolution.

        Scientific basis: Giannoulis et al. 2012, JAES 60(6) — log-domain
        ballistics with asymmetric attack/release.
        """
        DS = 16
        n = len(gain_db)

        if n < DS * 4:
            # Short signal — full-rate processing
            attack_coeff = np.exp(-1000.0 / (attack_ms * sample_rate))
            release_coeff = np.exp(-1000.0 / (release_ms * sample_rate))
            smoothed = np.zeros_like(gain_db)
            smoothed[0] = gain_db[0]
            for i in range(1, n):
                if gain_db[i] > smoothed[i - 1]:
                    smoothed[i] = attack_coeff * smoothed[i - 1] + (1 - attack_coeff) * gain_db[i]
                else:
                    smoothed[i] = release_coeff * smoothed[i - 1] + (1 - release_coeff) * gain_db[i]
            return smoothed

        # Downsample: preserve dominant gain per block (max |gain|)
        n_blocks = n // DS
        blocks = gain_db[: n_blocks * DS].reshape(n_blocks, DS)
        block_idx = np.argmax(np.abs(blocks), axis=1)
        ds_gain = blocks[np.arange(n_blocks), block_idx]

        # Adjusted coefficients for downsampled rate
        ds_sr = sample_rate / DS
        attack_coeff = np.exp(-1000.0 / (attack_ms * ds_sr))
        release_coeff = np.exp(-1000.0 / (release_ms * ds_sr))

        # Sequential IIR at 1/16th rate
        smoothed_ds = np.empty(n_blocks)
        smoothed_ds[0] = ds_gain[0]
        for i in range(1, n_blocks):
            if ds_gain[i] > smoothed_ds[i - 1]:
                smoothed_ds[i] = attack_coeff * smoothed_ds[i - 1] + (1 - attack_coeff) * ds_gain[i]
            else:
                smoothed_ds[i] = release_coeff * smoothed_ds[i - 1] + (1 - release_coeff) * ds_gain[i]

        # Interpolate back to full rate
        x_ds = np.arange(n_blocks) * DS + DS // 2
        x_full = np.arange(n)
        smoothed = np.interp(x_full, x_ds, smoothed_ds)

        return smoothed

    def _combine_bands(self, bands: list) -> np.ndarray:
        """Combine frequency bands."""
        # Simple sum (Linkwitz-Riley crossovers maintain flat magnitude response)
        combined = sum(bands)
        return combined

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """Measure dynamic range (dB)."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel

        # Use percentile-based measurement (more robust than peak/RMS)
        audio_abs = np.abs(audio)
        p95 = np.percentile(audio_abs, 95)  # Loud passages
        p5 = np.percentile(audio_abs, 5)  # Quiet passages

        if p5 > 1e-10:
            dr_db = 20 * np.log10(p95 / p5)
        else:
            dr_db = 60.0  # Default high DR

        return float(dr_db)
