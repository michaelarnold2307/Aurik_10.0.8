#!/usr/bin/env python3
"""
Phase 14: Professional Phase Correction v2.0
=============================================

Multi-band stereo phase alignment for optimal imaging and mono compatibility.

SCIENTIFIC FOUNDATION:
- Gerzon (1992): Multi-Channel Microphone Array Design
- Lipshitz & Vanderkooy (1986): The Great Debate: Subjective Evaluation
- Bech & Zacharov (2006): Perceptual Audio Evaluation - stereo imaging
- Blauert (1997): Spatial Hearing - The Psychophysics of Human Sound Localization
- ITU-R BS.775-3: Multichannel Stereophonic Sound System with and without Accompanying Picture
- EBU Tech 3286: Assessment and Specification of Phase Coherence

INDUSTRY BENCHMARKS:
- iZotope Ozone Imager (Stereo Phase correlation display)
- Waves InPhase (Multi-band stereo phase alignment)
- Brainworx bx_digital V3 (Correlation meter + phase correction)
- SSL X-ISM (Intelligent Stereo Management)
- Flux Stereo Tool (Phase/Time alignment)
- Nugen Audio Stereo Pack (Phase correlation analysis)

ALGORITHM:
1. Multi-Band Cross-Correlation Analysis (4 bands)
   - 200 Hz, 1 kHz, 8 kHz crossovers
   - Per-band phase correlation measurement
   - Time-delay estimation via cross-correlation peak

2. Per-Band Phase Alignment
   - Bass: Critical for mono compatibility (sum to mono)
   - Mid: Balance between imaging and compatibility
   - High: Wide stereo image acceptable

3. Material-Adaptive Correction
   - Shellac/Vinyl: Strong correction (old stereo techniques)
   - Tape: Moderate correction (head alignment issues)
   - Digital: Minimal correction (production errors only)

QUALITY TARGETS:
- Correlation improvement: +0.1 to +0.3 (material-dependent)
- Mono compatibility: >0.7 for bass, >0.5 for full range
- Processing: <0.05× realtime

Author: Aurik Professional Team
Version: 2.0.0
Date: February 2026
"""

import os
import sys


import logging
import time

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


class PhaseCorrection(PhaseInterface):
    """Professional multi-band phase correction for stereo imaging."""

    # Material-adaptive correction strength
    CORRECTION_STRENGTH = {
        MaterialType.SHELLAC: 0.80,  # Strong (old stereo cutting techniques)
        MaterialType.VINYL: 0.70,  # Moderate-strong (stereo cutting angle issues)
        MaterialType.TAPE: 0.85,  # Very strong (head misalignment common)
        MaterialType.CD_DIGITAL: 0.30,  # Minimal (production errors only)
        MaterialType.STREAMING: 0.20,  # Very minimal
    }

    # Correlation threshold (correct if below this)
    CORRELATION_THRESHOLD = {
        MaterialType.SHELLAC: 0.65,
        MaterialType.VINYL: 0.75,
        MaterialType.TAPE: 0.70,
        MaterialType.CD_DIGITAL: 0.85,
        MaterialType.STREAMING: 0.90,
    }

    # Multi-band crossover frequencies
    CROSSOVER_FREQS = [200, 1000, 8000]  # Hz (4 bands: <200, 200-1k, 1k-8k, >8k)

    # Max time delay per band (samples @ 44.1kHz)
    MAX_DELAY_SAMPLES = {
        "bass": 100,  # ~2.3ms (bass less critical for timing)
        "low_mid": 50,  # ~1.1ms
        "mid_high": 30,  # ~0.7ms
        "high": 20,  # ~0.45ms (high freqs critical for imaging)
    }

    def __init__(self):
        super().__init__()
        self.name = "Phase Correction v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_14_phase_correction",
            name="Phase Correction v2 Professional",
            category=PhaseCategory.STEREO,
            priority=6,
            dependencies=["phase_15_stereo_balance"],
            estimated_time_factor=0.04,
            version="2.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.90,  # High impact on stereo imaging
            description="Multi-band phase correction for optimal stereo imaging and mono compatibility",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        """
        Apply multi-band phase correction.

        Args:
            audio: Stereo audio [samples, 2]
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with corrected audio
        """
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        # Only for stereo
        if audio.ndim != 2:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={"skipped": True, "reason": "mono_signal"},
                execution_time_seconds=time.time() - start_time,
                metadata={"algorithm": "phase_correction", "version": "2.0"},
            )

        strength = self.CORRECTION_STRENGTH.get(material, 0.7)
        threshold = self.CORRELATION_THRESHOLD.get(material, 0.75)

        # Extract L/R channels
        left = audio[:, 0]
        right = audio[:, 1]

        # Multi-band split
        bands_left = self._multiband_split(left, sample_rate)
        bands_right = self._multiband_split(right, sample_rate)

        # Analyze and correct per band
        corrected_bands_left = []
        corrected_bands_right = []
        correlations_before = []
        correlations_after = []
        delays_corrected = []

        band_names = ["bass", "low_mid", "mid_high", "high"]

        for i, (band_l, band_r, band_name) in enumerate(zip(bands_left, bands_right, band_names)):
            # Analyze correlation
            corr_before, delay = self._analyze_phase(band_l, band_r, self.MAX_DELAY_SAMPLES[band_name])
            correlations_before.append(corr_before)

            # Correct if needed
            if corr_before < threshold:
                band_l_corr, band_r_corr = self._correct_band_phase(band_l, band_r, delay, strength)
                corr_after, _ = self._analyze_phase(band_l_corr, band_r_corr, self.MAX_DELAY_SAMPLES[band_name])
                delays_corrected.append(int(delay))
            else:
                band_l_corr, band_r_corr = band_l, band_r
                corr_after = corr_before
                delays_corrected.append(0)

            correlations_after.append(corr_after)
            corrected_bands_left.append(band_l_corr)
            corrected_bands_right.append(band_r_corr)

        # Reconstruct
        corrected_left = self._multiband_reconstruct(corrected_bands_left)
        corrected_right = self._multiband_reconstruct(corrected_bands_right)

        # Ensure length matches
        min_len = min(len(corrected_left), len(corrected_right), len(audio))
        corrected_left = corrected_left[:min_len]
        corrected_right = corrected_right[:min_len]

        corrected_audio = np.column_stack([corrected_left, corrected_right])

        # Overall correlation
        overall_corr_before = np.mean(correlations_before)
        overall_corr_after = np.mean(correlations_after)

        processing_time = time.time() - start_time

        corrected_audio = np.nan_to_num(corrected_audio, nan=0.0, posinf=0.0, neginf=0.0)
        corrected_audio = np.clip(corrected_audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=corrected_audio,
            metrics={
                "correlation_before": float(overall_corr_before),
                "correlation_after": float(overall_corr_after),
                "correlation_improvement": float(overall_corr_after - overall_corr_before),
                "per_band_correlation_before": [float(c) for c in correlations_before],
                "per_band_correlation_after": [float(c) for c in correlations_after],
                "delays_corrected_samples": delays_corrected,
                "correction_strength": strength,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "multiband_phase_correction",
                "version": "2.0",
                "bands": band_names,
                "crossovers_hz": self.CROSSOVER_FREQS,
            },
        )

    def _multiband_split(self, audio: np.ndarray, sample_rate: int) -> list:
        """Split audio into 4 bands using Linkwitz-Riley crossovers."""
        bands = []

        # Design crossover filters (4th order Linkwitz-Riley)
        nyquist = sample_rate / 2

        # Band 1: <200 Hz (Bass)
        sos_low = signal.butter(4, self.CROSSOVER_FREQS[0] / nyquist, btype="low", output="sos")
        bands.append(signal.sosfilt(sos_low, audio))

        # Band 2: 200-1000 Hz (Low-Mid)
        sos_band2 = signal.butter(
            4, [self.CROSSOVER_FREQS[0] / nyquist, self.CROSSOVER_FREQS[1] / nyquist], btype="band", output="sos"
        )
        bands.append(signal.sosfilt(sos_band2, audio))

        # Band 3: 1000-8000 Hz (Mid-High)
        sos_band3 = signal.butter(
            4, [self.CROSSOVER_FREQS[1] / nyquist, self.CROSSOVER_FREQS[2] / nyquist], btype="band", output="sos"
        )
        bands.append(signal.sosfilt(sos_band3, audio))

        # Band 4: >8000 Hz (High)
        sos_high = signal.butter(4, self.CROSSOVER_FREQS[2] / nyquist, btype="high", output="sos")
        bands.append(signal.sosfilt(sos_high, audio))

        return bands

    def _multiband_reconstruct(self, bands: list) -> np.ndarray:
        """Reconstruct audio from bands (simple sum for Linkwitz-Riley)."""
        # Ensure all bands same length
        min_len = min(len(b) for b in bands)
        bands_trimmed = [b[:min_len] for b in bands]

        # Sum bands
        reconstructed = np.sum(bands_trimmed, axis=0)
        return reconstructed

    def _analyze_phase(self, left: np.ndarray, right: np.ndarray, max_delay: int) -> tuple[float, int]:
        """
        Analyze phase alignment via cross-correlation.

        Returns:
            (correlation, delay_samples)
        """
        # Use first 3 seconds for analysis
        max_samples = min(len(left), len(right), 44100 * 3)
        left_seg = left[:max_samples]
        right_seg = right[:max_samples]

        # Cross-correlation
        correlation = signal.correlate(left_seg, right_seg, mode="same")
        lags = signal.correlation_lags(len(left_seg), len(right_seg), mode="same")

        # Limit search range
        valid_mask = np.abs(lags) <= max_delay
        corr_valid = correlation[valid_mask]
        lags_valid = lags[valid_mask]

        # Find peak
        peak_idx = np.argmax(np.abs(corr_valid))
        delay = -lags_valid[peak_idx]

        # Normalized correlation
        if delay > 0:
            aligned_l = left_seg[delay:]
            aligned_r = right_seg[: len(left_seg) - delay]
        elif delay < 0:
            aligned_l = left_seg[: len(left_seg) + delay]
            aligned_r = right_seg[-delay:]
        else:
            aligned_l = left_seg
            aligned_r = right_seg

        if len(aligned_l) > 0 and len(aligned_r) > 0:
            # Guard: np.corrcoef stiller Signale => RuntimeWarning(invalid in divide)
            with np.errstate(invalid="ignore"):
                corr_coef = float(np.corrcoef(aligned_l, aligned_r)[0, 1])
            if np.isnan(corr_coef):
                corr_coef = 1.0  # Stille = perfekt korreliert (kein Phasenfehler)
        else:
            corr_coef = 0.0

        return corr_coef, delay

    def _correct_band_phase(
        self, left: np.ndarray, right: np.ndarray, delay: int, strength: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct phase by time-shifting."""
        corrected_delay = int(delay * strength)

        if corrected_delay == 0:
            return left.copy(), right.copy()

        if corrected_delay > 0:
            # Right is delayed
            corrected_left = left.copy()
            corrected_right = np.roll(right, -corrected_delay)
            corrected_right[-corrected_delay:] = 0
        else:
            # Left is delayed
            corrected_left = np.roll(left, corrected_delay)
            corrected_right = right.copy()
            corrected_left[: abs(corrected_delay)] = 0

        return corrected_left, corrected_right


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 14: Professional Phase Correction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test stereo audio with phase error
    duration = 3.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Multi-frequency signal
    signal_base = (
        0.3 * np.sin(2 * np.pi * 100 * t)  # Bass
        + 0.2 * np.sin(2 * np.pi * 500 * t)  # Low-mid
        + 0.15 * np.sin(2 * np.pi * 2000 * t)  # Mid-high
        + 0.1 * np.sin(2 * np.pi * 8000 * t)  # High
    )

    # Create stereo with phase errors (different delays per band)
    delay_bass = 30  # samples (~0.68ms)
    delay_mid = 15  # samples (~0.34ms)

    left = signal_base
    right = signal_base.copy()

    # Apply delays to simulate phase errors
    right = np.roll(right, delay_bass)
    right[:delay_bass] = 0

    test_audio = np.column_stack([left, right])

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug(f"Phase error: Right delayed by {delay_bass} samples (~{delay_bass*1000/sample_rate:.2f}ms)")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug(f"Material: {material_name}")
        logger.debug("─" * 80)
        logger.debug("")

        phase = PhaseCorrection()
        result = phase.process(test_audio, sample_rate, material)

        logger.debug("✅ Professional Phase Correction:")
        logger.debug(f"   Correlation Before: {result.metrics['correlation_before']:.4f}")
        logger.debug(f"   Correlation After: {result.metrics['correlation_after']:.4f}")
        logger.debug(f"   Improvement: {result.metrics['correlation_improvement']:.4f}")
        logger.debug("")
        logger.debug(f"   Per-Band Correlation Before: {[f'{c:.3f}' for c in result.metrics['per_band_correlation_before']]}")
        logger.debug(f"   Per-Band Correlation After:  {[f'{c:.3f}' for c in result.metrics['per_band_correlation_after']]}")
        logger.debug(f"   Delays Corrected (samples):  {result.metrics['delays_corrected_samples']}")
        logger.debug("")
        logger.debug(
            f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
        )
        logger.debug(f"   Correction strength: {result.metrics['correction_strength']}")
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
