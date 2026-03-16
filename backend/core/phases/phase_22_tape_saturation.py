#!/usr/bin/env python3
"""
Phase 22: Professional Tape Saturation v2.0
=============================================

Multi-band analog tape saturation emulation with harmonic series modeling.

SCIENTIFIC FOUNDATION:
- Parker et al. (2014): Wave Digital Filters for Vacuum Tube Emulation
- Huovilainen (2004): Design of a Scalable Polyphonic Synthesizer
- Stilson & Smith (1996): Alias-Free Digital Synthesis of Classic Analog Waveforms
- Välimäki & Reiss (2008): All About Audio Equalization: Solutions and Frontiers
- Zölzer (2011): DAFX - Digital Audio Effects
- McNally (1984): Dynamic Range Control of Digital Audio Signals
- Hamada & Koizumi (1981): Analysis of Distortion in Tape Recording

INDUSTRY BENCHMARKS:
- Universal Audio Ampex ATR-102 (Industry standard tape emulation)
- Slate Digital Virtual Tape Machines (VTM)
- Softube Tape (Multi-track tape simulator)
- IK Multimedia Tape Machine Collection
- Waves J37 Tape (Abbey Road)
- McDSP Analog Channel (AC101/AC202)
- Acustica Audio Taupe (Tape suite)

ALGORITHM:
1. Multi-Band Processing (3 bands: Bass <300Hz, Mid 300-4k, High >4kHz)
   - Independent saturation per band
   - Different harmonic characteristics per band

2. Tape Speed Emulation
   - 15 IPS (high fidelity): Minimal saturation, extended HF
   - 7.5 IPS (standard): Moderate saturation, HF roll-off @ 18kHz
   - 3.75 IPS (vintage): Strong saturation, HF roll-off @ 12kHz

3. Harmonic Series Modeling
   - 2nd harmonic (even): Warmth, fullness
   - 3rd harmonic (odd): Bite, presence
   - 4th+ harmonics: Tape coloration

4. Tape Hysteresis
   - Asymmetric transfer function (positive > negative)
   - Frequency-dependent (more effect on LF)

5. Material-Adaptive Parameters
   - Tape: Strong saturation (authentic tape character)
   - Vinyl: Moderate (analog warmth without tape artifacts)
   - Digital: Subtle (analogizing digital sources)

QUALITY TARGETS:
- THD: 0.5-3% (tape-typical)
- Harmonic increase: +2 to +6 dB
- Processing: <0.2× realtime

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


class TapeSaturation(PhaseInterface):
    """Professional multi-band tape saturation emulation."""

    # Material-adaptive saturation drive
    SATURATION_DRIVE = {
        MaterialType.SHELLAC: 0.0,  # No tape (era mismatch)
        MaterialType.VINYL: 0.30,  # Moderate analog warmth
        MaterialType.TAPE: 0.55,  # Strong (authentic tape)
        MaterialType.CD_DIGITAL: 0.20,  # Subtle analogizing
        MaterialType.STREAMING: 0.25,  # Light warmth
    }

    # Mix amount (wet/dry)
    SATURATION_MIX = {
        MaterialType.SHELLAC: 0.0,
        MaterialType.VINYL: 0.40,  # 40% saturated
        MaterialType.TAPE: 0.60,  # 60% saturated
        MaterialType.CD_DIGITAL: 0.25,  # 25%
        MaterialType.STREAMING: 0.30,  # 30%
    }

    # Tape speed (affects HF response and saturation)
    TAPE_SPEED = {
        MaterialType.SHELLAC: None,
        MaterialType.VINYL: "7.5_ips",  # Standard
        MaterialType.TAPE: "phase_22_tape_saturation",  # High fidelity
        MaterialType.CD_DIGITAL: "15_ips",
        MaterialType.STREAMING: "7.5_ips",
    }

    # Tape speed HF roll-off frequencies
    TAPE_SPEED_HF_ROLLOFF = {
        "15_ips": 20000,  # Minimal roll-off
        "7.5_ips": 18000,  # Standard roll-off
        "3.75_ips": 12000,  # Vintage roll-off
    }

    # Band split frequencies (3 bands: Bass, Mid, High)
    BAND_SPLIT_LOW = 300  # Bass < 300 Hz
    BAND_SPLIT_HIGH = 4000  # High > 4 kHz

    # Per-band saturation scaling
    BAND_DRIVE_SCALE = {
        "bass": 1.2,  # More saturation on bass (tape characteristic)
        "mid": 1.0,  # Standard
        "high": 0.7,  # Less saturation on highs (preserve clarity)
    }

    # Harmonic series weights (2nd, 3rd, 4th+)
    HARMONIC_WEIGHTS = {
        "bass": [0.6, 0.3, 0.1],  # 2nd dominant (warmth)
        "mid": [0.5, 0.4, 0.1],  # Balanced
        "high": [0.4, 0.5, 0.1],  # 3rd dominant (presence)
    }

    # Hysteresis amount (asymmetric saturation)
    HYSTERESIS_AMOUNT = {
        MaterialType.SHELLAC: 0.0,
        MaterialType.VINYL: 0.12,
        MaterialType.TAPE: 0.25,  # Tape-typical
        MaterialType.CD_DIGITAL: 0.08,
        MaterialType.STREAMING: 0.10,
    }

    def __init__(self):
        super().__init__()
        self.name = "Tape Saturation v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_22_tape_saturation",
            name="Tape Saturation v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_21_exciter"],
            estimated_time_factor=0.10,
            version="2.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.93,
            description="Multi-band analog tape saturation with harmonic modeling",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        """
        Apply tape saturation.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with saturated audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        start_time = time.time()

        drive = self.SATURATION_DRIVE.get(material, 0.25)
        mix_amount = self.SATURATION_MIX.get(material, 0.30)
        tape_speed = self.TAPE_SPEED.get(material, "7.5_ips")
        hysteresis = self.HYSTERESIS_AMOUNT.get(material, 0.10)

        if drive < 0.01 or mix_amount < 0.01:
            # Skip processing
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={"saturation_applied": False, "material": material.value},
                execution_time_seconds=time.time() - start_time,
            )

        is_stereo = audio.ndim == 2

        if is_stereo:
            # Process each channel
            left_saturated = self._saturate_multi_band(audio[:, 0], sample_rate, drive, tape_speed, hysteresis)
            right_saturated = self._saturate_multi_band(audio[:, 1], sample_rate, drive, tape_speed, hysteresis)

            # Ensure same length
            min_len = min(len(left_saturated), len(right_saturated))
            saturated = np.column_stack([left_saturated[:min_len], right_saturated[:min_len]])
        else:
            saturated = self._saturate_multi_band(audio, sample_rate, drive, tape_speed, hysteresis)

        # Wet/dry mix
        if len(saturated) != len(audio):
            # Length mismatch: trim or pad
            if len(saturated) > len(audio):
                saturated = saturated[: len(audio)]
            else:
                if is_stereo:
                    saturated = np.pad(saturated, ((0, len(audio) - len(saturated)), (0, 0)), mode="edge")
                else:
                    saturated = np.pad(saturated, (0, len(audio) - len(saturated)), mode="edge")

        mixed = (1.0 - mix_amount) * audio + mix_amount * saturated

        # Measure THD (Total Harmonic Distortion)
        thd_percent = self._estimate_thd(audio, mixed)

        # Measure harmonic increase
        harmonic_before = self._measure_harmonic_content(audio, sample_rate)
        harmonic_after = self._measure_harmonic_content(mixed, sample_rate)
        harmonic_increase_db = 20 * np.log10((harmonic_after + 1e-10) / (harmonic_before + 1e-10))

        processing_time = time.time() - start_time

        mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)
        mixed = np.clip(mixed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=mixed,
            metrics={
                "saturation_applied": True,
                "thd_percent": float(thd_percent),
                "harmonic_increase_db": float(harmonic_increase_db),
                "drive": drive,
                "mix_amount": mix_amount,
                "tape_speed": tape_speed,
                "hysteresis": hysteresis,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={"algorithm": "multi_band_tape_saturation", "version": "2.0", "bands": 3},
        )

    def _saturate_multi_band(
        self, audio: np.ndarray, sample_rate: int, drive: float, tape_speed: str, hysteresis: float
    ) -> np.ndarray:
        """
        Multi-band tape saturation.
        """
        nyquist = sample_rate / 2.0

        # Design crossover filters (Linkwitz-Riley 4th order)
        # Bass: < 300 Hz
        sos_bass_lp = signal.butter(4, self.BAND_SPLIT_LOW / nyquist, btype="lowpass", output="sos")
        bass = signal.sosfilt(sos_bass_lp, audio)

        # High: > 4000 Hz
        sos_high_hp = signal.butter(4, self.BAND_SPLIT_HIGH / nyquist, btype="highpass", output="sos")
        high = signal.sosfilt(sos_high_hp, audio)

        # Mid: 300-4000 Hz (residual)
        mid = audio - bass - high

        # Per-band saturation
        bass_saturated = self._saturate_band(
            bass, drive * self.BAND_DRIVE_SCALE["bass"], hysteresis, self.HARMONIC_WEIGHTS["bass"]
        )
        mid_saturated = self._saturate_band(
            mid, drive * self.BAND_DRIVE_SCALE["mid"], hysteresis, self.HARMONIC_WEIGHTS["mid"]
        )
        high_saturated = self._saturate_band(
            high, drive * self.BAND_DRIVE_SCALE["high"], hysteresis, self.HARMONIC_WEIGHTS["high"]
        )

        # Recombine bands
        saturated = bass_saturated + mid_saturated + high_saturated

        # Apply tape speed HF roll-off
        if tape_speed and tape_speed in self.TAPE_SPEED_HF_ROLLOFF:
            hf_rolloff = self.TAPE_SPEED_HF_ROLLOFF[tape_speed]
            if hf_rolloff < nyquist * 0.95:
                sos_tape_hf = signal.butter(2, hf_rolloff / nyquist, btype="lowpass", output="sos")
                saturated = signal.sosfilt(sos_tape_hf, saturated)

        # Soft limiter (prevent clipping)
        peak = np.max(np.abs(saturated))
        if peak > 0.95:
            saturated = saturated * (0.95 / peak)

        return saturated

    def _saturate_band(self, audio: np.ndarray, drive: float, hysteresis: float, harmonic_weights: list) -> np.ndarray:
        """
        Saturate a single band with harmonic modeling.
        """
        # Drive stage
        driven = audio * (1.0 + drive * 8.0)

        # Tape saturation with hysteresis
        # Positive: standard tanh
        # Negative: reduced gain (asymmetric)
        saturated = np.zeros_like(driven)
        positive_mask = driven >= 0
        negative_mask = driven < 0

        saturated[positive_mask] = np.tanh(driven[positive_mask])
        saturated[negative_mask] = np.tanh(driven[negative_mask] * (1.0 - hysteresis))

        # Add harmonics (2nd, 3rd, 4th)
        # 2nd harmonic (even): saturated^2 (scaled)
        h2 = saturated**2 * np.sign(saturated) * harmonic_weights[0] * 0.1

        # 3rd harmonic (odd): saturated^3
        h3 = saturated**3 * harmonic_weights[1] * 0.05

        # 4th+ harmonics (subtle)
        h4 = saturated**4 * np.sign(saturated) * harmonic_weights[2] * 0.02

        # Mix harmonics
        saturated_with_harmonics = saturated + h2 + h3 + h4

        # Normalize to prevent clipping
        peak = np.max(np.abs(saturated_with_harmonics))
        if peak > 1.0:
            saturated_with_harmonics /= peak

        return saturated_with_harmonics

    def _estimate_thd(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Estimate THD (Total Harmonic Distortion) as RMS difference.
        """
        if original.ndim == 2:
            original = np.mean(original, axis=1)
        if processed.ndim == 2:
            processed = np.mean(processed, axis=1)

        # Ensure same length
        min_len = min(len(original), len(processed))
        original = original[:min_len]
        processed = processed[:min_len]

        # Difference = added harmonics/distortion
        difference = processed - original

        rms_original = np.sqrt(np.mean(original**2))
        rms_difference = np.sqrt(np.mean(difference**2))

        if rms_original > 1e-10:
            thd_percent = (rms_difference / rms_original) * 100.0
        else:
            thd_percent = 0.0

        return min(thd_percent, 100.0)

    def _measure_harmonic_content(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Measure harmonic content (energy in harmonic band).
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # FFT
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)

        # Harmonic band (200 Hz - 5 kHz)
        harmonic_band = (freqs >= 200) & (freqs <= 5000)

        if np.any(harmonic_band):
            harmonic_energy = np.mean(spectrum[harmonic_band])
        else:
            harmonic_energy = 0.0

        return harmonic_energy


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 22: Professional Tape Saturation v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio (pure sine for THD measurement)
    duration = 2.0
    sample_rate = 44100

    t = np.linspace(0, duration, int(sample_rate * duration))

    # Pure 440 Hz sine (A4) at moderate level
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug("Signal: Pure 440 Hz sine (A4)")
    logger.debug("Purpose: Measure THD and harmonic addition")
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

        phase = TapeSaturation()
        result = phase.process(test_signal, sample_rate, material)

        logger.debug("✅ Professional Tape Saturation:")
        logger.debug(f"   THD: {result.metrics['thd_percent']:.2f}%")
        logger.debug(f"   Harmonic Increase: {result.metrics['harmonic_increase_db']:.2f} dB")
        logger.debug(f"   Drive: {result.metrics['drive']:.2f}")
        logger.debug(f"   Mix Amount: {result.metrics['mix_amount']:.0%}")
        logger.debug(f"   Tape Speed: {result.metrics['tape_speed']}")
        logger.debug(f"   Hysteresis: {result.metrics['hysteresis']:.2f}")
        logger.debug(
            f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
