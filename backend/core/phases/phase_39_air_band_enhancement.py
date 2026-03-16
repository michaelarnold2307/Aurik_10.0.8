#!/usr/bin/env python3
"""
Phase 39: Air Band Enhancement v2.0 - Professional
High-frequency shimmer and "air" enhancement (12-20 kHz).

Algorithm Overview:
1. Frequency Focus: 12-20 kHz (air band)
2. Harmonic Excitation: Generate missing HF content
3. Shelving EQ: Smooth HF lift
4. Saturation: Subtle harmonic distortion for warmth
5. Material Adaptation:
   - Shellac: Strong (restore bandwidth-limited treble)
   - Vinyl: Moderate (add air and sparkle)
   - Tape: Light (tape often has natural HF roll-off)
   - Digital: Moderate (add analog-style air)

Scientific Foundation:
- Fastl & Zwicker (2007): Psychoacoustics - Facts and Models
- Gabrielsson & Sjögren (1979): Perceived Sound Quality of Sound-Reproducing Systems
- Toole (1986): Loudspeaker Measurements and Their Relationship to Listener Preferences
- Moore (2012): An Introduction to the Psychology of Hearing

Industry Benchmarks:
- Maag EQ4 (Famous "Air Band" @ 40 kHz downsampled)
- Aphex Aural Exciter (HF harmonic generator)
- BBE Sonic Maximizer (Phase compensation + HF boost)
- Pultec HLF-3C (High-frequency boost)
- Dangerous Music BAX EQ (Shelf filter mastering)

Quality Target: 0.82 → 0.93 (+13% improvement)
Performance Target: <0.08× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
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


class AirBandEnhancement(PhaseInterface):
    """
    Professional Air Band Enhancement Engine.

    Key Features:
    - High-frequency shelving (12-20 kHz)
    - Harmonic excitation (add sparkle)
    - Material-adaptive intensity
    - Psychoacoustic optimization
    - Analog-style warmth

    Use Cases:
    - Restore missing treble from bandwidth-limited sources
    - Add "air" and "shimmer" to vocals and instruments
    - Enhance perceived detail and clarity
    - Modernize vintage recordings

    Performance: <0.08× realtime on modern CPU
    """

    # Air band frequency range
    AIR_BAND_HZ = (12000, 20000)

    # Enhancement parameters (material-adaptive)
    AIR_CONFIG = {
        MaterialType.SHELLAC: {
            "shelf_gain_db": 6.0,  # Strong (restore missing HF)
            "shelf_freq_hz": 10000,
            "exciter_mix": 0.40,
            "saturation_drive": 0.30,
        },
        MaterialType.VINYL: {
            "shelf_gain_db": 4.0,
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.30,
            "saturation_drive": 0.20,
        },
        MaterialType.TAPE: {
            "shelf_gain_db": 3.0,
            "shelf_freq_hz": 13000,
            "exciter_mix": 0.20,
            "saturation_drive": 0.15,
        },
        MaterialType.CD_DIGITAL: {
            "shelf_gain_db": 3.5,
            "shelf_freq_hz": 12000,
            "exciter_mix": 0.25,
            "saturation_drive": 0.25,
        },
        MaterialType.STREAMING: {
            "shelf_gain_db": 4.0,
            "shelf_freq_hz": 11000,
            "exciter_mix": 0.30,
            "saturation_drive": 0.20,
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Air Band Enhancement v2 Professional"
        self._sos_air_cache: dict[int, np.ndarray] = {}
        self._shelf_coeffs: dict[tuple, tuple] = {}

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_39_air_band_enhancement",
            name="Air Band Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=[],
            estimated_time_factor=0.08,
            version="2.0.0",
            memory_requirement_mb=35,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.93,
            description="High-frequency shimmer and air enhancement (12-20 kHz)",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply air band enhancement to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        is_stereo = audio.ndim == 2
        config = self.AIR_CONFIG.get(material, self.AIR_CONFIG[MaterialType.CD_DIGITAL])

        # Measure initial HF energy
        hf_energy_before = self._measure_hf_energy(audio, sample_rate)

        # Process each channel
        if is_stereo:
            enhanced_left = self._enhance_channel(audio[:, 0], sample_rate, config)
            enhanced_right = self._enhance_channel(audio[:, 1], sample_rate, config)
            enhanced_audio = np.column_stack((enhanced_left, enhanced_right))
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        # Measure final HF energy
        hf_energy_after = self._measure_hf_energy(enhanced_audio, sample_rate)
        hf_boost_db = 20 * np.log10((hf_energy_after + 1e-10) / (hf_energy_before + 1e-10))

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        # ── HF-Kumulativ-Limit (Spec §8.2: Presence + Air kumulativ ≤ +4 dB) ──
        # Listening-Fatigue-Schutz: Gesamtanhebung 2–20 kHz limitieren
        hf_cumul_db = float(kwargs.get("hf_cumulative_gain_db", 0.0))
        MAX_HF_CUMUL_DB = 4.0
        if hf_cumul_db > MAX_HF_CUMUL_DB:
            logger.warning(
                "Phase 39: HF-Kumulativ-Limit erreicht (%.1f dB > %.1f dB) — "
                "Air-Band-Gain reduziert (Listening-Fatigue-Schutz, Spec §8.2)",
                hf_cumul_db,
                MAX_HF_CUMUL_DB,
            )
            # Gain-Korrektur: Überschuss rückgängig machen
            excess_db = hf_cumul_db - MAX_HF_CUMUL_DB
            gain_correction = 10 ** (-excess_db / 20.0)
            if enhanced_audio.ndim == 1:
                enhanced_audio = enhanced_audio * gain_correction
            else:
                enhanced_audio = enhanced_audio * gain_correction
            enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "hf_boost_db": float(hf_boost_db),
                "shelf_gain_db": float(config["shelf_gain_db"]),
                "shelf_freq_hz": float(config["shelf_freq_hz"]),
                "rt_factor": float(rt_factor),
                "hf_cumulative_db": float(hf_cumul_db),
            },
            warnings=[],
        )

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, float]) -> np.ndarray:
        """Enhance air band in a single audio channel."""
        # 1. Shelving EQ
        shelved = self._apply_high_shelf(audio, sample_rate, config["shelf_freq_hz"], config["shelf_gain_db"])

        # 2. Harmonic excitation
        excited = self._apply_exciter(audio, sample_rate, config["exciter_mix"], config["saturation_drive"])

        # Combine (weighted)
        enhanced = shelved * 0.7 + excited * 0.3

        return enhanced

    def _apply_high_shelf(self, audio: np.ndarray, sample_rate: int, freq_hz: float, gain_db: float) -> np.ndarray:
        """Apply high-frequency shelving filter (biquad coefficients cached per key)."""
        cache_key = (sample_rate, freq_hz, gain_db)
        if cache_key not in self._shelf_coeffs:
            w0 = 2 * np.pi * freq_hz / sample_rate
            A = 10 ** (gain_db / 40)
            alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1 / a0, a2 / a0])
            self._shelf_coeffs[cache_key] = (b, a)
        b, a = self._shelf_coeffs[cache_key]
        return signal.lfilter(b, a, audio)

    def _apply_exciter(self, audio: np.ndarray, sample_rate: int, mix: float, drive: float) -> np.ndarray:
        """Apply harmonic exciter to HF region (SOS filter cached per sample_rate)."""
        if sample_rate not in self._sos_air_cache:
            self._sos_air_cache[sample_rate] = signal.butter(
                4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
            )
        hf = signal.sosfilt(self._sos_air_cache[sample_rate], audio)
        excited_hf = np.tanh(hf * drive * 2) / (drive + 0.5)
        return audio + excited_hf * mix

    def _measure_hf_energy(self, audio: np.ndarray, sample_rate: int) -> float:
        """Measure high-frequency energy (12-20 kHz RMS, cached SOS filter)."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel
        if sample_rate not in self._sos_air_cache:
            self._sos_air_cache[sample_rate] = signal.butter(
                4, self.AIR_BAND_HZ, btype="band", fs=sample_rate, output="sos"
            )
        hf = signal.sosfilt(self._sos_air_cache[sample_rate], audio)
        return float(np.sqrt(np.mean(hf**2)))
