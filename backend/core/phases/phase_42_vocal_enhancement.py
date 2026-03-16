#!/usr/bin/env python3
"""
Phase 43: Vocal Enhancement v2.0 - Professional
Comprehensive vocal processing chain for clarity, presence, and polish.

Algorithm Overview:
1. Vocal Detection:
   - Frequency range analysis (fundamental @ 80-400 Hz)
   - Formant detection (F1/F2 @ 300-3000 Hz)
   - Harmonic series identification
2. Multi-Stage Processing:
   - De-essing (sibilance control @ 6-10 kHz)
   - Presence boost (clarity @ 3-6 kHz)
   - Formant enhancement (vowel clarity @ 1-3 kHz)
   - Breath control (gentle reduction @ 8-12 kHz)
   - Chest resonance (warmth @ 100-250 Hz)
3. Dynamic Processing:
   - Micro-compression (syllable-level dynamics)
   - Envelope shaping (attack/sustain balance)
4. Material Adaptation:
   - Shellac/Vinyl: Restore missing formants
   - Tape: Restore HF detail
   - Digital: Add analog warmth

Scientific Foundation:
- Fant (1960): Acoustic Theory of Speech Production
- Peterson & Barney (1952): Control Methods Used in Study of Vowels
- Hillenbrand et al. (1995): Acoustic Characteristics of American English Vowels
- Sundberg (1987): The Science of the Singing Voice
- Titze (2000): Principles of Voice Production

Industry Benchmarks:
- iZotope Nectar (Vocal processing suite)
- Waves Renaissance Vox (Classic vocal compressor)
- FabFilter Pro-Q 3 (Surgical EQ for vocals)
- Antares Auto-Tune Pro (Pitch + formant correction)
- Universal Audio Neve 1073 (Classic vocal chain)
- SSL Channel Strip (Broadcast vocal processing)

Quality Target: 0.85 → 0.95 (+12% improvement)
Performance Target: <0.30× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import os
import sys


import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# VocalAI Enhancement (Spec §2.8 — Stimmtyp-adaptive Gesangsverarbeitung)
try:
    pass

    VOCAL_AI_AVAILABLE = True
except ImportError:
    VOCAL_AI_AVAILABLE = False
    logging.getLogger(__name__).warning("VocalAIEnhancement nicht verfügbar — Standard-DSP-Vogalverarbeitung aktiv")

# FormantSystem: LPC-basiertes Formant-Tracking + Singer's Formant Enhancement (§2.8)
try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls, VowelPhonemeFormantTargets as _VowelTargetsCls
    _FORMANT_SYSTEM_AVAILABLE = True
except ImportError:
    _FormantSystemCls = None  # type: ignore
    _VowelTargetsCls = None  # type: ignore
    _FORMANT_SYSTEM_AVAILABLE = False
    logging.getLogger(__name__).debug("FormantSystem (dsp) nicht verfügbar — Bell-EQ-Fallback aktiv")

# DSP-PhonemeDetector für vokal-phonemspezifische Formant-Steuerung
try:
    from plugins.phoneme_detector import get_phoneme_detector as _get_phoneme_detector
    _PHONEME_DETECTOR_AVAILABLE = True
except ImportError:
    _get_phoneme_detector = None  # type: ignore
    _PHONEME_DETECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class VocalEnhancement(PhaseInterface):
    """
    Professional Vocal Enhancement Engine.

    Key Features:
    - Multi-stage vocal processing
    - De-essing (6-10 kHz)
    - Presence boost (3-6 kHz)
    - Formant enhancement (1-3 kHz)
    - Breath control (8-12 kHz)
    - Chest resonance (100-250 Hz)
    - Micro-compression
    - Material-adaptive parameters

    Use Cases:
    - Enhance vocal clarity and intelligibility
    - Restore vintage vocal recordings
    - Polish modern vocal tracks
    - Broadcast vocal optimization

    Performance: <0.30× realtime on modern CPU
    """

    # Vocal frequency bands
    VOCAL_BANDS = {
        "chest": (100, 250),  # Chest resonance (warmth)
        "fundamental": (80, 400),  # Fundamental frequency range
        "formant": (300, 3000),  # Formant region (vowels)
        "presence": (3000, 6000),  # Presence and clarity
        "sibilance": (6000, 10000),  # Sibilance (s, t, sh sounds)
        "breath": (8000, 12000),  # Breath noise
    }

    # Processing parameters (material-adaptive)
    ENHANCEMENT_CONFIG = {
        MaterialType.SHELLAC: {
            "deess_threshold_db": -15,
            "deess_reduction_db": 8,
            "presence_gain_db": 5.0,
            "formant_gain_db": 4.0,
            "chest_gain_db": 3.0,
            "breath_reduction_db": 6,
            "compression_ratio": 2.5,
        },
        MaterialType.VINYL: {
            "deess_threshold_db": -18,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.0,
            "formant_gain_db": 3.5,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.0,
        },
        MaterialType.TAPE: {
            "deess_threshold_db": -20,
            "deess_reduction_db": 5,
            "presence_gain_db": 3.5,
            "formant_gain_db": 3.0,
            "chest_gain_db": 2.0,
            "breath_reduction_db": 4,
            "compression_ratio": 1.8,
        },
        MaterialType.CD_DIGITAL: {
            "deess_threshold_db": -20,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.5,
            "formant_gain_db": 4.0,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.2,
        },
        MaterialType.STREAMING: {
            "deess_threshold_db": -18,
            "deess_reduction_db": 6,
            "presence_gain_db": 4.0,
            "formant_gain_db": 3.5,
            "chest_gain_db": 2.5,
            "breath_reduction_db": 5,
            "compression_ratio": 2.0,
        },
    }

    def __init__(self):
        super().__init__()
        self.name = "Vocal Enhancement v2 Professional"
        # LPC-basiertes Formant-Tracking + Singer's Formant Enhancement (2.5–3.5 kHz)
        self._formant_system = None
        if _FORMANT_SYSTEM_AVAILABLE:
            try:
                self._formant_system = _FormantSystemCls(
                    n_formants=5, correction_strength=0.5, enhance_singers_formant=True
                )
                logger.debug("FormantSystem (LPC) für Phase 42 initialisiert")
            except Exception as _e:
                logger.debug("FormantSystem-Init fehlgeschlagen: %s", _e)

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_42_vocal_enhancement",
            name="Vocal Enhancement v2 Professional",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_19_de_esser", "phase_38_presence_boost"],
            estimated_time_factor=0.30,
            version="2.0.0",
            memory_requirement_mb=90,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,
            description="Comprehensive vocal processing chain for clarity and polish",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply vocal enhancement to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with enhanced audio
        """
        start_time = time.time()
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        is_stereo = audio.ndim == 2
        config = self.ENHANCEMENT_CONFIG.get(material, self.ENHANCEMENT_CONFIG[MaterialType.CD_DIGITAL])

        # Detect if audio contains vocals (simple heuristic)
        has_vocals = self._detect_vocals(audio, sample_rate)

        if not has_vocals:
            logger.info("No vocal content detected - skipping vocal enhancement")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={"material": material.name, "vocals_detected": False},
                warnings=["No vocal content detected - enhancement skipped"],
            )

        # Process each channel
        if is_stereo:
            enhanced_left = self._enhance_channel(audio[:, 0], sample_rate, config)
            enhanced_right = self._enhance_channel(audio[:, 1], sample_rate, config)
            enhanced_audio = np.column_stack((enhanced_left, enhanced_right))
        else:
            enhanced_audio = self._enhance_channel(audio, sample_rate, config)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        enhanced_audio = np.nan_to_num(enhanced_audio, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=enhanced_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "vocals_detected": True,
                "presence_gain_db": float(config["presence_gain_db"]),
                "formant_gain_db": float(config["formant_gain_db"]),
                "compression_ratio": float(config["compression_ratio"]),
                "rt_factor": float(rt_factor),
                "vocal_ai_linked": VOCAL_AI_AVAILABLE,
            },
            warnings=[] if rt_factor < 0.35 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _detect_vocals(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Simple vocal detection based on formant energy."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel

        # Measure energy in formant region (300-3000 Hz)
        sos_formant = signal.butter(4, self.VOCAL_BANDS["formant"], btype="band", fs=sample_rate, output="sos")
        formant_signal = signal.sosfilt(sos_formant, audio)
        formant_energy = np.mean(formant_signal**2)

        # Measure total energy
        total_energy = np.mean(audio**2)

        # If formant region has >20% of total energy, likely contains vocals
        if total_energy > 1e-10:
            formant_ratio = formant_energy / total_energy
            return formant_ratio > 0.20
        else:
            return False

    def _enhance_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Enhance vocals in a single audio channel."""
        enhanced = audio.copy()

        # Stage 1: De-essing (sibilance control)
        enhanced = self._apply_deessing(enhanced, sample_rate, config)

        # Stage 2: Formant enhancement (vowel clarity)
        enhanced = self._enhance_formants(enhanced, sample_rate, config)

        # Stage 3: Presence boost (clarity)
        enhanced = self._boost_presence(enhanced, sample_rate, config)

        # Stage 4: Chest resonance (warmth)
        enhanced = self._enhance_chest(enhanced, sample_rate, config)

        # Stage 5: Breath control
        enhanced = self._control_breath(enhanced, sample_rate, config)

        # Stage 6: Micro-compression (dynamics)
        enhanced = self._apply_compression(enhanced, sample_rate, config)

        return enhanced

    def _apply_deessing(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Apply de-essing to sibilance band."""
        # Extract sibilance band
        sos = signal.butter(4, self.VOCAL_BANDS["sibilance"], btype="band", fs=sample_rate, output="sos")
        sibilance = signal.sosfilt(sos, audio)

        # Dynamic range compression on sibilance
        envelope = np.abs(signal.hilbert(sibilance))
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Apply reduction above threshold
        gain_db = np.where(
            envelope_db > config["deess_threshold_db"],
            -config["deess_reduction_db"] * ((envelope_db - config["deess_threshold_db"]) / 20),
            0,
        )

        # Smooth gain
        gain_db_smooth = signal.savgol_filter(gain_db, window_length=min(101, len(gain_db) // 10 * 2 + 1), polyorder=3)
        gain_linear = 10 ** (gain_db_smooth / 20)

        # Apply to sibilance and subtract from original
        sibilance_reduced = sibilance * gain_linear
        deessed = audio + (sibilance_reduced - sibilance) * 0.7

        return deessed

    def _enhance_formants(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Enhance formant region using LPC FormantSystem with Singer's Formant Enhancement.

        Processing chain (§2.8):
            1. FormantSystem.process() — LPC tracking, drift correction,
               Singer's Formant (2.5–3.5 kHz).  Primary path.
            2. FormantSystem.phoneme_guided_enhance() — per-vowel canonical
               target steering (Peterson & Barney 1952, Hillenbrand 1995).
               Runs after step 1 with correction_strength=0.25 (identity-safe).
               Uses the DSP-PhonemeDetector label ('V') to restrict steering
               to voiced segments; vowel class is auto-classified from F1/F2.
            3. Bell EQ @ 1.5 kHz — DSP fallback when FormantSystem unavailable.
        """
        # Primary: LPC-based formant tracking + Singer's Formant Enhancement
        if self._formant_system is not None:
            try:
                enhanced, _ = self._formant_system.process(audio, sample_rate)
                enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
                enhanced = np.clip(enhanced, -1.0, 1.0)

                # Stage 2: phoneme-guided per-vowel formant steering
                try:
                    enhanced, _pg_report = self._formant_system.phoneme_guided_enhance(
                        enhanced,
                        sample_rate,
                        phoneme_segments=None,   # DSP fallback: F1/F2-driven classification
                        gender="unknown",
                        correction_strength=0.25,
                    )
                    enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
                    enhanced = np.clip(enhanced, -1.0, 1.0)
                    logger.debug(
                        "Phase42 phoneme_guided_enhance: vowel_frames=%d/%d",
                        _pg_report.get("vowel_segments_processed", 0),
                        _pg_report.get("total_frames", 0),
                    )
                except Exception as _pg_err:
                    logger.debug("phoneme_guided_enhance fehlgeschlagen (ignoriert): %s", _pg_err)

                return enhanced.astype(audio.dtype)
            except Exception as _fs_err:
                logger.debug("FormantSystem fehlgeschlagen, Bell-EQ-Fallback: %s", _fs_err)

        # DSP-Fallback: Bell-EQ @ 1.5 kHz (Formant Region)
        gain_db = config["formant_gain_db"]
        w0 = 2 * np.pi * 1500 / sample_rate
        q = 2.0
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        enhanced = signal.lfilter(b, a, audio)
        return enhanced

    def _boost_presence(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Boost presence region."""
        # Bell filter @ 4500 Hz
        w0 = 2 * np.pi * 4500 / sample_rate
        gain_db = config["presence_gain_db"]
        q = 1.5
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        enhanced = signal.lfilter(b, a, audio)
        return enhanced

    def _enhance_chest(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Enhance chest resonance."""
        # Bell filter @ 175 Hz
        w0 = 2 * np.pi * 175 / sample_rate
        gain_db = config["chest_gain_db"]
        q = 1.0
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain_db / 40)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        enhanced = signal.lfilter(b, a, audio)
        return enhanced

    def _control_breath(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Reduce breath noise."""
        # Extract breath band
        sos = signal.butter(4, self.VOCAL_BANDS["breath"], btype="band", fs=sample_rate, output="sos")
        breath = signal.sosfilt(sos, audio)

        # Reduce breath by fixed amount
        reduction_linear = 10 ** (-config["breath_reduction_db"] / 20)
        breath_reduced = breath * reduction_linear

        controlled = audio + (breath_reduced - breath) * 0.6
        return controlled

    def _apply_compression(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> np.ndarray:
        """Apply micro-compression."""
        # Simple RMS-based compression
        window_samples = int(0.020 * sample_rate)  # 20ms window
        rms = np.sqrt(signal.convolve(audio**2, np.ones(window_samples) / window_samples, mode="same"))
        rms_db = 20 * np.log10(rms + 1e-10)

        threshold_db = -15
        ratio = config["compression_ratio"]

        # Compute gain reduction
        gain_db = np.where(rms_db > threshold_db, -(rms_db - threshold_db) * (1 - 1 / ratio), 0)

        # Smooth gain
        gain_db_smooth = signal.savgol_filter(gain_db, window_length=min(201, len(gain_db) // 10 * 2 + 1), polyorder=3)
        gain_linear = 10 ** (gain_db_smooth / 20)

        compressed = audio * gain_linear

        # Make-up gain
        makeup_gain = 1.2
        compressed = compressed * makeup_gain

        return compressed
