#!/usr/bin/env python3
"""
Phase 51: Drums/Percussion Enhancement v1.0 - Tier 1 ML-Hybrid
Professional drums and percussion processing for instrumental restoration.

Algorithm Overview:
1. Kick Drum Enhancement (20-80 Hz sub-bass punch)
2. Snare Crack Enhancement (200-400 Hz + 1-3 kHz articulation)
3. Hi-Hat Clarification (8-12 kHz clarity and presence)
4. Cymbal Shimmer Enhancement (12-20 kHz air and shimmer)
5. Transient shaping (attack/sustain optimization)

Components:
- KickDrumEnhancer: Sub-bass reinforcement
- SnareCrackEnhancer: Articulation and definition
- HiHatClarifier: High-frequency clarity
- CymbalShimmerEnhancer: Air band processing

Scientific Foundation:
- Masri (1996): Computer Modeling of Sound for Transformation and Synthesis of Musical Signals
- Duxbury et al. (2003): Complex Domain Onset Detection for Musical Signals
- Bello et al. (2005): A Tutorial on Onset Detection in Music Signals
- Amatriain et al. (2003): CLAM: A Framework for Efficient and Rapid Development of Cross-platform Audio Applications

Industry Benchmarks:
- iZotope Nectar/Trash (Transient shaping)
- Waves Dbx 160 (Drum compression)
- SPL Transient Designer (Transient control)
- SSL Drumstrip (Professional drum processing)
- Slate Digital Trigger (Drum enhancement)

Tier 1 Priority: PRIORITY 2 (after Bass, critical for groove and punch)
Quality Target: Transparenz 87% → 93% (+6% for percussive content)
Performance Target: <0.25× realtime

Author: Aurik Development Team  Phase 2.3 - Tier 1 ML-Hybrid
Version: 1.0.0
Date: 16. Februar 2026
"""

import os
import sys


import logging
import time

import numpy as np

from backend.core.defect_scanner import MaterialType
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Import Drums Enhancement DSP module
try:
    from dsp.drums_enhancement import DrumsEnhancementSystem
    DRUMS_ENHANCEMENT_AVAILABLE = True
except ImportError:
    DRUMS_ENHANCEMENT_AVAILABLE = False
    logging.warning("DrumsEnhancementSystem not available")

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls
    _FORMANT_SYSTEM_DRUMS: _FormantSystemCls | None = None
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_DRUMS = None

logger = logging.getLogger(__name__)


class DrumsEnhancementV1(PhaseInterface):
    """
    Professional Drums/Percussion Enhancement Engine (Tier 1).

    Key Features:
    - Kick drum sub-bass punch (20-80 Hz)
    - Snare articulation and crack (200-400 Hz, 1-3 kHz)
    - Hi-hat clarity (8-12 kHz)
    - Cymball shimmer (12-20 kHz)
    - Transient shaping
    - Material-adaptive intensity

    Use Cases:
    - Restore lost drum impact from over-processing
    - Enhance percussive elements in mixes
    - Improve groove and punch
    - Broadcast and mastering drum optimization

    Performance: <0.25× realtime on modern CPU
    """

    # Material-specific enhancement parameters
    ENHANCEMENT_CONFIG = {
        MaterialType.SHELLAC: {
            "kick_gain_db": 4.0,  # Strong (restore missing low-end)
            "snare_articulation": 0.9,  # High (restore attack)
            "hihat_clarity_db": 3.0,  # Strong (restore HF)
            "cymbal_shimmer_db": 2.5,
            "transient_enhancement": 0.8,
            "mix": 0.60,  # 60% enhancement
        },
        MaterialType.VINYL: {
            "kick_gain_db": 3.0,  # Moderate
            "snare_articulation": 0.75,
            "hihat_clarity_db": 2.5,
            "cymbal_shimmer_db": 2.0,
            "transient_enhancement": 0.7,
            "mix": 0.50,  # 50% enhancement
        },
        MaterialType.TAPE: {
            "kick_gain_db": 2.5,  # Gentle (tape has good bass)
            "snare_articulation": 0.6,
            "hihat_clarity_db": 2.0,  # Restore HF loss
            "cymbal_shimmer_db": 1.5,
            "transient_enhancement": 0.5,
            "mix": 0.40,  # 40% enhancement
        },
        MaterialType.CD_DIGITAL: {
            "kick_gain_db": 2.0,  # Subtle
            "snare_articulation": 0.5,
            "hihat_clarity_db": 1.5,
            "cymbal_shimmer_db": 1.0,
            "transient_enhancement": 0.4,
            "mix": 0.30,  # 30% enhancement
        },
        MaterialType.STREAMING: {
            "kick_gain_db": 1.5,  # Minimal (already processed)
            "snare_articulation": 0.4,
            "hihat_clarity_db": 1.0,
            "cymbal_shimmer_db": 0.8,
            "transient_enhancement": 0.3,
            "mix": 0.25,  # 25% enhancement
        },
    }

    DEFAULT_CONFIG = {
        "kick_gain_db": 2.5,
        "snare_articulation": 0.6,
        "hihat_clarity_db": 2.0,
        "cymbal_shimmer_db": 1.5,
        "transient_enhancement": 0.5,
        "mix": 0.40,
    }

    def __init__(self, sample_rate: int = 48000, **kwargs):
        """
        Initialize Drums Enhancement Phase.

        Args:
            sample_rate: Audio sample rate (Hz)
            **kwargs: Override parameters
        """
        super().__init__(sample_rate, **kwargs)
        self.enhancer = None  # Lazy init

    def process(
        self, audio: np.ndarray, material_type: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Enhance drums and percussion.

        Args:
            audio: Input audio (mono or stereo)
            material_type: Source material type
            **kwargs: Additional parameters

        Returns:
            PhaseResult with enhanced audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        # ── PANNs Drums-Confidence-Check (Spec §2.9: Schwellwert ≥ 0.5) ────
        panns_tags = kwargs.get("panns_tags", {})
        drums_confidence = 0.0
        for tag_name, conf in panns_tags.items():
            tag_lower = tag_name.lower()
            if any(k in tag_lower for k in ("drum", "percussion", "kick", "snare", "hihat", "cymbal")):
                drums_confidence = max(drums_confidence, float(conf))
        # Wenn PANNs-Tags vorhanden aber Drums nicht erkannt → Phase überspringen
        if panns_tags and drums_confidence < 0.50:
            logger.info(
                "Phase 51: Drums-Confidence %.2f < 0.50 — Phase übersprungen (Spec §2.9)",
                drums_confidence,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={"skipped": True, "drums_confidence": drums_confidence},
                execution_time_seconds=0.0,
                metadata={"algorithm": "skip_panns_confidence"},
                warnings=[],
                modifications={},
            )
        if drums_confidence > 0:
            logger.info("Phase 51: PANNs Drums-Confidence=%.2f ≥ 0.50 — Verarbeitung aktiv", drums_confidence)

        if not DRUMS_ENHANCEMENT_AVAILABLE:
            logger.warning("DrumsEnhancementSystem not available, bypassing...")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=False,
                audio=audio,
                execution_time_seconds=time.time() - start_time,
                warnings=["DrumsEnhancementSystem module not available"],
            )

        try:
            # Get material-specific config
            config = self.ENHANCEMENT_CONFIG.get(material_type, self.DEFAULT_CONFIG).copy()
            config.update(kwargs)  # Allow override

            # Lazy init enhancer
            if self.enhancer is None:
                self.enhancer = DrumsEnhancementSystem(
                    kick_gain_db=config["kick_gain_db"],
                    snare_articulation=config["snare_articulation"],
                    hihat_clarity_db=config["hihat_clarity_db"],
                    cymbal_shimmer_db=config["cymbal_shimmer_db"],
                )

            # Process audio
            processed_audio, report = self.enhancer.process(audio, self.sample_rate)

            # Mix with original (parallel processing)
            mix = config["mix"]
            enhanced = audio * (1.0 - mix) + processed_audio * mix

            execution_time = time.time() - start_time

            # Build metrics
            metrics = {
                "kick_energy_change_db": report.get("kick_energy_change_db", 0.0),
                "snare_energy_change_db": report.get("snare_energy_change_db", 0.0),
                "hihat_energy_change_db": report.get("hihat_energy_change_db", 0.0),
                "cymbal_energy_change_db": report.get("cymbal_energy_change_db", 0.0),
                "transient_enhancement": report.get("transient_enhancement", 0.0),
                "mix_ratio": mix,
                "material_type": material_type.value,
                "config_applied": config,
            }

            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            enhanced = np.clip(enhanced, -1.0, 1.0)

            # Instrument-guided formant enhancement (drums resonance targets: Rossing 1992)
            igt_frames = 0
            try:
                global _FORMANT_SYSTEM_DRUMS
                if _FormantSystemCls is not None:
                    if _FORMANT_SYSTEM_DRUMS is None:
                        _FORMANT_SYSTEM_DRUMS = _FormantSystemCls(enhance_singers_formant=False)
                    enhanced, igt_report = _FORMANT_SYSTEM_DRUMS.instrument_guided_enhance(
                        enhanced, self.sample_rate, instrument="drums", correction_strength=0.15
                    )
                    igt_frames = igt_report.get("frames_processed", 0)
                    logger.debug("Phase 51 InstrumentFormant: drums frames=%d", igt_frames)
            except Exception as _igt_exc:
                logger.debug("Phase 51 instrument_guided_enhance skipped: %s", _igt_exc)

            # Formant-Drift-Korrektur via DTW (Schritt 3)
            try:
                from dsp.instrument_formant_corrector import correct_instrument_formant_drift
                drift_result = correct_instrument_formant_drift(enhanced, sample_rate, instrument="drums")
                enhanced = drift_result.audio
                logger.debug(
                    "Phase 51 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                    drift_result.drift_detected, drift_result.n_frames_corrected,
                    drift_result.total_frames, drift_result.mean_drift_hz,
                )
            except Exception as _drift_exc:
                logger.debug("Phase 51 drift correction skipped: %s", _drift_exc)

            # Sub-Stem-Verarbeitung (Schritt 4)
            try:
                from backend.core.sub_stem_processor import process_sub_stems
                ss_result = process_sub_stems(enhanced, sample_rate, instrument="drums",
                                              processing_strength=0.30)
                enhanced = ss_result.audio
                logger.debug("Phase 51 sub-stem: bands=%d strength=%.2f",
                             ss_result.n_bands, ss_result.processing_strength)
            except Exception as _ss_exc:
                logger.debug("Phase 51 sub-stem skipped: %s", _ss_exc)

            # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
            try:
                from backend.core.physics_resonance_enhancer import enhance_physics_resonance
                pr_result = enhance_physics_resonance(enhanced, sample_rate, instrument="drums",
                                                      enhancement_strength=0.35)
                enhanced = pr_result.audio
                logger.debug("Phase 51 physics resonance: peaks=%d strength=%.2f",
                             pr_result.n_peaks, pr_result.enhancement_strength)
            except Exception as _pr_exc:
                logger.debug("Phase 51 physics resonance skipped: %s", _pr_exc)

            return PhaseResult(
                success=True,
                audio=enhanced,
                execution_time_seconds=execution_time,
                metadata={**metrics, "instrument_formant_frames": igt_frames},
                modifications={
                    "drums_enhanced": True,
                    "kick_enhanced": report.get("kick_energy_change_db", 0) > 0.5,
                    "snare_enhanced": report.get("snare_energy_change_db", 0) > 0.5,
                    "hihat_enhanced": report.get("hihat_energy_change_db", 0) > 0.3,
                },
            )

        except Exception as e:
            logger.error(f"Drums enhancement failed: {e}", exc_info=True)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=False,
                audio=audio,  # Return original audio
                execution_time_seconds=time.time() - start_time,
                warnings=[f"Drums enhancement failed: {str(e)}"],
            )

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_51_drums_enhancement",
            name="Drums/Percussion Enhancement (Tier 1)",
            category=PhaseCategory.ENHANCEMENT,
            priority=8,  # High priority (Tier 1)
            dependencies=[],  # Independent
            estimated_time_factor=0.15,  # 15% of audio duration
            version="1.0.0",
            memory_requirement_mb=100,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.85,  # High impact on percussive content
            description="Professional drums and percussion enhancement: kick punch, snare crack, hi-hat clarity, cymbal shimmer",
        )

    def supports_material(self, material_type: MaterialType) -> bool:
        """Check if material type is supported."""
        return material_type in self.ENHANCEMENT_CONFIG or material_type in [
            MaterialType.CD_DIGITAL,
            MaterialType.DAT,
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
            MaterialType.AAC,
            MaterialType.STREAMING,
        ]

    def estimate_time(self, audio_duration_seconds: float) -> float:
        """Estimate processing time."""
        return audio_duration_seconds * 0.15  # 15% of audio duration

    def validate_input(self, audio: np.ndarray) -> tuple[bool, str | None]:
        """Validate input audio."""
        if audio.size == 0:
            return False, "Empty audio input"
        if not np.isfinite(audio).all():
            return False, "Audio contains NaN or Inf values"
        if audio.ndim > 2:
            return False, "Audio must be mono or stereo"
        return True, None


# Export the class
__all__ = ["DrumsEnhancementV1"]
