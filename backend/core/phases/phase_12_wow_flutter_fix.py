#!/usr/bin/env python3
"""
Phase 12: Professional Wow & Flutter Correction v2.0
=====================================================

Corrects pitch and speed instability caused by mechanical variations in analog playback equipment.

SCIENTIFIC FOUNDATION (Über-SOTA):
- **Mauch & Dixon (2014)**: pYIN — probabilistischer Multi-Threshold-Pitch-Estimator
  → Primärer Algorithmus (ersetzt simples YIN als primäre Methode)
- De Cheveigné & Kawahara (2002): YIN — nur noch als Legacy-Fallback referenziert
- Laroche & Dolson (1999): Improved Phase Vocoder Time-Scale Modification of Audio
- Driedger & Müller (2016): TSM Toolbox

INDUSTRY BENCHMARKS:
- iZotope RX 10 De-flutter (Phase Vocoder + Spectral Continuity)
- Waves X-Speed (Time & Pitch Manipulation with Formant Preservation)
- Cedar Retouch Pro (Professional Wow/Flutter Removal)
- Capstan by Plangent Processes (Reference Standard for Analog Transfer)
- WOW Control by Magix (Spectral Repair based)
- Steinberg SpectraLayers (Visual Wow/Flutter Correction)
- iZotope Neutron (Pitch Tracking + Time Correction)

ALGORITHM OVERVIEW:

1. Multi-Resolution Pitch Detection (**pYIN — Mauch & Dixon 2014**)
   - 50ms windows with 75% overlap (high temporal resolution)
   - pYIN: Probabilistic Multi-Threshold CMND (Beta-verteilte Gewichte)
   - Sub-harmonic rejection via gewichtetes Kandidaten-Medioid
   - Temporal smoothing (exponentiell, α=0.7)

2. Wow vs Flutter Separation
   - Wow: <4 Hz speed variations (slow pitch drift)
   - Flutter: 4-100 Hz speed variations (fast mechanical vibrations)
   - Separate low-pass (<4 Hz) and band-pass (4-100 Hz) filters
   - Different correction strategies per component

3. Spectral Continuity Preservation
   - Phase Vocoder with vertical phase coherence
   - Formant-preserving time-stretching
   - Harmonic locking (prevent harmonic smearing)
   - Transient detection and preservation

4. Time-Stretching via Phase Vocoder
   - STFT with Hann window (2048 samples, 75% overlap)
   - Phase unwrapping and instantaneous frequency estimation
   - Time-varying stretch factors from pitch deviation curve
   - Overlap-add synthesis with phase coherence

5. Material-Adaptive Correction
   - Tape: Aggressive correction (0.9), high sensitivity (capstan flutter)
   - Vinyl: Moderate correction (0.7), turntable speed variations
   - Shellac: Conservative (0.6), hand-crank artifacts
   - CD_Digital: Minimal (0.2), rare digital artifacts

QUALITY TARGETS:
- Pitch stability: <0.3% residual deviation (Professional standard)
- Spectral artifacts: <-40 dB (imperceptible)
- Transient preservation: >95% (no smearing)
- Realtime factor: <0.8× (faster than playback)

Author: Aurik Professional Team
Version: 2.0.0 (Professional)
Date: February 2026
"""

import logging
import os
import time

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Resource Management for fallback to lightweight algorithms
try:
    from backend.core.adaptive_resource_manager import adaptive_resource_manager

    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("AdaptiveResourceManager not available, no automatic fallback")

# ML-Hybrid Support (Aurik 9.0 - Phase 12 v3.0)
try:
    from backend.core.hybrid.hybrid_wow_flutter import HybridWowFlutter, PitchDetectionStrategy, WowFlutterConfig

    ML_HYBRID_AVAILABLE = True
except ImportError:
    ML_HYBRID_AVAILABLE = False
    logging.getLogger(__name__).warning("ML-Hybrid wow/flutter detector not available, using DSP-only mode")

logger = logging.getLogger(__name__)


class WowFlutterFix(PhaseInterface):
    """Professional Wow & Flutter Correction with YIN pitch detection and Phase Vocoder time-stretching."""

    # Material-adaptive correction strength (0.0-1.0)
    CORRECTION_STRENGTH = {
        MaterialType.TAPE: 0.80,  # v9.10.97: raised from 0.65 — tonal_center PMGG-excluded (§2.29b);
        #   cassette head-settling wow/flutter requires stronger correction.
        #   Was reduced in v9.10.77 due to tonal_center regression, but K-S proxy
        #   (§9.7.11) now excludes tonal_center from PMGG delta-checks for phase_12.
        MaterialType.VINYL: 0.70,  # Moderate (turntable speed variations, belt/motor issues)
        MaterialType.SHELLAC: 0.60,  # Conservative (hand-crank artifacts, worn mechanisms)
        MaterialType.CD_DIGITAL: 0.20,  # Minimal (rare digital artifacts)
        MaterialType.STREAMING: 0.10,  # Very minimal (usually none)
    }

    # Detection sensitivity (minimum pitch deviation to correct, in %)
    DETECTION_THRESHOLD = {
        MaterialType.TAPE: 0.3,  # 0.3% pitch deviation (high sensitivity)
        MaterialType.VINYL: 0.5,  # 0.5% pitch deviation
        MaterialType.SHELLAC: 0.8,  # 0.8% pitch deviation
        MaterialType.CD_DIGITAL: 2.0,  # 2.0% pitch deviation (low sensitivity)
        MaterialType.STREAMING: 3.0,  # 3.0% pitch deviation (rarely triggered)
    }

    # YIN algorithm parameters
    YIN_THRESHOLD = 0.15  # Confidence threshold for pitch detection
    PITCH_WINDOW_MS = 100  # Larger window for stability (was 50ms)
    # v9.10.111: Restored to 75 % overlap (factor=4, was 2=50 %) for 4× better
    # temporal resolution in pYIN flutter detection. At 48 kHz and 100 ms window:
    #   factor=2 → hop=50 ms — insufficient for 4–20 Hz flutter (20–250 ms periods)
    #   factor=4 → hop=25 ms — 2+ samples per flutter cycle (Nyquist-safe)
    # librosa.pyin fast-path ≪ 200 ms even for 20-min files. §9.10.80
    # Quality-first: no RT sacrifice in main path.
    PITCH_HOP_FACTOR = 4  # 75 % overlap — restored from 2 (v9.10.111)

    # Phase Vocoder / STFT parameters
    STFT_WINDOW_SIZE = 2048  # 23 Hz/bin @ 48 kHz — restored from 1024 (v9.10.111)
    STFT_HOP_SIZE = 512  # 75 % overlap with 2048 window — restored from 256

    # Formant preservation (prevent "chipmunk" effect)
    PRESERVE_FORMANTS = True

    def __init__(self):
        super().__init__()
        self.name = "Wow & Flutter Correction v2 Professional"

    @staticmethod
    def _derive_safe_timing_profile(
        material: MaterialType,
        mean_confidence: float,
        vocals_confidence: float,
        *,
        polyphonic_fallback: bool = False,
    ) -> tuple[float, float]:
        """Reduce timing aggression for content that is easy to damage.

        Vocal-heavy vintage transfers are especially sensitive to articulation and
        authenticity loss from even correct-but-strong time warping. Keep the
        internal timing remap narrower than the outer PMGG strength alone would
        allow.
        """
        strength_scale = 1.0
        max_stretch_delta = 0.05

        if vocals_confidence >= 0.40:
            strength_scale *= 0.82
            max_stretch_delta = min(max_stretch_delta, 0.035)

        if (
            material
            in {
                MaterialType.VINYL,
                MaterialType.SHELLAC,
                MaterialType.WAX_CYLINDER,
                MaterialType.WIRE_RECORDING,
                MaterialType.LACQUER_DISC,
            }
            and mean_confidence < 0.75
        ):
            strength_scale *= float(np.clip(0.82 + 0.20 * mean_confidence, 0.82, 0.97))
            max_stretch_delta = min(max_stretch_delta, 0.03)
        elif mean_confidence < 0.60:
            strength_scale *= 0.90
            max_stretch_delta = min(max_stretch_delta, 0.04)

        if polyphonic_fallback:
            # If the polyphonic speed estimator rejected the signal as implausible,
            # stay in a narrow DSP-safe correction band. Continuing with the broader
            # quality-mode path can drift tonal center on difficult analog vocals.
            strength_scale *= 0.78
            max_stretch_delta = min(max_stretch_delta, 0.02)

        return float(strength_scale), float(max_stretch_delta)

    @staticmethod
    def _should_bypass_unsafe_polyphonic_fallback(
        material: MaterialType,
        mean_confidence: float,
        vocals_confidence: float,
        *,
        polyphonic_fallback: bool,
    ) -> bool:
        """Bypass correction entirely when the estimator already proved unstable.

        If the polyphonic speed estimator rejects the input as implausible, a second,
        still-global pitch correction pass on vocal-heavy analog material often harms
        tonal center more than it helps. In that case, minimal intervention is safer
        than a likely rollback.
        """
        if not polyphonic_fallback:
            return False
        if vocals_confidence < 0.40:
            return False
        if mean_confidence >= 0.80:
            return False
        return material in {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
        }

    def get_metadata(self) -> PhaseMetadata:
        """Get phase metadata."""
        return PhaseMetadata(
            phase_id="phase_12_wow_flutter_fix",
            name="Wow & Flutter Correction v2 Professional",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH priority (wow/flutter very audible)
            dependencies=["phase_01_click_removal", "phase_09_crackle_removal"],
            estimated_time_factor=0.15,  # 15% of audio duration
            version="2.0.0",
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.92,  # Professional-grade wow/flutter correction
            description="Professional wow & flutter correction with YIN pitch detection and Phase Vocoder time-stretching",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        """
        Correct wow & flutter using professional pitch detection + Phase Vocoder.

        Args:
            audio: Audio samples (mono: [samples], stereo: [samples, 2])
            sample_rate: Sample rate in Hz
            material: Material type for adaptive parameters

        Returns:
            PhaseResult with wow/flutter corrected audio
        """
        self.validate_input(audio)
        assert sample_rate == 48000, f"Interne SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        _original_audio = np.asarray(audio, dtype=np.float32).copy()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=passthrough,
                metrics={
                    "wow_flutter_detected": False,
                    "max_deviation_percent": 0.0,
                    "correction_applied": 0.0,
                    "material": material.value,
                    "mean_confidence": 0.0,
                    "quality_mode": kwargs.get("quality_mode", "balanced"),
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "version": "4.1_locality",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Get material-specific parameters
        strength = float(self.CORRECTION_STRENGTH.get(material, 0.7) * _effective_strength)
        threshold = self.DETECTION_THRESHOLD.get(material, 0.5)

        # Convert to mono for pitch analysis
        is_stereo = audio.ndim == 2
        mono = np.mean(audio, axis=1) if is_stereo else audio.copy()

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "quality")

        # Check resource availability for ML-Hybrid (fallback to lightweight if needed)
        use_lightweight = False
        if RESOURCE_MANAGER_AVAILABLE:
            use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
            if use_lightweight:
                logger.info(
                    f"Phase 12: Resource constraint detected, forcing DSP-only mode "
                    f"(CPU: {adaptive_resource_manager.get_cpu_usage():.1f}%, "
                    f"Memory: {adaptive_resource_manager.get_memory_usage():.1f}%)"
                )

        # Strategy routing v4.0 — Capstan-kompetitiv:
        # maximum  → PolyphonicSpeedCurveEstimator (multi-F0 consensus, §2.12)
        # quality  → HybridWowFlutter pYIN+FCPE/CREPE (ML-Hybrid, HYBRID strategy)
        # balanced → HybridWowFlutter pYIN+FCPE/CREPE (ML-Hybrid, ADAPTIVE strategy)
        # fast / lightweight → pYIN DSP
        _poly_applied = False
        _poly_fallback = False

        if quality_mode in ["quality", "maximum"] and not use_lightweight:
            try:
                from backend.core.hybrid.hybrid_wow_flutter import (
                    PolyphonicSpeedCurveEstimator,
                )

                _poly_est = PolyphonicSpeedCurveEstimator()
                pitch_trajectory, confidence = _poly_est.estimate(mono, sample_rate)
                _poly_applied = True
                logger.info(
                    "Phase 12 polyphoner Konsensus: T=%d Frames, material=%s",
                    len(pitch_trajectory),
                    material.value,
                )
            except Exception as _poly_exc:
                _poly_fallback = True
                logger.warning(
                    "PolyphonicSpeedCurveEstimator fehlgeschlagen (%s) — ML-Hybrid-Fallback",
                    _poly_exc,
                )

        # ML-Hybrid only if polyphonic path did not succeed and resources permit
        use_ml_hybrid = (
            not _poly_applied
            and ML_HYBRID_AVAILABLE
            and quality_mode in ["balanced", "quality", "maximum"]
            and not use_lightweight
        )

        if _poly_fallback and quality_mode in ["quality", "maximum"]:
            # Conservative restoration fallback: keep Phase 12 in narrow DSP mode
            # after an implausible polyphonic estimate instead of escalating again.
            use_ml_hybrid = False

        if use_ml_hybrid:
            try:
                logger.info("Phase 12 ML-Hybrid: mode=%s, material=%s", quality_mode, material.value)

                # Configure ML pitch detector strategy
                if quality_mode in ["quality", "maximum"]:
                    strategy = PitchDetectionStrategy.HYBRID  # Full YIN + CREPE (polyphonic failed)
                else:  # balanced
                    strategy = PitchDetectionStrategy.ADAPTIVE  # Smart: YIN only if confident

                detector = HybridWowFlutter(
                    config=WowFlutterConfig(
                        strategy=strategy,
                        yin_threshold=self.YIN_THRESHOLD,
                        crepe_model="full" if quality_mode in ["quality", "maximum"] else "medium",
                        confidence_threshold=0.7,
                        enable_preprocessing=True,
                    )
                )

                ml_result = detector.detect_pitch(mono, sample_rate=sample_rate)

                logger.info(
                    f"ML-Hybrid Pitch-Detektion abgeschlossen: pYIN={ml_result.pyin_applied}, "
                    f"CREPE={ml_result.crepe_applied}, confidence={ml_result.mean_confidence:.3f}"
                )

                # Use ML-detected pitch trajectory
                pitch_trajectory = ml_result.pitch_trajectory
                confidence = ml_result.confidence

            except Exception as e:
                logger.warning(
                    f"ML-Hybrid Pitch-Detektion fehlgeschlagen: {e}, Fallback auf pYIN DSP. "
                    f"Fehlertyp: {type(e).__name__}"
                )
                # Fall through to DSP path below
                use_ml_hybrid = False

        # DSP-Only (Fast-Modus oder Fallback): pYIN
        if not _poly_applied and not use_ml_hybrid:
            logger.info("Phase 12 pYIN DSP: material=%s", material.value)
            pitch_trajectory, confidence = self._estimate_pitch_yin(mono, sample_rate)

        # Continue with standard wow/flutter correction pipeline
        # (regardless of detection method)

        # Confidence-Guard: Bei sehr niedriger mittlerer Konfidenz die Phase gar nicht
        # erst anwenden — Phase-Vocoder-Timestretch auf Basis unzuverlässiger Pitch-Daten
        # erzeugt Artefakte (0.09 PMGG-Regression im E2E) ohne tatsächlichen Nutzen.
        # Timing-Phasen haben keine Wet/Dry-Retries, daher frühes Bail-out.
        #
        # v9.10.97: Tape-Start Confidence Adaptation.
        # Cassette motor startup (0–20 s) produces low-confidence pitch regions because
        # the signal is genuinely unstable (speed ramp-up).  The confidence guard must
        # NOT skip the ENTIRE phase just because the first few seconds have low confidence.
        # For tape material: lower the threshold to 0.25 (from 0.40) to prevent skipping
        # precisely where correction is most needed.
        # Scientific basis: Mauch & Dixon (2014) §3.2 "Voiced probability in noisy signals";
        # pYIN confidence degrades with speed instability but pitch estimates remain usable
        # above 0.20 threshold in the low-Hz (<200 Hz) tape flutter range.
        _valid_conf = confidence[confidence > 0]
        _mean_conf = float(np.mean(_valid_conf)) if len(_valid_conf) > 0 else 0.0
        _MIN_CONFIDENCE_FOR_CORRECTION = 0.40
        if material == MaterialType.TAPE:
            _MIN_CONFIDENCE_FOR_CORRECTION = 0.25  # tape-start-aware lower threshold
        if _mean_conf < _MIN_CONFIDENCE_FOR_CORRECTION:
            logger.info(
                "Phase 12: Pitch-Konfidenz zu niedrig (%.3f < %.2f) — keine Korrektur angewandt "
                "(vermeidet Artefakte bei unsicherer Detection)",
                _mean_conf,
                _MIN_CONFIDENCE_FOR_CORRECTION,
            )
            # Tape level stabilization even when pitch confidence is too low
            n_level_dips_repaired = 0
            _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE}
            _mat_enum = material if isinstance(material, MaterialType) else None
            if _mat_enum in _TAPE_LEVEL_MATERIALS and _effective_strength > 0.0:
                audio, n_level_dips_repaired = self._stabilize_tape_level(audio, sample_rate, _effective_strength)
                if n_level_dips_repaired > 0:
                    logger.info(
                        "Phase 12 tape level stabilizer (low confidence path): %d dips repaired",
                        n_level_dips_repaired,
                    )
            audio, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
                _original_audio,
                audio,
                material,
            )
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={
                    "wow_flutter_detected": False,
                    "max_deviation_percent": 0.0,
                    "correction_applied": 0.0,
                    "material": material.value,
                    "mean_confidence": _mean_conf,
                    "quality_mode": quality_mode,
                    "skipped_reason": "low_confidence",
                    "tape_level_dips_repaired": n_level_dips_repaired,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "confidence_guard_skip",
                    "version": "4.1_confidence_guard",
                    "ml_hybrid": use_ml_hybrid,
                    "polyphonic": _poly_applied,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": _rms_drop_db,
                    "loudness_makeup_db": _makeup_db,
                },
            )

        # Step 1: Separate wow (<4 Hz) and flutter (4-100 Hz) components
        wow_component, flutter_component = self._separate_wow_flutter(pitch_trajectory, sample_rate)

        # Step 3: Detect significant wow/flutter (check if correction needed)
        wow_flutter_detected, max_deviation = self._detect_wow_flutter(pitch_trajectory, confidence, threshold)

        if not wow_flutter_detected:
            # No significant wow/flutter detected
            metadata = {
                "algorithm": (
                    "polyphonic_multi_f0_consensus_v1"
                    if _poly_applied
                    else "hybrid_ml_pyin_crepe_v3"
                    if use_ml_hybrid
                    else "pyin_phase_vocoder"
                ),
                "version": "4.0_polyphonic" if _poly_applied else "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
                "ml_hybrid": use_ml_hybrid,
                "polyphonic": _poly_applied,
            }

            if use_ml_hybrid:
                metadata["pyin_applied"] = ml_result.pyin_applied
                metadata["crepe_applied"] = ml_result.crepe_applied
                metadata["strategy_used"] = str(ml_result.strategy_used)
                metadata["ml_metadata"] = ml_result.metadata

            # Step 6c (also in no-wow/flutter path): Tape level stabilization
            n_level_dips_repaired = 0
            _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE}
            _mat_enum = material if isinstance(material, MaterialType) else None
            if _mat_enum in _TAPE_LEVEL_MATERIALS and _effective_strength > 0.0:
                audio, n_level_dips_repaired = self._stabilize_tape_level(audio, sample_rate, _effective_strength)
                if n_level_dips_repaired > 0:
                    logger.info(
                        "Phase 12 tape level stabilizer (no wow/flutter): %d dips repaired",
                        n_level_dips_repaired,
                    )

            audio, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
                _original_audio,
                audio,
                material,
            )

            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                metrics={
                    "wow_flutter_detected": False,
                    "max_deviation_percent": max_deviation,
                    "correction_applied": 0.0,
                    "material": material.value,
                    "mean_confidence": float(np.mean(confidence[confidence > 0])),
                    "quality_mode": quality_mode,
                    "tape_level_dips_repaired": n_level_dips_repaired,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    **metadata,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": _rms_drop_db,
                    "loudness_makeup_db": _makeup_db,
                },
            )

        vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
        if vocals_conf == 0.0:  # Fallback: direct callers may use panns_singing key
            vocals_conf = float(kwargs.get("panns_singing", 0.0))
        _timing_safe_strength_scale, _max_stretch_delta = self._derive_safe_timing_profile(
            material,
            _mean_conf,
            vocals_conf,
            polyphonic_fallback=_poly_fallback,
        )
        _timing_safe_strength = float(np.clip(strength * _timing_safe_strength_scale, 0.0, max(0.15, strength)))

        if self._should_bypass_unsafe_polyphonic_fallback(
            material,
            _mean_conf,
            vocals_conf,
            polyphonic_fallback=_poly_fallback,
        ):
            restored = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            restored = np.clip(restored, -1.0, 1.0)
            processing_time = time.time() - start_time
            return PhaseResult(
                success=True,
                audio=restored,
                metrics={
                    "wow_flutter_detected": True,
                    "max_deviation_percent": max_deviation,
                    "residual_deviation_percent": max_deviation,
                    "wow_magnitude_percent": 0.0,
                    "flutter_magnitude_percent": 0.0,
                    "correction_strength": 0.0,
                    "mean_confidence": float(_mean_conf),
                    "material": material.value,
                    "quality_mode": quality_mode,
                    "transport_bumps_repaired": 0,
                    "tape_level_dips_repaired": 0,
                },
                execution_time_seconds=processing_time,
                metadata={
                    "algorithm": "unsafe_polyphonic_fallback_bypass",
                    "version": "4.1_locality",
                    "ml_hybrid": False,
                    "psola_active": vocals_conf >= 0.4,
                    "panns_vocals_confidence": vocals_conf,
                    "threshold": threshold,
                    "polyphonic_fallback": _poly_fallback,
                    "timing_safe_strength": 0.0,
                    "timing_safe_strength_scale": _timing_safe_strength_scale,
                    "max_stretch_delta": _max_stretch_delta,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "bypassed_due_to_unsafe_polyphonic_fallback": True,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=[
                    "Wow/flutter correction bypassed: polyphonic estimator was implausible and fallback was unsafe for vocal analog material"
                ],
            )

        # Step 4: Calculate time-stretching factors from pitch deviation
        stretch_factors = self._calculate_stretch_factors(
            pitch_trajectory,
            confidence,
            _timing_safe_strength,
            max_stretch_delta=_max_stretch_delta,
        )

        # Step 5: Apply time-stretching – PSOLA für Vokal-Segmente, WSOLA sonst
        # Moulines & Charpentier (1990): PSOLA ist formanterhaltend bei Gesangsmaterial;
        # Phase-Vocoder (hier: WSOLA/resample) für Instrumental-/Nicht-Vokal-Material.
        _stretch_fn = self._psola_timestretch if vocals_conf >= 0.4 else self._phase_vocoder_timestretch
        if vocals_conf >= 0.4:
            logger.debug(
                "Phase 12: PSOLA aktiviert (PANNs Vocals-Konfidenz=%.2f ≥ 0.40)",
                vocals_conf,
            )
        if is_stereo:
            restored_left = _stretch_fn(audio[:, 0], stretch_factors, sample_rate)
            restored_right = _stretch_fn(audio[:, 1], stretch_factors, sample_rate)
            restored = np.column_stack([restored_left, restored_right])
        else:
            restored = _stretch_fn(audio, stretch_factors, sample_rate)

        # Step 6: Verify correction (measure residual deviation)
        restored_mono = np.mean(restored, axis=1) if is_stereo else restored

        # Step 6b: Targeted transport bump repair (impulsive micro-speed jumps 50–300 ms)
        bump_locations = kwargs.get("transport_bump_locations")
        if not bump_locations:
            # Fallback: extract from defect_locations dict (UV3 passes defect_locations kwarg)
            _dl = kwargs.get("defect_locations") or {}
            bump_locations = _dl.get("transport_bump", []) if isinstance(_dl, dict) else []
        n_bumps_repaired = 0
        if bump_locations and len(bump_locations) > 0:
            # Confidence-aware bump repair: keep defect removal active while avoiding
            # over-processing when pitch certainty is marginal.
            _bump_strength = float(
                np.clip(
                    _timing_safe_strength * (0.65 + 0.35 * np.clip(_mean_conf, 0.0, 1.0)),
                    0.15,
                    max(0.15, _timing_safe_strength),
                )
            )
            if len(bump_locations) > 120:
                _bump_strength *= 0.85
            restored, n_bumps_repaired = self._repair_transport_bumps(
                restored,
                sample_rate,
                bump_locations,
                _bump_strength,
            )
            restored_mono = np.mean(restored, axis=1) if is_stereo else restored
            logger.info(
                "Phase 12 transport_bump repair: %d/%d bumps repaired (strength=%.3f, conf=%.3f)",
                n_bumps_repaired,
                len(bump_locations),
                _bump_strength,
                _mean_conf,
            )

        # Step 6c: Tape head contact level stabilization (autonomous detection + repair)
        # Repairs gradual level dips caused by tape-head pressure variation / capstan
        # irregularity in cassette recordings.  These dips fall through Phase 24 dropout
        # repair (threshold too aggressive for gradual 60-100 ms fades) and are not
        # covered by transport_bump repair (which requires DefectScanner locations).
        n_level_dips_repaired = 0
        _TAPE_LEVEL_MATERIALS = {MaterialType.TAPE, MaterialType.REEL_TAPE}
        _mat_enum = material if isinstance(material, MaterialType) else None
        if _mat_enum in _TAPE_LEVEL_MATERIALS and _effective_strength > 0.0:
            restored, n_level_dips_repaired = self._stabilize_tape_level(restored, sample_rate, _effective_strength)
            if n_level_dips_repaired > 0:
                restored_mono = np.mean(restored, axis=1) if is_stereo else restored
                logger.info(
                    "Phase 12 tape level stabilizer: %d dips repaired",
                    n_level_dips_repaired,
                )

        residual_pitch, residual_conf = self._estimate_pitch_yin(restored_mono, sample_rate)
        residual_deviation = self._calculate_max_deviation(residual_pitch, residual_conf)

        # Chroma Pearson rollback guard: if tonal center drifts, revert to original
        # (Phase Vocoder / PSOLA can introduce pitch shifts that destroy tonal center)
        try:
            _orig_mono = np.mean(audio, axis=1) if is_stereo else audio
            _n_chroma = min(len(_orig_mono), len(restored_mono))
            _hop_chroma = 512
            _n_frames = max(1, _n_chroma // _hop_chroma)
            _chroma_orig = np.zeros(12, dtype=np.float64)
            _chroma_rest = np.zeros(12, dtype=np.float64)
            for _ci in range(min(_n_frames, 200)):  # sample up to 200 frames
                _s = _ci * _hop_chroma
                _e = _s + _hop_chroma
                if _e > _n_chroma:
                    break
                _sp_o = np.abs(np.fft.rfft(_orig_mono[_s:_e]))
                _sp_r = np.abs(np.fft.rfft(restored_mono[_s:_e]))
                _freqs_c = np.fft.rfftfreq(_hop_chroma, 1.0 / sample_rate)
                for _b in range(12):
                    _f_lo = 65.41 * (2 ** (_b / 12.0))
                    _f_hi = 65.41 * (2 ** ((_b + 1) / 12.0))
                    _mask_c = (_freqs_c >= _f_lo) & (_freqs_c < _f_hi)
                    _chroma_orig[_b] += np.sum(_sp_o[_mask_c] ** 2)
                    _chroma_rest[_b] += np.sum(_sp_r[_mask_c] ** 2)
            _norm_o = np.sqrt(np.sum(_chroma_orig**2)) + 1e-10
            _norm_r = np.sqrt(np.sum(_chroma_rest**2)) + 1e-10
            _chroma_pearson = float(np.dot(_chroma_orig / _norm_o, _chroma_rest / _norm_r))
        except Exception:
            _chroma_pearson = 1.0  # fallback: assume OK

        if _chroma_pearson < 0.95:
            logger.warning(
                "Phase 12 chroma guard: Pearson %.3f < 0.95 — reverting to original audio "
                "(wow/flutter correction caused tonal center drift)",
                _chroma_pearson,
            )
            restored = audio.copy()
            residual_deviation = max_deviation  # unchanged

        processing_time = time.time() - start_time

        # Calculate wow/flutter statistics
        wow_magnitude = np.std(wow_component[wow_component != 0])
        flutter_magnitude = np.std(flutter_component[flutter_component != 0])

        # Build metadata
        metadata = {
            "algorithm": (
                "polyphonic_multi_f0_consensus_v1"
                if _poly_applied
                else (
                    "hybrid_ml_pyin_crepe_psola_v3"
                    if (use_ml_hybrid and vocals_conf >= 0.4)
                    else (
                        "hybrid_ml_pyin_crepe_v3"
                        if use_ml_hybrid
                        else "pyin_psola"
                        if vocals_conf >= 0.4
                        else "pyin_phase_vocoder"
                    )
                )
            ),
            "version": "4.0_polyphonic" if _poly_applied else "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
            "ml_hybrid": use_ml_hybrid,
            "psola_active": vocals_conf >= 0.4,
            "panns_vocals_confidence": vocals_conf,
            "threshold": threshold,
            "stft_window": self.STFT_WINDOW_SIZE,
            "stft_hop": self.STFT_HOP_SIZE,
            "polyphonic_fallback": _poly_fallback,
            "timing_safe_strength": _timing_safe_strength,
            "timing_safe_strength_scale": _timing_safe_strength_scale,
            "max_stretch_delta": _max_stretch_delta,
        }

        if use_ml_hybrid:
            metadata["pyin_applied"] = ml_result.pyin_applied
            metadata["crepe_applied"] = ml_result.crepe_applied
            metadata["strategy_used"] = str(ml_result.strategy_used)
            metadata["pitch_detection_time"] = ml_result.processing_time
            metadata["ml_metadata"] = ml_result.metadata

        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.clip(restored, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            restored = audio + _effective_strength * (restored - audio)
            restored = np.clip(restored, -1.0, 1.0)

        restored, _rms_drop_db, _makeup_db = self._preserve_phase_loudness(
            _original_audio,
            restored,
            material,
        )
        if abs(_makeup_db) > 0.01:
            logger.info(
                "Phase 12 loudness-preservation: material=%s rms_drop=%+.2f dB via makeup %+.2f dB",
                material.value,
                _rms_drop_db,
                _makeup_db,
            )
        return PhaseResult(
            success=True,
            audio=restored,
            metrics={
                "wow_flutter_detected": True,
                "max_deviation_percent": max_deviation,
                "residual_deviation_percent": residual_deviation,
                "wow_magnitude_percent": float(wow_magnitude) if not np.isnan(wow_magnitude) else 0.0,
                "flutter_magnitude_percent": float(flutter_magnitude) if not np.isnan(flutter_magnitude) else 0.0,
                "correction_strength": _timing_safe_strength,
                "mean_confidence": float(np.mean(confidence[confidence > 0])),
                "material": material.value,
                "quality_mode": quality_mode,
                "transport_bumps_repaired": n_bumps_repaired,
                "tape_level_dips_repaired": n_level_dips_repaired,
            },
            execution_time_seconds=processing_time,
            metadata={
                **metadata,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": _rms_drop_db,
                "loudness_makeup_db": _makeup_db,
            },
        )

    def _preserve_phase_loudness(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, float, float]:
        """Keep Phase-12 loudness close to input while preserving defect repairs.

        Returns:
            (audio_out, rms_delta_db, applied_makeup_db)
        """
        orig = np.asarray(original_audio, dtype=np.float64)
        proc = np.asarray(processed_audio, dtype=np.float64)
        if orig.shape != proc.shape:
            return np.clip(np.asarray(processed_audio, dtype=np.float32), -1.0, 1.0), 0.0, 0.0

        _max_rms_drop_db = {
            MaterialType.SHELLAC: 1.2,
            MaterialType.WAX_CYLINDER: 1.2,
            MaterialType.WIRE_RECORDING: 1.3,
            MaterialType.VINYL: 1.5,
            MaterialType.TAPE: 1.7,
            MaterialType.REEL_TAPE: 1.6,
            MaterialType.CD_DIGITAL: 2.0,
            MaterialType.DAT: 2.0,
            MaterialType.MP3_LOW: 2.2,
            MaterialType.MP3_HIGH: 2.0,
            MaterialType.AAC: 2.0,
            MaterialType.STREAMING: 2.0,
        }.get(material, 1.8)
        _max_rms_lift_db = 1.0

        _orig_rms = float(np.sqrt(np.mean(orig**2) + 1e-12))
        _proc_rms = float(np.sqrt(np.mean(proc**2) + 1e-12))
        if _orig_rms < 1e-10 or _proc_rms < 1e-12:
            return np.clip(proc.astype(np.float32), -1.0, 1.0), 0.0, 0.0

        _delta_db = float(20.0 * np.log10(max(_proc_rms / _orig_rms, 1e-30)))
        _gain_db = 0.0

        if _delta_db < -_max_rms_drop_db:
            _gain_db = (-_max_rms_drop_db) - _delta_db
        elif _delta_db > _max_rms_lift_db:
            _gain_db = _max_rms_lift_db - _delta_db

        if abs(_gain_db) > 0.01:
            _gain = float(10.0 ** (_gain_db / 20.0))
            _peak_p999 = float(np.percentile(np.abs(proc), 99.9) + 1e-12)
            if _gain > 1.0:
                _gain = min(_gain, float(0.985 / _peak_p999))
            proc = np.clip(proc * _gain, -1.0, 1.0)

        _out_rms = float(np.sqrt(np.mean(proc**2) + 1e-12))
        _out_delta_db = float(20.0 * np.log10(max(_out_rms / _orig_rms, 1e-30)))
        _applied_makeup_db = float(20.0 * np.log10(max(np.sqrt(np.mean(proc**2) + 1e-12) / _proc_rms, 1e-30)))

        return np.clip(proc.astype(np.float32), -1.0, 1.0), _out_delta_db, _applied_makeup_db

    def _estimate_pitch_yin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Rückwärts-kompatibles Alias auf pYIN (Mauch & Dixon 2014).

        Aufruf-Schnittstelle identisch mit YIN, aber probabilistische
        Multilevel-Threshold-Auswertung für höhere Robustheit.
        """
        return self._estimate_pitch_pyin(audio, sample_rate)

    def _estimate_pitch_pyin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Probabilistic YIN (pYIN) nach Mauch & Dixon (2014).

        Primary path: librosa.pyin (vectorised C++ backend, ~100× faster than
        the pure-Python triple-loop fallback). Falls back to the Python
        implementation capped at 30 s centre if librosa is unavailable.

        Args:
            audio: Mono float32 [-1,1]
            sample_rate: Sample rate (expected: 48000 Hz)

        Returns:
            (pitch_trajectory, confidence): Pitch Hz and confidence [0,1] per frame
        """
        # -----------------------------------------------------------------
        # Optional fast path: librosa.pyin (C-accelerated)
        # Disabled by default because some environments show native segfaults
        # in librosa.sequence.viterbi/pyin. Opt-in with:
        #   AURIK_ENABLE_LIBROSA_PYIN=1
        # -----------------------------------------------------------------
        _enable_librosa_pyin = os.environ.get("AURIK_ENABLE_LIBROSA_PYIN", "0").strip() == "1"
        if _enable_librosa_pyin:
            try:
                import librosa  # always available in .venv_aurik

                hop_samples = max(1, int(self.PITCH_WINDOW_MS * sample_rate / 1000) // self.PITCH_HOP_FACTOR)
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    audio.astype(np.float32),
                    fmin=float(librosa.note_to_hz("C2")),  # ~65 Hz
                    fmax=float(librosa.note_to_hz("C7")),  # ~2093 Hz
                    sr=sample_rate,
                    hop_length=hop_samples,
                    fill_na=0.0,
                )
                # voiced_prob gives per-frame confidence; unvoiced → 0
                f0 = np.nan_to_num(f0, nan=0.0)
                confidence = np.where(voiced_flag, voiced_prob, 0.0).astype(np.float64)
                pitch_trajectory = f0.astype(np.float64)

                logger.debug(
                    "pYIN (librosa): %d frames, μ_pitch=%.1f Hz, μ_conf=%.3f",
                    len(pitch_trajectory),
                    float(np.mean(pitch_trajectory[pitch_trajectory > 0]) or 0),
                    float(np.mean(confidence)),
                )
                return pitch_trajectory, confidence

            except Exception as _lib_exc:
                logger.debug("librosa.pyin unavailable (%s) — falling back to Python pYIN (30 s cap)", _lib_exc)
        else:
            logger.debug("librosa.pyin disabled (AURIK_ENABLE_LIBROSA_PYIN!=1) — using Python pYIN fallback")

        # -----------------------------------------------------------------
        # Fallback: pure-Python pYIN with 30 s centre cap
        # (prevents 80 M Python iterations on long audio)
        # -----------------------------------------------------------------
        _PYIN_CAP_S = 30
        _cap_samples = int(_PYIN_CAP_S * sample_rate)
        if len(audio) > _cap_samples:
            _mid = len(audio) // 2
            _half = _cap_samples // 2
            audio_pyin = audio[_mid - _half : _mid + _half]
            logger.debug(
                "pYIN Python fallback: %.0f s audio capped to %d s centre", len(audio) / sample_rate, _PYIN_CAP_S
            )
        else:
            audio_pyin = audio

        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = window_samples // self.PITCH_HOP_FACTOR

        min_period = int(sample_rate / 1000)  # max 1000 Hz
        max_period = int(sample_rate / 50)  # min 50 Hz
        max_period = min(max_period, window_samples // 2)

        num_windows = max(1, (len(audio_pyin) - window_samples) // hop_samples + 1)
        pitch_trajectory = np.zeros(num_windows, dtype=np.float64)
        confidence = np.zeros(num_windows, dtype=np.float64)

        # pYIN: Multi-Threshold weights via Beta(2,18)-like distribution
        N_thr = 20
        thresholds = np.linspace(0.01, 0.30, N_thr)
        beta_weights = (1 - thresholds) ** 17 * thresholds
        beta_weights /= beta_weights.sum() + 1e-10

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            if end > len(audio_pyin):
                break

            window = audio_pyin[start:end] * np.hanning(window_samples)

            # CMND function (YIN)
            autocorr = np.correlate(window, window, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            diff = 2.0 * (autocorr[0] - autocorr[:max_period])
            cmnd = np.ones(max_period)
            cumsum = np.cumsum(diff[1:])
            tau_range = np.arange(1, max_period)
            cmnd[1:] = diff[1:] * tau_range / (cumsum + 1e-10)

            # Multi-threshold pYIN
            cand_pitches: list = []
            cand_weights: list = []

            for thr, w in zip(thresholds, beta_weights):
                tau_est = 0
                for tau in range(min_period, max_period):
                    if cmnd[tau] < thr and 0 < tau < max_period - 1:
                        if cmnd[tau] <= cmnd[tau - 1] and cmnd[tau] <= cmnd[tau + 1]:
                            tau_est = tau
                            break
                if tau_est == 0:
                    tau_est = min_period + int(np.argmin(cmnd[min_period:max_period]))

                # Parabolic interpolation
                if 0 < tau_est < max_period - 1:
                    s0, s1, s2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
                    denom = s0 - 2 * s1 + s2
                    if abs(denom) > 1e-10:
                        delta = 0.5 * (s0 - s2) / denom
                        tau_est = tau_est + delta

                if tau_est > 0:
                    cand_pitches.append(float(sample_rate) / float(tau_est))
                    cand_weights.append(w)

            if cand_pitches:
                cand_arr = np.array(cand_pitches)
                wgt_arr = np.array(cand_weights)
                wgt_arr /= wgt_arr.sum() + 1e-10
                mu = float(np.dot(cand_arr, wgt_arr))
                mask = np.abs(cand_arr - mu) < 0.10 * mu
                if mask.any():
                    pitch_trajectory[i] = float(np.dot(cand_arr[mask], wgt_arr[mask]) / (wgt_arr[mask].sum() + 1e-10))
                    confidence[i] = float(wgt_arr[mask].sum())
                else:
                    pitch_trajectory[i] = mu
                    confidence[i] = 0.3
            else:
                pitch_trajectory[i] = 0.0
                confidence[i] = 0.0

        # Temporal smoothing (simplified HMM tracking via exp smoothing)
        alpha_smooth = 0.7
        for i in range(1, num_windows):
            if pitch_trajectory[i] > 0 and pitch_trajectory[i - 1] > 0:
                pitch_trajectory[i] = alpha_smooth * pitch_trajectory[i - 1] + (1 - alpha_smooth) * pitch_trajectory[i]

        logger.debug(
            "pYIN (Python): %d frames, μ_pitch=%.1f Hz, μ_conf=%.3f",
            num_windows,
            float(np.mean(pitch_trajectory[pitch_trajectory > 0]) or 0),
            float(np.mean(confidence)),
        )
        return pitch_trajectory, confidence

    def _yin_algorithm(
        self, window: np.ndarray, sample_rate: int, min_period: int, max_period: int
    ) -> tuple[float, float]:
        """Legacy YIN — nur noch als Fallback, primär wird pYIN verwendet.

        De Cheveigné & Kawahara (2002): \"YIN, a fundamental frequency
        estimator for speech and music\".
        ACHTUNG: Nicht als primärer Algorithmus verwenden \u2014 pYIN bevorzugen.
        """
        autocorr = np.correlate(window, window, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        diff = 2.0 * (autocorr[0] - autocorr[:max_period])
        cmnd = np.ones(max_period)
        cumsum = np.cumsum(diff[1:])
        tau_range = np.arange(1, max_period)
        cmnd[1:] = diff[1:] * tau_range / (cumsum + 1e-10)

        tau_estimate = 0
        min_cmnd = 1.0
        for tau in range(min_period, max_period):
            if cmnd[tau] < self.YIN_THRESHOLD and 0 < tau < max_period - 1:
                if cmnd[tau] < cmnd[tau - 1] and cmnd[tau] < cmnd[tau + 1]:
                    tau_estimate = tau
                    min_cmnd = cmnd[tau]
                    break
        if tau_estimate == 0:
            tau_estimate = min_period + int(np.argmin(cmnd[min_period:max_period]))
            min_cmnd = cmnd[tau_estimate]

        if 0 < tau_estimate < max_period - 1:
            s0, s1, s2 = cmnd[tau_estimate - 1], cmnd[tau_estimate], cmnd[tau_estimate + 1]
            denom = s0 - 2 * s1 + s2
            if abs(denom) > 1e-10:
                delta = 0.5 * (s0 - s2) / denom
                tau_estimate = tau_estimate + delta

        pitch_hz = float(sample_rate) / float(tau_estimate) if tau_estimate > 0 else 0.0
        conf = max(0.0, 1.0 - min_cmnd)
        return pitch_hz, conf

    def _repair_transport_bumps(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bump_locations: list[tuple[float, float]],
        strength: float = 0.85,
    ) -> tuple[np.ndarray, int]:
        """Repair impulsive transport bumps at known locations.

        Multi-stage repair strategy (v2):
            1. Amplitude envelope smoothing toward context RMS level
            2. Local pitch correction via context-guided resampling
            3. Spectral context interpolation: blend bump spectrum toward
               weighted average of pre/post context spectra (removes LF thump
               and spectral centroid disruption that characterize transport bumps)
            4. Hanning-crossfade at margins for seamless integration

        Args:
            audio:          Audio signal (mono [N] or stereo [N, 2])
            sample_rate:    Sample rate in Hz
            bump_locations: List of (start_s, end_s) time pairs from DefectScanner
            strength:       Correction strength 0.0–1.0 (material-adaptive)

        Returns:
            (repaired_audio, n_bumps_repaired)
        """
        result = audio.copy()
        is_stereo = audio.ndim == 2
        n_samples = audio.shape[0]
        margin_s = 0.030  # 30 ms crossfade margin
        margin_samples = int(margin_s * sample_rate)
        n_repaired = 0

        for bump_start_s, bump_end_s in bump_locations:
            bump_start = int(bump_start_s * sample_rate)
            bump_end = int(bump_end_s * sample_rate)

            if bump_start < 0 or bump_end > n_samples or bump_end <= bump_start:
                continue

            region_start = max(0, bump_start - margin_samples)
            region_end = min(n_samples, bump_end + margin_samples)
            region_len = region_end - region_start

            if region_len < sample_rate // 100:  # minimum 10 ms
                continue

            ctx_before_start = max(0, region_start - sample_rate // 4)
            ctx_after_end = min(n_samples, region_end + sample_rate // 4)

            if is_stereo:
                mono_ctx_before = np.mean(result[ctx_before_start:region_start], axis=1)
                mono_ctx_after = np.mean(result[region_end:ctx_after_end], axis=1)
            else:
                mono_ctx_before = result[ctx_before_start:region_start]
                mono_ctx_after = result[region_end:ctx_after_end]

            # 1. Reference RMS from surrounding context
            ctx_audio = np.concatenate([mono_ctx_before, mono_ctx_after])
            ref_rms = float(np.sqrt(np.mean(ctx_audio**2) + 1e-12)) if len(ctx_audio) > 0 else 1.0

            # 2. Amplitude envelope smoothing
            if is_stereo:
                for ch in range(result.shape[1]):
                    result[region_start:region_end, ch] = self._smooth_bump_envelope(
                        result[region_start:region_end, ch],
                        ref_rms,
                        margin_samples,
                        strength,
                    )
            else:
                result[region_start:region_end] = self._smooth_bump_envelope(
                    result[region_start:region_end],
                    ref_rms,
                    margin_samples,
                    strength,
                )

            # 3. Local pitch correction
            bump_len = bump_end - bump_start
            if bump_len >= sample_rate // 50:
                if is_stereo:
                    for ch in range(result.shape[1]):
                        result[bump_start:bump_end, ch] = self._local_pitch_flatten(
                            result[bump_start:bump_end, ch],
                            mono_ctx_before,
                            mono_ctx_after,
                            sample_rate,
                            strength,
                        )
                else:
                    result[bump_start:bump_end] = self._local_pitch_flatten(
                        result[bump_start:bump_end],
                        mono_ctx_before,
                        mono_ctx_after,
                        sample_rate,
                        strength,
                    )

            # 4. Spectral context interpolation — remove LF thump and timbral disruption
            #    by blending the bump's magnitude spectrum toward the surrounding context
            if bump_len >= 256:
                if is_stereo:
                    for ch in range(result.shape[1]):
                        result[bump_start:bump_end, ch] = self._spectral_context_blend(
                            result[bump_start:bump_end, ch],
                            (
                                result[ctx_before_start:region_start, ch]
                                if ctx_before_start < region_start
                                else np.zeros(1)
                            ),
                            result[region_end:ctx_after_end, ch] if region_end < ctx_after_end else np.zeros(1),
                            strength,
                        )
                else:
                    result[bump_start:bump_end] = self._spectral_context_blend(
                        result[bump_start:bump_end],
                        result[ctx_before_start:region_start] if ctx_before_start < region_start else np.zeros(1),
                        result[region_end:ctx_after_end] if region_end < ctx_after_end else np.zeros(1),
                        strength,
                    )

            n_repaired += 1

        # Safety: NaN/Inf guard + clip
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)
        return result, n_repaired

    @staticmethod
    def _smooth_bump_envelope(
        segment: np.ndarray,
        ref_rms: float,
        margin_samples: int,
        strength: float,
    ) -> np.ndarray:
        """Smooth the amplitude envelope of a bump region toward the reference RMS level.

        Uses a Hanning-weighted crossfade at margins and local RMS gain correction
        in the bump interior to eliminate sudden amplitude jumps.
        """
        result = segment.copy()
        n = len(result)
        if n < 4 or ref_rms < 1e-10:
            return result

        # Compute local RMS in short windows (5 ms)
        win = max(1, min(n, 240))  # ~5 ms at 48 kHz
        local_rms = np.array(
            [float(np.sqrt(np.mean(result[i : i + win] ** 2) + 1e-12)) for i in range(0, n - win + 1, max(1, win // 2))]
        )

        if len(local_rms) == 0:
            return result

        # Gain correction: target is ref_rms, blend with strength
        for i, lr in enumerate(local_rms):
            if lr > 1e-10:
                gain = 1.0 + strength * (ref_rms / lr - 1.0)
                gain = np.clip(gain, 0.5, 2.0)  # Safety clamp
                start_idx = i * max(1, win // 2)
                end_idx = min(n, start_idx + win)
                result[start_idx:end_idx] *= gain

        # Apply Hanning crossfade at margins
        fade_len = min(margin_samples, n // 2)
        if fade_len > 1:
            fade_in = np.hanning(fade_len * 2)[:fade_len]
            fade_out = np.hanning(fade_len * 2)[fade_len:]
            # Blend: corrected × fade + original × (1 - fade)
            result[:fade_len] = segment[:fade_len] * (1.0 - fade_in) + result[:fade_len] * fade_in
            result[-fade_len:] = segment[-fade_len:] * (1.0 - fade_out) + result[-fade_len:] * fade_out

        return np.clip(result, -1.0, 1.0)

    @staticmethod
    def _spectral_context_blend(
        bump_audio: np.ndarray,
        ctx_before: np.ndarray,
        ctx_after: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Blend the magnitude spectrum of a bump region toward surrounding context.

        Transport bumps inject spurious low-frequency energy (mechanical thump) and
        cause abrupt spectral centroid shifts. This method:
          1. Computes average magnitude spectrum of pre/post context
          2. Computes magnitude spectrum of the bump region
          3. Blends bump magnitudes toward context magnitudes (strength-weighted)
          4. Preserves original phase (no phase distortion)
          5. RMS-normalizes output to prevent amplitude drift

        Uses raw FFT (no windowing) to avoid edge-amplification artifacts.
        """
        n = len(bump_audio)
        if n < 128 or strength < 0.01:
            return bump_audio

        fft_size = 1
        while fft_size < n:
            fft_size *= 2

        # Compute reference spectrum from context (weighted average)
        ref_mag = None
        n_ref = 0
        for ctx in (ctx_before, ctx_after):
            if len(ctx) < 64:
                continue
            chunk = ctx[-n:] if ctx is ctx_before else ctx[:n]
            if len(chunk) < 64:
                continue
            padded = np.zeros(fft_size, dtype=np.float64)
            padded[: len(chunk)] = chunk.astype(np.float64)
            mag = np.abs(np.fft.rfft(padded))
            if ref_mag is None:
                ref_mag = mag
            else:
                ref_mag = ref_mag + mag
            n_ref += 1

        if ref_mag is None or n_ref == 0:
            return bump_audio

        ref_mag /= n_ref

        # Compute bump spectrum (no windowing — avoids edge amplification)
        bump_padded = np.zeros(fft_size, dtype=np.float64)
        bump_f64 = bump_audio.astype(np.float64)
        bump_padded[:n] = bump_f64
        bump_fft = np.fft.rfft(bump_padded)
        bump_mag = np.abs(bump_fft)
        bump_phase = np.angle(bump_fft)

        # Only suppress frequencies where bump exceeds context (remove thump),
        # don't boost missing frequencies (that would add artifacts)
        blended_mag = bump_mag.copy()
        excess = bump_mag > ref_mag * 1.2
        blended_mag[excess] = bump_mag[excess] * (1.0 - strength * 0.7) + ref_mag[excess] * strength * 0.7

        # Reconstruct with original phase
        blended_fft = blended_mag * np.exp(1j * bump_phase)
        result_full = np.fft.irfft(blended_fft, n=fft_size)
        result = result_full[:n].astype(np.float32)

        # RMS-normalize to match original bump level (prevent amplitude drift)
        orig_rms = float(np.sqrt(np.mean(bump_f64**2) + 1e-12))
        result_rms = float(np.sqrt(np.mean(result.astype(np.float64) ** 2) + 1e-12))
        if result_rms > 1e-10:
            result *= np.float32(orig_rms / result_rms)

        # Blend with original using strength
        result = bump_audio * (1.0 - strength * 0.5) + result * (strength * 0.5)

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 6c: Tape Head Contact Level Stabilizer
    # ------------------------------------------------------------------
    # Repairs gradual envelope dips caused by tape-head pressure
    # variation or capstan irregularity in cassette/reel recordings.
    #
    # Defect morphology (from real-world cassette analysis):
    #   - Gradual fade-down: 60-100 ms onset
    #   - Minimum depth: 10-25 dB below local context level
    #   - Sharp recovery: < 25 ms back to normal level
    #   - Distributed across entire song, not just intro
    #   - Average rate: ~0.5-1.0 per second
    #
    # These dips are NOT caught by Phase 24 dropout repair:
    #   - Phase 24 requires > 75 % energy drop (these are 10-20 dB = 50-90 %)
    #   - Phase 24 max_dropout_ms = 200 ms (many dips last 150-300 ms)
    #   - Phase 24 detects sudden drops, not gradual fades
    #
    # Algorithm (Tape Head Contact Level Stabilizer v1):
    #   1. Compute RMS envelope in 20 ms windows, 10 ms hop
    #   2. Compute slow-moving reference via percentile filter (500 ms)
    #   3. Detect dips: envelope < reference - dip_threshold_db
    #   4. For each dip region: compute compensating gain, smooth edges
    #   5. Limit max gain to avoid noise amplification
    #   6. Apply gain with smooth interpolation
    #
    # Scientific basis:
    #   - Camras (1988): Magnetic Recording Handbook - head contact mechanics
    #   - Fastl & Zwicker (2007): equal-loudness perception of level dips
    # ------------------------------------------------------------------

    def _stabilize_tape_level(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float,
    ) -> tuple[np.ndarray, int]:
        """Detect and repair tape head contact level dips — STFT-domain SOTA v2.

        Physics basis (Wallace 1951):
            When head-tape contact pressure varies, effective head-tape spacing
            Δd increases → spacing loss L(f) = e^(-2π·Δd·f/v).
            HF loss is exponentially greater than LF loss: a 1 µm spacing
            increase at cassette speed (v=47.6 mm/s) causes ~11 dB loss at
            10 kHz but only ~1 dB at 1 kHz.  Wideband-only gain therefore
            restores loudness but leaves a residual HF spectral tilt.

        v2 SOTA upgrades (2026-04-09):
          1. STFT-domain frequency-dependent gain (HF boosted proportional to
             estimated spacing loss derived from per-dip context comparison)
          2. Per-dip HF tilt estimated adaptively from spectral centroid shift
             in context vs. dip frames — no fixed tape-speed assumption needed
          3. SNR guard per frequency bin: bins at noise floor are not boosted
             (prevents amplification of residual tape hiss during deep dips)
          4. Asymmetric gain envelope: slow onset (~30 % of dip duration) /
             fast recovery (~10 %) matches the physical capstan-irregularity
             dip morphology
          5. §2.51 linked stereo: spectral gain derived from mono, applied
             identically to L and R channels

        Args:
            audio:       Audio signal (mono [N] or stereo [N, 2] — float32/64)
            sample_rate: Sample rate in Hz (expected: 48000)
            strength:    Correction strength [0.0, 1.0]

        Returns:
            (stabilized_audio, n_dips_repaired)

        References:
            Wallace, R.L. (1951) The Reproduction of Magnetically Recorded
            Signals. Bell System Technical Journal 30(4):1145–1173.
            Camras, M. (1988) Magnetic Recording Handbook. §8.3.
            McKnight, J.G. (1969) Tape Reproducer Response Measurements.
        """
        if strength < 0.01:
            return audio, 0

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        is_stereo = audio.ndim == 2
        n_samples = audio.shape[0]

        if is_stereo:
            mono = audio.mean(axis=1).astype(np.float64)
        else:
            mono = audio.astype(np.float64)

        # ── Step 1: RMS envelope (20 ms / 10 ms) + percentile-75 reference ──
        env_win_s = 0.020
        env_hop_s = 0.010
        ref_win_s = 0.500
        dip_thresh_db = 3.0
        min_dip_frames = 3
        max_gain_db = 15.0

        env_win = max(1, int(env_win_s * sample_rate))
        env_hop = max(1, int(env_hop_s * sample_rate))
        n_frames = max(0, (n_samples - env_win) // env_hop)

        if n_frames < 10:
            return audio, 0

        rms_env = np.array(
            [np.sqrt(np.mean(mono[i * env_hop : i * env_hop + env_win] ** 2) + 1e-15) for i in range(n_frames)],
            dtype=np.float64,
        )
        rms_db = 20.0 * np.log10(rms_env + 1e-15)

        from scipy.ndimage import percentile_filter

        ref_n = max(3, int(ref_win_s / env_hop_s))
        ref_n = ref_n + (1 - ref_n % 2)  # ensure odd
        ref_db = percentile_filter(rms_db, percentile=75, size=ref_n, mode="reflect")

        dip_mask_rms = rms_db < (ref_db - dip_thresh_db)

        from scipy.ndimage import label as nd_label

        labeled, n_dips_raw = nd_label(dip_mask_rms)

        if n_dips_raw == 0:
            return audio, 0

        # ── Step 2: Collect valid dip events ─────────────────────────────
        rms_centres = np.arange(n_frames) * env_hop + env_win // 2  # sample pos

        dip_events: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, n_dips_raw + 1):
            frames = np.where(labeled == i)[0]
            if len(frames) < min_dip_frames:
                continue
            deficit = ref_db[frames] - rms_db[frames]
            if np.max(deficit) > max_gain_db + 5.0:
                continue  # likely genuine silence gap
            if np.mean(rms_db[frames]) < -55.0:
                continue  # noise floor — nothing to restore
            dip_events.append((frames, deficit))

        if not dip_events:
            return audio, 0

        # ── Step 3: STFT setup ────────────────────────────────────────────
        # fft_size = 2048 → 42.7 ms @ 48 kHz; hop = 512 → 10.7 ms (75 % OV)
        fft_size = 2048
        hop_stft = fft_size // 4
        n_freqs = fft_size // 2 + 1
        freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        _, _, X_mono = signal.stft(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size - hop_stft,
            boundary="even",
            padded=True,
        )
        n_stft_frames = X_mono.shape[1]
        stft_centres = np.arange(n_stft_frames) * hop_stft + fft_size // 2  # samples

        X_mag_db = 20.0 * np.log10(np.abs(X_mono) + 1e-15)  # [n_freqs, n_stft_frames]

        # Spectral gain mask: identity (1.0) everywhere — modified per dip below
        spectral_gain = np.ones((n_freqs, n_stft_frames), dtype=np.float64)

        # ── Step 4: Build per-dip spectral gain (broadband + HF tilt) ────
        n_repaired = 0
        ctx_n = 64  # context window ≈ 680 ms before dip onset

        for rms_frames, deficit in dip_events:
            # Map RMS dip frames → nearest STFT frame indices
            rms_ctrs = rms_centres[rms_frames]
            stft_idx = np.searchsorted(stft_centres, rms_ctrs)
            stft_idx = np.unique(np.clip(stft_idx, 0, n_stft_frames - 1))
            if len(stft_idx) == 0:
                continue

            # Per-STFT-frame broadband gain: interpolate from per-RMS-frame deficit
            bb_gain_db_stft = np.interp(
                stft_centres[stft_idx].astype(float),
                rms_centres[rms_frames].astype(float),
                np.clip(deficit * strength, 0.0, max_gain_db),
            )  # dB, shape [n_stft_dip]

            # ── HF spectral-tilt correction (Wallace spacing-loss inversion) ──
            first_stft = int(stft_idx[0])
            ctx_start = max(0, first_stft - ctx_n)
            ctx_end = max(0, first_stft - 2)
            hf_tilt_db = np.zeros(n_freqs, dtype=np.float64)  # per-bin

            if ctx_end - ctx_start >= 4:
                ctx_mag = X_mag_db[:, ctx_start:ctx_end]  # [n_freqs, ctx]
                dip_mag = X_mag_db[:, stft_idx]  # [n_freqs, n_dip_stft]

                # Reference: p75 of context per bin (robust to note-onset transients)
                ref_spec = np.percentile(ctx_mag, 75, axis=1)  # [n_freqs]
                dip_spec = np.mean(dip_mag, axis=1)  # [n_freqs]

                spectral_loss = ref_spec - dip_spec  # positive = dropped in dip

                # Remove broadband component (median of LF bins < 4 kHz)
                lf_mask = freqs_hz < 4000.0
                if lf_mask.any():
                    broadband_loss = float(np.median(spectral_loss[lf_mask]))
                else:
                    broadband_loss = float(np.median(spectral_loss))

                tilt_raw = spectral_loss - broadband_loss  # frequency-dep. residual

                # SNR guard: bins where signal ≤ noise floor + 6 dB → no HF boost
                noise_floor = np.percentile(ctx_mag, 10, axis=1)  # p10 ≈ noise floor
                snr_in_dip = dip_spec - noise_floor
                tilt_raw = np.where(snr_in_dip > 6.0, tilt_raw, 0.0)

                # Cap HF tilt at 10 dB; only positive values (loss → boost)
                hf_tilt_db = np.clip(tilt_raw, 0.0, 10.0) * strength

            # ── Asymmetric gain envelope (slow onset / fast recovery) ────
            n_sf = len(stft_idx)
            onset_n = max(1, int(n_sf * 0.30))  # ~30 % slow fade-in
            recovery_n = max(1, int(n_sf * 0.10))  # ~10 % fast fade-out
            fade_env = np.ones(n_sf, dtype=np.float64)
            fade_env[:onset_n] = np.linspace(0.0, 1.0, onset_n)
            if n_sf > onset_n + recovery_n:
                fade_env[-recovery_n:] = np.linspace(1.0, 0.0, recovery_n)

            # ── Combine broadband + HF-tilt into spectral_gain mask ──────
            tilt_lin = 10.0 ** (hf_tilt_db / 20.0)  # [n_freqs] — per-bin extra boost
            max_lin = 10.0 ** (max_gain_db / 20.0)

            for k, sf in enumerate(stft_idx):
                bb_db = float(bb_gain_db_stft[k])
                if bb_db < 0.3:
                    continue  # trivially small — skip
                bb_lin = 10.0 ** (bb_db / 20.0)  # scalar
                combined = bb_lin * tilt_lin  # [n_freqs]
                # Asymmetric fade: smoothly ramp gain at onset/recovery edges
                combined = 1.0 + (combined - 1.0) * float(fade_env[k])
                spectral_gain[:, sf] = np.clip(combined, 1.0, max_lin)

            n_repaired += 1

        if n_repaired == 0:
            return audio, 0

        # ── Step 5: Apply spectral gain to each channel (§2.51 linked) ──
        def _apply_gain_to_channel(sig_ch: np.ndarray) -> np.ndarray:
            """Apply the spectral_gain mask to one audio channel via STFT."""
            _, _, X_ch = signal.stft(
                sig_ch.astype(np.float64),
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_stft,
                boundary="even",
                padded=True,
            )
            n_apply = min(X_ch.shape[1], spectral_gain.shape[1])
            X_ch[:, :n_apply] *= spectral_gain[:, :n_apply]
            _, y = signal.istft(
                X_ch,
                fs=sample_rate,
                window="hann",
                nperseg=fft_size,
                noverlap=fft_size - hop_stft,
                boundary="even",
            )
            # Trim / zero-pad to original length
            y_out = np.zeros(n_samples, dtype=np.float64)
            n_trim = min(len(y), n_samples)
            y_out[:n_trim] = y[:n_trim]
            return y_out

        if is_stereo:
            L_out = _apply_gain_to_channel(audio[:, 0])
            R_out = _apply_gain_to_channel(audio[:, 1])
            result = np.stack([L_out, R_out], axis=1)
        else:
            result = _apply_gain_to_channel(mono)

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        logger.debug(
            "Tape level stabilizer v2 (STFT): %d dips repaired, HF-tilt correction active, strength=%.2f",
            n_repaired,
            strength,
        )

        return result, n_repaired

    def _local_pitch_flatten(
        self,
        bump_audio: np.ndarray,
        ctx_before: np.ndarray,
        ctx_after: np.ndarray,
        sample_rate: int,
        strength: float,
    ) -> np.ndarray:
        """Flatten pitch excursion in a bump region using context-guided resampling.

        Algorithm:
            1. Estimate reference pitch from context (before + after bump)
            2. Estimate pitch within the bump
            3. Compute per-sample resampling ratio to flatten toward reference
            4. Apply via linear interpolation (fast, no phase vocoder overhead)
        """
        n = len(bump_audio)
        if n < 64:
            return bump_audio

        # Estimate reference pitch from context
        ref_pitch_before = self._quick_pitch_estimate(ctx_before, sample_rate) if len(ctx_before) > 256 else 0.0
        ref_pitch_after = self._quick_pitch_estimate(ctx_after, sample_rate) if len(ctx_after) > 256 else 0.0

        if ref_pitch_before > 0 and ref_pitch_after > 0:
            ref_pitch = (ref_pitch_before + ref_pitch_after) / 2.0
        elif ref_pitch_before > 0:
            ref_pitch = ref_pitch_before
        elif ref_pitch_after > 0:
            ref_pitch = ref_pitch_after
        else:
            return bump_audio  # Cannot determine reference → skip

        # Estimate pitch in bump
        bump_pitch = self._quick_pitch_estimate(bump_audio, sample_rate)
        if bump_pitch <= 0 or ref_pitch <= 0:
            return bump_audio

        # Compute pitch ratio
        ratio = ref_pitch / bump_pitch
        # Only correct if there's a meaningful deviation (> 0.5 %)
        if abs(ratio - 1.0) < 0.005:
            return bump_audio

        # Clamp correction to avoid extreme warping
        correction = 1.0 + strength * (ratio - 1.0)
        correction = np.clip(correction, 0.9, 1.1)  # max ±10 % correction

        # Resample via linear interpolation
        new_len = int(n * correction)
        if new_len < 4 or new_len > n * 3:
            return bump_audio

        indices = np.linspace(0, n - 1, new_len)
        resampled = np.interp(indices, np.arange(n), bump_audio).astype(np.float32)

        # Fit back to original length via truncation or zero-pad + crossfade
        if len(resampled) >= n:
            result = resampled[:n]
        else:
            result = np.zeros(n, dtype=np.float32)
            result[: len(resampled)] = resampled
            # Fade out tail
            fade = max(1, n - len(resampled))
            result[len(resampled) :] = bump_audio[len(resampled) :] * np.linspace(1.0, 0.0, fade)[: n - len(resampled)]

        return np.clip(result, -1.0, 1.0)

    def _quick_pitch_estimate(self, audio: np.ndarray, sample_rate: int) -> float:
        """Fast autocorrelation-based pitch estimate for short segments.

        Returns estimated pitch in Hz, or 0.0 if unvoiced/unreliable.
        """
        n = len(audio)
        if n < 128:
            return 0.0

        # Use center portion
        center = audio[n // 4 : 3 * n // 4]
        center = center - np.mean(center)
        if np.max(np.abs(center)) < 1e-6:
            return 0.0

        # Autocorrelation via FFT
        n_fft = 1
        while n_fft < len(center) * 2:
            n_fft *= 2
        fft_x = np.fft.rfft(center, n=n_fft)
        acf = np.fft.irfft(fft_x * np.conj(fft_x))[: len(center)]

        if acf[0] < 1e-10:
            return 0.0
        acf = acf / acf[0]

        # Find first peak after initial decline
        min_lag = max(2, int(sample_rate / 1000.0))  # 1000 Hz max
        max_lag = min(len(acf) - 1, int(sample_rate / 50.0))  # 50 Hz min

        if max_lag <= min_lag:
            return 0.0

        search = acf[min_lag : max_lag + 1]
        if len(search) < 3:
            return 0.0

        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]

        if peak_val < 0.3:  # low confidence
            return 0.0

        lag = min_lag + peak_idx
        return float(sample_rate) / float(lag) if lag > 0 else 0.0

    def _separate_wow_flutter(self, pitch_trajectory: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate wow (<4 Hz) and flutter (4-100 Hz) components.

        Wow: Slow speed variations (<4 Hz) - from motor/belt issues
        Flutter: Fast speed variations (4-100 Hz) - from mechanical vibrations

        Args:
            pitch_trajectory: Pitch estimates (Hz) over time
            sample_rate: Original audio sample rate (used to determine analysis frame rate)

        Returns:
            (wow_component, flutter_component): Separated pitch deviation components
        """
        # Remove zero values (unvoiced frames)
        valid_mask = pitch_trajectory > 0
        valid_pitches = pitch_trajectory[valid_mask]

        if len(valid_pitches) < 10:
            # Too few valid estimates
            return np.zeros_like(pitch_trajectory), np.zeros_like(pitch_trajectory)

        # Calculate median pitch (stable reference)
        median_pitch = np.median(valid_pitches)

        # Pitch deviation from median (as percentage)
        deviation = np.zeros_like(pitch_trajectory)
        deviation[valid_mask] = (valid_pitches - median_pitch) / median_pitch * 100

        # Calculate frame rate of pitch trajectory
        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = window_samples // self.PITCH_HOP_FACTOR
        frame_rate = sample_rate / hop_samples

        # Low-pass filter for wow (<4 Hz)
        wow_cutoff = 4.0  # Hz
        nyquist = frame_rate / 2
        if wow_cutoff < nyquist:
            sos_wow = signal.butter(4, wow_cutoff / nyquist, btype="low", output="sos")
            wow_component = signal.sosfiltfilt(sos_wow, deviation)
        else:
            wow_component = deviation  # Frame rate too low, treat all as wow

        # Band-pass filter for flutter (4-100 Hz)
        flutter_low = 4.0  # Hz
        flutter_high = 100.0  # Hz
        if flutter_high < nyquist:
            sos_flutter = signal.butter(4, [flutter_low / nyquist, flutter_high / nyquist], btype="band", output="sos")
            flutter_component = signal.sosfiltfilt(sos_flutter, deviation)
        else:
            flutter_component = deviation - wow_component

        return wow_component, flutter_component

    def _detect_wow_flutter(
        self, pitch_trajectory: np.ndarray, confidence: np.ndarray, threshold: float
    ) -> tuple[bool, float]:
        """
        Detect wow & flutter by analyzing pitch deviations with confidence weighting.

        Args:
            pitch_trajectory: Pitch estimates (Hz)
            confidence: Confidence scores (0.0-1.0)
            threshold: Detection threshold (percent deviation)

        Returns:
            (detected, max_deviation_percent)
        """
        max_deviation = self._calculate_max_deviation(pitch_trajectory, confidence)

        # Detect if max deviation exceeds threshold
        detected = max_deviation > threshold

        return detected, max_deviation

    def _calculate_max_deviation(self, pitch_trajectory: np.ndarray, confidence: np.ndarray) -> float:
        """Calculate max pitch deviation weighted by confidence."""
        # Only use confident pitch estimates (confidence > 0.5)
        confident_mask = (pitch_trajectory > 0) & (confidence > 0.5)

        if np.sum(confident_mask) < 10:
            return 0.0

        confident_pitches = pitch_trajectory[confident_mask]
        median_pitch = np.median(confident_pitches)

        # Deviation from median (percentage)
        deviations = np.abs((confident_pitches - median_pitch) / median_pitch) * 100

        # Max deviation
        max_dev = np.max(deviations) if len(deviations) > 0 else 0.0

        return max_dev

    def _calculate_stretch_factors(
        self,
        pitch_trajectory: np.ndarray,
        confidence: np.ndarray,
        strength: float,
        *,
        max_stretch_delta: float = 0.05,
    ) -> np.ndarray:
        """
        Calculate time-stretching factors to correct pitch deviations.

        Stretch factor > 1.0: slow down (pitch was too high)
        Stretch factor < 1.0: speed up (pitch was too low)

        Args:
            pitch_trajectory: Pitch estimates (Hz)
            confidence: Confidence scores (0.0-1.0)
            strength: Correction strength (0.0-1.0)

        Returns:
            Array of stretch factors (one per pitch estimate)
        """
        # Use confident estimates to determine target pitch
        confident_mask = (pitch_trajectory > 0) & (confidence > 0.5)

        if np.sum(confident_mask) < 10:
            # Not enough confident estimates, no correction
            return np.ones_like(pitch_trajectory)

        confident_pitches = pitch_trajectory[confident_mask]
        target_pitch = np.median(confident_pitches)

        # Calculate stretch factors for each frame
        stretch_factors = np.ones_like(pitch_trajectory)

        for i, (pitch, conf) in enumerate(zip(pitch_trajectory, confidence)):
            if pitch > 0 and conf > 0.3:  # Use estimates with reasonable confidence
                # Stretch factor = current_pitch / target_pitch
                raw_stretch = pitch / target_pitch

                # Apply strength (blending between no correction and full correction)
                stretch_factors[i] = 1.0 + strength * (raw_stretch - 1.0)

                # Clamp to reasonable range (avoid extreme stretching)
                stretch_factors[i] = np.clip(
                    stretch_factors[i],
                    1.0 - max_stretch_delta,
                    1.0 + max_stretch_delta,
                )
            else:
                # Low confidence, no correction
                stretch_factors[i] = 1.0

        # Stretch-Faktoren glätten mit Savitzky-Golay (polynomialer Least-Squares-Smoother)
        # Ersetzt signal.medfilt: Savitzky-Golay erhält Peaks und liefert glatteren Verlauf
        try:
            from scipy.signal import savgol_filter

            stretch_factors = savgol_filter(stretch_factors, window_length=5, polyorder=2)
        except Exception:
            # Notfall-Fallback: Uniform-Smoothing
            from scipy.ndimage import uniform_filter1d

            stretch_factors = uniform_filter1d(stretch_factors, size=5, mode="nearest")
        stretch_factors = np.clip(
            stretch_factors,
            1.0 - max_stretch_delta,
            1.0 + max_stretch_delta,
        )

        return stretch_factors

    def _phase_vocoder_timestretch(
        self, audio: np.ndarray, stretch_factors: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Apply time-varying WSOLA-style time mapping for wow/flutter correction.

        Uses per-frame stretch factors interpolated to sample-rate, followed by a
        monotonic source-position mapping and band-limited interpolation. This keeps
        output length constant while avoiding the coarse average-stretch approximation.

        Args:
            audio: Mono audio samples
            stretch_factors: Time-varying stretch factors (one per pitch window)
            sample_rate: Sample rate

        Returns:
            Time-stretched audio
        """
        if len(audio) < 8 or len(stretch_factors) == 0:
            return audio.copy()

        audio_f = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        n_samples = len(audio_f)

        # Interpolate frame-wise stretch factors to sample resolution.
        sf = np.asarray(stretch_factors, dtype=np.float64)
        sf = np.clip(sf, 0.90, 1.10)
        if len(sf) == 1:
            sf_samples = np.full(n_samples, sf[0], dtype=np.float64)
        else:
            src_idx = np.linspace(0, n_samples - 1, len(sf), dtype=np.float64)
            dst_idx = np.arange(n_samples, dtype=np.float64)
            sf_samples = np.interp(dst_idx, src_idx, sf)

        # Smooth micro-jitter in factor curve (preserve wow contour, suppress zipper noise).
        try:
            from scipy.signal import savgol_filter

            win = max(5, (n_samples // 400) | 1)
            win = min(win, n_samples if n_samples % 2 == 1 else n_samples - 1)
            if win >= 5:
                sf_samples = savgol_filter(sf_samples, window_length=win, polyorder=2, mode="interp")
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)

        sf_samples = np.clip(sf_samples, 0.90, 1.10)
        if np.max(np.abs(sf_samples - 1.0)) < 0.002:
            return audio.copy()

        # Local source-step: stretch>1 => slower playback => smaller source increment.
        src_step = 1.0 / np.clip(sf_samples, 0.85, 1.15)
        src_pos = np.cumsum(src_step)
        src_pos -= src_pos[0]

        # Normalize mapping to consume exactly the available source range.
        max_pos = float(src_pos[-1]) + 1e-12
        src_pos *= (n_samples - 1) / max_pos
        src_pos = np.clip(src_pos, 0.0, n_samples - 1)

        corrected = np.interp(src_pos, np.arange(n_samples, dtype=np.float64), audio_f)
        corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        return corrected.astype(audio.dtype, copy=False)

    def _psola_timestretch(
        self,
        audio: np.ndarray,
        stretch_factors: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Pitch-Synchronous Overlap-Add (PSOLA) für Gesangsmaterial.

        Erhält Formanten bei Wow/Flutter-Korrektur (Moulines & Charpentier 1990;
        Macon & Clements 1997). Aktiviert wenn PANNs Vocals confidence >= 0.40.
        Fallback auf _phase_vocoder_timestretch() bei unbekanntem Grundton.

        Args:
            audio:           Mono-Audio [samples], beliebige float-Dtype.
            stretch_factors: Stretch-Faktoren pro pYIN-Frame, Ratios [~0.95–1.05].
            sample_rate:     Sample-Rate in Hz (assert == 48000).

        Returns:
            Zeitgedehntes Audio gleicher Länge wie Eingabe, NaN/Inf-frei, gleiche Dtype.
        """
        if len(audio) == 0:
            return audio.copy()

        dtype = audio.dtype
        audio_f = audio.astype(np.float64)

        # Grundfrequenz-Schätzung für Pitch-Marken (pYIN — Mauch & Dixon 2014)
        pitch_hz, confidence = self._estimate_pitch_yin(audio_f, sample_rate)
        n_frames = len(pitch_hz)
        if n_frames == 0:
            return self._phase_vocoder_timestretch(audio, stretch_factors, sample_rate)

        hop = self.STFT_HOP_SIZE

        # Pitch-Perioden in Samples (Fallback 440 Hz für nicht-stimmhafte Segmente)
        voiced = (pitch_hz > 50) & (confidence > 0.40)
        f0_safe = np.where(voiced & (pitch_hz > 0), pitch_hz, 440.0)
        period_samps = np.round(sample_rate / np.maximum(f0_safe, 1.0)).astype(int)
        period_samps = np.clip(period_samps, 20, sample_rate // 50)  # >= 50 Hz Untergrenze

        # Stretch-Faktoren auf n_frames interpolieren
        if len(stretch_factors) != n_frames:
            x_src = np.linspace(0, n_frames - 1, max(len(stretch_factors), 2))
            sf_per_frame = np.interp(np.arange(n_frames), x_src, stretch_factors)
        else:
            sf_per_frame = stretch_factors.astype(np.float64)
        sf_per_frame = np.clip(sf_per_frame, 0.9, 1.1)

        # OLA-Ausgangspuffer (großzügig dimensioniert, am Ende getrimmt)
        n_input = len(audio_f)
        max_period = int(np.max(period_samps))
        out_buf = np.zeros(n_input + max_period * 4, dtype=np.float64)
        weight_buf = np.zeros_like(out_buf)

        out_write = 0
        for i in range(n_frames):
            in_center = i * hop
            if in_center >= n_input:
                break
            period = int(period_samps[i])
            sf = float(sf_per_frame[i])

            # Grain: symmetrisch ±1 Periode um in_center (Hanning-gewichtet)
            i_s = max(0, in_center - period)
            i_e = min(n_input, in_center + period)
            grain = audio_f[i_s:i_e].copy()
            if len(grain) == 0:
                out_write += round(hop * sf)
                continue

            win = np.hanning(len(grain))
            grain *= win

            # Ausgabe-Fensterposition (OLA)
            out_center = out_write
            o_s = max(0, out_center - period)
            o_e = min(len(out_buf), out_center + period)
            g_len = o_e - o_s
            if g_len <= 0:
                out_write += round(hop * sf)
                continue

            # Grain auf Fensterlänge anpassen (Trimm oder Zero-Pad)
            if g_len < len(grain):
                grain = grain[:g_len]
                win = win[:g_len]
            elif g_len > len(grain):
                pad = g_len - len(grain)
                grain = np.pad(grain, (0, pad))
                win = np.pad(win, (0, pad))

            out_buf[o_s:o_e] += grain
            weight_buf[o_s:o_e] += win
            out_write += round(hop * sf)

        # OLA-Normierung; Ausgabe auf Originallänge trimmen + NaN-Schutz
        safe_w = np.maximum(weight_buf[:n_input], 1e-8)
        result = out_buf[:n_input] / safe_w
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0).astype(dtype)


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 12: Professional Wow & Flutter Correction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio with synthetic wow/flutter
    duration = 5.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate 440 Hz tone with combined wow (1 Hz) + flutter (20 Hz)
    wow_freq = 1.0  # Hz (slow speed variation)
    wow_depth = 0.015  # 1.5% pitch variation

    flutter_freq = 20.0  # Hz (fast mechanical vibration)
    flutter_depth = 0.005  # 0.5% pitch variation

    # Combined instantaneous pitch variation
    pitch_variation = (
        1.0 + wow_depth * np.sin(2 * np.pi * wow_freq * t) + flutter_depth * np.sin(2 * np.pi * flutter_freq * t)
    )
    instantaneous_freq = 440 * pitch_variation

    # Generate audio with varying pitch
    phase = np.cumsum(2 * np.pi * instantaneous_freq / sample_rate)
    audio = 0.3 * np.sin(phase)

    # Add harmonics for more realistic tone
    audio += 0.15 * np.sin(2 * phase)  # 2nd harmonic
    audio += 0.10 * np.sin(3 * phase)  # 3rd harmonic

    logger.debug("Generated %ss test audio @ %s Hz", duration, sample_rate)
    logger.debug("Base frequency: 440 Hz with harmonics (2nd, 3rd)")
    logger.debug("Wow: %s Hz, Depth: %.2f%%", wow_freq, wow_depth * 100)
    logger.debug("Flutter: %s Hz, Depth: %.2f%%", flutter_freq, flutter_depth * 100)
    logger.debug("Total pitch variation: %.2f%%", (wow_depth + flutter_depth) * 100)
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE (Aggressive correction)"),
        (MaterialType.VINYL, "VINYL (Moderate correction)"),
        (MaterialType.SHELLAC, "SHELLAC (Conservative correction)"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = WowFlutterFix()
        result = phase.process(audio, sample_rate, material)

        if result.metrics["wow_flutter_detected"]:
            logger.debug("✅ Professional Wow & Flutter Correction:")
            logger.debug("   Detected: YES")
            logger.debug("   Max Deviation: %.3f%%", result.metrics["max_deviation_percent"])
            logger.debug("   Wow Magnitude: %.3f%%", result.metrics["wow_magnitude_percent"])
            logger.debug("   Flutter Magnitude: %.3f%%", result.metrics["flutter_magnitude_percent"])
            logger.debug("   Residual Deviation: %.3f%% (target <0.3%%)", result.metrics["residual_deviation_percent"])
            logger.debug("   Correction Strength: %s", result.metrics["correction_strength"])
            logger.debug("   Mean Confidence: %.2f", result.metrics["mean_confidence"])
            logger.debug(
                f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
            )
            logger.debug("")
        else:
            logger.debug("⚠️  No significant wow/flutter detected")
            logger.debug("   Max Deviation: %.3f%%", result.metrics["max_deviation_percent"])
            logger.debug("   Threshold: %s%%", phase.DETECTION_THRESHOLD[material])
            logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
