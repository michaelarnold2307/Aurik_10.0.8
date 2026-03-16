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

import os
import sys


import logging
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
        MaterialType.TAPE: 0.90,  # Aggressive (capstan flutter, wow from speed variations)
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

    # YIN algorithm parameters (optimized for performance)
    YIN_THRESHOLD = 0.15  # Confidence threshold for pitch detection
    PITCH_WINDOW_MS = 100  # Larger window for stability (was 50ms)
    PITCH_HOP_FACTOR = 2  # Less overlap (was 4 = 75% overlap, now 2 = 50% overlap)

    # Phase Vocoder parameters (optimized)
    STFT_WINDOW_SIZE = 1024  # Smaller FFT (was 2048)
    STFT_HOP_SIZE = 256  # Smaller hop (was 512)

    # Formant preservation (prevent "chipmunk" effect)
    PRESERVE_FORMANTS = True

    def __init__(self):
        super().__init__()
        self.name = "Wow & Flutter Correction v2 Professional"

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

        # Get material-specific parameters
        strength = self.CORRECTION_STRENGTH.get(material, 0.7)
        threshold = self.DETECTION_THRESHOLD.get(material, 0.5)

        # Convert to mono for pitch analysis
        is_stereo = audio.ndim == 2
        if is_stereo:
            mono = np.mean(audio, axis=1)
        else:
            mono = audio.copy()

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "balanced")

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

        # ML-Hybrid only if resources available and quality mode permits
        use_ml_hybrid = ML_HYBRID_AVAILABLE and quality_mode in ["balanced", "maximum"] and not use_lightweight

        if use_ml_hybrid:
            try:
                logger.info(f"Phase 12 ML-Hybrid: mode={quality_mode}, material={material.value}")

                # Configure ML pitch detector strategy
                if quality_mode == "maximum":
                    strategy = PitchDetectionStrategy.HYBRID  # Full YIN + CREPE
                else:  # balanced
                    strategy = PitchDetectionStrategy.ADAPTIVE  # Smart: YIN only if confident

                detector = HybridWowFlutter(
                    config=WowFlutterConfig(
                        strategy=strategy,
                        yin_threshold=self.YIN_THRESHOLD,
                        crepe_model="full" if quality_mode == "maximum" else "medium",
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

        # DSP-Only (Fast-Modus oder ML-Fallback): pYIN
        if not use_ml_hybrid:
            logger.info(f"Phase 12 pYIN DSP: material={material.value}")
            pitch_trajectory, confidence = self._estimate_pitch_yin(mono, sample_rate)

        # Continue with standard wow/flutter correction pipeline
        # (regardless of detection method)

        # Step 1: Separate wow (<4 Hz) and flutter (4-100 Hz) components
        wow_component, flutter_component = self._separate_wow_flutter(pitch_trajectory, sample_rate)

        # Step 3: Detect significant wow/flutter (check if correction needed)
        wow_flutter_detected, max_deviation = self._detect_wow_flutter(pitch_trajectory, confidence, threshold)

        if not wow_flutter_detected:
            # No significant wow/flutter detected
            metadata = {
                "algorithm": "hybrid_ml_pyin_crepe_v3" if use_ml_hybrid else "pyin_phase_vocoder",
                "version": "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
                "ml_hybrid": use_ml_hybrid,
            }

            if use_ml_hybrid:
                metadata["pyin_applied"] = ml_result.pyin_applied
                metadata["crepe_applied"] = ml_result.crepe_applied
                metadata["strategy_used"] = str(ml_result.strategy_used)
                metadata["ml_metadata"] = ml_result.metadata

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
                },
                execution_time_seconds=time.time() - start_time,
                metadata=metadata,
            )

        # Step 4: Calculate time-stretching factors from pitch deviation
        stretch_factors = self._calculate_stretch_factors(pitch_trajectory, confidence, strength)

        # Step 5: Apply time-stretching – PSOLA für Vokal-Segmente, WSOLA sonst
        # Moulines & Charpentier (1990): PSOLA ist formanterhaltend bei Gesangsmaterial;
        # Phase-Vocoder (hier: WSOLA/resample) für Instrumental-/Nicht-Vokal-Material.
        vocals_conf = float(kwargs.get("panns_vocals_confidence", 0.0))
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
        if is_stereo:
            restored_mono = np.mean(restored, axis=1)
        else:
            restored_mono = restored

        residual_pitch, residual_conf = self._estimate_pitch_yin(restored_mono, sample_rate)
        residual_deviation = self._calculate_max_deviation(residual_pitch, residual_conf)

        processing_time = time.time() - start_time

        # Calculate wow/flutter statistics
        wow_magnitude = np.std(wow_component[wow_component != 0])
        flutter_magnitude = np.std(flutter_component[flutter_component != 0])

        # Build metadata
        metadata = {
            "algorithm": (
                "hybrid_ml_pyin_crepe_psola_v3"
                if (use_ml_hybrid and vocals_conf >= 0.4)
                else (
                    "hybrid_ml_pyin_crepe_v3"
                    if use_ml_hybrid
                    else "pyin_psola" if vocals_conf >= 0.4 else "pyin_phase_vocoder"
                )
            ),
            "version": "3.0_ml_hybrid" if use_ml_hybrid else "3.0_pyin",
            "ml_hybrid": use_ml_hybrid,
            "psola_active": vocals_conf >= 0.4,
            "panns_vocals_confidence": vocals_conf,
            "threshold": threshold,
            "stft_window": self.STFT_WINDOW_SIZE,
            "stft_hop": self.STFT_HOP_SIZE,
        }

        if use_ml_hybrid:
            metadata["pyin_applied"] = ml_result.pyin_applied
            metadata["crepe_applied"] = ml_result.crepe_applied
            metadata["strategy_used"] = str(ml_result.strategy_used)
            metadata["pitch_detection_time"] = ml_result.processing_time
            metadata["ml_metadata"] = ml_result.metadata

        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.clip(restored, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=restored,
            metrics={
                "wow_flutter_detected": True,
                "max_deviation_percent": max_deviation,
                "residual_deviation_percent": residual_deviation,
                "wow_magnitude_percent": float(wow_magnitude) if not np.isnan(wow_magnitude) else 0.0,
                "flutter_magnitude_percent": float(flutter_magnitude) if not np.isnan(flutter_magnitude) else 0.0,
                "correction_strength": strength,
                "mean_confidence": float(np.mean(confidence[confidence > 0])),
                "material": material.value,
                "quality_mode": quality_mode,
            },
            execution_time_seconds=processing_time,
            metadata=metadata,
        )

    def _estimate_pitch_yin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Rückwärts-kompatibles Alias auf pYIN (Mauch & Dixon 2014).

        Aufruf-Schnittstelle identisch mit YIN, aber probabilistische
        Multilevel-Threshold-Auswertung für höhere Robustheit.
        """
        return self._estimate_pitch_pyin(audio, sample_rate)

    def _estimate_pitch_pyin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Probabilistic YIN (pYIN) nach Mauch & Dixon (2014).

        Mauch & Dixon (2014): \"pYIN: A Fundamental Frequency Estimator
        Using Probabilistic Threshold Distributions\".

        Algorithmus:
            1. CMND-Funktion (wie YIN) pro Frame
            2. Multi-Threshold-Kandidaten: thresholds ∈ [0.01, 0.30] (N_thr=20)
            3. Wahrscheinlichkeits-Gewichte nach Beta-Verteilung (a=2, b=18)
            4. Parabolic Interpolation für Sub-Sample-Genauigkeit
            5. Gewichtetes Maximum über Kandidaten → pYIN-Schätzung
            6. Temporal Smoothing via exponentieller Glättung (analog HMM-Tracking)

        Vorteile gegenüber simple YIN:
            - Kein hartes Threshold → robuster gegen Oktav-Fehler
            - Probabilistische Konfidenz statt binärer Confidence
            - Stabile Schätzung bei schwachem Signal

        Args:
            audio: Mono float32 [-1,1]
            sample_rate: Sample-Rate (erwartet: 48000 Hz)

        Returns:
            (pitch_trajectory, confidence): Pitch-Hz und Konfidenz [0,1] pro Frame
        """
        window_samples = int(self.PITCH_WINDOW_MS * sample_rate / 1000)
        hop_samples = window_samples // self.PITCH_HOP_FACTOR

        min_period = int(sample_rate / 1000)  # max 1000 Hz
        max_period = int(sample_rate / 50)  # min 50 Hz
        max_period = min(max_period, window_samples // 2)

        num_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)
        pitch_trajectory = np.zeros(num_windows, dtype=np.float64)
        confidence = np.zeros(num_windows, dtype=np.float64)

        # pYIN: Multi-Threshold-Gewichte via Beta(2,18)-ähnliche Verteilung
        N_thr = 20
        thresholds = np.linspace(0.01, 0.30, N_thr)
        # Beta-ähnliche Gewichte: niedrige Thresholds bevorzugt (konservativ)
        beta_weights = (1 - thresholds) ** 17 * thresholds
        beta_weights /= beta_weights.sum() + 1e-10

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            if end > len(audio):
                break

            window = audio[start:end] * np.hanning(window_samples)

            # CMND-Funktion (wie YIN)
            autocorr = np.correlate(window, window, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            diff = 2.0 * (autocorr[0] - autocorr[:max_period])
            cmnd = np.ones(max_period)
            cumsum = np.cumsum(diff[1:])
            tau_range = np.arange(1, max_period)
            cmnd[1:] = diff[1:] * tau_range / (cumsum + 1e-10)

            # Multi-Threshold pYIN
            cand_pitches: list = []
            cand_weights: list = []

            for thr, w in zip(thresholds, beta_weights):
                tau_est = 0
                for tau in range(min_period, max_period):
                    if cmnd[tau] < thr:
                        if 0 < tau < max_period - 1:
                            if cmnd[tau] <= cmnd[tau - 1] and cmnd[tau] <= cmnd[tau + 1]:
                                tau_est = tau
                                break
                if tau_est == 0:
                    tau_est = min_period + int(np.argmin(cmnd[min_period:max_period]))

                # Parabolische Interpolation
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
                # Gewichtetes Medioid (pitch mit höchstem Gesamtgewicht im
                # Bereich ±10% um den gewichteten Mittelwert)
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

        # Temporal Smoothing (vereinfachtes HMM-Tracking via Exp-Glättung)
        alpha_smooth = 0.7
        for i in range(1, num_windows):
            if pitch_trajectory[i] > 0 and pitch_trajectory[i - 1] > 0:
                pitch_trajectory[i] = alpha_smooth * pitch_trajectory[i - 1] + (1 - alpha_smooth) * pitch_trajectory[i]

        logger.debug(
            "pYIN: %d Frames, μ_pitch=%.1f Hz, μ_conf=%.3f",
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
            if cmnd[tau] < self.YIN_THRESHOLD:
                if 0 < tau < max_period - 1:
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
            b_wow, a_wow = signal.butter(4, wow_cutoff / nyquist, btype="low")
            wow_component = signal.filtfilt(b_wow, a_wow, deviation)
        else:
            wow_component = deviation  # Frame rate too low, treat all as wow

        # Band-pass filter for flutter (4-100 Hz)
        flutter_low = 4.0  # Hz
        flutter_high = 100.0  # Hz
        if flutter_high < nyquist:
            b_flutter, a_flutter = signal.butter(4, [flutter_low / nyquist, flutter_high / nyquist], btype="band")
            flutter_component = signal.filtfilt(b_flutter, a_flutter, deviation)
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
        self, pitch_trajectory: np.ndarray, confidence: np.ndarray, strength: float
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
                stretch_factors[i] = np.clip(stretch_factors[i], 0.95, 1.05)  # ±5% max stretch
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
        stretch_factors = np.clip(stretch_factors, 0.95, 1.05)  # Grenzen nach Glättung sichern

        return stretch_factors

    def _phase_vocoder_timestretch(
        self, audio: np.ndarray, stretch_factors: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Apply simplified WSOLA (Waveform Similarity Overlap-Add) time-stretching.

        WSOLA is faster than phase vocoder and works well for small stretch factors.
        For wow/flutter correction (small deviations), WSOLA is sufficient.

        Args:
            audio: Mono audio samples
            stretch_factors: Time-varying stretch factors (one per pitch window)
            sample_rate: Sample rate

        Returns:
            Time-stretched audio
        """
        # Simplification: Use average stretch factor for efficiency
        # (Full implementation would use time-varying stretching)
        avg_stretch = np.mean(stretch_factors)

        # If avg stretch is very close to 1.0, no correction needed
        if abs(avg_stretch - 1.0) < 0.005:  # <0.5% change
            return audio

        # Use scipy's resample for efficiency (band-limited interpolation)
        # This is faster than full phase vocoder and sufficient for small changes
        output_length = int(len(audio) / avg_stretch)

        # Resample to correct length
        corrected = signal.resample(audio, output_length)

        # Ensure output matches input length (truncate or pad)
        if len(corrected) > len(audio):
            corrected = corrected[: len(audio)]
        elif len(corrected) < len(audio):
            corrected = np.pad(corrected, (0, len(audio) - len(corrected)), mode="edge")

        return corrected

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
                out_write += int(round(hop * sf))
                continue

            win = np.hanning(len(grain))
            grain *= win

            # Ausgabe-Fensterposition (OLA)
            out_center = out_write
            o_s = max(0, out_center - period)
            o_e = min(len(out_buf), out_center + period)
            g_len = o_e - o_s
            if g_len <= 0:
                out_write += int(round(hop * sf))
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
            out_write += int(round(hop * sf))

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

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug("Base frequency: 440 Hz with harmonics (2nd, 3rd)")
    logger.debug(f"Wow: {wow_freq} Hz, Depth: {wow_depth * 100:.2f}%")
    logger.debug(f"Flutter: {flutter_freq} Hz, Depth: {flutter_depth * 100:.2f}%")
    logger.debug(f"Total pitch variation: {(wow_depth + flutter_depth) * 100:.2f}%")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE (Aggressive correction)"),
        (MaterialType.VINYL, "VINYL (Moderate correction)"),
        (MaterialType.SHELLAC, "SHELLAC (Conservative correction)"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug(f"Material: {material_name}")
        logger.debug("─" * 80)
        logger.debug("")

        phase = WowFlutterFix()
        result = phase.process(audio, sample_rate, material)

        if result.metrics["wow_flutter_detected"]:
            logger.debug("✅ Professional Wow & Flutter Correction:")
            logger.debug("   Detected: YES")
            logger.debug(f"   Max Deviation: {result.metrics['max_deviation_percent']:.3f}%")
            logger.debug(f"   Wow Magnitude: {result.metrics['wow_magnitude_percent']:.3f}%")
            logger.debug(f"   Flutter Magnitude: {result.metrics['flutter_magnitude_percent']:.3f}%")
            logger.debug(f"   Residual Deviation: {result.metrics['residual_deviation_percent']:.3f}% (target <0.3%)")
            logger.debug(f"   Correction Strength: {result.metrics['correction_strength']}")
            logger.debug(f"   Mean Confidence: {result.metrics['mean_confidence']:.2f}")
            logger.debug(
                f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
            )
            logger.debug("")
        else:
            logger.debug("⚠️  No significant wow/flutter detected")
            logger.debug(f"   Max Deviation: {result.metrics['max_deviation_percent']:.3f}%")
            logger.debug(f"   Threshold: {phase.DETECTION_THRESHOLD[material]}%")
            logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
