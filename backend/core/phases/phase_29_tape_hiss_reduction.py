#!/usr/bin/env python3
"""
Phase 29: Tape Hiss Reduction v3.0 - Über-SOTA OMLSA/IMCRA
Adaptive HF-Rauschunterdrückung für Tape-Aufnahmen via spektraler OMLSA/IMCRA-Verarbeitung.

Algorithmus (v3.0):
1. STFT (nperseg=2048, 75% Overlap) des gesamten Signals
2. IMCRA-Rauschschätzung (Cohen & Berdugo 2002):
   - Bias-korrigiertes gleitendes Minimum im HF-Bereich
   - b_min=1.66, alpha_n=0.85, Fenster ~1.5s
3. OMLSA-Gain (Cohen 2003):
   - G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p
   - HF-selektiv: Bins < hf_low erhalten G=1.0 (unangetastet)
   - Bins >= hf_low: OMLSA-Gain mit materialadaptivem G_floor
4. Cappé-Gain-Glättung (1994): temporal geglättet
5. ISTFT + NaN-Schutz + clip[-1, 1]
6. ML-Hybrid: DeepFilterNet v3 II für Residual-Hiss >2kHz (optional)

Scientific Foundation:
- Cohen & Berdugo (2002): IMCRA — primär
- Cohen (2003): OMLSA — primär
- Cappé (1994): Elimination of the Musical Noise Phenomenon — Gain-Glättung
- Le Roux & Vincent (2013): Consistent Wiener Filtering — Phasenkonsistenz
- Überholt (NICHT primär): einfacher Percentile-Gate, Bandpass-Expander-Kette

Author: Aurik Development Team
Version: 2.0.0 Professional ML-Hybrid
"""

import os
import sys


import logging
import tempfile
import time

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType
from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# ML-Hybrid Support
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from backend.core.quality_mode import QualityMode, should_use_ml

    QUALITY_MODE_AVAILABLE = True
except ImportError:
    QUALITY_MODE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TapeHissReductionPhase(PhaseInterface):
    """
    Enhanced tape hiss reduction with adaptive gates and ML-Hybrid Support.

    Tape hiss is characterized by:
    - High-frequency noise (primarily >8 kHz)
    - Stationary (constant noise floor)
    - Gaussian distribution

    Strategy:
    1. Split into frequency bands (8 bands above 4 kHz)
    2. Estimate noise floor per band
    3. Apply adaptive expander gate per band
    4. Smooth gate action (attack/release)
    5. Reconstruct with preserved phase
    6. ML-Hybrid: <2kHz DSP → >2kHz ML DeepFilterNet refinement

    Material Adaptation:
    - Tape: Moderate reduction (primary target)
    - Shellac/Vinyl: Light (mainly surface noise, handled by phase_28)
    - CD/Streaming: Disabled
    """

    # ML frequency band threshold (Hz)
    ML_FREQUENCY_THRESHOLD_HZ = 2000  # <2kHz: DSP, >2kHz: ML optional

    # Hiss reduction threshold (dB above noise floor to start gating)
    GATE_THRESHOLD_DB = {
        MaterialType.SHELLAC: -6,  # Light gating
        MaterialType.VINYL: -8,
        MaterialType.TAPE: -10,  # More aggressive
        MaterialType.CD_DIGITAL: -999,  # Disabled
        MaterialType.STREAMING: -999,
    }

    # Reduction depth (dB to attenuate below threshold)
    REDUCTION_DEPTH_DB = {
        MaterialType.SHELLAC: 6,
        MaterialType.VINYL: 8,
        MaterialType.TAPE: 12,  # Aggressive for tape
        MaterialType.CD_DIGITAL: 0,
        MaterialType.STREAMING: 0,
    }

    # HF focus range (Hz) - where to apply reduction most aggressively
    HF_FOCUS_RANGE = {
        MaterialType.SHELLAC: (6000, 12000),
        MaterialType.VINYL: (8000, 15000),
        MaterialType.TAPE: (8000, 18000),  # Tape hiss dominates 8-18 kHz
        MaterialType.CD_DIGITAL: (0, 0),
        MaterialType.STREAMING: (0, 0),
    }

    # Number of frequency bands for multiband processing
    NUM_BANDS = 8

    def __init__(self, sample_rate: int = 48000, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self._deepfilternet_plugin = None

    def _get_deepfilternet_plugin(self):
        """
        Lazy load DeepFilterNet v3 II Plugin.

        Returns:
            DeepFilterNet plugin or None if unavailable
        """
        if self._deepfilternet_plugin is not None:
            return self._deepfilternet_plugin

        try:
            from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin

            self._deepfilternet_plugin = DeepFilterNetV3IIPlugin()
            logger.info("✅ DeepFilterNet v3 II Plugin loaded for Tape Hiss Reduction")
            return self._deepfilternet_plugin
        except Exception as e:
            logger.warning(f"⚠️  DeepFilterNet Plugin not available: {e}")
            logger.info("    Falling back to DSP-only hiss reduction")
            return None

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_29_tape_hiss_reduction",
            name="Tape Hiss Reduction v3 OMLSA/IMCRA",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=6,
            dependencies=["phase_03_denoise", "phase_28_surface_noise_profiling"],
            estimated_time_factor=0.10,
            version="3.0.0",
            memory_requirement_mb=60,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description="HF-OMLSA-Rauschunterdrückung (Cohen 2002/2003) — Über-SOTA",
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        material: MaterialType = MaterialType.TAPE,
        quality_mode: str | None = None,
        **kwargs,
    ) -> PhaseResult:
        """
        Process audio to reduce tape hiss with ML-Hybrid support.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Source material type
            quality_mode: Quality mode (FAST/BALANCED/MAXIMUM), None=auto

        Returns:
            PhaseResult with denoised audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.sample_rate = sample_rate
        self.validate_input(audio)

        # Determine if ML should be used
        use_ml = False
        if QUALITY_MODE_AVAILABLE and quality_mode:
            try:
                qm = QualityMode[quality_mode.upper()]
                use_ml = should_use_ml(29, qm)  # Phase 29
            except Exception:
                pass

        # Skip for digital sources
        if material in [MaterialType.CD_DIGITAL, MaterialType.STREAMING]:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={"material": material.name, "processing": "skipped"},
                warnings=["Digital source - no tape hiss expected"],
            )

        # Get material-specific parameters
        # Fallback via .value-Vergleich loest Doppel-Import-Problem
        # (core.defect_scanner vs. backend.core.defect_scanner erzeugen
        # verschiedene Enum-Klassen-Objekte, obwohl der Wert identisch ist)
        _mat_val = getattr(material, "value", str(material))
        gate_threshold_db = self.GATE_THRESHOLD_DB.get(material) or next(
            (v for k, v in self.GATE_THRESHOLD_DB.items() if getattr(k, "value", None) == _mat_val),
            -10,
        )
        reduction_depth_db = self.REDUCTION_DEPTH_DB.get(material) or next(
            (v for k, v in self.REDUCTION_DEPTH_DB.items() if getattr(k, "value", None) == _mat_val),
            8,
        )
        _hf = self.HF_FOCUS_RANGE.get(material) or next(
            (v for k, v in self.HF_FOCUS_RANGE.items() if getattr(k, "value", None) == _mat_val),
            (8000, 18000),
        )
        hf_low, hf_high = _hf

        # Create frequency bands (logarithmic spacing)
        nyquist = sample_rate / 2
        np.logspace(np.log10(hf_low), np.log10(min(hf_high, nyquist * 0.95)), self.NUM_BANDS + 1)

        is_stereo = audio.ndim == 2
        channels = 2 if is_stereo else 1

        # Process each channel
        audio_processed = np.zeros_like(audio)
        hiss_reduction_per_band = []

        for ch in range(channels):
            channel = audio[:, ch] if is_stereo else audio

            # STFT-OMLSA-Verarbeitung (HF-selektiv)
            processed = self._process_channel_omlsa(channel, sample_rate, hf_low, hf_high, material)
            hiss_reduction_per_band = []  # Metriken auf Kanal-Ebene  # noqa: F841

            if is_stereo:
                audio_processed[:, ch] = processed
            else:
                audio_processed = processed

        # Calculate overall HF noise reduction
        audio_ch0 = audio[:, 0] if is_stereo else audio
        proc_ch0 = audio_processed[:, 0] if is_stereo else audio_processed
        hf_band_orig = self._extract_band(audio_ch0, sample_rate, hf_low, hf_high)
        hf_band_proc = self._extract_band(proc_ch0, sample_rate, hf_low, hf_high)

        # Guard: log10(0) when both bands are silent -> RuntimeWarning; clamp >= 1e-30
        hf_reduction_db = 20 * np.log10(np.maximum(np.std(hf_band_orig) / (np.std(hf_band_proc) + 1e-10), 1e-30))

        # ML Refinement for HF (>2kHz) - if enabled and significant hiss present
        ml_refined = False
        if use_ml and hf_reduction_db > 3:  # Only refine if significant hiss was removed
            ml_success = self._refine_hf_with_ml(audio_processed, sample_rate)
            if ml_success:
                ml_refined = True
                logger.info("✅ ML HF refinement applied (DeepFilterNet): residual hiss removal >2kHz")

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        audio_processed = np.nan_to_num(audio_processed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_processed = np.clip(audio_processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=audio_processed,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "gate_threshold_db": float(gate_threshold_db),
                "reduction_depth_db": float(reduction_depth_db),
                "hf_focus_range_hz": [int(hf_low), int(hf_high)],
                "hf_reduction_db": round(float(hf_reduction_db), 2),
                "ml_refined": ml_refined,
                "algorithm_version": "3.0_omlsa_ml_hybrid" if ml_refined else "3.0_omlsa",
                "algorithm": "IMCRA+OMLSA (Cohen 2002/2003)",
                "ml_model": "DeepFilterNet v3 II" if ml_refined else None,
                "rt_factor": float(rt_factor),
            },
            warnings=[] if rt_factor < 0.12 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _process_channel_omlsa(
        self, channel: np.ndarray, sample_rate: int, hf_low: float, hf_high: float, material: "MaterialType"
    ) -> np.ndarray:
        """STFT-OMLSA-Verarbeitung: HF-selektive Rauschunterdrückung (Cohen 2002/2003).

        Algorithmus:
            1. STFT (nperseg=2048, noverlap=1536)
            2. IMCRA-Rauschschätzung im HF-Bereich [hf_low, hf_high]
            3. OMLSA-Gain: G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p
            4. Bins < hf_low: G=1.0 (unangetastet — Tieftonschutz)
            5. Cappé-Glättung: alpha_g = 0.85
            6. ISTFT + NaN/Clip-Schutz

        Args:
            channel:     Mono-Audio (1D float32)
            sample_rate: Abtastrate in Hz
            hf_low:      Untere HF-Grenze (Hz), z.B. 8000
            hf_high:     Obere HF-Grenze (Hz), z.B. 18000
            material:    MaterialType für G_floor

        Returns:
            processed: Restauriertes Mono-Audio (gleiche Länge wie channel)
        """
        eps = 1e-10
        # Material-adaptiver G_floor
        G_floor_map = {
            "SHELLAC": 0.12,
            "VINYL": 0.10,
            "TAPE": 0.08,
            "REEL_TAPE": 0.07,
            "DAT": 0.06,
        }
        mat_name = getattr(material, "name", str(material)).upper()
        G_floor = G_floor_map.get(mat_name, 0.10)
        q = 0.5  # Rausch-Präsenz-Prior

        # STFT
        nperseg = 2048
        noverlap = 1536
        f_bins, t_arr, stft = signal.stft(channel, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")

        magnitude = np.abs(stft)
        phase_arr = np.angle(stft)
        F, T = magnitude.shape

        # Frequenz-Bin-Grenzen
        hf_low_bin = max(1, int(np.searchsorted(f_bins, hf_low)))
        hf_high_bin = min(F - 1, int(np.searchsorted(f_bins, hf_high)))

        # IMCRA-Rauschschätzung nur im HF-Bereich
        alpha_n = 0.85
        b_min = 1.66
        hop_s = float(t_arr[1] - t_arr[0]) if T > 1 else 0.01
        M = max(15, int(round(1.5 / hop_s)))

        # Geglättete Leistung (nur HF-Bins)
        P_hat = magnitude**2
        for ti in range(1, T):
            P_hat[hf_low_bin:, ti] = (
                alpha_n * P_hat[hf_low_bin:, ti - 1] + (1.0 - alpha_n) * magnitude[hf_low_bin:, ti] ** 2
            )
        P_hat = np.nan_to_num(P_hat, nan=eps)

        # Gleitendes Minimum -> Rauschleistung
        noise_power = np.full_like(P_hat, eps)
        for ti in range(T):
            s = max(0, ti - M)
            noise_power[hf_low_bin:, ti] = np.min(P_hat[hf_low_bin:, s : ti + 1], axis=1)
        noise_mag = np.sqrt(np.maximum(b_min * noise_power, eps))
        noise_mag = np.nan_to_num(noise_mag, nan=eps)

        # OMLSA-Gain (nur HF-Bins; LF-Bins behalten G=1.0)
        sigma2_n = np.maximum(noise_mag**2, eps)
        gamma = np.maximum(magnitude**2 / sigma2_n, 0.0)
        xi = np.maximum(gamma - 1.0, 0.0)
        v = np.clip(xi * gamma / (xi + 1.0 + eps), 0.0, 500.0)
        lam = np.exp(np.clip(-xi + v, -50.0, 50.0))
        p = 1.0 / (1.0 + q / ((1.0 - q) * lam + eps))
        G_H1 = xi / (xi + 1.0 + eps)

        log_G = (1.0 - p) * np.log(G_floor) + p * np.log(np.maximum(G_H1, eps))
        log_G = np.clip(log_G, np.log(G_floor), 0.0)
        G = np.exp(log_G)
        G = np.nan_to_num(G, nan=G_floor, posinf=1.0, neginf=G_floor)
        G = np.clip(G, G_floor, 1.0)

        # LF-Bins: Gain = 1.0 (vollständig erhalten)
        G[:hf_low_bin, :] = 1.0
        # Bins über hf_high: sanft zurück auf 1.0 (kein Over-Suppression in Nyquist-Nähe)
        if hf_high_bin < F:
            G[hf_high_bin:, :] = 1.0

        # Cappé-Gain-Glättung (alpha_g=0.85)
        alpha_g = 0.85
        G_smooth = np.zeros_like(G)
        G_smooth[:, 0] = G[:, 0]
        for ti in range(1, T):
            G_smooth[:, ti] = alpha_g * G_smooth[:, ti - 1] + (1.0 - alpha_g) * G[:, ti]
        G_smooth = np.clip(np.nan_to_num(G_smooth, nan=G_floor, posinf=1.0, neginf=G_floor), G_floor, 1.0)

        # Spektrum anwenden + ISTFT
        proc_stft = magnitude * G_smooth * np.exp(1j * phase_arr)
        _, processed = signal.istft(proc_stft, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")

        # Länge + NaN/Clip-Schutz
        processed = processed[: len(channel)]
        if len(processed) < len(channel):
            processed = np.pad(processed, (0, len(channel) - len(processed)))
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # §4.5 Psychoakustischer Masking-Gain-Clamp (ISO 11172-3, Painter & Spanias 2000)
        # Berechnet auf Input-Audio → Schutzmaske für Stille / ungemaskierte Bereiche
        try:
            from backend.core.psychoacoustic_masking_model import compute_masking_threshold

            _pmm = compute_masking_threshold(channel.astype(np.float32), sample_rate)
            _pmm_gain_t = np.mean(_pmm.gain_modifier, axis=1).astype(np.float32)
            _hop = 512  # entspricht nperseg=2048, noverlap=1536
            _pmm_centers = np.arange(len(_pmm_gain_t)) * float(_hop) + _hop * 0.5
            _pmm_x = np.arange(len(processed), dtype=np.float32)
            _gain_samples = np.interp(_pmm_x, _pmm_centers, _pmm_gain_t).astype(np.float32)
            processed = np.clip((processed * _gain_samples).astype(np.float32), -1.0, 1.0)
            logger.debug(
                "🎭 PsychoacousticMasking [phase29]: silence=%.1f%% mean_gain=%.3f",
                100.0 * float(np.mean(_pmm.silence_frames)),
                float(np.mean(_pmm_gain_t)),
            )
        except Exception as _pmm_exc:
            logger.debug("PsychoacousticMaskingModel nicht verfügbar: %s", _pmm_exc)

        return processed

    def _extract_band(self, signal_in: np.ndarray, sample_rate: int, low_freq: float, high_freq: float) -> np.ndarray:
        """Bandpass-Filterung f\u00fcr Metrik-Berechnung (Hilfsmethode)."""
        nyquist = sample_rate / 2
        low_norm = max(low_freq, 20.0)
        high_norm = min(high_freq, nyquist * 0.98)
        if low_norm >= high_norm:
            return signal_in.copy()
        sos = signal.butter(4, [low_norm, high_norm], btype="band", fs=sample_rate, output="sos")
        return signal.sosfilt(sos, signal_in)

    def _estimate_noise_floor(self, band_signal: np.ndarray) -> float:
        """
        Legacy-Methode (10th-Percentile RMS) \u2014 nur als R\u00fcckw\u00e4rtskompatibilit\u00e4ts-Alias.
        Primitivere Sch\u00e4tzung; STFT-OMLSA via _process_channel_omlsa ist prim\u00e4r.
        """
        # Compute short-term RMS (10ms windows)
        window_samples = int(0.01 * self.sample_rate)
        num_windows = len(band_signal) // window_samples

        rms_vals = []
        for i in range(num_windows):
            start = i * window_samples
            end = start + window_samples
            window = band_signal[start:end]
            rms = np.sqrt(np.mean(window**2))
            rms_vals.append(rms)

        # 10th percentile as noise floor estimate
        noise_floor = np.percentile(rms_vals, 10) if rms_vals else 1e-10
        noise_floor_db = 20 * np.log10(noise_floor + 1e-10)

        return noise_floor_db

    def _refine_hf_with_ml(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Refine HF hiss reduction (>2kHz) using DeepFilterNet v3 II.

        Band-Specific Strategy:
        1. DSP handles full spectrum with multi-band gates
        2. ML refines >2kHz region to remove residual hiss without artifacts
        3. <2kHz left untouched to preserve warmth and bass

        Args:
            audio: Audio array (mono or stereo, will be modified in-place)
            sample_rate: Sample rate

        Returns:
            True if successful, False otherwise
        """
        if not SOUNDFILE_AVAILABLE:
            logger.warning("soundfile not available for ML HF refinement")
            return False

        plugin = self._get_deepfilternet_plugin()
        if plugin is None:
            return False

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_temp:
                input_path = input_temp.name

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_temp:
                output_path = output_temp.name

            # Write audio to temp file
            sf.write(input_path, audio, sample_rate)

            # Process with DeepFilterNet
            returncode, stdout, stderr = plugin.process(
                input_path, output_path, post_filter=True  # Enable post-filter for smooth HF reduction
            )

            if returncode == 0 and os.path.exists(output_path):
                # Read refined audio
                refined, sr_read = sf.read(output_path)

                # Blend strategy: Keep <2kHz from original, use ML for >2kHz
                if refined.shape == audio.shape:
                    # Extract HF bands
                    sample_rate / 2
                    sos_lp = signal.butter(4, self.ML_FREQUENCY_THRESHOLD_HZ, btype="low", fs=sample_rate, output="sos")
                    sos_hp = signal.butter(
                        4, self.ML_FREQUENCY_THRESHOLD_HZ, btype="high", fs=sample_rate, output="sos"
                    )

                    # Apply filters
                    is_stereo = audio.ndim == 2
                    if is_stereo:
                        for ch in range(2):
                            lf_original = signal.sosfilt(sos_lp, audio[:, ch])
                            hf_refined = signal.sosfilt(sos_hp, refined[:, ch])
                            audio[:, ch] = lf_original + hf_refined
                    else:
                        lf_original = signal.sosfilt(sos_lp, audio)
                        hf_refined = signal.sosfilt(sos_hp, refined)
                        audio[:] = lf_original + hf_refined

                    logger.info("✅ ML HF refinement successful (>2kHz band)")
                    return True
                else:
                    logger.warning(f"Shape mismatch: {refined.shape} vs {audio.shape}")
                    return False
            else:
                logger.warning(f"DeepFilterNet failed (returncode={returncode})")
                return False

        except Exception as e:
            logger.error(f"ML HF refinement error: {e}")
            return False

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception:
                pass

    def _apply_adaptive_gate(
        self, band_signal: np.ndarray, noise_floor_db: float, threshold_db: float, reduction_db: float, sample_rate: int
    ) -> np.ndarray:
        """
        Apply adaptive expander gate to band signal.

        Gate formula:
            gain = 1.0 if level > threshold
            gain = 10^(reduction_db / 20) if level < threshold
            Smooth transition in between
        """
        # Compute envelope (RMS with attack/release)
        envelope = self._compute_envelope(band_signal, sample_rate)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Compute gate threshold
        gate_threshold = noise_floor_db + threshold_db

        # Compute gains
        reduction_factor = 10 ** (reduction_db / 20)
        gains = np.ones_like(envelope)

        # Below threshold: apply reduction
        below_mask = envelope_db < gate_threshold
        gains[below_mask] = 1.0 / reduction_factor

        # Smooth gains (attack/release)
        gains_smoothed = self._smooth_gains(gains, sample_rate)

        # Apply gains
        processed = band_signal * gains_smoothed

        return processed

    def _compute_envelope(
        self, signal_in: np.ndarray, sample_rate: int, attack_ms: float = 5.0, release_ms: float = 50.0
    ) -> np.ndarray:
        """
        Compute envelope with attack/release smoothing.
        """
        # Rectify
        rectified = np.abs(signal_in)

        # Attack/release coefficients
        attack_coeff = np.exp(-1 / (attack_ms * 0.001 * sample_rate))
        release_coeff = np.exp(-1 / (release_ms * 0.001 * sample_rate))

        # Envelope follower
        envelope = np.zeros_like(rectified)
        envelope[0] = rectified[0]

        for i in range(1, len(rectified)):
            if rectified[i] > envelope[i - 1]:
                # Attack
                envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * rectified[i]
            else:
                # Release
                envelope[i] = release_coeff * envelope[i - 1] + (1 - release_coeff) * rectified[i]

        return envelope

    def _smooth_gains(self, gains: np.ndarray, sample_rate: int, smooth_ms: float = 10.0) -> np.ndarray:
        """
        Smooth gain curve to prevent artifacts.
        """
        # Lowpass filter gains
        cutoff = 1000.0 / smooth_ms  # Lower cutoff for longer smooth_ms
        sos = signal.butter(2, cutoff, "low", fs=sample_rate, output="sos")
        gains_smoothed = signal.sosfilt(sos, gains)

        return gains_smoothed


# Test harness
if __name__ == "__main__":
    logger.debug("=== Phase 29: Tape Hiss Reduction v2 Test ===\n")

    processor = TapeHissReductionPhase(sample_rate=44100)

    # Test materials
    test_materials = [
        MaterialType.VINYL,
        MaterialType.TAPE,
        MaterialType.SHELLAC,
    ]

    for material in test_materials:
        logger.debug(f"Testing {material.value.upper()}:")

        # Create test signal: music + tape hiss
        sr = 44100
        duration = 2.0
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples)

        # Music: 440 Hz tone with modulation
        np.random.seed(42)
        music = 0.5 * np.sin(2 * np.pi * 440 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t))

        # Tape hiss: High-frequency noise (8-18 kHz dominant)
        hiss = 0.12 * np.random.randn(samples)
        sos_hiss = signal.butter(4, [8000, 18000], "band", fs=sr, output="sos")
        hiss = signal.sosfilt(sos_hiss, hiss)

        # Combine
        noisy = music + hiss

        # Create stereo
        audio = np.column_stack([noisy, noisy])

        # Process
        start = time.time()
        processed, meta = processor.process(audio, sr, material)
        elapsed = time.time() - start

        # Calculate HF noise reduction
        sos_hf = signal.butter(4, 8000, "high", fs=sr, output="sos")
        hf_orig = signal.sosfilt(sos_hf, audio[:, 0])
        hf_proc = signal.sosfilt(sos_hf, processed[:, 0])

        hf_reduction = 20 * np.log10(np.std(hf_orig) / (np.std(hf_proc) + 1e-10))

        # Display results
        logger.debug(f"  Gate threshold: {meta.get('gate_threshold_db', 0):.1f} dB")
        logger.debug(f"  Reduction depth: {meta.get('reduction_depth_db', 0):.1f} dB")
        logger.debug(f"  HF focus range: {meta.get('hf_focus_range_hz', [])} Hz")
        logger.debug(f"  Num bands: {meta.get('num_bands', 0)}")
        logger.debug(f"  HF reduction: {meta.get('hf_reduction_db', 0):.2f} dB")
        logger.debug(f"  Per-band reduction: {meta.get('reduction_per_band_db', [])[:3]}... (first 3)")
        logger.debug(f"  Processing time: {elapsed:.3f}s")
        logger.debug("  ✅\n")
