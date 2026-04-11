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

import logging
import os
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

try:
    from dsp.pghi import pghi_reconstruct_from_stft as _pghi_p29

    _PGHI_AVAILABLE_P29 = True
except ImportError:
    _PGHI_AVAILABLE_P29 = False

from scipy.ndimage import minimum_filter1d as _min_filter1d_p29  # vectorised sliding-min
from scipy.signal import lfilter as _lfilter_p29  # vectorised IIR smoothing (Cappé 1994)

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

    # MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
    _MRSA_ZONES: tuple = (
        # (name,       win_size, hop_size, f_low_hz, f_high_hz)
        ("sub_bass", 65536, 16384, 0, 250),
        ("mid_low", 16384, 4096, 250, 2500),
        ("mid", 8192, 2048, 2500, 8000),
        ("presence", 1024, 256, 8000, 16000),
        ("air", 128, 32, 16000, 24000),
    )
    _MRSA_CROSSFADE_BW_HZ: float = 100.0

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

    _MAX_RMS_DROP_DB = {
        "tape": 2.0,
        "reel_tape": 1.8,
        "cassette": 2.2,
        "vinyl": 1.5,
        "shellac": 1.2,
        "wax_cylinder": 1.0,
        "cd_digital": 1.2,
        "dat": 1.0,
        "mp3_low": 1.4,
        "mp3_high": 1.4,
        "aac": 1.4,
        "unknown": 1.5,
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
            from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin

            self._deepfilternet_plugin = get_deepfilternet_plugin()
            logger.info("✅ DeepFilterNet v3 II Plugin loaded for Tape Hiss Reduction")
            return self._deepfilternet_plugin
        except Exception as e:
            logger.warning("⚠️  DeepFilterNet Plugin not available: %s", e)
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
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

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

        # Locality-aware modulation from UV3.
        # Sparse hiss-related defect coverage -> conservative denoising outside affected regions.
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
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "processing": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                },
                warnings=["Tape hiss reduction skipped due to zero effective strength"],
            )

        # Create frequency bands (logarithmic spacing)
        nyquist = sample_rate / 2
        np.logspace(np.log10(hf_low), np.log10(min(hf_high, nyquist * 0.95)), self.NUM_BANDS + 1)

        is_stereo = audio.ndim == 2

        if is_stereo:
            # §2.51 Linked-Sidechain OMLSA: Gain-Maske aus Mid-Kanal berechnen,
            # identisch auf L und R anwenden. Verhindert stereo-inkohärente HF-Dämpfung
            # bei kanalasymmetrischem Tape-Rauschen (Phantom-Mitte-Instabilität).
            mid_channel = (audio[:, 0] + audio[:, 1]) * (1.0 / np.sqrt(2.0))
            mid_channel = mid_channel.astype(np.float32)

            # Psychoacoustic masking from L channel (dominant sidechain for mid)
            kwargs.get("masking_result")

            # Compute OMLSA gain on Mid sidechain — applied identically to both channels
            audio_processed = np.zeros_like(audio)
            for ch in range(2):
                channel = audio[:, ch]

                # §Psychoacoustic: select pre-computed masking result for this channel.
                if ch == 1:
                    _ch_masking = kwargs.get("masking_result_r") or kwargs.get("masking_result")
                else:
                    _ch_masking = kwargs.get("masking_result")

                # STFT-OMLSA on this channel but with Mid-linked gain sidechain
                audio_processed[:, ch] = self._process_channel_omlsa(
                    channel,
                    sample_rate,
                    hf_low,
                    hf_high,
                    material,
                    intensity_scale=_effective_strength,
                    masking_result=_ch_masking,
                    linked_sidechain=mid_channel,
                )
        else:
            # Mono: standard processing
            _ch_masking = kwargs.get("masking_result")
            audio_processed = self._process_channel_omlsa(
                audio,
                sample_rate,
                hf_low,
                hf_high,
                material,
                intensity_scale=_effective_strength,
                masking_result=_ch_masking,
            )

        # Calculate overall HF noise reduction
        audio_ch0 = audio[:, 0] if is_stereo else audio
        proc_ch0 = audio_processed[:, 0] if is_stereo else audio_processed
        hf_band_orig = self._extract_band(audio_ch0, sample_rate, hf_low, hf_high)
        hf_band_proc = self._extract_band(proc_ch0, sample_rate, hf_low, hf_high)

        # Guard: log10(0) when both bands are silent -> RuntimeWarning; clamp >= 1e-30
        hf_reduction_db = 20 * np.log10(np.maximum(np.std(hf_band_orig) / (np.std(hf_band_proc) + 1e-10), 1e-30))

        # HF over-suppression guard (Restoration): avoid excessive brilliance loss.
        # If tape hiss attenuation exceeds a material-adaptive ceiling, blend back
        # only the HF residual from original audio.
        hf_detail_blend = 0.0
        _mat_name = getattr(material, "name", str(material)).upper()
        _hf_ceiling_db = {
            "TAPE": 10.0,
            "REEL_TAPE": 10.5,
            "VINYL": 9.5,
            "SHELLAC": 8.5,
        }.get(_mat_name, 10.0)
        if hf_reduction_db > _hf_ceiling_db and _effective_strength > 0.0:
            _excess_db = float(hf_reduction_db - _hf_ceiling_db)
            # Max 28% HF back-blend, scaled by excess reduction and locality strength.
            hf_detail_blend = float(np.clip((_excess_db / 12.0) * 0.28 * _effective_strength, 0.0, 0.28))
            if hf_detail_blend > 0.0:
                if is_stereo:
                    for ch in range(2):
                        _orig_hf = self._extract_band(audio[:, ch], sample_rate, hf_low, hf_high)
                        _proc_hf = self._extract_band(audio_processed[:, ch], sample_rate, hf_low, hf_high)
                        audio_processed[:, ch] = np.clip(
                            audio_processed[:, ch] + hf_detail_blend * (_orig_hf - _proc_hf),
                            -1.0,
                            1.0,
                        )
                else:
                    _orig_hf = self._extract_band(audio, sample_rate, hf_low, hf_high)
                    _proc_hf = self._extract_band(audio_processed, sample_rate, hf_low, hf_high)
                    audio_processed = np.clip(
                        audio_processed + hf_detail_blend * (_orig_hf - _proc_hf),
                        -1.0,
                        1.0,
                    )

                # Recompute HF reduction after guard blend.
                proc_ch0 = audio_processed[:, 0] if is_stereo else audio_processed
                hf_band_proc = self._extract_band(proc_ch0, sample_rate, hf_low, hf_high)
                hf_reduction_db = 20 * np.log10(
                    np.maximum(np.std(hf_band_orig) / (np.std(hf_band_proc) + 1e-10), 1e-30)
                )

        # ML Refinement for HF (>2kHz) - if enabled and significant hiss present
        ml_refined = False
        if use_ml and _effective_strength > 0.0 and hf_reduction_db > 3:  # Only refine if significant hiss was removed
            ml_success = self._refine_hf_with_ml(audio_processed, sample_rate)
            if ml_success:
                ml_refined = True
                logger.info("✅ ML HF refinement applied (DeepFilterNet): residual hiss removal >2kHz")

        # Preserve PMGG strength control via wet/dry blending.
        if 0.0 < _effective_strength < 1.0:
            audio_processed = audio + _effective_strength * (audio_processed - audio)

        audio_processed, loudness_stats = self._apply_material_loudness_preservation(
            audio,
            audio_processed,
            material,
        )

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
                "hf_detail_blend": round(float(hf_detail_blend), 4),
                "ml_refined": ml_refined,
                "algorithm_version": "3.0_omlsa_ml_hybrid" if ml_refined else "3.0_omlsa",
                "algorithm": "IMCRA+OMLSA (Cohen 2002/2003)",
                "stereo_mode": "linked_mid_sidechain" if is_stereo else "mono",
                "ml_model": "DeepFilterNet v3 II" if ml_refined else None,
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": loudness_stats["rms_drop_db"],
                "loudness_makeup_db": loudness_stats["makeup_gain_db"],
            },
            warnings=[] if rt_factor < 0.12 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, dict[str, float]]:
        material_key = getattr(material, "name", str(material)).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        rms_in = float(np.sqrt(np.mean(np.asarray(original_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_drop_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30)) if rms_in > 1e-8 else 0.0
        makeup_gain_db = 0.0

        if rms_in > 1e-8 and rms_drop_db < -max_rms_drop_db:
            target_rms_drop_db = -max_rms_drop_db
            required_gain_db = target_rms_drop_db - rms_drop_db
            # §2.45a-II fix: apply full gain — peak-headroom cap disabled (see phase_05 fix).
            # §2.45a-III: soft-limiter only when peak99 > 0.98.
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, 6.0))
            if makeup_gain_db > 0.0:
                processed_audio = np.clip(
                    processed_audio * (10.0 ** (makeup_gain_db / 20.0)),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                current_peak = float(np.percentile(np.abs(processed_audio), 99.9))
                if current_peak > 0.98:
                    _abs_29 = np.abs(processed_audio)
                    _over_29 = _abs_29 > 0.92
                    if np.any(_over_29):
                        _sign_29 = np.sign(processed_audio)
                        processed_audio = np.where(
                            _over_29, _sign_29 * (0.92 + 0.08 * np.tanh((_abs_29 - 0.92) / 0.08)), processed_audio
                        )
                processed_audio = np.clip(processed_audio, -1.0, 1.0).astype(np.float32)
                rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
                rms_drop_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30))
                logger.info(
                    "Phase 29 loudness-preservation: material=%s rms_drop=%.2f dB via makeup %.2f dB",
                    material_key,
                    rms_drop_db,
                    makeup_gain_db,
                )

        return processed_audio, {
            "rms_drop_db": round(float(rms_drop_db), 3),
            "makeup_gain_db": round(float(makeup_gain_db), 3),
        }

    def _process_channel_omlsa(
        self,
        channel: np.ndarray,
        sample_rate: int,
        hf_low: float,
        hf_high: float,
        material: "MaterialType",
        intensity_scale: float = 1.0,
        masking_result=None,  # §Psychoacoustic: pre-computed MaskingResult for this channel
        linked_sidechain: np.ndarray | None = None,  # §2.51: Mid-channel for linked gain computation
    ) -> np.ndarray:
        """STFT-OMLSA-Verarbeitung: HF-selektive Rauschunterdrückung (Cohen 2002/2003).

        Algorithmus:
            1. STFT (nperseg=2048, noverlap=1536)
            2. IMCRA-Rauschschätzung im HF-Bereich [hf_low, hf_high]
            3. OMLSA-Gain: G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p
            4. Bins < hf_low: G=1.0 (unangetastet — Tieftonschutz)
            5. Cappé-Glättung: alpha_g = 0.85
            6. ISTFT + NaN/Clip-Schutz

        §2.51 Linked-Sidechain: when ``linked_sidechain`` (Mid) is provided, the
        IMCRA noise estimation and OMLSA gain are computed from the sidechain signal
        so that L and R receive the **identical** gain mask — stereo-coherent.

        Args:
            channel:           Mono-Audio (1D float32)
            sample_rate:       Abtastrate in Hz
            hf_low:            Untere HF-Grenze (Hz), z.B. 8000
            hf_high:           Obere HF-Grenze (Hz), z.B. 18000
            material:          MaterialType für G_floor
            linked_sidechain:  Optional Mid-channel for linked stereo gain computation

        Returns:
            processed: Restauriertes Mono-Audio (gleiche Länge wie channel)
        """
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
        intensity_scale = float(np.clip(intensity_scale, 0.0, 1.0))
        # Raise floor towards 1.0 for conservative locality handling.
        G_floor = float(np.clip(1.0 - intensity_scale * (1.0 - G_floor), 0.0, 1.0))

        # STFT + OMLSA via MRSA 5-zone processing (§DSP-Spezialregeln)
        # §2.51: When linked_sidechain is provided, IMCRA/OMLSA gain is computed on
        # the sidechain (Mid) but applied to this channel's STFT magnitudes.
        processed = self._process_channel_omlsa_mrsa(
            channel,
            sample_rate,
            hf_low,
            hf_high,
            material,
            intensity_scale,
            linked_sidechain=linked_sidechain,
        )
        processed = processed[: len(channel)]
        if len(processed) < len(channel):
            processed = np.pad(processed, (0, len(channel) - len(processed)))
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # §4.5 Psychoakustischer Masking-Gain-Clamp (ISO 11172-3, Painter & Spanias 2000)
        # Berechnet auf Input-Audio → Schutzmaske für Stille / ungemaskierte Bereiche
        try:
            _pmm = masking_result
            if _pmm is None:
                from backend.core.psychoacoustic_masking_model import compute_masking_threshold

                _pmm = compute_masking_threshold(channel.astype(np.float32), sample_rate)
            # Use max over Bark bands: if ANY band has active signal, the frame must not be suppressed.
            # mean() was incorrect — a 1 kHz sine only has energy in ~1 of 24 bands → mean ≈ 0.33,
            # causing 3× RMS reduction on clean tonal signals. max() gives 1.0 for active frames,
            # and correctly reduces only true silence frames (all bands near g_floor).
            _pmm_gain_t = np.max(_pmm.gain_modifier, axis=1).astype(np.float32)
            _hop = 512  # entspricht nperseg=2048, noverlap=1536
            _pmm_centers = np.arange(len(_pmm_gain_t)) * float(_hop) + _hop * 0.5
            _pmm_x = np.arange(len(processed), dtype=np.float32)
            _gain_samples = np.interp(_pmm_x, _pmm_centers, _pmm_gain_t).astype(np.float32)
            # §2.45a / §2.54: Scale masking suppression toward 1.0 by intensity_scale.
            # At low PMGG strength (e.g. 0.14) the masking clamp must be near-transparent
            # — otherwise full suppression runs regardless of strength causing unexpected
            # RMS drops, makeup-gain overshoot and TFS coherence degradation.
            _gain_samples_scaled = (1.0 + intensity_scale * (_gain_samples - 1.0)).astype(np.float32)
            processed = np.clip((processed * _gain_samples_scaled).astype(np.float32), -1.0, 1.0)
            logger.debug(
                "🎭 PsychoacousticMasking [phase29]: silence=%.1f%% mean_gain=%.3f scaled_mean=%.3f (scale=%.2f)",
                100.0 * float(np.mean(_pmm.silence_frames)),
                float(np.mean(_pmm_gain_t)),
                float(np.mean(_gain_samples_scaled)),
                intensity_scale,
            )
        except Exception as _pmm_exc:
            logger.debug("PsychoacousticMaskingModel nicht verfügbar: %s", _pmm_exc)

        return processed

    def _process_channel_omlsa_mrsa(
        self,
        channel: np.ndarray,
        sample_rate: int,
        hf_low: float,
        hf_high: float,
        material: "MaterialType",
        intensity_scale: float = 1.0,
        linked_sidechain: np.ndarray | None = None,
    ) -> np.ndarray:
        """MRSA 5-zone OMLSA/IMCRA tape-hiss reduction with PGHI phase reconstruction.

        Multi-Resolution Spectral Analysis (MRSA): each frequency zone is processed
        at its optimal time-frequency resolution. Zones below hf_low receive pass-through
        (gain=1.0), protecting low-frequency content. PGHI replaces plain iSTFT.

        §2.51 Linked-Sidechain: when ``linked_sidechain`` (Mid) is provided, IMCRA noise
        estimation runs on the sidechain signal, producing a stereo-coherent gain mask
        that is applied to the actual channel's STFT. This prevents L/R asymmetric
        gain modulation that causes phantom-center instability on tape material.

        Args:
            channel:           Mono audio [1D float32].
            sample_rate:       Must be 48000.
            hf_low:            Lower HF gate boundary (Hz), e.g. 8000.
            hf_high:           Upper HF gate boundary (Hz), e.g. 18000.
            material:          MaterialType for G_floor selection.
            intensity_scale:   Locality factor ∈ [0, 1].
            linked_sidechain:  Optional Mid-channel for linked stereo gain computation.

        Returns:
            Processed mono audio, same length as input.
        """
        n = len(channel)
        nyquist = float(sample_rate // 2)
        eps = 1e-10

        # Material-adaptive G_floor
        G_floor_map = {"SHELLAC": 0.12, "VINYL": 0.10, "TAPE": 0.08, "REEL_TAPE": 0.07, "DAT": 0.06}
        mat_name = getattr(material, "name", str(material)).upper()
        G_floor = G_floor_map.get(mat_name, 0.10)
        intensity_scale = float(np.clip(intensity_scale, 0.0, 1.0))
        G_floor = float(np.clip(1.0 - intensity_scale * (1.0 - G_floor), 0.0, 1.0))
        q = 0.5
        b_min = 1.66
        alpha_g = 0.85

        # Reference STFT (win=2048, 75 % overlap) — on channel (for magnitude application)
        REF_WIN = 2048
        REF_HOP = 512
        REF_NOVERLAP = REF_WIN - REF_HOP
        f_ref, _, Zxx_ref = signal.stft(channel, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann")
        n_bins, n_t = f_ref.shape[0], Zxx_ref.shape[1]

        # §2.51 Linked-Sidechain: compute gain from Mid sidechain for stereo coherence.
        # If no sidechain is provided, gain is computed from the channel itself (mono path).
        _gain_source = linked_sidechain if linked_sidechain is not None else channel

        G_acc = np.zeros((n_bins, n_t), dtype=np.float64)
        w_acc = np.zeros(n_bins, dtype=np.float64)

        for zone_name, zone_win, zone_hop, f_low, f_high in self._MRSA_ZONES:
            try:
                # §2.51: STFT for gain computation uses _gain_source (Mid sidechain
                # for stereo, or channel itself for mono).
                if n >= zone_win * 2:
                    zone_noverlap = zone_win - zone_hop
                    f_z, _, Zxx_z = signal.stft(
                        _gain_source, fs=sample_rate, nperseg=zone_win, noverlap=zone_noverlap, window="hann"
                    )
                else:
                    # Fallback to reference STFT — recompute from gain source if linked
                    if linked_sidechain is not None:
                        f_z, _, Zxx_z = signal.stft(
                            _gain_source, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann"
                        )
                    else:
                        f_z, Zxx_z = f_ref, Zxx_ref
                    zone_win, zone_hop = REF_WIN, REF_HOP

                mag_z = np.abs(Zxx_z)
                n_z_t = mag_z.shape[1]
                frames_per_sec_z = float(sample_rate / zone_hop)
                M_z = max(3, int(1.5 * frames_per_sec_z))

                # Vectorised IMCRA: sliding minimum as noise estimate (Cohen 2003)
                power_z = mag_z**2
                S_min_z = _min_filter1d_p29(power_z, size=M_z, axis=1, mode="reflect")
                noise_sq_z = np.maximum(b_min * S_min_z, eps)

                # Vectorised OMLSA gain
                gamma_z = power_z / noise_sq_z
                xi_z = np.maximum(gamma_z - 1.0, 0.0)
                nu_z = np.clip(xi_z * gamma_z / (xi_z + 1.0 + eps), 0.0, 500.0)
                lam_z = np.exp(np.clip(-xi_z + nu_z, -50.0, 50.0))
                p_z = 1.0 / (1.0 + q / ((1.0 - q) * lam_z + eps))
                G_H1_z = xi_z / (xi_z + 1.0 + eps)
                log_G_z = (1.0 - p_z) * np.log(G_floor + eps) + p_z * np.log(np.maximum(G_H1_z, eps))
                G_z = np.exp(np.clip(log_G_z, np.log(G_floor + eps), 0.0))
                G_z = np.clip(np.nan_to_num(G_z, nan=G_floor), G_floor, 1.0)

                # §v9.10.113: Stronger HF suppression in presence/air zones when DeepFilterNet absent.
                # DeepFilterNet removes residual hiss 2–16 kHz; without it, G_floor must be lower.
                # TAPE: 0.08 → 0.036, VINYL: 0.10 → 0.045, SHELLAC: 0.12 → 0.054 in these zones.
                if zone_name in ("presence", "air") and intensity_scale > 0.40:
                    _hf_floor = float(np.clip(G_floor * 0.45, 0.020, G_floor))
                    G_z = np.clip(G_z, _hf_floor, 1.0)

                # Zones below hf_low: pass-through (protect low frequencies)
                lf_mask_z = f_z < float(hf_low)
                G_z[lf_mask_z, :] = 1.0
                # Zones above hf_high (Nyquist region): pass-through
                if float(hf_high) < nyquist:
                    hf_mask_z = f_z > float(hf_high)
                    G_z[hf_mask_z, :] = 1.0

                # Cappé temporal smoothing via fast IIR
                G_z_sm = _lfilter_p29([1.0 - alpha_g], [1.0, -alpha_g], G_z, axis=1)
                G_z_sm = np.clip(np.nan_to_num(G_z_sm, nan=G_floor), G_floor, 1.0)

                # Extract zone frequency range
                zm_z = (f_z >= float(f_low)) & (f_z <= float(f_high))
                if not np.any(zm_z):
                    continue
                f_z_zone = f_z[zm_z]
                G_zone = G_z_sm[zm_z, :]

                # Reference bins for this zone (with crossfade bandwidth)
                ref_zm = (f_ref >= max(0.0, float(f_low) - self._MRSA_CROSSFADE_BW_HZ)) & (
                    f_ref <= min(nyquist, float(f_high) + self._MRSA_CROSSFADE_BW_HZ)
                )
                if not np.any(ref_zm):
                    continue
                f_ref_zone = f_ref[ref_zm]
                ref_indices = np.where(ref_zm)[0]
                n_ref_zone = len(ref_indices)

                # Temporal resampling
                if n_z_t != n_t and len(f_z_zone) > 0:
                    t_src = np.linspace(0.0, 1.0, n_z_t)
                    t_dst = np.linspace(0.0, 1.0, n_t)
                    G_zone_t = np.empty((len(f_z_zone), n_t), dtype=np.float64)
                    for k in range(len(f_z_zone)):
                        G_zone_t[k, :] = np.interp(t_dst, t_src, G_zone[k, :])
                else:
                    G_zone_t = G_zone.astype(np.float64)

                # Frequency interpolation
                G_ref_zone = np.empty((n_ref_zone, n_t), dtype=np.float64)
                if len(f_z_zone) >= 2:
                    for ti in range(n_t):
                        G_ref_zone[:, ti] = np.interp(
                            f_ref_zone,
                            f_z_zone,
                            G_zone_t[:, ti],
                            left=float(G_zone_t[0, ti]),
                            right=float(G_zone_t[-1, ti]),
                        )
                elif len(f_z_zone) == 1:
                    G_ref_zone[:, :] = G_zone_t[0:1, :]
                else:
                    continue

                # Hanning crossfade weights
                if n_ref_zone > 2:
                    hann_w = np.hanning(n_ref_zone + 2)[1:-1]
                    hann_w = np.clip(hann_w, 1e-3, 1.0)
                else:
                    hann_w = np.ones(n_ref_zone)

                for ki, k in enumerate(ref_indices):
                    w = float(hann_w[ki])
                    G_acc[k, :] += w * G_ref_zone[ki, :]
                    w_acc[k] += w

            except Exception as zone_exc:
                logger.warning("MRSA Phase 29 zone '%s' failed: %s", zone_name, zone_exc)
                continue

        # Combine zone gains; unprocessed bins → pass-through
        valid = w_acc > 0.0
        G_combined = np.ones((n_bins, n_t), dtype=np.float32)
        G_combined[valid, :] = (G_acc[valid, :] / w_acc[valid, np.newaxis]).astype(np.float32)
        G_combined = np.clip(np.nan_to_num(G_combined, nan=1.0), 0.0, 1.0)

        # HF detail protection: preserve salient tape harmonics/transients in 6-18 kHz
        # while still reducing stationary hiss in low-salience bins.
        _mat = getattr(material, "name", str(material)).upper()
        _base_floor_by_mat = {
            "TAPE": 0.11,
            "REEL_TAPE": 0.10,
            "VINYL": 0.09,
            "SHELLAC": 0.09,
            "DAT": 0.08,
        }
        _hf_guard_low = max(float(hf_low), 6000.0)
        _hf_guard_high = min(float(hf_high), 18000.0)
        _hf_guard_mask = (f_ref >= _hf_guard_low) & (f_ref <= _hf_guard_high)
        if np.any(_hf_guard_mask):
            _mag_hf = np.abs(Zxx_ref[_hf_guard_mask, :]).astype(np.float64)
            _bin_sal = np.median(_mag_hf, axis=1)
            _bin_den = float(np.percentile(_bin_sal, 95) + eps)
            _bin_sal_n = np.clip(_bin_sal / _bin_den, 0.0, 1.0)

            _frame_den = np.percentile(_mag_hf, 90, axis=1, keepdims=True) + eps
            _frame_sal_n = np.clip(_mag_hf / _frame_den, 0.0, 1.0)

            _base_floor = float(_base_floor_by_mat.get(_mat, 0.09))
            # With higher intensity we still preserve enough HF detail to avoid dullness.
            _floor_min = float(np.clip(_base_floor - 0.03 * intensity_scale, 0.07, 0.16))
            _bin_floor = np.clip(_floor_min + 0.12 * _bin_sal_n, _floor_min, 0.30)
            _dyn_floor = _bin_floor[:, None] + 0.08 * _frame_sal_n
            _dyn_floor = np.clip(_dyn_floor, _bin_floor[:, None], 0.36).astype(np.float32)
            G_combined[_hf_guard_mask, :] = np.maximum(G_combined[_hf_guard_mask, :], _dyn_floor)

        # Apply gain + PGHI reconstruction
        Zxx_proc = G_combined * np.abs(Zxx_ref) * np.exp(1j * np.angle(Zxx_ref))
        if _PGHI_AVAILABLE_P29:
            try:
                audio_out = _pghi_p29(
                    Zxx_proc.astype(np.complex64), sr=sample_rate, win_size=REF_WIN, hop=REF_HOP, n_samples=n
                )
            except Exception as pghi_exc:
                logger.warning("MRSA Phase 29: PGHI failed, iSTFT fallback: %s", pghi_exc)
                _, audio_out = signal.istft(
                    Zxx_proc, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann"
                )
        else:
            _, audio_out = signal.istft(Zxx_proc, fs=sample_rate, nperseg=REF_WIN, noverlap=REF_NOVERLAP, window="hann")

        audio_out = np.real(audio_out)
        audio_out = audio_out[:n]
        if len(audio_out) < n:
            audio_out = np.pad(audio_out, (0, n - len(audio_out)))
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0).astype(np.float32)

        logger.debug(
            "MRSA Phase 29: 5 zones processed, valid_bins=%d/%d, G_mean=%.3f, linked_sidechain=%s",
            int(np.sum(valid)),
            n_bins,
            float(np.mean(G_combined)),
            linked_sidechain is not None,
        )
        return audio_out

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
            returncode, _stdout, _stderr = plugin.process(
                input_path,
                output_path,
                post_filter=True,  # Enable post-filter for smooth HF reduction
            )

            if returncode == 0 and os.path.exists(output_path):
                # Read refined audio
                from backend.file_import import load_audio_file

                _res = load_audio_file(output_path, do_carrier_analysis=False)
                if not _res or "audio" not in _res:
                    return False
                refined = np.asarray(_res["audio"], dtype=np.float32)

                # Blend strategy: Keep <2kHz from original, use ML for >2kHz
                if refined.shape == audio.shape:
                    # Extract HF bands
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
                    logger.warning("Shape mismatch: %s vs %s", refined.shape, audio.shape)
                    return False
            else:
                logger.warning("DeepFilterNet failed (returncode=%s)", returncode)
                return False

        except Exception as e:
            logger.error("ML HF refinement error: %s", e)
            return False

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

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
        logger.debug("Testing %s:", material.value.upper())

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
        result = processor.process(audio, sr, material)
        processed = result.audio
        meta = result.metadata or {}
        elapsed = time.time() - start

        # Calculate HF noise reduction
        sos_hf = signal.butter(4, 8000, "high", fs=sr, output="sos")
        hf_orig = signal.sosfilt(sos_hf, audio[:, 0])
        hf_proc = signal.sosfilt(sos_hf, processed[:, 0])

        hf_reduction = 20 * np.log10(np.std(hf_orig) / (np.std(hf_proc) + 1e-10))

        # Display results
        logger.debug("  Gate threshold: %.1f dB", meta.get("gate_threshold_db", 0))
        logger.debug("  Reduction depth: %.1f dB", meta.get("reduction_depth_db", 0))
        logger.debug("  HF focus range: %s Hz", meta.get("hf_focus_range_hz", []))
        logger.debug("  Num bands: %s", meta.get("num_bands", 0))
        logger.debug("  HF reduction: %.2f dB", meta.get("hf_reduction_db", 0))
        logger.debug("  Per-band reduction: %s... (first 3)", meta.get("reduction_per_band_db", [])[:3])
        logger.debug("  Processing time: %.3fs", elapsed)
        logger.debug("  ✅\n")
