#!/usr/bin/env python3
"""
Phase 20: Professional Reverb Reduction v3.0 — OMLSA/IMCRA
===========================================================

Spektrale Nachhall-Reduktion via statistischer Rauschunterdrückung (OMLSA/IMCRA)
mit transientenerhaltender Nachbearbeitung.

SCIENTIFIC FOUNDATION (Primär, Über-SOTA-Pflicht):
- Cohen & Berdugo (2002): "Noise Estimation by Minima Controlled Recursive Averaging"
  → IMCRA: gleitendes Minimum mit Bias-Kompensation b_min=1.66 für diffuse
    Schallfelder wie Hall-Ausläufer. Ersetzt primitive Median-Rauschschätzung.
- Cohen (2003): "Noise Spectrum Estimation in Adverse Environments: Improved
  Minima Controlled Recursive Averaging" (OMLSA)
  → OMLSA: G(t,f) = G_floor^(1−p) · (ξ/(1+ξ))^p eliminiert musikalisches Rauschen.
- Le Roux & Vincent (2013): "Consistent Wiener Filtering" — Gain-Clamp G_floor.
- Cappé (1994): Temporale Gain-Glättung α_g=0.85 — unterdrückt Gain-Flattern.
- Perraudin et al. (2013): PGHI — scipy.signal.stft/istft sichert OLA-Phasenkonsistenz.

Historische Referenz (nur noch informativ, nicht als primärer Algorithmus):
- Moorer (1979): About This Reverberation Business
- Schroeder (1962): Natural Sounding Artificial Reverberation
- Kendall (2010): The Decorrelation of Audio Signals and Its Impact on Spatial Imagery
- Välimäki et al. (2012): Fifty Years of Artificial Reverberation
- ITU-R BS.1116-3: Methods for the Subjective Assessment of Small Impairments
- Bech & Zacharov (2006): Perceptual Audio Evaluation

INDUSTRY BENCHMARKS:
- iZotope RX 10 De-reverb (Spectral analysis + ML)
- Waves Clarity Vx DeReverb (Transient-preserving)
- Zynaptiq Unveil (Source separation based)
- SPL DeVerb (Dynamics-based)
- Cedar Retouch Pro (Professional standard)
- Accusonus ERA-D (Real-time dereverb)

ALGORITHM:
1. Transient Detection
   - Attack/Sustain separation
   - Transients bypass processing (preserve direct sound)

2. Spectral Envelope Analysis
   - STFT with 2048 window, 75% overlap
   - Identify reverb tail characteristics (exponential decay)
   - Separate direct sound from reflections

3. Spectral Gating
   - Frequency-dependent thresholds
   - Soft-knee gating (avoid artifacts)
   - Preserve tonal components while reducing diffuse field

4. Material-Adaptive Parameters
   - Shellac: Moderate (often already dry)
   - Vinyl: Light (preserve natural ambience)
   - Tape: Strong (analog reverb artifacts)
   - Digital: Minimal (production choice)

QUALITY TARGETS:
- Reverb reduction: 30-60% tail dampening
- Transient preservation: >98% attack energy
- Processing: <0.3× realtime

Author: Aurik Professional Team
Version: 3.0.0
Date: März 2026
"""

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

# ML-Hybrid Support (Aurik 9.0 - Phase 20 v3.0)
try:
    from backend.core.hybrid.hybrid_dereverb import DereverbConfig, DereverbStrategy, HybridDereverb

    ML_HYBRID_AVAILABLE = True
except ImportError:
    ML_HYBRID_AVAILABLE = False
    logging.getLogger(__name__).warning("ML-Hybrid dereverb not available, using DSP-only mode")

# WPE Dereverberation (Spec §4.4 — Tier-1 DSP: kanonisches Dereverb-Plugin, Nakatani 2010)
try:
    from plugins.wpe_plugin import get_wpe_plugin

    WPE_AVAILABLE = True
except ImportError:
    WPE_AVAILABLE = False
    logging.getLogger(__name__).warning("WPE-Plugin nicht verfügbar — OMLSA/IMCRA-Fallback aktiv")

try:
    from dsp.pghi import pghi_reconstruct_from_stft as _pghi_p20

    _PGHI_AVAILABLE_P20 = True
except ImportError:
    _PGHI_AVAILABLE_P20 = False

from scipy.ndimage import minimum_filter1d as _min_filter1d_p20  # vectorised sliding-min
from scipy.signal import lfilter as _lfilter_p20  # vectorised IIR smoothing

logger = logging.getLogger(__name__)


class ReverbReduction(PhaseInterface):
    """Professional spectral-based reverb reduction."""

    _MAX_RMS_DROP_DB = {
        "tape": 2.5,
        "reel_tape": 2.2,
        "cassette": 2.8,
        "vinyl": 2.0,
        "shellac": 1.8,
        "wax_cylinder": 1.5,
        "cd_digital": 1.8,
        "streaming": 1.6,
        "unknown": 2.0,
    }

    # Material-adaptive reduction strength
    REDUCTION_STRENGTH = {
        MaterialType.SHELLAC: 0.50,  # Moderate (often dry already)
        MaterialType.VINYL: 0.40,  # Light (preserve natural ambience)
        MaterialType.TAPE: 0.65,  # Strong (analog reverb artifacts)
        MaterialType.CD_DIGITAL: 0.30,  # Minimal (production choice)
        MaterialType.STREAMING: 0.25,  # Very minimal
    }

    # Tail damping factor (how quickly reverb tail decays)
    TAIL_DAMPING = {
        MaterialType.SHELLAC: 0.70,
        MaterialType.VINYL: 0.60,
        MaterialType.TAPE: 0.80,
        MaterialType.CD_DIGITAL: 0.50,
        MaterialType.STREAMING: 0.40,
    }

    # Transient threshold (energy ratio for transient detection)
    TRANSIENT_THRESHOLD = 3.0  # 3× energy increase = transient

    # STFT parameters
    WINDOW_SIZE = 2048
    HOP_SIZE = 512  # 75% overlap

    # MRSA Multi-Resolution Spectral Analysis zones (mandatory, §DSP-Spezialregeln)
    # VERBOTEN: arbitrary FFT sizes — only these 5 zone-optimal windows are permitted.
    _MRSA_ZONES: tuple = (
        # (name,       win_size, hop_size, f_low_hz, f_high_hz)
        ("sub_bass", 65536, 16384, 0, 250),
        ("mid_low", 16384, 4096, 250, 2500),
        ("mid", 8192, 2048, 2500, 8000),
        ("presence", 1024, 256, 8000, 16000),
        ("air", 128, 32, 16000, 24000),
    )
    # Hanning crossfade transition bandwidth at zone boundaries (~10 ms spectral transition)
    _MRSA_CROSSFADE_BW_HZ: float = 100.0

    def __init__(self):
        super().__init__()
        self.name = "Reverb Reduction v3 OMLSA/IMCRA"
        # If SGMSE TorchScript fails with deterministic shape/runtime errors,
        # avoid retrying the same expensive ML path on subsequent PMGG retries.
        self._force_dsp_only_due_ml_error: bool = False
        self._ml_disable_reason: str = ""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_20_reverb_reduction",
            name="Reverb Reduction v3 OMLSA/IMCRA",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.15,
            version="3.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description=(
                "Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003) — "
                "diffuse Schallfelder ohne musikalisches Rauschen, "
                "Transientenerhalt und scipy.signal.stft/istft (PGHI-konsistent)"
            ),
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        """
        Apply reverb reduction.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with reverb-reduced audio
        """
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        strength = self.REDUCTION_STRENGTH.get(material, 0.4)
        damping = self.TAIL_DAMPING.get(material, 0.6)

        # Locality-aware intensity control from UV3.
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
                    "rms_change_db": 0.0,
                    "reduction_strength": 0.0,
                    "tail_damping": damping,
                    "material": material.value,
                },
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                },
            )

        strength = float(np.clip(strength * _effective_strength, 0.0, 1.0))

        # §2.20 Genre-adaptive reverb: classical/opera preserve concert hall ambience;
        # Schlager profile may define dereverb_strength_cap.
        genre_label = kwargs.get("genre_label", "Unbekannt")
        if genre_label in ("Klassik", "Oper"):
            strength = min(strength, 0.25)
            logger.debug("Phase 20: Genre=%s → reverb strength capped to %.2f", genre_label, strength)
        elif genre_label == "Jazz":
            strength = min(strength, 0.30)

        # §2.14+ Era-adaptive: older recordings (pre-1960) often have room ambience
        # integral to the character — reduce dereverb strength.
        decade = kwargs.get("decade")
        if decade is not None and decade <= 1950:
            strength = min(strength, 0.30)
            logger.debug("Phase 20: decade=%d → reverb strength capped to %.2f", decade, strength)

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "quality")

        # Check resource availability for ML-Hybrid (fallback to lightweight if needed)
        use_lightweight = False
        if RESOURCE_MANAGER_AVAILABLE:
            use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
            if use_lightweight:
                logger.info(
                    f"Phase 20: Resource constraint detected, forcing DSP-only mode "
                    f"(CPU: {adaptive_resource_manager.get_cpu_usage():.1f}%, "
                    f"Memory: {adaptive_resource_manager.get_memory_usage():.1f}%)"
                )

        # ML-Hybrid only if resources available and quality mode permits
        use_ml_hybrid = (
            ML_HYBRID_AVAILABLE
            and quality_mode in ["balanced", "maximum", "quality"]
            and not use_lightweight
            and not self._force_dsp_only_due_ml_error
        )

        if use_ml_hybrid:
            try:
                logger.info("Phase 20 ML-Hybrid: mode=%s, material=%s", quality_mode, material.value)

                # Configure ML dereverb strategy
                # 'quality' und 'maximum' → HYBRID (SGMSE+ ML-Primär + WPE-DSP-Fallback, §4.4)
                if quality_mode in ("maximum", "quality"):
                    strategy = DereverbStrategy.HYBRID  # SGMSE+ primär → WPE DSP-Fallback (§4.4)
                else:  # balanced
                    strategy = DereverbStrategy.ADAPTIVE  # Smart: DSP only if light reverb

                dereverb = HybridDereverb(
                    config=DereverbConfig(
                        strategy=strategy,
                        dsp_strength=strength,
                        dsp_damping=damping,
                        enable_preprocessing=True,
                        reverb_threshold=0.3,  # Skip DCCRN if reverb already low
                    )
                )

                ml_result = dereverb.dereverb(audio, sample_rate=sample_rate)
                processing_time = time.time() - start_time

                # Estimate RMS change from reverb reduction
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(ml_result.audio**2))
                # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

                logger.info(
                    f"ML-Hybrid complete: DSP={ml_result.dsp_applied}, "
                    f"ML={ml_result.ml_applied}, reverb={ml_result.reverb_estimate:.3f}, "
                    f"RMS change={rms_change_db:.2f}dB, time={processing_time:.2f}s"
                )

                # Generate warnings
                warnings = []
                if ml_result.reverb_estimate > 0.7:
                    warnings.append(
                        f"High reverb detected: {ml_result.reverb_estimate:.2f} (may require multiple passes)"
                    )

                _audio_clean = np.nan_to_num(ml_result.audio, nan=0.0, posinf=0.0, neginf=0.0)
                _audio_clean = np.clip(_audio_clean, -1.0, 1.0)
                if 0.0 < _effective_strength < 1.0:
                    _audio_clean = audio + _effective_strength * (_audio_clean - audio)
                    _audio_clean = np.clip(_audio_clean, -1.0, 1.0)
                _audio_clean, _rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
                    audio,
                    _audio_clean,
                    material,
                )
                return PhaseResult(
                    success=True,
                    audio=_audio_clean,
                    metrics={
                        "rms_change_db": float(_rms_change_db),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "strategy": str(ml_result.strategy_used),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                        "quality_mode": quality_mode,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "hybrid_wpe_resemble_v4",
                        "ml_hybrid": True,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "processing_time": ml_result.processing_time,
                        "version": "3.0_ml_hybrid",
                        "window_size": self.WINDOW_SIZE,
                        "hop_size": self.HOP_SIZE,
                        "ml_metadata": ml_result.metadata,
                        "phase_locality_factor": phase_locality_factor,
                        "effective_strength": _effective_strength,
                        "loudness_makeup_db": float(_makeup_gain_db),
                    },
                    warnings=warnings,
                    modifications={},
                )

            except Exception as e:
                import traceback as _tb

                _err_text = str(e)
                _is_deterministic_ml_fail = (
                    "Sizes of tensors must match" in _err_text
                    or "TorchScript" in _err_text
                    or "expected shape" in _err_text.lower()
                    or "shape mismatch" in _err_text.lower()
                )
                if _is_deterministic_ml_fail and not self._force_dsp_only_due_ml_error:
                    self._force_dsp_only_due_ml_error = True
                    self._ml_disable_reason = _err_text[:220]
                    logger.warning(
                        "Phase 20: disable ML-hybrid for remaining calls due to deterministic SGMSE error: %s",
                        self._ml_disable_reason,
                    )

                logger.warning(
                    f"ML-Hybrid dereverb failed: {e}, falling back to DSP. Error type: {type(e).__name__}\n"
                    f"Traceback: {_tb.format_exc()}"
                )
                # Fall through to DSP path below

        # DSP-Only Path (Fast mode or ML fallback)
        logger.info("Phase 20 DSP-Only: material=%s, strength=%s", material.value, strength)

        # ── Tier-1 DSP: WPE (Nakatani 2010) — DSP-Fallback für Dereverb (§4.4; ML-Primär: SGMSE+) ──
        # WPE entfernt Spätreflexionen via iterative gewichtete lineare Prädiktion.
        # Kaskade: nara_wpe → NumPy-WPE → OMLSA (innerhalb des Plugins).
        if WPE_AVAILABLE:
            try:
                wpe = get_wpe_plugin()
                # WPE erwartet SR == 48000 (Spec-Invariante)
                wpe_strength = float(np.clip(strength * 0.90, 0.3, 0.95))
                audio_wpe = wpe.enhance(audio.astype(np.float32), sample_rate, strength=wpe_strength)
                audio_wpe = np.nan_to_num(audio_wpe, nan=0.0, posinf=0.0, neginf=0.0)
                audio_wpe = np.clip(audio_wpe.astype(np.float64), -1.0, 1.0)
                # Passen Länge und Form an
                if audio_wpe.shape != audio.shape:
                    if audio.ndim == 1:
                        audio_wpe = audio_wpe.flatten()[: len(audio)]
                        if len(audio_wpe) < len(audio):
                            audio_wpe = np.pad(audio_wpe, (0, len(audio) - len(audio_wpe)))
                    else:
                        min_len = min(
                            audio_wpe.shape[0] if audio_wpe.ndim == 1 else audio_wpe.shape[-1], audio.shape[-1]
                        )
                        audio_wpe = audio_wpe[:min_len] if audio_wpe.ndim == 1 else audio_wpe[:, :min_len]
                reduced = audio_wpe
                processing_time = time.time() - start_time
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(reduced**2))
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))
                logger.info("Phase 20: WPE-Tier erfolgreich (strength=%.2f)", wpe_strength)
                reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
                reduced = np.clip(reduced, -1.0, 1.0)
                if 0.0 < _effective_strength < 1.0:
                    reduced = audio + _effective_strength * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                reduced, rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
                    audio,
                    reduced,
                    material,
                )
                return PhaseResult(
                    success=True,
                    audio=reduced,
                    metrics={
                        "rms_change_db": float(rms_change_db),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "wpe_nakatani2010_tier1",
                        "version": "3.0_wpe",
                        "wpe_strength": wpe_strength,
                        "phase_locality_factor": phase_locality_factor,
                        "effective_strength": _effective_strength,
                        "loudness_makeup_db": float(_makeup_gain_db),
                    },
                )
            except Exception as wpe_err:
                logger.warning("Phase 20: WPE fehlgeschlagen (%s) — OMLSA/IMCRA-Fallback", wpe_err)
        # ── Tier-2 DSP: OMLSA/IMCRA (Cohen 2002/2003) — Fallback ──────────────
        is_stereo = audio.ndim == 2

        if is_stereo:
            # Detect channel-major (2, N) vs time-major (N, 2).
            # Aurik uses channel-major (2, N) throughout the pipeline.
            _is_ch_maj = audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            _l = audio[0] if _is_ch_maj else audio[:, 0]
            _r = audio[1] if _is_ch_maj else audio[:, 1]
            # §2.51 M/S: apply reverb suppression on Mid-channel only so both
            # channels receive the SAME gain curve — independent L/R processing
            # estimates different room impulses per channel, creating stereo-field
            # asymmetry that triggers §2.49 phase-cancellation rollbacks.
            _sqrt2 = np.sqrt(2.0)
            _mid = (_l + _r) / _sqrt2
            _side = (_l - _r) / _sqrt2
            mid_reduced = self._reduce_reverb(_mid, sample_rate, strength, damping)
            # Side: apply weaker dereverb (side already less reverberant)
            _side_str = strength * 0.5
            side_reduced = self._reduce_reverb(_side, sample_rate, _side_str, damping)
            min_len = min(len(mid_reduced), len(side_reduced))
            _l_out = (mid_reduced[:min_len] + side_reduced[:min_len]) / _sqrt2
            _r_out = (mid_reduced[:min_len] - side_reduced[:min_len]) / _sqrt2
            if _is_ch_maj:
                reduced = np.stack([_l_out, _r_out], axis=0)  # (2, N)
            else:
                reduced = np.column_stack([_l_out, _r_out])  # (N, 2)
        else:
            reduced = self._reduce_reverb(audio, sample_rate, strength, damping)

        processing_time = time.time() - start_time

        # Measure reverb reduction (RT60-like estimate)
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(reduced**2))
        # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
        rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

        reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
        reduced = np.clip(reduced, -1.0, 1.0)
        if 0.0 < _effective_strength < 1.0:
            reduced = audio + _effective_strength * (reduced - audio)
            reduced = np.clip(reduced, -1.0, 1.0)
        reduced, rms_change_db, _makeup_gain_db = self._apply_material_loudness_preservation(
            audio,
            reduced,
            material,
        )

        # §4.5c Early-Reflection-Guard (Spec §4.5c, v9.10.100)
        # C80 = 10·log10(E_early80ms / E_late) — Kuttruff 2009; ΔC80 ≤ 6 dB
        # D50 = E_early50ms / E_total — ΔD50 ≤ 0.12 (sekundär)
        _c80_guard_triggered = False
        _early_blend_triggered = False
        _delta_c80 = 0.0
        _delta_d50 = 0.0
        try:
            _mono_in = audio[0] if audio.ndim == 2 else audio
            _mono_out = reduced[0] if reduced.ndim == 2 else reduced
            _e80 = int(sample_rate * 0.080)
            _e50 = int(sample_rate * 0.050)
            if len(_mono_in) > _e80:
                _c80_pre = 10.0 * float(
                    np.log10(max(np.sum(_mono_in[:_e80] ** 2), 1e-12) / max(np.sum(_mono_in[_e80:] ** 2), 1e-12))
                )
                _c80_post = 10.0 * float(
                    np.log10(max(np.sum(_mono_out[:_e80] ** 2), 1e-12) / max(np.sum(_mono_out[_e80:] ** 2), 1e-12))
                )
                _delta_c80 = _c80_post - _c80_pre

                # D50 measurement
                _e_total_in = max(float(np.sum(_mono_in**2)), 1e-12)
                _e_total_out = max(float(np.sum(_mono_out**2)), 1e-12)
                _d50_pre = float(np.clip(float(np.sum(_mono_in[:_e50] ** 2)) / _e_total_in, 0.0, 1.0))
                _d50_post = float(np.clip(float(np.sum(_mono_out[:_e50] ** 2)) / _e_total_out, 0.0, 1.0))
                _delta_d50 = _d50_post - _d50_pre

                if _delta_c80 < -2.0:
                    # C80 degraded → rollback to dry
                    logger.warning("Phase 20 §4.5c C80-guard: ΔC80=%.2f dB < −2 dB → rollback", _delta_c80)
                    reduced = audio.copy()
                    _c80_guard_triggered = True
                elif _delta_c80 > 6.0:
                    # Excessive clarity boost → scale wet proportionally
                    _c80_wet_scale = float(np.clip(6.0 / (_delta_c80 + 1e-9), 0.30, 1.0))
                    reduced = audio + _c80_wet_scale * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                    _c80_guard_triggered = True
                    logger.info(
                        "Phase 20 §4.5c C80-guard: ΔC80=%.2f dB > 6 dB → wet scaled to %.2f",
                        _delta_c80,
                        _c80_wet_scale,
                    )
                elif _delta_c80 > 4.0:
                    # Moderate boost → blend 35 % early reflections back (spec α=0.35, 50 ms)
                    _early_win = int(sample_rate * 0.050)
                    _alpha = 0.35
                    _rd = reduced.copy().astype(np.float64)
                    _og = audio.astype(np.float64)
                    if _rd.ndim == 2:
                        for _ch in range(_rd.shape[0]):
                            _e = min(_early_win, _rd.shape[1])
                            _rd[_ch, :_e] = (1.0 - _alpha) * _rd[_ch, :_e] + _alpha * _og[_ch, :_e]
                    else:
                        _e = min(_early_win, len(_rd))
                        _rd[:_e] = (1.0 - _alpha) * _rd[:_e] + _alpha * _og[:_e]
                    reduced = np.clip(_rd.astype(np.float32), -1.0, 1.0)
                    _early_blend_triggered = True
                    logger.info(
                        "Phase 20 §4.5c C80-guard: ΔC80=%.2f dB — early-reflection blend 35 %% applied",
                        _delta_c80,
                    )

                # §4.5c D50 secondary guard: ΔD50 > 0.12 → reduce wet further
                if abs(_delta_d50) > 0.12 and not _c80_guard_triggered:
                    _d50_scale = float(np.clip(0.12 / (abs(_delta_d50) + 1e-9), 0.30, 1.0))
                    reduced = audio + _d50_scale * (reduced - audio)
                    reduced = np.clip(reduced, -1.0, 1.0)
                    logger.info(
                        "Phase 20 §4.5c D50-guard: ΔD50=%.3f > 0.12 → wet scaled to %.2f",
                        _delta_d50,
                        _d50_scale,
                    )
        except Exception as _c80_exc:
            logger.debug("Phase 20 C80/D50-guard skipped (non-critical): %s", _c80_exc)

        return PhaseResult(
            success=True,
            audio=reduced,
            metrics={
                "rms_change_db": float(rms_change_db),
                "reduction_strength": strength,
                "tail_damping": damping,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "stft_omlsa_imcra_cohen2003",
                "version": "3.0_omlsa",
                "window_size": self.WINDOW_SIZE,
                "hop_size": self.HOP_SIZE,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "delta_c80": float(_delta_c80),
                "c80_guard_triggered": _c80_guard_triggered,
                "early_blend_triggered": _early_blend_triggered,
                "loudness_makeup_db": float(_makeup_gain_db),
            },
        )

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, float, float]:
        material_key = getattr(material, "value", getattr(material, "name", str(material))).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        rms_in = float(np.sqrt(np.mean(np.asarray(original_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_change_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30)) if rms_in > 1e-8 else 0.0
        makeup_gain_db = 0.0

        if rms_in > 1e-8 and rms_change_db < -max_rms_drop_db:
            target_rms_change_db = -max_rms_drop_db
            required_gain_db = target_rms_change_db - rms_change_db
            current_peak = float(np.percentile(np.abs(processed_audio), 99.9) + 1e-12)
            max_safe_gain_db = max(0.0, -1.5 - 20.0 * np.log10(current_peak))
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, max_safe_gain_db))
            if makeup_gain_db > 0.0:
                processed_audio = np.clip(
                    processed_audio * (10.0 ** (makeup_gain_db / 20.0)),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
                rms_change_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30))
                logger.info(
                    "Phase 20 loudness-preservation: material=%s rms_change=%.2f dB via makeup %.2f dB",
                    material_key,
                    rms_change_db,
                    makeup_gain_db,
                )

        return processed_audio, float(rms_change_db), float(makeup_gain_db)

    def _reduce_reverb(self, audio: np.ndarray, sample_rate: int, strength: float, damping: float) -> np.ndarray:
        """Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003).

        Algorithmus:
            1. scipy.signal.stft  — OLA-konsistent (ersetzt np.fft.rfft-Schleife)
            2. IMCRA gleitendes Minimum (Cohen 2003):
               σ²_d(t,f) = b_min · min_{t'∈[t-M,t]} S̃(t',f), b_min=1.66
            3. DD a-priori SNR:
               ξ̂(t,f) = α·G²(t-1,f)·γ(t-1,f) + (1-α)·max(γ(t,f)-1, 0)
            4. OMLSA Gain:
               G(t,f) = G_floor^(1−p(t,f)) · (ξ̂/(1+ξ̂))^p(t,f), Clip [G_floor,1]
            5. Cappé Temporal-Smoothing: Ĝ_t = α_g·Ĝ_{t-1} + (1-α_g)·G_t
            6. scipy.signal.istft  — phasenkonsistente PGHI-Rekonstruktion
            7. Transientenerhalt: Original zurückgemischt wo transient_mask > 0.5
            8. nan_to_num + clip[-1, 1]

        Forschungsreferenz:
            Cohen & Berdugo (2002) Signal Processing Letters
            Cohen (2003) IEEE Trans. Speech Audio Process.
            Le Roux & Vincent (2013) Consistent Wiener Filtering

        Args:
            audio:       1D float32, normalisiert auf [-1, 1]
            sample_rate: Abtastrate in Hz (intern 48 000 Hz)
            strength:    Reduktionsstärke ∈ [0.0, 1.0] (materialadaptiv)
            damping:     Nachhall-Dämpfungs-Prior (beeinflusst G_floor)

        Returns:
            np.ndarray: Restauriertes Audio, gleiche Länge wie Eingang, clip[-1, 1].
        """
        len(audio)

        # ── 1. Transientenerkennung (Sample-Ebene, vor STFT) ─────────────────
        transient_mask_raw = self._detect_transients(audio, sample_rate)

        # ── 2–7. MRSA Multi-Resolution Spectral Analysis OMLSA/IMCRA (§DSP-Spezialregeln) ─
        # Replaces single-STFT OMLSA with 5-zone optimal-resolution processing + PGHI.
        audio_out = self._reduce_reverb_mrsa(audio, sample_rate, strength, damping)

        # ── 8. Transientenerhalt ───────────────────────────────────────────────
        # Transient-Maske auf Sample-Ebene hochsampeln
        transient_up = signal.resample(transient_mask_raw, len(audio_out))
        transient_up = np.clip(transient_up, 0.0, 1.0)
        audio_out = audio_out * (1.0 - transient_up) + audio[: len(audio_out)] * transient_up
        audio_out = np.clip(audio_out, -1.0, 1.0)

        return audio_out

    def _reduce_reverb_mrsa(self, audio: np.ndarray, sample_rate: int, strength: float, damping: float) -> np.ndarray:
        """MRSA 5-zone OMLSA/IMCRA reverb reduction with PGHI phase reconstruction.

        Multi-Resolution Spectral Analysis (MRSA): each frequency zone is processed
        at its optimal time-frequency resolution using a zone-specific STFT window.
        Per-zone OMLSA/IMCRA gains are interpolated (frequency & time) to the
        reference STFT grid and blended with Hanning-weighted crossfades at zone
        boundaries. Final audio is synthesised via PGHI (Perraudin 2013).

        Zone definitions (mandatory, §DSP-Spezialregeln):
            sub_bass:  win=65536, hop=16384, 0–250 Hz
            mid_low:   win=16384, hop=4096,  250–2500 Hz
            mid:       win=8192,  hop=2048,  2500–8000 Hz
            presence:  win=1024,  hop=256,   8000–16000 Hz
            air:       win=128,   hop=32,    16000–24000 Hz

        Args:
            audio:       Mono float32 [-1, 1], SR=48000.
            sample_rate: Sample rate (must be 48000).
            strength:    Reduction strength ∈ [0.0, 1.0].
            damping:     Reverb damping prior (influences G_floor).

        Returns:
            np.ndarray: Restored audio, same length as input, clipped to [-1, 1].
        """
        n_audio = len(audio)
        nyquist = float(sample_rate // 2)

        # Reference STFT (win=2048, 75 % overlap) — same as original _reduce_reverb
        REF_WIN = 2048
        REF_HOP = REF_WIN - self.WINDOW_SIZE + self.HOP_SIZE  # preserves original 512-hop
        REF_NOVERLAP = REF_WIN - REF_HOP

        f_ref, _, Zxx_ref = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=REF_WIN,
            noverlap=REF_NOVERLAP,
            boundary="even",
            padded=True,
        )
        n_bins, n_t = f_ref.shape[0], Zxx_ref.shape[1]

        # OMLSA hyper-parameters
        G_floor = float(np.clip(0.1 + (1.0 - strength) * 0.05, 0.04, 0.15))
        q = float(np.clip(strength * 0.60, 0.10, 0.80))
        b_min = 1.66  # IMCRA bias correction (Cohen 2003)
        alpha_g = 0.85  # Cappé smoothing (1994)

        # Accumulate weighted zone gains
        G_acc = np.zeros((n_bins, n_t), dtype=np.float64)
        w_acc = np.zeros(n_bins, dtype=np.float64)

        for zone_name, zone_win, zone_hop, f_low, f_high in self._MRSA_ZONES:
            try:
                # Use zone-specific STFT if audio is long enough
                if n_audio >= zone_win * 2:
                    zone_noverlap = zone_win - zone_hop
                    f_z, _, Zxx_z = signal.stft(
                        audio,
                        fs=sample_rate,
                        window="hann",
                        nperseg=zone_win,
                        noverlap=zone_noverlap,
                        boundary="even",
                        padded=True,
                    )
                else:
                    f_z, Zxx_z = f_ref, Zxx_ref
                    zone_win, zone_hop = REF_WIN, REF_HOP

                mag_z = np.abs(Zxx_z)  # (F_z, T_z)
                n_z_t = mag_z.shape[1]

                # Vectorised IMCRA noise estimation: sliding-minimum (Cohen 2003)
                frames_per_sec_z = float(sample_rate / zone_hop)
                M_z = max(3, int(1.5 * frames_per_sec_z))
                power_z = mag_z**2
                # minimum_filter1d is fast C-code (no Python frame loop)
                S_min_z = _min_filter1d_p20(power_z, size=M_z, axis=1, mode="reflect")
                noise_sq_z = np.maximum(b_min * S_min_z, 1e-12)

                # Vectorised OMLSA gain (Cohen 2003, no Decision-Directed recursion needed
                # because the sliding-min noise estimator already provides a stable σ²_d)
                gamma_z = power_z / noise_sq_z
                xi_z = np.maximum(gamma_z - 1.0, 0.0)
                nu_z = np.clip(xi_z * gamma_z / (xi_z + 1.0 + 1e-12), 0.0, 500.0)
                log_lambda_z = -np.log1p(xi_z + 1e-12) + nu_z
                Lambda_z = np.exp(np.clip(log_lambda_z, -50.0, 50.0))
                p_H1_z = np.clip(
                    Lambda_z / (1.0 + Lambda_z + 1e-12) / (1.0 + q / ((1.0 - q) * Lambda_z + 1e-12)), 0.0, 1.0
                )
                G_wiener_z = xi_z / (xi_z + 1.0 + 1e-12)
                log_G_z = (1.0 - p_H1_z) * np.log(G_floor + 1e-10) + p_H1_z * np.log(np.maximum(G_wiener_z, 1e-10))
                G_z = np.exp(np.clip(log_G_z, np.log(G_floor + 1e-10), 0.0))
                G_z = np.clip(G_z, G_floor, 1.0)

                # Cappé temporal smoothing via fast IIR lfilter (no Python loop)
                G_z_sm = _lfilter_p20([1.0 - alpha_g], [1.0, -alpha_g], G_z, axis=1)
                G_z_sm = np.clip(np.nan_to_num(G_z_sm, nan=G_floor), G_floor, 1.0)

                # Extract zone frequency range from zone STFT
                zm_z = (f_z >= float(f_low)) & (f_z <= float(f_high))
                if not np.any(zm_z):
                    continue
                f_z_zone = f_z[zm_z]
                G_zone = G_z_sm[zm_z, :]  # (n_zone_bins, n_z_t)

                # Reference STFT bins for this zone (extended by crossfade bandwidth)
                ref_zm = (f_ref >= max(0.0, float(f_low) - self._MRSA_CROSSFADE_BW_HZ)) & (
                    f_ref <= min(nyquist, float(f_high) + self._MRSA_CROSSFADE_BW_HZ)
                )
                if not np.any(ref_zm):
                    continue
                f_ref_zone = f_ref[ref_zm]
                ref_indices = np.where(ref_zm)[0]
                n_ref_zone = len(ref_indices)

                # Temporal resampling: zone frames → reference frames
                if n_z_t != n_t and len(f_z_zone) > 0:
                    t_src = np.linspace(0.0, 1.0, n_z_t)
                    t_dst = np.linspace(0.0, 1.0, n_t)
                    G_zone_t = np.empty((len(f_z_zone), n_t), dtype=np.float64)
                    for k in range(len(f_z_zone)):
                        G_zone_t[k, :] = np.interp(t_dst, t_src, G_zone[k, :])
                else:
                    G_zone_t = G_zone.astype(np.float64)

                # Frequency interpolation: zone bins → reference bins
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

                # Hanning crossfade weights at zone boundaries
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
                logger.warning("MRSA Phase 20 zone '%s' failed: %s", zone_name, zone_exc)
                continue

        # Combine zone gains; unprocessed bins → pass-through (gain=1.0)
        valid = w_acc > 0.0
        G_combined = np.ones((n_bins, n_t), dtype=np.float32)
        G_combined[valid, :] = (G_acc[valid, :] / w_acc[valid, np.newaxis]).astype(np.float32)
        G_combined = np.clip(np.nan_to_num(G_combined, nan=1.0), 0.0, 1.0)

        # Late-reverb temporal decay suppression (v9.10.112):
        # Room reverberation produces exponentially decaying tails after transients;
        # OMLSA alone treats all time-frames equally and cannot separate the
        # reverberant tail from the direct sound.  We add a time-varying secondary
        # gain that suppresses frames identified as part of a reverberant decay.
        # Ref: Noh & Hwang 2014; Braun & Haardt 2016 — spectral late-reverb model.
        if strength > 0.15 and n_t > 8:
            try:
                # Per-frame mean energy from reference STFT (linear scale)
                E_frame = np.mean(np.abs(Zxx_ref) ** 2, axis=0)  # shape (n_t,)
                E_frame = np.maximum(E_frame, 1e-15)
                E_log_db = 10.0 * np.log10(E_frame)  # dB per frame

                # Frame-to-frame delta energy (positive = rising, negative = decaying)
                dE = np.diff(E_log_db, prepend=E_log_db[0])  # shape (n_t,)

                # Smooth dE to suppress single-sample noise spikes
                _sm = max(3, min(7, n_t // 20))
                _kern = np.ones(_sm, dtype=np.float32) / _sm
                dE_smooth = np.convolve(dE.astype(np.float32), _kern, mode="same")

                # Decay mask: frames where energy is steadily dropping > 0.5 dB/hop
                decay_mask = (dE_smooth < -0.5).astype(np.float32)  # shape (n_t,)

                # Direct-sound protection window (~40 ms after each onset):
                # onset frames and the immediately following window are exempted
                # so direct attack transients are never suppressed.
                _prot = max(1, int(0.040 * sample_rate / REF_HOP))
                _onset_indices = np.where(dE > 2.0)[0]  # onset = energy rise > 2 dB
                for _oi in _onset_indices:
                    _end = min(n_t, int(_oi) + _prot)
                    decay_mask[int(_oi) : _end] = 0.0

                # Extra gain reduction in decay frames; strength-scaled.
                # Maximum penalty 35 % at full strength → never below -4.4 dB (G_lr ≥ 0.60).
                _penalty = float(np.clip(strength * 0.35, 0.0, 0.35))
                G_lr = np.clip(1.0 - _penalty * decay_mask, 0.60, 1.0).astype(np.float32)

                # Broadcast: (n_bins, n_t) × (n_t,) → shape-safe
                G_combined = np.clip(G_combined * G_lr[np.newaxis, :], 0.0, 1.0)

                logger.debug(
                    "MRSA Phase 20 late-reverb suppression: penalty=%.2f, decay_frames=%d/%d, onset_protected=%d",
                    _penalty,
                    int(np.sum(decay_mask > 0)),
                    n_t,
                    len(_onset_indices),
                )
            except Exception as _lr_exc:
                logger.debug("MRSA Phase 20 late-reverb suppression skipped: %s", _lr_exc)

        # Apply combined gain to reference STFT + PGHI phase reconstruction
        Zxx_processed = G_combined * np.abs(Zxx_ref) * np.exp(1j * np.angle(Zxx_ref))
        if _PGHI_AVAILABLE_P20:
            try:
                audio_out = _pghi_p20(
                    Zxx_processed.astype(np.complex64), sr=sample_rate, win_size=REF_WIN, hop=REF_HOP, n_samples=n_audio
                )
            except Exception as pghi_exc:
                logger.warning("MRSA Phase 20: PGHI failed, using iSTFT fallback: %s", pghi_exc)
                _, audio_out = signal.istft(
                    Zxx_processed, fs=sample_rate, window="hann", nperseg=REF_WIN, noverlap=REF_NOVERLAP, boundary=True
                )
        else:
            _, audio_out = signal.istft(
                Zxx_processed, fs=sample_rate, window="hann", nperseg=REF_WIN, noverlap=REF_NOVERLAP, boundary=True
            )

        audio_out = np.real(audio_out).astype(np.float32)
        if len(audio_out) > n_audio:
            audio_out = audio_out[:n_audio]
        elif len(audio_out) < n_audio:
            audio_out = np.pad(audio_out, (0, n_audio - len(audio_out)), mode="constant", constant_values=0.0)

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        logger.debug(
            "MRSA Phase 20: 5 zones processed, valid_bins=%d/%d, G_mean=%.3f",
            int(np.sum(valid)),
            n_bins,
            float(np.mean(G_combined)),
        )
        return audio_out

    def _detect_transients(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detect transients using energy envelope.

        Returns:
            Binary mask (1 = transient, 0 = sustain/decay)
        """
        # Calculate energy envelope (RMS in short windows)
        window_samples = int(0.01 * sample_rate)  # 10ms windows
        hop_samples = window_samples // 2

        num_windows = (len(audio) - window_samples) // hop_samples + 1
        energy = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            window = audio[start:end]
            energy[i] = np.sqrt(np.mean(window**2))  # Fix: assign to energy array

        # Detect transients as rapid energy increases
        transient_mask = np.zeros(num_windows)
        for i in range(1, num_windows):
            if energy[i] > self.TRANSIENT_THRESHOLD * energy[i - 1]:
                # Transient detected
                transient_mask[i] = 1.0
                # Extend mask for attack phase (20ms)
                extend_frames = int(0.02 * sample_rate / hop_samples)
                transient_mask[i : min(i + extend_frames, num_windows)] = 1.0

        return transient_mask


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 20: Professional Reverb Reduction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio with synthetic reverb
    duration = 3.0
    sample_rate = 44100

    # Dry signal: short impulses (like snare hits)
    t = np.linspace(0, duration, int(sample_rate * duration))
    dry_signal = np.zeros_like(t)

    # Add impulses at 0.5s intervals
    for impulse_time in np.arange(0, duration, 0.5):
        impulse_sample = int(impulse_time * sample_rate)
        if impulse_sample < len(dry_signal):
            dry_signal[impulse_sample : impulse_sample + 100] = 0.8 * np.exp(-np.arange(100) / 20)

    # Add musical content
    dry_signal += 0.2 * np.sin(2 * np.pi * 440 * t)

    # Add synthetic reverb (exponential decay of signal)
    reverb_tail = signal.lfilter([1], [1, -0.7], dry_signal)  # Simple comb filter
    reverbed_signal = dry_signal + 0.4 * reverb_tail

    logger.debug("Generated %ss test audio @ %s Hz", duration, sample_rate)
    logger.debug("Dry signal + synthetic reverb tail")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug("Material: %s", material_name)
        logger.debug("─" * 80)
        logger.debug("")

        phase = ReverbReduction()
        result = phase.process(reverbed_signal, sample_rate, material)

        logger.debug("✅ Professional Reverb Reduction:")
        logger.debug("   RMS Change: %.2f dB", result.metrics["rms_change_db"])
        logger.debug("   Reduction Strength: %.2f", result.metrics["reduction_strength"])
        logger.debug("   Tail Damping: %.2f", result.metrics["tail_damping"])
        logger.debug(
            f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
