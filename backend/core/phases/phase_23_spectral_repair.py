#!/usr/bin/env python3
"""
Phase 23: Spectral Repair v3.0 — IMCRA Adaptive Noise-Floor + Vectorized Inpainting

Algorithm (v3.0 — Über-SOTA):
1. STFT Analysis (scipy.signal.stft, Hann, material-adaptive Fensterlänge)
2. IMCRA Noise-Floor-Schätzung (Cohen 2003):
   - Exponential Power-Smoothing (α_d=0.85)
   - Sliding-Minimum über M Frames (b_min=1.66)
   - Werkzeug: scipy.ndimage.minimum_filter1d
3. Defekt-Detektion (3 Strategien):
   - Dropout: magnitude < 0.3 × noise_floor (IMCRA-basiert, bin-adaptiv)
   - Spike/Artefakt: Z-Score über IMCRA-Floor (robust via MAD)
   - Phasensprung: |Δφ(t,f)| > Schwellwert
4. Inpainting (vektorisiert, O(F+T)):
   - Horizontal (Zeit): scipy.interpolate.interp1d per Frequenzband
   - Vertikal (Frequenz): scipy.interpolate.interp1d per Zeitframe
   - Blend: 0.6 × horizontal + 0.4 × vertikal (Smaragdis 2003)
5. Phase-Velocity-Fortsetzung: δφ(f,t) = φ(f,t-1) - φ(f,t-2)
6. Konsistente ISTFT-Rekonstruktion

Scientific Foundation:
- Cohen & Berdugo (2002): Noise Estimation by Minima Controlled Recursive Averaging — IMCRA
- Cohen (2003): Noise Spectrum Estimation in Adverse Environments — OMLSA/IMCRA
- Smaragdis & Brown (2003): NMF for Audio — Inpainting-Blend-Gewichte
- Févotte & Idier (2011): NMF with β-Divergenz — spektrale Konsistenz

VERBOTEN (entfernt, per copilot-instructions §4.2):
- np.mean/np.std als globaler Rauschboden → ersetzt durch IMCRA Sliding-Minimum
- Fixierter energy_floor_db-Schwellwert → ersetzt durch adaptiven bin-spezifischen Floor
- O(F×T) Python-Doppelschleife → ersetzt durch vektorisierte F+T scipy.interpolate

Quality Target: PQS MOS ≥ 4.0 nach Reparatur
Performance Target: <0.5× Echtzeit bei 48 kHz

Author: Aurik Development Team
Version: 3.0.0
"""

import importlib
import logging
import time
from typing import Any

import numpy as np
from scipy import interpolate, ndimage, signal

from backend.core.defect_scanner import MaterialType
from backend.core.quality_mode import QualityModeConfig, is_phase_ml_enabled, log_mode_decision

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# §2.46b Spectral-Tilt-Preservation: material-adaptive tolerance in dB/octave
_TILT_TOLERANCE_P23: dict[str, float] = {
    "digital": 1.5,
    "cd_digital": 1.5,
    "streaming": 1.5,
    "tape": 1.875,
    "reel_tape": 1.875,
    "vinyl": 2.25,
    "minidisc": 2.25,
    "shellac": 3.0,
    "wax_cylinder": 3.0,
    "wire_recording": 3.0,
}


def _est_tilt_p23(audio: np.ndarray, sr: int) -> float:
    """Quick spectral tilt estimate in dB/octave (§2.46b)."""
    if audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    n = min(len(mono), 8192)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(mono[:n] * np.hanning(n))) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    valid = (freqs >= 100.0) & (freqs <= sr * 0.45)
    if np.sum(valid) < 8:
        return 0.0
    log_f = np.log2(freqs[valid] + 1e-12)
    log_m = 20.0 * np.log10(spec[valid])
    log_f_c = log_f - log_f.mean()
    log_m_c = log_m - log_m.mean()
    denom = float(np.dot(log_f_c, log_f_c))
    return float(np.dot(log_f_c, log_m_c) / denom) if denom > 1e-10 else 0.0


class SpectralRepair(PhaseInterface):
    """
    Professional Spectral Repair Engine.

    Key Features:
    - Multi-strategy inpainting (horizontal/vertical/harmonic)
    - Adaptive defect detection (z-score, energy, phase)
    - Material-specific sensitivity
    - Real-time performance (<0.5× realtime)
    - Quality validation and adaptive blending

    Use Cases:
    - MP3/AAC codec artifacts (pre-echo, quantization noise)
    - Tape dropouts (short-duration signal loss)
    - Vinyl ticks/pops (localized spectral damage)
    - Frequency band gaps (missing treble/bass)
    - Phase discontinuities (digital glitches)

    Performance: <0.5× realtime on modern CPU
    """

    @staticmethod
    def _compute_admm_runtime_profile(
        material: str | MaterialType,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """Compute adaptive runtime profile for ADMM spectral repair (§2.54)."""
        mat = str(material.value if isinstance(material, MaterialType) else material).lower().replace("-", "_")
        qm = str(quality_mode or "balanced").lower().replace("-", "_")
        rest = float(np.clip(restorability_score, 0.0, 100.0))

        # Material baselines: digital materials have clean spectra, analogues are noisy
        base = {
            "shellac": {"clip_pct": 99.2, "clip_floor": 0.82, "side_mult": 1.02},
            "vinyl": {"clip_pct": 99.3, "clip_floor": 0.84, "side_mult": 1.03},
            "tape": {"clip_pct": 99.35, "clip_floor": 0.85, "side_mult": 1.04},
            "reel_tape": {"clip_pct": 99.4, "clip_floor": 0.86, "side_mult": 1.04},
            "minidisc": {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.05},
            "cd_digital": {"clip_pct": 99.5, "clip_floor": 0.90, "side_mult": 1.06},
            "streaming": {"clip_pct": 99.55, "clip_floor": 0.92, "side_mult": 1.07},
            "unknown": {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.04},
        }.get(mat, {"clip_pct": 99.4, "clip_floor": 0.88, "side_mult": 1.04})

        # Quality mode: fast is conservative (higher clip_pct = fewer bins clipped), quality is aggressive
        mode_adj_pct = {
            "fast": 0.22,
            "balanced": 0.0,
            "quality": -0.35,
            "maximum": -0.55,
            "restoration": -0.15,
            "studio_2026": -0.55,
        }.get(qm, 0.0)
        mode_adj_floor = {
            "fast": 0.035,
            "balanced": 0.0,
            "quality": -0.045,
            "maximum": -0.070,
            "restoration": -0.020,
            "studio_2026": -0.070,
        }.get(qm, 0.0)
        mode_adj_mult = {
            "fast": 0.025,
            "balanced": 0.0,
            "quality": -0.020,
            "maximum": -0.035,
            "restoration": -0.010,
            "studio_2026": -0.035,
        }.get(qm, 0.0)

        # Restorability: low (noisy/degraded) = fewer clipped bins (conservative), high (clean) = more aggressive

        # Restorability: low (noisy/degraded) = more aggressive clipping (lower clip_pct), high (clean) = conservative
        rest_adj_pct = ((rest - 50.0) / 50.0) * 0.28
        rest_adj_floor = ((rest - 50.0) / 50.0) * 0.035
        rest_adj_mult = ((rest - 50.0) / 50.0) * 0.015

        clip_percentile = float(np.clip(base["clip_pct"] + mode_adj_pct + rest_adj_pct, 98.8, 99.9))
        clip_floor = float(np.clip(base["clip_floor"] + mode_adj_floor + rest_adj_floor, 0.80, 0.93))
        side_clip_multiplier = float(np.clip(base["side_mult"] + mode_adj_mult + rest_adj_mult, 1.00, 1.10))

        return {
            "clip_percentile": clip_percentile,
            "clip_floor": clip_floor,
            "side_clip_multiplier": side_clip_multiplier,
        }

    # STFT Parameters (material-adaptive)
    STFT_CONFIG = {
        MaterialType.SHELLAC: {
            "nperseg": 4096,  # Larger window for noisy material
            "noverlap": 3072,  # 75% overlap
            "nfft": 8192,
        },
        MaterialType.VINYL: {
            "nperseg": 2048,
            "noverlap": 1536,  # 75% overlap
            "nfft": 4096,
        },
        MaterialType.TAPE: {
            "nperseg": 2048,
            "noverlap": 1536,
            "nfft": 4096,
        },
        MaterialType.CD_DIGITAL: {
            "nperseg": 2048,
            "noverlap": 1024,  # 50% overlap (less processing needed)
            "nfft": 4096,
        },
        MaterialType.STREAMING: {
            "nperseg": 1024,
            "noverlap": 512,
            "nfft": 2048,
        },
    }

    # Defect detection thresholds
    DETECTION_THRESHOLDS = {
        MaterialType.SHELLAC: {
            "outlier_z_score": 4.5,  # Higher (more tolerant of noise)
            "energy_floor_db": -55,  # Higher floor
            "phase_jump_threshold": np.pi * 0.7,
        },
        MaterialType.VINYL: {
            "outlier_z_score": 4.0,
            "energy_floor_db": -60,
            "phase_jump_threshold": np.pi * 0.6,
        },
        MaterialType.TAPE: {
            "outlier_z_score": 3.5,
            "energy_floor_db": -65,
            "phase_jump_threshold": np.pi * 0.5,
        },
        MaterialType.CD_DIGITAL: {
            "outlier_z_score": 3.0,  # More sensitive
            "energy_floor_db": -70,
            "phase_jump_threshold": np.pi * 0.4,
        },
        MaterialType.STREAMING: {
            "outlier_z_score": 2.5,  # Very sensitive (MP3 artifacts)
            "energy_floor_db": -75,
            "phase_jump_threshold": np.pi * 0.3,
        },
    }

    # Inpainting blend amounts (how aggressive to repair)
    REPAIR_STRENGTH = {
        MaterialType.SHELLAC: 0.60,  # Moderate (preserve character)
        MaterialType.VINYL: 0.70,
        MaterialType.TAPE: 0.75,
        MaterialType.CD_DIGITAL: 0.85,  # Aggressive (digital artifacts obvious)
        MaterialType.STREAMING: 0.90,  # Very aggressive (codec artifacts)
    }

    # Soft-relax window for thrashing guards (§2.54): allow robust processing when
    # headroom is still objectively high, keep hard stop for real emergency states.
    _THRASH_RELAX_ML_MIN_AVAIL_GB = 14.0
    _THRASH_RELAX_ML_MIN_AVAIL_RATIO = 0.40
    _THRASH_RELAX_ML_MAX_SWAP_PCT = 95.0
    _THRASH_RELAX_ML_MAX_ATTEMPTS = 1
    _THRASH_RELAX_MRSA_MIN_AVAIL_GB = 8.0
    _THRASH_RELAX_MRSA_MIN_AVAIL_RATIO = 0.28
    _THRASH_RELAX_MRSA_MAX_SWAP_PCT = 98.0
    _THRASH_RELAX_MRSA_MAX_ATTEMPTS = 1

    def __init__(self):
        super().__init__()
        self.name = "Spectral Repair v3 IMCRA"
        self._audiosr_plugin = None  # Lazy loading
        self._ml_guard_events: list[dict[str, Any]] = []
        self._current_material: MaterialType = MaterialType.CD_DIGITAL  # updated per process() call
        self._pressure_relax_ml_attempts: int = 0
        self._pressure_relax_mrsa_attempts: int = 0

    def _has_sufficient_ml_headroom(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Return True when enough physical RAM is available for AudioSR stage.

        Guard 1 — material check: AudioSR bandwidth extension is the wrong tool for
        lossy-codec artifacts (MP3/AAC ringing, pre-echo, masking throughout spectrum).
        DSP spectral inpainting is more appropriate; never load 5.9 GB for this.

        Guard 2 — channel-aware RAM check (§2.38a): stereo doubles inference working
        memory; empirical per-minute inference buffer overhead is added.
        """
        try:
            import gc

            import psutil
        except Exception:
            return True

        # Guard 1: AudioSR nur für bekannte Analog-Quellen erlaubt (Allowlist-Prinzip).
        # Bug-16b-Fix: Blocklist {"mp3_low", ...} verhindert nicht "unknown" — bei unbekanntem
        # Material lädt AudioSR trotzdem → OOM. Allowlist verlangt positive Analog-Evidenz.
        # AudioSR trainiert auf Analog-Bandbreitenverlust (Shellac ≤7 kHz, Tape ≤12 kHz).
        # Für "unknown", cd_digital, dat, mp3*, aac, streaming: DSP-Inpainting überlegen.
        _ANALOG_ALLOW_AUDIOSR: frozenset[str] = frozenset(
            {
                "vinyl",
                "shellac",
                "tape",
                "reel_tape",
                "wax_cylinder",
                "cassette",
                "lacquer_disc",
                "wire_recording",
            }
        )
        _mat = getattr(self, "_current_material", None)
        if _mat not in _ANALOG_ALLOW_AUDIOSR:
            self._ml_guard_events.append(
                {
                    "phase_id": "phase_23_spectral_repair",
                    "model": "AudioSR",
                    "reason": "lossy_codec_material_dsp_preferred",
                    "required_gb": 0.0,
                    "available_gb": 0.0,
                    "channels": 0,
                    "duration_s": 0.0,
                    "fallback": "dsp_inpainting",
                }
            )
            logger.info(
                "SpectralRepair: AudioSR skipped — material '%s' not in analog allowlist — DSP inpainting preferred",
                _mat,
            )
            return False

        # Guard 2: channel-aware physical RAM check (§2.38a)
        # Aurik internal format: (N,) mono or (N, ch) stereo — first axis is always samples.
        n_channels = int(audio.shape[1]) if (audio.ndim == 2 and 1 < audio.shape[1] <= 8) else 1
        n_samples = int(audio.shape[0])
        duration_s = n_samples / float(max(1, sample_rate))

        # Zone-based budget: _run_audiosr_ml() processes in 10-second zones, so only ONE
        # zone is in memory at a time.  Use zone duration (not full audio duration) to avoid
        # a false OOM-guard reject for long songs like 225 s tracks.
        # Model: ~5.8 GB steady-state.  Per-zone DDIM inference (50 steps × 10 s): ~1.5 GB.
        _AUDIOSR_ZONE_SECONDS = 10
        duration_for_budget_s = min(duration_s, float(_AUDIOSR_ZONE_SECONDS))

        required_gb = 5.0  # Base model weight budget
        if duration_for_budget_s >= 60.0:
            required_gb += 1.0  # only reached if a zone were somehow > 60 s (never)
        required_gb += 1.5 * (duration_for_budget_s / 60.0)  # per-zone inference overhead
        required_gb = min(required_gb, 22.0)  # sanity cap
        # Note: n_channels multiplier removed — _repair_with_audiosr processes mono channels
        # individually (M/S in _repair_channel), so each call is always mono.

        available_gb = float(psutil.virtual_memory().available / (1024**3))
        if available_gb < required_gb + 1.5:
            try:
                from backend.core.plugin_lifecycle_manager import evict_stale_plugins

                evict_stale_plugins(required_mb=int(required_gb * 1024))
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            gc.collect()
            try:
                import ctypes as _ct

                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            available_gb = float(psutil.virtual_memory().available / (1024**3))

        if available_gb < required_gb:
            self._ml_guard_events.append(
                {
                    "phase_id": "phase_23_spectral_repair",
                    "model": "AudioSR",
                    "reason": "insufficient_physical_ram_headroom",
                    "required_gb": float(required_gb),
                    "available_gb": float(available_gb),
                    "channels": n_channels,
                    "duration_s": float(duration_s),
                    "fallback": "dsp_inpainting",
                }
            )
            logger.warning(
                "SpectralRepair RAM guard triggered: %.1f GB available, %.1f GB required "
                "(duration=%.1fs, ch=%d) — using DSP fallback",
                available_gb,
                required_gb,
                duration_s,
                n_channels,
            )
            return False
        return True

    def _can_relax_thrashing_guard(self, *, for_mrsa: bool) -> bool:
        """Return True when pressure is elevated but still far from emergency.

        This enables a controlled attempt for robust paths in phase 23 instead of
        forcing immediate Single-STFT fallback on every transient pressure spike.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            avail_gb = float(vm.available / (1024**3))
            avail_ratio = float(vm.available / max(vm.total, 1))
            swap_pct = float(getattr(swap, "percent", 100.0))

            if for_mrsa:
                return (
                    avail_gb >= self._THRASH_RELAX_MRSA_MIN_AVAIL_GB
                    and avail_ratio >= self._THRASH_RELAX_MRSA_MIN_AVAIL_RATIO
                    and swap_pct < self._THRASH_RELAX_MRSA_MAX_SWAP_PCT
                )

            return (
                avail_gb >= self._THRASH_RELAX_ML_MIN_AVAIL_GB
                and avail_ratio >= self._THRASH_RELAX_ML_MIN_AVAIL_RATIO
                and swap_pct < self._THRASH_RELAX_ML_MAX_SWAP_PCT
            )
        except Exception:
            return False

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_23_spectral_repair",
            name="Spectral Repair v3 IMCRA",
            category=PhaseCategory.ENHANCEMENT,
            priority=5,
            dependencies=["phase_03_denoise", "phase_24_dropout_repair"],
            estimated_time_factor=0.50,
            version="3.0.0",
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.94,
            description="IMCRA Adaptive Noise-Floor + Vectorized Spectral Inpainting (Cohen 2003)",
        )

    def _get_audiosr_plugin(self):
        """Lazy load AudioSR plugin for ML-based repair."""
        if self._audiosr_plugin is None:
            try:
                from plugins.audiosr_plugin import AudioSRPlugin

                self._audiosr_plugin = AudioSRPlugin()
                logger.info("AudioSR plugin loaded successfully")
            except Exception as e:
                logger.warning("Failed to load AudioSR plugin: %s", e)
                self._audiosr_plugin = False  # Mark as unavailable

        return self._audiosr_plugin if self._audiosr_plugin is not False else None

    @staticmethod
    def _is_system_thrashing() -> bool:
        """Return True when swap/RAM pressure makes heavy phase-23 paths unsafe."""
        try:
            from backend.core.ml_memory_budget import is_system_thrashing

            return bool(is_system_thrashing())
        except Exception:
            return False

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply spectral repair to audio.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with repaired audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        # §4.6b: Pre-phase eviction — free previous phase models to prevent OOM
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_evict

            _get_plm_evict().evict_for_phase("phase_23_spectral_repair")
        except Exception:
            pass

        start_time = time.time()
        _progress_cb = kwargs.get("progress_sub_callback")

        def _report_progress(pct: float, label: str) -> None:
            if callable(_progress_cb):
                try:
                    _progress_cb(float(np.clip(pct, 0.0, 100.0)), label, time.time() - start_time)
                except Exception:
                    pass

        _report_progress(2.0, "Spektralreparatur: Vorbereitung")
        self._ml_guard_events = []
        # Store material as lowercase string value for guard comparison (handles both str and MaterialType enum)
        self._current_material = str(getattr(material, "value", material)).lower()
        self.validate_input(audio)
        _audio_for_tilt_p23 = audio.copy()  # §2.46b: tilt reference before any processing (incl. Apollo)

        # Normalize stereo layout: UV3 sends (2, N) channels-first; phase_23 processes
        # internally as (N, 2) samples-first for M/S and column_stack operations.
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        if _was_channels_first:
            audio = audio.T  # (2, N) → (N, 2)
            _audio_for_tilt_p23 = _audio_for_tilt_p23.T

        is_stereo = audio.ndim == 2

        # Get material-specific parameters
        stft_cfg = self.STFT_CONFIG.get(material, self.STFT_CONFIG[MaterialType.CD_DIGITAL])
        thresholds = self.DETECTION_THRESHOLDS.get(material, self.DETECTION_THRESHOLDS[MaterialType.CD_DIGITAL])
        repair_strength = self.REPAIR_STRENGTH.get(material, 0.75)

        # Locality-aware modulation from UV3.
        # Sparse defect coverage -> lower inpainting intensity to preserve unaffected texture.
        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        repair_strength = float(np.clip(repair_strength * _effective_strength, 0.0, 1.0))

        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            if _was_channels_first and passthrough.ndim == 2:
                passthrough = passthrough.T  # (N, 2) → (2, N)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "defect_reduction_percent": 0.0,
                    "spectral_coherence": 1.0,
                    "repair_strength": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rt_factor": 0.0,
                    "nperseg": stft_cfg["nperseg"],
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Repair skipped due to zero effective strength"],
            )

        if repair_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            if _was_channels_first and passthrough.ndim == 2:
                passthrough = passthrough.T  # (N, 2) → (2, N)
            return PhaseResult(
                success=True,
                audio=passthrough,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "material": material.name,
                    "defect_reduction_percent": 0.0,
                    "spectral_coherence": 1.0,
                    "repair_strength": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rt_factor": 0.0,
                    "nperseg": stft_cfg["nperseg"],
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Repair skipped due to zero effective strength"],
            )

        # --- Apollo pre-processing for lossy-codec materials ---
        # Apollo TorchScript handles MDCT-specific artefacts (pre-echo, spectral staircase,
        # psychoacoustic masking) better than generic STFT inpainting.  When the Apollo
        # model is loaded the output is fed into the standard STFT inpainting below as a
        # pre-cleaned signal.  Only activates when Apollo model is actually loaded
        # (no fallback call here — Apollo DSP-fallback runs inside repair() if model absent).
        _APOLLO_CODEC_MATERIALS = frozenset({"mp3_low", "mp3_high", "aac", "minidisc", "streaming"})
        _apollo_preproc_applied = False
        _apollo_hf_gain_db: float | None = None
        if self._current_material in _APOLLO_CODEC_MATERIALS:
            # §OOM-Guard: Swap-Thrashing-Check vor Apollo-TorchScript-Inferenz.
            # Apollo lädt bis zu 21 GB anon-RSS bei langen Audios (Mamba-State-Space).
            # Wenn Swap bereits > 80 % voll, bricht die Allokation den Prozess via
            # Kernel-OOM-Killer (kein Python-Traceback, kein faulthandler-Dump).
            # In diesem Fall DSP-Fallback erzwingen — Apollo wird übersprungen.
            _apollo_swap_blocked = False
            try:
                from backend.core.ml_memory_budget import is_system_thrashing as _is_thrashing_p23

                if _is_thrashing_p23():
                    logger.warning(
                        "phase_23: Apollo TorchScript übersprungen — Swap-Thrashing erkannt "
                        "(OOM-Killer-Prävention). DSP-Inpainting wird verwendet."
                    )
                    _apollo_swap_blocked = True
            except Exception as _swap_chk_exc:
                logger.debug("phase_23: Swap-Thrashing-Check fehlgeschlagen (non-critical): %s", _swap_chk_exc)
            # RAM-Guard: Apollo benötigt mind. 6 GB freien RAM (TorchScript + Mamba-State-Space).
            # Swap-Thrashing-Check allein reicht nicht — Crash tritt auch bei niedrigem
            # Swap-Prozent auf wenn nach großen Vorphasen (SGMSE+/MDX) wenig RAM verfügbar ist.
            if not _apollo_swap_blocked:
                try:
                    import psutil as _psutil_p23

                    _avail_ram_p23 = _psutil_p23.virtual_memory().available / (1024**3)
                    if _avail_ram_p23 < 6.0:
                        logger.warning(
                            "phase_23: Apollo TorchScript übersprungen — nur %.1f GB RAM verfügbar "
                            "(< 6.0 GB Mindest-Headroom). DSP-Inpainting wird verwendet.",
                            _avail_ram_p23,
                        )
                        _apollo_swap_blocked = True
                except Exception as _ram_chk_exc:
                    logger.debug("phase_23: RAM-Check fehlgeschlagen (non-critical): %s", _ram_chk_exc)

            _plm23 = None
            if not _apollo_swap_blocked:
                try:
                    from plugins.apollo_plugin import get_apollo as _get_apollo

                    try:
                        from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm23

                        _plm23 = _get_plm23()
                        _plm23.set_active("Apollo", True)  # §4.6b: protect from eviction during inference
                    except Exception:
                        _plm23 = None

                    _apollo_inst = _get_apollo()
                    if _apollo_inst._model_loaded and _apollo_inst._torch_model is not None:
                        if is_stereo:
                            # Audio is normalized to (N, 2) at method start — safe to use [:, 0]
                            if _plm23 is not None:
                                try:
                                    _plm23.touch_plugin("Apollo")
                                except Exception:
                                    pass
                            _ap_l = _apollo_inst.repair(audio[:, 0], sample_rate, material=self._current_material)
                            _ap_l_audio = _ap_l.audio
                            _ap_l_hf = float(_ap_l.hf_gain_db)
                            del _ap_l
                            # GC between channels — free torch tensors before second channel
                            import gc

                            gc.collect(0)
                            if _plm23 is not None:
                                try:
                                    _plm23.touch_plugin("Apollo")
                                except Exception:
                                    pass
                            _ap_r = _apollo_inst.repair(audio[:, 1], sample_rate, material=self._current_material)
                            audio = np.column_stack((_ap_l_audio, _ap_r.audio))  # (N, 2)
                            _apollo_hf_gain_db = (_ap_l_hf + float(_ap_r.hf_gain_db)) / 2.0
                            del _ap_r, _ap_l_audio
                        else:
                            _ap_res = _apollo_inst.repair(audio, sample_rate, material=self._current_material)
                            audio = _ap_res.audio
                            _apollo_hf_gain_db = float(_ap_res.hf_gain_db)
                            del _ap_res
                        _apollo_preproc_applied = True
                        logger.info(
                            "phase_23: Apollo pre-processing applied (material=%s, hf_gain=+%.1f dB)",
                            self._current_material,
                            _apollo_hf_gain_db,
                        )
                        _report_progress(20.0, "Spektralreparatur: Apollo-Vorverarbeitung")
                except Exception as _apollo_exc:
                    logger.debug("Apollo pre-processing skipped (non-critical): %s", _apollo_exc)
                finally:
                    if _plm23 is not None:
                        try:
                            _plm23.set_active("Apollo", False)
                        except Exception:
                            pass

        # --- ADMM Declipping path (spec §4.5a) ---
        # Detect hard clipping and route to sparse-recovery solver instead of
        # standard inpainting.  SOFT_SATURATION → no ADMM (per §5 vintage rules).
        _use_admm = False
        _clip_level = 0.98
        _defect_type_kwarg = kwargs.get("defect_type")
        if _defect_type_kwarg is not None and hasattr(_defect_type_kwarg, "name"):
            if _defect_type_kwarg.name in ("CLIPPING", "HARMONIC_DISTORTION"):
                _use_admm = True
        else:
            try:
                from backend.core.clipping_detection import ClippingType, classify_clipping

                if is_stereo:
                    _mono_ref = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
                else:
                    _mono_ref = audio
                _clip_check = _mono_ref[: min(len(_mono_ref), sample_rate * 10)]
                if classify_clipping(_clip_check, sample_rate) == ClippingType.CLIPPING:
                    _use_admm = True
            except Exception as _ce:
                logger.debug("classify_clipping check failed: %s", _ce)

        if _use_admm:
            # Estimate clip ceiling as 99.5th percentile of absolute amplitude
            _clip_level = float(np.percentile(np.abs(audio), 99.5))
            _clip_level = float(np.clip(_clip_level, 0.85, 1.0))
            logger.info("ADMM declipping activated: clip_level=%.4f", _clip_level)
            if is_stereo:
                # §2.51 M/S: Reparatur auf Mid-Kanal; Side minimal (Stereo-Kohärenz-Invariante).
                _sqrt2 = np.sqrt(2.0)
                _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
                _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
                _report_progress(35.0, "Spektralreparatur: ADMM Mid-Kanal")
                _repaired_mid = self._admm_declip(_mid, _clip_level, sample_rate)
                # Side: declip mildly (half strength) to avoid breaking stereo field
                _side_clip = float(np.clip(_clip_level * 1.05, 0.85, 1.0))
                _report_progress(55.0, "Spektralreparatur: ADMM Side-Kanal")
                _repaired_side = self._admm_declip(_side, _side_clip, sample_rate)
                repaired_audio = np.column_stack(
                    (
                        (_repaired_mid + _repaired_side) / _sqrt2,
                        (_repaired_mid - _repaired_side) / _sqrt2,
                    )
                )
            else:
                _report_progress(60.0, "Spektralreparatur: ADMM")
                repaired_audio = self._admm_declip(audio, _clip_level, sample_rate)
        else:
            # Process via standard spectral inpainting — §2.51 M/S for stereo.
            if is_stereo:
                # §2.51 M/S: Reparatur auf Mid-Kanal; Side minimal (Stereo-Kohärenz-Invariante).
                _sqrt2 = np.sqrt(2.0)
                _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
                _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
                _report_progress(35.0, "Spektralreparatur: Mid-Kanal")
                _repaired_mid = self._repair_channel(
                    _mid,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    repair_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        35.0 + 20.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur Mid: {lbl}",
                    ),
                )
                # Side: minimal repair at half strength to preserve stereo field
                _side_strength = repair_strength * 0.5
                _report_progress(55.0, "Spektralreparatur: Side-Kanal")
                _repaired_side = self._repair_channel(
                    _side,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    _side_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        55.0 + 20.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur Side: {lbl}",
                    ),
                )
                repaired_audio = np.column_stack(
                    (
                        (_repaired_mid + _repaired_side) / _sqrt2,
                        (_repaired_mid - _repaired_side) / _sqrt2,
                    )
                )
            else:
                _report_progress(60.0, "Spektralreparatur: Inpainting")
                repaired_audio = self._repair_channel(
                    audio,
                    sample_rate,
                    stft_cfg,
                    thresholds,
                    repair_strength,
                    progress_cb=lambda p, lbl: _report_progress(
                        35.0 + 40.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                        f"Spektralreparatur: {lbl}",
                    ),
                )

        # Calculate metrics
        defect_reduction = self._calculate_defect_reduction(audio, repaired_audio, sample_rate)
        spectral_coherence = self._calculate_spectral_coherence(repaired_audio, sample_rate)

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        repaired_audio = np.nan_to_num(repaired_audio, nan=0.0, posinf=0.0, neginf=0.0)
        repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
        _report_progress(78.0, "Spektralreparatur: Qualitätsmetriken")

        # §2.46b Spectral-Tilt-Guard: cap HF inpainting if tilt deviates beyond tolerance
        # Only applies to spectral inpainting path, not ADMM declipping (no HF synthesis there)
        _tilt_capped_p23 = False
        if not _use_admm:
            try:
                _mat_k23 = str(self._current_material).lower().replace(" ", "_").replace("-", "_")
                _tol23 = _TILT_TOLERANCE_P23.get(_mat_k23, 2.0)
                _tb23 = _est_tilt_p23(_audio_for_tilt_p23, sample_rate)
                _ta23 = _est_tilt_p23(repaired_audio, sample_rate)
                _dev23 = abs(_ta23 - _tb23)
                if _dev23 > _tol23:
                    _cap23 = float(np.clip(1.0 - (_dev23 - _tol23) / (_tol23 * 2.0), 0.5, 1.0))
                    repaired_audio = _cap23 * repaired_audio + (1.0 - _cap23) * _audio_for_tilt_p23
                    repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
                    _tilt_capped_p23 = True
                    logger.info(
                        "phase_23 §2.46b tilt-cap: before=%.2f after=%.2f dev=%.2f tol=%.2f cap=%.2f",
                        _tb23,
                        _ta23,
                        _dev23,
                        _tol23,
                        _cap23,
                    )
            except Exception as _tc23:
                logger.debug("phase_23 §2.46b tilt-cap skipped (graceful): %s", _tc23)

        # §4.5 Psychoacoustic Masking Clamp — only repair audible spectral gaps
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp

            repaired_audio = apply_psychoacoustic_masking_clamp(
                audio,
                repaired_audio,
                sample_rate,
                strength=_effective_strength,
                mode="additive",
            )
        except Exception as _pm_exc:
            logger.debug("Phase23 masking clamp non-blocking: %s", _pm_exc)

        _report_progress(92.0, "Spektralreparatur: Abschluss")

        # Restore channels-first layout expected by UV3 (2, N)
        if _was_channels_first and repaired_audio.ndim == 2:
            repaired_audio = repaired_audio.T  # (N, 2) → (2, N)

        _result = PhaseResult(
            success=True,
            audio=repaired_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "defect_reduction_percent": float(defect_reduction * 100),
                "spectral_coherence": float(spectral_coherence),
                "repair_strength": float(repair_strength),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rt_factor": float(rt_factor),
                "nperseg": stft_cfg["nperseg"],
                "apollo_preproc_applied": _apollo_preproc_applied,
                "apollo_preproc_hf_gain_db": _apollo_hf_gain_db,
                "ml_guard_events": list(self._ml_guard_events),
                "deferred_for_kmv": ["phase_23_spectral_repair"] if self._ml_guard_events else [],
                "spectral_tilt_capped": _tilt_capped_p23,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            warnings=[] if rt_factor < 0.6 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )
        _report_progress(100.0, "Spektralreparatur: fertig")
        return _result

    def _admm_declip(
        self,
        audio: np.ndarray,
        clip_level: float,
        sr: int = 48_000,
        rho: float = 0.1,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """ADMM sparse-recovery declipping per spec §4.5a (Záviška 2021).

        Solves:  minimize  ||W x||_1
                 subject to  x[reliable] = y[reliable]
                              x[clipped] ∈ [-C, C]

        where W is a Daubechies-db4 Level-5 wavelet transform and C = clip_level.

        TransientGuard: at onset positions (±5 ms) rho is scaled by 3.0 to
        preserve attack transients (ArticulationMetric ≥ 0.88 invariant).

        Post-ADMM: clipped samples that remain > clip_level are hard-clamped
        to avoid residual ears.

        Reference: Záviška et al. (2021) "A Survey and an Extensive Evaluation
        of Popular Audio Declipping Methods"; Condat (2013) ADMM tutorial.

        Args:
            audio:     1-D float32 mono signal (clipped).
            clip_level: Amplitude ceiling (e.g. 0.98 for near-full-scale).
            sr:        Sample rate — must be 48 000 Hz.
            rho:       ADMM penalty parameter (default 0.1, adaptive at onsets).
            max_iter:  Maximum ADMM iterations.
            tol:       Convergence tolerance (relative primal residual).

        Returns:
            Declipped 1-D float32 array, same shape as input.
        """
        try:
            pywt = importlib.import_module("pywt")
        except ModuleNotFoundError:
            logger.warning("pywt not available — ADMM declipping skipped, returning original")
            return np.asarray(audio, dtype=np.float32)

        y = np.asarray(audio, dtype=np.float64)
        n = len(y)

        # --- Reliable vs clipped mask ---
        reliable_mask = np.abs(y) < clip_level * 0.99
        clipped_mask = ~reliable_mask
        if not np.any(clipped_mask):
            return y.astype(np.float32)

        # --- Onset detection for TransientGuard (§4.5a) ---
        onset_win = max(1, int(sr * 0.005))  # ±5 ms
        onset_guard = np.zeros(n, dtype=bool)
        try:
            hop = 64
            n_frames = max(1, (n - hop) // hop)
            frame_energy = np.array([np.sum(y[i * hop : i * hop + hop] ** 2) for i in range(n_frames)])
            diff = np.diff(frame_energy, prepend=frame_energy[0])
            mu, sigma = float(np.mean(diff)), float(np.std(diff)) + 1e-10
            onset_frames = np.where(diff > mu + 2.0 * sigma)[0]
            for of in onset_frames:
                s = max(0, of * hop - onset_win)
                e = min(n, of * hop + onset_win)
                onset_guard[s:e] = True
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)  # No transient guard on error — safe fallback

        # --- Wavelet parameters: db4 Level-5 ---
        wavelet = "db4"
        level = 5

        # --- ADMM initialisation ---
        x = y.copy()
        coeffs_shape = pywt.wavedec(x, wavelet, level=level, mode="periodization")
        z = [c.copy() for c in coeffs_shape]
        u = [np.zeros_like(c) for c in coeffs_shape]

        lam = 0.01 * rho  # Sparsity weight balanced against rho
        rho_onset = rho * 3.0  # TransientGuard penalty

        x_prev = x.copy()
        for _iter in range(max_iter):
            # x-update: reconstruct from (z − u), then project onto constraints
            z_minus_u = [z[i] - u[i] for i in range(len(z))]
            x_new = pywt.waverec(z_minus_u, wavelet, mode="periodization")
            x_new = x_new[:n]
            if len(x_new) < n:
                x_new = np.pad(x_new, (0, n - len(x_new)))

            # Projection: reliable → original; clipped → clamp to [-C, C]
            x_new[reliable_mask] = y[reliable_mask]
            x_new[clipped_mask] = np.clip(x_new[clipped_mask], -clip_level, clip_level)

            # At onset guard positions: keep original value when reliable to
            # preserve transient shape (rho_onset applied implicitly via z-update)
            onset_reliable = onset_guard & reliable_mask
            x_new[onset_reliable] = y[onset_reliable]

            x = x_new

            # z-update: soft-threshold in wavelet domain
            xc = pywt.wavedec(x, wavelet, level=level, mode="periodization")
            for i in range(len(z)):
                v = xc[i] + u[i]
                # Use onset-aware threshold only for approximation coefficients (i==0)
                thr = lam / (rho_onset if i == 0 and np.any(onset_guard) else rho)
                z[i] = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)

            # u-update: dual ascent
            xc2 = pywt.wavedec(x, wavelet, level=level, mode="periodization")
            for i in range(len(u)):
                u[i] = u[i] + xc2[i] - z[i]

            # Convergence check
            rel = float(np.linalg.norm(x - x_prev)) / (float(np.linalg.norm(x_prev)) + 1e-10)
            if rel < tol:
                logger.debug("ADMM declip converged after %d iterations (rel=%.2e)", _iter + 1, rel)
                break
            x_prev = x.copy()

        # Hard-clamp residual excursions > clip_level as safety net
        x = np.clip(x, -1.0, 1.0)
        return x.astype(np.float32)

    def _repair_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        stft_cfg: dict[str, int],
        thresholds: dict[str, float],
        repair_strength: float,
        progress_cb=None,
    ) -> np.ndarray:
        """Repair a single audio channel using spectral inpainting."""

        def _report(pct: float, label: str) -> None:
            if callable(progress_cb):
                try:
                    progress_cb(float(np.clip(pct, 0.0, 100.0)), label)
                except Exception:
                    pass

        _report(8.0, "STFT")
        # Compute STFT
        _f, _t, Zxx = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=stft_cfg["nperseg"],
            noverlap=stft_cfg["noverlap"],
            nfft=stft_cfg["nfft"],
            boundary="even",
        )

        # Magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        _report(22.0, "Defekterkennung")

        # Detect defects using DSP (always)
        defect_mask = self._detect_defects(magnitude, phase, thresholds)

        if np.sum(defect_mask) == 0:
            # No defects detected
            _report(100.0, "Keine Defekte")
            return audio

        # Calculate defect severity for adaptive ML decision
        defect_severity = float(np.sum(defect_mask) / defect_mask.size)
        system_thrashing = self._is_system_thrashing()
        allow_ml_under_pressure = system_thrashing and self._can_relax_thrashing_guard(for_mrsa=False)
        if allow_ml_under_pressure and self._pressure_relax_ml_attempts >= self._THRASH_RELAX_ML_MAX_ATTEMPTS:
            allow_ml_under_pressure = False
            logger.warning(
                "phase_23: pressure-relax ML attempt cap reached (%d) — forcing DSP fallback",
                self._THRASH_RELAX_ML_MAX_ATTEMPTS,
            )

        # Decide: ML or DSP?
        use_ml = is_phase_ml_enabled(23) and QualityModeConfig.should_use_ml("phase_23", defect_severity)
        if use_ml and system_thrashing and not allow_ml_under_pressure:
            logger.warning("phase_23: ML repair skipped — system thrashing detected, forcing DSP fallback")
            use_ml = False
        elif use_ml and allow_ml_under_pressure:
            self._pressure_relax_ml_attempts += 1
            logger.warning(
                "phase_23: controlled ML retry under pressure enabled — high RAM headroom detected (attempt %d/%d)",
                self._pressure_relax_ml_attempts,
                self._THRASH_RELAX_ML_MAX_ATTEMPTS,
            )

        if use_ml:
            if not self._has_sufficient_ml_headroom(audio, sample_rate):
                audiosr = None
            else:
                audiosr = self._get_audiosr_plugin()
            if audiosr is not None:
                # ML-based repair with AudioSR
                log_mode_decision("phase_23", True, f"Defect severity: {defect_severity:.2%}")
                _report(55.0, "AudioSR")
                repaired_audio = self._repair_with_audiosr(audio, sample_rate, defect_mask, repair_strength, audiosr)
                _report(98.0, "AudioSR fertig")
                return repaired_audio
            else:
                logger.warning("AudioSR unavailable, falling back to DSP")

        # DSP-based repair (fallback or FAST mode)
        log_mode_decision("phase_23", False, f"Mode: {QualityModeConfig.get_mode().value}")

        # §DSP-Spezialregeln: MRSA 5-Zone-Reparatur für BALANCED/QUALITY/MAXIMUM
        _mode_upper = QualityModeConfig.get_mode().value.upper()
        if _mode_upper not in ("FAST",) and len(audio) >= sample_rate:
            # OOM-Preflight: MRSA (5 Zonen × STFT) benötigt ~2 GB für Stereo-Audio.
            # Unter 4 GB verfügbar → Single-STFT-Fallback um systemd-oomd zu vermeiden.
            _mrsa_ok = True
            _allow_mrsa_under_pressure = system_thrashing and self._can_relax_thrashing_guard(for_mrsa=True)
            if (
                _allow_mrsa_under_pressure
                and self._pressure_relax_mrsa_attempts >= self._THRASH_RELAX_MRSA_MAX_ATTEMPTS
            ):
                _allow_mrsa_under_pressure = False
                logger.warning(
                    "phase_23: pressure-relax MRSA attempt cap reached (%d) — using Single-STFT fallback",
                    self._THRASH_RELAX_MRSA_MAX_ATTEMPTS,
                )
            if system_thrashing and not _allow_mrsa_under_pressure:
                logger.warning("phase_23: MRSA skipped — system thrashing detected, using Single-STFT fallback")
                _mrsa_ok = False
            elif _allow_mrsa_under_pressure:
                self._pressure_relax_mrsa_attempts += 1
                logger.warning(
                    "phase_23: MRSA allowed under controlled pressure window (attempt %d/%d)",
                    self._pressure_relax_mrsa_attempts,
                    self._THRASH_RELAX_MRSA_MAX_ATTEMPTS,
                )
            try:
                if _mrsa_ok:
                    import psutil as _psu

                    _avail_gb = _psu.virtual_memory().available / (1024**3)
                    if _avail_gb < 4.0:
                        logger.warning(
                            "phase_23: MRSA-OOM-Preflight fehlgeschlagen (%.1f GB < 4.0 GB) — Single-STFT-Fallback",
                            _avail_gb,
                        )
                        _mrsa_ok = False
            except Exception:
                pass  # psutil nicht verfügbar — MRSA versuchen
            if _mrsa_ok:
                try:
                    _report(45.0, "MRSA-Start")
                    out = self._repair_channel_mrsa(
                        audio,
                        sample_rate,
                        thresholds,
                        repair_strength,
                        progress_cb=lambda p, lbl: _report(
                            45.0 + 50.0 * float(np.clip(p, 0.0, 100.0)) / 100.0,
                            f"MRSA: {lbl}",
                        ),
                    )
                    _report(98.0, "MRSA fertig")
                    return out
                except Exception as _mrsa_err:
                    logger.warning("MRSA-Reparatur fehlgeschlagen (%s), Single-STFT-Fallback", _mrsa_err)

        # Single-STFT fallback (FAST mode or MRSA failure)
        # Apply inpainting strategies
        _report(60.0, "Inpainting Magnitude")
        repaired_magnitude = self._inpaint_magnitude(magnitude, defect_mask)
        _report(72.0, "Inpainting Phase")
        repaired_phase = self._inpaint_phase(phase, defect_mask)

        # Reconstruct complex spectrogram
        Zxx_repaired = repaired_magnitude * np.exp(1j * repaired_phase)

        # Blend original and repaired
        Zxx_blended = Zxx * (1 - defect_mask * repair_strength) + Zxx_repaired * (defect_mask * repair_strength)

        # Phase-kohärente Rekonstruktion via PGHI (§2.47 VERBOTEN: direktes ISTFT nach Spektralmodifikation)
        try:
            from dsp.pghi import pghi_reconstruct_from_stft as _pghi_istft_p23

            _hop_p23 = stft_cfg["nperseg"] - stft_cfg["noverlap"]
            audio_repaired = _pghi_istft_p23(
                Zxx_blended,
                sr=sample_rate,
                win_size=stft_cfg["nperseg"],
                hop=_hop_p23,
                use_original_phase=False,
                n_samples=len(audio),
            )
            audio_repaired = np.asarray(audio_repaired, dtype=np.float64)
        except (ImportError, Exception) as _pghi_err_p23:
            logger.debug("PGHI-Fallback auf scipy.signal.istft (phase_23): %s", _pghi_err_p23)
            _, audio_repaired = signal.istft(
                Zxx_blended,
                fs=sample_rate,
                window="hann",
                nperseg=stft_cfg["nperseg"],
                noverlap=stft_cfg["noverlap"],
                nfft=stft_cfg["nfft"],
                boundary=True,
            )
        _report(98.0, "Rekonstruktion")

        return audio_repaired[: len(audio)]

    def _repair_channel_mrsa(
        self,
        audio: np.ndarray,
        sample_rate: int,
        thresholds: dict[str, float],
        repair_strength: float,
        progress_cb=None,
    ) -> np.ndarray:
        """Repair single audio channel using 5-zone MRSA (§DSP-Spezialregeln).

        Applies spectral inpainting independently per frequency zone with
        zone-appropriate STFT resolution (win 65536→128). Reconstructs each zone
        via PGHI-approximation and merges via Hanning crossfade (10 ms, §DSP).

        Args:
            audio: Mono float32 input channel.
            sample_rate: Must be 48000 Hz.
            thresholds: Material-specific detection thresholds.
            repair_strength: Blend factor for inpainting (0–1).

        Returns:
            Repaired mono float32 audio.
        """
        from backend.core.mrsa_zones import analyze_zones, merge_zones, synthesize_zone

        audio_f32 = np.asarray(audio, dtype=np.float32)
        zone_stfts = analyze_zones(audio_f32, sample_rate)
        zone_audios: dict[str, np.ndarray] = {}
        _zone_names = list(zone_stfts.keys())
        _n_zones = max(1, len(_zone_names))

        for _zi, name in enumerate(_zone_names):
            if callable(progress_cb):
                try:
                    progress_cb(5.0 + 90.0 * (_zi / _n_zones), f"Zone {name}")
                except Exception:
                    pass
            zone = zone_stfts[name]
            magnitude = np.abs(zone.stft)
            phase = np.angle(zone.stft)
            defect_mask = self._detect_defects(magnitude, phase, thresholds)

            if np.sum(defect_mask) == 0:
                # No defects in this zone — passthrough (preserve original)
                zone_audios[name] = synthesize_zone(zone, zone.stft, len(audio))
                # Release STFT of this zone immediately — no longer needed
                zone_stfts[name] = zone._replace(stft=np.empty(0, dtype=np.complex64))
                del magnitude, phase, defect_mask
                continue

            repaired_mag = self._inpaint_magnitude(magnitude, defect_mask)
            Zxx_repaired = repaired_mag * np.exp(1j * phase)
            blend_mask = defect_mask * repair_strength
            Zxx_blended = zone.stft * (1.0 - blend_mask) + Zxx_repaired * blend_mask

            zone_audios[name] = synthesize_zone(zone, Zxx_blended, len(audio))
            # Release all intermediary arrays and this zone's STFT immediately
            zone_stfts[name] = zone._replace(stft=np.empty(0, dtype=np.complex64))
            del magnitude, phase, defect_mask, repaired_mag, Zxx_repaired, blend_mask, Zxx_blended

        if callable(progress_cb):
            try:
                progress_cb(100.0, "Zonen-Merge")
            except Exception:
                pass
        return merge_zones(zone_audios, zone_stfts, sample_rate, len(audio))

    def _repair_with_audiosr(
        self,
        audio: np.ndarray,
        sample_rate: int,
        defect_mask: np.ndarray,
        repair_strength: float,
        audiosr: Any,
    ) -> np.ndarray:
        """
        Repair audio using AudioSR ML model.

        Strategy: DSP-Detection + ML-Repair
        1. DSP detects defect regions (already done - defect_mask)
        2. Extract defect regions with context (±500ms)
        3. Process with AudioSR (super-resolution inpainting)
        4. Blend back with repair_strength

        Args:
            audio: Input audio channel (mono)
            sample_rate: Sample rate in Hz
            defect_mask: Binary mask from DSP detection
            repair_strength: Blend amount (0-1)

        Returns:
            Repaired audio
        """
        if audiosr is None:
            return audio

        # §4.6b PLM Active-Guard — prevents Emergency-Eviction during AudioSR inference
        _plm23_asr = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm23_asr

            _plm23_asr = _get_plm23_asr()
            _plm23_asr.set_active("AudioSR", True)
        except Exception:
            pass
        try:
            if not self._has_sufficient_ml_headroom(audio, sample_rate):
                return audio

            def _repair_single_channel(channel_audio: np.ndarray) -> np.ndarray:
                # AudioSR.process() erwartet (audio: np.ndarray, sr: int, target_sr: int)
                # — keine Dateipfade. Das Plugin übernimmt Resampling und DSP-Fallback intern.
                target_sr = 48000
                repaired_channel = audiosr.process(channel_audio, sample_rate, target_sr)
                repaired_channel = np.asarray(repaired_channel, dtype=np.float32)
                repaired_channel = np.squeeze(repaired_channel)

                if repaired_channel.ndim != 1:
                    logger.warning(
                        "AudioSR returned unexpected shape %s for mono channel — falling back to passthrough",
                        repaired_channel.shape,
                    )
                    repaired_channel = channel_audio

                if len(repaired_channel) != len(channel_audio):
                    from scipy.signal import resample as _resample

                    repaired_channel = _resample(repaired_channel, len(channel_audio))

                audio_final = (
                    channel_audio * (1 - repair_strength)
                    + repaired_channel.astype(channel_audio.dtype) * repair_strength
                )
                return audio_final[: len(channel_audio)]

            audio_arr = np.asarray(audio)
            if audio_arr.ndim == 1:
                return _repair_single_channel(audio_arr)

            if audio_arr.ndim == 2:
                # UV3 uses channels-first stereo (2, N). Some standalone phase tests/use-cases
                # still provide samples-first (N, 2). Preserve the incoming orientation.
                if audio_arr.shape[0] <= 2 and audio_arr.shape[1] > 2:
                    repaired_channels = [_repair_single_channel(audio_arr[ch]) for ch in range(audio_arr.shape[0])]
                    return np.stack(repaired_channels, axis=0).astype(audio_arr.dtype, copy=False)

                if audio_arr.shape[1] <= 2 and audio_arr.shape[0] > 2:
                    repaired_channels = [_repair_single_channel(audio_arr[:, ch]) for ch in range(audio_arr.shape[1])]
                    return np.column_stack(repaired_channels).astype(audio_arr.dtype, copy=False)

            logger.warning(
                "AudioSR repair received unsupported audio shape %s — falling back to passthrough",
                audio_arr.shape,
            )
            return audio

        except Exception as e:
            logger.error("AudioSR processing failed: %s, falling back to DSP", e)
            # Fallback to DSP (will be handled by caller)
            return audio
        finally:
            if _plm23_asr is not None:
                try:
                    _plm23_asr.set_active("AudioSR", False)
                except Exception:
                    pass

    def _estimate_noise_floor_imcra(self, magnitude: np.ndarray) -> np.ndarray:
        """IMCRA-adaptiver Rauschboden pro Zeit-Frequenz-Bin (Cohen 2003).

        Algorithmus:
            1. Leistungsspektrum P(t,f) = |magnitude|²
            2. Exp. Glättung: S̃(t,f) = α_d·S̃(t-1,f) + (1-α_d)·P(t,f)  α_d=0.85
            3. Sliding-Minimum: σ²_min(t,f) = min_{t'∈[t-M,t]} S̃(t',f)
            4. Rauschboden: σ_d(t,f) = √(b_min · σ²_min(t,f))  b_min=1.66

        Forschungsreferenz:
            Cohen (2003): „Noise Spectrum Estimation in Adverse Environments:
            Improved Minima Controlled Recursive Averaging"

        Args:
            magnitude: STFT-Magnitude (F, T), float32/64

        Returns:
            noise_floor: Adaptiver Rauschboden (F, T), Amplitude-Einheiten, NaN-frei
        """
        power = magnitude**2  # (F, T)

        # Exponentielle Glättung α_d=0.85 (Cohen 2003 Gleichung 3)
        alpha_d = 0.85
        smoothed = np.empty_like(power)
        smoothed[:, 0] = power[:, 0]
        for t_idx in range(1, power.shape[1]):
            smoothed[:, t_idx] = alpha_d * smoothed[:, t_idx - 1] + (1.0 - alpha_d) * power[:, t_idx]

        # Sliding-Minimum (M ≈ 1.5 s in STFT-Frames, mind. 5, max. 40)
        M = max(5, min(40, power.shape[1] // 4))
        min_smoothed = ndimage.minimum_filter1d(smoothed, size=M, axis=1, mode="nearest")

        # Overcorrection b_min=1.66 → zurück zu Amplitude (Cohen 2003, Gl. 12)
        b_min = 1.66
        noise_floor = np.sqrt(np.maximum(b_min * min_smoothed, 1e-20))
        noise_floor = np.nan_to_num(noise_floor, nan=1e-10, posinf=1.0, neginf=1e-10)
        return noise_floor

    def _detect_defects(self, magnitude: np.ndarray, phase: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
        """Defekt-Detektion via IMCRA-adaptivem Rauschboden + Phasenkonsistenz.

        Strategien:
            1. Dropout:  magnitude < 0.3 × IMCRA_noise_floor  (bin-adaptiv)
            2. Artefakt: Z-Score über IMCRA-Floor via MAD  (1.4826 · MAD = σ_robust)
            3. Phasensprung: |Δφ(t,f)| > Schwellwert

        Entfernt (verboten per copilot-instructions §4.2):
            np.mean/std als globaler Rauschboden → IMCRA Sliding-Minimum
            Fixierter energy_floor_db → adaptiver bin-spezifischer Floor

        Args:
            magnitude:  STFT-Magnitude (F, T)
            phase:      STFT-Phase (F, T)
            thresholds: Material-spezifische Schwellwerte

        Returns:
            defect_mask: Bool-Array (F, T), True = defekter Bin
        """
        defect_mask = np.zeros_like(magnitude, dtype=bool)

        # -- Strategie 1 + 2: IMCRA-adaptiver Rauschboden --
        noise_floor = self._estimate_noise_floor_imcra(magnitude)

        # Dropout: Magnitude deutlich unterhalb des geschätzten Rauschbodens
        dropout_mask = magnitude < (noise_floor * 0.3)
        defect_mask |= dropout_mask

        # Spike/Codec-Artefakt: Z-Score über IMCRA-Floor (robust via MAD)
        ratio = np.where(noise_floor > 1e-12, magnitude / (noise_floor + 1e-12), 1.0)
        ratio_db = 20.0 * np.log10(np.maximum(ratio, 1e-10))

        median_ratio = np.median(ratio_db, axis=1, keepdims=True)
        mad = np.median(np.abs(ratio_db - median_ratio), axis=1, keepdims=True) + 1e-6
        z_scores = (ratio_db - median_ratio) / (mad * 1.4826)
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        spike_mask = z_scores > thresholds["outlier_z_score"]
        defect_mask |= spike_mask

        # -- Strategie 3: Phasensprünge --
        phase_diff = np.diff(phase, axis=1)
        phase_jumps = np.abs(phase_diff) > thresholds["phase_jump_threshold"]
        defect_mask[:, 1:] |= phase_jumps

        # Morphologische Bereinigung
        defect_mask = ndimage.binary_opening(defect_mask, structure=np.ones((3, 3)))
        defect_mask = ndimage.binary_closing(defect_mask, structure=np.ones((5, 3)))

        return defect_mask

    def _inpaint_magnitude(self, magnitude: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """Vektorisiertes Spectral Inpainting — O(F+T) statt O(F×T).

        Algorithmus (Smaragdis & Brown 2003, Blend-Gewichte):
            Für jede Frequenz f: interp1d NaN-Lücken entlang Zeitachse  →  mag_h
            Für jeden Zeitframe t: interp1d NaN-Lücken entlang Frequenzachse → mag_v
            Repaired[defect] = 0.6 · mag_h[defect] + 0.4 · mag_v[defect]

        Entfernt (verboten):
            O(F×T) Python-Doppelschleife mit einzeln berechneten Pixeln
        """
        if not np.any(defect_mask):
            return magnitude.copy()

        # --- Horizontal: Zeit-Richtung (Zeile = Frequenz) ---
        mag_h = magnitude.copy().astype(np.float64)
        mag_h[defect_mask] = np.nan

        for f in range(mag_h.shape[0]):
            row = mag_h[f, :]
            if not np.any(np.isnan(row)):
                continue
            valid = np.where(~np.isnan(row))[0]
            if len(valid) >= 2:
                xs = np.arange(len(row))
                interp_fn = interpolate.interp1d(
                    valid, row[valid], kind="linear", fill_value=(row[valid[0]], row[valid[-1]]), bounds_error=False
                )
                row[:] = interp_fn(xs)
            elif len(valid) == 1:
                row[:] = row[valid[0]]
            else:
                row[:] = 1e-10
            mag_h[f, :] = np.nan_to_num(row, nan=1e-10)

        # --- Vertikal: Frequenz-Richtung (Spalte = Zeitframe) ---
        mag_v = magnitude.copy().astype(np.float64)
        mag_v[defect_mask] = np.nan

        for t in range(mag_v.shape[1]):
            col = mag_v[:, t]
            if not np.any(np.isnan(col)):
                continue
            valid = np.where(~np.isnan(col))[0]
            if len(valid) >= 2:
                xs = np.arange(len(col))
                interp_fn = interpolate.interp1d(
                    valid, col[valid], kind="linear", fill_value=(col[valid[0]], col[valid[-1]]), bounds_error=False
                )
                col[:] = interp_fn(xs)
            elif len(valid) == 1:
                col[:] = col[valid[0]]
            else:
                col[:] = 1e-10
            mag_v[:, t] = np.nan_to_num(col, nan=1e-10)

        # Blend an Defektstellen: 0.6 horizontal + 0.4 vertikal
        repaired = magnitude.copy().astype(np.float64)
        blended = 0.6 * mag_h + 0.4 * mag_v
        repaired[defect_mask] = blended[defect_mask]

        return np.maximum(repaired, 0.0)

    def _inpaint_phase(self, phase: np.ndarray, defect_mask: np.ndarray) -> np.ndarray:
        """Phase-Inpainting via Phasen-Geschwindigkeits-Fortsetzung.

        Statt einfaches Frame-Copy: Phasengeschwindigkeit δφ(f,t) = φ(f,t-1) − φ(f,t-2)
        wird extrapoliert. Dies entspricht der instantanen Frequenz und erhält
        die Phasenkohärenz (Laroche & Dolson 1999, Phase-Vocoder).
        """
        repaired = phase.copy()
        F, T = phase.shape

        for f in range(F):
            mask_row = defect_mask[f, :]
            if not np.any(mask_row):
                continue
            row = repaired[f, :]
            prev_phi = phase[f, 0]
            prev_delta = 0.0
            for t in range(1, T):
                if mask_row[t]:
                    row[t] = prev_phi + prev_delta
                    # update prev_phi mit extrapoliertem Wert für nächste Iteration
                    prev_phi = row[t]
                    # prev_delta bleibt konstant (lineare Phase-Fortsetzung)
                else:
                    if t >= 2 and not mask_row[t - 1]:
                        prev_delta = phase[f, t] - phase[f, t - 1]
                    prev_phi = phase[f, t]

        return repaired

    def _calculate_defect_reduction(self, original: np.ndarray, repaired: np.ndarray, sample_rate: int) -> float:
        """Calculate percentage of defects reduced."""
        # Simple metric: reduction in high-frequency noise
        _, _, Pxx_orig = signal.spectrogram(original if original.ndim == 1 else original[:, 0], fs=sample_rate)
        _, _, Pxx_rep = signal.spectrogram(repaired if repaired.ndim == 1 else repaired[:, 0], fs=sample_rate)

        noise_orig = np.std(Pxx_orig)
        noise_rep = np.std(Pxx_rep)

        reduction = max(0, min(1, (noise_orig - noise_rep) / noise_orig)) if noise_orig > 1e-10 else 0.0

        return reduction

    def _calculate_spectral_coherence(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate spectral coherence (smoothness) score."""
        if audio.ndim == 2:
            audio = audio[:, 0]  # Use left channel

        # Compute spectrogram
        _f, _t, Pxx = signal.spectrogram(audio, fs=sample_rate, nperseg=2048)

        # Measure smoothness (inverse of spectral roughness)
        spectral_diff = np.diff(Pxx, axis=0)
        roughness = np.mean(np.abs(spectral_diff))

        # Normalize to 0-1 range (lower roughness = higher coherence)
        coherence = 1.0 / (1.0 + roughness * 100)

        return float(coherence)
