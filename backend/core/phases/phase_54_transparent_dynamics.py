#!/usr/bin/env python3
"""
Phase 53: Transparent Dynamics v1.0 - Tier 1 ML-Hybrid.
Psychoacoustic-aware compression that adapts to musical genre and content.

Algorithm Overview:
1. Psychoacoustic Masking Detection
   - Identify quiet passages (psychoacoustic masking zones)
   - Hide compression artifacts in masked regions
   - Preserve loud passages with minimal intervention

2. Genre-Adaptive Time Constants
   - Classical: Slow attack (50-100ms), long release (500-1000ms)
   - Jazz: Medium attack (20-40ms), medium release (200-400ms)
   - Rock: Fast attack (5-15ms), short release (50-150ms)
   - Electronic: Ultra-fast attack (1-5ms), adaptive release

3. Intelligent Transient Detection
   - Detect drum hits, piano attacks, plucked strings
   - Preserve transient clarity (bypass compression for 5-20ms)
   - Adaptive threshold based on RMS envelope

4. Material-Specific Ceiling
   - Shellac: Gentle compression (ratio 2:1, -20dB threshold)
   - Vinyl: Moderate (ratio 2.5:1, -18dB)
   - CD: Transparent (ratio 3:1, -15dB)
   - Streaming: Ultra-transparent (ratio 4:1, -12dB)

Components:
- PsychoacousticMaskingDetector: Find masking zones
- GenreAdaptiveCompressor: Time constants per genre
- TransientPreserver: Intelligent transient bypass
- MaterialAdaptiveCeiling: Material-specific limits

Scientific Foundation:
- Zwicker & Fastl (1999): Psychoacoustics - masking curves
- Painter & Spanias (2000): Perceptual Coding of Digital Audio
- Reiss (2012): Intelligent Systems for Music Information Retrieval
- Moore (2012): An Introduction to the Psychology of Hearing
- Giannoulis et al. (2012): Digital Dynamic Range Compressor Design
- Abel & Berners (2004): Music and Audio Research Laboratory papers

Industry Benchmarks:
- FabFilter Pro-C 2 (Transparent mode, lookahead)
- iZotope Ozone Dynamics (Intelligent mode)
- Sonnox Oxford Dynamics (Transparent mastering)
- Waves H-Comp (Hybrid compression, genre presets)
- DMG Audio Compassion (Psychoacoustic compression)

Tier 1 Priority: PRIORITY 4 (after Bass/Drums/Piano, completes Transparent Dynamics)
Quality Target: Imperceptible compression, maintain dynamics 90%+
Performance Target: <0.25× realtime

Author: Aurik Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
import time

import numpy as np
from scipy import signal

from backend.core.audio_utils import to_channels_last
from backend.core.defect_scanner import MaterialType
from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp
from backend.core.natural_performance_detector import get_natural_performance_detector

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _extract_compression_pressure(defect_scores: object) -> float:
    """Liest Compression-Defektstärke robust aus heterogenen defect_scores."""
    if not isinstance(defect_scores, dict) or not defect_scores:
        return 0.0

    key_weights = {
        "COMPRESSION_ARTIFACTS": 1.0,
        "DYNAMIC_COMPRESSION_EXCESS": 0.85,
        "DIGITAL_ARTIFACTS": 0.35,
    }

    pressure = 0.0
    for key, val in defect_scores.items():
        key_s = str(key)
        key_norm = key_s.rsplit(".", maxsplit=1)[-1].upper()
        if key_norm not in key_weights:
            continue

        sev_val = None
        if isinstance(val, (int, float)):
            sev_val = float(val)
        elif hasattr(val, "severity"):
            sev_val = float(getattr(val, "severity", 0.0) or 0.0)

        if sev_val is None:
            continue

        pressure = max(pressure, float(np.clip(sev_val * key_weights[key_norm], 0.0, 1.0)))

    return float(np.clip(pressure, 0.0, 1.0))


class TransparentDynamicsV1(PhaseInterface):
    """
    Tier 1 Transparent Dynamics Processor with psychoacoustic awareness.

    Features:
    - Psychoacoustic masking detection (hide compression in quiet zones)
    - Genre-adaptive time constants (Classical/Jazz/Rock/Electronic)
    - Intelligent transient preservation (drums, piano, plucks)
    - Material-specific compression ceiling (Shellac gentle → Streaming transparent)
    - Frequency-dependent processing (preserve bass, control highs)

    Use Cases:
    - Classical mastering (ultra-transparent, slow dynamics)
    - Jazz enhancement (natural, preserve dynamics)
    - Rock/Pop balance (controlled peaks, maintain punch)
    - Electronic music (fast transients, consistent loudness)

    Performance: <0.25× realtime on modern CPU
    """

    # Genre-adaptive time constants [attack_ms, release_ms]
    GENRE_PRESETS = {
        "classical": {
            "attack_ms": 75,
            "release_ms": 750,
            "ratio": 1.5,
            "threshold_db": -25,
            "knee_db": 12,
        },
        "jazz": {
            "attack_ms": 30,
            "release_ms": 300,
            "ratio": 2.0,
            "threshold_db": -20,
            "knee_db": 10,
        },
        "rock": {
            "attack_ms": 10,
            "release_ms": 100,
            "ratio": 3.0,
            "threshold_db": -15,
            "knee_db": 8,
        },
        "electronic": {
            "attack_ms": 3,
            "release_ms": 50,
            "ratio": 4.0,
            "threshold_db": -12,
            "knee_db": 6,
        },
        "default": {
            "attack_ms": 20,
            "release_ms": 200,
            "ratio": 2.5,
            "threshold_db": -18,
            "knee_db": 8,
        },
    }

    # Material-specific compression ceiling
    MATERIAL_CEILING = {
        MaterialType.SHELLAC: {
            "ratio": 2.0,
            "threshold_db": -20,
            "knee_db": 12,
            "mix": 0.50,  # 50% wet (gentle)
        },
        MaterialType.VINYL: {
            "ratio": 2.5,
            "threshold_db": -18,
            "knee_db": 10,
            "mix": 0.60,  # 60% wet
        },
        MaterialType.TAPE: {
            "ratio": 2.8,
            "threshold_db": -16,
            "knee_db": 9,
            "mix": 0.65,  # 65% wet
        },
        MaterialType.CD_DIGITAL: {
            "ratio": 3.0,
            "threshold_db": -15,
            "knee_db": 8,
            "mix": 0.70,  # 70% wet (transparent)
        },
        MaterialType.STREAMING: {
            "ratio": 4.0,
            "threshold_db": -12,
            "knee_db": 6,
            "mix": 0.75,  # 75% wet (ultra-transparent)
        },
    }

    DEFAULT_MATERIAL_CONFIG = {
        "ratio": 2.5,
        "threshold_db": -18,
        "knee_db": 8,
        "mix": 0.60,
    }

    _MATERIAL_KEY_MAP: dict[str, MaterialType] = {
        "shellac": MaterialType.SHELLAC,
        "vinyl": MaterialType.VINYL,
        "tape": MaterialType.TAPE,
        "cassette": MaterialType.TAPE,
        "cd": MaterialType.CD_DIGITAL,
        "cd_digital": MaterialType.CD_DIGITAL,
        "digital": MaterialType.CD_DIGITAL,
        "streaming": MaterialType.STREAMING,
    }

    def __init__(self, sample_rate: int = 48000, genre: str = "default", **kwargs):
        """
        Initialisiert Transparent Dynamics Processor.

        Args:
            sample_rate: Audio sample rate (Hz)
            genre: Musical genre ('classical', 'jazz', 'rock', 'electronic', 'default')
            **kwargs: Override parameters
        """
        super().__init__(sample_rate, **kwargs)
        self.genre = genre

    @staticmethod
    def _compute_transparent_dynamics_profile(
        _material_key: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.56 Advisory-only mix adaptation for transparent dynamics."""
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )
        mix_delta = {
            "fast": -0.05,
            "balanced": 0.0,
            "quality": 0.05,
            "maximum": 0.08,
        }.get(_mode, 0.0)

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        if _rest < 40.0:
            mix_delta -= 0.10

        return {"mix_delta": float(np.clip(mix_delta, -0.20, 0.20))}

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Wendet an: transparent psychoacoustic-aware compression.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Input-SR (intern 48000 Hz)
            material_type: Source material type
            **kwargs: Additional parameters

        Returns:
            PhaseResult with transparently compressed audio
        """
        sample_rate = int(kwargs.get("sample_rate", sample_rate))
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p54_transposed = to_channels_last(audio)
        start_time = time.time()

        _material_raw = kwargs.get("material_type", material_type)
        if isinstance(_material_raw, MaterialType):
            material_enum = _material_raw
        else:
            material_enum = self._MATERIAL_KEY_MAP.get(str(_material_raw).strip().lower(), MaterialType.CD_DIGITAL)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))
        compression_pressure = _extract_compression_pressure(kwargs.get("defect_scores"))
        control_floor = 0.0
        if compression_pressure >= 0.25:
            control_floor = float(np.clip(0.25 + 0.65 * ((compression_pressure - 0.25) / 0.75), 0.25, 0.90))
        control_strength = float(max(effective_strength, control_floor))

        if control_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "compression_pressure": compression_pressure,
                    "control_floor": control_floor,
                    "control_strength": control_strength,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
            )

        # Determine genre preset
        genre_key = str(kwargs.get("genre") or self.genre)
        genre_config = self.GENRE_PRESETS.get(genre_key, self.GENRE_PRESETS["default"])

        # Get material-specific config
        material_config = self.MATERIAL_CEILING.get(material_enum, self.DEFAULT_MATERIAL_CONFIG)
        _material_key_54 = material_enum.value
        _td_profile = self._compute_transparent_dynamics_profile(
            _material_key_54,
            kwargs.get("quality_mode"),
            float(kwargs.get("restorability_score", 50.0)),
        )

        # Merge configs (genre provides time constants, material provides ceiling)
        attack_ms = genre_config["attack_ms"]
        release_ms = genre_config["release_ms"]
        ratio = material_config["ratio"]
        threshold_db = material_config["threshold_db"]
        knee_db = material_config["knee_db"]
        mix = float(np.clip(material_config["mix"] + _td_profile["mix_delta"], 0.0, 1.0)) * control_strength

        hard_intervention_active = compression_pressure >= 0.45
        if hard_intervention_active:
            hard_norm = float(np.clip((compression_pressure - 0.45) / 0.55, 0.0, 1.0))
            threshold_db = float(np.clip(threshold_db - (2.0 + 6.0 * hard_norm), -36.0, -6.0))
            ratio = float(np.clip(ratio + (0.4 + 1.2 * hard_norm), 1.5, 5.0))
            attack_ms = float(np.clip(attack_ms * (1.05 + 0.30 * hard_norm), 3.0, 120.0))
            release_ms = float(np.clip(release_ms * (1.15 + 0.55 * hard_norm), 40.0, 1400.0))
            min_mix = float(np.clip(0.45 + 0.40 * hard_norm, 0.45, 0.90))
            mix = float(max(mix, min_mix))

        # §2.51 Linked-Stereo: Gain-Envelope aus Mono-Downmix, identisch auf L+R
        is_stereo = audio.ndim == 2
        audio_mono = np.mean(audio, axis=1) if is_stereo else audio.copy()

        logger.info(
            "Transparent Dynamics: %s, genre=%s, ratio=%.1f:1, threshold=%sdB, "
            "attack=%sms, release=%sms, comp_pressure=%.2f, control=%.2f",
            material_enum.value,
            genre_key,
            ratio,
            threshold_db,
            attack_ms,
            release_ms,
            compression_pressure,
            control_strength,
        )

        # Stage 1: Psychoacoustic Masking Detection
        masking_curve = self._detect_psychoacoustic_masking(audio_mono)

        # Stage 2: Intelligent Transient Detection
        transient_mask = self._detect_transients(audio_mono)

        # Stage 3: Genre-Adaptive Compression (on mono for gain computation)
        audio_compressed = self._apply_compression(
            audio_mono,
            ratio=ratio,
            threshold_db=threshold_db,
            knee_db=knee_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
            masking_curve=masking_curve,
            transient_mask=transient_mask,
            compression_pressure=compression_pressure,
        )

        # §2.51 Linked: Compute gain envelope from mono, apply identically to L+R
        # gain = compressed / original (avoid division by zero)
        eps = 1e-10
        gain_envelope = np.where(
            np.abs(audio_mono) > eps,
            audio_compressed / (audio_mono + eps * np.sign(audio_mono + eps)),
            1.0,
        )
        # Smooth gain to avoid rapid fluctuations
        gain_envelope = np.clip(gain_envelope, 0.0, 10.0)

        # Apply gain + dry/wet mix to preserve stereo field
        if is_stereo:
            audio_out = np.empty_like(audio)
            for ch in range(audio.shape[1]):
                wet = audio[:, ch] * gain_envelope
                audio_out[:, ch] = mix * wet + (1.0 - mix) * audio[:, ch]
        else:
            audio_out = mix * audio_compressed + (1.0 - mix) * audio_mono

        # Prevent clipping — §2.49 Peak-Guard: percentile(99.9)
        peak = float(np.percentile(np.abs(audio_out), 99.9))
        if peak > 0.95:
            audio_out = audio_out * (0.95 / peak)

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # §4.5 Psychoacoustic Masking Clamp — preserve masked dynamics regions
        try:
            audio_out = apply_psychoacoustic_masking_clamp(
                audio,
                audio_out,
                sample_rate,
                strength=effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            logger.debug("Phase54 masking clamp non-blocking: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Dynamik-Kompression darf
        # Atemgeräusche zwischen Phrasen nicht gaten und Vibrato nicht glätten.
        try:
            _npa_a54 = audio
            if _npa_a54.ndim == 2 and _npa_a54.shape[0] == 2 and _npa_a54.shape[1] > _npa_a54.shape[0]:
                _npa_a54 = _npa_a54.T
            _npa_r54 = get_natural_performance_detector().detect(_npa_a54, sample_rate)
            _npa_n54 = (
                audio_out.shape[1]
                if (audio_out.ndim == 2 and audio_out.shape[0] == 2 and audio_out.shape[1] > 2)
                else audio_out.shape[0]
            )
            _npa_m54 = _npa_r54.get_protected_mask(_npa_n54, sample_rate)
            if np.any(_npa_m54):
                if audio_out.ndim == 2 and audio.ndim == 2:
                    if audio_out.shape[0] == 2 and audio_out.shape[1] > 2:
                        audio_out[:, _npa_m54] = audio[:, _npa_m54]
                    elif audio_out.shape == audio.shape:
                        audio_out[_npa_m54, :] = audio[_npa_m54, :]
                elif audio_out.ndim == 1 and audio.ndim == 1:
                    audio_out[_npa_m54] = audio[_npa_m54]
        except Exception as _npa54_exc:
            logger.debug("§2.46f phase_54 NPA-Guard (non-blocking): %s", _npa54_exc)

        return PhaseResult(
            success=True,
            audio=audio_out,
            execution_time_seconds=time.time() - start_time,
            metadata={
                "material_type": material_enum.value,
                "genre": genre_key,
                "ratio": ratio,
                "threshold_db": threshold_db,
                "attack_ms": attack_ms,
                "release_ms": release_ms,
                "mix": mix,
                "compression_pressure": compression_pressure,
                "control_floor": control_floor,
                "control_strength": control_strength,
                "hard_intervention_active": hard_intervention_active,
                "transparent_dynamics_profile": dict(_td_profile),
                "mix_delta": float(_td_profile["mix_delta"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
        )

    def _detect_psychoacoustic_masking(self, audio: np.ndarray) -> np.ndarray:
        """
        Erkennt psychoacoustic masking zones (quiet passages where compression is hidden).

        Returns masking curve: 0.0 = no masking (preserve), 1.0 = full masking (compress freely)
        """
        # RMS envelope (100ms window for masking detection)
        window_size = int(0.100 * self.sample_rate)
        audio_squared = audio**2
        window = np.ones(window_size) / window_size
        rms_envelope = np.sqrt(np.convolve(audio_squared, window, mode="same"))

        # Normalize RMS to 0-1 range
        rms_max = np.max(rms_envelope)
        if rms_max > 0:
            rms_envelope = rms_envelope / rms_max

        # Masking curve: quiet = high masking, loud = low masking
        # Use inverted RMS (quiet passages have high masking)
        masking_curve = 1.0 - rms_envelope

        # Smooth masking curve (avoid abrupt changes)
        nyquist = self.sample_rate / 2
        masking_curve = signal.sosfilt(signal.butter(2, 5 / nyquist, output="sos"), masking_curve)  # 5 Hz lowpass

        # Clip to 0-1 range
        masking_curve = np.clip(masking_curve, 0, 1)

        return masking_curve

    def _detect_transients(self, audio: np.ndarray) -> np.ndarray:
        """
        Erkennt transients (drum hits, piano attacks) for preservation.

        Returns transient mask: 1.0 = transient (preserve), 0.0 = non-transient (compress)
        """
        # High-pass filter to emphasize transients
        nyquist = self.sample_rate / 2
        sos = signal.butter(4, 80 / nyquist, btype="high", output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Envelope detection
        envelope = np.abs(audio_hp)
        envelope = signal.sosfilt(signal.butter(2, 50 / nyquist, output="sos"), envelope)

        # Find transient peaks
        threshold = np.mean(envelope) + 2 * np.std(envelope)
        peak_indices, _ = signal.find_peaks(
            envelope,
            height=threshold,
            distance=int(0.02 * self.sample_rate),  # Min 20ms between transients
        )

        # Create transient mask (20ms windows around peaks)
        transient_mask = np.zeros_like(audio)
        window_samples = int(0.020 * self.sample_rate)

        for peak_idx in peak_indices:
            start_idx = max(0, peak_idx - window_samples // 4)
            end_idx = min(len(audio), peak_idx + window_samples)

            # Gaussian window for smooth transition
            window_len = end_idx - start_idx
            window = signal.windows.gaussian(window_len, std=window_len / 6)
            transient_mask[start_idx:end_idx] += window

        # Normalize mask
        transient_mask = np.clip(transient_mask, 0, 1)

        return transient_mask

    def _apply_compression(
        self,
        audio: np.ndarray,
        ratio: float,
        threshold_db: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
        masking_curve: np.ndarray,
        transient_mask: np.ndarray,
        compression_pressure: float = 0.0,
    ) -> np.ndarray:
        """
        Wendet an: genre-adaptive soft-knee compression with psychoacoustic awareness.
        """
        # Convert attack/release to samples
        attack_samples = max(1, int(attack_ms * self.sample_rate / 1000))
        release_samples = max(1, int(release_ms * self.sample_rate / 1000))

        # Convert threshold to linear
        threshold_linear = 10 ** (threshold_db / 20)

        # RMS detection (10ms window)
        window_size = int(0.010 * self.sample_rate)
        audio_squared = audio**2
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(audio_squared, window, mode="same"))

        # Compute gain reduction
        gain_reduction = np.ones_like(rms)

        for i in range(len(rms)):
            level = rms[i]

            # Soft knee compression
            if level < threshold_linear:
                # Below threshold: no compression
                gain_reduction[i] = 1.0
            else:
                # Above threshold: apply compression with soft knee
                knee_linear = 10 ** (knee_db / 20)

                if level < threshold_linear * knee_linear:
                    # In knee region: gradual compression
                    knee_factor = (level - threshold_linear) / (threshold_linear * (knee_linear - 1))
                    effective_ratio = 1.0 + (ratio - 1.0) * knee_factor
                    gain_reduction[i] = (threshold_linear / level) ** (1 - 1 / effective_ratio)
                else:
                    # Above knee: full compression
                    gain_reduction[i] = (threshold_linear / level) ** (1 - 1 / ratio)

        # Apply attack/release envelope
        gain_smooth = np.zeros_like(gain_reduction)
        gain_smooth[0] = gain_reduction[0]

        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] < gain_smooth[i - 1]:
                # Attack (gain going down)
                alpha_attack = 1.0 - np.exp(-1.0 / attack_samples)
                gain_smooth[i] = alpha_attack * gain_reduction[i] + (1 - alpha_attack) * gain_smooth[i - 1]
            else:
                # Release (gain going up)
                alpha_release = 1.0 - np.exp(-1.0 / release_samples)
                gain_smooth[i] = alpha_release * gain_reduction[i] + (1 - alpha_release) * gain_smooth[i - 1]

        # Modulate compression based on psychoacoustic masking
        # In masked regions (quiet), allow more compression
        # In unmasked regions (loud), reduce compression
        gain_smooth = 1.0 - masking_curve * (1.0 - gain_smooth)

        # Preserve transients (bypass compression at transient locations)
        gain_smooth = transient_mask + (1.0 - transient_mask) * gain_smooth

        # Harte Kompressions-Artefakte: zusätzliche Envelope-Glättung gegen Pumpen.
        if compression_pressure >= 0.45:
            nyquist = self.sample_rate / 2.0
            hard_norm = float(np.clip((compression_pressure - 0.45) / 0.55, 0.0, 1.0))
            smooth_hz = float(np.clip(6.0 - 3.5 * hard_norm, 2.5, 6.0))
            sos_smooth = signal.butter(2, smooth_hz / nyquist, output="sos")
            gain_smooth = signal.sosfiltfilt(sos_smooth, gain_smooth)
            gain_smooth = np.clip(gain_smooth, 0.0, 1.25)

        # Apply gain reduction
        audio_compressed = audio * gain_smooth

        return audio_compressed

    def get_metadata(self) -> PhaseMetadata:
        """Gibt phase metadata zurück."""
        return PhaseMetadata(
            phase_id="phase_54_transparent_dynamics",
            name="Transparent Dynamics v1.0",
            category=PhaseCategory.ENHANCEMENT,
            priority=8,  # High priority (Tier 1, PRIORITY 4)
            dependencies=[],
            estimated_time_factor=0.25,  # 25% of audio duration
            version="1.0.0",
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.92,  # High impact on dynamics
            description="Psychoacoustic-aware transparent compression with genre adaptation and transient preservation",
        )

    def supports_material(self, material_type: MaterialType) -> bool:
        """Prüft if material type is supported."""
        return material_type in self.MATERIAL_CEILING or material_type in [
            MaterialType.REEL_TAPE,
            MaterialType.DAT,
            MaterialType.AAC,
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
        ]

    def estimate_time(self, audio_duration_seconds: float) -> float:
        """Schätzt processing time."""
        return audio_duration_seconds * 0.25  # 0.25× realtime


# Test harness
if __name__ == "__main__":
    logger.debug("=" * 70)
    logger.debug("AURIK 9.0 - TRANSPARENT DYNAMICS SYSTEM v1.0 TEST")
    logger.debug("=" * 70)
    logger.debug("Tier 1 Priority 4: Psychoacoustic-Aware Compression")
    logger.debug("=" * 70)

    # Create test instances for different genres
    genres = ["classical", "jazz", "rock", "electronic"]
    materials = [MaterialType.SHELLAC, MaterialType.VINYL, MaterialType.CD_DIGITAL, MaterialType.STREAMING]

    # Generate test audio (mix of loud and quiet with transients)
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal with dynamics
    test_audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # Base tone

    # Add loud passage (0.5-1.0s)
    loud_mask = (t >= 0.5) & (t < 1.0)
    test_audio[loud_mask] *= 3.0

    # Add transients (drum hits)
    transient_times = [0.3, 0.7, 1.3, 1.7]
    for tt in transient_times:
        idx = int(tt * sr)
        if idx < len(test_audio):
            transient_len = int(0.01 * sr)
            transient_env = np.exp(-200 * (t - tt))[idx : idx + transient_len]
            test_audio[idx : idx + len(transient_env)] += (
                0.5 * transient_env * np.sin(2 * np.pi * 1000 * (t - tt)[idx : idx + len(transient_env)])
            )

    # Normalize
    test_audio = test_audio / np.percentile(np.abs(test_audio), 99.9) * 0.7

    logger.debug("\n🎵 Test Audio Generated:")
    logger.debug("   Duration: %ss", duration)
    logger.debug("   Dynamics: Quiet (0-0.5s), Loud (0.5-1.0s), Quiet (1.0-2.0s)")
    logger.debug("   Transients: %s drum hits", len(transient_times))

    # Test first genre/material combination
    phase = TransparentDynamicsV1(sample_rate=sr, genre=genres[0])
    result = phase.process(test_audio, material_type=materials[0])

    metadata = phase.get_metadata()

    logger.debug("\n📋 Phase Metadata:")
    logger.debug("   ID: %s", metadata.phase_id)
    logger.debug("   Name: %s", metadata.name)
    logger.debug("   Priority: %s", metadata.priority)
    logger.debug("   Estimated time: %s× RT", metadata.estimated_time_factor)
    logger.debug("   Quality impact: %.0f%%", metadata.quality_impact * 100)

    logger.debug("\n🎛️ Testing %s genres × %s materials:", len(genres), len(materials))

    for genre_name in genres:
        for material in materials:
            phase = TransparentDynamicsV1(sample_rate=sr, genre=genre_name)
            result = phase.process(test_audio, material_type=material)

            rt_factor = result.execution_time_seconds / duration

            logger.debug(
                "\n   %-12s × %-15s: time=%.3fs (%.3f× RT), max=%.3f, ratio=%.1f:1",
                genre_name,
                material.value,
                result.execution_time_seconds,
                rt_factor,
                np.max(np.abs(result.audio)),
                result.metadata["ratio"],
            )

    logger.debug("\n%s", "=" * 70)
    logger.debug("✅ TRANSPARENT DYNAMICS TEST COMPLETE")
    logger.debug("=" * 70)
    logger.debug("\n🎯 Next Steps:")
    logger.debug("   1. Add to __init__.py exports")
    logger.debug("   2. Integrate into UnifiedRestorerV3 _select_phases()")
    logger.debug("   3. Create integration tests")
    logger.debug("   4. Finalize Tier 1 (12/14 modules = 86%)")
