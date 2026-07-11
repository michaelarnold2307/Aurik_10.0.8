"""
Hybrid NVSR (Neural Vocoder Super Resolution) - Phase 06/07
============================================================

Combines traditional DSP bandwidth extension (SBR + LPC) with ML-based
Neural Vocoder Super Resolution using AudioSR for high-quality frequency restoration.

Architecture
------------
- DSP Baseline: Spectral Band Replication (SBR) + LPC harmonics
- ML Enhancement: AudioSR neural super-resolution
- Hybrid: Combine SBR for stability + AudioSR for naturalness

Quality Modes
-------------
- FAST: DSP-only SBR + LPC (~0.5× RT)
- BALANCED: Adaptive (skip AudioSR if high bandwidth detected)
- MAXIMUM: Full AudioSR neural bandwidth extension (~2-3× RT)

Author: Aurik Phase 06/07 ML-Hybrid Integration
Version: 1.0
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.fft as np_fft
import scipy.signal as signal

logger = logging.getLogger(__name__)


class NVSRStrategy(Enum):
    """Strategy for bandwidth extension"""

    DSP_ONLY = "dsp_only"  # Pure SBR + LPC (fast)
    AUDIOSR_ONLY = "audiosr_only"  # Pure AudioSR (slow, high quality)
    ADAPTIVE = "adaptive"  # Skip AudioSR if bandwidth sufficient
    HYBRID = "hybrid"  # Blend SBR + AudioSR


@dataclass
class NVSRConfig:
    """Configuration for NVSR processing"""

    strategy: NVSRStrategy = NVSRStrategy.ADAPTIVE
    target_bandwidth_hz: int = 20000  # Target upper frequency
    bandwidth_threshold_hz: int = 12000  # Skip ML if > this
    audiosr_target_sr: int = 48000  # AudioSR output sample rate
    blend_ratio: float = 0.7  # AudioSR weight in hybrid (0.0-1.0)
    confidence_threshold: float = 0.65  # Skip AudioSR if DSP confidence above this


@dataclass
class NVSRResult:
    """Result of NVSR bandwidth extension"""

    restored_audio: np.ndarray
    strategy_used: str
    dsp_applied: bool
    audiosr_applied: bool
    detected_bandwidth_hz: float
    target_bandwidth_hz: int
    processing_time_sec: float
    skipped_reason: str | None = None
    metadata: dict[str, Any] | None = None


class HybridNVSR:
    """
    Hybrid NVSR bandwidth extension combining DSP and ML approaches.

    DSP Approach (SBR + LPC):
    - Fast, stable, predictable
    - Good for moderate bandwidth extension
    - Material-adaptive rolloff frequencies

    ML Approach (AudioSR):
    - High quality, natural harmonics
    - Best for severe bandwidth limitations
    - Expensive (~2-3× RT)

    Adaptive Strategy:
    - Analyze existing bandwidth
    - Skip AudioSR if bandwidth > 12 kHz
    - Use DSP only for fast path
    """

    def __init__(self, config: NVSRConfig | None = None) -> None:
        self.config = config or NVSRConfig()
        self.audiosr_plugin = None
        self._ml_guard_events: list[dict[str, Any]] = []

    def _init_audiosr(self) -> None:
        """Initialisiert AudioSR plugin lazily when needed."""
        try:
            from plugins.audiosr_plugin import AudioSRPlugin

            self.audiosr_plugin = AudioSRPlugin(timeout=300)  # type: ignore[assignment]
            logger.info("AudioSR plugin initialized for NVSR")
        except Exception as e:
            logger.warning("AudioSR plugin not available: %s", e)
            self.audiosr_plugin = None

    def _has_sufficient_ml_headroom(self, audio: np.ndarray, sample_rate: int, phase_id: str) -> bool:
        """Gibt True when enough physical RAM is available for AudioSR stage zurück."""
        try:
            import gc

            import psutil
        except Exception as e:
            logger.warning("hybrid_nvsr.py::_has_sufficient_ml_headroom fallback: %s", e)
            return True

        n_samples = int(
            audio.shape[-1]
            if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            else audio.shape[0]
        )
        n_channels = 2 if audio.ndim == 2 else 1
        duration_s = n_samples / float(max(1, sample_rate))

        required_gb = 6.0
        if n_channels >= 2:
            required_gb += 2.0
        if duration_s >= 180.0:
            required_gb += 2.0
        elif duration_s >= 60.0:
            required_gb += 1.0

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
                    "phase_id": phase_id,
                    "model": "AudioSR",
                    "reason": "insufficient_physical_ram_headroom",
                    "required_gb": float(required_gb),
                    "available_gb": float(available_gb),
                    "channels": int(n_channels),
                    "duration_s": float(duration_s),
                    "fallback": "dsp",
                }
            )
            logger.warning(
                "NVSR RAM guard triggered: %.1f GB available, %.1f GB required (duration=%.1fs channels=%d) - using DSP fallback",
                available_gb,
                required_gb,
                duration_s,
                n_channels,
            )
            return False

        return True

    def _get_audiosr_plugin(self, audio: np.ndarray, sample_rate: int, phase_id: str) -> Any:
        """Gibt AudioSR plugin only when guard allows ML stage zurück."""
        if not self._has_sufficient_ml_headroom(audio, sample_rate, phase_id):
            return None
        if self.audiosr_plugin is None:
            self._init_audiosr()
        return self.audiosr_plugin

    def restore_bandwidth(
        self,
        audio: np.ndarray,
        sample_rate: int,
        dsp_restored_audio: np.ndarray | None = None,
        material_type: str = "unknown",
    ) -> NVSRResult:
        """
        Haupt-entry point for bandwidth restoration.

        Args:
            audio: Input audio (may be low-bandwidth)
            sample_rate: Sample rate in Hz
            dsp_restored_audio: Optional pre-computed DSP restoration (SBR + LPC)
            material_type: Material type for adaptive processing

        Returns:
            NVSRResult with restored audio and metadata
        """
        start_time = time.time()
        self._ml_guard_events = []

        # Detect current bandwidth
        detected_bandwidth = self._detect_bandwidth(audio, sample_rate)
        logger.info("Detected bandwidth: %.0f Hz", detected_bandwidth)

        # Choose strategy
        strategy = self.config.strategy

        # Strategy routing
        if strategy == NVSRStrategy.DSP_ONLY:
            result = self._apply_dsp_only(dsp_restored_audio or audio, sample_rate, detected_bandwidth)
        elif strategy == NVSRStrategy.AUDIOSR_ONLY:
            result = self._apply_audiosr_only(audio, sample_rate, detected_bandwidth)
        elif strategy == NVSRStrategy.ADAPTIVE:
            result = self._apply_adaptive(audio, sample_rate, dsp_restored_audio, detected_bandwidth, material_type)
        elif strategy == NVSRStrategy.HYBRID:
            result = self._apply_hybrid(audio, sample_rate, dsp_restored_audio, detected_bandwidth)
        else:
            # Fallback to DSP
            result = self._apply_dsp_only(dsp_restored_audio or audio, sample_rate, detected_bandwidth)

        # Update processing time
        result.processing_time_sec = time.time() - start_time
        if result.metadata is None:
            result.metadata = {}
        if self._ml_guard_events:
            result.metadata["ml_guard_events"] = list(self._ml_guard_events)
            result.metadata["deferred_for_kmv"] = ["phase_06_frequency_restoration"]

        return result

    def _detect_bandwidth(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Erkennt the effective bandwidth of the audio signal.

        Returns frequency (Hz) where energy drops below -40 dB.
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Compute FFT
        audio_mono = np.asarray(audio, dtype=np.float64)
        fft_data = np_fft.rfft(audio_mono)
        freqs = np_fft.rfftfreq(len(audio), d=1.0 / sample_rate)

        # Compute magnitude spectrum in dB
        magnitude = np.asarray(np.abs(fft_data), dtype=np.float64)
        magnitude_db = np.asarray(20.0 * np.log10(magnitude + 1e-10), dtype=np.float64)

        # Normalize to peak
        magnitude_db -= np.max(magnitude_db)

        # Find rolloff frequency (-40 dB threshold)
        rolloff_threshold = -40.0
        rolloff_indices = np.where(magnitude_db > rolloff_threshold)[0]

        rolloff_freq = freqs[rolloff_indices[-1]] if len(rolloff_indices) > 0 else sample_rate / 2.0

        return float(rolloff_freq)

    def _apply_dsp_only(self, audio: np.ndarray, sample_rate: int, detected_bandwidth: float) -> NVSRResult:
        """DSP-only path (SBR + LPC already applied)"""
        return NVSRResult(
            restored_audio=audio,
            strategy_used="dsp_only",
            dsp_applied=True,
            audiosr_applied=False,
            detected_bandwidth_hz=detected_bandwidth,
            target_bandwidth_hz=self.config.target_bandwidth_hz,
            processing_time_sec=0.0,
            skipped_reason="DSP-only mode selected",
        )

    def _apply_audiosr_only(self, audio: np.ndarray, sample_rate: int, detected_bandwidth: float) -> NVSRResult:
        """AudioSR-only path."""
        plugin = self._get_audiosr_plugin(audio, sample_rate, "phase_06_frequency_restoration")
        if plugin is None:
            logger.warning("AudioSR not available, falling back to DSP")
            return self._apply_dsp_only(audio, sample_rate, detected_bandwidth)

        try:
            # Apply AudioSR
            restored = self._run_audiosr(audio, sample_rate, plugin)

            return NVSRResult(
                restored_audio=restored,
                strategy_used="audiosr_only",
                dsp_applied=False,
                audiosr_applied=True,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
            )
        except Exception as e:
            logger.error("AudioSR processing failed: %s", e)
            return self._apply_dsp_only(audio, sample_rate, detected_bandwidth)

    def _apply_adaptive(
        self,
        audio: np.ndarray,
        sample_rate: int,
        dsp_restored_audio: np.ndarray | None,
        detected_bandwidth: float,
        material_type: str,
    ) -> NVSRResult:
        """
        Adaptive strategy: Skip AudioSR if bandwidth already sufficient.
        """
        # Use DSP restoration if provided
        base_audio = dsp_restored_audio if dsp_restored_audio is not None else audio

        # Check if bandwidth is already sufficient
        if detected_bandwidth >= self.config.bandwidth_threshold_hz:
            logger.info(
                f"Bandwidth {detected_bandwidth:.0f} Hz >= {self.config.bandwidth_threshold_hz} Hz, skipping AudioSR"
            )
            return NVSRResult(
                restored_audio=base_audio,
                strategy_used="adaptive_dsp_only",
                dsp_applied=True,
                audiosr_applied=False,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
                skipped_reason=f"Bandwidth sufficient ({detected_bandwidth:.0f} Hz >= {self.config.bandwidth_threshold_hz} Hz)",
            )

        # Bandwidth insufficient, apply AudioSR
        logger.info(
            f"Bandwidth {detected_bandwidth:.0f} Hz < {self.config.bandwidth_threshold_hz} Hz, applying AudioSR"
        )

        plugin = self._get_audiosr_plugin(audio, sample_rate, "phase_06_frequency_restoration")
        if plugin is None:
            logger.warning("AudioSR not available, using DSP restoration")
            return NVSRResult(
                restored_audio=base_audio,
                strategy_used="adaptive_dsp_fallback",
                dsp_applied=True,
                audiosr_applied=False,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
                skipped_reason="AudioSR plugin not available",
            )

        try:
            # Apply AudioSR for bandwidth extension
            restored = self._run_audiosr(audio, sample_rate, plugin)

            return NVSRResult(
                restored_audio=restored,
                strategy_used="adaptive_audiosr",
                dsp_applied=True,
                audiosr_applied=True,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
            )
        except Exception as e:
            logger.error("AudioSR processing failed: %s", e)
            return NVSRResult(
                restored_audio=base_audio,
                strategy_used="adaptive_dsp_fallback",
                dsp_applied=True,
                audiosr_applied=False,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
                skipped_reason=f"AudioSR error: {e!s}",
            )

    def _apply_hybrid(
        self, audio: np.ndarray, sample_rate: int, dsp_restored_audio: np.ndarray | None, detected_bandwidth: float
    ) -> NVSRResult:
        """
        Hybrid strategy: Blend DSP (SBR) with AudioSR.

        Approach:
        - Use DSP restoration (SBR + LPC) for stability
        - Use AudioSR for naturalness
        - Blend: DSP in low frequencies, AudioSR in high frequencies
        """
        base_audio = dsp_restored_audio if dsp_restored_audio is not None else audio

        plugin = self._get_audiosr_plugin(audio, sample_rate, "phase_06_frequency_restoration")
        if plugin is None:
            logger.warning("AudioSR not available, falling back to DSP")
            return self._apply_dsp_only(base_audio, sample_rate, detected_bandwidth)

        try:
            # Apply AudioSR
            audiosr_result = self._run_audiosr(audio, sample_rate, plugin)

            # Blend DSP and AudioSR
            # Strategy: DSP for low frequencies (stable), AudioSR for high frequencies (natural)
            crossover_freq = 8000  # Hz
            blended = self._blend_audio(
                base_audio, audiosr_result, sample_rate, crossover_freq, self.config.blend_ratio
            )

            return NVSRResult(
                restored_audio=blended,
                strategy_used="hybrid",
                dsp_applied=True,
                audiosr_applied=True,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
            )
        except Exception as e:
            logger.error("Hybrid processing failed: %s", e)
            return self._apply_dsp_only(base_audio, sample_rate, detected_bandwidth)

    def _run_audiosr(self, audio: np.ndarray, sample_rate: int, plugin: Any) -> np.ndarray:
        """
        Führt aus: AudioSR neural super-resolution via array-based plugin API.

        AudioSRPlugin.process() accepts [samples] or [channels, samples] and
        handles resampling, NaN-guards and target_sr internally.
        """
        if not self._has_sufficient_ml_headroom(audio, sample_rate, "phase_06_frequency_restoration"):
            raise RuntimeError("AudioSR guard triggered before inference")
        # §4.6b PLM Active-Guard — prevents Emergency-Eviction during inference
        _plm_nvsr = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_nvsr

            _plm_nvsr = _get_plm_nvsr()
            _plm_nvsr.set_active("AudioSR", True)
        except Exception as e:
            logger.warning("hybrid_nvsr.py::_run_audiosr fallback: %s", e)
        try:
            return plugin.process(audio, sample_rate, target_sr=self.config.audiosr_target_sr)  # type: ignore[no-any-return]
        finally:
            if _plm_nvsr is not None:
                try:
                    _plm_nvsr.set_active("AudioSR", False)
                except Exception as e:
                    logger.warning("hybrid_nvsr.py::_run_audiosr fallback: %s", e)

    def _blend_audio(
        self, audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int, crossover_freq: float, blend_ratio: float
    ) -> np.ndarray:
        """
        Mischt two audio signals with crossover frequency.

        Args:
            audio_a: Base audio (DSP)
            audio_b: Enhanced audio (AudioSR)
            sample_rate: Sample rate
            crossover_freq: Crossover frequency (Hz)
            blend_ratio: Weight for audio_b (0.0-1.0)

        Returns:
            Blended audio
        """
        # Ensure same shape
        min_samples = min(audio_a.shape[-1], audio_b.shape[-1])
        audio_a = audio_a[..., :min_samples]
        audio_b = audio_b[..., :min_samples]

        # Simple crossover blend
        # Low frequencies from audio_a, high frequencies from audio_b
        nyquist = sample_rate / 2.0
        normalized_cutoff = crossover_freq / nyquist

        # Design lowpass filter for audio_a contribution
        sos_low = signal.butter(4, normalized_cutoff, btype="low", output="sos")

        # Design highpass filter for audio_b contribution
        sos_high = signal.butter(4, normalized_cutoff, btype="high", output="sos")

        # Apply filters — §2.51 Zero-phase crossover: sosfiltfilt prevents group-delay
        # mismatch between LP/HP bands that causes comb-filter coloration at crossover freq.
        if audio_a.ndim > 1:
            audio_a_low = np.array([signal.sosfiltfilt(sos_low, audio_a[ch]) for ch in range(audio_a.shape[0])])
            audio_b_high = np.array([signal.sosfiltfilt(sos_high, audio_b[ch]) for ch in range(audio_b.shape[0])])
        else:
            audio_a_low = signal.sosfiltfilt(sos_low, audio_a)
            audio_b_high = signal.sosfiltfilt(sos_high, audio_b)

        # Blend: keep DSP low frequencies, add AudioSR high frequencies
        blended = audio_a_low + blend_ratio * audio_b_high

        return blended  # type: ignore[no-any-return]


def create_nvsr_config(quality_mode: str = "balanced", material_type: str = "unknown") -> NVSRConfig:
    """
    Erstellt NVSR config based on quality mode and material type.

    Args:
        quality_mode: 'fast', 'balanced', 'quality', 'maximum', 'restoration', or 'studio_2026'
        material_type: 'tape', 'vinyl', 'shellac', etc.

    Returns:
        NVSRConfig instance
    """
    # Alias normalisation (§2.46 / §0a mode-differentiation)
    _qm = (quality_mode or "balanced").lower()
    if _qm == "restoration":
        _qm = "balanced"
    elif _qm == "studio_2026":
        _qm = "maximum"

    if _qm == "fast":
        return NVSRConfig(
            strategy=NVSRStrategy.DSP_ONLY,
            target_bandwidth_hz=20000,
            bandwidth_threshold_hz=12000,
            audiosr_target_sr=48000,
            blend_ratio=0.0,  # No AudioSR
            confidence_threshold=1.0,  # Always skip AudioSR
        )
    elif _qm == "maximum":
        # Material-adaptive target bandwidth
        target_bw = {
            "shellac": 10000,  # Shellac limited to ~10 kHz restoration
            "vinyl": 20000,  # Vinyl full bandwidth
            "tape": 20000,  # Tape full bandwidth
        }.get(material_type, 20000)

        return NVSRConfig(
            strategy=NVSRStrategy.HYBRID,
            target_bandwidth_hz=target_bw,
            bandwidth_threshold_hz=999999,  # Always apply AudioSR
            audiosr_target_sr=48000,
            blend_ratio=0.7,  # 70% AudioSR
            confidence_threshold=0.0,  # Never skip
        )
    else:  # balanced / quality
        return NVSRConfig(
            strategy=NVSRStrategy.ADAPTIVE,
            target_bandwidth_hz=20000,
            bandwidth_threshold_hz=12000,  # Skip AudioSR if > 12 kHz
            audiosr_target_sr=48000,
            blend_ratio=0.6,
            confidence_threshold=0.65,
        )
