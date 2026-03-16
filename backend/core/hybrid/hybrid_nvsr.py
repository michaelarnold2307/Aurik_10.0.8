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

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.fft import rfft, rfftfreq
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
        self._init_audiosr()

    def _init_audiosr(self) -> None:
        """Initialize AudioSR plugin if available"""
        try:
            from plugins.audiosr_plugin import AudioSRPlugin

            self.audiosr_plugin = AudioSRPlugin(timeout=300)
            logger.info("AudioSR plugin initialized for NVSR")
        except Exception as e:
            logger.warning(f"AudioSR plugin not available: {e}")
            self.audiosr_plugin = None

    def restore_bandwidth(
        self,
        audio: np.ndarray,
        sample_rate: int,
        dsp_restored_audio: np.ndarray | None = None,
        material_type: str = "unknown",
    ) -> NVSRResult:
        """
        Main entry point for bandwidth restoration.

        Args:
            audio: Input audio (may be low-bandwidth)
            sample_rate: Sample rate in Hz
            dsp_restored_audio: Optional pre-computed DSP restoration (SBR + LPC)
            material_type: Material type for adaptive processing

        Returns:
            NVSRResult with restored audio and metadata
        """
        start_time = time.time()

        # Detect current bandwidth
        detected_bandwidth = self._detect_bandwidth(audio, sample_rate)
        logger.info(f"Detected bandwidth: {detected_bandwidth:.0f} Hz")

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

        return result

    def _detect_bandwidth(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Detect the effective bandwidth of the audio signal.

        Returns frequency (Hz) where energy drops below -40 dB.
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Compute FFT
        fft_data = rfft(audio)
        freqs = rfftfreq(len(audio), d=1.0 / sample_rate)

        # Compute magnitude spectrum in dB
        magnitude_db = 20 * np.log10(np.abs(fft_data) + 1e-10)

        # Normalize to peak
        magnitude_db -= np.max(magnitude_db)

        # Find rolloff frequency (-40 dB threshold)
        rolloff_threshold = -40.0
        rolloff_indices = np.where(magnitude_db > rolloff_threshold)[0]

        if len(rolloff_indices) > 0:
            rolloff_freq = freqs[rolloff_indices[-1]]
        else:
            rolloff_freq = sample_rate / 2.0

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
        """AudioSR-only path"""
        if self.audiosr_plugin is None:
            logger.warning("AudioSR not available, falling back to DSP")
            return self._apply_dsp_only(audio, sample_rate, detected_bandwidth)

        try:
            # Apply AudioSR
            restored = self._run_audiosr(audio, sample_rate)

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
            logger.error(f"AudioSR processing failed: {e}")
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

        if self.audiosr_plugin is None:
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
            restored = self._run_audiosr(audio, sample_rate)

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
            logger.error(f"AudioSR processing failed: {e}")
            return NVSRResult(
                restored_audio=base_audio,
                strategy_used="adaptive_dsp_fallback",
                dsp_applied=True,
                audiosr_applied=False,
                detected_bandwidth_hz=detected_bandwidth,
                target_bandwidth_hz=self.config.target_bandwidth_hz,
                processing_time_sec=0.0,
                skipped_reason=f"AudioSR error: {str(e)}",
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

        if self.audiosr_plugin is None:
            logger.warning("AudioSR not available, falling back to DSP")
            return self._apply_dsp_only(base_audio, sample_rate, detected_bandwidth)

        try:
            # Apply AudioSR
            audiosr_result = self._run_audiosr(audio, sample_rate)

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
            logger.error(f"Hybrid processing failed: {e}")
            return self._apply_dsp_only(base_audio, sample_rate, detected_bandwidth)

    def _run_audiosr(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Run AudioSR neural super-resolution.

        Handles temporary file I/O for Docker-based plugin.
        """
        import soundfile as sf

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            input_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            # Write input audio
            sf.write(input_path, audio.T if audio.ndim > 1 else audio, sample_rate)

            # Run AudioSR
            self.audiosr_plugin.process(input_path, output_path, target_sr=self.config.audiosr_target_sr)

            # Read output audio
            restored, out_sr = sf.read(output_path, always_2d=True)
            restored = restored.T  # (channels, samples)

            # Resample if needed
            if out_sr != sample_rate:
                from scipy import signal as sp_signal

                num_samples = int(len(restored[0]) * sample_rate / out_sr)
                restored = np.array([sp_signal.resample(restored[ch], num_samples) for ch in range(restored.shape[0])])

            return restored
        finally:
            # Cleanup
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def _blend_audio(
        self, audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int, crossover_freq: float, blend_ratio: float
    ) -> np.ndarray:
        """
        Blend two audio signals with crossover frequency.

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

        # Apply filters
        if audio_a.ndim > 1:
            audio_a_low = np.array([signal.sosfilt(sos_low, audio_a[ch]) for ch in range(audio_a.shape[0])])
            audio_b_high = np.array([signal.sosfilt(sos_high, audio_b[ch]) for ch in range(audio_b.shape[0])])
        else:
            audio_a_low = signal.sosfilt(sos_low, audio_a)
            audio_b_high = signal.sosfilt(sos_high, audio_b)

        # Blend: keep DSP low frequencies, add AudioSR high frequencies
        blended = audio_a_low + blend_ratio * audio_b_high

        return blended


def create_nvsr_config(quality_mode: str = "balanced", material_type: str = "unknown") -> NVSRConfig:
    """
    Create NVSR config based on quality mode and material type.

    Args:
        quality_mode: 'fast', 'balanced', or 'maximum'
        material_type: 'tape', 'vinyl', 'shellac', etc.

    Returns:
        NVSRConfig instance
    """
    if quality_mode == "fast":
        return NVSRConfig(
            strategy=NVSRStrategy.DSP_ONLY,
            target_bandwidth_hz=20000,
            bandwidth_threshold_hz=12000,
            audiosr_target_sr=48000,
            blend_ratio=0.0,  # No AudioSR
            confidence_threshold=1.0,  # Always skip AudioSR
        )
    elif quality_mode == "maximum":
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
    else:  # balanced
        return NVSRConfig(
            strategy=NVSRStrategy.ADAPTIVE,
            target_bandwidth_hz=20000,
            bandwidth_threshold_hz=12000,  # Skip AudioSR if > 12 kHz
            audiosr_target_sr=48000,
            blend_ratio=0.6,
            confidence_threshold=0.65,
        )
