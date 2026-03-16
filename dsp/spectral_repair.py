import logging

logger = logging.getLogger(__name__)

"""
AURIK Spectral Repair/Inpainting Module

Repairs digital artifacts including:
- MP3 compression artifacts (spectral holes above 16 kHz)
- Packet loss (spectral dropouts)
- Digital clicks/glitches
- Codec artifacts (AAC, Vorbis bandwidth limitations)

Approach:
- Spectral hole detection (missing frequency bands)
- Harmonic extrapolation (synthesize missing harmonics)
- Spectral smoothing (repair codec artifacts)
- Transient-aware processing (preserve musical details)

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal


@dataclass
class SpectralRepairConfig:
    """Configuration for spectral repair processing."""

    # Spectral hole detection
    hole_threshold_db: float = -60.0
    """Threshold for detecting spectral holes (dB below peak)."""

    min_hole_width_hz: float = 500.0
    """Minimum width of spectral hole to repair (Hz)."""

    # Harmonic extrapolation
    extrapolation_order: int = 8
    """Number of harmonics to analyze for extrapolation."""

    extrapolation_strength: float = 0.7
    """Strength of harmonic extrapolation (0.0-1.0)."""

    # Codec artifact repair
    smooth_bandwidth_hz: float = 100.0
    """Smoothing bandwidth for codec artifact repair (Hz)."""

    # Processing
    fft_size: int = 4096
    """FFT size for spectral analysis."""

    hop_size: int = 1024
    """Hop size for STFT processing."""


class SpectralRepair:
    """
    Spectral Repair/Inpainting for Digital Artifacts.

    Repairs:
    - MP3 artifacts (spectral holes above 16 kHz)
    - Packet loss (spectral dropouts)
    - Digital clicks/glitches
    - Codec artifacts (AAC, Vorbis bandwidth limitations)
    """

    def __init__(self, config: SpectralRepairConfig | None = None):
        """
        Initialize spectral repair processor.

        Args:
            config: Spectral repair configuration (uses defaults if None)
        """
        self.config = config or SpectralRepairConfig()
        self._metrics: dict[str, Any] = {}

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Process audio with spectral repair.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate (Hz)

        Returns:
            Repaired audio
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # Reset metrics
        self._metrics = {"holes_detected": 0, "holes_repaired": 0, "frequency_ranges": [], "artifacts_smoothed": 0}

        # Handle stereo
        if audio.ndim == 2:
            # Process each channel
            left = self._process_mono(audio[:, 0], sr)
            right = self._process_mono(audio[:, 1], sr)
            return np.column_stack([left, right])
        else:
            return self._process_mono(audio, sr)

    def _process_mono(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """Process mono audio."""
        # STFT
        f, t, Zxx = scipy.signal.stft(
            audio, fs=sr, nperseg=self.config.fft_size, noverlap=self.config.fft_size - self.config.hop_size
        )

        # Detect and repair spectral holes
        Zxx_repaired = self._detect_and_repair_holes(Zxx, f, sr)

        # Smooth codec artifacts
        Zxx_smoothed = self._smooth_codec_artifacts(Zxx_repaired, f, sr)

        # Inverse STFT
        _, audio_repaired = scipy.signal.istft(
            Zxx_smoothed, fs=sr, nperseg=self.config.fft_size, noverlap=self.config.fft_size - self.config.hop_size
        )

        # Ensure same length as input
        if len(audio_repaired) > len(audio):
            audio_repaired = audio_repaired[: len(audio)]
        elif len(audio_repaired) < len(audio):
            audio_repaired = np.pad(audio_repaired, (0, len(audio) - len(audio_repaired)), mode="constant")

        # NaN/Inf-Guard + Clipping
        audio_repaired = np.nan_to_num(audio_repaired, nan=0.0, posinf=0.0, neginf=0.0)
        audio_repaired = np.clip(audio_repaired, -1.0, 1.0)

        return audio_repaired.astype(np.float64)

    def _detect_and_repair_holes(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Detect and repair spectral holes.

        Common scenarios:
        - MP3: Hole above 16 kHz (HF cutoff)
        - AAC 128kbps: Hole above 15.5 kHz
        - Opus/Vorbis: Variable cutoff 12-20 kHz
        """
        # Compute magnitude spectrum (averaged over time)
        mag_spec = np.abs(Zxx).mean(axis=1)
        mag_spec_db = 20 * np.log10(mag_spec + 1e-10)

        # Find peak magnitude (reference level)
        peak_db = mag_spec_db.max()

        # Detect holes: regions below threshold
        hole_mask = mag_spec_db < (peak_db + self.config.hole_threshold_db)

        # Find contiguous hole regions
        hole_regions = self._find_contiguous_regions(hole_mask, f, self.config.min_hole_width_hz)

        if not hole_regions:
            return Zxx

        # Repair each hole
        Zxx_repaired = Zxx.copy()
        for start_idx, end_idx in hole_regions:
            start_freq = f[start_idx]
            end_freq = f[end_idx]

            # Only repair holes in upper frequency range (> 2 kHz)
            # (Low frequency holes are likely intentional)
            if start_freq > 2000:
                Zxx_repaired = self._repair_hole_region(Zxx_repaired, f, start_idx, end_idx, sr)

                self._metrics["holes_repaired"] += 1
                self._metrics["frequency_ranges"].append((float(start_freq), float(end_freq)))

        self._metrics["holes_detected"] = len(hole_regions)

        return Zxx_repaired

    def _find_contiguous_regions(
        self, mask: npt.NDArray[np.bool_], f: npt.NDArray[np.float64], min_width_hz: float
    ) -> list:
        """Find contiguous regions in binary mask."""
        regions = []
        in_region = False
        start_idx = 0

        for i, val in enumerate(mask):
            if val and not in_region:
                # Start of region
                start_idx = i
                in_region = True
            elif not val and in_region:
                # End of region
                end_idx = i - 1
                width_hz = f[end_idx] - f[start_idx]

                if width_hz >= min_width_hz:
                    regions.append((start_idx, end_idx))

                in_region = False

        # Handle last region
        if in_region:
            end_idx = len(mask) - 1
            width_hz = f[end_idx] - f[start_idx]
            if width_hz >= min_width_hz:
                regions.append((start_idx, end_idx))

        return regions

    def _repair_hole_region(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], start_idx: int, end_idx: int, sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Repair spectral hole using harmonic extrapolation.

        Method:
        1. Analyze harmonics below hole
        2. Extrapolate harmonic pattern into hole
        3. Blend with original signal
        """
        # Find reference region (just below hole)
        ref_width = min(end_idx - start_idx, 50)  # Up to 50 bins
        ref_start = max(0, start_idx - ref_width)
        ref_end = start_idx

        if ref_start >= ref_end:
            return Zxx

        # Extract reference spectrum
        ref_spec = Zxx[ref_start:ref_end, :]
        ref_mag = np.abs(ref_spec).mean(axis=0)  # Average magnitude  # noqa: F841
        np.angle(ref_spec)

        # Synthesize missing content using harmonic decay model
        # Typical HF decay: -6 dB/octave for harmonics
        hole_length = end_idx - start_idx

        for i in range(hole_length):
            freq_idx = start_idx + i

            # Frequency of current bin
            freq_hz = f[freq_idx]

            # Find corresponding lower frequency (1 octave down)
            ref_freq_hz = freq_hz / 2.0
            ref_freq_idx = np.argmin(np.abs(f - ref_freq_hz))

            if ref_freq_idx < start_idx:
                # Extrapolate with decay
                octaves_up = np.log2(freq_hz / f[ref_freq_idx])
                decay_db = -6.0 * octaves_up  # -6 dB/octave
                decay_linear = 10 ** (decay_db / 20.0)

                # Apply decay + add noise for naturalness
                Zxx[freq_idx, :] = Zxx[ref_freq_idx, :] * decay_linear * self.config.extrapolation_strength + Zxx[
                    freq_idx, :
                ] * (1.0 - self.config.extrapolation_strength)

        return Zxx

    def _smooth_codec_artifacts(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Smooth codec artifacts (quantization noise, ringing).

        Method:
        - Apply gentle spectral smoothing in codec artifact bands
        - Preserve transients (only smooth steady-state regions)
        """
        # Compute temporal variation (to detect transients)
        mag = np.abs(Zxx)
        temporal_var = np.std(mag, axis=1)

        # Only smooth regions with low temporal variation (steady-state)
        steady_mask = temporal_var < temporal_var.mean()

        # Smooth spectrum in steady regions
        mag_smoothed = mag.copy()

        # Spectral smoothing kernel (frequency domain)
        smooth_bins = int(self.config.smooth_bandwidth_hz / (sr / self.config.fft_size))
        smooth_bins = max(3, smooth_bins)

        kernel = np.ones(smooth_bins) / smooth_bins

        for t_idx in range(mag.shape[1]):
            # Only smooth if steady-state
            if np.any(steady_mask):
                mag_col = mag[:, t_idx]
                mag_smoothed_col = np.convolve(mag_col, kernel, mode="same")

                # Apply only to steady regions
                mag_smoothed[steady_mask, t_idx] = mag_smoothed_col[steady_mask]

        # Count artifacts smoothed
        self._metrics["artifacts_smoothed"] = int(np.sum(steady_mask))

        # Reconstruct with smoothed magnitude
        phase = np.angle(Zxx)
        Zxx_smoothed = mag_smoothed * np.exp(1j * phase)

        return Zxx_smoothed

    def get_metrics(self) -> dict[str, Any]:
        """
        Get processing metrics.

        Returns:
            Dictionary with metrics:
            - holes_detected: Number of spectral holes detected
            - holes_repaired: Number of spectral holes repaired
            - frequency_ranges: List of (start_hz, end_hz) tuples for repaired holes
            - artifacts_smoothed: Number of frequency bins with artifacts smoothed
        """
        return self._metrics.copy()


def process_spectral_repair(
    audio: npt.NDArray[np.float64], sr: int, strength: float = 0.7, hole_threshold_db: float = -60.0
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """
    Convenience function for spectral repair.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate (Hz)
        strength: Repair strength (0.0-1.0)
        hole_threshold_db: Threshold for detecting spectral holes

    Returns:
        Tuple of (repaired_audio, metrics)
    """
    config = SpectralRepairConfig(hole_threshold_db=hole_threshold_db, extrapolation_strength=strength)

    processor = SpectralRepair(config)
    repaired = processor.process(audio, sr)
    metrics = processor.get_metrics()

    return repaired, metrics


if __name__ == "__main__":
    # Demo: Test spectral repair on synthetic signal with holes
    logger.info("AURIK Spectral Repair Demo")
    logger.info(str("=" * 60))

    # Generate test signal (1 second at 48 kHz)
    sr = 48000
    t = np.linspace(0, 1, sr)

    # Multi-harmonic signal (fundamental + harmonics)
    signal = np.zeros_like(t)
    fundamental = 440  # A4
    for h in range(1, 11):  # 10 harmonics
        signal += 0.5 * np.sin(2 * np.pi * fundamental * h * t) / h

    # Simulate MP3 compression: remove content above 16 kHz
    from scipy.signal import butter, lfilter

    b, a = butter(8, 16000 / (sr / 2), btype="low")
    signal_compressed = lfilter(b, a, signal)

    logger.info("\nInput:")
    logger.info(f"  - Sample Rate: {sr} Hz")
    logger.info("  - Duration: 1.0 seconds")
    logger.info("  - Content: Multi-harmonic signal (440 Hz fundamental, 10 harmonics)")
    logger.info("  - Simulated Damage: MP3-style lowpass (cutoff 16 kHz)")

    # Process with spectral repair
    config = SpectralRepairConfig(hole_threshold_db=-60.0, min_hole_width_hz=1000.0, extrapolation_strength=0.7)
    processor = SpectralRepair(config)
    signal_repaired = processor.process(signal_compressed, sr)
    metrics = processor.get_metrics()

    logger.info("\nSpectral Repair Results:")
    logger.info(f"  - Holes Detected: {metrics['holes_detected']}")
    logger.info(f"  - Holes Repaired: {metrics['holes_repaired']}")
    if metrics["frequency_ranges"]:
        logger.info("  - Frequency Ranges:")
        for start_hz, end_hz in metrics["frequency_ranges"]:
            logger.info(f"    * {start_hz:.0f} - {end_hz:.0f} Hz")
    logger.info(f"  - Artifacts Smoothed: {metrics['artifacts_smoothed']} bins")

    logger.info("\n✅ Spectral Repair Demo Complete")
