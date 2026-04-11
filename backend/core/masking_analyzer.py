#!/usr/bin/env python3
"""
Masking Analyzer - Psychoacoustic Masking Model
================================================

Implementiert Simultaneous und Temporal Masking nach psychoacoustischen Modellen.
Extrahiert aus Ultimate De-Esser für universelle Verwendung.

Masking Typen:
1. **Simultaneous Masking** (Frequency Masking):
   - Laute Töne maskieren leisere Töne in benachbarten Frequenzen
   - Spreading Function: Asymmetrisch (mehr nach oben)
   - Maskiert ~15-20 Bark breiter Bereich

2. **Temporal Masking**:
   - **Pre-Masking** (Backward): Maskierung vor Signal (~20ms)
   - **Post-Masking** (Forward): Maskierung nach Signal (~200ms)
   - Wird durch auditorische Persistenz verursacht

Anwendungen:
- Perceptual Audio Coding (MP3, AAC, Opus)
- Noise Reduction
- Dynamic Range Control
- Quality Assessment
- Audio Enhancement

Autor: Aurik v8.0 - Psychoacoustic Core
Lizenz: Proprietär
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASKING CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Temporal Masking Constants (nach psychoacustischen Studien)
PRE_MASKING_MS = 20.0  # Backward masking: ~20ms
POST_MASKING_MS = 200.0  # Forward masking: ~200ms

# Simultaneous Masking Slope (dB/Bark)
MASKING_SLOPE_LOWER = 27.0  # Slope below masker (steiler)
MASKING_SLOPE_UPPER = 10.0  # Slope above masker (flacher)

# Absolute Threshold of Hearing (ATH) in dB SPL
# Repräsentiert die leisteste Töne, die Menschen hören können
# Format: Frequency (Hz) → Threshold (dB SPL)
ABSOLUTE_THRESHOLD_HZ = [
    (20, 70),
    (25, 60),
    (31.5, 52),
    (40, 44),
    (50, 37),
    (63, 32),
    (80, 27),
    (100, 23),
    (125, 19),
    (160, 16),
    (200, 13),
    (250, 10),
    (315, 8),
    (400, 6),
    (500, 5),
    (630, 4),
    (800, 3),
    (1000, 2.5),
    (1250, 2),
    (1600, 1),
    (2000, 0),
    (2500, -1),
    (3150, -3),
    (4000, -5),
    (5000, -4),
    (6300, -2),
    (8000, 2),
    (10000, 8),
    (12500, 15),
    (16000, 25),
]


@dataclass
class MaskingProfile:
    """
    Represents a masking profile in time-frequency domain.

    Attributes:
        masking_threshold_db: Masking threshold in dB (STFT time-freq)
        signal_power_db: Signal power in dB (STFT time-freq)
        frequencies: Frequency bins (Hz)
        times: Time frames (seconds)
        masking_ratio_db: Signal-to-mask ratio in dB
    """

    masking_threshold_db: np.ndarray  # Shape: (freq_bins, time_frames)
    signal_power_db: np.ndarray  # Shape: (freq_bins, time_frames)
    frequencies: np.ndarray  # Shape: (freq_bins,)
    times: np.ndarray  # Shape: (time_frames,)
    masking_ratio_db: np.ndarray  # Shape: (freq_bins, time_frames)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns (freq_bins, time_frames)."""
        return self.masking_threshold_db.shape

    def get_audible_mask(self, threshold_db: float = 0.0) -> np.ndarray:
        """
        Get binary mask of audible components (signal > masking threshold).

        Args:
            threshold_db: Additional threshold in dB

        Returns:
            Boolean mask: True where signal is audible
        """
        return self.masking_ratio_db > threshold_db

    def get_masked_components_ratio(self) -> float:
        """
        Get ratio of masked components (inaudible).

        Returns:
            Ratio of masked components (0-1)
        """
        masked = np.sum(self.masking_ratio_db <= 0)
        total = self.masking_ratio_db.size
        return masked / total


@dataclass
class MaskingConfig:
    """
    Configuration for masking analysis.

    Attributes:
        enable_simultaneous: Enable simultaneous (frequency) masking
        enable_temporal: Enable temporal (pre/post) masking
        enable_spreading_function: Enable asymmetric spreading
        include_absolute_threshold: Add absolute threshold of hearing
        stft_nperseg: STFT window length
        stft_overlap: STFT overlap ratio (0-1)
        pre_masking_ms: Pre-masking duration in ms
        post_masking_ms: Post-masking duration in ms
    """

    enable_simultaneous: bool = True
    enable_temporal: bool = True
    enable_spreading_function: bool = True
    include_absolute_threshold: bool = True
    stft_nperseg: int = 2048
    stft_overlap: float = 0.5
    pre_masking_ms: float = PRE_MASKING_MS
    post_masking_ms: float = POST_MASKING_MS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASKING ANALYZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MaskingAnalyzer:
    """
    Psychoacoustic masking analyzer.

    Implements frequency and temporal masking models.

    Features:
    - Simultaneous masking with spreading function
    - Temporal pre/post masking
    - Absolute threshold of hearing
    - Signal-to-mask ratio computation
    - Audible component detection

    Example:
        >>> analyzer = MaskingAnalyzer()
        >>> profile = analyzer.analyze(audio, sr=48000)
        >>> masked_ratio = profile.get_masked_components_ratio()
        logger.debug("Masked components: %.1f%%", masked_ratio*100)
    """

    def __init__(self, config: MaskingConfig | None = None):
        """
        Initialize masking analyzer.

        Args:
            config: Masking configuration (uses defaults if None)
        """
        self.config = config or MaskingConfig()
        logger.debug(
            f"MaskingAnalyzer initialized (simultaneous={self.config.enable_simultaneous}, "
            f"temporal={self.config.enable_temporal})"
        )

    def analyze(self, audio: np.ndarray, sr: int) -> MaskingProfile:
        """
        Analyze audio and compute masking profile.

        Args:
            audio: Input audio (mono)
            sr: Sample rate

        Returns:
            MaskingProfile object
        """
        # Compute STFT
        nperseg = self.config.stft_nperseg
        noverlap = int(nperseg * self.config.stft_overlap)

        f, t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)

        # Convert to power (dB)
        power_linear = np.abs(Zxx) ** 2
        power_db = 10 * np.log10(power_linear + 1e-10)

        # Initialize masking threshold with absolute threshold
        if self.config.include_absolute_threshold:
            masking_threshold = self._compute_absolute_threshold(f)
            # Convert ATH from dB SPL to dBFS.
            # Standard assumption: 0 dBFS ≈ 94 dB SPL (ISO 1683, 1 Pa reference)
            # ATH values are in dB SPL, signal power is in dBFS.
            _SPL_TO_DBFS_OFFSET = 94.0
            masking_threshold = masking_threshold - _SPL_TO_DBFS_OFFSET
            # Broadcast to time dimension
            masking_threshold = masking_threshold[:, np.newaxis] + np.zeros((1, len(t)))
        else:
            masking_threshold = np.full_like(power_db, -80.0)  # Very low baseline

        # Simultaneous masking
        if self.config.enable_simultaneous:
            simultaneous_mask = self._compute_simultaneous_masking(power_db, f, sr)
            masking_threshold = np.maximum(masking_threshold, simultaneous_mask)

        # Temporal masking
        if self.config.enable_temporal:
            temporal_mask = self._compute_temporal_masking(power_db, t, sr)
            masking_threshold = np.maximum(masking_threshold, temporal_mask)

        # Signal-to-mask ratio
        masking_ratio_db = power_db - masking_threshold

        return MaskingProfile(
            masking_threshold_db=masking_threshold,
            signal_power_db=power_db,
            frequencies=f,
            times=t,
            masking_ratio_db=masking_ratio_db,
        )

    def _compute_absolute_threshold(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Compute absolute threshold of hearing for frequency bins.

        Args:
            frequencies: Frequency bins in Hz

        Returns:
            Threshold in dB SPL (shape: freq_bins)
        """
        # Interpolate ATH curve
        ath_freqs = np.array([f for f, _ in ABSOLUTE_THRESHOLD_HZ])
        ath_thresholds = np.array([t for _, t in ABSOLUTE_THRESHOLD_HZ])

        # Interpolate (log-space for frequency)
        threshold_db = np.interp(np.log10(frequencies + 1), np.log10(ath_freqs), ath_thresholds)  # +1 to avoid log(0)

        return threshold_db

    def _compute_simultaneous_masking(self, power_db: np.ndarray, frequencies: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute simultaneous (frequency) masking threshold.

        Uses simplified spreading function.

        Args:
            power_db: Signal power in dB (freq, time)
            frequencies: Frequency bins
            sr: Sample rate

        Returns:
            Masking threshold in dB (same shape as power_db)
        """
        # Simplified approach:
        # Masking threshold = Signal - 50dB (typical masking depth)
        # More sophisticated: Use actual spreading function

        if self.config.enable_spreading_function:
            # Proper roex-like spreading function (Moore & Glasberg 1997):
            # The spreading kernel must be a POSITIVE, NORMALIZED function
            # that represents how masking excitation spreads in frequency.
            # Asymmetric: upward masking spreads further (-10 dB/ERB)
            # than downward masking (-24 dB/ERB).
            #
            # We convert dB attenuation to linear spreading weights,
            # normalize, then convolve with linear power, and convert back.

            kernel_size = 15  # Frequency bins
            kernel = np.zeros(kernel_size)
            center = kernel_size // 2

            # Lower side (downward masking, steeper): -24 dB/ERB
            for i in range(center):
                atten_db = (center - i) * 3.0  # dB attenuation per bin
                kernel[i] = 10.0 ** (-atten_db / 10.0)

            # Center bin: full masking
            kernel[center] = 1.0

            # Upper side (upward masking, shallower): -10 dB/ERB
            for i in range(center + 1, kernel_size):
                atten_db = (i - center) * 1.5  # dB attenuation per bin
                kernel[i] = 10.0 ** (-atten_db / 10.0)

            # Normalize kernel so sum = 1 (energy-preserving convolution)
            kernel /= kernel.sum()

            # Convolve in LINEAR domain, convert back to dB, apply masking depth.
            # Masking depth: a component must exceed the local excitation by
            # this many dB to be perceived as audible. The spread includes
            # self-masking (the component's own energy), so without depth
            # offset masking_ratio ≈ 0 everywhere.
            # Standard values: tone-masking-tone ~14.5 dB, noise-masking-tone
            # ~5.5 dB.  For general music (mix of tonal and noise): ~12 dB.
            _MASKING_DEPTH_DB = 12.0

            masking_threshold = np.zeros_like(power_db)
            for t_idx in range(power_db.shape[1]):
                power_lin = 10.0 ** (power_db[:, t_idx] / 10.0)
                spread_lin = np.convolve(power_lin, kernel, mode="same")
                masking_threshold[:, t_idx] = 10.0 * np.log10(spread_lin + 1e-30) - _MASKING_DEPTH_DB
        else:
            # Simple: -50dB below signal
            masking_threshold = power_db - 50.0

        return masking_threshold

    def _compute_temporal_masking(self, power_db: np.ndarray, times: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute temporal (pre/post) masking threshold.

        Args:
            power_db: Signal power in dB (freq, time)
            times: Time frames
            sr: Sample rate

        Returns:
            Masking threshold in dB (same shape as power_db)
        """
        # Pre-Masking: 20ms backward
        # Post-Masking: 200ms forward

        # Convert ms to frames
        time_step = times[1] - times[0] if len(times) > 1 else 0.01
        pre_frames = max(1, int(self.config.pre_masking_ms / 1000 / time_step))
        post_frames = max(1, int(self.config.post_masking_ms / 1000 / time_step))

        # Apply temporal masking (maximum filter in time)
        # This spreads high energy regions forward and backward in time

        masking_threshold = np.zeros_like(power_db)

        # Apply separately per frequency
        for f_idx in range(power_db.shape[0]):
            signal_row = power_db[f_idx, :]

            # Forward masking (post)
            forward_mask = maximum_filter1d(signal_row, size=post_frames, mode="constant", cval=-80)
            # Decay: -10 dB per 50ms
            decay_rate = 10.0 / (50 / 1000 / time_step)  # dB per frame
            for i in range(len(forward_mask)):
                offset = min(i, post_frames)
                forward_mask[i] -= offset * decay_rate

            # Backward masking (pre)
            backward_mask = maximum_filter1d(signal_row[::-1], size=pre_frames, mode="constant", cval=-80)[::-1]
            # Steeper decay for pre-masking
            decay_rate_pre = 20.0 / (10 / 1000 / time_step)  # dB per frame
            for i in range(len(backward_mask)):
                # Look backward
                offset = min(len(backward_mask) - 1 - i, pre_frames)
                backward_mask[i] -= offset * decay_rate_pre

            # Combine (maximum)
            masking_threshold[f_idx, :] = np.maximum(forward_mask, backward_mask) - 30.0  # -30dB typical

        return masking_threshold

    def apply_masking(
        self, audio: np.ndarray, sr: int, profile: MaskingProfile | None = None, threshold_db: float = 0.0
    ) -> np.ndarray:
        """
        Remove masked (inaudible) components from audio.

        Args:
            audio: Input audio
            sr: Sample rate
            profile: Pre-computed masking profile (computes if None)
            threshold_db: Additional threshold in dB

        Returns:
            Audio with masked components removed
        """
        # Compute profile if not provided
        if profile is None:
            profile = self.analyze(audio, sr)

        # STFT
        nperseg = self.config.stft_nperseg
        noverlap = int(nperseg * self.config.stft_overlap)
        _f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)

        # Get audible mask
        audible = profile.get_audible_mask(threshold_db)

        # Zero out masked components
        Zxx_filtered = Zxx * audible

        # ISTFT
        _, audio_filtered = istft(Zxx_filtered, sr, nperseg=nperseg, noverlap=noverlap)

        # Handle potential NaN/Inf
        if np.any(~np.isfinite(audio_filtered)):
            logger.warning("NaN/Inf detected after masking, returning original audio")
            return audio

        # Match length
        if len(audio_filtered) < len(audio):
            audio_filtered = np.pad(audio_filtered, (0, len(audio) - len(audio_filtered)))
        elif len(audio_filtered) > len(audio):
            audio_filtered = audio_filtered[: len(audio)]

        return audio_filtered

    def compute_smr(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute Signal-to-Mask Ratio (SMR) for audio.

        SMR indicates how much of the signal is audible above masking.
        Higher SMR = more audible content.

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Average SMR in dB
        """
        profile = self.analyze(audio, sr)

        # Average SMR (where signal > threshold)
        audible = profile.masking_ratio_db > 0
        num_audible = np.sum(audible)

        if num_audible == 0:
            # All masked - return minimum SMR
            # Use median instead of 0 to avoid degenerate case
            med_smr = np.median(profile.masking_ratio_db)
            return float(med_smr) if np.isfinite(med_smr) else -40.0

        # Average SMR over audible components
        avg_smr = np.mean(profile.masking_ratio_db[audible])
        return float(avg_smr) if np.isfinite(avg_smr) else 20.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_masking(audio: np.ndarray, sr: int, enable_temporal: bool = True) -> MaskingProfile:
    """
    Quick masking analysis.

    Args:
        audio: Input audio (mono)
        sr: Sample rate
        enable_temporal: Enable temporal masking

    Returns:
        MaskingProfile object
    """
    config = MaskingConfig(enable_temporal=enable_temporal)
    analyzer = MaskingAnalyzer(config)
    return analyzer.analyze(audio, sr)


def compute_smr(audio: np.ndarray, sr: int) -> float:
    """
    Quick Signal-to-Mask Ratio computation.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Average SMR in dB
    """
    analyzer = MaskingAnalyzer()
    return analyzer.compute_smr(audio, sr)


if __name__ == "__main__":
    """Demo masking analyzer"""
    logger.debug("\n" + "=" * 70)
    logger.debug("MASKING ANALYZER - Demo")
    logger.debug("=" * 70 + "\n")

    # Generate test signal
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Signal: Strong 1 kHz tone + weak 1.2 kHz tone (should be masked)
    audio = 1.0 * np.sin(2 * np.pi * 1000 * t) + 0.05 * np.sin(  # Strong masker
        2 * np.pi * 1200 * t
    )  # Weak tone (masked)
    audio = audio / np.abs(audio).max()

    # Analyze
    analyzer = MaskingAnalyzer()
    profile = analyzer.analyze(audio, sr)

    logger.debug("Masking Profile Analysis:")
    logger.debug("  STFT Shape: %s freq bins × %s time frames", profile.shape[0], profile.shape[1])
    logger.debug("  Frequency Range: %.0f - %.0f Hz", profile.frequencies[0], profile.frequencies[-1])
    logger.debug("  Time Range: %.3f - %.3f s", profile.times[0], profile.times[-1])

    masked_ratio = profile.get_masked_components_ratio()
    logger.debug("\n  Masked Components: %.1f%%", masked_ratio * 100)

    avg_smr = analyzer.compute_smr(audio, sr)
    logger.debug("  Average SMR: %.1f dB", avg_smr)

    # Check if 1.2 kHz is masked
    freq_1200_idx = np.argmin(np.abs(profile.frequencies - 1200))
    smr_at_1200 = np.mean(profile.masking_ratio_db[freq_1200_idx, :])
    logger.debug("\n  SMR at 1.2 kHz: %.1f dB", smr_at_1200)
    if smr_at_1200 < 0:
        logger.debug("  → 1.2 kHz tone is MASKED (inaudible)")
    else:
        logger.debug("  → 1.2 kHz tone is AUDIBLE")

    # Apply masking (remove inaudible components)
    logger.debug("\n  Applying masking filter...")
    audio_filtered = analyzer.apply_masking(audio, sr, profile)

    logger.debug("    Original RMS: %.4f", np.sqrt(np.mean(audio**2)))
    logger.debug("    Filtered RMS: %.4f", np.sqrt(np.mean(audio_filtered**2)))

    logger.debug("\n" + "=" * 70)
    logger.debug("Demo complete!")
    logger.debug("=" * 70 + "\n")
