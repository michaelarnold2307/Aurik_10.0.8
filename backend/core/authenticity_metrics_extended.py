"""
authenticity_metrics_extended.py - Genre-Specific Authenticity Detectors

Detektiert und erhält genrespezifische Performance-Elemente:
- Finger Noise (Acoustic Guitar)
- Bow Noise (Violin/Cello)
- Pedal Noise (Piano)
- Brush Texture (Jazz Drums)
- Vinyl Character (Warmth vs Defects)

Author: AURIK Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional
import warnings

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# FINGER NOISE DETECTOR - ACOUSTIC GUITAR
# =============================================================================


class FingerNoiseDetector:
    """
    Detects and measures finger noise on acoustic guitar strings.

    Finger noise: Sliding/squeaking sounds from fingers moving on strings.
    - Frequency range: 2-6 kHz (harmonically rich)
    - Duration: 50-200ms (short sweeps)
    - Critical for: Jazz, Fingerstyle, Folk, Classical Guitar

    Target: >85% retention (preserve authenticity)

    References:
    - Barthet, M., et al. (2010). "Expressive Audio Transformations"
    - Grachten, M., et al. (2012). "Guitar Performance Analysis"
    """

    def __init__(self, sensitivity: float = 0.7, freq_range: tuple[float, float] = (2000.0, 6000.0)):
        """
        Initialize Finger Noise Detector.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0-1, default: 0.7)
        freq_range : tuple
            Frequency range for detection (Hz, default: 2-6 kHz)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.freq_range = freq_range

        self.metrics = {}

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Detect finger noise events in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Detection metrics
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Bandpass filter for finger noise frequency range
        sos = signal.butter(4, [self.freq_range[0], self.freq_range[1]], "bandpass", fs=sample_rate, output="sos")
        filtered = np.nan_to_num(signal.sosfilt(sos, audio), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute envelope
        envelope = np.abs(signal.hilbert(filtered))

        # Smooth envelope
        window_ms = 10
        window_samples = int(window_ms * sample_rate / 1000)
        envelope_smooth = signal.convolve(envelope, np.ones(window_samples) / window_samples, mode="same")

        # Detect transients (finger slides)
        threshold = np.percentile(envelope_smooth, 95) * self.sensitivity
        events = envelope_smooth > threshold

        # Count events and measure total duration
        event_starts = np.where(np.diff(events.astype(int)) == 1)[0]
        event_ends = np.where(np.diff(events.astype(int)) == -1)[0]

        # Match starts with ends
        n_events = min(len(event_starts), len(event_ends))
        if n_events > 0:
            event_durations = event_ends[:n_events] - event_starts[:n_events]
            total_duration_ms = np.sum(event_durations) / sample_rate * 1000
            avg_duration_ms = np.mean(event_durations) / sample_rate * 1000
        else:
            total_duration_ms = 0.0
            avg_duration_ms = 0.0

        # Compute finger noise energy ratio
        finger_noise_energy = np.sum(filtered**2)
        total_energy = np.sum(audio**2)
        energy_ratio = finger_noise_energy / (total_energy + 1e-8)

        self.metrics = {
            "finger_noise_detected": n_events > 0,
            "num_events": n_events,
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": avg_duration_ms,
            "energy_ratio": energy_ratio,
            "retention_target": 0.85,
        }

        return self.metrics


# =============================================================================
# BOW NOISE DETECTOR - VIOLIN/CELLO
# =============================================================================


class BowNoiseDetector:
    """
    Detects and measures bow noise on string instruments.

    Bow noise: Rosin/bow scraping, raspy texture from bow-string interaction.
    - Frequency range: Broadband 1-8 kHz (textural noise)
    - Character: Continuous, not transient
    - Critical for: Classical violin, folk fiddle, cello

    Target: >80% retention (authentic string performance)

    References:
    - Schoonderwaldt, E., & Demoucron, M. (2009). "Extraction of Bowing Parameters"
    - Maestre, E., et al. (2012). "Statistical Modeling of Bowing Control"
    """

    def __init__(self, sensitivity: float = 0.6, freq_range: tuple[float, float] = (1000.0, 8000.0)):
        """
        Initialize Bow Noise Detector.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0-1, default: 0.6)
        freq_range : tuple
            Frequency range for detection (Hz, default: 1-8 kHz)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.freq_range = freq_range

        self.metrics = {}

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Detect bow noise in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Detection metrics
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Highpass filter to isolate bow noise
        sos = signal.butter(4, self.freq_range[0], "highpass", fs=sample_rate, output="sos")
        filtered = np.nan_to_num(signal.sosfilt(sos, audio), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute spectral flatness (noise-like vs tonal)
        # Bow noise is broadband → high flatness
        nperseg = 2048
        f, t, Zxx = signal.stft(filtered, sample_rate, nperseg=nperseg)

        # Spectral flatness per frame
        mag = np.abs(Zxx) + 1e-8
        geometric_mean = np.exp(np.mean(np.log(mag), axis=0))
        arithmetic_mean = np.mean(mag, axis=0)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)

        # Bow noise has high flatness (>0.3)
        bow_noise_frames = spectral_flatness > (0.3 * self.sensitivity)
        bow_noise_ratio = np.mean(bow_noise_frames)

        # Compute bow noise energy
        bow_noise_energy = np.sum(filtered**2)
        total_energy = np.sum(audio**2)
        energy_ratio = bow_noise_energy / (total_energy + 1e-8)

        self.metrics = {
            "bow_noise_detected": bow_noise_ratio > 0.1,
            "bow_noise_ratio": float(bow_noise_ratio),
            "energy_ratio": float(energy_ratio),
            "spectral_flatness_mean": float(np.mean(spectral_flatness)),
            "retention_target": 0.80,
        }

        return self.metrics


# =============================================================================
# PEDAL NOISE DETECTOR - PIANO
# =============================================================================


class PedalNoiseDetector:
    """
    Detects and measures piano pedal mechanical sounds.

    Pedal noise: Sustain pedal clicks/thuds from mechanical operation.
    - Frequency range: 80-400 Hz (low, dull clicks)
    - Duration: Short impulses (10-50ms)
    - Critical for: Classical piano, jazz piano, singer-songwriter

    Target: >80% retention (natural piano performance)

    References:
    - Repp, B. H. (1997). "Expressive Timing in Piano Performance"
    - Goebl, W., & Bresin, R. (2003). "Measurement and Reproduction of Piano Pedaling"
    """

    def __init__(self, sensitivity: float = 0.7, freq_range: tuple[float, float] = (80.0, 400.0)):
        """
        Initialize Pedal Noise Detector.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0-1, default: 0.7)
        freq_range : tuple
            Frequency range for detection (Hz, default: 80-400 Hz)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.freq_range = freq_range

        self.metrics = {}

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Detect pedal noise events in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Detection metrics
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Bandpass filter for pedal noise frequency range
        sos = signal.butter(4, [self.freq_range[0], self.freq_range[1]], "bandpass", fs=sample_rate, output="sos")
        filtered = np.nan_to_num(signal.sosfilt(sos, audio), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute envelope
        envelope = np.abs(signal.hilbert(filtered))

        # Detect short impulses (onset detection)
        onset_envelope = np.diff(envelope, prepend=envelope[0])
        onset_envelope = np.maximum(onset_envelope, 0)  # Only positive changes

        # Threshold for pedal click detection
        threshold = np.percentile(onset_envelope, 99) * self.sensitivity

        # Find peaks (pedal clicks)
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            onset_envelope, height=threshold, distance=int(0.1 * sample_rate)  # Min 100ms between clicks
        )

        n_events = len(peaks)

        # Compute pedal noise energy
        pedal_energy = np.sum(filtered**2)
        total_energy = np.sum(audio**2)
        energy_ratio = pedal_energy / (total_energy + 1e-8)

        self.metrics = {
            "pedal_noise_detected": n_events > 0,
            "num_events": n_events,
            "energy_ratio": float(energy_ratio),
            "retention_target": 0.80,
        }

        return self.metrics


# =============================================================================
# BRUSH TEXTURE DETECTOR - JAZZ DRUMS
# =============================================================================


class BrushTextureDetector:
    """
    Detects and measures brush sweeps/texture on drums.

    Brush texture: Continuous sweeping/swishing sounds from brushes on snare.
    - Frequency range: 3-10 kHz (high-frequency texture)
    - Character: Continuous, sand-like texture
    - Critical for: Jazz drums, ballads, intimate settings

    Target: >85% retention (essential jazz character)

    References:
    - Dahl, S., & Altenmüller, E. (2008). "Finger Forces in Drumming"
    - Rossing, T. D. (2000). "Science of Percussion Instruments"
    """

    def __init__(self, sensitivity: float = 0.65, freq_range: tuple[float, float] = (3000.0, 10000.0)):
        """
        Initialize Brush Texture Detector.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0-1, default: 0.65)
        freq_range : tuple
            Frequency range for detection (Hz, default: 3-10 kHz)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.freq_range = freq_range

        self.metrics = {}

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Detect brush texture in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Detection metrics
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Bandpass filter for brush texture frequency range
        sos = signal.butter(4, [self.freq_range[0], self.freq_range[1]], "bandpass", fs=sample_rate, output="sos")
        filtered = np.nan_to_num(signal.sosfilt(sos, audio), nan=0.0, posinf=0.0, neginf=0.0)

        # Compute RMS envelope (continuous energy)
        window_ms = 50  # Brushes have longer temporal structure
        window_samples = int(window_ms * sample_rate / 1000)

        rms_envelope = np.sqrt(signal.convolve(filtered**2, np.ones(window_samples) / window_samples, mode="same"))

        # Brush texture is continuous, not impulsive
        # Measure temporal continuity
        threshold = np.percentile(rms_envelope, 70) * self.sensitivity
        active_frames = rms_envelope > threshold

        # Compute runs of active frames (continuous texture)
        from scipy.ndimage import label

        labeled, num_regions = label(active_frames)

        if num_regions > 0:
            region_lengths = [np.sum(labeled == i) for i in range(1, num_regions + 1)]
            avg_region_length_ms = np.mean(region_lengths) / sample_rate * 1000
            total_active_ms = np.sum(active_frames) / sample_rate * 1000
        else:
            avg_region_length_ms = 0.0
            total_active_ms = 0.0

        # Compute brush texture energy
        brush_energy = np.sum(filtered**2)
        total_energy = np.sum(audio**2)
        energy_ratio = brush_energy / (total_energy + 1e-8)

        self.metrics = {
            "brush_texture_detected": num_regions > 0 and avg_region_length_ms > 100,
            "num_regions": num_regions,
            "avg_region_length_ms": float(avg_region_length_ms),
            "total_active_ms": float(total_active_ms),
            "energy_ratio": float(energy_ratio),
            "retention_target": 0.85,
        }

        return self.metrics


# =============================================================================
# VINYL CHARACTER DETECTOR - WARMTH VS DEFECTS
# =============================================================================


class VinylCharacterDetector:
    """
    Distinguishes vinyl warmth (harmonic saturation) from defects (noise).

    Vinyl character: Desired "warmth" from tape/vinyl saturation vs unwanted noise.
    - Warmth: Harmonic distortion (2nd/3rd harmonics, <5%)
    - Defects: Broadband noise, crackle, pops
    - Critical for: Vintage remastering, lo-fi production

    Target: Preserve warmth (>90%), remove defects

    References:
    - Katz, B. (2002). "Mastering Audio: The Art and the Science"
    - Zölzer, U. (2011). "DAFX: Digital Audio Effects"
    """

    def __init__(self, sensitivity: float = 0.75, thd_threshold: float = 0.05):  # 5% THD
        """
        Initialize Vinyl Character Detector.

        Parameters
        ----------
        sensitivity : float
            Detection sensitivity (0-1, default: 0.75)
        thd_threshold : float
            Maximum THD for "warmth" (default: 5%)
        """
        self.sensitivity = np.clip(sensitivity, 0.0, 1.0)
        self.thd_threshold = thd_threshold

        self.metrics = {}

    def detect(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Detect vinyl character (warmth vs defects) in audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Detection metrics
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Compute spectrum
        N = len(audio)
        fft_audio = fft(audio)
        fft_audio = np.nan_to_num(fft_audio, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = fftfreq(N, 1 / sample_rate)

        # Analyze only positive frequencies
        positive_freqs = freqs[: N // 2]
        magnitude = np.abs(fft_audio[: N // 2])

        # Find fundamental (peak in 80-400 Hz range, typical for music)
        low_freq_mask = (positive_freqs >= 80) & (positive_freqs <= 400)
        if np.any(low_freq_mask):
            fundamental_idx = np.argmax(magnitude[low_freq_mask])
            fundamental_freq = positive_freqs[low_freq_mask][fundamental_idx]
            fundamental_mag = magnitude[low_freq_mask][fundamental_idx]

            # Find 2nd and 3rd harmonics
            harmonic_2_freq = fundamental_freq * 2
            harmonic_3_freq = fundamental_freq * 3

            # Search in ±20 Hz window
            def find_harmonic_magnitude(target_freq):
                mask = (positive_freqs >= target_freq - 20) & (positive_freqs <= target_freq + 20)
                if np.any(mask):
                    return np.max(magnitude[mask])
                return 0.0

            harmonic_2_mag = find_harmonic_magnitude(harmonic_2_freq)
            harmonic_3_mag = find_harmonic_magnitude(harmonic_3_freq)

            # Compute THD (simplified: 2nd + 3rd harmonics only)
            if fundamental_mag > 1e-8:
                thd = np.sqrt(harmonic_2_mag**2 + harmonic_3_mag**2) / fundamental_mag
            else:
                thd = 0.0
        else:
            thd = 0.0

        # Compute noise floor (high-frequency energy >10 kHz)
        noise_mask = positive_freqs > 10000
        if np.any(noise_mask):
            noise_energy = np.sum(magnitude[noise_mask] ** 2)
            total_energy = np.sum(magnitude**2)
            noise_ratio = noise_energy / (total_energy + 1e-8)
        else:
            noise_ratio = 0.0

        # Classify
        has_warmth = 0.01 < thd < self.thd_threshold  # Warmth: 1-5% THD
        has_defects = noise_ratio > 0.02  # Defects: >2% HF noise

        self.metrics = {
            "vinyl_character_detected": has_warmth or has_defects,
            "warmth_detected": has_warmth,
            "defects_detected": has_defects,
            "thd": float(thd),
            "noise_ratio": float(noise_ratio),
            "warmth_retention_target": 0.90,
            "defects_removal_target": 0.90,
        }

        return self.metrics


# =============================================================================
# UNIFIED API - AUTHENTICITY METRICS EXTENDED
# =============================================================================


class AuthenticityMetricsExtended:
    """
    Unified API for genre-specific authenticity detection.

    Combines all 5 detectors for comprehensive analysis:
    - FingerNoiseDetector
    - BowNoiseDetector
    - PedalNoiseDetector
    - BrushTextureDetector
    - VinylCharacterDetector
    """

    def __init__(self):
        """Initialize all detectors with default parameters."""
        self.finger_noise_detector = FingerNoiseDetector()
        self.bow_noise_detector = BowNoiseDetector()
        self.pedal_noise_detector = PedalNoiseDetector()
        self.brush_texture_detector = BrushTextureDetector()
        self.vinyl_character_detector = VinylCharacterDetector()

    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Analyze audio for all genre-specific authenticity metrics.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sample_rate : int
            Sample rate in Hz

        Returns
        -------
        metrics : dict
            Combined metrics from all detectors
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        # Handle stereo (use left channel for analysis)
        if audio.ndim == 2:
            audio_mono = audio[0]
        else:
            audio_mono = audio
        audio_mono = np.nan_to_num(audio_mono, nan=0.0, posinf=0.0, neginf=0.0)

        logger.debug("AuthenticityMetricsExtended: Analysiere genre-spezifische Authentizitätselemente")

        # Run all detectors
        finger_noise = self.finger_noise_detector.detect(audio_mono, sample_rate)
        bow_noise = self.bow_noise_detector.detect(audio_mono, sample_rate)
        pedal_noise = self.pedal_noise_detector.detect(audio_mono, sample_rate)
        brush_texture = self.brush_texture_detector.detect(audio_mono, sample_rate)
        vinyl_character = self.vinyl_character_detector.detect(audio_mono, sample_rate)

        # Summary
        detected_elements = []
        if finger_noise["finger_noise_detected"]:
            detected_elements.append("Finger Noise (Guitar)")
        if bow_noise["bow_noise_detected"]:
            detected_elements.append("Bow Noise (Strings)")
        if pedal_noise["pedal_noise_detected"]:
            detected_elements.append("Pedal Noise (Piano)")
        if brush_texture["brush_texture_detected"]:
            detected_elements.append("Brush Texture (Jazz Drums)")
        if vinyl_character["warmth_detected"]:
            detected_elements.append("Vinyl Warmth")
        if vinyl_character["defects_detected"]:
            detected_elements.append("Vinyl Defects")

        logger.debug(
            "AuthenticityMetricsExtended: Erkannte Elemente: %s",
            ", ".join(detected_elements) if detected_elements else "Keine",
        )

        return {
            "finger_noise": finger_noise,
            "bow_noise": bow_noise,
            "pedal_noise": pedal_noise,
            "brush_texture": brush_texture,
            "vinyl_character": vinyl_character,
            "detected_elements": detected_elements,
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Genre-Specific Authenticity Detector")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument(
        "--detector", choices=["finger", "bow", "pedal", "brush", "vinyl", "all"], default="all", help="Detector to use"
    )

    args = parser.parse_args()

    # Load audio
    logger.debug(f"Loading: {args.input}")
    audio, sr = sf.read(args.input, always_2d=False)

    # Transpose if stereo
    if audio.ndim == 2:
        audio = audio.T

    # Analyze
    if args.detector == "all":
        analyzer = AuthenticityMetricsExtended()
        metrics = analyzer.analyze(audio, sr)

        logger.debug(f"\n{'='*60}")
        logger.debug("GENRE-SPECIFIC AUTHENTICITY METRICS:")
        logger.debug(f"{'='*60}")
        for key, value in metrics.items():
            if key != "detected_elements":
                logger.debug(f"\n{key.upper()}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        logger.debug(f"  {k}: {v:.4f}")
                    else:
                        logger.debug(f"  {k}: {v}")
        logger.debug(f"{'='*60}\n")
    else:
        # Single detector
        if args.detector == "finger":
            detector = FingerNoiseDetector()
        elif args.detector == "bow":
            detector = BowNoiseDetector()
        elif args.detector == "pedal":
            detector = PedalNoiseDetector()
        elif args.detector == "brush":
            detector = BrushTextureDetector()
        elif args.detector == "vinyl":
            detector = VinylCharacterDetector()

        audio_mono = audio[0] if audio.ndim == 2 else audio
        metrics = detector.detect(audio_mono, sr)

        logger.debug(f"\n{'='*60}")
        logger.debug(f"{args.detector.upper()} DETECTOR METRICS:")
        logger.debug(f"{'='*60}")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.debug(f"{k}: {v:.4f}")
            else:
                logger.debug(f"{k}: {v}")
        logger.debug(f"{'='*60}\n")
