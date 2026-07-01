"""
Psychoacoustic Metrics Module - Aurik 9.0
==========================================

Objective audio quality metrics for validation and optimization.

Metrics:
- PESQ: Perceptual Evaluation of Speech Quality
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio
- Spectral Distortion: Log-Spectral Distance
- Roughness (Zwicker): Psychoacoustic roughness
- Sharpness (Aures): High-frequency emphasis
- Naturalness Score: Custom composite metric

Scientific Foundation:
- ITU-T P.862: PESQ standard
- Roux et al. (2019): SI-SDR for source separation
- Zwicker & Fastl (1999): Psychoacoustics of Hearing
- Aures (1985): Berechnungsverfahren für den sensorischen Wohlklang

Author: Aurik 9.0 Development Team
Date: 15. Februar 2026
"""

import logging

import numpy as np
from scipy import signal
from scipy.signal import hilbert, stft, welch

logger = logging.getLogger(__name__)


class PsychoAcousticMetrics:
    """
    Objective audio quality metrics for naturalness assessment.

    All methods return normalized scores (0-1 where higher = better)
    unless otherwise specified.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    # calculate_sisdr entfernt — explizit verboten §4.4+§10.2 (SI-SDR Sprach-Metrik,
    # auf Sprachkorpora trainiert/definiert; für Musik systematisch ungeeignet)

    def calculate_spectral_distortion(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """
        Log-Spectral Distortion (LSD) in dB.

        Measures spectral difference between reference and degraded.
        Lower = better (less spectral distortion).

        Args:
            reference: Clean reference signal
            degraded: Degraded/processed signal

        Returns:
            LSD in dB (typically 0-10 dB, lower = better)
        """
        # Convert stereo to mono — handle both (channels, samples) and (samples, channels)
        if reference.ndim > 1:
            # Use axis=0 if channels-first (shape[0] <= 2), else axis=-1
            reference = reference.mean(axis=0) if reference.shape[0] <= 2 else reference.mean(axis=-1)
        if degraded.ndim > 1:
            degraded = degraded.mean(axis=0) if degraded.shape[0] <= 2 else degraded.mean(axis=-1)
        # Flatten to 1D guard
        reference = np.asarray(reference, dtype=np.float32).ravel()
        degraded = np.asarray(degraded, dtype=np.float32).ravel()
        if len(reference) < 4 or len(degraded) < 4:
            return 5.0  # Cannot compute LSD on very short signals

        # Compute STFTs
        _, _, ref_stft = stft(reference, fs=self.sample_rate, nperseg=min(2048, len(reference)))
        _, _, deg_stft = stft(degraded, fs=self.sample_rate, nperseg=min(2048, len(degraded)))

        # Magnitude spectra
        ref_mag = np.abs(ref_stft)
        deg_mag = np.abs(deg_stft)

        # Guard: both must be 2D (freq × time) for shape[1] access
        if ref_mag.ndim < 2 or deg_mag.ndim < 2:
            return 5.0

        # Ensure same shape (handles different freq bins from different nperseg)
        min_shape = (
            min(ref_mag.shape[0], deg_mag.shape[0]),
            min(ref_mag.shape[1], deg_mag.shape[1]),
        )
        ref_mag = ref_mag[: min_shape[0], : min_shape[1]]
        deg_mag = deg_mag[: min_shape[0], : min_shape[1]]

        # Log-spectral distance
        lsd = np.sqrt(np.mean((20 * np.log10((ref_mag + 1e-10) / (deg_mag + 1e-10))) ** 2))
        # NaN/Inf-Guard (§3.1)
        lsd = np.nan_to_num(lsd, nan=5.0, posinf=20.0, neginf=0.0)
        return float(lsd)

    def calculate_roughness(self, audio: np.ndarray) -> float:
        """
        Psychoacoustic Roughness (simplified Zwicker model).

        Based on amplitude modulation in critical bands.
        Lower = smoother, more natural.

        Args:
            audio: Audio signal

        Returns:
            Roughness score (0-1, normalized, lower = better)
        """
        # Hilbert envelope
        analytic: np.ndarray = hilbert(np.asarray(audio, dtype=np.float64))  # type: ignore[call-overload]
        envelope = np.abs(analytic)

        # Envelope modulation (derivative)
        envelope_diff = np.abs(np.diff(envelope))

        # Roughness is proportional to envelope modulation
        # Zwicker: Roughness peaks at 70 Hz modulation
        # We use simplified metric: mean envelope variation
        roughness = np.mean(envelope_diff) / (np.mean(envelope) + 1e-10)

        # Normalize to 0-1 (lower = better)
        # Typical values: 0.01-0.1 for natural audio
        roughness_normalized = min(1.0, roughness * 10)
        # NaN/Inf-Guard (§3.1)
        roughness_normalized = np.nan_to_num(roughness_normalized, nan=0.5, posinf=1.0, neginf=0.0)
        return float(roughness_normalized)

    def calculate_sharpness(self, audio: np.ndarray) -> float:
        """
        Psychoacoustic Sharpness (Aures model).

        Measures high-frequency emphasis.
        Higher = more high-frequency content.

        Args:
            audio: Audio signal

        Returns:
            Sharpness score (0-1, normalized)
        """
        # Power spectral density
        _nperseg_sharpness = min(2048, max(1, len(audio)))
        f, psd = welch(audio, fs=self.sample_rate, nperseg=_nperseg_sharpness)

        # Aures weighting: higher frequencies weighted more
        weights = (f / 1000 + 1e-10) ** 1.5
        weighted_psd = psd * weights

        # Sharpness: ratio of weighted to unweighted energy
        sharpness = np.sum(weighted_psd) / (np.sum(psd) + 1e-10)

        # Normalize to 0-1
        # Typical values: 0.5-3.0 for natural audio
        sharpness_normalized = min(1.0, sharpness / 3.0)
        # NaN/Inf-Guard (§3.1)
        sharpness_normalized = np.nan_to_num(sharpness_normalized, nan=0.5, posinf=1.0, neginf=0.0)
        return float(sharpness_normalized)

    def calculate_spectral_flatness(self, audio: np.ndarray) -> float:
        """
        Spectral Flatness (Wiener Entropy).

        Ratio of geometric mean to arithmetic mean of spectrum.
        Higher = more noise-like (flat spectrum).
        Lower = more tonal (peaked spectrum).

        Args:
            audio: Audio signal

        Returns:
            Spectral flatness (0-1)
        """
        # Periodogram
        _nperseg_flatness = min(2048, max(1, len(audio)))
        _f, psd = welch(audio, fs=self.sample_rate, nperseg=_nperseg_flatness)

        # Geometric and arithmetic means
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)

        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        # NaN/Inf-Guard (§3.1)
        flatness = np.nan_to_num(flatness, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(flatness, 0, 1))

    def calculate_temporal_smoothness(self, audio: np.ndarray) -> float:
        """
        Temporal Smoothness (absence of abrupt changes).

        Measures presence of clicks, pops, artifacts.
        Higher = smoother, fewer artifacts.

        Args:
            audio: Audio signal

        Returns:
            Smoothness score (0-1, higher = better)
        """
        # First derivative (sample-to-sample change)
        diff = np.abs(np.diff(audio))

        # Count high-energy transients (outliers)
        threshold = np.percentile(diff, 99.5)
        artifacts: int = int(np.sum(diff > threshold))

        # Normalize: fewer artifacts = higher score
        max_expected_artifacts = len(audio) * 0.001  # 0.1% threshold
        smoothness = 1.0 - min(1.0, artifacts / max_expected_artifacts)
        # NaN/Inf-Guard (§3.1)
        smoothness = np.nan_to_num(smoothness, nan=0.8, posinf=1.0, neginf=0.0)
        return float(smoothness)

    def calculate_harmonic_coherence(self, audio: np.ndarray) -> float:
        """
        Harmonic Coherence (preservation of harmonic structure).

        Measures if harmonic relationships are intact.
        Higher = better harmonic structure.

        Args:
            audio: Audio signal

        Returns:
            Coherence score (0-1, higher = better)
        """
        # STFT
        _nperseg_harm = min(2048, max(1, len(audio)))
        _f, _t, Zxx = stft(audio, fs=self.sample_rate, nperseg=_nperseg_harm)
        magnitude = np.abs(Zxx)

        # Find spectral peaks (potential harmonics)
        spectral_peaks = np.sum(magnitude > np.percentile(magnitude, 95), axis=0)

        # Measure temporal consistency of peak count
        # Good harmonic structure = consistent peak count over time
        if len(spectral_peaks) < 2:
            return 0.5  # Not enough data

        peak_variance = np.std(spectral_peaks) / (np.mean(spectral_peaks) + 1e-10)

        # Lower variance = better coherence
        coherence = 1.0 / (1.0 + peak_variance * 0.5)
        # NaN/Inf-Guard (§3.1)
        coherence = np.nan_to_num(coherence, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(coherence, 0, 1))

    def calculate_noise_floor_consistency(self, audio: np.ndarray) -> float:
        """
        Noise Floor Consistency (stable background noise).

        Measures if noise floor is consistent (not modulated).
        Higher = more consistent.

        Args:
            audio: Audio signal

        Returns:
            Consistency score (0-1, higher = better)
        """
        # RMS envelope
        window_samples = int(0.05 * self.sample_rate)  # 50ms
        if window_samples % 2 == 0:
            window_samples += 1

        rms = np.sqrt(signal.convolve(audio**2, np.ones(window_samples) / window_samples, mode="same"))

        # Find quiet passages (below -40 dB)
        threshold = 0.01  # -40 dB
        quiet_passages = rms < threshold

        if np.sum(quiet_passages) < 100:
            return 0.5  # Not enough quiet passages

        quiet_rms = rms[quiet_passages]

        # Low variance in quiet RMS = consistent noise floor
        variance = np.std(quiet_rms) / (np.mean(quiet_rms) + 1e-10)
        consistency = 1.0 - min(1.0, variance * 10)  # type: ignore[operator]
        # NaN/Inf-Guard (§3.1)
        consistency = np.nan_to_num(consistency, nan=0.5, posinf=1.0, neginf=0.0)
        return float(consistency)

    def calculate_naturalness_score(self, audio: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float]:
        """
        Comprehensive Naturalness Score (composite metric).

        Combines multiple psychoacoustic metrics weighted for "naturalness".

        Args:
            audio: Processed audio signal
            reference: Optional clean reference (for comparative metrics)

        Returns:
            Dictionary with individual metrics and overall naturalness score
        """
        scores = {}

        # 1. Spectral Flatness (0.3 weight)
        scores["spectral_flatness"] = self.calculate_spectral_flatness(audio)

        # 2. Temporal Smoothness (0.3 weight)
        scores["temporal_smoothness"] = self.calculate_temporal_smoothness(audio)

        # 3. Harmonic Coherence (0.25 weight)
        scores["harmonic_coherence"] = self.calculate_harmonic_coherence(audio)

        # 4. Noise Floor Consistency (0.15 weight)
        scores["noise_floor_consistency"] = self.calculate_noise_floor_consistency(audio)

        # If reference available, add comparative metrics
        if reference is not None:
            try:
                # SI-SDR entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
                # Spectral Distortion (normalize, <2 dB is excellent)
                lsd = self.calculate_spectral_distortion(reference, audio)
                scores["spectral_distortion_db"] = lsd
                scores["spectral_distortion_normalized"] = 1.0 / (1.0 + lsd / 2)

            except Exception as e:
                logger.warning("Comparative metrics failed: %s", e)

        # Calculate overall naturalness (weighted combination)
        # SI-SDR entfernt §4.4+§10.2 — Gewichte auf verbleibende Metriken umverteilt
        if reference is not None and "spectral_distortion_normalized" in scores:
            # With reference: include spectral distortion (§4.4-konforme Vergleichsmetrik)
            naturalness = (
                scores["spectral_flatness"] * 0.25
                + scores["temporal_smoothness"] * 0.30
                + scores["harmonic_coherence"] * 0.25
                + scores["noise_floor_consistency"] * 0.10
                + scores["spectral_distortion_normalized"] * 0.10
            )
        else:
            # Without reference: intrinsic metrics only
            naturalness = (
                scores["spectral_flatness"] * 0.30
                + scores["temporal_smoothness"] * 0.30
                + scores["harmonic_coherence"] * 0.25
                + scores["noise_floor_consistency"] * 0.15
            )

        scores["naturalness_overall"] = float(np.nan_to_num(naturalness, nan=0.5, posinf=1.0, neginf=0.0))
        return scores

    def calculate_roughness_zwicker_detailed(self, audio: np.ndarray) -> float:
        """
        Zwicker roughness approximation via critical-band modulation analysis.

        Implements Zwicker & Fastl (1999) §5.7:
            R ≈ c_f · Σ_k w(f_mod,k) · ΔL_k
        where:
            - critical bands: 24 Bark bands (100–15500 Hz, Traunmüller 1990)
            - w(f_mod) = (f_mod/70) · exp(1 − f_mod/70)  — MTF peak at 70 Hz
            - ΔL_k: normalised modulation depth per band (DC-removed)
            - roughness range: 20–200 Hz modulation frequency

        Returns:
            Roughness ∈ [0, 1]  (0 = perfectly smooth, 1 = maximum roughness)
        """
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        mono = np.nan_to_num(np.asarray(mono, dtype=np.float32))

        if len(mono) < 2048:
            return self.calculate_roughness(audio)

        # 24 Bark critical-band boundaries [Hz] (Traunmüller 1990)
        bark_freqs = [
            100,
            200,
            300,
            400,
            510,
            630,
            770,
            920,
            1080,
            1270,
            1480,
            1720,
            2000,
            2320,
            2700,
            3150,
            3700,
            4400,
            5300,
            6400,
            7700,
            9500,
            12000,
            15500,
        ]

        # Fine-grained STFT: hop=64 → frame_rate=750 Hz → Nyquist 375 Hz > 200 Hz ✓
        nperseg = 1024
        hop = 64
        _, _, Zxx = stft(mono, fs=self.sample_rate, nperseg=nperseg, noverlap=nperseg - hop, window="hann")
        magnitude = np.abs(Zxx)  # (freq_bins, time_frames)
        n_freq_bins, n_frames = magnitude.shape

        if n_frames < 10:
            return self.calculate_roughness(audio)

        frame_rate = self.sample_rate / hop
        freq_res = self.sample_rate / nperseg
        roughness_sum = 0.0
        n_active = 0

        for i in range(len(bark_freqs) - 1):
            bin_lo = max(0, int(bark_freqs[i] / freq_res))
            bin_hi = min(n_freq_bins, int(bark_freqs[i + 1] / freq_res))
            if bin_hi <= bin_lo:
                continue

            envelope = magnitude[bin_lo:bin_hi, :].mean(axis=0)
            mean_env = float(np.mean(envelope))
            if mean_env < 1e-10:
                continue

            # Normalised envelope (relative modulation depth, DC ≈ 1.0)
            env_norm = envelope / mean_env
            env_fft = np.abs(np.fft.rfft(env_norm - 1.0))  # remove DC
            mod_freqs = np.fft.rfftfreq(len(env_norm), d=1.0 / frame_rate)

            rough_mask = (mod_freqs >= 20.0) & (mod_freqs <= 200.0)
            if not np.any(rough_mask):
                continue

            mod_f = mod_freqs[rough_mask]
            mod_amp = env_fft[rough_mask]

            # Zwicker MTF: peak at 70 Hz, Gamma-type normalised amplitude
            w = (mod_f / 70.0) * np.exp(1.0 - mod_f / 70.0)
            w = np.clip(w, 0.0, None)
            w_sum = float(np.sum(w)) + 1e-10

            roughness_sum += float(np.sum(w * mod_amp)) / w_sum
            n_active += 1

        if n_active == 0:
            return self.calculate_roughness(audio)

        # Sigmoid-like calibration: r_mean≈0.02 → smooth, r_mean≈0.2 → rough
        r_mean = roughness_sum / n_active
        roughness = float(np.clip(r_mean / (r_mean + 0.05), 0.0, 1.0))
        # NaN/Inf-Guard (§3.1)
        return float(np.nan_to_num(roughness, nan=0.0))


def measure_quality_improvement(
    original: np.ndarray, processed: np.ndarray, sample_rate: int = 44100
) -> dict[str, float]:
    """
    Misst quality improvement from processing.

    Args:
        original: Original audio
        processed: Processed audio
        sample_rate: Sample rate

    Returns:
        Dictionary with improvement metrics
    """
    metrics = PsychoAcousticMetrics(sample_rate)

    original_quality = metrics.calculate_naturalness_score(original)
    processed_quality = metrics.calculate_naturalness_score(processed, reference=original)

    improvement = {
        "original_naturalness": original_quality["naturalness_overall"],
        "processed_naturalness": processed_quality["naturalness_overall"],
        "improvement": processed_quality["naturalness_overall"] - original_quality["naturalness_overall"],
        "improvement_percent": (
            (processed_quality["naturalness_overall"] - original_quality["naturalness_overall"])
            / (original_quality["naturalness_overall"] + 1e-10)
            * 100
        ),
    }

    # Add detailed metrics
    for key in ["spectral_flatness", "temporal_smoothness", "harmonic_coherence", "noise_floor_consistency"]:
        improvement[f"original_{key}"] = original_quality[key]
        improvement[f"processed_{key}"] = processed_quality[key]

    # sisdr_db entfernt — verboten §4.4+§10.2 (SI-SDR Sprach-Metrik)
    if "spectral_distortion_db" in processed_quality:
        improvement["spectral_distortion_db"] = processed_quality["spectral_distortion_db"]

    return improvement


if __name__ == "__main__":
    # Test with synthetic audio
    logger.debug("\n" + "=" * 70)
    logger.debug("Psychoacoustic Metrics Test")
    logger.debug("=" * 70)

    # Generate test signals
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr))

    # Clean signal
    clean = np.sin(2 * np.pi * 440 * t) * 0.3

    # Degraded signal (with artifacts)
    degraded = clean.copy()
    degraded += np.random.randn(len(clean)) * 0.05  # Add noise

    # Add clicks
    for _ in range(10):
        pos = np.random.randint(0, len(clean))
        degraded[pos] += 0.5

    # Calculate metrics
    metrics = PsychoAcousticMetrics(sr)

    logger.debug("\nClean Signal:")
    clean_scores = metrics.calculate_naturalness_score(clean)
    for key, val in clean_scores.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\nDegraded Signal:")
    degraded_scores = metrics.calculate_naturalness_score(degraded, reference=clean)
    for key, val in degraded_scores.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\nImprovement Analysis:")
    improvement = measure_quality_improvement(degraded, clean, sr)
    for key, val in improvement.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\n" + "=" * 70)
    logger.debug("✅ Psychoacoustic Metrics Module operational")
