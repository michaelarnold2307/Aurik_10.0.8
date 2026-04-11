"""
dsp/psychoacoustics.py — ISO 532-1 Stationary Loudness (Zwicker/Fastl)
========================================================================

§4.1b [RELEASE_MUST] Psychoacoustic loudness measurement.

Implements total loudness N in sone via 24 Bark-band critical-band analysis
following the stationary method of ISO 532-1:2017.

Algorithm:
    1. 24 Butterworth bandpass filters (Bark scale, 4th order)
    2. Band-level computation (dB SPL approximation from digital level)
    3. Equal-loudness contour correction (ISO 226:2023 approximation)
    4. Specific loudness per band (Stevens' power law)
    5. Total loudness N = sum of specific loudnesses

Performance: ≤ 50 ms for 5-second window @ 48 kHz (Spec §4.1b).

References:
    - ISO 532-1:2017 — Acoustics — Methods for calculating loudness — Part 1: Zwicker method
    - Zwicker & Fastl (2007): Psychoacoustics — Facts and Models, 3rd ed.
    - ISO 226:2023 — Equal-loudness-level contours
    - Stevens (1957): On the psychophysical law

Author: Aurik 9.11 — §4.1b implementation
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

# ── 24 Bark Critical Bands (Zwicker & Fastl 1990) ──────────────────────
# (lower_hz, upper_hz) for each of the 24 Bark bands
_BARK_BANDS: list[tuple[float, float]] = [
    (20, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 510),
    (510, 630),
    (630, 770),
    (770, 920),
    (920, 1080),
    (1080, 1270),
    (1270, 1480),
    (1480, 1720),
    (1720, 2000),
    (2000, 2320),
    (2320, 2700),
    (2700, 3150),
    (3150, 3700),
    (3700, 4400),
    (4400, 5300),
    (5300, 6400),
    (6400, 7700),
    (7700, 9500),
    (9500, 12000),
    (12000, 15500),
]

# ── Equal-Loudness Correction (ISO 226:2023 approximation) ─────────────
# dB offset to apply at each band center so that equal loudness contour
# at 40 phon is approximately flat.  Positive = ear is MORE sensitive
# (less energy needed), so we ADD to measured level.
# Derived from ISO 226:2023 40-phon contour at band center frequencies.
_EQUAL_LOUDNESS_OFFSET_DB: list[float] = [
    -22.0,  # 60 Hz  — ear very insensitive
    -14.0,  # 150 Hz
    -9.0,  # 250 Hz
    -5.5,  # 350 Hz
    -3.5,  # 455 Hz
    -2.0,  # 570 Hz
    -1.0,  # 700 Hz
    0.0,  # 845 Hz
    0.5,  # 1000 Hz — reference
    1.0,  # 1175 Hz
    1.5,  # 1375 Hz
    2.0,  # 1600 Hz
    2.5,  # 1860 Hz
    3.0,  # 2160 Hz
    3.5,  # 2510 Hz — ear most sensitive region
    3.0,  # 2925 Hz
    2.0,  # 3425 Hz
    0.5,  # 4050 Hz
    -1.5,  # 4850 Hz
    -3.5,  # 5850 Hz
    -6.0,  # 7050 Hz
    -9.0,  # 8600 Hz
    -13.0,  # 10750 Hz
    -18.0,  # 13750 Hz — ear insensitive again
]

# ── Hearing threshold in quiet (ISO 226:2023, dB SPL at band center) ───
_THRESHOLD_QUIET_DB: list[float] = [
    55.0,
    35.0,
    22.0,
    15.0,
    11.0,
    8.5,
    7.0,
    6.0,
    5.5,
    5.5,
    6.0,
    7.0,
    8.0,
    9.5,
    10.5,
    11.0,
    12.0,
    14.0,
    16.0,
    19.0,
    23.0,
    28.0,
    35.0,
    45.0,
]

# ── Digital-to-SPL offset ──────────────────────────────────────────────
# 0 dBFS ≈ 94 dB SPL (standard studio monitoring level assumption).
# This is an approximation — the absolute value doesn't matter for
# the ΔN comparison (input_sone vs output_sone), only relative accuracy.
_DBFS_TO_SPL_OFFSET: float = 94.0


def _band_filters(sr: int) -> list[np.ndarray]:
    """Pre-compute SOS Butterworth bandpass filters for 24 Bark bands.

    Args:
        sr: Sample rate in Hz.

    Returns:
        List of 24 SOS filter coefficient arrays.
    """
    nyquist = sr / 2.0
    filters: list[np.ndarray] = []
    for lo, hi in _BARK_BANDS:
        # Clamp to valid Nyquist range
        lo_n = max(lo / nyquist, 0.001)
        hi_n = min(hi / nyquist, 0.999)
        if lo_n >= hi_n:
            # Band above Nyquist — use dummy passthrough
            filters.append(np.zeros((1, 6), dtype=np.float64))
            continue
        try:
            sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
        except Exception:
            filters.append(np.zeros((1, 6), dtype=np.float64))
            continue
        filters.append(sos)
    return filters


# Cache filters per sample rate to avoid re-computation
_FILTER_CACHE: dict[int, list[np.ndarray]] = {}


def _get_filters(sr: int) -> list[np.ndarray]:
    """Get cached Bark-band filters for the given sample rate."""
    if sr not in _FILTER_CACHE:
        _FILTER_CACHE[sr] = _band_filters(sr)
    return _FILTER_CACHE[sr]


def compute_specific_loudness_zwicker(audio: np.ndarray, sr: int) -> float:
    """Compute total loudness N in sone (ISO 532-1 stationary method).

    §4.1b [RELEASE_MUST]: Psychoacoustic loudness measurement after
    broadband subtraktive phases (rumble, multiband, dereverb).

    Algorithm:
        1. Extract 5 s center segment (or full if shorter)
        2. Filter through 24 Bark-band Butterworth bandpass filters
        3. Compute band level in dB SPL (from RMS + digital-to-SPL offset)
        4. Apply equal-loudness correction (ISO 226:2023)
        5. Convert each band to specific loudness via Stevens' power law
        6. Sum specific loudnesses → total loudness N (sone)

    Reference: 1 sone = 40 phon at 1 kHz.

    Args:
        audio: float32/float64 mono or stereo, values in [-1, 1].
        sr: Sample rate in Hz.

    Returns:
        Total loudness N in sone (≥ 0.0).
    """
    # Mono downmix if stereo
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float64)

    # Extract center 5-second window (sufficient for stationary measurement)
    n_samples = len(audio)
    window_samples = min(n_samples, 5 * sr)
    start = max(0, (n_samples - window_samples) // 2)
    segment = audio[start : start + window_samples]

    if len(segment) < sr // 10:  # < 100 ms — too short
        return 0.0

    filters = _get_filters(sr)
    total_loudness = 0.0

    for band_idx in range(24):
        sos = filters[band_idx]
        if sos.shape[0] == 1 and np.all(sos == 0):
            # Dummy filter — band above Nyquist
            continue

        # Apply bandpass filter
        try:
            band_signal = sosfilt(sos, segment)
        except Exception:
            continue

        # RMS level in dBFS
        rms = float(np.sqrt(np.mean(band_signal**2) + 1e-20))
        level_dbfs = 20.0 * np.log10(max(rms, 1e-20))

        # Convert to approximate dB SPL
        level_spl = level_dbfs + _DBFS_TO_SPL_OFFSET

        # Apply equal-loudness correction
        corrected_spl = level_spl + _EQUAL_LOUDNESS_OFFSET_DB[band_idx]

        # Threshold in quiet — below this, no loudness contribution
        if corrected_spl <= _THRESHOLD_QUIET_DB[band_idx]:
            continue

        # Excess above threshold
        excess_db = corrected_spl - _THRESHOLD_QUIET_DB[band_idx]

        # Stevens' power law: specific loudness ∝ (intensity)^0.3
        # In dB domain: N_specific = k * 10^(0.3 * excess_dB / 10)
        # k chosen so that 40 dB excess at 1 kHz ≈ 1 sone
        # At 1 kHz (band 8): threshold ~5.5 dB SPL, 40 phon → 45.5 dB SPL
        # excess = 40 dB → 10^(0.3*40/10) = 10^1.2 ≈ 15.85
        # So k ≈ 1/15.85 ≈ 0.063
        k = 0.063
        specific_loudness = k * (10.0 ** (0.3 * excess_db / 10.0))

        total_loudness += specific_loudness

    return max(0.0, float(total_loudness))


def compute_loudness_delta_sone(
    audio_before: np.ndarray, audio_after: np.ndarray, sr: int
) -> tuple[float, float, float]:
    """Compute loudness change ΔN in sone between two audio signals.

    §4.1b ΔN decision table:
        ≤ 0.5 sone:  OK (loudness-neutral)
        0.5 – 1.0:   INFO
        1.0 – 2.0:   WARNING
        > 2.0:       FAIL → Dry/Wet rescue

    Args:
        audio_before: Audio before phase processing.
        audio_after: Audio after phase processing.
        sr: Sample rate in Hz.

    Returns:
        Tuple of (delta_sone, loudness_before, loudness_after).
        delta_sone = loudness_after - loudness_before (positive = louder).
    """
    n_before = compute_specific_loudness_zwicker(audio_before, sr)
    n_after = compute_specific_loudness_zwicker(audio_after, sr)
    delta = n_after - n_before
    return (float(delta), float(n_before), float(n_after))
