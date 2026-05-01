"""
§2.60 Stereo Temporal Coherence Guard (STCG)
============================================

Prevents temporal misalignment between:
  1. L and R stereo channels (inter-channel delay, e.g. from independent-channel DSP)
  2. Vocal stem and instrumental stem during phase_42 recombination (processing-chain latency)

Two public entry points:
  - correct_interchannel_delay(audio, sr, phase_id) → aligned audio
  - align_stem_to_reference(processed_stem, original_stem, sr, stem_label) → latency-compensated stem

Algorithm:
  - FFT cross-correlation on a 10-second mid-song window (fast: <10ms for 48kHz 10s audio)
  - Parabolic interpolation of cross-correlation peak for sub-sample precision (Smith 2011)
  - scipy.ndimage.shift with cubic spline interpolation for the actual sub-sample correction
    (numerically stable, avoids Lagrange FIR boundary artefacts for |frac| ≤ 0.5)

Performance budget: ≤ 15ms per call for songs up to 10 min @ 48kHz.

§0 Minimal-Intervention: no correction when |delay| < 0.5 samples.
§2.51: All corrections are applied to both channels symmetrically (linked-stereo),
       not independently per channel.

Singleton: get_stereo_temporal_coherence_guard()

References:
  - Smith, J.O. (2011). Spectral Audio Signal Processing. CCRMA.
  - Laakso et al. (1996). Splitting the Unit Delay. IEEE Signal Processing Magazine 13(1), 30–60.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.ndimage import shift as _ndimage_shift

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Analysis window: 10 s from the mid-song (480 000 samples @ 48 kHz)
_ANALYSIS_WINDOW_S: float = 10.0

# Maximum inter-channel delay to search for: ±200 ms (9 600 samples @ 48 kHz)
# Covers tape-head azimuth errors, ML-plugin output latency, analogue playback chains,
# and pipeline-introduced phase-vocoder/PSOLA latency (observed: 8 777 samples = 182.9 ms).
_MAX_DELAY_SAMPLES: int = 9_600

# Correction threshold: delays ≤ this are left untouched (§0 Minimal-Intervention).
_CORRECTION_THRESHOLD_SAMPLES: float = 0.5

# Minimum cross-correlation peak magnitude for reliable estimation.
# Below this, signals are uncorrelated (e.g. independent instruments) — no correction.
_MIN_CORRELATION_CONFIDENCE: float = 0.04

# scipy.ndimage.shift interpolation order (3 = cubic spline, good quality/speed balance)
_INTERP_ORDER: int = 3


# ---------------------------------------------------------------------------
# Core DSP helpers
# ---------------------------------------------------------------------------


def _to_mono_analysis(audio: np.ndarray) -> np.ndarray:
    """Collapse audio to a 1-D float32 mono array for correlation analysis.

    Accepts (N,), (2, N) channels-first, or (N, 2) channels-last.
    Returns a contiguous float32 array of shape (N,).
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # Detect orientation: channels-first (2, N) vs channels-last (N, 2)
        if arr.shape[0] == 2 and arr.shape[1] > 2:
            return (arr[0] + arr[1]) * 0.5  # (2, N) → mono
        if arr.shape[1] == 2 and arr.shape[0] > 2:
            return (arr[:, 0] + arr[:, 1]) * 0.5  # (N, 2) → mono
        # Fallback: sum along shorter axis
        return arr.mean(axis=0 if arr.shape[0] <= arr.shape[1] else 1)
    # Unexpected rank — return first row/channel
    return arr.reshape(-1)[: arr.size // arr.shape[0]]


def _mid_window(signal: np.ndarray, sr: int) -> np.ndarray:
    """Extract a 10-second window from the middle of *signal*."""
    n_window = int(_ANALYSIS_WINDOW_S * sr)
    n = min(len(signal), n_window)
    start = max(0, (len(signal) - n) // 2)
    return signal[start : start + n]


def _gcc_phat(r: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, int]:
    """GCC-PHAT: Generalized Cross-Correlation with Phase Transform.

    Whitens the cross-spectrum so that the correlation peak becomes
    impulse-like regardless of the signal's spectral shape.  This makes
    parabolic sub-sample interpolation reliable for broadband signals
    (white noise, music) and for sinusoidal signals alike.

    Returns:
        (cc_shifted, center_idx):
            cc_shifted — full-length correlation array with zero-lag at center_idx.
            center_idx — index of the zero-lag element.

    Reference: Knapp & Carter (1976). IEEE TASLP 24(4), 320–327.
    """
    n = len(r)
    n_fft = int(2 ** np.ceil(np.log2(2 * n)))  # Next power-of-2 ≥ 2N (linear correlation)
    R = np.fft.rfft(r.astype(np.float64), n=n_fft)
    T = np.fft.rfft(t.astype(np.float64), n=n_fft)
    G = R * np.conj(T)
    # PHAT weighting: divide by |G| → normalised to unit magnitude per frequency bin
    G_phat = G / (np.abs(G) + 1e-9)
    # irfft output: cc[0] = zero-lag, cc[k] = lag +k, cc[n_fft-k] = lag -k
    cc = np.real(np.fft.irfft(G_phat, n=n_fft))
    # Rearrange so that zero-lag sits at the centre using np.roll
    # After np.roll(cc, n_fft//2): index n_fft//2 = original lag-0
    cc_shifted = np.roll(cc, n_fft // 2)
    center_idx = n_fft // 2
    return cc_shifted, center_idx


def _estimate_delay_subsample(ref: np.ndarray, target: np.ndarray, sr: int) -> float:
    """Estimate the fractional-sample delay of *target* relative to *ref*.

    Convention:
      positive  → target is AHEAD of ref (target occurred earlier in time)
      negative  → target is BEHIND ref (target occurred later in time)

    Returns 0.0 when:
      - Signal too short (< 250 ms) for reliable estimation
      - GCC-PHAT peak < _MIN_CORRELATION_CONFIDENCE (uncorrelated signals)

    Algorithm:
      GCC-PHAT cross-correlation on a 10-second mid-song window, then parabolic
      interpolation on the peak for sub-sample precision.  GCC-PHAT whitens the
      cross-spectrum so the correlation peak is impulse-like for both broadband
      and tonal content (Knapp & Carter 1976; Smith 2011 §3.7).
    """
    r_full = _to_mono_analysis(ref)
    t_full = _to_mono_analysis(target)

    r = _mid_window(r_full, sr)
    t = _mid_window(t_full, sr)

    n = min(len(r), len(t))
    if n < sr // 4:  # < 250 ms — not enough context
        return 0.0

    r = r[:n].astype(np.float32)
    t = t[:n].astype(np.float32)

    # Energy check — skip silence channels
    r_rms = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
    t_rms = float(np.sqrt(np.mean(t.astype(np.float64) ** 2)))
    if r_rms < 1e-8 or t_rms < 1e-8:
        return 0.0

    # GCC-PHAT: impulse-shaped peak for reliable sub-sample interpolation
    corr, center = _gcc_phat(r, t)

    # Restrict search to ±_MAX_DELAY_SAMPLES around zero lag
    lo = max(0, center - _MAX_DELAY_SAMPLES)
    hi = min(len(corr), center + _MAX_DELAY_SAMPLES + 1)
    region = corr[lo:hi]

    peak_local_idx = int(np.argmax(np.abs(region)))
    peak_val = float(region[peak_local_idx])

    if abs(peak_val) < _MIN_CORRELATION_CONFIDENCE:
        logger.debug(
            "STCG: GCC-PHAT peak=%.4f below threshold=%.4f — no correction",
            abs(peak_val),
            _MIN_CORRELATION_CONFIDENCE,
        )
        return 0.0

    # Global peak index in full correlation array
    global_peak = lo + peak_local_idx

    # Parabolic sub-sample interpolation (Smith 2011 §3.7)
    # With GCC-PHAT whitening the peak is narrow/impulse-like → parabola fits well
    if 1 <= global_peak < len(corr) - 1:
        y0 = corr[global_peak - 1]
        y1 = corr[global_peak]
        y2 = corr[global_peak + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        frac_offset = (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        # Clamp sub-sample offset to ±0.5 (parabola only valid near peak)
        frac_offset = float(np.clip(frac_offset, -0.5, 0.5))
    else:
        frac_offset = 0.0

    delay = float(global_peak - center) + frac_offset
    return delay


def _apply_correction_shift(signal: np.ndarray, shift_samples: float) -> np.ndarray:
    """Apply a signed fractional-sample shift to a 1-D float32 array.

    Positive shift_samples → signal content moves to later indices (rightward, delay).
    Negative shift_samples → signal content moves to earlier indices (advance).

    Uses scipy.ndimage.shift with cubic spline interpolation (order=3).
    Boundary condition: mode='constant', cval=0.0 (silence at edges).

    Args:
        signal: 1-D float32 array.
        shift_samples: Fractional shift in samples (signed).

    Returns:
        Shifted signal, same length as input, clipped to [-1, 1].
    """
    if abs(shift_samples) < 1e-4:
        return signal.astype(np.float32)

    shifted = _ndimage_shift(
        signal.astype(np.float64),
        shift=shift_samples,
        mode="constant",
        cval=0.0,
        order=_INTERP_ORDER,
    )
    return np.clip(
        np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0),
        -1.0,
        1.0,
    ).astype(np.float32)


def _apply_shift_to_audio(audio: np.ndarray, shift_samples: float) -> np.ndarray:
    """Apply fractional-sample shift to audio (mono or stereo, any orientation).

    For stereo, the SAME shift is applied to ALL channels (linked-stereo, §2.51):
    the inter-channel relationship is preserved; the whole signal is moved in time.

    Args:
        audio: 1-D, (2, N), or (N, 2) float32 array.
        shift_samples: Fractional shift in samples (positive = delay).

    Returns:
        Shifted audio, same shape and dtype as input.
    """
    if abs(shift_samples) < 1e-4:
        return audio

    orig_dtype = audio.dtype
    arr = np.asarray(audio, dtype=np.float32)

    if arr.ndim == 1:
        return _apply_correction_shift(arr, shift_samples).astype(orig_dtype)

    if arr.ndim == 2:
        channels_first = arr.shape[0] == 2 and arr.shape[1] > 2
        if channels_first:
            shifted_ch = np.vstack(
                [_apply_correction_shift(arr[i], shift_samples)[np.newaxis, :] for i in range(arr.shape[0])]
            )
        else:
            shifted_ch = np.column_stack(
                [_apply_correction_shift(arr[:, i], shift_samples) for i in range(arr.shape[1])]
            )
        return shifted_ch.astype(orig_dtype)

    # Unexpected rank — return unchanged
    logger.debug("STCG: unexpected audio rank %d — skipping shift", arr.ndim)
    return audio


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------


class StereoTemporalCoherenceGuard:
    """Ensures temporal coherence across stereo channels and between separated stems.

    Thread-safe. Instantiate via get_stereo_temporal_coherence_guard().

    Two responsibilities:
      1. correct_interchannel_delay  — detects and corrects L-R sample offset
                                       (call pre-pipeline and/or post-pipeline).
      2. align_stem_to_reference     — re-aligns a processed stem to the timing of
                                       its un-processed original (phase_42 stem remix).
    """

    # ------------------------------------------------------------------
    # 1. Inter-channel delay correction (L vs R)
    # ------------------------------------------------------------------

    def correct_interchannel_delay(
        self,
        audio: np.ndarray,
        sr: int,
        phase_id: str = "unknown",
    ) -> np.ndarray:
        """Detect and correct any L-R inter-channel temporal offset.

        The RIGHT channel is corrected to align with the LEFT channel as reference.
        LEFT channel is never modified (preserves mono-down-mix identity).

        §0 Minimal-Intervention: returns audio unchanged if |delay| < 0.5 samples.
        §2.51: Correction is applied only to R (linked-stereo, not independent).

        Args:
            audio: Audio array. Shape: mono (N,), channels-first (2, N),
                   or channels-last (N, 2). Mono → returned unchanged.
            sr:    Sample rate — must be 48000.
            phase_id: Label for logging (e.g. "pre_pipeline", "post_pipeline").

        Returns:
            Audio with corrected L-R alignment. Same shape and dtype as input.
        """
        assert sr == 48000, f"STCG requires sr=48000, got {sr}"

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2:
            return audio  # Mono — no inter-channel alignment needed

        # Detect stereo orientation
        channels_first = arr.shape[0] == 2 and arr.shape[1] > 2
        if channels_first:
            ch_l = arr[0]
            ch_r = arr[1]
        elif arr.shape[1] == 2 and arr.shape[0] > 2:
            ch_l = arr[:, 0]
            ch_r = arr[:, 1]
        else:
            return audio  # Ambiguous shape — skip

        if len(ch_l) < sr // 4:
            return audio  # < 250 ms — too short to measure

        delay = _estimate_delay_subsample(ch_l, ch_r, sr)
        delay_ms = delay / sr * 1000.0

        if abs(delay) < _CORRECTION_THRESHOLD_SAMPLES:
            logger.debug(
                "STCG [%s]: L-R delay=%.4f samples (%.3f ms) — within threshold, no correction",
                phase_id,
                delay,
                delay_ms,
            )
            return audio

        logger.info(
            "STCG [%s]: L-R delay=%.4f samples (%.3f ms) — correcting R channel",
            phase_id,
            delay,
            delay_ms,
        )

        # Positive delay means R is AHEAD of L → shift R to the right (delay R) to align
        ch_r_corrected = _apply_correction_shift(ch_r, shift_samples=delay)

        orig_dtype = audio.dtype
        if channels_first:
            result = np.vstack([ch_l[np.newaxis, :], ch_r_corrected[np.newaxis, :]])
        else:
            result = np.column_stack([ch_l, ch_r_corrected])

        return result.astype(orig_dtype)

    # ------------------------------------------------------------------
    # 2. Stem latency compensation (processed vs original)
    # ------------------------------------------------------------------

    def align_stem_to_reference(
        self,
        processed_stem: np.ndarray,
        original_stem: np.ndarray,
        sr: int,
        stem_label: str = "stem",
    ) -> np.ndarray:
        """Re-align *processed_stem* to the timing of *original_stem*.

        Used in phase_42 to compensate for any latency introduced by the vocal
        enhancement chain (STFT, ML inference, compression, formant EQ) before
        the enhanced vocals are mixed back with the instrumental stem.

        The original_stem is the un-processed stem (timing reference).
        The processed_stem is the same signal after enhancement/processing.

        §0 Minimal-Intervention: returns processed_stem unchanged if
        |delay| < 0.5 samples.

        Args:
            processed_stem: Enhanced/processed stem, any orientation.
            original_stem:  Original un-processed stem (timing reference).
            sr:             Sample rate — must be 48000.
            stem_label:     Label for logging.

        Returns:
            processed_stem aligned to original_stem's timing. Same shape as input.
        """
        assert sr == 48000, f"STCG requires sr=48000, got {sr}"

        mono_orig = _to_mono_analysis(original_stem)
        mono_proc = _to_mono_analysis(processed_stem)

        if len(mono_orig) < sr // 4 or len(mono_proc) < sr // 4:
            return processed_stem  # Too short

        # How many samples is processed AHEAD of original?
        # Positive = processed is ahead → shift processed to the right to align
        delay = _estimate_delay_subsample(mono_orig, mono_proc, sr)
        delay_ms = delay / sr * 1000.0

        if abs(delay) < _CORRECTION_THRESHOLD_SAMPLES:
            logger.debug(
                "STCG stem [%s]: delay=%.4f samples (%.3f ms) — within threshold, no correction",
                stem_label,
                delay,
                delay_ms,
            )
            return processed_stem

        logger.info(
            "STCG stem [%s]: processing latency=%.4f samples (%.3f ms) — compensating",
            stem_label,
            delay,
            delay_ms,
        )

        return _apply_shift_to_audio(processed_stem, shift_samples=delay)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_instance: StereoTemporalCoherenceGuard | None = None
_lock = threading.Lock()


def get_stereo_temporal_coherence_guard() -> StereoTemporalCoherenceGuard:
    """Return the process-wide StereoTemporalCoherenceGuard singleton."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StereoTemporalCoherenceGuard()
    return _instance
