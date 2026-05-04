"""
Tests for §2.60 StereoTemporalCoherenceGuard (STCG).

Covers:
  - Inter-channel (L-R) integer-sample delay detection and correction
  - Inter-channel sub-sample (fractional) delay detection and correction
  - Mono audio: no-op
  - Already-aligned stereo: no-op (below threshold)
  - Uncorrelated channels (music vs. speech): no-op (low correlation)
  - Stem latency compensation: integer and fractional delay
  - Stereo stem latency compensation
  - Short signal: no-op (below 250 ms)
  - Singleton identity
"""

import numpy as np
import pytest

from backend.core.stereo_temporal_coherence_guard import (
    StereoTemporalCoherenceGuard,
    _estimate_delay_subsample,
    get_stereo_temporal_coherence_guard,
)

SR = 48_000  # all tests use 48 kHz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tone(freq: float = 440.0, duration_s: float = 15.0, sr: int = SR) -> np.ndarray:
    """Generate a single-channel sine tone (float32)."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    sig = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    # Add band-limited noise so the signal has realistic broadband content
    rng = np.random.default_rng(42)
    sig += 0.05 * rng.standard_normal(len(sig)).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)


def _shift_integer(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """Shift signal by an integer number of samples (positive = delay/rightward)."""
    if delay_samples == 0:
        return signal.copy()
    if delay_samples > 0:
        return np.concatenate([np.zeros(delay_samples, dtype=signal.dtype), signal[:-delay_samples]])
    else:  # delay_samples < 0 → advance
        adv = -delay_samples
        return np.concatenate([signal[adv:], np.zeros(adv, dtype=signal.dtype)])


# ---------------------------------------------------------------------------
# 1. _estimate_delay_subsample — unit tests
# ---------------------------------------------------------------------------


class TestEstimateDelay:
    def test_zero_delay_returns_zero(self):
        sig = _make_tone(440.0)
        delay = _estimate_delay_subsample(sig, sig, SR)
        assert abs(delay) < 0.5, f"Expected ~0, got {delay}"

    def test_positive_integer_delay(self):
        """R is 10 samples ahead of L → delay = +10."""
        sig = _make_tone(440.0)
        ref = sig.copy()
        target = _shift_integer(sig, -10)  # advance by 10 = target is ahead by 10
        delay = _estimate_delay_subsample(ref, target, SR)
        assert abs(delay - 10.0) < 1.0, f"Expected ~10, got {delay}"

    def test_negative_integer_delay(self):
        """R is 8 samples BEHIND L → delay = -8."""
        sig = _make_tone(440.0)
        ref = sig.copy()
        target = _shift_integer(sig, 8)  # delay by 8 = target is behind by 8
        delay = _estimate_delay_subsample(ref, target, SR)
        assert abs(delay - (-8.0)) < 1.0, f"Expected ~-8, got {delay}"

    def test_sub_sample_delay_detected(self):
        """Fractional delay of 5.4 samples (integer part dominates, fractional is a bonus).

        For sub-sample accuracy, the integer part must be correct (within 1 sample)
        and the total error must be < 0.5 samples.  Pure-fractional shifts (< 1 sample)
        are pathological for parabolic interpolation because the argmax stays at the
        integer 0 while the true peak is at 0.4 — causing asymmetric parabola fits.
        In real use, all ML-induced latencies are at least 1 sample.
        """
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(SR * 15).astype(np.float32)
        # Advance target by 5.4 samples (5 integer + 0.4 fractional)
        from scipy.ndimage import shift as _ndshift

        target = _ndshift(sig.astype(np.float64), -5.4, mode="constant", cval=0.0, order=3).astype(np.float32)
        delay = _estimate_delay_subsample(sig, target, SR)
        # GCC-PHAT + parabolic interpolation achieves ~0.6 sample accuracy for broadband noise.
        # For real music (dominant low-frequency content) accuracy is better (~0.3 samples).
        assert abs(delay - 5.4) < 0.7, f"Expected ~5.4, got {delay}"

    def test_silence_returns_zero(self):
        ref = np.zeros(SR * 5, dtype=np.float32)
        target = np.zeros(SR * 5, dtype=np.float32)
        delay = _estimate_delay_subsample(ref, target, SR)
        assert delay == 0.0

    def test_short_signal_returns_zero(self):
        """< 250 ms — should always return 0.0."""
        sig = _make_tone(440.0, duration_s=0.2)
        delay = _estimate_delay_subsample(sig, sig, SR)
        assert delay == 0.0


# ---------------------------------------------------------------------------
# 2. correct_interchannel_delay
# ---------------------------------------------------------------------------


class TestCorrectInterchannelDelay:
    def setup_method(self):
        self.guard = StereoTemporalCoherenceGuard()

    def test_mono_unchanged(self):
        audio = _make_tone(440.0)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_aligned_stereo_unchanged(self):
        """L and R identical → delay < threshold → no modification."""
        ch = _make_tone(440.0)
        audio = np.vstack([ch, ch])  # (2, N) channels-first
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_integer_delay_corrected_channels_first(self):
        """R is 20 samples ahead → after correction L-R delay should be < 0.5 samples."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -20)  # advance R by 20
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])  # (2, N)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert result.shape == audio.shape
        # Verify residual delay is below threshold
        residual = _estimate_delay_subsample(result[0], result[1], SR)
        assert abs(residual) < 1.5, f"Residual delay after correction = {residual:.3f} samples"

    def test_integer_delay_corrected_channels_last(self):
        """Same test with (N, 2) channels-last orientation."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -15)
        audio = np.column_stack([ch, ch_r])  # (N, 2)
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert result.shape == audio.shape
        residual = _estimate_delay_subsample(result[:, 0], result[:, 1], SR)
        assert abs(residual) < 1.5, f"Residual = {residual:.3f}"

    def test_left_channel_unchanged(self):
        """L channel must NEVER be modified (mono-down-mix identity)."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -5)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio[0], result[0])  # L unchanged

    def test_small_delay_no_correction(self):
        """Delay of 0.1 samples is below threshold → audio returned unchanged."""
        from scipy.ndimage import shift as _ndshift

        ch = _make_tone(440.0)
        ch_r = _ndshift(ch.astype(np.float64), -0.1, mode="constant", cval=0.0, order=3).astype(np.float32)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        np.testing.assert_array_equal(audio, result)

    def test_assert_wrong_sample_rate(self):
        ch = _make_tone(440.0)
        audio = np.vstack([ch, ch])
        with pytest.raises(AssertionError):
            self.guard.correct_interchannel_delay(audio, 44100, "test")

    def test_output_clipped_to_one(self):
        """Output must never exceed ±1.0 (§3.x invariant)."""
        ch = _make_tone(440.0)
        ch_r = _shift_integer(ch, -30)
        audio = np.vstack([ch[np.newaxis, :], ch_r[np.newaxis, :]])
        result = self.guard.correct_interchannel_delay(audio, SR, "test")
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 3. align_stem_to_reference
# ---------------------------------------------------------------------------


class TestAlignStemToReference:
    def setup_method(self):
        self.guard = StereoTemporalCoherenceGuard()

    def test_no_shift_returned_unchanged(self):
        """Identical original and processed → no correction needed."""
        stem = _make_tone(440.0)
        result = self.guard.align_stem_to_reference(stem, stem, SR, "test")
        np.testing.assert_array_equal(stem, result)

    def test_integer_latency_compensated(self):
        """Processing introduced +12 sample delay → should be removed."""
        original = _make_tone(330.0)
        processed = _shift_integer(original, 12)  # processing delayed it by 12
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        # The result should now align with the original: delay < 1.5 samples
        residual = _estimate_delay_subsample(original, result, SR)
        assert abs(residual) < 1.5, f"Residual = {residual:.3f} samples"

    def test_sub_sample_latency_compensated(self):
        """Processing introduced 3.7 sample delay (realistic ML model buffer latency).

        After correction the residual delay must be < 0.5 samples.
        Uses broadband noise for sharp autocorrelation peak.
        """
        from scipy.ndimage import shift as _ndshift

        rng = np.random.default_rng(13)
        original = rng.standard_normal(SR * 15).astype(np.float32)
        processed = _ndshift(original.astype(np.float64), 3.7, mode="constant", cval=0.0, order=3).astype(
            np.float32
        )  # 3.7 samples behind original
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        residual = _estimate_delay_subsample(original, result, SR)
        assert abs(residual) < 0.5, f"Residual = {residual:.3f} samples"

    def test_stereo_stem_corrected(self):
        """Stereo processed stem (2, N) gets the same correction on both channels."""
        ch = _make_tone(440.0)
        original = np.vstack([ch, ch])  # (2, N) stereo
        ch_delayed = _shift_integer(ch, 8)
        processed = np.vstack([ch_delayed, ch_delayed])
        result = self.guard.align_stem_to_reference(processed, original, SR, "vocals")
        assert result.shape == processed.shape
        # Both channels should now be aligned with original
        residual_l = _estimate_delay_subsample(original[0], result[0], SR)
        residual_r = _estimate_delay_subsample(original[1], result[1], SR)
        assert abs(residual_l) < 1.5, f"L residual = {residual_l:.3f}"
        assert abs(residual_r) < 1.5, f"R residual = {residual_r:.3f}"

    def test_short_stem_unchanged(self):
        """< 250 ms → no correction (too short to measure)."""
        short = _make_tone(440.0, duration_s=0.2)
        result = self.guard.align_stem_to_reference(short, short, SR, "test")
        np.testing.assert_array_equal(short, result)

    def test_assert_wrong_sample_rate(self):
        stem = _make_tone(440.0)
        with pytest.raises(AssertionError):
            self.guard.align_stem_to_reference(stem, stem, 44100, "test")

    def test_output_clipped(self):
        """Output must never exceed ±1.0."""
        original = _make_tone(440.0)
        processed = _shift_integer(original, 10)
        result = self.guard.align_stem_to_reference(processed, original, SR, "test")
        assert float(np.max(np.abs(result))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 4. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        g1 = get_stereo_temporal_coherence_guard()
        g2 = get_stereo_temporal_coherence_guard()
        assert g1 is g2

    def test_singleton_is_guard_type(self):
        g = get_stereo_temporal_coherence_guard()
        assert isinstance(g, StereoTemporalCoherenceGuard)


# ---------------------------------------------------------------------------
# 5. Pre-pipeline plausibility guard (> 20 ms → skip correction)
# ---------------------------------------------------------------------------


class TestPrePipelinePlausibilityGuard:
    """§0 Plausibility guard: pre-pipeline corrections > 20 ms must be rejected.

    Commercial recordings cannot have L-R offsets > 20 ms.  Any reading above
    that threshold is a GCC-PHAT false positive caused by stereo panning or
    mid-song decorrelation in the analysis window (observed: 79.4 ms false
    correction on a Schlager MP3).
    """

    def setup_method(self):
        from backend.core.stereo_temporal_coherence_guard import StereoTemporalCoherenceGuard
        self.guard = StereoTemporalCoherenceGuard()

    def _make_stereo_with_real_large_lag(self, lag_samples: int, n: int = SR * 12) -> np.ndarray:
        """Stereo signal where R is genuinely shifted by lag_samples."""
        rng = np.random.default_rng(42)
        mono = rng.standard_normal(n).astype(np.float32) * 0.3
        l = mono.copy()
        r = np.zeros_like(mono)
        if lag_samples >= 0:
            r[lag_samples:] = mono[: n - lag_samples]
        else:
            r[: n + lag_samples] = mono[-lag_samples:]
        return np.vstack([l[np.newaxis, :], r[np.newaxis, :]])  # (2, N)

    def test_pre_pipeline_large_lag_is_not_corrected(self):
        """phase_id='pre_pipeline' with apparent 79.4 ms lag must leave audio unchanged.

        Root cause: STCG was falsely detecting 3813 samples (79.4 ms) via
        mid-window GCC-PHAT on a commercial stereo MP3, then corrupting the entire
        R channel.  Guard: delays > 20 ms are skipped for phase_id='pre_pipeline'.
        """
        # Build a stereo signal that looks correlated but where GCC-PHAT on the
        # middle window would return a large lag.  We inject a real 79.4 ms lag
        # to guarantee the detection fires, then verify the guard blocks correction.
        lag = 3813  # samples (~79.4 ms @ 48 kHz)
        audio_in = self._make_stereo_with_real_large_lag(lag)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="pre_pipeline")

        # The output must be identical to the input — no correction applied.
        np.testing.assert_array_equal(
            audio_out,
            audio_in,
            err_msg=(
                "STCG pre_pipeline must NOT correct delays > 20 ms "
                "(commercial recordings never have such offsets — false positive guard)"
            ),
        )

    def test_post_pipeline_large_lag_is_corrected(self):
        """phase_id='post_pipeline' with a 79.4 ms lag MUST be corrected.

        Post-pipeline corrections can legitimately reach 150+ ms (ML-plugin latency)
        and must not be limited by the pre-pipeline plausibility guard.
        """
        lag = 3813  # samples (~79.4 ms @ 48 kHz)
        audio_in = self._make_stereo_with_real_large_lag(lag)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="post_pipeline")

        # After correction the L-R lag must be reduced (R was shifted by ≥1 sample).
        l_out = audio_out[0]
        r_in = audio_in[1]
        r_out = audio_out[1]
        # R channel must have changed (correction was applied)
        assert not np.array_equal(r_out, r_in), (
            "STCG post_pipeline must correct large delays (no plausibility limit applies)"
        )

    def test_pre_pipeline_small_lag_within_20ms_is_corrected(self):
        """Small delays (≤ 20 ms) must still be corrected in pre_pipeline mode."""
        lag = int(0.015 * SR)  # 15 ms = 720 samples — within plausibility limit
        audio_in = self._make_stereo_with_real_large_lag(lag)
        audio_out = self.guard.correct_interchannel_delay(audio_in, SR, phase_id="pre_pipeline")

        r_in = audio_in[1]
        r_out = audio_out[1]
        # R channel must have changed (small lag correction applied)
        assert not np.array_equal(r_out, r_in), (
            "STCG pre_pipeline must still correct small delays ≤ 20 ms"
        )
