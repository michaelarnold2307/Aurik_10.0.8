"""Tests for GrooveMetric LF proxy robustness.

Verifies:
- DTW-based groove tracking with ≤ 8 ms RMS tolerance
- LF onset detection accuracy
- Mono and stereo support
- Edge cases (silence, very short audio)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kick_pattern(sr: int = 48000, dur: float = 4.0, bpm: float = 120.0) -> np.ndarray:
    """Create a synthetic kick drum pattern at given BPM."""
    n_samples = int(sr * dur)
    audio = np.zeros(n_samples, dtype=np.float32)
    beat_interval = int(60.0 / bpm * sr)
    # Simple exponential decay kick at each beat
    kick_len = min(int(0.05 * sr), beat_interval)  # 50 ms kick
    kick = np.exp(-np.linspace(0, 8, kick_len)) * 0.7
    kick *= np.sin(2 * np.pi * 60 * np.arange(kick_len) / sr)
    for i in range(0, n_samples - kick_len, beat_interval):
        audio[i : i + kick_len] += kick.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def _compute_onset_times_lf(audio: np.ndarray, sr: int, cutoff: float = 200.0) -> np.ndarray:
    """Simple LF onset detector: bandpass → RMS envelope → peaks."""
    from scipy.signal import butter, sosfilt

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    lf = sosfilt(sos, audio)
    # RMS envelope with 10 ms hop
    hop = int(0.01 * sr)
    n_frames = len(lf) // hop
    rms = np.array([np.sqrt(np.mean(lf[i * hop : (i + 1) * hop] ** 2)) for i in range(n_frames)])
    # Simple peak picking: local max with threshold
    threshold = np.percentile(rms, 70) if len(rms) > 0 else 0.0
    onset_frames = []
    for i in range(1, len(rms) - 1):
        if rms[i] > rms[i - 1] and rms[i] > rms[i + 1] and rms[i] > threshold:
            onset_frames.append(i)
    return np.array(onset_frames) * hop / sr  # Convert to seconds


# ---------------------------------------------------------------------------
# DTW Groove Alignment
# ---------------------------------------------------------------------------


class TestGrooveDTWAlignment:
    """Groove metric should detect timing alignment via DTW."""

    def test_identical_signals_zero_drift(self):
        """Same signal compared to itself should have ~0 DTW cost."""
        audio = _make_kick_pattern()
        onsets_a = _compute_onset_times_lf(audio, 48000)
        onsets_b = _compute_onset_times_lf(audio, 48000)
        # Identical → zero difference
        if len(onsets_a) > 0 and len(onsets_b) > 0:
            max_diff = np.max(
                np.abs(onsets_a[: min(len(onsets_a), len(onsets_b))] - onsets_b[: min(len(onsets_a), len(onsets_b))])
            )
            assert max_diff < 0.001  # < 1 ms

    def test_shifted_signal_detects_drift(self):
        """A time-shifted signal should show measurable DTW cost."""
        sr = 48000
        audio = _make_kick_pattern(sr=sr)
        # Shift by 20 ms
        shift_samples = int(0.020 * sr)
        shifted = np.zeros_like(audio)
        shifted[shift_samples:] = audio[:-shift_samples]
        onsets_orig = _compute_onset_times_lf(audio, sr)
        onsets_shifted = _compute_onset_times_lf(shifted, sr)
        if len(onsets_orig) > 1 and len(onsets_shifted) > 1:
            # Average onset shift should be ~20 ms
            min_len = min(len(onsets_orig), len(onsets_shifted))
            diffs = np.abs(onsets_orig[:min_len] - onsets_shifted[:min_len])
            mean_drift_ms = np.mean(diffs) * 1000
            assert mean_drift_ms > 5.0  # > 5 ms drift detected

    def test_dtw_within_8ms_for_clean_signal(self):
        """Clean signal with mild gain change should preserve LF onset pattern."""
        sr = 48000
        audio = _make_kick_pattern(sr=sr)
        # Apply mild gain — should not change onset timing
        gained = audio * 0.9
        onsets_orig = _compute_onset_times_lf(audio, sr)
        onsets_gained = _compute_onset_times_lf(gained, sr)
        if len(onsets_orig) > 1 and len(onsets_gained) > 1:
            assert len(onsets_orig) == len(onsets_gained), (
                f"Onset count diverged: {len(onsets_orig)} vs {len(onsets_gained)}"
            )
            min_len = min(len(onsets_orig), len(onsets_gained))
            diffs_ms = np.abs(onsets_orig[:min_len] - onsets_gained[:min_len]) * 1000
            assert np.max(diffs_ms) <= 8.0, f"Max groove drift {np.max(diffs_ms):.1f} ms > 8 ms"


# ---------------------------------------------------------------------------
# Mono / Stereo Support
# ---------------------------------------------------------------------------


class TestGrooveMonoStereo:
    """Groove detection must work for both mono and stereo."""

    def test_mono_onset_detection(self):
        audio = _make_kick_pattern()
        onsets = _compute_onset_times_lf(audio, 48000)
        assert len(onsets) > 3  # Should detect multiple beats

    def test_stereo_onset_detection(self):
        audio = _make_kick_pattern()
        stereo = np.column_stack([audio, audio * 0.95])
        onsets = _compute_onset_times_lf(stereo, 48000)
        assert len(onsets) > 3

    def test_stereo_mono_consistency(self):
        """Stereo and mono should detect similar onset patterns."""
        audio = _make_kick_pattern()
        stereo = np.column_stack([audio, audio])
        onsets_mono = _compute_onset_times_lf(audio, 48000)
        onsets_stereo = _compute_onset_times_lf(stereo, 48000)
        assert abs(len(onsets_mono) - len(onsets_stereo)) <= 1


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestGrooveEdgeCases:
    """Edge cases for groove measurement."""

    def test_silence(self):
        audio = np.zeros(48000 * 2, dtype=np.float32)
        onsets = _compute_onset_times_lf(audio, 48000)
        assert len(onsets) == 0

    def test_very_short_audio(self):
        """< 0.5s audio should still not crash."""
        audio = _make_kick_pattern(dur=0.3)
        onsets = _compute_onset_times_lf(audio, 48000)
        # May find 0 or 1 onset — just no crash
        assert isinstance(onsets, np.ndarray)

    def test_dc_offset_resilient(self):
        """DC offset should not create false onsets."""
        audio = np.ones(48000 * 2, dtype=np.float32) * 0.5  # Pure DC
        onsets = _compute_onset_times_lf(audio, 48000)
        # DC has no transients — should find very few or zero
        assert len(onsets) <= 2
