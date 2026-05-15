"""Tests for preventive Pegelexplosion protection.

§2.30b Stufe 5 — per-sample quiet-zone hard clamp in apply_musical_gain_envelope.
check_gain_safety() — pre-flight gain ceiling check.
UV3._musical_gain_envelope delegation to canonical function.
"""

import numpy as np

from backend.core.audio_utils import apply_musical_gain_envelope, check_gain_safety

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 48000
FRAME = 480  # 10 ms @ 48 kHz


def _make_vinyl_signal(duration_s: float = 2.0) -> np.ndarray:
    """Music body at -20 dBFS + fadeout (last 0.5 s) at -40 dBFS (vinyl surface noise)."""
    n = int(duration_s * SR)
    sig = np.zeros(n, dtype=np.float32)
    music_end = int((duration_s - 0.5) * SR)
    t = np.linspace(0, duration_s - 0.5, music_end)
    sig[:music_end] = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)  # -20 dBFS ≈ 0.1
    # Fadeout: vinyl surface noise at -40 dBFS (≈ 0.01 linear)
    noise_amp = 10.0 ** (-40.0 / 20.0)
    rng = np.random.RandomState(42)
    sig[music_end:] = (rng.randn(n - music_end) * noise_amp).astype(np.float32)
    return sig


def _rms_dbfs(x: np.ndarray) -> float:
    return float(20.0 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12))


# ---------------------------------------------------------------------------
# §2.30b Stufe 5 — per-sample quiet-zone hard clamp
# ---------------------------------------------------------------------------


def test_01_quiet_zone_never_amplified():
    """Fadeout frames at -40 dBFS must NOT be boosted even with gain=4.0."""
    sig = _make_vinyl_signal()
    fadeout = sig[int(1.5 * SR) :]  # last 0.5 s
    rms_before = _rms_dbfs(fadeout)

    result = apply_musical_gain_envelope(sig, gain=4.0, gate_dbfs=-36.0, sr=SR)

    fadeout_after = result[int(1.5 * SR) :]
    rms_after = _rms_dbfs(fadeout_after)
    # Fadeout must not be amplified at all (allow 0.5 dB for boundary crossfade)
    assert rms_after <= rms_before + 0.5, (
        f"Fadeout Pegelexplosion: {rms_before:.1f} dBFS → {rms_after:.1f} dBFS (delta={rms_after - rms_before:.1f} dB)"
    )


def test_02_musical_frames_still_boosted():
    """Musical frames (> -36 dBFS) must still receive the gain."""
    sig = _make_vinyl_signal()
    music_body = sig[: int(0.5 * SR)]
    rms_before = _rms_dbfs(music_body)

    result = apply_musical_gain_envelope(sig, gain=2.0, gate_dbfs=-36.0, sr=SR)

    music_after = result[: int(0.5 * SR)]
    rms_after = _rms_dbfs(music_after)
    # Should be roughly +6 dB (gain=2.0), allow ±3 dB
    assert rms_after >= rms_before + 3.0, f"Musical frames not boosted: {rms_before:.1f} → {rms_after:.1f} dBFS"


def test_03_smoothing_bleed_clamped():
    """After box-blur smoothing, boundary samples of quiet frames must be clamped."""
    # Create: 0.5 s music + 0.5 s silence at -42 dBFS
    n_music = int(0.5 * SR)
    n_quiet = int(0.5 * SR)
    sig = np.zeros(n_music + n_quiet, dtype=np.float32)
    t = np.linspace(0, 0.5, n_music)
    sig[:n_music] = (np.sin(2 * np.pi * 440 * t) * 0.08).astype(np.float32)
    rng = np.random.RandomState(7)
    sig[n_music:] = (rng.randn(n_quiet) * 10.0 ** (-42.0 / 20.0)).astype(np.float32)

    # Large gain to make the bleed obvious if not clamped
    result = apply_musical_gain_envelope(sig, gain=8.0, gate_dbfs=-36.0, sr=SR)

    # Quiet zone: every sample in the quiet region must not exceed the original * 1.05
    quiet_out = result[n_music + FRAME :]  # skip first frame (could be crossfade)
    quiet_in = sig[n_music + FRAME :]
    # No sample should have been amplified in the quiet zone
    assert np.all(np.abs(quiet_out) <= np.abs(quiet_in) * 1.05 + 1e-7), "Smoothing bleed into quiet zone not clamped"


def test_04_stereo_channels_first_quiet_zone():
    """Stereo (2×N channels-first) — quiet frames clamped in both channels."""
    sig = _make_vinyl_signal()
    stereo = np.stack([sig, sig * 0.9], axis=0)  # shape (2, N)
    result = apply_musical_gain_envelope(stereo, gain=3.0, gate_dbfs=-36.0, sr=SR)
    assert result.shape == stereo.shape
    # Fadeout region
    fadeout_in = stereo[:, int(1.5 * SR) :]
    fadeout_out = result[:, int(1.5 * SR) :]
    assert np.all(np.abs(fadeout_out) <= np.abs(fadeout_in) + 0.01), "Stereo channels-first: quiet zone amplified"


def test_05_stereo_samples_first_quiet_zone():
    """Stereo (N×2 samples-first) — quiet frames clamped."""
    sig = _make_vinyl_signal()
    stereo = np.stack([sig, sig * 0.85], axis=1)  # shape (N, 2)
    result = apply_musical_gain_envelope(stereo, gain=3.0, gate_dbfs=-36.0, sr=SR)
    assert result.shape == stereo.shape
    fadeout_in = stereo[int(1.5 * SR) :, :]
    fadeout_out = result[int(1.5 * SR) :, :]
    assert np.all(np.abs(fadeout_out) <= np.abs(fadeout_in) + 0.01), "Stereo samples-first: quiet zone amplified"


# ---------------------------------------------------------------------------
# check_gain_safety() — pre-flight
# ---------------------------------------------------------------------------


def test_06_preflight_passes_safe_gain():
    """Low-level signal + small gain → no clamping."""
    sig = np.ones(1024, dtype=np.float32) * 0.1  # -20 dBFS
    safe_gain, clamped = check_gain_safety(sig, requested_gain=2.0, max_peak_dbfs=-1.0)
    assert not clamped
    assert abs(safe_gain - 2.0) < 1e-6


def test_07_preflight_clamps_dangerous_gain():
    """Near-full-scale signal + large gain → clamped, safe_gain < requested_gain."""
    # Signal at -6 dBFS (0.5 linear); gain=4.0 would push to 2.0 (clipping)
    # Expected safe_gain = max_lin / 0.5 ≈ 0.891/0.5 ≈ 1.78 < 4.0
    sig = np.ones(1024, dtype=np.float32) * 0.5  # -6 dBFS
    safe_gain, clamped = check_gain_safety(sig, requested_gain=4.0, max_peak_dbfs=-1.0)
    assert clamped
    assert safe_gain < 4.0
    # Result must not push peak above max_peak_dbfs
    max_lin = float(10.0 ** (-1.0 / 20.0))
    assert safe_gain * 0.5 <= max_lin + 1e-6


def test_08_preflight_silent_returns_unity():
    """Silent signal → gain 1.0 to prevent boosting absolute silence."""
    sig = np.zeros(1024, dtype=np.float32)
    safe_gain, clamped = check_gain_safety(sig, requested_gain=3.0)
    assert clamped
    assert safe_gain == 1.0


def test_09_preflight_attenuation_not_changed():
    """Gain < 1 (attenuation) is never clamped."""
    sig = np.ones(256, dtype=np.float32) * 0.5
    safe_gain, clamped = check_gain_safety(sig, requested_gain=0.5)
    assert not clamped
    assert abs(safe_gain - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# UV3 _musical_gain_envelope — delegates to canonical
# ---------------------------------------------------------------------------


def test_10_uv3_delegates_to_canonical():
    """UV3._musical_gain_envelope must produce identical output to apply_musical_gain_envelope."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    sig = _make_vinyl_signal(1.0)
    gain = 3.0

    result_canon = apply_musical_gain_envelope(sig, gain, gate_dbfs=-36.0, sr=SR)
    result_uv3 = UnifiedRestorerV3._musical_gain_envelope(sig, gain, gate_dbfs=-36.0, sr=SR)

    np.testing.assert_array_almost_equal(
        result_canon,
        result_uv3,
        decimal=6,
        err_msg="UV3._musical_gain_envelope diverges from canonical apply_musical_gain_envelope",
    )


def test_11_uv3_quiet_zone_also_clamped():
    """UV3._musical_gain_envelope inherits the per-sample quiet-zone clamp via delegation."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    sig = _make_vinyl_signal()
    fadeout_before = sig[int(1.5 * SR) :]
    rms_before = _rms_dbfs(fadeout_before)

    result = UnifiedRestorerV3._musical_gain_envelope(sig, gain=5.0, gate_dbfs=-36.0, sr=SR)

    rms_after = _rms_dbfs(result[int(1.5 * SR) :])
    assert rms_after <= rms_before + 0.5, (
        f"UV3 delegation: fadeout Pegelexplosion {rms_before:.1f} → {rms_after:.1f} dBFS"
    )


def test_12_quiet_edge_reference_clamps_intentionally_quiet_intro_outro():
    """Boost protection must compare processed edges against the original song edges."""
    n = int(8.0 * SR)
    t = np.linspace(0.0, 8.0, n, endpoint=False)
    music = (0.18 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    reference = music.copy()
    reference[: int(1.0 * SR)] *= 0.10
    reference[-int(1.0 * SR) :] *= 0.08

    processed = music.copy()
    processed[: int(1.0 * SR)] *= 0.35
    processed[-int(1.0 * SR) :] *= 0.32

    out = apply_musical_gain_envelope(
        processed,
        gain=2.5,
        gate_dbfs=-36.0,
        sr=SR,
        reference_for_gate=reference,
    )

    intro_ref = _rms_dbfs(reference[: int(1.0 * SR)])
    intro_out = _rms_dbfs(out[: int(1.0 * SR)])
    outro_ref = _rms_dbfs(reference[-int(1.0 * SR) :])
    outro_out = _rms_dbfs(out[-int(1.0 * SR) :])
    body_ref = _rms_dbfs(reference[int(2.0 * SR) : int(6.0 * SR)])
    body_out = _rms_dbfs(out[int(2.0 * SR) : int(6.0 * SR)])

    assert intro_out <= intro_ref + 2.05, f"Quiet intro was over-boosted: {intro_out - intro_ref:.1f} dB"
    assert outro_out <= outro_ref + 2.05, f"Quiet outro was over-boosted: {outro_out - outro_ref:.1f} dB"
    assert body_out >= body_ref + 2.0, "Music body should still receive meaningful gain"


def test_13_stereo_right_channel_quiet_edges_are_clamped_independently():
    """A single over-boosted stereo channel at intro/outro must be clamped on that channel too."""
    n = int(8.0 * SR)
    t = np.linspace(0.0, 8.0, n, endpoint=False)
    base = (0.18 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    left_ref = base.copy()
    right_ref = (base * 0.92).astype(np.float32)
    left_ref[: int(1.0 * SR)] *= 0.10
    right_ref[: int(1.0 * SR)] *= 0.10
    left_ref[-int(1.0 * SR) :] *= 0.08
    right_ref[-int(1.0 * SR) :] *= 0.08
    reference = np.stack([left_ref, right_ref], axis=0)

    processed = reference.copy()
    processed[1, : int(1.0 * SR)] *= 3.8
    processed[1, -int(1.0 * SR) :] *= 3.4

    out = apply_musical_gain_envelope(
        processed,
        gain=2.5,
        gate_dbfs=-36.0,
        sr=SR,
        reference_for_gate=reference,
    )

    right_intro_ref = _rms_dbfs(reference[1, : int(1.0 * SR)])
    right_intro_out = _rms_dbfs(out[1, : int(1.0 * SR)])
    right_outro_ref = _rms_dbfs(reference[1, -int(1.0 * SR) :])
    right_outro_out = _rms_dbfs(out[1, -int(1.0 * SR) :])
    body_ref = _rms_dbfs(reference[1, int(2.0 * SR) : int(6.0 * SR)])
    body_out = _rms_dbfs(out[1, int(2.0 * SR) : int(6.0 * SR)])

    assert right_intro_out <= right_intro_ref + 2.05, (
        f"Right intro was over-boosted: {right_intro_out - right_intro_ref:.1f} dB"
    )
    assert right_outro_out <= right_outro_ref + 2.05, (
        f"Right outro was over-boosted: {right_outro_out - right_outro_ref:.1f} dB"
    )
    assert body_out >= body_ref + 2.0, "Right channel music body should still receive meaningful gain"
