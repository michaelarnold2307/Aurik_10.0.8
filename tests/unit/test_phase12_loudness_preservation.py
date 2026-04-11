import numpy as np

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


def _rms_db(x: np.ndarray) -> float:
    xr = np.asarray(x, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(xr**2) + 1e-12)))


def test_phase12_loudness_guard_limits_large_drop() -> None:
    phase = WowFlutterFix()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    original = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    processed = (original * 0.20).astype(np.float32)  # ~ -14 dB drop

    corrected, delta_db, makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.VINYL)

    assert corrected.shape == original.shape
    assert np.isfinite(corrected).all()
    assert delta_db >= -1.7  # vinyl cap + small numeric tolerance
    assert makeup_db > 0.0


def test_phase12_loudness_guard_limits_over_lift() -> None:
    phase = WowFlutterFix()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    original = (0.20 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    processed = (original * 2.0).astype(np.float32)  # strong lift

    corrected, delta_db, makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.TAPE)

    assert corrected.shape == original.shape
    assert np.isfinite(corrected).all()
    assert delta_db <= 1.1
    assert makeup_db < 0.0
    # Ensure guard actually changed output level relative to input processed signal.
    assert _rms_db(corrected) < _rms_db(processed)


def test_phase12_safe_timing_profile_dampens_vocal_vinyl() -> None:
    phase = WowFlutterFix()

    strength_scale, max_stretch_delta = phase._derive_safe_timing_profile(
        MaterialType.VINYL,
        mean_confidence=0.55,
        vocals_confidence=0.75,
    )

    assert strength_scale < 1.0
    assert max_stretch_delta <= 0.03


def test_phase12_stretch_factors_respect_safe_delta() -> None:
    phase = WowFlutterFix()
    pitch = np.array([220.0, 228.0, 212.0, 224.0, 216.0, 230.0, 210.0, 222.0, 218.0, 226.0, 214.0], dtype=np.float64)
    conf = np.ones_like(pitch, dtype=np.float64) * 0.95

    stretch = phase._calculate_stretch_factors(pitch, conf, 1.0, max_stretch_delta=0.03)

    assert np.isfinite(stretch).all()
    assert float(np.min(stretch)) >= 0.97 - 1e-6
    assert float(np.max(stretch)) <= 1.03 + 1e-6


def test_phase12_polyphonic_fallback_tightens_timing_profile() -> None:
    phase = WowFlutterFix()

    strength_scale, max_stretch_delta = phase._derive_safe_timing_profile(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.65,
        polyphonic_fallback=True,
    )

    assert strength_scale < 0.82
    assert max_stretch_delta <= 0.02


def test_phase12_unsafe_polyphonic_fallback_is_bypassed_for_vocal_vinyl() -> None:
    phase = WowFlutterFix()

    assert phase._should_bypass_unsafe_polyphonic_fallback(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.65,
        polyphonic_fallback=True,
    )


def test_phase12_unsafe_polyphonic_fallback_not_bypassed_for_nonvocal() -> None:
    phase = WowFlutterFix()

    assert not phase._should_bypass_unsafe_polyphonic_fallback(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.10,
        polyphonic_fallback=True,
    )
