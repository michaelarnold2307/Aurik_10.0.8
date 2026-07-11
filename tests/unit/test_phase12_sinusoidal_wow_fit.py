from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 48_000


def _pitch_from_cents(base_hz: float, cents: np.ndarray) -> np.ndarray:
    return (base_hz * np.power(2.0, cents / 1200.0)).astype(np.float64)


@pytest.mark.unit
def test_sinusoidal_wow_fit_reduces_noisy_transport_curve_error() -> None:
    phase = WowFlutterFix()
    frame_rate = SR / ((phase.PITCH_WINDOW_MS * SR // 1000) // phase.PITCH_HOP_FACTOR)
    n_frames = 240
    t = np.arange(n_frames, dtype=np.float64) / frame_rate
    true_cents = 18.0 * np.sin(2.0 * np.pi * 1.0 * t + 0.4)
    rng = np.random.default_rng(12)
    noisy_cents = true_cents + rng.normal(0.0, 3.0, size=n_frames)
    pitch = _pitch_from_cents(220.0, noisy_cents)
    confidence = np.full(n_frames, 0.92, dtype=np.float64)

    fitted_pitch, profile = phase._fit_sinusoidal_wow_curve(pitch, confidence, SR)

    fitted_cents = 1200.0 * np.log2(fitted_pitch / np.median(fitted_pitch))
    before_rmse = float(np.sqrt(np.mean((noisy_cents - true_cents) ** 2)))
    after_rmse = float(np.sqrt(np.mean((fitted_cents - true_cents) ** 2)))

    assert profile["applied"] is True
    assert 0.85 <= profile["frequency_hz"] <= 1.15
    assert 12.0 <= profile["amplitude_cents"] <= 24.0
    assert profile["r2"] >= 0.70
    assert after_rmse < before_rmse * 0.60


def test_sinusoidal_wow_fit_bypasses_melodic_pitch_span() -> None:
    phase = WowFlutterFix()
    low = np.full(80, 220.0, dtype=np.float64)
    high = np.full(80, 246.94165, dtype=np.float64)  # ca. +200 cents
    pitch = np.concatenate([low, high])
    confidence = np.full_like(pitch, 0.95)

    fitted_pitch, profile = phase._fit_sinusoidal_wow_curve(pitch, confidence, SR)

    assert profile["applied"] is False
    np.testing.assert_allclose(fitted_pitch, pitch)
