"""Unit tests — §2.47 Phase_29 SNR > 35 dB Dry-Signal Bypass.

Verifies that TapeHissReductionPhase skips OMLSA processing and returns
the unmodified audio when the estimated SNR exceeds 35 dB.  This is the
§2.47 RELEASE_MUST 'clean signal → dry bypass' invariant.
"""
from __future__ import annotations

import numpy as np
import pytest

SR = 48_000
DURATION_S = 3  # seconds of audio in tests


def _clean_mono(snr_db: float = 50.0) -> np.ndarray:
    """Return a mono sine wave with the specified broadband SNR (signal/noise power ratio)."""
    rng = np.random.default_rng(42)
    t = np.arange(SR * DURATION_S, dtype=np.float32) / SR
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Derive noise amplitude from desired SNR:  SNR_dB = 10*log10(P_s / P_n)
    signal_power = float(np.mean(signal**2))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_amp = float(np.sqrt(noise_power))
    noise = rng.normal(0.0, noise_amp, size=signal.shape).astype(np.float32)
    return (signal + noise).clip(-1.0, 1.0)


def _noisy_mono(snr_db: float = 10.0) -> np.ndarray:
    """Return a mono signal with low SNR to ensure phase_29 processes it."""
    return _clean_mono(snr_db=snr_db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhase29SnrBypass:
    """§2.47 SNR > 35 dB bypass invariant."""

    def test_high_snr_signal_is_bypassed(self) -> None:
        """Clean audio (SNR ≈ 50 dB) must be returned as-is with snr_bypass=True."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.defect_scanner import MaterialType

        audio = _clean_mono(snr_db=50.0)
        phase = TapeHissReductionPhase()
        result = phase.process(audio, SR, material=MaterialType.REEL_TAPE, strength=0.8)

        assert result.success, "Phase must report success on SNR bypass"
        meta = result.metadata or {}
        assert meta.get("snr_bypass") is True, "metadata['snr_bypass'] must be True for high-SNR audio"
        assert meta.get("processing") == "snr_bypass", "metadata['processing'] must be 'snr_bypass'"
        np.testing.assert_allclose(result.audio, np.clip(audio, -1.0, 1.0), atol=1e-5)

    def test_low_snr_signal_is_not_bypassed(self) -> None:
        """Noisy audio (SNR ≈ 10 dB) must NOT receive the SNR bypass."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.defect_scanner import MaterialType

        audio = _noisy_mono(snr_db=10.0)
        phase = TapeHissReductionPhase()

        result = phase.process(audio, SR, material=MaterialType.REEL_TAPE, strength=0.5)

        # snr_bypass key must either be absent or False
        meta = result.metadata or {}
        assert not meta.get("snr_bypass", False), (
            "Low-SNR audio must not be bypassed — tape hiss reduction should run"
        )

    def test_snr_bypass_for_stereo(self) -> None:
        """The bypass must work identically for stereo (2D) audio."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.defect_scanner import MaterialType

        mono = _clean_mono(snr_db=55.0)
        stereo = np.stack([mono, mono], axis=1)  # shape (N, 2)

        phase = TapeHissReductionPhase()
        result = phase.process(stereo, SR, material=MaterialType.TAPE, strength=0.8)

        assert result.success
        meta = result.metadata or {}
        assert meta.get("snr_bypass") is True, "Stereo high-SNR audio must also be bypassed"
        assert result.audio.ndim == 2, "Stereo output must remain stereo"

    def test_snr_bypass_rms_drop_is_zero(self) -> None:
        """Bypassed phase must report rms_drop_db=0.0 (§2.45a telemetry invariant)."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.defect_scanner import MaterialType

        audio = _clean_mono(snr_db=45.0)
        phase = TapeHissReductionPhase()
        result = phase.process(audio, SR, material=MaterialType.REEL_TAPE, strength=1.0)

        meta = result.metadata or {}
        if meta.get("snr_bypass"):
            assert meta.get("rms_drop_db", 0.0) == 0.0, "Bypassed phase must not report RMS drop"

    def test_digital_source_skipped_before_snr_check(self) -> None:
        """CD_DIGITAL material is skipped before reaching the SNR estimation block."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.defect_scanner import MaterialType

        audio = _noisy_mono(snr_db=5.0)  # very noisy — would NOT trigger SNR bypass
        phase = TapeHissReductionPhase()

        result = phase.process(audio, SR, material=MaterialType.CD_DIGITAL, strength=0.8)

        assert result.success
        meta = result.metadata or {}
        # Should be digital-source skip (not snr_bypass)
        processing = meta.get("processing", "")
        assert processing == "skipped", (
            f"CD_DIGITAL should use 'skipped' path, got: {processing}"
        )
        assert not meta.get("snr_bypass", False), "CD_DIGITAL skip ≠ SNR bypass path"
