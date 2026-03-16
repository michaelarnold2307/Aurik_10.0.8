"""
Unit-Tests für DSP-neugeschriebene Phasen mit PhaseResult-Return-Fix.

Abgedeckte Phasen:
  Phase 10  — Multi-Band Parallel Compression (PhaseResult-Return-Fix)
  Phase 34  — Mid/Side Processing (PhaseResult-Return-Fix)

Alle Tests laufen auf synthetischen Arrays (44100 Hz).
Phasen 43–53 werden kanonisch in test_phases_late_ext.py getestet.
"""

import numpy as np
np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.phases.phase_interface import PhaseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 44100
_N = SR // 4  # 11025 Samples (0.25s) — statt 1s vermeidet unnötige DSP-Arbeit


@pytest.fixture(scope="class")
def mono():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal(_N).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="class")
def stereo():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal((_N, 2)).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="class")
def stereo_quiet():
    """Sehr leises Stereo-Signal (für De-Esser über Schwelle)."""
    rng = np.random.default_rng(99)
    return np.clip(rng.standard_normal((_N, 2)).astype(np.float32) * 0.05, -1.0, 1.0)


def _assert_phase_result(result, orig_audio, check_clipping: bool = True):
    """Gemeinsame Validierung für alle PhaseResult-Objekte."""
    assert isinstance(result, PhaseResult), f"Erwartet PhaseResult, got {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Audio-Shape geändert: {orig_audio.shape} → {result.audio.shape}"
    assert result.audio.dtype == orig_audio.dtype, f"Dtype geändert: {orig_audio.dtype} → {result.audio.dtype}"
    assert isinstance(result.metadata, dict), "metadata muss dict sein"
    assert isinstance(result.metrics, dict), "metrics muss dict sein"
    assert float(result.execution_time_seconds) >= 0.0, "execution_time muss ≥ 0 sein"
    if check_clipping:
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-4, "Clipping: Audio > 1.0"


# ---------------------------------------------------------------------------
# Phase 10: Multi-Band Compression
# ---------------------------------------------------------------------------


class TestPhase10Compression:
    def setup_method(self):
        from backend.core.phases.phase_10_compression import CompressionPhase

        self.phase = CompressionPhase()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)  # Makeup-Gain kann > 1.0

    def test_metadata_keys(self, stereo):
        result = self.phase.process(stereo, SR)
        assert "rms_change_db" in result.metrics or "rms_change_db" in result.metadata

    def test_rms_changes(self, stereo):
        result = self.phase.process(stereo, SR)
        # Kompressor sollte RMS leicht reduzieren (negative rms_change_db)
        rms_change = result.metrics.get("rms_change_db", result.metadata.get("rms_change_db", 0))
        assert float(rms_change) <= 2.0, "Kompressor erhöht RMS stark — ungewöhnlich"


# ---------------------------------------------------------------------------
# Phase 34: Mid/Side Processing
# ---------------------------------------------------------------------------


class TestPhase34MidSide:
    def setup_method(self):
        from backend.core.phases.phase_34_mid_side_processing import MidSideProcessing

        self.phase = MidSideProcessing()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)  # M/S-Dynamik kann > 1.0

    def test_mono_compatibility_metric(self, stereo):
        result = self.phase.process(stereo, SR)
        mc = result.metrics.get("mono_compatibility")
        assert mc is not None, "mono_compatibility-Metrik fehlt"
        assert 0.0 <= float(mc) <= 1.5, f"mono_compatibility außerhalb Bereich: {mc}"
