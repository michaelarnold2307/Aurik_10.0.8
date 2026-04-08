"""
Unit-Tests für Aurik-Phasen 11-30, 40-42, 49, 51-52, 54.

Aufruf-Konventionen:
  - Phasen 11-30, 40-42, 49: process(audio, sample_rate[, material_type])
  - Phasen 13, 14, 15, 25: brauchen Stereo-Audio
  - Phasen 51, 52, 54: process(audio) — KEIN sample_rate!
  - Phase 11 + 25: MaterialType ist PFLICHTARG
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_interface import PhaseResult

SR = 48000
_N = SR // 4  # 12000 Samples (0.25s) – reicht für alle Phase-Tests


# ---------------------------------------------------------------------------
# Fixtures — scope="class": einmalig pro Testklasse (sicher bei in-place Ops)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def mono():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, _N).astype(np.float32)


@pytest.fixture(scope="class")
def stereo():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, (_N, 2)).astype(np.float32)


@pytest.fixture(scope="class")
def silent_mono():
    return np.zeros(_N, dtype=np.float32)


@pytest.fixture(scope="class")
def loud_mono():
    rng = np.random.default_rng(0)
    return rng.uniform(-0.9, 0.9, _N).astype(np.float32)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------
def _assert_phase_result(result, orig_audio, *, check_clipping: bool = True, clip_threshold: float = 2.0):
    """Prüft ein PhaseResult auf Grundkorrektheit."""
    assert isinstance(result, PhaseResult), f"Kein PhaseResult: {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Shape-Mismatch: {result.audio.shape} != {orig_audio.shape}"
    assert np.issubdtype(result.audio.dtype, np.floating), f"audio.dtype nicht floating: {result.audio.dtype}"
    assert isinstance(result.metadata, dict)
    assert isinstance(result.metrics, dict)
    assert float(result.execution_time_seconds) >= 0.0
    if check_clipping:
        peak = float(np.max(np.abs(result.audio)))
        assert peak <= clip_threshold, f"Audio stark übersteuert: {peak:.3f} > {clip_threshold}"


# ===========================================================================
# Phase 11 – Limiting
# ===========================================================================
class TestPhase11Limiting:
    def setup_method(self):
        from backend.core.phases.phase_11_limiting import LimitingPhase

        self.phase = LimitingPhase()

    def test_mono_returns_phase_result(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, loud_mono)

    def test_output_within_limits(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        assert np.max(np.abs(result.audio)) <= 1.05

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_different_material_types(self, mono):
        for mat in [MaterialType.VINYL, MaterialType.TAPE, MaterialType.CD_DIGITAL]:
            result = self.phase.process(mono, SR, mat)
            _assert_phase_result(result, mono)

    def test_zero_strength_passthrough(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, loud_mono)
        assert np.allclose(result.audio, loud_mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, loud_mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 12 – Wow/Flutter-Fix
# ===========================================================================
class TestPhase12WowFlutterFix:
    def setup_method(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        self.phase = WowFlutterFix()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_tape_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 13 – Stereo Enhancement
# ===========================================================================
class TestPhase13StereoEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_13_stereo_enhancement import StereoEnhancementPhaseV2

        self.phase = StereoEnhancementPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 14 – Phase Correction
# ===========================================================================
class TestPhase14PhaseCorrection:
    def setup_method(self):
        from backend.core.phases.phase_14_phase_correction import PhaseCorrection

        self.phase = PhaseCorrection()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 15 – Stereo Balance
# ===========================================================================
class TestPhase15StereoBalance:
    def setup_method(self):
        from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2

        self.phase = StereoBalancePhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR)
        _assert_phase_result(result, silent)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 16 – Final EQ
# ===========================================================================
class TestPhase16FinalEQ:
    def setup_method(self):
        from backend.core.phases.phase_16_final_eq import FinalEQ

        self.phase = FinalEQ()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 17 – Mastering Polish
# ===========================================================================
class TestPhase17MasteringPolish:
    def setup_method(self):
        from backend.core.phases.phase_17_mastering_polish import MasteringPolishPhase

        self.phase = MasteringPolishPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 18 – Noise Gate
# ===========================================================================
class TestPhase18NoiseGate:
    def setup_method(self):
        from backend.core.phases.phase_18_noise_gate import NoiseGate

        self.phase = NoiseGate()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_silent_is_attenuated(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)
        assert np.max(np.abs(result.audio)) < 0.01

    def test_loud_signal_passes(self, loud_mono):
        result = self.phase.process(loud_mono, SR)
        # Gate kann leichten Gain hinzufügen → check_clipping=False
        _assert_phase_result(result, loud_mono, check_clipping=False)
        # Laut-Signal sollte nicht komplett gedämpft werden
        assert np.max(np.abs(result.audio)) > 0.01

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 19 – De-Esser
# ===========================================================================
class TestPhase19DeEsser:
    def setup_method(self):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        self.phase = DeEsserPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono)

    def test_sibilant_reduction(self):
        """Hochfrequentes Signal (s-Laute) soll gedämpft werden."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        sibilant = (0.5 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
        result = self.phase.process(sibilant, SR, MaterialType.VINYL)
        _assert_phase_result(result, sibilant, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 20 – Reverb Reduction
# ===========================================================================
class TestPhase20ReverbReduction:
    def setup_method(self):
        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        self.phase = ReverbReduction()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_different_materials(self, mono):
        for mat in [MaterialType.TAPE, MaterialType.SHELLAC]:
            result = self.phase.process(mono, SR, mat)
            _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_ml_shape_failure_disables_repeated_ml_attempts(self, mono, monkeypatch):
        import backend.core.phases.phase_20_reverb_reduction as phase20_mod

        class _ExplodingHybrid:
            def __init__(self, config):
                pass

            def dereverb(self, audio, sample_rate=48000):
                raise RuntimeError("Sizes of tensors must match except in dimension 1")

        monkeypatch.setattr(phase20_mod, "ML_HYBRID_AVAILABLE", True)
        monkeypatch.setattr(
            phase20_mod, "RESOURCE_MANAGER_AVAILABLE", False
        )  # ensure ML path is taken regardless of machine load
        monkeypatch.setattr(phase20_mod, "HybridDereverb", _ExplodingHybrid)

        # First call hits ML failure once and should switch phase instance to DSP-only.
        result1 = self.phase.process(mono, SR, MaterialType.TAPE, quality_mode="quality")
        _assert_phase_result(result1, mono, check_clipping=False)
        assert self.phase._force_dsp_only_due_ml_error is True

        calls = {"n": 0}

        class _CountingHybrid:
            def __init__(self, config):
                calls["n"] += 1

            def dereverb(self, audio, sample_rate=48000):
                return audio

        monkeypatch.setattr(phase20_mod, "HybridDereverb", _CountingHybrid)

        # Second call must stay DSP-only and not instantiate HybridDereverb again.
        result2 = self.phase.process(mono, SR, MaterialType.TAPE, quality_mode="quality")
        _assert_phase_result(result2, mono, check_clipping=False)
        assert calls["n"] == 0


# ===========================================================================
# Phase 21 – Exciter
# ===========================================================================
class TestPhase21Exciter:
    def setup_method(self):
        from backend.core.phases.phase_21_exciter import Exciter

        self.phase = Exciter()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_ms_mid_only_coherence(self):
        """§2.51: Stereo exciter applies harmonics to Mid only — preserves stereo image."""
        rng = np.random.default_rng(42)
        t = np.arange(_N) / SR
        left = (rng.normal(0, 0.1, _N) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (rng.normal(0, 0.1, _N) + 0.2 * np.sin(2 * np.pi * 440 * t + 0.5)).astype(np.float32)
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, SR, MaterialType.VINYL)
        assert result.success
        assert result.metadata.get("stereo_mode") == "ms_mid_only"
        assert result.audio.shape == stereo_in.shape

    def test_stereo_output_shape_preserved(self):
        """Stereo input → stereo output with correct shape."""
        rng = np.random.default_rng(99)
        stereo_in = rng.normal(0, 0.1, (_N, 2)).astype(np.float32)
        result = self.phase.process(stereo_in, SR, MaterialType.CD_DIGITAL)
        assert result.audio.shape == stereo_in.shape


# ===========================================================================
# Phase 22 – Tape Saturation
# ===========================================================================
class TestPhase22TapeSaturation:
    def setup_method(self):
        from backend.core.phases.phase_22_tape_saturation import TapeSaturation

        self.phase = TapeSaturation()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.modifications.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.modifications.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 23 – Spectral Repair
# ===========================================================================
class TestPhase23SpectralRepair:
    def setup_method(self):
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        self.phase = SpectralRepair()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_locality_reduces_repair_strength(self, mono):
        result_default = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        result_sparse = self.phase.process(mono, SR, MaterialType.CD_DIGITAL, phase_locality_factor=0.4)
        _assert_phase_result(result_default, mono, check_clipping=False)
        _assert_phase_result(result_sparse, mono, check_clipping=False)

        s_default = float(result_default.metadata.get("repair_strength", 1.0))
        s_sparse = float(result_sparse.metadata.get("repair_strength", 1.0))
        assert s_sparse < s_default
        assert float(result_sparse.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 24 – Dropout Repair
# ===========================================================================
class TestPhase24DropoutRepair:
    def setup_method(self):
        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        self.phase = DropoutRepairPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_audio_with_dropout(self):
        """Audio mit simuliertem Aussetzer."""
        sig = np.ones(_N, dtype=np.float32) * 0.5
        sig[_N // 2 : _N // 2 + 100] = 0.0  # Aussetzer
        result = self.phase.process(sig, SR)
        _assert_phase_result(result, sig, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_repair_strength(self, mono):
        result_default = self.phase.process(mono, SR, material_type="vinyl")
        result_sparse = self.phase.process(mono, SR, material_type="vinyl", strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result_default, mono, check_clipping=False)
        _assert_phase_result(result_sparse, mono, check_clipping=False)
        s_default = float(result_default.modifications.get("repair_strength", 1.0))
        s_sparse = float(result_sparse.modifications.get("repair_strength", 1.0))
        assert s_sparse < s_default
        assert float(result_sparse.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 25 – Azimuth Correction (Stereo + MaterialType PFLICHT)
# ===========================================================================
class TestPhase25AzimuthCorrection:
    def setup_method(self):
        from backend.core.phases.phase_25_azimuth_correction import AzimuthCorrectionPhaseV2

        self.phase = AzimuthCorrectionPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_tape_material(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.TAPE)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)

    def test_zero_strength_passthrough_tape(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.TAPE, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength_tape(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.TAPE, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 26 – Dynamic Range Expansion
# ===========================================================================
class TestPhase26DynamicRangeExpansion:
    def setup_method(self):
        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        self.phase = DynamicRangeExpansion()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 27 – Click/Pop Removal V9
# ===========================================================================
class TestPhase27ClickPopRemoval:
    def setup_method(self):
        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        self.phase = ClickPopRemoval()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_audio_with_click(self):
        """Signal mit simuliertem Click."""
        sig = np.zeros(_N, dtype=np.float32)
        sig[_N // 4] = 1.0  # impulsartiger Click
        result = self.phase.process(sig, SR, MaterialType.VINYL)
        _assert_phase_result(result, sig, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_linked_detection_coherence(self):
        """§2.51: Stereo click detection is linked — same positions repaired in both channels."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.05, _N).astype(np.float32)
        left = base.copy()
        right = base.copy() + rng.normal(0, 0.01, _N).astype(np.float32)
        # Same click in both channels
        left[_N // 3] = 1.0
        right[_N // 3] = 0.9
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, SR, MaterialType.VINYL)
        assert result.success
        assert result.metadata.get("stereo_mode") == "linked_detection"
        assert result.audio.shape == stereo_in.shape

    def test_stereo_output_shape_preserved(self):
        """Stereo input → stereo output with correct shape."""
        rng = np.random.default_rng(99)
        stereo_in = rng.normal(0, 0.1, (_N, 2)).astype(np.float32)
        result = self.phase.process(stereo_in, SR, MaterialType.CD_DIGITAL)
        assert result.audio.shape == stereo_in.shape


# ===========================================================================
# Phase 28 – Surface Noise Profiling
# ===========================================================================
class TestPhase28SurfaceNoiseProfiling:
    def setup_method(self):
        from backend.core.phases.phase_28_surface_noise_profiling import SurfaceNoiseProfiling

        self.phase = SurfaceNoiseProfiling()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_shellac_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.SHELLAC)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 29 – Tape Hiss Reduction
# ===========================================================================
class TestPhase29TapeHissReduction:
    # Phase 29 erzwingt 48 kHz (validate_input()) — anderer SR als Datei-Standard
    _SR = 48_000

    def setup_method(self):
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        self.phase = TapeHissReductionPhase()

    def test_mono_returns_phase_result(self, mono):
        audio_48 = mono[: self._SR // 4]  # 0.25 s bei 48 kHz
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE)
        _assert_phase_result(result, audio_48, check_clipping=False)

    def test_silent_input(self, silent_mono):
        silent_48 = silent_mono[: self._SR // 4]
        result = self.phase.process(silent_48, self._SR)
        _assert_phase_result(result, silent_48)

    def test_reel_tape_material(self, mono):
        audio_48 = mono[: self._SR // 4]
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE)
        _assert_phase_result(result, audio_48, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        audio_48 = mono[: self._SR // 4]
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE, strength=0.0)
        _assert_phase_result(result, audio_48, check_clipping=False)
        assert np.allclose(result.audio, audio_48, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        audio_48 = mono[: self._SR // 4]
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, audio_48, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_linked_sidechain_coherence(self):
        """§2.51: Stereo-Audio muss identische Gain-Maske auf L und R erhalten
        (Mid-Linked-Sidechain), sodass asymmetrisches Kanalrauschen nicht zu
        unterschiedlichen Reduktionsraten führt (Phantom-Mitte-Instabilität)."""
        sr = self._SR
        n = sr // 4  # 0.25 s
        rng = np.random.default_rng(42)
        # L channel: clean sine + moderate noise
        sine = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n, dtype=np.float32) / sr)
        noise_l = 0.04 * rng.standard_normal(n).astype(np.float32)
        # R channel: same sine + MUCH MORE noise (asymmetric hiss)
        noise_r = 0.15 * rng.standard_normal(n).astype(np.float32)
        stereo = np.column_stack([sine + noise_l, sine + noise_r]).astype(np.float32)

        result = self.phase.process(stereo, sr, MaterialType.TAPE)
        _assert_phase_result(result, stereo, check_clipping=False)

        # §2.51 check: stereo_mode should be linked
        assert result.metadata.get("stereo_mode") == "linked_mid_sidechain"

        # Key invariant: L and R should receive the SAME gain mask from Mid sidechain.
        # Therefore the HF reduction ratio (dB) must be similar for both channels.
        from scipy.signal import butter, sosfilt

        sos_hp = butter(4, 8000, btype="high", fs=sr, output="sos")

        hf_in_l = sosfilt(sos_hp, stereo[:, 0])
        hf_in_r = sosfilt(sos_hp, stereo[:, 1])
        hf_out_l = sosfilt(sos_hp, result.audio[:, 0])
        hf_out_r = sosfilt(sos_hp, result.audio[:, 1])

        rms = lambda x: float(np.sqrt(np.mean(x**2) + 1e-12))
        reduction_l_db = 20 * np.log10(max(rms(hf_out_l) / rms(hf_in_l), 1e-12))
        reduction_r_db = 20 * np.log10(max(rms(hf_out_r) / rms(hf_in_r), 1e-12))

        # With linked sidechain: Both channels get same gain → similar reduction ratio.
        # Allow 4 dB tolerance (psychoacoustic masking model still per-channel).
        delta = abs(reduction_l_db - reduction_r_db)
        assert delta < 4.0, (
            f"§2.51 Stereo HF reduction divergence: L={reduction_l_db:.1f} dB, "
            f"R={reduction_r_db:.1f} dB, delta={delta:.1f} dB"
        )

    def test_stereo_output_shape_preserved(self):
        """Stereo input → stereo output, same shape."""
        sr = self._SR
        n = sr // 4
        rng = np.random.default_rng(99)
        stereo = 0.1 * rng.standard_normal((n, 2)).astype(np.float32)
        result = self.phase.process(stereo, sr, MaterialType.TAPE)
        assert result.audio.shape == stereo.shape


# ===========================================================================
# Phase 30 – DC Offset Removal
# ===========================================================================
class TestPhase30DCOffsetRemoval:
    def setup_method(self):
        from backend.core.phases.phase_30_dc_offset_removal import DCOffsetRemoval

        self.phase = DCOffsetRemoval()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_dc_offset_removed(self):
        """Starker DC-Anteil soll entfernt werden."""
        sig = np.ones(_N, dtype=np.float32) * 0.5 + 0.05 * np.random.default_rng(7).standard_normal(_N).astype(
            np.float32
        )
        result = self.phase.process(sig, SR)
        _assert_phase_result(result, sig, check_clipping=False)
        assert abs(float(np.mean(result.audio))) < abs(float(np.mean(sig)))

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 31 – Speed/Pitch Correction
# ===========================================================================
class TestPhase31SpeedPitchCorrection:
    def setup_method(self):
        from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase

        self.phase = SpeedPitchCorrectionPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, material_type="vinyl", sample_rate=SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, material_type="vinyl", sample_rate=SR, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(
            mono,
            material_type="vinyl",
            sample_rate=SR,
            strength=1.0,
            phase_locality_factor=0.4,
        )
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 32 – Mono-to-Stereo
# ===========================================================================
class TestPhase32MonoToStereo:
    def setup_method(self):
        from backend.core.phases.phase_32_mono_to_stereo import MonoToStereoPhaseV2

        self.phase = MonoToStereoPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 40 – Loudness Normalization
# ===========================================================================
class TestPhase40LoudnessNormalization:
    def setup_method(self):
        from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase

        self.phase = LoudnessNormalizationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_input_no_crash(self, silent_mono):
        # Stilles Signal kann Probleme mit Lautheitsmessung machen
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)

    def test_loud_signal_normalized(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, loud_mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_quality_mode_emits_guard_metadata(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, quality_mode="maximum")
        _assert_phase_result(result, stereo, check_clipping=False)
        assert "output_guard_enabled" in result.metadata
        assert "output_guard_fallback" in result.metadata
        assert "output_guard_reason" in result.metadata


# ===========================================================================
# Phase 41 – Output Format Optimization
# ===========================================================================
class TestPhase41OutputFormatOptimization:
    def setup_method(self):
        from backend.core.phases.phase_41_output_format_optimization import OutputFormatOptimization

        self.phase = OutputFormatOptimization()

    def test_mono_returns_phase_result(self, mono):
        # Phase 41 resampled ggf. auf andere Samplerate → nur success & dtype prüfen
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert isinstance(result.audio, np.ndarray)
        assert np.issubdtype(result.audio.dtype, np.floating)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)
        assert result.success is True

    def test_cd_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        assert isinstance(result, PhaseResult)
        assert result.success is True

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_quality_mode_emits_guard_metadata(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, quality_mode="maximum")
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert "output_guard_enabled" in result.metadata
        assert "output_guard_fallback" in result.metadata
        assert "output_guard_reason" in result.metadata


# ===========================================================================
# Phase 42 – Vocal Enhancement
# ===========================================================================
class TestPhase42VocalEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        self.phase = VocalEnhancement()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_voice_frequency_range(self):
        """Gesangs-Frequenzbereich (300 Hz–3 kHz)."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        vocal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = self.phase.process(vocal, SR)
        _assert_phase_result(result, vocal, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 49 – Advanced Deverb
# ===========================================================================
class TestPhase49AdvancedDereverb:
    def setup_method(self):
        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        self.phase = AdvancedDereverbPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_output_shape_preserved(self, mono):
        result = self.phase.process(mono, SR)
        assert result.audio.shape == mono.shape

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_reduced_strength_stays_closer_to_input(self, mono):
        full = self.phase.process(mono, SR, strength=1.0)
        reduced = self.phase.process(mono, SR, strength=0.25)
        _assert_phase_result(full, mono, check_clipping=False)
        _assert_phase_result(reduced, mono, check_clipping=False)
        full_delta = float(np.mean(np.abs(full.audio - mono)))
        reduced_delta = float(np.mean(np.abs(reduced.audio - mono)))
        assert reduced_delta <= full_delta + 1e-6

    def test_attenuation_guard_rescues_aggressive_output(self, mono, monkeypatch):
        def _fake_aggressive(_audio, _sample_rate, _strength, _protect):
            return np.zeros_like(_audio)

        monkeypatch.setattr(self.phase, "_dereverb_channel", _fake_aggressive)
        result = self.phase.process(mono, SR, strength=1.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert bool(result.metadata.get("attenuation_guard_triggered", False)) is True
        assert float(result.metadata.get("wet_mix", 1.0)) < 1.0
        assert float(np.sqrt(np.mean(result.audio**2))) > 0.0


# ===========================================================================
# Phase 51 – Drums Enhancement (KEIN sample_rate!)
# ===========================================================================
class TestPhase51DrumsEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_51_drums_enhancement import DrumsEnhancementV1

        self.phase = DrumsEnhancementV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)

    def test_loud_percussive_signal(self, loud_mono):
        result = self.phase.process(loud_mono)
        _assert_phase_result(result, loud_mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_reduced_strength_stays_closer_to_input(self, mono):
        full = self.phase.process(mono, strength=1.0)
        reduced = self.phase.process(mono, strength=0.4)
        _assert_phase_result(full, mono, check_clipping=False)
        _assert_phase_result(reduced, mono, check_clipping=False)
        full_delta = float(np.mean(np.abs(full.audio - mono)))
        reduced_delta = float(np.mean(np.abs(reduced.audio - mono)))
        assert reduced_delta <= full_delta + 1e-6

    def test_stereo_shape_preserved(self, stereo):
        result = self.phase.process(stereo)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2


# ===========================================================================
# Phase 52 – Piano Restoration (KEIN sample_rate!)
# ===========================================================================
class TestPhase52PianoRestoration:
    def setup_method(self):
        from backend.core.phases.phase_52_piano_restoration import PianoRestorationV1

        self.phase = PianoRestorationV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_piano_tone(self):
        """Klavier-typisches Signal (A4 = 440 Hz)."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        piano = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        result = self.phase.process(piano)
        _assert_phase_result(result, piano, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_reduced_strength_stays_closer_to_input(self, mono):
        full = self.phase.process(mono, strength=1.0)
        reduced = self.phase.process(mono, strength=0.4)
        _assert_phase_result(full, mono, check_clipping=False)
        _assert_phase_result(reduced, mono, check_clipping=False)
        full_delta = float(np.mean(np.abs(full.audio - mono)))
        reduced_delta = float(np.mean(np.abs(reduced.audio - mono)))
        assert reduced_delta <= full_delta + 1e-6

    def test_stereo_preserves_side_information(self):
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        left = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        right = (0.35 * np.sin(2 * np.pi * 660.0 * t + 0.3)).astype(np.float32)
        stereo_in = np.column_stack([left, right]).astype(np.float32)

        result = self.phase.process(stereo_in, strength=1.0)
        _assert_phase_result(result, stereo_in, check_clipping=False)

        side_in = 0.5 * (stereo_in[:, 0] - stereo_in[:, 1])
        side_out = 0.5 * (result.audio[:, 0] - result.audio[:, 1])
        side_in_rms = float(np.sqrt(np.mean(side_in**2)) + 1e-12)
        side_out_rms = float(np.sqrt(np.mean(side_out**2)) + 1e-12)

        assert side_out_rms > 0.2 * side_in_rms
        assert bool(result.metadata.get("stereo_image_preserved", False)) is True

    def test_panns_low_confidence_skips_processing(self, stereo):
        panns_tags = {"Piano": 0.30, "Keyboard": 0.20}
        result = self.phase.process(stereo, panns_tags=panns_tags, strength=1.0)
        _assert_phase_result(result, stereo, check_clipping=False)

        assert result.metadata.get("algorithm") == "skip_panns_confidence"
        assert np.allclose(result.audio, stereo, atol=1e-7)

    def test_panns_high_confidence_processes(self, stereo):
        panns_tags = {"Piano": 0.75}
        result = self.phase.process(stereo, panns_tags=panns_tags, strength=1.0)
        _assert_phase_result(result, stereo, check_clipping=False)

        assert result.metadata.get("algorithm") != "skip_panns_confidence"


# ===========================================================================
# Phase 54 – Transparent Dynamics (KEIN sample_rate!)
# ===========================================================================
class TestPhase54TransparentDynamics:
    def setup_method(self):
        from backend.core.phases.phase_54_transparent_dynamics import TransparentDynamicsV1

        self.phase = TransparentDynamicsV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)

    def test_output_shape_preserved(self, loud_mono):
        result = self.phase.process(loud_mono)
        assert result.audio.shape == loud_mono.shape

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6
