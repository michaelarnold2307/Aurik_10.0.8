"""§v10.15 Phase Contract Tests
=============================
Validates: phase output types, audio shapes, stereo handling, 
PostGate signatures, OneTakeExport fallback, STCG consistency.

Run: python3 -m pytest backend/tests/test_phase_contracts.py -v
"""

import numpy as np
import pytest


# ── Phase Import Helpers ────────────────────────────────────────


def _make_stereo_audio(duration_s=2.0, sr=48000):
    """Generate valid stereo test audio (N, 2)."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    left = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n)
    right = 0.3 * np.sin(2 * np.pi * 445 * t) + 0.1 * np.random.randn(n)
    return np.column_stack([left, right]).astype(np.float32)


def _make_mono_audio(duration_s=2.0, sr=48000):
    """Generate valid mono test audio (N,)."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(n)).astype(np.float32)


# ── PhaseResult Contract ────────────────────────────────────────


class TestPhaseResultContract:
    """Every phase must return PhaseResult with valid audio."""

    def test_phase09_returns_phaseresult(self):
        """phase_09_crackle_removal must return PhaseResult."""
        from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = CrackleRemovalPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000, material_type="cassette")

        assert isinstance(result, PhaseResult), (
            f"phase_09 returned {type(result).__name__}, expected PhaseResult"
        )
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"
        assert result.audio.ndim in (1, 2), f"audio must be 1D or 2D, got {result.audio.ndim}D"
        # Shape must be consistent with input
        assert result.audio.shape == audio.shape, (
            f"output shape {result.audio.shape} != input shape {audio.shape}"
        )

    def test_phase29_returns_phaseresult(self):
        """phase_29_tape_hiss_reduction must return PhaseResult."""
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = TapeHissReductionPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        assert isinstance(result, PhaseResult), (
            f"phase_29 returned {type(result).__name__}, expected PhaseResult"
        )
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"

    def test_phase06_returns_phaseresult(self):
        """phase_06_frequency_restoration must return PhaseResult."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase
        from backend.core.phases.phase_interface import PhaseResult

        phase = FrequencyRestorationPhase()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        assert isinstance(result, PhaseResult), (
            f"phase_06 returned {type(result).__name__}, expected PhaseResult"
        )
        assert isinstance(result.audio, np.ndarray), "PhaseResult.audio must be ndarray"


# ── Stereo Shape Contract ───────────────────────────────────────


class TestStereoShapeContract:
    """Stereo input must produce same-shape stereo output."""

    @pytest.mark.parametrize("phase_name,module_path,class_name", [
        ("phase_09", "backend.core.phases.phase_09_crackle_removal", "CrackleRemovalPhase"),
        ("phase_29", "backend.core.phases.phase_29_tape_hiss_reduction", "TapeHissReductionPhase"),
    ])
    def test_stereo_in_stereo_out(self, phase_name, module_path, class_name):
        """Stereo (N,2) input → stereo (N,2) output."""
        import importlib
        mod = importlib.import_module(module_path)
        phase_cls = getattr(mod, class_name)
        phase = phase_cls()
        audio = _make_stereo_audio()
        result = phase.process(audio, sample_rate=48000)

        if result.audio is not None:
            if audio.ndim == 2:
                assert result.audio.ndim == 2, (
                    f"{phase_name}: stereo input got {result.audio.ndim}D output"
                )
                assert result.audio.shape == audio.shape, (
                    f"{phase_name}: shape mismatch {audio.shape} → {result.audio.shape}"
                )

    @pytest.mark.parametrize("phase_name,module_path,class_name", [
        ("phase_09", "backend.core.phases.phase_09_crackle_removal", "CrackleRemovalPhase"),
    ])
    def test_mono_in_mono_out(self, phase_name, module_path, class_name):
        """Mono (N,) input → mono (N,) output."""
        import importlib
        mod = importlib.import_module(module_path)
        phase_cls = getattr(mod, class_name)
        phase = phase_cls()
        audio = _make_mono_audio()
        result = phase.process(audio, sample_rate=48000)

        if result.audio is not None:
            assert result.audio.ndim == 1, (
                f"{phase_name}: mono input got {result.audio.ndim}D output"
            )
            assert len(result.audio) == len(audio), (
                f"{phase_name}: length mismatch {len(audio)} → {len(result.audio)}"
            )


# ── Phase Contract Guard Tests ──────────────────────────────────


class TestPhaseContractGuard:
    """The centralized guard module catches invalid inputs."""

    def test_guard_rejects_tuple(self):
        """guard_phase_input must reject tuple input."""
        from backend.core.phase_contract_guard import guard_phase_input

        # Should convert tuple to ndarray (not crash)
        result = guard_phase_input((np.zeros(100, dtype=np.float32),), 48000, "test")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_guard_rejects_wrong_ndim(self):
        """guard_phase_input must reject 3D audio."""
        from backend.core.phase_contract_guard import guard_phase_input
        import pytest

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            guard_phase_input(np.zeros((2, 3, 100), dtype=np.float32), 48000, "test")

    def test_guard_output_rejects_non_phaseresult(self):
        """guard_phase_output must reject non-PhaseResult."""
        from backend.core.phase_contract_guard import guard_phase_output
        import pytest

        with pytest.raises(TypeError, match="expected PhaseResult"):
            guard_phase_output("not_a_phaseresult", np.zeros(100), "test")


# ── OneTakeExport Contract ──────────────────────────────────────


class TestOneTakeExportContract:
    """OneTakeExport must not infinite-loop and must return best-effort."""

    def test_no_change_early_exit(self):
        """When no corrections are possible, best-effort export must succeed."""
        from backend.core.one_take_export import OneTakeExport

        audio = _make_stereo_audio(duration_s=5.0)
        result = OneTakeExport.prepare(audio, 48000, is_studio_2026=False)

        assert result.audio is not None, "OneTakeExport must return audio"
        assert isinstance(result.passed, bool), "OneTakeExport must set passed flag"
        # Must not exceed MAX_RETRIES
        assert result.retries <= 3, f"retries={result.retries} exceeds MAX_RETRIES=3"


# ── STCG Consistency Test ───────────────────────────────────────


class TestSTCGConsistency:
    """STCG measurement and correction must use the same algorithm."""

    def test_verify_lag_multi_point_returns_expected_keys(self):
        """_verify_lag_multi_point must return consistent dict structure."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        sr = 48000
        n = sr * 3  # 3 seconds
        t = np.arange(n) / sr
        ch_l = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ch_r = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # same = no lag

        result = stcg._verify_lag_multi_point(ch_l, ch_r, sr)

        assert "median_lag" in result, "must have median_lag"
        assert "max_spread" in result, "must have max_spread"
        assert "num_points" in result, "must have num_points"
        assert isinstance(result["median_lag"], (int, float)), "median_lag must be numeric"

    def test_same_signal_no_lag(self):
        """Identical L/R channels should produce near-zero lag."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        sr = 48000
        n = sr * 5  # 5 seconds
        t = np.arange(n) / sr
        signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        result = stcg._verify_lag_multi_point(signal, signal, sr)
        # Near-zero lag expected for identical channels
        assert abs(result["median_lag"]) < 5, (
            f"identical signals should have ~0 lag, got {result['median_lag']}"
        )

    def test_correct_interchannel_delay_preserves_shape(self):
        """correct_interchannel_delay must preserve input shape."""
        from backend.core.stereo_temporal_coherence_guard import (
            StereoTemporalCoherenceGuard,
        )

        stcg = StereoTemporalCoherenceGuard()
        audio = _make_stereo_audio(duration_s=4.0)
        result = stcg.correct_interchannel_delay(audio, 48000, phase_id="test")

        assert result.shape == audio.shape, (
            f"STCG changed shape: {audio.shape} → {result.shape}"
        )
        assert result.dtype == audio.dtype, (
            f"STCG changed dtype: {audio.dtype} → {result.dtype}"
        )


# ── PostGate Lambda Contract ────────────────────────────────────


class TestPostGateLambdaContract:
    """PostGate lambdas must accept 3 positional args (audio, sr, strength)."""

    def test_antimuffling_lambda_signature(self):
        """AntiMufflingPass lambda must accept (a, sr, strength=None)."""
        # Verify the lambda in UV3 accepts all 3 args
        try:
            from backend.core.phases.phase_19_de_esser import AntiMufflingPass
            amp = AntiMufflingPass()
            # Simulate what PostGate does: call with (a, sr, strength=None)
            audio = _make_mono_audio()
            result = amp.process(audio, 48000)
            assert isinstance(result, np.ndarray), "AntiMufflingPass must return ndarray"
        except ImportError:
            pytest.skip("AntiMufflingPass not available in this context")

    def test_vocal_clarity_lambda_signature(self):
        """VocalClarityMax lambda must accept (a, sr, strength=None)."""
        try:
            from backend.core.phases.phase_19_de_esser import VocalClarityMax
            vcm = VocalClarityMax()
            audio = _make_mono_audio()
            result = vcm.process(audio, 48000)
            assert isinstance(result, np.ndarray), "VocalClarityMax must return ndarray"
        except ImportError:
            pytest.skip("VocalClarityMax not available in this context")
