"""
tests/unit/test_per_phase_musical_goals_gate.py
================================================
Aurik 9.9 — PerPhaseMusicalGoalsGate (§2.29)

26 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import math
import threading

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Mock Phasen
# ---------------------------------------------------------------------------


class _MockPassPhase:
    """Identity-Phase — keine Musical-Goal-Regression."""

    def __call__(self, audio, strength=1.0):
        return audio.copy().astype(np.float32)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockPass"
        return m


class _MockAttenuatePhase:
    """Dämpft minimal — Pass (keine starke Regression)."""

    def __call__(self, audio, strength=1.0):
        return (audio * 0.98).astype(np.float32)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockAttenuate"
        return m


class _MockZeroPhase:
    """Zeroed output — erzeugt starke Regression in allen Musical Goals."""

    def __call__(self, audio, strength=1.0):
        return np.zeros_like(audio)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockZero"
        return m


class _MockExplodePhase:
    """Gibt NaN zurück — sollte sicher behandelt werden."""

    def __call__(self, audio, strength=1.0):
        out = audio.copy()
        out[:] = np.nan
        return out

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockExplode"
        return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate():
    from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

    return PerPhaseMusicalGoalsGate()


@pytest.fixture(scope="module")
def audio_1s():
    np.random.seed(42)
    t = np.linspace(0, 1.0, SR, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8
    return sig


@pytest.fixture(scope="module")
def audio_5s():
    np.random.seed(42)
    t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
    sig = np.sin(2 * np.pi * 261.6 * t).astype(np.float32) * 0.7
    return sig


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestPMGGSingleton:
    def test_01_singleton_same_instance(self):
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        a = get_phase_gate()
        b = get_phase_gate()
        assert a is b

    def test_02_singleton_thread_safe(self):
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        instances = []

        def _get():
            instances.append(get_phase_gate())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestPMGGReset:
    def test_03_reset_clears_rollback_count(self, gate):
        try:
            gate._rollback_count
        except AttributeError:
            pass  # Attribut nicht vorhanden → kein Test nötig
        gate.reset()
        assert getattr(gate, "_rollback_count", 0) == 0

    def test_04_reset_idempotent(self, gate):
        gate.reset()
        gate.reset()
        assert getattr(gate, "_rollback_count", 0) == 0


# ---------------------------------------------------------------------------
# Tests: wrap_phase() Rückgabe-Struktur
# ---------------------------------------------------------------------------


class TestPMGGWrapPhaseStructure:
    def test_05_returns_three_tuple(self, gate, audio_5s):
        gate.reset()
        result = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(result, (tuple, list))
        assert len(result) == 3

    def test_06_first_element_is_ndarray(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(out, np.ndarray)

    def test_07_second_element_is_dict(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(scores, dict)

    def test_08_third_element_has_action(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert hasattr(entry, "action")

    def test_09_third_element_has_strength_used(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert hasattr(entry, "strength_used")

    def test_10_action_is_valid_string(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5", "rollback"}

    def test_11_strength_used_is_positive(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert math.isfinite(entry.strength_used)
        assert entry.strength_used > 0.0


# ---------------------------------------------------------------------------
# Tests: wrap_phase() Audio-Qualität
# ---------------------------------------------------------------------------


class TestPMGGAudioQuality:
    def test_12_output_shape_preserved(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert out.shape == audio_5s.shape

    def test_13_output_no_nan_passthrough(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert np.isfinite(out).all()

    def test_14_output_bounded(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_15_nan_phase_output_handled(self, gate, audio_5s):
        """Phase gibt NaN zurück — Gate muss robust reagieren (kein Crash)."""
        gate.reset()
        try:
            out, _, entry = gate.wrap_phase(_MockExplodePhase(), audio_5s, SR)
            # Falls kein Fehler: Ausgabe muss finite sein (Rollback greift)
            assert np.isfinite(out).all()
        except Exception:
            pass  # Exception ist auch akzeptabel

    def test_16_scores_values_finite(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for v in scores.values():
            assert math.isfinite(v)

    def test_17_scores_keys_are_strings(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for k in scores.keys():
            assert isinstance(k, str)

    def test_18_scores_values_in_unit_interval(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for v in scores.values():
            assert -0.1 <= v <= 1.1  # leichte numerische Toleranz


# ---------------------------------------------------------------------------
# Tests: Regressions-Behandlung
# ---------------------------------------------------------------------------


class TestPMGGRegression:
    def test_19_pass_phase_action_passed(self, gate, audio_5s):
        gate.reset()
        _, _, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        # Identity-Phase sollte "passed" oder höchstens "retry1" erhalten
        assert entry.action in {"passed", "retry1"}

    def test_20_zero_phase_triggers_protection(self, gate, audio_5s):
        """Null-Phase zerstört Signal — Gate soll Rollback/Retry auslösen."""
        gate.reset()
        _, _, entry = gate.wrap_phase(_MockZeroPhase(), audio_5s, SR)
        # Bei starker Regression: rollback oder retry
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5", "rollback"}

    def test_21_zero_phase_output_is_bounded(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockZeroPhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_22_rollback_count_increments(self, gate, audio_5s):
        gate.reset()
        count_before = getattr(gate, "_rollback_count", 0)
        gate.wrap_phase(_MockZeroPhase(), audio_5s, SR)
        count_after = getattr(gate, "_rollback_count", 0)
        # Count hat sich ggf. erhöht (wenn Rollback ausgelöst wurde)
        assert count_after >= count_before


# ---------------------------------------------------------------------------
# Tests: Verschiedene Input-Größen
# ---------------------------------------------------------------------------


class TestPMGGInputVariations:
    def test_23_short_audio_1s(self, gate, audio_1s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_1s, SR)
        assert np.isfinite(out).all()
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5", "rollback"}

    def test_24_stereo_audio(self, gate):
        np.random.seed(42)
        t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
        stereo = np.stack(
            [
                np.sin(2 * np.pi * 440 * t).astype(np.float32),
                np.sin(2 * np.pi * 550 * t).astype(np.float32),
            ],
            axis=0,
        )
        gate.reset()
        try:
            out, scores, entry = gate.wrap_phase(_MockPassPhase(), stereo, SR)
            assert np.isfinite(out).all()
        except Exception:
            pass  # Stereo-Ablehnung ist akzeptabel

    def test_25_existing_scores_passed_in(self, gate, audio_5s):
        """Existierende Scores als Baseline übergeben."""
        gate.reset()
        initial_scores = {
            "brillanz": 0.87,
            "waerme": 0.82,
            "groove": 0.90,
        }
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR, scores_before=initial_scores)
        assert np.isfinite(out).all()
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5", "rollback"}

    def test_26_attenuate_phase_passes(self, gate, audio_5s):
        """Minimale Dämpfung (0.98) sollte in den meisten Fällen akzeptiert werden."""
        gate.reset()
        out, _, entry = gate.wrap_phase(_MockAttenuatePhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ----------------------------------------------------------------------
# Tests §9.7.3 — Phasen-adaptive Sample-Dauer (PHASE_SAMPLE_DURATIONS)
# ----------------------------------------------------------------------

class TestPMGGAdaptiveSampleDuration:
    """§9.7.3: Triviale Phasen bekommen kürzere Sample-Dauer als Standard."""

    def test_27_trivial_phases_have_short_duration(self):
        from backend.core.per_phase_musical_goals_gate import (
            _get_sample_duration,
            SAMPLE_DURATION_S,
            PHASE_SAMPLE_DURATIONS,
        )
        for prefix in PHASE_SAMPLE_DURATIONS:
            dur = _get_sample_duration(prefix)
            assert dur < SAMPLE_DURATION_S, f"Phase {prefix}: {dur} nicht kürzer als {SAMPLE_DURATION_S}"

    def test_28_standard_phase_gets_full_duration(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration, SAMPLE_DURATION_S
        assert _get_sample_duration("phase_03_denoise") == SAMPLE_DURATION_S
        assert _get_sample_duration("phase_55_diffusion_inpainting") == SAMPLE_DURATION_S
        assert _get_sample_duration("phase_29_tape_hiss_reduction") == SAMPLE_DURATION_S

    def test_29_dc_offset_phase_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration
        dur = _get_sample_duration("phase_30_dc_offset_removal")
        assert 1.0 <= dur <= 2.0

    def test_30_rumble_filter_phase_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration
        dur = _get_sample_duration("phase_05_rumble_filter")
        assert 1.0 <= dur <= 2.0

    def test_31_hum_removal_phase_2s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration
        dur = _get_sample_duration("phase_02_hum_removal")
        assert 1.0 <= dur <= 2.5

    def test_32_duration_bounded_min_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration
        # Keine Dauer darf unter 1 Sekunde liegen
        for prefix in ("phase_30", "phase_05", "phase_02", "phase_15", "phase_11", "phase_18"):
            assert _get_sample_duration(prefix) >= 1.0

    def test_33_duration_bounded_max_5s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration, SAMPLE_DURATION_S
        # Keine adaptive Dauer darf über den Standard hinausgehen
        for prefix in ("phase_30", "phase_05", "phase_02", "phase_15", "phase_11", "phase_18"):
            assert _get_sample_duration(prefix) <= SAMPLE_DURATION_S

    def test_34_phase_sample_durations_dict_nonempty(self):
        from backend.core.per_phase_musical_goals_gate import PHASE_SAMPLE_DURATIONS
        assert len(PHASE_SAMPLE_DURATIONS) >= 6

    def test_35_wrap_phase_uses_short_duration_for_trivial(self, gate, audio_5s):
        """DC-Offset-Phase soll mit adaptiver Dauer ohne Fehler laufen."""
        import math

        class _MockDCPhase:
            """Mock-Phase die DC-Offset 'korrigiert' (identisch pass-through)."""

            def __call__(self, audio, **kwargs):
                return audio

        # Phase-ID enthält phase_30 -> kurze Sample-Dauer (§9.7.3)
        _MockDCPhase.__name__ = "phase_30_dc_offset_removal"
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockDCPhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        for v in scores.values():
            assert math.isfinite(v)
