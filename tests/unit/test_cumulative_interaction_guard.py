"""
tests/unit/test_cumulative_interaction_guard.py — §2.48 Interaktions-Guard Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

import numpy as np

SR = 48_000


def _audio(dur: float = 1.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _goals(nat=0.92, auth=0.90, tonal=0.96, timbre=0.89, artic=0.87):
    # Keys müssen mit measure_all()-Output übereinstimmen (deutsch, §2.48 Namen-Fix)
    return {
        "natuerlichkeit": nat,
        "authentizitaet": auth,
        "tonal_center": tonal,
        "timbre_authentizitaet": timbre,
        "artikulation": artic,
        "emotion": 0.85,
        "micro_dynamics": 0.88,
        "groove": 0.84,
    }


# ---------------------------------------------------------------------------


def test_00_import():
    from backend.core.cumulative_interaction_guard import (
        CumulativeInteractionGuard,
        get_interaction_guard,
    )

    assert CumulativeInteractionGuard is not None
    assert get_interaction_guard is not None


def test_01_singleton():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    g1 = get_interaction_guard()
    g2 = get_interaction_guard()
    assert g1 is g2


def test_02_reset_returns_fresh_state():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    assert state is not None
    assert len(state.rollback_log) == 0
    assert state.consecutive_rollbacks == 0
    assert not state.should_stop


def test_03_set_baseline():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    goals = _goals()
    guard.set_pre_pipeline_baseline(state, audio, goals)
    assert state.pre_pipeline_goals == goals
    assert state.best_checkpoint is not None
    assert state.best_checkpoint.phase_id == "__pre_pipeline__"


def test_04_no_rollback_on_improvement():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # Phase that improves goals
    improved = _goals(nat=0.94, auth=0.92)
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        improved,
        SR,
    )
    assert not rolled_back
    assert np.array_equal(result_audio, audio)


def test_05_rollback_on_cumulative_drift():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    # Phase that degrades naturalness by > 0.05
    degraded = _goals(nat=0.85)  # -0.07 drift
    bad_audio = audio * 0.5  # modified audio
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        bad_audio,
        degraded,
        SR,
    )
    assert rolled_back
    assert len(state.rollback_log) == 1
    assert state.rollback_log[0].phase_id == "phase_03_denoise"


def test_06_consecutive_rollbacks_stop_pipeline():
    from backend.core.cumulative_interaction_guard import MAX_CONSECUTIVE_ROLLBACKS, get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    degraded = _goals(nat=0.80)
    # MAX_CONSECUTIVE_ROLLBACKS = 3 — need exactly that many consecutive rollbacks
    rollback_phases = ["phase_03_denoise", "phase_29_tape_hiss_reduction", "phase_20_reverb_reduction"]
    for i, ph in enumerate(rollback_phases):
        guard.check_after_phase(state, ph, audio * 0.3, degraded, SR)
        if i < MAX_CONSECUTIVE_ROLLBACKS - 1:
            assert not state.should_stop, f"should_stop too early after rollback {i + 1}"
    assert state.should_stop


def test_07_rollback_resets_on_success():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    # First: degraded → rollback
    degraded = _goals(nat=0.80)
    guard.check_after_phase(state, "phase_03_denoise", audio * 0.5, degraded, SR)
    assert state.consecutive_rollbacks == 1
    # Second: good → reset counter
    improved = _goals(nat=0.93)
    guard.check_after_phase(state, "phase_07_harmonic_restoration", audio, improved, SR)
    assert state.consecutive_rollbacks == 0


def test_08_critical_pair_denoise_dereverb():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92)
    guard.set_pre_pipeline_baseline(state, audio, baseline)
    # Execute phase_03 — OK
    state.executed_phases.add("phase_03_denoise")
    # Now phase_20 triggers critical pair check
    degraded = _goals(nat=0.88)  # drift = -0.04 > -0.03 threshold
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_20_reverb_reduction",
        audio * 0.8,
        degraded,
        SR,
    )
    assert rolled_back


def test_09_critical_pair_no_trigger_if_good():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    state.executed_phases.add("phase_03_denoise")
    # phase_20 with good goals — no trigger
    good = _goals(nat=0.91)  # drift = -0.01, ok
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_20_reverb_reduction",
        audio,
        good,
        SR,
    )
    assert not rolled_back


def test_10_stft_phase_tracking():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio(2.0)
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # Execute 3 STFT phases — only SUBTRACTIVE/DYNAMICS/ENHANCEMENT count (§2.48a Architecture Inversion)
    # phase_07_harmonic_restoration is ADDITIVE → not tracked in stft_phases_executed (GDD invalid per §2.48a)
    for pid in ["phase_03_denoise", "phase_50_spectral_repair", "phase_29_tape_hiss_reduction"]:
        state.executed_phases.add(pid)
        guard.check_after_phase(state, pid, audio, _goals(), SR)
    assert len(state.stft_phases_executed) == 3


def test_11_checkpoint_updated_on_improvement():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    # Better phase
    improved_audio = audio * 1.01
    improved_goals = _goals(nat=0.95)
    guard.check_after_phase(state, "phase_01_click_removal", improved_audio, improved_goals, SR)
    assert state.best_checkpoint is not None
    assert state.best_checkpoint.phase_id == "phase_01_click_removal"


def test_12_metadata_empty_for_clean_run():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    meta = guard.get_rollback_metadata(state)
    assert meta["interaction_rollbacks"] == []
    assert meta["pipeline_stopped_early"] is False


def test_13_metadata_contains_rollback_info():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    degraded = _goals(nat=0.80)
    guard.check_after_phase(state, "phase_03_denoise", audio, degraded, SR)
    meta = guard.get_rollback_metadata(state)
    assert len(meta["interaction_rollbacks"]) == 1
    assert meta["interaction_rollbacks"][0]["phase_id"] == "phase_03_denoise"


def test_14_p1_p2_goals_constant():
    from backend.core.cumulative_interaction_guard import P1_P2_GOALS

    # Prüfe deutsche Namen (measure_all()-kompatibel, §2.48 Namen-Fix)
    assert "natuerlichkeit" in P1_P2_GOALS
    assert "authentizitaet" in P1_P2_GOALS
    assert "tonal_center" in P1_P2_GOALS
    assert "timbre_authentizitaet" in P1_P2_GOALS
    assert "artikulation" in P1_P2_GOALS
    assert len(P1_P2_GOALS) == 5


def test_15_stft_phases_constant():
    from backend.core.cumulative_interaction_guard import STFT_PHASES

    assert "phase_03_denoise" in STFT_PHASES
    assert "phase_07_harmonic_restoration" in STFT_PHASES
    assert "phase_29_tape_hiss_reduction" in STFT_PHASES


def test_16_critical_pairs_defined():
    from backend.core.cumulative_interaction_guard import CRITICAL_PAIRS

    assert len(CRITICAL_PAIRS) >= 5
    for pair_phases, guard_goal, description, max_reg in CRITICAL_PAIRS:
        assert isinstance(pair_phases, frozenset)
        assert isinstance(guard_goal, str)
        assert max_reg < 0


def test_17_group_delay_check_short_audio():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard

    guard = CumulativeInteractionGuard()
    short = np.zeros(100, dtype=np.float32)
    # Should return True (OK) for very short audio
    assert guard._check_group_delay(short, short, SR) is True


def test_18_group_delay_identical_audio():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard

    guard = CumulativeInteractionGuard()
    audio = _audio(1.0)
    assert guard._check_group_delay(audio, audio, SR) is True


def test_19_is_better_checkpoint_none():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard, InteractionGuardState

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    assert guard._is_better_checkpoint(state, _goals()) is True


def test_20_is_better_checkpoint_worse():
    from backend.core.cumulative_interaction_guard import (
        CumulativeInteractionGuard,
        InteractionGuardCheckpoint,
        InteractionGuardState,
    )

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.best_checkpoint = InteractionGuardCheckpoint(
        audio=_audio(),
        phase_id="best",
        goal_scores=_goals(nat=0.95),
    )
    # Worse goals
    assert guard._is_better_checkpoint(state, _goals(nat=0.80)) is False


def test_21_check_critical_pairs_no_match():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard, InteractionGuardState

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.executed_phases = {"phase_01_click_removal"}
    result = guard._check_critical_pairs(state, "phase_02_hum_removal", _goals())
    assert result is None


def test_22_should_stop_prevents_further_checks():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    state.should_stop = True
    audio = _audio()
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        _goals(),
        SR,
    )
    assert not rolled_back


def test_23_rollback_returns_checkpoint_audio():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    original_audio = _audio()
    guard.set_pre_pipeline_baseline(state, original_audio, _goals(nat=0.92))
    # Degraded phase
    bad_audio = original_audio * 0.1
    degraded = _goals(nat=0.80)
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        bad_audio,
        degraded,
        SR,
    )
    assert rolled_back
    # Should return checkpoint audio, not the degraded audio
    assert not np.array_equal(result_audio, bad_audio)


def test_24_no_baseline_no_crash():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    # No baseline set — should not crash
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        _goals(),
        SR,
    )
    assert not rolled_back


def test_25_max_drift_threshold():
    from backend.core.cumulative_interaction_guard import MAX_CUMULATIVE_DRIFT

    assert MAX_CUMULATIVE_DRIFT == -0.05


def test_26_max_consecutive_rollbacks():
    from backend.core.cumulative_interaction_guard import MAX_CONSECUTIVE_ROLLBACKS

    # §2.47 Adaptive-Intelligence: 3 erlaubt Pipeline-Fortschritt auf stark
    # degradiertem Vintage-Material (tape gen-3, SNR < 6 dB) ohne frühzeitigen Stop.
    assert MAX_CONSECUTIVE_ROLLBACKS == 3


def test_27_interaction_rollback_dataclass():
    from backend.core.cumulative_interaction_guard import InteractionRollback

    rb = InteractionRollback(
        phase_id="phase_03_denoise",
        reason="test",
        drift={"natuerlichkeit": -0.06},
        rolled_back_to="__pre_pipeline__",
    )
    assert rb.phase_id == "phase_03_denoise"
    assert rb.drift["natuerlichkeit"] == -0.06
    assert rb.drift["natuerlichkeit"] == -0.06


def test_28_spectral_subtraction_phases_constant():
    """§2.48: _SPECTRAL_SUBTRACTION_PHASES enthält die 4 betroffenen Phasen."""
    from backend.core.cumulative_interaction_guard import _SPECTRAL_SUBTRACTION_PHASES

    expected = {
        "phase_03_denoise",
        "phase_20_reverb_reduction",
        "phase_29_tape_hiss_reduction",
        "phase_49_advanced_dereverb",
    }
    assert expected == set(_SPECTRAL_SUBTRACTION_PHASES)


def test_29_spectral_threshold_higher_than_standard():
    """§2.48: Spektralsubtraktion-Phasen erhalten 10 ms, Standard 5 ms."""
    from backend.core.cumulative_interaction_guard import (
        MAX_GROUP_DELAY_DEVIATION_MS,
        MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL,
    )

    assert MAX_GROUP_DELAY_DEVIATION_MS == 5.0
    assert MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL == 10.0
    assert MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL > MAX_GROUP_DELAY_DEVIATION_MS


def test_30_group_delay_identical_audio_spectral_phase_passes():
    """§2.48: Identisches Audio → 0 ms Deviation → immer True, auch für Spektralphase."""
    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1

    guard = get_interaction_guard()
    ok = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")
    assert ok, "Identisches Audio muss bei Spektralphase passieren"


def test_31_group_delay_spectral_phase_uses_10ms_threshold():
    """§2.48: Spektralphase erhält 10 ms Threshold — Mock-Kontrolle der internen Messung."""
    from unittest.mock import patch

    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    guard = get_interaction_guard()

    # 7 ms in samples = 7 * 48000 / 1000 = 336 samples
    # Mit Mock liefern wir stets 336 als percentile-Wert → 7.0 ms Deviation
    controlled_samples = 7.0 * sr / 1000.0  # 336
    real_percentile = np.percentile
    calls: list[float] = []

    def _mock_percentile(a, q, *args, **kwargs):
        # Nur den gd_valid-Percentile-Call intercepten (1-D float array, q=95)
        if isinstance(a, np.ndarray) and a.ndim == 1 and q == 95:
            calls.append(float(controlled_samples))
            return controlled_samples
        return real_percentile(a, q, *args, **kwargs)

    with patch("backend.core.cumulative_interaction_guard.np.percentile", side_effect=_mock_percentile):
        ok_spectral = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")
        ok_standard = guard._check_group_delay(audio, audio, sr, phase_id="phase_07_harmonic_restoration")

    # Muss mindestens 1× intercepted worden sein
    assert len(calls) >= 1, "_mock_percentile nicht aufgerufen"
    # 7 ms < 10 ms → Spektralphase passiert
    assert ok_spectral, "phase_29 soll bei 7 ms (< 10 ms-Threshold) passieren"
    # 7 ms > 5 ms → Standard-Phase scheitert
    assert not ok_standard, "phase_07 soll bei 7 ms (> 5 ms-Threshold) scheitern"


def test_32_group_delay_spectral_phase_fails_above_10ms():
    """§2.48: Auch Spektralphase scheitert bei > 10 ms Deviation."""
    from unittest.mock import patch

    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    guard = get_interaction_guard()

    # 11 ms → 528 samples
    controlled_samples = 11.0 * sr / 1000.0
    real_percentile = np.percentile

    def _mock_percentile(a, q, *args, **kwargs):
        if isinstance(a, np.ndarray) and a.ndim == 1 and q == 95:
            return controlled_samples
        return real_percentile(a, q, *args, **kwargs)

    with patch("backend.core.cumulative_interaction_guard.np.percentile", side_effect=_mock_percentile):
        ok = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")

    assert not ok, "phase_29 soll bei 11 ms (> 10 ms-Threshold) scheitern"
