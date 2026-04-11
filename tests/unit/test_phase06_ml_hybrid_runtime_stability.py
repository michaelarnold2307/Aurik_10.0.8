from __future__ import annotations

import numpy as np
import psutil
import scipy.signal as signal

from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase


class _FreeMemVM:
    """Mocked psutil.virtual_memory — 32 GB free, bypasses phase_06 headroom guard."""

    available: int = 32 * 1024**3
    total: int = 32 * 1024**3
    percent: float = 10.0


class _FakeAudioSRPlugin:
    def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
        # Deterministic HF boost-like behavior for testability.
        _ = (sr, target_sr)
        return np.clip(np.asarray(audio, dtype=np.float32) * 1.05, -1.0, 1.0)


def _make_rolloff_audio(sr: int = 48_000, duration_s: float = 0.35) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / float(sr)
    src = (
        0.45 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 1100.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 4500.0 * t)
        + 0.18 * np.sin(2.0 * np.pi * 9000.0 * t)
        + 0.12 * np.sin(2.0 * np.pi * 14000.0 * t)
    ).astype(np.float32)
    src += 0.05 * np.random.RandomState(42).standard_normal(n).astype(np.float32)
    sos = signal.butter(8, 4200.0 / (sr / 2.0), btype="low", output="sos")
    rolled = signal.sosfiltfilt(sos, src)
    return np.column_stack([rolled, rolled * 0.98]).astype(np.float32)


def test_phase06_uses_ml_hybrid_when_audiosr_available(monkeypatch) -> None:
    # Mock psutil so the phase_06 headroom guard (requires 9.5 GB available) passes
    # on memory-constrained dev machines. Tests ML path logic, not RAM availability.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())
    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio()

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_audiosr_plugin",
        lambda: _FakeAudioSRPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
        audiosr_min_duration_s=0.0,
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "ml_hybrid"
    assert float(result.metadata.get("ml_blend_alpha", 0.0)) > 0.0


def test_phase06_falls_back_to_dsp_on_ml_error(monkeypatch) -> None:
    # Mock psutil so the phase_06 headroom guard passes, allowing AudioSR to be
    # attempted (and fail with RuntimeError) — which sets ml_error in metadata.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FreeMemVM())

    class _BrokenAudioSRPlugin:
        def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
            _ = (audio, sr, target_sr)
            raise RuntimeError("synthetic failure")

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio()

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_audiosr_plugin",
        lambda: _BrokenAudioSRPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
        audiosr_min_duration_s=0.0,
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "dsp_only"
    assert "ml_error" in result.metadata


def test_phase06_skips_audiosr_for_short_clip_guard(monkeypatch) -> None:
    called = {"plugin": False}

    class _ShouldNotBeCalledPlugin:
        def process(self, audio: np.ndarray, sr: int, target_sr: int = 48_000) -> np.ndarray:
            _ = (audio, sr, target_sr)
            called["plugin"] = True
            return np.asarray(audio, dtype=np.float32)

    phase = FrequencyRestorationPhase(sample_rate=48_000)
    audio = _make_rolloff_audio(duration_s=0.35)

    monkeypatch.setattr(
        "backend.core.phases.phase_06_frequency_restoration._get_audiosr_plugin",
        lambda: _ShouldNotBeCalledPlugin(),
    )

    result = phase.process(
        audio,
        sample_rate=48_000,
        material_type="shellac",
        quality_mode="maximum",
    )

    assert result.success
    assert result.modifications.get("frequency_restored") is True
    assert result.metadata.get("strategy_used") == "dsp_only"
    assert "short_clip_guard" in str(result.metadata.get("ml_reason", ""))
    assert called["plugin"] is False
