from __future__ import annotations

import numpy as np

from backend.core.phases import phase_55_diffusion_inpainting as phase55


def test_process_channel_skips_ml_plugins_on_thrashing(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    calls = {"cqtdiff": 0, "flow": 0}

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)

    def _cqtdiff(*_args, **_kwargs):
        calls["cqtdiff"] += 1
        return np.full(60, 0.5, dtype=np.float32)

    def _flow(*_args, **_kwargs):
        calls["flow"] += 1
        return np.full(60, 0.5, dtype=np.float32)

    monkeypatch.setattr(phase55, "_try_cqtdiff_plus_plugin", _cqtdiff)
    monkeypatch.setattr(phase55, "_try_flow_matching_plugin", _flow)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.zeros(60, dtype=np.float32))

    _repaired, stats = phase55._process_channel(audio, 48000, 20.0)

    assert calls["cqtdiff"] == 0
    assert calls["flow"] == 0
    assert stats["ml_thrashing_guard"] is True


def test_process_channel_damage_guard_replaces_risky_candidate(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.ones(60, dtype=np.float32))

    repaired, stats = phase55._process_channel(audio, 48000, 20.0)

    repaired_gap = repaired[200:260]
    assert stats["damage_guard_activations"] >= 1
    assert float(np.max(np.abs(repaired_gap))) < 0.2


def test_phase55_metadata_contains_damage_and_thrash_guards(monkeypatch):
    audio = np.full(512, 0.02, dtype=np.float32)
    audio[200:260] = 0.0

    monkeypatch.setattr(phase55, "_detect_gaps", lambda *_args, **_kwargs: [(200, 260)])
    monkeypatch.setattr(phase55, "_is_ml_thrashing", lambda: True)
    monkeypatch.setattr(phase55, "_inpaint_gap_dsp", lambda *_args, **_kwargs: np.ones(60, dtype=np.float32))

    phase = phase55.DiffusionInpaintingPhase()
    result = phase.process(audio, 48000)

    assert result.success is True
    assert int(result.metadata.get("damage_guard_activations", 0)) >= 1
    assert bool(result.metadata.get("ml_thrashing_guard", False)) is True
