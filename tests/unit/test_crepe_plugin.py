from __future__ import annotations

import sys
import types

import numpy as np
import pytest


@pytest.mark.unit
def test_crepe_falls_back_to_yin_when_pyin_fails(monkeypatch):
    from plugins.crepe_plugin import CrepePlugin

    def _pyin_fail(*_args, **_kwargs):
        raise RuntimeError("pyin failed")

    fake_librosa = types.SimpleNamespace(
        note_to_hz=lambda _note: 32.703195,
        pyin=_pyin_fail,
        yin=lambda audio, **_kwargs: np.full(max(1, len(audio) // 512), 220.0, dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    plugin = CrepePlugin()
    plugin._session = None

    audio = np.random.randn(48_000).astype(np.float32) * 0.01
    result = plugin.analyze(audio, 48_000)

    assert result.model_used == "dsp_yin"
    assert result.f0_hz.size > 0
    assert np.all(result.f0_hz >= 0.0)
    assert np.all(np.isfinite(result.voiced_prob))


def test_crepe_returns_empty_result_when_yin_also_fails(monkeypatch):
    from plugins.crepe_plugin import CrepePlugin

    def _fail(*_args, **_kwargs):
        raise RuntimeError("fallback failed")

    fake_librosa = types.SimpleNamespace(
        note_to_hz=lambda _note: 32.703195,
        pyin=_fail,
        yin=_fail,
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    plugin = CrepePlugin()
    plugin._session = None

    audio = np.random.randn(24_000).astype(np.float32) * 0.01
    result = plugin.analyze(audio, 48_000)

    assert result.model_used == "dsp_yin_failed"
    assert result.f0_hz.shape == (1,)
    assert np.all(result.f0_hz == 0.0)
