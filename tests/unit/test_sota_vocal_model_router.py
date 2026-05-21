"""Unit tests for §SMR-1 SotaVocalModelRouter."""

import types

import numpy as np


def _audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_router_prefers_bs_roformer_for_vocal_material(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeBsResult:
        stems = {
            "vocals": np.full(48000, 0.10, dtype=np.float32),
            "drums": np.full(48000, 0.20, dtype=np.float32),
            "bass": np.full(48000, 0.30, dtype=np.float32),
        }
        model_used = "bs_roformer"
        confidence = 0.88
        sdri_db = 8.5

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeBsResult()

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.bs_roformer_plugin",
        types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs()),
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(_audio(), 48000, panns_singing=0.8)
    assert result.success is True
    assert result.model_used == "bs_roformer"
    np.testing.assert_allclose(result.vocal, np.full(48000, 0.10, dtype=np.float32))
    np.testing.assert_allclose(result.instrumental, np.full(48000, 0.50, dtype=np.float32))
    assert result.metadata["confidence"] == 0.88
    assert "capability_status" in result.metadata


def test_router_skips_roformer_fallback_and_uses_demucs(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeFallbackBsResult:
        stems = {"vocals": np.full(48000, 0.10, dtype=np.float32), "other": np.full(48000, 0.20, dtype=np.float32)}
        model_used = "nmf_dsp_fallback"
        confidence = 0.20
        sdri_db = 0.0

    class _FakeBs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int, *, stems: list[str] | None = None):  # pylint: disable=unused-argument
            return _FakeFallbackBsResult()

    class _FakeDemucs:
        @staticmethod
        def separate(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:  # pylint: disable=unused-argument
            return {
                "vocals": np.full(48000, 0.30, dtype=np.float32),
                "other": np.full(48000, 0.40, dtype=np.float32),
            }

    sys_modules = __import__("sys").modules
    monkeypatch.setitem(
        sys_modules, "plugins.bs_roformer_plugin", types.SimpleNamespace(get_bs_roformer=lambda: _FakeBs())
    )
    monkeypatch.setitem(
        sys_modules, "plugins.demucs_v4_plugin", types.SimpleNamespace(get_demucs_plugin=lambda: _FakeDemucs())
    )

    result = SotaVocalModelRouter().separate_vocal_instrumental(_audio(), 48000, panns_singing=0.8)
    assert result.success is True
    assert result.model_used == "demucs_v4"
    assert "bs_roformer:nmf_dsp_fallback" in result.fallback_chain
    assert "capability_status" in result.metadata
    np.testing.assert_allclose(result.vocal, np.full(48000, 0.30, dtype=np.float32))


def test_router_vocal_nr_skips_unloaded_miipher_for_sgmse(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipher:
        _model_loaded = False

        @staticmethod
        def enhance(audio: np.ndarray, sr: int, noise_snr_db: float = 0.0) -> np.ndarray:  # pylint: disable=unused-argument
            raise AssertionError("MIIPHER stub must not run when no model is loaded")

    class _FakeSgmseResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = (audio * 0.5).astype(np.float32)
            self.model_used = "sgmse_plus_torchscript"

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
            return _FakeSgmseResult(audio)

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipher()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.sgmse_plugin",
        types.SimpleNamespace(get_sgmse_plugin=lambda: _FakeSgmse()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: (_ for _ in ()).throw(RuntimeError("no dfn"))),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0)
    assert result.success is True
    assert result.model_used == "sgmse_plus_torchscript"
    assert "miipher:not_loaded" in result.fallback_chain
    assert result.metadata["miipher_model_loaded"] is False
    assert result.metadata["sgmse_model_loaded"] is True
    assert result.metadata["miipher_compensation_active"] is True
    assert result.metadata["miipher_compensation_dfn_applied"] is False
    assert "capability_status" in result.metadata
    np.testing.assert_allclose(result.audio, audio * 0.5, atol=1e-7)


def test_router_vocal_nr_compensates_missing_miipher_with_dfn_and_hnr(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipher:
        _model_loaded = False

    class _FakeSgmseResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = (audio * 0.5).astype(np.float32)
            self.model_used = "sgmse_plus_torchscript"

    class _FakeSgmse:
        _model_loaded = True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int):  # pylint: disable=unused-argument
            return _FakeSgmseResult(audio)

    class _FakeDfn:
        @staticmethod
        def enhance(audio: np.ndarray, sr: int, energy_bias_db: float = -9.0):  # pylint: disable=unused-argument
            return (audio * 0.8).astype(np.float32)

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipher()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.sgmse_plugin",
        types.SimpleNamespace(get_sgmse_plugin=lambda: _FakeSgmse()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: _FakeDfn()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "backend.core.dsp.hnr_guard",
        types.SimpleNamespace(apply_hnr_blend=lambda pre, post, sr: (post * 0.9).astype(np.float32)),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0)
    assert result.success is True
    assert result.model_used.endswith("+deepfilternet_v3_ii+hnr_blend")
    assert result.model_used.startswith("sgmse_plus")
    assert result.metadata["miipher_compensation_active"] is True
    assert result.metadata["miipher_compensation_dfn_applied"] is True
    assert result.metadata["miipher_compensation_hnr_applied"] is True
    np.testing.assert_allclose(result.audio, audio * 0.36, atol=1e-7)


def test_router_accepts_productive_miipher_adapter_without_native_onnx(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    class _FakeMiipherAdapter:
        _model_loaded = False
        route_metadata = {
            "model_used": "miipher_sgmse_plus",
            "capability_status": "sota_fallback",
            "native_miipher_loaded": False,
        }

        @staticmethod
        def is_productive() -> bool:
            return True

        @staticmethod
        def enhance(audio: np.ndarray, sr: int, noise_snr_db: float = 0.0) -> np.ndarray:  # pylint: disable=unused-argument
            return (audio * 0.25).astype(np.float32)

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.miipher_plugin",
        types.SimpleNamespace(get_miipher_plugin=lambda: _FakeMiipherAdapter()),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_vocal(audio, 48000, energy_bias_db=-6.0, noise_snr_db=4.0)
    assert result.success is True
    assert result.model_used == "miipher_sgmse_plus"
    assert result.metadata["miipher_model_loaded"] is False
    assert result.metadata["miipher_adapter_productive"] is True
    assert result.metadata["capability_status"] == "sota_fallback"
    np.testing.assert_allclose(result.audio, audio * 0.25, atol=1e-7)


def test_router_instrumental_nr_falls_back_cleanly(monkeypatch):
    from backend.core.dsp.sota_vocal_model_router import SotaVocalModelRouter

    monkeypatch.setitem(
        __import__("sys").modules,
        "plugins.deepfilternet_v3_ii_plugin",
        types.SimpleNamespace(get_deepfilternet_plugin=lambda: (_ for _ in ()).throw(RuntimeError("no model"))),
    )

    audio = _audio()
    result = SotaVocalModelRouter().enhance_instrumental(audio, 48000)
    assert result.success is False
    assert result.model_used == "none"
    assert result.fallback_chain
    assert result.metadata["capability_status"] == "dsp_fallback"
    np.testing.assert_allclose(result.audio, audio, atol=1e-7)
