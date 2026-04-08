import os
import sys
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

# Third-party deprecations emitted by optional plugin/model imports during
# AdaptiveProcessingPipeline initialization (outside our code ownership).
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:torch\.nn\.utils\.weight_norm is deprecated in favor of torch\.nn\.utils\.parametrizations\.weight_norm\.:UserWarning"
    ),
    pytest.mark.filterwarnings(r"ignore:pkg_resources is deprecated as an API\.:UserWarning"),
]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.adaptive_pipeline import AdaptiveProcessingPipeline


def test_pipeline_initialization():
    pipeline = AdaptiveProcessingPipeline()
    assert pipeline.context_analyzer is not None
    assert pipeline.goal_engine is not None
    assert pipeline.quality_control is not None
    assert isinstance(pipeline.log, list)
    assert pipeline.logger is not None


# Erweiterung: Teste Policy-Engine (im __init__ gesetzt)
def test_policy_engine_exists():
    pipeline = AdaptiveProcessingPipeline()
    assert hasattr(pipeline, "policy_engine")


def test_run_uses_authoritative_medium_transfer_chain(monkeypatch):
    pipeline = AdaptiveProcessingPipeline()

    medium_result = SimpleNamespace(
        transfer_chain=["vinyl", "cassette", "mp3_low"],
        medium_confidences=[0.82, 0.71, 0.66],
        primary_material="vinyl",
        confidence=0.82,
    )

    audio = np.zeros(4800, dtype=np.float32)
    buffer = BytesIO()
    sf.write(buffer, audio, 48000, format="WAV")

    captured: dict[str, object] = {}

    def _stop_after_medium_capture(audio_np, sr_audio, file_path=None, metadata=None):
        captured["metadata"] = metadata
        raise RuntimeError("stop-after-medium")

    monkeypatch.setattr(pipeline.audio_monitor, "capture_baseline", _stop_after_medium_capture)

    with pytest.raises(RuntimeError, match="stop-after-medium"):
        pipeline.run(buffer.getvalue(), {"medium_result": medium_result, "file_path": "demo.wav"})

    assert pipeline.log[0]["step"] == "media_chain_detection"
    assert pipeline.log[0]["media_chain"] == [
        {"medium": "vinyl", "confidence": 0.82},
        {"medium": "cassette", "confidence": 0.71},
        {"medium": "mp3_low", "confidence": 0.66},
    ]
    assert captured["metadata"] == {
        "detected_medium": {"type": "vinyl", "confidence": 0.82},
        "user_profile": None,
    }
