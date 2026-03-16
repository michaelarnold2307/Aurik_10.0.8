import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.adaptive_pipeline import AdaptiveProcessingPipeline


def test_pipeline_initialization():
    pipeline = AdaptiveProcessingPipeline()
    assert pipeline.context_analyzer is not None
    assert pipeline.goal_engine is not None
    assert pipeline.quality_control is not None
    assert isinstance(pipeline.log, list)
    assert pipeline.logger is not None


# Erweiterung: Teste Policy-Engine, falls vorhanden
@pytest.mark.skipif(
    not hasattr(AdaptiveProcessingPipeline, "policy_engine"), reason="Policy-Engine nicht implementiert"
)
def test_policy_engine_exists():
    pipeline = AdaptiveProcessingPipeline()
    assert hasattr(pipeline, "policy_engine")
