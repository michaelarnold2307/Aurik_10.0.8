"""
Test für Multi-Model Ensemble Processor
"""

import unittest

from ensemble_processor import EnsembleProcessor
from fusion_engine import FusionEngine
import numpy as np

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin
from plugins.resemble_enhance_plugin import ResembleEnhancePlugin


class TestMultiModelEnsemble(unittest.TestCase):
    def test_process_tape_context(self):
        audio = np.random.randn(16000).astype(np.float32)
        dsp_modules = [ResembleEnhancePlugin() for _ in range(2)]
        ml_models = [DeepFilterNetV3IIPlugin() for _ in range(2)]
        fusion_engine = FusionEngine()

        # Kontext-Objekt simulieren (minimal)
        class Context:
            mode = "tape"
            model_registry = {
                "resemble_enhance": {"model": dsp_modules[0]},
                "deepfilternet_v3_ii": {"model": ml_models[0]},
            }

            def activate_team(self, chain):
                pass

        processor = EnsembleProcessor(Context(), fusion_engine)
        result, results, chain = processor.process(audio)
        assert isinstance(result, np.ndarray) or isinstance(result, float)

    def test_process_digital_context(self):
        audio = np.random.randn(16000).astype(np.float32)
        dsp_modules = [ResembleEnhancePlugin() for _ in range(2)]
        ml_models = [DeepFilterNetV3IIPlugin() for _ in range(2)]
        fusion_engine = FusionEngine()

        class Context:
            mode = "digital"
            model_registry = {
                "resemble_enhance": {"model": dsp_modules[0]},
                "deepfilternet_v3_ii": {"model": ml_models[0]},
            }

            def activate_team(self, chain):
                pass

        processor = EnsembleProcessor(Context(), fusion_engine)
        result, results, chain = processor.process(audio)
        assert isinstance(result, np.ndarray) or isinstance(result, float)

    def test_process_broadcast_context(self):
        audio = np.random.randn(16000).astype(np.float32)
        dsp_modules = [ResembleEnhancePlugin() for _ in range(2)]
        ml_models = [DeepFilterNetV3IIPlugin() for _ in range(2)]
        fusion_engine = FusionEngine()

        class Context:
            mode = "broadcast"
            model_registry = {
                "resemble_enhance": {"model": dsp_modules[0]},
                "deepfilternet_v3_ii": {"model": ml_models[0]},
            }

            def activate_team(self, chain):
                pass

        processor = EnsembleProcessor(Context(), fusion_engine)
        result, results, chain = processor.process(audio)
        assert isinstance(result, np.ndarray) or isinstance(result, float)

    def test_process_equal_weights(self):
        audio = np.random.randn(16000).astype(np.float32)
        dsp_modules = [ResembleEnhancePlugin() for _ in range(2)]
        ml_models = [DeepFilterNetV3IIPlugin() for _ in range(2)]
        fusion_engine = FusionEngine()

        class Context:
            mode = "restoration"
            model_registry = {
                "resemble_enhance": {"model": dsp_modules[0]},
                "deepfilternet_v3_ii": {"model": ml_models[0]},
            }

            def activate_team(self, chain):
                pass

        processor = EnsembleProcessor(Context(), fusion_engine)
        result, results, chain = processor.process(audio)
        assert isinstance(result, np.ndarray) or isinstance(result, float)

    def test_process_vinyl_context(self):
        audio = np.random.randn(16000).astype(np.float32)
        dsp_modules = [ResembleEnhancePlugin() for _ in range(2)]
        ml_models = [DeepFilterNetV3IIPlugin() for _ in range(2)]
        fusion_engine = FusionEngine()

        class Context:
            mode = "vinyl"
            model_registry = {
                "resemble_enhance": {"model": dsp_modules[0]},
                "deepfilternet_v3_ii": {"model": ml_models[0]},
            }

            def activate_team(self, chain):
                pass

        processor = EnsembleProcessor(Context(), fusion_engine)
        result, results, chain = processor.process(audio)
        assert isinstance(result, np.ndarray) or isinstance(result, float)


if __name__ == "__main__":
    unittest.main()
