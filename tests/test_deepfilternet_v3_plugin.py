import numpy as np

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin


def test_deepfilternet_v3_ii_plugin_aurik90():
    np.random.randn(16000).astype(np.float32)
    plugin = DeepFilterNetV3IIPlugin()
    assert isinstance(plugin, DeepFilterNetV3IIPlugin)
