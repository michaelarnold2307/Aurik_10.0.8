import numpy as np
import pytest


def test_madmom_plugin_desktop():
    """Testet, ob MadmomPlugin instanziiert werden kann und eine Dummy-Audioverarbeitung durchführt (Desktop-Only)."""
    try:
        from plugins.madmom_plugin import MadmomPlugin
    except ImportError:
        pytest.skip("MadmomPlugin nicht verfügbar oder nicht installiert.")
    plugin = MadmomPlugin()
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = plugin.process(audio, sr)
    assert result is not None
