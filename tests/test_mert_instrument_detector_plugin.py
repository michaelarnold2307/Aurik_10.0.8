import numpy as np
import pytest


def test_mert_instrument_detector_plugin_desktop():
    """Testet, ob MERTInstrumentDetectorPlugin instanziiert werden kann und eine Dummy-Audioverarbeitung durchführt (Desktop-Only)."""
    try:
        from plugins.mert_instrument_detector_plugin import MERTInstrumentDetectorPlugin
    except ImportError:
        pytest.skip("MERTInstrumentDetectorPlugin nicht verfügbar oder nicht installiert.")
    plugin = MERTInstrumentDetectorPlugin()
    sr = 16000
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    result = plugin.process(audio, sr)
    assert result is not None
