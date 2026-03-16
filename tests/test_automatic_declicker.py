import numpy as np
import pytest

from dsp.automatic_declicker import AutomaticDeclicker


def test_declicker_removes_clicks():
    sr = 44100
    audio = np.zeros(sr)
    click_positions = [100, 500, 1000]
    for pos in click_positions:
        audio[pos] = 1.0
    declicker = AutomaticDeclicker(threshold=0.5)
    processed = declicker.process(audio, sr)
    for pos in click_positions:
        assert abs(processed[pos]) < abs(audio[pos])


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_declicker_threshold_variation(threshold):
    sr = 44100
    audio = np.zeros(sr)
    audio[100] = 1.0
    declicker = AutomaticDeclicker(threshold=threshold)
    processed = declicker.process(audio, sr)
    assert processed.shape == audio.shape


def test_declicker_no_clicks():
    sr = 44100
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
    declicker = AutomaticDeclicker()
    processed = declicker.process(audio, sr)
    # Erlaube DSP-typische Abweichung durch Medianfilter, aber keine signifikante Veränderung
    mae = np.mean(np.abs(processed - audio))
    assert mae < 0.01
