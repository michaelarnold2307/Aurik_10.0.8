from hypothesis import given, strategies as st
import numpy as np

from backend.core.core_utils import normalize_audio


def arrays(dtype=np.float32, min_size=1, max_size=48000):
    return st.lists(st.floats(-1.0, 1.0), min_size=min_size, max_size=max_size).map(lambda l: np.array(l, dtype=dtype))


@given(audio=arrays())
def test_normalize_audio_property(audio):
    norm = normalize_audio(audio)
    if len(audio) == 0:
        assert np.all(norm == 0)
    else:
        assert np.max(np.abs(norm)) <= 1.0
        assert norm.shape == audio.shape
