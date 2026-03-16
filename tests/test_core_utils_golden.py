import numpy as np

from backend.core.core_utils import normalize_audio


# Golden Sample: Referenzsignal (z.B. Sinus)
def golden_sample():
    t = np.linspace(0, 1, 48000, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def test_normalize_audio_golden():
    ref = golden_sample()
    norm = normalize_audio(ref)
    # Erwartung: Maximalwert exakt 0.999 oder sehr nahe dran
    assert np.isclose(np.max(np.abs(norm)), 0.999, atol=1e-6)
    # Form und Länge müssen erhalten bleiben
    assert norm.shape == ref.shape
    # Signal darf nicht NaN oder Inf enthalten
    assert np.all(np.isfinite(norm))
