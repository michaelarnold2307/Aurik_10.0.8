import numpy as np

from dsp.spectral_denoiser import SpectralDenoiser


def test_spectral_denoiser_reduces_noise():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.2, sr)
    audio = clean + noise
    denoiser = SpectralDenoiser(reduction_db=30.0)
    processed = denoiser.process(audio, sr)
    # Prüfe, ob die Varianz nach Denoising kleiner ist als die des Originalsignals
    assert np.var(processed) < np.var(audio)


def test_spectral_denoiser_identity_on_clean():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Für saubere Signale: minimale Dämpfung, musikalische Integrität
    denoiser = SpectralDenoiser(reduction_db=1.0)
    processed = denoiser.process(clean, sr)
    mae = np.mean(np.abs(processed - clean))
    assert mae < 0.05
    # Hinweis: In produktiven Modulen sollte die Dämpfung und alle Parameter automatisch an Song, Genre und musikalische Ziele angepasst werden (siehe SOTA-Architektur).


def test_spectral_denoiser_extreme_noise():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 1.0, sr)
    audio = clean + noise
    denoiser = SpectralDenoiser(reduction_db=30.0)
    processed = denoiser.process(audio, sr)
    # Bei extremem Rauschen sollte die Varianz deutlich sinken
    assert np.var(processed - clean) < np.var(audio - clean)
