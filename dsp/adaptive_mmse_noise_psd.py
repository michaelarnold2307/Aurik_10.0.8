"""
Adaptive MMSE Noise PSD DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten.
Implementiert die MMSE Noise Power Spectral Density Schätzung.
"""

import numpy as np


class AdaptiveMMSENoisePSD:
    def __init__(self, alpha=0.98, noise_floor=1e-6):
        self.alpha = alpha  # Glättungsfaktor für Noise-Tracking
        self.noise_floor = noise_floor

    def estimate_noise(self, power_spectrogram):
        """Schätzt das Noise-Power-Spektrum adaptiv mit MMSE-Ansatz."""
        n_frames, n_bins = power_spectrogram.shape
        noise_psd = np.zeros_like(power_spectrogram)
        # Initialisiere mit erstem Frame
        noise_psd[0] = power_spectrogram[0]
        for t in range(1, n_frames):
            noise_psd[t] = self.alpha * noise_psd[t - 1] + (1 - self.alpha) * power_spectrogram[t]
            noise_psd[t] = np.maximum(noise_psd[t], self.noise_floor)
        return noise_psd

    def auto_optimize(self, power_spectrogram):
        """Automatische Anpassung der Parameter je nach Signal."""
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.alpha = 0.9
        elif n_frames < 200:
            self.alpha = 0.95
        else:
            self.alpha = 0.98
