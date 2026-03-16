"""
Adaptive Minimum Statistics DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten.
Implementiert das Minimum-Statistics-Verfahren für Noise Estimation.
"""

import numpy as np


class AdaptiveMinimumStatistics:
    def __init__(self, win_length=20, noise_floor=1e-6):
        self.win_length = win_length  # Fensterlänge für Minimumsuche
        self.noise_floor = noise_floor

    def estimate_noise(self, power_spectrogram):
        """Schätzt das Noise-Power-Spektrum adaptiv mit Minimum Statistics."""
        n_frames, n_bins = power_spectrogram.shape
        noise_psd = np.zeros_like(power_spectrogram)
        min_buffer = np.full((self.win_length, n_bins), np.inf)
        for t in range(n_frames):
            min_buffer[t % self.win_length] = power_spectrogram[t]
            min_psd = np.min(min_buffer, axis=0)
            noise_psd[t] = np.maximum(min_psd, self.noise_floor)
        return noise_psd

    def auto_optimize(self, power_spectrogram):
        """Automatische Anpassung der Parameter je nach Signal."""
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.win_length = 5
        elif n_frames < 200:
            self.win_length = 10
        else:
            self.win_length = 20
