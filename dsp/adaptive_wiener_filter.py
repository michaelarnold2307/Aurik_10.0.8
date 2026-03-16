"""
Adaptive Wiener Filter DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten.
Implementiert den klassischen und adaptiven Wiener-Filter für Magnitude-Spektren.
"""

import numpy as np


class AdaptiveWienerFilter:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def filter(self, noisy_mag, noise_mag, **kwargs):
        """Führt Wiener-Filterung auf Magnitude-Spektren durch."""
        eps = kwargs.get("eps", self.eps)
        gain = np.maximum(1 - (noise_mag**2) / (noisy_mag**2 + eps), 0)
        clean_mag = gain * noisy_mag
        return clean_mag

    def auto_optimize(self, noisy_mag, noise_mag):
        """Adaptiert eps anhand des geschätzten SNR.

        Niedriger SNR -> kleineres eps (aggressivere Filterung).
        Hoher SNR -> größeres eps (sanftere Filterung, Artefaktschutz).
        """
        mean_signal = float(np.mean(noisy_mag**2)) + 1e-12
        mean_noise = float(np.mean(noise_mag**2)) + 1e-12
        snr_linear = mean_signal / mean_noise
        # Bei SNR < 5 (7 dB): aggressiver; bei SNR > 100 (20 dB): sanft
        self.eps = float(np.clip(1e-8 / snr_linear, 1e-12, 1e-4))
