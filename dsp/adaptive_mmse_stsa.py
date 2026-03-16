"""
Adaptive Ephraim-Malah MMSE-STSA DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Modular vorbereitet für spätere Erweiterung und Integration von SOTA-Algorithmen.
Implementiert eine Basisversion des MMSE-STSA-Algorithmus für Magnitude-Spektren.
"""

import numpy as np
from scipy.special import expn


class AdaptiveMMSESTSA:
    def __init__(self, alpha=0.98, noise_floor=1e-8):
        self.alpha = alpha
        self.noise_floor = noise_floor

    def mmse_stsa(self, noisy_mag, noise_mag, **kwargs):
        """Berechnet das MMSE-STSA Gain für Magnitude-Spektren (vereinfachte Version)."""
        alpha = kwargs.get("alpha", self.alpha)
        noise_floor = kwargs.get("noise_floor", self.noise_floor)
        # A-priori SNR
        gamma = (noisy_mag**2) / (noise_mag**2 + noise_floor)
        xi = alpha * (noisy_mag**2) / (noise_mag**2 + noise_floor) + (1 - alpha) * np.maximum(gamma - 1, 0)
        # MMSE-STSA Gain (vereinfachte Formel)
        v = xi * gamma / (1 + xi)
        gain = (xi / (1 + xi)) * np.exp(0.5 * expn(1, v))
        clean_mag = gain * noisy_mag
        return clean_mag

    def auto_optimize(self, noisy_mag, noise_mag):
        """Adaptiert alpha (a-priori-SNR-Gewichtung) anhand der Signal-Dynamik (wie MMSE-LSA)."""
        mean_s = float(np.mean(noisy_mag**2)) + 1e-12
        mean_n = float(np.mean(noise_mag**2)) + 1e-12
        snr_db = float(10 * np.log10(mean_s / mean_n))
        self.alpha = float(np.clip(0.85 + 0.0065 * snr_db, 0.80, 0.99))
