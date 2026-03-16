"""
Adaptive Musical Noise Reduction DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Ermöglicht dynamische Anpassung der Parameter und Integration in adaptive Verarbeitungsketten.
Implementiert ein adaptives Verfahren zur Reduktion von musikalischem Rauschen.
"""

import numpy as np


class AdaptiveMusicalNoiseReduction:
    def __init__(self, median_filter_size=3, threshold=0.1):
        self.median_filter_size = median_filter_size
        self.threshold = threshold

    def reduce(self, mag_spectrogram, **kwargs):
        """Reduziert musikalisches Rauschen im Magnitude-Spektrogramm adaptiv."""
        median_filter_size = kwargs.get("median_filter_size", self.median_filter_size)
        threshold = kwargs.get("threshold", self.threshold)
        # Medianfilter über Frequenzachsen
        filtered = np.copy(mag_spectrogram)
        for t in range(mag_spectrogram.shape[0]):
            filtered[t] = np.median(
                mag_spectrogram[max(0, t - median_filter_size) : t + median_filter_size + 1],
                axis=0,
            )
        # Schwellenwert-Maske
        mask = np.abs(mag_spectrogram - filtered) < threshold * np.max(mag_spectrogram)
        output = np.where(mask, mag_spectrogram, filtered)
        return output

    def auto_optimize(self, mag_spectrogram):
        """Automatische Anpassung der Parameter je nach Signal."""
        std = np.std(mag_spectrogram)
        if std < 0.01:
            self.median_filter_size = 2
            self.threshold = 0.05
        elif std < 0.1:
            self.median_filter_size = 3
            self.threshold = 0.1
        else:
            self.median_filter_size = 5
            self.threshold = 0.2
