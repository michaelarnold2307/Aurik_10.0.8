"""
Total Harmonic Distortion (THD) Detector
=========================================

Detects harmonic distortion in audio.
"""

import numpy as np
from scipy.signal import welch

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class DistortionDetector(DefectDetector):
    """
    Detects harmonic distortion.

    Measures Total Harmonic Distortion (THD) by analyzing
    ratio of harmonic energy to fundamental energy.
    """

    def __init__(self):
        super().__init__(name="distortion_detector", defect_type=DefectType.DISTORTION)

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.15, **kwargs) -> list[DefectInstance]:
        """Detect harmonic distortion, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]
        freqs, psd = welch(audio, sr, nperseg=min(4096, len(audio) // 4))
        fundamental_idx = np.argmax(psd[10:]) + 10  # Skip DC
        fundamental_freq = freqs[fundamental_idx]
        if fundamental_freq < 50 or fundamental_freq > 5000:
            return []
        thd, harmonic_energies = self._calculate_thd(freqs, psd, fundamental_freq)
        severity = min(thd / 0.05, 1.0)
        if severity < tolerance:
            return []
        fundamental_energy = psd[fundamental_idx]
        mean_energy = np.mean(psd)
        fundamental_prominence = fundamental_energy / (mean_energy + 1e-10)
        confidence = min(0.4 + np.log10(fundamental_prominence + 1) * 0.2, 0.85)
        metrics = {
            "thd_percent": float(thd * 100),
            "fundamental_freq": float(fundamental_freq),
            "fundamental_energy_db": float(10 * np.log10(fundamental_energy + 1e-10)),
            "num_harmonics": len(harmonic_energies),
        }
        description = f"Harmonic distortion: THD={thd*100:.2f}% at {fundamental_freq:.1f} Hz"
        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]

    def _calculate_thd(self, freqs: np.ndarray, psd: np.ndarray, fundamental: float, max_harmonics: int = 6):
        """Calculate Total Harmonic Distortion."""

        # Find fundamental bin
        fund_idx = np.argmin(np.abs(freqs - fundamental))
        fund_energy = psd[fund_idx]

        harmonic_energies = []

        for n in range(2, max_harmonics + 1):
            harmonic_freq = fundamental * n

            if harmonic_freq > freqs[-1]:
                break

            # Find harmonic bin
            harm_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harm_energy = psd[harm_idx]
            harmonic_energies.append(harm_energy)

        # THD = sqrt(sum(harmonic_powers)) / fundamental_power
        if fund_energy > 0 and harmonic_energies:
            thd = np.sqrt(np.sum(harmonic_energies)) / fund_energy
        else:
            thd = 0.0

        return thd, harmonic_energies
