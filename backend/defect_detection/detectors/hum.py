"""
Hum Detector
============

Detects electrical hum (50/60 Hz and harmonics).
"""

import numpy as np
from scipy.signal import welch

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class HumDetector(DefectDetector):
    """
    Detects electrical hum (mains frequency and harmonics).

    Checks for 50 Hz and 60 Hz fundamental + harmonics up to 500 Hz.
    """

    def __init__(self, hum_freqs: list[float] | None = None):
        super().__init__(name="hum_detector", defect_type=DefectType.HUM)
        self.hum_freqs = hum_freqs if hum_freqs is not None else [50.0, 60.0]
        self.max_harmonics = 8

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.5, **kwargs) -> list[DefectInstance]:
        """Detect electrical hum, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]
        freqs, psd = welch(audio, sr, nperseg=min(16384, len(audio) // 2))
        detections = []
        for hum_freq in self.hum_freqs:
            hum_energy = self._detect_hum_frequency(freqs, psd, hum_freq)
            if hum_energy["severity"] > tolerance:
                detections.append(hum_energy)
        if not detections:
            return []
        strongest = max(detections, key=lambda x: x["severity"])
        return [
            self._create_instance(
                severity=strongest["severity"],
                confidence=strongest["confidence"],
                metrics=strongest["metrics"],
                description=strongest["description"],
            )
        ]

    def _detect_hum_frequency(self, freqs: np.ndarray, psd: np.ndarray, hum_freq: float) -> dict:
        """Detect hum at specific frequency and harmonics."""

        # Finde Grundfrequenz (stärkstes Signal über 100 Hz)
        high_freq_mask = freqs > 100
        if np.any(high_freq_mask):
            high_psd = psd[high_freq_mask]
            high_freqs = freqs[high_freq_mask]
            fundamental_idx = np.argmax(high_psd)
            fundamental_freq = high_freqs[fundamental_idx]
            fundamental_energy = high_psd[fundamental_idx]
        else:
            fundamental_freq = 0
            fundamental_energy = 0

        harmonic_energies = []
        harmonic_freqs = []

        for n in range(1, self.max_harmonics + 1):
            harmonic_freq = hum_freq * n

            if harmonic_freq > freqs[-1]:
                break

            # Find bin closest to harmonic
            bin_idx = np.argmin(np.abs(freqs - harmonic_freq))

            # Sum energy in ±2 bins around harmonic
            start_idx = max(0, bin_idx - 2)
            end_idx = min(len(psd), bin_idx + 3)
            energy = np.sum(psd[start_idx:end_idx])

            harmonic_energies.append(energy)
            harmonic_freqs.append(harmonic_freq)

        if not harmonic_energies:
            return {"severity": 0.0, "confidence": 0.0, "metrics": {}, "description": ""}

        # Total hum energy
        total_hum_energy = np.sum(harmonic_energies)

        # Overall signal energy
        total_signal_energy = np.sum(psd)

        # Hum ratio (what percentage of signal is hum)
        hum_ratio = total_hum_energy / (total_signal_energy + 1e-10)

        # Robustheitsprüfung: Wenn Grundfrequenz dominant ist (>100 Hz) und Hum schwach, ignoriere
        if fundamental_energy > 0 and fundamental_freq > 100:
            hum_to_fundamental_ratio = total_hum_energy / (fundamental_energy + 1e-10)
            # Wenn Hum weniger als 5% der Grundfrequenz-Energie ist, ignoriere
            if hum_to_fundamental_ratio < 0.05:
                return {"severity": 0.0, "confidence": 0.0, "metrics": {}, "description": ""}

        # Severity: 0.1% = 0.1, 10% = 1.0
        severity = min(hum_ratio * 10.0, 1.0)

        # Confidence: higher if multiple harmonics detected
        num_significant_harmonics = np.sum(np.array(harmonic_energies) > np.mean(psd) * 2)
        confidence = min(0.5 + (num_significant_harmonics / 8.0) * 0.5, 0.95)

        metrics = {
            "hum_frequency": float(hum_freq),
            "hum_ratio": float(hum_ratio),
            "num_harmonics": len(harmonic_energies),
            "num_significant_harmonics": int(num_significant_harmonics),
            "frequencies": [float(f) for f in harmonic_freqs[:num_significant_harmonics]],  # type: ignore[misc]
            "total_hum_energy_db": float(10 * np.log10(total_hum_energy + 1e-10)),
        }

        description = (
            f"Electrical hum at {hum_freq} Hz ({num_significant_harmonics} harmonics, {hum_ratio * 100:.2f}% of signal)"
        )

        return {
            "severity": severity,
            "confidence": confidence,
            "metrics": metrics,
            "description": description,
        }
