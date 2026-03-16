"""
Broadband Noise Detector
========================

Detects broadband background noise.
"""

import numpy as np
from scipy.signal import welch

from backend.defect_detection.base import DefectDetector, DefectInstance, DefectType


class BroadbandNoiseDetector(DefectDetector):
    """
    Detects broadband background noise.

    Analyzes spectral flatness and SNR to estimate noise floor.
    """

    def __init__(self):
        super().__init__(name="noise_detector", defect_type=DefectType.BROADBAND_NOISE)

    def detect(self, audio: np.ndarray, sr: int, tolerance: float = 0.1, **kwargs) -> list[DefectInstance]:
        """Detect broadband noise, kontextbewusst."""
        if audio.ndim == 2:
            audio = audio[:, 0]
        freqs, psd = welch(audio, sr, nperseg=min(2048, len(audio) // 4))
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        noise_floor_db = 10 * np.log10(np.percentile(psd, 10) + 1e-10)
        signal_rms = np.sqrt(np.mean(audio**2))
        signal_db = 20 * np.log10(signal_rms + 1e-10)
        snr_db = signal_db - noise_floor_db
        snr_severity = max(0.0, (60 - snr_db) / 40.0)
        flatness_severity = spectral_flatness / 0.5
        severity = min((snr_severity + flatness_severity) / 2.0, 1.0)
        if severity < tolerance:
            return []

        # Confidence based on consistency of measurement
        confidence = min(0.6 + severity * 0.3, 0.9)

        metrics = {
            "snr_db": float(snr_db),
            "noise_floor_db": float(noise_floor_db),
            "spectral_flatness": float(spectral_flatness),
            "signal_rms": float(signal_rms),
        }

        description = f"Broadband noise detected: SNR={snr_db:.1f} dB, Spectral Flatness={spectral_flatness:.2f}"

        return [
            self._create_instance(
                severity=severity,
                confidence=confidence,
                metrics=metrics,
                description=description,
            )
        ]
