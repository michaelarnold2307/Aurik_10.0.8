"""
Adaptive MCRA Noise Estimation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive MCRA-Rauschschätzung mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_mcra"
    category: str = "noise_estimation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveMCRA:
    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.98,
        win_length: int = 20,
        noise_floor: float = 1e-6,
    ):
        self.alpha = alpha  # Glättungsfaktor für Power-Schätzung
        self.beta = beta  # Glättungsfaktor für Minimum-Tracking
        self.win_length = win_length  # Fensterlänge für Minimumsuche
        self.noise_floor = noise_floor

    def estimate_noise(self, power_spectrogram: np.ndarray) -> np.ndarray:
        """Schätzt das Noise-Power-Spektrum adaptiv mit MCRA."""
        n_frames, n_bins = power_spectrogram.shape
        noise_psd = np.zeros_like(power_spectrogram)
        smoothed_psd = np.zeros(n_bins)
        min_buffer = np.full((self.win_length, n_bins), np.inf)
        for t in range(n_frames):
            smoothed_psd = self.alpha * smoothed_psd + (1 - self.alpha) * power_spectrogram[t]
            min_buffer[t % self.win_length] = smoothed_psd
            min_psd = np.min(min_buffer, axis=0)
            noise_psd[t] = self.beta * (noise_psd[t - 1] if t > 0 else smoothed_psd) + (1 - self.beta) * min_psd
            noise_psd[t] = np.maximum(noise_psd[t], self.noise_floor)
        return noise_psd

    def auto_optimize(self, power_spectrogram: np.ndarray) -> dict[str, float]:
        """Automatische Anpassung der Parameter je nach Signal."""
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.win_length = 5
        elif n_frames < 200:
            self.win_length = 10
        else:
            self.win_length = 20
