import logging

"""
Adaptive CREPE Neural Pitch DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Dummy-Implementierung für ML-basierte Pitch-Analyse (klassische DSP/ML, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_crepe_neural_pitch"
    category: str = "pitch_tracking"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveCREPENeuralPitch:
    """
    SOTA-konformes ML-basiertes Pitch-Tracking (CREPE, Dummy)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def track(self, x: np.ndarray) -> float:
        """Pitch-Tracking via YIN-Algorithmus (scipy/numpy, kein CREPE/DL).

        YIN (de Cheveigne & Kawahara 2002):
          1. Squared-Differenzfunktion d(tau)
          2. Kumulierte normalisierte Differenzfunktion (CMND)
          3. Erstes Minimum unter Schwellwert 0.1 -> Grundfrequenz
        """
        self.log_contract()
        if len(x) < 512:
            return 0.0
        frame = x[:1024].astype(np.float64)
        n = len(frame)
        max_tau = n // 2
        min_tau = max(2, int(self.sr / 1000))
        max_period = min(max_tau, int(self.sr / 60))
        # Differenzfunktion
        d = np.zeros(max_tau)
        for tau in range(1, max_tau):
            diff = frame[: n - tau] - frame[tau:n]
            d[tau] = float(np.sum(diff**2))
        # CMND
        cmnd = np.zeros(max_tau)
        cmnd[0] = 1.0
        running = 0.0
        for tau in range(1, max_tau):
            running += d[tau]
            cmnd[tau] = d[tau] * tau / (running + 1e-12)
        # Erstes lokales Minimum unter Schwellwert
        best_tau = 0
        for tau in range(min_tau, max_period):
            if cmnd[tau] < 0.1:
                while tau + 1 < max_period and cmnd[tau + 1] < cmnd[tau]:
                    tau += 1
                best_tau = tau
                break
        if best_tau < 2:
            best_tau = int(np.argmin(cmnd[min_tau:max_period])) + min_tau
        if best_tau < 2:
            return 0.0
        return round(float(np.clip(float(self.sr) / float(best_tau), 60.0, 1000.0)), 2)

    def auto_optimize(self, x: np.ndarray) -> None:
        """
        Setzt die Samplingrate auf 16 kHz (SOTA-Default für CREPE)
        :param x: Eingabesignal (np.ndarray)
        """
        self.sr = 16000
