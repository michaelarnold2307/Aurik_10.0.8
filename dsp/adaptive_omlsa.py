"""
adaptive_omlsa.py - SOTA-konformes OMLSA-Modul für Aurik 6.0
Dieses Modul implementiert adaptives OMLSA und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any

import numpy as np
from scipy.special import expn

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_omlsa"
    category: str = "omlsa"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_omlsa_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"alpha": 0.98, "noise_floor": 1e-8},
        "safe_ranges": {
            "alpha": {"min": 0.8, "max": 1.0},
            "noise_floor": {"min": 1e-12, "max": 1e-3},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Fehlanpassung bei falschem Noise-Floor",
            "expected_when": "noise_floor zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["omlsa_gain"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveOMLSA:
    def __init__(self, alpha=0.98, noise_floor=1e-8):
        self.alpha = alpha
        self.noise_floor = noise_floor

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_omlsa_contract))

    def omlsa(self, noisy_mag, noise_mag, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        alpha = kwargs.get("alpha", self.alpha)
        noise_floor = kwargs.get("noise_floor", self.noise_floor)
        # A-priori SNR
        gamma = (noisy_mag**2) / (noise_mag**2 + noise_floor)
        xi = alpha * (noisy_mag**2) / (noise_mag**2 + noise_floor) + (1 - alpha) * np.maximum(gamma - 1, 0)
        # OMLSA Gain (vereinfachte Formel, ähnlich MMSE-LSA)
        v = xi * gamma / (1 + xi)
        v = np.maximum(v, 1e-12)  # §3.1 NaN-Guard: expn(1, 0) → ∞, 0*∞ = NaN bei Stille
        gain = (xi / (1 + xi)) * np.exp(0.5 * expn(1, v))
        gain = np.nan_to_num(gain, nan=0.0, posinf=0.0, neginf=0.0)  # §3.1 NaN/Inf-Guard
        clean_mag = gain * noisy_mag
        return clean_mag

    def auto_optimize(self, noisy_mag, noise_mag):
        """
        Passt alpha und noise_floor adaptiv an den geschätzten SNR an.
        Hohes SNR → alpha nahe 1 (starkes Glätten), niedriger Rauschboden.
        Niedriges SNR → alpha kleiner (schnellere Adaptation), höherer Rauschboden.
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        # SNR-Schätzung: Mittlere Signal- zu Rauschmagnituden-Ratio
        snr = float(np.mean(noisy_mag) / (np.mean(noise_mag) + 1e-8))

        # alpha: 0.85 (niedriger SNR) … 0.99 (hoher SNR)
        self.alpha = float(np.clip(0.85 + 0.14 * np.tanh((snr - 5.0) / 5.0), 0.85, 0.99))

        # noise_floor: schärfer bei hohem SNR, lockerer bei niedrigem
        self.noise_floor = float(np.clip(1e-6 / (snr + 1.0), 1e-8, 1e-5))

        logger.info(
            f"adaptive_omlsa.auto_optimize: SNR={snr:.2f} → alpha={self.alpha:.4f}, "
            f"noise_floor={self.noise_floor:.2e}"
        )
