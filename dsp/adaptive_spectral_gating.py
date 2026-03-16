import logging

"""
adaptive_spectral_gating.py - SOTA-konformes Spectral Gating Modul für Aurik 6.0
Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_gating"
    category: str = "spectral_gating"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_gating_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold_db": -40, "reduction_db": -20},
        "safe_ranges": {
            "threshold_db": {"min": -80, "max": 0},
            "reduction_db": {"min": -60, "max": 0},
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
            "risk": "Verlust von Details",
            "expected_when": "reduction_db zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gating_effect"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralGating:
    def __init__(self, threshold_db=-40, reduction_db=-20):
        self.threshold_db = threshold_db
        self.reduction_db = reduction_db

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_spectral_gating_contract))

    def gate(self, mag_spectrogram, noise_floor=None, **kwargs):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        threshold_db = kwargs.get("threshold_db", self.threshold_db)
        reduction_db = kwargs.get("reduction_db", self.reduction_db)
        mag_db = 20 * np.log10(np.maximum(mag_spectrogram, 1e-8))
        if noise_floor is not None:
            threshold = 20 * np.log10(np.maximum(noise_floor, 1e-8)) + threshold_db
        else:
            threshold = threshold_db
        gated_db = np.where(mag_db < threshold, mag_db + reduction_db, mag_db)
        gated_mag = 10 ** (gated_db / 20)
        return gated_mag

    def auto_optimize(self, mag_spectrogram, noise_floor=None):
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        median_db = np.median(20 * np.log10(np.maximum(mag_spectrogram, 1e-8)))
        self.threshold_db = median_db - 20
        self.reduction_db = -30 if median_db < -40 else -20
