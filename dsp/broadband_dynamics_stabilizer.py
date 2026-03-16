from dataclasses import asdict, dataclass, field
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "broadband_dynamics_stabilizer"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


broadband_dynamics_stabilizer_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"target_rms": -18.0, "window_ms": 100.0},
        "safe_ranges": {
            "target_rms": {"min": -30.0, "max": -8.0},
            "window_ms": {"min": 10.0, "max": 500.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.02,
    },
    side_effects=[{"risk": "Pumpen", "expected_when": "window_ms < 30.0", "severity": 0.2}],
    reports={"self_metrics": ["dynamics_stability"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class BroadbandDynamicsStabilizer:
    """
    SOTA-konformer Broadband Dynamics Stabilizer:
    - Stabilisiert die Dynamik ohne Pumpen (z.B. RMS-Tracking, sanfte Gain-Riding)
    """

    def __init__(self, target_rms: float = -18.0, window_ms: float = 100.0):
        self.target_rms = target_rms
        self.window_ms = window_ms

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(broadband_dynamics_stabilizer_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        SOTA-Maximum: RMS-Tracking, Gain-Riding, Quality-Gate gegen Pumpen
        """
        self.log_contract()
        window = int(self.window_ms * sr / 1000)
        if window < 1:
            window = 1
        # RMS pro Fenster
        rms = np.sqrt(np.convolve(audio**2, np.ones(window) / window, mode="same"))
        target_lin = 10 ** (self.target_rms / 20)
        gain = np.where(rms > 0, target_lin / (rms + 1e-8), 1.0)
        # Sanftes Gain-Riding
        smoothed_gain = np.convolve(gain, np.ones(window) // window, mode="same")
        out = audio * smoothed_gain
        # Quality-Gate: Kein Pumpen (Varianz der Gain-Kurve)
        if np.std(smoothed_gain) > 0.5:
            logger.warning("[QualityGate] Warnung: Pumpen erkannt, Rollback aktiviert.")
            return audio
        return out
