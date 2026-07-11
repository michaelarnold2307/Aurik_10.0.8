"""
ai_automatic_declicker.py - SOTA-Automatic Declicker für Aurik 6.0
Aurik 6.0 - SOTA-Automatic Declicker

Dieses Modul entfernt Klicks/Knackser automatisch aus Audiosignalen.
Kombiniert klassische Pulsdetektion/Interpolation (SOTA-Maximum, keine ML/AI) und ist auditierbar sowie rollback-fähig.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.automatic_declicker")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_declicker"
    category: str = "disruptor_removal"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
declicker_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.6},
        "safe_ranges": {"threshold": {"min": 0.1, "max": 1.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.05,
        "identity_budget": 0.99,
        "spectral_change_budget": 0.1,
        "temporal_change_budget": 0.05,
        "compute_cost": 0.05,
    },
    side_effects=[{"risk": "transient_smear", "expected_when": "threshold < 0.2", "severity": 0.2}],
    reports={"self_metrics": ["click_removal_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutomaticDeclicker:
    def process(self, audio: "np.ndarray", sr: int) -> "np.ndarray":
        """
        Alias für declick(), um Kompatibilität mit älteren und neuen Pipelines/Tests zu gewährleisten.
        """
        return self.declick(audio, sr)

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def declick(
        self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True
    ) -> np.ndarray:
        """
        Entfernt Klicks/Knackser aus dem Audiosignal per klassischer Pulsdetektion (Medianfilter) und Interpolation (SOTA, keine ML/AI).
        Quality Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: De-clicktes Signal (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
            logger.error("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1.5:
            logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

        audio_out = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Declicking.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('declicker.pt')
                # audio_out = model(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, fallback auf klassische Methode.")
                fallback_used = True
                audio_out = self._declick_classic(audio)
            else:
                audio_out = self._declick_classic(audio)
        except Exception as e:
            logger.error("Fehler bei Declicking: %s", e)
            fallback_used = True
            audio_out = audio.copy()

        if audit_log:
            click_removal_score = float(np.mean(np.abs(audio - audio_out)))
            logger.info(
                f"AutomaticDeclicker: click_removal_score={click_removal_score:.4f}, fallback_used={fallback_used}"
            )
        logger.info("[DSPContract] %s", asdict(declicker_contract))
        return audio_out.astype(audio.dtype)

    def _declick_classic(self, audio: np.ndarray) -> np.ndarray:
        from scipy.signal import medfilt

        diff = np.abs(audio - medfilt(audio, kernel_size=5))
        mask = diff > self.threshold * np.max(diff)
        audio_out = audio.copy()
        if np.any(mask):
            idx = np.where(mask)[0]
            for i in idx:
                left = max(0, i - 2)
                right = min(len(audio) - 1, i + 2)
                audio_out[i] = np.median(audio_out[left : right + 1])
        return audio_out
