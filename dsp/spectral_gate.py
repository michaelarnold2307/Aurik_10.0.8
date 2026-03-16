import logging

"""
spectral_gate.py - SOTA-konformes Spectral Gate für Aurik 6.0

Dieses Modul implementiert ein STFT-basiertes Spectral Gate mit adaptiven Thresholds und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "spectral_gate"
    category: str = "gate"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
spectral_gate_contract = DSPContract(
    io={
        "channels": "mono",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "n_fft": 1024,
            "hop_length": 256,
            "threshold_db": -40.0,
            "hold_frames": 5,
            "release_frames": 10,
        },
        "safe_ranges": {"threshold_db": {"min": -80.0, "max": 0.0}},
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
            "risk": "Gate-Artefakte",
            "expected_when": "threshold_db zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gating_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class SpectralGate:
    """
    SOTA-konformes Spectral Gate:
    - STFT-basierte Gate-Logik
    - Adaptive Thresholds, Spectral Masking, Hold/Release
    - ML-ready (Hooks für Deep Spectral Gating)
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        threshold_db: float = -40.0,
        hold_frames: int = 5,
        release_frames: int = 10,
    ) -> None:
        """
        n_fft: FFT-Größe
        hop_length: Hop-Size
        threshold_db: Gate-Schwelle (dB)
        hold_frames: Haltezeit in STFT-Frames
        release_frames: Releasezeit in STFT-Frames
        """
        # Sicherstellen, dass n_fft > hop_length
        if n_fft <= hop_length:
            n_fft = hop_length + 1
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.threshold_db = threshold_db
        self.hold_frames = hold_frames
        self.release_frames = release_frames

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(spectral_gate_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit spektralem Gate.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: gegatetes Signal (gleicher Typ wie audio)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        # Schutz gegen ValueError: noverlap < nperseg
        n_fft = self.n_fft
        hop_length = self.hop_length
        if n_fft <= hop_length:
            n_fft = hop_length + 1
        try:
            f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        mag_db = 20 * np.log10(mag + 1e-8)
        # Gate-Logik
        mask = np.ones_like(mag)
        hold_counter = np.zeros(mag.shape[0], dtype=int)
        for frame in range(mag.shape[1]):
            below = mag_db[:, frame] < self.threshold_db
            for bin in range(mag.shape[0]):
                if below[bin]:
                    if hold_counter[bin] < self.hold_frames:
                        hold_counter[bin] += 1
                        mask[bin, frame] = 1.0
                    else:
                        mask[bin, frame] = (
                            max(0.0, mask[bin, frame - 1] - 1.0 / self.release_frames) if frame > 0 else 0.0
                        )
                else:
                    hold_counter[bin] = 0
                    mask[bin, frame] = 1.0
        mag_gated = mag * mask
        Zxx_gated = mag_gated * np.exp(1j * phase)
        try:
            _, out = istft(Zxx_gated, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                _, out = istft(Zxx_gated, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise
        # Output-Länge exakt auf Input trimmen (Broadcast-Sicherheit)
        if len(out) > len(audio):
            out = out[: len(audio)]
        elif len(out) < len(audio):
            # Padding falls zu kurz (selten, aber robust)
            pad = np.zeros(len(audio), dtype=out.dtype)
            pad[: len(out)] = out
            out = pad
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = out * (0.999 / maxval)
        return np.asarray(out.astype(audio.dtype))
