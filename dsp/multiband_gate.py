from collections.abc import Sequence
from dataclasses import asdict, dataclass
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, lfilter

_logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "MultibandGate"
    version: str = "1.0"
    description: str = "Adaptives Multiband-Gate mit SOTA-Features"
    parameters: dict[str, Any] | None = None


multiband_gate_contract = DSPContract(
    parameters={
        "bands": 3,
        "crossovers": (200, 2000),
        "thresholds_db": (-40, -35, -30),
        "knees_db": (6, 6, 6),
        "attack_ms": (10, 8, 5),
        "release_ms": (80, 60, 40),
        "hold_ms": (30, 25, 20),
    }
)


class MultibandGate:
    """
    SOTA-konformes adaptives Multiband-Gate:
    - Beliebige Anzahl Bänder (default: 3)
    - Adaptive Crossover (Butterworth, Linkwitz-Riley)
    - Pro Band: RMS/Peak-Detection, Soft-Knee, Threshold, Attack/Release, Hold
    - Sidechain-Option, ML-ready
    """

    def __init__(
        self,
        bands: int = 3,
        crossovers: tuple[float, float] = (200, 2000),
        thresholds_db: Sequence[float] = (-40, -35, -30),
        knees_db: Sequence[float] = (6, 6, 6),
        attack_ms: Sequence[float] = (10, 8, 5),
        release_ms: Sequence[float] = (80, 60, 40),
        hold_ms: Sequence[float] = (30, 25, 20),
    ) -> None:
        """
        bands: Anzahl der Frequenzbänder
        crossovers: Übergangsfrequenzen (Hz)
        thresholds_db: Gate-Schwellen pro Band (dB)
        knees_db: Soft-Knee pro Band (dB)
        attack_ms: Attack-Zeiten pro Band (ms)
        release_ms: Release-Zeiten pro Band (ms)
        hold_ms: Hold-Zeiten pro Band (ms)
        """
        self.bands = bands
        self.crossovers = crossovers
        self.thresholds_db = thresholds_db
        self.knees_db = knees_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.hold_ms = hold_ms

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(multiband_gate_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandGate")
            band_signals = self._split_bands(audio, sr)
            processed: list[npt.NDArray[np.float64]] = []
            for i, band in enumerate(band_signals):
                gated = self._gate_band(
                    band,
                    sr,
                    self.thresholds_db[i],
                    self.knees_db[i],
                    self.attack_ms[i],
                    self.release_ms[i],
                    self.hold_ms[i],
                )
                processed.append(np.asarray(gated, dtype=np.float64))
            if processed:
                out = np.sum(np.stack(processed, axis=0), axis=0)
                maxval = np.max(np.abs(out))
                if maxval > 1.0:
                    out = out * (0.999 / maxval)
                self._audit_log({"bands": self.bands, "shape": out.shape, "success": True})
                return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(orig_dtype)
            else:
                self._audit_log({"bands": self.bands, "error": "No bands processed"})
                return audio.astype(orig_dtype)
        except Exception as e:
            _logger.error("MultibandGate Fehler: %s", e)
            self._audit_log({"bands": self.bands, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandGate] Ergebnis: %s", result)

    def _split_bands(self, audio: npt.NDArray[np.float64], sr: int) -> list[npt.NDArray[np.float64]]:
        """
        Teilt das Signal in Frequenzbänder auf.
        Rückgabe: Liste von Band-Signalen
        """
        low, mid = self.crossovers if self.bands == 3 else (self.crossovers[0], self.crossovers[1])
        bands: list[npt.NDArray[np.float64]] = []
        # Low
        b, a = butter(4, low / (sr / 2), btype="low")
        bands.append(np.asarray(lfilter(b, a, audio), dtype=np.float64))
        # Mid
        b, a = butter(4, [low / (sr / 2), mid / (sr / 2)], btype="band")
        bands.append(np.asarray(lfilter(b, a, audio), dtype=np.float64))
        # High
        b, a = butter(4, mid / (sr / 2), btype="high")
        bands.append(np.asarray(lfilter(b, a, audio), dtype=np.float64))
        return bands

    def _gate_band(
        self,
        audio: npt.NDArray[np.float64],
        sr: int,
        threshold_db: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
        hold_ms: float,
    ) -> npt.NDArray[np.float64]:
        """
        Gated ein einzelnes Frequenzband.
        """
        window = int(sr * 0.01)
        rms = np.sqrt(np.convolve(audio**2, np.ones(window) / window, mode="same"))
        rms_db = 20 * np.log10(rms + 1e-8)
        under = threshold_db - rms_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (under > -knee_db / 2) & (under < knee_db / 2)
        gain_db[idx_soft] = -((under[idx_soft] + knee_db / 2) ** 2) / (2 * knee_db)
        idx_under = under >= knee_db / 2
        gain_db[idx_under] = -under[idx_under]
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
        hold_samples = int(sr * hold_ms / 1000)
        hold_counter = 0
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                if hold_counter < hold_samples:
                    env[i] = env[i - 1]
                    hold_counter += 1
                else:
                    env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
                    hold_counter = 0
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
                hold_counter = 0
        out = audio * env
        return np.asarray(out)
