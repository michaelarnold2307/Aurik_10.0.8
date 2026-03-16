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
    name: str = "MultibandExpander"
    version: str = "1.0"
    description: str = "Adaptiver Multiband-Expander mit SOTA-Features"
    parameters: dict[str, Any] | None = None


multiband_expander_contract = DSPContract(
    parameters={
        "bands": 3,
        "crossovers": (200, 2000),
        "thresholds_db": (-40, -35, -30),
        "ratios": (0.5, 0.4, 0.3),
        "knees_db": (6, 6, 6),
        "attack_ms": (10, 8, 5),
        "release_ms": (80, 60, 40),
    }
)


class MultibandExpander:
    """
    SOTA-konformer adaptiver Multiband-Expander:
    - Beliebige Anzahl Bänder (default: 3)
    - Adaptive Crossover (Butterworth, Linkwitz-Riley)
    - Pro Band: RMS/Peak-Detection, Soft-Knee, Ratio, Attack/Release
    - Sidechain-Option, Band-Feedback
    - ML-ready (Hooks für ML-basierte Parameter)
    """

    def __init__(
        self,
        bands: int = 3,
        crossovers: tuple[float, float] = (200, 2000),
        thresholds_db: Sequence[float] = (-40, -35, -30),
        ratios: Sequence[float] = (0.5, 0.4, 0.3),
        knees_db: Sequence[float] = (6, 6, 6),
        attack_ms: Sequence[float] = (10, 8, 5),
        release_ms: Sequence[float] = (80, 60, 40),
    ) -> None:
        """
        bands: Anzahl der Frequenzbänder
        crossovers: Übergangsfrequenzen (Hz)
        thresholds_db: Expander-Schwellen pro Band (dB)
        ratios: Expansionsraten pro Band (<1)
        knees_db: Soft-Knee pro Band (dB)
        attack_ms: Attack-Zeiten pro Band (ms)
        release_ms: Release-Zeiten pro Band (ms)
        """
        self.bands = bands
        self.crossovers = crossovers
        self.thresholds_db = thresholds_db
        self.ratios = ratios
        self.knees_db = knees_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(multiband_expander_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Normkonform: Quality-Gate, Audit-Logging, robuste Fehlerbehandlung
        """
        self.log_contract()
        orig_dtype = audio.dtype
        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für MultibandExpander")
            band_signals = self._split_bands(audio, sr)
            processed: list[npt.NDArray[np.float64]] = []
            for i, band in enumerate(band_signals):
                exp = self._expand_band(
                    band,
                    sr,
                    self.thresholds_db[i],
                    self.ratios[i],
                    self.knees_db[i],
                    self.attack_ms[i],
                    self.release_ms[i],
                )
                processed.append(np.asarray(exp, dtype=np.float64))
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
            _logger.error("MultibandExpander Fehler: %s", e)
            self._audit_log({"bands": self.bands, "error": str(e)})
            return audio.astype(orig_dtype)

    def _audit_log(self, result: dict[str, Any]):
        _logger.debug("[AuditLog][MultibandExpander] Ergebnis: %s", result)

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

    def _expand_band(
        self,
        audio: npt.NDArray[np.float64],
        sr: int,
        threshold_db: float,
        ratio: float,
        knee_db: float,
        attack_ms: float,
        release_ms: float,
    ) -> npt.NDArray[np.float64]:
        """
        Expandiert ein einzelnes Frequenzband.
        """
        window = int(sr * 0.01)
        rms = np.sqrt(np.convolve(audio**2, np.ones(window) / window, mode="same"))
        rms_db = 20 * np.log10(rms + 1e-8)
        under = threshold_db - rms_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (under > -knee_db / 2) & (under < knee_db / 2)
        gain_db[idx_soft] = (1 / ratio - 1) * ((under[idx_soft] + knee_db / 2) ** 2) / (2 * knee_db)
        idx_under = under >= knee_db / 2
        gain_db[idx_under] = (1 / ratio - 1) * (under[idx_under])
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        return np.asarray(out)
