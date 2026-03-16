import logging
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

# §4.4: MP-SENet 2023 ersetzt FullSubNetPlusPlugin + DCCRNPlugin (beide entfernt, verboten)
try:
    from plugins.mp_senet_plugin import MpSenetPlugin
except ImportError as e:  # pragma: no cover
    MpSenetPlugin = None  # type: ignore[misc,assignment]
    _log.warning("MpSenetPlugin nicht verfügbar: %s", e)

try:
    from plugins.demucs_v4_plugin import DemucsV4Plugin
except ImportError as e:  # pragma: no cover
    DemucsV4Plugin = None  # type: ignore[misc,assignment]
    _log.warning("DemucsV4Plugin nicht verfügbar: %s", e)

try:
    from plugins.wpe_plugin import WpePlugin
except ImportError as e:  # pragma: no cover
    WpePlugin = None  # type: ignore[misc,assignment]
    _log.warning("WpePlugin nicht verfügbar: %s", e)


class SOTAUniversalEnhancer:
    """
    Wählt und nutzt automatisch das optimale SOTA-Modell für Sprache, Musik oder gemischte Inhalte.
    """

    def __init__(self, mode: Literal["auto", "speech", "music", "mix"] = "auto"):
        self.mode = mode
        # §4.4: MP-SENet 2023 übernimmt Sprach-/Fallback-Enhancement (ersetzt FullSubNet+/DCCRN)
        self.speech_model = MpSenetPlugin() if MpSenetPlugin else None
        self.music_model = DemucsV4Plugin() if DemucsV4Plugin else None
        self.mix_model = WpePlugin() if WpePlugin else None
        self.fallback_model = MpSenetPlugin() if MpSenetPlugin else None

    def detect_type(self, audio: np.ndarray, sr: int) -> str:
        # Platzhalter: In der Praxis sollte hier ein KI- oder Heuristik-basierter Musik/Sprache/Mix-Detektor stehen
        # Für Demo: Wenn Varianz > 1.5 -> Musik, sonst Sprache
        if np.var(audio) > 1.5:
            return "music"
        return "speech"

    def process(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.mode == "auto":
            typ = self.detect_type(audio, sr)
        else:
            typ = self.mode
        if typ == "speech" and self.speech_model:
            # §4.4: MP-SENet 2023 — enhance(audio, sr) → MpSenetResult
            result = self.speech_model.enhance(audio, sr)
            enhanced = np.nan_to_num(
                result.audio if hasattr(result, "audio") else result,
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            return np.clip(enhanced, -1.0, 1.0)
        elif typ == "music" and self.music_model:
            return self.music_model.process(audio, sr)
        elif typ == "mix" and self.mix_model:
            # WpePlugin: enhance(audio, sr) → ndarray
            result = self.mix_model.enhance(audio, sr)
            return np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        elif self.fallback_model:
            result = self.fallback_model.enhance(audio, sr)
            out = result.audio if hasattr(result, "audio") else result
            return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        else:
            raise RuntimeError("Kein passendes SOTA-Modell für diesen Signaltyp verfügbar.")
