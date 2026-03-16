import logging
import os
from typing import Any

import audioread
import numpy as np
import soundfile as sf

from .carrier_forensics import analyze_carrier_forensics
from .carrier_ml_classifier import classify_carrier_ml

logger = logging.getLogger(__name__)


def detect_carrier(filepath: str, meta: dict[str, Any] | None = None) -> str:
    """
    Versucht, den Tonträger (Schallplatte, Kassette, CD, Band, etc.) anhand von Dateiname, Metadaten oder User-Tag zu erkennen.
    """
    name = os.path.basename(filepath).lower()
    meta = meta or {}
    # Heuristik: Dateiname
    if any(
        x in name
        for x in [
            "vinyl",
            "lp",
            "platte",
            "schallplatte",
            "33rpm",
            "45rpm",
            "78rpm",
            "shellac",
            "schellack",
        ]
    ):
        return "Schallplatte (Vinyl/Schellack)"
    if any(x in name for x in ["cassette", "mc", "kassette", "tape"]):
        return "Kassette/Band"
    if any(x in name for x in ["cd", "compactdisc", "compact_disc"]):
        return "CD"
    if any(x in name for x in ["minidisc", "md"]):
        return "MiniDisc"
    if any(x in name for x in ["dat", "digitalaudiotape"]):
        return "DAT"
    if any(x in name for x in ["reel", "tonband", "spule"]):
        return "Tonband/Reel"
    # Heuristik: Metadaten
    for k, v in (meta.items() if meta else []):
        vstr = str(v).lower()
        if "vinyl" in vstr or "lp" in vstr or "platte" in vstr or "schellack" in vstr:
            return "Schallplatte (Vinyl/Schellack)"
        if "cassette" in vstr or "mc" in vstr or "kassette" in vstr or "tape" in vstr:
            return "Kassette/Band"
        if "cd" in vstr:
            return "CD"
        if "minidisc" in vstr or "md" in vstr:
            return "MiniDisc"
        if "dat" in vstr:
            return "DAT"
        if "reel" in vstr or "tonband" in vstr or "spule" in vstr:
            return "Tonband/Reel"
    return "Unbekannt"


def is_supported_audio_file(filename: str) -> bool:
    """Prüft ob Datei ein unterstütztes Audioformat ist."""
    # Unterstützte Formate: WAV, MP3, FLAC, OGG, AAC, AIFF (erweiterbar)
    return filename.lower().endswith(
        (
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".aac",
            ".aiff",
            ".aif",
            ".wma",
            ".opus",
            ".m4a",
            ".alac",
            ".caf",
        )
    )


def load_audio_file(
    filepath: str,
    target_sr: int | None = None,
    mono: bool = False
) -> dict[str, Any] | None:
    """Lädt Audiodatei robust und gibt Dict mit allen Metadaten zurück.

    Args:
        filepath: Pfad zur Audiodatei
        target_sr: Ziel-Sample-Rate (optional, Resampling wenn nötig)
        mono: Wenn True, konvertiere zu Mono

    Returns:
        Dict mit 'audio', 'sr', 'channels', 'format', 'duration', 'meta', 'error'
    """
    result: dict[str, Any] = {
        "audio": None,
        "sr": None,
        "channels": None,
        "format": None,
        "duration": None,
        "meta": {},
        "carrier": None,
        "error": None,
    }
    try:
        if not os.path.isfile(filepath):
            result["error"] = "File not found"
            return result
        if not is_supported_audio_file(filepath):
            result["error"] = "Unsupported file type"
            return result
        # Metadaten via soundfile.info() – schnell, kein GStreamer/audioread-Hang
        # audioread.audio_open() für MP3 auf Linux (GStreamer-Backend) kann hängen → verboten
        try:
            _info = sf.info(filepath)
            result["format"] = _info.format
            result["channels"] = _info.channels
            result["sr"] = _info.samplerate
            result["duration"] = _info.duration
            result["meta"] = dict(getattr(_info, "extra_info", {}) or {})
        except Exception:
            # MP3/AAC/WMA: soundfile.info() schlägt fehl → nur Extension (kein audioread)
            result["format"] = os.path.splitext(filepath)[1][1:].upper()
        # Hauptweg: soundfile (liest auch Metadaten, Unicode, große Dateien)
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            if target_sr and sr != target_sr:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            if mono and audio.ndim > 1:
                audio = audio.mean(axis=-1)
            # NaN/Inf-Guard (§3.1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            result["audio"] = audio
            result["sr"] = sr
            result["channels"] = 1 if audio.ndim == 1 else audio.shape[-1]
            result["format"] = result["format"] or os.path.splitext(filepath)[1][1:].upper()
            result["duration"] = result["duration"] or (len(audio) / sr)
            # Tonträger-Erkennung (Heuristik)
            result["carrier_heuristic"] = detect_carrier(filepath, result["meta"])
            # Forensische Tonträger-Erkennung
            try:
                forensic = analyze_carrier_forensics(audio, sr)
                result["carrier_forensic"] = forensic["carrier_forensic"]
                result["carrier_forensic_score"] = forensic["score"]
                result["carrier_forensic_features"] = forensic["features"]
                # ML-basierte Tonträgerklassifikation
                ml = classify_carrier_ml(forensic["features"])
                result["carrier_ml"] = ml.get("carrier_ml", "Unbekannt")
                result["carrier_ml_confidence"] = ml.get("confidence", 0.0)
                result["carrier_ml_probas"] = ml.get("probas", {})
                result["carrier_ml_explain"] = ml.get("explain", "")
            except Exception as e:
                result["carrier_forensic"] = "Unbekannt"
                result["carrier_forensic_score"] = 0
                result["carrier_forensic_features"] = {}
                result["carrier_ml"] = "Unbekannt"
                result["carrier_ml_confidence"] = 0.0
                result["carrier_ml_probas"] = {}
                result["carrier_ml_explain"] = str(e)
        except Exception as e:
            result["error"] = f"Audio read error: {e}"
    except Exception as e:
        result["error"] = f"Unknown error: {e}"
    return result
