"""
Aurik 9 — ContentAwareProcessor (§2.36)
========================================
Verfeinert die PerceptualAttentionModel-Salienz-Karte anhand von
Wort-Zeitstempeln aus LyricsTranscriber (§2.36).

Prinzip:
    Betonte Frikative (z. B. /s/ in „süß", „sehnsucht") tragen semantischen
    Inhalt — ihre HF-Energie (Bark-Bänder 17–23, ~4–16 kHz) darf von NR
    kaum angetastet werden.  Unbetonte Silben und Stille-Frames können
    aggressiver behandelt werden.

Salienz-Boost-Faktoren (SALIENCY_BOOST, §2.36):
    fricative_stressed:   2.0   → HF-Bänder 17–23 × 2.0
    fricative_unstressed: 1.4
    vowel_stressed:       1.6
    vowel_unstressed:     1.0
    plosive:              1.5
    silence:              0.5
    mixed:                1.0

G_floor-Override (G_FLOOR_FRICATIVE_STRESSED = 0.90):
    Bei fricative+stressed: saliency_map[k, 17:24] mindestens 0.90,
    bevor NR-Gain angewendet wird.

Invarianten (§2.36, §3.1, §3.2):
    - Salienz-Werte ∈ [0.3, 2.0] (PAM-Invariante §2.22)
    - Kein Absenken unter 0.3 (kein völliges Unterdrücken)
    - NaN/Inf-safe überall
    - Thread-safe: Singleton (Double-Checked Locking §3.2)
    - Laufzeit: ≤ 0.5 s Overhead auf LyricsTranscriber-Dauer
    - Datenschutz: Lyrics-Text niemals geloggt
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

# Rückwärtskompatible Importe vom lyrics_transcriber_plugin
try:
    from plugins.lyrics_transcriber_plugin import (  # type: ignore[import]
        LyricsTranscriptionResult,
        WordTimestamp,
    )
except ImportError:
    try:
        from lyrics_transcriber_plugin import (  # type: ignore[import]  # noqa: PLC0415
            LyricsTranscriptionResult,
            WordTimestamp,
        )
    except ImportError:
        # Minimale Stub-Typen damit das Modul auch ohne Plugin ladbar ist
        class LyricsTranscriptionResult:  # type: ignore[no-redef]
            words: list = []
            fallback_used: bool = True
            duration_s: float = 0.0

        class WordTimestamp:  # type: ignore[no-redef]
            word: str = ""
            start_s: float = 0.0
            end_s: float = 0.0
            confidence: float = 0.0
            is_stressed: bool = False
            phoneme_type: str = "mixed"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten (§2.36)
# ---------------------------------------------------------------------------

N_BARK_BANDS: int = 24
"""Anzahl Bark-Bänder — identisch zu PerceptualAttentionModel (§2.22)."""

# HF-Bark-Bänder-Index-Bereich: Bänder 17–23 (~4–16 kHz, §2.36)
HF_BARK_START: int = 17
HF_BARK_END: int = 24   # exclusive (= N_BARK_BANDS)

# Clamp-Grenzen (PAM-Invariante §2.22)
_SALIENCY_MIN: float = 0.3
_SALIENCY_MAX: float = 2.0

# G_floor an fricative+stressed HF-Bins (§2.36)
G_FLOOR_FRICATIVE_STRESSED: float = 0.90

# Salienz-Boost-Tabelle (§2.36)
SALIENCY_BOOST: dict[str, float] = {
    "fricative_stressed":   2.0,
    "fricative_unstressed": 1.4,
    "vowel_stressed":       1.6,
    "vowel_unstressed":     1.0,
    "plosive":              1.5,
    "silence":              0.5,
    "mixed":                1.0,
}

# Frame-Konfiguration (identisch zu PAM §2.22: 500 ms, 250 ms Hop)
_FRAME_DURATION_S: float = 0.5
_FRAME_HOP_S: float = 0.25


# ---------------------------------------------------------------------------
# Öffentliche Hilfsfunktionen
# ---------------------------------------------------------------------------

def _resolve_boost_key(phoneme_type: str, is_stressed: bool) -> str:
    """Leitet den SALIENCY_BOOST-Schlüssel aus Phonem-Typ und Betonung ab.

    Args:
        phoneme_type: "vowel" | "fricative" | "plosive" | "silence" | "mixed"
        is_stressed:  True wenn das Wort betont ist

    Returns:
        Schlüssel aus SALIENCY_BOOST; "mixed" als Fallback.
    """
    if phoneme_type == "fricative":
        return "fricative_stressed" if is_stressed else "fricative_unstressed"
    if phoneme_type == "vowel":
        return "vowel_stressed" if is_stressed else "vowel_unstressed"
    if phoneme_type in SALIENCY_BOOST:
        return phoneme_type
    return "mixed"


def _find_word_at(
    words: list,
    t_center_s: float,
) -> Optional[object]:
    """Findet das Wort, das den Zeitpunkt t_center_s enthält.

    Args:
        words:      Liste von WordTimestamp-Objekten (start_s, end_s)
        t_center_s: Frame-Mittelpunkt in Sekunden

    Returns:
        Das erste passende WordTimestamp oder None.
    """
    for w in words:
        if w.start_s <= t_center_s < w.end_s:
            return w
    return None


# ---------------------------------------------------------------------------
# ContentAwareProcessor (§2.36)
# ---------------------------------------------------------------------------

class ContentAwareProcessor:
    """Verfeinert PAM-Salienz-Karte anhand von Wort-Zeitstempeln (§2.36).

    Anwendung:
        1. LyricsTranscriber.transcribe() → LyricsTranscriptionResult
        2. compute_lyrics_saliency(base_saliency, transcription, sr)
           → verfeinerte Salienz-Karte [n_frames × 24]
        3. PerceptualAttentionModel.apply_to_gain() erhält das Ergebnis
           als optionales lyrics_saliency-Argument

    Invarianten:
        - Salienz-Werte immer ∈ [0.3, 2.0] (PAM-Invariante §2.22)
        - Kein Absenken unter 0.3
        - Thread-safe (Singleton §3.2)
        - Laufzeit: ≤ 0.5 s Overhead
        - Kein direktes Schreiben von Lyrics-Text in Logs (Datenschutz)
    """

    def compute_lyrics_saliency(
        self,
        base_saliency: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int = 48_000,
    ) -> np.ndarray:
        """Verfeinert Salienz-Karte anhand von Wort-Zeitstempeln.

        Algorithmus (§2.36):
            1. Frame-Iteration (500 ms, 250 ms Hop) über base_saliency
            2. Für jeden Frame k: Wort-Mapping via _find_word_at()
            3. Boost-Faktor aus SALIENCY_BOOST
            4. HF-Bänder 17–23 × fricative_boost_factor
            5. G_floor 0.90 an HF-Bändern bei fricative+stressed
            6. Gesamt-clamp auf [0.3, 2.0]

        Args:
            base_saliency: Basis-Salienz-Karte [n_frames × 24]
            transcription: LyricsTranscriptionResult mit WordTimestamp-Liste
            sr:            Sample-Rate (für Frame-Zeitberechnung)

        Returns:
            Verfeinerte Salienz-Karte [n_frames × 24], float32, ∈ [0.3, 2.0]
        """
        # --- NaN-Guard Eingabe ---
        base_saliency = np.nan_to_num(
            base_saliency.astype(np.float32),
            nan=1.0, posinf=_SALIENCY_MAX, neginf=_SALIENCY_MIN,
        )

        if base_saliency.ndim != 2 or base_saliency.shape[1] != N_BARK_BANDS:
            # Rückgabe geclampter Basis bei falscher Form
            return np.clip(base_saliency, _SALIENCY_MIN, _SALIENCY_MAX).astype(np.float32)

        n_frames = base_saliency.shape[0]
        result   = base_saliency.copy()

        # Fallback: keine Wörter oder fallback_used → basis zurückgeben
        words = getattr(transcription, "words", [])
        if not words:
            return np.clip(result, _SALIENCY_MIN, _SALIENCY_MAX).astype(np.float32)

        # Frame-Konfiguration
        frame_hop_s = _FRAME_HOP_S

        for k in range(n_frames):
            t_center = (k + 0.5) * frame_hop_s

            word = _find_word_at(words, t_center)

            if word is None:
                # Kein Wort → silence-Boost auf alle Bänder
                silence_boost = SALIENCY_BOOST["silence"]
                result[k] = result[k] * silence_boost
            else:
                phoneme_type = getattr(word, "phoneme_type", "mixed")
                is_stressed  = bool(getattr(word, "is_stressed", False))
                boost_key    = _resolve_boost_key(phoneme_type, is_stressed)
                boost_factor = SALIENCY_BOOST.get(boost_key, 1.0)

                # HF-Bänder 17–23: Boost anwenden
                result[k, HF_BARK_START:HF_BARK_END] = (
                    result[k, HF_BARK_START:HF_BARK_END] * boost_factor
                )

                # G_floor 0.90 bei fricative+stressed (§2.36)
                if boost_key == "fricative_stressed":
                    hf_slice = result[k, HF_BARK_START:HF_BARK_END]
                    result[k, HF_BARK_START:HF_BARK_END] = np.maximum(
                        hf_slice, G_FLOOR_FRICATIVE_STRESSED
                    )

        # NaN-Guard + Clamp
        result = np.nan_to_num(result, nan=1.0, posinf=_SALIENCY_MAX, neginf=_SALIENCY_MIN)
        result = np.clip(result, _SALIENCY_MIN, _SALIENCY_MAX)
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton + Convenience (§3.2)
# ---------------------------------------------------------------------------

_instance: Optional[ContentAwareProcessor] = None
_lock = threading.Lock()


def get_content_aware_processor() -> ContentAwareProcessor:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ContentAwareProcessor()
    return _instance


def compute_lyrics_saliency(
    base_saliency: np.ndarray,
    transcription: LyricsTranscriptionResult,
    sr: int = 48_000,
) -> np.ndarray:
    """Convenience-Wrapper: verfeinert Salienz-Karte ohne Klassen-Instantiierung."""
    return get_content_aware_processor().compute_lyrics_saliency(
        base_saliency, transcription, sr
    )
