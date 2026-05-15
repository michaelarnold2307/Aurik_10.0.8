"""
Resemblyzer Plugin — Speaker-Embedding (d-vector, GE2E-Loss)
=============================================================

Wrapper um das Resemblyzer-Paket (d-vector, 256-dim) für §2.35c
Singer-Identity-Cosine und §gender_detection Genderklassifikation.

Modell: Resemblyzer VoiceEncoder (GE2E, 256-dim d-vector)
    - Eingabe: 16 kHz Mono float32 (beliebige Länge)
    - Ausgabe: 256-dim L2-normierter Embedding-Vektor

Spec-Referenzen:
    §2.35c:  singer_identity_cosine — VOR und NACH Pipeline messen;
             cos_sim < 0.92 → Phase-Rollback letzte Vokal-Phase
    §0j:     Resemblyzer ist leichtgewichtig → CPU-only (kein GPU-Overhead)
    §3.2:    Singleton + Double-Checked Locking, thread-safe
    §3.1:    NaN/Inf-Guard; Fallback ohne Absturz
    §4.4:    Resemblyzer (dvector, GE2E-Loss) als primäres Speaker-ID-Modell;
             DSP-Fallback: MFCC-Pearson × Centroid-Korrelation

DSP-Fallback:
    Wenn Resemblyzer nicht installiert oder Fehler → embed() gibt None zurück
    → Aufrufer nutzt _compute_singer_identity_dsp() aus vocal_quality_index.py
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, cast

import numpy as np

try:
    import librosa as _librosa
except Exception:
    _librosa = None

try:
    from resemblyzer import VoiceEncoder as _ResemblyzerVoiceEncoder
    from resemblyzer import preprocess_wav as _resemblyzer_preprocess_wav
except Exception:
    _ResemblyzerVoiceEncoder = None
    _resemblyzer_preprocess_wav = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton-Lock (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: ResemblyzerPlugin | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Plugin-Klasse
# ---------------------------------------------------------------------------
class ResemblyzerPlugin:
    """Resemblyzer VoiceEncoder — Speaker-Embedding für §2.35c Singer-Identity.

    CPU-only (§0j: leichtgewichtiges Modell, kein GPU-Overhead gerechtfertigt).
    Thread-sicherer Singleton via Double-Checked Locking (§3.2).

    Invarianten (§3.1):
        - embed() gibt None zurück wenn Modell nicht verfügbar (kein Absturz)
        - Alle Embedding-Vektoren sind L2-normiert ∈ [-1, 1]^256
        - NaN/Inf in Eingaben werden zu 0.0 bereinigt
    """

    # Resemblyzer erwartet 16 kHz Mono
    MODEL_SR: int = 16_000

    def __init__(self) -> None:
        self._encoder: Any | None = None
        self._preprocess_wav_fn: Callable[..., np.ndarray] | None = None
        self._load()

    # ------------------------------------------------------------------
    # Laden
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lädt VoiceEncoder einmalig lazy; warnt bei Fehler, kein Absturz."""
        try:
            if _ResemblyzerVoiceEncoder is None or _resemblyzer_preprocess_wav is None:
                raise ImportError("resemblyzer unavailable")

            # CPU-only: Resemblyzer ist leichtgewichtig (§0j)
            self._encoder = _ResemblyzerVoiceEncoder("cpu")
            self._preprocess_wav_fn = _resemblyzer_preprocess_wav
            logger.info("resemblyzer_plugin: VoiceEncoder loaded (256-dim d-vector, CPU, §2.35c)")
        except Exception as exc:
            logger.warning("resemblyzer_plugin: Resemblyzer nicht verfügbar — DSP-Fallback aktiv: %s", exc)
            self._encoder = None
            self._preprocess_wav_fn = None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True wenn Resemblyzer geladen und einsatzbereit."""
        return self._encoder is not None

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Berechnet 256-dim d-vector Embedding.

        Args:
            audio: Mono oder Stereo float32 ndarray, beliebige SR.
            sr:    Sample-Rate von audio.

        Returns:
            256-dim L2-normierter Embedding-Vektor (float32) oder None bei Fehler.
        """
        if self._encoder is None or self._preprocess_wav_fn is None:
            return None
        try:
            # Zu Mono normieren
            mono = _to_mono(audio)
            mono = np.asarray(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

            # Auf Resemblyzer-SR resamplen (16 kHz)
            if sr != self.MODEL_SR:
                if _librosa is None:
                    return None
                mono = np.asarray(_librosa.resample(mono, orig_sr=sr, target_sr=self.MODEL_SR), dtype=np.float32)

            # preprocess_wav → normiert + getrimmtes float32
            wav = np.asarray(
                self._preprocess_wav_fn(mono, source_sr=self.MODEL_SR),
                dtype=np.float32,
            )

            # embed_utterance → 256-dim d-vector
            emb = np.asarray(cast(Any, self._encoder).embed_utterance(wav), dtype=np.float32)
            emb = np.asarray(np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
            return emb

        except Exception as exc:
            logger.debug("resemblyzer_plugin: embed() Fehler — None zurückgegeben: %s", exc)
            return None

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """L2-normierte Cosinus-Ähnlichkeit ∈ [0, 1] zwischen zwei Embeddings.

        Args:
            emb_a: 256-dim Embedding.
            emb_b: 256-dim Embedding.

        Returns:
            Cosinus-Ähnlichkeit ∈ [0, 1]. NaN-safe.
        """
        a = np.array(emb_a, dtype=np.float64)
        b = np.array(emb_b, dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        cos = float(np.dot(a, b) / denom)
        # Resemblyzer-Embeddings sind L2-normiert → cos ∈ [-1, 1]; clippen auf [0, 1]
        return float(np.clip(cos, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Channels-first (2,N) oder samples-first (N,2) → mono (N,)."""
    if audio.ndim == 1:
        mono_audio = np.asarray(audio, dtype=np.float32)
        return mono_audio
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            mono_audio = np.asarray(audio.mean(axis=0), dtype=np.float32)
            return mono_audio
        if audio.shape[1] == 2:
            mono_audio = np.asarray(audio.mean(axis=1), dtype=np.float32)
            return mono_audio
        if audio.shape[0] == 1:
            mono_audio = np.asarray(audio[0], dtype=np.float32)
            return mono_audio
    mono_audio = np.asarray(audio.flatten(), dtype=np.float32)
    return mono_audio


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2)
# ---------------------------------------------------------------------------


def get_resemblyzer_plugin() -> ResemblyzerPlugin:
    """Gibt den thread-sicheren Singleton zurück (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ResemblyzerPlugin()
    return _instance
