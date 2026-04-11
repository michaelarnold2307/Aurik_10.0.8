"""
Phase 53: Semantic Audio Analysis v2.0 — DSP Feature Extraction
================================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub (aurik_ml.semantic → ImportError).

KATEGORIE: METADATA — Audio wird NICHT verändert, nur analysiert.

EXTRAHIERTE FEATURES:
  - BPM:           Auto-Korrelation der Onset-Stärke (Bereich 60–180 BPM)
  - Tonart (Key):  Chromagramm-Projektion auf Dur/Moll-Profil (Krumhansl 1990)
  - Genre-Hint:    Spektraler Zentroid + Crest Factor → Grobklassifikation
  - Loudness-Klasse: LUFS-Näherung + Crest Factor
  - Energie-Struktur: Überblick über HF/LF-Energie-Segmente

WICHTIG:
  - process() gibt audio UNVERÄNDERT zurück
  - Alle Analysen landen in PhaseResult.metadata + PhaseResult.metrics

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler Dur/Moll-Profile (1990)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio


def _estimate_bpm(mono: np.ndarray, sr: int) -> float:
    """Auto-Korrelation der Onset-Hüllkurve für BPM-Schätzung."""
    # Hüllkurve über RMS-Fenster (23 ms)
    frame = max(1, int(0.023 * sr))
    hop = max(1, frame // 2)
    n_frames = (len(mono) - frame) // hop
    if n_frames < 4:
        return 120.0
    onset = np.array([float(np.sqrt(np.mean(mono[i * hop : i * hop + frame] ** 2))) for i in range(n_frames)])
    # Auto-Korrelation
    onset -= onset.mean()
    ac = np.correlate(onset, onset, mode="full")
    ac = ac[len(ac) // 2 :]
    # Suchbereich: 60–180 BPM → Periode in Frames
    min_period = int(60.0 / 180.0 * sr / hop)
    max_period = int(60.0 / 60.0 * sr / hop)
    min_period = max(1, min(min_period, len(ac) - 2))
    max_period = max(min_period + 1, min(max_period, len(ac) - 1))
    best_period = min_period + int(np.argmax(ac[min_period:max_period]))
    bpm = 60.0 * sr / (best_period * hop)
    return float(np.clip(bpm, 60.0, 180.0))


def _estimate_key(mono: np.ndarray, sr: int) -> str:
    """Chromagramm + Krumhansl-Profile → Tonart-Schätzung."""
    n_fft = 4096
    hop = 1024
    f, _t, Zxx = sig.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    mag = np.abs(Zxx)

    # Frequenz → Chroma-Bin (12-stufige gleichmäßige Stimmung, A4=440 Hz)
    eps = 1e-8
    freqs = f[1:]  # Gleichstromanteil überspringen
    mag = mag[1:, :]  # entsprechend kürzen
    chroma = np.zeros(12)
    for i, freq in enumerate(freqs):
        if freq < 27.5:
            continue
        midi = 69 + 12 * np.log2(freq / 440.0 + eps)
        chroma_bin = round(midi) % 12
        chroma[chroma_bin] += float(np.mean(mag[i]))

    if chroma.sum() < eps:
        return "C major"

    chroma = chroma / (chroma.sum() + eps)

    # Verschiebe das Profil für alle 12 Tonarten, wähle besten Pearson-r
    best_r = -2.0
    best_key = "C"
    best_mode = "major"
    for root in range(12):
        shifted = np.roll(chroma, -root)
        r_maj = float(np.corrcoef(shifted, _MAJOR_PROFILE)[0, 1])
        r_min = float(np.corrcoef(shifted, _MINOR_PROFILE)[0, 1])
        if r_maj > best_r:
            best_r = r_maj
            best_key = _NOTE_NAMES[root]
            best_mode = "major"
        if r_min > best_r:
            best_r = r_min
            best_key = _NOTE_NAMES[root]
            best_mode = "minor"

    return f"{best_key} {best_mode}"


def _estimate_genre_hint(mono: np.ndarray, sr: int) -> str:
    """Grober Genre-Hint via Spektralzentroid + Crest Factor."""
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
    peak = float(np.max(np.abs(mono))) + 1e-12
    crest = peak / rms

    # Spektralzentroid
    f, psd = sig.periodogram(mono, fs=sr, window="hann")
    total_psd = np.sum(psd) + 1e-12
    centroid = float(np.sum(f * psd) / total_psd)

    if centroid < 1200.0 and crest > 8.0:
        return "classical_orchestral"
    if centroid < 1500.0 and crest > 5.0:
        return "jazz_acoustic"
    if centroid > 3000.0 and crest < 5.0:
        return "electronic_edm"
    if centroid > 2500.0:
        return "rock_metal"
    if centroid < 2000.0:
        return "pop_ballad"
    return "general"


class SemanticAudioPhase(PhaseInterface):
    """Reine Analyse-Phase: DSP Feature Extraction (BPM, Key, Genre, Loudness)."""

    PHASE_ID = "phase_53_semantic_audio"
    PHASE_NAME = "Semantic Audio Analysis"
    PHASE_DESCRIPTION = (
        "Analysiert Audio auf BPM, Tonart (Krumhansl-Profile), Genre-Hint "
        "(Spektralzentroid + Crest Factor) und Lautheitsprofil — OHNE die Audio-Daten zu verändern. "
        "Kein aurik_ml benötigt."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.METADATA,
            priority=2,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.08,
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.75,
            description=self.PHASE_DESCRIPTION,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Analysiert Audio semantisch, gibt unverändertes Audio zurück.

        PhaseResult.metadata enthält:
            bpm, key, genre_hint, loudness_class, crest_factor, spectral_centroid_hz
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        if effective_strength <= 1e-6:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        mono = _mono(audio.astype(np.float64))

        bpm = _estimate_bpm(mono, sample_rate)
        key = _estimate_key(mono, sample_rate)
        genre_hint = _estimate_genre_hint(mono, sample_rate)

        rms = float(np.sqrt(np.mean(mono**2))) + 1e-12
        peak = float(np.max(np.abs(mono))) + 1e-12
        crest_factor = peak / rms
        # Annähernde LUFS-Näherung (nur Energiebasis)
        lufs_approx = 20.0 * np.log10(rms) - 0.691  # vereinfacht

        if lufs_approx > -9.0:
            loudness_class = "very_loud_mastered"
        elif lufs_approx > -14.0:
            loudness_class = "loud_streaming"
        elif lufs_approx > -23.0:
            loudness_class = "moderate"
        else:
            loudness_class = "quiet_dynamic"

        f, psd = sig.periodogram(mono, fs=sample_rate, window="hann")
        centroid = float(np.sum(f * psd) / (np.sum(psd) + 1e-12))

        # ── Tier-0: BEATs iter3 Audio Tagging (SOTA §4.4) ───────────────────────────
        # AudioSet-527-Klassifikation für semantisch reichere Pipeline-Metadaten.
        # Fallback auf DSP-Chromagramm/Zentroid wenn BEATs nicht verfügbar.
        _beats_tags: dict[str, float] = {}
        _beats_model_used = "dsp_only"
        _beats_embedding: list[float] = []
        _beats_top_k: list[tuple[str, float]] = []
        try:
            from backend.core.ml_memory_budget import release as _release_53
            from backend.core.ml_memory_budget import try_allocate as _alloc_53
            from plugins.beats_plugin import get_beats_plugin as _beats_factory

            if _alloc_53("BEATs_phase53", 0.09):
                try:
                    _beats_result = _beats_factory().get_tags(audio, sample_rate, top_k=15)
                    _beats_result.tags
                    _beats_model_used = _beats_result.model_used
                    _beats_top_k = _beats_result.top_k
                    # Speichere die ersten 32 Embedding-Dimensionen (Transport-safe)
                    _beats_embedding = [float(x) for x in _beats_result.embeddings[:32].tolist()]
                    # Überschreibe Genre-Hint wenn BEATs einen klaren Tag liefert
                    if _beats_top_k:
                        _top_tag, _top_conf = _beats_top_k[0]
                        if _top_conf >= 0.40:
                            genre_hint = _top_tag  # BEATs-Tag ersetzt DSP-Heuristik
                    logger.info(
                        "Phase 53: BEATs OK (model=%s, top=%s)",
                        _beats_model_used,
                        [f"{t}({c:.2f})" for t, c in _beats_top_k[:3]],
                    )
                except Exception as _beats_err:
                    logger.debug("Phase 53: BEATs tagging fehlgeschlagen (%s) — DSP-Fallback", _beats_err)
                finally:
                    _release_53("BEATs_phase53")
        except Exception as _imp_err:
            logger.debug("Phase 53: BEATs-Import nicht verfügbar (%s) — DSP-only", _imp_err)

        meta = {
            "bpm": round(bpm, 1),
            "key": key,
            "genre_hint": genre_hint,
            "loudness_class": loudness_class,
            "lufs_approx": round(float(lufs_approx), 1),
            "crest_factor": round(float(crest_factor), 2),
            "spectral_centroid_hz": round(float(centroid), 1),
            "phase_locality_factor": phase_locality_factor,
            "effective_strength": effective_strength,
            # BEATs semantic tags (leer wenn BEATs nicht verfügbar)
            "beats_model_used": _beats_model_used,
            "beats_top_tags": [{"tag": t, "confidence": round(c, 3)} for t, c in _beats_top_k[:10]],
            "beats_embedding_32": _beats_embedding,
        }

        logger.info(
            "Phase 53 SemanticAudio: BPM=%.1f, Key=%s, Genre=%s, Loudness=%s",
            bpm,
            key,
            genre_hint,
            loudness_class,
        )

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=audio,  # UNVERÄNDERT — Kategorie METADATA
            execution_time_seconds=time.time() - t0,
            metadata=meta,
            metrics={"bpm": bpm, "crest_factor": crest_factor, "effective_strength": effective_strength},
        )
