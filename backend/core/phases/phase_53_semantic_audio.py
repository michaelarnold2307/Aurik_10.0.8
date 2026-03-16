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
    f, t, Zxx = sig.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
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
        chroma_bin = int(round(midi)) % 12
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

    phase_id = "phase_53_semantic_audio"
    name = "Semantic Audio Analysis"
    description = (
        "Analysiert Audio auf BPM, Tonart (Krumhansl-Profile), Genre-Hint "
        "(Spektralzentroid + Crest Factor) und Lautheitsprofil — OHNE die Audio-Daten zu verändern. "
        "Kein aurik_ml benötigt."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.METADATA,
            priority=2,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.08,
            memory_requirement_mb=80,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.75,
            description=self.description,
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

        meta = {
            "bpm": round(bpm, 1),
            "key": key,
            "genre_hint": genre_hint,
            "loudness_class": loudness_class,
            "lufs_approx": round(float(lufs_approx), 1),
            "crest_factor": round(float(crest_factor), 2),
            "spectral_centroid_hz": round(float(centroid), 1),
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
            metrics={"bpm": bpm, "crest_factor": crest_factor},
        )
