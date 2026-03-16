"""
backend/core/temporal_quality_coherence.py
Aurik 9 -- Spec §2.16: TemporalQualityCoherenceMetric

PQS-MOS-Konsistenz ueber Zeitachse gemaess Spec §2.16.
MOS-Spanne <= 0.30, sigma(MOS) <= 0.15.
Min. 3 Segmente (Dateien < 25 s werden nicht geprueft).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TEMPORAL_CONSISTENCY_THRESHOLD: float = 0.30
SIGMA_THRESHOLD: float = 0.15
MIN_FILE_DURATION_S: float = 25.0
SEGMENT_DURATION_S: float = 10.0
SEGMENT_HOP_S: float = 5.0


# ---------------------------------------------------------------------------
# Ergebnis-Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TemporalCoherenceResult:
    """Spec §2.16: Ergebnis der Temporalen Qualitaetskohaerenzmessung."""

    passed: bool  # True wenn max_span <= 0.30 UND sigma <= 0.15
    max_span: float  # max(MOS) - min(MOS) ueber alle Segmente
    sigma: float  # Standardabweichung sigma(MOS)
    mean_mos: float  # Mittlere PQS-MOS ueber alle Segmente
    n_segments: int  # Anzahl ausgewerteter Segmente
    segment_scores: List[float] = field(default_factory=list)  # MOS pro Segment
    skipped: bool = False  # True wenn Datei zu kurz (< 25 s)
    warnings: List[str] = field(default_factory=list)
    message: str = ""  # Laienverständliche Zusammenfassung (Deutsch)

    @property
    def passes(self) -> bool:
        """Alias fuer passed — Rueckwaertskompatibilitaet (§2.16)."""
        return self.passed


# ---------------------------------------------------------------------------
# Klasse
# ---------------------------------------------------------------------------


class TemporalQualityCoherenceMetric:
    """Spec §2.16: Prueft PQS-MOS-Konsistenz ueber die Zeitachse.

    Algorithmus:
        1. Audio in ueberlappende 10-s-Segmente (Hop 5 s)
        2. PQS-MOS-Schaetzung pro Segment (schnelle DSP-Schätzung)
        3. max_span = max(MOS) - min(MOS) <= 0.30
        4. sigma(MOS) <= 0.15
    """

    TEMPORAL_CONSISTENCY_THRESHOLD: float = TEMPORAL_CONSISTENCY_THRESHOLD
    SIGMA_THRESHOLD: float = SIGMA_THRESHOLD
    MIN_FILE_DURATION_S: float = MIN_FILE_DURATION_S

    def measure(self, audio: np.ndarray, sr: int) -> TemporalCoherenceResult:
        """Misst temporale Qualitaetskohaerentz.

        Args:
            audio: float32 1-D oder 2-D
            sr:    Sample-Rate (muss 48000 sein)

        Returns:
            TemporalCoherenceResult
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        duration_s = len(audio) / max(sr, 1)
        if duration_s < self.MIN_FILE_DURATION_S:
            return TemporalCoherenceResult(
                passed=True,
                max_span=0.0,
                sigma=0.0,
                mean_mos=4.0,
                n_segments=0,
                skipped=True,
                warnings=["Datei zu kurz fuer temporale Kohaarenzmessung (< 25 s)."],
            )

        seg_len = int(SEGMENT_DURATION_S * sr)
        hop_len = int(SEGMENT_HOP_S * sr)
        scores: List[float] = []
        i = 0
        while i + seg_len <= len(audio):
            seg = audio[i : i + seg_len]
            score = self._quick_mos(seg, sr)
            if math.isfinite(score):
                scores.append(score)
            else:
                logger.debug("Segment MOS nicht endlich – uebersprungen")
            i += hop_len

        if len(scores) < 3:
            return TemporalCoherenceResult(
                passed=True,
                max_span=0.0,
                sigma=0.0,
                mean_mos=float(np.mean(scores)) if scores else 4.0,
                n_segments=len(scores),
                skipped=True,
                warnings=["Zu wenige Segmente fuer temporale Kohaerenz (< 3)."],
            )

        arr = np.array(scores, dtype=np.float32)
        max_span = float(arr.max() - arr.min())
        sigma = float(arr.std())
        mean_mos = float(arr.mean())
        passes = (max_span <= self.TEMPORAL_CONSISTENCY_THRESHOLD) and (sigma <= self.SIGMA_THRESHOLD)

        return TemporalCoherenceResult(
            passed=passes,
            max_span=max_span,
            sigma=sigma,
            mean_mos=mean_mos,
            n_segments=len(scores),
            segment_scores=scores,
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethode: schnelle MOS-Schätzung via SNR-Proxy
    # ------------------------------------------------------------------

    def _quick_mos(self, seg: np.ndarray, sr: int) -> float:
        """Schnelle DSP-basierte MOS-Schätzung (kein ML); gibt float in [1,5] zurueck."""
        if len(seg) == 0:
            return 4.0
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < 1e-8:
            return 4.0  # Stille → neutral
        # Heuristik: SNR-Proxy aus Signal/Rauschboden
        # Rauschboden = 5. Perzentil der Frame-Energie
        frame = 1024
        if len(seg) < frame:
            return 4.0
        n_frames = len(seg) // frame
        frame_rms = np.array([np.sqrt(np.mean(seg[k * frame : (k + 1) * frame] ** 2)) for k in range(n_frames)])
        noise_floor = float(np.percentile(frame_rms, 5)) + 1e-10
        signal_level = float(np.percentile(frame_rms, 95))
        # Guard: signal_level muss > noise_floor sein (Dirac oder Stille-Artefakt)
        if signal_level <= noise_floor or not math.isfinite(signal_level / noise_floor):
            return 4.0
        snr_db = 20.0 * math.log10(signal_level / noise_floor)
        # Mapping SNR → MOS-Schätzung
        mos = 1.0 + 4.0 * self._sigmoid((snr_db - 15.0) / 10.0)
        return float(np.clip(mos, 1.0, 5.0))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[TemporalQualityCoherenceMetric] = None
_lock = threading.Lock()


def get_temporal_coherence_metric() -> TemporalQualityCoherenceMetric:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TemporalQualityCoherenceMetric()
    return _instance


def measure_temporal_coherence(audio: np.ndarray, sr: int) -> TemporalCoherenceResult:
    """Convenience-Wrapper."""
    return get_temporal_coherence_metric().measure(audio, sr)


# Convenience-Alias (Spec §3.2)
get_temporal_quality_coherence = get_temporal_coherence_metric

__all__ = [
    "TemporalQualityCoherenceMetric",
    "TemporalCoherenceResult",
    "get_temporal_coherence_metric",
    "get_temporal_quality_coherence",
    "measure_temporal_coherence",
    "TEMPORAL_CONSISTENCY_THRESHOLD",
    "SIGMA_THRESHOLD",
    "MIN_FILE_DURATION_S",
    "SEGMENT_DURATION_S",
    "SEGMENT_HOP_S",
]
