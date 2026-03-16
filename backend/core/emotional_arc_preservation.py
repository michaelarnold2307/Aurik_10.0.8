"""
core/emotional_arc_preservation.py — Aurik 9.9+ (§2.30 / §8.2 Punkt 12)

EmotionalArcPreservationMetric: Prüft, ob der emotionale Dynamik-Bogen
(Arousal-/Valence-Kurve) des Originals im restaurierten Signal erhalten bleibt.

Musikalische Werke haben einen emotionalen Spannungsbogen (sanfter Beginn →
Klimax → Auflösung). Restaurierung darf diesen Bogen nicht begradigen.

Invariante: Dateien < 30 s: Metrik deaktiviert. Pearson-Schwellen ≥ 0.85 (Arousal),
            ≥ 0.80 (Valence). Klimax-Peak-Abweichung ≤ 2 Segmente.

Referenz:
    Russell (1980): „A circumplex model of affect"
    Thayer (1989): „The biopsychology of mood and arousal"
    Kim & André (2008): „Emotion recognition based on physiological changes in music listening"
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class EmotionalArcResult:
    """Ergebnis der Emotionaler-Dynamik-Bogen-Analyse (§8.2 Punkt 12)."""

    arousal_pearson: float  # Korrelation Arousal-Profil ∈ [−1, 1]
    valence_pearson: float  # Korrelation Valence-Profil ∈ [−1, 1]
    klimax_peak_deviation: float  # Segmente | argmax(orig) − argmax(rest) |
    klimax_level_deviation_db: float  # dB-Abweichung Klimax-Peak
    arc_preserved: bool  # True wenn alle Schwellen erfüllt
    reason: str = ""  # Menschenlesbare Begründung (Deutsch)
    skipped: bool = False  # True wenn Datei < 30 s

    THRESHOLD_AROUSAL = 0.85
    THRESHOLD_VALENCE = 0.80
    MAX_KLIMAX_DEVIATION_SEGMENTS = 2
    MAX_KLIMAX_LEVEL_DB = 2.0

    def as_dict(self) -> dict:
        return {
            "arousal_pearson": self.arousal_pearson,
            "valence_pearson": self.valence_pearson,
            "klimax_peak_deviation": self.klimax_peak_deviation,
            "klimax_level_deviation_db": self.klimax_level_deviation_db,
            "arc_preserved": self.arc_preserved,
            "reason": self.reason,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class EmotionalArcPreservationMetric:
    """Misst Erhalt des emotionalen Dynamik-Bogens Original vs. Restauriert.

    Algorithmus (§8.2):
        1. Audio in 5-s-Segmente teilen (Hop: 2.5 s)
        2. Arousal-Proxy pro Segment:
           arousal(t) = 0.6 · rms_norm(t) + 0.4 · zcr_norm(t)
        3. Valence-Proxy: Harmonizitäts-Ratio (HPSS-Approximation via
           Spektral-Flachheit — hohe Spektral-Flachheit = weniger Harmonizität)
        4. Pearson-Korrelation arousal_orig ↔ arousal_rest ≥ 0.85
        5. Klimax-Peak-Erhalt:
           |argmax(arousal_orig) − argmax(arousal_rest)| ≤ 2 Segmente
           |max(arousal_orig) − max(arousal_rest)| ≤ 2 dB

    Invarianten:
        - Dateien < 30 s: Metrik deaktiviert (skipped=True, arc_preserved=True)
        - NaN im Segment → Segment überspringen
        - Alle Ausgaben sind NaN/Inf-frei
    """

    SEGMENT_S = 5.0  # Segment-Länge
    HOP_S = 2.5  # Hop-Größe
    MIN_DURATION_S = 30.0  # Minimum für sinnvolle Bogen-Messung

    THRESHOLD_AROUSAL = 0.85
    THRESHOLD_VALENCE = 0.80
    MAX_KLIMAX_DEVIATION = 2  # Segmente
    MAX_KLIMAX_LEVEL_DB = 2.0  # dB

    def measure(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
    ) -> EmotionalArcResult:
        """Misst den Erhalt des emotionalen Bogens.

        Args:
            original: Original-Audio (vor Restaurierung), float32, 1D oder 2D
            restored: Restauriertes Audio, float32, 1D oder 2D
            sr:       Sample-Rate, muss 48000 sein

        Returns:
            EmotionalArcResult mit Pearson-Korrelationen und Klimax-Analyse.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        # Mono-Konvertierung
        def to_mono(a: np.ndarray) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=0)
            return arr

        orig_mono = to_mono(original)
        rest_mono = to_mono(restored)

        # Längenabgleich
        n = min(len(orig_mono), len(rest_mono))
        orig_mono = orig_mono[:n]
        rest_mono = rest_mono[:n]

        duration_s = n / sr

        # Zu kurz → überspringen
        if duration_s < self.MIN_DURATION_S:
            return EmotionalArcResult(
                arousal_pearson=1.0,
                valence_pearson=1.0,
                klimax_peak_deviation=0.0,
                klimax_level_deviation_db=0.0,
                arc_preserved=True,
                reason="Datei kürzer als 30 s — Emotional-Arc-Prüfung nicht aktiv.",
                skipped=True,
            )

        # ----------------------------------------------------------------
        # Segment-Analyse
        # ----------------------------------------------------------------
        seg_len = int(self.SEGMENT_S * sr)
        hop_len = int(self.HOP_S * sr)

        arousal_orig, valence_orig = self._compute_features(orig_mono, sr, seg_len, hop_len)
        arousal_rest, valence_rest = self._compute_features(rest_mono, sr, seg_len, hop_len)

        n_segs = min(len(arousal_orig), len(arousal_rest))
        if n_segs < 3:
            return EmotionalArcResult(
                arousal_pearson=1.0,
                valence_pearson=1.0,
                klimax_peak_deviation=0.0,
                klimax_level_deviation_db=0.0,
                arc_preserved=True,
                reason="Zu wenige Segmente für Bogen-Messung.",
                skipped=True,
            )

        arousal_orig = arousal_orig[:n_segs]
        arousal_rest = arousal_rest[:n_segs]
        valence_orig = valence_orig[:n_segs]
        valence_rest = valence_rest[:n_segs]

        # ----------------------------------------------------------------
        # Pearson-Korrelationen
        # ----------------------------------------------------------------
        ar_pearson = self._pearson(arousal_orig, arousal_rest)
        val_pearson = self._pearson(valence_orig, valence_rest)

        # ----------------------------------------------------------------
        # Klimax-Peak-Analyse
        # ----------------------------------------------------------------
        peak_orig = int(np.argmax(arousal_orig))
        peak_rest = int(np.argmax(arousal_rest))
        klimax_dev = abs(peak_orig - peak_rest)

        orig_peak_db = float(20.0 * math.log10(max(arousal_orig[peak_orig], 1e-9)))
        rest_peak_db = float(20.0 * math.log10(max(arousal_rest[peak_rest], 1e-9)))
        klimax_level_dev = abs(orig_peak_db - rest_peak_db)

        # ----------------------------------------------------------------
        # Urteil
        # ----------------------------------------------------------------
        arc_preserved = (
            ar_pearson >= self.THRESHOLD_AROUSAL
            and val_pearson >= self.THRESHOLD_VALENCE
            and klimax_dev <= self.MAX_KLIMAX_DEVIATION
            and klimax_level_dev <= self.MAX_KLIMAX_LEVEL_DB
        )

        reason_parts = []
        if ar_pearson < self.THRESHOLD_AROUSAL:
            reason_parts.append(f"Arousal-Korrelation zu niedrig ({ar_pearson:.2f} < {self.THRESHOLD_AROUSAL})")
        if val_pearson < self.THRESHOLD_VALENCE:
            reason_parts.append(f"Valence-Korrelation zu niedrig ({val_pearson:.2f} < {self.THRESHOLD_VALENCE})")
        if klimax_dev > self.MAX_KLIMAX_DEVIATION:
            reason_parts.append(f"Klimax-Verschiebung: {klimax_dev} Segmente (Max: {self.MAX_KLIMAX_DEVIATION})")
        if klimax_level_dev > self.MAX_KLIMAX_LEVEL_DB:
            reason_parts.append(f"Klimax-Pegel-Abweichung: {klimax_level_dev:.1f} dB")

        reason = (
            "Emotionaler Bogen vollständig erhalten."
            if arc_preserved
            else "Emotionaler Bogen teilweise verändert: " + "; ".join(reason_parts)
        )

        return EmotionalArcResult(
            arousal_pearson=round(ar_pearson, 4),
            valence_pearson=round(val_pearson, 4),
            klimax_peak_deviation=float(klimax_dev),
            klimax_level_deviation_db=round(klimax_level_dev, 2),
            arc_preserved=arc_preserved,
            reason=reason,
            skipped=False,
        )

    # ----------------------------------------------------------------
    # Hilfsmethoden
    # ----------------------------------------------------------------

    def _compute_features(
        self,
        mono: np.ndarray,
        sr: int,
        seg_len: int,
        hop_len: int,
    ):
        """Berechnet Arousal- und Valence-Proxy-Profile als Arrays."""
        arousal_list = []
        valence_list = []

        positions = list(range(0, len(mono) - seg_len + 1, hop_len))
        for start in positions:
            seg = mono[start : start + seg_len]
            if not np.isfinite(seg).all():
                continue

            # Arousal: 0.6 · RMS_norm + 0.4 · ZCR_norm
            rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            zcr = float(np.mean(np.abs(np.diff(np.sign(seg + 1e-12))) / 2.0))
            arousal_list.append(rms * 0.6 + zcr * 0.4)

            # Valence: Harmonizitäts-Proxy via Spektral-Flachheit-Inverse
            # (hohe Spektral-Flachheit = viel Rauschen = geringe Harmonizität)
            try:
                n_fft = min(2048, len(seg))
                spec = np.abs(np.fft.rfft(seg[:n_fft], n=n_fft)) + 1e-9
                geo_mean = np.exp(np.mean(np.log(spec + 1e-9)))
                arith_mean = np.mean(spec)
                flatness = float(geo_mean / (arith_mean + 1e-12))
                harmonicity = 1.0 - float(np.clip(flatness, 0.0, 1.0))
                valence_list.append(harmonicity)
            except Exception:
                valence_list.append(0.5)

        return (
            np.array(arousal_list, dtype=np.float32),
            np.array(valence_list, dtype=np.float32),
        )

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson-Korrelation, NaN-sicher."""
        try:
            if len(a) < 2 or len(b) < 2:
                return 1.0
            std_a = np.std(a)
            std_b = np.std(b)
            if std_a < 1e-9 or std_b < 1e-9:
                return 1.0
            corr = float(np.corrcoef(a, b)[0, 1])
            if not math.isfinite(corr):
                return 0.0
            return float(np.clip(corr, -1.0, 1.0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: Optional[EmotionalArcPreservationMetric] = None
_lock = threading.Lock()


def get_emotional_arc_metric() -> EmotionalArcPreservationMetric:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EmotionalArcPreservationMetric()
    return _instance


def measure_emotional_arc(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
) -> EmotionalArcResult:
    """Convenience-Wrapper: Erhalt des emotionalen Bogens prüfen.

    Args:
        original: Original-Audio vor Restaurierung, float32, SR = 48000
        restored: Restauriertes Audio, float32, SR = 48000
        sr:       48000 (Pflicht)

    Returns:
        EmotionalArcResult mit Pearson-Korrelationen und Klimax-Analyse.
    """
    return get_emotional_arc_metric().measure(original, restored, sr)
