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

import logging
import math
import threading
from dataclasses import dataclass

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

        arousal_orig, valence_orig, _centroids_orig = self._compute_features(orig_mono, sr, seg_len, hop_len)
        arousal_rest, valence_rest, _centroids_rest = self._compute_features(rest_mono, sr, seg_len, hop_len)

        # §9.10.119: Re-normalize arousal centroid component per-song.
        # After denoising, the noise floor drops → centroid shifts upward
        # globally → false arousal increase that flattens apparent dynamics.
        # Fix: scale restored centroid median to match original median.
        if len(_centroids_orig) >= 3 and len(_centroids_rest) >= 3:
            _med_orig = float(np.median(_centroids_orig))
            _med_rest = float(np.median(_centroids_rest))
            if _med_rest > 1e-3 and abs(_med_rest - _med_orig) / max(_med_orig, 1.0) > 0.05:
                _centroid_correction = _med_orig / _med_rest
                # Re-compute arousal_rest with corrected centroid weight
                _arousal_corr = []
                for i, start in enumerate(range(0, len(rest_mono) - seg_len + 1, hop_len)):
                    if i >= len(arousal_rest):
                        break
                    seg = rest_mono[start : start + seg_len]
                    rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
                    _c_hz = _centroids_rest[i] * _centroid_correction if i < len(_centroids_rest) else 0.0
                    _c_norm = float(np.clip(_c_hz / max(sr / 2.0, 1.0), 0.0, 1.0))
                    _arousal_corr.append(rms * 0.55 + _c_norm * 0.45)
                if _arousal_corr:
                    arousal_rest = np.array(_arousal_corr[: len(arousal_rest)], dtype=np.float32)

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
        # §Y4 Local segment Δarousal check: no segment should drop > 8%
        # Ignore the last ≤3 s tail segment (under-filled HOP window)
        # ----------------------------------------------------------------
        _n_tail_segs = max(0, int(3.0 / self.HOP_S))  # segments covering last 3 s
        _check_segs = n_segs - _n_tail_segs
        _local_drop_ok = True
        _worst_local_drop = 0.0
        _worst_seg_idx = -1
        if _check_segs > 0:
            _orig_clip = np.maximum(arousal_orig[:_check_segs], 1e-9)
            _delta_local = arousal_rest[:_check_segs] / _orig_clip - 1.0
            _worst_local_drop = float(np.min(_delta_local))
            _worst_seg_idx = int(np.argmin(_delta_local))
            if _worst_local_drop < -0.08:
                _local_drop_ok = False

        # ----------------------------------------------------------------
        # Urteil
        # ----------------------------------------------------------------
        arc_preserved = (
            ar_pearson >= self.THRESHOLD_AROUSAL
            and val_pearson >= self.THRESHOLD_VALENCE
            and klimax_dev <= self.MAX_KLIMAX_DEVIATION
            and klimax_level_dev <= self.MAX_KLIMAX_LEVEL_DB
            and _local_drop_ok  # §Y4: no segment arousal collapsed by > 8%
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
        if not _local_drop_ok:
            reason_parts.append(
                f"Lokaler Arousal-Einbruch Segment {_worst_seg_idx + 1}/{_check_segs}: "
                f"Δ={_worst_local_drop:.3f} (Schwelle: -0.08)"
            )

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
        _centroid_hz_list: list[float] = []  # §9.10.119: collect for post-normalization

        positions = list(range(0, len(mono) - seg_len + 1, hop_len))
        for start in positions:
            seg = mono[start : start + seg_len]
            if not np.isfinite(seg).all():
                continue

            # Arousal: 0.55·RMS + 0.45·spectral_centroid_norm (v9.10.114)
            # §9.10.119: Centroid normalized relative to per-song median instead
            # of Nyquist.  After denoising, the noise floor drops → centroid
            # shifts upward → false arousal increase → flattened dynamic arc.
            # Per-song normalization makes the proxy robust to global spectral
            # shifts caused by NR (Blanchini 2011; empirical correction).
            rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
            _n_fft_a = min(1024, len(seg))
            _spec_a = np.abs(np.fft.rfft(seg[:_n_fft_a], n=_n_fft_a)) + 1e-9
            _freqs_a = np.fft.rfftfreq(_n_fft_a, 1.0 / sr)
            _spec_sum = float(np.sum(_spec_a))
            _centroid_hz = float(np.sum(_freqs_a * _spec_a) / max(_spec_sum, 1e-12))
            _centroid_hz_list.append(_centroid_hz)
            # Normalize centroid to [0,1] by Nyquist (sr/2) — deferred below
            _centroid_norm = float(np.clip(_centroid_hz / max(sr / 2.0, 1.0), 0.0, 1.0))
            arousal_list.append(rms * 0.55 + _centroid_norm * 0.45)

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
            _centroid_hz_list,
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

    def correct_arc(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sr: int,
        max_gain_db: float = 6.0,
        damping: float = 0.7,
    ) -> tuple[np.ndarray, EmotionalArcResult]:
        """Correct emotional arc via macro-level RMS-based gain envelope.

        Operates at 5 s segment timescale, complementing MDEM's 400 ms
        micro-dynamics.  Only the RMS component (0.6 weight in arousal
        proxy) responds to gain correction; ZCR (0.4 weight) is spectral.

        Algorithm (Thayer 1989 / Kim & André 2008):
            1. Per-segment RMS for original + restored (5 s, hop 2.5 s)
            2. gain_db = 20·log₁₀(rms_orig / rms_rest) · damping, ±max_gain_db
            3. Savitzky-Golay smooth → linear interpolation to sample-level
            4. Apply per-channel, NaN-guard, clip ±1.0
            5. Re-measure; revert if arousal worsened

        Args:
            original: Pre-repair reference audio (float32, 1D/2D)
            restored: Post-MDEM restored audio (float32, 1D/2D)
            sr:       48000
            max_gain_db: Max per-segment gain (default 6 dB)
            damping:  Correction fraction 0–1 (default 0.7 = 70 % correction)

        Returns:
            (corrected_audio, EmotionalArcResult after correction)
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        def _to_mono(a: np.ndarray) -> np.ndarray:
            arr = np.asarray(a, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return np.mean(arr, axis=0) if arr.ndim == 2 else arr

        orig_mono = _to_mono(original)
        rest_mono = _to_mono(restored)

        n = min(len(orig_mono), len(rest_mono))
        orig_mono = orig_mono[:n]
        rest_mono = rest_mono[:n]

        # Too short for meaningful arc correction
        if n / sr < self.MIN_DURATION_S:
            return restored.copy(), self.measure(original, restored, sr)

        seg_len = int(self.SEGMENT_S * sr)
        hop_len = int(self.HOP_S * sr)
        positions = list(range(0, n - seg_len + 1, hop_len))

        if len(positions) < 3:
            return restored.copy(), self.measure(original, restored, sr)

        # ---- Per-segment RMS ----
        rms_orig = np.empty(len(positions), dtype=np.float32)
        rms_rest = np.empty(len(positions), dtype=np.float32)
        for i, start in enumerate(positions):
            end = start + seg_len
            rms_orig[i] = float(np.sqrt(np.mean(orig_mono[start:end] ** 2) + 1e-12))
            rms_rest[i] = float(np.sqrt(np.mean(rest_mono[start:end] ** 2) + 1e-12))

        # ---- Gain in dB, damped ----
        eps = 1e-9
        gain_db = 20.0 * np.log10((rms_orig + eps) / (rms_rest + eps))
        gain_db = np.clip(gain_db.astype(np.float64) * damping, -max_gain_db, max_gain_db)

        # Guard: suppress positive gain on near-silent restored segments.
        # After denoising the tail is genuinely silent; applying a positive
        # gain here (to match original tape/vinyl noise floor) inflates the
        # integrated LUFS measurement and forces phase_40 to attenuate the
        # musical content in compensation (regression).
        _silence_rms_thresh = 10.0 ** (-60.0 / 20.0)  # −60 dBFS ≈ noise floor
        # §2.30 Bug-Fix §13 — Fade-out denoising guard (companion to MDEM guard):
        # Restored segment is moderately quiet (< -42 dBFS) AND original is
        # significantly louder (> 6 dB difference). This is the "denoised fade-out"
        # pattern: original had noise+signal, restored has only the clean signal.
        # Applying positive gain would reconstruct the noise floor and cause an
        # audible volume jump at the end of the song.
        _quiet_rms_thresh = 10.0 ** (-42.0 / 20.0)  # −42 dBFS = quiet/fade-out zone
        _noise_diff_thresh_db = 6.0
        for _i in range(len(gain_db)):
            if gain_db[_i] > 0.0:
                if rms_rest[_i] < _silence_rms_thresh:
                    gain_db[_i] = 0.0
                elif rms_rest[_i] < _quiet_rms_thresh:
                    _diff = 20.0 * np.log10((rms_orig[_i] + eps) / (rms_rest[_i] + eps))
                    if _diff >= _noise_diff_thresh_db:
                        gain_db[_i] = 0.0  # Denoised fade-out: no boost

        # Savitzky-Golay smooth (boxcar fallback)
        if len(gain_db) >= 7:
            try:
                from scipy.signal import savgol_filter

                gain_db = savgol_filter(gain_db, window_length=7, polyorder=2)
            except Exception:
                kernel = np.ones(5, dtype=np.float64) / 5.0
                gain_db = np.convolve(gain_db, kernel, mode="same")

        gain_db = np.clip(gain_db, -max_gain_db, max_gain_db).astype(np.float32)

        # ---- Interpolate to sample-level via segment centres ----
        centres = np.array([start + seg_len // 2 for start in positions], dtype=np.float64)
        sample_idx = np.arange(n, dtype=np.float64)
        gain_db_interp = np.interp(sample_idx, centres, gain_db).astype(np.float32)
        gain_linear = np.float32(10.0) ** (gain_db_interp / np.float32(20.0))

        # ---- Apply gain ----
        out = np.asarray(restored, dtype=np.float32).copy()
        if out.ndim == 2:
            for ch in range(out.shape[0]):
                ch_n = min(out.shape[1], len(gain_linear))
                out[ch, :ch_n] *= gain_linear[:ch_n]
        else:
            ch_n = min(len(out), len(gain_linear))
            out[:ch_n] *= gain_linear[:ch_n]

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        # ---- Re-measure; safety-revert if correction worsened arousal ----
        arc_before = self.measure(original, restored, sr)
        arc_after = self.measure(original, out, sr)

        if not arc_before.skipped:
            if arc_after.arousal_pearson < arc_before.arousal_pearson - 0.02:
                logger.warning(
                    "EmotionalArc correction worsened arousal (%.3f → %.3f) — reverting",
                    arc_before.arousal_pearson,
                    arc_after.arousal_pearson,
                )
                return restored.copy(), arc_before

        logger.info(
            "EmotionalArc correction applied: arousal %.3f→%.3f  valence %.3f→%.3f  max_gain=±%.1f dB",
            arc_before.arousal_pearson,
            arc_after.arousal_pearson,
            arc_before.valence_pearson,
            arc_after.valence_pearson,
            float(np.max(np.abs(gain_db))),
        )
        return out, arc_after


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: EmotionalArcPreservationMetric | None = None
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


def correct_emotional_arc(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int,
    max_gain_db: float = 6.0,
    damping: float = 0.7,
) -> tuple[np.ndarray, EmotionalArcResult]:
    """Convenience wrapper: correct emotional arc macro-dynamics.

    Args:
        original: Pre-repair reference audio, float32, SR=48000
        restored: Post-MDEM restored audio, float32, SR=48000
        sr:       48000 (mandatory)
        max_gain_db: Max per-segment gain (default 6 dB)
        damping:  Correction fraction 0–1 (default 0.7)

    Returns:
        (corrected_audio, EmotionalArcResult)
    """
    return get_emotional_arc_metric().correct_arc(original, restored, sr, max_gain_db=max_gain_db, damping=damping)
