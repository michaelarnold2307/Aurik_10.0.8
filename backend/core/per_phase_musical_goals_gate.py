"""
PerPhaseMusicalGoalsGate (PMGG) — Aurik 9.0 §2.29
===================================================

Prüft Musical Goals nach JEDER Phase via 5-s-Stichprobe.
Verhindert kumulative Degradation über 56 Phasen.

PROBLEM:
--------
Jede Phase kann Musical Goals minimal verslechtern (z.B. Δ-0.01).
Über 20+ aktive Phasen kumuliert das zu -0.20 → ein Ziel fällt unter
den Pflicht-Schwellwert. Der End-Check kann das nicht mehr korrigieren.

ALGORITHMUS:
-----------
Pro Phase (wrap_phase()):
    1. 5-s-Stichprobe aus Mitte des Audios
    2. Phase ausführen: audio_after = phase(audio_before)
    3. Schnell-Check (6 Ziele, ≤ 200 ms):
       Brillanz, Wärme, Groove, TonalCenter,
       Natürlichkeit (MFCC-Proxy), Timbre-Authentizität
    4. Δ = score_after − score_before für jedes Ziel
       Falls Δ < −REGRESSION_THRESHOLD (adaptiv je nach Restorability):
         Retry-1: Phase mit strength × 0.65
         Retry-2: Phase mit strength × 0.50  (v9.15-B3: sanfterer Gradient)
         Retry-3: Phase mit strength × 0.35
         Retry-4: Phase mit strength × 0.20
         Retry-5 (Last-Resort): Phase mit strength × 0.10
         Falls immer noch: Rollback + phase_gate_log-Eintrag

KONSTANTEN:
-----------
REGRESSION_THRESHOLD = 0.025  (adaptiv: 0.012 / 0.040 / 0.060 je Restorability)
SAMPLE_DURATION_S    = 5.0
MAX_RETRIES          = 5  (v9.15-B3: 5 Retries mit sanftem Stärkegradienten)

OVERHEAD: max. 56 × 200 ms = 11.2 s pro Verarbeitungsdurchlauf
DEAKTIVIERUNG: --no-phase-gate (Debugging/Benchmarking)

WICHTIG: MERT wird im Schnell-Check NICHT verwendet (zu langsam: 800 ms)
Vollständige 14-Ziele-Prüfung bleibt am Pipeline-Ende (MusicalGoalsChecker)

Autor: Aurik 9.0 Development Team / v9.15
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konstanten (§2.29) — restorability-adaptive Schwellwerte
# ---------------------------------------------------------------------------
# Feste Einzel-Schwelle (Legacy-Fallback, nicht mehr primär verwendet)
REGRESSION_THRESHOLD: float = 0.025

# Restorability-adaptive Schwellwerte (§2.29 Spec)
REGRESSION_THRESHOLD_GOOD: float = 0.012  # v9.11: verschärft von 0.018 — Exzellenz-Präzision | restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = 0.040  # restorability 40–69 (entspannter)
REGRESSION_THRESHOLD_POOR: float = 0.060  # restorability < 40 (maximal tolerant)

SAMPLE_DURATION_S: float = 5.0
MAX_RETRIES: int = 5  # v9.15-B3: 5 Retries mit sanftem Stärkegradienten (0.65→0.50→0.35→0.20→0.10)

# ---------------------------------------------------------------------------
# §9.7.3 Phasen-adaptive Sample-Dauer — triviale Phasen brauchen < 5 s
# ---------------------------------------------------------------------------
PHASE_SAMPLE_DURATIONS: dict[str, float] = {
    # Triviale Phasen: Zeiteffekt ist lokal messbar in 1–2 s
    "phase_30": 1.5,  # DC-Offset-Removal
    "phase_05": 1.5,  # Rumble-Filter (< 20 Hz)
    "phase_02": 2.0,  # Hum-Removal (50/60 Hz Kammfilter)
    "phase_15": 1.5,  # Stereo-Balance L/R
    "phase_11": 1.5,  # Limiting (True-Peak)
    "phase_18": 2.0,  # Noise-Gate
    # Standard: SAMPLE_DURATION_S = 5.0 für alle anderen Phasen
}


def _get_sample_duration(phase_id: str) -> float:
    """Gibt phasen-adaptive Stichprobenlänge zurück (§9.7.3).

    Minimale Sample-Dauer: 1.0 s (kein Unterschreiten).
    Maximale Sample-Dauer: SAMPLE_DURATION_S (5.0 s).
    Phase-ID-Matching via startswith — robust gegen Suffix-Varianten.
    """
    for prefix, dur in PHASE_SAMPLE_DURATIONS.items():
        if phase_id.startswith(prefix):
            return max(1.0, min(dur, SAMPLE_DURATION_S))
    return SAMPLE_DURATION_S


# Strength-Faktoren für Retry-Durchgänge
_RETRY_STRENGTHS: list[float] = [
    0.65,
    0.50,
    0.35,
    0.20,
    0.10,
]  # v9.15-B3: sanfterer Gradient, 0.50 als 2. Stufe ergänzt


def _get_adaptive_threshold(restorability_score: float) -> float:
    """Restorability-adaptiver REGRESSION_THRESHOLD (§2.29).

    Args:
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100]

    Returns:
        Adaptiver Schwellwert: 0.025 / 0.040 / 0.060.
    """
    if restorability_score >= 70.0:
        return REGRESSION_THRESHOLD_GOOD
    if restorability_score >= 40.0:
        return REGRESSION_THRESHOLD_FAIR
    return REGRESSION_THRESHOLD_POOR


# Schnell-Subset der 14 Musical Goals (ohne MERT-abhängige Ziele)
FAST_GOALS_SUBSET: list[str] = [
    "brillanz",
    "waerme",
    "groove",
    "tonal_center",
    "natuerlichkeit_mfcc_proxy",
    "timbre_authentizitaet",
]


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PhaseGateLogEntry:
    """Eintrag im phase_gate_log für eine Phase."""

    phase_id: str
    action: str  # "passed" | "retry1" | "retry2" | "rollback"
    goal_regressions: Dict[str, float]  # Ziel → Δ-Score
    strength_used: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PhaseGateResult:
    """Ergebnis der wrap_phase()-Operation."""

    audio: np.ndarray
    scores_after: Dict[str, float]
    log_entry: PhaseGateLogEntry
    rolled_back: bool


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: Optional[PerPhaseMusicalGoalsGate] = None
_lock = threading.Lock()


def get_phase_gate() -> "PerPhaseMusicalGoalsGate":
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerPhaseMusicalGoalsGate()
    return _instance


# ---------------------------------------------------------------------------
# Schnell-Metriken (ohne MERT, ohne CDPAM, ohne externe ML-Modelle)
# ---------------------------------------------------------------------------


def _measure_quick(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Misst 6 Musical Goals auf einer 5-s-Stichprobe in ≤ 200 ms.

    Alle Messungen sind DSP-only (kein MERT, kein CREPE).
    NaN-sicher: fehlerhafte Einzelmessungen werden auf 0.5 (neutral) gesetzt.

    Args:
        audio: Mono oder Stereo, float32, beliebige Länge
        sr: 48000 Hz

    Returns:
        Dict mit 6 Scores ∈ [0, 1]
    """
    mono = audio[:, 0] if audio.ndim == 2 else audio
    mono = np.nan_to_num(mono, nan=0.0).astype(np.float32)

    scores: Dict[str, float] = {}

    # ── Brillanz (HF-Energie > 8 kHz) ─────────────────────────────────
    try:
        fft_mag = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
        hf_energy = float(np.mean(fft_mag[freqs > 8000] ** 2))
        tot_energy = float(np.mean(fft_mag**2)) + 1e-12
        scores["brillanz"] = float(np.clip(hf_energy / tot_energy / 0.3 + 0.4, 0.0, 1.0))
    except Exception:
        scores["brillanz"] = 0.5

    # ── Wärme (Mid-Range-Energie 200–2000 Hz) ──────────────────────────
    try:
        mid_energy = float(np.mean(fft_mag[(freqs >= 200) & (freqs <= 2000)] ** 2))
        scores["waerme"] = float(np.clip(mid_energy / tot_energy / 0.6 + 0.3, 0.0, 1.0))
    except Exception:
        scores["waerme"] = 0.5

    # ── Groove (Onset-Energie-Regularität via Autokorrelation) ─────────
    try:
        env = np.abs(mono)
        # Hüllkurven-Autokorrelation
        hop = sr // 100  # 10 ms
        rms_env = np.array([float(np.mean(env[i : i + hop] ** 2)) for i in range(0, len(env) - hop, hop)])
        if len(rms_env) > 10:
            autocorr = np.correlate(rms_env, rms_env, mode="full")
            autocorr = autocorr[len(rms_env) - 1 :]
            autocorr /= autocorr[0] + 1e-12
            # Regularität: Autokorrelations-Peak bei ~0.5 s (typisch Groove)
            lag_05 = min(50, len(autocorr) - 1)  # 50 × 10 ms = 500 ms
            scores["groove"] = float(np.clip(autocorr[lag_05] * 0.5 + 0.5, 0.0, 1.0))
        else:
            scores["groove"] = 0.5
    except Exception:
        scores["groove"] = 0.5

    # ── Tonales Zentrum (Chroma-Konzentration) ─────────────────────────
    try:
        n_fft_chroma = 4096
        spec_mag = np.abs(np.fft.rfft(mono, n=n_fft_chroma))
        spec_freqs = np.fft.rfftfreq(n_fft_chroma, d=1.0 / sr)
        # Chroma-Bins approximieren
        chroma = np.zeros(12, dtype=np.float32)
        for b, f in enumerate(spec_freqs):
            if 27.5 < f < 4186:
                note = int(round(12.0 * math.log2(f / 440.0 + 1e-12))) % 12
                chroma[note] += spec_mag[b]
        if chroma.sum() > 1e-8:
            chroma /= chroma.sum()
            # Konzentration = 1 − Entropie/log(12)
            entropy = -float(np.sum(chroma * np.log(chroma + 1e-12)))
            tonal_score = 1.0 - entropy / math.log(12.0)
            scores["tonal_center"] = float(np.clip(tonal_score, 0.0, 1.0))
        else:
            scores["tonal_center"] = 0.5
    except Exception:
        scores["tonal_center"] = 0.5

    # ── Natürlichkeit (MFCC-Proxy: spektrale Glattheit) ───────────────
    try:
        n_mfcc = min(20, len(fft_mag) // 2)
        mfcc_approx = np.log(np.convolve(fft_mag[: len(fft_mag) // 2], np.ones(10) / 10, mode="valid") + 1e-12)
        if len(mfcc_approx) > n_mfcc:
            smoothness = 1.0 - float(np.std(np.diff(mfcc_approx[:n_mfcc]))) / (
                float(np.std(mfcc_approx[:n_mfcc])) + 1e-12
            )
            scores["natuerlichkeit_mfcc_proxy"] = float(np.clip(smoothness, 0.0, 1.0))
        else:
            scores["natuerlichkeit_mfcc_proxy"] = 0.5
    except Exception:
        scores["natuerlichkeit_mfcc_proxy"] = 0.5

    # ── Timbre-Authentizität (MFCC-basiert: Pearson auf log-Mel) ──────
    try:
        # Proxy: Spectral Centroid-Stabilität über kurze Fenster
        hop_t = sr // 50  # 20 ms
        centroids = []
        for i in range(0, len(mono) - hop_t, hop_t):
            w = mono[i : i + hop_t]
            w_fft = np.abs(np.fft.rfft(w))
            w_freqs = np.fft.rfftfreq(len(w), d=1.0 / sr)
            centroid = float(np.sum(w_freqs * w_fft) / (np.sum(w_fft) + 1e-12))
            centroids.append(centroid)
        if len(centroids) > 2:
            cv = float(np.std(centroids)) / (float(np.mean(centroids)) + 1e-12)
            # Niedrige CV → stabiles Timbre → hoher Score
            scores["timbre_authentizitaet"] = float(np.clip(1.0 - min(cv, 1.0), 0.0, 1.0))
        else:
            scores["timbre_authentizitaet"] = 0.5
    except Exception:
        scores["timbre_authentizitaet"] = 0.5

    # NaN-Schutz (§3.1)
    for k in FAST_GOALS_SUBSET:
        if k not in scores or not math.isfinite(scores[k]):
            scores[k] = 0.5

    return scores


def _extract_sample(audio: np.ndarray, sr: int, duration_s: float = SAMPLE_DURATION_S) -> np.ndarray:
    """Extrahiert repräsentative 5-s-Stichprobe aus der Mitte des Audios."""
    n = len(audio) if audio.ndim == 1 else len(audio)
    sample_len = min(int(duration_s * sr), n)
    if n <= sample_len:
        return audio
    start = (n - sample_len) // 2
    return audio[start : start + sample_len] if audio.ndim == 1 else audio[start : start + sample_len]


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PerPhaseMusicalGoalsGate:
    """
    Wraps PhaseInterface.process() mit Musical-Goals-Prüfung.

    Alle Methoden sind thread-sicher und NaN/Inf-sicher.
    """

    def __init__(self) -> None:
        self._rollback_count: int = 0  # Pro Restaurierungsaufruf
        self._user_warned: bool = False  # Nutzer-Warnung einmalig

    def reset(self) -> None:
        """Setzt Zähler für neuen Restaurierungsaufruf zurück."""
        self._rollback_count = 0
        self._user_warned = False

    def wrap_phase(
        self,
        phase: Any,  # PhaseInterface-Instanz
        audio: np.ndarray,
        sr: int,
        scores_before: Optional[Dict[str, float]] = None,
        phase_kwargs: Optional[Dict[str, Any]] = None,
        restorability_score: float = 70.0,
        applicable_goals: Optional[Set[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], PhaseGateLogEntry]:
        """
        Führt eine Phase aus und prüft Musical-Goals-Regression.

        Args:
            phase: PhaseInterface-Instanz mit process(audio) → PhaseResult
            audio: Input-Audio (float32)
            sr: 48000 Hz
            scores_before: Bekannte Scores vor der Phase (werden gemessen
                           wenn nicht übergeben)
            phase_kwargs: Zusätzliche kwargs für den Phase-Aufruf (z.B. sample_rate, material_type)
            restorability_score: RestorabilityEstimator-Score ∈ [0, 100] — bestimmt
                                 adaptiven REGRESSION_THRESHOLD (§2.29).
            applicable_goals: Aus GoalApplicabilityFilter — nur diese Ziele werden
                              geprüft. None = alle FAST_GOALS_SUBSET-Ziele.

        Returns:
            (audio_out, scores_after, log_entry)
        """
        if sr != 48000:
            logger.debug(f"PMGG: SR={sr} (nicht 48000) — Goal-Messung läuft trotzdem")

        # Vor-Scores messen (wenn nicht übergeben)
        sample_before = _extract_sample(audio, sr)
        if scores_before is None:
            scores_before = _measure_quick(sample_before, sr)

        if phase_kwargs is None:
            phase_kwargs = {}

        phase_id = self._get_phase_id(phase)
        t0 = time.time()

        # Adaptiven Threshold bestimmen (§2.29)
        threshold = _get_adaptive_threshold(restorability_score)

        # §9.7.3 Phasen-adaptive Sample-Dauer
        _sample_dur = _get_sample_duration(phase_id)

        # Effective goal set: Schnitt aus FAST_GOALS_SUBSET + applicable_goals
        if applicable_goals is not None:
            effective_goals = [g for g in FAST_GOALS_SUBSET if g in applicable_goals]
            if not effective_goals:
                effective_goals = FAST_GOALS_SUBSET  # Fallback: alle
        else:
            effective_goals = FAST_GOALS_SUBSET

        # Phase ausführen + Regression prüfen
        audio_out, scores_after, action, strength = self._run_with_retry(
            phase,
            audio,
            sr,
            scores_before,
            phase_id,
            phase_kwargs,
            threshold=threshold,
            effective_goals=effective_goals,
            sample_duration_s=_sample_dur,
        )

        # Rollback-Zähler
        if action in ("rollback",):
            self._rollback_count += 1
            if self._rollback_count > 3 and not self._user_warned:
                self._user_warned = True
                logger.warning("ℹ️ Einige Verarbeitungsschritte wurden angepasst, " "um den Klang zu schützen.")

        goal_regressions = {
            g: scores_after.get(g, 0.5) - scores_before.get(g, 0.5)
            for g in effective_goals
            if scores_after.get(g, 0.5) - scores_before.get(g, 0.5) < -threshold
        }

        log_entry = PhaseGateLogEntry(
            phase_id=phase_id,
            action=action,
            goal_regressions=goal_regressions,
            strength_used=strength,
        )

        elapsed = time.time() - t0
        logger.debug(
            "PMGG: %s → %s (%.0f ms, strength=%.2f)",
            phase_id,
            action,
            elapsed * 1000,
            strength,
        )

        return audio_out, scores_after, log_entry

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _run_with_retry(
        self,
        phase: Any,
        audio: np.ndarray,
        sr: int,
        scores_before: Dict[str, float],
        phase_id: str,
        phase_kwargs: Optional[Dict[str, Any]] = None,
        *,
        threshold: float = REGRESSION_THRESHOLD_GOOD,
        effective_goals: Optional[list] = None,
        sample_duration_s: float = SAMPLE_DURATION_S,  # §9.7.3 phasen-adaptiv
    ) -> Tuple[np.ndarray, Dict[str, float], str, float]:
        """
        Führt Phase aus, ggf. mit Retry bei Regression.

        Args:
            threshold: Adaptiver REGRESSION_THRESHOLD (§2.29).
            effective_goals: Subset aus FAST_GOALS_SUBSET, das geprüft wird.
            sample_duration_s: Stichprobenlänge (§9.7.3 phasen-adaptiv, 1.0–5.0 s).

        Returns:
            (audio_out, scores_after, action_label, strength_used)
        """
        if phase_kwargs is None:
            phase_kwargs = {}
        if effective_goals is None:
            effective_goals = FAST_GOALS_SUBSET
        # Erster Versuch mit normaler Stärke (strength=1.0)
        audio_out = self._run_phase(phase, audio, 1.0, phase_kwargs)
        scores_after = _measure_quick(_extract_sample(audio_out, sr, duration_s=sample_duration_s), sr)

        regression = self._max_regression(scores_before, scores_after, effective_goals)
        if regression <= threshold:
            return audio_out, scores_after, "passed", 1.0

        # Retry-Schleife
        for attempt, strength in enumerate(_RETRY_STRENGTHS):
            action_label = f"retry{attempt + 1}"
            logger.debug(
                "PMGG: %s Retry %d mit strength=%.2f (Regression=%.4f, threshold=%.3f)",
                phase_id,
                attempt + 1,
                strength,
                regression,
                threshold,
            )
            audio_retry = self._run_phase(phase, audio, strength, phase_kwargs)
            scores_retry = _measure_quick(_extract_sample(audio_retry, sr, duration_s=sample_duration_s), sr)
            regression_retry = self._max_regression(scores_before, scores_retry, effective_goals)
            if regression_retry <= threshold:
                return audio_retry, scores_retry, action_label, strength

        # Rollback
        logger.warning(
            "⚠️ PMGG: %s übersprungen — Musical Goal Regression %.4f überschreitet Limit",
            phase_id,
            regression,
        )
        return audio, scores_before, "rollback", 0.0

    @staticmethod
    def _run_phase(
        phase: Any,
        audio: np.ndarray,
        strength: float,
        phase_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Führt Phase aus; gibt bei Fehler das Original zurück."""
        if phase_kwargs is None:
            phase_kwargs = {}
        try:
            result = phase(audio, strength=strength, **phase_kwargs)
            if hasattr(result, "audio"):
                out = result.audio
            elif hasattr(result, "processed_audio"):
                out = result.processed_audio
            elif isinstance(result, np.ndarray):
                out = result
            else:
                return audio

            if out is None or not isinstance(out, np.ndarray):
                return audio

            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            out = np.clip(out, -1.0, 1.0).astype(np.float32)

            # Länge sicherstellen
            if len(out) != len(audio):
                if len(out) > len(audio):
                    out = out[: len(audio)]
                else:
                    out = np.pad(out, (0, len(audio) - len(out)))

            return out
        except Exception as exc:
            logger.debug("PMGG: Phase-Ausführung fehlgeschlagen: %s", exc)
            return audio

    @staticmethod
    def _max_regression(
        before: Dict[str, float],
        after: Dict[str, float],
        goals: Optional[list] = None,
    ) -> float:
        """Maximale negative Differenz in Musical Goals (positiv = Regression)."""
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET
        max_reg = 0.0
        for g in check_goals:
            delta = after.get(g, 0.5) - before.get(g, 0.5)
            if delta < 0:
                max_reg = max(max_reg, -delta)
        return max_reg

    @staticmethod
    def _get_phase_id(phase: Any) -> str:
        """Extrahiert Phase-ID aus MetaDaten oder Klassennamen."""
        try:
            meta = phase.get_metadata()
            return getattr(meta, "phase_id", type(phase).__name__)
        except Exception:
            return type(phase).__name__


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------


def wrap_phase(
    phase: Any,
    audio: np.ndarray,
    sr: int,
    scores_before: Optional[Dict[str, float]] = None,
    restorability_score: float = 70.0,
    applicable_goals: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, Dict[str, float], PhaseGateLogEntry]:
    """
    Convenience-Wrapper: Führt eine Phase aus mit Musical-Goals-Schutz.

    Args:
        phase: PhaseInterface-Instanz
        audio: Input-Audio (float32, 48 kHz)
        sr: 48000 Hz
        scores_before: Vorherige Goal-Scores (optional)
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100], bestimmt
                             adaptiven REGRESSION_THRESHOLD (§2.29).
        applicable_goals: Aus GoalApplicabilityFilter — nur diese Ziele geprüft.

    Returns:
        (audio_out, scores_after, log_entry)
    """
    return get_phase_gate().wrap_phase(
        phase,
        audio,
        sr,
        scores_before,
        restorability_score=restorability_score,
        applicable_goals=applicable_goals,
    )
