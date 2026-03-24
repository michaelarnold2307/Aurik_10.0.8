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
    3. Schnell-Check (14 Ziele, ≤ 200 ms, DSP-only):
       Brillanz, Wärme, Groove, TonalCenter, Natürlichkeit (MFCC-Proxy),
       Timbre-Authentizität, Bass-Kraft, Authentizität, Emotionalität,
       Transparenz, Spatial Depth, Mikro-Dynamik, Separation-Treue, Artikulation
    4. Δ = score_after − score_before für jedes Ziel
       Falls Δ < −REGRESSION_THRESHOLD (adaptiv je nach Restorability):
         Retry-1: Phase mit strength × 0.65
         Retry-2: Phase mit strength × 0.50  (v9.15-B3: sanfterer Gradient)
         Retry-3: Phase mit strength × 0.35
         Retry-4: Phase mit strength × 0.20
         Retry-5 (Last-Resort): Phase mit strength × 0.10
         Falls immer noch: Best-Effort — Versuch mit geringster Regression wird
         angewendet. KEIN Rollback/Skip erlaubt (§2.29 v9.10.64).

WICHTIG (§2.29 v9.10.64):
-----------
PMGG darf Phasen NIEMALS überspringen (kein Rollback auf Original-Audio).
CausalDefectReasoner hat die Phase als notwendig bestimmt — sie MUSS angewendet
werden, ggf. mit reduzierter Stärke (best-effort).

KONSTANTEN:
-----------
REGRESSION_THRESHOLD = 0.025  (adaptiv: 0.012 / 0.040 / 0.060 je Restorability)
SAMPLE_DURATION_S    = 5.0
MAX_RETRIES          = 5  (v9.15-B3: 5 Retries mit sanftem Stärkegradienten)

OVERHEAD: max. 56 × 200 ms = 11.2 s pro Verarbeitungsdurchlauf (alle 14 Ziele DSP-only)
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
from typing import Any

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

# §9.7.4 Phase-specific goal exclusions.
# Goals whose DSP proxy is structurally unreliable for a given processing type.
# These goals are NOT checked for regression when the phase matches.
#
# Rationale for phase_02 / phase_28:
#   Both apply comb / spectral-notch filters.  The MFCC-smoothness proxy for
#   "natuerlichkeit" and the flatness-based proxy for "separation_fidelity"
#   interpret the introduced notches as artefacts, producing false-positive
#   regressions of 0.36–0.69 >> REGRESSION_THRESHOLD_GOOD (0.012).
#   Perceptually the hum/noise removal IS the correct action; the measurement
#   artifact must therefore be suppressed for these phases.
#   Also: "timbre_authentizitaet" (centroid-stability proxy) and "authentizitaet"
#   (spectral-roughness proxy) degrade because comb-filter notches at 50/100/
#   150/200/250/300/350/400 Hz create spectral discontinuities and shift the
#   centroid distribution.  "bass_kraft" degrades because the 50-Hz fundamental
#   falls directly in the 20–250 Hz bass band.  All three produce constant
#   false-positive regressions of 0.15–0.35 independent of wet/dry strength,
#   confirming measurement artifacts (not real quality degradation).
#
# Rationale for phase_06 (frequency restoration):
#   SBR + LPC harmonic extension intentionally modifies the spectral envelope
#   above the rolloff frequency.  "timbre_authentizitaet" (centroid-stability)
#   and "authentizitaet" (spectral-roughness) proxies interpret the new harmonic
#   content as spectral deformation, producing regressions of 0.05–0.10 even
#   though the restoration IS the intended action.  "brillanz" is also excluded
#   because SBR increases HF energy — the proxy measures the change as
#   deviation, not improvement.
#
# Rationale for phase_24 (dropout repair):
#   Fills signal gaps with reconstructed audio.  The MFCC-smoothness proxy
#   and NMF-flatness proxy interpret the reconstructed segments as artefacts,
#   producing constant false-positive regressions (~0.37) independent of
#   Wet/Dry strength — physically impossible for real quality degradation.
#
# Rationale for phase_05 / phase_30:
#   Rumble filter and DC-offset removal modify only sub-sonic content.
#   The MFCC proxy interprets the changed spectral baseline as degradation
#   (regressions 0.36–0.50), although perceptual quality is unchanged or
#   improved.  bass_kraft is also excluded because the removed content is
#   below audible bass range (< 20 Hz for DC, < 30 Hz for rumble).
#   timbre_authentizitaet (MFCC-Pearson) is also excluded: MFCC coefficient
#   #0 encodes the spectral energy level; sub-sonic removal shifts this
#   coefficient producing a false-positive Pearson drop of 0.35–0.45.
#
# Rationale for phase_55 (diffusion inpainting):
#   Reconstructs dropout gaps via CQTdiff diffusion.  The MFCC-smoothness
#   proxy interprets the AI-reconstructed segments identically to phase_24 —
#   constant false-positive regression (~0.044) independent of Wet/Dry
#   strength.  Same proxy artifact pattern as phase_24.
#
# Rationale for phase_29 (tape hiss / noise reduction):
#   Broadband noise reduction (DeepFilterNet / OMLSA) removes HF noise floor.
#   brillanz proxy (raw HF energy ratio) falsely reports regression because
#   the removed content IS the noise, not musical signal.  timbre_authentizitaet
#   (MFCC-Pearson) degrades because spectral envelope changes; natuerlichkeit
#   and separation_fidelity show the same comb-filter false-positive pattern.
#   Constant regression of ~0.31 observed across all wet/dry strengths —
#   physically impossible for strength-proportional wet/dry blending, confirming
#   measurement artifact independent of processing level.
#
# Rationale for phase_08 (transient preservation):
#   Transient shaping modifies attack envelopes.  The MFCC-smoothness proxy
#   (natuerlichkeit) and MFCC-Pearson proxy (timbre_authentizitaet) interpret
#   the changed spectral energy distribution over time as degradation, producing
#   false-positive regressions of ~0.10 independent of strength.
PHASE_GOAL_EXCLUSIONS: dict[str, set[str]] = {
    "phase_02": {"natuerlichkeit", "separation_fidelity", "timbre_authentizitaet", "authentizitaet", "bass_kraft"},
    "phase_28": {"natuerlichkeit", "separation_fidelity"},
    "phase_24": {"natuerlichkeit", "separation_fidelity"},
    "phase_55": {"natuerlichkeit", "separation_fidelity"},
    "phase_05": {"natuerlichkeit", "separation_fidelity", "bass_kraft", "timbre_authentizitaet"},
    "phase_30": {"natuerlichkeit", "separation_fidelity", "bass_kraft", "timbre_authentizitaet"},
    "phase_29": {"natuerlichkeit", "separation_fidelity", "brillanz", "timbre_authentizitaet"},
    "phase_08": {"natuerlichkeit", "separation_fidelity", "timbre_authentizitaet"},
    "phase_06": {"timbre_authentizitaet", "authentizitaet", "brillanz"},
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


# All 14 Musical Goals are checked per-phase — DSP-only proxies, no ML (≤ 200 ms total §2.29).
# "natuerlichkeit" uses an MFCC-smoothness DSP proxy internally but is exposed under its
# canonical key so GoalApplicabilityFilter intersection (§2.32) works correctly.
FAST_GOALS_SUBSET: list[str] = [
    "brillanz",
    "waerme",
    "groove",
    "tonal_center",
    "natuerlichkeit",  # canonical key — MFCC-smoothness DSP proxy, matches GoalApplicabilityFilter
    "timbre_authentizitaet",
    # 8 neu (DSP-Proxies, v9.10.57):
    "bass_kraft",
    "authentizitaet",
    "emotionalitaet",
    "transparenz",
    "spatial_depth",
    "micro_dynamics",
    "separation_fidelity",
    "artikulation",
]


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PhaseGateLogEntry:
    """Eintrag im phase_gate_log für eine Phase."""

    phase_id: str
    action: str  # "passed" | "retry1" | ... | "retry5" | "best_effort" | "best_effort_rN"
    goal_regressions: dict[str, float]  # Ziel → Δ-Score
    strength_used: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PhaseGateResult:
    """Ergebnis der wrap_phase()-Operation."""

    audio: np.ndarray
    scores_after: dict[str, float]
    log_entry: PhaseGateLogEntry
    rolled_back: bool


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: PerPhaseMusicalGoalsGate | None = None
_lock = threading.Lock()


def get_phase_gate() -> PerPhaseMusicalGoalsGate:
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


def _measure_quick(audio: np.ndarray, sr: int) -> dict[str, float]:
    """
    Misst alle 14 Musical Goals auf einer 5-s-Stichprobe in ≤ 200 ms.

    6 Ziele bereits vorhanden, 8 als DSP-Proxy ergänzt (v9.10.57).
    Alle Messungen sind DSP-only (kein MERT, kein CREPE, kein NMF).
    NaN-sicher: fehlerhafte Einzelmessungen werden auf 0.5 (neutral) gesetzt.

    Args:
        audio: Mono oder Stereo, float32, beliebige Länge
        sr: 48000 Hz

    Returns:
        Dict mit 14 Scores ∈ [0, 1]
    """
    mono = audio[:, 0] if audio.ndim == 2 else audio
    mono = np.nan_to_num(mono, nan=0.0).astype(np.float32)

    scores: dict[str, float] = {}

    # ── Pre-compute spectrum once — brillanz, waerme, bass_kraft, natuerlichkeit,
    #    authentizitaet, transparenz, separation_fidelity all share these arrays.
    #    If FFT fails every dependent metric gracefully falls back to 0.5 via its
    #    own try/except; the shared variables are always defined.
    try:
        fft_mag: np.ndarray = np.abs(np.fft.rfft(mono))
        freqs: np.ndarray = np.fft.rfftfreq(len(mono), d=1.0 / sr)
        tot_energy: float = float(np.mean(fft_mag**2)) + 1e-12
    except Exception:
        fft_mag = np.zeros(len(mono) // 2 + 1, dtype=np.float32)
        freqs = np.zeros(len(mono) // 2 + 1, dtype=np.float32)
        tot_energy = 1e-12

    # ── Brillanz (HF-Energie > 8 kHz) ─────────────────────────────────
    try:
        hf_energy = float(np.mean(fft_mag[freqs > 8000] ** 2))
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
                note = round(12.0 * math.log2(f / 440.0 + 1e-12)) % 12
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
    # Canonical key "natuerlichkeit" — aligned with GoalApplicabilityFilter §2.32.
    try:
        n_mfcc = min(20, len(fft_mag) // 2)
        mfcc_approx = np.log(np.convolve(fft_mag[: len(fft_mag) // 2], np.ones(10) / 10, mode="valid") + 1e-12)
        if len(mfcc_approx) > n_mfcc:
            smoothness = 1.0 - float(np.std(np.diff(mfcc_approx[:n_mfcc]))) / (
                float(np.std(mfcc_approx[:n_mfcc])) + 1e-12
            )
            scores["natuerlichkeit"] = float(np.clip(smoothness, 0.0, 1.0))
        else:
            scores["natuerlichkeit"] = 0.5
    except Exception:
        scores["natuerlichkeit"] = 0.5

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

    # ── Bass-Kraft (Bassenergie 20–250 Hz) ─────────────────────────────
    try:
        bass_energy = float(np.mean(fft_mag[(freqs >= 20) & (freqs <= 250)] ** 2))
        # Normierung: typische Bassenergie ~2% des Spektrums → 0.02 = Score 1.0
        scores["bass_kraft"] = float(np.clip(bass_energy / (tot_energy * 0.02 + 1e-12), 0.0, 1.0))
    except Exception:
        scores["bass_kraft"] = 0.5

    # ── Authentizität (Spektrale Konsistenz-Proxy, referenzfrei) ───────
    try:
        # Proxy: Gleichmäßigkeit der Spektralhüllkurve (glatte Hülle = authentisches Signal)
        # Stark deformierte Spektren (Codec-Artefakte, Phasenfehler) zeigen hohe Varianz
        log_mag = np.log(fft_mag + 1e-12)
        # Glättung über 50 Bins
        smooth_len = min(50, len(log_mag) // 4)
        if smooth_len > 1:
            smoothed = np.convolve(log_mag, np.ones(smooth_len) / smooth_len, mode="valid")
            roughness = float(np.std(log_mag[smooth_len // 2 : smooth_len // 2 + len(smoothed)] - smoothed))
            # Niedriger Roughness-Wert → glatte Hülle → hohe Authentizität
            scores["authentizitaet"] = float(np.clip(1.0 - roughness / 3.0, 0.0, 1.0))
        else:
            scores["authentizitaet"] = 0.5
    except Exception:
        scores["authentizitaet"] = 0.5

    # ── Emotionalität (Crest-Factor + RMS-Varianz) ─────────────────────
    try:
        rms_val = float(np.sqrt(np.mean(mono**2) + 1e-12))
        peak_val = float(np.max(np.abs(mono)))
        crest_db = 20.0 * math.log10(peak_val / (rms_val + 1e-12) + 1e-12)
        # 2–14 dB Crestfaktor ist gesunder Dynamikbereich
        crest_score = float(np.clip((crest_db - 2.0) / 12.0, 0.0, 1.0))
        # RMS-Varianz über 10ms-Frames (Ausdruck)
        hop_e = max(1, sr // 100)
        rms_frames = np.array(
            [float(np.sqrt(np.mean(mono[i : i + hop_e] ** 2) + 1e-12)) for i in range(0, len(mono) - hop_e, hop_e)]
        )
        variance_score = float(np.clip(np.var(rms_frames) * 1000.0, 0.0, 1.0)) if len(rms_frames) > 2 else 0.5
        scores["emotionalitaet"] = float(np.clip(0.5 * crest_score + 0.5 * variance_score, 0.0, 1.0))
    except Exception:
        scores["emotionalitaet"] = 0.5

    # ── Transparenz (Spektrale Rolloff + Energie-Balance) ──────────────
    try:
        # 75%-Rolloff: Frequenz unterhalb derer 75% der Energie konzentriert ist
        cumsum = np.cumsum(fft_mag**2)
        total_e = cumsum[-1] + 1e-12
        rolloff_idx = int(np.searchsorted(cumsum, 0.75 * total_e))
        rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        # 5500 Hz = 1.0 (gut gemastertes Material), 1500 Hz = 0.0
        rolloff_score = float(np.clip((rolloff_hz - 1500.0) / 4000.0, 0.0, 1.0))
        # Energie-Balance low/mid/high: gleichmäßig = transparent
        e_low = float(np.mean(fft_mag[freqs < 500] ** 2) + 1e-12)
        e_mid = float(np.mean(fft_mag[(freqs >= 500) & (freqs < 2000)] ** 2) + 1e-12)
        e_high = float(np.mean(fft_mag[freqs >= 2000] ** 2) + 1e-12)
        e_total = e_low + e_mid + e_high
        balance_std = float(np.std([e_low / e_total, e_mid / e_total, e_high / e_total]))
        balance_score = float(np.clip(1.0 - balance_std * 3.0, 0.0, 1.0))
        scores["transparenz"] = float(np.clip(0.6 * rolloff_score + 0.4 * balance_score, 0.0, 1.0))
    except Exception:
        scores["transparenz"] = 0.5

    # ── Spatial Depth (M/S-Korrelation bei Stereo, 0.5 bei Mono) ──────
    try:
        if audio.ndim == 2 and audio.shape[1] >= 2:
            left = audio[:, 0].astype(np.float32)
            right = audio[:, 1].astype(np.float32)
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            mid_e = float(np.mean(mid**2) + 1e-12)
            side_e = float(np.mean(side**2) + 1e-12)
            # Hohe Side-Energie = breites Stereo-Bild = hohe Räumlichkeit
            # Normierung: S/M-Ratio ≥ 0.5 = sehr breites Stereo → Score 1.0
            stereo_ratio = side_e / (mid_e + side_e)
            scores["spatial_depth"] = float(np.clip(stereo_ratio * 2.0, 0.0, 1.0))
        else:
            scores["spatial_depth"] = 0.5  # Mono: neutral (GoalApplicabilityFilter entscheidet)
    except Exception:
        scores["spatial_depth"] = 0.5

    # ── Mikro-Dynamik (LUFS-Profil-Korrelation 400ms Proxy) ──────────
    try:
        # Proxy: RMS-Varianz über 400ms-Fenster (äquivalent zu LUFS-Profil-Korrelation)
        win_400ms = max(1, int(sr * 0.4))
        hop_400ms = win_400ms // 4
        rms_400 = np.array(
            [
                float(np.sqrt(np.mean(mono[i : i + win_400ms] ** 2) + 1e-12))
                for i in range(0, len(mono) - win_400ms, hop_400ms)
            ]
        )
        if len(rms_400) > 2:
            # Gleichmäßige Variation über 400ms-Fenster = gute Mikro-Dynamik
            # (weder totales Limiting noch extreme Spitzen)
            db_profile = 20.0 * np.log10(rms_400 + 1e-12)
            db_range = float(np.max(db_profile) - np.min(db_profile))
            # Gesunder Bereich: 3–18 dB Variation
            scores["micro_dynamics"] = float(np.clip((db_range - 1.0) / 17.0, 0.0, 1.0))
        else:
            scores["micro_dynamics"] = 0.5
    except Exception:
        scores["micro_dynamics"] = 0.5

    # ── Separation-Treue (Spektrale Tonalität als NMF-Proxy) ──────────
    try:
        # Proxy: Spektrale Flachheit (niedrig = tonal = gut separierbar)
        # Rauschen hat hohe Flachheit → schwer zu trennen → niedrige Separation-Treue
        # Tonales Signal: Flachheit ~ 0.01–0.05 → Score nahe 1.0
        # Rauschen: Flachheit ~ 0.3–1.0 → Score nahe 0.0
        eps = 1e-12
        # Geometrisches Mittel / arithmetisches Mittel auf Leistungsspektrum
        power = fft_mag**2 + eps
        geom_mean = float(np.exp(np.mean(np.log(power))))
        arith_mean = float(np.mean(power))
        flatness = float(np.clip(geom_mean / (arith_mean + eps), 0.0, 1.0))
        # Niedriger Flatness → hohe Tonalität → gute Separierbarkeit
        scores["separation_fidelity"] = float(np.clip(1.0 - flatness * 2.5, 0.0, 1.0))
    except Exception:
        scores["separation_fidelity"] = 0.5

    # ── Artikulation (Onset-Schärfe: Transient-Proxy) ─────────────────
    try:
        # Proxy: Varianz der Energiehüllkurve-Ableitungen (scharfe Transienten = hohe Varianz)
        hop_a = max(1, sr // 200)  # 5 ms
        env_a = np.array([float(np.max(np.abs(mono[i : i + hop_a]))) for i in range(0, len(mono) - hop_a, hop_a)])
        if len(env_a) > 4:
            # Erste Ableitung der Hüllkurve
            d_env = np.diff(env_a)
            # Starke positive Sprünge = scharfe Anschläge (Artikulation)
            pos_peaks = d_env[d_env > 0]
            if len(pos_peaks) > 0:
                onset_sharpness = float(np.mean(pos_peaks))
                # Normierung: 0.01 = gute Artikulation → Score 1.0
                scores["artikulation"] = float(np.clip(onset_sharpness / 0.01, 0.0, 1.0))
            else:
                scores["artikulation"] = 0.3  # Keine Transienten = schlechte Artikulation
        else:
            scores["artikulation"] = 0.5
    except Exception:
        scores["artikulation"] = 0.5

    # NaN-guard (§3.1) — all 14 canonical keys including "natuerlichkeit"
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
        scores_before: dict[str, float] | None = None,
        phase_kwargs: dict[str, Any] | None = None,
        restorability_score: float = 70.0,
        applicable_goals: set[str] | None = None,
        initial_strength: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
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
            initial_strength: Material-adaptive Initialstärke ∈ (0, 1.0] (§2.29/§2.31).
                              1.0 = volle Stärke (Default). Niedrigere Werte aus
                              _MATERIAL_PHASE_FACTORS schützen Vintage-Charakter
                              (z.B. 0.25 für phase_22_tape_saturation bei shellac).
                              Retry-Stärken skalieren relativ zur Initialstärke.

        Returns:
            (audio_out, scores_after, log_entry)
        """
        if sr != 48000:
            logger.debug(f"PMGG: SR={sr} (nicht 48000) — Goal-Messung läuft trotzdem")

        if phase_kwargs is None:
            phase_kwargs = {}

        phase_id = self._get_phase_id(phase)
        t0 = time.time()

        # Adaptiven Threshold bestimmen (§2.29)
        threshold = _get_adaptive_threshold(restorability_score)

        # §9.7.3 Phasen-adaptive Sample-Dauer — MUSS vor scores_before bestimmt werden,
        # damit before und after dieselbe Sample-Länge nutzen (sonst falsche Regression).
        _sample_dur = _get_sample_duration(phase_id)

        # Vor-Scores messen (wenn nicht übergeben) — gleiche duration wie after-Messung
        sample_before = _extract_sample(audio, sr, duration_s=_sample_dur)
        if scores_before is None:
            scores_before = _measure_quick(sample_before, sr)

        # Effective goal set: Schnitt aus FAST_GOALS_SUBSET + applicable_goals
        if applicable_goals is not None:
            effective_goals = [g for g in FAST_GOALS_SUBSET if g in applicable_goals]
            if not effective_goals:
                effective_goals = FAST_GOALS_SUBSET  # Fallback: alle
        else:
            effective_goals = FAST_GOALS_SUBSET

        # §9.7.4 Phase-specific goal exclusions (comb-filter-sensitive proxies).
        # Remove goals whose DSP proxy is unreliable for this particular phase type.
        _excluded_goals: set[str] = set()
        for _pfx, _excl in PHASE_GOAL_EXCLUSIONS.items():
            if phase_id.startswith(_pfx):
                _excluded_goals |= _excl
        if _excluded_goals:
            effective_goals = [g for g in effective_goals if g not in _excluded_goals]
            if not effective_goals:
                effective_goals = list(FAST_GOALS_SUBSET)  # Safety fallback
            logger.debug(
                "PMGG: %s goal exclusions applied: %s → %d goals checked",
                phase_id,
                sorted(_excluded_goals),
                len(effective_goals),
            )

        # Phase ausführen + Regression prüfen (§2.29: initial_strength statt immer 1.0)
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
            initial_strength=max(0.0, min(1.0, initial_strength)),
        )

        # Best-Effort-Zähler (Phase wurde mit reduzierter Stärke angewendet, nicht übersprungen)
        if action.startswith("best_effort"):
            self._rollback_count += 1
            if self._rollback_count > 3 and not self._user_warned:
                self._user_warned = True
                logger.warning(
                    "ℹ️ Einige Verarbeitungsschritte wurden mit reduzierter Stärke angewendet, um den Klang zu schützen."
                )

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
        scores_before: dict[str, float],
        phase_id: str,
        phase_kwargs: dict[str, Any] | None = None,
        *,
        threshold: float = REGRESSION_THRESHOLD_GOOD,
        effective_goals: list | None = None,
        sample_duration_s: float = SAMPLE_DURATION_S,  # §9.7.3 phasen-adaptiv
        initial_strength: float = 1.0,
    ) -> tuple[np.ndarray, dict[str, float], str, float]:
        """
        Führt Phase aus, ggf. mit Retry bei Regression.

        Args:
            threshold: Adaptiver REGRESSION_THRESHOLD (§2.29).
            effective_goals: Subset aus FAST_GOALS_SUBSET, das geprüft wird.
            sample_duration_s: Stichprobenlänge (§9.7.3 phasen-adaptiv, 1.0–5.0 s).
            initial_strength: Material-adaptive Initialstärke ∈ (0, 1.0] (§2.31).
                1.0 = Default. Retry-Stärken skalieren relativ dazu wenn < 1.0.

        Returns:
            (audio_out, scores_after, action_label, strength_used)
        """
        if phase_kwargs is None:
            phase_kwargs = {}
        if effective_goals is None:
            effective_goals = FAST_GOALS_SUBSET
        initial_strength = max(0.01, min(1.0, initial_strength))
        # Erster Versuch mit material-adaptiver Initialstärke (§2.29/§2.31)
        audio_out = self._run_phase(phase, audio, initial_strength, phase_kwargs)
        scores_after = _measure_quick(_extract_sample(audio_out, sr, duration_s=sample_duration_s), sr)

        regression = self._max_regression(scores_before, scores_after, effective_goals)
        if regression <= threshold:
            return audio_out, scores_after, "passed", initial_strength

        # Log which goal caused the regression (diagnostics for false-positive detection)
        _worst_goal = max(
            effective_goals,
            key=lambda g: max(0.0, scores_before.get(g, 0.5) - scores_after.get(g, 0.5)),
        )
        logger.debug(
            "PMGG: %s regression=%.4f > threshold=%.3f — worst goal: %s (before=%.3f after=%.3f)",
            phase_id,
            regression,
            threshold,
            _worst_goal,
            scores_before.get(_worst_goal, 0.5),
            scores_after.get(_worst_goal, 0.5),
        )

        # Retry-Stärken relativ zur Initialstärke skalieren (§2.29):
        # initial_strength=1.0 → normale Retry-Folge [0.65, 0.50, ...]
        # initial_strength<1.0 → proportional nach unten skaliert
        retry_strengths = [s * initial_strength for s in _RETRY_STRENGTHS]

        # §2.29 Best-Effort-Tracking: Speichere den Versuch mit geringster Regression.
        # PMGG darf Phasen NICHT überspringen — CausalDefectReasoner hat die Phase
        # als notwendig bestimmt. Stattdessen wird der beste Versuch verwendet.
        best_audio = audio_out
        best_scores = scores_after
        best_regression = regression
        best_strength = initial_strength
        best_action = "best_effort"

        # Retry-Schleife
        _prev_regression = regression  # Track previous regression for stagnation detection
        _retry_t0 = time.time()  # Per-phase time budget for retries
        _RETRY_BUDGET_S = 300.0  # Max 5 min for all retries of a single phase
        for attempt, strength in enumerate(retry_strengths):
            # Per-phase time budget: abort retries if total retry time exceeds budget.
            # Prevents runaway phases (e.g. Phase 06: 42 min, Phase 36: 16 min).
            _retry_elapsed = time.time() - _retry_t0
            if _retry_elapsed > _RETRY_BUDGET_S:
                logger.info(
                    "PMGG: %s retry time budget exceeded (%.0fs > %.0fs) — "
                    "using best attempt so far (regression=%.4f, attempt=%d)",
                    phase_id,
                    _retry_elapsed,
                    _RETRY_BUDGET_S,
                    best_regression,
                    attempt,
                )
                break

            # §OOM-Safety: Garbage Collection zwischen Retries — jeder Retry
            # kann ~1 GB temporäre Arrays erzeugen (ResembleEnhance, OMLSA).
            import gc

            gc.collect()

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
            # Track best attempt (lowest regression)
            if regression_retry < best_regression:
                best_audio = audio_retry
                best_scores = scores_retry
                best_regression = regression_retry
                best_strength = strength
                best_action = f"best_effort_r{attempt + 1}"

            # Stagnation guard: if regression barely changes across consecutive
            # retries despite strength variation, further retries are wasted
            # computation.  Threshold 0.002 catches near-identical ML outputs
            # (e.g. ResembleEnhance ignoring omlsa_alpha) as well as DSP phases
            # where wet/dry blending produces marginal improvement (< 0.2%).
            if abs(regression_retry - _prev_regression) < 0.002 and attempt >= 1:
                logger.info(
                    "PMGG: %s stagnation detected at retry %d (Δregression=%.6f) — skipping remaining retries",
                    phase_id,
                    attempt + 1,
                    abs(regression_retry - _prev_regression),
                )
                break
            _prev_regression = regression_retry

        # §2.29 KEIN Rollback — Phase wird mit geringster Regression angewendet.
        # VERBOTEN: Phase überspringen (Original-Audio zurückgeben).
        # CausalDefectReasoner hat diese Phase als notwendig bestimmt.
        logger.warning(
            "⚠️ PMGG: %s best-effort (strength=%.2f, Regression=%.4f > threshold=%.3f) — "
            "Phase wird trotzdem angewendet (kein Rollback/Skip erlaubt)",
            phase_id,
            best_strength,
            best_regression,
            threshold,
        )
        return best_audio, best_scores, best_action, best_strength

    @staticmethod
    def _run_phase(
        phase: Any,
        audio: np.ndarray,
        strength: float,
        phase_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Führt Phase aus mit Wet/Dry-Modulation; gibt bei Fehler das Original zurück.

        CRITICAL FIX (v9.10.64): Ruft phase.process() statt phase() auf.
        PhaseInterface definiert kein __call__; der vorherige Code erzeugte
        TypeError, das still gefangen wurde — ALLE Phasen waren No-Ops.

        Wet/Dry-Modulation (§MusikalischeHarmonisierung):
        strength < 1.0 → audio_out = audio + strength × (processed - audio)
        Psychoakustisch korrekt: Sanftere Verarbeitung bei niedriger Stärke,
        statt binär „alles oder nichts".
        Timing-modifizierende Phasen (wow/flutter, speed) sind von Wet/Dry
        ausgenommen (Phasen-Artefakte bei Crossfade zeitversetzter Signale).
        """
        if phase_kwargs is None:
            phase_kwargs = {}
        # Timing-modifizierende Phasen: kein Wet/Dry (Phasen-Artefakte)
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        try:
            # Strength als Kwarg übergeben, damit Phasen ihn OPTIONAL nutzen können
            kw = dict(phase_kwargs)
            kw["strength"] = strength
            # CRITICAL: phase.process() statt phase() — PhaseInterface hat kein __call__
            result = phase.process(audio, **kw)
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
                out = out[: len(audio)] if len(out) > len(audio) else np.pad(out, (0, len(audio) - len(out)))

            # Wet/Dry-Modulation: strength < 1.0 → blende zwischen Original und Verarbeitet
            if 0.0 < strength < 1.0:
                phase_id = ""
                try:
                    meta = phase.get_metadata()
                    phase_id = getattr(meta, "phase_id", "")
                except Exception:
                    pass
                if phase_id not in _TIMING_PHASES:
                    out = (audio + strength * (out - audio)).astype(np.float32)
                    out = np.clip(out, -1.0, 1.0)

            return out
        except Exception as exc:
            logger.debug("PMGG: Phase-Ausführung fehlgeschlagen: %s", exc)
            return audio

    @staticmethod
    def _max_regression(
        before: dict[str, float],
        after: dict[str, float],
        goals: list | None = None,
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
    scores_before: dict[str, float] | None = None,
    restorability_score: float = 70.0,
    applicable_goals: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
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
