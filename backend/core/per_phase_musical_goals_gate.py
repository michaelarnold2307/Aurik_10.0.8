"""
PerPhaseMusicalGoalsGate (PMGG) — Aurik 9.0 §2.29.

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

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PRECISE_METRICS_LOCK = threading.Lock()
_PRECISE_METRICS: dict[str, Any] | None = None
_PRECISE_OVERRIDE_WARN_MS: float = 200.0


# ---------------------------------------------------------------------------
# Konstanten (§2.29) — restorability-adaptive Schwellwerte
# ---------------------------------------------------------------------------
# Feste Einzel-Schwelle (Legacy-Fallback, nicht mehr primär verwendet)
REGRESSION_THRESHOLD: float = 0.025

# Restorability-adaptive Schwellwerte (§2.29 Spec)
# v9.10.76: 0.012 → 0.030 (DSP-Proxy-Messrauschen 0.01–0.05).
# v9.10.77: 0.030 → 0.020 — §9.7.5 Reference-Aware Preservation Corrections
# eliminieren den größten Teil des Messrauschens; engere Schwellwerte fangen
# nun echte Regressionen zuverlässiger ab ohne False-Positives.
REGRESSION_THRESHOLD_GOOD: float = 0.020  # restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = 0.035  # restorability 40–69 (entspannter)
REGRESSION_THRESHOLD_POOR: float = (
    0.040  # restorability < 40 (maximal tolerant) — reduced from 0.055 to prevent best_effort cascades
)

# §2.54 Material-bonus: analog/physical carriers need more tolerance because
# carrier-repair phases intentionally shift spectral fingerprints (Reference-
# Paradoxon §2.44). CD/DAT need no bonus — proxy metrics are reliable there.
_MATERIAL_THRESHOLD_BONUS: dict[str, float] = {
    "wax_cylinder": 0.022,  # most degraded — carrier-repair phases radically alter signal
    "shellac": 0.018,
    "wire_recording": 0.016,
    "optical_film": 0.010,
    "vinyl": 0.009,
    "reel_tape": 0.008,
    "tape": 0.007,
    "radio_broadcast": 0.006,
    "cassette": 0.006,
    "mp3_low": 0.005,  # codec artefacts → repair changes look regressive to proxies
    "minidisc": 0.004,
    "mp3_high": 0.002,
    "cd_digital": 0.000,
    "dat": 0.000,
    "unknown": 0.003,
}

# ---------------------------------------------------------------------------
# §2.29 v9.10.77: Priority-aware Retry-Budget
# ---------------------------------------------------------------------------
# P1/P2 regressions trigger full retry cascade (4 Retries + Emergency).
# P3 regressions trigger max 2 retries with 1.5× relaxed threshold.
# P4/P5 regressions never trigger retries — only logged.
#
# Begründung (Pareto-Analyse): Hohe P3–P5-Schwellwerte verursachten unnötige
# PMGG-Retries (CPU-Verschwendung) und Cross-Goal-Damage (Natürlichkeit/
# Authentizität-Regression durch Over-Optimization nachrangiger Ziele).
# GoalPriorityProtocol.PRIORITY_MAP ist die Autoritätsquelle.
# ---------------------------------------------------------------------------
_PRIORITY_MAX_RETRIES: dict[int, int] = {
    1: 4,  # P1: Natürlichkeit, Authentizität — volle Retry-Kaskade
    2: 4,  # P2: TonalCenter, Timbre, Artikulation — volle Retry-Kaskade
    3: 2,  # P3: Emotionalität, MicroDynamics, Groove — max 2 Retries
    4: 0,  # P4: Transparenz, Wärme, Bass-Kraft, SepFidelity — kein Retry
    5: 0,  # P5: Brillanz, SpatialDepth — kein Retry
}

# Regression-Toleranz-Multiplikator pro Priorität.
# P3-Ziele haben 1.5× mehr Toleranz als P1/P2, bevor ein Retry ausgelöst wird.
_PRIORITY_THRESHOLD_FACTOR: dict[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.5,
    4: 99.0,  # Effektiv kein Retry (Threshold × 99 = immer unter)
    5: 99.0,
}

# §2.47b JND-Effektivitätsschwelle — Sub-Threshold Phase Marking
# If ALL applicable goal-deltas are ≥ 0 and < JND → "sub_threshold" (no retry, accept)
JND_MIN_DELTA: dict[str, float] = {
    "natuerlichkeit": 0.015,  # 1.5 % Timbre-JND (Zwicker 1990)
    "authentizitaet": 0.015,
    "tonal_center": 0.010,  # Tonal centre: more sensitive (Krumhansl 1990)
    "timbre_authentizitaet": 0.015,
    "artikulation": 0.012,  # Transient timing: more sensitive than long-term spectrum
    "emotionalitaet": 0.018,
    "micro_dynamics": 0.015,
    "groove": 0.012,
    "transparenz": 0.015,
    "waerme": 0.020,  # Warmth perception: slower integration
    "bass_kraft": 0.015,
    "separation_fidelity": 0.018,
    "brillanz": 0.020,
    "spatial_depth": 0.025,  # Room impression: weakest JND sensitivity
}

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
# v9.10.77: Exclusions significantly reduced thanks to §9.7.5 reference-aware
# preservation corrections.  Goals with spectral/temporal correlation support
# are now checked even for phases that previously triggered false positives.
# Only goals where processing FUNDAMENTALLY changes the measured quantity
# (and correlation cannot distinguish intentional change from degradation)
# remain excluded.
#
# Rationale for remaining exclusions:
#
# phase_02 (hum removal): 50/100/.../400 Hz comb-filter creates spectral
#   notches directly in the bass band → bass_kraft LF correlation still sees
#   notches as degradation because they ARE spectral removal (intentional).
#   authentizitaet excluded: comb-filter notches create spectral roughness
#   that is the intended action, not degradation.
#
# phase_04 (EQ correction): Spectral redistribution IS the core function.
#   transparenz (rolloff + balance) changes deliberately.
#
# phase_06 (frequency restoration): SBR intentionally adds HF content that
#   the reference doesn't have → correlation is LOW by design.
#   brillanz excluded because the increase IS the goal.
#
# phase_18 / phase_26 / phase_36: Dynamics-modifying phases intentionally
#   change the temporal envelope → micro_dynamics measures the intended change.
PHASE_GOAL_EXCLUSIONS: dict[str, set[str]] = {
    # Hum removal: comb-filter notches in bass band + spectral roughness.
    # natuerlichkeit excluded: CREPE voicing analysis in NatuerlichkeitMetric
    # flags 50/100 Hz notch-induced spectral-flatness changes as P1 regression.
    # transparenz excluded: 50/100/150/250/300 Hz hum harmonics are narrow spectral
    # peaks in the 250-500 Hz band (5th and 6th harmonic of 50 Hz hum sit at exactly
    # 250 and 300 Hz).  Notch filters remove these peaks, which lowers p95 in the
    # first octave band of the §9.7.13 multi-band crest proxy → false P4 regression
    # even though audio quality has improved.  Unlike broadband denoising (phase_03)
    # where noise fills the ENTIRE band floor (elevating p50), hum-notch removal only
    # reduces isolated peaks → net crest DECREASE in the 250-500 Hz band.  The
    # §9.7.13 fix does not cover narrowband-notch-induced peak removal.
    # groove excluded (P3 root cause, 2026-03-30): hum removal does not affect
    # timing or rhythmic events, but the GrooveMetric onset/DTW proxy is sensitive
    # to LF spectral energy changes (50–200 Hz). Real-run stagnation Δ=0.000000
    # across all retries confirms filter-independence — this is a measurement
    # artifact. Groove 0.1526 regression proved to produce false catastrophic
    # PMGG cascades. Export gate still enforces GrooveMetric threshold globally.
    # timbre_authentizitaet excluded (P2 root cause, 2026-03-30): spectral notches
    # directly disturb the MFCC-Pearson and spectral-centroid correlation proxies.
    # Even a shallow notch at 50 Hz shifts lower MFCC coefficients, driving the
    # timbre proxy below threshold despite no perceptual timbre degradation.
    "phase_02": {
        "bass_kraft",
        "authentizitaet",
        "natuerlichkeit",
        "transparenz",
        "groove",
        "timbre_authentizitaet",
        "artikulation",  # hum-notch removal changes onset rise-time in notched bands (LF energy alters ArticulationMetric transient-rise proxy) — CIG has this, PMGG sync §2.54
    },
    # Reconstruction phases: spectral correlation handles reconstruction well;
    # only keep exclusions where AI-generated content has low correlation by design
    # natuerlichkeit excluded: gap-fill synthesis produces content absent from
    # reference; CREPE voicing score on synthesised audio is unreliable.
    # artikulation excluded (P2 root cause, 2026-03-29): dropout repair inserts
    # newly synthesised transients inside missing regions. ArticulationMetric
    # compares transient-shape correlation against the pre-repair signal where
    # those transients are absent by definition, causing false catastrophic P2
    # regressions (worst_goal=artikulation ~0.23) despite musically improved
    # continuity after repair.
    # brillanz excluded: synthesised fill content can have different HF spectral
    # distribution than the surrounding noisy reference → false brillanz drop.
    # authentizitaet excluded (belt+suspenders for flatness proxy): dropout silence
    # has near-zero amplitude → fft_mag ≈ 0 → flatness undefined/high → scores_before
    # may be artificially low; after AudioSR synthesis tonal content increases.
    # The flatness-based proxy handles this correctly in practice but the exclusion
    # prevents edge-cases in very short silence segments where the 2.5-s sample
    # window captures mostly dropout.
    "phase_24": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "artikulation",
        "timbre_authentizitaet",
        "transparenz",
        "tonal_center",
        "groove",  # AudioSR synthesis fills 5981 dropout gaps with new audio patches; formerly silent/corrupted dropout frames had 0 onsets → GrooveMetric onset-DTW autocorr[lag_05] registers onset-density increase as rhythm disruption → false P3 regression. Regression constant at all strengths (stagnation Δ=0.000004, 2026-04-10) → PMGG reduces strength to 0.22 (best_effort), leaving >5000 dropouts unrepaired. Identical mechanism to phase_09/phase_18 groove exclusion.
        "emotionalitaet",  # Dropout silence gaps score high in crest-factor (silence/near-zero amplitude between notes amplifies peak-to-RMS ratio in degraded reference). After AudioSR synthesis, formerly silent patches receive normal signal amplitude → crest-factor ratio drops → false P3 emotionalitaet regression. Identical mechanism to phase_09 (broadband transitions from near-silence) and phase_18 (noise-gate silencing). Regression invariant to strength → confirmed stagnation pattern.
    },  # Dropout repair: synthesised HF content; timbre_authentizitaet: AudioSR synthesis creates new spectral content → MFCC correlation against damaged reference is meaningless; transparenz: dropout silence regions inflate spectral clarity proxy (silence = perfect rolloff) → after AudioSR fill slight noise floor added → proxy drops (false P4); tonal_center: dropout silence has undefined/near-zero chroma → K-S key detection unstable; after AudioSR tonal synthesis K-S locks onto different key estimate → false tonal regression despite musically unchanged pitch centre (stagnation 0.3137 confirmed, 2026-04-08). groove + emotionalitaet: added 2026-04-10 — see inline comments above.
    "phase_28": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
    },  # Surface noise profiling (vinyl): broadband noise events look like transients to ArticulationMetric → after profiling/removal pseudo-transients gone → false P1 regression (catastrophic 0.4222 confirmed, 2026-04-08); natuerlichkeit: broadband spectral denoising (same MFCC-smoothness mechanism as phase_03/phase_29); timbre_authentizitaet: spectral envelope changes when broadband surface noise removed → MFCC-Pearson + centroid-CV shift; authentizitaet: §2.44 Reference-Paradoxon — broadband surface noise smooths log-spectrum valleys → roughness proxy scores HIGH before profiling; after removal true valleys reappear → false P1 cascade (identical mechanism to phase_03/phase_29, aligned with CIG §2.48 exclusions, 2026-04-09)
    # Diffusion inpainting: synthesised content has no transient reference →
    # ArticulationMetric correlation vs pre-inpainting fragment is meaningless.
    # micro_dynamics excluded: inpainting inserts new content with its own
    # envelope that intentionally differs from the surrounding material.
    "phase_55": {
        "artikulation",
        "micro_dynamics",
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "timbre_authentizitaet",
        "tonal_center",
    },  # Diffusion inpainting: synthesised content → identical root-causes as phase_23/phase_24 (AudioSR); MFCC-smoothness vs. damaged reference meaningless; brillanz crest-proxy scores against absent HF pre-synthesis; authentizitaet flatness-proxy reference-mismatch; timbre_authentizitaet MFCC-Pearson/centroid meaningless for synthesised spectral content; tonal_center excluded (§9.7.11 extension, 2026-04-10): CQTdiff+ fills bandwidth-loss gaps with synthesized HF content — pre-inpainting audio (band-limited vinyl ≤8-12 kHz) has near-zero chroma energy in high-register bins; after inpainting, newly filled HF bins shift K-S key-template correlation → false catastrophic P2 regression (Δ=0.8333 confirmed, 06:34 run). Musical key is unchanged; only chroma-bin distribution shifts due to spectral extension
    # Sub-sonic removal: reference LF correlation handles bass preservation check
    "phase_05": {
        "natuerlichkeit",
        "authentizitaet",
    },  # Rumble filter: sub-sonic removal shifts MFCC-smoothness baseline + sub-bass chroma removal causes minor chromagram shift — CIG sync §2.54
    "phase_30": {
        "natuerlichkeit",
        "authentizitaet",
    },  # DC-offset removal: zero-phase highpass slightly shifts ZCR/MFCC-smoothness + minimal chromagram baseline — CIG sync §2.54
    # Broadband denoise: reference HF/LF correlation distinguishes noise from music
    # natuerlichkeit excluded: broadband denoising shifts spectral flatness and
    # ZCR, causing the CREPE-based NatuerlichkeitMetric to report false P1
    # regressions (~0.28) even at near-dry wet-mix.  DSP proxy with §9.7.5
    # reference-aware preservation correctly evaluates naturalness for denoise.
    # artikulation excluded: ArticulationMetric(reference=noisy_tape) measures
    # transient-shape correlation between the denoised output and the noisy input.
    # Denoising IS supposed to reshape transients (ResembleEnhance, OMLSA spectral
    # weighting) — scores_before(reference-free)≈0.67 vs scores_after(ref-based)≈0.13
    # produces a false P2 regression of ~0.54 that drives PMGG into best_effort at
    # strength=0.06 (virtually no denoising applied).  Root cause confirmed in debug
    # logs (2026-03-28): worst_goal=artikulation, before=0.665, after=0.126.
    # brillanz excluded: broadband denoising removes HF noise energy → brillanz DSP
    # proxy drops from ~0.9 (noise-inflated) to ~0.1 (clean).
    # authentizitaet excluded (P1 root cause, v9.10.79): broadband noise raises the
    # spectral noise-floor uniformly → log-spectrum valleys are filled → roughness
    # proxy scores HIGH before denoising.  After denoising the true spectral valleys
    # are revealed → roughness INCREASES → authentizitaet drops ~0.75 → false P1
    # catastrophic cascade (0.8884 regression, phase runs at 6% strength).
    # This is the INTENDED outcome of denoising — not a musical-quality regression.
    # transparenz excluded: HF noise inflates spectral rolloff → DSP proxy scores
    # scores_before too high; after denoising rolloff drops to true musical level
    # → false P4 regression triggering unnecessary retries.
    # timbre_authentizitaet excluded (P2 root cause, 2026-03-30): denoise phases
    # intentionally alter spectral-centroid variance and fine texture while reducing
    # hiss. The PMGG short-window timbre proxy can overreact on tape material and
    # report false catastrophic P2 regressions (~0.09 > 0.08) despite improved
    # perceptual clarity.
    # tonal_center excluded (§9.7.11 extension, v9.10.93): K-S is invariant to ADDITIVE
    # white noise (uniform spectral floor lifts all chroma bins equally → ratios preserved).
    # OMLSA/ResembleEnhance apply FREQUENCY-SELECTIVE suppression (gain G(f) varies per
    # frequency band) which effectively acts as a noise-adaptive EQ → chroma energy
    # distribution shifts → K-S key template correlation changes. Real-run confirmed:
    # catastrophic tonal_center regression Δ=0.1043 on 1930s tape (SNR≈15 dB, 1/f hiss).
    # The musical key did not change; K-S measurement is disturbed by shaped NR.
    # brillanz+transparenz: §9.7.12/13 crest-factor proxies are SNR-robust → kept.
    "phase_03": {
        "natuerlichkeit",
        "artikulation",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
    },  # OMLSA/ResembleEnhance: CREPE-Load-State + transient-shape mismatch + K-S NOT invariant for shaped NR §9.7.11 ext + MFCC-Pearson/Centroid-CV disturbed by spectral-envelope change after NR (v9.10.96 canonical — groove/emotionalitaet entfernt: P3-Quick-Proxy-Robustheit hinreichend)
    # DeepFilterNet HF-removal intentionally reduces HF energy → brillanz drops.
    # artikulation excluded for same reason as phase_03: reference=hissy_tape vs
    # denoised output gives misleadingly low transient-correlation score.
    # authentizitaet excluded: same root-cause as phase_03 — tape hiss smooths the
    # log-spectrum (spectral valleys filled by noise floor); after DeepFilterNet v3 II
    # removes hiss, true valleys reappear → roughness rises → false P1 catastrophic
    # regression (0.5661 observed).
    # transparenz excluded: HF hiss inflates rolloff proxy → DSP rolloff score drops
    # after hiss removal → false P4 regression triggering unnecessary retries.
    # natuerlichkeit excluded: MFCC-smoothness DSP proxy unreliable during HF-removal
    # (same root cause as phase_02 and phase_03).
    # tonal_center excluded (§9.7.11 extension, v9.10.93): DeepFilterNet v3 II is a
    # learned frequency-selective filter — identical mechanism to OMLSA (see phase_03).
    # HF-targeted tape-hiss removal reduces energy in high-register chroma bins
    # (C5-B7) while leaving low-register bins less affected → K-S correlation shifts
    # even though the musical key is unchanged. Real-run stagnation (Δ=0.000311) at
    # strength=0.78 confirms the regression is measurement-driven, not musical.
    "phase_29": {
        "artikulation",
        "authentizitaet",
        "natuerlichkeit",
        "tonal_center",
        "timbre_authentizitaet",
    },  # DeepFilterNet Tape-Hiss — gleiche Root-Causes wie phase_03: MFCC-Pearson + centroid-CV + K-S shaped-NR-instabilität (v9.10.96 canonical — groove/emotionalitaet entfernt)
    # Phases with RADICAL spectral changes where even correlation can't help:
    # phase_04 EQ: redistributes the entire spectrum — brillanz (HF cut/boost)
    # and waerme (mid cut/boost) are intentional outcomes, not regressions.
    # authentizitaet excluded: EQ notch/shelf filters create spectral non-uniformity
    # in the log-domain → roughness proxy rises → false P1 catastrophic regression
    # (0.5503 observed).  EQ-induced spectral shaping IS the intended restoration
    # action — not a musical-quality regression.
    # natuerlichkeit excluded: MFCC-smoothness DSP proxy is directly disturbed by
    # EQ notches (same mechanism as phase_02 comb-filter notches).
    # timbre_authentizitaet excluded: EQ shifts spectral centroid trajectory — the
    # centroid-CV proxy treats any centroid change as timbre degradation, but EQ
    # correction is intentional spectral-shape restoration.
    # phase_16 final_eq: mirrors phase_04 EQ exclusions + tonal_center (see phase_03).
    # Confirmed catastrophic tonal_center regression Δ=0.4708 (P2) in real run.
    # Final mastering EQ with presence boost (3-5 kHz) strengthens upper harmonics of
    # each note → those harmonics land in specific semitone bins → chroma distribution
    # shifts → K-S correlation changes. Not a musical key change.
    "phase_16": {
        "transparenz",
        "brillanz",
        "waerme",
        "authentizitaet",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "tonal_center",
    },  # Final EQ: same spectral redistribution as phase_04 + K-S chroma-shift (§9.7.11 ext)
    "phase_04": {
        "transparenz",
        "brillanz",
        "waerme",
        "authentizitaet",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "artikulation",
    },  # EQ deliberately redistributes spectrum (§9.7.11 K-S: tonal_center not yet observed failing here); artikulation: EQ spectral reshaping modifies frequency distribution of transient attacks → ArticulationMetric transient-shape correlation changes as spectral envelope of attacks shifts (catastrophic P2 regression 0.2515 confirmed, 2026-04-08)
    "phase_06": {
        "timbre_authentizitaet",
    },  # SBR/bandwidth extension adds new HF harmonics: brillanz excluded rationale no longer applies here because §9.7.12 crest-proxy correctly handles synthesis improvement. timbre_authentizitaet: adding sub-10kHz HF harmonics via SBR changes MFCC-Pearson + spectral-centroid-CV (intentional spectral content addition = false P2 by design, confirmed catastrophic regression 0.2185 on timbre_authentizitaet P1 in E2E, 2026-04-08)
    "phase_07": {
        "artikulation",
        "timbre_authentizitaet",
    },  # Harmonic restoration: H2-H4 waveshaping adds new harmonic partials → onset-sharpness proxy saturates at 1.0 pre-phase (mean_peaks/0.01 clips) then drops after harmonic addition (new spectral energy reshapes attack envelope) → false P2 artikulation catastrophic regression (0.2532 observed, 2026-04-02). timbre_authentizitaet: harmonic synthesis intentionally changes MFCC-Pearson + spectral-centroid-CV.
    # Click removal: replaces impulse artifacts with interpolated audio.
    # artikulation excluded: clicks are high-amplitude transients in the damaged
    # signal — ArticulationMetric sees them as "transients". After removal they're
    # absent → transient-shape correlation drops → false P2 regression despite
    # genuine quality improvement. The proxy compares damage-transients vs. repair.
    # natuerlichkeit excluded (P1 root cause, 2026-04-07): click removal applies
    # spectral interpolation over the removed impulse locations. NatuerlichkeitMetric
    # MFCC-smoothness proxy evaluates local short-window coherence; the transition
    # from reconstructed frames to undamaged surroundings creates MFCC trajectory
    # discontinuities that score as "unnatural" relative to the click-bearing
    # reference. Real-run confirmed: worst_goal=natuerlichkeit, regression=0.267 (P1),
    # PMGG dithered to strength=0.17 (virtually no click removal applied).
    # Same root cause as phase_02 comb-notch → CREPE/MFCC-smoothness mismatch.
    "phase_01": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
        "tonal_center",  # §2.44 Reference-Paradox: 22965 click events are broadband impulses; spectral interpolation at scale alters chromagram → K-S key-template correlation drops despite pitch structure being preserved/improved. Identical mechanism to phase_12/phase_49/phase_58. CIG P2 rollback confirmed (rollbacks=1, strength→0.07, 2026-04-10).
        "groove",  # Clicks appear as spurious onset events in GrooveMetric onset-based DTW proxy. High-severity click removal reduces onset count/density → autocorr[lag_05] DTW changes → false P3 regression. Identical mechanism to phase_09 groove exclusion (confirmed: 22965 clicks on gen=7 vinyl).
    },  # Click removal: impulse transients → ArticulationMetric false P2; spectral interpolation → NatuerlichkeitMetric MFCC-smoothness false P1; timbre_authentizitaet: MFCC-Pearson shift at repaired click locations; authentizitaet: §2.44 reference-paradox roughness shift vs. click-bearing reference. tonal_center + groove: see inline comments above.
    # Click/pop removal: identical mechanism to phase_01 (different algorithm,
    # same false-regression root cause for all excluded proxies).
    "phase_27": {
        "artikulation",
        "natuerlichkeit",
        "timbre_authentizitaet",
        "authentizitaet",
        "tonal_center",  # Same mechanism as phase_01: click/pop interpolation at scale alters K-S chroma correlation → false P2 CIG rollback. Confirmed: phase_27 rollbacks=1 on same run (2026-04-10).
        "groove",  # Same mechanism as phase_01 + phase_09: impulse removal changes onset density → onset-DTW false P3 regression. (phase_27 already handled as P99 tolerance, explicit exclusion prevents CIG-level rollback cascade.)
    },  # Click/pop removal: same proxy limitations as phase_01 — tonal_center (K-S false P2 via chroma shift) + groove (onset-DTW false P3 via impulse removal) both added 2026-04-10.
    # BANQUET blind denoising: full-band neural diffusion-based crackle/noise removal.
    # natuerlichkeit excluded: BANQUET modifies the full spectral envelope (same root
    # cause as phase_03/phase_29 — MFCC-smoothness proxy disturbed by denoising).
    # groove excluded (P1 root cause, 2026-04-07): BANQUET removes crackle events
    # that appear as periodic impulsive onsets. GrooveMetric onset-based DTW proxy
    # registers the change in LF onset density as rhythmic disruption. Real-run
    # confirmed: worst_goal=groove, regression=0.291 (P1), stagnation across all
    # retries, strength=0.15 (virtually no crackle removal). Same mechanism as
    # phase_02 groove exclusion — LF spectral energy changes fool onset-DTW proxy.
    # authentizitaet excluded: crackle adds broadband noise floor → log-spectrum
    # valleys filled high before BANQUET; after processing, valleys reappear →
    # roughness rises → false P1 cascade. Identical to phase_03/phase_29.
    # timbre_authentizitaet excluded: MFCC-Pearson/centroid-CV disturbed by
    # full-band spectral envelope modification (same as phase_29).
    "phase_09": {
        "natuerlichkeit",
        "groove",
        "authentizitaet",
        "timbre_authentizitaet",
        "artikulation",  # crackle removal changes onset energy envelope (same mechanism as phase_01/phase_27) — CIG sync §2.54
        "tonal_center",  # broadband crackle inflates K-S chroma bins; after removal chroma estimate shifts vs. crackle-bearing checkpoint — CIG sync §2.54
    },  # BANQUET blind denoising: full-band spectral mod → natuerlichkeit MFCC-smoothness false P1 (0.291, 2026-04-07); groove onset-DTW false P1 (0.291); authentizitaet log-spectrum valley mechanism; timbre MFCC-Pearson/centroid-CV
    # Spectral repair (STFT inpainting via bin interpolation):
    # Replaces isolated spike-bins with linear interpolation from ±2 neighbours.
    # This is DSP-only (no ML synthesis), so natuerlichkeit/authentizitaet proxies
    # are unaffected (no spectral envelope synthesis). Only artikulation is at risk:
    # isolated spike-bins can appear as transient onsets to the proxy; after repair
    # those spikes are smoothed → false P2 regression for heavily corrupted sections.
    "phase_50": {
        "artikulation"
    },  # STFT spectral inpainting: isolated spike-bins appear as transients → smoothing causes false ArticulationMetric P2 regression
    # Harmonic exciter: synthesises H2–H4 harmonics to enhance presence/air.
    # timbre_authentizitaet excluded: adding harmonics intentionally changes
    # MFCC-Pearson (the timbre IS changing) and spectral-centroid-CV
    # (HF partial energy increases) → false P2 regression vs. pre-exciter reference.
    "phase_21": {
        "timbre_authentizitaet"
    },  # Harmonic exciter: H2-H4 synthesis intentionally changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Tape saturation: tanh-waveshaping (soft saturation) + harmonic series modeling.
    # timbre_authentizitaet: harmonics added intentionally → MFCC-Pearson changes.
    # emotionalitaet: tanh reduces peak amplitude relative to RMS (peak compression)
    # → crest-factor ratio drops → false P3 regression despite intended enhancement.
    "phase_22": {
        "timbre_authentizitaet",
        "emotionalitaet",
    },  # Tape saturation: tanh waveshaping compresses peaks → crest-factor drops (false P3); harmonic synthesis changes MFCC-Pearson (false P2)
    # Bass enhancement: low-shelf EQ + sub-harmonic synthesis + soft saturation.
    # tonal_center excluded: sub-harmonic synthesis creates tones an octave below
    # fundamentals — K-S chroma template correlation changes because the pitch-class
    # weight distribution shifts (added energy at sub-octave positions).
    # timbre_authentizitaet: LF energy addition shifts lower-order MFCC coefficients.
    # waerme excluded: bass boost directly increases energy in the 200–800 Hz warmth
    # band → warmth ratio E(200-800)/E(800-3000) changes → false P4 regression.
    # emotionalitaet: LF boost raises RMS significantly relative to peaks → crest
    # drops → false P3 regression (same mechanism as phase_22 tanh saturation).
    "phase_37": {
        "timbre_authentizitaet",
        "tonal_center",
        "waerme",
        "emotionalitaet",
    },  # Bass enhancement: sub-harmonic synthesis → K-S chroma shift + MFCC change; LF energy boost → warmth-ratio + crest-factor false regressions
    # Presence/mid-range clarity EQ (1–4 kHz dynamic boost + Bell EQ).
    # timbre_authentizitaet: boosting 1-4 kHz changes MFCC c1-c3 (dominant spectral
    # range) + centroid-CV directly → false P2 regression vs. pre-boost reference.
    # waerme excluded: presence boost raises energy in 800–3000 Hz band → warmth
    # ratio E(200-800)/E(800-3000) changes → false P4 regression.
    "phase_38": {
        "timbre_authentizitaet",
        "waerme",
    },  # Presence EQ: 1-4 kHz boost changes MFCC c1-c3 + warmth ratio E(200-800)/E(800-3000) → false P2/P4 regressions
    # Air band enhancement: shelving EQ + harmonic synthesis for 12–20 kHz.
    # timbre_authentizitaet: HF centroid-CV increases when 12-20 kHz is boosted
    # (centroid shifts upward) → MFCC higher-order coefficients change → false P2.
    "phase_39": {
        "timbre_authentizitaet"
    },  # Air band HF enhancement: 12-20 kHz boost shifts centroid-CV + MFCC higher-order coefficients → false P2 timbre regression
    # De-esser (DSP primary + MP-SENet ML refinement): targets sibilant 4–8 kHz.
    # artikulation excluded: /s/, /f/ fricatives ARE transients — the de-esser
    # specifically attenuates their peaks → ArticulationMetric registers this as
    # transient-shape regression vs. pre-processing reference despite it being repair.
    # timbre_authentizitaet: 4-8 kHz spectral reduction changes centroid-CV + MFCC
    # higher-order coefficients → false P2 regression.
    # emotionalitaet: de-essing reduces crest specifically at sibilant peaks
    # → crest-factor ratio drops → false P3 regression.
    "phase_43": {
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
    },  # De-esser: 4-8 kHz sibilant attenuation → artikulation (fricative transients attenuated) + timbre (centroid-CV + MFCC) + emotionalitaet (crest-factor drop) false regressions
    # Guitar enhancement: spectral shaping for guitar timbre (distortion, presence).
    # timbre_authentizitaet: guitar-specific spectral shaping intentionally changes
    # the MFCC-Pearson + centroid-CV profile → false P2 vs. pre-enhancement.
    "phase_44": {
        "timbre_authentizitaet"
    },  # Guitar enhancement: spectral shaping changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Brass enhancement: HP-filtered formant enhancement + spectral shaping.
    # timbre_authentizitaet: brass formant enhancement changes spectral envelope
    # deliberately → MFCC-Pearson/centroid-CV proxy registers as P2 regression.
    "phase_45": {
        "timbre_authentizitaet"
    },  # Brass enhancement: formant spectral shaping changes MFCC-Pearson + centroid-CV → false P2 timbre regression
    # Spectral tilt: global broadband EQ tilt (boost LF / cut HF or vice versa).
    # timbre_authentizitaet: global spectral tilt directly changes lower-order MFCC
    # c1-c3 (dominant LF/MF energy) + spectral centroid location → false P2.
    # waerme excluded: spectral tilt shifts E(200-800)/E(800-3000) warmth ratio
    # depending on tilt direction → false P4 regression.
    # emotionalitaet: broad energy redistribution changes RMS vs. peak balance
    # → crest-factor ratio shifts → false P3 regression.
    # phase_53_semantic_audio: METADATA-only phase — audio is returned UNCHANGED.
    # process() computes BPM, key, genre-hint and writes results to PhaseResult.metadata.
    # No spectral or dynamics modification → scores_before == scores_after for all 14 goals
    # → no PMGG regression possible → exclusions are structurally unnecessary.
    "phase_53": set(),  # SemanticAudioPhase is metadata-only (audio unchanged) → no goal can regress
    # Spectral Band Gap Repair (HEAD_WEAR defect): harmonics interpolated via
    # Fletcher partial model + NMF-β refinement.
    # Mechanistically identical to phase_23 (AudioSR spectral inpainting) for all
    # synthesis-reference-mismatch root causes.
    # natuerlichkeit: synthesised partial harmonics differ from pre-repair damaged
    # reference → MFCC smoothness proxy unreliable.
    # brillanz: synthesised HF band energy distribution may differ from the HF gap
    # reference → crest proxy scores against a damaged (near-zero HF) baseline.
    # authentizitaet: spectral gap has near-zero amplitude → flatness undefined;
    # after repair, tonal content increases → reference-mismatch-driven transition.
    # timbre_authentizitaet: MFCC-Pearson meaningless against damaged gap reference.
    "phase_56": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "timbre_authentizitaet",
    },  # HEAD_WEAR band gap repair: harmonic interpolation synthesis → identical reference-mismatch root causes as phase_23/phase_55
    # Print-through reduction (bidirectional LMS, reel_tape only):
    # Removes magnetic pre/post-echo from tape print-through artifact.
    # Mechanistically identical to phase_49 (Advanced Dereverb) for:
    # authentizitaet: echo tail spread energy across spectrum → smooths log-spectrum
    # valleys; after removal, valleys reappear → roughness rises → false P1 cascade.
    # emotionalitaet: echo tail adds residual energy to quiet segments between musical
    # events → scores_before crest-factor elevated; after removal, quiet segments
    # become true silence → crest-factor ratio shifts → false P3 regression.
    "phase_57_print_through_reduction": {
        "authentizitaet",
        "emotionalitaet",
    },  # Print-through reduction: echo-tail/pre-echo removal → authentizitaet (log-spectrum valley mechanism) + emotionalitaet (crest-factor in silence segments) false regressions — identical to phase_49
    # Lyrics-Guided Enhancement (§2.36 PFLICHT): phoneme-aligned DSP per segment class.
    # timbre_authentizitaet excluded: fricative ramp-gain (4-8 kHz), vowel formant
    # shelving (LPC Burg Ord.30-40), plosive burst boost (100-350 Hz) all intentionally
    # change the spectral envelope → MFCC-Pearson + centroid-CV register as P2 regression
    # vs. the pre-enhancement reference where these phoneme targets were under-enhanced.
    # artikulation excluded: plosive TransientShapeGuard bypasses onset-window (gain=1.0)
    # but burst boost (×1.40) and aspiration boost (3-8 kHz ×1.20) modify the plosive
    # shape → ArticulationMetric transient-shape correlation registers change vs. baseline.
    # emotionalitaet excluded: fricative high-frequency ramp-gain raises HF energy
    # selectively at sibilant positions → local crest-factor ratio shifts → false P3
    # regression despite intended timbral improvement.
    "phase_58_lyrics_guided_enhancement": {
        "tonal_center",  # §Y5: fricative ramp-gain (4–8 kHz) shifts HF energy profile → K-S key-label flip (SNR-sensitive K-S already excluded from shaped-NR phases)
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
    },  # LGE §2.36: phoneme-specific spectral ops (fricative ramp, plosive burst, formant shelving) → MFCC-Pearson + transient-shape + local crest false regressions
    # M/S dynamics: compresses BOTH Mid AND Side channels independently per 4 bands.
    # Mid compression (ratio 2.0–3.0) directly affects the mono sum (L+R)/2 = Mid.
    # micro_dynamics excluded: multi-band Mid/Side compression intentionally reshapes
    # envelope — same mechanism as phase_10 (multiband compression).
    # groove excluded: Mid compression changes inter-beat RMS periodicity in the
    # mono sum → autocorr[lag_05] disrupted (same mechanism as phase_17).
    # emotionalitaet excluded: Mid compression reduces crest-factor in mono sum
    # → false P3 regression (same mechanism as phase_17/phase_10).
    # timbre_authentizitaet excluded: multiband Mid+Side spectral shaping with
    # different gain-reduction per band alters the spectral envelope of the
    # mono sum → MFCC c1-c3 + centroid-CV register false P2 regression.
    "phase_34": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
        "timbre_authentizitaet",
    },  # M/S dynamics: Mid-channel compression (ratio 2-3) affects mono sum → groove+emotionalitaet+micro_dynamics (same as phase_10) + timbre_authentizitaet (per-band spectral shaping)
    # Loudness normalization (ITU-R BS.1770-4 / EBU R128):
    # Pure LUFS gain scaling is scale-invariant for ALL ratio-based proxies
    # (groove autocorr, crest-factor, tonal K-S, timbre MFCC-Pearson are unchanged
    # by global gain). BUT includes multi-band loudness shaping (frequency-dependent
    # gain adjustment) which is essentially a spectral EQ → timbre risk.
    # timbre_authentizitaet excluded: multi-band frequency-dependent loudness shaping
    # shifts spectral envelope → MFCC c1-c3 + centroid-CV register false P2.
    "phase_40": {
        "timbre_authentizitaet"
    },  # Loudness normalization: pure LUFS gain is scale-invariant; multi-band frequency shaping changes spectral envelope → timbre_authentizitaet false P2 regression
    # + transient enhancement. Three false-regression root causes on degraded material:
    # micro_dynamics excluded: transient enhancement intentionally reshapes the LUFS
    # micro-profile — that change IS the intended TDP effect, not a regression.
    # artikulation excluded: transient-shaping BY DEFINITION changes transient shapes;
    # comparing after-TDP transients against before-TDP baseline is meaningless since
    # TDP is supposed to alter transient characteristics.
    "phase_08": {"micro_dynamics", "artikulation"},  # TDP transient preservation (§9.7.11 K-S: tonal_center resolved)
    # Dynamics-modifying phases: intentional temporal envelope changes
    # phase_18 noise gate: removes background noise (incl. HF noise) between
    # musical events → brillanz drops from noise-inflated value → false regression.
    # authentizitaet excluded: same log-spectrum valley mechanism as phase_03 —
    # noise in silence gaps smooths log-spectrum; after gating, silence is silent
    # (zeros) → the FFT sample captures more musical-content frames → valleys
    # become visible → roughness rises → false P1 regression.
    # transparenz excluded: HF noise in silence sections inflates rolloff proxy;
    # after gating those sections, average rolloff drops → false P4 regression.
    # emotionalitaet excluded: noise gate deliberately changes crest factor by
    # silencing quiet sections between musical phrases → crest_score shift is
    # the intended effect, not a dynamics regression.
    # groove excluded (P3 root cause, 2026-03-30): noise gate silences inter-beat
    # quiet sections → rms_env becomes discontinuous at gate-on/off boundaries
    # → gate-zero segments inflate autocorr[0] variance → normalized autocorr[lag_05]
    # drops even at minimal gate strength (Δ=0.002226 stagnation, regression 0.1721
    # observed; best_effort at 0.19 strength = noise gate effectively disabled).
    # Groove proxy measures rhythmic periodicity; VAD-gated silence IS the intended
    # noise-gate effect and cannot be decoupled from the rhythm signal in 2.5 s windows.
    "phase_18": {
        "micro_dynamics",
        "authentizitaet",
        "emotionalitaet",
        "groove",
        "timbre_authentizitaet",  # noise gate inserts silence between phrases → spectral centroid/MFCC changes vs. continuous-noise reference — CIG sync §2.54
    },  # Noise gate (§9.7.11 K-S: tonal_center resolved — K-S key-detection is SNR-invariant; §9.7.12/13: brillanz+transparenz crest proxies SNR-robust → removed)
    "phase_26": {
        "micro_dynamics",
        "artikulation",
        "groove",
        "emotionalitaet",
    },  # Dynamic expansion: expander opens transient/decay gap → RMS-env autocorr[lag_05] disrupted + crest-factor shift → false P3 regressions (same mechanisms as phase_18 noise gate)
    "phase_36": {
        "micro_dynamics",
        "artikulation",
        "groove",
        "emotionalitaet",
    },  # Transient shaper: transient-boost raises peaks vs. RMS floor → crest-factor ratio shifts + RMS-peak timing changes → false P3 regressions (same mechanisms as phase_08 + phase_18)
    # Multiband parallel compression: attack/release envelopes directly modify
    # inter-beat RMS envelope periodicity → autocorr[lag_05] changes → false P3 groove
    # regression (identical mechanism to phase_17 multiband mastering compressor).
    # Crest-factor reduction via compression → false P3 emotionalitaet regression.
    # micro_dynamics excluded: by design compression changes envelope dynamics.
    "phase_10": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # Multiband parallel compression: envelope modification → groove autocorr[lag_05] disrupted + crest-factor uniformly reduced → false P3 regressions (identical mechanism to phase_17)
    # 4-band limiting: brick-wall limiter is an extreme compressor with ∞:1 ratio.
    # Peaks clipped → crest-factor drops → false P3 emotionalitaet regression.
    # Inter-beat periodicity changes when loud transient peaks are attenuated
    # differently per band → groove autocorr[lag_05] disrupted.
    "phase_11": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # Multi-band limiting: extreme compression (∞:1) → crest-factor drops + RMS-envelope periodicity changes → false P3 regressions (identical mechanism to phase_17 + phase_10)
    # TruePeak limiter: clamps sample peaks above a threshold via 4× oversampling.
    # Aggressive application (near 0 dBFS ceiling) extensively clips transient peaks
    # → crest-factor drops significantly → false P3 emotionalitaet regression.
    # Loud transient beats attenuated → inter-beat amplitude contrast reduced →
    # autocorr[lag_05] misreads periodic beat pattern → false P3 groove regression.
    "phase_47": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # TruePeak limiter: peak-clamping reduces crest-factor + inter-beat peak contrast → false P3 regressions (same mechanism as phase_11)
    # 4-band independent compression with upward/downward compander:
    # mechanistically identical to phase_10 (multi-band parallel compression) and
    # phase_17 (mastering compressor). Envelope modification per band.
    "phase_35": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # 4-band multiband compression: independent band compander → RMS envelope periodicity disrupted + crest-factor reduced → false P3 regressions (identical mechanism to phase_10/phase_17)
    # Psychoacoustic-aware compression (genre-adaptive, masking-aware):
    # applies RMS-envelope-adaptive threshold and alters dynamics per masked region.
    # Despite perceptual optimisation the proxy mechanisms are identical:
    # crest-factor reduction + inter-beat envelope change → false P3 regressions.
    "phase_54": {
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # Psychoacoustic compression: genre-adaptive envelope modification → crest-factor drops + groove autocorr[lag_05] disrupted → false P3 regressions (identical mechanism to phase_17 + phase_35)
    # Mastering: intentional dynamics compression + spectral shaping.
    # tonal_center excluded (§9.7.11 ext, v9.10.93): multiband compression +
    # presence/air EQ redistribute chroma energy → K-S detects apparent key shift.
    # artikulation excluded: multiband compression is designed to change attack
    # envelopes (faster attack → softer onset, slower release → sustain boost);
    # ArticulationMetric's transient-shape correlation measures this change as
    # regression, but the mastering effect IS the intended outcome. Real-run:
    # catastrophic P2 regression Δ=0.2092 (worst_goal=artikulation).
    "phase_17": {
        "micro_dynamics",
        "natuerlichkeit",
        "tonal_center",
        "artikulation",
        "groove",
        "emotionalitaet",
    },  # groove: multiband compression changes inter-beat RMS envelope periodicity → false P3 regression (Δ=0.0251 observed); emotionalitaet: compression reduces crest-factor uniformly → false P3 regression (identical mechanism to phase_18) — confirmed 2026-03-31
    # Vocal enhancement: Stages 2-6 intentionally alter spectral shape and dynamics;
    # natuerlichkeit/timbre proxies are unreliable for deliberate vocal-presence boosts.
    "phase_19": {
        "natuerlichkeit",
        "timbre_authentizitaet",
        "micro_dynamics",
        "groove",
        "emotionalitaet",
    },  # Vocal enhancement: Stage 6 micro-compression shifts crest-factor + Stage 2 breath-gating changes inter-beat RMS periodicity → false P3 regressions (same mechanisms as phase_17/phase_18)
    # BSRoFormer vocal stem separation + vocal enhancement with micro-compression
    # (syllable-level, ratio 1.8–2.5) + envelope shaping + FormantSystem enhancement.
    # Mechanistically similar to phase_19 (compression/breathing) + phase_23 (synthesis).
    # natuerlichkeit excluded: BSRoFormer stem synthesis + formant enhancement alter
    # MFCC smoothness proxy on the separated signal vs. mixed reference.
    # authentizitaet excluded: stem separation changes spectral flatness (separation
    # exposes vocal harmonics previously masked by instrumentation → flatness shifts).
    # timbre_authentizitaet excluded: formant enhancement + spectral envelope change
    # from stem isolation shifts MFCC-Pearson + centroid-CV proxy.
    # groove excluded: syllable-level micro-compression modifies inter-syllable RMS
    # periodicity → autocorr[lag_05] false regression (same mechanism as phase_19).
    # emotionalitaet excluded: micro-compression reduces crest-factor at syllable level
    # → false P3 regression (identical mechanism to phase_17/phase_19).
    "phase_42": {
        "natuerlichkeit",
        "authentizitaet",
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
        "artikulation",
    },  # BSRoFormer vocal enhancement: stem separation + micro-compression + formant shaping → false regressions via synthesis/crest/MFCC mechanisms (identical to phase_19 + phase_23); artikulation: BSRoFormer stem resynthesis reshapes transient content → ArticulationMetric transient-shape correlation vs. original meaningless for ML-synthesized output (catastrophic P2 regression 0.2043 confirmed, 2026-04-08)
    # Drums/percussion enhancement: transient shaping (attack/sustain) + DrumsEnhancementSystem
    # which includes compression (Dbx-style). Beat-synchronous transient shaping alters
    # the inter-beat RMS contrast → groove autocorr[lag_05] disrupted.
    # Drum spectral enhancement (punch/snap synthesis) changes timbre MFCC proxy.
    # Compression on beats reduces crest-factor at beat positions → false P3 emotionalitaet.
    "phase_51": {
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
    },  # Drums enhancement: transient shaping + compression → inter-beat RMS changes + crest-factor drops → false P3 regressions; timbre_authentizitaet: punch/snap synthesis changes spectral envelope
    # Piano restoration: dynamic range restoration via material-adaptive expansion
    # (velocity curve optimization, expansion ratios 1.2–1.3, compression artifact removal).
    # Expansion is upward dynamic expansion (inverse compression) — increases crest-factor
    # and changes note-to-note RMS envelope periodicity → false P3 groove regression
    # (same root cause as phase_26 dynamic expansion). String resonance modeling and
    # spectral enhancement change timbre MFCC proxy.
    "phase_52": {
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
    },  # Piano restoration: dynamic expansion (1.2–1.3) + string resonance synthesis → inter-beat RMS periodicity changes + crest-factor shifts → false P3 regressions; timbre_authentizitaet: string resonance modeling changes MFCC-spectral-envelope proxy
    # Dereverb: removes room impulse response; reverb contributes diffuse HF energy
    # and room resonances (warmth). After dereverb brillanz and waerme both drop
    # legitimately — these are intentional improvements, not regressions.
    # authentizitaet excluded: reverb tail spreads energy across spectrum → smooths
    # log-spectrum (fills valleys with diffuse energy) → scores_before artificially
    # high; after dereverb true spectral valleys reappear → roughness rises →
    # false P1 catastrophic regression (0.5502 observed).
    # transparenz excluded: reverb-contributed diffuse HF energy inflates spectral
    # rolloff → scores_before elevated; after dereverb rolloff drops → false P4
    # regression triggering unnecessary strength reductions.
    "phase_49": {
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "natuerlichkeit",
    },  # Advanced dereverb: tonal_center excluded (§9.7.11 ext, 2026-04-10): WPE/spectral-subtraction removes reverb energy from high-register chroma bins unevenly → K-S correlation shifts; catastrophic P2 regression 0.4667/0.5530 confirmed in real run. timbre_authentizitaet: reverb tail shifts MFCC-Pearson at all cepstral coefficients → removal changes spectral-centroid-CV (identical mechanism to phase_03/phase_29). artikulation: reverb tail blurs transient attacks → pre-removal ArticulationMetric(reverberant reference) vs de-reverbed output shows false correlation drop. natuerlichkeit: spectral-subtraction dereverb applies frequency-selective gain G(f) → MFCC-smoothness instability
    # Reverb reduction (SGMSE+ primary / WPE-DSP fallback): mechanistically identical
    # to phase_49 Advanced Dereverb — both remove room impulse response energy.
    # brillanz excluded: reverb tail contributes diffuse HF energy across the spectrum
    # → brillanz proxy scores HIGH before removal (noise-inflated); after SGMSE+ the
    # dry direct signal no longer carries that diffuse HF → false brillanz drop.
    # waerme excluded: reverb mid-band tail (early reflections 200–2000 Hz) lifts the
    # waerme proxy before processing; removal exposes the dry mid energy → false drop.
    # authentizitaet excluded: reverb smooths log-spectrum valleys (same mechanism as
    # broadband noise in phase_03); after removal true valleys reappear → flatness-proxy
    # perceives this as reduced tonality → false P1 cascade (same 0.55 regression
    # observed in production for phase_49 and structurally identical for phase_20).
    # transparenz excluded: reverb contributes diffuse HF inflating 75%-rolloff proxy;
    # after removal rolloff drops legitimately → false P4 regression.
    # natuerlichkeit excluded: SGMSE+ spectral deconvolution can introduce slight
    # harmonic smearing on ambiguous reverb vs. body resonance segments → MFCC
    # smoothness proxy reacts on the 5-s short window even when result is perceptually
    # correct. Same MFCC-smoothness instability as phase_02/phase_03 root causes.
    "phase_20": {
        "authentizitaet",
        "natuerlichkeit",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
    },  # SGMSE+ reverb reduction: tonal_center excluded (§9.7.11 ext, 2026-04-10): SGMSE+ U-Net applies learned frequency-selective deconvolution → high-register chroma bins attenuated unevenly → K-S correlation shifts; P2 catastrophic regression 0.5530 confirmed. timbre_authentizitaet + artikulation: identical mechanism to phase_49 (reverb tail MFCC/transient-shape mismatch vs dry reference)
    # Spectral inpainting (AudioSR gap-fill): synthesises new frequency content for
    # spectral holes (codec artefacts, digital clipping reconstruction, missing HF).
    # Identical synthesised-content mechanism to phase_24 (AudioSR dropout repair).
    # natuerlichkeit excluded: gap-fill synthesis produces content absent from the
    # noisy/damaged reference; MFCC-smoothness proxy on the synthesised region is
    # unreliable vs. the pre-repair (damaged) reference.
    # brillanz excluded: synthesised HF fill may have different spectral distribution
    # than the surrounding damaged reference → false brillanz regression against a
    # damaged-signal baseline.
    # authentizitaet excluded: spectral gaps have near-zero amplitude → fft_mag ≈ 0
    # → flatness undefined; after AudioSR synthesis tonal content increases →
    # authentizitaet score transition is reference-mismatch-driven, not a regression.
    # artikulation excluded: inpainting inserts new spectral content in regions where
    # (by definition) the reference has damaged/missing content → transient-shape
    # correlation against the pre-inpainting fragment is meaningless.
    "phase_23": {
        "natuerlichkeit",
        "brillanz",
        "authentizitaet",
        "artikulation",
        "timbre_authentizitaet",
    },  # AudioSR spectral inpainting / gap-fill; timbre_authentizitaet: synthesised fill content has different spectral envelope than damaged reference
    # Wow/flutter correction: time-stretching/resampling shifts chroma energy
    # distribution → K-S key correlation changes despite unchanged musical key.
    # Regression variance 0.067→0.833 across runs of same audio PROVES this is
    # pure proxy noise, not a real quality issue.
    # timbre_authentizitaet: speed correction alters spectral centroid trajectory.
    "phase_12": {
        "tonal_center",
        "timbre_authentizitaet",
        "authentizitaet",
        "natuerlichkeit",
        "artikulation",
    },  # Wow/flutter fix: K-S volatile after pitch/speed correction + centroid-CV disturbed; reference-paradox affects authentizitaet/natuerlichkeit/artikulation proxies.
    # Speed/pitch correction: global time-stretch + resampling — mechanistically
    # identical to phase_12 for all proxy false-regression root causes.
    # tonal_center excluded: global pitch-shift moves ALL chroma bins proportionally
    # → K-S key template correlation changes even when the musical key interpretation
    # is correct (only absolute frequency changes, not musical class).
    # timbre_authentizitaet excluded: pitch-shift alters spectral centroid directly
    # (f0 × ratio → centroid × ratio) → centroid-CV proxy registers as timbre change.
    # groove excluded: time-stretch changes absolute frame timing of RMS peaks;
    # autocorr[lag_05] measures periodicity in absolute sample-time units, not
    # musical-beat units → tempo-corrected audio appears less periodic to the proxy.
    # emotionalitaet excluded: global speed change uniformly scales all envelope
    # segments → crest-factor ratio shifts because loud/quiet segment durations change
    # → false P3 regression despite identical musical dynamics after correction.
    # artikulation excluded: PSOLA/time-stretch modifies transient shapes by
    # design — that IS the correction. TransientShapeCorrelation vs. pre-correction
    # reference is meaningless (same root cause as phase_08 TDP).
    "phase_31": {
        "tonal_center",
        "timbre_authentizitaet",
        "groove",
        "emotionalitaet",
        "artikulation",
        "natuerlichkeit",  # speed correction shifts tempo → MFCC-smoothness temporal consistency changes vs. speed-deviated reference — CIG sync §2.54
        "authentizitaet",  # speed/pitch correction fundamentally changes chromagram vs. pitch-deviated reference (carrier-chain inversion §2.46 — mirror of phase_12) — CIG sync §2.54
    },  # Speed/pitch correction: global time-stretch identical mechanisms to phase_12 + emotionalitaet/artikulation via envelope/transient change (2026-03-31)
    # Stereo enhancement (multi-band M/S + Haas cross-feed delays + Blumlein shuffling):
    # The Haas effect simulation (5–35 ms inter-channel delays) adds delayed copies of
    # one channel to the other → cross-feed creates comb-filtering artifacts in the
    # mono sum (L+R)/2 → spectral balance changes → MFCC-Pearson + centroid-CV
    # register change vs. pre-enhancement reference → false P2 timbre regression.
    # Transient-preserving Side enhancement (attack/decay-aware lateral widening)
    # additionally modifies the L/R spectral content independently of the mono sum.
    "phase_13": {
        "timbre_authentizitaet"
    },  # Stereo enhancement: Haas cross-feed delays (5–35 ms) create comb-filter in mono sum + transient-aware Side shaping → MFCC-Pearson + centroid-CV change vs reference → false P2 timbre regression
    # Phase correction (multi-band all-pass / fractional-delay alignment):
    # Phase-only operations preserve per-channel spectral magnitude, but correcting
    # inter-channel misalignment changes the constructive/destructive interference
    # pattern in the mono sum (L+R)/2. Before correction: channel misalignment causes
    # spectral cancellation notches in the M-channel. After correction: channels align
    # → cancellation resolved → M-channel spectral valleys fill in → MFCC-Pearson vs.
    # the misaligned reference detects a spectral-shape change → false P2 regression.
    "phase_14": {
        "timbre_authentizitaet",
        "authentizitaet",  # phase correction resolves mono-sum cancellation notches → stereo correlation fingerprint changes vs. phase-misaligned reference — CIG sync §2.54
    },  # Phase correction: all-pass/fractional-delay alignment resolves mono-sum cancellation notches → spectral shape changes vs. misaligned reference → MFCC-Pearson + centroid-CV false P2 timbre regression
    # Stereo balance correction: re-balancing L/R channel levels intentionally changes stereo field.
    # authentizitaet: stereo correlation fingerprint changes vs. imbalanced carrier reference (§2.44 Carrier-Chain-Inversion).
    # timbre_authentizitaet: per-channel spectral balance change shifts MFCC of stereo mix vs. imbalanced reference.
    "phase_15": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Stereo balance correction: L/R re-balancing changes stereo-field fingerprint → authentizitaet + MFCC-Pearson false P2 regression vs. imbalanced reference (§2.44)
    # Azimuth correction (tape head misalignment: fractional delay + HF restoration):
    # HF restoration via spectral prediction adds energy in the 5–20 kHz range —
    # mechanistically identical to phase_39 (air band HF enhancement). MFCC
    # higher-order coefficients (c7–c13 HF-sensitive) change when true HF content
    # is restored from azimuth-caused HF dropout; spectral centroid shifts upward.
    # The proxy compares against the reference measured BEFORE azimuth correction
    # (with reduced HF due to destructive interference) → false P2 timbre regression
    # despite genuine quality improvement.
    "phase_25": {
        "timbre_authentizitaet",
        "authentizitaet",  # azimuth correction changes stereo HF balance vs. mis-aligned reference → chromagram fingerprint shifts (§2.44 carrier-chain inversion) — CIG sync §2.54
    },  # Azimuth correction: fractional-delay + HF spectral restoration changes MFCC higher-order coefficients + centroid-CV vs. azimuth-degraded reference → false P2 timbre regression (identical mechanism to phase_39 air band)
    # Mono-to-stereo (Lauridsen pseudo-stereo + HF harmonics + Schroeder decorrelation):
    # Schroeder reverb structures and comb-filter frequency-dependent phase shifts used
    # for decorrelation change the mono sum (L+R)/2 through cross-feed comb patterns
    # → MFCC-Pearson + centroid-CV shift vs. original mono reference. Additionally,
    # optional HF harmonics ("air", tape warmth, vinyl sheen) add spectral energy in
    # the MFCC-sensitive high-register → false P2 timbre regression vs. mono reference.
    "phase_32": {
        "timbre_authentizitaet"
    },  # Mono-to-stereo: Schroeder decorrelation (comb-filter in mono sum) + HF harmonic synthesis → MFCC-Pearson + centroid-CV change vs. mono reference → false P2 timbre regression
    # Stereo width limiter (M/S soft-knee Side-channel compression):
    # Scales the Side channel (S = (L−R)/2) with a frequency-dependent gain ≤ 1.
    # Reconstructed: L = M + S×gain, R = M − S×gain. The mono sum M = (L+R)/2 is
    # unaffected, but individual L/R channels change spectral content proportionally
    # to the applied Side gain. If the MFCC proxy runs on the LEFT channel (or the
    # first channel of the np.ndarray), spectral content of L shifts vs. the
    # pre-limiting reference → MFCC-Pearson + centroid-CV register change → false P2.
    "phase_33": {
        "timbre_authentizitaet"
    },  # Stereo width limiter: M/S Side soft-knee compression changes L/R channel spectral distribution (mono sum M preserved) → MFCC-Pearson + centroid-CV vs. wide-stereo reference → false P2 timbre regression
    # Output format optimization (multi-band loudness shaping + TruePeak + SRC + dither):
    # Pure LUFS gain and lossless SRC are scale-invariant for all ratio-based proxies.
    # However, format-specific multi-band frequency-dependent loudness shaping shifts
    # the spectral envelope → MFCC c1–c3 + centroid-CV register change — identical
    # root cause to phase_40 (loudness normalization). TruePeak limiting at −1 dBTP
    # is normally very light (prevents intersample overs only) → micro_dynamics, groove,
    # emotionalitaet not excluded unless confirmed in production logs.
    "phase_41": {
        "artikulation",
        "timbre_authentizitaet",
    },  # Output format optimization: multi-band loudness shaping + TruePeak limiting shifts spectral envelope → MFCC c1-c3 + centroid-CV false P2 (identical root cause to phase_40). artikulation excluded: loudness normalization + dithering modifies onset-energy peaks → quick proxy saturates at 1.0 pre-phase (mean_peaks/0.01 clips) then drops after processing → false P1 catastrophic regression (0.4803 observed, before=1.000→after=0.520, 2026-04-02).
    # Spatial enhancement (cross-feed early reflections + Schroeder all-pass diffusion):
    # 4 early reflections (6–22 ms, −8 to −16 dB, dry_wet=0.18) are cross-fed:
    # L_out += gain × dry_wet × delayed_R (and vice versa). This DOES change the mono
    # sum M = (L+R)/2 by introducing delayed cross-channel copies → comb-filtering
    # pattern in M → MFCC-Pearson + centroid-CV shift → false P2 timbre regression.
    # emotionalitaet excluded: cross-feed reflections add short-decay tail after
    # transients → crest-factor (peak/RMS ratio) decreases → false P3 regression
    # (same mechanism as phase_22: adding post-peak tail energy compresses crest).
    # waerme excluded: early reflections add diffuse energy across the spectrum
    # including the 200–800 Hz warmth band → warmth ratio E(200-800)/E(800-3000)
    # shifts → false P4 regression.
    "phase_46": {
        "timbre_authentizitaet",
        "emotionalitaet",
        "waerme",
    },  # Spatial enhancement: cross-feed early reflections modify mono sum (comb-filter in M) + add post-transient tail energy (crest-factor drop) + mid-band reflection energy (warmth ratio change) → false P2/P3/P4 regressions
    # Stereo width enhancer (STFT-based frequency-dependent M/S width:
    # LF×0.6 / MF×1.0 / HF×1.15, plus allpass decorrelation delays 17.1/19.7/23.3 ms):
    # STFT-based frequency-dependent Side scaling changes the spectral distribution of
    # both L and R channels (L = M + S×freq_factor). The HF enhancement (×1.15 above
    # 8 kHz) raises HF energy in the stereo image → centroid-CV shifts upward in L and
    # R → MFCC higher-order coefficients change vs. the unwidened reference → false P2
    # timbre regression (identical mechanism to phase_39 air band, phase_21/phase_25).
    "phase_48": {
        "timbre_authentizitaet"
    },  # Stereo width enhancer: STFT frequency-dependent M/S Side scaling (HF ×1.15) changes L/R spectral distribution → MFCC higher-order coefficients + centroid-CV shift → false P2 timbre regression (identical mechanism to phase_39 air band)
    # ── §2.54 CIG-PMGG-Synchronisation: Phasen ohne bisher existierende PMGG-Einträge ───────────────────
    # Groove-echo cancellation (inner-groove vinyl pre-echo): removes pre-echo artefact.
    # authentizitaet: pre-echo creates phantom chroma artefacts; removal changes chromagram (§2.44).
    # timbre_authentizitaet: pre-echo spectral coloration is removed → MFCC-Pearson shifts vs. pre-echo-bearing reference.
    "phase_61": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Groove-echo cancellation: pre-echo phantom chroma removed → chromagram + MFCC fingerprint change vs. pre-echo-distorted reference (§2.44)
    # Crosstalk cancellation (early stereo channel leakage repair): removes inter-channel contamination.
    # authentizitaet: stereo-field chromagram fingerprint changes vs. crosstalk-distorted reference (§2.46).
    # timbre_authentizitaet: spectral crosstalk coloration removed → MFCC-Pearson shifts intentionally.
    "phase_62": {
        "authentizitaet",
        "timbre_authentizitaet",
    },  # Crosstalk cancellation: inter-channel spectral leakage removed → stereo fingerprint change vs. crosstalk-distorted reference (§2.46)
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


# ---------------------------------------------------------------------------
# §PMGG-Restorative: Phasen die Defekte ENTFERNEN statt Klang zu formen.
# Defekte erhöhen viele Metriken künstlich über ihren sauberen Wert:
#   Rauschen füllt Spektraltäler → AuthentizitaetProxy erscheint HOCH.
#   HF-Rauschen → BrillanzProxy erscheint HOCH.
#   Hall → Wärme/Transparenz erscheinen HOCH.
# Nach Restaurierung fallen diese Scores auf reale Werte → PMGG wertet es
# als Regression obwohl es eine Verbesserung ist.
# Lösung: Für restorative Phasen wird scores_before auf die normativen
# Qualitäts-Schwellwerte gedeckelt (§14 Musical Goals, Restoration-Modus).
# Dadurch kann keine defekt-inflationierte Baseline eine false-positive
# Regression auslösen. Echter Schaden (Score unter Schwelle) wird weiterhin erkannt.
# ---------------------------------------------------------------------------
# §TFS: Phases where Temporal Fine Structure coherence is measured.
# These are heavy spectral-modification phases that can disrupt sub-1.5 kHz
# instantaneous phase relationships (pitch, binaural cues, consonant texture).
# Scientific basis: Moore (2008) JARO 9(4), Lorenzi et al. (2006) PNAS 103(49).
_TFS_SENSITIVE_PHASES: frozenset[str] = frozenset(
    {
        "phase_03",  # Broadband denoise — spectral shaping disrupts TFS
        "phase_09",  # BANQUET blind denoising — full-band spectral mod
        "phase_20",  # Reverb reduction (SGMSE+) — diffuse field removal affects phase
        "phase_29",  # Tape hiss reduction (DeepFilterNet) — HF removal cascades into TFS
        "phase_49",  # Advanced dereverb — aggressive spectral subtraction
    }
)
_TFS_COHERENCE_THRESHOLD: float = 0.85  # Below this → phase disrupted fine structure
_TFS_RETRY_TRIGGER: float = 0.15  # TFS delta > this AND P1/P2 regression → extra retry

_RESTORATIVE_PHASES: frozenset[str] = frozenset(
    {
        "phase_01",  # Click removal
        "phase_02",  # Hum removal (Kammfilter)
        "phase_03",  # Broadband denoise (OMLSA + ResembleEnhance)
        "phase_04",  # EQ correction (RIAA/NAB de-emphasis inversion) — HF/LF energy redistribution inflates brillanz/waerme proxies
        "phase_05",  # Rumble filter (subtractive LF cleanup)
        "phase_09",  # BANQUET blind denoising
        "phase_12",  # Wow/flutter correction (§2.44 Reference-Paradoxon: pitch dewarping changes chroma vs. wobble-distorted reference)
        "phase_14",  # Stereo phase correction (multi-band alignment) — fixes carrier phase misalignment; stereo-fingerprint changes vs. mis-aligned reference
        "phase_15",  # Stereo balance correction — corrects L/R imbalance defect; energy shift changes authentizitaet proxy vs. imbalanced reference
        "phase_18",  # Noise gate (Silero VAD)
        "phase_19",  # De-esser — sibilance carrier distortion (vinyl HF, cassette) inflates brillanz; post-reduction drop is defect-removal, not regression
        "phase_20",  # Reverb reduction (SGMSE+)
        "phase_23",  # Spectral inpainting / gap-fill (AudioSR)
        "phase_24",  # Dropout repair (AudioSR)
        "phase_25",  # Azimuth correction — tape head misalignment repair; HF balance changes vs. mis-aligned reference
        "phase_27",  # Click/pop removal
        "phase_28",  # Surface noise profiling (vinyl — broadband noise inflates proxy baselines identically to phase_03/phase_29)
        "phase_29",  # Tape hiss reduction (DeepFilterNet v3 II)
        "phase_30",  # DC offset / near-DC drift removal
        "phase_31",  # Speed/pitch correction (pYIN + WSOLA) — corrects turntable/tape speed deviation; tonal_center/groove proxies change vs. pitch-deviated checkpoint (§2.44 Reference-Paradoxon identical to phase_12)
        "phase_49",  # Advanced dereverb
        "phase_50",  # STFT spectral inpainting (bin interpolation)
        "phase_55",  # Diffusion inpainting (CQTdiff+) — gap reconstruction: silence-baseline inflates tonal_center/waerme
        "phase_56",  # Spectral band gap repair (HEAD_WEAR)
        "phase_57_print_through_reduction",  # Print-through reduction (bidirectional LMS)
        "phase_59",  # Tape modulation noise reduction — carrier-induced FM noise removal inflates tonal_center proxy
        "phase_60",  # Inner groove distortion repair (vinyl) — THD reduction changes spectral fingerprint vs. distorted reference
        "phase_62",  # Crosstalk cancellation — early stereo channel separation repair; stereo fingerprint changes vs. crosstalk-distorted reference
        "phase_63",  # Intermodulation distortion reduction (M/S-domain) — IMD artefact removal changes spectral energy vs. distorted reference
    }
)

# Normative Mindest-Schwellwerte (§14 Musical Goals, §9.10.77 Pareto-Differenzierung).
# Werden als Baseline-Deckel für restorative Phasen genutzt.
# Scores_before über diesem Wert werden auf diesen Wert gedeckelt,
# da höhere Werte nur durch Defekt-Inflation entstehen können.
#
# P1/P2: identisch für beide Modi.
# P3–P5: Restoration senkt auf physikalisch erreichbare Werte (Pareto-Konflikte);
#         Studio 2026 behält ambitionierte Ziele.
_CANONICAL_THRESHOLDS_RESTORATION: dict[str, float] = {
    # P1
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    # P2
    "tonal_center": 0.95,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.85,
    # P3
    "emotionalitaet": 0.82,
    "micro_dynamics": 0.88,
    "groove": 0.83,
    # P4
    "transparenz": 0.82,
    "waerme": 0.75,
    "bass_kraft": 0.78,
    "separation_fidelity": 0.78,
    # P5
    "brillanz": 0.78,
    "spatial_depth": 0.70,
}

_CANONICAL_THRESHOLDS_STUDIO2026: dict[str, float] = {
    # P1 — identical
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    # P2 — identical
    "tonal_center": 0.97,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.85,
    # P3 — Studio 2026: higher targets
    "emotionalitaet": 0.87,
    "micro_dynamics": 0.92,
    "groove": 0.88,
    # P4 — Studio 2026: higher targets
    "transparenz": 0.89,
    "waerme": 0.80,
    "bass_kraft": 0.85,
    "separation_fidelity": 0.82,
    # P5 — Studio 2026: higher targets
    "brillanz": 0.85,
    "spatial_depth": 0.75,
}

# Default alias for backward compatibility (Restoration-Modus)
_CANONICAL_THRESHOLDS: dict[str, float] = _CANONICAL_THRESHOLDS_RESTORATION


def _get_canonical_thresholds(is_studio_2026: bool = False) -> dict[str, float]:
    """Return mode-appropriate canonical thresholds (§9.10.77 Pareto-Differenzierung).

    P1/P2 are identical for both modes.
    P3–P5 are higher in Studio 2026 (ambitious targets), lower in Restoration
    (Pareto-conflict-aware, physically achievable).
    """
    if is_studio_2026:
        return _CANONICAL_THRESHOLDS_STUDIO2026
    return _CANONICAL_THRESHOLDS_RESTORATION


# Strength-Faktoren für Retry-Durchgänge
# v9.10.79: 5 Stufen für 5 vollständige Retries (0–4). Floor = 0.15 für Last-Resort.
# Psychoakustik: strength ≥ 0.15 still perceivable (−18 dB Wet bleiben unter Maskierungsschwelle).
# Nach 5 fehlgeschlagenen Retries: best-effort Anwendung (Spec §2.29 v9.10.64).
_RETRY_STRENGTHS: list[float] = [
    0.65,
    0.50,
    0.35,
    0.25,
    0.15,
]  # v9.10.79: 5 Stufen (Retry-Index 0–4), Floor 0.15 last-resort

# §2.29a ML-deterministische Phasen: Inference-Output ist bei gleichem Input
# identisch, unabhängig vom strength-Parameter.  Bei PMGG-Retries wird nur
# Wet/Dry-Reblending variiert — keine Re-Inferenz.
# Phase-ID-Prefixes (startswith-Match) für robustes Matching.
_ML_DETERMINISTIC_PHASES: frozenset[str] = frozenset(
    {
        "phase_03",  # OMLSA + ResembleEnhance (ML-Hybrid Denoising)
        "phase_06",  # AudioSR (neurale Bandwidth-Extension)
        "phase_09",  # BANQUET ONNX (Blind-Denoising)
        "phase_12",  # FCPE/CREPE/pYIN (f₀-Schätzung) — Timing-Phase, kein Wet/Dry
        "phase_18",  # Silero VAD (Binary-Mask)
        "phase_19",  # De-Esser+VocalStack: process() ignoriert strength → Wet/Dry reicht
        "phase_20",  # SGMSE+ (Reverb-Separation) — nur ML-deterministisch wenn SGMSE+ geladen
        # WPE-Fallback ist strength-abhängiger DSP → _phase20_is_ml_active() prüft zur Laufzeit
        "phase_23",  # AudioSR Inpainting (Spektral-Lückenfüllung)
        "phase_29",  # DeepFilterNet v3 II (HF-Denoising)
        "phase_42",  # BSRoFormer (Stem-Separation)
        "phase_56",  # FCPE/CREPE + Synthese (Spectral Band Gap Repair)
    }
)


def _material_key_from_phase_kwargs(phase_kwargs: dict[str, Any] | None) -> str:
    """Return normalized material key from phase kwargs."""
    if not isinstance(phase_kwargs, dict):
        return "unknown"
    _raw = phase_kwargs.get("material_type", phase_kwargs.get("material", "unknown"))
    _txt = str(getattr(_raw, "value", _raw) or "unknown").strip().lower()
    if _txt.startswith("materialtype."):
        _txt = _txt.split(".", 1)[1]
    return _txt or "unknown"


def _phase_safe_strength_cap(phase_id: str, phase_kwargs: dict[str, Any] | None) -> float:
    """Conservative phase-specific cap to reduce P1/P2 drift cascades.

    These caps are intentionally material-adaptive and only applied to known
    high-risk phases (02/03/12/24/29/55) that repeatedly triggered rollback cascades.
    """
    _mat = _material_key_from_phase_kwargs(phase_kwargs)
    _caps: dict[str, dict[str, float]] = {
        "phase_02_hum_removal": {
            "vinyl": 0.34,
            "tape": 0.36,
            "reel_tape": 0.34,
            "shellac": 0.32,
            "wax_cylinder": 0.30,
            "wire_recording": 0.30,
            "cassette": 0.36,
            "cd_digital": 0.40,
            "dat": 0.40,
            "mp3_low": 0.38,
            "mp3_high": 0.40,
            "aac": 0.40,
            "streaming": 0.40,
            "unknown": 0.36,
        },
        "phase_03_denoise": {
            "vinyl": 0.42,
            "tape": 0.44,
            "reel_tape": 0.42,
            "shellac": 0.40,
            "wax_cylinder": 0.38,
            "wire_recording": 0.38,
            "cassette": 0.44,
            "cd_digital": 0.48,
            "dat": 0.48,
            "mp3_low": 0.46,
            "mp3_high": 0.48,
            "aac": 0.48,
            "streaming": 0.48,
            "unknown": 0.44,
        },
        "phase_12_wow_flutter_fix": {
            "vinyl": 0.62,
            "tape": 0.70,
            "reel_tape": 0.66,
            "shellac": 0.56,
            "wax_cylinder": 0.52,
            "wire_recording": 0.52,
            "cassette": 0.68,
            "lacquer_disc": 0.58,
            "unknown": 0.60,
        },
        "phase_24_dropout_repair": {
            "vinyl": 0.58,
            "tape": 0.62,
            "reel_tape": 0.60,
            "shellac": 0.54,
            "wax_cylinder": 0.52,
            "wire_recording": 0.52,
            "cassette": 0.62,
            "cd_digital": 0.64,
            "dat": 0.64,
            "mp3_low": 0.66,
            "mp3_high": 0.64,
            "aac": 0.64,
            "streaming": 0.64,
            "unknown": 0.60,
        },
        "phase_29_tape_hiss_reduction": {
            "vinyl": 0.34,
            "tape": 0.36,
            "reel_tape": 0.35,
            "shellac": 0.32,
            "wax_cylinder": 0.30,
            "wire_recording": 0.30,
            "cassette": 0.36,
            "cd_digital": 0.40,
            "dat": 0.40,
            "mp3_low": 0.38,
            "mp3_high": 0.40,
            "aac": 0.40,
            "streaming": 0.40,
            "unknown": 0.36,
        },
        "phase_55_diffusion_inpainting": {
            "vinyl": 0.46,
            "tape": 0.50,
            "reel_tape": 0.48,
            "shellac": 0.42,
            "wax_cylinder": 0.40,
            "wire_recording": 0.40,
            "cassette": 0.50,
            "cd_digital": 0.54,
            "dat": 0.54,
            "mp3_low": 0.56,
            "mp3_high": 0.54,
            "aac": 0.54,
            "streaming": 0.54,
            "unknown": 0.48,
        },
    }
    _for_phase = _caps.get(phase_id)
    if not _for_phase:
        return 1.0
    return float(_for_phase.get(_mat, _for_phase.get("unknown", 1.0)))


def _resolve_team_context_policy(phase_id: str, phase_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Return PMGG team-coordination policy derived from prior phase context.

    The policy is advisory and only affects PMGG retry/goal-check behavior.
    It does not disable final export gates and does not bypass safety guards.
    """
    _policy: dict[str, Any] = {
        "goal_exclusions": set(),
        "threshold_multiplier": 1.0,
        "strength_cap": 1.0,
        "reason": "",
    }
    if not isinstance(phase_kwargs, dict):
        return _policy

    _ctx = phase_kwargs.get("prior_phase_context")
    if not isinstance(_ctx, dict) or not _ctx:
        return _policy

    _is_phase50 = str(phase_id).startswith("phase_50")
    _hf_chain_applied = bool(
        _ctx.get("harmonic_restoration_applied")
        or _ctx.get("frequency_restoration_applied")
        or _ctx.get("spectral_super_resolution_applied")
    )

    # Generic all-phase transition policy (module/phase complete coverage)
    # ---------------------------------------------------------------
    # Uses phase ontology types to derive conservative PMGG adjustments for
    # potentially conflicting transition pairs. This keeps behavior centralized
    # and avoids manual per-phase hotfixes.
    try:
        from backend.core.phase_ontology import get_phase_type

        _cur_t = getattr(get_phase_type(str(phase_id)), "name", "")
        _prev_t = str(_ctx.get("last_phase_type", "") or "")
        _transition = (_prev_t, _cur_t)
        _TRANSITION_POLICY: dict[tuple[str, str], dict[str, Any]] = {
            # Prior additive reconstruction followed by subtractive cleanup:
            # avoid over-penalizing intentional HF/timbre changes.
            ("ADDITIVE", "SUBTRACTIVE"): {
                "goal_exclusions": {"brillanz", "transparenz"},
                "threshold_multiplier": 1.08,
                "strength_cap": 0.90,
                "reason": "transition_additive_to_subtractive",
            },
            # Diffusion/ML-generated content followed by subtractive cleanup:
            # articulation/micro-dynamics proxies often overreact.
            ("ML_GENERATIVE", "SUBTRACTIVE"): {
                "goal_exclusions": {"artikulation", "micro_dynamics"},
                "threshold_multiplier": 1.08,
                "strength_cap": 0.90,
                "reason": "transition_mlgen_to_subtractive",
            },
            # Dynamics processing after additive synthesis: preserve reconstructed
            # transients and avoid aggressive PMGG-driven attenuation.
            ("ADDITIVE", "DYNAMICS"): {
                "goal_exclusions": {"artikulation"},
                "threshold_multiplier": 1.05,
                "strength_cap": 0.92,
                "reason": "transition_additive_to_dynamics",
            },
            # Corrective after additive: tonal/timbre proxies can reflect intentional
            # spectral re-centering rather than real degradation.
            ("ADDITIVE", "CORRECTIVE"): {
                "goal_exclusions": {"timbre_authentizitaet"},
                "threshold_multiplier": 1.05,
                "strength_cap": 0.94,
                "reason": "transition_additive_to_corrective",
            },
        }
        _tp = _TRANSITION_POLICY.get(_transition)
        if isinstance(_tp, dict):
            _policy["goal_exclusions"] |= set(_tp.get("goal_exclusions", set()))
            _policy["threshold_multiplier"] = max(
                float(_policy["threshold_multiplier"]), float(_tp.get("threshold_multiplier", 1.0))
            )
            _policy["strength_cap"] = min(float(_policy["strength_cap"]), float(_tp.get("strength_cap", 1.0)))
            if not _policy["reason"]:
                _policy["reason"] = str(_tp.get("reason", ""))
    except Exception:
        pass

    # Team rule: if prior phases already restored HF content, phase_50 should
    # avoid treating those bins as "damage" via indirect metric pressure.
    if _is_phase50 and _hf_chain_applied:
        _policy["goal_exclusions"] = {"brillanz", "transparenz", "timbre_authentizitaet"}
        _policy["threshold_multiplier"] = 1.15
        _policy["strength_cap"] = 0.80
        _policy["reason"] = "phase50_after_hf_restoration"

    return _policy


def _allow_emergency_retries(
    phase_id: str,
    worst_priority: int,
    best_regression: float,
    catastrophic_threshold: float,
    team_policy: dict[str, Any] | None,
) -> bool:
    """Return whether PMGG emergency retries should run for this phase.

    Team-policy can disable emergency retries when a measured regression is likely
    a proxy artifact caused by intentional prior restoration steps.
    """
    if not (best_regression > catastrophic_threshold and worst_priority <= 2):
        return False

    if isinstance(team_policy, dict):
        _reason = str(team_policy.get("reason", ""))
        # phase_50 after HF restoration (phase_06/phase_07/phase_23):
        # emergency low-strength loops are typically wasted because the observed
        # P1/P2 proxy drop stems from intentional HF changes, not real damage.
        if phase_id.startswith("phase_50") and _reason == "phase50_after_hf_restoration":
            return False

    return True


def _phase20_is_ml_active() -> bool:
    """Return True when SGMSE+ is currently loaded in the ML budget (§2.29a).

    phase_20 is ML-deterministic only when the SGMSE+ model is actually resident
    in memory.  When SGMSE+ was blocked by ml_memory_budget (OOM pressure) and
    the WPE-DSP fallback is active instead, wet/dry blending cannot represent the
    full range of WPE's strength-dependent predictor-order parameter.  In that
    case phase_20 must be treated as a strength-dependent DSP phase — re-run on
    every PMGG retry.
    """
    try:
        from backend.core.ml_memory_budget import get_status

        return "SGMSE+" in get_status().get("models", {})
    except Exception:
        return False  # Safe default: DSP path — must re-run


def _get_adaptive_threshold(restorability_score: float, material_type: str = "unknown") -> float:
    """§2.29/§2.54 Material- und Restorability-adaptiver REGRESSION_THRESHOLD.

    Args:
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100]
        material_type: Carrier-Materialklasse (z.B. 'vinyl', 'shellac', 'cd_digital')

    Returns:
        Adaptiver Schwellwert ∈ [0.012, 0.070].
        Analog/physische Träger erhalten einen Material-Bonus, da Carrier-Repair-
        Phasen das Signal intentional ändern (Referenz-Paradoxon §2.44).
    """
    # Restorability-tier Basis
    if restorability_score >= 70.0:
        base = REGRESSION_THRESHOLD_GOOD
    elif restorability_score >= 40.0:
        base = REGRESSION_THRESHOLD_FAIR
    else:
        base = REGRESSION_THRESHOLD_POOR
    # Material-Bonus: analog-physische Träger benötigen mehr Toleranz (§2.54)
    bonus = _MATERIAL_THRESHOLD_BONUS.get(material_type.lower(), 0.003)
    threshold = base + bonus
    # Hard-Cap: nie enger als 0.012 (Messrauschen), nie lockerer als 0.070
    return float(np.clip(threshold, 0.012, 0.070))


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
    metadata: dict[str, Any] = field(default_factory=dict)  # TFS coherence, vocal_intimacy, etc.


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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-Korrelation mit Längen-Matching und NaN/Inf-Sicherheit.

    Returns 0.0 bei Fehler oder zu wenig Daten.
    """
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    try:
        av = a[:n].ravel()
        bv = b[:n].ravel()
        if float(np.std(av)) < 1e-12 or float(np.std(bv)) < 1e-12:
            return 1.0 if np.allclose(av, bv, atol=1e-12, rtol=1e-6) else 0.0
        r = float(np.corrcoef(av, bv)[0, 1])
        return r if math.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _get_precise_metric_instances() -> dict[str, Any]:
    """Lazy-load a small set of production musical-goal metrics for PMGG.

    These are used selectively for the most decision-critical goals where local
    DSP proxies are materially less precise than the canonical metric.
    """
    global _PRECISE_METRICS
    if _PRECISE_METRICS is None:
        with _PRECISE_METRICS_LOCK:
            if _PRECISE_METRICS is None:
                try:
                    from backend.core.musical_goals.musical_goals_metrics import (
                        ArticulationMetric,
                        MicroDynamicsMetric,
                        SeparationFidelityMetric,
                    )

                    _PRECISE_METRICS = {
                        # brillanz intentionally omitted: §9.7.12 HF-crest-factor quick proxy
                        # is symmetric and SNR-robust for PMGG delta checks.  The absolute
                        # BrillanzMetric._measure_absolute() (ISO-226 HF-energy ratio) is still
                        # SNR-dependent → would show false drop after denoising even without the
                        # reference-preservation penalty.  Both scores_before and scores_after
                        # now use the crest-factor proxy consistently → symmetric, no false regressions.
                        # The canonical BrillanzMetric still runs in the final export gate.
                        # waerme intentionally omitted: §9.7.14 warmth-ratio quick proxy
                        # (E_200-800 / E_800-3000) is reverb-invariant.  WaermeMetric._measure_absolute()
                        # uses ISO-226 mid/total ratio which drops after dereverb → false regression.
                        # transparenz intentionally omitted: §9.7.13 multi-band crest-factor quick
                        # proxy is SNR-robust.  TransparenzMetric.measure() also has no reference=
                        # parameter → precise override was silently failing (TypeError) already.
                        # natuerlichkeit intentionally omitted: NatuerlichkeitMetric uses
                        # CREPE ML inference (1–4 s/call) with dynamic weight switching
                        # based on CREPE load state.  Between scores_before (CREPE not
                        # yet loaded → w_crepe=0.0) and scores_after (CREPE loaded →
                        # w_crepe=0.18) the absolute score shifts non-deterministically,
                        # creating systematic false P1 regressions in phase_03/phase_02.
                        # The DSP proxy in _measure_quick with §9.7.5 reference-aware
                        # preservation correction is more reliable for PMGG delta checks.
                        # The canonical NatuerlichkeitMetric still runs in the final
                        # export quality gate (MusicalGoalsChecker).
                        #
                        # tonal_center intentionally omitted (§2.29b, v9.10.93):
                        # TonalCenterMetric uses librosa.feature.chroma_stft and applies a
                        # binary key-shift penalty (1 semitone → score ≤ 0.50; ≥2 → 0.0).
                        # This causes systematic catastrophic false P2 regressions in phases
                        # that legitimately change harmonic-percussive balance or energy
                        # distribution without changing the musical key:
                        #   - phase_08 TDP/HPSS: Δ=0.5612 observed (transient reshaping
                        #     shifts dominant chroma class in librosa by 1 semitone)
                        #   - phase_36 transient shaper: Δ=0.3231 observed (same mechanism)
                        #   - phase_49 advanced dereverb: Δ=0.5312 observed (reverb decay
                        #     energy removal changes chroma frame distribution)
                        # Root cause: librosa chroma_stft is sensitive to energy-envelope
                        # changes even when pitch content is unchanged.  A 1-semitone
                        # apparent shift (e.g. A→A# due to energy redistribution) triggers
                        # the 50% penalty → catastrophic threshold breached → 5+ retries +
                        # emergency retries → Watchdog timeout.
                        # The K-S quick proxy in _measure_quick is the correct PMGG tool:
                        # it uses a multi-frame chroma sum → argmax is stable under energy
                        # redistribution that does not change the dominant pitch class.
                        # The canonical TonalCenterMetric still runs in the final export gate.
                        "micro_dynamics": MicroDynamicsMetric(),
                        "artikulation": ArticulationMetric(),
                        "separation_fidelity": SeparationFidelityMetric(),
                    }
                except Exception as exc:
                    logger.debug("PMGG precise metrics unavailable: %s", exc)
                    _PRECISE_METRICS = {}
    return _PRECISE_METRICS


def _apply_precise_metric_overrides(
    scores: dict[str, float],
    audio: np.ndarray,
    sr: int,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Refine selected quick scores using canonical metric implementations."""
    t0 = time.perf_counter()
    precise_metrics = _get_precise_metric_instances()
    if not precise_metrics:
        return scores

    # §9.7.7 Audio length cap: 2.5 s is sufficient for all precise metrics and
    # avoids long NMF/onset-detection runs in SeparationFidelityMetric /
    # ArticulationMetric on long audio samples.
    _cap = int(2.5 * sr)
    if audio.ndim == 1 and len(audio) > _cap:
        audio = audio[:_cap]
    elif audio.ndim == 2 and audio.shape[-1] > _cap:
        audio = audio[..., :_cap]
    if reference is not None:
        if reference.ndim == 1 and len(reference) > _cap:
            reference = reference[:_cap]
        elif reference.ndim == 2 and reference.shape[-1] > _cap:
            reference = reference[..., :_cap]

    refined = dict(scores)
    for goal_name, metric in precise_metrics.items():
        try:
            if goal_name == "micro_dynamics":
                # Always reference-free: scores_before is measured without reference,
                # so scores_after must use the same absolute mode for a fair comparison.
                # Reference-based MicroDynamicsMetric gives 0.60+ baseline vs ~0.75×corr
                # for scores_after, creating systematic false regressions in PMGG.
                refined[goal_name] = float(metric.measure(audio, sr))
            elif goal_name in {
                "tonal_center",
                "artikulation",
                "separation_fidelity",
            }:
                refined[goal_name] = float(metric.measure(audio, sr, reference=reference))
            else:
                refined[goal_name] = float(metric.measure(audio, sr))
        except Exception as exc:
            logger.debug("PMGG precise metric override failed for %s: %s", goal_name, exc)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > _PRECISE_OVERRIDE_WARN_MS:
        logger.warning(
            "PMGG precise overrides slow: %.1f ms for %d goals",
            elapsed_ms,
            len(precise_metrics),
        )
    return refined


def _measure_quick(
    audio: np.ndarray, sr: int, reference: np.ndarray | None = None, *, precise_override: bool = True
) -> dict[str, float]:
    """
    Misst alle 14 Musical Goals auf einer 5-s-Stichprobe in ≤ 200 ms.

    §9.7.5 (v9.10.77): Referenz-aware Preservation-Korrekturen.
    Wenn ``reference`` übergeben wird, erhalten anfällige Goals einen
    Preservation-Bonus basierend auf spektraler Korrelation.  Dies beseitigt
    False-Positive-Regressionen bei Noise-Removal, EQ, Dynamics-Phasen
    und ermöglicht breitere Goal-Prüfung mit weniger Exclusions.

    Prinzip: Wenn die Korrelation zwischen Original und Verarbeitetem hoch ist
    (musikalischer Inhalt erhalten), wird der absolute Score nach oben korrigiert.
    Bei niedriger Korrelation (echte Degradation) bleibt der absolute Score.

    Args:
        audio: Mono oder Stereo, float32, beliebige Länge
        sr: 48000 Hz
        reference: Original-Audio vor Phasen-Verarbeitung (gleiche Länge).
            None = rein absolute Messung (für scores_before).

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

    # §9.7.5 Pre-compute reference spectrum for preservation corrections.
    # Computed once; used by all reference-aware goal branches below.
    _ref_fft: np.ndarray | None = None
    _ref_mono: np.ndarray | None = None
    if reference is not None:
        try:
            _rm = reference[:, 0] if reference.ndim == 2 else reference
            _rm = np.nan_to_num(_rm, nan=0.0).astype(np.float32)
            _ml = min(len(mono), len(_rm))
            _ref_mono = _rm[:_ml]
            _ref_fft = np.abs(np.fft.rfft(_ref_mono))
        except Exception:
            _ref_fft = None
            _ref_mono = None

    # ── Brillanz (§9.7.12 HF Spectral Crest Factor, 2–16 kHz) ────────
    # Root-cause of prior false regressions: the old HF-energy-ratio proxy was
    # SNR-dependent.  Broadband noise raises the HF energy floor uniformly →
    # high ratio before denoising; after denoising only musical peaks remain →
    # lower absolute energy → false drop of 0.2–0.5.
    #
    # Fix §9.7.12: Spectral crest factor = p95 / p50 within 2–16 kHz band.
    #   • Noise floor lifts the MEDIAN (p50) while leaving p95 ≈ music peaks
    #     → noisy audio: crest LOW (2–3).
    #   • After denoising, noise floor drops → p50 falls toward musical valleys
    #     → crest INCREASES (5–30) → score improves → no false regression.
    # Scientific basis: Fastl & Zwicker, "Psychoacoustics: Facts and Models",
    # 2007 §8.3 Sharpness — crest factor as perceptual brightness indicator.
    # Calibration: crest ≥ 15 → score 1.0; crest 1.5 → score 0.0.
    try:
        _hf_mask_b = (freqs >= 2000) & (freqs <= 16000)
        _hf_bins_b = fft_mag[_hf_mask_b]
        if len(_hf_bins_b) > 20:
            _p95_b = float(np.percentile(_hf_bins_b, 95))
            _p50_b = float(np.median(_hf_bins_b)) + 1e-9
            _hf_crest_b = _p95_b / _p50_b
            scores["brillanz"] = float(np.clip((_hf_crest_b - 1.5) / 13.5, 0.0, 1.0))
        else:
            scores["brillanz"] = 0.5
    except Exception:
        scores["brillanz"] = 0.5

    # ── Wärme (§9.7.14 Warmth Ratio: E_200-800 / E_800-3000 Hz) ──────
    # Root-cause of prior false regressions: the old mid/total-energy ratio was
    # reverb-sensitive.  Reverb tail adds diffuse energy across the mid band →
    # high ratio before dereverb; after removal dry signal has less mid energy →
    # false drop in waerme.
    #
    # Fix §9.7.14: Warmth ratio = E(200–800 Hz) / E(800–3000 Hz).
    #   • Reverb affects BOTH sub-bands proportionally (air absorption is gradual
    #     at these frequencies, early reflections span 200–3000 Hz uniformly) →
    #     the ratio stays stable during dereverb → reverb-invariant.
    #   • Only genuine spectral-balance changes (EQ, vinyl roll-off) shift the ratio
    #     in a perceptually meaningful way.
    # Scientific basis: Moore & Glasberg (1983) auditory filter bandwidths;
    # Fletcher & Rossing vocal formant structure (warmth ≈ F1/F2 energy balance).
    # Calibration: ratio 1.5 → score 1.0 (warm); ratio 0 → score 0.0 (thin).
    try:
        _e_low_mid = float(np.mean(fft_mag[(freqs >= 200) & (freqs < 800)] ** 2)) + 1e-9
        _e_upper_mid = float(np.mean(fft_mag[(freqs >= 800) & (freqs < 3000)] ** 2)) + 1e-9
        scores["waerme"] = float(np.clip(_e_low_mid / _e_upper_mid / 1.5, 0.0, 1.0))
    except Exception:
        scores["waerme"] = 0.5

    # ── Groove (Onset-Energie-Regularität via Autokorrelation) ─────────
    try:
        env = np.abs(mono)
        # Hüllkurven-Autokorrelation
        hop = sr // 100  # 10 ms
        # Vectorized: non-overlapping frames via reshape (replaces Python list comprehension)
        _nf_g = (len(env) - 1) // hop
        rms_env = (
            np.mean(env[: _nf_g * hop].reshape(_nf_g, hop) ** 2, axis=1) if _nf_g > 0 else np.empty(0, dtype=np.float32)
        )
        if len(rms_env) > 10:
            # §9.7.9 LF-Robustheit: 5-Frame-Glättung der Einhüllkurve (50 ms).
            # Hintergrund: Hum (50/100 Hz) erzeugt 100/200 Hz-Modulation in |mono|.
            # Bei 10 ms hop je ~0.5–1 Perioden/Frame → frame-to-frame-Varianz.
            # Diese Varianz erhöht autocorr[0] (Gesamtenergie-Normierungsbasis)
            # ohne die 500 ms-Rhythmusperiodizität zu verändern → normiertes
            # autocorr[lag_05] wird durch LF-Spektraländerungen beeinflusst.
            # Fix: 5 × 10 ms = 50 ms Tiefpass entfernt ≥ 20 Hz Hüllkurvenkomponenten
            # (Hum-Modulation) → Normierungsbasis repräsentiert nur Rhythmusenergie.
            # Musikalischer Groove: 0.5–8 Hz (120–1920 BPM) → unverändert.
            _sw = min(5, len(rms_env) // 4)
            if _sw >= 2:
                rms_env = np.convolve(rms_env, np.ones(_sw) / float(_sw), mode="valid")
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

    # ── Tonales Zentrum (Krumhansl-Schmuckler Key Detection, §9.7.11) ──
    # Scientific basis: Krumhansl & Schmuckler 1990, Temperley 2001,
    # Müller "Fundamentals of Music Processing" 2015 §5.3.
    #
    # WHY the previous entropy-based proxy was wrong for PMGG delta-checks:
    #   entropy = -Σ(chroma * log chroma)  measures chroma CONCENTRATION
    #   → SNR-dependent: noise spreads energy uniformly across all 12 bins
    #     → low entropy (flat chroma) BEFORE denoise; tonal signal revealed
    #     AFTER → entropy changes even though musical key is preserved.
    #   → Result: false P2 regression on EVERY noise-reducing phase at ANY
    #     strength (Δ≈0 stagnation confirmed in production logs 2026-03-30).
    #
    # K-S key detection is SNR-invariant: uniform noise raises ALL 24 major/
    # minor correlation scores equally → argmax is unchanged. Only a genuine
    # key-shift (pitch transposition) changes the dominant key label.
    #
    # Algorithm (vectorized):
    #   1. Build chroma vector from log-domain FFT magnitude (Hann window)
    #   2. Correlate against 24 Krumhansl-Schmuckler major/minor profiles
    #      (normalized to unit-variance for Pearson equivalence)
    #   3. key_before = argmax of 24 scores in _ref, key_after = argmax in proc
    #   4. Circular semitone distance d = min(|k_a − k_b| mod 12, 12 − ...) ∈ [0,6]
    #   5. tonal_center = 1 − d/6   (0 = tritone/max shift, 1 = same key)
    #   6. Fallback (no reference available): best correlation score, normalized.
    #
    # Krumhansl-Schmuckler major/minor profiles (canonical, from Krumhansl 1990
    # Table 1 + Temperley 2001 re-normalisation).
    _KS_MAJOR: np.ndarray = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        dtype=np.float32,
    )
    _KS_MINOR: np.ndarray = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        dtype=np.float32,
    )
    # Pre-normalise profiles once (zero-mean, unit-variance)
    _ks_maj_n: np.ndarray = _KS_MAJOR - _KS_MAJOR.mean()
    _ks_maj_n /= _ks_maj_n.std() + 1e-12
    _ks_min_n: np.ndarray = _KS_MINOR - _KS_MINOR.mean()
    _ks_min_n /= _ks_min_n.std() + 1e-12

    def _ks_key(signal_mono: np.ndarray, n_fft: int = 4096, sr_inner: int = 48000) -> int:
        """Return dominant key label 0–23 (0–11 major, 12–23 minor, root = C).

        Uses multi-segment averaging (8 windows) for stability across
        the entire signal, rather than a single center window which is
        vulnerable to phase-specific spectral changes (e.g. TP limiting).

        Returns -1 on failure (too short / silence).
        """
        n_seg = 8
        seg_len = n_fft

        if len(signal_mono) < seg_len:
            # Short signal: single segment (original behaviour)
            segments = [signal_mono]
        else:
            # Distribute n_seg segments evenly across the signal
            step = max(1, (len(signal_mono) - seg_len) // max(1, n_seg - 1))
            segments = []
            for i in range(n_seg):
                start = min(i * step, len(signal_mono) - seg_len)
                segments.append(signal_mono[start : start + seg_len])

        # Accumulate chroma across all segments
        chroma_acc = np.zeros(12, dtype=np.float64)
        for seg in segments:
            win = np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(seg * win, n=n_fft))
            freqs_k = np.fft.rfftfreq(n_fft, d=1.0 / sr_inner)
            _kb = np.where((freqs_k > 27.5) & (freqs_k < 4186.0))[0]
            if len(_kb) == 0:
                continue
            _kn = np.round(12.0 * np.log2(freqs_k[_kb] / 440.0 + 1e-12)).astype(np.int32) % 12
            _chroma_seg = np.zeros(12, dtype=np.float64)
            np.add.at(_chroma_seg, _kn, spec[_kb].astype(np.float64))
            seg_sum = _chroma_seg.sum()
            if seg_sum > 1e-8:
                chroma_acc += _chroma_seg / seg_sum  # normalize each segment contribution

        s = chroma_acc.sum()
        if s < 1e-8:
            return -1
        chroma_k = (chroma_acc / s).astype(np.float32)
        # Zero-mean + unit-variance normalisation of the chroma vector
        chroma_k -= chroma_k.mean()
        std_c = chroma_k.std()
        if std_c < 1e-12:
            return -1
        chroma_k /= std_c
        # Correlate against all 12 rotations of major and minor profiles
        best_score = -np.inf
        best_key = 0
        for root in range(12):
            maj_rot = np.roll(_ks_maj_n, root)
            min_rot = np.roll(_ks_min_n, root)
            r_maj = float(np.dot(chroma_k, maj_rot))
            r_min = float(np.dot(chroma_k, min_rot))
            if r_maj > best_score:
                best_score, best_key = r_maj, root  # major: 0–11
            if r_min > best_score:
                best_score, best_key = r_min, root + 12  # minor: 12–23
        return best_key

    try:
        _n_fft_ks = 4096
        _key_proc = _ks_key(mono, n_fft=_n_fft_ks, sr_inner=sr)
        if _key_proc == -1:
            scores["tonal_center"] = 0.5
        elif _ref_mono is not None:
            # Delta-mode: compare dominant key before vs. after processing.
            # Circular semitone distance on root (0–11), mode ignored for primary check
            # (mode-shift rare in restoration; penalised lightly via +6 offset if needed).
            _key_ref = _ks_key(_ref_mono, n_fft=_n_fft_ks, sr_inner=sr)
            if _key_ref == -1:
                scores["tonal_center"] = 0.5
            else:
                _root_proc = _key_proc % 12
                _root_ref = _key_ref % 12
                _d = abs(_root_proc - _root_ref)
                _d = min(_d, 12 - _d)  # circular distance ∈ [0, 6]
                # Mode mismatch (major ↔ minor): add 1 semitone equivalent penalty
                _mode_penalty = 0 if (_key_proc // 12 == _key_ref // 12) else 1
                _d = min(6, _d + _mode_penalty)
                scores["tonal_center"] = float(np.clip(1.0 - _d / 6.0, 0.0, 1.0))
        else:
            # No reference available: use normalised best K-S correlation score
            # as absolute quality indicator (0 = atonal noise, 1 = strongly tonal).
            # Re-compute the best score for absolute interpretation.
            _spec_abs = np.abs(np.fft.rfft(mono * np.hanning(len(mono)), n=_n_fft_ks))
            _freqs_abs = np.fft.rfftfreq(_n_fft_ks, d=1.0 / sr)
            _chroma_abs = np.zeros(12, dtype=np.float32)
            _kb2 = np.where((_freqs_abs > 27.5) & (_freqs_abs < 4186.0))[0]
            if len(_kb2) > 0:
                _kn2 = np.round(12.0 * np.log2(_freqs_abs[_kb2] / 440.0 + 1e-12)).astype(np.int32) % 12
                np.add.at(_chroma_abs, _kn2, _spec_abs[_kb2])
            _s2 = _chroma_abs.sum()
            if _s2 > 1e-8:
                _chroma_abs /= _s2
                _chroma_abs -= _chroma_abs.mean()
                _std2 = _chroma_abs.std()
                if _std2 > 1e-12:
                    _chroma_abs /= _std2
                    _best_r = float(
                        max(
                            max(float(np.dot(_chroma_abs, np.roll(_ks_maj_n, r))) for r in range(12)),
                            max(float(np.dot(_chroma_abs, np.roll(_ks_min_n, r))) for r in range(12)),
                        )
                    )
                    # K-S scores range roughly ‑1…+1; clamp to [0, 1]
                    scores["tonal_center"] = float(np.clip((_best_r + 1.0) / 2.0, 0.0, 1.0))
                else:
                    scores["tonal_center"] = 0.5
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
        # §9.7.5 Preservation: Log-spectral envelope correlation
        if _ref_fft is not None:
            _fl = min(len(fft_mag), len(_ref_fft))
            if _fl > 20:
                _log_proc = np.log(fft_mag[:_fl] + 1e-12)
                _log_ref = np.log(_ref_fft[:_fl] + 1e-12)
                _r = _safe_pearson(_log_ref, _log_proc)
                if _r > 0.7:
                    scores["natuerlichkeit"] = min(1.0, scores["natuerlichkeit"] + (_r - 0.7) * 0.5)
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
        # §9.7.5 Preservation: Centroid trajectory correlation with reference
        if _ref_mono is not None and len(centroids) > 2:
            _rm_ml = min(len(mono), len(_ref_mono))
            _ref_centroids = []
            for i in range(0, _rm_ml - hop_t, hop_t):
                _rw = _ref_mono[i : i + hop_t]
                _rw_fft = np.abs(np.fft.rfft(_rw))
                _rw_freqs = np.fft.rfftfreq(len(_rw), d=1.0 / sr)
                _ref_centroids.append(float(np.sum(_rw_freqs * _rw_fft) / (np.sum(_rw_fft) + 1e-12)))
            if len(_ref_centroids) > 2:
                _r = _safe_pearson(np.array(_ref_centroids), np.array(centroids[: len(_ref_centroids)]))
                if _r > 0.7:
                    scores["timbre_authentizitaet"] = min(1.0, scores["timbre_authentizitaet"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["timbre_authentizitaet"] = 0.5

    # ── Bass-Kraft (Bassenergie 20–250 Hz) ─────────────────────────────
    try:
        bass_energy = float(np.mean(fft_mag[(freqs >= 20) & (freqs <= 250)] ** 2))
        # Normierung: typische Bassenergie ~2% des Spektrums → 0.02 = Score 1.0
        scores["bass_kraft"] = float(np.clip(bass_energy / (tot_energy * 0.02 + 1e-12), 0.0, 1.0))
        # §9.7.5 Preservation: LF spectral correlation (20-500 Hz)
        if _ref_fft is not None:
            _lf = (freqs[: len(_ref_fft)] >= 20) & (freqs[: len(_ref_fft)] <= 500)
            if np.sum(_lf) > 5:
                _r = _safe_pearson(_ref_fft[_lf], fft_mag[: len(_ref_fft)][_lf])
                if _r > 0.7:
                    scores["bass_kraft"] = min(1.0, scores["bass_kraft"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["bass_kraft"] = 0.5

    # ── Authentizität (Spektrale Flachheit — aligned mit kanonischer AuthentizitaetMetric)
    # §2.29b Root-Fix (v9.10.79): Der frühere Proxy maß spektrale Rauheit (Abweichung vom
    # geglätteten Log-Spektrum). Das war invertiert für PMGG-Delta-Checks:
    #   - Rauschen füllt spektrale Täler → glatte Log-Amplitude → NIEDRIGE Rauheit
    #     → scores_before HOCH (≈ 0.93) — künstlich inflationiert durch Noise-Floor.
    #   - Sauberes Signal mit Harmonischen → tiefe Täler → HOHE Rauheit
    #     → scores_after NIEDRIG (≈ 0.07) → Pseudo-Regression 0.86 → P1-Katastrophe.
    # Kanonische AuthentizitaetMetric (ohne Reference) verwendet spectral_flatness:
    #   flatness = geom_mean(amplitude) / arith_mean(amplitude)
    #   music → flatness 0.001–0.03 (tonal = authentisch)
    #   noise/codec → flatness 0.10+   (inauthentic)
    # Dieser Proxy repliziert das und verhält sich korrekt für alle Denoise-/Dereverb-Phasen:
    #   scores_before (noisy): flatness ≈ 0.04 → auth ≈ 0.60
    #   scores_after  (clean): flatness ≈ 0.01 → auth ≈ 0.90 → keine Regression ✓
    #   scores_after  (codec-damaged): flatness steigt → auth sinkt → echte Regression ✓
    try:
        _amp = fft_mag + 1e-12
        _geom_auth = float(np.exp(np.mean(np.log(_amp))))
        _arith_auth = float(np.mean(_amp))
        _flatness_auth = float(np.clip(_geom_auth / (_arith_auth + 1e-12), 0.0, 1.0))
        # Calibrated to canonical AuthentizitaetMetric: score = 0 at flatness ≥ 0.10
        scores["authentizitaet"] = float(np.clip(1.0 - _flatness_auth / 0.10, 0.0, 1.0))
        # §9.7.5 Preservation: log-spectral correlation as supplemental signal.
        # Belt + suspenders: even if flatness mis-scores a specific phase output,
        # high spectral correlation with reference boosts the score.
        if _ref_fft is not None:
            _fl = min(len(fft_mag), len(_ref_fft))
            if _fl > 20:
                _r = _safe_pearson(
                    np.log(_ref_fft[:_fl] + 1e-12),
                    np.log(fft_mag[:_fl] + 1e-12),
                )
                if _r > 0.7:
                    scores["authentizitaet"] = min(1.0, scores["authentizitaet"] + (_r - 0.7) * 0.3)
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
        # §9.7.5 Preservation: RMS-envelope correlation (dynamics preservation)
        if _ref_mono is not None:
            _rm_ml = min(len(mono), len(_ref_mono))
            _ref_rms = np.array(
                [
                    float(np.sqrt(np.mean(_ref_mono[i : i + hop_e] ** 2) + 1e-12))
                    for i in range(0, _rm_ml - hop_e, hop_e)
                ]
            )
            _proc_rms = np.array(
                [float(np.sqrt(np.mean(mono[i : i + hop_e] ** 2) + 1e-12)) for i in range(0, _rm_ml - hop_e, hop_e)]
            )
            _r = _safe_pearson(_ref_rms, _proc_rms)
            if _r > 0.7:
                scores["emotionalitaet"] = min(1.0, scores["emotionalitaet"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["emotionalitaet"] = 0.5

    # ── Transparenz (§9.7.13 Multi-Band Spectral Crest Factor, 5 octaves) ─
    # Root-cause of prior false regressions: the 75%-rolloff proxy was
    # SNR-dependent.  Broadband noise raises the high-frequency content
    # → rolloff climbs to 8–12 kHz before denoising; after denoising only
    # musical content remains → rolloff drops to 3–5 kHz → false P4 regression.
    # The §9.7.5 rolloff-floor fix (85 % of reference) only partially mitigated
    # this, and didn't help for phases processed without a reference snapshot.
    #
    # Fix §9.7.13: Multi-band spectral crest factor across 5 octave bands
    #   (250–500 · 500–1k · 1k–2k · 2k–4k · 4k–8k Hz).
    #   • Noise fills each band's floor (raises p50 toward p95) → low crest → low score.
    #   • Denoising clears each band's floor → p50 drops toward musical valleys
    #     → crest rises in ALL bands → score improves → no false regression.
    #   • Reference-free by design: both scores_before and scores_after use the
    #     same absolute formula → symmetric delta even without a clean reference.
    # Scientific basis: Moore & Glasberg (1983); ITU-T P.862 spectral clarity.
    # Calibration: mean crest 1.2 → score 0.0; mean crest 10.0 → score 1.0.
    try:
        _oct_bands_t = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
        _band_crests_t: list[float] = []
        for _fl_t, _fh_t in _oct_bands_t:
            _b_t = fft_mag[(freqs >= _fl_t) & (freqs < _fh_t)]
            if len(_b_t) > 5:
                _p95_t = float(np.percentile(_b_t, 95))
                _p50_t = float(np.median(_b_t)) + 1e-9
                _band_crests_t.append(float(np.clip((_p95_t / _p50_t - 1.2) / 8.8, 0.0, 1.0)))
        if _band_crests_t:
            scores["transparenz"] = float(np.clip(float(np.mean(_band_crests_t)), 0.0, 1.0))
        else:
            scores["transparenz"] = 0.5
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
        # §9.7.5 Preservation: Full-band spectral magnitude coherence
        if _ref_fft is not None:
            _fl = min(len(fft_mag), len(_ref_fft))
            if _fl > 20:
                _r = _safe_pearson(_ref_fft[:_fl], fft_mag[:_fl])
                if _r > 0.7:
                    scores["separation_fidelity"] = min(1.0, scores["separation_fidelity"] + (_r - 0.7) * 0.5)
    except Exception:
        scores["separation_fidelity"] = 0.5

    # ── Artikulation (Onset-Schärfe: Transient-Proxy) ─────────────────
    try:
        # Proxy: Varianz der Energiehüllkurve-Ableitungen (scharfe Transienten = hohe Varianz)
        hop_a = max(1, sr // 200)  # 5 ms
        # Vectorized: non-overlapping peak envelope via reshape
        _nf_a = (len(mono) - 1) // hop_a
        env_a = (
            np.max(np.abs(mono[: _nf_a * hop_a].reshape(_nf_a, hop_a)), axis=1)
            if _nf_a > 0
            else np.empty(0, dtype=np.float32)
        )
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

    if precise_override:
        scores = _apply_precise_metric_overrides(scores, audio, sr, reference=reference)

    for k in FAST_GOALS_SUBSET:
        if k not in scores or not math.isfinite(scores[k]):
            scores[k] = 0.5

    return scores


# Timing phases: intentional time-warping makes *any* correlation metric unreliable.
# 163 transport bumps → envelope reordering → corr≈0.265 even on perfect correction.
# Excluding corr_pen for these phases prevents false Content-Guard rollbacks (§2.48 §2.54).
_TIMING_CORR_EXCLUDE: frozenset[str] = frozenset(
    {
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
    }
)

# LF-subtractive phases: intentional broadband RMS reduction when low-end noise dominates.
# phase_02 (hum), phase_05 (rumble): removing sub-bass / 50 Hz hum CAN reduce broadband RMS
# by 20-30 dB if that noise dominated the signal — this is CORRECT behaviour (§0).
# Using broadband RMS for the drop-penalty creates false Content-Guard rollbacks.
# These phases already have internal §2.45a guards; drop-penalty in PMGG is redundant.
_LF_SUBTRACTIVE_DROP_SKIP: frozenset[str] = frozenset(
    {
        "phase_02_hum_removal",
        "phase_05_rumble_filter",
    }
)


def _content_integrity_penalty(
    reference: np.ndarray,
    processed: np.ndarray,
    skip_corr_check: bool = False,
    skip_drop_check: bool = False,
) -> tuple[float, dict[str, float]]:
    """Detect catastrophic content loss independently from Musical-Goal proxies.

    The guard is intentionally conservative and only reacts to severe failures:
    large broadband energy collapse and/or very low waveform correlation.
    It protects PMGG when many P1/P2 goals are excluded for a phase.

    §9.11.2: Uses RMS-envelope correlation instead of raw sample correlation.
    Time-domain phases (wow/flutter phase vocoder, time-stretch) shift samples
    in time without destroying content — raw corrcoef yields ~0 (false positive).
    10 ms RMS-envelope correlation is time-shift-tolerant while still detecting
    genuine content loss (energy distribution, spectral balance changes).

    skip_corr_check: When True (timing phases with intentional global time-warp),
    the corr_pen component is zeroed — only the RMS-drop component remains active.
    This prevents false Content-Guard rollbacks when 100+ transport bumps are
    corrected (§2.48 Carrier-Repair-Exclusions, §2.54 adaptive thresholds).
    """
    try:
        _ref = np.asarray(reference, dtype=np.float32)
        _out = np.asarray(processed, dtype=np.float32)
        if _ref.ndim == 2 and _ref.shape[1] >= 2:
            _ref_mono = ((_ref[:, 0] + _ref[:, 1]) / np.sqrt(2.0)).astype(np.float32)
        else:
            _ref_mono = (_ref[:, 0] if _ref.ndim == 2 else _ref).astype(np.float32)
        if _out.ndim == 2 and _out.shape[1] >= 2:
            _out_mono = ((_out[:, 0] + _out[:, 1]) / np.sqrt(2.0)).astype(np.float32)
        else:
            _out_mono = (_out[:, 0] if _out.ndim == 2 else _out).astype(np.float32)

        _n = min(len(_ref_mono), len(_out_mono))
        if _n < 256:
            return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}
        _ref_mono = np.nan_to_num(_ref_mono[:_n], nan=0.0, posinf=0.0, neginf=0.0)
        _out_mono = np.nan_to_num(_out_mono[:_n], nan=0.0, posinf=0.0, neginf=0.0)

        _rms_ref = float(np.sqrt(np.mean(_ref_mono**2) + 1e-12))
        _rms_out = float(np.sqrt(np.mean(_out_mono**2) + 1e-12))
        if _rms_ref < 1e-5:
            return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}
        _rms_drop_db = float(20.0 * np.log10((_rms_ref + 1e-12) / (_rms_out + 1e-12)))

        # §9.11.2 RMS-envelope correlation (time-shift-tolerant).
        # 10 ms frames at 48 kHz = 480 samples per frame.
        _hop = max(256, min(480, _n // 16))
        _n_frames = _n // _hop
        if _n_frames >= 4:
            _ref_env = np.array(
                [np.sqrt(np.mean(_ref_mono[i * _hop : (i + 1) * _hop] ** 2) + 1e-12) for i in range(_n_frames)],
                dtype=np.float32,
            )
            _out_env = np.array(
                [np.sqrt(np.mean(_out_mono[i * _hop : (i + 1) * _hop] ** 2) + 1e-12) for i in range(_n_frames)],
                dtype=np.float32,
            )
            _std_env_ref = float(np.std(_ref_env))
            _std_env_out = float(np.std(_out_env))
            if _std_env_ref < 1e-7 or _std_env_out < 1e-7:
                _corr = 1.0 if abs(_std_env_ref - _std_env_out) < 1e-7 else 0.0
            else:
                _corr = float(np.corrcoef(_ref_env, _out_env)[0, 1])
                if not np.isfinite(_corr):
                    _corr = 0.0
        else:
            # Too few frames — fall back to sample correlation
            _std_ref = float(np.std(_ref_mono))
            _std_out = float(np.std(_out_mono))
            if _std_ref < 1e-7 or _std_out < 1e-7:
                _corr = 1.0 if abs(_std_ref - _std_out) < 1e-7 else 0.0
            else:
                _corr = float(np.corrcoef(_ref_mono, _out_mono)[0, 1])
                if not np.isfinite(_corr):
                    _corr = 0.0

        # Conservative thresholds: trigger only catastrophic changes.
        # skip_drop_check: LF-subtractive phases (rumble, hum) can lower broadband RMS
        # dramatically when noise dominates the signal — this is correct (§0 §2.45a).
        _drop_pen = 0.0 if skip_drop_check else max(0.0, min(1.0, (_rms_drop_db - 12.0) / 12.0))
        # skip_corr_check: timing phases with intentional global time-warp produce
        # low envelope correlation by design — do NOT penalise (§2.48/§2.54).
        _corr_pen = 0.0 if skip_corr_check else max(0.0, min(1.0, (0.55 - _corr) / 0.55))
        _penalty = float(max(_drop_pen, _corr_pen))
        return _penalty, {"rms_drop_db": _rms_drop_db, "corr": _corr}
    except Exception:
        return 0.0, {"rms_drop_db": 0.0, "corr": 1.0}


def _extract_sample(
    audio: np.ndarray,
    sr: int,
    duration_s: float = SAMPLE_DURATION_S,
    defect_locations: dict[str, list[tuple[float, float]]] | None = None,
    phase_id: str = "",
) -> np.ndarray:
    """Extrahiert repräsentative Stichprobe aus dem Audio.

    For dropout/transport-bump phases (§9.1a), the sample is centred on the
    first known defect location rather than the audio midpoint, so that the
    PMGG regression check actually evaluates the repaired region.

    For all other phases, the classic centre-crop is used.
    """
    n = len(audio)
    sample_len = min(int(duration_s * sr), n)
    if n <= sample_len:
        return audio

    # §9.1a Non-stationary defect targeting: centre the sample on the first
    # defect location for phases that repair sparse, localised defects.
    _SPARSE_DEFECT_PHASES = frozenset(
        {
            "phase_24",
            "phase_27",
            "phase_55",  # dropout_repair, diffusion_inpainting
            "phase_09",  # crackle_removal (event-based)
            "phase_01",  # click_removal (event-based)
        }
    )
    if defect_locations and any(phase_id.startswith(p) for p in _SPARSE_DEFECT_PHASES):
        # Find the first defect location relevant to this phase
        _PHASE_DEFECT_KEYS = {
            "phase_24": ("DROPOUTS", "DROPOUT", "dropouts"),
            "phase_27": ("DROPOUTS", "DROPOUT", "dropouts"),
            "phase_55": ("DROPOUTS", "DROPOUT", "dropouts", "SPECTRAL_HOLES", "spectral_holes"),
            "phase_09": ("CRACKLE", "crackle", "CLICKS", "clicks"),
            "phase_01": ("CLICKS", "clicks", "CLICK", "click"),
        }
        _keys = _PHASE_DEFECT_KEYS.get(next((p for p in _SPARSE_DEFECT_PHASES if phase_id.startswith(p)), ""), ())
        _best_start_s = None
        for _dk in _keys:
            locs = defect_locations.get(_dk, [])
            if locs and isinstance(locs[0], (tuple, list)) and len(locs[0]) >= 1:
                _best_start_s = float(locs[0][0])  # first defect location start time
                break
        if _best_start_s is not None:
            # Centre the sample window on the defect location
            _defect_sample = int(_best_start_s * sr)
            start = max(0, min(_defect_sample - sample_len // 2, n - sample_len))
            return audio[start : start + sample_len]

    # Default: centre-crop
    start = (n - sample_len) // 2
    return audio[start : start + sample_len]


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PerPhaseMusicalGoalsGate:
    """
    Wraps PhaseInterface.process() mit Musical-Goals-Prüfung.

    Alle Methoden sind thread-sicher und NaN/Inf-sicher.
    """

    def __init__(self) -> None:
        """Initialize PMGG with zeroed rollback counters."""
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
        phase_id: str | None = None,
        scores_before: dict[str, float] | None = None,
        phase_kwargs: dict[str, Any] | None = None,
        restorability_score: float = 70.0,
        applicable_goals: set[str] | None = None,
        initial_strength: float = 1.0,
        is_studio_2026: bool = False,
        goal_weights: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
        """
        Führt eine Phase aus und prüft Musical-Goals-Regression.

        Args:
            phase: PhaseInterface-Instanz mit process(audio) → PhaseResult
            audio: Input-Audio (float32)
            sr: 48000 Hz
            phase_id: Optional explicit phase id for backward-compatible callers.
                      If omitted, id is resolved from phase metadata.
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
            is_studio_2026: True if Studio 2026 mode (§9.10.77 Pareto-Differenzierung).
                            Selects higher P3–P5 thresholds. Default: Restoration.

        Returns:
            (audio_out, scores_after, log_entry)
        """
        if sr != 48000:
            logger.debug("PMGG: SR=%s (nicht 48000) — Goal-Messung läuft trotzdem", sr)

        if phase_kwargs is None:
            phase_kwargs = {}

        phase_id = phase_id or self._get_phase_id(phase)
        t0 = time.time()

        # §2.29/§2.54 Material- und Restorability-adaptiven Threshold bestimmen
        _mat_kw_thresh = (phase_kwargs or {}).get("material_type") or (phase_kwargs or {}).get("material")
        _mat_str_thresh = (
            (_mat_kw_thresh.value if hasattr(_mat_kw_thresh, "value") else str(_mat_kw_thresh)).lower()
            if _mat_kw_thresh
            else "unknown"
        )
        threshold = _get_adaptive_threshold(restorability_score, _mat_str_thresh)

        # §2.31a SongCal-Threshold-Feinjustage: global_scalar aus dem Song-Kalibrierungsprofil
        # erlaubt engere Schutzzone bei nahe-sauberem Audio und lockert sie bei stark
        # beschädigtem Material, um unnötige Retry-Zyklen zu vermeiden.
        _calpro_kw = (phase_kwargs or {}).get("song_calibration_profile", {})
        if isinstance(_calpro_kw, dict) and _calpro_kw:
            _gs = float(_calpro_kw.get("global_scalar", 1.0))
            if _gs < 0.85:
                # Near-clean: tighter threshold guards musical purity
                threshold = max(0.015, threshold * 0.85)
            elif _gs > 1.20:
                # Heavy damage: looser threshold reduces wasted retry cycles
                threshold = min(0.070, threshold * 1.15)

        # §9.7.3 Phasen-adaptive Sample-Dauer — MUSS vor scores_before bestimmt werden,
        # damit before und after dieselbe Sample-Länge nutzen (sonst falsche Regression).
        _sample_dur = _get_sample_duration(phase_id)

        # §9.1a Non-stationary: extract defect_locations for targeted sampling
        _defect_locs = (phase_kwargs or {}).get("defect_locations")

        # Vor-Scores messen (wenn nicht übergeben) — gleiche duration wie after-Messung
        sample_before = _extract_sample(
            audio, sr, duration_s=_sample_dur, defect_locations=_defect_locs, phase_id=phase_id
        )
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
        # §2.31b Material-adaptive exclusion relaxation (v9.10.85):
        # High-quality digital sources (cd_digital, dat) have no broadband hiss.
        # Noise-derived false-regression root-causes (brillanz/authentizitaet/
        # transparenz/tonal_center) do not apply to these materials.
        # Only CREPE-load-state (natuerlichkeit) and transient-shape mismatch
        # (artikulation) remain as stable, material-independent exclusions.
        if _excluded_goals:
            _mat_kw = (phase_kwargs or {}).get("material_type") or (phase_kwargs or {}).get("material")
            _mat_str = (_mat_kw.value if hasattr(_mat_kw, "value") else str(_mat_kw)) if _mat_kw else ""
            if _mat_str in {"cd_digital", "dat"} and (
                phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
            ):
                _excluded_goals &= {"natuerlichkeit", "artikulation"}
            # Analog-noise adaptive extension (2026-03-30): phase_03 on hiss-heavy
            # analog media can produce false timbre_authentizitaet regressions in the
            # short PMGG window although denoise improves perceptual quality.
            # Extended to phase_29 (2026-03-30): DeepFilterNet tape-hiss removal has
            # identical HF-removal → centroid-CV-disturbance mechanism as phase_03.
            # Both phases alter spectral-centroid variance on analog material where
            # hiss dominates HF → timbre proxy overreacts → false P2 cascade.
            if _mat_str in {"vinyl", "shellac", "tape", "reel_tape", "cassette"} and (
                phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
            ):
                _excluded_goals.add("timbre_authentizitaet")

        # §2.54 Team-Koordination: Folgephase berücksichtigt Vorphasen-Kontext.
        # Verhindert, dass PMGG Retry-Logik bewusst wiederhergestellte HF-Anteile
        # als Regression wertet (phase_50 nach phase_06/phase_07/phase_23).
        _team_policy = _resolve_team_context_policy(phase_id, phase_kwargs)
        _team_goal_exclusions = _team_policy.get("goal_exclusions")
        if isinstance(_team_goal_exclusions, set) and _team_goal_exclusions:
            _excluded_goals |= _team_goal_exclusions

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

        _team_threshold_mult = _team_policy.get("threshold_multiplier", 1.0)
        if isinstance(_team_threshold_mult, (int, float)) and float(_team_threshold_mult) > 1.0:
            _old_threshold = threshold
            threshold = min(0.090, float(threshold) * float(_team_threshold_mult))
            logger.debug(
                "PMGG team-policy: %s threshold %.3f -> %.3f (reason=%s)",
                phase_id,
                _old_threshold,
                threshold,
                _team_policy.get("reason", "unknown"),
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
            defect_locations=_defect_locs,
            is_studio_2026=is_studio_2026,
            goal_weights=goal_weights,
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

        # §2.29e Team-Telemetrie: Policyinformationen in log_entry.metadata schreiben
        # damit UV3 nach der Pipeline team_coordination_events extrahieren kann.
        _te_reason = str(_team_policy.get("reason", "") or "")
        if _te_reason:
            log_entry.metadata["team_policy_reason"] = _te_reason
            log_entry.metadata["team_excluded_goals"] = sorted(
                _team_goal_exclusions if isinstance(_team_goal_exclusions, set) else set()
            )
            log_entry.metadata["team_threshold_mult"] = round(float(_team_policy.get("threshold_multiplier", 1.0)), 3)
            log_entry.metadata["team_strength_cap"] = round(float(_team_policy.get("strength_cap", 1.0)), 3)

        # §TFS: Temporal Fine Structure coherence check for spectral-modification phases.
        # Measures whether sub-1.5 kHz instantaneous phase (pitch/binaural cues)
        # survives the restoration phase. Moore (2008): TFS encodes pitch perception,
        # binaural localisation, and consonant texture — invisible to envelope metrics.
        if any(phase_id.startswith(pfx) for pfx in _TFS_SENSITIVE_PHASES):
            try:
                from backend.core.tfs_preservation_guard import get_tfs_preservation_guard

                _tfs_guard = get_tfs_preservation_guard()
                # Use same sample duration as Musical Goals (consistency)
                _tfs_sample_before = _extract_sample(audio, sr, duration_s=min(_sample_dur, 2.5))
                _tfs_sample_after = _extract_sample(audio_out, sr, duration_s=min(_sample_dur, 2.5))
                _tfs_result = _tfs_guard.measure(_tfs_sample_before, _tfs_sample_after, sr)

                log_entry.metadata["tfs_coherence"] = round(_tfs_result.mean_coherence, 4)
                log_entry.metadata["tfs_min_coherence"] = round(_tfs_result.min_coherence, 4)
                log_entry.metadata["tfs_n_bands"] = _tfs_result.n_bands
                log_entry.metadata["tfs_passes"] = _tfs_result.passes_threshold

                if not _tfs_result.passes_threshold:
                    logger.warning(
                        "PMGG TFS: %s TFS coherence degraded (mean=%.4f < %.2f) — "
                        "phase may have disrupted temporal fine structure",
                        phase_id,
                        _tfs_result.mean_coherence,
                        _TFS_COHERENCE_THRESHOLD,
                    )
                else:
                    logger.info(
                        "PMGG TFS: %s coherence=%.4f (passes)",
                        phase_id,
                        _tfs_result.mean_coherence,
                    )
            except Exception as _tfs_exc:
                logger.debug("PMGG TFS: %s measurement failed: %s", phase_id, _tfs_exc)

        elapsed = time.time() - t0
        logger.debug(
            "PMGG: %s → %s (%.0f ms, strength=%.2f)",
            phase_id,
            action,
            elapsed * 1000,
            strength,
        )

        # §9.7.14 Wärme-Validierung: log waerme delta at INFO level so real-run
        # AMRB field validation of the reverb-invariant warmth-ratio proxy is
        # visible without enabling full debug logging.
        if "waerme" in effective_goals:
            _w_before = scores_before.get("waerme", float("nan"))
            _w_after = scores_after.get("waerme", float("nan"))
            _w_delta = (
                _w_after - _w_before
                if (
                    not (isinstance(_w_before, float) and _w_before != _w_before)
                    and not (isinstance(_w_after, float) and _w_after != _w_after)
                )
                else float("nan")
            )
            logger.info(
                "PMGG waerme §9.7.14  phase=%s  before=%.4f  after=%.4f  delta=%+.4f  action=%s  strength=%.2f",
                phase_id,
                _w_before,
                _w_after,
                _w_delta if not (isinstance(_w_delta, float) and _w_delta != _w_delta) else 0.0,
                action,
                strength,
            )

        # §2.47b: propagate sub_threshold marking into log_entry metadata
        if action == "sub_threshold":
            log_entry.metadata.setdefault("sub_threshold_phases", []).append(phase_id)

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
        defect_locations: dict[str, list[tuple[float, float]]] | None = None,
        is_studio_2026: bool = False,
        goal_weights: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, float], str, float]:
        """
        Führt Phase aus, ggf. mit Retry bei Regression.

        Args:
            threshold: Adaptiver REGRESSION_THRESHOLD (§2.29).
            effective_goals: Subset aus FAST_GOALS_SUBSET, das geprüft wird.
            sample_duration_s: Stichprobenlänge (§9.7.3 phasen-adaptiv, 1.0–5.0 s).
            initial_strength: Material-adaptive Initialstärke ∈ (0, 1.0] (§2.31).
                1.0 = Default. Retry-Stärken skalieren relativ dazu wenn < 1.0.
            goal_weights: §2.56 Song-specific goal importance weights.
                Per-goal float ∈ [0.3, 2.0]. weight > 1.0 = stricter threshold,
                weight < 1.0 = more lenient. None = uniform (1.0 for all).

        Returns:
            (audio_out, scores_after, action_label, strength_used)
        """
        if phase_kwargs is None:
            phase_kwargs = {}
        if effective_goals is None:
            effective_goals = FAST_GOALS_SUBSET
        initial_strength = max(0.01, min(1.0, initial_strength))
        _team_policy = _resolve_team_context_policy(phase_id, phase_kwargs)
        _team_cap = _team_policy.get("strength_cap", 1.0)
        if isinstance(_team_cap, (int, float)) and float(_team_cap) < 0.999:
            _old_strength = initial_strength
            initial_strength = min(initial_strength, float(_team_cap))
            if initial_strength + 1e-9 < _old_strength:
                logger.debug(
                    "PMGG team-cap: %s strength %.2f -> %.2f (reason=%s)",
                    phase_id,
                    _old_strength,
                    initial_strength,
                    _team_policy.get("reason", "unknown"),
                )
        _safe_cap = _phase_safe_strength_cap(phase_id, phase_kwargs)
        if _safe_cap < 0.999:
            _old_strength = initial_strength
            initial_strength = min(initial_strength, _safe_cap)
            if initial_strength + 1e-9 < _old_strength:
                logger.info(
                    "PMGG safe-cap: %s material=%s strength %.2f -> %.2f",
                    phase_id,
                    _material_key_from_phase_kwargs(phase_kwargs),
                    _old_strength,
                    initial_strength,
                )

        # §PMGG-Restorative: Für defektentfernende Phasen (denoise, dereverb, hiss,
        # hum, noise gate, dropout) deckeln wir scores_before auf normative Mindest-
        # schwellwerte. Defekte erhöhen Metriken künstlich über den sauberen Wert:
        # Rauschen füllt Spektraltäler → Authentizität SCHEINT hoch. Nach Denoise
        # sinkt der Score auf den echten Wert → PMGG würde false-positive P1-
        # Regression melden und die Phase auf 6% Wet drosseln.
        # Lösung: Baseline kann nie höher sein als das normative Qualitätsziel.
        # Echter Schaden (Score nach Phase UNTER Schwelle) wird weiterhin erkannt.
        # §2.29c §2.48a Architektur-Inversion: Ist diese Phase restorative?
        # Ableitung aus phase_ontology (intrinsischer Typ), nicht aus Ausnahmeliste.
        # Legacy-Fallback: _RESTORATIVE_PHASES für Phasen noch nicht im Ontologie-Register.
        from backend.core.phase_ontology import BASELINE_CAPPING_VALID_TYPES, get_phase_type

        _phase_type = get_phase_type(phase_id)
        _is_restorative = _phase_type in BASELINE_CAPPING_VALID_TYPES or any(
            phase_id.startswith(p) for p in _RESTORATIVE_PHASES
        )
        _thresholds = _get_canonical_thresholds(is_studio_2026)
        if _is_restorative:
            effective_scores_before = {g: min(v, _thresholds.get(g, v) + 0.05) for g, v in scores_before.items()}
            _capped_goals = [
                g for g in scores_before if scores_before[g] > _thresholds.get(g, scores_before[g]) + 0.001
            ]
            if _capped_goals:
                logger.debug(
                    "PMGG restorative baseline cap (%s): %s — defect-inflated scores capped at"
                    " canonical thresholds to prevent false-positive regressions",
                    phase_id,
                    {g: round(scores_before[g], 3) for g in _capped_goals},
                )
        else:
            effective_scores_before = scores_before

        # §2.29a ML-Inference-Caching: ML-deterministische Phasen werden nur
        # einmal mit strength=1.0 ausgeführt.  Retries variieren Wet/Dry-Blending.
        # Strength-abhängige DSP-Phasen müssen bei jedem Retry neu ausgeführt
        # werden, da strength dort Algorithmus-Parameter steuert (z.B. Filterfrequenz,
        # Kompressionsratio), nicht nur das Mischverhältnis.
        _is_ml_deterministic = phase_id.startswith(tuple(_ML_DETERMINISTIC_PHASES))
        # §2.29a Sonderfall phase_20: SGMSE+ (ML) ist deterministisch, aber WPE-DSP-Fallback
        # verwendet strength*0.90 als algorithmus-internen Prädiktor-Parameter → must re-run.
        # Zur Laufzeit: nur wenn SGMSE+ im ML-Budget alloziert ist, ML-Pfad verwenden.
        if _is_ml_deterministic and phase_id.startswith("phase_20"):
            _is_ml_deterministic = _phase20_is_ml_active()

        # §9.7.5 Referenz-Stichprobe für preservation-aware Messung.
        # Einmal berechnen, für alle scores_after/scores_retry wiederverwenden.
        _defect_locs = defect_locations
        _ref_sample = _extract_sample(
            audio, sr, duration_s=sample_duration_s, defect_locations=_defect_locs, phase_id=phase_id
        )

        if _is_ml_deterministic:
            # ML-Pfad: Einmalige Inferenz mit strength=1.0, Wet/Dry für Stärke
            audio_full = self._run_phase(phase, audio, 1.0, phase_kwargs)
            if initial_strength < 1.0:
                audio_out = self._wet_dry_blend(audio, audio_full, initial_strength, phase)
            else:
                audio_out = audio_full
        else:
            # DSP-Pfad: Direkte Ausführung mit material-adaptiver Stärke
            audio_out = self._run_phase(phase, audio, initial_strength, phase_kwargs)
            audio_full = None  # kein Cache benötigt

        # §2.45/§2.54 Passthrough-Erkennung: Phasen die kein Pitch/Defekt finden geben
        # das Audio bit-identisch zurück (z.B. phase_31 bei CREPE confidence=0.0).
        # In diesem Fall: kein Goal-Scoring, kein Retry, kein StrictConflictDecay.
        # np.array_equal ist exakt + schnell (kein float-Toleranz-Drift).
        if np.array_equal(audio, audio_out):
            logger.debug(
                "PMGG %s: audio_out identisch mit input (passthrough) — direkt passed, kein Retry",
                phase_id,
            )
            return audio_out, scores_before, "passed", initial_strength

        scores_after = _measure_quick(
            _extract_sample(
                audio_out, sr, duration_s=sample_duration_s, defect_locations=_defect_locs, phase_id=phase_id
            ),
            sr,
            reference=_ref_sample,
        )

        regression = self._max_regression(
            effective_scores_before, scores_after, effective_goals, goal_weights=goal_weights
        )
        _skip_corr = phase_id in _TIMING_CORR_EXCLUDE
        _skip_drop = phase_id in _LF_SUBTRACTIVE_DROP_SKIP
        _ci_penalty, _ci_meta = _content_integrity_penalty(
            audio, audio_out, skip_corr_check=_skip_corr, skip_drop_check=_skip_drop
        )
        if _ci_penalty > 0.0:
            # Force retry path for catastrophic content loss even when many goals are excluded.
            regression = max(regression, threshold + 0.001 + 0.05 * _ci_penalty)
            logger.warning(
                "PMGG Content-Guard: %s triggered (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                phase_id,
                _ci_meta.get("rms_drop_db", 0.0),
                _ci_meta.get("corr", 1.0),
                _ci_penalty,
            )

        # §2.47b JND Sub-Threshold Check: if all applicable goal-deltas are ≥ 0 and < JND
        # → phase produces no perceptually detectable improvement → accept but mark sub_threshold
        # VERBOTEN: sub-threshold logic must NOT fire when _ci_penalty > 0 (content loss)
        if _ci_penalty == 0.0:
            _applicable_jnd = [g for g in effective_goals if g in effective_scores_before and g in scores_after]
            if _applicable_jnd:
                _deltas = {g: scores_after[g] - effective_scores_before[g] for g in _applicable_jnd}
                _all_below_jnd = all(d >= 0.0 for d in _deltas.values()) and all(
                    abs(d) < JND_MIN_DELTA.get(g, 0.015) for g, d in _deltas.items()
                )
                if _all_below_jnd:
                    logger.debug(
                        "PMGG %s: sub_threshold — all %d goal-deltas ≥ 0 and < JND, accepting",
                        phase_id,
                        len(_applicable_jnd),
                    )
                    return audio_out, scores_after, "sub_threshold", initial_strength

        if regression <= threshold:
            return audio_out, scores_after, "passed", initial_strength

        # §2.29 v9.10.77: Priority-aware regression check.
        # Determine worst priority among regressed goals to set retry budget.
        _reg_pa, _worst_prio = self._max_regression_priority_aware(
            effective_scores_before, scores_after, effective_goals, threshold, goal_weights=goal_weights
        )

        # Log which goal caused the regression (diagnostics for false-positive detection)
        _worst_goal = max(
            effective_goals,
            key=lambda g: max(0.0, effective_scores_before.get(g, 0.5) - scores_after.get(g, 0.5)),
        )
        if _ci_penalty > 0.0:
            _worst_prio = min(_worst_prio, 2)
            _worst_goal = "content_integrity_guard"
        logger.debug(
            "PMGG: %s regression=%.4f > threshold=%.3f — worst goal: %s (P%d, before=%.3f after=%.3f)",
            phase_id,
            regression,
            threshold,
            _worst_goal,
            _worst_prio,
            effective_scores_before.get(_worst_goal, 0.5),
            scores_after.get(_worst_goal, 0.5),
        )

        # §2.29 v9.10.77: If ONLY P4/P5 goals regressed (priority-adjusted threshold
        # not exceeded), skip retries entirely — these are best-effort goals.
        if _worst_prio >= 4:
            logger.info(
                "PMGG: %s regression only in P%d goals (%s) — no retry (best-effort priority)",
                phase_id,
                _worst_prio,
                _worst_goal,
            )
            log_action = "passed_p4p5_tolerated"
            return audio_out, scores_after, log_action, initial_strength

        # Priority-based max retries (§2.29 v9.10.77):
        _max_retries_for_prio = _PRIORITY_MAX_RETRIES.get(_worst_prio, 4)

        # §2.31a SongCal P3-Retry-Feinjustage: restorability_tier moduliert den
        # Retry-Etat für P3-Ziele (Groove, MicroDynamics, Emotionalität).
        # Good-Material: 2 → 3 Retries (stabil genug, um Verbesserung rauszuholen).
        # Poor-Material:  2 → 1 Retry  (P3-Regressionen oft unabwendbar — Zeit sparen).
        # P1/P2/P4/P5 bleiben unverändert.
        if _worst_prio == 3:
            _cal_p3 = (phase_kwargs or {}).get("song_calibration_profile", {})
            if isinstance(_cal_p3, dict) and _cal_p3:
                _rtier = _cal_p3.get("restorability_tier", "fair")
                if _rtier == "good":
                    _max_retries_for_prio = min(4, _max_retries_for_prio + 1)  # 2 → 3
                elif _rtier == "poor":
                    _max_retries_for_prio = max(1, _max_retries_for_prio - 1)  # 2 → 1

        # Retry-Stärken relativ zur Initialstärke skalieren (§2.29):
        # initial_strength=1.0 → normale Retry-Folge [0.65, 0.50, ...]
        # initial_strength<1.0 → proportional nach unten skaliert
        # §2.31a SongCal: Wurde initial_strength bereits durch Kalibrierung reduziert
        # (< 0.90), sanftere Ankerpunkte verwenden um Doppel-Reduktion zu vermeiden.
        # Beispiel: initial=0.70 → alt: 0.65×0.70=0.455; neu: 0.80×0.70=0.560
        if initial_strength < 0.90:
            _retry_anchors = [0.80, 0.65, 0.50, 0.35, 0.20]
        else:
            _retry_anchors = list(_RETRY_STRENGTHS)
        retry_strengths = [s * initial_strength for s in _retry_anchors[:_max_retries_for_prio]]

        # §2.29 Best-Effort-Tracking: Speichere den Versuch mit geringster Regression.
        # PMGG darf Phasen NICHT überspringen — CausalDefectReasoner hat die Phase
        # als notwendig bestimmt. Stattdessen wird der beste Versuch verwendet.
        best_audio = audio_out
        best_scores = scores_after
        best_regression = regression
        best_strength = initial_strength
        best_action = "best_effort"

        # §2.29a Fix: ML-deterministische Timing-Phasen (phase_12, phase_31)
        # können NICHT per Wet/Dry retried werden, da Timing-Phasen kein Blending
        # erlauben (Phasen-Artefakte bei Crossfade zeitversetzter Signale).
        # Alle Retries würden identisches Audio produzieren → sofort Best-Effort.
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        if _is_ml_deterministic and phase_id in _TIMING_PHASES:
            logger.info(
                "PMGG: %s is ML-deterministic timing phase — Wet/Dry retries not applicable, "
                "using best-effort (regression=%.4f > threshold=%.3f)",
                phase_id,
                regression,
                threshold,
            )
            return best_audio, best_scores, "best_effort", initial_strength

        # Retry-Schleife
        # ML-deterministische Phasen: Wet/Dry-Reblend des gecachten audio_full
        #   (spart ~60 s pro Retry bei OMLSA + ResembleEnhance etc.)
        # DSP-Phasen: Erneuter process()-Aufruf mit geändertem strength
        #   (nichtlineare DSP-Operationen: wet/dry ≠ Neuberechnung)
        _prev_regression = regression
        _retry_t0 = time.time()
        _RETRY_BUDGET_S = 300.0  # Max 5 min für alle Retries einer Phase
        for attempt, strength in enumerate(retry_strengths):
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

            import gc

            gc.collect()

            action_label = f"retry{attempt + 1}"

            if _is_ml_deterministic:
                # §2.29a: Wet/Dry-Reblend — keine erneute ML-Inferenz
                logger.debug(
                    "PMGG: %s Retry %d mit strength=%.2f (Wet/Dry-Reblend, keine Re-Inferenz)",
                    phase_id,
                    attempt + 1,
                    strength,
                )
                audio_retry = self._wet_dry_blend(audio, audio_full, strength, phase)
            else:
                # DSP-Phase: Neu ausführen mit reduziertem strength
                logger.debug(
                    "PMGG: %s Retry %d mit strength=%.2f (DSP Re-Run)",
                    phase_id,
                    attempt + 1,
                    strength,
                )
                audio_retry = self._run_phase(phase, audio, strength, phase_kwargs)

            _retry_sample = _extract_sample(
                audio_retry,
                sr,
                duration_s=sample_duration_s,
                defect_locations=_defect_locs,
                phase_id=phase_id,
            )
            scores_retry = _measure_quick(_retry_sample, sr, reference=_ref_sample, precise_override=False)
            regression_retry = self._max_regression(
                effective_scores_before, scores_retry, effective_goals, goal_weights=goal_weights
            )
            _ci_penalty_retry, _ci_meta_retry = _content_integrity_penalty(audio, audio_retry)
            if _ci_penalty_retry > 0.0:
                regression_retry = max(regression_retry, threshold + 0.001 + 0.05 * _ci_penalty_retry)
                logger.debug(
                    "PMGG Content-Guard retry: %s r%d (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                    phase_id,
                    attempt + 1,
                    _ci_meta_retry.get("rms_drop_db", 0.0),
                    _ci_meta_retry.get("corr", 1.0),
                    _ci_penalty_retry,
                )
            if regression_retry <= threshold:
                # Apply precise overrides once for accurate score propagation to next phase
                scores_retry = _apply_precise_metric_overrides(scores_retry, _retry_sample, sr, reference=_ref_sample)
                return audio_retry, scores_retry, action_label, strength
            # Track best attempt (lowest regression)
            if regression_retry < best_regression:
                best_audio = audio_retry
                best_scores = scores_retry
                best_regression = regression_retry
                best_strength = strength
                best_action = f"best_effort_r{attempt + 1}"

            # Stagnation guard: if regression barely changes across consecutive
            # retries despite strength variation, further retries are wasted.
            # §2.31a: Stagnation-Delta proportional zum Threshold — armes Material
            # (höherer Threshold) bricht früher ab; gutes Material (niedriger Threshold)
            # ist geduldiger (wartet auf kleinere Verbesserungen).
            _stagnation_delta = max(0.002, threshold * 0.15)
            if abs(regression_retry - _prev_regression) < _stagnation_delta and attempt >= 1:
                logger.info(
                    "PMGG: %s stagnation detected at retry %d (Δregression=%.6f) — skipping remaining retries",
                    phase_id,
                    attempt + 1,
                    abs(regression_retry - _prev_regression),
                )
                break
            _prev_regression = regression_retry

        # §2.31b Dynamic catastrophic threshold (v9.10.85):
        # Proportional to adaptive threshold so Good material (0.020) triggers
        # emergency retries at 0.08 — earlier quality protection. Poor material
        # (0.055) produces 0.22, matching the old hard-coded value.
        # Floor 0.08 prevents über-aggressive cascades on very clean material.
        _CATASTROPHIC_THRESHOLD = max(0.08, 4.0 * threshold)
        _team_thr_mult = _team_policy.get("threshold_multiplier", 1.0)
        if isinstance(_team_thr_mult, (int, float)) and float(_team_thr_mult) > 1.0:
            _CATASTROPHIC_THRESHOLD = min(0.25, _CATASTROPHIC_THRESHOLD * float(_team_thr_mult))

        _EMERGENCY_STRENGTHS = [0.15 * initial_strength, 0.10 * initial_strength]
        if _allow_emergency_retries(
            phase_id,
            _worst_prio,
            best_regression,
            _CATASTROPHIC_THRESHOLD,
            _team_policy,
        ):
            logger.warning(
                "PMGG: %s catastrophic regression %.4f > %.2f (worst goal: %s P%d) — attempting emergency low-strength retries",
                phase_id,
                best_regression,
                _CATASTROPHIC_THRESHOLD,
                _worst_goal,
                _worst_prio,
            )
            for _em_strength in _EMERGENCY_STRENGTHS:
                _retry_elapsed = time.time() - _retry_t0
                if _retry_elapsed > _RETRY_BUDGET_S:
                    break
                if _is_ml_deterministic:
                    audio_em = self._wet_dry_blend(
                        audio, audio_full if audio_full is not None else best_audio, _em_strength, phase
                    )
                else:
                    audio_em = self._run_phase(phase, audio, _em_strength, phase_kwargs)
                _em_sample = _extract_sample(
                    audio_em,
                    sr,
                    duration_s=sample_duration_s,
                    defect_locations=_defect_locs,
                    phase_id=phase_id,
                )
                scores_em = _measure_quick(_em_sample, sr, reference=_ref_sample, precise_override=False)
                regression_em = self._max_regression(
                    effective_scores_before, scores_em, effective_goals, goal_weights=goal_weights
                )
                _ci_penalty_em, _ci_meta_em = _content_integrity_penalty(audio, audio_em)
                if _ci_penalty_em > 0.0:
                    regression_em = max(regression_em, threshold + 0.001 + 0.05 * _ci_penalty_em)
                    logger.debug(
                        "PMGG Content-Guard emergency: %s (rms_drop=%.2f dB corr=%.3f penalty=%.3f)",
                        phase_id,
                        _ci_meta_em.get("rms_drop_db", 0.0),
                        _ci_meta_em.get("corr", 1.0),
                        _ci_penalty_em,
                    )
                if regression_em <= threshold:
                    if audio_full is not None:
                        del audio_full
                    scores_em = _apply_precise_metric_overrides(scores_em, _em_sample, sr, reference=_ref_sample)
                    return audio_em, scores_em, f"emergency_s{_em_strength:.2f}", _em_strength
                if regression_em < best_regression:
                    best_audio = audio_em
                    best_scores = scores_em
                    best_regression = regression_em
                    best_strength = _em_strength
                    best_action = "best_effort_emergency"
        elif best_regression > _CATASTROPHIC_THRESHOLD and _worst_prio <= 2:
            logger.info(
                "PMGG: %s catastrophic path skipped by team policy (reason=%s, regression=%.4f, threshold=%.3f)",
                phase_id,
                _team_policy.get("reason", "none") if isinstance(_team_policy, dict) else "none",
                best_regression,
                _CATASTROPHIC_THRESHOLD,
            )

        # §2.29 KEIN Rollback — Phase wird mit geringster Regression angewendet.
        # VERBOTEN: Phase überspringen (Original-Audio zurückgeben).
        # CausalDefectReasoner hat diese Phase als notwendig bestimmt.
        # Sofortige Freigabe: audio_full (+86 MB bei 225s) nicht bis GC halten.
        if audio_full is not None:
            del audio_full
        # Apply precise overrides once for accurate score propagation to next phase
        _best_sample = _extract_sample(
            best_audio,
            sr,
            duration_s=sample_duration_s,
            defect_locations=_defect_locs,
            phase_id=phase_id,
        )
        best_scores = _apply_precise_metric_overrides(best_scores, _best_sample, sr, reference=_ref_sample)
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
            if out.shape[0] != audio.shape[0]:
                target_len = int(audio.shape[0])
                if out.shape[0] > target_len:
                    out = out[:target_len, ...]
                else:
                    pad_rows = target_len - int(out.shape[0])
                    pad_spec = [(0, pad_rows)] + [(0, 0)] * (max(out.ndim, 1) - 1)
                    out = np.pad(out, pad_spec)

            # Wet/Dry-Modulation: strength < 1.0 → blende zwischen Original und Verarbeitet
            if 0.0 < strength < 1.0:
                phase_id = ""
                try:
                    meta = phase.get_metadata()
                    phase_id = getattr(meta, "phase_id", "")
                except Exception as _meta_exc:
                    logger.debug("PMGG: Phase-Metadata-Zugriff fehlgeschlagen: %s", _meta_exc)
                if phase_id not in _TIMING_PHASES:
                    out = (audio + strength * (out - audio)).astype(np.float32)
                    out = np.clip(out, -1.0, 1.0)

            return out
        except Exception as exc:
            logger.debug("PMGG: Phase-Ausführung fehlgeschlagen: %s", exc)
            return audio

    @staticmethod
    def _wet_dry_blend(
        dry: np.ndarray,
        wet: np.ndarray,
        strength: float,
        phase: Any = None,
    ) -> np.ndarray:
        """Phase-aware Wet/Dry-Blending (§9.10.118 — Kammfilter-Schutz).

        Bei niedrigen Strengths (< 0.30) erzeugt lineare Zeitdomänen-
        Interpolation Kammfilter-Artefakte (Original + phasenverschobenes
        Signal). Stattdessen: STFT-Magnitude-Interpolation mit bewahrter
        Originalphase.

        Wissensch. Basis: Wiener-Filtertheorie — Magnitude-Blending bei
        erhaltener Phase minimiert perceptual distortion (Ephraim & Malah
        1984).  Lineare Interpolation bleibt bei strength >= 0.30 (Kammfilter
        dort vernachlässigbar da Wet-Anteil dominiert).

        Timing-modifizierende Phasen (wow/flutter, speed) sind ausgenommen,
        da Crossfade zeitversetzter Signale Phasen-Artefakte erzeugt.
        """
        _TIMING_PHASES = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        dry = np.asarray(dry, dtype=np.float32)
        wet = np.asarray(wet, dtype=np.float32)

        def _match_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
            if x.shape[0] == target_len:
                return x
            if x.shape[0] > target_len:
                return x[:target_len, ...]
            pad_rows = target_len - int(x.shape[0])
            pad_spec = [(0, pad_rows)] + [(0, 0)] * (max(x.ndim, 1) - 1)
            return np.pad(x, pad_spec)

        # Time axis must always match before blending.
        wet = _match_time_axis(wet, int(dry.shape[0]))
        if strength >= 1.0:
            return np.clip(wet, -1.0, 1.0).astype(np.float32)
        if strength <= 0.0:
            return dry.copy()
        # Timing-Phasen: kein Blend
        phase_id = ""
        if phase is not None:
            try:
                meta = phase.get_metadata()
                phase_id = getattr(meta, "phase_id", "")
            except Exception as _meta_exc:
                logger.debug("PMGG: Wet/Dry-Blend Phase-Metadata-Zugriff fehlgeschlagen: %s", _meta_exc)
        if phase_id in _TIMING_PHASES:
            return np.clip(wet, -1.0, 1.0).astype(np.float32)

        # Stereo-safe handling: never run STFT blend on channel axis.
        if dry.ndim == 2 or wet.ndim == 2:
            if dry.ndim != 2 or wet.ndim != 2:
                logger.debug(
                    "PMGG Wet/Dry-Blend ndim mismatch dry=%s wet=%s; using linear fallback",
                    dry.shape,
                    wet.shape,
                )
                if dry.ndim == 2 and wet.ndim == 1:
                    wet = np.tile(wet[:, None], (1, dry.shape[1]))
                elif dry.ndim == 1 and wet.ndim == 2:
                    wet = wet.mean(axis=1)
                out_lin = (dry + strength * (wet - dry)).astype(np.float32)
                return np.clip(out_lin, -1.0, 1.0)

            if dry.shape[1] != wet.shape[1]:
                logger.debug(
                    "PMGG Wet/Dry-Blend channel mismatch dry=%s wet=%s; using linear fallback",
                    dry.shape,
                    wet.shape,
                )
                n_ch = min(dry.shape[1], wet.shape[1])
                out = dry.copy()
                out[:, :n_ch] = dry[:, :n_ch] + strength * (wet[:, :n_ch] - dry[:, :n_ch])
                return np.clip(out.astype(np.float32), -1.0, 1.0)

            if strength < 0.30 and dry.shape[0] >= 2048:
                ch_out = []
                for ch in range(dry.shape[1]):
                    ch_out.append(
                        PerPhaseMusicalGoalsGate._wet_dry_blend(
                            dry[:, ch],
                            wet[:, ch],
                            strength,
                            phase=None,
                        )
                    )
                return np.clip(np.stack(ch_out, axis=1).astype(np.float32), -1.0, 1.0)

            out = (dry + strength * (wet - dry)).astype(np.float32)
            return np.clip(out, -1.0, 1.0)

        # §9.10.118: phase-aware STFT blending for low strengths to prevent
        # comb-filter artifacts from time-domain mixing of phase-shifted signals.
        if strength < 0.30 and len(dry) >= 2048:
            try:
                win_size = 2048
                hop = win_size // 4
                from scipy.signal import istft as _istft
                from scipy.signal import stft as _stft

                _, _, Zxx_dry = _stft(dry, fs=48000, nperseg=win_size, noverlap=win_size - hop)
                _, _, Zxx_wet = _stft(wet, fs=48000, nperseg=win_size, noverlap=win_size - hop)

                # §2.43 Phase-Preserved Wet/Dry-Blend:
                # M_blend = (1−α)·M_dry + α·M_wet, Phase vom Wet-Signal
                mag_dry = np.abs(Zxx_dry)
                mag_wet = np.abs(Zxx_wet)
                phase_wet = np.angle(Zxx_wet)

                mag_blend = mag_dry + strength * (mag_wet - mag_dry)
                Zxx_blend = mag_blend * np.exp(1j * phase_wet)

                _, out = _istft(Zxx_blend, fs=48000, nperseg=win_size, noverlap=win_size - hop)
                # Length matching
                if len(out) > len(dry):
                    out = out[: len(dry)]
                elif len(out) < len(dry):
                    out = np.pad(out, (0, len(dry) - len(out)))
                return np.clip(out.astype(np.float32), -1.0, 1.0)
            except Exception as _stft_exc:
                logger.debug("PMGG STFT-Blend fallback to linear: %s", _stft_exc)

        out = (dry + strength * (wet - dry)).astype(np.float32)
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _max_regression(
        before: dict[str, float],
        after: dict[str, float],
        goals: list | None = None,
        goal_weights: dict[str, float] | None = None,
    ) -> float:
        """Maximale negative Differenz in Musical Goals (positiv = Regression).

        §2.56 Song-specific goal weighting: if goal_weights is provided,
        each goal's regression is multiplied by its weight before taking the max.
        weight > 1.0 → regression is amplified (stricter for important goals).
        weight < 1.0 → regression is dampened (lenient for less relevant goals).
        """
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET
        max_reg = 0.0
        for g in check_goals:
            delta = after.get(g, 0.5) - before.get(g, 0.5)
            if delta < 0:
                raw_reg = -delta
                # §2.56: Apply song-specific weight
                w = goal_weights.get(g, 1.0) if goal_weights else 1.0
                weighted_reg = raw_reg * w
                max_reg = max(max_reg, weighted_reg)
        return max_reg

    @staticmethod
    def _max_regression_priority_aware(
        before: dict[str, float],
        after: dict[str, float],
        goals: list | None = None,
        threshold: float = 0.020,
        goal_weights: dict[str, float] | None = None,
    ) -> tuple[float, int]:
        """Priority-aware regression: returns (max_regression, worst_priority).

        Only considers goals whose priority-adjusted threshold is exceeded.
        Returns the highest priority level (lowest number) among regressed goals.

        §2.56: goal_weights modulate the effective threshold per goal.
        weight > 1.0 → effective threshold is lower (stricter for important goals).
        weight < 1.0 → effective threshold is higher (lenient).

        Args:
            before: Scores before phase.
            after: Scores after phase.
            goals: Subset of goals to check.
            threshold: Base regression threshold.
            goal_weights: Per-goal importance weights ∈ [0.3, 2.0].

        Returns:
            (max_regression_value, worst_priority) where worst_priority is 1–5
            (1 = most critical). Returns (0.0, 99) if no regression detected.
        """
        from backend.core.goal_priority_protocol import get_goal_priority_protocol

        gpp = get_goal_priority_protocol()
        check_goals = goals if goals is not None else FAST_GOALS_SUBSET
        max_reg = 0.0
        worst_prio = 99
        for g in check_goals:
            delta = after.get(g, 0.5) - before.get(g, 0.5)
            if delta < 0:
                raw_reg = -delta
                # §2.56: weight amplifies regression for important goals
                w = goal_weights.get(g, 1.0) if goal_weights else 1.0
                weighted_reg = raw_reg * w
                prio = gpp.priority_of(g)
                prio_threshold = threshold * _PRIORITY_THRESHOLD_FACTOR.get(prio, 1.0)
                if weighted_reg > prio_threshold:
                    if prio < worst_prio:
                        worst_prio = prio
                    max_reg = max(max_reg, weighted_reg)
        return max_reg, worst_prio

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
    phase_id: str | None = None,
    scores_before: dict[str, float] | None = None,
    restorability_score: float = 70.0,
    applicable_goals: set[str] | None = None,
    is_studio_2026: bool = False,
    goal_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], PhaseGateLogEntry]:
    """
    Convenience-Wrapper: Führt eine Phase aus mit Musical-Goals-Schutz.

    Args:
        phase: PhaseInterface-Instanz
        audio: Input-Audio (float32, 48 kHz)
        sr: 48000 Hz
        phase_id: Optional explicit phase id for backward-compatible callers.
        scores_before: Vorherige Goal-Scores (optional)
        restorability_score: RestorabilityEstimator-Score ∈ [0, 100], bestimmt
                             adaptiven REGRESSION_THRESHOLD (§2.29).
        applicable_goals: Aus GoalApplicabilityFilter — nur diese Ziele geprüft.
        is_studio_2026: True for Studio 2026 mode (higher P3–P5 thresholds).
        goal_weights: §2.56 Song-specific goal importance weights.

    Returns:
        (audio_out, scores_after, log_entry)
    """
    return get_phase_gate().wrap_phase(
        phase,
        audio,
        sr,
        phase_id=phase_id,
        scores_before=scores_before,
        restorability_score=restorability_score,
        applicable_goals=applicable_goals,
        is_studio_2026=is_studio_2026,
        goal_weights=goal_weights,
    )
