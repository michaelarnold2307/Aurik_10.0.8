"""
Aurik 9 — Kumulative-Phasen-Interaktions-Guard §2.48 [RELEASE_MUST]
====================================================================
Guards against destructive cumulative effects of multiple phases:
- P1/P2 cumulative drift monitoring with material-adaptive rollback (§2.54)
- Critical phase pair interaction detection
- STFT phase coherence after ≥3 STFT phases
- Checkpoint management with adaptive consecutive-rollback limit

The guard is a **Notbremse** (emergency brake), not the routine pipeline steering.
Routine steering is handled by PhaseConductor (§2.52), PMGG (§2.29), and
SongCalibration (§2.47). Drift tolerances are COMPUTED from the concrete
song context — never hard-coded constants (§2.54).

Reference: Spec 02 §2.48, §2.54 (Adaptives Phasen-Optimum)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from backend.core.phase_ontology import (
    BASELINE_CAPPING_VALID_TYPES,
    GDD_VALID_TYPES,
    P1P2_DRIFT_CHECK_INVALID_TYPES,
    get_phase_type,
)

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────
_instance: CumulativeInteractionGuard | None = None
_lock = threading.Lock()


def get_interaction_guard() -> CumulativeInteractionGuard:
    """Thread-safe Singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CumulativeInteractionGuard()
    return _instance


# ── Constants ──────────────────────────────────────────────────────────────

# P1/P2 goals (§2.29d — hard regime, no degradation allowed)
# WICHTIG: Keys müssen mit measure_all()-Output von MusicalGoalsChecker
# übereinstimmen (deutsch). Englische Keys (naturalness, authenticity …)
# führten zu stiller Nicht-Prüfung aller Goals außer tonal_center.
P1_P2_GOALS = frozenset(
    {
        "natuerlichkeit",  # Natürlichkeit (Ziel P1, Schwellwert 0.90)
        "authentizitaet",  # Authentizität  (Ziel P1, Schwellwert 0.88)
        "tonal_center",  # TonalCenter    (Ziel P2, Schwellwert 0.95)
        "timbre_authentizitaet",  # Timbre         (Ziel P2, Schwellwert 0.87)
        "artikulation",  # Artikulation   (Ziel P2, Schwellwert 0.85)
    }
)

# §2.29c: tonal_center wird durch Breitbandrauschen künstlich erhöht
# (gleichmäßige Chroma-Lifts). Nach Denoise sinkt der Wert auf den echten
# musikimmanenten Level — das ist KEIN Artefakt, sondern Entlarvung des
# Rausch-Inflationseffekts. Für SUBTRACTIVE-Phasen daher aus Drift-Check
# ausschließen. PMGG §2.29c regelt die Per-Phase-Messung mit Baseline-Capping.
_DEFECT_INFLATED_SUBTRACTIVE_GOALS: frozenset[str] = frozenset(
    {
        "tonal_center",
    }
)

# Early carrier-cleanup phases can show temporary authenticity-proxy drops even when
# they only remove non-musical low-frequency defects (DC offset, subsonic rumble).
# Guard against false-positive rollbacks for this specific pattern.
#
# §2.44 Referenz-Paradoxon: Phases that correct time-domain carrier defects (wow/flutter,
# dropout, spectral inpainting) change chroma/centroid signatures vs. the defective
# checkpoint — AuthentizitaetMetric legitimately drops because it compares against the
# still-damaged rollback anchor (phase_30).  These are **not** quality regressions;
# they are artefacts of measuring corrected audio against corrupted reference.
# Excluding "authentizitaet" here prevents false rollbacks while PMGG still guards
# hard P1/P2 goals (natuerlichkeit, artikulation) as real signals.
_PHASE_SPECIFIC_DRIFT_EXCLUSIONS: dict[str, frozenset[str]] = {
    # Prefix-based keys (startswith), robust for "phase_30" and "phase_30_dc_offset_removal".
    "phase_30": frozenset({"authentizitaet", "natuerlichkeit"}),
    "phase_05": frozenset({"authentizitaet", "natuerlichkeit"}),
    # Wow/flutter correction shifts chroma AND amplitude envelope vs. defective checkpoint:
    # - authentizitaet: chromagram correlation drops because pitch-wobble artefacts are removed
    # - natuerlichkeit: spectral/temporal consistency changes vs. wow-distorted reference
    # - artikulation: energy envelope aligns differently once periodic pitch wobble is removed
    # - timbre_authentizitaet: spectral centroid/MFCC shifts as periodic pitch-wobble is removed (same §2.44 mechanism)
    # - tonal_center: K-S chroma unstable vs. wow-distorted checkpoint (pitch deviation corrupts chroma bins)
    # All five are reference-paradoxon false positives (§2.44, §2.47).
    "phase_12": frozenset(
        {"authentizitaet", "natuerlichkeit", "artikulation", "timbre_authentizitaet", "tonal_center"}
    ),
    # Dropout repair fills silent gaps → onset profiles / spectral continuity differ
    # vs. checkpoint with gaps:
    # - authentizitaet: chromagram changes because silence regions are now filled
    # - artikulation: energy envelope shape changes as dropouts are repaired
    # - natuerlichkeit: continuity increases vs. gap-corrupted anchor (reference paradox)
    # - timbre_authentizitaet: AudioSR fills gaps with synthesized spectral content → MFCC-Pearson +
    #   spectral centroid-CV change vs. dropout-bearing reference (same root as phase_23 AudioSR).
    # Dropout repair (AudioSR): fills silent dropout gaps → chroma, onset-density and crest-factor
    # all shift vs. the dropout-bearing checkpoint (§2.44 Reference-Paradoxon).
    # tonal_center: dropout silence has near-zero chroma → K-S unstable in checkpoint; after synthesis
    # K-S locks onto new estimate → cumulative tonal drift is measurement artefact, not regression.
    # groove: onset-density increase in repaired gaps → cumulative autocorr[lag_05] drift expected.
    "phase_24": frozenset(
        {"authentizitaet", "artikulation", "natuerlichkeit", "tonal_center", "groove", "timbre_authentizitaet"}
    ),
    # Diffusion inpainting reconstructs masked regions → spectral fingerprint mismatch
    # against corrupted reference is expected and not a real identity violation.
    # §2.44 Reference Paradox FULL SCOPE (aligns with PMGG 7-goal exclusion for phase_55):
    # natuerlichkeit: CQTdiff+ fills BW-loss gaps with synthesized HF content → MFCC-smoothness
    #   proxy measures synthesized spectrum vs. band-limited reference (same root cause as phase_23).
    # tonal_center: synthesized HF bins shift K-S key-detection; pre-inpainting vinyl-bw-limited
    #   audio has near-zero energy in high-register chroma bins → K-S unstable (confirmed Δ=0.8333, 2026-04-10).
    # timbre_authentizitaet: synthesized spectral content has different MFCC-Pearson + centroid-CV
    #   vs. band-limited reference (identical mechanism to phase_23 AudioSR gap-fill).
    # artikulation: diffusion synthesis inserts new spectral content where reference has damaged/missing
    #   content → transient-shape correlation vs. pre-inpainting fragment is meaningless.
    # natuerlichkeit confirmed mismatch: PMGG excluded it, CIG did not → cumulative nat drift
    #   accumulated through phase_55 → spurious CIG rollback at later non-excluded phase (e.g. phase_61).
    "phase_55": frozenset(
        {"authentizitaet", "natuerlichkeit", "tonal_center", "timbre_authentizitaet", "artikulation"}
    ),
    # §2.44 Reference-Paradoxon + §2.46 Carrier-Chain-Inversion:
    # All carrier-defect-removal phases change the chromagram/spectral fingerprint vs.
    # the still-damaged checkpoint.  authentizitaet measures correlation against the
    # degraded reference — diverging IS the goal, not a regression.
    # Click removal changes transient profile and onset energy envelope:
    # tonal_center: 22965 broadband click events alter K-S chroma template correlation;
    # after removal the chromagram shifts vs. click-distorted CIG checkpoint.
    # CIG P2 rollback confirmed (rollbacks=1, strength→0.07 in production, 2026-04-10).
    # natuerlichkeit: PMGG correctly excludes natuerlichkeit for phase_01 (MFCC-smoothness
    #   unreliable vs. click-bearing reference); CIG must exclude it too — otherwise accumulated
    #   natuerlichkeit drift (phase_55→phase_01) triggers spurious CIG rollback at later phase.
    "phase_01": frozenset(
        {"authentizitaet", "artikulation", "timbre_authentizitaet", "tonal_center", "natuerlichkeit"}
    ),
    # Crackle removal modifies broadband impulse energy → chroma correlation drops:
    # natuerlichkeit: PMGG excludes natuerlichkeit for phase_09 (broadband spectral mod → MFCC-smoothness
    #   proxy disturbed by crackle removal, identical mechanism to phase_03/phase_29).
    #   CIG must align: without exclusion, crackle's contribution to cumulative nat drift is counted
    #   and can tip the total over _drift_tolerance at a later non-excluded phase.
    "phase_09": frozenset(
        {"authentizitaet", "artikulation", "timbre_authentizitaet", "tonal_center", "natuerlichkeit"}
    ),
    # Click/pop removal (same carrier-chain defect class as phase_01):
    # natuerlichkeit: identical mechanism to phase_01 — PMGG excludes natuerlichkeit but CIG did not;
    #   same CIG-PMGG mismatch pattern (2026-04-10).
    "phase_27": frozenset(
        {"authentizitaet", "artikulation", "timbre_authentizitaet", "tonal_center", "natuerlichkeit"}
    ),
    # Surface noise profiling/reduction changes broadband energy floor → spectral
    # fingerprint shifts vs. noisy reference (same as tonal_center inflation §2.29c):
    # natuerlichkeit: uniform noise-floor → metric sees "broadband = natural"; clean audio scores lower (Reference Paradox §2.44)
    "phase_28": frozenset({"authentizitaet", "artikulation", "timbre_authentizitaet", "natuerlichkeit"}),
    # Tape hiss reduction removes broadband carrier noise → spectral divergence
    # from noisy checkpoint is intentional.
    # artikulation: silence-between-notes energy envelope changes as noise floor drops.
    # natuerlichkeit: same Reference Paradox as phase_28 — noise floor artificially inflates natuerlichkeit proxy.
    # tonal_center: broadband tape hiss adds uniform chroma-bin energy → K-S correlation vs. hissy checkpoint elevated;
    #   after hiss reduction K-S converges to clean key estimate (same mechanism as phase_03).
    "phase_29": frozenset(
        {"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit", "tonal_center"}
    ),
    # Denoise (broadband) — same reference-paradoxon as tape hiss:
    # artikulation: silence regions become quieter after noise removal → onset gap detectability changes.
    # natuerlichkeit: broadband noise creates artificially uniform spectrum → metric scores it as "natural" (Reference Paradox §2.44).
    # tonal_center: broadband noise adds uniform energy across chroma bins → K-S template correlation
    #   vs. noisy checkpoint artificially elevated; after denoising chromagram converges to clean key estimate.
    "phase_03": frozenset(
        {"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit", "tonal_center"}
    ),
    # Hum removal changes spectral fingerprint (notch series) vs. hum-distorted reference.
    # artikulation: harmonic energy in hum bands is removed → onset rise-time changes.
    # natuerlichkeit: notch-filtered spectrum has altered MFCC-smoothness vs. hum-bearing reference
    #   (Reference Paradox §2.44 — hum harmonics contribute to "uniform broadband" metric baseline).
    "phase_02": frozenset({"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit"}),
    # Reverb reduction on carrier-saturated material (tape flutter reflections, vinyl
    # room coupling) — SGMSE+ deconvolution shifts spectral fingerprint vs. reverberant ref.
    # artikulation: note decay envelope shortened intentionally (reverb tails removed).
    # natuerlichkeit: room-air / early reflections in degraded recording score as "natural" before dereverb (Reference Paradox §2.44).
    # tonal_center: reverb smears chroma bins → K-S template inflated vs. reverberant checkpoint; after derev key estimate sharpens.
    "phase_20": frozenset(
        {"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit", "tonal_center"}
    ),
    # Noise gate (Silero VAD) removes low-energy carrier noise between phrases —
    # silence insertion shifts chroma/energy fingerprint vs. noisy-floor reference:
    "phase_18": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # Groove-echo cancellation removes pre-echo artefact (vinyl inner-groove distortion) —
    # spectral fingerprint diverges from pre-echo-distorted reference checkpoint:
    "phase_61": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # Advanced dereverb (phase_49) — same class as phase_20.
    # artikulation: reverb-tail removal shortens perceived note decay vs. reverberant checkpoint.
    # natuerlichkeit: same Reference Paradox as phase_20 (reverberant room = "natural" to metric before dereverb).
    # tonal_center: reverb smears chroma bins → cumulative K-S template shift (same mechanism as phase_20 tonal_center).
    "phase_49": frozenset(
        {"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit", "tonal_center"}
    ),
    # Azimuth correction (phase_25) shifts spectral balance vs. mis-aligned reference:
    "phase_25": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # RIAA/NAB de-emphasis inversion (phase_04) redistributes HF/LF energy vs. pre-correction reference.
    # Carrier EQ correction is §2.46 Carrier-Chain-Inversion (§2.44 Reference-Paradoxon applies):
    # authentizitaet: chromagram shifts because tonal balance changes dramatically after EQ inversion.
    # timbre_authentizitaet: spectral centroid and shape change vs. RIAA-distorted checkpoint.
    # artikulation: EQ inversion alters transient rise-time in HF band → onset energy envelope differs vs. RIAA-distorted ref.
    # natuerlichkeit: RIAA-distorted spectrum has different MFCC-smoothness baseline → metric shift after correction.
    "phase_04": frozenset({"authentizitaet", "timbre_authentizitaet", "artikulation", "natuerlichkeit"}),
    # Speed/pitch correction (phase_31) corrects turntable/tape transport speed deviation (§2.46).
    # Mirror of phase_12 (Wow/Flutter) — same Carrier-Chain-Inversion family:
    # authentizitaet: chromagram fundamentally changes when correcting pitch deviation vs. pitch-deviated checkpoint.
    # natuerlichkeit: tempo/groove feel shifts intentionally away from speed-deviated reference.
    # artikulation: articulation patterns are different at correct vs. deviated pitch.
    # tonal_center: pitch-deviated checkpoint has wrong key center → K-S correlation against deviated reference meaningless after correction.
    "phase_31": frozenset(
        {"authentizitaet", "natuerlichkeit", "artikulation", "timbre_authentizitaet", "tonal_center"}
    ),
    # Stereo phase correction (phase_14) aligns L/R phase vs. mis-aligned carrier reference:
    # authentizitaet: stereo correlation fingerprint changes vs. phase-misaligned checkpoint
    # timbre_authentizitaet: comb-filter spectral coloration is removed → fingerprint shifts
    "phase_14": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # Stereo balance correction (phase_15) equalises L/R levels vs. imbalanced carrier reference:
    # authentizitaet: stereo field correlation changes as channels are re-balanced
    "phase_15": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # Crosstalk cancellation (phase_62) repairs early stereo channel leakage (§2.46 §2.44):
    # authentizitaet: chromagram and stereo-fingerprint diverge from crosstalk-distorted checkpoint
    # timbre_authentizitaet: spectral crosstalk coloration is removed → fingerprint shifts intentionally
    "phase_62": frozenset({"authentizitaet", "timbre_authentizitaet"}),
    # ── §2.54 PMGG→CIG SYNCHRONISATION BLOCK ──────────────────────────────────────────────
    # Architectural invariant: CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[phase] ⊇ PMGG.PHASE_GOAL_EXCLUSIONS[phase] ∩ P1P2
    # All entries below were added in v9.11.3 to close the CIG-PMGG mismatch gap.
    # timbre_authentizitaet is the most common false-positive: spectral centroid/MFCC-Pearson
    # correlation changes any time a phase modifies the spectral envelope (EQ, BW-restoration,
    # enhancement) vs. the degraded checkpoint — this is §2.44 Reference-Paradoxon.
    # ─────────────────────────────────────────────────────────────────────────────────────
    # Frequency restoration: HF synthesis shifts spectral centroid → timbre proxy unreliable vs. BW-limited ref.
    "phase_06": frozenset({"timbre_authentizitaet"}),
    # Harmonic restoration: adds overtones → spectral shape differs vs. BW-limited reference.
    # artikulation: synthesised harmonic energy changes transient rise-time profile.
    "phase_07": frozenset({"artikulation", "timbre_authentizitaet"}),
    # TDP/HPSS transient detection: harmonic-percussive separation shifts onset energy profile.
    # artikulation: HPSS alters transient sharpness vs. mixed-domain checkpoint.
    "phase_08": frozenset({"artikulation"}),
    # Surface noise / crackle profiling: wideband noise profiling shifts spectral baseline.
    "phase_13": frozenset({"timbre_authentizitaet"}),
    # Era-adaptive tone shaping / retro EQ: deliberate spectral balance change → all P1/P2 spectral proxies shift.
    # authentizitaet: chroma distribution changes with EQ.
    # natuerlichkeit: MFCC-smoothness changes with EQ curve.
    # tonal_center: K-S correlation vs. pre-EQ checkpoint shifts.
    "phase_16": frozenset({"authentizitaet", "natuerlichkeit", "timbre_authentizitaet", "tonal_center"}),
    # Mastering / loudness normalisation: gain/compression shifts energy distribution.
    # artikulation: dynamic range change alters onset contrast ratio.
    # natuerlichkeit: compression smooths spectral variance → metric shift.
    # tonal_center: loudness normalisation may alter chroma bin weighting.
    "phase_17": frozenset({"artikulation", "natuerlichkeit", "tonal_center"}),
    # Carrier noise gate (secondary): gate openings/closings alter spectral continuity.
    # natuerlichkeit: gated silence has different MFCC-smoothness vs. continuous carrier-noise floor.
    "phase_19": frozenset({"natuerlichkeit", "timbre_authentizitaet"}),
    # Harmonic exciter: adds synthetic harmonics → spectral shape diverges from pre-exciter reference.
    "phase_21": frozenset({"timbre_authentizitaet"}),
    # Presence / air band EQ: HF shelving shifts spectral centroid and MFCC.
    "phase_22": frozenset({"timbre_authentizitaet"}),
    # Spectral repair / AudioSR upsampling: synthesised broadband content shifts all spectral fingerprints.
    # artikulation: synthesised HF energy fills transient profiles not in BW-limited checkpoint.
    # authentizitaet: chroma shifts as BW is extended (same class as phase_06).
    # natuerlichkeit: MFCC-smoothness of synthesised spectrum differs from BW-limited reference.
    "phase_23": frozenset({"artikulation", "authentizitaet", "natuerlichkeit", "timbre_authentizitaet"}),
    # Carrier noise / reel-splice click repair (secondary): similar to phase_01 family.
    # artikulation: click regions have altered onset energy after repair.
    "phase_26": frozenset({"artikulation"}),
    # Quantisation noise reduction (ADC artefact): dithering changes noise-floor texture → spectral fingerprint.
    # artikulation: dither changes LSB energy envelope in transient tails.
    # tonal_center: quantisation noise adds white-noise component to chroma → K-S probe shift after removal.
    "phase_32": frozenset({"timbre_authentizitaet"}),
    # Stereo width / imaging: M/S processing shifts stereo-field spectral balance.
    "phase_33": frozenset({"timbre_authentizitaet"}),
    # De-esser: HF reduction in sibilant bands shifts spectral centroid.
    "phase_34": frozenset({"timbre_authentizitaet"}),
    # Transient shaper: attack/sustain shaping changes onset energy envelope and spectral shape.
    # artikulation: attack shaping directly affects onset rise-time measurements.
    "phase_36": frozenset({"artikulation"}),
    # Bass enhancement / warmth: LF boost shifts spectral centroid and MFCC low-band.
    # tonal_center: LF boost shifts chroma bin weighting → K-S template correlation changes.
    "phase_37": frozenset({"timbre_authentizitaet", "tonal_center"}),
    # Spectral smoothing / micro-noise reduction: fine-grained spectral changes.
    "phase_38": frozenset({"timbre_authentizitaet"}),
    # Vocal clarity / presence enhancement: formant shaping changes spectral shape.
    "phase_39": frozenset({"timbre_authentizitaet"}),
    # Saturation / soft-clip: harmonic addition shifts spectral shape.
    "phase_40": frozenset({"timbre_authentizitaet"}),
    # Parallel compression: dynamic spectral modification.
    # artikulation: parallel compression changes onset-to-sustain ratio.
    "phase_41": frozenset({"artikulation", "timbre_authentizitaet"}),
    # Carrier-formant decay inversion (stage 0.5): zero-phase Bell-EQ on F1-F4 →
    # spectral shape deliberately restored from carrier degradation (§2.47, §2.52 Hebel 4).
    # All four metrics shift intentionally vs. carrier-degraded checkpoint (§2.44 Reference Paradox full scope).
    "phase_42": frozenset({"artikulation", "authentizitaet", "natuerlichkeit", "timbre_authentizitaet"}),
    # MP-SENet vocal enhancement: spectral shape modified for vocal intelligibility.
    # artikulation: phone-level onset sharpening changes articulation metric vs. degraded reference.
    "phase_43": frozenset({"artikulation", "timbre_authentizitaet"}),
    # Stereo enhancement / Haas effect: stereo imaging changes spectral balance per-channel.
    "phase_44": frozenset({"timbre_authentizitaet"}),
    # Mono compatibility / mid enhancement: M/S recombination shifts spectral shape.
    "phase_45": frozenset({"timbre_authentizitaet"}),
    # Loudness maximiser / brickwall limiter: gain reduction changes spectral dynamics.
    "phase_46": frozenset({"timbre_authentizitaet"}),
    # Mid/side EQ: deliberate spectral balance change in M/S domain.
    "phase_48": frozenset({"timbre_authentizitaet"}),
    # Spectral inpainting (CQTdiff+ / NMF): synthesised content fills masked regions.
    # artikulation: inpainted regions have new onset profiles not present in masked checkpoint.
    "phase_50": frozenset({"artikulation"}),
    # Vintage EQ / analogue-chain modelling: deliberate spectral colouring.
    "phase_51": frozenset({"timbre_authentizitaet"}),
    # Tape saturation / analogue warmth modelling: harmonic addition shifts spectral shape.
    "phase_52": frozenset({"timbre_authentizitaet"}),
    # Era-style mastering chain (reference-based): all spectral proxies shift towards reference target.
    # authentizitaet: chroma-based similarity vs. pre-mastering checkpoint drops intentionally.
    # natuerlichkeit: mastering chain smooths MFCC vs. unmastered reference.
    "phase_56": frozenset({"authentizitaet", "natuerlichkeit", "timbre_authentizitaet"}),
    # Print-through reduction (reel tape): LF modulation artefact removal shifts spectral baseline.
    # authentizitaet: print-through echo phantom creates chroma artefacts; removal changes chromagram.
    "phase_57_print_through_reduction": frozenset({"authentizitaet"}),
    # Lyrics-guided enhancement (§2.36): formant shaping guided by phoneme sequence.
    # artikulation: phone-boundary energy redistribution directly affects articulation metric.
    # timbre_authentizitaet: formant-targeted EQ changes MFCC-Pearson vs. pre-enhancement reference.
    # tonal_center: vowel formant emphasis can shift chroma bin weighting (§2.44 Reference Paradox).
    "phase_58_lyrics_guided_enhancement": frozenset({"artikulation", "timbre_authentizitaet", "tonal_center"}),
}


def _resolve_phase_specific_drift_exclusions(phase_id: str) -> frozenset[str]:
    """Return goal exclusions for known false-positive phase patterns (prefix-based)."""
    for prefix, exclusions in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS.items():
        if phase_id.startswith(prefix):
            return exclusions
    return frozenset()


# §2.48 Carrier-Repair phases: rollbacks caused by the Reference-Paradox (§2.44)
# should NOT increment consecutive_rollbacks.  These phases intentionally diverge from
# the degraded checkpoint (that is the goal). Counting them as 'stuck loop' failures
# would stop multi-tier restoration (e.g. vinyl→tape→mp3 chain) long before all
# carrier defects are removed.
_CARRIER_REPAIR_PHASE_PREFIXES: tuple[str, ...] = (
    "phase_01",  # click removal
    "phase_02",  # hum removal
    "phase_03",  # broadband denoise
    "phase_09",  # crackle removal
    "phase_18",  # noise gate
    "phase_20",  # reverb reduction
    "phase_24",  # dropout repair
    "phase_25",  # azimuth correction
    "phase_27",  # click/pop removal
    "phase_28",  # surface noise profiling
    "phase_29",  # tape hiss reduction
    "phase_49",  # advanced dereverb
    "phase_55",  # diffusion inpainting
)


# Max cumulative drift before rollback (§2.48 / §2.54)
# This is the FALLBACK value when no material context is available.
# The actual tolerance is computed via compute_adaptive_drift_tolerance().
MAX_CUMULATIVE_DRIFT = -0.05

# Max consecutive rollbacks (§2.54 adaptive formula).
# Fallback constant — actual limit is max(5, n_carrier_phases + 2).
MAX_CONSECUTIVE_ROLLBACKS = 5


def compute_adaptive_drift_tolerance(
    restorability_score: float = 50.0,
    material_type: str = "cd_digital",
    defect_severity_mean: float = 0.0,
    n_active_phases: int = 10,
) -> float:
    """§2.54 Adaptive Drift-Toleranz — ersetzt feste -0.05-Konstante.

    Computes how much cumulative P1/P2 proxy-drift is acceptable before
    the emergency brake (rollback) fires.  More degraded material needs
    more tolerance because carrier-repair phases intentionally shift
    spectral fingerprints away from the corrupted checkpoint (§2.44
    Reference-Paradoxon).

    Returns:
        Negative float, e.g. -0.03 (CD, light) to -0.25 (shellac, heavily degraded).
    """
    # Base tolerance by material class
    _MATERIAL_BASE: dict[str, float] = {
        "cd_digital": -0.03,
        "dat": -0.03,
        "minidisc": -0.04,
        "mp3_high": -0.04,
        "mp3_low": -0.06,
        "cassette": -0.07,
        "tape": -0.08,
        "reel_tape": -0.09,
        "vinyl": -0.10,
        "shellac": -0.15,
        "wax_cylinder": -0.18,
        "wire_recording": -0.15,
        "optical_film": -0.10,
        "radio_broadcast": -0.08,
        "unknown": -0.06,
    }
    base = _MATERIAL_BASE.get(material_type, -0.06)

    # Restorability factor: lower restorability → more tolerance needed
    # Continuous (not step-function): linear interpolation 0→100 maps to 1.8→0.8
    restorability_clamped = float(np.clip(restorability_score, 0.0, 100.0))
    restorability_factor = 1.8 - (restorability_clamped / 100.0)  # 1.8 at 0, 0.8 at 100

    # Defect severity factor: higher mean severity → more tolerance
    severity_factor = 1.0 + 0.5 * float(np.clip(defect_severity_mean, 0.0, 1.0))

    # Phase count factor: more phases → more cumulative drift is normal
    phase_factor = 1.0 + 0.02 * max(0, n_active_phases - 5)  # +2% per phase above 5

    tolerance = base * restorability_factor * severity_factor * phase_factor

    # Clamp to sane range: never tighter than -0.02, never looser than -0.30
    tolerance = float(np.clip(tolerance, -0.30, -0.02))

    return tolerance


def compute_adaptive_max_rollbacks(
    n_carrier_phases: int = 3,
) -> int:
    """§2.54 Adaptive max consecutive rollbacks.

    Multi-generation material (vinyl→tape→mp3) needs more carrier-repair
    phases, each of which may individually trigger a rollback due to the
    Reference-Paradoxon.
    """
    return max(5, n_carrier_phases + 2)


# STFT-based phases (§2.48) — group delay coherence check after ≥3.
# §2.48a Architektur-Inversion: STFT_PHASES ist jetzt ein Hinweis-Set;
# der GDD-Check konsultiert zusätzlich GDD_VALID_TYPES aus phase_ontology.
# ML_GENERATIVE Phasen (SGMSE+ / phase_20, FlowMatching / phase_55) sind
# explizit NICHT in GDD_VALID_TYPES: Diffusionsausgang ist per Design
# nicht STFT-phasenkohärent (Richter et al., TASLP 2022).


STFT_PHASES = frozenset(
    {
        "phase_03_denoise",
        "phase_07_harmonic_restoration",
        "phase_23_spectral_repair",
        "phase_24_dropout_repair",
        "phase_29_tape_hiss_reduction",
        "phase_35_multiband_compression",
        "phase_49_advanced_dereverb",
        "phase_50_spectral_repair",  # STFT bin-interpolation (SUBTRACTIVE, GDD valid)
        # phase_20_reverb_reduction: SGMSE+ primär  → ML_GENERATIVE → GDD invalide
        # → nicht mehr in STFT_PHASES (Richter 2022).
        # Verbleibt als Legacy-Eintrag auskommentiert für Audit-Zwecke:
        # "phase_20_reverb_reduction",
    }
)

# Max group delay deviation in ms (§2.48)
# 5 ms = realistic limit for FFT-based spectral processing at 48 kHz.
# 2 ms was too tight: a standard 2048-point STFT hop of 512 samples already
# introduces 10.7 ms framing latency; spectral-subtraction filters routinely
# shift per-bin phase by 3–8 ms without introducing perceptible artefacts.
# The 95th-percentile measurement below means we tolerate up to 5 ms *peak*
# group-delay change across the audio spectrum — any more signals a genuine
# phase-distortion problem (e.g. cascaded IIR filters with different cut-offs
# operating independently on L and R, violating §2.51).
MAX_GROUP_DELAY_DEVIATION_MS = 5.0

# Spectral-subtraction phases have an inherently higher group-delay deviation
# because frequency-domain magnitude modifications displace phase by 3–8 ms
# (STFT window 2048 samples / 48 kHz = 42.6 ms frame; bin-level phase shift
# from Wiener/OMLSA mask application is not a sign of quality degradation).
# §2.48 spec explicitly acknowledges this range — use 10 ms for these phases
# to prevent spurious rollbacks on tape/hiss reduction.
MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL = 10.0

# Phases that use spectral subtraction / frequency-domain masking and are
# therefore exempt from the 5 ms threshold — they get 10 ms instead.
_SPECTRAL_SUBTRACTION_PHASES = frozenset(
    {
        "phase_03_denoise",
        "phase_20_reverb_reduction",
        "phase_29_tape_hiss_reduction",
        "phase_49_advanced_dereverb",
    }
)

# Critical interaction pairs (§2.48 Table)
CRITICAL_PAIRS: list[tuple[frozenset[str], str, str, float]] = [
    # (phases, guard_goal, guard_description, max_regression)
    (
        frozenset({"phase_03_denoise", "phase_20_reverb_reduction"}),
        "natuerlichkeit",
        "Cumulative room removal (De-Hiss + De-Reverb)",
        -0.03,
    ),
    (
        frozenset({"phase_03_denoise", "phase_49_advanced_dereverb"}),
        "natuerlichkeit",
        "Cumulative room removal (De-Hiss + Advanced De-Reverb)",
        -0.03,
    ),
    (
        frozenset({"phase_29_tape_hiss_reduction", "phase_03_denoise"}),
        "natuerlichkeit",  # proxy for over-denoising
        "Over-denoising (NR + De-Hiss)",
        -0.03,
    ),
    (
        frozenset({"phase_35_multiband_compression", "phase_40_lufs_normalization"}),
        "micro_dynamics",
        "Dynamics loss (Multiband-Comp + LUFS)",
        -0.04,
    ),
    (
        frozenset({"phase_07_harmonic_restoration", "phase_42_vocal_ai_enhancement"}),
        "timbre_authentizitaet",
        "Frequency doubling (Harmonic + Vocal-AI)",
        -0.03,
    ),
]


@dataclass
class InteractionGuardCheckpoint:
    """Audio checkpoint with goal scores."""

    audio: np.ndarray
    phase_id: str
    goal_scores: dict[str, float]
    stft_phase_count: int = 0


@dataclass
class InteractionRollback:
    """Record of a rollback event."""

    phase_id: str
    reason: str
    drift: dict[str, float]
    rolled_back_to: str


@dataclass
class InteractionGuardState:
    """Mutable guard state for one pipeline run."""

    pre_pipeline_goals: dict[str, float] = field(default_factory=dict)
    best_checkpoint: InteractionGuardCheckpoint | None = None
    stft_phases_executed: list[str] = field(default_factory=list)
    executed_phases: set[str] = field(default_factory=set)
    consecutive_rollbacks: int = 0
    rollback_log: list[InteractionRollback] = field(default_factory=list)
    should_stop: bool = False
    # §2.54 Adaptive parameters — set in set_pre_pipeline_baseline()
    adaptive_drift_tolerance: float = MAX_CUMULATIVE_DRIFT  # fallback
    adaptive_max_rollbacks: int = MAX_CONSECUTIVE_ROLLBACKS  # fallback
    # §2.54 Material context for adaptive GDD threshold
    restorability_score: float = 50.0
    material_type: str = "unknown"
    # §2.56 Song-Goal-Importance: per-goal weights for weighted drift
    goal_weights: dict[str, float] | None = None


class CumulativeInteractionGuard:
    """§2.48 Kumulative-Phasen-Interaktions-Guard.

    Monitors cumulative P1/P2 drift, critical phase pair interactions,
    and STFT phase coherence across the pipeline.
    """

    def reset(self) -> InteractionGuardState:
        """Create fresh state for a new pipeline run."""
        return InteractionGuardState()

    def set_pre_pipeline_baseline(
        self,
        state: InteractionGuardState,
        audio: np.ndarray,
        goal_scores: dict[str, float],
        *,
        material_type: str = "unknown",
        restorability_score: float = 50.0,
        defect_severity_mean: float = 0.0,
        n_active_phases: int = 10,
        n_carrier_phases: int = 3,
        goal_weights: dict[str, float] | None = None,
    ) -> None:
        """Set the pre-pipeline P1/P2 baseline scores before any phases run.

        §2.29c Baseline-Capping: Defekte inflationieren bestimmte Metriken
        künstlich (Rauschen → tonal_center erhöht durch gleichmäßige Chroma-Lifts;
        Hall → waerme/transparenz erhöht). Subtractive Phasen werden gegen diesen
        inflationierten Wert gemessen → false Regression → Rollback auf verbessertes Audio.

        Lösung: Baseline wird auf canonical_threshold + 0.05 Headroom gedeckelt.
        Dieser Schritt greift nur auf P1/P2 Goals für spätere Drift-Checks.

        §2.54: Computes material-adaptive drift tolerance and max rollbacks
        from the concrete song context — not hard-coded constants.
        """
        # §2.29c canonical thresholds (Restoration — Pareto-Differenzierung §9.10.77)
        # Keys müssen mit measure_all()-Output übereinstimmen (deutsch).
        _CANONICAL_BASELINES: dict[str, float] = {
            "natuerlichkeit": 0.90,
            "authentizitaet": 0.88,
            "tonal_center": 0.95,
            "timbre_authentizitaet": 0.87,
            "artikulation": 0.85,
        }
        capped: dict[str, float] = {}
        for goal, measured in goal_scores.items():
            if goal in P1_P2_GOALS:
                canonical = _CANONICAL_BASELINES.get(goal, 0.85)
                capped[goal] = min(measured, canonical + 0.05)
            else:
                capped[goal] = measured

        state.pre_pipeline_goals = capped

        # §2.54 Adaptive parameters from song context
        state.adaptive_drift_tolerance = compute_adaptive_drift_tolerance(
            restorability_score=restorability_score,
            material_type=material_type,
            defect_severity_mean=defect_severity_mean,
            n_active_phases=n_active_phases,
        )
        state.adaptive_max_rollbacks = compute_adaptive_max_rollbacks(
            n_carrier_phases=n_carrier_phases,
        )
        # Persist for adaptive GDD threshold in _check_group_delay (§2.54)
        state.restorability_score = float(restorability_score)
        state.material_type = str(material_type)
        # §2.56 Song-Goal-Importance: store for weighted drift in check_after_phase
        state.goal_weights = dict(goal_weights) if goal_weights else None

        state.best_checkpoint = InteractionGuardCheckpoint(
            audio=audio.copy(),
            phase_id="__pre_pipeline__",
            goal_scores=dict(goal_scores),  # unkapped — für echte P1/P2-Ziele
            stft_phase_count=0,
        )
        logger.debug(
            "§2.48 InteractionGuard: baseline set (capped) — P1/P2 goals: %s, "
            "adaptive_drift_tol=%.3f, adaptive_max_rollbacks=%d",
            {k: f"{v:.3f}" for k, v in capped.items() if k in P1_P2_GOALS},
            state.adaptive_drift_tolerance,
            state.adaptive_max_rollbacks,
        )

    def check_after_phase(
        self,
        state: InteractionGuardState,
        phase_id: str,
        current_audio: np.ndarray,
        current_goals: dict[str, float],
        sr: int,
    ) -> tuple[np.ndarray, bool]:
        """Check cumulative drift after a phase.

        Returns:
            (audio_to_use, was_rolled_back): either current_audio or best_checkpoint
        """
        if state.should_stop:
            return current_audio, False

        state.executed_phases.add(phase_id)

        # §2.48a: Drift-Check überspringen für ANALYSIS_ONLY (Audio unverändert)
        _phase_type = get_phase_type(phase_id)
        if _phase_type in P1P2_DRIFT_CHECK_INVALID_TYPES:
            return current_audio, False

        # Track STFT phases (nur wenn GDD-Check valide für diesen Typ)
        _gdd_valid_for_type = _phase_type in GDD_VALID_TYPES
        if phase_id in STFT_PHASES and _gdd_valid_for_type:
            state.stft_phases_executed.append(phase_id)

        # 1. Check cumulative P1/P2 drift
        # §2.29c: SUBTRACTIVE-Phasen können defekt-inflationierte Goals scheinbar
        # verschlechtern (tonal_center: Rauschen → künstlich hohe Chroma-Lifts → Inflation).
        # Nach Denoise wird echter niedrigerer Wert sichtbar — kein Artefakt.
        # Diese Goals SUBTRACTIVE-Phase-selektiv aus Drift-Check ausschließen.
        _is_subtractive = _phase_type in BASELINE_CAPPING_VALID_TYPES  # SUBTRACTIVE
        rolled_back = False
        # §2.54: Use material-adaptive drift tolerance (computed in set_pre_pipeline_baseline)
        _drift_tolerance = state.adaptive_drift_tolerance
        _max_rollbacks = state.adaptive_max_rollbacks
        if state.pre_pipeline_goals:
            cumulative_drift = {}
            _phase_exclusions = _resolve_phase_specific_drift_exclusions(phase_id)
            # §2.56: Per-goal weights amplify drift for important goals
            _gw = getattr(state, "goal_weights", None)
            for g in P1_P2_GOALS:
                if _is_subtractive and g in _DEFECT_INFLATED_SUBTRACTIVE_GOALS:
                    continue  # tonal_center für SUBTRACTIVE-Phasen nicht prüfen (§2.29c)
                if g in _phase_exclusions:
                    continue
                if g in current_goals and g in state.pre_pipeline_goals:
                    raw_drift = current_goals[g] - state.pre_pipeline_goals[g]
                    # §2.56: Weight amplifies negative drift for important goals
                    w = _gw.get(g, 1.0) if _gw else 1.0
                    cumulative_drift[g] = raw_drift * w if raw_drift < 0 else raw_drift

            worst_drift = min(cumulative_drift.values()) if cumulative_drift else 0.0
            if worst_drift < _drift_tolerance:
                logger.warning(
                    "§2.48 P1/P2 cumulative drift after %s: %s (tol=%.3f) → rollback to %s",
                    phase_id,
                    {k: f"{v:+.3f}" for k, v in cumulative_drift.items() if v < _drift_tolerance},
                    _drift_tolerance,
                    state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                )
                state.rollback_log.append(
                    InteractionRollback(
                        phase_id=phase_id,
                        reason=f"P1/P2 cumulative drift {worst_drift:+.3f} (tol={_drift_tolerance:+.3f})",
                        drift=cumulative_drift,
                        rolled_back_to=state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                    )
                )
                # §2.48 Carrier-Repair-Ausnahme: Reference Paradox (§2.44) — intentionale
                # Divergenz vom beschädigten Checkpoint ist kein Versagen.
                _is_carrier_repair_phase = any(phase_id.startswith(p) for p in _CARRIER_REPAIR_PHASE_PREFIXES)
                if _is_carrier_repair_phase:
                    logger.debug(
                        "§2.48 Carrier-repair rollback (%s) — consecutive_rollbacks NICHT inkrementiert (§2.44)",
                        phase_id,
                    )
                else:
                    state.consecutive_rollbacks += 1
                rolled_back = True
                # §2.48 STFT-Zähler korrigieren: Phase war im Audio-Zustand nicht
                # persistent — Zähler muss Checkpoint-Stand widerspiegeln.
                if phase_id in STFT_PHASES and phase_id in state.stft_phases_executed:
                    state.stft_phases_executed.remove(phase_id)

                if state.consecutive_rollbacks >= _max_rollbacks:
                    state.should_stop = True
                    logger.warning(
                        "§2.48 Max consecutive rollbacks (%d) reached — pipeline stop, export on best checkpoint",
                        _max_rollbacks,
                    )

                if state.best_checkpoint is not None:
                    return state.best_checkpoint.audio.copy(), True
                return current_audio, True

        # 2. Check critical interaction pairs
        pair_rollback = self._check_critical_pairs(state, phase_id, current_goals)
        if pair_rollback:
            logger.warning(
                "§2.48 Critical pair interaction after %s: %s → rollback",
                phase_id,
                pair_rollback,
            )
            state.rollback_log.append(
                InteractionRollback(
                    phase_id=phase_id,
                    reason=f"Critical pair: {pair_rollback}",
                    drift={},
                    rolled_back_to=state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                )
            )
            # §2.48 STFT-Zähler korrigieren bei Critical-Pair-Rollback
            if phase_id in STFT_PHASES and phase_id in state.stft_phases_executed:
                state.stft_phases_executed.remove(phase_id)
            # §2.48 Carrier-Repair-Ausnahme: Reference Paradox (§2.44)
            _is_carrier_repair_cp = any(phase_id.startswith(p) for p in _CARRIER_REPAIR_PHASE_PREFIXES)
            if _is_carrier_repair_cp:
                logger.debug(
                    "§2.48 Carrier-repair critical-pair rollback (%s) — consecutive_rollbacks NICHT inkrementiert",
                    phase_id,
                )
            else:
                state.consecutive_rollbacks += 1
            rolled_back = True

            if state.consecutive_rollbacks >= _max_rollbacks:
                state.should_stop = True

            if state.best_checkpoint is not None:
                return state.best_checkpoint.audio.copy(), True
            return current_audio, True

        # 3. Check STFT phase coherence after ≥3 STFT phases
        # §2.48a: GDD nur prüfen wenn Typ valide und Phase in STFT_PHASES.
        if len(state.stft_phases_executed) >= 3 and phase_id in STFT_PHASES and _gdd_valid_for_type:
            gdd_ok = self._check_group_delay(
                state.best_checkpoint.audio if state.best_checkpoint else current_audio,
                current_audio,
                sr,
                phase_id=phase_id,
                state=state,
            )
            _gdd_threshold = self._compute_gdd_threshold(phase_id, state)
            if not gdd_ok:
                logger.warning(
                    "§2.48 STFT group delay deviation > %.1f ms after %d STFT phases (%s) → rollback",
                    _gdd_threshold,
                    len(state.stft_phases_executed),
                    phase_id,
                )
                state.rollback_log.append(
                    InteractionRollback(
                        phase_id=phase_id,
                        reason=f"STFT group delay deviation after {len(state.stft_phases_executed)} STFT phases",
                        drift={},
                        rolled_back_to=state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                    )
                )
                state.consecutive_rollbacks += 1
                # §2.48 STFT-Zähler korrigieren: Group-Delay-Rollback bedeutet,
                # diese Phase ist nicht im Audio-Zustand persistent.  Ohne
                # Korrektur bleibt len(stft_phases_executed) dauerhaft ≥ 3 und
                # jede weitere STFT-Phase löst sofort erneut einen Rollback aus
                # → Pipeline-Stop nach nur 2 STFT-Rollbacks.
                if phase_id in state.stft_phases_executed:
                    state.stft_phases_executed.remove(phase_id)
                rolled_back = True

                if state.consecutive_rollbacks >= _max_rollbacks:
                    state.should_stop = True

                if state.best_checkpoint is not None:
                    return state.best_checkpoint.audio.copy(), True
                return current_audio, True

        # No rollback — update best checkpoint if goals improved
        if not rolled_back:
            state.consecutive_rollbacks = 0  # reset on success
            if self._is_better_checkpoint(state, current_goals):
                state.best_checkpoint = InteractionGuardCheckpoint(
                    audio=current_audio.copy(),
                    phase_id=phase_id,
                    goal_scores=dict(current_goals),
                    stft_phase_count=len(state.stft_phases_executed),
                )
                logger.debug(
                    "§2.48 Updated best_checkpoint to %s (P1/P2 mean=%.3f)",
                    phase_id,
                    np.mean([current_goals.get(g, 0.0) for g in P1_P2_GOALS if g in current_goals]),
                )
            # §2.44 Reference-Paradox rebase: after an accepted carrier-repair phase,
            # absorb its legitimate goal-drops into the cumulative baseline so that
            # subsequent phases are not penalised for earlier intentional defect removal.
            # Only rebase if the goal actually dropped (no upward rebase).
            _accepted_exclusions = _resolve_phase_specific_drift_exclusions(phase_id)
            if _accepted_exclusions and state.pre_pipeline_goals:
                for _g in _accepted_exclusions:
                    if _g in current_goals and _g in state.pre_pipeline_goals:
                        if current_goals[_g] < state.pre_pipeline_goals[_g]:
                            state.pre_pipeline_goals[_g] = current_goals[_g]
                            logger.debug(
                                "§2.44 Carrier-repair rebase: %s %s baseline → %.3f",
                                phase_id,
                                _g,
                                current_goals[_g],
                            )

        return current_audio, False

    def get_rollback_metadata(self, state: InteractionGuardState) -> dict:
        """Get metadata for RestorationResult."""
        return {
            "interaction_rollbacks": [
                {
                    "phase_id": rb.phase_id,
                    "reason": rb.reason,
                    "drift": rb.drift,
                    "rolled_back_to": rb.rolled_back_to,
                }
                for rb in state.rollback_log
            ],
            "pipeline_stopped_early": state.should_stop,
            "stft_phases_count": len(state.stft_phases_executed),
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_critical_pairs(
        self,
        state: InteractionGuardState,
        phase_id: str,
        goals: dict[str, float],
    ) -> str | None:
        """Check if current phase + already-executed phases form a critical pair.

        §2.54: max_reg is scaled adaptively for material/restorability so that
        carrier-repair pairs (denoise+hiss) on analog material don't fire false-positive
        rollbacks (Reference Paradox §2.44).
        """
        for pair_phases, guard_goal, description, max_reg in CRITICAL_PAIRS:
            if phase_id in pair_phases:
                other_phases = pair_phases - {phase_id}
                if other_phases.issubset(state.executed_phases):
                    # Both phases of the pair have executed
                    if guard_goal in goals and guard_goal in state.pre_pipeline_goals:
                        drift = goals[guard_goal] - state.pre_pipeline_goals[guard_goal]
                        # §2.54: adaptive threshold — analog material tolerates more drift
                        effective_max_reg = self._compute_adaptive_pair_threshold(max_reg, state)
                        if drift < effective_max_reg:
                            return (
                                f"{description}: {guard_goal} drift={drift:+.3f} < "
                                f"{effective_max_reg:.3f} (base={max_reg:.3f})"
                            )
        return None

    def _compute_adaptive_pair_threshold(
        self,
        base_threshold: float,
        state: InteractionGuardState,
    ) -> float:
        """§2.54 Scale critical-pair regression threshold for material/restorability.

        Analog carrier materials are inherently more affected by broadband-NR
        pairs (denoise+hiss-reduction) — their natuerlichkeit metric drops because
        broadband noise was artificially elevating the "uniform-spectrum = natural"
        proxy score.  This is the Reference Paradox (§2.44), not real degradation.

        Returns a negative float (threshold); more negative = more permissive.
        Never tighter than base_threshold, never looser than 5× base_threshold.
        """
        _ANALOG_SCALE: dict[str, float] = {
            "wax_cylinder": 5.0,
            "shellac": 4.0,
            "wire_recording": 3.5,
            "vinyl": 3.0,
            "reel_tape": 2.5,
            "tape": 2.5,
            "optical_film": 2.0,
            "cassette": 2.0,
            "radio_broadcast": 1.8,
            "mp3_low": 1.5,
        }
        material = str(getattr(state, "material_type", "unknown"))
        mat_scale = _ANALOG_SCALE.get(material, 1.0)

        # Restorability factor: lower restorability → more tolerance
        restorability = float(np.clip(getattr(state, "restorability_score", 50.0), 0.0, 100.0))
        rest_factor = 1.0 + max(0.0, (55.0 - restorability) / 55.0)  # 1.0–2.0 range

        effective = base_threshold * mat_scale * rest_factor
        # base_threshold is negative; clip: most permissive (5×) ← → strictest (1×)
        return float(np.clip(effective, 5.0 * base_threshold, base_threshold))

    def _compute_gdd_threshold(self, phase_id: str, state: InteractionGuardState) -> float:
        """§2.54 Adaptive GDD threshold based on material and restorability.

        Spectral-subtraction phases on heavily degraded analog material generate
        genuinely high group-delay deviations because noise-dominated bins change
        phase after NR — not a real phase-coherence problem.  The threshold must
        account for this.

        Base thresholds:
          - General STFT phases: MAX_GROUP_DELAY_DEVIATION_MS (5 ms)
          - Spectral-subtraction (_SPECTRAL_SUBTRACTION_PHASES): MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL (10 ms)

        Adaptive factors:
          - Restorability < 70 → +50 % per 10 points below 70 (capped at ×2.5)
          - Analog material (vinyl, shellac, tape, reel_tape) → ×1.4 additional
        """
        base = (
            MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL
            if phase_id in _SPECTRAL_SUBTRACTION_PHASES
            else MAX_GROUP_DELAY_DEVIATION_MS
        )
        # Restorability factor: lower → more noise-phase contamination
        restorability = float(getattr(state, "restorability_score", 50.0))
        if restorability < 70.0:
            rest_factor = 1.0 + min(1.5, (70.0 - restorability) / 20.0)  # 1.0–2.5
        else:
            rest_factor = 1.0
        # Analog-material factor: noise-dominated bins produce false GDD spikes
        _ANALOG_MATERIALS = {"vinyl", "shellac", "tape", "reel_tape", "wire_recording", "wax_cylinder"}
        material = str(getattr(state, "material_type", "unknown"))
        mat_factor = 1.4 if material in _ANALOG_MATERIALS else 1.0
        return base * rest_factor * mat_factor

    def _check_group_delay(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        sr: int,
        phase_id: str = "",
        state: InteractionGuardState | None = None,
    ) -> bool:
        """Check STFT group delay deviation stays within per-phase threshold.

        §2.54: threshold is adaptive — computed via _compute_gdd_threshold() from
        material type and restorability score.  Falls back to fixed constants when
        state is None.

        Returns True if OK, False if deviation exceeded.
        """
        if state is not None:
            threshold_ms = self._compute_gdd_threshold(phase_id, state)
        else:
            threshold_ms = (
                MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL
                if phase_id in _SPECTRAL_SUBTRACTION_PHASES
                else MAX_GROUP_DELAY_DEVIATION_MS
            )
        ref_mono = reference if reference.ndim == 1 else np.mean(reference, axis=0)
        cur_mono = current if current.ndim == 1 else np.mean(current, axis=0)
        min_len = min(len(ref_mono), len(cur_mono))
        if min_len < 2048:
            return True  # too short to measure meaningfully

        ref_mono = ref_mono[:min_len]
        cur_mono = cur_mono[:min_len]

        n_fft = 2048
        # Use a representative segment from the middle
        mid = min_len // 2
        start = max(0, mid - n_fft)
        end = start + n_fft

        win = np.hanning(n_fft).astype(np.float32)
        ref_fft = np.fft.rfft(ref_mono[start:end] * win)
        cur_fft = np.fft.rfft(cur_mono[start:end] * win)

        # Group delay = -d(phase)/d(omega)
        # Approximate: phase difference between consecutive bins
        ref_phase = np.unwrap(np.angle(ref_fft))
        cur_phase = np.unwrap(np.angle(cur_fft))

        phase_diff = cur_phase - ref_phase
        # Group delay deviation in samples
        # dphase/dbin approximation
        gd_deviation_samples = np.abs(np.diff(phase_diff)) / (2.0 * np.pi / n_fft)

        # Ignore extreme outliers (DC, Nyquist region)
        gd_valid = gd_deviation_samples[5:-5]
        if len(gd_valid) == 0:
            return True

        max_deviation_samples = float(np.percentile(gd_valid, 95))
        max_deviation_ms = 1000.0 * max_deviation_samples / sr

        logger.debug(
            "§2.48 STFT group delay deviation: %.2f ms (threshold: %.1f ms, phase=%s)",
            max_deviation_ms,
            threshold_ms,
            phase_id or "unknown",
        )

        return max_deviation_ms <= threshold_ms

    def _is_better_checkpoint(
        self,
        state: InteractionGuardState,
        goals: dict[str, float],
    ) -> bool:
        """Check if current goals are better than best checkpoint."""
        if state.best_checkpoint is None:
            return True

        # Mean P1/P2 score comparison
        curr_mean = np.mean([goals.get(g, 0.0) for g in P1_P2_GOALS if g in goals])
        best_mean = np.mean(
            [
                state.best_checkpoint.goal_scores.get(g, 0.0)
                for g in P1_P2_GOALS
                if g in state.best_checkpoint.goal_scores
            ]
        )
        return float(curr_mean) >= float(best_mean) - 0.001  # allow tiny tolerance
