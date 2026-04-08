"""
Aurik 9 — Kumulative-Phasen-Interaktions-Guard §2.48 [RELEASE_MUST]
====================================================================
Guards against destructive cumulative effects of multiple phases:
- P1/P2 cumulative drift monitoring with rollback
- Critical phase pair interaction detection
- STFT phase coherence after ≥3 STFT phases
- Checkpoint management with max 2 consecutive rollbacks → pipeline stop

Reference: Spec 02 §2.48, Spec 02 §2.29d (P1/P2 hart)
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

# Max cumulative drift before rollback (§2.48)
MAX_CUMULATIVE_DRIFT = -0.05

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

# Max consecutive rollbacks before pipeline stop.
# 3 = material-adaptive headroom for heavily degraded vintage material
# (tape gen-3, SNR < 6 dB, Wow/Flutter=1.0) — §2.47 Adaptive-Intelligence.
# 2 caused premature Pipeline-Stop on difficult tape after only phase_29+phase_20.
MAX_CONSECUTIVE_ROLLBACKS = 3


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
    ) -> None:
        """Set the pre-pipeline P1/P2 baseline scores before any phases run.

        §2.29c Baseline-Capping: Defekte inflationieren bestimmte Metriken
        künstlich (Rauschen → tonal_center erhöht durch gleichmäßige Chroma-Lifts;
        Hall → waerme/transparenz erhöht). Subtractive Phasen werden gegen diesen
        inflationierten Wert gemessen → false Regression → Rollback auf verbessertes Audio.

        Lösung: Baseline wird auf canonical_threshold + 0.05 Headroom gedeckelt.
        Dieser Schritt greift nur auf P1/P2 Goals für spätere Drift-Checks.
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
        state.best_checkpoint = InteractionGuardCheckpoint(
            audio=audio.copy(),
            phase_id="__pre_pipeline__",
            goal_scores=dict(goal_scores),  # unkapped — für echte P1/P2-Ziele
            stft_phase_count=0,
        )
        logger.debug(
            "§2.48 InteractionGuard: baseline set (capped) — P1/P2 goals: %s",
            {k: f"{v:.3f}" for k, v in capped.items() if k in P1_P2_GOALS},
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
        if state.pre_pipeline_goals:
            cumulative_drift = {}
            for g in P1_P2_GOALS:
                if _is_subtractive and g in _DEFECT_INFLATED_SUBTRACTIVE_GOALS:
                    continue  # tonal_center für SUBTRACTIVE-Phasen nicht prüfen (§2.29c)
                if g in current_goals and g in state.pre_pipeline_goals:
                    cumulative_drift[g] = current_goals[g] - state.pre_pipeline_goals[g]

            worst_drift = min(cumulative_drift.values()) if cumulative_drift else 0.0
            if worst_drift < MAX_CUMULATIVE_DRIFT:
                logger.warning(
                    "§2.48 P1/P2 cumulative drift after %s: %s → rollback to %s",
                    phase_id,
                    {k: f"{v:+.3f}" for k, v in cumulative_drift.items() if v < MAX_CUMULATIVE_DRIFT},
                    state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                )
                state.rollback_log.append(
                    InteractionRollback(
                        phase_id=phase_id,
                        reason=f"P1/P2 cumulative drift {worst_drift:+.3f}",
                        drift=cumulative_drift,
                        rolled_back_to=state.best_checkpoint.phase_id if state.best_checkpoint else "pre_pipeline",
                    )
                )
                state.consecutive_rollbacks += 1
                rolled_back = True
                # §2.48 STFT-Zähler korrigieren: Phase war im Audio-Zustand nicht
                # persistent — Zähler muss Checkpoint-Stand widerspiegeln.
                if phase_id in STFT_PHASES and phase_id in state.stft_phases_executed:
                    state.stft_phases_executed.remove(phase_id)

                if state.consecutive_rollbacks >= MAX_CONSECUTIVE_ROLLBACKS:
                    state.should_stop = True
                    logger.warning(
                        "§2.48 Max consecutive rollbacks (%d) reached — pipeline stop, export on best checkpoint",
                        MAX_CONSECUTIVE_ROLLBACKS,
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
            state.consecutive_rollbacks += 1
            # §2.48 STFT-Zähler korrigieren bei Critical-Pair-Rollback
            if phase_id in STFT_PHASES and phase_id in state.stft_phases_executed:
                state.stft_phases_executed.remove(phase_id)
            rolled_back = True

            if state.consecutive_rollbacks >= MAX_CONSECUTIVE_ROLLBACKS:
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
            )
            _gdd_threshold = (
                MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL
                if phase_id in _SPECTRAL_SUBTRACTION_PHASES
                else MAX_GROUP_DELAY_DEVIATION_MS
            )
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

                if state.consecutive_rollbacks >= MAX_CONSECUTIVE_ROLLBACKS:
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
        """Check if current phase + already-executed phases form a critical pair."""
        for pair_phases, guard_goal, description, max_reg in CRITICAL_PAIRS:
            if phase_id in pair_phases:
                other_phases = pair_phases - {phase_id}
                if other_phases.issubset(state.executed_phases):
                    # Both phases of the pair have executed
                    if guard_goal in goals and guard_goal in state.pre_pipeline_goals:
                        drift = goals[guard_goal] - state.pre_pipeline_goals[guard_goal]
                        if drift < max_reg:
                            return f"{description}: {guard_goal} drift={drift:+.3f} < {max_reg}"
        return None

    def _check_group_delay(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        sr: int,
        phase_id: str = "",
    ) -> bool:
        """Check STFT group delay deviation stays within per-phase threshold.

        Spectral-subtraction phases (_SPECTRAL_SUBTRACTION_PHASES) use
        MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL (10 ms); all others use
        MAX_GROUP_DELAY_DEVIATION_MS (5 ms).

        Returns True if OK, False if deviation exceeded.
        """
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
