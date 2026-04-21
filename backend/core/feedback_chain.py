from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modul-Konstanten (Spec §2.16 / §9.5)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_SCORE: float = 0.70  # Standard-Qualitätsschwelle (MOS-normalisiert)
EXCELLENCE_TARGET_SCORE: float = 0.85  # Verschärftes Ziel im Excellence-Modus
MUSIC_OVR_EXCELLENCE_THRESHOLD: float = 0.90  # Musik-OVR Schwelle für Excellence
HEADROOM_THRESHOLD: float = 0.03  # §2.33 PhysicalCeilingEstimator: Δ < 3 % → früher Abbruch


@dataclass
class FeedbackChainResult:
    audio: np.ndarray
    iterations: int
    converged: bool
    mos_history: list[float] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    phase_executions: list[dict] = field(default_factory=list)  # Ausgeführte Phasen-Protokolle
    overall_score: float = 0.0  # Gesamt-Score nach allen Iterationen
    total_retries: int = 0  # Kompatibilitaet: Alias fuer iterations
    total_time_s: float = 0.0  # Gesamtdauer der Feedback-Schleife in Sekunden
    ceiling_reached: bool = False  # §2.33: True wenn PhysicalCeiling frühzeitig erreicht
    analytics_overhead_s: float = 0.0  # Time spent in goal-measurement calls (excluded from RT budget)


class FeedbackChain:
    """Iterative quality loop with conservative convergence control."""

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_delta: float = 0.02,
        *,
        sample_rate: int = 48000,
        target_score: float | None = None,
        excellence_mode: bool = False,
        material: str = "auto",
        use_mert: bool = False,
        use_pqs_in_loop: bool = False,
        use_versa_in_loop: bool = True,  # §VERBOTEN: VERSA muss immer aktiv sein (§2.44)
        max_retries: int | None = None,
        restorability_score: float = 50.0,
        defect_severity_mean: float = 0.3,
    ) -> None:
        # Legacy-Kompatibilitaet: max_retries entspricht max_iterations.
        if max_retries is not None:
            max_iterations = int(max_retries)
        self.max_iterations = max(1, int(max_iterations))
        self.convergence_delta = max(1e-6, float(convergence_delta))
        self.sample_rate = int(sample_rate)
        self.excellence_mode = bool(excellence_mode)
        self.material = str(material)
        self.restorability_score = float(np.clip(restorability_score, 0.0, 100.0))
        self.defect_severity_mean = float(np.clip(defect_severity_mean, 0.0, 1.0))
        self.use_mert = bool(use_mert)
        self.use_pqs_in_loop = bool(use_pqs_in_loop)
        self.use_versa_in_loop = bool(use_versa_in_loop)
        self.goal_priority_callback: Callable[[np.ndarray, np.ndarray], tuple[bool, str]] | None = None
        self.goal_weights: dict[str, float] | None = None  # §2.56 Song-Goal-Importance
        self._pqs_score_fn: Callable[[np.ndarray, int], object] | None = None
        self._versa_score_fn: Callable[[np.ndarray, int], object] | None = None
        self._last_score_source: str = "heuristic_rms"
        if self.use_pqs_in_loop:
            try:
                from backend.core.perceptual_quality_scorer import score_audio_absolute

                self._pqs_score_fn = score_audio_absolute
            except Exception as exc:
                logger.debug("FeedbackChain: PQS scorer unavailable, heuristic fallback active: %s", exc)
        if self.use_versa_in_loop:
            try:
                from plugins.versa_plugin import get_versa_plugin

                _versa_plugin = get_versa_plugin()
                if _versa_plugin is not None:
                    self._versa_score_fn = _versa_plugin.score
            except Exception as exc:
                logger.debug("FeedbackChain: VERSA scorer unavailable, fallback active: %s", exc)
        # target_score: explizit gesetzt oder aus excellence_mode abgeleitet
        excellence_target = EXCELLENCE_TARGET_SCORE if excellence_mode else DEFAULT_TARGET_SCORE
        if target_score is not None:
            self.target_score = float(max(target_score, excellence_target))
        else:
            self.target_score = excellence_target

    @staticmethod
    def compute_perceptual_score(audio: np.ndarray) -> float:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
        return float(np.clip(1.0 + 4.0 * (1.0 - np.exp(-8.0 * rms)), 1.0, 5.0))

    def _compute_iteration_score(self, audio: np.ndarray, sr: int) -> float:
        """Computes loop score with PQS-first strategy and heuristic fallback.

        Primary: VERSA mos (if enabled) or PerceptualQualityScorer.score_audio_absolute(...).
        Fallback: legacy RMS heuristic from compute_perceptual_score().
        """
        if self._versa_score_fn is not None:
            try:
                versa_mos = self._compute_versa_segmented_score(audio, sr)
                if np.isfinite(versa_mos):
                    self._last_score_source = "versa_segmented"
                    return float(np.clip(versa_mos, 1.0, 5.0))
            except Exception as exc:
                logger.debug("FeedbackChain: VERSA loop score failed, trying PQS fallback: %s", exc)
        if self._pqs_score_fn is not None:
            try:
                pqs = self._pqs_score_fn(audio, sr)
                pqs_mos = float(getattr(pqs, "pqs_mos", getattr(pqs, "mos", np.nan)))
                if np.isfinite(pqs_mos):
                    self._last_score_source = "pqs_absolute"
                    return float(np.clip(pqs_mos, 1.0, 5.0))
            except Exception as exc:
                logger.debug("FeedbackChain: PQS loop score failed, fallback active: %s", exc)
        self._last_score_source = "heuristic_rms"
        return self.compute_perceptual_score(audio)

    def _compute_versa_segmented_score(self, audio: np.ndarray, sr: int) -> float:
        """Compute VERSA MOS on up to 5 representative segments, aggregate via min.

        Motivation: avoid local quality collapses being hidden by a single global MOS.
        """
        if self._versa_score_fn is None:
            return float("nan")

        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            if arr.shape[1] <= 2 and arr.shape[0] > arr.shape[1]:
                mono = arr.mean(axis=1)
            elif arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]:
                mono = arr.mean(axis=0)
            else:
                mono = arr.mean(axis=-1)
        else:
            mono = arr.ravel()

        win = int(sr * 30)  # align with SingMOS 30 s design window
        if mono.size <= win:
            versa = self._versa_score_fn(mono, sr)
            return float(getattr(versa, "mos", np.nan))

        n_segments = int(np.clip(np.ceil(mono.size / win), 3, 5))
        half = win // 2
        centers = np.linspace(half, mono.size - half, n_segments, dtype=int)

        seg_scores: list[float] = []
        for c in centers:
            s = int(max(0, c - half))
            e = int(min(mono.size, s + win))
            seg = mono[s:e]
            if seg.size < int(sr * 5):
                continue
            versa = self._versa_score_fn(seg, sr)
            mos = float(getattr(versa, "mos", np.nan))
            if np.isfinite(mos):
                seg_scores.append(float(np.clip(mos, 1.0, 5.0)))

        if not seg_scores:
            versa = self._versa_score_fn(mono, sr)
            return float(getattr(versa, "mos", np.nan))

        # Conservative aggregation: bottleneck quality dominates listener perception.
        return float(np.min(seg_scores))

    def _adaptive_convergence_delta(self, current_mos: float) -> float:
        """Adaptive convergence threshold based on current MOS level and §2.56 goal_weights.

        High-quality audio (MOS > 4.0) uses tighter delta to squeeze out
        remaining improvements. Low-quality uses relaxed delta to avoid
        wasting iterations on negligible gains.

        §2.56: P1/P2-heavy songs (naturalness/authenticity) get a tighter
        delta at high MOS to extract maximum perceptual quality on the
        goals that matter most. P4/P5-heavy songs remain at standard delta.
        """
        if current_mos >= 4.0:
            base_delta = max(1e-6, self.convergence_delta * 0.25)  # 0.005 for default 0.02
        elif current_mos >= 3.5:
            base_delta = self.convergence_delta  # 0.02 default
        else:
            base_delta = min(0.05, self.convergence_delta * 2.5)  # 0.05 for poor audio

        # §2.56: tighten convergence for P1/P2-dominant songs at high quality
        if current_mos >= 4.0 and isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = ("natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation")
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            if _p1p2_mean > 1.1:
                # Tighten by up to 40% for strongly P1/P2-dominant songs
                _tighten = float(min(0.40, (_p1p2_mean - 1.0) * 0.40))
                base_delta = max(1e-8, base_delta * (1.0 - _tighten))

        return base_delta

    # -------------------------------------------------------------------
    # §2.54 Adaptive thresholds — material/restorability/defect-aware
    # -------------------------------------------------------------------
    _POOR_MATERIALS = frozenset(
        {
            "shellac",
            "wax_cylinder",
            "wire_recording",
            "acetate_disc",
        }
    )
    _ANALOG_MATERIALS = frozenset(
        {
            "vinyl",
            "tape",
            "reel_tape",
            "cassette",
            "minidisc",
        }
    )

    def _compute_adaptive_prune_threshold(self, is_restorative: bool) -> float:
        """§2.54 + §2.56: Material- and goal-importance-adaptive pruning threshold.

        Restorative phases on severely degraded material need much more
        lenient thresholds — their MOS-proxy drop is expected (removing
        energy that was defect, not content).

        §2.56: When P1/P2 goals (natuerlichkeit, authentizitaet, tonal_center,
        timbre_authentizitaet, artikulation) carry high weight for this song,
        we apply a *conservative bias*: the threshold is tightened (less negative)
        so that phases that improve these critical goals are less likely to be
        pruned based on an incomplete MOS proxy.  Conversely, a P4/P5-dominated
        profile (brillanz, raumtiefe) loosens the threshold because minor MOS
        fluctuations there are tolerable.

        Returns a negative float (more negative = more lenient).
        """
        # Base: -0.01 enhancement, -0.05 restorative (legacy fallback)
        base = -0.05 if is_restorative else -0.01

        # Material factor: poor materials get 2x–3x more lenient
        mat = self.material.lower() if self.material else "unknown"
        if mat in self._POOR_MATERIALS:
            mat_factor = 3.0
        elif mat in self._ANALOG_MATERIALS:
            mat_factor = 2.0
        else:
            mat_factor = 1.0

        # Restorability factor: lower restorability → more lenient
        # restorability_score 0–100: 0=pristine, 100=heavily degraded
        rest_factor = 1.0 + (self.restorability_score / 100.0) * 1.5  # [1.0, 2.5]

        # Defect severity: higher → more lenient
        sev_factor = 1.0 + self.defect_severity_mean * 1.0  # [1.0, 2.0]

        # Combined: base * max(factors) — use max to avoid over-compounding
        adaptive = base * max(mat_factor, rest_factor, sev_factor)

        # §2.56 Goal-importance bias: P1/P2 heavy → tighten threshold (conservative).
        # A song where naturalness/authenticity matter most should not aggressively prune
        # phases that might be nudging those delicate goals in the right direction.
        _gw_bias = 0.0
        if isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = (
                "natuerlichkeit",
                "authentizitaet",
                "tonal_center",
                "timbre_authentizitaet",
                "artikulation",
            )
            _P4P5_KEYS = (
                "brillanz",
                "raumtiefe",
                "waerme",
                "bassgewalt",
            )
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p4p5_vals = [self.goal_weights.get(k, 1.0) for k in _P4P5_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            _p4p5_mean = float(sum(_p4p5_vals) / max(len(_p4p5_vals), 1))
            # bias ∈ [-0.05, +0.05]: positive bias = tighten (less pruning for P1/P2 songs)
            _gw_bias = float((_p1p2_mean - _p4p5_mean) * 0.025)
            _gw_bias = float(max(-0.05, min(0.05, _gw_bias)))

        # Apply bias: tighten (toward 0) when P1/P2 heavy, loosen when P4/P5 heavy.
        adaptive_biased = adaptive + _gw_bias
        # Clamp: never more lenient than -0.30, never stricter than -0.005
        return float(max(-0.30, min(-0.005, adaptive_biased)))

    def _compute_adaptive_mos_regression_tolerance(self) -> float:
        """§2.54 + §2.56: Material- and goal-importance-adaptive MOS regression tolerance.

        Poor material with heavy defects needs more tolerance — each
        iteration may transiently worsen MOS as it repairs deeper damage.

        §2.56: Songs with dominant P1/P2 goals get a small tolerance reduction
        to prevent accepting spurious regressions on critical perceptual goals.

        Returns a positive float (higher = more tolerant).
        """
        # Base: 0.05 (legacy fallback)
        base = 0.05

        mat = self.material.lower() if self.material else "unknown"
        if mat in self._POOR_MATERIALS:
            mat_bonus = 0.10  # shellac/wax allow up to 0.15 regression
        elif mat in self._ANALOG_MATERIALS:
            mat_bonus = 0.05  # vinyl/tape allow up to 0.10
        else:
            mat_bonus = 0.0

        # Higher restorability (more degraded) → more tolerance
        rest_bonus = (self.restorability_score / 100.0) * 0.08  # up to +0.08

        # Higher defect severity → more tolerance
        sev_bonus = self.defect_severity_mean * 0.05  # up to +0.05

        tolerance = base + max(mat_bonus, rest_bonus, sev_bonus)

        # §2.56: P1/P2 heavy songs → tighten tolerance slightly (max -0.015)
        # so we don't accept regressions in the most perceptually critical goals.
        if isinstance(self.goal_weights, dict) and self.goal_weights:
            _P1P2_KEYS = ("natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation")
            _p1p2_vals = [self.goal_weights.get(k, 1.0) for k in _P1P2_KEYS]
            _p1p2_mean = float(sum(_p1p2_vals) / max(len(_p1p2_vals), 1))
            if _p1p2_mean > 1.0:
                # Over-weighted P1/P2 → reduce tolerance proportionally, capped at -0.015
                _p1p2_reduction = float(min(0.015, (_p1p2_mean - 1.0) * 0.01))
                tolerance = max(base, tolerance - _p1p2_reduction)

        # Clamp: [0.03, 0.25] — never allow unlimited regression
        return float(np.clip(tolerance, 0.03, 0.25))

    def run(
        self,
        audio: np.ndarray,
        phases_or_fn: Callable[[np.ndarray, int], np.ndarray] | list,
        sr: int | None = None,
        ceiling: float | None = None,
    ) -> FeedbackChainResult:
        """Führt die Feedback-Schleife aus.

        Akzeptiert zwei Aufruf-Varianten:
          - run(audio, improve_fn, sr)           – klassisch
          - run(audio, [(phase_id, fn, kwargs)]) – Phasen-Listen-Modus
        """
        _sr = sr if sr is not None else self.sample_rate
        assert _sr == 48000, f"FeedbackChain.run() erwartet SR=48000, erhalten: {_sr}"

        # --- Adaptive Per-Phase Pruning for phase-list mode ---
        # In the first iteration, evaluate each phase individually.
        # Phases that degrade MOS (Δ < -0.01) are pruned from subsequent iterations.
        # This prevents a harmful phase from cancelling gains of helpful ones.
        _phase_list_mode = isinstance(phases_or_fn, list)
        _active_phases: list = list(phases_or_fn) if _phase_list_mode else []
        _pruned_phases: list[str] = []
        _phase_deltas: dict[str, float] = {}  # phase_id → MOS delta from first iteration

        # ── §2.47 GP-Advisory Strength Lookup ──────────────────────────────
        # Consult GP memory for material-genre-specific strength priors before
        # running the loop.  If GP has learned good parameters from previous
        # songs with the same material, inject them as advisory kwargs hints.
        _gp_advisory_applied = False
        if _phase_list_mode and len(_active_phases) > 0:
            try:
                from backend.core.gp_parameter_optimizer import get_optimizer as _get_gp_opt

                _gp_opt = _get_gp_opt()
                _mat = self.material if self.material != "auto" else "unknown"
                _proposal = _gp_opt.propose(material=_mat, n_init=5)
                if _proposal is not None and hasattr(_proposal, "params") and _proposal.params:
                    _gp_proposal = dict(_proposal.params)
                    _strength_keys = {
                        "noise_reduction_strength": ("phase_03",),
                        "reverb_reduction_strength": ("phase_49", "phase_20"),
                        "eq_correction_strength": ("phase_04", "phase_06"),
                        "harmonic_preservation": ("phase_07", "phase_08"),
                        "transient_strength": ("phase_08",),
                    }
                    _hints_applied = 0
                    for gp_key, phase_prefixes in _strength_keys.items():
                        if gp_key in _gp_proposal:
                            gp_val = float(np.clip(_gp_proposal[gp_key], 0.1, 1.0))
                            for idx, (_pid, _fn, _kw) in enumerate(_active_phases):
                                pid_str = str(_pid)
                                if any(pid_str.startswith(pp) for pp in phase_prefixes):
                                    if "strength" not in (_kw or {}):
                                        _kw_new = dict(_kw) if _kw else {}
                                        _kw_new["gp_advisory_strength"] = gp_val
                                        _active_phases[idx] = (_pid, _fn, _kw_new)
                                        _hints_applied += 1
                    if _hints_applied > 0:
                        _gp_advisory_applied = True
                        logger.info(
                            "FeedbackChain: GP advisory applied %d strength hints (material=%s)",
                            _hints_applied,
                            _mat,
                        )
            except Exception as _gp_exc:
                logger.debug("FeedbackChain: GP advisory lookup non-blocking: %s", _gp_exc)

        if _phase_list_mode:

            def _build_combined_fn(active_phase_list: list):
                """Build improve_fn from currently active phases."""

                def _combined_fn(a: np.ndarray, _sr2: int) -> np.ndarray:
                    out = a
                    for _pid, _fn, _kw in active_phase_list:
                        try:
                            out = _fn(out, _sr2, **_kw) if _kw else _fn(out, _sr2)
                        except Exception as phase_exc:
                            logger.debug(
                                "FeedbackChain: phase callable failed (%s): %s",
                                _pid,
                                phase_exc,
                            )
                    return out

                return _combined_fn

            improve_fn: Callable[[np.ndarray, int], np.ndarray] = _build_combined_fn(_active_phases)
        else:
            improve_fn = phases_or_fn  # type: ignore[assignment]

        _t0 = time.perf_counter()

        current = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        # §Performance-Budget: ≤120s per minute audio for FeedbackChain (all iterations).
        # Spec §2.38: FeedbackChain ≤ 120 s per minute audio.
        _audio_dur_s = float(max(current.shape) if current.ndim == 2 else len(current)) / float(_sr)
        _time_budget_s = max(120.0, 2.0 * _audio_dur_s)  # 120s per minute, min 120s
        best = current.copy()
        _t_before_init_score = time.perf_counter()
        best_mos = self._compute_iteration_score(best, _sr)
        _init_score_elapsed = time.perf_counter() - _t_before_init_score
        if _init_score_elapsed > 30.0:
            logger.warning(
                "FeedbackChain: initial score call took %.1fs (audio=%.0fs) — "
                "likely ML scorer without length cap; iterations will be skipped if budget exhausted",
                _init_score_elapsed,
                _audio_dur_s,
            )
        history = [best_mos]
        _score_sources = [self._last_score_source]
        _ceiling_reached = False

        # §2.34 GoalPriorityProtocol — Stufe-1/2-Regression löst sofortigen Rollback aus
        _gpp = None
        try:
            from backend.core.goal_priority_protocol import GoalPriorityProtocol

            _gpp = GoalPriorityProtocol()
        except Exception as gpp_exc:
            logger.debug("FeedbackChain: GoalPriorityProtocol unavailable: %s", gpp_exc)

        _prev_goals: dict[str, float] = {}
        _goal_priority_log: list[str] = []
        _phase_executions: list[dict] = []

        # Max audio window for goal regression checks — 30 s is sufficient to
        # detect P1/P2 regressions; measuring the full signal wastes CPU budget.
        _GOAL_WINDOW_SAMPLES = int(_sr * 30.0)

        def _goal_window(a: np.ndarray) -> np.ndarray:
            """Return a centre-slice ≤ 30 s for goal measurement."""
            total = a.shape[-1] if a.ndim == 2 else len(a)
            if total <= _GOAL_WINDOW_SAMPLES:
                return a
            start = (total - _GOAL_WINDOW_SAMPLES) // 2
            return (
                a[..., start : start + _GOAL_WINDOW_SAMPLES] if a.ndim == 2 else a[start : start + _GOAL_WINDOW_SAMPLES]
            )

        # §9.8 Goal-vector candidate selection — track how many goals pass thresholds
        _best_goal_pass_count: int = -1  # -1 = not yet measured
        _curr_goal_pass_count: int = -1

        converged = False
        for i in range(1, self.max_iterations + 1):
            # §Performance-Budget: abort if time budget exceeded
            _elapsed = time.perf_counter() - _t0
            if _elapsed > _time_budget_s:
                logger.warning(
                    "FeedbackChain: time budget exceeded (%.1fs > %.1fs) — aborting at iteration %d",
                    _elapsed,
                    _time_budget_s,
                    i,
                )
                break

            # --- Adaptive Pruning: after iteration 1, evaluate each phase individually ---
            # On the first iteration, run all phases as a bundle to get the combined effect.
            # Then measure each phase's individual contribution and prune harmful ones.
            if _phase_list_mode and i == 2 and len(_active_phases) > 1:
                _pre_prune_audio = current.copy()
                _base_mos = self._compute_iteration_score(_pre_prune_audio, _sr)
                _surviving_phases = []
                for _pid, _fn, _kw in _active_phases:
                    try:
                        _test_out = _fn(_pre_prune_audio, _sr, **_kw) if _kw else _fn(_pre_prune_audio, _sr)
                        _test_out = np.clip(np.nan_to_num(np.asarray(_test_out, dtype=np.float32)), -1.0, 1.0)
                        _test_mos = self._compute_iteration_score(_test_out, _sr)
                        _delta = _test_mos - _base_mos
                        _phase_deltas[str(_pid)] = float(_delta)
                        # §2.54: Restorative phases (denoise, click, dropout) intentionally
                        # remove energy → MOS proxy may drop slightly vs. defect-laden
                        # reference. Use material-adaptive threshold (§2.54) to avoid
                        # pruning legitimate carrier-repair phases.
                        _RESTORATIVE_PREFIXES = (
                            "phase_01",
                            "phase_02",
                            "phase_03",
                            "phase_05",
                            "phase_09",
                            "phase_12",
                            "phase_18",
                            "phase_20",
                            "phase_23",
                            "phase_24",
                            "phase_27",
                            "phase_28",
                            "phase_29",
                            "phase_30",
                            "phase_49",
                            "phase_50",
                            "phase_55",
                            "phase_56",
                            "phase_57",
                        )
                        _is_restorative = any(str(_pid).startswith(rp) for rp in _RESTORATIVE_PREFIXES)
                        _prune_threshold = self._compute_adaptive_prune_threshold(_is_restorative)
                        if _delta >= _prune_threshold:
                            _surviving_phases.append((_pid, _fn, _kw))
                            logger.debug(
                                "FeedbackChain: phase %s kept (Δ=%.4f)",
                                _pid,
                                _delta,
                            )
                        else:
                            _pruned_phases.append(str(_pid))
                            logger.info(
                                "FeedbackChain: phase %s pruned — degraded MOS by %.4f",
                                _pid,
                                _delta,
                            )
                    except Exception as _eval_exc:
                        _surviving_phases.append((_pid, _fn, _kw))
                        logger.debug(
                            "FeedbackChain: phase %s evaluation failed (%s) — keeping",
                            _pid,
                            _eval_exc,
                        )
                if _surviving_phases and len(_surviving_phases) < len(_active_phases):
                    _active_phases = _surviving_phases
                    improve_fn = _build_combined_fn(_active_phases)
                    logger.info(
                        "FeedbackChain: pruned %d/%d phases, %d remaining",
                        len(_pruned_phases),
                        len(_pruned_phases) + len(_active_phases),
                        len(_active_phases),
                    )
                elif not _surviving_phases:
                    logger.info("FeedbackChain: all phases degrade quality — converging early")
                    converged = True
                    break

            candidate = improve_fn(current, _sr)
            candidate = np.clip(np.nan_to_num(np.asarray(candidate, dtype=np.float32)), -1.0, 1.0)
            mos = self._compute_iteration_score(candidate, _sr)
            history.append(mos)
            _score_sources.append(self._last_score_source)
            _phase_executions.append({"iteration": i, "mos": float(mos)})

            # Optionaler externer Priority-Callback (z.B. aus UnifiedRestorerV3).
            if callable(self.goal_priority_callback):
                try:
                    _cb_abort, _cb_reason = self.goal_priority_callback(current, candidate)
                    if _cb_abort:
                        _log_entry = (
                            f"FeedbackChain Iteration {i} abgebrochen: {_cb_reason or 'goal-priority callback'}"
                        )
                        _goal_priority_log.append(_log_entry)
                        logger.warning("⚠ %s", _log_entry)
                        break
                except Exception as _cb_exc:
                    logger.debug("FeedbackChain goal_priority_callback fehlgeschlagen: %s", _cb_exc)

            # §2.34 GoalPriorityProtocol: Stufe-1/2-Ziele schützen
            # Skip internal GPP check when external goal_priority_callback is wired
            # (UV3 provides its own GPP callback that already calls measure_all).
            if _gpp is not None and _prev_goals and not callable(self.goal_priority_callback):
                try:
                    import time as _time_fc

                    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

                    _checker = MusicalGoalsChecker()
                    _t_goals = _time_fc.perf_counter()
                    _curr_goals = _checker.measure_all(_goal_window(candidate), _sr)
                    _analytics_dt = _time_fc.perf_counter() - _t_goals
                    self._last_analytics_overhead_s = getattr(self, "_last_analytics_overhead_s", 0.0) + _analytics_dt
                    abort_result = _gpp.should_abort_iteration(_prev_goals, _curr_goals, goal_weights=self.goal_weights)
                    if abort_result.should_abort:
                        _log_entry = f"FeedbackChain Iteration {i} abgebrochen: {abort_result.reason}"
                        _goal_priority_log.append(_log_entry)
                        logger.warning("⚠ %s", _log_entry)
                        break  # Rollback auf best (§2.34)
                    _prev_goals = _curr_goals
                    _curr_goal_pass_count = sum(
                        1 for g, v in _curr_goals.items() if v >= _checker.thresholds.get(g, 0.85)
                    )
                except Exception as _gpp_exc:
                    logger.debug("GoalPriorityProtocol in FeedbackChain nicht verfügbar: %s", _gpp_exc)
            elif _gpp is not None and not _prev_goals and not callable(self.goal_priority_callback):
                try:
                    import time as _time_fc

                    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

                    _checker = MusicalGoalsChecker()
                    _t_goals = _time_fc.perf_counter()
                    _prev_goals = _checker.measure_all(_goal_window(candidate), _sr)
                    _analytics_dt = _time_fc.perf_counter() - _t_goals
                    self._last_analytics_overhead_s = getattr(self, "_last_analytics_overhead_s", 0.0) + _analytics_dt
                    _curr_goal_pass_count = sum(
                        1 for g, v in _prev_goals.items() if v >= _checker.thresholds.get(g, 0.85)
                    )
                except Exception as mg_exc:
                    logger.debug("FeedbackChain: initial musical-goals read failed: %s", mg_exc)

            # §9.8 Goal-aware candidate selection: prefer candidates passing more goals
            _candidate_better = mos > best_mos
            if _curr_goal_pass_count >= 0 and _best_goal_pass_count >= 0:
                if _curr_goal_pass_count > _best_goal_pass_count:
                    _candidate_better = True  # more goals passed → accept
                elif _curr_goal_pass_count < _best_goal_pass_count:
                    _candidate_better = mos > best_mos + 0.05  # need significant MOS gain
            if _candidate_better:
                best_mos = mos
                best = candidate.copy()
                if _curr_goal_pass_count >= 0:
                    _best_goal_pass_count = _curr_goal_pass_count

            # §2.33 PhysicalCeilingEstimator: Frühzeitiger Abbruch wenn Ceiling erreicht
            # Tight headroom: allow iterations to push closer to ceiling
            _adaptive_headroom = 0.01
            if ceiling is not None and best_mos >= ceiling - _adaptive_headroom:
                _ceiling_reached = True
                converged = True
                logger.debug(
                    "FeedbackChain: ceiling=%.3f reached (MOS=%.3f) — Frühzeitiger Abbruch",
                    ceiling,
                    best_mos,
                )
                break

            if abs(history[-1] - history[-2]) < self._adaptive_convergence_delta(best_mos):
                converged = True
                break

            # §2.54 Adaptive regression guard — material/restorability-aware
            _mos_regression_tol = self._compute_adaptive_mos_regression_tolerance()
            if history[-1] < history[-2] - _mos_regression_tol:
                break

            current = candidate

        return FeedbackChainResult(
            audio=best,
            iterations=len(history) - 1,
            converged=converged,
            mos_history=history,
            metadata={
                "best_mos": best_mos,
                "goal_priority_log": _goal_priority_log,
                "score_source": _score_sources[-1] if _score_sources else self._last_score_source,
                "score_sources_seen": list(dict.fromkeys(_score_sources)),
                "score_fallback_used": bool(
                    (self.use_pqs_in_loop or self.use_versa_in_loop)
                    and any(src == "heuristic_rms" for src in _score_sources)
                ),
                "pruned_phases": _pruned_phases,
                "phase_deltas": _phase_deltas,
                "gp_advisory_applied": _gp_advisory_applied,
            },
            phase_executions=_phase_executions,
            overall_score=float(best_mos),
            total_retries=max(0, len(history) - 1),
            total_time_s=float(time.perf_counter() - _t0),
            ceiling_reached=_ceiling_reached,
            analytics_overhead_s=float(getattr(self, "_last_analytics_overhead_s", 0.0)),
        )


_instance: FeedbackChain | None = None
_lock = threading.Lock()


def get_feedback_chain() -> FeedbackChain:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FeedbackChain()
    return _instance


def compute_perceptual_score(
    original: np.ndarray,
    degraded: np.ndarray,
    *,
    sample_rate: int = 48000,
) -> dict:
    """Berechnet Perceptual-Score-Dict für original vs. degraded Audio (Spec §2.6).

    Rückgabe-Schlüssel:
        sisnr_db       SI-SNR in dB (Scale-Invariant SNR)
        spectral_flatness  Spektrale Flachheit ∈ [0, 1]
        snr_db         SNR in dB
        transient_score    Hüllkurven-Korrelation ∈ [0, 1]
        combined       Gewichteter Gesamt-Score ∈ [0, 1]
    """
    import numpy as _np

    _ = sample_rate  # wird für zukünftige SR-abhängige Metriken genutzt

    orig = _np.nan_to_num(_np.asarray(original, dtype=_np.float32)).ravel()
    deg = _np.nan_to_num(_np.asarray(degraded, dtype=_np.float32)).ravel()
    n = min(len(orig), len(deg))
    orig, deg = orig[:n], deg[:n]

    # — SI-SNR (Scale-Invariant SNR) ——————————————————————————————————————
    orig64 = orig.astype(_np.float64)
    deg64 = deg.astype(_np.float64)
    dot = float(_np.dot(orig64, orig64)) + 1e-12
    s_target = (_np.dot(deg64, orig64) / dot) * orig64
    e_noise = deg64 - s_target
    sisnr = 10.0 * float(_np.log10((_np.dot(s_target, s_target) + 1e-12) / (_np.dot(e_noise, e_noise) + 1e-12)))

    # — SNR ——————————————————————————————————————————————————————————————
    signal_power = float(_np.mean(orig64**2)) + 1e-12
    noise_power = float(_np.mean((deg64 - orig64) ** 2)) + 1e-12
    raw_snr = 10.0 * _np.log10(signal_power / noise_power)
    snr_db = float(_np.nan_to_num(raw_snr, nan=0.0, posinf=60.0, neginf=-60.0))

    # — Spectral Flatness ——————————————————————————————————————————————
    n_fft = min(2048, max(4, len(deg) // 4))
    spec = _np.abs(_np.fft.rfft(deg, n=n_fft)) + 1e-12
    spectral_flatness = float(
        _np.clip(
            _np.exp(float(_np.mean(_np.log(spec)))) / (float(_np.mean(spec)) + 1e-12),
            0.0,
            1.0,
        )
    )

    # — Transient Score (Hüllkurven-Korrelation) ——————————————————————
    hop = max(1, len(orig) // 200)
    env_o = _np.array(
        [float(_np.max(_np.abs(orig[i : i + hop]))) for i in range(0, len(orig) - hop, hop)],
        dtype=_np.float64,
    )
    env_d = _np.array(
        [float(_np.max(_np.abs(deg[i : i + hop]))) for i in range(0, len(deg) - hop, hop)],
        dtype=_np.float64,
    )
    ml = min(len(env_o), len(env_d))
    if ml > 1 and _np.std(env_o[:ml]) > 1e-10 and _np.std(env_d[:ml]) > 1e-10:
        cov = _np.corrcoef(env_o[:ml], env_d[:ml])
        _raw_corr = float(cov[0, 1])
        transient_score = float(_np.clip((_raw_corr + 1.0) / 2.0, 0.0, 1.0)) if _np.isfinite(_raw_corr) else 0.5
    elif ml > 1 and _np.std(env_o[:ml]) < 1e-10 and _np.std(env_d[:ml]) < 1e-10:
        transient_score = 1.0  # Both silent — trivially matched
    else:
        transient_score = 0.5

    # — Combined ———————————————————————————————————————————————————————
    sisnr_norm = float(_np.clip((sisnr + 20.0) / 80.0, 0.0, 1.0))
    combined = float(
        _np.clip(
            0.4 * sisnr_norm + 0.3 * transient_score + 0.3 * (1.0 - spectral_flatness),
            0.0,
            1.0,
        )
    )

    return {
        "sisnr_db": float(sisnr),
        "spectral_flatness": spectral_flatness,
        "snr_db": snr_db,
        "transient_score": transient_score,
        "combined": combined,
    }


# Convenience-Konstante: kritische Phasen-IDs (Spec §2.2 — TIER_1 + Dropout-Repair)
FEEDBACK_CRITICAL_PHASES: frozenset[int] = frozenset(
    {
        1,  # click_removal
        2,  # hum_removal
        3,  # denoise
        9,  # crackle_removal
        12,  # wow_flutter_fix
        24,  # dropout_repair
        29,  # tape_hiss_reduction
        55,  # diffusion_inpainting
    }
)

__all__ = [
    "DEFAULT_TARGET_SCORE",
    "EXCELLENCE_TARGET_SCORE",
    "FEEDBACK_CRITICAL_PHASES",
    "MUSIC_OVR_EXCELLENCE_THRESHOLD",
    "FeedbackChain",
    "FeedbackChainResult",
    "compute_perceptual_score",
    "get_feedback_chain",
]
