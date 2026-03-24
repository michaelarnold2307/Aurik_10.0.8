from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import threading
import time

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
        use_versa_in_loop: bool = False,
        max_retries: int | None = None,
    ) -> None:
        # Legacy-Kompatibilitaet: max_retries entspricht max_iterations.
        if max_retries is not None:
            max_iterations = int(max_retries)
        self.max_iterations = max(1, int(max_iterations))
        self.convergence_delta = max(1e-6, float(convergence_delta))
        self.sample_rate = int(sample_rate)
        self.excellence_mode = bool(excellence_mode)
        self.material = str(material)
        self.use_mert = bool(use_mert)
        self.use_pqs_in_loop = bool(use_pqs_in_loop)
        self.use_versa_in_loop = bool(use_versa_in_loop)
        self.goal_priority_callback: Callable[[np.ndarray, np.ndarray], tuple[bool, str]] | None = None
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
                versa = self._versa_score_fn(audio, sr)
                versa_mos = float(getattr(versa, "mos", np.nan))
                if np.isfinite(versa_mos):
                    self._last_score_source = "versa"
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

    def _adaptive_convergence_delta(self, current_mos: float) -> float:
        """Adaptive convergence threshold based on current MOS level.

        High-quality audio (MOS > 4.0) uses tighter delta to squeeze out
        remaining improvements. Low-quality uses relaxed delta to avoid
        wasting iterations on negligible gains.
        """
        if current_mos >= 4.0:
            return max(1e-6, self.convergence_delta * 0.25)  # 0.005 for default 0.02
        if current_mos >= 3.5:
            return self.convergence_delta  # 0.02 default
        return min(0.05, self.convergence_delta * 2.5)  # 0.05 for poor audio

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

        # Phasen-Listen-Modus: Liste von (id, fn, kwargs)-Tupeln
        if isinstance(phases_or_fn, list):

            def _combined_fn(a: np.ndarray, _sr2: int) -> np.ndarray:
                out = a
                for _pid, _fn, _kw in phases_or_fn:
                    try:
                        out = _fn(out, _sr2, **_kw) if _kw else _fn(out, _sr2)
                    except Exception as phase_exc:
                        logger.debug(
                            "FeedbackChain: phase callable failed (%s): %s",
                            _pid,
                            phase_exc,
                        )
                return out

            improve_fn: Callable[[np.ndarray, int], np.ndarray] = _combined_fn
        else:
            improve_fn = phases_or_fn  # type: ignore[assignment]

        _t0 = time.perf_counter()

        current = np.nan_to_num(np.asarray(audio, dtype=np.float32))

        # §Performance-Budget: ≤60s per minute audio for FeedbackChain (all iterations).
        _audio_dur_s = float(max(current.shape) if current.ndim == 2 else len(current)) / float(_sr)
        _time_budget_s = max(60.0, 60.0 * (_audio_dur_s / 60.0))  # 60s per minute, min 60s
        best = current.copy()
        best_mos = self._compute_iteration_score(best, _sr)
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
                    abort_result = _gpp.should_abort_iteration(_prev_goals, _curr_goals)
                    if abort_result.should_abort:
                        _log_entry = f"FeedbackChain Iteration {i} abgebrochen: {abort_result.reason}"
                        _goal_priority_log.append(_log_entry)
                        logger.warning("⚠ %s", _log_entry)
                        break  # Rollback auf best (§2.34)
                    _prev_goals = _curr_goals
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
                except Exception as mg_exc:
                    logger.debug("FeedbackChain: initial musical-goals read failed: %s", mg_exc)

            if mos > best_mos:
                best_mos = mos
                best = candidate.copy()

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

            # Regression guard
            if history[-1] < history[-2] - 0.05:
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
    if ml > 1:
        cov = _np.corrcoef(env_o[:ml], env_d[:ml])
        transient_score = float(_np.clip((cov[0, 1] + 1.0) / 2.0, 0.0, 1.0))
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
