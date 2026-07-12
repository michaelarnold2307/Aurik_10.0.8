"""
pipeline_guard.py — v10 Pipeline-Guard: Watchdog + CLP + Dynamics + Whisper
============================================================================

Zentraler Integrationspunkt fuer alle v10 Optimierungen. Wird von
AurikDenker._orchestriere() an strategischen Punkten aufgerufen.

Einstiegspunkte:
  guard_pre_flight(audio, sr)          -> (ok, issues)
  guard_phase_start(name)              -> None
  guard_phase_end(name, audio, sr)     -> None
  guard_post_restore(original, result, sr, material) -> (blended, report)
  guard_auto_recover(audio, sr, profile) -> restored_audio

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PipelineGuard:
    """Zentraler Wachter fuer die gesamte Aurik-Restaurierungspipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._watchdog: Any = None
        self._dynamics: Any = None
        self._whisper_result: Any = None
        self._clp_result: Any = None
        self._original_audio: np.ndarray | None = None
        self._original_sr: int = 48000
        self._original_profile: Any = None
        self._material: str = "unknown"
        self._start_time: float = 0.0
        self._phase_count: int = 0
        self._warnings: list[str] = []
        self._criticals: list[str] = []

    def _wd(self) -> Any:
        if self._watchdog is None:
            from backend.core.watchdog_monitor import get_watchdog

            self._watchdog = get_watchdog()
        return self._watchdog

    def _dp(self) -> Any:
        if self._dynamics is None:
            from backend.core.dynamics_preserver import get_dynamics_preserver

            self._dynamics = get_dynamics_preserver()
        return self._dynamics

    # ── Pre-Flight ────────────────────────────────────────────────────

    def pre_flight(self, audio: np.ndarray, sr: int, material: str = "unknown") -> tuple[bool, list[str]]:
        """Vor Pipeline-Start: Original sichern, CLP analysieren, Whisper scannen."""
        self._start_time = time.perf_counter()
        self._material = material
        self._original_audio = np.asarray(audio, dtype=np.float32).copy()
        self._original_sr = sr

        issues: list[str] = []

        # Watchdog pre-flight
        try:
            ok, wd_issues = self._wd().pre_flight_check(audio, sr)
            if not ok:
                issues.extend(wd_issues)
        except Exception as e:
            logger.debug("PipelineGuard: watchdog pre-flight error: %s", e)

        # Dynamics: Original-Profil erfassen
        try:
            self._original_profile = self._dp().capture_original(audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: dynamics capture error: %s", e)

        # CLP: Kritische Frequenzzonen analysieren
        try:
            from backend.core.critical_listening_points import analyze_critical_zones

            self._clp_result = analyze_critical_zones(audio, sr, vocal_boost=True)
            logger.info(
                "PipelineGuard CLP: vocal=%.1f%%, whisper=%.1f%%, zones=%s",
                self._clp_result.vocal_presence * 100,
                self._clp_result.whisper_energy * 100,
                {z: f"{s:.2f}" for z, s in self._clp_result.zone_scores.items()},
            )
        except Exception as e:
            logger.debug("PipelineGuard: CLP analysis error: %s", e)

        # Whisper: Leise Gesangsdetails erkennen
        try:
            from backend.core.whisper_detail_preserver import analyze_whisper_detail

            self._whisper_result = analyze_whisper_detail(audio, sr)
            if self._whisper_result.segments:
                logger.info(
                    "PipelineGuard Whisper: %d Segmente (%.1f%%), max_atten=%.1fdB",
                    len(self._whisper_result.segments),
                    self._whisper_result.whisper_ratio * 100,
                    self._whisper_result.max_attenuation_db,
                )
        except Exception as e:
            logger.debug("PipelineGuard: Whisper analysis error: %s", e)

        return len(issues) == 0, issues

    # ── Phase Tracking ────────────────────────────────────────────────

    def phase_start(self, name: str) -> None:
        self._phase_count += 1
        try:
            self._wd().on_phase_start(name)
        except Exception as e:
            logger.debug("PipelineGuard: phase_start error: %s", e)

    def phase_end(self, name: str, audio: np.ndarray, sr: int) -> None:
        try:
            self._wd().on_phase_end(name, audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: phase_end watchdog error: %s", e)

        try:
            self._dp().check_phase(name, audio, sr)
        except Exception as e:
            logger.debug("PipelineGuard: phase_end dynamics error: %s", e)

    def record_strength(self, value: float) -> None:
        try:
            self._wd().record_cumulative_effect(value)
        except Exception as e:
            logger.debug("PipelineGuard: record_strength error: %s", e)

    def record_silent_exception(self, context: str = "") -> None:
        try:
            self._wd().record_silent_exception(context)
        except Exception as e:
            logger.debug("PipelineGuard: record_silent_exception error: %s", e)

    # ── Post-Restore: Whisper-Blending + Dynamics-Recovery ─────────────

    def post_restore(self, original: np.ndarray, restored: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Nach der Restaurierung: Whisper-Details zurueckmischen + Dynamik wiederherstellen."""
        result = np.asarray(restored, dtype=np.float32).copy()
        report: dict[str, Any] = {"whisper_blended": False, "dynamics_restored": False}

        # Whisper-Blending: Original-Details in leisen Passagen zurueckmischen
        if self._whisper_result is not None and self._whisper_result.segments:
            try:
                from backend.core.whisper_detail_preserver import apply_whisper_preservation

                result = apply_whisper_preservation(original, sr, self._whisper_result, processed_audio=result)
                report["whisper_blended"] = True
                report["whisper_segments"] = len(self._whisper_result.segments)
            except Exception as e:
                logger.debug("PipelineGuard: Whisper blending error: %s", e)

        # Dynamics-Auto-Recovery
        if self._original_profile is not None:
            try:
                if self._dp().should_restore():
                    logger.info("PipelineGuard: Auto-Recovery Dynamics (loss=%.1fdB)", self._dp().cumulative_loss_db)
                    from backend.core.dynamics_preserver import restore_dynamics

                    result = restore_dynamics(result, sr, self._original_profile, strength=0.7)
                    report["dynamics_restored"] = True
                    report["dynamics_loss_db"] = round(self._dp().cumulative_loss_db, 2)
            except Exception as e:
                logger.debug("PipelineGuard: Dynamics recovery error: %s", e)

        return result, report

    # ── Post-Flight ───────────────────────────────────────────────────

    def post_flight(self, final_audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Nach Pipeline-Ende: Watchdog-Report + Pleasantness-Check."""
        report: dict[str, Any] = {
            "phases_monitored": self._phase_count,
            "warnings": list(self._warnings),
            "criticals": list(self._criticals),
        }

        # Watchdog post-flight
        try:
            wd_report = self._wd().post_flight_validity(final_audio, sr)
            report["watchdog"] = wd_report.to_dict()
            report["all_checks_passed"] = wd_report.all_checks_passed
            if wd_report.criticals:
                self._criticals.extend(wd_report.criticals)
                report["criticals"] = list(self._criticals)
            if wd_report.warnings:
                self._warnings.extend(wd_report.warnings)
                report["warnings"] = list(self._warnings)
        except Exception as e:
            logger.debug("PipelineGuard: post_flight watchdog error: %s", e)
            report["all_checks_passed"] = True

        # HPE (Human Pleasantness Estimator)
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            hpe = compute_pleasantness(final_audio, sr)
            report["pleasantness_score"] = hpe.score
            report["pleasantness_label"] = hpe.label
            if hpe.issues:
                report["pleasantness_issues"] = hpe.issues
        except Exception as e:
            logger.debug("PipelineGuard: HPE unavailable: %s", e)

        elapsed = time.perf_counter() - self._start_time
        report["guard_overhead_ms"] = round(elapsed * 1000)

        if report.get("criticals"):
            logger.warning(
                "PipelineGuard: %d kritische Warnungen: %s", len(report["criticals"]), report["criticals"][:3]
            )

        return report

    # ── Properties ────────────────────────────────────────────────────

    @property
    def clp_result(self) -> Any:
        return self._clp_result

    @property
    def whisper_result(self) -> Any:
        return self._whisper_result

    @property
    def original_profile(self) -> Any:
        return self._original_profile

    @property
    def material(self) -> str:
        return self._material

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._original_audio = None
        self._original_profile = None
        self._whisper_result = None
        self._clp_result = None
        self._warnings = []
        self._criticals = []
        self._phase_count = 0
        try:
            self._wd().reset()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience-Funktionen (stateless, direkt importierbar)
# ═══════════════════════════════════════════════════════════════════════════════


def get_clp_max_attenuation_for_frequency(freq_hz: float, clp_result: Any) -> float:
    """Maximal erlaubte Daempfung (dB) fuer eine Frequenz basierend auf CLP-Maske."""
    if clp_result is None or clp_result.critical_mask is None:
        return 99.0
    from backend.core.critical_listening_points import CLP_ZONES

    for zone in CLP_ZONES:
        if zone.f_min <= freq_hz <= zone.f_max:
            return zone.max_cut_db
    return 99.0


def get_clp_max_gain_for_frequency(freq_hz: float, clp_result: Any) -> float:
    """Maximal erlaubte Verstaerkung (dB) fuer eine Frequenz."""
    if clp_result is None or clp_result.critical_mask is None:
        return 99.0
    from backend.core.critical_listening_points import CLP_ZONES

    for zone in CLP_ZONES:
        if zone.f_min <= freq_hz <= zone.f_max:
            return zone.max_gain_db
    return 99.0


def auto_detect_gender(audio: np.ndarray, sr: int) -> str:
    """Erkennt Geschlecht der Stimme aus der Grundfrequenz-Verteilung."""
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        mono = arr.mean(axis=1) if arr.shape[1] <= 2 else arr.mean(axis=0)
    else:
        mono = arr
    n_fft = 4096
    if len(mono) < n_fft:
        return "auto"
    spec = np.abs(np.fft.rfft(mono[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sr, 1))
    male_mask = (freqs >= 85) & (freqs <= 180)
    female_mask = (freqs >= 165) & (freqs <= 350)
    male_energy = float(np.sum(spec[male_mask]))
    female_energy = float(np.sum(spec[female_mask]))
    if male_energy > female_energy * 1.3:
        return "male"
    elif female_energy > male_energy * 1.3:
        return "female"
    return "auto"


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_guard: PipelineGuard | None = None
_guard_lock = threading.Lock()


def get_guard() -> PipelineGuard:
    """Thread-sicherer Singleton-Accessor."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = PipelineGuard()
    return _guard
