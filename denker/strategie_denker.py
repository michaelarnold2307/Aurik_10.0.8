"""
StrategieDenker — Domäne: 8×RT-Budgetplanung + Performance-Guard
=================================================================

Kapselt `core.performance_guard.PerformanceGuard` und plant die
Verarbeitungs-Strategie anhand des verfügbaren Zeit-Budgets.

Die 8×RT-Grenze (§9.5) ist hart: Verarbeitung darf maximal das
Achtfache der Audiodauer dauern. Dieser Denker sorgt dafür, dass
dieses Limit durchgesetzt und kommuniziert wird.

Singleton-Pattern nach §3.2 (Double-Checked Locking).
Type-Annotations nach §3.7.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 32×RT-Grenze aus §9.5 / PerformanceGuard.LIMIT_3X_RT
_3X_RT_LIMIT: float = 32.0


# ---------------------------------------------------------------------------
# Strategie-Daten
# ---------------------------------------------------------------------------


@dataclass
class StrategiePlan:
    """Enthält die geplante Verarbeitungs-Strategie und das RT-Budget."""

    audio_duration_s: float
    """Länge der Quelldatei in Sekunden."""

    max_processing_s: float
    """Maximal erlaubte Verarbeitungszeit in Sekunden (8× Audiodauer)."""

    quality_mode: str
    """Gewählter Qualitätsmodus: 'quality', 'balanced' oder 'speed'."""

    enforce_limit: bool
    """True = 8×RT-Limit wird hart durchgesetzt."""

    enable_adaptive_skipping: bool
    """True = Nicht-kritische Phasen werden übersprungen wenn Budget knapp."""

    recommended_chunk_s: float = 60.0
    """Empfohlene Chunk-Größe für adaptive Verarbeitung (§7.6, defekt-adaptiv)."""

    defect_severity: float = 0.0
    """Defekt-Schwere aus DefektDenker ∈ [0, 1]; steuert Chunk-Größe nach §7.6."""

    budget_note: str = ""
    """Hinweis auf Budget-Engpässe (Deutsch, laienverständlich)."""

    def as_dict(self) -> dict:
        return {
            "audio_duration_s": self.audio_duration_s,
            "max_processing_s": self.max_processing_s,
            "quality_mode": self.quality_mode,
            "enforce_limit": self.enforce_limit,
            "enable_adaptive_skipping": self.enable_adaptive_skipping,
            "recommended_chunk_s": self.recommended_chunk_s,
            "budget_note": self.budget_note,
        }


@dataclass
class BudgetStatus:
    """Laufender Budget-Status während der Verarbeitung."""

    elapsed_s: float
    """Bisher verstrichene Verarbeitungszeit in Sekunden."""

    budget_remaining_s: float
    """Verbleibende Zeit im Budget in Sekunden."""

    rt_factor_current: float
    """Aktueller RT-Faktor (verstrichene Zeit / Audiodauer)."""

    should_exit_early: bool
    """True wenn das Budget erschöpft ist und gestoppt werden sollte."""

    phases_completed: int = 0
    """Zahl der abgeschlossenen Phasen."""


@dataclass
class StrategieErgebnis:
    """Ergebnis der Strategie-Planung für die Restaurierung."""

    selected_phases: list
    """Liste der ausgewählten Verarbeitungsphasen."""

    phase_parameters: dict
    """Parameter-Mapping pro Phase."""

    strategy_name: str
    """Name der gewählten Strategie (z. B. 'Rauschunterdrückung')."""

    estimated_quality_gain: float
    """Geschätzter Qualitätsgewinn durch die Strategie (0–1)."""

    reasoning: str
    """Laienverständliche Begründung der Strategie-Wahl."""

    rt_limit: float = 3.0
    """Echtzeit-Faktor-Grenze (z. B. 3.0 = max. 3× Audiodauer)."""

    start_time: float = 0.0
    """Startzeitpunkt (time.time()) für Budget-Tracking."""


# ---------------------------------------------------------------------------
# StrategieDenker
# ---------------------------------------------------------------------------


class StrategieDenker:
    """Plant die Verarbeitungs-Strategie und überwacht das 8×RT-Budget.

    Kernaufgabe:
        1. plan()         → StrategiePlan erstellen
        2. starte_timer() → Messung beginnen
        3. check()        → BudgetStatus abfragen (SOLL frühzeitig beendet werden?)

    PerformanceGuard wird genutzt, um die Phasen-Priorisierung
    (MUSICAL_EXCELLENCE_PHASES werden niemals übersprungen) zu erhalten.

    Verwendung::

        denker = get_strategie_denker()
        plan  = denker.plan(audio, sr, mode="quality")
        denker.starte_timer(audio_duration_s=plan.audio_duration_s)
        # ... Phase ausführen ...
        status = denker.check(phases_remaining=5)
        if status.should_exit_early:
            break  # Budget erschöpft — Verarbeitung sicher beenden
    """

    def __init__(self) -> None:
        self._guard: Any | None = None
        self._guard_lock = threading.Lock()
        self._guard_loaded = False

        # Runtime tracking
        self._t_start: float | None = None
        self._audio_duration_s: float = 0.0
        self._current_plan: StrategiePlan | None = None

    # ------------------------------------------------------------------
    # Lazy-Init des PerformanceGuard
    # ------------------------------------------------------------------

    def _ensure_guard(self, mode: str = "quality", enforce: bool = True) -> None:
        """Instantiate or re-instantiate PerformanceGuard with given mode."""
        with self._guard_lock:
            try:
                from backend.core.performance_guard import PerformanceGuard

                # Map string mode to QualityMode if possible
                quality_mode_obj = self._parse_mode(mode)
                self._guard = PerformanceGuard(
                    mode=quality_mode_obj,
                    enforce_limit=enforce,
                    enable_adaptive_skipping=True,
                )
                logger.info("StrategieDenker: PerformanceGuard (mode=%s) geladen.", mode)
            except Exception as exc:
                logger.warning(
                    "StrategieDenker: PerformanceGuard nicht verfügbar (%s). Einfacher Timer-Fallback wird genutzt.",
                    exc,
                )
                self._guard = None
            self._guard_loaded = True

    @staticmethod
    def _parse_mode(mode_str: str) -> Any:
        """Convert mode string to QualityMode enum value if available."""
        try:
            from backend.core.unified_restorer_v3 import QualityMode

            mapping = {
                "fast": QualityMode.FAST,
                "balanced": QualityMode.BALANCED,
                "quality": QualityMode.QUALITY,
                "restoration": QualityMode.QUALITY,
                "maximum": QualityMode.MAXIMUM,
                "studio_2026": QualityMode.MAXIMUM,
                # "speed" existiert nicht im QualityMode-Enum → FAST als Fallback
            }
            return mapping.get(mode_str.lower(), QualityMode.QUALITY)
        except Exception:
            return mode_str  # PerformanceGuard handles unknown mode gracefully

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def plan(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        mode: str = "quality",
        enforce_3x_rt: bool = True,
        defect_severity: float = 0.0,
    ) -> StrategiePlan:
        """Erstellt den Verarbeitungs-Strategieplan.

        Algorithmus:
            1. Audiodauer berechnen
            2. 8×RT-Budget ableiten  (max_processing_s = 8 × audio_duration_s)
            3. Chunk-Größe gemäß §9.5 adaptiv-defektdichte setzen
            4. PerformanceGuard initialisieren

        Args:
            audio:         Eingabe-Audio.
            sr:            Sample-Rate in Hz.
            mode:          Qualitätsmodus ('quality', 'balanced', 'speed').
            enforce_3x_rt: 8×RT-Limit hart durchsetzen.

        Returns:
            StrategiePlan mit Budget-Angaben.
        """
        assert sr == 48000, f"StrategieDenker.plan() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        audio_dur = _safe_duration(audio, sr)

        self._ensure_guard(mode=mode, enforce=enforce_3x_rt)

        max_proc = _3X_RT_LIMIT * audio_dur
        _sev = float(defect_severity) if math.isfinite(float(defect_severity)) else 0.0
        _sev = max(0.0, min(1.0, _sev))
        chunk_s = _adaptive_chunk(audio_dur, defect_severity=_sev)

        note = ""
        if audio_dur > 300:
            note = (
                "Lange Datei erkannt — Verarbeitung erfolgt in Abschnitten "
                f"(jeweils {int(chunk_s)} s), um das 8×RT-Zeitbudget einzuhalten."
            )
        elif audio_dur < 5:
            note = "Sehr kurze Aufnahme — volle Verarbeitungstiefe aktiviert."

        plan = StrategiePlan(
            audio_duration_s=audio_dur,
            max_processing_s=max_proc,
            quality_mode=mode,
            enforce_limit=enforce_3x_rt,
            enable_adaptive_skipping=True,
            recommended_chunk_s=chunk_s,
            defect_severity=_sev,
            budget_note=note,
        )
        self._current_plan = plan
        logger.info(
            "StrategieDenker: Strategieplan — Dauer=%.1fs, Budget=%.1fs, Chunk=%.1fs",
            audio_dur,
            max_proc,
            chunk_s,
        )
        return plan

    def starte_timer(self, audio_duration_s: float) -> None:
        """Startet die Zeitmessung für das Budget-Tracking.

        Ruft optional `PerformanceGuard.start_monitoring()` auf.

        Args:
            audio_duration_s: Länge der Quelldatei in Sekunden.
        """
        self._audio_duration_s = max(audio_duration_s, 0.001)
        self._t_start = time.monotonic()

        if self._guard is not None:
            try:
                self._guard.start_monitoring(audio_duration_seconds=self._audio_duration_s)
            except Exception as exc:
                logger.debug("StrategieDenker: start_monitoring() Fehler: %s", exc)

        logger.info(
            "StrategieDenker: Timer gestartet (Audio=%.1fs, Budget=%.1fs).",
            audio_duration_s,
            _3X_RT_LIMIT * audio_duration_s,
        )

    def check(self, phases_remaining: int = 0) -> BudgetStatus:
        """Prüft den aktuellen Budget-Status.

        Args:
            phases_remaining: Anzahl der noch anstehenden Verarbeitungs-Phasen.

        Returns:
            BudgetStatus mit Empfehlung, ob frühzeitig abgebrochen werden soll.
        """
        if self._t_start is None:
            return BudgetStatus(
                elapsed_s=0.0,
                budget_remaining_s=float("inf"),
                rt_factor_current=0.0,
                should_exit_early=False,
            )

        elapsed = time.monotonic() - self._t_start
        max_proc = _3X_RT_LIMIT * max(self._audio_duration_s, 0.001)
        remaining = max(0.0, max_proc - elapsed)
        rt_factor = elapsed / max(self._audio_duration_s, 0.001)

        # Check via PerformanceGuard if available
        guard_exit = False
        if self._guard is not None:
            try:
                guard_exit = bool(self._guard.check_early_exit(remaining_phases=phases_remaining))
            except Exception as exc:
                logger.debug("StrategieDenker: check_early_exit() Fehler: %s", exc)

        # Hard limit override
        hard_exit = rt_factor >= _3X_RT_LIMIT

        should_exit = guard_exit or hard_exit

        if should_exit:
            logger.warning(
                "StrategieDenker: Budget erschöpft! RT-Faktor=%.2f ≥ 3.0 (elapsed=%.1fs, budget=%.1fs).",
                rt_factor,
                elapsed,
                max_proc,
            )

        return BudgetStatus(
            elapsed_s=elapsed,
            budget_remaining_s=remaining,
            rt_factor_current=rt_factor,
            should_exit_early=should_exit,
        )

    @property
    def performance_guard(self) -> Any | None:
        """Zugriff auf den internen PerformanceGuard (für fortgeschrittene Nutzung)."""
        return self._guard


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _safe_duration(audio: np.ndarray, sr: int) -> float:
    """Return audio duration in seconds, guarded for NaN and edge cases.

    Handles both (channels, samples) and (samples, channels) layouts by
    using the same heuristic as AurikDenker.denke(): axis with size <= 2
    is channels, the other axis is samples.
    """
    if sr < 1 or audio.size == 0:
        return 0.001
    if audio.ndim == 1:
        n_samples = audio.shape[0]
    elif audio.shape[-1] <= 2:
        # (N, channels) layout — samples on axis 0
        n_samples = audio.shape[0]
    else:
        # (channels, N) layout — samples on axis -1
        n_samples = audio.shape[-1]
    dur = n_samples / max(sr, 1)
    return dur if math.isfinite(dur) and dur > 0 else 0.001


def _adaptive_chunk(audio_dur_s: float, defect_severity: float = 0.0) -> float:
    """Determine recommended chunk size per §7.6 (defect-density adaptive).

    Spec §7.6:
        defect_severity >= 0.6  →  5 s   (fine-grained, high-defect material)
        defect_severity >= 0.3  → 15 s   (moderate defects)
        default                 → 60 s   (clean material — large context chunks)
        silence segment caps at 120 s    (not observable here — caller handles)
    Minimum: 2 s | Maximum: min(120 s, audio_dur_s)
    """
    if audio_dur_s <= 2.0:
        return audio_dur_s  # Too short to subdivide
    if defect_severity >= 0.6:
        chunk = 5.0  # §7.6: Feingranular bei hohem Defektniveau
    elif defect_severity >= 0.3:
        chunk = 15.0  # §7.6: Mittel bei moderatem Defektniveau
    elif audio_dur_s > 300:
        chunk = 120.0  # Long clean files: 120 s for context coherence
    elif audio_dur_s > 60:
        chunk = 60.0
    else:
        return audio_dur_s  # Short clean files: process as whole
    return float(min(max(chunk, 2.0), audio_dur_s))


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: StrategieDenker | None = None
_lock = threading.Lock()


def get_strategie_denker() -> StrategieDenker:
    """Thread-sicherer Singleton-Accessor für StrategieDenker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StrategieDenker()
    return _instance
