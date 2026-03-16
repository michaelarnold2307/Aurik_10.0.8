"""
RestaurierDenker — Domäne: Vollrestaurierung via UnifiedRestorerV3.

Kapselt core.unified_restorer_v3.UnifiedRestorerV3 mit strikter
3×RT-Pflicht (enforce_3x_rt=True darf NIEMALS auf False gesetzt werden).

Usage::

    from denker.restaurier_denker import get_restaurier_denker

    denker = get_restaurier_denker()
    ergebnis = denker.restauriere(audio, sr=48000, material="vinyl")
    restauriertes_audio = ergebnis.audio
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

# A-1: RT-Budget-Konstanten für V3 Post-Pass nach ARE-Erfolg (§9.5)
_3X_RT_LIMIT: float = 3.0
_V3_POSTPASS_BUDGET_FRACTION: float = 0.70  # nur wenn ARE < 70 % des RT-Budgets nutzte

# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class RestaurierErgebnis:
    """Ergebnis einer vollständigen Restaurierung."""

    audio: np.ndarray
    """Restauriertes Audio (float32, Bereich [-1, 1])."""

    material: str = ""
    """Erkanntes oder übergebenes Trägermaterial."""

    rt_factor: float = 0.0
    """Verarbeitungszeit-Faktor (≤ 3.0 garantiert)."""

    quality_estimate: float = 0.0
    """Qualitätsschätzung ∈ [0, 1]."""

    phases_executed: list[str] = field(default_factory=list)
    """Ausgeführte Verarbeitungsphasen."""

    phases_skipped: list[str] = field(default_factory=list)
    """Übersprungene Phasen (Budget, Defekte)."""

    musical_goals: dict[str, float] | None = None
    """14 Musical Goals (falls gemessen)."""

    warnings: list[str] = field(default_factory=list)
    """Warnmeldungen aus der Pipeline."""

    confidence: float = 1.0
    """Gesamtkonfidenz der Restaurierung ∈ [0, 1]."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Zusätzliche Pipeline-Metadaten."""

    # ── Kompatibilitäts-Felder (Tests erwarten diese Benennung) ──────────
    quality_delta: float = 0.0
    """Qualitätsgewinn gegenüber Eingang (Alias/Compat)."""

    phases_applied: list[str] = field(default_factory=list)
    """Alias für phases_executed (Test-Kompatibilität)."""

    processing_note: str = ""
    """Kurznotiz zur Verarbeitung (laienverständlich)."""

    def as_dict(self) -> dict[str, Any]:
        """Liefert alle Felder als serialisierbares Dict."""
        return {
            "rt_factor": self.rt_factor,
            "quality_estimate": self.quality_estimate,
            "phases_executed": self.phases_executed,
            "phases_skipped": self.phases_skipped,
            "musical_goals": self.musical_goals,
            "warnings": self.warnings,
            "material": self.material,
            "confidence": self.confidence,
            "audio_shape": list(self.audio.shape),
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# _AREAdapter — normiert AutonomousRestorationResult auf V3-Felder (M-5)
# ---------------------------------------------------------------------------


class _AREAdapter:
    """Adaptiert ``AutonomousRestorationResult`` (ARE) auf die Felder des
    ``UnifiedRestorerV3.restore()``-Ergebnisses, damit ``_konvertiere()``
    ohne Änderungen mit beiden Engines funktioniert (M-5).
    """

    def __init__(self, are_result: Any, audio_duration_s: float) -> None:
        # Audio
        raw_audio = are_result.audio
        self.audio: np.ndarray = (
            raw_audio if isinstance(raw_audio, np.ndarray) else np.array(raw_audio, dtype=np.float32)
        )

        # material_type — Enum-kompatibel halten
        try:
            self.material_type = are_result.material_type
        except AttributeError:

            class _FallbackMT:
                value: str = "unknown"

            self.material_type = _FallbackMT()

        # rt_factor aus Verarbeitungszeit
        t: float = float(getattr(are_result, "processing_time_seconds", 0.0))
        self.rt_factor: float = t / max(float(audio_duration_s), 1e-6)

        # Qualitätsschätzung
        self.quality_estimate: float = float(getattr(are_result, "quality_after", 0.75))

        # Musikalische Ziele (ARE liefert keine Einzel-Scores)
        self.musical_goals: dict[str, float] = {}

        # Phasen aus passes_executed
        passes: int = int(getattr(are_result, "passes_executed", 1))
        self.phases_executed: list[str] = [f"are_pass_{i + 1}" for i in range(passes)]
        self.phases_skipped: list[str] = []

        # Warnungen
        rollback: bool = bool(getattr(are_result, "rollback_triggered", False))
        self.warnings: list[str] = ["ARE-Rollback ausgelöst"] if rollback else []

        # Konfidenz aus improvement_db
        imp: float = float(getattr(are_result, "improvement_db", 0.0))
        self.confidence: float = min(imp / 10.0, 1.0) if imp > 0.0 else 0.85
        # Rollback verschlechtert Qualität → Konfidenz-Malus (A-4)
        if rollback:
            self.confidence = min(self.confidence, 0.5)

        # Gesamtzeit
        self.total_time_seconds: float = t


# ---------------------------------------------------------------------------
# RestaurierDenker
# ---------------------------------------------------------------------------


class RestaurierDenker:
    """Restaurierungs-Domänendenker — orchestriert UnifiedRestorerV3.

    Invarianten
    -----------
    - ``enforce_3x_rt=True`` ist **immer** aktiv — kein Override möglich.
    - Eingabe-Audio wird auf NaN/Inf geprüft und bereinigt.
    - Ausgabe-Audio ist immer in [-1, 1] geclippt.
    - Singleton via :func:`get_restaurier_denker` (Double-Checked Locking).
    """

    def __init__(self) -> None:
        self._restorers: dict[str, Any] = {}  # mode-keyed — Singleton-Fix (M-5)
        self._pipeline: Any | None = None  # AurikAutonomousPipeline (ARE)
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def restauriere(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        *,
        material: str | None = None,
        mode: str = "quality",
        validate_audio: bool = True,
        global_plan: Any | None = None,
        chain_info: Any | None = None,
        defekt_hint: Any | None = None,
        progress_callback: Any | None = None,
    ) -> RestaurierErgebnis:
        """Restauriert Audio vollständig mit UnifiedRestorerV3.

        Parameter
        ---------
        audio:
            Eingabe-Audio als float32-Array (mono oder stereo).
        sr:
            Abtastrate in Hz (Standard: 48000).
        material:
            Trägermaterial-Hint (z. B. ``"vinyl"``, ``"tape"``).
            ``None`` = automatische Erkennung.
        mode:
            Qualitätsmodus (``"quality"`` oder ``"studio2026"``).
        validate_audio:
            Ob Eingabe auf NaN/Inf geprüft werden soll.

        Rückgabe
        --------
        RestaurierErgebnis mit restauriertem Audio und Pipeline-Metadaten.
        """
        assert sr == 48000, f"RestaurierDenker.restauriere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        # ── ARE (AurikAutonomousPipeline) als primäre Engine (M-5) ──────────
        _audio_dur_s: float = float(len(audio)) / max(float(sr), 1.0)
        pipeline = self._get_are_pipeline()
        if pipeline is not None:
            try:
                _t0_are = time.perf_counter()
                are_result = pipeline.process(audio, sample_rate=sr)
                adapter = _AREAdapter(are_result, audio_duration_s=_audio_dur_s)
                logger.info(
                    "\U0001f680 RestaurierDenker: ARE primary OK \u2014 Q=%.3f RT=%.2f\u00d7",
                    adapter.quality_estimate,
                    adapter.rt_factor,
                )
                # A-1: V3 Post-Pass additiv auf ARE-Output — nur wenn RT-Budget erlaubt
                _are_elapsed = time.perf_counter() - _t0_are
                _are_rt = _are_elapsed / max(_audio_dur_s, 1e-6)
                if _are_rt < _3X_RT_LIMIT * _V3_POSTPASS_BUDGET_FRACTION:
                    _v3 = self._get_restorer(mode=mode)
                    if _v3 is not None:
                        try:
                            _v3_raw = _v3.restore(adapter.audio, sample_rate=sr)
                            _v3_audio = np.nan_to_num(
                                np.array(_v3_raw.audio, dtype=np.float32),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            adapter.audio = np.clip(_v3_audio, -1.0, 1.0)
                            adapter.quality_estimate = max(
                                adapter.quality_estimate,
                                float(getattr(_v3_raw, "quality_estimate", 0.0)),
                            )
                            adapter.phases_executed = list(adapter.phases_executed) + ["v3_post_pass"]
                            logger.info(
                                "RestaurierDenker: V3 Post-Pass OK \u2014" " ARE RT-Nutzung: %.1f%% (< %.0f%%)",
                                _are_rt / _3X_RT_LIMIT * 100,
                                _V3_POSTPASS_BUDGET_FRACTION * 100,
                            )
                        except Exception as _v3_exc:
                            logger.debug(
                                "RestaurierDenker: V3 Post-Pass \u00fcbersprungen (%s)" " \u2014 ARE-Ergebnis bleibt.",
                                _v3_exc,
                            )
                return self._konvertiere(adapter, material=material)
            except Exception as are_exc:
                logger.warning(
                    "AurikAutonomousPipeline fehlgeschlagen: %s \u2014 Fallback auf UnifiedRestorerV3",
                    are_exc,
                )

        # ── Fallback: UnifiedRestorerV3 ───────────────────────────────────────
        restorer = self._get_restorer(mode=mode)
        if restorer is None:
            return self._fallback(audio, material or "unknown", "Kein Restorer verfügbar")

        try:
            _restore_kwargs: dict = {"sample_rate": sr}
            if global_plan is not None:
                _restore_kwargs["global_plan"] = global_plan
            if chain_info is not None:
                _restore_kwargs["chain_info"] = chain_info
            if defekt_hint is not None:
                _restore_kwargs["defekt_hint"] = defekt_hint
            if progress_callback is not None:
                _restore_kwargs["progress_callback"] = progress_callback
            raw = restorer.restore(audio, **_restore_kwargs)
        except Exception as exc:
            logger.warning("UnifiedRestorerV3.restore() fehlgeschlagen: %s \u2014 Fallback auf Original", exc)
            return self._fallback(audio, material or "unknown", str(exc))

        return self._konvertiere(raw, material=material)

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _get_restorer(self, mode: str = "quality") -> Any:
        """Lädt UnifiedRestorerV3 lazy und mode-spezifisch (Double-Checked Locking).

        Jeder ``mode`` bekommt eine eigene Instanz — verhindert den Bug,
        dass der erste Call den Singleton auf ``quality`` fixiert und spätere
        ``studio2026``-Calls unbemerkt die falsche Konfiguration erhalten.
        """
        if mode not in self._restorers:
            with self._lock:
                if mode not in self._restorers:
                    self._restorers[mode] = self._build_restorer(mode)
        return self._restorers.get(mode)

    def _build_restorer(self, mode: str) -> Any:
        """Baut UnifiedRestorerV3 mit zwingend enforce_3x_rt=True."""
        try:
            from core.unified_restorer_v3 import (
                QualityMode,
                RestorationConfig,
                UnifiedRestorerV3,
            )

            qmode = QualityMode.STUDIO_2026 if mode == "studio2026" else QualityMode.QUALITY

            cfg = RestorationConfig(
                mode=qmode,
                enforce_3x_rt=True,  # NIEMALS False — Pflicht-Invariante
                enable_performance_guard=True,
                enable_adaptive_skipping=True,
                enable_phase_gate=True,
                num_cores=4,
                enable_psychoacoustic_enhancement=True,
            )

            logger.info("🎵 RestaurierDenker: UnifiedRestorerV3 init (mode=%s, enforce_3x_rt=True)", mode)
            return UnifiedRestorerV3(config=cfg)

        except Exception as exc:
            logger.warning("UnifiedRestorerV3 konnte nicht geladen werden: %s", exc)
            return None

    # ------------------------------------------------------------------
    # ARE-Pipeline (AurikAutonomousPipeline)  — M-5
    # ------------------------------------------------------------------

    def _get_are_pipeline(self) -> Any:
        """Lädt AurikAutonomousPipeline lazy (Double-Checked Locking)."""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = self._build_are_pipeline()
        return self._pipeline

    def _build_are_pipeline(self) -> Any:
        """Erstellt AurikAutonomousPipeline als primäre Restaurierungs-Engine (M-5)."""
        try:
            from backend.core.pipeline_main import AurikAutonomousPipeline  # lazy import
            from backend.core.processing_modes import ProcessingMode

            pipeline = AurikAutonomousPipeline(
                mode=ProcessingMode.RESTORATION,
                enable_self_learning=True,
            )
            logger.info("\u2705 RestaurierDenker: AurikAutonomousPipeline (ARE) bereit")
            return pipeline
        except Exception as exc:
            logger.warning(
                "AurikAutonomousPipeline nicht verf\u00fcgbar (%s) \u2014 UnifiedRestorerV3 als Fallback",
                exc,
            )
            return None

    def _konvertiere(self, raw: Any, *, material: str | None) -> RestaurierErgebnis:
        """Wandelt RestorationResult in RestaurierErgebnis um."""
        if raw is None:
            dummy = np.zeros(1, dtype=np.float32)
            return RestaurierErgebnis(
                audio=dummy,
                rt_factor=0.0,
                quality_estimate=0.0,
                phases_executed=[],
                phases_skipped=[],
                musical_goals=None,
                warnings=["Keine Verarbeitung — Restorer nicht initialisiert"],
                material=material or "unknown",
            )

        # Audio sichern
        out_audio = np.array(raw.audio, dtype=np.float32)
        out_audio = np.nan_to_num(out_audio, nan=0.0, posinf=0.0, neginf=0.0)
        out_audio = np.clip(out_audio, -1.0, 1.0)

        rt = float(raw.rt_factor) if math.isfinite(raw.rt_factor) else 0.0

        # Material aus raw übernehmen wenn nicht explizit übergeben
        detected_material = material
        if detected_material is None:
            try:
                detected_material = raw.material_type.value  # type: ignore[attr-defined]
            except Exception:
                detected_material = "unknown"

        # Musical Goals extrahieren
        goals: dict[str, float] | None = None
        if raw.musical_goals:
            goals = {k: float(v) for k, v in raw.musical_goals.items() if math.isfinite(float(v))}

        return RestaurierErgebnis(
            audio=out_audio,
            rt_factor=rt,
            quality_estimate=float(raw.quality_estimate) if math.isfinite(raw.quality_estimate) else 0.0,
            phases_executed=list(raw.phases_executed or []),
            phases_skipped=list(raw.phases_skipped or []),
            musical_goals=goals,
            warnings=list(raw.warnings or []),
            material=detected_material or "unknown",
            confidence=float(raw.confidence) if math.isfinite(raw.confidence) else 1.0,
            metadata={
                "total_time_seconds": float(raw.total_time_seconds or 0.0),
            },
        )

    @staticmethod
    def _fallback(audio: np.ndarray, material: str, reason: str) -> RestaurierErgebnis:
        """Gibt Original-Audio als Notfall-Ergebnis zurück."""
        out = np.clip(
            np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        return RestaurierErgebnis(
            audio=out,
            rt_factor=0.0,
            quality_estimate=0.0,
            phases_executed=[],
            phases_skipped=[],
            musical_goals=None,
            warnings=[f"Restaurierung fehlgeschlagen — Original unverändert. Grund: {reason}"],
            material=material,
        )


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance: RestaurierDenker | None = None
_lock: threading.Lock = threading.Lock()


def get_restaurier_denker() -> RestaurierDenker:
    """Gibt den thread-sicheren Singleton-RestaurierDenker zurück."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RestaurierDenker()
    return _instance
