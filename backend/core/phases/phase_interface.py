"""
Phase Interface — Basisklassen für alle Aurik-Verarbeitungsphasen (§7.1).

Definiert:
  - PhaseCategory  (Enum): Kategorisierung der Phasen
  - PhaseMetadata  (dataclass): Beschreibende Metadaten einer Phase
  - PhaseResult    (dataclass): Ausgabe einer Phase
    - PhaseInterface (ABC): Abstrakte Basisklasse für alle 64 Phasen
  - create_phase_result(): Convenience-Factory

Aurik 9.10.46 — Kanonische Implementierung (core/phases/phase_interface.py)
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseCategory — Kategorisiert jede Phase nach ihrer Funktion (§7.1)
# ---------------------------------------------------------------------------
class PhaseCategory(Enum):
    """Funktionale Kategorie einer Restaurierungsphase."""

    DEFECT_REMOVAL = auto()  # Klicks, Rauschen, Brumm, Crackle …
    FREQUENCY = auto()  # EQ, Bandbreiten-Erweiterung, Rumble …
    RESTORATION = auto()  # Dropout, Inpainting, Spektralreparatur …
    DYNAMICS = auto()  # Kompression, Expansion, Limiting …
    ENHANCEMENT = auto()  # Exciter, Gesang, Instrumente, Air …
    STEREO = auto()  # Stereo-Balance, Mid/Side, Breite …
    METADATA = auto()  # Normalisierung, Format-Optimierung …


# ---------------------------------------------------------------------------
# PhaseMetadata — Beschreibende Informationen einer Phase
# ---------------------------------------------------------------------------
@dataclass
class PhaseMetadata:
    """Metadaten zu einer Verarbeitungsphase (unveränderlich nach Erstellung)."""

    phase_id: str  # z.B. "phase_01_click_removal"
    name: str  # Anzeigename
    category: PhaseCategory  # Funktionale Kategorie
    priority: int  # 1 (niedrig) – 10 (hoch)
    version: str = "1.0.0"
    dependencies: list[str] = field(default_factory=list)
    estimated_time_factor: float = 0.05  # Anteil Verarbeitungszeit (0–1)
    memory_requirement_mb: int = 64
    is_cpu_intensive: bool = True
    is_io_intensive: bool = False
    quality_impact: float = 0.85  # Erwarteter Qualitätsbeitrag (0–1)
    description: str = ""
    defect_types: list[str] = field(default_factory=list)
    musical_goals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert phase metadata to a plain dictionary."""
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "category": self.category.name,
            "priority": self.priority,
            "version": self.version,
            "dependencies": self.dependencies,
            "estimated_time_factor": self.estimated_time_factor,
            "memory_requirement_mb": self.memory_requirement_mb,
            "is_cpu_intensive": self.is_cpu_intensive,
            "is_io_intensive": self.is_io_intensive,
            "quality_impact": self.quality_impact,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# PhaseResult — Ausgabe einer Verarbeitungsphase
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    """Ergebnis einer Phase-Verarbeitung — immer NaN/Inf-frei und geclippt.

    §2.59: time_range ermöglicht chirurgische Verarbeitung.
    Wenn gesetzt, wurde die Phase NUR auf diesen Zeitbereich angewendet.
    None = Phase hat gesamtes Audio verarbeitet (global).
    """

    audio: np.ndarray  # Verarbeitetes Audio (float32, [-1,1])
    modifications: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metrics ist ein echtes Feld als Alias fuer metadata-Inhalte.
    # Wird es beim Konstruktor-Aufruf uebergeben, landet der Inhalt
    # in metadata (via __post_init__).
    time_range: tuple[float, float] | None = None  # §2.59: (start_s, end_s) oder None=global
    metrics: dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    ml_used: bool = False
    quality_estimate: float = 1.0  # 0–1
    success: bool = True  # True = Phase erfolgreich abgeschlossen

    def __post_init__(self) -> None:
        # Sicherheits-Invarianten: NaN/Inf bereinigen, clipping (§3.1)
        if not isinstance(self.audio, np.ndarray):
            self.audio = np.asarray(self.audio, dtype=np.float32)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.clip(self.audio, -1.0, 1.0)
        if self.audio.dtype != np.float32:
            self.audio = self.audio.astype(np.float32)
        # metrics und metadata synchronisieren: metrics erhaelt Vorrang
        # wenn explizit gesetzt, sonst wird metrics mit metadata befüllt.
        if self.metrics and not self.metadata:
            self.metadata = self.metrics
        elif self.metadata and not self.metrics:
            self.metrics = self.metadata

    def as_dict(self) -> dict[str, Any]:
        """Serialisiert the phase result payload to a plain dictionary."""
        return {
            "modifications": self.modifications,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "execution_time_seconds": self.execution_time_seconds,
            "ml_used": self.ml_used,
            "quality_estimate": self.quality_estimate,
        }


# ---------------------------------------------------------------------------
# create_phase_result — Convenience-Factory (NaN/Inf-sicher)
# ---------------------------------------------------------------------------
def create_phase_result(
    audio: np.ndarray,
    modifications: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_time_seconds: float = 0.0,
    ml_used: bool = False,
    quality_estimate: float = 1.0,
) -> PhaseResult:
    """Erzeugt ein NaN/Inf-bereinigtes PhaseResult.

    Args:
        audio:                  Verarbeitetes Audio-Signal (float32)
        modifications:          Dict mit Phase-spezifischen Änderungen
        warnings:               Liste von Warnungen
        metadata:               Zusätzliche Metadaten
        execution_time_seconds: Verarbeitungszeit in Sekunden
        ml_used:                Ob ML-Modell verwendet wurde
        quality_estimate:       Qualitätsschätzung 0–1

    Returns:
        PhaseResult mit bereinigtem Audio
    """
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    audio = np.clip(audio, -1.0, 1.0)

    return PhaseResult(
        audio=audio,
        modifications=modifications or {},
        warnings=warnings or [],
        metadata=metadata or {},
        execution_time_seconds=execution_time_seconds,
        ml_used=ml_used,
        quality_estimate=float(np.clip(quality_estimate, 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# PhaseInterface — Abstrakte Basisklasse für alle 64 Phasen (§7.1)
# ---------------------------------------------------------------------------
class PhaseInterface(abc.ABC):
    """Abstrakte Basisklasse für alle Aurik-Verarbeitungsphasen.

    Jede Phase implementiert:
        get_metadata() -> PhaseMetadata
        process(audio, sample_rate, material_type, **kwargs) -> PhaseResult

    Invarianten (§3.1):
        - Ausgang immer float32 im Bereich [-1, 1]
        - Kein NaN/Inf in Ausgang
        - sample_rate == 48000 wird vorausgesetzt
        - Kein direktes Netzwerk-I/O
    """

    def __init__(self, sample_rate: int = 48000, **_kwargs) -> None:
        """Basisinitialisierung für alle Phasen.

        Args:
            sample_rate: Sample-Rate (Standard 48000 Hz). Wird von Subklassen
                         via super().__init__(sample_rate) weitergegeben.
            **_kwargs:   Zusätzliche Konfigurations-Parameter (werden ignoriert,
                         aber akzeptiert, damit Subklassen **kwargs weiterreichen).
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._name_override: str | None = None
        # sample_rate für Subklassen verfügbar machen (ohne Pflicht es zu nutzen)
        self._sample_rate: int = sample_rate

    @property
    def sample_rate(self) -> int:
        """Sample-Rate dieser Phase (Standard 48000 Hz).

        Property für Rückwärtskompatibilität: Phasen die ``self.sample_rate``
        nutzen, funktionieren ohne Änderung. Intern gespeichert als ``_sample_rate``.
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        """Setzt sample_rate (ermöglicht phase_02-Muster: self.sample_rate = sr)."""
        self._sample_rate = value

    @property
    def metadata(self) -> PhaseMetadata:
        """Gibt Phasen-Metadaten als Attribut zurück (delegiert an get_metadata())."""
        return self.get_metadata()

    @abc.abstractmethod
    def get_metadata(self) -> PhaseMetadata:
        """Gibt beschreibende Metadaten dieser Phase zurück."""

    @abc.abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """Verarbeitet Audio und gibt PhaseResult zurück.

        Args:
            audio:        float32 np.ndarray, mono [N] oder stereo [2, N] / [N, 2]
            sample_rate:  Sample-Rate in Hz (intern immer 48000)
            material_type: Träger-Material z.B. "tape", "vinyl", "unknown"
            **kwargs:     Phase-spezifische Parameter

        Returns:
            PhaseResult mit bereinigtem Audio, NaN/Inf-frei, geclippt auf [-1, 1]
        """

    # ------------------------------------------------------------------
    # Konkrete Hilfsmethoden (von allen Phasen geerbt)
    # ------------------------------------------------------------------

    @staticmethod
    def surgical_dispatch(
        phase: PhaseInterface,
        audio: np.ndarray,
        sample_rate: int,
        material_type: str,
        time_ranges: list[tuple[float, float]],
        context_ms: float = 20.0,
        crossfade_ms: float = 5.0,
        **kwargs,
    ) -> np.ndarray:
        """§2.59.14: Führt eine Phase chirurgisch aus — nur auf Zeitfenstern.

        Extrahiert jedes Zeitfenster mit Kontext, ruft phase.process()
        auf das Fenster auf, und blended das Ergebnis via Cosine-Crossfade
        nahtlos zurück ins Gesamtsignal.

        Args:
            phase: Die auszuführende Phase (muss PhaseInterface sein)
            audio: Vollständiges Audio (channels, samples) oder (samples,)
            sample_rate: Sample-Rate in Hz
            material_type: Material-Typ für die Phase
            time_ranges: Liste von (start_s, end_s) Zeitfenstern
            context_ms: Kontext vor/nach jedem Fenster
            crossfade_ms: Dauer des Crossfades an den Rändern
            **kwargs: Werden an phase.process() weitergereicht

        Returns:
            Audio mit chirurgisch reparierten Zonen (gleiche Shape wie Input)
        """
        import numpy as np

        was_mono = audio.ndim == 1
        if was_mono:
            audio = audio.reshape(1, -1)
        result = audio.copy()
        total_samples = audio.shape[1]
        ctx_samples = int(context_ms * sample_rate / 1000)
        fade_samples = int(crossfade_ms * sample_rate / 1000)
        repaired = 0
        skipped = 0

        for start_s, end_s in sorted(time_ranges, key=lambda x: x[0]):
            s0 = max(0, int(start_s * sample_rate) - ctx_samples)
            s1 = min(total_samples, int(end_s * sample_rate) + ctx_samples)
            if s1 - s0 < 32:  # Minimum für DSP
                skipped += 1
                continue

            segment = audio[:, s0:s1].copy()
            original = segment.copy()

            try:
                proc_result = phase.process(segment, sample_rate, material_type, **kwargs)
                if isinstance(proc_result, np.ndarray):
                    segment = proc_result
                elif hasattr(proc_result, "audio"):
                    segment = proc_result.audio
            except Exception:
                skipped += 1
                continue

            # Safety-Clamp: ≤2× Original-Amplitude
            import numpy as _np

            abs_orig = _np.maximum(_np.abs(original), 1e-10)
            limit = abs_orig * 2.0
            _np.clip(segment, -limit, limit, out=segment)

            # Cosine-Crossfade an den Rändern
            if segment.shape[1] >= fade_samples * 2:
                ramp_in = 0.5 * (1 - _np.cos(_np.pi * _np.arange(fade_samples) / fade_samples))
                ramp_out = ramp_in[::-1]
                for ch in range(segment.shape[0]):
                    segment[ch, :fade_samples] = (
                        original[ch, :fade_samples] * (1 - ramp_in) + segment[ch, :fade_samples] * ramp_in
                    )
                    segment[ch, -fade_samples:] = (
                        original[ch, -fade_samples:] * (1 - ramp_out) + segment[ch, -fade_samples:] * ramp_out
                    )

            result[:, s0:s1] = segment
            repaired += 1

        if was_mono:
            result = result[0]
        return result.astype(np.float32)

    def _safe_process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """Wrapper mit Timing, Exception-Handling, NaN-Guard, ComfortGuard und VocalQualityGate.

        §Rolls-Royce-Phantom: Jede Phase wird automatisch auf Hörkomfort und
        Gesangsqualität geprüft. Kein manuelles Eingreifen nötig.
        """
        assert sample_rate == 48000, f"Interne SR muss 48000 Hz sein, erhalten: {sample_rate}"
        t0 = time.monotonic()
        try:
            result = self.process(audio, sample_rate, material_type, **kwargs)
        except Exception as exc:
            self._logger.warning(
                "Phase %s fehlgeschlagen (%s) — Pass-Through",
                self.get_metadata().phase_id,
                exc,
            )
            result = create_phase_result(
                audio=audio,
                warnings=[f"Phase fehlgeschlagen: {exc}"],
                quality_estimate=0.95,
            )

        # ── ComfortGuard: Automatische Hörmüdungs-Prävention ──────────
        try:
            from backend.core.comfort_guard import apply_comfort_guard

            result.audio = apply_comfort_guard(result.audio, sample_rate)
        except Exception as _cg_exc:
            self._logger.debug("ComfortGuard skipped: %s", _cg_exc)

        # ── VocalQualityGate: Gesangsqualität prüfen (nur bei Vokal-Phasen) ─
        phase_id = self.get_metadata().phase_id
        if any(kw in phase_id for kw in ("42", "65", "vocal", "voice", "deess")):
            try:
                from backend.core.vocal_quality_gate import get_vocal_quality_gate

                gate = get_vocal_quality_gate()
                decision = gate.evaluate(
                    pre_audio=audio,
                    post_audio=result.audio,
                    sr=sample_rate,
                    phase_name=phase_id,
                )
                if decision.rollback_needed:
                    result.warnings.append(f"VocalQualityGate: Rollback empfohlen (Δ={decision.naturalness_delta:.1f})")
                    result.warnings.extend(decision.warnings)
                    # Leichte Qualitätsabwertung bei Rollback
                    result.quality_estimate = max(0.5, result.quality_estimate - 0.1)
                if decision.recommendations:
                    result.metadata["vocal_recommendations"] = decision.recommendations
            except Exception as _vqg_exc:
                self._logger.debug("VocalQualityGate skipped: %s", _vqg_exc)

        result.execution_time_seconds = time.monotonic() - t0
        return result

    @property
    def phase_id(self) -> str:
        """Kurz-ID dieser Phase."""
        return self.get_metadata().phase_id

    @property
    def name(self) -> str:
        """Anzeigename dieser Phase."""
        if self._name_override is not None:
            return self._name_override
        return self.get_metadata().name

    @name.setter
    def name(self, value: str) -> None:
        """Erlaubt Subklassen self.name = '...' im __init__ zu setzen."""
        self._name_override = value

    def validate_input(self, audio: np.ndarray) -> tuple[bool, str | None]:
        """Validiert Eingangs-Audio auf Korrektheits-Invarianten.

        Returns:
            (True, None) wenn valide, (False, Fehlermeldung) sonst.
        """
        if audio.size == 0:
            return False, "Empty audio input"
        if not np.isfinite(audio).all():
            return False, "Audio contains NaN or Inf values"
        if audio.ndim > 2:
            return False, "Audio must be mono or stereo"
        return True, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.phase_id!r})"
