"""
Phase Interface — Basisklassen für alle Aurik-Verarbeitungsphasen (§7.1).

Definiert:
  - PhaseCategory  (Enum): Kategorisierung der Phasen
  - PhaseMetadata  (dataclass): Beschreibende Metadaten einer Phase
  - PhaseResult    (dataclass): Ausgabe einer Phase
  - PhaseInterface (ABC): Abstrakte Basisklasse für alle 56 Phasen
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
        """Serialize phase metadata to a plain dictionary."""
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
    """Ergebnis einer Phase-Verarbeitung — immer NaN/Inf-frei und geclippt."""

    audio: np.ndarray  # Verarbeitetes Audio (float32, [-1,1])
    modifications: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # metrics ist ein echtes Feld als Alias fuer metadata-Inhalte.
    # Wird es beim Konstruktor-Aufruf uebergeben, landet der Inhalt
    # in metadata (via __post_init__).
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
        """Serialize the phase result payload to a plain dictionary."""
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
# PhaseInterface — Abstrakte Basisklasse für alle 56 Phasen (§7.1)
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

    def _safe_process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs: Any,
    ) -> PhaseResult:
        """Wrapper mit Timing, Exception-Handling und NaN-Guard."""
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
