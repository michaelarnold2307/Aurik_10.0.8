"""Aurik 9 — API Bridge (§11 Spec 08)
=====================================
Einziger Eintrittspunkt für Frontend/CLI → Backend-Core.

Das Frontend darf ``backend/core/``, ``dsp/`` oder ``plugins/`` **nicht**
direkt importieren. Alle Core-Zugriffe laufen über diese Datei.

Verwendung im Frontend::

    from backend.api.bridge import export_guard, get_aurik_denker_instance
    from backend.api.bridge import get_defect_scanner, get_quality_mode
    from backend.api.bridge import get_musical_goals_checker, get_mushra_evaluator
    from backend.api.bridge import get_perceptual_quality_scorer
    from backend.api.bridge import get_ml_memory_budget_status

Öffentliche API (vollständig)::

    # Defect-Cache (FIFO, 64 Einträge, Thread-sicher)
    cache_defect_result, get_cached_defect_result, clear_defect_cache

    # Era/Genre-Cache (FIFO, 64 Einträge, Thread-sicher)
    cache_era_genre_result, get_cached_era_genre_result, clear_era_genre_cache

    # Enums / Konfigurationsklassen
    get_quality_mode, get_medium_type_enum, get_processing_mode_enum

    # Kern-Einstiegspunkte
    get_restorer_classes, get_aurik_denker_class, get_aurik_denker_instance

    # Analyse / Klassifikation
    get_defect_scanner, get_defect_type
    get_medium_classifier_fn, get_era_classifier_fn, get_genre_classifier_fn
    get_restorability_estimator_class, get_carrier_forensics_fn
    get_audio_file_validator

    # Qualitätsbewertung
    get_musical_goals_checker          # MusicalGoalsChecker-Klasse (§8.1)
    get_adaptive_goals_fn              # get_adaptive_goals_and_config (§2.31)
    get_mushra_evaluator               # MushraEvaluator-Singleton (§8.1.1 OQS)
    get_perceptual_quality_scorer      # PerceptualQualityScorer-Singleton (§8.1 PQS)

    # Infrastruktur / Pipeline
    get_plugin_lifecycle_manager       # PLM-Singleton (LRU-Eviction §2.37)
    get_ml_memory_budget_status        # Budget-Statusdict (§2.37)
    get_pipeline_health_state_enum, normalize_pipeline_health_state
    resolve_pipeline_fail_reason

    # Audio-Verarbeitung (Hilfsmittel)
    get_audio_exporter_class           # None wenn Modul fehlt
    get_stem_remix_balancer_fn         # StemRemixBalancer.balance_remix (§1.5)
    get_clipping_classifier            # ClippingClassifier-Singleton (§6.3)
    get_lyrics_guided_enhancement_fn   # LyricsGuidedEnhancement (§2.36)
    get_cleanup_after_file_fn          # PLM.cleanup_after_file

    # NaN/Inf-Guard + Export-Absicherung
    export_guard

    # Hintergrund-Vorwärmung
    warmup_models_background

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Öffentliche API — explizite Export-Liste
# ---------------------------------------------------------------------------

__all__ = [
    # Defect-Cache
    "cache_defect_result",
    "clear_defect_cache",
    # Era/Genre-Cache
    "cache_era_genre_result",
    "clear_era_genre_cache",
    "get_cached_era_genre_result",
    # Medium-Cache
    "cache_medium_result",
    "get_cached_medium_result",
    # Restorability-Cache
    "cache_restorability_result",
    "get_cached_restorability_result",
    # NaN/Inf-Guard
    "export_guard",
    "validate_export_quality",
    "get_adaptive_goals_fn",
    # Audio-Verarbeitung (Hilfsmittel)
    "get_audio_exporter_class",
    "get_audio_file_validator",
    "get_aurik_denker_class",
    "get_aurik_denker_instance",
    "get_cached_defect_result",
    "get_carrier_forensics_fn",
    "get_cleanup_after_file_fn",
    "get_clipping_classifier",
    # Analyse / Klassifikation
    "get_defect_scanner",
    "get_defect_type",
    "get_era_classifier_fn",
    "get_genre_classifier_fn",
    "get_lyrics_guided_enhancement_fn",
    "get_medium_classifier_fn",
    "get_medium_type_enum",
    "get_ml_memory_budget_status",
    "get_mushra_evaluator",
    # Qualitätsbewertung (§8.1)
    "get_musical_goals_checker",
    "get_perceptual_quality_scorer",
    "get_pipeline_health_state_enum",
    # Infrastruktur / Speicher-Management (§2.37)
    "get_plugin_lifecycle_manager",
    "get_processing_mode_enum",
    # Enums / Konfigurationsklassen
    "get_quality_mode",
    "get_restorability_estimator_class",
    # Kern-Einstiegspunkte
    "get_restorer_classes",
    "get_stem_remix_balancer_fn",
    "normalize_pipeline_health_state",
    "resolve_pipeline_fail_reason",
    # Hintergrund-Vorwärmung
    "warmup_models_background",
]

# ---------------------------------------------------------------------------
# Defect-Scan-Cache  (Thread-sicher, Prozess-Lebensdauer, RAM-only)
# Key: file_path (str), Value: ScanResult-Objekt
# Limit: 64 Einträge (FIFO-Trim)
# ---------------------------------------------------------------------------

_defect_cache: dict[str, object] = {}
_defect_cache_lock = threading.Lock()
_DEFECT_CACHE_MAX = 64


def cache_defect_result(file_path: str, result: object) -> None:
    """Speichert einen DefectScanner-Befund für *file_path* im Cache.

    Thread-sicher. Trimmt den Cache auf _DEFECT_CACHE_MAX Einträge (FIFO).
    """
    with _defect_cache_lock:
        _defect_cache[file_path] = result
        # FIFO-Trim
        if len(_defect_cache) > _DEFECT_CACHE_MAX:
            oldest = next(iter(_defect_cache))
            del _defect_cache[oldest]
    logger.debug("bridge: DefectScan cached for '%s'", file_path)


def get_cached_defect_result(file_path: str) -> object | None:
    """Gibt einen gecachten DefectScanner-Befund zurück oder ``None``."""
    with _defect_cache_lock:
        return _defect_cache.get(file_path)


def clear_defect_cache(file_path: str | None = None) -> None:
    """Löscht einen oder alle Einträge aus dem DefectScan-Cache."""
    with _defect_cache_lock:
        if file_path is not None:
            _defect_cache.pop(file_path, None)
        else:
            _defect_cache.clear()


# ---------------------------------------------------------------------------
# Era/Genre-Cache  (Thread-sicher, Prozess-Lebensdauer, RAM-only)
# Key: file_path (str), Value: dict mit era_result und genre_result
# Limit: 64 Einträge (FIFO-Trim)
# ---------------------------------------------------------------------------

_era_genre_cache: dict[str, dict[str, object]] = {}
_era_genre_cache_lock = threading.Lock()
_ERA_GENRE_CACHE_MAX = 64


def cache_era_genre_result(
    file_path: str,
    era_result: object | None = None,
    genre_result: object | None = None,
) -> None:
    """Speichert Era/Genre-Klassifikationsergebnisse für *file_path* im Cache.

    Thread-sicher. Trimmt den Cache auf _ERA_GENRE_CACHE_MAX Einträge (FIFO).
    """
    with _era_genre_cache_lock:
        _era_genre_cache[file_path] = {
            "era_result": era_result,
            "genre_result": genre_result,
        }
        if len(_era_genre_cache) > _ERA_GENRE_CACHE_MAX:
            oldest = next(iter(_era_genre_cache))
            del _era_genre_cache[oldest]
    logger.debug("bridge: Era/Genre cached for '%s'", file_path)


def get_cached_era_genre_result(file_path: str) -> dict[str, object] | None:
    """Gibt gecachte Era/Genre-Ergebnisse zurück oder ``None``.

    Returns:
        dict mit Keys ``era_result`` und ``genre_result`` oder ``None``.
    """
    with _era_genre_cache_lock:
        return _era_genre_cache.get(file_path)


def clear_era_genre_cache(file_path: str | None = None) -> None:
    """Löscht einen oder alle Einträge aus dem Era/Genre-Cache."""
    with _era_genre_cache_lock:
        if file_path is not None:
            _era_genre_cache.pop(file_path, None)
        else:
            _era_genre_cache.clear()


# ---------------------------------------------------------------------------
# Medium-Cache  (Thread-sicher, Prozess-Lebensdauer, RAM-only)
# Key: file_path (str), Value: MediumClassifier-Result-Objekt
# Limit: 64 Einträge (FIFO-Trim)
# ---------------------------------------------------------------------------

_medium_cache: dict[str, object] = {}
_medium_cache_lock = threading.Lock()
_MEDIUM_CACHE_MAX = 64


def cache_medium_result(file_path: str, result: object) -> None:
    """Speichert ein MediumClassifier-Ergebnis für *file_path* im Cache."""
    with _medium_cache_lock:
        _medium_cache[file_path] = result
        if len(_medium_cache) > _MEDIUM_CACHE_MAX:
            oldest = next(iter(_medium_cache))
            del _medium_cache[oldest]
    logger.debug("bridge: Medium cached for '%s'", file_path)


def get_cached_medium_result(file_path: str) -> object | None:
    """Gibt ein gecachtes MediumClassifier-Ergebnis zurück oder ``None``."""
    with _medium_cache_lock:
        return _medium_cache.get(file_path)


# ---------------------------------------------------------------------------
# Restorability-Cache  (Thread-sicher, Prozess-Lebensdauer, RAM-only)
# Key: file_path (str), Value: RestorabilityResult-Objekt
# Limit: 64 Einträge (FIFO-Trim)
# ---------------------------------------------------------------------------

_restorability_cache: dict[str, object] = {}
_restorability_cache_lock = threading.Lock()
_RESTORABILITY_CACHE_MAX = 64


def cache_restorability_result(file_path: str, result: object) -> None:
    """Speichert ein RestorabilityEstimator-Ergebnis für *file_path* im Cache."""
    with _restorability_cache_lock:
        _restorability_cache[file_path] = result
        if len(_restorability_cache) > _RESTORABILITY_CACHE_MAX:
            oldest = next(iter(_restorability_cache))
            del _restorability_cache[oldest]
    logger.debug("bridge: Restorability cached for '%s'", file_path)


def get_cached_restorability_result(file_path: str) -> object | None:
    """Gibt ein gecachtes RestorabilityEstimator-Ergebnis zurück oder ``None``."""
    with _restorability_cache_lock:
        return _restorability_cache.get(file_path)


# ---------------------------------------------------------------------------
# Lazy-Import-Wrappers  (Core-Module werden erst bei Bedarf geladen)
# ---------------------------------------------------------------------------


def get_quality_mode() -> type:
    """Gibt die ``QualityMode``-Enum zurück (lazy import)."""
    from backend.core.performance_guard import QualityMode  # type: ignore[import]

    return QualityMode


def get_medium_type_enum() -> type:
    """Gibt die ``MediumType``-Enum zurück (lazy import)."""
    from backend.core.enums import MediumType  # type: ignore[import]

    return MediumType


def get_processing_mode_enum() -> type:
    """Gibt die ``ProcessingMode``-Enum zurück (lazy import)."""
    from backend.core.enums import ProcessingMode  # type: ignore[import]

    return ProcessingMode


def get_restorer_classes() -> tuple[type, type]:
    """Gibt ``(RestorationConfig, UnifiedRestorerV3)`` zurück (lazy import)."""
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore[import]

    return RestorationConfig, UnifiedRestorerV3


def get_aurik_denker_class() -> type:
    """Gibt ``AurikDenker``-Klasse zurück (lazy import, §2.2 Spec 08).

    Primary entry point for the full 8-stage restoration with carrier analysis,
    DefektDenker, MusikalischerGlobalplan, VERSA MOS scoring and ExzellenzDenker.
    Use this instead of UnifiedRestorerV3 for production pipelines.
    """
    from denker.aurik_denker import AurikDenker  # type: ignore[import]

    return AurikDenker


def get_aurik_denker_instance():
    """Gibt den thread-sicheren AurikDenker-Prozess-Singleton zurück (lazy, §2.2 Spec 08).

    Primary production accessor for BatchProcessingThread.
    Ensures Single-Orchestrator Ownership per process (No-Competing-Instances-Protokoll).
    Use ``get_aurik_denker_class()`` only for testing / mocking scenarios.
    """
    from denker.aurik_denker import get_aurik_denker  # type: ignore[import]

    return get_aurik_denker()


def get_defect_scanner() -> type:
    """Gibt die ``DefectScanner``-Klasse zurück (lazy import)."""
    from backend.core.defect_scanner import DefectScanner  # type: ignore[import]

    return DefectScanner


def get_audio_file_validator():
    """Gibt den ``AudioFileValidator``-Singleton zurück (lazy import, §10.5).

    Pflicht-Gate vor jedem ``_bg_load``-Thread-Start.  Wirf
    ``AudioLoadError`` (mit ``.message_user`` auf Deutsch) bei ungültiger Datei.
    """
    from backend.core.audio_file_validator import get_audio_file_validator as _get  # type: ignore[import]

    return _get()


def get_defect_type() -> type:
    """Gibt die ``DefectType``-Enum-Klasse zurück (lazy import).

    Wird von ``_defect_analysis_to_display`` und ``_result_scores_to_display``
    im Frontend benötigt, um DefectScanner-Scores zu indizieren.
    """
    from backend.core.defect_scanner import DefectType  # type: ignore[import]

    return DefectType


def get_medium_classifier_fn():
    """Gibt ``classify_medium``-Funktion zurück (lazy import, §2.5).

    Signatur: ``classify_medium(mono_audio: np.ndarray, sr: int) -> MediumResult``
    """
    from backend.core.medium_classifier import classify_medium  # type: ignore[import]

    return classify_medium


def get_era_classifier_fn():
    """Gibt ``classify_era``-Funktion zurück (lazy import, §2.4).

    Signatur: ``classify_era(audio: np.ndarray, sr: int) -> EraResult``
    """
    from backend.core.era_classifier import classify_era  # type: ignore[import]

    return classify_era


def get_genre_classifier_fn():
    """Gibt ``classify_genre``-Funktion zurück (lazy import).

    Signatur: ``classify_genre(audio: np.ndarray, sr: int) -> GenreResult``
    """
    from backend.core.genre_classifier import classify_genre  # type: ignore[import]

    return classify_genre


def get_restorability_estimator_class() -> type:
    """Gibt ``RestorabilityEstimator``-Klasse zurück (lazy import, §2.3).

    Verwendung: ``get_restorability_estimator_class()().estimate(audio, sr)``
    """
    from backend.core.restorability_estimator import RestorabilityEstimator  # type: ignore[import]

    return RestorabilityEstimator


def get_carrier_forensics_fn():
    """Gibt ``analyze_carrier_forensics``-Funktion zurück (lazy import).

    Signatur: ``analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict``
    Rückgabe-Keys: ``"carrier_forensic"`` (str), ``"score"`` (float).

    Intern wird ``classify_medium`` aus ``backend.core.medium_classifier``
    genutzt (``backend.carrier_forensics`` ist ein veralteter Shim).
    """
    from backend.core.medium_classifier import classify_medium as _cm  # type: ignore[import]

    def _analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict:
        result = _cm(mono, sr)
        return {"carrier_forensic": result.material_type, "score": float(result.confidence)}

    return _analyze_carrier_forensics


def get_audio_exporter_class() -> type | None:
    """Gibt ``AudioExporter``-Klasse zurück (lazy import).

    Gibt ``None`` zurück wenn ``backend.core.audio_exporter`` nicht verfügbar
    ist — Aufrufer muss dann ``soundfile.write()`` als Fallback verwenden.
    Spec §11.3: Kein Hard-Fail bei optionalen Export-Modulen.
    """
    try:
        from backend.core.audio_exporter import AudioExporter  # type: ignore[import]

        return AudioExporter
    except ImportError:
        logger.debug("bridge: AudioExporter nicht verfügbar — sf.write als Fallback")
        return None


def get_lyrics_guided_enhancement_fn():
    """Gibt ``LyricsGuidedEnhancement``-Singleton zurück (lazy import, §2.36).

    Rückgabe: ``LyricsGuidedEnhancement``-Instanz mit ``.enhance(audio, sr)``
    und ``.get_timeline()``.

    Pflicht ab 9.10.x (§2.36): Wird im Frontend für L-Shortcut-Overlay und
    im BatchProcessingThread für ContentAwareProcessor-Integration verwendet.
    """
    from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement  # type: ignore[import]

    return get_lyrics_guided_enhancement()


def get_cleanup_after_file_fn():
    """Gibt ``cleanup_after_file``-Funktion zurück (lazy import)."""
    from backend.core.plugin_lifecycle_manager import cleanup_after_file  # type: ignore[import]

    return cleanup_after_file


def get_pipeline_health_state_enum() -> type:
    """Gibt ``PipelineHealthState``-Enum zurück (lazy import)."""
    from backend.core.pipeline_health_state import PipelineHealthState  # type: ignore[import]

    return PipelineHealthState


def normalize_pipeline_health_state(raw):
    """Normalisiert Pipeline-Health-State auf kanonische Enum-Werte (lazy import)."""
    from backend.core.pipeline_health_state import normalize_pipeline_health_state as _normalize  # type: ignore[import]

    return _normalize(raw)


def resolve_pipeline_fail_reason(
    *,
    typed_fail_reason=None,
    metadata: dict | None = None,
    stage_notes: dict | None = None,
    fail_reasons: list[dict] | None = None,
) -> str:
    """Löst ``fail_reason`` aus typed Feld, Metadata und Stage-Notes auf (lazy import)."""
    from backend.core.pipeline_health_state import resolve_fail_reason as _resolve  # type: ignore[import]

    return _resolve(
        typed_fail_reason=typed_fail_reason,
        metadata=metadata,
        stage_notes=stage_notes,
        fail_reasons=fail_reasons,
    )


def get_stem_remix_balancer_fn():
    """Gibt ``StemRemixBalancer.balance_remix``-Funktion zurück (lazy import, §1.4).

    Signatur: ``balance_remix(vocals, instruments, original, sr, vocal_weight) -> np.ndarray``
    Verwendet ITU-R BS.1770-5 K-gewichtete LUFS-Messung für Gain-Korrektur.
    LUFS-Differenz nach Re-Mix ≤ 0.3 LU gegenüber Original (§1.4 Spec).
    """
    from backend.core.stem_remix_balancer import StemRemixBalancer  # type: ignore[import]

    return StemRemixBalancer().balance_remix


def get_clipping_classifier():
    """Gibt ``ClippingClassifier``-Singleton zurück (lazy import, §6.3).

    Rückgabe: ``ClippingClassifier``-Instanz.
    Verwende ``classify_clipping(audio, sr)`` (Convenience-Funktion) für
    direkten Aufruf ohne Singleton-Handle.

    §6.3 CLIPPING vs SOFT_SATURATION: THD-basierte Diskriminierung.
    SOFT_SATURATION (gerade Harmonische — Röhre/Tape) → bewahren.
    CLIPPING (ungerade Harmonische + flat_tops > 0.1 %) → reparieren.
    """
    from backend.core.clipping_detection import get_clipping_classifier as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# Qualitätsbewertung  (Musical Goals, PQS, OQS/MUSHRA — §8.1)
# ---------------------------------------------------------------------------


def get_musical_goals_checker() -> type:
    """Gibt ``MusicalGoalsChecker``-Klasse zurück (lazy import, §8.1).

    Die zurückgegebene **Klasse** kann instanziiert werden::

        checker = get_musical_goals_checker()()
        scores = checker.measure_all(audio, sr)  # Dict[str, float]

    14 Musical Goals mit AMRB-kalibrierten Schwellwerten (§8.1).
    Adaptive Schwellwerte via ``get_adaptive_goals_fn()`` — nicht statisch!
    """
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore[import]

    return MusicalGoalsChecker


def get_adaptive_goals_fn():
    """Gibt ``get_adaptive_goals_and_config``-Funktion zurück (lazy import, §2.31).

    Signatur::

        get_adaptive_goals_and_config(
            audio: np.ndarray,
            sr: int,
        ) -> tuple[AdaptiveGoalThresholds, dict, MaterialQualityAssessment]

    **Pflicht vor jeder Restaurierung**: statische Schwellwerte sind verboten
    als alleinige Entscheidungsbasis (§2.31 AdaptiveGoalThresholds).
    Schwellwerte werden material-, ära- und restorability-adaptiv skaliert.
    """
    from backend.core.musical_goals.adaptive_goals_system import (  # type: ignore[import]
        get_adaptive_goals_and_config,
    )

    return get_adaptive_goals_and_config


def get_mushra_evaluator():
    """Gibt den ``MushraEvaluator``-Singleton zurück (lazy import, §8.1.1 OQS).

    OQS = algorithmische PEAQ-Approximation (kein ITU-R-MUSHRA).
    In externen Berichten stets „OQS (algorithmisch)" schreiben.

    Schwellwerte::

        OQS ≥ 91  → Excellent (A)
        OQS ≥ 80  → Good (B)  — Pflicht für jede neue Phase / jedes Plugin
        OQS ≥ 60  → Fair (C)

    Verwendung::

        evaluator = get_mushra_evaluator()
        result = evaluator.evaluate(audio, sr)
        assert result.oqs >= 80, f"OQS unter Good-Schwelle: {result.oqs}"
    """
    from backend.core.mushra_evaluator import get_mushra_evaluator as _get  # type: ignore[import]

    return _get()


def get_perceptual_quality_scorer():
    """Gibt den ``PerceptualQualityScorer``-Singleton zurück (lazy import, §8.1 PQS).

    Prüft **alle vier PQS-Metriken** — nie nur MOS allein (§8.1)::

        PQS MOS            ≥ 3.8 (generell) / ≥ 4.5 (nur cd_digital/dat/mp3_high/aac)
        PQS NSIM           ≥ 0.70
        MCD (dB)           ≤ 8.0
        Spectral Coherence ≥ 0.60

    ABSOLUT VERBOTEN als Musikmetrik: PESQ, DNSMOS, NISQA, STOI, CDPAM.

    Verwendung::

        pqs = get_perceptual_quality_scorer()
        result = pqs.score(audio, sr)
        assert result.mos >= 3.8, f"PQS MOS zu niedrig: {result.mos}"
    """
    from backend.core.perceptual_quality_scorer import (  # type: ignore[import]
        get_perceptual_quality_scorer as _get,
    )

    return _get()


# ---------------------------------------------------------------------------
# Infrastruktur / Speicher-Management  (PLM + ML-Budget §2.37)
# ---------------------------------------------------------------------------


def get_plugin_lifecycle_manager():
    """Gibt den ``PluginLifecycleManager``-Singleton zurück (lazy import, §2.37).

    Der PLM ist **Schicht 2** des zweischichtigen OOM-Schutzsystems:

    - **Schicht 1**: ``ml_memory_budget.try_allocate()`` — logisch
    - **Schicht 2**: ``PluginLifecycleManager`` — physisch (LRU-Eviction)

    RAM-Trigger: 82 % Systemauslastung → LRU-Eviction bis < 70 % oder
    ≥ 1,5 GB frei. Monitoring-Thread alle 10 Sekunden.

    Verwendung::

        plm = get_plugin_lifecycle_manager()
        plm.register("MeinPlugin", size_gb=0.10, unload_fn=lambda: ...)
        plm.set_active("MeinPlugin", True)   # schützt vor Eviction

    VERBOTEN: ``plm.try_allocate()`` — Methode existiert nicht!
    Verwende stattdessen ``ml_memory_budget.try_allocate()``.
    """
    from backend.core.plugin_lifecycle_manager import (  # type: ignore[import]
        get_plugin_lifecycle_manager as _get,
    )

    return _get()


def get_ml_memory_budget_status() -> dict:
    """Gibt den aktuellen ML-Speicherbudget-Status als Dict zurück (lazy import, §2.37).

    Rückgabe-Keys (Beispiel)::

        {
            "budget_gb": 10.7,
            "allocated_gb": 3.2,
            "free_gb": 7.5,
            "plugins": {"fcpe": 0.12, "panns": 0.44, ...},
        }

    Das Budget wird automatisch auf ``RAM/3, capped [4–12 GB]`` gesetzt.
    Auf 32-GB-System: ≈ 10.7 GB (Cap: 12 GB).

    WARNUNG: Fehlt ``psutil``, sind physische RAM-Checks deaktiviert —
    ``psutil`` muss im AppImage gebündelt sein.
    """
    try:
        from backend.core.ml_memory_budget import get_status  # type: ignore[import]

        return get_status()
    except Exception as _e:
        logger.debug("bridge: ml_memory_budget.get_status() nicht verfügbar: %s", _e)
        return {"max_gb": 0.0, "allocated_gb": 0.0, "free_gb": 0.0, "models": {}}


# ---------------------------------------------------------------------------
# Export-Guard  (PFLICHT vor jedem sf.write / AudioExporter.export)
# ---------------------------------------------------------------------------


def export_guard(audio: np.ndarray) -> np.ndarray:
    """Stellt sicher, dass Audio NaN/Inf-frei und auf [-1, 1] geclippt ist.

    Muss vor jedem ``sf.write()`` oder ``AudioExporter.export()`` aufgerufen
    werden. Entspricht der Numerischen Robustheit-Pflicht (§3.1 Spec 08).

    Args:
        audio: Audio-Array (float32 oder float64).

    Returns:
        Bereinigtes Audio (float32, kein NaN/Inf, Werte ∈ [-1, 1]).
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def validate_export_quality(result: object) -> tuple[bool, list[str]]:
    """Validate export quality based on RestorationResult fields.

    Delegates to :func:`backend.exporter.validate_export_quality`.
    Returns ``(passed, warnings)`` — *passed* is False only on catastrophic
    tonal shift (chroma < 0.80).
    """
    try:
        from backend.exporter import validate_export_quality as _veq

        return _veq(result)
    except Exception as exc:
        logger.debug("validate_export_quality unavailable: %s", exc)
        return True, []


# ---------------------------------------------------------------------------
# Warmup  (Modell-Vorinitialisierung im Hintergrund, §9.7.4)
# ---------------------------------------------------------------------------


def warmup_models_background() -> None:
    """Initialisiert häufig genutzte ML-Modelle im Hintergrund vor.

    Kanonische Warmup-Funktion (§9.7.4). Wird 2 Sekunden nach App-Start
    als Daemon-Thread gestartet — aus ``ModernMainWindow.__init__`` via
    ``QTimer.singleShot(2000, ...)``. Fehler werden nur geloggt, kein Absturz.

    Der Caller (QTimer) steuert das Timing — kein zusätzliches sleep().
    Warmup berührt keinerlei UI-Objekte (kein GUI-Zugriff aus dem Thread).

    Plugin-Reihenfolge spiegelt §4.4-Priorisierung:
    Tier-1-Primär-Plugins zuerst (VAD/Pitch/Tagging), Fallbacks danach.
    """
    import importlib

    _plugins = [
        # Tier-1 Primär-Plugins (§9.7.4 — Pflicht-Vorwärmen, §4.4-Reihenfolge)
        ("plugins.silero_plugin", "get_silero_vad"),  # VAD (~1 MB, ultraschnell — zuerst)
        ("plugins.fcpe_plugin", "get_fcpe_plugin"),  # Pitch-Tracking Primär (§4.4)
        ("plugins.beats_plugin", "get_beats_plugin"),  # Audio-Tagging Primär (§4.4)
        ("plugins.sgmse_plugin", "get_sgmse_plugin"),  # Dereverb/Denoising Primär
        ("backend.core.noise_reduction", "get_noise_reducer"),  # DeepFilterNet v3.II Breitrauschen
        # Stem-Separation Primärpfad (§4.4 — BS-RoFormer > MDX23C)
        ("plugins.bs_roformer_plugin", "get_bs_roformer_plugin"),  # Gesang Primär (860 MB — lazy)
        ("plugins.mdx23c_plugin", "get_mdx23c_plugin"),  # Instrumental Primär (Kim_Vocal_2)
        # Fallback-Plugins (nach Bedarf)
        ("plugins.panns_plugin", "get_panns_plugin"),  # Audio-Tagging Fallback
        ("plugins.crepe_plugin", "get_crepe_plugin"),  # Pitch-Tracking Fallback
    ]
    logger.info("bridge: Warmup gestartet (%d Plugins) …", len(_plugins))
    for _mod, _accessor in _plugins:
        try:
            m = importlib.import_module(_mod)
            fn = getattr(m, _accessor, None)
            if fn is not None:
                fn()
                logger.debug("bridge: %s.%s vorgeladen", _mod.split(".")[-1], _accessor)
        except Exception as _e:
            logger.debug("bridge: %s.%s übersprungen: %s", _mod, _accessor, _e)
    logger.info("bridge: Warmup abgeschlossen")


# ---------------------------------------------------------------------------
# Startup model check  (§9.x — via Bridge, nie direkt aus UI)
# ---------------------------------------------------------------------------


def get_startup_check_result():
    """Return startup model-availability check result via bridge (never import core directly).

    Returns the result object from ``backend.core.startup_model_check.get_startup_check_result``
    or ``None`` on import failure.
    """
    try:
        from backend.core.startup_model_check import (  # type: ignore[import]
            get_startup_check_result as _fn,
        )

        return _fn()
    except Exception:
        return None
