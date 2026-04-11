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

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

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
    "clear_medium_cache",
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
    "get_medium_detector",  # §6.1 MediumDetector forensic chain
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
    "get_experience_insights",
    # Hintergrund-Vorwärmung
    "warmup_models_background",
    # §2.38 KMV / §2.39 OOM-Recovery / §2.37 RAM-Budget
    "get_deferred_refinement_job_class",
    "get_era_medium_constraint",
    "get_ml_memory_budget",
    "get_model_downloader",
    "get_recovery_checkpoint_fns",
    "get_save_checkpoint_fn",
    # §11 erweiterte Core-Module (bisher nicht bridge-zugänglich)
    "get_german_schlager_classifier_fn",
    "get_harmonic_preservation_guard",
    "get_feedback_chain",
    "get_physical_ceiling_estimator",
    "get_per_phase_musical_goals_gate",
    "get_emotional_arc_metric",
    "get_micro_dynamics_em",
    "get_goal_applicability_filter",
    "get_perceptual_salience_estimator",
    # Startup-/Self-Heal
    "get_startup_check_result",
    # Content-Addressed LRU Cache — Utility
    "content_cache_key",
    # Pre-Analysis — single authoritative entry point (§pre_analysis)
    "run_pre_analysis",
    "PreAnalysisResult",
    # Audio-Import — canonical cascade (§11 VERBOTEN: sf.read / librosa.load direkt)
    "get_load_audio_fn",
]

# ---------------------------------------------------------------------------
# _AnalysisLruCache — Unified Thread-safe LRU Cache mit Content-Addressing
#
# Ersetzt die vier früheren FIFO-Dict-Caches durch eine gemeinsame Klasse:
# - LRU-Eviction statt FIFO: heiße Einträge bleiben, kalte fliegen raus
# - Content-Addressing: selbes Audio unter zwei Pfaden trifft denselben Slot
# - Ein Lock statt vier separater Locks
# - Optionaler Path→ContentKey-Alias für schnelle path-basierte Lookups
# ---------------------------------------------------------------------------

_ANALYSIS_CACHE_MAX = 64
_CONTENT_CHUNK = 4096  # Bytes vom Anfang + Ende für SHA-256 Content-Key


class _AnalysisLruCache:
    """Thread-safe LRU cache keyed by content-hash (or arbitrary string).

    Stores analysis results under a content-addressed key so that the same
    audio file is not re-analysed when its path changes (e.g. rename before
    OOM-checkpoint resume).  Path→key aliases are maintained for fast
    backward-compatible path lookups.

    Args:
        maxsize: Maximum number of entries before LRU eviction.
    """

    def __init__(self, maxsize: int = _ANALYSIS_CACHE_MAX) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._path_to_key: dict[str, str] = {}  # path → content_key
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def put(self, key: str, value: Any, path_alias: str | None = None) -> None:
        """Insert *value* under *key*, evicting LRU entry when full."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if path_alias:
                self._path_to_key[path_alias] = key
            while len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                # Clean up alias mapping for evicted key
                self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != evicted_key}

    def get(self, key: str) -> Any | None:
        """Return cached value for *key* and promote to MRU, or ``None``."""
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def get_by_path(self, path: str) -> Any | None:
        """Return cached value using a path alias, or ``None``."""
        with self._lock:
            key = self._path_to_key.get(path)
            if key is None or key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def remove(self, key_or_path: str) -> None:
        """Remove entry by content-key or path alias."""
        with self._lock:
            # Try as path alias first
            key = self._path_to_key.pop(key_or_path, key_or_path)
            self._data.pop(key, None)
            # Also remove any alias pointing to same key
            self._path_to_key = {p: k for p, k in self._path_to_key.items() if k != key}

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._data.clear()
            self._path_to_key.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def content_cache_key(file_path: str) -> str:
    """Compute a content-addressed cache key for *file_path*.

    Uses SHA-256 over the first and last ``_CONTENT_CHUNK`` bytes of the
    file (fast, file-size independent).  Falls back to the path itself when
    the file is not readable (e.g. missing/locked).

    Args:
        file_path: Absolute path to an audio file.

    Returns:
        A 64-character hex string suitable as a cache key, or the path
        itself on I/O error.
    """
    try:
        size = os.path.getsize(file_path)
        with open(file_path, "rb") as fh:
            head = fh.read(_CONTENT_CHUNK)
            if size > _CONTENT_CHUNK * 2:
                fh.seek(-_CONTENT_CHUNK, 2)
                tail = fh.read(_CONTENT_CHUNK)
            else:
                tail = b""
        return hashlib.sha256(head + tail + str(size).encode()).hexdigest()
    except OSError:
        return file_path


# Singleton caches — one per analysis type for independent eviction
_defect_lru: _AnalysisLruCache = _AnalysisLruCache()
_era_genre_lru: _AnalysisLruCache = _AnalysisLruCache()
_medium_lru: _AnalysisLruCache = _AnalysisLruCache()
_restorability_lru: _AnalysisLruCache = _AnalysisLruCache()


# ---------------------------------------------------------------------------
# Defect-Scan-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_defect_result(file_path: str, result: object) -> None:
    """Cache a DefectScanner result under a content-addressed key.

    Thread-safe.  Uses LRU eviction (max 64 entries).  Identical audio
    stored under a different path will hit the same cache slot.
    """
    key = content_cache_key(file_path)
    _defect_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: DefectScan cached for '%s' (key=%.8s…)", file_path, key)


def get_cached_defect_result(file_path: str) -> object | None:
    """Return a cached DefectScanner result or ``None``."""
    key = content_cache_key(file_path)
    result = _defect_lru.get(key)
    if result is None:
        result = _defect_lru.get_by_path(file_path)
    return result


def clear_defect_cache(file_path: str | None = None) -> None:
    """Remove one entry (by path) or all entries from the defect cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _defect_lru.remove(key)
    else:
        _defect_lru.clear()


# ---------------------------------------------------------------------------
# Era/Genre-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_era_genre_result(
    file_path: str,
    era_result: object | None = None,
    genre_result: object | None = None,
) -> None:
    """Cache Era/Genre classification results for *file_path*.

    Thread-safe, LRU-evicting, content-addressed.
    """
    key = content_cache_key(file_path)
    _era_genre_lru.put(
        key,
        {"era_result": era_result, "genre_result": genre_result},
        path_alias=file_path,
    )
    logger.debug("bridge: Era/Genre cached for '%s' (key=%.8s…)", file_path, key)


def get_cached_era_genre_result(file_path: str) -> dict[str, object] | None:
    """Return cached Era/Genre results or ``None``.

    Returns:
        dict with keys ``era_result`` and ``genre_result``, or ``None``.
    """
    key = content_cache_key(file_path)
    result = _era_genre_lru.get(key)
    if result is None:
        result = _era_genre_lru.get_by_path(file_path)
    return result


def clear_era_genre_cache(file_path: str | None = None) -> None:
    """Remove one entry (by path) or all entries from the Era/Genre cache."""
    if file_path is not None:
        key = content_cache_key(file_path)
        _era_genre_lru.remove(key)
    else:
        _era_genre_lru.clear()


# ---------------------------------------------------------------------------
# Medium-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_medium_result(file_path: str, result: object) -> None:
    """Cache a MediumClassifier result for *file_path*."""
    key = content_cache_key(file_path)
    _medium_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: Medium cached for '%s' (key=%.8s…)", file_path, key)


def get_cached_medium_result(file_path: str) -> object | None:
    """Return a cached MediumClassifier result or ``None``."""
    key = content_cache_key(file_path)
    result = _medium_lru.get(key)
    if result is None:
        result = _medium_lru.get_by_path(file_path)
    return result


def clear_medium_cache(file_path: str | None = None) -> None:
    """Invalidate medium cache entry for *file_path*, or entire cache when ``None``."""
    if file_path is None:
        _medium_lru.clear()
        logger.debug("bridge: Medium-Cache vollst\u00e4ndig geleert.")
    else:
        key = content_cache_key(file_path)
        _medium_lru.remove(key)
        _medium_lru.remove(file_path)  # remove() handles path-alias too
        logger.debug("bridge: Medium-Cache f\u00fcr '%s' geleert.", file_path)


# ---------------------------------------------------------------------------
# Restorability-Cache  (Thread-sicher, LRU, content-addressed)
# ---------------------------------------------------------------------------


def cache_restorability_result(file_path: str, result: object) -> None:
    """Cache a RestorabilityEstimator result for *file_path*."""
    key = content_cache_key(file_path)
    _restorability_lru.put(key, result, path_alias=file_path)
    logger.debug("bridge: Restorability cached for '%s' (key=%.8s…)", file_path, key)


def get_cached_restorability_result(file_path: str) -> object | None:
    """Return a cached RestorabilityEstimator result or ``None``."""
    key = content_cache_key(file_path)
    result = _restorability_lru.get(key)
    if result is None:
        result = _restorability_lru.get_by_path(file_path)
    return result


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
    """Gibt einen MediumDetector-basierten Legacy-Kompat-Callable zurück.

    Signatur-kompatibel zu ``classify_medium(mono_audio, sr)`` für Altaufrufer,
    intern jedoch detector-only (kein direkter MediumClassifier-Aufruf).
    """
    from forensics.medium_detector import get_medium_detector as _get_md  # type: ignore[import]

    class _CompatMediumResult:
        def __init__(self, primary_material: str, confidence: float, transfer_chain: list[str], chain_label: str):
            self.material_type = primary_material
            self.material = primary_material
            self.primary_material = primary_material
            self.confidence = float(confidence)
            self.transfer_chain = list(transfer_chain)
            self.chain_label = chain_label

    def _classify_medium_compat(mono_audio: np.ndarray, sr: int) -> _CompatMediumResult:
        _res = _get_md().detect(mono_audio, sr, file_ext="")
        _chain = list(getattr(_res, "transfer_chain", None) or [str(_res.primary_material)])
        _chain_label = str(getattr(_res, "chain_label", " -> ".join(_chain)))
        return _CompatMediumResult(
            primary_material=str(_res.primary_material),
            confidence=float(getattr(_res, "confidence", 0.0)),
            transfer_chain=_chain,
            chain_label=_chain_label,
        )

    return _classify_medium_compat


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


def get_medium_detector():
    """Return the ``MediumDetector`` singleton (lazy import, §6.1 / §11.1).

    Canonical forensic carrier-chain detector.  Preferred over
    ``get_medium_classifier_fn()`` in all production paths because
    ``MediumDetector.detect()`` supplies the required ``file_ext`` context
    for codec-format digital-file prior adjustment (§6.7b).

    Invariante: ``primary_material`` is always a key from SUPPORTED_MATERIALS
    (cassette → tape, reel_wire → wire_recording, etc. normalised internally).

    Usage::

        md = get_medium_detector()
        result = md.detect(audio, sr, file_ext=Path(file_path).suffix)
        material = result.primary_material  # e.g. "tape", "vinyl"
    """
    try:
        from forensics.medium_detector import get_medium_detector as _get  # type: ignore[import]

        return _get()
    except ImportError:
        logger.debug("bridge: MediumDetector nicht importierbar — stub aktiv")
        return None


def get_carrier_forensics_fn():
    """Gibt ``analyze_carrier_forensics``-Funktion zurück (lazy import).

    Signatur: ``analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict``
    Rückgabe-Keys: ``"carrier_forensic"`` (str), ``"score"`` (float).

    Intern wird ``MediumDetector.detect`` genutzt (detector-only).
    """
    from forensics.medium_detector import get_medium_detector as _get_md  # type: ignore[import]

    def _analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict:
        result = _get_md().detect(mono, sr, file_ext="")
        return {"carrier_forensic": str(result.primary_material), "score": float(result.confidence)}

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


def get_experience_insights(result: Any) -> dict[str, Any]:
    """Extract normalized joy/fatigue/recommendation insights from a result object.

    Frontend-safe helper for AurikErgebnis/RestorationResult-like objects.
    Returns stable keys even if metadata is partially missing.
    """
    _meta = getattr(result, "metadata", None)
    if not isinstance(_meta, dict):
        _meta = {}

    _joy = _meta.get("joy_runtime_index") if isinstance(_meta.get("joy_runtime_index"), dict) else {}
    _auto = (
        _meta.get("auto_improvement_recommendations")
        if isinstance(_meta.get("auto_improvement_recommendations"), dict)
        else {}
    )
    _song_cal = _meta.get("song_calibration") if isinstance(_meta.get("song_calibration"), dict) else {}
    _cluster = _song_cal.get("cluster_policy") if isinstance(_song_cal.get("cluster_policy"), dict) else {}

    _recommendations = _auto.get("recommendations") if isinstance(_auto.get("recommendations"), list) else []

    def _safe01(v: Any) -> float:
        try:
            vf = float(v)
        except Exception:
            return 0.0
        if not np.isfinite(vf):
            return 0.0
        return float(np.clip(vf, 0.0, 1.0))

    _normalized_recommendations: list[dict[str, Any]] = []
    for _r in _recommendations:
        if not isinstance(_r, dict):
            continue
        _normalized_recommendations.append(
            {
                "priority": str(_r.get("priority", "info") or "info"),
                "focus": str(_r.get("focus", "") or ""),
                "reason": str(_r.get("reason", "") or ""),
                "action": str(_r.get("action", "") or ""),
            }
        )

    _cnt_raw = _auto.get("count", len(_normalized_recommendations))
    try:
        _cnt = int(_cnt_raw)
    except Exception:
        _cnt = len(_normalized_recommendations)
    _cnt = max(_cnt, len(_normalized_recommendations), 0)

    _tc = _meta.get("team_coordination") if isinstance(_meta.get("team_coordination"), dict) else {}
    _tc_events_raw = _tc.get("events") if isinstance(_tc.get("events"), list) else []
    _tc_events: list[dict[str, Any]] = []
    for _tce in _tc_events_raw:
        if not isinstance(_tce, dict):
            continue
        _tc_events.append(
            {
                "phase_id": str(_tce.get("phase_id", "") or ""),
                "action": str(_tce.get("action", "") or ""),
                "reason": str(_tce.get("reason", "") or ""),
                "excluded_goals": list(_tce.get("excluded_goals", []) or []),
            }
        )
    try:
        _tc_count = int(_tc.get("event_count", len(_tc_events)))
    except Exception:
        _tc_count = len(_tc_events)
    _pt_summary = dict(_tc.get("phase_type_summary", {}) or {})

    return {
        "joy_index": _safe01(_joy.get("joy_index", 0.0)),
        "fatigue_index": _safe01(_joy.get("fatigue_index", 0.0)),
        "cluster_key": str(_song_cal.get("cluster_key", "") or ""),
        "cluster_policy": dict(_cluster) if isinstance(_cluster, dict) else {},
        "recommendations": _normalized_recommendations,
        "recommendation_count": _cnt,
        "team_coordination": {
            "event_count": _tc_count,
            "events": _tc_events,
            "phase_type_summary": _pt_summary,
        },
    }


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
    logger.info("bridge: warmup started (%d plugins) …", len(_plugins))
    for _mod, _accessor in _plugins:
        try:
            m = importlib.import_module(_mod)
            fn = getattr(m, _accessor, None)
            if fn is not None:
                fn()
                logger.debug("bridge: %s.%s vorgeladen", _mod.split(".")[-1], _accessor)
        except Exception as _e:
            logger.debug("bridge: %s.%s übersprungen: %s", _mod, _accessor, _e)
    logger.info("bridge: warmup complete")


# ---------------------------------------------------------------------------
# §2.38 KMV + §2.39 OOM-Recovery + §2.37 RAM-Budget  (Lazy-Wrapper)
# ---------------------------------------------------------------------------


def get_deferred_refinement_job_class() -> type:
    """Return ``DeferredRefinementJob`` class (lazy import, §2.38 KMV Stufe 2).

    Used by MLRefinementThread and ModernMainWindow._maybe_start_kmv_refinement.
    """
    from backend.core.deferred_refinement_job import DeferredRefinementJob  # type: ignore[import]

    return DeferredRefinementJob


def get_save_checkpoint_fn():
    """Return ``save_checkpoint`` from recovery_checkpoint (lazy, §2.39)."""
    from backend.core.recovery_checkpoint import save_checkpoint  # type: ignore[import]

    return save_checkpoint


def get_recovery_checkpoint_fns() -> tuple:
    """Return ``(cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint)`` (lazy, §2.39).

    Usage::

        cleanup_fn, find_fn, delete_fn = get_recovery_checkpoint_fns()
        cleanup_fn()
        checkpoints = find_fn()
        delete_fn(input_path)
    """
    from backend.core.recovery_checkpoint import (  # type: ignore[import]
        cleanup_expired_checkpoints,
        delete_checkpoint,
        find_pending_checkpoints,
    )

    return cleanup_expired_checkpoints, find_pending_checkpoints, delete_checkpoint


def get_era_medium_constraint() -> tuple:
    """Return ``(MEDIUM_DECADE_FLOOR, constrain_era_to_medium)`` from era_classifier (lazy import).

    Usage::

        floor_map, constrain_fn = get_era_medium_constraint()
        era = constrain_fn(era_result, medium_type)
        floor = floor_map.get(medium_type)
    """
    from backend.core.era_classifier import (  # type: ignore[import]
        MEDIUM_DECADE_FLOOR,
        constrain_era_to_medium,
    )

    return MEDIUM_DECADE_FLOOR, constrain_era_to_medium


def get_ml_memory_budget():
    """Return the ``MlMemoryBudget`` singleton (lazy import, §2.37).

    Usage::

        budget = get_ml_memory_budget()
        ok = budget.try_allocate("kmv_job", size_gb)
        budget.release("kmv_job")

    VERBOTEN: ``get_plugin_lifecycle_manager().try_allocate()`` — existiert nicht.
    """
    from backend.core.ml_memory_budget import get_ml_memory_budget as _get  # type: ignore[import]

    return _get()


def get_model_downloader():
    """Return the ``ModelDownloader`` singleton (lazy import, §9.x / §13.x).

    Used in Aurik startup self-heal to repair missing/corrupted bundled models.
    """
    from backend.core.model_downloader import get_model_downloader as _get  # type: ignore[import]

    return _get()


# ---------------------------------------------------------------------------
# §11 erweiterte Core-Module  (bisher nicht über Bridge zugänglich)
# Alle 9 Getter sind lazy — kein Import-Overhead beim Bridge-Load.
# ---------------------------------------------------------------------------


def get_german_schlager_classifier_fn():
    """Return the ``GermanSchlagerClassifier`` singleton (lazy, §2.1 Pipeline).

    Alias wrapper around ``backend.core.german_schlager_classifier``.
    The canonical implementation lives in ``backend.core.genre_classifier``.

    Usage::

        clf = get_german_schlager_classifier_fn()
        result = clf.classify(audio, sr)
        profile = clf.get_restoration_profile(result)
    """
    from backend.core.german_schlager_classifier import (  # type: ignore[import]
        get_german_schlager_classifier,
    )

    return get_german_schlager_classifier()


def get_harmonic_preservation_guard():
    """Return the ``HarmonicPreservationGuard`` singleton (lazy).

    Guards all spectral modifications against harmonic structure loss.
    Run ``guard.protect(audio, sr, fn)`` to wrap any processing function.

    Usage::

        guard = get_harmonic_preservation_guard()
        restored = guard.protect(audio, sr, my_phase_fn)
    """
    from backend.core.harmonic_preservation_guard import (  # type: ignore[import]
        get_harmonic_preservation_guard as _get,
    )

    return _get()


def get_feedback_chain():
    """Return the ``FeedbackChain`` singleton (lazy, §2.33 FeedbackChain-Rollback).

    Manages iterative quality improvement with automatic rollback when
    MOS degrades by more than 0.05 (§8.2 universelle Garantien).

    Usage::

        fc = get_feedback_chain()
        result = fc.run(audio, sr, phase_fns, target_score=0.78)
    """
    from backend.core.feedback_chain import get_feedback_chain as _get  # type: ignore[import]

    return _get()


def get_physical_ceiling_estimator():
    """Return the ``PhysicalCeilingEstimator`` singleton (lazy).

    Estimates the theoretical maximum quality achievable for a given
    audio fragment given its material degradation state.
    Terminates FeedbackChain when ceiling Δ < 3 % (§2.31).

    Usage::

        pce = get_physical_ceiling_estimator()
        ceiling = pce.estimate(audio, sr, material_type)
        assert ceiling.delta_achievable >= 0.03, "Ceiling reached — terminate"
    """
    from backend.core.physical_ceiling_estimator import (  # type: ignore[import]
        get_physical_ceiling_estimator as _get,
    )

    return _get()


def get_per_phase_musical_goals_gate():
    """Return the ``PerPhaseMusicalGoalsGate`` singleton (lazy, §2.29 PMGG).

    The PMGG wraps individual restoration phases and enforces Musical Goal
    regression checks with retry cascades (P1 4 retries, P2 4 retries,
    P3 1–3 retries, P4/P5 logged only).

    Usage::

        gate = get_per_phase_musical_goals_gate()
        result_audio = gate.wrap_phase("phase_03", audio, sr, phase_fn, ...)
    """
    from backend.core.per_phase_musical_goals_gate import get_phase_gate  # type: ignore[import]

    return get_phase_gate()


def get_emotional_arc_metric():
    """Return the ``EmotionalArcPreservationMetric`` singleton (lazy, §8.3).

    Measures arousal/valence arc preservation (Pearson ≥ 0.85/0.80).
    Also exposes ``correct_emotional_arc(original, restored, sr)`` post-MDEM
    macro-gain correction.

    Usage::

        metric = get_emotional_arc_metric()
        arc = metric.measure(audio, sr)
        corrected = metric.correct(original, restored, sr)
    """
    from backend.core.emotional_arc_preservation import (  # type: ignore[import]
        get_emotional_arc_metric as _get,
    )

    return _get()


def get_micro_dynamics_em():
    """Return the ``MicroDynamicsEnvelopeMorphing`` singleton (lazy, §8.3 MDEM).

    400 ms LUFS-profile morphing: recovers micro-dynamic envelope lost
    during denoising/dereverb.  Gain limit: 4 dB (Restoration), 6 dB (Studio).

    Usage::

        mdem = get_micro_dynamics_em()
        morphed = mdem.morph(restored, original, sr)
    """
    from backend.core.micro_dynamics_envelope_morphing import get_mdem  # type: ignore[import]

    return get_mdem()


def get_goal_applicability_filter():
    """Return the ``GoalApplicabilityFilter`` singleton (lazy, §2.31).

    Determines which of the 14 Musical Goals are applicable for a given
    audio fragment based on material, era, and content type.
    Mono-era recordings have SpatialDepthMetric deactivated automatically.

    Usage::

        gaf = get_goal_applicability_filter()
        result = gaf.evaluate(audio, sr, material_type, era_decade)
        active_goals = result.active_goals  # set[str]
    """
    from backend.core.goal_applicability_filter import get_goal_filter  # type: ignore[import]

    return get_goal_filter()


def get_perceptual_salience_estimator():
    """Return the ``PerceptualSalienceEstimator`` singleton (lazy, §9.1c).

    Annotates each detected defect with a psychoacoustic salience score
    (Fastl & Zwicker 2007).  Masked defects receive reduced severity:
    ``severity * (0.3 + 0.7 * mean_salience)``.

    Usage::

        pse = get_perceptual_salience_estimator()
        annotations = pse.annotate(defect_list, audio, sr)
    """
    from backend.core.perceptual_salience import (  # type: ignore[import]
        get_perceptual_salience_estimator as _get,
    )

    return _get()


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


# ---------------------------------------------------------------------------
# Pre-Analysis — single authoritative entry point re-export
# ---------------------------------------------------------------------------


def run_pre_analysis(
    audio_native,
    sr_native: int,
    *,
    audio_48k=None,
    file_path: str = "",
    progress_callback=None,
    scan_progress_callback=None,
    store_in_bridge_cache: bool = True,
):
    """Bridge re-export of backend.core.pre_analysis.run_pre_analysis.

    See that module for full documentation.

    Args:
        scan_progress_callback: Optional ``(pct: float) -> None`` forwarded to
            ``DefectScanner.scan()`` for fine-grained scan progress (0–100).
    """
    from backend.core.pre_analysis import run_pre_analysis as _fn

    return _fn(
        audio_native,
        sr_native,
        audio_48k=audio_48k,
        file_path=file_path,
        progress_callback=progress_callback,
        scan_progress_callback=scan_progress_callback,
        store_in_bridge_cache=store_in_bridge_cache,
    )


# PreAnalysisResult is imported lazily; expose as a module-level name via
# a try-block so bridge consumers can do:
#   from backend.api.bridge import PreAnalysisResult
try:
    from backend.core.pre_analysis import PreAnalysisResult
except Exception:  # pragma: no cover
    PreAnalysisResult = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Audio-Import — Kaskade soundfile → pedalboard/FFmpeg → pydub
# ---------------------------------------------------------------------------


def get_load_audio_fn():
    """Return ``load_audio_file`` from backend.file_import (lazy).

    The returned function signature::

        load_audio_file(filepath, target_sr=None, mono=False) -> dict | None

    The dict contains keys ``audio`` (np.ndarray float32) and ``sr`` (int).
    Falls back to ``None`` when the import chain fails so callers can degrade
    gracefully.
    """
    from backend.file_import import load_audio_file  # type: ignore[import]

    return load_audio_file
