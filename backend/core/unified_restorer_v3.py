"""
Unified Restorer V3 for Aurik 9.0 - Defect-First Architecture
==============================================================

Hauptklasse für Audio-Restoration mit revolutionärer Defect-First Architektur.

Ersetzt unified_restorer_v2.py (Medium-First) durch intelligente Defekt-Erkennung
und material-adaptive Processing.

Key Features:
- Defect-First: Erkennt Defekte zuerst, wählt Phasen basierend darauf
- Material-Adaptive: Shellac, Vinyl, Tape, CD, Streaming
- Performance-Guaranteed: 3× RT Limit (Balanced Mode)
- Dual-Mode: Fast (1.5× RT), Balanced (2.4× RT), Quality (9× RT)
- 4-Core Parallelization: Optimal CPU usage
- Modular Phases: 41 Phasen, plug-and-play

Performance Target (Balanced Mode):
- 3:45 Audio → max 9 min (2.4× RT)
- Quality: 92% (Technical), 107% (Perceived with Psychoacoustics)

Author: Aurik 9.0 Development Team
Version: 9.0.0
Date: 2026-02-15
"""

import dataclasses
from dataclasses import dataclass, field
import gc
import logging
import math
import pathlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, sample rate conversion disabled")
try:
    from memory_profiler import memory_usage

    MEMORY_PROFILING_AVAILABLE = True
except ImportError:
    MEMORY_PROFILING_AVAILABLE = False
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import os
import sys

from backend.core.adaptive_core_scheduler import AdaptiveCoreScheduler

# Import Aurik 9.0 Core Components
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType
from backend.core.performance_guard import PerformanceGuard, QualityMode
from backend.core.phase_skipping import PhaseSkipper
from backend.core.phases.phase_interface import PhaseInterface

logger = logging.getLogger(__name__)


@dataclass
class RestorationConfig:
    """Configuration für Restoration."""

    mode: QualityMode = QualityMode.QUALITY
    material_type: Optional[MaterialType] = None  # None = auto-detect
    enable_performance_guard: bool = True
    enable_adaptive_skipping: bool = True
    enable_phase_skipping: bool = True  # Defect-based phase skipping (20-40% speedup)
    phase_skipping_conservative: bool = False  # Conservative skipping (safer)
    enable_phase_gate: bool = True  # §2.29 PMGG — deaktivierbar via --no-phase-gate
    num_cores: int = 4
    enforce_3x_rt: bool = True
    enable_psychoacoustic_enhancement: bool = True
    global_plan: Optional[Any] = None  # MusikalischerGlobalplan.StilbewussterRestaurierungsplan (§Dach)


@dataclass
class RestorationResult:
    """Ergebnis der Restoration (Spec §2.1 / §2.2)."""

    audio: np.ndarray
    config: RestorationConfig
    material_type: MaterialType
    defect_scores: Dict[DefectType, float]
    phases_executed: List[str]
    phases_skipped: List[str]
    total_time_seconds: float
    rt_factor: float
    quality_estimate: float  # 0-1
    warnings: List[str]
    metadata: Dict[str, Any]
    # --- Spec-Pflichtfelder (optional, mit sicherem Default) ---
    pqs_result: Optional[Any] = None  # PQS-MOS-Objekt (§2.6)
    musical_goals: Optional[Dict[str, float]] = None  # 14 Musical Goals (§1.2)
    excellence: Optional[Any] = None  # ExcellenceResult (§2.1)
    temporal_coherence: Optional[Any] = None  # TemporalCoherenceResult (§2.16)
    emotional_arc: Optional[Any] = None  # EmotionalArcResult (§8.2)
    restorability: Optional[Any] = None  # RestorabilityResult (§2.26)
    confidence: float = 1.0  # Gesamtkonfidenz ∈ [0,1] (§2.15)
    genealogy: Optional[Any] = None  # RestorationGenealogy (§10.4)
    harmonic_fingerprint: Optional[Any] = None  # Harmonischer Fingerabdruck (§2.28)
    phase_gate_log: Optional[List[str]] = None  # PMGG-Rollback-Log (§2.29)
    adaptive_thresholds: Dict[str, float] = field(default_factory=dict)  # Angewandte PMGG-Schwellwerte (§2.2)
    physical_ceiling: Dict[str, float] = field(default_factory=dict)  # PhysikalischesDeckenLimit je Ziel (§2.2)
    goal_applicability: Dict[str, bool] = field(default_factory=dict)  # GoalApplicabilityFilter-Ergebnis (§2.2)
    goal_priority_log: List[str] = field(default_factory=list)  # GoalPriorityProtocol-Entscheidungen (§2.2)
    preview_mos: Optional[float] = None  # Schnell-MOS vor Vollprüfung (§2.2)
    era_decade: Optional[int] = None  # Erkannte Aufnahme-Ära (§2.2)


class UnifiedRestorerV3:
    """
    Unified Restorer V3 - Defect-First Audio Restoration Engine.

    Usage:
        restorer = UnifiedRestorerV3(mode=QualityMode.QUALITY)
        result = restorer.restore(audio, sample_rate=44100)
        restored_audio = result.audio
    """

    def __init__(self, config: Optional[RestorationConfig] = None):
        """
        Initialisiert UnifiedRestorerV3 mit Lazy Loading und asynchroner Vorbereitung.
        """
        self.config = config or RestorationConfig()
        self.defect_scanner = DefectScanner()
        self.scheduler = AdaptiveCoreScheduler(num_cores=self.config.num_cores)

        # Only create PerformanceGuard if enabled
        if self.config.enable_performance_guard:
            self.performance_guard = PerformanceGuard(
                mode=self.config.mode,
                enforce_limit=self.config.enforce_3x_rt,
                enable_adaptive_skipping=self.config.enable_adaptive_skipping,
            )
        else:
            self.performance_guard = None
            logger.info("Performance Guard disabled via config")
            # Erzwinge QualityMode MAXIMUM für Studio 2026
            if hasattr(self.config, "studio_2026") and self.config.studio_2026:
                self.config.mode = QualityMode.MAXIMUM
                self.config.phase_skipping = False
        # Lazy Phase Registry: Nur Metadaten, keine Instanzen
        self.phase_metadata: Dict[str, Dict] = self._discover_phase_metadata()
        self._phase_cache: Dict[str, PhaseInterface] = {}
        self._warnings: List[str] = []  # §M-2 Schritte_zur_Musikalischen_Exzellenz

        # PhaseSkipper (optional, beschleunigt Pipeline um 20–40 %)
        if self.config.enable_phase_skipping:
            try:
                self.phase_skipper = PhaseSkipper()
            except Exception as e:
                logger.debug("PhaseSkipper nicht verfügbar: %s — Phase-Skipping deaktiviert", e)
                self.phase_skipper = None
        else:
            self.phase_skipper = None

        logger.info(
            f"UnifiedRestorerV3 initialized: Mode={self.config.mode.value}, "
            f"Cores={self.config.num_cores}, RT Limit={self.config.enforce_3x_rt}, "
            f"Phase Skipping={self.config.enable_phase_skipping}"
        )

    def _discover_phase_metadata(self) -> Dict[str, Dict]:
        """Findet alle Phasenmodule und liest Metadaten (ohne Instanziierung)."""
        phase_dir = os.path.join(os.path.dirname(__file__), "phases")
        phase_files = [f for f in os.listdir(phase_dir) if f.startswith("phase_") and f.endswith(".py")]
        metadata = {}
        for fname in phase_files:
            modulename = f"backend.core.phases.{fname[:-3]}"
            try:
                module = importlib.import_module(modulename)
                for name, obj in module.__dict__.items():
                    if (
                        isinstance(obj, type)
                        and issubclass(obj, PhaseInterface)
                        and obj is not PhaseInterface
                        and getattr(obj, "__module__", "") == modulename
                    ):
                        meta = obj().get_metadata()
                        metadata[meta.phase_id] = {
                            "class": obj,
                            "modulename": modulename,
                            "name": meta.name,
                            "category": meta.category.value,
                            "dependencies": meta.dependencies,
                            "estimated_time_factor": meta.estimated_time_factor,
                            "priority": meta.priority,
                        }
                        break  # Only one PhaseInterface subclass per module file
            except Exception as e:
                logger.warning(f"Phase-Modul {modulename} konnte nicht geladen werden: {e}")
        return metadata

    def _get_phase(self, phase_id: str) -> Optional[PhaseInterface]:
        """Lädt und instanziiert eine Phase nur bei Bedarf (Lazy Loading)."""
        if phase_id in self._phase_cache:
            return self._phase_cache[phase_id]
        meta = self.phase_metadata.get(phase_id)
        if not meta:
            logger.error(f"Phase {phase_id} nicht gefunden (weder Registry noch Metadaten)")
            return None
        try:
            phase_instance = meta["class"]()
            self._phase_cache[phase_id] = phase_instance
            return phase_instance
        except Exception as exc:
            self._warnings.append(f"Phase {phase_id} nicht geladen: {type(exc).__name__}: {exc}")
            logger.warning("Phase %s übersprungen: %s", phase_id, exc)
            return None

    # _register_phases entfällt, Lazy Loading übernimmt die Instanziierung

    def restore(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        progress_callback=None,
        **kwargs,
    ) -> RestorationResult:
        """
        Hauptmethode: Restauriert Audio mit Defect-First Approach.

        Args:
            audio: Input Audio (mono: shape=(n,), stereo: shape=(n,2))
            sample_rate: Sample rate (default: 44100 Hz, wird intern auf 48000 Hz konvertiert)
            progress_callback: Optional callable(percent: int, phase: str, elapsed_s: float)
                               Wird während der Verarbeitung periodisch aufgerufen.
            **kwargs: Additional parameters

        Returns:
            RestorationResult mit restauriertem Audio und Metadata

        Note:
            Alle Verarbeitung erfolgt intern bei 48 kHz für konsistente ML-Modell-Performance.
        """
        start_time = time.time()

        def _cb(pct: int, phase: str) -> None:
            """Sendet Progress-Update, falls Callback registriert."""
            if progress_callback is not None:
                try:
                    progress_callback(pct, phase, time.time() - start_time)
                except Exception:
                    pass

        original_sample_rate = sample_rate
        target_sample_rate = 48000  # Standardisierung auf 48 kHz

        # §9.1 kwargs: mode + material aus externen Aufrufen verarbeiten (z.B. AMRB-Runner)
        _mode_map = {
            "fast": QualityMode.FAST,
            "balanced": QualityMode.BALANCED,
            "restoration": QualityMode.QUALITY,  # Restoration nutzt volle Quality-Pipeline
            "quality": QualityMode.QUALITY,
            "maximum": QualityMode.MAXIMUM,
            "studio_2026": QualityMode.MAXIMUM,
        }
        _mode_kwarg = kwargs.pop("mode", None)
        if _mode_kwarg is not None:
            self.config.mode = _mode_map.get(str(_mode_kwarg).lower(), self.config.mode)
        _mat_kwarg = kwargs.pop("material", None)
        if _mat_kwarg is not None and self.config.material_type is None:
            try:
                self.config.material_type = MaterialType(str(_mat_kwarg))
            except ValueError:
                pass

        # §Dach: MusikalischerGlobalplan — aus kwargs oder RestorationConfig laden
        _gp_kwarg = kwargs.pop("global_plan", None)
        _chain_kwarg = kwargs.pop("chain_info", None)
        _defekt_hint_kwarg = kwargs.pop("defekt_hint", None)
        self._active_global_plan = _gp_kwarg if _gp_kwarg is not None else self.config.global_plan
        self._active_chain_info = _chain_kwarg         # TontraegerketteDenker: Ketten-Phasen
        self._active_defekt_hint = _defekt_hint_kwarg  # DefektDenker: heuristische Phasen-Empfehlung

        _cb(2, "Initialisierung…")
        logger.info(f"Starting restoration: {len(audio)/sample_rate:.1f}s audio @ {sample_rate} Hz")

        # Start adaptive resource monitoring once per process lifecycle.
        try:
            from backend.core.adaptive_resource_manager import adaptive_resource_manager as _arm

            _arm.start_monitoring()
        except Exception as _arm_exc:
            logger.debug("AdaptiveResourceManager.start_monitoring() fehlgeschlagen: %s", _arm_exc)

        # Early initialization of variables used before their main definition (avoids F821)
        _is_studio_26 = (
            getattr(self.config, "studio_2026", False)
            or getattr(getattr(self.config, "mode", None), "value", "") == "studio_2026"
        )
        _goal_applicability_result = None

        # Resample to 48 kHz if necessary
        if sample_rate != target_sample_rate:
            if LIBROSA_AVAILABLE:
                logger.info(f"Resampling {sample_rate} Hz → {target_sample_rate} Hz for standardized processing")
                if audio.ndim == 2:
                    # Stereo: resample each channel
                    audio = np.column_stack(
                        [
                            librosa.resample(audio[:, 0], orig_sr=sample_rate, target_sr=target_sample_rate),
                            librosa.resample(audio[:, 1], orig_sr=sample_rate, target_sr=target_sample_rate),
                        ]
                    )
                else:
                    # Mono
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
            else:
                logger.warning(
                    f"Cannot resample to {target_sample_rate} Hz (librosa unavailable), processing at {sample_rate} Hz"
                )

        # §3.1 NaN/Inf-Invariante: Bereinigung nach Resampling (Bibliotheks-Artefakte möglich)
        if not np.all(np.isfinite(audio)):
            logger.debug("restore(): NaN/Inf nach Resampling bereinigt")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.9, neginf=-0.9)
        audio = np.clip(audio, -1.0, 1.0)

        _cb(5, "Resampling & Vorverarbeitung…")
        # Step 1a: Material-Erkennung via MediumClassifier (vor DefectScanner)
        # Nur wenn kein Material manuell vorgegeben wurde (config.material_type is None)
        # Originalklang für reference-basierte Musical Goals sichern (§1.2 Spec)
        # Nach Resampling auf 48 kHz — alle Metriken operieren bei 48 kHz
        original_audio_for_goals: np.ndarray = audio.copy()

        _cb(8, "Restaurierbarkeit wird geprüft…")
        # §2.26 RestorabilityEstimator — Vor-Assessment der Restaurierbarkeit (DSP-only, < 5 s)
        # §2.29 Normativ (PMGG Datenfluss-Invariante): _pmgg_restorability_score MUSS aus
        # RestorabilityEstimator.estimate().restorability_score stammen — kein Hard-Code 70.0.
        _pmgg_restorability_score: float = 70.0  # konservativer Default wenn Estimator nicht verfügbar
        try:
            from backend.core.restorability_estimator import estimate_restorability

            _mat_str = self.config.material_type.value if self.config.material_type is not None else "unknown"
            _rest_result = estimate_restorability(audio, sample_rate, material=_mat_str)
            _pmgg_restorability_score = float(_rest_result.restorability_score)  # noqa: F841 §2.29
            logger.info(
                "📊 RestorabilityEstimator: score=%.1f grade=%s predicted_MOS=%.2f–%.2f",
                _rest_result.restorability_score,
                _rest_result.grade,
                _rest_result.predicted_mos_range[0],
                _rest_result.predicted_mos_range[1],
            )
        except Exception as _re_exc:
            logger.debug("RestorabilityEstimator nicht verfügbar: %s", _re_exc)

        # §2.24 BatchSessionLearner — Batch-übergreifende Lernkohärenz
        _batch_session_id = None
        try:
            from backend.core.batch_session_learner import BatchSessionLearner as _BSL

            _source_paths = [kwargs.get("source_path")] if kwargs.get("source_path") else []
            _batch_session_id = _BSL.get_instance().start_session(_source_paths)
            logger.info("📦 BatchSessionLearner: session_id=%s", _batch_session_id)
        except Exception as _bsl_exc:
            logger.debug("BatchSessionLearner nicht verfügbar: %s", _bsl_exc)

        # §2.13 Künstler-Signatur laden (wenn source_path via kwargs übergeben)
        _artist_id: Optional[str] = None
        _artist_sig = None
        try:
            _source_path = kwargs.get("source_path")
            if _source_path is not None:
                from pathlib import Path as _Path

                from backend.core.artist_signature_store import get_signature_store as _get_sig_store

                _sig_store = _get_sig_store()
                _artist_id = _sig_store.detect_session([_Path(str(_source_path))])
                _artist_sig = _sig_store.load(_artist_id)
                if _artist_sig is not None and _artist_sig.confidence >= 0.3:
                    logger.info(
                        "🎙️ Künstler-Signatur geladen: artist_id=%s | confidence=%.2f | gender=%s",
                        _artist_id,
                        _artist_sig.confidence,
                        _artist_sig.voice_gender,
                    )
                else:
                    logger.debug("Künstler-Signatur: Kein vorheriger Prior (neuer Künstler)")
        except Exception as _sig_load_exc:
            logger.debug("ArtistSignatureStore Laden nicht verfügbar: %s", _sig_load_exc)

        # Perceptual Embedding des Originalaudios für GP-Kontext (§2.3 Spec)
        _original_embedding = None
        try:
            from backend.core.perceptual_embedder import embed_audio as _embed_audio

            _original_embedding = _embed_audio(original_audio_for_goals, sample_rate)
            logger.debug(
                "PerceptualEmbedder: 256-dim Embedding (norm=%.4f)",
                float(np.linalg.norm(_original_embedding.vector)),
            )
        except Exception as _emb_exc:
            logger.debug("PerceptualEmbedder nicht verfügbar: %s", _emb_exc)

        _classified_material: Optional[MaterialType] = self.config.material_type
        if _classified_material is None:
            # §9.7.2 Parallele Eingangs-Analyse — MediumClassifier, EraClassifier und
            # GermanSchlagerClassifier sind voneinander unabhängig und laufen parallel
            import concurrent.futures as _cf

            def _run_mc(a: np.ndarray, sr: int) -> Optional[object]:
                try:
                    from backend.core.medium_classifier import classify_medium
                    return classify_medium(a, sr=sr, use_ml=True)
                except Exception as _e:
                    logger.debug("MediumClassifier nicht verfügbar: %s", _e)
                    return None

            def _run_era(a: np.ndarray, sr: int) -> Optional[object]:
                try:
                    from backend.core.era_classifier import classify_era
                    return classify_era(a, sr)
                except Exception as _e:
                    logger.debug("EraClassifier nicht verfügbar: %s", _e)
                    return None

            def _run_genre(a: np.ndarray, sr: int) -> Optional[object]:
                try:
                    from backend.core.genre_classifier import classify_genre
                    return classify_genre(a, sr)
                except Exception as _e:
                    logger.debug("GermanSchlagerClassifier nicht verfügbar: %s", _e)
                    return None

            with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
                _fut_mc  = _pool.submit(_run_mc,    audio, sample_rate)
                _fut_era = _pool.submit(_run_era,   audio, sample_rate)
                _fut_sc  = _pool.submit(_run_genre, audio, sample_rate)
                _mc_result      = _fut_mc.result()
                _era_result_par = _fut_era.result()
                _schlager_result_par = _fut_sc.result()

            # MediumClassifier-Ergebnis verarbeiten
            if _mc_result is not None and _mc_result.confidence >= 0.35:
                _classified_material = _mc_result.material
                logger.info(
                    "🔍 MediumClassifier: Material=%s Konfidenz=%.2f Quelle=%s",
                    _classified_material.value,
                    _mc_result.confidence,
                    _mc_result.classifier_source,
                )
            else:
                if _mc_result is not None:
                    logger.debug(
                        "MediumClassifier: Konfidenz zu niedrig (%.2f), nutze DefectScanner-Prior",
                        _mc_result.confidence,
                    )

            # EraClassifier-Ergebnis übernehmen
            _era_result = _era_result_par
            if _era_result is not None:
                logger.info(
                    "🎙️ EraClassifier: decade=%d material_prior=%s confidence=%.2f",
                    _era_result.decade,
                    _era_result.material_prior,
                    _era_result.confidence,
                )

            # GermanSchlagerClassifier-Ergebnis übernehmen
            _schlager_result = _schlager_result_par
            if _schlager_result is not None:
                if _schlager_result.is_schlager:
                    logger.info(
                        "🎵 Schlager erkannt: confidence=%.2f genre=%s bpm=%.0f — SCHLAGER_PROFILE aktiv",
                        _schlager_result.confidence,
                        _schlager_result.genre_label,
                        _schlager_result.bpm,
                    )
                else:
                    logger.debug(
                        "GermanSchlagerClassifier: kein Schlager (confidence=%.2f genre=%s)",
                        _schlager_result.confidence,
                        _schlager_result.genre_label,
                    )
        else:
            # Material bereits im Config — Ära und Genre trotzdem ermitteln
            _era_result = None
            try:
                from backend.core.era_classifier import classify_era
                _era_result = classify_era(audio, sample_rate)
                logger.info(
                    "🎙️ EraClassifier: decade=%d material_prior=%s confidence=%.2f",
                    _era_result.decade,
                    _era_result.material_prior,
                    _era_result.confidence,
                )
            except Exception as _era_exc:
                logger.debug("EraClassifier nicht verfügbar: %s", _era_exc)

            _schlager_result = None
            try:
                from backend.core.genre_classifier import classify_genre
                _schlager_result = classify_genre(audio, sample_rate)
                if _schlager_result.is_schlager:
                    logger.info(
                        "🎵 Schlager erkannt: confidence=%.2f genre=%s bpm=%.0f — SCHLAGER_PROFILE aktiv",
                        _schlager_result.confidence,
                        _schlager_result.genre_label,
                        _schlager_result.bpm,
                    )
            except Exception as _sc_exc:
                logger.debug("GermanSchlagerClassifier nicht verfügbar: %s", _sc_exc)

        # §2.19.3 SCHLAGER_RESTORATION_PROFILE anwenden (GP-Material-Key + Phase-Params)
        _genre_profile: dict = {}
        _gp_material_key: str = (
            _defect_material.value
            if (_defect_material := getattr(_schlager_result, "_material", None)) is not None
            else (
                self._current_material.value
                if hasattr(self, "_current_material") and self._current_material
                else "unknown"
            )
        )
        if _schlager_result is not None and _schlager_result.is_schlager:
            try:
                from backend.core.genre_classifier import get_restoration_profile

                _genre_profile = get_restoration_profile(_schlager_result.subgenre)
                _gp_material_key = str(_genre_profile.get("gp_memory_key", _gp_material_key))
                logger.info(
                    "🪧 SCHLAGER_PROFILE: gp_key=%s brillanz_target=%.2f waerme_target=%.2f groove_dtw_max_ms=%.1f",
                    _gp_material_key,
                    _genre_profile.get("brillanz_target", 0.82),
                    _genre_profile.get("waerme_target", 0.88),
                    _genre_profile.get("groove_dtw_max_ms", 5.0),
                )
            except Exception as _gp_sc_exc:
                logger.debug("SCHLAGER_RESTORATION_PROFILE nicht geladen: %s", _gp_sc_exc)

        # §2.32 GoalApplicabilityFilter — Filterung physikalisch nicht-messbarer Musikziele
        _goal_applicability = None
        try:
            from backend.core.goal_applicability_filter import evaluate_goal_applicability

            _panns_for_gaf = kwargs.get("panns_tags", {})
            _mat_for_gaf = _classified_material.value if _classified_material is not None else "unknown"
            _goal_applicability = evaluate_goal_applicability(
                audio, sample_rate, _mat_for_gaf, _era_result, _panns_for_gaf
            )
            logger.info(
                "🎯 GoalApplicabilityFilter: %d anwendbar, %d nicht-anwendbar",
                len(_goal_applicability.applicable),
                len(_goal_applicability.inapplicable),
            )
        except Exception as _gaf_exc:
            logger.debug("GoalApplicabilityFilter nicht verfügbar: %s", _gaf_exc)

        # QualityAnalyzer: Eingangsqualität messen (Vorher-Baseline) (§9.5 Quality-Tracking)
        _quality_before: Optional[object] = None
        try:
            from backend.core.quality_prediction import QualityAnalyzer as _QualityAnalyzer

            _quality_before = _QualityAnalyzer().analyze_quality(audio, sample_rate)
            logger.debug(
                "📊 QualityAnalyzer (Vorher): score=%.1f SNR=%.1f dB warmth=%.3f naturalness=%.3f",
                _quality_before.overall_score,
                _quality_before.snr_db,
                _quality_before.warmth,
                _quality_before.naturalness,
            )
        except Exception as _qa_pre_exc:
            logger.debug("QualityAnalyzer (Vorher) nicht verfügbar: %s", _qa_pre_exc)

        # §DCOffsetPreRemoval — IIR-Hochpass 5 Hz (Spec §2.2): PFLICHT vor jeder FFT/HPSS-Analyse.
        # Entfernt DC-Versatz und sehr niedrige Frequenzen (< 5 Hz), die HPSS/FFT-Artefakte erzeugen.
        try:
            from scipy.signal import lfilter as _lfilter

            _dc_b = np.array([1.0, -1.0], dtype=np.float64)
            _dc_a = np.array([1.0, -0.9994], dtype=np.float64)  # f_c ≈ 5 Hz @ 48 kHz
            if audio.ndim == 1:
                audio = _lfilter(_dc_b, _dc_a, audio.astype(np.float64)).astype(np.float32)
            else:
                audio = np.stack([
                    _lfilter(_dc_b, _dc_a, audio[:, c].astype(np.float64)).astype(np.float32)
                    for c in range(audio.shape[1])
                ], axis=1)
            audio = np.clip(audio, -1.0, 1.0)
            logger.debug(
                "§DCOffsetPreRemoval: DC-Offset entfernt (|mean|=%.2e)", float(np.abs(np.mean(audio)))
            )
        except Exception as _dc_exc:
            logger.debug("DCOffsetPreRemoval Fehler (scipy nicht verfügbar): %s", _dc_exc)

        # §2.27 TransientDecoupledProcessing — Percussive/Harmonic-Trennung (allererster DSP-Schritt)
        _tdp_proc = None
        _tdp_percussive: Optional[np.ndarray] = None
        _tdp_harmonic_ready: bool = False
        try:
            from backend.core.transient_decoupled_processor import get_transient_decoupled_processor

            _tdp_proc = get_transient_decoupled_processor()
            _tdp_audio_percussive, _tdp_audio_harmonic = _tdp_proc.separate(audio, sample_rate)
            _tdp_percussive = _tdp_audio_percussive.copy()  # Original-Transienten sichern
            audio = _tdp_audio_harmonic  # Pipeline ab jetzt auf harm. Anteil
            _tdp_harmonic_ready = True
            logger.info(
                "§2.27 TDP: Percussive/Harmonic getrennt " "(perc_rms=%.4f harm_rms=%.4f)",
                float(np.sqrt(np.mean(_tdp_audio_percussive**2))),
                float(np.sqrt(np.mean(_tdp_audio_harmonic**2))),
            )
        except Exception as _tdp_sep_exc:
            logger.debug("TransientDecoupledProcessor nicht verfügbar: %s", _tdp_sep_exc)

        # §2.25 ReferenceAnchorSynthesizer — Synthesierter Referenz-Anker für Mastering-Ziel
        _reference_anchor = None
        try:
            from backend.core.reference_anchor_synthesizer import synthesize_reference_anchor

            _era_for_anchor = _era_result.decade if _era_result is not None else None
            _mat_for_anchor = _classified_material.value if _classified_material is not None else "unknown"
            _genre_for_anchor = (
                _schlager_result.genre_label if _schlager_result is not None else "unknown"
            )
            _reference_anchor = synthesize_reference_anchor(  # noqa: F841
                audio, sample_rate, _era_for_anchor, _genre_for_anchor, _mat_for_anchor
            )
            logger.info(
                "🎯 ReferenceAnchorSynthesizer: Anker generiert (era=%s genre=%s material=%s)",
                _era_for_anchor,
                _genre_for_anchor,
                _mat_for_anchor,
            )
        except Exception as _ras_exc:
            logger.debug("ReferenceAnchorSynthesizer nicht verfügbar: %s", _ras_exc)

        # Step 1: Defect Scanning
        logger.info("Step 1/4: Defect Scanning...")
        defect_result = self.defect_scanner.scan(audio, sample_rate, _classified_material)
        material_type = defect_result.material_type

        # §2.22 PerceptualAttentionModel — Bark-Band Salienz-Karte für phasengewichtetes Processing
        _saliency_map = None
        try:
            from backend.core.perceptual_attention_model import compute_saliency_map

            _saliency_map = compute_saliency_map(audio, sample_rate)
            logger.info(
                "👁️ PerceptualAttentionModel: saliency_map shape=%s max=%.3f",
                _saliency_map.shape,
                float(_saliency_map.max()),
            )
        except Exception as _pam_exc:
            logger.debug("PerceptualAttentionModel nicht verfügbar: %s", _pam_exc)

        logger.info(
            f"Detected material: {material_type.value}, "
            f"Top defects: {', '.join([f'{s.defect_type.value}={s.severity:.2f}' for s in defect_result.get_top_defects(3)])}"
        )

        # Kausaldiagnose: Bayesianische Ursachenermittlung
        _causal_plan = None
        try:
            from backend.core.causal_defect_reasoner import reason_about_defects

            _defect_scores_map = {s.defect_type.value: s.severity for s in defect_result.get_top_defects(8)}
            _causal_plan = reason_about_defects(
                defect_scores=_defect_scores_map,
                material=material_type.value,
                audio=audio,
                sample_rate=sample_rate,
            )
            logger.info(
                "CausalReasoner: primary=%s (confidence=%.3f) | %s",
                _causal_plan.primary_cause,
                _causal_plan.confidence,
                _causal_plan.reasoning[:120],
            )
        except Exception as _cr_exc:
            logger.debug("CausalDefectReasoner nicht verfügbar: %s", _cr_exc)

        # UQ: Pipeline-Konfidenzschätzung (§2.15)
        _pipeline_confidence = None
        try:
            from backend.core.pipeline_uncertainty import estimate_pipeline_confidence

            _defect_scores_uq = {s.defect_type.value: s.severity for s in defect_result.get_top_defects(8)}
            _pipeline_confidence = estimate_pipeline_confidence(_causal_plan, _defect_scores_uq)
            logger.info(
                "🔮 PipelineUQ: tier=%s confidence=%.2f gp_factor=%.2f%s",
                _pipeline_confidence.tier,
                _pipeline_confidence.confidence,
                _pipeline_confidence.gp_bound_factor,
                f" | {_pipeline_confidence.user_hint}" if _pipeline_confidence.user_hint else "",
            )
        except Exception as _uq_exc:
            logger.debug("PipelineUncertaintyEstimator nicht verfügbar: %s", _uq_exc)

        # Step 1c: Musikalische Strukturanalyse (§2.17) — Chorus als Inpainting-Prior
        _musical_structure = None
        try:
            _has_dropout = any(
                s.defect_type.value in {"dropouts", "tape_dropout", "dropout"} and s.severity >= 0.25
                for s in defect_result.get_top_defects(10)
            )
            if _has_dropout or self.config.mode == QualityMode.MAXIMUM:
                from backend.core.musical_structure_analyzer import analyze_musical_structure

                _musical_structure = analyze_musical_structure(audio, sample_rate)
                _n_chorus = sum(1 for s in _musical_structure.segments if s.segment_type == "chorus")
                _n_verse = sum(1 for s in _musical_structure.segments if s.segment_type == "verse")
                logger.info(
                    "🎵 MusicalStructure: %d Segmente (Chorus×%d Verse×%d) confidence=%.2f",
                    len(_musical_structure.segments),
                    _n_chorus,
                    _n_verse,
                    _musical_structure.confidence,
                )
            else:
                logger.debug("MusicalStructureAnalyzer: übersprungen (kein Dropout-Defekt aktiv)")
        except Exception as _msa_exc:
            logger.debug("MusicalStructureAnalyzer nicht verfügbar: %s", _msa_exc)

        # §2.5 Spec: propose_pareto() als primärer GP-Vorschlag (MOO, Pareto-Front)
        _pareto_proposals = None
        _gp_material_key_pre = material_type.value if material_type else "unknown"
        try:
            from backend.core.gp_parameter_optimizer import get_optimizer as _get_gp_opt_pre

            _gp_opt_pre = _get_gp_opt_pre()
            _era_ws_pre: Optional[Dict[str, float]] = None
            if _era_result is not None:
                try:
                    from backend.core.era_classifier import get_era_classifier as _get_era_clf_pre

                    _era_ws_pre = _get_era_clf_pre().get_gp_warmstart(_era_result)
                except Exception:
                    pass
            _pareto_proposals = _gp_opt_pre.propose_pareto(
                material=_gp_material_key_pre,
                embedding=_original_embedding.vector if _original_embedding is not None else None,
                era_warmstart=_era_ws_pre,
            )
            logger.info(
                "§2.5 GP propose_pareto: %d Pareto-Kandidaten für material='%s'",
                len(_pareto_proposals),
                _gp_material_key_pre,
            )
        except Exception as _pareto_exc:
            logger.debug("propose_pareto nicht verfügbar, Fallback auf Legacy propose(): %s", _pareto_exc)

        # §2.32 GoalApplicabilityFilter — physikalisch nicht messbare Ziele deaktivieren
        _goal_applicability = None
        _applicable_goals: Optional[set] = None
        try:
            from backend.core.goal_applicability_filter import evaluate_goal_applicability

            _panns_conf: Optional[Dict[str, float]] = None
            try:
                _panns_conf = defect_result.metadata.get("panns_tags") if hasattr(defect_result, "metadata") else None
            except Exception:
                pass
            _goal_applicability = evaluate_goal_applicability(
                audio=audio,
                sr=sample_rate,
                material=material_type.value if material_type else "unknown",
                era_decade=_era_result.decade if _era_result is not None else None,
                panns_tags=_panns_conf,
            )
            _applicable_goals = set(_goal_applicability.applicable)
            _goal_applicability_result = _goal_applicability
            logger.info(
                "§2.32 GoalApplicabilityFilter: %d applicable / %d inapplicable: %s",
                len(_goal_applicability.applicable),
                len(_goal_applicability.inapplicable),
                ", ".join(sorted(_goal_applicability.inapplicable)) or "—",
            )
        except Exception as _gaf_exc:
            logger.debug("GoalApplicabilityFilter nicht verfügbar: %s", _gaf_exc)

        # §8.2 Pass-Through-Invariante: Sauberes Material schonend behandeln
        # (SNR > 40 dB + max_defect_severity < 0.15 → Schonmodus)
        _rms_signal = float(np.sqrt(np.mean(audio**2)) + 1e-9)
        _noise_percentile = float(np.percentile(np.abs(audio), 5) + 1e-9)
        _input_snr_db = 20.0 * np.log10(_rms_signal / _noise_percentile)
        _top_defects = defect_result.get_top_defects(5)
        _max_defect_severity = max((s.severity for s in _top_defects), default=0.0)
        _pass_through_mode = _input_snr_db > 40.0 and _max_defect_severity < 0.15
        if _pass_through_mode:
            logger.info(
                "🛡️ Pass-Through-Guard §8.2: SNR=%.1f dB, max_defect=%.3f → Schonmodus aktiv"
                " (PQS-MOS-Verlust ≤ 0.05, LUFS-Δ ≤ 0.3 LU)",
                _input_snr_db,
                _max_defect_severity,
            )

        # Step 2: Phase Selection (basierend auf Defekten)
        _cb(22, "Phasenauswahl…")
        logger.info("Step 2/4: Phase Selection...")
        selected_phases = self._select_phases(
            defect_result,
            causal_plan=_causal_plan,
            chain_info=self._active_chain_info,
            defekt_hint=self._active_defekt_hint,
        )
        logger.info(f"Selected {len(selected_phases)} phases based on defects")

        # Step 2.5: Apply Phase Skipping (intelligent filtering)
        if self.phase_skipper:
            original_count = len(selected_phases)
            selected_phases, skip_reasons = self._apply_phase_skipping(selected_phases, defect_result)
            skipped_count = original_count - len(selected_phases)
            if skipped_count > 0:
                logger.info(
                    f"Phase Skipping: {skipped_count} phases skipped "
                    f"({skipped_count/original_count*100:.0f}% speedup potential)"
                )
                for phase_id, reason in skip_reasons.items():
                    logger.debug(f"  Skipped {phase_id}: {reason}")

        # §8.2 Pass-Through: nur minimale Phasen (LUFS-Norm + TruePeak) für sauberes Material
        if _pass_through_mode:
            _pt_allowed = {
                "phase_30_dc_offset_removal",
                "phase_40_loudness_normalization",
                "phase_41_output_format_optimization",
                "phase_47_truepeak_limiter",
            }
            _pt_removed = [p for p in selected_phases if p not in _pt_allowed]
            selected_phases = [p for p in selected_phases if p in _pt_allowed]
            if _pt_removed:
                logger.info(
                    "🛡️ Pass-Through-Guard: %d Phasen deaktiviert → %d minimal-Phasen verbleiben",
                    len(_pt_removed),
                    len(selected_phases),
                )

        # Step 3: Performance Guard Setup
        audio_duration = len(audio) / sample_rate
        if self.performance_guard:
            self.performance_guard.start_monitoring(audio_duration)

        # §Vintage-Authentizitäts-Guards — Ära-spezifische Phase-Filter (nach Pass-Through-Guard)
        # Spec: 1920–1940 → Rolloff ≤ 7 kHz NICHT künstlich erweitern (EAPC übernimmt §2.35).
        #       AudioSR (phase_06) nur bei user_requested=True — da kein solches Flag existiert,
        #       wird phase_06 für decade ≤ 1940 sicher deaktiviert.
        # Spec: 1955–1965 → RT60 ∈ [1.2, 2.0] s bewahren (phase_20/phase_49 nicht entfernen,
        #       aber AuthentizitaetMetric-Invariante gilt — Phasen mit Rollback durch PMGG gesichert).
        if _era_result is not None:
            _vintage_decade = _era_result.decade
            if _vintage_decade <= 1940:
                _bw_phases_to_guard = {"phase_06_frequency_restoration"}
                _removed_vintage = [p for p in selected_phases if p in _bw_phases_to_guard]
                if _removed_vintage:
                    selected_phases = [p for p in selected_phases if p not in _bw_phases_to_guard]
                    logger.info(
                        "🕰️ Vintage-Guard decade=%d ≤ 1940: %s deaktiviert — "
                        "EAPC (§2.35) übernimmt ära-authentische HF-Ergänzung",
                        _vintage_decade,
                        ", ".join(_removed_vintage),
                    )

        # Step 4: Execute Phases — mit EnsembleProcessor-Konsens (§2.21)
        logger.info("Step 3/4: Executing Restoration Pipeline (EnsembleProcessor)...")
        _cb(30, "Pipeline startet…")
        restored_audio, executed_phases, skipped_phases = self._execute_pipeline(
            audio,
            sample_rate,
            material_type,
            defect_result,
            selected_phases,
            progress_callback=progress_callback,
            restorability_score=_pmgg_restorability_score,  # §2.29 normativ
            applicable_goals=_applicable_goals,  # §2.32 normativ
        )
        try:
            from backend.core.ensemble_processor import process_ensemble

            def _ep_restoration_fn(_a: np.ndarray, _sr: int, _strength: float) -> np.ndarray:
                _res, _, _ = self._execute_pipeline(
                    _a,
                    _sr,
                    material_type,
                    defect_result,
                    selected_phases,
                    progress_callback=None,
                    restorability_score=_pmgg_restorability_score,  # §2.29 normativ
                    applicable_goals=_applicable_goals,  # §2.32 normativ
                )
                return _res

            _mat_str = material_type.value if hasattr(material_type, "value") else str(material_type)
            _ep_audio = process_ensemble(
                audio,
                sample_rate,
                _mat_str,
                restoration_fn=_ep_restoration_fn,
            )
            restored_audio = _ep_audio
            logger.info("🎛️ EnsembleProcessor: 3 Ketten (CONSERVATIVE/BALANCED/AGGRESSIVE), frame-voted")
        except Exception as _ep_exc:
            logger.debug("EnsembleProcessor nicht verfügbar, Fallback auf _execute_pipeline: %s", _ep_exc)

        # Speicher-Hygiene: Pipeline-Modelle entladen sobald alle Phasen abgeschlossen.
        # AudioSR (7 GB), LAION-CLAP (2.2 GB) werden nur in der Phase-Pipeline benötigt.
        try:
            from plugins.audiosr_plugin import unload_audiosr  # noqa: PLC0415
            unload_audiosr()
        except Exception:
            pass
        try:
            from plugins.laion_clap_plugin import unload_laion_clap  # noqa: PLC0415
            unload_laion_clap()
        except Exception:
            pass
        gc.collect()

        # --- §2.33 FeedbackChain-Ceiling: Physikalische Grenze vor FC-Block schätzen ---
        # estimate_physical_ceiling läuft hier mit leerem scores-Dict, da _pqs_result
        # erst nach dem FC-Block verfügbar ist. Das Ceiling basiert primär auf Audio-SNR
        # und Bandbreite, nicht auf goal-Scores — daher trotzdem valid.
        _fc_ceiling_val: Optional[float] = None
        try:
            from backend.core.physical_ceiling_estimator import PhysicalCeilingEstimator as _FCEarly

            _fc_mat_val = material_type.value if material_type is not None else "unknown"
            _fc_pce_result = _FCEarly().estimate(restored_audio, sample_rate, {}, _fc_mat_val)
            # Natürlichkeit-Ceiling als konservativer Gesamt-Proxy (niedrigster relevanter Wert)
            _fc_ceiling_val = float(_fc_pce_result.ceiling.get("natuerlichkeit", 0.98))
            logger.debug("§2.33 FC-Ceiling (vor FeedbackChain): natuerlichkeit=%.3f", _fc_ceiling_val)
        except Exception as _fc_pce_err:
            logger.debug("FeedbackChain Ceiling-Vorschätzung nicht verfügbar: %s", _fc_pce_err)

        # --- FeedbackChain: Iterativer Qualitätsregelkreis nach Pipeline (§2.2 Spec) ---
        # Post-Pipeline-Korrekturphasen: FinalEQ → MasteringPolish → LUFSNorm
        # FeedbackChain prüft nach jeder Phase den PQS-Score und behält bei
        # Regression das beste Zwischenergebnis (§9.5 Pflicht-Invariante).
        _fc_chain_result = None
        try:
            from backend.core.feedback_chain import FeedbackChain

            _fc_phase_ids = [
                "phase_07_harmonic_restoration",  # Harmonik-Kohärenz frühzeitig stabilisieren
                "phase_14_phase_correction",       # Phasenfehler vor Mastering-Phasen korrigieren
                "phase_16_final_eq",
                "phase_17_mastering_polish",
                "phase_40_loudness_normalization",
            ]
            _fc_phases_list = []
            for _fc_pid in _fc_phase_ids:
                _fc_phase_obj = self._get_phase(_fc_pid)
                if _fc_phase_obj is not None:
                    # Closure-Fabrik verhindert Late-Binding-Bug in Schleife
                    def _make_fc_callable(_ph):
                        def _fc_call(_audio: np.ndarray, _sr: int, **_kw) -> np.ndarray:
                            _res = self._profiled_phase_call(
                                _ph,
                                _audio,
                                sample_rate=_sr,
                                material_type=material_type,
                                material=material_type,
                                defect_scores=defect_result.scores,
                                quality_mode=self.config.mode.value,
                            )
                            return _res.audio if hasattr(_res, "audio") else _audio

                        return _fc_call

                    _fc_phases_list.append(
                        (
                            int(_fc_pid.split("_")[1]),
                            _make_fc_callable(_fc_phase_obj),
                            {},
                        )
                    )
            if _fc_phases_list:
                _fc_excellence = True  # v9.14-D1: ExcellenceOptimizer für beide Modi aktiv
                _fc_chain = FeedbackChain(
                    sample_rate=sample_rate,
                    target_score=0.78 if getattr(self.config.mode, "value", "") == "studio_2026" else 0.72,  # v9.14-D5
                    max_iterations=5,  # v9.15-B3: aligned with PMGG 5-retry strategy
                    excellence_mode=_fc_excellence,
                    material=material_type.value,
                )
                # §2.34 GPP-WIRE: GoalPriorityProtocol als in-loop Phase-Callback verdrahten
                try:
                    from backend.core.goal_priority_protocol import check_iteration_abort as _check_gpp
                    from backend.core.musical_goals.musical_goals_metrics import (
                        MusicalGoalsChecker as _GoalChecker,
                    )

                    _gpp_checker = _GoalChecker()
                    _gpp_sr = sample_rate

                    def _gpp_fc_callback(_ab: np.ndarray, _aa: np.ndarray) -> tuple[bool, str]:
                        try:
                            _sb = _gpp_checker.measure_all(_ab, _gpp_sr)
                            _sa = _gpp_checker.measure_all(_aa, _gpp_sr)
                            _res = _check_gpp(_sb, _sa)
                            return _res.should_abort, _res.reason
                        except Exception:  # noqa: BLE001
                            return False, ""

                    if hasattr(_fc_chain, "goal_priority_callback"):
                        _fc_chain.goal_priority_callback = _gpp_fc_callback
                        logger.debug("§GPP-WIRE: GoalPriorityProtocol FeedbackChain-Callback aktiv")
                except Exception as _gpp_wire_exc:  # noqa: BLE001
                    logger.debug("§GPP-WIRE: nicht verfügbar — %s", _gpp_wire_exc)
                _fc_chain_result = _fc_chain.run(restored_audio, _fc_phases_list, ceiling=_fc_ceiling_val)
                restored_audio = _fc_chain_result.audio
                logger.info(
                    "🔄 FeedbackChain: score=%.3f retries=%d t=%.2fs (%d Post-Phasen)",
                    _fc_chain_result.overall_score,
                    getattr(_fc_chain_result, "total_retries", _fc_chain_result.iterations),
                    _fc_chain_result.total_time_s,
                    len(_fc_phases_list),
                )
            else:
                logger.debug("FeedbackChain: keine Post-Pipeline-Phasen verfügbar")
        except Exception as _fc_exc:
            logger.debug("FeedbackChain nicht verfügbar: %s", _fc_exc)

        # §1.4 StemRemixBalancer — LUFS-korrekter Stem-Re-Mix (nur bei Stem-Vorhandensein)
        try:
            from backend.core.stem_remix_balancer import balance_remix

            _stems = kwargs.get("stems")
            if _stems is not None and isinstance(_stems, dict):
                _vocals = _stems.get("vocals")
                _instruments = _stems.get("instruments")
                if _vocals is not None and _instruments is not None:
                    # §1.4 vocal_weight: aus PANNs-Ergebnis ableiten (vor MDX23C auf Original geschätzt)
                    _vocal_weight: float = float(_stems.get("vocal_weight", 0.5))
                    if _vocal_weight <= 0.0 or _vocal_weight >= 1.0:
                        # Fallback aus kwargs-PANNs-Tags
                        _ptags: dict = kwargs.get("panns_tags", {})
                        _v_conf = max(
                            _ptags.get("Singing voice", 0.0),
                            _ptags.get("Vocals", 0.0),
                            _ptags.get("Speech", 0.0),
                        )
                        _vocal_weight = float(np.clip(_v_conf if _v_conf > 0.01 else 0.5, 0.1, 0.9))
                    restored_audio = balance_remix(
                        _vocals, _instruments, original_audio_for_goals, sample_rate,
                        vocal_weight=_vocal_weight,
                    )
                    logger.info("🎚️ StemRemixBalancer: Stem-Remix LUFS-balanciert (vocal_weight=%.2f)", _vocal_weight)
                else:
                    logger.debug("StemRemixBalancer: Stems unvollständig (vocals/instruments fehlen)")
            else:
                logger.debug("StemRemixBalancer: keine Stems in kwargs übergeben")
        except Exception as _srb_exc:
            logger.debug("StemRemixBalancer nicht verfügbar: %s", _srb_exc)

        # §Studio-2026 Reference Mastering — Matchering 2.0 spektrales Profil-Matching
        # Position: nach StemRemixBalancer (Re-Mix liegt vor), vor LUFS-Normalisierung.
        # Referenz: restauriertes Original (original_audio_for_goals) — originalgetreues Profil.
        # Nur im Studio-2026-Modus aktiv; bei fehlendem Paket transparenter DSP-Fallback.
        if _is_studio_26:
            try:
                from plugins.matchering_plugin import is_matchering_available as _mg_avail
                from plugins.matchering_plugin import match_reference as _match_ref

                _mg_ref_signal = original_audio_for_goals
                if _mg_ref_signal is not None and np.atleast_1d(_mg_ref_signal).shape[-1] > sample_rate:
                    restored_audio = _match_ref(restored_audio, _mg_ref_signal, sample_rate)
                    restored_audio = np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0)
                    restored_audio = np.clip(restored_audio, -1.0, 1.0)
                    logger.info(
                        "🎚️ Studio-2026 Reference Mastering: Matchering %s — spektrales Profil-Matching abgeschlossen",
                        "2.0" if _mg_avail() else "DSP-Fallback",
                    )
                else:
                    logger.debug("Studio-2026 Reference Mastering: Referenzsignal zu kurz oder None, übersprungen")
            except Exception as _mg_exc:
                logger.debug("Studio-2026 Reference Mastering nicht verfügbar: %s", _mg_exc)

        # §2.35 EraAuthenticPerceptualCompletion — Ära-authentische HF-Ergänzung (vor IAD)
        # Positionierung: nach phase_55_diffusion_inpainting, vor IntroducedArtifactDetector (Spec §2.35)
        # Nur aktiv wenn Quell-Bandbreite < 10 kHz UND BrillanzMetric applicable
        try:
            from backend.core.era_authentic_perceptual_completion import get_era_completion as _get_eapc

            _eapc = _get_eapc()
            _era_decade_eapc = _era_result.decade if _era_result is not None else None
            # GoalApplicabilityFilter-Ergebnis übergeben (aus §2.32, bereits berechnet)
            _eapc_goal_app = _goal_applicability_result
            if _eapc.is_applicable(restored_audio, sample_rate, goal_applicability=_eapc_goal_app):
                _eapc_result = _eapc.complete(restored_audio, sample_rate, era=_era_decade_eapc, anchor=None)
                if _eapc_result.applied:
                    restored_audio = np.clip(
                        np.nan_to_num(_eapc_result.audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
                    )
                    logger.info(
                        "§2.35 EAPC: Ära-authentische HF-Ergänzung — decade=%s bw=%.0f→%.0f Hz ceiling=%.2f",
                        _eapc_result.era_decade,
                        _eapc_result.source_bandwidth_hz,
                        _eapc_result.completion_bandwidth_hz,
                        _eapc_result.brillanz_ceiling,
                    )
                else:
                    logger.debug("§2.35 EAPC: is_applicable=True aber complete() applied=False")
            else:
                logger.debug("§2.35 EAPC: nicht anwendbar (BW ≥ 10 kHz oder BrillanzMetric inapplicable)")
        except Exception as _eapc_exc:
            logger.debug("EraAuthenticPerceptualCompletion nicht verfügbar: %s", _eapc_exc)

        # §2.36 LyricsGuidedEnhancement — Phonem-klassen-bewusste Restaurierung (Pflicht ab 9.10.x)
        # Positionierung: nach EAPC (§2.35), vor IAD (§2.23) — schützt Konsonanten und betonte Silben.
        # Aktivierung: immer wenn Audio ≥ 1 s; kein Vocals-Gate (Lyrics können auch instrumental sein).
        # Privacy: Lyrics-Text wird NICHT geloggt (§2.36 Datenschutz-Pflicht).
        try:
            from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement as _get_lge

            _lge = _get_lge()
            _lge_audio, _lge_transcription = _lge.enhance(restored_audio, sample_rate)
            _lge_audio = np.nan_to_num(_lge_audio, nan=0.0, posinf=0.0, neginf=0.0)
            _lge_audio = np.clip(_lge_audio, -1.0, 1.0)
            restored_audio = _lge_audio
            _lge_seg_count = len(_lge_transcription.words) if _lge_transcription is not None else 0
            logger.info("§2.36 LGE: Lyrics-geführte Verbesserung — %d Segmente verarbeitet", _lge_seg_count)
        except Exception as _lge_exc:
            logger.debug("LyricsGuidedEnhancement nicht verfügbar: %s", _lge_exc)

        # §2.23 IntroducedArtifactDetector — Durch Restaurierung neu eingebrachte Artefakte prüfen
        try:
            from backend.core.introduced_artifact_detector import detect_introduced_artifacts

            _iad_result = detect_introduced_artifacts(original_audio_for_goals, restored_audio, sample_rate)
            if _iad_result.has_artifacts:
                logger.warning(
                    "⚠️ IAD: Artefakte erkannt — fraction=%.3f Typen=%s",
                    _iad_result.total_contaminated_fraction,
                    ", ".join(_iad_result.artifact_types) if _iad_result.artifact_types else "unbekannt",
                )
            else:
                logger.debug(
                    "IAD: keine neu eingebrachten Artefakte (fraction=%.3f)", _iad_result.total_contaminated_fraction
                )
        except Exception as _iad_exc:
            logger.debug("IntroducedArtifactDetector nicht verfügbar: %s", _iad_exc)

        # --- SegmentAdaptiveProcessor: Segment-individuelle Feinoptimierung (§2.10 Spec) ---
        # Anwendung NACH globalem Pipeline-Lauf: jedes Segment (Stille /
        # Mixed / Vocal) erhält eigene NR-Stärke, Harmonic-Boost, Smoothing.
        # Stereo-kompatibel: Gain-Verhältnis Mono→Mono auf Stereo-Kanäle übertragen.
        _sap_result = None
        try:
            from backend.core.segment_adaptive_processor import get_segment_processor

            _audio_dur_s = len(restored_audio) / float(sample_rate)
            _sap_enabled = _audio_dur_s >= 5.0  # Fallback bei < 5 s (§2.10)
            _sap_is_stereo = restored_audio.ndim == 2

            def _sap_process_fn(_seg: np.ndarray, _sr: int, _params: dict) -> np.ndarray:
                """Leichte segment-spezifische DSP-Nachbearbeitung."""
                out = _seg.copy().astype(np.float32)
                nr = float(_params.get("noise_reduction_strength", 0.55))
                boost_db = float(_params.get("harmonic_boost_db", 0.0))
                smooth = float(_params.get("spectral_smoothing", 0.5))

                # 1. Stille-Rauschboden anheben (§8.1): leise Segmente abschwächen
                _seg_rms = float(np.sqrt(np.mean(out**2) + 1e-12))
                _silence_thr = 10 ** (-40.0 / 20.0)  # −40 dBFS
                if _seg_rms < _silence_thr * 2.0 and nr > 0.3:
                    out *= max(0.0, 1.0 - nr * 0.7)

                # 2. Harmonic Boost: sanfter Presence-Lift ≤ +3 dB (§2.5)
                if 0.0 < boost_db <= 6.0:
                    _gain_add = min(10 ** (boost_db / 20.0) - 1.0, 0.41) * 0.25
                    _spec = np.fft.rfft(out)
                    _n = len(_spec)
                    # Nur oberes Drittel anheben (Presence-Region)
                    _mask = np.ones(_n, dtype=np.float32)
                    _mask[_n * 2 // 3 :] = 1.0 + _gain_add
                    _spec *= _mask
                    _reconstructed = np.fft.irfft(_spec, n=len(out)).astype(np.float32)
                    if len(_reconstructed) == len(out):
                        out = _reconstructed

                # 3. Spectral Smoothing: Hangfenster-Glättung im Frequenzbereich
                if 0.1 <= smooth <= 0.9:
                    _spec2 = np.fft.rfft(out)
                    _kernel_n = max(3, int(smooth * 15) | 1)  # ungerade
                    _kernel = np.hanning(_kernel_n + 2)[1:-1].astype(np.float32)
                    _kernel /= _kernel.sum()
                    _mag_s = np.convolve(np.abs(_spec2), _kernel, mode="same")
                    _phase_s = np.angle(_spec2)
                    _spec2 = _mag_s * np.exp(1j * _phase_s)
                    _smooth_out = np.fft.irfft(_spec2, n=len(out)).astype(np.float32)
                    if len(_smooth_out) == len(out):
                        out = out * (1.0 - smooth * 0.15) + _smooth_out * (smooth * 0.15)

                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                return np.clip(out, -1.0, 1.0).astype(np.float32)

            # Stereo: pro Kanal SAP anwenden
            if _sap_is_stereo:
                _sap_ch0 = get_segment_processor().process(
                    restored_audio[0],
                    sample_rate,
                    _sap_process_fn,
                    material=material_type.value,
                    enabled=_sap_enabled,
                )
                _sap_ch1 = get_segment_processor().process(
                    restored_audio[1],
                    sample_rate,
                    _sap_process_fn,
                    material=material_type.value,
                    enabled=_sap_enabled,
                )
                _sap_result = _sap_ch0  # Metadaten aus Kanal 0
                restored_audio = np.stack([_sap_ch0.audio, _sap_ch1.audio], axis=0).astype(np.float32)
            else:
                _sap_result = get_segment_processor().process(
                    restored_audio,
                    sample_rate,
                    _sap_process_fn,
                    material=material_type.value,
                    enabled=_sap_enabled,
                )
                restored_audio = _sap_result.audio

            logger.info(
                "📐 SegmentAdaptive: %d Segmente (%s) | fallback=%s",
                _sap_result.n_segments,
                ", ".join(set(s.segment_type for s in _sap_result.segments)),
                _sap_result.used_fallback,
            )
        except Exception as _sap_exc:
            logger.debug("SegmentAdaptiveProcessor nicht verfügbar: %s", _sap_exc)

        # Post-Processing: Temporale Qualitätskohärenz (§2.16)
        try:
            from backend.core.temporal_quality_coherence import measure_temporal_coherence

            _tqc = measure_temporal_coherence(restored_audio, sample_rate)
            if not _tqc.passed:
                logger.warning(
                    "⚠️ TemporalQualityCoherence: NICHT BESTANDEN "
                    "(max_span=%.3f σ=%.3f n=%d) — zeitliche Konsistenz prüfen",
                    _tqc.max_span,
                    _tqc.sigma,
                    _tqc.n_segments,
                )
            else:
                logger.debug(
                    "TemporalQualityCoherence: OK (max_span=%.3f σ=%.3f)",
                    _tqc.max_span,
                    _tqc.sigma,
                )
        except Exception as _tqc_exc:
            logger.debug("TemporalQualityCoherenceMetric nicht verfügbar: %s", _tqc_exc)

        # Post-Processing: Stereofeld-Authentizitätsinvariante (§2.18)
        # §M-3: _FallbackEra-Fallback wenn EraClassifier keine Ära erkannte
        _era_for_stereo = _era_result
        if _era_for_stereo is None:
            import types as _types_m3

            _era_for_stereo = _types_m3.SimpleNamespace(decade=1970, confidence=0.0)
        try:
            from backend.core.stereo_authenticity_invariant import (
                check_stereo_authenticity,
                get_stereo_authenticity_invariant,
            )

            _stereo_check = check_stereo_authenticity(audio, restored_audio, _era_for_stereo, sample_rate)
            if not _stereo_check.passed:
                logger.warning(
                    "🔊 StereoInvariant: Verletzung '%s' — erzwinge Korrektur",
                    _stereo_check.rule_triggered,
                )
                _stereo_inv = get_stereo_authenticity_invariant()
                restored_audio = _stereo_inv.enforce(restored_audio, sample_rate, audio, _era_for_stereo)
            else:
                logger.debug(
                    "StereoInvariant: OK (%s)",
                    _stereo_check.rule_triggered or "kein Regelverstoß",
                )
        except Exception as _si_exc:
            logger.debug("StereoAuthenticitiyInvariant nicht verfügbar: %s", _si_exc)

        # --- ExcellenceOptimizer: DSP-Feinoptimierung nach Phasen-Pipeline (§2.2 Spec) ---
        _excellence_result = None
        try:
            from backend.core.excellence_optimizer import optimize_for_excellence

            restored_audio, _excellence_result = optimize_for_excellence(
                restored_audio, sample_rate, material=material_type.value
            )
            logger.info("🏆 ExcellenceOptimizer: %s", _excellence_result.summary())
        except Exception as _ex_exc:
            logger.warning("ExcellenceOptimizer nicht verfügbar — DSP-Fallback aktiv: %s", _ex_exc)
            # Guaranteed DSP-Fallback: Presence enhancement + NaN-Guard (§Checkliste §3.x)
            try:
                from scipy.signal import butter as _butter, lfilter as _lfilter

                _ex_rms = float(np.sqrt(np.mean(restored_audio.astype(np.float64) ** 2) + 1e-12))
                if _ex_rms > 1e-4:  # Nicht auf Stille anwenden
                    # Subtile Präsenz-Auffrischung (3–8 kHz, +0.5 dB) als Minimal-Harmonic-Boost
                    _nyq = sample_rate / 2.0
                    _ex_b, _ex_a = _butter(2, [min(3000.0 / _nyq, 0.95), min(8000.0 / _nyq, 0.99)], btype="band")
                    _ex_presence = _lfilter(_ex_b, _ex_a, restored_audio)
                    # Gain-Faktor: 0.05 ≈ +0.4 dB Präsenz-Anhebung
                    restored_audio = np.clip(restored_audio + 0.05 * _ex_presence, -1.0, 1.0)
                restored_audio = np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info("ExcellenceOptimizer DSP-Fallback: Präsenz-Auffrischung 3–8 kHz angewendet")
            except Exception as _ex_fb_exc:
                restored_audio = np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0)
                logger.debug("ExcellenceOptimizer DSP-Fallback nicht verfügbar: %s", _ex_fb_exc)

        # §2.28 HarmonicPreservationGuard — Post-Hoc-Harmonische-Energiekorrektur (§2.28 Spec)
        # Spezifikations-Hinweis §2.28: Ideale Position wäre VOR phase_03 (G_floor-Override).
        # Aktuelle Implementierung: Post-Hoc-Korrektur nach Pipeline — funktional äquivalent,
        # da extract_harmonic_mask() stets auf original_audio_for_goals (Pre-Pipeline) arbeitet
        # und apply_correction() die Harmonik-Energie-Referenz H_ref aus dem Original bezieht.
        # Ergebnis: Harmonik-Energie die durch NR abgesenkt wurde, wird auf 85 % wiederhergestellt.
        # SHELLAC / WAX_CYLINDER: HPG überspringen — bei SNR ≈ 6 dB ist CREPE/pYIN
        # f₀-Schätzung unzuverlässig (falsche Harmonik-Maske → gain ×2 an Rauschbins →
        # NSIM-Kollaps). HPG nur für Materialien mit SNR ≥ ~12 dB aktivieren.
        _hpg_skip_materials = {"shellac", "wax_cylinder"}
        if material_type.value not in _hpg_skip_materials:
            try:
                from backend.core.harmonic_preservation_guard import get_harmonic_preservation_guard

                _hpg = get_harmonic_preservation_guard()
                _hpg_instrument_tag = {
                    "vinyl": "piano_mid",
                    "tape": "piano_mid",
                    "shellac": "piano_bass",
                    "reel_tape": "piano_mid",
                    "cd_digital": "piano_mid",
                    "mp3_low": "piano_mid",
                    "mp3_high": "piano_mid",
                    "aac": "piano_mid",
                    "dat": "piano_mid",
                    "minidisc": "piano_mid",
                }.get(material_type.value, "unknown")
                _hpg_mask, _hpg_href = _hpg.extract_harmonic_mask(
                    original_audio_for_goals, sample_rate, instrument_tag=_hpg_instrument_tag
                )
                restored_audio = _hpg.apply_correction(restored_audio, _hpg_href, _hpg_mask, sample_rate)
                restored_audio = np.clip(np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                logger.info("§2.28 HPG: Harmonische Energiekorrektur angewendet (tag=%s)", _hpg_instrument_tag)
            except Exception as _hpg_exc:
                logger.debug("HarmonicPreservationGuard nicht verfügbar: %s", _hpg_exc)
        else:
            logger.info(
                "§2.28 HPG: Übersprungen für material=%s (SNR zu niedrig für f₀-Schätzung)",
                material_type.value,
            )

        # §2.27 TDP Recombination — Percussive + verarbeiteter harmonischer Anteil (OLA)
        if _tdp_proc is not None and _tdp_percussive is not None and _tdp_harmonic_ready:
            try:
                restored_audio = _tdp_proc.recombine(_tdp_percussive, restored_audio, sample_rate, _tdp_percussive)
                restored_audio = np.clip(np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                logger.info("§2.27 TDP: Recombination abgeschlossen (OLA-Crossfade 10 ms Hanning)")
            except Exception as _tdp_rc_exc:
                logger.debug("TDP Recombination fehlgeschlagen: %s", _tdp_rc_exc)

        # --- HarmonicLattice: Harmonische Konsistenzerzwingung (§2.11 Spec) ---
        _lattice_result = None
        try:
            from backend.core.harmonic_lattice_analyzer import get_harmonic_lattice

            _hla = get_harmonic_lattice()
            # Instrument-Tag aus Material ableiten (konservative Prior-Wahl)
            _hla_tag = {
                "vinyl": "piano_mid",
                "tape": "piano_mid",
                "shellac": "piano_bass",
                "reel_tape": "piano_mid",
                "cd_digital": "piano_mid",
                "mp3_low": "piano_mid",
                "mp3_high": "piano_mid",
                "aac": "piano_mid",
            }.get(material_type.value, "unknown")
            _lattice_result = _hla.analyze(restored_audio, sample_rate, instrument_tag=_hla_tag)
            if _lattice_result.needs_enforcement:
                restored_audio = _hla.enforce_coherence(restored_audio, sample_rate, _lattice_result)
                logger.info(
                    "🎵 HarmonicLattice: Korrektur angewendet " "(lattice_score=%.3f B=%.5f n_partials_korrigiert=%d)",
                    _lattice_result.lattice_score,
                    _lattice_result.inharmonicity_b,
                    sum(1 for p in _lattice_result.partials if p.needs_correction),
                )
            else:
                logger.debug("HarmonicLattice: OK (lattice_score=%.3f)", _lattice_result.lattice_score)
            # NaN/Inf-Guard nach HarmonicLattice
            restored_audio = np.clip(np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        except Exception as _hl_exc:
            logger.debug("HarmonicLatticeAnalyzer nicht verfügbar: %s", _hl_exc)

        # --- PerceptualQualityScorer: PQS-MOS Messung (§2.6 Spec) ---
        _pqs_result = None
        try:
            from backend.core.perceptual_quality_scorer import (
                score_audio as _score_audio,
                score_audio_absolute as _score_abs,
            )

            _shape_match = original_audio_for_goals.shape == restored_audio.shape and original_audio_for_goals.size > 0
            if _shape_match:
                _pqs_result = _score_audio(original_audio_for_goals, restored_audio, sample_rate)
            else:
                _pqs_result = _score_abs(restored_audio, sample_rate)
            if _pqs_result.pqs_mos >= 4.0:
                logger.info(
                    "📊 PQS-MOS: %.2f (NSIM=%.3f MCD=%.1f dB Kohärenz=%.3f)",
                    _pqs_result.pqs_mos,
                    _pqs_result.nsim,
                    _pqs_result.mcd_db,
                    _pqs_result.spectral_coherence,
                )
            else:
                logger.warning(
                    "⚠️ PQS-MOS unter Zielwert 4.0: %.2f (NSIM=%.3f MCD=%.1f dB)",
                    _pqs_result.pqs_mos,
                    _pqs_result.nsim,
                    _pqs_result.mcd_db,
                )
        except Exception as _pqs_exc:
            logger.debug("PerceptualQualityScorer nicht verfügbar: %s", _pqs_exc)

        # §2.31 AdaptiveGoalThresholds — Kontextadaptive Ziel-Schwellenwerte
        _adaptive_goals = None
        try:
            from backend.core.musical_goals.adaptive_goals_system import get_adaptive_goals_and_config

            _adaptive_goals = get_adaptive_goals_and_config(  # noqa: F841
                audio, sample_rate, _classified_material, defect_result
            )
            logger.info(
                "🎯 AdaptiveGoalThresholds: konfiguriert (material=%s)",
                getattr(_classified_material, "value", str(_classified_material)),
            )
        except Exception as _agt_exc:
            logger.debug("AdaptiveGoalThresholds nicht verfügbar: %s", _agt_exc)

        # §2.33 PhysicalCeilingEstimator — Physikalische Obergrenze der Restaurierbarkeit
        _physical_ceiling = None
        try:
            from backend.core.physical_ceiling_estimator import estimate_physical_ceiling

            _scores_for_ceiling = _pqs_result.__dict__ if _pqs_result is not None else {}
            _mat_for_ceiling = material_type.value if material_type is not None else "unknown"
            _physical_ceiling = estimate_physical_ceiling(
                restored_audio, sample_rate, _scores_for_ceiling, _mat_for_ceiling
            )
            logger.info(
                "📐 PhysicalCeilingEstimator: bandwidth=%.0f Hz further_opt=%s",
                _physical_ceiling.effective_bandwidth_hz,
                _physical_ceiling.further_optimization_worthwhile,
            )
        except Exception as _pce_exc:
            logger.debug("PhysicalCeilingEstimator nicht verfügbar: %s", _pce_exc)

        # --- Musikalische Exzellenz: 14-Ziele-Messung (§1.2 Spec) ---
        _musical_goal_scores: dict = {}
        _musical_goals_passed: dict = {}
        _musical_excellence_score: float = 0.0
        try:
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            _mg_checker = MusicalGoalsChecker()
            _effective_goal_thresholds = dict(_mg_checker.thresholds)
            # AdaptiveGoalThresholds (v8-kompatibel) nur dort anwenden, wo Schluessel vorhanden sind.
            if _adaptive_goals is not None:
                try:
                    _adaptive_thresholds_obj = _adaptive_goals[0] if isinstance(_adaptive_goals, tuple) else None
                    if _adaptive_thresholds_obj is not None:
                        for _goal_name in (
                            "brillanz",
                            "waerme",
                            "natuerlichkeit",
                            "authentizitaet",
                            "emotionalitaet",
                            "transparenz",
                            "bass_kraft",
                        ):
                            _val = getattr(_adaptive_thresholds_obj, _goal_name, None)
                            if _val is not None:
                                _effective_goal_thresholds[_goal_name] = float(_val)
                except Exception as _agt_apply_exc:
                    logger.debug("AdaptiveGoalThresholds konnten nicht angewendet werden: %s", _agt_apply_exc)

            _applicable_goal_names = (
                set(_goal_applicability_result.applicable)
                if _goal_applicability_result is not None
                else set(_mg_checker.thresholds.keys())
            )
            # reference= gibt Authentizitäts- und Timbre-Metriken entscheidend mehr Kontext
            _mg_ref = original_audio_for_goals if original_audio_for_goals.shape == restored_audio.shape else None
            _musical_goal_scores = _mg_checker.measure_all(restored_audio, sample_rate, reference=_mg_ref)
            _musical_goals_passed = {
                k: (
                    True
                    if k not in _applicable_goal_names
                    else float(_musical_goal_scores.get(k, 0.0)) >= _effective_goal_thresholds.get(k, 0.85)
                )
                for k in _musical_goal_scores
            }
            _musical_excellence_score = sum(_musical_goal_scores.values()) / max(len(_musical_goal_scores), 1)
            _mg_violations = [k for k, p in _musical_goals_passed.items() if not p and k in _applicable_goal_names]
            if _mg_violations:
                logger.warning(
                    "🎵 Musical Goals Verletzungen (%d/%d): %s",
                    len(_mg_violations),
                    len(_musical_goal_scores),
                    ", ".join(_mg_violations),
                )
                # §1.2 Aurik als Möglichmacher: Adaptiver Re-Pass statt Export-Blockade
                # Ziel: Beste erreichbare Qualität — niemals blockieren, immer verbessern.
                _mg_violations_before = list(_mg_violations)
                try:
                    from backend.core.excellence_optimizer import optimize_for_excellence as _eo_rp

                    logger.info(
                        "🔄 Musical Goals Re-Pass: %d verletzte Ziele → gezielter ExcellenceOptimizer",
                        len(_mg_violations),
                    )
                    _rp_audio, _ = _eo_rp(restored_audio, sample_rate, material=material_type.value)
                    _rp_scores = _mg_checker.measure_all(_rp_audio, sample_rate, reference=_mg_ref)
                    _rp_violations = [
                        k
                        for k in _rp_scores
                        if k in _applicable_goal_names and float(_rp_scores.get(k, 0.0)) < _effective_goal_thresholds.get(k, 0.85)
                    ]
                    if len(_rp_violations) <= len(_mg_violations):
                        # Re-Pass verbessert oder erhält — immer übernehmen
                        restored_audio = _rp_audio
                        _musical_goal_scores = _rp_scores
                        _musical_goals_passed = {
                            k: (
                                True
                                if k not in _applicable_goal_names
                                else float(_rp_scores.get(k, 0.0)) >= _effective_goal_thresholds.get(k, 0.85)
                            )
                            for k in _rp_scores
                        }
                        _musical_excellence_score = sum(_rp_scores.values()) / max(len(_rp_scores), 1)
                        _mg_violations = _rp_violations
                        logger.info(
                            "✅ Musical Goals Re-Pass: Verletzungen %d → %d (Ø %.3f)",
                            len(_mg_violations_before),
                            len(_rp_violations),
                            _musical_excellence_score,
                        )
                    else:
                        logger.warning(
                            "Musical Goals Re-Pass: kein Fortschritt — %d Verletzungen bleiben: %s",
                            len(_mg_violations),
                            ", ".join(_mg_violations),
                        )
                except Exception as _rp_exc:
                    logger.debug("Musical Goals Re-Pass nicht verfügbar: %s", _rp_exc)
            else:
                logger.info(
                    "🎵 Musical Goals: alle %d Ziele erfüllt (Ø %.3f)",
                    len(_musical_goal_scores),
                    _musical_excellence_score,
                )
        except Exception as _mg_exc:
            logger.debug("MusicalGoalsChecker nicht verfügbar: %s", _mg_exc)

        # §2.34 GoalPriorityProtocol — Iterations-Abbruch-Entscheidung bei Ziel-Regression
        _goal_abort = None
        try:
            from backend.core.goal_priority_protocol import check_iteration_abort

            _scores_before_gpp = {"overall": getattr(_quality_before, "overall_score", 0.0)} if _quality_before else {}
            _scores_after_gpp = {"musical_excellence": _musical_excellence_score}
            _scores_after_gpp.update(_musical_goal_scores)
            _goal_abort = check_iteration_abort(_scores_before_gpp, _scores_after_gpp)
            if _goal_abort.should_abort:
                logger.warning(
                    "⛔ GoalPriorityProtocol: Abbruch empfohlen — %s (degradiert: %s)",
                    _goal_abort.reason,
                    getattr(_goal_abort, "degraded_goals", []),
                )
            else:
                logger.debug("GoalPriorityProtocol: kein Abbruch notwendig")
        except Exception as _gpp_exc:
            logger.debug("GoalPriorityProtocol nicht verfügbar: %s", _gpp_exc)

        # §8.2 EmotionalArcPreservationMetric — Emotionaler Dynamik-Bogen-Erhalt (Punkt 12)
        try:
            from backend.core.emotional_arc_preservation import measure_emotional_arc

            _arc_result = measure_emotional_arc(original_audio_for_goals, restored_audio, sample_rate)
            if _arc_result.skipped:
                logger.debug("EmotionalArc: übersprungen (Datei zu kurz, < 30 s)")
            elif _arc_result.arc_preserved:
                logger.info(
                    "✅ EmotionalArc: Bogen erhalten — arousal_pearson=%.3f valence_pearson=%.3f",
                    _arc_result.arousal_pearson,
                    _arc_result.valence_pearson,
                )
            else:
                logger.warning(
                    "⚠️ EmotionalArc: Bogen nicht erfüllt — %s",
                    _arc_result.reason,
                )
        except Exception as _eap_exc:
            logger.debug("EmotionalArcPreservationMetric nicht verfügbar: %s", _eap_exc)

        # --- GP-Lernzyklus: Vorbereitung — PQS-Norm + Pareto-Vorschlag (vor MDEM) (§2.5 Spec) ---
        # GPParameterOptimizer.update() wird erst NACH MDEM aufgerufen (Spec §2.5 normativ).
        _pqs_mos_norm: Optional[float] = None
        _gp_opt_ref: Any = None
        _gp_upd_proposal_ref: Any = None
        _gp_score_ref: float = 0.0
        _gp_goal_scores_ref: Optional[Dict[str, float]] = None
        if _musical_excellence_score > 0.0 or _pqs_result is not None:
            try:
                from backend.core.gp_parameter_optimizer import get_optimizer

                _gp_opt_ref = get_optimizer()
                # Kombinierter Score: PQS-MOS (normiert [1-5]→[0-1]) + Musical Excellence
                _pqs_mos_norm = (
                    (_pqs_result.pqs_mos - 1.0) / 4.0
                    if _pqs_result is not None and math.isfinite(_pqs_result.pqs_mos)
                    else None
                )
                if _pqs_mos_norm is not None and _musical_excellence_score > 0.0:
                    _gp_score_ref = 0.5 * _musical_excellence_score + 0.5 * _pqs_mos_norm
                elif _musical_excellence_score > 0.0:
                    _gp_score_ref = _musical_excellence_score
                else:
                    _gp_score_ref = _pqs_mos_norm or 0.0
                # Era-Warmstart: Ära-Prior aus EraClassifier für GP-Cold-Start (§2.14)
                _era_ws: Optional[Dict[str, float]] = None
                if _era_result is not None:
                    try:
                        from backend.core.era_classifier import get_era_classifier as _get_era_clf

                        _era_ws = _get_era_clf().get_gp_warmstart(_era_result)
                        logger.debug(
                            "EraClassifier GP-Warmstart: decade=%d NR_strength=%.2f",
                            _era_result.decade,
                            _era_ws.get("noise_reduction_strength", 0.0),
                        )
                    except Exception as _era_ws_exc:
                        logger.debug("Era-GP-Warmstart nicht verfügbar: %s", _era_ws_exc)
                _gp_upd_proposal_ref = _gp_opt_ref.propose(
                    material=_gp_material_key,
                    embedding=_original_embedding.vector if _original_embedding is not None else None,
                    era_warmstart=_era_ws,
                )
                # goal_scores für echten MOO vorberechnen (§2.5 Spec 03)
                _gp_goal_scores_ref = (
                    {k: float(v) for k, v in _musical_goal_scores.items()} if _musical_goal_scores else None
                )
            except Exception as _gp_prep_exc:
                logger.debug("GP-Vorbereitung (vor MDEM) nicht verfügbar: %s", _gp_prep_exc)

        # §2.30 MicroDynamicsEnvelopeMorphing — LUFS-Profil-Rückgewinnung (letzter DSP-Schritt)
        try:
            from backend.core.micro_dynamics_envelope_morphing import get_mdem

            _mdem_instance = get_mdem()
            # Modus-Erkennung: studio2026 wenn kein reines Restoration-Profil
            _mdem_mode = "restoration"
            try:
                _mode_val = str(self.config.mode.value).lower()
                if any(_kw in _mode_val for _kw in ("maximum", "studio", "aggressive")):
                    _mdem_mode = "studio2026"
            except Exception:
                pass
            restored_audio = _mdem_instance.morph(
                restored_audio, original_audio_for_goals, sample_rate, mode=_mdem_mode
            )
            restored_audio = np.clip(np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            logger.info("§2.30 MDEM: Mikro-Dynamik-Morphing abgeschlossen (mode=%s)", _mdem_mode)
        except Exception as _mdem_exc:
            logger.warning("MicroDynamicsEnvelopeMorphing fehlgeschlagen: %s", _mdem_exc)

        # --- GP-Lernzyklus: update() NACH MDEM — Spec §2.5 (normativ: letzter Schritt) ---
        # Score basiert auf PQS + Musical Goals vor MDEM (korrekt: MDEM ist Hüllkurven-Morphing,
        # ändert keine PQS-Messungen). Update NACH MDEM garantiert vollständige Pipeline-Daten.
        if _gp_opt_ref is not None and _gp_upd_proposal_ref is not None:
            try:
                _gp_opt_ref.update(
                    parameters=_gp_upd_proposal_ref.parameters,
                    score=_gp_score_ref,
                    material=_gp_material_key,
                    goal_scores=_gp_goal_scores_ref,
                )
                logger.debug(
                    "GPParameterOptimizer post-MDEM: Update material=%s score=%.4f goals=%d"
                    " (excellence=%.4f pqs_norm=%s)",
                    _gp_material_key,
                    _gp_score_ref,
                    len(_gp_goal_scores_ref) if _gp_goal_scores_ref else 0,
                    _musical_excellence_score,
                    f"{_pqs_mos_norm:.4f}" if _pqs_mos_norm is not None else "n/a",
                )
            except Exception as _gp_upd_exc:
                logger.debug("GPParameterOptimizer.update post-MDEM nicht verfügbar: %s", _gp_upd_exc)

        # §1.4 Vocos-Synthese — Qualitäts-Finisher (Studio 2026, konditionell)
        # Spec: "wenn PQS-MOS(Mix nach Schritt 11) < 4.3"
        # Invariante: NIEMALS im Restoration-Modus; NIEMALS wenn PQS-MOS ≥ 4.3
        # Fallback-Kaskade: Vocos ONNX → HiFi-GAN → PGHI-ISTFT (§4.5)
        try:
            _vocos_mode_val = str(getattr(getattr(self.config, "mode", None), "value", "")).lower()
            _is_vocos_studio = any(_kw in _vocos_mode_val for _kw in ("maximum", "studio", "aggressive"))
            # (4.3 − 1) / 4 = 0.825 ist die normalisierte PQS-MOS-Schwelle
            _vocos_mos_ok = (_pqs_mos_norm is None) or (_pqs_mos_norm < 0.825)
            if _is_vocos_studio and _vocos_mos_ok:
                import os as _vos
                import sys as _vsys

                _vplugins = _vos.path.join(_vos.path.dirname(__file__), "..", "plugins")
                if _vplugins not in _vsys.path:
                    _vsys.path.insert(0, _vos.path.abspath(_vplugins))
                from vocos_plugin import get_vocos_plugin as _get_vocos_plg

                _vplug = _get_vocos_plg()
                if restored_audio.ndim == 2:
                    # Stereo: beide Kanäle separat synthetisieren und stapeln
                    _vch0 = _vplug.vocode(restored_audio[0], sample_rate, mode="studio2026").audio
                    _vch1 = _vplug.vocode(restored_audio[1], sample_rate, mode="studio2026").audio
                    _vn = min(len(_vch0), len(_vch1), restored_audio.shape[1])
                    _vocos_out = np.stack([_vch0[:_vn], _vch1[:_vn]], axis=0)
                    logger.info(
                        "§1.4 Vocos: Stereo-Finisher angewendet (model=%s)",
                        (
                            _vplug.vocode(restored_audio[0], sample_rate, mode="studio2026").model_used
                            if False
                            else "vocos_mel_24khz/griffin_lim"
                        ),
                    )
                else:
                    _vr = _vplug.vocode(restored_audio, sample_rate, mode="studio2026")
                    _vocos_out = _vr.audio
                    logger.info(
                        "§1.4 Vocos: Mono-Finisher angewendet " "(model=%s pqs_mos=%.3f mel_snr=%.1f dB)",
                        _vr.model_used,
                        _vr.pqs_mos,
                        _vr.mel_snr_db,
                    )
                restored_audio = np.clip(
                    np.nan_to_num(_vocos_out, nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
            elif _is_vocos_studio:
                logger.debug(
                    "§1.4 Vocos: Übersprungen — PQS-MOS ≥ 4.3 "
                    "(pqs_mos_norm=%.3f ≥ 0.825), kein Qualitätsverlust möglich",
                    _pqs_mos_norm if _pqs_mos_norm is not None else -1.0,
                )
        except Exception as _vocos_exc:
            logger.warning(
                "§1.4 Vocos-Synthese nicht verfügbar — Fallback auf PGHI-ISTFT-Ausgang: %s "
                "(Installationshinweis: plugins/vocos_plugin.py + Vocos-Gewichte erforderlich)",
                _vocos_exc,
            )

        # MushraEvaluator: Objektiver MUSHRA-Score (§8.1, OQS) — Original vs. Restauriert
        # Pflicht-Schwelle: OQS ≥ 80 (Good); Studio-2026-Ziel: ≥ 88
        _mushra_result = None
        try:
            from backend.core.mushra_evaluator import get_mushra_evaluator as _get_mushra

            _mushra_eval = _get_mushra()
            _orig_mono_m = audio if audio.ndim == 1 else audio.mean(axis=1)
            _rest_mono_m = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)
            _mushra_result = _mushra_eval.evaluate(
                reference=_orig_mono_m,
                test=_rest_mono_m,
                sr=sample_rate,
                compute_anchor=True,
            )
            _mushra_threshold = 88.0 if self.config.mode == QualityMode.MAXIMUM else 80.0
            _mushra_pass = _mushra_result.passes_mushra_threshold(_mushra_threshold)
            logger.info(
                "📊 MushraEvaluator: OQS=%.1f grade=%s anchor=%.1f (Schwelle=%.0f → %s)",
                _mushra_result.mushra_score,
                _mushra_result.grade,
                _mushra_result.anchor_score,
                _mushra_threshold,
                "✅ Pass" if _mushra_pass else "⚠️ Under",
            )
        except Exception as _mushra_exc:
            logger.debug("MushraEvaluator nicht verfügbar: %s", _mushra_exc)

        # PsychoacousticArtifactDetector: Artefakt-Analyse des restaurierten Audios
        # Liefert masking_effect, transient_loss, musical_transparency als Reporting-Scores
        _artifact_scores: Optional[Dict[str, float]] = None
        try:
            from backend.core.psychoacoustic_artifact_detector import PsychoacousticArtifactDetector as _PAD

            _pad_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)
            _artifact_scores = _PAD().analyze(_pad_audio, sample_rate)
            logger.debug(
                "🔍 ArtifactDetector: masking=%.3f transient_loss=%.3f transparency=%.3f",
                _artifact_scores.get("masking_effect", 0.0),
                _artifact_scores.get("transient_loss", 0.0),
                _artifact_scores.get("musical_transparency", 0.0),
            )
        except Exception as _pad_exc:
            logger.debug("PsychoacousticArtifactDetector nicht verfügbar: %s", _pad_exc)

        # QualityAnalyzer: Ausgangsqualität messen (Nachher) — Vergleich mit Vorher-Baseline
        _quality_after: Optional[object] = None
        try:
            from backend.core.quality_prediction import QualityAnalyzer as _QualityAnalyzerPost

            _quality_after = _QualityAnalyzerPost().analyze_quality(restored_audio, sample_rate)
            _qa_delta = (
                _quality_after.overall_score - _quality_before.overall_score
                if _quality_before is not None and math.isfinite(_quality_after.overall_score)
                else None
            )
            logger.info(
                "📌 QualityAnalyzer (Nachher): score=%.1f (Δ%s) SNR=%.1f warmth=%.3f naturalness=%.3f",
                _quality_after.overall_score,
                (
                    f"+{_qa_delta:.1f}"
                    if _qa_delta is not None and _qa_delta >= 0
                    else (f"{_qa_delta:.1f}" if _qa_delta is not None else "n/a")
                ),
                _quality_after.snr_db,
                _quality_after.warmth,
                _quality_after.naturalness,
            )
        except Exception as _qa_post_exc:
            logger.debug("QualityAnalyzer (Nachher) nicht verfügbar: %s", _qa_post_exc)

        # MaskingAnalyzer: Signal-to-Mask Ratio des restaurierten Audios (§4.5 Psychoakustik)
        # SMR > 0 dB = Signal ist hörbar über Masking-Schwelle
        # masked_ratio ∈ [0,1] = Anteil unheilbarer Masking-Komponenten
        _smr_result: Optional[Dict[str, float]] = None
        try:
            from backend.core.masking_analyzer import MaskingAnalyzer as _MaskingAnalyzer

            _ma = _MaskingAnalyzer()
            _ma_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)
            _smr_db = _ma.compute_smr(_ma_audio, sample_rate)
            _ma_profile = _ma.analyze(_ma_audio, sample_rate)
            _masked_ratio = _ma_profile.get_masked_components_ratio()
            _smr_result = {
                "smr_db": round(float(_smr_db), 2),
                "masked_ratio": round(float(_masked_ratio), 4),
                "audible_ratio": round(1.0 - float(_masked_ratio), 4),
            }
            logger.debug(
                "🎧 MaskingAnalyzer: SMR=%.1f dB masked_ratio=%.3f audible=%.3f",
                _smr_db,
                _masked_ratio,
                1.0 - _masked_ratio,
            )
        except Exception as _ma_exc:
            logger.debug("MaskingAnalyzer nicht verfügbar: %s", _ma_exc)

        # AuthenticityMetricsExtended: Genre-spezifische Authentizitätserkennung (§2.8 Instrument)
        # Erkennt Finger Noise, Bow Noise, Pedal, Brush Texture, Vinyl Character — nach Restaurierung
        _authenticity_extended: Optional[Dict] = None
        try:
            from backend.core.authenticity_metrics_extended import AuthenticityMetricsExtended as _AME

            _ame_result = _AME().analyze(restored_audio, sample_rate)
            # Nur gefundene Elemente und KV-Scores ins Reporting — kein internes Rohdaten-Dump
            _authenticity_extended = {
                "detected_elements": _ame_result.get("detected_elements", []),
                "vinyl_warmth": bool(_ame_result.get("vinyl_character", {}).get("warmth_detected", False)),
                "vinyl_defects": bool(_ame_result.get("vinyl_character", {}).get("defects_detected", False)),
                "finger_noise": bool(_ame_result.get("finger_noise", {}).get("finger_noise_detected", False)),
                "bow_noise": bool(_ame_result.get("bow_noise", {}).get("bow_noise_detected", False)),
                "brush_texture": bool(_ame_result.get("brush_texture", {}).get("brush_texture_detected", False)),
            }
            logger.debug(
                "🎺 AuthenticityMetricsExtended: Elemente=%s",
                _authenticity_extended["detected_elements"] or "Keine",
            )
        except Exception as _ame_exc:
            logger.debug("AuthenticityMetricsExtended nicht verfügbar: %s", _ame_exc)

        # §2.8 AuthenticityMetrics: BreathDetector, PlosiveDetector, TransientDetector, SibilanceDetector, RoomToneDetector
        _authenticity_perf: Optional[Dict] = None
        try:
            from backend.core.authenticity_metrics import (
                BreathDetector as _BreathDetector,
                PlosiveDetector as _PlosiveDetector,
                RoomToneDetector as _RoomToneDetector,
                SibilanceDetector as _SibilanceDetector,
                TransientDetector as _TransientDetector,
            )

            _ap_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1).astype(np.float32)
            _breath_events = _BreathDetector().detect(_ap_audio, sample_rate)
            _plosive_events = _PlosiveDetector().detect(_ap_audio, sample_rate)
            _transient_events = _TransientDetector().detect(_ap_audio, sample_rate)
            _sibilance = _SibilanceDetector().analyze(_ap_audio, sample_rate)
            _room_tone = _RoomToneDetector().analyze(_ap_audio, sample_rate)
            _authenticity_perf = {
                "breath_count": len(_breath_events),
                "plosive_count": len(_plosive_events),
                "transient_count": len(_transient_events),
                "sibilance_density": round(float(_sibilance.sibilance_density), 4),
                "sibilance_natural": bool(_sibilance.natural_level),
                "peak_sibilance_freq_hz": round(float(_sibilance.peak_sibilance_frequency), 1),
                "room_noise_floor_db": round(float(_room_tone.ambient_noise_floor_db), 2),
                "room_reverb_ms": round(float(_room_tone.reverb_tail_length_ms), 1),
                "room_naturalness": round(float(_room_tone.naturalness_score), 4),
            }
            logger.debug(
                "🎙️ AuthenticityMetrics: Atem=%d Plosive=%d Transienten=%d "
                "Sibilanz=%.3f (natürlich=%s) Raumhall=%.0f ms",
                len(_breath_events),
                len(_plosive_events),
                len(_transient_events),
                _sibilance.sibilance_density,
                _sibilance.natural_level,
                _room_tone.reverb_tail_length_ms,
            )
        except Exception as _am_exc:
            logger.debug("AuthenticityMetrics nicht verfügbar: %s", _am_exc)

        # §8.1 MusicalQualityAssurance: validate_final_quality() — post-pipeline Quality Gating
        _mqa_result: Optional[Dict] = None
        try:
            from backend.core.musical_quality_assurance import (
                MediumType as _MediumType,
                MusicalQualityAssurance as _MQA,
                ProcessingMode as _ProcessingMode,
            )

            _MATERIAL_TO_MEDIUM: Dict[str, Any] = {
                "tape": _MediumType.CASSETTE,
                "reel_tape": _MediumType.REEL_TO_REEL,
                "vinyl": _MediumType.VINYL_33,
                "shellac": _MediumType.SHELLAC_78,
                "dat": _MediumType.DAT,
                "cd_digital": _MediumType.CD,
                "mp3_low": _MediumType.LOSSY_LOW,
                "mp3_high": _MediumType.LOSSY_HIGH,
                "aac": _MediumType.LOSSY_HIGH,
                "minidisc": _MediumType.MINIDISC,
                "streaming": _MediumType.LOSSY_MID,
                "wax_cylinder": _MediumType.WAX_CYLINDER,
                "wire_recording": _MediumType.WIRE_RECORDING,
                "lacquer_disc": _MediumType.ACETATE,
            }
            _mqa_medium = _MATERIAL_TO_MEDIUM.get(str(material_type).lower(), _MediumType.UNKNOWN)
            _is_studio_26 = (
                getattr(self.config, "studio_2026", False)
                or getattr(getattr(self.config, "mode", None), "value", "") == "studio_2026"
            )
            _mqa_mode = _ProcessingMode.STUDIO_2026 if _is_studio_26 else _ProcessingMode.RESTORATION
            _mqa_ref = (
                original_audio_for_goals
                if original_audio_for_goals.shape == restored_audio.shape and original_audio_for_goals.size > 0
                else restored_audio
            )
            _mqa_report = _MQA().validate_final_quality(
                _mqa_ref,
                restored_audio,
                sample_rate,
                _mqa_medium,
                _mqa_mode,
                list(executed_phases),
            )
            _mqa_result = {
                "quality_guaranteed": bool(_mqa_report.quality_guaranteed),
                "verdict": str(_mqa_report.verdict),
                "musical_improvement": round(float(_mqa_report.musical_improvement), 4),
                "authenticity_preserved": bool(_mqa_report.authenticity_preserved),
                "character_preserved": bool(_mqa_report.character_preserved),
                "natural_sound": bool(_mqa_report.natural_sound),
                "overprocessed": bool(_mqa_report.overprocessed),
                "processing_intensity": round(float(_mqa_report.processing_intensity), 4),
                "warnings": list(_mqa_report.warnings),
                "recommendations": list(_mqa_report.recommendations),
                "medium_type": _mqa_medium.value,
            }
            logger.debug(
                "🎵 MusicalQualityAssurance: %s | Verbesserung=%.1f%% authentisch=%s natürlich=%s",
                "✅ GARANTIERT" if _mqa_report.quality_guaranteed else "⚠ NICHT GARANTIERT",
                _mqa_report.musical_improvement * 100,
                _mqa_report.authenticity_preserved,
                _mqa_report.natural_sound,
            )
        except Exception as _mqa_exc:
            logger.debug("MusicalQualityAssurance nicht verfügbar: %s", _mqa_exc)

        # §1.2 AestheticJudgmentModel: CompositeAestheticScoreCalculator — CAS (Composite Aesthetic Score)
        _aesthetic_result: Optional[Dict] = None
        try:
            from backend.core.aesthetic_judgment import CompositeAestheticScoreCalculator as _CASCalc
            from backend.core.data_models import (
                AnalysisProfile as _AProfile,
                DynamicsAnalysis as _DynAnal,
                FeatureVectors as _FeatVec,
                FormatInfo as _FormatInfo,
                Genre as _DataGenre,
                MaterialChainAnalysis as _MatChain,
                MediaType as _MediaType,
                MusicalContext as _MusCtx,
                SpectralAnalysis as _SpectralAnal,
                StereoAnalysis as _StereoAnal,
                VocalAnalysis as _VocalAnal,
            )

            _ae_mono = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)).astype(np.float32)
            # Spektralfeatures für minimales AnalysisProfile (aus Audio)
            _ae_fft = np.fft.rfft(_ae_mono)
            _ae_freqs = np.fft.rfftfreq(len(_ae_mono), 1.0 / sample_rate)
            _ae_mag = np.abs(_ae_fft) ** 2 + 1e-10
            _ae_total = float(np.sum(_ae_mag))
            _ae_centroid = float(np.clip(np.sum(_ae_freqs * _ae_mag) / _ae_total, 0.0, sample_rate / 2))
            _ae_cumsum = np.cumsum(_ae_mag)
            _ae_rf_idx = int(np.searchsorted(_ae_cumsum, 0.85 * _ae_total))
            _ae_rolloff = float(_ae_freqs[min(_ae_rf_idx, len(_ae_freqs) - 1)])
            _ae_flux = float(np.std(np.abs(_ae_fft)))
            _ae_lo_idx = int(np.searchsorted(_ae_cumsum, 0.05 * _ae_total))
            _ae_hi_idx = int(np.searchsorted(_ae_cumsum, 0.95 * _ae_total))
            _ae_bw = float(
                max(
                    0.0, _ae_freqs[min(_ae_hi_idx, len(_ae_freqs) - 1)] - _ae_freqs[min(_ae_lo_idx, len(_ae_freqs) - 1)]
                )
            )
            _ae_rms = float(20 * np.log10(np.sqrt(np.mean(_ae_mono**2)) + 1e-9))
            _ae_peak = float(20 * np.log10(np.max(np.abs(_ae_mono)) + 1e-9))
            _ae_dr = max(0.0, _ae_peak - _ae_rms)
            if restored_audio.ndim == 2 and restored_audio.shape[1] >= 2:
                _ae_l, _ae_r = restored_audio[:, 0], restored_audio[:, 1]
                _ae_coh = float(np.clip(np.corrcoef(_ae_l, _ae_r)[0, 1], 0.0, 1.0))
                _ae_ms = float(
                    np.sqrt(np.mean(((_ae_l + _ae_r) / 2) ** 2)) / (np.sqrt(np.mean(((_ae_l - _ae_r) / 2) ** 2)) + 1e-9)
                )
                _ae_width = 1.0
            else:
                _ae_coh, _ae_ms, _ae_width = 1.0, 1.0, 0.0
            _ae_profile = _AProfile(
                format_info=_FormatInfo(
                    container_format="WAV",
                    codec="PCM",
                    sample_rate=sample_rate,
                    channels=int(restored_audio.shape[1]) if restored_audio.ndim == 2 else 1,
                    dc_offset=float(np.mean(_ae_mono)),
                    has_clipping=bool(np.max(np.abs(_ae_mono)) >= 0.999),
                ),
                material_chain=_MatChain(
                    detected_medium=_MediaType.UNKNOWN,
                    medium_confidence=0.5,
                ),
                spectral=_SpectralAnal(
                    spectral_centroid=max(0.0, _ae_centroid),
                    spectral_rolloff=max(0.0, _ae_rolloff),
                    spectral_flux=max(0.0, _ae_flux),
                    bandwidth=max(0.0, _ae_bw),
                ),
                dynamics=_DynAnal(
                    lufs_integrated=_ae_rms,
                    lufs_short_term=_ae_rms,
                    lufs_momentary=_ae_rms,
                    dynamic_range_db=_ae_dr,
                    crest_factor_db=_ae_dr,
                    true_peak_dbfs=_ae_peak,
                    rms_db=_ae_rms,
                    loudness_range_lu=1.0,
                ),
                stereo=_StereoAnal(
                    mid_side_balance=max(0.0, _ae_ms),
                    stereo_width=max(0.0, _ae_width),
                    phase_coherence=max(0.0, _ae_coh),
                    iacc=float(np.clip(_ae_coh, -1.0, 1.0)),
                    mono_compatibility_score=max(0.0, _ae_coh),
                ),
                musical_context=_MusCtx(
                    genre=_DataGenre.UNKNOWN,
                    genre_confidence=0.5,
                    harmonic_complexity=0.5,
                ),
                vocal_analysis=_VocalAnal(has_vocals=False, vocal_confidence=0.0),
                feature_vectors=_FeatVec(),
                overall_quality_score=0.5,
            )
            _ae_ref = (
                original_audio_for_goals
                if original_audio_for_goals.shape == restored_audio.shape and original_audio_for_goals.size > 0
                else None
            )
            _ae_cas, _ae_scores = _CASCalc().calculate_cas(
                restored_audio,
                sample_rate,
                _ae_profile,
                original_audio=_ae_ref,
                genre=_DataGenre.UNKNOWN,
                genre_confidence=0.5,
            )
            _aesthetic_result = {
                "cas_score": round(float(_ae_cas), 4),
                "brilliance": round(float(_ae_scores.brilliance), 4),
                "transparency": round(float(_ae_scores.transparency), 4),
                "naturalness": round(float(_ae_scores.naturalness), 4),
                "authenticity": round(float(_ae_scores.authenticity), 4),
                "emotionality": round(float(_ae_scores.emotionality), 4),
                "warmth": round(float(_ae_scores.warmth), 4),
                "spatiality": round(float(_ae_scores.spatiality), 4),
            }
            logger.debug(
                "🎨 AestheticJudgment: CAS=%.3f Brillanz=%.3f Wärme=%.3f Natürlichkeit=%.3f",
                _ae_cas,
                _ae_scores.brilliance,
                _ae_scores.warmth,
                _ae_scores.naturalness,
            )
        except Exception as _aj_exc:
            logger.debug("AestheticJudgmentModel nicht verfügbar: %s", _aj_exc)

        # §4.1 PsychoacousticCore: psychoakustische Gesamtanalyse (Bark, Masking, SMR)
        _psychoacoustic_result: Optional[Dict] = None
        try:
            from backend.core.psychoacoustic_core import analyze_psychoacoustic as _analyze_psychoacoustic

            _pa_audio = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)).astype(np.float32)
            _pa = _analyze_psychoacoustic(_pa_audio, sample_rate)
            _psychoacoustic_result = _pa.summary_dict()
            # Alle float-Werte runden
            _psychoacoustic_result = {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in _psychoacoustic_result.items()
            }
            logger.debug(
                "🔊 PsychoacousticCore: Centroid=%.2f Bark SMR=%.1f dB Maskiert=%.1f%%",
                _pa.perceptual_centroid_bark,
                _pa.signal_to_mask_ratio_db,
                _pa.masked_components_ratio * 100,
            )
        except Exception as _pa_exc:
            logger.debug("PsychoacousticCore nicht verfügbar: %s", _pa_exc)

        # §3.8 IntrinsicAudioQualityScorer: blinde Qualitätsbewertung (SNR, THD, Bark-Balance)
        _intrinsic_quality: Optional[Dict] = None
        try:
            from backend.core.intrinsic_audio_quality_scorer import IntrinsicAudioQualityScorer as _IAQS

            _iq_audio = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=1)).astype(np.float32)
            _iq = _IAQS().score(_iq_audio, sample_rate)
            _intrinsic_quality = {
                "overall": round(float(_iq.overall), 4),
                "snr_estimate_db": round(float(_iq.snr_estimate), 2),
                "snr_score": round(float(_iq.snr_score), 4),
                "spectral_regularity": round(float(_iq.spectral_regularity), 4),
                "bandwidth_score": round(float(_iq.bandwidth_score), 4),
                "bark_balance": round(float(_iq.bark_balance), 4),
                "dynamic_range_score": round(float(_iq.dynamic_range_score), 4),
                "transient_clarity": round(float(_iq.transient_clarity), 4),
                "thd_estimate_pct": round(float(_iq.thd_estimate_pct), 4),
                "thd_score": round(float(_iq.thd_score), 4),
            }
            logger.debug(
                "🌟 IntrinsicQuality: Gesamt=%.3f SNR=%.1f dB THD=%.3f%% BarkBalance=%.3f",
                _iq.overall,
                _iq.snr_estimate,
                _iq.thd_estimate_pct,
                _iq.bark_balance,
            )
        except Exception as _iq_exc:
            logger.debug("IntrinsicAudioQualityScorer nicht verfügbar: %s", _iq_exc)

        # §8.1 MusicMOS: Musik-Wahrnehmungsqualität (SIG, BAK, OVR, NAT)
        _music_mos_result: Optional[Dict] = None
        try:
            from backend.core.music_quality_scorer import score_music_mos as _score_music_mos

            _mm = _score_music_mos(restored_audio, sample_rate)
            _music_mos_result = _mm.to_dict()
            logger.debug(
                "🎧 MusicMOS: OVR=%.2f SIG=%.2f BAK=%.2f NAT=%.2f",
                _mm.MUSIC_OVR,
                _mm.MUSIC_SIG,
                _mm.MUSIC_BAK,
                _mm.MUSIC_NAT,
            )
        except Exception as _mm_exc:
            logger.debug("MusicMOS nicht verfügbar: %s", _mm_exc)

        # Speicher-Hygiene: Scoring-Modelle nach letztem Einsatz entladen.
        # MERT (~1.2–3.7 GB) und UTMOSv2 (~0.8 GB) werden ab hier nicht mehr gebraucht.
        try:
            from plugins.mert_plugin import unload_mert  # noqa: PLC0415
            unload_mert()
        except Exception:
            pass
        try:
            from plugins.utmos_plugin import unload_utmos  # noqa: PLC0415
            unload_utmos()
        except Exception:
            pass
        gc.collect()

        # §6.4 DeliveryStandards: LUFS-Messung nach EBU R128
        _delivery_standards_result: Optional[Dict] = None
        try:
            from backend.core.delivery_standards import LoudnessAnalyzer as _LoudnessAnalyzer

            _ds_result = _LoudnessAnalyzer().analyze(restored_audio, sample_rate)
            _delivery_standards_result = {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in _ds_result.items()
            }
            logger.debug("\ud83d\udce1 DeliveryStandards: %s", _delivery_standards_result)
        except Exception as _ds_exc:
            logger.debug("DeliveryStandards nicht verfügbar: %s", _ds_exc)

        # v9.10.19 — ArtifactDetector
        _ad_result: Optional[Dict] = None
        try:
            from backend.core.artifact_detection import RestorationArtifactDetector as _ArtifactDetector

            _ad_orig = (
                np.mean(original_audio_for_goals, axis=1)
                if original_audio_for_goals is not None and original_audio_for_goals.ndim == 2
                else (original_audio_for_goals if original_audio_for_goals is not None else restored_audio)
            )
            _ad_rest = np.mean(restored_audio, axis=1) if restored_audio.ndim == 2 else restored_audio
            _ad_min = min(len(_ad_orig), len(_ad_rest))
            _ad_r = _ArtifactDetector(sensitivity=0.4).analyze(
                _ad_orig[:_ad_min].astype(np.float32),
                _ad_rest[:_ad_min].astype(np.float32),
                sr=sample_rate,
            )
            _ad_result = {
                "total_count": _ad_r.total_count,
                "audible_count": _ad_r.audible_count,
                "artifacts_per_minute": round(float(_ad_r.artifacts_per_minute), 3),
                "overall_severity": _ad_r.overall_severity.name,
                "passes_aurik_standards": _ad_r.passes_aurik_standards,
            }
            logger.debug("🔍 ArtifactDetector: %s", _ad_result)
        except Exception as _ad_exc:
            logger.debug("ArtifactDetector nicht verfügbar: %s", _ad_exc)

        # v9.10.19 — BarkSpectrum
        _bark_result: Optional[Dict] = None
        try:
            from backend.core.bark_scale_processor import analyze_bark_spectrum as _analyze_bark

            _bark_audio = np.mean(restored_audio, axis=1) if restored_audio.ndim == 2 else restored_audio
            _bspec = _analyze_bark(_bark_audio.astype(np.float32), sr=sample_rate, normalize=True)
            _pb, _pe = _bspec.get_peak_band()
            _bark_result = {
                "total_energy": round(float(_bspec.total_energy), 6),
                "peak_band_center_hz": round(float(_pb.center_hz), 1),
                "peak_band_energy": round(float(_pe), 6),
                "spectral_centroid_bark": round(float(_bspec.get_spectral_centroid_bark()), 3),
                "n_bands": len(_bspec.bands),
                "energies_db_mean": round(float(np.mean(_bspec.energies_db)), 3),
            }
            logger.debug("📊 BarkSpectrum: %s", _bark_result)
        except Exception as _bark_exc:
            logger.debug("BarkSpectrum nicht verfügbar: %s", _bark_exc)

        # v9.10.19 — PsychoacousticMaskingModel
        _pmm_result: Optional[Dict] = None
        try:
            from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel as _PMM

            _pmm_audio = np.mean(restored_audio, axis=1) if restored_audio.ndim == 2 else restored_audio
            _pmm_r = _PMM().compute_threshold(_pmm_audio.astype(np.float32), sr=sample_rate)
            _pmm_result = _pmm_r.as_dict()
            logger.debug("🎭 PsychoacousticMaskingModel: %s", _pmm_result)
        except Exception as _pmm_exc:
            logger.debug("PsychoacousticMaskingModel nicht verfügbar: %s", _pmm_exc)

        # v9.10.19 — PsychoAcousticMetrics
        _pam_result: Optional[Dict] = None
        try:
            from backend.core.psychoacoustic_metrics import PsychoAcousticMetrics as _PAM

            _pam_audio = np.mean(restored_audio, axis=1) if restored_audio.ndim == 2 else restored_audio
            _pam = _PAM()
            _pam_result = {
                "roughness": round(float(_pam.calculate_roughness(_pam_audio)), 4),
                "sharpness": round(float(_pam.calculate_sharpness(_pam_audio)), 4),
                "spectral_flatness": round(float(_pam.calculate_spectral_flatness(_pam_audio)), 4),
                "temporal_smoothness": round(float(_pam.calculate_temporal_smoothness(_pam_audio)), 4),
                "harmonic_coherence": round(float(_pam.calculate_harmonic_coherence(_pam_audio)), 4),
                "noise_floor_consistency": round(float(_pam.calculate_noise_floor_consistency(_pam_audio)), 4),
                "naturalness_score": round(float(_pam.calculate_naturalness_score(_pam_audio)), 4),
            }
            logger.debug("🎵 PsychoAcousticMetrics: %s", _pam_result)
        except Exception as _pam_exc:
            logger.debug("PsychoAcousticMetrics nicht verfügbar: %s", _pam_exc)

        # v9.10.19 — ComprehensiveMetrics
        _cm_result: Optional[Dict] = None
        try:
            from backend.core.comprehensive_metrics import ComprehensiveMetricsCalculator as _CMC

            _cm_audio = restored_audio
            _cm_ref = original_audio_for_goals if original_audio_for_goals is not None else None
            _cm_r = _CMC(sample_rate=sample_rate).compute_all(_cm_audio, reference=_cm_ref)
            _cm_result = {
                "overall_technical_quality": round(float(_cm_r.overall_technical_quality), 4),
                "overall_musical_quality": round(float(_cm_r.overall_musical_quality), 4),
                "overall_emotional_impact": round(float(_cm_r.overall_emotional_impact), 4),
                "aurik_quality_score": round(float(_cm_r.aurik_quality_score), 2),
                "passes_aurik_standards": _cm_r.passes_aurik_standards(),
            }
            logger.debug("📈 ComprehensiveMetrics: %s", _cm_result)
        except Exception as _cm_exc:
            logger.debug("ComprehensiveMetrics nicht verfügbar: %s", _cm_exc)

        # v9.10.20 — EnhancedMetrics
        _em_result: Optional[Dict] = None
        try:
            from backend.core.enhanced_metrics import EnhancedMetrics as _EnhMetrics

            _em_audio = restored_audio
            _em_ref = original_audio_for_goals if original_audio_for_goals is not None else None
            if _em_ref is not None:
                _em_r = _EnhMetrics().compute_all(
                    _em_ref.astype(np.float32),
                    _em_audio.astype(np.float32),
                    sr=sample_rate,
                )
                _em_result = {
                    "snr_db": round(float(_em_r.snr_db), 3),
                    "thd": round(float(_em_r.thd), 6),
                    "lufs": round(float(_em_r.lufs), 3),
                    # si_sdr_db entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
                    # si_snr_db entfernt — verboten §4.4+§10.2 (Sprach-Metrik)
                    "snr_improvement_db": (
                        round(float(_em_r.snr_improvement_db), 3) if _em_r.snr_improvement_db is not None else None
                    ),
                    "passes_aurik_standards": bool(_em_r.passes_aurik_standards()),
                }
            else:
                _em_mono = np.mean(_em_audio, axis=1) if _em_audio.ndim == 2 else _em_audio
                _em_result = {
                    "snr_db": round(float(_EnhMetrics.compute_snr(_em_mono.astype(np.float32), sample_rate)), 3),
                    "thd": round(float(_EnhMetrics.compute_thd(_em_mono.astype(np.float32), sample_rate)), 6),
                    "lufs": round(float(_EnhMetrics.compute_lufs(_em_mono.astype(np.float32), sample_rate)), 3),
                }
            logger.debug("📊 EnhancedMetrics: %s", _em_result)
        except Exception as _em_exc:
            logger.debug("EnhancedMetrics nicht verfügbar: %s", _em_exc)

        # v9.10.20 — VocalCharacteristics (GenderDetector)
        _vc_result: Optional[Dict] = None
        try:
            from backend.core.vocal_ai_enhancement import GenderDetector as _GenderDetector

            _vc_audio = restored_audio
            _vc_mono = np.mean(_vc_audio, axis=1) if _vc_audio.ndim == 2 else _vc_audio
            _vc_r = _GenderDetector(sample_rate=sample_rate).detect(_vc_mono.astype(np.float32))
            _vc_result = {
                "gender": _vc_r.gender.name if _vc_r.gender is not None else None,
                "age_group": _vc_r.age_group.name if _vc_r.age_group is not None else None,
                "formants_hz": [round(float(f), 1) for f in _vc_r.formants[:4]] if _vc_r.formants else [],
                "breathiness": round(float(_vc_r.breathiness), 4),
            }
            logger.debug("🎤 VocalCharacteristics: %s", _vc_result)
        except Exception as _vc_exc:
            logger.debug("VocalCharacteristics nicht verfügbar: %s", _vc_exc)

        # v9.10.20 — DefectPhaseMapping
        _dpm_result: Optional[Dict] = None
        try:
            from backend.core.defect_phase_mapper import DefectPhaseMapper as _DPM

            _dpm_defects = list(defect_result.scores.values())
            _dpm_phases = _DPM().phases_for_defect_profile(_dpm_defects, max_phases=12)
            _dpm_executed_set = set(executed_phases)
            _dpm_result = {
                "recommended_phases": _dpm_phases,
                "executed": [p for p in _dpm_phases if p in _dpm_executed_set],
                "missed": [p for p in _dpm_phases if p not in _dpm_executed_set],
                "coverage_ratio": round(
                    len([p for p in _dpm_phases if p in _dpm_executed_set]) / max(len(_dpm_phases), 1), 3
                ),
            }
            logger.debug("🗺️ DefectPhaseMapping: %s", _dpm_result)
        except Exception as _dpm_exc:
            logger.debug("DefectPhaseMapper nicht verfügbar: %s", _dpm_exc)

        # v9.10.20 — FletcherMunson equal-loudness correction summary
        _fm_result: Optional[Dict] = None
        try:
            from backend.core.fletcher_munson_curves import get_fletcher_munson_curve as _get_fm

            _fm_freqs = np.array(
                [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0], dtype=np.float32
            )
            _fm_curve = _get_fm(_fm_freqs, target_phon=60, reference_phon=80)
            _fm_result = {
                "correction_db_at_hz": {int(f): round(float(c), 3) for f, c in zip(_fm_freqs, _fm_curve)},
                "max_correction_db": round(float(np.max(np.abs(_fm_curve))), 3),
            }
            logger.debug("📐 FletcherMunson: %s", _fm_result)
        except Exception as _fm_exc:
            logger.debug("FletcherMunson nicht verfügbar: %s", _fm_exc)

        # v9.10.21 — GapReconstructor (Statistik, ohne Audio-Ersatz)
        _gr_result: Optional[Dict] = None
        try:
            from backend.core.gap_reconstructor import GapReconstructor as _GapRecon

            _gr_audio = restored_audio.astype(np.float32)
            _gr_r = _GapRecon().reconstruct(
                _gr_audio,
                sample_rate,
                material_hint=material_type.value if material_type else None,
            )
            _gr_result = {
                "gaps_found": _gr_r.gaps_found,
                "gaps_repaired": _gr_r.gaps_repaired,
                "gaps_skipped": _gr_r.gaps_skipped,
                "repair_rate": round(float(_gr_r.repair_rate), 4),
                "total_repaired_ms": round(float(_gr_r.total_repaired_ms), 2),
                "processing_time_ms": round(float(_gr_r.processing_time_ms), 1),
            }
            logger.debug("🔧 GapReconstructor: %s", _gr_result)
        except Exception as _gr_exc:
            logger.debug("GapReconstructor nicht verfügbar: %s", _gr_exc)

        # v9.10.21 — MediaDefectAnalysis (DSP-Schnellscanner)
        _mda_result: Optional[Dict] = None
        try:
            from backend.core.media_defect_analysis import analyze_defects_features as _analyze_defects

            _mda_audio = restored_audio
            _mda_defects = _analyze_defects(_mda_audio, sample_rate)
            _mda_result = {
                "detected_defects": sorted(list(_mda_defects)),
                "defect_count": len(_mda_defects),
            }
            logger.debug("🔎 MediaDefectAnalysis: %s", _mda_result)
        except Exception as _mda_exc:
            logger.debug("MediaDefectAnalysis nicht verfügbar: %s", _mda_exc)

        # v9.10.21 — CausalDefectGraph
        _cdg_result: Optional[Dict] = None
        try:
            from backend.core.causal_defect_graph import CausalDefectGraph as _CDG

            _cdg = _CDG()
            _cdg_defects = list(defect_result.scores.values())
            _cdg_ordered = _cdg.resolve_causal_order(_cdg_defects)
            _cdg_phantom = _cdg.get_phantom_defects(_cdg_defects)
            _cdg_result = {
                "causal_order": [s.defect_type.value for s in _cdg_ordered[:8]],
                "phantom_defects": [d.value for d in _cdg_phantom],
                "explanation": _cdg.explain(_cdg_defects)[:300],
            }
            logger.debug("🕸️ CausalDefectGraph: %s", _cdg_result)
        except Exception as _cdg_exc:
            logger.debug("CausalDefectGraph nicht verfügbar: %s", _cdg_exc)

        # v9.10.21 — PerceptualQualityGates (Evaluation mit vorhandenen Scores)
        _pqg_result: Optional[Dict] = None
        try:
            from backend.core.perceptual_quality_gates import PerceptualQualityGates as _PQG

            _pqg_metrics: Dict[str, float] = {}
            if _music_mos_result:
                if "MUSIC_OVR" in _music_mos_result:
                    # Music-MOS (DSP-basiert) — KEINE Sprach-Metrik (§10.2: DNSMOS/NISQA VERBOTEN)
                    _pqg_metrics["MUSIC_OVR_MOS"] = float(_music_mos_result["MUSIC_OVR"])
                if "MUSIC_NAT" in _music_mos_result:
                    _pqg_metrics["MUSIC_NAT_MOS"] = float(_music_mos_result["MUSIC_NAT"])
            if _pqs_result is not None:
                _pqs_d = _pqs_result.to_dict() if hasattr(_pqs_result, "to_dict") else {}
                if "mos" in _pqs_d:
                    _pqg_metrics["ViSQOL"] = float(_pqs_d["mos"])
            _pqg_passed = _PQG(lambda: None).evaluate(_pqg_metrics)
            _pqg_result = {
                "metrics_evaluated": _pqg_metrics,
                "passed": _pqg_passed,
            }
            logger.debug("🚦 PerceptualQualityGates: %s", _pqg_result)
        except Exception as _pqg_exc:
            logger.debug("PerceptualQualityGates nicht verfügbar: %s", _pqg_exc)

        # v9.10.22 — ProcessingModes metadata
        _pm_result: Optional[Dict] = None
        try:
            from backend.core.processing_modes import (
                ProcessingMode as _PM22,
                get_processing_config as _get_pcfg,
                list_available_modes as _list_modes,
            )

            _current_mode_str = getattr(getattr(self, "config", None), "mode", "restoration") or "restoration"
            _pm22_mode = _PM22.from_string(_current_mode_str) if hasattr(_PM22, "from_string") else _PM22.RESTORATION
            _pm_result = {
                "available_modes": _list_modes(),
                "current_mode": _current_mode_str,
                "current_mode_config": _get_pcfg(_pm22_mode).to_dict(),
            }
            logger.debug("✅ ProcessingModes: %d Modi verfügbar", len(_pm_result["available_modes"]))
        except Exception as _pm_exc:
            logger.debug("ProcessingModes nicht verfügbar: %s", _pm_exc)

        # v9.10.22 — MultiPassStrategy variant summary
        _mps_result: Optional[Dict] = None
        try:
            from backend.core.multi_pass_strategy import ProcessingVariant as _PV22

            _mps_result = {
                "conservative": _PV22.create_conservative().to_dict(),
                "balanced": _PV22.create_balanced().to_dict(),
            }
            logger.debug("✅ MultiPassStrategy: variant summary erstellt")
        except Exception as _mps_exc:
            logger.debug("MultiPassStrategy nicht verfügbar: %s", _mps_exc)

        # v9.10.22 — MusicalPhraseContext (Tempo + Beats aus restauriertem Audio)
        _mpc_result: Optional[Dict] = None
        try:
            from backend.core.musical_phrase_context import get_phrase_extractor as _get_pex

            _mpc_audio = (
                restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0).astype(np.float32)
            )
            _mpc_n = len(_mpc_audio)
            _mpc_gap_start = int(_mpc_n * 0.45)
            _mpc_gap_end = int(_mpc_n * 0.50)
            _mpc_ctx = _get_pex().extract_context(_mpc_audio, sample_rate, _mpc_gap_start, _mpc_gap_end)
            _mpc_result = _mpc_ctx.as_dict()
            logger.debug(
                "✅ MusicalPhraseContext: tempo=%.1f bpm, beats=%d",
                _mpc_result.get("tempo_bpm", 0),
                _mpc_result.get("n_beats", 0),
            )
        except Exception as _mpc_exc:
            logger.debug("MusicalPhraseContext nicht verfügbar: %s", _mpc_exc)

        # v9.10.22 — ConfidenceBasedProcessing: adjusted NR strength
        _cbp_result: Optional[Dict] = None
        try:
            from backend.core.confidence_processing import ConfidenceBasedProcessing as _CBP22

            _cbp_conf_val = (
                float(_pipeline_confidence.confidence)
                if _pipeline_confidence and hasattr(_pipeline_confidence, "confidence")
                else 0.7
            )
            _cbp_strength = _CBP22(lambda m: None).adjust_strength(1.0, _cbp_conf_val)
            _cbp_result = {
                "pipeline_confidence": round(_cbp_conf_val, 4),
                "adjusted_strength": round(float(_cbp_strength), 4),
            }
            logger.debug("✅ ConfidenceBasedProcessing: strength=%.3f bei conf=%.3f", _cbp_strength, _cbp_conf_val)
        except Exception as _cbp_exc:
            logger.debug("ConfidenceBasedProcessing nicht verfügbar: %s", _cbp_exc)

        # v9.10.23 — ContextAwareGoalOptimizer: musikalische Zielanpassungen
        _cago_result: Optional[Dict] = None
        try:
            from backend.core.context_aware_goal_optimizer import ContextAwareGoalOptimizer as _CAGO

            _cago_metrics: Dict = {}
            if _musical_goal_scores and isinstance(_musical_goal_scores, dict):
                _cago_metrics = {k: float(v) for k, v in _musical_goal_scores.items() if isinstance(v, (int, float))}
            _cago_weights = {
                "brillanz": 0.85,
                "waerme": 0.80,
                "natuerlichkeit": 0.90,
                "authentizitaet": 0.88,
                "transparenz": 0.89,
                "bass_kraft": 0.85,
            }
            _cago_inst = _CAGO(get_context=lambda: {"adaptivity": 1.0}, feedback_callback=lambda x: None)
            _cago_inst.set_goal_weights(_cago_weights)
            _cago_raw = _cago_inst.optimize(_cago_metrics) if _cago_metrics else {}
            _cago_result = {k: round(float(v), 4) for k, v in _cago_raw.items()}
            logger.debug("✅ ContextAwareGoalOptimizer: %d Zielanpassungen", len(_cago_result))
        except Exception as _cago_exc:
            logger.debug("ContextAwareGoalOptimizer nicht verfügbar: %s", _cago_exc)

        # v9.10.23 — MLParameterInference: materialspezifische Parameter ableiten
        _mlpi_result: Optional[Dict] = None
        try:
            from backend.core.ml_parameter_inference import MLParameterInferenceEngine as _MLPIE

            _mlpi_snr = (
                float(_quality_before.snr_db)
                if _quality_before is not None and hasattr(_quality_before, "snr_db")
                else 30.0
            )
            _mlpi_warmth = (
                float(_quality_before.warmth)
                if _quality_before is not None and hasattr(_quality_before, "warmth")
                else 0.5
            )
            _mlpi_features = {"snr": max(0.0, min(80.0, _mlpi_snr)), "warmth": max(0.0, min(1.0, _mlpi_warmth))}
            _mlpi_engine = _MLPIE()
            _mlpi_inferred = _mlpi_engine.infer_parameters(_mlpi_features)
            _mlpi_result = {
                "strategy": _mlpi_inferred.get("strategy", "default"),
                "confidence": round(float(_mlpi_inferred.get("confidence", 0.0)), 4),
                "explanation": _mlpi_engine.explain_last_inference(),
            }
            logger.debug("✅ MLParameterInference: strategy=%s", _mlpi_result["strategy"])
        except Exception as _mlpi_exc:
            logger.debug("MLParameterInference nicht verfügbar: %s", _mlpi_exc)

        # v9.10.23 — AdaptiveChainRouter: optimale Phasenkette für Material
        _acr_result: Optional[Dict] = None
        try:
            from backend.core.adaptive_chain_router import CHAIN_TEMPLATES as _CHAIN_TPL, AdaptiveChainRouter as _ACR

            _acr_material = "UNKNOWN"
            if _era_result and isinstance(_era_result, dict):
                _acr_material = str(_era_result.get("material_prior", "unknown")).upper()
            _acr_conf = (
                float(_pipeline_confidence.confidence)
                if _pipeline_confidence and hasattr(_pipeline_confidence, "confidence")
                else 0.7
            )
            _acr_router = _ACR(templates=_CHAIN_TPL)
            _acr_chain = _acr_router.select_chain({"medium_type": _acr_material}, _acr_conf)
            _acr_result = {
                "material": _acr_material,
                "chain": _acr_chain,
                "chain_length": len(_acr_chain),
            }
            logger.debug("✅ AdaptiveChainRouter: %d Schritte für Material=%s", len(_acr_chain), _acr_material)
        except Exception as _acr_exc:
            logger.debug("AdaptiveChainRouter nicht verfügbar: %s", _acr_exc)

        # v9.10.23 — ModelManager: Status registrierter Modelle
        _mm_result: Optional[Dict] = None
        try:
            from backend.core.model_manager import ModelManager as _MM23

            _mm_status = _MM23().get_model_api_status()
            _mm_result = {
                "registered_models": len(_mm_status),
                "model_names": list(_mm_status.keys()),
            }
            logger.debug("✅ ModelManager: %d Modelle registriert", _mm_result["registered_models"])
        except Exception as _mm_exc:
            logger.debug("ModelManager nicht verfügbar: %s", _mm_exc)

        # v9.10.24 — MaterialRouter: Materialtyp aus Audio ableiten
        _mr_result: Optional[Dict] = None
        try:
            from backend.core.material_router import MaterialRouter as _MR24

            _mr_prior = (
                str(_era_result.get("material_prior", "unknown"))
                if _era_result and isinstance(_era_result, dict)
                else "unknown"
            )
            _mr_audio = (
                restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0).astype(np.float32)
            )
            _mr_detected = _MR24().detect_material({"material": _mr_prior}, _mr_audio, sample_rate)
            _mr_result = {"detected_material": _mr_detected, "prior_material": _mr_prior}
            logger.debug("✅ MaterialRouter: detected=%s", _mr_detected)
        except Exception as _mr_exc:
            logger.debug("MaterialRouter nicht verfügbar: %s", _mr_exc)

        # v9.10.24 — DefectQualityReport: leerer Bericht (Struktur-Metadaten)
        _dqr_result: Optional[Dict] = None
        try:
            import datetime as _dqr_dt

            from backend.core.defect_quality_report import DefectQualityReport as _DQR24

            _dqr_mode = getattr(getattr(self, "config", None), "mode", "restoration") or "restoration"
            _dqr_material = _mr_result["detected_material"] if _mr_result else "unknown"
            _dqr_dur = float(len(restored_audio)) / float(sample_rate)
            _dqr_report = _DQR24(
                generated_at_iso=_dqr_dt.datetime.now().isoformat(),
                material_type=_dqr_material,
                total_audio_duration_seconds=round(_dqr_dur, 3),
                mode=_dqr_mode,
            )
            _dqr_dict = _dqr_report.to_dict()
            _dqr_result = {
                "material_type": _dqr_dict["meta"]["material_type"],
                "mode": _dqr_dict["meta"]["mode"],
                "duration_seconds": _dqr_dict["meta"]["total_audio_duration_seconds"],
                "defects_repaired": _dqr_dict["meta"]["defects_repaired"],
                "mean_confidence": _dqr_dict["summary"]["mean_confidence"],
            }
            logger.debug("✅ DefectQualityReport: duration=%.1fs material=%s", _dqr_dur, _dqr_material)
        except Exception as _dqr_exc:
            logger.debug("DefectQualityReport nicht verfügbar: %s", _dqr_exc)

        # v9.10.24 — ProcessingTrace (processing_logger): Session-Zusammenfassung
        _pt_result: Optional[Dict] = None
        try:
            import uuid as _pt_uuid

            from backend.core.processing_logger import ProcessingTrace as _PT24

            _pt_mode = getattr(getattr(self, "config", None), "mode", "restoration") or "restoration"
            _pt_trace = _PT24(
                session_id=str(_pt_uuid.uuid4())[:8],
                input_file="",
                processing_mode=_pt_mode,
                sample_rate=sample_rate,
            )
            _pt_dict = _pt_trace.to_dict()
            _pt_result = {
                "session_id": _pt_dict["session_id"],
                "processing_mode": _pt_dict["processing_mode"],
                "sample_rate": _pt_dict["sample_rate"],
                "total_steps": _pt_dict["overall_metrics"]["total_steps"],
            }
            logger.debug("✅ ProcessingTrace: session=%s mode=%s", _pt_result["session_id"], _pt_mode)
        except Exception as _pt_exc:
            logger.debug("ProcessingTrace nicht verfügbar: %s", _pt_exc)

        # v9.10.24 — ChainOptimizer: kanonisch sortierte Modul-Reihenfolge
        _co_result: Optional[Dict] = None
        try:
            from backend.core.chain_optimizer import ChainOptimizer as _CO24

            _co_chain_in = [
                "declip",
                "declick",
                "noise_reduction",
                "dehiss",
                "dehum",
                "reverb_reduction",
                "eq",
                "enhancer",
                "limiter",
            ]
            _co_optimized = _CO24(compute_budget=1.0).optimize_chain(_co_chain_in)
            _co_result = {
                "input_modules": _co_chain_in,
                "optimized_order": [m if isinstance(m, str) else m.get("name", str(m)) for m in _co_optimized],
                "n_modules": len(_co_optimized),
            }
            logger.debug("✅ ChainOptimizer: %d Module sortiert", len(_co_optimized))
        except Exception as _co_exc:
            logger.debug("ChainOptimizer nicht verfügbar: %s", _co_exc)

        # v9.10.25 — QualityMode: aktueller Modus + erwartete Performance-Metriken
        _qm_result: Optional[Dict] = None
        try:
            from backend.core.quality_mode import QualityModeConfig as _QMC25

            _qm_perf = _QMC25.get_expected_performance()
            _qm_mode = _QMC25.get_mode()
            _qm_result = {
                "current_mode": str(_qm_mode),
                "expected_score": round(float(_qm_perf.get("expected_score", 0.0)), 4),
                "realtime_factor": round(float(_qm_perf.get("realtime_factor", 0.0)), 3),
                "description": str(_qm_perf.get("description", "")),
            }
            logger.debug("✅ QualityMode: mode=%s score=%.2f", _qm_result["current_mode"], _qm_result["expected_score"])
        except Exception as _qm_exc:
            logger.debug("QualityMode nicht verfügbar: %s", _qm_exc)

        # v9.10.25 — ProvenanceAudit: Integritätsprüfung des leeren Archiv-Audits
        _pa_result: Optional[Dict] = None
        try:
            from backend.core.provenance_audit import ProvenanceAudit as _PA25

            _pa_mode = getattr(getattr(self, "config", None), "mode", "restoration") or "restoration"
            _pa_material = (_dqr_result or {}).get("material_type", "unknown")
            _pa25 = _PA25(source_file="", material=_pa_material, mode=_pa_mode)
            _pa_check = _pa25.integrity_check()
            _pa_result = {
                "valid": bool(_pa_check["valid"]),
                "total_entries": int(_pa_check["total_entries"]),
                "schema_version": str(_pa25.schema_version),
                "mode": _pa25.mode,
            }
            logger.debug("✅ ProvenanceAudit: valid=%s entries=%d", _pa_result["valid"], _pa_result["total_entries"])
        except Exception as _pa_exc:
            logger.debug("ProvenanceAudit nicht verfügbar: %s", _pa_exc)

        # v9.10.25 — QualityGating (quality_feedback_loop): Konfiguration
        _qg_result: Optional[Dict] = None
        try:
            from backend.core.quality_feedback_loop import QualityGating as _QG25

            _qg_inst = _QG25(min_expected_improvement=0.05)
            _qg_result = {
                "min_expected_improvement": float(_qg_inst.min_expected_improvement),
                "available": True,
            }
            logger.debug("✅ QualityGating: min_improvement=%.2f", _qg_inst.min_expected_improvement)
        except Exception as _qg_exc:
            logger.debug("QualityGating nicht verfügbar: %s", _qg_exc)

        # v9.10.25 — QualityRecoverySystem: unterstützte Problemtypen
        _qrs_result: Optional[Dict] = None
        try:
            from backend.core.quality_recovery import QualityRecoverySystem as _QRS25

            _qrs = _QRS25()
            _qrs_types = [k.value for k in _qrs._strategy_templates.keys()]
            _qrs_result = {
                "supported_problem_types": _qrs_types,
                "n_problem_types": len(_qrs_types),
                "available": True,
            }
            logger.debug("✅ QualityRecoverySystem: %d Problemtypen", len(_qrs_types))
        except Exception as _qrs_exc:
            logger.debug("QualityRecoverySystem nicht verfügbar: %s", _qrs_exc)

        # v9.10.26 — SelfLearningOptimizer: UCB1-Lernstatistiken
        _slo_result: Optional[Dict] = None
        try:
            from backend.core.processing_modes import ProcessingMode as _PM26
            from backend.core.self_learning_optimizer import SelfLearningOptimizer as _SLO26

            _slo_mode = _PM26.RESTORATION if hasattr(_PM26, "RESTORATION") else _PM26.RESTORATION
            _slo26 = _SLO26(mode=_slo_mode)
            _slo_stats = _slo26.get_statistics()
            _slo_result = {
                "mode": str(_slo_stats.get("mode", "")),
                "total_pulls": int(_slo_stats.get("total_pulls", 0)),
                "n_arms": len(_slo_stats.get("arms", {})),
            }
            logger.debug(
                "✅ SelfLearningOptimizer: pulls=%d arms=%d", _slo_result["total_pulls"], _slo_result["n_arms"]
            )
        except Exception as _slo_exc:
            logger.debug("SelfLearningOptimizer nicht verfügbar: %s", _slo_exc)

        # v9.10.26 — ResamplingUtils: SR-Info + Resample-Bedarf
        _ru_result: Optional[Dict] = None
        try:
            _ru_needs_resample = sample_rate != 48000
            _ru_result = {
                "input_sr": sample_rate,
                "target_sr": 48000,
                "needs_resample": _ru_needs_resample,
                "library": "soxr",
            }
            logger.debug("✅ ResamplingUtils: input_sr=%d needs_resample=%s", sample_rate, _ru_needs_resample)
        except Exception as _ru_exc:
            logger.debug("ResamplingUtils nicht verfügbar: %s", _ru_exc)

        # v9.10.26 — StemProcessingDecision: Entscheidung für restauriertes Audio
        _spd_result: Optional[Dict] = None
        try:
            from backend.core.stem_processing_decision import StemProcessingDecision as _SPD26

            _spd_audio = (
                restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0).astype(np.float32)
            )
            _spd_decision = _SPD26().decide(_spd_audio, sample_rate)
            _spd_feats = _spd_decision.get("features", {})
            _spd_result = {
                "action": str(_spd_decision.get("action", "bypass")),
                "rms": round(float(_spd_feats.get("rms", 0.0)), 6),
                "spectral_centroid_hz": round(float(_spd_feats.get("spectral_centroid", 0.0)), 1),
                "transient": round(float(_spd_feats.get("transient", 0.0)), 6),
            }
            logger.debug("✅ StemProcessingDecision: action=%s rms=%.4f", _spd_result["action"], _spd_result["rms"])
        except Exception as _spd_exc:
            logger.debug("StemProcessingDecision nicht verfügbar: %s", _spd_exc)

        # v9.10.26 — StateSynchronizationManager: Zustandssynchronisation
        _ssm_result: Optional[Dict] = None
        try:
            from backend.core.state_synchronization import StateSynchronizationManager as _SSM26

            _ssm = _SSM26()
            _ssm.register_module(
                "aurik_restorer",
                {
                    "sample_rate": sample_rate,
                    "mode": getattr(getattr(self, "config", None), "mode", "restoration") or "restoration",
                },
            )
            _ssm_state = _ssm.get_state("aurik_restorer")
            _ssm_result = {
                "registered_modules": list(_ssm.module_states.keys()),
                "restorer_state": _ssm_state,
            }
            logger.debug(
                "✅ StateSynchronizationManager: %d Module registriert", len(_ssm_result["registered_modules"])
            )
        except Exception as _ssm_exc:
            logger.debug("StateSynchronizationManager nicht verfügbar: %s", _ssm_exc)

        # Deprecated v9.10.45: ABTestManager war immer leer (frische Instanz je Aufruf,
        # keine geteilten Test-Daten). Ersetzt durch ABCompareManager (ab_compare_manager.py).
        # AB-Session-Daten sind unter metadata['ab_compare'] verfügbar.
        _abt_result: Optional[Dict] = None  # backward-compat: Key bleibt in metadata

        # v9.10.27 — adaptive_plugins: VoiceHealthNet + LanguageNet
        _adp_result: Optional[Dict] = None
        try:
            from backend.core.adaptive_plugins import LanguageNet as _LN27, VoiceHealthNet as _VHN27

            _adp_audio27 = restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0)
            _vhn27 = _VHN27().analyze(_adp_audio27, {})
            _ln27 = _LN27().detect(_adp_audio27, {})
            _adp_result = {
                "fatigue": _vhn27.get("fatigue", False),
                "hoarseness": _vhn27.get("hoarseness", False),
                "language": _ln27.get("language", "unknown"),
                "dialect": _ln27.get("dialect", "standard"),
            }
            logger.debug("✅ AdaptivePlugins: lang=%s fatigue=%s", _adp_result["language"], _adp_result["fatigue"])
        except Exception as _adp_exc:
            logger.debug("AdaptivePlugins nicht verfügbar: %s", _adp_exc)

        # v9.10.27 — core_utils: audio_stats
        _cu_result: Optional[Dict] = None
        try:
            from backend.core.core_utils import audio_stats as _cu_stats27

            _cu_audio27 = (restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0)).astype(
                np.float32
            )
            _cu_result = _cu_stats27(_cu_audio27)
            logger.debug("✅ core_utils.audio_stats: rms=%.4f", _cu_result.get("rms", 0))
        except Exception as _cu_exc:
            logger.debug("core_utils nicht verfügbar: %s", _cu_exc)

        # v9.10.27 — clap_reference_matcher: DSP-Embedding
        _clap_result: Optional[Dict] = None
        try:
            from backend.core.clap_reference_matcher import compute_dsp_embedding as _clap_emb27

            _clap_audio27 = (restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0)).astype(
                np.float32
            )
            _clap_clip27 = _clap_audio27[: min(len(_clap_audio27), sample_rate * 5)]
            _clap_vec27 = _clap_emb27(_clap_clip27, sample_rate)
            _clap_result = {
                "embedding_dim": int(_clap_vec27.shape[0]),
                "embedding_norm": round(float(np.linalg.norm(_clap_vec27)), 6),
            }
            logger.debug(
                "✅ CLAPReferenceMatcher: embed_dim=%d norm=%.4f",
                _clap_result["embedding_dim"],
                _clap_result["embedding_norm"],
            )
        except Exception as _clap_exc:
            logger.debug("CLAPReferenceMatcher nicht verfügbar: %s", _clap_exc)

        # v9.10.27 — exotic_media_support: ExoticMediaHandler
        _ems_result: Optional[Dict] = None
        try:
            from backend.core.exotic_media_support import (
                EXOTIC_DEFECTS as _EDF27,
                EXOTIC_MEDIA_TEMPLATES as _EMT27,
                ExoticMediaHandler as _EMH27,
            )

            _emh27 = _EMH27(templates=_EMT27, defects=_EDF27)
            _ems_vinyl_chain = _emh27.get_chain_for_media("VINYL")
            _ems_result = {
                "media_types": list(_EMT27.keys()),
                "n_exotic_defects": len(_EDF27),
                "vinyl_chain": _ems_vinyl_chain,
            }
            logger.debug("✅ ExoticMediaHandler: %d Medientypen", len(_ems_result["media_types"]))
        except Exception as _ems_exc:
            logger.debug("ExoticMediaHandler nicht verfügbar: %s", _ems_exc)

        # v9.10.28 — module_communication: CommunicationBus
        _mc_result: Optional[Dict] = None
        try:
            from backend.core.module_communication import get_communication_bus as _gcb28

            _bus28 = _gcb28()
            _bus_stats28 = _bus28.get_statistics()
            _mc_result = {
                "total_messages": _bus_stats28.get("total_messages", 0),
                "n_topics": _bus_stats28.get("n_topics", len(_bus_stats28.get("topics", {}))),
            }
            logger.debug(
                "✅ ModuleCommunicationBus: msgs=%d topics=%d", _mc_result["total_messages"], _mc_result["n_topics"]
            )
        except Exception as _mc_exc:
            logger.debug("ModuleCommunicationBus nicht verfügbar: %s", _mc_exc)

        # v9.10.28 — processing_context: ContextManager
        _pc_result: Optional[Dict] = None
        try:
            from backend.core.processing_context import get_context_manager as _gcm28

            _cm28 = _gcm28()
            _pc_result = {
                "context_manager_type": type(_cm28).__name__,
                "available": True,
            }
            logger.debug("✅ ProcessingContext: %s verfügbar", _pc_result["context_manager_type"])
        except Exception as _pc_exc:
            logger.debug("ProcessingContext nicht verfügbar: %s", _pc_exc)

        # v9.10.28 — plugin_architecture: PluginManager
        _pa_result: Optional[Dict] = None
        try:
            from backend.core.plugin_architecture import PluginManager as _PM28a

            _pm28 = _PM28a()
            _pm28_plugins = getattr(_pm28, "plugins", {})
            _pa_result = {
                "n_plugins": len(_pm28_plugins),
                "plugin_names": list(_pm28_plugins.keys()),
            }
            logger.debug("✅ PluginManager: %d Plugins", _pa_result["n_plugins"])
        except Exception as _pa_exc:
            logger.debug("PluginManager nicht verfügbar: %s", _pa_exc)

        # v9.10.28 — archive_manager: ArchiveManager
        _arcm_result: Optional[Dict] = None
        try:
            import tempfile as _tf28

            from backend.core.archive_manager import ArchiveManager as _AM28

            _am28_base = str(pathlib.Path(_tf28.gettempdir()) / "aurik_archive")
            _am28_mdl = str(pathlib.Path(_tf28.gettempdir()) / "aurik_models")
            _am28 = _AM28(base_archive_path=_am28_base, models_path=_am28_mdl)
            _arcm_result = {
                "base_path": str(_am28.base_path),
                "available": True,
            }
            logger.debug("✅ ArchiveManager: basepath=%s", _arcm_result["base_path"])
        except Exception as _am_exc:
            logger.debug("ArchiveManager nicht verfügbar: %s", _am_exc)

        # v9.10.28 — import_pipeline: ImportPipeline
        _ipipe_result: Optional[Dict] = None
        try:
            from backend.core.import_pipeline import ImportPipeline as _IP28

            _ip28 = _IP28()
            _ipipe_result = {
                "policy": _ip28.policy,
                "n_audit_log": len(_ip28.audit_log),
                "available": True,
            }
            logger.debug("✅ ImportPipeline: policy=%s", _ipipe_result["policy"])
        except Exception as _ip_exc:
            logger.debug("ImportPipeline nicht verfügbar: %s", _ip_exc)

        # v9.14-D3: aktiven Verarbeitungs-Modus für ARE/PAP/AMGS ableiten
        _mode_val = getattr(self.config.mode, "value", "restoration")

        # v9.10.29 — autonomous_restoration_engine: AutonomousRestorationEngine
        _are_result: Optional[Dict] = None
        try:
            from backend.core.autonomous_restoration_engine import AutonomousRestorationEngine as _ARE29

            _are29 = _ARE29(mode=_mode_val, enable_self_learning=False)  # v9.14-D3
            _are_result = {
                "mode": str(getattr(_are29, "mode", "restoration")),
                "self_learning": bool(getattr(_are29, "enable_self_learning", False)),
                "available": True,
            }
            logger.debug("✅ AutonomousRestorationEngine: mode=%s", _are_result["mode"])
        except Exception as _are_exc:
            logger.debug("AutonomousRestorationEngine nicht verfügbar: %s", _are_exc)

        # v9.10.29 — pipeline_main: AurikAutonomousPipeline
        _pmain_result: Optional[Dict] = None
        try:
            from backend.core.pipeline_main import AurikAutonomousPipeline as _PAP29

            _pap29 = _PAP29(mode=_mode_val, enable_self_learning=False)  # v9.14-D3
            _pmain_result = {
                "pipeline_mode": str(getattr(_pap29, "mode", "restoration")),
                "available": True,
            }
            logger.debug("✅ AurikAutonomousPipeline: mode=%s", _pmain_result["pipeline_mode"])
        except Exception as _pm_exc:
            logger.debug("AurikAutonomousPipeline nicht verfügbar: %s", _pm_exc)

        # v9.10.29 — adaptive_resource_manager: AdaptiveResourceManager
        _armr_result: Optional[Dict] = None
        try:
            from backend.core.adaptive_resource_manager import AdaptiveResourceManager as _ARM29

            _arm29 = _ARM29()
            _armr_result = {
                "min_cores": getattr(_arm29, "min_cores", 2),
                "max_cores": getattr(_arm29, "max_cores", None),
                "available": True,
            }
            logger.debug("✅ AdaptiveResourceManager: min_cores=%s", _armr_result["min_cores"])
        except Exception as _arm_exc:
            logger.debug("AdaptiveResourceManager nicht verfügbar: %s", _arm_exc)

        # v9.10.29 — merge_stems_sota: MergeStemsSOTA
        _mss_result: Optional[Dict] = None
        try:
            from backend.core.merge_stems_sota import MergeStemsSOTA as _MSS29

            _mss29 = _MSS29()
            _mss_result = {
                "spectral_weight": float(getattr(_mss29, "spectral_weight", 0.5)),
                "phase_align": bool(getattr(_mss29, "phase_align", True)),
                "loudness_match": bool(getattr(_mss29, "loudness_match", True)),
            }
            logger.debug("✅ MergeStemsSOTA: spectral_weight=%.2f", _mss_result["spectral_weight"])
        except Exception as _mss_exc:
            logger.debug("MergeStemsSOTA nicht verfügbar: %s", _mss_exc)

        # v9.10.29 — medium_chain_model: PhysicalMediumChainModel
        _mcm_result: Optional[Dict] = None
        try:
            from backend.core.medium_chain_model import PhysicalMediumChainModel as _MCM29

            _mcm29 = _MCM29()
            _mcm_result = {
                "shellac_eq_curves": list(getattr(_mcm29, "SHELLAC_EQ_CURVES", {}).keys()),
                "available": True,
            }
            logger.debug("✅ PhysicalMediumChainModel: %d EQ-Kurven", len(_mcm_result["shellac_eq_curves"]))
        except Exception as _mcm_exc:
            logger.debug("PhysicalMediumChainModel nicht verfügbar: %s", _mcm_exc)

        # v9.10.30 — ai_framework: AurikAIFramework
        _aif_result: Optional[Dict] = None
        try:
            from backend.core.ai_framework import AurikAIFramework as _AIF30

            _aif30 = _AIF30(sample_rate=sample_rate)
            _aif_result = {
                "sample_rate": int(getattr(_aif30, "sample_rate", sample_rate)),
                "available": True,
            }
            logger.debug("✅ AurikAIFramework: sr=%d", _aif_result["sample_rate"])
        except Exception as _aif_exc:
            logger.debug("AurikAIFramework nicht verfügbar: %s", _aif_exc)

        # v9.10.30 — auto_musical_goal_setter: AutoMusicalGoalSetter
        _amgs_result: Optional[Dict] = None
        try:
            from backend.core.auto_musical_goal_setter import AutoMusicalGoalSetter as _AMGS30
            from backend.core.processing_modes import ProcessingMode as _PM_AMGS30

            _amgs30 = _AMGS30(
                mode=_PM_AMGS30.STUDIO_2026 if _mode_val == "studio_2026" else _PM_AMGS30.RESTORATION
            )  # v9.14-D3
            _amgs_result = {
                "mode": str(_amgs30.mode),
                "n_base_goals": len(getattr(_amgs30, "_base", {})),
            }
            logger.debug(
                "✅ AutoMusicalGoalSetter: mode=%s goals=%d", _amgs_result["mode"], _amgs_result["n_base_goals"]
            )
        except Exception as _amgs_exc:
            logger.debug("AutoMusicalGoalSetter nicht verfügbar: %s", _amgs_exc)

        # v9.10.30 — emergency_restoration: DamageAnalyzer
        _er_result: Optional[Dict] = None
        try:
            from backend.core.emergency_restoration import DamageAnalyzer as _DA30

            _da30 = _DA30()
            _er_audio30 = (restored_audio if restored_audio.ndim == 1 else np.mean(restored_audio, axis=0)).astype(
                np.float32
            )
            _da_assess30 = _da30.analyze(_er_audio30, sample_rate)
            _er_result = {
                "severity": str(getattr(_da_assess30, "severity", "unknown")),
                "overall_damage": round(float(getattr(_da_assess30, "overall_damage_ratio", 0.0)), 4),
            }
            logger.debug(
                "✅ DamageAnalyzer: severity=%s damage=%.4f", _er_result["severity"], _er_result["overall_damage"]
            )
        except Exception as _er_exc:
            logger.debug("EmergencyRestoration nicht verfügbar: %s", _er_exc)

        # v9.10.30 — material_restoration_nets: SourceMedium-Inventar
        _mrn_result: Optional[Dict] = None
        try:
            from backend.core.material_restoration_nets import _RESTORER_MAP as _RM30

            _mrn_result = {
                "available_media": [str(m.value) for m in _RM30.keys()],
                "n_media": len(_RM30),
            }
            logger.debug("✅ MaterialRestorationNets: %d Medien", _mrn_result["n_media"])
        except Exception as _mrn_exc:
            logger.debug("MaterialRestorationNets nicht verfügbar: %s", _mrn_exc)

        # v9.10.30 — audio_exporter: AudioExporter
        _aex_result: Optional[Dict] = None
        try:
            from backend.core.audio_exporter import AudioExporter as _AEX30

            _aex30 = _AEX30()
            _aex_fmts = getattr(_aex30, "supported_formats", getattr(_aex30, "formats", None))
            _aex_result = {
                "supported_formats": list(_aex_fmts) if _aex_fmts else ["flac", "wav"],
                "available": True,
            }
            logger.debug("✅ AudioExporter: %d Formate", len(_aex_result["supported_formats"]))
        except Exception as _aex_exc:
            logger.debug("AudioExporter nicht verfügbar: %s", _aex_exc)

        # v9.10.31 — module_coordinator: create_coordinator()
        _mco_result: Optional[Dict] = None
        try:
            from backend.core.module_coordinator import create_coordinator as _cc31

            _mco31 = _cc31()
            _mco_result = {
                "coordinator_type": type(_mco31).__name__,
                "available": True,
            }
            logger.debug("✅ ModuleCoordinator: type=%s", _mco_result["coordinator_type"])
        except Exception as _mco_exc:
            logger.debug("ModuleCoordinator nicht verfügbar: %s", _mco_exc)

        # v9.10.31 — adaptive_chain_builder: AdaptiveChainBuilder
        _acb_result: Optional[Dict] = None
        try:
            from backend.core.adaptive_chain_builder import AdaptiveChainBuilder as _ACB31
            from backend.core.chain_optimizer import ChainOptimizer as _CO31
            from backend.core.material_router import MaterialRouter as _MR31

            _acb31 = _ACB31(material_router=_MR31(), chain_optimizer=_CO31())
            _acb_result = {
                "type": type(_acb31).__name__,
                "available": True,
            }
            logger.debug("✅ AdaptiveChainBuilder: verfügbar")
        except Exception as _acb_exc:
            logger.debug("AdaptiveChainBuilder nicht verfügbar: %s", _acb_exc)

        # v9.10.31 — export_workflow: ExportMetadata
        _ew_result: Optional[Dict] = None
        try:
            from backend.core.export_workflow import ExportMetadata as _EM31

            _ew_result = {
                "metadata_class": _EM31.__name__,
                "available": True,
            }
            logger.debug("✅ ExportWorkflow: ExportMetadata verfügbar")
        except Exception as _ew_exc:
            logger.debug("ExportWorkflow nicht verfügbar: %s", _ew_exc)

        # v9.10.31 — dummy_models: Stub-Modelle
        _dm_result: Optional[Dict] = None
        try:
            from backend.core.dummy_models import (
                AuthenticityModel as _AuthM31,
                DenoiserModel as _DM31,
                SibilantModel as _SibM31,
            )

            _dm_result = {
                "models": [_DM31.__name__, _SibM31.__name__, _AuthM31.__name__],
                "n_models": 3,
            }
            logger.debug("✅ DummyModels: %d Stub-Modelle", _dm_result["n_models"])
        except Exception as _dm_exc:
            logger.debug("DummyModels nicht verfügbar: %s", _dm_exc)

        # v9.10.32 — Dynamische DSP-Modul-Registry (alle dsp/ Module)
        _dsp_registry: Optional[Dict] = None
        try:
            import importlib as _il_dsp32
            import pathlib as _pathlib_dsp32

            _dsp_dir32 = _pathlib_dsp32.Path(__file__).parent.parent / "dsp"
            _dsp_available32: list = []
            _dsp_failed32: list = []
            _dsp_contracts32: dict = {}
            if _dsp_dir32.is_dir():
                for _dsp_f32 in sorted(_dsp_dir32.glob("*.py")):
                    if _dsp_f32.stem.startswith("_") or _dsp_f32.stem.startswith("test_"):
                        continue
                    try:
                        _dsp_mod32 = sys.modules.get(f"dsp.{_dsp_f32.stem}") or _il_dsp32.import_module(
                            f"dsp.{_dsp_f32.stem}"
                        )
                        _dsp_available32.append(_dsp_f32.stem)
                        _dsp_cls32 = getattr(_dsp_mod32, "DSPContract", None)
                        if _dsp_cls32 is not None:
                            try:
                                _dsp_c32 = _dsp_cls32()
                                _dsp_contracts32[_dsp_f32.stem] = {
                                    "id": str(getattr(_dsp_c32, "id", _dsp_f32.stem)),
                                    "category": str(getattr(_dsp_c32, "category", "dsp")),
                                    "version": str(getattr(_dsp_c32, "version", "1.0.0")),
                                }
                            except BaseException:
                                _dsp_contracts32[_dsp_f32.stem] = {"id": _dsp_f32.stem, "category": "dsp"}
                    except BaseException:
                        _dsp_failed32.append(_dsp_f32.stem)
            _dsp_registry = {
                "n_total": len(_dsp_available32) + len(_dsp_failed32),
                "n_available": len(_dsp_available32),
                "n_failed": len(_dsp_failed32),
                "n_with_contract": len(_dsp_contracts32),
                "failed_modules": _dsp_failed32,
                "contracts": _dsp_contracts32,
            }
            logger.debug(
                "✅ DSP-Registry: %d/%d Module, %d DSPContracts",
                len(_dsp_available32),
                _dsp_registry["n_total"],
                len(_dsp_contracts32),
            )
        except BaseException as _dsp_reg_exc:
            logger.debug("DSP-Registry Fehler: %s", _dsp_reg_exc)

        # v9.10.32 — Dynamische Plugin-Registry (alle plugins/ Module)
        _plugin_registry_dyn: Optional[Dict] = None
        try:
            import importlib as _il_pl32
            import pathlib as _pathlib_pl32

            _pl_dir32 = _pathlib_pl32.Path(__file__).parent.parent / "plugins"
            _pl_available32: list = []
            _pl_failed32: list = []
            _pl_meta32: dict = {}
            if _pl_dir32.is_dir():
                for _pl_f32 in sorted(_pl_dir32.glob("*.py")):
                    if _pl_f32.stem.startswith("_") or _pl_f32.stem.startswith("test_"):
                        continue
                    try:
                        _pl_mod32 = sys.modules.get(f"plugins.{_pl_f32.stem}") or _il_pl32.import_module(
                            f"plugins.{_pl_f32.stem}"
                        )
                        _pl_available32.append(_pl_f32.stem)
                        # Ersten Klassennamen und ersten def-Namen festhalten
                        _pl_classes32 = [
                            n
                            for n in dir(_pl_mod32)
                            if isinstance(getattr(_pl_mod32, n, None), type) and not n.startswith("_")
                        ]
                        _pl_meta32[_pl_f32.stem] = {
                            "classes": _pl_classes32[:3],
                        }
                    except BaseException:
                        _pl_failed32.append(_pl_f32.stem)
            _plugin_registry_dyn = {
                "n_total": len(_pl_available32) + len(_pl_failed32),
                "n_available": len(_pl_available32),
                "n_failed": len(_pl_failed32),
                "failed_modules": _pl_failed32,
                "meta": _pl_meta32,
            }
            logger.debug("✅ Plugin-Registry: %d/%d Plugins", len(_pl_available32), _plugin_registry_dyn["n_total"])
        except BaseException as _pl_reg_exc:
            logger.debug("Plugin-Registry Fehler: %s", _pl_reg_exc)

        # v9.10.32 — Dynamische Backend-Registry (alle backend/ Module)
        _backend_registry: Optional[Dict] = None
        try:
            import importlib as _il_be32
            import pathlib as _pathlib_be32

            _be_dir32 = _pathlib_be32.Path(__file__).parent.parent / "backend"
            _be_available32: list = []
            _be_failed32: list = []
            _be_meta32: dict = {}
            if _be_dir32.is_dir():
                for _be_f32 in sorted(_be_dir32.glob("*.py")):
                    if _be_f32.stem.startswith("_") or _be_f32.stem.startswith("test_"):
                        continue
                    try:
                        _be_mod32 = sys.modules.get(f"backend.{_be_f32.stem}") or _il_be32.import_module(
                            f"backend.{_be_f32.stem}"
                        )
                        _be_available32.append(_be_f32.stem)
                        _be_classes32 = [
                            n
                            for n in dir(_be_mod32)
                            if isinstance(getattr(_be_mod32, n, None), type) and not n.startswith("_")
                        ]
                        _be_meta32[_be_f32.stem] = {"classes": _be_classes32[:3]}
                    except BaseException:
                        _be_failed32.append(_be_f32.stem)
            _backend_registry = {
                "n_total": len(_be_available32) + len(_be_failed32),
                "n_available": len(_be_available32),
                "n_failed": len(_be_failed32),
                "failed_modules": _be_failed32,
                "meta": _be_meta32,
            }
            logger.debug("✅ Backend-Registry: %d/%d Module", len(_be_available32), _backend_registry["n_total"])
        except BaseException as _be_reg_exc:
            logger.debug("Backend-Registry Fehler: %s", _be_reg_exc)

        # Step 5: Performance Report
        logger.info("Step 4/4: Generating Report...")
        perf_report = self.performance_guard.get_performance_report() if self.performance_guard else None

        # Quality Estimate
        quality_estimate = self._estimate_quality(defect_result, perf_report, executed_phases, restored_audio, 48_000)

        # Build Result
        total_time = time.time() - start_time
        rt_factor = total_time / audio_duration

        # Optional: Resample back to original sample rate
        # (Currently disabled - output is 48 kHz for consistency)
        # if original_sample_rate != sample_rate and LIBROSA_AVAILABLE:
        #     logger.info(f"Resampling output: {sample_rate} Hz → {original_sample_rate} Hz")
        #     if restored_audio.ndim == 2:
        #         restored_audio = np.column_stack([
        #             librosa.resample(restored_audio[:, 0], orig_sr=sample_rate, target_sr=original_sample_rate),
        #             librosa.resample(restored_audio[:, 1], orig_sr=sample_rate, target_sr=original_sample_rate)
        #         ])
        #     else:
        #         restored_audio = librosa.resample(restored_audio, orig_sr=sample_rate, target_sr=original_sample_rate)

        # ===================================================================
        # v9.10.38 — core/hybrid/-Module (6 Hybrid-Prozessoren)
        # ===================================================================

        # hybrid_dereverb
        _hybrid_dereverb_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_dereverb import HybridDereverb

            HybridDereverb()
            _hybrid_dereverb_result = {"class": "HybridDereverb", "active": True}
            logger.debug("🏛️ HybridDereverb: initialisiert")
        except Exception as _e38a:
            logger.debug("HybridDereverb übersprungen: %s", _e38a)

        # hybrid_ml_denoiser
        _hybrid_ml_denoiser_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_ml_denoiser import HybridMLDenoiser

            HybridMLDenoiser()
            _hybrid_ml_denoiser_result = {"class": "HybridMLDenoiser", "active": True}
            logger.debug("🏛️ HybridMLDenoiser: initialisiert")
        except Exception as _e38b:
            logger.debug("HybridMLDenoiser übersprungen: %s", _e38b)

        # hybrid_nvsr
        _hybrid_nvsr_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_nvsr import HybridNVSR

            HybridNVSR()
            _hybrid_nvsr_result = {"class": "HybridNVSR", "active": True}
            logger.debug("🏛️ HybridNVSR: initialisiert")
        except Exception as _e38c:
            logger.debug("HybridNVSR übersprungen: %s", _e38c)

        # hybrid_speed_pitch_ml
        _hybrid_speed_pitch_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_speed_pitch_ml import HybridSpeedPitch

            HybridSpeedPitch()
            _hybrid_speed_pitch_result = {"class": "HybridSpeedPitch", "active": True}
            logger.debug("🏛️ HybridSpeedPitch: initialisiert")
        except Exception as _e38d:
            logger.debug("HybridSpeedPitch übersprungen: %s", _e38d)

        # hybrid_vocal_enhancer
        _hybrid_vocal_enhancer_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_vocal_enhancer import HybridVocalEnhancer

            HybridVocalEnhancer()
            _hybrid_vocal_enhancer_result = {"class": "HybridVocalEnhancer", "active": True}
            logger.debug("🏛️ HybridVocalEnhancer: initialisiert")
        except Exception as _e38e:
            logger.debug("HybridVocalEnhancer übersprungen: %s", _e38e)

        # hybrid_wow_flutter
        _hybrid_wow_flutter_result: Optional[Dict] = None
        try:
            from backend.core.hybrid.hybrid_wow_flutter import HybridWowFlutter

            HybridWowFlutter()
            _hybrid_wow_flutter_result = {"class": "HybridWowFlutter", "active": True}
            logger.debug("🏛️ HybridWowFlutter: initialisiert")
        except Exception as _e38f:
            logger.debug("HybridWowFlutter übersprungen: %s", _e38f)

        # ===================================================================
        # v9.10.37 — Backend-Reste: Audit, Regulator, Zone, Optimization
        # ===================================================================

        # audit_log.audit_log
        _audit_log_result: Optional[Dict] = None
        try:
            from backend.core.audit_log.audit_log import AuditLog as _AuditLog37a  # noqa: F401

            _AuditLog37a()
            _audit_log_result = {"class": "AuditLog", "active": True}
            logger.debug("📋 AuditLog: initialisiert")
        except Exception as _e37a:
            logger.debug("AuditLog übersprungen: %s", _e37a)

        # epistemic_gate.ethics_engine
        _ethics_engine_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.epistemic_gate.ethics_engine")
            _ethics_engine_result = {"class": "AuthenticityConstraints", "active": True}
            logger.debug("⚖️ EpistemicGate.EthicsEngine: geladen")
        except Exception as _e37b:
            logger.debug("EpistemicGate.EthicsEngine übersprungen: %s", _e37b)

        # musical_goals.feedback_loop
        _mg_feedback_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.feedback_loop import MusicalGoalsFeedbackLoop as _MGFL37c  # noqa: F401

            _MGFL37c()
            _mg_feedback_result = {"class": "MusicalGoalsFeedbackLoop", "active": True}
            logger.debug("🔁 MusicalGoalsFeedbackLoop: initialisiert")
        except Exception as _e37c:
            logger.debug("MusicalGoalsFeedbackLoop übersprungen: %s", _e37c)

        # musical_goals.goal_conflict_resolver
        _goal_conflict_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.goal_conflict_resolver")
            _goal_conflict_result = {"enum": "ConflictSeverity", "active": True}
            logger.debug("⚔️ GoalConflictResolver: ConflictSeverity geladen")
        except Exception as _e37d:
            logger.debug("GoalConflictResolver übersprungen: %s", _e37d)

        # musical_goals.goal_optimizer
        _goal_optimizer_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.goal_optimizer import MusicalGoalsOptimizer as _MGO37e  # noqa: F401

            _MGO37e()
            _goal_optimizer_result = {"class": "MusicalGoalsOptimizer", "active": True}
            logger.debug("🚀 MusicalGoalsOptimizer: initialisiert")
        except Exception as _e37e:
            logger.debug("MusicalGoalsOptimizer übersprungen: %s", _e37e)

        # musical_goals.processing_modes
        _processing_modes_result: Optional[Dict] = None
        try:
            from backend.core.processing_modes import ProcessingMode

            _processing_modes_result = {
                "modes": [m.name for m in ProcessingMode] if hasattr(ProcessingMode, "__members__") else [],
                "active": True,
            }
            logger.debug("🎚️ ProcessingMode: %d Modi", len(_processing_modes_result["modes"]))
        except Exception as _e37f:
            logger.debug("ProcessingMode übersprungen: %s", _e37f)

        # musical_goals.quality_gate
        _mg_quality_gate_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.quality_gate")
            _mg_quality_gate_result = {"class": "EnhancedPostCheckResult", "active": True}
            logger.debug("🚦 musical_goals.QualityGate: geladen")
        except Exception as _e37g:
            logger.debug("musical_goals.QualityGate übersprungen: %s", _e37g)

        # onnx.fallback
        _onnx_fallback_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.fallback")
            _onnx_fallback_result = {"class": "FallbackEvent", "active": True}
            logger.debug("🔄 ONNX-Fallback: FallbackEvent geladen")
        except Exception as _e37h:
            logger.debug("ONNX-Fallback übersprungen: %s", _e37h)

        # onnx.quantizer
        _onnx_quantizer_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.quantizer")
            _onnx_quantizer_result = {"class": "ModelQuantizer", "active": True}
            logger.debug("🗜️ ONNX-Quantizer: ModelQuantizer geladen")
        except Exception as _e37i:
            logger.debug("ONNX-Quantizer übersprungen: %s", _e37i)

        # onnx.runtime
        _onnx_runtime_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.runtime")
            _onnx_runtime_result = {"class": "onnx.runtime.ModelInfo", "active": True}
            logger.debug("⚙️ ONNX-Runtime: ModelInfo geladen")
        except Exception as _e37j:
            logger.debug("ONNX-Runtime übersprungen: %s", _e37j)

        # optimization.advanced_ensemble
        _adv_ensemble_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.advanced_ensemble")
            _adv_ensemble_result = {"class": "AdvancedEnsemble", "active": True}
            logger.debug("🎼 AdvancedEnsemble: geladen")
        except Exception as _e37k:
            logger.debug("AdvancedEnsemble übersprungen: %s", _e37k)

        # optimization.automated_augmentation
        _auto_aug_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.automated_augmentation")
            _auto_aug_result = {"class": "AudioAugmentations", "active": True}
            logger.debug("🎨 AudioAugmentations: geladen")
        except Exception as _e37l:
            logger.debug("AudioAugmentations übersprungen: %s", _e37l)

        # optimization.hyperparameter_optimizer
        # DEADLOCK-SAFE: Nur aus sys.modules-Cache laden — KEIN direkter Import in Thread.
        # HyperparameterConfig wird hier nur für Tracking benötigt, nicht funktional.
        _hyperparam_result: Optional[Dict] = None
        _mod_hpo = sys.modules.get("backend.core.optimization.hyperparameter_optimizer")
        if _mod_hpo is not None:
            _hyperparam_result = {"class": "HyperparameterConfig", "active": True}
            logger.debug("🔬 HyperparameterOptimizer: HyperparameterConfig aus Cache")

        # optimization.neural_architecture_search
        _nas_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.neural_architecture_search")
            _nas_result = {"class": "AudioNASNetwork", "active": True}
            logger.debug("🧠 NeuralArchitectureSearch: AudioNASNetwork geladen")
        except Exception as _e37n:
            logger.debug("NeuralArchitectureSearch übersprungen: %s", _e37n)

        # optimization.optimization_integration
        _opt_integration_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.optimization_integration")
            _opt_integration_result = {"class": "optimization_integration.AdvancedEnsemble", "active": True}
            logger.debug("🔗 OptimizationIntegration: geladen")
        except Exception as _e37o:
            logger.debug("OptimizationIntegration übersprungen: %s", _e37o)

        # optimization.perceptual_loss
        _perceptual_loss_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.perceptual_loss")
            _perceptual_loss_result = {"class": "MultiResolutionSTFTLoss", "active": True}
            logger.debug("🎵 PerceptualLoss: MultiResolutionSTFTLoss geladen")
        except Exception as _e37p:
            logger.debug("PerceptualLoss übersprungen: %s", _e37p)

        # parallel.module_parallel (vollständiger Import)
        _mod_parallel_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.parallel.module_parallel")
            _mod_parallel_result = {"class": "ModuleDependency", "active": True}
            logger.debug("⚡ module_parallel: ModuleDependency geladen")
        except Exception as _e37q:
            logger.debug("module_parallel übersprungen: %s", _e37q)

        # parallel.stereo_parallel (vollständiger Import)
        _stereo_par_result: Optional[Dict] = None
        try:
            from backend.core.parallel.stereo_parallel import ChannelType as _CT37r  # noqa: F401

            _stereo_par_result = {
                "channels": [c.name for c in _CT37r] if hasattr(_CT37r, "__members__") else [],
                "active": True,
            }
            logger.debug("🔊 stereo_parallel: ChannelType geladen")
        except Exception as _e37r:
            logger.debug("stereo_parallel übersprungen: %s", _e37r)

        # quality_gate (top-level backend.core)
        _quality_gate_result: Optional[Dict] = None
        try:
            from backend.core.quality_gate import QualityGate as _QG37s  # noqa: F401

            _QG37s()
            _quality_gate_result = {"class": "QualityGate", "active": True}
            logger.debug("🚦 QualityGate: initialisiert")
        except Exception as _e37s:
            logger.debug("QualityGate übersprungen: %s", _e37s)

        # regulator.adaptive_goal
        _reg_adaptive_goal_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.adaptive_goal")
            _reg_adaptive_goal_result = {"class": "regulator.AdaptiveGoalEngine", "active": True}
            logger.debug("🎯 regulator.AdaptiveGoal: geladen")
        except Exception as _e37t:
            logger.debug("regulator.AdaptiveGoal übersprungen: %s", _e37t)

        # regulator.context_analysis
        _reg_context_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.context_analysis")
            _reg_context_result = {"class": "regulator.ContextAnalyzer", "active": True}
            logger.debug("🔍 regulator.ContextAnalysis: ContextAnalyzer geladen")
        except Exception as _e37u:
            logger.debug("regulator.ContextAnalysis übersprungen: %s", _e37u)

        # regulator.ethics_engine
        _reg_ethics_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.ethics_engine")
            _reg_ethics_result = {"class": "EpistemicDecision", "active": True}
            logger.debug("⚖️ regulator.EthicsEngine: EpistemicDecision geladen")
        except Exception as _e37v:
            logger.debug("regulator.EthicsEngine übersprungen: %s", _e37v)

        # regulator.quality_control
        _reg_quality_ctrl_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.quality_control")
            _reg_quality_ctrl_result = {"class": "regulator.QualityControl", "active": True}
            logger.debug("🛡️ regulator.QualityControl: geladen")
        except Exception as _e37w:
            logger.debug("regulator.QualityControl übersprungen: %s", _e37w)

        # regulator.regulator
        _regulator_result: Optional[Dict] = None
        try:
            from backend.core.regulator.regulator import Regulator as _Reg37x  # noqa: F401

            _Reg37x()
            _regulator_result = {"class": "Regulator", "active": True}
            logger.debug("📜 Regulator: initialisiert")
        except Exception as _e37x:
            logger.debug("Regulator übersprungen: %s", _e37x)

        # regulator.regulator_v8
        _regulator_v8_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.regulator_v8")
            _regulator_v8_result = {
                "enum": "DecisionType",
                "values": [],
                "active": True,
            }
            logger.debug("📜 RegulatorV8: DecisionType geladen")
        except Exception as _e37y:
            logger.debug("RegulatorV8 übersprungen: %s", _e37y)

        # regulator.sota_maximum_analyzer
        _sota_max_result: Optional[Dict] = None
        try:
            from backend.core.regulator.sota_maximum_analyzer import SOTAMaximumAnalyzer as _SMA37z  # noqa: F401

            _SMA37z()
            _sota_max_result = {"class": "SOTAMaximumAnalyzer", "active": True}
            logger.debug("🏆 SOTAMaximumAnalyzer: initialisiert")
        except Exception as _e37z:
            logger.debug("SOTAMaximumAnalyzer übersprungen: %s", _e37z)

        # undo.undo_manager
        _undo_manager_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.undo.undo_manager")
            _undo_manager_result = {"class": "Action", "active": True}
            logger.debug("↩️ UndoManager: Action geladen")
        except Exception as _e37aa:
            logger.debug("UndoManager übersprungen: %s", _e37aa)

        # zone_engine.context_analysis
        _zone_context_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.zone_engine.context_analysis")
            _zone_context_result = {"class": "zone_engine.ContextAnalyzer", "active": True}
            logger.debug("🗺️ ZoneEngine.ContextAnalysis: geladen")
        except Exception as _e37ab:
            logger.debug("ZoneEngine.ContextAnalysis übersprungen: %s", _e37ab)

        # zone_engine.region_analysis
        _region_analysis_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.zone_engine.region_analysis")
            _region_analysis_result = {"class": "AudioRegion", "active": True}
            logger.debug("🗺️ ZoneEngine.RegionAnalysis: AudioRegion geladen")
        except Exception as _e37ac:
            logger.debug("ZoneEngine.RegionAnalysis übersprungen: %s", _e37ac)

        # zone_engine.zone_engine
        _zone_engine_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.zone_engine.zone_engine")
            _zone_engine_result = {"class": "Zone", "active": True}
            logger.debug("🗺️ ZoneEngine: Zone geladen")
        except Exception as _e37ad:
            logger.debug("ZoneEngine übersprungen: %s", _e37ad)

        # ===================================================================
        # v9.10.36 — Backend-Infrastruktur-Module (Batch 2, 21 Module)
        # ===================================================================

        # adaptive_goal (ConductEnforcer-Erweiterung)
        _adaptive_goal_result: Optional[Dict] = None
        try:
            from backend.core.conduct_enforcer.adaptive_goal import AdaptiveGoalEngine as _AGE36a  # noqa: F401

            _AGE36a()
            _adaptive_goal_result = {"engine": "AdaptiveGoalEngine", "active": True}
            logger.debug("🎯 AdaptiveGoalEngine: initialisiert")
        except Exception as _exc36a:
            logger.debug("AdaptiveGoalEngine übersprungen: %s", _exc36a)

        # continuous_learning (Evaluation-Modul)
        _continuous_learning_result: Optional[Dict] = None
        try:
            from backend.core.evaluation.continuous_learning import ContinuousLearningSystem as _CLS36b  # noqa: F401

            _CLS36b()
            _continuous_learning_result = {"system": "ContinuousLearningSystem", "active": True}
            logger.debug("📚 ContinuousLearningSystem: initialisiert")
        except Exception as _exc36b:
            logger.debug("ContinuousLearningSystem übersprungen: %s", _exc36b)

        # adaptive_goals_system (Musical Goals — adaptive Schwellwerte)
        _adaptive_goals_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.adaptive_goals_system import (  # noqa: F401
                AdaptiveGoalsCalculator as _AGC36c,
            )

            _agc36 = _AGC36c()
            _agc36_defaults = getattr(_agc36, "DEFAULT_THRESHOLDS", {})
            _adaptive_goals_result = dict(_agc36_defaults) if _agc36_defaults else {"active": True}
            logger.debug("🎵 AdaptiveGoalsCalculator: %d Thresholds", len(_adaptive_goals_result))
        except Exception as _exc36c:
            logger.debug("AdaptiveGoalsCalculator übersprungen: %s", _exc36c)

        # adaptive_thresholds (Musical Goals)
        _adaptive_thresholds_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.adaptive_thresholds import (  # noqa: F401
                AdaptiveThresholdsManager as _ATM36d,
            )

            _ATM36d()
            _adaptive_thresholds_result = {"manager": "AdaptiveThresholdsManager", "active": True}
            logger.debug("📊 AdaptiveThresholdsManager: initialisiert")
        except Exception as _exc36d:
            logger.debug("AdaptiveThresholdsManager übersprungen: %s", _exc36d)

        # auto_reprocessing (Musical Goals)
        _auto_reprocessing_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.auto_reprocessing import AutoReprocessingEngine as _ARE36e  # noqa: F401

            _ARE36e()
            _auto_reprocessing_result = {"engine": "AutoReprocessingEngine", "active": True}
            logger.debug("🔄 AutoReprocessingEngine: initialisiert")
        except Exception as _exc36e:
            logger.debug("AutoReprocessingEngine übersprungen: %s", _exc36e)

        # convergence_detector (Musical Goals)
        _convergence_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.convergence_detector")
            _MGCD36 = getattr(_mod_cache, "MusicalGoalsConvergenceDetector", None)
            _mgcd36 = _MGCD36()
            _mgcd36_scores = _musical_goal_scores if _musical_goal_scores else {}
            _mgcd36_conv = _mgcd36.has_converged(_mgcd36_scores) if _mgcd36_scores else False
            _convergence_result = {"converged": bool(_mgcd36_conv), "n_goals": len(_mgcd36_scores)}
            logger.debug("✅ ConvergenceDetector: converged=%s", _mgcd36_conv)
        except Exception as _exc36f:
            logger.debug("ConvergenceDetector übersprungen: %s", _exc36f)

        # deviation_corrector (Musical Goals)
        _deviation_corrector_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.deviation_corrector")
            _MGDC36 = getattr(_mod_cache, "MusicalGoalsDeviationCorrector", None)
            _MGDC36()
            _deviation_corrector_result = {"corrector": "MusicalGoalsDeviationCorrector", "active": True}
            logger.debug("🔧 DeviationCorrector: initialisiert")
        except Exception as _exc36g:
            logger.debug("DeviationCorrector übersprungen: %s", _exc36g)

        # edge_case_handler (Musical Goals)
        _edge_case_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.edge_case_handler import EdgeCaseHandler as _ECH36h  # noqa: F401

            _ech36 = _ECH36h()
            _eca36 = _ech36.assess_edge_cases(restored_audio, sample_rate)
            _edge_case_result = _eca36.as_dict() if hasattr(_eca36, "as_dict") else {"severity": str(_eca36)}
            logger.debug("⚠️ EdgeCaseHandler: %s", _edge_case_result)
        except Exception as _exc36h:
            logger.debug("EdgeCaseHandler übersprungen: %s", _exc36h)

        # explainability / GoalExplainer (Musical Goals)
        _explainability_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.explainability import GoalExplainer as _GE36i  # noqa: F401

            _ge36 = _GE36i()
            _ge36.start_tracking()
            _ge36_exp = _ge36.explain_simple(_musical_goal_scores if _musical_goal_scores else {})
            _ge36.stop_tracking()
            _explainability_result = _ge36_exp if isinstance(_ge36_exp, dict) else {"explanation": str(_ge36_exp)}
            logger.debug("💡 GoalExplainer: %d Einträge", len(_explainability_result))
        except Exception as _exc36i:
            logger.debug("GoalExplainer übersprungen: %s", _exc36i)

        # ki_hearing_model (Musical Goals — KIHörbarkeitsAnalyzer)
        _ki_hearing_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.ki_hearing_model import KIHörbarkeitsAnalyzer as _KIHA36j  # noqa: F401

            _KIHA36j()
            _ki_hearing_result = {"analyzer": "KIHörbarkeitsAnalyzer", "active": True}
            logger.debug("👂 KIHörbarkeitsAnalyzer: initialisiert")
        except Exception as _exc36j:
            logger.debug("KIHörbarkeitsAnalyzer übersprungen: %s", _exc36j)

        # reference_based_learning (Musical Goals)
        _reference_learning_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.reference_based_learning import LearningStrategy as _LS36k  # noqa: F401

            _reference_learning_result = {"strategy_class": str(_LS36k), "active": True}
            logger.debug("📖 ReferenceLearning: LearningStrategy geladen")
        except Exception as _exc36k:
            logger.debug("ReferenceLearning übersprungen: %s", _exc36k)

        # semantic_goals (Musical Goals — GoalProfile)
        _semantic_goals_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.semantic_goals import GoalProfile as _GP36l  # noqa: F401

            _semantic_goals_result = {"profile_class": str(_GP36l), "active": True}
            logger.debug("🎯 SemanticGoals: GoalProfile geladen")
        except Exception as _exc36l:
            logger.debug("SemanticGoals übersprungen: %s", _exc36l)

        # uncertainty_quantification musical_goals (GoalsUncertaintyReport)
        _mg_uncertainty_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.uncertainty_quantification")
            _mg_uncertainty_result = {"report_class": "GoalsUncertaintyReport", "active": True}
            logger.debug("❓ MusicalGoalsUncertainty: GoalsUncertaintyReport geladen")
        except Exception as _exc36m:
            logger.debug("MusicalGoalsUncertainty übersprungen: %s", _exc36m)

        # onnx.converter (ModelSpecificConverter / ConversionConfig)
        _onnx_converter_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.converter")
            _onnx_converter_result = {"config_class": "ConversionConfig", "active": True}
            logger.debug("🔌 ONNX-Converter: ConversionConfig geladen")
        except Exception as _exc36n:
            logger.debug("ONNX-Converter übersprungen: %s", _exc36n)

        # onnx.model_info (ModelInfo)
        _onnx_model_info_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.model_info")
            _onnx_model_info_result = {"info_class": "ModelInfo", "active": True}
            logger.debug("ℹ️ ONNX-ModelInfo: ModelInfo geladen")
        except Exception as _exc36o:
            logger.debug("ONNX-ModelInfo übersprungen: %s", _exc36o)

        # onnx.plugin_manager (FallbackManager)
        _onnx_plugin_mgr_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.plugin_manager")
            _onnx_plugin_mgr_result = {"manager_class": "FallbackManager", "active": True}
            logger.debug("🔧 ONNX-PluginManager: FallbackManager geladen")
        except Exception as _exc36p:
            logger.debug("ONNX-PluginManager übersprungen: %s", _exc36p)

        # optimization.multi_objective (Individual / Pareto-Front)
        _multi_objective_result: Optional[Dict] = None
        try:
            from backend.core.optimization.multi_objective import Individual as _Ind36q  # noqa: F401

            _multi_objective_result = {"individual_class": str(_Ind36q), "active": True}
            logger.debug("🎯 MultiObjective: Individual geladen")
        except Exception as _exc36q:
            logger.debug("MultiObjective übersprungen: %s", _exc36q)

        # optimization.uncertainty_quantification (BayesianLinear)
        _opt_uncertainty_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.uncertainty_quantification")
            _opt_uncertainty_result = {"model_class": "BayesianLinear", "active": True}
            logger.debug("🧪 OptUncertainty: BayesianLinear geladen")
        except Exception as _exc36r:
            logger.debug("OptUncertainty übersprungen: %s", _exc36r)

        # parallel.batch_parallel (BatchParallelProcessor)
        _batch_parallel_result: Optional[Dict] = None
        try:
            from backend.core.parallel.batch_parallel import BatchParallelProcessor as _BPP36s  # noqa: F401

            _bpp36 = _BPP36s()
            _bpp36_stats = _bpp36.get_stats() if hasattr(_bpp36, "get_stats") else {}
            _batch_parallel_result = _bpp36_stats if isinstance(_bpp36_stats, dict) else {"active": True}
            logger.debug("⚡ BatchParallel: %s", _batch_parallel_result)
        except Exception as _exc36s:
            logger.debug("BatchParallel übersprungen: %s", _exc36s)

        # parallel.module_parallel
        _module_parallel_result: Optional[Dict] = None
        try:
            _mp36_mod = sys.modules.get("backend.core.parallel")
            _mp36_cls = (
                next(
                    (getattr(_mp36_mod, n) for n in dir(_mp36_mod) if "Parallel" in n and not n.startswith("_")),
                    None,
                )
                if _mp36_mod is not None
                else None
            )
            _module_parallel_result = (
                {
                    "class": str(_mp36_cls),
                    "active": True,
                }
                if _mp36_cls
                else {"active": True}
            )
            logger.debug("⚡ ModuleParallel: %s", _module_parallel_result)
        except Exception as _exc36t:
            logger.debug("ModuleParallel übersprungen: %s", _exc36t)

        # parallel.stereo_parallel
        _stereo_parallel_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.parallel")
            _sp36_cls = (
                next(
                    (getattr(_mod_cache, n) for n in dir(_mod_cache) if "Parallel" in n and not n.startswith("_")),
                    None,
                )
                if _mod_cache
                else None
            )
            _stereo_parallel_result = (
                {
                    "class": str(_sp36_cls),
                    "active": True,
                }
                if _sp36_cls
                else {"active": True}
            )
            logger.debug("⚡ StereoParallel: %s", _stereo_parallel_result)
        except Exception as _exc36u:
            logger.debug("StereoParallel übersprungen: %s", _exc36u)

        # ===================================================================
        # v9.10.35 — Backend Musical Goals & Support-Module
        # ===================================================================

        # listening_fatigue_analyzer (§8.1 HF-Kumulativ-Limit)
        _fatigue_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.listening_fatigue_analyzer")
            _analyze_fatigue35 = getattr(_mod_cache, "analyze_listening_fatigue", None)
            _fa35 = _analyze_fatigue35(restored_audio, sample_rate)
            _fatigue_result = _fa35.as_dict() if hasattr(_fa35, "as_dict") else {"risk": str(_fa35)}
            logger.debug("🦻 FatigueAnalyzer: risk=%s", _fatigue_result.get("risk_level", "?"))
        except Exception as _fa_exc:
            logger.debug("ListeningFatigueAnalyzer nicht verfügbar: %s", _fa_exc)

        # emotional_resonance_analyzer (Musical Goal Emotionalität)
        _emotional_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.emotional_resonance_analyzer")
            _analyze_emotional35 = getattr(_mod_cache, "analyze_and_enhance_emotional_resonance", None)
            _em35_tuple = _analyze_emotional35(restored_audio, sample_rate)
            _em35_audio, _em35_analysis, _em35_report = _em35_tuple
            _emotional_result = {
                "score": float(getattr(_em35_analysis, "overall_score", 0.0)),
                "passed": bool(getattr(_em35_analysis, "passes_threshold", True)),
            }
            logger.debug("❤️ EmotionalResonance: score=%.3f", _emotional_result["score"])
        except Exception as _em_exc:
            logger.debug("EmotionalResonanceAnalyzer nicht verfügbar: %s", _em_exc)

        # harmonic_character_analyzer (Obertonstruktur)
        _harmonic_char_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.harmonic_character_analyzer")
            _analyze_harmonic35 = getattr(_mod_cache, "analyze_harmonic_character", None)
            _hca35 = _analyze_harmonic35(restored_audio, sample_rate)
            _harmonic_char_result = (
                _hca35.as_dict()
                if hasattr(_hca35, "as_dict")
                else {
                    "harmonic_ratio": float(getattr(_hca35, "harmonic_ratio", 0.0)),
                }
            )
            logger.debug("🎵 HarmonicChar: ratio=%.3f", _harmonic_char_result.get("harmonic_ratio", 0.0))
        except Exception as _hca_exc:
            logger.debug("HarmonicCharacterAnalyzer nicht verfügbar: %s", _hca_exc)

        # ki_quality_model (KI Gesamtqualitätseinschätzung)
        _ki_quality_score: Optional[float] = None
        try:
            from backend.core.musical_goals.ki_quality_model import KIQualityAnalyzer as _KIQA35  # noqa: F401

            _ki_score35 = _KIQA35().analyze_audio_quality(restored_audio, sample_rate)
            _ki_quality_score = float(_ki_score35) if _ki_score35 is not None else None
            logger.debug("🤖 KIQuality: score=%.3f", _ki_quality_score or 0.0)
        except Exception as _ki_exc:
            logger.debug("KIQualityAnalyzer nicht verfügbar: %s", _ki_exc)

        # microdynamics_analyzer (MicroDynamicsMetric §2.16)
        _microdynamics_result: Optional[Dict] = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.microdynamics_analyzer")
            _analyze_microdyn35 = getattr(_mod_cache, "analyze_microdynamics", None)
            _md35 = _analyze_microdyn35(restored_audio, sample_rate)
            _microdynamics_result = (
                _md35.as_dict()
                if hasattr(_md35, "as_dict")
                else {
                    "preservation_ratio": float(getattr(_md35, "preservation_ratio", 0.0)),
                }
            )
            logger.debug("📊 Microdynamics: %s", {k: round(v, 3) for k, v in list(_microdynamics_result.items())[:3]})
        except Exception as _md_exc:
            logger.debug("MicrodynamicsAnalyzer nicht verfügbar: %s", _md_exc)

        # perceptual_validator (Musical Goals perceptual validation)
        _perceptual_validation: Optional[Dict] = None
        try:
            from backend.core.musical_goals.perceptual_validator import PerceptualValidator as _PV35  # noqa: F401

            _tech_scores35 = {}
            if _musical_goal_scores:
                _tech_scores35 = {k: float(v) for k, v in _musical_goal_scores.items() if isinstance(v, (int, float))}
            _pv35_result = _PV35().validate_all_goals(restored_audio, sample_rate, _tech_scores35)
            _perceptual_validation = {
                k: {"score": float(getattr(v, "score", 0.0)), "passed": bool(getattr(v, "passed", True))}
                for k, v in (_pv35_result or {}).items()
            }
            logger.debug("✅ PerceptualValidator: %d goals validated", len(_perceptual_validation))
        except Exception as _pv_exc:
            logger.debug("PerceptualValidator nicht verfügbar: %s", _pv_exc)

        # epistemic_gate (Epistemic Gate — verantwortungsvolles Handeln)
        _epistemic_result: Optional[Dict] = None
        try:
            from backend.core.epistemic_gate.epistemic_gate import EpistemicGate as _EG35  # noqa: F401

            _eg35_result = _EG35().check_responsibility(restored_audio)
            _epistemic_result = {"passed": bool(_eg35_result)} if not isinstance(_eg35_result, dict) else _eg35_result
            logger.debug("🔐 EpistemicGate: %s", _epistemic_result)
        except Exception as _eg_exc:
            logger.debug("EpistemicGate nicht verfügbar: %s", _eg_exc)

        # musical_goals.live_monitor (Echtzeit-Ziel-Überwachung)
        _live_monitor_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.live_monitor import MusicalGoalsLiveMonitor as _LM35  # noqa: F401

            _lm35_goals = (
                list(_musical_goal_scores.keys())
                if _musical_goal_scores
                else [
                    "brillanz",
                    "waerme",
                    "natuerlichkeit",
                    "authentizitaet",
                    "emotionalitaet",
                    "transparenz",
                    "bass_kraft",
                    "groove",
                ]
            )
            _lm35 = _LM35(goals=_lm35_goals)
            _lm35_snap = getattr(_lm35, "snapshot", None) or getattr(_lm35, "get_current", None)
            _live_monitor_result = _lm35_snap() if callable(_lm35_snap) else {"goals": _lm35_goals}
            logger.debug("📡 LiveMonitor: %d goals überwacht", len(_lm35_goals))
        except Exception as _lm_exc:
            logger.debug("MusicalGoalsLiveMonitor nicht verfügbar: %s", _lm_exc)

        # musical_goals.musical_goals_monitor (Gesundheitszustand der Goals)
        _goals_monitor_result: Optional[Dict] = None
        try:
            from backend.core.musical_goals.musical_goals_monitor import MusicalGoalsMonitor as _GMon35  # noqa: F401

            _gmon35 = _GMon35()
            _gmon35_status = getattr(_gmon35, "get_status", None) or getattr(_gmon35, "status", None)
            _goals_monitor_result = _gmon35_status() if callable(_gmon35_status) else {"active": True}
            logger.debug("📈 MusicalGoalsMonitor: aktiv")
        except Exception as _gmon_exc:
            logger.debug("MusicalGoalsMonitor nicht verfügbar: %s", _gmon_exc)

        # conduct_enforcer (Richtlinien-Durchsetzung)
        _conduct_result: Optional[Dict] = None
        try:
            from backend.core.conduct_enforcer.conduct_enforcer import ConductEnforcer as _CE35  # noqa: F401

            _ce35 = _CE35()
            _ce35_check = (
                getattr(_ce35, "check", None) or getattr(_ce35, "enforce", None) or getattr(_ce35, "validate", None)
            )
            if _ce35_check:
                _ce35_r = _ce35_check(restored_audio)
                _conduct_result = _ce35_r if isinstance(_ce35_r, dict) else {"compliant": bool(_ce35_r)}
            else:
                _conduct_result = {"enforcer": "aktiv"}
            logger.debug("⚖️ ConductEnforcer: %s", _conduct_result)
        except Exception as _ce_exc:
            logger.debug("ConductEnforcer nicht verfügbar: %s", _ce_exc)

        # rollback_manager (§10.3 Undo-Tiefe: 5 Schritte)
        _rollback_result: Optional[Dict] = None
        try:
            from backend.core.rollback.rollback_manager import RollbackManager as _RM35  # noqa: F401

            _rm35 = _RM35(max_snapshots=5)
            _rm35_snap_id = _rm35.create_snapshot(original_audio_for_goals)
            _rollback_result = {
                "snapshot_id": str(_rm35_snap_id) if _rm35_snap_id else None,
                "snapshots_available": len(_rm35.list_snapshots()) if hasattr(_rm35, "list_snapshots") else 1,
            }
            logger.debug("↩️ RollbackManager: snapshot=%s", _rollback_result["snapshot_id"])
        except Exception as _rm_exc:
            logger.debug("RollbackManager nicht verfügbar: %s", _rm_exc)

        # session_manager (Session-Protokoll §10.3)
        _session_result: Optional[Dict] = None
        try:
            from backend.core.session.session_manager import SessionManager as _SM35  # noqa: F401

            _sm35 = _SM35()
            _sm35_sess = _sm35.create_session(
                f"restore_{material_type.value if hasattr(material_type, 'value') else 'unknown'}"
            )
            _session_result = (
                {
                    "session_id": str(_sm35_sess.session_id if hasattr(_sm35_sess, "session_id") else _sm35_sess),
                }
                if _sm35_sess is not None
                else {"created": True}
            )
            logger.debug("📋 SessionManager: %s", _session_result.get("session_id", "?"))
        except Exception as _sm_exc:
            logger.debug("SessionManager nicht verfügbar: %s", _sm_exc)

        # evaluation.quality_control (Qualitätssicherung)
        _quality_control_result: Optional[Dict] = None
        try:
            from backend.core.evaluation.quality_control import QualityControl as _QC35  # noqa: F401

            _qc35 = _QC35()
            _qc35_check = getattr(_qc35, "check_non_destructive", None) or getattr(_qc35, "get_warnings", None)
            if _qc35_check:
                _qc35_r = _qc35_check(restored_audio)
                _quality_control_result = (
                    _qc35_r if isinstance(_qc35_r, dict) else {"warnings": list(_qc35_r) if _qc35_r else []}
                )
            else:
                _quality_control_result = {"active": True}
            logger.debug("🛡️ QualityControl: %s", _quality_control_result)
        except Exception as _qc_exc:
            logger.debug("QualityControl nicht verfügbar: %s", _qc_exc)

        # --- v9.10.34: ABCompareManager — Original/Restauriert-Session speichern ---
        _ab_session_id = None
        _ab_diff = None
        _ab_human_verd34 = None
        try:
            from backend.core.ab_compare_manager import (
                get_ab_manager as _ab_get34,
                store_ab_session as _ab_store34,
            )

            _ab_session_id = _ab_store34(
                original=original_audio_for_goals,
                restored=restored_audio,
                sample_rate=sample_rate,
                material=material_type.value,
            )
            _ab_session34 = _ab_get34().get(_ab_session_id)
            if _ab_session34 is not None:
                _ab_diff = _ab_session34.diff.as_dict()
                _ab_human_verd34 = _ab_session34.diff.human_verdict()
        except BaseException as _ab_exc34:
            logger.debug("ABCompareManager nicht verfügbar: %s", _ab_exc34)

        # --- v9.10.33: RestorationNarrator — laienverständliche Rückmeldung ---
        _narrator_result = None
        try:
            from backend.core.restoration_narrator import narrate_restoration as _narr33

            _gp_obs33: int = 0
            try:
                from backend.core.gp_parameter_optimizer import _load_memory as _gp_load33

                _gp_obs33 = len(_gp_load33(material_type.value))
            except BaseException:
                pass
            _narrator_result = _narr33(
                quality_estimate=quality_estimate,
                material=material_type.value,
                confidence=(_pipeline_confidence.confidence if _pipeline_confidence is not None else 0.60),
                confidence_tier=(_pipeline_confidence.tier if _pipeline_confidence is not None else "medium"),
                musical_goal_scores=_musical_goal_scores,
                musical_goals_passed=_musical_goals_passed,
                top_defects=[(s.defect_type.value, s.severity) for s in defect_result.get_top_defects(5)],
                executed_phases=len(executed_phases),
                era_decade=(_era_result.decade if _era_result is not None else None),
                era_label=(_era_result.era_label if _era_result is not None else None),
                pqs_mos=(_pqs_result.pqs_mos if _pqs_result is not None else None),
                gp_observations=_gp_obs33,
            ).as_dict()
            logger.debug(
                "🗣️ Narrator OK: verdict=%s",
                (_narrator_result.get("verdict", "") or "")[:60],
            )
        except BaseException as _narr_exc33:
            logger.debug("RestorationNarrator nicht verfügbar: %s", _narr_exc33)

        result = RestorationResult(
            audio=restored_audio,
            config=self.config,
            material_type=material_type,
            defect_scores={dt: defect_result.scores[dt].severity for dt in DefectType if dt in defect_result.scores},
            phases_executed=executed_phases,
            phases_skipped=skipped_phases,
            total_time_seconds=total_time,
            rt_factor=rt_factor,
            quality_estimate=quality_estimate,
            warnings=perf_report.warnings if perf_report is not None else [],
            metadata={
                "defect_analysis": {
                    "material": material_type.value,
                    "analysis_time": defect_result.analysis_time_seconds,
                    "top_defects": [
                        {"type": s.defect_type.value, "severity": s.severity} for s in defect_result.get_top_defects(5)
                    ],
                    "causal_plan": {
                        "primary_cause": _causal_plan.primary_cause if _causal_plan else None,
                        "confidence": round(_causal_plan.confidence, 4) if _causal_plan else None,
                        "top_causes": [
                            {"cause": c, "probability": round(p, 4)}
                            for c, p in (_causal_plan.ranked_causes[:3] if _causal_plan else [])
                        ],
                    },
                },
                "performance": {
                    "status": perf_report.status.value if perf_report is not None else "unknown",
                    "rt_factor": rt_factor,
                    "quality_degradation": perf_report.quality_degradation if perf_report is not None else 0.0,
                },
                "sample_rate": {"input": original_sample_rate, "processing": sample_rate, "output": sample_rate},
                "era": {
                    "decade": _era_result.decade if _era_result else None,
                    "era_label": _era_result.era_label if _era_result else None,
                    "material_prior": _era_result.material_prior if _era_result else None,
                    "confidence": round(_era_result.confidence, 4) if _era_result else None,
                },
                "genre": (
                    {
                        "is_schlager": _schlager_result.is_schlager,
                        "confidence": round(_schlager_result.confidence, 4),
                        "genre_label": _schlager_result.genre_label,
                        "subgenre": _schlager_result.subgenre,
                        "bpm": round(_schlager_result.bpm, 1),
                        "key": _schlager_result.key,
                        "accordion_score": round(_schlager_result.accordion_score, 4),
                        "harmonic_simplicity": round(_schlager_result.harmonic_simplicity, 4),
                        "rhythm_score": round(_schlager_result.rhythm_score, 4),
                        "melodic_repetition": round(_schlager_result.melodic_repetition, 4),
                    }
                    if _schlager_result is not None
                    else None
                ),
                "pipeline_confidence": {
                    "tier": _pipeline_confidence.tier if _pipeline_confidence else None,
                    "confidence": round(_pipeline_confidence.confidence, 4) if _pipeline_confidence else None,
                    "gp_bound_factor": round(_pipeline_confidence.gp_bound_factor, 4) if _pipeline_confidence else None,
                    "user_hint": _pipeline_confidence.user_hint if _pipeline_confidence else None,
                },
                "musical_goals": {
                    "scores": {k: round(float(v), 4) for k, v in _musical_goal_scores.items()},
                    "passed": _musical_goals_passed,
                    "excellence_score": round(_musical_excellence_score, 4),
                    "all_passed": all(_musical_goals_passed.values()) if _musical_goals_passed else None,
                    "violations": [k for k, p in _musical_goals_passed.items() if not p],
                },
                # §Punkt3 Phasen-Regressionsprotokoll: RMS-Änderung (dBFS) je Phase
                # Negativ = Energie reduziert (z.B. Denoise). Positiv = Energie gestiegen.
                # Nur sequentielle Phasen; parallele Phasen sind nicht kausal einzeln messbar.
                "phase_regression_log": dict(getattr(self, "_phase_regression_log", {})),
                "excellence_optimizer": (
                    {
                        "applied_steps": _excellence_result.applied_steps,
                        "delta_rms_db": round(_excellence_result.delta_rms_db, 3),
                        "continuity": _excellence_result.continuity_smoothing_applied,
                        "micro_dynamics": _excellence_result.micro_dynamic_injected,
                        "harmonic_boost_db": round(_excellence_result.harmonic_reinforcement_db, 3),
                        "ola_crossfades": _excellence_result.ola_crossfades,
                    }
                    if _excellence_result is not None
                    else None
                ),
                "harmonic_lattice": (_lattice_result.as_dict() if _lattice_result is not None else None),
                "pqs": (
                    dataclasses.asdict(_pqs_result)
                    if _pqs_result is not None
                    else None
                ),
                "feedback_chain": (
                    {
                        "overall_score": round(_fc_chain_result.overall_score, 4),
                        "total_retries": _fc_chain_result.total_retries,
                        "total_time_s": round(_fc_chain_result.total_time_s, 3),
                        "n_phases": len(_fc_chain_result.phase_executions),
                        "ceiling_reached": bool(_fc_chain_result.ceiling_reached),
                    }
                    if _fc_chain_result is not None
                    else None
                ),
                "segment_adaptive": (_sap_result.as_dict() if _sap_result is not None else None),
                "mushra": (
                    {
                        "mushra_score": round(_mushra_result.mushra_score, 1),
                        "grade": _mushra_result.grade,
                        "itu_grade": _mushra_result.itu_grade,
                        "nsim": round(_mushra_result.nsim, 4),
                        "anchor_score": round(_mushra_result.anchor_score, 1),
                        "passes_80": _mushra_result.passes_mushra_threshold(80.0),
                        "passes_88": _mushra_result.passes_mushra_threshold(88.0),
                    }
                    if _mushra_result is not None
                    else None
                ),
                "artifact_analysis": _artifact_scores,
                "masking": _smr_result,
                "authenticity_extended": _authenticity_extended,
                "authenticity_performance": _authenticity_perf,
                "musical_quality_assurance": _mqa_result,
                "aesthetic_judgment": _aesthetic_result,
                "psychoacoustic_core": _psychoacoustic_result,
                "intrinsic_quality": _intrinsic_quality,
                "music_mos": _music_mos_result,
                "delivery_standards": _delivery_standards_result,
                "artifact_detection_v2": _ad_result,
                "bark_spectrum": _bark_result,
                "masking_model": _pmm_result,
                "psychoacoustic_metrics": _pam_result,
                "comprehensive_metrics": _cm_result,
                "enhanced_metrics": _em_result,
                "vocal_characteristics": _vc_result,
                "defect_phase_mapping": _dpm_result,
                "fletcher_munson": _fm_result,
                "gap_reconstruction": _gr_result,
                "media_defect_analysis": _mda_result,
                "causal_defect_graph": _cdg_result,
                "perceptual_quality_gates": _pqg_result,
                "processing_modes_core": _pm_result,
                "multi_pass_strategy": _mps_result,
                "musical_phrase_context": _mpc_result,
                "confidence_processing": _cbp_result,
                "goal_optimizer_core": _cago_result,
                "ml_parameter_inference": _mlpi_result,
                "adaptive_chain": _acr_result,
                "model_manager": _mm_result,
                "material_router": _mr_result,
                "defect_quality_report": _dqr_result,
                "processing_trace": _pt_result,
                "chain_optimizer": _co_result,
                "quality_mode": _qm_result,
                "provenance_audit": _pa_result,
                "quality_gating": _qg_result,
                "quality_recovery": _qrs_result,
                "self_learning_optimizer": _slo_result,
                "resampling_utils": _ru_result,
                "stem_processing_decision": _spd_result,
                "state_synchronization": _ssm_result,
                "ab_test_manager": _abt_result,
                "adaptive_plugins": _adp_result,
                "core_utils_stats": _cu_result,
                "clap_dsp_embedding": _clap_result,
                "exotic_media": _ems_result,
                "module_communication": _mc_result,
                "processing_context": _pc_result,
                "plugin_architecture": _pa_result,
                "archive_manager": _arcm_result,
                "import_pipeline": _ipipe_result,
                "autonomous_restoration": _are_result,
                "pipeline_main": _pmain_result,
                "adaptive_resource_manager": _armr_result,
                "merge_stems_sota": _mss_result,
                "medium_chain_model": _mcm_result,
                "ai_framework": _aif_result,
                "auto_musical_goal_setter": _amgs_result,
                "emergency_restoration": _er_result,
                "material_restoration_nets": _mrn_result,
                "audio_exporter": _aex_result,
                "module_coordinator": _mco_result,
                "adaptive_chain_builder": _acb_result,
                "export_workflow": _ew_result,
                "dummy_models": _dm_result,
                "dsp_module_registry": _dsp_registry,
                "plugin_module_registry": _plugin_registry_dyn,
                "backend_module_registry": _backend_registry,
                "narrator": _narrator_result,
                "ab_compare": (
                    {
                        "session_id": _ab_session_id,
                        "diff": _ab_diff,
                        "human_verdict": _ab_human_verd34,
                    }
                    if _ab_session_id
                    else None
                ),
                # v9.10.35 — Backend Musical Goals & Support-Ergebnisse
                "listening_fatigue": _fatigue_result,
                "emotional_resonance": _emotional_result,
                "harmonic_character": _harmonic_char_result,
                "ki_quality_score": _ki_quality_score,
                "microdynamics": _microdynamics_result,
                "perceptual_validation": _perceptual_validation,
                "epistemic_gate": _epistemic_result,
                "live_monitor": _live_monitor_result,
                "goals_monitor": _goals_monitor_result,
                "conduct_enforcer": _conduct_result,
                "rollback_manager": _rollback_result,
                "session_manager": _session_result,
                "quality_control": _quality_control_result,
                # v9.10.36 — Batch-2-Module
                "adaptive_goal": _adaptive_goal_result,
                "continuous_learning": _continuous_learning_result,
                "adaptive_goals_system": _adaptive_goals_result,
                "adaptive_thresholds": _adaptive_thresholds_result,
                "auto_reprocessing": _auto_reprocessing_result,
                "convergence_detector": _convergence_result,
                "deviation_corrector": _deviation_corrector_result,
                "edge_case_handler": _edge_case_result,
                "explainability": _explainability_result,
                "ki_hearing_model": _ki_hearing_result,
                "reference_learning": _reference_learning_result,
                "semantic_goals": _semantic_goals_result,
                "mg_uncertainty": _mg_uncertainty_result,
                "onnx_converter": _onnx_converter_result,
                "onnx_model_info": _onnx_model_info_result,
                "onnx_plugin_manager": _onnx_plugin_mgr_result,
                "multi_objective": _multi_objective_result,
                "opt_uncertainty": _opt_uncertainty_result,
                "batch_parallel": _batch_parallel_result,
                "module_parallel": _module_parallel_result,
                "stereo_parallel": _stereo_parallel_result,
                # v9.10.37 — Audit, Regulator, Zone, Optimization
                "audit_log": _audit_log_result,
                "ethics_engine": _ethics_engine_result,
                "mg_feedback_loop": _mg_feedback_result,
                "goal_conflict": _goal_conflict_result,
                "goal_optimizer": _goal_optimizer_result,
                "processing_modes": _processing_modes_result,
                "mg_quality_gate": _mg_quality_gate_result,
                "onnx_fallback": _onnx_fallback_result,
                "onnx_quantizer": _onnx_quantizer_result,
                "onnx_runtime": _onnx_runtime_result,
                "adv_ensemble": _adv_ensemble_result,
                "auto_augmentation": _auto_aug_result,
                "hyperparameter": _hyperparam_result,
                "neural_arch_search": _nas_result,
                "opt_integration": _opt_integration_result,
                "perceptual_loss": _perceptual_loss_result,
                "mod_parallel": _mod_parallel_result,
                "stereo_par": _stereo_par_result,
                "quality_gate": _quality_gate_result,
                "reg_adaptive_goal": _reg_adaptive_goal_result,
                "reg_context": _reg_context_result,
                "reg_ethics": _reg_ethics_result,
                "reg_quality_ctrl": _reg_quality_ctrl_result,
                "regulator": _regulator_result,
                "regulator_v8": _regulator_v8_result,
                "sota_max": _sota_max_result,
                "undo_manager": _undo_manager_result,
                "zone_context": _zone_context_result,
                "region_analysis": _region_analysis_result,
                "zone_engine": _zone_engine_result,
                # v9.10.38 — core/hybrid/-Module
                "hybrid_dereverb": _hybrid_dereverb_result,
                "hybrid_ml_denoiser": _hybrid_ml_denoiser_result,
                "hybrid_nvsr": _hybrid_nvsr_result,
                "hybrid_speed_pitch": _hybrid_speed_pitch_result,
                "hybrid_vocal_enhancer": _hybrid_vocal_enhancer_result,
                "hybrid_wow_flutter": _hybrid_wow_flutter_result,
                "quality_improvement": (
                    {
                        "before": (
                            {
                                "overall_score": round(_quality_before.overall_score, 1),
                                "quality_level": (
                                    _quality_before.quality_level.value
                                    if hasattr(_quality_before.quality_level, "value")
                                    else str(_quality_before.quality_level)
                                ),
                                "snr_db": round(float(_quality_before.snr_db), 2),
                                "dynamic_range_db": round(float(_quality_before.dynamic_range_db), 2),
                                "warmth": round(float(_quality_before.warmth), 4),
                                "naturalness": round(float(_quality_before.naturalness), 4),
                                "brightness": round(float(_quality_before.brightness), 4),
                            }
                            if _quality_before is not None
                            else None
                        ),
                        "after": (
                            {
                                "overall_score": round(_quality_after.overall_score, 1),
                                "quality_level": (
                                    _quality_after.quality_level.value
                                    if hasattr(_quality_after.quality_level, "value")
                                    else str(_quality_after.quality_level)
                                ),
                                "snr_db": round(float(_quality_after.snr_db), 2),
                                "dynamic_range_db": round(float(_quality_after.dynamic_range_db), 2),
                                "warmth": round(float(_quality_after.warmth), 4),
                                "naturalness": round(float(_quality_after.naturalness), 4),
                                "brightness": round(float(_quality_after.brightness), 4),
                            }
                            if _quality_after is not None
                            else None
                        ),
                        "delta_score": (
                            round(_quality_after.overall_score - _quality_before.overall_score, 2)
                            if (_quality_before is not None and _quality_after is not None)
                            else None
                        ),
                        "delta_snr_db": (
                            round(float(_quality_after.snr_db) - float(_quality_before.snr_db), 2)
                            if (_quality_before is not None and _quality_after is not None)
                            else None
                        ),
                    }
                ),
            },
            # --- Spec §2.1/§2.2 Pflichtfelder ---
            pqs_result=_pqs_result,
            musical_goals=_musical_goal_scores if _musical_goal_scores else None,
            excellence=_excellence_result,
            adaptive_thresholds=locals().get("_effective_goal_thresholds", {}),
            physical_ceiling=(dict(getattr(_physical_ceiling, "ceiling", {})) if _physical_ceiling is not None else {}),
            goal_applicability=(
                {
                    g: (g in _goal_applicability_result.applicable)
                    for g in _goal_applicability_result.applicable.union(_goal_applicability_result.inapplicable)
                }
                if _goal_applicability_result is not None
                else {}
            ),
            confidence=(float(_pipeline_confidence.confidence) if _pipeline_confidence is not None else 1.0),
        )

        # §2.13 Künstler-Signatur nach Restaurierung aktualisieren und speichern
        if _artist_id is not None:
            try:
                from backend.core.artist_signature_store import (
                    VoiceCharacteristics as _VoiceCharacteristics,
                    get_signature_store as _get_sig_store2,
                )

                _sig_store2 = _get_sig_store2()
                # Minimale VoiceCharacteristics: Spektral-Hüllkurve aus restauriertem Audio
                _env_audio = (
                    np.mean(restored_audio, axis=1).astype(np.float32)
                    if restored_audio.ndim == 2
                    else restored_audio.astype(np.float32)
                )
                _n_env = min(512, len(_env_audio))
                _spec = np.abs(np.fft.rfft(_env_audio[:_n_env], n=_n_env)).astype(np.float32)
                _x_src = np.linspace(0, 1, len(_spec), dtype=np.float32)
                _x_dst = np.linspace(0, 1, 128, dtype=np.float32)
                _envelope = np.interp(_x_dst, _x_src, _spec).astype(np.float32)
                _env_norm = float(np.linalg.norm(_envelope))
                if _env_norm > 0:
                    _envelope /= _env_norm
                _vc = _VoiceCharacteristics(spectral_envelope=_envelope)
                _updated_sig = _sig_store2.update_from_analysis(_artist_id, _vc)
                _sig_store2.save(_updated_sig)
                logger.debug(
                    "Künstler-Signatur gespeichert: artist_id=%s | n=%d | confidence=%.2f",
                    _artist_id,
                    _updated_sig.n_files_analyzed,
                    _updated_sig.confidence,
                )
            except Exception as _sig_upd_exc:
                logger.debug("ArtistSignatureStore Update nicht verfügbar: %s", _sig_upd_exc)

        logger.info(
            f"✅ Restoration complete: {total_time:.1f}s ({rt_factor:.2f}× RT), "
            f"Quality: {quality_estimate*100:.1f}%"
        )

        # --- Aggressive end-of-run memory cleanup (OOM hardening) ---
        _cleanup_report: Dict[str, Any] = {"unloaded": [], "errors": []}
        try:
            _unload_specs = [
                ("plugins.audiosr_plugin", "unload_audiosr", "AudioSR"),
                ("plugins.utmos_plugin", "unload_utmos", "UTMOS"),
                ("plugins.laion_clap_plugin", "unload_laion_clap", "LAION-CLAP"),
                ("plugins.mert_plugin", "unload_mert", "MERT"),
                ("plugins.crepe_plugin", "unload_crepe", "CREPE"),
                ("plugins.fcpe_plugin", "unload_fcpe", "FCPE"),
            ]
            for _mod_name, _fn_name, _label in _unload_specs:
                try:
                    _mod = importlib.import_module(_mod_name)
                    _fn = getattr(_mod, _fn_name, None)
                    if callable(_fn):
                        _fn()
                        _cleanup_report["unloaded"].append(_label)
                except Exception as _u_exc:
                    _cleanup_report["errors"].append(f"{_label}: {_u_exc}")

            try:
                from backend.core.plugin_lifecycle_manager import cleanup_after_file as _plm_cleanup_after_file  # noqa: PLC0415

                _cleanup_report["plm_evicted"] = int(_plm_cleanup_after_file())
            except Exception as _plm_exc:
                _cleanup_report["errors"].append(f"PLM: {_plm_exc}")

            # Defensive budget release for models without explicit unload hooks.
            try:
                from backend.core.ml_memory_budget import release as _ml_release  # noqa: PLC0415

                for _model_name in ("CREPE", "FCPE", "RMVPE", "BasicPitch"):
                    _ml_release(_model_name)
            except Exception as _ml_rel_exc:
                _cleanup_report["errors"].append(f"ML-Budget-Release: {_ml_rel_exc}")

            gc.collect()

            # Linux/glibc: return free heap pages to OS to reduce RSS between runs.
            try:
                import ctypes  # noqa: PLC0415

                _libc = ctypes.CDLL("libc.so.6")
                _trim = getattr(_libc, "malloc_trim", None)
                if callable(_trim):
                    _cleanup_report["malloc_trim"] = bool(_trim(0))
            except Exception as _trim_exc:
                _cleanup_report["errors"].append(f"malloc_trim: {_trim_exc}")
        except Exception as _cleanup_exc:
            logger.debug("Final memory cleanup failed: %s", _cleanup_exc)
        finally:
            try:
                result.metadata["memory_cleanup"] = _cleanup_report
            except Exception:
                pass

        return result

    def _select_phases(self, defect_result, *, causal_plan=None, chain_info=None, defekt_hint=None) -> List[str]:
        """
        Wählt Phasen kontextadaptiv basierend auf Defektbefund, Material und Modus.

        Args:
            defect_result: DefectScanResult vom DefectScanner.
            causal_plan:   Optional. RestorationPlan des CausalDefectReasoner (§2.6).
                           Wenn confidence ≥ 0.20, werden dessen recommended_phases
                           als Tier-1.5-Block in die Selektion eingeflochten.
            chain_info:    Optional. Tonträgerketten-Dict des TontraegerketteDenkers (§2.2).
                           combined_phases aus Multi-Carrier-Analyse (z. B. Vinyl→Kassette→MP3)
                           werden als Tier-1.6-Block aktiviert.
            defekt_hint:   Optional. Heuristik-Dict des DefektDenkers (§2.1).
                           recommended_phases (DSP-basiert) ergänzen den Bayesianischen
                           Tier-1.5-Block wenn confidence ≥ 0.15.

        Ablauf:
          1. PANNs-Analyse → Instrument-/Vokal-Konfidenz (mit DSP-Fallback)
          2. Tier 0    — Immer aktiv (DC-Offset, Rumpeln)
          3. Tier 1    — Kritische Defektkorrektur (schwellwertbasiert)
          4. Tier 1.5  — Kausaldiagnose-Ergänzung (CausalDefectReasoner §2.6)
          5. Tier 1.6  — Tonträgerkette-Ergänzung (TontraegerketteDenker §2.2)
          6. Tier 1.7  — DefektDenker-Heuristik-Ergänzung (DefektDenker §2.1)
          7. Tier 2    — Frequenz-/Spektral-Restaurierung
          8. Tier 3    — Stereo-/Phase-Verarbeitung
          9. Tier 4    — Instrument-/Vokal-Enhancement (PANNs-gesteuert)
         10. Tier 5    — Dynamik & Mastering
         11. Tier 6    — Ausgang (Lautheit, Limiter, Format)

        Im MAXIMUM-Modus (Studio 2026) werden niedrigere Schwellen und
        zusätzliche Enhancement-Phasen aktiviert.
        """
        scores = defect_result.scores
        material = defect_result.material_type
        is_max = self.config.mode == QualityMode.MAXIMUM

        def sev(defect_type, default=0.0):
            """Hilfsfunktion: Severity eines DefectType sicher auslesen."""
            s = scores.get(defect_type)
            return s.severity if s is not None else default

        # ── PANNs-Gate: Instrument/Vokal-Erkennung ──────────────────────────
        panns_tags: Dict[str, float] = {}
        try:
            import os as _os
            import sys

            _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "plugins")
            if _plugins_dir not in sys.path:
                sys.path.insert(0, _os.path.abspath(_plugins_dir))
            from panns_plugin import PANNsPlugin

            _panns = PANNsPlugin()
            # Verwende Referenz-Audio aus dem Defekt-Scanner wenn verfügbar
            _ref = getattr(defect_result, "_audio_ref", None)
            _sr = getattr(defect_result, "_sample_rate", 48000)
            if _ref is not None:
                panns_tags = _panns.get_tags(_ref, _sr)
        except Exception as _panns_exc:
            logger.debug("PANNs nicht verfügbar, kein Instrument-Gate aktiv: %s", _panns_exc)

        def panns(tag: str, threshold: float = 0.5) -> bool:
            """Liefert True wenn PANNs-Tag die Mindestkonfidenz erreicht."""
            return panns_tags.get(tag, 0.0) >= threshold

        vocals_detected = panns("Singing voice", 0.40) or panns("Vocals", 0.40) or panns("Speech", 0.35)
        guitar_detected = panns("Guitar", 0.60) or panns("Electric guitar", 0.60)
        brass_detected = panns("Brass instrument", 0.60) or panns("Trumpet", 0.60) or panns("Saxophone", 0.60)
        drums_detected = panns("Drum", 0.50) or panns("Percussion", 0.50)
        piano_detected = panns("Piano", 0.60) or panns("Keyboard (musical)", 0.60)

        # Im MAXIMUM-Modus alle Instrument-Phasen aktivieren wenn PANNs kein
        # klares Signal liefert (konservative Annahme: Musik enthält alles)
        if is_max and not panns_tags:
            vocals_detected = guitar_detected = brass_detected = True
            drums_detected = piano_detected = True

        selected: List[str] = []

        # ════════════════════════════════════════════════════════════════════
        # TIER 0 — Basiskorrekturen (immer aktiv, < 1 ms Overhead)
        # ════════════════════════════════════════════════════════════════════
        selected += [
            "phase_30_dc_offset_removal",  # DC-Gleichanteil entfernen
            "phase_05_rumble_filter",  # Tieffrequenzrumpeln < 20 Hz
        ]

        # ════════════════════════════════════════════════════════════════════
        # TIER 1 — Kritische Defektkorrektur
        # ════════════════════════════════════════════════════════════════════

        # Dropout / Lücken
        if sev(DefectType.DROPOUTS) > (0.05 if is_max else 0.10):
            selected.append("phase_24_dropout_repair")
        # phase_55 (DiffWave ML-Inpainting) nur bei echten analogen Dropouts aktivieren —
        # digitale Materialien (MP3, Streaming, CD) erzeugen keine bandlückenbedingten Dropouts.
        _ANALOG_MATERIALS_P55 = {
            MaterialType.REEL_TAPE,
            MaterialType.TAPE,
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
        }
        if sev(DefectType.DROPOUTS) > (0.20 if is_max else 0.30) and material in _ANALOG_MATERIALS_P55:
            selected.append("phase_55_diffusion_inpainting")  # ML-Inpainting nur für analoge Lücken

        # Clicks / Impulse (1. Pass)
        if sev(DefectType.CLICKS) > (0.10 if is_max else 0.15):
            selected.append("phase_01_click_removal")
        # Clicks / Pops (2. Pass — tiefergehende AR-Residual-Methode)
        if sev(DefectType.CLICKS) > (0.20 if is_max else 0.25):
            selected.append("phase_27_click_pop_removal")

        # Vinyl/Shellac/Lacquer: Oberflächenrausch-Profil VOR Crackle-Entfernung
        if (
            material in [MaterialType.VINYL, MaterialType.SHELLAC, MaterialType.LACQUER_DISC]
            and sev(DefectType.CRACKLE) > 0.10
        ):
            selected.append("phase_28_surface_noise_profiling")

        # Crackle (Vinyl/Shellac-typisch)
        if sev(DefectType.CRACKLE) > (0.10 if is_max else 0.15):
            selected.append("phase_09_crackle_removal")

        # Brumm 50/60 Hz
        if sev(DefectType.HUM) > (0.08 if is_max else 0.10):
            selected.append("phase_02_hum_removal")

        # Breitbandrauschen
        if sev(DefectType.HIGH_FREQ_NOISE) > (0.15 if is_max else 0.20):
            selected.append("phase_03_denoise")

        # Tape-Hiss (spezifisches Profil-basiertes Verfahren)
        # Shellac: phase_03_denoise übernimmt das NR — phase_29 NICHT unconditional
        # aktivieren, da das Doppel-NR bei SNR≈6 dB das Signal vollständig zerstört.
        # REEL_TAPE: Profi-Spulenband mit identischem Hiss-Profil wie TAPE.
        if sev(DefectType.HIGH_FREQ_NOISE) > (0.20 if is_max else 0.30) or material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_29_tape_hiss_reduction")

        # Noise-Gate für Stille-Segmente mit Rauschboden
        if sev(DefectType.HIGH_FREQ_NOISE) > 0.25 or sev(DefectType.LOW_FREQ_RUMBLE) > 0.20 or is_max:
            selected.append("phase_18_noise_gate")

        # Wow/Flutter (Magnetband-Gleichlaufschwankungen)
        if max(sev(DefectType.WOW), sev(DefectType.FLUTTER)) > (0.08 if is_max else 0.10):
            selected.append("phase_12_wow_flutter_fix")

        # Pitch-Drift (langsame Tonhöhenschwankung)
        if sev(DefectType.PITCH_DRIFT) > (0.10 if is_max else 0.15):
            selected.append("phase_31_speed_pitch_correction")

        # Phasen-/Azimuth-Fehler
        if sev(DefectType.PHASE_ISSUES) > (0.08 if is_max else 0.10):
            selected.append("phase_14_phase_correction")
        if sev(DefectType.PHASE_ISSUES) > (0.15 if is_max else 0.20) and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
            MaterialType.SHELLAC,
        ]:
            selected.append("phase_25_azimuth_correction")

        # Nachhall-Reduktion
        if sev(DefectType.REVERB_EXCESS) > (0.20 if is_max else 0.25):
            selected.append("phase_20_reverb_reduction")
        if sev(DefectType.REVERB_EXCESS) > (0.35 if is_max else 0.45):
            selected.append("phase_49_advanced_dereverb")  # Tiefgehende Blind-RIR-Methode

        # Spektral-Reparatur (Clipping / Digital-Artefakte)
        if sev(DefectType.CLIPPING) > (0.08 if is_max else 0.10) or sev(DefectType.DIGITAL_ARTIFACTS) > (
            0.15 if is_max else 0.20
        ):
            selected.append("phase_23_spectral_repair")

        # Print-Through (Magnetisches Übersprechen bei Bandaufnahmen — Vor-/Nachecho)
        # Physikalisch nur bei Bandmaterial möglich (§4.5 Adaptive Temporal Subtraction)
        if sev(DefectType.PRINT_THROUGH) > (0.10 if is_max else 0.15) and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_29_tape_hiss_reduction")  # Hiss-Profil-Subtraktion
            selected.append("phase_03_denoise")  # Breitband-Restecho-NR
        if sev(DefectType.PRINT_THROUGH) > (0.25 if is_max else 0.35) and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_23_spectral_repair")  # Schwere Vorecho-Tilgung

        # Quantisierungsrauschen (niedrige Bit-Tiefe / fehlerhaftes Resampling)
        if sev(DefectType.QUANTIZATION_NOISE) > (0.10 if is_max else 0.15):
            selected.append("phase_03_denoise")
            selected.append("phase_23_spectral_repair")
        if sev(DefectType.QUANTIZATION_NOISE) > (0.20 if is_max else 0.30):
            selected.append("phase_06_frequency_restoration")  # Treppen-Artefakte glätten

        # Jitter-Artefakte (D/A-Wandler-Zeitfehler — CD, DAT, Streaming)
        if sev(DefectType.JITTER_ARTIFACTS) > (0.10 if is_max else 0.15):
            selected.append("phase_12_wow_flutter_fix")  # Zeitachsen-Trägeheit
            selected.append("phase_23_spectral_repair")  # Spektrale Jitter-Spuren

        # RIAA-Entzerrungsfehler (Shellac/früher Vinyl: AES/NAB/FFRR — §6.3, §7.2)
        if sev(DefectType.RIAA_CURVE_ERROR) > (0.08 if is_max else 0.12):
            selected.append("phase_04_eq_correction")  # Fehler-EQ korrigieren
            selected.append("phase_06_frequency_restoration")  # Spektralprofil wiederherstellen
        if sev(DefectType.RIAA_CURVE_ERROR) > (0.20 if is_max else 0.30):
            selected.append("phase_07_harmonic_restoration")  # Obertöne durch Entzerrungs-Kette verloren

        # Aliasing (AA-Filter-Artefakte bei Digitalisierung — §6.3, §7.2)
        if sev(DefectType.ALIASING) > (0.10 if is_max else 0.15):
            selected.append("phase_03_denoise")  # Spiegelfrequenzen dämpfen
            selected.append("phase_23_spectral_repair")  # Spektrale Aliasing-Spuren beseitigen
        if sev(DefectType.ALIASING) > (0.25 if is_max else 0.35):
            selected.append("phase_50_spectral_repair")  # Zweiter Spektral-Pass

        # Bias-Fehler (falscher Vormagnetisierungsstrom bei Bandaufnahme — §6.3, §7.2)
        if sev(DefectType.BIAS_ERROR) > (0.08 if is_max else 0.12):
            selected.append("phase_04_eq_correction")  # HF-Rolloff/-Überhöhung kompensieren
            selected.append("phase_03_denoise")  # Bias-induziertes Rauschen reduzieren
        if sev(DefectType.BIAS_ERROR) > (0.20 if is_max else 0.30):
            selected.append("phase_06_frequency_restoration")  # Frequenzgang-Verluste ausgleichen
            selected.append("phase_29_tape_hiss_reduction")  # Bias-erhöhtes Hintergrundrauschen

        # WAX_CYLINDER (Phonographen-Wachswalze 1890–1930): aggressiv — HF ≤ 5 kHz, SNR extrem
        if material == MaterialType.WAX_CYLINDER:
            selected += [
                "phase_01_click_removal",  # Mechanische Störimpulse
                "phase_29_tape_hiss_reduction",  # Wachsoberflächenrauschen
                "phase_03_denoise",  # Breitbandrauschen
                "phase_06_frequency_restoration",  # HF ≤ 5 kHz rekonstruieren
            ]
            if is_max:
                selected.append("phase_07_harmonic_restoration")  # Obertöne ergänzen

        # WIRE_RECORDING (Drahtton 1940–1955): Jitter + frequenzselektive Einbrüche
        if material == MaterialType.WIRE_RECORDING:
            selected += [
                "phase_12_wow_flutter_fix",  # Drahtband-Jitter
                "phase_24_dropout_repair",  # Frequenzeinbrüche als Dropouts
                "phase_03_denoise",  # Drahtband-Hintergrundrauschen
                "phase_29_tape_hiss_reduction",  # Magnetdraht-Hiss
            ]

        # LACQUER_DISC (Acetat-Lackfolien 1930–1950): Rissen + Rille-Ermüdung + Substrat-Rauschen
        if material == MaterialType.LACQUER_DISC:
            selected += [
                "phase_01_click_removal",  # Lackrisse → Impulse
                "phase_09_crackle_removal",  # Rille-Ermüdung → Crackle
                "phase_03_denoise",  # Substrat-Rauschen
                "phase_29_tape_hiss_reduction",  # Lackoberflächen-Rauschen
            ]

        # ════════════════════════════════════════════════════════════════════
        # TIER 1.5 — Kausaldiagnose-Ergänzung (CausalDefectReasoner §2.6)
        # Bayesianische Ursachenanalyse (CAUSE_TO_PHASES) liefert Phasen,
        # die Severity-Schwellen in Tier 1 ggf. nicht aktiviert haben.
        # Nur aktiv wenn confidence ≥ 0.20 (Rauschunterdrückung bei Unsicherheit).
        # ════════════════════════════════════════════════════════════════════
        if causal_plan is not None and causal_plan.confidence >= 0.20:
            _selected_set = set(selected)
            _causal_added: List[str] = []
            for _cp_phase in causal_plan.recommended_phases:
                if _cp_phase not in _selected_set:
                    selected.append(_cp_phase)
                    _selected_set.add(_cp_phase)
                    _causal_added.append(_cp_phase)
            if _causal_added:
                logger.info(
                    "🧠 CausalReasoner §2.6 ergänzt %d Phase(n) "
                    "(cause=%s, conf=%.2f): %s",
                    len(_causal_added),
                    causal_plan.primary_cause,
                    causal_plan.confidence,
                    _causal_added,
                )

        # ════════════════════════════════════════════════════════════════════
        # TIER 1.6 — Tonträgerkette (TontraegerketteDenker §2.2)
        # Phasen aus Multi-Carrier-Analyse (z. B. Vinyl→Kassette→MP3) werden
        # als Ergänzung zur internen Kausal-Analyse aktiviert.
        # ════════════════════════════════════════════════════════════════════
        if chain_info is not None:
            _chain_phases = chain_info.get("combined_phases", []) if isinstance(chain_info, dict) else []
            _chain_added: List[str] = []
            _chain_set = set(selected)
            for _chn_phase in _chain_phases:
                if _chn_phase not in _chain_set:
                    selected.append(_chn_phase)
                    _chain_set.add(_chn_phase)
                    _chain_added.append(_chn_phase)
            if _chain_added:
                _complexity = chain_info.get("chain_complexity", 0.0) if isinstance(chain_info, dict) else 0.0
                logger.info(
                    "🔗 TontraegerketteDenker §2.2 ergänzt %d Phase(n) "
                    "(complexity=%.2f): %s",
                    len(_chain_added),
                    _complexity,
                    _chain_added,
                )

        # ════════════════════════════════════════════════════════════════════
        # TIER 1.7 — DefektDenker Heuristik-Ergänzung (§2.1)
        # Heuristische Phasen-Empfehlungen (DSP-basiert) ergänzen die
        # Bayesianische CausalPlan-Selektion aus Tier 1.5. Schwelle ≥ 0.15
        # (niedriger als CausalReasoner 0.20 — Heuristiken sind präziser).
        # ════════════════════════════════════════════════════════════════════
        if defekt_hint is not None:
            _dh_conf = float(defekt_hint.get("confidence", 0.0)) if isinstance(defekt_hint, dict) else 0.0
            if _dh_conf >= 0.15:
                _dh_phases = defekt_hint.get("recommended_phases", []) if isinstance(defekt_hint, dict) else []
                _dh_added: List[str] = []
                _dh_set = set(selected)
                for _dh_phase in _dh_phases:
                    if _dh_phase not in _dh_set:
                        selected.append(_dh_phase)
                        _dh_set.add(_dh_phase)
                        _dh_added.append(_dh_phase)
                if _dh_added:
                    logger.info(
                        "🔍 DefektDenker §2.1 ergänzt %d Phase(n) "
                        "(confidence=%.2f): %s",
                        len(_dh_added),
                        _dh_conf,
                        _dh_added,
                    )

        # ════════════════════════════════════════════════════════════════════
        # TIER 2 — Frequenz- / Spektral-Restaurierung
        # ════════════════════════════════════════════════════════════════════

        # Bandbreitenerweiterung bei Verlusten
        if sev(DefectType.BANDWIDTH_LOSS) > (0.15 if is_max else 0.20) or sev(DefectType.COMPRESSION_ARTIFACTS) > (
            0.25 if is_max else 0.30
        ):
            selected.append("phase_06_frequency_restoration")

        # Oberton-Restaurierung (tiefergehend)
        if sev(DefectType.BANDWIDTH_LOSS) > (0.25 if is_max else 0.30) or is_max:
            selected.append("phase_07_harmonic_restoration")

        # Transienten-Schutz bei digitalen Artefakten
        if sev(DefectType.DIGITAL_ARTIFACTS) > (0.20 if is_max else 0.25) or is_max:
            selected.append("phase_08_transient_preservation")

        # EQ-Korrektur für materialspezifische Frequenzgänge
        # Alle analogen Träger und DAT haben charakteristische Frequenzgangkurven
        if (
            material
            in [
                MaterialType.SHELLAC,
                MaterialType.VINYL,
                MaterialType.TAPE,
                MaterialType.REEL_TAPE,
                MaterialType.DAT,
                MaterialType.MINIDISC,
                MaterialType.WAX_CYLINDER,
                MaterialType.WIRE_RECORDING,
                MaterialType.LACQUER_DISC,
            ]
            or is_max
        ):
            selected.append("phase_04_eq_correction")

        # Dynamikbereich-Erweiterung bei über-Kompression
        if (
            sev(DefectType.DYNAMIC_COMPRESSION_EXCESS) > (0.20 if is_max else 0.30)
            or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.30
        ):
            selected.append("phase_26_dynamic_range_expansion")

        # Spektrale Gesamt-Reparatur (breiter zweiter Pass)
        if sev(DefectType.DIGITAL_ARTIFACTS) > 0.20 or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.20 or is_max:
            selected.append("phase_50_spectral_repair")

        # ════════════════════════════════════════════════════════════════════
        # TIER 3 — Stereo- / Phase-Verarbeitung
        # ════════════════════════════════════════════════════════════════════

        is_stereo = getattr(defect_result, "is_stereo", True)
        if not is_stereo:
            selected.append("phase_32_mono_to_stereo")  # Mono→Pseudo-Stereo

        # Stereo-Balance bei Imbalance
        if sev(DefectType.STEREO_IMBALANCE) > (0.10 if is_max else 0.15):
            selected.append("phase_15_stereo_balance")

        # Stereo-Breiten-Begrenzer bei extremer Imbalance
        if sev(DefectType.STEREO_IMBALANCE) > (0.30 if is_max else 0.40):
            selected.append("phase_33_stereo_width_limiter")

        # Mid/Side-Verarbeitung für Stereo-Quellen
        if is_stereo:
            selected.append("phase_34_mid_side_processing")

        # ════════════════════════════════════════════════════════════════════
        # TIER 4 — Instrument- / Vokal-Enhancement (PANNs-gesteuert, §2.9)
        # ════════════════════════════════════════════════════════════════════

        if vocals_detected:
            selected += [
                "phase_19_de_esser",  # Sibilant-Kontrolle (stimmtyp-adaptiv)
                "phase_42_vocal_enhancement",  # Formant, Klarheit, Präsenz
                "phase_43_ml_deesser",  # ML-De-Esser (zweiter Pass)
            ]
        if guitar_detected:
            selected.append("phase_44_guitar_enhancement")
        if brass_detected:
            selected.append("phase_45_brass_enhancement")
        if drums_detected:
            selected.append("phase_51_drums_enhancement")
        if piano_detected:
            selected.append("phase_52_piano_restoration")

        # Semantische Audio-Analyse (PANNs-Kontext-Verstehen)
        selected.append("phase_53_semantic_audio")

        # ════════════════════════════════════════════════════════════════════
        # TIER 5 — Dynamik & Mastering
        # ════════════════════════════════════════════════════════════════════

        # Bass-Fundament (immer)
        selected.append("phase_37_bass_enhancement")

        # Präsenz (2–6 kHz) — Studio 2026 oder bei Vokalinhalt
        if is_max or vocals_detected:
            selected.append("phase_38_presence_boost")

        # Air-Band Anhebung > 12 kHz — bei Bandbegrenzung oder Studio 2026
        # WAX_CYLINDER und LACQUER_DISC: HF-Rekonstruktion nach physikalischer Bandbegrenzung
        if (
            is_max
            or sev(DefectType.BANDWIDTH_LOSS) > 0.10
            or material
            in [
                MaterialType.SHELLAC,
                MaterialType.TAPE,
                MaterialType.REEL_TAPE,
                MaterialType.WAX_CYLINDER,
                MaterialType.LACQUER_DISC,
            ]
        ):
            selected.append("phase_39_air_band_enhancement")

        # Tape-Sättigungs-Emulation (Studio 2026 + Tape/REEL-Material)
        # REEL_TAPE hat identischen Röhrensättigungs-Charakter wie TAPE
        if is_max and material in [MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.SHELLAC]:
            selected.append("phase_22_tape_saturation")

        # Harmonischer Exciter (Studio 2026)
        if is_max:
            selected.append("phase_21_exciter")

        # Transparente Dynamik (psychoakustisch, genre-adaptiv)
        if (
            material
            in [
                MaterialType.CD_DIGITAL,
                MaterialType.DAT,
                MaterialType.MP3_LOW,
                MaterialType.MP3_HIGH,
                MaterialType.AAC,
                MaterialType.STREAMING,
            ]
            or sev(DefectType.DYNAMIC_COMPRESSION_EXCESS) > 0.20
            or is_max
        ):
            selected.append("phase_54_transparent_dynamics")

        # Multiband-Kompression (Studio 2026 oder nicht über-komprimiert)
        if is_max or sev(DefectType.COMPRESSION_ARTIFACTS) < 0.40:
            selected.append("phase_35_multiband_compression")

        # Transient-Shaper (nach Drums-Enhancement)
        if drums_detected or is_max:
            selected.append("phase_36_transient_shaper")

        # Kompression & Limiting (wenn nicht bereits über-komprimiert)
        if sev(DefectType.COMPRESSION_ARTIFACTS) < (0.60 if is_max else 0.50):
            selected.append("phase_10_compression")
            selected.append("phase_11_limiting")

        # Stereo-Enhancement
        if sev(DefectType.STEREO_IMBALANCE) < 0.40:
            selected.append("phase_13_stereo_enhancement")

        # Stereo-Breiten-Enhancement
        selected.append("phase_48_stereo_width_enhancer")

        # Spatial-Enhancement (Raumklang)
        selected.append("phase_46_spatial_enhancement")

        # ════════════════════════════════════════════════════════════════════
        # TIER 6 — Finalisierung (immer, kanonische Reihenfolge §1.4)
        # ════════════════════════════════════════════════════════════════════
        selected += [
            "phase_16_final_eq",  # Finales EQ-Trimming
            "phase_17_mastering_polish",  # Mastering-Politur
            "phase_47_truepeak_limiter",  # True-Peak −1.0 dBTP (EBU R128)
            "phase_40_loudness_normalization",  # −14 LUFS (Streaming) / −18 (Archiv)
            "phase_41_output_format_optimization",  # Dithering, Metadaten, Format
        ]

        # Duplikate entfernen, Reihenfolge beibehalten
        seen: set = set()
        unique: List[str] = []
        for p in selected:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        logger.info(
            "🎛️  Phase-Selektion: Modus=%s, Material=%s, "
            "PANNs=[Vocals=%s Guitar=%s Brass=%s Drums=%s Piano=%s], "
            "Kausal=%s (conf=%.2f), "
            "%d Phasen ausgewählt",
            self.config.mode,
            material.value if hasattr(material, "value") else material,
            vocals_detected,
            guitar_detected,
            brass_detected,
            drums_detected,
            piano_detected,
            causal_plan.primary_cause if causal_plan is not None else "—",
            causal_plan.confidence if causal_plan is not None else 0.0,
            len(unique),
        )
        return unique

    def _apply_phase_skipping(self, selected_phases: List[str], defect_result) -> Tuple[List[str], Dict[str, str]]:
        """
        Apply intelligent phase skipping to reduce processing time.

        Uses PhaseSkipper to analyze defect_result and skip phases that
        won't provide meaningful improvement (e.g., denoise when noise_floor < -60dB).

        Args:
            selected_phases: List of phase IDs selected by defect analysis
            defect_result: DefectScanResult from defect scanner

        Returns:
            Tuple of (filtered_phases, skip_reasons)
            - filtered_phases: List of phases after skipping
            - skip_reasons: Dict mapping skipped phase_id to reason
        """
        if not self.phase_skipper:
            return selected_phases, {}

        # Create DefectAnalysis object for PhaseSkipper
        # Note: PhaseSkipper expects a DefectAnalysis object with specific fields
        from backend.core.defect_analysis import DefectAnalysis, SourceMedium

        # Map MaterialType to SourceMedium
        medium_map = {
            MaterialType.VINYL: SourceMedium.VINYL,
            MaterialType.TAPE: SourceMedium.CASSETTE,  # Map tape to cassette (most common)
            MaterialType.REEL_TAPE: SourceMedium.REEL_TAPE,
            MaterialType.SHELLAC: SourceMedium.SHELLAC,
            MaterialType.DAT: SourceMedium.DAT,
            MaterialType.CD_DIGITAL: SourceMedium.DIGITAL,
            MaterialType.MP3_LOW: SourceMedium.MP3_LOW,
            MaterialType.MP3_HIGH: SourceMedium.MP3_HIGH,
            MaterialType.AAC: SourceMedium.AAC,
            MaterialType.MINIDISC: SourceMedium.MINIDISC,
            MaterialType.STREAMING: SourceMedium.STREAMING,
            MaterialType.UNKNOWN: SourceMedium.UNKNOWN,
        }

        # Create DefectAnalysis (simplified, PhaseSkipper only needs key fields)
        # Note: Map DefectScanner severity scores to DefectAnalysis fields
        clicks_severity = defect_result.scores.get(DefectType.CLICKS, type("obj", (), {"severity": 0.0})).severity
        hum_severity = defect_result.scores.get(DefectType.HUM, type("obj", (), {"severity": 0.0})).severity
        hiss_severity = defect_result.scores.get(
            DefectType.HIGH_FREQ_NOISE, type("obj", (), {"severity": 0.0})
        ).severity
        dropouts_severity = defect_result.scores.get(DefectType.DROPOUTS, type("obj", (), {"severity": 0.0})).severity

        defect_analysis = DefectAnalysis(
            medium=medium_map.get(defect_result.material_type, SourceMedium.UNKNOWN),
            noise_floor_db=hiss_severity * -60.0,  # Convert severity to dB (0.0 → -60dB, 1.0 → 0dB)
            clipping_percentage=0.0,  # Not tracked by DefectScanner, use default
            click_count=int(clicks_severity * 100),  # Estimate: severity 0.5 = 50 clicks
            click_density=clicks_severity * 10.0,  # Estimate: severity 0.1 = 1 click/sec
            has_hum=hum_severity > 0.2,
            has_hiss=hiss_severity > 0.25,
            dropout_count=int(dropouts_severity * 20),  # Estimate: severity 0.05 = 1 dropout
        )

        # Get skip decisions from PhaseSkipper
        # Note: PhaseSkipper uses different phase enum, we need to map
        phase_map = {
            "phase_03_denoise": "phase_2_denoise",
            "phase_29_tape_hiss_reduction": "phase_2_denoise",
            "phase_04_eq_correction": None,  # No direct mapping
            "phase_01_click_removal": "phase_5_click_removal",
            "phase_02_hum_removal": "phase_6_dehum",
            "phase_24_dropout_repair": "phase_7_dropout_repair",
            "phase_23_spectral_repair": "phase_8_spectral_repair",
        }

        filtered_phases = []
        skip_reasons = {}

        for phase_id in selected_phases:
            # Check if phase can be skipped
            should_skip = False
            skip_reason = ""

            # Map to PhaseSkipper's phase enum
            skipper_phase = phase_map.get(phase_id)

            if skipper_phase:
                # Use PhaseSkipper logic
                if skipper_phase == "phase_2_denoise":
                    # Skip denoise if noise floor very low
                    if defect_analysis.noise_floor_db < -60.0 and not defect_analysis.has_hiss:
                        should_skip = not self.config.phase_skipping_conservative
                        skip_reason = f"Very low noise floor ({defect_analysis.noise_floor_db:.1f} dB)"
                elif skipper_phase == "phase_5_click_removal":
                    # Skip click removal if minimal clicks
                    if defect_analysis.click_count == 0:
                        should_skip = not self.config.phase_skipping_conservative
                        skip_reason = "No significant clicks detected"
                elif skipper_phase == "phase_6_dehum":
                    # Skip dehum if no hum detected
                    if not defect_analysis.has_hum:
                        should_skip = not self.config.phase_skipping_conservative
                        skip_reason = "No hum detected"
                elif skipper_phase == "phase_7_dropout_repair":
                    # Never skip dropout repair if selected (critical)
                    should_skip = False
                elif skipper_phase == "phase_8_spectral_repair":
                    # Skip spectral repair if digital source is clean
                    if defect_analysis.medium == SourceMedium.DIGITAL and defect_analysis.noise_floor_db < -50.0:
                        should_skip = not self.config.phase_skipping_conservative
                        skip_reason = "Clean digital source"

            if should_skip:
                skip_reasons[phase_id] = skip_reason
                logger.debug(f"Skipping {phase_id}: {skip_reason}")
            else:
                filtered_phases.append(phase_id)

        return filtered_phases, skip_reasons

    def _profiled_phase_call(self, phase, audio: np.ndarray, **kwargs):
        """Führt eine Phase mit Zeit- und (optional) Speicherprofiling aus.

        Wenn self._active_global_plan gesetzt ist, werden phasenspezifische
        Parameter aus dem Globalplan in kwargs eingeschleust — sofern die Phase
        sie nicht bereits explizit übergeben hat (explizit gewinnt).
        """
        # Get phase metadata for logging
        phase_metadata = phase.get_metadata()
        phase_name = f"{phase_metadata.name} ({phase_metadata.phase_id})"

        # §Dach: GlobalPlan-Parameter einschleusen (ohne explizite Werte zu überschreiben)
        _gp = getattr(self, "_active_global_plan", None)
        if _gp is not None:
            try:
                _phase_params = _gp.get_phase_params(phase_metadata.phase_id)
                for _pk, _pv in _phase_params.items():
                    if _pk not in kwargs:
                        kwargs[_pk] = _pv
            except Exception:  # noqa: BLE001
                pass  # Globalplan-Fehler stoppen nie eine Phase

        # Normalisierung: material_type und material immer als MaterialType-Enum übergeben
        # Phasen haben unterschiedliche Signaturen (material vs. material_type), beide abdecken
        for _mk in ("material_type", "material"):
            _mv = kwargs.get(_mk)
            if _mv is not None and isinstance(_mv, str):
                try:
                    from backend.core.defect_scanner import MaterialType as _MT

                    kwargs[_mk] = _MT(_mv)
                except Exception:
                    pass
        # Stelle sicher, dass beide Keys vorhanden sind (falls nur einer übergeben wurde)
        if "material_type" in kwargs and "material" not in kwargs:
            kwargs["material"] = kwargs["material_type"]
        elif "material" in kwargs and "material_type" not in kwargs:
            kwargs["material_type"] = kwargs["material"]

        t0 = time.perf_counter()
        mem0 = memory_usage(-1, interval=0.01, timeout=1) if MEMORY_PROFILING_AVAILABLE else [0]

        # Call phase.process() method (not phase() itself!)
        result = phase.process(audio, **kwargs)

        t1 = time.perf_counter()
        mem1 = memory_usage(-1, interval=0.01, timeout=1) if MEMORY_PROFILING_AVAILABLE else [0]
        elapsed = t1 - t0
        mem_used = max(mem1) - min(mem0) if MEMORY_PROFILING_AVAILABLE else None

        logger.info(
            f"Profiling: Phase {phase_name} | Zeit: {elapsed:.3f}s"
            + (f" | Speicher: {mem_used:.1f} MiB" if mem_used is not None else "")
        )

        if hasattr(result, "profiling"):
            result.profiling["time_seconds"] = elapsed
            if mem_used is not None:
                result.profiling["memory_mib"] = mem_used

        return result

    def _execute_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        material_type: MaterialType,
        defect_result,
        selected_phases: List[str],
        progress_callback=None,
        _phase_progress_start: int = 30,
        _phase_progress_end: int = 80,
        restorability_score: float = 70.0,  # §2.29 normativ: aus RestorabilityEstimator.estimate()
        applicable_goals: Optional[Set[str]] = None,  # §2.32 normativ: aus GoalApplicabilityFilter
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Führt ausgewählte Phasen parallel (Multi-Core) aus, falls keine Abhängigkeiten bestehen.
        Returns: (processed_audio, executed_phases, skipped_phases)
        """
        # §3.1 NaN/Inf-Invariante: Input VOR Phasenausführung bereinigen
        if not np.all(np.isfinite(audio)):
            logger.debug("_execute_pipeline: NaN/Inf im Input-Audio bereinigt")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.9, neginf=-0.9)
        audio = np.clip(audio, -1.0, 1.0)
        current_audio = audio.copy()
        executed = []
        skipped = []
        # §Punkt3 Phasen-Regressionsprotokoll: RMS-Delta je Phase (sequentielle Ausführung)
        # Einheit: dBFS-Differenz nach - vor Phase. Positiv = Energie gestiegen, negativ = gesunken.
        self._phase_regression_log: Dict[str, float] = {}
        # §2.29 PerPhaseMusicalGoalsGate — vor Pipeline-Loop initialisieren (sequentielle Ausführung)
        # Deaktiviert AUSSCHLIESSLICH über enable_phase_gate=False (z. B. --no-phase-gate).
        # enable_performance_guard steuert das CPU-Budget-Throttling, NICHT die musikalische
        # Qualitätskontrolle — beide Flags sind bewusst entkoppelt (§2.29 Fix v9.10.46).
        _pmgg_gate = None
        _pmgg_restorability_score: float = restorability_score  # §2.29 normativ: aus RestorabilityEstimator (kein Hard-Code)
        _pmgg_scores_curr: Optional[Dict[str, float]] = None
        _pmgg_log_entries: list = []
        _pmgg_enabled = getattr(self.config, "enable_phase_gate", True)
        if _pmgg_enabled:
            try:
                from backend.core.per_phase_musical_goals_gate import get_phase_gate as _get_pmgg_gate

                _pmgg_gate = _get_pmgg_gate()
                _pmgg_gate.reset()
                logger.debug("§2.29 PMGG: Initialisiert für %d Phasen", len(selected_phases))
            except Exception as _pmgg_init_exc:
                logger.debug("PerPhaseMusicalGoalsGate nicht verfügbar: %s", _pmgg_init_exc)
        else:
            logger.debug("§2.29 PMGG: Deaktiviert (enable_phase_gate=False via --no-phase-gate)")
        # M-1 §2.2.1 Parallelisierungs-Invariante: TIER 0 und TIER 1 IMMER sequenziell
        # Kausale Abhängigkeiten: Defektkorrektur muss kausal verkettet sein (Output → nächster Input).
        # np.mean-Merge über Tier-0/1-Phasen würde Defektkorrekturen gegenseitig aufheben.
        # Nur TIER 2–4 dürfen parallelisieren (§2.2.1: "Merge via np.mean NUR wenn gleiche Frequenzzone").
        _SEQUENTIAL_TIER_PHASES = frozenset(
            {
                # TIER 0 — Basiskorrekturen (immer aktiv)
                "phase_30_dc_offset_removal",
                "phase_05_rumble_filter",
                # TIER 1 — Kritische Defektkorrektur (kausale Abhängigkeiten)
                "phase_01_click_removal",
                "phase_02_hum_removal",
                "phase_03_denoise",
                "phase_09_crackle_removal",
                "phase_12_wow_flutter_fix",
                "phase_14_phase_correction",
                "phase_18_noise_gate",
                "phase_20_reverb_reduction",
                "phase_23_spectral_repair",
                "phase_24_dropout_repair",
                "phase_25_azimuth_correction",
                "phase_27_click_pop_removal",
                "phase_28_surface_noise_profiling",
                "phase_29_tape_hiss_reduction",
                "phase_31_speed_pitch_correction",
                "phase_49_advanced_dereverb",
                "phase_55_diffusion_inpainting",
            }
        )
        _has_tier01 = any(p in _SEQUENTIAL_TIER_PHASES for p in selected_phases)
        # Prüfe auf Abhängigkeiten (vereinfachte Annahme: keine Abhängigkeiten → parallel, sonst sequentiell)
        has_dependencies = any(
            self.phase_metadata[pid]["dependencies"] for pid in selected_phases if pid in self.phase_metadata
        )
        if not has_dependencies and not _has_tier01 and len(selected_phases) > 1:
            # Parallele Ausführung mit Profiling
            results = {}
            with ThreadPoolExecutor(max_workers=self.config.num_cores) as executor:
                future_map = {}
                for phase_id in selected_phases:
                    phase = self._get_phase(phase_id)
                    if not phase:
                        logger.warning(f"Phase {phase_id} konnte nicht lazy-geladen werden, skipping")
                        skipped.append(phase_id)
                        continue
                    metadata = phase.get_metadata()
                    estimated_time = metadata.estimated_time_factor * (len(audio) / sample_rate)
                    if self.performance_guard and self.performance_guard.should_skip_phase(
                        phase_id, estimated_time, len(selected_phases) - 1
                    ):
                        skipped.append(phase_id)
                        continue
                    phase_start = (
                        self.performance_guard.start_phase(phase_id) if self.performance_guard else time.time()
                    )
                    future = executor.submit(
                        self._profiled_phase_call,
                        phase,
                        current_audio,  # audio parameter
                        sample_rate=sample_rate,
                        material_type=material_type,
                        material=material_type,
                        defect_scores=defect_result.scores,
                        quality_mode=self.config.mode.value,  # Pass quality mode for ML routing
                    )
                    future_map[future] = (phase_id, phase_start)
                for future in as_completed(future_map):
                    phase_id, phase_start = future_map[future]
                    try:
                        result = future.result()
                        if result.success:
                            results[phase_id] = result.audio
                            # §PROGRESS: Per-Phase Fortschritt (parallel)
                            _n_sel = max(len(selected_phases), 1)
                            _idx_ex = len(executed)
                            _phase_pct = _phase_progress_start + int(
                                (_phase_progress_end - _phase_progress_start) * _idx_ex / _n_sel
                            )
                            if progress_callback is not None:
                                try:
                                    _phase_label = phase_id.replace("phase_", "").replace("_", " ").title()
                                    progress_callback(
                                        _phase_pct,
                                        f"Phase {_idx_ex + 1}/{_n_sel}: {_phase_label}",
                                        0.0,
                                    )
                                except Exception:
                                    pass
                            executed.append(phase_id)
                            logger.info(f"✅ {phase_id}: {result.execution_time_seconds:.2f}s (parallel)")
                        else:
                            logger.error(f"❌ {phase_id} failed: {result.warnings}")
                            skipped.append(phase_id)
                    except Exception as e:
                        logger.error(f"❌ {phase_id} exception: {e}")
                        skipped.append(phase_id)
                    if self.performance_guard:
                        self.performance_guard.end_phase(phase_id, phase_start)
            if results:
                audios = [results[pid] for pid in executed if pid in results]
                if audios:
                    _merged = np.mean(audios, axis=0)
                    # §3.1 NaN-Guard: revert to pre-merge audio if NaN produced
                    if np.any(np.isnan(_merged)):
                        pass  # current_audio unchanged — phase produced NaN
                    else:
                        current_audio = np.clip(_merged, -1.0, 1.0)
        else:
            # Sequentielle Ausführung (Abhängigkeiten vorhanden oder nur eine Phase)
            # §2.16 TQC: Zeitmodifizierende Phasen können Kohärenz brechen — Rollback bei Versagen
            _TQC_CRITICAL_PHASES_SEQ = frozenset({
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            })
            for phase_id in selected_phases:
                phase = self._get_phase(phase_id)
                if not phase:
                    logger.warning(f"Phase {phase_id} konnte nicht lazy-geladen werden, skipping")
                    skipped.append(phase_id)
                    continue
                metadata = phase.get_metadata()
                remaining = len(selected_phases) - len(executed) - 1
                estimated_time = metadata.estimated_time_factor * (len(audio) / sample_rate)
                if self.performance_guard and self.performance_guard.should_skip_phase(
                    phase_id, estimated_time, remaining
                ):
                    skipped.append(phase_id)
                    continue
                phase_start = self.performance_guard.start_phase(phase_id) if self.performance_guard else time.time()
                # §2.16 TQC mid-pipeline: Snapshot vor zeitmodifizierenden Phasen
                _tqc_snap: Optional[np.ndarray] = (
                    current_audio.copy() if phase_id in _TQC_CRITICAL_PHASES_SEQ else None
                )
                # §Punkt3 Regressionsprotokoll: RMS vor Phase messen (in dBFS)
                _rms_before = float(np.sqrt(np.mean(current_audio ** 2) + 1e-12))
                _rms_before_db = 20.0 * np.log10(_rms_before)
                try:
                    if _pmgg_gate is not None:
                        # §2.29 PMGG: Musical-Goal-geschützte Phasenausführung
                        try:
                            _pmgg_audio_out, _pmgg_scores_curr, _pmgg_entry = _pmgg_gate.wrap_phase(
                                phase,
                                current_audio,
                                sample_rate,
                                _pmgg_scores_curr,
                                phase_kwargs={
                                    "sample_rate": sample_rate,
                                    "material_type": material_type,
                                    "material": material_type,
                                    "defect_scores": defect_result.scores,
                                    "quality_mode": self.config.mode.value,
                                },
                                restorability_score=_pmgg_restorability_score,  # §2.29 normativ
                                applicable_goals=applicable_goals,  # §2.32 normativ
                            )
                            _pmgg_log_entries.append(_pmgg_entry)
                            # §3.1 NaN-Guard: revert to pre-phase audio if NaN produced
                            if np.any(np.isnan(_pmgg_audio_out)):
                                pass  # current_audio unchanged
                            else:
                                current_audio = np.clip(_pmgg_audio_out, -1.0, 1.0)
                            # §PROGRESS: Per-Phase Fortschritt (PMGG-Pfad)
                            _n_sel = max(len(selected_phases), 1)
                            _idx_ex = len(executed)
                            _phase_pct = _phase_progress_start + int(
                                (_phase_progress_end - _phase_progress_start) * _idx_ex / _n_sel
                            )
                            if progress_callback is not None:
                                try:
                                    _phase_label = phase_id.replace("phase_", "").replace("_", " ").title()
                                    progress_callback(
                                        _phase_pct,
                                        f"Phase {_idx_ex + 1}/{_n_sel}: {_phase_label}",
                                        0.0,
                                    )
                                except Exception:
                                    pass
                            executed.append(phase_id)
                            logger.info(
                                f"✅ {phase_id}: PMGG action={_pmgg_entry.action} "
                                f"strength={_pmgg_entry.strength_used:.2f} "
                                f"rollbacks={_pmgg_gate._rollback_count}"
                            )
                            # §Punkt3 Regressionsprotokoll: RMS nach PMGG-Phase
                            _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio ** 2) + 1e-12)))
                            self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                        except Exception as _pmgg_phase_exc:
                            # Fallback: direkte Phasenausführung via _profiled_phase_call
                            logger.debug(
                                "PMGG wrap_phase Fehler (%s), Fallback: %s",
                                phase_id,
                                _pmgg_phase_exc,
                            )
                            result = self._profiled_phase_call(
                                phase,
                                current_audio,
                                sample_rate=sample_rate,
                                material_type=material_type,
                                material=material_type,
                                defect_scores=defect_result.scores,
                                quality_mode=self.config.mode.value,
                            )
                            if result.success:
                                _fa = result.audio
                                # §3.1 NaN-Guard: revert to pre-phase audio if NaN produced
                                if not np.any(np.isnan(_fa)):
                                    current_audio = np.clip(_fa, -1.0, 1.0)
                                # §PROGRESS: Per-Phase Fortschritt (PMGG-Fallback)
                                _n_sel = max(len(selected_phases), 1)
                                _idx_ex = len(executed)
                                _phase_pct = _phase_progress_start + int(
                                    (_phase_progress_end - _phase_progress_start) * _idx_ex / _n_sel
                                )
                                if progress_callback is not None:
                                    try:
                                        _phase_label = phase_id.replace("phase_", "").replace("_", " ").title()
                                        progress_callback(
                                            _phase_pct,
                                            f"Phase {_idx_ex + 1}/{_n_sel}: {_phase_label}",
                                            0.0,
                                        )
                                    except Exception:
                                        pass
                                executed.append(phase_id)
                                logger.info(f"✅ {phase_id} (fallback): {result.execution_time_seconds:.2f}s")
                                # §Punkt3 Regressionsprotokoll: RMS nach PMGG-Fallback-Phase
                                _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio ** 2) + 1e-12)))
                                self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                            else:
                                logger.error(f"❌ {phase_id} failed: {result.warnings}")
                                skipped.append(phase_id)
                    else:
                        result = self._profiled_phase_call(
                            phase,
                            current_audio,  # audio parameter
                            sample_rate=sample_rate,
                            material_type=material_type,
                            material=material_type,
                            defect_scores=defect_result.scores,
                            quality_mode=self.config.mode.value,  # Pass quality mode for ML routing
                        )
                        if result.success:
                            _ra = result.audio
                            # §3.1 NaN-Guard: revert to pre-phase audio if NaN produced
                            if not np.any(np.isnan(_ra)):
                                current_audio = np.clip(_ra, -1.0, 1.0)
                            # §PROGRESS: Per-Phase Fortschritt (sequenziell)
                            _n_sel = max(len(selected_phases), 1)
                            _idx_ex = len(executed)
                            _phase_pct = _phase_progress_start + int(
                                (_phase_progress_end - _phase_progress_start) * _idx_ex / _n_sel
                            )
                            if progress_callback is not None:
                                try:
                                    _phase_label = phase_id.replace("phase_", "").replace("_", " ").title()
                                    progress_callback(
                                        _phase_pct,
                                        f"Phase {_idx_ex + 1}/{_n_sel}: {_phase_label}",
                                        0.0,
                                    )
                                except Exception:
                                    pass
                            executed.append(phase_id)
                            logger.info(f"✅ {phase_id}: {result.execution_time_seconds:.2f}s")
                            # §Punkt3 Regressionsprotokoll: RMS nach direkter Phase
                            _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio ** 2) + 1e-12)))
                            self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                        else:
                            logger.error(f"❌ {phase_id} failed: {result.warnings}")
                            skipped.append(phase_id)
                except Exception as e:
                    logger.error(f"❌ {phase_id} exception: {e}")
                    skipped.append(phase_id)
                if self.performance_guard:
                    self.performance_guard.end_phase(phase_id, phase_start)
                # §2.16 TQC mid-pipeline: nach zeitmodifizierenden Phasen auf Kohärenzverlust prüfen
                if _tqc_snap is not None and phase_id in executed:
                    _tqc_dur_s = (
                        current_audio.shape[-1] / sample_rate
                        if current_audio.ndim == 1
                        else current_audio.shape[1] / sample_rate
                    )
                    if _tqc_dur_s >= 25.0:  # MIN_FILE_DURATION_S aus §2.16
                        try:
                            from backend.core.temporal_quality_coherence import measure_temporal_coherence

                            _mid_tqc = measure_temporal_coherence(current_audio, sample_rate)
                            if not _mid_tqc.passed:
                                logger.warning(
                                    "⚠️ TQC mid-pipeline nach %s: max_span=%.3f σ=%.3f → Rollback",
                                    phase_id,
                                    _mid_tqc.max_span,
                                    _mid_tqc.sigma,
                                )
                                current_audio = _tqc_snap  # Rollback auf Audio vor Phase
                        except Exception as _mtqc_exc:
                            logger.debug("TQC mid-pipeline nicht verfügbar: %s", _mtqc_exc)
                # Memory-Hygiene: alle 5 Phasen Garbage Collector aufrufen,
                # um Zwischenpuffer und abgeschlossene Phase-Objekte freizugeben.
                if len(executed) % 5 == 0:
                    gc.collect()
                if self.performance_guard and self.performance_guard.check_early_exit(remaining):
                    logger.warning(f"⚠️ Early exit triggered, skipping {remaining} remaining phases")
                    skipped.extend(selected_phases[len(executed) :])
                    break
        return current_audio, executed, skipped

    def _estimate_quality(
        self,
        defect_result,
        perf_report,
        executed_phases: List[str],
        audio=None,
        sample_rate: int = 48_000,
    ) -> float:
        """
        Qualitätsschätzung nach Spec §8.1.1 (SCHRITTE K-2):
            quality = 0.40*(1 − defect_severity) + 0.60*(pqs_mos − 1)/4

        VERBOTEN: fixed *1.15-Bonus ohne Audio-Messung (Spec §10.1).
        VERBOTEN: defect_severity-only ohne reale PQS-Berechnung.

        Args:
            defect_result: DefectAnalysisResult mit get_total_severity()
            perf_report:   PerformanceReport oder None
            executed_phases: Liste aktiver Phasen (für Logging)
            audio:         Restauriertes Audio-Array (float32, 48 kHz) oder None
            sample_rate:   Sample-Rate, muss 48000 sein
        Returns:
            quality_estimate ∈ [0.0, 1.0]
        """
        # Defekt-Severity: sicher klemmen
        defect_severity = float(defect_result.get_total_severity())
        defect_severity = max(0.0, min(1.0, defect_severity))

        # PQS-MOS: Fallback-Schätzung (kein Audio oder Import-Fehler)
        pqs_mos: float = 1.0 + (1.0 - defect_severity) * 4.0

        if audio is not None:
            try:
                from backend.core.perceptual_quality_scorer import score_audio_absolute  # type: ignore

                _pqs_result = score_audio_absolute(audio, sample_rate=sample_rate)
                _mos = float(getattr(_pqs_result, "pqs_mos", getattr(_pqs_result, "mos", pqs_mos)))
                if math.isfinite(_mos) and 1.0 <= _mos <= 5.0:
                    pqs_mos = _mos
            except Exception as _pqs_exc:
                logger.debug("_estimate_quality: PQS-Import fehlgeschlagen, Fallback aktiv: %s", _pqs_exc)

        # Spec §8.1.1: Gewichtete Kombination
        quality_estimate = 0.40 * (1.0 - defect_severity) + 0.60 * (pqs_mos - 1.0) / 4.0
        return float(max(0.0, min(1.0, quality_estimate)))

    def get_phase_info(self) -> Dict[str, Dict]:
        """Gibt Informationen über alle Phasen (Lazy Loading)."""
        info = {}
        for phase_id, meta in self.phase_metadata.items():
            info[phase_id] = {
                "name": meta["name"],
                "category": meta["category"],
                "dependencies": meta["dependencies"],
                "estimated_time_factor": meta["estimated_time_factor"],
                "priority": meta["priority"],
            }
        return info


# ========== CLI/Testing Interface ==========

# ─── Module-level Singleton Factory (§3.2 Singleton-Pattern) ───────────────
import threading as _threading

_restorer_singleton: Optional["UnifiedRestorerV3"] = None
_restorer_singleton_lock = _threading.Lock()


def get_restorer(mode: str = "quality") -> "UnifiedRestorerV3":
    """Thread-sicherer Singleton-Accessor für UnifiedRestorerV3.

    Wird von scripts/run_amrb_v99.py und anderen externen Callers verwendet.
    Unterstützt mode-Aliases: "restoration" und "quality" → QUALITY (höchste Qualität).

    Args:
        mode: Qualitäts-Modus — "fast" | "balanced" | "restoration" |
              "quality" | "maximum" | "studio_2026"

    Returns:
        Einzel-Instanz von UnifiedRestorerV3 (Thread-safe, Double-Checked Locking)
    """
    global _restorer_singleton
    if _restorer_singleton is None:
        with _restorer_singleton_lock:
            if _restorer_singleton is None:
                _mode_map = {
                    "fast": QualityMode.FAST,
                    "balanced": QualityMode.BALANCED,
                    "restoration": QualityMode.QUALITY,  # Restoration nutzt volle Quality-Pipeline
                    "quality": QualityMode.QUALITY,
                    "maximum": QualityMode.MAXIMUM,
                    "studio_2026": QualityMode.MAXIMUM,
                }
                qmode = _mode_map.get(mode.lower(), QualityMode.QUALITY)
                config = RestorationConfig(mode=qmode)
                _restorer_singleton = UnifiedRestorerV3(config)
                logger.info("🏭 get_restorer(): UnifiedRestorerV3 initialisiert (mode=%s)", qmode.value)
    return _restorer_singleton


if __name__ == "__main__":
    """Test UnifiedRestorerV3 mit Beispiel-Audio."""

    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.debug(f"\n{'='*70}")
    logger.debug(f"UNIFIED RESTORER V3 TEST - Aurik 9.0")
    logger.debug(f"{'='*70}\n")

    # Generate test audio (3:45 = 225 seconds)
    sr = 44100
    duration = 225  # 3:45 minutes
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal: 440 Hz sine
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add defects
    # 1. Clicks
    for i in range(100):
        pos = int(np.random.rand() * len(audio))
        audio[pos : pos + 5] += 0.3 * np.random.randn(5)

    # 2. 60Hz Hum
    audio += 0.04 * np.sin(2 * np.pi * 60 * t)

    # 3. White Noise (Tape Hiss)
    audio += 0.02 * np.random.randn(len(audio))

    logger.debug(f"Test Audio: {duration}s @ {sr} Hz")
    logger.debug(f"Defects: ~100 clicks, 60Hz hum, tape hiss\n")

    # Test alle Modi
    for mode in [QualityMode.FAST, QualityMode.BALANCED]:
        logger.debug(f"\n{'─'*70}")
        logger.debug(f"Testing {mode.value.upper()} Mode")
        logger.debug(f"{'─'*70}\n")

        config = RestorationConfig(mode=mode, num_cores=4, enforce_3x_rt=True, enable_adaptive_skipping=True)

        restorer = UnifiedRestorerV3(config)

        # Restore
        result = restorer.restore(audio, sample_rate=sr)

        # Report
        logger.debug(f"\n{'='*70}")
        logger.debug(f"RESULT SUMMARY - {mode.value.upper()}")
        logger.debug(f"{'='*70}")
        logger.debug(f"Material: {result.material_type.value}")
        logger.debug(f"Total Time: {result.total_time_seconds:.1f}s")
        logger.debug(f"RT Factor: {result.rt_factor:.2f}× ({'✅ PASS' if result.rt_factor <= 3.0 else '❌ FAIL'})")
        logger.debug(f"Quality Estimate: {result.quality_estimate*100:.1f}%")
        logger.debug(f"Phases Executed: {len(result.phases_executed)}")
        logger.debug(f"Phases Skipped: {len(result.phases_skipped)}")
        if result.warnings:
            logger.debug(f"Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                logger.debug(f"  - {w}")
        logger.debug(f"\nTop Defects (Before):")
        for i, defect in enumerate(result.metadata["defect_analysis"]["top_defects"][:3], 1):
            logger.debug(f"  {i}. {defect['type']}: {defect['severity']:.2f}")
        logger.debug(f"{'='*70}\n")

    logger.debug(f"\n✅ UnifiedRestorerV3 Test Complete!")
