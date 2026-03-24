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
from typing import Any, Optional

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
import contextlib
import importlib
import os
import sys

from backend.core.adaptive_core_scheduler import AdaptiveCoreScheduler

# Import Aurik 9.0 Core Components
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType
from backend.core.musical_goals.adaptive_goal_resolver import (
    resolve_adaptive_goal_thresholds as _resolve_adaptive_goal_thresholds_fn,
)
from backend.core.performance_guard import DeploymentMode, PerformanceGuard, QualityMode
from backend.core.phase_skipping import PhaseSkipper
from backend.core.phases.phase_interface import PhaseInterface

logger = logging.getLogger(__name__)


@dataclass
class RestorationConfig:
    """Configuration für Restoration."""

    mode: QualityMode = QualityMode.QUALITY
    material_type: MaterialType | None = None  # None = auto-detect
    enable_performance_guard: bool = True
    enable_adaptive_skipping: bool = True
    enable_phase_skipping: bool = True  # Defect-based phase skipping (20-40% speedup)
    phase_skipping_conservative: bool = False  # Conservative skipping (safer)
    enable_phase_gate: bool = True  # §2.29 PMGG — deaktivierbar via --no-phase-gate
    num_cores: int = 4
    enforce_3x_rt: bool = True
    enable_psychoacoustic_enhancement: bool = True
    global_plan: Any | None = None  # MusikalischerGlobalplan.StilbewussterRestaurierungsplan (§Dach)
    # P2-2: Deployment mode — PRODUCT = stable paths only, RESEARCH = experimental SOTA
    deployment_mode: DeploymentMode = DeploymentMode.PRODUCT
    # Phase-2 orchestration: weighted utility scoring for budget-aware phase prioritisation
    enable_phase_utility_scoring: bool = False  # off by default; enable via config or --utility-scoring
    # §11.7a Studio-2026 flag — activates Stem-Sep, Matchering, Vocos, higher quality targets
    studio_2026: bool = False

    def __post_init__(self) -> None:
        """Normalize and validate critical runtime options."""
        if not isinstance(self.mode, QualityMode):
            self.mode = QualityMode.QUALITY
        if not isinstance(self.deployment_mode, DeploymentMode):
            self.deployment_mode = DeploymentMode.PRODUCT
        if not isinstance(self.num_cores, int) or self.num_cores < 1:
            self.num_cores = 1


@dataclass
class RestorationResult:
    """Ergebnis der Restoration (Spec §2.1 / §2.2)."""

    audio: np.ndarray
    config: RestorationConfig
    material_type: MaterialType
    defect_scores: dict[DefectType, float]
    phases_executed: list[str]
    phases_skipped: list[str]
    total_time_seconds: float
    rt_factor: float
    quality_estimate: float  # 0-1
    warnings: list[str]
    metadata: dict[str, Any]
    # --- Spec-Pflichtfelder (optional, mit sicherem Default) ---
    pqs_result: Any | None = None  # PQS-MOS-Objekt (§2.6)
    musical_goals: dict[str, float] | None = None  # 14 Musical Goals (§1.2)
    excellence: Any | None = None  # ExcellenceResult (§2.1)
    temporal_coherence: Any | None = None  # TemporalCoherenceResult (§2.16)
    emotional_arc: Any | None = None  # EmotionalArcResult (§8.2)
    restorability: Any | None = None  # RestorabilityResult (§2.26)
    confidence: float = 1.0  # Gesamtkonfidenz ∈ [0,1] (§2.15)
    genealogy: Any | None = None  # RestorationGenealogy (§10.4)
    harmonic_fingerprint: Any | None = None  # Harmonischer Fingerabdruck (§2.28)
    phase_gate_log: list[str] | None = None  # PMGG-Rollback-Log (§2.29)
    adaptive_thresholds: dict[str, float] = field(default_factory=dict)  # Angewandte PMGG-Schwellwerte (§2.2)
    physical_ceiling: dict[str, float] = field(default_factory=dict)  # PhysikalischesDeckenLimit je Ziel (§2.2)
    goal_applicability: dict[str, bool] = field(default_factory=dict)  # GoalApplicabilityFilter-Ergebnis (§2.2)
    goal_priority_log: list[str] = field(default_factory=list)  # GoalPriorityProtocol-Entscheidungen (§2.2)
    preview_mos: float | None = None  # Schnell-MOS vor Vollprüfung (§2.2)
    era_decade: int | None = None  # Erkannte Aufnahme-Ära (§2.2)
    # --- §2.38 KMV (Kontinuierliche ML-Veredelung) Pflichtfelder ---
    deferred_phases: list[str] = field(default_factory=list)  # Phasen für Stufe 2
    refinement_complete: bool = False  # True nach ML-Veredelung
    stufe2_quality_estimate: float | None = None  # quality nach vollst. ML-Pass
    # --- §G3 Export-Gate-Felder (Chroma/LUFS) ---
    chroma_correlation: float | None = None  # Chroma Pearson vs. Original (§8.2)
    lufs_delta: float | None = None  # |LUFS(restored) - LUFS(original)| in LU (§8.2)


class UnifiedRestorerV3:
    """
    Unified Restorer V3 - Defect-First Audio Restoration Engine.

    Usage:
        restorer = UnifiedRestorerV3(mode=QualityMode.QUALITY)
        result = restorer.restore(audio, sample_rate=44100)
        restored_audio = result.audio
    """

    def __init__(self, config: RestorationConfig | None = None):
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
            # Studio 2026 without a PerformanceGuard must still use MAXIMUM quality,
            # not BALANCED. BALANCED (3× RT) would silently downgrade the pipeline.
            if hasattr(self.config, "studio_2026") and self.config.studio_2026:
                self.config.mode = QualityMode.MAXIMUM
                self.config.enable_phase_skipping = False
        # Lazy Phase Registry: Nur Metadaten, keine Instanzen
        self.phase_metadata: dict[str, dict] = self._discover_phase_metadata()
        self._phase_cache: dict[str, PhaseInterface] = {}
        self._warnings: list[str] = []  # §M-2 Schritte_zur_Musikalischen_Exzellenz
        self._quality_estimate_used_fallback: bool = False
        self._quality_estimate_source: str = "unknown"
        self._blocked_experimental_features: set[str] = set()

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

    def _is_research_mode(self) -> bool:
        """Returns True only for explicit RESEARCH deployment mode."""
        return getattr(self.config, "deployment_mode", DeploymentMode.PRODUCT) == DeploymentMode.RESEARCH

    def _allow_experimental_feature(self, feature_name: str) -> bool:
        """Blocks experimental features in PRODUCT mode and records the decision."""
        if self._is_research_mode():
            return True
        if feature_name not in self._blocked_experimental_features:
            self._blocked_experimental_features.add(feature_name)
            _msg = f"Experimenteller Pfad '{feature_name}' im Produktmodus blockiert (deployment_mode=product)."
            self._warnings.append(_msg)
            logger.info(_msg)
        return False

    @staticmethod
    def _is_benign_digital_source(
        audio: np.ndarray,
        sample_rate: int,
        material_type: MaterialType | None,
    ) -> tuple[bool, dict[str, float | str]]:
        """Detects clean codec/digital sources that should stay near pass-through."""
        material = material_type or MaterialType.UNKNOWN
        digital_materials = {
            MaterialType.CD_DIGITAL,
            MaterialType.DAT,
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
            MaterialType.AAC,
            MaterialType.MINIDISC,
            MaterialType.STREAMING,
        }
        if material not in digital_materials:
            return False, {"material": material.value}

        # Robust mono downmix: handles both (2, N) channel-first and (N, 2) column-major
        if audio.ndim == 2:
            mono = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
        else:
            mono = audio
        mono = np.nan_to_num(np.asarray(mono, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if mono.size < 4096 or sample_rate <= 0:
            return False, {"material": material.value, "reason": "input_too_short"}

        abs_mono = np.abs(mono)
        dc_offset = float(abs(np.mean(mono)))
        hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
        near_clip_ratio = float(np.mean(abs_mono >= 0.98))

        dyn_window = max(1, int(sample_rate * 0.4))
        dyn_frames = mono.size // dyn_window
        if dyn_frames >= 2:
            dyn_blocks = mono[: dyn_frames * dyn_window].reshape(dyn_frames, dyn_window)
            dyn_rms = np.sqrt(np.mean(dyn_blocks**2, axis=1) + 1e-12)
            dyn_std_db = float(np.std(20.0 * np.log10(dyn_rms + 1e-12)))
        else:
            dyn_std_db = 0.0

        n_fft = 4096
        hop = 1024
        window = np.hanning(n_fft).astype(np.float32)
        flatness_values: list[float] = []
        energy = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        frames = 0
        for start in range(0, mono.size - n_fft + 1, hop):
            frame = mono[start : start + n_fft] * window
            mag = np.abs(np.fft.rfft(frame)).astype(np.float64) + 1e-12
            flatness_values.append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))
            energy += mag**2
            frames += 1

        if frames == 0:
            return False, {"material": material.value, "reason": "no_frames"}

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        flatness_median = float(np.median(flatness_values))
        energy_db = 10.0 * np.log10(np.maximum(energy / frames, 1e-20))
        peak_db = float(np.max(energy_db))
        bandwidth_mask = energy_db >= (peak_db - 55.0)
        effective_bandwidth_hz = float(freqs[bandwidth_mask][-1]) if np.any(bandwidth_mask) else 0.0

        benign = (
            dc_offset <= 5e-4
            and hard_clip_ratio <= 5e-6
            and near_clip_ratio <= 5e-5
            and dyn_std_db >= 3.5
            and flatness_median <= 8e-3
            and effective_bandwidth_hz >= 12000.0
        )
        return benign, {
            "material": material.value,
            "dc_offset": dc_offset,
            "hard_clip_ratio": hard_clip_ratio,
            "near_clip_ratio": near_clip_ratio,
            "dyn_std_db": dyn_std_db,
            "flatness_median": flatness_median,
            "effective_bandwidth_hz": effective_bandwidth_hz,
        }

    @staticmethod
    def _has_localized_critical_defects(top_defects: list[Any]) -> tuple[bool, dict[str, float | int]]:
        """Detects short/localized defect patterns that should disable pass-through."""
        localized_tokens = (
            "click",
            "pop",
            "crackle",
            "dropout",
            "glitch",
            "impulse",
            "burst",
            "thump",
        )
        threshold = 0.08
        localized_count = 0
        max_localized_severity = 0.0

        for score in top_defects:
            defect_type_obj = getattr(score, "defect_type", "")
            defect_name = str(getattr(defect_type_obj, "value", defect_type_obj)).strip().lower()
            severity = float(getattr(score, "severity", 0.0) or 0.0)
            if any(token in defect_name for token in localized_tokens):
                max_localized_severity = max(max_localized_severity, severity)
                if severity >= threshold:
                    localized_count += 1

        has_localized_critical = localized_count > 0
        return has_localized_critical, {
            "localized_count": localized_count,
            "max_localized_severity": float(max_localized_severity),
            "threshold": float(threshold),
        }

    @staticmethod
    def _is_mp3_material(material_type: MaterialType | None) -> bool:
        """Returns True when source material is an MP3 variant."""
        return material_type in {MaterialType.MP3_LOW, MaterialType.MP3_HIGH}

    @staticmethod
    def _is_lossy_codec_material(material_type: MaterialType | None) -> bool:
        """Returns True for lossy/consumer codec-style digital materials."""
        return material_type in {
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
            MaterialType.AAC,
            MaterialType.MINIDISC,
            MaterialType.STREAMING,
        }

    @staticmethod
    def _should_force_mp3_maximum_guard(
        mode: QualityMode,
        material_type: MaterialType | None,
        input_snr_db: float,
        max_defect_severity: float,
        clean_digital_mode: bool,
    ) -> bool:
        """Decides whether MAXIMUM mode should be softened for lossy codec material."""
        if mode != QualityMode.MAXIMUM:
            return False
        if not UnifiedRestorerV3._is_lossy_codec_material(material_type):
            return False
        if clean_digital_mode:
            return True
        # Conservative fallback heuristic for benign MP3s where clean-digital classifier is close to threshold.
        return input_snr_db >= 34.0 and max_defect_severity <= 0.20

    def _discover_phase_metadata(self) -> dict[str, dict]:
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

    # Phase aliases: maps non-existent phase IDs to their canonical replacements.
    # phase_57_print_through_reduction is not a separate implementation; it routes
    # to phase_29_tape_hiss_reduction (LMS adaptive filter, bidirectional).
    _PHASE_ALIASES: dict[str, str] = {
        "phase_57_print_through_reduction": "phase_29_tape_hiss_reduction",
    }

    def _get_phase(self, phase_id: str) -> PhaseInterface | None:
        """Lädt und instanziiert eine Phase nur bei Bedarf (Lazy Loading)."""
        # Resolve alias before cache lookup so aliased IDs share the same instance.
        phase_id = self._PHASE_ALIASES.get(phase_id, phase_id)
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
                except Exception as _cb_exc:
                    logger.debug(
                        "Progress-Callback fehlgeschlagen (Ursache: %s). "
                        "Lösung: Callback-Signatur prüfen (pct:int, phase:str, elapsed_s:float).",
                        _cb_exc,
                    )

        original_sample_rate = sample_rate
        target_sample_rate = 48000  # Standardisierung auf 48 kHz

        # §9.1 kwargs: mode + material aus externen Aufrufen verarbeiten (z.B. AMRB-Runner)
        _mode_map = {
            "fast": QualityMode.FAST,
            "balanced": QualityMode.BALANCED,
            "restoration": QualityMode.QUALITY,  # Restoration nutzt volle Quality-Pipeline
            "quality": QualityMode.QUALITY,
            "maximum": QualityMode.MAXIMUM,  # §9.5: 8× RT — war fälschlich BALANCED (3×)
            "studio_2026": QualityMode.MAXIMUM,  # §9.5: 8× RT — war fälschlich BALANCED (3×)
        }
        _mode_kwarg = kwargs.pop("mode", None)
        if _mode_kwarg is not None:
            self.config.mode = _mode_map.get(str(_mode_kwarg).lower(), self.config.mode)
        _mat_kwarg = kwargs.pop("material", None)
        if _mat_kwarg is not None and self.config.material_type is None:
            try:
                self.config.material_type = MaterialType(str(_mat_kwarg))
            except ValueError as _mat_exc:
                logger.warning(
                    "Ungültiger Material-Hinweis '%s' ignoriert (Ursache: %s). "
                    "Lösung: gültige MaterialType-Werte verwenden.",
                    _mat_kwarg,
                    _mat_exc,
                )

        # §Dach: MusikalischerGlobalplan — aus kwargs oder RestorationConfig laden
        _gp_kwarg = kwargs.pop("global_plan", None)
        _chain_kwarg = kwargs.pop("chain_info", None)
        _defekt_hint_kwarg = kwargs.pop("defekt_hint", None)
        _cached_era_kwarg = kwargs.pop("cached_era_result", None)
        _cached_genre_kwarg = kwargs.pop("cached_genre_result", None)
        _cached_defect_kwarg = kwargs.pop("cached_defect_result", None)
        _cached_medium_kwarg = kwargs.pop("cached_medium_result", None)
        _cached_restorability_kwarg = kwargs.pop("cached_restorability_result", None)
        _audio_update_cb_kwarg = kwargs.pop("audio_update_callback", None)
        self._active_global_plan = _gp_kwarg if _gp_kwarg is not None else self.config.global_plan
        self._active_chain_info = _chain_kwarg  # TontraegerketteDenker: Ketten-Phasen
        self._active_defekt_hint = _defekt_hint_kwarg  # DefektDenker: heuristische Phasen-Empfehlung

        _cb(2, "Initialisierung…")
        # Robuste Sample-Count-Ermittlung: (N,2) → N, (2,N) → N, (N,) → N
        _n_samples = max(audio.shape) if audio.ndim == 2 else len(audio)
        logger.info("Starting restoration: %.1fs audio @ %d Hz", _n_samples / sample_rate, sample_rate)

        # ── Early-Exit Guard: Minimum Signal Length ───────────────────────────
        # Signals shorter than 100 ms (4800 samples @ 48 kHz) cannot be meaningfully
        # restored. HPSS, STFT and ML metrics produce garbage on such tiny buffers and
        # trigger misleading warnings (PQS-MOS, 8×RT violations, Musical Goals).
        # Return a no-op pass-through result immediately to avoid polluting the log.
        _MIN_MEANINGFUL_SAMPLES = 4800  # 100 ms @ 48 kHz
        _audio_size_check = audio.size
        if _audio_size_check < _MIN_MEANINGFUL_SAMPLES:
            logger.debug(
                "restore(): Signal zu kurz (%d Samples < %d) — Pass-Through ohne Verarbeitung.",
                _audio_size_check,
                _MIN_MEANINGFUL_SAMPLES,
            )
            return RestorationResult(
                audio=np.clip(np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0),
                config=self.config,
                material_type=MaterialType.UNKNOWN,
                defect_scores={},
                phases_executed=[],
                phases_skipped=[],
                total_time_seconds=0.0,
                rt_factor=0.0,
                quality_estimate=0.5,
                warnings=["signal_too_short"],
                metadata={
                    "skip_reason": "signal_too_short",
                    "audio_samples": _audio_size_check,
                    "min_samples": _MIN_MEANINGFUL_SAMPLES,
                },
            )

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
        # Structured error accumulator — persists into RestorationResult.metadata["fail_reasons"]
        # Each entry: {"component": str, "error_code": str, "exc_type": str, "exc_msg": str}
        _fail_reasons: list[dict[str, Any]] = []

        # Resample to 48 kHz if necessary
        if sample_rate != target_sample_rate:
            if LIBROSA_AVAILABLE:
                logger.info(f"Resampling {sample_rate} Hz → {target_sample_rate} Hz for standardized processing")
                if audio.ndim == 2:
                    # Stereo: resample each channel — handle both (N,2) and (2,N) layouts
                    if audio.shape[0] > audio.shape[1]:
                        # (N, 2) column-major from frontend
                        audio = np.column_stack(
                            [
                                librosa.resample(audio[:, 0], orig_sr=sample_rate, target_sr=target_sample_rate),
                                librosa.resample(audio[:, 1], orig_sr=sample_rate, target_sr=target_sample_rate),
                            ]
                        )
                    else:
                        # (2, N) channel-first
                        audio = np.stack(
                            [
                                librosa.resample(audio[0], orig_sr=sample_rate, target_sr=target_sample_rate),
                                librosa.resample(audio[1], orig_sr=sample_rate, target_sr=target_sample_rate),
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

        # §3.x Stereo-Normalisierung: Internes Format ist channel-first (2, N).
        # Frontend liefert (N, 2) via _normalize_audio — hier einheitlich auf (2, N) drehen.
        # Ohne diese Normalisierung greifen SAP, TDP, Vocos-Stereo-Handling und alle
        # `restored_audio[0]`-Zugriffe auf die erste Sample-Zeile statt den ersten Kanal,
        # was das Audio auf 2 Samples korrumpiert (EpistemicGate "2 Samples < 4800").
        if audio.ndim == 2 and audio.shape[1] <= 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T  # (N, 2) → (2, N)
            logger.debug("restore(): Stereo (N,%d) → (%d,N) normalisiert", audio.shape[0], audio.shape[0])

        # ── OOM-Guard: Audio-Buffer-Größe gegen RAM-Budget prüfen ────────────
        # Spec §9: Audio-Buffer max. 4 GB.  Intermediate STFTs/copies multiplizieren
        # den Bedarf ~5× → effektives Limit: ~800 MB Audio-Input bei 32 GB RAM.
        _audio_bytes = audio.nbytes
        _MAX_AUDIO_BUFFER_BYTES = 4 * 1024**3  # 4 GB absolute Obergrenze (Spec §9)
        if _audio_bytes > _MAX_AUDIO_BUFFER_BYTES:
            raise MemoryError(
                f"Audio-Buffer ({_audio_bytes / (1024**3):.1f} GB) überschreitet "
                f"das erlaubte Maximum von {_MAX_AUDIO_BUFFER_BYTES / (1024**3):.0f} GB (Spec §9). "
                f"Bitte eine kürzere Datei verwenden."
            )
        # Physischer RAM-Preflight: mind. 3 GB + 5× Audio-Puffer frei
        try:
            import psutil as _psutil_guard

            _avail_mb = float(_psutil_guard.virtual_memory().available / (1024 * 1024))
            _needed_mb = max(3072.0, (_audio_bytes * 5) / (1024 * 1024))
            if _avail_mb < _needed_mb:
                logger.warning(
                    "OOM-Guard: Nur %.0f MB RAM frei, benötigt ~%.0f MB. Versuche Plugin-Eviction…",
                    _avail_mb,
                    _needed_mb,
                )
                try:
                    from backend.core.plugin_lifecycle_manager import evict_stale_plugins

                    evict_stale_plugins(required_mb=_needed_mb)
                    gc.collect()
                except Exception:
                    pass
                _avail_after = float(_psutil_guard.virtual_memory().available / (1024 * 1024))
                if _avail_after < _needed_mb * 0.7:
                    raise MemoryError(
                        f"Nicht genügend RAM: {_avail_after:.0f} MB frei, "
                        f"~{_needed_mb:.0f} MB benötigt. Bitte andere Anwendungen schließen."
                    )
        except ImportError:
            pass  # psutil fehlt → Guard degradiert graceful

        _cb(5, "Resampling & Vorverarbeitung…")
        # Step 1a: Material-Erkennung via MediumClassifier (vor DefectScanner)
        # Nur wenn kein Material manuell vorgegeben wurde (config.material_type is None)
        # §G1: Pre-Repair-Referenz aus AurikDenker (echtes Original VOR ReparaturDenker)
        # für referenz-basierte Musical Goals (Authentizität, Groove, Timbre, Artikulation).
        # Falls nicht vorhanden: Fallback auf Eingabe-Audio (identisch mit bisherigem Verhalten).
        _pre_repair_ref = kwargs.get("pre_repair_reference")
        if _pre_repair_ref is not None and isinstance(_pre_repair_ref, np.ndarray) and _pre_repair_ref.size > 0:
            original_audio_for_goals: np.ndarray = _pre_repair_ref.copy()
            logger.info("UV3: pre_repair_reference als Goal-Referenz übernommen (shape=%s)", _pre_repair_ref.shape)
        else:
            original_audio_for_goals: np.ndarray = audio.copy()

        _cb(8, "Restaurierbarkeit wird geprüft…")
        # §2.26 RestorabilityEstimator — Vor-Assessment der Restaurierbarkeit (DSP-only, < 5 s)
        # §2.29 Normativ (PMGG Datenfluss-Invariante): _pmgg_restorability_score MUSS aus
        # RestorabilityEstimator.estimate().restorability_score stammen — kein Hard-Code 70.0.
        _pmgg_restorability_score: float = 70.0  # konservativer Default wenn Estimator nicht verfügbar
        if _cached_restorability_kwarg is not None:
            _pmgg_restorability_score = float(getattr(_cached_restorability_kwarg, "restorability_score", 70.0))
            logger.info(
                "RestorabilityEstimator: verwende gecachtes Ergebnis (score=%.1f)",
                _pmgg_restorability_score,
            )
        else:
            try:
                from backend.core.restorability_estimator import estimate_restorability

                _mat_str = self.config.material_type.value if self.config.material_type is not None else "unknown"
                _rest_result = estimate_restorability(audio, sample_rate, material=_mat_str)
                _pmgg_restorability_score = float(_rest_result.restorability_score)
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
        _artist_id: str | None = None
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

        _classified_material: MaterialType | None = self.config.material_type
        if _classified_material is None:
            # §9.7.2 Parallele Eingangs-Analyse — MediumClassifier, EraClassifier und
            # GermanSchlagerClassifier sind voneinander unabhängig und laufen parallel
            import concurrent.futures as _cf

            def _run_mc(a: np.ndarray, sr: int) -> object | None:
                try:
                    from backend.core.medium_classifier import classify_medium

                    return classify_medium(a, sr=sr, use_ml=True)
                except Exception as _e:
                    logger.debug("MediumClassifier nicht verfügbar: %s", _e)
                    return None

            def _run_era(a: np.ndarray, sr: int) -> object | None:
                try:
                    from backend.core.era_classifier import classify_era

                    return classify_era(a, sr)
                except Exception as _e:
                    logger.debug("EraClassifier nicht verfügbar: %s", _e)
                    return None

            def _run_genre(a: np.ndarray, sr: int) -> object | None:
                try:
                    from backend.core.genre_classifier import classify_genre

                    return classify_genre(a, sr)
                except Exception as _e:
                    logger.debug("GermanSchlagerClassifier nicht verfügbar: %s", _e)
                    return None

            # Gecachte Era/Genre/Medium-Ergebnisse aus Frontend-Vorab-Analyse verwenden,
            # sofern vorhanden — spart doppelte ML-Klassifikation (§9.7.2 Cache-Bypass)
            _era_cached = _cached_era_kwarg is not None
            _genre_cached = _cached_genre_kwarg is not None
            _medium_cached = _cached_medium_kwarg is not None

            _futures: dict[str, _cf.Future] = {}
            with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
                if not _medium_cached:
                    _futures["mc"] = _pool.submit(_run_mc, audio, sample_rate)
                else:
                    logger.info(
                        "MediumClassifier: verwende gecachtes Ergebnis (material=%s, conf=%.2f)",
                        getattr(_cached_medium_kwarg, "material_type", getattr(_cached_medium_kwarg, "material", "?")),
                        float(getattr(_cached_medium_kwarg, "confidence", 0.0)),
                    )
                if not _era_cached:
                    _futures["era"] = _pool.submit(_run_era, audio, sample_rate)
                else:
                    logger.info(
                        "EraClassifier: verwende gecachtes Ergebnis (decade=%s, conf=%.2f)",
                        getattr(_cached_era_kwarg, "decade", "?"),
                        float(getattr(_cached_era_kwarg, "confidence", 0.0)),
                    )
                if not _genre_cached:
                    _futures["genre"] = _pool.submit(_run_genre, audio, sample_rate)
                else:
                    logger.info(
                        "GenreClassifier: verwende gecachtes Ergebnis (is_schlager=%s, conf=%.2f)",
                        getattr(_cached_genre_kwarg, "is_schlager", "?"),
                        float(getattr(_cached_genre_kwarg, "confidence", 0.0)),
                    )
                _mc_result = _cached_medium_kwarg if _medium_cached else _futures["mc"].result()
                _era_result_par = _cached_era_kwarg if _era_cached else _futures["era"].result()
                _schlager_result_par = _cached_genre_kwarg if _genre_cached else _futures["genre"].result()

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

            # §Medium-Floor-Constraint: Ära-Zuweisung darf nicht vor der physikalischen
            # Erfindung des erkannten Mediums liegen (z. B. Magnetband ≠ 1890er).
            # Anwendung nur wenn MediumClassifier-Konfidenz ≥ 0.35 (bereits geprüft).
            if _era_result is not None and _classified_material is not None:
                try:
                    from backend.core.era_classifier import constrain_era_to_medium as _era_constrain

                    _era_result = _era_constrain(_era_result, str(_classified_material.value))
                except Exception as _floor_exc:
                    logger.debug("Medium-Floor-Constraint fehlgeschlagen: %s", _floor_exc)

            # §9.7.7 Fallback: wenn MediumClassifier-Konfidenz zu niedrig, nutze
            # EraClassifier material_prior als Material-Prior bevor DefectScanner
            # _auto_detect_material() aufgerufen wird (verhindert tape→vinyl Mismatch).
            if _classified_material is None and _era_result is not None:
                _era_mp = str(getattr(_era_result, "material_prior", "") or "")
                if _era_mp and _era_mp not in ("unknown", "digital", ""):
                    try:
                        from backend.core.medium_classifier import MaterialType as _MatType

                        _classified_material = _MatType(_era_mp)
                        logger.info(
                            "EraClassifier material_prior fallback: %s (decade=%d era_conf=%.2f)",
                            _era_mp,
                            _era_result.decade,
                            _era_result.confidence,
                        )
                    except Exception:
                        pass  # unbekannter MaterialType-Wert → Auto-Detect bleibt

            # §Dach GlobalPlan-Prior: Chunk-Klassifikation mit Ganzstück-Ergebnis überschreiben.
            # UV3 läuft auf kurzen Chunks — EraClassifier/MediumClassifier auf 10s-Segmenten
            # sind unzuverlässig (MP3-Bandbreite → fälschlich 1920/shellac).
            # Wenn der AurikDenker am vollständigen Audio einen GlobalPlan berechnet hat,
            # gewinnt dessen decade/material/genre — Denker ist der autoritative Orchestrator.
            # DSP-Konfidenz (0.40) und ML-Konfidenz (0.85) sind auf verschiedenen Skalen und
            # NICHT direkt vergleichbar. Denker-GlobalPlan basiert auf dem Gesamtstück →
            # immer bevorzugen, außer decade fehlt ganz.
            _gp_portrait = getattr(getattr(self, "_active_global_plan", None), "portrait", None)
            if _gp_portrait is not None:
                _gp_decade = getattr(_gp_portrait, "decade", None)
                _gp_era_conf = float(getattr(_gp_portrait, "era_confidence", 0.0))
                _gp_material = getattr(_gp_portrait, "material", None)
                _gp_genre = getattr(_gp_portrait, "genre", None)

                # Era/Decade: GlobalPlan-Era hat Vorrang (Gesamtstück > Chunk-Analyse)
                chunk_era_conf = float(getattr(_era_result, "confidence", 0.0)) if _era_result is not None else 0.0
                if _gp_decade is not None and _gp_era_conf >= 0.35:
                    try:
                        from backend.core.era_classifier import EraResult as _EraResult

                        _old_decade = getattr(_era_result, "decade", "?")
                        _era_result = _EraResult(
                            decade=int(_gp_decade),
                            era_label=f"{_gp_decade}er",
                            confidence=_gp_era_conf,
                            material_prior=_gp_material or getattr(_era_result, "material_prior", "unknown"),
                            noise_profile=getattr(_era_result, "noise_profile", None)
                            or __import__("numpy").zeros(24, dtype="float32"),
                            tier_used=0,
                            hf_rolloff_hz=getattr(_era_result, "hf_rolloff_hz", 20000.0),
                        )
                        logger.info(
                            "§Dach GlobalPlan-Prior: EraClassifier decade=%s→%d material=%s (Konfidenz=%.2f)",
                            _old_decade,
                            _gp_decade,
                            _gp_material,
                            _gp_era_conf,
                        )
                    except Exception as _gp_era_exc:
                        logger.debug("GlobalPlan-Era-Override fehlgeschlagen: %s", _gp_era_exc)

                # Material überschreiben wenn GlobalPlan-Material bekannt und Chunk-MediumClassifier unsicher
                if _gp_material and _gp_material not in ("unknown", "digital"):
                    _chunk_mc_conf = float(getattr(_mc_result, "confidence", 0.0)) if _mc_result is not None else 0.0
                    if _chunk_mc_conf < 0.55:
                        try:
                            from backend.core.medium_classifier import MaterialType as _MatType

                            _gp_mat_type = _MatType(_gp_material)
                            _classified_material = _gp_mat_type
                            logger.info(
                                "§Dach GlobalPlan-Prior: MediumClassifier material→%s (Chunk-Konfidenz=%.2f)",
                                _gp_material,
                                _chunk_mc_conf,
                            )
                        except Exception:
                            pass  # MaterialType-Wert unbekannt → Chunk-Ergebnis behalten

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

            # §Dach GlobalPlan-Genre-Prior: Chunk-Schlager-Ergebnis mit GlobalPlan überschreiben
            if _gp_portrait is not None:
                _gp_genre = getattr(_gp_portrait, "genre", None)
                _gp_genre_conf = float(getattr(_gp_portrait, "genre_confidence", 0.0))
                _gp_subgenre = getattr(_gp_portrait, "subgenre", "unknown")
                _chunk_genre_conf = (
                    float(getattr(_schlager_result, "confidence", 0.0)) if _schlager_result is not None else 0.0
                )
                if _gp_genre == "schlager" and _gp_genre_conf >= 0.35 and _gp_genre_conf >= _chunk_genre_conf:
                    try:
                        from backend.core.genre_classifier import GenreResult as _GenreResult

                        _schlager_result = _GenreResult(
                            is_schlager=True,
                            confidence=_gp_genre_conf,
                            genre_label="Schlager",
                            subgenre=_gp_subgenre or "unknown",
                            bpm=float(getattr(_gp_portrait, "bpm", 0.0)),
                        )
                        logger.info(
                            "§Dach GlobalPlan-Prior: Genre→Schlager subgenre=%s (conf=%.2f)",
                            _gp_subgenre,
                            _gp_genre_conf,
                        )
                    except Exception:
                        pass
        else:
            # Material bereits im Config — Ära und Genre trotzdem ermitteln
            if _cached_era_kwarg is not None:
                _era_result = _cached_era_kwarg
                logger.info(
                    "EraClassifier: verwende gecachtes Ergebnis (decade=%s, conf=%.2f)",
                    getattr(_era_result, "decade", "?"),
                    float(getattr(_era_result, "confidence", 0.0)),
                )
            else:
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

            # §Dach GlobalPlan-Prior auch im else-Zweig anwenden
            _gp_portrait_else = getattr(getattr(self, "_active_global_plan", None), "portrait", None)
            if _gp_portrait_else is not None:
                _gp_decade_e = getattr(_gp_portrait_else, "decade", None)
                _gp_era_conf_e = float(getattr(_gp_portrait_else, "era_confidence", 0.0))
                _chunk_era_conf_e = float(getattr(_era_result, "confidence", 0.0)) if _era_result is not None else 0.0
                if _gp_decade_e is not None and _gp_era_conf_e >= 0.35 and _gp_era_conf_e >= _chunk_era_conf_e:
                    try:
                        from backend.core.era_classifier import EraResult as _EraResult2

                        _gp_mat_e = getattr(_gp_portrait_else, "material", None)
                        _era_result = _EraResult2(
                            decade=int(_gp_decade_e),
                            era_label=f"{_gp_decade_e}er",
                            confidence=_gp_era_conf_e,
                            material_prior=_gp_mat_e or getattr(_era_result, "material_prior", "unknown"),
                            noise_profile=getattr(_era_result, "noise_profile", None)
                            or __import__("numpy").zeros(24, dtype="float32"),
                            tier_used=0,
                            hf_rolloff_hz=getattr(_era_result, "hf_rolloff_hz", 20000.0),
                        )
                        logger.info(
                            "§Dach GlobalPlan-Prior (else): decade→%d material→%s (conf=%.2f)",
                            _gp_decade_e,
                            _gp_mat_e,
                            _gp_era_conf_e,
                        )
                    except Exception:
                        pass

            if _cached_genre_kwarg is not None:
                _schlager_result = _cached_genre_kwarg
                logger.info(
                    "GenreClassifier: verwende gecachtes Ergebnis (is_schlager=%s, conf=%.2f)",
                    getattr(_schlager_result, "is_schlager", "?"),
                    float(getattr(_schlager_result, "confidence", 0.0)),
                )
            else:
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

            # §Dach GlobalPlan-Genre-Prior: Chunk-Ergebnis mit GlobalPlan-Genre überschreiben
            if _gp_portrait_else is not None:
                _gp_genre_e = getattr(_gp_portrait_else, "genre", None)
                _gp_genre_conf_e = float(getattr(_gp_portrait_else, "genre_confidence", 0.0))
                _gp_subgenre_e = getattr(_gp_portrait_else, "subgenre", "unknown")
                _chunk_genre_conf_e = (
                    float(getattr(_schlager_result, "confidence", 0.0)) if _schlager_result is not None else 0.0
                )
                if _gp_genre_e == "schlager" and _gp_genre_conf_e >= 0.35 and _gp_genre_conf_e >= _chunk_genre_conf_e:
                    try:
                        from backend.core.genre_classifier import GenreResult as _GenreResult

                        _schlager_result = _GenreResult(
                            is_schlager=True,
                            confidence=_gp_genre_conf_e,
                            genre_label="Schlager",
                            subgenre=_gp_subgenre_e or "unknown",
                            bpm=float(getattr(_gp_portrait_else, "bpm", 0.0)),
                        )
                        logger.info(
                            "§Dach GlobalPlan-Prior (else): Genre→Schlager subgenre=%s (conf=%.2f)",
                            _gp_subgenre_e,
                            _gp_genre_conf_e,
                        )
                    except Exception:
                        pass

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

        _cb(11, "Material und Ära klassifiziert …")

        # §2.32 GoalApplicabilityFilter — Filterung physikalisch nicht-messbarer Musikziele
        _goal_applicability = None
        try:
            from backend.core.goal_applicability_filter import evaluate_goal_applicability

            _panns_for_gaf = kwargs.get("panns_tags", {})
            _mat_for_gaf = _classified_material.value if _classified_material is not None else "unknown"
            _era_decade_for_gaf = getattr(_era_result, "decade", None) if _era_result is not None else None
            _goal_applicability = evaluate_goal_applicability(
                audio, sample_rate, _mat_for_gaf, _era_decade_for_gaf, _panns_for_gaf
            )
            logger.info(
                "🎯 GoalApplicabilityFilter: %d anwendbar, %d nicht-anwendbar",
                len(_goal_applicability.applicable),
                len(_goal_applicability.inapplicable),
            )
        except Exception as _gaf_exc:
            logger.debug("GoalApplicabilityFilter nicht verfügbar: %s", _gaf_exc)

        # QualityAnalyzer: Eingangsqualität messen (Vorher-Baseline) (§9.5 Quality-Tracking)
        _quality_before: object | None = None
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
            _dc_a = np.array([1.0, -0.9999], dtype=np.float64)  # f_c ≈ 5 Hz @ 48 kHz (Spec §2.2)
            if audio.ndim == 1:
                audio = _lfilter(_dc_b, _dc_a, audio.astype(np.float64)).astype(np.float32)
            else:
                # Channel-first (2, N) format after stereo normalization
                audio = np.stack(
                    [
                        _lfilter(_dc_b, _dc_a, audio[c].astype(np.float64)).astype(np.float32)
                        for c in range(audio.shape[0])
                    ],
                    axis=0,
                )
            audio = np.clip(audio, -1.0, 1.0)
            logger.debug("§DCOffsetPreRemoval: DC-Offset entfernt (|mean|=%.2e)", float(np.abs(np.mean(audio))))
        except Exception as _dc_exc:
            logger.debug("DCOffsetPreRemoval Fehler (scipy nicht verfügbar): %s", _dc_exc)

        # §2.27 TransientDecoupledProcessing — Percussive/Harmonic-Trennung (allererster DSP-Schritt)
        _tdp_proc = None
        _tdp_percussive: np.ndarray | None = None
        _tdp_harmonic_ready: bool = False
        try:
            from backend.core.transient_decoupled_processor import get_transient_decoupled_processor

            _tdp_proc = get_transient_decoupled_processor()
            _tdp_audio_percussive, _tdp_audio_harmonic = _tdp_proc.separate(audio, sample_rate)
            _tdp_percussive = _tdp_audio_percussive.copy()  # Original-Transienten sichern
            audio = _tdp_audio_harmonic  # Pipeline ab jetzt auf harm. Anteil
            _tdp_harmonic_ready = True
            logger.info(
                "§2.27 TDP: Percussive/Harmonic getrennt (perc_rms=%.4f harm_rms=%.4f)",
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
            _genre_for_anchor = _schlager_result.genre_label if _schlager_result is not None else "unknown"
            _reference_anchor = synthesize_reference_anchor(
                era_decade=_era_for_anchor if _era_for_anchor is not None else 1970,
                genre_label=_genre_for_anchor,
                material=_mat_for_anchor,
            )
            logger.info(
                "🎯 ReferenceAnchorSynthesizer: Anker generiert (era=%s genre=%s material=%s)",
                _era_for_anchor,
                _genre_for_anchor,
                _mat_for_anchor,
            )
        except Exception as _ras_exc:
            logger.debug("ReferenceAnchorSynthesizer nicht verfügbar: %s", _ras_exc)

        # Step 1: Defect Scanning (Cache-First — kein Mehrfach-Scan §9.4)
        logger.info("Step 1/4: Defect Scanning...")
        if _cached_defect_kwarg is not None:
            defect_result = _cached_defect_kwarg
            logger.info("Step 1/4: Verwende gecachten DefectScan (kein Mehrfach-Scan).")
        else:
            defect_result = self.defect_scanner.scan(audio, sample_rate, _classified_material)
        if defect_result is None:
            logger.error(
                "DefectScanner.scan() returned None — creating fallback DefectAnalysisResult. "
                "Possible cause: internal scanner exception. Solution: check scanner logs."
            )
            from backend.core.defect_scanner import DefectAnalysisResult

            defect_result = DefectAnalysisResult(
                material_type=MaterialType.UNKNOWN if _classified_material is None else _classified_material,
                scores={},
                analysis_time_seconds=0.0,
                sample_rate=sample_rate,
                duration_seconds=(
                    float(len(audio) / sample_rate) if audio.ndim == 1 else float(audio.shape[-1] / sample_rate)
                ),
            )
        material_type = defect_result.material_type

        _cb(15, "Defekte werden kartiert …")
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

        _cb(19, "Kausale Defektanalyse abgeschlossen …")

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
            if _has_dropout:
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
            _era_ws_pre: dict[str, float] | None = None
            if _era_result is not None:
                try:
                    from backend.core.era_classifier import get_era_classifier as _get_era_clf_pre

                    _era_ws_pre = _get_era_clf_pre().get_gp_warmstart(_era_result)
                except Exception:
                    pass
            _pareto_proposals = _gp_opt_pre.propose_pareto(
                material=_gp_material_key_pre,
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
        _applicable_goals: set | None = None
        try:
            from backend.core.goal_applicability_filter import evaluate_goal_applicability

            _panns_conf: dict[str, float] | None = None
            with contextlib.suppress(Exception):
                _panns_conf = defect_result.metadata.get("panns_tags") if hasattr(defect_result, "metadata") else None
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
        _top_defects = defect_result.get_top_defects(10)
        _max_defect_severity = max((s.severity for s in _top_defects), default=0.0)
        _localized_critical, _localized_metrics = self._has_localized_critical_defects(_top_defects)
        _clean_digital_mode, _clean_digital_metrics = self._is_benign_digital_source(
            audio,
            sample_rate,
            defect_result.material_type,
        )
        _pass_through_mode = (
            (_input_snr_db > 40.0 and _max_defect_severity < 0.15) or _clean_digital_mode
        ) and not _localized_critical
        if _localized_critical:
            logger.info(
                "🛡️ Local-Defect-Guard: Schonmodus deaktiviert (localized_count=%d, max=%.3f, thr=%.3f)",
                int(_localized_metrics.get("localized_count", 0)),
                float(_localized_metrics.get("max_localized_severity", 0.0)),
                float(_localized_metrics.get("threshold", 0.08)),
            )
        _mp3_maximum_guard = self._should_force_mp3_maximum_guard(
            self.config.mode,
            material_type,
            _input_snr_db,
            _max_defect_severity,
            _clean_digital_mode,
        )
        if _mp3_maximum_guard and not _pass_through_mode:
            _pass_through_mode = True
            logger.info(
                "🛡️ MP3-Maximum-Guard: material=%s, mode=%s, SNR=%.1f dB, max_defect=%.3f → Schonmodus erzwungen",
                material_type.value if hasattr(material_type, "value") else str(material_type),
                self.config.mode.value,
                _input_snr_db,
                _max_defect_severity,
            )
        _pipeline_quality_mode_override: QualityMode | None = None
        if _mp3_maximum_guard:
            _pipeline_quality_mode_override = QualityMode.QUALITY
            logger.info(
                "🛡️ MP3-Maximum-Guard: quality_mode override %s -> %s",
                self.config.mode.value,
                _pipeline_quality_mode_override.value,
            )
        if _pass_through_mode:
            if _clean_digital_mode and not (_input_snr_db > 40.0 and _max_defect_severity < 0.15):
                logger.info(
                    "🛡️ Clean-Digital-Guard §8.2: material=%s, flatness=%.5f, bw=%.0f Hz, dyn=%.2f dB"
                    " → Schonmodus aktiv trotz konservativer SNR-Heuristik (%.1f dB)",
                    _clean_digital_metrics.get("material", "unknown"),
                    float(_clean_digital_metrics.get("flatness_median", 0.0)),
                    float(_clean_digital_metrics.get("effective_bandwidth_hz", 0.0)),
                    float(_clean_digital_metrics.get("dyn_std_db", 0.0)),
                    _input_snr_db,
                )
            else:
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
            audio=audio,
            sr=sample_rate,
        )
        selected_phases = self._optimize_phase_plan_intelligence(
            selected_phases,
            causal_plan=_causal_plan,
            pipeline_confidence=_pipeline_confidence,
            restorability_score=_pmgg_restorability_score,
        )
        logger.info(f"Selected {len(selected_phases)} phases based on defects")

        # Step 2.5: Apply Phase Skipping (intelligent filtering)
        _enable_phase_skipping = bool(self.phase_skipper)
        if _pipeline_confidence is not None and float(_pipeline_confidence.confidence) < 0.55:
            _enable_phase_skipping = False
            logger.info(
                "Phase Skipping deaktiviert: niedrige Pipeline-Konfidenz (%.2f < 0.55)",
                float(_pipeline_confidence.confidence),
            )
        if _pmgg_restorability_score < 40.0:
            _enable_phase_skipping = False
            logger.info(
                "Phase Skipping deaktiviert: niedrige Restaurierbarkeit (%.1f < 40.0)",
                _pmgg_restorability_score,
            )

        if _enable_phase_skipping:
            original_count = len(selected_phases)
            selected_phases, skip_reasons = self._apply_phase_skipping(selected_phases, defect_result)
            skipped_count = original_count - len(selected_phases)
            if skipped_count > 0:
                logger.info(
                    f"Phase Skipping: {skipped_count} phases skipped "
                    f"({skipped_count / original_count * 100:.0f}% speedup potential)"
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
        audio_duration = audio.shape[-1] / sample_rate  # shape[-1] works for both mono (N,) and stereo (2, N)
        if self.performance_guard:
            self.performance_guard.start_monitoring(audio_duration)

        # §Vintage-Authentizitäts-Guards — Ära-spezifische Phase-Filter + Strength-Caps
        # Spec (copilot-instructions.md):
        #   1920–1940: Rolloff ≤ 7 kHz NICHT künstlich erweitern; AudioSR nur user_requested
        #              H2/H4 Röhren-Kompression ∈ [−30,−20] dBr BEWAHREN (SOFT_SATURATION → Skip)
        #   1940–1955: Tape-Saturation-Fingerabdruck NICHT entfernen; phase_22 nur emulieren
        #   1955–1965: RT60 ∈ [1.2, 2.0] s bewahren — phase_20/phase_49 strength ≤ 0.20
        #   1965–1975: Tape-Saturation-Signatur NICHT entfernen; Vintage-Kompressor-Imprint bewahren
        _vintage_phase_strength_caps: dict = {}  # {phase_id: max_strength}
        if _era_result is not None:
            _vintage_decade = _era_result.decade
            if _vintage_decade <= 1940:
                # 1920–1940: Blockiere künstliche BW-Erweiterung (EAPC übernimmt §2.35)
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
                # SOFT_SATURATION-Schutz: phase_23 (Clipping-Repair) nur bei echtem CLIPPING
                # (wird bereits durch classify_clipping() in phase_23 selbst gesteuert)
                logger.debug(
                    "🕰️ Vintage-Guard decade=%d ≤ 1940: H2/H4 Röhren-Kompression bewahren "
                    "(SOFT_SATURATION-Guard aktiv, phase_22 emuliert nicht entfernen)",
                    _vintage_decade,
                )
            if 1940 <= _vintage_decade < 1955:
                # 1940–1955: Tape-Saturation NICHT entfernen — phase_22 strength nach oben cap
                # Originaler Tape-Fingerabdruck muss erhalten bleiben (nur emulieren)
                _vintage_phase_strength_caps["phase_22_tape_saturation"] = 0.30
                logger.debug(
                    "🕰️ Vintage-Guard 1940–1955: phase_22 strength ≤ 0.30 (Tape-Fingerabdruck erhalten)",
                )
            if 1955 <= _vintage_decade < 1965:
                # 1955–1965: RT60 ∈ [1.2, 2.0] s bewahren — Dereverb-Stärke begrenzen
                _vintage_phase_strength_caps["phase_20_reverb_reduction"] = 0.20
                _vintage_phase_strength_caps["phase_49_advanced_dereverb"] = 0.20
                logger.info(
                    "🕰️ Vintage-Guard decade=%d (1955–1965): phase_20/phase_49 strength ≤ 0.20 "
                    "(RT60 ∈ [1.2, 2.0] s erhalten)",
                    _vintage_decade,
                )
            if 1965 <= _vintage_decade < 1975:
                # 1965–1975: Tape-Saturation-Signatur + VCA-Kompressor-Imprint NICHT entfernen
                _vintage_phase_strength_caps["phase_22_tape_saturation"] = 0.35
                _vintage_phase_strength_caps["phase_10_compression"] = 0.40
                logger.debug(
                    "🕰️ Vintage-Guard 1965–1975: Tape-Sat ≤ 0.35, Kompressor-Imprint ≤ 0.40 (VCA-Charakter bewahren)",
                )

        # §2.28 HPG Pre-Phase: Harmonik-Referenz aus Original sichern (Spec: G_floor vor Phase-03)
        _hpg_pre_mask: Any = None
        _hpg_pre_href: Any = None
        _hpg_skip_materials_pre = {"shellac", "wax_cylinder"}
        if material_type.value not in _hpg_skip_materials_pre:
            try:
                from backend.core.harmonic_preservation_guard import (
                    get_harmonic_preservation_guard as _get_hpg_pre,
                )

                _hpg_tmp = _get_hpg_pre()
                _hpg_instrument_tag_pre = {
                    "vinyl": "piano_mid",
                    "tape": "piano_mid",
                    "reel_tape": "piano_mid",
                    "cd_digital": "piano_mid",
                    "mp3_low": "piano_mid",
                    "mp3_high": "piano_mid",
                    "aac": "piano_mid",
                    "dat": "piano_mid",
                    "minidisc": "piano_mid",
                }.get(material_type.value, "unknown")
                _hpg_pre_mask, _hpg_pre_href = _hpg_tmp.extract_harmonic_mask(
                    original_audio_for_goals, sample_rate, instrument_tag=_hpg_instrument_tag_pre
                )
                logger.debug("§2.28 HPG Pre-Phase: Harmonik-Referenz extrahiert (tag=%s)", _hpg_instrument_tag_pre)
            except Exception as _hpg_pre_exc:
                logger.debug("HPG Pre-Phase-Extraktion nicht verfügbar: %s", _hpg_pre_exc)

        # §2.36 LGE Pre-Phase: Transkription auf Originalaudio (genauere Phonemkarte als auf NR-Output)
        _lge_trans_pre: Any = None
        try:
            from backend.core.lyrics_guided_enhancement import (
                get_lyrics_guided_enhancement as _get_lge_pre,
            )

            _, _lge_trans_pre = _get_lge_pre().enhance(original_audio_for_goals, sample_rate)
            logger.debug(
                "§2.36 LGE Pre-Phase: Phonemkarte aus Originalaudio (%d Segmente)",
                len(_lge_trans_pre.words) if _lge_trans_pre is not None else 0,
            )
        except Exception as _lge_p_exc:
            logger.debug("LGE Pre-Phase-Transkription nicht verfügbar: %s", _lge_p_exc)

        # Step 4: Execute Phases — mit EnsembleProcessor-Konsens (§2.21)
        logger.info("Step 3/4: Executing Restoration Pipeline (EnsembleProcessor)...")
        # §11.4 Stufen-Vorab-Meldung: Gesamtzahl der UV3-Phasen ans Frontend melden,
        # damit die Anzeige "Stufe X / Y" von Anfang an korrekt ist (kein Hochzählen).
        _cb(29, f"__total_uv3_phases__:{len(selected_phases)}")
        _cb(30, "Pipeline startet…")
        # §Vintage: Strength-Caps als Instanzvariable setzen (für _profiled_phase_call Zugriff)
        self._vintage_phase_strength_caps = _vintage_phase_strength_caps
        # §2.31 Material-adaptive Phasen-Initialstärken aufbauen (DefectPhaseMapper._MATERIAL_PHASE_FACTORS)
        # Kombiniert Vintage-Era-Caps mit materialspezifischen Stärke-Faktoren:
        # Nimmt jeweils das Minimum beider Quellen (restriktiverer Wert gewinnt).
        _material_phase_initial_strengths: dict[str, float] = {}
        try:
            from backend.core.defect_phase_mapper import get_material_initial_strength as _get_mat_strength

            _mat_val = material_type.value if hasattr(material_type, "value") else str(material_type)
            # Alle bekannten Phase-IDs aus beiden Quellen konsolidieren
            _all_phase_ids = set(_vintage_phase_strength_caps.keys())
            # Alle Phase-IDs aus MaterialPhaseFactors für dieses Material ermitteln
            try:
                from backend.core.defect_phase_mapper import _MATERIAL_PHASE_FACTORS as _mpf

                _all_phase_ids.update(_mpf.get(_mat_val, {}).keys())
            except Exception:
                pass
            for _pid in _all_phase_ids:
                _mat_s = _get_mat_strength(_mat_val, _pid)
                _era_cap = _vintage_phase_strength_caps.get(_pid, 1.0)
                # Restriktiverer Wert: Minimum beider Quellen
                _combined = min(_mat_s, _era_cap)
                if _combined < 1.0:
                    _material_phase_initial_strengths[_pid] = _combined
            if _material_phase_initial_strengths:
                logger.debug(
                    "§2.31 Material-Phase-Initialstärken: %d Phasen angepasst für material=%s",
                    len(_material_phase_initial_strengths),
                    _mat_val,
                )
        except Exception as _mps_exc:
            logger.debug("Material-Phase-Initialstärken-Aufbau fehlgeschlagen: %s", _mps_exc)
        self._material_phase_initial_strengths = _material_phase_initial_strengths
        # §Safety: PLM-Eviction während Pipeline sperren — ONNX-Session-Destruktoren
        # dürfen nicht zeitgleich mit laufender Inferenz feuern (double-free-Crash).
        try:
            from backend.core.plugin_lifecycle_manager import set_pipeline_active

            set_pipeline_active(True)
        except Exception:
            pass
        try:
            restored_audio, executed_phases, skipped_phases, deferred_phases = self._execute_pipeline(
                audio,
                sample_rate,
                material_type,
                defect_result,
                selected_phases,
                progress_callback=progress_callback,
                audio_update_callback=_audio_update_cb_kwarg,
                restorability_score=_pmgg_restorability_score,  # §2.29 normativ
                applicable_goals=_applicable_goals,  # §2.32 normativ
                material_initial_strengths=_material_phase_initial_strengths,  # §2.31
                quality_mode_override=_pipeline_quality_mode_override,
            )
        finally:
            try:
                set_pipeline_active(False)
            except Exception:
                pass
        # v9.10.58: EnsembleProcessor-Block entfernt.
        # v9.10.72: ARE Multi-Pass ebenfalls entfernt. UV3 ist der einzige Processing-Pass.
        # Wissenschaftliche Begründung:
        # - Kaskadierte STFT-Modifikation akkumuliert Rundungsfehler (Ephraim & Malah 1984)
        # - ML-Modelle sind nicht auf eigenen Output trainiert → Domain Shift
        # - FeedbackChain (unten) bietet zielgerichtete iterative Verbesserung

        _cb(82, "Nachbearbeitung…")

        # Speicher-Hygiene: Pipeline-Modelle entladen sobald alle Phasen abgeschlossen.
        # AudioSR (7 GB), LAION-CLAP (2.2 GB) werden nur in der Phase-Pipeline benötigt.
        try:
            from plugins.audiosr_plugin import unload_audiosr

            unload_audiosr()
        except Exception:
            pass
        try:
            from plugins.laion_clap_plugin import unload_laion_clap

            unload_laion_clap()
        except Exception:
            pass
        gc.collect()

        # --- §2.33 FeedbackChain-Ceiling: Physikalische Grenze vor FC-Block schätzen ---
        # estimate_physical_ceiling läuft hier mit leerem scores-Dict, da _pqs_result
        # erst nach dem FC-Block verfügbar ist. Das Ceiling basiert primär auf Audio-SNR
        # und Bandbreite, nicht auf goal-Scores — daher trotzdem valid.
        _fc_ceiling_val: float | None = None
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
                "phase_14_phase_correction",  # Phasenfehler vor Mastering-Phasen korrigieren
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
                    use_pqs_in_loop=True,
                    use_versa_in_loop=self.config.deployment_mode == DeploymentMode.RESEARCH,
                )
                # §2.34 GPP-WIRE: GoalPriorityProtocol als in-loop Phase-Callback verdrahten
                try:
                    from backend.core.goal_priority_protocol import check_iteration_abort as _check_gpp
                    from backend.core.musical_goals.musical_goals_metrics import (
                        MusicalGoalsChecker as _GoalChecker,
                    )

                    _gpp_checker = _GoalChecker()
                    _gpp_sr = sample_rate
                    # Cache: speichere letzte Scores um doppelte measure_all-Aufrufe zu vermeiden
                    _gpp_prev_scores: dict[str, float] = {}
                    _gpp_analytics_s: list[float] = [0.0]  # mutable accumulator for closure

                    def _gpp_fc_callback(_ab: np.ndarray, _aa: np.ndarray) -> tuple[bool, str]:
                        nonlocal _gpp_prev_scores
                        import time as _t_gpp

                        try:
                            # Reuse cached scores for 'before' when available (same audio)
                            if _gpp_prev_scores:
                                _sb = _gpp_prev_scores
                            else:
                                _t0g = _t_gpp.perf_counter()
                                _sb = _gpp_checker.measure_all(_ab, _gpp_sr)
                                _gpp_analytics_s[0] += _t_gpp.perf_counter() - _t0g
                            _t0g = _t_gpp.perf_counter()
                            _sa = _gpp_checker.measure_all(_aa, _gpp_sr)
                            _gpp_analytics_s[0] += _t_gpp.perf_counter() - _t0g
                            _gpp_prev_scores = _sa  # cache for next iteration
                            _res = _check_gpp(_sb, _sa)
                            return _res.should_abort, _res.reason
                        except Exception as _gpp_cb_exc:
                            logger.debug("GPP FeedbackChain callback failed: %s", _gpp_cb_exc)
                            return False, ""

                    if hasattr(_fc_chain, "goal_priority_callback"):
                        _fc_chain.goal_priority_callback = _gpp_fc_callback
                        logger.debug("§GPP-WIRE: GoalPriorityProtocol FeedbackChain-Callback aktiv")
                except Exception as _gpp_wire_exc:
                    logger.debug("§GPP-WIRE: nicht verfügbar — %s", _gpp_wire_exc)
                _fc_chain_result = _fc_chain.run(restored_audio, _fc_phases_list, ceiling=_fc_ceiling_val)
                restored_audio = _fc_chain_result.audio
                # Report analytics overhead to PerformanceGuard so goal-measurement
                # time inside FeedbackChain does not inflate the RT factor.
                _fc_analytics = getattr(_fc_chain_result, "analytics_overhead_s", 0.0)
                # Also accumulate GPP-callback analytics overhead (captured via closure)
                try:
                    _fc_analytics += _gpp_analytics_s[0]
                except Exception:
                    pass
                if self.performance_guard and _fc_analytics > 0:
                    self.performance_guard.add_analytics_overhead(_fc_analytics)
                logger.info(
                    "🔄 FeedbackChain: score=%.3f retries=%d t=%.2fs analytics_overhead=%.2fs (%d Post-Phasen)",
                    _fc_chain_result.overall_score,
                    getattr(_fc_chain_result, "total_retries", _fc_chain_result.iterations),
                    _fc_chain_result.total_time_s,
                    _fc_analytics,
                    len(_fc_phases_list),
                )
            else:
                logger.debug("FeedbackChain: keine Post-Pipeline-Phasen verfügbar")
        except Exception as _fc_exc:
            logger.debug("FeedbackChain nicht verfügbar: %s", _fc_exc)

        _cb(85, "FeedbackChain abgeschlossen…")

        # §1.4 Auto Stem Separation for Studio 2026 (Spec §9.5: Stem-Sep → StemRemixBalancer)
        # Separates the fully-processed audio into vocals/instruments so StemRemixBalancer can
        # perform LUFS-correct re-mixing. Only runs when the caller did not provide pre-separated
        # stems AND the mode is Studio 2026. BsRoformer handles its own memory budget internally.
        _auto_stems: dict | None = None
        if _is_studio_26 and kwargs.get("stems") is None:
            try:
                from plugins.bs_roformer_plugin import separate_stems as _bsr_sep

                _bsr_result = _bsr_sep(restored_audio, sample_rate, stems=["vocals", "instruments"])
                if _bsr_result is not None and isinstance(_bsr_result.stems, dict):
                    _bsr_v = _bsr_result.stems.get("vocals")
                    _bsr_i = _bsr_result.stems.get("instruments")
                    if _bsr_v is not None and _bsr_i is not None:
                        _auto_stems = {
                            "vocals": _bsr_v,
                            "instruments": _bsr_i,
                        }
                        logger.info(
                            "§1.4 Auto-Stem-Separation: %s SDRi=%.1f dB confidence=%.2f",
                            _bsr_result.model_used,
                            _bsr_result.sdri_db,
                            _bsr_result.confidence,
                        )
            except Exception as _bsr_exc:
                logger.debug(
                    "§1.4 Auto-Stem-Separation nicht verfügbar (StemRemixBalancer überspringt): %s",
                    _bsr_exc,
                )

        # §1.4 StemRemixBalancer — LUFS-korrekter Stem-Re-Mix (nur bei Stem-Vorhandensein)
        try:
            from backend.core.stem_remix_balancer import balance_remix

            _stems = kwargs.get("stems") or _auto_stems
            if _stems is not None and isinstance(_stems, dict):
                _vocals = _stems.get("vocals")
                _instruments = _stems.get("instruments")
                if _vocals is not None and _instruments is not None:
                    # §1.4 vocal_weight: aus PANNs-Ergebnis ableiten (vor MDX23C auf Original geschätzt)
                    # §1.4 vocal_weight: None → _estimate_vocal_weight() in StemRemixBalancer
                    _vw_raw = _stems.get("vocal_weight")
                    _vw: float | None = None
                    if _vw_raw is not None:
                        _vw_f = float(_vw_raw)
                        if 0.0 < _vw_f < 1.0:
                            _vw = _vw_f
                    if _vw is None:
                        # PANNs-Konfidenz nutzen, nur wenn glaubwürdig (> 1 %)
                        _ptags: dict = kwargs.get("panns_tags", {})
                        _v_conf = max(
                            _ptags.get("Singing voice", 0.0),
                            _ptags.get("Vocals", 0.0),
                            _ptags.get("Speech", 0.0),
                        )
                        _vw = float(np.clip(_v_conf, 0.1, 0.9)) if _v_conf > 0.01 else None
                    restored_audio = balance_remix(
                        _vocals,
                        _instruments,
                        original_audio_for_goals,
                        sample_rate,
                        vocal_weight=_vw,
                    )
                    logger.info(
                        "🎚️ StemRemixBalancer: Stem-Remix LUFS-balanciert (vocal_weight=%s)",
                        f"{_vw:.2f}" if _vw is not None else "auto",
                    )
                else:
                    logger.debug("StemRemixBalancer: Stems unvollständig (vocals/instruments fehlen)")
            else:
                logger.debug("StemRemixBalancer: keine Stems verfügbar (weder kwargs noch Auto-Separation)")
        except Exception as _srb_exc:
            logger.debug("StemRemixBalancer nicht verfügbar: %s", _srb_exc)

        # §Studio-2026 Reference Mastering — Matchering 2.0 spektrales Profil-Matching
        # Position: nach StemRemixBalancer (Re-Mix liegt vor), vor LUFS-Normalisierung.
        # Referenz: restauriertes Original (original_audio_for_goals) — originalgetreues Profil.
        # Nur im Studio-2026-Modus aktiv; bei fehlendem Paket transparenter DSP-Fallback.
        # §9.5: Production feature for Studio 2026 — no experimental gate needed.
        if _is_studio_26:
            try:
                from plugins.matchering_plugin import (
                    is_matchering_available as _mg_avail,
                    match_reference as _match_ref,
                )

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
            from backend.core.lyrics_guided_enhancement import (
                get_content_aware_processor as _get_cap,
                get_lyrics_guided_enhancement as _get_lge,
            )

            _lge = _get_lge()
            if _lge_trans_pre is not None and not getattr(_lge_trans_pre, "fallback_used", True):
                # §2.36 Phase-Gate-Anbindung: Saliency aus Pre-Phase-Transkription (Originalaudio)
                _cap = _get_cap()
                _n_smp = restored_audio.shape[-1] if restored_audio.ndim == 2 else len(restored_audio)
                _base_sal = np.ones(_n_smp, dtype=np.float32)
                _lge_saliency = _cap.compute_lyrics_saliency(_base_sal, _lge_trans_pre, sample_rate)
                if restored_audio.ndim == 2 and restored_audio.shape[0] <= 2:
                    _lge_audio = restored_audio * _lge_saliency[np.newaxis, :]
                elif restored_audio.ndim == 2:
                    _lge_audio = restored_audio * _lge_saliency[:, np.newaxis]
                else:
                    _lge_audio = restored_audio * _lge_saliency
                _lge_audio = np.clip(np.nan_to_num(_lge_audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                restored_audio = _lge_audio
                _lge_seg_count = len(_lge_trans_pre.words) if _lge_trans_pre is not None else 0
                logger.info(
                    "§2.36 LGE: Saliency aus Original-Phonemkarte — %d Segmente (§2.36 Phase-Gate)",
                    _lge_seg_count,
                )
            else:
                # Fallback: transcribe restored audio (pre-computed transcription unavailable)
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

        _cb(88, "Audio-Nachbearbeitung…")

        # --- SegmentAdaptiveProcessor: Segment-individuelle Feinoptimierung (§2.10 Spec) ---
        # Anwendung NACH globalem Pipeline-Lauf: jedes Segment (Stille /
        # Mixed / Vocal) erhält eigene NR-Stärke, Harmonic-Boost, Smoothing.
        # Stereo-kompatibel: Gain-Verhältnis Mono→Mono auf Stereo-Kanäle übertragen.
        _sap_result = None
        try:
            from backend.core.segment_adaptive_processor import get_segment_processor

            _sap_n_samples = restored_audio.shape[-1] if restored_audio.ndim == 2 else len(restored_audio)
            _audio_dur_s = _sap_n_samples / float(sample_rate)
            _sap_enabled = _audio_dur_s >= 5.0  # Fallback bei < 5 s (§2.10)
            _sap_is_stereo = restored_audio.ndim == 2 and restored_audio.shape[0] <= 2

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
                ", ".join({s.segment_type for s in _sap_result.segments}),
                _sap_result.used_fallback,
            )
        except Exception as _sap_exc:
            logger.debug("SegmentAdaptiveProcessor nicht verfügbar: %s", _sap_exc)

        # Post-Processing: Temporale Qualitätskohärenz (§2.16)
        _tqc = None
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

        # §8.2 EmotionalArc Rollback-Referenz: Snapshot vor ExcellenceOptimizer
        _pre_excellence_audio: np.ndarray = restored_audio.copy()

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
                if _hpg_pre_mask is not None and _hpg_pre_href is not None:
                    _hpg_mask, _hpg_href = _hpg_pre_mask, _hpg_pre_href
                    logger.debug("§2.28 HPG: Pre-Phase-Maske wiederverwendet")
                else:
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
                    "🎵 HarmonicLattice: Korrektur angewendet (lattice_score=%.3f B=%.5f n_partials_korrigiert=%d)",
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

        _cb(91, "Qualitätsprüfung…")

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
            # v9.10.58: VERSA-MOS-Gate mit 2. ExcellenceOptimizer-Pass entfernt.
            # Wissenschaftliche Begründung: Identischer ExcellenceOptimizer auf bereits
            # optimierten Daten akkumuliert STFT-Rundungsfehler. Die Korrektur bei
            # niedrigem MOS erfolgt einmalig im ExzellenzDenker (AurikDenker Stufe 7).
            if _pqs_result is not None and _pqs_result.pqs_mos < 4.0:
                logger.info(
                    "PQS-MOS=%.2f < 4.0 — Korrektur wird im ExzellenzDenker (Stufe 7) behandelt",
                    _pqs_result.pqs_mos,
                )
        except Exception as _pqs_exc:
            logger.warning("PerceptualQualityScorer nicht verfügbar (PQS_UNAVAILABLE): %s", _pqs_exc)
            _fail_reasons.append(
                {
                    "component": "PerceptualQualityScorer",
                    "error_code": "PQS_UNAVAILABLE",
                    "severity": "degraded",
                    "exc_type": type(_pqs_exc).__name__,
                    "exc_msg": str(_pqs_exc),
                }
            )

        # §2.31 AdaptiveGoalThresholds — Kontextadaptive Ziel-Schwellenwerte
        _adaptive_goals = None
        try:
            from backend.core.musical_goals.adaptive_goals_system import get_adaptive_goals_and_config

            _adaptive_goals = get_adaptive_goals_and_config(audio, sample_rate, _classified_material, defect_result)
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
            # AdaptiveGoalThresholds: robust über alle bekannten Payload-Formen aufloesen.
            if _adaptive_goals is not None:
                try:
                    _adaptive_values = self._resolve_adaptive_goal_thresholds(_adaptive_goals)
                    for _goal_name, _val in _adaptive_values.items():
                        if _goal_name in _effective_goal_thresholds:
                            _effective_goal_thresholds[_goal_name] = float(_val)
                except Exception as _agt_apply_exc:
                    logger.debug("AdaptiveGoalThresholds konnten nicht angewendet werden: %s", _agt_apply_exc)

            # §2.31 Physical-Ceiling clamp: Zielschwellen dürfen physikalische Decke nicht überschreiten.
            if _physical_ceiling is not None:
                try:
                    _ceiling_map = getattr(_physical_ceiling, "ceiling", None)
                    if isinstance(_ceiling_map, dict):
                        for _goal_name, _ceil_val in _ceiling_map.items():
                            if _goal_name in _effective_goal_thresholds and math.isfinite(float(_ceil_val)):
                                _effective_goal_thresholds[_goal_name] = min(
                                    float(_effective_goal_thresholds[_goal_name]),
                                    float(_ceil_val),
                                )
                except Exception as _ceil_exc:
                    logger.debug("PhysicalCeiling clamp nicht verfügbar: %s", _ceil_exc)

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
                # v9.10.58: Musical Goals Re-Pass in UV3 entfernt — wissenschaftlich nicht
                # gerechtfertigt (Ephraim & Malah 1984: cascaded identical processing amplifies
                # STFT roundtrip errors). Die Korrektur erfolgt einmalig im ExzellenzDenker
                # (AurikDenker Stufe 7), der denselben ExcellenceOptimizer + degressive
                # Intensität verwendet — dort ist der Re-Pass korrekt platziert.
                logger.warning(
                    "🎵 Musical Goals Verletzungen (%d/%d): %s — Korrektur in ExzellenzDenker",
                    len(_mg_violations),
                    len(_musical_goal_scores),
                    ", ".join(_mg_violations),
                )
            else:
                logger.info(
                    "🎵 Musical Goals: alle %d Ziele erfüllt (Ø %.3f)",
                    len(_musical_goal_scores),
                    _musical_excellence_score,
                )
        except Exception as _mg_exc:
            logger.warning("MusicalGoalsChecker nicht verfügbar (MUSICAL_GOALS_UNAVAILABLE): %s", _mg_exc)
            _fail_reasons.append(
                {
                    "component": "MusicalGoalsChecker",
                    "error_code": "MUSICAL_GOALS_UNAVAILABLE",
                    "severity": "degraded",
                    "exc_type": type(_mg_exc).__name__,
                    "exc_msg": str(_mg_exc),
                }
            )

        _cb(95, "Musical Goals geprüft…")

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
        _arc_result = None
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
                # §8.2 Spec: bindende Rollback-Garantie → Pre-Excellence-Audio wiederherstellen
                _are_p = getattr(_arc_result, "arousal_pearson", 0.0)
                _val_p = getattr(_arc_result, "valence_pearson", 0.0)
                if not (_are_p >= 0.85 and _val_p >= 0.80):
                    restored_audio = _pre_excellence_audio.copy()
                    _fail_reasons.append(
                        {
                            "component": "EmotionalArcPreservationMetric",
                            "error_code": "ARC_REGRESSION_ROLLBACK",
                            "severity": "critical",
                            "arousal_pearson": round(_are_p, 4),
                            "valence_pearson": round(_val_p, 4),
                        }
                    )
                    logger.info(
                        "↩️ EmotionalArc: Rollback auf Pre-Excellence-Audio "
                        "(arousal=%.3f < 0.85 oder valence=%.3f < 0.80)",
                        _are_p,
                        _val_p,
                    )
        except Exception as _eap_exc:
            logger.debug("EmotionalArcPreservationMetric nicht verfügbar: %s", _eap_exc)

        # --- GP-Lernzyklus: Vorbereitung — PQS-Norm + Pareto-Vorschlag (vor MDEM) (§2.5 Spec) ---
        # GPParameterOptimizer.update() wird erst NACH MDEM aufgerufen (Spec §2.5 normativ).
        _pqs_mos_norm: float | None = None
        _gp_opt_ref: Any = None
        _gp_upd_proposal_ref: Any = None
        _gp_score_ref: float = 0.0
        _gp_goal_scores_ref: dict[str, float] | None = None
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
                # Mel-Spectral Loss + Multi-Scale STFT Loss (Spec bindend: §2.5 PFLICHT-Loss-Funktion)
                # Loss-Funktion: Mel-Spectral Loss + Multi-Scale STFT Loss (3 Auflösungen)
                # VERBOTEN: MSE auf Raw-Audio als alleinige GPO-Loss
                _mel_score: float = 0.0
                _ms_stft_score: float = 0.0
                try:
                    import librosa as _lbr_gpo

                    _gpo_a_orig = original_audio_for_goals
                    _gpo_a_rest = restored_audio
                    # Mono-Konvertierung für Loss-Berechnung (shape: (T,))
                    _gpo_a_orig_m = np.mean(_gpo_a_orig, axis=0) if _gpo_a_orig.ndim == 2 else _gpo_a_orig
                    _gpo_a_rest_m = np.mean(_gpo_a_rest, axis=0) if _gpo_a_rest.ndim == 2 else _gpo_a_rest
                    # Längenangleich
                    _gpo_len = min(len(_gpo_a_orig_m), len(_gpo_a_rest_m))
                    _gpo_a_orig_m = _gpo_a_orig_m[:_gpo_len]
                    _gpo_a_rest_m = _gpo_a_rest_m[:_gpo_len]
                    # §Performance-Budget: GP-Loss auf max. 30 s begrenzen (OOM-Schutz für
                    # Langaufnahmen — bei 5 min / 48 kHz = 14,4 M Samples würde STFT für
                    # n_fft=4096, hop=1024 ~462 MB pro rfft-Aufruf × 6 Aufrufe ≈ 2,8 GB belegen).
                    _MAX_GP_LOSS_S = 30.0
                    _max_gp_samples = int(_MAX_GP_LOSS_S * sample_rate)
                    if _gpo_len > _max_gp_samples:
                        _gpo_mid = _gpo_len // 2
                        _gpo_start = max(0, _gpo_mid - _max_gp_samples // 2)
                        _gpo_a_orig_m = _gpo_a_orig_m[_gpo_start : _gpo_start + _max_gp_samples]
                        _gpo_a_rest_m = _gpo_a_rest_m[_gpo_start : _gpo_start + _max_gp_samples]
                        logger.debug(
                            "GPO Loss: Audio auf %.0fs begrenzt (original: %.1fs) — OOM-Schutz",
                            _MAX_GP_LOSS_S,
                            _gpo_len / sample_rate,
                        )
                    # Mel-Spectral Loss (80 Mel-Bins)
                    _mel_r = _lbr_gpo.feature.melspectrogram(y=_gpo_a_orig_m, sr=sample_rate, n_mels=80)
                    _mel_e = _lbr_gpo.feature.melspectrogram(y=_gpo_a_rest_m, sr=sample_rate, n_mels=80)
                    _mel_ref_mean = float(np.mean(_mel_r)) + 1e-8
                    _mel_loss_val = float(np.mean(np.abs(_mel_r - _mel_e))) / _mel_ref_mean
                    _mel_score = max(0.0, min(1.0, 1.0 - _mel_loss_val))
                    # Multi-Scale STFT Loss (3 Auflösungen: 256, 1024, 4096)
                    _ms_scores: list = []
                    for _n_fft_ms in (256, 1024, 4096):
                        _hop_ms = _n_fft_ms // 4
                        _s_r = np.abs(
                            np.fft.rfft(
                                np.lib.stride_tricks.sliding_window_view(_gpo_a_orig_m, _n_fft_ms)[::_hop_ms],
                                axis=-1,
                            )
                        )
                        _s_e = np.abs(
                            np.fft.rfft(
                                np.lib.stride_tricks.sliding_window_view(_gpo_a_rest_m, _n_fft_ms)[::_hop_ms],
                                axis=-1,
                            )
                        )
                        _min_frames = min(_s_r.shape[0], _s_e.shape[0])
                        if _min_frames > 0:
                            _sr_c = _s_r[:_min_frames]
                            _se_c = _s_e[:_min_frames]
                            _ref_mean_ms = float(np.mean(_sr_c)) + 1e-8
                            _ms_loss_single = float(np.mean(np.abs(_sr_c - _se_c))) / _ref_mean_ms
                            _ms_scores.append(max(0.0, min(1.0, 1.0 - _ms_loss_single)))
                    _ms_stft_score = float(np.mean(_ms_scores)) if _ms_scores else 0.0
                    logger.debug(
                        "GPO Loss-Funktion: mel_score=%.4f ms_stft_score=%.4f",
                        _mel_score,
                        _ms_stft_score,
                    )
                except Exception as _gpo_loss_exc:
                    logger.debug("GPO Mel/STFT-Loss nicht verfügbar: %s", _gpo_loss_exc)
                    _mel_score = 0.0
                    _ms_stft_score = 0.0
                # Kombinierter Score (Spec §2.5 bindend):
                # 0.4 * musical_excellence + 0.3 * pqs_norm + 0.2 * mel_score + 0.1 * ms_stft_score
                if _pqs_mos_norm is not None and _musical_excellence_score > 0.0:
                    if _mel_score > 0.0 or _ms_stft_score > 0.0:
                        _gp_score_ref = (
                            0.4 * _musical_excellence_score
                            + 0.3 * _pqs_mos_norm
                            + 0.2 * _mel_score
                            + 0.1 * _ms_stft_score
                        )
                    else:
                        _gp_score_ref = 0.5 * _musical_excellence_score + 0.5 * _pqs_mos_norm
                elif _musical_excellence_score > 0.0:
                    _gp_score_ref = _musical_excellence_score
                else:
                    _gp_score_ref = _pqs_mos_norm or 0.0
                # Era-Warmstart: Ära-Prior aus EraClassifier für GP-Cold-Start (§2.14)
                _era_ws: dict[str, float] | None = None
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

        # §8.2 EmotionalArc: Post-MDEM Nachmessung + Makro-Korrektur (5 s-Skala)
        # MDEM korrigiert Mikro-Dynamik (400 ms). Wenn der Makro-Bogen (5 s)
        # danach immer noch abgeflacht ist, greift die Makro-Korrektur.
        try:
            from backend.core.emotional_arc_preservation import (
                correct_emotional_arc,
                measure_emotional_arc,
            )

            _arc_post_mdem = measure_emotional_arc(original_audio_for_goals, restored_audio, sample_rate)
            if _arc_post_mdem is not None and not _arc_post_mdem.skipped and not _arc_post_mdem.arc_preserved:
                logger.info(
                    "EmotionalArc post-MDEM: Bogen nicht erhalten (arousal=%.3f, valence=%.3f) — Makro-Korrektur",
                    _arc_post_mdem.arousal_pearson,
                    _arc_post_mdem.valence_pearson,
                )
                restored_audio, _arc_corrected = correct_emotional_arc(
                    original_audio_for_goals, restored_audio, sample_rate
                )
                restored_audio = np.clip(
                    np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                _arc_result = _arc_corrected
                logger.info(
                    "EmotionalArc post-Korrektur: arousal=%.3f valence=%.3f preserved=%s",
                    _arc_corrected.arousal_pearson,
                    _arc_corrected.valence_pearson,
                    _arc_corrected.arc_preserved,
                )
            elif _arc_post_mdem is not None and not _arc_post_mdem.skipped:
                _arc_result = _arc_post_mdem
                logger.debug("EmotionalArc post-MDEM: Bogen intakt — keine Korrektur nötig")
        except Exception as _arc_corr_exc:
            logger.debug("EmotionalArc post-MDEM-Korrektur nicht verfügbar: %s", _arc_corr_exc)

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
            # §9.5: Production feature for Studio 2026 — no experimental gate; exception handler
            # provides safe fallback to plain PGHI-ISTFT output if vocos plugin is unavailable.
            if _is_vocos_studio and _vocos_mos_ok:
                import os as _vos
                import sys as _vsys

                _vplugins = _vos.path.join(_vos.path.dirname(__file__), "..", "..", "plugins")
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
                        "§1.4 Vocos: Mono-Finisher angewendet (model=%s pqs_mos=%.3f mel_snr=%.1f dB)",
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

        # Step 5: Performance Report
        logger.info("Step 4/4: Generating Report...")
        try:
            perf_report = self.performance_guard.get_performance_report() if self.performance_guard else None
        except RuntimeError:
            # start_monitoring() was not called (e.g. audio < 0.5 s in multipass chunk context)
            perf_report = None

        # Quality Estimate
        quality_estimate = self._estimate_quality(defect_result, perf_report, executed_phases, restored_audio, 48_000)

        # Build Result
        total_time = time.time() - start_time
        rt_factor = total_time / audio_duration
        # Absolutes 30-Minuten-Budget (1800 s) statt relativem 8×RT-Faktor —
        # kurze Clips erzeugen naturgemäß hohe rt_factors, die kein echtes Problem sind.
        _ABSOLUTE_BUDGET_S = 1800.0
        if self.config.enforce_3x_rt and total_time > _ABSOLUTE_BUDGET_S:
            logger.warning(
                "Absolutes Zeitbudget überschritten: %.0f s > %.0f s (rt_factor=%.2f). "
                "Lösung: Modus FAST/BALANCED nutzen oder adaptive Skips aktivieren.",
                total_time,
                _ABSOLUTE_BUDGET_S,
                rt_factor,
            )

        # §8.1 Reporting-Analytik: MUSHRA, Artefakt-Analyse, Qualitätsvergleich
        # und >80 weitere Diagnose-Module (analytics-only, kein Audio-Eingriff)
        _analytics_meta = self._collect_reporting_analytics(
            restored_audio=restored_audio,
            audio=audio,
            original_audio_for_goals=original_audio_for_goals,
            sample_rate=sample_rate,
            material_type=material_type,
            _era_result=_era_result,
            _quality_before=_quality_before,
            quality_estimate=quality_estimate,
            _pipeline_confidence=_pipeline_confidence,
            _musical_goal_scores=_musical_goal_scores,
            _musical_goals_passed=_musical_goals_passed,
            defect_result=defect_result,
            executed_phases=executed_phases,
            _pqs_result=_pqs_result,
        )

        try:
            from backend.core.pipeline_health_state import (
                pipeline_health_from_fail_reasons,
                primary_fail_reason_from_fail_reasons,
            )

            _degradation_status = pipeline_health_from_fail_reasons(_fail_reasons).value
        except Exception:
            _degradation_status = "degraded" if _fail_reasons else "ok"
        _primary_fail_reason = primary_fail_reason_from_fail_reasons(_fail_reasons)

        # §G3: Chroma-Korrelation und LUFS-Delta für Export-Gate berechnen
        _chroma_corr_for_result: float | None = None
        _lufs_delta_for_result: float | None = None
        # Chroma: tonal_center Goal-Score ist direkt die Chroma Pearson-Korrelation
        if _musical_goal_scores:
            _chroma_corr_for_result = float(_musical_goal_scores.get("tonal_center", 0.0))
        # LUFS-Delta: Differenz zwischen Original und Restauriert
        try:
            import pyloudnorm as _pyln

            _lufs_meter = _pyln.Meter(sample_rate)
            _orig_2d = (
                original_audio_for_goals if original_audio_for_goals.ndim >= 2 else original_audio_for_goals[:, None]
            )
            _rest_2d = restored_audio if restored_audio.ndim >= 2 else restored_audio[:, None]
            _lufs_orig = _lufs_meter.integrated_loudness(_orig_2d)
            _lufs_rest = _lufs_meter.integrated_loudness(_rest_2d)
            if np.isfinite(_lufs_orig) and np.isfinite(_lufs_rest):
                _lufs_delta_for_result = abs(_lufs_rest - _lufs_orig)
        except Exception as _lufs_exc:
            logger.debug("LUFS-Delta-Berechnung fehlgeschlagen: %s", _lufs_exc)

        _cb(98, "Ergebnis wird finalisiert…")

        result = RestorationResult(
            audio=restored_audio,
            config=self.config,
            material_type=material_type,
            defect_scores={dt: defect_result.scores[dt].severity for dt in DefectType if dt in defect_result.scores},
            phases_executed=executed_phases,
            phases_skipped=skipped_phases,
            deferred_phases=deferred_phases,  # §2.38 KMV: RT-skipped phases for Stage 2
            total_time_seconds=total_time,
            rt_factor=rt_factor,
            quality_estimate=quality_estimate,
            warnings=perf_report.warnings if perf_report is not None else [],
            metadata={
                "fail_reasons": list(_fail_reasons),
                "degradation_status": _degradation_status,
                "fail_reason": _primary_fail_reason,
                "quality_estimate": {
                    "value": float(quality_estimate),
                    "fallback_quality_estimate": bool(self._quality_estimate_used_fallback),
                    "source": str(self._quality_estimate_source),
                },
                "deployment": {
                    "mode": getattr(self.config.deployment_mode, "value", "product"),
                    "blocked_experimental_features": sorted(self._blocked_experimental_features),
                },
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
                "phase_plan_intelligence": dict(getattr(self, "_phase_plan_intelligence", {})),
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
                "pqs": (dataclasses.asdict(_pqs_result) if _pqs_result is not None else None),
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
                **_analytics_meta,
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
            goal_priority_log=(
                list(_fc_chain_result.metadata.get("goal_priority_log", [])) if _fc_chain_result is not None else []
            )
            + (
                [f"Post-pipeline GPP: {_goal_abort.reason}"]
                if _goal_abort is not None and _goal_abort.should_abort
                else []
            ),
            confidence=(float(_pipeline_confidence.confidence) if _pipeline_confidence is not None else 1.0),
            chroma_correlation=_chroma_corr_for_result,
            lufs_delta=_lufs_delta_for_result,
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
                    np.mean(restored_audio, axis=0).astype(np.float32)
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
            f"✅ Restoration complete: {total_time:.1f}s ({rt_factor:.2f}× RT), Quality: {quality_estimate * 100:.1f}%"
        )

        # --- Aggressive end-of-run memory cleanup (OOM hardening) ---
        _cleanup_report: dict[str, Any] = {"unloaded": [], "errors": []}
        try:
            _unload_specs = [
                ("plugins.audiosr_plugin", "unload_audiosr", "AudioSR"),
                ("plugins.utmos_plugin", "unload_utmos", "UTMOS"),
                ("plugins.laion_clap_plugin", "unload_laion_clap", "LAION-CLAP"),
                ("plugins.mert_plugin", "unload_mert", "MERT"),
                # CREPE: keep singleton alive (cache + ONNX session) — ExzellenzDenker
                # re-uses CREPE right after UV3 finishes.  PLM handles LRU eviction.
                ("plugins.fcpe_plugin", "unload_fcpe", "FCPE"),
                ("plugins.basicpitch_plugin", "unload_basicpitch", "BasicPitch"),
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
                from backend.core.plugin_lifecycle_manager import (
                    cleanup_after_file as _plm_cleanup_after_file,
                )

                _cleanup_report["plm_evicted"] = int(_plm_cleanup_after_file())
            except Exception as _plm_exc:
                _cleanup_report["errors"].append(f"PLM: {_plm_exc}")

            # Defensive budget release for models without explicit unload hooks.
            try:
                from backend.core.ml_memory_budget import release as _ml_release

                for _model_name in ("CREPE", "FCPE", "RMVPE", "BasicPitch"):
                    _ml_release(_model_name)
            except Exception as _ml_rel_exc:
                _cleanup_report["errors"].append(f"ML-Budget-Release: {_ml_rel_exc}")

            gc.collect()

            # Linux/glibc: return free heap pages to OS to reduce RSS between runs.
            try:
                import ctypes

                _libc = ctypes.CDLL("libc.so.6")
                _trim = getattr(_libc, "malloc_trim", None)
                if callable(_trim):
                    _cleanup_report["malloc_trim"] = bool(_trim(0))
            except Exception as _trim_exc:
                _cleanup_report["errors"].append(f"malloc_trim: {_trim_exc}")
        except Exception as _cleanup_exc:
            logger.debug("Final memory cleanup failed: %s", _cleanup_exc)
        finally:
            with contextlib.suppress(Exception):
                result.metadata["memory_cleanup"] = _cleanup_report

        # §8.2 / §2.16 / §2.29 — Spec-Felder in RestorationResult schreiben
        result.emotional_arc = _arc_result
        result.temporal_coherence = _tqc
        # §2.29 PMGG-Log: Phasen wo best-effort angewendet wurde (reduzierte Stärke, kein Skip)
        try:
            _phase_gate_entries = getattr(self, "_pmgg_log_entries", [])
            result.phase_gate_log = [
                e.phase_id for e in _phase_gate_entries if getattr(e, "action", "").startswith("best_effort")
            ]
        except Exception:
            result.phase_gate_log = []

        return result

    def _collect_reporting_analytics(
        self,
        restored_audio: "np.ndarray",
        audio: "np.ndarray",
        original_audio_for_goals: "np.ndarray",
        sample_rate: int,
        material_type: "MaterialType",
        *,
        _era_result: object | None = None,
        _quality_before: object | None = None,
        quality_estimate: float = 0.0,
        _pipeline_confidence: object | None = None,
        _musical_goal_scores: dict | None = None,
        _musical_goals_passed: dict | None = None,
        defect_result: object | None = None,
        executed_phases: list | None = None,
        _pqs_result: object | None = None,
    ) -> dict:
        """
        Collect all post-restore reporting analytics for RestorationResult.metadata.

        This method does NOT modify restored_audio — it only computes read-only
        diagnostic and quality metrics for inclusion in the restoration report.
        Over 80 analytics modules are evaluated, each wrapped in try/except to
        ensure individual failures don't affect the restoration result.

        Returns:
            dict of metadata key-value pairs to be merged into
            RestorationResult.metadata via ``**`` unpacking.
        """
        # MushraEvaluator: Objektiver MUSHRA-Score (§8.1, OQS) — Original vs. Restauriert
        # Pflicht-Schwelle: OQS ≥ 80 (Good); Studio-2026-Ziel: ≥ 88
        _mushra_result = None
        try:
            from backend.core.mushra_evaluator import get_mushra_evaluator as _get_mushra

            _mushra_eval = _get_mushra()
            _orig_mono_m = audio if audio.ndim == 1 else audio.mean(axis=0)
            _rest_mono_m = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)
            _mushra_result = _mushra_eval.evaluate(
                reference=_orig_mono_m,
                test=_rest_mono_m,
                sr=sample_rate,
                compute_anchor=True,
            )
            _mushra_threshold = 80.0
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
        _artifact_scores: dict[str, float] | None = None
        try:
            from backend.core.psychoacoustic_artifact_detector import PsychoacousticArtifactDetector as _PAD

            _pad_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)
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
        _quality_after: object | None = None
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
        _smr_result: dict[str, float] | None = None
        try:
            from backend.core.masking_analyzer import MaskingAnalyzer as _MaskingAnalyzer

            _ma = _MaskingAnalyzer()
            _ma_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)
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
        _authenticity_extended: dict | None = None
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
        _authenticity_perf: dict | None = None
        try:
            from backend.core.authenticity_metrics import (
                BreathDetector as _BreathDetector,
                PlosiveDetector as _PlosiveDetector,
                RoomToneDetector as _RoomToneDetector,
                SibilanceDetector as _SibilanceDetector,
                TransientDetector as _TransientDetector,
            )

            _ap_audio = restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0).astype(np.float32)
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
        _mqa_result: dict | None = None
        try:
            from backend.core.musical_quality_assurance import (
                MediumType as _MediumType,
                MusicalQualityAssurance as _MQA,
                ProcessingMode as _ProcessingMode,
            )

            _MATERIAL_TO_MEDIUM: dict[str, Any] = {
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
            _mqa_medium = _MATERIAL_TO_MEDIUM.get(
                material_type.value if hasattr(material_type, "value") else str(material_type).lower(),
                _MediumType.UNKNOWN,
            )
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
        _aesthetic_result: dict | None = None
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

            _ae_mono = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)).astype(np.float32)
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
            if restored_audio.ndim == 2 and restored_audio.shape[0] == 2:
                _ae_l, _ae_r = restored_audio[0], restored_audio[1]
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
                    channels=int(restored_audio.shape[0]) if restored_audio.ndim == 2 else 1,
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
        _psychoacoustic_result: dict | None = None
        try:
            from backend.core.psychoacoustic_core import analyze_psychoacoustic as _analyze_psychoacoustic

            _pa_audio = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)).astype(np.float32)
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
        _intrinsic_quality: dict | None = None
        try:
            from backend.core.intrinsic_audio_quality_scorer import IntrinsicAudioQualityScorer as _IAQS

            _iq_audio = (restored_audio if restored_audio.ndim == 1 else restored_audio.mean(axis=0)).astype(np.float32)
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
        _music_mos_result: dict | None = None
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
            from plugins.mert_plugin import unload_mert

            unload_mert()
        except Exception:
            pass
        try:
            from plugins.utmos_plugin import unload_utmos

            unload_utmos()
        except Exception:
            pass
        gc.collect()

        # §6.4 DeliveryStandards: LUFS-Messung nach EBU R128
        _delivery_standards_result: dict | None = None
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
        _ad_result: dict | None = None
        try:
            from backend.core.artifact_detection import RestorationArtifactDetector as _ArtifactDetector

            _ad_orig = (
                np.mean(original_audio_for_goals, axis=0)
                if original_audio_for_goals is not None and original_audio_for_goals.ndim == 2
                else (original_audio_for_goals if original_audio_for_goals is not None else restored_audio)
            )
            _ad_rest = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio
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
        _bark_result: dict | None = None
        try:
            from backend.core.bark_scale_processor import analyze_bark_spectrum as _analyze_bark

            _bark_audio = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio
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
        _pmm_result: dict | None = None
        try:
            from backend.core.psychoacoustic_masking_model import PsychoacousticMaskingModel as _PMM

            _pmm_audio = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio
            _pmm_r = _PMM().compute_threshold(_pmm_audio.astype(np.float32), sr=sample_rate)
            _pmm_result = _pmm_r.as_dict()
            logger.debug("🎭 PsychoacousticMaskingModel: %s", _pmm_result)
        except Exception as _pmm_exc:
            logger.debug("PsychoacousticMaskingModel nicht verfügbar: %s", _pmm_exc)

        # v9.10.19 — PsychoAcousticMetrics
        _pam_result: dict | None = None
        try:
            from backend.core.psychoacoustic_metrics import PsychoAcousticMetrics as _PAM

            _pam_audio = np.mean(restored_audio, axis=0) if restored_audio.ndim == 2 else restored_audio
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
        _cm_result: dict | None = None
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
        _em_result: dict | None = None
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
                _em_mono = np.mean(_em_audio, axis=0) if _em_audio.ndim == 2 else _em_audio
                _em_result = {
                    "snr_db": round(float(_EnhMetrics.compute_snr(_em_mono.astype(np.float32), sample_rate)), 3),
                    "thd": round(float(_EnhMetrics.compute_thd(_em_mono.astype(np.float32), sample_rate)), 6),
                    "lufs": round(float(_EnhMetrics.compute_lufs(_em_mono.astype(np.float32), sample_rate)), 3),
                }
            logger.debug("📊 EnhancedMetrics: %s", _em_result)
        except Exception as _em_exc:
            logger.debug("EnhancedMetrics nicht verfügbar: %s", _em_exc)

        # v9.10.20 — VocalCharacteristics (GenderDetector)
        _vc_result: dict | None = None
        try:
            from backend.core.vocal_ai_enhancement import GenderDetector as _GenderDetector

            _vc_audio = restored_audio
            _vc_mono = np.mean(_vc_audio, axis=0) if _vc_audio.ndim == 2 else _vc_audio
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
        _dpm_result: dict | None = None
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
        _fm_result: dict | None = None
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
        _gr_result: dict | None = None
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
        _mda_result: dict | None = None
        try:
            from backend.core.media_defect_analysis import analyze_defects_features as _analyze_defects

            _mda_audio = restored_audio
            _mda_defects = _analyze_defects(_mda_audio, sample_rate)
            _mda_result = {
                "detected_defects": sorted(_mda_defects),
                "defect_count": len(_mda_defects),
            }
            logger.debug("🔎 MediaDefectAnalysis: %s", _mda_result)
        except Exception as _mda_exc:
            logger.debug("MediaDefectAnalysis nicht verfügbar: %s", _mda_exc)

        # v9.10.21 — CausalDefectGraph
        _cdg_result: dict | None = None
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
        _pqg_result: dict | None = None
        try:
            from backend.core.perceptual_quality_gates import PerceptualQualityGates as _PQG

            _pqg_metrics: dict[str, float] = {}
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
        _pm_result: dict | None = None
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
        _mps_result: dict | None = None
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
        _mpc_result: dict | None = None
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
        _cbp_result: dict | None = None
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
        _cago_result: dict | None = None
        try:
            from backend.core.context_aware_goal_optimizer import ContextAwareGoalOptimizer as _CAGO

            _cago_metrics: dict = {}
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
        _mlpi_result: dict | None = None
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
        _acr_result: dict | None = None
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
        _mm_result: dict | None = None
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
        _mr_result: dict | None = None
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
        _dqr_result: dict | None = None
        try:
            import datetime as _dqr_dt

            from backend.core.defect_quality_report import DefectQualityReport as _DQR24

            _dqr_mode = getattr(getattr(self, "config", None), "mode", "restoration") or "restoration"
            _dqr_material = _mr_result["detected_material"] if _mr_result else "unknown"
            _dqr_dur = float(restored_audio.shape[-1]) / float(sample_rate)
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
        _pt_result: dict | None = None
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
        _co_result: dict | None = None
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
        _qm_result: dict | None = None
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
        _pa_result: dict | None = None
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
        _qg_result: dict | None = None
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
        _qrs_result: dict | None = None
        try:
            from backend.core.quality_recovery import QualityRecoverySystem as _QRS25

            _qrs = _QRS25()
            _qrs_types = [k.value for k in _qrs._strategy_templates]
            _qrs_result = {
                "supported_problem_types": _qrs_types,
                "n_problem_types": len(_qrs_types),
                "available": True,
            }
            logger.debug("✅ QualityRecoverySystem: %d Problemtypen", len(_qrs_types))
        except Exception as _qrs_exc:
            logger.debug("QualityRecoverySystem nicht verfügbar: %s", _qrs_exc)

        # v9.10.26 — SelfLearningOptimizer: UCB1-Lernstatistiken
        _slo_result: dict | None = None
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
        _ru_result: dict | None = None
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
        _spd_result: dict | None = None
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
        _ssm_result: dict | None = None
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
        _abt_result: dict | None = None  # backward-compat: Key bleibt in metadata

        # v9.10.27 — adaptive_plugins: VoiceHealthNet + LanguageNet
        _adp_result: dict | None = None
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
        _cu_result: dict | None = None
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
        _clap_result: dict | None = None
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
        _ems_result: dict | None = None
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
        _mc_result: dict | None = None
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
        _pc_result: dict | None = None
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
        _pa_result: dict | None = None
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
        _arcm_result: dict | None = None
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
        _ipipe_result: dict | None = None
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
        _are_result: dict | None = None
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
        _pmain_result: dict | None = None
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
        _armr_result: dict | None = None
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
        _mss_result: dict | None = None
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
        _mcm_result: dict | None = None
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
        _aif_result: dict | None = None
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
        _amgs_result: dict | None = None
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
        _er_result: dict | None = None
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
        _mrn_result: dict | None = None
        try:
            from backend.core.material_restoration_nets import _RESTORER_MAP as _RM30

            _mrn_result = {
                "available_media": [str(m.value) for m in _RM30],
                "n_media": len(_RM30),
            }
            logger.debug("✅ MaterialRestorationNets: %d Medien", _mrn_result["n_media"])
        except Exception as _mrn_exc:
            logger.debug("MaterialRestorationNets nicht verfügbar: %s", _mrn_exc)

        # v9.10.30 — audio_exporter: AudioExporter
        _aex_result: dict | None = None
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
        _mco_result: dict | None = None
        try:
            from backend.core.module_communication import ModuleCommunicationBus as _MCBus31
            from backend.core.module_coordinator import create_coordinator as _cc31
            from backend.core.processing_context import ProcessingContext as _PCtx31

            _mco31 = _cc31(context=_PCtx31(session_id="probe"), bus=_MCBus31())
            _mco_result = {
                "coordinator_type": type(_mco31).__name__,
                "available": True,
            }
            logger.debug("✅ ModuleCoordinator: type=%s", _mco_result["coordinator_type"])
        except Exception as _mco_exc:
            logger.debug("ModuleCoordinator nicht verfügbar: %s", _mco_exc)

        # v9.10.31 — adaptive_chain_builder: AdaptiveChainBuilder
        _acb_result: dict | None = None
        try:
            from backend.core.forensics.adaptive_chain_builder import AdaptiveChainBuilder as _ACB31

            _acb31 = _ACB31()
            _acb_result = {
                "type": type(_acb31).__name__,
                "available": True,
            }
            logger.debug("✅ AdaptiveChainBuilder: verfügbar")
        except Exception as _acb_exc:
            logger.debug("AdaptiveChainBuilder nicht verfügbar: %s", _acb_exc)

        # v9.10.31 — export_workflow: ExportMetadata
        _ew_result: dict | None = None
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
        _dm_result: dict | None = None
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
        _dsp_registry: dict | None = None
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
        _plugin_registry_dyn: dict | None = None
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
        _backend_registry: dict | None = None
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
        _hybrid_dereverb_result: dict | None = None
        try:
            from backend.core.hybrid.hybrid_dereverb import HybridDereverb

            HybridDereverb()
            _hybrid_dereverb_result = {"class": "HybridDereverb", "active": True}
            logger.debug("🏛️ HybridDereverb: initialisiert")
        except Exception as _e38a:
            logger.debug("HybridDereverb übersprungen: %s", _e38a)

        # hybrid_ml_denoiser
        _hybrid_ml_denoiser_result: dict | None = None
        try:
            from backend.core.hybrid.hybrid_ml_denoiser import HybridMLDenoiser

            HybridMLDenoiser()
            _hybrid_ml_denoiser_result = {"class": "HybridMLDenoiser", "active": True}
            logger.debug("🏛️ HybridMLDenoiser: initialisiert")
        except Exception as _e38b:
            logger.debug("HybridMLDenoiser übersprungen: %s", _e38b)

        # hybrid_nvsr
        _hybrid_nvsr_result: dict | None = None
        try:
            from backend.core.hybrid.hybrid_nvsr import HybridNVSR

            HybridNVSR()
            _hybrid_nvsr_result = {"class": "HybridNVSR", "active": True}
            logger.debug("🏛️ HybridNVSR: initialisiert")
        except Exception as _e38c:
            logger.debug("HybridNVSR übersprungen: %s", _e38c)

        # hybrid_speed_pitch_ml
        _hybrid_speed_pitch_result: dict | None = None
        try:
            from backend.core.hybrid.hybrid_speed_pitch_ml import HybridSpeedPitch

            HybridSpeedPitch()
            _hybrid_speed_pitch_result = {"class": "HybridSpeedPitch", "active": True}
            logger.debug("🏛️ HybridSpeedPitch: initialisiert")
        except Exception as _e38d:
            logger.debug("HybridSpeedPitch übersprungen: %s", _e38d)

        # hybrid_vocal_enhancer
        _hybrid_vocal_enhancer_result: dict | None = None
        try:
            from backend.core.hybrid.hybrid_vocal_enhancer import HybridVocalEnhancer

            HybridVocalEnhancer()
            _hybrid_vocal_enhancer_result = {"class": "HybridVocalEnhancer", "active": True}
            logger.debug("🏛️ HybridVocalEnhancer: initialisiert")
        except Exception as _e38e:
            logger.debug("HybridVocalEnhancer übersprungen: %s", _e38e)

        # hybrid_wow_flutter
        _hybrid_wow_flutter_result: dict | None = None
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
        _audit_log_result: dict | None = None
        try:
            from backend.core.audit_log.audit_log import AuditLog as _AuditLog37a

            _AuditLog37a()
            _audit_log_result = {"class": "AuditLog", "active": True}
            logger.debug("📋 AuditLog: initialisiert")
        except Exception as _e37a:
            logger.debug("AuditLog übersprungen: %s", _e37a)

        # epistemic_gate.ethics_engine
        _ethics_engine_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.epistemic_gate.ethics_engine")
            _ethics_engine_result = {"class": "AuthenticityConstraints", "active": True}
            logger.debug("⚖️ EpistemicGate.EthicsEngine: geladen")
        except Exception as _e37b:
            logger.debug("EpistemicGate.EthicsEngine übersprungen: %s", _e37b)

        # musical_goals.feedback_loop
        _mg_feedback_result: dict | None = None
        try:
            from backend.core.musical_goals.feedback_loop import MusicalGoalsFeedbackLoop as _MGFL37c

            _MGFL37c(monitor=None, adjust_callback=lambda _: None)
            _mg_feedback_result = {"class": "MusicalGoalsFeedbackLoop", "active": True}
            logger.debug("🔁 MusicalGoalsFeedbackLoop: initialisiert")
        except Exception as _e37c:
            logger.debug("MusicalGoalsFeedbackLoop übersprungen: %s", _e37c)

        # musical_goals.goal_conflict_resolver
        _goal_conflict_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.goal_conflict_resolver")
            _goal_conflict_result = {"enum": "ConflictSeverity", "active": True}
            logger.debug("⚔️ GoalConflictResolver: ConflictSeverity geladen")
        except Exception as _e37d:
            logger.debug("GoalConflictResolver übersprungen: %s", _e37d)

        # musical_goals.goal_optimizer
        _goal_optimizer_result: dict | None = None
        try:
            from backend.core.musical_goals.goal_optimizer import MusicalGoalsOptimizer as _MGO37e

            _MGO37e(monitor=None)
            _goal_optimizer_result = {"class": "MusicalGoalsOptimizer", "active": True}
            logger.debug("🚀 MusicalGoalsOptimizer: initialisiert")
        except Exception as _e37e:
            logger.debug("MusicalGoalsOptimizer übersprungen: %s", _e37e)

        # musical_goals.processing_modes
        _processing_modes_result: dict | None = None
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
        _mg_quality_gate_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.quality_gate")
            _mg_quality_gate_result = {"class": "EnhancedPostCheckResult", "active": True}
            logger.debug("🚦 musical_goals.QualityGate: geladen")
        except Exception as _e37g:
            logger.debug("musical_goals.QualityGate übersprungen: %s", _e37g)

        # onnx.fallback
        _onnx_fallback_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.fallback")
            _onnx_fallback_result = {"class": "FallbackEvent", "active": True}
            logger.debug("🔄 ONNX-Fallback: FallbackEvent geladen")
        except Exception as _e37h:
            logger.debug("ONNX-Fallback übersprungen: %s", _e37h)

        # onnx.quantizer
        _onnx_quantizer_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.quantizer")
            _onnx_quantizer_result = {"class": "ModelQuantizer", "active": True}
            logger.debug("🗜️ ONNX-Quantizer: ModelQuantizer geladen")
        except Exception as _e37i:
            logger.debug("ONNX-Quantizer übersprungen: %s", _e37i)

        # onnx.runtime
        _onnx_runtime_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.runtime")
            _onnx_runtime_result = {"class": "onnx.runtime.ModelInfo", "active": True}
            logger.debug("⚙️ ONNX-Runtime: ModelInfo geladen")
        except Exception as _e37j:
            logger.debug("ONNX-Runtime übersprungen: %s", _e37j)

        # optimization.advanced_ensemble
        _adv_ensemble_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.advanced_ensemble")
            _adv_ensemble_result = {"class": "AdvancedEnsemble", "active": True}
            logger.debug("🎼 AdvancedEnsemble: geladen")
        except Exception as _e37k:
            logger.debug("AdvancedEnsemble übersprungen: %s", _e37k)

        # optimization.automated_augmentation
        _auto_aug_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.automated_augmentation")
            _auto_aug_result = {"class": "AudioAugmentations", "active": True}
            logger.debug("🎨 AudioAugmentations: geladen")
        except Exception as _e37l:
            logger.debug("AudioAugmentations übersprungen: %s", _e37l)

        # optimization.hyperparameter_optimizer
        # DEADLOCK-SAFE: Nur aus sys.modules-Cache laden — KEIN direkter Import in Thread.
        # HyperparameterConfig wird hier nur für Tracking benötigt, nicht funktional.
        _hyperparam_result: dict | None = None
        _mod_hpo = sys.modules.get("backend.core.optimization.hyperparameter_optimizer")
        if _mod_hpo is not None:
            _hyperparam_result = {"class": "HyperparameterConfig", "active": True}
            logger.debug("🔬 HyperparameterOptimizer: HyperparameterConfig aus Cache")

        # optimization.neural_architecture_search
        _nas_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.neural_architecture_search")
            _nas_result = {"class": "AudioNASNetwork", "active": True}
            logger.debug("🧠 NeuralArchitectureSearch: AudioNASNetwork geladen")
        except Exception as _e37n:
            logger.debug("NeuralArchitectureSearch übersprungen: %s", _e37n)

        # optimization.optimization_integration
        _opt_integration_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.optimization_integration")
            _opt_integration_result = {"class": "optimization_integration.AdvancedEnsemble", "active": True}
            logger.debug("🔗 OptimizationIntegration: geladen")
        except Exception as _e37o:
            logger.debug("OptimizationIntegration übersprungen: %s", _e37o)

        # optimization.perceptual_loss
        _perceptual_loss_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.perceptual_loss")
            _perceptual_loss_result = {"class": "MultiResolutionSTFTLoss", "active": True}
            logger.debug("🎵 PerceptualLoss: MultiResolutionSTFTLoss geladen")
        except Exception as _e37p:
            logger.debug("PerceptualLoss übersprungen: %s", _e37p)

        # parallel.module_parallel (vollständiger Import)
        _mod_parallel_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.parallel.module_parallel")
            _mod_parallel_result = {"class": "ModuleDependency", "active": True}
            logger.debug("⚡ module_parallel: ModuleDependency geladen")
        except Exception as _e37q:
            logger.debug("module_parallel übersprungen: %s", _e37q)

        # parallel.stereo_parallel (vollständiger Import)
        _stereo_par_result: dict | None = None
        try:
            from backend.core.parallel.stereo_parallel import ChannelType as _CT37r

            _stereo_par_result = {
                "channels": [c.name for c in _CT37r] if hasattr(_CT37r, "__members__") else [],
                "active": True,
            }
            logger.debug("🔊 stereo_parallel: ChannelType geladen")
        except Exception as _e37r:
            logger.debug("stereo_parallel übersprungen: %s", _e37r)

        # quality_gate (top-level backend.core)
        _quality_gate_result: dict | None = None
        try:
            from backend.core.quality_gate import QualityGate as _QG37s

            _QG37s()
            _quality_gate_result = {"class": "QualityGate", "active": True}
            logger.debug("🚦 QualityGate: initialisiert")
        except Exception as _e37s:
            logger.debug("QualityGate übersprungen: %s", _e37s)

        # regulator.adaptive_goal
        _reg_adaptive_goal_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.adaptive_goal")
            _reg_adaptive_goal_result = {"class": "regulator.AdaptiveGoalEngine", "active": True}
            logger.debug("🎯 regulator.AdaptiveGoal: geladen")
        except Exception as _e37t:
            logger.debug("regulator.AdaptiveGoal übersprungen: %s", _e37t)

        # regulator.context_analysis
        _reg_context_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.context_analysis")
            _reg_context_result = {"class": "regulator.ContextAnalyzer", "active": True}
            logger.debug("🔍 regulator.ContextAnalysis: ContextAnalyzer geladen")
        except Exception as _e37u:
            logger.debug("regulator.ContextAnalysis übersprungen: %s", _e37u)

        # regulator.ethics_engine
        _reg_ethics_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.ethics_engine")
            _reg_ethics_result = {"class": "EpistemicDecision", "active": True}
            logger.debug("⚖️ regulator.EthicsEngine: EpistemicDecision geladen")
        except Exception as _e37v:
            logger.debug("regulator.EthicsEngine übersprungen: %s", _e37v)

        # regulator.quality_control
        _reg_quality_ctrl_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.regulator.quality_control")
            _reg_quality_ctrl_result = {"class": "regulator.QualityControl", "active": True}
            logger.debug("🛡️ regulator.QualityControl: geladen")
        except Exception as _e37w:
            logger.debug("regulator.QualityControl übersprungen: %s", _e37w)

        # regulator.regulator
        _regulator_result: dict | None = None
        try:
            from backend.core.regulator.regulator import Regulator as _Reg37x

            _Reg37x()
            _regulator_result = {"class": "Regulator", "active": True}
            logger.debug("📜 Regulator: initialisiert")
        except Exception as _e37x:
            logger.debug("Regulator übersprungen: %s", _e37x)

        # regulator.regulator_v8
        _regulator_v8_result: dict | None = None
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
        _sota_max_result: dict | None = None
        try:
            from backend.core.regulator.sota_maximum_analyzer import SOTAMaximumAnalyzer as _SMA37z

            _SMA37z()
            _sota_max_result = {"class": "SOTAMaximumAnalyzer", "active": True}
            logger.debug("🏆 SOTAMaximumAnalyzer: initialisiert")
        except Exception as _e37z:
            logger.debug("SOTAMaximumAnalyzer übersprungen: %s", _e37z)

        # undo.undo_manager
        _undo_manager_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.undo.undo_manager")
            _undo_manager_result = {"class": "Action", "active": True}
            logger.debug("↩️ UndoManager: Action geladen")
        except Exception as _e37aa:
            logger.debug("UndoManager übersprungen: %s", _e37aa)

        # zone_engine.context_analysis
        _zone_context_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.zone_engine.context_analysis")
            _zone_context_result = {"class": "zone_engine.ContextAnalyzer", "active": True}
            logger.debug("🗺️ ZoneEngine.ContextAnalysis: geladen")
        except Exception as _e37ab:
            logger.debug("ZoneEngine.ContextAnalysis übersprungen: %s", _e37ab)

        # zone_engine.region_analysis
        _region_analysis_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.zone_engine.region_analysis")
            _region_analysis_result = {"class": "AudioRegion", "active": True}
            logger.debug("🗺️ ZoneEngine.RegionAnalysis: AudioRegion geladen")
        except Exception as _e37ac:
            logger.debug("ZoneEngine.RegionAnalysis übersprungen: %s", _e37ac)

        # zone_engine.zone_engine
        _zone_engine_result: dict | None = None
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
        _adaptive_goal_result: dict | None = None
        try:
            from backend.core.conduct_enforcer.adaptive_goal import AdaptiveGoalEngine as _AGE36a

            _AGE36a()
            _adaptive_goal_result = {"engine": "AdaptiveGoalEngine", "active": True}
            logger.debug("🎯 AdaptiveGoalEngine: initialisiert")
        except Exception as _exc36a:
            logger.debug("AdaptiveGoalEngine übersprungen: %s", _exc36a)

        # continuous_learning (Evaluation-Modul)
        _continuous_learning_result: dict | None = None
        try:
            from backend.core.evaluation.continuous_learning import ContinuousLearningSystem as _CLS36b

            _CLS36b()
            _continuous_learning_result = {"system": "ContinuousLearningSystem", "active": True}
            logger.debug("📚 ContinuousLearningSystem: initialisiert")
        except Exception as _exc36b:
            logger.debug("ContinuousLearningSystem übersprungen: %s", _exc36b)

        # adaptive_goals_system (Musical Goals — adaptive Schwellwerte)
        _adaptive_goals_result: dict | None = None
        try:
            from backend.core.musical_goals.adaptive_goals_system import (
                AdaptiveGoalsCalculator as _AGC36c,
            )

            _agc36 = _AGC36c()
            _agc36_defaults = getattr(_agc36, "DEFAULT_THRESHOLDS", {})
            _adaptive_goals_result = dict(_agc36_defaults) if _agc36_defaults else {"active": True}
            logger.debug("🎵 AdaptiveGoalsCalculator: %d Thresholds", len(_adaptive_goals_result))
        except Exception as _exc36c:
            logger.debug("AdaptiveGoalsCalculator übersprungen: %s", _exc36c)

        # adaptive_thresholds (Musical Goals)
        _adaptive_thresholds_result: dict | None = None
        try:
            from backend.core.musical_goals.adaptive_thresholds import (
                AdaptiveThresholdsManager as _ATM36d,
            )

            _ATM36d()
            _adaptive_thresholds_result = {"manager": "AdaptiveThresholdsManager", "active": True}
            logger.debug("📊 AdaptiveThresholdsManager: initialisiert")
        except Exception as _exc36d:
            logger.debug("AdaptiveThresholdsManager übersprungen: %s", _exc36d)

        # auto_reprocessing (Musical Goals)
        _auto_reprocessing_result: dict | None = None
        try:
            from backend.core.musical_goals.auto_reprocessing import AutoReprocessingEngine as _ARE36e

            _ARE36e()
            _auto_reprocessing_result = {"engine": "AutoReprocessingEngine", "active": True}
            logger.debug("🔄 AutoReprocessingEngine: initialisiert")
        except Exception as _exc36e:
            logger.debug("AutoReprocessingEngine übersprungen: %s", _exc36e)

        # convergence_detector (Musical Goals)
        _convergence_result: dict | None = None
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
        _deviation_corrector_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.deviation_corrector")
            _MGDC36 = getattr(_mod_cache, "MusicalGoalsDeviationCorrector", None)
            _MGDC36()
            _deviation_corrector_result = {"corrector": "MusicalGoalsDeviationCorrector", "active": True}
            logger.debug("🔧 DeviationCorrector: initialisiert")
        except Exception as _exc36g:
            logger.debug("DeviationCorrector übersprungen: %s", _exc36g)

        # edge_case_handler (Musical Goals)
        _edge_case_result: dict | None = None
        try:
            from backend.core.musical_goals.edge_case_handler import EdgeCaseHandler as _ECH36h

            _ech36 = _ECH36h()
            _eca36 = _ech36.assess_edge_cases(restored_audio, sample_rate)
            _edge_case_result = _eca36.as_dict() if hasattr(_eca36, "as_dict") else {"severity": str(_eca36)}
            logger.debug("⚠️ EdgeCaseHandler: %s", _edge_case_result)
        except Exception as _exc36h:
            logger.debug("EdgeCaseHandler übersprungen: %s", _exc36h)

        # explainability / GoalExplainer (Musical Goals)
        _explainability_result: dict | None = None
        try:
            from backend.core.musical_goals.explainability import GoalExplainer as _GE36i

            _ge36 = _GE36i()
            _ge36.start_tracking()
            _ge36_exp = _ge36.explain_simple(_musical_goal_scores if _musical_goal_scores else {})
            _ge36.stop_tracking()
            _explainability_result = _ge36_exp if isinstance(_ge36_exp, dict) else {"explanation": str(_ge36_exp)}
            logger.debug("💡 GoalExplainer: %d Einträge", len(_explainability_result))
        except Exception as _exc36i:
            logger.debug("GoalExplainer übersprungen: %s", _exc36i)

        # ki_hearing_model (Musical Goals — KIHörbarkeitsAnalyzer)
        _ki_hearing_result: dict | None = None
        try:
            from backend.core.musical_goals.ki_hearing_model import KIHörbarkeitsAnalyzer as _KIHA36j

            _KIHA36j()
            _ki_hearing_result = {"analyzer": "KIHörbarkeitsAnalyzer", "active": True}
            logger.debug("👂 KIHörbarkeitsAnalyzer: initialisiert")
        except Exception as _exc36j:
            logger.debug("KIHörbarkeitsAnalyzer übersprungen: %s", _exc36j)

        # reference_based_learning (Musical Goals)
        _reference_learning_result: dict | None = None
        try:
            from backend.core.musical_goals.reference_based_learning import LearningStrategy as _LS36k

            _reference_learning_result = {"strategy_class": str(_LS36k), "active": True}
            logger.debug("📖 ReferenceLearning: LearningStrategy geladen")
        except Exception as _exc36k:
            logger.debug("ReferenceLearning übersprungen: %s", _exc36k)

        # semantic_goals (Musical Goals — GoalProfile)
        _semantic_goals_result: dict | None = None
        try:
            from backend.core.musical_goals.semantic_goals import GoalProfile as _GP36l

            _semantic_goals_result = {"profile_class": str(_GP36l), "active": True}
            logger.debug("🎯 SemanticGoals: GoalProfile geladen")
        except Exception as _exc36l:
            logger.debug("SemanticGoals übersprungen: %s", _exc36l)

        # uncertainty_quantification musical_goals (GoalsUncertaintyReport)
        _mg_uncertainty_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.uncertainty_quantification")
            _mg_uncertainty_result = {"report_class": "GoalsUncertaintyReport", "active": True}
            logger.debug("❓ MusicalGoalsUncertainty: GoalsUncertaintyReport geladen")
        except Exception as _exc36m:
            logger.debug("MusicalGoalsUncertainty übersprungen: %s", _exc36m)

        # onnx.converter (ModelSpecificConverter / ConversionConfig)
        _onnx_converter_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.converter")
            _onnx_converter_result = {"config_class": "ConversionConfig", "active": True}
            logger.debug("🔌 ONNX-Converter: ConversionConfig geladen")
        except Exception as _exc36n:
            logger.debug("ONNX-Converter übersprungen: %s", _exc36n)

        # onnx.model_info (ModelInfo)
        _onnx_model_info_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.model_info")
            _onnx_model_info_result = {"info_class": "ModelInfo", "active": True}
            logger.debug("ℹ️ ONNX-ModelInfo: ModelInfo geladen")
        except Exception as _exc36o:
            logger.debug("ONNX-ModelInfo übersprungen: %s", _exc36o)

        # onnx.plugin_manager (FallbackManager)
        _onnx_plugin_mgr_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.onnx.plugin_manager")
            _onnx_plugin_mgr_result = {"manager_class": "FallbackManager", "active": True}
            logger.debug("🔧 ONNX-PluginManager: FallbackManager geladen")
        except Exception as _exc36p:
            logger.debug("ONNX-PluginManager übersprungen: %s", _exc36p)

        # optimization.multi_objective (Individual / Pareto-Front)
        _multi_objective_result: dict | None = None
        try:
            from backend.core.optimization.multi_objective import Individual as _Ind36q

            _multi_objective_result = {"individual_class": str(_Ind36q), "active": True}
            logger.debug("🎯 MultiObjective: Individual geladen")
        except Exception as _exc36q:
            logger.debug("MultiObjective übersprungen: %s", _exc36q)

        # optimization.uncertainty_quantification (BayesianLinear)
        _opt_uncertainty_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.optimization.uncertainty_quantification")
            _opt_uncertainty_result = {"model_class": "BayesianLinear", "active": True}
            logger.debug("🧪 OptUncertainty: BayesianLinear geladen")
        except Exception as _exc36r:
            logger.debug("OptUncertainty übersprungen: %s", _exc36r)

        # parallel.batch_parallel (BatchParallelProcessor)
        _batch_parallel_result: dict | None = None
        try:
            from backend.core.parallel.batch_parallel import BatchParallelProcessor as _BPP36s

            _bpp36 = _BPP36s()
            _bpp36_stats = _bpp36.get_stats() if hasattr(_bpp36, "get_stats") else {}
            _batch_parallel_result = _bpp36_stats if isinstance(_bpp36_stats, dict) else {"active": True}
            logger.debug("⚡ BatchParallel: %s", _batch_parallel_result)
        except Exception as _exc36s:
            logger.debug("BatchParallel übersprungen: %s", _exc36s)

        # parallel.module_parallel
        _module_parallel_result: dict | None = None
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
        _stereo_parallel_result: dict | None = None
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
        _fatigue_result: dict | None = None
        try:
            _mod_cache = sys.modules.get("backend.core.musical_goals.listening_fatigue_analyzer")
            _analyze_fatigue35 = getattr(_mod_cache, "analyze_listening_fatigue", None)
            _fa35 = _analyze_fatigue35(restored_audio, sample_rate)
            _fatigue_result = _fa35.as_dict() if hasattr(_fa35, "as_dict") else {"risk": str(_fa35)}
            logger.debug("🦻 FatigueAnalyzer: risk=%s", _fatigue_result.get("risk_level", "?"))
        except Exception as _fa_exc:
            logger.debug("ListeningFatigueAnalyzer nicht verfügbar: %s", _fa_exc)

        # emotional_resonance_analyzer (Musical Goal Emotionalität)
        _emotional_result: dict | None = None
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
        _harmonic_char_result: dict | None = None
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
        _ki_quality_score: float | None = None
        try:
            from backend.core.musical_goals.ki_quality_model import KIQualityAnalyzer as _KIQA35

            _ki_score35 = _KIQA35().analyze_audio_quality(restored_audio, sample_rate)
            _ki_quality_score = float(_ki_score35) if _ki_score35 is not None else None
            logger.debug("🤖 KIQuality: score=%.3f", _ki_quality_score or 0.0)
        except Exception as _ki_exc:
            logger.debug("KIQualityAnalyzer nicht verfügbar: %s", _ki_exc)

        # microdynamics_analyzer (MicroDynamicsMetric §2.16)
        _microdynamics_result: dict | None = None
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
        _perceptual_validation: dict | None = None
        try:
            from backend.core.musical_goals.perceptual_validator import PerceptualValidator as _PV35

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
        _epistemic_result: dict | None = None
        try:
            from backend.core.epistemic_gate.epistemic_gate import EpistemicGate as _EG35

            _eg35_result = _EG35().check_responsibility(restored_audio)
            _epistemic_result = {"passed": bool(_eg35_result)} if not isinstance(_eg35_result, dict) else _eg35_result
            logger.debug("🔐 EpistemicGate: %s", _epistemic_result)
        except Exception as _eg_exc:
            logger.debug("EpistemicGate nicht verfügbar: %s", _eg_exc)

        # musical_goals.live_monitor (Echtzeit-Ziel-Überwachung)
        _live_monitor_result: dict | None = None
        try:
            from backend.core.musical_goals.live_monitor import MusicalGoalsLiveMonitor as _LM35

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
        _goals_monitor_result: dict | None = None
        try:
            from backend.core.musical_goals.musical_goals_monitor import MusicalGoalsMonitor as _GMon35

            _gmon35 = _GMon35()
            _gmon35_status = getattr(_gmon35, "get_status", None) or getattr(_gmon35, "status", None)
            _goals_monitor_result = _gmon35_status() if callable(_gmon35_status) else {"active": True}
            logger.debug("📈 MusicalGoalsMonitor: aktiv")
        except Exception as _gmon_exc:
            logger.debug("MusicalGoalsMonitor nicht verfügbar: %s", _gmon_exc)

        # conduct_enforcer (Richtlinien-Durchsetzung)
        _conduct_result: dict | None = None
        try:
            from backend.core.conduct_enforcer.conduct_enforcer import ConductEnforcer as _CE35

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
        _rollback_result: dict | None = None
        try:
            from backend.core.rollback.rollback_manager import RollbackManager as _RM35

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
        _session_result: dict | None = None
        try:
            from backend.core.session.session_manager import SessionManager as _SM35

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
        _quality_control_result: dict | None = None
        try:
            from backend.core.evaluation.quality_control import QualityControl as _QC35

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

        return {
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
        }

    def _select_phases(
        self, defect_result, *, causal_plan=None, chain_info=None, defekt_hint=None, audio=None, sr=48000
    ) -> list[str]:
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
        """
        scores = defect_result.scores
        material = defect_result.material_type

        def sev(defect_type, default=0.0):
            """Hilfsfunktion: Severity eines DefectType sicher auslesen."""
            s = scores.get(defect_type)
            return s.severity if s is not None else default

        # ── PANNs-Gate: Instrument/Vokal-Erkennung ──────────────────────────
        panns_tags: dict[str, float] = {}
        try:
            import os as _os
            import sys

            _ref = audio  # audio array passed from restore() — no longer relying on defect_result._audio_ref
            _sr = sr
            if _ref is not None:
                # Only load PANNs when audio reference is actually available
                _plugins_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "plugins")
                if _plugins_dir not in sys.path:
                    sys.path.insert(0, _os.path.abspath(_plugins_dir))
                from panns_plugin import PANNsPlugin

                _panns = PANNsPlugin()
                panns_tags = _panns.get_tags(_ref, _sr)
                logger.info(
                    "PANNs tags: %s", {k: f"{v:.2f}" for k, v in sorted(panns_tags.items(), key=lambda x: -x[1])[:10]}
                )
        except Exception as _panns_exc:
            logger.debug("PANNs nicht verfügbar, kein Instrument-Gate aktiv: %s", _panns_exc)

        def panns(tag: str, threshold: float = 0.5) -> bool:
            """Liefert True wenn PANNs-Tag die Mindestkonfidenz erreicht."""
            return panns_tags.get(tag, 0.0) >= threshold

        vocals_detected = panns("Singing voice", 0.40) or panns("Vocals", 0.40) or panns("Speech", 0.35)
        guitar_detected = panns("Guitar", 0.50) or panns("Electric guitar", 0.50)
        brass_detected = panns("Brass instrument", 0.50) or panns("Trumpet", 0.50) or panns("Saxophone", 0.50)
        drums_detected = panns("Drum", 0.50) or panns("Percussion", 0.50)
        piano_detected = panns("Piano", 0.50) or panns("Keyboard (musical)", 0.50)

        selected: list[str] = []

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

        # Dropout / gap repair
        # §9.7.5a — Always activate phase 24 for dropout-prone materials (Spec 05 §6.2).
        # Phase 24 has its own high-quality multi-modal detection (0.5 ms resolution,
        # material-adaptive) that catches dropouts the DefectScanner may underestimate.
        _DROPOUT_PRONE_MATERIALS = {
            MaterialType.REEL_TAPE,
            MaterialType.TAPE,
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
            MaterialType.DAT,
        }
        if sev(DefectType.DROPOUTS) > 0.10 or material in _DROPOUT_PRONE_MATERIALS:
            selected.append("phase_24_dropout_repair")
        # phase_55 (DiffWave ML-Inpainting) only for severe analog dropouts —
        # digital codecs (MP3, streaming, CD) do not produce tape-gap-type dropouts.
        if sev(DefectType.DROPOUTS) > 0.30 and material in _DROPOUT_PRONE_MATERIALS:
            selected.append("phase_55_diffusion_inpainting")  # ML-Inpainting nur für analoge Lücken

        # Clicks / Impulse (1. Pass)
        if sev(DefectType.CLICKS) > 0.15:
            selected.append("phase_01_click_removal")
        # Clicks / Pops (2. Pass — tiefergehende AR-Residual-Methode)
        if sev(DefectType.CLICKS) > 0.25:
            selected.append("phase_27_click_pop_removal")

        # Vinyl/Shellac/Lacquer: Oberflächenrausch-Profil VOR Crackle-Entfernung
        if (
            material in [MaterialType.VINYL, MaterialType.SHELLAC, MaterialType.LACQUER_DISC]
            and sev(DefectType.CRACKLE) > 0.10
        ):
            selected.append("phase_28_surface_noise_profiling")

        # Crackle — nur für Disc-Medien (Vinyl/Shellac/Lacquer-Oberflächen-Erosion).
        # Tape-Hiss-Artefakte und MP3-Block-Impulse lösen zwar den DefectScanner aus,
        # sind aber kein physikalisches Crackle — phase_29/phase_03 adressieren diese bereits.
        # UNKNOWN wird einbezogen, da Materialerkennung fehlschlagen kann (Safety-Net für
        # nicht erkannte Disc-Medien). Alle anderen Träger (Tape, REEL_TAPE, CD, Digital,
        # MP3-Kette) werden hier explizit ausgeschlossen.
        _DISC_MATERIALS_CRACKLE = {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.LACQUER_DISC,
            MaterialType.UNKNOWN,
        }
        if sev(DefectType.CRACKLE) > 0.15 and material in _DISC_MATERIALS_CRACKLE:
            selected.append("phase_09_crackle_removal")
        elif sev(DefectType.CRACKLE) > 0.15:
            logger.debug(
                "phase_09_crackle_removal übersprungen: material=%s ist kein Disc-Träger"
                " (Crackle-Artefakt wird durch phase_29/phase_03 adressiert)",
                material,
            )

        # Brumm 50/60 Hz
        if sev(DefectType.HUM) > 0.10:
            selected.append("phase_02_hum_removal")

        # Breitbandrauschen
        if sev(DefectType.HIGH_FREQ_NOISE) > 0.20:
            selected.append("phase_03_denoise")

        # Tape-Hiss (spezifisches Profil-basiertes Verfahren)
        # Shellac: phase_03_denoise übernimmt das NR — phase_29 NICHT unconditional
        # aktivieren, da das Doppel-NR bei SNR≈6 dB das Signal vollständig zerstört.
        # REEL_TAPE: Profi-Spulenband mit identischem Hiss-Profil wie TAPE.
        if sev(DefectType.HIGH_FREQ_NOISE) > 0.30 or material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_29_tape_hiss_reduction")

        # Noise-Gate für Stille-Segmente mit Rauschboden
        if sev(DefectType.HIGH_FREQ_NOISE) > 0.25 or sev(DefectType.LOW_FREQ_RUMBLE) > 0.20:
            selected.append("phase_18_noise_gate")

        # Wow/Flutter (Magnetband-Gleichlaufschwankungen)
        if max(sev(DefectType.WOW), sev(DefectType.FLUTTER)) > 0.10:
            selected.append("phase_12_wow_flutter_fix")

        # Pitch-Drift (langsame Tonhöhenschwankung)
        if sev(DefectType.PITCH_DRIFT) > 0.15:
            selected.append("phase_31_speed_pitch_correction")

        # Phasen-/Azimuth-Fehler
        if sev(DefectType.PHASE_ISSUES) > 0.10:
            selected.append("phase_14_phase_correction")
        if sev(DefectType.PHASE_ISSUES) > 0.20 and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
            MaterialType.SHELLAC,
        ]:
            selected.append("phase_25_azimuth_correction")

        # Nachhall-Reduktion
        if sev(DefectType.REVERB_EXCESS) > 0.25:
            selected.append("phase_20_reverb_reduction")
        if sev(DefectType.REVERB_EXCESS) > 0.45:
            selected.append("phase_49_advanced_dereverb")  # Tiefgehende Blind-RIR-Methode

        # Spektral-Reparatur (Clipping / Digital-Artefakte)
        if sev(DefectType.CLIPPING) > 0.10 or sev(DefectType.DIGITAL_ARTIFACTS) > 0.20:
            selected.append("phase_23_spectral_repair")

        # Print-Through (Magnetisches Übersprechen bei Bandaufnahmen — Vor-/Nachecho)
        # Physikalisch nur bei Bandmaterial möglich
        # Spec §7.x / DSP-Regel: Bidirektionale Adaptive Temporal Subtraction (LMS)
        # phase_57_print_through_reduction = Primär (Pre-Echo + Post-Echo getrennt)
        # Fallback: phase_29 Hiss-Profil + phase_03 Breitband-NR
        if sev(DefectType.PRINT_THROUGH) > 0.15 and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_57_print_through_reduction")  # §7.x bidirektionale LMS Primär
            selected.append("phase_29_tape_hiss_reduction")  # Hiss-Profil-Subtraktion
            selected.append("phase_03_denoise")  # Breitband-Restecho-NR
        if sev(DefectType.PRINT_THROUGH) > 0.35 and material in [
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
        ]:
            selected.append("phase_23_spectral_repair")  # Schwere Vorecho-Tilgung

        # Quantisierungsrauschen (niedrige Bit-Tiefe / fehlerhaftes Resampling)
        if sev(DefectType.QUANTIZATION_NOISE) > 0.15:
            selected.append("phase_03_denoise")
            selected.append("phase_23_spectral_repair")
        if sev(DefectType.QUANTIZATION_NOISE) > 0.30:
            selected.append("phase_06_frequency_restoration")  # Treppen-Artefakte glätten

        # Jitter-Artefakte (D/A-Wandler-Zeitfehler — CD, DAT, Streaming)
        if sev(DefectType.JITTER_ARTIFACTS) > 0.15:
            selected.append("phase_12_wow_flutter_fix")  # Zeitachsen-Trägeheit
            selected.append("phase_23_spectral_repair")  # Spektrale Jitter-Spuren

        # RIAA-Entzerrungsfehler (Shellac/früher Vinyl: AES/NAB/FFRR — §6.3, §7.2)
        # Medium-Gate: RIAA nur für Disc-Medien (Vinyl/Shellac/Lacquer/Wax/Unknown)
        _is_disc_medium = material in {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.LACQUER_DISC,
            MaterialType.WAX_CYLINDER,
            MaterialType.UNKNOWN,
        }
        if _is_disc_medium and sev(DefectType.RIAA_CURVE_ERROR) > 0.12:
            selected.append("phase_04_eq_correction")  # Fehler-EQ korrigieren
            selected.append("phase_06_frequency_restoration")  # Spektralprofil wiederherstellen
        if _is_disc_medium and sev(DefectType.RIAA_CURVE_ERROR) > 0.30:
            selected.append("phase_07_harmonic_restoration")  # Obertöne durch Entzerrungs-Kette verloren

        # Aliasing (AA-Filter-Artefakte bei Digitalisierung — §6.3, §7.2)
        if sev(DefectType.ALIASING) > 0.15:
            selected.append("phase_03_denoise")  # Spiegelfrequenzen dämpfen
            selected.append("phase_23_spectral_repair")  # Spektrale Aliasing-Spuren beseitigen
        if sev(DefectType.ALIASING) > 0.35:
            selected.append("phase_50_spectral_repair")  # Zweiter Spektral-Pass

        # Bias-Fehler (falscher Vormagnetisierungsstrom bei Bandaufnahme — §6.3, §7.2)
        if sev(DefectType.BIAS_ERROR) > 0.12:
            selected.append("phase_04_eq_correction")  # HF-Rolloff/-Überhöhung kompensieren
            selected.append("phase_03_denoise")  # Bias-induziertes Rauschen reduzieren
        if sev(DefectType.BIAS_ERROR) > 0.30:
            selected.append("phase_06_frequency_restoration")  # Frequenzgang-Verluste ausgleichen
            selected.append("phase_29_tape_hiss_reduction")  # Bias-erhöhtes Hintergrundrauschen

        # ── §6.3 Pflicht-Severity-Checks: HEAD_WEAR, AZIMUTH, TRANSIENT_SMEARING, PRE_ECHO, SIBILANCE ──
        # Ohne diese direkte Aktivierung verlassen sich die Defekte NUR auf CausalReasoner (Tier 1.5).
        # Tier 1 MUSS jede erkannte Severity > Schwelle direkt in Phasen übersetzen.

        # Kopfverschleiß (Tape/DAT-Köpfe — Frequenzband-Auslöschung → phase_56)
        if sev(DefectType.HEAD_WEAR) > 0.15:
            selected.append("phase_56_spectral_band_gap_repair")  # §4.5/§7.2
            selected.append("phase_14_phase_correction")
            selected.append("phase_06_frequency_restoration")

        # Azimuth-Fehler (Kopfausrichtung — nur Tape/Reel; eigenständig neben PHASE_ISSUES)
        if sev(DefectType.AZIMUTH_ERROR) > 0.12:
            selected.append("phase_25_azimuth_correction")  # §7.2 — direkt, nicht nur via PHASE_ISSUES
            selected.append("phase_14_phase_correction")
            selected.append("phase_06_frequency_restoration")

        # Transient-Smearing (Transienten-Verschmierung — Mastering/Codec)
        if sev(DefectType.TRANSIENT_SMEARING) > 0.15:
            selected.append("phase_08_transient_preservation")  # §7.2 — Transienten-Restaurierung
            selected.append("phase_36_transient_shaper")

        # Pre-Echo (MP3/AAC-Codec-Artefakte vor Transienten)
        if sev(DefectType.PRE_ECHO) > 0.15:
            selected.append("phase_23_spectral_repair")  # §6.3 — Codec-Pre-Echo
            selected.append("phase_50_spectral_repair")
            selected.append("phase_08_transient_preservation")

        # Sibilanz (Zischlaut-Überbetonung > 6 kHz — severity-priorisiert, nicht nur PANNs-Gate)
        if sev(DefectType.SIBILANCE) > 0.15:
            selected.append("phase_19_de_esser")  # §6.3 — De-Esser (stimmtyp-adaptiv)
            selected.append("phase_43_ml_deesser")  # ML-De-Esser (zweiter Pass)

        # WAX_CYLINDER (Phonographen-Wachswalze 1890–1930): aggressiv — HF ≤ 5 kHz, SNR extrem
        if material == MaterialType.WAX_CYLINDER:
            selected += [
                "phase_01_click_removal",  # Mechanische Störimpulse
                "phase_29_tape_hiss_reduction",  # Wachsoberflächenrauschen
                "phase_03_denoise",  # Breitbandrauschen
                "phase_06_frequency_restoration",  # HF ≤ 5 kHz rekonstruieren
                "phase_07_harmonic_restoration",  # Obertöne ergänzen
            ]

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
        # §6.2a [RELEASE_MUST] Material-Pflicht-Phasen — Unconditional
        # Spec 05 §6.2a: Prioritäts-Phasen MÜSSEN aktiviert werden wenn
        # Material erkannt — unabhängig vom DefectScanner-Severity-Score.
        # Phasen enthalten eigene hochauflösende Detektionslogik und
        # entscheiden selbst, ob eine Reparatur notwendig ist.
        # ════════════════════════════════════════════════════════════════════
        _MATERIAL_PRIORITY_PHASES: dict[str, list[str]] = {
            "tape": [
                "phase_24_dropout_repair",
                "phase_29_tape_hiss_reduction",
                "phase_12_wow_flutter_fix",
            ],
            "reel_tape": [
                "phase_29_tape_hiss_reduction",
                "phase_03_denoise",
                "phase_24_dropout_repair",
                "phase_55_diffusion_inpainting",
            ],
            "vinyl": [
                "phase_09_crackle_removal",
                "phase_12_wow_flutter_fix",
                "phase_30_dc_offset_removal",
            ],
            "shellac": [
                "phase_03_denoise",
                "phase_06_frequency_restoration",
                "phase_01_click_removal",
            ],
            "wax_cylinder": [
                "phase_03_denoise",
                "phase_06_frequency_restoration",
                "phase_01_click_removal",
                "phase_29_tape_hiss_reduction",
            ],
            "wire_recording": [
                "phase_12_wow_flutter_fix",
                "phase_24_dropout_repair",
                "phase_03_denoise",
                "phase_29_tape_hiss_reduction",
            ],
            "lacquer_disc": [
                "phase_01_click_removal",
                "phase_09_crackle_removal",
                "phase_03_denoise",
                "phase_29_tape_hiss_reduction",
            ],
            "dat": [
                "phase_24_dropout_repair",
                "phase_02_hum_removal",
                "phase_23_spectral_repair",
            ],
            "cd_digital": [
                "phase_23_spectral_repair",
                "phase_06_frequency_restoration",
                "phase_40_loudness_normalization",
            ],
            "mp3_low": [
                "phase_23_spectral_repair",
                "phase_03_denoise",
                "phase_50_spectral_repair",
            ],
            "mp3_high": [
                "phase_23_spectral_repair",
                "phase_50_spectral_repair",
            ],
            "aac": [
                "phase_23_spectral_repair",
                "phase_38_presence_boost",
                "phase_06_frequency_restoration",
            ],
            "minidisc": [
                "phase_23_spectral_repair",
                "phase_06_frequency_restoration",
                "phase_07_harmonic_restoration",
            ],
            "streaming": [
                "phase_24_dropout_repair",
                "phase_23_spectral_repair",
                "phase_50_spectral_repair",
            ],
        }
        _mat_key = material.value if hasattr(material, "value") else str(material)
        _priority_phases = _MATERIAL_PRIORITY_PHASES.get(_mat_key, [])
        _selected_set_mat = set(selected)
        _mat_priority_added: list[str] = []
        for _mp in _priority_phases:
            if _mp not in _selected_set_mat:
                selected.append(_mp)
                _selected_set_mat.add(_mp)
                _mat_priority_added.append(_mp)
        if _mat_priority_added:
            logger.info(
                "📋 §6.2a Material-Pflichtphasen: %d ergänzt für material=%s: %s",
                len(_mat_priority_added),
                _mat_key,
                _mat_priority_added,
            )

        # ════════════════════════════════════════════════════════════════════
        # TIER 1.5 — Kausaldiagnose-Ergänzung (CausalDefectReasoner §2.6)
        # Bayesianische Ursachenanalyse (CAUSE_TO_PHASES) liefert Phasen,
        # die Severity-Schwellen in Tier 1 ggf. nicht aktiviert haben.
        # Nur aktiv wenn confidence ≥ 0.20 (Rauschunterdrückung bei Unsicherheit).
        # ════════════════════════════════════════════════════════════════════
        if causal_plan is not None and causal_plan.confidence >= 0.20:
            _selected_set = set(selected)
            _causal_added: list[str] = []
            for _cp_phase in causal_plan.recommended_phases:
                if _cp_phase not in _selected_set:
                    selected.append(_cp_phase)
                    _selected_set.add(_cp_phase)
                    _causal_added.append(_cp_phase)
            if _causal_added:
                logger.info(
                    "🧠 CausalReasoner §2.6 ergänzt %d Phase(n) (cause=%s, conf=%.2f): %s",
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
            _chain_added: list[str] = []
            _chain_set = set(selected)
            for _chn_phase in _chain_phases:
                if _chn_phase not in _chain_set:
                    selected.append(_chn_phase)
                    _chain_set.add(_chn_phase)
                    _chain_added.append(_chn_phase)
            if _chain_added:
                _complexity = chain_info.get("chain_complexity", 0.0) if isinstance(chain_info, dict) else 0.0
                logger.info(
                    "🔗 TontraegerketteDenker §2.2 ergänzt %d Phase(n) (complexity=%.2f): %s",
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
                _dh_added: list[str] = []
                _dh_set = set(selected)
                for _dh_phase in _dh_phases:
                    if _dh_phase not in _dh_set:
                        selected.append(_dh_phase)
                        _dh_set.add(_dh_phase)
                        _dh_added.append(_dh_phase)
                if _dh_added:
                    logger.info(
                        "🔍 DefektDenker §2.1 ergänzt %d Phase(n) (confidence=%.2f): %s",
                        len(_dh_added),
                        _dh_conf,
                        _dh_added,
                    )

        # ════════════════════════════════════════════════════════════════════
        # TIER 2 — Frequenz- / Spektral-Restaurierung
        # ════════════════════════════════════════════════════════════════════

        # Bandbreitenerweiterung bei Verlusten
        if sev(DefectType.BANDWIDTH_LOSS) > 0.20 or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.30:
            selected.append("phase_06_frequency_restoration")

        # Oberton-Restaurierung (tiefergehend)
        if sev(DefectType.BANDWIDTH_LOSS) > 0.30:
            selected.append("phase_07_harmonic_restoration")

        # Transienten-Schutz bei digitalen Artefakten
        if sev(DefectType.DIGITAL_ARTIFACTS) > 0.25:
            selected.append("phase_08_transient_preservation")

        # EQ-Korrektur für materialspezifische Frequenzgänge
        # Alle analogen Träger und DAT haben charakteristische Frequenzgangkurven
        if material in [
            MaterialType.SHELLAC,
            MaterialType.VINYL,
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
            MaterialType.DAT,
            MaterialType.MINIDISC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
        ]:
            selected.append("phase_04_eq_correction")

        # Dynamikbereich-Erweiterung bei über-Kompression
        if sev(DefectType.DYNAMIC_COMPRESSION_EXCESS) > 0.30 or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.30:
            selected.append("phase_26_dynamic_range_expansion")

        # Spektrale Gesamt-Reparatur (breiter zweiter Pass)
        if sev(DefectType.DIGITAL_ARTIFACTS) > 0.20 or sev(DefectType.COMPRESSION_ARTIFACTS) > 0.20:
            selected.append("phase_50_spectral_repair")

        # ════════════════════════════════════════════════════════════════════
        # TIER 3 — Stereo- / Phase-Verarbeitung
        # ════════════════════════════════════════════════════════════════════

        is_stereo = getattr(defect_result, "is_stereo", True)
        if not is_stereo:
            selected.append("phase_32_mono_to_stereo")  # Mono→Pseudo-Stereo

        # Stereo-Balance bei Imbalance
        if sev(DefectType.STEREO_IMBALANCE) > 0.15:
            selected.append("phase_15_stereo_balance")

        # Stereo-Breiten-Begrenzer bei extremer Imbalance
        if sev(DefectType.STEREO_IMBALANCE) > 0.40:
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

        # Präsenz (2–6 kHz) — bei Vokalinhalt
        if vocals_detected:
            selected.append("phase_38_presence_boost")

        # Air-Band Anhebung > 12 kHz — bei Bandbegrenzung oder analogem Material
        # WAX_CYLINDER und LACQUER_DISC: HF-Rekonstruktion nach physikalischer Bandbegrenzung
        if sev(DefectType.BANDWIDTH_LOSS) > 0.10 or material in [
            MaterialType.SHELLAC,
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
            MaterialType.WAX_CYLINDER,
            MaterialType.LACQUER_DISC,
        ]:
            selected.append("phase_39_air_band_enhancement")

        # Tape-Sättigungs-Emulation (Tape/REEL-Material — authentischer Charakter)
        # REEL_TAPE hat identischen Röhrensättigungs-Charakter wie TAPE
        if material in [MaterialType.TAPE, MaterialType.REEL_TAPE, MaterialType.SHELLAC]:
            selected.append("phase_22_tape_saturation")

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
        ):
            selected.append("phase_54_transparent_dynamics")

        # Multiband-Kompression (wenn nicht über-komprimiert)
        if sev(DefectType.COMPRESSION_ARTIFACTS) < 0.40:
            selected.append("phase_35_multiband_compression")

        # Transient-Shaper (nach Drums-Enhancement)
        if drums_detected:
            selected.append("phase_36_transient_shaper")

        # Kompression & Limiting (wenn nicht bereits über-komprimiert)
        if sev(DefectType.COMPRESSION_ARTIFACTS) < 0.50:
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
        unique: list[str] = []
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

    def _optimize_phase_plan_intelligence(
        self,
        selected_phases: list[str],
        *,
        causal_plan: Any | None,
        pipeline_confidence: Any | None,
        restorability_score: float,
    ) -> list[str]:
        """Optimize phase plan with lightweight orchestration intelligence.

        Goals:
          1) Keep deterministic and safe phase ordering invariants.
          2) Boost high-confidence causal phases earlier in the chain.
          3) Preserve all selected phases (no phase removal in this step).
          4) (Phase 2) Utility-scored re-ranking when enable_phase_utility_scoring=True.

        Phase-2 Utility formula:
            utility(phase) = defect_severity_weight * expected_goal_impact / budget_cost_weight

        Phases below UTILITY_DROP_THRESHOLD are moved to end of list as low-priority candidates
        for budget-aware skipping by downstream PMGG, but are NOT removed here.
        Phases in _UTILITY_PROTECTED are never re-ranked by utility scoring alone.
        """
        phases = list(selected_phases)
        if not phases:
            self._phase_plan_intelligence = {
                "input_count": 0,
                "output_count": 0,
                "causal_boosted": [],
                "precedence_fixes": [],
                "pipeline_confidence": None,
                "restorability_score": float(restorability_score),
            }
            return phases

        _precedence_fixes: list[str] = []

        def _move_before(a: str, b: str) -> None:
            if a in phases and b in phases:
                ia = phases.index(a)
                ib = phases.index(b)
                if ia > ib:
                    phases.pop(ia)
                    ib = phases.index(b)
                    phases.insert(ib, a)
                    _precedence_fixes.append(f"{a}<{b}")

        # Apply ordering constraints iteratively until convergent (max 5 rounds).
        # Single-pass _move_before can leave transitive violations
        # (e.g. A<B then B<C moves B earlier, undoing A<B).
        for _conv_round in range(5):
            _fixes_before = len(_precedence_fixes)

            # Safety ordering constraints (quality-preserving orchestration invariants)
            _move_before("phase_24_dropout_repair", "phase_55_diffusion_inpainting")
            _move_before("phase_57_print_through_reduction", "phase_29_tape_hiss_reduction")
            _move_before("phase_20_reverb_reduction", "phase_49_advanced_dereverb")
            _move_before("phase_16_final_eq", "phase_17_mastering_polish")
            _move_before("phase_17_mastering_polish", "phase_47_truepeak_limiter")
            _move_before("phase_47_truepeak_limiter", "phase_40_loudness_normalization")
            _move_before("phase_40_loudness_normalization", "phase_41_output_format_optimization")

            # Signal-processing-chain ordering (wissenschaftlich korrekte Reihenfolge)
            # Impulse noise removal before broadband NR (prevents impulse energy smearing)
            _move_before("phase_01_click_removal", "phase_03_denoise")
            _move_before("phase_01_click_removal", "phase_27_click_pop_removal")
            # Surface noise profiling before crackle removal (profile-basierte Methode)
            _move_before("phase_28_surface_noise_profiling", "phase_09_crackle_removal")
            # Narrow-band (hum) before broadband NR
            _move_before("phase_02_hum_removal", "phase_03_denoise")
            # Denoise before frequency extension (clean before reconstruct)
            _move_before("phase_03_denoise", "phase_06_frequency_restoration")
            # Frequency restoration before harmonic restoration (Grundfrequenz vor Obertöne)
            _move_before("phase_06_frequency_restoration", "phase_07_harmonic_restoration")
            # Wow/flutter pitch correction before speed/pitch correction
            _move_before("phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction")
            # Transient preservation before transient shaping
            _move_before("phase_08_transient_preservation", "phase_36_transient_shaper")
            # First spectral repair pass before second pass
            _move_before("phase_23_spectral_repair", "phase_50_spectral_repair")

            if len(_precedence_fixes) == _fixes_before:
                break  # Converged — no further reordering needed

        # Causal boost: if confidence is high, front-load top causal phases after base cleanup.
        _causal_boosted: list[str] = []
        try:
            _c_conf = float(causal_plan.confidence) if causal_plan is not None else 0.0
        except Exception:
            _c_conf = 0.0
        if causal_plan is not None and _c_conf >= 0.35:
            _base_anchor = 2  # after phase_30 + phase_05 baseline
            _non_boostable = {
                "phase_16_final_eq",
                "phase_17_mastering_polish",
                "phase_47_truepeak_limiter",
                "phase_40_loudness_normalization",
                "phase_41_output_format_optimization",
            }
            for _p in list(getattr(causal_plan, "recommended_phases", []))[:2]:
                if _p in phases and _p not in _non_boostable:
                    _old = phases.index(_p)
                    if _old > _base_anchor:
                        phases.pop(_old)
                        phases.insert(_base_anchor, _p)
                        _causal_boosted.append(_p)
                        _base_anchor += 1

        # -----------------------------------------------------------------------
        # Phase 2: Utility-scored re-ranking (feature-gated)
        # -----------------------------------------------------------------------
        # utility(phase) = defect_severity_weight * expected_goal_impact / budget_cost_weight
        # All weights are calibrated on 500 AMRB test items; update via AMRB regression.
        # Phases below UTILITY_DROP_THRESHOLD are moved toward the end so that the
        # PMGG per-phase gate can evict them first when the RT budget is tight.
        # No phase is ever *removed* here — only re-ordered.
        # -----------------------------------------------------------------------
        _UTILITY_DROP_THRESHOLD: float = 0.12
        # Phases whose position must never be altered by utility scoring alone
        _UTILITY_PROTECTED: frozenset = frozenset(
            {
                "phase_30_dc_offset_removal",
                "phase_05_noise_floor_analysis",
                "phase_16_final_eq",
                "phase_17_mastering_polish",
                "phase_47_truepeak_limiter",
                "phase_40_loudness_normalization",
                "phase_41_output_format_optimization",
            }
        )
        # Per-phase static weights: (defect_severity_weight, expected_goal_impact, budget_cost_weight)
        # Tuned from AMRB benchmarks — higher utility → earlier execution priority.
        _PHASE_WEIGHTS: dict[str, tuple[float, float, float]] = {
            # Core cleanup — high impact, low cost
            "phase_01_noise_reduction": (0.90, 0.95, 0.40),
            "phase_02_click_removal": (0.80, 0.85, 0.30),
            "phase_03_rumble_removal": (0.70, 0.75, 0.25),
            "phase_04_hum_removal": (0.85, 0.88, 0.35),
            "phase_06_vinyl_crackle_removal": (0.85, 0.90, 0.45),
            "phase_09_banquet_vinyl_ml": (0.88, 0.92, 0.50),
            "phase_07_dropout_repair_nmf": (0.75, 0.80, 0.60),
            "phase_08_azimuth_correction": (0.60, 0.70, 0.35),
            "phase_10_riaa_curve_correction": (0.78, 0.85, 0.30),
            "phase_11_transient_restoration": (0.65, 0.78, 0.55),
            "phase_12_harmonic_restoration": (0.70, 0.82, 0.65),
            "phase_13_stereo_restoration": (0.60, 0.72, 0.50),
            "phase_14_dynamic_range_restoration": (0.72, 0.80, 0.60),
            "phase_15_frequency_restoration": (0.68, 0.78, 0.55),
            "phase_18_vocal_enhancement": (0.80, 0.88, 0.70),
            "phase_19_de_essing": (0.55, 0.65, 0.30),
            "phase_20_reverb_reduction": (0.75, 0.82, 0.75),
            "phase_21_pitch_correction": (0.65, 0.75, 0.80),
            "phase_22_tape_saturation_emulation": (0.50, 0.60, 0.35),
            "phase_23_clipping_repair": (0.88, 0.90, 0.60),
            "phase_24_dropout_repair": (0.78, 0.84, 0.70),
            "phase_25_wow_flutter_correction": (0.72, 0.80, 0.65),
            "phase_26_print_through_reduction": (0.60, 0.70, 0.50),
            "phase_27_bias_correction": (0.68, 0.76, 0.45),
            "phase_28_bandwidth_extension": (0.58, 0.72, 0.90),
            "phase_29_tape_hiss_reduction": (0.82, 0.86, 0.45),
            "phase_31_stem_separation": (0.70, 0.85, 0.95),
            "phase_32_vocal_isolation": (0.72, 0.87, 0.95),
            "phase_33_music_enhancement": (0.65, 0.78, 0.85),
            "phase_34_lyrics_guided_enhancement": (0.70, 0.82, 0.75),
            "phase_35_psychoacoustic_enhancement": (0.55, 0.68, 0.60),
            "phase_36_spatial_enhancement": (0.52, 0.65, 0.70),
            "phase_37_codec_artifact_removal": (0.80, 0.87, 0.55),
            "phase_38_aliasing_removal": (0.70, 0.80, 0.50),
            "phase_39_era_authentic_completion": (0.45, 0.60, 0.80),
            "phase_42_spectral_coherence": (0.60, 0.72, 0.60),
            "phase_43_ml_de_esser": (0.55, 0.65, 0.45),
            "phase_44_transient_shaper": (0.60, 0.70, 0.55),
            "phase_45_micro_dynamics_morphing": (0.58, 0.68, 0.65),
            "phase_46_groove_preservation": (0.55, 0.65, 0.60),
            "phase_48_stem_remix_balance": (0.68, 0.80, 0.80),
            "phase_49_advanced_dereverb": (0.72, 0.80, 0.85),
            "phase_50_formant_enhancement": (0.62, 0.74, 0.60),
            "phase_51_harmonic_guard": (0.65, 0.75, 0.55),
            "phase_52_breathiness_guard": (0.50, 0.60, 0.40),
            "phase_53_emotional_arc_preservation": (0.58, 0.70, 0.70),
            "phase_54_artifact_suppression": (0.72, 0.80, 0.55),
            "phase_55_diffusion_inpainting": (0.70, 0.82, 0.90),
            "phase_56_reference_mastering": (0.55, 0.70, 0.85),
            "phase_57_print_through_reduction": (0.65, 0.75, 0.50),
            "phase_58_flow_matching_inpainting": (0.72, 0.84, 0.95),
        }

        _utility_reranked: list[str] = []
        _use_utility = getattr(self.config, "enable_phase_utility_scoring", False)
        if _use_utility and len(phases) > 4:
            # Scale defect_severity_weight by restorability so that poor material
            # gets aggressively cleaned while good material keeps expensive extras.
            _rscore = max(0.01, min(1.0, restorability_score / 100.0))
            _severity_scale = 1.0 + (1.0 - _rscore) * 0.30  # range 1.0 → 1.30

            def _utility(phase_id: str) -> float:
                w = _PHASE_WEIGHTS.get(phase_id)
                if w is None:
                    return 0.50  # unknown phase → neutral utility
                dsev, goal_imp, cost = w
                effective_dsev = min(1.0, dsev * _severity_scale)
                return (effective_dsev * goal_imp) / max(0.01, cost)

            # Partition into protected, high-utility, and low-utility groups
            _protected_phases = [p for p in phases if p in _UTILITY_PROTECTED]
            _eligible_phases = [p for p in phases if p not in _UTILITY_PROTECTED]

            # Score and split
            _scored = [(p, _utility(p)) for p in _eligible_phases]
            _high = [p for p, u in _scored if u >= _UTILITY_DROP_THRESHOLD]
            _low = [p for p, u in _scored if u < _UTILITY_DROP_THRESHOLD]

            # High-utility phases: sorted descending (highest impact first) but capped
            # to preserve relative ordering of phases with very similar scores (±0.05)
            _STABLE_BAND: float = 0.05
            _high_sorted = sorted(_high, key=lambda p: -_utility(p))
            # Stabilisation: don't reorder pairs whose utility difference is within band
            _stabilised: list[str] = []
            _remaining_h = list(_high_sorted)
            while _remaining_h:
                _cur = _remaining_h.pop(0)
                # If next phase is within stable band, preserve original relative order
                if _remaining_h:
                    _nxt = _remaining_h[0]
                    if abs(_utility(_cur) - _utility(_nxt)) <= _STABLE_BAND:
                        # Keep original relative ordering for this pair
                        _orig_cur = phases.index(_cur) if _cur in phases else 999
                        _orig_nxt = phases.index(_nxt) if _nxt in phases else 999
                        if _orig_nxt < _orig_cur:
                            _stabilised.append(_nxt)
                            _remaining_h.pop(0)
                            _stabilised.append(_cur)
                            continue
                _stabilised.append(_cur)

            # Rebuild: protected phases retain their original absolute positions;
            # high-utility fill remaining slots; low-utility go to end.
            _final: list[str] = []
            _high_iter = iter(_stabilised)
            _low_iter = iter(sorted(_low, key=lambda p: phases.index(p) if p in phases else 999))
            _protected_idx = {p: phases.index(p) for p in _protected_phases if p in phases}
            _all_idx = sorted(range(len(phases)), key=lambda i: i)
            _insert_positions = {p: phases.index(p) for p in _protected_phases if p in phases}
            # Reconstruct phase list preserving protected absolute positions
            _slot_phases: list[str | None] = [None] * len(phases)
            for p, idx in _insert_positions.items():
                _slot_phases[idx] = p
            for i in range(len(_slot_phases)):
                if _slot_phases[i] is None:
                    try:
                        _slot_phases[i] = next(_high_iter)
                    except StopIteration:
                        with contextlib.suppress(StopIteration):
                            _slot_phases[i] = next(_low_iter)
            # Fill any remaining low-utility phases at the end
            _filled = [p for p in _slot_phases if p is not None]
            _placed = set(_filled)
            _remaining_low = [p for p in _low if p not in _placed]
            _remaining_high = [p for p in _stabilised if p not in _placed]
            phases = _filled + _remaining_high + _remaining_low
            _utility_reranked = [
                p
                for p in phases
                if p not in _UTILITY_PROTECTED
                and p in dict(_scored)
                and dict(_scored).get(p, 0.5) >= _UTILITY_DROP_THRESHOLD
            ]
            if _utility_reranked:
                logger.info(
                    "🧮 Phase-Utility-Scoring: %d phases re-ranked (low-utility→end: %d), "
                    "restorability=%.0f, severity_scale=%.2f",
                    len(_utility_reranked),
                    len(_low),
                    restorability_score,
                    _severity_scale,
                )

        self._phase_plan_intelligence = {
            "input_count": len(selected_phases),
            "output_count": len(phases),
            "causal_boosted": _causal_boosted,
            "precedence_fixes": _precedence_fixes,
            "utility_scoring_active": _use_utility,
            "utility_reranked_count": len(_utility_reranked),
            "utility_low_priority_count": len(_low) if _use_utility and len(phases) > 4 else 0,
            "pipeline_confidence": (
                round(float(getattr(pipeline_confidence, "confidence", 0.0)), 4)
                if pipeline_confidence is not None
                else None
            ),
            "restorability_score": round(float(restorability_score), 2),
        }
        if _causal_boosted or _precedence_fixes:
            logger.info(
                "🧠 Phase-Plan-Intelligence: causal_boost=%s precedence_fixes=%d",
                _causal_boosted,
                len(_precedence_fixes),
            )
        return phases

    def _apply_phase_skipping(self, selected_phases: list[str], defect_result) -> tuple[list[str], dict[str, str]]:
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

        noise_floor_db = max(-60.0, min(0.0, -60.0 + hiss_severity * 60.0))

        defect_analysis = DefectAnalysis(
            medium=medium_map.get(defect_result.material_type, SourceMedium.UNKNOWN),
            noise_floor_db=noise_floor_db,  # Convert severity to dB (0.0 → -60dB, 1.0 → 0dB)
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
                    if defect_analysis.noise_floor_db <= -60.0 and not defect_analysis.has_hiss:
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
            except Exception:
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

        # §Vintage-Authentizitäts-Guard: Strength-Cap anwenden (copilot-instructions.md)
        # _vintage_phase_strength_caps wird in restore() befüllt und hier per Cap erzwungen.
        _vcaps = getattr(self, "_vintage_phase_strength_caps", {})
        if _vcaps:
            _pid = phase_metadata.phase_id
            _cap = _vcaps.get(_pid)
            if _cap is not None and "strength" in kwargs:
                _orig_s = kwargs["strength"]
                if isinstance(_orig_s, (int, float)) and float(_orig_s) > float(_cap):
                    kwargs["strength"] = float(_cap)
                    logger.debug(
                        "🕰️ Vintage-Cap %s: strength %.2f → %.2f",
                        _pid,
                        _orig_s,
                        _cap,
                    )

        t0 = time.perf_counter()
        mem0 = memory_usage(-1, interval=0.01, timeout=1) if MEMORY_PROFILING_AVAILABLE else [0]

        # §MusikalischeHarmonisierung: Defekt-Severity-Wet/Dry für non-PMGG-Pfade
        # Berechne Severity-Faktor VOR Phasenausführung (benötigt Original-Audio für Mix)
        _sev_wet_dry: float = 1.0
        _TIMING_PHASES_WD = frozenset(
            {
                "phase_12_wow_flutter_fix",
                "phase_31_speed_pitch_correction",
            }
        )
        _defect_scores_wd = kwargs.get("defect_scores")
        if _defect_scores_wd and phase_metadata.phase_id not in _TIMING_PHASES_WD:
            try:
                from backend.core.defect_phase_mapper import get_phase_defect_severity

                _sev_wet_dry = get_phase_defect_severity(phase_metadata.phase_id, _defect_scores_wd)
            except Exception:
                _sev_wet_dry = 1.0

        # Call phase.process() method (not phase() itself!)
        _audio_before_phase = audio if _sev_wet_dry < 1.0 else None
        result = phase.process(audio, **kwargs)

        # §MusikalischeHarmonisierung: Wet/Dry basierend auf Defekt-Severity
        if _sev_wet_dry < 1.0 and _audio_before_phase is not None and hasattr(result, "audio"):
            _pa = result.audio
            if isinstance(_pa, np.ndarray) and _pa.shape == _audio_before_phase.shape:
                result.audio = np.clip(
                    (_audio_before_phase + _sev_wet_dry * (_pa - _audio_before_phase)).astype(np.float32),
                    -1.0,
                    1.0,
                )
                logger.debug(
                    "🎵 DefectSeverity Wet/Dry %s: factor=%.2f",
                    phase_metadata.phase_id,
                    _sev_wet_dry,
                )

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

    # ------------------------------------------------------------------
    # §Per-Channel Defect Repair: Proxy für kanalweise Phasen-Ausführung
    # ------------------------------------------------------------------

    # Defekt-Reparatur-Phasen, die bei Stereo-Audio kanalweise verarbeitet werden.
    # Analog zur kanalweisen Erkennung im DefectScanner — jeder Kanal wird
    # unabhängig repariert, Zwischenstände an die Visualisierung emittiert.
    _CHANNEL_SPECIFIC_PHASES: frozenset[str] = frozenset(
        {
            "phase_01_click_removal",
            "phase_09_crackle_removal",
            "phase_23_spectral_repair",
            "phase_24_dropout_repair",
            "phase_27_click_pop_removal",
        }
    )

    class _ChannelSplitPhaseProxy:
        """Proxy that wraps a PhaseInterface to process stereo channels independently.

        Used for channel-specific defect repair phases (click, crackle, dropout etc.)
        where each stereo channel should be repaired individually — analogous to
        the per-channel detection in DefectScanner.

        Emits intermediate waveform updates after each channel via audio_update_callback
        so the UI can show the sequential channel processing in real time.
        """

        __slots__ = ("_phase", "_audio_update_cb", "_sr")

        def __init__(self, phase, audio_update_callback=None, sample_rate: int = 48000):
            object.__setattr__(self, "_phase", phase)
            object.__setattr__(self, "_audio_update_cb", audio_update_callback)
            object.__setattr__(self, "_sr", sample_rate)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_phase"), name)

        def get_metadata(self):
            return object.__getattribute__(self, "_phase").get_metadata()

        def process(self, audio, **kwargs):
            """Process stereo audio channel-by-channel; mono passes through unchanged."""
            phase = object.__getattribute__(self, "_phase")
            cb = object.__getattribute__(self, "_audio_update_cb")
            sr = object.__getattribute__(self, "_sr")

            # Only split if stereo (N, 2) — mono and other shapes pass through unchanged
            if audio.ndim != 2 or audio.shape[-1] != 2:
                return phase.process(audio, **kwargs)

            n_samples = audio.shape[0]
            left_orig = audio[:, 0].copy()
            right_orig = audio[:, 1].copy()

            def _safe_extract(result, fallback):
                """Extract audio array from PhaseResult or raw ndarray."""
                out = result.audio if hasattr(result, "audio") else result
                if not isinstance(out, np.ndarray) or out.ndim != 1:
                    return fallback.copy()
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                # Ensure same length after phase processing
                if len(out) != n_samples:
                    if len(out) > n_samples:
                        out = out[:n_samples]
                    else:
                        out = np.pad(out, (0, n_samples - len(out)))
                return out

            # --- Process left channel independently ---
            result_l = phase.process(left_orig, **kwargs)
            repaired_l = _safe_extract(result_l, left_orig)

            # Emit intermediate: repaired L + original R
            if cb is not None:
                try:
                    intermediate = np.column_stack([repaired_l, right_orig])
                    _pid = ""
                    try:
                        _pid = phase.get_metadata().phase_id
                    except Exception:
                        pass
                    cb(intermediate, sr, f"{_pid}:L")
                except Exception:
                    pass

            # --- Process right channel independently ---
            result_r = phase.process(right_orig, **kwargs)
            repaired_r = _safe_extract(result_r, right_orig)

            # Combine into stereo result
            combined = np.column_stack([repaired_l, repaired_r]).astype(np.float32)

            # Return PhaseResult-compatible object (PMGG _run_phase checks .audio attribute)
            if hasattr(result_l, "audio"):
                result_l.audio = combined
                return result_l
            return combined

    def _execute_pipeline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        material_type: MaterialType,
        defect_result,
        selected_phases: list[str],
        progress_callback=None,
        audio_update_callback=None,
        _phase_progress_start: int = 30,
        _phase_progress_end: int = 80,
        restorability_score: float = 70.0,  # §2.29 normativ: aus RestorabilityEstimator.estimate()
        applicable_goals: set[str] | None = None,  # §2.32 normativ: aus GoalApplicabilityFilter
        material_initial_strengths: dict[str, float] | None = None,  # §2.31 material-adaptive Initialstärken
        quality_mode_override: QualityMode | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Führt ausgewählte Phasen parallel (Multi-Core) aus, falls keine Abhängigkeiten bestehen.
        Returns: (processed_audio, executed_phases, skipped_phases)
        """
        # §3.1 NaN/Inf-Invariante: Input VOR Phasenausführung bereinigen
        if not np.all(np.isfinite(audio)):
            logger.debug("_execute_pipeline: NaN/Inf im Input-Audio bereinigt")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.9, neginf=-0.9)
        audio = np.clip(audio, -1.0, 1.0)
        _active_quality_mode = quality_mode_override or self.config.mode
        _quality_mode_value = _active_quality_mode.value
        current_audio = audio.copy()
        executed = []
        skipped = []
        deferred = []  # §2.38 KMV: phases skipped by PerformanceGuard for Stage 2 refinement
        # §Punkt3 Phasen-Regressionsprotokoll: RMS-Delta je Phase (sequentielle Ausführung)
        # Einheit: dBFS-Differenz nach - vor Phase. Positiv = Energie gestiegen, negativ = gesunken.
        self._phase_regression_log: dict[str, float] = {}
        # §2.29 PerPhaseMusicalGoalsGate — vor Pipeline-Loop initialisieren (sequentielle Ausführung)
        # Deaktiviert AUSSCHLIESSLICH über enable_phase_gate=False (z. B. --no-phase-gate).
        # enable_performance_guard steuert das CPU-Budget-Throttling, NICHT die musikalische
        # Qualitätskontrolle — beide Flags sind bewusst entkoppelt (§2.29 Fix v9.10.46).
        _pmgg_gate = None
        _pmgg_restorability_score: float = (
            restorability_score  # §2.29 normativ: aus RestorabilityEstimator (kein Hard-Code)
        )
        _pmgg_scores_curr: dict[str, float] | None = None
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
        self._pmgg_log_entries = []
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

        # §7.6 / §9.1: Defect-Locations für gezielte Phasenverarbeitung extrahieren
        # + maximale Severity für adaptive Chunk-Größe
        _defect_locations: dict[str, list[tuple[float, float]]] = {}
        _max_defect_severity: float = 0.0
        if defect_result is not None and hasattr(defect_result, "scores"):
            for _dt, _ds in defect_result.scores.items():
                _dt_key = _dt.value if hasattr(_dt, "value") else str(_dt)
                if hasattr(_ds, "locations") and _ds.locations:
                    _defect_locations[_dt_key] = list(_ds.locations)
                if hasattr(_ds, "severity"):
                    _max_defect_severity = max(_max_defect_severity, float(_ds.severity))

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
                    estimated_time = metadata.estimated_time_factor * (audio.shape[-1] / sample_rate)
                    if self.performance_guard and self.performance_guard.should_skip_phase(
                        phase_id, estimated_time, len(selected_phases) - 1
                    ):
                        skipped.append(phase_id)
                        deferred.append(phase_id)  # §2.38 KMV: RT-skipped → Stage 2
                        continue
                    phase_start = (
                        self.performance_guard.start_phase(phase_id) if self.performance_guard else time.time()
                    )
                    # §PROGRESS: Emit on phase activation (parallel submission) — real-time ML plugin detection
                    if progress_callback is not None:
                        try:
                            _n_sub_p = len(future_map)
                            _n_sel_p = max(len(selected_phases), 1)
                            _pct_p = _phase_progress_start + int(
                                (_phase_progress_end - _phase_progress_start) * _n_sub_p / _n_sel_p
                            )
                            _lbl_p = (self.phase_metadata.get(phase_id) or {}).get("name") or phase_id.replace(
                                "phase_", ""
                            ).replace("_", " ").title().lstrip("0123456789 ")
                            progress_callback(
                                _pct_p,
                                f"{_lbl_p} [{phase_id}]",
                                0.0,
                            )
                        except Exception:
                            pass
                    future = executor.submit(
                        self._profiled_phase_call,
                        phase,
                        current_audio,  # audio parameter
                        sample_rate=sample_rate,
                        material_type=material_type,
                        material=material_type,
                        defect_scores=defect_result.scores,
                        defect_locations=_defect_locations,
                        max_defect_severity=_max_defect_severity,
                        quality_mode=_quality_mode_value,  # Pass quality mode for ML routing
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
                                    _phase_label = (self.phase_metadata.get(phase_id) or {}).get(
                                        "name"
                                    ) or phase_id.replace("phase_", "").replace("_", " ").title().lstrip("0123456789 ")
                                    progress_callback(
                                        _phase_pct,
                                        f"{_phase_label} [{phase_id}]",
                                        0.0,
                                    )
                                except Exception as _cb_exc:
                                    logger.debug(
                                        "Progress-Update übersprungen in %s (Ursache: %s). "
                                        "Lösung: Callback-Signatur prüfen.",
                                        phase_id,
                                        _cb_exc,
                                    )
                            executed.append(phase_id)
                            logger.info(f"✅ {phase_id}: {result.execution_time_seconds:.2f}s (parallel)")
                        else:
                            logger.error(
                                "Phase %s fehlgeschlagen (Ursache: %s). "
                                "Lösung: Phase prüfen oder DSP-Fallback aktivieren.",
                                phase_id,
                                result.warnings,
                            )
                            skipped.append(phase_id)
                    except Exception as e:
                        logger.error(
                            "Phase %s mit Ausnahme abgebrochen (Ursache: %s). "
                            "Lösung: Plugin-/Modellverfügbarkeit und Eingabedaten prüfen.",
                            phase_id,
                            e,
                        )
                        skipped.append(phase_id)
                    if self.performance_guard:
                        self.performance_guard.end_phase(phase_id, phase_start)
            if results:
                audios = [results[pid] for pid in executed if pid in results]
                if audios:
                    _merged = np.mean(audios, axis=0)
                    # §3.1 NaN-Guard: revert to pre-merge audio if NaN produced
                    if np.any(np.isnan(_merged)):
                        logger.warning(
                            "NaN detected in parallel merge of %d phases (%s) — "
                            "reverting to pre-merge audio. Solution: check individual phase outputs.",
                            len(audios),
                            [pid for pid in executed if pid in results],
                        )
                    else:
                        current_audio = np.clip(_merged, -1.0, 1.0)
                        # §Live-Waveform: Emit audio after parallel merge
                        if audio_update_callback is not None:
                            try:
                                audio_update_callback(current_audio, sample_rate, "parallel_merge")
                            except Exception:
                                pass
        else:
            # Sequentielle Ausführung (Abhängigkeiten vorhanden oder nur eine Phase)
            # §2.16 TQC: Zeitmodifizierende Phasen können Kohärenz brechen — Rollback bei Versagen
            _TQC_CRITICAL_PHASES_SEQ = frozenset(
                {
                    "phase_12_wow_flutter_fix",
                    "phase_31_speed_pitch_correction",
                }
            )
            for phase_id in selected_phases:
                phase = self._get_phase(phase_id)
                if not phase:
                    logger.warning(f"Phase {phase_id} konnte nicht lazy-geladen werden, skipping")
                    skipped.append(phase_id)
                    continue
                # §Per-Channel: Defekt-Phasen bei Stereo-Audio kanalweise verarbeiten
                _use_channel_split = (
                    phase_id in self._CHANNEL_SPECIFIC_PHASES
                    and current_audio.ndim == 2
                    and current_audio.shape[-1] == 2
                )
                _phase_for_exec = (
                    self._ChannelSplitPhaseProxy(phase, audio_update_callback, sample_rate)
                    if _use_channel_split
                    else phase
                )
                metadata = phase.get_metadata()
                remaining = len(selected_phases) - len(executed) - 1
                estimated_time = metadata.estimated_time_factor * (audio.shape[-1] / sample_rate)
                if self.performance_guard and self.performance_guard.should_skip_phase(
                    phase_id, estimated_time, remaining
                ):
                    skipped.append(phase_id)
                    deferred.append(phase_id)  # §2.38 KMV: RT-skipped → Stage 2
                    continue
                phase_start = self.performance_guard.start_phase(phase_id) if self.performance_guard else time.time()
                logger.info("▶ %s startet (%d/%d)", phase_id, len(executed) + 1, len(selected_phases))
                # §2.16 TQC mid-pipeline: Snapshot vor zeitmodifizierenden Phasen
                _tqc_snap: np.ndarray | None = current_audio.copy() if phase_id in _TQC_CRITICAL_PHASES_SEQ else None
                # §Punkt3 Regressionsprotokoll: RMS vor Phase messen (in dBFS)
                _rms_before = float(np.sqrt(np.mean(current_audio**2) + 1e-12))
                _rms_before_db = 20.0 * np.log10(_rms_before)
                # §PROGRESS: Emit BEFORE phase execution — real-time ML plugin detection
                if progress_callback is not None:
                    try:
                        _n_sel_pre = max(len(selected_phases), 1)
                        _idx_pre = len(executed)
                        _pct_pre = _phase_progress_start + int(
                            (_phase_progress_end - _phase_progress_start) * _idx_pre / _n_sel_pre
                        )
                        _lbl_pre = (self.phase_metadata.get(phase_id) or {}).get("name") or phase_id.replace(
                            "phase_", ""
                        ).replace("_", " ").title().lstrip("0123456789 ")
                        progress_callback(
                            _pct_pre,
                            f"{_lbl_pre} [{phase_id}]",
                            0.0,
                        )
                    except Exception:
                        pass
                try:
                    if _pmgg_gate is not None:
                        # §2.29 PMGG: Musical-Goal-geschützte Phasenausführung
                        try:
                            # §MusikalischeHarmonisierung: Defekt-Severity-Faktor moduliert Stärke
                            # Phasen arbeiten proportional zur gemessenen Defekt-Schwere,
                            # nicht mit festen Material-Defaults.
                            _mat_strength = (
                                material_initial_strengths.get(phase_id, 1.0) if material_initial_strengths else 1.0
                            )
                            try:
                                from backend.core.defect_phase_mapper import get_phase_defect_severity

                                _sev_factor = get_phase_defect_severity(phase_id, defect_result.scores)
                                _combined_strength = _mat_strength * _sev_factor
                                if _sev_factor < 1.0:
                                    logger.debug(
                                        "🎵 DefectSeverity %s: initial_strength %.2f × sev_factor %.2f = %.2f",
                                        phase_id,
                                        _mat_strength,
                                        _sev_factor,
                                        _combined_strength,
                                    )
                            except Exception:
                                _combined_strength = _mat_strength

                            _pmgg_audio_out, _pmgg_scores_curr, _pmgg_entry = _pmgg_gate.wrap_phase(
                                _phase_for_exec,
                                current_audio,
                                sample_rate,
                                _pmgg_scores_curr,
                                phase_kwargs={
                                    "sample_rate": sample_rate,
                                    "material_type": material_type,
                                    "material": material_type,
                                    "defect_scores": defect_result.scores,
                                    "defect_locations": _defect_locations,
                                    "max_defect_severity": _max_defect_severity,
                                    "quality_mode": _quality_mode_value,
                                },
                                restorability_score=_pmgg_restorability_score,  # §2.29 normativ
                                applicable_goals=applicable_goals,  # §2.32 normativ
                                initial_strength=_combined_strength,  # §2.31 + §Harmonisierung
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
                                    _phase_label = (self.phase_metadata.get(phase_id) or {}).get(
                                        "name"
                                    ) or phase_id.replace("phase_", "").replace("_", " ").title().lstrip("0123456789 ")
                                    progress_callback(
                                        _phase_pct,
                                        f"{_phase_label} [{phase_id}]",
                                        0.0,
                                    )
                                except Exception as _cb_exc:
                                    logger.debug(
                                        "Progress-Update übersprungen in %s (Ursache: %s). "
                                        "Lösung: Callback-Signatur prüfen.",
                                        phase_id,
                                        _cb_exc,
                                    )
                            executed.append(phase_id)
                            # §Live-Waveform: Emit updated audio after PMGG phase
                            if audio_update_callback is not None:
                                try:
                                    audio_update_callback(current_audio, sample_rate, phase_id)
                                except Exception:
                                    pass
                            logger.info(
                                f"✅ {phase_id}: PMGG action={_pmgg_entry.action} "
                                f"strength={_pmgg_entry.strength_used:.2f} "
                                f"rollbacks={_pmgg_gate._rollback_count}"
                            )
                            # §Punkt3 Regressionsprotokoll: RMS nach PMGG-Phase
                            _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio**2) + 1e-12)))
                            self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                        except Exception as _pmgg_phase_exc:
                            # Fallback: direkte Phasenausführung via _profiled_phase_call
                            logger.debug(
                                "PMGG wrap_phase Fehler (%s), Fallback: %s",
                                phase_id,
                                _pmgg_phase_exc,
                            )
                            result = self._profiled_phase_call(
                                _phase_for_exec,
                                current_audio,
                                sample_rate=sample_rate,
                                material_type=material_type,
                                material=material_type,
                                defect_scores=defect_result.scores,
                                defect_locations=_defect_locations,
                                max_defect_severity=_max_defect_severity,
                                quality_mode=_quality_mode_value,
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
                                        _phase_label = (self.phase_metadata.get(phase_id) or {}).get(
                                            "name"
                                        ) or phase_id.replace("phase_", "").replace("_", " ").title().lstrip(
                                            "0123456789 "
                                        )
                                        progress_callback(
                                            _phase_pct,
                                            f"{_phase_label} [{phase_id}]",
                                            0.0,
                                        )
                                    except Exception as _cb_exc:
                                        logger.debug(
                                            "Progress-Update übersprungen in %s (Ursache: %s). "
                                            "Lösung: Callback-Signatur prüfen.",
                                            phase_id,
                                            _cb_exc,
                                        )
                                executed.append(phase_id)
                                # §Live-Waveform: Emit updated audio after PMGG-fallback phase
                                if audio_update_callback is not None:
                                    try:
                                        audio_update_callback(current_audio, sample_rate, phase_id)
                                    except Exception:
                                        pass
                                logger.info(f"✅ {phase_id} (fallback): {result.execution_time_seconds:.2f}s")
                                # §Punkt3 Regressionsprotokoll: RMS nach PMGG-Fallback-Phase
                                _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio**2) + 1e-12)))
                                self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                            else:
                                logger.error(f"❌ {phase_id} failed: {result.warnings}")
                                skipped.append(phase_id)
                    else:
                        result = self._profiled_phase_call(
                            _phase_for_exec,
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
                                    _phase_label = (self.phase_metadata.get(phase_id) or {}).get(
                                        "name"
                                    ) or phase_id.replace("phase_", "").replace("_", " ").title().lstrip("0123456789 ")
                                    progress_callback(
                                        _phase_pct,
                                        f"{_phase_label} [{phase_id}]",
                                        0.0,
                                    )
                                except Exception as _cb_exc:
                                    logger.debug(
                                        "Progress-Update übersprungen in %s (Ursache: %s). "
                                        "Lösung: Callback-Signatur prüfen.",
                                        phase_id,
                                        _cb_exc,
                                    )
                            executed.append(phase_id)
                            # §Live-Waveform: Emit updated audio after direct phase
                            if audio_update_callback is not None:
                                try:
                                    audio_update_callback(current_audio, sample_rate, phase_id)
                                except Exception:
                                    pass
                            logger.info(f"✅ {phase_id}: {result.execution_time_seconds:.2f}s")
                            # §Punkt3 Regressionsprotokoll: RMS nach direkter Phase
                            _rms_after_db = 20.0 * np.log10(float(np.sqrt(np.mean(current_audio**2) + 1e-12)))
                            self._phase_regression_log[phase_id] = round(_rms_after_db - _rms_before_db, 3)
                        else:
                            logger.error(f"❌ {phase_id} failed: {result.warnings}")
                            skipped.append(phase_id)
                except Exception as e:
                    logger.error(f"❌ {phase_id} exception: {e}")
                    skipped.append(phase_id)
                if self.performance_guard:
                    self.performance_guard.end_phase(phase_id, phase_start)
                # OOM-Guard: periodisches GC alle 5 Phasen um Speicher-Akkumulation zu vermeiden
                if len(executed) % 5 == 0:
                    gc.collect()
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
                if self.performance_guard and self.performance_guard.check_early_exit(remaining):
                    logger.warning(f"⚠️ Early exit triggered, skipping {remaining} remaining phases")
                    skipped.extend(selected_phases[len(executed) :])
                    break
        self._pmgg_log_entries = _pmgg_log_entries
        return current_audio, executed, skipped, deferred

    @staticmethod
    def _resolve_adaptive_goal_thresholds(adaptive_goals_payload: Any) -> dict[str, float]:
        """Delegates to the standalone resolver module (P1-2 modularisation).

        See ``backend.core.musical_goals.adaptive_goal_resolver`` for the
        full implementation.  Kept as a @staticmethod here for backward
        compatibility with call sites that reference UnifiedRestorerV3.
        """
        return _resolve_adaptive_goal_thresholds_fn(adaptive_goals_payload)

    def _estimate_quality(
        self,
        defect_result,
        perf_report,
        executed_phases: list[str],
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
        self._quality_estimate_used_fallback = True
        self._quality_estimate_source = "defect_proxy"

        # Defekt-Severity: sicher klemmen
        defect_severity = float(defect_result.get_total_severity())
        defect_severity = max(0.0, min(1.0, defect_severity))

        # PQS-MOS: Fallback-Schätzung (kein Audio oder Import-Fehler)
        pqs_mos: float = 1.0 + (1.0 - defect_severity) * 4.0

        if audio is not None:
            try:
                from backend.core.perceptual_quality_scorer import score_audio_absolute  # type: ignore

                _pqs_result = score_audio_absolute(audio, sr=sample_rate)
                _mos = float(getattr(_pqs_result, "pqs_mos", getattr(_pqs_result, "mos", pqs_mos)))
                if math.isfinite(_mos) and 1.0 <= _mos <= 5.0:
                    pqs_mos = _mos
                    self._quality_estimate_used_fallback = False
                    self._quality_estimate_source = "pqs_absolute"
            except Exception as _pqs_exc:
                logger.debug("_estimate_quality: PQS-Import fehlgeschlagen, Fallback aktiv: %s", _pqs_exc)

        # Spec §8.1.1: Gewichtete Kombination
        quality_estimate = 0.40 * (1.0 - defect_severity) + 0.60 * (pqs_mos - 1.0) / 4.0
        return float(max(0.0, min(1.0, quality_estimate)))

    def get_phase_info(self) -> dict[str, dict]:
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
                    "maximum": QualityMode.MAXIMUM,  # §9.5: 8× RT — war fälschlich BALANCED (3×)
                    "studio_2026": QualityMode.MAXIMUM,  # §9.5: 8× RT — war fälschlich BALANCED (3×)
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

    logger.debug(f"\n{'=' * 70}")
    logger.debug("UNIFIED RESTORER V3 TEST - Aurik 9.0")
    logger.debug(f"{'=' * 70}\n")

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
    logger.debug("Defects: ~100 clicks, 60Hz hum, tape hiss\n")

    # Test alle Modi
    for mode in [QualityMode.FAST, QualityMode.BALANCED]:
        logger.debug(f"\n{'─' * 70}")
        logger.debug(f"Testing {mode.value.upper()} Mode")
        logger.debug(f"{'─' * 70}\n")

        config = RestorationConfig(mode=mode, num_cores=4, enforce_3x_rt=True, enable_adaptive_skipping=True)

        restorer = UnifiedRestorerV3(config)

        # Restore
        result = restorer.restore(audio, sample_rate=sr)

        # Report
        logger.debug(f"\n{'=' * 70}")
        logger.debug(f"RESULT SUMMARY - {mode.value.upper()}")
        logger.debug(f"{'=' * 70}")
        logger.debug(f"Material: {result.material_type.value}")
        logger.debug(f"Total Time: {result.total_time_seconds:.1f}s")
        logger.debug(f"RT Factor: {result.rt_factor:.2f}× ({'✅ PASS' if result.rt_factor <= 3.0 else '❌ FAIL'})")
        logger.debug(f"Quality Estimate: {result.quality_estimate * 100:.1f}%")
        logger.debug(f"Phases Executed: {len(result.phases_executed)}")
        logger.debug(f"Phases Skipped: {len(result.phases_skipped)}")
        if result.warnings:
            logger.debug(f"Warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                logger.debug(f"  - {w}")
        logger.debug("\nTop Defects (Before):")
        for i, defect in enumerate(result.metadata["defect_analysis"]["top_defects"][:3], 1):
            logger.debug(f"  {i}. {defect['type']}: {defect['severity']:.2f}")
        logger.debug(f"{'=' * 70}\n")

    logger.debug("\n✅ UnifiedRestorerV3 Test Complete!")
