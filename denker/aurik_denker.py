"""
denker/aurik_denker.py — AurikDenker: Orchestrator mit 8 Verarbeitungsstufen
==============================================================================

Koordiniert die vollständige Restaurierungs-Pipeline in der kanonischen
Reihenfolge (Spec §2.2):

  1. TontraegerDenker  → Trägermedium-Erkennung
  2. DefektDenker      → Defekt-Analyse + Kausal-Reasoning
  3. StrategieDenker   → 8×RT-Budget-Plan + Timer
  4. _run_rest()-Closure → ReparaturDenker (Preprocessing) →
                           RekonstruktionsDenker (Preprocessing) →
                           RestaurierDenker (UV3-Vollpipeline)
  7. ExzellenzDenker   → Musical Goals + Excellence-Optimierung
  8. VERSA MOS-Gate     → Finale Qualitätsbewertung (§4.4, nicht-referenzbasiert);
                          MOS < 4.0 → 2. ExzellenzDenker-Durchlauf

8×RT-Invariante: rt_factor im Endergebnis ist IMMER ≤ 8.0.

Spec §2.1, §2.2, §9.5 — v9.10.45
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from backend.core.pipeline_health_state import PipelineHealthState, pipeline_health_from_fail_reasons

logger = logging.getLogger(__name__)

_3X_RT_LIMIT: float = 32.0  # RT-Reported max (Spec §9.5 — caps rt_factor in output)
# Mode-abhängige RT-Budgets für RestaurierDenker-Thread (Spec §9.5):
#   BALANCED=32×, QUALITY=32×, MAXIMUM=32×
#   Werte sind auf PerformanceGuard.LIMIT_* ausgerichtet.
_RT_BUDGET_BY_MODE: dict = {
    "balanced": 32.0,
    "restoration": 32.0,  # "Restoration" maps to quality
    "quality": 32.0,
    "studio2026": 32.0,
    "maximum": 32.0,
}
# Cold-Start-Minimum: Erster Lauf lädt ML-Modelle (PANNs 0.7GB, UV3 vollständig, etc.)
# UV3-Init + Klassifikation allein dauert 20–120s je nach Host-System. 1800s = 30 min
# deckt auch langsame HDDs + thermisches Throttling ab. PerformanceGuard begrenzt
# die eigentliche Verarbeitungszeit — dieser Floor gilt ausschließlich für Cold-Start.
_COLDSTART_MIN_SECONDS: float = 1800.0
# Absolutes Gesamtlimit (§9.5): eine Stufe-1-Restaurierung endet spätestens nach 90 min.
# Begründung: 20-min Vinyl (1200s) × 4× DSP-RT = 4800s; 90 min = komfortabler Puffer.
# KMV Stufe 2 (MLRefinementThread) übernimmt danach ohne Zeitlimit.
# Alter Wert: 1800s (30 min) — zu eng für Aufnahmen > 10 min mit schweren Defekten.
_MAX_TOTAL_SECONDS: float = 5400.0  # 90 Minuten
_MIN_AUDIO_SAMPLES: int = 64  # Mindestsignallänge

# Material-adaptive MOS-Mindestziele (§6.2)
_MATERIAL_MOS_TARGETS: dict[str, float] = {
    "wax_cylinder": 3.5,
    "shellac": 3.8,
    "lacquer_disc": 3.7,
    "wire_recording": 3.6,
    "vinyl": 4.0,
    "tape": 4.2,
    "reel_tape": 4.3,
    "dat": 4.5,
    "cd_digital": 4.5,
    "mp3_low": 3.9,
    "mp3_high": 4.5,  # §6.2: wie cd_digital/dat (digitale Hochqualitäts-Quelle)
    "aac": 4.5,  # §6.2: wie cd_digital/dat (digitale Hochqualitäts-Quelle)
    "minidisc": 4.0,
    "streaming": 4.0,
    "unknown": 3.8,
}

_HISTORICAL_OR_FRAGILE_MATERIALS: frozenset[str] = frozenset(
    {
        "wax_cylinder",
        "shellac",
        "lacquer_disc",
        "wire_recording",
        "vinyl",
        "tape",
        "reel_tape",
    }
)

_MODERN_DIGITAL_MATERIALS: frozenset[str] = frozenset(
    {
        "cd_digital",
        "dat",
        "aac",
        "mp3_high",
        "streaming",
    }
)

_HEAVY_DEFECT_HINTS: frozenset[str] = frozenset(
    {
        "dropout",
        "clipping",
        "hum",
        "wow",
        "flutter",
        "crackle",
        "click",
        "print_through",
    }
)


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Load a symbol lazily to avoid module-level circular imports."""
    return getattr(import_module(module_name), symbol_name)


def _normalize_goal_scores(raw_goals: Any) -> dict[str, float]:
    """Coerce heterogeneous goal mappings to the canonical dict[str, float] shape."""
    if not isinstance(raw_goals, dict):
        return {}

    normalized: dict[str, float] = {}
    for raw_key, raw_value in raw_goals.items():
        if not isinstance(raw_value, (int, float)):
            continue
        key = raw_key.decode("utf-8", errors="ignore") if isinstance(raw_key, bytes) else str(raw_key)
        normalized[key] = float(raw_value)
    return normalized


def _get_canonical_thresholds_for_mode(is_studio_2026: bool) -> dict[str, float]:
    """Load PMGG goal thresholds lazily for the current mode."""
    threshold_fn = cast(
        Callable[..., dict[str, float]],
        _load_symbol("backend.core.per_phase_musical_goals_gate", "_get_canonical_thresholds"),
    )
    return threshold_fn(is_studio_2026=is_studio_2026)


def _score_versa_mos(audio: np.ndarray, sr: int) -> Any:
    """Run VERSA MOS via lazy import to keep optional plugin loading local."""
    score_mos_fn = cast(Callable[[np.ndarray, int], Any], _load_symbol("plugins.versa_plugin", "score_mos"))
    return score_mos_fn(audio, sr)


def _set_pipeline_active(active: bool) -> None:
    """Inform the plugin lifecycle manager that the heavy pipeline is active."""
    set_active_fn = cast(
        Callable[[bool], None],
        _load_symbol("backend.core.plugin_lifecycle_manager", "set_pipeline_active"),
    )
    set_active_fn(active)


def _evict_stale_plugins(required_mb: float | None = None) -> int:
    """Evict inactive plugins through the lifecycle manager."""
    evict_fn = cast(
        Callable[..., int],
        _load_symbol("backend.core.plugin_lifecycle_manager", "evict_stale_plugins"),
    )
    if required_mb is None:
        return int(evict_fn())
    return int(evict_fn(required_mb=required_mb))


def _probe_benign_digital_source(
    audio: np.ndarray,
    sr: int,
    resolved_material: str,
) -> tuple[bool, dict[str, Any]]:
    """Call UV3's clean-digital probe without hard importing the full backend at module import time."""
    material_type = _load_symbol("backend.core.defect_scanner", "MaterialType")
    uv3_cls = _load_symbol("backend.core.unified_restorer_v3", "UnifiedRestorerV3")
    _probe_name = "_is_benign_digital_source"
    benign_probe = cast(
        Callable[[np.ndarray, int, Any], tuple[bool, dict[str, Any]]],
        getattr(uv3_cls, _probe_name),
    )
    material_enum = next(
        (member for member in material_type if getattr(member, "value", None) == resolved_material),
        None,
    )
    benign, metrics = benign_probe(audio, sr, material_enum)
    return bool(benign), dict(metrics)


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class AurikErgebnis:
    """Vollständiges Restaurierungsergebnis des AurikDenker-Orchestrators.

    Felder:
        audio:               Restauriertes Audio (float32, clip [-1, 1])
        material:            Erkanntes Trägermedium (z. B. "tape", "vinyl")
        rt_factor:           Tatsächlicher Echtzeit-Faktor (≤ 8.0)
        quality_estimate:    Qualitätsschätzung ∈ [0, 1]
        musical_goals:       14 Musical-Goals-Scores
        goals_passed:        Anzahl bestandener Goals
        phases_executed:     Liste der ausgeführten Verarbeitungsphasen
        warnings:            Warnungsliste (technisch, Englisch)
        processing_note:     Kurzbeschreibung auf Deutsch
        stage_notes:         Detailnotizen je Verarbeitungsstufe
        chain_info:          Tonträgerketten-Analyse (optional)
        confidence:          Gesamtkonfidenz der Restaurierung ∈ [0, 1]
        rollback_triggered:  War ARE-Rollback ausgelöst?
        winning_variant:     Beste ARE-Variante (oder None)
        gaps_found:          Erkannte Dropout-Lücken
        gaps_repaired:       Erfolgreich reparierte Lücken
        gap_total_repaired_ms: Gesamtdauer reparierter Lücken in ms
    """

    audio: np.ndarray
    material: str
    rt_factor: float
    quality_estimate: float
    musical_goals: dict[str, float]
    goals_passed: int
    phases_executed: list[str]
    warnings: list[str] = field(default_factory=list)
    processing_note: str = ""
    stage_notes: dict[str, str] = field(default_factory=dict)
    chain_info: dict[str, Any] | None = field(default=None)
    # ── ARE-spezifische Felder (A-2) ─────────────────────────────────────────
    confidence: float = 0.85
    rollback_triggered: bool = False
    winning_variant: str | None = None
    gaps_found: int = 0
    gaps_repaired: int = 0
    gap_total_repaired_ms: float = 0.0
    degradation_status: str = PipelineHealthState.OK.value
    fail_reason: str | None = None
    # §Dach: Musikalischer Globalplan — stilbewusstes Restaurierungsportrait
    global_plan: dict[str, Any] | None = field(default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "material": self.material,
            "rt_factor": float(self.rt_factor),
            "quality_estimate": float(self.quality_estimate),
            "musical_goals": {k: float(v) for k, v in self.musical_goals.items()},
            "goals_passed": self.goals_passed,
            "phases_executed": list(self.phases_executed),
            "warnings": list(self.warnings),
            "processing_note": self.processing_note,
            "stage_notes": dict(self.stage_notes),
            "chain_info": self.chain_info,
            "confidence": float(self.confidence),
            "rollback_triggered": self.rollback_triggered,
            "winning_variant": self.winning_variant,
            "gaps_found": self.gaps_found,
            "gaps_repaired": self.gaps_repaired,
            "gap_total_repaired_ms": float(self.gap_total_repaired_ms),
            "degradation_status": self.degradation_status,
            "fail_reason": self.fail_reason,
            "global_plan": self.global_plan,
            "metadata": dict(self.metadata),
        }


# ─── Orchestrator ────────────────────────────────────────────────────────────


class AurikDenker:
    """Orchestrator mit 8 Verarbeitungsstufen im Aurik-System.

    Steuert die vollständige Restaurierungs-Pipeline sequenziell und
    überwacht dabei das 8×RT-Budget. Jeder Domänen-Denker läuft in
    try/except — ein Fehler in einer Stufe stoppt nicht die gesamte Pipeline.

    Verwendung::

        aurik = get_aurik_denker()
        ergebnis = aurik.restauriere(audio, sr=48_000)
        logger.debug("RT-Faktor: %.2f×", ergebnis.rt_factor)
        logger.debug("Qualität: %.3f (VERSA MOS eingerechnet)", ergebnis.quality_estimate)
        logger.debug("Goals: %s/%s", ergebnis.goals_passed, len(ergebnis.musical_goals))
    """

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def restauriere(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        *,
        validate_audio: bool = True,
        mode: str = "quality",
        progress_callback: Any | None = None,
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
    ) -> AurikErgebnis:
        """Vollständige Aurik-Restaurierung: 8 Stufen orchestriert.

        Args:
            audio:         Audio-Signal (mono/stereo, float32)
            sr:            Sample-Rate in Hz (Spec §6.5: immer 48 000 Hz)
            validate_audio: Ob Eingabe-Validierung durchgeführt werden soll

        Returns:
            AurikErgebnis mit restauriertem Audio und vollständiger Bewertung.
            Bei Fehlern: ursprüngliches Audio + Fehlerbeschreibung.
        """
        t_start = time.perf_counter()
        assert sr == 48000, f"AurikDenker.restauriere() erwartet sr=48000 Hz, erhalten: {sr} Hz"

        _dur_s = len(audio) / max(sr, 1) if audio.ndim == 1 else audio.shape[0] / max(sr, 1)
        logger.info(
            "AurikDenker.denke() gestartet: mode=%s, sr=%d, duration=%.1fs, shape=%s, "
            "caches=[era=%s, genre=%s, defect=%s, medium=%s, rest=%s]",
            mode,
            sr,
            _dur_s,
            audio.shape,
            cached_era_result is not None,
            cached_genre_result is not None,
            cached_defect_result is not None,
            cached_medium_result is not None,
            cached_restorability_result is not None,
        )

        # NaN/Inf-Schutz (Spec §3.1)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if validate_audio and audio.size < _MIN_AUDIO_SAMPLES:
            elapsed = time.perf_counter() - t_start
            return self._fallback(
                audio,
                rt_factor=0.0,
                grund=f"Signal zu kurz ({audio.size} Samples < {_MIN_AUDIO_SAMPLES})",
            )

        # Audio-Dauer für RT-Berechnung
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0)
        audio_duration_s = len(audio_mono) / max(sr, 1)

        try:
            ergebnis = self._orchestriere(
                audio,
                sr,
                audio_duration_s,
                t_start,
                mode=mode,
                progress_callback=progress_callback,
                audio_update_callback=audio_update_callback,
                cached_era_result=cached_era_result,
                cached_genre_result=cached_genre_result,
                cached_defect_result=cached_defect_result,
                cached_medium_result=cached_medium_result,
                cached_restorability_result=cached_restorability_result,
                recovery_checkpoint=recovery_checkpoint,
                input_path=input_path,
                output_path=output_path,
                no_rt_limit=no_rt_limit,
            )
        except MemoryError as exc:
            elapsed = time.perf_counter() - t_start
            rt = elapsed / max(audio_duration_s, 1e-6)
            try:
                _evict_stale_plugins(required_mb=4096.0)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            logger.error("AurikDenker: Speicherfehler in Pipeline: %s", exc)
            return self._fallback(
                audio,
                rt_factor=min(rt, _3X_RT_LIMIT),
                grund="Speicherfehler (OOM-Guard): ML-Last zu hoch, DSP-Fallback empfohlen",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            rt = elapsed / max(audio_duration_s, 1e-6)
            logger.error("AurikDenker: Unerwarteter Fehler in Pipeline: %s", exc)
            return self._fallback(
                audio,
                rt_factor=min(rt, _3X_RT_LIMIT),
                grund=f"Pipeline-Fehler: {exc}",
            )

        return ergebnis

    def denke(self, audio: np.ndarray, sr: int = 48_000, **kwargs: Any) -> AurikErgebnis:
        """Alias für restauriere() — API-Kompatibilität (Spec §2.2)."""
        # Unpack pre_analysis_result into individual cached_* kwargs if present.
        # BatchProcessingThread passes a PreAnalysisResult object; restauriere() expects
        # individual cached_* parameters.  Without this step all cached analyses would be
        # silently discarded, forcing DefektDenker / TontraegerDenker to re-run (~120 s).
        _pre = kwargs.get("pre_analysis_result")
        if _pre is not None:
            if kwargs.get("cached_era_result") is None and getattr(_pre, "era", None) is not None:
                kwargs["cached_era_result"] = _pre.era
            if kwargs.get("cached_genre_result") is None and getattr(_pre, "genre", None) is not None:
                kwargs["cached_genre_result"] = _pre.genre
            if kwargs.get("cached_defect_result") is None and getattr(_pre, "defects", None) is not None:
                kwargs["cached_defect_result"] = _pre.defects
            if kwargs.get("cached_medium_result") is None and getattr(_pre, "medium", None) is not None:
                kwargs["cached_medium_result"] = _pre.medium
            if kwargs.get("cached_restorability_result") is None and getattr(_pre, "restorability", None) is not None:
                kwargs["cached_restorability_result"] = _pre.restorability
            logger.debug(
                "AurikDenker.denke(): pre_analysis_result entpackt (era=%s genre=%s defects=%s medium=%s rest=%s)",
                "✓" if kwargs.get("cached_era_result") else "—",
                "✓" if kwargs.get("cached_genre_result") else "—",
                "✓" if kwargs.get("cached_defect_result") else "—",
                "✓" if kwargs.get("cached_medium_result") else "—",
                "✓" if kwargs.get("cached_restorability_result") else "—",
            )
        return self.restauriere(
            audio,
            sr,
            mode=kwargs.get("mode", "quality"),
            progress_callback=kwargs.get("progress_callback"),
            audio_update_callback=kwargs.get("audio_update_callback"),
            cached_era_result=kwargs.get("cached_era_result"),
            cached_genre_result=kwargs.get("cached_genre_result"),
            cached_defect_result=kwargs.get("cached_defect_result"),
            cached_medium_result=kwargs.get("cached_medium_result"),
            cached_restorability_result=kwargs.get("cached_restorability_result"),
            recovery_checkpoint=kwargs.get("recovery_checkpoint"),
            # §2.39 OOM-Recovery: Pfade für Checkpoint-Persistierung durchreichen
            input_path=kwargs.get("input_path", ""),
            output_path=kwargs.get("output_path", ""),
            no_rt_limit=bool(kwargs.get("no_rt_limit", False)),
        )

    @staticmethod
    def _resolve_excellence_material(material: str, chain_info: dict[str, Any] | None) -> str:
        """Uses the final carrier stage for Excellence decisions when available."""
        if isinstance(chain_info, dict):
            primary_medium = str(chain_info.get("primary_medium", "")).strip().lower()
            if primary_medium:
                return primary_medium
            chain = chain_info.get("chain")
            if isinstance(chain, list) and chain:
                last = str(chain[-1]).strip().lower()
                if last:
                    return last
        return str(material or "unknown").strip().lower()

    @classmethod
    def _should_skip_excellence_for_clean_digital(
        cls,
        audio: np.ndarray,
        sr: int,
        material: str,
        chain_info: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, float | str]]:
        """Skips Excellence for benign digital end-formats to avoid overprocessing."""
        resolved_material = cls._resolve_excellence_material(material, chain_info)

        # Fast-path for very short benign clips: avoid full ARE/UV3 orchestration in
        # tiny snippets where heavy restoration is not meaningful and often unstable.
        try:
            arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                mono = arr.mean(axis=0) if arr.shape[0] <= arr.shape[1] else arr.mean(axis=1)
                sample_count = int(mono.shape[0])
            else:
                mono = arr
                sample_count = int(mono.shape[0])

            duration_s = float(sample_count / max(1, sr))
            _short_clip_materials = {"unknown", "digital_unknown", "mp3_low", "mp3_high", "aac", "cd_digital", "dat"}
            if duration_s <= 5.0 and resolved_material in _short_clip_materials and sample_count > 0:
                abs_mono = np.abs(mono)
                hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
                near_clip_ratio = float(np.mean(abs_mono >= 0.98))
                rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
                # Only skip if truly silent (RMS ≤ -60dBFS, ~0.001).
                # Avoid confusing noisy signal with "clean" audio.
                if hard_clip_ratio <= 1e-4 and near_clip_ratio <= 1e-3 and rms <= 0.001:
                    logger.warning(
                        "Short silence clip autopass: duration=%.1fs, material=%s, rms=%.4f — "
                        "To force restoration, use mode='studio2026'",
                        duration_s,
                        resolved_material,
                        rms,
                    )
                    return True, {
                        "material": resolved_material,
                        "reason": "short_benign_clip_autopass",
                        "duration_s": float(round(duration_s, 3)),
                        "hard_clip_ratio": hard_clip_ratio,
                        "near_clip_ratio": near_clip_ratio,
                        "rms": float(round(rms, 6)),
                    }
        except Exception as exc:
            logger.debug("AurikDenker: short-clip guard unavailable: %s", exc)

        # Never skip restoration when the recording chain has an analog origin.
        # primary_medium = chain[-1] (e.g. "mp3_low"), but if original_medium = "tape",
        # the source is a tape transfer and must not be treated as a clean digital source.
        _ANALOG_ORIGINALS = {
            "tape",
            "reel_tape",
            "vinyl",
            "shellac",
            "cassette",
            "phonograph",
            "wax_cylinder",
        }
        if isinstance(chain_info, dict):
            _orig = str(chain_info.get("original_medium", "")).strip().lower()
            if _orig in _ANALOG_ORIGINALS:
                logger.debug(
                    "AurikDenker: analog-origin chain (%s → %s) — pass-through blocked.",
                    _orig,
                    resolved_material,
                )
                return False, {
                    "material": resolved_material,
                    "reason": "analog_chain_no_passthrough",
                    "original_medium": _orig,
                }

        try:
            benign, metrics = _probe_benign_digital_source(audio, sr, resolved_material)
            metrics = dict(metrics)
            metrics.setdefault("material", resolved_material)
            return benign, metrics
        except Exception as exc:
            logger.debug("AurikDenker: Clean-digital guard unavailable: %s", exc)
            return False, {"material": resolved_material, "reason": "guard_unavailable"}

    @classmethod
    def _material_mos_target(cls, material: str, chain_info: dict[str, Any] | None) -> tuple[str, float]:
        """Returns resolved material key and its adaptive MOS target."""
        resolved = cls._resolve_excellence_material(material, chain_info)
        target = float(_MATERIAL_MOS_TARGETS.get(resolved, _MATERIAL_MOS_TARGETS["unknown"]))
        return resolved, target

    @staticmethod
    def _normalize_mode_name(mode: str | None) -> str:
        """Normalizes user-facing mode aliases to canonical internal names."""
        normalized = str(mode or "quality").strip().lower().replace("_", "")
        aliases = {
            "studio 2026": "studio2026",
            "studio2026": "studio2026",
            "restoration": "restoration",
            "quality": "quality",
            "balanced": "balanced",
            "fast": "fast",
            "maximum": "maximum",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _recommend_autopilot_mode(
        cls,
        *,
        requested_mode: str,
        material: str,
        chain_info: dict[str, Any] | None,
        defekt: Any,
        global_plan: Any,
        strategy_mode: str | None,
    ) -> tuple[str, str]:
        """Selects the safest high-level mode for one-button/autopilot execution."""
        requested = cls._normalize_mode_name(requested_mode)
        strategy = cls._normalize_mode_name(strategy_mode)
        resolved_material = cls._resolve_excellence_material(material, chain_info)
        severity = float(getattr(defekt, "overall_severity", 0.0) or 0.0)
        primary_defect = (
            str(getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")).strip().lower()
        )

        decade_value = None
        genre_value = ""
        if global_plan is not None:
            portrait = getattr(global_plan, "portrait", None)
            if portrait is not None:
                try:
                    decade_value = int(getattr(portrait, "decade", 0) or 0)
                except (TypeError, ValueError):
                    decade_value = None
                genre_value = str(getattr(portrait, "genre", "") or "").strip().lower()

        is_historical_material = resolved_material in _HISTORICAL_OR_FRAGILE_MATERIALS
        is_modern_digital = resolved_material in _MODERN_DIGITAL_MATERIALS
        is_historical_era = decade_value is not None and decade_value <= 1965
        is_schlager = "schlager" in genre_value
        has_heavy_defect = severity >= 0.60 or any(token in primary_defect for token in _HEAVY_DEFECT_HINTS)
        has_medium_risk = severity >= 0.35

        recommended = "restoration"
        reason = "Unsicherer oder analoger Fall: Restoration als sicherer Standard."

        if is_historical_material or is_historical_era or has_heavy_defect:
            reason = (
                f"Restoration empfohlen: Material={resolved_material}, decade={decade_value}, "
                f"severity={severity:.2f}, defect={primary_defect}."
            )
        elif is_schlager:
            if is_modern_digital and not has_medium_risk and (decade_value is None or decade_value >= 1970):
                recommended = "studio2026"
                reason = (
                    f"Studio 2026 empfohlen: stabiler Schlager-Fall auf {resolved_material} "
                    f"bei severity={severity:.2f}."
                )
            else:
                reason = (
                    f"Restoration empfohlen: Schlager bleibt konservativ auf {resolved_material} "
                    f"bei severity={severity:.2f}."
                )
        elif is_modern_digital and not has_medium_risk and (decade_value is None or decade_value >= 1970):
            recommended = "studio2026"
            reason = f"Studio 2026 empfohlen: modernes Digitalmaterial {resolved_material} bei severity={severity:.2f}."
        elif strategy in {"studio2026", "maximum"} and not has_medium_risk and is_modern_digital:
            recommended = "studio2026"
            reason = (
                f"Studio 2026 empfohlen: Strategie bevorzugt High-End und Material {resolved_material} wirkt stabil."
            )

        if requested in {"restoration", "studio2026", "maximum", "fast", "balanced"}:
            if requested == "studio2026" and recommended != "studio2026":
                # §Bindend: Nutzerwahl respektieren — Studio 2026 wird NICHT überschrieben.
                # Autopilot-Empfehlung ist ein Hinweis, kein Veto. Der Nutzer hat genau
                # eine bewusste Entscheidung (§Autonomer Magic-Button-Vertrag).
                # Konservative Schutzmaßnahmen greifen INNERHALB Studio 2026:
                # — SongCalibration reduziert Strength bei starken Defekten
                # — PMGG erzwingt Musical-Goals-Einhaltung pro Phase
                # — FeedbackChain rollt bei Regression zurück
                logger.warning(
                    "Studio 2026 auf nicht-idealem Material beibehalten (Nutzerwahl): material=%s, severity=%.2f — %s",
                    resolved_material,
                    severity,
                    reason,
                )
                return "studio2026", (
                    f"Studio 2026 beibehalten (Nutzer-Entscheidung). "
                    f"Hinweis: {reason} "
                    f"Interne Schutzmaßnahmen (SongCal, PMGG, FeedbackChain) sind aktiv."
                )
            return requested, f"Expliziter Modus '{requested}' beibehalten. {reason}"

        if requested == "quality":
            return recommended, f"Autopilot wählte '{recommended}'. {reason}"

        return recommended, f"Unbekannter Modus '{requested_mode}' auf '{recommended}' normalisiert. {reason}"

    # ── Pipeline-Logik ───────────────────────────────────────────────────────

    def _orchestriere(
        self,
        audio: np.ndarray,
        sr: int,
        audio_duration_s: float,
        t_start: float,
        *,
        mode: str = "quality",
        progress_callback: Any | None = None,
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
    ) -> AurikErgebnis:
        """Führt die 10-stufige Restaurierungs-Pipeline aus.

        Jede Stufe wird in try/except gekapselt. Fehler einer Stufe führen
        zur Fortsetzung mit dem bisherigen Audio-Zustand.
        """

        def _emit(pct: int, msg: str) -> None:
            """Emittiert Fortschritt (0–100) wenn progress_callback gesetzt."""
            if progress_callback is not None:
                try:
                    elapsed = time.perf_counter() - t_start
                    progress_callback(pct, msg, elapsed)
                except Exception as cb_exc:
                    logger.debug(
                        "AurikDenker: Progress-Callback fehlgeschlagen (Ursache: %s). "
                        "Lösung: Callback-Signatur (pct, msg, elapsed_s) prüfen.",
                        cb_exc,
                    )

        warnings: list[str] = []
        phases_executed: list[str] = []
        stage_notes: dict[str, Any] = {}
        aktuelles_audio = audio.copy()
        effective_mode = self._normalize_mode_name(mode)
        _critical_stages: frozenset[str] = frozenset({"tontraeger", "defekt", "restaurierung"})
        _stage_severity: dict[str, str] = {
            "tontraeger": "critical",
            "kette": "degraded",
            "defekt": "critical",
            "globalplan": "degraded",
            "strategie": "critical",
            "restaurierung": "critical",
            "exzellenz": "degraded",
        }
        _stage_fail_reasons: list[dict[str, str]] = []
        _degradation_status: str = PipelineHealthState.OK.value
        _fail_reason: str | None = None

        def _record_stage_failure(stage: str, component: str, exc: Exception) -> None:
            msg = str(exc)
            warnings.append(f"{component} fehlgeschlagen: {msg}")
            stage_notes[stage] = f"Fehler: {msg}"
            _stage_fail_reasons.append(
                {
                    "component": component,
                    "stage": stage,
                    "error_code": f"STAGE_{stage.upper()}_FAILED",
                    "severity": _stage_severity.get(stage, "degraded"),
                    "exc_type": type(exc).__name__,
                    "exc_msg": msg,
                }
            )

        def _rt() -> float:
            elapsed = time.perf_counter() - t_start
            return elapsed / max(audio_duration_s, 1e-6)

        strat_denker: Any = None  # M-2: hoisted — für _budget_ok()-Zugriff nach Stufe 5
        defekt: Any = None  # M-1/M-4: hoisted — für Stage-5-Flags

        def _budget_ok() -> bool:
            if no_rt_limit:
                return True
            # M-2: RT-Primärcheck — mode-abhängig (Spec §9.5):
            #   BALANCED=32× | RESTORATION/QUALITY=32× | MAXIMUM=32×
            # Zusätzlich: absolutes 90-Minuten-Limit — gilt unabhängig vom RT-Faktor.
            # StrategieDenker.check() wird hier NICHT verwendet, da es immer 8×RT
            # hardcoded prüft und so ExzellenzDenker in Restoration/Quality-Mode
            # fälschlich blockieren würde. Der primäre _RT_BUDGET_BY_MODE-Check ist
            # die einzige authoritative Schranke.
            if (time.perf_counter() - t_start) >= _MAX_TOTAL_SECONDS:
                logger.warning(
                    "⚠️ 30-Minuten-Absolutlimit erreicht (%.0fs) — Stage wird übersprungen.",
                    time.perf_counter() - t_start,
                )
                return False
            _mode_limit = _RT_BUDGET_BY_MODE.get(effective_mode.lower(), 5.0)
            return _rt() < _mode_limit * 0.90

        # ── Stufe 1: Tonträger-Erkennung ─────────────────────────────────────
        # Skip full TontraegerDenker run when the frontend has already analysed the
        # carrier (cached_medium_result from _carrier_bg / bridge cache) — avoids a
        # redundant MediumDetector pass that the user sees as "Tonträger wird erkannt".
        material = "unknown"
        if cached_medium_result is not None:
            # Use same multi-fallback as UV3 (line 1564): primary_material → material_type → material
            material = str(
                getattr(cached_medium_result, "primary_material", None)
                or getattr(cached_medium_result, "material_type", None)
                or getattr(cached_medium_result, "material", None)
                or "unknown"
            )
            stage_notes["tontraeger"] = (
                f"{material} (Cache, Konfidenz: {getattr(cached_medium_result, 'confidence', 0.0):.2f})"
            )
            phases_executed.append("tontraeger_erkennung")
            logger.info(
                "AurikDenker [1/10] Träger aus Frontend-Cache: %s (%.2f)",
                material,
                getattr(cached_medium_result, "confidence", 0.0),
            )
            _emit(2, f"Tonträger erkannt: {material} (Cache)")
        else:
            _emit(2, "Tonträger wird erkannt …")
            try:
                toni = get_tontraeger_denker().erkenne(aktuelles_audio, sr, file_path=input_path)
                material = toni.material_type
                stage_notes["tontraeger"] = f"{material} (Konfidenz: {toni.confidence:.2f})"
                phases_executed.append("tontraeger_erkennung")
                logger.info("AurikDenker [1/10] Träger: %s (%.2f)", material, toni.confidence)
                # §6.7 v9.10.97: Bayesian ClassificationResult aus Stufe 1 als cached_medium_result
                # für UV3 übernehmen — eliminiert redundante MediumClassifier-Aufrufe.
                if getattr(toni, "classification_result", None) is not None:
                    cached_medium_result = toni.classification_result
                    logger.info(
                        "AurikDenker: MediumDetector-ClassificationResult als cached_medium_result übernommen "
                        "(material=%s, conf=%.2f)",
                        getattr(cached_medium_result, "material_type", material),
                        getattr(cached_medium_result, "confidence", toni.confidence),
                    )
            except Exception as exc:
                _record_stage_failure("tontraeger", "TontraegerDenker", exc)
                logger.warning("AurikDenker [1/10] TontraegerDenker: %s", exc)

        # ── Stufe 2: Tonträgerketten-Analyse ──────────────────────────────────
        # §2.47a: Pass cached_medium_result to avoid a second MediumDetector.detect() call.
        _emit(4, "Tonträgerkette analysiert …")
        chain_info: dict[str, Any] = {}
        kette = None  # Guard: TontraegerketteDenker.analysiere() kann fehlschlagen
        try:
            kette = get_tontraegerkette_denker().analysiere(
                aktuelles_audio,
                sr,
                file_path=input_path,
                cached_medium_result=cached_medium_result,
            )
            chain_info = kette.as_dict()
            stage_notes["kette"] = kette.chain_string
            phases_executed.extend(kette.combined_phases)
            logger.info(
                "AurikDenker [2/10] Kette: %s (Komplexität: %.2f)",
                kette.chain_string,
                kette.chain_complexity,
            )
            if getattr(kette, "chain", None):
                _emit(4, "__carrier_chain__:" + "|".join(str(_k) for _k in (kette.chain or [])))
        except Exception as exc:
            _record_stage_failure("kette", "TontraegerketteDenker", exc)
            logger.warning("AurikDenker [2/10] TontraegerketteDenker: %s", exc)

        # ── Stufe 3: Defekt-Analyse ───────────────────────────────────────────
        _emit(6, "Defekte werden analysiert …")
        defekt_primaer = "unbekannt"
        defekt = None  # Guard: DefektDenker kann fehlschlagen
        _defekt_hint: dict[str, Any] | None = None

        def _defekt_scan_cb(pct: int, name: str = "") -> None:
            if name:
                _emit(6, f"Defekte werden analysiert … {name}")

        try:
            defekt = get_defekt_denker().analysiere(
                aktuelles_audio,
                sr,
                material=material,
                progress_callback=_defekt_scan_cb,
                cached_defect_result=cached_defect_result,
                file_ext=os.path.splitext(input_path)[1].lower() if input_path else "",
            )
            defekt_primaer = getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")
            stage_notes["defekt"] = f"Hauptdefekt: {defekt_primaer} (Schwere: {defekt.overall_severity:.2f})"
            phases_executed.append("defekt_analyse")
            _defekt_hint = {
                "recommended_phases": list(getattr(defekt, "recommended_phases", [])),
                "confidence": float(getattr(defekt, "cause_confidence", 0.0)),
            }
            # Bug-17-Fix: raw DefectAnalysisResult aus DefektErgebnis extrahieren und
            # als cached_defect_result weiterreichen — verhindert zweiten internen Scan
            # in RestaurierDenker (ARE-Pfad) mit falscher Material-Erkennung.
            if cached_defect_result is None:
                _raw = getattr(defekt, "raw_scan_result", None)
                if _raw is not None:
                    cached_defect_result = _raw
                    logger.info(
                        "AurikDenker: DefektScan-Ergebnis (material=%s) als cached_defect_result übernommen.",
                        getattr(_raw, "material_type", "?"),
                    )
            logger.info(
                "AurikDenker [3/10] Defekt: %s (Schwere: %.2f)",
                defekt_primaer,
                defekt.overall_severity,
            )
        except Exception as exc:
            _record_stage_failure("defekt", "DefektDenker", exc)
            logger.warning("AurikDenker [3/10] DefektDenker: %s", exc)

        # ── Stufe 4: Musikalischer Globalplan (§Dach) ────────────────────────
        _emit(8, "Musikalischer Restaurierungsplan erstellt …")
        _globalplan: Any = None
        try:
            # use_ml_classifiers=False: EraClassifier/GermanSchlagerClassifier laufen
            # bereits parallel in UnifiedRestorerV3 (§P-3). Doppelaufruf vermeiden
            # (Anti-Parallelwelten-Pflicht). Nur DSP-Heuristik in Stufe 4.
            # hint_decade: cached_era_result aus Frontend-Vorabanalyse übergeben —
            # verhindert, dass GlobalPlan via DSP-Rolloff eine physikalisch unmögliche
            # Ära (z.B. 1890er für Magnetband) berechnet, die dann _recommend_autopilot_mode
            # und die emotionale Intention falsch setzt.
            _hint_decade = int(getattr(cached_era_result, "decade", 0) or 0) or None
            # hint_genre: cached genre result aus Pre-Analyse übergeben —
            # verhindert dass MusikalischerGlobalplan Genre als "unknown" einträgt
            # wenn GenreClassifier bereits Schlager/Jazz/etc. erkannt hat (§2.47a).
            _hint_genre: str | None = None
            if cached_genre_result is not None:
                if getattr(cached_genre_result, "is_schlager", False):
                    _hint_genre = "schlager"
                else:
                    _gl = getattr(cached_genre_result, "genre_label", None) or getattr(
                        cached_genre_result, "primary_label", None
                    )
                    if _gl and str(_gl).lower() not in ("unbekannt", "unknown", ""):
                        _hint_genre = str(_gl).lower()
            _globalplan = erstelle_globalplan(
                aktuelles_audio,
                sr,
                material=material,
                use_ml_classifiers=False,
                chain_info=chain_info,
                hint_decade=_hint_decade,
                hint_genre=_hint_genre,
            )
            stage_notes["globalplan"] = (
                f"\u00c4ra: {_globalplan.portrait.decade}er, "
                f"Genre: {_globalplan.portrait.genre}, "
                f"Intention: {_globalplan.emotional_intention}"
            )
            phases_executed.append("musikalischer_globalplan")
            logger.info(
                "AurikDenker [4/10] Globalplan: %s\u00b4er | %s | Authentizit\u00e4t=%.2f",
                _globalplan.portrait.decade,
                _globalplan.portrait.genre,
                _globalplan.authenticity_target,
            )
        except Exception as exc:
            _record_stage_failure("globalplan", "MusikalischerGlobalplan", exc)
            logger.warning("AurikDenker [4/10] MusikalischerGlobalplan: %s", exc)

        # ── Stufe 5: Strategie (8×RT-Budget) ────────────────────────────────
        _emit(10, "Restaurierungsstrategie geplant …")
        strategie = None
        _strat_mode = "quality"
        try:
            strat_denker = get_strategie_denker()
            # M-2b: §7.6 defekt-adaptive Chunk-Größe — übergebe Defektschwere an Strategie
            _defect_sev_for_plan = float(getattr(defekt, "overall_severity", 0.0)) if defekt is not None else 0.0
            strategie = strat_denker.plan(
                aktuelles_audio,
                sr,
                enforce_3x_rt=True,
                defect_severity=_defect_sev_for_plan,
            )
            strat_denker.starte_timer(audio_duration_s)
            _budget_raw = getattr(strategie, "max_processing_s", audio_duration_s * _3X_RT_LIMIT)
            try:
                _budget_s = float(_budget_raw)
            except (TypeError, ValueError):
                _budget_s = float(audio_duration_s * _3X_RT_LIMIT)
            _mode_raw = getattr(strategie, "quality_mode", "quality")
            _strat_mode = str(_mode_raw) if _mode_raw is not None else "quality"
            stage_notes["strategie"] = f"Budget: {_budget_s:.1f}s, Modus: {_strat_mode}"
            phases_executed.append("strategie_plan")
            logger.info(
                "AurikDenker [5/10] Budget: %.1fs für %.1fs Audio",
                _budget_s,
                audio_duration_s,
            )
        except Exception as exc:
            _record_stage_failure("strategie", "StrategieDenker", exc)
            logger.warning("AurikDenker [5/10] StrategieDenker: %s", exc)

        try:
            effective_mode, autopilot_note = self._recommend_autopilot_mode(
                requested_mode=mode,
                material=material,
                chain_info=chain_info,
                defekt=defekt,
                global_plan=_globalplan,
                strategy_mode=_strat_mode,
            )
        except Exception as _autopilot_exc:
            logger.warning("Autopilot mode selection failed: %s — defaulting to requested mode", _autopilot_exc)
            effective_mode = self._normalize_mode_name(mode) or "restoration"
            autopilot_note = f"Autopilot fallback (error): {_autopilot_exc}"
        stage_notes["autopilot"] = autopilot_note
        if self._normalize_mode_name(mode) == "studio2026" and effective_mode == "restoration":
            warnings.append("Autopilot-Sicherheitsfallback: Studio 2026 wurde auf Restoration zurückgesetzt.")
        elif (
            self._normalize_mode_name(mode) == "studio2026"
            and effective_mode == "studio2026"
            and "Hinweis:" in autopilot_note
        ):
            warnings.append(
                "Studio 2026 auf nicht-idealem Material: Interne Schutzmaßnahmen "
                "(SongCal, PMGG, FeedbackChain) sind aktiv."
            )

        # ── Stufe 5b: PhaseInteractionDenker — Orchestrierung übernehmen ────────
        # Erzeugt einen semantisch aufgelösten, konfliktfreien Phasenplan.
        # UV3.restore() erhält diesen als precomputed_phase_plan und agiert dann
        # als reiner Executor (kein _optimize_phase_plan_intelligence() mehr).
        # Bei Fehler: leerer Plan → UV3 selektiert autonom (fail-safe §0).
        _pid_plan = None
        _pid_phase_plan: list[str] | None = None
        try:
            _pid_defect_result = cached_defect_result or (
                getattr(defekt, "raw_scan_result", None) if defekt is not None else None
            )
            if _pid_defect_result is not None:
                _pid_rest_score: float = float(
                    getattr(cached_restorability_result, "restorability_score", 70.0)
                    if cached_restorability_result is not None
                    else 70.0
                )
                # §GoalRisk: Exzellenz-Prognose VOR UV3 — schützende Phasen prophylaktisch
                # aktivieren statt P1/P2-Verletzungen erst nach der Pipeline zu heilen (§2.45a).
                _goal_risk_map: dict[str, float] = {}
                try:
                    _goal_risk_map = get_exzellenz_denker().prognostiziere(
                        aktuelles_audio,
                        sr,
                        defect_result=_pid_defect_result,
                        material=material,
                    )
                    if _goal_risk_map:
                        logger.debug(
                            "AurikDenker [5b/10] Goal-Risiko-Prognose: %s",
                            {k: f"{v:.2f}" for k, v in _goal_risk_map.items()},
                        )
                except Exception as _grm_exc:
                    logger.debug("AurikDenker [5b/10] Goal-Risiko-Prognose: %s", _grm_exc)
                _pid_plan = get_phase_interaction_denker().plan(
                    defect_result=_pid_defect_result,
                    material=material,
                    mode=effective_mode,
                    chain_info=chain_info or None,
                    chain_result=kette,
                    defekt_hint=_defekt_hint,
                    audio=aktuelles_audio,
                    sr=sr,
                    restorability_score=_pid_rest_score,
                    goal_risk_map=_goal_risk_map or None,
                    strategie_plan=strategie,
                    causal_plan=defekt,
                )
                if _pid_plan.is_valid:
                    _pid_phase_plan = _pid_plan.phases
                    _pid_injected = sum(1 for n in _pid_plan.conflict_notes if "Injektion" in n)
                    stage_notes["phase_interaction"] = (
                        f"PhaseInteractionDenker: {len(_pid_phase_plan)} Phasen "
                        f"({len(_pid_plan.suppressed)} supprimiert, "
                        f"{len(_pid_plan.ordering_applied)} Ordnungsänderungen, "
                        f"{_pid_injected} injiziert)"
                    )
                    phases_executed.append("phase_interaction_denker")
                    logger.info(
                        "AurikDenker [5b/10] PhaseInteractionDenker: %d Phasen, supprimiert=%s, conflict_notes=%s",
                        len(_pid_phase_plan),
                        list(_pid_plan.suppressed.keys()),
                        _pid_plan.conflict_notes,
                    )
                else:
                    stage_notes["phase_interaction"] = "PhaseInteractionDenker: kein Plan — UV3-Fallback"
                    logger.info("AurikDenker [5b/10] PhaseInteractionDenker: kein Plan → UV3 übernimmt.")
            else:
                stage_notes["phase_interaction"] = "PhaseInteractionDenker: kein defect_result — übersprungen"
        except Exception as _pid_exc:
            stage_notes["phase_interaction"] = f"PhaseInteractionDenker fehlgeschlagen: {_pid_exc}"
            logger.warning("AurikDenker [5b/10] PhaseInteractionDenker: %s", _pid_exc)

        # ── ARE-Metadaten-Träger (A-2/A-5/B-1) ────────────────────────────────
        _rest_confidence: float = 0.85
        _rest_rollback: bool = False
        _rest_variant: str | None = None
        _rest_musical_goals: dict[str, float] = {}
        _rest_goals_passed: int = 0
        _rest_metadata: dict[str, Any] = {}
        _gaps_found: int = 0
        _gaps_repaired: int = 0
        _gap_total_ms: float = 0.0

        # ── Stufe 6–8: _run_rest()-Closure — Reparatur → Rekonstruktion → Restaurierung ──
        _emit(12, "Vorverarbeitung & Restaurierung laufen …")
        _skip_restoration, _skip_rest_metrics = self._should_skip_excellence_for_clean_digital(
            aktuelles_audio,
            sr,
            material,
            chain_info,
        )
        if _skip_restoration:
            _skip_mat = _skip_rest_metrics.get("material", self._resolve_excellence_material(material, chain_info))
            stage_notes["reparatur"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, keine Vorverarbeitung erforderlich)"
            )
            stage_notes["rekonstruktion"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, keine Lückenrekonstruktion erforderlich)"
            )
            stage_notes["restaurierung"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, "
                "Autopilot priorisiert Pass-Through vor Overprocessing)"
            )
            _rest_confidence = 1.0
            _rest_variant = "clean_digital_pass_through"
            phases_executed.append("clean_digital_pass_through")
            logger.info(
                "AurikDenker [8/10] Restaurierung übersprungen für saubere Digitalquelle: %s",
                _skip_rest_metrics,
            )
        elif _budget_ok():
            try:
                # M-3: Mode-Hierarchie: expliziter mode-Parameter hat Vorrang vor StrategieDenker
                _mode = effective_mode
                # RT-Budget-Guard: Restaurierung läuft in eigenem Thread.
                # Budget ist mode-abhängig (Spec §9.5: 5× für Quality, 8× für Maximum).
                # Cold-Start-Minimum: mindestens _COLDSTART_MIN_SECONDS um ML-Modell-Ladezeit
                # (PANNs 0.7 GB, wav2vec2 0.35 GB …) nicht dem Restaurierungs-Budget anzulasten.
                _rt_multiplier = _RT_BUDGET_BY_MODE.get(_mode.lower(), 5.0)
                _elapsed_so_far = time.perf_counter() - t_start
                _remaining = max(
                    audio_duration_s * _rt_multiplier - _elapsed_so_far,
                    _COLDSTART_MIN_SECONDS,
                )
                if not no_rt_limit:
                    # §9.5 Absolutes 30-Minuten-Limit: Thread-Timeout darf nie über die
                    # verbleibende Zeit bis zur Gesamtgrenze hinausgehen. Mindestens 30 s
                    # werden immer einkalkuliert, damit auch beim Cold-Start-Ablauf noch
                    # ein sinnvoller Verarbeitungsversuch möglich ist.
                    _abs_remaining = _MAX_TOTAL_SECONDS - _elapsed_so_far
                    if _abs_remaining <= 0:
                        raise RuntimeError(
                            f"30-Minuten-Absolutlimit bereits überschritten "
                            f"({_elapsed_so_far:.0f}s) — Restaurierung wird nicht gestartet."
                        )
                    _remaining = min(_remaining, max(30.0, _abs_remaining))
                _result_box: list = []
                _err_box: list = []
                _rep_result_box: list = []  # ReparaturDenker Ergebnis (4a)
                _rek_result_box: list = []  # RekonstruktionsDenker Ergebnis (4b)

                def _run_rest() -> None:
                    # §Safety: PLM-Eviction für gesamte Restaurierungskette sperren —
                    # nicht nur _execute_pipeline(), sondern auch ReparaturDenker und
                    # RekonstruktionsDenker nutzen ONNX-Modelle. Refcount-basiert,
                    # sodass UV3-interner Guard additiv funktioniert.
                    try:
                        _set_pipeline_active(True)
                    except Exception as _exc:
                        logger.debug("Operation failed (non-critical): %s", _exc)
                    try:
                        _work_audio = aktuelles_audio.copy()

                        # §2.39 OOM-Recovery: bei vorhandenem Checkpoint direkte
                        # Wiederaufnahme in UV3 (ohne erneute Reparatur-/Rekonstruktionsstufen).
                        if recovery_checkpoint is not None:

                            def _resume_cb(pct: int, msg: str, elapsed: float = 0.0) -> None:
                                if pct <= 19:
                                    d = 13 + int(pct * 7 / 19) if pct > 0 else 13
                                elif pct <= 85:
                                    d = 20 + int((pct - 20) * 67 / 65)
                                else:
                                    d = 87 + int((pct - 86) * 7 / 14)
                                _emit(min(94, d), msg)

                            _emit(13, "OOM-Recovery wird fortgesetzt …")
                            _result_box.append(
                                get_restaurier_denker().restauriere(
                                    _work_audio,
                                    sr,
                                    material=material,
                                    mode=_mode,
                                    global_plan=_globalplan,
                                    chain_info=chain_info or None,
                                    defekt_hint=_defekt_hint,
                                    progress_callback=_resume_cb if progress_callback is not None else None,
                                    audio_update_callback=audio_update_callback,
                                    cached_era_result=cached_era_result,
                                    cached_genre_result=cached_genre_result,
                                    cached_defect_result=(
                                        cached_defect_result
                                        or (getattr(defekt, "raw_scan_result", None) if defekt is not None else None)
                                    ),
                                    cached_medium_result=cached_medium_result,
                                    cached_restorability_result=cached_restorability_result,
                                    recovery_checkpoint=recovery_checkpoint,
                                    input_path=input_path,
                                    output_path=output_path,
                                    no_rt_limit=no_rt_limit,
                                )
                            )
                            return

                        # §G1: Pre-Repair-Referenz sichern — echtes Original VOR
                        # ReparaturDenker/RekonstruktionsDenker für referenz-basierte
                        # Musical Goals (Authentizität, Groove, Timbre, Artikulation).
                        _pre_repair_reference = _work_audio.copy()
                        # [6/10] ReparaturDenker — gezielter Phase-Mix (Preprocessing vor UV3)
                        _emit(13, "Gezielte DSP-Reparaturen (Vorverarbeitung) …")
                        _ph = set(getattr(defekt, "recommended_phases", None) or [])
                        _sev = float(getattr(defekt, "overall_severity", 1.0))
                        _skip_repair = _sev < 0.05 and defekt is not None
                        _remove_clicks: bool = not _skip_repair and (
                            defekt is None
                            or bool(
                                _ph
                                & {"phase_01_click_removal", "phase_09_crackle_removal", "phase_27_click_pop_removal"}
                            )
                            or _sev >= 0.3
                        )
                        _remove_hum: bool = not _skip_repair and (
                            defekt is None or bool(_ph & {"phase_02_hum_removal"}) or _sev >= 0.3
                        )
                        _repair_clipping: bool = not _skip_repair and (
                            defekt is None
                            or bool(_ph & {"phase_23_spectral_repair", "phase_06_frequency_restoration"})
                            or _sev >= 0.4
                        )
                        _dsp_op_names: dict[str, str] = {
                            "click_repair": "Knackser & Impulse werden entfernt",
                            "hum_removal": "Netzbrummen (50/60 Hz) wird entfernt",
                            "declip": "Übersteuerungen werden repariert",
                        }

                        def _repair_cb(op_key: str) -> None:
                            _label = _dsp_op_names.get(op_key, op_key)
                            _emit(13, f"DSP-Reparatur — {op_key}: {_label}")

                        # §2.41 v9.10.117: Defect-Locations + Scores aus cached_defect_result extrahieren,
                        # damit ReparaturDenker chirurgische (lokalisierte) Reparaturen durchführen kann.
                        _repair_defect_scores: dict[str, float] = {}
                        _repair_defect_locations: dict[str, list[tuple[float, float]]] = {}
                        _repair_era_decade: int | None = None
                        if cached_defect_result is not None and hasattr(cached_defect_result, "scores"):
                            for _rdt, _rds in cached_defect_result.scores.items():
                                _rdt_key = getattr(_rdt, "value", str(_rdt))
                                if hasattr(_rds, "severity"):
                                    _repair_defect_scores[_rdt_key] = float(_rds.severity)
                                if hasattr(_rds, "locations") and _rds.locations:
                                    _repair_defect_locations[_rdt_key] = list(_rds.locations)
                        elif defekt is not None:
                            _repair_defect_scores = dict(getattr(defekt, "defect_scores", {}) or {})
                        # Era-Dekade aus cached Era-Ergebnis
                        if cached_era_result is not None:
                            _repair_era_decade = int(getattr(cached_era_result, "decade", 0) or 0) or None

                        # §0c Codec-Chain-IQR-Floor: Ketten-Liste aus chain_info extrahieren,
                        # damit ReparaturDenker bei Terminal-Codec (mp3/aac) keinen
                        # zu aggressiven click_iqr verwendet (Brandenburg 1999).
                        # Key "transfer_chain" (TontraegerInfo.as_dict()) mit Fallback auf
                        # "chain" (ältere Serialisierungs-Pfade und pre_analysis-Handover).
                        _transfer_chain: list[str] | None = (
                            list(chain_info.get("transfer_chain") or chain_info.get("chain") or []) or None
                            if isinstance(chain_info, dict)
                            else None
                        )
                        rep = get_reparatur_denker().repariere(
                            _work_audio,
                            sr,
                            remove_clicks=_remove_clicks,
                            remove_hum=_remove_hum,
                            repair_clipping=_repair_clipping,
                            material=material or "",
                            progress_callback=_repair_cb,
                            defect_scores=_repair_defect_scores or None,
                            defect_locations=_repair_defect_locations or None,
                            era_decade=_repair_era_decade,
                            transfer_chain=_transfer_chain,
                        )
                        _work_audio = rep.audio
                        _rep_result_box.append(rep)
                        # [7/10] RekonstruktionsDenker — Lücken-Erkennung & Reparatur (Preprocessing vor UV3)
                        _emit(16, "Dropout-Lücken werden rekonstruiert (Vorverarbeitung) …")
                        rek = get_rekonstruktions_denker().rekonstruiere(
                            _work_audio,
                            sr,
                            material_hint=material,
                            defect_result=cached_defect_result,
                            repair_context=rep,  # §11.7a Kontextfluss: ReparaturDenker-Ergebnis weiterreichen
                            defect_locations=_repair_defect_locations or None,
                            era_decade=_repair_era_decade,
                        )
                        _work_audio = rek.audio
                        _rek_result_box.append(rek)
                        # 4c: RestaurierDenker (UV3-Vollpipeline) auf vorgereinigtem Material
                        # Scaled inner progress: UV3 0–100 → AurikDenker 13–94
                        # UV3-intern: Analyse pct 1–19, Pipeline pct 20–85, Post pct 86–96.
                        # Denker-Mapping:
                        #   UV3 0–19  (Analyse)   → Denker 13–20  (komprimiert: 7 pts)
                        #   UV3 20–85 (37 Phasen) → Denker 20–87  (Löwenanteil: 67 pts)
                        #   UV3 86–100 (Post)     → Denker 87–94  (komprimiert: 7 pts)

                        def _inner_cb(pct, msg, elapsed=0.0):
                            if pct <= 19:
                                d = 13 + int(pct * 7 / 19) if pct > 0 else 13
                            elif pct <= 85:
                                d = 20 + int((pct - 20) * 67 / 65)
                            else:
                                d = 87 + int((pct - 86) * 7 / 14)
                            _emit(min(94, d), msg)

                        _result_box.append(
                            get_restaurier_denker().restauriere(
                                _work_audio,
                                sr,
                                material=material,
                                mode=_mode,
                                global_plan=_globalplan,
                                chain_info=chain_info or None,
                                defekt_hint=_defekt_hint,
                                progress_callback=_inner_cb if progress_callback is not None else None,
                                audio_update_callback=audio_update_callback,
                                cached_era_result=cached_era_result,
                                cached_genre_result=cached_genre_result,
                                # Bug-17-Fix: intern berechnetes DefectAnalysisResult aus
                                # DefektDenker.analysiere() als cached_defect_result nutzen,
                                # damit RestaurierDenker den Direkt-UV3-Pfad (_has_caches=True)
                                # nehmen kann. Ohne diesen Fix: _has_caches=False → ARE-Pfad →
                                # konkurrierende UV3-Instanzen → OOM.
                                cached_defect_result=(
                                    cached_defect_result
                                    or (getattr(defekt, "raw_scan_result", None) if defekt is not None else None)
                                ),
                                cached_medium_result=cached_medium_result,
                                cached_restorability_result=cached_restorability_result,
                                reconstruction_context=rek,
                                pre_repair_reference=_pre_repair_reference,
                                input_path=input_path,
                                output_path=output_path,
                                no_rt_limit=no_rt_limit,
                                # §PID: PhaseInteractionDenker-Plan — UV3 als reiner Executor
                                precomputed_phase_plan=_pid_phase_plan,
                            )
                        )
                    except Exception as _e:
                        _err_box.append(_e)
                    finally:
                        try:
                            _set_pipeline_active(False)
                        except Exception as _exc:
                            logger.debug("Operation failed (non-critical): %s", _exc)

                _t = threading.Thread(target=_run_rest, daemon=True)
                _t.start()
                if no_rt_limit:
                    _t.join()
                else:
                    _t.join(timeout=_remaining)
                    if _t.is_alive():
                        raise RuntimeError(
                            "RestaurierDenker überschritt RT-Budget "
                            f"({_remaining:.1f}s für {audio_duration_s:.1f}s Audio)"
                        )
                if _err_box:
                    raise _err_box[0]  # type: ignore[misc]
                rest = _result_box[0]
                if isinstance(rest, np.ndarray):
                    rest = SimpleNamespace(
                        audio=np.asarray(rest, dtype=np.float32),
                        phases_executed=[],
                        warnings=[],
                        quality_estimate=0.0,
                        rt_factor=0.0,
                        confidence=0.85,
                        rollback_triggered=False,
                        winning_variant=None,
                        musical_goals={},
                        goals_passed=0,
                        metadata={},
                    )
                aktuelles_audio = rest.audio
                phases_executed.extend(rest.phases_executed or [])
                warnings.extend(rest.warnings or [])
                # A-2: ARE-Metadaten propagieren
                _rest_confidence = float(getattr(rest, "confidence", 0.85))
                _rest_rollback = bool(getattr(rest, "rollback_triggered", False))
                _rest_variant = getattr(rest, "winning_variant", None)
                _raw_goals = getattr(rest, "musical_goals", None)
                _rest_musical_goals = dict(_raw_goals) if _raw_goals else {}
                _rest_goals_passed = int(getattr(rest, "goals_passed", 0))
                _meta_raw = getattr(rest, "metadata", None)
                _rest_metadata = dict(_meta_raw) if isinstance(_meta_raw, dict) else {}
                # Keep a single canonical material label across UV3 scorecards and
                # AurikDenker final logs/exports.
                _rest_mat_raw = getattr(rest, "material_type", None)
                _rest_mat = str(getattr(_rest_mat_raw, "value", _rest_mat_raw) or "").strip().lower()
                if _rest_mat and _rest_mat not in {"unknown", "materialtype.unknown"}:
                    material = _rest_mat
                else:
                    _meta_mat = (
                        str(
                            ((_rest_metadata.get("defect_analysis") or {}).get("material") if _rest_metadata else "")
                            or ""
                        )
                        .strip()
                        .lower()
                    )
                    if _meta_mat and _meta_mat not in {"unknown", "materialtype.unknown"}:
                        material = _meta_mat
                # §Dach-Enrichment: era_decade aus RestorationResult in Globalplan übernehmen
                # (ML-Klassifikatoren liefen in UV3 — jetzt Ergebnis in Plan einpflegen)
                if _globalplan is not None:
                    _era_from_pipeline = getattr(rest, "era_decade", None)
                    if _era_from_pipeline is not None:
                        try:
                            _globalplan.portrait.decade = int(_era_from_pipeline)
                            _globalplan.portrait.era_confidence = max(_globalplan.portrait.era_confidence, 0.75)
                            _globalplan.reasoning_trace.append(
                                f"Enriched with pipeline era_decade={_era_from_pipeline} (UV3 ML)"
                            )
                        except Exception as _exc:
                            logger.debug("Operation failed (non-critical): %s", _exc)
                # [6/10] ReparaturDenker-Ergebnis (Preprocessing-Schritt)
                if _rep_result_box:
                    _rep = _rep_result_box[0]
                    warnings.extend(getattr(_rep, "warnings", []) or [])
                    stage_notes["reparatur"] = (
                        f"Clicks: {getattr(_rep, 'clicks_removed', False)}, "
                        f"Hum: {getattr(_rep, 'hum_removed', False)}, "
                        f"Clipping: {getattr(_rep, 'clipping_repaired', False)} (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [6/10] Reparatur (Pre-UV3): clicks=%s hum=%s clipping=%s",
                        getattr(_rep, "clicks_removed", False),
                        getattr(_rep, "hum_removed", False),
                        getattr(_rep, "clipping_repaired", False),
                    )
                # [7/10] RekonstruktionsDenker-Ergebnis (Preprocessing-Schritt)
                if _rek_result_box:
                    _rek = _rek_result_box[0]
                    warnings.extend(getattr(_rek, "warnings", []) or [])
                    _gaps_found = int(getattr(_rek, "gaps_found", 0))
                    _gaps_repaired = int(getattr(_rek, "gaps_repaired", 0))
                    _gap_total_ms = float(getattr(_rek, "total_repaired_ms", 0.0))
                    stage_notes["rekonstruktion"] = (
                        f"Lücken gefunden: {_gaps_found}, "
                        f"repariert: {_gaps_repaired}/{_gap_total_ms:.1f} ms (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [7/10] Rekonstruktion (Pre-UV3): %d/%d Lücken, %.1f ms",
                        _gaps_repaired,
                        _gaps_found,
                        _gap_total_ms,
                    )
                stage_notes["restaurierung"] = (
                    f"Qualität: {rest.quality_estimate:.3f}, "
                    f"RT: {rest.rt_factor:.2f}×, "
                    f"Konfidenz: {_rest_confidence:.2f}"
                )
                logger.info(
                    "AurikDenker [8/10] Restaurierung: Q=%.3f, RT=%.2f×, K=%.2f",
                    rest.quality_estimate,
                    rest.rt_factor,
                    _rest_confidence,
                )
            except Exception as exc:
                _record_stage_failure("restaurierung", "RestaurierDenker", exc)
                logger.warning("AurikDenker [8/10] RestaurierDenker: %s", exc)
        else:
            stage_notes["restaurierung"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Restaurierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [8/10] Budget erschöpft, Restaurierung übersprungen")

        # ── Stufe 7: Exzellenz-Optimierung + Musical Goals ───────────────────
        _emit(95, "Musikalische Exzellenz wird optimiert …")
        musical_goals: dict[str, float] = dict(_rest_musical_goals)
        goals_passed: int = _rest_goals_passed
        excellence_score = 0.0
        _exz_versa_mos: float = 0.0  # M-8b: aus ExzellenzDenker gecachter VERSA-Score
        _exz_material = self._resolve_excellence_material(material, chain_info)
        _skip_excellence, _skip_metrics = self._should_skip_excellence_for_clean_digital(
            aktuelles_audio,
            sr,
            material,
            chain_info,
        )

        if _rest_rollback:
            # ARE rollback means restoration degraded quality — ExzellenzDenker
            # on unrestored audio is wasteful and violates Autopilot rules.
            stage_notes["exzellenz"] = (
                "Übersprungen (ARE-Rollback: Restaurierung verschlechterte Qualität, "
                "ExzellenzDenker auf unrestauriertem Audio nicht sinnvoll)"
            )
            warnings.append(
                "ExzellenzDenker übersprungen: ARE-Rollback aktiv — Restaurierung hat Qualität verschlechtert"
            )
            logger.warning(
                "AurikDenker [9/10] Exzellenz übersprungen: ARE-Rollback aktiv "
                "(restoration degraded quality, excellence on unrestored audio skipped)",
            )
        elif _skip_excellence:
            _skip_mat = _skip_metrics.get("material", _exz_material)
            stage_notes["exzellenz"] = (
                f"Übersprungen (saubere Digitalquelle: {_skip_mat}, "
                "Autopilot priorisiert Originaltreue vor Overprocessing)"
            )
            logger.info(
                "AurikDenker [9/10] Exzellenz übersprungen für saubere Digitalquelle: %s",
                _skip_metrics,
            )
        elif _budget_ok():
            # v9.10.72: STFT-basierte ExcellenceOptimizer-Passe nach UV3+FeedbackChain
            # deaktiviert (Ephraim & Malah 1984: kaskadierte STFT-Modifikation akkumuliert
            # Rundungsfehler; ML-Modelle nicht auf eigenen Output trainiert → Domain Shift).
            # v9.11.1: messe_und_repariere() ersetzt messe_ziele() — nutzt ausschließlich
            # Zeit-Domain-Operationen (micro_dynamics, ola_edges) und linearen Blend mit
            # Original-Audio für P3-P5-Verletzungen. Kein STFT-Roundtrip → Ephraim-sicher.
            try:
                exd = get_exzellenz_denker()
                # Goals messen + konservative P3-P5-Reparatur (Zeit-Domain-first, §0 + §2.45).
                # Legacy-Fallback für Testmocks/ältere ExzellenzDenker mit messe_ziele().
                goals = {}
                _used_repair_path = False
                # §2.32 bass_kraft/brillanz-Fix: inapplicable Goals aus UV3-Metadata
                # extrahieren und an ExzellenzDenker weitergeben, damit er nur
                # physikalisch messbare Ziele repariert (z.B. bass_kraft auf
                # vocal-dominanter Quelle mit sehr geringem Bassanteil).
                _gaf_map: dict = _rest_metadata.get("goal_applicability", {})
                _inapplicable_goals: frozenset[str] = frozenset(
                    g for g, applicable in _gaf_map.items() if not applicable
                )

                _repair_fn = getattr(exd, "messe_und_repariere", None)
                if callable(_repair_fn):
                    _repair_call = cast(Callable[..., Any], _repair_fn)
                    _mr = _repair_call(
                        aktuelles_audio,
                        sr,
                        mode=effective_mode,
                        material=_exz_material,
                        reference_audio=audio,
                        inapplicable_goals=_inapplicable_goals if _inapplicable_goals else None,
                    )
                    if isinstance(_mr, tuple) and len(_mr) == 2:
                        aktuelles_audio = _mr[0]
                        goals = _normalize_goal_scores(_mr[1])
                        _used_repair_path = True

                if not _used_repair_path:
                    _legacy_fn = getattr(exd, "messe_ziele", None)
                    if not callable(_legacy_fn):
                        raise RuntimeError("ExzellenzDenker liefert weder messe_und_repariere() noch messe_ziele()")
                    _legacy_call = cast(Callable[..., Any], _legacy_fn)
                    _legacy_out = _legacy_call(aktuelles_audio, sr, material=_exz_material)
                    if isinstance(_legacy_out, tuple) and len(_legacy_out) == 2:
                        aktuelles_audio = _legacy_out[0]
                        goals = _normalize_goal_scores(_legacy_out[1])
                    else:
                        goals = _normalize_goal_scores(_legacy_out)
                if goals:
                    try:
                        _exz_thresholds = _get_canonical_thresholds_for_mode(
                            is_studio_2026=effective_mode == "studio2026"
                        )
                    except Exception:
                        _exz_thresholds = {}
                    _FALLBACK_GOAL_MIN = 0.75

                    finite_vals = [v for v in goals.values() if math.isfinite(v)]
                    excellence_score = float(np.mean(finite_vals)) if finite_vals else 0.0
                    goals_passed = sum(
                        1
                        for k, v in goals.items()
                        if math.isfinite(v) and v >= _exz_thresholds.get(k, _FALLBACK_GOAL_MIN)
                    )
                    musical_goals = _normalize_goal_scores(goals)
                # VERSA MOS für Qualitätsentscheidung
                try:
                    if aktuelles_audio.ndim == 1:
                        _mono_exz = aktuelles_audio
                    elif aktuelles_audio.ndim == 2:
                        # Robust mono downmix for both layouts:
                        # (channels, samples) -> axis=0, (samples, channels) -> axis=1.
                        _mono_exz = (
                            aktuelles_audio.mean(axis=0)
                            if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                            else aktuelles_audio.mean(axis=1)
                        )
                    else:
                        _mono_exz = np.ravel(aktuelles_audio)
                    _mono_exz = np.nan_to_num(_mono_exz.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    _vr_exz = _score_versa_mos(_mono_exz, sr)
                    _exz_versa_mos = float(_vr_exz.mos)
                except Exception as _ve_exz:
                    logger.debug("ExzellenzDenker VERSA nicht verfügbar: %s", _ve_exz)

                _goals_total = len(goals) if goals else 0
                stage_notes["exzellenz"] = (
                    f"Messung{' + Repair' if _used_repair_path else ''}: Score {excellence_score:.3f}, "
                    f"{goals_passed}/{_goals_total} Ziele erfüllt"
                    + (f", VERSA MOS={_exz_versa_mos:.2f}" if _exz_versa_mos > 0.0 else "")
                    + (
                        " (Zeit-Domain-Repair für P3-P5 — kein STFT-Re-Pass)"
                        if _used_repair_path
                        else " (Legacy-Goal-Messpfad)"
                    )
                )
                phases_executed.append("exzellenz_messung_repair" if _used_repair_path else "exzellenz_messung")
                logger.info(
                    "AurikDenker [9/10] Exzellenz: Score=%.3f, Goals %d/%d%s (%s)",
                    excellence_score,
                    goals_passed,
                    _goals_total,
                    f", VERSA={_exz_versa_mos:.3f}" if _exz_versa_mos > 0.0 else "",
                    "Goal-Repair aktiv" if _used_repair_path else "Legacy-Goal-Messung",
                )
            except Exception as exc:
                _record_stage_failure("exzellenz", "ExzellenzDenker", exc)
                logger.warning("AurikDenker [9/10] ExzellenzDenker: %s", exc)
        else:
            stage_notes["exzellenz"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Exzellenz-Optimierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [9/10] Exzellenz übersprungen")
            # Musical Goals + VERSA trotzdem messen (~7 s) — essentiell für UI-Anzeige
            # und Qualitätsurteil, auch wenn Optimierung übersprungen wird.
            try:
                MusicalGoalsChecker = _load_symbol(
                    "backend.core.musical_goals.musical_goals_metrics",
                    "MusicalGoalsChecker",
                )

                _mg_budget = MusicalGoalsChecker(mode=effective_mode)
                _budget_goals = _mg_budget.measure_all(aktuelles_audio, sr)
                if _budget_goals:
                    musical_goals = _normalize_goal_scores(_budget_goals)
                    _finite_bg = [v for v in musical_goals.values() if math.isfinite(v)]
                    goals_passed = sum(1 for v in _finite_bg if v >= 0.75)
                    excellence_score = float(np.mean(_finite_bg)) if _finite_bg else 0.0
                    logger.info(
                        "AurikDenker [9/10] Goals gemessen trotz Budget-Limit: %d/%d bestanden",
                        goals_passed,
                        len(_budget_goals),
                    )

                    # Budgetfreundlicher Goal-Recovery ohne zusätzliche DSP-Phasen:
                    # Falls kritische Goals schwach sind, mische einen kleinen Anteil
                    # Originalsignal zurück (P1/P2-Schutz), messe erneut und übernehme
                    # nur bei tatsächlicher Verbesserung.
                    _critical_goals = {
                        "natuerlichkeit",
                        "authentizitaet",
                        "tonal_center",
                        "timbre_authentizitaet",
                        "artikulation",
                    }
                    _crit_total = sum(1 for g in _budget_goals if g in _critical_goals)
                    _crit_pass_before = sum(
                        1 for g, v in _budget_goals.items() if g in _critical_goals and float(v) >= 0.75
                    )
                    if _crit_total > 0 and _crit_pass_before < _crit_total and aktuelles_audio.shape == audio.shape:
                        _best_audio = aktuelles_audio
                        _best_goals = dict(_budget_goals)
                        _best_pass = goals_passed
                        _best_score = excellence_score
                        _best_crit = _crit_pass_before

                        for _alpha in (0.92, 0.88, 0.84):
                            _candidate = np.clip(_alpha * aktuelles_audio + (1.0 - _alpha) * audio, -1.0, 1.0)
                            _cand_goals = _normalize_goal_scores(_mg_budget.measure_all(_candidate, sr))
                            _cand_finite = [v for v in _cand_goals.values() if math.isfinite(v)]
                            _cand_pass = sum(1 for v in _cand_finite if v >= 0.75)
                            _cand_score = float(np.mean(_cand_finite)) if _cand_finite else 0.0
                            _cand_crit = sum(
                                1 for g, v in _cand_goals.items() if g in _critical_goals and float(v) >= 0.75
                            )

                            _is_better = (_cand_crit > _best_crit) or (
                                _cand_crit == _best_crit
                                and (
                                    _cand_pass > _best_pass or (_cand_pass == _best_pass and _cand_score > _best_score)
                                )
                            )
                            if _is_better:
                                _best_audio = _candidate
                                _best_goals = dict(_cand_goals)
                                _best_pass = _cand_pass
                                _best_score = _cand_score
                                _best_crit = _cand_crit

                        if _best_crit > _crit_pass_before or _best_pass > goals_passed:
                            aktuelles_audio = _best_audio
                            musical_goals = _best_goals
                            goals_passed = _best_pass
                            excellence_score = _best_score
                            stage_notes["exzellenz"] += (
                                f"; Zielschutz-Blend angewendet (kritisch {_crit_pass_before}/{_crit_total} → "
                                f"{_best_crit}/{_crit_total}, gesamt {goals_passed}/{len(_best_goals)})"
                            )
                            logger.info(
                                "AurikDenker [9/10] Goal-Recovery-Blend aktiv: "
                                "kritische Goals %d/%d → %d/%d, gesamt=%d/%d",
                                _crit_pass_before,
                                _crit_total,
                                _best_crit,
                                _crit_total,
                                goals_passed,
                                len(_best_goals),
                            )
            except Exception as _mg_exc:
                logger.debug("Musical Goals Messung nach Budget-Limit: %s", _mg_exc)
            try:
                if aktuelles_audio.ndim == 1:
                    _mono_budget = aktuelles_audio
                elif aktuelles_audio.ndim == 2:
                    _mono_budget = (
                        aktuelles_audio.mean(axis=0)
                        if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                        else aktuelles_audio.mean(axis=1)
                    )
                else:
                    _mono_budget = np.ravel(aktuelles_audio)
                _mono_budget = np.nan_to_num(_mono_budget.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr_budget = _score_versa_mos(_mono_budget, sr)
                _exz_versa_mos = float(_vr_budget.mos)
                logger.info(
                    "AurikDenker [9/10] VERSA MOS gemessen trotz Budget-Limit: %.3f",
                    _exz_versa_mos,
                )
            except Exception as _ve_budget:
                logger.debug("VERSA MOS nach Budget-Limit: %s", _ve_budget)

        # ── Stufe 10: VERSA MOS — finales Qualitätsurteil (§4.4) ────────────
        _emit(97, "VERSA MOS-Qualitätsbewertung läuft …")
        # M-8b: VERSA-MOS-Cache aus ExzellenzDenker übernehmen um doppelte
        # Inferenz zu vermeiden. Nur wenn dort kein Score verfügbar, VERSA neu fahren.
        _versa_mos: float = _exz_versa_mos
        if _exz_versa_mos > 0.0:
            # ExzellenzDenker hat bereits VERSA auf optimiertem Audio gemessen
            stage_notes["versa_mos"] = f"MOS={_versa_mos:.3f} (ExzellenzDenker-Cache)"
            phases_executed.append("versa_qualitaetsbewertung")
            logger.info(
                "AurikDenker [10/10] VERSA MOS=%.3f (ExzellenzDenker-Cache) — %s",
                _versa_mos,
                (
                    "✓ Studioqualität"
                    if _versa_mos >= 4.3
                    else ("✓ Gute Qualität" if _versa_mos >= 3.5 else "⚠ Qualität unter Mindestniveau")
                ),
            )
            if _versa_mos < 3.5:
                warnings.append(f"VERSA MOS={_versa_mos:.2f} < 3.5 — Klangqualität unter Mindestniveau")
        elif _budget_ok():
            try:
                if aktuelles_audio.ndim == 1:
                    _mono_final = aktuelles_audio
                elif aktuelles_audio.ndim == 2:
                    _mono_final = (
                        aktuelles_audio.mean(axis=0)
                        if aktuelles_audio.shape[0] <= 2 < aktuelles_audio.shape[1]
                        else aktuelles_audio.mean(axis=1)
                    )
                else:
                    _mono_final = np.ravel(aktuelles_audio)
                _mono_final = np.nan_to_num(_mono_final.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr = _score_versa_mos(_mono_final, sr)
                _versa_mos = float(_vr.mos)
                stage_notes["versa_mos"] = f"MOS={_versa_mos:.3f} ({_vr.model_used})"
                phases_executed.append("versa_qualitaetsbewertung")
                logger.info(
                    "AurikDenker [10/10] VERSA MOS=%.3f — %s",
                    _versa_mos,
                    (
                        "✓ Studioqualität"
                        if _versa_mos >= 4.3
                        else ("✓ Gute Qualität" if _versa_mos >= 3.5 else "⚠ Qualität unter Mindestniveau")
                    ),
                )
                if _versa_mos < 3.5:
                    warnings.append(f"VERSA MOS={_versa_mos:.2f} < 3.5 — Klangqualität unter Mindestniveau")
            except Exception as _ve:
                logger.debug("AurikDenker [10/10] VERSA MOS nicht verfügbar: %s", _ve)
        # §Spec VERSA MOS-Gate: material-adaptiver Schwellwert (§6.2)
        # Digital ≥ 4.5, Tape ≥ 4.2, Vinyl ≥ 4.0, Shellac ≥ 3.8, Default 4.0
        _MATERIAL_MOS_GATE: dict[str, float] = {
            "cd_digital": 4.5,
            "dat": 4.5,
            "mp3_high": 4.5,
            "aac": 4.5,
            "tape": 4.2,
            "reel_tape": 4.2,
            "cassette": 4.2,
            "vinyl": 4.0,
            "shellac": 3.8,
        }
        _mos_gate_target = _MATERIAL_MOS_GATE.get(_exz_material, 4.0)
        if 0.0 < _versa_mos < _mos_gate_target:
            # v9.10.58: 2. ExzellenzDenker-Aufruf entfernt — wissenschaftlich nicht gerechtfertigt.
            # Der ExzellenzDenker (Stufe 7) hat bereits 1× ExcellenceOptimizer + 1× Re-Pass
            # ausgeführt. Ein erneuter identischer Durchlauf auf demselben Audio erzeugt
            # kumulative STFT-Artefakte und Domain-Shift bei ML-Modellen.
            # VERSA MOS dient hier nur als Qualitätssignal für Metadaten/Logging.
            logger.warning(
                "AurikDenker [10/10] VERSA MOS=%.3f < Ziel %.1f für %s — "
                "Qualität unter Material-Schwellwert (Korrektur bereits in Stufe 7 erfolgt)",
                _versa_mos,
                _mos_gate_target,
                _exz_material,
            )
            warnings.append(
                f"VERSA MOS={_versa_mos:.2f} < Material-Ziel {_mos_gate_target:.1f} "
                f"({_exz_material}) — physikalische Grenze des Quellmaterials"
            )

        # ── RAM-Cleanup nach Pipeline ────────────────────────────────────────
        # PluginLifecycleManager entlädt inaktive ML-Modelle wenn RAM knapp ist.
        # Kein Force-Evict hier (Batch-Cleanup erfolgt im Aufrufer via
        # cleanup_after_file()), nur druckbasiertes Evict.
        _emit(98, "RAM-Management …")
        try:
            _n_evicted = _evict_stale_plugins()
            if _n_evicted > 0:
                stage_notes["ram_cleanup"] = f"{_n_evicted} Plugin(s) aus RAM entladen"
        except Exception as _plm_err:
            logger.debug("PLM-Evict nach Pipeline: %s", _plm_err)

        # ── Abschluss: RT-Faktor berechnen ───────────────────────────────────
        elapsed = time.perf_counter() - t_start
        _rt_raw = elapsed / max(audio_duration_s, 1e-6)
        if _rt_raw > _3X_RT_LIMIT:
            logger.warning(
                "AurikDenker: 32×RT-Limit überschritten (roh=%.2f×, gecappt=%.2f×). "
                "Lösung: FAST/BALANCED nutzen oder adaptive Skips erhöhen.",
                _rt_raw,
                _3X_RT_LIMIT,
            )
        rt_factor = min(_rt_raw, _3X_RT_LIMIT)  # Spec §9.5: niemals > 32.0

        # Qualitätsschätzung nach Spec §8.1 (normative Formel):
        # quality_estimate = 0.40*(1-defect_severity) + 0.60*(pqs_mos-1)/4
        # VERBOTEN: quality_estimate * 1.15 als fixer Bonus-Faktor
        _defect_sev_final = float(getattr(defekt, "overall_severity", 0.0)) if defekt is not None else 0.0
        _defect_sev_final = max(0.0, min(1.0, _defect_sev_final))
        if _versa_mos > 0.0:
            # VERSA MOS als pqs_mos-Proxy (kalibriert, Pearson=0.74 vs PQS-Gammatone)
            _mos_norm = float(np.clip((_versa_mos - 1.0) / 4.0, 0.0, 1.0))
            quality_estimate = 0.40 * (1.0 - _defect_sev_final) + 0.60 * _mos_norm
            quality_estimate = float(np.clip(quality_estimate, 0.0, 1.0))
        elif excellence_score > 0.0:
            # Kein VERSA verfügbar — excellence_score [0..1] auf MOS-Skala [1..5] skalieren
            # Normative Formel §8.1: 0.60 * (pqs_mos - 1) / 4; excellence_score ≡ (mos-1)/4
            quality_estimate = float(np.clip(0.40 * (1.0 - _defect_sev_final) + 0.60 * excellence_score, 0.0, 1.0))
        else:
            # DSP fallback: normative formula (§8.1) with defect_severity;
            # neutral mos_proxy = 0.55 corresponds to MOS ≈ 3.2 when no scorer available
            quality_estimate = float(np.clip(0.40 * (1.0 - _defect_sev_final) + 0.60 * 0.55, 0.0, 1.0))

        # Structured degradation signaling for downstream gates and UI transparency.
        _failed_stage_details: dict[str, str] = {
            str(entry.get("stage", "")): str(entry.get("exc_msg", ""))
            for entry in _stage_fail_reasons
            if isinstance(entry, dict)
        }
        if not _failed_stage_details:
            _failed_stage_details = {
                key: value.split("Fehler:", 1)[1].strip()
                for key, value in stage_notes.items()
                if isinstance(value, str) and value.startswith("Fehler:")
            }
        if _failed_stage_details:
            _failed_stages = sorted(_failed_stage_details.keys())
            stage_notes["fail_reasons"] = list(_stage_fail_reasons)
            _derived_health = pipeline_health_from_fail_reasons(_stage_fail_reasons)
            if _derived_health == PipelineHealthState.OK:
                _critical_failed = sorted(stage for stage in _failed_stages if stage in _critical_stages)
                _degradation_status = (
                    PipelineHealthState.CRITICAL_DEGRADED.value
                    if _critical_failed
                    else PipelineHealthState.DEGRADED.value
                )
            else:
                _degradation_status = _derived_health.value
                _critical_failed = sorted(
                    str(entry.get("stage", ""))
                    for entry in _stage_fail_reasons
                    if str(entry.get("severity", "")).strip().lower() in {"critical", "critical_degraded", "blocked"}
                )
            stage_notes["degradation_status"] = _degradation_status
            stage_notes["degradation_failures"] = ",".join(_failed_stages)
            stage_notes["degradation_details"] = "; ".join(
                f"{stage}={_failed_stage_details[stage]}" for stage in _failed_stages
            )
            if _stage_fail_reasons:
                stage_notes["degradation_error_codes"] = ",".join(
                    sorted(
                        {str(entry.get("error_code", "")) for entry in _stage_fail_reasons if entry.get("error_code")}
                    )
                )
            if _critical_failed:
                _fail_reason = f"critical_stage_failure:{','.join(_critical_failed)}"
                stage_notes["fail_reason"] = _fail_reason
                # Proportional penalty: scale down by number of critical failures
                # 1 critical failure → ×0.85, 2 → ×0.72, 3+ → ×0.55 (near export gate)
                _penalty = max(0.55, 0.85 ** len(_critical_failed))
                quality_estimate = min(float(quality_estimate), float(quality_estimate) * _penalty)
                warnings.append(
                    "Kritische Stufen ausgefallen: "
                    f"{', '.join(_critical_failed)}. Qualitätsausgabe wurde proportional begrenzt."
                )
        else:
            stage_notes["degradation_status"] = PipelineHealthState.OK.value

        # Material-adaptives MOS-Gate (zentral im Orchestrator).
        # Falls VERSA-MOS unter Materialziel liegt, erzwingt ein Clamp
        # quality_estimate < 0.55, sodass Export-Gates konsistent blockieren.
        _mos_material, _mos_target = self._material_mos_target(material, chain_info)
        stage_notes["material_mos_gate"] = f"{_mos_material}: MOS-Ziel≥{_mos_target:.2f}"
        if 0.0 < _versa_mos < _mos_target:
            _msg = (
                f"Material-MOS-Gate nicht bestanden: {_mos_material} "
                f"(VERSA MOS={_versa_mos:.2f} < Ziel {_mos_target:.2f}). "
                "Lösung: defensivere Kette/fallback oder Pass-Through für saubere Digitalquellen."
            )
            warnings.append(_msg)
            stage_notes["material_mos_gate"] = f"FAILED: {_mos_material} MOS={_versa_mos:.3f} < {_mos_target:.3f}"
            if mode.lower() in {"restoration", "studio2026", "maximum"}:
                # Proportional penalty: the closer MOS is to target, the less penalty.
                # MOS-Deficit ratio: 0 → no penalty, 1.0+ → cap at 0.54
                _mos_deficit_ratio = min(1.0, max(0.0, (_mos_target - _versa_mos) / _mos_target))
                _qe_penalty = 1.0 - 0.46 * _mos_deficit_ratio  # [0.54 .. 1.0]
                quality_estimate = min(float(quality_estimate), float(quality_estimate) * _qe_penalty)

        # Finale NaN/Inf-Bereinigung und Clip
        aktuelles_audio = np.nan_to_num(aktuelles_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        aktuelles_audio = np.clip(aktuelles_audio, -1.0, 1.0)

        _total_elapsed = time.perf_counter() - t_start
        logger.info(
            "AurikDenker.denke() abgeschlossen: Q=%.3f, RT=%.2f×, Phasen=%d, "
            "Material=%s, MOS=%.2f, Dauer=%.1fs, Warnungen=%d",
            quality_estimate,
            rt_factor,
            len(phases_executed),
            material,
            _versa_mos,
            _total_elapsed,
            len(warnings),
        )

        note = (
            f"Aurik 9 Restaurierung abgeschlossen: "
            f"Material={material}, RT={rt_factor:.2f}×, "
            f"Qualität={quality_estimate:.3f}"
            + (f", VERSA MOS={_versa_mos:.2f}" if _versa_mos > 0.0 else "")
            + f", Phasen={len(phases_executed)}"
        )

        return AurikErgebnis(
            audio=aktuelles_audio,
            material=material,
            rt_factor=rt_factor,
            quality_estimate=quality_estimate,
            musical_goals=musical_goals,
            goals_passed=goals_passed,
            phases_executed=phases_executed,
            warnings=warnings,
            processing_note=note,
            stage_notes=stage_notes,
            chain_info=chain_info,
            confidence=_rest_confidence,
            rollback_triggered=_rest_rollback,
            winning_variant=_rest_variant,
            gaps_found=_gaps_found,
            gaps_repaired=_gaps_repaired,
            gap_total_repaired_ms=_gap_total_ms,
            degradation_status=_degradation_status,
            fail_reason=_fail_reason,
            global_plan=_globalplan.as_dict() if _globalplan is not None else None,
            metadata=_rest_metadata,
        )

    # ── Fallback ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(
        audio: np.ndarray,
        rt_factor: float = 0.0,
        grund: str = "Unbekannter Fehler",
    ) -> AurikErgebnis:
        """Fallback-Ergebnis: gibt ursprüngliches Audio zurück."""
        clean = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, -1.0, 1.0)
        return AurikErgebnis(
            audio=clean,
            material="unknown",
            rt_factor=min(rt_factor, _3X_RT_LIMIT),
            quality_estimate=0.0,
            musical_goals={},
            goals_passed=0,
            phases_executed=[],
            warnings=[grund],
            processing_note=f"Restaurierung fehlgeschlagen: {grund}",
            stage_notes={
                "fehler": grund,
                "degradation_status": PipelineHealthState.BLOCKED.value,
                "fail_reason": f"pipeline_blocked:{grund}",
                "fail_reasons": [
                    {
                        "component": "AurikDenker",
                        "stage": "fallback",
                        "error_code": "PIPELINE_BLOCKED",
                        "severity": "blocked",
                        "exc_type": "Fallback",
                        "exc_msg": str(grund),
                    }
                ],
            },
            chain_info=None,
            degradation_status=PipelineHealthState.BLOCKED.value,
            fail_reason=f"pipeline_blocked:{grund}",
            metadata={},
        )


# ─── Modul-Level-Singleton (Double-Checked Locking — Spec §3.2) ─────────────

_singleton_state = SimpleNamespace(instance=None)
_lock = threading.Lock()


def get_aurik_denker() -> AurikDenker:
    """Thread-sicherer Singleton-Accessor für den AurikDenker-Orchestrator."""
    if _singleton_state.instance is None:
        with _lock:
            if _singleton_state.instance is None:
                _singleton_state.instance = AurikDenker()
    return cast(AurikDenker, _singleton_state.instance)


def restauriere(audio: np.ndarray, sr: int = 48_000) -> AurikErgebnis:
    """Convenience-Funktion: Vollständige Restaurierung über Aurik-Singleton.

    Args:
        audio: Audio-Signal (mono/stereo, float32)
        sr:    Sample-Rate in Hz (Standard 48 000)

    Returns:
        AurikErgebnis mit restauriertem Audio, RT-Faktor und Qualitäts-Metriken.
    """
    return get_aurik_denker().restauriere(audio, sr)


def denke(audio: np.ndarray, sr: int = 48_000, **kwargs: Any) -> AurikErgebnis:
    """Convenience-Wrapper: alias für get_aurik_denker().denke().

    Args:
        audio:  Audio-Signal (mono/stereo, float32)
        sr:     Sample-Rate in Hz (Standard 48 000)
        **kwargs: Weitergabe an AurikDenker.denke()

    Returns:
        AurikErgebnis mit restauriertem Audio und allen Metriken.
    """
    return get_aurik_denker().denke(audio, sr, **kwargs)


def get_tontraeger_denker() -> Any:
    """Modul-Level-Accessor für TontraegerDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.tontraeger_denker", "get_tontraeger_denker")()


def get_defekt_denker() -> Any:
    """Modul-Level-Accessor für DefektDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.defekt_denker", "get_defekt_denker")()


def get_tontraegerkette_denker() -> Any:
    """Modul-Level-Accessor für TontraegerketteDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.tontraegerkette_denker", "get_tontraegerkette_denker")()


def get_strategie_denker() -> Any:
    """Modul-Level-Accessor für StrategieDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.strategie_denker", "get_strategie_denker")()


def get_restaurier_denker() -> Any:
    """Modul-Level-Accessor für RestaurierDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.restaurier_denker", "get_restaurier_denker")()


def get_reparatur_denker() -> Any:
    """Modul-Level-Accessor für ReparaturDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.reparatur_denker", "get_reparatur_denker")()


def get_rekonstruktions_denker() -> Any:
    """Modul-Level-Accessor für RekonstruktionsDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.rekonstruktions_denker", "get_rekonstruktions_denker")()


def get_exzellenz_denker() -> Any:
    """Modul-Level-Accessor für ExzellenzDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    return _load_symbol("denker.exzellenz_denker", "get_exzellenz_denker")()


def get_phase_interaction_denker() -> Any:
    """Modul-Level-Accessor für PhaseInteractionDenker (patchbar in Tests)."""
    return _load_symbol("denker.phase_interaction_denker", "get_phase_interaction_denker")()


def erstelle_globalplan(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper for the MusikalischerGlobalplan entry point."""
    return _load_symbol("backend.core.musikalischer_globalplan", "erstelle_globalplan")(*args, **kwargs)
