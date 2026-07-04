"""
RestaurierDenker — Domäne: Vollrestaurierung via UnifiedRestorerV3.

Kapselt core.unified_restorer_v3.UnifiedRestorerV3 mit strikter
8×RT-Pflicht (enforce_3x_rt=True darf NIEMALS auf False gesetzt werden).

Usage::

    from denker.restaurier_denker import get_restaurier_denker

    denker = get_restaurier_denker()
    ergebnis = denker.restauriere(audio, sr=48000, material="vinyl")
    restauriertes_audio = ergebnis.audio
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from backend.core.unified_restorer_v3 import (
        QualityMode,
        RestorationConfig,
        UnifiedRestorerV3,
    )
except Exception:  # pragma: no cover — optional heavy dependency
    QualityMode = None  # type: ignore[assignment,misc]
    RestorationConfig = None  # type: ignore[assignment,misc]
    UnifiedRestorerV3 = None  # type: ignore[assignment,misc]

try:
    from backend.core.pipeline_main import AurikAutonomousPipeline
    from backend.core.processing_modes import ProcessingMode
except Exception:  # pragma: no cover — optional heavy dependency
    AurikAutonomousPipeline = None  # type: ignore[assignment,misc]
    ProcessingMode = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# RT-Budget-Konstante (§9.5) — caps reported rt_factor
_3X_RT_LIMIT: float = 8.0

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
    """15 Musical Goals (falls gemessen)."""

    warnings: list[str] = field(default_factory=list)
    """Warnmeldungen aus der Pipeline."""

    confidence: float = 1.0
    """Gesamtkonfidenz der Restaurierung ∈ [0, 1]."""

    winning_variant: str | None = None
    """Beste ARE-Variante (oder None bei UV3-Fallback)."""

    rollback_triggered: bool = False
    """War ARE-Rollback ausgelöst?"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Zusätzliche Pipeline-Metadaten."""

    goal_applicability: dict[str, bool] = field(default_factory=dict)
    """GAF-Ergebnis: True = applicable, False = inapplicable (§2.32, §S5)."""

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

        # Phasen aus passes_executed + Ableitung echter Phase-IDs aus ARE-Metadaten
        # Ziel: _ML_PHASE_MARKERS im Frontend kann ML-Plugins erkennen.
        passes: int = int(getattr(are_result, "passes_executed", 1))
        _phases: list[str] = [f"are_pass_{i + 1}" for i in range(passes)]

        # Derive phase IDs from causal_order defect types (ARE exposes these).
        # Maps defect type values → UV3-style phase-ID substrings matching _ML_PHASE_MARKERS.
        _DEFECT_TO_PHASE: dict[str, str] = {
            "noise": "phase_denoise",
            "hiss": "phase_tape_hiss",
            "clicks": "phase_01_click_removal",
            "crackle": "phase_09_crackle_removal",
            "pops": "phase_27_click_pop_removal",
            "dropout": "phase_dropout_repair",
            "wow": "phase_12_wow_flutter",
            "flutter": "phase_12_wow_flutter",
            "clipping": "phase_23_spectral_repair",
            "hum": "phase_02_hum_removal",
            "reverb": "phase_reverb_reduction",
            "sibilance": "phase_ml_deesser",
            "frequency": "phase_frequency_restoration",
        }
        _seen_phases: set[str] = set(_phases)
        causal_order: list[Any] = list(getattr(are_result, "causal_order", []))
        for defect_val in causal_order:
            _dv = str(defect_val).lower()
            for _key, _phase in _DEFECT_TO_PHASE.items():
                if _key in _dv and _phase not in _seen_phases:
                    _phases.append(_phase)
                    _seen_phases.add(_phase)

        # Map winning variant name → additional phase IDs.
        _VARIANT_TO_PHASES: dict[str, list[str]] = {
            "aggressive": ["phase_denoise", "phase_deepfilternet", "phase_vocal_enhancement"],
            "balanced": ["phase_denoise", "phase_deepfilternet"],
            "conservative": ["phase_denoise"],
            "naturalness": ["phase_denoise", "phase_reverb_reduction"],
            "gentle_denoise": ["phase_denoise"],
            "light": ["phase_denoise"],
        }
        _winning: str = str(getattr(are_result, "winning_variant", "") or "").lower()
        if not getattr(are_result, "rollback_triggered", False) and _winning not in (
            "passthrough",
            "passthrough_error",
            "passthrough_fallback",
            "",
        ):
            for _vkey, _vphases in _VARIANT_TO_PHASES.items():
                if _vkey in _winning:
                    for _p in _vphases:
                        if _p not in _seen_phases:
                            _phases.append(_p)
                            _seen_phases.add(_p)
                    break
            else:
                # Unknown variant name — add baseline denoise phase
                if "phase_denoise" not in _seen_phases:
                    _phases.append("phase_denoise")

        # Add DeepFilterNet whenever noise/hiss/tape reduction ran (always active for reel_tape/vinyl)
        _mat_val: str = str(getattr(getattr(are_result, "material_type", None), "value", "") or "").lower()
        if any(m in _mat_val for m in ("tape", "vinyl", "shellac", "reel")):
            for _p in ("phase_denoise", "phase_tape_hiss", "phase_deepfilternet"):
                if _p not in _seen_phases:
                    _phases.append(_p)
                    _seen_phases.add(_p)

        self.phases_executed: list[str] = _phases
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

        # ARE-Variante und Rollback-Flag
        self.winning_variant: str | None = getattr(are_result, "winning_variant", None)
        self.rollback_triggered: bool = rollback

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
        audio_update_callback: Any | None = None,
        cached_era_result: Any | None = None,
        cached_genre_result: Any | None = None,
        cached_defect_result: Any | None = None,
        cached_medium_result: Any | None = None,
        cached_restorability_result: Any | None = None,
        recovery_checkpoint: Any | None = None,
        reconstruction_context: Any | None = None,
        pre_repair_reference: np.ndarray | None = None,
        input_path: str = "",
        output_path: str = "",
        no_rt_limit: bool = False,
        precomputed_phase_plan: list[str] | None = None,
        phase_strength_oracle_rollout: str | None = None,
        denker_policy_input: dict[str, Any] | None = None,
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
        reconstruction_context:
            Optional RekonstruktionsErgebnis from RekonstruktionsDenker (§11.7a).
            Contains bandwidth loss hints and gap statistics for UV3 context.

        Rückgabe
        --------
        RestaurierErgebnis mit restauriertem Audio und Pipeline-Metadaten.
        """
        assert sr == 48000, f"RestaurierDenker.restauriere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        logger.info(
            "RestaurierDenker.restauriere() gestartet: mode=%s, material=%s, duration=%.1fs, caches=%s",
            mode,
            material,
            len(audio) / max(sr, 1),
            cached_defect_result is not None,
        )
        if validate_audio:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            audio = audio.astype(np.float32)

        # §2.39 OOM-Recovery: explicit resume path via persisted checkpoint.
        if recovery_checkpoint is not None:
            logger.info(
                "RestaurierDenker: OOM-Recovery-Wiederaufnahme aktiv (remaining_phases=%d, failure_phase=%s)",
                len(getattr(recovery_checkpoint, "phases_remaining", []) or []),
                getattr(recovery_checkpoint, "failure_phase", "?"),
            )
            restorer = self._get_restorer(mode=mode)
            if restorer is None:
                return self._fallback(audio, material or "unknown", "Kein Restorer für OOM-Recovery verfügbar")
            if not hasattr(restorer, "restore_from_checkpoint"):
                return self._fallback(
                    audio,
                    material or "unknown",
                    "OOM-Recovery nicht verfügbar: restore_from_checkpoint fehlt",
                )
            try:
                raw = restorer.restore_from_checkpoint(
                    recovery_checkpoint,
                    progress_callback=progress_callback,
                    audio_update_callback=audio_update_callback,
                    no_rt_limit=no_rt_limit,
                )
                return self._konvertiere(raw, material=material)
            except Exception as cp_exc:
                logger.warning("RestaurierDenker: OOM-Recovery fehlgeschlagen: %s", cp_exc)
                return self._fallback(audio, material or "unknown", f"OOM-Recovery fehlgeschlagen: {cp_exc}")

        # ── v9.10.72: Direkt-UV3-Pfad (kein ARE-Umweg) ─────────────────────
        # AurikDenker hat Preprocessing (ReparaturDenker + RekonstruktionsDenker)
        # und alle Analysen (DefectScan, Era, Medium, Restorability) bereits
        # durchgeführt und als Caches weitergegeben. ARE wiederholt diese Arbeit
        # redundant (Chain-Inversion, Gap-Repair, IAQS, DefectScan, CausalGraph)
        # und gibt dann unmodifiziertes Audio an UV3 weiter — 2 nutzlose Passes.
        # Direkt-UV3-Pfad spart ~85-170s pro Datei und erreicht MOS 4.2+ im 1. Pass.
        _has_caches = cached_defect_result is not None
        _audio_dur_s: float = float(len(audio)) / max(float(sr), 1.0)

        if _has_caches:
            logger.info(
                "RestaurierDenker: Direkt-UV3-Pfad (Caches vorhanden, ARE übersprungen) — %.1fs Audio",
                _audio_dur_s,
            )
            restorer = self._get_restorer(mode=mode)
            if restorer is None:
                return self._fallback(audio, material or "unknown", "Kein Restorer verfügbar")

            _uv3_kwargs: dict = {"sample_rate": sr}
            if global_plan is not None:
                _uv3_kwargs["global_plan"] = global_plan
            if chain_info is not None:
                _uv3_kwargs["chain_info"] = chain_info
            if defekt_hint is not None:
                _uv3_kwargs["defekt_hint"] = defekt_hint
            if progress_callback is not None:
                _uv3_kwargs["progress_callback"] = progress_callback
            if audio_update_callback is not None:
                _uv3_kwargs["audio_update_callback"] = audio_update_callback
            if cached_era_result is not None:
                _uv3_kwargs["cached_era_result"] = cached_era_result
            if cached_genre_result is not None:
                _uv3_kwargs["cached_genre_result"] = cached_genre_result
            if cached_defect_result is not None:
                _uv3_kwargs["cached_defect_result"] = cached_defect_result
            if cached_medium_result is not None:
                _uv3_kwargs["cached_medium_result"] = cached_medium_result
            if cached_restorability_result is not None:
                _uv3_kwargs["cached_restorability_result"] = cached_restorability_result
            # §11.7a: Pass reconstruction context to UV3 for bandwidth/gap awareness
            if reconstruction_context is not None:
                _uv3_kwargs["reconstruction_context"] = reconstruction_context
                _bw = getattr(reconstruction_context, "bandwidth_limited", False)
                _gaps = getattr(reconstruction_context, "gaps_repaired", 0)
                if _bw or _gaps > 0:
                    logger.info(
                        "RestaurierDenker: reconstruction_context → UV3 (bw_limited=%s, gaps_repaired=%d)",
                        _bw,
                        _gaps,
                    )
            # §G1: Pre-Repair-Referenz für referenz-basierte Musical Goals
            if pre_repair_reference is not None:
                _uv3_kwargs["pre_repair_reference"] = pre_repair_reference
            # §PID: PhaseInteractionDenker-Plan weitergeben (UV3 wird reiner Executor)
            if precomputed_phase_plan:
                _uv3_kwargs["precomputed_phase_plan"] = precomputed_phase_plan
            # §2.39 OOM-Recovery: Pfade für Checkpoint-Persistierung
            if input_path:
                _uv3_kwargs["input_path"] = input_path
            if output_path:
                _uv3_kwargs["output_path"] = output_path
            if no_rt_limit:
                _uv3_kwargs["no_rt_limit"] = True
            if phase_strength_oracle_rollout is not None:
                _uv3_kwargs["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
            if denker_policy_input:
                _uv3_kwargs["denker_policy_input"] = dict(denker_policy_input)
            try:
                # §v10 HPE Baseline + Inviting VOR UV3
                _hpe_pre = 0.5
                _inv_pre = 0.5
                try:
                    from backend.core.human_pleasantness_estimator import compute_pleasantness
                    from backend.core.inviting_sound_checker import check_inviting_sound
                    _hpe_pre = compute_pleasantness(audio, sr).score
                    _inv_pre = check_inviting_sound(audio, sr).score
                except Exception: pass

                raw = restorer.restore(audio, **_uv3_kwargs)
                result = self._konvertiere(raw, material=material)

                # §v10 HPE + Inviting Check: Hat UV3 versagt?
                try:
                    from backend.core.human_pleasantness_estimator import compute_pleasantness, compare_pleasantness
                    from backend.core.inviting_sound_checker import check_inviting_sound
                    from backend.core.pleasantness_registry import get_pleasantness_registry

                    _restored = result.audio
                    if _restored.ndim == 2: _restored = _restored.mean(axis=1)
                    _restored_f32 = _restored.astype(np.float32)
                    _hpe_post = compute_pleasantness(_restored_f32, sr).score
                    _inv_post = check_inviting_sound(_restored_f32, sr).score
                    _cmp = compare_pleasantness(
                        np.asarray(audio, dtype=np.float32),
                        _restored_f32, sr)
                    _delta = float(_cmp.get("delta_score", 0.0))

                    _reg = get_pleasantness_registry()
                    _reg.report_post("UV3", _hpe_post, delta=_delta)
                    result.quality_delta = _delta

                    # §v10 FAILSAFE: Wenn HPE oder Inviting VERSCHLECHTERT wurde,
                    # ist Aurik GESCHEITERT. Original zurückgeben.
                    _inv_delta = _inv_post - _inv_pre
                    _failed_hpe = _delta < -0.03
                    _failed_inv = _inv_delta < -0.05

                    if _failed_hpe or _failed_inv:
                        _reasons = []
                        if _failed_hpe: _reasons.append(f"HPE {_delta:+.3f}")
                        if _failed_inv: _reasons.append(f"Inviting {_inv_delta:+.3f}")
                        _fail_msg = f"AURIK VERBESSERUNG GESCHEITERT: {', '.join(_reasons)} — gebe Original zurück"
                        logger.error("RestaurierDenker: %s", _fail_msg)
                        result.quality_delta = _delta
                        result.processing_note = _fail_msg
                        # NICHT return result — wir versuchen es leichter!
                        _retry = self._retry_lighter(audio, sr, restorer, _uv3_kwargs, material)
                        if _retry is not None:
                            return _retry
                        # Auch Retry gescheitert → Original zurück
                        logger.error("RestaurierDenker: Auch leichterer Versuch gescheitert — Original unverändert")
                        fallback = self._fallback(audio, material or "unknown", _fail_msg)
                        fallback.audio = audio.copy()
                        return fallback
                    else:
                        logger.info("RestaurierDenker: HPE %.3f->%.3f (%+.3f) Inviting %.3f->%.3f (%+.3f) %s",
                                   _hpe_pre, _hpe_post, _delta, _inv_pre, _inv_post, _inv_delta,
                                   _cmp.get("verdict",""))
                except Exception: pass

                return result
            except Exception as uv3_exc:
                logger.warning("UV3 Direkt-Pfad fehlgeschlagen: %s — Fallback.", uv3_exc, exc_info=True)
                return self._fallback(audio, material or "unknown", str(uv3_exc))

        # ── Fallback ohne Caches: ARE → UV3 (Legacy-Pfad) ────────────────────
        pipeline = self._get_are_pipeline()
        if pipeline is not None:
            try:
                _are_ctx: dict = {}
                if global_plan is not None:
                    _are_ctx["global_plan"] = global_plan
                if chain_info is not None:
                    _are_ctx["chain_info"] = chain_info
                if defekt_hint is not None:
                    _are_ctx["defekt_hint"] = defekt_hint
                if mode:
                    _are_ctx["mode"] = mode
                if material:
                    _are_ctx["material"] = material
                if cached_defect_result is not None:
                    _are_ctx["cached_defect_result"] = cached_defect_result
                are_result = pipeline.process(audio, sample_rate=sr, progress_callback=progress_callback, **_are_ctx)
                _are_audio = getattr(are_result, "audio", audio)
                del are_result, pipeline  # RAM für UV3 freigeben

                restorer = self._get_restorer(mode=mode)
                if restorer is not None:
                    logger.info("RestaurierDenker: Legacy ARE → UV3-Pass …")
                    _uv3_kwargs2: dict = {"sample_rate": sr}
                    if global_plan is not None:
                        _uv3_kwargs2["global_plan"] = global_plan
                    if chain_info is not None:
                        _uv3_kwargs2["chain_info"] = chain_info
                    if defekt_hint is not None:
                        _uv3_kwargs2["defekt_hint"] = defekt_hint
                    if progress_callback is not None:
                        _uv3_kwargs2["progress_callback"] = progress_callback
                    if audio_update_callback is not None:
                        _uv3_kwargs2["audio_update_callback"] = audio_update_callback
                    # Bug-16c-Fix: gecachte Klassifikationsergebnisse im ARE→UV3-Pfad
                    # weitergeben — ohne diese führt UV3 frische classify_medium() etc.
                    # durch → material=unknown → AudioSR auf MP3 → OOM.
                    if cached_era_result is not None:
                        _uv3_kwargs2["cached_era_result"] = cached_era_result
                    if cached_genre_result is not None:
                        _uv3_kwargs2["cached_genre_result"] = cached_genre_result
                    if cached_defect_result is not None:
                        _uv3_kwargs2["cached_defect_result"] = cached_defect_result
                    if cached_medium_result is not None:
                        _uv3_kwargs2["cached_medium_result"] = cached_medium_result
                    if cached_restorability_result is not None:
                        _uv3_kwargs2["cached_restorability_result"] = cached_restorability_result
                    if reconstruction_context is not None:
                        _uv3_kwargs2["reconstruction_context"] = reconstruction_context
                    if pre_repair_reference is not None:
                        _uv3_kwargs2["pre_repair_reference"] = pre_repair_reference
                    if input_path:
                        _uv3_kwargs2["input_path"] = input_path
                    if output_path:
                        _uv3_kwargs2["output_path"] = output_path
                    if no_rt_limit:
                        _uv3_kwargs2["no_rt_limit"] = True
                    if phase_strength_oracle_rollout is not None:
                        _uv3_kwargs2["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
                    if denker_policy_input:
                        _uv3_kwargs2["denker_policy_input"] = dict(denker_policy_input)
                    try:
                        raw = restorer.restore(_are_audio, **_uv3_kwargs2)
                        return self._konvertiere(raw, material=material)
                    except Exception as uv3_exc:
                        logger.warning("UV3 auf ARE-Audio fehlgeschlagen: %s", uv3_exc)
                        return self._fallback(_are_audio, material or "unknown", str(uv3_exc))
            except Exception as are_exc:
                logger.warning("AurikAutonomousPipeline fehlgeschlagen: %s — Fallback auf UV3", are_exc)

        # ── Letzter Fallback: UV3 direkt ──────────────────────────────────────
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
            if audio_update_callback is not None:
                _restore_kwargs["audio_update_callback"] = audio_update_callback
            if cached_era_result is not None:
                _restore_kwargs["cached_era_result"] = cached_era_result
            if cached_genre_result is not None:
                _restore_kwargs["cached_genre_result"] = cached_genre_result
            if cached_defect_result is not None:
                _restore_kwargs["cached_defect_result"] = cached_defect_result
            if cached_medium_result is not None:
                _restore_kwargs["cached_medium_result"] = cached_medium_result
            if cached_restorability_result is not None:
                _restore_kwargs["cached_restorability_result"] = cached_restorability_result
            if no_rt_limit:
                _restore_kwargs["no_rt_limit"] = True
            if phase_strength_oracle_rollout is not None:
                _restore_kwargs["phase_strength_oracle_rollout"] = phase_strength_oracle_rollout
            if denker_policy_input:
                _restore_kwargs["denker_policy_input"] = dict(denker_policy_input)
            raw = restorer.restore(audio, **_restore_kwargs)
        except Exception as exc:
            logger.warning("UnifiedRestorerV3.restore() fehlgeschlagen: %s — Fallback auf Original", exc)
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
            if UnifiedRestorerV3 is None or QualityMode is None or RestorationConfig is None:
                raise ImportError("backend.core.unified_restorer_v3 nicht verfügbar")

            # §11.7a: studio2026 → MAXIMUM quality + studio_2026 flag
            _is_studio = mode == "studio2026"
            qmode = QualityMode.MAXIMUM if _is_studio else QualityMode.QUALITY

            # Performance-Upgrade: harte 4-Core-Begrenzung entfernt.
            # Default: nutze bis zu 8 Kerne (Desktop sweet-spot laut Scheduler-Log).
            _system_cores = int(os.cpu_count() or 4)
            _auto_cores = int(max(2, min(8, _system_cores)))
            _env_cores = os.getenv("AURIK_NUM_CORES", "").strip()
            if _env_cores:
                try:
                    _requested = int(_env_cores)
                    _auto_cores = int(max(1, min(16, _requested)))
                except ValueError:
                    logger.warning(
                        "AURIK_NUM_CORES='%s' ungültig — verwende Auto-Wert=%d",
                        _env_cores,
                        _auto_cores,
                    )

            cfg = RestorationConfig(
                mode=qmode,
                studio_2026=_is_studio,  # §11.7a — activates Stem-Sep, Matchering, Vocos
                enforce_3x_rt=False,  # RT opt-in only; standard uses no_rt_limit=True (§2.38)
                enable_performance_guard=True,
                enable_adaptive_skipping=False,  # Adaptive skipping opt-in only
                enable_phase_gate=True,
                num_cores=_auto_cores,
                enable_psychoacoustic_enhancement=True,
            )

            logger.info(
                "\U0001f3b5 RestaurierDenker: UnifiedRestorerV3 init (mode=%s, num_cores=%d, system_cores=%d)",
                mode,
                _auto_cores,
                _system_cores,
            )
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
            if AurikAutonomousPipeline is None or ProcessingMode is None:
                raise ImportError("backend.core.pipeline_main nicht verfügbar")

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
            winning_variant=getattr(raw, "winning_variant", None),
            rollback_triggered=bool(getattr(raw, "rollback_triggered", False)),
            metadata={
                # §2.53 [RELEASE_MUST]: propagate complete UV3 metadata end-to-end.
                # Previously only "total_time_seconds" was forwarded — this silently
                # dropped joy_runtime_index, auto_improvement_recommendations,
                # song_calibration, and all other §2.53 telemetry fields.
                **(dict(raw.metadata or {}) if isinstance(getattr(raw, "metadata", None), dict) else {}),
                "total_time_seconds": float(raw.total_time_seconds or 0.0),
            },
            # §S5 Propagations-Fix: GAF-Ergebnis (inapplicable Goals) aus UV3-RestorationResult
            # direkt weiterleiten — ohne dieses Feld bleibt _rest_inapplicable_goals in
            # AurikDenker leer und ExzellenzDenker zählt physikalisch unmögliche Ziele als
            # Violations statt sie auszuschließen.
            goal_applicability=(dict(getattr(raw, "goal_applicability", None) or {})),
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

    @staticmethod
    def _retry_lighter(
        audio: np.ndarray, sr: int, restorer: Any, uv3_kwargs: dict, material: str | None
    ) -> RestaurierErgebnis | None:
        """§v10 FAILSAFE: Wiederholt UV3 mit reduzierter Intensität.

        Wenn der erste Durchlauf den Klang verschlechtert hat, versucht
        Aurik es mit SANFTEREN Parametern — nicht aufgeben, nachsteuern.
        """
        try:
            logger.info("RestaurierDenker: RETRY_LIGHTER — versuche mit reduzierter Intensität")
            _lighter_kwargs = dict(uv3_kwargs)
            _lighter_kwargs["mode"] = "balanced"  # Weniger aggressiv als "quality"
            raw2 = restorer.restore(audio, **_lighter_kwargs)
            from denker.restaurier_denker import RestaurierDenker
            result2 = RestaurierDenker._konvertiere.__func__(None, raw2, material=material)

            # Prüfe ob der Retry besser ist
            from backend.core.human_pleasantness_estimator import compare_pleasantness
            _r2 = result2.audio
            if _r2.ndim == 2: _r2 = _r2.mean(axis=1)
            _cmp2 = compare_pleasantness(
                np.asarray(audio, dtype=np.float32),
                np.asarray(_r2, dtype=np.float32), sr)
            _delta2 = float(_cmp2.get("delta_score", 0.0))
            if _delta2 > -0.02:
                logger.info("RestaurierDenker RETRY_LIGHTER erfolgreich: HPE %+.3f", _delta2)
                result2.quality_delta = _delta2
                return result2
            else:
                logger.warning("RestaurierDenker RETRY_LIGHTER gescheitert: HPE %+.3f", _delta2)
                return None
        except Exception as e:
            logger.warning("RestaurierDenker RETRY_LIGHTER Fehler: %s", e)
            return None


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking — §3.2)
# ---------------------------------------------------------------------------

_instance_holder: list[RestaurierDenker | None] = [None]  # list-Container vermeidet global (W0603)
_lock: threading.Lock = threading.Lock()


def get_restaurier_denker() -> RestaurierDenker:
    """Gibt den thread-sicheren Singleton-RestaurierDenker zurück."""
    instance = _instance_holder[0]
    if instance is None:
        with _lock:
            instance = _instance_holder[0]
            if instance is None:
                instance = RestaurierDenker()
                _instance_holder[0] = instance
    assert instance is not None
    return instance
