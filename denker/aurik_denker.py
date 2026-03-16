"""
denker/aurik_denker.py — AurikDenker: Orchestrator mit 8 Verarbeitungsstufen
==============================================================================

Koordiniert die vollständige Restaurierungs-Pipeline in der kanonischen
Reihenfolge (Spec §2.2):

  1. TontraegerDenker  → Trägermedium-Erkennung
  2. DefektDenker      → Defekt-Analyse + Kausal-Reasoning
  3. StrategieDenker   → 3×RT-Budget-Plan + Timer
  4. _run_rest()-Closure → ReparaturDenker (Preprocessing) →
                           RekonstruktionsDenker (Preprocessing) →
                           RestaurierDenker (UV3-Vollpipeline)
  7. ExzellenzDenker   → Musical Goals + Excellence-Optimierung
  8. VERSA MOS-Gate     → Finale Qualitätsbewertung (§4.4, nicht-referenzbasiert);
                          MOS < 4.0 → 2. ExzellenzDenker-Durchlauf

3×RT-Invariante: rt_factor im Endergebnis ist IMMER ≤ 3.0.

Spec §2.1, §2.2, §9.5 — v9.10.45
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_3X_RT_LIMIT: float = 3.0  # Maximaler Echtzeit-Faktor (Spec §9.5)
_MIN_AUDIO_SAMPLES: int = 64  # Mindestsignallänge


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class AurikErgebnis:
    """Vollständiges Restaurierungsergebnis des AurikDenker-Orchestrators.

    Felder:
        audio:               Restauriertes Audio (float32, clip [-1, 1])
        material:            Erkanntes Trägermedium (z. B. "tape", "vinyl")
        rt_factor:           Tatsächlicher Echtzeit-Faktor (≤ 3.0)
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
    # §Dach: Musikalischer Globalplan — stilbewusstes Restaurierungsportrait
    global_plan: dict[str, Any] | None = field(default=None)

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
        }


# ─── Orchestrator ────────────────────────────────────────────────────────────


class AurikDenker:
    """Orchestrator mit 8 Verarbeitungsstufen im Aurik-System.

    Steuert die vollständige Restaurierungs-Pipeline sequenziell und
    überwacht dabei das 3×RT-Budget. Jeder Domänen-Denker läuft in
    try/except — ein Fehler in einer Stufe stoppt nicht die gesamte Pipeline.

    Verwendung::

        aurik = get_aurik_denker()
        ergebnis = aurik.restauriere(audio, sr=48_000)
        logger.debug(f"RT-Faktor: {ergebnis.rt_factor:.2f}×")
        logger.debug(f"Qualität: {ergebnis.quality_estimate:.3f} (VERSA MOS eingerechnet)")
        logger.debug(f"Goals: {ergebnis.goals_passed}/{len(ergebnis.musical_goals)}")
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
            ergebnis = self._orchestriere(audio, sr, audio_duration_s, t_start,
                                          mode=mode, progress_callback=progress_callback)
        except MemoryError as exc:
            elapsed = time.perf_counter() - t_start
            rt = elapsed / max(audio_duration_s, 1e-6)
            try:
                from backend.core.plugin_lifecycle_manager import evict_stale_plugins  # noqa: PLC0415

                evict_stale_plugins(required_mb=2048.0)
            except Exception:
                pass
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
        return self.restauriere(
            audio, sr,
            mode=kwargs.get("mode", "quality"),
            progress_callback=kwargs.get("progress_callback", None),
        )

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
    ) -> AurikErgebnis:
        """Führt die 8-stufige Restaurierungs-Pipeline aus.

        Jede Stufe wird in try/except gekapselt. Fehler einer Stufe führen
        zur Fortsetzung mit dem bisherigen Audio-Zustand.
        """

        def _emit(pct: int, msg: str) -> None:
            """Emittiert Fortschritt (0–100) wenn progress_callback gesetzt."""
            if progress_callback is not None:
                try:
                    elapsed = time.perf_counter() - t_start
                    progress_callback(pct, msg, elapsed)
                except Exception:  # noqa: BLE001
                    pass
        warnings: list[str] = []
        phases_executed: list[str] = []
        stage_notes: dict[str, str] = {}
        aktuelles_audio = audio.copy()

        def _rt() -> float:
            elapsed = time.perf_counter() - t_start
            return elapsed / max(audio_duration_s, 1e-6)

        strat_denker: Any = None  # M-2: hoisted — für _budget_ok()-Zugriff nach Stufe 3
        defekt: Any = None  # M-1/M-4: hoisted — für Stage-5-Flags

        def _budget_ok() -> bool:
            # M-2: RT-Primärcheck — immer aktiv
            if _rt() >= _3X_RT_LIMIT * 0.90:
                return False
            # M-2: StrategieDenker-Budget-Gate — aktiv sobald Stufe 3 abgeschlossen
            if strat_denker is not None:
                try:
                    if strat_denker.check(phases_remaining=0).should_exit_early:
                        logger.info("AurikDenker: StrategieDenker signalisiert Budget-Abbruch")
                        return False
                except Exception:
                    pass  # defensiv: lieber fortfahren als abstürzen
            return True

        # ── Stufe 1: Tonträger-Erkennung ─────────────────────────────────────
        _emit(2, "Tonträger wird erkannt …")
        material = "unknown"
        try:
            toni = get_tontraeger_denker().erkenne(aktuelles_audio, sr)
            material = toni.material_type
            stage_notes["tontraeger"] = f"{material} (Konfidenz: {toni.confidence:.2f})"
            phases_executed.append("tontraeger_erkennung")
            logger.info("AurikDenker [1/8] Träger: %s (%.2f)", material, toni.confidence)
        except Exception as exc:
            warnings.append(f"TontraegerDenker fehlgeschlagen: {exc}")
            stage_notes["tontraeger"] = f"Fehler: {exc}"
            logger.warning("AurikDenker [1/8] TontraegerDenker: %s", exc)

        # ── Stufe 1b: Tonträgerketten-Analyse ────────────────────────────────
        _emit(4, "Tonträgerkette analysiert …")
        chain_info: dict[str, Any] = {}
        try:
            kette = get_tontraegerkette_denker().analysiere(aktuelles_audio, sr)
            chain_info = kette.as_dict()
            stage_notes["kette"] = kette.chain_string
            phases_executed.extend(kette.combined_phases)
            logger.info(
                "AurikDenker [1b/8] Kette: %s (Komplexität: %.2f)",
                kette.chain_string,
                kette.chain_complexity,
            )
        except Exception as exc:
            warnings.append(f"TontraegerketteDenker fehlgeschlagen: {exc}")
            stage_notes["kette"] = f"Fehler: {exc}"
            logger.warning("AurikDenker [1b/8] TontraegerketteDenker: %s", exc)

        # ── Stufe 2: Defekt-Analyse ───────────────────────────────────────────
        _emit(6, "Defekte werden analysiert …")
        defekt_primär = "unbekannt"
        defekt = None  # Guard: DefektDenker kann fehlschlagen
        _defekt_hint: dict[str, Any] | None = None
        try:
            defekt = get_defekt_denker().analysiere(aktuelles_audio, sr, material=material)
            defekt_primär = getattr(defekt, "primary_defect", None) or getattr(defekt, "primary_cause", "unknown")
            stage_notes["defekt"] = f"Hauptdefekt: {defekt_primär} " f"(Schwere: {defekt.overall_severity:.2f})"
            phases_executed.append("defekt_analyse")
            _defekt_hint = {
                "recommended_phases": list(getattr(defekt, "recommended_phases", [])),
                "confidence": float(getattr(defekt, "cause_confidence", 0.0)),
            }
            logger.info(
                "AurikDenker [2/8] Defekt: %s (Schwere: %.2f)",
                defekt_primär,
                defekt.overall_severity,
            )
        except Exception as exc:
            warnings.append(f"DefektDenker fehlgeschlagen: {exc}")
            stage_notes["defekt"] = f"Fehler: {exc}"
            logger.warning("AurikDenker [2/8] DefektDenker: %s", exc)

        # ── Stufe 2b: Musikalischer Globalplan (§Dach) ───────────────────────
        _emit(8, "Musikalischer Restaurierungsplan erstellt …")
        _globalplan: Any = None
        try:
            from backend.core.musikalischer_globalplan import erstelle_globalplan as _erstelle_gp
            # use_ml_classifiers=False: EraClassifier/GermanSchlagerClassifier laufen
            # bereits parallel in UnifiedRestorerV3 (§P-3). Doppelaufruf vermeiden
            # (Anti-Parallelwelten-Pflicht). Nur DSP-Heuristik in Stufe 2b.
            _globalplan = _erstelle_gp(
                aktuelles_audio, sr, material=material, use_ml_classifiers=False
            )
            stage_notes["globalplan"] = (
                f"\u00c4ra: {_globalplan.portrait.decade}er, "
                f"Genre: {_globalplan.portrait.genre}, "
                f"Intention: {_globalplan.emotional_intention}"
            )
            phases_executed.append("musikalischer_globalplan")
            logger.info(
                "AurikDenker [2b/8] Globalplan: %s\u00b4er | %s | Authentizit\u00e4t=%.2f",
                _globalplan.portrait.decade,
                _globalplan.portrait.genre,
                _globalplan.authenticity_target,
            )
        except Exception as exc:
            warnings.append(f"MusikalischerGlobalplan fehlgeschlagen: {exc}")
            stage_notes["globalplan"] = f"Fehler: {exc}"
            logger.warning("AurikDenker [2b/8] MusikalischerGlobalplan: %s", exc)

        # ── Stufe 3: Strategie (3×RT-Budget) ────────────────────────────────
        _emit(10, "Restaurierungsstrategie geplant …")
        strategie = None
        try:
            strat_denker = get_strategie_denker()
            strategie = strat_denker.plan(aktuelles_audio, sr, enforce_3x_rt=True)
            strat_denker.starte_timer(audio_duration_s)
            stage_notes["strategie"] = f"Budget: {strategie.max_processing_s:.1f}s, " f"Modus: {strategie.quality_mode}"
            phases_executed.append("strategie_plan")
            logger.info(
                "AurikDenker [3/8] Budget: %.1fs für %.1fs Audio",
                strategie.max_processing_s,
                audio_duration_s,
            )
        except Exception as exc:
            warnings.append(f"StrategieDenker fehlgeschlagen: {exc}")
            stage_notes["strategie"] = f"Fehler: {exc}"
            logger.warning("AurikDenker [3/8] StrategieDenker: %s", exc)

        # ── ARE-Metadaten-Träger (A-2/A-5/B-1) ────────────────────────────────
        _rest_confidence: float = 0.85
        _rest_rollback: bool = False
        _rest_variant: str | None = None
        _rest_musical_goals: dict[str, float] = {}
        _rest_goals_passed: int = 0
        _gaps_found: int = 0
        _gaps_repaired: int = 0
        _gap_total_ms: float = 0.0

        # ── Stufe 4: _run_rest()-Closure — Reparatur → Rekonstruktion → Restaurierung ──
        _emit(12, "Vorverarbeitung & Restaurierung laufen …")
        if _budget_ok():
            try:
                # M-3: Mode-Hierarchie: expliziter mode-Parameter hat Vorrang vor StrategieDenker
                _strat_mode = strategie.quality_mode if strategie is not None else "quality"
                _mode = mode if mode != "quality" else _strat_mode  # Studio 2026 überschreibt
                # RT-Budget-Guard: Restaurierung läuft in eigenem Thread, damit
                # ein langsamer Kaltstart (Modul-Import-Kaskade) das Gesamt-
                # Budget nicht sprengt und den pytest-Timeout auslöst.
                _remaining = max(
                    audio_duration_s * _3X_RT_LIMIT - (time.perf_counter() - t_start),
                    0.5,
                )
                _result_box: list = []
                _err_box: list = []
                _rep_result_box: list = []   # ReparaturDenker Ergebnis (4a)
                _rek_result_box: list = []   # RekonstruktionsDenker Ergebnis (4b)

                def _run_rest() -> None:
                    try:
                        _work_audio = aktuelles_audio.copy()
                        # 4a: ReparaturDenker — gezielter Phase-Mix (Preprocessing vor UV3)
                        _emit(13, "Gezielte DSP-Reparaturen (Vorverarbeitung) …")
                        _ph = set(getattr(defekt, "recommended_phases", None) or [])
                        _sev = float(getattr(defekt, "overall_severity", 1.0))
                        _skip_repair = _sev < 0.05 and defekt is not None
                        _remove_clicks: bool = not _skip_repair and (
                            defekt is None
                            or bool(_ph & {"phase_01_click_removal", "phase_09_crackle_removal", "phase_27_click_pop_removal"})
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
                        rep = get_reparatur_denker().repariere(
                            _work_audio, sr,
                            remove_clicks=_remove_clicks,
                            remove_hum=_remove_hum,
                            repair_clipping=_repair_clipping,
                        )
                        _work_audio = rep.audio
                        _rep_result_box.append(rep)
                        # 4b: RekonstruktionsDenker — Lücken-Erkennung & Reparatur (Preprocessing vor UV3)
                        _emit(16, "Dropout-Lücken werden rekonstruiert (Vorverarbeitung) …")
                        rek = get_rekonstruktions_denker().rekonstruiere(
                            _work_audio, sr, material_hint=material
                        )
                        _work_audio = rek.audio
                        _rek_result_box.append(rek)
                        # 4c: RestaurierDenker (UV3-Vollpipeline) auf vorgereinigtem Material
                        # Scaled inner progress: UV3 0–100 → AurikDenker 18–80
                        if progress_callback is not None:
                            def _inner_cb(pct: int, msg: str, elapsed: float = 0.0) -> None:
                                _emit(18 + int(pct * 0.62), msg)
                        else:
                            _inner_cb = None
                        _result_box.append(
                            get_restaurier_denker().restauriere(
                                _work_audio, sr,
                                material=material,
                                mode=_mode,
                                global_plan=_globalplan,
                                chain_info=chain_info or None,
                                defekt_hint=_defekt_hint,
                                progress_callback=_inner_cb,
                            )
                        )
                    except Exception as _e:  # noqa: BLE001
                        _err_box.append(_e)

                _t = threading.Thread(target=_run_rest, daemon=True)
                _t.start()
                _t.join(timeout=_remaining)
                if _t.is_alive():
                    raise RuntimeError(
                        f"RestaurierDenker überschritt RT-Budget "
                        f"({_remaining:.1f}s für {audio_duration_s:.1f}s Audio)"
                    )
                if _err_box:
                    raise _err_box[0]  # type: ignore[misc]
                rest = _result_box[0]
                aktuelles_audio = rest.audio
                phases_executed.extend(rest.phases_executed or [])
                warnings.extend(rest.warnings or [])
                # A-2: ARE-Metadaten propagieren
                _rest_confidence = float(getattr(rest, "confidence", 0.85))
                _rest_rollback = bool(getattr(rest, "rollback_triggered", False))
                _rest_variant = getattr(rest, "winning_variant", None)
                _rest_musical_goals = dict(getattr(rest, "musical_goals", {}))
                _rest_goals_passed = int(getattr(rest, "goals_passed", 0))
                # §Dach-Enrichment: era_decade aus RestorationResult in Globalplan übernehmen
                # (ML-Klassifikatoren liefen in UV3 — jetzt Ergebnis in Plan einpflegen)
                if _globalplan is not None:
                    _era_from_pipeline = getattr(rest, "era_decade", None)
                    if _era_from_pipeline is not None:
                        try:
                            _globalplan.portrait.decade = int(_era_from_pipeline)
                            _globalplan.portrait.era_confidence = max(
                                _globalplan.portrait.era_confidence, 0.75
                            )
                            _globalplan.reasoning_trace.append(
                                f"Enriched with pipeline era_decade={_era_from_pipeline} (UV3 ML)"
                            )
                        except Exception:  # noqa: BLE001
                            pass
                # 4a: ReparaturDenker-Ergebnis (Preprocessing-Schritt)
                if _rep_result_box:
                    _rep = _rep_result_box[0]
                    warnings.extend(_rep.warnings or [])
                    stage_notes["reparatur"] = (
                        f"Clicks: {_rep.clicks_removed}, "
                        f"Hum: {_rep.hum_removed}, "
                        f"Clipping: {_rep.clipping_repaired} (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [4a/8] Reparatur (Pre-UV3): clicks=%s hum=%s clipping=%s",
                        _rep.clicks_removed, _rep.hum_removed, _rep.clipping_repaired,
                    )
                # 4b: RekonstruktionsDenker-Ergebnis (Preprocessing-Schritt)
                if _rek_result_box:
                    _rek = _rek_result_box[0]
                    warnings.extend(_rek.warnings or [])
                    _gaps_found = int(getattr(_rek, "gaps_found", 0))
                    _gaps_repaired = int(getattr(_rek, "gaps_repaired", 0))
                    _gap_total_ms = float(getattr(_rek, "total_repaired_ms", 0.0))
                    stage_notes["rekonstruktion"] = (
                        f"Lücken gefunden: {_gaps_found}, "
                        f"repariert: {_gaps_repaired}/{_gap_total_ms:.1f} ms (Vorverarbeitung)"
                    )
                    logger.info(
                        "AurikDenker [4b/8] Rekonstruktion (Pre-UV3): %d/%d Lücken, %.1f ms",
                        _gaps_repaired, _gaps_found, _gap_total_ms,
                    )
                stage_notes["restaurierung"] = (
                    f"Qualität: {rest.quality_estimate:.3f}, "
                    f"RT: {rest.rt_factor:.2f}×, "
                    f"Konfidenz: {_rest_confidence:.2f}"
                )
                logger.info(
                    "AurikDenker [4/8] Restaurierung: Q=%.3f, RT=%.2f×, K=%.2f",
                    rest.quality_estimate,
                    rest.rt_factor,
                    _rest_confidence,
                )
            except Exception as exc:
                warnings.append(f"RestaurierDenker fehlgeschlagen: {exc}")
                stage_notes["restaurierung"] = f"Fehler: {exc}"
                logger.warning("AurikDenker [4/8] RestaurierDenker: %s", exc)
        else:
            stage_notes["restaurierung"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Restaurierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [4/8] Budget erschöpft, Restaurierung übersprungen")

        # ── Stufe 7: Exzellenz-Optimierung + Musical Goals ───────────────────
        _emit(91, "Musikalische Exzellenz wird optimiert …")
        musical_goals: dict[str, float] = dict(_rest_musical_goals)
        goals_passed: int = _rest_goals_passed
        excellence_score = 0.0

        if _budget_ok():
            try:
                exz = get_exzellenz_denker().optimiere(aktuelles_audio, sr, material=material)
                aktuelles_audio = exz.audio
                # A-5: ExzellenzDenker nur wenn er bessere Goals liefert
                if exz.musical_goals and exz.goals_passed >= goals_passed:
                    musical_goals = exz.musical_goals
                    goals_passed = exz.goals_passed
                excellence_score = exz.excellence_score
                warnings.extend(exz.warnings or [])
                stage_notes["exzellenz"] = exz.processing_note
                phases_executed.append("exzellenz_optimierung")
                logger.info(
                    "AurikDenker [7/8] Exzellenz: Score=%.3f, Goals %d/%d",
                    excellence_score,
                    goals_passed,
                    exz.goals_total,
                )
            except Exception as exc:
                warnings.append(f"ExzellenzDenker fehlgeschlagen: {exc}")
                stage_notes["exzellenz"] = f"Fehler: {exc}"
                logger.warning("AurikDenker [7/8] ExzellenzDenker: %s", exc)
        else:
            stage_notes["exzellenz"] = "Übersprungen (RT-Budget ausgeschöpft)"
            warnings.append("Exzellenz-Optimierung übersprungen: RT-Budget ausgeschöpft")
            logger.warning("AurikDenker [7/8] Exzellenz übersprungen")

        # ── Stufe 8: VERSA MOS — finales Qualitätsurteil (§4.4) ─────────────
        _emit(95, "VERSA MOS-Qualitätsbewertung läuft …")
        # VERSA 2024 liefert eine unabhängige, nicht-referenzbasierte MOS-Bewertung.
        # Dieser Score fließt in quality_estimate ein und entscheidet ob eine
        # Warnung für den Benutzer erzeugt wird.
        _versa_mos: float = 0.0
        if _budget_ok():
            try:
                from plugins.versa_plugin import score_mos  # noqa: PLC0415

                _mono_final = aktuelles_audio if aktuelles_audio.ndim == 1 else aktuelles_audio.mean(axis=-1)
                _mono_final = np.nan_to_num(_mono_final.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr = score_mos(_mono_final, sr)
                _versa_mos = float(_vr.mos)
                stage_notes["versa_mos"] = f"MOS={_versa_mos:.3f} ({_vr.model_used})"
                phases_executed.append("versa_qualitaetsbewertung")
                logger.info(
                    "AurikDenker [8/8] VERSA MOS=%.3f — %s",
                    _versa_mos,
                    "✓ Studioqualität" if _versa_mos >= 4.3 else (
                        "✓ Gute Qualität" if _versa_mos >= 3.5 else "⚠ Qualität unter Mindestniveau"
                    ),
                )
                if _versa_mos < 3.5:
                    warnings.append(
                        f"VERSA MOS={_versa_mos:.2f} < 3.5 — Klangqualität unter Mindestniveau"
                    )
                # §Spec VERSA MOS-Gate: MOS < 4.0 → zweiter ExzellenzDenker-Durchlauf
                if _versa_mos > 0.0 and _versa_mos < 4.0 and _budget_ok():
                    _emit(97, "VERSA MOS-Gate: 2. Exzellenz-Optimierungspass …")
                    try:
                        exz2 = get_exzellenz_denker().optimiere(aktuelles_audio, sr, material=material)
                        aktuelles_audio = exz2.audio
                        if exz2.musical_goals and exz2.goals_passed >= goals_passed:
                            musical_goals = exz2.musical_goals
                            goals_passed = exz2.goals_passed
                        if exz2.excellence_score > excellence_score:
                            excellence_score = exz2.excellence_score
                        warnings.extend(exz2.warnings or [])
                        stage_notes["exzellenz_2"] = (
                            f"MOS-Gate (MOS={_versa_mos:.2f} < 4.0): "
                            f"Score={exz2.excellence_score:.3f}, Goals {exz2.goals_passed}/{exz2.goals_total}"
                        )
                        phases_executed.append("exzellenz_optimierung_2")
                        logger.info(
                            "AurikDenker [8/8] MOS-Gate: 2. Exzellenz Score=%.3f, Goals %d/%d",
                            exz2.excellence_score, goals_passed, exz2.goals_total,
                        )
                    except Exception as exc:
                        warnings.append(f"ExzellenzDenker 2. Durchlauf fehlgeschlagen: {exc}")
                        stage_notes["exzellenz_2"] = f"MOS-Gate Fehler: {exc}"
                        logger.warning("AurikDenker [8/8] MOS-Gate ExzellenzDenker 2. Pass: %s", exc)
            except Exception as _ve:  # noqa: BLE001
                logger.debug("AurikDenker [8/8] VERSA MOS nicht verfügbar: %s", _ve)

        # ── Stufe 8b: RAM-Cleanup nach Pipeline ──────────────────────────────
        # PluginLifecycleManager entlädt inaktive ML-Modelle wenn RAM knapp ist.
        # Kein Force-Evict hier (Batch-Cleanup erfolgt im Aufrufer via
        # cleanup_after_file()), nur druckbasiertes Evict.
        _emit(98, "RAM-Management …")
        try:
            from backend.core.plugin_lifecycle_manager import evict_stale_plugins  # noqa: PLC0415
            _n_evicted = evict_stale_plugins()
            if _n_evicted > 0:
                stage_notes["ram_cleanup"] = f"{_n_evicted} Plugin(s) aus RAM entladen"
        except Exception as _plm_err:
            logger.debug("PLM-Evict nach Pipeline: %s", _plm_err)

        # ── Abschluss: RT-Faktor berechnen ───────────────────────────────────
        elapsed = time.perf_counter() - t_start
        rt_factor = elapsed / max(audio_duration_s, 1e-6)
        rt_factor = min(rt_factor, _3X_RT_LIMIT)  # Spec §9.5: niemals > 3.0

        # Qualitätsschätzung: Priorität VERSA MOS → excellence_score → DSP-basiert
        if _versa_mos > 0.0:
            # VERSA MOS [1,5] → [0,1] skaliert, mit Excellence-Bonus gewichtet
            _versa_norm = float(np.clip((_versa_mos - 1.0) / 4.0, 0.0, 1.0))
            quality_estimate = 0.6 * _versa_norm + 0.4 * excellence_score if excellence_score > 0.0 else _versa_norm
        elif excellence_score > 0.0:
            quality_estimate = excellence_score
        else:
            # Einfache DSP-basierte Schätzung (SNR-Proxy)
            rms = float(np.sqrt(np.mean(aktuelles_audio.astype(np.float64) ** 2)))
            quality_estimate = min(max(rms * 4.0, 0.55), 0.95)

        # Finale NaN/Inf-Bereinigung und Clip
        aktuelles_audio = np.nan_to_num(aktuelles_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        aktuelles_audio = np.clip(aktuelles_audio, -1.0, 1.0)

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
            global_plan=_globalplan.as_dict() if _globalplan is not None else None,
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
            stage_notes={"fehler": grund},
            chain_info=None,
        )


# ─── Modul-Level-Singleton (Double-Checked Locking — Spec §3.2) ─────────────

_instance: AurikDenker | None = None
_lock = threading.Lock()


def get_aurik_denker() -> AurikDenker:
    """Thread-sicherer Singleton-Accessor für den AurikDenker-Orchestrator."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AurikDenker()
    return _instance


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
    from denker.tontraeger_denker import get_tontraeger_denker as _get

    return _get()


def get_defekt_denker() -> Any:
    """Modul-Level-Accessor für DefektDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.defekt_denker import get_defekt_denker as _get

    return _get()


def get_tontraegerkette_denker() -> Any:
    """Modul-Level-Accessor für TontraegerketteDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.tontraegerkette_denker import get_tontraegerkette_denker as _get

    return _get()


def get_strategie_denker() -> Any:
    """Modul-Level-Accessor für StrategieDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.strategie_denker import get_strategie_denker as _get

    return _get()


def get_restaurier_denker() -> Any:
    """Modul-Level-Accessor für RestaurierDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.restaurier_denker import get_restaurier_denker as _get

    return _get()


def get_reparatur_denker() -> Any:
    """Modul-Level-Accessor für ReparaturDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.reparatur_denker import get_reparatur_denker as _get

    return _get()


def get_rekonstruktions_denker() -> Any:
    """Modul-Level-Accessor für RekonstruktionsDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.rekonstruktions_denker import get_rekonstruktions_denker as _get

    return _get()


def get_exzellenz_denker() -> Any:
    """Modul-Level-Accessor für ExzellenzDenker (patchbar in Tests).

    Lazy-Import: wird erst beim ersten Aufruf importiert.
    """
    from denker.exzellenz_denker import get_exzellenz_denker as _get

    return _get()
