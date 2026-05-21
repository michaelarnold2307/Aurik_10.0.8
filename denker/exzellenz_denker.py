"""
denker/exzellenz_denker.py — Exzellenz-Denker für Aurik 9
==========================================================

Verbessert restauriertes Audio bis an die physikalische Qualitätsgrenze:
  • ExcellenceOptimizer: Spectral Continuity, Micro-Dynamics, Harmonic Boost, OLA
    • MusicalGoalsChecker: Messung aller 15 musikalischen Qualitätsziele

Spec §1.2, §2.5, §2.29, §8.1 — v9.10.45
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Eager top-level import verhindert Import-Lock-Deadlock in langen Test-Läufen
# (§3.4 Lazy Imports / Spec Threading-Invariante)
try:
    from backend.core.vocal_ai_enhancement import (
        EmotionPreservationMode,
        UnifiedVocalAIEnhancer,
    )
except Exception:  # ImportError, AttributeError, etc.
    UnifiedVocalAIEnhancer = None  # type: ignore[assignment]
    EmotionPreservationMode = None  # type: ignore[assignment]

_3X_RT_LIMIT: float = 32.0  # Maximaler RT-Faktor (Spec §9.5)


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class ExzellenzErgebnis:
    """Ergebnis der Exzellenz-Optimierung mit Musical-Goals-Bewertung.

    Felder:
        audio:           Optimiertes Audio (float32, clip [-1, 1])
        excellence_score: Gesamt-Exzellenz-Score ∈ [0, 1]
        musical_goals:   14 Musical-Goals-Scores (Dict[str, float])
        goals_passed:    Anzahl bestandener Goals
        goals_total:     Gesamtanzahl gemessener Goals
        improvements:    Liste der angewendeten Optimierungsschritte (Deutsch)
        processing_note: Kurzbeschreibung auf Deutsch
        warnings:        Warnungsliste (technisch, Englisch)
    """

    audio: np.ndarray
    excellence_score: float
    musical_goals: dict[str, float]
    goals_passed: int
    goals_total: int
    improvements: list[str]
    processing_note: str
    warnings: list[str] = field(default_factory=list)
    versa_mos: float = 0.0
    """VERSA MOS-Score \u2208 [1, 5] gemessen am optimierten Audio (0.0 = nicht verfügbar).
    Wird von AurikDenker wiederverwendet, um doppelte VERSA-Inferenz zu vermeiden.
    """
    frisson_index: float = 0.0
    """Gänsehaut-Propensity \u2208 [0, 1] aus Goals-Proxy (Blood & Zatorre 2001, §2.53).
    0.0 = nicht berechnet oder keine Goals verfügbar.
    """
    mert_proxy_used: bool = False
    """True wenn VERSA fehlschlug und MERT als Proxy-Fallback verwendet wurde (§2.44)."""

    def as_dict(self) -> dict:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "excellence_score": float(self.excellence_score),
            "musical_goals": {k: float(v) for k, v in self.musical_goals.items()},
            "goals_passed": self.goals_passed,
            "goals_total": self.goals_total,
            "improvements": list(self.improvements),
            "processing_note": self.processing_note,
            "warnings": list(self.warnings),
            "frisson_index": float(self.frisson_index),
            "mert_proxy_used": bool(self.mert_proxy_used),
        }


# ─── Denker-Klasse ───────────────────────────────────────────────────────────


class ExzellenzDenker:
    """Optimiert restauriertes Audio auf höchstes musikalisches Niveau.

    Kombiniert den ExcellenceOptimizer (vier DSP-Pfade) mit dem
    MusicalGoalsChecker (14 Qualitätsziele) zu einer vollständigen
    Exzellenz-Bewertungseinheit.

    Verwendung::

        exd = get_exzellenz_denker()
        ergebnis = exd.optimiere(audio, sr=48_000, material="tape")
        logger.debug(ergebnis.musical_goals)
    """

    def __init__(self) -> None:
        self._optimizer: Any = None
        self._opt_lock = threading.Lock()
        self._checker: Any = None
        self._checker_lock = threading.Lock()

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def optimiere(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        *,
        material: str = "auto",
        _messe_ziele_vorab: bool = True,
    ) -> ExzellenzErgebnis:
        """Optimiert Audio auf CEDAR Cambridge-Niveau und misst alle 15 Goals.

        Ablauf:
            1. NaN/Inf-Schutz
            2. Optional: Musical Goals vor Optimierung messen (Baseline)
            3. ExcellenceOptimizer: Spectral Continuity + Micro-Dynamics +
               Harmonic Boost + OLA-Crossfade
            4. Musical Goals nach Optimierung messen
            5. Excellence-Score aus Goals berechnen

        Args:
            audio:              Audio-Signal (mono/stereo, float32)
            sr:                 Sample-Rate in Hz (Standard 48 000)
            material:           Trägermedium-Bezeichnung für GP-Priors
            messe_ziele_vorab:  Ob Baseline-Goals gemessen werden sollen

        Returns:
            ExzellenzErgebnis mit optimiertem Audio und allen Metriken
        """
        assert sr == 48000, f"ExzellenzDenker.optimiere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        # NaN/Inf-Schutz (Spec §3.1)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size == 0:
            return self._fallback(audio, "Leeres Eingabe-Audio")

        warnings: list[str] = []
        improvements: list[str] = []
        _metadata: dict = {}  # §2.44 MERT-Proxy-Flag und interne Telemetrie

        # Schritt 1 — Excellence-Optimierung
        try:
            opt = self._get_optimizer(sr=sr, material=material)
            optimiertes_audio, result = opt.optimize(audio)
            improvements = list(getattr(result, "applied_steps", []))
            logger.info(
                "ExzellenzDenker: Optimierung abgeschlossen — %s",
                getattr(result, "summary", lambda: "ok")(),
            )
        except Exception as exc:
            warnings.append(f"ExcellenceOptimizer nicht verfügbar: {exc}")
            optimiertes_audio = audio.copy()
            logger.warning("ExzellenzDenker: Excellence-Optimierung fehlgeschlagen: %s", exc)

        # NaN/Inf nach Optimierung prüfen
        optimiertes_audio = np.nan_to_num(optimiertes_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        optimiertes_audio = np.clip(optimiertes_audio, -1.0, 1.0)

        # ── Vokal-KI-Verbesserung (material-abhängig) ─────────────────────────
        _VOCAL_MATERIALS = {"gesang", "sprache", "vocal", "singer", "speech", "voice"}
        if material.lower().strip() in _VOCAL_MATERIALS:
            try:
                if UnifiedVocalAIEnhancer is None or EmotionPreservationMode is None:
                    raise ImportError("core.vocal_ai_enhancement nicht verfügbar")
                _vocal_enh = UnifiedVocalAIEnhancer(sample_rate=sr)
                _vocal_result = _vocal_enh.enhance(
                    optimiertes_audio,
                    emotion_mode=EmotionPreservationMode.BALANCED,
                    breath_preservation=0.7,
                    sibilance_reduction=True,
                )
                optimiertes_audio = _vocal_result.audio
                logger.info(
                    "ExzellenzDenker Vokal-KI: Qualität +%.3f, Phasen=%s",
                    _vocal_result.quality_improvement,
                    _vocal_result.processing_applied,
                )
            except Exception as _exc:
                logger.warning("ExzellenzDenker VocalAIEnhancer: %s", _exc)

        # ── VERSA MOS: Qualitätssignal für Entscheidung über Nachbearbeitung (§4.4) ──
        # VERSA 2024 (non-reference MOS) gibt ein unabhängiges Klangsignal,
        # das entscheidet ob eine weitere Verarbeitungsrunde sinnvoll ist.
        versa_mos: float = 0.0
        try:
            from plugins.versa_plugin import score_mos

            _mono = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
            _mono = np.nan_to_num(_mono.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            _vr = score_mos(_mono, sr)
            versa_mos = float(_vr.mos)
            logger.info(
                "ExzellenzDenker VERSA MOS=%.3f (%s) — Entscheidungssignal",
                versa_mos,
                _vr.model_used,
            )
            if versa_mos < 3.5:
                warnings.append(
                    f"VERSA MOS={versa_mos:.2f} < 3.5 — Qualität unter Mindestniveau, Nachbearbeitung empfohlen"
                )
            elif versa_mos >= 4.3:
                improvements.append(f"VERSA MOS={versa_mos:.2f} ≥ 4.3 — Studioqualität erreicht")
        except Exception as _ve:
            logger.debug("ExzellenzDenker: VERSA MOS nicht verfügbar: %s", _ve)
            # §2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar.
            # Hier: VERSA fehlgeschlagen → MERT als Proxy-Fallback (§2.44).
            try:
                from plugins.mert_plugin import get_mert_plugin as _get_mert

                _mert = _get_mert()
                _mono_mert = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
                _mono_mert = np.nan_to_num(_mono_mert.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _mert_analysis = _mert.analyze(_mono_mert, sr)
                versa_mos = float(_mert_analysis.naturalness_score) * 5.0  # MERT [0,1] → MOS-Skala
                logger.info(
                    "ExzellenzDenker MERT-Proxy MOS=%.3f (VERSA unavailable, §2.44 fallback)",
                    versa_mos,
                )
                _metadata["mert_proxy_used"] = True  # §2.44 VERBOTEN-Invariante
            except Exception as _mert_exc:
                logger.debug("ExzellenzDenker: MERT-Proxy MOS nicht verfügbar: %s", _mert_exc)

        # Schritt 2 — Musical Goals messen
        goals = self.messe_ziele(optimiertes_audio, sr)

        # Excellence-Score: Mittelwert aller Goals
        # §09.2 Material-adaptive Mindest-Score (nicht hardcoded 0.75 — Shellac-Ceiling 0.70,
        # Vinyl 0.88, CD 0.95 → 0.75 wäre für schwierige Materialien systematisch falsch).
        try:
            from backend.core.calibration_matrix import _MATERIAL_QUALITY_CEILING as _MQC

            _mat_norm_ed = str(material or "auto").strip().lower()
            _goal_min_raw = _MQC.get(_mat_norm_ed, _MQC.get("vinyl", 0.75)) * 0.85
            _GOAL_MIN: float = float(np.clip(_goal_min_raw, 0.50, 0.75))
        except Exception:
            _GOAL_MIN = 0.75
        if goals:
            finite_vals = [v for v in goals.values() if np.isfinite(v)]
            score = float(np.mean(finite_vals)) if finite_vals else 0.0
            passed = sum(1 for v in finite_vals if v >= _GOAL_MIN)
        else:
            score = 0.0
            passed = 0

        # Schritt 3 — Musical Goals Re-Pass bei Violations (max. 1 Runde)
        # v9.10.58: Von 3 auf 1 Re-Pass reduziert — wissenschaftliche Begründung:
        # Kaskadierte identische Verarbeitung akkumuliert STFT-Rundungsfehler
        # (Ephraim & Malah 1984) und verschiebt ML-Modelle in untrainierte Domains.
        # Ein einzelner korrigierender Pass mit degressiver Intensität ist optimal.
        _violations = [k for k, v in goals.items() if math.isfinite(v) and v < _GOAL_MIN] if goals else []
        _MAX_RE_PASSES = 1
        for _re_pass_i in range(1, _MAX_RE_PASSES + 1):
            _violations = [k for k, v in goals.items() if math.isfinite(v) and v < _GOAL_MIN] if goals else []
            if not _violations:
                break
            try:
                _opt_rp = self._get_optimizer(sr=sr, material=material)
                # Degressive intensity: reduce harmonic boost + modulation each pass
                _opt_rp._harm_boost_db *= max(0.3, 1.0 - 0.3 * _re_pass_i)
                _opt_rp._modulation_strength *= max(0.3, 1.0 - 0.25 * _re_pass_i)
                _rp_audio, _rp_result = _opt_rp.optimize(optimiertes_audio)
                _rp_audio = np.clip(
                    np.nan_to_num(_rp_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                _rp_goals = self.messe_ziele(_rp_audio, sr)
                _rp_passed = sum(1 for v in _rp_goals.values() if math.isfinite(v) and v >= _GOAL_MIN)
                # Re-Pass nur übernehmen wenn er Goals verbessert
                if _rp_passed >= passed:
                    optimiertes_audio = _rp_audio
                    goals = _rp_goals
                    finite_vals = [v for v in goals.values() if math.isfinite(v)]
                    score = float(np.mean(finite_vals)) if finite_vals else score
                    passed = _rp_passed
                    improvements.append(
                        f"Re-Pass {_re_pass_i}: {len(_violations)} Violations → {passed}/{len(goals)} Goals bestanden"
                    )
                    logger.info(
                        "ExzellenzDenker Re-Pass %d: %d Violations korrigiert → %d/%d Goals",
                        _re_pass_i,
                        len(_violations),
                        passed,
                        len(goals),
                    )
                else:
                    logger.debug(
                        "ExzellenzDenker Re-Pass %d: keine Verbesserung (%d→%d Goals), abgebrochen",
                        _re_pass_i,
                        passed,
                        _rp_passed,
                    )
                    break  # No improvement → stop re-passing
            except Exception as _rp_exc:
                logger.debug("ExzellenzDenker Re-Pass %d fehlgeschlagen: %s", _re_pass_i, _rp_exc)
                break

        # §2.53a VERSA Re-measurement nach Goal-Re-Pass: optimiertes_audio könnte durch
        # Re-Pass geändert worden sein. VERSA neu messen für post-repair MOS.
        # (Die initiale VERSA-Messung L186–200 war VOR Re-Pass → hier re-messen.)
        if optimiertes_audio.shape != audio.shape or not np.array_equal(optimiertes_audio, audio):
            try:
                from plugins.versa_plugin import score_mos as _score_mos_post

                _mono_post = optimiertes_audio if optimiertes_audio.ndim == 1 else optimiertes_audio.mean(axis=-1)
                _mono_post = np.nan_to_num(_mono_post.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                _vr_post = _score_mos_post(_mono_post, sr)
                _mos_post = float(getattr(_vr_post, "mos", 1.0))
                if math.isfinite(_mos_post) and 1.0 <= _mos_post <= 5.0:
                    logger.info(
                        "ExzellenzDenker VERSA post-repair re-measurement: MOS=%.3f (pre-repair war %.3f)",
                        _mos_post,
                        versa_mos,
                    )
                    versa_mos = _mos_post  # Überschreibe pre-repair Wert
                else:
                    logger.debug("ExzellenzDenker: VERSA post-repair MOS ungültig: %.1f", _mos_post)
            except Exception as _ve:
                logger.debug("ExzellenzDenker: VERSA post-repair Messung fehlgeschlagen: %s", _ve)

        # §2.53 Frisson-Index (Gänsehaut-Propensity) aus Goals-Proxy
        # Literature: Blood & Zatorre 2001, Grewe 2007, Harrison & Loui 2014
        # Proxy-Gewichte analog UV3 _compute_joy_runtime_index() fallback-Pfad
        _fi_micro = float(goals.get("micro_dynamics", 0.0)) if goals else 0.0
        _fi_emo = float(goals.get("emotionalitaet", 0.0)) if goals else 0.0
        # emotional_arc ≈ Mittel aus emotionalitaet + micro_dynamics (kein direkter Goal)
        _fi_arc = 0.5 * _fi_emo + 0.5 * _fi_micro
        _fi_art = float(goals.get("artikulation", 0.0)) if goals else 0.0
        _fi_spa = float(goals.get("spatial_depth", 0.0)) if goals else 0.0
        _fi_trans = float(goals.get("transparenz", 0.0)) if goals else 0.0
        _fi_tonal = float(goals.get("tonal_center", 0.0)) if goals else 0.0
        frisson_index = float(
            np.clip(
                0.26 * _fi_arc
                + 0.18 * _fi_micro
                + 0.14 * _fi_emo
                + 0.14 * _fi_art
                + 0.10 * _fi_spa
                + 0.08 * _fi_trans
                + 0.10 * _fi_tonal,
                0.0,
                1.0,
            )
        )
        logger.info(
            "ExzellenzDenker frisson_index=%.3f (arc=%.2f micro=%.2f emo=%.2f art=%.2f spa=%.2f)",
            frisson_index,
            _fi_arc,
            _fi_micro,
            _fi_emo,
            _fi_art,
            _fi_spa,
        )

        note = (
            f"Exzellenz-Optimierung abgeschlossen: Score {score:.3f}, "
            f"{passed}/{len(goals)} Ziele erfüllt"
            + (f", VERSA MOS={versa_mos:.2f}" if versa_mos > 0.0 else "")
            + (f", Gänsehaut={frisson_index:.0%}" if frisson_index > 0.0 else "")
            + "."
        )

        return ExzellenzErgebnis(
            audio=optimiertes_audio,
            excellence_score=score,
            musical_goals=goals,
            goals_passed=passed,
            goals_total=len(goals),
            improvements=improvements,
            processing_note=note,
            warnings=warnings,
            versa_mos=versa_mos,
            frisson_index=frisson_index,
            mert_proxy_used=bool(_metadata.get("mert_proxy_used", False)),
        )

    def messe_ziele(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Misst alle 15 Musical Goals für das übergebene Audio.

        Args:
            audio: Audio-Signal (mono/stereo, float32)
            sr:    Sample-Rate in Hz

        Returns:
            Dict mit Goal-Namen → Score ∈ [0, 1].
            Leerer Dict bei Fehler (NaN-sicher).
        """
        # NaN/Inf-Schutz
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if audio.size == 0:
            return {}

        import concurrent.futures as _cf_mz

        try:
            checker = self._get_checker()
            # Bug-Fix §10c: thread-based timeout — prevents O(N²) or deadlock
            # in any metric from blocking the pipeline forever.  120 s covers
            # all 15 metrics even on long tracks; normal run < 30 s.
            with _cf_mz.ThreadPoolExecutor(max_workers=1) as _exec_mz:
                _fut_mz = _exec_mz.submit(checker.measure_all, audio, sr)
                try:
                    raw = _fut_mz.result(timeout=120.0)
                except _cf_mz.TimeoutError:
                    logger.warning(
                        "ExzellenzDenker: messe_ziele() Timeout (120 s) — "
                        "leeres Dict zurückgegeben (Goal-Messung abgebrochen)"
                    )
                    return {}
            # Nur finite Werte behalten
            return {k: float(v) for k, v in raw.items() if np.isfinite(v)}
        except Exception as exc:
            logger.warning("ExzellenzDenker: Goal-Messung fehlgeschlagen: %s", exc)
            return {}

    def messe_und_repariere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        mode: str = "restoration",
        material: str = "auto",
        reference_audio: np.ndarray | None = None,
        inapplicable_goals: frozenset[str] | set[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Misst Goals und führt konservative Ziel-Reparatur für P3-P5-Verletzungen durch.

        Strategie (§0 + §2.45):
          1. Goals messen — sofort zurück wenn alle ≥ 0.75 (Minimal-Interventions-Prinzip).
          2. Nur P3-P5-Verletzungen mit Defizit ≥ 0.03 adressieren.
          3. Zeit-Domain-Reparatur (ExcellenceOptimizer, micro_dynamics + ola_edges):
             Kein STFT-Roundtrip-Fehler — sicher nach beliebig vielen Pipeline-STFT-Phasen
             (Ephraim & Malah 1984 gilt nur für spektrale Subtraktion, nicht für
             Hüllkurven-Modulation oder Overlap-Add-Crossfade).
          4. Frequenz-Blend-Reparatur (α ≥ 0.95, falls reference_audio vorhanden):
             Für waerme/brillanz/bass_kraft ohne neue STFT-Kaskade.
             Minimalste Rückmischung (max. 7 % Originalanteil).
          5. Bestes Kandidaten-Audio nur übernehmen wenn:
             goals_passed_after ≥ goals_passed_before UND
             kein Ziel um mehr als 0.02 schlechter als vor der Reparatur (§0).

        Warum sicher nach UV3-FeedbackChain (Abgrenzung zu v9.10.72):
          - v9.10.72 deaktivierte den ExcellenceOptimizer *global* nach UV3, weil alle
            vier Schritte auf STFT-Basis liefen und kaskadieren würden.
          - Hier werden *ausschließlich* Zeit-Domain-Schritte (micro_dynamics, ola_edges)
            verwendet — diese akkumulieren per Definition keine STFT-Rundungsfehler.
          - Der Blend-Schritt mischt Linear-PCM (kein Spektrum) → ebenfalls STFT-frei.

        Args:
            audio:           Verarbeitetes Audio (nach UV3 + FeedbackChain), float32.
            sr:              Sample-Rate (muss 48000 sein).
            material:        Trägermedium für ExcellenceOptimizer-Profile.
            reference_audio: Original-Audio vor der Restaurierung (für Blend-Reparatur).

        Returns:
            (audio_out, goals_dict): Im Erfolgsfall ggf. verbessertes Audio + Goals.
            Im Fehlerfall: (audio unverändert, ursprüngliche Goals oder {}).
        """
        # Mode-adaptive PMGG-Canonical-Schwellen (§9.10.77) — ersetzen den
        # vereinfachten 0.75-Floor. P1-Verletzungen (z.B. natuerlichkeit=0.82 < 0.90)
        # werden jetzt korrekt erkannt. P2: tonal_center < 0.95 wird als Verletzung
        # klassifiziert; Zeit-Domain-Ops helfen nur begrenzt, aber Erkennung ist Pflicht.
        _is_studio = str(mode).lower() in {"studio", "studio_2026", "studio2026"}
        _FALLBACK_MIN: float = 0.75  # Fallback für unbekannte Goals
        try:
            from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds as _gct

            _thresholds: dict[str, float] = _gct(is_studio_2026=_is_studio)
        except Exception:
            _thresholds = {}  # Import-Fehler → Fallback
        _MIN_DEFICIT: float = 0.03  # Reparatur nur wenn Score < threshold − 0.03
        _REPAIR_TRIGGER_FLOOR: float = 0.75  # §2.45: Borderline nahe 0.75 nicht nachbearbeiten

        # P3-P5 goals addressable by time-domain or blend repair (not P1/P2 — those
        # are already protected by UV3's P1/P2 blend cascade §9.8)
        _P3P5_REPAIR_TARGETS: frozenset[str] = frozenset(
            {
                # P3 — time-domain micro_dynamics helps directly
                "micro_dynamics",
                "groove",
                "emotionalitaet",
                # P4 — frequency blend helps (spectral envelope closer to reference)
                "waerme",
                "bass_kraft",
                "transparenz",
                "separation_fidelity",
                # P5 — frequency blend helps
                "brillanz",
                "spatial_depth",
            }
        )
        # Goals that benefit from micro_dynamics injection (time-domain envelope)
        _MICRO_DYN_GOALS: frozenset[str] = frozenset({"micro_dynamics", "groove", "emotionalitaet"})
        # Goals that benefit from OLA-crossfade edge smoothing (time-domain)
        _OLA_GOALS: frozenset[str] = frozenset({"artikulation", "natuerlichkeit"})
        # Goals that benefit from blend with reference (restores spectral warmth/brightness)
        _BLEND_GOALS: frozenset[str] = frozenset({"waerme", "bass_kraft", "brillanz", "transparenz"})

        # §2.32 Inapplicable-Filter: Goals die GAF als physikalisch nicht messbar
        # markiert hat (z.B. brillanz bei BW<8kHz ohne AudioSR, bass_kraft bei
        # vocal-dominanter Quelle) aus Reparatur UND Messung ausschließen.
        _inappl: frozenset[str] = frozenset(inapplicable_goals) if inapplicable_goals else frozenset()
        _eff_repair_targets: frozenset[str] = _P3P5_REPAIR_TARGETS - _inappl

        # NaN/Inf-Schutz
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            return audio, {}

        # Step 1: Basis-Messung
        goals_initial = self.messe_ziele(audio, sr)
        if not goals_initial:
            return audio, {}

        _passed_initial = sum(
            1 for k, v in goals_initial.items() if math.isfinite(v) and v >= _thresholds.get(k, _FALLBACK_MIN)
        )
        _total = len(goals_initial)

        # P3-P5 violations with meaningful deficit (avoids micro-adjustments on borderline goals)
        _p35_violations: set[str] = {
            k
            for k, v in goals_initial.items()
            if k in _eff_repair_targets
            and math.isfinite(v)
            and v < min(_thresholds.get(k, _FALLBACK_MIN), _REPAIR_TRIGGER_FLOOR) - _MIN_DEFICIT
        }
        if not _p35_violations:
            # §2.45: Minimal-Intervention — kein Eingriff wenn Goals erfüllt / Grenzfall
            return audio, goals_initial

        logger.info(
            "ExzellenzDenker messe_und_repariere: %d/%d Goals | %d P3-P5-Verletzungen: %s",
            _passed_initial,
            _total,
            len(_p35_violations),
            ", ".join(sorted(_p35_violations)),
        )

        _best_audio: np.ndarray = audio
        _best_goals: dict[str, float] = dict(goals_initial)
        _best_passed: int = _passed_initial

        def _is_improvement(candidate_goals: dict[str, float]) -> bool:
            """Accept only if goals_passed ≥ before AND no goal regresses > 0.02 (§0)."""
            _cand_passed = sum(
                1 for k, v in candidate_goals.items() if math.isfinite(v) and v >= _thresholds.get(k, _FALLBACK_MIN)
            )
            if _cand_passed < _best_passed:
                return False
            # Regression guard for ALL goals (not just P1/P2)
            for _g, _v_init in goals_initial.items():
                if math.isfinite(_v_init):
                    _v_cand = candidate_goals.get(_g, _v_init)
                    if math.isfinite(_v_cand) and _v_cand < _v_init - 0.02:
                        return False
            return True

        # Step 2: Zeit-Domain-Reparatur (micro_dynamics + ola_edges — kein STFT)
        _needs_td = bool(_p35_violations & (_MICRO_DYN_GOALS | _OLA_GOALS))
        if _needs_td:
            try:
                from backend.core.excellence_optimizer import ExcellenceOptimizer

                _apply_md = bool(_p35_violations & _MICRO_DYN_GOALS)
                _apply_ola = bool(_p35_violations & _OLA_GOALS)
                _opt_td = ExcellenceOptimizer(
                    sample_rate=sr,
                    apply_continuity=False,  # Kein STFT — Ephraim & Malah 1984
                    apply_micro_dynamics=_apply_md,
                    apply_harmonic_boost=False,  # Kein STFT — Ephraim & Malah 1984
                    apply_ola_edges=_apply_ola,
                    material=material,
                )
                # Konservative Stärke: §0 Minimal-Intervention
                _opt_td._modulation_strength = (  # pylint: disable=protected-access
                    getattr(_opt_td, "_modulation_strength", 0.3) * 0.55
                )
                _td_out, _ = _opt_td.optimize(_best_audio)
                _td_out = np.clip(
                    np.nan_to_num(_td_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
                _td_goals = self.messe_ziele(_td_out, sr)
                if _td_goals and _is_improvement(_td_goals):
                    _td_passed = sum(1 for v in _td_goals.values() if math.isfinite(v) and v >= _FALLBACK_MIN)
                    _best_audio = _td_out
                    _best_goals = _td_goals
                    _best_passed = _td_passed
                    logger.info(
                        "ExzellenzDenker Goal-Repair (Zeit-Domain): %d → %d Goals bestanden",
                        _passed_initial,
                        _best_passed,
                    )
                else:
                    logger.debug(
                        "ExzellenzDenker Goal-Repair (Zeit-Domain): keine Verbesserung (%d Goals)",
                        _passed_initial,
                    )
            except Exception as _td_exc:
                logger.debug("ExzellenzDenker Goal-Repair (Zeit-Domain) fehlgeschlagen: %s", _td_exc)

        # Step 3: Frequenz-Blend-Reparatur für waerme/brillanz/bass_kraft/transparenz
        _needs_blend = bool(_p35_violations & _BLEND_GOALS)
        # §0: Rauschbehaftetes Original NICHT remischen — Blend hebt Träger-Rauschen zurück.
        # shellac/wax_cylinder/wire_recording haben SNR < 20 dB → max. 4 % remixte Rauschenergie
        # wäre hörbarer Qualitätsverlust (Primum non nocere).
        _HIGH_NOISE_MATERIALS: frozenset[str] = frozenset({"shellac", "wax_cylinder", "wire_recording"})
        if str(material).lower() in _HIGH_NOISE_MATERIALS:
            _needs_blend = False
            logger.debug("ExzellenzDenker Blend-Reparatur übersprungen — Hochrausch-Träger: %s", material)
        if _needs_blend and reference_audio is not None and reference_audio.shape == _best_audio.shape:
            try:
                _ref_f32 = np.nan_to_num(reference_audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                for _alpha in (0.96, 0.93):
                    _blended = np.clip(
                        _alpha * _best_audio + (1.0 - _alpha) * _ref_f32,
                        -1.0,
                        1.0,
                    )
                    _blend_goals = self.messe_ziele(_blended, sr)
                    if not _blend_goals:
                        continue
                    _blend_passed = sum(1 for v in _blend_goals.values() if math.isfinite(v) and v >= _FALLBACK_MIN)
                    if _is_improvement(_blend_goals) and _blend_passed > _best_passed:
                        _best_audio = _blended
                        _best_goals = _blend_goals
                        _best_passed = _blend_passed
                        logger.info(
                            "ExzellenzDenker Goal-Repair (Blend α=%.2f): %d → %d Goals bestanden",
                            _alpha,
                            _passed_initial,
                            _best_passed,
                        )
                        break  # Erste Verbesserung genügt (§2.45 Minimal-Intervention)
            except Exception as _bl_exc:
                logger.debug("ExzellenzDenker Goal-Repair (Blend) fehlgeschlagen: %s", _bl_exc)

        return _best_audio, _best_goals

    def prognostiziere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        defect_result: object | None = None,
        material: str = "auto",
    ) -> dict[str, float]:
        """Schätzt Goal-Risikowahrscheinlichkeiten VOR der Restaurierung.

        Schnelle DSP-Proxies (≤ 200 ms) als Frühwarnsystem für den
        PhaseInteractionDenker. Ermöglicht prophylaktische Phasen-Aktivierung
        bevor UV3 läuft — statt P1/P2-Verletzungen erst danach zu heilen (§2.45a).

        Kein ML, kein STFT-Roundtrip — reine Spektralstatistik auf Kurzfenster.

        Gemessene Risiken:
            Rauschboden       → Natuerlichkeit / Authentizitaet
            HF-Verlust        → Brillanz / Timbre
            Transient-Armut   → Groove / MicroDynamics
            Dropout-Schwere   → Artikulation

        Returns:
            Dict goal_name → Risikowahrscheinlichkeit ∈ [0, 1].
            0.0 = sicher, 1.0 = sicher verletzt ohne schützende Phase.
            Leeres Dict bei internem Fehler (fail-safe).
        """
        try:
            audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if audio.size == 0:
                return {}

            mono = audio if audio.ndim == 1 else audio.mean(axis=0)
            risk: dict[str, float] = {}

            # ── Rauschboden → Natuerlichkeit / Authentizitaet ─────────────────────
            # 5. Perzentil der FFT-Magnituden als Rauschboden-Proxy.
            # Referenz: -70 dBFS = sauber; -40 dBFS = hohes Rauschen.
            try:
                n_fft = min(4096, mono.size)
                if n_fft >= 128:
                    fft_mag = np.abs(np.fft.rfft(mono[:n_fft]))
                    noise_floor_db = float(np.percentile(20.0 * np.log10(fft_mag + 1e-10), 5))
                    # Lineares Risiko zwischen -70 dBFS (0.0) und -40 dBFS (1.0)
                    noise_risk = float(np.clip((-noise_floor_db - 40.0) / 30.0, 0.0, 1.0))
                    risk["natuerlichkeit"] = noise_risk
                    risk["authentizitaet"] = round(noise_risk * 0.80, 3)
            except Exception:
                pass

            # ── HF-Verlust → Brillanz / Timbre ───────────────────────────
            # Welch-PSD: Energie 8 kHz+ / Breitband-Energie.
            # Referenz: CD (1990er) ≅ 0.10-0.25; Shellac ≅ 0.01-0.03.
            try:
                if sr > 0 and mono.size >= sr // 8:
                    from scipy.signal import welch as _welch

                    _nperseg = min(1024, mono.size)
                    freqs, psd = _welch(mono, fs=sr, nperseg=_nperseg)
                    bb_energy = float(np.sum(psd[freqs >= 20])) + 1e-10
                    hf_energy = float(np.sum(psd[freqs >= 8000]))
                    hf_ratio = hf_energy / bb_energy
                    # Risiko steigt wenn hf_ratio < 0.08 (≥ Kassetten/Shellac-Niveau)
                    hf_risk = float(np.clip(1.0 - hf_ratio / 0.08, 0.0, 1.0))
                    risk["brillanz"] = hf_risk
                    risk["timbre"] = round(hf_risk * 0.70, 3)
            except Exception:
                pass

            # ── Transient-Armut → Groove / MicroDynamics ─────────────────
            # Onset-Rate in 10-ms-Fenstern. Referenz: lebendige Musik ≥ 1.5 Onsets/s.
            # Tiefes Rauschen oder starker Dropout senken die Rate.
            try:
                if mono.size >= 4800:
                    hop = int(sr * 0.010)  # 10 ms
                    max_frames = min(len(mono) // hop, 1000)  # max 10 s analysieren
                    env = np.abs(mono[: max_frames * hop])
                    frames = env.reshape(max_frames, hop)
                    frame_rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-8
                    onsets = float(np.sum(frame_rms[1:] > 3.0 * frame_rms[:-1]))
                    duration_s = max_frames * hop / max(sr, 1)
                    onset_rate = onsets / max(duration_s, 0.001)
                    # Risiko steigt wenn Onset-Rate < 0.8/s (stark gedämpftes Signal)
                    transient_risk = float(np.clip(1.0 - onset_rate / 0.8, 0.0, 1.0))
                    risk["groove"] = round(transient_risk * 0.70, 3)
                    risk["micro_dynamics"] = round(transient_risk * 0.80, 3)
            except Exception:
                pass

            # ── Dropout-Schwere → Artikulation ──────────────────────────
            # Dropout-Severity direkt aus DefectResult (falls verfügbar).
            try:
                if defect_result is not None:
                    scores = getattr(defect_result, "scores", {}) or {}
                    dropout_sev = 0.0
                    for k, v in scores.items():
                        if "dropout" in str(k).lower():
                            sev = float(getattr(v, "severity", 0.0) if not isinstance(v, (int, float)) else v)
                            dropout_sev = max(dropout_sev, sev)
                    if dropout_sev > 0.0:
                        risk["artikulation"] = float(np.clip(dropout_sev, 0.0, 1.0))
            except Exception:
                pass

            logger.debug(
                "ExzellenzDenker.prognostiziere(): material=%s %d Risikofelder: %s",
                material,
                len(risk),
                {k: f"{v:.2f}" for k, v in risk.items()},
            )
            return risk
        except Exception as exc:
            logger.debug("ExzellenzDenker.prognostiziere() fehlgeschlagen: %s", exc)
            return {}

    # ── Interne Hilfsmethoden ────────────────────────────────────────────────

    def _get_optimizer(self, sr: int = 48_000, material: str = "auto") -> Any:
        """Double-Checked Locking — lazy ExcellenceOptimizer-Init.

        Der Optimizer wird je sr+material neu erstellt wenn sr/material sich ändern.
        Für die häufigsten Fälle (konstantes sr=48000, material="auto") wird
        die bestehende Instanz wiederverwendet.
        """
        if self._optimizer is not None:
            # Quick-check ob sample_rate und material noch stimmen
            existing_sr = getattr(self._optimizer, "sample_rate", None)
            existing_mat = getattr(self._optimizer, "material", None)
            if existing_sr == sr and existing_mat == material.lower().strip():
                return self._optimizer

        with self._opt_lock:
            # Erneuter Check unter Lock
            if self._optimizer is not None:
                existing_sr = getattr(self._optimizer, "sample_rate", None)
                existing_mat = getattr(self._optimizer, "material", None)
                if existing_sr == sr and existing_mat == material.lower().strip():
                    return self._optimizer
            self._optimizer = self._build_optimizer(sr, material)
        return self._optimizer

    @staticmethod
    def _build_optimizer(sr: int, material: str) -> Any:
        """Erstellt einen neuen ExcellenceOptimizer."""
        from backend.core.excellence_optimizer import ExcellenceOptimizer  # lazy import

        return ExcellenceOptimizer(
            sample_rate=sr,
            apply_continuity=True,
            apply_micro_dynamics=True,
            apply_harmonic_boost=True,
            apply_ola_edges=True,
            material=material,
            use_mert=material.lower().strip()
            in {  # M-7b: MERT nur für Vokal-Material
                "gesang",
                "sprache",
                "vocal",
                "singer",
                "speech",
                "voice",
            },
        )

    def _get_checker(self) -> Any:
        """Double-Checked Locking — lazy MusicalGoalsChecker-Init."""
        if self._checker is None:
            with self._checker_lock:
                if self._checker is None:
                    self._checker = self._build_checker()
        return self._checker

    @staticmethod
    def _build_checker() -> Any:
        """Erstellt MusicalGoalsChecker."""
        from backend.core.musical_goals.musical_goals_metrics import (  # lazy import
            MusicalGoalsChecker,
        )

        return MusicalGoalsChecker()

    @staticmethod
    def _fallback(audio: np.ndarray, grund: str) -> ExzellenzErgebnis:
        """Fallback-Ergebnis bei leerem oder fehlerhaftem Audio."""
        return ExzellenzErgebnis(
            audio=audio,
            excellence_score=0.0,
            musical_goals={},
            goals_passed=0,
            goals_total=0,
            improvements=[],
            processing_note=f"Keine Optimierung möglich: {grund}",
            warnings=[grund],
        )


# ─── Modul-Level-Singleton (Double-Checked Locking — Spec §3.2) ─────────────

_instance: ExzellenzDenker | None = None
_lock = threading.Lock()


def get_exzellenz_denker() -> ExzellenzDenker:
    """Thread-sicherer Singleton-Accessor für den ExzellenzDenker."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ExzellenzDenker()
    return _instance


def optimiere(
    audio: np.ndarray,
    sr: int = 48_000,
    *,
    material: str = "auto",
) -> ExzellenzErgebnis:
    """Convenience-Wrapper: Exzellenz-Optimierung über Singleton."""
    return get_exzellenz_denker().optimiere(audio, sr, material=material)


def messe_ziele(audio: np.ndarray, sr: int = 48_000) -> dict[str, float]:
    """Convenience-Wrapper: Musical Goals messen über Singleton."""
    return get_exzellenz_denker().messe_ziele(audio, sr)
