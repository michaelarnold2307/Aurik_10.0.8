"""
denker/exzellenz_denker.py — Exzellenz-Denker für Aurik 9
==========================================================

Verbessert restauriertes Audio bis an die physikalische Qualitätsgrenze:
  • ExcellenceOptimizer: Spectral Continuity, Micro-Dynamics, Harmonic Boost, OLA
  • MusicalGoalsChecker: Messung aller 14 musikalischen Qualitätsziele

Spec §1.2, §2.5, §2.29, §8.1 — v9.10.45
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading

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
    UnifiedVocalAIEnhancer = None  # type: ignore[assignment,misc]
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
        self._optimizer: object | None = None
        self._opt_lock = threading.Lock()
        self._checker: object | None = None
        self._checker_lock = threading.Lock()

    # ── Öffentliche API ──────────────────────────────────────────────────────

    def optimiere(
        self,
        audio: np.ndarray,
        sr: int = 48_000,
        *,
        material: str = "auto",
        messe_ziele_vorab: bool = True,
    ) -> ExzellenzErgebnis:
        """Optimiert Audio auf CEDAR Cambridge-Niveau und misst alle 14 Goals.

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
                if UnifiedVocalAIEnhancer is None:
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

        # Schritt 2 — Musical Goals messen
        goals = self.messe_ziele(optimiertes_audio, sr)

        # Excellence-Score: Mittelwert aller Goals
        _GOAL_MIN = 0.75  # Mindest-Score je Goal (Spec §8.x)
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

        note = (
            f"Exzellenz-Optimierung abgeschlossen: Score {score:.3f}, "
            f"{passed}/{len(goals)} Ziele erfüllt" + (f", VERSA MOS={versa_mos:.2f}" if versa_mos > 0.0 else "") + "."
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
        )

    def messe_ziele(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Misst alle 14 Musical Goals für das übergebene Audio.

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

        try:
            checker = self._get_checker()
            raw = checker.measure_all(audio, sr)
            # Nur finite Werte behalten
            return {k: float(v) for k, v in raw.items() if np.isfinite(v)}
        except Exception as exc:
            logger.warning("ExzellenzDenker: Goal-Messung fehlgeschlagen: %s", exc)
            return {}

    # ── Interne Hilfsmethoden ────────────────────────────────────────────────

    def _get_optimizer(self, sr: int = 48_000, material: str = "auto") -> object:
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
    def _build_optimizer(sr: int, material: str) -> object:
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

    def _get_checker(self) -> object:
        """Double-Checked Locking — lazy MusicalGoalsChecker-Init."""
        if self._checker is None:
            with self._checker_lock:
                if self._checker is None:
                    self._checker = self._build_checker()
        return self._checker

    @staticmethod
    def _build_checker() -> object:
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
    global _instance
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
