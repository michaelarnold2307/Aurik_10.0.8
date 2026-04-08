"""
backend/core/goal_applicability_filter.py
Aurik 9 -- Spec §2.32: GoalApplicabilityFilter

Filtert physikalisch nicht messbare Musical Goals heraus.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoalApplicabilityResult:
    """Spec §2.32"""

    applicable: frozenset[str]
    inapplicable: frozenset[str]
    reasons: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        """Serialisierungsformat f\u00fcr Logging und API."""
        return {
            "applicable": sorted(self.applicable),
            "inapplicable": sorted(self.inapplicable),
            "reasons": dict(self.reasons),
        }


_ALWAYS_APPLICABLE: frozenset[str] = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "timbre_authentizitaet",
        "artikulation",
    }
)

_ALL_GOALS: frozenset[str] = frozenset(
    {
        "brillanz",
        "waerme",
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "bass_kraft",
        "groove",
        "spatial_depth",
        "timbre_authentizitaet",
        "tonal_center",
        "micro_dynamics",
        "separation_fidelity",
        "artikulation",
    }
)

_MONO_MATERIALS = {
    "wax_cylinder",
    "wire_recording",
    "shellac",
    "lacquer_disc",
}


class GoalApplicabilityFilter:
    """Spec §2.32: Filtert physikalisch nicht messbare Ziele.

    Deaktivierungs-Regeln:
        SpatialDepthMetric:  Mono-Aufnahme (decade <= 1950 oder mono-Material)
        BrillanzMetric:      Quell-BW < 8 kHz
        TonalCenterMetric:   wax_cylinder (Fix K v9.10.100: SNR-Bedingung entfernt —
                             K-S-Key-Detection ist SNR-invariant gemäß §9.7.11)
        GrooveMetric:        <10 s oder keine Percussion
        MicroDynamicsMetric: <20 s oder stark komprimiert
        SeparationFidelityMetric: Mono oder <2 Instrumente
    """

    ALWAYS_APPLICABLE: frozenset[str] = _ALWAYS_APPLICABLE

    def evaluate(
        self,
        audio: np.ndarray | None = None,
        sr: int = 48000,
        material: str = "unknown",
        era_decade: int | None = None,
        panns_tags: dict[str, float] | None = None,
        audiosr_available: bool = False,
    ) -> GoalApplicabilityResult:
        """Spec §2.32: Wertet aus, welche Goals anwendbar sind.

        Args:
            audio: float32 ndarray
            sr: Sample-Rate
            material: MaterialType-Label
            era_decade: Jahrzehnt (aus EraClassifier)
            panns_tags: PANNs-Tag-Konfidenz Dict
            audiosr_available: ob AudioSR geladen ist

        Returns:
            GoalApplicabilityResult
        """
        inapplicable: dict[str, str] = {}

        # Audio-Basis-Analysen
        duration_s = 0.0
        bw_hz = 20000.0
        is_mono_signal = False

        if audio is not None:
            arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
            if arr.ndim == 2:
                # Normierung auf (channels, samples) — toleriert (N, C) und (C, N)
                if arr.shape[0] > arr.shape[1]:  # (samples, channels) → transponieren
                    arr = arr.T
                duration_s = arr.shape[1] / sr
                # Stereo-Analyse
                if arr.shape[0] >= 2:
                    corr = float(np.corrcoef(arr[0], arr[1])[0, 1])
                    # Schwellwert 0.995 (vorher 0.97 — zu strict: typische
                    # Popmusik mit zentriertem Gesang hat corr ≈ 0.97–0.99
                    # und ist dennoch kein Mono-Signal).
                    is_mono_signal = bool(np.isnan(corr) or corr >= 0.995)
                else:
                    is_mono_signal = True
                mono = arr.mean(axis=0)
            else:
                duration_s = len(arr) / sr
                is_mono_signal = True
                mono = arr

            # SNR-Schätzung
            if len(mono) >= 512:
                spec = np.abs(np.fft.rfft(mono[: min(len(mono), 8192)])) ** 2 + 1e-15
                freqs = np.fft.rfftfreq(min(len(mono), 8192), d=1.0 / sr)
                sorted_e = np.sort(spec)
                n_floor = max(1, len(sorted_e) // 10)
                noise_e = float(np.mean(sorted_e[:n_floor]))
                signal_e = float(np.mean(sorted_e[-n_floor:]))
                10.0 * math.log10(max(signal_e / max(noise_e, 1e-15), 1e-15))

                # Effektive Bandbreite: höchste Frequenz mit Energie > Rauschboden + 20 dB.
                # Vorheriger Algorithmus (99th-Perzentil-Energie-Schwerpunkt) unterschätzte
                # die BW für typische Musik systematisch: Bassenergie dominiert →
                # 99 % der Energie unterhalb ~2–4 kHz → brillanz fälschlich deaktiviert.
                noise_floor_db = 10.0 * math.log10(float(np.percentile(spec, 10)) + 1e-15)
                bw_hz = 0.0
                for _bi in range(len(freqs) - 1, 0, -1):
                    _e_db = 10.0 * math.log10(float(spec[_bi]) + 1e-15)
                    if _e_db > noise_floor_db + 18.0:  # 18 dB über Rauschboden
                        bw_hz = float(freqs[_bi])
                        break

        # REGEL: SpatialDepthMetric
        era_mono = era_decade is not None and era_decade <= 1950
        mat_mono = material in _MONO_MATERIALS
        # §GoalApplicability Mono-Fix: Materialien vor 1960 die in _MONO_MATERIALS sind,
        # werden auch dann als mono behandelt, wenn die Signal-Korrelation < 0.97 liegt
        # (z. B. Schellack mit Raumambience über Stereo-A/D-Wandler).
        mat_era_mono = mat_mono and (era_decade is not None and era_decade <= 1960)
        if era_mono or is_mono_signal or mat_mono or mat_era_mono:
            inapplicable["spatial_depth"] = "Mono-Aufnahme — Raumtiefe nicht messbar."

        # REGEL: BrillanzMetric
        if bw_hz < 8000.0 and not audiosr_available:
            inapplicable["brillanz"] = (
                "Hochfrequenz war nicht aufgezeichnet — Brillanz wird nach Bandbreiten-Erweiterung neu bewertet."
            )

        # REGEL: TonalCenterMetric
        # Fix K (v9.10.100): SNR-Bedingung entfernt — K-S-Key-Detection ist SNR-invariant
        # gemäß §9.7.11; Deaktivierung bei SNR < −5 dB war inkonsistent mit der
        # K-S-Invarianz-Aussage und hätte tonal_center auf stark degradiertem Material
        # blind abgeschaltet. Nur WAX_CYLINDER wird weiterhin deaktiviert (proprietäres
        # Tonleitersystem mit festen K, kein Western-Durtonart-Profil anwendbar).
        if material == "wax_cylinder":
            inapplicable["tonal_center"] = (
                "Wachswalze — K-S-Durtonart-Profil nicht anwendbar: proprietäres Tonleitersystem, kein Western-Key."
            )

        # REGEL: GrooveMetric
        no_percussion = (
            panns_tags is not None and panns_tags.get("Percussion", 0.0) < 0.15 and panns_tags.get("Drum", 0.0) < 0.15
        )
        if duration_s < 10.0 or no_percussion:
            reason = "Zu kurz fuer Beat-Tracking." if duration_s < 10.0 else "Kein messbares Rhythmusmuster erkannt."
            inapplicable["groove"] = reason

        # REGEL: MicroDynamicsMetric
        # §2.47: < 10 s → Groove/MicroDyn off (spec-konform; vorher fälschlich 20 s)
        if duration_s < 10.0:
            inapplicable["micro_dynamics"] = "Zu kurz fuer LUFS-Profil-Korrelation (<10 s)."

        # REGEL: SeparationFidelityMetric
        if is_mono_signal or mat_mono:
            inapplicable["separation_fidelity"] = "Mono-Quelle — kein mehrkanaliges Signal auflösbar."

        # _ALWAYS_APPLICABLE NIE deaktivieren
        for g in _ALWAYS_APPLICABLE:
            inapplicable.pop(g, None)

        applicable = _ALL_GOALS - frozenset(inapplicable.keys())

        return GoalApplicabilityResult(
            applicable=frozenset(applicable),
            inapplicable=frozenset(inapplicable.keys()),
            reasons=inapplicable,
        )


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: GoalApplicabilityFilter | None = None
_lock = threading.Lock()


def get_goal_filter() -> GoalApplicabilityFilter:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoalApplicabilityFilter()
    return _instance


def evaluate_goal_applicability(
    audio: np.ndarray | None = None,
    sr: int = 48000,
    material: str = "unknown",
    era_decade: int | None = None,
    panns_tags: dict[str, float] | None = None,
    audiosr_available: bool = False,
) -> GoalApplicabilityResult:
    """Convenience-Funktion."""
    return get_goal_filter().evaluate(
        audio=audio,
        sr=sr,
        material=material,
        era_decade=era_decade,
        panns_tags=panns_tags,
        audiosr_available=audiosr_available,
    )


# ── Öffentliche Aliase (Spec §2.32 — ohne führenden Unterstrich) ────────────
ALL_GOALS: frozenset[str] = _ALL_GOALS
ALWAYS_APPLICABLE: frozenset[str] = _ALWAYS_APPLICABLE

# Spec-konformer Funktionsname-Alias (§3.2)
get_goal_applicability_filter = get_goal_filter  # Alias für get_goal_filter() — Spec-konformer Name (§3.2)

__all__ = [
    "ALL_GOALS",
    "ALWAYS_APPLICABLE",
    "_ALL_GOALS",
    "_ALWAYS_APPLICABLE",
    "GoalApplicabilityFilter",
    "GoalApplicabilityResult",
    "evaluate_goal_applicability",
    "get_goal_applicability_filter",
    "get_goal_filter",
]
