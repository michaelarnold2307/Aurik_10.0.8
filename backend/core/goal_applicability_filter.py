"""
backend/core/goal_applicability_filter.py
Aurik 9 -- Spec §2.32: GoalApplicabilityFilter

Filtert physikalisch nicht messbare Musical Goals heraus.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Dict, FrozenSet, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoalApplicabilityResult:
    """Spec §2.32"""

    applicable: FrozenSet[str]
    inapplicable: FrozenSet[str]
    reasons: Dict[str, str]

    def as_dict(self) -> Dict[str, object]:
        """Serialisierungsformat f\u00fcr Logging und API."""
        return {
            "applicable": sorted(self.applicable),
            "inapplicable": sorted(self.inapplicable),
            "reasons": dict(self.reasons),
        }


_ALWAYS_APPLICABLE: FrozenSet[str] = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "timbre_authentizitaet",
        "artikulation",
    }
)

_ALL_GOALS: FrozenSet[str] = frozenset(
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
        TonalCenterMetric:   SNR < -5 dB oder wax_cylinder
        GrooveMetric:        <10 s oder keine Percussion
        MicroDynamicsMetric: <20 s oder stark komprimiert
        SeparationFidelityMetric: Mono oder <2 Instrumente
    """

    ALWAYS_APPLICABLE: FrozenSet[str] = _ALWAYS_APPLICABLE

    def evaluate(
        self,
        audio: Optional[np.ndarray] = None,
        sr: int = 48000,
        material: str = "unknown",
        era_decade: Optional[int] = None,
        panns_tags: Optional[Dict[str, float]] = None,
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
        inapplicable: Dict[str, str] = {}

        # Audio-Basis-Analysen
        duration_s = 0.0
        snr_db = 30.0
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
                    is_mono_signal = bool(np.isnan(corr) or corr >= 0.97)
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
                snr_db = 10.0 * math.log10(max(signal_e / max(noise_e, 1e-15), 1e-15))

                # Effektive Bandbreite
                cumsum = np.cumsum(spec[::-1])[::-1]
                total = cumsum[0]
                if total > 0:
                    for i, f in enumerate(freqs):
                        if cumsum[i] / total < 0.01:
                            bw_hz = float(f)
                            break

        # REGEL: SpatialDepthMetric
        era_mono = era_decade is not None and era_decade <= 1950
        mat_mono = material in _MONO_MATERIALS
        if era_mono or is_mono_signal or mat_mono:
            inapplicable["spatial_depth"] = "Mono-Aufnahme — Raumtiefe nicht messbar."

        # REGEL: BrillanzMetric
        if bw_hz < 8000.0 and not audiosr_available:
            inapplicable["brillanz"] = (
                "Hochfrequenz war nicht aufgezeichnet — " "Brillanz wird nach Bandbreiten-Erweiterung neu bewertet."
            )

        # REGEL: TonalCenterMetric
        if snr_db < -5.0 or material == "wax_cylinder":
            inapplicable["tonal_center"] = (
                "Aufnahme zu stark beschaedigt, um die Tonart zuverlaessig "
                "zu vergleichen — Tonalitaet wird geschuetzt, nicht gemessen."
            )

        # REGEL: GrooveMetric
        no_percussion = (
            panns_tags is not None and panns_tags.get("Percussion", 0.0) < 0.15 and panns_tags.get("Drum", 0.0) < 0.15
        )
        if duration_s < 10.0 or no_percussion:
            reason = "Zu kurz fuer Beat-Tracking." if duration_s < 10.0 else "Kein messbares Rhythmusmuster erkannt."
            inapplicable["groove"] = reason

        # REGEL: MicroDynamicsMetric
        if duration_s < 20.0:
            inapplicable["micro_dynamics"] = "Zu kurz fuer LUFS-Profil-Korrelation (<20 s)."

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

_instance: Optional[GoalApplicabilityFilter] = None
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
    audio: Optional[np.ndarray] = None,
    sr: int = 48000,
    material: str = "unknown",
    era_decade: Optional[int] = None,
    panns_tags: Optional[Dict[str, float]] = None,
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
ALL_GOALS: FrozenSet[str] = _ALL_GOALS
ALWAYS_APPLICABLE: FrozenSet[str] = _ALWAYS_APPLICABLE

# Spec-konformer Funktionsname-Alias (§3.2)
get_goal_applicability_filter = get_goal_filter
"""Alias für get_goal_filter() — Spec-konformer Name (§3.2)."""

__all__ = [
    "GoalApplicabilityFilter",
    "GoalApplicabilityResult",
    "ALL_GOALS",
    "ALWAYS_APPLICABLE",
    "get_goal_applicability_filter",
    "get_goal_filter",
    "evaluate_goal_applicability",
    "_ALWAYS_APPLICABLE",
    "_ALL_GOALS",
]
