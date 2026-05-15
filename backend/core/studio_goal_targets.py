"""
backend/core/studio_goal_targets.py
Aurik 9 — Per-song studio-day goal targets for phase steering.

Computes target values for all 14 musical goals so PMGG can evaluate whether a
phase moves toward or away from the intended studio-day profile instead of only
checking absolute score drops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.calibration_matrix import (
    blend_targets_with_confidence,
    compute_ibs,
    compute_tcci,
)
from backend.core.material_canonical import canonical_material_key

_TARGET_VERSION = "v1.1"
_GOAL_KEY_ALIASES: dict[str, str] = {
    "spatial_depth": "raumtiefe",
}

# §09.11 PHYSICAL_CEILING — Physikalisch maximal erreichbare Goal-Scores pro Material.
# Nur Materialien/Goals mit nachgewiesenen physikalischen Grenzen eingetragen.
# Leeres Dict = kein Ceiling (CD/DAT/FLAC).
_PHYSICAL_CEILING: dict[str, dict[str, float]] = {
    "shellac": {
        "brillanz": 0.72,
        "transparenz": 0.72,
        "spatial_depth": 0.55,
        "raumtiefe": 0.55,
        "artikulation": 0.78,
        "separation_fidelity": 0.60,
    },
    "wax_cylinder": {
        "brillanz": 0.55,
        "transparenz": 0.60,
        "spatial_depth": 0.45,
        "raumtiefe": 0.45,
        "artikulation": 0.70,
    },
    "vinyl": {
        "brillanz": 0.86,
        "transparenz": 0.84,
        "spatial_depth": 0.80,
        "raumtiefe": 0.80,
    },
    "tape": {
        # §9.12.7 [BUG-FIX v9.12.7] Ceiling-Rekalibrierung nach material-adaptiver Formel:
        # Die material-adaptive BrillanzMetric-Formel (§9.12.7) kalibriert den HF-Crest
        # auf das tape-spezifische Rausch-/Signalverhältnis um. Mit G_floor=0.22 und
        # crest_peak≈8 ergibt die neue Formel score≈0.67 (vorher: 0.20 mit CD-Formel).
        # Ceilings angepasst auf neue Formeldynamik:
        #   tape/cassette: brillanz ≤0.78 (crest_peak≈12 → 0.816 bei exzellenter Kassette)
        #   transparenz ≤0.50 (Band-Crest-Faktor mit Tape-Rauschboden)
        "brillanz": 0.78,
        "transparenz": 0.50,
    },
    "reel_tape": {
        # §9.12.7: Reel-Tape mit besserem SNR; reel_tape-Formel (offset=0.05, divisor=1.40).
        # crest_peak≈15 → (1.176-0.05)/1.40 = 0.804; exzellent crest_peak≈20 → 0.893.
        "brillanz": 0.85,
        "transparenz": 0.62,
    },
    "mp3_low": {
        "brillanz": 0.80,
        "transparenz": 0.78,
        "artikulation": 0.82,
    },
}


@dataclass(frozen=True)
class SongGoalTargets:
    """Container for per-song goal targets used by phase steering."""

    targets: dict[str, float]
    confidence: float
    derived: dict[str, float] | None = None
    version: str = _TARGET_VERSION


def _safe_float(v: object, default: float = 1.0) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except Exception:
        return default
    if not np.isfinite(f):
        return default
    return f


def _safe_int(v: object, default: int | None = None) -> int | None:
    try:
        if v is None:
            return default
        _f = float(v)  # type: ignore[arg-type]
        return int(_f)
    except Exception:
        return default


def _canonical_goal_key(goal: object) -> str:
    key = str(goal or "").strip().lower()
    return _GOAL_KEY_ALIASES.get(key, key)


def _lookup_goal_value(mapping: dict[str, float], goal: str) -> float:
    if goal in mapping:
        return float(mapping[goal])
    canonical = _canonical_goal_key(goal)
    if canonical in mapping:
        return float(mapping[canonical])
    for alias, canonical_name in _GOAL_KEY_ALIASES.items():
        if canonical_name == goal and alias in mapping:
            return float(mapping[alias])
    return 0.0


def _goal_bias_from_era(era_decade: int | None) -> dict[str, float]:
    if era_decade is None:
        return {}
    if era_decade <= 1949:
        return {
            "brillanz": -0.28,
            "transparenz": -0.18,
            "raumtiefe": -0.14,
            "waerme": 0.14,
            "authentizitaet": 0.10,
            "natuerlichkeit": 0.08,
        }
    if era_decade <= 1969:
        return {
            "brillanz": -0.14,
            "transparenz": -0.08,
            "waerme": 0.10,
            "authentizitaet": 0.08,
        }
    if era_decade >= 1990:
        return {
            "brillanz": 0.10,
            "transparenz": 0.10,
            "artikulation": 0.06,
            "waerme": -0.04,
        }
    return {
        "brillanz": 0.04,
        "transparenz": 0.04,
        "waerme": 0.02,
    }


def _goal_bias_from_genre(genre_label: str) -> dict[str, float]:
    g = str(genre_label or "").strip().lower()
    if g in {"klassik", "oper"}:
        return {
            "raumtiefe": 0.18,
            "natuerlichkeit": 0.12,
            "mikrodynamik": 0.10,
            "brillanz": -0.08,
        }
    if g in {"jazz", "folk"}:
        return {
            "waerme": 0.12,
            "natuerlichkeit": 0.10,
            "authentizitaet": 0.10,
            "transparenz": -0.04,
        }
    if g in {"reggae", "gospel"}:
        return {
            "raumtiefe": 0.10,
            "waerme": 0.08,
            "brillanz": -0.06,
        }
    if g in {"pop", "electronic", "hip-hop", "rock"}:
        return {
            "transparenz": 0.08,
            "artikulation": 0.08,
            "brillanz": 0.08,
        }
    return {}


def _goal_bias_from_material(material_type: object) -> dict[str, float]:
    mk = canonical_material_key(material_type)
    if mk in {"shellac", "wax_cylinder", "wire_recording"}:
        return {
            "brillanz": -0.24,
            "transparenz": -0.12,
            "waerme": 0.10,
            "authentizitaet": 0.10,
        }
    if mk in {"vinyl", "tape", "reel_tape", "cassette"}:
        return {
            "waerme": 0.10,
            "brillanz": -0.06,
            "authentizitaet": 0.08,
        }
    if mk in {"cd_digital", "dat", "streaming", "aac", "mp3_high", "flac"}:
        return {
            "transparenz": 0.08,
            "artikulation": 0.06,
            "brillanz": 0.06,
        }
    return {}


def estimate_song_goal_targets(
    *,
    is_studio_2026: bool,
    goal_weights: dict[str, float] | None,
    restorability_score: float | None,
    era_decade: int | None = None,
    genre_label: str = "",
    material_type: object = None,
    transfer_chain: list[str] | None = None,
) -> SongGoalTargets:
    """Estimate per-song targets for all goals.

    Formula:
        target = clip(
            floor + kappa * max(0, weight-1) * (upper-floor),
            floor,
            upper,
        )

    Where:
        floor = canonical restoration threshold
        upper = min(0.97, studio_threshold + 0.06)
        kappa = 0.45 (restoration), 0.65 (studio_2026)
    """
    from backend.core.per_phase_musical_goals_gate import (  # pylint: disable=import-outside-toplevel
        _CANONICAL_THRESHOLDS_STUDIO2026,
        _get_canonical_thresholds,
    )

    floors = dict(_get_canonical_thresholds(False))
    studio = dict(_CANONICAL_THRESHOLDS_STUDIO2026)
    weights = {_canonical_goal_key(k): v for k, v in (goal_weights or {}).items()}
    era_i = _safe_int(era_decade, None)
    _bias_era = _goal_bias_from_era(era_i)
    _bias_genre = _goal_bias_from_genre(genre_label)
    _bias_material = _goal_bias_from_material(material_type)
    _chain_depth = len(transfer_chain or [])

    kappa = 0.65 if is_studio_2026 else 0.45
    context_blend = 0.45 if is_studio_2026 else 0.60
    targets: dict[str, float] = {}

    for g, floor in floors.items():
        upper = float(min(0.97, _safe_float(studio.get(g, floor), floor) + 0.06))
        w = float(np.clip(_safe_float(weights.get(g, 1.0), 1.0), 0.30, 2.00))
        extra = max(0.0, w - 1.0)
        t = floor + kappa * extra * (upper - floor)
        _bias = float(
            _lookup_goal_value(_bias_era, g)
            + _lookup_goal_value(_bias_genre, g)
            + _lookup_goal_value(_bias_material, g)
        )
        t += context_blend * _bias * (upper - floor)
        targets[g] = float(np.clip(t, floor, upper))

    rest = float(np.clip(_safe_float(restorability_score, 50.0), 0.0, 100.0))
    tcci = compute_tcci(transfer_chain)
    defect_proxy = float(np.clip(1.0 - (rest / 100.0), 0.0, 1.0))
    ibs = compute_ibs(rest, defect_proxy, tcci)

    # Guard against over-driving targets on hard material: high intervention budget
    # softly pulls target deltas back toward canonical floors.
    if ibs > 0.60:
        pullback = float(np.clip((ibs - 0.60) / 0.35, 0.0, 1.0))
        for g, floor in floors.items():
            tg = float(targets.get(g, floor))
            targets[g] = float((1.0 - 0.35 * pullback) * tg + (0.35 * pullback) * float(floor))

    # Conservative confidence envelope to allow fallback in edge-cases.
    conf = float(np.clip(0.55 + 0.40 * (rest / 100.0), 0.55, 0.95))
    if _chain_depth >= 3:
        conf = float(np.clip(conf - 0.05 - min(0.08, 0.02 * float(_chain_depth - 3)), 0.45, 0.95))
    conf = float(np.clip(conf - 0.10 * tcci, 0.35, 0.95))

    # Target-Confidence-Blend (§09.10c): uncertain context falls back to canonical floors.
    targets = blend_targets_with_confidence(
        canonical=floors,
        song_targets=targets,
        medium_conf=conf,
        era_conf=conf,
        genre_conf=conf,
    )

    for alias, canonical in _GOAL_KEY_ALIASES.items():
        if canonical in targets:
            targets[alias] = float(targets[canonical])

    # §09.11 PHYSICAL_CEILING-Clamp: Restoration targets must not exceed the
    # physically achievable maximum for the carrier material.
    # Studio 2026 is exempt (deliberate enhancement beyond carrier limits is allowed).
    if not is_studio_2026:
        mk = canonical_material_key(material_type)
        material_ceiling = _PHYSICAL_CEILING.get(mk, {})
        for g, ceiling_val in material_ceiling.items():
            if g in targets and targets[g] > ceiling_val:
                targets[g] = float(ceiling_val)

    return SongGoalTargets(
        targets=targets,
        confidence=conf,
        derived={
            "tcci": float(tcci),
            "ibs": float(ibs),
        },
    )


__all__ = ["SongGoalTargets", "estimate_song_goal_targets"]
