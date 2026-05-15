"""Global calibration helpers for derived universal meta-parameters (§09.10).

All functions are pure and bounded. They only use existing pipeline inputs
and are safe to call from gates and target estimators.
"""

from __future__ import annotations

import numpy as np


def compute_tcci(transfer_chain: list[str] | None) -> float:
    """Transfer-Chain-Complexity-Index in [0, 1]."""
    chain = [str(m).strip().lower() for m in (transfer_chain or []) if str(m).strip()]
    n = max(1, len(chain))
    lossy = sum(1 for m in chain if m in {"mp3_low", "aac", "streaming"})
    analog = sum(
        1 for m in chain if m in {"wax_cylinder", "shellac", "vinyl", "tape", "cassette", "wire_recording", "reel_tape"}
    )
    score = 0.18 * float(n - 1) + 0.22 * float(lossy) + 0.10 * float(max(0, analog - 1))
    return float(np.clip(score, 0.0, 1.0))


def compute_ibs(restorability: float, defect_severity_mean: float, tcci: float) -> float:
    """Intervention-Budget-Scalar in [0.15, 0.95]."""
    r = 1.0 - float(np.clip(restorability / 100.0, 0.0, 1.0))
    d = float(np.clip(defect_severity_mean, 0.0, 1.0))
    c = float(np.clip(tcci, 0.0, 1.0))
    budget = 0.55 * r + 0.30 * d + 0.15 * c
    return float(np.clip(budget, 0.15, 0.95))


def blend_targets_with_confidence(
    canonical: dict[str, float],
    song_targets: dict[str, float],
    medium_conf: float,
    era_conf: float,
    genre_conf: float,
) -> dict[str, float]:
    """Blend per-song targets with canonical floors based on confidence."""
    conf = float(np.clip(0.45 * medium_conf + 0.30 * era_conf + 0.25 * genre_conf, 0.0, 1.0))
    blended: dict[str, float] = {}
    for goal, floor in canonical.items():
        t = float(song_targets.get(goal, floor))
        blended[goal] = float((1.0 - conf) * float(floor) + conf * t)
    return blended


def compute_cpb(material_ceiling: float, current_value: float, mode: str) -> float:
    """Ceiling-Proximity-Budget in [0, material_ceiling]."""
    mc = max(0.0, float(material_ceiling))
    cv = max(0.0, float(current_value))
    margin = max(0.0, mc - cv)
    safety = 0.70 if str(mode).strip().lower() == "restoration" else 0.50
    return float(np.clip(safety * margin, 0.0, mc))


def compute_retry_temperature(restorability: float, tcci: float, artifact_freedom_score: float) -> float:
    """Retry aggressiveness temperature in [0, 1]."""
    hard_song = 1.0 - float(np.clip(restorability / 100.0, 0.0, 1.0))
    chain = float(np.clip(tcci, 0.0, 1.0))
    artifact_risk = 1.0 - float(np.clip(artifact_freedom_score, 0.0, 1.0))
    t = 0.50 * hard_song + 0.30 * chain + 0.20 * artifact_risk
    return float(np.clip(t, 0.0, 1.0))


def compute_export_reliability(
    hpi: float,
    artifact_freedom: float,
    passed_goals: int,
    total_goals: int,
    reference_confidence: float,
) -> float:
    """Export reliability score in [0, 1]."""
    total = int(total_goals)
    ratio = 0.0 if total <= 0 else float(np.clip(float(passed_goals) / float(total), 0.0, 1.0))
    score = (
        0.35 * float(np.clip(hpi, 0.0, 1.0))
        + 0.30 * float(np.clip(artifact_freedom, 0.0, 1.0))
        + 0.20 * ratio
        + 0.15 * float(np.clip(reference_confidence, 0.0, 1.0))
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_goal_coverage_index(musical_goals_passed: dict[str, bool] | None) -> float:
    """Priority-weighted musical-goal coverage in [0, 1]."""
    passed = dict(musical_goals_passed or {})
    if not passed:
        return 0.0

    weights = {
        # P1
        "natuerlichkeit": 1.4,
        "authentizitaet": 1.4,
        # P2
        "tonal_center": 1.2,
        "tonalcenter": 1.2,
        "timbre_authentizitaet": 1.2,
        "artikulation": 1.2,
        # P3
        "emotionalitaet": 1.0,
        "mikrodynamik": 1.0,
        "micro_dynamics": 1.0,
        "groove": 1.0,
        # P4
        "transparenz": 0.8,
        "waerme": 0.8,
        "basskraft": 0.8,
        "bass_kraft": 0.8,
        "separation_fidelity": 0.8,
        # P5
        "brillanz": 0.6,
        "raumtiefe": 0.6,
        "spatial_depth": 0.6,
    }

    score = 0.0
    total = 0.0
    for g, ok in passed.items():
        k = str(g).strip().lower()
        w = float(weights.get(k, 1.0))
        total += w
        if bool(ok):
            score += w

    if total <= 0.0:
        return 0.0
    return float(np.clip(score / total, 0.0, 1.0))


def compute_reference_confidence(target_confidence: float, tcci: float, carrier_chain_recovery_ratio: float) -> float:
    """Calibrated reference confidence in [0, 1] from existing reliability signals."""
    tc = float(np.clip(target_confidence, 0.0, 1.0))
    chain_stability = 1.0 - float(np.clip(tcci, 0.0, 1.0))
    # High carrier-recovery-ratio means stronger intentional divergence from degraded input.
    # This lowers confidence in strict input-referenced proxies.
    carrier_stability = 1.0 - float(np.clip(carrier_chain_recovery_ratio / 0.35, 0.0, 1.0))
    conf = 0.65 * tc + 0.20 * chain_stability + 0.15 * carrier_stability
    return float(np.clip(conf, 0.0, 1.0))


def compute_recovery_pressure_index(
    fallback_attempts: int,
    rollback_count: int,
    goal_deficit_ratio: float,
) -> float:
    """Recovery pressure in [0, 1] based on recovery attempts, rollbacks and missing goals."""
    fa = float(np.clip(float(fallback_attempts) / 3.0, 0.0, 1.0))
    rb = float(np.clip(float(rollback_count) / 8.0, 0.0, 1.0))
    gd = float(np.clip(goal_deficit_ratio, 0.0, 1.0))
    rpi = 0.40 * fa + 0.35 * rb + 0.25 * gd
    return float(np.clip(rpi, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §09.1 Kanonische Schwellwerte (Single Source of Truth)
# ---------------------------------------------------------------------------

CANONICAL_THRESHOLDS_RESTORATION: dict[str, float] = {
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    "tonal_center": 0.95,
    "tonalcenter": 0.95,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.85,
    "emotionalitaet": 0.82,
    "mikrodynamik": 0.88,
    "micro_dynamics": 0.88,
    "groove": 0.83,
    "transparenz": 0.82,
    "waerme": 0.75,
    "bass_kraft": 0.78,
    "basskraft": 0.78,
    "separation_fidelity": 0.78,
    "brillanz": 0.78,
    "raumtiefe": 0.70,
    "spatial_depth": 0.70,
}

CANONICAL_THRESHOLDS_STUDIO2026: dict[str, float] = {
    "natuerlichkeit": 0.92,
    "authentizitaet": 0.90,
    "tonal_center": 0.96,
    "tonalcenter": 0.96,
    "timbre_authentizitaet": 0.89,
    "artikulation": 0.87,
    "emotionalitaet": 0.84,
    "mikrodynamik": 0.90,
    "micro_dynamics": 0.90,
    "groove": 0.85,
    "transparenz": 0.85,
    "waerme": 0.78,
    "bass_kraft": 0.80,
    "basskraft": 0.80,
    "separation_fidelity": 0.80,
    "brillanz": 0.82,
    "raumtiefe": 0.74,
    "spatial_depth": 0.74,
}

# ---------------------------------------------------------------------------
# §09.2 Per-Song Goal-Targets — Era × Material × Genre Bias-Tabellen
# ---------------------------------------------------------------------------

_ERA_BIAS: dict[str, dict[str, float]] = {
    "1920s": {
        "brillanz": -0.28,
        "transparenz": -0.18,
        "raumtiefe": -0.14,
        "waerme": +0.14,
        "authentizitaet": +0.10,
        "natuerlichkeit": +0.08,
    },
    "1950s": {
        "brillanz": -0.14,
        "transparenz": -0.08,
        "waerme": +0.10,
        "authentizitaet": +0.08,
    },
    "1970s": {
        "brillanz": +0.04,
        "transparenz": +0.04,
        "waerme": +0.02,
    },
    "1990s": {
        "brillanz": +0.10,
        "transparenz": +0.10,
        "artikulation": +0.06,
        "waerme": -0.04,
    },
}

_MATERIAL_BIAS: dict[str, dict[str, float]] = {
    # Ultra-analog (Shellac, Wax, Wire, Lacquer)
    # P1/P2: Authentizität und Timbre stark richtungsgebend (Trägersignatur), aber BW stark limitiert.
    # P3/P4: Physikalische Grenzen des Trägers — keine SGT-Kalibrierung ohne diese Biases führt
    # dazu, dass groove/bass_kraft/sep_fidelity auf kanonischen Werten (0.83/0.78/0.78) verbleiben,
    # die für Shellac/Wax physikalisch unerreichbar sind (§0a Material-Ceiling Ebene 1).
    "ultra_analog": {
        "brillanz": -0.24,
        "transparenz": -0.12,
        "waerme": +0.10,
        "authentizitaet": +0.10,
        # §09.2 Material-adaptive floor: Shellac SNR~15dB, BW~7kHz → natuerlichkeit physikalisch ≤0.72
        # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.72-0.90)/0.27 = -0.667
        "natuerlichkeit": -0.67,
        # P3 — dynamisch/rhythmisch limitiert durch Wow/Flutter, surface noise floor, limited DR
        "groove": -0.18,  # Wow/Flutter, Crackle-Onsets maskieren rhythmische Präzision
        "micro_dynamics": -0.14,  # Rauschboden und limitierte DR reduzieren Dynamik-Headroom
        "emotionalitaet": -0.08,  # Begrenzte tonale + dynamische Bandbreite reduziert Arousal-Range
        # P4 — spektral/strukturell limitiert
        "bass_kraft": -0.22,  # LF-Rolloff: Shellac ≤ 100 Hz praktisch, Wax ≤ 80 Hz (§0 §6.2c)
        "separation_fidelity": -0.24,  # Mono oder Mono-kompatibles Narrow-Stereo → keine Stem-Sep möglich
        "raumtiefe": -0.15,  # Mono-Quelle: kein echtes Stereo-Feld → Raumtiefe inherent limitiert
    },
    # Normal-analog Vinyl (LP, Vinyl)
    # Vinyl: Schneidebeschränkungen <80 Hz (bass_kraft), leichtes Wow/Flutter (groove)
    "analog": {
        "waerme": +0.10,
        "brillanz": -0.06,
        "authentizitaet": +0.08,
        # §09.2 Material-adaptive floor: Vinyl SNR~55dB, BW~16kHz → natuerlichkeit physikalisch ≤0.82
        # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.82-0.90)/0.27 = -0.296
        "natuerlichkeit": -0.30,
        # P3/P4: milde Biases — physikalische Einschränkungen des Carriers ohne Extremwerte
        "groove": -0.04,  # Leichtes Wow/Flutter
        "bass_kraft": -0.06,  # Vinyl-Schneidebeschränkung LF
        "separation_fidelity": -0.06,  # Narrow-Stereo, Early-Stereo-Artefakte
    },
    # Tape (Reel-Tape, Kassette) — physikalisch stärker limitiert als Vinyl
    # §09.2 (v9.12.5): Separatklasse tape_analog, da Tape-spezifische Einschränkungen
    # sich grundlegend von Vinyl unterscheiden:
    # - Tape-Hiss-NR reduziert HF stärker als Vinyl-Rausch-NR → brillanz Ceiling tiefer
    # - Wow/Flutter bei Tape stärker → groove/artikulation eingeschränkter
    # - Tape-Sättigung (H2/H4) ist authentic, aber ISO-226-gewichtetes Wärme-Verhältnis
    #   bleibt niedrig (Waerme-Bias 0.0, nicht +0.10 wie bei Vinyl) — Metrik-Divisor
    #   übernimmt die Anpassung (§9.12.8 WaermeMetric material_type)
    # - Narrow/Mono-Stereo → SepFidelity stark eingeschränkt
    "tape_analog": {
        "waerme": +0.00,  # Kein Bias: Metrik-Divisor (§9.12.8) übernimmt Anpassung
        "brillanz": -0.22,  # Tape-Hiss-NR reduziert HF: floor ~0.72 bei kappa_min
        "authentizitaet": +0.06,  # Tape-Sättigung ist authentisch
        "natuerlichkeit": -0.30,  # Gleich wie Vinyl (SNR~40-50 dB nach NR)
        "groove": -0.08,  # Stärkeres Wow/Flutter als Vinyl
        "bass_kraft": -0.06,  # Tape-Bias-Sättigung LF
        "separation_fidelity": -0.22,  # Tape Narrow/Mono: SDR<3dB triggert ref-free path
        "artikulation": -0.15,  # Dropout+Hiss maskiert Transienten
        "timbre_authentizitaet": -0.04,
        "spatial_depth": -0.04,
    },
    # Digital (CD, DAT, Streaming) — near-lossless; natuerlichkeit at full canonical floor 0.90
    "digital": {
        "transparenz": +0.08,
        "artikulation": +0.06,
        "brillanz": +0.06,
    },
    # Lossy (mp3_low, mp3_high, aac, minidisc) — codec artifacts reduce naturalness
    # §09.2 Material-adaptive floor: mp3_low 128kbps → natuerlichkeit physikalisch ≤0.78
    # Formula: floor = canonical(0.90) + kappa_min(0.27) * bias → bias = (0.78-0.90)/0.27 = -0.444
    "lossy": {
        "natuerlichkeit": -0.44,
        "transparenz": +0.04,
        "artikulation": +0.04,
        "brillanz": +0.04,
    },
}

_MATERIAL_CLASS: dict[str, str] = {
    "wax_cylinder": "ultra_analog",
    "shellac": "ultra_analog",
    "wire_recording": "ultra_analog",
    "lacquer_disc": "ultra_analog",
    "vinyl": "analog",
    "lp": "analog",
    "tape": "tape_analog",  # §09.2 (v9.12.5): eigene Klasse, nicht mit Vinyl zusammen
    "reel_tape": "tape_analog",  # §09.2 (v9.12.5): tape_analog mit korrekten HF/Sep-Böden
    "cassette": "tape_analog",  # §09.2 (v9.12.5): Kassette physikalisch näher an Tape als Vinyl
    "kassette": "tape_analog",  # Alias
    "cd_digital": "digital",
    "cd": "digital",
    "dat": "digital",
    "mp3_low": "lossy",
    "mp3_high": "lossy",
    "aac": "lossy",
    "minidisc": "lossy",
    "streaming": "lossy",
}

_GENRE_BIAS: dict[str, dict[str, float]] = {
    "klassik": {
        "raumtiefe": +0.18,
        "natuerlichkeit": +0.12,
        "mikrodynamik": +0.10,
        "brillanz": -0.08,
    },
    "jazz": {
        "waerme": +0.12,
        "natuerlichkeit": +0.10,
        "authentizitaet": +0.10,
        "transparenz": -0.04,
    },
    "schlager": {
        # bass_kraft −0.32: Schlager hat physikalisch geringen Bassanteil (heller Vokal-Mix,
        # Vinyl-Schneidebeschränkungen <80 Hz). bass_ratio typisch 0.002–0.005 statt 0.05
        # → Score max. ~0.20 ohne künstliche Bass-Anhebung (§0-Verletzung).
        # Threshold 0.78 → 0.46 ist genre-realistisch und §0-konform.
        "waerme": +0.10,
        "groove": +0.06,
        "authentizitaet": +0.08,
        "bass_kraft": -0.32,
    },
    "pop": {
        # bass_kraft −0.12: Heller Pop-Mix (1970s–1990s) hat weniger Bassanteil als
        # Rock/Electronic. Threshold leicht senken ohne §0-Verletzung.
        "transparenz": +0.08,
        "artikulation": +0.08,
        "brillanz": +0.08,
        "bass_kraft": -0.12,
    },
    "rock": {
        "bass_kraft": +0.08,
        "mikrodynamik": +0.06,
        "groove": +0.08,
    },
    "electronic": {
        "transparenz": +0.10,
        "bass_kraft": +0.10,
        "brillanz": +0.08,
    },
    "folk": {
        "natuerlichkeit": +0.12,
        "authentizitaet": +0.10,
        "waerme": +0.08,
    },
}


def _era_key(decade: int | None) -> str:
    """Map decade to bias bucket."""
    if decade is None:
        return "1970s"
    if decade < 1950:
        return "1920s"
    if decade < 1970:
        return "1950s"
    if decade < 1990:
        return "1970s"
    return "1990s"


def estimate_song_goal_targets(
    *,
    is_studio_2026: bool = False,
    goal_weights: dict[str, float] | None = None,
    restorability_score: float = 70.0,
    era_decade: int | None = None,
    genre_label: str | None = None,
    material_type: str | None = None,
    transfer_chain: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-song goal targets as studio-day reconstruction targets.

    Returns a dict mapping each goal name to its target value ∈ [0.30, 0.99].
    Targets are blended from canonical floors (§09.1), era-/material-/genre-biases
    (§09.2), goal importance weights (§2.56) and restorability.

    These targets indicate where the pipeline *should stop* — the reconstructed
    studio-day score for this specific song.  They are NOT hard gates; they inform
    PhaseConductor strength recommendations and PMGG over-processing detection.

    Args:
        is_studio_2026: Studio 2026 mode uses higher canonical floors.
        goal_weights:   Per-song goal importance from §2.56 (1.0 = default).
        restorability_score: 0–100 from RestorabilityEstimator.
        era_decade:     Decade (e.g. 1970) from EraClassifier.
        genre_label:    Genre string (e.g. "schlager") from GenreClassifier.
        material_type:  Primary material (e.g. "vinyl") from MediumDetector.
        transfer_chain: Full chain list (e.g. ["vinyl","tape","mp3_low"]).

    Returns:
        dict[str, float]: Per-goal targets, same keys as CANONICAL_THRESHOLDS.
    """
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio_2026 else CANONICAL_THRESHOLDS_RESTORATION
    weights = goal_weights or {}
    rest_norm = float(np.clip(restorability_score / 100.0, 0.0, 1.0))

    # Resolve material from chain if not given directly
    primary_mat = (str(material_type or "").strip().lower()) or (
        str((transfer_chain or [""])[0]).strip().lower() if transfer_chain else ""
    )
    mat_class = _MATERIAL_CLASS.get(primary_mat, "analog")
    era_bucket = _era_key(era_decade)
    genre_key = str(genre_label or "").strip().lower()

    # Accumulate biases
    bias: dict[str, float] = {}
    for b_dict in [
        _ERA_BIAS.get(era_bucket, {}),
        _MATERIAL_BIAS.get(mat_class, {}),
        _GENRE_BIAS.get(genre_key, {}),
    ]:
        for goal, delta in b_dict.items():
            bias[goal] = bias.get(goal, 0.0) + float(delta)

    # §Bug#2 Alias-Propagation: bias tables use legacy keys (raumtiefe, mikrodynamik,
    # tonalcenter, basskraft). Canonical dicts carry both old and new keys.
    # Propagate biases so new-key lookups (spatial_depth, micro_dynamics, etc.) work.
    _OLD_TO_NEW: dict[str, str] = {
        "raumtiefe": "spatial_depth",
        "mikrodynamik": "micro_dynamics",
        "tonalcenter": "tonal_center",
        "basskraft": "bass_kraft",
    }
    _NEW_TO_OLD: dict[str, str] = {v: k for k, v in _OLD_TO_NEW.items()}
    for old_k, new_k in _OLD_TO_NEW.items():
        if old_k in bias and new_k not in bias:
            bias[new_k] = bias[old_k]
        elif new_k in bias and old_k not in bias:
            bias[old_k] = bias[new_k]

    # kappa: how strongly biases are applied (low restorability → more conservative)
    # Restoration: 0.45; Studio 2026: 0.65; modulated by restorability
    kappa_base = 0.65 if is_studio_2026 else 0.45
    kappa = kappa_base * (0.60 + 0.40 * rest_norm)  # range: [0.27, 0.65] Restoration

    targets: dict[str, float] = {}
    for goal, floor in canonical.items():
        b = bias.get(goal, 0.0)
        w = float(weights.get(goal, 1.0))
        # goal weight > 1.0 → song needs this goal → stay closer to or above floor
        # goal weight < 1.0 → goal less important for this song → can tolerate lower target
        weight_shift = (w - 1.0) * 0.06  # ±0.06 max per unit weight deviation
        target = floor + kappa * b + weight_shift
        # Hard bounds: never below 0.30, never above 0.99 (1.0 is unreachable)
        targets[goal] = float(np.clip(target, 0.30, 0.99))

    return targets


# ---------------------------------------------------------------------------
# §09.7 Expected Quality Score (UI Baseline Prediction)
# ---------------------------------------------------------------------------

_MATERIAL_QUALITY_CEILING: dict[str, float] = {
    "wax_cylinder": 0.55,
    "shellac": 0.70,
    "lacquer_disc": 0.68,
    "wire_recording": 0.65,
    "vinyl": 0.88,
    "lp": 0.88,
    "tape": 0.85,
    "reel_tape": 0.86,
    "cassette": 0.80,
    "cd_digital": 0.95,
    "cd": 0.95,
    "dat": 0.92,
    "minidisc": 0.85,
    "mp3_low": 0.78,
    "mp3_high": 0.88,
}


# ---------------------------------------------------------------------------
# §09.8 Material-adaptive goal floors — get_material_floor (RELEASE_MUST)
# ---------------------------------------------------------------------------


def get_material_floor(
    material_type: str,
    goal: str,
    is_studio_2026: bool = False,
) -> float:
    """Return the minimum achievable goal floor for a given material type (§09.1).

    Unlike the canonical thresholds (CD-equivalent), this returns the
    material-adjusted minimum floor accounting for physical medium limits.
    REQUIRED: callers MUST use this instead of CANONICAL_THRESHOLDS directly
    when evaluating goals against material-specific limits (§0a, §2.44, §2.45b).

    Args:
        material_type: e.g. "shellac", "vinyl", "cd_digital"
        goal: goal key e.g. "brillanz", "natuerlichkeit"
        is_studio_2026: mode flag

    Returns:
        float: minimum achievable floor ∈ [0.30, 0.99]

    Examples:
        shellac brillanz → ~0.72 (physical BW limit)
        vinyl  brillanz → ~0.76
        cd     brillanz → ~0.80
    """
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio_2026 else CANONICAL_THRESHOLDS_RESTORATION
    floor = float(canonical.get(goal, 0.70))

    mat = str(material_type or "").strip().lower()
    mat_class = _MATERIAL_CLASS.get(mat, "analog")

    # Apply material bias at minimum restorability kappa (worst-case → lowest floor).
    # kappa_min = kappa_base * 0.60 (restorability=0, §09.2 kappa range).
    kappa_base = 0.65 if is_studio_2026 else 0.45
    kappa_min = kappa_base * 0.60

    bias_val = float(_MATERIAL_BIAS.get(mat_class, {}).get(goal, 0.0))
    material_floor = floor + kappa_min * bias_val

    # §0a/§9.12.3 Studio2026-Invarianz: Studio2026-Floor darf NIEMALS unter dem
    # Restoration-Floor für dasselbe Material liegen. Studio2026 ist eine Obermenge
    # von Restoration (macht alles was Restoration tut + Enhancement). Wenn der größere
    # kappa_base (0.65 vs 0.45) zusammen mit negativem Material-Bias die Studio2026-
    # Formel unter den Restoration-Wert drückt, ist das ein Kalibrierungsfehler:
    # vinyl/natuerlichkeit: studio=0.92-0.39*0.30=0.803 < restoration=0.90-0.27*0.30=0.819.
    # Fix: Studio2026-Floor = max(computed, restoration_floor) → Monotonie-Invariante.
    if is_studio_2026:
        rest_canonical = float(CANONICAL_THRESHOLDS_RESTORATION.get(goal, 0.70))
        rest_kappa_min = 0.45 * 0.60  # kappa_min für Restoration
        rest_floor = rest_canonical + rest_kappa_min * bias_val
        rest_floor = float(np.clip(rest_floor, 0.30, 0.99))
        material_floor = max(material_floor, rest_floor)

    return float(np.clip(material_floor, 0.30, 0.99))


# ---------------------------------------------------------------------------
# §09.9 Material-adaptive phase strength ranges — get_phase_strength_range
# ---------------------------------------------------------------------------

# (min_strength, max_strength) per material class and phase.
# max_strength caps over-processing on fragile media (§2.45b, §0a BW-Ceiling).
# min_strength ensures the phase is effective enough to be worth running.
# Phases not listed fall back to _DEFAULT_STRENGTH_RANGE.
_PHASE_STRENGTH_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "ultra_analog": {
        # Shellac / wax_cylinder / wire_recording / lacquer_disc
        # BW ≤ 8 kHz, SNR ~15 dB, Mono — aggressive processing would hallucinate.
        "phase_03_denoise": (0.15, 0.60),  # OMLSA/DFN: hard cap — SNR ~15 dB
        "phase_06_frequency_restoration": (0.05, 0.35),  # BW-Ceiling ≤ 8 kHz (§6.2c)
        "phase_07_harmonic_restoration": (0.05, 0.30),  # BW-Ceiling strict
        "phase_09_crackle_removal": (0.20, 0.75),  # Key phase for shellac/vinyl crackle
        "phase_12_wow_flutter_fix": (0.10, 0.55),  # Wow/Flutter present but gentle
        "phase_20_reverb_reduction": (0.05, 0.20),  # Studio early reflections protect
        "phase_23_spectral_repair": (0.10, 0.50),  # Spectral gaps limited
        "phase_26_dynamic_range_expansion": (0.00, 0.25),  # DR ceiling shellac ≤ 45 dB
        "phase_29_tape_hiss_reduction": (0.10, 0.55),  # Surface noise, not tape hiss
        "phase_35_multiband_compression": (0.05, 0.20),  # Gentle — original dynamics key
        "phase_46_spatial_enhancement": (0.00, 0.10),  # Mono source — no stereo widening
        "phase_48_stereo_width_enhancer": (0.00, 0.00),  # VERBOTEN on mono material
        "phase_49_advanced_dereverb": (0.05, 0.20),  # Early reflections cap §2.46f
        "phase_55_diffusion_inpainting": (0.10, 0.45),  # Dropout repair — careful
    },
    "analog": {
        # Vinyl / tape / reel_tape / cassette
        # Moderate limitations — standard restoration range.
        "phase_03_denoise": (0.10, 0.75),
        "phase_06_frequency_restoration": (0.10, 0.60),  # BW vinyl ≤ 16 kHz
        "phase_07_harmonic_restoration": (0.10, 0.55),
        "phase_09_crackle_removal": (0.20, 0.85),
        "phase_12_wow_flutter_fix": (0.15, 0.70),
        "phase_20_reverb_reduction": (0.05, 0.35),
        "phase_23_spectral_repair": (0.15, 0.70),
        "phase_26_dynamic_range_expansion": (0.00, 0.50),  # vinyl ≤ 70 dB
        "phase_29_tape_hiss_reduction": (0.15, 0.70),
        "phase_35_multiband_compression": (0.05, 0.40),
        "phase_49_advanced_dereverb": (0.05, 0.35),
        "phase_55_diffusion_inpainting": (0.15, 0.65),
    },
    "lossy": {
        # mp3_low / mp3_high / aac / minidisc
        # Codec artifacts — spectral repair is key, denoise moderate.
        "phase_03_denoise": (0.05, 0.55),
        "phase_06_frequency_restoration": (0.15, 0.70),  # HF reconstruction primary
        "phase_07_harmonic_restoration": (0.15, 0.65),
        "phase_23_spectral_repair": (0.25, 0.90),  # Primary phase for lossy
        "phase_35_multiband_compression": (0.05, 0.40),
        "phase_50_spectral_repair": (0.25, 0.85),
    },
    "digital": {
        # cd_digital / dat — near-lossless, minimal intervention (§2.45b)
        "phase_03_denoise": (0.05, 0.35),
        "phase_06_frequency_restoration": (0.05, 0.40),
        "phase_07_harmonic_restoration": (0.05, 0.35),
        "phase_09_crackle_removal": (0.05, 0.30),
        "phase_23_spectral_repair": (0.10, 0.55),
        "phase_26_dynamic_range_expansion": (0.00, 0.40),
        "phase_35_multiband_compression": (0.05, 0.30),
        "phase_49_advanced_dereverb": (0.05, 0.30),
    },
}

_DEFAULT_STRENGTH_RANGE: tuple[float, float] = (0.05, 1.0)


def get_phase_strength_range(
    phase_id: str,
    material_type: str | None = None,
    restorability_score: float = 70.0,
) -> tuple[float, float]:
    """Return (min_strength, max_strength) for a phase given material and restorability.

    The max_strength caps over-processing on fragile material (§0a, §0h).
    The min_strength ensures the phase applies enough correction to be meaningful.
    High restorability reduces max_strength toward near-passthrough (§2.45b).

    Args:
        phase_id: e.g. "phase_03_denoise"
        material_type: e.g. "shellac", "vinyl", "cd_digital"
        restorability_score: 0–100 from RestorabilityEstimator

    Returns:
        tuple[float, float]: (min_strength, max_strength) both ∈ [0.0, 1.0]
    """
    mat = str(material_type or "").strip().lower()
    mat_class = _MATERIAL_CLASS.get(mat, "analog")

    mat_ranges = _PHASE_STRENGTH_RANGES.get(mat_class, {})
    min_s, max_s = mat_ranges.get(phase_id, _DEFAULT_STRENGTH_RANGE)

    # §2.45b: high restorability → near-passthrough → scale down max_strength.
    # Above 80 the scaling is linear from 1.0 to 0.30 at restorability=100.
    rest = float(np.clip(restorability_score, 0.0, 100.0))
    if rest > 80.0:
        passthrough_factor = 1.0 - 0.70 * ((rest - 80.0) / 20.0)
        passthrough_factor = float(np.clip(passthrough_factor, 0.30, 1.0))
        max_s = float(np.clip(max_s * passthrough_factor, min_s, max_s))

    return (float(min_s), float(max_s))


def predict_quality_score(
    material_type: str,
    restorability: float,
    defect_severity_mean: float,
    is_studio_2026: bool = False,
) -> float:
    """Predict expected OQS (Overall Quality Score) after full pipeline.

    Pure function — safe to call from UI/Bridge before processing starts.
    Returns a value ∈ [0.0, 0.99].
    """
    mat = str(material_type or "").strip().lower()
    ceiling = _MATERIAL_QUALITY_CEILING.get(mat, 0.75)
    rest_norm = float(np.clip(restorability / 100.0, 0.0, 1.0))
    defect_penalty = float(np.clip(defect_severity_mean, 0.0, 1.0)) * 0.15
    studio_boost = 0.08 if is_studio_2026 else 0.0
    score = rest_norm * ceiling - defect_penalty + studio_boost
    return float(np.clip(score, 0.0, 0.99))


__all__ = [
    "blend_targets_with_confidence",
    "compute_cpb",
    "compute_export_reliability",
    "compute_goal_coverage_index",
    "compute_ibs",
    "compute_recovery_pressure_index",
    "compute_reference_confidence",
    "compute_retry_temperature",
    "compute_tcci",
    "estimate_song_goal_targets",
    "get_material_floor",
    "get_phase_strength_range",
    "predict_quality_score",
]
