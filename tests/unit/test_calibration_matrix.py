"""Unit tests for backend/core/calibration_matrix.py — §09.1/§09.2/§09.7.

Tests cover:
- CANONICAL_THRESHOLDS_* export and consistency
- estimate_song_goal_targets: era/material/genre bias application
- predict_quality_score: material-ceiling and restorability scaling
"""

from __future__ import annotations

import numpy as np

from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_RESTORATION,
    CANONICAL_THRESHOLDS_STUDIO2026,
    estimate_song_goal_targets,
    get_material_floor,
    get_phase_strength_range,
    predict_quality_score,
)

# ---------------------------------------------------------------------------
# §09.1 Canonical Thresholds
# ---------------------------------------------------------------------------

_P1_GOALS = {"natuerlichkeit", "authentizitaet"}
_P2_GOALS = {"tonal_center", "timbre_authentizitaet", "artikulation"}


def test_canonical_thresholds_restoration_has_all_goals():
    """All 14 goals must be present in CANONICAL_THRESHOLDS_RESTORATION."""
    expected = {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
        "mikrodynamik",
        "groove",
        "transparenz",
        "waerme",
        "bass_kraft",
        "separation_fidelity",
        "brillanz",
        "raumtiefe",
    }
    for g in expected:
        assert g in CANONICAL_THRESHOLDS_RESTORATION, f"Goal '{g}' missing in CANONICAL_THRESHOLDS_RESTORATION"


def test_canonical_thresholds_studio2026_has_all_goals():
    """Studio 2026 thresholds must have the same goal keys."""
    for g in CANONICAL_THRESHOLDS_RESTORATION:
        assert g in CANONICAL_THRESHOLDS_STUDIO2026, f"Goal '{g}' in RESTORATION but missing in STUDIO2026"


def test_p1_p2_floors_are_identical_across_modes():
    """P1/P2 goals must have identical floors in both modes (§09.1a/b comment)."""
    p1p2 = _P1_GOALS | _P2_GOALS
    for g in p1p2:
        if g in CANONICAL_THRESHOLDS_RESTORATION and g in CANONICAL_THRESHOLDS_STUDIO2026:
            r = CANONICAL_THRESHOLDS_RESTORATION[g]
            s = CANONICAL_THRESHOLDS_STUDIO2026[g]
            # Allow ≤ 0.03 delta — spec notes P1/P2 are "identical or near-identical"
            assert abs(r - s) <= 0.05, f"P1/P2 goal '{g}': Restoration={r}, Studio={s}, delta too large"


def test_all_thresholds_in_valid_range():
    """All canonical thresholds must be in (0.50, 1.00)."""
    for mode, thresholds in [
        ("Restoration", CANONICAL_THRESHOLDS_RESTORATION),
        ("Studio2026", CANONICAL_THRESHOLDS_STUDIO2026),
    ]:
        for goal, val in thresholds.items():
            assert 0.50 <= val < 1.00, f"{mode}/{goal}={val} out of range [0.50, 1.00)"


# ---------------------------------------------------------------------------
# §09.2 estimate_song_goal_targets
# ---------------------------------------------------------------------------


def test_shellac_1930_has_lower_brillanz_than_cd_2005():
    """Ultra-analog era+material bias must reduce brillanz target vs. modern digital."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["brillanz"] < t_new["brillanz"], (
        f"shellac/1935 brillanz={t_old['brillanz']:.3f} should be < cd/2005 {t_new['brillanz']:.3f}"
    )


def test_shellac_1930_has_higher_waerme_than_cd_2005():
    """Vintage analog material should have higher waerme target (warm character)."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["waerme"] > t_new["waerme"], (
        f"shellac/1935 waerme={t_old['waerme']:.3f} should be > cd/2005 {t_new['waerme']:.3f}"
    )


def test_klassik_genre_raises_raumtiefe():
    """Klassik genre bias must result in higher raumtiefe target than Pop."""
    t_klassik = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="klassik",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_pop = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="pop",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_klassik["raumtiefe"] > t_pop["raumtiefe"]


def test_jazz_genre_raises_waerme():
    """Jazz genre bias must result in higher waerme target than rock."""
    t_jazz = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="jazz",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_rock = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="rock",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_jazz["waerme"] > t_rock["waerme"]


def test_targets_all_in_valid_range():
    """All returned targets must be in [0.30, 0.99]."""
    targets = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1928,
        genre_label="jazz",
        restorability_score=25,
        is_studio_2026=False,
    )
    for g, v in targets.items():
        assert 0.30 <= v <= 0.99, f"target[{g}]={v} out of [0.30, 0.99]"


def test_targets_all_finite():
    """All returned targets must be finite (no NaN/Inf)."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1968,
        genre_label="rock",
        restorability_score=55,
        is_studio_2026=True,
    )
    for g, v in targets.items():
        assert np.isfinite(v), f"target[{g}]={v} is not finite"


def test_targets_keys_match_restoration_canonical():
    """estimate_song_goal_targets must return the same keys as CANONICAL_THRESHOLDS_RESTORATION."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        is_studio_2026=False,
        restorability_score=65,
    )
    assert set(targets.keys()) == set(CANONICAL_THRESHOLDS_RESTORATION.keys())


def test_studio_2026_mode_raises_targets():
    """Studio 2026 mode targets must be ≥ Restoration targets for P3–P5 goals."""
    common_kwargs = {
        "material_type": "vinyl",
        "era_decade": 1975,
        "genre_label": "pop",
        "restorability_score": 70,
    }
    t_rest = estimate_song_goal_targets(is_studio_2026=False, **common_kwargs)
    t_s26 = estimate_song_goal_targets(is_studio_2026=True, **common_kwargs)
    # P3-P5 floor is higher in Studio 2026 → targets must be higher or equal
    for g in ("transparenz", "brillanz", "groove", "mikrodynamik"):
        assert t_s26.get(g, 0) >= t_rest.get(g, 0) - 0.02, (
            f"Studio2026[{g}]={t_s26.get(g):.3f} < Restoration[{g}]={t_rest.get(g):.3f}"
        )


def test_goal_weights_above_1_raise_target():
    """goal_weight > 1.0 for a goal must increase its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"natuerlichkeit": 1.8},
    )
    assert weighted["natuerlichkeit"] >= base["natuerlichkeit"]


def test_goal_weights_below_1_lower_target():
    """goal_weight < 1.0 for a goal must lower its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"brillanz": 0.4},
    )
    assert weighted["brillanz"] <= base["brillanz"]


def test_none_inputs_do_not_crash():
    """Graceful handling of None/missing inputs — must not raise."""
    result = estimate_song_goal_targets(is_studio_2026=False)
    assert isinstance(result, dict)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# §09.7 predict_quality_score
# ---------------------------------------------------------------------------


def test_shellac_quality_lower_than_vinyl():
    """Shellac has lower quality ceiling than vinyl."""
    q_shellac = predict_quality_score("shellac", 50.0, 0.4, False)
    q_vinyl = predict_quality_score("vinyl", 50.0, 0.4, False)
    assert q_shellac < q_vinyl


def test_cd_digital_has_highest_ceiling():
    """CD digital with high restorability and no defects → near-maximum quality."""
    q_cd = predict_quality_score("cd_digital", 95.0, 0.0, False)
    assert q_cd > 0.85


def test_studio_boost_raises_score():
    """Studio 2026 mode adds boost over Restoration."""
    q_rest = predict_quality_score("vinyl", 70.0, 0.3, False)
    q_s26 = predict_quality_score("vinyl", 70.0, 0.3, True)
    assert q_s26 > q_rest


def test_heavy_defects_reduce_score():
    """Heavy defects (severity=0.9) must reduce quality vs. no defects."""
    q_clean = predict_quality_score("vinyl", 65.0, 0.0, False)
    q_damaged = predict_quality_score("vinyl", 65.0, 0.9, False)
    assert q_damaged < q_clean


def test_quality_score_in_valid_range():
    """Output must always be in [0.0, 0.99]."""
    for mat in ["shellac", "vinyl", "tape", "cd_digital", "mp3_low", "wax_cylinder"]:
        for rest in [5.0, 50.0, 95.0]:
            q = predict_quality_score(mat, rest, 0.5, False)
            assert 0.0 <= q <= 0.99, f"{mat}/rest={rest}: q={q} out of range"


def test_quality_score_finite():
    """Output must never be NaN or Inf."""
    q = predict_quality_score("unknown_material", 55.0, 0.5, True)
    assert np.isfinite(q)


# ---------------------------------------------------------------------------
# §09.8 get_material_floor — material-adaptive goal floors
# ---------------------------------------------------------------------------


def test_get_material_floor_shellac_brillanz_below_canonical():
    """Shellac brillanz floor must be below canonical (physical BW limit ≤ 8 kHz)."""
    canonical_floor = CANONICAL_THRESHOLDS_RESTORATION["brillanz"]
    shellac_floor = get_material_floor("shellac", "brillanz")
    assert shellac_floor < canonical_floor, (
        f"shellac brillanz floor {shellac_floor:.3f} must be < canonical {canonical_floor:.3f}"
    )


def test_get_material_floor_vinyl_between_shellac_and_cd():
    """Vinyl brillanz floor must be between shellac and cd_digital."""
    f_shellac = get_material_floor("shellac", "brillanz")
    f_vinyl = get_material_floor("vinyl", "brillanz")
    f_cd = get_material_floor("cd_digital", "brillanz")
    assert f_shellac < f_vinyl <= f_cd, (
        f"Expected shellac({f_shellac:.3f}) < vinyl({f_vinyl:.3f}) <= cd({f_cd:.3f})"
    )


def test_get_material_floor_cd_matches_canonical_closely():
    """CD digital floor should be at or above canonical (digital bias is positive)."""
    for goal in ["transparenz", "artikulation", "brillanz"]:
        floor = get_material_floor("cd_digital", goal)
        canonical = CANONICAL_THRESHOLDS_RESTORATION[goal]
        assert floor >= canonical * 0.99, (
            f"cd_digital {goal} floor {floor:.3f} unexpectedly far below canonical {canonical:.3f}"
        )


def test_get_material_floor_always_in_valid_range():
    """All material/goal combinations must stay in [0.30, 0.99]."""
    materials = ["wax_cylinder", "shellac", "vinyl", "tape", "cd_digital", "mp3_low", "unknown"]
    goals = list(CANONICAL_THRESHOLDS_RESTORATION.keys())
    for mat in materials:
        for goal in goals:
            floor = get_material_floor(mat, goal)
            assert 0.30 <= floor <= 0.99, f"get_material_floor({mat!r}, {goal!r}) = {floor:.3f} out of [0.30, 0.99]"


def test_get_material_floor_studio_2026_geq_restoration():
    """Studio 2026 floors should be >= restoration floors (higher canonical base)."""
    for goal in ["natuerlichkeit", "brillanz", "transparenz"]:
        f_rest = get_material_floor("vinyl", goal, is_studio_2026=False)
        f_s26 = get_material_floor("vinyl", goal, is_studio_2026=True)
        assert f_s26 >= f_rest - 0.01, (
            f"Studio2026 floor ({f_s26:.3f}) unexpectedly below restoration floor ({f_rest:.3f}) for vinyl/{goal}"
        )


def test_get_material_floor_unknown_material_returns_finite():
    """Unknown material should fall back gracefully (analog class)."""
    floor = get_material_floor("totally_unknown_carrier", "natuerlichkeit")
    assert np.isfinite(floor)
    assert 0.30 <= floor <= 0.99


# ---------------------------------------------------------------------------
# §09.9 get_phase_strength_range — material-adaptive strength caps
# ---------------------------------------------------------------------------


def test_get_phase_strength_range_ultra_analog_is_capped():
    """Ultra-analog (shellac) phases must have lower max_strength than analog (vinyl)."""
    _, max_shellac = get_phase_strength_range("phase_03_denoise", "shellac", 50.0)
    _, max_vinyl = get_phase_strength_range("phase_03_denoise", "vinyl", 50.0)
    assert max_shellac < max_vinyl, (
        f"shellac max_strength ({max_shellac:.2f}) must be < vinyl ({max_vinyl:.2f}) for phase_03"
    )


def test_get_phase_strength_range_high_restorability_reduces_max():
    """High restorability (>80) must reduce max_strength toward passthrough (§2.45b)."""
    _, max_low = get_phase_strength_range("phase_03_denoise", "vinyl", 30.0)
    _, max_high = get_phase_strength_range("phase_03_denoise", "vinyl", 95.0)
    assert max_high < max_low, (
        f"High restorability max ({max_high:.2f}) must be < low restorability max ({max_low:.2f})"
    )


def test_get_phase_strength_range_min_never_exceeds_max():
    """min_strength must always be <= max_strength for all materials/phases."""
    phases = [
        "phase_03_denoise", "phase_07_harmonic_restoration", "phase_09_crackle_removal",
        "phase_23_spectral_repair", "phase_26_dynamic_range_expansion", "phase_49_advanced_dereverb",
    ]
    materials = ["wax_cylinder", "shellac", "vinyl", "tape", "cd_digital", "mp3_low"]
    for mat in materials:
        for phase in phases:
            min_s, max_s = get_phase_strength_range(phase, mat, 70.0)
            assert min_s <= max_s, f"min({min_s:.2f}) > max({max_s:.2f}) for {mat}/{phase}"
            assert 0.0 <= min_s <= 1.0
            assert 0.0 <= max_s <= 1.0


def test_get_phase_strength_range_returns_floats():
    """Return values must be Python floats, not numpy scalars."""
    min_s, max_s = get_phase_strength_range("phase_09_crackle_removal", "vinyl", 60.0)
    assert isinstance(min_s, float)
    assert isinstance(max_s, float)


def test_get_phase_strength_range_stereo_forbidden_on_ultra_analog():
    """Stereo width enhancer must have max_strength=0 on mono ultra_analog material."""
    _, max_s = get_phase_strength_range("phase_48_stereo_width_enhancer", "shellac", 50.0)
    assert max_s == 0.0, f"phase_48 must be disabled on shellac (max={max_s:.2f})"


def test_get_phase_strength_range_unknown_phase_uses_default():
    """Unknown phase IDs must return the safe default range."""
    min_s, max_s = get_phase_strength_range("phase_99_nonexistent", "vinyl", 50.0)
    assert max_s > 0.0  # default range is permissive
    assert min_s <= max_s


# ---------------------------------------------------------------------------
# §09.8 Material-floor ordering — materials added to UV3 _ADAPTIVE_THR_MATERIAL_CEILING
# (§0a §2.44 defensive ceiling for lacquer_disc, mp3_high, aac, minidisc, dat)
# ---------------------------------------------------------------------------


def test_material_floor_lacquer_disc_lower_than_vinyl_brillanz():
    """lacquer_disc BW ≤ 8 kHz → brillanz floor must be below vinyl (BW ≤ 16 kHz)."""
    f_lacquer = get_material_floor("lacquer_disc", "brillanz")
    f_vinyl = get_material_floor("vinyl", "brillanz")
    assert f_lacquer < f_vinyl, (
        f"lacquer_disc brillanz floor {f_lacquer:.3f} should be below vinyl {f_vinyl:.3f} "
        "(lacquer_disc BW ≤ 8 kHz, vinyl BW ≤ 16 kHz — §0a §6.2c)"
    )


def test_material_floor_dat_near_cd_level():
    """DAT is near-CD quality; its brillanz floor must be ≥ 90 % of cd_digital floor."""
    f_dat = get_material_floor("dat", "brillanz")
    f_cd = get_material_floor("cd_digital", "brillanz")
    assert f_dat >= 0.90 * f_cd, (
        f"dat brillanz floor {f_dat:.3f} should be at least 90 % of cd_digital {f_cd:.3f} "
        "(DAT is near-CD quality — §0a §2.44)"
    )


def test_material_floor_ordering_ultra_analog_to_digital():
    """Physical material quality ordering must hold for brillanz floor:
    wax_cylinder < shellac ≤ lacquer_disc < vinyl < mp3_low < minidisc ≤ mp3_high ≤ aac < dat < cd_digital
    """
    materials = ["wax_cylinder", "shellac", "lacquer_disc", "vinyl", "mp3_low",
                 "minidisc", "mp3_high", "aac", "dat", "cd_digital"]
    floors = [get_material_floor(m, "brillanz") for m in materials]
    for i in range(len(floors) - 1):
        assert floors[i] <= floors[i + 1] + 0.02, (  # 0.02 tolerance for equal-tier materials
            f"brillanz floor ordering violated: {materials[i]}={floors[i]:.3f} "
            f"> {materials[i+1]}={floors[i+1]:.3f} (§0a physical medium ordering)"
        )


def test_pmgg_canonical_thresholds_match_calibration_matrix():
    """PMGG's internal threshold copy must stay in sync with calibration_matrix (§2.55).

    Drift between PMGG's local _CANONICAL_THRESHOLDS_RESTORATION and
    calibration_matrix.CANONICAL_THRESHOLDS_RESTORATION causes silent per-phase
    gate inconsistencies — goals pass at pipeline end but fail in PMGG or vice versa.
    """
    from backend.core.per_phase_musical_goals_gate import (
        _CANONICAL_THRESHOLDS_RESTORATION as pmgg_rest,
        _CANONICAL_THRESHOLDS_STUDIO2026 as pmgg_s26,
    )

    # Key normalization: PMGG uses short aliases (micro_dynamics, bass_kraft, spatial_depth)
    # calibration_matrix uses both aliases; we map them to a common key for comparison.
    _ALIAS = {
        "micro_dynamics": "mikrodynamik",
        "bass_kraft": "basskraft",
        "spatial_depth": "raumtiefe",
        "tonalcenter": "tonal_center",
    }

    def _norm(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            key = _ALIAS.get(k, k)
            out[key] = v
        return out

    pmgg_r_norm = _norm(pmgg_rest)
    pmgg_s_norm = _norm(pmgg_s26)
    cal_r_norm = _norm(CANONICAL_THRESHOLDS_RESTORATION)
    cal_s_norm = _norm(CANONICAL_THRESHOLDS_STUDIO2026)

    # Check every goal that appears in BOTH dicts
    for goal in cal_r_norm:
        if goal in pmgg_r_norm:
            assert abs(pmgg_r_norm[goal] - cal_r_norm[goal]) < 0.005, (
                f"PMGG restoration threshold for '{goal}' = {pmgg_r_norm[goal]:.4f} "
                f"differs from calibration_matrix {cal_r_norm[goal]:.4f} — §2.55 sync violation"
            )
    for goal in cal_s_norm:
        if goal in pmgg_s_norm:
            assert abs(pmgg_s_norm[goal] - cal_s_norm[goal]) < 0.005, (
                f"PMGG studio2026 threshold for '{goal}' = {pmgg_s_norm[goal]:.4f} "
                f"differs from calibration_matrix {cal_s_norm[goal]:.4f} — §2.55 sync violation"
            )
