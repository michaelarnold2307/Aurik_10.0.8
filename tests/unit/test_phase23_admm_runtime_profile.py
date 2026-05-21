from __future__ import annotations

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_23_spectral_repair import SpectralRepair


def test_admm_runtime_profile_keys_and_bounds() -> None:
    p = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.VINYL,
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {"clip_percentile", "clip_floor", "side_clip_multiplier"}
    assert 98.8 <= p["clip_percentile"] <= 99.9
    assert 0.80 <= p["clip_floor"] <= 0.93
    assert 1.00 <= p["side_clip_multiplier"] <= 1.10


def test_low_restorability_reduces_clip_percentile() -> None:
    low = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.TAPE,
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["clip_percentile"] < high["clip_percentile"]


def test_fast_mode_more_conservative_than_quality() -> None:
    fast = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.CD_DIGITAL,
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.CD_DIGITAL,
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["clip_percentile"] > quality["clip_percentile"]
    assert fast["clip_floor"] > quality["clip_floor"]
    assert fast["side_clip_multiplier"] > quality["side_clip_multiplier"]


def test_lossy_material_raises_clip_floor_vs_analog() -> None:
    lossy = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.STREAMING,
        quality_mode="balanced",
        restorability_score=60.0,
    )
    analog = SpectralRepair._compute_admm_runtime_profile(
        material=MaterialType.VINYL,
        quality_mode="balanced",
        restorability_score=60.0,
    )

    assert lossy["clip_floor"] > analog["clip_floor"]


def test_cassette_bw_ceiling_is_material_adaptive() -> None:
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.CASSETTE) == 12000.0
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.TAPE) == 15000.0
    assert SpectralRepair._material_bw_ceiling_hz(MaterialType.REEL_TAPE) == 18000.0
