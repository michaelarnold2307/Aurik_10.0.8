"""Tests fuer Policy-Drift-Guard in backend.adaptive_pipeline."""

from __future__ import annotations

from backend.adaptive_pipeline import _enforce_canonical_policy_route
from policy.ml_policy_engine import (
    CANONICAL_INSTRUMENTAL_NR_ROUTE,
    CANONICAL_REPAIR_ROUTE,
    CANONICAL_SEPARATION_ROUTE,
    CANONICAL_VOCAL_NR_ROUTE,
)


def test_enforce_denoise_legacy_to_vocal_route() -> None:
    got = _enforce_canonical_policy_route("denoise", "resemble_enhance", {"has_vocals": True})
    assert got == CANONICAL_VOCAL_NR_ROUTE


def test_enforce_denoise_legacy_to_instrumental_route() -> None:
    got = _enforce_canonical_policy_route("denoise", "deepfilternet", {"has_vocals": False})
    assert got == CANONICAL_INSTRUMENTAL_NR_ROUTE


def test_keep_canonical_denoise_route_unchanged() -> None:
    got = _enforce_canonical_policy_route("denoise", CANONICAL_VOCAL_NR_ROUTE, {"has_vocals": True})
    assert got == CANONICAL_VOCAL_NR_ROUTE


def test_enforce_repair_route() -> None:
    got = _enforce_canonical_policy_route("repair", "dccrn", {"has_vocals": False})
    assert got == CANONICAL_REPAIR_ROUTE


def test_enforce_separation_route() -> None:
    got = _enforce_canonical_policy_route("separation", "mdx23c", {"has_vocals": True})
    assert got == CANONICAL_SEPARATION_ROUTE


def test_enforce_enhancement_legacy_to_instrumental_route() -> None:
    got = _enforce_canonical_policy_route("enhancement", "gacela", {"has_vocals": False})
    assert got == CANONICAL_INSTRUMENTAL_NR_ROUTE
