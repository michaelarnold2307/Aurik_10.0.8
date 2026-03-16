#!/usr/bin/env python3
"""
Policy-Engine Decision Logic Test
==================================

Tests whether Policy-Engine makes correct model selections
based on different audio contexts (without Docker execution).

This validates the BRAIN of the system - the decision logic.
"""

import logging
import sys

from policy.ml_policy_engine import MLModelPolicyEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_denoise_selections():
    """Test denoise model selection logic (🔬 semantic-aware)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Denoise Model Selection (Semantic-Aware)")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    # 🎯 SEMANTIC-AWARE TEST CASES (replaces genre-based logic)
    test_cases = [
        {
            "name": "Classical Piano (Sustained)",
            "context": {
                "content_character": "HIGHLY_SUSTAINED",
                "dominant_instrument": "KEYS",
                "has_vocals": False,
                "has_ambient": False,
                "noise_type": "broadband",
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": "resemble_enhance",  # Balanced for keys/piano
        },
        {
            "name": "Speech/Podcast (Vocals)",
            "context": {
                "content_character": "BALANCED",
                "dominant_instrument": "SPEECH",
                "has_vocals": True,
                "noise_type": "broadband",
                "sample_rate": 16000,
            },
            "goal": {"target_quality": 0.85},
            "expected": "resemble_enhance",  # Voice-optimized
        },
        {
            "name": "Vinyl Jazz",
            "context": {
                "content_character": "BALANCED",
                "detected_medium": "vinyl",
                "has_vocals": False,
                "sample_rate": 44100,
            },
            "goal": {"target_quality": 0.85},
            "expected": "banquet",  # Vinyl-specialized
        },
        {
            "name": "Pop Music (Vocals)",
            "context": {
                "content_character": "BALANCED",
                "dominant_instrument": "VOCALS",
                "has_vocals": True,
                "noise_type": "unknown",
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": "resemble_enhance",  # Voice-optimized
        },
        {
            "name": "Drum Kit (Transient-Rich)",
            "context": {
                "content_character": "HIGHLY_TRANSIENT",
                "dominant_instrument": "DRUMS",
                "has_drums": True,
                "has_vocals": False,
                "preserve_transients": True,
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": "dccrn",  # Preserves transients
        },
        {
            "name": "Ambient/Drone (Sustained)",
            "context": {
                "content_character": "HIGHLY_SUSTAINED",
                "dominant_instrument": "AMBIENT",
                "has_ambient": True,
                "has_vocals": False,
                "sample_rate": 48000,
            },
            "goal": {"target_quality": 0.85},
            "expected": "deepfilternet",  # Aggressive smoothing
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_denoise_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    return results


def test_repair_selections():
    """Test repair model selection logic."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Repair Model Selection")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    # Note: repair_model selection uses has_vocals, no semantic changes needed
    test_cases = [
        {
            "name": "Speech with Clipping",
            "context": {"has_vocals": True},
            "goal": {"target_quality": 0.85},
            "expected": "fullsubnet",
        },
        {
            "name": "Music with Clipping",
            "context": {"has_vocals": False},
            "goal": {"target_quality": 0.85},
            "expected": "dccrn",
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_repair_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    return results


def test_separation_selections():
    """Test separation model selection logic."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Separation Model Selection")
    logger.info("=" * 60)

    engine = MLModelPolicyEngine()

    test_cases = [
        {
            "name": "High Quality (6-stem)",
            "context": {"sample_rate": 44100},
            "goal": {"target_quality": 0.9, "stems": 6},  # FIXED: added stems parameter
            "expected": "demucs",
        },
        {
            "name": "Fast Processing",
            "context": {"sample_rate": 44100},
            "goal": {"quality_level": "maximal", "priority": "speed"},  # FIXED: added quality_level
            "expected": "mdx23c",
        },
    ]

    results = {}
    for test in test_cases:
        selected = engine.select_separation_model(test["context"], test["goal"])
        expected = test["expected"]
        match = selected == expected

        status = "✅" if match else "❌"
        logger.info(f"{status} {test['name']}: {selected} (expected: {expected})")

        results[test["name"]] = match

    return results


def main():
    """Run all policy decision tests."""
    logger.info("=" * 60)
    logger.info("POLICY-ENGINE DECISION LOGIC TEST SUITE")
    logger.info("=" * 60)

    all_results = {}

    # Test denoise selections
    all_results.update(test_denoise_selections())

    # Test repair selections
    all_results.update(test_repair_selections())

    # Test separation selections
    all_results.update(test_separation_selections())

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    total_pass = sum(all_results.values())
    total_tests = len(all_results)

    for test_name, passed in all_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL: {total_pass}/{total_tests} tests passed ({total_pass/total_tests*100:.0f}%)")
    logger.info("=" * 60)

    if total_pass == total_tests:
        logger.info("\n🎉 ALL POLICY TESTS PASSED!")
        logger.info("Policy-Engine is making correct decisions!")
        return 0
    else:
        logger.warning("\n⚠️  Some policy tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
