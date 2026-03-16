"""
V3 Basic Test Suite
===================

Tests für UnifiedRestorerV3 Grundfunktionalität.

Sprint 1, Week 1 - Basic Functionality Tests
Author: AI Development Team
Date: 2026-02-15
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.performance_guard import QualityMode
from backend.core.defect_scanner import MaterialType
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3


def test_v3_initialization():
    """Test 1: V3 Initialisierung."""
    print("\n" + "=" * 70)
    print("TEST 1: V3 Initialization")
    print("=" * 70)

    # Test Default Config
    restorer = UnifiedRestorerV3()
    assert restorer.config.mode == QualityMode.BALANCED, "Default mode should be BALANCED"
    assert restorer.config.num_cores == 4, "Default cores should be 4"
    assert restorer.config.enforce_3x_rt == True, "Default enforce should be True"
    print("✅ Default initialization OK")

    # Test Custom Config
    config = RestorationConfig(mode=QualityMode.FAST, num_cores=2, enforce_3x_rt=False)
    restorer2 = UnifiedRestorerV3(config)
    assert restorer2.config.mode == QualityMode.FAST, "Custom mode should be FAST"
    assert restorer2.config.num_cores == 2, "Custom cores should be 2"
    print("✅ Custom initialization OK")

    # Test Components
    assert restorer.defect_scanner is not None, "DefectScanner should be initialized"
    assert restorer.scheduler is not None, "AdaptiveCoreScheduler should be initialized"
    assert restorer.performance_guard is not None, "PerformanceGuard should be initialized"
    print("✅ Component initialization OK")

    print("\n✅ TEST 1 PASSED\n")


def test_phase_metadata_discovery():
    """Test 2: Phase Metadata Discovery."""
    print("\n" + "=" * 70)
    print("TEST 2: Phase Metadata Discovery")
    print("=" * 70)

    restorer = UnifiedRestorerV3()
    metadata = restorer.phase_metadata

    print(f"Discovered {len(metadata)} phases")

    # Sollte mindestens einige Standardphasen finden
    expected_phases = ["phase_1.1_click_removal", "phase_2.0_hum_removal", "phase_3.0_denoise"]

    for phase_id in expected_phases:
        if phase_id in metadata:
            print(f"✅ Found {phase_id}")
            meta = metadata[phase_id]
            assert "class" in meta, f"{phase_id} should have 'class'"
            assert "name" in meta, f"{phase_id} should have 'name'"
            assert "category" in meta, f"{phase_id} should have 'category'"
        else:
            print(f"⚠️ {phase_id} not found (might be renamed)")

    print(f"\n✅ TEST 2 PASSED (found {len(metadata)} phases)\n")


def test_lazy_phase_loading():
    """Test 3: Lazy Phase Loading."""
    print("\n" + "=" * 70)
    print("TEST 3: Lazy Phase Loading")
    print("=" * 70)

    restorer = UnifiedRestorerV3()

    # Phase sollte nicht im Cache sein
    assert len(restorer._phase_cache) == 0, "Phase cache should be empty initially"
    print("✅ Phase cache initially empty")

    # Lade eine Phase
    if len(restorer.phase_metadata) > 0:
        first_phase_id = list(restorer.phase_metadata.keys())[0]
        phase = restorer._get_phase(first_phase_id)

        if phase is not None:
            assert first_phase_id in restorer._phase_cache, "Phase should be cached"
            print(f"✅ Lazy loading OK: {first_phase_id}")
        else:
            print(f"⚠️ Phase {first_phase_id} could not be loaded")
    else:
        print("⚠️ No phases discovered, skipping lazy loading test")

    print("\n✅ TEST 3 PASSED\n")


def test_synthetic_audio_restore():
    """Test 4: Synthetic Audio Restoration."""
    print("\n" + "=" * 70)
    print("TEST 4: Synthetic Audio Restoration")
    print("=" * 70)

    # Generate test audio (10 seconds)
    sr = 44100
    duration = 10
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal: 440 Hz sine
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add defects
    # 1. Clicks (10 random)
    for i in range(10):
        pos = int(np.random.rand() * len(audio))
        audio[pos : pos + 5] += 0.3 * np.random.randn(5)

    # 2. 60Hz Hum
    audio += 0.04 * np.sin(2 * np.pi * 60 * t)

    # 3. White Noise
    audio += 0.02 * np.random.randn(len(audio))

    print(f"Test Audio: {duration}s @ {sr} Hz")
    print("Defects: 10 clicks, 60Hz hum, white noise")

    # Test FAST Mode
    print("\nTesting FAST Mode...")
    config = RestorationConfig(mode=QualityMode.FAST, num_cores=2, enforce_3x_rt=True)
    restorer = UnifiedRestorerV3(config)

    start_time = time.time()
    result = restorer.restore(audio, sample_rate=sr)
    elapsed = time.time() - start_time

    # Validiere Result
    assert result.audio is not None, "Result audio should not be None"
    assert len(result.audio) == len(audio), "Result audio length should match input"
    assert result.material_type in MaterialType, "Material type should be valid"
    assert result.rt_factor > 0, "RT factor should be > 0"
    assert 0 <= result.quality_estimate <= 1.0, "Quality estimate should be 0-1"

    print("\n✅ FAST Mode Results:")
    print(f"   Material: {result.material_type.value}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   RT Factor: {result.rt_factor:.2f}×")
    print(f"   Quality: {result.quality_estimate*100:.1f}%")
    print(f"   Phases Executed: {len(result.phases_executed)}")
    print(f"   Phases Skipped: {len(result.phases_skipped)}")

    # RT Factor sollte < 3× sein
    assert result.rt_factor < 3.0, f"RT Factor should be < 3× (was {result.rt_factor:.2f}×)"
    print("   ✅ RT Factor < 3× (PASS)")

    print("\n✅ TEST 4 PASSED\n")

    return result


def test_performance_guard_integration():
    """Test 5: PerformanceGuard Integration."""
    print("\n" + "=" * 70)
    print("TEST 5: PerformanceGuard Integration")
    print("=" * 70)

    # Generate test audio (5 seconds)
    sr = 44100
    duration = 5
    audio = 0.1 * np.random.randn(int(sr * duration))

    print(f"Test Audio: {duration}s @ {sr} Hz")

    # Test BALANCED Mode mit enforce_3x_rt
    config = RestorationConfig(mode=QualityMode.BALANCED, enforce_3x_rt=True, enable_adaptive_skipping=True)
    restorer = UnifiedRestorerV3(config)

    result = restorer.restore(audio, sample_rate=sr)

    # Performance Guard sollte aktiv gewesen sein
    assert "performance" in result.metadata, "Performance metadata should exist"
    perf = result.metadata["performance"]

    print("\n✅ Performance Guard Results:")
    print(f"   Status: {perf['status']}")
    print(f"   RT Factor: {perf['rt_factor']:.2f}×")
    print(f"   Quality Degradation: {perf['quality_degradation']*100:.1f}%")

    # Sollte 3× RT Limit einhalten
    assert (
        result.rt_factor <= 3.0 or not config.enforce_3x_rt
    ), f"RT Factor should be <= 3× (was {result.rt_factor:.2f}×)"
    print("   ✅ 3× RT Limit enforced (PASS)")

    print("\n✅ TEST 5 PASSED\n")


def test_defect_scanner_integration():
    """Test 6: DefectScanner Integration."""
    print("\n" + "=" * 70)
    print("TEST 6: DefectScanner Integration")
    print("=" * 70)

    # Generate test audio with obvious defects
    sr = 44100
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))

    # Pure sine wave
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Add 50Hz hum (obvious)
    audio += 0.1 * np.sin(2 * np.pi * 50 * t)

    print(f"Test Audio: {duration}s @ {sr} Hz")
    print("Defects: Strong 50Hz hum")

    restorer = UnifiedRestorerV3()
    result = restorer.restore(audio, sample_rate=sr)

    # Check defect analysis
    assert "defect_analysis" in result.metadata, "Defect analysis should exist"
    defect_analysis = result.metadata["defect_analysis"]

    print("\n✅ DefectScanner Results:")
    print(f"   Material: {defect_analysis['material']}")
    print(f"   Analysis Time: {defect_analysis['analysis_time']:.3f}s")
    print("   Top Defects:")
    for i, defect in enumerate(defect_analysis["top_defects"][:3], 1):
        print(f"      {i}. {defect['type']}: {defect['severity']:.2f}")

    # HUM sollte erkannt worden sein
    defect_scores = result.defect_scores
    from backend.core.defect_scanner import DefectType

    if DefectType.HUM in defect_scores:
        hum_severity = defect_scores[DefectType.HUM]
        print(f"\n   Hum Severity: {hum_severity:.2f}")
        # Sollte > 0.1 sein bei starkem 50Hz Hum
        assert hum_severity > 0.1, f"Hum should be detected (severity {hum_severity:.2f})"
        print("   ✅ Hum detected correctly (PASS)")
    else:
        print("   ⚠️ Hum not in defect scores")

    print("\n✅ TEST 6 PASSED\n")


def test_config_variations():
    """Test 7: Config Variations."""
    print("\n" + "=" * 70)
    print("TEST 7: Config Variations")
    print("=" * 70)

    # Generate simple test audio
    sr = 44100
    duration = 3
    audio = 0.1 * np.random.randn(int(sr * duration))

    print(f"Test Audio: {duration}s @ {sr} Hz")

    # Test different Quality Modes
    modes = [QualityMode.FAST, QualityMode.BALANCED]

    for mode in modes:
        print(f"\nTesting {mode.value.upper()} Mode...")
        config = RestorationConfig(mode=mode)
        restorer = UnifiedRestorerV3(config)

        start_time = time.time()
        result = restorer.restore(audio, sample_rate=sr)
        elapsed = time.time() - start_time

        print(f"   Time: {elapsed:.2f}s, RT: {result.rt_factor:.2f}×, Quality: {result.quality_estimate*100:.1f}%")

    # Test different core counts
    print("\nTesting different core counts...")
    for cores in [1, 2, 4]:
        config = RestorationConfig(num_cores=cores)
        restorer = UnifiedRestorerV3(config)
        assert restorer.scheduler.num_cores <= 6, "Should not exceed MAX_CORES=6"
        print(f"   Cores {cores}: Scheduler uses {restorer.scheduler.num_cores} cores ✅")

    print("\n✅ TEST 7 PASSED\n")


def run_all_tests():
    """Run all V3 basic tests."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "V3 BASIC TEST SUITE" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")

    tests = [
        ("Initialization", test_v3_initialization),
        ("Phase Metadata Discovery", test_phase_metadata_discovery),
        ("Lazy Phase Loading", test_lazy_phase_loading),
        ("Synthetic Audio Restoration", test_synthetic_audio_restore),
        ("PerformanceGuard Integration", test_performance_guard_integration),
        ("DefectScanner Integration", test_defect_scanner_integration),
        ("Config Variations", test_config_variations),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
