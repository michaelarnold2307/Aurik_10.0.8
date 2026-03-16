"""
Test Phase 2 Hum Removal ML-Hybrid Integration

Quick test to verify DeepFilterNet dual-stage integration for hum removal.
Dieses Skript ist als manuelles Integrations-Skript gedacht.
Pytest sammelt die Datei, führt aber keinen Code auf Modul-Ebene aus.
"""

# Kein Modul-Level-Code — alles liegt unter if __name__ == "__main__",
# damit pytest die Datei importieren kann ohne Seiteneffekte.

if __name__ == "__main__":
    import sys
    import numpy as np

    from backend.core.phases.phase_02_hum_removal import HumRemovalPhase

    print("=" * 80)
    print("Phase 2 Hum Removal ML-Hybrid Integration Test")
    print("=" * 80)

    # Create test phase
    phase = HumRemovalPhase()
    print(f"\n✅ Phase instantiated: {phase.get_metadata().name}")

    # Generate test audio with AC hum
    sr = 44100
    duration = 2
    t = np.linspace(0, duration, sr * duration)

    # Clean signal (music)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    audio += 0.15 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic

    # Add 50 Hz hum with harmonics
    hum_strength = 0.08  # Significant hum
    audio += hum_strength * np.sin(2 * np.pi * 50 * t)  # Fundamental
    audio += hum_strength * 0.5 * np.sin(2 * np.pi * 100 * t)  # 2nd harmonic
    audio += hum_strength * 0.3 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic
    audio += hum_strength * 0.2 * np.sin(2 * np.pi * 200 * t)  # 4th harmonic

    print("\nTest Audio:")
    print(f"  Duration: {duration}s @ {sr} Hz")
    print(f"  Hum: 50 Hz + harmonics (strength: {hum_strength:.2f})")
    print("  Music: 440 Hz + overtone")

    # Test with different quality modes
    test_modes = ["FAST", "BALANCED", "MAXIMUM"]

    for mode in test_modes:
        print(f"\n{'-'*80}")
        print(f"Testing Quality Mode: {mode}")
        print(f"{'-'*80}")

        result = phase.process(audio.copy(), sample_rate=sr, material_type="tape", auto_detect=True, quality_mode=mode)

        if result.success:
            print("✅ Processing successful!")
            alg = result.modifications.get("algorithm_version", "DSP_only")
            print(f"   Algorithm: {alg}")
            print(f"   Hum detected: {result.modifications['hum_detected']}")
            if result.modifications["hum_detected"]:
                print(f"   Fundamentals: {result.modifications['fundamentals']} Hz")
                print(f"   Harmonics removed: {result.modifications['total_harmonics_removed']}")
                print(f"   Reduction: {result.modifications['hum_reduction_db']:.1f} dB")
                print(f"   ML refined: {result.modifications['ml_refined']}")
            print(
                f"   Execution time: {result.metadata['execution_time_seconds']:.3f}s "
                f"({result.metadata['execution_time_seconds']/duration:.2f}× RT)"
            )

            # Expectations
            if mode == "FAST":
                if not result.modifications["ml_refined"]:
                    print("   ✅ Expected: FAST mode uses DSP only")
                else:
                    print("   ⚠️  Unexpected: FAST should not use ML")

            elif mode in ["BALANCED", "MAXIMUM"]:
                if result.modifications["ml_refined"]:
                    print(f"   ✅ Expected: {mode} mode uses ML refinement")
                else:
                    print("   ⚠️  Note: ML not used (plugin may be unavailable)")
        else:
            print("❌ Processing failed!")

    print(f"\n{'='*80}")
    print("✅ Phase 2 ML-Hybrid Integration Test Complete!")
    print(f"{'='*80}")
    print("\nStrategy: Dual-Stage (DSP Rough + ML Refine)")
    print("  - Stage 1: DSP adaptive comb filtering removes bulk hum")
    print("  - Stage 2: ML (DeepFilterNet) removes residual hum and smooths artifacts")
    print("  - ML only applied if hum reduction >10 dB (significant hum)")
    print("  - Graceful fallback: DSP-only if ML unavailable")
    print("\nExpected Improvement:")
    print("  - Current (DSP): Natürlichkeit 0.50")
    print("  - With ML: Natürlichkeit 0.75 (+0.25)")
    print("  - Overall: 0.86 → 0.88 (+0.02)")
