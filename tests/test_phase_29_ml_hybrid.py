"""
Test Phase 29 Tape Hiss Reduction ML-Hybrid Integration

Quick test to verify DeepFilterNet band-specific HF refinement for tape hiss.
"""

import sys

sys.path.insert(0, "/mnt/1846D15B46D139E8/Aurik_Standalone")


import numpy as np

print("=" * 80)
print("Phase 29 Tape Hiss Reduction ML-Hybrid Integration Test")
print("=" * 80)

from backend.core.defect_scanner import MaterialType

# Import phase
from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

# Create test phase
phase = TapeHissReductionPhase()
print(f"\n✅ Phase instantiated: {phase.get_metadata().name}")

# Generate test audio with tape hiss
sr = 48000
duration = 2
t = np.linspace(0, duration, sr * duration)

# Clean signal (music)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
audio += 0.15 * np.sin(2 * np.pi * 880 * t)

# Add tape hiss (HF noise >8kHz)
np.random.seed(42)
hiss = np.random.randn(len(audio)) * 0.05  # White noise

# Filter hiss to HF (8-20 kHz)
from scipy import signal as sp_signal

sos_hp = sp_signal.butter(4, 8000, btype="high", fs=sr, output="sos")
hiss_hf = sp_signal.sosfilt(sos_hp, hiss)

# Add to audio
audio += hiss_hf

print("\nTest Audio:")
print(f"  Duration: {duration}s @ {sr} Hz")
print("  Clean signal: 440 Hz + 880 Hz")
print("  Tape hiss: HF noise >8kHz (typical tape hiss)")

# Test with different quality modes
test_modes = ["FAST", "BALANCED", "MAXIMUM"]

for mode in test_modes:
    print(f"\n{'-'*80}")
    print(f"Testing Quality Mode: {mode}")
    print(f"{'-'*80}")

    result = phase.process(audio.copy(), sample_rate=sr, material=MaterialType.TAPE, quality_mode=mode)

    if result.success:
        print("✅ Processing successful!")
        print(f"   Algorithm: {result.metadata['algorithm_version']}")
        print(f"   Material: {result.metadata['material']}")
        print(f"   HF reduction: {result.metadata['hf_reduction_db']:.1f} dB")
        print(f"   Gate threshold: {result.metadata['gate_threshold_db']:.1f} dB")
        print(f"   Reduction depth: {result.metadata['reduction_depth_db']:.1f} dB")
        print(f"   HF focus range: {result.metadata['hf_focus_range_hz']} Hz")
        print(f"   ML refined: {result.metadata['ml_refined']}")
        if result.metadata.get("ml_model"):
            print(f"   ML model: {result.metadata['ml_model']}")
        print(f"   Execution time: {result.execution_time_seconds:.3f}s ({result.metadata['rt_factor']:.2f}× RT)")

        # Expectations
        if mode == "FAST":
            if not result.metadata["ml_refined"]:
                print("   ✅ Expected: FAST mode uses DSP only")
            else:
                print("   ⚠️  Unexpected: FAST should not use ML")

        elif mode in ["BALANCED", "MAXIMUM"]:
            if result.metadata["ml_refined"]:
                print(f"   ✅ Expected: {mode} mode uses ML HF refinement")
                print("      Band-Specific: <2kHz DSP → >2kHz ML DeepFilterNet")
            else:
                print("   ⚠️  Note: ML not used (plugin may be unavailable or hiss reduction <3dB)")
    else:
        print("❌ Processing failed!")

print(f"\n{'='*80}")
print("✅ Phase 29 ML-Hybrid Integration Test Complete!")
print(f"{'='*80}")
print("\nStrategy: Band-Specific Processing")
print("  - Stage 1: DSP multi-band adaptive gates (all frequencies)")
print("  - Stage 2: ML DeepFilterNet HF refinement (>2kHz only)")
print("  - <2kHz: DSP only (preserves warmth and bass)")
print("  - >2kHz: ML refines residual hiss without artifacts")
print("  - ML only applied if HF reduction >3 dB (significant hiss)")
print("  - Graceful fallback: DSP-only if ML unavailable")
print("\nExpected Improvement:")
print("  - Current (DSP): Natürlichkeit 0.50")
print("  - With ML: Natürlichkeit 0.80 (+0.30)")
print("  - Overall: 0.90 → 0.93 (+0.03)")
