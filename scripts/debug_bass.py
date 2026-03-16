#!/usr/bin/env python3
"""
Debug Bass Enhancement - welche Stage hängt?
"""

from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔍 Bass Enhancement Debug")
print("=" * 60)

# Load audio
audio_file = Path("input/vinyl_test_01.wav")
print(f"Loading: {audio_file.name}")
audio, sr = sf.read(audio_file)
if audio.ndim == 1:
    audio = np.stack([audio, audio], axis=-1)
print(f"✓ {audio.shape[0]} samples @ {sr} Hz ({audio.shape[0]/sr:.1f}s)\n")

# Test each stage individually
print("Testing Bass Enhancement Stages:")
print("-" * 60)

# Stage 1: SubBassEnhancer
try:
    print("1. SubBassEnhancer...", end=" ", flush=True)
    from dsp.bass_enhancement import SubBassEnhancer

    start = time.time()
    stage1 = SubBassEnhancer(gain_db=3.0)
    result1, report1 = stage1.process(audio, sr)
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
except Exception as e:
    print(f"✗ ERROR: {str(e)[:80]}")
    sys.exit(1)

# Stage 2: MidBassClarifier
try:
    print("2. MidBassClarifier...", end=" ", flush=True)
    from dsp.bass_enhancement import MidBassClarifier

    start = time.time()
    stage2 = MidBassClarifier(clarity=0.8)
    result2, report2 = stage2.process(result1, sr)
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
except Exception as e:
    print(f"✗ ERROR: {str(e)[:80]}")
    sys.exit(1)

# Stage 3: BassHarmonicsEnhancer
try:
    print("3. BassHarmonicsEnhancer...", end=" ", flush=True)
    from dsp.bass_enhancement import BassHarmonicsEnhancer

    start = time.time()
    stage3 = BassHarmonicsEnhancer(gain_db=2.0)
    result3, report3 = stage3.process(result2, sr)
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
except Exception as e:
    print(f"✗ ERROR: {str(e)[:80]}")
    sys.exit(1)

# Stage 4: BassDynamicsController
try:
    print("4. BassDynamicsController...", end=" ", flush=True)
    from dsp.bass_enhancement import BassDynamicsController

    start = time.time()
    stage4 = BassDynamicsController(compression_ratio=2.0)
    result4, report4 = stage4.process(result3, sr)
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
except Exception as e:
    print(f"✗ ERROR: {str(e)[:80]}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All stages completed successfully!")
