#!/usr/bin/env python3
"""
Minimaler Phase 2.3 Test - Nur die 6 Komponenten direkt
"""

from pathlib import Path
import sys

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🎵 Phase 2.3 Quick Test\n")

# Generate simple test signal
sr = 48000
duration = 0.5  # Short signal
t = np.linspace(0, duration, int(sr * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # Simple 440Hz tone
audio_stereo = np.stack([audio, audio], axis=-1)

print(f"Test signal: {duration}s @ {sr} Hz, {audio_stereo.shape}\n")

# Test each component
results = {}

# 1. Bass Enhancement
try:
    from dsp.bass_enhancement import BassEnhancementSystem

    bass = BassEnhancementSystem()
    proc, report = bass.process(audio_stereo, sr)
    results["Bass"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"1. Bass Enhancement:   ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Bass"] = f"✗ {str(e)[:50]}"
    print(f"1. Bass Enhancement:   ✗ {str(e)[:50]}")

# 2. Drums Enhancement
try:
    from dsp.drums_enhancement import DrumsEnhancementSystem

    drums = DrumsEnhancementSystem()
    proc, report = drums.process(audio_stereo, sr)
    results["Drums"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"2. Drums Enhancement:  ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Drums"] = f"✗ {str(e)[:50]}"
    print(f"2. Drums Enhancement:  ✗ {str(e)[:50]}")

# 3. Guitar Enhancement
try:
    from dsp.guitar_enhancement import GuitarEnhancementSystem

    guitar = GuitarEnhancementSystem()
    proc, report = guitar.process(audio_stereo, sr)
    results["Guitar"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"3. Guitar Enhancement: ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Guitar"] = f"✗ {str(e)[:50]}"
    print(f"3. Guitar Enhancement: ✗ {str(e)[:50]}")

# 4. Piano Restoration
try:
    from dsp.piano_restoration import PianoRestorationSystem

    piano = PianoRestorationSystem()
    proc, report = piano.process(audio_stereo, sr)
    results["Piano"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"4. Piano Restoration:  ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Piano"] = f"✗ {str(e)[:50]}"
    print(f"4. Piano Restoration:  ✗ {str(e)[:50]}")

# 5. Brass Enhancement
try:
    from dsp.brass_enhancement import BrassEnhancementSystem

    brass = BrassEnhancementSystem()
    proc, report = brass.process(audio_stereo, sr)
    results["Brass"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"5. Brass Enhancement:  ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Brass"] = f"✗ {str(e)[:50]}"
    print(f"5. Brass Enhancement:  ✗ {str(e)[:50]}")

# 6. Spatial Enhancement
try:
    from dsp.spatial_enhancement import SpatialEnhancementSystem

    spatial = SpatialEnhancementSystem()
    proc, report = spatial.process(audio_stereo, sr)
    results["Spatial"] = "✓"
    rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
    print(f"6. Spatial Enhancement: ✓ ({rms_change:+.2f} dB)")
except Exception as e:
    results["Spatial"] = f"✗ {str(e)[:50]}"
    print(f"6. Spatial Enhancement: ✗ {str(e)[:50]}")

# Summary
print(f"\n{'='*50}")
success = sum(1 for v in results.values() if v == "✓")
total = len(results)
print(f"Result: {success}/{total} components working")

if success == total:
    print("✅ ALL SYSTEMS OPERATIONAL!")
    sys.exit(0)
else:
    print(f"⚠️  {total - success} components failed")
    sys.exit(1)
