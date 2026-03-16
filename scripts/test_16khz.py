#!/usr/bin/env python3
"""
Test alle 6 Phase 2.3 Komponenten bei 16kHz Sample-Rate
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.bass_enhancement import BassEnhancementSystem
from dsp.brass_enhancement import BrassEnhancementSystem
from dsp.drums_enhancement import DrumsEnhancementSystem
from dsp.guitar_enhancement import GuitarEnhancementSystem
from dsp.piano_restoration import PianoRestorationSystem
from dsp.spatial_enhancement import SpatialEnhancementSystem

print("🎵 Phase 2.3 Test @ 16kHz (Nyquist=8000Hz)")
print("=" * 50 + "\n")

# Generate test signal at 16kHz
sr = 16000  # CRITICAL: Low sample rate
duration = 0.5
t = np.linspace(0, duration, int(sr * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
audio_stereo = np.stack([audio, audio], axis=-1)

print(f"Test signal: {duration}s @ {sr} Hz")
print(f"Nyquist frequency: {sr/2} Hz\n")

# Test each component
systems = [
    ("Bass", BassEnhancementSystem()),
    ("Drums", DrumsEnhancementSystem()),
    ("Guitar", GuitarEnhancementSystem()),
    ("Piano", PianoRestorationSystem()),
    ("Brass", BrassEnhancementSystem()),
    ("Spatial", SpatialEnhancementSystem()),
]

results = {}
for name, system in systems:
    try:
        proc, report = system.process(audio_stereo, sr)
        rms_change = 20 * np.log10(np.sqrt(np.mean(proc**2)) / (np.sqrt(np.mean(audio_stereo**2)) + 1e-10))
        results[name] = "✓"
        print(f"{name:8s}: ✓ ({rms_change:+.2f} dB)")
    except Exception as e:
        results[name] = f"✗ {str(e)[:60]}"
        print(f"{name:8s}: ✗")
        print(f"          Error: {str(e)[:60]}")

# Summary
print(f"\n{'='*50}")
success = sum(1 for v in results.values() if v == "✓")
total = len(results)
print(f"Result: {success}/{total} working at 16kHz")

if success == total:
    print("✅ All systems work at low sample rates!")
    sys.exit(0)
else:
    print(f"⚠️  {total - success} systems still have issues")
    sys.exit(1)
