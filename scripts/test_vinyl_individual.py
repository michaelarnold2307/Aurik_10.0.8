#!/usr/bin/env python3
"""
Test jede Phase 2.3 Komponente einzeln auf vinyl_test_01.wav
"""

from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.bass_enhancement import BassEnhancementSystem
from dsp.brass_enhancement import BrassEnhancementSystem
from dsp.drums_enhancement import DrumsEnhancementSystem
from dsp.guitar_enhancement import GuitarEnhancementSystem
from dsp.piano_restoration import PianoRestorationSystem
from dsp.spatial_enhancement import SpatialEnhancementSystem

print("🎵 Testing individual components on vinyl_test_01.wav")
print("=" * 60)

# Load audio
audio_file = Path("input/vinyl_test_01.wav")
if not audio_file.exists():
    print(f"❌ File not found: {audio_file}")
    sys.exit(1)

print(f"Loading: {audio_file.name}")
audio, sr = sf.read(audio_file)
if audio.ndim == 1:
    audio = np.stack([audio, audio], axis=-1)
print(f"✓ Loaded: {audio.shape[0]} samples @ {sr} Hz")
print(f"  Duration: {audio.shape[0]/sr:.1f}s\n")

# Test each component individually with timeout
systems = [
    ("Bass", BassEnhancementSystem),
    ("Drums", DrumsEnhancementSystem),
    ("Guitar", GuitarEnhancementSystem),
    ("Piano", PianoRestorationSystem),
    ("Brass", BrassEnhancementSystem),
    ("Spatial", SpatialEnhancementSystem),
]

for name, SystemClass in systems:
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        start = time.time()
        system = SystemClass()
        processed, report = system.process(audio, sr)
        elapsed = time.time() - start

        rms_orig = np.sqrt(np.mean(audio**2))
        rms_proc = np.sqrt(np.mean(processed**2))
        gain_db = 20 * np.log10(rms_proc / (rms_orig + 1e-10))

        print(f"✓ {gain_db:+.2f} dB ({elapsed:.1f}s)")

    except Exception as e:
        print(f"✗ ERROR")
        print(f"  {str(e)[:100]}")
        import traceback

        traceback.print_exc()
        break

print("\n" + "=" * 60)
print("✅ Test complete")
