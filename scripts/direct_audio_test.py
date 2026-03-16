#!/usr/bin/env python3
"""
Direkter Test mit echtem Audio - ohne UnifiedRestorerV2 Pipeline
"""

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.bass_enhancement import BassEnhancementSystem
from dsp.spatial_enhancement import SpatialEnhancementSystem

# Find audio files
audio_dirs = [Path("input"), Path("test_audio"), Path("audio_examples")]

audio_files = []
for d in audio_dirs:
    if d.exists():
        audio_files.extend(d.glob("*.wav"))
        audio_files.extend(d.glob("*.mp3"))

if not audio_files:
    print("❌ No audio files found")
    sys.exit(1)

# Test on first file
test_file = audio_files[0]
print(f"🎵 Testing Phase 2.3 on real audio")
print(f"File: {test_file.name}\n")

# Load audio
print("[1/4] Loading...", end=" ", flush=True)
audio, sr = sf.read(test_file)
if audio.ndim == 1:
    audio = np.stack([audio, audio], axis=-1)
print(f"✓ ({audio.shape[0]} samples @ {sr} Hz)")

# RMS before
rms_before = np.sqrt(np.mean(audio**2))

# Apply Bass Enhancement
print("[2/4] Bass Enhancement...", end=" ", flush=True)
bass = BassEnhancementSystem()
audio_bass, report_bass = bass.process(audio, sr)
rms_bass = np.sqrt(np.mean(audio_bass**2))
gain_bass = 20 * np.log10(rms_bass / (rms_before + 1e-10))
print(f"✓ ({gain_bass:+.2f} dB)")

# Apply Spatial Enhancement
print("[3/4] Spatial Enhancement...", end=" ", flush=True)
spatial = SpatialEnhancementSystem()
audio_spatial, report_spatial = spatial.process(audio_bass, sr)
rms_spatial = np.sqrt(np.mean(audio_spatial**2))
gain_spatial = 20 * np.log10(rms_spatial / (rms_before + 1e-10))
print(f"✓ ({gain_spatial:+.2f} dB)")

# Save result
print("[4/4] Saving...", end=" ", flush=True)
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"phase_2_3_test_{test_file.stem}.wav"
sf.write(output_file, audio_spatial, sr)
print(f"✓")

print(f"\n{'='*50}")
print(f"✅ Success!")
print(f"Output: {output_file}")
print(f"Total gain: {gain_spatial:+.2f} dB")
