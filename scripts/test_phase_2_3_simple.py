#!/usr/bin/env python3
"""
Simple Phase 2.3 Testing Script
Tests nur die Phase 2.3 Enhancement-Komponenten direkt.
"""

from pathlib import Path
import sys

import librosa
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dsp.bass_enhancement import BassEnhancementSystem
from dsp.brass_enhancement import BrassEnhancementSystem
from dsp.drums_enhancement import DrumsEnhancementSystem
from dsp.guitar_enhancement import GuitarEnhancementSystem
from dsp.piano_restoration import PianoRestorationSystem
from dsp.spatial_enhancement import SpatialEnhancementSystem


def test_component(component_name, system, audio, sr):
    """Test a single Phase 2.3 component."""
    print(f"\n{'='*60}")
    print(f"Testing: {component_name}")
    print(f"{'='*60}")

    try:
        processed, report = system.process(audio, sr)
        print(f"  ✓ Processing successful")
        print(f"  📊 Report keys: {list(report.keys())}")

        # Calculate RMS change
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(processed**2))
        rms_change_db = 20 * np.log10((rms_after + 1e-10) / (rms_before + 1e-10))
        print(f"  🎚️  RMS change: {rms_change_db:+.2f} dB")

        return True, processed, report
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None, None


def main():
    print("\n" + "=" * 60)
    print("🎵 Phase 2.3 Component Testing")
    print("=" * 60 + "\n")

    # Find test audio file
    test_files = [
        project_root / "test_audio" / "child_voice.wav",
        project_root / "input" / "test.wav",
        project_root / "test_audio" / "dummy_input.wav",
    ]

    test_file = None
    for f in test_files:
        if f.exists():
            test_file = f
            break

    if not test_file:
        print("❌ No test audio file found!")
        return 1

    print(f"📁 Using test file: {test_file.name}\n")

    # Load audio
    audio, sr = librosa.load(str(test_file), sr=48000, mono=False)

    # Convert to stereo if mono
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)
    elif audio.shape[0] == 2:  # Channels first
        audio = audio.T

    print(f"  Duration: {audio.shape[0]/sr:.2f}s")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Channels: {audio.shape[1]}")

    # Initialize all systems
    systems = {
        "Bass Enhancement": BassEnhancementSystem(),
        "Drums Enhancement": DrumsEnhancementSystem(),
        "Guitar Enhancement": GuitarEnhancementSystem(),
        "Piano Restoration": PianoRestorationSystem(),
        "Brass Enhancement": BrassEnhancementSystem(),
        "Spatial Enhancement": SpatialEnhancementSystem(),
    }

    # Test each system
    results = {}
    for name, system in systems.items():
        success, processed, report = test_component(name, system, audio, sr)
        results[name] = {"success": success, "processed": processed, "report": report}

    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}\n")

    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)

    print(f"Total: {total}")
    print(f"Successful: {successful} ✓")
    print(f"Failed: {total - successful} ✗")

    if successful == total:
        print("\n✅ ALL SYSTEMS WORKING!")
        return 0
    else:
        print(f"\n⚠️  {total - successful} SYSTEMS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
