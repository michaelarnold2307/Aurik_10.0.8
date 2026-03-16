#!/usr/bin/env python
"""Quick validation: Die 2 AURIK Magic Buttons"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.core.processing_modes import ProcessingMode, get_processing_config, list_available_modes

print("\n" + "=" * 80)
print("AURIK MAGIC BUTTONS - VALIDATION")
print("=" * 80)

# List all modes
modes = list_available_modes()
print(f"\n✓ AURIK Magic Buttons: {len(modes)}\n")

assert len(modes) == 2, f"❌ Erwartet: 2 Magic Buttons, gefunden: {len(modes)}"

for mode_name, description in modes.items():
    print(f"  🎛️  {mode_name.upper():<20}")
    print(f"     {description}")

# Validate each mode has a config
print("\n" + "=" * 80)
print("CONFIGURATION VALIDATION")
print("=" * 80)

all_valid = True
for mode in ProcessingMode:
    try:
        config = get_processing_config(mode)
        print(f"\n✓ {mode.value.upper()}")
        print(f"    Denoise: {config.denoise_strength:.2f}")
        print(f"    Compression: {config.compression_ratio:.1f}:1")
        print(f"    Target LUFS: {config.target_lufs if config.target_lufs else 'Original'}")
        print(f"    High Freq: {config.high_freq_boost_db:+.1f} dB")
        print(f"    Enhancement: {config.enhancement_strength:.2f}")
    except Exception as e:
        print(f"❌ {mode.value.upper()} - ERROR: {e}")
        all_valid = False

print("\n" + "=" * 80)
if all_valid:
    print("✅ BEIDE MAGIC BUTTONS SIND KORREKT KONFIGURIERT!")
    print("")
    print("Magic Button 1: RESTORATION")
    print("  → Originalgetreue Restauration, sanft, natürlich")
    print("")
    print("Magic Button 2: STUDIO_2026")
    print("  → Moderner Highend-Studio-Sound, streaming-ready")
else:
    print("❌ CONFIGURATION ERRORS DETECTED!")
print("=" * 80)
