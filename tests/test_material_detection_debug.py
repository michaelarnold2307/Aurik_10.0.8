"""Debug script for material detection"""

import logging

import librosa
import numpy as np

from backend.core.defect_scanner import DefectScanner, MaterialType

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

# Load test files
vinyl_file = "test_audio/vinyl/jazz_1950s_scratched.wav"
tape_file = "test_audio/tape/cassette_1980s_wow.wav"

print("=" * 80)
print("MATERIAL DETECTION DEBUG")
print("=" * 80)

for name, file_path, expected in [
    ("Vinyl Jazz", vinyl_file, MaterialType.VINYL),
    ("Tape Cassette", tape_file, MaterialType.TAPE),
]:
    print(f"\n{name}: {file_path}")
    print("-" * 80)

    # Load audio
    audio, sr = librosa.load(file_path, sr=None, mono=False, duration=5.0)
    print(f"Original: shape={audio.shape}, sr={sr}")

    # Resample to 48kHz (like in unified_restorer_v3)
    if sr != 48000:
        if audio.ndim == 2:
            audio = np.column_stack(
                [
                    librosa.resample(audio[0, :], orig_sr=sr, target_sr=48000),
                    librosa.resample(audio[1, :], orig_sr=sr, target_sr=48000),
                ]
            )
        else:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    print(f"After resample: shape={audio.shape}, sr={sr}")
    print(f"ndim: {audio.ndim}, dtype: {audio.dtype}")

    # Initialize scanner
    scanner = DefectScanner(sample_rate=sr, material_type=None)

    # Test auto-detection with detailed diagnostics
    print("\nAudio properties:")
    print(f"  - Channels: {2 if audio.ndim == 2 else 1}")
    print(f"  - Duration: {audio.shape[-1] / sr:.2f}s")
    print(f"  - RMS: {np.sqrt(np.mean(audio**2)):.6f}")

    # Extract features manually to see what's being detected
    from scipy import signal

    # Click rate
    diff = np.abs(np.diff(audio))
    click_rate = np.sum(diff > np.percentile(diff, 99.9)) / len(audio) * sr

    # Spectral features
    freqs, psd = signal.welch(audio, sr, nperseg=2048)
    high_freq_energy = np.sum(psd[freqs > 8000]) / np.sum(psd)
    rumble_energy = np.sum(psd[freqs < 60]) / np.sum(psd)
    hum_energy = np.sum(psd[(freqs >= 50) & (freqs <= 70)]) / np.sum(psd)

    print("\nFeature values:")
    print(f"  - click_rate: {click_rate:.3f} clicks/sec")
    print(f"  - high_freq_energy: {high_freq_energy:.4f}")
    print(f"  - rumble_energy: {rumble_energy:.4f}")
    print(f"  - hum_energy: {hum_energy:.4f}")

    # Get crackle and wow/flutter scores from scanner
    crackle_result = scanner._detect_crackle(audio)
    wow_flutter_result = scanner._detect_wow_flutter(audio)

    print(f"  - crackle_score: {crackle_result.severity:.4f}")
    print(f"  - wow_flutter_score: {wow_flutter_result.severity:.4f}")

    detected = scanner._auto_detect_material(audio)

    print(f"\nExpected: {expected.value}")
    print(f"Detected: {detected.value}")
    print(f"Result: {'✅ CORRECT' if detected == expected else '❌ WRONG'}")
