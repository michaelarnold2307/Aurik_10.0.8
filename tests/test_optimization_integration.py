#!/usr/bin/env python3
"""
Test script for Balanced Optimization integration
"""

import sys
import tempfile

import numpy as np
import soundfile as sf


def test_optimization_integration():
    """Test that optimization works end-to-end"""
    import os

    import pytest

    if not os.path.exists("orchestrator_and_cli.py"):
        pytest.skip("orchestrator_and_cli.py nicht gefunden – Test uebersprungen")

    print("=" * 70)
    print("BALANCED OPTIMIZATION INTEGRATION TEST")
    print("=" * 70)

    # 1. Generate test audio
    print("\n[1/4] Generating test audio...")
    sr = 48000
    duration = 5  # 5 seconds
    audio = np.random.randn(sr * duration).astype(np.float32) * 0.1

    # Create temp files
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
        input_file = tmp_input.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_output:
        output_file = tmp_output.name

    try:
        # Save test audio
        sf.write(input_file, audio, sr)
        print(f"✓ Test audio saved: {input_file}")

        # 2. Test WITHOUT optimization
        print("\n[2/4] Testing WITHOUT optimization...")
        # Security: Use subprocess.run() with list instead of os.system() (CWE-78)
        import subprocess

        subprocess.run(
            [
                "python",
                "orchestrator_and_cli.py",
                "--input",
                input_file,
                "--output",
                output_file,
                "--optimization",
                "none",
                "--log",
                "/dev/null",
            ],
            check=False,
            stderr=subprocess.DEVNULL,
        )

        if os.path.exists(output_file):
            result_audio, result_sr = sf.read(output_file)
            print("✓ Processing complete (no optimization)")
            print(f"  Output: {len(result_audio)} samples @ {result_sr} Hz")
        else:
            print("❌ Processing failed (no output file)")
            return False

        # 3. Test WITH balanced optimization
        print("\n[3/4] Testing WITH balanced optimization...")
        os.remove(output_file)

        # Security: Use subprocess.run() with list instead of os.system() (CWE-78)
        import subprocess

        cmd_list = [
            "/mnt/1846D15B46D139E8/Aurik_Standalone/.venv_aurik/bin/python",
            "orchestrator_and_cli.py",
            "--input",
            input_file,
            "--output",
            output_file,
            "--optimization",
            "balanced",
            "--genre",
            "rock",
            "--log",
            "/dev/null",
        ]
        print(f"Command: {' '.join(cmd_list)}")
        ret = subprocess.run(cmd_list, check=False).returncode

        if ret == 0 and os.path.exists(output_file):
            result_audio, result_sr = sf.read(output_file)
            print("✓ Processing complete (balanced optimization)")
            print(f"  Output: {len(result_audio)} samples @ {result_sr} Hz")

            # Quality checks
            if np.isfinite(result_audio).all():
                print("✓ Output is finite (no NaN/Inf)")
            else:
                print("❌ Output contains NaN/Inf")
                return False

            if np.abs(result_audio).max() <= 1.0:
                print(f"✓ Output amplitude OK ({np.abs(result_audio).max():.3f})")
            else:
                print(f"⚠️  Output amplitude high ({np.abs(result_audio).max():.3f})")
        else:
            print(f"❌ Processing failed (return code: {ret})")
            return False

        # 4. Test gentle and aggressive presets
        print("\n[4/4] Testing other presets...")
        for preset in ["gentle", "aggressive"]:
            os.remove(output_file)
            # Security: Use subprocess.run() with list instead of os.system() (CWE-78)
            cmd_list = [
                "/mnt/1846D15B46D139E8/Aurik_Standalone/.venv_aurik/bin/python",
                "orchestrator_and_cli.py",
                "--input",
                input_file,
                "--output",
                output_file,
                "--optimization",
                preset,
                "--genre",
                "jazz",
                "--log",
                "/dev/null",
            ]
            ret = subprocess.run(cmd_list, check=False).returncode

            if ret == 0 and os.path.exists(output_file):
                print(f"✓ Preset '{preset}' works")
            else:
                print(f"❌ Preset '{preset}' failed")
                return False

        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 70)
        return True

    finally:
        # Cleanup
        for f in [input_file, output_file]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == "__main__":
    success = test_optimization_integration()
    sys.exit(0 if success else 1)
