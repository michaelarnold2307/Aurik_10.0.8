"""
Test suite for core/export_workflow.py - Export functions
Tests audio export and audit log export
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.export_workflow import export_audio, export_audit_log


@pytest.fixture
def temp_export_dir():
    """Create temporary export directory"""
    temp_dir = tempfile.mkdtemp()
    export_dir = os.path.join(temp_dir, "export")
    logs_dir = os.path.join(temp_dir, "logs")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Change to temp dir
    orig_dir = os.getcwd()
    os.chdir(temp_dir)

    yield temp_dir

    # Cleanup
    os.chdir(orig_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_audio_basic(temp_export_dir):
    """Test basic audio export"""
    audio = np.random.randn(1000) * 0.5
    sr = 48000
    filename = "test_audio.wav"

    export_path = export_audio(audio, sr, filename)

    # Check file was created
    assert os.path.exists(export_path)
    assert export_path == os.path.join("export", filename)


def test_export_audio_stereo(temp_export_dir):
    """Test stereo audio export"""
    audio = np.random.randn(1000, 2) * 0.5  # Stereo
    sr = 44100
    filename = "test_stereo.wav"

    export_path = export_audio(audio, sr, filename)

    # Check file exists
    assert os.path.exists(export_path)


def test_export_audio_different_sample_rates(temp_export_dir):
    """Test audio export with different sample rates"""
    audio = np.random.randn(500) * 0.5

    # Test 44.1 kHz
    path1 = export_audio(audio, 44100, "test_44k.wav")
    assert os.path.exists(path1)

    # Test 48 kHz
    path2 = export_audio(audio, 48000, "test_48k.wav")
    assert os.path.exists(path2)

    # Test 96 kHz
    path3 = export_audio(audio, 96000, "test_96k.wav")
    assert os.path.exists(path3)


def test_export_audio_high_res(temp_export_dir):
    """Test high-resolution audio export"""
    audio = np.random.randn(2000) * 0.5
    sr = 192000
    filename = "test_hires.wav"

    export_path = export_audio(audio, sr, filename)

    assert os.path.exists(export_path)


def test_export_audit_log_basic(temp_export_dir):
    """Test basic audit log export"""
    audit_log = [{"step": "denoising", "status": "success"}, {"step": "restoration", "status": "complete"}]
    filename = "test_audit.log"

    log_path = export_audit_log(audit_log, filename)

    # Check file was created
    assert os.path.exists(log_path)
    assert log_path == os.path.join("logs", filename)


def test_export_audit_log_content(temp_export_dir):
    """Test audit log exports correct content"""
    audit_log = [{"step": "analysis", "result": "vinyl_detected"}, {"step": "filtering", "params": {"freq": 50}}]
    filename = "test_content.log"

    log_path = export_audit_log(audit_log, filename)

    # Read file and check content
    with open(log_path) as f:
        content = f.read()

    assert "analysis" in content
    assert "vinyl_detected" in content
    assert "filtering" in content


def test_export_audit_log_empty(temp_export_dir):
    """Test audit log export with empty log"""
    audit_log = []
    filename = "test_empty.log"

    log_path = export_audit_log(audit_log, filename)

    # File should exist but be empty
    assert os.path.exists(log_path)


def test_export_audit_log_append(temp_export_dir):
    """Test audit log appends to existing file"""
    filename = "test_append.log"

    # First export
    audit_log1 = [{"entry": 1}]
    log_path = export_audit_log(audit_log1, filename)

    # Second export (should append)
    audit_log2 = [{"entry": 2}]
    export_audit_log(audit_log2, filename)

    # Check both entries are in file
    with open(log_path) as f:
        content = f.read()

    assert "entry" in content
    # Should have 2 lines (one per entry)
    lines = content.strip().split("\n")
    assert len(lines) >= 2


def test_export_audio_return_path(temp_export_dir):
    """Test export_audio returns correct path"""
    audio = np.random.randn(500) * 0.5
    sr = 48000
    filename = "test_return.wav"

    path = export_audio(audio, sr, filename)

    # Path should contain export directory and filename
    assert "export" in path
    assert filename in path
    assert path.endswith(filename)


def test_export_audit_log_return_path(temp_export_dir):
    """Test export_audit_log returns correct path"""
    audit_log = [{"test": "data"}]
    filename = "test_return.log"

    path = export_audit_log(audit_log, filename)

    # Path should contain logs directory and filename
    assert "logs" in path
    assert filename in path
    assert path.endswith(filename)


def test_export_audio_zero_signal(temp_export_dir):
    """Test audio export with zero (silent) signal"""
    audio = np.zeros(1000)
    sr = 48000
    filename = "test_silent.wav"

    export_path = export_audio(audio, sr, filename)

    assert os.path.exists(export_path)


def test_export_audit_log_complex_entries(temp_export_dir):
    """Test audit log with complex nested entries"""
    audit_log = [
        {
            "step": "restoration",
            "params": {"mode": "vinyl", "settings": {"denoise": True, "declicker": True}},
            "result": {"quality": 0.92, "snr": 35.5},
        }
    ]
    filename = "test_complex.log"

    log_path = export_audit_log(audit_log, filename)

    # Check file exists and has content
    assert os.path.exists(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "restoration" in content
    assert "denoise" in content
