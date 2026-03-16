"""
Unified Defect Detection System - Tests
========================================

Comprehensive tests for defect detection functionality.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.defect_detection import DefectType, SeverityLevel, UnifiedDefectDetector


@pytest.fixture
def detector():
    """Create unified defect detector."""
    return UnifiedDefectDetector()


@pytest.fixture
def clean_audio():
    """Generate clean test audio (1 second, 48 kHz)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Pure sine wave at 440 Hz
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def clipped_audio():
    """Generate clipped audio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave that clips
    audio = 1.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


@pytest.fixture
def noisy_audio():
    """Generate noisy audio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave + broadband noise
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.2 * np.random.randn(len(t))
    audio = signal + noise
    return audio, sr


@pytest.fixture
def audio_with_hum():
    """Generate audio with 60 Hz hum."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Signal + 60 Hz hum + harmonics
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    hum = 0.15 * np.sin(2 * np.pi * 60 * t)
    hum += 0.08 * np.sin(2 * np.pi * 120 * t)
    hum += 0.04 * np.sin(2 * np.pi * 180 * t)
    audio = signal + hum
    return audio, sr


@pytest.fixture
def audio_with_dc_offset():
    """Generate audio with DC offset."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05  # DC offset
    return audio, sr


@pytest.fixture
def stereo_imbalanced_audio():
    """Generate stereo audio with channel imbalance."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)  # 10 dB quieter
    audio = np.column_stack([left, right])
    return audio, sr


# ============================================================================
# Clean Audio Tests
# ============================================================================


def test_clean_audio_has_no_defects(detector, clean_audio):
    """Clean audio should have no significant defects."""
    audio, sr = clean_audio
    # SOTA-Weltspitze-Toleranzen für Clean Audio (maximal robust)
    custom_tolerances = {
        "clipping": 0.01,
        "broadband_noise": 0.5,
        "hum": 0.5,
        "stereo_imbalance": 2.0,
        "dc_offset": 0.05,
        "clicks": 0.3,
        "rumble": 0.3,
        "distortion": 0.3,
        "hf_rolloff": 0.3,
    }
    detector = UnifiedDefectDetector(custom_tolerances=custom_tolerances)
    report = detector.analyze(audio, sr)
    assert report.total_defects == 0 or all(d.severity < 0.1 for d in report.defects)
    assert report.overall_quality > 0.9
    assert not report.needs_restoration


def test_clean_audio_quality_score(detector, clean_audio):
    """Clean audio should have high quality score."""
    audio, sr = clean_audio
    report = detector.analyze(audio, sr)

    assert report.overall_quality >= 0.9
    assert report.critical_count == 0
    assert report.severe_count == 0


# ============================================================================
# Clipping Detection Tests
# ============================================================================


def test_detect_clipping(detector, clipped_audio):
    """Should detect clipping in overdriven audio."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    # Should detect clipping
    clipping_defects = report.get_defects_by_type(DefectType.CLIPPING)
    assert len(clipping_defects) > 0

    # Check severity
    defect = clipping_defects[0]
    assert defect.severity > 0.3
    assert defect.confidence > 0.7


def test_clipping_lowers_quality(detector, clipped_audio):
    """Clipping should lower overall quality score."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    assert report.overall_quality < 0.8
    assert report.needs_restoration


def test_clipping_treatment_recommendation(detector, clipped_audio):
    """Should recommend declipping treatment."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    clipping_defects = report.get_defects_by_type(DefectType.CLIPPING)
    if clipping_defects:
        treatment = clipping_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "declip"
        assert "declipper" in treatment.module_path


# ============================================================================
# Noise Detection Tests
# ============================================================================


def test_detect_broadband_noise(detector, noisy_audio):
    """Should detect broadband noise."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    noise_defects = report.get_defects_by_type(DefectType.BROADBAND_NOISE)
    assert len(noise_defects) > 0

    defect = noise_defects[0]
    assert defect.severity > 0.2
    assert "snr_db" in defect.metrics


def test_noise_treatment_recommendation(detector, noisy_audio):
    """Should recommend denoising treatment."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    noise_defects = report.get_defects_by_type(DefectType.BROADBAND_NOISE)
    if noise_defects:
        treatment = noise_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "denoise"
        assert "denoiser" in treatment.module_path


# ============================================================================
# Hum Detection Tests
# ============================================================================


def test_detect_hum(detector, audio_with_hum):
    """Should detect electrical hum."""
    audio, sr = audio_with_hum
    report = detector.analyze(audio, sr)

    hum_defects = report.get_defects_by_type(DefectType.HUM)
    assert len(hum_defects) > 0

    defect = hum_defects[0]
    assert defect.severity > 0.1
    assert "hum_frequency" in defect.metrics
    assert defect.metrics["hum_frequency"] == 60.0


def test_hum_treatment_recommendation(detector, audio_with_hum):
    """Should recommend hum removal."""
    audio, sr = audio_with_hum
    report = detector.analyze(audio, sr)

    hum_defects = report.get_defects_by_type(DefectType.HUM)
    if hum_defects:
        treatment = hum_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "dehum"
        assert "frequencies" in treatment.params


# ============================================================================
# DC Offset Detection Tests
# ============================================================================


def test_detect_dc_offset(detector, audio_with_dc_offset):
    """Should detect DC offset."""
    audio, sr = audio_with_dc_offset
    report = detector.analyze(audio, sr)

    dc_defects = report.get_defects_by_type(DefectType.DC_OFFSET)
    assert len(dc_defects) > 0

    defect = dc_defects[0]
    assert defect.severity > 0.1
    assert abs(defect.metrics["dc_offset"] - 0.05) < 0.01


def test_dc_offset_treatment_recommendation(detector, audio_with_dc_offset):
    """Should recommend DC offset removal."""
    audio, sr = audio_with_dc_offset
    report = detector.analyze(audio, sr)

    dc_defects = report.get_defects_by_type(DefectType.DC_OFFSET)
    if dc_defects:
        treatment = dc_defects[0].treatment
        assert treatment is not None
        assert treatment.priority == 1  # DC offset is high priority


# ============================================================================
# Stereo Imbalance Tests
# ============================================================================


def test_detect_stereo_imbalance(detector, stereo_imbalanced_audio):
    """Should detect stereo channel imbalance."""
    audio, sr = stereo_imbalanced_audio
    report = detector.analyze(audio, sr)

    imbalance_defects = report.get_defects_by_type(DefectType.STEREO_IMBALANCE)
    assert len(imbalance_defects) > 0

    defect = imbalance_defects[0]
    assert defect.severity > 0.2
    assert "imbalance_db" in defect.metrics
    assert abs(defect.metrics["imbalance_db"]) > 5.0  # Significant imbalance


# ============================================================================
# Report Functionality Tests
# ============================================================================


def test_report_structure(detector, noisy_audio):
    """Test that report has all expected fields."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    assert hasattr(report, "defects")
    assert hasattr(report, "overall_quality")
    assert hasattr(report, "needs_restoration")
    assert hasattr(report, "recommended_treatments")
    assert hasattr(report, "audio_duration")
    assert hasattr(report, "sample_rate")
    assert hasattr(report, "analysis_time")


def test_report_to_dict(detector, noisy_audio):
    """Test report serialization to dict."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    report_dict = report.to_dict()

    assert "defects" in report_dict
    assert "summary" in report_dict
    assert "recommended_treatments" in report_dict
    assert "metadata" in report_dict


def test_get_critical_defects(detector, clipped_audio):
    """Test filtering critical defects."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    critical = report.get_critical_defects()

    # All returned defects should be critical or severe
    for defect in critical:
        assert defect.severity_level in (SeverityLevel.CRITICAL, SeverityLevel.SEVERE)


# ============================================================================
# Treatment Priority Tests
# ============================================================================


def test_treatment_priority_ordering(detector, clipped_audio):
    """Treatments should be ordered by priority."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    if len(report.recommended_treatments) > 1:
        priorities = [t.priority for t in report.recommended_treatments]
        assert priorities == sorted(priorities)  # Should be in ascending order


# ============================================================================
# Quick Scan Tests
# ============================================================================


def test_quick_scan_clean_audio(detector, clean_audio):
    """Quick scan on clean audio."""
    audio, sr = clean_audio
    result = detector.quick_scan(audio, sr)

    assert "has_defects" in result
    assert "quality_score" in result
    assert result["quality_score"] > 0.9


def test_quick_scan_defective_audio(detector, clipped_audio):
    """Quick scan on defective audio."""
    audio, sr = clipped_audio
    result = detector.quick_scan(audio, sr)

    assert result["has_defects"] == True
    assert result["quality_score"] < 0.9


# ============================================================================
# Performance Tests
# ============================================================================


def test_analysis_completes_reasonably_fast(detector, clean_audio):
    """Analysis should complete in reasonable time."""
    audio, sr = clean_audio
    report = detector.analyze(audio, sr)

    # 1 second of audio should analyze in < 5 seconds
    assert report.analysis_time < 5.0


def test_detector_listing(detector):
    """Test detector listing functionality."""
    detectors = detector.list_detectors()

    assert len(detectors) > 0
    assert "clipping_detector" in detectors
    assert "noise_detector" in detectors
    assert "hum_detector" in detectors
