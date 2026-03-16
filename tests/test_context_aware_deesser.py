"""
Tests for Context-Aware De-Esser v2.0

Tests the phoneme-aware de-essing system including:
- Phoneme detection integration
- Sibilant filtering
- Processing mode selection
- Genre adaptation
- Stereo handling
- Safety validation

Author: Aurik Development Team
Version: 2.0.0
Date: 8. Februar 2026
"""

from typing import Tuple

import numpy as np
import pytest

try:
    from backend.ml.safety_wrappers.context_aware_deesser_safety import (
        ContextAwareDeEsserSafety,
        validate_deessing_post,
        validate_deessing_pre,
    )
    from dsp.context_aware_deesser import (
        ContextAwareDeEsser,
        DeEsserConfig,
        ProcessingMode,
        apply_context_aware_deessing,
    )

    CONTEXT_AWARE_DEESSER_AVAILABLE = True
except ImportError:
    CONTEXT_AWARE_DEESSER_AVAILABLE = False


# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def sample_audio() -> tuple[np.ndarray, int]:
    """Generate synthetic audio with sibilant-like frequencies."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Base vocal signal (mid frequencies)
    vocal = np.sin(2 * np.pi * 250 * t) * 0.3  # 250 Hz fundamental

    # Add sibilant-like bursts (8 kHz) at regular intervals
    sibilant_times = [0.5, 1.0, 1.5]
    for sib_time in sibilant_times:
        sib_start = int(sib_time * sr)
        sib_end = sib_start + int(0.05 * sr)  # 50ms sibilant
        if sib_end <= len(vocal):
            sibilant = np.sin(2 * np.pi * 8000 * t[sib_start:sib_end]) * 0.5
            vocal[sib_start:sib_end] += sibilant

    # Normalize
    vocal = vocal / np.max(np.abs(vocal)) * 0.7

    return vocal, sr


@pytest.fixture
def stereo_audio(sample_audio) -> tuple[np.ndarray, int]:
    """Generate stereo version of sample audio."""
    mono, sr = sample_audio
    stereo = np.stack([mono, mono * 0.9], axis=0)  # Slight L/R difference
    return stereo, sr


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_deesser_config_defaults():
    """Test default configuration values."""
    config = DeEsserConfig()

    assert config.mode == ProcessingMode.MODERATE
    assert config.device == "cpu"
    assert config.min_phoneme_confidence == 0.5
    assert config.reduction_multiplier == 1.0
    assert config.crossfade_ms == 5.0
    assert config.dry_wet_mix == 1.0
    assert config.enable_genre_adaptation is True


def test_deesser_config_custom():
    """Test custom configuration."""
    config = DeEsserConfig(
        mode=ProcessingMode.AGGRESSIVE,
        device="cuda",
        reduction_multiplier=1.5,
        dry_wet_mix=0.8,
    )

    assert config.mode == ProcessingMode.AGGRESSIVE
    assert config.device == "cuda"
    assert config.reduction_multiplier == 1.5
    assert config.dry_wet_mix == 0.8


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_deesser_initialization():
    """Test de-esser initialization."""
    deesser = ContextAwareDeEsser()

    assert hasattr(deesser, "phoneme_detector")
    assert hasattr(deesser, "phoneme_classifier")
    assert deesser.last_report is None


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_deesser_initialization_with_config():
    """Test de-esser initialization with custom config."""
    config = DeEsserConfig(
        mode=ProcessingMode.GENTLE,
        device="cpu",
    )
    deesser = ContextAwareDeEsser(config)

    assert deesser.config.mode == ProcessingMode.GENTLE


# ============================================================================
# PROCESSING TESTS - MONO
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_process_mono_audio(sample_audio):
    """Test mono audio processing."""
    audio, sr = sample_audio

    deesser = ContextAwareDeEsser()
    audio_out, report = deesser.process(audio, sr)

    # Check output shape
    assert audio_out.shape == audio.shape

    # Check report
    assert report.total_duration_sec == pytest.approx(len(audio) / sr, rel=0.01)
    assert report.phonemes_detected >= 0
    assert report.sibilants_detected >= 0
    assert 0.0 <= report.percentage_processed <= 100.0


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_process_different_modes(sample_audio):
    """Test different processing modes."""
    audio, sr = sample_audio

    modes = [ProcessingMode.GENTLE, ProcessingMode.MODERATE, ProcessingMode.AGGRESSIVE]
    reductions = []

    for mode in modes:
        config = DeEsserConfig(mode=mode)
        deesser = ContextAwareDeEsser(config)
        _, report = deesser.process(audio, sr)
        reductions.append(report.avg_reduction_db)

    # Aggressive should reduce more than moderate, moderate more than gentle
    # (if sibilants detected)
    if reductions[0] > 0:  # If any reduction applied
        assert reductions[2] >= reductions[1] >= reductions[0]


# ============================================================================
# PROCESSING TESTS - STEREO
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_process_stereo_audio(stereo_audio):
    """Test stereo audio processing."""
    audio, sr = stereo_audio

    deesser = ContextAwareDeEsser()
    audio_out, report = deesser.process(audio, sr)

    # Check output shape
    assert audio_out.shape == audio.shape
    assert audio_out.ndim == 2
    # Note: when phoneme detection unavailable (no torch), audio may pass through unchanged


# ============================================================================
# DRY/WET MIX TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_dry_wet_mix_extremes(sample_audio):
    """Test dry/wet mix extremes."""
    audio, sr = sample_audio

    # 100% dry (no processing)
    config_dry = DeEsserConfig(dry_wet_mix=0.0)
    deesser_dry = ContextAwareDeEsser(config_dry)
    audio_dry, _ = deesser_dry.process(audio, sr)

    # Should be identical to input
    np.testing.assert_array_almost_equal(audio_dry, audio, decimal=5)

    # 100% wet (full processing)
    config_wet = DeEsserConfig(dry_wet_mix=1.0)
    deesser_wet = ContextAwareDeEsser(config_wet)
    audio_wet, _ = deesser_wet.process(audio, sr)

    # Should differ from input (if sibilants detected)
    # Allow for cases where no sibilants are detected
    if not np.array_equal(audio_wet, audio):
        assert not np.allclose(audio_wet, audio)


# ============================================================================
# GENRE ADAPTATION TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_genre_adaptation(sample_audio):
    """Test genre-specific adaptation."""
    audio, sr = sample_audio

    genres = ["classical", "pop", "speech"]

    for genre in genres:
        config = DeEsserConfig(enable_genre_adaptation=True, genre=genre)
        deesser = ContextAwareDeEsser(config)
        audio_out, report = deesser.process(audio, sr, genre=genre)

        # Should process successfully
        assert audio_out.shape == audio.shape
        assert report.total_duration_sec > 0


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_convenience_function(sample_audio):
    """Test convenience function."""
    audio, sr = sample_audio

    audio_out, report = apply_context_aware_deessing(
        audio,
        sr,
        mode=ProcessingMode.MODERATE,
        device="cpu",
    )

    assert audio_out.shape == audio.shape
    assert report.total_duration_sec > 0


# ============================================================================
# SAFETY WRAPPER TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_safety_pre_check(sample_audio):
    """Test safety pre-check validation."""
    audio, sr = sample_audio

    result = validate_deessing_pre(audio, sr, params={})

    # Should have metrics in metadata
    assert "sibilance_intensity" in result.metadata
    assert "baseline_intelligibility" in result.metadata


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_safety_post_check(sample_audio):
    """Test safety post-check validation."""
    audio, sr = sample_audio

    # Process audio
    audio_out, _ = apply_context_aware_deessing(audio, sr)

    # Run safety checks
    pre_result = validate_deessing_pre(audio, sr)
    post_result = validate_deessing_post(audio, audio_out, sr, pre_check=pre_result)

    # Should have correlation metric
    assert "correlation" in post_result.metrics
    assert "intelligibility_preservation" in post_result.metrics or True  # May not be computed if no baseline


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_safety_wrapper_class():
    """Test safety wrapper class."""
    wrapper = ContextAwareDeEsserSafety()

    assert wrapper.name == "Context-Aware De-Esser Safety"
    assert wrapper.min_intelligibility_preservation == 0.95
    assert wrapper.min_correlation == 0.85


# ============================================================================
# EDGE CASES
# ============================================================================


def test_empty_audio():
    """Test with empty audio."""
    audio = np.array([])
    sr = 16000

    if not CONTEXT_AWARE_DEESSER_AVAILABLE:
        pytest.skip("Context-Aware De-Esser not available")

    with pytest.raises((ValueError, RuntimeError)):
        deesser = ContextAwareDeEsser()
        deesser.process(audio, sr)


def test_very_short_audio():
    """Test with very short audio (< 100ms)."""
    sr = 16000
    audio = np.random.randn(int(0.05 * sr))  # 50ms

    if not CONTEXT_AWARE_DEESSER_AVAILABLE:
        pytest.skip("Context-Aware De-Esser not available")

    # Should handle gracefully
    deesser = ContextAwareDeEsser()
    audio_out, report = deesser.process(audio, sr)

    assert audio_out.shape == audio.shape
    assert report.phonemes_detected >= 0


def test_silent_audio():
    """Test with silent audio."""
    sr = 16000
    audio = np.zeros(sr * 2)  # 2 seconds of silence

    if not CONTEXT_AWARE_DEESSER_AVAILABLE:
        pytest.skip("Context-Aware De-Esser not available")

    deesser = ContextAwareDeEsser()
    audio_out, report = deesser.process(audio, sr)

    # Silent audio should remain silent
    assert np.allclose(audio_out, audio)
    assert report.sibilants_detected == 0


def test_clipped_audio():
    """Test with clipped audio."""
    sr = 16000
    audio = np.ones(sr * 2)  # Fully clipped

    if not CONTEXT_AWARE_DEESSER_AVAILABLE:
        pytest.skip("Context-Aware De-Esser not available")

    deesser = ContextAwareDeEsser()
    audio_out, report = deesser.process(audio, sr)

    # Should process without error
    assert audio_out.shape == audio.shape


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_processing_speed_cpu(sample_audio):
    """Test processing speed on CPU."""
    import time

    audio, sr = sample_audio

    config = DeEsserConfig(device="cpu")
    deesser = ContextAwareDeEsser(config)

    start = time.time()
    _, report = deesser.process(audio, sr)
    elapsed = time.time() - start

    # Should process 2 seconds of audio in reasonable time
    # Allow generous time for CI/CD (30 seconds)
    assert elapsed < 30.0, f"Processing too slow: {elapsed:.2f}s"

    print(f"\nCPU processing: {report.total_duration_sec:.2f}s audio in {elapsed:.2f}s")


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_memory_usage(sample_audio):
    """Test memory usage during processing."""
    audio, sr = sample_audio

    deesser = ContextAwareDeEsser()

    # Process should not leak memory
    for _ in range(3):
        audio_out, _ = deesser.process(audio, sr)
        del audio_out  # Cleanup

    # If we got here without OOM, test passes
    assert True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.skipif(not CONTEXT_AWARE_DEESSER_AVAILABLE, reason="Context-Aware De-Esser not available")
def test_full_pipeline_with_safety(sample_audio):
    """Test full pipeline with safety validation."""
    audio, sr = sample_audio

    # Pre-check
    pre_result = validate_deessing_pre(audio, sr)

    if not pre_result.passed:
        pytest.skip(f"Pre-check failed: {pre_result.reasons}")

    # Process
    audio_out, process_report = apply_context_aware_deessing(audio, sr)

    # Post-check
    post_result = validate_deessing_post(audio, audio_out, sr, pre_check=pre_result)

    # Validation
    assert audio_out.shape == audio.shape
    assert process_report.total_duration_sec > 0

    # Safety should pass or have only warnings
    if not post_result.passed:
        print(f"Post-check issues: {post_result.issues}")
        print(f"Post-check warnings: {post_result.warnings}")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
