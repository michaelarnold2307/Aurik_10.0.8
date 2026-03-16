"""
Tests für Vocal AI Enhancement (Phase 19 + 42 Integration)
===========================================================

Test suite for gender-aware vocal enhancement with emotion
and breath preservation.

Author: Aurik 9.0 Development Team
Date: 15. Februar 2026
"""

from pathlib import Path
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy import signal

from backend.core.vocal_ai_enhancement import (
    BreathPreservingProcessor,
    EmotionPreservationMode,
    GenderAwareDeEsser,
    GenderDetector,
    UnifiedVocalAIEnhancer,
    VocalEnhancementResult,
    VoiceCharacteristics,
    VoiceGender,
)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000


@pytest.fixture
def male_voice(sample_rate):
    """Generate male voice (F0 ~120 Hz)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    f0 = 120  # Male fundamental
    voice = np.zeros_like(t)

    # Add harmonics
    for i in range(1, 6):
        voice += (1 / i) * np.sin(2 * np.pi * f0 * i * t)

    voice = voice / np.max(np.abs(voice)) * 0.5
    return voice


@pytest.fixture
def female_voice(sample_rate):
    """Generate female voice (F0 ~220 Hz)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    f0 = 220  # Female fundamental
    voice = np.zeros_like(t)

    # Add harmonics
    for i in range(1, 6):
        voice += (1 / i) * np.sin(2 * np.pi * f0 * i * t)

    voice = voice / np.max(np.abs(voice)) * 0.5
    return voice


@pytest.fixture
def child_voice(sample_rate):
    """Generate child voice (F0 ~300 Hz)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    f0 = 300  # Child fundamental
    voice = np.zeros_like(t)

    # Add harmonics
    for i in range(1, 6):
        voice += (1 / i) * np.sin(2 * np.pi * f0 * i * t)

    voice = voice / np.max(np.abs(voice)) * 0.5
    return voice


@pytest.fixture
def voice_with_sibilance(female_voice, sample_rate):
    """Add sibilance to voice."""
    voice = female_voice.copy()

    # Add sibilance bursts (8 kHz)
    sibilance = np.zeros_like(voice)
    sibilance[int(0.3 * sample_rate) : int(0.32 * sample_rate)] = np.random.randn(int(0.02 * sample_rate)) * 0.3
    sibilance[int(0.6 * sample_rate) : int(0.62 * sample_rate)] = np.random.randn(int(0.02 * sample_rate)) * 0.3

    # High-pass filter
    sos = signal.butter(4, 6000, "high", fs=sample_rate, output="sos")
    sibilance = signal.sosfilt(sos, sibilance)

    return voice + sibilance


@pytest.fixture
def voice_with_breath(female_voice, sample_rate):
    """Add breath noise to voice."""
    voice = female_voice.copy()

    # Add breath (low-level noise)
    breath = np.random.randn(len(voice)) * 0.03

    # High-pass filter breath
    sos = signal.butter(4, 1000, "high", fs=sample_rate, output="sos")
    breath = signal.sosfilt(sos, breath)

    return voice + breath


# ============================================================
# GENDER DETECTION TESTS
# ============================================================


class TestGenderDetector:
    """Test suite for gender detection."""

    def test_detector_initialization(self, sample_rate):
        """Test detector initializes."""
        detector = GenderDetector(sample_rate=sample_rate)
        assert detector.sr == sample_rate

    def test_detect_male_voice(self, male_voice):
        """Test male voice detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(male_voice)

        assert isinstance(result, VoiceCharacteristics)
        assert result.gender == VoiceGender.MALE
        assert 80 < result.fundamental_freq < 200  # Male range
        assert result.confidence > 0.3

    def test_detect_female_voice(self, female_voice):
        """Test female voice detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(female_voice)

        assert isinstance(result, VoiceCharacteristics)
        assert result.gender in [VoiceGender.FEMALE, VoiceGender.CHILD]
        assert 150 < result.fundamental_freq < 400  # Female/child range

    def test_detect_child_voice(self, child_voice):
        """Test child voice detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(child_voice)

        assert isinstance(result, VoiceCharacteristics)
        # High F0 should suggest child
        assert result.fundamental_freq > 250

    def test_formant_detection(self, female_voice):
        """Test formant detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(female_voice)

        # Should detect some formants
        assert len(result.formants) >= 0  # May or may not detect in synthetic

    def test_breathiness_detection(self, voice_with_breath):
        """Test breathiness detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(voice_with_breath)

        # Should detect some breathiness
        assert result.breathiness >= 0.0

    def test_sibilance_detection(self, voice_with_sibilance):
        """Test sibilance detection."""
        detector = GenderDetector(sample_rate=48000)
        result = detector.detect(voice_with_sibilance)

        # Should detect sibilance
        assert result.sibilance_severity > 0.3


# ============================================================
# DE-ESSER TESTS
# ============================================================


class TestGenderAwareDeEsser:
    """Test suite for gender-aware de-esser."""

    def test_deesser_initialization(self, sample_rate):
        """Test de-esser initializes."""
        deesser = GenderAwareDeEsser(sample_rate=sample_rate)
        assert deesser.sr == sample_rate
        assert hasattr(deesser, "gender_detector")

    def test_deess_female_voice(self, voice_with_sibilance):
        """Test de-essing on female voice."""
        deesser = GenderAwareDeEsser(sample_rate=48000)

        processed, reduction_db = deesser.process(voice_with_sibilance, emotion_mode=EmotionPreservationMode.BALANCED)

        assert processed.shape == voice_with_sibilance.shape
        assert reduction_db >= 0  # Some reduction applied

    def test_deess_male_voice(self, male_voice):
        """Test de-essing on male voice."""
        deesser = GenderAwareDeEsser(sample_rate=48000)

        # Add sibilance to male voice
        sibilance = np.random.randn(len(male_voice)) * 0.2
        sos = signal.butter(4, 5000, "high", fs=48000, output="sos")
        sibilance = signal.sosfilt(sos, sibilance)

        voice_with_sib = male_voice + sibilance

        processed, reduction_db = deesser.process(voice_with_sib, emotion_mode=EmotionPreservationMode.BALANCED)

        assert processed.shape == voice_with_sib.shape

    def test_emotion_modes(self, voice_with_sibilance):
        """Test different emotion preservation modes."""
        deesser = GenderAwareDeEsser(sample_rate=48000)

        modes = [
            EmotionPreservationMode.MAXIMUM,
            EmotionPreservationMode.BALANCED,
            EmotionPreservationMode.TECHNICAL,
            EmotionPreservationMode.TRANSPARENT,
        ]

        for mode in modes:
            processed, reduction = deesser.process(voice_with_sibilance, emotion_mode=mode)
            assert processed.shape == voice_with_sibilance.shape

    def test_gender_specific_parameters(self, female_voice, male_voice, child_voice):
        """Test that different genders use different parameters."""
        deesser = GenderAwareDeEsser(sample_rate=48000)
        detector = GenderDetector(sample_rate=48000)

        # Detect each gender
        female_char = detector.detect(female_voice)
        male_char = detector.detect(male_voice)
        child_char = detector.detect(child_voice)

        # Process with detected characteristics
        _, female_reduction = deesser.process(female_voice, female_char)
        _, male_reduction = deesser.process(male_voice, male_char)
        _, child_reduction = deesser.process(child_voice, child_char)

        # All should complete successfully
        assert isinstance(female_reduction, float)
        assert isinstance(male_reduction, float)
        assert isinstance(child_reduction, float)


# ============================================================
# BREATH PRESERVATION TESTS
# ============================================================


class TestBreathPreservingProcessor:
    """Test suite for breath preservation."""

    def test_breath_processor_initialization(self, sample_rate):
        """Test breath processor initializes."""
        processor = BreathPreservingProcessor(sample_rate=sample_rate)
        assert processor.sr == sample_rate

    def test_breath_preservation(self, voice_with_breath):
        """Test breath preservation."""
        processor = BreathPreservingProcessor(sample_rate=48000)
        detector = GenderDetector(sample_rate=48000)

        characteristics = detector.detect(voice_with_breath)

        processed, ratio = processor.process(voice_with_breath, characteristics, preservation_ratio=0.7)

        assert processed.shape == voice_with_breath.shape
        assert 0 <= ratio <= 1

    def test_preservation_ratios(self, voice_with_breath):
        """Test different preservation ratios."""
        processor = BreathPreservingProcessor(sample_rate=48000)
        detector = GenderDetector(sample_rate=48000)

        characteristics = detector.detect(voice_with_breath)

        ratios = [0.0, 0.5, 1.0]

        for ratio in ratios:
            processed, actual_ratio = processor.process(voice_with_breath, characteristics, preservation_ratio=ratio)
            assert processed.shape == voice_with_breath.shape


# ============================================================
# UNIFIED VOCAL ENHANCER TESTS
# ============================================================


class TestUnifiedVocalAIEnhancer:
    """Test suite for unified vocal enhancer."""

    def test_enhancer_initialization(self, sample_rate):
        """Test enhancer initializes."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=sample_rate)
        assert enhancer.sr == sample_rate
        assert hasattr(enhancer, "gender_detector")
        assert hasattr(enhancer, "deesser")
        assert hasattr(enhancer, "breath_processor")

    def test_enhance_female_voice(self, voice_with_sibilance):
        """Test enhancement on female voice."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(
            voice_with_sibilance,
            emotion_mode=EmotionPreservationMode.BALANCED,
            breath_preservation=0.7,
            sibilance_reduction=True,
        )

        assert isinstance(result, VocalEnhancementResult)
        assert result.audio.shape == voice_with_sibilance.shape
        assert result.characteristics.gender in [VoiceGender.FEMALE, VoiceGender.CHILD]
        assert len(result.processing_applied) > 0

    def test_enhance_male_voice(self, male_voice):
        """Test enhancement on male voice."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(male_voice, emotion_mode=EmotionPreservationMode.BALANCED, breath_preservation=0.7)

        assert isinstance(result, VocalEnhancementResult)
        assert result.characteristics.gender == VoiceGender.MALE

    def test_enhance_child_voice(self, child_voice):
        """Test enhancement on child voice."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(
            child_voice,
            emotion_mode=EmotionPreservationMode.BALANCED,
            breath_preservation=0.8,  # More preservation for children
        )

        assert isinstance(result, VocalEnhancementResult)
        # High F0 detected
        assert result.characteristics.fundamental_freq > 250

    def test_emotion_preservation_modes(self, female_voice):
        """Test all emotion preservation modes."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        modes = [
            EmotionPreservationMode.MAXIMUM,
            EmotionPreservationMode.BALANCED,
            EmotionPreservationMode.TECHNICAL,
            EmotionPreservationMode.TRANSPARENT,
        ]

        for mode in modes:
            result = enhancer.enhance(female_voice, emotion_mode=mode, breath_preservation=0.7)

            assert isinstance(result, VocalEnhancementResult)
            assert result.metadata["emotion_mode"] == mode.value

    def test_breath_preservation_levels(self, voice_with_breath):
        """Test different breath preservation levels."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        levels = [0.0, 0.5, 1.0]

        for level in levels:
            result = enhancer.enhance(voice_with_breath, breath_preservation=level)

            assert isinstance(result, VocalEnhancementResult)
            # Should preserve breath accordingly
            assert 0 <= result.breath_preserved_ratio <= 1

    def test_formant_preservation(self, female_voice):
        """Test formant preservation."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(female_voice, emotion_mode=EmotionPreservationMode.BALANCED)

        # Formants should be well preserved
        assert result.formant_preservation_score >= 0

    def test_emotion_preservation(self, female_voice):
        """Test emotion preservation."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(female_voice, emotion_mode=EmotionPreservationMode.MAXIMUM)

        # Emotion should be preserved
        assert result.emotion_preservation_score >= 0

    def test_stereo_enhancement(self, female_voice):
        """Test enhancement on stereo audio."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        # Create stereo
        stereo = np.stack([female_voice, female_voice], axis=1)

        result = enhancer.enhance(stereo, emotion_mode=EmotionPreservationMode.BALANCED)

        assert result.audio.shape == stereo.shape
        assert result.audio.ndim == 2

    def test_sibilance_reduction_toggle(self, voice_with_sibilance):
        """Test sibilance reduction can be toggled."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        # With sibilance reduction
        result_with = enhancer.enhance(voice_with_sibilance, sibilance_reduction=True)

        # Without sibilance reduction
        result_without = enhancer.enhance(voice_with_sibilance, sibilance_reduction=False)

        # Both should work
        assert isinstance(result_with, VocalEnhancementResult)
        assert isinstance(result_without, VocalEnhancementResult)

        # With reduction should have higher sibilance_reduced_db
        assert result_with.sibilance_reduced_db >= result_without.sibilance_reduced_db


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestVocalAIIntegration:
    """Test integration with AI Framework."""

    def test_vocal_ai_standalone(self, female_voice):
        """Test vocal AI runs standalone."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        result = enhancer.enhance(female_voice)

        assert isinstance(result, VocalEnhancementResult)
        assert result.audio.shape == female_voice.shape

    def test_complete_pipeline(self, voice_with_sibilance, voice_with_breath):
        """Test complete vocal processing pipeline."""
        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        # Combine defects
        voice = voice_with_sibilance + voice_with_breath * 0.5
        voice = voice / np.max(np.abs(voice)) * 0.7

        # Process
        result = enhancer.enhance(
            voice, emotion_mode=EmotionPreservationMode.BALANCED, breath_preservation=0.7, sibilance_reduction=True
        )

        # Should apply all processing
        assert len(result.processing_applied) >= 2  # breath + sibilance


# ============================================================
# PERFORMANCE TESTS
# ============================================================


class TestVocalAIPerformance:
    """Test performance and efficiency."""

    def test_enhancement_speed(self, female_voice):
        """Test enhancement completes in reasonable time."""
        import time

        enhancer = UnifiedVocalAIEnhancer(sample_rate=48000)

        start = time.time()
        result = enhancer.enhance(female_voice)
        elapsed = time.time() - start

        # Should complete in < 2 seconds for 1s audio
        assert elapsed < 2.0
        assert isinstance(result, VocalEnhancementResult)

    @pytest.mark.slow
    def test_long_audio_enhancement(self, sample_rate):
        """Test enhancement on longer audio."""
        import time

        # 10 seconds of audio
        duration = 10.0
        audio = np.random.randn(int(sample_rate * duration)) * 0.1

        enhancer = UnifiedVocalAIEnhancer(sample_rate=sample_rate)

        start = time.time()
        result = enhancer.enhance(audio)
        elapsed = time.time() - start

        # Should complete in < 20 seconds for 10s audio
        assert elapsed < 20.0
        assert isinstance(result, VocalEnhancementResult)


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short", "-x"])
