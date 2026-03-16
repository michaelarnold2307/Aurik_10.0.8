"""
Tests für Multi-Pass Processing Strategy & Adaptive Selection

Test Coverage:
1. ProcessingVariant creation und variants generation
2. ObjectiveScorer mit verschiedenen Metriken
3. ConfidenceCalculator für top-2 variance
4. MultiPassEngine End-to-End
5. Integration mit verschiedenen ProcessingModes

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import numpy as np
import pytest

from backend.core.multi_pass_strategy import (
    ConfidenceCalculator,
    MultiPassEngine,
    ObjectiveScore,
    ObjectiveScorer,
    ProcessingVariant,
    VariantStrategy,
    create_default_variants,
)
from backend.core.processing_modes import ProcessingMode


class TestProcessingVariant:
    """Test ProcessingVariant creation und Konfiguration."""

    def test_create_conservative(self):
        """Test conservative variant creation."""
        variant = ProcessingVariant.create_conservative(ProcessingMode.RESTORATION)

        assert variant.name == "conservative"
        assert variant.strategy == VariantStrategy.CONSERVATIVE
        assert variant.config.denoise_strength < 0.5  # Ultra-conservative
        assert variant.config.preserve_breaths is True
        assert variant.config.preserve_room_tone is True
        assert variant.weight == 1.0

    def test_create_balanced(self):
        """Test balanced variant creation."""
        variant = ProcessingVariant.create_balanced(ProcessingMode.RESTORATION)

        assert variant.name == "balanced"
        assert variant.strategy == VariantStrategy.BALANCED
        assert variant.config is not None
        assert 0.0 <= variant.config.denoise_strength <= 1.0

    def test_create_aggressive(self):
        """Test aggressive variant creation."""
        variant = ProcessingVariant.create_aggressive(ProcessingMode.STUDIO_2026)

        assert variant.name == "aggressive"
        assert variant.strategy == VariantStrategy.AGGRESSIVE
        # Aggressive sollte höhere werte haben
        assert variant.config.enhancement_strength > 0.5

    def test_create_gentle_denoise(self):
        """Test gentle denoise variant."""
        variant = ProcessingVariant.create_gentle_denoise()

        assert variant.name == "gentle_denoise"
        assert variant.strategy == VariantStrategy.GENTLE_DENOISE
        assert variant.config.denoise_strength < 0.2  # Ultra-gentle
        assert variant.config.preserve_analog_character is True

    def test_create_strong_dynamics(self):
        """Test strong dynamics variant."""
        variant = ProcessingVariant.create_strong_dynamics()

        assert variant.name == "strong_dynamics"
        assert variant.strategy == VariantStrategy.STRONG_DYNAMICS
        assert variant.config.compression_ratio > 4.0  # Strong compression
        assert variant.config.target_lufs <= -14.0  # Streaming standard

    def test_variant_to_dict(self):
        """Test variant serialization."""
        variant = ProcessingVariant.create_balanced()
        variant_dict = variant.to_dict()

        assert "name" in variant_dict
        assert "strategy" in variant_dict
        assert "config" in variant_dict
        assert variant_dict["strategy"] == "balanced"

    def test_variant_different_parameters(self):
        """Test dass verschiedene Varianten verschiedene Parameter haben."""
        conservative = ProcessingVariant.create_conservative()
        aggressive = ProcessingVariant.create_aggressive()

        # Conservative sollte schwächere Parameter haben
        assert conservative.config.denoise_strength < aggressive.config.denoise_strength
        assert conservative.config.compression_ratio < aggressive.config.compression_ratio


class TestObjectiveScore:
    """Test ObjectiveScore dataclass."""

    def test_objective_score_creation(self):
        """Test ObjectiveScore creation."""
        score = ObjectiveScore(
            cdpam_score=0.25,
            dnsmos_score=3.8,
            musical_goals_avg=0.75,
            snr_db=25.0,
            thd_percent=0.5,
            composite_score=0.82,
            confidence=0.9,
            variant_name="test_variant",
        )

        assert score.variant_name == "test_variant"
        assert score.composite_score == 0.82
        assert score.confidence == 0.9

    def test_objective_score_to_dict(self):
        """Test ObjectiveScore serialization."""
        score = ObjectiveScore(variant_name="test", composite_score=0.75)

        score_dict = score.to_dict()
        assert "variant_name" in score_dict
        assert "composite_score" in score_dict
        assert score_dict["variant_name"] == "test"

    def test_objective_score_str(self):
        """Test ObjectiveScore string representation."""
        score = ObjectiveScore(
            variant_name="balanced", composite_score=0.80, confidence=0.85, cdpam_score=0.28, dnsmos_score=3.9
        )

        score_str = str(score)
        assert "balanced" in score_str
        assert "0.800" in score_str  # composite
        assert "0.85" in score_str  # confidence


class TestObjectiveScorer:
    """Test ObjectiveScorer mit mock audio."""

    @pytest.fixture
    def mock_audio(self):
        """Generate mock audio signal."""
        duration = 2.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simple sine wave + noise
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        audio += 0.05 * np.random.randn(len(t))  # Small noise

        return audio, sample_rate

    def test_scorer_initialization(self):
        """Test ObjectiveScorer initialization.

        §10.2/§4.4: DNSMOS ist für Musik VERBOTEN — per Spec immer enable_dnsmos=False.
        VERSA (non-reference MOS) und Musical Goals sind die primären Musik-Qualitätsmetriken.
        """
        # Standard-Konfiguration: VERSA + Musical Goals (§4.4: VERSA ersetzt CDPAM)
        scorer = ObjectiveScorer(enable_cdpam=True, enable_dnsmos=False, enable_musical_goals=True)

        assert scorer.enable_cdpam is True
        assert scorer.enable_dnsmos is False  # §10.2: DNSMOS VERBOTEN für Musik
        assert scorer.enable_musical_goals is True

    def test_scorer_without_plugins(self):
        """Test ObjectiveScorer ohne externe plugins."""
        # Disable external plugins (für CI ohne VERSA/DNSMOS)
        scorer = ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=True)

        # §4.4: versa_plugin ersetzt cdpam_plugin
        assert scorer.versa_plugin is None
        assert scorer.dnsmos_plugin is None

    def test_score_audio_basic(self, mock_audio):
        """Test basic audio scoring."""
        audio, sr = mock_audio

        # Scorer ohne externe plugins (für robuste tests)
        scorer = ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)

        score = scorer.score(audio=audio, sample_rate=sr, variant_name="test_variant")

        assert score.variant_name == "test_variant"
        assert 0.0 <= score.composite_score <= 1.0

    def test_score_with_reference(self, mock_audio):
        """Test scoring mit reference audio (für CDPAM)."""
        audio, sr = mock_audio
        reference = audio.copy()

        scorer = ObjectiveScorer(
            enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False  # Mock, da CDPAM evtl nicht installiert
        )

        score = scorer.score(audio=audio, sample_rate=sr, variant_name="test", reference_audio=reference)

        assert score is not None

    def test_composite_calculation(self):
        """Test composite score calculation."""
        scorer = ObjectiveScorer()

        # Mock score mit bekannten Werten (§10.2: DNSMOS VERBOTEN, dnsmos_score=0.0)
        test_score = ObjectiveScore(
            cdpam_score=0.2,        # Good (niedrig) — CDPAM als Musik-Wahrnehmungsmetrik (§4.4)
            dnsmos_score=0.0,       # §10.2: DNSMOS VERBOTEN für Musik — kein Beitrag zum Composite
            musical_goals_avg=0.8,  # Good (hoch) — 14 Musical Goals (§1.2)
            snr_db=30.0,            # Good (hoch)
            thd_percent=0.5,        # Good (niedrig)
        )

        composite = scorer._calculate_composite(test_score)

        # Sollte relativ hoch sein da alle Metriken gut sind
        assert composite > 0.7
        assert 0.0 <= composite <= 1.0


class TestConfidenceCalculator:
    """Test ConfidenceCalculator für variance-based confidence."""

    def test_confidence_high_gap(self):
        """Test confidence mit großem Gap zwischen top-2."""
        score1 = ObjectiveScore(composite_score=0.90, variant_name="best")
        score2 = ObjectiveScore(composite_score=0.70, variant_name="second")
        score3 = ObjectiveScore(composite_score=0.65, variant_name="third")

        scores = [score1, score2, score3]

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence(scores, score1)

        # Large gap → high confidence
        assert confidence > 0.7
        assert 0.0 <= confidence <= 1.0

    def test_confidence_small_gap(self):
        """Test confidence mit kleinem Gap."""
        score1 = ObjectiveScore(composite_score=0.75, variant_name="best")
        score2 = ObjectiveScore(composite_score=0.73, variant_name="second")
        score3 = ObjectiveScore(composite_score=0.72, variant_name="third")

        scores = [score1, score2, score3]

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence(scores, score1)

        # Small gap → lower confidence
        assert confidence < 0.8

    def test_confidence_single_variant(self):
        """Test confidence mit nur 1 Variante."""
        score1 = ObjectiveScore(composite_score=0.85, variant_name="only")

        calc = ConfidenceCalculator()
        confidence = calc.calculate_confidence([score1], score1)

        # Not enough data → medium confidence
        assert confidence == 0.5


class TestMultiPassEngine:
    """Test MultiPassEngine End-to-End."""

    @pytest.fixture
    def mock_audio(self):
        """Generate mock audio."""
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio += 0.05 * np.random.randn(len(t))
        return audio, sample_rate

    @pytest.fixture
    def mock_process_func(self):
        """Mock processing function für tests."""

        def process(audio, sample_rate, config):
            # Simplified mock: apply small gain based on config strength
            gain = 1.0 + (config.enhancement_strength * 0.1)
            return audio * gain

        return process

    def test_engine_initialization(self):
        """Test MultiPassEngine initialization."""
        engine = MultiPassEngine()

        assert engine.scorer is not None
        assert engine.confidence_calc is not None

    def test_engine_with_custom_scorer(self):
        """Test engine mit custom scorer."""
        scorer = ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False)
        engine = MultiPassEngine(scorer=scorer)

        assert engine.scorer is scorer

    def test_process_with_3_variants(self, mock_audio, mock_process_func):
        """Test processing mit 3 Varianten."""
        audio, sr = mock_audio

        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        # Validate result structure
        assert "audio" in result
        assert "variant_name" in result
        assert "confidence" in result
        assert "composite_score" in result
        assert "all_scores" in result

        # Validate result values
        assert result["audio"] is not None
        assert len(result["audio"]) > 0
        assert result["variant_name"] in ["conservative", "balanced", "aggressive"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["composite_score"] <= 1.0
        assert len(result["all_scores"]) == 3

    def test_process_with_5_variants(self, mock_audio, mock_process_func):
        """Test processing mit 5 Varianten."""
        audio, sr = mock_audio

        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=5)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        assert len(result["all_scores"]) == 5
        assert len(result["processing_times"]) == 5

    def test_process_no_variants_error(self, mock_audio):
        """Test dass leere Varianten-Liste Error wirft."""
        audio, sr = mock_audio

        engine = MultiPassEngine()

        with pytest.raises(ValueError, match="Mindestens 1 ProcessingVariant"):
            engine.process_with_variants(audio=audio, sample_rate=sr, variants=[])

    def test_process_with_reference(self, mock_audio, mock_process_func):
        """Test processing mit reference audio."""
        audio, sr = mock_audio
        reference = audio.copy()

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False))  # Mock

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, reference_audio=reference, process_func=mock_process_func
        )

        assert result is not None

    def test_scores_sorted_by_composite(self, mock_audio, mock_process_func):
        """Test dass Scores sortiert sind (best first)."""
        audio, sr = mock_audio

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        all_scores = result["all_scores"]

        # Sollte descending sortiert sein
        for i in range(len(all_scores) - 1):
            assert all_scores[i].composite_score >= all_scores[i + 1].composite_score

    def test_processing_times_recorded(self, mock_audio, mock_process_func):
        """Test dass processing times recorded werden."""
        audio, sr = mock_audio

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine()

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=mock_process_func
        )

        times = result["processing_times"]

        assert len(times) == 3
        assert "conservative" in times
        assert "balanced" in times
        assert "aggressive" in times

        # All times should be > 0
        for variant_name, time_sec in times.items():
            assert time_sec > 0.0


class TestCreateDefaultVariants:
    """Test create_default_variants helper function."""

    def test_create_3_variants(self):
        """Test creating 3 variants."""
        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        assert len(variants) == 3
        assert variants[0].name == "conservative"
        assert variants[1].name == "balanced"
        assert variants[2].name == "aggressive"

    def test_create_5_variants(self):
        """Test creating 5 variants."""
        variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=5)

        assert len(variants) == 5
        names = [v.name for v in variants]
        assert "conservative" in names
        assert "gentle_denoise" in names
        assert "balanced" in names
        assert "aggressive" in names
        assert "strong_dynamics" in names

    def test_create_invalid_num_variants(self):
        """Test dass invalid num_variants Error wirft."""
        with pytest.raises(ValueError):
            create_default_variants(num_variants=4)  # Only 3 or 5 supported


class TestIntegrationScenarios:
    """Integration tests für realistic scenarios."""

    @pytest.fixture
    def realistic_audio(self):
        """Generate realistic test audio mit some "defects"."""
        duration = 3.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Music-like signal: fundamental + harmonics
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
        audio += 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
        audio += 0.1 * np.sin(2 * np.pi * 880 * t)  # A5

        # Add "defects"
        audio += 0.08 * np.random.randn(len(t))  # Noise

        # Add some clicks (simplified)
        click_positions = np.random.choice(len(audio), size=10, replace=False)
        audio[click_positions] *= 3.0

        return audio, sample_rate

    def test_realistic_restoration_scenario(self, realistic_audio):
        """Test realistic restoration scenario."""
        audio, sr = realistic_audio

        def simple_denoise_process(audio, sample_rate, config):
            # Simplified denoising: High-pass filter
            from scipy import signal

            strength = config.denoise_strength
            cutoff = 80 * (1 + strength)  # Higher strength → higher cutoff

            sos = signal.butter(3, cutoff, "high", fs=sample_rate, output="sos")

            return signal.sosfilt(sos, audio)

        variants = create_default_variants(num_variants=3)

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(
            audio=audio, sample_rate=sr, variants=variants, process_func=simple_denoise_process
        )

        # Should select one variant
        assert result["variant_name"] is not None
        assert result["confidence"] > 0.0

        # Output should be similar length
        assert len(result["audio"]) == len(audio)

    def test_different_processing_modes(self, realistic_audio):
        """Test variants mit verschiedenen ProcessingModes."""
        audio, sr = realistic_audio

        def mock_process(audio, sr, config):
            return audio * 0.99  # Simplified

        # RESTORATION mode
        variants_restore = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

        # STUDIO_2026 mode
        variants_studio = create_default_variants(base_mode=ProcessingMode.STUDIO_2026, num_variants=3)

        # Configs should differ
        assert variants_restore[0].config.target_lufs != variants_studio[0].config.target_lufs


# === Performance & Edge Cases ===


class TestEdgeCases:
    """Test edge cases und error handling."""

    def test_silent_audio(self):
        """Test mit silent audio."""
        audio = np.zeros(16000 * 2)  # 2 sec silence
        sr = 16000

        variants = create_default_variants(num_variants=3)

        def mock_process(audio, sr, config):
            return audio

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        # Should still work (wählt irgendeine Variante)
        assert result["variant_name"] is not None

    def test_very_short_audio(self):
        """Test mit sehr kurzer audio."""
        audio = np.random.randn(1000)  # ~62ms @ 16kHz
        sr = 16000

        variants = [ProcessingVariant.create_balanced()]

        def mock_process(audio, sr, config):
            return audio

        engine = MultiPassEngine(
            scorer=ObjectiveScorer(enable_cdpam=False, enable_dnsmos=False, enable_musical_goals=False)
        )

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        assert len(result["audio"]) == len(audio)

    def test_clipped_audio(self):
        """Test mit clipped audio."""
        audio = np.clip(np.random.randn(16000 * 2) * 2.0, -1.0, 1.0)
        sr = 16000

        variants = create_default_variants(num_variants=3)

        def mock_process(audio, sr, config):
            return audio * 0.9  # Simple gain reduction

        engine = MultiPassEngine()

        result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants, process_func=mock_process)

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
