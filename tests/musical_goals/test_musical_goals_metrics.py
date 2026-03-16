"""
AURIK v8 Musical Goals Metrics - Automated Test Suite
======================================================

Comprehensive test suite for all 7 musical goals metrics:
1. Bass-Kraft (20-250 Hz)
2. Brillanz (8-20 kHz)
3. Wärme (200-2000 Hz)
4. Natürlichkeit (Spectral properties)
5. Authentizität (Voice/Spectral fingerprint)
6. Emotionalität (Dynamics)
7. Transparenz (Clarity/Separation)

Test Categories:
- Unit Tests: Individual metric correctness
- Range Tests: Score bounds (0.0-1.0)
- Stability Tests: Consistent results
- Regression Tests: Prevent degradation
- Golden Sample Tests: Real-world validation

Quelle: Finalisierungs_Roadmap.md - Component 0.9.1
Autor: AI Team
Datum: 8. Februar 2026
"""

import numpy as np
import pytest

from backend.core.musical_goals import (
    AuthentizitaetMetric,
    BassKraftMetric,
    BrillanzMetric,
    EmotionalitaetMetric,
    MusicalGoalsChecker,
    NatuerlichkeitMetric,
    TransparenzMetric,
    WaermeMetric,
)


class TestBassKraftMetric:
    """Test suite for Bass-Kraft metric (20-250 Hz)."""

    @pytest.fixture
    def metric(self):
        return BassKraftMetric(threshold=0.85)

    @pytest.fixture
    def bass_heavy_audio(self):
        """Audio with strong bass (100 Hz)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Heavy bass at 100 Hz
        audio = 0.8 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
        return audio, sr

    @pytest.fixture
    def bass_light_audio(self):
        """Audio with weak bass (mostly high frequencies)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Mostly high frequencies
        audio = 0.1 * np.sin(2 * np.pi * 100 * t) + 0.9 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_bass_kraft_score_range(self, metric, bass_heavy_audio):
        """Test that bass kraft score is in valid range [0.0, 1.0]."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bass_heavy_high_score(self, metric, bass_heavy_audio):
        """Test that bass-heavy audio gets high score."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Bass-heavy audio should score >0.7, got {score}"

    def test_bass_light_low_score(self, metric, bass_light_audio):
        """Test that bass-light audio gets low score."""
        audio, sr = bass_light_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Bass-light audio should score <0.5, got {score}"

    def test_bass_preservation_check(self, metric, bass_heavy_audio):
        """Test bass preservation check."""
        audio, sr = bass_heavy_audio
        # Simulate processing that reduces bass
        processed = audio * np.array([0.5 if i < len(audio) // 4 else 1.0 for i in range(len(audio))])

        passed, loss, details = metric.check_preservation(audio, processed, sr)
        assert 0.0 <= loss <= 1.0, "Loss should be in [0.0, 1.0]"
        assert "original_score" in details
        assert "processed_score" in details

    def test_measurement_stability(self, metric, bass_heavy_audio):
        """Test that multiple measurements are consistent."""
        audio, sr = bass_heavy_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestBrillanzMetric:
    """Test suite for Brillanz metric (8-20 kHz)."""

    @pytest.fixture
    def metric(self):
        return BrillanzMetric(threshold=0.85)

    @pytest.fixture
    def bright_audio(self):
        """Audio with strong high frequencies."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.8 * np.sin(2 * np.pi * 10000 * t)
        return audio, sr

    @pytest.fixture
    def dull_audio(self):
        """Audio with weak high frequencies."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.9 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 10000 * t)
        return audio, sr

    def test_brillanz_score_range(self, metric, bright_audio):
        """Test that brillanz score is in valid range."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bright_audio_high_score(self, metric, bright_audio):
        """Test that bright audio gets high score."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Bright audio should score >0.6, got {score}"

    def test_dull_audio_low_score(self, metric, dull_audio):
        """Test that dull audio gets low score."""
        audio, sr = dull_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Dull audio should score <0.5, got {score}"

    def test_measurement_stability(self, metric, bright_audio):
        """Test measurement consistency."""
        audio, sr = bright_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestWaermeMetric:
    """Test suite for Wärme metric (200-2000 Hz)."""

    @pytest.fixture
    def metric(self):
        return WaermeMetric(threshold=0.80)

    @pytest.fixture
    def warm_audio(self):
        """Audio with strong mid-range (warm)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_waerme_score_range(self, metric, warm_audio):
        """Test that wärme score is in valid range."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_warm_audio_high_score(self, metric, warm_audio):
        """Test that warm audio gets high score."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Warm audio should score >0.6, got {score}"


class TestNatuerlichkeitMetric:
    """Test suite for Natürlichkeit metric."""

    @pytest.fixture
    def metric(self):
        return NatuerlichkeitMetric(threshold=0.90)

    @pytest.fixture
    def natural_audio(self):
        """Natural audio with harmonics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Fundamental + harmonics (natural sound)
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t)
        )
        return audio, sr

    @pytest.fixture
    def unnatural_audio(self):
        """Unnatural audio (white noise)."""
        sr = 48000
        audio = np.random.randn(sr)
        return audio, sr

    def test_natuerlichkeit_score_range(self, metric, natural_audio):
        """Test that natürlichkeit score is in valid range."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_natural_audio_high_score(self, metric, natural_audio):
        """Test that natural audio gets high score."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Natural audio should score >0.7, got {score}"

    def test_unnatural_audio_low_score(self, metric, unnatural_audio):
        """Test that unnatural audio gets low score."""
        audio, sr = unnatural_audio
        score = metric.measure(audio, sr)
        assert score < 0.6, f"Unnatural audio should score <0.6, got {score}"


class TestAuthentizitaetMetric:
    """Test suite for Authentizität metric."""

    @pytest.fixture
    def metric(self):
        return AuthentizitaetMetric(threshold=0.88)

    @pytest.fixture
    def test_audio(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = np.sin(2 * np.pi * 440 * t)
        return audio, sr

    def test_authentizitaet_score_range(self, metric, test_audio):
        """Test that authentizität score is in valid range."""
        audio, sr = test_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_with_reference_audio(self, metric, test_audio):
        """Test authentizität with reference audio."""
        audio, sr = test_audio
        reference = audio.copy()
        score = metric.measure(audio, sr, reference=reference)
        # Same audio should have high authenticity (>= 0.75 wegen möglichem DSP-Fallback ohne skimage)
        assert score >= 0.75, f"Identical audio should score >=0.75, got {score}"


class TestEmotionalitaetMetric:
    """Test suite for Emotionalität metric."""

    @pytest.fixture
    def metric(self):
        return EmotionalitaetMetric(threshold=0.87)

    @pytest.fixture
    def dynamic_audio(self):
        """Audio with high dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Varying amplitude (emotional)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        audio = envelope * np.sin(2 * np.pi * 440 * t)
        return audio, sr

    @pytest.fixture
    def flat_audio(self):
        """Audio with low dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Constant amplitude
        return audio, sr

    def test_emotionalitaet_score_range(self, metric, dynamic_audio):
        """Test that emotionalität score is in valid range."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_dynamic_audio_high_score(self, metric, dynamic_audio):
        """Test that dynamic audio gets high score."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert score > 0.5, f"Dynamic audio should score >0.5, got {score}"

    def test_flat_audio_low_score(self, metric, flat_audio):
        """Test that flat audio gets low score."""
        audio, sr = flat_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Flat audio should score <0.5, got {score}"


class TestTransparenzMetric:
    """Test suite for Transparenz metric."""

    @pytest.fixture
    def metric(self):
        return TransparenzMetric(threshold=0.89)

    @pytest.fixture
    def clear_audio(self):
        """Clear audio with good separation."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Clear frequencies, well-separated
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_transparenz_score_range(self, metric, clear_audio):
        """Test that transparenz score is in valid range."""
        audio, sr = clear_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


class TestMusicalGoalsChecker:
    """Integration tests for MusicalGoalsChecker."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    @pytest.fixture
    def test_audio(self):
        """Multi-frequency test audio."""
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_measure_all_returns_all_goals(self, checker, test_audio):
        """Test that measure_all returns all 14 goals (v9.9.9 Spec)."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        # 14 Musical Goals gemäß Spec §1.2 / §8.1 — deutsche Schlüssel
        expected_goals = {
            "bass_kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "groove",                # v9.9 Groove-Metrik
            "spatial_depth",         # v9.9 Raumtiefe
            "timbre_authentizitaet",  # v9.9 Timbre-Authentizität (deutsch)
            "tonal_center",          # v9.9.5 Tonales Zentrum
            "micro_dynamics",        # v9.9.5 Mikro-Dynamik
            "separation_fidelity",   # v9.9.9 Separation-Treue
            "artikulation",          # v9.9.9 Artikulation
        }
        assert set(scores.keys()) == expected_goals, (
            f"Missing or extra goals. \n"
            f"Got: {sorted(scores.keys())}\n"
            f"Expected: {sorted(expected_goals)}"
        )

    def test_all_scores_in_valid_range(self, checker, test_audio):
        """Test that all scores are in [0.0, 1.0]."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{goal} score {score} out of range"

    def test_check_all_preserved(self, checker, test_audio):
        """Test check_all_preserved with minimal degradation."""
        audio, sr = test_audio

        # Slightly degraded audio (98% of original)
        degraded = audio * 0.98

        passed, violations = checker.check_all_preserved(audio, degraded, sr)
        # Should have some violations but not catastrophic
        assert isinstance(passed, bool)
        assert isinstance(violations, dict)

    def test_measure_single_goal(self, checker, test_audio):
        """Test measuring single goal."""
        audio, sr = test_audio
        result = checker.measure_single("brillanz", audio, sr)

        assert result.goal_name == "brillanz"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.threshold == checker.thresholds["brillanz"]


class TestRegressionPrevention:
    """Regression tests to prevent metric degradation."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    def test_reference_scores_stability(self, checker):
        """Test that reference audio has consistent scores over time."""
        # Reference audio (stored baseline scores)
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )

        # Expected baseline scores — UPDATED v9.10 after formula recalibration
        # (BrillanzMetric ceiling fix, EmotionalitaetMetric dB crest, TransparenzMetric rolloff)
        # Signal: 100+500+2000+8000 Hz tones, amplitudes 0.3/0.3/0.2/0.2
        baseline_scores = {
            "bass_kraft": (0.90, 1.05),  # Bass-heavy signal always near 1.0
            "brillanz": (0.75, 0.92),  # 8000 Hz = ~15% energy → hf_score=1.0, centroid~2180 Hz
            "waerme": (0.90, 1.05),  # Mid-heavy signal always near 1.0
            "natuerlichkeit": (0.89, 1.00),  # Low flatness (pure tones) → high naturalness
            "authentizitaet": (0.63, 0.79),  # No-reference heuristic: moderate for 4-tone mix
            "emotionalitaet": (0.22, 0.32),  # dB crest fix: 4-tone mix crest ~8.9 dB → 0.27
            "transparenz": (0.56, 0.71),  # 75% rolloff, bandwidth ≥ 4000 Hz after fix
        }

        scores = checker.measure_all(audio, sr)

        for goal, (min_score, max_score) in baseline_scores.items():
            assert (
                min_score <= scores[goal] <= max_score
            ), f"Regression detected in {goal}: {scores[goal]} not in [{min_score}, {max_score}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
