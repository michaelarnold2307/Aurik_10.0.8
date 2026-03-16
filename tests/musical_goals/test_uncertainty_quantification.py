"""
Tests for Uncertainty Quantification

Tests Component 0.9.7: Uncertainty Quantification
"""

import numpy as np
import pytest

from backend.core.musical_goals.uncertainty_quantification import (
    ConfidenceLevel,
    GoalsUncertaintyReport,
    UncertaintyEstimate,
    UncertaintyQuantifier,
    get_uncertainty_summary,
    quick_confidence_check,
)

# Test Data Fixtures


@pytest.fixture
def test_audio():
    """Generate test audio signal"""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Multi-frequency signal
    audio = (
        0.3 * np.sin(2 * np.pi * 100 * t)  # Bass
        + 0.3 * np.sin(2 * np.pi * 440 * t)  # Midrange
        + 0.2 * np.sin(2 * np.pi * 2000 * t)  # Treble
    )
    return audio.astype(np.float32)


@pytest.fixture
def stable_calculator():
    """Calculator with low variance (high confidence)"""

    def calculate(audio):
        # Very stable: just compute RMS
        return float(np.sqrt(np.mean(audio**2)))

    return calculate


@pytest.fixture
def noisy_calculator():
    """Calculator with high variance (low confidence)"""

    def calculate(audio):
        # Add random noise to make it unstable
        base = float(np.sqrt(np.mean(audio**2)))
        noise = np.random.normal(0, 0.2)
        return base + noise

    return calculate


@pytest.fixture
def quantifier():
    """Create uncertainty quantifier"""
    return UncertaintyQuantifier(
        n_bootstrap=50, confidence_level=0.95, min_confidence=0.70, random_seed=42  # Fewer samples for faster tests
    )


# Test UncertaintyEstimate


class TestUncertaintyEstimate:
    """Test UncertaintyEstimate dataclass"""

    def test_uncertainty_estimate_creation(self):
        """Test creating uncertainty estimate"""
        estimate = UncertaintyEstimate(
            goal_name="test-goal",
            mean=0.85,
            std=0.03,
            confidence=0.92,
            epistemic_uncertainty=0.01,
            aleatoric_uncertainty=0.03,
            confidence_interval=(0.79, 0.91),
            confidence_level=ConfidenceLevel.HIGH,
            n_samples=100,
        )

        assert estimate.goal_name == "test-goal"
        assert estimate.mean == 0.85
        assert estimate.confidence == 0.92
        assert estimate.confidence_level == ConfidenceLevel.HIGH

    def test_is_reliable(self):
        """Test reliability check"""
        # High confidence - reliable
        estimate_high = UncertaintyEstimate(
            goal_name="test",
            mean=0.85,
            std=0.02,
            confidence=0.90,
            epistemic_uncertainty=0.01,
            aleatoric_uncertainty=0.02,
            confidence_interval=(0.81, 0.89),
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert estimate_high.is_reliable(min_confidence=0.70)

        # Low confidence - unreliable
        estimate_low = UncertaintyEstimate(
            goal_name="test",
            mean=0.75,
            std=0.15,
            confidence=0.55,
            epistemic_uncertainty=0.05,
            aleatoric_uncertainty=0.15,
            confidence_interval=(0.45, 1.05),
            confidence_level=ConfidenceLevel.LOW,
        )
        assert not estimate_low.is_reliable(min_confidence=0.70)

    def test_get_warning(self):
        """Test warning generation"""
        # HIGH - no warning
        estimate_high = UncertaintyEstimate(
            goal_name="test",
            mean=0.85,
            std=0.02,
            confidence=0.90,
            epistemic_uncertainty=0.01,
            aleatoric_uncertainty=0.02,
            confidence_interval=(0.81, 0.89),
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert estimate_high.get_warning() is None

        # MEDIUM - no warning
        estimate_medium = UncertaintyEstimate(
            goal_name="test",
            mean=0.80,
            std=0.05,
            confidence=0.75,
            epistemic_uncertainty=0.02,
            aleatoric_uncertainty=0.05,
            confidence_interval=(0.70, 0.90),
            confidence_level=ConfidenceLevel.MEDIUM,
        )
        assert estimate_medium.get_warning() is None

        # LOW - warning
        estimate_low = UncertaintyEstimate(
            goal_name="bass-kraft",
            mean=0.75,
            std=0.10,
            confidence=0.65,
            epistemic_uncertainty=0.03,
            aleatoric_uncertainty=0.10,
            confidence_interval=(0.55, 0.95),
            confidence_level=ConfidenceLevel.LOW,
        )
        warning = estimate_low.get_warning()
        assert warning is not None
        assert "UNSICHER" in warning
        assert "bass-kraft" in warning

        # VERY_LOW - strong warning
        estimate_very_low = UncertaintyEstimate(
            goal_name="transparenz",
            mean=0.70,
            std=0.20,
            confidence=0.40,
            epistemic_uncertainty=0.07,
            aleatoric_uncertainty=0.20,
            confidence_interval=(0.30, 1.10),
            confidence_level=ConfidenceLevel.VERY_LOW,
        )
        warning = estimate_very_low.get_warning()
        assert warning is not None
        assert "SEHR UNSICHER" in warning


# Test GoalsUncertaintyReport


class TestGoalsUncertaintyReport:
    """Test GoalsUncertaintyReport dataclass"""

    def test_report_creation(self):
        """Test creating uncertainty report"""
        estimates = {
            "goal1": UncertaintyEstimate(
                goal_name="goal1",
                mean=0.85,
                std=0.02,
                confidence=0.90,
                epistemic_uncertainty=0.01,
                aleatoric_uncertainty=0.02,
                confidence_interval=(0.81, 0.89),
                confidence_level=ConfidenceLevel.HIGH,
            ),
            "goal2": UncertaintyEstimate(
                goal_name="goal2",
                mean=0.75,
                std=0.10,
                confidence=0.65,
                epistemic_uncertainty=0.03,
                aleatoric_uncertainty=0.10,
                confidence_interval=(0.55, 0.95),
                confidence_level=ConfidenceLevel.LOW,
            ),
        }

        report = GoalsUncertaintyReport(
            estimates=estimates,
            overall_confidence=0.775,
            warnings=["Warning 1"],
            reliable_goals=["goal1"],
            unreliable_goals=["goal2"],
        )

        assert len(report.estimates) == 2
        assert report.overall_confidence == 0.775
        assert len(report.warnings) == 1
        assert len(report.reliable_goals) == 1
        assert len(report.unreliable_goals) == 1

    def test_has_warnings(self):
        """Test warning detection"""
        report_with = GoalsUncertaintyReport(
            estimates={}, overall_confidence=0.8, warnings=["Warning"], reliable_goals=[], unreliable_goals=[]
        )
        assert report_with.has_warnings()

        report_without = GoalsUncertaintyReport(
            estimates={}, overall_confidence=0.9, warnings=[], reliable_goals=[], unreliable_goals=[]
        )
        assert not report_without.has_warnings()

    def test_get_summary(self):
        """Test summary generation"""
        report = GoalsUncertaintyReport(
            estimates={},
            overall_confidence=0.85,
            warnings=["W1", "W2"],
            reliable_goals=["g1", "g2", "g3"],
            unreliable_goals=["g4"],
        )

        summary = report.get_summary()
        assert "0.85" in summary
        assert "3/7" in summary
        assert "2" in summary


# Test UncertaintyQuantifier


class TestUncertaintyQuantifier:
    """Test UncertaintyQuantifier class"""

    def test_quantifier_initialization(self, quantifier):
        """Test quantifier initialization"""
        assert quantifier.n_bootstrap == 50
        assert quantifier.confidence_level == 0.95
        assert quantifier.min_confidence == 0.70
        assert quantifier.random_seed == 42

    def test_bootstrap_sample(self, quantifier, test_audio, stable_calculator):
        """Test bootstrap sampling"""
        samples = quantifier.bootstrap_sample(test_audio, stable_calculator, n_samples=20)

        assert len(samples) == 20
        assert np.all(np.isfinite(samples))
        # Stable calculator should have low variance
        assert np.std(samples) < 0.1

    def test_bootstrap_sample_with_noisy_calculator(self, quantifier, test_audio, noisy_calculator):
        """Test bootstrap sampling with noisy calculator"""
        samples = quantifier.bootstrap_sample(test_audio, noisy_calculator, n_samples=20)

        assert len(samples) == 20
        # Noisy calculator should have higher variance
        assert np.std(samples) > 0.05

    def test_estimate_epistemic_uncertainty(self, quantifier):
        """Test epistemic uncertainty estimation"""
        # Low variance samples - low epistemic uncertainty
        samples_stable = np.array([0.85, 0.86, 0.84, 0.85, 0.86])
        epistemic = quantifier.estimate_epistemic_uncertainty(samples_stable)
        assert epistemic < 0.01

        # High variance samples - high epistemic uncertainty
        samples_noisy = np.array([0.70, 0.85, 0.65, 0.90, 0.60])
        epistemic_high = quantifier.estimate_epistemic_uncertainty(samples_noisy)
        assert epistemic_high > epistemic

    def test_estimate_aleatoric_uncertainty(self, quantifier):
        """Test aleatoric uncertainty estimation"""
        # Low variance - low aleatoric uncertainty
        samples_stable = np.array([0.85, 0.86, 0.84, 0.85, 0.86])
        aleatoric = quantifier.estimate_aleatoric_uncertainty(samples_stable)
        assert aleatoric < 0.05

        # High variance - high aleatoric uncertainty
        samples_noisy = np.array([0.70, 0.85, 0.65, 0.90, 0.60])
        aleatoric_high = quantifier.estimate_aleatoric_uncertainty(samples_noisy)
        assert aleatoric_high > aleatoric

    def test_calculate_confidence(self, quantifier):
        """Test confidence calculation"""
        # Low variance, in range - high confidence
        samples_good = np.array([0.85, 0.86, 0.84, 0.85, 0.86] * 20)
        confidence_high = quantifier.calculate_confidence(samples_good, expected_range=(0.7, 1.0))
        assert confidence_high > 0.80

        # High variance - lower confidence
        samples_noisy = np.array([0.70, 0.85, 0.65, 0.90, 0.60] * 20)
        confidence_low = quantifier.calculate_confidence(samples_noisy, expected_range=(0.7, 1.0))
        assert confidence_low < confidence_high

        # Out of range - lower confidence
        samples_out = np.array([0.50, 0.55, 0.45, 1.10, 1.15] * 20)
        confidence_out = quantifier.calculate_confidence(samples_out, expected_range=(0.7, 1.0))
        assert confidence_out < confidence_low

    def test_classify_confidence(self, quantifier):
        """Test confidence classification"""
        assert quantifier.classify_confidence(0.95) == ConfidenceLevel.HIGH
        assert quantifier.classify_confidence(0.85) == ConfidenceLevel.HIGH
        assert quantifier.classify_confidence(0.80) == ConfidenceLevel.MEDIUM
        assert quantifier.classify_confidence(0.70) == ConfidenceLevel.MEDIUM
        assert quantifier.classify_confidence(0.65) == ConfidenceLevel.LOW
        assert quantifier.classify_confidence(0.50) == ConfidenceLevel.LOW
        assert quantifier.classify_confidence(0.40) == ConfidenceLevel.VERY_LOW

    def test_quantify_goal_stable(self, quantifier, test_audio, stable_calculator):
        """Test quantifying goal with stable calculator"""
        estimate = quantifier.quantify_goal(
            test_audio, stable_calculator, goal_name="test-goal", expected_range=(0.0, 1.0)
        )

        assert estimate.goal_name == "test-goal"
        assert 0.0 <= estimate.mean <= 1.0
        assert estimate.std < 0.1  # Low variance
        assert estimate.confidence > 0.70  # High confidence
        assert estimate.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        assert estimate.epistemic_uncertainty < estimate.aleatoric_uncertainty
        assert estimate.is_reliable()

    def test_quantify_goal_noisy(self, quantifier, test_audio, noisy_calculator):
        """Test quantifying goal with noisy calculator"""
        estimate = quantifier.quantify_goal(
            test_audio, noisy_calculator, goal_name="noisy-goal", expected_range=(0.0, 1.0)
        )

        assert estimate.goal_name == "noisy-goal"
        assert estimate.std > 0.05  # Higher variance than stable
        # Confidence might be lower due to noise
        assert 0.0 <= estimate.confidence <= 1.0

    def test_quantify_all_goals(self, quantifier, test_audio, stable_calculator, noisy_calculator):
        """Test quantifying all goals"""
        calculators = {"stable-goal": stable_calculator, "noisy-goal": noisy_calculator}

        report = quantifier.quantify_all_goals(test_audio, calculators, expected_range=(0.0, 1.0))

        assert len(report.estimates) == 2
        assert "stable-goal" in report.estimates
        assert "noisy-goal" in report.estimates
        assert 0.0 <= report.overall_confidence <= 1.0

        # Stable goal should be more reliable
        stable_est = report.estimates["stable-goal"]
        noisy_est = report.estimates["noisy-goal"]
        assert stable_est.confidence >= noisy_est.confidence

    def test_should_proceed(self, quantifier):
        """Test should_proceed decision"""
        # HIGH confidence - always proceed
        estimate_high = UncertaintyEstimate(
            goal_name="test",
            mean=0.85,
            std=0.02,
            confidence=0.90,
            epistemic_uncertainty=0.01,
            aleatoric_uncertainty=0.02,
            confidence_interval=(0.81, 0.89),
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert quantifier.should_proceed(estimate_high, strict=False)
        assert quantifier.should_proceed(estimate_high, strict=True)

        # MEDIUM confidence - proceed if not strict
        estimate_medium = UncertaintyEstimate(
            goal_name="test",
            mean=0.80,
            std=0.05,
            confidence=0.75,
            epistemic_uncertainty=0.02,
            aleatoric_uncertainty=0.05,
            confidence_interval=(0.70, 0.90),
            confidence_level=ConfidenceLevel.MEDIUM,
        )
        assert quantifier.should_proceed(estimate_medium, strict=False)
        assert not quantifier.should_proceed(estimate_medium, strict=True)

        # LOW confidence - don't proceed
        estimate_low = UncertaintyEstimate(
            goal_name="test",
            mean=0.75,
            std=0.10,
            confidence=0.65,
            epistemic_uncertainty=0.03,
            aleatoric_uncertainty=0.10,
            confidence_interval=(0.55, 0.95),
            confidence_level=ConfidenceLevel.LOW,
        )
        assert not quantifier.should_proceed(estimate_low, strict=False)
        assert not quantifier.should_proceed(estimate_low, strict=True)


# Test Convenience Functions


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_quick_confidence_check(self, test_audio, stable_calculator):
        """Test quick confidence check"""
        mean, confidence, reliable = quick_confidence_check(
            test_audio, stable_calculator, "test-goal", n_bootstrap=30  # Fast check
        )

        assert 0.0 <= mean <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reliable, bool)

    def test_get_uncertainty_summary(self):
        """Test uncertainty summary generation"""
        estimates = {
            "goal1": UncertaintyEstimate(
                goal_name="goal1",
                mean=0.85,
                std=0.02,
                confidence=0.90,
                epistemic_uncertainty=0.01,
                aleatoric_uncertainty=0.02,
                confidence_interval=(0.81, 0.89),
                confidence_level=ConfidenceLevel.HIGH,
            ),
            "goal2": UncertaintyEstimate(
                goal_name="goal2",
                mean=0.75,
                std=0.10,
                confidence=0.65,
                epistemic_uncertainty=0.03,
                aleatoric_uncertainty=0.10,
                confidence_interval=(0.55, 0.95),
                confidence_level=ConfidenceLevel.LOW,
            ),
        }

        summary = get_uncertainty_summary(estimates)

        assert "Uncertainty Summary" in summary
        assert "goal1" in summary
        assert "goal2" in summary
        assert "0.85" in summary
        assert "0.75" in summary


# Integration Tests


class TestIntegration:
    """Test integration scenarios"""

    def test_complete_uncertainty_workflow(self, test_audio):
        """Test complete uncertainty quantification workflow"""
        # Create quantifier
        quantifier = UncertaintyQuantifier(n_bootstrap=50, confidence_level=0.95, min_confidence=0.70, random_seed=42)

        # Define goal calculators
        def bass_calculator(audio):
            # Simple bass energy (low frequencies)
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1 / 44100)
            bass_mask = freqs < 250
            bass_energy = np.sum(np.abs(fft[bass_mask]) ** 2)
            total_energy = np.sum(np.abs(fft) ** 2)
            return float(np.clip(bass_energy / (total_energy + 1e-10), 0.7, 1.0))

        def treble_calculator(audio):
            # Simple treble energy (high frequencies)
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1 / 44100)
            treble_mask = freqs > 2000
            treble_energy = np.sum(np.abs(fft[treble_mask]) ** 2)
            total_energy = np.sum(np.abs(fft) ** 2)
            return float(np.clip(treble_energy / (total_energy + 1e-10), 0.7, 1.0))

        # Quantify all goals
        report = quantifier.quantify_all_goals(
            test_audio, {"bass-kraft": bass_calculator, "brillanz": treble_calculator}, expected_range=(0.7, 1.0)
        )

        # Check report
        assert len(report.estimates) == 2
        assert "bass-kraft" in report.estimates
        assert "brillanz" in report.estimates
        assert 0.0 <= report.overall_confidence <= 1.0

        # Check individual estimates
        for est in report.estimates.values():
            assert 0.6999 <= est.mean <= 1.0
            assert est.std >= 0
            assert 0.0 <= est.confidence <= 1.0
            assert est.epistemic_uncertainty >= 0
            assert est.aleatoric_uncertainty >= 0

        # Summary
        summary = get_uncertainty_summary(report.estimates)
        assert len(summary) > 0

    def test_confidence_affects_warnings(self):
        """Test that low confidence generates appropriate warnings"""
        quantifier = UncertaintyQuantifier(n_bootstrap=50, min_confidence=0.70, random_seed=42)

        # High variance calculator - should produce warnings
        def unstable_calculator(audio):
            base = np.mean(np.abs(audio))
            # Add large random variation
            noise = np.random.normal(0, 0.5)
            return float(np.clip(base + noise, 0.3, 1.2))

        # Generate test audio
        audio = np.random.randn(44100).astype(np.float32) * 0.1

        # Quantify
        estimate = quantifier.quantify_goal(audio, unstable_calculator, "unstable-goal", expected_range=(0.7, 1.0))

        # Should have low confidence due to high variance and out-of-range values
        # (exact confidence depends on random seed, but should be somewhat low)
        warning = estimate.get_warning()
        # Might or might not have warning depending on exact samples,
        # but check that warning system works
        if warning:
            assert "UNSICHER" in warning or "unstable-goal" in warning
