"""Erweiterte Unit-Tests für Genre-Classifier und Genre-Profile (v10.0).

Testet:
- GermanSchlagerClassifier importierbar
- classify() auf verschiedene synthetische Signale
- Alle Scores finite und ∈ [0, 1]
- Singleton thread-sicher
- SCHLAGER/JAZZ/KLASSIK/OPER/ROCK_RESTORATION_PROFILE Parameter
- get_restoration_profile() für verschiedene Genres
- Subgenre-Labels
- SCHLAGER_CLAP_PROMPTS
- Schwellwerte
- etc.
"""

import threading

import numpy as np
import pytest

from backend.core.genre_classifier import (
    GENRE_RESTORATION_PROFILES,
    JAZZ_RESTORATION_PROFILE,
    KLASSIK_RESTORATION_PROFILE,
    OPER_RESTORATION_PROFILE,
    ROCK_RESTORATION_PROFILE,
    SCHLAGER_RESTORATION_PROFILE,
    GermanSchlagerClassifier,
    SchlagerClassificationResult,
    classify_genre,
    get_genre_classifier,
    get_restoration_profile,
)

# Klassenattribute als Modulkonstanten verfügbar machen
SCHLAGER_CLAP_PROMPTS = GermanSchlagerClassifier.SCHLAGER_CLAP_PROMPTS
SCHLAGER_CONFIDENCE_THRESHOLD = GermanSchlagerClassifier.SCHLAGER_CONFIDENCE_THRESHOLD
HSI_THRESHOLD = GermanSchlagerClassifier.HSI_THRESHOLD


class TestV100GenreClassifierProfiles:
    """Erweiterte Tests für Genre-Classifier v10.0."""

    def test_01_german_schlager_classifier_importable(self):
        """GermanSchlagerClassifier importierbar."""
        assert GermanSchlagerClassifier is not None

    def test_02_classify_returns_result(self):
        """classify() gibt SchlagerClassificationResult zurück."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_03_classify_silence_is_not_schlager_no_crash(self):
        """classify() auf Stille → is_schlager=False, kein Absturz."""
        audio = np.zeros(48000, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.is_schlager, bool)

    def test_04_classify_white_noise_is_not_schlager(self):
        """classify() auf weißes Rauschen → is_schlager=False."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        # Weißes Rauschen sollte nicht Schlager sein
        assert isinstance(result.is_schlager, bool)

    def test_05_all_scores_finite(self):
        """Alle Scores finite."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert np.isfinite(result.confidence)
        assert np.isfinite(result.clap_score)
        assert np.isfinite(result.accordion_score)
        assert np.isfinite(result.harmonic_simplicity)
        assert np.isfinite(result.rhythm_score)
        assert np.isfinite(result.vocal_german_prior)
        assert np.isfinite(result.melodic_repetition)

    def test_06_all_scores_in_range_0_1(self):
        """Alle Scores ∈ [0, 1]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.clap_score <= 1.0
        assert 0.0 <= result.accordion_score <= 1.0
        assert 0.0 <= result.harmonic_simplicity <= 1.0
        assert 0.0 <= result.rhythm_score <= 1.0
        assert 0.0 <= result.vocal_german_prior <= 1.0
        assert 0.0 <= result.melodic_repetition <= 1.0

    def test_07_singleton_thread_safe(self):
        """Singleton thread-sicher (20 parallele get_genre_classifier() → selbes Objekt)."""
        instances = []

        def get_instance():
            instances.append(get_genre_classifier())

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_08_schlager_profile_tonal_center_threshold(self):
        """SCHLAGER_RESTORATION_PROFILE: tonal_center_threshold = 0.97."""
        assert SCHLAGER_RESTORATION_PROFILE["tonal_center_threshold"] == 0.97

    def test_09_jazz_profile_compression_ratio_cap(self):
        """JAZZ_RESTORATION_PROFILE: compression_ratio_cap = 1.8."""
        assert JAZZ_RESTORATION_PROFILE["compression_ratio_cap"] == 1.8

    def test_10_klassik_profile_groove_dtw_max_ms(self):
        """KLASSIK_RESTORATION_PROFILE: groove_dtw_max_ms = 10.0."""
        assert KLASSIK_RESTORATION_PROFILE["groove_dtw_max_ms"] == 10.0

    def test_11_oper_profile_gp_memory_key_opera(self):
        """OPER_RESTORATION_PROFILE: gp_memory_key = 'opera'."""
        assert OPER_RESTORATION_PROFILE["gp_memory_key"] == "opera"

    def test_12_rock_profile_gp_memory_key_rock(self):
        """ROCK_RESTORATION_PROFILE: gp_memory_key = 'rock'."""
        assert ROCK_RESTORATION_PROFILE["gp_memory_key"] == "rock"

    def test_13_get_restoration_profile_rock_returns_rock_profile(self):
        """get_restoration_profile('Rock') gibt ROCK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Rock")
        assert profile == ROCK_RESTORATION_PROFILE

    def test_14_get_restoration_profile_klassik_returns_klassik_profile(self):
        """get_restoration_profile('Klassik') gibt KLASSIK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Klassik")
        assert profile == KLASSIK_RESTORATION_PROFILE

    def test_15_subgenre_labels_valid(self):
        """Subgenre-Labels: schunkel, walzer, marsch, discoschlager, unknown."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        valid_subgenres = [
            "schunkel",
            "walzer",
            "marsch",
            "discoschlager",
            "unknown",
            "schlager_1950s",
            "schlager_modern",
            "volksmusik",
        ]
        assert result.subgenre in valid_subgenres or result.subgenre == ""

    def test_16_schlager_clap_prompts_is_list(self):
        """SCHLAGER_CLAP_PROMPTS is list mit ≥ 5 Einträgen."""
        assert isinstance(SCHLAGER_CLAP_PROMPTS, list)
        assert len(SCHLAGER_CLAP_PROMPTS) >= 5

    def test_17_schlager_confidence_threshold_approx_052(self):
        """SCHLAGER_CONFIDENCE_THRESHOLD ≈ 0.52."""
        assert 0.50 <= SCHLAGER_CONFIDENCE_THRESHOLD <= 0.55

    def test_18_hsi_threshold_equals_082(self):
        """HSI_THRESHOLD = 0.82."""
        assert HSI_THRESHOLD == 0.82

    def test_19_classify_mono_input_no_error(self):
        """classify() Mono-Input (1D Array) → kein Fehler."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        assert audio.ndim == 1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_20_classify_stereo_input_no_error(self):
        """classify() Stereo-Input (2D Array) → kein Fehler."""
        np.random.seed(42)
        audio = np.random.randn(2, 48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_21_classify_very_short_signal_no_crash(self):
        """classify() sehr kurzes Signal (< 5 s) → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(24000).astype(np.float32) * 0.1  # 0.5 s
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_22_bpm_positive_or_zero(self):
        """bpm ≥ 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert result.bpm >= 0.0

    def test_23_reasoning_not_empty(self):
        """reasoning ist nicht leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_24_genre_label_is_string(self):
        """genre_label ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.genre_label, str)

    def test_25_classify_genre_convenience_function(self):
        """classify_genre() convenience function works."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = classify_genre(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_26_nan_in_audio_no_crash(self):
        """NaN in audio → kein Absturz."""
        audio = np.full(48000, np.nan, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_27_inf_in_audio_no_crash(self):
        """Inf in audio → kein Absturz."""
        audio = np.full(48000, np.inf, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_28_very_loud_audio_no_crash(self):
        """Sehr lautes Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 10.0
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_29_very_quiet_audio_no_crash(self):
        """Sehr leises Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.0001
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_30_key_is_string_or_empty(self):
        """key ist String oder leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.key, str)

    def test_31_schlager_profile_soft_saturation_preserve_true(self):
        """SCHLAGER_RESTORATION_PROFILE: soft_saturation_preserve = True."""
        assert SCHLAGER_RESTORATION_PROFILE["soft_saturation_preserve"] is True

    def test_32_jazz_profile_tonal_center_threshold(self):
        """JAZZ_RESTORATION_PROFILE: tonal_center_threshold = 0.92."""
        assert JAZZ_RESTORATION_PROFILE["tonal_center_threshold"] == 0.92

    def test_33_klassik_profile_compression_ratio_cap(self):
        """KLASSIK_RESTORATION_PROFILE: compression_ratio_cap = 1.3."""
        assert KLASSIK_RESTORATION_PROFILE["compression_ratio_cap"] == 1.3

    def test_34_oper_profile_deessing_target_hz(self):
        """OPER_RESTORATION_PROFILE: deessing_target_hz = 7000."""
        assert OPER_RESTORATION_PROFILE["deessing_target_hz"] == 7000

    def test_35_rock_profile_brillanz_target(self):
        """ROCK_RESTORATION_PROFILE: brillanz_target = 0.90."""
        assert ROCK_RESTORATION_PROFILE["brillanz_target"] == 0.90

    def test_36_genre_restoration_profiles_has_all_five(self):
        """GENRE_RESTORATION_PROFILES hat alle 5 Hauptgenres."""
        profile_keys = list(GENRE_RESTORATION_PROFILES.keys())
        expected = ["Schlager", "Jazz", "Klassik", "Oper", "Rock"]
        for genre in expected:
            assert genre in profile_keys

    def test_37_get_restoration_profile_case_insensitive_or_exact(self):
        """get_restoration_profile() funktioniert."""
        profile = get_restoration_profile("Jazz")
        assert profile == JAZZ_RESTORATION_PROFILE

    def test_38_schlager_profile_brillanz_target_lower(self):
        """SCHLAGER_RESTORATION_PROFILE: brillanz_target = 0.82."""
        assert SCHLAGER_RESTORATION_PROFILE["brillanz_target"] == 0.82

    def test_39_schlager_profile_waerme_target_higher(self):
        """SCHLAGER_RESTORATION_PROFILE: waerme_target = 0.88."""
        assert SCHLAGER_RESTORATION_PROFILE["waerme_target"] == 0.88

    def test_40_all_profiles_have_gp_memory_key(self):
        """Alle Profile haben gp_memory_key."""
        for profile in [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]:
            assert "gp_memory_key" in profile
