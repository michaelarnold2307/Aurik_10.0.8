"""
Test Multi-Language Support for Lyrics-Guided Vocal Enhancement
================================================================

Tests German, English, French, Spanish, and Italian language support.

Author: AURIK Development Team
Date: 11. Februar 2026
"""

import numpy as np
import pytest

from backend.lyrics_guided import (
    LyricsAligner,
    create_integrated_vocal_timeline,
)


class TestMultiLanguageSupport:
    """Test suite for multi-language support."""

    def setup_method(self):
        """Initialize for each test."""
        self.sr = 48000
        self.duration = 3.0
        self.audio = self._create_test_audio()

    def _create_test_audio(self):
        """Create test audio with vocals."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        audio = np.zeros_like(t, dtype=np.float32)

        # Vocal segment
        mask = (t >= 0.5) & (t < 2.5)
        f0 = 200 + 20 * np.sin(2 * np.pi * 3 * t[mask])
        audio[mask] = 0.3 * np.sin(2 * np.pi * f0 * (t[mask] - 0.5))

        # Background noise
        audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)

        return audio

    # ========================================================================
    # ENGLISH LANGUAGE SUPPORT
    # ========================================================================

    def test_english_language_detection(self):
        """Test English language is supported."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="en")

        alignment = aligner.align(self.audio, self.sr, lyrics="Hello world")

        assert alignment.language in ["en", "unknown"]

    def test_english_explicit_language(self):
        """Test explicitly setting English language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Testing", language="en")

        assert alignment.language == "en"

    def test_integrated_timeline_english(self):
        """Test integrated timeline with English."""
        timeline = create_integrated_vocal_timeline(self.audio, self.sr, language="en")

        assert timeline is not None
        assert timeline.total_duration > 0

    # ========================================================================
    # GERMAN LANGUAGE SUPPORT
    # ========================================================================

    def test_german_language_detection(self):
        """Test German language is supported."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="de")

        alignment = aligner.align(self.audio, self.sr, lyrics="Hallo Welt")

        assert alignment.language in ["de", "unknown"]

    def test_german_explicit_language(self):
        """Test explicitly setting German language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Test", language="de")

        assert alignment.language == "de"

    def test_integrated_timeline_german(self):
        """Test integrated timeline with German."""
        timeline = create_integrated_vocal_timeline(self.audio, self.sr, language="de")

        assert timeline is not None
        assert timeline.total_duration > 0

    def test_german_text_processing(self):
        """Test German text (umlauts) is handled correctly."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="de")

        # German text with umlauts
        german_text = "Über den Wölken muss die Freiheit wohl grenzenlos sein"

        alignment = aligner.align(self.audio, self.sr, lyrics=german_text)

        assert alignment.text == german_text
        assert alignment.language == "de"

    # ========================================================================
    # FRENCH LANGUAGE SUPPORT
    # ========================================================================

    def test_french_language_detection(self):
        """Test French language is supported."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="fr")

        alignment = aligner.align(self.audio, self.sr, lyrics="Bonjour le monde")

        assert alignment.language in ["fr", "unknown"]

    def test_french_explicit_language(self):
        """Test explicitly setting French language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Test", language="fr")

        assert alignment.language == "fr"

    def test_integrated_timeline_french(self):
        """Test integrated timeline with French."""
        timeline = create_integrated_vocal_timeline(self.audio, self.sr, language="fr")

        assert timeline is not None
        assert timeline.total_duration > 0

    def test_french_text_processing(self):
        """Test French text (accents) is handled correctly."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="fr")

        # French text with accents
        french_text = "Liberté, égalité, fraternité"

        alignment = aligner.align(self.audio, self.sr, lyrics=french_text)

        assert alignment.text == french_text
        assert alignment.language == "fr"

    # ========================================================================
    # SPANISH LANGUAGE SUPPORT
    # ========================================================================

    def test_spanish_language_detection(self):
        """Test Spanish language is supported."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="es")

        alignment = aligner.align(self.audio, self.sr, lyrics="Hola mundo")

        assert alignment.language in ["es", "unknown"]

    def test_spanish_explicit_language(self):
        """Test explicitly setting Spanish language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Prueba", language="es")

        assert alignment.language == "es"

    def test_integrated_timeline_spanish(self):
        """Test integrated timeline with Spanish."""
        timeline = create_integrated_vocal_timeline(self.audio, self.sr, language="es")

        assert timeline is not None
        assert timeline.total_duration > 0

    def test_spanish_text_processing(self):
        """Test Spanish text (special chars) is handled correctly."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="es")

        # Spanish text with special characters
        spanish_text = "¿Cómo estás? ¡Muy bien!"

        alignment = aligner.align(self.audio, self.sr, lyrics=spanish_text)

        assert alignment.text == spanish_text
        assert alignment.language == "es"

    # ========================================================================
    # ITALIAN LANGUAGE SUPPORT
    # ========================================================================

    def test_italian_language_detection(self):
        """Test Italian language is supported."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="it")

        alignment = aligner.align(self.audio, self.sr, lyrics="Ciao mondo")

        assert alignment.language in ["it", "unknown"]

    def test_italian_explicit_language(self):
        """Test explicitly setting Italian language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Test", language="it")

        assert alignment.language == "it"

    def test_integrated_timeline_italian(self):
        """Test integrated timeline with Italian."""
        timeline = create_integrated_vocal_timeline(self.audio, self.sr, language="it")

        assert timeline is not None
        assert timeline.total_duration > 0

    def test_italian_text_processing(self):
        """Test Italian text (accents) is handled correctly."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="it")

        # Italian text with accents
        italian_text = "Perché è così bello"

        alignment = aligner.align(self.audio, self.sr, lyrics=italian_text)

        assert alignment.text == italian_text
        assert alignment.language == "it"

    # ========================================================================
    # MULTI-LANGUAGE SWITCHING
    # ========================================================================

    def test_language_switching(self):
        """Test switching between all 5 languages."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        # English
        alignment_en = aligner.align(self.audio, self.sr, lyrics="Hello", language="en")
        assert alignment_en.language == "en"

        # German
        alignment_de = aligner.align(self.audio, self.sr, lyrics="Hallo", language="de")
        assert alignment_de.language == "de"

        # French
        alignment_fr = aligner.align(self.audio, self.sr, lyrics="Bonjour", language="fr")
        assert alignment_fr.language == "fr"

        # Spanish
        alignment_es = aligner.align(self.audio, self.sr, lyrics="Hola", language="es")
        assert alignment_es.language == "es"

        # Italian
        alignment_it = aligner.align(self.audio, self.sr, lyrics="Ciao", language="it")
        assert alignment_it.language == "it"

    def test_auto_language_detection(self):
        """Test auto language detection (falls back to en)."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Test")

        # Should default to English or unknown
        assert alignment.language in ["en", "unknown"]

    # ========================================================================
    # MFA MODEL AVAILABILITY
    # ========================================================================

    def test_mfa_models_mapping_exists(self):
        """Test MFA models mapping contains all 5 languages."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        assert hasattr(aligner, "MFA_MODELS")
        assert "en" in aligner.MFA_MODELS
        assert "de" in aligner.MFA_MODELS
        assert "fr" in aligner.MFA_MODELS
        assert "es" in aligner.MFA_MODELS
        assert "it" in aligner.MFA_MODELS

    def test_mfa_model_structure(self):
        """Test MFA model structure for each language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        for lang_code in ["en", "de", "fr", "es", "it"]:
            assert lang_code in aligner.MFA_MODELS
            model = aligner.MFA_MODELS[lang_code]

            assert "dictionary" in model
            assert "acoustic" in model
            assert "name" in model
            assert isinstance(model["dictionary"], str)
            assert isinstance(model["acoustic"], str)
            assert isinstance(model["name"], str)

    def test_english_mfa_model_name(self):
        """Test English MFA model name."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        en_model = aligner.MFA_MODELS["en"]
        assert en_model["dictionary"] == "english_us_arpa"
        assert en_model["acoustic"] == "english_us_arpa"
        assert en_model["name"] == "English (US)"

    def test_german_mfa_model_name(self):
        """Test German MFA model name."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        de_model = aligner.MFA_MODELS["de"]
        assert de_model["dictionary"] == "german_mfa"
        assert de_model["acoustic"] == "german_mfa"
        assert de_model["name"] == "German"

    def test_french_mfa_model_name(self):
        """Test French MFA model name."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        fr_model = aligner.MFA_MODELS["fr"]
        assert fr_model["dictionary"] == "french_mfa"
        assert fr_model["acoustic"] == "french_mfa"
        assert fr_model["name"] == "French"

    def test_spanish_mfa_model_name(self):
        """Test Spanish MFA model name."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        es_model = aligner.MFA_MODELS["es"]
        assert es_model["dictionary"] == "spanish_mfa"
        assert es_model["acoustic"] == "spanish_mfa"
        assert es_model["name"] == "Spanish"

    def test_italian_mfa_model_name(self):
        """Test Italian MFA model name."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        it_model = aligner.MFA_MODELS["it"]
        assert it_model["dictionary"] == "italian_mfa"
        assert it_model["acoustic"] == "italian_mfa"
        assert it_model["name"] == "Italian"

    # ========================================================================
    # ENCODING TESTS
    # ========================================================================

    def test_utf8_encoding_german(self):
        """Test UTF-8 encoding for German special characters."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="de")

        # All German special characters
        german_text = "äöüß ÄÖÜ"

        alignment = aligner.align(self.audio, self.sr, lyrics=german_text)

        assert alignment.text == german_text

    def test_utf8_encoding_english(self):
        """Test UTF-8 encoding for English."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="en")

        english_text = "Hello world with special chars: café"

        alignment = aligner.align(self.audio, self.sr, lyrics=english_text)

        assert alignment.text == english_text

    def test_utf8_encoding_french(self):
        """Test UTF-8 encoding for French special characters."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="fr")

        # French special characters
        french_text = "àâäéèêëïîôùûüÿ ÀÂÄÉÈÊËÏÎÔÙÛÜŸ"

        alignment = aligner.align(self.audio, self.sr, lyrics=french_text)

        assert alignment.text == french_text

    def test_utf8_encoding_spanish(self):
        """Test UTF-8 encoding for Spanish special characters."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="es")

        # Spanish special characters
        spanish_text = "áéíóúñü ÁÉÍÓÚÑÜ ¿¡"

        alignment = aligner.align(self.audio, self.sr, lyrics=spanish_text)

        assert alignment.text == spanish_text

    def test_utf8_encoding_italian(self):
        """Test UTF-8 encoding for Italian special characters."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="it")

        # Italian special characters
        italian_text = "àèéìòù ÀÈÉÌÒÙ"

        alignment = aligner.align(self.audio, self.sr, lyrics=italian_text)

        assert alignment.text == italian_text

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    def test_full_workflow_english(self):
        """Test complete workflow with English."""
        timeline = create_integrated_vocal_timeline(
            self.audio, sr=self.sr, aurik_mode="restoration", language="en", provided_lyrics="Hello world testing"
        )

        assert timeline is not None
        assert len(timeline.segments) > 0
        assert timeline.total_duration == pytest.approx(self.duration, rel=0.1)

    def test_full_workflow_german(self):
        """Test complete workflow with German."""
        timeline = create_integrated_vocal_timeline(
            self.audio, sr=self.sr, aurik_mode="restoration", language="de", provided_lyrics="Hallo Welt Test"
        )

        assert timeline is not None
        assert len(timeline.segments) > 0
        assert timeline.total_duration == pytest.approx(self.duration, rel=0.1)

    def test_full_workflow_french(self):
        """Test complete workflow with French."""
        timeline = create_integrated_vocal_timeline(
            self.audio, sr=self.sr, aurik_mode="restoration", language="fr", provided_lyrics="Bonjour monde test"
        )

        assert timeline is not None
        assert len(timeline.segments) > 0
        assert timeline.total_duration == pytest.approx(self.duration, rel=0.1)

    def test_full_workflow_spanish(self):
        """Test complete workflow with Spanish."""
        timeline = create_integrated_vocal_timeline(
            self.audio, sr=self.sr, aurik_mode="restoration", language="es", provided_lyrics="Hola mundo prueba"
        )

        assert timeline is not None
        assert len(timeline.segments) > 0
        assert timeline.total_duration == pytest.approx(self.duration, rel=0.1)

    def test_full_workflow_italian(self):
        """Test complete workflow with Italian."""
        timeline = create_integrated_vocal_timeline(
            self.audio, sr=self.sr, aurik_mode="restoration", language="it", provided_lyrics="Ciao mondo test"
        )

        assert timeline is not None
        assert len(timeline.segments) > 0
        assert timeline.total_duration == pytest.approx(self.duration, rel=0.1)

    def test_processing_consistency_across_languages(self):
        """Test processing is consistent across all languages."""
        # Create timelines for all languages
        timeline_en = create_integrated_vocal_timeline(self.audio, self.sr, language="en")
        timeline_de = create_integrated_vocal_timeline(self.audio, self.sr, language="de")
        timeline_fr = create_integrated_vocal_timeline(self.audio, self.sr, language="fr")
        timeline_es = create_integrated_vocal_timeline(self.audio, self.sr, language="es")
        timeline_it = create_integrated_vocal_timeline(self.audio, self.sr, language="it")

        # Should have similar structure (same audio)
        assert len(timeline_en.segments) == len(timeline_de.segments)
        assert len(timeline_en.segments) == len(timeline_fr.segments)
        assert len(timeline_en.segments) == len(timeline_es.segments)
        assert len(timeline_en.segments) == len(timeline_it.segments)

        # All should have same duration
        assert timeline_en.total_duration == timeline_de.total_duration
        assert timeline_en.total_duration == timeline_fr.total_duration
        assert timeline_en.total_duration == timeline_es.total_duration
        assert timeline_en.total_duration == timeline_it.total_duration

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    def test_unsupported_language_fallback(self):
        """Test fallback for unsupported language."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        # Should not crash, should handle gracefully
        alignment = aligner.align(self.audio, self.sr, lyrics="Test", language="unknown")

        assert alignment is not None

    def test_language_none_defaults_to_english(self):
        """Test None language defaults to English."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False)

        alignment = aligner.align(self.audio, self.sr, lyrics="Test")

        # Should default to 'en' or 'unknown'
        assert alignment.language in ["en", "unknown"]


class TestLanguageSpecificFeatures:
    """Test language-specific features."""

    def setup_method(self):
        """Initialize for each test."""
        self.sr = 48000

    def test_german_phoneme_types(self):
        """Test German-specific phoneme types."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="de")

        # German has specific phonemes (ü, ö, ä, ß)
        # The aligner should handle these
        alignment = aligner.align(np.random.randn(self.sr).astype(np.float32), self.sr, lyrics="über")

        assert alignment is not None

    def test_english_phoneme_types(self):
        """Test English-specific phoneme types."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="en")

        alignment = aligner.align(np.random.randn(self.sr).astype(np.float32), self.sr, lyrics="testing")

        assert alignment is not None

    def test_french_phoneme_types(self):
        """Test French-specific phoneme types."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="fr")

        alignment = aligner.align(np.random.randn(self.sr).astype(np.float32), self.sr, lyrics="français")

        assert alignment is not None

    def test_spanish_phoneme_types(self):
        """Test Spanish-specific phoneme types."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="es")

        alignment = aligner.align(np.random.randn(self.sr).astype(np.float32), self.sr, lyrics="español")

        assert alignment is not None

    def test_italian_phoneme_types(self):
        """Test Italian-specific phoneme types."""
        aligner = LyricsAligner(use_whisper=False, use_mfa=False, language="it")

        alignment = aligner.align(np.random.randn(self.sr).astype(np.float32), self.sr, lyrics="italiano")

        assert alignment is not None

    def test_language_in_timeline_strategy(self):
        """Test language is considered in processing strategy."""
        audio = np.random.randn(self.sr * 2).astype(np.float32) * 0.1

        # Test all 5 languages
        for lang in ["en", "de", "fr", "es", "it"]:
            timeline = create_integrated_vocal_timeline(audio, self.sr, language=lang)

            # All should have restoration strategies
            assert timeline.restoration_strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
