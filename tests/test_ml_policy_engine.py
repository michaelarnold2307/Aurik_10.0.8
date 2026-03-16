"""
Tests for ML Policy Engine - Docker Phase 7

Tests:
- Model selection for all 28 Docker plugins
- Context-based routing decisions
- Quality assessment model selection
- Stem separation policy
- Enhancement policy
"""

import pytest

from policy.ml_policy_engine import MLModelPolicyEngine, get_recommended_models


@pytest.fixture
def policy_engine():
    """Create ML Policy Engine instance."""
    return MLModelPolicyEngine()


class TestDenoiseModelSelection:
    """Test denoise model selection logic."""

    def test_vinyl_selects_banquet(self, policy_engine):
        """Vinyl medium should select Banquet (vinyl-specialized)."""
        context = {"detected_medium": "vinyl"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "banquet"

    def test_vocals_selects_resemble_enhance(self, policy_engine):
        """Vocals/Speech should select Resemble Enhance."""
        context = {"has_vocals": True}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "resemble_enhance"

    def test_drums_selects_dccrn(self, policy_engine):
        """Drums/Transients should select DCCRN (transient-preserving)."""
        context = {"has_drums": True, "content_character": "HIGHLY_TRANSIENT"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "dccrn"

    def test_ambient_selects_deepfilternet(self, policy_engine):
        """Ambient content should select DeepFilterNet (aggressive smoothing)."""
        context = {"has_ambient": True, "content_character": "HIGHLY_SUSTAINED"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "deepfilternet"

    def test_processing_strategy_override(self, policy_engine):
        """Processing strategy should override other factors."""
        context = {"processing_strategy": "PRESERVE_TRANSIENTS"}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "dccrn"

    def test_dominant_instrument_mapping(self, policy_engine):
        """Dominant instrument should guide model selection."""
        test_cases = [
            ("DRUMS", "dccrn"),
            ("VOCALS", "resemble_enhance"),
            ("BASS", "deepfilternet"),
            ("STRINGS", "resemble_enhance"),
        ]
        for instrument, expected_model in test_cases:
            context = {"dominant_instrument": instrument}
            model = policy_engine.select_denoise_model(context, {})
            assert model == expected_model, f"Failed for {instrument}"

    def test_fallback_to_resemble_enhance(self, policy_engine):
        """Unknown context should fallback to Resemble Enhance."""
        context = {}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "resemble_enhance"


class TestRepairModelSelection:
    """Test repair/declipping model selection."""

    def test_speech_selects_fullsubnet(self, policy_engine):
        """Speech should select FullSubNet+."""
        context = {"has_vocals": True}
        model = policy_engine.select_repair_model(context, {})
        assert model == "fullsubnet"

    def test_music_selects_dccrn(self, policy_engine):
        """Music should select DCCRN."""
        context = {"has_vocals": False}
        model = policy_engine.select_repair_model(context, {})
        assert model == "dccrn"


class TestStemSeparationSelection:
    """Test stem separation model selection."""

    def test_speech_separation_selects_convtasnet(self, policy_engine):
        """Speech separation should select Conv-TasNet."""
        context = {"has_vocals": True}
        goal = {"num_stems": 2}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == "convtasnet"

    def test_fast_processing_selects_mdx23c(self, policy_engine):
        """Fast processing should select MDX23C."""
        context = {}
        goal = {"quality_level": "fast"}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == "mdx23c"

    def test_ultra_hq_selects_uvr_mdxnet(self, policy_engine):
        """Ultra-HQ should select UVR MDX-Net HQ4."""
        context = {}
        goal = {"quality_level": "ultra"}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == "uvr_mdxnet"

    def test_six_stems_selects_demucs(self, policy_engine):
        """6+ stems should select Demucs v4."""
        context = {}
        goal = {"num_stems": 6}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == "demucs"

    def test_default_selects_mdx23c(self, policy_engine):
        """Standard separation should select MDX23C (SOTA)."""
        context = {}
        goal = {}
        model = policy_engine.select_stem_separation_model(context, goal)
        assert model == "mdx23c"


class TestEnhancementSelection:
    """Test enhancement model selection."""

    def test_speech_enhancement_selects_resemble(self, policy_engine):
        """Speech enhancement should select Resemble Enhance."""
        context = {"has_vocals": True}
        goal = {"enhancement_type": "speech"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == "resemble_enhance"

    def test_super_resolution_selects_audiosr(self, policy_engine):
        """Super-resolution should select AudioSR."""
        context = {}
        goal = {"enhancement_type": "super_resolution"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == "audiosr"

    def test_diffusion_enhancement_selects_wpe(self, policy_engine):
        """Diffusion enhancement should select WPE."""
        context = {}
        goal = {"enhancement_type": "diffusion"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == "wpe"

    def test_general_enhancement_selects_gacela(self, policy_engine):
        """General enhancement should select GACELA."""
        context = {}
        goal = {"enhancement_type": "general"}
        model = policy_engine.select_enhancement_model(context, goal)
        assert model == "gacela"


class TestQualityAssessmentSelection:
    """Test quality assessment model selection (§4.4 Spec: nur musik-geeignete Metriken)."""

    def test_always_includes_cdpam(self, policy_engine):
        """CDPAM ist immer dabei — musik-spezifische Wahrnehmungsmetrik (§4.4)."""
        context = {}
        goal = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "cdpam" in models

    def test_never_includes_dnsmos(self, policy_engine):
        """DNSMOS ist für Musik VERBOTEN (§4.4/§10.2): trainiert auf Sprachkorpora."""
        context = {}
        goal = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "dnsmos" not in models

    def test_never_includes_nisqa(self, policy_engine):
        """NISQA ist für Musik VERBOTEN (§4.4/§10.2): Sprachqualitäts-CNN."""
        context = {"has_vocals": True}
        goal = {}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "nisqa" not in models

    def test_never_includes_pesq(self, policy_engine):
        """PESQ ist für Musik VERBOTEN (§4.4/§10.2): Telefonband 300–3400 Hz."""
        context = {"has_vocals": True}
        goal = {"has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "pesq" not in models

    def test_with_reference_includes_visqol(self, policy_engine):
        """Bei vorhandener Referenz: ViSQOL v3 (--audio Mode zwingend, §4.4)."""
        context = {"has_vocals": False}
        goal = {"has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        assert "visqol" in models

    def test_full_assessment_uses_music_metrics(self, policy_engine):
        """Vollständige Bewertung: CDPAM + ViSQOL + PEAQ + FAD — keine Sprachmetriken (§4.4)."""
        context = {}
        goal = {"assessment_type": "full", "has_reference": True}
        models = policy_engine.select_quality_assessment_model(context, goal)
        expected = {"cdpam", "visqol", "peaq", "fad"}
        assert set(models) == expected
        # Verbotene Metriken dürfen nicht enthalten sein
        assert "dnsmos" not in models
        assert "nisqa" not in models
        assert "pesq" not in models


class TestVocoderSelection:
    """Test vocoder model selection."""

    def test_fast_selects_vocos(self, policy_engine):
        """Fast vocoding: Vocos (ConvNeXt-iSTFT, 8× schneller als BigVGAN-v2, §4.5)."""
        context = {}
        goal = {"quality_level": "fast"}
        model = policy_engine.select_vocoder_model(context, goal)
        assert model == "vocos"

    def test_high_quality_selects_vocos(self, policy_engine):
        """High-quality vocoding: Vocos 0.1.0 als Primär-Vocoder (§4.5)."""
        context = {}
        goal = {"quality_level": "high"}
        model = policy_engine.select_vocoder_model(context, goal)
        assert model == "vocos"


class TestSpecializedModels:
    """Test specialized model selection."""

    def test_audio_tagging_selects_panns(self, policy_engine):
        """Audio tagging should select PANNS."""
        context = {}
        goal = {}
        model = policy_engine.select_audio_tagging_model(context, goal)
        assert model == "panns"

    def test_mastering_selects_matchering(self, policy_engine):
        """Mastering should select Matchering."""
        context = {}
        goal = {}
        model = policy_engine.select_mastering_model(context, goal)
        assert model == "matchering"

    def test_pitch_detection_selects_crepe(self, policy_engine):
        """Pitch detection should select CREPE."""
        context = {}
        goal = {}
        model = policy_engine.select_pitch_detection_model(context, goal)
        assert model == "crepe"


class TestGenerativeModels:
    """Test generative model selection."""

    def test_music_generation_selects_vampnet(self, policy_engine):
        """Music generation should select VampNet."""
        context = {}
        goal = {"generation_type": "music"}
        model = policy_engine.select_generative_model(context, goal)
        assert model == "vampnet"

    def test_text_to_audio_selects_audioldm2(self, policy_engine):
        """Text-to-audio should select AudioLDM2."""
        context = {}
        goal = {"generation_type": "text_to_audio"}
        model = policy_engine.select_generative_model(context, goal)
        assert model == "audioldm2"


class TestMediumSpecific:
    """Test medium-specific model selection."""

    def test_vinyl_medium_specific(self, policy_engine):
        """Vinyl should select Banquet for medium-specific restoration."""
        context = {"detected_medium": "vinyl"}
        goal = {}
        model = policy_engine.select_medium_specific_model(context, goal)
        assert model == "banquet"

    def test_unknown_medium_fallback_to_denoise(self, policy_engine):
        """Unknown medium should fallback to denoise model."""
        context = {"detected_medium": "unknown"}
        goal = {}
        model = policy_engine.select_medium_specific_model(context, goal)
        # Should return one of the denoise models
        assert model in ["resemble_enhance", "deepfilternet", "dccrn", "wpe", "banquet"]


class TestSelectAllModels:
    """Test comprehensive model selection."""

    def test_select_multiple_tasks(self, policy_engine):
        """Should select appropriate models for multiple tasks."""
        context = {"has_vocals": True, "detected_medium": "vinyl"}
        tasks = ["denoise", "separation", "quality"]

        models = policy_engine.select_all_models(context, tasks)

        # Should have selected models for all tasks
        assert "denoise" in models
        assert "separation" in models
        assert "quality" in models

        # Denoise should be banquet (vinyl)
        assert models["denoise"] == "banquet"

        # Quality should be list
        assert isinstance(models["quality"], list)

    def test_select_all_task_types(self, policy_engine):
        """Test selection for all supported task types."""
        context = {}
        tasks = [
            "denoise",
            "repair",
            "separation",
            "enhancement",
            "quality",
            "vocoder",
            "tagging",
            "mastering",
            "generation",
            "pitch",
            "medium_specific",
        ]

        models = policy_engine.select_all_models(context, tasks)

        # Should have all 11 tasks covered
        assert len(models) == 11


class TestConvenienceFunctions:
    """Test convenience helper functions."""

    def test_get_recommended_models_basic(self):
        """Should recommend basic denoise + quality models."""
        context = {}
        recommendations = get_recommended_models(context)

        # Should have denoise and quality
        assert "denoise" in recommendations
        assert "quality" in recommendations

        # Denoise should be string
        assert isinstance(recommendations["denoise"], str)

        # Quality should be list
        assert isinstance(recommendations["quality"], list)

    def test_get_recommended_models_with_vocals(self):
        """Should add separation for vocal content."""
        context = {"has_vocals": True}
        recommendations = get_recommended_models(context)

        # Should include separation
        assert "separation" in recommendations


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context(self, policy_engine):
        """Empty context should use fallback models."""
        context = {}
        goal = {}

        # Should not raise errors
        denoise = policy_engine.select_denoise_model(context, goal)
        repair = policy_engine.select_repair_model(context, goal)
        separation = policy_engine.select_stem_separation_model(context, goal)

        # All should return valid model names
        assert denoise in ["resemble_enhance", "deepfilternet", "dccrn", "wpe", "banquet"]
        assert repair in ["dccrn", "fullsubnet"]
        assert separation in ["mdx23c", "demucs", "uvr_mdxnet"]

    def test_conflicting_context_signals(self, policy_engine):
        """Conflicting signals should follow priority order."""
        # Vinyl + vocals → Vinyl should win (higher priority)
        context = {"detected_medium": "vinyl", "has_vocals": True}
        model = policy_engine.select_denoise_model(context, {})
        assert model == "banquet"

    def test_invalid_goal_parameters(self, policy_engine):
        """Invalid goal parameters should use defaults."""
        context = {}
        goal = {"invalid_key": "invalid_value"}

        # Should not crash, use defaults
        model = policy_engine.select_denoise_model(context, goal)
        assert isinstance(model, str)


# Integration Test
def test_full_workflow_integration():
    """Test full workflow: context → policy → models."""
    policy = MLModelPolicyEngine()

    # Scenario: Professional vinyl restoration
    context = {"detected_medium": "vinyl", "has_vocals": True, "genre": "jazz", "content_character": "BALANCED"}

    # Get models for restoration workflow
    denoise = policy.select_denoise_model(context, {"quality_level": "maximal"})
    quality = policy.select_quality_assessment_model(context, {"has_reference": True})

    # Verify vinyl-specific model selected
    assert denoise == "banquet"

    # Verify music-appropriate quality assessment (§4.4/§10.2: DNSMOS/NISQA/PESQ verboten)
    assert "cdpam" in quality  # Musik-Wahrnehmungsmetrik (primär)
    assert "visqol" in quality  # ViSQOL v3 (--audio Mode) mit Referenz
    assert "dnsmos" not in quality  # VERBOTEN: Sprachkorpus
    assert "nisqa" not in quality  # VERBOTEN: Sprachqualitäts-CNN
    assert "pesq" not in quality  # VERBOTEN: Telefonband


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
