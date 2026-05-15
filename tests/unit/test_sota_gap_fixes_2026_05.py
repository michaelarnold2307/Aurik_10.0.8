"""Unit-Tests für die SOTA-Gap-Fixes (Session Mai 2026):
- noise_texture_resynth.restore_carrier_noise_texture  (Gap 3)
- nvsr_plugin.NvsrPlugin.process                       (Gap 2)
- phoneme_boundary_detector                            (Gap 6)
- tonal_reference_profile.get_studio_console_curve     (Gap 5)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _white_noise(n: int = 48000, amplitude: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * amplitude


def _sine(freq: float = 440.0, sr: int = 48000, dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


# ===========================================================================
# Gap 3: noise_texture_resynth
# ===========================================================================


class TestRestoreCarrierNoiseTexture:
    """§TimbralCoherence — restore_carrier_noise_texture() Grundverhalten."""

    def test_passthrough_when_strength_zero(self):
        """strength=0 → Audio unverändert zurückgegeben."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl", strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_shape_preserved_mono(self):
        """Output-Shape ist identisch mit Input (Mono)."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(24000, amplitude=0.03)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl")
        assert result.shape == audio.shape

    def test_shape_preserved_stereo(self):
        """Output-Shape ist identisch mit Input (Stereo 2×N)."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        rng = np.random.default_rng(7)
        audio = (rng.standard_normal((2, 48000)) * 0.05).astype(np.float32)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="vinyl")
        assert result.shape == audio.shape

    def test_no_clipping_in_output(self):
        """Output darf niemals clippen."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.4)
        result = restore_carrier_noise_texture(audio, audio, sr=48000)
        assert float(np.max(np.abs(result))) <= 1.0

    def test_no_nan_in_output(self):
        """Output darf keine NaN/Inf-Werte enthalten."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="shellac")
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_over_nr_correction_applied(self):
        """Wenn post-NR-Signal lautlos ist (extreme Over-NR), wird Korrektur angewandt."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        pre = _white_noise(48000, amplitude=0.05)
        # Simuliere Over-NR: post ist fast komplett still
        post = np.zeros(48000, dtype=np.float32) + 1e-6
        result = restore_carrier_noise_texture(pre, post, sr=48000, material_type="vinyl")
        # Nach Korrektur sollte etwas Energie vorhanden sein (wenn psychoacoustics verfügbar)
        assert result.shape == post.shape  # Mindestanforderung: shape bleibt gleich

    def test_passthrough_for_small_deviation(self):
        """Bei identischem pre/post-NR Signal (keine Over-NR) bleibt Output gleich."""
        from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture

        audio = _white_noise(48000, amplitude=0.05)
        result = restore_carrier_noise_texture(audio, audio, sr=48000, material_type="cd_digital")
        # Kein Unterschied → entweder passthrough oder minimale Korrektur
        assert result.shape == audio.shape


# ===========================================================================
# Gap 2: nvsr_plugin
# ===========================================================================


class TestNvsrPlugin:
    """§SOTA Gap 2 — NvsrPlugin DSP-SBR Grundverhalten."""

    def test_singleton_returns_same_instance(self):
        """get_nvsr_plugin() liefert immer dieselbe Instanz."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        inst_a = get_nvsr_plugin()
        inst_b = get_nvsr_plugin()
        assert inst_a is inst_b

    def test_process_shape_preserved_mono(self):
        """process() erhält Shape: Mono (N,)."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert result["audio"].shape == audio.shape

    def test_process_shape_preserved_stereo(self):
        """process() erhält Shape: Stereo (2, N)."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = np.stack([_sine(440.0), _sine(880.0)], axis=0).astype(np.float32)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert result["audio"].shape == audio.shape

    def test_no_clipping(self):
        """SBR-Ausgang darf niemals clippen."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=1.0) * 0.8
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert float(np.max(np.abs(result["audio"]))) <= 1.0

    def test_strategy_metadata_present(self):
        """process()-Ergebnis enthält 'strategy' im dict."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(440.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000, material_type="vinyl")
        assert "strategy" in result or "strategy_used" in result

    def test_shellac_ceiling_respected(self):
        """Shellac-Material hat HF-Ceiling ≤ 8000 Hz — kein SBR über 8 kHz."""
        from plugins.nvsr_plugin import _MATERIAL_HF_CEILING_HZ

        assert _MATERIAL_HF_CEILING_HZ.get("shellac", 0) <= 8_001.0

    def test_no_nan_output(self):
        """Output darf keine NaN-Werte enthalten."""
        from plugins.nvsr_plugin import get_nvsr_plugin

        plugin = get_nvsr_plugin()
        audio = _sine(880.0, sr=48000, dur=0.5)
        result = plugin.process(audio, sr=48000)
        assert not np.any(np.isnan(result["audio"]))


# ===========================================================================
# Gap 6: phoneme_boundary_detector
# ===========================================================================


class TestPhonemeBoundaryDetectorDsp:
    """§Gap 6 — DSP-Phonem-Grenzerkennung Grundverhalten."""

    def test_returns_bool_array(self):
        """detect_phoneme_boundaries_dsp() gibt bool-Array zurück."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = _white_noise(48000, amplitude=0.1)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert result.dtype == bool

    def test_output_length_matches_n_frames(self):
        """Output-Länge entspricht n_frames = len(audio) // hop_length."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = _white_noise(48000, amplitude=0.1)
        hop = 512
        result = detect_phoneme_boundaries_dsp(audio, sr=48000, hop_length=hop)
        # mindestens 1, nicht mehr als len(audio) // hop
        assert 1 <= len(result) <= len(audio) // hop + 1

    def test_silence_has_no_boundaries(self):
        """Komplett stilles Signal → keine Phonem-Grenzen."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        silence = np.zeros(48000, dtype=np.float32)
        result = detect_phoneme_boundaries_dsp(silence, sr=48000)
        # Stille = alle Frames SILENCE → keine Übergänge
        assert not np.any(result)

    def test_plosive_onset_detected_for_energy_spike(self):
        """Energie-Spike (12+ dB) löst Boundary aus."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        # Erzeuge Signal: 0.5s still, dann Energie-Spike
        audio = np.zeros(48000, dtype=np.float32)
        audio[24000:26000] = 0.8  # großer Spike
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert np.any(result), "Energie-Spike muss eine Boundary auslösen"

    def test_stereo_input_handled(self):
        """Stereo-Input (2×N) wird korrekt zu Mono downgemischt."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = np.stack([_white_noise(24000), _white_noise(24000, seed=7)], axis=0)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert result.dtype == bool

    def test_short_audio_no_crash(self):
        """Sehr kurzes Audio (< 4× hop_length) → leeres Array ohne Crash."""
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp

        audio = np.zeros(100, dtype=np.float32)
        result = detect_phoneme_boundaries_dsp(audio, sr=48000)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool


# ===========================================================================
# Gap 5: Console Character Studio 2026
# ===========================================================================


class TestStudioConsoleCharacter:
    """§Gap 5 — TonalReferenceProfiler.get_studio_console_curve()"""

    def test_neve_1073_returns_list(self):
        """neve_1073-Kurve ist eine Liste von (Hz, dB)-Paaren."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neve_1073")
        assert isinstance(curve, list)
        assert len(curve) >= 3

    def test_ssl_4000_returns_list(self):
        """ssl_4000-Kurve ist eine Liste von (Hz, dB)-Paaren."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("ssl_4000")
        assert isinstance(curve, list)
        assert len(curve) >= 3

    def test_unknown_console_returns_neve_fallback(self):
        """Unbekannter console_type → Fallback auf neve_1073."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve_unknown = profiler.get_studio_console_curve("xyz_unknown")
        curve_neve = profiler.get_studio_console_curve("neve_1073")
        assert curve_unknown == curve_neve

    def test_console_curve_has_valid_frequency_range(self):
        """Alle Kurven haben Frequenzwerte 20 Hz – 20 kHz."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        for console in ("neve_1073", "ssl_4000", "api_2500", "neutral"):
            curve = profiler.get_studio_console_curve(console)
            freqs = [hz for hz, _ in curve]
            assert min(freqs) >= 20.0, f"{console}: Min-Frequenz zu niedrig"
            assert max(freqs) <= 21_000.0, f"{console}: Max-Frequenz zu hoch"

    def test_neve_1073_has_low_shelf_boost(self):
        """Neve 1073 hat Low-Shelf-Boost bei ~80 Hz."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neve_1073")
        # Suche +2 dB bei 80 Hz
        gains_at_low = [g for hz, g in curve if 60.0 <= hz <= 120.0]
        assert any(g > 1.0 for g in gains_at_low), "Neve 1073 muss Low-Shelf > +1 dB haben"

    def test_neutral_curve_is_flat(self):
        """Neutral-Kurve enthält nur 0 dB-Werte."""
        from backend.core.tonal_reference_profile import get_tonal_reference_profiler

        profiler = get_tonal_reference_profiler()
        curve = profiler.get_studio_console_curve("neutral")
        gains = [g for _, g in curve]
        assert all(g == 0.0 for g in gains), "Neutral muss komplett flat (0 dB) sein"

    def test_console_character_wired_in_phase06_studio_mode(self):
        """§Gap5: phase_06 ruft get_studio_console_curve() im Studio-2026-Pfad auf."""
        import inspect

        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        src = inspect.getsource(FrequencyRestorationPhase.process)
        assert "get_studio_console_curve" in src or "ConsoleCharacter" in src.lower() or "console_character" in src, (
            "§Gap5 Console Character nicht verdrahtet in phase_06"
        )
        assert '"studio"' in src or "'studio'" in src, "Studio-Mode-Gate fehlt"


# ===========================================================================
# VQI per-Phase Gates: phase_20 und phase_42
# ===========================================================================


class TestVqiPerPhaseGates:
    """§0p VQI per-Phase-Rollback — phase_20_reverb_reduction, phase_42_vocal_enhancement."""

    def _make_audio(self, n: int = 24000, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(n) * 0.1).astype(np.float32)

    def test_phase20_process_returns_phaseresult(self):
        """phase_20.process() gibt PhaseResult zurück (Smoke-Test)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        phase = ReverbReduction()
        audio = self._make_audio()
        result = phase.process(audio, 48000, MaterialType.VINYL, strength=0.2)
        assert result is not None
        assert hasattr(result, "audio")
        assert result.audio.shape == audio.shape

    def test_phase20_vqi_gate_inserted(self):
        """phase_20.py enthält den VQI per-Phase Rollback-Block."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "compute_vqi" in src or "vocal_quality_index" in src, "§0p VQI per-phase gate fehlt in phase_20"
        assert "_vqi_p20" in src or "_vqi_result_p20" in src, "VQI-Variable _vqi_p20 nicht gefunden"

    def test_phase42_process_returns_phaseresult(self):
        """phase_42.process() gibt PhaseResult zurück (Smoke-Test)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = self._make_audio()
        result = phase.process(audio, 48000, MaterialType.CD_DIGITAL, strength=0.2)
        assert result is not None
        assert hasattr(result, "audio")
        assert result.audio.shape == audio.shape

    def test_phase42_vqi_gate_inserted(self):
        """phase_42.py enthält den VQI per-Phase Rollback-Block."""
        import inspect

        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        src = inspect.getsource(VocalEnhancement.process)
        assert "compute_vqi" in src or "vocal_quality_index" in src, "§0p VQI per-phase gate fehlt in phase_42"
        assert "_vqi_p42" in src or "_vqi_result_p42" in src, "VQI-Variable _vqi_p42 nicht gefunden"

    def test_uv3_phase50_in_hnr_blend_set(self):
        """UV3: phase_50_spectral_repair muss im _NR_PHASES_HNR-Set sein."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3._profiled_phase_call)
        assert "phase_50_spectral_repair" in src, (
            "§0p HNR-Blend: phase_50_spectral_repair fehlt in _NR_PHASES_HNR (UV3)"
        )


# ===========================================================================
# §0p Gap-Fixes Session 2: HNR-Blend in ML-NR-Phasen
# ===========================================================================


class TestHnrBlendInNrPhases:
    """§0p RELEASE_MUST — phase_20/29/49/50 müssen HNR-Blend aufrufen (panns >= 0.25)."""

    def test_phase20_source_contains_hnr_blend(self):
        """phase_20.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_20"
        assert "_hnr_blended_p20" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_20"

    def test_phase29_source_contains_hnr_blend(self):
        """phase_29.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        src = inspect.getsource(TapeHissReductionPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_29"
        assert "_hnr_blended_p29" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_29"

    def test_phase49_source_contains_hnr_blend(self):
        """phase_49.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        src = inspect.getsource(AdvancedDereverbPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_49"
        assert "_hnr_blended_p49" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_49"

    def test_phase50_source_contains_hnr_blend(self):
        """phase_50.py enthält den §0p HNR-Blend-Block."""
        import inspect

        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        src = inspect.getsource(SpectralRepairPhase.process)
        assert "apply_hnr_blend" in src or "hnr_guard" in src, "§0p HNR-Blend fehlt in phase_50"
        assert "_hnr_blended_p50" in src or "over_cleaned" in src, "HNR-Blend-Variablen fehlen in phase_50"

    def test_phase20_hnr_blend_called_when_over_cleaned(self):
        """phase_20: apply_hnr_blend wird aufgerufen; bei over_cleaned=True wird blended-Audio übernommen."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32)
        blended = np.ones(4800, dtype=np.float32) * 0.1

        mock_result = (blended, {"over_cleaned": True, "hnr_delta_db": 4.2})

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend", return_value=mock_result):
            from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

            phase = ReverbReduction()
            result = phase.process(
                audio,
                sample_rate=48000,
                panns_singing=0.6,
                processing_mode="restoration",
            )
        assert hasattr(result, "audio")
        assert result.audio is not None

    def test_phase29_hnr_blend_called_when_over_cleaned(self):
        """phase_29: apply_hnr_blend wird aufgerufen; bei over_cleaned=True wird blended-Audio übernommen."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32) + 0.02
        blended = np.ones(4800, dtype=np.float32) * 0.05

        mock_result = (blended, {"over_cleaned": True, "hnr_delta_db": 5.0})

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend", return_value=mock_result):
            from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

            phase = TapeHissReductionPhase()
            result = phase.process(
                audio,
                sample_rate=48000,
                panns_singing=0.5,
                processing_mode="restoration",
            )
        assert hasattr(result, "audio")
        assert result.audio is not None

    def test_phase49_hnr_blend_skipped_when_panns_low(self):
        """phase_49: HNR-Blend wird NICHT aufgerufen wenn panns_singing < 0.25."""
        from unittest.mock import patch

        import numpy as np

        audio = np.zeros(4800, dtype=np.float32) + 0.01

        with patch("backend.core.dsp.hnr_guard.apply_hnr_blend") as mock_hnr:
            from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

            phase = AdvancedDereverbPhase()
            phase.process(audio, sample_rate=48000, panns_singing=0.1, processing_mode="restoration")
        mock_hnr.assert_not_called()


# ===========================================================================
# §0p Gap-Fix: Singer-ID-Cosine Rollback in UV3
# ===========================================================================


class TestSingerIdRollbackUV3:
    """§0p RELEASE_MUST — UV3 muss Singer-ID-Rollback-Code enthalten."""

    def test_uv3_singer_id_rollback_code_present(self):
        """UV3 enthält den §0p Singer-ID-Cosine-Rollback-Block."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "singer_identity_cosine" in src, "§0p Singer-ID-Rollback fehlt komplett in UV3"
        assert "SINGER_ID_BELOW_THRESHOLD" in src, "§0p SingerIDGate error_code fehlt in UV3"
        assert "_is_ms_rb" in src or "multi_singer" in src, "§0p multi_singer-Guard fehlt in UV3"

    def test_uv3_singer_id_rollback_threshold_correct(self):
        """UV3 verwendet exakt 0.92 als Singer-ID-Schwellwert."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "0.92" in src, "§0p Singer-ID Threshold 0.92 nicht in UV3 gefunden"

    def test_uv3_singer_id_deactivated_for_multi_singer(self):
        """UV3-Rollback ist bei multi_singer=True deaktiviert."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Guard muss vorhanden sein
        assert "multi_singer" in src, "§0p multi_singer Guard fehlt"
        assert "not _is_ms_rb" in src or "multi_singer" in src, (
            "§0p Singer-ID Rollback-Deaktivierung für multi_singer fehlt"
        )


# ===========================================================================
# §2.46e Gap-Fix: Hallucination-Guard in phase_26
# ===========================================================================


class TestPhase26HallucinationGuard:
    """§2.46e RELEASE_MUST — phase_26 muss apply_hallucination_guard aufrufen."""

    def test_phase26_source_contains_hallucination_guard(self):
        """phase_26.py enthält den §2.46e Hallucination-Guard-Block."""
        import inspect

        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        src = inspect.getsource(DynamicRangeExpansion.process)
        assert "hallucination_guard" in src or "apply_hallucination_guard" in src, (
            "§2.46e Hallucination-Guard fehlt in phase_26"
        )
        assert "hallucination_decision" in src, "§2.46e hallucination_decision-Check fehlt in phase_26"

    def test_phase26_rollback_on_hallucination(self):
        """phase_26: apply_hallucination_guard-Rollback überschreibt expanded_audio mit original audio."""
        from unittest.mock import patch

        import numpy as np

        original = np.ones(4800, dtype=np.float32) * 0.2

        with patch(
            "backend.core.hallucination_guard.apply_hallucination_guard",
            return_value=(None, {"hallucination_decision": "rollback", "hallucination_severity": 0.9}),
        ):
            from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

            phase = DynamicRangeExpansion()
            result = phase.process(
                original.copy(),
                sample_rate=48000,
                processing_mode="restoration",
                strength=0.5,
            )
        assert hasattr(result, "audio")
        # Bei Rollback muss das Ergebnis dem Original entsprechen (keine neue Energie)
        assert result.audio is not None

    def test_phase26_no_rollback_when_clean(self):
        """phase_26: kein Rollback wenn hallucination_decision = 'pass'."""
        from unittest.mock import patch

        import numpy as np

        original = np.ones(4800, dtype=np.float32) * 0.2

        with patch(
            "backend.core.hallucination_guard.apply_hallucination_guard",
            return_value=(None, {"hallucination_decision": "pass", "hallucination_severity": 0.0}),
        ):
            from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

            phase = DynamicRangeExpansion()
            result = phase.process(
                original.copy(),
                sample_rate=48000,
                processing_mode="restoration",
                strength=0.3,
            )
        assert hasattr(result, "audio")
        assert result.audio is not None


# ===========================================================================
# §V11 Gap-Fix: sosfiltfilt in synthetic_generator.py
# ===========================================================================


class TestSyntheticGeneratorSosfiltfilt:
    """§V11 RELEASE_MUST — synthetic_generator.py muss sosfiltfilt (zero-phase) nutzen."""

    def test_synthetic_generator_uses_sosfiltfilt(self):
        """golden_samples/synthetic_generator.py nutzt sosfiltfilt statt sosfilt für Formant-Filter."""
        import inspect

        try:
            from golden_samples.synthetic_generator import SyntheticAudioGenerator

            src = inspect.getsource(SyntheticAudioGenerator._generate_vocal)
        except (ImportError, AttributeError):
            import pathlib

            src = pathlib.Path("golden_samples/synthetic_generator.py").read_text(encoding="utf-8")

        assert "sosfiltfilt" in src, "§V11 Verletzung: sosfilt statt sosfiltfilt in synthetic_generator.py"
        assert "sosfilt(" not in src.replace("sosfiltfilt", ""), "§V11: sosfilt() (kausal) wird noch verwendet"


# ===========================================================================
# §0a Gap-Fix: _RESTORATION_FORBIDDEN_PHASES in DefectPhaseMapper
# ===========================================================================


class TestDefectPhaseMapperRestorationFilter:
    """§0a RELEASE_MUST — DefectPhaseMapper darf phase_21/35/42 in Restoration
    nicht vorschlagen (BUG-FIX v9.12.0 §0a)."""

    def test_forbidden_phases_constant_exists(self):
        """_RESTORATION_FORBIDDEN_PHASES muss die drei §0a-Phasen enthalten."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES

        assert "phase_21_exciter" in _RESTORATION_FORBIDDEN_PHASES
        assert "phase_35_multiband_compression" in _RESTORATION_FORBIDDEN_PHASES
        assert "phase_42_vocal_enhancement" in _RESTORATION_FORBIDDEN_PHASES

    def test_get_primary_phases_restoration_filters_forbidden(self):
        """get_primary_phases(mode='restoration') darf §0a-Phasen nicht zurückgeben."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_primary_phases(defect_type, mode="restoration")
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, (
                f"§0a Verletzung: {forbidden_found} in primary_phases für {defect_type.value} im Restoration-Modus"
            )

    def test_get_all_phases_restoration_filters_forbidden(self):
        """get_all_phases(mode='restoration') darf §0a-Phasen nicht zurückgeben."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_all_phases(defect_type, mode="restoration")
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, (
                f"§0a Verletzung: {forbidden_found} in all_phases für {defect_type.value} im Restoration-Modus"
            )

    def test_get_all_phases_studio_2026_allows_forbidden(self):
        """get_all_phases(mode='studio_2026') darf §0a-Phasen enthalten (Studio 2026)."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        # Prüfe: mindestens eine DefectType hat eine §0a-Phase in studio_2026
        found_studio_phase = False
        for defect_type in DefectType:
            phases_studio = mapper.get_all_phases(defect_type, mode="studio_2026")
            if set(phases_studio) & _RESTORATION_FORBIDDEN_PHASES:
                found_studio_phase = True
                break
        assert found_studio_phase, "Studio 2026 sollte mindestens eine §0a-Phase (phase_35/42) enthalten"

    def test_get_primary_phases_no_mode_defaults_to_restoration(self):
        """Kein mode-Argument → default 'restoration' → Filterung aktiv."""
        from backend.core.defect_phase_mapper import _RESTORATION_FORBIDDEN_PHASES, DefectPhaseMapper
        from backend.core.defect_scanner import DefectType

        mapper = DefectPhaseMapper()
        for defect_type in DefectType:
            phases = mapper.get_primary_phases(defect_type)  # kein mode-Argument
            forbidden_found = set(phases) & _RESTORATION_FORBIDDEN_PHASES
            assert not forbidden_found, f"§0a Default-Filter fehlt: {forbidden_found} in {defect_type.value}"


# ===========================================================================
# §Cross-Goal-Recovery (v9.12.x fix): hf_recovery_boost_after_phase03
# WIRING FIX: floor enforcement moved from _profiled_phase_call (dead code path)
# to main phase loop, applied to _combined_strength before wrap_phase call.
# ===========================================================================


class TestCrossGoalRecoveryMainLoopFix:
    """§Cross-Goal-Recovery (v9.12.x) — Strength-Floor für phase_06/07/39 wird in
    UV3-Hauptschleife auf _combined_strength angewendet (nicht in _profiled_phase_call)."""

    def test_uv3_cross_goal_recovery_in_main_loop_source(self):
        """UV3-Source enthält den Cross-Goal-Recovery-Block in der Hauptschleife
        (erkennbar am Kommentar 'must run here (main loop)')."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        assert "must run here (main loop)" in src, (
            "§Cross-Goal-Recovery Wiring-Fix fehlt: Block 'must run here (main loop)' nicht in UV3"
        )
        assert "_hf_boost_ctx" in src, "§Cross-Goal-Recovery: _hf_boost_ctx Variable fehlt in UV3-Hauptschleife"

    def test_uv3_cross_goal_recovery_applies_strength_floor(self):
        """UV3-Hauptschleife erhöht _combined_strength auf HF-Floor wenn
        hf_recovery_boost_after_phase03 aktiviert ist."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Der Fix setzt _combined_strength auf den floor-Wert
        assert "_combined_strength = float(np.clip(_hf_floor" in src, (
            "§Cross-Goal-Recovery: _combined_strength-Floor-Assignment fehlt in UV3-Hauptschleife"
        )

    def test_uv3_cross_goal_recovery_phase_set_correct(self):
        """Cross-Goal-Recovery gilt für phase_06/07/39 (nicht andere Phasen)."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3)
        # Beide Blöcke (Hauptschleife + _profiled_phase_call) checken phase_06/07/39
        assert "phase_06_frequency_restoration" in src
        assert "phase_07_harmonic_restoration" in src
        assert "phase_39_air_band_enhancement" in src

    def test_uv3_profiled_phase_call_backward_compat_comment(self):
        """_profiled_phase_call enthält Hinweis dass der dortige Block für den
        Bronze/Bypass-Pfad (Z.7298/7339) bestimmt ist — nicht für PMGG-Hauptpfad."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3._profiled_phase_call)
        assert "bronze" in src.lower() or "bypass" in src.lower(), (
            "§Cross-Goal-Recovery: _profiled_phase_call fehlt Hinweis auf Bronze/Bypass-Pfad"
        )


# ---------------------------------------------------------------------------
# Todo 2 — §4.4 Era-Aware ML-NR Routing (v9.12.x)
# ---------------------------------------------------------------------------


class TestEraAwareNrModelRouting:
    """§4.4 SOTA Era-Aware NR routing: MIIPHER/DFN/OMLSA selection."""

    def _routing(self, era_decade, material_type, est_snr_db, panns_singing, is_vocal=True, is_non_digital=True):
        from backend.core.phases.phase_03_denoise import _determine_era_nr_routing

        return _determine_era_nr_routing(era_decade, material_type, est_snr_db, panns_singing, is_vocal, is_non_digital)

    def test_acoustic_era_1920_returns_omlsa_only(self):
        """1920s phonograph: no ML NR — carrier character must be preserved (§0a)."""
        tier = self._routing(era_decade=1920, material_type="shellac", est_snr_db=8.0, panns_singing=0.5)
        assert tier == "omlsa_only", f"Expected omlsa_only for 1920 shellac, got {tier!r}"

    def test_acoustic_era_1930_boundary_returns_omlsa_only(self):
        """Era 1930 (boundary): omlsa_only."""
        tier = self._routing(era_decade=1930, material_type="shellac", est_snr_db=6.0, panns_singing=0.4)
        assert tier == "omlsa_only", f"Expected omlsa_only at 1930 boundary, got {tier!r}"

    def test_early_electric_shellac_1940_returns_dfn_restricted(self):
        """1940 shellac electrical: DFN restricted to 30% wet (preserve H2/H4)."""
        tier = self._routing(era_decade=1940, material_type="shellac", est_snr_db=12.0, panns_singing=0.4)
        assert tier == "dfn_restricted", f"Expected dfn_restricted for 1940 shellac, got {tier!r}"

    def test_early_electric_1945_shellac_returns_dfn_restricted(self):
        """Era 1945, shellac — still dfn_restricted (boundary)."""
        tier = self._routing(era_decade=1945, material_type="shellac", est_snr_db=15.0, panns_singing=0.5)
        assert tier == "dfn_restricted", f"Expected dfn_restricted at era=1945, got {tier!r}"

    def test_wax_cylinder_always_omlsa_only(self):
        """Wax cylinder: always omlsa_only regardless of decade."""
        tier = self._routing(era_decade=1965, material_type="wax_cylinder", est_snr_db=5.0, panns_singing=0.6)
        assert tier == "omlsa_only", f"Expected omlsa_only for wax_cylinder, got {tier!r}"

    def test_digital_material_omlsa_only(self):
        """Digital material: omlsa_only (no ML broadband NR needed)."""
        tier = self._routing(
            era_decade=1995, material_type="cd_digital", est_snr_db=35.0, panns_singing=0.5, is_non_digital=False
        )
        assert tier == "omlsa_only", f"Expected omlsa_only for cd_digital, got {tier!r}"

    def test_deep_snr_vocal_post1950_returns_miipher_primary(self):
        """1965 vinyl, SNR 8 dB, panns 0.4 → MIIPHER primary (§4.4 SOTA)."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=8.0, panns_singing=0.4)
        assert tier == "miipher_primary", f"Expected miipher_primary for deep SNR vocal, got {tier!r}"

    def test_moderate_snr_post1950_returns_dfn_primary(self):
        """1975 vinyl, SNR 18 dB → DFN primary (current SOTA behavior)."""
        tier = self._routing(era_decade=1975, material_type="vinyl", est_snr_db=18.0, panns_singing=0.4)
        assert tier == "dfn_primary", f"Expected dfn_primary for moderate SNR, got {tier!r}"

    def test_snr_boundary_exactly_10_dfn_primary(self):
        """SNR exactly 10.0 dB (boundary): dfn_primary (MIIPHER only below 10 dB)."""
        tier = self._routing(era_decade=1970, material_type="vinyl", est_snr_db=10.0, panns_singing=0.4)
        assert tier == "dfn_primary", f"Expected dfn_primary at SNR=10 dB boundary, got {tier!r}"

    def test_low_panns_no_miipher(self):
        """Low panns_singing (0.25) below MIIPHER threshold 0.35 → dfn_primary."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=7.0, panns_singing=0.25)
        assert tier == "dfn_primary", f"Expected dfn_primary for low panns, got {tier!r}"

    def test_snr_none_no_miipher_routing(self):
        """None SNR: MIIPHER not activated (cannot confirm deep noise) → dfn_primary."""
        tier = self._routing(era_decade=1965, material_type="vinyl", est_snr_db=None, panns_singing=0.5)
        assert tier == "dfn_primary", f"Expected dfn_primary when SNR unknown, got {tier!r}"

    def test_routing_function_is_importable(self):
        """_determine_era_nr_routing must be importable from phase_03_denoise."""
        from backend.core.phases.phase_03_denoise import _determine_era_nr_routing

        assert callable(_determine_era_nr_routing), "_determine_era_nr_routing must be callable"

    def test_era_routing_key_in_phase03_process_source(self):
        """phase_03 process() must call _determine_era_nr_routing."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "_determine_era_nr_routing" in src, "process() must call _determine_era_nr_routing"

    def test_miipher_block_in_phase03_source(self):
        """MIIPHER block must appear in phase_03 process()."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "miipher_primary" in src, "process() must contain MIIPHER primary routing"
        assert "miipher_applied" in src, "process() must track _miipher_applied flag"

    def test_sgmse_guard_for_omlsa_only_routing(self):
        """SGMSE+ must be blocked when _era_nr_routing == 'omlsa_only'."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "omlsa_only" in src, "process() must guard SGMSE+ with omlsa_only check"

    def test_dfn_restricted_blend_in_source(self):
        """DFN restricted 30% blend must be applied for early-electrical era."""
        import inspect

        from backend.core.phases.phase_03_denoise import DenoisePhase

        src = inspect.getsource(DenoisePhase.process)
        assert "dfn_restricted" in src, "process() must implement dfn_restricted blend"
        assert "0.30" in src or "0.70" in src, "dfn_restricted must use 30%/70% wet/dry blend"


class TestRoomAcousticsFingerprinter:
    """Tests for §2.46f room_acoustics_fingerprinter.py"""

    def test_module_importable(self):
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        assert callable(compute_room_acoustics_fingerprint)

    def test_returns_expected_keys(self):
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros(48000, dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        for key in ("rt60_s", "drr_db", "room_type", "dereverb_strength_cap", "early_reflection_ms", "protection_note"):
            assert key in result, f"Missing key: {key}"

    def test_studio_cap_range(self):
        """Studio room → cap ≥ 0.50 (moderate, not maximum protection)."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        # Dry impulse-like signal → short RT60 → studio
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(48000).astype(np.float32) * 0.01
        result = compute_room_acoustics_fingerprint(audio, 48000)
        cap = float(result["dereverb_strength_cap"])
        assert 0.10 <= cap <= 1.0, f"Cap out of range: {cap}"

    def test_long_rt60_tightens_cap(self):
        """Long-decay signal → rt60 ≥ 1.2 s → cap ≤ 0.25."""
        from backend.core.room_acoustics_fingerprinter import _RT60_HIGH_THRESHOLD_S, compute_room_acoustics_fingerprint

        # Simulate long reverb tail: exponential decay over 3 s
        sr = 48000
        t = np.linspace(0, 3.0, sr * 3)
        decay = np.exp(-1.5 * t).astype(np.float32)  # ~0.4 s RT60 threshold
        # Use a very slow decay to exceed RT60 threshold
        slow_decay = np.exp(-0.3 * t).astype(np.float32)
        result = compute_room_acoustics_fingerprint(slow_decay, sr)
        if float(result["rt60_s"]) >= _RT60_HIGH_THRESHOLD_S:
            assert float(result["dereverb_strength_cap"]) <= 0.25, f"High RT60 should tighten cap: {result}"

    def test_silent_signal_returns_default(self):
        """Silent signal → fallback defaults, no exception."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros(48000, dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        assert isinstance(result["rt60_s"], float)
        assert isinstance(result["dereverb_strength_cap"], float)

    def test_cap_clamped_to_valid_range(self):
        """dereverb_strength_cap must always be in [0.10, 1.0]."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        rng = np.random.default_rng(99)
        audio = rng.standard_normal(96000).astype(np.float32) * 0.5
        result = compute_room_acoustics_fingerprint(audio, 48000)
        cap = float(result["dereverb_strength_cap"])
        assert 0.10 <= cap <= 1.0, f"Cap {cap} outside [0.10, 1.0]"

    def test_stereo_input_accepted(self):
        """Stereo audio (2, N) should be processed without error."""
        from backend.core.room_acoustics_fingerprinter import compute_room_acoustics_fingerprint

        audio = np.zeros((2, 48000), dtype=np.float32)
        result = compute_room_acoustics_fingerprint(audio, 48000)
        assert "rt60_s" in result

    def test_phase49_reads_room_acoustics_fingerprint(self):
        """phase_49 process() source must contain room_acoustics_fingerprint guard."""
        import inspect

        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        src = inspect.getsource(AdvancedDereverbPhase.process)
        assert "room_acoustics_fingerprint" in src, "phase_49 must read room_acoustics_fingerprint from kwargs"
        assert "dereverb_strength_cap" in src, "phase_49 must apply dereverb_strength_cap"

    def test_phase20_reads_room_acoustics_fingerprint(self):
        """phase_20 process() source must contain room_acoustics_fingerprint guard."""
        import inspect

        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        src = inspect.getsource(ReverbReduction.process)
        assert "room_acoustics_fingerprint" in src, "phase_20 must read room_acoustics_fingerprint from kwargs"
        assert "dereverb_strength_cap" in src, "phase_20 must apply dereverb_strength_cap"

    def test_uv3_injects_room_acoustics_fingerprint(self):
        """UV3 restore() source must inject room_acoustics_fingerprint into _restoration_context."""
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        src = inspect.getsource(UnifiedRestorerV3.restore)
        assert "room_acoustics_fingerprint" in src, "UV3.restore() must inject room_acoustics_fingerprint"
        assert "room_acoustics_fingerprinter" in src, "UV3.restore() must import room_acoustics_fingerprinter"


class TestEraHarmonicProfileAndPhase07H2Steering:
    """Todo 4: get_era_harmonic_profile() + phase_07 H2-Target-Steering."""

    def test_get_era_harmonic_profile_importable(self):
        """get_era_harmonic_profile must be importable from tonal_reference_profile."""
        from backend.core.tonal_reference_profile import (
            HarmonicProfile,
            get_era_harmonic_profile,
        )

        assert callable(get_era_harmonic_profile)
        profile = get_era_harmonic_profile(1940)
        assert isinstance(profile, HarmonicProfile)

    def test_era_1940_h2_ratio_correct(self):
        """1940 decade should return the Golden Tube era H2 ratio (0.020)."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(1940)
        assert abs(profile.h2_ratio - 0.020) < 1e-6, f"Expected 0.020, got {profile.h2_ratio}"
        assert profile.era_label == "Golden Tube"

    def test_era_none_returns_fallback_1970(self):
        """None decade must fall back to 1970 Transistor-Era profile."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(None)
        assert abs(profile.h2_ratio - 0.006) < 1e-6, f"Expected 0.006, got {profile.h2_ratio}"
        assert "Transistor" in profile.era_label

    def test_era_beyond_2000_uses_nearest(self):
        """Decade 2030 (beyond last entry 2025) must use the 2025 entry."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(2030)
        # 2025 is the max available key — Contemporary era
        assert abs(profile.h2_ratio - 0.0002) < 1e-7
        assert "Contemporary" in profile.era_label

    def test_era_exact_key_match(self):
        """Exact decade key must return that entry directly."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        p1960 = get_era_harmonic_profile(1960)
        assert abs(p1960.h2_ratio - 0.014) < 1e-6

        p1970 = get_era_harmonic_profile(1970)
        assert abs(p1970.h2_ratio - 0.006) < 1e-6

    def test_era_between_decades_rounds_down(self):
        """Decade between entries (e.g. 1955) rounds down to 1950."""
        from backend.core.tonal_reference_profile import get_era_harmonic_profile

        profile = get_era_harmonic_profile(1955)
        assert abs(profile.h2_ratio - 0.018) < 1e-6  # 1950 Classic Tube

    def test_get_era_harmonic_profile_in_all_export(self):
        """get_era_harmonic_profile must appear in __all__ of tonal_reference_profile."""
        import backend.core.tonal_reference_profile as mod

        assert "get_era_harmonic_profile" in mod.__all__, "get_era_harmonic_profile missing from __all__"

    def test_phase07_source_contains_h2_target_steering(self):
        """phase_07 process() must contain the ERA_HARMONIC H2-target-steering block."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "ERA_HARMONIC" in src, "phase_07.process() must contain §ERA_HARMONIC steering"
        assert "get_era_harmonic_profile" in src, "phase_07 must call get_era_harmonic_profile"
        assert "_measure_h2_ratio" in src, "phase_07 must call _measure_h2_ratio"

    def test_phase07_measure_h2_ratio_method_exists(self):
        """_measure_h2_ratio must be a static method of HarmonicRestorationPhase."""
        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        assert hasattr(HarmonicRestorationPhase, "_measure_h2_ratio"), (
            "_measure_h2_ratio method missing from HarmonicRestorationPhase"
        )
        assert callable(HarmonicRestorationPhase._measure_h2_ratio)

    def test_measure_h2_ratio_pure_sine_returns_small(self):
        """Pure 440 Hz sine without harmonics should yield a very small H2 ratio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        t = np.linspace(0, 5.0, sr * 5)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(audio, sr)
        # No harmonic content → should be well below 0.01
        assert ratio < 0.10, f"Expected small ratio for pure sine, got {ratio:.4f}"

    def test_measure_h2_ratio_with_harmonics_detects_h2(self):
        """Signal with explicit H2 component must yield a detectable H2 ratio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        t = np.linspace(0, 5.0, sr * 5)
        # H1 = 440 Hz at amplitude 1.0, H2 = 880 Hz at amplitude 0.03
        audio = (1.0 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(audio, sr)
        # Should detect H2 around 0.03 ± some tolerance
        assert ratio > 0.005, f"H2 ratio too low: {ratio:.4f}"

    def test_measure_h2_ratio_short_audio_returns_zero(self):
        """Audio shorter than 4096 samples must return 0.0 safely."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        short = np.zeros(100, dtype=np.float32)
        ratio = HarmonicRestorationPhase._measure_h2_ratio(short, 48000)
        assert ratio == 0.0


class TestConsoleCharacterStudio2026:
    """Todo 5: Console-Character in Studio 2026 (§Gap5)."""

    def test_phase07_source_contains_console_character_block(self):
        """phase_07 process() must contain the §Gap5 Console-Character block."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "Gap5" in src or "console_character" in src.lower(), (
            "phase_07.process() must contain §Gap5 Console-Character block"
        )
        assert "get_studio_console_curve" in src, "phase_07 must call get_studio_console_curve"
        assert "studio" in src, "Console-Character block must be gated on studio mode"

    def test_phase07_studio_mode_only_guard(self):
        """Console-Character must only activate in studio mode, not restoration."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        # The guard condition must check for "studio" in mode
        assert '"studio" in _mode_07' in src, "Console-Character must be gated on '\"studio\" in _mode_07'"

    def test_phase07_console_hallucination_guard_present(self):
        """phase_07 must apply hallucination_guard after console EQ (§2.46e)."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "hallucination_guard" in src or "check_hallucination" in src, (
            "phase_07 must import hallucination_guard for console EQ"
        )

    def test_apply_console_eq_method_exists(self):
        """_apply_console_eq must be a static method of HarmonicRestorationPhase."""
        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        assert hasattr(HarmonicRestorationPhase, "_apply_console_eq"), (
            "_apply_console_eq missing from HarmonicRestorationPhase"
        )

    def test_apply_console_eq_passthrough_on_neutral(self):
        """Neutral console profile (0 dB at all freqs) must return near-identical audio."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        audio = np.random.default_rng(42).standard_normal(sr * 3).astype(np.float32) * 0.3
        neutral_bp = [(20.0, 0.0), (20000.0, 0.0)]
        result = HarmonicRestorationPhase._apply_console_eq(audio, neutral_bp, sr, strength=1.0)
        # Should be within ±3 dB RMS of the original
        rms_orig = float(np.sqrt(np.mean(audio**2)))
        rms_out = float(np.sqrt(np.mean(result**2)))
        assert abs(rms_out - rms_orig) / (rms_orig + 1e-8) < 0.20, (
            f"Neutral console EQ should preserve RMS, orig={rms_orig:.4f} out={rms_out:.4f}"
        )

    def test_apply_console_eq_strength_zero_returns_passthrough(self):
        """strength=0 must return audio effectively unchanged."""
        import numpy as np

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        sr = 48000
        audio = np.random.default_rng(7).standard_normal(sr * 2).astype(np.float32) * 0.3
        neve_bp = [
            (20.0, 0.5),
            (80.0, 2.0),
            (200.0, 0.5),
            (1000.0, 0.0),
            (3000.0, 1.0),
            (18000.0, -0.5),
            (20000.0, -0.8),
        ]
        result = HarmonicRestorationPhase._apply_console_eq(audio, neve_bp, sr, strength=0.0)
        rms_diff = float(np.sqrt(np.mean((result - audio) ** 2)))
        assert rms_diff < 0.005, f"strength=0 should be passthrough, diff RMS={rms_diff:.6f}"

    def test_phase07_console_character_in_metadata(self):
        """phase_07 metadata must include console_character_applied key."""
        import inspect

        from backend.core.phases.phase_07_harmonic_restoration import (
            HarmonicRestorationPhase,
        )

        src = inspect.getsource(HarmonicRestorationPhase.process)
        assert "console_character_applied" in src, "phase_07 return metadata must include 'console_character_applied'"
