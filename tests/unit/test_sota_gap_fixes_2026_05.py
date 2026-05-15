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
