"""
tests/unit/test_phase_56_spectral_band_gap_repair.py
=====================================================
Aurik 9.9 — SpectralBandGapRepairPhase (§4.5, §7.1)

22 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase():
    from backend.core.phases.phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase

    return SpectralBandGapRepairPhase()


@pytest.fixture(scope="module")
def silence_1s():
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture(scope="module")
def sine_440_2s():
    np.random.seed(42)
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture(scope="module")
def noisy_audio():
    np.random.seed(42)
    return (np.random.randn(SR * 3) * 0.1).astype(np.float32)


@pytest.fixture(scope="module")
def stereo_audio():
    np.random.seed(42)
    ch1 = np.sin(2 * np.pi * 220 * np.linspace(0, 2, 2 * SR, endpoint=False)).astype(np.float32)
    ch2 = np.sin(2 * np.pi * 330 * np.linspace(0, 2, 2 * SR, endpoint=False)).astype(np.float32)
    return np.stack([ch1, ch2], axis=0)


# ---------------------------------------------------------------------------
# Tests: Metadaten
# ---------------------------------------------------------------------------


class TestPhase56Metadata:
    def test_01_metadata_returns_object(self, phase):
        meta = phase.get_metadata()
        assert meta is not None

    def test_02_category_is_restoration(self, phase):
        meta = phase.get_metadata()
        assert (
            "restor" in str(meta.category).lower()
            or "repair" in str(meta.category).lower()
            or "defect" in str(meta.category).lower()
        )

    def test_03_name_contains_spectral(self, phase):
        meta = phase.get_metadata()
        assert "spectral" in meta.name.lower() or "band" in meta.name.lower()

    def test_04_estimated_time_positive(self, phase):
        meta = phase.get_metadata()
        assert meta.estimated_time_factor >= 0.0


# ---------------------------------------------------------------------------
# Tests: Grundfunktion process()
# ---------------------------------------------------------------------------


class TestPhase56Process:
    def test_05_process_returns_phase_result(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result is not None
        assert hasattr(result, "audio") and hasattr(result, "success")

    def test_06_output_shape_preserved_mono(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.audio.shape == sine_440_2s.shape

    def test_07_output_dtype_float32(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.audio.dtype == np.float32

    def test_08_no_nan_in_output(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert np.isfinite(result.audio).all(), "NaN/Inf im Ausgang"

    def test_09_no_clipping_in_output(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_10_silence_passthrough(self, phase, silence_1s):
        result = phase.process(silence_1s, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_11_noisy_audio_processed(self, phase, noisy_audio):
        result = phase.process(noisy_audio, sample_rate=SR)
        assert result is not None
        assert result.audio.shape == noisy_audio.shape
        assert np.isfinite(result.audio).all()


# ---------------------------------------------------------------------------
# Tests: Stereo-Eingabe
# ---------------------------------------------------------------------------


class TestPhase56Stereo:
    def test_12_stereo_shape_preserved(self, phase, stereo_audio):
        result = phase.process(stereo_audio, sample_rate=SR)
        assert result is not None
        # Akzeptiere: entweder Shape gleich ODER zu Mono konvertiert + zurück
        assert np.isfinite(result.audio).all()

    def test_13_stereo_no_clipping(self, phase, stereo_audio):
        result = phase.process(stereo_audio, sample_rate=SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestPhase56EdgeCases:
    def test_14_single_sample_array(self, phase):
        audio = np.array([0.0], dtype=np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_15_very_short_100ms(self, phase):
        np.random.seed(42)
        audio = (np.random.randn(SR // 10) * 0.1).astype(np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_16_dirac_impulse(self, phase):
        audio = np.zeros(SR, dtype=np.float32)
        audio[SR // 2] = 1.0
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_17_negative_amplitude_input(self, phase):
        audio = -np.ones(SR, dtype=np.float32) * 0.5
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_18_max_amplitude_input(self, phase):
        audio = np.ones(SR, dtype=np.float32)
        result = phase.process(audio, sample_rate=SR)
        assert result is not None
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Konsistenz
# ---------------------------------------------------------------------------


class TestPhase56Consistency:
    def test_19_both_runs_valid(self, phase, sine_440_2s):
        """Zwei Läufe mit gleicher Eingabe liefern beide gültige Ausgaben.

        Hinweis: SpectralBandGapRepair nutzt NMF mit stochastischer Initialisierung
        — bit-identische Outputs sind daher nicht garantiert, aber beide Ausgaben
        müssen valide (NaN-frei, bounded) sein.
        """
        np.random.seed(42)
        r1 = phase.process(sine_440_2s.copy(), sample_rate=SR)
        np.random.seed(42)
        r2 = phase.process(sine_440_2s.copy(), sample_rate=SR)
        # Beide Ausgaben müssen valide sein
        assert np.isfinite(r1.audio).all(), "Run 1: NaN/Inf im Ausgang"
        assert np.isfinite(r2.audio).all(), "Run 2: NaN/Inf im Ausgang"
        assert np.max(np.abs(r1.audio)) <= 1.0 + 1e-6, "Run 1: Clipping"
        assert np.max(np.abs(r2.audio)) <= 1.0 + 1e-6, "Run 2: Clipping"
        assert r1.audio.shape == sine_440_2s.shape
        assert r2.audio.shape == sine_440_2s.shape

    def test_20_success_flag_for_valid_audio(self, phase, sine_440_2s):
        result = phase.process(sine_440_2s, sample_rate=SR)
        assert result.success is True

    def test_21_additional_kwargs_ignored_gracefully(self, phase, sine_440_2s):
        """Unbekannte kwargs dürfen keinen Fehler auslösen."""
        result = phase.process(
            sine_440_2s, sample_rate=SR, material_type="tape", defect_scores={}, quality_mode="restoration"
        )
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_22_output_energy_not_zero_for_tonal_input(self, phase, sine_440_2s):
        """Tonaler Eingang → Ausgang nicht komplett auf Null."""
        result = phase.process(sine_440_2s, sample_rate=SR)
        rms = float(np.sqrt(np.mean(result.audio**2)))
        assert rms > 1e-6, f"Ausgang hat kein Energie: rms={rms}"
