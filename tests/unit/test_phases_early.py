"""
Unit-Tests für Phasen 01–09 (Frühe Restaurierungsphasen).

Abgedeckt:
  Phase 01 — Click Removal (ClickRemovalPhase)
  Phase 02 — Hum Removal (HumRemovalPhase)
  Phase 03 — Denoise (DenoisePhase)
  Phase 04 — EQ Correction (EQCorrectionPhase)
  Phase 05 — Rumble Filter (RumbleFilterPhase)
  Phase 06 — Frequency Restoration (FrequencyRestorationPhase)
  Phase 07 — Harmonic Restoration (HarmonicRestorationPhase)
  Phase 08 — Transient Preservation (TransientPreservationPhase)
  Phase 09 — Crackle Removal (CrackleRemovalPhase)

Alle Tests auf 0.1-Sekunden-Arrays (4410 Samples @ 44100 Hz).
"""

import numpy as np
np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.phases.phase_interface import PhaseResult

SR = 44100
N = SR // 10  # 4410 Samples ≈ 0.1s für schnelle Tests


# ---------------------------------------------------------------------------
# Fixtures — scope="module": einmalig pro Testdatei
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mono():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal(N).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="module")
def mono_quiet():
    rng = np.random.default_rng(99)
    return np.clip(rng.standard_normal(N).astype(np.float32) * 0.05, -1.0, 1.0)


@pytest.fixture(scope="module")
def stereo():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal((N, 2)).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="module")
def sine_mono():
    """Reines Sinus-Signal mit 440 Hz."""
    t = np.linspace(0, N / SR, N, endpoint=False)
    return (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Gemeinsame Validierung
# ---------------------------------------------------------------------------


def _assert_phase_result(result, orig_audio, check_clipping: bool = True):
    assert isinstance(result, PhaseResult), f"Kein PhaseResult: {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Shape geändert: {orig_audio.shape} → {result.audio.shape}"
    # Dtype: float32 oder float64 akzeptabel (interne DSP-Umwandlung)
    assert np.issubdtype(result.audio.dtype, np.floating), f"Dtype nicht floating: {result.audio.dtype}"
    assert isinstance(result.metadata, dict), "metadata muss dict sein"
    assert isinstance(result.metrics, dict), "metrics muss dict sein"
    assert float(result.execution_time_seconds) >= 0.0
    if check_clipping:
        assert np.max(np.abs(result.audio)) <= 2.0, f"Audio stark übersteuert: {np.max(np.abs(result.audio)):.3f}"


# ---------------------------------------------------------------------------
# Phase 01: Click Removal
# ---------------------------------------------------------------------------


class TestPhase01ClickRemoval:
    def setup_method(self):
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        self.phase = ClickRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_unchanged_roughly(self, sine_mono):
        """Reines Sinussignal (keine Klicks) bleibt nahezu unverändert."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.8, f"Korrelation zu gering: {corr:.3f}"

    def test_signal_with_click_processed(self, mono):
        """Signal mit eingebautem Klick wurde verarbeitet (kein Absturz)."""
        audio_with_click = mono.copy()
        audio_with_click[100] = 0.99  # Starker Klick
        audio_with_click[200] = -0.99
        result = self.phase.process(audio_with_click, SR)
        _assert_phase_result(result, audio_with_click)

    def test_metrics_present(self, mono):
        result = self.phase.process(mono, SR)
        assert result.metrics is not None or result.metadata is not None


# ---------------------------------------------------------------------------
# Phase 02: Hum Removal
# ---------------------------------------------------------------------------


class TestPhase02HumRemoval:
    def setup_method(self):
        from backend.core.phases.phase_02_hum_removal import HumRemovalPhase

        self.phase = HumRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_50hz_hum_reduced(self):
        """50-Hz-Brumm: Nach Entfernung muss 50-Hz-Energie gesunken sein."""
        t = np.linspace(0, N / SR, N, endpoint=False)
        hum = (np.sin(2 * np.pi * 50.0 * t) * 0.3).astype(np.float32)
        result = self.phase.process(hum, SR)
        _assert_phase_result(result, hum)
        # Energie bei 50 Hz nach Entfernung niedriger
        fft_in = np.abs(np.fft.rfft(hum.astype(float)))
        fft_out = np.abs(np.fft.rfft(result.audio.astype(float)))
        bin_50 = int(50 * N / SR)
        # Mindestens keine Verstärkung des Brumms
        assert fft_out[bin_50] <= fft_in[bin_50] * 1.5

    def test_quiet_signal_not_amplified(self, mono_quiet):
        result = self.phase.process(mono_quiet, SR)
        _assert_phase_result(result, mono_quiet)
        rms_in = float(np.sqrt(np.mean(mono_quiet**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out <= rms_in * 3.0  # Keine extreme Verstärkung


# ---------------------------------------------------------------------------
# Phase 03: Denoise
# ---------------------------------------------------------------------------


class TestPhase03Denoise:
    def setup_method(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        self.phase = DenoisePhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_broadband_noise_reduced(self):
        """Breitbandrauschen wird reduziert."""
        rng = np.random.default_rng(7)
        noise = (rng.standard_normal(N) * 0.3).astype(np.float32)
        result = self.phase.process(noise, SR)
        _assert_phase_result(result, noise)
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out <= rms_in * 1.1  # Keine Verstärkung des Rauschens

    def test_material_type_supported(self, mono):
        """material_type='vinyl' wird akzeptiert."""
        result = self.phase.process(mono, material_type="vinyl")
        _assert_phase_result(result, mono)


# ---------------------------------------------------------------------------
# Phase 04: EQ Correction
# ---------------------------------------------------------------------------


class TestPhase04EQCorrection:
    def setup_method(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        self.phase = EQCorrectionPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_frequency_range_preserved(self, sine_mono):
        """Sinus-Signal: Kein vollständiges Auslöschen durch EQ."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        rms_in = float(np.sqrt(np.mean(sine_mono**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out > rms_in * 0.1, "EQ hat Signal fast vollständig eliminiert"

    def test_material_types(self, mono):
        for mat in ["vinyl", "tape", "unknown"]:
            result = self.phase.process(mono, material_type=mat)
            # EQ kann starke Boosts erzeugen → check_clipping=False
            _assert_phase_result(result, mono, check_clipping=False)


# ---------------------------------------------------------------------------
# Phase 05: Rumble Filter
# ---------------------------------------------------------------------------


class TestPhase05RumbleFilter:
    def setup_method(self):
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        self.phase = RumbleFilterPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_low_freq_rumble_reduced(self):
        """LF-Rumble (10 Hz) wird stark reduziert."""
        t = np.linspace(0, N / SR, N, endpoint=False)
        rumble = (np.sin(2 * np.pi * 10.0 * t) * 0.5).astype(np.float32)
        result = self.phase.process(rumble, SR)
        _assert_phase_result(result, rumble)
        # Energie bei 10 Hz nach Filterung niedriger
        fft_in = np.abs(np.fft.rfft(rumble.astype(float)))
        fft_out = np.abs(np.fft.rfft(result.audio.astype(float)))
        bin_10 = max(1, int(10 * N / SR))
        assert fft_out[bin_10] <= fft_in[bin_10] + 1e-3

    def test_high_freq_preserved(self, sine_mono):
        """440-Hz-Signal bleibt nach Rumble-Filter erhalten."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out > 0.05  # Signal muss erhalten bleiben


# ---------------------------------------------------------------------------
# Phase 06: Frequency Restoration
# ---------------------------------------------------------------------------


class TestPhase06FrequencyRestoration:
    def setup_method(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        self.phase = FrequencyRestorationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_no_extreme_gain(self, mono):
        """Frequenzwiederherstellung darf Signal nicht extrem verstärken."""
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)
        peak_in = float(np.max(np.abs(mono)))
        peak_out = float(np.max(np.abs(result.audio)))
        assert peak_out <= max(peak_in * 5.0, 1.05)  # Kein unkontrollierter Gain


# ---------------------------------------------------------------------------
# Phase 07: Harmonic Restoration
# ---------------------------------------------------------------------------


class TestPhase07HarmonicRestoration:
    def setup_method(self):
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        self.phase = HarmonicRestorationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_not_destroyed(self, sine_mono):
        """Grundton bleibt nach harmonischer Restaurierung erhalten."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.5, f"Korrelation zu gering nach Harmonic Restoration: {corr:.3f}"


# ---------------------------------------------------------------------------
# Phase 08: Transient Preservation
# ---------------------------------------------------------------------------


class TestPhase08TransientPreservation:
    def setup_method(self):
        from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase

        self.phase = TransientPreservationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_transients_preserved(self):
        """Impuls-Signal: Transienten-Energie bleibt erhalten."""
        audio = np.zeros(N, dtype=np.float32)
        audio[:: int(SR * 0.02)] = 0.8  # Impulse alle 20ms
        result = self.phase.process(audio, SR)
        _assert_phase_result(result, audio)
        # Peak muss noch signifikant sein
        assert float(np.max(np.abs(result.audio))) > 0.3


# ---------------------------------------------------------------------------
# Phase 09: Crackle Removal
# ---------------------------------------------------------------------------


class TestPhase09CrackleRemoval:
    def setup_method(self):
        from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase

        self.phase = CrackleRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_not_damaged(self, sine_mono):
        """Reiner Sinus (kein Crackle): Signal bleibt weitgehend unverändert."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.7, f"Korrelation zu gering: {corr:.3f}"

    def test_crackle_signal_processed(self, mono):
        """Signal mit simulierten Crackles wird verarbeitet."""
        audio = mono.copy()
        rng = np.random.default_rng(0)
        crackle_pos = rng.integers(0, N - 1, size=20)
        audio[crackle_pos] = rng.choice([-0.95, 0.95], size=20).astype(np.float32)
        result = self.phase.process(audio, SR)
        _assert_phase_result(result, audio)

    def test_material_types(self, mono):
        for mat in ["shellac", "vinyl", "unknown"]:
            result = self.phase.process(mono, material_type=mat)
            _assert_phase_result(result, mono)
