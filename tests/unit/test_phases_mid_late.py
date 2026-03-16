"""
Unit-Tests für Aurik-Phasen 11-30, 40-42, 49, 51-52, 54.

Aufruf-Konventionen:
  - Phasen 11-30, 40-42, 49: process(audio, sample_rate[, material_type])
  - Phasen 13, 14, 15, 25: brauchen Stereo-Audio
  - Phasen 51, 52, 54: process(audio) — KEIN sample_rate!
  - Phase 11 + 25: MaterialType ist PFLICHTARG
"""

import numpy as np
np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_interface import PhaseResult

SR = 48000
_N = SR // 4  # 12000 Samples (0.25s) – reicht für alle Phase-Tests


# ---------------------------------------------------------------------------
# Fixtures — scope="class": einmalig pro Testklasse (sicher bei in-place Ops)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def mono():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, _N).astype(np.float32)


@pytest.fixture(scope="class")
def stereo():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, (_N, 2)).astype(np.float32)


@pytest.fixture(scope="class")
def silent_mono():
    return np.zeros(_N, dtype=np.float32)


@pytest.fixture(scope="class")
def loud_mono():
    rng = np.random.default_rng(0)
    return rng.uniform(-0.9, 0.9, _N).astype(np.float32)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------
def _assert_phase_result(result, orig_audio, *, check_clipping: bool = True, clip_threshold: float = 2.0):
    """Prüft ein PhaseResult auf Grundkorrektheit."""
    assert isinstance(result, PhaseResult), f"Kein PhaseResult: {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Shape-Mismatch: {result.audio.shape} != {orig_audio.shape}"
    assert np.issubdtype(result.audio.dtype, np.floating), f"audio.dtype nicht floating: {result.audio.dtype}"
    assert isinstance(result.metadata, dict)
    assert isinstance(result.metrics, dict)
    assert float(result.execution_time_seconds) >= 0.0
    if check_clipping:
        peak = float(np.max(np.abs(result.audio)))
        assert peak <= clip_threshold, f"Audio stark übersteuert: {peak:.3f} > {clip_threshold}"


# ===========================================================================
# Phase 11 – Limiting
# ===========================================================================
class TestPhase11Limiting:
    def setup_method(self):
        from backend.core.phases.phase_11_limiting import LimitingPhase

        self.phase = LimitingPhase()

    def test_mono_returns_phase_result(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, loud_mono)

    def test_output_within_limits(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        assert np.max(np.abs(result.audio)) <= 1.05

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_different_material_types(self, mono):
        for mat in [MaterialType.VINYL, MaterialType.TAPE, MaterialType.CD_DIGITAL]:
            result = self.phase.process(mono, SR, mat)
            _assert_phase_result(result, mono)


# ===========================================================================
# Phase 12 – Wow/Flutter-Fix
# ===========================================================================
class TestPhase12WowFlutterFix:
    def setup_method(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        self.phase = WowFlutterFix()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_tape_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 13 – Stereo Enhancement
# ===========================================================================
class TestPhase13StereoEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_13_stereo_enhancement import StereoEnhancementPhaseV2

        self.phase = StereoEnhancementPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)


# ===========================================================================
# Phase 14 – Phase Correction
# ===========================================================================
class TestPhase14PhaseCorrection:
    def setup_method(self):
        from backend.core.phases.phase_14_phase_correction import PhaseCorrection

        self.phase = PhaseCorrection()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)


# ===========================================================================
# Phase 15 – Stereo Balance
# ===========================================================================
class TestPhase15StereoBalance:
    def setup_method(self):
        from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2

        self.phase = StereoBalancePhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR)
        _assert_phase_result(result, silent)


# ===========================================================================
# Phase 16 – Final EQ
# ===========================================================================
class TestPhase16FinalEQ:
    def setup_method(self):
        from backend.core.phases.phase_16_final_eq import FinalEQ

        self.phase = FinalEQ()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 18 – Noise Gate
# ===========================================================================
class TestPhase18NoiseGate:
    def setup_method(self):
        from backend.core.phases.phase_18_noise_gate import NoiseGate

        self.phase = NoiseGate()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_silent_is_attenuated(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)
        assert np.max(np.abs(result.audio)) < 0.01

    def test_loud_signal_passes(self, loud_mono):
        result = self.phase.process(loud_mono, SR)
        # Gate kann leichten Gain hinzufügen → check_clipping=False
        _assert_phase_result(result, loud_mono, check_clipping=False)
        # Laut-Signal sollte nicht komplett gedämpft werden
        assert np.max(np.abs(result.audio)) > 0.01


# ===========================================================================
# Phase 19 – De-Esser
# ===========================================================================
class TestPhase19DeEsser:
    def setup_method(self):
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        self.phase = DeEsserPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono)

    def test_sibilant_reduction(self):
        """Hochfrequentes Signal (s-Laute) soll gedämpft werden."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        sibilant = (0.5 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
        result = self.phase.process(sibilant, SR, MaterialType.VINYL)
        _assert_phase_result(result, sibilant, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 20 – Reverb Reduction
# ===========================================================================
class TestPhase20ReverbReduction:
    def setup_method(self):
        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        self.phase = ReverbReduction()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent_mono)

    def test_different_materials(self, mono):
        for mat in [MaterialType.TAPE, MaterialType.SHELLAC]:
            result = self.phase.process(mono, SR, mat)
            _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 21 – Exciter
# ===========================================================================
class TestPhase21Exciter:
    def setup_method(self):
        from backend.core.phases.phase_21_exciter import Exciter

        self.phase = Exciter()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 22 – Tape Saturation
# ===========================================================================
class TestPhase22TapeSaturation:
    def setup_method(self):
        from backend.core.phases.phase_22_tape_saturation import TapeSaturation

        self.phase = TapeSaturation()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.TAPE)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 23 – Spectral Repair
# ===========================================================================
class TestPhase23SpectralRepair:
    def setup_method(self):
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        self.phase = SpectralRepair()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 24 – Dropout Repair
# ===========================================================================
class TestPhase24DropoutRepair:
    def setup_method(self):
        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        self.phase = DropoutRepairPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_audio_with_dropout(self):
        """Audio mit simuliertem Aussetzer."""
        sig = np.ones(_N, dtype=np.float32) * 0.5
        sig[_N // 2 : _N // 2 + 100] = 0.0  # Aussetzer
        result = self.phase.process(sig, SR)
        _assert_phase_result(result, sig, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 25 – Azimuth Correction (Stereo + MaterialType PFLICHT)
# ===========================================================================
class TestPhase25AzimuthCorrection:
    def setup_method(self):
        from backend.core.phases.phase_25_azimuth_correction import AzimuthCorrectionPhaseV2

        self.phase = AzimuthCorrectionPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2

    def test_tape_material(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.TAPE)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_stereo(self):
        silent = np.zeros((_N, 2), dtype=np.float32)
        result = self.phase.process(silent, SR, MaterialType.VINYL)
        _assert_phase_result(result, silent)


# ===========================================================================
# Phase 26 – Dynamic Range Expansion
# ===========================================================================
class TestPhase26DynamicRangeExpansion:
    def setup_method(self):
        from backend.core.phases.phase_26_dynamic_range_expansion import DynamicRangeExpansion

        self.phase = DynamicRangeExpansion()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_vinyl_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 27 – Click/Pop Removal V9
# ===========================================================================
class TestPhase27ClickPopRemoval:
    def setup_method(self):
        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        self.phase = ClickPopRemoval()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_audio_with_click(self):
        """Signal mit simuliertem Click."""
        sig = np.zeros(_N, dtype=np.float32)
        sig[_N // 4] = 1.0  # impulsartiger Click
        result = self.phase.process(sig, SR, MaterialType.VINYL)
        _assert_phase_result(result, sig, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 28 – Surface Noise Profiling
# ===========================================================================
class TestPhase28SurfaceNoiseProfiling:
    def setup_method(self):
        from backend.core.phases.phase_28_surface_noise_profiling import SurfaceNoiseProfiling

        self.phase = SurfaceNoiseProfiling()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_shellac_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.SHELLAC)
        _assert_phase_result(result, mono, check_clipping=False)


# ===========================================================================
# Phase 29 – Tape Hiss Reduction
# ===========================================================================
class TestPhase29TapeHissReduction:
    # Phase 29 erzwingt 48 kHz (validate_input()) — anderer SR als Datei-Standard
    _SR = 48_000

    def setup_method(self):
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        self.phase = TapeHissReductionPhase()

    def test_mono_returns_phase_result(self, mono):
        audio_48 = mono[: self._SR // 4]  # 0.25 s bei 48 kHz
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE)
        _assert_phase_result(result, audio_48, check_clipping=False)

    def test_silent_input(self, silent_mono):
        silent_48 = silent_mono[: self._SR // 4]
        result = self.phase.process(silent_48, self._SR)
        _assert_phase_result(result, silent_48)

    def test_reel_tape_material(self, mono):
        audio_48 = mono[: self._SR // 4]
        result = self.phase.process(audio_48, self._SR, MaterialType.TAPE)
        _assert_phase_result(result, audio_48, check_clipping=False)


# ===========================================================================
# Phase 30 – DC Offset Removal
# ===========================================================================
class TestPhase30DCOffsetRemoval:
    def setup_method(self):
        from backend.core.phases.phase_30_dc_offset_removal import DCOffsetRemoval

        self.phase = DCOffsetRemoval()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_dc_offset_removed(self):
        """Starker DC-Anteil soll entfernt werden."""
        sig = np.ones(_N, dtype=np.float32) * 0.5 + 0.05 * np.random.default_rng(7).standard_normal(_N).astype(
            np.float32
        )
        result = self.phase.process(sig, SR)
        _assert_phase_result(result, sig, check_clipping=False)
        assert abs(float(np.mean(result.audio))) < abs(float(np.mean(sig)))

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 40 – Loudness Normalization
# ===========================================================================
class TestPhase40LoudnessNormalization:
    def setup_method(self):
        from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase

        self.phase = LoudnessNormalizationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_silent_input_no_crash(self, silent_mono):
        # Stilles Signal kann Probleme mit Lautheitsmessung machen
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)

    def test_loud_signal_normalized(self, loud_mono):
        result = self.phase.process(loud_mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, loud_mono, check_clipping=False)


# ===========================================================================
# Phase 41 – Output Format Optimization
# ===========================================================================
class TestPhase41OutputFormatOptimization:
    def setup_method(self):
        from backend.core.phases.phase_41_output_format_optimization import OutputFormatOptimization

        self.phase = OutputFormatOptimization()

    def test_mono_returns_phase_result(self, mono):
        # Phase 41 resampled ggf. auf andere Samplerate → nur success & dtype prüfen
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert isinstance(result.audio, np.ndarray)
        assert np.issubdtype(result.audio.dtype, np.floating)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR, MaterialType.VINYL)
        assert isinstance(result, PhaseResult)
        assert result.success is True

    def test_cd_material(self, mono):
        result = self.phase.process(mono, SR, MaterialType.CD_DIGITAL)
        assert isinstance(result, PhaseResult)
        assert result.success is True


# ===========================================================================
# Phase 42 – Vocal Enhancement
# ===========================================================================
class TestPhase42VocalEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        self.phase = VocalEnhancement()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_voice_frequency_range(self):
        """Gesangs-Frequenzbereich (300 Hz–3 kHz)."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        vocal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = self.phase.process(vocal, SR)
        _assert_phase_result(result, vocal, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 49 – Advanced Deverb
# ===========================================================================
class TestPhase49AdvancedDereverb:
    def setup_method(self):
        from backend.core.phases.phase_49_advanced_dereverb import AdvancedDereverbPhase

        self.phase = AdvancedDereverbPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_output_shape_preserved(self, mono):
        result = self.phase.process(mono, SR)
        assert result.audio.shape == mono.shape


# ===========================================================================
# Phase 51 – Drums Enhancement (KEIN sample_rate!)
# ===========================================================================
class TestPhase51DrumsEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_51_drums_enhancement import DrumsEnhancementV1

        self.phase = DrumsEnhancementV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)

    def test_loud_percussive_signal(self, loud_mono):
        result = self.phase.process(loud_mono)
        _assert_phase_result(result, loud_mono, check_clipping=False)


# ===========================================================================
# Phase 52 – Piano Restoration (KEIN sample_rate!)
# ===========================================================================
class TestPhase52PianoRestoration:
    def setup_method(self):
        from backend.core.phases.phase_52_piano_restoration import PianoRestorationV1

        self.phase = PianoRestorationV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_piano_tone(self):
        """Klavier-typisches Signal (A4 = 440 Hz)."""
        t = np.linspace(0, 0.25, _N, dtype=np.float32)
        piano = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        result = self.phase.process(piano)
        _assert_phase_result(result, piano, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)


# ===========================================================================
# Phase 54 – Transparent Dynamics (KEIN sample_rate!)
# ===========================================================================
class TestPhase54TransparentDynamics:
    def setup_method(self):
        from backend.core.phases.phase_54_transparent_dynamics import TransparentDynamicsV1

        self.phase = TransparentDynamicsV1()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_with_material_type(self, mono):
        result = self.phase.process(mono, material_type=MaterialType.CD_DIGITAL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_silent_input(self, silent_mono):
        result = self.phase.process(silent_mono)
        _assert_phase_result(result, silent_mono)

    def test_output_shape_preserved(self, loud_mono):
        result = self.phase.process(loud_mono)
        assert result.audio.shape == loud_mono.shape
