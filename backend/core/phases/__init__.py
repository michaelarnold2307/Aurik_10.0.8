"""
Aurik 9.0 - Phase Module Exports
==================================

Central import hub for all 42 processing phases.

Usage:
    from backend.core.phases import ClickRemovalPhase, DenoisePhase
    from backend.core.phases import PhaseInterface, PhaseCategory, PhaseMetadata

Author: Aurik 9.0 Development Team
Version: 9.0.0
Date: 2026-02-15
"""

# Phase 01-09: Defect Removal & Basic Restoration
from .phase_01_click_removal import ClickRemovalPhase
from .phase_02_hum_removal import HumRemovalPhase
from .phase_03_denoise import DenoisePhase
from .phase_04_eq_correction import EQCorrectionPhase
from .phase_05_rumble_filter import RumbleFilterPhase
from .phase_06_frequency_restoration import FrequencyRestorationPhase
from .phase_07_harmonic_restoration import HarmonicRestorationPhase
from .phase_08_transient_preservation import TransientPreservationPhase
from .phase_09_crackle_removal import CrackleRemovalPhase

# Phase 10-19: Dynamics & Spatial Processing
from .phase_10_compression import CompressionPhase
from .phase_11_limiting import LimitingPhase
from .phase_12_wow_flutter_fix import WowFlutterFix
from .phase_13_stereo_enhancement import StereoEnhancementPhaseV2
from .phase_14_phase_correction import PhaseCorrection
from .phase_15_stereo_balance import StereoBalancePhaseV2
from .phase_16_final_eq import FinalEQ
from .phase_17_mastering_polish import MasteringPolishPhase
from .phase_18_noise_gate import NoiseGate
from .phase_19_de_esser import DeEsserPhase

# Phase 20-29: Advanced Restoration & Material-Specific
from .phase_20_reverb_reduction import ReverbReduction
from .phase_21_exciter import Exciter
from .phase_22_tape_saturation import TapeSaturation
from .phase_23_spectral_repair import SpectralRepair
from .phase_24_dropout_repair import DropoutRepairPhase
from .phase_25_azimuth_correction import AzimuthCorrectionPhaseV2
from .phase_26_dynamic_range_expansion import DynamicRangeExpansion
from .phase_27_click_pop_removal import ClickPopRemoval
from .phase_28_surface_noise_profiling import SurfaceNoiseProfiling
from .phase_29_tape_hiss_reduction import TapeHissReductionPhase

# Phase 30-39: Format & Enhancement
from .phase_30_dc_offset_removal import DCOffsetRemoval
from .phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase
from .phase_32_mono_to_stereo import MonoToStereoPhaseV2
from .phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2
from .phase_34_mid_side_processing import MidSideProcessing
from .phase_35_multiband_compression import MultibandCompressionPhase
from .phase_36_transient_shaper import TransientShaper
from .phase_37_bass_enhancement import BassEnhancement
from .phase_38_presence_boost import PresenceBoost
from .phase_39_air_band_enhancement import AirBandEnhancement

# Phase 40-42: Final Output Processing
from .phase_40_loudness_normalization import LoudnessNormalizationPhase
from .phase_41_output_format_optimization import OutputFormatOptimization
from .phase_42_vocal_enhancement import VocalEnhancement
from .phase_43_ml_deesser import MLDeEsserPhase
from .phase_44_guitar_enhancement import GuitarEnhancementPhase
from .phase_45_brass_enhancement import BrassEnhancementPhase
from .phase_46_spatial_enhancement import SpatialEnhancementPhase
from .phase_47_truepeak_limiter import TruePeakLimiterPhase
from .phase_48_stereo_width_enhancer import StereoWidthEnhancerPhase
from .phase_49_advanced_dereverb import AdvancedDereverbPhase
from .phase_50_spectral_repair import SpectralRepairPhase

# Phase 50+: Tier 1 ML-Hybrid Enhancement (NEW)
from .phase_51_drums_enhancement import DrumsEnhancementV1
from .phase_52_piano_restoration import PianoRestorationV1
from .phase_54_transparent_dynamics import TransparentDynamicsV1
from .phase_55_diffusion_inpainting import DiffusionInpaintingPhase

# Phase Interface & Base Classes
from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)

# Phase 56: HEAD_WEAR Spectral Band Gap Repair (v9.9.8)
try:
    from .phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase

    _PHASE56_OK = True
except ImportError as _p56_err:
    import logging as _logging

    _logging.getLogger(__name__).debug("Phase 56 nicht verfügbar: %s", _p56_err)
    _PHASE56_OK = False
    SpectralBandGapRepairPhase = None  # type: ignore[assignment,misc]

# Exported symbols
__all__ = [
    # Base Classes
    "PhaseInterface",
    "PhaseCategory",
    "PhaseMetadata",
    "PhaseResult",
    # Phase 01-09
    "ClickRemovalPhase",
    "HumRemovalPhase",
    "DenoisePhase",
    "EQCorrectionPhase",
    "RumbleFilterPhase",
    "FrequencyRestorationPhase",
    "HarmonicRestorationPhase",
    "TransientPreservationPhase",
    "CrackleRemovalPhase",
    # Phase 10-19
    "CompressionPhase",
    "LimitingPhase",
    "WowFlutterFix",
    "StereoEnhancementPhaseV2",
    "PhaseCorrection",
    "StereoBalancePhaseV2",
    "FinalEQ",
    "MasteringPolishPhase",
    "NoiseGate",
    "DeEsserPhase",
    # Phase 20-29
    "ReverbReduction",
    "Exciter",
    "TapeSaturation",
    "SpectralRepair",
    "DropoutRepairPhase",
    "AzimuthCorrectionPhaseV2",
    "DynamicRangeExpansion",
    "ClickPopRemoval",
    "SurfaceNoiseProfiling",
    "TapeHissReductionPhase",
    # Phase 30-39
    "DCOffsetRemoval",
    "SpeedPitchCorrectionPhase",
    "MonoToStereoPhaseV2",
    "StereoWidthLimiterPhaseV2",
    "MidSideProcessing",
    "MultibandCompressionPhase",
    "TransientShaper",
    "BassEnhancement",
    "PresenceBoost",
    "AirBandEnhancement",
    # Phase 40-48
    "LoudnessNormalizationPhase",
    "OutputFormatOptimization",
    "VocalEnhancement",
    "MLDeEsserPhase",
    "GuitarEnhancementPhase",
    "BrassEnhancementPhase",
    "SpatialEnhancementPhase",
    "TruePeakLimiterPhase",
    "StereoWidthEnhancerPhase",
    "AdvancedDereverbPhase",
    # Phase 50+ (Tier 1 ML-Hybrid)
    "SpectralRepairPhase",
    "DrumsEnhancementV1",
    "PianoRestorationV1",
    "TransparentDynamicsV1",
    "DiffusionInpaintingPhase",
    "SpectralBandGapRepairPhase",
]

# Version info
__version__ = "9.0.0"
__author__ = "Aurik 9.0 Development Team"
__date__ = "2026-02-15"
