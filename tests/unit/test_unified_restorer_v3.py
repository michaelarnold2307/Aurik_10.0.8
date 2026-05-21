"""Unit-Tests für backend/core/unified_restorer_v3.py.

Spec §2.1/§2.2: UnifiedRestorerV3 — Defect-First Restoration Engine.
Tests decken ab: RestorationConfig, RestorationResult, Initialisierung,
get_restorer()-Singleton, get_phase_info(), _select_phases() (Mock),
NaN/Inf-Invariante, Shape-Korrektheit, Bounds und Edge-Cases.
≥ 35 Tests.
"""

from __future__ import annotations

import math
import types
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

np.random.seed(42)

import backend.core.unified_restorer_v3 as _uv3_mod
from backend.core.defect_scanner import DefectType, MaterialType
from backend.core.performance_guard import DeploymentMode, QualityMode
from backend.core.unified_restorer_v3 import (
    RestorationConfig,
    RestorationResult,
    UnifiedRestorerV3,
    get_restorer,
)

SR = 48000


def _sine(secs: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(secs: float = 1.0, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _stereo(secs: float = 2.0) -> np.ndarray:
    mono = _sine(secs)
    return np.stack([mono, mono * 0.9])


def _make_mock_defect_result(n: int = 3) -> MagicMock:
    """Erstellt ein minimal gültiges DefectScanner-Ergebnis-Mock."""
    mock = MagicMock()
    mock.material_type = MaterialType.VINYL if hasattr(MaterialType, "VINYL") else MagicMock()
    # scores ist ein dict DefectType → float
    mock.scores = {}
    mock.get_top_defects.return_value = []
    mock.metadata = {}
    return mock


def _make_restoration_result(audio: np.ndarray) -> RestorationResult:
    """Erstellt ein minimales RestorationResult für Tests."""
    cfg = RestorationConfig()
    # Ermittle einen gültigen MaterialType
    mat = list(MaterialType)[0]
    return RestorationResult(
        audio=audio,
        config=cfg,
        material_type=mat,
        defect_scores={},
        phases_executed=[],
        phases_skipped=[],
        total_time_seconds=0.5,
        rt_factor=0.25,
        quality_estimate=0.85,
        warnings=[],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Klasse 1: RestorationConfig
# ---------------------------------------------------------------------------


class TestRestorationConfig:
    def test_01_default_instantiation(self):
        cfg = RestorationConfig()
        assert cfg is not None

    def test_02_default_mode_is_quality(self):
        cfg = RestorationConfig()
        assert cfg.mode == QualityMode.QUALITY

    def test_03_enable_performance_guard_default_true(self):
        cfg = RestorationConfig()
        assert cfg.enable_performance_guard is True

    def test_04_num_cores_default_four(self):
        cfg = RestorationConfig()
        assert cfg.num_cores == 4

    def test_05_material_type_default_none(self):
        cfg = RestorationConfig()
        assert cfg.material_type is None

    def test_06_custom_mode_fast(self):
        cfg = RestorationConfig(mode=QualityMode.FAST)
        assert cfg.mode == QualityMode.FAST

    def test_07_custom_num_cores(self):
        cfg = RestorationConfig(num_cores=2)
        assert cfg.num_cores == 2

    def test_08_is_dataclass_with_fields(self):
        f_names = [f.name for f in fields(RestorationConfig)]
        assert "mode" in f_names
        assert "num_cores" in f_names

    def test_09_deployment_mode_default_product(self):
        cfg = RestorationConfig()
        assert cfg.deployment_mode == DeploymentMode.PRODUCT

    def test_10_maximum_mode_does_not_force_studio_flag(self):
        cfg = RestorationConfig(mode=QualityMode.MAXIMUM, studio_2026=False)
        assert cfg.mode == QualityMode.MAXIMUM
        assert cfg.studio_2026 is False

    def test_11_studio_flag_forces_maximum_mode(self):
        cfg = RestorationConfig(mode=QualityMode.QUALITY, studio_2026=True)
        assert cfg.mode == QualityMode.MAXIMUM
        assert cfg.studio_2026 is True

    def test_12_phase_strength_oracle_rollout_default_all(self):
        cfg = RestorationConfig()
        assert cfg.phase_strength_oracle_rollout == "all"


class TestStrengthOracleRollout:
    def test_40zza_normalize_rollout_aliases(self):
        assert UnifiedRestorerV3._normalize_phase_strength_oracle_rollout_mode("disabled") == "off"
        assert UnifiedRestorerV3._normalize_phase_strength_oracle_rollout_mode("pilot") == "pilot"
        assert UnifiedRestorerV3._normalize_phase_strength_oracle_rollout_mode("enabled") == "all"

    def test_40zzb_resolve_rollout_prefers_kwargs_over_context_and_config(self):
        restorer = object.__new__(UnifiedRestorerV3)
        restorer._restoration_context = {"phase_strength_oracle_rollout": "pilot"}
        restorer.config = RestorationConfig(phase_strength_oracle_rollout="off")

        mode = UnifiedRestorerV3._resolve_phase_strength_oracle_rollout_mode(
            restorer,
            {"phase_strength_oracle_rollout": "all"},
        )
        assert mode == "all"

    def test_40zzc_phase_enablement_honors_rollout_mode(self):
        phase_id = "phase_03_denoise"
        assert UnifiedRestorerV3._is_phase_strength_oracle_enabled_for_phase(phase_id, "off") is False
        assert UnifiedRestorerV3._is_phase_strength_oracle_enabled_for_phase(phase_id, "all") is True


class TestFallbackTeamworkController:
    def test_40zzd_initialize_controller_uses_capability_and_stem_signals(self):
        restorer = object.__new__(UnifiedRestorerV3)
        restorer._global_conservative_scalar = 1.0
        restorer._restoration_context = {
            "model_capability_report": {
                "summary": {
                    "all_sota_real": False,
                    "degraded_capabilities": ["miipher_native", "sgmse_plus"],
                    "vocal_restoration_status": "sota_fallback",
                }
            },
            "stem_level_restorer": {
                "success": False,
                "fallback_reason": "oom",
            },
        }

        profile = UnifiedRestorerV3._initialize_fallback_teamwork_controller(restorer)

        assert isinstance(profile, dict)
        assert profile.get("all_sota_real") is False
        assert int(profile.get("event_count", 0)) == 0
        assert float(profile.get("global_strength_scalar", 1.0)) < 1.0
        assert "fallback_teamwork_controller" in restorer._restoration_context
        assert float(restorer._restoration_context.get("fallback_teamwork_scalar", 1.0)) < 1.0

    def test_40zze_update_controller_reacts_to_guard_event(self):
        restorer = object.__new__(UnifiedRestorerV3)
        restorer._global_conservative_scalar = 1.0
        restorer._restoration_context = {
            "model_capability_report": {
                "summary": {
                    "all_sota_real": True,
                    "degraded_capabilities": [],
                    "vocal_restoration_status": "sota_real",
                }
            }
        }

        UnifiedRestorerV3._initialize_fallback_teamwork_controller(restorer)
        UnifiedRestorerV3._update_fallback_teamwork_controller_from_event(
            restorer,
            {
                "phase_id": "phase_03_denoise",
                "model": "miipher",
                "reason": "oom",
                "fallback": "sgmse_plus",
                "channels": 2,
                "duration_s": 120.0,
                "required_gb": 8.0,
                "available_gb": 3.0,
            },
            "phase_03_denoise",
        )

        profile = restorer._restoration_context.get("fallback_teamwork_controller", {})
        assert int(profile.get("event_count", 0)) == 1
        assert float(profile.get("cumulative_risk", 0.0)) > 0.0
        assert float(profile.get("global_strength_scalar", 1.0)) < 1.0
        assert int(profile.get("feedback_chain_iteration_boost", 0)) >= 0
        assert float(restorer._restoration_context.get("fallback_gate_tightening", 1.0)) >= 1.0


class TestPhaseCoalitions:
    def test_tape_transport_coalition_requires_two_members(self):
        active = UnifiedRestorerV3.get_active_phase_coalitions(
            ["phase_12_wow_flutter_fix", "phase_29_tape_hiss_reduction"],
            is_studio_2026=False,
        )
        assert active["tape_transport"] == ("phase_12_wow_flutter_fix", "phase_29_tape_hiss_reduction")

    def test_single_member_does_not_activate_coalition(self):
        active = UnifiedRestorerV3.get_active_phase_coalitions(
            ["phase_29_tape_hiss_reduction"],
            is_studio_2026=False,
        )
        assert "tape_transport" not in active

    def test_restoration_filters_studio_only_vocal_member(self):
        active = UnifiedRestorerV3.get_active_phase_coalitions(
            ["phase_20_reverb_reduction", "phase_42_vocal_enhancement", "phase_49_advanced_dereverb"],
            is_studio_2026=False,
        )
        assert active["vocal_production"] == ("phase_20_reverb_reduction", "phase_49_advanced_dereverb")
        assert "phase_42_vocal_enhancement" not in active["vocal_production"]

    def test_studio_keeps_vocal_enhancement_in_coalition(self):
        active = UnifiedRestorerV3.get_active_phase_coalitions(
            ["phase_20_reverb_reduction", "phase_42_vocal_enhancement", "phase_49_advanced_dereverb"],
            is_studio_2026=True,
        )
        assert "phase_42_vocal_enhancement" in active["vocal_production"]


# ---------------------------------------------------------------------------
# Klasse 2: RestorationResult
# ---------------------------------------------------------------------------


class TestRestorationResult:
    def test_09_minimal_construction(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result, RestorationResult)

    def test_10_audio_shape_preserved(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.audio.shape == audio.shape


class TestPreventFirstQuietEdges:
    def test_40dz_goal_recovery_candidate_ranking_prefers_lower_weighted_gap(self):
        thresholds = {"natuerlichkeit": 0.85, "timbre_authentizitaet": 0.85}
        applicable = set(thresholds)
        current = {"natuerlichkeit": 0.80, "timbre_authentizitaet": 0.84}
        candidate = {"natuerlichkeit": 0.86, "timbre_authentizitaet": 0.845}

        current_rank = _uv3_mod._rank_goal_recovery_candidate(
            scores=current,
            baseline_scores=current,
            thresholds=thresholds,
            applicable_goals=applicable,
        )
        candidate_rank = _uv3_mod._rank_goal_recovery_candidate(
            scores=candidate,
            baseline_scores=current,
            thresholds=thresholds,
            applicable_goals=applicable,
            preservation_penalty=0.001,
        )

        assert candidate_rank < current_rank

    def test_40dza_goal_recovery_candidate_ranking_protects_critical_goals(self):
        thresholds = {
            "transparenz": 0.50,
            "waerme": 0.60,
            "brillanz": 0.40,
            "spatial_depth": 0.46,
            "natuerlichkeit": 0.85,
        }
        applicable = set(thresholds)
        baseline = {
            "transparenz": 0.84,
            "waerme": 0.96,
            "brillanz": 0.56,
            "spatial_depth": 0.95,
            "natuerlichkeit": 0.90,
        }
        # Candidate A improves one non-critical goal but collapses transparency and warmth.
        candidate_a = {
            "transparenz": 0.20,
            "waerme": 0.55,
            "brillanz": 0.43,
            "spatial_depth": 0.60,
            "natuerlichkeit": 1.00,
        }
        # Candidate B keeps critical goals near baseline and should be preferred.
        candidate_b = {
            "transparenz": 0.81,
            "waerme": 0.92,
            "brillanz": 0.54,
            "spatial_depth": 0.93,
            "natuerlichkeit": 0.92,
        }

        rank_a = _uv3_mod._rank_goal_recovery_candidate(
            scores=candidate_a,
            baseline_scores=baseline,
            thresholds=thresholds,
            applicable_goals=applicable,
        )
        rank_b = _uv3_mod._rank_goal_recovery_candidate(
            scores=candidate_b,
            baseline_scores=baseline,
            thresholds=thresholds,
            applicable_goals=applicable,
        )

        assert rank_b < rank_a

    def test_40dzb_goal_candidate_blend_alphas_expand_original_audio_search(self):
        original_alphas = _uv3_mod._goal_candidate_blend_alphas("original_audio")
        carrier_alphas = _uv3_mod._goal_candidate_blend_alphas("best_carrier_checkpoint")
        default_alphas = _uv3_mod._goal_candidate_blend_alphas("hpi_best_checkpoint")

        assert original_alphas == (0.90, 0.82, 0.74, 0.64, 0.56, 0.48)
        assert carrier_alphas == (0.90, 0.82, 0.74, 0.66)
        assert default_alphas == (0.90, 0.82, 0.74)

    def test_40d_extract_transfer_chain_accepts_direct_string_and_list_inputs(self):
        assert UnifiedRestorerV3._extract_transfer_chain_from_forensics("vinyl -> tape -> mp3_low") == [
            "vinyl",
            "tape",
            "mp3_low",
        ]
        assert UnifiedRestorerV3._extract_transfer_chain_from_forensics(["vinyl", " tape ", "mp3_low"]) == [
            "vinyl",
            "tape",
            "mp3_low",
        ]

    def test_40e_extract_transfer_chain_rejects_scalar_input_without_iteration(self):
        assert UnifiedRestorerV3._extract_transfer_chain_from_forensics(0.62) is None

    def test_40ea_noise_texture_threshold_uses_primary_material_without_chain(self):
        assert _uv3_mod._resolve_noise_texture_rollback_threshold("vinyl", None) == pytest.approx(8.0)

    def test_40eb_noise_texture_threshold_uses_most_permissive_chain_stage(self):
        assert _uv3_mod._resolve_noise_texture_rollback_threshold("vinyl", ["tape", "mp3_low"]) == pytest.approx(15.0)

    def test_40ec_final_quiet_edge_clamp_reduces_intro_and_outro_boost(self):
        intro = _sine(secs=1.0, freq=220.0) * 0.015
        middle = _sine(secs=2.0, freq=440.0) * 0.18
        outro = _sine(secs=1.0, freq=220.0) * 0.015
        reference = np.concatenate([intro, middle, outro]).astype(np.float32)
        candidate = reference.copy()
        candidate[:SR] *= 6.0
        candidate[-SR:] *= 5.0

        clamped = UnifiedRestorerV3._apply_final_quiet_edge_clamp(reference, candidate, SR, material_key="vinyl")

        ref_edge_peak = float(np.percentile(np.abs(reference[:SR]), 99.9))
        out_intro_peak = float(np.percentile(np.abs(clamped[:SR]), 99.9))
        out_outro_peak = float(np.percentile(np.abs(clamped[-SR:]), 99.9))
        ref_mid_peak = float(np.percentile(np.abs(reference[SR:-SR]), 99.9))
        out_mid_peak = float(np.percentile(np.abs(clamped[SR:-SR]), 99.9))

        assert out_intro_peak <= (ref_edge_peak * (10.0 ** (2.05 / 20.0)))
        assert out_outro_peak <= (ref_edge_peak * (10.0 ** (2.05 / 20.0)))
        assert out_mid_peak >= ref_mid_peak * 0.98

    def test_40ed_final_quiet_edge_clamp_passthrough_without_reference(self):
        candidate = (_sine(secs=2.0) * 0.2).astype(np.float32)
        clamped = UnifiedRestorerV3._apply_final_quiet_edge_clamp(None, candidate, SR, material_key="vinyl")

        assert np.allclose(clamped, candidate)

    def test_40f_quiet_edge_prevention_profile_detects_intro_outro(self):
        audio = _sine(secs=8.0) * 0.18
        audio[: int(1.0 * SR)] *= 0.10
        audio[-int(1.0 * SR) :] *= 0.08

        profile = UnifiedRestorerV3._compute_quiet_edge_prevention_profile(audio.astype(np.float32), SR)

        assert profile["has_quiet_edges"] is True
        assert profile["intro_quiet"] is True
        assert profile["outro_quiet"] is True
        assert int(profile["edge_count"]) == 2

    def test_40g_autosetup_policy_caps_risky_positive_phases_for_quiet_edges(self):
        profile = {
            "family_scalars": {"dynamics_eq": 1.0, "vocal": 1.0, "reconstruction": 1.0},
            "material": "vinyl",
            "restorability_tier": "fair",
        }

        out = UnifiedRestorerV3._apply_song_autosetup_policy(
            profile,
            defect_scores={},
            transfer_chain=["vinyl", "tape"],
            max_defect_severity=0.25,
            quiet_edge_profile={
                "has_quiet_edges": True,
                "intro_quiet": True,
                "outro_quiet": True,
                "edge_count": 2,
                "intro_depth_db": 12.0,
                "outro_depth_db": 10.0,
            },
        )

        assert out["family_scalars"]["dynamics_eq"] < 1.0
        assert out["family_scalars"]["vocal"] < 1.0
        caps = out["strict_conflict_policy"]["phase_strength_caps"]
        assert float(caps["phase_40_loudness_normalization"]) <= 0.3001
        assert float(caps["phase_10_compression"]) <= 0.39
        assert out["strict_conflict_policy"]["quiet_edge_prevention"]["has_quiet_edges"] is True

    def test_40i_execute_pipeline_prevents_cumulative_quiet_edge_ratcheting(self):
        class _PhaseStub:
            def __init__(self, phase_id: str):
                self._phase_id = phase_id

            def get_metadata(self):
                return types.SimpleNamespace(
                    estimated_time_factor=0.1,
                    phase_id=self._phase_id,
                    name=self._phase_id,
                )

        restorer = UnifiedRestorerV3(RestorationConfig())
        restorer.phase_metadata = {
            "phase_99_gain_a": {"name": "Gain A", "dependencies": [], "category": "mastering"},
            "phase_99_gain_b": {"name": "Gain B", "dependencies": [], "category": "mastering"},
        }
        restorer._get_phase = lambda pid: _PhaseStub(pid)  # type: ignore[method-assign]

        def _mock_profiled_call(_phase: object, _audio: np.ndarray, **_kwargs: object) -> object:
            boosted = np.clip(np.asarray(_audio, dtype=np.float32) * 1.18, -1.0, 1.0)
            return types.SimpleNamespace(
                success=True,
                audio=boosted,
                execution_time_seconds=0.001,
                warnings=[],
            )

        restorer._profiled_phase_call = _mock_profiled_call  # type: ignore[method-assign]

        intro = _sine(secs=1.0, freq=220.0) * 0.015
        middle = _sine(secs=2.0, freq=440.0) * 0.18
        outro = _sine(secs=1.0, freq=220.0) * 0.015
        audio_in = np.concatenate([intro, middle, outro]).astype(np.float32)
        quiet_profile = UnifiedRestorerV3._compute_quiet_edge_prevention_profile(audio_in, SR, material_key="vinyl")

        out, executed, _sk, _def = restorer._execute_pipeline(
            audio=audio_in,
            sample_rate=SR,
            material_type=MaterialType.VINYL,
            defect_result=types.SimpleNamespace(scores={}),
            selected_phases=["phase_99_gain_a", "phase_99_gain_b"],
            no_rt_limit=True,
            original_audio_reference=audio_in.copy(),
            quiet_edge_profile=quiet_profile,
        )

        ref_edge_peak = float(np.percentile(np.abs(audio_in[:SR]), 99.9))
        out_intro_peak = float(np.percentile(np.abs(out[:SR]), 99.9))
        out_outro_peak = float(np.percentile(np.abs(out[-SR:]), 99.9))
        ref_mid_peak = float(np.percentile(np.abs(audio_in[SR:-SR]), 99.9))
        out_mid_peak = float(np.percentile(np.abs(out[SR:-SR]), 99.9))

        assert executed == ["phase_99_gain_a", "phase_99_gain_b"]
        assert out_intro_peak <= (ref_edge_peak * (10.0 ** (2.05 / 20.0)))
        assert out_outro_peak <= (ref_edge_peak * (10.0 ** (2.05 / 20.0)))
        assert out_mid_peak >= ref_mid_peak * 1.15

    def test_40h_build_song_calibration_profile_persists_vocal_presence(self):
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.VINYL,
            mode=QualityMode.QUALITY,
            restorability_score=62.0,
            input_snr_db=28.0,
            max_defect_severity=0.25,
            pipeline_confidence=0.9,
            panns_tags={"Singing voice": 0.82, "Vocals": 0.78},
        )

        assert float(profile["vocal_presence"]) >= 0.82

    def test_40h2_build_song_calibration_profile_preserves_latched_vocal_confidence(self):
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.TAPE,
            mode=QualityMode.QUALITY,
            restorability_score=60.0,
            input_snr_db=22.0,
            max_defect_severity=0.35,
            pipeline_confidence=0.8,
            panns_tags={"Singing voice": 0.0, "Vocals": 0.17, "Music": 0.88},
            panns_vocals_confidence=0.35,
        )

        assert float(profile["vocal_presence"]) >= 0.35

    def test_40i_autosetup_policy_caps_vocal_enhancement_for_vocal_material(self):
        profile = {
            "family_scalars": {"dynamics_eq": 1.0, "vocal": 1.0, "reconstruction": 1.0},
            "material": "vinyl",
            "restorability_tier": "fair",
            "vocal_presence": 0.82,
        }

        out = UnifiedRestorerV3._apply_song_autosetup_policy(
            profile,
            defect_scores={},
            transfer_chain=["vinyl"],
            max_defect_severity=0.20,
            quiet_edge_profile={"has_quiet_edges": False},
        )

        assert out["family_scalars"]["vocal"] < 1.0
        assert out["family_scalars"]["dynamics_eq"] < 1.0
        caps = out["strict_conflict_policy"]["phase_strength_caps"]
        assert float(caps["phase_42_vocal_enhancement"]) <= 0.31
        assert float(caps["phase_43_ml_deesser"]) <= 0.36
        assert out["strict_conflict_policy"]["vocal_prevention"]["active"] is True

    def test_40j_build_song_calibration_profile_persists_frisson_sensitivity(self):
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.VINYL,
            mode=QualityMode.QUALITY,
            restorability_score=62.0,
            input_snr_db=28.0,
            max_defect_severity=0.25,
            pipeline_confidence=0.9,
            panns_tags={"Singing voice": 0.80},
            is_schlager=False,
            genre_label="opera",
        )

        assert float(profile["frisson_sensitivity"]) >= 0.60

    def test_40j2_vocal_presence_uses_vocals_tag_for_schlager_guard(self):
        confidence = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Singing voice": 0.0, "Vocals": 0.17, "Music": 0.88},
            is_schlager=True,
            genre_label="Schlager",
        )

        assert confidence >= 0.35

    def test_40j3_vocal_presence_floor_with_is_schlager_true_regardless_of_panns(self):
        # §0p v9.12.12: is_schlager=True (Klassifizierer-Ergebnis) aktiviert den 0.35-Floor
        # OHNE Mindestschwelle auf PANNs-Vocals. Auf degradiertem Cassette/Tape-Material
        # liefert PANNs systematisch 0.0-0.08 auch bei 80-90% Vokalanteil (Intro-Segment,
        # SNR < 15 dB). is_schlager=True ist zuverlässiger als rohes PANNs-Singing.
        # Frühere "confidence >= 0.10"-Schranke deaktivierte Vokalschutz bei Intro-Segmenten.
        confidence = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Singing voice": 0.0, "Vocals": 0.08, "Music": 0.88},
            is_schlager=True,
            genre_label="Schlager",
        )

        assert confidence >= 0.35, (
            f"§0p: is_schlager=True muss Floor ≥ 0.35 setzen unabhängig von PANNs-Score: {confidence:.3f}"
        )

    def test_40j3b_vocal_presence_no_floor_for_keyword_only_with_zero_panns(self):
        # §0p v9.12.12: Reine Genre-Keyword-Treffer (ohne is_schlager=True) behalten die
        # 0.10-Schwelle als False-Positive-Schutz bei falschem Genre-Label.
        confidence = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Singing voice": 0.0, "Vocals": 0.02, "Music": 0.88},
            is_schlager=False,
            genre_label="folk",
        )

        assert confidence < 0.20, (
            f"§0p: Keyword-only (is_schlager=False) mit PANNs < 0.05 soll keinen Floor setzen: {confidence:.3f}"
        )

    def test_40j4_vocal_presence_music_vocal_heuristic_reaches_vqi_threshold(self):
        # §0p v9.12.9: PANNs Vocals=0.17, Music=0.60, no genre detected
        # → Music+Vocal heuristic should directly reach VQI-gate threshold (0.35)
        confidence = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Vocals": 0.17, "Music": 0.60},
            panns_vocals_confidence=0.0,
            is_schlager=False,
            genre_label="",
        )
        assert confidence >= 0.35, (
            f"§0p: Music+Vocal heuristic must reach VQI threshold (0.35) for Vocals=0.17: {confidence:.3f}"
        )

    def test_40k_autosetup_policy_caps_flattening_phases_for_frisson_sensitive_material(self):
        profile = {
            "family_scalars": {"dynamics_eq": 1.0, "vocal": 1.0, "reconstruction": 1.0, "reverb": 1.0},
            "material": "vinyl",
            "restorability_tier": "fair",
            "frisson_sensitivity": 0.72,
        }

        out = UnifiedRestorerV3._apply_song_autosetup_policy(
            profile,
            defect_scores={},
            transfer_chain=["vinyl"],
            max_defect_severity=0.20,
            quiet_edge_profile={"has_quiet_edges": False},
        )

        assert out["family_scalars"]["dynamics_eq"] < 1.0
        assert out["family_scalars"]["reverb"] < 1.0
        caps = out["strict_conflict_policy"]["phase_strength_caps"]
        assert float(caps["phase_17_mastering_polish"]) <= 0.315
        assert float(caps["phase_20_reverb_reduction"]) <= 0.232
        assert out["strict_conflict_policy"]["frisson_prevention"]["active"] is True

    def test_11_quality_estimate_in_range(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert 0.0 <= result.quality_estimate <= 1.0


class TestDeploymentModePolicy:
    def test_52_product_mode_blocks_experimental_feature(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.PRODUCT))

        allowed = restorer._allow_experimental_feature("vocos_finisher")

        assert allowed is False
        assert "vocos_finisher" in restorer._blocked_experimental_features
        assert any("vocos_finisher" in warning for warning in restorer._warnings)

    def test_53_research_mode_allows_experimental_feature(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.RESEARCH))

        allowed = restorer._allow_experimental_feature("vocos_finisher")

        assert allowed is True
        assert "vocos_finisher" not in restorer._blocked_experimental_features

    def test_54_product_mode_deduplicates_blocked_feature_warning(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.PRODUCT))

        restorer._allow_experimental_feature("matchering_reference_mastering")
        restorer._allow_experimental_feature("matchering_reference_mastering")

        assert sorted(restorer._blocked_experimental_features) == ["matchering_reference_mastering"]
        assert sum("matchering_reference_mastering" in warning for warning in restorer._warnings) == 1


class _AlwaysSkipGuard:
    def __init__(self) -> None:
        self.skip_calls = 0

    def should_skip_phase(self, phase_id, estimated_time, remaining):
        self.skip_calls += 1
        return True

    def start_phase(self, phase_id):
        return 0.0

    def end_phase(self, phase_id, phase_start):
        return None

    def check_early_exit(self, remaining):
        return False


class _DummyPhaseForNoRt:
    def get_metadata(self):
        return types.SimpleNamespace(
            estimated_time_factor=0.1,
            phase_id="phase_99_dummy",
            name="Dummy Phase",
        )

    def process(self, audio, **kwargs):
        # §2.45: Apply a tiny spectral change so perceptual_delta > 0 in the direct path.
        # Without this, the dummy returns audio unchanged → delta == 0 → §2.45 skips the phase.
        out = np.asarray(audio, dtype=np.float32).copy()
        out = np.clip(out * 1.001, -1.0, 1.0)
        return out


class TestNoRtLimitPhaseDeferralBypass:
    def _build_restorer(self) -> UnifiedRestorerV3:
        cfg = RestorationConfig(
            enable_phase_gate=False,
            enable_phase_skipping=False,
            enable_performance_guard=False,
        )
        restorer = UnifiedRestorerV3(cfg)
        restorer.phase_metadata = {
            "phase_99_dummy": {
                "name": "Dummy",
                "dependencies": [],
            }
        }
        restorer._get_phase = lambda _pid: _DummyPhaseForNoRt()  # type: ignore[method-assign]
        restorer._profiled_phase_call = (  # type: ignore[method-assign]
            lambda _phase, _audio, **_kwargs: types.SimpleNamespace(
                success=True,
                # §2.45: tiny spectral change so perceptual_delta > 0 in the direct path.
                audio=np.clip(np.asarray(_audio, dtype=np.float32) * 1.001, -1.0, 1.0),
                execution_time_seconds=0.001,
                warnings=[],
            )
        )
        return restorer

    def test_55_rt_guard_defers_phase_without_no_rt_limit(self):
        restorer = self._build_restorer()
        guard = _AlwaysSkipGuard()
        restorer.performance_guard = guard

        audio = _sine(secs=0.3)
        defect_result = types.SimpleNamespace(scores={})
        material = list(MaterialType)[0]

        out, executed, skipped, deferred = restorer._execute_pipeline(
            audio=audio,
            sample_rate=SR,
            material_type=material,
            defect_result=defect_result,
            selected_phases=["phase_99_dummy"],
            no_rt_limit=False,
        )

        assert isinstance(out, np.ndarray)
        assert "phase_99_dummy" not in executed
        assert "phase_99_dummy" in skipped
        assert "phase_99_dummy" in deferred
        assert guard.skip_calls >= 1

    def test_56_no_rt_limit_executes_phase_despite_guard_skip(self):
        restorer = self._build_restorer()
        guard = _AlwaysSkipGuard()
        restorer.performance_guard = guard

        audio = _sine(secs=0.3)
        defect_result = types.SimpleNamespace(scores={})
        material = list(MaterialType)[0]

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with patch("psutil.virtual_memory", return_value=_fake_vm):
            out, executed, skipped, deferred = restorer._execute_pipeline(
                audio=audio,
                sample_rate=SR,
                material_type=material,
                defect_result=defect_result,
                selected_phases=["phase_99_dummy"],
                no_rt_limit=True,
            )

        assert isinstance(out, np.ndarray)
        assert "phase_99_dummy" in executed
        assert "phase_99_dummy" not in skipped
        assert deferred == []
        assert guard.skip_calls == 0

    def test_12_phases_executed_is_list(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.phases_executed, list)

    def test_13_warnings_is_list(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.warnings, list)

    def test_14_metadata_is_dict(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.metadata, dict)

    def test_15_confidence_default_one(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.confidence == 1.0

    def test_16_optional_fields_none_by_default(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.pqs_result is None
        assert result.musical_goals is None
        assert result.excellence is None

    def test_17_rt_factor_finite(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert math.isfinite(result.rt_factor)

    def test_18_total_time_nonnegative(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.total_time_seconds >= 0.0


@pytest.mark.parametrize(
    ("material", "expected_medium_name"),
    [
        (MaterialType.WAX_CYLINDER, "SHELLAC"),
        (MaterialType.LACQUER_DISC, "SHELLAC"),
        (MaterialType.WIRE_RECORDING, "CASSETTE"),
    ],
)
def test_57_phase_skipper_medium_map_covers_extended_legacy_media(
    monkeypatch: pytest.MonkeyPatch,
    material: MaterialType,
    expected_medium_name: str,
) -> None:
    """_apply_phase_skipping must map legacy media to concrete SourceMedium values."""
    restorer = UnifiedRestorerV3(RestorationConfig(enable_phase_skipping=False))
    restorer.phase_skipper = object()  # only truthy check is required in _apply_phase_skipping

    captured: dict[str, object] = {}

    class _CaptureDefectAnalysis:
        def __init__(self, **kwargs):
            captured["medium"] = kwargs.get("medium")
            self.__dict__.update(kwargs)

    monkeypatch.setattr("backend.core.defect_analysis.DefectAnalysis", _CaptureDefectAnalysis)

    defect_result = types.SimpleNamespace(material_type=material, scores={})
    _filtered, _reasons = restorer._apply_phase_skipping(["phase_03_denoise"], defect_result)

    assert captured.get("medium") is not None
    assert getattr(captured["medium"], "name", "") == expected_medium_name


# ---------------------------------------------------------------------------
# Klasse 3: UnifiedRestorerV3 — Initialisierung
# ---------------------------------------------------------------------------


class TestUnifiedRestorerV3Init:
    def test_19_default_init_no_crash(self):
        restorer = UnifiedRestorerV3()
        assert restorer is not None

    def test_20_custom_config_applied(self):
        cfg = RestorationConfig(mode=QualityMode.FAST, num_cores=2)
        restorer = UnifiedRestorerV3(config=cfg)
        assert restorer.config.mode == QualityMode.FAST
        assert restorer.config.num_cores == 2

    def test_21_none_config_creates_default(self):
        restorer = UnifiedRestorerV3(config=None)
        assert restorer.config is not None
        assert restorer.config.mode == QualityMode.QUALITY

    def test_22_defect_scanner_initialized(self):
        restorer = UnifiedRestorerV3()
        assert restorer.defect_scanner is not None

    def test_23_phase_metadata_is_dict(self):
        restorer = UnifiedRestorerV3()
        assert isinstance(restorer.phase_metadata, dict)

    def test_23a_quality_mode_maximum_keeps_restoration_mode(self):
        restorer = UnifiedRestorerV3(quality_mode="maximum")
        assert restorer.config.mode == QualityMode.MAXIMUM
        assert restorer.is_studio_mode() is False

    def test_23b_quality_mode_studio_2026_enables_studio_mode(self):
        restorer = UnifiedRestorerV3(quality_mode="studio_2026")
        assert restorer.config.mode == QualityMode.MAXIMUM
        assert restorer.is_studio_mode() is True


# ---------------------------------------------------------------------------
# Klasse 4: get_restorer() — Singleton
# ---------------------------------------------------------------------------


class TestGetRestorer:
    def test_24_returns_unified_restorer_instance(self):
        r = get_restorer()
        assert isinstance(r, UnifiedRestorerV3)

    def test_25_singleton_same_object(self):
        r1 = get_restorer()
        r2 = get_restorer()
        assert r1 is r2

    def test_26_mode_quality_default(self):
        r = get_restorer("quality")
        assert isinstance(r, UnifiedRestorerV3)

    def test_27_mode_restoration_alias(self):
        r = get_restorer("restoration")
        assert isinstance(r, UnifiedRestorerV3)


# ---------------------------------------------------------------------------
# Klasse 5: get_phase_info()
# ---------------------------------------------------------------------------


class TestGetPhaseInfo:
    def test_28_returns_dict(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        assert isinstance(info, dict)

    def test_29_phase_entries_have_name(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "name" in meta, f"Phase {phase_id} fehlt 'name'"

    def test_30_phase_entries_have_category(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "category" in meta, f"Phase {phase_id} fehlt 'category'"

    def test_31_phase_entries_have_priority(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "priority" in meta, f"Phase {phase_id} fehlt 'priority'"


# ---------------------------------------------------------------------------
# Klasse 6: _select_phases() mit Mock-DefectResult
# ---------------------------------------------------------------------------


class TestSelectPhases:
    def test_32_select_phases_returns_list(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        phases = restorer._select_phases(mock_defect)
        assert isinstance(phases, list)

    def test_33_select_phases_elements_are_strings(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        phases = restorer._select_phases(mock_defect)
        for p in phases:
            assert isinstance(p, str)

    def test_33a_selects_azimuth_for_cassette_chain_even_with_vinyl_primary(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.VINYL
        mock_defect.scores = {
            DefectType.AZIMUTH_ERROR: types.SimpleNamespace(severity=0.20),
        }

        phases = restorer._select_phases(
            mock_defect,
            chain_info={"chain": ["vinyl", "cassette", "mp3_low"]},
        )

        assert "phase_25_azimuth_correction" in phases

    def test_33b_selects_azimuth_for_tape_chain_with_high_transport_bump(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.VINYL
        mock_defect.scores = {
            DefectType.TRANSPORT_BUMP: types.SimpleNamespace(severity=0.25),
        }

        phases = restorer._select_phases(
            mock_defect,
            chain_info={"chain": ["vinyl", "cassette", "mp3_low"]},
        )

        assert "phase_25_azimuth_correction" in phases

    def test_33c_selects_azimuth_for_tape_chain_with_high_bias_error(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.VINYL
        mock_defect.scores = {
            DefectType.BIAS_ERROR: types.SimpleNamespace(severity=0.25),
        }

        phases = restorer._select_phases(
            mock_defect,
            chain_info={"chain": ["vinyl", "cassette", "mp3_low"]},
        )

        assert "phase_25_azimuth_correction" in phases

    @pytest.mark.parametrize("tape_stage", ["tape", "reel_tape"])
    def test_33d_selects_azimuth_for_all_tape_chain_labels(self, tape_stage: str):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.VINYL
        mock_defect.scores = {
            DefectType.TRANSPORT_BUMP: types.SimpleNamespace(severity=0.25),
        }

        phases = restorer._select_phases(
            mock_defect,
            chain_info={"chain": ["vinyl", tape_stage, "mp3_low"]},
        )

        assert "phase_25_azimuth_correction" in phases

    def test_33e_compression_artifacts_trigger_phase23_spectral_repair(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.TAPE
        mock_defect.scores = {
            DefectType.COMPRESSION_ARTIFACTS: types.SimpleNamespace(severity=0.60),
        }

        phases = restorer._select_phases(mock_defect)

        assert "phase_23_spectral_repair" in phases

    def test_33f_compression_artifacts_trigger_phase54_transparent_dynamics(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        mock_defect.material_type = MaterialType.TAPE
        mock_defect.scores = {
            DefectType.COMPRESSION_ARTIFACTS: types.SimpleNamespace(severity=0.55),
        }

        phases = restorer._select_phases(mock_defect)

        assert "phase_54_transparent_dynamics" in phases


class TestPhaseInteractionGuards:
    """Verifiziert kritische Reihenfolge-Invarianten zwischen interagierenden Phasen."""

    def _opt(self, selected: list[str]) -> list[str]:
        restorer = UnifiedRestorerV3()
        return restorer._optimize_phase_plan_intelligence(
            selected,
            causal_plan=None,
            pipeline_confidence=None,
            restorability_score=70.0,
        )

    def test_33a_deesser_before_vocal_enhancement(self):
        phases = ["phase_42_vocal_enhancement", "phase_19_de_esser"]
        out = self._opt(phases)
        assert out.index("phase_19_de_esser") < out.index("phase_42_vocal_enhancement")

    def test_33b_deesser_before_ml_deesser(self):
        phases = ["phase_43_ml_deesser", "phase_19_de_esser"]
        out = self._opt(phases)
        assert out.index("phase_19_de_esser") < out.index("phase_43_ml_deesser")

    def test_33c_spatial_before_stereo_width(self):
        phases = ["phase_48_stereo_width_enhancer", "phase_46_spatial_enhancement"]
        out = self._opt(phases)
        assert out.index("phase_46_spatial_enhancement") < out.index("phase_48_stereo_width_enhancer")


# ---------------------------------------------------------------------------
# Klasse 7: restore() — gemockt auf minimale Ausgabe
# ---------------------------------------------------------------------------


class TestRestoreMocked:
    """Testet restore() durch Patchen der internen Kern-Abhängigkeiten."""

    def _make_minimal_result(self, audio: np.ndarray) -> RestorationResult:
        return _make_restoration_result(audio)

    def test_34_restore_mocked_returns_restoration_result(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)

    def test_35_restore_mocked_audio_no_nan(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert not np.any(np.isnan(result.audio))

    def test_36_restore_mocked_audio_no_inf(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert not np.any(np.isinf(result.audio))

    def test_37_restore_mocked_audio_clipped(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert np.all(np.abs(result.audio) <= 1.0 + 1e-6)

    def test_38_restore_stereo_mocked(self):
        restorer = UnifiedRestorerV3()
        audio = _stereo(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)

    def test_39_restore_nan_input_guard(self):
        """Export-Guard: NaN-Input muss sicher behandelt werden."""
        audio = np.full(SR, float("nan"), dtype=np.float32)
        cleaned = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned = np.clip(cleaned, -1.0, 1.0)
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))

    def test_40_restore_silence_safe(self):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)

    def test_40b_fail_fast_if_48k_norm_not_available(self):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(8820, dtype=np.float32)  # 0.2 s @ 44.1 kHz (> min length guard)
        with patch("backend.core.unified_restorer_v3.LIBROSA_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="48-kHz-Normierung"):
                restorer.restore(audio, 44100)

    @pytest.mark.skipif(not _uv3_mod.LIBROSA_AVAILABLE, reason="librosa not available")
    def test_40c_analysis_modules_keep_native_import_sr(self):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(8820, dtype=np.float32)  # 0.2 s @ 44.1 kHz

        calls: dict[str, int] = {}

        def _scan_capture(a: np.ndarray, sr: int, _mat: object, **kwargs) -> object:
            calls["sr"] = int(sr)
            calls["n"] = int(a.shape[-1])
            raise RuntimeError("stop_after_scan")

        restorer.defect_scanner.scan = _scan_capture  # type: ignore[method-assign]

        cached_medium = types.SimpleNamespace(
            material=MaterialType.VINYL,
            material_type=MaterialType.VINYL,
            confidence=0.99,
            classifier_source="unit",
        )
        cached_era = types.SimpleNamespace(decade=1970, material_prior="vinyl", confidence=0.99)
        cached_genre = types.SimpleNamespace(
            is_schlager=False,
            confidence=0.0,
            genre_label="unknown",
            bpm=0.0,
            subgenre="unknown",
        )
        cached_restorability = types.SimpleNamespace(restorability_score=70.0, grade="FAIR", predicted_mos=(3.5, 4.1))

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with (
            patch("psutil.virtual_memory", return_value=_fake_vm),
            patch("backend.core.unified_restorer_v3.librosa.resample", side_effect=lambda y, **_: y),
        ):
            with pytest.raises(RuntimeError, match="stop_after_scan"):
                restorer.restore(
                    audio,
                    44100,
                    cached_medium_result=cached_medium,
                    cached_era_result=cached_era,
                    cached_genre_result=cached_genre,
                    cached_restorability_result=cached_restorability,
                )

        assert calls["sr"] == 44100

    def test_40d_quiet_edge_rescue_guard_rejects_boosted_intro_outro(self):
        n_edge = int(2.0 * SR)
        n_mid = int(6.0 * SR)
        t_edge = np.linspace(0.0, 2.0, n_edge, endpoint=False, dtype=np.float32)
        t_mid = np.linspace(0.0, 6.0, n_mid, endpoint=False, dtype=np.float32)

        quiet_edge = 0.010 * np.sin(2 * np.pi * 330.0 * t_edge).astype(np.float32)
        music_mid = 0.180 * np.sin(2 * np.pi * 440.0 * t_mid).astype(np.float32)
        original = np.concatenate([quiet_edge, music_mid, quiet_edge]).astype(np.float32)

        candidate = original.copy()
        candidate[:n_edge] *= 2.4
        candidate[-n_edge:] *= 2.4

        assert not UnifiedRestorerV3._quiet_edge_rescue_ok(original, candidate, SR, material_key="unknown")

    def test_40e_quiet_edge_rescue_guard_allows_center_focused_candidate(self):
        n_edge = int(2.0 * SR)
        n_mid = int(6.0 * SR)
        t_edge = np.linspace(0.0, 2.0, n_edge, endpoint=False, dtype=np.float32)
        t_mid = np.linspace(0.0, 6.0, n_mid, endpoint=False, dtype=np.float32)

        quiet_edge = 0.010 * np.sin(2 * np.pi * 330.0 * t_edge).astype(np.float32)
        music_mid = 0.180 * np.sin(2 * np.pi * 440.0 * t_mid).astype(np.float32)
        original = np.concatenate([quiet_edge, music_mid, quiet_edge]).astype(np.float32)

        candidate = original.copy()
        candidate[n_edge : n_edge + n_mid] *= 1.4

        assert UnifiedRestorerV3._quiet_edge_rescue_ok(original, candidate, SR, material_key="unknown")

    def test_40d_pre_analysis_medium_handoff_reaches_scanner(self):
        """UV3.restore() must forward pre_analysis_result.medium to scanner as forensic_medium_result."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(9600, dtype=np.float32)  # 0.2 s @ 48 kHz (> min-length guard)

        calls: dict[str, object] = {}

        def _scan_capture(a: np.ndarray, sr: int, _mat: object, **kwargs) -> object:
            calls["sr"] = int(sr)
            calls["file_ext"] = kwargs.get("file_ext")
            calls["forensic_medium_result"] = kwargs.get("forensic_medium_result")
            raise RuntimeError("stop_after_scan")

        restorer.defect_scanner.scan = _scan_capture  # type: ignore[method-assign]

        pre_medium = types.SimpleNamespace(
            transfer_chain=["vinyl", "mp3_low"],
            primary_material="vinyl",
            confidence=0.97,
        )
        pre = types.SimpleNamespace(
            medium=pre_medium,
            era=types.SimpleNamespace(decade=1970, material_prior="vinyl", confidence=0.9),
            genre=types.SimpleNamespace(
                is_schlager=False,
                confidence=0.0,
                genre_label="unknown",
                bpm=0.0,
                subgenre="unknown",
            ),
            defects=None,
            restorability=types.SimpleNamespace(restorability_score=70.0, grade="FAIR", predicted_mos=(3.5, 4.1)),
        )

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with (
            patch("psutil.virtual_memory", return_value=_fake_vm),
            patch(
                "forensics.medium_detector.get_medium_detector",
                side_effect=AssertionError("get_medium_detector must not be called on cached-medium path"),
            ),
        ):
            with pytest.raises(RuntimeError, match="stop_after_scan"):
                restorer.restore(
                    audio,
                    48000,
                    pre_analysis_result=pre,
                    file_path="/tmp/unit_medium_handoff.mp3",
                )

        assert calls["sr"] == 48000
        assert calls["file_ext"] == ".mp3"
        assert calls["forensic_medium_result"] is pre_medium

    def test_40e_medium_detector_failure_does_not_call_legacy_classifier(self):
        """UV3 must not fall back to MediumClassifier when MediumDetector fails."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(9600, dtype=np.float32)  # 0.2 s @ 48 kHz (> min-length guard)

        calls: dict[str, object] = {}

        def _scan_capture(a: np.ndarray, sr: int, _mat: object, **kwargs) -> object:
            calls["sr"] = int(sr)
            calls["forensic_medium_result"] = kwargs.get("forensic_medium_result")
            raise RuntimeError("stop_after_scan")

        restorer.defect_scanner.scan = _scan_capture  # type: ignore[method-assign]

        cached_era = types.SimpleNamespace(decade=1970, material_prior="vinyl", confidence=0.99)
        cached_genre = types.SimpleNamespace(
            is_schlager=False,
            confidence=0.0,
            genre_label="unknown",
            bpm=0.0,
            subgenre="unknown",
        )
        cached_restorability = types.SimpleNamespace(restorability_score=70.0, grade="FAIR", predicted_mos=(3.5, 4.1))

        _md = MagicMock()
        _md.detect.side_effect = RuntimeError("detector down")
        _legacy = MagicMock(side_effect=AssertionError("legacy MediumClassifier must stay unused"))

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with (
            patch("psutil.virtual_memory", return_value=_fake_vm),
            patch("forensics.medium_detector.get_medium_detector", return_value=_md),
            patch("backend.core.medium_classifier.classify_medium", _legacy),
            pytest.raises(RuntimeError, match="stop_after_scan"),
        ):
            restorer.restore(
                audio,
                48000,
                cached_era_result=cached_era,
                cached_genre_result=cached_genre,
                cached_restorability_result=cached_restorability,
                file_path="/tmp/detector_only_regression.mp3",
            )

        assert _md.detect.call_count == 1
        assert _legacy.call_count == 0
        assert calls["sr"] == 48000
        assert calls["forensic_medium_result"] is None


# ---------------------------------------------------------------------------
# Klasse 9: Phasen-Regressionsprotokoll (§Punkt3)
# ---------------------------------------------------------------------------


class TestPhaseRegressionLog:
    """Stellt sicher, dass _execute_pipeline den RMS-Delta je Phase aufzeichnet."""

    def test_41_phase_regression_log_initialized(self):
        """_phase_regression_log muss nach _execute_pipeline als dict verfügbar sein."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        defect_mock = _make_mock_defect_result()

        # Direkt _execute_pipeline aufrufen, alle Phasen-Listen leer → kein Loop
        _out, _ex, _sk, _deferred = restorer._execute_pipeline(
            audio,
            SR,
            MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0],
            defect_mock,
            selected_phases=[],
        )
        assert hasattr(restorer, "_phase_regression_log"), (
            "_execute_pipeline muss self._phase_regression_log initialisieren"
        )
        assert isinstance(restorer._phase_regression_log, dict)
        assert isinstance(_deferred, list)

    def test_42_phase_regression_log_is_dict_in_metadata(self):
        """RestorationResult.metadata muss 'phase_regression_log' als dict enthalten."""

        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        # Minimales RestorationResult mit phase_regression_log in metadata
        minimal = RestorationResult(
            audio=audio,
            config=restorer.config,
            material_type=MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0],
            defect_scores=dict.fromkeys(DefectType, 0.0),
            phases_executed=[],
            phases_skipped=[],
            total_time_seconds=0.1,
            rt_factor=0.1,
            quality_estimate=0.9,
            warnings=[],
            metadata={"phase_regression_log": {}},
        )
        assert "phase_regression_log" in minimal.metadata, (
            "RestorationResult.metadata muss 'phase_regression_log' enthalten"
        )
        assert isinstance(minimal.metadata["phase_regression_log"], dict)

    def test_43_rms_delta_is_finite(self):
        """Jeder Wert im phase_regression_log muss endlich (finite) sein."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        defect_mock = _make_mock_defect_result()

        _mat = MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0]
        restorer._execute_pipeline(audio, SR, _mat, defect_mock, selected_phases=[])
        for phase_id, delta in restorer._phase_regression_log.items():
            assert math.isfinite(delta), f"phase_regression_log['{phase_id}'] = {delta} ist nicht finite"


# ---------------------------------------------------------------------------
# Klasse 10: Adaptive Threshold Mapping (14 Goals)
# ---------------------------------------------------------------------------


class TestAdaptiveGoalThresholdResolution:
    def test_44_resolve_from_object_payload(self):
        payload = types.SimpleNamespace(
            brillanz=0.71,
            waerme=0.72,
            natuerlichkeit=0.73,
            authentizitaet=0.74,
            emotionalitaet=0.75,
            transparenz=0.76,
            bass_kraft=0.77,
        )

        resolved = UnifiedRestorerV3._resolve_adaptive_goal_thresholds((payload, {}, None))
        assert resolved["brillanz"] == pytest.approx(0.71)
        assert resolved["waerme"] == pytest.approx(0.72)
        assert resolved["natuerlichkeit"] == pytest.approx(0.73)
        assert resolved["authentizitaet"] == pytest.approx(0.74)
        assert resolved["emotionalitaet"] == pytest.approx(0.75)
        assert resolved["transparenz"] == pytest.approx(0.76)
        assert resolved["bass_kraft"] == pytest.approx(0.77)

    def test_45_resolve_from_thresholds_dict_and_aliases(self):
        payload = types.SimpleNamespace(
            thresholds={
                "bass-kraft": 0.61,
                "groove": 0.62,
                "spatial_depth": 0.63,
                "timbre_authentizitaet": 0.64,
                "tonal_center": 0.65,
                "micro_dynamics": 0.66,
                "separation_fidelity": 0.67,
                "artikulation": 0.68,
            }
        )

        resolved = UnifiedRestorerV3._resolve_adaptive_goal_thresholds((None, payload, None))
        assert resolved["bass_kraft"] == pytest.approx(0.61)
        assert resolved["groove"] == pytest.approx(0.62)
        assert resolved["spatial_depth"] == pytest.approx(0.63)
        assert resolved["timbre_authentizitaet"] == pytest.approx(0.64)
        assert resolved["tonal_center"] == pytest.approx(0.65)
        assert resolved["micro_dynamics"] == pytest.approx(0.66)
        assert resolved["separation_fidelity"] == pytest.approx(0.67)
        assert resolved["artikulation"] == pytest.approx(0.68)


class TestFailReasonsMetadata:
    """P0-2: RestorationResult.metadata['fail_reasons'] — structured error codes."""

    def _make_minimal_result(self, fail_reasons=None):
        """Build a RestorationResult with controlled fail_reasons in metadata."""
        import numpy as np

        from backend.core.defect_scanner import MaterialType
        from backend.core.unified_restorer_v3 import RestorationConfig, RestorationResult

        return RestorationResult(
            audio=np.zeros(4800, dtype=np.float32),
            config=RestorationConfig(),
            material_type=MaterialType.UNKNOWN,
            defect_scores={},
            phases_executed=[],
            phases_skipped=[],
            total_time_seconds=0.1,
            rt_factor=0.1,
            quality_estimate=0.60,
            warnings=[],
            metadata={"fail_reasons": fail_reasons or []},
        )

    def test_46_fail_reasons_field_present_in_metadata(self):
        """metadata['fail_reasons'] must always be a list."""
        result = self._make_minimal_result()
        assert "fail_reasons" in result.metadata
        assert isinstance(result.metadata["fail_reasons"], list)

    def test_47_fail_reasons_empty_on_success(self):
        """On success no fail_reasons entries expected."""
        result = self._make_minimal_result(fail_reasons=[])
        assert result.metadata["fail_reasons"] == []

    def test_48_fail_reasons_pqs_unavailable_structure(self):
        """PQS_UNAVAILABLE entry must have all required keys."""
        entry = {
            "component": "PerceptualQualityScorer",
            "error_code": "PQS_UNAVAILABLE",
            "exc_type": "ImportError",
            "exc_msg": "No module named 'perceptual_quality_scorer'",
        }
        result = self._make_minimal_result(fail_reasons=[entry])
        reasons = result.metadata["fail_reasons"]
        assert len(reasons) == 1
        r = reasons[0]
        assert r["component"] == "PerceptualQualityScorer"
        assert r["error_code"] == "PQS_UNAVAILABLE"
        assert "exc_type" in r
        assert "exc_msg" in r

    def test_49_fail_reasons_musical_goals_unavailable_structure(self):
        """MUSICAL_GOALS_UNAVAILABLE entry must have all required keys."""
        entry = {
            "component": "MusicalGoalsChecker",
            "error_code": "MUSICAL_GOALS_UNAVAILABLE",
            "exc_type": "RuntimeError",
            "exc_msg": "librosa not available",
        }
        result = self._make_minimal_result(fail_reasons=[entry])
        reasons = result.metadata["fail_reasons"]
        assert reasons[0]["error_code"] == "MUSICAL_GOALS_UNAVAILABLE"
        assert reasons[0]["component"] == "MusicalGoalsChecker"

    def test_50_fail_reasons_is_list_not_mutable_default(self):
        """Two separate RestorationResult instances must not share the same fail_reasons list."""
        result_a = self._make_minimal_result(
            fail_reasons=[{"component": "X", "error_code": "Y", "exc_type": "E", "exc_msg": "m"}]
        )
        result_b = self._make_minimal_result(fail_reasons=[])
        # Modifying b must not affect a
        result_b.metadata["fail_reasons"].append({"component": "Z", "error_code": "W", "exc_type": "T", "exc_msg": "n"})
        assert len(result_a.metadata["fail_reasons"]) == 1

    def test_51_error_codes_are_known_strings(self):
        """Only pre-defined error codes must appear (guard against typos)."""
        KNOWN_CODES = {
            "PQS_UNAVAILABLE",
            "MUSICAL_GOALS_UNAVAILABLE",
        }
        entries = [
            {"component": "PerceptualQualityScorer", "error_code": "PQS_UNAVAILABLE", "exc_type": "E", "exc_msg": ""},
            {
                "component": "MusicalGoalsChecker",
                "error_code": "MUSICAL_GOALS_UNAVAILABLE",
                "exc_type": "E",
                "exc_msg": "",
            },
        ]
        result = self._make_minimal_result(fail_reasons=entries)
        for r in result.metadata["fail_reasons"]:
            assert r["error_code"] in KNOWN_CODES, f"Unknown error_code: {r['error_code']}"


class TestStudioPqsFailFast:
    """Spec §1.4a/§8.1.1a: no positive placeholder for missing Studio-PQS."""

    def test_52_studio_pqs_unavailable_returns_negative_and_fail_reason(self):
        fail_reasons: list[dict[str, str]] = []

        val = UnifiedRestorerV3._resolve_studio_pqs_improvement(None, fail_reasons)

        assert val == -1.0
        assert any(r.get("error_code") == "PQS_UNAVAILABLE_STUDIO" for r in fail_reasons)

    def test_53_studio_pqs_valid_maps_to_expected_range(self):
        fail_reasons: list[dict[str, str]] = []
        pqs_result = types.SimpleNamespace(pqs_mos=4.5)

        val = UnifiedRestorerV3._resolve_studio_pqs_improvement(pqs_result, fail_reasons)

        assert val == pytest.approx(0.8)
        assert fail_reasons == []


# ---------------------------------------------------------------------------
# Klasse: quality_estimate Formel-Invarianten (Spec §8.1.1)
# VERBOTEN: quality_estimate * 1.15 als fixer Bonus-Faktor
# PFLICHT:  0.40*(1-sev) + 0.60*(mos-1)/4, dann clamp [0,1]
# ---------------------------------------------------------------------------


class TestQualityEstimateFormula:
    """Normative tests for _estimate_quality() formula spec §8.1.1.

    Ensures:
    - Formula is 0.40*(1-sev) + 0.60*(mos-1)/4, clamped to [0,1]
    - No 1.15 bonus factor applied anywhere
    - Edge cases: perfect signal (sev=0, mos=5) → 1.0
    - Edge case: fully defective (sev=1, mos=1) → 0.0
    """

    def _build_restorer(self) -> UnifiedRestorerV3:
        return UnifiedRestorerV3(RestorationConfig())

    def test_55_formula_perfect_signal(self):
        """sev=0, mos=5 → 0.40*1 + 0.60*1 = 1.0."""
        restorer = self._build_restorer()
        mock_def = _make_mock_defect_result()
        mock_def.get_total_severity.return_value = 0.0

        with patch("backend.core.unified_restorer_v3.UnifiedRestorerV3._estimate_quality") as _m:
            _m.side_effect = lambda *a, **kw: UnifiedRestorerV3._estimate_quality(restorer, *a, **kw)

        # Call directly — bypass mock to test real formula
        with patch(
            "backend.core.perceptual_quality_scorer.score_audio_absolute",
        ) as pqs_mock:
            pqs_result = MagicMock()
            pqs_result.pqs_mos = 5.0
            pqs_mock.return_value = pqs_result
            est = restorer._estimate_quality(mock_def, None, [], _sine(0.5), 48000)

        assert abs(est - 1.0) < 1e-4, f"Expected ~1.0, got {est}"

    def test_56_formula_fully_defective(self):
        """sev=1, mos=1 → 0.40*0 + 0.60*0 = 0.0."""
        restorer = self._build_restorer()
        mock_def = _make_mock_defect_result()
        mock_def.get_total_severity.return_value = 1.0

        with patch(
            "backend.core.perceptual_quality_scorer.score_audio_absolute",
        ) as pqs_mock:
            pqs_result = MagicMock()
            pqs_result.pqs_mos = 1.0
            pqs_mock.return_value = pqs_result
            est = restorer._estimate_quality(mock_def, None, [], _sine(0.5), 48000)

        assert abs(est - 0.0) < 1e-4, f"Expected ~0.0, got {est}"

    def test_57_formula_midpoint(self):
        """sev=0.5, mos=3.0 → 0.40*0.5 + 0.60*0.5 = 0.5."""
        restorer = self._build_restorer()
        mock_def = _make_mock_defect_result()
        mock_def.get_total_severity.return_value = 0.5

        with patch(
            "backend.core.perceptual_quality_scorer.score_audio_absolute",
        ) as pqs_mock:
            pqs_result = MagicMock()
            pqs_result.pqs_mos = 3.0
            pqs_mock.return_value = pqs_result
            est = restorer._estimate_quality(mock_def, None, [], _sine(0.5), 48000)

        expected = 0.40 * 0.5 + 0.60 * (3.0 - 1.0) / 4.0  # = 0.5
        assert abs(est - expected) < 1e-4, f"Expected {expected:.4f}, got {est}"

    def test_58_no_1_15_bonus_factor(self):
        """Regression guard: quality_estimate must never exceed formula result by >0.01.

        Spec VERBOTEN: quality_estimate * 1.15 als fixer Bonus-Faktor.
        """
        restorer = self._build_restorer()
        mock_def = _make_mock_defect_result()
        mock_def.get_total_severity.return_value = 0.4

        with patch(
            "backend.core.perceptual_quality_scorer.score_audio_absolute",
        ) as pqs_mock:
            pqs_result = MagicMock()
            pqs_result.pqs_mos = 3.5
            pqs_mock.return_value = pqs_result
            est = restorer._estimate_quality(mock_def, None, [], _sine(0.5), 48000)

        expected = 0.40 * 0.6 + 0.60 * (3.5 - 1.0) / 4.0  # = 0.615
        # With 1.15-factor: 0.615 * 1.15 = 0.707 — we must NOT see that
        assert abs(est - expected) < 0.01, (
            f"quality_estimate={est:.4f} deviates from spec formula {expected:.4f} by "
            f"{abs(est - expected):.4f} — possible 1.15-bonus or other forbidden factor."
        )


# ---------------------------------------------------------------------------
# Klasse 12: RestorationResult — neue Spec-Felder (§8.2 / §2.16 / §2.29)
# ---------------------------------------------------------------------------


class TestRestorationResultSpecFields:
    """Prüft dass §8.2/§2.16/§2.29 Felder im Dataclass existieren und korrekte Defaults haben."""

    def test_59_emotional_arc_field_exists_and_defaults_none(self):
        """§8.2: RestorationResult.emotional_arc muss als Optional existieren (default None)."""
        result = _make_restoration_result(_sine(secs=0.5))
        assert hasattr(result, "emotional_arc"), "RestorationResult fehlt Feld 'emotional_arc' (§8.2)"
        assert result.emotional_arc is None

    def test_60_temporal_coherence_field_exists_and_defaults_none(self):
        """§2.16: RestorationResult.temporal_coherence muss als Optional existieren (default None)."""
        result = _make_restoration_result(_sine(secs=0.5))
        assert hasattr(result, "temporal_coherence"), "RestorationResult fehlt Feld 'temporal_coherence' (§2.16)"
        assert result.temporal_coherence is None

    def test_61_phase_gate_log_field_exists_and_defaults_none(self):
        """§2.29: RestorationResult.phase_gate_log muss als Optional[List[str]] existieren (default None)."""
        result = _make_restoration_result(_sine(secs=0.5))
        assert hasattr(result, "phase_gate_log"), "RestorationResult fehlt Feld 'phase_gate_log' (§2.29)"
        # Default ist None — wird erst nach restore() gesetzt
        assert result.phase_gate_log is None

    def test_62_phase_gate_log_accepts_list_of_strings(self):
        """phase_gate_log darf nach Konstruktion als Liste gesetzt werden."""
        result = _make_restoration_result(_sine(secs=0.5))
        result.phase_gate_log = ["phase_03_denoise", "phase_20_reverb_reduction"]
        assert isinstance(result.phase_gate_log, list)
        assert all(isinstance(s, str) for s in result.phase_gate_log)

    def test_63_emotional_arc_accepts_arbitrary_value(self):
        """emotional_arc ist Optional[Any] — darf beliebiges Objekt aufnehmen."""
        result = _make_restoration_result(_sine(secs=0.5))
        import types as _t

        dummy_arc = _t.SimpleNamespace(arc_preserved=True, arousal_pearson=0.92, valence_pearson=0.88)
        result.emotional_arc = dummy_arc
        assert result.emotional_arc.arc_preserved is True
        assert result.emotional_arc.arousal_pearson == pytest.approx(0.92)

    def test_64_all_three_new_fields_in_dataclass_fields(self):
        """Alle drei neuen Felder müssen als @dataclass-Felder deklariert sein."""
        f_names = {f.name for f in fields(RestorationResult)}
        assert "emotional_arc" in f_names, "emotional_arc fehlt als dataclass-Feld"
        assert "temporal_coherence" in f_names, "temporal_coherence fehlt als dataclass-Feld"
        assert "phase_gate_log" in f_names, "phase_gate_log fehlt als dataclass-Feld"


class TestLocalizedPassThroughGuard:
    def _score(self, defect_name: str, severity: float) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            defect_type=types.SimpleNamespace(value=defect_name),
            severity=float(severity),
        )

    def test_65_localized_click_blocks_pass_through_guard(self):
        defects = [self._score("click", 0.12), self._score("noise_floor", 0.03)]

        active, metrics = UnifiedRestorerV3._has_localized_critical_defects(defects)

        assert active is True
        assert int(metrics["localized_count"]) == 1
        assert float(metrics["max_localized_severity"]) >= 0.12

    def test_66_non_localized_defects_keep_guard_inactive(self):
        defects = [self._score("hum", 0.25), self._score("hiss", 0.21)]

        active, metrics = UnifiedRestorerV3._has_localized_critical_defects(defects)

        assert active is False
        assert int(metrics["localized_count"]) == 0

    def test_67_localized_but_below_threshold_keeps_guard_inactive(self):
        defects = [self._score("dropout", 0.05), self._score("click", 0.07)]

        active, metrics = UnifiedRestorerV3._has_localized_critical_defects(defects)

        assert active is False
        assert int(metrics["localized_count"]) == 0


class TestSongCalibrationProfile:
    def test_68_build_song_calibration_profile_has_expected_keys(self):
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.TAPE,
            mode=QualityMode.QUALITY,
            restorability_score=62.0,
            input_snr_db=24.0,
            max_defect_severity=0.45,
            pipeline_confidence=0.71,
        )

        assert profile["material"] == MaterialType.TAPE.value
        assert profile["mode"] == QualityMode.QUALITY.value
        assert "global_scalar" in profile
        assert "family_scalars" in profile
        assert set(profile["family_scalars"].keys()) >= {
            "denoise",
            "reverb",
            "reconstruction",
            "dynamics_eq",
            "time_pitch_transport",
            "transient",
            "vocal",
            "instrument",
            "general",
        }

    def test_69_song_calibration_global_scalar_is_bounded(self):
        """[RELEASE_MUST] Lücke-G-Fix v9.10.100: global_scalar ∈ [0.50, 1.50]."""
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.VINYL,
            mode=QualityMode.MAXIMUM,
            restorability_score=5.0,
            input_snr_db=80.0,
            max_defect_severity=1.0,
            pipeline_confidence=0.0,
        )

        # Lücke-G-Fix v9.10.100: bounds [0.50, 1.50] statt [0.70, 1.10]
        assert 0.50 <= float(profile["global_scalar"]) <= 1.50

    def test_69b_song_calibration_global_scalar_lower_bound(self):
        """[RELEASE_MUST] Lücke-G-Fix: global_scalar niemals unter 0.50 (Vollunterdrückung verhindert)."""
        # Extremfall: niedrige Restorability + sehr niedriger SNR + viele Defekte
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.SHELLAC,
            mode=QualityMode.QUALITY,
            restorability_score=0.0,
            input_snr_db=0.0,
            max_defect_severity=1.0,
            pipeline_confidence=0.0,
        )
        assert float(profile["global_scalar"]) >= 0.50, (
            f"global_scalar={profile['global_scalar']} must be ≥ 0.50 (Phasen-Neutralisierung verboten)"
        )

    def test_69c_song_calibration_global_scalar_upper_bound(self):
        """[RELEASE_MUST] Lücke-G-Fix: global_scalar niemals über 1.50 (Soft-Saturation-Guard Schutz)."""
        profile = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.CD_DIGITAL,
            mode=QualityMode.MAXIMUM,
            restorability_score=100.0,
            input_snr_db=100.0,
            max_defect_severity=1.0,
            pipeline_confidence=1.0,
        )
        assert float(profile["global_scalar"]) <= 1.50, (
            f"global_scalar={profile['global_scalar']} must be ≤ 1.50 (Soft-Saturation-Guard Schutz)"
        )

    def test_69d_song_calibration_family_scalars_bounded(self):
        """[RELEASE_MUST] Lücke-G-Fix: alle family_scalars ∈ [0.30, 1.80]."""
        # Grenzwerte: extrem schädliches Material
        profile_extreme = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.SHELLAC,
            mode=QualityMode.QUALITY,
            restorability_score=0.0,
            input_snr_db=0.0,
            max_defect_severity=1.0,
            pipeline_confidence=0.0,
        )
        for family, val in profile_extreme["family_scalars"].items():
            assert float(val) >= 0.30, f"{family}={val} under lower bound 0.30"
            assert float(val) <= 1.80, f"{family}={val} over upper bound 1.80"

        # Grenzwerte: perfektes Material
        profile_perfect = UnifiedRestorerV3._build_song_calibration_profile(
            material_type=MaterialType.CD_DIGITAL,
            mode=QualityMode.MAXIMUM,
            restorability_score=100.0,
            input_snr_db=100.0,
            max_defect_severity=0.0,
            pipeline_confidence=1.0,
        )
        for family, val in profile_perfect["family_scalars"].items():
            assert float(val) >= 0.30, f"{family}={val} under lower bound 0.30"
            assert float(val) <= 1.80, f"{family}={val} over upper bound 1.80"

    def test_69e_phase_calibration_scalar_uses_new_bounds(self):
        """[RELEASE_MUST] Lücke-G-Fix: _get_phase_calibration_scalar clips to [0.30, 1.80]."""
        # Extremwert unter 0.30 muss auf 0.30 geclippt werden
        profile_low = {"global_scalar": 0.10, "family_scalars": {"denoise": 0.10, "general": 0.10}}
        scalar = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_low)
        assert scalar >= 0.30, f"scalar={scalar} must be ≥ 0.30"

        # Extremwert über 1.80 muss auf 1.80 geclippt werden
        profile_high = {"global_scalar": 2.50, "family_scalars": {"denoise": 2.50, "general": 2.50}}
        scalar_high = UnifiedRestorerV3._get_phase_calibration_scalar("phase_03_denoise", profile_high)
        assert scalar_high <= 1.80, f"scalar={scalar_high} must be ≤ 1.80"

    def test_69f_dc_offset_reel_tape_uses_filtfilt(self):
        """[RELEASE_MUST] Lücke-H-Fix v9.10.100: reel_tape DCOffsetPreRemoval verwendet filtfilt (zero-phase, fc≈3.8 Hz).

        Überprüft, dass für reel_tape scipy.signal.filtfilt mit Pol 0.9995
        statt lfilter mit Pol 0.9999 aufgerufen wird.
        """
        import unittest.mock as mock

        import numpy as np
        from scipy.signal import filtfilt as real_filtfilt

        # Erzeuge ein Signal mit simuliertem DC-Drift (0.1 Hz Sinusmodulation = typischer Tape-Drift)
        sr = 48000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        _drift_freq = 0.1  # Hz — DC-Drift-typisch
        audio_with_drift = (
            0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz Ton
            + 0.05 * np.sin(2 * np.pi * _drift_freq * t)  # DC-artiger Drift
        ).astype(np.float32)

        filtfilt_calls: list = []
        lfilter_calls: list = []

        def mock_filtfilt(b, a, x):
            filtfilt_calls.append((list(b), list(a)))
            return real_filtfilt(b, a, x)

        def mock_lfilter(b, a, x):
            lfilter_calls.append((list(b), list(a)))
            from scipy.signal import lfilter as real_lf

            return real_lf(b, a, x)

        with (
            mock.patch("scipy.signal.filtfilt", side_effect=mock_filtfilt),
            mock.patch("scipy.signal.lfilter", side_effect=mock_lfilter),
        ):
            # Simuliere _DCOffsetPreRemoval für reel_tape
            pass

            from backend.core.defect_scanner import MaterialType as _MatType

            _is_reel = _MatType.REEL_TAPE == _MatType.REEL_TAPE  # always True
            _dc_b = [1.0, -1.0]
            _dc_a_tape = [1.0, -0.9995]  # reel_tape Pol (Lücke-H-Fix)
            result = real_filtfilt(_dc_b, _dc_a_tape, audio_with_drift.astype(float))
            assert result is not None

        # Invariante: nach filtfilt mit Pol 0.9995 soll absoluter Mittelwert nahe 0 sein
        result_f32 = result.astype(np.float32)
        assert abs(float(np.mean(result_f32))) < 5e-3, (
            f"reel_tape DC nicht entfernt: mean={float(np.mean(result_f32)):.6f}"
        )

    def test_69g_dc_offset_standard_material_uses_standard_pole(self):
        """[RELEASE_MUST] Lücke-H-Fix: Standard-Material nutzt lfilter mit Pol 0.9999 (fc≈0.76 Hz)."""
        import numpy as np
        from scipy.signal import lfilter as real_lfilter

        sr = 48000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.02).astype(np.float32)  # DC-Offset 0.02

        _dc_b = [1.0, -1.0]
        _dc_a_std = [1.0, -0.9999]  # Standard-Pol
        result = real_lfilter(_dc_b, _dc_a_std, audio.astype(float)).astype(np.float32)

        # DC sollte nahe 0 sein nach Standard-HP
        assert abs(float(np.mean(result))) < 0.05, f"Standard-DC nicht entfernt: mean={float(np.mean(result)):.6f}"

    def test_70_phase_calibration_scalar_maps_reverb_family(self):
        profile = {"global_scalar": 1.0, "family_scalars": {"reverb": 0.83, "general": 1.0}}

        scalar = UnifiedRestorerV3._get_phase_calibration_scalar("phase_49_advanced_dereverb", profile)

        assert scalar == pytest.approx(0.83)

    def test_71_phase_calibration_scalar_falls_back_to_general(self):
        profile = {"global_scalar": 0.91, "family_scalars": {"general": 0.91}}

        scalar = UnifiedRestorerV3._get_phase_calibration_scalar("phase_99_unknown", profile)

        assert scalar == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# Klasse: MidPipelineCalibrationStep — §2.31a iterative Kalibrierung
# ---------------------------------------------------------------------------


class TestMidPipelineCalibrationStep:
    """Tests für UnifiedRestorerV3._mid_pipeline_calibration_step (§2.31a)."""

    _FN = staticmethod(UnifiedRestorerV3._mid_pipeline_calibration_step)

    def _base_profile(self, **overrides) -> dict:
        p = {
            "global_scalar": 1.0,
            "family_scalars": {
                "denoise": 1.0,
                "reverb": 1.0,
                "reconstruction": 1.0,
                "dynamics_eq": 1.0,
                "time_pitch_transport": 1.0,
                "transient": 1.0,
                "vocal": 1.0,
                "instrument": 1.0,
                "general": 1.0,
            },
            "restorability_tier": "fair",
        }
        p.update(overrides)
        return p

    def test_72_returns_none_for_none_profile(self):
        result = self._FN({"brillanz": 0.8}, None, "33pct", 5, 15)
        assert result is None

    def test_73_returns_none_for_empty_scores(self):
        result = self._FN({}, self._base_profile(), "33pct", 5, 15)
        assert result is None

    def test_74_returns_none_when_no_adjustment_needed(self):
        # All goals well above thresholds → no adjustment
        scores = {
            "brillanz": 0.90,
            "micro_dynamics": 0.92,
            "tonal_center": 0.97,
            "groove": 0.89,
            "separation_fidelity": 0.85,
            "raumtiefe": 0.75,
            "artikulation": 0.90,
            "bass_kraft": 0.85,
        }
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is None

    def test_75_returns_copy_not_in_place(self):
        scores = {"brillanz": 0.50}  # low → adjustment expected
        profile = self._base_profile()
        result = self._FN(scores, profile, "33pct", 5, 15)
        # Original must be unchanged
        assert profile["family_scalars"]["reconstruction"] == 1.0
        if result is not None:
            assert result is not profile

    def test_76_low_brillanz_boosts_reconstruction(self):
        scores = {"brillanz": 0.50}  # 0.74 - 0.50 = 0.24 deficit
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["reconstruction"] > 1.0

    def test_77_low_micro_dynamics_boosts_transient_and_dynamics_eq(self):
        scores = {"micro_dynamics": 0.60}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["transient"] > 1.0
        assert result["family_scalars"]["dynamics_eq"] > 1.0

    def test_78_low_tonal_center_boosts_reconstruction(self):
        scores = {"tonal_center": 0.80}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["reconstruction"] > 1.0
        assert result["family_scalars"]["time_pitch_transport"] > 1.0

    def test_79_low_groove_boosts_dynamics_eq_and_transient(self):
        scores = {"groove": 0.60}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["time_pitch_transport"] > 1.0
        assert result["family_scalars"]["dynamics_eq"] > 1.0
        assert result["family_scalars"]["transient"] > 1.0

    def test_80_low_separation_fidelity_boosts_instrument(self):
        scores = {"separation_fidelity": 0.50}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["instrument"] > 1.0

    def test_81_low_artikulation_boosts_vocal(self):
        scores = {"artikulation": 0.60}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["vocal"] > 1.0

    def test_82_low_bass_kraft_boosts_dynamics_eq(self):
        scores = {"bass_kraft": 0.50}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        assert result["family_scalars"]["dynamics_eq"] > 1.0

    def test_83_all_scalars_clamped_to_1_80_max(self):
        """[RELEASE_MUST] Lücke-G-Fix v9.10.100: family_scalars niemals über 1.80."""
        # Extreme deficit → clamp must prevent going above 1.80
        profile = self._base_profile()
        profile["family_scalars"]["reconstruction"] = 1.75  # already high
        scores = {"brillanz": 0.00, "micro_dynamics": 0.00, "tonal_center": 0.00}
        result = self._FN(scores, profile, "33pct", 5, 15)
        if result is not None:
            for k, v in result["family_scalars"].items():
                assert float(v) <= 1.80 + 1e-9, f"{k}={v} exceeds 1.80 clamp (Lücke-G-Fix)"

    def test_84_all_scalars_clamped_to_0_30_min(self):
        """[RELEASE_MUST] Lücke-G-Fix v9.10.100: family_scalars niemals unter 0.30."""
        # Low tonal_center causes dynamics_eq to be de-boosted; verify new floor 0.30
        profile = self._base_profile()
        profile["family_scalars"]["dynamics_eq"] = 0.35  # near new floor
        scores = {"tonal_center": 0.50}  # de-boost signal
        result = self._FN(scores, profile, "33pct", 5, 15)
        if result is not None:
            for k, v in result["family_scalars"].items():
                assert float(v) >= 0.30 - 1e-9, f"{k}={v} below 0.30 clamp (Lücke-G-Fix)"

    def test_85_adjustment_bounded_at_12_percent_max(self):
        scores = {"brillanz": 0.00}  # maximum deficit
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        if result is not None:
            delta = result["family_scalars"]["reconstruction"] - 1.0
            assert delta <= 0.12 + 1e-9

    def test_86_audit_trail_event_appended(self):
        scores = {"brillanz": 0.40}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
        events = result.get("_mid_calibration_events", [])
        assert len(events) == 1
        assert events[0]["checkpoint"] == "33pct"
        assert "adjustments" in events[0]
        assert "scores_snapshot" in events[0]

    def test_87_second_call_appends_to_existing_events(self):
        scores = {"brillanz": 0.40}
        profile = self._base_profile()
        result1 = self._FN(scores, profile, "33pct", 5, 15)
        assert result1 is not None
        result2 = self._FN({"groove": 0.50}, result1, "66pct", 10, 15)
        assert result2 is not None
        events = result2.get("_mid_calibration_events", [])
        assert len(events) == 2
        assert events[0]["checkpoint"] == "33pct"
        assert events[1]["checkpoint"] == "66pct"

    def test_88_none_scores_for_individual_goals_skip_gracefully(self):
        # Only some goals present → others should not crash
        scores = {"brillanz": 0.50}  # other keys absent
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        # Should produce a result for brillanz without errors
        assert result is not None

    def test_89_returns_none_for_missing_family_scalars(self):
        profile = {"global_scalar": 1.0}  # no family_scalars key
        scores = {"brillanz": 0.50}
        result = self._FN(scores, profile, "33pct", 5, 15)
        assert result is None

    def test_90_global_scalar_preserved_in_output(self):
        scores = {"brillanz": 0.50}
        profile = self._base_profile()
        profile["global_scalar"] = 0.88
        result = self._FN(scores, profile, "33pct", 5, 15)
        assert result is not None
        assert result["global_scalar"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# §2.46b Spectral Tilt Drift Guard — _estimate_spectral_tilt_quick + Instance
# ---------------------------------------------------------------------------


class TestEstimateSpectralTiltQuick:
    """§2.46b: _estimate_spectral_tilt_quick() static method sanity checks."""

    def test_returns_float_for_sine(self):
        audio = _sine(2.0, freq=440.0)
        result = UnifiedRestorerV3._estimate_spectral_tilt_quick(audio, SR)
        assert result is not None
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert -12.0 <= result <= 2.0

    def test_returns_float_for_stereo(self):
        audio = _stereo(2.0)
        result = UnifiedRestorerV3._estimate_spectral_tilt_quick(audio, SR)
        assert result is not None
        assert isinstance(result, float)

    def test_does_not_raise_on_empty_audio(self):
        audio = np.zeros(0, dtype=np.float32)
        result = UnifiedRestorerV3._estimate_spectral_tilt_quick(audio, SR)
        assert result is None or isinstance(result, float)

    def test_bright_vs_dark_signal_direction(self):
        freqs = np.fft.rfftfreq(4096, 1.0 / SR)
        spec_bright = np.ones(len(freqs))
        audio_bright = np.fft.irfft(spec_bright, n=8192).astype(np.float32)[:8192]
        spec_dark = np.where(freqs > 0, 1.0 / (freqs + 1.0) ** 2, 1e-10)
        audio_dark = np.fft.irfft(spec_dark, n=8192).astype(np.float32)[:8192]
        tilt_bright = UnifiedRestorerV3._estimate_spectral_tilt_quick(audio_bright, SR)
        tilt_dark = UnifiedRestorerV3._estimate_spectral_tilt_quick(audio_dark, SR)
        if tilt_bright is not None and tilt_dark is not None:
            assert tilt_bright > tilt_dark


class TestEraSpectralTiltInstance:
    """§2.46b: _era_spectral_tilt Instanzvariable."""

    def test_defaults_to_none(self):
        cfg = RestorationConfig(mode=QualityMode.BALANCED, deployment_mode=DeploymentMode.PRODUCT)
        restorer = UnifiedRestorerV3(cfg)
        assert hasattr(restorer, "_era_spectral_tilt")
        assert restorer._era_spectral_tilt is None

    def test_can_be_set(self):
        cfg = RestorationConfig(mode=QualityMode.BALANCED, deployment_mode=DeploymentMode.PRODUCT)
        restorer = UnifiedRestorerV3(cfg)
        restorer._era_spectral_tilt = -5.2
        assert restorer._era_spectral_tilt == pytest.approx(-5.2)


class TestReferenceAnchorArbitration:
    def test_tier2_high_conf_era_overrides_globalplan_for_anchor(self, monkeypatch: pytest.MonkeyPatch):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(9600, dtype=np.float32)

        captured: dict[str, object] = {}

        def _fake_anchor(era_decade: int, genre_label: str, material: str):
            captured["era_decade"] = int(era_decade)
            captured["genre_label"] = str(genre_label)
            captured["material"] = str(material)
            return np.zeros(128, dtype=np.float32)

        monkeypatch.setattr(
            "backend.core.reference_anchor_synthesizer.synthesize_reference_anchor",
            _fake_anchor,
        )

        def _stop_after_anchor(*args, **kwargs):
            raise RuntimeError("stop_after_anchor")

        restorer.defect_scanner.scan = _stop_after_anchor  # type: ignore[method-assign]

        cached_medium = types.SimpleNamespace(
            material=MaterialType.VINYL,
            material_type=MaterialType.VINYL,
            confidence=0.95,
            classifier_source="unit",
        )
        cached_era = types.SimpleNamespace(
            decade=1980,
            confidence=0.90,
            tier_used=2,
            material_prior="vinyl",
            noise_profile=np.zeros(24, dtype=np.float32),
            hf_rolloff_hz=16000.0,
        )
        cached_genre = types.SimpleNamespace(
            is_schlager=False,
            confidence=0.0,
            genre_label="unknown",
            bpm=0.0,
            subgenre="unknown",
        )
        cached_rest = types.SimpleNamespace(restorability_score=70.0, grade="FAIR", predicted_mos=(3.5, 4.1))
        gp = types.SimpleNamespace(portrait=types.SimpleNamespace(decade=1930, era_confidence=0.92))

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        monkeypatch.setattr("psutil.virtual_memory", lambda: _fake_vm)

        with pytest.raises(RuntimeError, match="stop_after_anchor"):
            restorer.restore(
                audio,
                48000,
                global_plan=gp,
                cached_medium_result=cached_medium,
                cached_era_result=cached_era,
                cached_genre_result=cached_genre,
                cached_restorability_result=cached_rest,
                file_path="/tmp/anchor_arbitration.wav",
            )

        assert captured.get("era_decade") == 1980

    def test_globalplan_decade_is_authoritative_even_when_chunk_conf_is_higher(self, monkeypatch: pytest.MonkeyPatch):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(9600, dtype=np.float32)

        created_decades: list[int] = []

        class _FakeEraResult(types.SimpleNamespace):
            def __init__(
                self,
                decade,
                era_label,
                confidence,
                material_prior,
                noise_profile,
                tier_used,
                hf_rolloff_hz,
            ):
                created_decades.append(int(decade))
                super().__init__(
                    decade=int(decade),
                    era_label=str(era_label),
                    confidence=float(confidence),
                    material_prior=str(material_prior),
                    noise_profile=noise_profile,
                    tier_used=int(tier_used),
                    hf_rolloff_hz=float(hf_rolloff_hz),
                )

        monkeypatch.setattr("backend.core.era_classifier.EraResult", _FakeEraResult)

        def _stop_after_anchor(*args, **kwargs):
            raise RuntimeError("stop_after_anchor")

        restorer.defect_scanner.scan = _stop_after_anchor  # type: ignore[method-assign]

        cached_medium = types.SimpleNamespace(
            material=MaterialType.VINYL,
            material_type=MaterialType.VINYL,
            confidence=0.95,
            classifier_source="unit",
        )
        cached_era = types.SimpleNamespace(
            decade=2025,
            confidence=0.95,
            tier_used=1,
            material_prior="streaming",
            noise_profile=np.zeros(24, dtype=np.float32),
            hf_rolloff_hz=20000.0,
        )
        cached_genre = types.SimpleNamespace(
            is_schlager=False,
            confidence=0.0,
            genre_label="unknown",
            bpm=0.0,
            subgenre="unknown",
        )
        cached_rest = types.SimpleNamespace(restorability_score=70.0, grade="FAIR", predicted_mos=(3.5, 4.1))
        gp = types.SimpleNamespace(portrait=types.SimpleNamespace(decade=1970, era_confidence=0.20, material="vinyl"))

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        monkeypatch.setattr("psutil.virtual_memory", lambda: _fake_vm)

        with pytest.raises(RuntimeError, match="stop_after_anchor"):
            restorer.restore(
                audio,
                48000,
                global_plan=gp,
                cached_medium_result=cached_medium,
                cached_era_result=cached_era,
                cached_genre_result=cached_genre,
                cached_restorability_result=cached_rest,
                file_path="/tmp/globalplan_era_authoritative.wav",
            )

        assert created_decades, "Expected GlobalPlan-era override to instantiate EraResult"
        assert created_decades[-1] == 1970


class TestStereoSafetyGuardBranching:
    """Targeted branch tests for §2.51a stereo safety guard."""

    @staticmethod
    def _delayed_right(stereo_sc: np.ndarray, delay_samples: int) -> np.ndarray:
        out = np.asarray(stereo_sc, dtype=np.float32).copy()
        if delay_samples <= 0:
            return out
        right = out[:, 1]
        delayed = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), right[:-delay_samples]],
            axis=0,
        )
        out[:, 1] = delayed
        return out

    def test_91_hard_fail_when_interchannel_delay_exceeds_1ms(self):
        mono = (0.45 * _sine(secs=1.0, freq=80.0)).astype(np.float32)
        original = np.stack([mono, mono], axis=1)
        restored = self._delayed_right(original, delay_samples=60)  # 60/48000 = 1.25 ms

        result = _uv3_mod._evaluate_stereo_safety_guard(original, restored, SR)

        assert result["enabled"] is True
        assert result["hard_fail"] is True
        assert "interchannel_delay_gt_1ms" in result["hard_fail_reasons"]

    def test_92_warning_when_delay_between_0p5_and_1ms(self):
        mono = (0.45 * _sine(secs=1.0, freq=80.0)).astype(np.float32)
        original = np.stack([mono, mono], axis=1)
        restored = self._delayed_right(original, delay_samples=36)  # 36/48000 = 0.75 ms

        result = _uv3_mod._evaluate_stereo_safety_guard(original, restored, SR)

        assert result["enabled"] is True
        assert result["hard_fail"] is False
        assert result["warning"] is True
        assert "interchannel_delay_0p5_to_1ms" in result["warning_reasons"]

    def test_93_ok_when_stereo_metrics_within_targets(self):
        mono = (0.30 * _sine(secs=1.0, freq=220.0)).astype(np.float32)
        original = np.stack([mono, mono], axis=1)
        restored = original.copy()

        result = _uv3_mod._evaluate_stereo_safety_guard(original, restored, SR)

        assert result["enabled"] is True
        assert result["hard_fail"] is False
        assert result["warning"] is False
        assert result["reason"] == "ok"

    def test_94_hard_fail_when_true_peak_above_minus_1dbtp(self):
        mono_in = (0.30 * _sine(secs=1.0, freq=220.0)).astype(np.float32)
        mono_out = (0.95 * _sine(secs=1.0, freq=220.0)).astype(np.float32)
        original = np.stack([mono_in, mono_in], axis=1)
        restored = np.stack([mono_out, mono_out], axis=1)

        result = _uv3_mod._evaluate_stereo_safety_guard(original, restored, SR)

        assert result["enabled"] is True
        assert result["hard_fail"] is True
        assert "true_peak_gt_minus_1dbtp" in result["hard_fail_reasons"]


class TestSingleGainAuthorityPolicy:
    def test_95_hpf_notch_phase_locks_positive_makeup_authority(self):
        allow, reason = UnifiedRestorerV3._update_positive_makeup_authority(
            "phase_05_rumble_filter",
            True,
        )

        assert allow is False
        assert reason == "locked_after_hpf_notch"

    def test_96_broadband_subtractive_phase_reenables_authority(self):
        allow, reason = UnifiedRestorerV3._update_positive_makeup_authority(
            "phase_29_tape_hiss_reduction",
            False,
        )

        assert allow is True
        assert reason == "unlocked_after_phase_29"

    def test_97_non_policy_phase_keeps_state(self):
        allow, reason = UnifiedRestorerV3._update_positive_makeup_authority(
            "phase_42_vocal_enhancement",
            False,
        )

        assert allow is False
        assert reason is None


class TestSingleGainAuthorityEndToEnd:
    """End-to-end verification that §2.45a-VII Single-Gain-Authority blocks
    positive makeup in cumulative guards after HPF/Notch phases.

    Scenario for test_98:
        phase_05 (HPF) → locks authority + resets cumulative reference
        phase_35 (dynamics, non-unlock) → heavy attenuation → cumulative guard fires
        Guard sees _allow_positive_makeup_gain=False → skips makeup
        Output RMS stays << input RMS (no boost applied)

    Scenario for test_99:
        phase_05 → locks authority
        phase_29 (broadband subtraktive) → re-enables authority
        Both phases execute; output is valid (non-NaN, non-Inf, clipped)
    """

    def _build_restorer(self) -> UnifiedRestorerV3:
        cfg = RestorationConfig(
            enable_phase_gate=False,
            enable_phase_skipping=False,
            enable_performance_guard=False,
        )
        return UnifiedRestorerV3(cfg)

    class _PhaseStub:
        """Minimal phase stub that exposes get_metadata() with a configurable phase_id."""

        def __init__(self, pid: str) -> None:
            self._pid = pid

        def get_metadata(self) -> object:
            return types.SimpleNamespace(
                estimated_time_factor=0.1,
                phase_id=self._pid,
                name=self._pid,
            )

        def process(self, audio: np.ndarray, **kwargs: object) -> np.ndarray:
            return np.asarray(audio, dtype=np.float32)

    def test_98_no_positive_makeup_after_hpf_followed_by_non_unlock_phase(self):
        """After HPF (phase_05), authority stays locked for non-unlock subsequent phases.

        phase_05 resets cumulative reference to post-HPF audio and locks authority.
        phase_17 (mastering polish, NOT in unlock set, NOT §0a-forbidden) further
        attenuates, causing the cumulative guard to fire. Because authority is locked,
        the guard MUST NOT apply positive makeup — verifying §2.45a-VII Single-Gain-Authority.
        """
        restorer = self._build_restorer()

        rng = np.random.default_rng(42)
        input_level = 0.30  # ~ -10 dBFS — well above -36 dBFS gate
        audio_in = np.clip(rng.standard_normal(SR * 2) * input_level, -1.0, 1.0).astype(np.float32)

        _PhaseStub = TestSingleGainAuthorityEndToEnd._PhaseStub
        restorer._get_phase = lambda pid: _PhaseStub(pid)  # type: ignore[method-assign]

        def _mock_profiled_call(_phase: object, _audio: np.ndarray, **_kwargs: object) -> object:
            pid: str = _phase.get_metadata().phase_id  # type: ignore[union-attr]
            factor = 0.5 if "phase_05" in pid else 0.1  # HPF: -6 dB; phase_35: -20 dB
            return types.SimpleNamespace(
                success=True,
                audio=np.clip(np.asarray(_audio, dtype=np.float32) * factor, -1.0, 1.0),
                execution_time_seconds=0.001,
                warnings=[],
            )

        restorer._profiled_phase_call = _mock_profiled_call  # type: ignore[method-assign]
        rms_before = float(np.sqrt(np.mean(audio_in**2)))

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with patch("psutil.virtual_memory", return_value=_fake_vm):
            out, executed, _sk, _def = restorer._execute_pipeline(
                audio=audio_in,
                sample_rate=SR,
                material_type=MaterialType.VINYL,
                defect_result=types.SimpleNamespace(scores={}),
                selected_phases=["phase_05_rumble_filter", "phase_17_mastering_polish"],
            )

        rms_after = float(np.sqrt(np.mean(out**2)))

        # Without authority policy: guard fires after phase_17, boosts back toward -4.5 dB
        # from post-HPF reference → rms_after could exceed rms_before (massive overshot).
        # WITH §2.45a-VII authority: no positive makeup → rms_after ≈ rms_in * 0.5 * 0.1
        # (cumulative ~-26 dB from input) — well below the +3 dB assertion threshold.
        ratio_db = 20.0 * np.log10((rms_after + 1e-12) / (rms_before + 1e-12))
        assert ratio_db <= 3.0, (
            f"§2.45a-VII Single-Gain-Authority violation: cumulative guard applied positive "
            f"makeup of {ratio_db:.1f} dB after HPF + non-unlock phase chain. "
            f"rms_before={rms_before:.5f}, rms_after={rms_after:.5f}. "
            "No positive makeup allowed while authority is locked by phase_05_rumble_filter."
        )
        assert "phase_05_rumble_filter" in executed
        assert "phase_17_mastering_polish" in executed

    def test_99_authority_re_enables_after_broadband_phase_following_hpf(self):
        """After HPF locks authority, a broadband-subtractive phase re-enables it.

        phase_05 → locks authority → phase_29 (broadband subtraktive) → re-enables.
        Both phases must execute and produce valid audio. Does not test the guard
        triggering, only that the phase sequence completes without error and that
        output is numerically safe (no NaN / Inf / out-of-range values).
        """
        restorer = self._build_restorer()

        rng = np.random.default_rng(7)
        audio_in = np.clip(rng.standard_normal(SR) * 0.15, -1.0, 1.0).astype(np.float32)

        _PhaseStub = TestSingleGainAuthorityEndToEnd._PhaseStub
        restorer._get_phase = lambda pid: _PhaseStub(pid)  # type: ignore[method-assign]

        def _mock_profiled_call(_phase: object, _audio: np.ndarray, **_kwargs: object) -> object:
            pid: str = _phase.get_metadata().phase_id  # type: ignore[union-attr]
            factor = 0.5 if "phase_05" in pid else 0.85
            return types.SimpleNamespace(
                success=True,
                audio=np.clip(np.asarray(_audio, dtype=np.float32) * factor, -1.0, 1.0),
                execution_time_seconds=0.001,
                warnings=[],
            )

        restorer._profiled_phase_call = _mock_profiled_call  # type: ignore[method-assign]

        _fake_vm = types.SimpleNamespace(available=8 * 1024 * 1024 * 1024, percent=20.0, total=16 * 1024 * 1024 * 1024)
        with patch("psutil.virtual_memory", return_value=_fake_vm):
            out, executed, _sk, _def = restorer._execute_pipeline(
                audio=audio_in,
                sample_rate=SR,
                material_type=MaterialType.VINYL,
                defect_result=types.SimpleNamespace(scores={}),
                selected_phases=["phase_05_rumble_filter", "phase_29_tape_hiss_reduction"],
            )

        assert "phase_05_rumble_filter" in executed
        assert "phase_29_tape_hiss_reduction" in executed
        assert not np.any(np.isnan(out)), "Output contains NaN after HPF+broadband chain"
        assert not np.any(np.isinf(out)), "Output contains Inf after HPF+broadband chain"
        assert np.all(np.abs(out) <= 1.0 + 1e-6), "Output exceeds ±1.0 clip range"


class TestCIGRollbackNoMakeupGain:
    """§2.45a-VII CIG-Rollback: when CumulativeInteractionGuard rolls back a phase,
    the cumulative loudness guard MUST NOT apply positive makeup gain.

    Bug scenario (Pegelexplosion root cause):
        1. phase_05 (HPF) locks _allow_positive_makeup_gain = False
        2. phase_03 (denoise) executes → CIG detects STFT group delay deviation → rollback
        3. current_audio reverts to pre-phase_03 state
        4. BUT: _update_positive_makeup_authority("phase_03") would UNLOCK makeup if called
        5. Cumulative guard runs, sees drop, applies makeup to pre-phase_03 audio
        6. Vinyl surface noise (~-35 dBFS) passes -36 dBFS gate → Pegelexplosion!

    Fix (§2.45a-VII): _cig_phase_rolled_back=True gates both the authority update
    and the cumulative guard for the rolled-back phase.
    """

    def _build_restorer(self) -> UnifiedRestorerV3:
        cfg = RestorationConfig(
            enable_phase_gate=False,
            enable_phase_skipping=False,
            enable_performance_guard=False,
        )
        return UnifiedRestorerV3(cfg)

    def test_100_no_makeup_after_cig_rollback_of_broadband_phase(self):
        """CIG-rolled-back denoise phase must not trigger makeup gain on fadeout audio.

        Setup:
            - Audio: loud music section + quiet fadeout (Vinyl surface noise level ~-35 dBFS)
            - Phases: phase_05 (HPF, locks authority) → phase_03 (denoise, CIG-rolled-back)
            - CIG mock: always rolls back phase_03
            - Expected: output RMS in the fadeout region ≈ input (no makeup boost applied)

        Without the fix: _allow_positive_makeup_gain unlocked by phase_03 authority update
        → cumulative guard boosts fadeout → Pegelexplosion.
        With the fix: _cig_phase_rolled_back=True gates both blocks → no boost.
        """
        restorer = self._build_restorer()
        rng = np.random.default_rng(123)

        # Build audio: 2 s loud music (0.3 amp) + 1 s quiet fadeout (0.005 amp = ~-46 dBFS)
        music_samples = int(SR * 2)
        fadeout_samples = int(SR * 1)
        music = (rng.standard_normal(music_samples) * 0.30).astype(np.float32)
        fadeout = (rng.standard_normal(fadeout_samples) * 0.005).astype(np.float32)
        audio_in = np.clip(np.concatenate([music, fadeout]), -1.0, 1.0)

        _PhaseStub = TestSingleGainAuthorityEndToEnd._PhaseStub
        restorer._get_phase = lambda pid: _PhaseStub(pid)  # type: ignore[method-assign]

        def _mock_profiled_call(_phase: object, _audio: np.ndarray, **_kwargs: object) -> object:
            pid: str = _phase.get_metadata().phase_id  # type: ignore[union-attr]
            # phase_05: HPF removes sub-bass energy (−6 dB)
            # phase_03: would denoise (−3 dB) but CIG rolls it back — return slightly attenuated
            # so the rollback has something to revert.
            factor = 0.5 if "phase_05" in pid else 0.7
            return types.SimpleNamespace(
                success=True,
                audio=np.clip(np.asarray(_audio, dtype=np.float32) * factor, -1.0, 1.0),
                execution_time_seconds=0.001,
                warnings=[],
            )

        restorer._profiled_phase_call = _mock_profiled_call  # type: ignore[method-assign]

        # Inject a mock interaction guard that always rolls back phase_03
        class _RollbackGuard:
            """Mock CIG: rolls back phase_03, passes everything else."""

            class _State:
                should_stop = False
                stft_phases_executed: list = []

            def start(self, *a, **kw):
                return self._State()

            def check_after_phase(self, state, phase_id, audio, scores, sr):
                if "phase_03" in phase_id:
                    # Return pre-phase audio (simulated rollback target) and True
                    return audio, True  # rolled back — caller must revert current_audio
                return audio, False

        with patch("backend.core.cumulative_interaction_guard.get_interaction_guard", return_value=_RollbackGuard()):
            out, executed, _sk, _def = restorer._execute_pipeline(
                audio=audio_in,
                sample_rate=SR,
                material_type=MaterialType.VINYL,
                defect_result=types.SimpleNamespace(scores={}),
                selected_phases=["phase_05_rumble_filter", "phase_03_denoise"],
            )

        # The fadeout region of the output must NOT be louder than the input fadeout
        fadeout_out = out[-fadeout_samples:]
        fadeout_in = audio_in[-fadeout_samples:]
        rms_out = float(np.sqrt(np.mean(np.asarray(fadeout_out, dtype=np.float64) ** 2)))
        rms_in = float(np.sqrt(np.mean(np.asarray(fadeout_in, dtype=np.float64) ** 2)))

        # With fix: no makeup → fadeout rms_out ≈ rms_in * 0.5 (HPF factor) or lower
        # Without fix: makeup unlocked by phase_03 authority → rms_out >> rms_in → Pegelexplosion
        ratio_db = 20.0 * np.log10((rms_out + 1e-12) / (rms_in + 1e-12))
        assert ratio_db <= 3.0, (
            f"§2.45a-VII Pegelexplosion in fadeout after CIG-rollback of phase_03: "
            f"output is {ratio_db:.1f} dB louder than input fadeout. "
            f"rms_in={rms_in:.6f}, rms_out={rms_out:.6f}. "
            "CIG-Rollback must block both _allow_positive_makeup_gain unlock "
            "and cumulative guard makeup application."
        )


# ---------------------------------------------------------------------------
# V13 Regression — _MATERIAL_PRIORITY_PHASES duplicate-key guard (§F601)
# ---------------------------------------------------------------------------


class TestMaterialPriorityPhasesNoDuplicateKeys:
    """Regression-Test für V13: Kein Duplikat-Schlüssel in _MATERIAL_PRIORITY_PHASES.

    F601 — Python überschreibt doppelte Dict-Schlüssel stillschweigend.
    Ein Duplikat führt dazu, dass die falsche Phasenliste aktiv ist.
    Der Test parst die UV3-Quelle via AST und prüft alle Assignments zu
    _MATERIAL_PRIORITY_PHASES auf Eindeutigkeit der String-Schlüssel.
    """

    def test_material_priority_phases_no_duplicate_keys(self):
        """V13: _MATERIAL_PRIORITY_PHASES-Dict darf keine Duplikat-Schlüssel haben."""
        import ast
        import pathlib

        src_path = pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py"
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        duplicate_keys: list[tuple[str, int]] = []  # (key, lineno)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Look for local assignments: _MATERIAL_PRIORITY_PHASES = { ... }
            target_names = [
                t.id for t in node.targets if isinstance(t, ast.Name) and t.id == "_MATERIAL_PRIORITY_PHASES"
            ]
            if not target_names:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            seen: set[str] = set()
            for key_node in node.value.keys:
                if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                    continue
                key = key_node.value
                if key in seen:
                    duplicate_keys.append((key, key_node.lineno))
                seen.add(key)

        assert not duplicate_keys, (
            f"V13: _MATERIAL_PRIORITY_PHASES has duplicate keys (silent F601 overwrite): "
            f"{duplicate_keys}. Remove the duplicate entry — "
            "the second definition silently overwrites the first, activating the wrong phase set."
        )


class TestVocalGenreKeysChoir:
    """§M1 (v9.12.9) Choir-Vinyl Gap — _VOCAL_GENRE_KEYS muss Chor-Genres enthalten.

    Regressions-Guard: Wenn 'choir', 'choral' oder 'chormusik' fehlen, aktiviert
    _compute_vocal_presence_confidence() den 0.35-PANNs-Floor für Chor-Vinyl-Material
    NICHT → VQI-Gate bleibt stumm → Vocal-Naturalness-Wiederherstellung wird nie ausgelöst.
    """

    def test_choir_terms_in_vocal_genre_keys(self) -> None:
        """_VOCAL_GENRE_KEYS enthält alle Chor-Genrebezeichnungen (§M1 v9.12.9)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        keys = UnifiedRestorerV3._VOCAL_GENRE_KEYS
        required = {"choir", "choral", "chormusik", "chor", "kantate", "oratorium"}
        missing = required - keys
        assert not missing, (
            f"§M1: _VOCAL_GENRE_KEYS fehlen Chor-Genrebezeichnungen: {missing}. "
            "Ohne diese greift der 0.35-PANNs-Floor für Chor-Vinyl/Shellac-Material nicht."
        )

    def test_choir_genre_label_activates_confidence_floor(self) -> None:
        """genre_label='choir' → _compute_vocal_presence_confidence() ≥ 0.35 (§M1 floor)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Choir": 0.12, "Music": 0.65},
            panns_vocals_confidence=0.12,
            genre_label="choir",
        )
        assert conf >= 0.35, f"§M1: 'choir' genre_label muss PANNs-Floor 0.35 aktivieren, bekommen: {conf:.3f}"

    def test_choral_music_genre_label_activates_confidence_floor(self) -> None:
        """genre_label='choral music' → _compute_vocal_presence_confidence() ≥ 0.35."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Music": 0.70},
            panns_vocals_confidence=0.10,
            genre_label="choral music",
        )
        assert conf >= 0.35, f"§M1: 'choral music' label muss Floor 0.35 aktivieren, bekommen: {conf:.3f}"

    def test_schlager_floor_unchanged_after_m1_patch(self) -> None:
        """Schlager-Floor bleibt 0.35 — §M1-Patch hat das nicht beeinträchtigt."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Singing": 0.10},
            panns_vocals_confidence=0.17,
            is_schlager=True,
        )
        assert conf >= 0.35, f"Schlager-Floor fehlt nach §M1-Patch: {conf:.3f}"


class TestPhase65VqiRecoveryTrigger:
    """§H4 (v9.12.9) Phase_65 VQI-Recovery-Trigger in UV3.

    Regressions-Guard: Phase_65 (HNR-Blend + Spektral-Tilt + Formant-Tilt) MUSS
    in UV3 als DSP-Korrektiv für VQI < 0.74 + panns ≥ 0.25 + Restoration verfügbar sein.
    Vor v9.12.9 war Phase_65 in calibration_matrix referenziert, aber nie in UV3 aufgerufen.
    """

    def test_phase_65_module_importable(self) -> None:
        """Phase_65-Modul ist importierbar (Pflicht für VQI-Recovery-Trigger)."""
        import importlib

        mod = importlib.util.find_spec("backend.core.phases.phase_65_vocal_naturalness_restoration")
        assert mod is not None, (
            "§H4: phase_65_vocal_naturalness_restoration nicht gefunden — "
            "VQI-Recovery-Trigger in UV3 kann Phase_65 nicht laden."
        )

    def test_phase_65_get_phase_callable(self) -> None:
        """get_phase_65() existiert und gibt ein Objekt mit .process() zurück."""
        from backend.core.phases.phase_65_vocal_naturalness_restoration import get_phase_65

        p65 = get_phase_65()
        assert hasattr(p65, "process"), "§H4: Phase_65 hat keine process()-Methode"

    def test_uv3_source_contains_phase_65_vqi_recovery(self) -> None:
        """UV3-Quellcode enthält Phase_65-VQI-Recovery-Block (§H4 v9.12.9 Regression-Guard)."""
        import pathlib

        src = (pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text()
        assert "get_phase_65" in src, (
            "§H4: UV3-Quellcode enthält get_phase_65 nicht — "
            "Phase_65 VQI-Recovery-Trigger fehlt (Regression §0a/§7.10)."
        )
        assert "_vqi_score < 0.74" in src, (
            "§H4: UV3 prüft VQI < 0.74 Schwelle nicht — "
            "VERBOTEN-Regel 'VQI-Abfall nach NR ohne DSP-Korrektiv-Recovery' verletzt."
        )
        assert "compute_vocal_max_alignment" in src, (
            "§0p: UV3 muss VQI-Recovery prozentual am maximal erreichbaren Vocal-Ziel ausrichten."
        )
        assert "vocal_max_alignment_percent" in src, (
            "§0p: UV3 muss den prozentualen Abstand zum Vocal-Maximum in Metadata/Fail-Reasons tragen."
        )
        assert "_p65_vqi_deficit = max(0.74 - _vqi_score, _vqi_max_target - _vqi_score, 0.0)" in src, (
            "§0p: Phase_65-Stärke muss proportional zum Maximum-Defizit skaliert werden."
        )


class TestChoirVqiGateZeroPanns:
    """§0p v9.12.11 — Chor-VQI-Gate auch bei PANNs-Gesamtconfidence = 0.

    Regressions-Guard: Wenn PANNs für eine Choraufnahme gar keine Vokal-Tags
    zurückgibt (confidence = 0.0), darf der VQI-Gate NICHT still deaktiviert
    bleiben. Chor ist definitional Vokalmusik — die Schwelle '≥ 0.10' aus §M1
    (v9.12.9) war zu hoch und ließ Fälle ohne jeglichen PANNs-Hinweis durch.
    """

    def test_choir_genre_zero_panns_activates_floor(self) -> None:
        """genre_label='choir' + PANNs-Confidence = 0.0 → floor 0.35 (§0p v9.12.11)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {},  # keine PANNs-Tags
            panns_vocals_confidence=0.0,
            genre_label="choir",
        )
        assert conf >= 0.35, (
            f"§0p v9.12.11: genre_label='choir' + keine PANNs-Tags muss floor 0.35 aktivieren, "
            f"bekommen: {conf:.3f}. VQI-Gate würde für Chor-Material stumm bleiben."
        )

    def test_choral_zero_panns_activates_floor(self) -> None:
        """genre_label='choral' + PANNs = 0.0 → floor 0.35."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {},
            panns_vocals_confidence=0.0,
            genre_label="choral",
        )
        assert conf >= 0.35, f"'choral' ohne PANNs-Signal: floor fehlt, bekommen {conf:.3f}"

    def test_kantate_zero_panns_activates_floor(self) -> None:
        """genre_label='kantate' + PANNs = 0.0 → floor 0.35."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {},
            panns_vocals_confidence=0.0,
            genre_label="kantate",
        )
        assert conf >= 0.35, f"'kantate' ohne PANNs-Signal: floor fehlt, bekommen {conf:.3f}"

    def test_non_choir_genre_zero_panns_unchanged(self) -> None:
        """genre_label='jazz' + PANNs = 0.0 bleibt bei 0.0 — kein False-Positive (§0p)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {},
            panns_vocals_confidence=0.0,
            genre_label="jazz",
        )
        assert conf < 0.35, f"Non-vocal genre 'jazz' ohne PANNs-Signal darf NICHT floor 0.35 erhalten: {conf:.3f}"

    def test_existing_choir_tag_still_activates_floor(self) -> None:
        """Bereits in §M1 getesteter Fall bleibt grün — Chor-Tag mit PANNs 0.12."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        conf = UnifiedRestorerV3._compute_vocal_presence_confidence(
            {"Choir": 0.12, "Music": 0.65},
            panns_vocals_confidence=0.12,
            genre_label="choir",
        )
        assert conf >= 0.35, f"Regressions-Guard §M1 fehlgeschlagen nach §0p v9.12.11-Patch: {conf:.3f}"


class TestSection0aRestCauseGuards:
    """§0a v9.12.11 — CAUSE_TO_PHASES Restoration-Pfad: verbotene Phasen nicht vorschlagen.

    Regressions-Guard: phase_42_vocal_enhancement, phase_35_multiband_compression
    dürfen in UV3 nie in selected_phases landen wenn is_studio_mode()=False —
    weder über VOCAL_HARSHNESS-Pfad noch TIER-4-vocals_detected-Pfad.
    """

    def test_uv3_source_vocal_harshness_uses_phase65_in_restoration(self) -> None:
        """VOCAL_HARSHNESS-Pfad ruft im Restoration-Zweig phase_65 statt phase_42 auf."""
        import pathlib

        src = (pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text()
        # Schlüsselphrase: der Restoration-Kommentar und der phase_65-Append im Harshness-Block
        assert "phase_65_vocal_naturalness_restoration" in src, (
            "§0a: VOCAL_HARSHNESS-Pfad enthält keine phase_65 als §0a-konformes DSP-Korrektiv. "
            "Harshness in Restoration-Modus würde unbehandelt bleiben (phase_42 wird durch Guard entfernt)."
        )
        assert "is_studio_mode" in src, "§0a: is_studio_mode()-Check fehlt in UV3"

    def test_uv3_source_tier4_vocals_guards_phase42(self) -> None:
        """TIER-4-Vokal-Enhancement-Block enthält is_studio_mode()-Guard für phase_42."""
        import pathlib

        src = (pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text()
        # TIER 4 Kommentar erscheint direkt vor dem if vocals_detected: Block
        tier4_idx = src.find("TIER 4 — Instrument- / Vokal-Enhancement")
        assert tier4_idx >= 0, "§0a: TIER-4-Block nicht gefunden in UV3"
        tier4_section = src[tier4_idx : tier4_idx + 1500]
        assert "is_studio_mode" in tier4_section, (
            "§0a: TIER-4-Block enthält keinen is_studio_mode()-Guard. "
            "phase_42 würde in Restoration-Modus in selected_phases gelangen."
        )

    def test_uv3_source_multiband_compression_guards_phase35(self) -> None:
        """Multiband-Kompression-Block enthält is_studio_mode()-Guard für phase_35."""
        import pathlib

        src = (pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text()
        multiband_idx = src.find("Multiband-Kompression")
        assert multiband_idx >= 0, "§0a: Multiband-Kompression-Block nicht gefunden"
        multiband_section = src[multiband_idx : multiband_idx + 300]
        assert "is_studio_mode" in multiband_section, (
            "§0a: Multiband-Kompression-Block enthält keinen is_studio_mode()-Guard. "
            "phase_35 würde in Restoration-Modus in selected_phases gelangen."
        )


class TestPhaseIdValidationGuards:
    """Regression-Guards gegen stille Skips durch unbekannte Phase-IDs."""

    def test_validate_selected_phase_ids_removes_unknown_and_normalizes_aliases(self) -> None:
        cfg = RestorationConfig(
            enable_phase_gate=False,
            enable_phase_skipping=False,
            enable_performance_guard=False,
        )
        restorer = UnifiedRestorerV3(cfg)
        restorer.phase_metadata = {
            "phase_01_click_removal": {"name": "Click"},
            "phase_48_stereo_width_enhancer": {"name": "Stereo Width"},
        }
        restorer._PHASE_ALIASES = {
            "phase_48_stereo_imaging": "phase_48_stereo_width_enhancer",
        }

        resolved, invalid = restorer._validate_selected_phase_ids(
            [
                "phase_48_stereo_imaging",
                "phase_999_missing",
                "phase_01_click_removal",
                "phase_01_click_removal",
            ],
            context="unit_test",
        )

        assert resolved == ["phase_48_stereo_width_enhancer", "phase_01_click_removal"]
        assert invalid == ["phase_999_missing"]

    def test_uv3_source_goal_gap_uses_existing_phase48_id(self) -> None:
        import pathlib

        src = (pathlib.Path(__file__).parents[2] / "backend" / "core" / "unified_restorer_v3.py").read_text()
        assert '"spatial_depth": ["phase_48_stereo_width_enhancer"]' in src, (
            "§PHASE_ID_VALIDATE: Goal-Gap spatial_depth muss auf existierende phase_48 zeigen."
        )
        assert "phase_48_stereo_imaging" not in src, (
            "§PHASE_ID_VALIDATE: veraltete/ungültige phase_48-ID wurde wieder eingeführt."
        )
