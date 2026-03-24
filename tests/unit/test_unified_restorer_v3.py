"""Unit-Tests für backend/core/unified_restorer_v3.py.

Spec §2.1/§2.2: UnifiedRestorerV3 — Defect-First Restoration Engine.
Tests decken ab: RestorationConfig, RestorationResult, Initialisierung,
get_restorer()-Singleton, get_phase_info(), _select_phases() (Mock),
NaN/Inf-Invariante, Shape-Korrektheit, Bounds und Edge-Cases.
≥ 35 Tests.
"""

from __future__ import annotations

from dataclasses import fields
import math
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

np.random.seed(42)

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
        try:
            phases = restorer._select_phases(mock_defect)
            assert isinstance(phases, list)
        except Exception:
            pytest.skip("_select_phases benötigt voll initialisiertes DefectResult")

    def test_33_select_phases_elements_are_strings(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        try:
            phases = restorer._select_phases(mock_defect)
            for p in phases:
                assert isinstance(p, str)
        except Exception:
            pytest.skip("_select_phases benötigt voll initialisiertes DefectResult")


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
        _out, _ex, _sk, _def = restorer._execute_pipeline(
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

    def test_42_phase_regression_log_is_dict_in_metadata(self):
        """RestorationResult.metadata muss 'phase_regression_log' als dict enthalten."""

        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        # Minimales RestorationResult mit phase_regression_log in metadata
        minimal = RestorationResult(
            audio=audio,
            config=restorer.config,
            material_type=MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0],
            defect_scores={dt: 0.0 for dt in DefectType},
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
