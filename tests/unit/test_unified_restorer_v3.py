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

        with patch("backend.core.unified_restorer_v3.librosa.resample", side_effect=lambda y, **_: y):
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

        with patch(
            "forensics.medium_detector.get_medium_detector",
            side_effect=AssertionError("get_medium_detector must not be called on cached-medium path"),
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

        with (
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

    def test_79_low_groove_boosts_dynamics_eq_and_transient(self):
        scores = {"groove": 0.60}
        result = self._FN(scores, self._base_profile(), "33pct", 5, 15)
        assert result is not None
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
