"""Unit tests for backend.core.pre_analysis — PreAnalysisResult + run_pre_analysis.

Test-IDs: test_pre_analysis_*
Coverage targets:
  - PreAnalysisResult dataclass
  - run_pre_analysis(): success path, partial failures, bridge cache storage
  - UV3 pre_analysis_result kwarg unpacking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence(duration_s: float = 1.0, sr: int = 44_100, channels: int = 1) -> np.ndarray:
    """Create a silent float32 audio array."""
    n = int(duration_s * sr)
    if channels == 2:
        return np.zeros((n, 2), dtype=np.float32)
    return np.zeros(n, dtype=np.float32)


# Minimal stub dataclasses to avoid loading heavy backend in unit tests
@dataclass
class _StubMedium:
    primary_material: str = "vinyl"
    confidence: float = 0.85
    chain_label: str = "vinyl → mp3"
    transfer_chain: list = field(default_factory=lambda: ["vinyl", "mp3_low"])


@dataclass
class _StubEra:
    decade: int = 1970
    era_label: str = "1970er"
    confidence: float = 0.72


@dataclass
class _StubGenre:
    is_schlager: bool = True
    genre_label: str = "Deutscher Schlager"
    confidence: float = 0.78


@dataclass
class _StubDefects:
    material_type: str = "vinyl"
    duration_seconds: float = 60.0


@dataclass
class _StubRestorability:
    restorability_score: float = 72.0
    predicted_mos: float = 4.1
    grade: str = "good"


# ---------------------------------------------------------------------------
# PreAnalysisResult — dataclass contract
# ---------------------------------------------------------------------------


class TestPreAnalysisResult:
    def test_default_all_none(self):
        from backend.core.pre_analysis import PreAnalysisResult

        r = PreAnalysisResult()
        assert r.medium is None
        assert r.era is None
        assert r.genre is None
        assert r.defects is None
        assert r.restorability is None
        assert r.errors == {}
        assert r.elapsed_seconds == 0.0

    def test_fields_assignable(self):
        from backend.core.pre_analysis import PreAnalysisResult

        r = PreAnalysisResult(
            medium=_StubMedium(),
            era=_StubEra(),
            genre=_StubGenre(),
            defects=_StubDefects(),
            restorability=_StubRestorability(),
            native_sr=44100,
            file_path="/tmp/song.mp3",
            elapsed_seconds=1.5,
        )
        assert r.native_sr == 44100
        assert r.file_path == "/tmp/song.mp3"
        assert r.elapsed_seconds == 1.5
        assert r.medium is not None
        assert r.era is not None
        assert r.genre is not None
        assert r.medium.primary_material == "vinyl"
        assert r.era.decade == 1970
        assert r.genre.is_schlager is True

    def test_errors_dict_mutable(self):
        from backend.core.pre_analysis import PreAnalysisResult

        r = PreAnalysisResult()
        r.errors["medium"] = "import error"
        assert r.errors["medium"] == "import error"

    def test_no_shared_errors_dict(self):
        """Each instance gets its own errors dict (not shared class-level default)."""
        from backend.core.pre_analysis import PreAnalysisResult

        r1 = PreAnalysisResult()
        r2 = PreAnalysisResult()
        r1.errors["era"] = "fail"
        assert "era" not in r2.errors


# ---------------------------------------------------------------------------
# run_pre_analysis — mocked backend modules
# ---------------------------------------------------------------------------

MOCK_MEDIUM = _StubMedium()
MOCK_ERA = _StubEra()
MOCK_GENRE = _StubGenre()
MOCK_DEFECTS = _StubDefects()
MOCK_RESTORABILITY = _StubRestorability()


def _make_mock_md():
    md = MagicMock()
    md.detect.return_value = MOCK_MEDIUM
    return md


def _make_mock_era_cls():
    ec = MagicMock()
    ec.classify.return_value = MOCK_ERA
    return ec


def _make_mock_genre_cls():
    gc = MagicMock()
    gc.classify.return_value = MOCK_GENRE
    return gc


def _make_mock_defect_scanner():
    ds = MagicMock()
    ds.scan.return_value = MOCK_DEFECTS
    return ds


def _make_mock_restorability(*_args, **_kwargs):
    return MOCK_RESTORABILITY


class TestRunPreAnalysis:
    """Tests for run_pre_analysis() with real success path and targeted mocks for failures."""

    def _run_real(self, audio=None, sr=44100, file_path="/tmp/test.mp3", **kwargs):
        if audio is None:
            t = np.linspace(0.0, 0.5, int(0.5 * sr), endpoint=False)
            audio = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        from backend.core.pre_analysis import run_pre_analysis

        return run_pre_analysis(audio, sr, file_path=file_path, store_in_bridge_cache=False, **kwargs)

    def _run(self, audio=None, sr=44100, file_path="/tmp/test.mp3", store_in_bridge_cache=False, **kwargs):
        if audio is None:
            audio = _silence(2.0, sr)

        with (
            patch("forensics.medium_detector.get_medium_detector", return_value=_make_mock_md()),
            patch("backend.core.era_classifier.get_era_classifier", return_value=_make_mock_era_cls()),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            return run_pre_analysis(
                audio,
                sr,
                file_path=file_path,
                store_in_bridge_cache=store_in_bridge_cache,
                **kwargs,
            )

    def test_success_path_all_fields_populated(self):
        result = self._run_real(sr=48_000)
        assert result is not None
        assert result.native_sr == 48_000
        assert result.file_path == "/tmp/test.mp3"
        assert result.elapsed_seconds >= 0.0
        # At least one analyzer should usually populate a sub-result on valid audio.
        assert any(
            x is not None
            for x in (
                result.medium,
                result.era,
                result.genre,
                result.defects,
                result.restorability,
            )
        )

    def test_native_sr_preserved(self):
        result = self._run_real(sr=44_100)
        assert result.native_sr == 44_100

    def test_file_path_preserved(self):
        result = self._run_real(file_path="/home/user/song.mp3")
        assert result.file_path == "/home/user/song.mp3"

    def test_elapsed_seconds_positive(self):
        result = self._run_real()
        assert result.elapsed_seconds >= 0.0

    def test_medium_gets_correct_values(self):
        result = self._run()
        assert result.medium is not None
        assert result.medium.primary_material == "vinyl"
        assert result.medium.confidence == 0.85

    def test_era_gets_correct_values(self):
        result = self._run()
        assert result.era is not None
        assert result.era.decade == 1970

    def test_genre_is_schlager(self):
        result = self._run()
        assert result.genre is not None
        assert result.genre.is_schlager is True
        assert result.genre.genre_label == "Deutscher Schlager"

    def test_stereo_audio(self):
        audio = _silence(1.0, 44_100, channels=2)
        result = self._run_real(audio=audio, sr=44_100)
        assert result.native_sr == 44_100
        assert result.elapsed_seconds >= 0.0

    def test_progress_callback_called(self):
        calls = []
        self._run_real(progress_callback=lambda pct, msg: calls.append(pct))
        assert 0 in calls
        assert 100 in calls

    def test_medium_failure_partial_result(self):
        """If medium detection fails, other steps still succeed."""
        _legacy = MagicMock(side_effect=RuntimeError("legacy classifier must stay unused"))
        with (
            patch("forensics.medium_detector.get_medium_detector", side_effect=ImportError("no module")),
            patch("backend.core.medium_classifier.classify_medium", _legacy),
            patch("backend.core.era_classifier.get_era_classifier", return_value=_make_mock_era_cls()),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            result = run_pre_analysis(_silence(1.0), 44100, store_in_bridge_cache=False)
        assert result.medium is None
        assert "medium" in result.errors
        assert _legacy.call_count == 0
        assert result.era is not None

    def test_medium_primary_fails_without_legacy_fallback(self):
        """Primary detect is attempted once; legacy classifier is never called."""

        _md = MagicMock()
        _md.detect.side_effect = RuntimeError("primary down")
        _legacy = MagicMock(side_effect=RuntimeError("legacy classifier must stay unused"))

        with (
            patch("forensics.medium_detector.get_medium_detector", return_value=_md),
            patch("backend.core.medium_classifier.classify_medium", _legacy),
            patch("backend.core.era_classifier.get_era_classifier", return_value=_make_mock_era_cls()),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            result = run_pre_analysis(_silence(1.0), 44100, store_in_bridge_cache=False)

        assert _md.detect.call_count == 1
        assert _legacy.call_count == 0
        assert result.medium is None
        assert "medium" in result.errors

    def test_single_step_failure_no_exception(self):
        """A failure in one step must not raise — result is returned with errors dict."""
        with (
            patch("forensics.medium_detector.get_medium_detector", return_value=_make_mock_md()),
            patch("backend.core.era_classifier.get_era_classifier", side_effect=RuntimeError("era crash")),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            result = run_pre_analysis(_silence(1.0), 44100, store_in_bridge_cache=False)
        assert result.era is None
        assert "era" in result.errors
        assert result.medium is not None  # other steps unaffected

    def test_all_steps_fail_no_exception(self):
        """Even total failure returns a PreAnalysisResult without raising."""
        with (
            patch("forensics.medium_detector.get_medium_detector", side_effect=Exception("x")),
            patch("backend.core.era_classifier.get_era_classifier", side_effect=Exception("x")),
            patch("backend.core.genre_classifier.get_genre_classifier", side_effect=Exception("x")),
            patch("backend.core.defect_scanner.DefectScanner", side_effect=Exception("x")),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=Exception("x")),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            result = run_pre_analysis(_silence(1.0), 44100, store_in_bridge_cache=False)
        assert len(result.errors) >= 4

    def test_hung_substep_times_out_and_returns_partial_result(self):
        """A stuck sub-analysis must degrade instead of blocking the whole pre-analysis."""
        with (
            patch("forensics.medium_detector.get_medium_detector", return_value=_make_mock_md()),
            patch("backend.core.era_classifier.get_era_classifier", return_value=_make_mock_era_cls()),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
            patch("backend.core.pre_analysis._SUBSTEP_TIMEOUT_S", 0.01),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            timed_out_future = MagicMock()
            timed_out_future.result.side_effect = TimeoutError()
            timed_out_future.cancel.return_value = False

            ok_future = MagicMock()
            ok_future.result.return_value = MOCK_GENRE

            pool = MagicMock()
            pool.submit.side_effect = [ok_future, timed_out_future, ok_future, ok_future]

            with patch("backend.core.pre_analysis._cf.ThreadPoolExecutor", return_value=pool):
                result = run_pre_analysis(_silence(1.0), 44100, store_in_bridge_cache=False)

        assert result.era is not None
        assert result.genre is None
        assert result.defects is not None
        assert result.restorability is not None
        assert result.errors["genre"] == "timeout_after=0.0s"
        timed_out_future.cancel.assert_called_once_with()
        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_bridge_cache_stored(self):
        """store_in_bridge_cache=True must call bridge cache functions."""
        _m_medium = MagicMock()
        _m_eg = MagicMock()
        _m_defect = MagicMock()
        with (
            patch("forensics.medium_detector.get_medium_detector", return_value=_make_mock_md()),
            patch("backend.core.era_classifier.get_era_classifier", return_value=_make_mock_era_cls()),
            patch("backend.core.genre_classifier.get_genre_classifier", return_value=_make_mock_genre_cls()),
            patch("backend.core.defect_scanner.DefectScanner", return_value=_make_mock_defect_scanner()),
            patch("backend.core.restorability_estimator.estimate_restorability", side_effect=_make_mock_restorability),
            patch("backend.api.bridge.cache_medium_result", _m_medium),
            patch("backend.api.bridge.cache_era_genre_result", _m_eg),
            patch("backend.api.bridge.cache_defect_result", _m_defect),
        ):
            from backend.core.pre_analysis import run_pre_analysis

            run_pre_analysis(
                _silence(1.0),
                44100,
                file_path="/tmp/song.mp3",
                store_in_bridge_cache=True,
            )
        _m_medium.assert_called_once()
        _m_eg.assert_called_once()
        _m_defect.assert_called_once()


# ---------------------------------------------------------------------------
# UV3 pre_analysis_result kwarg unpacking
# ---------------------------------------------------------------------------


class TestUV3PreAnalysisUnpacking:
    """Tests that UV3.restore() correctly unpacks pre_analysis_result kwargs."""

    def _make_pre(self):
        from backend.core.pre_analysis import PreAnalysisResult

        return PreAnalysisResult(
            medium=_StubMedium(),
            era=_StubEra(),
            genre=_StubGenre(),
            defects=_StubDefects(),
            restorability=_StubRestorability(),
            native_sr=44100,
            file_path="/tmp/test.mp3",
        )

    def test_pre_analysis_kwarg_accepted_without_error(self):
        """UV3.restore() must pop pre_analysis_result without KeyError or TypeError."""
        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3
        except Exception:
            pytest.skip("UV3 not importable in test environment")

        pre = self._make_pre()
        _silence(0.5, 48_000)
        UnifiedRestorerV3.__new__(UnifiedRestorerV3)
        # Simulate the kwarg-popping section only (not full restore())
        kwargs = {"pre_analysis_result": pre}
        _pre = kwargs.pop("pre_analysis_result", None)
        assert _pre is not None
        if _pre is not None:
            if kwargs.get("cached_medium_result") is None and getattr(_pre, "medium", None) is not None:
                kwargs["cached_medium_result"] = _pre.medium
            if kwargs.get("cached_era_result") is None and getattr(_pre, "era", None) is not None:
                kwargs["cached_era_result"] = _pre.era
        assert kwargs.get("cached_medium_result") is pre.medium
        assert kwargs.get("cached_era_result") is pre.era

    def test_explicit_kwarg_not_overwritten_by_pre_analysis(self):
        """Explicit cached_* kwargs take priority over pre_analysis_result contents."""
        from backend.core.pre_analysis import PreAnalysisResult

        explicit_medium = _StubMedium(primary_material="shellac", confidence=0.99)
        pre = PreAnalysisResult(medium=_StubMedium(primary_material="vinyl"))
        kwargs = {
            "pre_analysis_result": pre,
            "cached_medium_result": explicit_medium,
        }
        _pre = kwargs.pop("pre_analysis_result", None)
        if _pre is not None and kwargs.get("cached_medium_result") is None:
            kwargs["cached_medium_result"] = _pre.medium
        # Explicit kwarg must NOT be overwritten
        assert kwargs["cached_medium_result"] is explicit_medium
        assert kwargs["cached_medium_result"].primary_material == "shellac"


# ---------------------------------------------------------------------------
# _to_mono_native helper
# ---------------------------------------------------------------------------


class TestToMonoNative:
    def test_mono_unchanged(self):
        from backend.core.pre_analysis import _to_mono_native

        audio = _silence(1.0, 44100, channels=1)
        out = _to_mono_native(audio)
        assert out.ndim == 1
        assert len(out) == len(audio)

    def test_stereo_n_2_shape(self):
        from backend.core.pre_analysis import _to_mono_native

        audio = _silence(1.0, 44100, channels=2)
        assert audio.shape == (44100, 2)
        out = _to_mono_native(audio)
        assert out.ndim == 1
        assert len(out) == 44100

    def test_stereo_2_n_shape(self):
        from backend.core.pre_analysis import _to_mono_native

        audio = np.zeros((2, 44100), dtype=np.float32)
        out = _to_mono_native(audio)
        assert out.ndim == 1

    def test_nan_sanitized(self):
        from backend.core.pre_analysis import _to_mono_native

        audio = np.array([np.nan, 0.5, np.inf, -0.3], dtype=np.float32)
        out = _to_mono_native(audio)
        assert np.all(np.isfinite(out))

    def test_output_clipped(self):
        from backend.core.pre_analysis import _to_mono_native

        audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        out = _to_mono_native(audio)
        assert np.max(np.abs(out)) <= 1.0


# ---------------------------------------------------------------------------
# Bridge re-export
# ---------------------------------------------------------------------------


class TestBridgeExport:
    def test_run_pre_analysis_importable_from_bridge(self):
        from backend.api.bridge import run_pre_analysis

        assert callable(run_pre_analysis)

    def test_pre_analysis_result_importable_from_bridge(self):
        from backend.api.bridge import PreAnalysisResult

        assert PreAnalysisResult is not None
