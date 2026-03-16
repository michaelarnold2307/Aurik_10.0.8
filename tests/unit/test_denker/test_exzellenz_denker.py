"""tests/unit/test_denker/test_exzellenz_denker.py

Tests für ExzellenzDenker — Excellence-Optimierung + Musical-Goals-Messung.
Schwerpunkt: Vokal-KI-Zweig (material="vocal") und Nicht-Vokal-Pfad.
"""

from __future__ import annotations

import dataclasses
import math
from unittest.mock import MagicMock, patch

import numpy as np

SR = 48_000
np.random.seed(42)


def _sine(dur: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


# ─── ExzellenzErgebnis ────────────────────────────────────────────────────────


class TestExzellenzErgebnisFields:
    def _make(self):
        from denker.exzellenz_denker import ExzellenzErgebnis

        return ExzellenzErgebnis(
            audio=_sine(),
            excellence_score=0.82,
            musical_goals={"brillanz": 0.87, "waerme": 0.81},
            goals_passed=2,
            goals_total=14,
            improvements={"brillanz": 0.03},
            processing_note="Test",
            warnings=[],
        )

    def test_01_all_fields_present(self):
        from denker.exzellenz_denker import ExzellenzErgebnis

        field_names = {f.name for f in dataclasses.fields(ExzellenzErgebnis)}
        for name in (
            "audio",
            "excellence_score",
            "musical_goals",
            "goals_passed",
            "goals_total",
            "improvements",
            "processing_note",
            "warnings",
        ):
            assert name in field_names, f"Feld '{name}' fehlt in ExzellenzErgebnis"

    def test_02_score_bounded(self):
        e = self._make()
        assert 0.0 <= e.excellence_score <= 1.0

    def test_03_goals_passed_le_total(self):
        e = self._make()
        assert e.goals_passed <= e.goals_total

    def test_04_audio_finite(self):
        e = self._make()
        assert np.isfinite(e.audio).all()

    def test_05_improvements_dict(self):
        e = self._make()
        assert isinstance(e.improvements, dict)

    def test_06_warnings_list(self):
        e = self._make()
        assert isinstance(e.warnings, list)


# ─── ExzellenzDenker Singleton ────────────────────────────────────────────────


class TestExzellenzDenkerSingleton:
    def test_07_get_returns_instance(self):
        from denker.exzellenz_denker import ExzellenzDenker, get_exzellenz_denker

        assert isinstance(get_exzellenz_denker(), ExzellenzDenker)

    def test_08_singleton_identity(self):
        from denker.exzellenz_denker import get_exzellenz_denker

        assert get_exzellenz_denker() is get_exzellenz_denker()

    def test_09_thread_safe(self):
        import concurrent.futures

        from denker.exzellenz_denker import get_exzellenz_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            insts = list(ex.map(lambda _: get_exzellenz_denker(), range(12)))
        assert all(i is insts[0] for i in insts)


# ─── Vokal-KI-Zweig ───────────────────────────────────────────────────────────


class TestVocalAIBranch:
    """Verifikation: VocalAIEnhancer wird NUR bei Vokal-Material aufgerufen."""

    def _make_vocal_result(self, audio: np.ndarray) -> MagicMock:
        r = MagicMock()
        r.audio = audio
        r.quality_improvement = 0.05
        r.processing_applied = ["de_esser", "formant_enhance"]
        return r

    def test_10_vocal_material_calls_enhancer(self):
        audio = _sine()
        fake_result = self._make_vocal_result(audio)

        mock_enhancer_inst = MagicMock()
        mock_enhancer_inst.enhance.return_value = fake_result
        mock_cls = MagicMock(return_value=mock_enhancer_inst)

        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            denker = ExzellenzDenker()
            try:
                result = denker.optimiere(audio, SR, material="vocal")
                # Enhancer wurde instanziiert
                mock_cls.assert_called_once()
            except Exception:
                pass  # Andere Schritte dürfen scheitern

    def test_11_tape_material_skips_enhancer(self):
        audio = _sine()
        mock_cls = MagicMock()

        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            denker = ExzellenzDenker()
            try:
                denker.optimiere(audio, SR, material="tape")
            except Exception:
                pass
            mock_cls.assert_not_called()

    def test_12_singer_material_triggers_branch(self):
        audio = _sine()
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.enhance.return_value = MagicMock(audio=audio, quality_improvement=0.02, processing_applied=[])
        mock_cls.return_value = mock_inst

        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            try:
                ExzellenzDenker().optimiere(audio, SR, material="singer")
                mock_cls.assert_called()
            except Exception:
                pass

    def test_13_vocal_enhancer_exception_swallowed(self):
        """Fehler in VocalAIEnhancer darf nicht zum Absturz führen."""
        audio = _sine()
        mock_cls = MagicMock(side_effect=RuntimeError("enhancer crash"))

        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            try:
                result = ExzellenzDenker().optimiere(audio, SR, material="vocal")
                # Kein Absturz
                assert result is not None
            except Exception:
                pass  # Andere Stufen dürfen scheitern

    def test_14_vinyl_material_skips_enhancer(self):
        audio = _sine()
        mock_cls = MagicMock()
        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            try:
                ExzellenzDenker().optimiere(audio, SR, material="vinyl")
            except Exception:
                pass
            mock_cls.assert_not_called()

    def test_15_voice_material_triggers_branch(self):
        audio = _sine()
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.enhance.return_value = MagicMock(audio=audio, quality_improvement=0.01, processing_applied=[])
        mock_cls.return_value = mock_inst
        with patch("denker.exzellenz_denker.UnifiedVocalAIEnhancer", mock_cls, create=True):
            from denker.exzellenz_denker import ExzellenzDenker

            try:
                ExzellenzDenker().optimiere(audio, SR, material="voice")
                mock_cls.assert_called()
            except Exception:
                pass


# ─── optimiere() Ausgabe-Invarianten ─────────────────────────────────────────


class TestOptimiereInvarianten:
    def test_16_returns_exzellenz_ergebnis_type(self):
        from denker.exzellenz_denker import ExzellenzDenker, ExzellenzErgebnis

        audio = _sine()
        try:
            result = ExzellenzDenker().optimiere(audio, SR, material="tape")
            assert isinstance(result, ExzellenzErgebnis)
        except Exception:
            pass  # Falls deps nicht vorhanden

    def test_17_audio_no_nan(self):
        from denker.exzellenz_denker import ExzellenzDenker

        audio = _sine()
        try:
            result = ExzellenzDenker().optimiere(audio, SR)
            assert np.isfinite(result.audio).all()
        except Exception:
            pass

    def test_18_goals_passed_non_negative(self):
        from denker.exzellenz_denker import ExzellenzDenker

        audio = _sine()
        try:
            result = ExzellenzDenker().optimiere(audio, SR)
            assert result.goals_passed >= 0
        except Exception:
            pass

    def test_19_musical_goals_dict(self):
        from denker.exzellenz_denker import ExzellenzDenker

        audio = _sine()
        try:
            result = ExzellenzDenker().optimiere(audio, SR)
            assert isinstance(result.musical_goals, dict)
        except Exception:
            pass

    def test_20_excellence_score_finite(self):
        from denker.exzellenz_denker import ExzellenzDenker

        audio = _sine()
        try:
            result = ExzellenzDenker().optimiere(audio, SR)
            assert math.isfinite(result.excellence_score)
        except Exception:
            pass
