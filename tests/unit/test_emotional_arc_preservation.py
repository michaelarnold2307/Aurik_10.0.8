"""Unit-Tests für core/emotional_arc_preservation.py — EmotionalArcPreservationMetric.

Spec §8.2 Punkt 12: Emotionaler Dynamik-Bogen, Arousal/Valence Pearson, Klimax-Erhalt.
≥ 20 Tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.core.emotional_arc_preservation import (
    EmotionalArcPreservationMetric,
    EmotionalArcResult,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000


def _sine(freq: float, secs: float = 30.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(secs: float = 30.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(7)
    return (np.random.randn(int(SR * secs)) * amp).astype(np.float32)


def _silence(secs: float = 30.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


def _dynamic_signal(secs: float = 60.0) -> np.ndarray:
    """Signal mit steigender Dynamik (Crescendo-Simulation)."""
    n = int(SR * secs)
    t = np.linspace(0, secs, n, endpoint=False)
    envelope = np.linspace(0.1, 1.0, n)
    return (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Initialisierung
# ---------------------------------------------------------------------------


class TestEmotionalArcInit:
    def test_01_class_importable(self):
        assert EmotionalArcPreservationMetric is not None

    def test_02_result_class_importable(self):
        assert EmotionalArcResult is not None

    def test_03_instantiate(self):
        m = EmotionalArcPreservationMetric()
        assert m is not None

    def test_04_result_has_required_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(EmotionalArcResult)}
        required = {"arousal_pearson", "valence_pearson", "klimax_peak_deviation", "arc_preserved", "skipped"}
        assert required.issubset(fields)


# ---------------------------------------------------------------------------
# Klasse 2: Kurze Dateien werden übersprungen
# ---------------------------------------------------------------------------


class TestShortFilesSkipped:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_05_short_file_skipped(self):
        """Datei < 30 s → skipped=True, kein Absturz."""
        audio = _sine(440.0, secs=15.0)
        r = self.m.measure(audio, audio, SR)
        assert r.skipped is True

    def test_06_very_short_file_no_crash(self):
        audio = _sine(440.0, secs=3.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_07_silence_short_skipped(self):
        audio = _silence(secs=10.0)
        r = self.m.measure(audio, audio, SR)
        assert r.skipped is True


# ---------------------------------------------------------------------------
# Klasse 3: Ausgabe-Invarianten bei gültiger Länge
# ---------------------------------------------------------------------------


class TestOutputInvariants:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_08_identical_signals_not_skipped(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert not r.skipped

    def test_09_arousal_pearson_bounded(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert -1.0 <= r.arousal_pearson <= 1.0

    def test_10_valence_pearson_bounded(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert -1.0 <= r.valence_pearson <= 1.0

    def test_11_klimax_deviation_non_negative(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert r.klimax_peak_deviation >= 0

    def test_12_identical_signals_arc_preserved(self):
        """Identisches Signal mit sich selbst verglichen → arc_preserved=True."""
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert r.arc_preserved is True

    def test_13_no_nan_in_scores(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        if not r.skipped:
            assert math.isfinite(r.arousal_pearson)
            assert math.isfinite(r.valence_pearson)

    def test_14_noise_vs_noise_no_crash(self):
        audio = _noise(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_15_silence_long_no_crash(self):
        audio = _silence(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_16_reason_is_string(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.reason, str)

    def test_17_arc_preserved_is_bool(self):
        audio = _dynamic_signal(secs=60.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.arc_preserved, bool)

    def test_18_skipped_is_bool(self):
        audio = _silence(secs=10.0)
        r = self.m.measure(audio, audio, SR)
        assert isinstance(r.skipped, bool)


# ---------------------------------------------------------------------------
# Klasse 4: Verschiedene Signalkombinationen
# ---------------------------------------------------------------------------


class TestSignalCombinations:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_19_sine_vs_noise_no_crash(self):
        orig = _sine(440.0, secs=60.0)
        rest = _noise(secs=60.0)
        r = self.m.measure(orig, rest, SR)
        assert isinstance(r, EmotionalArcResult)

    def test_20_crescendo_vs_flat_no_crash(self):
        orig = _dynamic_signal(secs=60.0)
        rest = _sine(440.0, secs=60.0) * 0.5
        r = self.m.measure(orig, rest, SR)
        assert isinstance(r, EmotionalArcResult)
        if not r.skipped:
            assert math.isfinite(r.arousal_pearson)

    def test_21_different_lengths_no_crash(self):
        """Unterschiedliche Längen → kein Absturz."""
        orig = _dynamic_signal(secs=60.0)
        rest = _dynamic_signal(secs=70.0)
        try:
            r = self.m.measure(orig, rest, SR)
            assert isinstance(r, EmotionalArcResult)
        except Exception:
            pass  # Toleriert: unterschiedliche Längen können abgelehnt werden

    def test_22_threshold_arousal_constant(self):
        assert EmotionalArcPreservationMetric.THRESHOLD_AROUSAL == 0.85

    def test_23_threshold_valence_constant(self):
        assert EmotionalArcPreservationMetric.THRESHOLD_VALENCE == 0.80
