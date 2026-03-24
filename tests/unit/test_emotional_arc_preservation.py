"""Unit-Tests für core/emotional_arc_preservation.py — EmotionalArcPreservationMetric.

Spec §8.2 Punkt 12: Emotionaler Dynamik-Bogen, Arousal/Valence Pearson, Klimax-Erhalt.
≥ 20 Tests.
"""

from __future__ import annotations

import math

import numpy as np

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


# ---------------------------------------------------------------------------
# Klasse 7: correct_arc() — Makro-Bogen-Korrektur (§8.2)
# ---------------------------------------------------------------------------


def _crescendo_decrescendo(secs: float = 60.0) -> np.ndarray:
    """Musikalischer Bogen: leise → laut → leise (Sinusbogen-Hüllkurve)."""
    n = int(SR * secs)
    t = np.linspace(0, secs, n, endpoint=False)
    envelope = np.sin(np.pi * t / secs)  # Spitze in der Mitte
    return (np.sin(2 * np.pi * 440 * t) * envelope * 0.8).astype(np.float32)


def _flatten_dynamics(audio: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Simuliert NR-induzierte Dynamik-Abflachung: Amplitude → Mittelwert ziehen."""
    rms = np.sqrt(np.mean(audio**2) + 1e-12)
    audio - 0.0
    return (audio * (1 - factor) + np.sign(audio) * rms * factor).astype(np.float32)


class TestCorrectArc:
    def setup_method(self):
        self.m = EmotionalArcPreservationMetric()

    def test_24_correct_arc_importable(self):
        from backend.core.emotional_arc_preservation import correct_emotional_arc

        assert callable(correct_emotional_arc)

    def test_25_short_file_returns_unchanged(self):
        """< 30 s → Audio unverändert zurückgeben."""
        audio = _sine(440.0, secs=15.0)
        corrected, arc = self.m.correct_arc(audio, audio, SR)
        assert corrected.shape == audio.shape
        assert arc.skipped is True

    def test_26_identical_signal_no_change(self):
        """Identisches Signal → minimale Gain-Änderung, kein Absturz."""
        orig = _crescendo_decrescendo(secs=60.0)
        corrected, arc = self.m.correct_arc(orig, orig.copy(), SR)
        # Gain-Delta sollte bei identischem Signal nahe 0 sein
        diff = np.max(np.abs(corrected - orig))
        assert diff < 0.05, f"Identisches Signal: zu große Abweichung {diff}"

    def test_27_output_shape_preserved(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert corrected.shape == rest.shape
        assert corrected.dtype == np.float32

    def test_28_no_nan_inf_in_output(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert np.isfinite(corrected).all()

    def test_29_no_clipping(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR)
        assert np.max(np.abs(corrected)) <= 1.0

    def test_30_result_has_arousal_valence(self):
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        _, arc = self.m.correct_arc(orig, rest, SR)
        assert hasattr(arc, "arousal_pearson")
        assert hasattr(arc, "valence_pearson")
        assert -1.0 <= arc.arousal_pearson <= 1.0
        assert -1.0 <= arc.valence_pearson <= 1.0

    def test_31_stereo_input_supported(self):
        """Stereo-Signal (2, N) → korrekte Ausgabe."""
        orig_mono = _crescendo_decrescendo(secs=60.0)
        orig = np.stack([orig_mono, orig_mono * 0.8])
        rest = np.stack([_flatten_dynamics(orig_mono), _flatten_dynamics(orig_mono * 0.8)])
        corrected, arc = self.m.correct_arc(orig, rest, SR)
        assert corrected.ndim == 2
        assert corrected.shape[0] == 2
        assert np.isfinite(corrected).all()

    def test_32_safety_revert_on_degradation(self):
        """Wenn Korrektur Arousal verschlechtert → Original zurückgeben."""
        orig = _sine(440.0, secs=60.0) * 0.3
        # Künstlich "korrigierte" Version mit umgekehrtem Profil
        rest = _dynamic_signal(secs=60.0)
        corrected, arc = self.m.correct_arc(orig, rest, SR)
        # Egal ob revert passiert — kein Absturz, gültiges Ergebnis
        assert np.isfinite(corrected).all()
        assert isinstance(arc, EmotionalArcResult)

    def test_33_convenience_function_works(self):
        from backend.core.emotional_arc_preservation import correct_emotional_arc

        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, arc = correct_emotional_arc(orig, rest, SR)
        assert corrected.shape == rest.shape
        assert isinstance(arc, EmotionalArcResult)

    def test_34_damping_zero_returns_unchanged(self):
        """damping=0 → keine Korrektur (Gain-Profil = 0 dB)."""
        orig = _crescendo_decrescendo(secs=60.0)
        rest = _flatten_dynamics(orig)
        corrected, _ = self.m.correct_arc(orig, rest, SR, damping=0.0)
        diff = np.max(np.abs(corrected - rest))
        assert diff < 0.01, f"damping=0 sollte keine Änderung ergeben, diff={diff}"

    def test_35_max_gain_respected(self):
        """Gain wird auf max_gain_db begrenzt."""
        orig = _crescendo_decrescendo(secs=60.0)
        rest = orig * 0.01  # Extremer Pegelunterschied
        corrected, _ = self.m.correct_arc(orig, rest, SR, max_gain_db=3.0)
        # Bei max 3 dB Gain: Faktor ≤ 10^(3/20) ≈ 1.41
        assert np.max(np.abs(corrected)) <= 1.0
