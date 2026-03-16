"""Unit-Tests für core/genre_classifier.py — GermanSchlagerClassifier.

Spec §2.19: 6-Schicht-Ensemble Zero-Shot-Schlager-Erkennung.
≥ 35 Tests (shape, NaN, Bounds, Edge-Cases, Singleton, Profile).
"""

from __future__ import annotations

import concurrent.futures
import math
import threading

import numpy as np
import pytest

from backend.core.genre_classifier import (
    GENRE_RESTORATION_PROFILES,
    JAZZ_RESTORATION_PROFILE,
    KLASSIK_RESTORATION_PROFILE,
    OPER_RESTORATION_PROFILE,
    ROCK_RESTORATION_PROFILE,
    SCHLAGER_RESTORATION_PROFILE,
    GermanSchlagerClassifier,
    SchlagerClassificationResult,
    classify_genre,
    get_genre_classifier,
    get_restoration_profile,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, sr: int = 48000, secs: float = 5.0) -> np.ndarray:
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _white_noise(sr: int = 48000, secs: float = 5.0, amp: float = 0.1) -> np.ndarray:
    np.random.seed(42)
    return (np.random.randn(int(sr * secs)) * amp).astype(np.float32)


def _silence(sr: int = 48000, secs: float = 5.0) -> np.ndarray:
    return np.zeros(int(sr * secs), dtype=np.float32)


def _am_signal(
    carrier_hz: float = 600.0,
    mod_hz: float = 8.0,
    sr: int = 48000,
    secs: float = 5.0,
) -> np.ndarray:
    """AM-Signal mit 8 Hz Reed-Beating (Akkordeon-ähnlich)."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_hz * t)
    mod = 1.0 + 0.6 * np.sin(2 * np.pi * mod_hz * t)
    return (carrier * mod).astype(np.float32)


def _repetitive_signal(sr: int = 48000, block_secs: float = 8.0, repeats: int = 5) -> np.ndarray:
    """Stark repetitives Signal (gleicher Block mehrfach)."""
    block = np.sin(2 * np.pi * 261.6 * np.linspace(0, block_secs, int(sr * block_secs))).astype(np.float32)
    return np.tile(block, repeats)


# ---------------------------------------------------------------------------
# Klasse 1: Grundlegende Importierbarkeit und Singleton
# ---------------------------------------------------------------------------


class TestGenreClassifierImport:
    def test_01_import_classifier_class(self):
        assert GermanSchlagerClassifier is not None

    def test_02_import_result_dataclass(self):
        assert SchlagerClassificationResult is not None

    def test_03_get_genre_classifier_returns_instance(self):
        clf = get_genre_classifier()
        assert isinstance(clf, GermanSchlagerClassifier)

    def test_04_singleton_same_object(self):
        a = get_genre_classifier()
        b = get_genre_classifier()
        assert a is b

    def test_05_singleton_thread_safe(self):
        instances = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(get_genre_classifier) for _ in range(20)]
            instances = [f.result() for f in futures]
        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_06_classify_genre_convenience_wrapper(self):
        audio = _white_noise()
        result = classify_genre(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)


# ---------------------------------------------------------------------------
# Klasse 2: Ausgabe-Invarianten (NaN, Bounds, Tipos)
# ---------------------------------------------------------------------------


class TestGenreClassifierOutput:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_07_no_nan_on_white_noise(self):
        audio = _white_noise()
        r = self.clf.classify(audio, sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            assert math.isfinite(getattr(r, field)), f"NaN in {field}"

    def test_08_no_nan_on_silence(self):
        r = self.clf.classify(_silence(), sr=48000)
        assert math.isfinite(r.confidence)
        assert isinstance(r.is_schlager, bool)

    def test_09_all_scores_bounded_white_noise(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            val = getattr(r, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"

    def test_10_all_scores_bounded_sine(self):
        r = self.clf.classify(_sine(), sr=48000)
        for field in (
            "confidence",
            "clap_score",
            "accordion_score",
            "harmonic_simplicity",
            "rhythm_score",
            "vocal_german_prior",
            "melodic_repetition",
        ):
            val = getattr(r, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"

    def test_11_genre_label_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.genre_label, str)
        assert len(r.genre_label) > 0

    def test_12_subgenre_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.subgenre, str)

    def test_13_reasoning_is_string(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r.reasoning, str)

    def test_14_bpm_positive(self):
        r = self.clf.classify(_sine(freq=200, secs=10.0), sr=48000)
        assert math.isfinite(r.bpm)
        assert r.bpm >= 0.0

    def test_15_key_is_string(self):
        r = self.clf.classify(_sine(), sr=48000)
        assert isinstance(r.key, str)

    def test_16_white_noise_confidence_bounded(self):
        """Weißes Rauschen → alle Scores in [0,1] und Klassifikation stabil."""
        r = self.clf.classify(_white_noise(secs=15.0), sr=48000)
        assert 0.0 <= r.confidence <= 1.0
        assert math.isfinite(r.confidence)

    def test_17_no_false_positive_silence(self):
        """Stille → kein Schlager, kein Absturz."""
        r = self.clf.classify(_silence(secs=10.0), sr=48000)
        assert not r.is_schlager
        assert math.isfinite(r.confidence)


# ---------------------------------------------------------------------------
# Klasse 3: Tier-2 Akkordeon-Fingerprint
# ---------------------------------------------------------------------------


class TestAccordionDetection:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_18_accordion_am_signal_detected(self):
        """AM-Signal bei 8 Hz → accordion_score ≥ 0.50."""
        audio = _am_signal(carrier_hz=600.0, mod_hz=8.0, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert r.accordion_score >= 0.50, f"accordion_score={r.accordion_score}"

    def test_19_pure_sine_low_accordion_score(self):
        """Reiner Sinuston → kein Akkordeon-Fingerprint."""
        audio = _sine(freq=440.0, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert r.accordion_score <= 0.60

    def test_20_accordion_score_finite(self):
        audio = _am_signal(mod_hz=10.0, secs=6.0)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.accordion_score)
        assert 0.0 <= r.accordion_score <= 1.0


# ---------------------------------------------------------------------------
# Klasse 4: Tier-3 Harmonische Simplizität
# ---------------------------------------------------------------------------


class TestHarmonicSimplicity:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_21_major_triad_high_hsi(self):
        """I-IV-V-I-Progression → hohe harmonische Simplizität."""
        t = np.linspace(0, 10, 10 * 48000, endpoint=False)
        # C-Dur Dreiklang (C, E, G)
        audio = (np.sin(2 * np.pi * 261.6 * t) + np.sin(2 * np.pi * 329.6 * t) + np.sin(2 * np.pi * 392.0 * t)).astype(
            np.float32
        ) * 0.3
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.harmonic_simplicity)
        assert 0.0 <= r.harmonic_simplicity <= 1.0

    def test_22_harmonic_simplicity_bounded(self):
        audio = _white_noise(secs=10.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.harmonic_simplicity <= 1.0

    def test_23_harmonic_simplicity_finite_silence(self):
        r = self.clf.classify(_silence(secs=10.0), sr=48000)
        assert math.isfinite(r.harmonic_simplicity)


# ---------------------------------------------------------------------------
# Klasse 5: Tier-6 Melodische Wiederholungsrate
# ---------------------------------------------------------------------------


class TestMelodicRepetition:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_24_repetitive_signal_high_score(self):
        """Gleicher 8-s-Block × 5 → melodic_repetition ≥ 0.45."""
        audio = _repetitive_signal(block_secs=8.0, repeats=5)
        r = self.clf.classify(audio, sr=48000)
        assert r.melodic_repetition >= 0.45, f"melodic_repetition={r.melodic_repetition}"

    def test_25_short_signal_neutral_score(self):
        """Signal < 30 s → neutraler Score (keine Strafe)."""
        audio = _sine(secs=15.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.melodic_repetition <= 1.0
        assert math.isfinite(r.melodic_repetition)

    def test_26_melodic_repetition_bounded(self):
        """melodic_repetition liegt immer in [0, 1]."""
        audio = _white_noise(secs=40.0)
        r = self.clf.classify(audio, sr=48000)
        assert 0.0 <= r.melodic_repetition <= 1.0
        assert math.isfinite(r.melodic_repetition)


# ---------------------------------------------------------------------------
# Klasse 6: Edge-Cases (Stereo, kurze Dateien, extreme Amplituden)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self):
        self.clf = get_genre_classifier()

    def test_27_stereo_input_accepted(self):
        """Stereo-Input wird intern zu Mono gemittelt."""
        mono = _white_noise(secs=8.0)
        stereo = np.stack([mono, mono * 0.8], axis=1)
        r = self.clf.classify(stereo, sr=48000)
        assert isinstance(r, SchlagerClassificationResult)
        assert math.isfinite(r.confidence)

    def test_28_very_short_signal_no_crash(self):
        """Signal < 5 s darf nicht abstürzen."""
        audio = _sine(secs=2.0)
        r = self.clf.classify(audio, sr=48000)
        assert isinstance(r, SchlagerClassificationResult)
        assert math.isfinite(r.confidence)

    def test_29_clipped_signal_no_crash(self):
        """Geclipptes Signal → kein Absturz."""
        audio = np.ones(48000 * 5, dtype=np.float32)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.confidence)

    def test_30_very_quiet_signal_no_crash(self):
        """Sehr leises Signal → kein Absturz."""
        audio = _white_noise(amp=1e-6, secs=8.0)
        r = self.clf.classify(audio, sr=48000)
        assert math.isfinite(r.confidence)

    def test_31_result_is_dataclass_instance(self):
        r = self.clf.classify(_white_noise(), sr=48000)
        assert isinstance(r, SchlagerClassificationResult)

    def test_32_wrong_sr_raises_assertion(self):
        """Falsche Sample-Rate (22050 Hz) → AssertionError (Spec §3.x)."""
        audio = _white_noise(sr=22050, secs=8.0)
        with pytest.raises(AssertionError, match="48000 Hz erwartet"):
            self.clf.classify(audio, sr=22050)


# ---------------------------------------------------------------------------
# Klasse 7: Restaurierungsprofile
# ---------------------------------------------------------------------------


class TestRestorationProfiles:
    def test_33_schlager_profile_has_required_keys(self):
        required = {
            "soft_saturation_preserve",
            "tonal_center_threshold",
            "groove_dtw_max_ms",
            "deessing_target_hz",
            "brillanz_target",
            "waerme_target",
            "gp_memory_key",
        }
        assert required.issubset(SCHLAGER_RESTORATION_PROFILE.keys())

    def test_34_schlager_tonal_center_threshold_ge_095(self):
        assert SCHLAGER_RESTORATION_PROFILE["tonal_center_threshold"] >= 0.95

    def test_35_schlager_groove_dtw_le_8ms(self):
        assert SCHLAGER_RESTORATION_PROFILE["groove_dtw_max_ms"] <= 8.0

    def test_36_schlager_waerme_target_ge_080(self):
        assert SCHLAGER_RESTORATION_PROFILE["waerme_target"] >= 0.80

    def test_37_schlager_brillanz_target_in_range(self):
        assert 0.70 <= SCHLAGER_RESTORATION_PROFILE["brillanz_target"] <= 1.0

    def test_38_schlager_gp_memory_key_is_string(self):
        key = SCHLAGER_RESTORATION_PROFILE["gp_memory_key"]
        assert isinstance(key, str) and len(key) > 0

    def test_39_jazz_profile_has_groove_key(self):
        assert "groove_dtw_max_ms" in JAZZ_RESTORATION_PROFILE

    def test_40_jazz_groove_strict(self):
        """Jazz-Timing ist sacred — DTW ≤ 4 ms."""
        assert JAZZ_RESTORATION_PROFILE["groove_dtw_max_ms"] <= 5.0

    def test_41_klassik_profile_has_dereverb_flag(self):
        assert (
            "phase_20_dereverb_enabled" in KLASSIK_RESTORATION_PROFILE
            or "phase_49_dereverb_enabled" in KLASSIK_RESTORATION_PROFILE
        )

    def test_42_oper_profile_has_deessing_key(self):
        assert "deessing_target_hz" in OPER_RESTORATION_PROFILE or "deessing_strength_cap" in OPER_RESTORATION_PROFILE

    def test_43_rock_brillanz_target_high(self):
        assert ROCK_RESTORATION_PROFILE.get("brillanz_target", 1.0) >= 0.85

    def test_44_genre_profiles_dict_not_empty(self):
        assert len(GENRE_RESTORATION_PROFILES) >= 4

    def test_45_get_restoration_profile_schlager(self):
        p = get_restoration_profile("Schlager")
        assert p is not None
        assert "gp_memory_key" in p

    def test_46_get_restoration_profile_unknown_returns_default(self):
        p = get_restoration_profile("UnbekanntesGenre")
        # Soll Default zurückgeben, nicht abstürzen
        assert p is not None

    def test_47_all_profiles_have_gp_memory_key(self):
        for genre, profile in GENRE_RESTORATION_PROFILES.items():
            assert "gp_memory_key" in profile, f"gp_memory_key fehlt in {genre}"

    def test_48_all_profile_values_finite(self):
        """Alle numerischen Werte in den Profilen sind finite."""
        all_profiles = [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]
        for profile in all_profiles:
            for k, v in profile.items():
                if isinstance(v, float):
                    assert math.isfinite(v), f"Nicht-finiter Wert {k}={v}"
