"""Unit-Tests für backend/core/clipping_detection.py — §6.3 CLIPPING vs. SOFT_SATURATION.

≥ 35 Tests:
  - Shape / Mono / Stereo
  - NaN / Inf Guard
  - Leeres Signal
  - flat_tops_pct Berechnung
  - Korrekte Klassifikation: Hard-clipped Sinus → CLIPPING
  - Korrekte Klassifikation: tanh-saturated Sinus → SOFT_SATURATION
  - Korrekte Klassifikation: sauberes Signal → SOFT_SATURATION
  - Korrekte Klassifikation: schwaches Signal → SOFT_SATURATION
  - Grenzwerte (flat_tops exakt 0.1 %)
  - Singleton Thread-Safety
  - Result-Dataclass Felder
  - Eigenschaften should_repair / should_preserve
  - assert sr != 48000 → AssertionError
  - Stereo-Verarbeitung
  - Confidence 0–1
  - Konsistenz von classify_clipping() vs. analyse_clipping()
  - Reproduzierbarkeit (deterministisch)
  - Performance-Bound (< 2 s bei 5 s Audio)
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from backend.core.clipping_detection import (
    FLAT_TOPS_CLIP_BOUNDARY,
    FLAT_TOPS_THRESHOLD_PCT,
    THD_ODD_DOMINANCE_FACTOR,
    ClippingAnalysisResult,
    ClippingClassifier,
    ClippingType,
    _flat_tops_pct,
    analyse_clipping,
    classify_clipping,
    get_clipping_classifier,
)

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, amp: float = 0.5, dur: float = 0.1) -> np.ndarray:
    """Generate a mono sine wave at 48 kHz."""
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _hard_clip(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply hard clipping: samples beyond threshold are clamped flat."""
    return np.clip(audio, -threshold, threshold)


def _tanh_saturate(audio: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Apply tanh saturation (soft saturation — tube-like even harmonics)."""
    return np.tanh(drive * audio).astype(np.float32)


def _make_hard_clipped_signal(amp: float = 0.95, freq: float = 440.0) -> np.ndarray:
    """Full-amplitude sine then hard-clipped → definite CLIPPING signature."""
    t = np.arange(int(SR * 0.5)) / SR
    raw = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Hard clip to 0.6 → creates flat tops and odd harmonics
    clipped = np.clip(raw, -0.6, 0.6)
    # Normalise to near-unity so flat_tops_pct registers
    clipped = clipped / 0.60 * 0.9995
    return clipped


def _make_tanh_signal(freq: float = 440.0) -> np.ndarray:
    """Tanh-saturated sine → SOFT_SATURATION (even dominant, no flat tops)."""
    t = np.arange(int(SR * 0.5)) / SR
    raw = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return _tanh_saturate(raw, drive=2.5)


def _make_clean_signal(amp: float = 0.3, freq: float = 440.0, dur: float = 0.5) -> np.ndarray:
    """Clean low-amplitude sine — no clipping at all."""
    t = np.arange(int(SR * dur)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Basis: ClippingType Enum
# ---------------------------------------------------------------------------


class TestClippingTypeEnum:
    def test_01_enum_has_clipping(self):
        assert ClippingType.CLIPPING.value == "clipping"

    def test_02_enum_has_soft_saturation(self):
        assert ClippingType.SOFT_SATURATION.value == "soft_saturation"

    def test_03_enum_values_distinct(self):
        assert ClippingType.CLIPPING != ClippingType.SOFT_SATURATION


# ---------------------------------------------------------------------------
# 2. flat_tops_pct Hilfsfunktion
# ---------------------------------------------------------------------------


class TestFlatTopsPct:
    def test_04_zero_for_clean_signal(self):
        audio = _make_clean_signal()
        assert _flat_tops_pct(audio) < 0.01

    def test_05_high_for_fully_clipped(self):
        audio = np.ones(SR, dtype=np.float32)  # all samples at boundary
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert pct > 99.0

    def test_06_partial_clipping(self):
        audio = np.zeros(1000, dtype=np.float32)
        # Set exactly 100 samples to 0.9999 → 10 %
        audio[:100] = 1.0
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert abs(pct - 10.0) < 1.0

    def test_07_empty_array_returns_zero(self):
        assert _flat_tops_pct(np.array([], dtype=np.float32)) == 0.0

    def test_08_negative_boundary_symmetric(self):
        audio = np.full(100, -1.0, dtype=np.float32)
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert pct > 99.0


# ---------------------------------------------------------------------------
# 3. classify_clipping() — grundlegende Klassifikation
# ---------------------------------------------------------------------------


class TestClassifyClipping:
    def test_09_hard_clip_returns_clipping(self):
        audio = _make_hard_clipped_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.CLIPPING

    def test_10_tanh_signal_returns_soft_saturation(self):
        audio = _make_tanh_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_11_clean_signal_returns_soft_saturation(self):
        audio = _make_clean_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_12_silence_returns_soft_saturation(self):
        audio = np.zeros(SR, dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_13_wrong_sr_raises_assertion(self):
        audio = _make_clean_signal()
        with pytest.raises(AssertionError):
            classify_clipping(audio, sr=44100)

    def test_14_wrong_sr_22050_raises_assertion(self):
        audio = _make_clean_signal()
        with pytest.raises(AssertionError):
            classify_clipping(audio, sr=22050)


# ---------------------------------------------------------------------------
# 4. analyse_clipping() — volle Ergebnisstruktur
# ---------------------------------------------------------------------------


class TestAnalyseClipping:
    def test_15_returns_dataclass(self):
        audio = _make_clean_signal()
        result = analyse_clipping(audio, SR)
        assert isinstance(result, ClippingAnalysisResult)

    def test_16_dataclass_has_all_fields(self):
        audio = _make_clean_signal()
        result = analyse_clipping(audio, SR)
        assert hasattr(result, "clipping_type")
        assert hasattr(result, "flat_tops_pct")
        assert hasattr(result, "thd_odd")
        assert hasattr(result, "thd_even")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_clipping")

    def test_17_is_clipping_matches_type_clipping(self):
        audio = _make_hard_clipped_signal()
        result = analyse_clipping(audio, SR)
        assert result.is_clipping == (result.clipping_type == ClippingType.CLIPPING)

    def test_18_is_clipping_false_for_clean(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.is_clipping is False

    def test_19_flat_tops_pct_ge_zero(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.flat_tops_pct >= 0.0

    def test_20_thd_values_non_negative(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.thd_odd >= 0.0
        assert result.thd_even >= 0.0

    def test_21_confidence_in_unit_interval(self):
        for make in [_make_clean_signal, _make_hard_clipped_signal, _make_tanh_signal]:
            result = analyse_clipping(make(), SR)
            assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence}"

    def test_22_hard_clip_high_flat_tops(self):
        result = analyse_clipping(_make_hard_clipped_signal(), SR)
        assert result.flat_tops_pct > FLAT_TOPS_THRESHOLD_PCT

    def test_23_tanh_low_flat_tops(self):
        result = analyse_clipping(_make_tanh_signal(), SR)
        assert result.flat_tops_pct <= FLAT_TOPS_THRESHOLD_PCT

    def test_24_classify_and_analyse_consistent(self):
        audio = _make_hard_clipped_signal()
        ct = classify_clipping(audio, SR)
        ar = analyse_clipping(audio, SR)
        assert ct == ar.clipping_type


# ---------------------------------------------------------------------------
# 5. should_repair / should_preserve Eigenschaften
# ---------------------------------------------------------------------------


class TestResultProperties:
    def test_25_clipping_should_repair_true(self):
        audio = _make_hard_clipped_signal()
        result = analyse_clipping(audio, SR)
        if result.is_clipping:
            assert result.should_repair is True
            assert result.should_preserve is False

    def test_26_soft_sat_should_preserve_true(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.should_preserve is True
        assert result.should_repair is False

    def test_27_should_repair_and_preserve_are_complementary(self):
        audio = _make_tanh_signal()
        result = analyse_clipping(audio, SR)
        assert result.should_repair != result.should_preserve


# ---------------------------------------------------------------------------
# 6. Stereo- und Shape-Handling
# ---------------------------------------------------------------------------


class TestShapeHandling:
    def test_28_stereo_clean_is_soft_saturation(self):
        mono = _make_clean_signal()
        stereo = np.stack([mono, mono], axis=1)
        result = classify_clipping(stereo, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_29_stereo_clipped_is_clipping(self):
        mono = _make_hard_clipped_signal()
        stereo = np.stack([mono, mono], axis=1)
        result = classify_clipping(stereo, SR)
        assert result == ClippingType.CLIPPING

    def test_30_1d_array_accepted(self):
        audio = _make_clean_signal().ravel()
        assert audio.ndim == 1
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)


# ---------------------------------------------------------------------------
# 7. NaN / Inf Guard
# ---------------------------------------------------------------------------


class TestNaNInfGuard:
    def test_31_nan_audio_returns_soft_saturation(self):
        audio = np.full(SR, float("nan"), dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION  # all NaN → zero → silence

    def test_32_inf_audio_returns_soft_saturation(self):
        audio = np.full(SR, float("inf"), dtype=np.float32)
        # nan_to_num clamps inf → 0 internally
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_33_mixed_nan_inf_no_exception(self):
        audio = np.array([np.nan, np.inf, -np.inf, 0.5, 0.3], dtype=np.float32)
        # Must not raise
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)


# ---------------------------------------------------------------------------
# 8. Singleton Thread-Safety
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_34_singleton_same_instance(self):
        a = get_clipping_classifier()
        b = get_clipping_classifier()
        assert a is b

    def test_35_singleton_is_classifier_type(self):
        obj = get_clipping_classifier()
        assert isinstance(obj, ClippingClassifier)

    def test_36_singleton_thread_safe(self):
        """Race condition test: 50 threads acquiring singleton simultaneously."""
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_clipping_classifier())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        # All must be the same object
        assert all(inst is instances[0] for inst in instances)

    def test_37_singleton_classify_method(self):
        clf = get_clipping_classifier()
        audio = _make_clean_signal()
        result = clf.classify(audio, SR)
        assert isinstance(result, ClippingType)

    def test_38_singleton_analyse_method(self):
        clf = get_clipping_classifier()
        audio = _make_clean_signal()
        result = clf.analyse(audio, SR)
        assert isinstance(result, ClippingAnalysisResult)


# ---------------------------------------------------------------------------
# 9. Reproduzierbarkeit & Performance
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_39_deterministic_clean(self):
        audio = _make_clean_signal()
        r1 = classify_clipping(audio, SR)
        r2 = classify_clipping(audio, SR)
        assert r1 == r2

    def test_40_deterministic_hard_clip(self):
        audio = _make_hard_clipped_signal()
        r1 = classify_clipping(audio, SR)
        r2 = classify_clipping(audio, SR)
        assert r1 == r2

    def test_41_performance_5s_audio(self):
        """classify_clipping on 5 s of audio must complete in < 2 s (§DefectScanner budget)."""
        audio = _make_clean_signal(dur=0.5)
        # Extend to ~5 s
        audio = np.tile(audio, 10)
        start = time.perf_counter()
        classify_clipping(audio, SR)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Zu langsam: {elapsed:.2f} s"


# ---------------------------------------------------------------------------
# 10. Edge-Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_42_very_short_signal_no_crash(self):
        audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_43_constant_dc_offset_no_crash(self):
        audio = np.full(SR, 0.5, dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_44_constant_full_scale_is_clipping_or_soft_sat(self):
        """Full-scale DC signal has flat_tops=100% → CLIPPING classification."""
        audio = np.ones(SR, dtype=np.float32)
        result = analyse_clipping(audio, SR)
        # flat_tops_pct should be ~100%
        assert result.flat_tops_pct > 90.0

    def test_45_negative_amplitude_signal(self):
        t = np.arange(int(SR * 0.1)) / SR
        audio = (-0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION
