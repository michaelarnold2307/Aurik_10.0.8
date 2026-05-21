"""Regressionstests fuer gemeinsamen De-Esser-Intensitaetsrechner und Phasen-Integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.defect_scanner import MaterialType

SR = 48_000


def _make_sibilant(freq: float = 7000.0, duration: float = 0.4, amp: float = 0.45) -> np.ndarray:
    t = np.linspace(0.0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    env = np.sin(np.linspace(0.0, np.pi, t.size, dtype=np.float32)) ** 2
    return (amp * env * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


@dataclass
class _DummyWord:
    phoneme_type: str
    is_stressed: bool = False


@dataclass
class _DummyTimeline:
    words: list[_DummyWord]
    language: str = "de"


@dataclass
class _DummySegment:
    phoneme_class: str
    is_stressed: bool = False
    phoneme_ipa: str = ""


@dataclass
class _DummySegmentTimeline:
    segments: list[_DummySegment]
    language: str = "de"


def test_helper_intensity_scales_with_sibilance_pressure_and_affricate_drive():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=7000.0)
    mild = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.55,
        defect_scores={"sibilance": 0.15},
        fricative_snr_db=0.5,
        freq_low=5000.0,
        freq_high=11000.0,
    )
    hot = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.55,
        defect_scores={"sibilance": 0.95, "vocal_harshness": 0.80},
        fricative_snr_db=8.0,
        freq_low=5000.0,
        freq_high=11000.0,
    )

    assert hot.intensity > mild.intensity
    assert hot.threshold_db_delta > mild.threshold_db_delta
    assert hot.threshold_ratio_scale < mild.threshold_ratio_scale
    assert hot.affricate_drive >= mild.affricate_drive


def test_helper_boosts_german_phoneme_timeline_for_affricates():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=6500.0)
    plain = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
    )
    german = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        language_hint="de",
        phoneme_timeline=_DummyTimeline(words=[_DummyWord("plosive", True), _DummyWord("fricative", True)]),
    )

    assert german.phoneme_drive > plain.phoneme_drive
    assert german.intensity > plain.intensity
    assert german.threshold_db_delta > plain.threshold_db_delta
    assert german.threshold_ratio_scale < plain.threshold_ratio_scale


def test_helper_boosts_explicit_sibilant_segments_for_s_sound():
    from backend.core.dsp.deesser_intensity import compute_optimal_deesser_intensity

    audio = _make_sibilant(freq=7200.0)
    fricative = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        phoneme_timeline=_DummySegmentTimeline(segments=[_DummySegment("fricative_stressed", True)]),
    )
    s_sound = compute_optimal_deesser_intensity(
        audio,
        SR,
        effective_strength=0.35,
        defect_scores={"sibilance": 0.35},
        fricative_snr_db=3.0,
        freq_low=5000.0,
        freq_high=11000.0,
        phoneme_timeline=_DummySegmentTimeline(segments=[_DummySegment("sibilant", True, "s")]),
    )

    assert s_sound.intensity > fricative.intensity
    assert s_sound.threshold_db_delta > fricative.threshold_db_delta
    assert s_sound.ratio_multiplier > fricative.ratio_multiplier
    assert s_sound.threshold_ratio_scale < fricative.threshold_ratio_scale


def test_phase19_uses_stronger_intensity_for_hot_sibilance():
    from backend.core.phases.phase_19_de_esser import DeEsserPhase

    phase = DeEsserPhase(gender_type="female")
    audio = _make_sibilant(freq=7000.0)

    mild = phase.process(audio, SR, MaterialType.VINYL, strength=0.35, defect_scores_raw={"sibilance": 0.10})
    hot = phase.process(
        audio,
        SR,
        MaterialType.VINYL,
        strength=0.35,
        defect_scores_raw={"sibilance": 0.95, "vocal_harshness": 0.80},
    )

    assert float(hot.metadata["deesser_intensity"]) > float(mild.metadata["deesser_intensity"])
    assert float(hot.metadata["max_reduction_db"]) < float(mild.metadata["max_reduction_db"])
    assert float(hot.metadata["threshold_ratio"]) < float(mild.metadata["threshold_ratio"])


def test_phase43_uses_stronger_intensity_for_hot_sibilance():
    from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

    phase = MLDeEsserPhase()
    audio = _make_sibilant(freq=7000.0)

    mild = phase.process(audio, SR, strength=0.35, defect_scores={"sibilance": 0.10})
    hot = phase.process(audio, SR, strength=0.35, defect_scores={"sibilance": 0.95, "vocal_harshness": 0.80})

    assert float(hot.metadata["deesser_intensity"]) > float(mild.metadata["deesser_intensity"])
    assert float(hot.metadata["control_strength"]) >= float(mild.metadata["control_strength"])
    assert float(hot.metadata["strength_cap"]) <= float(mild.metadata["strength_cap"])
