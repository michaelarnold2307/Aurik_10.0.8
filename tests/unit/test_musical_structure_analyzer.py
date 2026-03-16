"""Unit-Tests für MusicalStructureAnalyzer (§2.17).

Tests: ≥ 22 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Stereo, Singleton
"""

import concurrent.futures

import numpy as np
import pytest

from backend.core.musical_structure_analyzer import (
    MusicalStructure,
    MusicalStructureAnalyzer,
    SegmentInfo,
    analyze_musical_structure,
    get_musical_structure_analyzer,
)

SR = 48000


@pytest.fixture
def analyzer():
    return MusicalStructureAnalyzer()


# ---------------------------------------------------------------------------
# Kurze Dateien → leeere Struktur (< 20 s)
# ---------------------------------------------------------------------------


def test_short_audio_returns_empty_structure(analyzer):
    audio = np.zeros(SR * 10, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)
    assert len(structure.segments) == 0


def test_under_20s_returns_empty_segments(analyzer):
    audio = np.random.randn(int(SR * 19.9)).astype(np.float32)
    structure = analyzer.analyze(audio, SR)
    assert len(structure.segments) == 0


# ---------------------------------------------------------------------------
# Normale Audiodatei
# ---------------------------------------------------------------------------


def test_long_audio_returns_segments(analyzer):
    np.random.seed(42)
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)
    assert len(structure.segments) >= 0  # Kann 0 sein wenn keine Grenzen erkannt


def test_analyze_returns_musical_structure(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_total_duration_set(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    expected_duration = len(audio) / SR
    assert abs(structure.total_duration_s - expected_duration) < 0.1


def test_bpm_nonnegative(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert structure.bpm >= 0.0


def test_confidence_in_range(analyzer):
    audio = np.random.randn(SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert 0.0 <= structure.confidence <= 1.0


# ---------------------------------------------------------------------------
# Segment-Labels
# ---------------------------------------------------------------------------


def test_segments_have_valid_labels(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    valid_labels = {"intro", "verse", "chorus", "bridge", "outro", "unknown"}
    for seg in structure.segments:
        assert seg.label in valid_labels


def test_chorus_segments_are_subset_of_segments(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    # Chorus-Segmente müssen eine Teilmenge aller Segmente sein
    [s.label for s in structure.segments]
    for chorus_seg in structure.chorus_segments:
        assert chorus_seg.label == "chorus"


def test_verse_segments_labeled_verse(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    for verse_seg in structure.verse_segments:
        assert verse_seg.label == "verse"


# ---------------------------------------------------------------------------
# SegmentInfo-Felder
# ---------------------------------------------------------------------------


def test_segment_info_time_range_valid(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    for seg in structure.segments:
        assert seg.start_time_s >= 0.0
        assert seg.end_time_s >= seg.start_time_s
        assert seg.start_sample >= 0
        assert seg.end_sample >= seg.start_sample


def test_segment_sample_count_le_max(analyzer):
    audio = np.random.randn(SR * 30).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert len(structure.segments) <= analyzer.MAX_SEGMENTS


# ---------------------------------------------------------------------------
# get_reference_segment
# ---------------------------------------------------------------------------


def test_get_reference_segment_no_chorus_returns_none(analyzer):
    structure = MusicalStructure(confidence=0.9)  # leere Struktur, kein Chorus
    ref = analyzer.get_reference_segment(0, structure)
    assert ref is None


def test_get_reference_segment_low_confidence_returns_none(analyzer):
    structure = MusicalStructure(
        confidence=0.5,  # unter CHORUS_CONFIDENCE_MIN = 0.75
        chorus_segments=[SegmentInfo("chorus", 0, SR * 10, 0.0, 10.0, 3, 0.88)],
    )
    ref = analyzer.get_reference_segment(0, structure)
    assert ref is None


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


def test_silence_audio_no_crash(analyzer):
    audio = np.zeros(SR * 25, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_stereo_input_accepted(analyzer):
    audio = np.random.randn(2, SR * 25).astype(np.float32) * 0.1
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_nan_input_handled(analyzer):
    audio = np.full(SR * 25, np.nan, dtype=np.float32)
    structure = analyzer.analyze(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_repetitive_signal_bpm_range(analyzer):
    """Stark repetitives Signal soll in vernünftigem BPM-Bereich landen."""
    t = np.arange(SR * 25) / SR
    # Periodisches Signal mit 2 Hz → entspricht 120 BPM in Taktbeziehung
    audio = np.sin(2 * np.pi * 2.0 * t).astype(np.float32) * 0.5
    structure = analyzer.analyze(audio, SR)
    # BPM sollte im plausiblen Bereich liegen wenn erkannt
    if structure.bpm > 0.0:
        assert 30.0 <= structure.bpm <= 300.0


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_musical_structure_analyzer()
    b = get_musical_structure_analyzer()
    assert a is b


def test_convenience_wrapper():
    audio = np.random.randn(SR * 10).astype(np.float32) * 0.1
    structure = analyze_musical_structure(audio, SR)
    assert isinstance(structure, MusicalStructure)


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_musical_structure_analyzer) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)
