"""
Tests für SongStructureAnalyzer (§2.52b).
"""

import numpy as np
import pytest


@pytest.fixture()
def long_audio():
    """3 Minuten Sinus-Audio bei 48 kHz für Struktur-Tests."""
    sr = 48000
    duration = 180  # 3 Minuten
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    # Dynamik simulieren: Energie variiert über Zeit
    envelope = 0.3 + 0.4 * np.sin(2 * np.pi * t / 30)  # 30 s Periodik
    audio = (envelope * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture()
def short_audio():
    """20 Sekunden Audio für Edge-Case-Tests."""
    sr = 48000
    t = np.linspace(0, 20.0, sr * 20, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


class TestSongStructureAnalyzerImport:
    def test_import_ok(self):
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        assert callable(get_song_structure_analyzer)

    def test_singleton(self):
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        a = get_song_structure_analyzer()
        b = get_song_structure_analyzer()
        assert a is b

    def test_dataclass_import(self):
        from backend.core.song_structure_analyzer import SongSegment

        seg = SongSegment(
            start_s=0.0,
            end_s=30.0,
            label="verse",
            energy_level=0.5,
            has_vocals=True,
            is_climax=False,
        )
        assert seg.label == "verse"
        assert seg.start_s == 0.0


class TestAnalyzeStructure:
    def test_returns_list(self, short_audio):
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(audio, sr)

        assert isinstance(segments, list)
        assert len(segments) >= 1

    def test_segments_cover_full_duration(self, short_audio):
        """Segments sollen die gesamte Dauer abdecken."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(audio, sr)

        assert segments[0].start_s == pytest.approx(0.0, abs=0.5)
        duration = len(audio) / sr
        assert segments[-1].end_s == pytest.approx(duration, abs=1.5)

    def test_segments_sorted_by_time(self, short_audio):
        """Segmente müssen aufsteigend nach start_s sortiert sein."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(audio, sr)

        starts = [s.start_s for s in segments]
        assert starts == sorted(starts)

    def test_energy_level_in_range(self, short_audio):
        """energy_level MUSS in [0, 1] liegen."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(audio, sr)

        for seg in segments:
            assert 0.0 <= seg.energy_level <= 1.0, f"energy_level={seg.energy_level} out of range"

    def test_valid_label(self, short_audio):
        """Label muss zu den erlaubten Werten gehören."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(audio, sr)

        valid_labels = {"intro", "verse", "chorus", "bridge", "outro", "instrumental", "silence", "unknown"}
        for seg in segments:
            assert seg.label in valid_labels, f"Unbekanntes Label: {seg.label}"

    def test_stereo_input(self):
        """Stereo-Input (2, N) wird ohne Fehler verarbeitet."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        sr = 48000
        t = np.linspace(0, 30.0, sr * 30, endpoint=False)
        mono = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([mono, mono])

        analyzer = get_song_structure_analyzer()
        segments = analyzer.analyze_structure(stereo, sr)

        assert isinstance(segments, list)
        assert len(segments) >= 1

    def test_panns_singing_influence(self, short_audio):
        """Hohe PANNs-Konfidenz erhöht has_vocals-Wahrscheinlichkeit."""
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        audio, sr = short_audio
        analyzer = get_song_structure_analyzer()

        segs_no_vocal = analyzer.analyze_structure(audio, sr, panns_singing_confidence=0.0)
        segs_with_vocal = analyzer.analyze_structure(audio, sr, panns_singing_confidence=0.9)

        n_vocal_no = sum(1 for s in segs_no_vocal if s.has_vocals)
        n_vocal_with = sum(1 for s in segs_with_vocal if s.has_vocals)
        # Mit hoher Konfidenz sollten mehr Segmente als vokal markiert sein
        assert n_vocal_with >= n_vocal_no


class TestGetStrengthScalar:
    def test_none_returns_1(self):
        from backend.core.song_structure_analyzer import get_song_structure_analyzer

        analyzer = get_song_structure_analyzer()
        assert analyzer.get_strength_scalar(None) == pytest.approx(1.0)

    def test_verse_nr_strength(self):
        from backend.core.song_structure_analyzer import SongSegment, get_song_structure_analyzer

        analyzer = get_song_structure_analyzer()
        seg = SongSegment(
            start_s=30.0,
            end_s=60.0,
            label="verse",
            energy_level=0.4,
            has_vocals=True,
            is_climax=False,
        )
        scalar = analyzer.get_strength_scalar(seg, "nr_strength")
        assert scalar == pytest.approx(1.15, abs=1e-3)

    def test_climax_overrides_label(self):
        """Climax-Flag überschreibt Label-Scalar."""
        from backend.core.song_structure_analyzer import SongSegment, get_song_structure_analyzer

        analyzer = get_song_structure_analyzer()
        seg = SongSegment(
            start_s=60.0,
            end_s=90.0,
            label="verse",
            energy_level=0.9,
            has_vocals=True,
            is_climax=True,
        )
        scalar = analyzer.get_strength_scalar(seg, "nr_strength")
        # Climax MUSS 0.85 sein (§2.52b)
        assert scalar == pytest.approx(0.85, abs=1e-3)

    def test_bounded_range(self):
        """Scalar MUSS im Hard-Bound [0.70, 1.30] liegen."""
        from backend.core.song_structure_analyzer import SongSegment, get_song_structure_analyzer

        analyzer = get_song_structure_analyzer()
        for label in ["intro", "verse", "chorus", "bridge", "outro", "instrumental", "silence", "unknown"]:
            seg = SongSegment(
                start_s=0.0,
                end_s=10.0,
                label=label,
                energy_level=0.5,
                has_vocals=True,
                is_climax=False,
            )
            scalar = analyzer.get_strength_scalar(seg)
            assert 0.70 <= scalar <= 1.30, f"label={label} scalar={scalar} out of [0.70, 1.30]"
