"""§2.61 SectionGoalAdapter — verbindet MusicalStructureAnalyzer mit dem Fahrplan.

Der Adapter:
  1. Nimmt Audio + Samplerate
  2. Ruft MusicalStructureAnalyzer.analyze() für SSM-basierte Segmentierung
  3. Gibt Sektionen im Fahrplan-Format zurück: [(start_s, end_s, label), ...]

Minimal-Interface: eine Funktion `adapt(audio, sr) -> list[tuple[float, float, str]]`.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_sections(
    audio: np.ndarray,
    sr: int,
    *,
    min_duration_s: float = 20.0,
) -> list[tuple[float, float, str]]:
    """Analysiert Audio und gibt Sektionen für den Fahrplan zurück.

    Args:
        audio: (samples,) oder (channels, samples) ndarray
        sr: Sample-Rate
        min_duration_s: Audio kürzer als das → Fallback auf eine "full"-Sektion

    Returns:
        Liste von (start_s, end_s, label) Tupeln, z.B.:
        [(0.0, 22.3, "intro"), (22.3, 67.8, "verse"), ...]
    """
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    n = arr.shape[0] if arr.ndim == 1 else max(arr.shape)
    duration_s = n / max(sr, 1)

    # Kurzes Audio → eine Sektion
    if duration_s < min_duration_s:
        return [(0.0, duration_s, "full")]

    try:
        from backend.core.musical_structure_analyzer import MusicalStructureAnalyzer

        analyzer = MusicalStructureAnalyzer()
        structure = analyzer.analyze(arr, sr)

        if not structure.segments:
            return [(0.0, duration_s, "full")]

        sections: list[tuple[float, float, str]] = []
        for seg in structure.segments:
            start = float(seg.start_s)
            end = float(seg.end_s)
            label = str(seg.label).lower().strip()
            # Normalisiere Label auf Fahrplan-kompatible Kategorien
            label = _normalize_label(label)
            sections.append((start, end, label))

        return _merge_adjacent(sections)

    except Exception as exc:
        logger.debug("SectionGoalAdapter: Analyse fehlgeschlagen → full: %s", exc)
        return [(0.0, duration_s, "full")]


def _normalize_label(label: str) -> str:
    """Vereinheitlicht Labels auf Fahrplan-kompatible Namen."""
    mapping: dict[str, str] = {
        "intro": "intro",
        "outro": "outro",
        "verse": "verse",
        "chorus": "chorus",
        "bridge": "bridge",
        "pre-chorus": "chorus",
        "pre_chorus": "chorus",
        "prechorus": "chorus",
        "post-chorus": "chorus",
        "post_chorus": "chorus",
        "solo": "bridge",  # Solo → konservativ wie Bridge
        "instrumental": "verse",
        "break": "bridge",
        "interlude": "bridge",
        "fade": "outro",
        "fade_out": "outro",
        "fadeout": "outro",
        "silence": "silence",
        "quiet": "outro",
        "unknown": "full",
        "full": "full",
    }
    return mapping.get(label, "full")


def _merge_adjacent(
    sections: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge benachbarte Sektionen mit gleichem Label."""
    if len(sections) <= 1:
        return sections

    merged: list[tuple[float, float, str]] = []
    for start, end, label in sections:
        if merged and merged[-1][2] == label:
            prev_start, _, prev_label = merged.pop()
            merged.append((prev_start, end, prev_label))
        else:
            merged.append((start, end, label))
    return merged
