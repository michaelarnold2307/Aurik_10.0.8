"""
core.musical_phrase_context_extractor — VERALTET (Kompatibilitäts-Shim)
========================================================================

Dieses Modul ist ein reiner Re-Export-Shim für ``core.musical_phrase_context``.
Alle Klassen, Funktionen und Konstanten werden von dort bezogen.

Migrationsanleitung::

    # Alt:
    from backend.core.musical_phrase_context_extractor import MusicalPhraseContextExtractor
    # Neu:
    from backend.core.musical_phrase_context import MusicalPhraseContextExtractor

Dieser Shim wird in einer zukünftigen Version entfernt.
Referenz: §2.12 Aurik-9-Spec (v9.9.5)
"""

import warnings as _warnings

_warnings.warn(
    "core.musical_phrase_context_extractor ist veraltet. " "Verwende stattdessen core.musical_phrase_context.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.core.musical_phrase_context import (  # noqa: F401, E402
    CHROMA_JUMP_THRESHOLD,
    ENERGY_DELTA_DB,
    GAP_FRACTION_THRESHOLD,
    GAP_PHRASE_RATIO_MAX,
    MAX_CONTEXT_DURATION_S,
    MAX_GAP_DURATION_MS,
    MIN_FILE_DURATION_S,
    MIN_PHRASE_BEATS,
    N_CHROMA,
    MusicalPhraseContextExtractor,
    PhraseBoundary,
    PhraseContext,
    extract_phrase_context,
    get_phrase_context_extractor,
    get_phrase_extractor,
)

__all__ = [
    "PhraseContext",
    "PhraseBoundary",
    "MusicalPhraseContextExtractor",
    "get_phrase_extractor",
    "get_phrase_context_extractor",
    "extract_phrase_context",
    "MIN_FILE_DURATION_S",
    "MAX_CONTEXT_DURATION_S",
    "MIN_PHRASE_BEATS",
    "MAX_GAP_DURATION_MS",
    "GAP_FRACTION_THRESHOLD",
    "N_CHROMA",
    "CHROMA_JUMP_THRESHOLD",
    "ENERGY_DELTA_DB",
    "GAP_PHRASE_RATIO_MAX",
]
