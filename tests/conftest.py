"""
Aurik 9.0 - Zentralisierte Test Fixtures
=========================================

Pytest fixtures für alle Aurik-Tests.
Utilities und Generators sind in test_utils.py definiert.

Author: Aurik Testing Team
Date: 14. Februar 2026
Version: 1.0.0 - INITIAL CONSOLIDATION
"""

import gc
import os
import sys
from typing import Tuple

import numpy as np
import pytest

# Add tests directory to path for test_utils import
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Add project root — benötigt für Modul-Level-Imports wie `from backend.core.X import ...`
# in Testdateien, die mit --import-mode=importlib gesammelt werden.
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add backend/ to path — ermöglicht `from backend.core.X import ...` (= backend/core/X.py)
# in Testdateien und im Produktionscode (z.B. backend/core/causal_defect_graph.py).
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Add src/ to path — enthält validate_musical_goals, orchestrator_and_cli, etc.
src_dir = os.path.join(os.path.dirname(tests_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import utilities from test_utils
from test_utils import (
    MEDIUM_SPECIFIC_THRESHOLDS,
    generate_audio_by_quality,
    generate_medium_specific_audio,
)

# Import Aurik components
try:
    from backend.core.musical_goals.adaptive_goals_system import MaterialQuality
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

    AURIK_COMPONENTS_AVAILABLE = True
except ImportError:
    AURIK_COMPONENTS_AVAILABLE = False
    MaterialQuality = None
    MusicalGoalsChecker = None


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def musical_goals_checker():
    """Fixture: MusicalGoalsChecker instance"""
    if not AURIK_COMPONENTS_AVAILABLE:
        pytest.skip("Aurik components not available")
    return MusicalGoalsChecker()


@pytest.fixture(scope="module", params=["PRISTINE", "EXCELLENT", "GOOD", "FAIR", "POOR", "VERY_POOR", "EXTREME"])
def audio_by_quality_level(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for all 7 Material Quality levels.
    scope=module: pro Modul nur einmal generiert (7× statt N×7).

    Returns:
        Tuple of (audio, sr, quality_level)
    """
    quality_level = request.param
    sr = 48000
    audio = generate_audio_by_quality(quality_level, sr=sr, duration=0.25)
    return audio, sr, quality_level


@pytest.fixture(
    scope="module",
    params=[
        "VINYL_LP_STEREO",
        "CASSETTE_TYPE_I",
        "CD_STANDARD",
        "MP3_320",
        "SACD_DSD",
        "CYLINDER_EDISON",
        "TAPE_15IPS",
        "FM_STEREO",
    ],
)
def audio_by_medium_type_short(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for major medium types (short list for quick tests).
    scope=module + 0.25s: butter/filtfilt auf 12k statt 96k Samples.

    Returns:
        Tuple of (audio, sr, medium_type)
    """
    medium_type = request.param
    sr = 48000
    audio = generate_medium_specific_audio(medium_type, sr=sr, duration=0.25)
    return audio, sr, medium_type


@pytest.fixture(scope="module", params=list(MEDIUM_SPECIFIC_THRESHOLDS.keys()))
def audio_by_medium_type_full(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for ALL 30+ medium types.
    scope=module + 0.25s: butter/filtfilt auf 12k statt 96k Samples (8× schneller).

    Returns:
        Tuple of (audio, sr, medium_type)
    """
    medium_type = request.param
    sr = 48000
    audio = generate_medium_specific_audio(medium_type, sr=sr, duration=0.25)
    return audio, sr, medium_type


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED PERFORMANCE FIXTURES
# Werden pro Worker einmalig erstellt — vermeiden wiederholte np.sin()-Generierung
# ═══════════════════════════════════════════════════════════════════════════

_SESSION_SR = 44100
_SESSION_DURATION = 0.5  # Sekunden — kurz genug für schnelle Tests


@pytest.fixture(scope="session")
def session_sr() -> int:
    """Session-Fixture: Standard-Samplerate 44100 Hz"""
    return _SESSION_SR


@pytest.fixture(scope="session")
def session_mono(session_sr) -> np.ndarray:
    """
    Session-Fixture: Mono-440-Hz-Sinus, 0.5s@44100Hz, float32.
    Einmal pro Worker erstellt — für alle Unit-Tests wiederverwendbar.
    """
    n = int(session_sr * _SESSION_DURATION)
    t = np.linspace(0, _SESSION_DURATION, n, endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture(scope="session")
def session_stereo(session_mono) -> np.ndarray:
    """
    Session-Fixture: Stereo-Audio (2×N), L=440Hz, R=890Hz (leicht verschoben).
    Basiert auf session_mono für Cache-Effizienz.
    """
    n = session_mono.shape[0]
    t = np.linspace(0, _SESSION_DURATION, n, endpoint=False, dtype=np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.stack([session_mono, right])


@pytest.fixture(scope="session")
def session_silence(session_sr) -> np.ndarray:
    """Session-Fixture: Stille (Nullvektor), 0.5s@44100Hz, float32."""
    n = int(session_sr * _SESSION_DURATION)
    return np.zeros(n, dtype=np.float32)


@pytest.fixture(scope="session")
def session_noise(session_sr) -> np.ndarray:
    """Session-Fixture: Weißes Rauschen, 0.5s@44100Hz, float32."""
    rng = np.random.default_rng(seed=42)
    n = int(session_sr * _SESSION_DURATION)
    return rng.standard_normal(n).astype(np.float32) * 0.1


# ═══════════════════════════════════════════════════════════════════════════
# SPEICHERSCHUTZ — verhindert VS Code Crash bei großen Testläufen
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _gc_after_test():
    """Autouse-Fixture: Garbage Collection nach jedem Test.

    Verhindert Speicher-Akkumulation über 267+ Testdateien.
    numpy-Arrays und importierte Module werden freigegeben.
    Overhead: ~1–3 ms pro Test — vernachlässigbar.
    """
    yield
    try:
        gc.collect()
    except (KeyboardInterrupt, SystemExit, Exception):
        pass
