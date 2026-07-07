"""Unit-Tests für musiclover-Exportoptimierung im AudioExporter.

Testet _apply_musiclover_export_optimizations direkt — keine sf.write-Mocks nötig.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.audio_exporter import _apply_musiclover_export_optimizations


def _anti_phase_stereo(n: int = 128) -> np.ndarray:
    return np.column_stack(
        [np.ones(n, dtype=np.float32), -np.ones(n, dtype=np.float32)]
    )


def test_mono_guard_applies_side_softening() -> None:
    """Anti-Phasen-Stereo: Side-Pegel wird reduziert (kein harter Mono-Downmix)."""
    audio = _anti_phase_stereo(96)
    metadata = {
        "quality_gate_musiclover_mono_warning": "True",
        "quality_gate_musiclover_mono_softened": "False",
        "quality_gate_musiclover_vqi": "1.0",
        "quality_gate_musiclover_temporal_hotspots": "0",
        "quality_gate_musiclover_remaining_goals": "0",
    }

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    assert result.shape == audio.shape
    # Side softening: Links > 0.92 (original 1.0 minus side_scale), Rechts < -0.92
    assert float(result[:, 0].mean()) == pytest.approx(0.92, abs=1e-6)
    assert float(result[:, 1].mean()) == pytest.approx(-0.92, abs=1e-6)


def test_mono_guard_skipped_when_already_softened() -> None:
    """Bereits gemildertes Audio wird nicht erneut modifiziert (Passthrough)."""
    audio = _anti_phase_stereo(96)
    metadata = {
        "quality_gate_musiclover_mono_warning": "True",
        "quality_gate_musiclover_mono_softened": "True",
        "quality_gate_musiclover_vqi": "1.0",
    }

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    assert result.shape == audio.shape
    # Keine Änderung — already_softened=True → Passthrough
    assert float(result[:, 0].mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(result[:, 1].mean()) == pytest.approx(-1.0, abs=1e-6)


def test_mono_input_unchanged() -> None:
    """Mono-Audio wird nicht verändert."""
    audio = np.ones((96, 1), dtype=np.float32) * 0.5
    metadata = {"quality_gate_musiclover_mono_warning": "True"}

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    np.testing.assert_array_almost_equal(result, audio)


def test_no_warning_passthrough() -> None:
    """Ohne Mono-Warnung: Passthrough."""
    audio = _anti_phase_stereo(96)
    metadata: dict[str, str] = {}

    result = _apply_musiclover_export_optimizations(audio, metadata, None)

    np.testing.assert_array_almost_equal(result, audio)
