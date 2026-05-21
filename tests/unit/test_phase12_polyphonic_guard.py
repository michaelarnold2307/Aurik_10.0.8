from __future__ import annotations

import numpy as np

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


def test_polyphonic_estimate_is_insufficient_for_single_frame() -> None:
    pitch = np.array([220.0], dtype=np.float64)
    conf = np.array([0.95], dtype=np.float64)
    assert WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)


def test_polyphonic_estimate_is_insufficient_for_sparse_valid_frames() -> None:
    pitch = np.array([220.0, 0.0, 0.0, 0.0, 221.0, 0.0], dtype=np.float64)
    conf = np.array([0.16, 0.01, 0.0, 0.03, 0.14, 0.0], dtype=np.float64)
    assert WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)


def test_polyphonic_estimate_is_sufficient_for_stable_sequence() -> None:
    pitch = np.array([220.0, 221.0, 219.5, 220.8, 220.2, 221.1], dtype=np.float64)
    conf = np.array([0.8, 0.85, 0.78, 0.82, 0.8, 0.81], dtype=np.float64)
    assert not WowFlutterFix._polyphonic_estimate_is_insufficient(pitch, conf)
