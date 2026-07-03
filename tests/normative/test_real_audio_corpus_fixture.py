"""Normative Real-Audio-Korpus-Fixture fuer R5-R12-Folgegates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_real_audio_corpus_fixture_covers_multiple_local_sources(
    real_audio_corpus_cases: list[dict[str, Any]],
) -> None:
    paths = [Path(str(case["path"])) for case in real_audio_corpus_cases]
    categories = {path.parent.name for path in paths}

    assert len(paths) >= 4
    assert {"tape", "vinyl", "digital", "vocals"} & categories

    for case in real_audio_corpus_cases:
        audio = np.asarray(case["audio"], dtype=np.float32)
        sr = int(case["sr"])
        assert sr == 48_000
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        assert audio.shape[0] >= sr * 2
        assert np.isfinite(audio).all()
        assert float(np.max(np.abs(audio))) <= 1.0 + 1e-6
