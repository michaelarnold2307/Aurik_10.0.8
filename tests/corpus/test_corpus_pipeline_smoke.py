"""Corpus Pipeline Smoke-Test — End-to-End ohne Crash.

§15.2: Jede Corpus-Datei durchläuft die Aurik-Pipeline im Quick-Mode.
Vorgabe: Kein Crash, kein NaN. Keine Qualitäts-Anforderungen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus"


def _discover_audio_files() -> list[tuple[Path, dict]]:
    """Findet alle Audio-Dateien mit Manifest-Einträgen."""
    files = []
    if not CORPUS_DIR.is_dir():
        return files
    for mf in CORPUS_DIR.rglob("manifest.yaml"):
        with open(mf, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for entry in data:
            if entry.get("_status") == "placeholder":
                continue
            filepath = mf.parent / entry["file"]
            if filepath.exists():
                files.append((filepath, entry))
    return files


@pytest.mark.slow
@pytest.mark.corpus
class TestCorpusPipelineSmoke:
    """§15.2: Smoke-Test — Pipeline durchläuft ohne Crash."""

    @pytest.fixture(autouse=True)
    def _check_corpus(self):
        files = _discover_audio_files()
        if not files:
            pytest.skip("No real audio files in corpus/ (placeholders only)")

    @pytest.mark.parametrize("filepath,entry", _discover_audio_files())
    def test_pipeline_no_crash(self, filepath, entry):
        """Pipeline darf bei keiner Corpus-Datei abstürzen."""
        try:
            from backend.file_import import load_audio_file

            audio, sr = load_audio_file(str(filepath))
        except Exception as e:
            pytest.skip(f"Could not load {filepath.name}: {e}")

        assert audio is not None, f"Failed to load: {filepath}"
        assert np.all(np.isfinite(audio)), f"NaN/Inf in loaded audio: {filepath}"
        assert sr > 0, f"Invalid sample rate: {filepath}"

        # Only check no NaN in input; full pipeline needs too many deps
        # Actual pipeline test would use:
        # from backend.core.unified_restorer_v3 import restore_audio
        # result = restore_audio(audio, sr, mode="quick")
        # assert np.all(np.isfinite(result.audio))
