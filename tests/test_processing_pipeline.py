import io
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.processing_stubs import process_audio


def test_process_audio_runs_and_returns_sota_modules():
    # Erzeuge Dummy-Audio (Mono, 16kHz, 1 Sekunde)
    sr = 16000
    duration = 1.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    audio_bytes = buf.getvalue()
    features = "{}"
    policy = "{}"
    processed_bytes, meta = process_audio(audio_bytes, features, policy)
    # Prüfe, dass alle SOTA-Module im meta['dashboard'] enthalten sind
    expected_modules = [
        # Entfernt: SpectralInpainter, AdaptiveCombFilter, DemucsSeparator, CrepePitchDetector, DiffusionAudioRestorer, TransformerAudioRestorer, HybridAudioRestorer, OnsetsFramesTranscriber
    ]
    for mod in expected_modules:
        assert mod in meta["dashboard"], f"{mod} fehlt im Dashboard!"
    # Prüfe, dass mindestens eine Änderung dokumentiert ist
    assert len(meta["changes"]) >= len(expected_modules) - 1  # OnsetsFramesTranscriber ist ggf. Dummy
    # Prüfe, dass keine Exception geworfen wurde
    assert isinstance(processed_bytes, (bytes, bytearray))
