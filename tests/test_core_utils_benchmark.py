import time

import numpy as np

from backend.core.core_utils import normalize_audio


def benchmark_normalize_audio():
    audio = np.random.uniform(-1, 1, 48000 * 60).astype(np.float32)  # 1 Minute Audio
    start = time.perf_counter()
    norm = normalize_audio(audio)
    duration = time.perf_counter() - start
    print(f"normalize_audio: {duration:.6f} Sekunden für 1min Audio")
    assert norm.shape == audio.shape
    assert np.max(np.abs(norm)) <= 0.999 + 1e-6


def test_benchmark_normalize_audio():
    benchmark_normalize_audio()
