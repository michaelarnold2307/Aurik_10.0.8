"""
backend/core/adaptive_chunk_processor.py — Aurik 9 §7.6: Severity-adaptive Chunk-Verarbeitung

Provides chunk-size computation and a generic chunked-processing wrapper
that phases can opt into.  Chunk size is driven by defect severity:

  - silence  → 120 s
  - sev ≥ 0.6 →  5 s  (fine-grained surgical repair)
  - sev ≥ 0.3 → 15 s
  - else      → 60 s  (min 2 s / max 120 s)

Crossfade between chunks uses Hanning window (10 ms) to prevent
boundary artefacts (§4.5 MRSA-Zonen-Spec).

Reference: copilot-instructions.md §7.6 (Chunk-Größe).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §7.6 Chunk-size constants
# ---------------------------------------------------------------------------

CHUNK_SILENCE_S: float = 120.0
CHUNK_HIGH_SEV_S: float = 5.0  # severity ≥ 0.6
CHUNK_MED_SEV_S: float = 15.0  # severity ≥ 0.3
CHUNK_LOW_SEV_S: float = 60.0  # default
CHUNK_MIN_S: float = 2.0
CHUNK_MAX_S: float = 120.0
CROSSFADE_S: float = 0.010  # 10 ms Hanning crossfade


def compute_chunk_size_s(max_severity: float, is_silence: bool = False) -> float:
    """Return adaptive chunk size in seconds per §7.6.

    Args:
        max_severity: Highest defect severity relevant for the current phase (0.0–1.0).
        is_silence:   True if audio is (near-)silence.

    Returns:
        Chunk duration in seconds, clamped to [CHUNK_MIN_S, CHUNK_MAX_S].
    """
    if is_silence:
        return CHUNK_SILENCE_S
    if max_severity >= 0.6:
        return CHUNK_HIGH_SEV_S
    if max_severity >= 0.3:
        return CHUNK_MED_SEV_S
    return CHUNK_LOW_SEV_S


def _is_near_silence(audio: np.ndarray, threshold_db: float = -55.0) -> bool:
    """Check whether *audio* is near-silent (RMS below threshold)."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-15))
    db = 20.0 * np.log10(rms + 1e-15)
    return db < threshold_db


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChunkProcessingResult:
    """Result of chunked phase processing."""

    audio: np.ndarray
    chunk_size_s: float
    n_chunks: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_in_adaptive_chunks(
    phase_fn,
    audio: np.ndarray,
    sr: int,
    max_severity: float,
    *,
    phase_kwargs: dict | None = None,
    crossfade_s: float = CROSSFADE_S,
) -> ChunkProcessingResult:
    """Run *phase_fn* on severity-adaptive chunks with overlap-add crossfade.

    This is an OPT-IN utility.  Phases that benefit from fine-grained
    chunk processing (NR, enhancement, spectral repair) can delegate
    their main loop here.

    Args:
        phase_fn:      Callable(audio_chunk, **phase_kwargs) → np.ndarray
        audio:         Full audio array (1D or 2D [channels, samples]).
        sr:            Sample rate (must be 48000 for processing phases).
        max_severity:  Highest relevant defect severity (0.0–1.0).
        phase_kwargs:  Extra keyword arguments forwarded to *phase_fn*.
        crossfade_s:   Crossfade duration in seconds (default 10 ms).

    Returns:
        ChunkProcessingResult with stitched audio.
    """
    if phase_kwargs is None:
        phase_kwargs = {}

    is_stereo = audio.ndim == 2
    n_samples = audio.shape[-1]
    duration_s = n_samples / sr

    is_silence = _is_near_silence(audio)
    chunk_s = compute_chunk_size_s(max_severity, is_silence=is_silence)

    # If audio fits in a single chunk, skip chunking overhead
    if duration_s <= chunk_s + crossfade_s:
        result_audio = phase_fn(audio, **phase_kwargs)
        return ChunkProcessingResult(
            audio=np.asarray(result_audio, dtype=np.float32),
            chunk_size_s=chunk_s,
            n_chunks=1,
        )

    chunk_samples = int(chunk_s * sr)
    fade_samples = max(1, int(crossfade_s * sr))
    hop_samples = max(1, chunk_samples - fade_samples)  # overlap = fade_samples

    # Hanning fade windows
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    # Output buffer
    if is_stereo:
        out = np.zeros_like(audio, dtype=np.float32)
    else:
        out = np.zeros(n_samples, dtype=np.float32)
    weight = np.zeros(n_samples, dtype=np.float32)

    n_chunks = 0
    pos = 0
    while pos < n_samples:
        end = min(pos + chunk_samples, n_samples)
        if is_stereo:
            chunk = audio[:, pos:end].copy()
        else:
            chunk = audio[pos:end].copy()

        # Process chunk
        processed = phase_fn(chunk, **phase_kwargs)
        processed = np.asarray(processed, dtype=np.float32)

        # Build weight envelope for this chunk
        chunk_len = end - pos
        w = np.ones(chunk_len, dtype=np.float32)
        # Fade-in (except first chunk)
        if pos > 0 and fade_samples < chunk_len:
            w[:fade_samples] = fade_in[:fade_samples]
        # Fade-out (except last chunk)
        if end < n_samples and fade_samples < chunk_len:
            w[-fade_samples:] = fade_out[:fade_samples]

        # Accumulate
        if is_stereo:
            for ch in range(processed.shape[0]):
                p_len = min(processed.shape[1], chunk_len)
                out[ch, pos : pos + p_len] += processed[ch, :p_len] * w[:p_len]
        else:
            p_len = min(len(processed), chunk_len)
            out[pos : pos + p_len] += processed[:p_len] * w[:p_len]
        weight[pos : pos + chunk_len] += w

        n_chunks += 1
        pos += hop_samples

    # Normalize by accumulated weight (avoid division by zero)
    weight = np.maximum(weight, 1e-8)
    if is_stereo:
        out /= weight[np.newaxis, :]
    else:
        out /= weight

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    logger.debug(
        "§7.6 AdaptiveChunk: severity=%.2f → chunk=%.1fs, n_chunks=%d, crossfade=%.0fms",
        max_severity,
        chunk_s,
        n_chunks,
        crossfade_s * 1000,
    )

    return ChunkProcessingResult(audio=out, chunk_size_s=chunk_s, n_chunks=n_chunks)


# ---------------------------------------------------------------------------
# Thread-safe singleton (§3.2)
# ---------------------------------------------------------------------------

_instance: AdaptiveChunkProcessor | None = None
_lock = threading.Lock()


class AdaptiveChunkProcessor:
    """Singleton wrapper for adaptive chunk processing."""

    def compute_chunk_size(self, max_severity: float, is_silence: bool = False) -> float:
        return compute_chunk_size_s(max_severity, is_silence=is_silence)

    def process(self, phase_fn, audio, sr, max_severity, **kwargs):
        return process_in_adaptive_chunks(phase_fn, audio, sr, max_severity, **kwargs)


def get_adaptive_chunk_processor() -> AdaptiveChunkProcessor:
    """Thread-safe singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AdaptiveChunkProcessor()
    return _instance
