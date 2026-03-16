"""Shim — forwards to canonical backend.core.regulator.mastering."""

from backend.core.regulator.mastering import (  # noqa: F401
    adaptive_eq,
    dither,
    limiter,
    loudness_normalize,
    lufs_normalize,
    mastering_chain,
    multiband_compress,
    simple_compressor,
    simple_eq,
    stereo_enhance,
)

__all__ = [
    "mastering_chain",
    "lufs_normalize",
    "multiband_compress",
    "adaptive_eq",
    "limiter",
    "stereo_enhance",
    "loudness_normalize",
    "dither",
    "simple_eq",
    "simple_compressor",
]
