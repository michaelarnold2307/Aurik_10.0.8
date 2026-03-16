"""
Optimized DSP Module for AURIK v8
==================================

High-performance DSP operations using:
- NumExpr: 2× speedup for vectorized operations
- Cython: 3-5× speedup for critical loops
- pyFFTW: 1.5-2× speedup for FFT operations

Combined speedup: 2-5× faster DSP processing
"""

# NumExpr optimizations (always available)
from .numexpr_ops import OptimizedDSP, hard_threshold, soft_threshold, spectral_gate

# FFT caching
try:
    from .fft_cache import CachedFFT, irfft, istft, rfft, stft

    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    CachedFFT = None
    rfft = None
    irfft = None
    stft = None
    istft = None

# Cython loops (requires compilation)
try:
    from . import cython_loops

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    cython_loops = None

__all__ = [
    # NumExpr
    "OptimizedDSP",
    "spectral_gate",
    "soft_threshold",
    "hard_threshold",
    # FFT
    "CachedFFT",
    "rfft",
    "irfft",
    "stft",
    "istft",
    # Cython
    "cython_loops",
    # Flags
    "HAS_PYFFTW",
    "HAS_CYTHON",
]
