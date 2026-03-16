#!/usr/bin/env python3
"""
Clipping Detection — §6.3 Spec: CLIPPING vs. SOFT_SATURATION Discrimination.

Critical distinction (copilot-instructions.md §6.3):
    SOFT_SATURATION = Tube/Tape character → PRESERVE
    CLIPPING        = Amplitude damage    → REPAIR

Algorithm:
    1. flat_tops_pct: fraction of samples at hard-clip boundary (|x| ≥ 0.999)
    2. THD analysis via STFT-based harmonic energy estimation:
       - THD_odd  = energy at odd harmonics  (clipping signature: square-wave-like)
       - THD_even = energy at even harmonics (saturation signature: tanh-like)
    3. Classification rule (normative per §6.3):
       CLIPPING:        flat_tops > 0.1 % AND THD_odd > THD_even × 1.5
       SOFT_SATURATION: flat_tops < 0.1 % OR  THD_even > THD_odd

Singleton: thread-safe, Double-Checked Locking (§3.x).

Author: Aurik Development Team
Version: 9.10.57
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (normative per §6.3)
# ---------------------------------------------------------------------------

FLAT_TOPS_THRESHOLD_PCT: float = 0.1        # > this → potential hard clipping
FLAT_TOPS_CLIP_BOUNDARY: float = 0.999      # samples within this range are "at ceiling"
THD_ODD_DOMINANCE_FACTOR: float = 1.5      # THD_odd > THD_even × this → CLIPPING
ANALYSIS_WINDOW_SAMPLES: int = 2048        # FFT window size for harmonic analysis
ANALYSIS_HOP_SAMPLES: int = 512            # hop between analysis windows
FUNDAMENTAL_MIN_HZ: float = 40.0          # lowest fundamental to search (E1 ~ 41 Hz)
FUNDAMENTAL_MAX_HZ: float = 800.0         # highest fundamental (G5 ~ 784 Hz)
MAX_HARMONIC_ORDER: int = 10              # analyse up to this harmonic order
HARMONIC_BIN_RADIUS: int = 1             # bins around harmonic centre to include


class ClippingType(Enum):
    """Discrimination result per §6.3."""

    CLIPPING = "clipping"                  # Hard amplitude clipping — REPAIR (→ phase_23)
    SOFT_SATURATION = "soft_saturation"   # Tube/Tape saturation character — PRESERVE


@dataclass
class ClippingAnalysisResult:
    """Full analysis result returned by classify_clipping()."""

    clipping_type: ClippingType
    flat_tops_pct: float          # % of samples at ±1.0 boundary
    thd_odd: float                # Total odd-harmonic energy (summed, normalised)
    thd_even: float               # Total even-harmonic energy (summed, normalised)
    confidence: float             # 0.0–1.0 — distance from decision boundary
    is_clipping: bool             # True when clipping_type == CLIPPING

    @property
    def should_repair(self) -> bool:
        """True when the pipeline should activate Clipping-Repair (phase_23)."""
        return self.is_clipping

    @property
    def should_preserve(self) -> bool:
        """True when the pipeline should skip Clipping-Repair and preserve saturation."""
        return not self.is_clipping


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert (N,) or (N, C) audio to 1-D mono float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def _flat_tops_pct(mono: np.ndarray, boundary: float = FLAT_TOPS_CLIP_BOUNDARY) -> float:
    """
    Return the percentage of samples sitting at or above the clip boundary.

    A sample is considered 'at the ceiling' when |x| >= boundary.
    For hard digital clipping this can exceed several percent.
    For analogue soft saturation this is typically < 0.05 %.
    """
    if len(mono) == 0:
        return 0.0
    ratio = float(np.mean(np.abs(mono) >= boundary))
    return ratio * 100.0


def _find_dominant_fundamental_hz(mono: np.ndarray, sr: int) -> Optional[float]:
    """
    Estimate dominant fundamental frequency via normalised autocorrelation (AMDF).

    Returns None when no clear fundamental within [FUNDAMENTAL_MIN_HZ, FUNDAMENTAL_MAX_HZ]
    is detectable (polyphonic / noise-dominated material).
    """
    n = min(len(mono), 4096)
    if n < 256:
        return None

    segment = mono[:n].astype(np.float64)
    # Normalised autocorrelation
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[n - 1:]  # keep causal half
    if np.max(np.abs(corr)) < 1e-8:
        return None
    corr = corr / (corr[0] + 1e-12)

    lag_min = max(1, int(sr / FUNDAMENTAL_MAX_HZ))
    lag_max = min(n - 1, int(sr / FUNDAMENTAL_MIN_HZ))
    if lag_max <= lag_min:
        return None

    search_region = corr[lag_min:lag_max + 1]
    peak_rel = int(np.argmax(search_region))
    peak_lag = lag_min + peak_rel
    peak_val = float(corr[peak_lag])

    # Require a strong correlation peak to confirm a clear fundamental
    if peak_val < 0.40:
        return None

    return float(sr) / float(peak_lag)


def _harmonic_energies(
    fft_mag: np.ndarray,
    fundamental_hz: float,
    sr: int,
    fft_size: int,
) -> tuple[float, float]:
    """
    Sum FFT magnitude energy at odd and even harmonics above the fundamental.

    Returns (thd_odd_energy, thd_even_energy) normalised by fundamental energy.
    Energy is summed over bins [k-HARMONIC_BIN_RADIUS, k+HARMONIC_BIN_RADIUS]
    around each harmonic centre bin.
    """
    bin_hz = sr / fft_size
    fundamental_bin = max(1, round(fundamental_hz / bin_hz))

    # Fundamental energy (normalisation)
    f_lo = max(0, fundamental_bin - HARMONIC_BIN_RADIUS)
    f_hi = min(len(fft_mag), fundamental_bin + HARMONIC_BIN_RADIUS + 1)
    fundamental_energy = float(np.sum(fft_mag[f_lo:f_hi] ** 2)) + 1e-12

    odd_energy = 0.0
    even_energy = 0.0

    for n in range(2, MAX_HARMONIC_ORDER + 1):
        harmonic_bin = round(fundamental_hz * n / bin_hz)
        if harmonic_bin >= len(fft_mag):
            break
        lo = max(0, harmonic_bin - HARMONIC_BIN_RADIUS)
        hi = min(len(fft_mag), harmonic_bin + HARMONIC_BIN_RADIUS + 1)
        energy = float(np.sum(fft_mag[lo:hi] ** 2))
        if n % 2 == 0:
            even_energy += energy
        else:
            odd_energy += energy

    return odd_energy / fundamental_energy, even_energy / fundamental_energy


def _polyphonic_odd_even_estimate(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Broadband odd/even harmonic ratio for polyphonic material without clear fundamental.

    Approach: window the signal, detect tonal peaks in each frame, and accumulate
    harmonic energy estimates over all detected peaks. Returns averaged ratio.

    Falls back to (1.0, 1.0) — neutral — if no tonal peaks are reliably found.
    """
    fft_size = ANALYSIS_WINDOW_SAMPLES
    window = np.hanning(fft_size)
    n_frames = max(1, (len(mono) - fft_size) // ANALYSIS_HOP_SAMPLES + 1)
    n_frames = min(n_frames, 32)  # cap at 32 frames for speed

    odd_acc = 0.0
    even_acc = 0.0
    valid_frames = 0

    for k in range(n_frames):
        start = k * ANALYSIS_HOP_SAMPLES
        end = start + fft_size
        if end > len(mono):
            break
        frame = mono[start:end] * window
        if np.max(np.abs(frame)) < 0.01:
            continue  # silent frame

        fft_mag = np.abs(np.fft.rfft(frame, n=fft_size))
        # Find peak (potential local fundamental)
        peak_bin = int(np.argmax(fft_mag[1:]) + 1)  # skip DC
        peak_hz = float(peak_bin) * sr / fft_size

        if peak_hz < FUNDAMENTAL_MIN_HZ or peak_hz > FUNDAMENTAL_MAX_HZ:
            continue

        o, e = _harmonic_energies(fft_mag, peak_hz, sr, fft_size)
        odd_acc += o
        even_acc += e
        valid_frames += 1

    if valid_frames == 0:
        return 1.0, 1.0

    return odd_acc / valid_frames, even_acc / valid_frames


def _compute_thd(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Compute (thd_odd, thd_even) for the audio signal.

    Strategy:
    1. Try single stable fundamental via autocorrelation (works for voice/single-instrument).
    2. Fall back to polyphonic frame-by-frame analysis.
    """
    fundamental_hz = _find_dominant_fundamental_hz(mono, sr)

    if fundamental_hz is not None:
        # Use global FFT for single-fundamental case
        n = min(len(mono), 16384)
        fft_mag = np.abs(np.fft.rfft(mono[:n], n=n))
        return _harmonic_energies(fft_mag, fundamental_hz, sr, n)
    else:
        return _polyphonic_odd_even_estimate(mono, sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_clipping(audio: np.ndarray, sr: int) -> ClippingType:
    """
    Discriminate CLIPPING from SOFT_SATURATION via harmonic analysis (§6.3).

    Classification rules (normative):
        CLIPPING:        flat_tops > 0.1 % AND THD_odd > THD_even × 1.5
        SOFT_SATURATION: flat_tops < 0.1 % OR  THD_even > THD_odd

    SOFT_SATURATION → Pipeline SKIPS clipping repair completely.
    CLIPPING        → Pipeline activates phase_23 (spectral repair / declipping).

    Args:
        audio: Audio signal as float32 numpy array, shape (N,) or (N, 2).
               Expected range [-1.0, 1.0].
        sr:    Sample rate — MUST be 48000 Hz.

    Returns:
        ClippingType.CLIPPING or ClippingType.SOFT_SATURATION.

    Raises:
        AssertionError: When sr != 48000.
        ValueError:     When audio is empty or all-zero.
    """
    return analyse_clipping(audio, sr).clipping_type


def analyse_clipping(audio: np.ndarray, sr: int) -> ClippingAnalysisResult:
    """
    Full clipping analysis with intermediate metrics.

    Returns ClippingAnalysisResult with flat_tops_pct, thd_odd, thd_even,
    confidence, and the final ClippingType decision.

    Args:
        audio: Audio signal, float32, shape (N,) or (N, C).
        sr:    Sample rate — MUST be 48000 Hz.
    """
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    mono = _to_mono(audio)

    if len(mono) == 0 or np.max(np.abs(mono)) < 1e-8:
        # Silence or empty signal — no clipping possible
        return ClippingAnalysisResult(
            clipping_type=ClippingType.SOFT_SATURATION,
            flat_tops_pct=0.0,
            thd_odd=0.0,
            thd_even=0.0,
            confidence=1.0,
            is_clipping=False,
        )

    # --- Step 1: flat top detection ---
    flat_pct = _flat_tops_pct(mono)
    logger.debug("ClippingDetector: flat_tops_pct=%.4f%%", flat_pct)

    # --- Step 2: harmonic distortion analysis ---
    thd_odd, thd_even = _compute_thd(mono, sr)
    thd_odd = float(np.clip(thd_odd, 0.0, 1e6))
    thd_even = float(np.clip(thd_even, 0.0, 1e6))
    logger.debug("ClippingDetector: thd_odd=%.4f thd_even=%.4f", thd_odd, thd_even)

    # --- Step 3: classify (normative rule §6.3) ---
    flat_tops_exceeded = flat_pct > FLAT_TOPS_THRESHOLD_PCT
    odd_dominant = thd_odd > thd_even * THD_ODD_DOMINANCE_FACTOR

    is_clipping = flat_tops_exceeded and odd_dominant
    clipping_type = ClippingType.CLIPPING if is_clipping else ClippingType.SOFT_SATURATION

    # --- Confidence: distance from both thresholds ---
    flat_distance = abs(flat_pct - FLAT_TOPS_THRESHOLD_PCT) / max(flat_pct, FLAT_TOPS_THRESHOLD_PCT, 1e-6)
    if thd_odd + thd_even < 1e-8:
        thd_ratio_distance = 1.0
    else:
        thd_ratio = thd_odd / (thd_even * THD_ODD_DOMINANCE_FACTOR + 1e-12)
        thd_ratio_distance = abs(math.log(max(thd_ratio, 1e-6)))
        thd_ratio_distance = min(thd_ratio_distance / 2.0, 1.0)  # normalise to [0,1]

    confidence = float(np.clip(
        min(flat_distance, thd_ratio_distance + 0.1),
        0.0, 1.0
    ))

    logger.info(
        "ClippingDetector: result=%s flat_tops=%.3f%% odd=%.3f even=%.3f confidence=%.2f",
        clipping_type.value, flat_pct, thd_odd, thd_even, confidence,
    )

    return ClippingAnalysisResult(
        clipping_type=clipping_type,
        flat_tops_pct=flat_pct,
        thd_odd=thd_odd,
        thd_even=thd_even,
        confidence=confidence,
        is_clipping=is_clipping,
    )


# ---------------------------------------------------------------------------
# Singleton (§3.x — thread-safe, Double-Checked Locking)
# ---------------------------------------------------------------------------

class ClippingClassifier:
    """
    Stateless wrapper providing the singleton access point per §3.x.

    All state lives in module-level functions; this class exists only to
    satisfy the Singleton-Pattern requirement and provide a discoverable API.
    """

    def classify(self, audio: np.ndarray, sr: int) -> ClippingType:
        """Classify audio as CLIPPING or SOFT_SATURATION."""
        return classify_clipping(audio, sr)

    def analyse(self, audio: np.ndarray, sr: int) -> ClippingAnalysisResult:
        """Return full analysis result including intermediate metrics."""
        return analyse_clipping(audio, sr)


_instance: Optional[ClippingClassifier] = None
_lock = threading.Lock()


def get_clipping_classifier() -> ClippingClassifier:
    """Return the singleton ClippingClassifier instance (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ClippingClassifier()
    return _instance
