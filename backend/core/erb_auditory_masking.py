"""ERB-scaled auditory masking model for frequency-dependent salience estimation.

Replaces fixed broadband masking thresholds with a psychoacoustically correct
frequency-dependent model based on Equivalent Rectangular Bandwidth (ERB)
critical-band filters and the power-spectrum model of masking.

Scientific basis:
- Glasberg, B.R. & Moore, B.C.J. (1990). "Derivation of auditory filter
  shapes from notched-noise data". *Hearing Research* 47, 103-138.
- Moore, B.C.J. & Glasberg, B.R. (1983). "Suggested formulae for calculating
  auditory-filter bandwidths and excitation patterns". *JASA* 74(3), 750-753.
- Moore, B.C.J., Glasberg, B.R. & Baer, T. (1997). "A model for the prediction
  of thresholds, loudness, and partial loudness". *JAES* 45(4), 224-240.
- Brungart, D.S. (2001). "Informational and energetic masking effects in the
  perception of two simultaneous talkers". *JASA* 109(3), 1101-1109.

Key improvements over fixed-threshold model:
1. **Frequency-dependent masking spread** via ERB excitation patterns
   (low frequencies mask wider than high frequencies — critical-ratio asymmetry)
2. **Exponential temporal decay** (not step function) for forward masking
   with 3:1 forward/backward asymmetry (Jesteadt, Bacon & Lehman 1982)
3. **Informational masking bonus** for harmonically related content
   (Brungart 2001; reduces salience for defects in tonal passages)

Module invariants (§3.x compliant):
- Thread-safe singleton via double-checked locking
- NaN/Inf guard on all numeric outputs
- No audio modification — produces masking thresholds only
- No sample-rate assertion (analysis module — works at native import SR)
- English docstrings and log messages
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERB auditory filter model (Glasberg & Moore 1990)
# ---------------------------------------------------------------------------


def erb_hz(f_hz: float) -> float:
    """Equivalent Rectangular Bandwidth at frequency *f_hz*.

    ERB(f) = 24.7 * (4.37 * f/1000 + 1)   [Glasberg & Moore 1990, Eq. 3]
    """
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)


def erb_rate(f_hz: float) -> float:
    """ERB-rate number (Cams) for frequency *f_hz*.

    N(f) = 21.4 * log10(0.00437*f + 1)     [Glasberg & Moore 1990, Eq. 4]
    """
    return 21.4 * np.log10(0.00437 * f_hz + 1.0)


def erb_rate_to_hz(n_cams: float) -> float:
    """Convert ERB-rate (Cams) back to Hz."""
    return (10.0 ** (n_cams / 21.4) - 1.0) / 0.00437


# ---------------------------------------------------------------------------
# Excitation spreading function
# ---------------------------------------------------------------------------


def _spreading_function_db(
    fc_masker: float,
    fc_signal: float,
) -> float:
    """Masking spread in dB as a function of frequency distance.

    Simplified roex(p) spreading function (Moore & Glasberg 1997):
    - Upper skirt (signal above masker): −24 dB/ERB
    - Lower skirt (signal below masker): −10 dB/ERB (upward masking is asymmetric)

    Returns the attenuation in dB of masking effect at *fc_signal*
    due to a masker at *fc_masker*.  A return of 0 means full masking;
    large negative values mean no masking.
    """
    delta_erb = erb_rate(fc_signal) - erb_rate(fc_masker)

    if abs(delta_erb) < 0.01:
        return 0.0  # same band — full masking
    elif delta_erb > 0:
        # Signal above masker (upward masking): shallower spread
        # Upward masking is stronger — excitation spreads UP (Moore 1997)
        return -10.0 * delta_erb
    else:
        # Signal below masker (downward masking): steeper drop
        return -24.0 * abs(delta_erb)


# ---------------------------------------------------------------------------
# Temporal masking decay
# ---------------------------------------------------------------------------


def _forward_masking_decay_db(dt_ms: float) -> float:
    """Forward masking decay in dB as a function of time after masker offset.

    Exponential decay model (Jesteadt, Bacon & Lehman 1982):
    ΔL = -20 * log10(1 + dt/τ)  where τ ≈ 10 ms for loud maskers.

    Effective range: ~200 ms.  Beyond that, returns -inf (no masking).
    """
    if dt_ms <= 0:
        return 0.0
    if dt_ms > 200.0:
        return -100.0
    tau_ms = 10.0
    return float(-20.0 * np.log10(1.0 + dt_ms / tau_ms))


def _backward_masking_decay_db(dt_ms: float) -> float:
    """Backward masking decay in dB.

    Much shorter than forward: effective range ~20 ms, roughly 1/3
    the strength of forward masking (Moore 2003).
    """
    if dt_ms <= 0:
        return 0.0
    if dt_ms > 20.0:
        return -100.0
    tau_ms = 3.3  # ~1/3 of forward masking τ
    return float(-20.0 * np.log10(1.0 + dt_ms / tau_ms))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ERBMaskingThreshold:
    """Masking threshold at a specific frequency band and time."""

    centre_freq_hz: float
    erb_width_hz: float
    threshold_db: float  # defect level must exceed this to be audible
    masking_type: str  # "simultaneous" | "temporal_forward" | "temporal_backward" | "combined"
    informational_bonus_db: float = 0.0  # extra masking for tonal content


@dataclass
class ERBMaskingResult:
    """Frequency-dependent masking analysis for a single defect event."""

    band_thresholds: list[ERBMaskingThreshold] = field(default_factory=list)
    mean_threshold_db: float = -100.0
    max_threshold_db: float = -100.0
    salience: float = 1.0  # 0.0 = fully masked, 1.0 = fully exposed
    dominant_masking_type: str = "none"


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------


class ERBAuditoryMaskingModel:
    """Frequency-dependent masking model using ERB critical-band filters.

    Replaces the fixed broadband thresholds (-12/-8/-6 dB) in
    PerceptualSalienceEstimator with a psychoacoustically correct model
    that accounts for:

    1.  Frequency-dependent critical bandwidth (narrow at low frequencies,
        wide at high frequencies)
    2.  Asymmetric spreading (upward masking stronger than downward)
    3.  Exponential temporal decay (not step function)
    4.  Informational masking for harmonically structured content
    """

    _N_BANDS = 24  # ERB bands spanning 50 Hz – 16 kHz
    _F_LOW = 50.0
    _F_HIGH = 16000.0
    _CONTEXT_WINDOW_S = 0.4  # ±400 ms context for simultaneous masking
    _FRAME_HOP_S = 0.050  # 50 ms hop for spectral analysis

    # Signal-to-masker ratio for threshold (Moore & Glasberg 1997)
    _SMR_ABSOLUTE_DB = 5.0  # defect must be ≥5 dB above masked threshold to be salient

    def compute_masking_threshold(
        self,
        audio: np.ndarray,
        sr: int,
        defect_start_s: float,
        defect_end_s: float,
        defect_freq_range: tuple[float, float] | None = None,
    ) -> ERBMaskingResult:
        """Compute frequency-dependent masking threshold at a defect location.

        Parameters
        ----------
        audio : np.ndarray
            Mono or stereo audio at native sample rate.
        sr : int
            Sample rate in Hz.
        defect_start_s, defect_end_s : float
            Temporal location of the defect in seconds.
        defect_freq_range : tuple[float, float] | None
            If known, the frequency range of the defect (Hz).
            If None, all ERB bands are evaluated.

        Returns
        -------
        ERBMaskingResult with per-band thresholds and aggregate salience.
        """
        mono = self._to_mono_f64(audio)
        n_samples = len(mono)
        duration_s = n_samples / sr
        nyquist = sr / 2.0

        # ERB centre frequencies
        centres = self._erb_centres(min(self._F_HIGH, nyquist - 1.0))

        # Compute short-time power spectrum around defect and context
        defect_power = self._band_power_at_time(
            mono,
            sr,
            centres,
            defect_start_s,
            defect_end_s,
        )

        # Context: ±400 ms around defect (excluding defect itself)
        ctx_before_start = max(0.0, defect_start_s - self._CONTEXT_WINDOW_S)
        ctx_after_end = min(duration_s, defect_end_s + self._CONTEXT_WINDOW_S)

        ctx_power_before = self._band_power_at_time(
            mono,
            sr,
            centres,
            ctx_before_start,
            defect_start_s,
        )
        ctx_power_after = self._band_power_at_time(
            mono,
            sr,
            centres,
            defect_end_s,
            ctx_after_end,
        )

        # Temporal distances for masking decay
        dt_forward_ms = max(0.0, (defect_start_s - ctx_before_start) * 1000.0 * 0.5)
        dt_backward_ms = max(0.0, (ctx_after_end - defect_end_s) * 1000.0 * 0.5)

        # Tonality estimate for informational masking
        tonality = self._estimate_tonality(mono, sr, defect_start_s, defect_end_s)

        band_thresholds: list[ERBMaskingThreshold] = []
        max_masking_db = -100.0
        dominant_type = "none"

        for i, fc in enumerate(centres):
            if defect_freq_range is not None:
                f_lo, f_hi = defect_freq_range
                ew = erb_hz(fc)
                if fc + 0.5 * ew < f_lo or fc - 0.5 * ew > f_hi:
                    continue  # defect doesn't occupy this band

            ew = erb_hz(fc)

            # Simultaneous masking: spreading from all context bands
            simul_mask_db = -100.0
            for j, fc_ctx in enumerate(centres):
                spread = _spreading_function_db(fc_ctx, fc)
                ctx_level = max(
                    self._power_to_db(ctx_power_before[j]),
                    self._power_to_db(ctx_power_after[j]),
                )
                mask_at_band = ctx_level + spread
                simul_mask_db = max(simul_mask_db, mask_at_band)

            # Forward temporal masking from pre-defect content
            fwd_decay = _forward_masking_decay_db(dt_forward_ms)
            fwd_mask_db = self._power_to_db(ctx_power_before[i]) + fwd_decay

            # Backward temporal masking from post-defect content
            bwd_decay = _backward_masking_decay_db(dt_backward_ms)
            bwd_mask_db = self._power_to_db(ctx_power_after[i]) + bwd_decay

            # Combined threshold: maximum of all masking contributions
            threshold_db = max(simul_mask_db, fwd_mask_db, bwd_mask_db)

            # Informational masking bonus for tonal content (Brungart 2001)
            info_bonus = 0.0
            if tonality > 0.5:
                info_bonus = 3.0 * tonality  # up to +3 dB extra masking
                threshold_db += info_bonus

            # Determine dominant masking type
            if threshold_db == simul_mask_db or (info_bonus > 0 and simul_mask_db >= max(fwd_mask_db, bwd_mask_db)):
                m_type = "simultaneous"
            elif fwd_mask_db >= bwd_mask_db:
                m_type = "temporal_forward"
            else:
                m_type = "temporal_backward"

            if threshold_db > max_masking_db:
                max_masking_db = threshold_db
                dominant_type = m_type

            band_thresholds.append(
                ERBMaskingThreshold(
                    centre_freq_hz=float(fc),
                    erb_width_hz=float(ew),
                    threshold_db=float(np.nan_to_num(threshold_db, nan=-100.0)),
                    masking_type=m_type,
                    informational_bonus_db=float(info_bonus),
                )
            )

        if not band_thresholds:
            return ERBMaskingResult(salience=1.0, dominant_masking_type="none")

        thresholds_arr = np.array([bt.threshold_db for bt in band_thresholds])
        mean_thresh = float(np.mean(thresholds_arr))
        max_thresh = float(np.max(thresholds_arr))

        # Compute aggregate salience
        # Defect is salient if its power exceeds masking threshold + SMR
        defect_levels = []
        for i, bt in enumerate(band_thresholds):
            idx = list(centres).index(bt.centre_freq_hz) if bt.centre_freq_hz in centres else i
            if idx < len(defect_power):
                dl = self._power_to_db(defect_power[idx])
                defect_levels.append(dl - bt.threshold_db)

        if defect_levels:
            # How much defect exceeds masking threshold on average
            excess = np.mean(defect_levels)
            # Map excess to salience: <0 dB → masked, >SMR → fully salient
            salience = float(
                np.clip(
                    (excess + self._SMR_ABSOLUTE_DB) / (2.0 * self._SMR_ABSOLUTE_DB),
                    0.0,
                    1.0,
                )
            )
        else:
            salience = 1.0

        salience = float(np.nan_to_num(salience, nan=0.5))

        result = ERBMaskingResult(
            band_thresholds=band_thresholds,
            mean_threshold_db=float(np.nan_to_num(mean_thresh, nan=-100.0)),
            max_threshold_db=float(np.nan_to_num(max_thresh, nan=-100.0)),
            salience=salience,
            dominant_masking_type=dominant_type,
        )

        logger.debug(
            "ERB masking: %.0f–%.0f Hz, %d bands, mean_thresh=%.1f dB, salience=%.3f, dominant=%s, tonality=%.2f",
            centres[0] if len(centres) > 0 else 0,
            centres[-1] if len(centres) > 0 else 0,
            len(band_thresholds),
            mean_thresh,
            salience,
            dominant_type,
            tonality,
        )

        return result

    # ------------------------------------------------------------------
    # Convenience: salience for broadband defect
    # ------------------------------------------------------------------

    def estimate_salience(
        self,
        audio: np.ndarray,
        sr: int,
        defect_start_s: float,
        defect_end_s: float,
    ) -> float:
        """Quick salience estimate (0.0–1.0) for a broadband defect.

        Convenience wrapper around compute_masking_threshold.
        """
        result = self.compute_masking_threshold(
            audio,
            sr,
            defect_start_s,
            defect_end_s,
        )
        return result.salience

    # ------------------------------------------------------------------
    # Internal: spectral analysis
    # ------------------------------------------------------------------

    def _erb_centres(self, f_max: float) -> np.ndarray:
        """Generate ERB centre frequencies up to *f_max*."""
        f_high = min(self._F_HIGH, f_max)
        n_low = erb_rate(self._F_LOW)
        n_high = erb_rate(f_high)
        n_vals = np.linspace(n_low, n_high, self._N_BANDS)
        return (10.0 ** (n_vals / 21.4) - 1.0) / 0.00437

    def _band_power_at_time(
        self,
        mono: np.ndarray,
        sr: int,
        centres: np.ndarray,
        t_start: float,
        t_end: float,
    ) -> np.ndarray:
        """Compute power in each ERB band for the given time range.

        Returns array of shape (n_bands,) with mean power per band.
        """
        s = max(0, int(t_start * sr))
        e = min(len(mono), int(t_end * sr))
        if e <= s:
            return np.full(len(centres), 1e-15, dtype=np.float64)

        segment = mono[s:e]

        # FFT
        n_fft = max(256, min(8192, len(segment)))
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))

        spectrum = np.abs(np.fft.rfft(segment[:n_fft])) ** 2
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        powers = np.full(len(centres), 1e-15, dtype=np.float64)
        for i, fc in enumerate(centres):
            ew = erb_hz(fc)
            f_lo = max(0.0, fc - 0.5 * ew)
            f_hi = fc + 0.5 * ew
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if np.any(mask):
                powers[i] = max(1e-15, float(np.mean(spectrum[mask])))

        return powers

    def _estimate_tonality(
        self,
        mono: np.ndarray,
        sr: int,
        t_start: float,
        t_end: float,
    ) -> float:
        """Estimate tonality of audio segment (0.0 = noise, 1.0 = pure tone).

        Uses spectral flatness (Wiener entropy) as a proxy.
        Low flatness = tonal, high flatness = noise-like.
        """
        s = max(0, int(t_start * sr))
        e = min(len(mono), int(t_end * sr))
        if e - s < 256:
            return 0.5

        segment = mono[s:e]
        n_fft = min(4096, len(segment))
        spectrum = np.abs(np.fft.rfft(segment[:n_fft])) ** 2 + 1e-15

        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / (arith_mean + 1e-15)

        # Convert flatness to tonality: flat=1 → tonal=0, flat=0 → tonal=1
        tonality = float(np.clip(1.0 - flatness, 0.0, 1.0))
        return float(np.nan_to_num(tonality, nan=0.5))

    @staticmethod
    def _power_to_db(power: float) -> float:
        """Convert power to dB with floor."""
        return float(10.0 * np.log10(max(power, 1e-15)))

    @staticmethod
    def _to_mono_f64(audio: np.ndarray) -> np.ndarray:
        """Convert to mono float64 with NaN/Inf guard.

        Aurik canonical shape is (N, channels) — axis 1 is the channel dimension.
        For (N, 2): shape[0]=N >> shape[1]=2 → mean(axis=1) → N-element mono.
        For (2, N): shape[0]=2 <  shape[1]=N  → mean(axis=0) → N-element mono.
        WRONG: mean(axis=0) on (N,2) returns a 2-element vector → all downstream
        FFT/band-power computations collapse → salience=0.000 on every call.
        """
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 2:
            # Detect Aurik (N, channels) vs. legacy (channels, N):
            # The channel count is always ≤ 2; the sample count is always >> 2.
            arr = arr.mean(axis=1) if arr.shape[0] > arr.shape[1] else arr.mean(axis=0)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Thread-safe singleton (double-checked locking — §3.2)
# ---------------------------------------------------------------------------

_instance: ERBAuditoryMaskingModel | None = None
_lock = threading.Lock()


def get_erb_auditory_masking_model() -> ERBAuditoryMaskingModel:
    """Return thread-safe singleton ERBAuditoryMaskingModel."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ERBAuditoryMaskingModel()
    return _instance
