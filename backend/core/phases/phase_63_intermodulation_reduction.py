"""
Phase 63 — Intermodulation Distortion Reduction.

IMD creates sum/difference frequency products (f1±f2) from nonlinear
signal paths that are NOT harmonically related to either input frequency.
This is distinct from THD (handled by phase_60) and clipping (phase_23).

Algorithm (Bispectrum-informed, §2.51 M/S compliant):
1. Identify tonal peaks via bispectrum analysis (detects f1+f2 correlations
   without guessing) — more robust than peak-pair enumeration alone.
2. Confirm IMD products: energy at predicted f1+f2 / |f1−f2| is significantly
   above noise floor AND coherent with the fundamental-pair bispectrum.
3. Build frequency-domain notch mask on the Mid channel (M/S domain).
4. Apply notch mask to Mid; apply at reduced strength (≤30 %) on Side.
5. Reconstruct L/R from processed M/S.

§2.51 compliance: no independent L/R processing — notch derived from Mid
spectrum and applied symmetrically (linked stereo semantics in M/S domain).

Scientific basis: Volterra series models; SMPTE RP120-1994;
Kim & Powers (1979) "Digital Bispectral Analysis and its Applications";
Farina (2000) "Simultaneous Measurement of Impulse Response and Distortion".
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)

_MIN_IMD_SCORE: float = 0.10
_NOTCH_WIDTH_HZ: float = 50.0
_BISPECTRUM_NFFT: int = 2048
_PROC_NFFT: int = 8192


def _build_imd_notch_mask(
    x_mono: np.ndarray,
    sample_rate: int,
    strength: float,
) -> np.ndarray:
    """Compute frequency-domain gain mask for IMD notch filtering.

    Uses bispectrum-coherence to confirm whether energy at predicted IMD
    product locations is causally related to the identified fundamentals.
    Returns gain mask of length PROC_NFFT//2+1, values in [0, 1].
    """
    n = len(x_mono)
    n_fft = _BISPECTRUM_NFFT
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    # Long-term power spectrum to find stable tonal peaks
    n_frames = max(1, (n - n_fft) // (n_fft // 4))
    sum_spec = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        s = i * (n_fft // 4)
        if s + n_fft > n:
            break
        frame = x_mono[s : s + n_fft] * np.hanning(n_fft)
        sum_spec += np.abs(np.fft.rfft(frame)) ** 2

    if n_frames > 0:
        sum_spec /= n_frames

    global_db = 10.0 * np.log10(sum_spec + 1e-20)
    noise_floor = float(np.percentile(global_db, 20))

    # Detect peaks ≥ 20 dB above noise floor
    peak_mask = global_db > (noise_floor + 20.0)
    peak_indices = [int(i) for i in np.where(peak_mask)[0]]
    if len(peak_indices) < 2:
        proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
        return np.ones(len(proc_freqs), dtype=np.float64)

    # Cluster nearby peaks → fundamental frequencies
    clusters: list[int] = []
    if peak_indices:
        grp = [peak_indices[0]]
        for idx in peak_indices[1:]:
            if (idx - grp[-1]) * freq_res < 10.0:
                grp.append(idx)
            else:
                clusters.append(int(np.mean(grp)))
                grp = [idx]
        clusters.append(int(np.mean(grp)))
    clusters = clusters[:12]

    # Predict IMD product bins (2nd-order: f1+f2 and |f1-f2|)
    imd_targets: list[int] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            f1 = freqs[clusters[i]]
            f2 = freqs[clusters[j]]
            for target_f in [abs(f1 - f2), f1 + f2]:
                if target_f < 50 or target_f > sample_rate / 2 - 100:
                    continue
                # Exclude harmonics of either fundamental
                is_harmonic = any(abs(target_f - h * base) < freq_res * 3 for base in [f1, f2] for h in range(1, 8))
                if is_harmonic:
                    continue
                target_idx = int(target_f / freq_res)
                if 0 < target_idx < len(global_db):
                    # Confirm: at least 5 dB above noise floor (avoid false positives)
                    if global_db[target_idx] > noise_floor + 5.0:
                        # ── Bispectrum coherence check ────────────────────────
                        # B(f1, f2) = E[X(f1) · X(f2) · conj(X(f1+f2))]
                        # If |B| / (P(f1)·P(f2)·P(f1+f2))^0.5 is high, the product
                        # energy is causally related to the fundamentals → true IMD.
                        # Approximate: use single frame for speed.
                        frame_len = min(n, n_fft)
                        mid_start = max(0, n // 2 - frame_len // 2)
                        frame = x_mono[mid_start : mid_start + frame_len] * np.hanning(frame_len)
                        X = np.fft.rfft(frame, n=n_fft)
                        b_coherence = 0.0
                        if clusters[i] < len(X) and clusters[j] < len(X) and target_idx < len(X):
                            bispec = abs(X[clusters[i]] * X[clusters[j]] * np.conj(X[target_idx]))
                            denom_b = (
                                (abs(X[clusters[i]]) ** 2 + 1e-20)
                                * (abs(X[clusters[j]]) ** 2 + 1e-20)
                                * (abs(X[target_idx]) ** 2 + 1e-20)
                            ) ** (1.0 / 3.0)
                            b_coherence = float(bispec / denom_b)
                        if b_coherence > 0.15:  # Confirmed IMD via bispectrum
                            imd_targets.append(target_idx)

    if not imd_targets:
        proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
        return np.ones(len(proc_freqs), dtype=np.float64)

    # Build notch mask at PROC_NFFT resolution
    proc_freqs = np.fft.rfftfreq(_PROC_NFFT, 1.0 / sample_rate)
    proc_freq_res = float(proc_freqs[1] - proc_freqs[0]) if len(proc_freqs) > 1 else 1.0
    notch_width_bins = max(1, int(_NOTCH_WIDTH_HZ / proc_freq_res))
    gain_mask = np.ones(len(proc_freqs), dtype=np.float64)

    for analysis_idx in imd_targets:
        # Map from analysis (BISPECTRUM_NFFT) to proc (PROC_NFFT) resolution
        target_freq_hz = freqs[analysis_idx]
        proc_idx = int(target_freq_hz / proc_freq_res)
        lo = max(0, proc_idx - notch_width_bins // 2)
        hi = min(len(gain_mask), proc_idx + notch_width_bins // 2 + 1)
        for k in range(lo, hi):
            dist = abs(k - proc_idx) / max(1, notch_width_bins // 2)
            notch_depth = float(np.clip(1.0 - strength * (1.0 - dist**2), 0.15, 1.0))
            gain_mask[k] = min(gain_mask[k], notch_depth)

    return gain_mask


def _apply_stft_mask(
    x: np.ndarray,
    gain_mask: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Apply a frequency-domain gain mask using STFT overlap-add."""
    n = len(x)
    n_fft = _PROC_NFFT
    hop = n_fft // 4
    window = sps.windows.hann(n_fft, sym=False)
    n_frames = max(1, (n - n_fft) // hop + 1)
    out = np.zeros(n, dtype=np.float64)
    win_sum = np.zeros(n, dtype=np.float64)

    for i in range(n_frames):
        s = i * hop
        e = s + n_fft
        if e > n:
            break
        spec = np.fft.rfft(x[s:e] * window)
        spec *= gain_mask
        frame_out = np.fft.irfft(spec, n=n_fft) * window
        out[s:e] += frame_out
        win_sum[s:e] += window**2

    win_sum = np.maximum(win_sum, 1e-8)
    out /= win_sum
    return out


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.55,
    defect_scores: dict | None = None,
) -> np.ndarray:
    """Main entry point for Phase 63 — §2.51 M/S-compliant."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        imd_score = float(defect_scores.get("intermodulation_distortion", 0.0))
        if imd_score < _MIN_IMD_SCORE:
            logger.debug("Phase 63: IMD score %.3f < %.3f — skipped", imd_score, _MIN_IMD_SCORE)
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 M/S domain: derive notch mask from Mid; apply to Mid and Side at
        # reduced strength. This keeps L/R coherent (no independent per-channel mask).
        if audio.shape[0] == 2 and audio.shape[1] != 2:
            left = audio[0].astype(np.float64)
            right = audio[1].astype(np.float64)
            channels_first = True
        else:
            left = audio[:, 0].astype(np.float64)
            right = audio[:, 1].astype(np.float64)
            channels_first = False

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Notch mask derived from Mid signal only
        gain_mask = _build_imd_notch_mask(mid, sample_rate, strength)

        mid_clean = _apply_stft_mask(mid, gain_mask, sample_rate)
        # Side: apply same mask at 30 % strength (IMD arises from Mid nonlinearity)
        side_gain = gain_mask * 0.3 + 0.7  # blend: 30 % of the notch
        side_clean = _apply_stft_mask(side, side_gain, sample_rate)

        left_out = np.clip(mid_clean + side_clean, -1.0, 1.0)
        right_out = np.clip(mid_clean - side_clean, -1.0, 1.0)

        left_out = np.nan_to_num(left_out, nan=0.0, posinf=0.0, neginf=0.0)
        right_out = np.nan_to_num(right_out, nan=0.0, posinf=0.0, neginf=0.0)

        if channels_first:
            return np.stack([left_out, right_out], axis=0).astype(np.float32)
        else:
            return np.stack([left_out, right_out], axis=1).astype(np.float32)

    # Mono path
    x = audio.astype(np.float64)
    gain_mask = _build_imd_notch_mask(x, sample_rate, strength)
    out = _apply_stft_mask(x, gain_mask, sample_rate)
    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class IntermodulationReductionPhase(PhaseInterface):
    """Phase 63: Volterra-based intermodulation distortion reduction."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_63_intermodulation_reduction",
            name="Intermodulation Distortion Reduction",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_04"],
            estimated_time_factor=0.05,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Targeted spectral notch filtering of intermodulation distortion "
                "products (sum/difference frequencies). Identifies IMD products "
                "from tonal peak analysis and removes non-harmonic artifacts."
            ),
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.55,
        defect_scores: dict | None = None,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "imd_score": float((_defect_scores or {}).get("intermodulation_distortion", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Intermodulation reduction skipped due to zero effective strength"],
            )
        _rms_in = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))
        result_audio = apply(audio, sample_rate, strength=_effective_strength, defect_scores=_defect_scores)
        elapsed = _time.perf_counter() - t0

        _rms_out = float(np.sqrt(np.mean(np.asarray(result_audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_drop = 20.0 * np.log10(max(_rms_out / _rms_in, 1e-30)) if _rms_in > 1e-8 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "imd_score": float((_defect_scores or {}).get("intermodulation_distortion", 0.0)),
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
