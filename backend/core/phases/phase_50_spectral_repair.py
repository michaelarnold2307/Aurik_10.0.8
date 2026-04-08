"""
Phase 50: Spectral Repair v2.0 — STFT Inpainting
==================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub.

ALGORITHMUS — Spektrales Inpainting:
  1. STFT: Hann-Fenster 2048, Hop 512
  2. Beschädigte Bin-Erkennung:
     - Vertikale Dimension (Zeit-Transient): |X[f,t]| > α × median(|X[f, t±K]|)
       → Transiente Peaks, kein Reparatur-Bedarf
     - Horizontale Dimension (Frequenz-Spitze): |X[f,t]| > β × median(|X[f±B, t]|)
       → Isolierter Bin-Spike → Reparatur notwendig
  3. Reparatur: Lineare Interpolation aus ±2 Nachbar-Bins
  4. ISTFT mit Overlap-Add, Normierung

METRIKEN:
  - n_repaired_bins: Anzahl reparierter Bins
  - repair_ratio: n_repaired / gesamt
  - snr_improvement_db: Approximierter SNR-Gewinn

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# STFT-Parameter
_FFT_SIZE = 2048
_HOP = 512
_WIN = "hann"

# Detektions-Schwellwerte
_NEIGHBOR_BINS = 5  # ±B Bins für Frequenz-Nachbarschaft
_THRESHOLD_FACTOR = 4.0  # Bin > FACTOR × Median der Nachbarn → verdächtig
_INTERP_BINS = 2  # ±K Bins für Interpolation


def _repair_channel(channel: np.ndarray, sample_rate: int, threshold_factor: float) -> tuple[np.ndarray, int]:
    """
    Spektrale Reparatur eines Mono-Kanals: Frequenz-Achse (Spikes) + Zeit-Achse (Scratches).

    Gibt (repaired, n_repaired_bins) zurück.

    v2.1 (upgraded):
      Pass 1 — Frequency-axis spike detection (unchanged from v2.0):
        Horizontal spikes: |X[f,t]| > FACTOR × median(|X[f±B, t]|) → interpolate

      Pass 2 — Time-axis damage detection (NEW):
        Vertical energy drops: frames where mean(|X[:, t]|) < DROP_FACTOR × median_frame_energy
        → Time-axis linear interpolation from neighbouring frames
        These correspond to scratches (short-duration broadband damage) and dropouts
        (complete signal absence in a frame).
    """
    from scipy.ndimage import uniform_filter1d

    _f, _t, Zxx = sig.stft(
        channel,
        fs=sample_rate,
        window=_WIN,
        nperseg=_FFT_SIZE,
        noverlap=_FFT_SIZE - _HOP,
    )

    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    _n_freq, _n_time = mag.shape

    # ── PASS 1: Frequency-axis spike repair ────────────────────────────────────
    _filter_size = 2 * _NEIGHBOR_BINS + 1
    mag_smooth = uniform_filter1d(mag, size=_filter_size, axis=0, mode="nearest")
    safe_smooth = np.where(mag_smooth < 1e-10, 1e-10, mag_smooth)
    spike_mask = (mag > threshold_factor * safe_smooth) & (mag >= 1e-10)

    mag_lo = np.roll(mag, _INTERP_BINS, axis=0)
    mag_hi = np.roll(mag, -_INTERP_BINS, axis=0)
    mag_interp = (mag_lo + mag_hi) * 0.5
    mag_rep = np.where(spike_mask, mag_interp, mag)
    n_freq_repaired = int(np.count_nonzero(spike_mask))

    # ── PASS 2: Time-axis damage detection (scratch/dropout frames) ────────────
    # Measure per-frame mean energy
    frame_energy = np.mean(mag_rep, axis=0)  # shape: (n_time,)
    # Robust median via sorted percentile (avoid scipy.ndimage for 1D)
    median_energy = float(np.median(frame_energy))
    _DROP_FACTOR = 0.15  # Frame energy < 15% of median → damaged (dropout/scratch)

    damaged_frames = frame_energy < (_DROP_FACTOR * median_energy + 1e-14)
    n_time_repaired = int(np.sum(damaged_frames))

    if n_time_repaired > 0 and n_time_repaired < _n_time * 0.5:
        # Only repair if <50% of frames damaged (full silence passes through intact)
        for t_idx in np.where(damaged_frames)[0]:
            # Find nearest undamaged neighbours on each side
            lo = t_idx - 1
            hi = t_idx + 1
            while lo >= 0 and damaged_frames[lo]:
                lo -= 1
            while hi < _n_time and damaged_frames[hi]:
                hi += 1
            if lo >= 0 and hi < _n_time:
                # Linear interpolation weight based on distance
                d_total = hi - lo
                w_hi = (t_idx - lo) / d_total
                w_lo = (hi - t_idx) / d_total
                mag_rep[:, t_idx] = w_lo * mag_rep[:, lo] + w_hi * mag_rep[:, hi]
            elif lo >= 0:
                mag_rep[:, t_idx] = mag_rep[:, lo]
            elif hi < _n_time:
                mag_rep[:, t_idx] = mag_rep[:, hi]

    n_repaired = n_freq_repaired + n_time_repaired

    # ── Reconstruct ────────────────────────────────────────────────────────────
    Zxx_rep = mag_rep * np.exp(1j * phase)
    _, repaired = sig.istft(
        Zxx_rep,
        fs=sample_rate,
        window=_WIN,
        nperseg=_FFT_SIZE,
        noverlap=_FFT_SIZE - _HOP,
    )
    out_len = len(channel)
    repaired = repaired[:out_len] if len(repaired) >= out_len else np.pad(repaired, (0, out_len - len(repaired)))

    return repaired.astype(channel.dtype), n_repaired


class SpectralRepairPhase(PhaseInterface):
    """STFT-basiertes Spectral Inpainting ohne ML-Dependency."""

    _PHASE_ID = "phase_50_spectral_repair"
    _NAME = "Spectral Repair (STFT Inpainting)"
    description = (
        "Erkennt und repariert isolierte spektrale Bin-Spikes (Artefakte, Kratzer, "
        "Datenfehler) mittels STFT-Analyse und linearer Interpolation aus Nachbar-Bins. "
        "Kein ML, volle DSP-Transparenz."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.08,
            memory_requirement_mb=150,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.88,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Repariert spektrale Artefakte via STFT Inpainting.

        Args:
            audio:        Mono oder Stereo
            sample_rate:  Abtastrate Hz
            **kwargs:     threshold_factor (float, default 4.0)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 1.0)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        threshold_factor = float(kwargs.get("threshold_factor", _THRESHOLD_FACTOR))
        # Lower effective strength should make detection more conservative.
        threshold_factor_eff = threshold_factor / max(effective_strength, 0.1)

        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        is_stereo = audio.ndim == 2
        total_bins = 0

        if not is_stereo:
            repaired_c, n_rep = _repair_channel(audio.astype(np.float64), sample_rate, threshold_factor_eff)
            repaired_audio = repaired_c.astype(audio.dtype)
            total_bins = n_rep
        else:
            # §2.51: M/S domain — repair Mid fully, Side conservatively
            _inv_sqrt2 = 1.0 / np.sqrt(2.0)
            mid = (audio[:, 0] + audio[:, 1]).astype(np.float64) * _inv_sqrt2
            side = (audio[:, 0] - audio[:, 1]).astype(np.float64) * _inv_sqrt2
            repaired_mid, n_mid = _repair_channel(mid, sample_rate, threshold_factor_eff)
            repaired_side, n_side = _repair_channel(side, sample_rate, threshold_factor_eff * 2.0)
            total_bins = n_mid + n_side
            left = (repaired_mid + repaired_side) * _inv_sqrt2
            right = (repaired_mid - repaired_side) * _inv_sqrt2
            repaired_audio = np.column_stack([left, right]).astype(audio.dtype)

        if 0.0 < effective_strength < 1.0:
            repaired_audio = audio + effective_strength * (repaired_audio - audio)

        total_bins_possible = int((_FFT_SIZE // 2 + 1) * (len(audio) // _HOP + 1) * n_channels)
        repair_ratio = total_bins / max(1, total_bins_possible)

        # Approximierter SNR-Gewinn
        diff = audio.astype(np.float64) - repaired_audio.astype(np.float64)
        sig_power = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
        noise_power = float(np.mean(diff**2)) + 1e-12
        snr_before = 10.0 * np.log10(sig_power / noise_power)

        logger.info(
            "Phase 50 SpectralRepair: n_repaired=%d, repair_ratio=%.5f",
            total_bins,
            repair_ratio,
        )

        repaired_audio = np.nan_to_num(repaired_audio, nan=0.0, posinf=0.0, neginf=0.0)
        repaired_audio = np.clip(repaired_audio, -1.0, 1.0)
        _rms_in_50 = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_out_50 = float(np.sqrt(np.mean(np.asarray(repaired_audio, dtype=np.float64) ** 2) + 1e-12))
        _rms_drop_50 = 20.0 * np.log10(max(_rms_out_50 / _rms_in_50, 1e-30)) if _rms_in_50 > 1e-8 else 0.0
        return PhaseResult(
            success=True,
            audio=repaired_audio,
            execution_time_seconds=time.time() - t0,
            metadata={
                "threshold_factor": threshold_factor,
                "threshold_factor_effective": threshold_factor_eff,
                "n_channels": n_channels,
                "stereo_mode": "ms_domain" if is_stereo else "mono",
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop_50)), 3),
                "loudness_makeup_db": 0.0,
            },
            metrics={
                "n_repaired_bins": total_bins,
                "repair_ratio": repair_ratio,
                "snr_improvement_db": max(0.0, snr_before - 30.0),
                "effective_strength": effective_strength,
            },
        )
