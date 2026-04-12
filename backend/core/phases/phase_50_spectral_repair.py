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


def _repair_channel(
    channel: np.ndarray,
    sample_rate: int,
    threshold_factor: float,
    hf_protected_bin_start: int = 0,
) -> tuple[np.ndarray, int]:
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

    Args:
        channel:               Mono audio (1D float array).
        sample_rate:           Sample rate (Hz).
        threshold_factor:      Spike threshold — magnitude ratio to smooth envelope.
        hf_protected_bin_start: Frequency-bin index above which Pass 1 spike detection is
                                 disabled.  0 = no protection (default).

    §PriorPhase-Guard (phase_07/phase_06 harmonics):
        When a prior phase (phase_07 harmonic restoration, phase_06 SBR) adds content
        at frequencies above the material's natural rolloff, those bins look like
        "isolated spikes" to Pass 1 because the 11-bin smooth envelope is still near
        the noise floor (only 1 bin elevated out of 11).  Pass 1 would flag and remove
        them — reverting the prior restoration.

        Fix: the caller passes hf_protected_bin_start = bin_index(material_rolloff × 0.85).
        Bins above that index are excluded from Pass 1.  Pass 2 (frame energy dropout)
        remains active everywhere: it works on whole-frame RMS, not per-bin ratios, and
        is insensitive to isolated harmonic additions.
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

    # §PriorPhase-Guard: Exclude HF bins above material rolloff from Pass 1.
    # Phase_07 (harmonic restoration) synthesises content at frequencies that were
    # previously near noise floor.  Those isolated peaks trigger the spike_mask
    # (1 bin elevated / 11-bin window → ratio ≈ 11 >> threshold_factor).
    # Pass 2 (frame dropout) is NOT masked — it is based on whole-frame RMS,
    # not per-bin ratio, and does not produce false positives on HF harmonics.
    if hf_protected_bin_start > 0:
        hf_start = min(hf_protected_bin_start, _n_freq)
        spike_mask[hf_start:, :] = False

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
        # ── Initial seed: linear interpolation between nearest undamaged frames ──
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

        # ── STFT-Consistency projection (Siedenburg & Dörfler 2013, JASA) ────────
        # Iteratively enforce time-frequency consistency for the inpainted frames
        # while keeping undamaged frames anchored to their original values.
        # This propagates spectral structure from known frames into gaps, exploiting
        # the redundancy of the STFT frame (each sample contributes to multiple frames).
        _N_CONSISTENCY_ITER = 5
        _known_mask = ~damaged_frames           # (n_time,) — True = undamaged
        _mag_anchor = mag[:, _known_mask].copy()  # original magnitudes for known frames
        _ph_anchor = phase[:, _known_mask].copy()  # original phases for known frames

        for _ci in range(_N_CONSISTENCY_ITER):
            # Reconstruct time-domain signal from current spectrogram estimate
            _Zxx_ci = mag_rep * np.exp(1j * phase)
            try:
                _, _td_ci = sig.istft(
                    _Zxx_ci,
                    fs=sample_rate,
                    window=_WIN,
                    nperseg=_FFT_SIZE,
                    noverlap=_FFT_SIZE - _HOP,
                )
                _n_need = len(channel)
                if len(_td_ci) >= _n_need:
                    _td_ci = _td_ci[:_n_need]
                else:
                    _td_ci = np.pad(_td_ci, (0, _n_need - len(_td_ci)))
                # Re-STFT — updated phase arises naturally from consistent TD signal
                _, _, _Zxx_new = sig.stft(
                    _td_ci,
                    fs=sample_rate,
                    window=_WIN,
                    nperseg=_FFT_SIZE,
                    noverlap=_FFT_SIZE - _HOP,
                )
                _mag_new = np.abs(_Zxx_new)
                _ph_new = np.angle(_Zxx_new)
                # Trim / pad to original STFT dimensions
                if _mag_new.shape[1] < _n_time:
                    _mag_new = np.pad(_mag_new, ((0, 0), (0, _n_time - _mag_new.shape[1])), mode="edge")
                    _ph_new = np.pad(_ph_new, ((0, 0), (0, _n_time - _ph_new.shape[1])), mode="edge")
                mag_rep = _mag_new[:_n_freq, :_n_time]
                phase = _ph_new[:_n_freq, :_n_time]
                # Projection step: re-impose known-frame constraint (POCS)
                mag_rep[:, _known_mask] = _mag_anchor
                phase[:, _known_mask] = _ph_anchor
            except Exception:
                break  # Consistency iteration failed; keep current estimate

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
        "Datenfehler) mittels STFT-Analyse. "
        "Pass 1: Frequenz-Achsen-Spike-Reparatur per Nachbar-Bin-Interpolation. "
        "Pass 2: Zeit-Achsen-Dropout-Reparatur via iterativer STFT-Konsistenz-Projektion "
        "(Siedenburg & Dörfler 2013). "
        "Kein ML, volle DSP-Transparenz."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="2.1.0",
            dependencies=[],
            estimated_time_factor=0.10,
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
        # §2.54 Material-adaptive threshold: degraded analog needs more aggressive repair
        # (lower threshold_factor = more bins detected as holes = more repairs).
        _mat_50 = kwargs.get("material_type") or kwargs.get("material")
        _mat_str_50 = str(_mat_50).lower() if _mat_50 is not None else ""
        _MATERIAL_THRESHOLD_CAPS_50: dict[str, float] = {
            "wax_cylinder": 2.0,
            "shellac": 2.5,
            "optical_film": 2.5,
            "wire_recording": 2.5,
            "vinyl": 3.0,
            "reel_tape": 3.0,
            "tape": 3.0,
            "cassette": 3.0,
            "radio_broadcast": 3.5,
            "mp3_low": 3.5,
            "minidisc": 3.5,
            "mp3_high": 4.5,
            "cd_digital": 4.5,
            "dat": 4.5,
        }
        for _mk50, _mv50 in _MATERIAL_THRESHOLD_CAPS_50.items():
            if _mk50 in _mat_str_50:
                threshold_factor = min(threshold_factor, _mv50)
                break
        # Lower effective strength should make detection more conservative.
        threshold_factor_eff = threshold_factor / max(effective_strength, 0.1)

        # §PriorPhase-Guard: Protect HF bins from Pass-1 false-positive spike detection.
        # Phase_07 (harmonic restoration) and Phase_06 (SBR) synthesise content at
        # frequencies that were near noise floor before they ran.  The 11-bin local-average
        # smooth envelope is still near noise floor at those frequencies → ratio ≫ 4.0 →
        # Pass 1 would flag and inpaint Phase_07/06 output, reverting the restoration.
        # Fix: exclude bins above 85 % of the material's natural rolloff from Pass 1.
        # (All codec-artifact spikes occur WITHIN the natural bandwidth, so Pass 1 still
        # catches them.  Phase_07/06 harmonics are exclusively ABOVE the rolloff.)
        _ANALOG_ROLLOFF_PROTECTION_HZ: dict[str, float] = {
            "wax_cylinder": 5_000.0,
            "wire_recording": 6_000.0,
            "shellac": 7_000.0,
            "lacquer_disc": 8_000.0,
            "cassette": 13_000.0,
            "vinyl": 14_000.0,
            "tape": 14_000.0,
            "reel_tape": 16_000.0,
        }
        _bin_hz = sample_rate / _FFT_SIZE  # Hz per STFT bin
        _rolloff_protection_hz = _ANALOG_ROLLOFF_PROTECTION_HZ.get(_mat_str_50, 0.0)
        # Also check prefix matching for compound material strings
        if _rolloff_protection_hz == 0.0:
            for _k, _v in _ANALOG_ROLLOFF_PROTECTION_HZ.items():
                if _k in _mat_str_50:
                    _rolloff_protection_hz = _v
                    break
        _hf_protected_bin_start: int = 0
        if _rolloff_protection_hz > 0.0:
            _hf_protected_bin_start = max(0, int(_rolloff_protection_hz * 0.85 / _bin_hz))

        n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        is_stereo = audio.ndim == 2
        total_bins = 0

        if not is_stereo:
            repaired_c, n_rep = _repair_channel(
                audio.astype(np.float64), sample_rate, threshold_factor_eff, _hf_protected_bin_start
            )
            repaired_audio = repaired_c.astype(audio.dtype)
            total_bins = n_rep
        else:
            # §2.51: M/S domain — repair Mid fully, Side conservatively
            _inv_sqrt2 = 1.0 / np.sqrt(2.0)
            mid = (audio[:, 0] + audio[:, 1]).astype(np.float64) * _inv_sqrt2
            side = (audio[:, 0] - audio[:, 1]).astype(np.float64) * _inv_sqrt2
            repaired_mid, n_mid = _repair_channel(mid, sample_rate, threshold_factor_eff, _hf_protected_bin_start)
            repaired_side, n_side = _repair_channel(
                side, sample_rate, threshold_factor_eff * 2.0, _hf_protected_bin_start
            )
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

        # §4.5 Psychoacoustic Masking Clamp — only repair where audible (§0 Primum non nocere)
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp
            repaired_audio = apply_psychoacoustic_masking_clamp(
                audio, repaired_audio, sample_rate,
                strength=effective_strength, mode="additive",
            )
        except Exception as _pm_exc:
            logger.debug("Phase50 masking clamp non-blocking: %s", _pm_exc)

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
                "hf_protected_bin_start": _hf_protected_bin_start,
                "hf_protection_rolloff_hz": round(_rolloff_protection_hz, 1),
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
