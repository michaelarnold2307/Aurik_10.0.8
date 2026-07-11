"""
Album Consistency Pass — Post-batch timbral and loudness alignment.

After restoring multiple songs from the same album individually, the results
can drift: one song ends up 2 LU louder, another slightly brighter, a third
slightly darker.  A human mastering engineer would catch this in a final
listen-through and apply gentle, transparent corrections.

This module does exactly that — as a final, purely additive pass over the
set of already-restored output files:

1. **Measure** per-song LUFS, spectral tilt (dB/oct), dynamic range
2. **Derive album targets** via median (robust to outliers)
3. **Apply corrections** only to outlier songs (> threshold from median)
   - LUFS correction: simple gain scalar (max ±3 dB)
   - Tilt correction: first-order high-shelf EQ (max ±1.5 dB shelf gain)
4. **Re-write** corrected files in-place + update sidecar metadata

§0 Primum non nocere is central:
  - Songs already within the album median ± threshold are NEVER touched.
  - Max single-song correction: ±3 dB gain, ±1.5 dB shelf.
  - No dynamic range compression, no phase changes.
  - Corrections are always logged in metadata; the pass can be disabled via
    ``min_songs`` (default 3) — a 1- or 2-song "album" makes no sense.

Singleton: get_album_consistency_pass()
Author: Aurik Development Team
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import sosfiltfilt

from backend.file_import import load_audio_file as _load_audio_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (§0: all corrections are bounded and transparent)
# ---------------------------------------------------------------------------

# LUFS: outlier threshold and max correction
_LUFS_OUTLIER_THRESHOLD_LU: float = 2.0  # songs > ±2 LU from median → correct
_LUFS_MAX_CORRECTION_DB: float = 3.0  # hard cap on gain correction

# Spectral tilt: outlier threshold and max correction
_TILT_OUTLIER_THRESHOLD: float = 1.5  # songs > ±1.5 dB/oct from median → correct
_TILT_MAX_CORRECTION_DB: float = 1.5  # shelf gain hard cap (±1.5 dB shelf)
_TILT_SHELF_FREQ_HZ: float = 3000.0  # high-shelf transition frequency

# Minimum number of songs for the pass to make sense
_MIN_SONGS: int = 3

# Peak safety ceiling after all corrections
_PEAK_SAFETY: float = 0.989  # −0.1 dBTP (same as AudioExporter)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SongConsistencyProfile:
    """Per-song measurements and derived corrections."""

    file_path: str
    lufs: float  # measured integrated loudness
    spectral_tilt: float  # dB/octave (log2-regression 200–16 kHz)
    dynamic_range_db: float  # P95peak / gated-RMS in dB

    # Corrections derived after album-median computation (may stay 0.0)
    lufs_correction_db: float = 0.0
    tilt_correction_db: float = 0.0  # signed shelf gain (+ = brighter, - = darker)
    correction_applied: bool = False


@dataclass
class AlbumConsistencyReport:
    """Summary of an album consistency pass."""

    n_songs: int
    album_lufs_median: float
    album_tilt_median: float
    album_dr_median: float
    songs: list[SongConsistencyProfile] = field(default_factory=list)
    corrections_applied: int = 0
    skipped_insufficient_songs: bool = False
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class AlbumConsistencyPass:
    """Post-batch album consistency pass (§ Album-Konsistenz-Pass)."""

    # ---------------------------------------------------------------------------
    # Measurement helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _measure_lufs(audio: np.ndarray, sr: int) -> float:
        """Integrated loudness via BS.1770-4 K-weighting + gating.

        Falls back to pyloudnorm if available, otherwise uses the built-in
        K-weighting implementation from dsp/loudness_matching.py.
        Returns -70.0 for silence.
        """
        try:
            from dsp.loudness_matching import AiLoudnessMatching  # pylint: disable=import-outside-toplevel

            return AiLoudnessMatching().measure_lufs(audio, sr)  # type: ignore[no-any-return]
        except Exception as _exc:
            logger.debug("AiLoudnessMatching unavailable (%s), using fallback LUFS", _exc)
        # Minimal fallback: RMS in dBFS (not gated, but adequate for relative comparison)
        if audio.ndim == 2:
            mono = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        else:
            mono = audio
        rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        if rms < 1e-10:
            return -70.0
        return float(20.0 * np.log10(rms))

    @staticmethod
    def _measure_spectral_tilt(audio: np.ndarray, sr: int) -> float:
        """Spectral tilt in dB/octave via log2-linear regression (200 Hz – 16 kHz).

        Reuses the canonical implementation from era_classifier.py.
        """
        try:
            from backend.core.era_classifier import _estimate_spectral_tilt  # pylint: disable=import-outside-toplevel

            if audio.ndim == 2:
                mono = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
            else:
                mono = audio
            return float(_estimate_spectral_tilt(mono.astype(np.float32), sr))
        except Exception as _exc:
            logger.debug("_estimate_spectral_tilt unavailable (%s), using fallback", _exc)
        # Minimal fallback: ratio of energy above/below 1 kHz
        try:
            n_fft = min(4096, len(audio.ravel()))
            mono = audio.ravel()[:n_fft]
            spec = np.abs(np.fft.rfft(mono * np.hanning(n_fft))) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            lo = spec[(freqs >= 200) & (freqs <= 1000)].mean()
            hi = spec[(freqs > 1000) & (freqs <= 8000)].mean()
            if lo > 1e-20 and hi > 1e-20:
                return float(10.0 * np.log10(hi / lo) / np.log2(8000.0 / 500.0))
        except Exception as e:
            logger.warning("album_consistency.py::_measure_spectral_tilt fallback: %s", e)
        return -4.0  # conservative neutral

    @staticmethod
    def _measure_dynamic_range(audio: np.ndarray) -> float:
        """Dynamic range: 95th-percentile peak vs. gated-RMS (dB)."""
        if audio.ndim == 2:
            mono = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
        else:
            mono = audio
        peak = float(np.percentile(np.abs(mono), 99.5))
        # Gated RMS: frames > -50 dBFS only
        frame = 480
        active = [
            mono[i : i + frame]
            for i in range(0, len(mono) - frame, frame)
            if 20.0 * np.log10(np.sqrt(np.mean(mono[i : i + frame] ** 2)) + 1e-10) > -50.0
        ]
        if not active or peak < 1e-10:
            return 0.0
        rms = float(np.sqrt(np.mean([np.mean(r**2) for r in active])))
        if rms < 1e-10:
            return 0.0
        return float(20.0 * np.log10(peak / (rms + 1e-10)))

    # ---------------------------------------------------------------------------
    # Correction helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
        """Wendet einfachen linearen Gain (dB) an, dann Peak-sicheres Clipping."""
        if abs(gain_db) < 0.01:
            return audio
        g = float(10.0 ** (gain_db / 20.0))
        out = audio * g
        # Peak safety: if clipping risk, reduce gain proportionally
        peak = float(np.percentile(np.abs(out), 99.9))
        if peak > _PEAK_SAFETY:
            out = out * (_PEAK_SAFETY / peak)
        return out.astype(audio.dtype)  # type: ignore[no-any-return]

    @staticmethod
    def _build_shelf_sos(shelf_gain_db: float, shelf_freq_hz: float, sr: int) -> np.ndarray:
        """Build a first-order high-shelf SOS (max ±1.5 dB shelf gain).

        Based on Audio EQ Cookbook (Zölzer 2011):
        High shelf with parametric gain G dB at shelf_freq_hz.
        """
        G = float(np.clip(shelf_gain_db, -_TILT_MAX_CORRECTION_DB, _TILT_MAX_CORRECTION_DB))
        if abs(G) < 0.05:
            # Identity filter
            return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])  # type: ignore[no-any-return]

        A = 10.0 ** (G / 40.0)  # shelf midpoint amplitude
        w0 = 2.0 * np.pi * shelf_freq_hz / sr
        cos_w0 = np.cos(w0)
        alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.9 - 1.0) + 2.0)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])  # type: ignore[no-any-return]

    def _apply_tilt_correction(self, audio: np.ndarray, tilt_correction_db: float, sr: int) -> np.ndarray:
        """Wendet an: high-shelf correction to compensate spectral tilt deviation."""
        if abs(tilt_correction_db) < 0.05:
            return audio
        sos = self._build_shelf_sos(tilt_correction_db, _TILT_SHELF_FREQ_HZ, sr)
        if audio.ndim == 1:
            out = sosfiltfilt(sos, audio.astype(np.float64)).astype(audio.dtype)
        else:
            # Process each channel independently (shelf is a linear amplitude operation)
            if audio.shape[0] <= 2:  # (channels, samples)
                out = np.stack(
                    [sosfiltfilt(sos, audio[c].astype(np.float64)).astype(audio.dtype) for c in range(audio.shape[0])]
                )
            else:  # (samples, channels)
                out = np.stack(
                    [
                        sosfiltfilt(sos, audio[:, c].astype(np.float64)).astype(audio.dtype)
                        for c in range(audio.shape[1])
                    ],
                    axis=1,
                )
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

    # ---------------------------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------------------------

    def analyze(
        self,
        audio_list: list[np.ndarray],
        sr_list: list[int],
        file_paths: list[str],
    ) -> AlbumConsistencyReport:
        """Misst per-song metrics and compute album-level correction targets.

        Args:
            audio_list:  Restored audio arrays (one per song).
            sr_list:     Sample rates (one per song).
            file_paths:  Output file paths (for reporting only).

        Returns:
            AlbumConsistencyReport with per-song SongConsistencyProfile and
            corrections already set (but NOT yet applied to audio).
        """
        n = len(audio_list)
        if n < _MIN_SONGS:
            logger.info(
                "AlbumConsistencyPass: only %d song(s) — minimum %d required, pass skipped.",
                n,
                _MIN_SONGS,
            )
            return AlbumConsistencyReport(
                n_songs=n,
                album_lufs_median=float("nan"),
                album_tilt_median=float("nan"),
                album_dr_median=float("nan"),
                skipped_insufficient_songs=True,
            )

        profiles: list[SongConsistencyProfile] = []
        for audio, sr, fp in zip(audio_list, sr_list, file_paths):
            try:
                lufs = self._measure_lufs(audio, sr)
                tilt = self._measure_spectral_tilt(audio, sr)
                dr = self._measure_dynamic_range(audio)
            except Exception as _exc:
                logger.warning("AlbumConsistencyPass: measurement failed for %s: %s", fp, _exc)
                lufs, tilt, dr = -18.0, -4.0, 12.0
            profiles.append(
                SongConsistencyProfile(
                    file_path=fp,
                    lufs=lufs,
                    spectral_tilt=tilt,
                    dynamic_range_db=dr,
                )
            )

        # Album targets: median of valid measurements
        lufs_vals = np.array([p.lufs for p in profiles if p.lufs > -69.0])
        tilt_vals = np.array([p.spectral_tilt for p in profiles])
        dr_vals = np.array([p.dynamic_range_db for p in profiles if p.dynamic_range_db > 0.0])

        album_lufs = float(np.median(lufs_vals)) if len(lufs_vals) > 0 else -18.0
        album_tilt = float(np.median(tilt_vals)) if len(tilt_vals) > 0 else -4.0
        album_dr = float(np.median(dr_vals)) if len(dr_vals) > 0 else 12.0

        logger.info(
            "AlbumConsistencyPass analysis: %d songs | LUFS median=%.1f | tilt median=%.2f dB/oct | DR median=%.1f dB",
            n,
            album_lufs,
            album_tilt,
            album_dr,
        )

        # Compute per-song corrections
        for p in profiles:
            lufs_dev = p.lufs - album_lufs
            tilt_dev = p.spectral_tilt - album_tilt

            # LUFS: correct outliers (§0: do not over-process, cap at max)
            if abs(lufs_dev) > _LUFS_OUTLIER_THRESHOLD_LU:
                p.lufs_correction_db = float(np.clip(-lufs_dev, -_LUFS_MAX_CORRECTION_DB, _LUFS_MAX_CORRECTION_DB))
                logger.info(
                    "  LUFS outlier: %s | measured=%.1f | deviation=%.1f LU | correction=%.2f dB",
                    Path(p.file_path).name,
                    p.lufs,
                    lufs_dev,
                    p.lufs_correction_db,
                )

            # Tilt: correct outliers (§0: gentle shelf only)
            if abs(tilt_dev) > _TILT_OUTLIER_THRESHOLD:
                # Negative tilt_dev means song is darker than album → positive shelf to brighten
                p.tilt_correction_db = float(
                    np.clip(-tilt_dev * 0.5, -_TILT_MAX_CORRECTION_DB, _TILT_MAX_CORRECTION_DB)
                )
                logger.info(
                    "  Tilt outlier:  %s | measured=%.2f | deviation=%.2f dB/oct | shelf=%.2f dB",
                    Path(p.file_path).name,
                    p.spectral_tilt,
                    tilt_dev,
                    p.tilt_correction_db,
                )

        return AlbumConsistencyReport(
            n_songs=n,
            album_lufs_median=album_lufs,
            album_tilt_median=album_tilt,
            album_dr_median=album_dr,
            songs=profiles,
        )

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        profile: SongConsistencyProfile,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Wendet Album-Korrekturen auf einen einzelnen Song an.

        Returns:
            (corrected_audio, correction_metadata)
        """
        meta: dict[str, Any] = {
            "lufs_before": profile.lufs,
            "tilt_before": profile.spectral_tilt,
            "lufs_correction_db": profile.lufs_correction_db,
            "tilt_correction_db": profile.tilt_correction_db,
        }
        out = audio.copy()

        # 1. Spectral tilt correction first (linear phase shelf)
        if abs(profile.tilt_correction_db) >= 0.05:
            out = self._apply_tilt_correction(out, profile.tilt_correction_db, sr)
            meta["tilt_corrected"] = True

        # 2. LUFS correction (gain scalar)
        if abs(profile.lufs_correction_db) >= 0.05:
            out = self._apply_gain(out, profile.lufs_correction_db)
            meta["lufs_corrected"] = True

        # Final safety clip
        out = np.clip(out, -1.0, 1.0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(audio.dtype)

        profile.correction_applied = (meta.get("tilt_corrected") or meta.get("lufs_corrected")) is True
        meta["correction_applied"] = profile.correction_applied

        # Re-measure to confirm
        if profile.correction_applied:
            meta["lufs_after"] = self._measure_lufs(out, sr)
            meta["tilt_after"] = self._measure_spectral_tilt(out, sr)

        return out, meta

    def process_output_files(
        self,
        output_files: list[str],
        sr: int = 48000,  # pylint: disable=unused-argument
        dry_run: bool = False,
    ) -> AlbumConsistencyReport:
        """Full album consistency pass over a list of already-written output files.

        Reads each file, analyzes, computes corrections, applies them (unless
        dry_run=True), and re-writes in-place.

        Args:
            output_files:  Paths to restored WAV/FLAC files.
            sr:            Expected sample rate (default 48000).
            dry_run:       If True, only measure and report — do not write.

        Returns:
            AlbumConsistencyReport with all corrections documented.
        """
        t0 = time.monotonic()
        audio_list: list[np.ndarray] = []
        sr_list: list[int] = []
        valid_files: list[str] = []

        for fp in output_files:
            try:
                _ld = _load_audio_file(fp)
                if _ld is None or _ld.get("audio") is None:
                    raise ValueError(f"Cannot load output file: {fp}")
                data = _ld["audio"].astype(np.float32)
                file_sr = int(_ld["sr"])
                audio_list.append(data)
                sr_list.append(file_sr)
                valid_files.append(fp)
            except Exception as _exc:
                logger.warning("AlbumConsistencyPass: cannot read %s: %s", fp, _exc)

        if len(valid_files) < _MIN_SONGS:
            logger.info(
                "AlbumConsistencyPass: %d valid output file(s) — minimum %d required, pass skipped.",
                len(valid_files),
                _MIN_SONGS,
            )
            return AlbumConsistencyReport(
                n_songs=len(valid_files),
                album_lufs_median=float("nan"),
                album_tilt_median=float("nan"),
                album_dr_median=float("nan"),
                skipped_insufficient_songs=True,
                elapsed_seconds=time.monotonic() - t0,
            )

        report = self.analyze(audio_list, sr_list, valid_files)
        if report.skipped_insufficient_songs:
            report.elapsed_seconds = time.monotonic() - t0
            return report

        for audio, file_sr, profile in zip(audio_list, sr_list, report.songs):
            needs_correction = abs(profile.lufs_correction_db) >= 0.05 or abs(profile.tilt_correction_db) >= 0.05
            if not needs_correction:
                continue

            corrected, corr_meta = self.apply(audio, file_sr, profile)

            if dry_run:
                logger.info(
                    "AlbumConsistencyPass DRY-RUN: %s | LUFS Δ=%.2f dB | tilt Δ=%.2f dB",
                    Path(profile.file_path).name,
                    profile.lufs_correction_db,
                    profile.tilt_correction_db,
                )
                report.corrections_applied += 1
                continue

            # Write corrected audio back in-place
            try:
                info = sf.info(profile.file_path)
                subtype = info.subtype if info.subtype else "PCM_16"
                sf.write(profile.file_path, corrected, file_sr, subtype=subtype)

                # Sidecar metadata update
                _sidecar_path = Path(profile.file_path).with_suffix(".metadata.json")
                if _sidecar_path.exists():
                    try:
                        with open(_sidecar_path, encoding="utf-8") as _f:
                            _sidecar = json.load(_f)
                    except Exception:
                        _sidecar = {}
                    _sidecar["album_consistency_pass"] = {
                        "album_lufs_median": report.album_lufs_median,
                        "album_tilt_median": report.album_tilt_median,
                        **corr_meta,
                    }
                    with open(_sidecar_path, "w", encoding="utf-8") as _f:
                        json.dump(_sidecar, _f, indent=2)

                report.corrections_applied += 1
                logger.info(
                    "AlbumConsistencyPass: corrected %s | LUFS Δ=%.2f dB | tilt Δ=%.2f dB shelf",
                    Path(profile.file_path).name,
                    profile.lufs_correction_db,
                    profile.tilt_correction_db,
                )
            except Exception as _exc:
                logger.warning(
                    "AlbumConsistencyPass: failed to re-write %s: %s",
                    profile.file_path,
                    _exc,
                )

        report.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "AlbumConsistencyPass complete: %d/%d songs corrected in %.1fs",
            report.corrections_applied,
            report.n_songs,
            report.elapsed_seconds,
        )
        return report


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: AlbumConsistencyPass | None = None
_lock = threading.Lock()


def get_album_consistency_pass() -> AlbumConsistencyPass:
    """Thread-safe singleton accessor (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AlbumConsistencyPass()
    return _instance
