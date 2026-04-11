"""
core/phase_defect_verifier.py
Phase Defect Verifier (PDV) — Ursache 7: Post-Phase Defekt-Verifikation
========================================================================

After each restorative phase runs, verifies that the phase did not measurably
WORSEN the defect type(s) it was selected to fix.

Rationale (§2.45, §0 Primum non nocere):
  PMGG guards musical goals (natürlichkeit, timbre, etc.) and AFG guards
  artifact freedom, but neither checks whether the targeted physical defect
  (e.g. crackle, hum, DC offset) actually improved.  A phase can pass both
  gates while its targeted defect score increases — this is a silent
  quality regression.

Design:
  - Uses lightweight DSP proxies only (< 5 ms per defect type at 48 kHz).
  - No ML, no full DefectScanner.scan().
  - Advisory + conditional rollback: rolls back ONLY when a proxy worsens
    by more than _ROLLBACK_THRESHOLD (25 %) relative to the pre-phase value.
  - All results stored in a per-session telemetry dict for metadata export.

Proxies implemented (lower = better defect level, unless noted):
  CLICKS / CRACKLE        → impulse-ratio: p99.9(|audio|) / RMS
  HIGH_FREQ_NOISE         → HF noise floor: 5th-pct short-time energy 4–16 kHz
  HUM                     → hum energy at 50/60 Hz harmonics (up to 7th)
  LOW_FREQ_RUMBLE         → energy below 80 Hz
  DC_OFFSET               → abs(mean(audio))
  DROPOUTS                → fraction of 10-ms frames with RMS < –60 dBFS
  PHASE_ISSUES            → mid/side imbalance as mono-compat ratio (higher = better)

DefectTypes without a fast proxy (WOW, FLUTTER, PITCH_DRIFT, CLIPPING,
BANDWIDTH_LOSS, COMPRESSION_ARTIFACTS, …) are silently skipped.

Thread-safety: per-session telemetry is protected by a reentrant lock;
    the verifier singleton is double-checked-locking.

Author: Aurik Development Team
Version: 1.0.0 (v9.11.1 — Ursache 7)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rollback threshold: if proxy_after > proxy_before * (1 + threshold), rollback.
# 25 % relative increase = the phase measurably WORSENED the targeted defect.
# ---------------------------------------------------------------------------
_ROLLBACK_THRESHOLD: float = 0.25

# For PHASE_ISSUES proxy (mono_compat): rollback if ratio DROPS by > threshold.
# (mono_compat: higher = better, so worsening = decrease)
_ROLLBACK_THRESHOLD_COMPAT: float = 0.20


# ---------------------------------------------------------------------------
# Fast proxy implementations (all operate on mono mix if stereo)
# ---------------------------------------------------------------------------


def _as_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        return np.mean(arr, axis=0)
    return arr


def _proxy_impulse_ratio(audio: np.ndarray) -> float:
    """99.9th-pct amplitude / RMS — click/crackle indicator. Lower = better."""
    try:
        mono = _as_mono(audio)
        rms = float(np.sqrt(np.mean(mono**2)) + 1e-9)
        p999 = float(np.percentile(np.abs(mono), 99.9))
        return float(p999 / rms)
    except Exception:
        return 1.0


def _proxy_hf_noise_floor(audio: np.ndarray, sr: int) -> float:
    """5th-percentile short-time energy in the 4–16 kHz band. Lower = better."""
    try:
        mono = _as_mono(audio)
        n_fft: int = 2048
        hop: int = 1024
        bin_4k = max(int(4000 * n_fft / sr), 1)
        bin_16k = min(int(16000 * n_fft / sr), n_fft // 2 + 1)
        if bin_16k <= bin_4k:
            return 0.0
        frame_energies: list[float] = []
        for i in range(0, max(0, len(mono) - n_fft), hop):
            chunk = mono[i : i + n_fft]
            if len(chunk) < n_fft:
                break
            mag = np.abs(np.fft.rfft(chunk * np.hanning(n_fft)))
            frame_energies.append(float(np.mean(mag[bin_4k:bin_16k])))
        if frame_energies:
            return float(np.percentile(frame_energies, 5))
    except Exception:
        pass
    return 0.0


def _proxy_hum_energy(audio: np.ndarray, sr: int) -> float:
    """Sum of energy at 50 Hz and 60 Hz harmonic series (up to 7th). Lower = better."""
    try:
        mono = _as_mono(audio)
        n = len(mono)
        if n < 2:
            return 0.0
        mag = np.abs(np.fft.rfft(mono.astype(np.float32)))
        energy = 0.0
        for f0 in (50.0, 60.0):
            for harmonic in range(1, 8):
                freq = f0 * harmonic
                if freq >= sr / 2:
                    break
                b_center = int(round(freq * n / sr))
                for b in range(max(0, b_center - 1), min(b_center + 2, len(mag))):
                    energy += float(mag[b])
        return float(energy)
    except Exception:
        return 0.0


def _proxy_low_freq_energy(audio: np.ndarray, sr: int) -> float:
    """Mean spectral energy below 80 Hz. Lower = better."""
    try:
        mono = _as_mono(audio)
        n = len(mono)
        mag = np.abs(np.fft.rfft(mono.astype(np.float32)))
        bin_80 = int(80 * n / sr)
        if bin_80 > 0 and bin_80 < len(mag):
            return float(np.mean(mag[:bin_80]))
    except Exception:
        pass
    return 0.0


def _proxy_dc_offset(audio: np.ndarray) -> float:
    """Absolute mean — DC offset indicator. Lower = better."""
    try:
        mono = _as_mono(audio)
        return float(abs(np.mean(mono)))
    except Exception:
        return 0.0


def _proxy_dropout_ratio(audio: np.ndarray, sr: int) -> float:
    """Fraction of 10-ms frames whose RMS < –60 dBFS. Lower = better."""
    try:
        mono = _as_mono(audio)
        frame_len = max(1, int(0.010 * sr))
        n_frames = len(mono) // frame_len
        if n_frames == 0:
            return 0.0
        threshold = 10.0 ** (-60.0 / 20.0)  # ≈ 0.001
        silent = 0
        for i in range(n_frames):
            chunk = mono[i * frame_len : (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            if rms < threshold:
                silent += 1
        return float(silent / n_frames)
    except Exception:
        return 0.0


def _proxy_mono_compat(audio: np.ndarray) -> float:
    """Mid energy / (Mid + Side energy) ratio ∈ [0, 1]. Higher = better mono compat."""
    try:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2:
            return 1.0
        mid = 0.5 * (arr[0] + arr[1])
        side = 0.5 * (arr[0] - arr[1])
        mid_rms = float(np.sqrt(np.mean(mid**2)) + 1e-12)
        side_rms = float(np.sqrt(np.mean(side**2)) + 1e-12)
        return float(np.clip(mid_rms / (mid_rms + side_rms + 1e-12), 0.0, 1.0))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# DefectType name → proxy function dispatch
# (uses string names to avoid circular imports when DefectType changes)
# ---------------------------------------------------------------------------

_PROXY_DISPATCH: dict[str, str] = {
    "CLICKS": "impulse_ratio",
    "CRACKLE": "impulse_ratio",
    "HIGH_FREQ_NOISE": "hf_noise_floor",
    "HUM": "hum_energy",
    "LOW_FREQ_RUMBLE": "low_freq_energy",
    "DC_OFFSET": "dc_offset",
    "DROPOUTS": "dropout_ratio",
    "PHASE_ISSUES": "mono_compat",
}

# Proxies where HIGHER value = better (i.e., worsening = decrease)
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"mono_compat"})


def _compute_proxy(proxy_name: str, audio: np.ndarray, sr: int) -> float:
    if proxy_name == "impulse_ratio":
        return _proxy_impulse_ratio(audio)
    if proxy_name == "hf_noise_floor":
        return _proxy_hf_noise_floor(audio, sr)
    if proxy_name == "hum_energy":
        return _proxy_hum_energy(audio, sr)
    if proxy_name == "low_freq_energy":
        return _proxy_low_freq_energy(audio, sr)
    if proxy_name == "dc_offset":
        return _proxy_dc_offset(audio)
    if proxy_name == "dropout_ratio":
        return _proxy_dropout_ratio(audio, sr)
    if proxy_name == "mono_compat":
        return _proxy_mono_compat(audio)
    return 0.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DefectVerificationResult:
    """Result of one phase's post-defect check."""

    phase_id: str
    """Phase that was verified."""

    targeted_defects: list[str]
    """DefectType names for which a proxy was available."""

    proxies_before: dict[str, float]
    """Proxy values measured before the phase."""

    proxies_after: dict[str, float]
    """Proxy values measured after the phase."""

    worst_defect: str
    """DefectType name with the worst relative change (worsening)."""

    worst_relative_change: float
    """Relative change of the worst defect.
    Positive = worsening (proxy increased for lower-is-better, or decreased for higher).
    """

    rollback_triggered: bool
    """True if current_audio was reverted to pre-phase state."""

    skipped_defects: list[str] = field(default_factory=list)
    """DefectTypes that had no proxy and were skipped."""


# ---------------------------------------------------------------------------
# PhaseDefectVerifier singleton
# ---------------------------------------------------------------------------

_pdv_instance: PhaseDefectVerifier | None = None
_pdv_lock = threading.Lock()


def get_phase_defect_verifier() -> PhaseDefectVerifier:
    """Thread-safe singleton accessor."""
    global _pdv_instance
    if _pdv_instance is None:
        with _pdv_lock:
            if _pdv_instance is None:
                _pdv_instance = PhaseDefectVerifier()
    return _pdv_instance


class PhaseDefectVerifier:
    """Verifies that each restorative phase does not worsen its targeted defects.

    Usage in _execute_pipeline():
        # Before phase:
        _pdv_ref = current_audio  # already available as _afg_phase_input
        # After phase + quality intervention:
        current_audio = get_phase_defect_verifier().check(
            phase_id, _pdv_ref, current_audio, sample_rate, metadata_dict
        )
    """

    def __init__(self) -> None:
        self._session_telemetry: list[dict] = []
        self._telem_lock = threading.RLock()
        self._reverse_map: dict[str, list[str]] | None = None
        self._rmap_lock = threading.Lock()

    def _get_reverse_map(self) -> dict[str, list[str]]:
        """Lazy-load reverse phase→defect-type map (string names, no circular import)."""
        if self._reverse_map is None:
            with self._rmap_lock:
                if self._reverse_map is None:
                    try:
                        from backend.core.defect_phase_mapper import get_reverse_phase_map

                        raw = get_reverse_phase_map()
                        # Convert DefectType enum values to string names
                        self._reverse_map = {pid: [dt.name for dt in dtypes] for pid, dtypes in raw.items()}
                    except Exception as exc:
                        logger.debug("PDV: reverse map load failed: %s", exc)
                        self._reverse_map = {}
        return self._reverse_map or {}

    def measure_proxies(self, phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Compute defect proxies for all targeted defect types of phase_id.

        Returns an empty dict if phase_id is not a restorative phase or if
        audio/sr are invalid.  Never raises.
        """
        try:
            reverse_map = self._get_reverse_map()
            defect_names = reverse_map.get(phase_id, [])
            if not defect_names:
                return {}
            result: dict[str, float] = {}
            for dname in defect_names:
                pname = _PROXY_DISPATCH.get(dname)
                if pname is None:
                    continue
                result[dname] = _compute_proxy(pname, audio, sr)
            return result
        except Exception as exc:
            logger.debug("PDV.measure_proxies(%s) failed: %s", phase_id, exc)
            return {}

    def check(
        self,
        phase_id: str,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        metadata_store: dict | None = None,
    ) -> np.ndarray:
        """Check whether the phase worsened its targeted defects.

        Args:
            phase_id:       Full phase ID (e.g. 'phase_09_crackle_removal').
            audio_before:   Audio BEFORE the phase ran (pre-phase reference).
            audio_after:    Audio AFTER the phase + quality-intervention.
            sr:             Sample rate (must be 48000 in processing context).
            metadata_store: If provided, appended with a 'phase_defect_verification'
                            entry for this phase (non-blocking, best-effort).

        Returns:
            audio_after normally, or audio_before if a rollback was triggered.
            Never raises.
        """
        try:
            proxies_before = self.measure_proxies(phase_id, audio_before, sr)
            if not proxies_before:
                # No proxies available for this phase — skip silently.
                return audio_after

            reverse_map = self._get_reverse_map()
            all_defects = reverse_map.get(phase_id, [])
            skipped: list[str] = [d for d in all_defects if d not in proxies_before]

            proxies_after = self.measure_proxies(phase_id, audio_after, sr)

            worst_defect = ""
            worst_change = 0.0
            rollback = False

            for dname, val_before in proxies_before.items():
                val_after = proxies_after.get(dname, val_before)
                pname = _PROXY_DISPATCH.get(dname, "")

                if pname in _HIGHER_IS_BETTER:
                    # Worsening = decrease
                    if val_before > 1e-9:
                        rel_change = (val_before - val_after) / val_before
                    else:
                        rel_change = 0.0
                    threshold = _ROLLBACK_THRESHOLD_COMPAT
                else:
                    # Worsening = increase
                    if val_before > 1e-9:
                        rel_change = (val_after - val_before) / val_before
                    else:
                        rel_change = 0.0
                    threshold = _ROLLBACK_THRESHOLD

                if rel_change > worst_change:
                    worst_change = rel_change
                    worst_defect = dname

                if rel_change > threshold:
                    rollback = True

            result = DefectVerificationResult(
                phase_id=phase_id,
                targeted_defects=list(proxies_before.keys()),
                proxies_before=proxies_before,
                proxies_after=proxies_after,
                worst_defect=worst_defect,
                worst_relative_change=round(worst_change, 4),
                rollback_triggered=rollback,
                skipped_defects=skipped,
            )

            if rollback:
                logger.warning(
                    "§PDV rollback: %s worsened %s by %.1f%% (pre=%.4f, post=%.4f) — reverting to pre-phase audio",
                    phase_id,
                    worst_defect,
                    worst_change * 100.0,
                    proxies_before.get(worst_defect, 0.0),
                    proxies_after.get(worst_defect, 0.0),
                )
            elif worst_change > 0.0:
                logger.debug(
                    "§PDV: %s minor drift on %s (+%.1f%%) — below rollback threshold, keeping",
                    phase_id,
                    worst_defect,
                    worst_change * 100.0,
                )

            self._record_telemetry(result)

            if metadata_store is not None:
                try:
                    pdv_list = metadata_store.setdefault("phase_defect_verification", [])
                    pdv_list.append(
                        {
                            "phase_id": phase_id,
                            "targeted_defects": result.targeted_defects,
                            "worst_defect": result.worst_defect,
                            "worst_relative_change": result.worst_relative_change,
                            "rollback": result.rollback_triggered,
                        }
                    )
                except Exception:
                    pass

            return audio_before if rollback else audio_after

        except Exception as exc:
            logger.debug("PDV.check(%s) failed: %s — returning audio_after unchanged", phase_id, exc)
            return audio_after

    def _record_telemetry(self, result: DefectVerificationResult) -> None:
        try:
            with self._telem_lock:
                self._session_telemetry.append(
                    {
                        "phase_id": result.phase_id,
                        "worst_defect": result.worst_defect,
                        "worst_relative_change": result.worst_relative_change,
                        "rollback": result.rollback_triggered,
                    }
                )
        except Exception:
            pass

    def get_session_summary(self) -> dict:
        """Return a summary of all PDV checks in this session."""
        try:
            with self._telem_lock:
                entries = list(self._session_telemetry)
            rollbacks = [e for e in entries if e.get("rollback")]
            misses = [e for e in entries if e.get("worst_relative_change", 0.0) > 0.0]
            return {
                "total_checked": len(entries),
                "rollback_count": len(rollbacks),
                "miss_count": len(misses),
                "rollback_phases": [e["phase_id"] for e in rollbacks],
                "miss_phases": [e["phase_id"] for e in misses],
            }
        except Exception:
            return {}

    def reset_session(self) -> None:
        """Reset session telemetry (call before each new restoration run)."""
        with self._telem_lock:
            self._session_telemetry.clear()
