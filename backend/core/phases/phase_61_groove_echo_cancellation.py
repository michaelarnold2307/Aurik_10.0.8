"""
Phase 61 — Groove Echo Cancellation.

Groove echo (pre-echo) occurs when a loud passage deforms the adjacent groove
wall, creating a ghost image ~1.8 s before (at 33⅓ rpm).  This is fundamentally
different from codec pre-echo (5–35 ms, handled by phase_23).

Algorithm:
1. Detect transient peaks (loud passages)
2. For each peak, compute revolution delay (RPM-dependent: 1.8s/1.35s/0.77s)
3. Template-match the ghost signal at the expected delay
4. Subtract the ghost component using adaptive spectral subtraction
5. Apply spectral gating to remove residual echo energy

Scientific basis: McDermott (2005) "Record Groove Physics";
Cannam (2006) "Echo Removal in Gramophone Recordings".
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np

logger = logging.getLogger(__name__)


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = sig.mean(axis=0 if sig.shape[0] <= 2 else 1).astype(np.float32)
    else:
        _mono = np.asarray(sig, dtype=np.float32).ravel()
    _frame = 480  # 10 ms @ 48 kHz
    _n_frames = len(_mono) // _frame
    if _n_frames == 0:
        return -96.0
    _frames = _mono[: _n_frames * _frame].reshape(_n_frames, _frame)
    _frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(_frames**2, axis=1)) + 1e-10)
    _mask = _frame_rms_db > -50.0
    if not np.any(_mask):
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(_frames[_mask] ** 2)) + 1e-10))


_MIN_GROOVE_ECHO_SCORE: float = 0.10
_REVOLUTION_DELAYS_S: list[float] = [1.8, 1.35, 0.77]  # 33⅓, 45, 78 RPM
_SPECTRAL_SUBTRACTION_FLOOR_DB: float = -40.0


def _apply_groove_echo_mono(
    x_mono: np.ndarray,
    sample_rate: int,
    strength: float,
    defect_scores: dict | None,
    spectral_subtraction_floor_db: float,
) -> np.ndarray:
    """Echo-Cancellation auf einem Mono-Kanal. Interne Hilfsfunktion für §2.51."""
    x = np.asarray(x_mono, dtype=np.float32)
    n = len(x)
    sr = sample_rate
    out = np.copy(x)

    # Find transient peaks (top 5% of envelope)
    win = max(1, int(0.010 * sr))
    envelope = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    peak_thresh = float(np.percentile(envelope, 95))
    peak_indices = np.where(envelope > peak_thresh)[0]

    # Deduplicate peaks (>500 ms apart)
    min_gap = int(0.5 * sr)
    deduped: list[int] = []
    if len(peak_indices) > 0:
        deduped = [int(peak_indices[0])]
        for p in peak_indices[1:]:
            if int(p) - deduped[-1] > min_gap:
                deduped.append(int(p))
    peaks = deduped[:30]

    if not peaks:
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    for delay_s in _REVOLUTION_DELAYS_S:
        delay_samples = int(delay_s * sr)
        search_window = int(0.05 * sr)

        for peak_idx in peaks:
            ghost_center = peak_idx - delay_samples
            if ghost_center < search_window or ghost_center + search_window >= n:
                continue

            template_len = int(0.03 * sr)
            t_start = peak_idx
            t_end = min(n, t_start + template_len)
            template = x[t_start:t_end]
            if len(template) < 64:
                continue

            g_start = max(0, ghost_center - search_window)
            g_end = min(n, ghost_center + search_window + len(template))
            ghost_region = x[g_start:g_end]

            if len(ghost_region) < len(template):
                continue

            corr = np.correlate(ghost_region, template, mode="valid")
            if len(corr) == 0:
                continue
            best_offset = int(np.argmax(np.abs(corr)))
            best_corr = float(np.abs(corr[best_offset]))
            template_energy = float(np.sum(template**2))
            if template_energy < 1e-12:
                continue
            norm_corr = best_corr / (np.sqrt(template_energy * np.sum(ghost_region**2)) + 1e-12)

            if norm_corr < 0.15:
                continue

            ghost_start = g_start + best_offset
            ghost_end = min(n, ghost_start + len(template))
            ghost_len = ghost_end - ghost_start

            if ghost_len < 32:
                continue

            alpha = float(np.clip(strength * norm_corr * 1.5, 0.0, 0.8))
            ghost = x[ghost_start:ghost_end]
            ref = template[:ghost_len]

            n_fft = min(2048, ghost_len)
            spec_ghost = np.fft.rfft(ghost[:n_fft])
            spec_ref = np.fft.rfft(ref[:n_fft])

            mag_ghost = np.abs(spec_ghost)
            mag_ref = np.abs(spec_ref)
            ratio = mag_ghost / (mag_ref + 1e-12)
            scale = float(np.median(ratio[ratio < np.percentile(ratio, 80)]))
            scale = float(np.clip(scale, 0.01, 0.5))

            spec_clean = spec_ghost - alpha * scale * spec_ref
            floor = 10 ** (spectral_subtraction_floor_db / 20.0)
            mag_clean = np.maximum(floor, np.abs(spec_clean))
            spec_clean = mag_clean * np.exp(1j * np.angle(spec_ghost))

            cleaned = np.fft.irfft(spec_clean, n=n_fft)
            out[ghost_start : ghost_start + n_fft] = cleaned

    result = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.6,
    defect_scores: dict | None = None,
    min_groove_echo_score: float = _MIN_GROOVE_ECHO_SCORE,
    spectral_subtraction_floor_db: float = _SPECTRAL_SUBTRACTION_FLOOR_DB,
) -> np.ndarray:
    """Main entry point for Phase 61."""
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if defect_scores is not None:
        ge_score = float(defect_scores.get("groove_echo", 0.0))
        if ge_score < min_groove_echo_score:
            logger.debug("Phase 61: groove_echo score %.3f < %.3f — skipped", ge_score, min_groove_echo_score)
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 M/S-Domain: Groove-Echo ist ein physikalisches Nadelphänomen,
        # das symmetrisch in beiden Kanälen auftritt → im Mid-Kanal konzentriert.
        # Verarbeitung: Echo-Cancellation auf Mid; Side mit 30 % Stärke (korreliert).
        if audio.shape[0] == 2 and audio.shape[1] != 2:
            left = audio[0].astype(np.float32)
            right = audio[1].astype(np.float32)
        else:
            left = audio[:, 0].astype(np.float32)
            right = audio[:, 1].astype(np.float32)
        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        # Echo-Erkennung + Subtraktion auf Mid-Kanal
        mid_clean = _apply_groove_echo_mono(mid, sample_rate, strength, defect_scores, spectral_subtraction_floor_db)
        # Side mit reduzierter Stärke (korreliertes, symmetrisches Phänomen)
        side_clean = _apply_groove_echo_mono(
            side, sample_rate, strength * 0.3, defect_scores, spectral_subtraction_floor_db
        )

        # M/S → L/R rekonstruieren
        left_out = (mid_clean + side_clean).astype(np.float32)
        right_out = (mid_clean - side_clean).astype(np.float32)
        result = np.clip(np.stack([left_out, right_out], axis=0), -1.0, 1.0).astype(np.float32)
        return result

    # Mono-Verarbeitung via Hilfsfunktion
    return _apply_groove_echo_mono(audio, sample_rate, strength, defect_scores, spectral_subtraction_floor_db)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class GrooveEchoCancellationPhase(PhaseInterface):
    """Phase 61: Template-based groove echo (pre-echo) cancellation for vinyl."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_61_groove_echo_cancellation",
            name="Groove Echo Cancellation",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_01"],
            estimated_time_factor=0.06,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.55,
            description=(
                "Template-based groove echo cancellation via spectral subtraction. "
                "Removes pre-echo artifacts (~1.8 s delay at 33⅓ rpm) caused by "
                "adjacent groove deformation on vinyl records."
            ),
        )

    @staticmethod
    def _compute_groove_echo_profile(
        material_key: str, quality_mode: str, restorability_score: float
    ) -> dict[str, float]:
        material = str(material_key or "unknown").strip().lower()
        mode = str(quality_mode or "balanced").strip().lower()
        mode = {"restoration": "balanced", "studio_2026": "maximum"}.get(mode, mode)

        if "vinyl" in material:
            min_score = 0.10
            floor_db = -44.0
        elif "shellac" in material:
            min_score = 0.12
            floor_db = -42.0
        elif any(token in material for token in ("cd_digital", "dat", "flac", "streaming")):
            min_score = 0.18
            floor_db = -32.0
        else:
            min_score = 0.15
            floor_db = -38.0

        rest_norm = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0)) / 100.0
        min_score += (rest_norm - 0.5) * 0.10
        floor_db += (rest_norm - 0.5) * 8.0

        score_off, floor_off = {
            "fast": (0.03, 8.0),
            "balanced": (0.00, 0.0),
            "quality": (-0.02, -6.0),
            "maximum": (-0.04, -10.0),
        }.get(mode, (0.0, 0.0))
        min_score += score_off
        floor_db += floor_off

        return {
            "min_groove_echo_score": float(np.clip(min_score, 0.05, 0.25)),
            "spectral_subtraction_floor_db": float(np.clip(floor_db, -60.0, -20.0)),
        }

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.6,
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
        _profile_61 = self._compute_groove_echo_profile(
            str(kwargs.get("material_type") or kwargs.get("material") or "unknown"),
            str(kwargs.get("quality_mode", "balanced")),
            float(kwargs.get("restorability_score", 50.0)),
        )
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "groove_echo_score": float((_defect_scores or {}).get("groove_echo", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Groove echo cancellation skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_groove_echo_score=_profile_61["min_groove_echo_score"],
            spectral_subtraction_floor_db=_profile_61["spectral_subtraction_floor_db"],
        )
        elapsed = _time.perf_counter() - t0

        # §2.46f NPA-Guard: Early-Reflections (0-50 ms nach Onset) sind Recording-Chain-Signatur
        # und dürfen nicht durch Groove-Echo-Subtraktion entfernt werden (§2.46f Kategorie 3).
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector
            _mono61 = audio.mean(axis=0) if audio.ndim == 2 else audio
            _npa_mask61 = get_natural_performance_detector().detect(
                _mono61, sample_rate
            ).get_protected_mask(len(_mono61), sample_rate)
            if _npa_mask61 is not None and _npa_mask61.any():
                if result_audio.ndim == 2:
                    result_audio[:, _npa_mask61] = audio[:, _npa_mask61]
                else:
                    result_audio[_npa_mask61] = audio[_npa_mask61]
        except Exception as _npa61_exc:
            logger.debug("§2.46f Phase61 NPA-Guard (non-blocking): %s", _npa61_exc)

        # §2.62 Psychoakustischer Masking-Guard: Spektral-Subtraktion entfernt
        # keine Komponenten die vom Musiksignal maskiert werden (G_floor ≥ 0.10).
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp
            result_audio = apply_psychoacoustic_masking_clamp(
                audio, result_audio, sample_rate, mode="restoration"
            )
        except Exception as _pmask61_exc:
            logger.debug("§2.62 Phase61 Masking-Guard (non-blocking): %s", _pmask61_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0

        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={"groove_echo_score": float((_defect_scores or {}).get("groove_echo", 0.0)), "strength": strength},
            metadata={
                "groove_echo_profile": dict(_profile_61),
                "min_groove_echo_score": float(_profile_61["min_groove_echo_score"]),
                "spectral_subtraction_floor_db": float(_profile_61["spectral_subtraction_floor_db"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,  # Targeted, kein Makeup-Gain nötig
                "strength": strength,
            },
        )
