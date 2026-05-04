"""
Phase 59 — Modulation Noise Reduction.

Signal-dependent modulation noise (unique to analog tape) where noise level
tracks the signal envelope.  Unlike stationary tape hiss (phase_29), modulation
noise requires signal-adaptive noise gating that varies its threshold with the
signal level.

Algorithm (Esquef & Biscainho 2006):
1. Estimate signal envelope (10 ms RMS)
2. Estimate noise floor as a function of signal level
3. Apply signal-dependent spectral gating: G(f) = max(G_floor, 1 - alpha * N(f)/S(f))
   where N(f) is estimated noise at frequency f given current signal level
4. Wet/dry blend with strength parameter

Scientific basis: Esquef & Biscainho (2006), Czyzewski et al. (2020).
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

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
    # Vectorized: reshape → (n_frames, 480), per-frame RMS in one pass — no Python loop
    _frames = _mono[: _n_frames * _frame].reshape(_n_frames, _frame)
    _frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(_frames**2, axis=1)) + 1e-10)
    _mask = _frame_rms_db > -50.0
    if not np.any(_mask):
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(_frames[_mask] ** 2)) + 1e-10))


_MIN_MODULATION_NOISE_SCORE: float = 0.10
_G_FLOOR: float = 0.10  # Minimum spectral gain — §2.62: VERBOTEN < 0.10


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.7,
    defect_scores: dict | None = None,
    min_modulation_noise_score: float = _MIN_MODULATION_NOISE_SCORE,
    g_floor: float = _G_FLOOR,
) -> np.ndarray:
    """Main entry point for Phase 59.

    Args:
        audio:        Input audio float32, mono or stereo
        sample_rate:  Must be 48000 Hz
        strength:     Processing strength 0–1
        defect_scores: Defect scan scores (optional, for gating)

    Returns:
        Processed audio, same shape as input
    """
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Gate: only process if modulation noise is detected
    if defect_scores is not None:
        mn_score = float(defect_scores.get("modulation_noise", 0.0))
        if mn_score < min_modulation_noise_score:
            logger.debug(
                "Phase 59: modulation_noise score %.3f < %.3f — skipped",
                mn_score,
                min_modulation_noise_score,
            )
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: Noise-Modell aus Mid, identischer STFT-Gain auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_denoised = apply(mono_mix, sample_rate, strength=strength, defect_scores=defect_scores)
        _eps_mn = 1e-10
        _gain_mn = np.where(
            np.abs(mono_mix) > _eps_mn,
            mono_denoised / (mono_mix + _eps_mn * np.sign(mono_mix + _eps_mn)),
            1.0,
        )
        _gain_mn = np.clip(_gain_mn, 0.0, 10.0)
        return np.clip(np.stack([audio[0] * _gain_mn, audio[1] * _gain_mn], axis=0), -1.0, 1.0).astype(np.float32)

    x = np.asarray(audio, dtype=np.float32)
    n = len(x)

    # STFT parameters
    n_fft = 2048
    hop = n_fft // 4

    # Vectorized STFT via scipy — replaces Python frame-loop (~21 000 iterations for 225 s)
    _, _, stft = sps.stft(x, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="even")
    stft = stft.astype(np.complex64)

    mag = np.abs(stft).astype(np.float32)
    phase = np.angle(stft).astype(np.float32)

    # Signal envelope (per-frame RMS)
    frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)

    # Estimate noise model: noise_level(f) = alpha * signal_level
    # Use low-energy frames to calibrate the noise/signal ratio
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)

    # Signal-dependent noise estimate per frame
    alpha = float(np.clip(strength * 0.8, 0.1, 0.9))
    noise_estimate = noise_floor * (frame_rms / (np.median(frame_rms) + 1e-12))[np.newaxis, :]

    # Spectral gating: reduce noise proportional to signal level
    gain = np.maximum(g_floor, 1.0 - alpha * noise_estimate / (mag + 1e-12))
    gain = np.clip(gain, g_floor, 1.0).astype(np.float32)

    # §2.62 Psychoakustischer Masking-Guard (ISO 11172-3) — per-Band Floor (non-blocking)
    try:
        from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172 as _cmask_p59
        _mask_ratio_p59 = _cmask_p59(x, sample_rate, n_fft=n_fft, hop_length=hop)
        _mfloor_p59 = np.mean(_mask_ratio_p59, axis=1).astype(np.float32)  # (n_freq,)
        _mfreqs_p59 = np.linspace(0.0, sample_rate / 2.0, _mask_ratio_p59.shape[0], dtype=np.float32)
        _mfloor_interp59 = np.interp(
            np.linspace(0.0, sample_rate / 2.0, gain.shape[0], dtype=np.float32),
            _mfreqs_p59, _mfloor_p59
        ).astype(np.float32)
        gain = np.maximum(gain, _mfloor_interp59[:, np.newaxis])
    except Exception:
        pass  # nie pipeline-blockierend

    # Apply gain and reconstruct
    stft_clean = (mag * gain * np.exp(1j * phase)).astype(np.complex64)

    # Vectorized ISTFT via scipy — replaces Python OLA-loop (~21 000 iterations for 225 s)
    _, out_full = sps.istft(
        stft_clean, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=True
    )
    out_full = np.asarray(out_full, dtype=np.float32)
    out = out_full[:n] if len(out_full) >= n else np.pad(out_full, (0, n - len(out_full)))

    # Wet/dry blend
    result = x * (1.0 - strength) + out * strength
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class ModulationNoiseReductionPhase(PhaseInterface):
    """Phase 59: Signal-dependent modulation noise reduction for analog tape."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_59_modulation_noise_reduction",
            name="Modulation Noise Reduction",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_03"],
            estimated_time_factor=0.05,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.65,
            description=(
                "Signal-dependent modulation noise reduction using adaptive spectral "
                "gating that tracks signal envelope. Targets noise that varies with "
                "signal level (unique to analog tape recordings)."
            ),
        )

    @staticmethod
    def _compute_modulation_noise_profile(
        material_key: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive modulation-noise profile."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("tape", "reel_tape", "cassette")):
            min_modulation_noise_score = 0.10
            g_floor = 0.10  # §2.62 hard min
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_modulation_noise_score = 0.18
            g_floor = 0.12
        else:
            min_modulation_noise_score = 0.14
            g_floor = 0.10

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_modulation_noise_score += (_rest_norm - 0.5) * 0.10
        g_floor += (0.5 - _rest_norm) * 0.08

        _mode_offsets = {
            "fast": (0.04, 0.03),
            "balanced": (0.0, 0.0),
            "quality": (-0.03, -0.02),
            "maximum": (-0.05, -0.03),
        }
        _score_off, _g_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_modulation_noise_score += _score_off
        g_floor += _g_off

        return {
            "min_modulation_noise_score": float(np.clip(min_modulation_noise_score, 0.05, 0.25)),
            "g_floor": float(np.clip(g_floor, 0.10, 0.30)),  # §2.62 hard min
        }

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.7,
        defect_scores: dict | None = None,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        _profile_59 = self._compute_modulation_noise_profile(
            str(kwargs.get("material_type", kwargs.get("material", "unknown"))).lower(),
            kwargs.get("quality_mode"),
            float(kwargs.get("restorability_score", 50.0)),
        )
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
                    "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Modulation noise reduction skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_modulation_noise_score=_profile_59["min_modulation_noise_score"],
            g_floor=_profile_59["g_floor"],
        )
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible modulation noise
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp

            result_audio = apply_psychoacoustic_masking_clamp(
                audio,
                result_audio,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            import logging as _log59

            _log59.getLogger(__name__).debug("Phase59 masking clamp non-blocking: %s", _pm_exc)

        # §2.36 Phonem-Schutz: Plosiv-/Frikativ-Frames via get_phoneme_mask() schützen.
        # Das signal-adaptive Spektral-Gating dämpft Konsonanten-Bursts wie Modulationsrauschen.
        try:
            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_59

            _hop_59 = 512
            _mono_59: np.ndarray
            if audio.ndim == 2:
                _mono_59 = np.mean(audio, axis=0) if audio.shape[0] == 2 else np.mean(audio, axis=1)
            else:
                _mono_59 = audio
            _pmask_59 = _get_pmask_59(_mono_59.astype(np.float32), sample_rate, hop_length=_hop_59)
            if np.any(_pmask_59):
                _n59 = len(_mono_59)
                _smask_59 = np.zeros(_n59, dtype=bool)
                for _fi59, _fp59 in enumerate(_pmask_59):
                    if _fp59:
                        _fs59 = _fi59 * _hop_59
                        _fe59 = min(_n59, _fs59 + _hop_59)
                        _smask_59[_fs59:_fe59] = True
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _smask_59] = audio[:, _smask_59]
                    else:
                        result_audio[_smask_59, :] = audio[_smask_59, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_smask_59] = audio[_smask_59]
                import logging as _log59p

                _log59p.getLogger(__name__).debug(
                    "§2.36 phase_59 Phonem-Schutz: %d/%d Frames restauriert",
                    int(np.sum(_pmask_59)),
                    len(_pmask_59),
                )
        except Exception as _pmask59_exc:
            import logging as _log59q

            _log59q.getLogger(__name__).debug("§2.36 phase_59 Phonem-Mask (non-blocking): %s", _pmask59_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Atemgeräusche und Vibrato schützen.
        # Das Gating dämpft Atemgeräusche zwischen Phrasen als "niedrig-pegel Rauschen".
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector

            _npa_a59 = audio
            if _npa_a59.ndim == 2 and _npa_a59.shape[0] == 2 and _npa_a59.shape[1] > _npa_a59.shape[0]:
                _npa_a59 = _npa_a59.T
            _npa_r59 = get_natural_performance_detector().detect(_npa_a59, sample_rate)
            _npa_n59 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m59 = _npa_r59.get_protected_mask(_npa_n59, sample_rate)
            if np.any(_npa_m59):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m59] = audio[:, _npa_m59]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m59, :] = audio[_npa_m59, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m59] = audio[_npa_m59]
                import logging as _log59r

                _log59r.getLogger(__name__).debug(
                    "§2.46f phase_59 NPA: %d protected samples restauriert", int(np.sum(_npa_m59))
                )
        except Exception as _npa59_exc:
            import logging as _log59s

            _log59s.getLogger(__name__).debug("§2.46f phase_59 NPA-Guard (non-blocking): %s", _npa59_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "modulation_noise_profile": dict(_profile_59),
                "min_modulation_noise_score": float(_profile_59["min_modulation_noise_score"]),
                "g_floor": float(_profile_59["g_floor"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
