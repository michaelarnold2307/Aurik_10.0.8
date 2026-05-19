"""
Phase 65 — DSP-Vocal-Naturalness-Restaurierung (KORREKTIV, §0a Restoration-only).

Spec §7.10 [RELEASE_MUST] (Spec 06, v9.12.0)

Zweck: Behebt VQI-Abfall nach ML-NR-Phasen (phase_03, phase_29, phase_20) durch
3-stufigen DSP-Korrektiv-Prozess ohne Halluzination, §0a-konform für Restoration-Modus.

§0a-Invariante:
    VERBOTEN in studio_2026-Modus (dort: phase_42_vocal_enhancement).
    KORREKTIV — kein Energiegewinn über Input hinaus erlaubt (subraktiver Fallback).

3-stufiger Algorithmus (§7.10a):
    Stufe 1 — Spektral-Tilt-Korrektur:
        estimate_spectral_tilt(pre_nr) − estimate_spectral_tilt(post_nr) → Δtilt
        wenn |Δtilt| > 1.5 dB: Shelving-EQ ±3 dB, F_shelf=2000 Hz
    Stufe 2 — HNR-Blend:
        compute_hnr(pre_nr) − compute_hnr(post_nr) → ΔHNR
        wenn ΔHNR > 2.5 dB: blend = clip(ΔHNR/10, 0, 0.35)
        apply_hnr_blend() (kanonisch via backend.core.dsp.hnr_guard)
    Stufe 3 — Formant-Tilt-Korrektur:
        LPC F1–F4 pre/post via check_formant_shift_db() + lpc_formant_enhance()
        wenn |Δ| > 1.5 dB: narrow shelving ±2.5 dB, Q=6

VQI-Guard: compute_vqi(pre_nr, result) < compute_vqi(pre_nr, post_nr) → Rollback
Aktivierungsgate: panns_singing ≥ 0.25 AND (ΔHNR > 2.5 OR |tilt_delta| > 1.5)
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import scipy.signal as sps

from backend.core.audio_utils import to_channels_last

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algorithmus-Konstanten (§7.10a normativ)
# ---------------------------------------------------------------------------
_HNR_DELTA_THRESHOLD: float = 2.5  # dB — ab wann HNR-Blend aktiv
_TILT_DELTA_THRESHOLD: float = 1.5  # dB — ab wann Spektral-Tilt-Korrektur aktiv
_TILT_SHELF_HZ: float = 2000.0  # Hz — Shelving-Frequenz
_TILT_MAX_BOOST_DB: float = 3.0  # dB — Maximaler Tilt-Boost (limitierend)
_FORMANT_MAX_BOOST_DB: float = 2.5  # dB — Maximaler Formant-Boost
_FORMANT_SHIFT_THRESHOLD_DB: float = 1.5  # dB — Formant-Shift-Gate
_HNR_BLEND_MAX: float = 0.35  # Maximaler Dry-Wet-Blend-Faktor
_PANNS_SINGING_GATE: float = 0.25  # Mindest-Vocal-Confidence


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono_float64(audio: np.ndarray) -> np.ndarray:
    """Kanal-unabhängige Mono-Konvertierung nach float64."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[0] <= 2 and a.shape[1] > a.shape[0]:
            # [C, T] Layout (nach to_channels_last)
            return a.mean(axis=0)  # type: ignore[no-any-return]
        return a.mean(axis=1)  # type: ignore[no-any-return]
    return a.flatten()


def _estimate_spectral_tilt_db(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt Spektral-Tilt als Steigung der Power-Spektrumdichte (dB/oct).

    Positive Werte = mehr Energie bei tiefen Frequenzen (dunkler Klang).
    Negative Werte = mehr Energie bei hohen Frequenzen (heller Klang).
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 512:
        return 0.0
    window = np.hanning(n_fft)
    # Mittelwert über mehrere Frames für Stabilität
    n_frames = max(1, len(audio_mono) // n_fft)
    spectra = []
    for i in range(min(n_frames, 8)):
        start = i * n_fft
        seg = audio_mono[start : start + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        spectra.append(np.abs(np.fft.rfft(seg * window)) ** 2)

    if not spectra:
        return 0.0

    mean_spec = np.mean(spectra, axis=0) + 1e-20
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Lineare Regression über log-Frequenz vs dB-Energie (nur 100 Hz – 8 kHz)
    mask = (freqs >= 100.0) & (freqs <= 8000.0)
    if mask.sum() < 8:
        return 0.0

    log_f = np.log2(freqs[mask] + 1e-9)
    db_e = 10.0 * np.log10(mean_spec[mask])
    # Lineare Regression
    coef = np.polyfit(log_f, db_e, 1)
    return float(coef[0])  # dB/octave Steigung


def _apply_shelving_eq(
    audio: np.ndarray,
    sr: int,
    shelf_hz: float,
    boost_db: float,
    shelf_type: str = "low",
) -> np.ndarray:
    """Wendet ein einfaches Low- oder High-Shelf-EQ an.

    Args:
        audio: [T] oder [C, T] Layout
        sr:    Abtastrate
        shelf_hz: Shelf-Frequenz
        boost_db: Gain in dB (positiv = Boost, negativ = Cut)
        shelf_type: "low" oder "high"
    """
    # Biquad-Shelf-Filter via scipy.signal
    # A = 10^(dB/40) (Standard-Shelf-Gain)
    A = 10.0 ** (boost_db / 40.0)
    w0 = 2.0 * np.pi * shelf_hz / sr
    S = 1.0  # Shelf-Slope = 1 (flache Flanke)
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    if shelf_type == "low":
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    else:  # "high"
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    # Normalisieren
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])

    audio_f64 = np.asarray(audio, dtype=np.float64)
    if audio_f64.ndim == 1:
        return sps.sosfiltfilt(sos, audio_f64).astype(audio.dtype)  # type: ignore[no-any-return]
    if audio_f64.ndim == 2:
        if audio_f64.shape[0] <= 2 and audio_f64.shape[1] > audio_f64.shape[0]:
            # [C, T] — nach to_channels_last
            return np.stack([sps.sosfiltfilt(sos, audio_f64[c]) for c in range(audio_f64.shape[0])]).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([sps.sosfiltfilt(sos, audio_f64[:, c]) for c in range(audio_f64.shape[1])], axis=1).astype(  # type: ignore[no-any-return]
            audio.dtype
        )
    return audio


# ---------------------------------------------------------------------------
# PhaseInterface-Klasse
# ---------------------------------------------------------------------------


class VocalNaturalnessRestorationPhase(PhaseInterface):
    """Phase 65: DSP-Vocal-Naturalness-Restaurierung.

    Spec §7.10 (Spec 06 v9.12.0).
    KORREKTIV — Restoration-only (§0a). Kein Energiegewinn über Input hinaus.
    """

    _PHASE_ID = "phase_65_vocal_naturalness_restoration"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name="Vocal Naturalness Restoration (DSP-Korrektiv)",
            category=PhaseCategory.RESTORATION,
            priority=6,
            version="1.0.0",
            dependencies=["phase_03", "phase_29", "phase_20"],
            estimated_time_factor=0.03,
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.88,
            description=(
                "§7.10 DSP-Korrektiv (3-Stufen): Spektral-Tilt + HNR-Blend + Formant-Tilt-Korrektur. "
                "Activation gate: panns_singing ≥ 0.25 AND (ΔHNR > 2.5 OR |tilt_delta| > 1.5). "
                "Restoration-only — §0a VERBOTEN in Studio 2026. "
                "VQI-Guard: Rollback wenn VQI nach Verarbeitung < VQI vorher."
            ),
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:  # pylint: disable=arguments-differ  # type: ignore[override]
        """DSP-Vocal-Naturalness-Restaurierung.

        Args:
            audio:       Mono oder Stereo
            sample_rate: Abtastrate Hz (MUSS 48000)
            **kwargs:
                panns_singing (float): PANNs-Singing-Konfidenz [0, 1]
                pre_nr_audio (np.ndarray): Audio VOR NR-Phasen (aus _restoration_context)
                quality_mode (str): "restoration" | "studio_2026"
                strength (float): Phase-Strength [0, 1]
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"Phase65 SR MUSS 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p65_transposed = to_channels_last(audio)
        self.validate_input(audio)
        t0 = time.time()

        _p65_meta: dict = {
            "algorithm": "vocal_naturalness_restoration_dsp",
            "stages_applied": [],
            "activation_triggered": False,
        }

        # §0a Guard: Phase_65 ist VERBOTEN in Studio 2026
        quality_mode = str(kwargs.get("quality_mode", "restoration")).strip().lower()
        if quality_mode in ("studio_2026", "studio2026"):
            logger.debug("§0a Phase65: skipped in studio_2026 mode (use phase_42_vocal_enhancement instead)")
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_studio2026_mode"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        # Vocal-Gate: nur bei panns_singing >= 0.25
        panns_singing = float(kwargs.get("panns_singing", kwargs.get("panns_singing_confidence", 0.0)))
        if panns_singing < _PANNS_SINGING_GATE:
            logger.debug("Phase65: vocal gate not met (panns_singing=%.3f < %.2f)", panns_singing, _PANNS_SINGING_GATE)
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_no_vocal"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        # Pre-NR-Checkpoint (erforderlich für alle 3 Stufen)
        pre_nr_audio_raw = kwargs.get("pre_nr_audio")
        if pre_nr_audio_raw is None:
            logger.debug("Phase65: pre_nr_audio nicht in kwargs — skipped")
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta["algorithm"] = "skipped_no_pre_nr_audio"
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0},
            )

        pre_nr_audio = np.asarray(pre_nr_audio_raw, dtype=np.float32)
        # to_channels_last für pre_nr_audio
        pre_nr_ch, _ = to_channels_last(pre_nr_audio)

        # Mono für Analyse
        pre_mono = _to_mono_float64(pre_nr_ch.astype(np.float64))
        post_mono = _to_mono_float64(audio.astype(np.float64))

        # --- Vorab: ΔHNR und Tilt-Delta berechnen ---
        hnr_pre = 0.0
        hnr_post = 0.0
        tilt_pre = 0.0
        tilt_post = 0.0

        try:
            from backend.core.dsp.hnr_guard import (  # pylint: disable=import-outside-toplevel
                compute_hnr as _compute_hnr65,
            )

            hnr_pre = float(_compute_hnr65(pre_mono.astype(np.float32), sample_rate))
            hnr_post = float(_compute_hnr65(post_mono.astype(np.float32), sample_rate))
        except Exception as _hnr_exc:
            logger.debug("Phase65 compute_hnr non-blocking: %s", _hnr_exc)

        tilt_pre = _estimate_spectral_tilt_db(pre_mono, sample_rate)
        tilt_post = _estimate_spectral_tilt_db(post_mono, sample_rate)
        delta_hnr = hnr_pre - hnr_post
        tilt_delta = tilt_post - tilt_pre  # positiv = post wurde dunkler

        # Aktivierungsgate (§7.10a)
        activation_triggered = (delta_hnr > _HNR_DELTA_THRESHOLD) or (abs(tilt_delta) > _TILT_DELTA_THRESHOLD)

        if not activation_triggered:
            logger.debug("Phase65: activation gate not met (delta_hnr=%.2f, tilt_delta=%.2f)", delta_hnr, tilt_delta)
            audio_out = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio_out = np.clip(audio_out, -1.0, 1.0)
            if _p65_transposed:
                audio_out = audio_out.T
            _p65_meta.update(
                {
                    "activation_triggered": False,
                    "delta_hnr": round(delta_hnr, 3),
                    "tilt_delta": round(tilt_delta, 3),
                    "algorithm": "skipped_gate_not_met",
                }
            )
            return PhaseResult(
                success=True,
                audio=audio_out,
                execution_time_seconds=time.time() - t0,
                metadata=_p65_meta,
                metrics={"activation_triggered": 0, "delta_hnr": delta_hnr, "tilt_delta": tilt_delta},
            )

        _p65_meta["activation_triggered"] = True
        _p65_meta["delta_hnr"] = round(delta_hnr, 3)
        _p65_meta["tilt_delta"] = round(tilt_delta, 3)

        result = audio.copy().astype(np.float32)
        effective_strength = float(np.clip(float(kwargs.get("strength", 1.0)), 0.0, 1.0))

        # ---- Stufe 1: Spektral-Tilt-Korrektur ----
        if abs(tilt_delta) > _TILT_DELTA_THRESHOLD:
            try:
                # tilt_delta > 0 → post wurde dunkler (HF verloren) → High-Shelf-Boost
                # tilt_delta < 0 → post wurde heller → Low-Shelf-Boost
                boost_db = float(np.clip(-tilt_delta * 0.5, -_TILT_MAX_BOOST_DB, _TILT_MAX_BOOST_DB))
                boost_db *= effective_strength
                if abs(boost_db) > 0.3:
                    shelf_type = "high" if boost_db > 0 else "low"
                    result_tilted = _apply_shelving_eq(result, sample_rate, _TILT_SHELF_HZ, abs(boost_db), shelf_type)
                    result = result_tilted.astype(np.float32)
                    _p65_meta["stages_applied"].append(f"tilt_correction_{shelf_type}_{boost_db:.1f}dB")
                    logger.debug("Phase65 Stufe1 Tilt: boost=%.2f dB %s-shelf", boost_db, shelf_type)
            except Exception as _tilt_exc:
                logger.debug("Phase65 Stufe1 tilt non-blocking: %s", _tilt_exc)

        # ---- Stufe 2: HNR-Blend ----
        if delta_hnr > _HNR_DELTA_THRESHOLD:
            try:
                from backend.core.dsp.hnr_guard import (  # pylint: disable=import-outside-toplevel
                    apply_hnr_blend as _apply_hnr65,
                )

                _hnr_blended65, _hnr_diag65 = _apply_hnr65(
                    pre_nr_ch.astype(np.float32),
                    result,
                    sample_rate,
                )
                if _hnr_diag65.get("over_cleaned"):
                    # Angepasster Blend basierend auf ΔHNR
                    manual_blend = float(np.clip(delta_hnr / 10.0, 0.0, _HNR_BLEND_MAX)) * effective_strength
                    result = ((1.0 - manual_blend) * result + manual_blend * pre_nr_ch.astype(np.float32)).astype(
                        np.float32
                    )
                    _p65_meta["stages_applied"].append(f"hnr_blend_{manual_blend:.3f}")
                    logger.debug("Phase65 Stufe2 HNR-Blend: blend=%.3f", manual_blend)
                elif _hnr_diag65.get("blend_applied"):
                    result = _hnr_blended65
                    _p65_meta["stages_applied"].append("hnr_blend_auto")
            except Exception as _hnr65_exc:
                logger.debug("Phase65 Stufe2 HNR-Blend non-blocking: %s", _hnr65_exc)

        # ---- Stufe 3: Formant-Tilt-Korrektur ----
        try:
            from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                check_formant_shift_db as _check_fshift65,
            )
            from backend.core.dsp.lpc_formant_tracker import (  # pylint: disable=import-outside-toplevel
                lpc_formant_enhance as _lpc_enhance65,
            )

            _rollback_needed, _max_shift = _check_fshift65(
                pre_nr_ch.astype(np.float32),
                result,
                sample_rate,
                threshold_db=_FORMANT_SHIFT_THRESHOLD_DB,
            )
            if _rollback_needed and _max_shift > _FORMANT_SHIFT_THRESHOLD_DB:
                # Formant-Tilt-Korrektur: sanfte LPC-Boost-Rekonstruktion
                _formant_boost_db = float(np.clip(_max_shift * 0.4, 0.1, _FORMANT_MAX_BOOST_DB)) * effective_strength
                result_enhanced = _lpc_enhance65(
                    result,
                    sample_rate,
                    max_boost_db=_formant_boost_db,
                )
                result = result_enhanced.astype(np.float32)
                _p65_meta["stages_applied"].append(f"formant_tilt_{_formant_boost_db:.2f}dB")
                logger.debug(
                    "Phase65 Stufe3 Formant-Tilt: shift=%.2f dB, boost=%.2f dB",
                    _max_shift,
                    _formant_boost_db,
                )
        except Exception as _ft65_exc:
            logger.debug("Phase65 Stufe3 formant-tilt non-blocking: %s", _ft65_exc)

        # NaN-Safety + Clip
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)

        # ---- VQI-Guard: Rollback wenn VQI schlechter geworden ----
        _vqi_before: float = -1.0
        _vqi_after: float = -1.0
        try:
            from backend.core.musical_goals.vocal_quality_index import (  # pylint: disable=import-outside-toplevel
                compute_vqi as _compute_vqi65,
            )

            _vqi_res_before = _compute_vqi65(pre_nr_ch.astype(np.float32), audio.astype(np.float32), sample_rate)
            _vqi_res_after = _compute_vqi65(pre_nr_ch.astype(np.float32), result, sample_rate)
            _vqi_before = float(_vqi_res_before.get("vqi", -1.0))
            _vqi_after = float(_vqi_res_after.get("vqi", -1.0))
            if _vqi_before > 0 and _vqi_after > 0 and _vqi_after < _vqi_before:
                logger.warning(
                    "Phase65 VQI-Guard: vqi_after=%.3f < vqi_before=%.3f → Rollback",
                    _vqi_after,
                    _vqi_before,
                )
                result = audio.copy().astype(np.float32)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                result = np.clip(result, -1.0, 1.0)
                _p65_meta["vqi_rollback"] = True
        except Exception as _vqi65_exc:
            logger.debug("Phase65 VQI-Guard non-blocking: %s", _vqi65_exc)

        _p65_meta["vqi_before"] = round(_vqi_before, 4)
        _p65_meta["vqi_after"] = round(_vqi_after, 4)

        if _p65_transposed:
            result = result.T

        return PhaseResult(
            success=True,
            audio=result,
            execution_time_seconds=time.time() - t0,
            metadata=_p65_meta,
            metrics={
                "activation_triggered": 1 if activation_triggered else 0,
                "delta_hnr": round(delta_hnr, 3),
                "tilt_delta": round(tilt_delta, 3),
                "vqi_before": round(_vqi_before, 4),
                "vqi_after": round(_vqi_after, 4),
                "stages_applied": len(_p65_meta.get("stages_applied", [])),
            },
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: VocalNaturalnessRestorationPhase | None = None
_lock = threading.Lock()


def get_phase_65() -> VocalNaturalnessRestorationPhase:
    """Thread-safe Singleton accessor (§0 Kopilot-Instructions Singleton-Pattern)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocalNaturalnessRestorationPhase()
    return _instance
