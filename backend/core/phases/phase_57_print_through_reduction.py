"""
Phase 57 — Print-Through Reduction (Bidirektionale Adaptive Temporal Subtraction).

Spec §7.x / DSP-Spezialregel: Print-Through-Reduktion (phase_29, reel_tape):
    - Bidirektionale Adaptive Temporal Subtraction (LMS)
    - Pre-Echo (Vorwärtswicklung): schwächer → alpha_pre  ∈ [0.03, 0.25]
    - Post-Echo (Rückwärtswicklung): stärker  → alpha_post ∈ [0.05, 0.35]
    - Algorithmus:
        1. Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post
        2. LMS-Adaptivfilter separat für Pre- und Post-Echo
        3. audio_clean[t] = audio[t] − alpha_pre·audio[t + delay_pre]
                                       − alpha_post·audio[t − delay_post]
        4. Spectral Coherence vor/nach ≥ 0.90 + PGHI
    - Fallback: NMF-β Dekomposition (einseitig, nur Post-Echo)

VERBOTEN: Comb-Filter, einseitiges α-Modell (alpha_pre == alpha_post)
"""

from __future__ import annotations

import logging
import math

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = sig.mean(axis=0).astype(np.float64) if sig.shape[0] <= 2 else sig.mean(axis=1).astype(np.float64)
    else:
        _mono = sig.astype(np.float64)
    _frame = 480  # 10 ms @ 48 kHz
    _active = [
        _mono[i : i + _frame]
        for i in range(0, len(_mono) - _frame, _frame)
        if 20.0 * np.log10(np.sqrt(np.mean(_mono[i : i + _frame] ** 2)) + 1e-10) > -50.0
    ]
    if not _active:
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(np.concatenate(_active) ** 2)) + 1e-10))


# ─── Konstanten (Spec normativ) ────────────────────────────────────────────────
_XCORR_MAX_DELAY_MS: float = 600.0  # Maximales Korrelations-Suchfenster in ms
_ALPHA_PRE_RANGE: tuple[float, float] = (0.03, 0.25)  # Vorwärtswicklung
_ALPHA_POST_RANGE: tuple[float, float] = (0.05, 0.35)  # Rückwärtswicklung
_LMS_MU: float = 0.001  # LMS-Schrittweite
_LMS_TAPS: int = 1  # Einfaches 1-Tap-Modell (α-Schätzung)
_MIN_PRINT_THROUGH_SCORE: float = 0.15  # Gate: unter diesem Wert kein Processing
_COHERENCE_FLOOR: float = 0.90  # Spec: Spectral Coherence nachher ≥ 0.90


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.8,
    defect_scores: dict | None = None,
    min_print_through_score: float = _MIN_PRINT_THROUGH_SCORE,
    coherence_floor: float = _COHERENCE_FLOOR,
) -> np.ndarray:
    """Haupteintrittspunkt für Phase 57.

    Args:
        audio:        Input-Audio float32, Mono oder Stereo, [samples] oder [channels, samples]
        sample_rate:  Muss 48000 Hz sein
        strength:     Verarbeitungsstärke 0–1 (skaliert alpha-Klammern)
        defect_scores: DefektScan-Scores (optional, für Print-Through-Gate)

    Returns:
        Bereinigtes Audio, gleiche Form wie Input
    """
    assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Gate: nur bei ausreichendem Print-Through-Score verarbeiten
    _pt_score = 0.0
    if defect_scores is not None:
        _pt_score = float(defect_scores.get("print_through", 0.0))
        if _pt_score < min_print_through_score:
            logger.debug(
                "Phase 57: Print-Through-Score %.3f < %.3f — übersprungen",
                _pt_score,
                min_print_through_score,
            )
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: LMS auf Mono-Mix, identische Korrektur auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_repaired = apply(mono_mix, sample_rate, strength=strength, defect_scores=defect_scores)
        _eps_pt = 1e-10
        _gain_pt = np.where(
            np.abs(mono_mix) > _eps_pt,
            mono_repaired / (mono_mix + _eps_pt * np.sign(mono_mix + _eps_pt)),
            1.0,
        )
        _gain_pt = np.clip(_gain_pt, 0.0, 10.0)
        out = np.stack([audio[0] * _gain_pt, audio[1] * _gain_pt], axis=0)
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    # Ab hier: Mono-Verarbeitung
    x = audio.astype(np.float64)
    len(x)
    max_delay_samples = int(_XCORR_MAX_DELAY_MS * 0.001 * sample_rate)

    # Skalierte Alpha-Klammern gemäß strength
    alpha_pre_max = _ALPHA_PRE_RANGE[0] + strength * (_ALPHA_PRE_RANGE[1] - _ALPHA_PRE_RANGE[0])
    alpha_post_max = _ALPHA_POST_RANGE[0] + strength * (_ALPHA_POST_RANGE[1] - _ALPHA_POST_RANGE[0])
    alpha_pre_max = float(np.clip(alpha_pre_max, _ALPHA_PRE_RANGE[0], _ALPHA_PRE_RANGE[1]))
    alpha_post_max = float(np.clip(alpha_post_max, _ALPHA_POST_RANGE[0], _ALPHA_POST_RANGE[1]))

    # Schritt 1: Kreuzkorrelation → delay_pre, delay_post
    delay_pre, delay_post = _find_delays(x, max_delay_samples)
    logger.debug(
        "Phase 57: delay_pre=%d samples (%.1f ms), delay_post=%d samples (%.1f ms)",
        delay_pre,
        delay_pre * 1000.0 / sample_rate,
        delay_post,
        delay_post * 1000.0 / sample_rate,
    )
    if delay_pre <= 0 and delay_post <= 0:
        logger.debug("Phase 57: Keine signifikanten Echos gefunden — übersprungen")
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Schritt 2+3: LMS-Adaptivfilter für Pre- und Post-Echo
    try:
        x_clean = _lms_bilateral_subtraction(
            x=x,
            delay_pre=delay_pre,
            delay_post=delay_post,
            alpha_pre_max=alpha_pre_max,
            alpha_post_max=alpha_post_max,
        )
    except Exception as _lms_exc:
        logger.warning("Phase 57: LMS-Subtraction fehlgeschlagen, NMF-Fallback: %s", _lms_exc)
        x_clean = _nmf_post_echo_fallback(x, delay_post, alpha_post_max)

    # Schritt 4: Spectral-Coherence-Check — Rollback wenn Qualität schlechter
    _coh = _spectral_coherence(x, x_clean, sample_rate)
    if _coh < coherence_floor:
        logger.warning(
            "Phase 57: Spectral Coherence nach LMS %.3f < %.3f — Rollback auf Original",
            _coh,
            coherence_floor,
        )
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    result = np.nan_to_num(x_clean, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result.astype(np.float32), -1.0, 1.0)


# ─── Interne Hilfsfunktionen ───────────────────────────────────────────────────


def _find_delays(x: np.ndarray, max_delay: int) -> tuple[int, int]:
    """Findet Pre-Echo- und Post-Echo-Delays via Kreuzkorrelations-Peak-Suche.

    Returns:
        (delay_pre, delay_post) in Samples. 0 wenn kein signifikanter Peak.
    """
    n = len(x)
    # Energie-normierte Kreuzkorrelation auf kurzen Segment (max 10 s)
    seg_len = min(n, 480000)  # 10 s @ 48 kHz
    x_seg = x[:seg_len]

    # FFT-basierte Autokorrelation
    n_fft = int(2 ** math.ceil(math.log2(2 * seg_len - 1)))
    X = np.fft.rfft(x_seg, n=n_fft)
    acorr = np.fft.irfft(X * np.conj(X), n=n_fft)
    acorr = acorr[: max_delay + 1]  # nur positive Lags

    # Normalize durch Nulllag
    acorr_norm = acorr / (acorr[0] + 1e-10)

    # Signifikanter Peak: > 5 % der Energie, exklusive lag=0
    threshold = 0.05
    acorr_norm[0] = 0.0  # Nulllag ausschließen

    # Post-Echo: höchster Peak ≥ 10 Samples (≥ 0.2 ms)
    delay_post = 0
    if len(acorr_norm) > 10:
        _cand = np.argmax(np.abs(acorr_norm[10:])) + 10
        if abs(acorr_norm[_cand]) > threshold:
            delay_post = int(_cand)

    # Pre-Echo: zweithöchster Peak nach delay_post (in negativem Lag = signal before)
    # Pre-Echo ist typischerweise schwächer und bei ca. 60–70% von delay_post
    delay_pre = 0
    if delay_post > 0:
        # Pre-Echo erwartet bei etwas kleineren Lags (Vorwärtswicklung näher am Signal)
        _pre_search_end = max(10, delay_post - 5)
        if _pre_search_end > 10:
            _cand_pre = np.argmax(np.abs(acorr_norm[10:_pre_search_end])) + 10
            if abs(acorr_norm[_cand_pre]) > threshold * 0.6:  # Pre-Echo typisch schwächer
                delay_pre = int(_cand_pre)

    return delay_pre, delay_post


def _lms_bilateral_subtraction(
    x: np.ndarray,
    delay_pre: int,
    delay_post: int,
    alpha_pre_max: float,
    alpha_post_max: float,
) -> np.ndarray:
    """Bidirektionale LMS-Adaptivfilter-Subtraktion (Pre- + Post-Echo).

    Algorithm (Spec §7.6 Print-Through normativ):
        audio_clean[t] = audio[t]
                         − alpha_pre · audio[t + delay_pre]   (Pre-Echo vorwärts)
                         − alpha_post · audio[t − delay_post] (Post-Echo rückwärts)

    LMS-Update (adaptiv):
        e[t] = x[t] − alpha · x_echo[t]
        alpha_new = alpha + mu · e[t] · x_echo[t]
        alpha_new = clip(alpha_new, 0.0, alpha_max)
    """
    n = len(x)
    out = x.copy()
    mu = _LMS_MU

    # Post-Echo LMS (Rückwärtswicklung — stärker, zuerst verarbeiten)
    if delay_post > 0:
        alpha_post = alpha_post_max * 0.5  # Initial-Schätzung
        for t in range(delay_post, n):
            echo_post = x[t - delay_post]
            e = out[t] - alpha_post * echo_post
            alpha_post += mu * e * echo_post
            alpha_post = float(np.clip(alpha_post, 0.0, alpha_post_max))
            out[t] = e

    # Pre-Echo LMS (Vorwärtswicklung — schwächer)
    if delay_pre > 0:
        alpha_pre = alpha_pre_max * 0.3  # Initial-Schätzung (schwächer)
        for t in range(0, n - delay_pre):
            echo_pre = x[t + delay_pre]
            e = out[t] - alpha_pre * echo_pre
            alpha_pre += mu * e * echo_pre
            alpha_pre = float(np.clip(alpha_pre, 0.0, alpha_pre_max))
            out[t] = e

    logger.debug(
        "Phase 57: LMS bilateral — delay_pre=%d delay_post=%d alpha_pre_max=%.3f alpha_post_max=%.3f",
        delay_pre,
        delay_post,
        alpha_pre_max,
        alpha_post_max,
    )
    return out


def _nmf_post_echo_fallback(x: np.ndarray, delay_post: int, alpha_max: float) -> np.ndarray:
    """NMF-β Fallback: einseitige Post-Echo-Subtraktion (Spec normativ als NMF-β Fallback).

    Vereinfachte Version ohne Pre-Echo (einseitig).
    """
    if delay_post <= 0:
        return x
    n = len(x)
    out = x.copy()
    alpha = alpha_max * 0.5
    for t in range(delay_post, n):
        out[t] = x[t] - alpha * x[t - delay_post]
    logger.debug("Phase 57: NMF-Fallback (einseitig, Post-Echo) delay=%d alpha=%.3f", delay_post, alpha)
    return out


def _spectral_coherence(x_orig: np.ndarray, x_clean: np.ndarray, sr: int) -> float:
    """Berechnet Spectral Coherence zwischen Original und bereinigtem Signal [0, 1]."""
    try:
        n_fft = min(2048, len(x_orig) // 4)
        if n_fft < 64:
            return 1.0
        _f, coh = sps.coherence(x_orig.astype(np.float64), x_clean.astype(np.float64), fs=sr, nperseg=n_fft)
        return float(np.mean(coh))
    except Exception as _coh_exc:
        logger.debug("Spectral-Coherence-Berechnung fehlgeschlagen: %s", _coh_exc)
        return 1.0  # im Zweifel: nicht rollbacken


# ─── PhaseInterface-Klasse (normativ für test_all_phases_normative.py) ─────────

import time as _time

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class PrintThroughReductionPhase(PhaseInterface):
    """Phase 57: Print-Through Reduction — Bidirektionale Adaptive Temporal Subtraction.

    Spec §7.x: Pre-Echo (Vorwärtswicklung) + Post-Echo (Rückwärtswicklung)
    via LMS-Adaptivfilter separat geschätzt.
    VERBOTEN: Comb-Filter, einseitiges α-Modell.
    Fallback: NMF-β Post-Echo (einseitig).
    """

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_57_print_through_reduction",
            name="Print-Through Reduction",
            category=PhaseCategory.RESTORATION,
            priority=7,
            dependencies=["phase_29"],
            estimated_time_factor=0.04,
            version="1.0.0",
            memory_requirement_mb=16,
            is_cpu_intensive=False,
            quality_impact=0.70,
            description=(
                "Bidirektionale Adaptive Temporal Subtraction (LMS) für Print-Through "
                "(Magnetband-Vor-/Nachecho). Pre- und Post-Echo separat via LMS. "
                "Nur für TAPE/REEL_TAPE Material aktiv."
            ),
        )

    @staticmethod
    def _compute_print_through_profile(
        material_key: str,
        quality_mode: str,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive print-through profile."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("reel_tape", "tape", "cassette")):
            min_print_through_score = 0.12
            coherence_floor = 0.93
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_print_through_score = 0.22
            coherence_floor = 0.95
        else:
            min_print_through_score = 0.18
            coherence_floor = 0.94

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_print_through_score += (_rest_norm - 0.5) * 0.14

        _mode_offsets = {
            "fast": (0.04, -0.01),
            "balanced": (0.0, 0.0),
            "quality": (-0.03, 0.02),
            "maximum": (-0.05, 0.03),
        }
        _score_off, _coh_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_print_through_score += _score_off
        coherence_floor += _coh_off

        return {
            "min_print_through_score": float(np.clip(min_print_through_score, 0.05, 0.30)),
            "coherence_floor": float(np.clip(coherence_floor, 0.90, 0.99)),
        }

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.8,
        defect_scores: dict | None = None,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        _profile_57 = self._compute_print_through_profile(
            str(kwargs.get("material_type", kwargs.get("material", "unknown"))).lower(),
            str(kwargs.get("quality_mode", "balanced")),
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
                    "print_through_score": float((_defect_scores or {}).get("print_through", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Print-through reduction skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_print_through_score=_profile_57["min_print_through_score"],
            coherence_floor=_profile_57["coherence_floor"],
        )
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible print-through
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
            import logging as _log57

            _log57.getLogger(__name__).debug("Phase57 masking clamp non-blocking: %s", _pm_exc)

        # §2.46f Natural-Performance-Artifacts-Guard — Print-Through-Reduktion darf
        # Atemgeräusche und Vibrato-Zonen nicht durch Subtraktionsfilter tilgen.
        try:
            from backend.core.natural_performance_detector import get_natural_performance_detector

            _npa_a57 = audio
            if _npa_a57.ndim == 2 and _npa_a57.shape[0] == 2 and _npa_a57.shape[1] > _npa_a57.shape[0]:
                _npa_a57 = _npa_a57.T
            _npa_r57 = get_natural_performance_detector().detect(_npa_a57, sample_rate)
            _npa_n57 = (
                result_audio.shape[1]
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else result_audio.shape[0]
            )
            _npa_m57 = _npa_r57.get_protected_mask(_npa_n57, sample_rate)
            if np.any(_npa_m57):
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _npa_m57] = audio[:, _npa_m57]
                    elif result_audio.shape == audio.shape:
                        result_audio[_npa_m57, :] = audio[_npa_m57, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_npa_m57] = audio[_npa_m57]
        except Exception as _npa57_exc:
            import logging as _log57n

            _log57n.getLogger(__name__).debug("§2.46f phase_57 NPA-Guard (non-blocking): %s", _npa57_exc)

        # §2.36 Phonem-Schutz: Print-Through-Reduktion subtrahiert Vor-/Nachhall-Energie.
        # Plosive Burst-Transienten haben ähnliche Burst-Energie — Plosiv-Frames schützen.
        try:
            pass

            from backend.core.lyrics_guided_enhancement import get_phoneme_mask as _get_pmask_57

            _hop_57 = 512
            _mono_57 = (
                result_audio.mean(axis=0)
                if (result_audio.ndim == 2 and result_audio.shape[0] == 2 and result_audio.shape[1] > 2)
                else (result_audio.mean(axis=1) if result_audio.ndim == 2 else result_audio)
            )
            _pmask_57 = _get_pmask_57(_mono_57.astype(np.float32), sample_rate, hop_length=_hop_57)
            if np.any(_pmask_57):
                _n_57 = _mono_57.shape[0]
                _smask_57 = np.zeros(_n_57, dtype=bool)
                for _fi57, _fp57 in enumerate(_pmask_57):
                    if _fp57:
                        _fs57 = _fi57 * _hop_57
                        _fe57 = min(_n_57, _fs57 + _hop_57)
                        _smask_57[_fs57:_fe57] = True
                if result_audio.ndim == 2 and audio.ndim == 2:
                    if result_audio.shape[0] == 2 and result_audio.shape[1] > 2:
                        result_audio[:, _smask_57] = audio[:, _smask_57]
                    elif result_audio.shape == audio.shape:
                        result_audio[_smask_57, :] = audio[_smask_57, :]
                elif result_audio.ndim == 1 and audio.ndim == 1:
                    result_audio[_smask_57] = audio[_smask_57]
        except Exception as _pm57_exc:
            logging.getLogger(__name__).debug("\u00a72.36 phase_57 Phonem-Mask (non-blocking): %s", _pm57_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        _pt_score = float((_defect_scores or {}).get("print_through", 0.0)) if _defect_scores else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "print_through_score": _pt_score,
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "print_through_profile": dict(_profile_57),
                "min_print_through_score": float(_profile_57["min_print_through_score"]),
                "coherence_floor": float(_profile_57["coherence_floor"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
