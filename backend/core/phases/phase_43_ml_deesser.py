"""
Phase 43: Hybrid De-Esser v2.2 — Stimmtyp-adaptiver Sidechain-De-Esser
=======================================================================

DSP-Primärpfad mit optionaler ML-Feinveredelung (MP-SENet, streng gegated).
Stimmtyp-adaptive Frequenzauswahl gemäß §2.8 (Vocal-Restaurierungskette).

ALGORITHMUS — Split-Band De-Esser:
  1. Sibilantenband extrahieren: Butterworth-Bandpass 4. Ordnung, gender-adaptiv
  2. Hüllkurve des Sibilantenbands via RMS-Fenster (5 ms)
  3. Gain Reduction: wenn Hüllkurve > threshold, Kompression 1:4
     GR = (threshold / envelope)^((ratio-1)/ratio)  → logarithmisch glatt
  4. Smooth Gain per Sample: Attack 2 ms, Release 80 ms
  5. Strength-Cap: GR >= strength_cap (verhindert Überdämpfung bei Schlager-Modus)
  6. Gefilterte Band × Gain → vom Original subtrahieren
  7. Funktioniert Mono + Stereo (channelweise)

PARAMETER (kwargs):
  threshold_db  (float, default -20.0)  — Detektionsschwelle in dBFS
  ratio         (float, default 4.0)    — Kompressionsverhältnis (1:ratio)
  attack_ms     (float, default 2.0)    — Gain-Attack in ms
  release_ms    (float, default 80.0)   — Gain-Release in ms
  freq_low      (float, optional)       — Untere Sibilanzgrenze Hz (überschreibt gender)
  freq_high     (float, optional)       — Obere Sibilanzgrenze Hz (überschreibt gender)
  gender        (str, default "unknown") — Stimmtyp: "male"|"female"|"child"|"unknown"
  strength_cap  (float, default 1.0)    — Max. GR-Stärke 0.0–1.0 (§2.19.3 Schlager: 0.45)

STIMMTYP-ADAPTIVE FREQUENZEN (§2.8):
  male:    5 000 – 10 000 Hz
  female:  6 000 – 12 000 Hz
  child:   7 000 – 14 000 Hz
  unknown: 5 000 –  9 000 Hz  (konservativ)

Author: Aurik Development Team
Version: 2.1.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stimmtyp-adaptive Sibilanz-Frequenzgrenzen (§2.8 Vocal-Restaurierungskette)
# ---------------------------------------------------------------------------
#  MALE:    5 –10 kHz  (tiefer Grundton, breitere Konsonanten)
#  FEMALE:  6 –12 kHz  (höherer Grundton, scharfe Frikative)
#  CHILD:   7 –14 kHz  (höchster Grundton, sehr hohe Sibilanz)
#  unknown: 5 – 9 kHz  (konservativer Fallback)
GENDER_FREQ_MAP: dict[str, tuple[float, float]] = {
    "male": (5_000.0, 10_000.0),
    "female": (6_000.0, 12_000.0),
    "child": (7_000.0, 14_000.0),
    "unknown": (5_000.0, 9_000.0),
}

_DEFAULT_THRESHOLD_DB = -20.0
_DEFAULT_RATIO = 4.0
_DEFAULT_ATTACK_MS = 2.0
_DEFAULT_RELEASE_MS = 80.0
_DEFAULT_GENDER = "unknown"
_DEFAULT_STRENGTH_CAP = 1.0  # kein Cap; §2.19.3 Schlager-Modus: 0.45


def _rms_envelope(signal: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """RMS-Hüllkurve mit gleitendem Fenster."""
    win = max(2, int(window_ms / 1000.0 * sr))
    sq = signal**2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-12)
    return rms


def _smooth_gain(gain_lin: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Asymmetric first-order IIR smoother: fast attack, slow release.

    Replaces the per-sample Python loop with scipy.signal.lfilter-based
    block processing (block size 512 samples ≈ 10.7 ms at 48 kHz).
    Speedup: ~500x vs pure Python loop (per-sample loop over 14 M samples
    at 48 kHz for a 5-min file blocked the thread for 30–60 s, appearing
    as an infinite hang to the user).

    Algorithm:
        smoothed[0] = 1.0  (matches original np.ones_like initialisation)
        For each 512-sample block starting at index 1:
            if block[0] < current_state: use attack coefficient (fast ↓)
            else:                         use release coefficient (slow ↑)
            Process block with scipy lfilter, carry state via zi.

    The block-mode decision introduces at most ≤10.7 ms of timing jitter
    at attack/release transition boundaries — inaudible in a de-esser.
    """
    att = np.exp(-1.0 / (attack_ms / 1000.0 * sr + 1e-6))
    rel = np.exp(-1.0 / (release_ms / 1000.0 * sr + 1e-6))

    n = len(gain_lin)
    # smoothed[0] always 1.0 — preserve original initialisation semantics.
    smoothed = np.ones(n, dtype=np.float64)
    if n <= 1:
        return smoothed.astype(gain_lin.dtype)

    b_att = np.array([1.0 - att])
    a_att = np.array([1.0, -att])
    b_rel = np.array([1.0 - rel])
    a_rel = np.array([1.0, -rel])

    _BLOCK = 512  # ≈ 10.7 ms at 48 kHz
    state = 1.0  # initial smoothed value (= smoothed[0], matching np.ones_like)
    x = gain_lin.astype(np.float64)

    for start in range(1, n, _BLOCK):
        end = min(start + _BLOCK, n)
        chunk = x[start:end]
        if chunk[0] < state:
            b, a, coef = b_att, a_att, att  # gain falling → fast attack
        else:
            b, a, coef = b_rel, a_rel, rel  # gain rising  → slow release
        zi = np.array([coef * state])
        chunk_out, _ = sig.lfilter(b, a, chunk, zi=zi)
        smoothed[start:end] = chunk_out
        state = float(chunk_out[-1])

    return np.clip(smoothed, 0.0, 1.0).astype(gain_lin.dtype)


def _estimate_breathiness(audio: np.ndarray, sr: int) -> float:
    """Schätzt Breathiness anhand des Spektralabfalls (0.0 = klar, 1.0 = sehr hauchig).

    Proxy: Spektraler Slope (dB/Oktave) im Vokalbereich 500–6000 Hz.
    Slope ≈ −6 dB/Okt → normalsprachlich (breathiness=0.0)
    Slope < −18 dB/Okt → stark hauchig (breathiness→1.0)
    """
    mono = audio if audio.ndim == 1 else (audio[:, 0] if audio.ndim == 2 else audio.mean(axis=0))
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    n_fft = min(4096, len(mono))
    if n_fft < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= 500.0) & (freqs <= 6000.0) & (spectrum > 1e-10)
    if mask.sum() < 4:
        return 0.0
    log_freqs = np.log2(freqs[mask] + 1e-6)
    log_amps = 20.0 * np.log10(spectrum[mask] + 1e-10)
    slope = float(np.polyfit(log_freqs, log_amps, 1)[0])  # dB/Oktave
    # slope ≈ −6 dB/Okt → Schwellwert; slope < −18 dB/Okt → Maximalhauchigkeit
    return float(np.clip((-slope - 6.0) / 12.0, 0.0, 1.0))


def _deess_channel(
    ch: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    freq_low: float,
    freq_high: float,
    strength_cap: float = 1.0,
) -> tuple[np.ndarray, float]:
    """De-Esser auf einem Mono-Kanal. Gibt (processed, avg_gain_reduction_db) zurück.

    Sibilantenband-Extraktion via sosfiltfilt (Zero-Phase / §4.5) — vermeidet
    Phasenversatz zwischen Original und Bandpass-Signal.
    """
    # 1. Sibilantenband — Zero-Phase-Filter (sosfiltfilt, offline-verarbeitung)
    sos = sig.butter(4, [freq_low, freq_high], btype="band", fs=sr, output="sos")
    try:
        sib_band = sig.sosfiltfilt(sos, ch)
    except ValueError:
        # Fallback für sehr kurze Signale (< filter-Transiente)
        sib_band = sig.sosfilt(sos, ch)

    # 2. Hüllkurve
    envelope = _rms_envelope(sib_band, sr, 5.0)

    # 3. Gain Reduction (linker Arm: über Schwelle → komprimieren)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    gr = np.where(
        envelope > threshold_lin,
        (threshold_lin / (envelope + 1e-12)) ** ((ratio - 1.0) / ratio),
        1.0,
    )

    # 4. Smooth
    gr_smooth = _smooth_gain(gr, sr, attack_ms, release_ms)

    # 5. Strength-Cap (§2.19.3): GR darf nicht stärker als strength_cap
    #    strength_cap = 1.0 → kein Cap; = 0.45 → max. 55 % GR
    if strength_cap < 1.0:
        gr_smooth = np.maximum(gr_smooth, strength_cap)

    # 6. Anwenden: Sibilantenband dämpfen, zum Restsignal addieren
    processed = ch - sib_band + sib_band * gr_smooth

    avg_gr_db = float(np.mean(20.0 * np.log10(gr_smooth + 1e-12)))
    return processed.astype(ch.dtype), avg_gr_db


def _band_rms(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Return RMS in a frequency band (zero-phase when possible)."""
    if audio.ndim == 2:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    mono = np.nan_to_num(mono.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    nyq = sr / 2.0
    high_hz = min(high_hz, nyq * 0.98)
    low_hz = min(max(low_hz, 20.0), high_hz * 0.9)
    sos = sig.butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    try:
        band = sig.sosfiltfilt(sos, mono)
    except ValueError:
        band = sig.sosfilt(sos, mono)
    return float(np.sqrt(np.mean(band**2) + 1e-12))


def _overall_rms(audio: np.ndarray) -> float:
    """Return overall RMS for mono or stereo."""
    x = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _try_mp_senet_refine(audio: np.ndarray, sr: int) -> tuple[np.ndarray | None, str]:
    """Try MP-SENet refinement with ml_memory_budget guard. Returns (audio_or_none, model_used)."""
    # §2.47 ml_memory_budget guard (250 MB for MP-SENet)
    _dfn_release = None
    try:
        from backend.core.ml_memory_budget import try_allocate as _try_alloc_43, release as _rel_43
        if not _try_alloc_43("MpSeNet_phase43", 0.25):
            logger.debug("MP-SENet phase_43: ml_memory_budget insufficient — DSP-Fallback")
            return None, "unavailable"
        _dfn_release = _rel_43
    except ImportError:
        pass  # budget tracking unavailable — allow inference
    try:
        from plugins.mp_senet_plugin import get_mp_senet_plugin

        plugin = get_mp_senet_plugin()
        result = plugin.enhance(audio, sr)
        return result.audio, result.model_used
    except Exception as exc:
        logger.debug("Phase 43 MP-SENet refinement unavailable: %s", exc)
        return None, "unavailable"
    finally:
        if _dfn_release is not None:
            _dfn_release("MpSeNet_phase43")


class AdaptiveDeEsserPhase(PhaseInterface):
    """Stimmtyp-adaptiver Hybrid-De-Esser (DSP primär + optional ML refinement, §2.8)."""

    PHASE_ID = "phase_43_ml_deesser"
    PHASE_NAME = "Adaptive De-Esser (DSP+ML Hybrid, stimmtyp-adaptiv)"
    PHASE_DESCRIPTION = (
        "Split-Band De-Esser mit Butterworth-Bandpass gender-adaptiver Frequenzauswahl "
        "(§2.8: MALE 5–10 kHz / FEMALE 6–12 kHz / CHILD 7–14 kHz). "
        "RMS-Hüllkurve, Gain-Reduction 1:4, Attack 2 ms / Release 80 ms, Strength-Cap. "
        "Optional: MP-SENet refinement mit Sicherheits-Gate (nur bei nachgewiesener Verbesserung)."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.ENHANCEMENT,
            priority=6,
            version="2.2.0",
            dependencies=[],
            estimated_time_factor=0.04,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.88,
            description=self.PHASE_DESCRIPTION,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        De-Essing: Sibilanten reduzieren (stimmtyp-adaptiv, §2.8).

        Args:
            audio:        Mono oder Stereo float32/64
            sample_rate:  Hz (muss 48 000 sein)
            **kwargs:
                gender        (str)   "male"|"female"|"child"|"unknown"
                threshold_db  (float) Detektionsschwelle dBFS, Default -20.0
                ratio         (float) Kompressionsverhältnis, Default 4.0
                attack_ms     (float) Attack in ms, Default 2.0
                release_ms    (float) Release in ms, Default 80.0
                strength_cap  (float) Max. GR-Stärke 0–1: §2.19.3 Schlager 0.45
                freq_low      (float) Überschreibt gender-Auswahl (Hz)
                freq_high     (float) Überschreibt gender-Auswahl (Hz)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.astype(audio.dtype),
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"avg_gain_reduction_db": 0.0},
            )

        # Parameter
        gender: str = str(kwargs.get("gender", _DEFAULT_GENDER)).lower()
        threshold_db: float = float(kwargs.get("threshold_db", _DEFAULT_THRESHOLD_DB))
        ratio: float = float(kwargs.get("ratio", _DEFAULT_RATIO))
        ratio = float(1.0 + (ratio - 1.0) * _effective_strength)
        attack_ms: float = float(kwargs.get("attack_ms", _DEFAULT_ATTACK_MS))
        release_ms: float = float(kwargs.get("release_ms", _DEFAULT_RELEASE_MS))
        strength_cap: float = float(kwargs.get("strength_cap", _DEFAULT_STRENGTH_CAP))
        strength_cap = float(np.clip(strength_cap, 0.0, 1.0))
        strength_cap = float(max(strength_cap, 1.0 - 0.55 * _effective_strength))

        # Stimmtyp-adaptive Frequenzauswahl (§2.8); explizite freq_low/freq_high überschreiben
        default_low, default_high = GENDER_FREQ_MAP.get(gender, GENDER_FREQ_MAP["unknown"])
        # §2.36a PhonemeTimeline: language-specific sibilant band overrides gender-freq defaults
        _ptl_43 = kwargs.get("phoneme_timeline")
        if _ptl_43 is not None:
            try:
                _ptl_low, _ptl_high = _ptl_43.sibilant_band_hz()
                default_low = float(_ptl_low)
                default_high = float(_ptl_high)
                logger.debug(
                    "Phase 43: sibilant_band_hz override → %.0f–%.0f Hz (language=%s)",
                    default_low,
                    default_high,
                    getattr(_ptl_43, "language", "?"),
                )
            except Exception as _ptl_exc:
                logger.debug("Phase 43: sibilant_band_hz fallback to gender-based: %s", _ptl_exc)
        freq_low: float = float(kwargs.get("freq_low", default_low))
        freq_high: float = float(kwargs.get("freq_high", default_high))

        # Nyquist-Sicherung
        nyquist = sample_rate / 2.0
        freq_high = min(freq_high, nyquist * 0.98)
        freq_low = min(freq_low, freq_high * 0.90)

        x = audio.astype(np.float64)

        # Breathiness-Guard (§2.8): Hauchige Stimmen (spectral slope < −18 dB/Okt)
        # dürfen nicht über-de-esst werden. Bei breathiness > 0.4 wird strength_cap
        # auf max. 0.50–0.60 begrenzt, um hauchige Vokale natürlich zu erhalten.
        _breathiness = _estimate_breathiness(x, sample_rate)
        if _breathiness > 0.4:
            _breath_cap = float(np.clip(0.60 - (_breathiness - 0.4) * 0.10, 0.50, 1.0))
            strength_cap = max(strength_cap, _breath_cap)
            logger.info(
                "Phase 43 Breathiness-Guard: breathiness=%.2f → strength_cap angepasst auf %.2f",
                _breathiness,
                strength_cap,
            )

        gr_dbs: list[float] = []

        if x.ndim == 1:
            processed_ch, gr_db = _deess_channel(
                x,
                sample_rate,
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                freq_low,
                freq_high,
                strength_cap,
            )
            processed = processed_ch
            gr_dbs.append(gr_db)
        else:
            # §2.51 Linked-Stereo: Sibilanz-Detektion auf Mono-Mix, identische GR auf L+R
            mono_mix = np.mean(x, axis=1)
            _mono_deessed, gr_db_linked = _deess_channel(
                mono_mix,
                sample_rate,
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                freq_low,
                freq_high,
                strength_cap,
            )
            # Compute linked gain from mono
            _eps_ds = 1e-10
            _gain_ds = np.where(
                np.abs(mono_mix) > _eps_ds,
                _mono_deessed / (mono_mix + _eps_ds * np.sign(mono_mix + _eps_ds)),
                1.0,
            )
            _gain_ds = np.clip(_gain_ds, 0.0, 10.0)
            processed = np.column_stack([x[:, ch] * _gain_ds for ch in range(x.shape[1])])
            gr_dbs.append(gr_db_linked)

        if 0.0 < _effective_strength < 1.0 and processed.shape == x.shape:
            processed = x + _effective_strength * (processed - x)

        processed = np.clip(processed, -1.0, 1.0).astype(audio.dtype)
        avg_gr = float(np.mean(gr_dbs))

        # §2.36a Segment-selective gate: apply de-essing only within sibilant_segments() windows.
        # Non-sibilant regions revert to the original signal so harmonic vowel and instrumental
        # passages are not inadvertently de-essed (iZotope RX-class time-domain gating).
        _ptl_gate43 = kwargs.get("phoneme_timeline")
        if _ptl_gate43 is not None:
            _sib_segs43 = _ptl_gate43.sibilant_segments()
            if _sib_segs43:
                _n43 = x.shape[0] if x.ndim >= 1 else len(x)
                _gate43 = np.zeros(_n43, dtype=np.float32)
                _fade43 = max(2, int(sample_rate * 0.005))  # 5 ms cosine fade
                for _seg43 in _sib_segs43:
                    _s43 = max(0, int(_seg43.start_s * sample_rate))
                    _e43 = min(_n43, int(_seg43.end_s * sample_rate))
                    if _e43 <= _s43:
                        continue
                    _gate43[_s43:_e43] = 1.0
                    _fi43 = min(_fade43, _e43 - _s43)
                    _gate43[_s43 : _s43 + _fi43] = np.sin(np.linspace(0.0, np.pi / 2.0, _fi43)) ** 2
                    _fo43 = min(_fade43, _e43 - _s43)
                    _gate43[_e43 - _fo43 : _e43] = np.cos(np.linspace(0.0, np.pi / 2.0, _fo43)) ** 2
                _x_ref43 = x.astype(processed.dtype)
                if processed.ndim == 2:
                    _mask43_2d = _gate43[:, np.newaxis]
                    processed = (_mask43_2d * processed + (1.0 - _mask43_2d) * _x_ref43).astype(audio.dtype)
                else:
                    processed = (_gate43 * processed + (1.0 - _gate43) * _x_ref43).astype(audio.dtype)
                logger.debug(
                    "Phase 43 segment-gate: %d sibilant windows, %.1f%% gated",
                    len(_sib_segs43),
                    100.0 * float(np.mean(_gate43)),
                )

        # Optional ML refinement with strict safety guard:
        # accept only if sibilance reduces and vocal core band is preserved.
        ml_refine_applied = False
        ml_refine_bypassed = False
        ml_refine_model = "disabled"
        ml_blend = 0.0
        ml_refine_bypass_reason = "disabled"
        if bool(kwargs.get("enable_ml_refine", True)):
            ml_candidate, ml_refine_model = _try_mp_senet_refine(processed, sample_rate)
            ml_refine_bypass_reason = "unavailable" if ml_candidate is None else "shape_mismatch"
            if ml_refine_model != "mp_senet_onnx":
                ml_refine_bypassed = True
                ml_refine_bypass_reason = ml_refine_model
            elif ml_candidate is not None and ml_candidate.shape == processed.shape:
                sibilance_before = _band_rms(processed, sample_rate, freq_low, freq_high)
                sibilance_after = _band_rms(ml_candidate, sample_rate, freq_low, freq_high)
                vocal_core_before = _band_rms(processed, sample_rate, 300.0, 3000.0)
                vocal_core_after = _band_rms(ml_candidate, sample_rate, 300.0, 3000.0)

                rms_before = _overall_rms(processed)
                rms_after = _overall_rms(ml_candidate)
                rms_delta_db = float(20.0 * np.log10((rms_after + 1e-12) / (rms_before + 1e-12)))

                sibilance_improvement = float((sibilance_before - sibilance_after) / (sibilance_before + 1e-12))
                core_ratio = float((vocal_core_after + 1e-12) / (vocal_core_before + 1e-12))

                # Acceptance criteria tuned conservative to avoid musical-goal regressions.
                # 1) Sibilance must improve by at least 2%
                # 2) Vocal core band must not lose more than 3%
                # 3) Overall loudness shift must stay within +-1.0 dB
                if sibilance_improvement >= 0.02 and core_ratio >= 0.97 and abs(rms_delta_db) <= 1.0:
                    # Adaptive blend: stronger blend only with stronger measured improvement.
                    ml_blend = float(np.clip(0.10 + 0.80 * sibilance_improvement, 0.10, 0.35))
                    ml_blend *= _effective_strength
                    processed = processed + ml_blend * (ml_candidate - processed)
                    processed = np.clip(np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    ml_refine_applied = True
                    ml_refine_bypass_reason = "applied"
                else:
                    ml_refine_bypassed = True
                    ml_refine_bypass_reason = "safety_gate"
                    logger.debug(
                        "Phase 43 ML refinement rejected: sib_impr=%.3f core_ratio=%.3f rms_delta_db=%.2f",
                        sibilance_improvement,
                        core_ratio,
                        rms_delta_db,
                    )

        logger.info(
            "Phase 43 DeEsser: gender=%s freq=[%.0f–%.0f Hz] "
            "threshold=%.1f dB ratio=%.1f strength_cap=%.2f avg_GR=%.2f dB",
            gender,
            freq_low,
            freq_high,
            threshold_db,
            ratio,
            strength_cap,
            avg_gr,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "gender": gender,
                "threshold_db": threshold_db,
                "ratio": ratio,
                "attack_ms": attack_ms,
                "release_ms": release_ms,
                "freq_low_hz": freq_low,
                "freq_high_hz": freq_high,
                "strength_cap": strength_cap,
                "ml_refine_enabled": bool(kwargs.get("enable_ml_refine", True)),
                "ml_refine_applied": ml_refine_applied,
                "ml_refine_bypassed": ml_refine_bypassed,
                "ml_refine_bypass_reason": ml_refine_bypass_reason,
                "ml_refine_model": ml_refine_model,
                "ml_refine_blend": ml_blend,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={"avg_gain_reduction_db": avg_gr},
        )


class MLDeEsserPhase(AdaptiveDeEsserPhase):
    """Backward-compatible alias for older imports/tests."""
