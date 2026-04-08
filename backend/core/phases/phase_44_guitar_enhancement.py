"""
Phase 44: Guitar Enhancement v2.0 — Transient Boost + Harmonic Exciter + Presence EQ
======================================================================================

Korrektur des alten Stubs: falscher Rückgabetyp (rohes ndarray statt PhaseResult),
defekte Local-Imports (np nicht in Scope in Helper-Methoden), kein aurik_ml.

ALGORITHMUS:
  1. Spektralzentroid-basierte Genre-Einstufung (Rock / Jazz / Pop)
     → amplitude-gewichteter Schwerpunkt des Magnitude-Spektrums

  2. Transienten-Boost:
     - Hilbert-Hüllkurve × 0.15 zum Signal addieren
     - Betont Anschlag/Pick-Transiente

  3. Genre-adaptiver Harmonic Exciter:
     - Rock:  Soft-Clip via tanh (ungerade Harmonics, Gain 0.12)
     - Jazz:  Subtile Sinus-Verzerrung (Gain 0.07)
     - Pop:   Vollwellengleichrichter (gerade Harmonics, Gain 0.09)

  4. Presence-EQ:
     - Rock/Pop: +2 dB Peak bei 3 kHz, Q=1.8
     - Jazz:     +1 dB Peak bei 1.5 kHz, Q=2.5

  5. Normalisierungs-Pass, np.clip, PhaseResult

Author: Aurik Development Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls

    _FORMANT_SYSTEM_GUITAR = None
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_GUITAR = None

logger = logging.getLogger(__name__)


def _spectral_centroid(audio: np.ndarray) -> float:
    """Amplitudengewichteter Spektralzentroid (normiert 0–1)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mag = np.abs(np.fft.rfft(mono[: min(len(mono), 65536)]))
    total = mag.sum() + 1e-12
    freqs = np.arange(len(mag), dtype=np.float64)
    return float((freqs * mag).sum() / total / len(mag))


def _peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Biquad Peaking-EQ (Audio-EQ-Cookbook)."""
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = b1
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    if x.ndim == 1:
        return sig.lfilter(b, a, x)
    return np.column_stack([sig.lfilter(b, a, x[:, ch]) for ch in range(x.shape[1])])


class GuitarEnhancementPhase(PhaseInterface):
    """Transient-Boost + genre-adaptiver Harmonic Exciter + Presence-EQ für Gitarre."""

    PHASE_ID = "phase_44_guitar_enhancement"
    PHASE_NAME = "Guitar Enhancement (Transient + Exciter + EQ)"
    PHASE_DESCRIPTION = (
        "Genre-adaptive Gitarren-Verbesserung: Hilbert-Transient-Boost, "
        "Harmonic Exciter (tanh/sin/abs je nach Genre) und Presence-EQ. "
        "Kein aurik_ml, vollständig DSP."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name=self.PHASE_NAME,
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.03,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.82,
            description=self.PHASE_DESCRIPTION,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Guitar Enhancement: Transient + Exciter + EQ.

        Args:
            audio:        Mono oder Stereo
            sample_rate:  Hz
            **kwargs:     transient_gain (float, default 0.15)
                          exciter_gain   (float, default 1.0 = Genre-Default)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        phase_locality_factor: float = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength: float = float(kwargs.get("strength", 1.0))
        _effective_strength: float = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

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
                metrics={"spectral_centroid": 0.0},
            )

        transient_gain: float = float(kwargs.get("transient_gain", 0.15))
        exciter_gain: float = float(kwargs.get("exciter_gain", 1.0))
        transient_gain = float(transient_gain * _effective_strength)
        exciter_gain = float(exciter_gain * _effective_strength)

        x = audio.astype(np.float64)
        mono = x.mean(axis=1) if x.ndim == 2 else x

        # 1. Genre-Klassifikation via Spektralzentroid + Crest Factor
        centroid = _spectral_centroid(audio)
        rms = float(np.sqrt(np.mean(mono**2)))
        peak = float(np.max(np.abs(mono)))
        crest_db = 20.0 * np.log10(peak / (rms + 1e-10) + 1e-10)
        # Rock: bright centroid + moderate crest (distorted pickups)
        # Jazz: dark centroid + high crest (clean dynamics)
        # Pop: mid centroid
        if centroid > 0.40 and crest_db < 20.0:
            genre = "Rock"
        elif centroid < 0.28 and crest_db > 16.0:
            genre = "Jazz"
        else:
            genre = "Pop"

        peak_in = float(np.max(np.abs(audio)))

        # 2. Transient-Boost via attack-envelope differentiation (NOT Hilbert envelope)
        # True transient boost: detect fast amplitude rises and amplify the onset
        # HP-filter → envelope derivative → onset mask → mix scaled signal
        sos_hp = sig.butter(2, 200.0, btype="high", fs=sample_rate, output="sos")

        def _transient_boost_channel(ch_audio: np.ndarray) -> np.ndarray:
            hp = sig.sosfilt(sos_hp, ch_audio)
            # Rectified envelope
            env = np.abs(hp)
            smooth_win = max(1, int(0.003 * sample_rate))  # 3ms smoothing
            env_smooth = np.convolve(env, np.ones(smooth_win) / smooth_win, mode="same")
            # Attack detector: positive derivative of smoothed envelope
            d_env = np.diff(env_smooth, prepend=env_smooth[0])
            attack = np.maximum(0.0, d_env)
            # Normalize attack to [0, 1]
            a_max = float(np.max(attack))
            if a_max > 1e-10:
                attack = attack / a_max
            # Boost signal at attack moments (wet/dry)
            return ch_audio * (1.0 + transient_gain * attack)

        if x.ndim == 1:
            x = _transient_boost_channel(x)
        else:
            for ch in range(x.shape[1]):
                x[:, ch] = _transient_boost_channel(x[:, ch])

        # 3. Genre-adaptiver Harmonic Exciter (band-limited to body resonance region)
        # Limit exciter to 200-5000 Hz to avoid mud (< 200 Hz) and hash (> 5 kHz)
        sos_body = sig.butter(4, [200.0, 5000.0], btype="band", fs=sample_rate, output="sos")
        g = exciter_gain

        def _excite_channel(ch_audio: np.ndarray) -> np.ndarray:
            body_band = sig.sosfilt(sos_body, ch_audio)
            if genre == "Rock":
                # Soft-clip for odd harmonics (pick attack grit)
                excited = np.tanh(body_band * 2.5) * 0.12 * g
            elif genre == "Jazz":
                # Subtle even harmonics (warmth, not distortion)
                excited = (np.abs(body_band) - body_band**2 * 0.5) * 0.07 * g
            else:  # Pop
                # Mild saturation
                excited = np.tanh(body_band * 1.5) * 0.09 * g
            return ch_audio + excited

        if x.ndim == 1:
            x = _excite_channel(x)
        else:
            for ch in range(x.shape[1]):
                x[:, ch] = _excite_channel(x[:, ch])

        # 4. Presence-EQ (genre-adaptive)
        if genre == "Rock":
            # Pick attack clarity at 3 kHz, string bite at 5 kHz
            x = _peaking_eq(x, sample_rate, freq=3000.0, gain_db=2.0, q=1.8)
            x = _peaking_eq(x, sample_rate, freq=5000.0, gain_db=1.2, q=2.5)
        elif genre == "Jazz":
            # Warm body at 800 Hz, subtle presence at 2 kHz
            x = _peaking_eq(x, sample_rate, freq=800.0, gain_db=1.0, q=2.0)
            x = _peaking_eq(x, sample_rate, freq=2000.0, gain_db=1.0, q=2.5)
        else:  # Pop
            x = _peaking_eq(x, sample_rate, freq=3000.0, gain_db=1.5, q=2.0)

        # 5. Normalisierung + Clip — §2.49 Peak-Guard: percentile(99.9) for peak-preservation
        peak_in = float(np.percentile(np.abs(audio), 99.9))
        peak_out = float(np.percentile(np.abs(x), 99.9))
        if peak_out > 1e-8 and peak_in > 1e-8:
            x = x * (peak_in / peak_out)
        processed = np.clip(x, -1.0, 1.0).astype(audio.dtype)

        logger.info(
            "Phase 44 GuitarEnhancement: genre=%s, centroid=%.3f, transient_gain=%.2f",
            genre,
            centroid,
            transient_gain,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # Instrument-guided formant enhancement (InstrumentFormantTargets — guitar body resonances)
        igt_frames = 0
        try:
            global _FORMANT_SYSTEM_GUITAR
            if _FormantSystemCls is not None:
                if _FORMANT_SYSTEM_GUITAR is None:
                    _FORMANT_SYSTEM_GUITAR = _FormantSystemCls(enhance_singers_formant=False)
                formant_system = _FORMANT_SYSTEM_GUITAR
                if formant_system is not None:
                    processed, igt_report = formant_system.instrument_guided_enhance(
                        processed, sample_rate, instrument="guitar", correction_strength=0.20
                    )
                    igt_frames = igt_report.get("frames_processed", 0)
                    logger.debug("Phase 44 InstrumentFormant: guitar frames=%d", igt_frames)
        except Exception as _igt_exc:
            logger.debug("Phase 44 instrument_guided_enhance skipped: %s", _igt_exc)

        # Formant-Drift-Korrektur via DTW (Schritt 3)
        try:
            from dsp.instrument_formant_corrector import correct_instrument_formant_drift

            drift_result = correct_instrument_formant_drift(processed, sample_rate, instrument="guitar")
            processed = drift_result.audio
            logger.debug(
                "Phase 44 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                drift_result.drift_detected,
                drift_result.n_frames_corrected,
                drift_result.total_frames,
                drift_result.mean_drift_hz,
            )
        except Exception as _drift_exc:
            logger.debug("Phase 44 drift correction skipped: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            from backend.core.sub_stem_processor import process_sub_stems

            ss_result = process_sub_stems(processed, sample_rate, instrument="guitar", processing_strength=0.35)
            processed = ss_result.audio
            logger.debug("Phase 44 sub-stem: bands=%d strength=%.2f", ss_result.n_bands, ss_result.processing_strength)
        except Exception as _ss_exc:
            logger.debug("Phase 44 sub-stem skipped: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            from backend.core.physics_resonance_enhancer import enhance_physics_resonance

            pr_result = enhance_physics_resonance(
                processed, sample_rate, instrument="guitar", enhancement_strength=0.40
            )
            processed = pr_result.audio
            logger.debug(
                "Phase 44 physics resonance: peaks=%d strength=%.2f", pr_result.n_peaks, pr_result.enhancement_strength
            )
        except Exception as _pr_exc:
            logger.debug("Phase 44 physics resonance skipped: %s", _pr_exc)

        if 0.0 < _effective_strength < 1.0 and processed.shape == audio.shape:
            processed = audio + _effective_strength * (processed - audio)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "genre": genre,
                "spectral_centroid": centroid,
                "transient_gain": transient_gain,
                "instrument_formant_frames": igt_frames,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={"genre": genre, "spectral_centroid": centroid},
        )
