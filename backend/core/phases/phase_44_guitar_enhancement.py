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
    _FORMANT_SYSTEM_GUITAR: _FormantSystemCls | None = None
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

    phase_id = "phase_44_guitar_enhancement"
    name = "Guitar Enhancement (Transient + Exciter + EQ)"
    description = (
        "Genre-adaptive Gitarren-Verbesserung: Hilbert-Transient-Boost, "
        "Harmonic Exciter (tanh/sin/abs je nach Genre) und Presence-EQ. "
        "Kein aurik_ml, vollständig DSP."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.03,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.82,
            description=self.description,
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

        transient_gain: float = float(kwargs.get("transient_gain", 0.15))
        exciter_gain: float = float(kwargs.get("exciter_gain", 1.0))

        x = audio.astype(np.float64)
        mono = x.mean(axis=1) if x.ndim == 2 else x

        # 1. Genre-Klassifikation via Spektralzentroid
        centroid = _spectral_centroid(audio)
        rms = float(np.sqrt(np.mean(mono**2)))
        if centroid > 0.45 and rms > 0.08:
            genre = "Rock"
        elif centroid < 0.25:
            genre = "Jazz"
        else:
            genre = "Pop"

        # 2. Transienten-Boost via Hilbert-Hüllkurve
        if x.ndim == 1:
            env = np.abs(sig.hilbert(x))
            x = x + transient_gain * env
        else:
            for ch in range(x.shape[1]):
                env = np.abs(sig.hilbert(x[:, ch]))
                x[:, ch] = x[:, ch] + transient_gain * env

        # 3. Genre-adaptiver Harmonic Exciter
        g = exciter_gain
        if genre == "Rock":
            x = x + 0.12 * g * np.tanh(x * 2.5)
        elif genre == "Jazz":
            x = x + 0.07 * g * np.sin(x * np.pi)
        else:  # Pop
            x = x + 0.09 * g * np.abs(x)

        # 4. Presence-EQ
        if genre == "Rock" or genre == "Pop":
            x = _peaking_eq(x, sample_rate, freq=3000.0, gain_db=2.0, q=1.8)
        else:
            x = _peaking_eq(x, sample_rate, freq=1500.0, gain_db=1.0, q=2.5)

        # 5. Normalisierung + Clip
        peak_in = float(np.max(np.abs(audio)))
        peak_out = float(np.max(np.abs(x)))
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
                processed, igt_report = _FORMANT_SYSTEM_GUITAR.instrument_guided_enhance(
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
                drift_result.drift_detected, drift_result.n_frames_corrected,
                drift_result.total_frames, drift_result.mean_drift_hz,
            )
        except Exception as _drift_exc:
            logger.debug("Phase 44 drift correction skipped: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            from backend.core.sub_stem_processor import process_sub_stems
            ss_result = process_sub_stems(processed, sample_rate, instrument="guitar",
                                          processing_strength=0.35)
            processed = ss_result.audio
            logger.debug("Phase 44 sub-stem: bands=%d strength=%.2f",
                         ss_result.n_bands, ss_result.processing_strength)
        except Exception as _ss_exc:
            logger.debug("Phase 44 sub-stem skipped: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            from backend.core.physics_resonance_enhancer import enhance_physics_resonance
            pr_result = enhance_physics_resonance(processed, sample_rate, instrument="guitar",
                                                  enhancement_strength=0.40)
            processed = pr_result.audio
            logger.debug("Phase 44 physics resonance: peaks=%d strength=%.2f",
                         pr_result.n_peaks, pr_result.enhancement_strength)
        except Exception as _pr_exc:
            logger.debug("Phase 44 physics resonance skipped: %s", _pr_exc)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={"genre": genre, "spectral_centroid": centroid, "transient_gain": transient_gain,
                      "instrument_formant_frames": igt_frames},
            metrics={"genre": genre, "spectral_centroid": centroid},
        )
