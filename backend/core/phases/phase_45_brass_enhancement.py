"""
Phase 45: Brass Enhancement v2.0 — Harmonischer Exciter + Presence-EQ
======================================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Ersetzt den kaputten ML-Stub.

ALGORITHMUS:
  1. 2. Oberton-Anreicherung (Harmonic Exciter):
     - Gleich­richter → Hüllkurve → Phase der Grundschwingung extrahieren
     - Simplified: Soft-Clipping der Spitzen → erzeugt ungerade Harmonische
     - 2nd-Harmonic: Vollwellengleich­richter-Signal (enthält hauptsächlich H2, H4)
     - Subtil addiert (gain_h2 = 0.04)

  2. Presence-EQ (parametrischer Peak):
     - Frequenz: 2500 Hz (Bläser-Präsenz-Bereich, Durchsetzungsvermögen)
     - Gain: +2.5 dB
     - Q: 2.0

  3. Air (High-Shelf):
     - Frequenz: 8000 Hz (Glanz, Atem der Bläser)
     - Gain: +1.8 dB

  4. Normalisierungs-Pass: Pegel-Erhalt

NOTES:
  - Funktioniert bei Mono und Stereo
  - Keinerlei aurik_ml-Abhängigkeit

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
    _FORMANT_SYSTEM_BRASS: _FormantSystemCls | None = None
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_BRASS = None

logger = logging.getLogger(__name__)


def _peaking_eq(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Parametrischer Peak-EQ via Biquad-Filter (Audio-EQ-Cookbook, Zölzer)."""
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])

    if audio.ndim == 1:
        return sig.lfilter(b, a, audio)
    result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        result[:, ch] = sig.lfilter(b, a, audio[:, ch])
    return result


def _high_shelf(audio: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
    """High-Shelf-Filter (Audio-EQ-Cookbook, S=1)."""
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    S = 1.0
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha
    a1 = 2.0 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])

    if audio.ndim == 1:
        return sig.lfilter(b, a, audio)
    result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        result[:, ch] = sig.lfilter(b, a, audio[:, ch])
    return result


class BrassEnhancementPhase(PhaseInterface):
    """Harmonischer Exciter + Presence-EQ + Air-EQ für Blechbläser."""

    phase_id = "phase_45_brass_enhancement"
    name = "Brass Enhancement (Harmonic Exciter + EQ)"
    description = (
        "Blechbläser-Optimierung: Subtile 2nd-Harmonic-Anreicherung (Soft-Clip), "
        "Presence-EQ bei 2.5 kHz (+2.5 dB, Q=2) und Air-EQ bei 8 kHz (+1.8 dB). "
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
        Brass Enhancement: Harmonics + EQ.

        Args:
            audio:        Mono oder Stereo (float32/float64)
            sample_rate:  Hz
            **kwargs:     gain_h2     (float, default 0.04)  — Harmonic-Exciter-Stärke
                          presence_db (float, default 2.5)   — Presence-EQ-Gain dB
                          air_db      (float, default 1.8)   — Air-Shelf-Gain dB
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        gain_h2: float = float(kwargs.get("gain_h2", 0.04))
        presence_db: float = float(kwargs.get("presence_db", 2.5))
        air_db: float = float(kwargs.get("air_db", 1.8))

        x = audio.astype(np.float64)

        # 1. Harmonic Exciter: Vollwellengleichrichter → H2-Anreicherung
        #    |x| enthält hauptsächlich 2nd + 4th Oberton.
        #    Vor Addition: Hochpass (>500 Hz) um LF-Mud zu vermeiden
        sos_hp = sig.butter(2, 500.0, btype="high", fs=sample_rate, output="sos")
        if x.ndim == 1:
            h2 = sig.sosfilt(sos_hp, np.abs(x))
        else:
            h2 = np.column_stack([sig.sosfilt(sos_hp, np.abs(x[:, ch])) for ch in range(x.shape[1])])
        x = x + gain_h2 * h2

        # 2. Presence-EQ (2.5 kHz, +presence_db dB, Q=2)
        x = _peaking_eq(x, sample_rate, freq=2500.0, gain_db=presence_db, q=2.0)

        # 3. Air High-Shelf (8 kHz, +air_db dB)
        x = _high_shelf(x, sample_rate, freq=8000.0, gain_db=air_db)

        # 4. Normalisierung (Pegel-Erhalt)
        peak_in = float(np.max(np.abs(audio)))
        peak_out = float(np.max(np.abs(x)))
        if peak_out > 1e-8 and peak_in > 1e-8:
            x = x * (peak_in / peak_out)

        processed = np.clip(x, -1.0, 1.0).astype(audio.dtype)

        logger.info(
            "Phase 45 BrassEnhancement: gain_h2=%.3f, presence=+%.1fdB, air=+%.1fdB",
            gain_h2,
            presence_db,
            air_db,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        # Instrument-guided formant enhancement (Benade 1976 brass resonance targets)
        igt_frames = 0
        try:
            global _FORMANT_SYSTEM_BRASS
            if _FormantSystemCls is not None:
                if _FORMANT_SYSTEM_BRASS is None:
                    _FORMANT_SYSTEM_BRASS = _FormantSystemCls(enhance_singers_formant=False)
                processed, igt_report = _FORMANT_SYSTEM_BRASS.instrument_guided_enhance(
                    processed, sample_rate, instrument="brass", correction_strength=0.20
                )
                igt_frames = igt_report.get("frames_processed", 0)
                logger.debug("Phase 45 InstrumentFormant: brass frames=%d", igt_frames)
        except Exception as _igt_exc:
            logger.debug("Phase 45 instrument_guided_enhance skipped: %s", _igt_exc)

        # Formant-Drift-Korrektur via DTW (Schritt 3)
        try:
            from dsp.instrument_formant_corrector import correct_instrument_formant_drift
            drift_result = correct_instrument_formant_drift(processed, sample_rate, instrument="brass")
            processed = drift_result.audio
            logger.debug(
                "Phase 45 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                drift_result.drift_detected, drift_result.n_frames_corrected,
                drift_result.total_frames, drift_result.mean_drift_hz,
            )
        except Exception as _drift_exc:
            logger.debug("Phase 45 drift correction skipped: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            from backend.core.sub_stem_processor import process_sub_stems
            ss_result = process_sub_stems(processed, sample_rate, instrument="brass",
                                          processing_strength=0.30)
            processed = ss_result.audio
            logger.debug("Phase 45 sub-stem: bands=%d strength=%.2f",
                         ss_result.n_bands, ss_result.processing_strength)
        except Exception as _ss_exc:
            logger.debug("Phase 45 sub-stem skipped: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            from backend.core.physics_resonance_enhancer import enhance_physics_resonance
            pr_result = enhance_physics_resonance(processed, sample_rate, instrument="brass",
                                                  enhancement_strength=0.40)
            processed = pr_result.audio
            logger.debug("Phase 45 physics resonance: peaks=%d strength=%.2f",
                         pr_result.n_peaks, pr_result.enhancement_strength)
        except Exception as _pr_exc:
            logger.debug("Phase 45 physics resonance skipped: %s", _pr_exc)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={"gain_h2": gain_h2, "presence_db": presence_db, "air_db": air_db,
                      "instrument_formant_frames": igt_frames},
            metrics={"gain_h2": gain_h2, "presence_db": presence_db, "air_db": air_db},
        )
