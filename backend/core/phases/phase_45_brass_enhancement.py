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
from typing import Any

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import to_channels_last

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

try:
    from dsp.formant_system import FormantSystem as _FormantSystemCls

    _FORMANT_SYSTEM_BRASS: list[Any] = [None]  # mutable container — avoids global statement
except Exception:
    _FormantSystemCls = None  # type: ignore[assignment,misc]
    _FORMANT_SYSTEM_BRASS = [None]

try:
    from dsp.instrument_formant_corrector import correct_instrument_formant_drift as _correct_instrument_formant_drift
except Exception:
    _correct_instrument_formant_drift = None  # type: ignore[assignment]

try:
    from backend.core.sub_stem_processor import process_sub_stems as _process_sub_stems
except Exception:
    _process_sub_stems = None  # type: ignore[assignment]

try:
    from backend.core.physics_resonance_enhancer import enhance_physics_resonance as _enhance_physics_resonance
except Exception:
    _enhance_physics_resonance = None  # type: ignore[assignment]

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

    # §2.51 zero-phase: filtfilt eliminates causal group-delay → no L/R interchannel lag.
    if audio.ndim == 1:
        return sig.filtfilt(b, a, audio)
    result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        result[:, ch] = sig.filtfilt(b, a, audio[:, ch])
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

    # §2.51 zero-phase: filtfilt eliminates causal group-delay → no L/R interchannel lag.
    if audio.ndim == 1:
        return sig.filtfilt(b, a, audio)
    result = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        result[:, ch] = sig.filtfilt(b, a, audio[:, ch])
    return result


class BrassEnhancementPhase(PhaseInterface):
    """Harmonischer Exciter + Presence-EQ + Air-EQ für Blechbläser."""

    _PHASE_ID = "phase_45_brass_enhancement"
    _NAME = "Brass Enhancement (Harmonic Exciter + EQ)"
    description = (
        "Blechbläser-Optimierung: Subtile 2nd-Harmonic-Anreicherung (Soft-Clip), "
        "Presence-EQ bei 2.5 kHz (+2.5 dB, Q=2) und Air-EQ bei 8 kHz (+1.8 dB). "
        "Kein aurik_ml, vollständig DSP."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
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

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Brass Enhancement: Harmonics + EQ.

        Args:
            audio:        Mono oder Stereo (float32/float64)
            sample_rate:  Hz
            **kwargs:     gain_h2     (float, default 0.04)  — Harmonic-Exciter-Stärke
                          presence_db (float, default 2.5)   — Presence-EQ-Gain dB
                          air_db      (float, default 1.8)   — Air-Shelf-Gain dB
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        audio, _p45_transposed = to_channels_last(audio)
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
                metrics={"gain_h2": 0.0, "presence_db": 0.0, "air_db": 0.0},
            )

        gain_h2: float = float(kwargs.get("gain_h2", 0.04))
        presence_db: float = float(kwargs.get("presence_db", 2.5))
        air_db: float = float(kwargs.get("air_db", 1.8))
        gain_h2 = float(gain_h2 * _effective_strength)
        presence_db = float(presence_db * _effective_strength)
        air_db = float(air_db * _effective_strength)

        x = audio.astype(np.float64)

        # 1. Phase-Coherent Harmonic Exciter for Brass H2
        #    OLD (WRONG): |x| via full-wave rectifier → produces only even harmonics,
        #      BUT without preserving phase of the fundamental → wrong phase relationship,
        #      sounds phasey not warm
        #    NEW (CORRECT): Hilbert-based instantaneous phase → compute H2 with 2× phase
        #      x(t) = A(t)*cos(φ(t))  →  H2(t) = A(t)*cos(2φ(t))
        #      This inserts the second harmonic at the correct phase relationship to the fundamental.
        #    Band-limited to 500–4000 Hz HP-filtered copy to avoid LF mud.
        try:
            _hilbert = sig.hilbert
            # §2.51 Anti-Zeitversatz: sosfiltfilt — BP-gefiltertes Signal liefert Phase/Amplitude
            # für H2-Synthese; h2 wird auf x/mid aufaddiert; sosfilt erzeugt Zeitversatz.
            sos_bp = sig.butter(2, [300.0, 4000.0], btype="band", fs=sample_rate, output="sos")
            if x.ndim == 1:
                x_bp = sig.sosfiltfilt(sos_bp, x)
                _analytic = np.asarray(_hilbert(x_bp))
                amplitude = np.sqrt(_analytic.real**2 + _analytic.imag**2)
                phase = np.unwrap(np.arctan2(_analytic.imag, _analytic.real))
                h2 = amplitude * np.cos(2.0 * phase)
            else:
                # §2.51 M/S-Domain: H2-Synthese nur auf Mid (wie phase_07)
                mid = (x[:, 0] + x[:, 1]) / np.sqrt(2.0)
                side = (x[:, 0] - x[:, 1]) / np.sqrt(2.0)
                x_bp = sig.sosfiltfilt(sos_bp, mid)
                _a = np.asarray(_hilbert(x_bp))
                h2 = np.sqrt(_a.real**2 + _a.imag**2) * np.cos(2.0 * np.unwrap(np.arctan2(_a.imag, _a.real)))
                mid = mid + gain_h2 * h2
                x = np.column_stack(
                    [
                        (mid + side) / np.sqrt(2.0),
                        (mid - side) / np.sqrt(2.0),
                    ]
                )
            x = x + gain_h2 * h2
        except Exception:
            # Fallback: classic rectifier if hilbert fails (e.g. very short signal)
            # §2.51 Anti-Zeitversatz: sosfiltfilt — h2_fb wird auf x aufaddiert.
            sos_hp = sig.butter(2, 500.0, btype="high", fs=sample_rate, output="sos")
            if x.ndim == 1:
                h2_fb = sig.sosfiltfilt(sos_hp, np.abs(x))
            else:
                h2_fb = np.column_stack([sig.sosfiltfilt(sos_hp, np.abs(x[:, ch])) for ch in range(x.shape[1])])
            x = x + gain_h2 * h2_fb

        # 2. Presence-EQ (2.5 kHz, +presence_db dB, Q=2)
        x = _peaking_eq(x, sample_rate, freq=2500.0, gain_db=presence_db, q=2.0)

        # 3. Air High-Shelf (8 kHz, +air_db dB)
        x = _high_shelf(x, sample_rate, freq=8000.0, gain_db=air_db)

        # 4. Normalisierung (Pegel-Erhalt) — §2.49 Peak-Guard: percentile(99.9)
        peak_in = float(np.percentile(np.abs(audio), 99.9))
        peak_out = float(np.percentile(np.abs(x), 99.9))
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
            if _FormantSystemCls is not None:
                if _FORMANT_SYSTEM_BRASS[0] is None:
                    _FORMANT_SYSTEM_BRASS[0] = _FormantSystemCls(enhance_singers_formant=False)
                processed, igt_report = _FORMANT_SYSTEM_BRASS[0].instrument_guided_enhance(
                    processed, sample_rate, instrument="brass", correction_strength=0.20
                )
                igt_frames = igt_report.get("frames_processed", 0)
                logger.debug("Phase 45 InstrumentFormant: brass frames=%d", igt_frames)
        except Exception as _igt_exc:
            logger.debug("Phase 45 instrument_guided_enhance skipped: %s", _igt_exc)

        # Formant-Drift-Korrektur via DTW (Schritt 3)
        try:
            if _correct_instrument_formant_drift is not None:
                drift_result = _correct_instrument_formant_drift(processed, sample_rate, instrument="brass")
                processed = drift_result.audio
                logger.debug(
                    "Phase 45 drift correction: detected=%s frames=%d/%d drift=%.1fHz",
                    drift_result.drift_detected,
                    drift_result.n_frames_corrected,
                    drift_result.total_frames,
                    drift_result.mean_drift_hz,
                )
        except Exception as _drift_exc:
            logger.debug("Phase 45 drift correction skipped: %s", _drift_exc)

        # Sub-Stem-Verarbeitung (Schritt 4)
        try:
            if _process_sub_stems is not None:
                # SubStemProcessor MAX_STRENGTH=0.60; 0.55 for brass (bright harmonics — conservative)
                # outer blend (_effective_strength) is the PMGG control knob.
                ss_result = _process_sub_stems(processed, sample_rate, instrument="brass", processing_strength=0.55)
                processed = ss_result.audio
                logger.debug(
                    "Phase 45 sub-stem: bands=%d strength=%.2f",
                    ss_result.n_bands,
                    ss_result.processing_strength,
                )
        except Exception as _ss_exc:
            logger.debug("Phase 45 sub-stem skipped: %s", _ss_exc)

        # Physics-Resonanz (Schritt 5 — Biquad Body Resonance)
        try:
            if _enhance_physics_resonance is not None:
                # PhysicsResonanceEnhancer MAX_STRENGTH=1.0; 0.65 for brass (internal 4 dB ceiling).
                pr_result = _enhance_physics_resonance(
                    processed, sample_rate, instrument="brass", enhancement_strength=0.65
                )
                processed = pr_result.audio
                logger.debug(
                    "Phase 45 physics resonance: peaks=%d strength=%.2f",
                    pr_result.n_peaks,
                    pr_result.enhancement_strength,
                )
        except Exception as _pr_exc:
            logger.debug("Phase 45 physics resonance skipped: %s", _pr_exc)

        if 0.0 < _effective_strength < 1.0 and processed.shape == audio.shape:
            processed = audio + _effective_strength * (processed - audio)

        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "gain_h2": gain_h2,
                "presence_db": presence_db,
                "air_db": air_db,
                "instrument_formant_frames": igt_frames,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={"gain_h2": gain_h2, "presence_db": presence_db, "air_db": air_db},
        )
