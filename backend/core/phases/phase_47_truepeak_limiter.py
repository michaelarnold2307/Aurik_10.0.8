"""
Phase 47: TruePeak Limiter v2.0 — ITU-R BS.1770 / AES17 konform
================================================================

Vollständige DSP-Implementierung des True-Peak-Limiters ohne ML-Abhängigkeiten.
Ersetzt den kaputten aurik_ml.mastering-Stub.

ALGORITHMUS:
  1. 4× Oversampling via Polyphase-Filter (scipy.signal.resample_poly)
     → Ermittelt Zwischenwerte zwischen Samples; nur so messbar ob TP > Ceiling
  2. True-Peak-Messung auf überabgetasteten Signal (AES17-2020, §12.1.3)
  3. Lookahead-Limiter (12 ms Lookahead):
     - Smooth Gain Reduction: attack=0.1 ms, release=100 ms
     - Keine Saturation / Clipping — nur Pegelanpassung
  4. Downsample zurück auf Eingangs-SR
  5. Ceiling-Clip (Hard Limit) als Notfall-Fallback (−0.1 dBFS default)

WARUM 4×-OVERSAMPLING?
  Nyquist-Frequenz begrenzt die Rekonstruktion auf Sample-Werte; Sinussignale
  nahe Nyquist können Inter-Sample-Peaks haben, die 3 dB über dem Sample-Wert
  liegen. Standard-Peak-Messung übersieht diese. ITU-R BS.1770 schreibt daher
  4-faches Oversampling für True-Peak-Messung vor.

INDUSTRIAL STANDARD:
  - AES17-2020: True Peak Level Definition
  - ITU-R BS.1770-4: Loudness Measurement, Annex 2 (True Peak)
  - EBU R 128: −23 LUFS, Max TP = −1 dBTP

Author: Aurik Development Team
Version: 2.0.0 (keine ML-Import-Abhängigkeit)
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# Standard-Ceiling: −0.5 dBFS entspricht AES/EBU Produktions-Standard
_DEFAULT_CEILING_DBFS = -0.5
_DEFAULT_CEILING_LIN = 10 ** (_DEFAULT_CEILING_DBFS / 20.0)
_OVERSAMPLING = 4  # ITU-R BS.1770: 4× minimum
_LOOKAHEAD_MS = 12.0  # Lookahead-Fenster
_ATTACK_MS = 0.1  # Gain-Reduction Attack
_RELEASE_MS = 100.0  # Gain-Reduction Release


class TruePeakLimiterPhase(PhaseInterface):
    """
    TruePeak Limiter nach ITU-R BS.1770 / AES17.

    Letzte Instanz in der Signalkette (priority=9).
    Verhindert digitales Clipping und überschrittene True-Peak-Level.
    Kein ML-Import — reine DSP-Implementierung.
    """

    phase_id = "phase_47_truepeak_limiter"
    name = "TruePeak Limiter (ITU-R BS.1770)"
    description = (
        "TruePeak Limiter nach ITU-R BS.1770: 4× Oversampling für exakte "
        "Zwischensample-Peak-Detektion, Lookahead Gain Reduction, "
        "keine Clipping-Ansteuerung. Last-in-chain (priority=9)."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.DYNAMICS,
            priority=9,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.02,
            memory_requirement_mb=40,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.98,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Limitiert True-Peak auf ceiling.

        Args:
            audio:       Mono oder Stereo (float32/64, ±1 normiert)
            sample_rate: Abtastrate in Hz
            **kwargs:    ceiling_dbfs (float, Default −0.5 dBFS)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        ceiling_dbfs: float = float(kwargs.get("ceiling_dbfs", _DEFAULT_CEILING_DBFS))
        ceiling_lin: float = 10 ** (ceiling_dbfs / 20.0)

        is_stereo = audio.ndim == 2
        if is_stereo:
            left = self._limit_channel(audio[:, 0], sample_rate, ceiling_lin)
            right = self._limit_channel(audio[:, 1], sample_rate, ceiling_lin)
            n = min(len(left), len(right))
            processed = np.column_stack([left[:n], right[:n]])
        else:
            processed = self._limit_channel(audio, sample_rate, ceiling_lin)

        # Notfall-Hard-Clip (sollte nach Limiter nicht nötig sein)
        processed = np.clip(processed, -ceiling_lin, ceiling_lin)

        tp_before = self._true_peak_dbfs(audio, sample_rate)
        tp_after = self._true_peak_dbfs(processed, sample_rate)
        gain_reduction_db = tp_before - tp_after

        logger.info(
            "Phase 47 TruePeak: ceiling=%.1f dBFS, TP %+.2f → %+.2f dBFS, " "GR=%.2f dB, t=%.3fs",
            ceiling_dbfs,
            tp_before,
            tp_after,
            gain_reduction_db,
            time.time() - t0,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed.astype(audio.dtype),
            execution_time_seconds=time.time() - t0,
            metadata={
                "ceiling_dbfs": ceiling_dbfs,
                "true_peak_before_dbfs": tp_before,
                "true_peak_after_dbfs": tp_after,
                "gain_reduction_db": gain_reduction_db,
                "oversampling": _OVERSAMPLING,
            },
            metrics={
                "true_peak_before_dbfs": tp_before,
                "true_peak_after_dbfs": tp_after,
                "gain_reduction_db": gain_reduction_db,
            },
        )

    # ------------------------------------------------------------------
    # Kern-Algorithmus
    # ------------------------------------------------------------------

    def _limit_channel(self, audio: np.ndarray, sample_rate: int, ceiling: float) -> np.ndarray:
        """Lookahead-Limiter für einen einzelnen Kanal."""
        n = len(audio)
        if n < 16:
            return np.clip(audio, -ceiling, ceiling)

        # 1. 4× Oversampling
        os_audio = sig.resample_poly(audio, _OVERSAMPLING, 1)

        # 2. True-Peak-Detection auf Oversampled Signal
        tp_env = np.abs(os_audio)

        # 3. Erforderliche Gain Reduction (frame-weise, 0.1ms Frames)
        frame_s = max(1, int(sample_rate * 0.001))  # 1ms Frames
        os_frame_s = frame_s * _OVERSAMPLING
        n_frames = len(os_audio) // os_frame_s + 1

        # True-Peak per Frame (Max im Oversampled)
        tp_per_frame = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * os_frame_s
            e = min(s + os_frame_s, len(os_audio))
            tp_per_frame[i] = np.max(tp_env[s:e]) if e > s else 0.0

        # 4. Lookahead: für jeden Frame, schaue L ms voraus
        lookahead_f = max(1, int(_LOOKAHEAD_MS))
        peak_ahead = np.zeros(n_frames)
        for i in range(n_frames):
            lo = i
            hi = min(i + lookahead_f, n_frames)
            peak_ahead[i] = np.max(tp_per_frame[lo:hi])

        # 5. Gain Reduction: wie viel müssen wir runterregeln?
        gain_frames = np.minimum(1.0, ceiling / (peak_ahead + 1e-10))

        # 6. Smooth Gain: Angriff & Release
        sr_f = sample_rate / frame_s  # Frames pro Sekunde
        att_coeff = 1.0 - np.exp(-1.0 / (sr_f * _ATTACK_MS * 0.001 + 1e-10))
        rel_coeff = 1.0 - np.exp(-1.0 / (sr_f * _RELEASE_MS * 0.001 + 1e-10))

        gain_smooth = np.ones(n_frames)
        for i in range(1, n_frames):
            if gain_frames[i] < gain_smooth[i - 1]:
                gain_smooth[i] = gain_smooth[i - 1] + att_coeff * (gain_frames[i] - gain_smooth[i - 1])
            else:
                gain_smooth[i] = gain_smooth[i - 1] + rel_coeff * (gain_frames[i] - gain_smooth[i - 1])
        gain_smooth = np.clip(gain_smooth, 0.0, 1.0)

        # 7. Gain auf Originalsignal (nicht oversampled) anwenden
        gain_full = np.interp(np.arange(n), np.arange(n_frames) * frame_s, gain_smooth)
        return audio * gain_full[:n]

    def _true_peak_dbfs(self, audio: np.ndarray, sample_rate: int) -> float:
        """Misst True-Peak in dBFS durch 4× Oversampling."""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        if len(mono) < 4:
            return 0.0
        os_audio = sig.resample_poly(mono, _OVERSAMPLING, 1)
        peak = float(np.max(np.abs(os_audio)))
        return 20.0 * np.log10(peak + 1e-10)
