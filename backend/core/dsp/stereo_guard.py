"""§MKI (V23) Mono-Kompatibilitätsprüfung — Stereo-Guard.

Prüft vor dem Export auf Phasenlöschung im 300 Hz–5 kHz Band bei
Vokal-Stereo-Material. Kein Veto — nur WARNING + Metadata-Flag.

Kanonische Nutzung (UV3 pre-export hook):
    from backend.core.dsp.stereo_guard import check_mono_compatibility, MonoCompatResult
    result = check_mono_compatibility(audio, sr)
    if result.phase_cancellation_db > 3.0:
        metadata["mono_compatibility_warning"] = True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class MonoCompatResult:
    """Ergebnis der Mono-Kompatibilitätsprüfung.

    Attributes:
        phase_cancellation_db: Energieverlust (dB) beim Summieren L+R im 300–5000 Hz Band.
            Positiv = Energieverlust (Phasenlöschung). Grenzwert: 3.0 dB → WARNING.
        ok: True wenn phase_cancellation_db <= 3.0.
        mono_rms: RMS des Mono-Summensignals (bandgefiltert).
        stereo_rms: RMS des Stereo-Originalsignals (bandgefiltert, L/R gemittelt).
    """

    phase_cancellation_db: float
    ok: bool
    mono_rms: float = 0.0
    stereo_rms: float = 0.0


def check_mono_compatibility(
    audio: np.ndarray,
    sr: int,
) -> MonoCompatResult:
    """Prüft Mono-Kompatibilität im 300 Hz–5 kHz Band.

    Args:
        audio: Stereo-Audio [2, N] oder [N, 2] oder Mono [N].
            Mono-Signale werden direkt als kompatibel zurückgegeben.
        sr: Sample-Rate (muss 48000 sein).

    Returns:
        MonoCompatResult mit phase_cancellation_db und ok-Flag.
    """
    assert sr == 48000
    _fallback = MonoCompatResult(phase_cancellation_db=0.0, ok=True)

    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Layout-Normierung: immer [2, N]
        if audio.ndim == 1:
            return _fallback  # Mono ist per Definition kompatibel
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                ch_l, ch_r = audio[0], audio[1]
            elif audio.shape[1] == 2:
                ch_l, ch_r = audio[:, 0], audio[:, 1]
            else:
                return _fallback
        else:
            return _fallback

        # Bandpass 300 Hz – 5 kHz (Butterworth 4. Ordnung, zero-phase)
        nyq = sr / 2.0
        sos = butter(4, [300.0 / nyq, 5000.0 / nyq], btype="band", output="sos")

        l_bp = sosfiltfilt(sos, ch_l).astype(np.float32)
        r_bp = sosfiltfilt(sos, ch_r).astype(np.float32)

        # Mono-Summe (standard broadcasting)
        mono_bp = (l_bp + r_bp) * 0.5

        mono_rms = float(np.sqrt(np.mean(mono_bp**2) + 1e-12))
        stereo_rms = float(np.sqrt(np.mean((l_bp**2 + r_bp**2) * 0.5) + 1e-12))

        if stereo_rms < 1e-9:
            return _fallback

        # Phasenlöschung = Energieverlust beim Summieren
        cancellation_db = float(20.0 * np.log10((stereo_rms + 1e-12) / (mono_rms + 1e-12)))
        cancellation_db = float(np.nan_to_num(cancellation_db, nan=0.0, posinf=0.0, neginf=0.0))
        ok = cancellation_db <= 3.0

        if not ok:
            logger.info(
                "§V23 Mono-Kompatibilität: Phasenlöschung=%.1f dB > 3.0 dB (300–5000 Hz) → WARNING",
                cancellation_db,
            )

        return MonoCompatResult(
            phase_cancellation_db=round(cancellation_db, 2),
            ok=ok,
            mono_rms=round(mono_rms, 6),
            stereo_rms=round(stereo_rms, 6),
        )

    except Exception as exc:
        logger.debug("check_mono_compatibility non-blocking: %s", exc)
        return _fallback
