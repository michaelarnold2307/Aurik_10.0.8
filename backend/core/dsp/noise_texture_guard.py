"""§NTI (V19) Noise-Textur-Invariante.

Prüft nach NR-Phasen ob das Residualrauschen noch zur Materialklasse passt.
Spektralfarbe (1/f-Steigung) des Restgeräuschs wird gegen Materialprofile
verglichen; zu große Abweichung → Whitening-Warnung + Strength-Reduktion.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance
    dist = compute_noise_texture_distance(residual, material)
    if dist > 0.25:  # → nr_strength × 0.5 (WARNING)
        ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Erwartete spektrale Steigung (dB/Oktave) pro Materialklasse.
# Werte: (min_slope, max_slope) — Rauschen innerhalb dieser Spanne gilt als materialkonform.
_MATERIAL_SLOPE_RANGES: dict[str, tuple[float, float]] = {
    "shellac": (-8.5, -3.5),
    "wax_cylinder": (-10.0, -4.5),
    "lacquer_disc": (-7.5, -3.5),
    "wire_recording": (-9.0, -4.0),
    "reel_tape": (-4.5, -1.0),
    "tape": (-4.5, -1.0),
    "vinyl": (-5.5, -2.0),
    "cassette": (-4.0, -0.5),
    "minidisc": (-2.5, 1.0),
    "cd_digital": (-1.5, 1.5),
    "dat": (-1.5, 1.5),
    "mp3_low": (-3.0, 0.5),
    "mp3_high": (-2.0, 1.0),
    "unknown": (-6.0, 1.5),
}

# Maximaler Steigungsbereich für Normierungszwecke
_MAX_DEVIATION = 8.0  # dB/oct


def _estimate_spectral_slope(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt die Spektralsteigung (dB/oct) via log-log Regression.

    Args:
        audio_mono: Mono-Audio-Signal (float32).
        sr: Sample-Rate.

    Returns:
        Steigung in dB/Oktave (negativ = Roll-Off nach oben).
    """
    n_fft = min(8192, len(audio_mono))
    if n_fft < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio_mono[:n_fft].astype(np.float32), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # Analyse im Band 100 Hz – 8 kHz
    mask = (freqs >= 100.0) & (freqs <= 8000.0) & (spectrum > 1e-14)
    if mask.sum() < 8:
        return 0.0
    log_f = np.log2(freqs[mask])
    log_p = 10.0 * np.log10(spectrum[mask] + 1e-14)
    try:
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    except Exception:
        slope = 0.0
    return float(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0))


def compute_noise_texture_distance(
    residual: np.ndarray,
    material: str,
    sr: int = 48000,
) -> float:
    """Berechnet die Distanz zwischen dem Residualrauschen und dem erwarteten Materialprofil.

    Args:
        residual: Differenz pre_audio − post_audio (Rausch-Residuum). Shape [N] oder [2, N].
        material: Materialklasse (z.B. ``"vinyl"``, ``"shellac"``).
        sr: Sample-Rate. Standardmäßig 48000 Hz (wird nicht assertions-geprüft,
            da auch in Analyse-Kontexten aufrufbar).

    Returns:
        Normierte Distanz [0.0 … 1.0]. 0 = perfekt materialkonform, > 0.25 = Whitening-Warnung.
    """
    try:
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if residual.ndim == 2:
            residual_mono = residual.mean(axis=0).astype(np.float32)
        else:
            residual_mono = residual.astype(np.float32)

        if len(residual_mono) < 256 or float(np.abs(residual_mono).max()) < 1e-9:
            return 0.0

        slope = _estimate_spectral_slope(residual_mono, sr)

        mat_key = str(material).lower().strip()
        lo, hi = _MATERIAL_SLOPE_RANGES.get(mat_key, _MATERIAL_SLOPE_RANGES["unknown"])

        if lo <= slope <= hi:
            return 0.0  # materialkonform

        # Abstand zur nächsten Grenze normieren auf [0, 1]
        dist = max(0.0, lo - slope) if slope < lo else max(0.0, slope - hi)
        normalized = float(np.clip(dist / _MAX_DEVIATION, 0.0, 1.0))
        return float(np.nan_to_num(normalized, nan=0.0))

    except Exception as exc:
        logger.debug("compute_noise_texture_distance non-blocking: %s", exc)
        return 0.0
