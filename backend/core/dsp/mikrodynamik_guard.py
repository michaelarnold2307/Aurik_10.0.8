"""§MKK (V20) Mikrodynamik-Korrelations-Guard.

Prüft nach NR/Dynamics-Phasen auf Vokal-Material, ob die Frame-Energie-Korrelation
zwischen pre und post im Voiced-Bereich ≥ 0.97 liegt. Unterschreitung →
Dry-Wet-Blend: wet = min(1.0, (corr - 0.90) / 0.07).

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation
    corr = frame_energy_correlation(pre, post, sr, frame_ms=10)
    if corr < 0.97:
        wet = min(1.0, (corr - 0.90) / 0.07)
        result.audio = wet * result.audio + (1 - wet) * audio
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Schwellwert: Korrelation ≥ 0.97 auf Voiced-Frames
MIKRODYNAMIK_THRESHOLD = 0.97
# Voiced-Frame-Schwellwert: Frames mit Energie über diesem Wert
_VOICED_ENERGY_PERCENTILE = 25.0


def frame_energy_correlation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 10.0,
) -> float:
    """Berechnet Pearson-Korrelation der Frame-Energien auf Voiced-Zonen.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate (muss 48000 sein).
        frame_ms: Frame-Länge in ms (Standard: 10 ms — §2.75).

    Returns:
        Pearson-Korrelation [0.0 … 1.0]. Grenzwert: 0.97.
        Bei Fehler oder sehr kurzem Signal: 1.0 (kein Eingriff).
    """
    assert sr == 48000

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if pre.shape != post.shape or pre.size < 512:
            return 1.0

        pre_mono = pre.mean(axis=0) if pre.ndim == 2 else pre
        post_mono = post.mean(axis=0) if post.ndim == 2 else post

        frame_len = max(64, int(sr * frame_ms / 1000.0))
        n = len(pre_mono)
        n_frames = n // frame_len

        if n_frames < 4:
            return 1.0

        pre_energy = np.array(
            [float(np.mean(pre_mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)],
            dtype=np.float32,
        )
        post_energy = np.array(
            [float(np.mean(post_mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)],
            dtype=np.float32,
        )

        # Nur Voiced-Frames: Frames über dem 25. Perzentil der pre-Energie
        voiced_threshold = float(np.percentile(pre_energy, _VOICED_ENERGY_PERCENTILE))
        voiced_mask = pre_energy > voiced_threshold

        if voiced_mask.sum() < 4:
            return 1.0

        pre_voiced = pre_energy[voiced_mask]
        post_voiced = post_energy[voiced_mask]

        # Pearson-Korrelation
        pre_mean = float(np.mean(pre_voiced))
        post_mean = float(np.mean(post_voiced))
        pre_std = float(np.std(pre_voiced) + 1e-12)
        post_std = float(np.std(post_voiced) + 1e-12)

        corr = float(np.mean((pre_voiced - pre_mean) * (post_voiced - post_mean)) / (pre_std * post_std))
        corr = float(np.clip(np.nan_to_num(corr, nan=1.0), -1.0, 1.0))

        if corr < MIKRODYNAMIK_THRESHOLD:
            wet_recommended = float(np.clip((corr - 0.90) / 0.07, 0.0, 1.0))
            logger.info(
                "§V20 Mikrodynamik: Korrelation=%.3f < %.2f auf Voiced-Frames → wet=%.2f",
                corr,
                MIKRODYNAMIK_THRESHOLD,
                wet_recommended,
            )

        return corr

    except Exception as exc:
        logger.debug("frame_energy_correlation non-blocking: %s", exc)
        return 1.0
