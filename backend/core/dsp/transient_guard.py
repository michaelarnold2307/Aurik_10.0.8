"""§PEP (V22) Pre-Echo-Prevention — Transient-Shift-Detektor.

Prüft nach additiven ML-Phasen (phase_06, phase_07, phase_23), ob Transient-
Onsets zeitlich verschoben wurden (Pre-Echo). Shift > ±2 ms → blend_reduction
als Metadata-Flag; kein Rollback (non-blocking WARNING).

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.transient_guard import detect_transient_shifts, TransientShiftResult
    result = detect_transient_shifts(pre, post, sr)
    # result.max_shift_ms > 2.0 → metadata["onset_shift_ms"] setzen
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optionale Abhängigkeit
    librosa = None

logger = logging.getLogger(__name__)

# Toleranzgrenzwert für Onset-Verschiebung
TRANSIENT_SHIFT_THRESHOLD_MS = 2.0
# Maximale Suchfenster-Breite für Onset-Matching
_MATCH_WINDOW_MS = 30.0


@dataclass
class TransientShiftResult:
    """Ergebnis der Transient-Shift-Detektion.

    Attributes:
        max_shift_ms: Maximale Onset-Verschiebung in ms (positiv = nach vorne = Pre-Echo).
        onset_count: Anzahl erkannter Onsets.
        ok: True wenn max_shift_ms <= 2.0 ms.
        blend_reduction: Empfohlene Wet-Reduktion (0.0–1.0, 0 = kein Eingriff).
    """

    max_shift_ms: float
    onset_count: int
    ok: bool
    blend_reduction: float = 0.0
    shifts_ms: list[float] = field(default_factory=list)


def _detect_onsets_simple(audio_mono: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """Einfache Onset-Detektion via Spektralflussnorm (ohne librosa-Dependency als Pflicht)."""
    try:
        if librosa is None:
            raise RuntimeError("librosa nicht verfügbar")

        onsets = librosa.onset.onset_detect(y=audio_mono, sr=sr, hop_length=hop, units="samples", backtrack=True)
        return np.asarray(onsets, dtype=np.int64)
    except Exception:
        pass

    # Fallback: Differenz der Frame-Energie
    frame_len = hop
    n = len(audio_mono)
    energies = []
    for i in range(0, n - frame_len, frame_len):
        energies.append(float(np.sum(audio_mono[i : i + frame_len] ** 2)))
    energies = np.array(energies, dtype=np.float32)
    diff = np.diff(energies, prepend=energies[:1])
    threshold = float(np.mean(diff) + 1.5 * np.std(diff))
    onset_frames = np.where(diff > threshold)[0]
    return (onset_frames * frame_len).astype(np.int64)


def detect_transient_shifts(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
) -> TransientShiftResult:
    """Erkennt zeitliche Verschiebungen von Transient-Onsets zwischen pre und post.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase (same shape as pre).
        sr: Sample-Rate (muss 48000 sein).

    Returns:
        TransientShiftResult. ok=False wenn max_shift_ms > 2.0 ms.
    """
    assert sr == 48000
    _fallback = TransientShiftResult(max_shift_ms=0.0, onset_count=0, ok=True, blend_reduction=0.0)

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        pre_mono = pre.mean(axis=0) if pre.ndim == 2 else pre
        post_mono = post.mean(axis=0) if post.ndim == 2 else post

        if len(pre_mono) < 512:
            return _fallback

        hop = 256
        pre_onsets = _detect_onsets_simple(pre_mono, sr, hop)
        post_onsets = _detect_onsets_simple(post_mono, sr, hop)

        if len(pre_onsets) == 0:
            return _fallback

        match_window_samples = int(_MATCH_WINDOW_MS / 1000.0 * sr)
        shifts_ms: list[float] = []

        for onset_pre in pre_onsets:
            # Nächsten Onset in post innerhalb Suchfenster finden
            candidates = post_onsets[
                (post_onsets >= onset_pre - match_window_samples) & (post_onsets <= onset_pre + match_window_samples)
            ]
            if len(candidates) == 0:
                continue
            # Nächsten Kandidaten wählen
            nearest = candidates[int(np.argmin(np.abs(candidates - onset_pre)))]
            shift_samples = int(nearest) - int(onset_pre)
            shift_ms = float(shift_samples) / sr * 1000.0
            shifts_ms.append(shift_ms)

        if not shifts_ms:
            return _fallback

        max_shift = float(np.max(np.abs(shifts_ms)))
        ok = max_shift <= TRANSIENT_SHIFT_THRESHOLD_MS

        # Blend-Reduktion: proportional zur Überschreitung
        blend_reduction = 0.0
        if not ok:
            blend_reduction = float(np.clip(max_shift / (TRANSIENT_SHIFT_THRESHOLD_MS * 2.0), 0.0, 1.0))
            logger.info(
                "§V22 Pre-Echo: max_shift=%.2f ms > %.0f ms → blend_reduction=%.2f",
                max_shift,
                TRANSIENT_SHIFT_THRESHOLD_MS,
                blend_reduction,
            )

        return TransientShiftResult(
            max_shift_ms=round(max_shift, 3),
            onset_count=len(pre_onsets),
            ok=ok,
            blend_reduction=round(blend_reduction, 3),
            shifts_ms=[round(s, 3) for s in shifts_ms[:10]],  # max 10 für Metadata
        )

    except Exception as exc:
        logger.debug("detect_transient_shifts non-blocking: %s", exc)
        return _fallback
