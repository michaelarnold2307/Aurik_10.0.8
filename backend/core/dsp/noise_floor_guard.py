"""§MNF (V21) Mindestrauschboden-Guard.

Stellt nach NR-Phasen sicher, dass analoge Materialien nicht auf digitale Stille
fallen. Pause-Zonen werden mit materialkonformem Rauschen auf den erwarteten
Rauschboden angehoben.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.noise_floor_guard import apply_noise_floor_minimum
    result.audio = apply_noise_floor_minimum(result.audio, sr, material)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Material-Rauschböden in dBFS.
# None = digitales Material → kein Mindestboden nötig.
_MATERIAL_FLOORS_DBFS: dict[str, float | None] = {
    "shellac": -42.0,
    "wax_cylinder": -38.0,
    "lacquer_disc": -45.0,
    "wire_recording": -40.0,
    "reel_tape": -52.0,
    "tape": -52.0,
    "vinyl": -55.0,
    "cassette": -50.0,
    # Digital → kein analoger Rauschboden nötig
    "minidisc": None,
    "cd_digital": None,
    "dat": None,
    "mp3_low": None,
    "mp3_high": None,
    "unknown": None,
}

# Low-Pass-Eckfrequenz pro Material (rauschfärbend)
_MATERIAL_NOISE_CUTOFF_HZ: dict[str, float] = {
    "shellac": 7000.0,
    "wax_cylinder": 5000.0,
    "lacquer_disc": 8000.0,
    "wire_recording": 6000.0,
    "reel_tape": 16000.0,
    "tape": 16000.0,
    "vinyl": 18000.0,
    "cassette": 12000.0,
}


def _shaped_noise(n_samples: int, sr: int, cutoff_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Einfaches pink-geformtes Rauschen (Butterworth LP) für Rauschboden-Injektion."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    sos = butter(2, cutoff_hz / (sr / 2.0), btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, white), dtype=np.float32)


def apply_noise_floor_minimum(
    audio: np.ndarray,
    sr: int,
    material: str,
    floor_dbfs: float | None = None,
    *,
    frame_ms: float = 20.0,
) -> np.ndarray:
    """Hebt vollständig stille Frames auf den erwarteten Materialrauschboden an.

    Args:
        audio: Audio nach NR-Phase. Shape [N] oder [2, N].
        sr: Sample-Rate (muss 48000 sein).
        material: Materialklasse (z.B. ``"vinyl"``).
        floor_dbfs: Optionaler Override für den Rauschboden in dBFS.
            Wenn None, wird der materialspezifische Wert aus ``_MATERIAL_FLOORS_DBFS`` verwendet.
        frame_ms: Frame-Länge in ms für Stille-Detektion.

    Returns:
        Audio mit angehobenem Rauschboden (Float32, geclippt auf [-1.0, 1.0]).
    """
    assert sr == 48000
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        mat_key = str(material).lower().strip()

        # Floor bestimmen
        effective_floor: float | None = floor_dbfs
        if effective_floor is None:
            effective_floor = _MATERIAL_FLOORS_DBFS.get(mat_key)
        if effective_floor is None:
            return audio  # Kein analoger Boden nötig

        floor_linear = float(10.0 ** (effective_floor / 20.0))
        # Stille-Schwelle: 6 dB unter dem Mindestboden
        silence_threshold = floor_linear * 0.5

        frame_len = max(64, int(sr * frame_ms / 1000.0))
        cutoff = _MATERIAL_NOISE_CUTOFF_HZ.get(mat_key, 18000.0)

        rng = np.random.default_rng(seed=42)  # Deterministisch für Tests
        is_stereo = audio.ndim == 2

        def _process_channel(ch: np.ndarray) -> np.ndarray:
            ch_out = ch.copy()
            n = len(ch)
            n_frames = n // frame_len
            noise = _shaped_noise(n, sr, cutoff, rng)
            for i in range(n_frames):
                start = i * frame_len
                end = start + frame_len
                rms = float(np.sqrt(np.mean(ch[start:end] ** 2) + 1e-12))
                if rms < silence_threshold:
                    # Rauschen auf exaktem Bodenpegel skalieren
                    noise_seg = noise[start:end]
                    noise_rms = float(np.sqrt(np.mean(noise_seg**2) + 1e-12))
                    scale = floor_linear / (noise_rms + 1e-12)
                    ch_out[start:end] = noise_seg * scale
            return ch_out

        if is_stereo:
            ch0 = _process_channel(audio[0])
            ch1 = _process_channel(audio[1])
            result = np.stack([ch0, ch1], axis=0)
        else:
            result = _process_channel(audio)

        out: np.ndarray = np.clip(result, -1.0, 1.0).astype(np.float32)
        return out

    except Exception as exc:
        logger.debug("apply_noise_floor_minimum non-blocking: %s", exc)
        return audio
