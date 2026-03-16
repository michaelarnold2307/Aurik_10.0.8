import numpy as np


def normalize_audio(audio: np.ndarray, peak: float = 0.999) -> np.ndarray:
    """Normiert das Audiosignal auf den angegebenen Peak."""
    maxval = np.max(np.abs(audio)) + 1e-8
    if maxval > 0:
        audio = audio / maxval * peak
    return audio.astype(audio.dtype)


def compute_rms(audio: np.ndarray) -> float:
    """Berechnet den RMS-Wert eines Audiosignals."""
    return float(np.sqrt(np.mean(audio**2)))


def compute_loudness(audio: np.ndarray) -> float:
    """Berechnet eine einfache Lautheitsschätzung (LUFS-Approximation)."""
    rms = compute_rms(audio)
    return 20 * np.log10(rms + 1e-8)


def audio_stats(audio: np.ndarray) -> dict:
    """Gibt zentrale Statistiken (Peak, RMS, Loudness) zurück."""
    return {
        "peak": float(np.max(np.abs(audio))),
        "rms": compute_rms(audio),
        "loudness": compute_loudness(audio),
    }


def log_message(msg: str, logfile: str = "aurik6.log"):
    """Schreibt eine Lognachricht in eine Datei."""
    with open(logfile, "a") as f:
        f.write(msg + "\n")
