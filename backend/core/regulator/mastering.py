"""
SOTA-Mastering-Chain für AURIK: Modular, erweiterbar, produktiv.
Enthält: LUFS-Normalisierung, Multiband-Kompression, adaptives EQing, Limiter, Stereo-Enhancement.
"""

from typing import Any

import librosa
import numpy as np


def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    try:
        import pyloudnorm as pyln
    except ImportError:
        raise RuntimeError("pyloudnorm muss installiert sein für LUFS-Normalisierung.")
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    gain = target_lufs - loudness
    audio = audio * (10 ** (gain / 20))
    return audio


def multiband_compress(
    audio: np.ndarray, sr: int, bands=((20, 250), (250, 4000), (4000, 20000)), ratio=2.0
) -> np.ndarray:
    # Einfache Multiband-Kompression (SOTA: für Produktion durch spezialisierte Module ersetzen)
    from scipy.signal import butter, lfilter

    out = np.zeros_like(audio)
    for low, high in bands:
        b, a = butter(2, [low / (sr / 2), high / (sr / 2)], btype="band")
        band = lfilter(b, a, audio)
        # Soft-Knee Kompression (vereinfachtes Modell)
        threshold = np.percentile(np.abs(band), 90)
        band = np.where(
            np.abs(band) > threshold,
            np.sign(band) * (threshold + (np.abs(band) - threshold) / ratio),
            band,
        )
        out += band
    return out


def adaptive_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    # SOTA: ML-basiertes Target-EQing möglich, hier spektrale Glättung
    S = np.abs(librosa.stft(audio, n_fft=2048))
    mean_spectrum = np.mean(S, axis=1)
    target = np.median(mean_spectrum)
    gain = target / (mean_spectrum + 1e-6)
    S_eq = S * gain[:, None]
    audio_eq = librosa.istft(S_eq * np.exp(1j * np.angle(librosa.stft(audio, n_fft=2048))))
    return librosa.util.fix_length(audio_eq, len(audio))


def limiter(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    # True Peak Limiter (vereinfachte Version)
    peak = np.max(np.abs(audio))
    if peak > threshold:
        audio = audio * (threshold / peak)
    return audio


def stereo_enhance(audio: np.ndarray, width: float = 1.1) -> np.ndarray:
    # Stereo-Enhancement (nur für Stereo)
    if audio.ndim == 1:
        return audio
    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2 * width
    left = mid + side
    right = mid - side
    return np.vstack([left, right])


def mastering_chain(audio: np.ndarray, sr: int, config: dict[str, Any] = None) -> np.ndarray:
    """
    Vollständige SOTA-Mastering-Chain.
    Args:
        audio: np.ndarray (mono oder stereo)
        sr: Sample-Rate
        config: optionale Parameter für einzelne Stufen
    Returns:
        gemastertes Audio
    """
    config = config or {}
    audio = lufs_normalize(audio, sr, target_lufs=config.get("target_lufs", -14.0))
    audio = multiband_compress(audio, sr)
    audio = adaptive_eq(audio, sr)
    audio = limiter(audio)
    if audio.ndim == 2:
        audio = stereo_enhance(audio)
    return audio


# ---------------------------------------------------------------------------
# Legacy-Funktionen (portiert aus backend/mastering.py) — Rückwärtskompatibilität
# ---------------------------------------------------------------------------


def loudness_normalize(audio: np.ndarray, target_loudness: float = -18.0) -> np.ndarray:
    """
    Einfache RMS-basierte Lautheits-Normalisierung (Legacy; bevorzuge lufs_normalize).
    """
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms == 0:
        return audio
    target_rms = 10 ** (target_loudness / 20)
    return audio * (target_rms / rms)


def dither(audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """
    TPDF-Dithering (Triangular Probability Density Function) for quantization
    noise shaping before bit-depth reduction.

    16-bit: ±1 LSB TPDF (two independent uniform distributions summed)
    24-bit: ±1 LSB TPDF at 24-bit resolution (1/8388608)
    32-bit float: no dithering required (no quantization noise)
    """
    if bit_depth == 16:
        # Echtes TPDF: Summe zweier unabhängiger Gleichverteilungen → Dreiecksverteilung
        lsb_16 = 1.0 / 32768.0
        noise = (
            np.random.uniform(-lsb_16, lsb_16, size=audio.shape)
            + np.random.uniform(-lsb_16, lsb_16, size=audio.shape)
        )
        return audio + noise
    elif bit_depth == 24:
        # TPDF für 24-Bit: 1 LSB = 1/2^23
        lsb_24 = 1.0 / 8_388_608.0
        noise = (
            np.random.uniform(-lsb_24, lsb_24, size=audio.shape)
            + np.random.uniform(-lsb_24, lsb_24, size=audio.shape)
        )
        return audio + noise
    # 32-bit float: kein Quantisierungsrauschen, kein Dithering nötig
    return audio


def simple_eq(
    audio: np.ndarray,
    sr: int,
    bass_gain_db: float = -2.0,
    treble_gain_db: float = 2.0,
) -> np.ndarray:
    """Einfacher EQ: Bass absenken, Höhen anheben (Butterworth-Shelf-Filter)."""
    from scipy.signal import butter, lfilter

    b, a = butter(2, 200 / (sr / 2), btype="low")
    bass = lfilter(b, a, audio)
    audio = audio + (10 ** (bass_gain_db / 20) - 1) * bass
    b, a = butter(2, 4000 / (sr / 2), btype="high")
    treble = lfilter(b, a, audio)
    audio = audio + (10 ** (treble_gain_db / 20) - 1) * treble
    return audio


def simple_compressor(
    audio: np.ndarray,
    threshold: float = 0.6,
    ratio: float = 4.0,
    makeup_gain: float = 1.0,
) -> np.ndarray:
    """Einfacher Downward-Kompressor (Legacy; bevorzuge multiband_compress)."""
    abs_audio = np.abs(audio)
    over = abs_audio > threshold
    audio[over] = np.sign(audio[over]) * (threshold + (abs_audio[over] - threshold) / ratio)
    return audio * makeup_gain
