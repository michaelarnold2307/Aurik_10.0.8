import numpy as np
import soxr


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> (np.ndarray, int):
    """
    Resample ein beliebiges Audiosignal auf 48 kHz (Mono oder Stereo) mit soxr.
    Gibt das resampelte Signal und die neue Abtastrate zurück.
    """
    target_sr = 48000
    if orig_sr == target_sr:
        return audio, orig_sr
    if audio.ndim == 1:
        resampled = soxr.resample(audio, orig_sr, target_sr)
    else:
        # Stereo/Mehrkanal: Kanalweise resamplen
        resampled = np.vstack([soxr.resample(audio[ch], orig_sr, target_sr) for ch in range(audio.shape[0])])
    return resampled, target_sr
