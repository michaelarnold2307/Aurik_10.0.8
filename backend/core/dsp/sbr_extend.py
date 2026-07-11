import librosa
import numpy as np


def _sbr_extend(audio: np.ndarray, sr: int) -> np.ndarray:
    """SBR (Spectral Band Replication): DSP-Bandbreitenerweiterung.

    Kopiert Energie aus 2–6 kHz, transponiert um eine Oktave (→4–12 kHz),
    blended mit Quell-Spektrum. Natürlicher als tanh()-Exciter, kein NaN.
    """
    try:
        n_fft = 2048
        hop = n_fft // 4
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        mag, phase = np.abs(D), np.angle(D)

        freq_per_bin = sr / n_fft
        src_lo = int(2000 / freq_per_bin)
        src_hi = int(6000 / freq_per_bin)
        dst_lo = src_lo * 2
        dst_hi = min(mag.shape[0] - 1, src_hi * 2)

        if dst_lo >= mag.shape[0] or src_lo >= src_hi:
            return audio

        src_mag = mag[src_lo:src_hi, :]
        src_len = src_hi - src_lo
        dst_len = dst_hi - dst_lo
        env = 10.0 ** (-0.15 * np.arange(dst_len) / max(1, dst_len))

        for i in range(min(src_len, dst_len)):
            if dst_lo + i < mag.shape[0]:
                mag[dst_lo + i, :] = np.maximum(
                    mag[dst_lo + i, :],
                    src_mag[min(i, src_len - 1), :] * env[i] * 0.4,
                )

        D_new = mag * np.exp(1j * phase)
        y = librosa.istft(D_new, hop_length=hop, length=len(audio))
        return np.asarray(y, dtype=np.float32)
    except Exception as e:
        logger.warning("sbr_extend.py::_sbr_extend fallback: %s", e)
        return audio
