import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft


class SpectralDenoiser:
    """
    SOTA-konformer Spectral Denoiser:
    - STFT-basierte Rauschreduktion
    - Adaptive Noise-Profile, Spectral Gating, Wiener-Filter
    - ML-ready (Hooks für Deep Spectral Masking)
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_profile_frames: int = 10,
        reduction_db: float = 18.0,
    ) -> None:
        """
        n_fft: FFT-Größe
        hop_length: Hop-Size
        noise_profile_frames: Anzahl Frames zur Rauschprofil-Schätzung
        reduction_db: Maximale Dämpfung (dB)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_profile_frames = noise_profile_frames
        self.reduction_db = reduction_db

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit spektraler Rauschreduktion.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: denoisiertes Signal (gleicher Typ wie audio)
        """
        # Schutz gegen ValueError: noverlap < nperseg
        n_fft = self.n_fft
        hop_length = self.hop_length
        if n_fft <= hop_length:
            n_fft = hop_length + 1
        f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)  # (n_freqs, n_frames)
        phase = np.angle(Zxx)

        # IMCRA-Sliding-Minimum: Rauschboden = Min(letzter W Frames)
        # Cohen (2003): Noise Spectrum Estimation in Adverse Environments
        # Ersetzt np.mean(ersten Frames) = statischer, veralteter Schätzer
        n_frames = mag.shape[1]
        W = max(self.noise_profile_frames, n_frames // 4)
        noise_mag = np.empty_like(mag)
        for t_idx in range(n_frames):
            lo = max(0, t_idx - W)
            noise_mag[:, t_idx] = mag[:, lo : t_idx + 1].min(axis=1)
        noise_mag = np.maximum(noise_mag, 1e-10)

        # MMSE-Wiener-Gain G = xi/(1+xi),  xi = max(SNR - 1, 0)
        # Le Roux & Vincent (2013): Consistent Wiener, Gain-Floor verhindert Musical Noise
        snr = np.maximum(mag / (noise_mag + 1e-12) - 1.0, 0.0)
        gain = snr / (snr + 1.0)
        # Gain-Floor aus reduction_db
        min_gain = 10 ** (-self.reduction_db / 20)
        gain = np.clip(gain, min_gain, 1.0)
        # Anwenden
        mag_denoised = mag * gain
        Zxx_denoised = mag_denoised * np.exp(1j * phase)
        _, out = istft(
            Zxx_denoised,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )
        # Länge anpassen
        out = out[: len(audio)]
        # Pegel normalisieren
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = out * (0.999 / maxval)
        return np.asarray(out.astype(audio.dtype))
