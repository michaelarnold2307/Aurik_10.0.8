import numpy as np
import numpy.typing as npt
from scipy.signal import istft, stft


class SpectralSubtractor:
    """
    SOTA-konformer Spectral Subtractor:
    - STFT-basierte Subtraktion
    - Adaptive Noise-Profile, Spectral Floor, ML-ready
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        noise_profile_frames: int = 10,
        spectral_floor: float = 0.02,
    ) -> None:
        """
        n_fft: FFT-Größe
        hop_length: Hop-Size
        noise_profile_frames: Anzahl Frames zur Rauschprofil-Schätzung
        spectral_floor: Minimaler Restpegel (0...1)
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noise_profile_frames = noise_profile_frames
        self.spectral_floor = spectral_floor

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit spektraler Subtraktion.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: subtrahiertes Signal (gleicher Typ wie audio)
        """
        # Schutz gegen ValueError: noverlap < nperseg
        n_fft = self.n_fft
        hop_length = self.hop_length
        if n_fft <= hop_length:
            n_fft = hop_length + 1
        try:
            f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        # Noise-Profil aus den ersten Frames schätzen
        noise_mag = np.mean(mag[:, : self.noise_profile_frames], axis=1, keepdims=True)
        # Subtraktion
        mag_sub = mag - noise_mag
        mag_sub = np.maximum(mag_sub, self.spectral_floor * np.max(mag))
        Zxx_sub = mag_sub * np.exp(1j * phase)
        try:
            _, out = istft(Zxx_sub, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        except ValueError as e:
            if "noverlap must be less than nperseg" in str(e):
                n_fft = hop_length + 1
                _, out = istft(Zxx_sub, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
            else:
                raise
        # Output-Länge exakt auf Input trimmen (Broadcast-Sicherheit)
        if len(out) > len(audio):
            out = out[: len(audio)]
        elif len(out) < len(audio):
            pad = np.zeros(len(audio), dtype=out.dtype)
            pad[: len(out)] = out
            out = pad
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = out * (0.999 / maxval)
        return np.asarray(out.astype(audio.dtype))
