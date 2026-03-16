import numpy as np
import numpy.typing as npt


class DynamicRangeExpander:
    """
    SOTA-konformer Dynamic Range Expander:
    - RMS/Peak-Detection, Soft-Knee, Ratio, Attack/Release, ML-ready
    """

    def __init__(
        self,
        threshold_db: float = -40.0,
        ratio: float = 0.5,
        knee_db: float = 6.0,
        attack_ms: float = 10.0,
        release_ms: float = 80.0,
    ) -> None:
        """
        threshold_db: Expander-Schwelle (dB)
        ratio: Expansionsrate (<1)
        knee_db: Soft-Knee (dB)
        attack_ms: Attack-Zeit (ms)
        release_ms: Release-Zeit (ms)
        """
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit Dynamikexpansion.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: expandiertes Signal (gleicher Typ wie audio)
        """
        # RMS-Detection
        window = int(sr * 0.01)
        rms = np.sqrt(np.convolve(audio**2, np.ones(window) / window, mode="same"))
        rms_db = 20 * np.log10(rms + 1e-8)
        under = self.threshold_db - rms_db
        gain_db = np.zeros_like(rms_db)
        # Soft-Knee
        idx_soft = (under > -self.knee_db / 2) & (under < self.knee_db / 2)
        gain_db[idx_soft] = (1 / self.ratio - 1) * ((under[idx_soft] + self.knee_db / 2) ** 2) / (2 * self.knee_db)
        idx_under = under >= self.knee_db / 2
        gain_db[idx_under] = (1 / self.ratio - 1) * (under[idx_under])
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        attack_coeff = np.exp(-1.0 / (sr * self.attack_ms / 1000))
        release_coeff = np.exp(-1.0 / (sr * self.release_ms / 1000))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = attack_coeff * env[i - 1] + (1 - attack_coeff) * gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        # Pegel normalisieren
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = out * (0.999 / maxval)
        return np.asarray(out.astype(audio.dtype))
