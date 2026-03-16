import numpy as np


class IntelligentLimiter:
    """
    Lookahead-Limiter mit Soft-Knee (Studio-Algorithmus)
    """

    def __init__(self, ceiling=-1.0, lookahead_ms=2.0, knee_db=6.0):
        self.ceiling = ceiling
        self.lookahead_ms = lookahead_ms
        self.knee_db = knee_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        lookahead = int(sr * self.lookahead_ms / 1000)
        padded = np.pad(audio, (lookahead, 0), mode="constant")
        shifted = padded[:-lookahead] if lookahead > 0 else audio
        peak = np.abs(shifted)
        peak_db = 20 * np.log10(peak + 1e-8)
        over = peak_db - self.ceiling
        gain_db = np.zeros_like(peak_db)
        idx_soft = (over > -self.knee_db / 2) & (over < self.knee_db / 2)
        gain_db[idx_soft] = -((over[idx_soft] + self.knee_db / 2) ** 2) / (2 * self.knee_db)
        idx_over = over >= self.knee_db / 2
        gain_db[idx_over] = -over[idx_over]
        gain_lin = 10 ** (gain_db / 20)
        env = np.ones_like(gain_lin)
        release_coeff = np.exp(-1.0 / (sr * 0.05))
        for i in range(1, len(env)):
            if gain_lin[i] < env[i - 1]:
                env[i] = gain_lin[i]
            else:
                env[i] = release_coeff * env[i - 1] + (1 - release_coeff) * gain_lin[i]
        out = audio * env
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = out * (0.999 / maxval)
        return np.asarray(out)
