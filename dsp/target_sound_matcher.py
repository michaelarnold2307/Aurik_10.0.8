import numpy as np


class TargetSoundMatcher:
    """
    SOTA Target Sound Matching (Studio-Algorithmus):
    - Passt das Spektrum an ein Referenzsignal an (z.B. modernes Studio-Master)
    """

    def __init__(self, reference_audio: np.ndarray | None = None):
        self.reference_audio = reference_audio

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.reference_audio is None:
            return audio
        # Spektralanalyse
        S = np.abs(np.fft.rfft(audio))
        S_ref = np.abs(np.fft.rfft(self.reference_audio))
        # Matching-Kurve
        match_curve = S_ref / (S + 1e-8)
        # Anwenden im Frequenzbereich
        audio_fft = np.fft.rfft(audio)
        matched_fft = audio_fft * match_curve
        matched = np.fft.irfft(matched_fft)
        return matched[: len(audio)]
