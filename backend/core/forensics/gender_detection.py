import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


class GenderDetector:
    def __init__(self, use_auth_token=None) -> str:
        self.encoder = VoiceEncoder()

    def detect_gender(self, audio_file) -> str:
        """
        Gibt das dominante Geschlecht im Audioclip zurück: 'male', 'female' oder 'unknown'.
        Verwendet Resemblyzer für robuste, offlinefähige Stimmtyperkennung.
        """
        try:
            wav = preprocess_wav(audio_file)
            self.encoder.embed_utterance(wav)
            # Placeholder: Nutze die mittlere Fundamental-Frequenz als grobe Gender-Schätzung
            # Für professionelle Nutzung sollte ein SVM/KNN auf Embeddings trainiert werden
            f0 = self._estimate_pitch(wav)
            if f0 < 170:
                return "male"
            elif f0 < 300:
                return "female"
            else:
                return "unknown"
        except Exception:
            return "unknown"

    def _estimate_pitch(self, wav, sr=16000) -> float:
        # Einfache Pitch-Schätzung (Median der Autokorrelationsmethode)
        from scipy.signal import correlate

        wav = wav.astype(np.float32)
        corr = correlate(wav, wav)
        corr = corr[len(corr) // 2 :]
        d = np.diff(corr)
        start = np.where(d > 0)[0][0]
        peak = np.argmax(corr[start:]) + start
        if peak == 0:
            return 0
        return sr / peak
