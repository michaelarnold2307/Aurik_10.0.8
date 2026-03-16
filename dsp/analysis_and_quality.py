"""
---
modul_name: AnalysisAndQuality
aufgabe: SOTA-konforme Analyse- und Qualitätsmetriken für Musikrestaurierung
ein_ausgabe_typen:
        input: np.ndarray (Audio), int (SampleRate)
        output: dict (Metriken)
staerken: Vielseitige Analyse, SOTA-Metriken
schwaechen: Nur Analyse, keine Bearbeitung
abhaengigkeiten: [numpy]
---
"""

import logging
import numpy as np
logger = logging.getLogger(__name__)


class SpectralCentroid:
    """Berechnet den spektralen Schwerpunkt eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für SpectralCentroid")
            magnitude = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            centroid = np.sum(magnitude * freqs) / (np.sum(magnitude) + 1e-10)
            result = np.array([centroid])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[SpectralCentroid][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][SpectralCentroid] Ergebnis: {result} | SR: {sr}")


class SpectralRolloff:
    """Berechnet den Roll-Off-Frequenzpunkt (z.B. 85%) eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int, roll_percent: float = 0.85) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für SpectralRolloff")
            magnitude = np.abs(np.fft.rfft(audio))
            total_energy = np.sum(magnitude)
            threshold = roll_percent * total_energy
            cumulative = np.cumsum(magnitude)
            rolloff_idx = np.where(cumulative >= threshold)[0][0]
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            result = np.array([freqs[rolloff_idx]])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[SpectralRolloff][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][SpectralRolloff] Ergebnis: {result} | SR: {sr}")


class RMSEnergy:
    """Berechnet die RMS-Energie eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für RMSEnergy")
            result = np.array([np.sqrt(np.mean(audio**2))])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[RMSEnergy][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][RMSEnergy] Ergebnis: {result} | SR: {sr}")


class ZeroCrossingRate:
    """Berechnet die Zero-Crossing-Rate eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für ZeroCrossingRate")
            zero_crossings = np.where(np.diff(np.sign(audio)))[0]
            result = np.array([len(zero_crossings) / len(audio)])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[ZeroCrossingRate][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][ZeroCrossingRate] Ergebnis: {result} | SR: {sr}")


class SpectralFlatness:
    """Berechnet die spektrale Flatness eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für SpectralFlatness")
            mag = np.abs(np.fft.rfft(audio)) + 1e-10
            geo_mean = np.exp(np.mean(np.log(mag)))
            arith_mean = np.mean(mag)
            flatness = geo_mean / arith_mean
            result = np.array([flatness])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[SpectralFlatness][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][SpectralFlatness] Ergebnis: {result} | SR: {sr}")


class SpectralContrast:
    """Berechnet den spektralen Kontrast eines Audiosignals (Banddynamik)."""

    def process(self, audio: np.ndarray, sr: int, n_bands: int = 6) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für SpectralContrast")
            mag = np.abs(np.fft.rfft(audio))
            bands = np.array_split(mag, n_bands)
            contrast = [np.max(b) - np.min(b) for b in bands if len(b) > 0]
            result = np.array(contrast)
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[SpectralContrast][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][SpectralContrast] Ergebnis: {result} | SR: {sr}")


class CrestFactor:
    """Berechnet den Crest-Faktor eines Audiosignals."""

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Ungültige Eingabe für CrestFactor")
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))
            crest = peak / (rms + 1e-10)
            result = np.array([crest])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[CrestFactor][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][CrestFactor] Ergebnis: {result} | SR: {sr}")


class LoudnessLUFS:
    """Stub für Loudness-Berechnung nach EBU R128 (LUFS)."""

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für LoudnessLUFS")
            # Vereinfachte Annäherung (echte Implementierung benötigt Filterbank)
            rms = np.sqrt(np.mean(audio**2))
            lufs = -0.691 + 10 * np.log10(rms + 1e-10)
            result = np.array([lufs])
            self._audit_log(result, sr)
            return result
        except Exception as e:
            logger.error(f"[LoudnessLUFS][Fehler] {e}")
            self._audit_log(np.array([-1.0]), sr)
            return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][LoudnessLUFS] Ergebnis: {result} | SR: {sr}")


class SNR:
    """Berechnet das Signal-Rausch-Verhältnis (SNR) eines Audiosignals gegenüber Referenz."""

    def process(self, audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
        try:
            if (
                not isinstance(audio, np.ndarray)
                or not isinstance(reference, np.ndarray)
                or audio.size == 0
                or reference.size == 0
            ):
                raise ValueError("Ungültige Eingabe für SNR")
            noise = audio - reference
            snr = 10 * np.log10(np.sum(reference**2) / (np.sum(noise**2) + 1e-10))
            result = np.array([snr])
            self._audit_log(result)
            return result
        except Exception as e:
            logger.error(f"[SNR][Fehler] {e}")
            self._audit_log(np.array([-1.0]))
            return np.array([-1.0])

    def _audit_log(self, result):
        logger.info(f"[AuditLog][SNR] Ergebnis: {result}")


class SISDR:
    """§10.2 STUB: SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) VERBOTEN (§4.4+§10.2).
    SI-SDR ist eine Sprach-Trennungs-Metrik — nicht für Musikqualitätsbewertung geeignet.
    Klasse bleibt für Import-Kompatibilität, Berechnung vollständig deaktiviert."""

    def process(self, audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
        # §4.4+§10.2: SI-SDR verboten — Sprach-Trennungs-Metrik (VERBOTEN für Musik §4.4+§10.2).
        # Neutralwert (kein Quality-Gate-Einfluss). Bitte SNR oder Musical Goals verwenden.
        self._audit_log(np.array([0.0]))
        return np.array([0.0])  # type: ignore[return-value]

    def _audit_log(self, result):
        logger.info(f"[AuditLog][SISDR] §10.2-Stub — Ergebnis: {result}")


class PESQStub:
    """Stub für Perceptual Evaluation of Speech Quality (PESQ)."""

    def process(self, audio: np.ndarray, reference: np.ndarray, sr: int) -> np.ndarray:
        logger.info("[PESQStub] Kein PESQ-Modul verfügbar – Quality-Gate nicht bestanden.")
        self._audit_log(np.array([-1.0]), sr)
        return np.array([-1.0])

    def _audit_log(self, result, sr):
        logger.info(f"[AuditLog][PESQStub] Ergebnis: {result} | SR: {sr}")


class MOSStub:
    """Stub für Mean Opinion Score (MOS, subjektive Bewertung)."""

    def process(self, audio: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        logger.info("[MOSStub] Keine MOS-Bewertung verfügbar – Quality-Gate nicht bestanden.")
        self._audit_log(np.array([-1.0]))
        return np.array([-1.0])

    def _audit_log(self, result):
        logger.info(f"[AuditLog][MOSStub] Ergebnis: {result}")


MODEL_PATH_P808 = "../../models/dnsmos/dnsmos_p808.onnx"
MODEL_PATH_P835 = "../../models/dnsmos/dnsmos_p835.onnx"
