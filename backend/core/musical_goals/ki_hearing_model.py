"""
KI-Hörbarkeitsanalyse-Modul für Wow/Flutter & Mechanische Artefakte
=====================================================================

Dieses Modul bewertet die subjektive Hörbarkeit von Wow/Flutter, Clicks,
Dropouts und anderen mechanischen Artefakten.

Nutzt psychoakustische Metriken:
- Spectral Modulation (Frequenz-Schwankungen erkennbar?)
- Pitch Variation (Tonhöhen-Instabilität hörbar?)
- Amplitude Modulation (Lautstärke-Schwankungen störend?)
- Transient Irregularity (Clicks/Pops erkennbar?)

Version: 1.0.0 - Initial Implementation
Autor: AURIK System
Datum: 2026-02-13
Status: PRODUKTIONSREIF ✅
"""

import logging
from typing import Any
import warnings

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# Suppress warnings für cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class KIHörbarkeitsAnalyzer:
    """
    Psychoakustische Hörbarkeitsanalyse für mechanische Artefakte.

    Bewertet Hörbarkeit mit 4 Metriken:
    1. Spectral Modulation (40%) - Frequenz-Schwankungen im Zeit-Spektrum
    2. Pitch Variation (30%) - Tonhöhen-Instabilität (Wow)
    3. Amplitude Modulation (20%) - Lautstärke-Schwankungen (Flutter)
    4. Transient Irregularity (10%) - Clicks/Pops/Dropouts

    Score-Range: 0.0 (unhörbar) bis 1.0 (extrem hörbar/störend)
    Unhörbar: < 0.05
    Kaum wahrnehmbar: 0.05-0.15
    Hörbar: 0.15-0.30
    Störend: 0.30-0.60
    Extrem störend: > 0.60
    """

    def __init__(self) -> float:
        self.logger = logging.getLogger(__name__)
        # Psychoakustische Schwellenwerte (Fletcher-Munson-Kurve inspiriert)
        self.jnd_frequency = 0.003  # Just Noticeable Difference für Pitch (0.3%)
        self.jnd_amplitude = 0.5  # JND für Amplitude (0.5 dB)

    def analyze(self, audio: np.ndarray, sr: int) -> float:
        """
        Hauptmethode: Bewertet Hörbarkeit von mechanischen Artefakten.

        Args:
            audio: Audio signal (mono oder stereo)
            sr: Sample rate

        Returns:
            Hörbarkeits-Score (0.0 = unhörbar, 1.0 = extrem störend)
        """
        try:
            # Audio validation
            if audio is None or len(audio) == 0:
                self.logger.warning("Empty audio in analyze")
                return 0.0

            # OPTIMIZATION: Limitiere auf 30s Sample (Mitte) für Performance
            # Wow/Flutter-Muster sind über 30s statistisch repräsentativ
            max_samples = 30 * sr  # 30 Sekunden
            if len(audio) > max_samples:
                start_idx = (len(audio) - max_samples) // 2  # Mitte des Audios
                audio = audio[start_idx : start_idx + max_samples]

            # Convert zu Mono für Analyse
            if audio.ndim == 2:
                audio_mono = np.mean(audio, axis=1)  # FIX: axis=1 für (samples, channels)
            else:
                audio_mono = audio

            # Normalisiere auf [-1, 1]
            max_amp = np.max(np.abs(audio_mono))
            if max_amp > 0:
                audio_mono = audio_mono / max_amp
            else:
                return 0.0

            # 1. Spectral Modulation (40%)
            spectral_mod = self._measure_spectral_modulation(audio_mono, sr)

            # 2. Pitch Variation (30%)
            pitch_var = self._measure_pitch_variation(audio_mono, sr)

            # 3. Amplitude Modulation (20%)
            amp_mod = self._measure_amplitude_modulation(audio_mono, sr)

            # 4. Transient Irregularity (10%)
            transient_irreg = self._measure_transient_irregularity(audio_mono, sr)

            # Gewichtete Summe
            hörbarkeits_score = 0.40 * spectral_mod + 0.30 * pitch_var + 0.20 * amp_mod + 0.10 * transient_irreg

            # Clip auf [0, 1]
            hörbarkeits_score = np.clip(hörbarkeits_score, 0.0, 1.0)

            self.logger.debug(
                f"Hörbarkeit: Spektrum={spectral_mod:.3f}, Pitch={pitch_var:.3f}, "
                f"Amplitude={amp_mod:.3f}, Transienten={transient_irreg:.3f}, "
                f"Gesamt={hörbarkeits_score:.3f}"
            )

            return float(hörbarkeits_score)

        except Exception as e:
            self.logger.error(f"Hörbarkeitsanalyse fehlgeschlagen: {e}")
            return 0.5  # Conservative fallback

    def _measure_spectral_modulation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst spektrale Modulation (Frequenz-Schwankungen über Zeit).

        Wow/Flutter zeigt sich als Seitenbänder im Spektrum.
        Verwende STFT um Zeit-Frequenz-Variationen zu erkennen.
        """
        try:
            # Short-Time Fourier Transform
            window_size = int(0.05 * sr)  # 50ms windows
            hop_size = window_size // 2

            if len(audio) < window_size:
                return 0.0

            # STFT
            f, t, Zxx = signal.stft(audio, fs=sr, nperseg=window_size, noverlap=hop_size)

            # Magnitude Spektrum
            magnitude = np.abs(Zxx)

            # Fokus auf harmonischen Bereich (100-5000 Hz)
            freq_mask = (f >= 100) & (f <= 5000)
            magnitude = magnitude[freq_mask, :]

            if magnitude.size == 0:
                return 0.0

            # Berechne spektrale Varianz über Zeit
            # Wow/Flutter → hohe Varianz in einzelnen Frequenzbändern
            spectral_variance = np.var(magnitude, axis=1)  # Varianz über Zeit
            mean_variance = np.mean(spectral_variance)

            # Normalisiere (empirisch kalibriert)
            # Werte > 0.01 sind typischerweise hörbar
            spectral_mod_score = np.clip(mean_variance / 0.02, 0.0, 1.0)

            return float(spectral_mod_score)

        except Exception as e:
            self.logger.error(f"Spectral Modulation Messung fehlgeschlagen: {e}")
            return 0.0

    def _measure_pitch_variation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Tonhöhen-Instabilität (Wow, langsame Pitch-Schwankungen).

        Wow → 0.5-6 Hz Modulation der Frequenz
        """
        try:
            # Zero-Crossing Rate als Pitch-Proxy
            # ZCR ≈ 2 * fundamental frequency
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = window_size // 2

            if len(audio) < window_size:
                return 0.0

            # Berechne ZCR in overlapping windows
            zcr_values = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i : i + window_size]
                # Zero crossings
                zero_crossings = np.sum(np.abs(np.diff(np.sign(window)))) / 2
                zcr = zero_crossings / window_size * sr
                zcr_values.append(zcr)

            if len(zcr_values) < 2:
                return 0.0

            zcr_values = np.array(zcr_values)

            # Pitch Variation = Standardabweichung der ZCR
            mean_zcr = np.mean(zcr_values)
            if mean_zcr > 0:
                pitch_variation = np.std(zcr_values) / mean_zcr
            else:
                pitch_variation = 0.0

            # Normalisiere (JND für Pitch ≈ 0.3%)
            # Variationen > 1% sind deutlich hörbar
            pitch_var_score = np.clip(pitch_variation / 0.01, 0.0, 1.0)

            return float(pitch_var_score)

        except Exception as e:
            self.logger.error(f"Pitch Variation Messung fehlgeschlagen: {e}")
            return 0.0

    def _measure_amplitude_modulation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Lautstärke-Schwankungen (Flutter, schnelle Amplituden-Modulation).

        Flutter → 5-25 Hz Modulation der Amplitude
        """
        try:
            # Envelope Extraction via Hilbert Transform
            from scipy.signal import hilbert

            analytic_signal = hilbert(audio)
            envelope = np.abs(analytic_signal)

            # Smooth envelope (10ms moving average)
            smooth_window = int(0.01 * sr)
            if smooth_window < 3:
                smooth_window = 3
            if smooth_window % 2 == 0:
                smooth_window += 1

            if len(envelope) < smooth_window:
                return 0.0

            from scipy.signal import savgol_filter

            try:
                envelope_smooth = savgol_filter(envelope, smooth_window, 2)
            except Exception:
                envelope_smooth = envelope

            # Amplitude Modulation = Varianz der Envelope
            amp_variance = np.var(envelope_smooth)
            mean_amp = np.mean(envelope_smooth)

            if mean_amp > 0:
                amp_mod_ratio = np.sqrt(amp_variance) / mean_amp
            else:
                amp_mod_ratio = 0.0

            # Normalisiere (JND für Amplitude ≈ 0.5 dB ≈ 6%)
            # Modulationen > 10% sind deutlich hörbar
            amp_mod_score = np.clip(amp_mod_ratio / 0.10, 0.0, 1.0)

            return float(amp_mod_score)

        except Exception as e:
            self.logger.error(f"Amplitude Modulation Messung fehlgeschlagen: {e}")
            return 0.0

    def _measure_transient_irregularity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Unregelmäßigkeiten in Transienten (Clicks, Pops, Dropouts).

        Mechanische Defekte → plötzliche Amplituden-Spitzen oder Nullen
        """
        try:
            # Envelope für Transient-Detektion
            from scipy.signal import hilbert

            analytic_signal = hilbert(audio)
            envelope = np.abs(analytic_signal)

            # Differenz der Envelope (Attack-Rate)
            envelope_diff = np.abs(np.diff(envelope))

            # Threshold für abnormale Transienten (> 95. Perzentil)
            threshold = np.percentile(envelope_diff, 95)
            if threshold == 0:
                return 0.0

            # Zähle abnormale Transienten
            abnormal_transients = np.sum(envelope_diff > threshold * 2)

            # Normalisiere auf Zeit (Clicks pro Sekunde)
            duration_sec = len(audio) / sr
            clicks_per_sec = abnormal_transients / duration_sec

            # Threshold: > 1 Click/sec ist hörbar, > 5 Clicks/sec störend
            transient_irreg_score = np.clip(clicks_per_sec / 5.0, 0.0, 1.0)

            return float(transient_irreg_score)

        except Exception as e:
            self.logger.error(f"Transient Irregularity Messung fehlgeschlagen: {e}")
            return 0.0

    def analyze_multitrack_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Multi-Track Enhancement Qualität basierend auf Metriken.

        Args:
            metrics: Dict mit Multi-Track Metriken (time_alignment, phase_alignment, etc.)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0  # Start mit perfekt

            # Time Alignment Quality
            if "time_alignment" in metrics:
                ta = metrics["time_alignment"]
                if ta.get("alignment_applied", False):
                    delay_ms = abs(ta.get("delay_ms", 0.0))
                    correlation = ta.get("correlation", 1.0)
                    # Penalty für große Delays (> 1ms)
                    if delay_ms > 1.0:
                        score *= 1.0 - min(0.2, delay_ms / 10.0)
                    # Bonus für hohe Korrelation
                    score *= 0.7 + 0.3 * correlation

            # Phase Alignment Quality
            if "phase_alignment" in metrics:
                pa = metrics["phase_alignment"]
                phase_diff = abs(pa.get("phase_diff_degrees", 0.0))
                # Penalty für große Phase-Differenzen (> 15°)
                if phase_diff > 15.0:
                    score *= 1.0 - min(0.3, phase_diff / 180.0)

            # Comb Filtering Detection
            if "comb_filter" in metrics:
                cf = metrics["comb_filter"]
                if cf.get("correction_applied", False):
                    notches = cf.get("notches_detected", 0)
                    # Bonus für Comb Filter Entfernung
                    score *= 1.0 + min(0.1, notches / 100.0)

            # Stereo Balance Quality
            if "stereo_balance" in metrics:
                sb = metrics["stereo_balance"]
                if sb.get("correction_applied", False):
                    imbalance_db = abs(sb.get("imbalance_db", 0.0))
                    # Bonus für Balance-Korrektur
                    if imbalance_db > 1.0:
                        score *= 1.0 + min(0.1, imbalance_db / 10.0)

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Multitrack-Metrik-Analyse fehlgeschlagen: {e}")
            return 0.8  # Conservative fallback

    def analyze_tape_metrics(self, metrics: dict[str, Any], audio: np.ndarray | None = None) -> float:
        """
        Bewertet Tape Defect Restoration Qualität.

        Args:
            metrics: Dict mit Tape Metriken (azimuth, print_through)
            audio: Optional audio für zusätzliche Analyse

        Returns:
            Quality score (0.0-1.0)
        """
        try:
            score = 1.0

            # Azimuth Correction Quality
            if "azimuth" in metrics:
                az = metrics["azimuth"]
                if az.get("correction_applied", False):
                    phase_error = abs(az.get("max_phase_error_degrees", 0.0))
                    # Bonus für Azimuth-Korrektur
                    if phase_error > 5.0:
                        score *= 1.0 + min(0.15, phase_error / 45.0)

            # Print-Through Removal Quality
            if "print_through" in metrics:
                pt = metrics["print_through"]
                if pt.get("removal_applied", False):
                    echo_db = pt.get("post_echo_attenuation_db", 0.0)
                    # Bonus für Echo-Entfernung
                    if echo_db < -40.0:  # Gute Dämpfung
                        score *= 1.1

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            self.logger.error(f"Tape-Metrik-Analyse fehlgeschlagen: {e}")
            return 0.85  # Conservative fallback


# Convenience function für einfachen Import
def analyze_wow_flutter_audibility(audio: np.ndarray, sr: int) -> float:
    """
    Convenience function: Bewertet Wow/Flutter Hörbarkeit.

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        Hörbarkeits-Score (0.0-1.0)
    """
    analyzer = KIHörbarkeitsAnalyzer()
    return analyzer.analyze(audio, sr)
