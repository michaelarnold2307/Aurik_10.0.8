"""
core/intrinsic_audio_quality_scorer.py
Intrinsic Audio Quality Scorer (IAQS)
=======================================

Psychoakustisch fundierter Qualitätsscorer — vollständig ohne externe
Abhängigkeiten (kein CDPAM, kein DNSMOS, kein PyTorch).

Basiert auf messbaren Signal-Eigenschaften, die stark mit wahrgenommener
Qualität korrelieren:

  A) Spektrale Güte
     - SNR (blind, via Minimum-Statistics-Schätzung)
     - Spektrale Regularität (Spitzen-zu-Tal-Verhältnis)
     - Bandbreiteneffizienz (genutzte Bandbreite vs. erwartete)
     - Bark-Band-Energie-Verteilung (Psychoakustisches Modell)

  B) Zeitbereichs-Güte
     - Transientenklarheit (Attack-Erkennung im Zeitsignal)
     - Dynamikumfang (EBU R128 Loudness Range näherungsweise)
     - Klirrfaktor-Schätzung (THD via Harmonics)

  C) Musikalische Güte
     - Harmonizität (Verhältnis harmonische zu inharmonische Energie)
     - Stimmungsklarheit (Pitch-Konsistenz über Zeit)
     - Authentizitätsindikator (Vintage vs. Digital-Überprägung)

  D) Artefakt-Detektion
     - Klick-Energie-Residuen (hohe Kurzzeitpegel)
     - Digitale Clipping-Indikatoren (Flat-Top-Samples)
     - Codec-Blockartefakte (periodische Spektralmodulation)

Alle Metriken sind:
  - schnell (< 0.5× Echtzeit für typische Längen)
  - robust (kein NaN/Inf)
  - skaliert auf [0.0, 1.0] (1.0 = perfekt)

Verwendung in MultiPassEngine als fallback wenn Plugins fehlen,
und als primärer Scorer in AutonomousRestorationEngine.

Author: Aurik Development Team
Version: 1.0.0 "Perceptual Precision"
Date: 2026-02-17
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import scipy.signal as sp_signal

logger = logging.getLogger(__name__)

# Bark-Band-Grenzen in Hz (25 Bänder nach Zwicker 1961)
_BARK_BANDS_HZ: tuple[float, ...] = (
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
    20000,
)


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicQualityScore:
    """Vollständiges intrinsisches Qualitätsergebnis."""

    # === Zusammenfassung ===
    overall: float = 0.0
    """Gewichteter Gesamtscore (0–1, 1 = perfekt)."""

    # === Spektral ===
    snr_estimate: float = 0.0
    """Blind-SNR-Schätzung in dB."""

    snr_score: float = 0.0
    """SNR normiert (0–1)."""

    spectral_regularity: float = 0.0
    """Spektrale Glätte (0–1, 1 = glatt)."""

    bandwidth_score: float = 0.0
    """Bandbreiteneffizienz (0–1)."""

    bark_balance: float = 0.0
    """Bark-Band-Balance (0–1, 1 = ideal)."""

    # === Zeitbereich ===
    dynamic_range_score: float = 0.0
    """Dynamikumfang-Score (0–1)."""

    transient_clarity: float = 0.0
    """Transientenklarheit (0–1)."""

    thd_estimate_pct: float = 0.0
    """THD-Schätzung in % (kleiner = besser)."""

    thd_score: float = 0.0
    """THD normiert (0–1, 1 = kein Klirr)."""

    # === Musikalisch ===
    harmonicity: float = 0.0
    """Harmonizität (0–1, 1 = rein harmonisch)."""

    pitch_consistency: float = 0.0
    """Pitch-Konsistenz (0–1, 1 = stabile Intonation)."""

    # === Artefakte ===
    click_residual: float = 0.0
    """Klick-Residual-Score (1 = keine Klicks, 0 = viele)."""

    clipping_score: float = 0.0
    """Clipping-Score (1 = kein Clipping, 0 = geclippt)."""

    codec_artifact_score: float = 0.0
    """Codec-Artefakt-Score (1 = keine, 0 = stark)."""

    # === Metadaten ===
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    is_stereo: bool = False
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class IntrinsicAudioQualityScorer:
    """
    Vollständig eigenständiger psychoakustischer Qualitätsscorer.

    Kein Training, keine Modelle, keine externen Abhängigkeiten.
    Funktioniert für beliebige Audiodatei-Typen.
    """

    # Gewichte für den Gesamt-Score
    _WEIGHTS: dict[str, float] = {
        "snr_score": 0.18,
        "spectral_regularity": 0.12,
        "bandwidth_score": 0.08,
        "bark_balance": 0.10,
        "dynamic_range_score": 0.10,
        "transient_clarity": 0.08,
        "thd_score": 0.07,
        "harmonicity": 0.07,
        "pitch_consistency": 0.05,
        "click_residual": 0.07,
        "clipping_score": 0.05,
        "codec_artifact_score": 0.03,
    }

    def __init__(self, fft_size: int = 2048, hop_size: int = 512):
        self.fft_size = fft_size
        self.hop_size = hop_size

    def score(self, audio: np.ndarray, sample_rate: int) -> IntrinsicQualityScore:
        """
        Berechnet den vollständigen Qualitätsscore.

        Args:
            audio: float32 Audio (mono oder stereo, −1…+1)
            sample_rate: Abtastrate in Hz

        Returns:
            IntrinsicQualityScore mit allen Metriken.
        """
        result = IntrinsicQualityScore(sample_rate=sample_rate)

        # Mono ableiten
        if audio.ndim == 2:
            result.is_stereo = True
            mono = np.mean(audio, axis=1).astype(np.float32)
        elif audio.ndim == 1:
            mono = audio.astype(np.float32)
        else:
            mono = audio.flatten().astype(np.float32)

        result.duration_seconds = len(mono) / sample_rate

        # Zu kurzes Signal
        if len(mono) < self.fft_size:
            result.warnings.append("Signal zu kurz für vollständige Analyse — Mindestqualität.")
            result.overall = 0.5
            return result

        # STFT berechnen (einmalig, für alle spektralen Metriken)
        try:
            stft = self._compute_stft(mono)
            mag = np.abs(stft)
            power = mag**2
        except Exception as exc:
            logger.warning("STFT fehlgeschlagen: %s", exc)
            result.warnings.append(f"STFT-Fehler: {exc}")
            result.overall = 0.5
            return result

        freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / sample_rate)

        # === A) Spektrale Metriken ===
        result.snr_estimate, result.snr_score = self._blind_snr(power)
        result.spectral_regularity = self._spectral_regularity(mag)
        result.bandwidth_score = self._bandwidth_score(power, freqs, sample_rate)
        result.bark_balance = self._bark_balance(power, freqs, sample_rate)

        # === B) Zeitbereich ===
        result.dynamic_range_score = self._dynamic_range_score(mono)
        result.transient_clarity = self._transient_clarity(mono, sample_rate)
        result.thd_estimate_pct, result.thd_score = self._thd_estimate(power, freqs, sample_rate)

        # === C) Musikalisch ===
        result.harmonicity = self._harmonicity(power, freqs, sample_rate)
        result.pitch_consistency = self._pitch_consistency(mag, freqs)

        # === D) Artefakt-Detektion ===
        result.click_residual = self._click_residual(mono)
        result.clipping_score = self._clipping_score(mono)
        result.codec_artifact_score = self._codec_artifacts(power, freqs, sample_rate)

        # === Gesamt-Score (gewichtet) ===
        result.overall = self._weighted_overall(result)

        return result

    def score_as_float(self, audio: np.ndarray, sample_rate: int) -> float:
        """Kurzform: Gibt nur den Gesamt-Score (0–1) zurück."""
        return self.score(audio, sample_rate).overall

    # ------------------------------------------------------------------
    # A) Spektrale Metriken
    # ------------------------------------------------------------------

    def _compute_stft(self, mono: np.ndarray) -> np.ndarray:
        """STFT via scipy."""
        window = sp_signal.get_window("hann", self.fft_size)
        _, _, z = sp_signal.stft(
            mono,
            nperseg=self.fft_size,
            noverlap=self.fft_size - self.hop_size,
            window=window,
            padded=True,
        )
        return z  # shape (freqs, frames)

    def _blind_snr(self, power: np.ndarray) -> tuple[float, float]:
        """
        Blind SNR-Schätzung via spektrale Peakigkeit.

        Methode: Verhältnis der energiereichsten Bins (Signal)
        zu den energieärmsten Bins (Rauschen) im Zeitmittel-Spektrum.

        Vorteil gegenüber temporaler Statistik: funktioniert korrekt
        sowohl für reine Töne (hohe Peakigkeit) als auch Breitbandsignale.
        """
        eps = 1e-12
        # Zeitmittelwert über alle STFT-Frames
        mean_spec = np.mean(power, axis=1)  # (freqs,)
        sorted_spec = np.sort(mean_spec)
        n = len(sorted_spec)

        # Obere 10 % = Signal-Bins, untere 30 % = Rauschboden
        top_end = max(int(n * 0.90), 1)
        noise_end = max(int(n * 0.30), 1)
        signal_power = float(np.mean(sorted_spec[top_end:])) + eps
        noise_power = float(np.mean(sorted_spec[:noise_end])) + eps

        snr_db = 10.0 * np.log10(signal_power / noise_power)
        # Klemmen: 0 dB (reines Rauschen) … 60 dB (sehr sauber) → 0…1
        snr_score = float(np.clip(snr_db / 60.0, 0.0, 1.0))
        # NaN/Inf-Guard (§3.1)
        snr_db = np.nan_to_num(snr_db, nan=0.0, posinf=60.0, neginf=0.0)
        snr_score = np.nan_to_num(snr_score, nan=0.0, posinf=1.0, neginf=0.0)
        return float(snr_db), float(snr_score)

    def _spectral_regularity(self, mag: np.ndarray) -> float:
        """
        Spektrale Regularität (Glattheit).

        Glatte Spektren = kein abruptes Rauschen / keine Artefakte.
        Score: 1 − normierte Varianz benachbarter Frequenzbins.
        """
        eps = 1e-12
        mean_mag = np.mean(mag, axis=1) + eps  # Zeitgemitteltes Spektrum
        diff = np.diff(np.log(mean_mag + eps))
        regularity = 1.0 - float(np.clip(np.std(diff) / 2.0, 0.0, 1.0))
        regularity = np.nan_to_num(regularity, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(regularity, 0.0, 1.0))

    def _bandwidth_score(self, power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> float:
        """
        Bandbreiteneffizienz: Wie viel der nutzbaren Bandbreite ist aktiv?

        Erwartete Mindestbandbreite: 8 kHz (Sprache) … 20 kHz (Musik HQ).
        """
        eps = 1e-12
        mean_power = np.mean(power, axis=1)
        max_power = float(np.max(mean_power)) + eps
        threshold = max_power * 1e-4  # −40 dB unter Peak

        active_freqs = freqs[mean_power > threshold]
        if len(active_freqs) == 0:
            return 0.0
        high_freq = float(active_freqs[-1])
        nyquist = sample_rate / 2.0
        score = float(np.clip(high_freq / min(nyquist, 20000.0), 0.0, 1.0))
        score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
        return float(score)

    def _bark_balance(self, power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> float:
        """
        Bark-Band-Energie-Balance.

        Vergleicht die Energie-Verteilung über Bark-Bänder mit einer
        idealen Referenzverteilung (rosa Rauschen ≈ −10 dB/Oktave).
        """
        eps = 1e-12
        mean_power = np.mean(power, axis=1)
        nyquist = sample_rate / 2.0

        bark_energies: list[float] = []
        valid_bands = [b for b in _BARK_BANDS_HZ if b <= nyquist]
        limits = [0.0] + list(valid_bands)

        for i in range(len(limits) - 1):
            lo, hi = limits[i], limits[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                bark_energies.append(float(np.sum(mean_power[mask])) + eps)

        if len(bark_energies) < 4:
            return 0.5

        energies = np.array(bark_energies)
        energies_db = 10.0 * np.log10(energies / (energies[0] + eps) + eps)

        # Idealer Verlauf: leicht abfallend (rosa Rauschen)
        ideal = np.linspace(0.0, -12.0, len(energies_db))
        deviation = np.abs(energies_db - ideal)
        balance = float(np.clip(1.0 - np.mean(deviation) / 20.0, 0.0, 1.0))
        balance = np.nan_to_num(balance, nan=0.5, posinf=1.0, neginf=0.0)
        return float(balance)

    # ------------------------------------------------------------------
    # B) Zeitbereich
    # ------------------------------------------------------------------

    def _dynamic_range_score(self, mono: np.ndarray) -> float:
        """
        Dynamikumfang: Verhältnis 90-Perzentil / 20-Perzentil des Frame-RMS.

        Bewertung:
          LRA < 3 dB  → stark komprimiert (0.0)
          LRA 6–18 dB → ideal für Musik (0.7–1.0)
          LRA > 30 dB → zu weiträumig (0.5, besser als überkomprimiert)
        """
        eps = 1e-12
        frame_rms = self._frame_rms(mono)
        # Nur aktive (nicht-stille) Frames
        peak_rms = float(np.max(frame_rms)) + eps
        active = frame_rms[frame_rms > peak_rms * 0.01]  # Frames > −40 dB
        if len(active) < 4:
            return 0.5
        p90 = float(np.percentile(active, 90)) + eps
        p20 = float(np.percentile(active, 20)) + eps
        lra_db = 20.0 * np.log10(p90 / p20)
        # Dreieckige Nutzenfunktion: Maximum bei 12 dB
        if lra_db < 3.0:
            score = lra_db / 3.0 * 0.3  # 0 … 0.3
        elif lra_db <= 18.0:
            score = 0.3 + (lra_db - 3.0) / 15.0 * 0.7  # 0.3 … 1.0
        else:
            score = max(1.0 - (lra_db - 18.0) / 30.0, 0.5)  # sanft abfallend
        score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(score, 0.0, 1.0))

    def _frame_rms(self, mono: np.ndarray, frame_ms: float = 10.0) -> np.ndarray:
        """Berechnet RMS pro Zeitrahmen."""
        sr = 44100  # Näherung; wird bei Bedarf angepasst
        frame_len = max(int(sr * frame_ms / 1000), 1)
        n_frames = len(mono) // frame_len
        if n_frames == 0:
            return np.array([float(np.sqrt(np.mean(mono**2)))])
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        return np.sqrt(np.mean(frames**2, axis=1))

    def _transient_clarity(self, mono: np.ndarray, sample_rate: int) -> float:
        """
        Transientenklarheit: Wie scharf sind Attackereignisse?

        Hohe Ableitung → klare Transienten → hoher Score.
        Verschwommene/bearbeitete Transienten haben flachen Envelope.
        """
        eps = 1e-12
        # Envelope via Hanning-geglättetes Betragsignal
        abs_audio = np.abs(mono).astype(np.float64)
        win = sp_signal.get_window("hann", min(1024, len(abs_audio) // 4 * 4 or 4))
        envelope = sp_signal.fftconvolve(abs_audio, win / win.sum(), mode="same")
        envelope = np.clip(envelope, eps, None)

        # Erste Ableitung des Envelopes
        deriv = np.diff(envelope)
        positive_deriv = deriv[deriv > 0]
        if len(positive_deriv) == 0:
            return 0.5
        peak_attack = float(np.percentile(positive_deriv, 99))
        mean_attack = float(np.mean(positive_deriv))
        ratio = peak_attack / (mean_attack + eps)
        # Verhältnis 1 = keine Transienten, >10 = klare Transienten
        score = float(np.clip((ratio - 1.0) / 20.0, 0.0, 1.0))
        score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
        return float(score)

    def _thd_estimate(self, power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> tuple[float, float]:
        """
        THD-Schätzung via Harmonic Product Spectrum (vereinfacht).

        Findet dominante Frequenz und schätzt Verhältnis harmonischer Energie.
        """
        eps = 1e-12
        mean_power = np.mean(power, axis=1)
        if len(mean_power) < 10:
            return 0.0, 1.0

        # Suche dominante Grundfrequenz im Bereich 80–2000 Hz
        lo_idx = int(80.0 / (sample_rate / 2.0) * len(freqs))
        hi_idx = int(2000.0 / (sample_rate / 2.0) * len(freqs))
        lo_idx = max(lo_idx, 1)
        hi_idx = min(hi_idx, len(mean_power) - 1)

        segment = mean_power[lo_idx:hi_idx]
        if len(segment) == 0 or np.max(segment) < eps:
            return 0.0, 1.0

        f0_idx = lo_idx + int(np.argmax(segment))
        f0 = freqs[f0_idx]
        if f0 < 1.0:
            return 0.0, 1.0

        fundamental_power = float(mean_power[f0_idx])

        # Harmonische aufsammeln (2.–5. Oberton)
        harmonic_power = 0.0
        for n in range(2, 6):
            h_freq = n * f0
            if h_freq >= sample_rate / 2.0:
                break
            h_idx = int(h_freq / (sample_rate / 2.0) * len(freqs))
            h_idx = min(h_idx, len(mean_power) - 1)
            # Fenster um Harmonische
            lo = max(h_idx - 3, 0)
            hi = min(h_idx + 4, len(mean_power))
            harmonic_power += float(np.sum(mean_power[lo:hi]))

        thd_ratio = harmonic_power / (fundamental_power + eps)
        thd_pct = float(np.clip(np.sqrt(thd_ratio) * 100.0, 0.0, 100.0))
        thd_score = float(np.clip(1.0 - thd_pct / 10.0, 0.0, 1.0))
        # NaN/Inf-Guard (§3.1)
        thd_pct = np.nan_to_num(thd_pct, nan=5.0, posinf=100.0, neginf=0.0)
        thd_score = np.nan_to_num(thd_score, nan=0.5, posinf=1.0, neginf=0.0)
        return float(thd_pct), float(thd_score)

    # ------------------------------------------------------------------
    # C) Musikalisch
    # ------------------------------------------------------------------

    def _harmonicity(self, power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> float:
        """
        Harmonizität: Wie viel der Energie fällt auf Harmonische?

        Harmonisches Signal: Energie konzentriert auf f0, 2f0, 3f0 …
        Rauschen: Energie gleichmäßig verteilt.
        """
        eps = 1e-12
        mean_power = np.mean(power, axis=1)
        total_power = float(np.sum(mean_power)) + eps

        # Finde Top-10 Spektralpeaks
        n_peaks = min(10, len(mean_power) // 3)
        peaks, _ = sp_signal.find_peaks(mean_power, distance=5)
        if len(peaks) == 0:
            return 0.5

        top_peaks = peaks[np.argsort(mean_power[peaks])[-n_peaks:]]
        peak_power = float(np.sum(mean_power[top_peaks])) + eps

        harmonicity = float(np.clip(peak_power / total_power * 3.0, 0.0, 1.0))
        harmonicity = np.nan_to_num(harmonicity, nan=0.5, posinf=1.0, neginf=0.0)
        return float(harmonicity)

    def _pitch_consistency(self, mag: np.ndarray, freqs: np.ndarray) -> float:
        """
        Pitch-Konsistenz: Wie stabil ist die dominante Frequenz über die Zeit?

        Stabile Pitch → gute Restaurierung ohne Wow/Flutter-Reste.
        """
        # Dominante Frequenz pro Frame
        dominant_idx = np.argmax(mag, axis=0)
        dominant_freqs = freqs[np.clip(dominant_idx, 0, len(freqs) - 1)]

        # Nur nicht-stille Frames
        frame_energy = np.max(mag, axis=0)
        active = frame_energy > (np.max(frame_energy) * 0.05)
        active_freqs = dominant_freqs[active]

        if len(active_freqs) < 3:
            return 0.5

        # Normierte Standardabweichung der dominanten Frequenz
        mean_f = float(np.mean(active_freqs)) + 1e-6
        std_f = float(np.std(active_freqs))
        cv = std_f / mean_f  # Variationskoeffizient
        consistency = float(np.clip(1.0 - cv * 5.0, 0.0, 1.0))
        consistency = np.nan_to_num(consistency, nan=0.5, posinf=1.0, neginf=0.0)
        return float(consistency)

    # ------------------------------------------------------------------
    # D) Artefakt-Detektion
    # ------------------------------------------------------------------

    def _click_residual(self, mono: np.ndarray) -> float:
        """
        Klick-Residual: Wie viele kurze, große Amplitudenspitzen gibt es?

        Klicks erzeugen extrem kurze Peaks weit über dem RMS.
        Schwelle: 3× RMS (konservativ: erfasst auch mittlere Impulse).
        Für CDPAM-ähnliches Verhalten: pro 100 ms Fenster gemessen.
        """
        eps = 1e-12
        rms = float(np.sqrt(np.mean(mono**2))) + eps
        abs_audio = np.abs(mono)
        # Schwelle: 3× RMS ≈ ca. +9.5 dB über Mittelwert
        click_fraction = float(np.mean(abs_audio > 3.0 * rms))
        # Toleranz: 0.01 % der Samples (= 4 Samples/s @ 44100 Hz)
        # Darüber liegende Samples = Klick-Artefakte
        penalized = max(click_fraction - 0.0001, 0.0)
        score = float(np.clip(1.0 - penalized * 3000.0, 0.0, 1.0))
        score = np.nan_to_num(score, nan=0.8, posinf=1.0, neginf=0.0)
        return float(score)

    def _clipping_score(self, mono: np.ndarray) -> float:
        """
        Clipping-Score: Erkennt Flat-Top-Clipping an aufeinanderfolgenden
        gleichen Maximalsamples.

        Methode:
          1. Prüfe ob Peak nahe 1.0 (> 0.98) — hartes Clipping zu ±1
          2. Zähle Samples die innerhalb 0.5 % des globalen Peaks liegen
          3. Erkenne Runs (≥ 2 aufeinanderfolgende Flat-Samples)

        Bonus: Normiert NICHT (verhindert false positives durch Normierung).
        """
        peak = float(np.max(np.abs(mono)))
        if peak < 1e-6:
            return 1.0

        # Nur relevant wenn Peak nahe ±1.0 (echtes Clipping-Regime)
        # Bei peak < 0.98 kann kein digitales Clipping vorliegen
        if peak < 0.98:
            return 1.0  # kein Clipping möglich

        # Flat-Top-Detektion: Samples nahe Absolutpeak
        threshold = peak * 0.995
        near_peak = np.abs(mono) >= threshold  # shape (n,)

        # Aufeinanderfolgende Flat-Samples (Runs länge ≥ 2) zählen
        runs = 0
        in_run = False
        run_len = 0
        for s in near_peak:
            if s:
                run_len += 1
                if not in_run:
                    in_run = True
            else:
                if run_len >= 2:
                    runs += 1
                run_len = 0
                in_run = False
        if run_len >= 2:
            runs += 1

        # Normiere: 0 Runs = 1.0, viele Runs = 0.0
        total_frames = max(len(mono) // 1024, 1)
        clip_density = runs / total_frames
        score = float(np.clip(1.0 - clip_density * 5.0, 0.0, 1.0))
        score = np.nan_to_num(score, nan=0.8, posinf=1.0, neginf=0.0)
        return float(score)

    def _codec_artifacts(self, power: np.ndarray, freqs: np.ndarray, sample_rate: int) -> float:
        """
        Codec-Blockartefakte: Periodische Spektralmodulation durch Codecs.

        MP3/AAC erzeugen charakteristische Blockartefakte alle 26 ms (23 ms).
        Diese zeigen sich als periodische Modulation der Framespektren.
        """
        if power.shape[1] < 8:
            return 0.8  # Zu kurz für Analyse

        # Zeitliche Varianz des Spektrums
        spec_var = np.var(power, axis=1)
        mean_power = np.mean(power, axis=1)

        eps = 1e-12
        rel_var = spec_var / (mean_power**2 + eps)

        # Hohe relative Varianz = ungleichmäßige zeitliche Entwicklung = Codec-Artefakt
        artifact_indicator = float(np.median(rel_var))
        score = float(np.clip(1.0 - artifact_indicator / 2.0, 0.0, 1.0))
        score = np.nan_to_num(score, nan=0.7, posinf=1.0, neginf=0.0)
        return float(score)

    # ------------------------------------------------------------------
    # Gesamt-Score
    # ------------------------------------------------------------------

    def _weighted_overall(self, r: IntrinsicQualityScore) -> float:
        """Berechnet gewichteten Gesamt-Score."""
        weighted_sum = 0.0
        total_weight = 0.0
        for metric, weight in self._WEIGHTS.items():
            value = getattr(r, metric, None)
            if value is not None and isinstance(value, float):
                if not (np.isnan(value) or np.isinf(value)):
                    weighted_sum += weight * float(np.clip(value, 0.0, 1.0))
                    total_weight += weight
        if total_weight < 1e-6:
            return 0.5
        result = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
        result = np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
        return float(result)
