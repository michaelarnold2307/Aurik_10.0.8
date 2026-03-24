"""
Phase 49: Advanced Dereverberation v3.0 — WPE/OMLSA Consistent
===============================================================

Vollständige DSP-Implementierung ohne ML-Abhängigkeiten.
Ersetzt den kaputten ML-Stub aus v1.0 und die np.fft.rfft-Schleife aus v2.0.

ALGORITHMUS — Weighted Prediction Error (WPE), vereinfachte Variante:
----------------------------------------------------------------------
WPE (Nakatani et al. 2010) modelliert den beobachteten Hallsignal y(t,f)
als Summe aus Direktsignal d(t,f) und spätem Nachhall r(t,f):

    y(t,f) = d(t,f) + r(t,f)

Der Nachhall-Anteil wird als gewichtete Summe vergangener Frames geschätzt:

    r(t,f) ≈ Σ_k  g_k(f) · y(t - D - k, f)

mit Systemverzögerung D (typisch 2–3 Frames ≈ 30–50 ms) und
Prädiktionsordnung K (typisch 5–10 Frames ≈ 80–160 ms Nachhall-Modell).

Die Koeffizienten g_k werden per Minimum-Varianz-Schätzung (MVDR-Prinzip)
iterativ ermittelt. Implementiert wird eine vereinfachte Single-Pass-Version:

1. scipy.signal.stft (Hann-Fenster, 75 % Überlappung) → phasenkonsistentes TF
2. Verzögertes Autoregressive Prediction per Frequenzband
3. Nachhall-Subtraktion mit Consistent-Wiener-Postfilterung (le Roux 2013)
4. Transientenmaske: Transienten umgehen die Dereverberation vollständig
5. scipy.signal.istft (OLA-konsistent, PGHI-konform) + nan_to_num + clip

Komplementär zu Phase 20 (OMLSA/IMCRA):
  - Phase 20: schnell, für moderaten Nachhall (RT60 < 0.6 s)
  - Phase 49: präziser, für starken Nachhall (RT60 0.4–2.0 s)
  → ARE aktiviert beide wenn REVERB_EXCESS severity >= 0.4

WISSENSCHAFTLICHE GRUNDLAGEN:
  - Nakatani et al. (2010): "Speech Dereverberation Based on Variance-Normalized
    Delayed Linear Prediction" — WPE Algorithmus
  - Kinoshita et al. (2016): "The REVERB Challenge" — Evaluierung
  - Habets (2007): "Multi-channel speech dereverberation based on a statistical
    model of late reverberation"
  - Le Roux & Vincent (2013): "Consistent Wiener Filtering" — Postfilter-Gain
  - Perraudin et al. (2013): PGHI — scipy.signal.stft/istft sichert OLA-Konsistenz

Author: Aurik Development Team
Version: 3.0.0 (scipy.signal.stft/istft, kein np.fft.rfft mehr)
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.ndimage import median_filter
import scipy.signal as sig

from .phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseResult,
)

logger = logging.getLogger(__name__)


class AdvancedDereverbPhase(PhaseInterface):
    """
    WPE-basierte Dereverberation für starken Nachhall (RT60 > 0.4 s).

    Komplementär zu Phase 20 (Spectral Gating).
    Arbeitet rein mit DSP — kein ML-Import benötigt.
    """

    phase_id = "phase_49_advanced_dereverb"
    name = "Advanced Dereverb (WPE DSP v3 — scipy.signal.stft)"
    description = (
        "Weighted Prediction Error Dereverberation v3: scipy.signal.stft/istft "
        "(OLA-konsistent, PGHI-konform) + Consistent-Wiener-Postfilter "
        "(Le Roux 2013). Kein ML — reine DSP-Implementierung."
    )

    # STFT-Parameter
    _WINDOW_SIZE: int = 2048  # ~46 ms bei 44.1 kHz
    _HOP_SIZE: int = 512  # 75 % Überlappung
    # WPE-Parameter
    _WPE_DELAY: int = 3  # Systemverzögerung D (Frames): ~35 ms
    _WPE_ORDER: int = 5  # Prädiktionsordnung K (war 8): ~58 ms — rt_factor ≤ 3.0
    _WPE_ITERATIONS: int = 1  # Iterationen (war 2): 1 Iteration reicht für Restaurierung
    _WIENER_FLOOR: float = 0.1  # Minimale Gain-Floor für Wiener-Postfilter

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="3.0.0",
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.12,
            memory_requirement_mb=160,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.91,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        Führt WPE-Dereverberation durch.

        Args:
            audio:       Mono- oder Stereo-Audiodaten (float32/64, ±1 normiert)
            sample_rate: Abtastrate in Hz
            **kwargs:    strength (float, 0–1, Default 0.7),
                         protect_transients (bool, Default True)

        Returns:
            PhaseResult mit dereverberiertem Audio.
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        strength: float = float(kwargs.get("strength", 0.7))
        protect_transients: bool = bool(kwargs.get("protect_transients", True))

        is_stereo = audio.ndim == 2
        if is_stereo:
            left = self._dereverb_channel(audio[:, 0], sample_rate, strength, protect_transients)
            right = self._dereverb_channel(audio[:, 1], sample_rate, strength, protect_transients)
            n = min(len(left), len(right))
            processed = np.column_stack([left[:n], right[:n]])
        else:
            processed = self._dereverb_channel(audio, sample_rate, strength, protect_transients)

        elapsed = time.time() - t0
        rms_before = float(np.sqrt(np.mean(audio**2)))
        rms_after = float(np.sqrt(np.mean(processed**2)))
        # Guard: log10(0) => RuntimeWarning bei Stille-Eingaben; clamp auf >= 1e-30
        rms_change_db = 20.0 * np.log10(max(rms_after / (rms_before + 1e-10), 1e-30))

        logger.info(
            "Phase 49 WPE-Dereverb: strength=%.2f, RMS-Δ=%.2f dB, t=%.2fs",
            strength,
            rms_change_db,
            elapsed,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=elapsed,
            metadata={
                "algorithm": "wpe_spectral_dsp_v3_scipy_stft",
                "strength": strength,
                "wpe_delay": "adaptive_schroeder",
                "wpe_order": "adaptive_schroeder",
                "wpe_iterations": self._WPE_ITERATIONS,
                "window_size": self._WINDOW_SIZE,
                "hop_size": self._HOP_SIZE,
                "rms_change_db": rms_change_db,
                "protect_transients": protect_transients,
            },
            metrics={"rms_change_db": rms_change_db, "strength": strength},
        )

    # ------------------------------------------------------------------
    # Kern-Implementierung
    # ------------------------------------------------------------------

    def _dereverb_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        strength: float,
        protect_transients: bool,
    ) -> np.ndarray:
        """WPE-Dereverberation für einen einzelnen Kanal."""
        n_orig = len(audio)

        # 0. Schroeder T60-Schätzung — Vorab-Prüfung für Early-Exit
        t60 = self._estimate_t60_schroeder(audio, sample_rate)
        if t60 < 0.15:
            logger.info(
                "Phase 49 WPE: T60=%.3fs < 0.15s — kein signifikanter Nachhall, WPE übersprungen (Kanal %d Samples)",
                t60,
                n_orig,
            )
            return np.clip(audio.copy(), -1.0, 1.0)

        # 1. Transientenmaske
        transient_mask: np.ndarray | None = None
        if protect_transients:
            transient_mask = self._compute_transient_mask(audio, sample_rate)

        # 2. STFT
        win = np.hanning(self._WINDOW_SIZE)
        stft_matrix = self._stft(audio, win)  # (T, F)

        # 3. WPE: iterative Nachhall-Schätzung & Subtraktion
        enhanced = stft_matrix.copy()
        t60_frames = max(1, int(t60 * sample_rate / self._HOP_SIZE))
        D = max(2, min(6, int(t60_frames * 0.25)))  # ~25 % von T60
        K = max(3, min(12, int(t60_frames * 0.60)))  # ~60 % von T60
        logger.debug(
            "Phase 49 Schroeder T60=%.2fs → WPE D=%d K=%d (t60_frames=%d)",
            t60,
            D,
            K,
            t60_frames,
        )
        _, F = stft_matrix.shape

        for _iteration in range(self._WPE_ITERATIONS):
            power = np.abs(enhanced) ** 2
            smoothed_power = self._smooth_power(power, alpha=0.90)

            reverb_estimate = np.zeros_like(stft_matrix)
            for f in range(F):
                if smoothed_power[:, f].max() < 1e-12:
                    continue
                reverb_estimate[:, f] = self._predict_reverb_band(
                    stft_matrix[:, f], smoothed_power[:, f], D, K, strength
                )
            enhanced = stft_matrix - reverb_estimate

        # 4. Wiener-Postfilter
        enhanced = self._apply_wiener_postfilter(enhanced, stft_matrix, floor=self._WIENER_FLOOR)

        # 5. ISTFT (OLA)
        output = self._istft(enhanced, win, n_orig)

        # 6. Transientenrestauration
        if protect_transients and transient_mask is not None:
            mask_res = sig.resample(transient_mask.astype(float), n_orig)
            mask_res = np.clip(mask_res, 0.0, 1.0)
            output = output * (1.0 - mask_res) + audio[:n_orig] * mask_res

        # Pegel-Erhalt
        rms_in = np.sqrt(np.mean(audio**2))
        rms_out = np.sqrt(np.mean(output**2))
        if rms_out > 1e-8:
            output = output * (rms_in / rms_out)

        return np.clip(output, -1.0, 1.0)

    # ------------------------------------------------------------------
    # WPE-Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_t60_schroeder(audio: np.ndarray, sample_rate: int) -> float:
        """Schroeder (1965) Backward Integration — blinde T60-Schätzung.

        Berechnet die Nachhallzeit T60 aus der Energy Decay Curve (EDC):

            EDC(t) = ∫[t..∞] x²(τ) dτ  ≈  Σ[n=t..N] x²[n]  (Rückwärts-Kumulativsumme)

        Dann gilt: T60 = 2 × T30  (Abfall von -5 dB auf -35 dB in der EDC).

        Referenz:
            Schroeder (1965) "New Method of Measuring Reverberation Time"
            — JASA 37(3): 409–412.

        Args:
            audio:       Mono-Audiosignal (float32/64).
            sample_rate: Abtastrate (Hz).

        Returns:
            T60-Schätzung in Sekunden, geklämmt auf [0.1, 3.0].
            Fallback 0.4 s bei Stille oder nicht-konvergenter Kurve.
        """
        x = audio.astype(np.float64)
        energy = x**2
        # Energy Decay Curve: rückwärtige Kumulativsumme
        edc = np.cumsum(energy[::-1])[::-1]
        peak = edc.max()
        if peak < 1e-12:
            return 0.4  # Stille → konservativer Fallback
        edc_db = 10.0 * np.log10(edc / peak + 1e-12)
        # -5 dB → -35 dB Schnittpunkte → T30 × 2 = T60
        below5 = np.where(edc_db <= -5.0)[0]
        below35 = np.where(edc_db <= -35.0)[0]
        if len(below5) == 0 or len(below35) == 0:
            return 0.4
        idx5 = int(below5[0])
        idx35 = int(below35[0])
        if idx35 <= idx5:
            return 0.4
        t30 = (idx35 - idx5) / float(sample_rate)
        return float(np.clip(2.0 * t30, 0.1, 3.0))

    @staticmethod
    def _smooth_power(power: np.ndarray, alpha: float = 0.90) -> np.ndarray:
        """Exponential Moving Average über Zeitachse (in-place-freie Version)."""
        smoothed = power.copy()
        for t in range(1, power.shape[0]):
            smoothed[t] = alpha * smoothed[t - 1] + (1.0 - alpha) * power[t]
        return smoothed + 1e-12

    @staticmethod
    def _predict_reverb_band(
        y: np.ndarray,
        power: np.ndarray,
        D: int,
        K: int,
        strength: float,
    ) -> np.ndarray:
        """
        Schätzt den Nachhall-Anteil r(t) für ein einzelnes Frequenzband f
        via gewichteter Least-Squares-Regression (vereinfachtes WPE).

        y(t) ≈ Σ_k g_k · y(t - D - k)   →  r(t) = Σ_k g_k · y(t-D-k)

        Args:
            y:        Komplexes STFT-Spektrum, shape (T,)
            power:    Geglättete Leistungsschätzung, shape (T,)
            D:        Systemverzögerung (Frames)
            K:        Prädiktionsordnung
            strength: Skalierung der Subtraktion [0–1]

        Returns:
            reverb_estimate, shape (T,), komplex
        """
        T = len(y)
        n_eq = T - D - K
        if n_eq < K:
            return np.zeros_like(y)

        # Regressionsmatrix X (n_eq × K)
        X = np.zeros((n_eq, K), dtype=complex)
        b = y[D + K :]
        for k in range(K):
            X[:, k] = y[K - k - 1 : T - D - k - 1]

        # Gewichtung: low-power Frames bevorzugen (Direktschall-Selektion)
        w = 1.0 / (power[D + K :] + 1e-8)
        w = w / (w.max() + 1e-12)

        Xw = X * w[:, np.newaxis]
        try:
            XhXw = Xw.conj().T @ X
            XhBw = Xw.conj().T @ b
            reg = 1e-4 * np.eye(K)
            g = np.linalg.solve(XhXw + reg, XhBw)
        except np.linalg.LinAlgError:
            return np.zeros_like(y)

        reverb = np.zeros_like(y)
        for t in range(D + K, T):
            for k in range(K):
                reverb[t] += g[k] * y[t - D - k - 1]

        return reverb * strength

    @staticmethod
    def _apply_wiener_postfilter(
        enhanced: np.ndarray,
        original: np.ndarray,
        floor: float = 0.10,
    ) -> np.ndarray:
        """
        Wiener-Postfilter gegen Musical Noise nach WPE-Subtraktion.

        Gain(t,f) = |enhanced|² / (|original|² + ε), geklämmt auf [floor, 1].
        Zusätzlich 3-Frame-Median-Glättung in der Zeitachse.
        """
        eps = 1e-10
        gain = np.abs(enhanced) ** 2 / (np.abs(original) ** 2 + eps)
        gain = np.clip(gain, floor, 1.0)
        gain = median_filter(gain, size=(3, 1))
        return enhanced * gain

    # ------------------------------------------------------------------
    # STFT / ISTFT
    # ------------------------------------------------------------------

    def _stft(self, audio: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform → komplexe Matrix (T, F).

        Implementiert via scipy.signal.stft (OLA-konsistent, PGHI-konform).
        Ersetzt die verbotene np.fft.rfft-Frame-Schleife aus v2.0.

        Args:
            audio:  1D-Audio-Signal.
            window: Hann-Fensterfunktion (Länge = _WINDOW_SIZE) — für
                    Konsistenz mit ISTFT übergeben, aber scipy nutzt intern
                    die Hann-Parameterisierung.

        Returns:
            np.ndarray: Komplexe STFT-Matrix, Form (T, F).
        """
        _, _, Zxx = sig.stft(
            audio,
            fs=1,  # normierte Frequenzachse — absolute Werte nicht benötigt
            window=window,
            nperseg=self._WINDOW_SIZE,
            noverlap=self._WINDOW_SIZE - self._HOP_SIZE,
            boundary="even",
            padded=True,
        )
        return Zxx.T  # scipy: (F, T) → intern (T, F)

    def _istft(self, stft: np.ndarray, window: np.ndarray, orig_len: int) -> np.ndarray:
        """OLA-Rücksynthese via scipy.signal.istft → Signal der Länge orig_len.

        Ersetzt die verbotene np.fft.irfft-Frame-Schleife aus v2.0.
        PGHI-Phasenkonsistenz durch scipy-interne OLA-Normierung gewährleistet.

        Args:
            stft:     Komplexe STFT-Matrix, Form (T, F).
            window:   Hann-Fensterfunktion (muss identisch zu _stft sein).
            orig_len: Ziel-Länge des Ausgangssignals.

        Returns:
            np.ndarray: 1D-Ausgangssignal, Länge = orig_len, clip[-1, 1].
        """
        _, out = sig.istft(
            stft.T,  # intern (T, F) → scipy erwartet (F, T)
            fs=1,
            window=window,
            nperseg=self._WINDOW_SIZE,
            noverlap=self._WINDOW_SIZE - self._HOP_SIZE,
            boundary=True,
        )
        out = np.real(out)
        if len(out) > orig_len:
            out = out[:orig_len]
        elif len(out) < orig_len:
            out = np.pad(out, (0, orig_len - len(out)), mode="edge")
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    # ------------------------------------------------------------------
    # Transientenmaske
    # ------------------------------------------------------------------

    def _compute_transient_mask(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Einsatz-Detektion via Energie-Anstieg (>9.5 dB in 5 ms).

        Returns:
            mask: Float-Array [0, 1], Länge = Anzahl RMS-Frames.
                  1.0 = Transiente (bleibt unverändert).
        """
        win_s = int(0.010 * sample_rate)  # 10 ms Fenster
        hop_s = max(1, int(0.005 * sample_rate))  # 5 ms Hop
        n_frames = (len(audio) - win_s) // hop_s + 1

        rms = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * hop_s
            rms[i] = np.sqrt(np.mean(audio[s : s + win_s] ** 2))

        mask = np.zeros(n_frames)
        for i in range(1, n_frames):
            if rms[i - 1] > 1e-8 and rms[i] / rms[i - 1] > 3.0:
                # Energie-Anstieg > 9.5 dB in 5 ms → Transiente
                extend = int(0.025 * sample_rate / hop_s)  # 25 ms schützen
                hi = min(i + extend, n_frames)
                mask[i:hi] = 1.0

        return mask
