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
import scipy.signal as sig
from scipy.ndimage import median_filter

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

    _PHASE_ID = "phase_49_advanced_dereverb"
    _NAME = "Advanced Dereverb (WPE DSP v3 — scipy.signal.stft)"
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
    _MAX_RMS_DROP_DB = {
        "tape": 2.5,
        "reel_tape": 2.2,
        "cassette": 2.8,
        "vinyl": 2.0,
        "shellac": 1.8,
        "wax_cylinder": 1.5,
        "unknown": 2.0,
    }

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
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

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        effective_strength = float(kwargs.get("strength", 0.7)) * phase_locality_factor
        effective_strength = float(np.clip(effective_strength, 0.0, 1.0))

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry,
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        strength = effective_strength
        protect_transients: bool = bool(kwargs.get("protect_transients", True))
        # Store material type for EMA-alpha selection in _dereverb_channel
        self._current_material = str(kwargs.get("material_type", "unknown"))

        is_stereo = audio.ndim == 2
        if is_stereo:
            # §2.51 M/S: derive room envelope from Mid-channel only so both
            # channels are attenuated by the SAME gain curve — independent L/R
            # processing assigns different room estimates and causes stereo-field
            # asymmetry that triggers §2.49 phase-cancellation rollbacks.
            _sqrt2 = np.sqrt(2.0)
            _mid = (audio[:, 0] + audio[:, 1]) / _sqrt2
            _side = (audio[:, 0] - audio[:, 1]) / _sqrt2
            _mid_dry = self._dereverb_channel(_mid, sample_rate, strength, protect_transients)
            # Side: apply weaker dereverb (side information is already less reverberant)
            _side_strength = strength * 0.5
            _side_dry = self._dereverb_channel(_side, sample_rate, _side_strength, protect_transients)
            _n = min(len(_mid_dry), len(_side_dry))
            _l = (_mid_dry[:_n] + _side_dry[:_n]) / _sqrt2
            _r = (_mid_dry[:_n] - _side_dry[:_n]) / _sqrt2
            processed = np.column_stack([_l, _r])
        else:
            processed = self._dereverb_channel(audio, sample_rate, strength, protect_transients)

        # Make PMGG strength retries audibly monotonic: explicit wet/dry blend at
        # the phase output, independent from internal WPE model scaling.
        wet_mix = float(np.clip(effective_strength, 0.0, 1.0))
        attenuation_guard_triggered = False
        attenuation_guard_factor = 1.0

        rms_before = float(np.sqrt(np.mean(audio**2)))
        rms_processed = float(np.sqrt(np.mean(processed**2)))
        rms_drop_db = 20.0 * np.log10(max(rms_processed / (rms_before + 1e-10), 1e-30))

        # Safety rescue for catastrophic dereverb attenuation. If the raw processed
        # signal collapses energy too aggressively, reduce wet mix preemptively.
        _max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(self._current_material, self._MAX_RMS_DROP_DB["unknown"]))
        if rms_drop_db < -_max_rms_drop_db and wet_mix > 0.0:
            attenuation_guard_triggered = True
            attenuation_guard_factor = float(np.clip(_max_rms_drop_db / (abs(rms_drop_db) + 1e-9), 0.35, 1.0))
            wet_mix *= attenuation_guard_factor

        # --- §4.5c Early-Reflection-Guard: C80/D50 clarity-based wet-mix limiting ---
        # Measure C80 (Kuttruff 2009) on original and processed signal.
        # Spec limits: ΔC80 ≤ 6 dB, ΔD50 ≤ 0.12.  Values exceeding bounds → reduce wet_mix
        # or blend early reflections back (alpha=0.35 for first 50 ms).
        c80_pre = self._measure_c80_proxy(audio, sample_rate)
        c80_post = self._measure_c80_proxy(processed, sample_rate)
        delta_c80 = c80_post - c80_pre
        c80_guard_triggered = False
        early_blend_triggered = False

        # §4.5c D50 Guard — ΔD50 ≤ 0.12 (Deutlichkeitsmaß)
        d50_pre = self._measure_d50_proxy(audio, sample_rate)
        d50_post = self._measure_d50_proxy(processed, sample_rate)
        delta_d50 = d50_post - d50_pre

        if delta_c80 < -2.0:
            # C80 degraded — full rollback to original dry signal
            logger.warning("Phase 49 C80-guard: ΔC80=%.2f dB below −2 dB → rollback to dry", delta_c80)
            processed = audio.copy()
            wet_mix = 0.0
            c80_guard_triggered = True
        elif delta_c80 > 6.0:
            # Excessive clarity boost → scale wet_mix proportionally
            _c80_scale = float(np.clip(6.0 / (delta_c80 + 1e-9), 0.30, 1.0))
            wet_mix *= _c80_scale
            c80_guard_triggered = True
            logger.info(
                "Phase 49 C80-guard: ΔC80=%.2f dB > 6 dB → wet_mix scaled to %.3f",
                delta_c80,
                wet_mix,
            )
        elif delta_c80 > 4.0:
            # Moderate clarity boost → blend 35 % of original early reflections
            # back into the first 50 ms (spec §4.5c alpha=0.35 early-reflection blend)
            early_blend_triggered = True
            _early_win = int(sample_rate * 0.050)
            _alpha = 0.35
            _proc64 = processed.copy().astype(np.float64)
            _orig64 = audio.copy().astype(np.float64)
            if _proc64.ndim == 2:
                for _ch in range(_proc64.shape[1]):
                    _e = min(_early_win, _proc64.shape[0])
                    _proc64[:_e, _ch] = (1.0 - _alpha) * _proc64[:_e, _ch] + _alpha * _orig64[:_e, _ch]
            else:
                _e = min(_early_win, len(_proc64))
                _proc64[:_e] = (1.0 - _alpha) * _proc64[:_e] + _alpha * _orig64[:_e]
            processed = _proc64.astype(np.float32)
            logger.info("Phase 49 C80-guard: ΔC80=%.2f dB — early-reflection blend 35 %% applied (50 ms)", delta_c80)

        # §4.5c D50 secondary guard: ΔD50 > 0.12 → further reduce wet
        if abs(delta_d50) > 0.12 and not c80_guard_triggered:
            _d50_scale = float(np.clip(0.12 / (abs(delta_d50) + 1e-9), 0.30, 1.0))
            wet_mix *= _d50_scale
            logger.info(
                "Phase 49 D50-guard: ΔD50=%.3f > 0.12 → wet_mix scaled to %.3f",
                delta_d50,
                wet_mix,
            )

        if wet_mix < 1.0:
            processed = audio + wet_mix * (processed - audio)

        # Final dry/wet rescue: keep dereverb effective, but do not allow an
        # audible loudness collapse after all clarity guards have been applied.
        rms_after_blend = float(np.sqrt(np.mean(processed**2)))
        rms_drop_after_blend_db = 20.0 * np.log10(max(rms_after_blend / (rms_before + 1e-10), 1e-30))
        if rms_drop_after_blend_db < -_max_rms_drop_db and wet_mix > 0.0:
            _rescue_wet = float(
                np.clip(wet_mix * (_max_rms_drop_db / (abs(rms_drop_after_blend_db) + 1e-9)), 0.20, wet_mix)
            )
            processed = audio + _rescue_wet * (processed - audio)
            wet_mix = _rescue_wet

        elapsed = time.time() - t0
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
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "wpe_delay": "adaptive_schroeder",
                "wpe_order": "adaptive_schroeder",
                "wpe_iterations": self._WPE_ITERATIONS,
                "window_size": self._WINDOW_SIZE,
                "hop_size": self._HOP_SIZE,
                "wet_mix": wet_mix,
                "attenuation_guard_triggered": attenuation_guard_triggered,
                "attenuation_guard_factor": attenuation_guard_factor,
                "rms_change_db": rms_change_db,
                "rms_drop_db": round(float(min(0.0, rms_change_db)), 3),  # §2.45a telemetry
                "loudness_makeup_db": 0.0,  # phase_49 uses wet-rescue, not makeup gain
                "protect_transients": protect_transients,
                "c80_pre": float(c80_pre),
                "c80_post": float(c80_post),
                "delta_c80": float(delta_c80),
                "c80_guard_triggered": c80_guard_triggered,
                "early_blend_triggered": early_blend_triggered,
            },
            metrics={"rms_change_db": rms_change_db, "strength": strength, "effective_strength": effective_strength},
        )

    # ------------------------------------------------------------------
    # Kern-Implementierung
    # ------------------------------------------------------------------

    def _measure_c80_proxy(self, audio: np.ndarray, sr: int) -> float:
        """Estimate Clarity C80 from time-domain energy ratio (Kuttruff 2009).

        C80 = 10 × log10(E_early / E_late)
        where E_early = energy in first 80 ms, E_late = energy from 80 ms onward.

        Used by spec §4.5c Early-Reflection-Guard to limit dereverberation depth.
        Returns 0.0 when audio is silent or shorter than 80 ms.
        """
        mono = audio[:, 0] if audio.ndim == 2 else audio
        early_win = int(sr * 0.080)  # 80 ms
        if len(mono) <= early_win:
            return 0.0
        e_early = float(np.sum(mono[:early_win] ** 2))
        e_late = float(np.sum(mono[early_win:] ** 2))
        if e_late < 1e-12:
            return 0.0
        return 10.0 * float(np.log10(max(e_early, 1e-12) / e_late))

    def _measure_d50_proxy(self, audio: np.ndarray, sr: int) -> float:
        """Estimate Definition D50 from time-domain energy ratio.

        D50 = E_early50 / E_total  where E_early50 = energy in first 50 ms.
        Returns 0.0 when audio is silent.
        """
        mono = audio[:, 0] if audio.ndim == 2 else audio
        early_win = int(sr * 0.050)  # 50 ms
        e_total = float(np.sum(mono**2))
        if e_total < 1e-12:
            return 0.0
        e_early50 = float(np.sum(mono[:early_win] ** 2))
        return float(np.clip(e_early50 / e_total, 0.0, 1.0))

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
            # alpha=0.90 is fast (high noise-floor reactivity) — good for vinyl/cassette
            # reverb bursts.  Tape material has a longer reverb tail and gentler room
            # impulse; alpha=0.93 averages over more frames, preventing over-dereverb
            # that strips authentic Tape room character (§0 Authenticity).
            _ema_alpha = 0.93 if getattr(self, "_current_material", "") in ("tape", "reel_tape") else 0.90
            smoothed_power = self._smooth_power(power, alpha=_ema_alpha)

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
        """Exponential Moving Average über Zeitachse — vectorized via lfilter."""
        # EMA: y[t] = alpha * y[t-1] + (1 - alpha) * x[t]
        # As IIR filter: b = [1 - alpha], a = [1, -alpha]
        b = np.array([1.0 - alpha])
        a = np.array([1.0, -alpha])
        # Apply per frequency bin (power shape: (T, F) or (T,))
        if power.ndim == 1:
            smoothed = sig.lfilter(b, a, power)
        else:
            smoothed = sig.lfilter(b, a, power, axis=0)
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

        # Vectorized reverb prediction: r(t) = Σ_k g_k · y(t-D-k-1)
        # The inner loop is a FIR filter (convolution) of y with g (reversed)
        g_rev = g[::-1]  # reverse g for convolution
        # Convolve y with g_rev: output[t] = Σ_k g_rev[k] * y[t-K+1+k] = Σ_k g[K-1-k] * y[t-K+1+k]
        # We need: reverb[t] = Σ_k g[k] * y[t-D-k-1] for t >= D+K
        # Shift: let s = t - D - 1, then reverb[s+D+1] = Σ_k g[k] * y[s-k]
        # This is: convolve(y, g) at position s, valid for s >= K-1
        np.convolve(y, g_rev, mode="full")  # length T + K - 1
        # conv_full[s] = Σ_k g_rev[k] * y[s-k] = Σ_k g[K-1-k] * y[s-k]
        # We want reverb[t] = Σ_k g[k] * y[t-D-k-1]
        # Let s = t - D - 1: reverb[t] = conv_g[s] where conv_g[s] = Σ_k g[k] * y[s-k]
        # conv_g = convolve(y, g) — use g directly (not reversed) since numpy convolve already flips
        conv_result = np.convolve(y, g, mode="full")  # length T + K - 1
        # conv_result[s] = Σ_k g[k] * y[s-k], valid when s >= K-1
        # reverb[t] = conv_result[t - D - 1] for t >= D + K (i.e. t-D-1 >= K-1)
        reverb = np.zeros_like(y)
        valid_start = D + K
        valid_end = min(T, T)  # conv_result has enough entries
        src_start = valid_start - D - 1  # = K - 1
        src_end = valid_end - D - 1
        if src_end > src_start and src_end <= len(conv_result):
            reverb[valid_start:valid_end] = conv_result[src_start:src_end]

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
