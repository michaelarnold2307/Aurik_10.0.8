#!/usr/bin/env python3
"""
Phase 28: Surface Noise Profiling v3.0 - Über-SOTA OMLSA/IMCRA
Adaptive spectral noise profiling via IMCRA minimum statistics
und OMLSA-Gain (Optimally Modified Log-Spectral Amplitude).

Algorithm Overview (v3.0):
1. IMCRA-Rauscheschätzung (Iterative Minimum Controlled Recursive Averaging):
   - Bias-korrigiertes gleitendes Minimum (≈1.5s Fenster, b_min=1.66)
   - Exponentielle Glättung α_n=0.85
   - Gibt F×T-Rauschleistungsmatrix zurück
2. OMLSA-Gain (Cohen 2003):
   - γ(t,f) = |Y(t,f)|² / σ²_n(t,f)
   - ξ(t,f) = max(γ-1, 0)  [a-priori SNR]
   - v = ξ·γ / (1+ξ)
   - Λ(t,f) = exp(-ξ + v)  [likelihood]
   - p(t,f) = 1/(1 + q/präsenzÜberβ(Λ))  [Sprachpräsenz]
   - G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p   G_floor=0.1
3. Cappe-Gain-Glättung (1994):
   - Temporale Glättung mit materialadaptiver Zeitkonstante
4. Material-Adaptierung:
   - Shellac: aggressiv (hohes Oberflächenrauschen)
   - Vinyl: ausgewogen (Crackle + Oberflächenrauschen)
   - Tape: konservativ (hauptsächlich Tape-Hiss)
   - Digital: minimal (Dithering-Rauschen)

Scientific Foundation:
- Cohen & Berdugo (2002): IMCRA — primär
- Cohen (2003): OMLSA — primär
- Cappé (1994): Elimination of the Musical Noise Phenomenon — Gain-Glättung
- Le Roux & Vincent (2013): Consistent Wiener Filtering — Phasenkonsistenz
- Ephraim & Malah (1984): historische Referenz — NICHT primär eingesetzt
- Martin (2001): Minimum Statistics — Basis-Konzept für IMCRA

Industry Benchmarks:
- iZotope RX Spectral De-noise ($399)
- Cedar DNS (Adaptive noise suppressor, $2000+)
- Waves Z-Noise ($49)
- Accusonus ERA-N ($99)
- Acon Digital DeNoise ($99)

Quality Target: 0.78 → 0.94 (+20% improvement)
Performance Target: <0.25× realtime

Author: Aurik Development Team
Version: 2.0.0 Professional
"""

import logging
import time
from typing import Any

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# PGHI phase reconstruction after spectral gain application (Spec §DSP — PFLICHT)
try:
    from dsp.pghi import pghi_reconstruct_from_stft as _pghi_from_stft

    _PGHI_AVAILABLE = True
except Exception:
    _PGHI_AVAILABLE = False
    logger.warning("PGHI not available; scipy.signal.istft fallback active for phase_28")


class SurfaceNoiseProfiling(PhaseInterface):
    """
    Professional Surface Noise Profiling with Wiener Filtering.

    Key Features:
    - Multi-pass VAD for noise-only region detection
    - Minimum statistics noise tracking
    - Wiener filter with SNR-based gains
    - Over-subtraction with frequency-dependent flooring
    - Temporal gain smoothing (reduce musical noise)
    - Material-adaptive parameters

    Use Cases:
    - Vinyl surface noise reduction
    - Shellac crackle suppression
    - Tape hiss removal
    - Digital dithering noise cleanup

    Performance: <0.25× realtime on modern CPU
    """

    # STFT parameters
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    # Material-adaptive noise reduction configurations
    NOISE_CONFIG = {
        MaterialType.SHELLAC: {
            "over_subtraction_alpha": 2.8,
            "spectral_floor": 0.12,
            "vad_threshold_db": -38,
            "smoothing_frames": 8,
            "noise_learn_duration_s": 1.5,
        },
        MaterialType.VINYL: {
            "over_subtraction_alpha": 2.2,
            "spectral_floor": 0.08,
            "vad_threshold_db": -42,
            "smoothing_frames": 10,
            "noise_learn_duration_s": 1.2,
        },
        MaterialType.TAPE: {
            "over_subtraction_alpha": 1.8,
            "spectral_floor": 0.06,
            "vad_threshold_db": -48,
            "smoothing_frames": 12,
            "noise_learn_duration_s": 1.0,
        },
        MaterialType.CD_DIGITAL: {
            "over_subtraction_alpha": 1.3,
            "spectral_floor": 0.04,
            "vad_threshold_db": -55,
            "smoothing_frames": 15,
            "noise_learn_duration_s": 0.8,
        },
        MaterialType.STREAMING: {
            "over_subtraction_alpha": 1.2,
            "spectral_floor": 0.03,
            "vad_threshold_db": -60,
            "smoothing_frames": 15,
            "noise_learn_duration_s": 0.5,
        },
    }

    _MAX_RMS_DROP_DB = {
        "tape": 2.0,
        "reel_tape": 1.8,
        "cassette": 2.2,
        "vinyl": 1.5,
        "shellac": 1.2,
        "wax_cylinder": 1.0,
        "cd_digital": 1.2,
        "streaming": 1.2,
        "unknown": 1.5,
    }

    def __init__(self):
        super().__init__()
        self.name = "Surface Noise Profiling v2 Professional"

    def get_metadata(self) -> PhaseMetadata:
        """Return phase metadata."""
        return PhaseMetadata(
            phase_id="phase_28_surface_noise_profiling",
            name="Surface Noise Profiling v3 OMLSA/IMCRA",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=5,
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.25,
            version="3.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.95,
            description="IMCRA-Rauschsch\u00e4tzung + OMLSA-Gain (Cohen 2002/2003) \u2014 \u00dcber-SOTA",
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.CD_DIGITAL, **kwargs
    ) -> PhaseResult:
        """
        Apply spectral noise profiling and removal.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type for adaptive processing

        Returns:
            PhaseResult with denoised audio
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()
        self.validate_input(audio)

        phase_locality_factor = float(kwargs.get("phase_locality_factor", 1.0))
        phase_locality_factor = float(np.clip(phase_locality_factor, 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", 1.0))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))

        if _effective_strength <= 0.0:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                execution_time_seconds=time.time() - start_time,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "material": material.name,
                    "noise_reduction_db": 0.0,
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": _effective_strength,
                },
                warnings=[],
            )

        is_stereo = audio.ndim == 2
        config = dict(self.NOISE_CONFIG.get(material, self.NOISE_CONFIG[MaterialType.CD_DIGITAL]))
        config["over_subtraction_alpha"] = float(1.0 + (config["over_subtraction_alpha"] - 1.0) * _effective_strength)
        config["spectral_floor"] = float(
            np.clip(1.0 - (1.0 - config["spectral_floor"]) * _effective_strength, 0.03, 1.0)
        )
        config["smoothing_frames"] = int(max(3, round(config["smoothing_frames"] + (1.0 - _effective_strength) * 6.0)))

        # Process each channel
        if is_stereo:
            denoised_left, noise_db_left = self._denoise_channel(audio[:, 0], sample_rate, config)
            denoised_right, noise_db_right = self._denoise_channel(audio[:, 1], sample_rate, config)
            denoised_audio = np.column_stack((denoised_left, denoised_right))
            avg_noise_db = (noise_db_left + noise_db_right) / 2
        else:
            denoised_audio, avg_noise_db = self._denoise_channel(audio, sample_rate, config)

        if 0.0 < _effective_strength < 1.0:
            denoised_audio = audio + _effective_strength * (denoised_audio - audio)

        denoised_audio, loudness_stats = self._apply_material_loudness_preservation(
            audio,
            denoised_audio,
            material,
        )

        execution_time = time.time() - start_time
        rt_factor = execution_time / (len(audio) / sample_rate)

        denoised_audio = np.nan_to_num(denoised_audio, nan=0.0, posinf=0.0, neginf=0.0)
        denoised_audio = np.clip(denoised_audio, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=denoised_audio,
            execution_time_seconds=execution_time,
            metadata={
                "material": material.name,
                "noise_reduction_db": float(avg_noise_db),
                "over_subtraction_alpha": float(config["over_subtraction_alpha"]),
                "rt_factor": float(rt_factor),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": loudness_stats["rms_drop_db"],
                "loudness_makeup_db": loudness_stats["makeup_gain_db"],
            },
            warnings=[] if rt_factor < 0.30 else [f"Performance sub-optimal: {rt_factor:.2f}× realtime"],
        )

    def _apply_material_loudness_preservation(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        material: MaterialType,
    ) -> tuple[np.ndarray, dict[str, float]]:
        material_key = getattr(material, "name", str(material)).lower()
        max_rms_drop_db = float(self._MAX_RMS_DROP_DB.get(material_key, self._MAX_RMS_DROP_DB["unknown"]))

        rms_in = float(np.sqrt(np.mean(np.asarray(original_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
        rms_drop_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30)) if rms_in > 1e-8 else 0.0
        makeup_gain_db = 0.0

        if rms_in > 1e-8 and rms_drop_db < -max_rms_drop_db:
            target_rms_drop_db = -max_rms_drop_db
            required_gain_db = target_rms_drop_db - rms_drop_db
            current_peak = float(np.percentile(np.abs(processed_audio), 99.9) + 1e-12)
            max_safe_gain_db = max(0.0, -1.5 - 20.0 * np.log10(current_peak))
            makeup_gain_db = float(np.clip(required_gain_db, 0.0, max_safe_gain_db))
            if makeup_gain_db > 0.0:
                processed_audio = np.clip(
                    processed_audio * (10.0 ** (makeup_gain_db / 20.0)),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                rms_out = float(np.sqrt(np.mean(np.asarray(processed_audio, dtype=np.float64) ** 2) + 1e-12))
                rms_drop_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-30))
                logger.info(
                    "Phase 28 loudness-preservation: material=%s rms_drop=%.2f dB via makeup %.2f dB",
                    material_key,
                    rms_drop_db,
                    makeup_gain_db,
                )

        return processed_audio, {
            "rms_drop_db": round(float(rms_drop_db), 3),
            "makeup_gain_db": round(float(makeup_gain_db), 3),
        }

    def _denoise_channel(self, audio: np.ndarray, sample_rate: int, config: dict[str, Any]) -> tuple[np.ndarray, float]:
        """Entfernt Oberflächenrauschen via IMCRA-Rauschschätzung + OMLSA-Gain.

        Algorithmus (v3.0):
            1. STFT mit 75% Overlap (nperseg=2048, noverlap=1536)
            2. IMCRA: Bias-korrigiertes gleitendes Minimum → F×T Rauschleistung
            3. OMLSA: G(t,f) = G_floor^(1-p) * G_H1^p, G_floor=0.1
            4. Cappé-Glättung: Gain temporal geglättet
            5. ISTFT + NaN-Schutz + clip
        """
        # Step 1: STFT (75% Overlap)
        nperseg = self.FRAME_SIZE
        noverlap = nperseg - self.HOP_SIZE
        _, t_arr, stft = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")

        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Step 2: IMCRA-Rauschschätzung (F×T-Matrix)
        noise_mag = self._estimate_noise_imcra(magnitude, t_arr, config)

        # Step 3: OMLSA-Gain
        gain = self._compute_omlsa_gain(magnitude, noise_mag, config)

        # Step 4: Cappé-Gain-Glättung
        alpha_g = 1.0 - 1.0 / max(config["smoothing_frames"], 1)
        gain_smooth = np.zeros_like(gain)
        gain_smooth[:, 0] = gain[:, 0]
        for ti in range(1, gain.shape[1]):
            gain_smooth[:, ti] = alpha_g * gain_smooth[:, ti - 1] + (1.0 - alpha_g) * gain[:, ti]
        gain_smooth = np.nan_to_num(gain_smooth, nan=1.0, posinf=1.0, neginf=0.1)
        gain_smooth = np.clip(gain_smooth, 0.1, 1.0)

        # Step 5: Spectrum anwenden
        cleaned_mag = magnitude * gain_smooth
        cleaned_stft = cleaned_mag * np.exp(1j * phase)

        # Step 6: PGHI-Rekonstruktion (Perraudin 2013) — bevorzugt statt direktem iSTFT
        # PGHI bewahrt interaurale Phasendifferenzen → Raumtiefe + Tiefen-Immersion (Spec §8.3)
        if _PGHI_AVAILABLE:
            try:
                denoised = _pghi_from_stft(
                    cleaned_stft, sr=sample_rate, win_size=nperseg, hop=noverlap, n_samples=len(audio)
                )
            except Exception as _pghi_exc:
                logger.debug("phase_28 PGHI failed, fallback to istft: %s", _pghi_exc)
                _, denoised = signal.istft(
                    cleaned_stft, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann"
                )
        else:
            _, denoised = signal.istft(cleaned_stft, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")

        # Länge anpassen + NaN/Clipping-Schutz
        denoised = denoised[: len(audio)]
        if len(denoised) < len(audio):
            denoised = np.pad(denoised, (0, len(audio) - len(denoised)))
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0)
        denoised = np.clip(denoised, -1.0, 1.0)

        # Rauschreduktion schätzen
        rms_in = np.sqrt(np.mean(audio**2) + 1e-12)
        rms_out = np.sqrt(np.mean(denoised**2) + 1e-12)
        noise_reduction_db = 20.0 * np.log10(rms_in / rms_out)
        if not np.isfinite(noise_reduction_db):
            noise_reduction_db = 0.0

        return denoised, noise_reduction_db

    def _estimate_noise_imcra(self, magnitude: np.ndarray, t_arr: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """IMCRA-Rauschschätzung: Bias-korrigiertes gleitendes Minimum (Cohen & Berdugo 2002).

        Algorithmus:
            sigma²_n(t,f) = b_min * min_{tau in [t-M, t]}( P_hat(tau,f) )
            P_hat(t,f) alpha_n * P_hat(t-1,f) + (1-alpha_n) * |Y(t,f)|²

        Args:
            magnitude: STFT-Betrag (F × T)
            t_arr:     Zeitstempel der STFT-Frames (T,)
            config:    Phasen-Konfiguration (enthält 'smoothing_frames')

        Returns:
            noise_mag: Rauschbetrag (F × T), NaN-frei
        """
        F, T = magnitude.shape
        b_min = 1.66  # Bias-Korrekturfaktor (Cohen 2003)
        alpha_n = 0.85  # Glättungskoeffizient für Rauschleistung
        eps = 1e-10

        # Fensterbreite ≈1.5s oder mind. 15 Frames
        hop_s = float(t_arr[1] - t_arr[0]) if T > 1 and len(t_arr) > 1 else 0.01
        M = max(15, round(1.5 / hop_s))

        # Geglättete Leistung P_hat (F × T)
        P_hat = magnitude**2
        for ti in range(1, T):
            P_hat[:, ti] = alpha_n * P_hat[:, ti - 1] + (1.0 - alpha_n) * magnitude[:, ti] ** 2
        P_hat = np.nan_to_num(P_hat, nan=eps)

        # Gleitendes Minimum über M Frames
        noise_power = np.zeros((F, T), dtype=np.float64)
        for ti in range(T):
            start = max(0, ti - M)
            noise_power[:, ti] = np.min(P_hat[:, start : ti + 1], axis=1)

        # Bias-Korrektur + Wurzel → Rauschbetrag
        noise_mag = np.sqrt(np.maximum(b_min * noise_power, eps))
        return np.nan_to_num(noise_mag, nan=eps, posinf=eps, neginf=eps)

    def _compute_omlsa_gain(self, magnitude: np.ndarray, noise_mag: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        """OMLSA-Gain: G(t,f) = G_floor^(1-p) * (xi/(1+xi))^p  (Cohen 2003).

        Formel:
            gamma = |Y|² / sigma²_n          (a-posteriori SNR)
            xi    = max(gamma - 1, 0)          (a-priori SNR, Decision-Directed)
            v     = clip(xi * gamma / (1+xi), 0, 500)
            Lambda = exp(-xi + v)              (Likelihood-Verhältnis)
            q     = materialabhängige Rauschprior  (aus config)
            p     = 1 / (1 + q/((1-q)*Lambda + eps))
            G_H1  = xi / (1 + xi)
            G     = exp((1-p)*ln(G_floor) + p*ln(G_H1 + eps))

        Args:
            magnitude: STFT-Betrag (F × T)
            noise_mag: Rauschbetrag aus IMCRA (F × T)
            config:    Enthält 'spectral_floor' (= G_floor, material-adaptiv)

        Returns:
            G: OMLSA-Gain (F × T) in [G_floor, 1.0]
        """
        G_floor = float(config.get("spectral_floor", 0.1))
        G_floor = max(G_floor, 0.05)
        # q: Rausch-Präsenz-Prior (material-adaptiv: Shellac aggressiver)
        q = 1.0 - float(config.get("spectral_floor", 0.1))  # höheres floor → mehr Rauschen erwartet
        q = np.clip(q, 0.05, 0.95)
        eps = 1e-10

        sigma2_n = np.maximum(noise_mag**2, eps)
        gamma = np.maximum(magnitude**2 / sigma2_n, 0.0)  # a-posteriori SNR
        xi = np.maximum(gamma - 1.0, 0.0)  # a-priori SNR
        v = np.clip(xi * gamma / (xi + 1.0 + eps), 0.0, 500.0)
        lam = np.exp(np.clip(-xi + v, -50.0, 50.0))  # Likelihood-Verhältnis
        p = 1.0 / (1.0 + q / ((1.0 - q) * lam + eps))  # Sprachpräsenzwahrsch.
        G_H1 = xi / (xi + 1.0 + eps)  # Wiener-Gain bei Signal

        log_G = (1.0 - p) * np.log(G_floor) + p * np.log(np.maximum(G_H1, eps))
        log_G = np.clip(log_G, np.log(G_floor), 0.0)
        G = np.exp(log_G)
        G = np.nan_to_num(G, nan=G_floor, posinf=1.0, neginf=G_floor)
        return np.clip(G, G_floor, 1.0)
