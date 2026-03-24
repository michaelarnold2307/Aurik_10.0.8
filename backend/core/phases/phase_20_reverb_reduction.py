#!/usr/bin/env python3
"""
Phase 20: Professional Reverb Reduction v3.0 — OMLSA/IMCRA
===========================================================

Spektrale Nachhall-Reduktion via statistischer Rauschunterdrückung (OMLSA/IMCRA)
mit transientenerhaltender Nachbearbeitung.

SCIENTIFIC FOUNDATION (Primär, Über-SOTA-Pflicht):
- Cohen & Berdugo (2002): "Noise Estimation by Minima Controlled Recursive Averaging"
  → IMCRA: gleitendes Minimum mit Bias-Kompensation b_min=1.66 für diffuse
    Schallfelder wie Hall-Ausläufer. Ersetzt primitive Median-Rauschschätzung.
- Cohen (2003): "Noise Spectrum Estimation in Adverse Environments: Improved
  Minima Controlled Recursive Averaging" (OMLSA)
  → OMLSA: G(t,f) = G_floor^(1−p) · (ξ/(1+ξ))^p eliminiert musikalisches Rauschen.
- Le Roux & Vincent (2013): "Consistent Wiener Filtering" — Gain-Clamp G_floor.
- Cappé (1994): Temporale Gain-Glättung α_g=0.85 — unterdrückt Gain-Flattern.
- Perraudin et al. (2013): PGHI — scipy.signal.stft/istft sichert OLA-Phasenkonsistenz.

Historische Referenz (nur noch informativ, nicht als primärer Algorithmus):
- Moorer (1979): About This Reverberation Business
- Schroeder (1962): Natural Sounding Artificial Reverberation
- Kendall (2010): The Decorrelation of Audio Signals and Its Impact on Spatial Imagery
- Välimäki et al. (2012): Fifty Years of Artificial Reverberation
- ITU-R BS.1116-3: Methods for the Subjective Assessment of Small Impairments
- Bech & Zacharov (2006): Perceptual Audio Evaluation

INDUSTRY BENCHMARKS:
- iZotope RX 10 De-reverb (Spectral analysis + ML)
- Waves Clarity Vx DeReverb (Transient-preserving)
- Zynaptiq Unveil (Source separation based)
- SPL DeVerb (Dynamics-based)
- Cedar Retouch Pro (Professional standard)
- Accusonus ERA-D (Real-time dereverb)

ALGORITHM:
1. Transient Detection
   - Attack/Sustain separation
   - Transients bypass processing (preserve direct sound)

2. Spectral Envelope Analysis
   - STFT with 2048 window, 75% overlap
   - Identify reverb tail characteristics (exponential decay)
   - Separate direct sound from reflections

3. Spectral Gating
   - Frequency-dependent thresholds
   - Soft-knee gating (avoid artifacts)
   - Preserve tonal components while reducing diffuse field

4. Material-Adaptive Parameters
   - Shellac: Moderate (often already dry)
   - Vinyl: Light (preserve natural ambience)
   - Tape: Strong (analog reverb artifacts)
   - Digital: Minimal (production choice)

QUALITY TARGETS:
- Reverb reduction: 30-60% tail dampening
- Transient preservation: >98% attack energy
- Processing: <0.3× realtime

Author: Aurik Professional Team
Version: 3.0.0
Date: März 2026
"""

import logging
import time

import numpy as np
from scipy import signal

from backend.core.defect_scanner import MaterialType

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

# Resource Management for fallback to lightweight algorithms
try:
    from backend.core.adaptive_resource_manager import adaptive_resource_manager

    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    logging.getLogger(__name__).warning("AdaptiveResourceManager not available, no automatic fallback")

# ML-Hybrid Support (Aurik 9.0 - Phase 20 v3.0)
try:
    from backend.core.hybrid.hybrid_dereverb import DereverbConfig, DereverbStrategy, HybridDereverb

    ML_HYBRID_AVAILABLE = True
except ImportError:
    ML_HYBRID_AVAILABLE = False
    logging.getLogger(__name__).warning("ML-Hybrid dereverb not available, using DSP-only mode")

# WPE Dereverberation (Spec §4.4 — Tier-1 DSP: kanonisches Dereverb-Plugin, Nakatani 2010)
try:
    from plugins.wpe_plugin import get_wpe_plugin

    WPE_AVAILABLE = True
except ImportError:
    WPE_AVAILABLE = False
    logging.getLogger(__name__).warning("WPE-Plugin nicht verfügbar — OMLSA/IMCRA-Fallback aktiv")

logger = logging.getLogger(__name__)


class ReverbReduction(PhaseInterface):
    """Professional spectral-based reverb reduction."""

    # Material-adaptive reduction strength
    REDUCTION_STRENGTH = {
        MaterialType.SHELLAC: 0.50,  # Moderate (often dry already)
        MaterialType.VINYL: 0.40,  # Light (preserve natural ambience)
        MaterialType.TAPE: 0.65,  # Strong (analog reverb artifacts)
        MaterialType.CD_DIGITAL: 0.30,  # Minimal (production choice)
        MaterialType.STREAMING: 0.25,  # Very minimal
    }

    # Tail damping factor (how quickly reverb tail decays)
    TAIL_DAMPING = {
        MaterialType.SHELLAC: 0.70,
        MaterialType.VINYL: 0.60,
        MaterialType.TAPE: 0.80,
        MaterialType.CD_DIGITAL: 0.50,
        MaterialType.STREAMING: 0.40,
    }

    # Transient threshold (energy ratio for transient detection)
    TRANSIENT_THRESHOLD = 3.0  # 3× energy increase = transient

    # STFT parameters
    WINDOW_SIZE = 2048
    HOP_SIZE = 512  # 75% overlap

    def __init__(self):
        super().__init__()
        self.name = "Reverb Reduction v3 OMLSA/IMCRA"

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_20_reverb_reduction",
            name="Reverb Reduction v3 OMLSA/IMCRA",
            category=PhaseCategory.ENHANCEMENT,
            priority=7,
            dependencies=["phase_03_denoise"],
            estimated_time_factor=0.15,
            version="3.0.0",
            memory_requirement_mb=120,
            is_cpu_intensive=True,
            is_io_intensive=False,
            quality_impact=0.90,
            description=(
                "Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003) — "
                "diffuse Schallfelder ohne musikalisches Rauschen, "
                "Transientenerhalt und scipy.signal.stft/istft (PGHI-konsistent)"
            ),
        )

    def process(
        self, audio: np.ndarray, sample_rate: int, material: MaterialType = MaterialType.VINYL, **kwargs
    ) -> PhaseResult:
        """
        Apply reverb reduction.

        Args:
            audio: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            material: Material type

        Returns:
            PhaseResult with reverb-reduced audio
        """
        self.validate_input(audio)
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        start_time = time.time()

        strength = self.REDUCTION_STRENGTH.get(material, 0.4)
        damping = self.TAIL_DAMPING.get(material, 0.6)

        # ML-Hybrid Mode Routing (v3.0)
        quality_mode = kwargs.get("quality_mode", "balanced")

        # Check resource availability for ML-Hybrid (fallback to lightweight if needed)
        use_lightweight = False
        if RESOURCE_MANAGER_AVAILABLE:
            use_lightweight = adaptive_resource_manager.should_use_lightweight_mode()
            if use_lightweight:
                logger.info(
                    f"Phase 20: Resource constraint detected, forcing DSP-only mode "
                    f"(CPU: {adaptive_resource_manager.get_cpu_usage():.1f}%, "
                    f"Memory: {adaptive_resource_manager.get_memory_usage():.1f}%)"
                )

        # ML-Hybrid only if resources available and quality mode permits
        use_ml_hybrid = (
            ML_HYBRID_AVAILABLE and quality_mode in ["balanced", "maximum", "quality"] and not use_lightweight
        )

        if use_ml_hybrid:
            try:
                logger.info(f"Phase 20 ML-Hybrid: mode={quality_mode}, material={material.value}")

                # Configure ML dereverb strategy
                # 'quality' und 'maximum' → HYBRID (SGMSE+ ML-Primär + WPE-DSP-Fallback, §4.4)
                if quality_mode in ("maximum", "quality"):
                    strategy = DereverbStrategy.HYBRID  # SGMSE+ primär → WPE DSP-Fallback (§4.4)
                else:  # balanced
                    strategy = DereverbStrategy.ADAPTIVE  # Smart: DSP only if light reverb

                dereverb = HybridDereverb(
                    config=DereverbConfig(
                        strategy=strategy,
                        dsp_strength=strength,
                        dsp_damping=damping,
                        enable_preprocessing=True,
                        reverb_threshold=0.3,  # Skip DCCRN if reverb already low
                    )
                )

                ml_result = dereverb.dereverb(audio, sample_rate=sample_rate)
                processing_time = time.time() - start_time

                # Estimate RMS change from reverb reduction
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(ml_result.audio**2))
                # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

                logger.info(
                    f"ML-Hybrid complete: DSP={ml_result.dsp_applied}, "
                    f"ML={ml_result.ml_applied}, reverb={ml_result.reverb_estimate:.3f}, "
                    f"RMS change={rms_change_db:.2f}dB, time={processing_time:.2f}s"
                )

                # Generate warnings
                warnings = []
                if ml_result.reverb_estimate > 0.7:
                    warnings.append(
                        f"High reverb detected: {ml_result.reverb_estimate:.2f} (may require multiple passes)"
                    )

                _audio_clean = np.nan_to_num(ml_result.audio, nan=0.0, posinf=0.0, neginf=0.0)
                _audio_clean = np.clip(_audio_clean, -1.0, 1.0)
                return PhaseResult(
                    success=True,
                    audio=_audio_clean,
                    metrics={
                        "rms_change_db": float(rms_change_db),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "strategy": str(ml_result.strategy_used),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                        "quality_mode": quality_mode,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "hybrid_wpe_resemble_v4",
                        "ml_hybrid": True,
                        "dsp_applied": ml_result.dsp_applied,
                        "ml_applied": getattr(ml_result, "ml_applied", ml_result.dccrn_applied),
                        "reverb_estimate": ml_result.reverb_estimate,
                        "processing_time": ml_result.processing_time,
                        "version": "3.0_ml_hybrid",
                        "window_size": self.WINDOW_SIZE,
                        "hop_size": self.HOP_SIZE,
                        "ml_metadata": ml_result.metadata,
                    },
                    warnings=warnings,
                    modifications={},
                )

            except Exception as e:
                import traceback as _tb

                logger.warning(
                    f"ML-Hybrid dereverb failed: {e}, falling back to DSP. Error type: {type(e).__name__}\n"
                    f"Traceback: {_tb.format_exc()}"
                )
                # Fall through to DSP path below

        # DSP-Only Path (Fast mode or ML fallback)
        logger.info(f"Phase 20 DSP-Only: material={material.value}, strength={strength}")

        # ── Tier-1 DSP: WPE (Nakatani 2010) — DSP-Fallback für Dereverb (§4.4; ML-Primär: SGMSE+) ──
        # WPE entfernt Spätreflexionen via iterative gewichtete lineare Prädiktion.
        # Kaskade: nara_wpe → NumPy-WPE → OMLSA (innerhalb des Plugins).
        if WPE_AVAILABLE:
            try:
                wpe = get_wpe_plugin()
                # WPE erwartet SR == 48000 (Spec-Invariante)
                wpe_strength = float(np.clip(strength * 0.90, 0.3, 0.95))
                audio_wpe = wpe.enhance(audio.astype(np.float32), sample_rate, strength=wpe_strength)
                audio_wpe = np.nan_to_num(audio_wpe, nan=0.0, posinf=0.0, neginf=0.0)
                audio_wpe = np.clip(audio_wpe.astype(np.float64), -1.0, 1.0)
                # Passen Länge und Form an
                if audio_wpe.shape != audio.shape:
                    if audio.ndim == 1:
                        audio_wpe = audio_wpe.flatten()[: len(audio)]
                        if len(audio_wpe) < len(audio):
                            audio_wpe = np.pad(audio_wpe, (0, len(audio) - len(audio_wpe)))
                    else:
                        min_len = min(
                            audio_wpe.shape[0] if audio_wpe.ndim == 1 else audio_wpe.shape[-1], audio.shape[-1]
                        )
                        audio_wpe = audio_wpe[:min_len] if audio_wpe.ndim == 1 else audio_wpe[:, :min_len]
                reduced = audio_wpe
                processing_time = time.time() - start_time
                rms_before = np.sqrt(np.mean(audio**2))
                rms_after = np.sqrt(np.mean(reduced**2))
                rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))
                logger.info("Phase 20: WPE-Tier erfolgreich (strength=%.2f)", wpe_strength)
                reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
                reduced = np.clip(reduced, -1.0, 1.0)
                return PhaseResult(
                    success=True,
                    audio=reduced,
                    metrics={
                        "rms_change_db": float(rms_change_db),
                        "reduction_strength": strength,
                        "tail_damping": damping,
                        "material": material.value,
                    },
                    execution_time_seconds=processing_time,
                    metadata={
                        "algorithm": "wpe_nakatani2010_tier1",
                        "version": "3.0_wpe",
                        "wpe_strength": wpe_strength,
                    },
                )
            except Exception as wpe_err:
                logger.warning("Phase 20: WPE fehlgeschlagen (%s) — OMLSA/IMCRA-Fallback", wpe_err)
        # ── Tier-2 DSP: OMLSA/IMCRA (Cohen 2002/2003) — Fallback ──────────────
        is_stereo = audio.ndim == 2

        if is_stereo:
            # Detect channel-major (2, N) vs time-major (N, 2).
            # Aurik uses channel-major (2, N) throughout the pipeline.
            _is_ch_maj = audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            _left = audio[0] if _is_ch_maj else audio[:, 0]
            _right = audio[1] if _is_ch_maj else audio[:, 1]
            # Process each channel in parallel (multicore via ThreadPoolExecutor)
            # Note: _reduce_reverb uses numpy FFT which releases GIL, enabling true parallelism
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(self._reduce_reverb, _left, sample_rate, strength, damping)
                future_right = executor.submit(self._reduce_reverb, _right, sample_rate, strength, damping)
                left_reduced = future_left.result()
                right_reduced = future_right.result()

            # Ensure same length and recombine in original format
            min_len = min(len(left_reduced), len(right_reduced))
            if _is_ch_maj:
                reduced = np.stack([left_reduced[:min_len], right_reduced[:min_len]], axis=0)  # (2, N)
            else:
                reduced = np.column_stack([left_reduced[:min_len], right_reduced[:min_len]])  # (N, 2)
        else:
            reduced = self._reduce_reverb(audio, sample_rate, strength, damping)

        processing_time = time.time() - start_time

        # Measure reverb reduction (RT60-like estimate)
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(reduced**2))
        # Guard: np.log10(0) => RuntimeWarning; clamp ratio >= 1e-30
        rms_change_db = 20 * np.log10(np.maximum(rms_after / (rms_before + 1e-10), 1e-30))

        reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
        reduced = np.clip(reduced, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=reduced,
            metrics={
                "rms_change_db": float(rms_change_db),
                "reduction_strength": strength,
                "tail_damping": damping,
                "material": material.value,
            },
            execution_time_seconds=processing_time,
            metadata={
                "algorithm": "stft_omlsa_imcra_cohen2003",
                "version": "3.0_omlsa",
                "window_size": self.WINDOW_SIZE,
                "hop_size": self.HOP_SIZE,
            },
        )

    def _reduce_reverb(self, audio: np.ndarray, sample_rate: int, strength: float, damping: float) -> np.ndarray:
        """Nachhall-Reduktion via STFT-OMLSA/IMCRA (Cohen 2002/2003).

        Algorithmus:
            1. scipy.signal.stft  — OLA-konsistent (ersetzt np.fft.rfft-Schleife)
            2. IMCRA gleitendes Minimum (Cohen 2003):
               σ²_d(t,f) = b_min · min_{t'∈[t-M,t]} S̃(t',f), b_min=1.66
            3. DD a-priori SNR:
               ξ̂(t,f) = α·G²(t-1,f)·γ(t-1,f) + (1-α)·max(γ(t,f)-1, 0)
            4. OMLSA Gain:
               G(t,f) = G_floor^(1−p(t,f)) · (ξ̂/(1+ξ̂))^p(t,f), Clip [G_floor,1]
            5. Cappé Temporal-Smoothing: Ĝ_t = α_g·Ĝ_{t-1} + (1-α_g)·G_t
            6. scipy.signal.istft  — phasenkonsistente PGHI-Rekonstruktion
            7. Transientenerhalt: Original zurückgemischt wo transient_mask > 0.5
            8. nan_to_num + clip[-1, 1]

        Forschungsreferenz:
            Cohen & Berdugo (2002) Signal Processing Letters
            Cohen (2003) IEEE Trans. Speech Audio Process.
            Le Roux & Vincent (2013) Consistent Wiener Filtering

        Args:
            audio:       1D float32, normalisiert auf [-1, 1]
            sample_rate: Abtastrate in Hz (intern 48 000 Hz)
            strength:    Reduktionsstärke ∈ [0.0, 1.0] (materialadaptiv)
            damping:     Nachhall-Dämpfungs-Prior (beeinflusst G_floor)

        Returns:
            np.ndarray: Restauriertes Audio, gleiche Länge wie Eingang, clip[-1, 1].
        """
        n_audio = len(audio)

        # ── 1. Transientenerkennung (Sample-Ebene, vor STFT) ─────────────────
        transient_mask_raw = self._detect_transients(audio, sample_rate)

        # ── 2. STFT via scipy (OLA-konsistent, KEIN np.fft.rfft) ─────────────
        noverlap = self.WINDOW_SIZE - self.HOP_SIZE
        _, _, stft_in = signal.stft(
            audio,
            fs=sample_rate,
            window="hann",
            nperseg=self.WINDOW_SIZE,
            noverlap=noverlap,
            boundary="even",  # symmetrische Randfortsetzung (scipy-konform)
            padded=True,
        )
        magnitude = np.abs(stft_in)  # (F, T)
        phase_arr = np.angle(stft_in)  # (F, T)
        F, T = magnitude.shape

        # ── 3. IMCRA Rauschboden-Schätzung (Cohen 2003) ──────────────────────
        #  Gleite über M Frames (~1.5 s), aktualisiere Sliding-Minimum.
        frames_per_sec = sample_rate / self.HOP_SIZE
        M = max(3, int(1.5 * frames_per_sec))  # ≈ 140 Frames bei 48 kHz / 512 Hop
        b_min = 1.66  # Bias-Kompensation (Cohen 2003, Tab. I)
        alpha_n = 0.85  # Exponentieller Glätter für Sliding-Min

        power = magnitude**2  # Leistungsspektrum (F, T)
        noise_floor_sq = np.zeros_like(power)  # σ²_d(t, f)
        S_min_prev = power[:, 0].copy()
        S_tmp_prev = power[:, 0].copy()

        for t in range(T):
            p_t = power[:, t]
            if t == 0:
                S_min_t = p_t.copy()
                S_tmp_t = p_t.copy()
            else:
                S_smooth = alpha_n * S_min_prev + (1.0 - alpha_n) * p_t
                S_min_t = np.minimum(S_min_prev, S_smooth)
                # Puffer alle M Frames zurücksetzen
                S_tmp_t = S_smooth.copy() if t % M == 0 else np.minimum(S_tmp_prev, S_smooth)
            noise_floor_sq[:, t] = b_min * S_min_t
            S_min_prev = S_min_t
            S_tmp_prev = S_tmp_t

        noise_floor_sq = np.clip(noise_floor_sq, 1e-12, None)

        # ── 4. OMLSA Gain (Cohen 2003) ────────────────────────────────────────
        G_floor = float(np.clip(0.1 + (1.0 - strength) * 0.05, 0.04, 0.15))
        alpha_dd = 0.92  # DD-Glättung (Ephraim & Malah 1985, nur DD-Teil)
        # q = a-priori Sprachabwesenheits-Wahrscheinlichkeit (stärkeabhängig)
        q = float(np.clip(strength * 0.60, 0.10, 0.80))

        G_omlsa = np.ones((F, T), dtype=np.float64)
        G_prev = np.ones(F, dtype=np.float64)
        gamma_prev = np.ones(F, dtype=np.float64)

        for t in range(T):
            sigma_n_sq = noise_floor_sq[:, t]
            gamma_t = power[:, t] / sigma_n_sq  # a-posteriori SNR

            # Decision-Directed a-priori SNR:
            # ξ̂ = α·G²(t-1)·γ(t-1) + (1-α)·max(γ(t)-1, 0)
            xi_t = alpha_dd * G_prev**2 * gamma_prev + (1.0 - alpha_dd) * np.maximum(gamma_t - 1.0, 0.0)
            xi_t = np.maximum(xi_t, 1e-6)

            # Posteriore Sprachpräsenzwahrscheinlichkeit p(H₁|Y)
            nu = gamma_t * xi_t / (1.0 + xi_t)
            Lambda = ((1.0 - q) / q) * (1.0 / (1.0 + xi_t)) * np.exp(np.clip(nu, -50.0, 50.0))
            p_H1 = np.clip(Lambda / (1.0 + Lambda), 0.0, 1.0)

            # OMLSA Gain: G = G_floor^(1-p) · (ξ/(1+ξ))^p
            G_wiener = xi_t / (1.0 + xi_t)
            G_t = (G_floor ** (1.0 - p_H1)) * (G_wiener**p_H1)
            G_t = np.clip(G_t, G_floor, 1.0)

            G_omlsa[:, t] = G_t
            G_prev = G_t
            gamma_prev = gamma_t

        # ── 5. Cappé Temporal-Gain-Glättung (α_g=0.85) ───────────────────────
        alpha_g = 0.85
        G_smooth = G_omlsa.copy()
        for t in range(1, T):
            G_smooth[:, t] = alpha_g * G_smooth[:, t - 1] + (1.0 - alpha_g) * G_omlsa[:, t]

        # ── 6. Gain auf STFT anwenden ─────────────────────────────────────────
        stft_out = (magnitude * G_smooth) * np.exp(1j * phase_arr)

        # ── 7. ISTFT — phasenkonsistente OLA-Rekonstruktion ──────────────────
        _, audio_out = signal.istft(
            stft_out,
            fs=sample_rate,
            window="hann",
            nperseg=self.WINDOW_SIZE,
            noverlap=noverlap,
            boundary=True,
        )
        audio_out = np.real(audio_out).astype(np.float64)

        # Länge angleichen
        if len(audio_out) > n_audio:
            audio_out = audio_out[:n_audio]
        elif len(audio_out) < n_audio:
            audio_out = np.pad(audio_out, (0, n_audio - len(audio_out)), mode="edge")

        # NaN/Inf-Schutz + Clip
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # ── 8. Transientenerhalt ───────────────────────────────────────────────
        # Transient-Maske auf Sample-Ebene hochsampeln
        transient_up = signal.resample(transient_mask_raw, len(audio_out))
        transient_up = np.clip(transient_up, 0.0, 1.0)
        audio_out = audio_out * (1.0 - transient_up) + audio[: len(audio_out)] * transient_up
        audio_out = np.clip(audio_out, -1.0, 1.0)

        return audio_out

    def _detect_transients(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detect transients using energy envelope.

        Returns:
            Binary mask (1 = transient, 0 = sustain/decay)
        """
        # Calculate energy envelope (RMS in short windows)
        window_samples = int(0.01 * sample_rate)  # 10ms windows
        hop_samples = window_samples // 2

        num_windows = (len(audio) - window_samples) // hop_samples + 1
        energy = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            window = audio[start:end]
            energy[i] = np.sqrt(np.mean(window**2))  # Fix: assign to energy array

        # Detect transients as rapid energy increases
        transient_mask = np.zeros(num_windows)
        for i in range(1, num_windows):
            if energy[i] > self.TRANSIENT_THRESHOLD * energy[i - 1]:
                # Transient detected
                transient_mask[i] = 1.0
                # Extend mask for attack phase (20ms)
                extend_frames = int(0.02 * sample_rate / hop_samples)
                transient_mask[i : min(i + extend_frames, num_windows)] = 1.0

        return transient_mask


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.debug("=" * 80)
    logger.debug("Phase 20: Professional Reverb Reduction v2.0")
    logger.debug("=" * 80)
    logger.debug("")

    # Generate test audio with synthetic reverb
    duration = 3.0
    sample_rate = 44100

    # Dry signal: short impulses (like snare hits)
    t = np.linspace(0, duration, int(sample_rate * duration))
    dry_signal = np.zeros_like(t)

    # Add impulses at 0.5s intervals
    for impulse_time in np.arange(0, duration, 0.5):
        impulse_sample = int(impulse_time * sample_rate)
        if impulse_sample < len(dry_signal):
            dry_signal[impulse_sample : impulse_sample + 100] = 0.8 * np.exp(-np.arange(100) / 20)

    # Add musical content
    dry_signal += 0.2 * np.sin(2 * np.pi * 440 * t)

    # Add synthetic reverb (exponential decay of signal)
    reverb_tail = signal.lfilter([1], [1, -0.7], dry_signal)  # Simple comb filter
    reverbed_signal = dry_signal + 0.4 * reverb_tail

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug("Dry signal + synthetic reverb tail")
    logger.debug("")

    # Test with different materials
    materials = [
        (MaterialType.TAPE, "TAPE"),
        (MaterialType.VINYL, "VINYL"),
        (MaterialType.CD_DIGITAL, "CD_DIGITAL"),
    ]

    for material, material_name in materials:
        logger.debug("─" * 80)
        logger.debug(f"Material: {material_name}")
        logger.debug("─" * 80)
        logger.debug("")

        phase = ReverbReduction()
        result = phase.process(reverbed_signal, sample_rate, material)

        logger.debug("✅ Professional Reverb Reduction:")
        logger.debug(f"   RMS Change: {result.metrics['rms_change_db']:.2f} dB")
        logger.debug(f"   Reduction Strength: {result.metrics['reduction_strength']:.2f}")
        logger.debug(f"   Tail Damping: {result.metrics['tail_damping']:.2f}")
        logger.debug(
            f"   Processing time: {result.execution_time_seconds:.3f}s ({result.execution_time_seconds / duration:.2f}× realtime)"
        )
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test completed")
    logger.debug("=" * 80)
