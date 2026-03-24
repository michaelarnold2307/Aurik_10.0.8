"""
Hybrid Dereverb - AURIK 9.0 Phase 20 ML-Hybrid
===============================================

Two-stage dereverb: DSP spectral gating + ML refinement (SGMSE+ / ResembleEnhance).

Architecture:
1. Stage 1: DSP Spectral Gating (fast, ~0.3× RT)
   - Transient detection
   - Frequency-dependent gating
   - Tail damping

2. Stage 2: ML Refinement (slower, ~2.0× RT)
   - SGMSE+ (primary) or ResembleEnhance (fallback)
   - Preserves direct sound, removes reflections

Strategy Modes:
- DSP_ONLY: Fast spectral gating only
- ML_ONLY: Pure ML (no DSP pre-processing)
- HYBRID: DSP → ML pipeline
- ADAPTIVE: Choose based on reverb severity

Author: Aurik 9.0 Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

from dataclasses import dataclass
from enum import Enum
import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# §3.x Singleton — Thread-sicher, Double-Checked Locking
_instance: Optional["HybridDereverb"] = None
_lock = threading.Lock()


def get_hybrid_dereverb() -> "HybridDereverb":
    """Thread-sicheres Singleton für HybridDereverb (§3.x Double-Checked Locking).

    Returns:
        HybridDereverb singleton instance.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HybridDereverb()
    return _instance


class DereverbStrategy(Enum):
    """Dereverb strategy selection."""

    DSP_ONLY = "dsp_only"  # Fast spectral gating
    DCCRN_ONLY = "dccrn_only"  # Pure ML
    HYBRID = "hybrid"  # DSP → DCCRN
    ADAPTIVE = "adaptive"  # Auto-select based on reverb level


@dataclass
class DereverbConfig:
    """Configuration for hybrid dereverb."""

    strategy: DereverbStrategy = DereverbStrategy.ADAPTIVE
    dsp_strength: float = 0.5  # DSP reduction strength (0-1)
    dsp_damping: float = 0.6  # DSP tail damping (0-1)
    dccrn_model: str = "dccrn.onnx"  # DCCRN model name
    enable_preprocessing: bool = True  # Enable DSP pre-processing
    reverb_threshold: float = 0.3  # Threshold for adaptive mode (0-1)


@dataclass
class DereverbResult:
    """Result from hybrid dereverb."""

    audio: np.ndarray
    strategy_used: DereverbStrategy
    dsp_applied: bool
    dccrn_applied: bool  # backward-compat alias (True == ml_applied)
    processing_time: float
    reverb_estimate: float  # Estimated reverb level (0-1)
    metadata: dict[str, Any]
    ml_applied: bool = False  # True wenn ResembleEnhance aktiv war


class HybridDereverb:
    """
    Hybrid Dereverb: DSP + SGMSE+ ML (Primär, §4.4) + ResembleEnhance (Fallback 1).

    SOTA-Priorität gemäß §4.4 DSP-Mindeststandards (Vocal Enhancement / Dereverb):
        Primär:    SGMSE+ ONNX         (sgmse_plugin,          ~120 MB)
        Fallback 1: Resemble-Enhance ONNX (resemble_enhance_plugin, ~722 MB)
        Fallback 2: WPE DSP             (kein ML erforderlich)

    Combines fast DSP spectral gating with SGMSE+ ML refinement.
    SGMSE+ (Score-Based Generative Model for Speech Enhancement, Richter 2022)
    ist primärer Vocal/Music Dereverb+Enhancement-Algorithmus.
    Adaptive strategy selects optimal processing based on reverb severity.
    """

    def __init__(self, config: DereverbConfig | None = None) -> None:
        """
        Initialize hybrid dereverb.

        Args:
            config: Dereverb configuration
        """
        self.config = config or DereverbConfig()
        self._sgmse_active: bool = False  # True wenn SGMSE+ ONNX geladen (Primär §4.4)

        # Lazy-load ML-Stufe: SGMSE+ primär, ResembleEnhance als Fallback 1
        self.dccrn = None  # backward-compat Name beibehalten
        if self.config.strategy in [DereverbStrategy.DCCRN_ONLY, DereverbStrategy.HYBRID, DereverbStrategy.ADAPTIVE]:
            self._init_dccrn()

    def _init_dccrn(self) -> None:
        """ML-Stufe initialisieren — SGMSE+ primär (§4.4), ResembleEnhance Fallback 1, DSP Fallback 2.

        SOTA-Reihenfolge (§4.4 Vocal Enhancement / Dereverb):
            1. SGMSE+ ONNX (sgmse_plugin)          — Primär
            2. Resemble-Enhance ONNX                — Fallback 1
            3. WPE DSP (self.dccrn = None)          — Fallback 2
        """
        # Stufe 1: SGMSE+ ONNX (§4.4 Primär — Score-Based Generative Model for Speech Enhancement)
        try:
            from plugins.sgmse_plugin import get_sgmse_plus_plugin

            self.dccrn = get_sgmse_plus_plugin()
            self._sgmse_active = True
            logger.info("✅ SGMSE+ geladen als Dereverb-Primärmodul (§4.4) — ResembleEnhance als Fallback bereit")
            return
        except ImportError as e:
            logger.debug("SGMSE+ import fehlgeschlagen (%s) — versuche ResembleEnhance Fallback 1", e)
        except Exception as e:
            logger.warning("SGMSE+ Init-Fehler (%s) — versuche ResembleEnhance Fallback 1", e)

        # Stufe 2: Resemble-Enhance ONNX (§4.4 Fallback 1)
        try:
            from plugins.resemble_enhance_plugin import ResembleEnhancePlugin

            self.dccrn = ResembleEnhancePlugin()
            self._sgmse_active = False
            logger.info("ResembleEnhance ML-Stufe für Dereverb geladen (§4.4 Fallback 1)")
        except ImportError as e:
            logger.info("ResembleEnhance nicht verfügbar (%s) — WPE-DSP-Fallback 2 aktiv", e)
            self.dccrn = None
        except Exception as e:
            logger.warning("ResembleEnhance-Init fehlgeschlagen (%s) — DSP-only Fallback 2", e)
            self.dccrn = None

    def dereverb(self, audio: np.ndarray, sample_rate: int = 48000) -> DereverbResult:
        """
        Apply hybrid dereverb.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz

        Returns:
            DereverbResult with processed audio and metadata
        """
        import time

        start_time = time.time()

        # Determine strategy
        strategy = self._determine_strategy(audio, sample_rate)

        dsp_applied = False
        dccrn_applied = False
        ml_applied = False
        reverb_estimate = self._estimate_reverb_level(audio, sample_rate)
        metadata = {}

        # Stage 1: DSP pre-processing (if enabled)
        if strategy in [DereverbStrategy.DSP_ONLY, DereverbStrategy.HYBRID]:
            logger.info("Stage 1: Applying DSP spectral gating...")
            audio, dsp_meta = self._apply_dsp_dereverb(audio, sample_rate)
            dsp_applied = True
            metadata["dsp"] = dsp_meta

            # Re-estimate reverb after DSP
            reverb_after_dsp = self._estimate_reverb_level(audio, sample_rate)
            metadata["reverb_after_dsp"] = reverb_after_dsp

            logger.info(f"DSP complete: reverb {reverb_estimate:.3f} → {reverb_after_dsp:.3f}")

            # Skip DCCRN if reverb already low enough
            if reverb_after_dsp < self.config.reverb_threshold and strategy == DereverbStrategy.HYBRID:
                logger.info(f"Reverb sufficient ({reverb_after_dsp:.3f}), skipping ML refinement")
                strategy = DereverbStrategy.DSP_ONLY

        # Stage 2: ML refinement (SGMSE+ / ResembleEnhance, if needed)
        if strategy in [DereverbStrategy.DCCRN_ONLY, DereverbStrategy.HYBRID]:
            if self.dccrn is not None:
                logger.info("Stage 2: ResembleEnhance ML-Dereverb-Stufe...")
                audio, dccrn_meta = self._apply_dccrn(audio, sample_rate)
                dccrn_applied = True
                ml_applied = True
                metadata["ml"] = dccrn_meta

                # Re-estimate reverb after DCCRN
                reverb_after_dccrn = self._estimate_reverb_level(audio, sample_rate)
                metadata["reverb_after_dccrn"] = reverb_after_dccrn

                logger.info(f"ML dereverb complete: reverb → {reverb_after_dccrn:.3f}")
            else:
                logger.info("ResembleEnhance nicht verfügbar — WPE-DSP-Ergebnis wird verwendet")

        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time

        return DereverbResult(
            audio=audio,
            strategy_used=strategy,
            dsp_applied=dsp_applied,
            dccrn_applied=dccrn_applied,
            ml_applied=ml_applied,
            processing_time=processing_time,
            reverb_estimate=reverb_estimate,
            metadata=metadata,
        )

    def _determine_strategy(self, audio: np.ndarray, sample_rate: int) -> DereverbStrategy:
        """Determine optimal dereverb strategy."""
        if self.config.strategy != DereverbStrategy.ADAPTIVE:
            return self.config.strategy

        # Analyze reverb level to decide strategy
        reverb_level = self._estimate_reverb_level(audio, sample_rate)

        if reverb_level < 0.2:
            # Light reverb - DSP sufficient
            logger.info(f"Light reverb ({reverb_level:.3f}), DSP only")
            return DereverbStrategy.DSP_ONLY
        elif reverb_level < 0.5:
            # Moderate reverb - Hybrid recommended
            logger.info(f"Moderate reverb ({reverb_level:.3f}), Hybrid mode")
            return DereverbStrategy.HYBRID
        else:
            # Heavy reverb - Full ML dereverb
            logger.info(f"Heavy reverb ({reverb_level:.3f}), ML-only")
            return DereverbStrategy.DCCRN_ONLY if self.dccrn else DereverbStrategy.HYBRID

    def _apply_dsp_dereverb(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Apply DSP spectral gating dereverb.

        Uses existing DSP logic from phase_20_reverb_reduction.py.
        """
        from backend.core.phases.phase_20_reverb_reduction import ReverbReduction

        metadata = {}

        # Create phase instance
        phase = ReverbReduction()

        # Detect stereo format: channel-major (2, N) vs time-major (N, 2).
        # Aurik uses channel-major (2, N) throughout the pipeline.
        _is_ch_maj = audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]

        # Process with DSP — extract correct mono channel
        if audio.ndim == 1:
            _left_ch = audio
        elif _is_ch_maj:
            _left_ch = audio[0]  # channel-major: first row = left channel
        else:
            _left_ch = audio[:, 0]  # time-major: first column = left channel

        result = phase._reduce_reverb(
            _left_ch,
            sample_rate,
            strength=self.config.dsp_strength,
            damping=self.config.dsp_damping,
        )

        # Handle stereo: extract right channel and recombine in original format
        if audio.ndim == 2:
            _right_ch = audio[1] if _is_ch_maj else audio[:, 1]
            result_right = phase._reduce_reverb(
                _right_ch, sample_rate, strength=self.config.dsp_strength, damping=self.config.dsp_damping
            )
            _n = min(len(result), len(result_right))
            if _is_ch_maj:
                result = np.stack([result[:_n], result_right[:_n]], axis=0)  # (2, N)
            else:
                result = np.column_stack([result[:_n], result_right[:_n]])  # (N, 2)

        metadata["strength"] = self.config.dsp_strength
        metadata["damping"] = self.config.dsp_damping

        return result, metadata

    def _apply_dccrn(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
        """ML-Dereverb via SGMSE+ (primär §4.4) oder ResembleEnhance (Fallback 1)."""
        metadata: dict[str, Any] = {}
        try:
            audio_in = audio.astype(np.float32)
            # Detect channel-major (2, N) vs time-major (N, 2) format.
            # For channel-major: mean along axis=0 collapses channels → (N,).
            # For time-major:  mean along axis=1 (=axis=-1) collapses channels → (N,).
            _dccrn_ch_maj = audio_in.ndim == 2 and audio_in.shape[0] <= 2 and audio_in.shape[1] > audio_in.shape[0]
            if audio_in.ndim == 2:
                mono_in = audio_in.mean(axis=0) if _dccrn_ch_maj else audio_in.mean(axis=1)
            else:
                mono_in = audio_in

            if self._sgmse_active:
                # §4.4 Primär: SGMSE+ — enhance(audio, sr) → SGMSEResult
                result = self.dccrn.enhance(mono_in, sample_rate)
                enhanced = np.asarray(
                    result.audio if hasattr(result, "audio") else result,
                    dtype=np.float32,
                )
                metadata["model"] = "sgmse_plus"
                metadata["model_used"] = getattr(result, "model_used", "sgmse_plus_torchscript")
            else:
                # §4.4 Fallback 1: ResembleEnhance
                enhanced = self.dccrn.enhance(mono_in, sample_rate)
                enhanced = np.asarray(enhanced, dtype=np.float32)
                metadata["model"] = "resemble_enhance"

            # Stereo wiederherstellen — in der gleichen Format-Konvention wie der Eingang.
            if audio.ndim == 2:
                if _dccrn_ch_maj:
                    enhanced = np.stack([enhanced[: audio.shape[1]], enhanced[: audio.shape[1]]], axis=0)  # (2, N)
                else:
                    enhanced = np.stack([enhanced, enhanced], axis=-1)  # (N, 2)
            elif audio.ndim == 1 and enhanced.ndim > 1:
                enhanced = np.mean(enhanced, axis=-1)

            metadata["success"] = True
            return np.clip(np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), metadata

        except Exception as e:
            logger.warning("HybridDereverb ML-Stufe fehlgeschlagen (%s) — DSP-Ergebnis unverändert", e)
            metadata["success"] = False
            metadata["error"] = str(e)
            return audio, metadata

    def _estimate_reverb_level(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Estimate reverb level in audio (0-1 scale).

        Uses RT60-like analysis: measure decay time of energy envelope.
        """
        from scipy import signal

        # Convert to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Calculate energy envelope (RMS in sliding windows)
        window_samples = int(0.05 * sample_rate)  # 50ms windows
        hop_samples = window_samples // 2

        num_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)
        energy = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * hop_samples
            end = min(start + window_samples, len(audio))
            if end > start:
                window = audio[start:end]
                energy[i] = np.sqrt(np.mean(window**2))

        # Smooth energy envelope
        energy_smooth = signal.medfilt(energy, kernel_size=min(5, len(energy))) if len(energy) > 0 else energy

        # Estimate decay time (RT60-like)
        # Find peak
        if len(energy_smooth) == 0 or np.max(energy_smooth) == 0:
            return 0.0

        # STATIONARITY CHECK: Tape hiss / background noise has nearly flat energy
        # → relative std < 20% → classify as no reverb to avoid false-positive DCCRN.
        # Real room reverb has exponential decay → much higher relative variance.
        if len(energy_smooth) > 5:
            rel_std = float(np.std(energy_smooth) / (np.mean(energy_smooth) + 1e-12))
            if rel_std < 0.20:
                return 0.0  # Stationary noise, not reverb

        peak_idx = np.argmax(energy_smooth)

        # Measure decay after peak
        if peak_idx >= len(energy_smooth) - 1:
            return 0.0

        decay_envelope = energy_smooth[peak_idx:]
        peak_energy = decay_envelope[0]

        if peak_energy == 0:
            return 0.0

        # Find time to decay to noise floor (RT60-like, noise-floor-relative).
        # Using noise_floor prevents false-positives: -60 dB re peak is unreachable
        # in any real recording with floor noise, causing decay_time = len(array).
        noise_floor = float(np.percentile(energy_smooth, 15))
        signal_above_floor = peak_energy - noise_floor
        if signal_above_floor < noise_floor * 2.0:  # Peak barely above noise floor
            return 0.0
        # Target: signal decays to noise floor + 0.1% of signal range (~-60 dB re signal)
        target_energy = noise_floor + signal_above_floor * 0.001

        decay_time = 0
        for i, e in enumerate(decay_envelope):
            if e < target_energy:
                decay_time = i
                break
        else:
            # Never reached target — estimate from first/last quarter trend
            if len(decay_envelope) > 4:
                first_q = float(np.mean(decay_envelope[: len(decay_envelope) // 4]))
                last_q = float(np.mean(decay_envelope[3 * len(decay_envelope) // 4 :]))
                if last_q < first_q:  # Some decay present → estimate conservatively
                    decay_time = len(decay_envelope) // 3
                else:  # Truly flat after peak → no reverb
                    decay_time = 0
            else:
                decay_time = 0

        # Normalize to 0-1 scale
        # Short decay (< 0.1s) = low reverb
        # Long decay (> 1.0s) = high reverb
        decay_seconds = decay_time * hop_samples / sample_rate
        reverb_level = np.clip(decay_seconds / 1.0, 0, 1)

        return float(reverb_level)


if __name__ == "__main__":
    """Test hybrid dereverb."""

    logger.debug("=" * 80)
    logger.debug("Hybrid Dereverb Test")
    logger.debug("=" * 80)

    # Generate test audio with synthetic reverb
    duration = 5.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Dry signal: impulses + sine
    dry = np.zeros_like(t)
    for impulse_time in np.arange(0, duration, 0.5):
        idx = int(impulse_time * sample_rate)
        if idx < len(dry):
            dry[idx : idx + 100] = 0.8 * np.exp(-np.arange(100) / 20)
    dry += 0.2 * np.sin(2 * np.pi * 440 * t)

    # Add reverb (simple comb filter)
    from scipy import signal as sp_signal

    reverb_tail = sp_signal.lfilter([1], [1, -0.7], dry)
    reverbed = dry + 0.5 * reverb_tail

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug("")

    # Test strategies
    strategies = [
        (DereverbStrategy.DSP_ONLY, "DSP Only"),
        (DereverbStrategy.HYBRID, "Hybrid (DSP + DCCRN)"),
    ]

    for strategy, name in strategies:
        logger.debug("-" * 80)
        logger.debug(f"Strategy: {name}")
        logger.debug("-" * 80)

        config = DereverbConfig(strategy=strategy)
        dereverb = HybridDereverb(config)

        result = dereverb.dereverb(reverbed, sample_rate)

        logger.debug(f"✅ Strategy used: {result.strategy_used.value}")
        logger.debug(f"   DSP applied: {result.dsp_applied}")
        logger.debug(f"   DCCRN applied: {result.dccrn_applied}")
        logger.debug(f"   Reverb estimate: {result.reverb_estimate:.3f}")
        logger.debug(f"   Processing time: {result.processing_time:.2f}s")
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test complete")
