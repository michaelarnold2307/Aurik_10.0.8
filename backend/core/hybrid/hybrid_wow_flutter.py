"""
Hybrid Wow/Flutter Correction - AURIK 9.0 Phase 12 ML-Hybrid
=============================================================

Zwei-Stufen-Pitch-Detektion: pYIN (DSP) + CREPE (ML) für überlegene Genauigkeit.
§4.2-konform: klassisches YIN (de Cheveigné 2002) ist VERBOTEN als primäre Methode.

Architektur:
1. Stufe 1: pYIN Pitch-Detektion (Mauch & Dixon 2014)
   - HMM-basierte Voiced/Unvoiced-Klassifikation
   - Kumulative mittlere normalisierte Differenz (probabilistisch)
   - Robust bei verrauschten/historischen Signalen

2. Stufe 2: CREPE ML Pitch-Detektion (Kim et al. 2018)
   - CNN-basiertes Pitch-Tracking
   - ±1 Cent Genauigkeit
   - Verhindert Oktavfehler
   - Besser bei komplexen/harmonisch dichten Signalen

Strategy-Modi:
- PYIN_ONLY: Schnelle pYIN-DSP-Detektion
- CREPE_ONLY: Reines ML (kein DSP-Preprocessing)
- HYBRID: pYIN → CREPE-Verfeinerung für unsichere Regionen
- ADAPTIVE: Auswahl nach Konfidenz-Scores

Korrektur-Pipeline:
1. Pitch-Detektion (pYIN oder CREPE)
2. Wow/Flutter-Trennung (< 4 Hz vs. 4-100 Hz)
3. Phase-Vocoder Zeitstreckung (Korrektur)

Author: Aurik 9.0 Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

import csv
from dataclasses import dataclass
from enum import Enum
import logging
import os
import tempfile
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PitchDetectionStrategy(Enum):
    """Pitch detection strategy selection."""

    PYIN_ONLY = "pyin_only"  # pYIN DSP (Mauch & Dixon 2014) — §4.2 konform
    CREPE_ONLY = "crepe_only"  # Pure ML CNN
    HYBRID = "hybrid"  # pYIN → CREPE-Verfeinerung
    ADAPTIVE = "adaptive"  # Auto-Auswahl nach Konfidenz
    # Backward-Alias (deprecated)
    YIN_ONLY = "pyin_only"  # Alias → PYIN_ONLY


@dataclass
class WowFlutterConfig:
    """Configuration for hybrid wow/flutter correction."""

    strategy: PitchDetectionStrategy = PitchDetectionStrategy.ADAPTIVE
    pyin_confidence_threshold: float = 0.4  # pYIN Mindest-Konfidenz
    crepe_model: str = "full"  # CREPE model size
    confidence_threshold: float = 0.7  # Mindest-Konfidenz für Pitch-Schätzwert
    enable_preprocessing: bool = True  # pYIN-Preprocessing aktivieren
    # Backward-Alias
    yin_threshold: float = 0.15  # unused — nur für Kompatibilität


@dataclass
class WowFlutterResult:
    """Result from hybrid wow/flutter correction."""

    pitch_trajectory: np.ndarray  # Pitch-Schätzwerte (Hz)
    confidence: np.ndarray  # Konfidenz-Scores (0-1)
    strategy_used: PitchDetectionStrategy
    pyin_applied: bool  # pYIN (Mauch & Dixon 2014) angewendet
    crepe_applied: bool
    processing_time: float
    mean_confidence: float
    metadata: dict[str, Any]

    @property
    def yin_applied(self) -> bool:
        """Backward-Alias für pyin_applied."""
        return self.pyin_applied


class HybridWowFlutter:
    """
    Hybrid Wow/Flutter Detection: YIN + CREPE.

    Combines fast YIN DSP detection with high-accuracy CREPE ML refinement.
    Adaptive strategy selects optimal detection based on signal characteristics.
    """

    def __init__(self, config: WowFlutterConfig | None = None) -> None:
        """
        Initialize hybrid wow/flutter detector.

        Args:
            config: Wow/Flutter configuration
        """
        self.config = config or WowFlutterConfig()

        # Lazy-load CREPE plugin
        self.crepe = None
        if self.config.strategy in [
            PitchDetectionStrategy.CREPE_ONLY,
            PitchDetectionStrategy.HYBRID,
            PitchDetectionStrategy.ADAPTIVE,
        ]:
            self._init_crepe()

    def _init_crepe(self) -> None:
        """Initialize FCPE/CREPE pitch plugin (lazy-loading, FCPE preferred)."""
        try:
            from plugins.fcpe_plugin import get_fcpe_plugin  # noqa: PLC0415
            self.crepe = get_fcpe_plugin()
            logger.info("FCPE pitch plugin loaded for wow/flutter detection (model=%s)", self.crepe.model_used)
            return
        except Exception as e:  # noqa: BLE001
            logger.debug("FCPE-Plugin nicht verfügbar (%s) — CREPE-Fallback", e)
        try:
            from plugins.crepe_plugin import get_crepe_plugin  # noqa: PLC0415
            self.crepe = get_crepe_plugin()
            logger.info("CREPE plugin geladen für wow/flutter-Detektion")
        except Exception as e:  # noqa: BLE001
            logger.warning("Kein Pitch-ML-Plugin verfügbar (%s) — pYIN-Fallback", e)
            self.crepe = None

    def detect_pitch(self, audio: np.ndarray, sample_rate: int = 48000) -> WowFlutterResult:
        """
        Pitch-Trajektorie via hybrides pYIN + CREPE detektieren.

        Args:
            audio: Eingangs-Audio (mono oder stereo)
            sample_rate: Abtastrate in Hz

        Returns:
            WowFlutterResult mit Pitch-Trajektorie und Metadaten
        """
        import time

        start_time = time.time()

        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        strategy = self._determine_strategy(audio, sample_rate)

        pyin_applied = False
        crepe_applied = False
        metadata = {}

        # Stufe 1: pYIN-Detektion (Mauch & Dixon 2014) — §4.2 konform
        if strategy in [PitchDetectionStrategy.PYIN_ONLY, PitchDetectionStrategy.HYBRID]:
            logger.info("Stufe 1: pYIN-Pitch-Detektion (Mauch & Dixon 2014)...")
            pitch_pyin, confidence_pyin = self._apply_pyin(audio, sample_rate)
            pyin_applied = True
            # Direkt als Basis setzen (wird von CREPE ggf. überschrieben)
            pitch_trajectory = pitch_pyin
            confidence = confidence_pyin
            valid_pyin = confidence_pyin[confidence_pyin > 0]
            metadata["pyin"] = {
                "mean_confidence": float(np.mean(valid_pyin)) if len(valid_pyin) > 0 else 0.0,
                "num_estimates": int(np.sum(pitch_pyin > 0)),
            }

            mean_confidence = float(np.mean(valid_pyin)) if len(valid_pyin) > 0 else 0.0
            logger.info(f"pYIN abgeschlossen: mean confidence={mean_confidence:.3f}")

            if mean_confidence >= self.config.confidence_threshold and strategy == PitchDetectionStrategy.HYBRID:
                logger.info(f"pYIN-Konfidenz ausreichend ({mean_confidence:.3f}), CREPE überspringen")
                strategy = PitchDetectionStrategy.PYIN_ONLY

        # Stufe 2: CREPE ML-Verfeinerung (falls nötig)
        if strategy in [PitchDetectionStrategy.CREPE_ONLY, PitchDetectionStrategy.HYBRID]:
            if self.crepe is not None:
                logger.info("Stufe 2: CREPE ML-Pitch-Detektion (Kim et al. 2018)...")
                pitch_crepe, confidence_crepe = self._apply_crepe(audio, sample_rate)
                crepe_applied = True
                valid_crepe = confidence_crepe[confidence_crepe > 0]
                metadata["crepe"] = {
                    "mean_confidence": float(np.mean(valid_crepe)) if len(valid_crepe) > 0 else 0.0,
                    "num_estimates": int(np.sum(pitch_crepe > 0)),
                    "model": self.config.crepe_model,
                }

                mean_confidence_crepe = float(np.mean(valid_crepe)) if len(valid_crepe) > 0 else 0.0
                logger.info(f"CREPE abgeschlossen: mean confidence={mean_confidence_crepe:.3f}")

                pitch_trajectory = pitch_crepe
                confidence = confidence_crepe

                if strategy == PitchDetectionStrategy.HYBRID and pyin_applied:
                    logger.info("pYIN + CREPE Ergebnisse werden gemischt...")
                    pitch_trajectory, confidence = self._blend_pitch_estimates(
                        pitch_pyin, confidence_pyin, pitch_crepe, confidence_crepe
                    )
            else:
                logger.warning("CREPE nicht verfügbar, nutze pYIN-Ergebnis")
                if not pyin_applied:
                    pitch_trajectory, confidence = self._apply_pyin(audio, sample_rate)
                    pyin_applied = True

        # Sicherheitsnetz: pYIN Fallback
        if not pyin_applied and not crepe_applied:
            pitch_trajectory, confidence = self._apply_pyin(audio, sample_rate)
            pyin_applied = True

        processing_time = time.time() - start_time
        valid = confidence[confidence > 0]
        mean_confidence = float(np.mean(valid)) if len(valid) > 0 else 0.0
        metadata["processing_time"] = processing_time

        return WowFlutterResult(
            pitch_trajectory=pitch_trajectory,
            confidence=confidence,
            strategy_used=strategy,
            pyin_applied=pyin_applied,
            crepe_applied=crepe_applied,
            processing_time=processing_time,
            mean_confidence=mean_confidence,
            metadata=metadata,
        )

    def _determine_strategy(self, audio: np.ndarray, sample_rate: int) -> PitchDetectionStrategy:
        """Optimale Pitch-Detektions-Strategie bestimmen."""
        if self.config.strategy != PitchDetectionStrategy.ADAPTIVE:
            return self.config.strategy

        if self.crepe is not None:
            logger.info("Adaptiv: CREPE verfügbar, nutze HYBRID-Modus")
            return PitchDetectionStrategy.HYBRID
        else:
            logger.info("Adaptiv: CREPE nicht verfügbar, nutze PYIN_ONLY-Modus")
            return PitchDetectionStrategy.PYIN_ONLY

    def _apply_pyin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        pYIN-Pitch-Detektion via Phase 12 (Mauch & Dixon 2014).

        Delegiert an WowFlutterFix._estimate_pitch_yin() welche intern
        _estimate_pitch_pyin() via librosa.pyin aufruft.

        Args:
            audio: Mono-Audio
            sample_rate: Abtastrate

        Returns:
            (pitch_trajectory, confidence) als np.ndarray
        """
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        phase = WowFlutterFix()
        pitch_trajectory, confidence = phase._estimate_pitch_yin(audio, sample_rate)
        return pitch_trajectory, confidence

    # Backward-Compat Alias
    def _apply_yin(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Backward-Compat: delegiert an _apply_pyin."""
        return self._apply_pyin(audio, sample_rate)

    def _apply_crepe(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Apply FCPE/CREPE ML pitch detection (numpy-API, kein Subprocess)."""
        try:
            result = self.crepe.analyze(audio, sample_rate)
            # CrepeResult.f0_hz / .voiced_prob sind die finalen Arrays
            f0 = np.nan_to_num(result.f0_hz.astype(np.float32))
            conf = np.clip(np.nan_to_num(result.voiced_prob.astype(np.float32)), 0.0, 1.0)
            return f0, conf
        except Exception as exc:  # noqa: BLE001
            logger.warning("FCPE/CREPE Pitch-Inferenz fehlgeschlagen (%s) — pYIN Fallback", exc)
            return self._apply_pyin(audio, sample_rate)

    def _blend_pitch_estimates(
        self, pitch_pyin: np.ndarray, conf_pyin: np.ndarray, pitch_crepe: np.ndarray, conf_crepe: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        pYIN- und CREPE-Pitch-Schätzwerte konfidenzgewichtet mischen.

        Strategie:
        - CREPE bei hoher Konfidenz (> 0.8)
        - pYIN bei niedriger CREPE-Konfidenz oder fehlenden Schätzwerten
        - Gewichtetes Mischen in unsicheren Regionen
        """
        if len(pitch_crepe) != len(pitch_pyin):
            from scipy import signal as sp_signal

            pitch_crepe = sp_signal.resample(pitch_crepe, len(pitch_pyin))
            conf_crepe = sp_signal.resample(conf_crepe, len(pitch_pyin))

        blended_pitch = np.zeros_like(pitch_pyin)
        blended_conf = np.zeros_like(conf_pyin)

        for i in range(len(pitch_pyin)):
            if conf_crepe[i] > 0.8:
                blended_pitch[i] = pitch_crepe[i]
                blended_conf[i] = conf_crepe[i]
            elif conf_pyin[i] > conf_crepe[i]:
                blended_pitch[i] = pitch_pyin[i]
                blended_conf[i] = conf_pyin[i]
            else:
                total = conf_crepe[i] + conf_pyin[i] + 1e-10
                w_crepe = conf_crepe[i] / total
                w_pyin = 1.0 - w_crepe
                blended_pitch[i] = w_crepe * pitch_crepe[i] + w_pyin * pitch_pyin[i]
                blended_conf[i] = max(conf_crepe[i], conf_pyin[i])

        return blended_pitch, blended_conf


if __name__ == "__main__":
    """Test hybrid wow/flutter detection."""

    logger.debug("=" * 80)
    logger.debug("Hybrid Wow/Flutter Detection Test")
    logger.debug("=" * 80)

    # Generate test audio with pitch variation (simulated wow/flutter)
    duration = 5.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Base frequency (440 Hz A4)
    base_freq = 440.0

    # Add simulated wow (slow pitch drift, <4 Hz)
    wow_freq = 2.0  # 2 Hz wow
    wow_amount = 0.02  # 2% pitch variation
    pitch_variation = 1.0 + wow_amount * np.sin(2 * np.pi * wow_freq * t)

    # Generate audio with pitch variation
    phase = np.cumsum(2 * np.pi * base_freq * pitch_variation / sample_rate)
    audio = 0.5 * np.sin(phase)

    logger.debug(f"Generated {duration}s test audio @ {sample_rate} Hz")
    logger.debug(f"Base frequency: {base_freq} Hz with {wow_amount*100:.1f}% wow at {wow_freq} Hz")
    logger.debug("")

    # Test strategies
    strategies = [
        (PitchDetectionStrategy.PYIN_ONLY, "pYIN Only (Mauch & Dixon 2014)"),
        (PitchDetectionStrategy.HYBRID, "Hybrid (pYIN + CREPE)"),
    ]

    for strategy, name in strategies:
        logger.debug("-" * 80)
        logger.debug(f"Strategy: {name}")
        logger.debug("-" * 80)

        config = WowFlutterConfig(strategy=strategy)
        detector = HybridWowFlutter(config)

        result = detector.detect_pitch(audio, sample_rate)

        logger.debug(f"✅ Strategy used: {result.strategy_used.value}")
        logger.debug(f"   pYIN applied: {result.pyin_applied}")
        logger.debug(f"   CREPE applied: {result.crepe_applied}")
        logger.debug(f"   Mean confidence: {result.mean_confidence:.3f}")
        logger.debug(f"   Pitch estimates: {len(result.pitch_trajectory[result.pitch_trajectory > 0])}")
        logger.debug(f"   Processing time: {result.processing_time:.2f}s")
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("Test complete")
