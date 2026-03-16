"""
Quality Mode System - Aurik 9.0
================================

Zentrale Steuerung für DSP vs ML-Hybrid Modi.

Modes:
- FAST: Pure DSP (0.7× RT, Score 0.83)
- BALANCED: Adaptive Hybrid (1.8× RT, Score ~0.90)
- MAXIMUM: Pure ML (4.5× RT, Score ~0.92)

Author: Aurik 9.0 Development Team
Date: 15. Februar 2026
"""

from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class QualityMode(Enum):
    """Quality modes for processing."""

    FAST = "fast"  # Pure DSP only
    BALANCED = "balanced"  # Adaptive Hybrid (Default)
    MAXIMUM = "maximum"  # Pure ML where available

    @classmethod
    def from_string(cls, mode_str: str) -> "QualityMode":
        """Convert string to QualityMode."""
        mode_map = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "maximum": cls.MAXIMUM,
            "dsp": cls.FAST,
            "hybrid": cls.BALANCED,
            "ml": cls.MAXIMUM,
        }
        return mode_map.get(mode_str.lower(), cls.BALANCED)

    def __str__(self) -> str:
        return self.value


class QualityModeConfig:
    """Global configuration for quality mode."""

    _current_mode: QualityMode = QualityMode.BALANCED

    @classmethod
    def set_mode(cls, mode: QualityMode) -> None:
        """Set global quality mode."""
        cls._current_mode = mode
        logger.info(f"Quality mode set to: {mode.value}")

    @classmethod
    def get_mode(cls) -> QualityMode:
        """Get current quality mode."""
        return cls._current_mode

    @classmethod
    def should_use_ml(cls, phase_name: str, defect_severity: float = 0.5) -> bool:
        """
        Determine if ML should be used for a specific phase.

        Args:
            phase_name: Name of the phase
            defect_severity: Severity of defects (0.0-1.0)

        Returns:
            True if ML should be used, False for DSP only
        """
        mode = cls._current_mode

        if mode == QualityMode.FAST:
            return False  # Always DSP

        elif mode == QualityMode.MAXIMUM:
            return True  # Always ML if available

        else:  # BALANCED
            # Adaptive: Use ML for severe defects
            return defect_severity > 0.6

    @classmethod
    def get_expected_performance(cls) -> dict[str, Any]:
        """Get expected performance metrics for current mode."""
        performance = {
            QualityMode.FAST: {
                "realtime_factor": 0.7,
                "expected_score": 0.83,
                "natuerlichkeit": 0.55,
                "description": "Pure DSP - Schnellste Verarbeitung",
            },
            QualityMode.BALANCED: {
                "realtime_factor": 1.8,
                "expected_score": 0.90,
                "natuerlichkeit": 0.80,
                "description": "Adaptive Hybrid - Balance zwischen Speed & Quality",
            },
            QualityMode.MAXIMUM: {
                "realtime_factor": 4.5,
                "expected_score": 0.92,
                "natuerlichkeit": 0.85,
                "description": "Pure ML - Höchste Qualität",
            },
        }
        return performance[cls._current_mode]


# ML Model Availability Registry
ML_MODELS_AVAILABLE = {
    "audiosr": True,  # Phase 23, 24
    "deepfilternet": True,  # Phase 2, 3, 29
    "silero_vad": True,  # Phase 18
    "banquet": True,  # Phase 9
    "mp_senet": True,  # Phase 1, 3, 29 — §4.4: MP-SENet 2023 (ersetzt DCCRN + FullSubNet+)
}


def check_ml_available(model_name: str) -> bool:
    """Check if ML model is available."""
    return ML_MODELS_AVAILABLE.get(model_name, False)


# Phase-specific ML recommendations
PHASE_ML_CONFIG = {
    1: {  # Click Removal
        "ml_model": "mp_senet",  # §4.4: MP-SENet 2023 ersetzt DCCRN
        "hybrid_strategy": "dsp_detect_ml_repair",
        "improvement": 0.25,
        "critical": False,
    },
    2: {  # Hum Removal
        "ml_model": "deepfilternet",
        "hybrid_strategy": "dual_stage",
        "improvement": 0.25,
        "critical": False,
    },
    3: {  # Denoise
        "ml_model": "deepfilternet",
        "hybrid_strategy": "snr_based_routing",
        "improvement": 0.12,
        "critical": False,
    },
    9: {  # Crackle Removal
        "ml_model": "banquet",
        "hybrid_strategy": "material_adaptive",
        "improvement": 0.35,
        "critical": True,  # Vinyl restoration
    },
    18: {  # Noise Gate
        "ml_model": "silero_vad",
        "hybrid_strategy": "ml_vad_dsp_gate",
        "improvement": 0.35,
        "critical": True,  # High impact, easy implementation
    },
    23: {  # Spectral Repair
        "ml_model": "audiosr",
        "hybrid_strategy": "dsp_detect_ml_repair",
        "improvement": 0.45,
        "critical": True,  # HIGHEST PRIORITY - worst phase
    },
    24: {  # Dropout Repair
        "ml_model": "audiosr",
        "hybrid_strategy": "length_based_routing",
        "improvement": 0.30,
        "critical": False,
    },
    29: {  # Tape Hiss Reduction
        "ml_model": "deepfilternet",
        "hybrid_strategy": "band_specific",
        "improvement": 0.30,
        "critical": False,
    },
}


def get_phase_ml_config(phase_number: int) -> dict[str, Any] | None:
    """Get ML configuration for a specific phase."""
    return PHASE_ML_CONFIG.get(phase_number)


def is_phase_ml_enabled(phase_number: int) -> bool:
    """Check if ML is enabled for a phase based on current mode."""
    config = get_phase_ml_config(phase_number)
    if not config:
        return False

    mode = QualityModeConfig.get_mode()

    if mode == QualityMode.FAST:
        return False
    elif mode == QualityMode.MAXIMUM:
        return check_ml_available(config["ml_model"])
    else:  # BALANCED
        # Only critical phases in balanced mode
        return config.get("critical", False) and check_ml_available(config["ml_model"])


# Logging helper
def log_mode_decision(phase_name: str, use_ml: bool, reason: str) -> None:
    """Log quality mode decision for debugging."""
    mode = QualityModeConfig.get_mode()
    ml_status = "ML" if use_ml else "DSP"
    logger.debug(f"Phase {phase_name} | Mode: {mode.value} | Using: {ml_status} | Reason: {reason}")


if __name__ == "__main__":
    # Test quality modes
    logger.debug("Quality Mode System Test\n" + "=" * 50)

    for mode in QualityMode:
        QualityModeConfig.set_mode(mode)
        perf = QualityModeConfig.get_expected_performance()

        logger.debug(f"\nMode: {mode.value.upper()}")
        logger.debug(f"  RT Factor: {perf['realtime_factor']}×")
        logger.debug(f"  Score: {perf['expected_score']}")
        logger.debug(f"  Natürlichkeit: {perf['natuerlichkeit']}")
        logger.debug(f"  {perf['description']}")

        # Test phase decisions
        logger.debug("\n  ML-Enabled Phases:")
        for phase_num in [1, 3, 9, 18, 23, 29]:
            config = get_phase_ml_config(phase_num)
            if config:
                enabled = is_phase_ml_enabled(phase_num)
                status = "✓" if enabled else "✗"
                logger.debug(f"    {status} Phase {phase_num:02d}: {config['ml_model']}")

    logger.debug("\n" + "=" * 50)
    logger.debug("✅ Quality Mode System initialized")
