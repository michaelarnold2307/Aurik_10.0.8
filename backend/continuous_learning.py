"""
Rückwärtskompatibilitäts-Shim (§9.4 Anti-Parallelwelten-Prinzip).

Die kanonische Implementierung liegt in
``backend/core/evaluation/continuous_learning.py``.
"""

from backend.core.evaluation.continuous_learning import (  # noqa: F401
    ContinuousLearningSystem,
    ConfidenceCalibrator,
    LearningRecommendation,
    LearningReport,
    PerformanceMetricsAggregator,
    ProcessingStrategy,
    StrategyWeightOptimizer,
    SuccessPatternAnalyzer,
)

__all__ = [
    "ProcessingStrategy",
    "LearningRecommendation",
    "LearningReport",
    "SuccessPatternAnalyzer",
    "StrategyWeightOptimizer",
    "ConfidenceCalibrator",
    "PerformanceMetricsAggregator",
    "ContinuousLearningSystem",
]
