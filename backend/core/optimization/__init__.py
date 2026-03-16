"""
Optimization Package für Aurik 8.0

Stellt alle Optimierungswerkzeuge bereit:
- Perceptual Loss Functions
- End-to-End Optimization
- Hyperparameter Optimization
- Neural Architecture Search (NAS)
- Advanced Ensemble Strategies
- Multi-Objective Optimization
- Uncertainty Quantification
- Automated Data Augmentation

Autor: Aurik Backend-Team
Version: 8.2
Datum: 14. Februar 2026
"""

from .advanced_ensemble import (
    AdvancedEnsemble,
    AttentionWeightPredictor,
    DynamicEnsembleSelector,
    EnsembleMember,
    MetaLearner,
    MixtureOfExperts,
)
from .automated_augmentation import (
    AudioAugmentations,
    AugmentationPolicy,
    AutoAugment,
    ConsistencyTraining,
    RandAugment,
)
from .e2e_optimizer import DifferentiableCompressor, DifferentiableEQ, DifferentiableNoiseGate, E2EOptimizationFramework
from .hyperparameter_optimizer import HyperparameterConfig, MaterialSpecificOptimizer, MultiMaterialOptimizer
from .multi_objective import NSGAII, Individual, ObjectiveFunction, create_audio_restoration_moo
from .neural_architecture_search import AudioNASNetwork, DARTSCell, MixedOp, NASTrainer
from .perceptual_loss import (
    CombinedPerceptualLoss,
    MultiResolutionSTFTLoss,
    MusicalFeatureLoss,
    PANNsPerceptualLoss,
    PsychoacousticMaskingLoss,
)
from .uncertainty_quantification import (
    BayesianLinear,
    BayesianNN,
    EnsembleUncertainty,
    MCDropoutModel,
    TemperatureScaling,
    UncertaintyMetrics,
    UncertaintyQuantifier,
)

__all__ = [
    # Perceptual Loss
    "MultiResolutionSTFTLoss",
    "PANNsPerceptualLoss",
    "PsychoacousticMaskingLoss",
    "MusicalFeatureLoss",
    "CombinedPerceptualLoss",
    # E2E Optimization
    "DifferentiableEQ",
    "DifferentiableCompressor",
    "DifferentiableNoiseGate",
    "E2EOptimizationFramework",
    # Hyperparameter Optimization
    "HyperparameterConfig",
    "MaterialSpecificOptimizer",
    "MultiMaterialOptimizer",
    # Neural Architecture Search
    "MixedOp",
    "DARTSCell",
    "AudioNASNetwork",
    "NASTrainer",
    # Advanced Ensemble
    "EnsembleMember",
    "MetaLearner",
    "AttentionWeightPredictor",
    "DynamicEnsembleSelector",
    "MixtureOfExperts",
    "AdvancedEnsemble",
    # Multi-Objective Optimization
    "Individual",
    "ObjectiveFunction",
    "NSGAII",
    "create_audio_restoration_moo",
    # Uncertainty Quantification
    "MCDropoutModel",
    "BayesianLinear",
    "BayesianNN",
    "EnsembleUncertainty",
    "TemperatureScaling",
    "UncertaintyMetrics",
    "UncertaintyQuantifier",
    # Automated Augmentation
    "AudioAugmentations",
    "AugmentationPolicy",
    "RandAugment",
    "AutoAugment",
    "ConsistencyTraining",
]

__version__ = "8.2.0"
