"""
AURIK v8 Processing Pipeline: Complete Integration
===================================================

Integrates all v8 components into a unified processing pipeline:

1. Zone Engine: Confidence-based zone classification (A/B/C)
2. Musical Goals Measurement: Measure all 7 goals
3. Conduct Enforcer: Validate conduct rules & musical goals
4. Regulator: Pre-validate, adjust parameters, hard stops
5. Monitoring: Continuous tracking with checkpoints
6. Rollback: Snapshot management with undo capability

Architecture Flow:
  Audio Input
       ↓
  Zone Classification (confidence → Zone A/B/C)
       ↓
  Musical Goals Baseline (measure current state)
       ↓
  Pre-Validation (predict impact of processing)
       ↓
  Regulator Decision (ALLOW / ADJUST / HARD_STOP)
       ↓
  [ IF ALLOW/ADJUST ]
  Snapshot (save current state)
       ↓
  Processing (with adjusted parameters if needed)
       ↓
  Checkpoint Monitoring (track goals during processing)
       ↓
  Post-Validation (verify results)
       ↓
  Conduct Enforcement (final validation)
       ↓
  [ IF VIOLATIONS ]
  Rollback (restore previous state)
       ↓
  Audio Output

Quelle: Finalisierungs_Roadmap.md - Component 0.8
Autor: AI Team
Datum: 8. Februar 2026
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for processing pipeline."""

    algorithm: str
    strength: float = 0.8
    aggressiveness: float = 0.7
    medium: str | None = None
    enable_rollback: bool = True
    enable_monitoring: bool = True
    enable_pre_validation: bool = True


@dataclass
class PipelineResult:
    """Result from v8 processing pipeline."""

    success: bool
    audio: np.ndarray
    sr: int
    original_goals: dict[str, float]
    final_goals: dict[str, float]
    zone_classification: str
    confidence: float
    regulator_decisions: list[str]
    violations: list[str]
    rollback_occurred: bool
    checkpoints: list[str]
    processing_log: list[dict]


class ProcessingPipelineV8:
    """
    Complete v8 processing pipeline with all components integrated.

    Example:
        >>> pipeline = ProcessingPipelineV8()
        >>>
        >>> # Configure processing
        >>> config = ProcessingConfig(
        ...     algorithm='DeepFilterNet',
        ...     strength=0.8,
        ...     medium='vinyl'
        ... )
        >>>
        >>> # Run processing
        >>> result = pipeline.process(
        ...     audio=audio,
        ...     sr=48000,
        ...     confidence=0.85,
        ...     config=config,
        ...     processing_func=lambda a, sr, params: process_audio(a, sr, params)
        ... )
        >>>
        >>> if result.success:
        ...     print(f"Processing successful! Zone: {result.zone_classification}")
        ...     print(f"Final musical goals: {result.final_goals}")
        ... else:
        ...     print(f"Processing failed: {result.violations}")
    """

    def __init__(self):
        """Initialize v8 pipeline with all components."""
        try:
            from backend.core.conduct_enforcer import ConductEnforcer
            from backend.core.musical_goals import MusicalGoalsChecker, MusicalGoalsMonitor
            from backend.core.regulator.regulator_v8 import RegulatorV8
            from backend.core.rollback import RollbackManager
            from backend.core.zone_engine import ZoneEngine
        except ModuleNotFoundError:
            # Imports bereits via backend.core korrekt — kein Fallback nötig
            raise

        self.zone_engine = ZoneEngine()
        self.goals_checker = MusicalGoalsChecker()
        self.goals_monitor = MusicalGoalsMonitor()
        self.conduct_enforcer = ConductEnforcer()
        self.regulator = RegulatorV8()
        self.rollback_manager = RollbackManager(max_snapshots=5)

        logger.info("ProcessingPipelineV8 initialized with all v8 components")

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        confidence: float,
        config: ProcessingConfig,
        processing_func: Callable[[np.ndarray, int, dict], np.ndarray],
    ) -> PipelineResult:
        """
        Process audio through v8 pipeline.

        Args:
            audio: Input audio
            sr: Sample rate
            confidence: Epistemic confidence (0.0 - 1.0)
            config: Processing configuration
            processing_func: Processing function (audio, sr, params) → processed_audio

        Returns:
            PipelineResult with complete processing report
        """
        # SR-Invariante (§3.1 Copilot-Instructions)
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        # NaN/Inf-Guard am Eingang (§3.1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        processing_log = []
        regulator_decisions = []
        checkpoints = []

        # Step 1: Zone Classification
        logger.info("Step 1: Zone Classification")
        zone_classification = self.zone_engine.classify_zone(confidence)
        processing_log.append(
            {
                "step": "zone_classification",
                "zone": zone_classification.zone.name,
                "confidence": confidence,
                "cas_multiplier": zone_classification.cas_multiplier,
                "musical_goals_multiplier": zone_classification.musical_goals_multiplier,
            }
        )

        # Get zone-specific thresholds
        thresholds = self.zone_engine.get_musical_goals_for_zone(zone=zone_classification.zone, medium=config.medium)
        processing_log.append({"step": "thresholds", "thresholds": thresholds})

        # Step 2: Measure Baseline Musical Goals
        logger.info("Step 2: Measure Baseline Musical Goals")
        original_goals = self.goals_checker.measure_all(audio, sr)
        processing_log.append({"step": "baseline_goals", "goals": original_goals})

        # Step 3: Pre-Validation (if enabled)
        if config.enable_pre_validation:
            logger.info("Step 3: Pre-Validation")
            pre_validation = self.goals_monitor.pre_validate(
                original_audio=audio,
                sr=sr,
                processing_config={"algorithm": config.algorithm, "strength": config.strength},
                thresholds=thresholds,
            )
            processing_log.append(
                {
                    "step": "pre_validation",
                    "predicted_goals": pre_validation.predicted_goals,
                    "confidence": pre_validation.confidence,
                    "recommendations": pre_validation.recommendations,
                }
            )

            # Step 4: Regulator Pre-Validation Decision
            logger.info("Step 4: Regulator Pre-Validation Decision")
            regulator_decision = self.regulator.pre_validate(
                predicted_goals=pre_validation.predicted_goals,
                thresholds=thresholds,
                current_parameters={"strength": config.strength, "aggressiveness": config.aggressiveness},
            )

            regulator_decisions.append(regulator_decision.reasoning)
            processing_log.append(
                {
                    "step": "regulator_pre_validation",
                    "decision": regulator_decision.decision.value,
                    "allowed": regulator_decision.allowed,
                    "reasoning": regulator_decision.reasoning,
                }
            )

            # Hard stop if rejected
            if not regulator_decision.allowed:
                logger.error("Processing rejected by regulator (pre-validation)")
                # NaN/Inf-Guard + Clip vor Return (§3.1)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                audio = np.clip(audio, -1.0, 1.0)
                return PipelineResult(
                    success=False,
                    audio=audio,  # Return original
                    sr=sr,
                    original_goals=original_goals,
                    final_goals=original_goals,
                    zone_classification=zone_classification.zone.name,
                    confidence=confidence,
                    regulator_decisions=regulator_decisions,
                    violations=regulator_decision.violations.violated_goals if regulator_decision.violations else [],
                    rollback_occurred=False,
                    checkpoints=checkpoints,
                    processing_log=processing_log,
                )

            # Apply parameter adjustments if needed
            if regulator_decision.parameter_adjustments:
                config.strength = regulator_decision.parameter_adjustments.get("strength", config.strength)
                config.aggressiveness = regulator_decision.parameter_adjustments.get(
                    "aggressiveness", config.aggressiveness
                )
                logger.info(
                    f"Applied parameter adjustments: "
                    f"strength={config.strength:.2f}, aggressiveness={config.aggressiveness:.2f}"
                )

        # Step 5: Create Snapshot (if rollback enabled)
        if config.enable_rollback:
            logger.info("Step 5: Create Snapshot")
            self.rollback_manager.create_snapshot(
                name="pre_processing",
                audio=audio,
                sr=sr,
                musical_goals=original_goals,
                metadata={"config": config.__dict__},
            )
            processing_log.append({"step": "snapshot_created", "name": "pre_processing"})

        # Step 6: Processing
        logger.info(f"Step 6: Processing ({config.algorithm})")
        try:
            processed_audio = processing_func(
                audio, sr, {"strength": config.strength, "aggressiveness": config.aggressiveness}
            )
            # NaN/Inf-Guard nach Verarbeitung (§3.1)
            processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.0, neginf=0.0)
            processed_audio = np.clip(processed_audio, -1.0, 1.0)
            processing_log.append(
                {
                    "step": "processing",
                    "algorithm": config.algorithm,
                    "parameters": {"strength": config.strength, "aggressiveness": config.aggressiveness},
                }
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            # NaN/Inf-Guard + Clip vor Return (§3.1)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            audio = np.clip(audio, -1.0, 1.0)
            return PipelineResult(
                success=False,
                audio=audio,
                sr=sr,
                original_goals=original_goals,
                final_goals=original_goals,
                zone_classification=zone_classification.zone.name,
                confidence=confidence,
                regulator_decisions=regulator_decisions,
                violations=["processing_error"],
                rollback_occurred=False,
                checkpoints=checkpoints,
                processing_log=processing_log,
            )

        # Step 7: Checkpoint Monitoring (if enabled)
        if config.enable_monitoring:
            logger.info("Step 7: Checkpoint Monitoring")
            self.goals_monitor.add_checkpoint(
                step_name=f"{config.algorithm}_complete", audio=processed_audio, sr=sr, confidence=confidence
            )
            checkpoints.append(f"{config.algorithm}_complete")
            processing_log.append({"step": "checkpoint", "name": f"{config.algorithm}_complete"})

        # Step 8: Post-Validation
        logger.info("Step 8: Post-Validation")
        final_goals = self.goals_checker.measure_all(processed_audio, sr)
        processing_log.append({"step": "post_validation_goals", "goals": final_goals})

        # Step 9: Regulator Post-Validation
        logger.info("Step 9: Regulator Post-Validation")
        post_regulator_decision = self.regulator.post_validate(
            original_goals=original_goals, processed_goals=final_goals, thresholds=thresholds
        )

        regulator_decisions.append(post_regulator_decision.reasoning)
        processing_log.append(
            {
                "step": "regulator_post_validation",
                "decision": post_regulator_decision.decision.value,
                "allowed": post_regulator_decision.allowed,
                "reasoning": post_regulator_decision.reasoning,
            }
        )

        # Step 10: Conduct Enforcement
        logger.info("Step 10: Conduct Enforcement")
        conduct_validation = self.conduct_enforcer.validate_step(
            step_name=f"{config.algorithm}_final",
            musical_goals_pre=original_goals,
            musical_goals_predicted=final_goals,
            zone=zone_classification.zone.value,
            confidence=confidence,
            medium=config.medium,
        )

        processing_log.append(
            {
                "step": "conduct_enforcement",
                "allowed": conduct_validation.allowed,
                "reason": conduct_validation.reason,
                "violations": conduct_validation.violations,
            }
        )

        # Step 11: Rollback Decision
        violations_list = []
        if post_regulator_decision.violations:
            violations_list.extend(post_regulator_decision.violations.violated_goals)
        if conduct_validation.violations:
            violations_list.extend(conduct_validation.violations)

        violations_list = list(set(violations_list))  # Deduplicate

        if violations_list and config.enable_rollback:
            logger.warning(f"Violations detected: {violations_list}. Rolling back...")
            rolled_back_audio, rolled_back_sr, rolled_back_goals = self.rollback_manager.rollback_to_snapshot(
                "pre_processing", reason=f"Musical goal violations: {', '.join(violations_list)}"
            )
            # NaN/Inf-Guard nach Rollback (§3.1)
            rolled_back_audio = np.nan_to_num(rolled_back_audio, nan=0.0, posinf=0.0, neginf=0.0)
            rolled_back_audio = np.clip(rolled_back_audio, -1.0, 1.0)
            processing_log.append({"step": "rollback", "reason": f"Violations: {', '.join(violations_list)}"})

            return PipelineResult(
                success=False,
                audio=rolled_back_audio,
                sr=rolled_back_sr,
                original_goals=original_goals,
                final_goals=rolled_back_goals,
                zone_classification=zone_classification.zone.name,
                confidence=confidence,
                regulator_decisions=regulator_decisions,
                violations=violations_list,
                rollback_occurred=True,
                checkpoints=checkpoints,
                processing_log=processing_log,
            )

        # Step 12: Success
        logger.info("Step 12: Processing Complete (success)")
        # NaN/Inf-Guard + Clip vor finalem Return (§3.1)
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.0, neginf=0.0)
        processed_audio = np.clip(processed_audio, -1.0, 1.0)
        return PipelineResult(
            success=True,
            audio=processed_audio,
            sr=sr,
            original_goals=original_goals,
            final_goals=final_goals,
            zone_classification=zone_classification.zone.name,
            confidence=confidence,
            regulator_decisions=regulator_decisions,
            violations=[],
            rollback_occurred=False,
            checkpoints=checkpoints,
            processing_log=processing_log,
        )


if __name__ == "__main__":
    # Test v8 Pipeline Integration
    logger.info("=== AURIK v8 Pipeline Integration Test ===\n")

    # Create test audio
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = (
        0.3 * np.sin(2 * np.pi * 100 * t)
        + 0.3 * np.sin(2 * np.pi * 500 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.2 * np.sin(2 * np.pi * 8000 * t)
    )

    # Mock processing function (adds slight noise)
    def mock_processing(audio_in, sr_in, params):
        logger.info(f"   Processing with params: {params}")
        # Add slight noise (simulates processing)
        noise = np.random.normal(0, 0.01, len(audio_in))
        return audio_in * 0.98 + noise  # Slight degradation

    # Initialize pipeline
    pipeline = ProcessingPipelineV8()

    # Configure processing
    config = ProcessingConfig(
        algorithm="MockProcessor",
        strength=0.6,
        aggressiveness=0.5,
        medium="vinyl",
        enable_rollback=True,
        enable_monitoring=True,
        enable_pre_validation=True,
    )

    # Run processing
    logger.info("Running v8 pipeline...")
    result = pipeline.process(
        audio=audio, sr=sr, confidence=0.85, config=config, processing_func=mock_processing  # Zone B
    )

    # Print results
    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE RESULT")
    logger.info(f"{'='*70}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Zone: {result.zone_classification}")
    logger.info(f"Confidence: {result.confidence:.3f}")
    logger.info(f"Rollback occurred: {result.rollback_occurred}")
    logger.info(f"Violations: {result.violations if result.violations else 'None'}")
    logger.info(f"\nOriginal Musical Goals:")
    for goal, score in result.original_goals.items():
        logger.info(f"  {goal:20s}: {score:.3f}")
    logger.info(f"\nFinal Musical Goals:")
    for goal, score in result.final_goals.items():
        delta = score - result.original_goals[goal]
        delta_str = f"({delta:+.3f})" if delta != 0 else ""
        logger.info(f"  {goal:20s}: {score:.3f} {delta_str}")
    logger.info(f"\nRegulator Decisions:")
    for i, decision in enumerate(result.regulator_decisions, 1):
        logger.info(f"  [{i}] {decision}")
    logger.info(f"\nProcessing Steps: {len(result.processing_log)}")
    logger.info(f"Checkpoints: {len(result.checkpoints)}")
    logger.info(f"{'='*70}\n")

    logger.info("=== Test complete ===")
