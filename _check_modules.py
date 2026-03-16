"""Einmaliger Import-Check aller Pflicht-Module laut Spec."""
import sys
sys.path.insert(0, '/media/michael/Software 4TB/Aurik_Standalone')

modules_to_check = [
    ('backend.core.transient_decoupled_processor', 'TransientDecoupledProcessing'),
    ('backend.core.harmonic_preservation_guard', 'HarmonicPreservationGuard'),
    ('backend.core.per_phase_musical_goals_gate', 'PerPhaseMusicalGoalsGate'),
    ('backend.core.micro_dynamics_envelope_morphing', 'MicroDynamicsEnvelopeMorphing'),
    ('backend.core.musical_goals.adaptive_goals_system', 'AdaptiveGoalThresholds'),
    ('backend.core.musical_goals.goal_applicability_filter', 'GoalApplicabilityFilter'),
    ('backend.core.musical_goals.physical_ceiling_estimator', 'PhysicalCeilingEstimator'),
    ('backend.core.musical_goals.goal_priority_protocol', 'GoalPriorityProtocol'),
    ('backend.core.era_authentic_perceptual_completion', 'EraAuthenticPerceptualCompletion'),
    ('backend.core.lyrics_guided_enhancement', 'LyricsTranscriber'),
    ('backend.core.lyrics_guided_enhancement', 'ContentAwareProcessor'),
    ('backend.core.genre_classifier', 'GermanSchlagerClassifier'),
    ('backend.core.remaster_detector', 'RemasterDetector'),
    ('backend.core.ensemble_processor', 'EnsembleProcessor'),
    ('backend.core.perceptual_attention_model', 'PerceptualAttentionModel'),
    ('backend.core.introduced_artifact_detector', 'IntroducedArtifactDetector'),
    ('backend.core.batch_session_learner', 'BatchSessionLearner'),
    ('backend.core.reference_anchor_synthesizer', 'ReferenceAnchorSynthesizer'),
    ('backend.core.restorability_estimator', 'RestorabilityEstimator'),
    ('backend.core.emotional_arc_preservation', 'EmotionalArcPreservationMetric'),
    ('backend.core.temporal_quality_coherence', 'TemporalQualityCoherenceMetric'),
    ('backend.core.stem_remix_balancer', 'StemRemixBalancer'),
    ('backend.core.progressive_quality_mode', 'ProgressiveQualityMode'),
    ('backend.core.optimization.uncertainty_quantification', 'UncertaintyQuantifier'),
    ('backend.core.spectral_band_gap_repair', 'SpectralBandGapRepair'),
    ('backend.error_notifier', 'setup_error_notifier'),
]

ok, fail = [], []
for modname, cls in modules_to_check:
    try:
        mod = __import__(modname, fromlist=[cls])
        getattr(mod, cls)
        ok.append(f'  OK  {modname}.{cls}')
    except Exception as e:
        fail.append(f'  FAIL {modname}.{cls}: {e}')

print(f'OK: {len(ok)}/{len(modules_to_check)}')
if fail:
    print('FEHLER:')
    for f in fail:
        print(f)
else:
    print('Alle Pflicht-Module importierbar!')
