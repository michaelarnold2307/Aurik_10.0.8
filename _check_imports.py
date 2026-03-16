#!/usr/bin/env python3
"""Schneller Import-Check aller Kern-Module – ohne pytest-Overhead."""
import sys
import traceback

MODULES = [
    "backend.core.unified_restorer_v3",
    "backend.core.transient_decoupled_processor",
    "backend.core.harmonic_preservation_guard",
    "backend.core.per_phase_musical_goals_gate",
    "backend.core.micro_dynamics_envelope_morphing",
    "backend.core.stem_remix_balancer",
    "backend.core.ensemble_processor",
    "backend.core.perceptual_attention_model",
    "backend.core.introduced_artifact_detector",
    "backend.core.goal_applicability_filter",
    "backend.core.goal_priority_protocol",
    "backend.core.physical_ceiling_estimator",
    "backend.core.restorability_estimator",
    "backend.core.era_authentic_perceptual_completion",
    "backend.core.genre_classifier",
    "backend.core.content_aware_processor",
    "backend.core.defect_scanner",
    "backend.core.causal_defect_reasoner",
    "backend.core.gp_parameter_optimizer",
    "backend.core.perceptual_quality_scorer",
    "backend.core.excellence_optimizer",
    "backend.core.feedback_chain",
    "backend.core.medium_classifier",
    "backend.core.musical_goals.musical_goals_metrics",
    "backend.core.musical_goals.adaptive_goals_system",
    "plugins.panns_plugin",
    "plugins.crepe_plugin",
    "plugins.diffwave_plugin",
    "plugins.vocos_plugin",
    "plugins.apollo_plugin",
    "denker",
    "denker.aurik_denker",
    "denker.defekt_denker",
    "denker.exzellenz_denker",
]

ok = 0
err = 0
for m in MODULES:
    try:
        __import__(m)
        print(f"OK   {m}")
        ok += 1
    except Exception as e:
        tb = traceback.format_exc().strip().split("\n")
        short_tb = "\n      ".join(tb[-3:])
        print(f"ERR  {m}\n      {short_tb}")
        err += 1

print(f"\n{'='*60}")
print(f"OK: {ok}  |  FEHLER: {err}  |  GESAMT: {ok+err}")
