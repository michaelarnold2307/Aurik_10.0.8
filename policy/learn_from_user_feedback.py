"""
User-Feedback-Integration für Policy-Lernen
Liest User-Feedback (JSON) ein und passt Policy/Parameter adaptiv an.
"""

import json
import os

import yaml

POLICY_PATH = "config_dsp_chain_example.yaml"
FEEDBACK_PATH = "user_feedback.json"

# Lade aktuelle Policy
with open(POLICY_PATH, "r") as f:
    policy = yaml.safe_load(f)

# Lade User-Feedback (z.B. [{"file":..., "feedback": {"dull": true, "artefacts": false, ...}}, ...])
if not os.path.exists(FEEDBACK_PATH):
    print("[Feedback] Keine user_feedback.json gefunden. Bitte Feedbackdatei anlegen.")
    exit(0)

with open(FEEDBACK_PATH, "r") as f:
    feedbacks = json.load(f)

# Feedback auswerten und Policy anpassen
for entry in feedbacks:
    fb = entry.get("feedback", {})
    # Dumpfheit: min_hf_ratio erhöhen
    if fb.get("dull", False):
        policy.setdefault("quality", {})["min_hf_ratio"] = min(
            1.0, policy.get("quality", {}).get("min_hf_ratio", 0.7) + 0.05
        )
    # Artefakte: max_artefact senken
    if fb.get("artefacts", False):
        policy.setdefault("quality", {})["max_artefact"] = max(
            0.01, policy.get("quality", {}).get("max_artefact", 0.2) * 0.9
        )
    # Sibilanz: Deesser-Threshold senken
    if fb.get("sibilance", False):
        policy.setdefault("deesser", {})["threshold"] = max(0.05, policy.get("deesser", {}).get("threshold", 0.2) * 0.9)
    # Zu wenig Rauschunterdrückung: Denoiser-Stärke erhöhen
    if fb.get("noise_left", False):
        policy.setdefault("denoiser", {})["strength"] = min(1.0, policy.get("denoiser", {}).get("strength", 0.8) * 1.1)
    # Zu viel Rauschunterdrückung: Denoiser-Stärke senken
    if fb.get("overdenoised", False):
        policy.setdefault("denoiser", {})["strength"] = max(0.1, policy.get("denoiser", {}).get("strength", 0.8) * 0.9)

# Speichere die angepasste Policy
with open(POLICY_PATH, "w") as f:
    yaml.dump(policy, f)

print("[Feedback-Lernen] Policy/Parameter wurden durch User-Feedback angepasst:")
print(policy)
