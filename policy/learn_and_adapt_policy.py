"""
Lern- und Anpassungsmechanismus für SOTA-DSP-Ketten
Wertet Audit-Reports aus und passt Policy/Parameter automatisch an.
"""

import glob
import json
import os

import yaml

# Verzeichnis mit Audit-Reports und Policy-Datei
AUDIT_DIR = "./"
POLICY_PATH = "config_dsp_chain_example.yaml"

# Lade aktuelle Policy
with open(POLICY_PATH, "r") as f:
    policy = yaml.safe_load(f)

# Sammle alle Audit-Reports
audit_files = glob.glob(os.path.join(AUDIT_DIR, "*_audit.json"))


# Statistiken sammeln (global und modul-spezifisch)
hf_ratios = []
corrs = []
artefacts = []
deesser_sib_ratios = []
denoiser_strengths = []
for afile in audit_files:
    with open(afile, "r") as f:
        report = json.load(f)
    hf_ratios.append(report.get("hf_ratio", 1.0))
    corrs.append(report.get("correlation", 1.0))
    artefacts.append(report.get("artefact_energy", 0.0))
    # Modul-spezifische Werte auslesen, falls vorhanden
    if "module_metrics" in report:
        mm = report["module_metrics"]
        if "deesser" in mm and "sib_ratio" in mm["deesser"]:
            deesser_sib_ratios.append(mm["deesser"]["sib_ratio"])
        if "denoiser" in mm and "strength" in mm["denoiser"]:
            denoiser_strengths.append(mm["denoiser"]["strength"])


# Einfache Regel: Passe Schwellenwerte adaptiv an (z.B. 10% unter Median)
import numpy as np

if hf_ratios:
    new_min_hf = max(0.5, float(np.median(hf_ratios)) * 0.9)
    policy.setdefault("quality", {})["min_hf_ratio"] = round(new_min_hf, 3)
if corrs:
    new_min_trans = max(0.5, float(np.median(corrs)) * 0.9)
    policy["quality"]["min_transparency"] = round(new_min_trans, 3)
if artefacts:
    new_max_artefact = min(0.5, float(np.median(artefacts)) * 1.1)
    policy["quality"]["max_artefact"] = round(new_max_artefact, 3)

# Modul-spezifische Parameter adaptiv anpassen
if deesser_sib_ratios:
    # Ziel: Deesser-Threshold so wählen, dass sib_ratio im Zielbereich liegt
    target_sib = 0.7
    median_sib = float(np.median(deesser_sib_ratios))
    # Beispiel: Wenn zu viel Sibilanz übrig, Threshold senken
    if median_sib > target_sib:
        policy.setdefault("deesser", {})["threshold"] = max(0.05, policy.get("deesser", {}).get("threshold", 0.2) * 0.9)
    elif median_sib < target_sib * 0.8:
        policy.setdefault("deesser", {})["threshold"] = min(0.5, policy.get("deesser", {}).get("threshold", 0.2) * 1.1)
if denoiser_strengths:
    # Ziel: Denoiser-Stärke so wählen, dass keine Über- oder Unterdämpfung
    target_strength = 0.8
    median_strength = float(np.median(denoiser_strengths))
    if median_strength < target_strength * 0.9:
        policy.setdefault("denoiser", {})["strength"] = min(1.0, policy.get("denoiser", {}).get("strength", 0.8) * 1.1)
    elif median_strength > target_strength * 1.1:
        policy.setdefault("denoiser", {})["strength"] = max(0.1, policy.get("denoiser", {}).get("strength", 0.8) * 0.9)

# Speichere die angepasste Policy
with open(POLICY_PATH, "w") as f:
    yaml.dump(policy, f)

print("[Lernen] Policy/Parameter wurden automatisch angepasst:")
print(policy)
