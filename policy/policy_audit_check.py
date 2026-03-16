from glob import glob
import json
import os

import yaml

POLICY_OVERVIEW = os.path.join(os.path.dirname(__file__), "dsp_policy_contracts_overview.yaml")
AUDIT_DIR = "."

with open(POLICY_OVERVIEW, "r") as f:
    overview = yaml.safe_load(f)

policies = overview.get("policies", [])
results = []
for policy in policies:
    name = policy["name"]
    media = policy["media"]
    defects = policy.get("defects", [])
    q = policy["quality_targets"]
    audit_required = policy["audit"].get("required", False)
    min_modules = policy["compliance"].get("min_modules", 1)
    sota_required = policy["compliance"].get("sota_required", False)

    # Suche nach Audit-Reports für diese Medienart (YAML und JSON, inkl. Kombis/Härtefälle)
    audit_files_yaml = [f for f in glob(os.path.join(AUDIT_DIR, f"*{media}*.yaml"))]
    audit_files_json = [f for f in glob(os.path.join(AUDIT_DIR, f"*{media}*.json"))]
    audit_files = audit_files_yaml + audit_files_json
    audits_ok = len(audit_files) > 0 if audit_required else True
    failed_audits = []
    for afile in audit_files:
        if afile.endswith(".yaml"):
            with open(afile, "r") as af:
                arep = yaml.safe_load(af)
        else:
            with open(afile, "r") as af:
                arep = json.load(af)
        # Policy-Block und Qualitätsziele auslesen
        qrep = arep.get("policy", {}).get("quality", {})
        # Prüfe Qualitätsziele
        if (
            qrep.get("hf_ratio", 1.0) < q["min_hf_ratio"]
            or qrep.get("min_transparency", 1.0) < q["min_transparency"]
            or qrep.get("artefact_energy", 0.0) > q["max_artefact"]
        ):
            failed_audits.append(afile)
    result = {
        "policy": name,
        "media": media,
        "audit_reports_found": len(audit_files),
        "audit_reports_failed": len(failed_audits),
        "audit_required": audit_required,
        "audits_ok": audits_ok and len(failed_audits) == 0,
        "failed_audits": failed_audits,
    }
    results.append(result)

# Ausgabe
print("\nPolicy/Audit-Check Report:")
for r in results:
    print(
        f"- Policy: {r['policy']} | Media: {r['media']} | Audits gefunden: {r['audit_reports_found']} | Fehlerhafte Audits: {r['audit_reports_failed']} | OK: {r['audits_ok']}"
    )
    if r["failed_audits"]:
        print(f"  Fehlerhafte Audit-Reports: {r['failed_audits']}")

# Optional: JSON-Export
with open("policy_audit_check_report.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nReport gespeichert als policy_audit_check_report.json")
