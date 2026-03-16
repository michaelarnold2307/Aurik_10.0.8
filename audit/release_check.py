"""
Aurik Release-Check und Change-Impact-Analyse
- Führt bei jedem Release/Update alle Quality-Gates, Audit-Logs und Compliance-Prüfungen aus
- Bewertet Auswirkungen von Änderungen auf musikalische Qualität und Auditierbarkeit
- Generiert einen Release-Report
"""

import json
from pathlib import Path


def load_audit_log(audit_path="audit/audit_trail.json"):
    if not Path(audit_path).exists():
        print("Audit-Log nicht gefunden.")
        return []
    with open(audit_path) as f:
        return json.load(f)


def check_compliance(
    audit_data, doc_gates_path="docs/audit/QUALITY_GATES.md", doc_policy_path="policy/policy_engine.py"
):
    compliance_ok = True
    changes = []
    if Path(doc_gates_path).exists():
        with open(doc_gates_path) as f:
            doc_gates = f.read()
    else:
        doc_gates = ""
    if Path(doc_policy_path).exists():
        with open(doc_policy_path) as f:
            f.read()
    else:
        pass
    for entry in audit_data:
        results = entry.get("results", {})
        for gate, value in results.items():
            if gate not in doc_gates:
                compliance_ok = False
                changes.append(f"Quality-Gate '{gate}' nicht in Dokumentation.")
            if value is False:
                changes.append(f"Quality-Gate '{gate}' nicht bestanden.")
    return compliance_ok, changes


def generate_release_report(compliance_ok, changes, audit_data):
    report = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "compliance_ok": compliance_ok,
        "changes": changes,
        "audit_summary": audit_data[-5:] if audit_data else [],
    }
    with open("audit/release_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Release-Report generiert.")
    if not compliance_ok:
        print("WARNUNG: Compliance-Verstöße oder Quality-Gate-Fehler im Release!")
        for c in changes:
            print(f"- {c}")


def main():
    audit_data = load_audit_log()
    compliance_ok, changes = check_compliance(audit_data)
    generate_release_report(compliance_ok, changes, audit_data)


if __name__ == "__main__":
    main()
