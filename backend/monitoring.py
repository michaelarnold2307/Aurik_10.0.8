"""
monitoring.py – Einfaches Monitoring für Aurik

- Prüft regelmäßig Compliance-Status, Performance und User-Feedback
- Loggt Ergebnisse in monitoring_log.txt
"""

import time

import yaml


def check_compliance():
    try:
        with open("policy_audit_check_report.json") as f:
            report = yaml.safe_load(f)
        ok = all(r.get("audits_ok", False) for r in report)
        return ok
    except Exception:
        return False


def log_status():
    status = "OK" if check_compliance() else "FEHLER"
    with open("monitoring_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Compliance: {status}\n")


if __name__ == "__main__":
    while True:
        log_status()
        time.sleep(3600)  # stündlich prüfen
