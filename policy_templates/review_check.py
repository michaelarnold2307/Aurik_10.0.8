#!/usr/bin/env python3
"""
CLI-Review-Check für Policy-Templates: Führt die automatisierte Prüfung aller YAML-Templates im policy_templates/-Verzeichnis aus und gibt einen Review-Report aus.
"""

import glob
import os
import sys

"""
Hinweis: Die Importe aus aurik4 sind entfernt. Sobald das Testmodul nach aurik6 migriert wurde,
kann es wie folgt importiert werden:
# from aurik6.testing.test_policy_templates import validate_policy_template
"""

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "policy_templates")
REPORT_PATH = os.path.join(TEMPLATE_DIR, "review_report.txt")


def main():
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    any_errors = False
    for path in glob.glob(os.path.join(TEMPLATE_DIR, "*.yaml")):
        errors = validate_policy_template(path)  # noqa: F821
        if errors:
            any_errors = True
            import logging

            logging.error(f"Fehler in {os.path.basename(path)}:")
            for err in errors:
                logging.error(f"  - {err}")
    if any_errors:
        logging.info(f"\nSiehe Review-Report: {REPORT_PATH}")
        sys.exit(1)
    logging.info("Alle Policy-Templates sind fehlerfrei.")


if __name__ == "__main__":
    main()
