import logging
"""
Aurik 6.0 – SOTA-System- und Plugin-Check

Dieses Skript prüft die Kernmodule und Plugins gemäß SOTA-Standards und Dokumentation.
Alle Ausgaben und Begriffe sind an die aktuelle Architektur angepasst.
"""

import glob
import importlib
import os
import sys
from typing import List

import requests

logger = logging.getLogger(__name__)

logger.info("Aurik 6.0 – SOTA System- und Plugin-Check")

# Kernabhängigkeiten
try:
    import soundfile

    logger.info("Check: soundfile OK")
except Exception as e:
    logger.error(f"ERROR: soundfile: {e}")
    sys.exit(1)
try:
    import numpy as np

    logger.info("Check: numpy OK")
except Exception as e:
    logger.error(f"ERROR: numpy: {e}")
    sys.exit(1)
try:
    import onnxruntime

    logger.info("Check: onnxruntime OK")
except Exception as e:
    logger.error(f"ERROR: onnxruntime: {e}")
    sys.exit(1)

# SOTA-Plugin- und Health-Check
plugin_dir = os.path.join("Aurik_Standalone", "plugins")
failed: List[str] = []
for f in glob.glob(os.path.join(plugin_dir, "*.py")):
    mod = os.path.splitext(os.path.basename(f))[0]
    if mod == "__init__":
        continue
    try:
        importlib.import_module(f"plugins.{mod}")
        logger.info(f"[SOTA-Check] Plugin: {mod} OK")
    except Exception as e:
        logger.error(f"[SOTA-Check] ERROR: {mod}: {e}")
        failed.append(mod)

# Health-Endpoint prüfen
try:
    r = requests.get("http://localhost:8000/health", timeout=2)
    if r.status_code == 200 and "ok" in r.text:
        logger.info("[SOTA-Check] Health-Endpoint OK")
    else:
        logger.error(f"[SOTA-Check] Health-Endpoint Fehler: {r.status_code} {r.text}")
        failed.append("health-endpoint")
except Exception as e:
    logger.info(f"[SOTA-Check] Health-Endpoint nicht erreichbar: {e}")
    failed.append("health-endpoint")

if failed:
    logger.error(f"[SOTA-Check] Fehlerhafte Komponenten: {failed}")
    sys.exit(1)
else:
    logger.info("[SOTA-Check] Alle Plugins, Kernmodule und Health-Checks erfolgreich geladen.")
    sys.exit(0)
