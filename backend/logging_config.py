import logging
from logging.handlers import RotatingFileHandler
import os

from .error_notifier import setup_error_notifier

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "aurik_backend.log")

# Rotierendes Logfile: 5MB, 5 Backups
handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(handler)

# Fehlerbenachrichtigung aktivieren (E-Mail), falls konfiguriert
setup_error_notifier()

# Optional: Fehler-Alerts (z.B. per E-Mail) können hier ergänzt werden


def get_logger(name=None, level: int = logging.INFO) -> logging.Logger:
    """
    Gibt einen konfigurierten Logger zurück.
    Kombiniert globales Logfile-Setup mit individuellem Level-Support
    (portiert aus backend.core.regulator.logging_config).
    """
    lg = logging.getLogger(name)
    lg.setLevel(level)
    return lg
