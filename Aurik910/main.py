#!/usr/bin/env python3
"""
AURIK Professional - Main Application Entry Point
Launch the desktop application for audio restoration
"""

from pathlib import Path
import sys
import logging
import threading
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Logging-Setup: File + Konsole ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / "aurik_backend.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

# Datei-Handler (5 MB, Rotation)
from logging.handlers import RotatingFileHandler as _RFH
_fh = _RFH(str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
_root_logger.addHandler(_fh)

# Konsole-Handler (nur WARNING+)
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
_root_logger.addHandler(_ch)
# ──────────────────────────────────────────────────────────────────────────────

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from Aurik910.ui.modern_window import ModernMainWindow


def _warmup_models_background() -> None:
    """Lädt ONNX-Sessions aller Pflicht-Plugins im Hintergrund (§9.7.4).

    2 s Verzögerung stellt sicher, dass App-Fenster vollständig sichtbar
    ist bevor Hintergrundlast beginnt. Kein Absturz bei fehlendem Plugin.
    """
    time.sleep(2)
    import importlib

    for _mod, _accessor in (
        ("plugins.panns_plugin", "get_panns_plugin"),
        ("plugins.crepe_plugin", "get_crepe_plugin"),
        ("plugins.deepfilternet_v3_ii_plugin", "get_deepfilternet"),
    ):
        try:
            m = importlib.import_module(_mod)
            fn = getattr(m, _accessor, None)
            if fn is not None:
                fn()
        except Exception:
            pass  # Lazy-Load uebernimmt bei Bedarf


def main():
    """Launch AURIK Professional"""
    # Enable high DPI scaling (PyQt5-Stubs kennen diese Attribute nicht -> ignore)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("AURIK Professional")
    app.setOrganizationName("AURIK")
    app.setApplicationVersion("9.10.51")

    # Set dark theme style
    app.setStyle("Fusion")
    # Global QToolTip styling — prevents the default black/system-colored tooltip box.
    # Border uses the app's purple accent; background matches the dark UI palette.
    app.setStyleSheet(
        "QToolTip {"
        "  background-color: #1a1a2e;"
        "  color: #d8d8f0;"
        "  border: 1px solid #4a3878;"
        "  border-radius: 6px;"
        "  padding: 6px 10px;"
        "  font-size: 9pt;"
        "}"
    )

    # Create and show main window
    window = ModernMainWindow()
    window.show()

    # §9.7.4 Modell-Warmup im Hintergrund (daemon=True -> endet mit App)
    threading.Thread(
        target=_warmup_models_background,
        daemon=True,
        name="AurikWarmup",
    ).start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
