#!/usr/bin/env python3
"""
AURIK Professional - Main Application Entry Point
Launch the desktop application for audio restoration
"""

import logging
from pathlib import Path
import sys
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

logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from Aurik910.ui.modern_window import ModernMainWindow


def _run_startup_model_check(app: QApplication) -> None:
    """Prüft ML-Modelle vor dem Fensteraufbau — zeigt deutschen Dialog bei Problemen.

    Nicht-blockierend bei fehlenden optionalen Modellen.
    Warnung bei fehlenden Primär-Modellen (DSP-Fallback aktiv).
    """
    try:
        from backend.api.bridge import get_startup_check_result  # type: ignore[import]

        result = get_startup_check_result()
        if result is None:
            return
        if not result.all_ok and result.user_message_de:
            icon = QMessageBox.Icon.Critical if result.is_critical else QMessageBox.Icon.Warning
            box = QMessageBox()
            box.setWindowTitle("AURIK — " + result.user_title_de)
            box.setText(result.user_message_de)
            box.setIcon(icon)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            box.exec_()
    except Exception as exc:
        # Startup-Check darf niemals den App-Start blockieren
        logger.warning("Startup-Modell-Check fehlgeschlagen (non-fatal): %s", exc)


def _warmup_models_background() -> None:
    """Start background warmup of ML models — non-blocking daemon thread."""
    import threading

    def _run() -> None:
        try:
            from backend.api.bridge import warmup_models_background as _wb  # type: ignore[import]

            _wb()
        except Exception as _e:
            logger.debug("Warmup fehlgeschlagen (non-fatal): %s", _e)

    t = threading.Thread(target=_run, daemon=True, name="AurikWarmup")
    t.start()


def _process_events_ms(app: "QApplication", ms: int) -> None:
    """Process Qt events for *ms* milliseconds — keeps splash animation alive."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.016)  # ~60 fps


def main():
    """Launch AURIK Professional"""
    # Enable high DPI scaling (PyQt5-Stubs kennen diese Attribute nicht -> ignore)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("AURIK Professional")
    app.setOrganizationName("AURIK")
    app.setApplicationVersion("9.10.57")

    # Application icon (taskbar, dock, alt-tab)
    _icon_path = Path(__file__).parent / "resources" / "icon.png"
    if _icon_path.exists():
        from PyQt5.QtGui import QIcon

        app.setWindowIcon(QIcon(str(_icon_path)))

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

    # ── Splash screen ─────────────────────────────────────────────────────────
    splash = None
    try:
        from Aurik910.ui.splash_screen import AurikSplashScreen

        splash = AurikSplashScreen()
        splash.setWindowOpacity(0.0)
        splash.show()
        app.processEvents()

        # Fade in: 32 steps × 18 ms ≈ 580 ms
        for _step in range(32):
            splash.setWindowOpacity((_step + 1) / 32.0)
            app.processEvents()
            time.sleep(0.018)

        _process_events_ms(app, 120)  # hold visible for a moment

    except Exception as _exc:
        # Splash must never block the application from starting
        logger.warning("Splash konnte nicht geladen werden (non-fatal): %s", _exc)
        splash = None

    # ── Startup model check ───────────────────────────────────────────────────
    if splash:
        splash.set_status("Modelle werden geprüft...")
        app.processEvents()

    _run_startup_model_check(app)

    # ── Build main window ─────────────────────────────────────────────────────
    if splash:
        splash.set_status("Benutzeroberfläche wird aufgebaut...")
        app.processEvents()

    window = ModernMainWindow()

    if splash:
        splash.set_status("Bereit.")
        app.processEvents()
        _process_events_ms(app, 280)  # show "Bereit." briefly

    # Show main window beneath splash, then fade out splash
    window.show()
    app.processEvents()

    if splash:
        # Fade out: 28 steps × 20 ms ≈ 560 ms
        # The main window becomes visible as splash becomes transparent.
        splash.stop_animation()
        for _step in range(28):
            splash.setWindowOpacity(1.0 - (_step + 1) / 28.0)
            app.processEvents()
            time.sleep(0.020)
        splash.close()
        splash = None

    # §9.7.4 Modell-Warmup: ModernMainWindow.__init__ startet via QTimer.singleShot(2000)
    # warmup_models_background() aus backend.api.bridge — kein zweiter Thread hier.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
