#!/usr/bin/env python3
"""
AURIK Professional - Main Application Entry Point
Launch the desktop application for audio restoration
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# ── OpenBLAS/OMP thread-safety: must be set BEFORE any numpy/scipy import ────
# When multiple Python threads call into numpy simultaneously (pre-analysis
# ThreadPoolExecutor, pipeline workers), OpenBLAS spawning its own threads
# causes race conditions and segfaults (faulthandler confirmed: numpy
# _wrapreduction thread + psychoacoustic_masking_model seg-fault pattern).
# Capping OpenBLAS/OMP to 1 internal thread makes each numpy call single-
# threaded but prevents the OpenBLAS-vs-Python-thread race. PyTorch uses its
# own thread pool (set via torch.set_num_threads) and is unaffected.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ── PyTorch global thread-pool — must be set once at startup (§VERBOTEN) ─────
try:
    import torch as _torch

    _torch.set_num_threads(os.cpu_count() or 4)
except Exception:  # torch not installed / import error on first run
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Logging-Setup: File + Konsole ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / "aurik_backend.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

# AURIK_DEBUG=1 → alle Handler auf DEBUG, dediziertes Timestamp-Log, volle Quelle
_DEBUG_MODE = os.getenv("AURIK_DEBUG", "0") == "1"

# Datei-Handler (5 MB, Rotation)
from logging.handlers import RotatingFileHandler as _RFH

_fh = _RFH(str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setLevel(logging.DEBUG if _DEBUG_MODE else logging.INFO)
_fh.setFormatter(
    logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if _DEBUG_MODE
    else logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
)
_root_logger.addHandler(_fh)

# Debug-Session-Log: eigene Datei mit Timestamp — nie rotiert, immer vollständig
if _DEBUG_MODE:
    import datetime as _dt

    _debug_log_file = _LOG_DIR / f"aurik_debug_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _dfh = logging.FileHandler(str(_debug_log_file), mode="w", encoding="utf-8")
    _dfh.setLevel(logging.DEBUG)
    _dfh.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _root_logger.addHandler(_dfh)
    # Pfad für shell-seitiges tail -f: in bekannter Datei hinterlegen
    (_LOG_DIR / "aurik_debug_latest.log").write_text(str(_debug_log_file), encoding="utf-8")

    # ── Noisy 3rd-party loggers: auf WARNING kappen — kein DSP/Aurik-Nutzen ─
    # Numba JIT-Compiler gibt bei jeder Erstcompilierung hunderte DEBUG-Zeilen aus.
    # PIL/fonttools/matplotlib werden durch Qt-Ressourcen gelegentlich importiert.
    # triton/torch.fx sind reine Framework-Internals ohne Diagnosewert für Aurik.
    for _noisy_logger in (
        "numba",
        "numba.core",
        "numba.core.byteflow",
        "numba.core.interpreter",
        "numba.core.ssa",
        "numba.core.analysis",
        "numba.typed",
        "PIL",
        "fonttools",
        "matplotlib",
        "matplotlib.font_manager",
        "triton",
        "torch.fx",
        "torch._dynamo",
        "urllib3",
        "h5py",
        "audioread",
        "librosa",
        "librosa.core",
        "pedalboard",
        "backend.core.erb_auditory_masking",
        "dsp.pghi",
    ):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# Konsole-Handler (WARNING+ im Normalbetrieb; DEBUG im Debug-Modus → live sichtbar)
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.DEBUG if _DEBUG_MODE else logging.WARNING)
_ch.setFormatter(
    logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if _DEBUG_MODE
    else logging.Formatter("%(levelname)s %(name)s: %(message)s")
)
_root_logger.addHandler(_ch)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _enable_crash_forensics() -> None:
    """Enable native-crash diagnostics for hard faults (SIGSEGV/SIGABRT)."""
    try:
        import faulthandler

        crash_log = _LOG_DIR / "python_faulthandler.log"
        _fh = open(crash_log, "a", encoding="utf-8")
        faulthandler.enable(file=_fh, all_threads=True)
        for _sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
            try:
                faulthandler.register(_sig, file=_fh, all_threads=True)
            except Exception:
                # Not all platforms allow explicit registration for every signal.
                pass
        logger.info("Crash forensics active (faulthandler): %s", crash_log)
    except Exception as exc:
        logger.warning("Faulthandler setup failed (non-fatal): %s", exc)


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from Aurik910 import __version__
from Aurik910.ui.modern_window import ModernMainWindow


def _run_startup_model_check(app: QApplication) -> None:
    """Prüft ML-Modelle vor dem Fensteraufbau — zeigt deutschen Dialog bei Problemen.

    Nicht-blockierend bei fehlenden optionalen Modellen.
    Warnung bei fehlenden Primär-Modellen (DSP-Fallback aktiv).
    """
    try:
        import time

        from backend.api.bridge import get_model_downloader, get_startup_check_result

        def run_self_heal(missing_primary, missing_optional):
            dl = get_model_downloader()
            to_repair = [m["name"] for m in (missing_primary + missing_optional)]
            for i, name in enumerate(to_repair):
                entry = dl.get_entry(name)
                if entry is not None:
                    # Download SOTA-Upgrade falls konfiguriert, sonst bundled
                    dl.schedule_sota_upgrade(entry)
                # Fortschritt anzeigen (optional: Splash-Text, hier nur Sleep)
                time.sleep(0.2)
            # Warten bis alle Downloads abgeschlossen (vereinfachte Variante)
            time.sleep(2.0)

        result = get_startup_check_result()
        if result is None:
            return
        if not result.all_ok and result.user_message_de:
            icon = QMessageBox.Icon.Critical if result.is_critical else QMessageBox.Icon.Warning
            box = QMessageBox()
            box.setWindowTitle("AURIK — " + result.user_title_de)
            box.setText(result.user_message_de)
            box.setIcon(icon)
            # Zusätzlicher Button für Selbstheilung
            repair_btn = box.addButton("Modelle reparieren", QMessageBox.ActionRole)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            box.exec_()
            if box.clickedButton() == repair_btn:
                # Automatische Selbstheilung starten
                run_self_heal(result.missing_primary, result.missing_optional)
                # Nach Abschluss Check wiederholen
                result2 = get_startup_check_result()
                if result2 is not None and result2.all_ok:
                    QMessageBox.information(
                        None,
                        "AURIK — Modelle repariert",
                        "Alle ML-Modelle wurden erfolgreich repariert und geladen. Die Restaurierung ist jetzt freigeschaltet.",
                    )
                else:
                    QMessageBox.critical(
                        None,
                        "AURIK — Reparatur fehlgeschlagen",
                        "Einige Modelle konnten nicht repariert werden. Bitte prüfen Sie Ihre Internetverbindung oder wenden Sie sich an den Support.",
                    )
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


# ---------------------------------------------------------------------------
# §3.9.2  SIGTERM handler — checkpoint + graceful Qt shutdown
# ---------------------------------------------------------------------------


def _emergency_checkpoint_if_running() -> None:
    """Non-blocking best-effort checkpoint on SIGTERM (§3.9.2).

    Only fires when a BatchProcessingThread is currently running.
    Uses threading.Event.wait(timeout=0) to avoid blocking the signal handler.
    Writes checkpoint atomically (.tmp → os.replace) if audio is available.
    """
    try:
        # Import lazily to avoid circular imports at module level
        from PyQt5.QtWidgets import QApplication

        from Aurik910.ui.modern_window import ModernMainWindow  # type: ignore[import]

        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if isinstance(widget, ModernMainWindow):
                bt = getattr(widget, "batch_thread", None)
                if bt is not None and bt.isRunning():
                    try:
                        # best-effort: trigger checkpoint without waiting
                        save_fn = getattr(bt, "request_emergency_checkpoint", None)
                        if callable(save_fn):
                            save_fn()
                    except Exception as _ce:
                        logger.debug("Emergency checkpoint failed (non-fatal): %s", _ce)
                break
    except Exception as _exc:
        logger.debug("_emergency_checkpoint_if_running skipped: %s", _exc)


def _sigterm_handler(signum: int, frame: object) -> None:
    """SIGTERM → emergency checkpoint + graceful Qt shutdown (§3.9.2)."""
    logger.warning("SIGTERM received (signum=%d) — initiating emergency checkpoint", signum)
    _emergency_checkpoint_if_running()
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        _app = QApplication.instance()
        if _app:
            QTimer.singleShot(0, _app.quit)
    except Exception as _exc:
        logger.debug("Qt-Shutdown nach SIGTERM fehlgeschlagen: %s", _exc)


def _process_events_ms(app: "QApplication", ms: int) -> None:
    """Process Qt events for *ms* milliseconds — keeps splash animation alive."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.016)  # ~60 fps


def main():
    """Launch AURIK Professional"""
    _enable_crash_forensics()

    # Linux hardening: prefer software OpenGL to reduce GPU driver related Qt crashes.
    # Can be disabled for troubleshooting via: AURIK_FORCE_SOFTWARE_OPENGL=0
    _force_sw_gl = os.getenv("AURIK_FORCE_SOFTWARE_OPENGL", "1") == "1"

    # Enable high DPI scaling (PyQt5-Stubs kennen diese Attribute nicht -> ignore)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore[attr-defined]
    if sys.platform.startswith("linux") and _force_sw_gl:
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)  # type: ignore[attr-defined]
    # Prevent XDG/GTK portal hang on Linux: native file dialogs must not be used
    # (portal daemon can block the main thread before DontUseNativeDialog takes effect)
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("AURIK Professional")
    app.setOrganizationName("AURIK")
    app.setApplicationVersion(__version__)

    # §3.9.2: Register SIGTERM handler for graceful shutdown + emergency checkpoint.
    # Must be installed after QApplication to avoid race with Qt's signal handling.
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Application icon (taskbar, dock, alt-tab)
    _res = Path(__file__).parent / "resources"
    for _icon_path in (_res / "vinyl_gold.png", _res / "icon_premium.svg", _res / "icon.png"):
        if _icon_path.exists():
            from PyQt5.QtGui import QIcon

            app.setWindowIcon(QIcon(str(_icon_path)))
            break

    # Set dark theme style
    app.setStyle("Fusion")

    # Capture Qt warnings/errors into Python logs for post-mortem analysis.
    try:
        from PyQt5.QtCore import qInstallMessageHandler

        def _qt_msg_handler(mode, context, message):
            _file = getattr(context, "file", "?") if context is not None else "?"
            _line = getattr(context, "line", 0) if context is not None else 0
            logger.warning("QtMessage[%s] %s:%s %s", int(mode), _file, _line, message)

        qInstallMessageHandler(_qt_msg_handler)
    except Exception as exc:
        logger.debug("Qt message handler setup skipped: %s", exc)

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
