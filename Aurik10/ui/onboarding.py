"""
Aurik10/ui/onboarding.py — Erststart-Assistent für Aurik 9.

Zeigt beim ersten Start einen freundlichen 3-Schritt-Wizard der erklärt,
was Aurik macht und wie man es benutzt. Speichert den Status via QSettings,
damit der Wizard nur einmal erscheint.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from Aurik10.i18n import t


class OnboardingWizard(QtWidgets.QDialog):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(t("app.name"))
        self.setMinimumSize(560, 420)
        self.resize(600, 460)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.CustomizeWindowHint
            | QtCore.Qt.WindowType.WindowTitleHint
        )

        self._current_page = 0
        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked pages
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._page_welcome())
        self._stack.addWidget(self._page_how())
        self._stack.addWidget(self._page_ready())
        layout.addWidget(self._stack, 1)

        # Navigation bar
        nav = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav)
        nav_layout.setContentsMargins(20, 12, 20, 16)

        # Dot indicators
        self._dots: list[QtWidgets.QLabel] = []
        dots_widget = QtWidgets.QWidget()
        dots_layout = QtWidgets.QHBoxLayout(dots_widget)
        dots_layout.setSpacing(8)
        for _ in range(3):
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet("font-size: 14px; color: #555;")
            dots_layout.addWidget(dot)
            self._dots.append(dot)
        nav_layout.addWidget(dots_widget)

        nav_layout.addStretch()

        self._btn_back = QtWidgets.QPushButton(t("onboarding.back") if False else "← Zurück")
        self._btn_back.setVisible(False)
        self._btn_back.clicked.connect(self._prev_page)
        nav_layout.addWidget(self._btn_back)

        self._btn_next = QtWidgets.QPushButton(t("onboarding.next") if False else "Weiter →")
        self._btn_next.clicked.connect(self._next_page)
        nav_layout.addWidget(self._btn_next)

        layout.addWidget(nav)
        self._update_dots()

    def _page_welcome(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel("🎵 Willkommen bei Aurik!")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Deine Musik verdient den besten Klang.")
        subtitle.setStyleSheet("font-size: 14px; color: #8894A8; margin-top: 8px;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        intro = QtWidgets.QLabel(
            "Aurik entfernt Knackser, Rauschen und andere Störungen\n"
            "aus deinen Aufnahmen – vollautomatisch, direkt auf deinem\n"
            "Computer, ohne Internet."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; line-height: 1.6;")
        layout.addWidget(intro)

        layout.addSpacing(20)

        # Features grid
        features = [
            ("🔇", "Entfernt Knackser & Rauschen"),
            ("🎚️", "Automatisch – kein Fachwissen nötig"),
            ("🔒", "100% offline – deine Daten bleiben privat"),
            ("💿", "Für Vinyl, Kassette, CD & MP3"),
        ]

        for icon, text in features:
            row = QtWidgets.QHBoxLayout()
            icon_label = QtWidgets.QLabel(icon)
            icon_label.setStyleSheet("font-size: 20px;")
            row.addWidget(icon_label)
            text_label = QtWidgets.QLabel(text)
            text_label.setStyleSheet("font-size: 13px;")
            row.addWidget(text_label)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()
        return page

    def _page_how(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel("⚡ So einfach geht's")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(20)

        # Mode cards
        for mode_id, mode_name, desc, icon, color in [
            (
                "restoration",
                "💿 Restoration",
                "Behutsam · Original bleibt erhalten\nFür Vinyl, Kassetten & Archiv",
                "💿",
                "#FFB300",
            ),
            ("studio", "🎯 Studio 2026", "Modern · Klar & kraftvoll\nFür Spotify, YouTube & Handy", "🎯", "#00B0FF"),
        ]:
            card = QtWidgets.QFrame()
            card.setStyleSheet(
                f"background: rgba(14,18,36,0.8); border: 1px solid {color}; border-radius: 10px; padding: 12px;"
            )
            card_layout = QtWidgets.QVBoxLayout(card)
            card_title = QtWidgets.QLabel(f"{icon}  {mode_name}")
            card_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {color};")
            card_layout.addWidget(card_title)
            card_desc = QtWidgets.QLabel(desc.replace("\n", "<br>"))
            card_desc.setStyleSheet("font-size: 12px;")
            card_layout.addWidget(card_desc)
            layout.addWidget(card)

        layout.addSpacing(15)

        steps = QtWidgets.QLabel(
            "1. 📂  Datei öffnen oder hierher ziehen\n"
            "2. 🎯  Modus wählen (Restoration oder Studio 2026)\n"
            "3. ▶  Starten – Aurik macht den Rest!"
        )
        steps.setStyleSheet("font-size: 13px; line-height: 1.8;")
        layout.addWidget(steps)

        layout.addStretch()
        return page

    def _page_ready(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel("🚀 Bereit!")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(15)

        msg = QtWidgets.QLabel(
            "Alles klar! Du kannst jederzeit auf ❓  Hilfe klicken,\n"
            "wenn du Fragen hast.\n\n"
            "Viel Spaß mit deiner Musik! 🎵"
        )
        msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(msg)

        layout.addSpacing(20)

        self._chk_skip = QtWidgets.QCheckBox("Beim nächsten Start nicht mehr anzeigen")
        layout.addWidget(self._chk_skip)

        layout.addStretch()
        return page

    def _next_page(self):
        if self._current_page < 2:
            self._current_page += 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_dots()

            if self._current_page == 2:
                self._btn_next.setText("Fertig ✓")
                self._btn_next.clicked.disconnect()
                self._btn_next.clicked.connect(self._finish)
        else:
            self._finish()

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_dots()

    def _update_dots(self):
        for i, dot in enumerate(self._dots):
            color = "#667eea" if i == self._current_page else "#555"
            dot.setStyleSheet(f"font-size: 14px; color: {color};")
        self._btn_back.setVisible(self._current_page > 0)
        if self._current_page < 2:
            self._btn_next.setText("Weiter →")

    def _finish(self):
        if self._chk_skip.isChecked():
            from PyQt5.QtCore import QSettings

            s = QSettings("AURIK", "AURIK Professional")
            s.setValue("onboarding/shown", True)
        self.finished.emit()
        self.accept()

    def _apply_theme(self):
        self.setStyleSheet("""
            QDialog {
                background: #080a18;
                color: #c9d1d9;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #5b6fd4;
            }
            QCheckBox {
                color: #8894A8;
            }
        """)


def should_show_onboarding() -> bool:
    """Prüft, ob der Onboarding-Wizard angezeigt werden soll."""
    try:
        from PyQt5.QtCore import QSettings

        s = QSettings("AURIK", "AURIK Professional")
        return not s.value("onboarding/shown", False, type=bool)
    except Exception:
        logger.warning("onboarding.py::should_show_onboarding fallback", exc_info=True)
        return True
