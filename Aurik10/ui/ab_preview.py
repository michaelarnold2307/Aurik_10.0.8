"""
Aurik10/ui/ab_preview.py — A/B-Vorher/Nachher-Vorschau in der GUI.

Zeigt vor der vollen Restaurierung eine schnelle Vorschau
der ersten 30 Sekunden, damit der Nutzer hören kann,
was Aurik mit seiner Aufnahme macht.

Nutzt das vorhandene StreamingAudioPlayer-System für
gapless A/B-Umschaltung.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PyQt5 import QtCore, QtWidgets

from Aurik10.i18n import t

logger = logging.getLogger(__name__)


class ABPreviewWidget(QtWidgets.QWidget):
    """A/B-Vorher/Nachher-Vorschau-Widget."""

    restoration_requested = QtCore.pyqtSignal(str)  # mode name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original_audio: np.ndarray | None = None
        self._preview_audio: np.ndarray | None = None
        self._sr: int = 48000
        self._file_path: str = ""
        self._generating: bool = False

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QtWidgets.QLabel(t("ab_preview.title"))
        header.setStyleSheet("font-size: 15pt; font-weight: bold;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        desc = QtWidgets.QLabel(t("ab_preview.description"))
        desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # A/B Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(20)

        self.btn_a = QtWidgets.QPushButton(t("ab_preview.btn_original"))
        self.btn_a.setMinimumSize(200, 60)
        self.btn_a.setStyleSheet(self._button_style("#FFB300"))
        self.btn_a.clicked.connect(self._play_original)
        self.btn_a.setEnabled(False)
        btn_row.addWidget(self.btn_a)

        self.btn_b = QtWidgets.QPushButton(t("ab_preview.btn_preview"))
        self.btn_b.setMinimumSize(200, 60)
        self.btn_b.setStyleSheet(self._button_style("#4CAF50"))
        self.btn_b.clicked.connect(self._play_preview)
        self.btn_b.setEnabled(False)
        btn_row.addWidget(self.btn_b)

        layout.addLayout(btn_row)

        # Status
        self.status_label = QtWidgets.QLabel(t("ab_preview.loading"))
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Start button
        self.btn_start = QtWidgets.QPushButton(t("ab_preview.start_full"))
        self.btn_start.setMinimumHeight(45)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #5B9FE8);
                color: white; border-radius: 8px; font-size: 12pt;
                padding: 10px 30px; font-weight: bold;
            }
            QPushButton:hover { background: #7B9BEE; }
        """)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setEnabled(False)
        layout.addWidget(self.btn_start)

        layout.addStretch()

    @staticmethod
    def _button_style(color: str) -> str:
        return f"""
            QPushButton {{
                background: rgba(0,0,0,0.3);
                color: {color};
                border: 2px solid {color};
                border-radius: 12px;
                font-size: 14pt; font-weight: bold;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: rgba(0,0,0,0.5);
            }}
            QPushButton:disabled {{
                color: #666;
                border-color: #444;
            }}
        """

    def set_audio(self, audio: np.ndarray, sr: int, file_path: str):
        """Audio für Vorschau setzen."""
        self._original_audio = audio.astype(np.float32)
        self._sr = sr
        self._file_path = file_path
        self.btn_a.setEnabled(True)
        self.status_label.setText(t("ab_preview.generating"))
        self.progress.setVisible(True)
        self._generate_preview()

    def _generate_preview(self):
        """Erzeugt Vorschau in Hintergrund-Thread."""
        if self._original_audio is None:
            return
        self._generating = True

        def _work():
            try:
                preview_len = min(len(self._original_audio), int(30 * self._sr))
                segment = self._original_audio[:preview_len].copy()

                # Leichte, schnelle Vorverarbeitung
                try:
                    from backend.api.bridge import get_human_pleasantness_estimator as _hpe

                    compute_pleasantness = _hpe()
                    result = compute_pleasantness(segment, self._sr)
                    logger.info("Preview HPE: %.2f", result.score)
                except Exception:
                    logger.warning("ab_preview.py::_work fallback", exc_info=True)

                # Schnelle Vorverarbeitung: einfaches Gate + Soft-Knee
                try:
                    from backend.api.bridge import get_audio_utils_gain_envelope as _ge

                    apply_musical_gain_envelope = _ge()
                    segment = apply_musical_gain_envelope(segment, self._sr, gate_db=-30, knee_db=6, crossfade_ms=200)
                except Exception:
                    logger.warning("ab_preview.py::_work fallback", exc_info=True)

                self._preview_audio = segment.astype(np.float32)
            except Exception as e:
                logger.warning("Preview generation failed: %s", e)
                self._preview_audio = None
            finally:
                self._generating = False

        t = threading.Thread(target=_work, daemon=True)
        t.start()

        # Poll for completion
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._check_preview_ready)
        self._poll_timer.start(200)

    def _check_preview_ready(self):
        if self._generating:
            return
        self._poll_timer.stop()
        self.progress.setVisible(False)
        if self._preview_audio is not None:
            self.btn_b.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.status_label.setText(t("ab_preview.ready"))
        else:
            self.status_label.setText(t("ab_preview.unavailable"))

    def _play_original(self):
        if self._original_audio is None:
            return
        try:
            import sounddevice as sd

            sd.stop()
            sd.play(self._original_audio[: min(len(self._original_audio), int(30 * self._sr))], self._sr)
            self.btn_a.setStyleSheet(self._button_style("#FFD54F"))
            self.btn_b.setStyleSheet(self._button_style("#4CAF50"))
        except Exception as e:
            logger.warning("Playback error: %s", e)

    def _play_preview(self):
        if self._preview_audio is None:
            return
        try:
            import sounddevice as sd

            sd.stop()
            sd.play(self._preview_audio, self._sr)
            self.btn_b.setStyleSheet(self._button_style("#81C784"))
            self.btn_a.setStyleSheet(self._button_style("#FFB300"))
        except Exception as e:
            logger.warning("Playback error: %s", e)

    def _on_start(self):
        from PyQt5.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            t("ab_preview.mode_title"),
            t("ab_preview.mode_question"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.restoration_requested.emit("RESTORATION")
        else:
            self.restoration_requested.emit("STUDIO_2026")
