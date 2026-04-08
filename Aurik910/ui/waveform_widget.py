"""
Waveform Display Widget using Matplotlib
Real-time audio waveform visualization
"""

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from backend.file_import import load_audio_file

from ..i18n import t


class WaveformCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for waveform display"""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        # Create figure with dark background
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="#1e1e1e")
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        # Configure axes for dark theme
        self.axes.set_facecolor("#0d0d0d")
        self.axes.tick_params(colors="#888888", which="both")
        self.axes.spines["bottom"].set_color("#444444")
        self.axes.spines["top"].set_color("#444444")
        self.axes.spines["left"].set_color("#444444")
        self.axes.spines["right"].set_color("#444444")

        # Labels
        self.axes.set_xlabel(t("legacy.waveform.time"), color="#cccccc", fontsize=9)
        self.axes.set_ylabel(t("legacy.waveform.amplitude"), color="#cccccc", fontsize=9)
        self.axes.grid(True, color="#333333", linestyle="--", linewidth=0.5, alpha=0.3)

        self.fig.tight_layout()

        # Store audio data
        self.audio = None
        self.sr = None

    def plot_waveform(self, audio, sr, title=None):
        """
        Plot audio waveform

        Parameters
        ----------
        audio : np.ndarray
            Audio data (mono or stereo)
        sr : int
            Sample rate
        title : str
            Plot title
        """
        self.audio = audio
        self.sr = sr

        # Clear previous plot
        self.axes.clear()

        # Handle stereo/mono
        if audio.ndim == 2:
            # Stereo - plot both channels
            time = np.linspace(0, len(audio) / sr, len(audio))

            # Downsample for performance if needed
            if len(audio) > 50000:
                # Plot every nth sample
                step = len(audio) // 50000
                time_ds = time[::step]
                audio_ds = audio[::step]
            else:
                time_ds = time
                audio_ds = audio

            # Plot channels
            self.axes.plot(time_ds, audio_ds[:, 0], color="#00ff88", linewidth=0.5, alpha=0.8, label="Left")
            self.axes.plot(time_ds, audio_ds[:, 1], color="#0088ff", linewidth=0.5, alpha=0.8, label="Right")
            self.axes.legend(
                loc="upper right", fontsize=8, facecolor="#1e1e1e", edgecolor="#444444", labelcolor="#cccccc"
            )
        else:
            # Mono
            time = np.linspace(0, len(audio) / sr, len(audio))

            # Downsample for performance
            if len(audio) > 50000:
                step = len(audio) // 50000
                time_ds = time[::step]
                audio_ds = audio[::step]
            else:
                time_ds = time
                audio_ds = audio

            self.axes.plot(time_ds, audio_ds, color="#00ff88", linewidth=0.5, alpha=0.9)

        # Configure axes
        self.axes.set_xlim(0, len(audio) / sr)
        self.axes.set_ylim(-1.1, 1.1)
        self.axes.set_xlabel(t("legacy.waveform.time"), color="#cccccc", fontsize=9)
        self.axes.set_ylabel(t("legacy.waveform.amplitude"), color="#cccccc", fontsize=9)
        if title is None:
            title = t("legacy.waveform.audio_waveform")
        self.axes.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=10)
        self.axes.grid(True, color="#333333", linestyle="--", linewidth=0.5, alpha=0.3)

        # Style
        self.axes.set_facecolor("#0d0d0d")
        self.axes.tick_params(colors="#888888", which="both")
        for spine in self.axes.spines.values():
            spine.set_color("#444444")

        self.fig.tight_layout()
        self.draw()

    def plot_spectrogram(self, audio, sr, title=None):
        """
        Plot spectrogram

        Parameters
        ----------
        audio : np.ndarray
            Audio data (mono or stereo)
        sr : int
            Sample rate
        title : str
            Plot title
        """
        # Clear previous plot
        self.axes.clear()

        # Convert to mono if stereo
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # Create spectrogram
        self.axes.specgram(audio_mono, Fs=sr, cmap="viridis", scale="dB", mode="magnitude")

        # Configure
        self.axes.set_xlabel(t("legacy.waveform.time"), color="#cccccc", fontsize=9)
        self.axes.set_ylabel(t("legacy.waveform.frequency"), color="#cccccc", fontsize=9)
        if title is None:
            title = t("legacy.waveform.spectrogram")
        self.axes.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=10)

        # Style
        self.axes.set_facecolor("#0d0d0d")
        self.axes.tick_params(colors="#888888", which="both")
        for spine in self.axes.spines.values():
            spine.set_color("#444444")

        self.fig.tight_layout()
        self.draw()

    def clear_plot(self):
        """Clear the plot"""
        self.axes.clear()
        self.axes.text(
            0.5,
            0.5,
            t("legacy.waveform.load_to_see"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.axes.transAxes,
            color="#888888",
            fontsize=12,
        )
        self.axes.set_facecolor("#0d0d0d")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_color("#444444")

        self.fig.tight_layout()
        self.draw()


class MatplotlibWaveformWidget(QWidget):
    """Matplotlib-basiertes Waveform-Widget (Legacy, genutzt von main_window.py).

    Das feature-reiche WaveformWidget mit Defekt-Overlay, Lyrics-Timeline und
    Custom-Painter ist in modern_window.py definiert.
    """

    progress = pyqtSignal(int)  # Fortschrittssignal (0-100)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Matplotlib canvas
        self.canvas = WaveformCanvas(self, width=8, height=4, dpi=100)
        layout.addWidget(self.canvas)

        # Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                spacing: 3px;
            }
            QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                color: #cccccc;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #666666;
            }
            QToolButton:pressed {
                background-color: #2d2d2d;
            }
        """)
        layout.addWidget(self.toolbar)

        # View buttons
        btn_layout = QHBoxLayout()

        self.btn_waveform = QPushButton(t("legacy.waveform.waveform_button"))
        self.btn_waveform.clicked.connect(self.show_waveform)
        self.btn_waveform.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)

        self.btn_spectrogram = QPushButton(t("legacy.waveform.spectrogram_button"))
        self.btn_spectrogram.clicked.connect(self.show_spectrogram)
        self.btn_spectrogram.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)

        btn_layout.addWidget(self.btn_waveform)
        btn_layout.addWidget(self.btn_spectrogram)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Initial empty plot
        self.canvas.clear_plot()

    def load_audio(self, filepath):
        """
        Load and display audio file, send progress in 1%-Schritten
        """
        try:
            self.progress.emit(10)
            _loaded = load_audio_file(filepath, do_carrier_analysis=False)
            if _loaded is None or _loaded.get("error"):
                raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {filepath}")
            audio, sr = _loaded["audio"], int(_loaded["sr"])
            self.progress.emit(40)
            # Simulierte Schrittweite für große Dateien (optional sleep für Demo)
            # import time; time.sleep(0.1)
            self.canvas.plot_waveform(audio, sr, title=t("legacy.waveform.waveform_file", file=filepath.split("/")[-1]))
            self.progress.emit(70)
            # Noch ein Schritt für "Player laden" (wird im MainWindow gemacht)
            self.progress.emit(90)
            self.progress.emit(100)
            return True
        except Exception as e:
            import logging

            logging.error(f"Error loading audio: {e}")
            self.progress.emit(0)
            return False

    def show_waveform(self):
        """Show waveform view"""
        if self.canvas.audio is not None:
            self.canvas.plot_waveform(self.canvas.audio, self.canvas.sr, title=t("legacy.waveform.audio_waveform"))

            # Update button styles
            self.btn_waveform.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """)
            self.btn_spectrogram.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4d4d4d;
                }
            """)

    def show_spectrogram(self):
        """Show spectrogram view"""
        if self.canvas.audio is not None:
            self.canvas.plot_spectrogram(self.canvas.audio, self.canvas.sr, title=t("legacy.waveform.spectrogram"))

            # Update button styles
            self.btn_waveform.setStyleSheet("""
                QPushButton {
                    background-color: #3d3d3d;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #4d4d4d;
                }
            """)
            self.btn_spectrogram.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
            """)

    def clear(self):
        """Clear the display"""
        self.canvas.clear_plot()
