"""
Audio Preview Player Widget
Real-time audio playback with before/after comparison
"""

from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from backend.api.bridge import (
        get_aurik_denker_class,
        get_aurik_denker_instance,
    )
    from backend.api.bridge import (
        get_load_audio_fn as _bridge_get_load_audio_fn,
    )

    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

    def get_aurik_denker_class() -> Any:  # type: ignore[misc]
        return None

    def get_aurik_denker_instance() -> Any:  # type: ignore[misc]
        return None

    def _bridge_get_load_audio_fn() -> Any:  # type: ignore[misc]
        return None


# Direct fallback import for when bridge is unavailable
try:
    from backend.file_import import load_audio_file as _direct_load_audio_file
except ImportError:
    _direct_load_audio_file = None  # type: ignore[assignment]


from ..i18n import t

# Optional sounddevice import (requires PortAudio)
try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    SOUNDDEVICE_AVAILABLE = False
    import logging

    logging.warning(f"Warning: sounddevice not available ({e}). Audio playback disabled.")


class AudioPreviewThread(QThread):
    """Background thread for audio preview processing"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, int)
    error = pyqtSignal(str)

    def __init__(self, input_file: str, settings: dict):
        super().__init__()
        self.input_file = input_file
        self.settings = settings

    def run(self):
        """Process preview audio"""
        try:
            import numpy as np

            # §2.2 No-Competing-Instances: Singleton-Zugriff statt direkter Instanziierung
            denker = get_aurik_denker_instance()
            if denker is None:
                raise RuntimeError("AurikDenker nicht verfügbar — Backend-Import fehlgeschlagen.")

            self.progress.emit(10)

            # Load audio via bridge cascade (soundfile → pedalboard/FFmpeg → pydub)
            # VERBOTEN: sf.read() direkt (§Architektur-RELEASE_MUST)
            _load_fn = _bridge_get_load_audio_fn()
            if _load_fn is not None:
                _loaded = _load_fn(self.input_file, target_sr=48_000, do_carrier_analysis=False)
                if _loaded is None:
                    raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {self.input_file}")
                audio = np.asarray(_loaded["audio"], dtype=np.float32)
                sr = int(_loaded["sr"])
            else:
                # Bridge nicht verfügbar — load_audio_file direkt
                if _direct_load_audio_file is not None:
                    _loaded = _direct_load_audio_file(self.input_file, target_sr=48_000, do_carrier_analysis=False)
                    if _loaded is None or _loaded.get("error"):
                        raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {self.input_file}")
                    audio = np.asarray(_loaded["audio"], dtype=np.float32)
                    sr = int(_loaded["sr"])
                else:
                    raise RuntimeError("Kein Audio-Loader verfügbar (Bridge und file_import fehlen)")
            self.progress.emit(30)

            self.progress.emit(50)

            # Process (first 30 seconds for preview)
            max_samples = int(30 * sr)
            audio_preview = audio[:max_samples] if len(audio) > max_samples else audio

            # Mono sicherstellen
            if audio_preview.ndim > 1:
                audio_preview = np.mean(audio_preview, axis=1)
            audio_preview = audio_preview.astype(np.float32)

            result = denker.denke(audio_preview, sr, mode="quality")
            self.progress.emit(90)

            # V3 gibt RestorationResult zurück — .audio enthält das Audio-Array
            audio_out = result.audio if hasattr(result, "audio") else result
            self.finished.emit(audio_out, sr)
            self.progress.emit(100)

        except Exception as e:
            self.error.emit(t("legacy.audio.load_error", error=str(e)))


class AudioPlayer(QWidget):
    """Audio player widget with before/after comparison"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_before = None
        self.audio_after = None
        self.sr = None
        self.current_mode = "before"  # before, after, split
        self.is_playing = False
        self.current_position = 0
        self.stream = None
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.update_position)
        self.preview_thread = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(self.title_label)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Progress bar (for preview processing)
        self.preview_progress = QProgressBar()
        self.preview_progress.setMaximum(100)
        self.preview_progress.hide()
        self.preview_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
                color: #cccccc;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        layout.addWidget(self.preview_progress)

        # Mode selector
        mode_layout = QHBoxLayout()

        self.mode_label = QLabel()
        self.mode_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        mode_layout.addWidget(self.mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo.setEnabled(False)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 3px 8px;
            }
        """)
        mode_layout.addWidget(self.mode_combo, 1)

        layout.addLayout(mode_layout)

        # Seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(1000)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self.on_seek)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444444;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #005a9e;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """)
        layout.addWidget(self.seek_slider)

        # Time labels
        time_layout = QHBoxLayout()
        self.time_current = QLabel("0:00")
        self.time_current.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.time_total = QLabel("0:00")
        self.time_total.setStyleSheet("color: #cccccc; font-size: 10px;")
        time_layout.addWidget(self.time_current)
        time_layout.addStretch()
        time_layout.addWidget(self.time_total)
        layout.addLayout(time_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.btn_play = QPushButton()
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px 20px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton()
        self.btn_stop.clicked.connect(self.stop)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                color: white;
                padding: 8px 20px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #dd4444;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        controls_layout.addWidget(self.btn_stop)

        layout.addLayout(controls_layout)

        # Volume control
        volume_layout = QHBoxLayout()

        volume_label = QLabel("🔊")
        volume_label.setStyleSheet("font-size: 14px;")
        volume_layout.addWidget(volume_label)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(75)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444444;
                height: 4px;
                background: #1e1e1e;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00aa00;
                border: 1px solid #008800;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
        """)
        volume_layout.addWidget(self.volume_slider, 1)

        self.volume_label = QLabel("75%")
        self.volume_label.setStyleSheet("color: #cccccc; font-size: 10px; min-width: 35px;")
        self.volume_slider.valueChanged.connect(lambda v: self.volume_label.setText(f"{v}%"))
        volume_layout.addWidget(self.volume_label)

        layout.addLayout(volume_layout)

        # Generate Preview button
        self.btn_generate = QPushButton()
        self.btn_generate.clicked.connect(self.generate_preview)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                padding: 8px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        layout.addWidget(self.btn_generate)
        self._apply_i18n_texts()

    def _apply_i18n_texts(self):
        """Apply translated labels and button texts."""
        self.title_label.setText(f"<b>{t('legacy.audio.title')}</b>")
        if self.audio_before is None:
            self.status_label.setText(t("legacy.audio.load_to_preview"))

        self.mode_label.setText(t("legacy.audio.mode_label"))
        self.mode_combo.blockSignals(True)
        current = self.mode_combo.currentIndex()
        self.mode_combo.clear()
        self.mode_combo.addItem(t("legacy.audio.mode_before"), "before")
        self.mode_combo.addItem(t("legacy.audio.mode_after"), "after")
        self.mode_combo.addItem(t("legacy.audio.mode_split"), "split")
        self.mode_combo.setCurrentIndex(max(0, current))
        self.mode_combo.blockSignals(False)

        self.btn_play.setText(t("legacy.audio.pause_button") if self.is_playing else t("legacy.audio.play_button"))
        self.btn_stop.setText(t("legacy.audio.stop_button"))
        self.btn_generate.setText(t("legacy.audio.generate_preview"))

    def load_audio(self, filepath: str) -> bool:
        """Load audio file for preview"""
        try:
            self.stop()

            # Load via bridge cascade (soundfile → pedalboard/FFmpeg → pydub)
            _load_fn = _bridge_get_load_audio_fn()
            if _load_fn is not None:
                _loaded = _load_fn(filepath, do_carrier_analysis=False)
                if _loaded is None:
                    raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {filepath}")
                audio = np.asarray(_loaded["audio"], dtype=np.float32)
                sr = int(_loaded["sr"])
            else:
                # Bridge nicht verfügbar — load_audio_file direkt
                if _direct_load_audio_file is not None:
                    _loaded = _direct_load_audio_file(filepath, do_carrier_analysis=False)
                    if _loaded is None or _loaded.get("error"):
                        raise RuntimeError(f"Audio-Datei konnte nicht geladen werden: {filepath}")
                    audio = np.asarray(_loaded["audio"], dtype=np.float32)
                    sr = int(_loaded["sr"])
                else:
                    raise RuntimeError("Kein Audio-Loader verfügbar (Bridge und file_import fehlen)")
            self.audio_before = audio
            self.sr = sr
            self.audio_after = None
            self.current_position = 0

            # Update UI
            duration = len(audio) / sr
            self.time_total.setText(self.format_time(duration))
            self.status_label.setText(t("legacy.audio.loaded", file=Path(filepath).name))

            self.btn_play.setEnabled(True)
            self.btn_generate.setEnabled(True)
            self.seek_slider.setEnabled(True)
            self.mode_combo.setCurrentIndex(0)
            self.mode_combo.setEnabled(False)  # Enable after preview generated

            return True

        except Exception as e:
            self.status_label.setText(t("legacy.audio.load_error", error=str(e)))
            return False

    def generate_preview(self):
        """Generate processed preview"""
        if self.audio_before is None:
            return

        # This will be connected to get settings from main window
        self.status_label.setText(t("legacy.audio.generating_preview"))

        # For now, just duplicate the audio as a placeholder
        # In real implementation, this will call preview_thread with settings
        self.audio_after = self.audio_before.copy()
        self.mode_combo.setEnabled(True)
        self.status_label.setText(t("legacy.audio.preview_ready"))

    def toggle_play(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playback"""
        if not SOUNDDEVICE_AVAILABLE:
            self.status_label.setText(t("legacy.audio.playback_unavailable"))
            return

        if self.audio_before is None:
            return

        # Get audio based on mode
        audio = self.get_current_audio()
        if audio is None:
            return

        # Apply volume
        volume = self.volume_slider.value() / 100.0

        try:
            # Start from current position
            start_sample = int(self.current_position * len(audio) / 1000)
            audio_to_play = audio[start_sample:] * volume

            # Start playback
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=audio_to_play.shape[1] if audio_to_play.ndim > 1 else 1,
                callback=self.audio_callback,
            )
            self.stream.start()

            self.is_playing = True
            self.btn_play.setText(t("legacy.audio.pause_button"))
            self.btn_stop.setEnabled(True)
            self.play_timer.start(100)  # Update every 100ms

        except Exception as e:
            self.status_label.setText(t("legacy.audio.playback_error", error=str(e)))

    def pause(self):
        """Pause playback"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.is_playing = False
        self.btn_play.setText(t("legacy.audio.play_button"))
        self.play_timer.stop()

    def stop(self):
        """Stop playback"""
        self.pause()
        self.current_position = 0
        self.seek_slider.setValue(0)
        self.time_current.setText("0:00")
        self.btn_stop.setEnabled(False)

    def audio_callback(self, outdata, frames, time, status):
        """Audio stream callback (placeholder)"""
        # This is a simplified callback - full implementation would be more complex

    def update_position(self):
        """Update playback position"""
        if self.is_playing and self.audio_before is not None:
            # Estimate position (simplified)
            self.current_position += 10  # ~100ms

            if self.current_position >= 1000:
                self.stop()
                return

            self.seek_slider.setValue(self.current_position)

            # Update time
            audio = self.get_current_audio()
            if audio is not None:
                duration = len(audio) / self.sr
                current_time = (self.current_position / 1000.0) * duration
                self.time_current.setText(self.format_time(current_time))

    def on_seek(self, position):
        """Handle seek slider move"""
        self.current_position = position

        if self.audio_before is not None:
            audio = self.get_current_audio()
            if audio is not None:
                duration = len(audio) / self.sr
                current_time = (position / 1000.0) * duration
                self.time_current.setText(self.format_time(current_time))

    def on_mode_changed(self, index):
        """Handle mode change"""
        mode = self.mode_combo.itemData(index)
        self.current_mode = mode if mode in {"before", "after", "split"} else "before"

        # Restart playback if playing
        if self.is_playing:
            self.pause()
            self.play()

    def get_current_audio(self) -> np.ndarray | None:
        """Get audio based on current mode"""
        if self.current_mode == "before":
            return self.audio_before
        elif self.current_mode == "after":
            return self.audio_after if self.audio_after is not None else self.audio_before
        else:  # split
            # A/B split: alternates every 2 seconds
            # Simplified implementation
            return self.audio_before

    def format_time(self, seconds: float) -> str:
        """Format time as M:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def cleanup(self):
        """Cleanup audio resources"""
        self.stop()
        if self.stream:
            self.stream.close()
