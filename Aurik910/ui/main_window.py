"""
Main Window for AURIK Professional
Professional audio restoration interface
"""

from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import soundfile as sf
from backend.api.bridge import get_restorer_classes

from ..core.preset_manager import Preset, PresetManager
from ..core.queue_manager import QueueManager, QueueStatus
from .audio_player import AudioPlayer
from .preset_browser import PresetBrowserWidget
from .queue_widget import QueueWidget
from .waveform_widget import MatplotlibWaveformWidget as WaveformWidget  # Legacy-Widget (Matplotlib-basiert)


class ProcessingThread(QThread):
    """Background thread for audio processing"""

    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_file, output_file, settings):
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file
        self.settings = settings

    def run(self):
        """Process audio in background"""
        try:
            _, UnifiedRestorerV3 = get_restorer_classes()

            self.progress.emit(10)

            # Load audio
            audio, sr = sf.read(self.input_file)
            self.progress.emit(20)

            # Create restorer
            restorer = UnifiedRestorerV3()
            self.progress.emit(30)

            # Process
            result = restorer.restore(audio, sample_rate=sr)
            self.progress.emit(80)

            # Save
            sf.write(self.output_file, result.audio, sr)
            self.progress.emit(100)

            self.finished.emit(f"Processed: {Path(self.input_file).name}")

        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class BatchProcessingThread(QThread):
    """Background thread for batch queue processing"""

    item_started = pyqtSignal(str)  # item_id
    item_progress = pyqtSignal(str, int)  # item_id, progress
    item_finished = pyqtSignal(str)  # item_id
    item_error = pyqtSignal(str, str)  # item_id, error_message
    all_finished = pyqtSignal()

    def __init__(self, queue_manager: QueueManager):
        super().__init__()
        self.queue_manager = queue_manager
        self._stop_requested = False

    def run(self):
        """Process all items in queue"""
        _, UnifiedRestorerV3 = get_restorer_classes()

        while not self._stop_requested:
            # Get next item
            item = self.queue_manager.get_next_item()
            if item is None:
                break

            try:
                # Mark as processing
                self.queue_manager.update_item_status(item.id, QueueStatus.PROCESSING, 0)
                self.item_started.emit(item.id)

                # Load audio
                audio, sr = sf.read(item.input_file)
                self.item_progress.emit(item.id, 20)
                self.queue_manager.update_item_status(item.id, QueueStatus.PROCESSING, 20)

                # Create restorer
                restorer = UnifiedRestorerV3()
                self.item_progress.emit(item.id, 30)
                self.queue_manager.update_item_status(item.id, QueueStatus.PROCESSING, 30)

                # Process
                result = restorer.restore(audio, sample_rate=sr)
                self.item_progress.emit(item.id, 80)
                self.queue_manager.update_item_status(item.id, QueueStatus.PROCESSING, 80)

                # Save
                sf.write(item.output_file, result.audio, sr)
                self.item_progress.emit(item.id, 100)

                # Mark as completed
                self.queue_manager.update_item_status(item.id, QueueStatus.COMPLETED, 100)
                self.item_finished.emit(item.id)

            except Exception as e:
                error_msg = str(e)
                self.queue_manager.update_item_status(item.id, QueueStatus.FAILED, 0, error_msg)
                self.item_error.emit(item.id, error_msg)

        self.all_finished.emit()

    def stop(self):
        """Request stop"""
        self._stop_requested = True


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_file = None
        self.processing_thread = None
        self.batch_thread = None
        self.queue_manager = QueueManager()
        self.preset_manager = PresetManager()
        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("AURIK Professional - Audio Restoration")
        self.setGeometry(100, 100, 1600, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left panel: File list and controls
        left_panel = self.create_left_panel()

        # Middle panel: Waveform and settings
        middle_panel = self.create_right_panel()

        # Right panel: Queue
        right_panel = self.create_queue_panel()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left: 25%
        splitter.setStretchFactor(1, 2)  # Middle: 50%
        splitter.setStretchFactor(2, 1)  # Right: 25%

        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusBar.addPermanentWidget(self.progress_bar)

    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Title
        title = QLabel("<h2>🎚️ AURIK Professional</h2>")
        layout.addWidget(title)

        # File list
        file_group = QGroupBox("Audio Files")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_list)

        # File buttons
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Files")
        btn_add.clicked.connect(self.add_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.file_list.clear)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_clear)
        file_layout.addLayout(btn_layout)

        layout.addWidget(file_group)

        # Medium selection
        medium_group = QGroupBox("Medium Type")
        medium_layout = QVBoxLayout()
        medium_group.setLayout(medium_layout)

        self.medium_combo = QComboBox()
        self.medium_combo.addItems(["Vinyl", "Cassette Tape", "DAT", "CD", "MP3", "Shellac 78rpm", "Wire Recording"])
        medium_layout.addWidget(self.medium_combo)

        layout.addWidget(medium_group)

        # Processing mode
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Gentle (Preserve Character)",
                "Balanced (Recommended)",
                "Aggressive (Maximum Cleanup)",
                "Archive (Maximum Preservation)",
                "Mastering (Subtle Enhancement)",
            ]
        )
        self.mode_combo.setCurrentIndex(1)  # Balanced
        mode_layout.addWidget(self.mode_combo)

        layout.addWidget(mode_group)

        # Process button
        self.btn_process = QPushButton("▶ Process Now")
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_process.clicked.connect(self.process_audio)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)

        # Add to Queue button
        self.btn_add_queue = QPushButton("➕ Add to Queue")
        self.btn_add_queue.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.btn_add_queue.clicked.connect(self.add_to_queue)
        self.btn_add_queue.setEnabled(False)
        layout.addWidget(self.btn_add_queue)

        # Preset browser
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        preset_group.setLayout(preset_layout)

        self.preset_browser = PresetBrowserWidget(self.preset_manager)
        self.preset_browser.preset_applied.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_browser)

        layout.addWidget(preset_group)

        return panel

    def create_right_panel(self):
        """Create right settings panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Tabs for Waveform and Preview
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444;
                border-radius: 3px;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: #cccccc;
                padding: 8px 20px;
                border: 1px solid #444444;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #0078d4;
                color: white;
            }
            QTabBar::tab:hover {
                background: #3d3d3d;
            }
        """)

        # Waveform tab
        waveform_tab = QWidget()
        waveform_layout = QVBoxLayout()
        waveform_tab.setLayout(waveform_layout)

        self.waveform_widget = WaveformWidget()
        waveform_layout.addWidget(self.waveform_widget)

        tabs.addTab(waveform_tab, "📊 Waveform")

        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout()
        preview_tab.setLayout(preview_layout)

        self.audio_player = AudioPlayer()
        preview_layout.addWidget(self.audio_player)

        tabs.addTab(preview_tab, "🎧 Preview")

        layout.addWidget(tabs)

        # Musical Goals
        goals_group = QGroupBox("Musical Goals")
        goals_layout = QVBoxLayout()
        goals_group.setLayout(goals_layout)

        self.goal_sliders = {}
        goals = [
            ("Brillanz (Brilliance)", 0.87),
            ("Wärme (Warmth)", 0.82),
            ("Natürlichkeit (Naturalness)", 0.85),
            ("Authentizität (Authenticity)", 0.88),
            ("Emotionalität (Emotion)", 0.83),
            ("Transparenz (Clarity)", 0.89),
            ("Bass-Kraft (Bass Power)", 0.75),
        ]

        for goal_name, default_val in goals:
            slider_layout = QHBoxLayout()
            label = QLabel(goal_name)
            label.setMinimumWidth(200)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(default_val * 100))
            value_label = QLabel(f"{int(default_val * 100)}%")
            value_label.setMinimumWidth(40)

            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}%"))

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)

            goals_layout.addLayout(slider_layout)
            self.goal_sliders[goal_name.split()[0]] = slider

        layout.addWidget(goals_group)

        # Phase 2.3 Enhancement
        enhancement_group = QGroupBox("Instrumental Enhancement (Phase 2.3)")
        enhancement_layout = QVBoxLayout()
        enhancement_group.setLayout(enhancement_layout)

        enhance_grid = QHBoxLayout()

        self.enhancement_checks = {}
        enhancements = ["Bass", "Drums", "Guitar", "Piano", "Brass", "Spatial"]

        for enhancement in enhancements:
            checkbox = QCheckBox(enhancement)
            enhance_grid.addWidget(checkbox)
            self.enhancement_checks[enhancement] = checkbox

        enhancement_layout.addLayout(enhance_grid)
        layout.addWidget(enhancement_group)

        layout.addStretch()

        return panel

    def add_files(self):
        """Add audio files to process"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac *.aiff);;All Files (*)"
        )

        for file in files:
            self.file_list.addItem(file)

        if self.file_list.count() > 0:
            self.btn_process.setEnabled(True)
            self.btn_add_queue.setEnabled(True)

    def on_file_selected(self):
        """Handle file selection"""
        items = self.file_list.selectedItems()
        if items:
            self.current_file = items[0].text()
            self.load_waveform()

    def load_waveform(self):
        """Load and display waveform with progress bar"""
        try:

            # Premium-Look für den Ladebalken
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #444;
                    border-radius: 10px;
                    text-align: center;
                    font: bold 14px "Segoe UI", "Arial";
                    color: #fff;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #232526, stop:1 #414345);
                }
                QProgressBar::chunk {
                    border-radius: 8px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #00c6ff, stop:0.5 #0072ff, stop:1 #005bea);
                    box-shadow: 0px 2px 8px #0072ff88;
                }
            """)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("<b>Lade Datei...</b> %p%")
            self.progress_bar.show()
            self.statusBar.showMessage("Lade Datei...")

            # Fortschritts-Signal verbinden
            self.waveform_widget.progress.connect(self.update_progress)
            # Start Ladeprozess (sendet 10, 40, 70, 90, 100)
            success_waveform = self.waveform_widget.load_audio(self.current_file)
            # Player laden (90%)
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            success_player = self.audio_player.load_audio(self.current_file)
            # Abschluss
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            self.progress_bar.hide()
            self.statusBar.showMessage("Bereit")

            if not success_waveform or not success_player:
                QMessageBox.warning(self, "Error", f"Failed to load audio file: {self.current_file}")

            # Fortschritts-Signal trennen, um Mehrfachverbindungen zu vermeiden
            self.waveform_widget.progress.disconnect(self.update_progress)

        except Exception as e:
            self.progress_bar.hide()
            self.statusBar.showMessage("Fehler beim Laden der Wellenform")
            QMessageBox.critical(self, "Error", f"Error loading waveform:\n{str(e)}")

    def process_audio(self):
        """Process selected audio files"""
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "Already processing a file!")
            return

        items = self.file_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select a file to process")
            return

        input_file = items[0].text()

        # Get output file
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Audio",
            str(Path(input_file).stem) + "_restored.wav",
            "WAV Files (*.wav);;All Files (*)",
        )

        if not output_file:
            return

        # Get settings
        medium_map = {
            "Vinyl": "VINYL",
            "Cassette Tape": "CASSETTE_TAPE",
            "DAT": "DAT",
            "CD": "CD",
            "MP3": "MP3",
            "Shellac 78rpm": "SHELLAC_78RPM",
            "Wire Recording": "WIRE_RECORDING",
        }

        mode_map = {
            "Gentle (Preserve Character)": "GENTLE",
            "Balanced (Recommended)": "BALANCED",
            "Aggressive (Maximum Cleanup)": "AGGRESSIVE",
            "Archive (Maximum Preservation)": "ARCHIVE",
            "Mastering (Subtle Enhancement)": "MASTERING",
        }

        settings = {
            "medium_type": medium_map[self.medium_combo.currentText()],
            "processing_mode": mode_map[self.mode_combo.currentText()],
        }

        # Start processing
        self.processing_thread = ProcessingThread(input_file, output_file, settings)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)

        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.btn_process.setEnabled(False)
        self.statusBar.showMessage("Processing...")

        self.processing_thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def on_processing_finished(self, message):
        """Handle processing completion"""
        self.progress_bar.hide()
        self.btn_process.setEnabled(True)
        self.statusBar.showMessage("Ready")
        QMessageBox.information(self, "Success", message)

    def on_processing_error(self, error_message):
        """Handle processing error"""
        self.progress_bar.hide()
        self.btn_process.setEnabled(True)
        self.statusBar.showMessage("Error")
        QMessageBox.critical(self, "Processing Error", error_message)

    def create_queue_panel(self):
        """Create queue management panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Queue widget
        self.queue_widget = QueueWidget(self.queue_manager)
        self.queue_widget.process_queue_requested.connect(self.process_queue)
        self.queue_widget.clear_queue_requested.connect(self.clear_queue)
        self.queue_widget.remove_item_requested.connect(self.remove_queue_item)
        layout.addWidget(self.queue_widget)

        return panel

    def add_to_queue(self):
        """Add selected files to processing queue"""
        # Get selected files or all files
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            # Add all files
            files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        else:
            files = [item.text() for item in selected_items]

        if not files:
            QMessageBox.warning(self, "No Files", "No files selected to add to queue.")
            return

        # Get current settings
        from backend.api.bridge import get_medium_type_enum, get_processing_mode_enum

        MediumType = get_medium_type_enum()
        ProcessingMode = get_processing_mode_enum()

        medium_map = {
            "Vinyl": MediumType.VINYL,
            "Cassette Tape": MediumType.CASSETTE,
            "DAT": MediumType.DAT,
            "CD": MediumType.CD,
            "MP3": MediumType.MP3,
            "Shellac 78rpm": MediumType.SHELLAC,
            "Wire Recording": MediumType.WIRE,
        }

        mode_map = {
            "Gentle (Preserve Character)": ProcessingMode.GENTLE,
            "Balanced (Recommended)": ProcessingMode.BALANCED,
            "Aggressive (Maximum Cleanup)": ProcessingMode.AGGRESSIVE,
            "Archive (Maximum Preservation)": ProcessingMode.ARCHIVE,
            "Mastering (Subtle Enhancement)": ProcessingMode.MASTERING,
        }

        settings = {
            "medium_type": medium_map[self.medium_combo.currentText()],
            "processing_mode": mode_map[self.mode_combo.currentText()],
        }

        # Choose output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(Path.home()))

        if not output_dir:
            return

        # Add files to queue
        added_count = 0
        for input_file in files:
            filename = Path(input_file).stem
            ext = Path(input_file).suffix
            output_file = str(Path(output_dir) / f"{filename}_restored{ext}")

            item = self.queue_manager.add_item(input_file, output_file, settings)
            self.queue_widget.add_item(item)
            added_count += 1

        self.statusBar.showMessage(f"Added {added_count} file(s) to queue", 3000)

    def process_queue(self):
        """Start batch processing of queue"""
        if self.batch_thread and self.batch_thread.isRunning():
            QMessageBox.warning(self, "Processing", "Queue is already being processed!")
            return

        stats = self.queue_manager.get_queue_stats()
        if stats["pending"] == 0:
            QMessageBox.information(self, "Queue Empty", "No pending items in queue.")
            return

        # Start batch processing
        self.batch_thread = BatchProcessingThread(self.queue_manager)
        self.batch_thread.item_started.connect(self.on_queue_item_started)
        self.batch_thread.item_progress.connect(self.on_queue_item_progress)
        self.batch_thread.item_finished.connect(self.on_queue_item_finished)
        self.batch_thread.item_error.connect(self.on_queue_item_error)
        self.batch_thread.all_finished.connect(self.on_queue_all_finished)

        self.statusBar.showMessage("Processing queue...")
        self.batch_thread.start()

    def on_queue_item_started(self, item_id):
        """Handle queue item start"""
        self.queue_widget.update_item(item_id, status=QueueStatus.PROCESSING)
        item = self.queue_manager.get_item_by_id(item_id)
        if item:
            self.statusBar.showMessage(f"Processing: {Path(item.input_file).name}")

    def on_queue_item_progress(self, item_id, progress):
        """Handle queue item progress"""
        self.queue_widget.update_item(item_id, progress=progress)

    def on_queue_item_finished(self, item_id):
        """Handle queue item completion"""
        self.queue_widget.update_item(item_id, status=QueueStatus.COMPLETED, progress=100)

    def on_queue_item_error(self, item_id, error_message):
        """Handle queue item error"""
        self.queue_widget.update_item(item_id, status=QueueStatus.FAILED)
        # Don't show error dialog during batch processing

    def on_queue_all_finished(self):
        """Handle queue completion"""
        stats = self.queue_manager.get_queue_stats()
        self.statusBar.showMessage(f"Queue complete: {stats['completed']} succeeded, {stats['failed']} failed", 5000)

        QMessageBox.information(
            self,
            "Queue Complete",
            f"Batch processing complete!\n\n" f"✅ Completed: {stats['completed']}\n" f"❌ Failed: {stats['failed']}",
        )

    def clear_queue(self):
        """Clear completed items from queue"""
        self.queue_manager.clear_queue(clear_completed=False)
        self.queue_widget.refresh_display()
        self.statusBar.showMessage("Cleared completed items", 2000)

    def remove_queue_item(self, item_id):
        """Remove item from queue"""
        if self.queue_manager.remove_item(item_id):
            self.queue_widget.remove_item(item_id)
            self.statusBar.showMessage("Removed item from queue", 2000)
        else:
            QMessageBox.warning(self, "Cannot Remove", "Cannot remove item that is currently processing.")

    def apply_preset(self, preset: Preset):
        """Apply preset to UI settings"""

        # Map medium type
        medium_map_reverse = {"VINYL": 0, "CASSETTE": 1, "DAT": 2, "CD": 3, "MP3": 4, "SHELLAC": 5, "WIRE": 6}

        # Map processing mode
        mode_map_reverse = {"GENTLE": 0, "BALANCED": 1, "AGGRESSIVE": 2, "ARCHIVE": 3, "MASTERING": 4}

        # Set medium type
        if preset.medium_type in medium_map_reverse:
            self.medium_combo.setCurrentIndex(medium_map_reverse[preset.medium_type])

        # Set processing mode
        if preset.processing_mode in mode_map_reverse:
            self.mode_combo.setCurrentIndex(mode_map_reverse[preset.processing_mode])

        # Set musical goals
        for goal_name, slider in self.goal_sliders.items():
            if goal_name in preset.musical_goals:
                value = int(preset.musical_goals[goal_name] * 100)
                slider.setValue(value)

        # Set enhancements
        for enhancement_name, checkbox in self.enhancement_checks.items():
            if enhancement_name in preset.enhancements:
                checkbox.setChecked(preset.enhancements[enhancement_name])

        # Show message
        self.statusBar.showMessage(f"Applied preset: {preset.name}", 3000)
        QMessageBox.information(
            self, "Preset Applied", f"Preset '{preset.name}' has been applied.\n\n{preset.description}"
        )

    def apply_dark_theme(self):
        """Apply dark color scheme"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.black)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        self.setPalette(palette)
