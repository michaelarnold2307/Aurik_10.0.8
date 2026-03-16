"""
Queue Widget for Batch Processing
Visual queue management interface
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.queue_manager import QueueItem, QueueManager, QueueStatus


class QueueItemWidget(QWidget):
    """Widget for displaying a single queue item"""

    def __init__(self, item: QueueItem, parent=None):
        super().__init__(parent)
        self.item = item
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Filename
        filename = self.item.input_file.split("/")[-1]
        self.label_filename = QLabel(f"<b>{filename}</b>")
        self.label_filename.setStyleSheet("color: #ffffff;")
        layout.addWidget(self.label_filename)

        # Status + Progress
        status_layout = QHBoxLayout()

        self.label_status = QLabel(self.item.status.value.upper())
        self.update_status_color()
        status_layout.addWidget(self.label_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(self.item.progress)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet("""
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
        status_layout.addWidget(self.progress_bar, 1)

        layout.addLayout(status_layout)

        # Error message
        if self.item.error_message:
            self.label_error = QLabel(f"Error: {self.item.error_message}")
            self.label_error.setStyleSheet("color: #ff4444; font-size: 9px;")
            self.label_error.setWordWrap(True)
            layout.addWidget(self.label_error)

    def update_status_color(self):
        """Update status label color based on status"""
        colors = {
            QueueStatus.PENDING: "#888888",
            QueueStatus.PROCESSING: "#0078d4",
            QueueStatus.COMPLETED: "#00ff00",
            QueueStatus.FAILED: "#ff4444",
            QueueStatus.CANCELLED: "#ffaa00",
        }
        color = colors.get(self.item.status, "#888888")
        self.label_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 9px;")

    def update_progress(self, progress: int):
        """Update progress bar"""
        self.progress_bar.setValue(progress)

    def update_status(self, status: QueueStatus):
        """Update status display"""
        self.item.status = status
        self.label_status.setText(status.value.upper())
        self.update_status_color()


class QueueWidget(QWidget):
    """Queue management widget"""

    # Signals
    process_queue_requested = pyqtSignal()
    clear_queue_requested = pyqtSignal()
    remove_item_requested = pyqtSignal(str)  # item_id

    def __init__(self, queue_manager: QueueManager, parent=None):
        super().__init__(parent)
        self.queue_manager = queue_manager
        self.item_widgets = {}  # item_id -> widget
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title + Stats
        title_layout = QHBoxLayout()

        title_label = QLabel("<b>Processing Queue</b>")
        title_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        title_layout.addWidget(title_label)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #888888; font-size: 10px;")
        title_layout.addWidget(self.stats_label, 1, Qt.AlignRight)

        layout.addLayout(title_layout)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333333;
            }
            QListWidget::item:selected {
                background-color: #2d2d2d;
            }
        """)
        self.queue_list.setMinimumHeight(200)
        layout.addWidget(self.queue_list)

        # Overall progress
        progress_layout = QHBoxLayout()

        progress_label = QLabel("Overall Progress:")
        progress_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        progress_layout.addWidget(progress_label)

        self.overall_progress = QProgressBar()
        self.overall_progress.setMaximum(100)
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
                color: #cccccc;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #00aa00;
            }
        """)
        progress_layout.addWidget(self.overall_progress, 1)

        layout.addLayout(progress_layout)

        # Buttons
        button_layout = QHBoxLayout()

        self.btn_process = QPushButton("▶ Process Queue")
        self.btn_process.clicked.connect(self.on_process_clicked)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        button_layout.addWidget(self.btn_process)

        self.btn_remove = QPushButton("✕ Remove Selected")
        self.btn_remove.clicked.connect(self.on_remove_clicked)
        self.btn_remove.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                color: white;
                padding: 8px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #dd4444;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        button_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear Completed")
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                padding: 8px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        button_layout.addWidget(self.btn_clear)

        layout.addLayout(button_layout)

        self.update_ui()

    def add_item(self, item: QueueItem):
        """Add item to display"""
        # Create list item
        list_item = QListWidgetItem()
        self.queue_list.addItem(list_item)

        # Create widget
        item_widget = QueueItemWidget(item)
        list_item.setSizeHint(item_widget.sizeHint())
        self.queue_list.setItemWidget(list_item, item_widget)

        # Store reference
        self.item_widgets[item.id] = (list_item, item_widget)

        self.update_ui()

    def update_item(self, item_id: str, progress: int = None, status: QueueStatus = None):
        """Update item display"""
        if item_id in self.item_widgets:
            _, item_widget = self.item_widgets[item_id]

            if progress is not None:
                item_widget.update_progress(progress)

            if status is not None:
                item_widget.update_status(status)

            self.update_ui()

    def remove_item(self, item_id: str):
        """Remove item from display"""
        if item_id in self.item_widgets:
            list_item, _ = self.item_widgets[item_id]
            row = self.queue_list.row(list_item)
            self.queue_list.takeItem(row)
            del self.item_widgets[item_id]
            self.update_ui()

    def clear_display(self):
        """Clear all items from display"""
        self.queue_list.clear()
        self.item_widgets.clear()
        self.update_ui()

    def refresh_display(self):
        """Refresh entire display from queue manager"""
        self.clear_display()
        for item in self.queue_manager.queue:
            self.add_item(item)

    def update_ui(self):
        """Update UI state"""
        stats = self.queue_manager.get_queue_stats()

        # Update stats
        self.stats_label.setText(
            f"Total: {stats['total']} | "
            f"Pending: {stats['pending']} | "
            f"Completed: {stats['completed']} | "
            f"Failed: {stats['failed']}"
        )

        # Update overall progress
        progress = self.queue_manager.get_overall_progress()
        self.overall_progress.setValue(int(progress))

        # Update button states
        self.btn_process.setEnabled(stats["pending"] > 0)
        self.btn_remove.setEnabled(self.queue_list.currentRow() >= 0)
        self.btn_clear.setEnabled(stats["completed"] > 0 or stats["failed"] > 0)

    def on_process_clicked(self):
        """Handle process button click"""
        self.process_queue_requested.emit()

    def on_remove_clicked(self):
        """Handle remove button click"""
        current_row = self.queue_list.currentRow()
        if current_row >= 0:
            list_item = self.queue_list.item(current_row)
            # Find item ID
            for item_id, (li, _) in self.item_widgets.items():
                if li == list_item:
                    self.remove_item_requested.emit(item_id)
                    break

    def on_clear_clicked(self):
        """Handle clear button click"""
        self.clear_queue_requested.emit()
