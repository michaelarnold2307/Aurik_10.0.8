"""
Preset Browser Widget
UI for browsing and managing presets
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.preset_manager import Preset, PresetCategory, PresetManager


class PresetBrowserWidget(QWidget):
    """Preset browser and management widget"""

    # Signals
    preset_selected = pyqtSignal(Preset)
    preset_applied = pyqtSignal(Preset)

    def __init__(self, preset_manager: PresetManager, parent=None):
        super().__init__(parent)
        self.preset_manager = preset_manager
        self.current_preset = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title_label = QLabel("<b>Preset Browser</b>")
        title_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(title_label)

        # Category filter
        filter_layout = QHBoxLayout()

        filter_label = QLabel("Category:")
        filter_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        filter_layout.addWidget(filter_label)

        self.category_combo = QComboBox()
        self.category_combo.addItems(["All", "Factory", "User", "Imported"])
        self.category_combo.currentTextChanged.connect(self.filter_presets)
        self.category_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 5px;
            }
        """)
        filter_layout.addWidget(self.category_combo, 1)

        layout.addLayout(filter_layout)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.itemSelectionChanged.connect(self.on_preset_selected)
        self.preset_list.itemDoubleClicked.connect(self.on_preset_double_clicked)
        self.preset_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 3px;
                color: #cccccc;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333333;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #2d2d2d;
            }
        """)
        layout.addWidget(self.preset_list)

        # Preset details
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(100)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 3px;
                font-size: 10px;
                padding: 5px;
            }
        """)
        details_layout.addWidget(self.details_text)

        layout.addWidget(details_group)

        # Buttons
        button_layout = QVBoxLayout()

        # Apply button
        self.btn_apply = QPushButton("✓ Apply Preset")
        self.btn_apply.clicked.connect(self.apply_preset)
        self.btn_apply.setEnabled(False)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                padding: 8px;
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
        button_layout.addWidget(self.btn_apply)

        # Management buttons
        mgmt_layout = QHBoxLayout()

        self.btn_save = QPushButton("💾 Save Current")
        self.btn_save.clicked.connect(self.save_current_settings)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                padding: 6px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        mgmt_layout.addWidget(self.btn_save)

        self.btn_delete = QPushButton("🗑 Delete")
        self.btn_delete.clicked.connect(self.delete_preset)
        self.btn_delete.setEnabled(False)
        self.btn_delete.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                color: white;
                padding: 6px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #dd4444;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        mgmt_layout.addWidget(self.btn_delete)

        button_layout.addLayout(mgmt_layout)

        # Import/Export buttons
        import_export_layout = QHBoxLayout()

        self.btn_import = QPushButton("📥 Import")
        self.btn_import.clicked.connect(self.import_preset)
        self.btn_import.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                padding: 6px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        import_export_layout.addWidget(self.btn_import)

        self.btn_export = QPushButton("📤 Export")
        self.btn_export.clicked.connect(self.export_preset)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                padding: 6px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #999999;
            }
        """)
        import_export_layout.addWidget(self.btn_export)

        button_layout.addLayout(import_export_layout)

        layout.addLayout(button_layout)

        # Load presets
        self.refresh_preset_list()

    def refresh_preset_list(self):
        """Refresh preset list"""
        self.preset_list.clear()

        category_filter = self.category_combo.currentText()

        for preset in self.preset_manager.get_all_presets():
            # Apply filter
            if category_filter != "All":
                if category_filter.lower() != preset.category.value:
                    continue

            # Add item
            item = QListWidgetItem()

            # Format name with category icon
            icon = {PresetCategory.FACTORY: "🏭", PresetCategory.USER: "👤", PresetCategory.IMPORTED: "📦"}.get(
                preset.category, ""
            )

            item.setText(f"{icon} {preset.name}")
            item.setData(Qt.UserRole, preset.name)

            self.preset_list.addItem(item)

    def filter_presets(self):
        """Filter presets by category"""
        self.refresh_preset_list()

    def on_preset_selected(self):
        """Handle preset selection"""
        items = self.preset_list.selectedItems()
        if not items:
            self.current_preset = None
            self.details_text.clear()
            self.btn_apply.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_export.setEnabled(False)
            return

        # Get preset
        preset_name = items[0].data(Qt.UserRole)
        self.current_preset = self.preset_manager.get_preset(preset_name)

        if self.current_preset:
            # Update details
            details = f"""
<b>{self.current_preset.name}</b><br>
<i>{self.current_preset.description}</i><br>
<br>
<b>Medium:</b> {self.current_preset.medium_type}<br>
<b>Mode:</b> {self.current_preset.processing_mode}<br>
<b>Category:</b> {self.current_preset.category.value.title()}<br>
<b>Author:</b> {self.current_preset.author}
            """
            self.details_text.setHtml(details)

            # Enable buttons
            self.btn_apply.setEnabled(True)
            self.btn_export.setEnabled(True)

            # Can only delete user/imported presets
            can_delete = self.current_preset.category in [PresetCategory.USER, PresetCategory.IMPORTED]
            self.btn_delete.setEnabled(can_delete)

            # Emit signal
            self.preset_selected.emit(self.current_preset)

    def on_preset_double_clicked(self, item):
        """Handle preset double-click (apply)"""
        self.apply_preset()

    def apply_preset(self):
        """Apply selected preset"""
        if self.current_preset:
            self.preset_applied.emit(self.current_preset)

    def save_current_settings(self):
        """Save current settings as new preset"""
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")

        if not ok or not name:
            return

        # Check if name exists
        if self.preset_manager.get_preset(name):
            reply = QMessageBox.question(
                self, "Overwrite?", f"Preset '{name}' already exists. Overwrite?", QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        description, ok = QInputDialog.getText(self, "Preset Description", "Enter description (optional):")

        if not ok:
            pass

        # This will be connected to the main window to get current settings
        # For now, just show success message
        QMessageBox.information(
            self,
            "Not Implemented",
            "This feature requires connection to main window settings.\n" "Will be implemented in integration.",
        )

    def delete_preset(self):
        """Delete selected preset"""
        if not self.current_preset:
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Delete Preset", f"Delete preset '{self.current_preset.name}'?", QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete
        if self.preset_manager.delete_preset(self.current_preset.name):
            QMessageBox.information(self, "Deleted", "Preset deleted successfully.")
            self.refresh_preset_list()
        else:
            QMessageBox.critical(self, "Error", "Failed to delete preset.")

    def import_preset(self):
        """Import preset from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Preset", str(Path.home()), "JSON Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        preset = self.preset_manager.import_preset(Path(filepath))
        if preset:
            QMessageBox.information(self, "Imported", f"Preset '{preset.name}' imported successfully.")
            self.refresh_preset_list()
        else:
            QMessageBox.critical(self, "Error", "Failed to import preset.")

    def export_preset(self):
        """Export selected preset"""
        if not self.current_preset:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Preset",
            str(Path.home() / f"{self.current_preset.name.replace(' ', '_')}.json"),
            "JSON Files (*.json);;All Files (*)",
        )

        if not filepath:
            return

        if self.preset_manager.export_preset(self.current_preset.name, Path(filepath)):
            QMessageBox.information(self, "Exported", "Preset exported successfully.")
        else:
            QMessageBox.critical(self, "Error", "Failed to export preset.")
