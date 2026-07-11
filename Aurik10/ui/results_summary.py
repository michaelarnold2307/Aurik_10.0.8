"""
Aurik10/ui/results_summary.py — Verständliches Ergebnis-Feedback für Laien.

Zeigt nach der Restaurierung in einfacher Sprache, was Aurik getan hat.
Keine technischen Metriken — nur das, was der Nutzer wissen will.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from Aurik10.i18n import t

logger = logging.getLogger(__name__)


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Zeigt das Restaurierungsergebnis in einfacher, verständlicher Sprache."""

    play_requested = QtCore.pyqtSignal()
    open_folder_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        result_data: dict,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data = result_data
        self._build_ui()
        self.setWindowTitle(t("results.title"))
        self.setMinimumSize(520, 420)
        self.setModal(True)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 24)

        # ── Header ────────────────────────────────────────────────────────
        header = QtWidgets.QLabel(t("results.header_done"))
        header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #82B89A;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # ── File info line ────────────────────────────────────────────────
        d = self._data
        file_name = d.get("file_name", "?")
        duration = d.get("duration_seconds", 0)
        material = d.get("material_detected", "")
        mins = int(duration // 60)
        secs = int(duration % 60)

        info_parts = []
        info_parts.append(f"📂 {file_name}")
        if duration > 0:
            info_parts.append(f"🕐 {mins}:{secs:02d}")
        if material:
            mat_label = t(f"material.{material}") if material else ""
            if mat_label and mat_label != f"material.{material}":
                info_parts.append(f"💿 {mat_label}")
        info_line = QtWidgets.QLabel("  |  ".join(info_parts))
        info_line.setStyleSheet("font-size: 10pt; color: #8894A8;")
        info_line.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        info_line.setWordWrap(True)
        layout.addWidget(info_line)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("background: rgba(130, 184, 154, 0.2); max-height: 1px;")
        layout.addWidget(sep)

        # ── What was done section ─────────────────────────────────────────
        what_label = QtWidgets.QLabel(t("results.what_done"))
        what_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(what_label)

        improvements = self._build_improvements()
        layout.addWidget(improvements)

        # ── Quality indicator ─────────────────────────────────────────────
        quality_before = d.get("quality_before", 50)
        quality_after = d.get("quality_after", 85)

        if quality_after > quality_before:
            delta = quality_after - quality_before
            quality_text = t("results.quality_improved").format(
                before=int(quality_before), after=int(quality_after), delta=int(delta)
            )
        else:
            quality_text = t("results.quality_ok")

        quality_label = QtWidgets.QLabel(f"⭐ {quality_text}")
        quality_label.setStyleSheet(
            "font-size: 11pt; color: #B8A068; padding: 8px;background: rgba(184, 160, 104, 0.08); border-radius: 6px;"
        )
        quality_label.setWordWrap(True)
        layout.addWidget(quality_label)

        # ── Output path ───────────────────────────────────────────────────
        output = d.get("output_path", "")
        if output:
            fmt = d.get("export_format", "FLAC")
            output_label = QtWidgets.QLabel(t("results.saved_as").format(path=output, fmt=fmt))
            output_label.setStyleSheet("font-size: 10pt; color: #8894A8;")
            output_label.setWordWrap(True)
            layout.addWidget(output_label)

        # ── Spacer ────────────────────────────────────────────────────────
        layout.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        listen_btn = QtWidgets.QPushButton(t("results.listen"))
        listen_btn.setStyleSheet(
            "QPushButton { background: #667eea; color: white; border: none;"
            "border-radius: 8px; padding: 10px 24px; font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background: #7B93F0; }"
        )
        listen_btn.clicked.connect(self.play_requested.emit)
        btn_row.addWidget(listen_btn)

        if output:
            folder_btn = QtWidgets.QPushButton(t("results.open_folder"))
            folder_btn.setStyleSheet(
                "QPushButton { background: transparent; color: #8894A8;"
                "border: 1px solid rgba(136, 148, 168, 0.3); border-radius: 8px;"
                "padding: 10px 24px; font-size: 11pt; }"
                "QPushButton:hover { border-color: #667eea; color: #c9d1d9; }"
            )
            folder_btn.clicked.connect(lambda: self.open_folder_requested.emit(str(Path(output).parent)))
            btn_row.addWidget(folder_btn)

        ok_btn = QtWidgets.QPushButton(t("results.ok"))
        ok_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8894A8;"
            "border: 1px solid rgba(136, 148, 168, 0.3); border-radius: 8px;"
            "padding: 10px 24px; font-size: 11pt; }"
            "QPushButton:hover { border-color: #82B89A; color: #82B89A; }"
        )
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _build_improvements(self) -> QtWidgets.QWidget:
        """Baut die menschenlesbare Liste von Verbesserungen."""
        d = self._data
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setSpacing(6)
        vbox.setContentsMargins(0, 0, 0, 0)

        items = []

        # Defects
        defects_found = d.get("defects_found", 0)
        defects_fixed = d.get("defects_fixed", 0)
        if defects_fixed > 0:
            items.append(("✓", t("results.defects_fixed").format(n=defects_fixed), "#82B89A"))
        elif defects_found == 0:
            items.append(("✓", t("results.no_defects"), "#82B89A"))

        # Noise reduction
        noise_reduction = d.get("noise_reduction_pct", 0)
        if noise_reduction > 0:
            items.append(("✓", t("results.noise_reduced").format(pct=int(noise_reduction)), "#82B89A"))

        # Clarity improvement
        clarity_delta = d.get("clarity_improvement", 0)
        if clarity_delta > 0:
            items.append(("✓", t("results.clarity_improved"), "#82B89A"))

        # HPE naturalness
        hpe_before = d.get("hpe_before", 0)
        hpe_after = d.get("hpe_after", 0)
        if hpe_after > hpe_before + 0.03:
            items.append(("🎧", t("results.naturalness_improved"), "#7B93B8"))

        # Mode
        mode = d.get("mode", "")
        if mode:
            mode_name = "Studio 2026" if "STUDIO" in str(mode).upper() else "Restoration"
            items.append(("⚙️", t("results.mode_used").format(mode=mode_name), "#8894A8"))

        # Era
        era = d.get("era_detected", "")
        if era:
            items.append(("🕰️", t("results.era_detected").format(era=era), "#8894A8"))

        for icon, text, color in items:
            row = QtWidgets.QHBoxLayout()
            icon_label = QtWidgets.QLabel(icon)
            icon_label.setStyleSheet(f"color: {color}; font-size: 12pt; min-width: 24px;")
            icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            row.addWidget(icon_label)

            text_label = QtWidgets.QLabel(text)
            text_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
            text_label.setWordWrap(True)
            row.addWidget(text_label, 1)
            vbox.addLayout(row)

        vbox.addStretch()
        return widget


def build_results_data(
    *,
    file_name: str = "",
    duration_seconds: float = 0,
    defects_found: int = 0,
    defects_fixed: int = 0,
    quality_before: float = 50,
    quality_after: float = 85,
    material_detected: str = "",
    era_detected: str = "",
    mode: str = "",
    output_path: str = "",
    export_format: str = "FLAC",
    noise_reduction_pct: float = 0,
    clarity_improvement: float = 0,
    hpe_before: float = 0,
    hpe_after: float = 0,
) -> dict:
    """Baut das data-dict für den ResultsSummaryDialog."""
    return {
        "file_name": file_name,
        "duration_seconds": duration_seconds,
        "defects_found": defects_found,
        "defects_fixed": defects_fixed,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "material_detected": material_detected,
        "era_detected": era_detected,
        "mode": mode,
        "output_path": output_path,
        "export_format": export_format,
        "noise_reduction_pct": noise_reduction_pct,
        "clarity_improvement": clarity_improvement,
        "hpe_before": hpe_before,
        "hpe_after": hpe_after,
    }
