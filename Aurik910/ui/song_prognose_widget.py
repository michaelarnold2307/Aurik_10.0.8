"""
SongPrognoseWidget — Pre-flight deep analysis tab for Aurik 9.

Displays material, era/genre, restorability score, detected defects,
phase prognosis and an overall 'Chancen-Score' before restoration starts.
Updated asynchronously as background analysis threads complete.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (Aurik dark theme)
# ---------------------------------------------------------------------------
_C_BG = "rgba(14, 18, 36, 0.92)"
_C_CARD = "rgba(22, 28, 50, 0.82)"
_C_BORDER = "rgba(102, 126, 234, 0.22)"
_C_GREEN = "#82B89A"
_C_AMBER = "#C8A84B"
_C_RED = "#B87A7A"
_C_BLUE = "#667EEA"
_C_MUTED = "#6E839D"
_C_TEXT = "#D0DCFF"
_C_TEXT_DIM = "#8898BB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _card_style(border_color: str = _C_BORDER) -> str:
    return f"background:{_C_CARD}; border:1px solid {border_color}; border-radius:10px; padding:10px 14px;"


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_C_TEXT_DIM}; font-size:8pt; font-weight:600; letter-spacing:0.8px; background:transparent; padding:0;"
    )
    return lbl


class _ScoreDial(QWidget):
    """Small circular score indicator (0–100)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._score: float = 0.0
        self._color = _C_MUTED
        self.setFixedSize(80, 80)

    def set_score(self, score: float, color: str) -> None:
        self._score = float(np.clip(score, 0.0, 100.0))
        self._color = color
        self.update()

    def paintEvent(self, _event: Any) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = min(w, h) // 2 - 6
        cx, cy = w // 2, h // 2

        # Track
        pen = QPen(QColor("#1E253D"), 7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, -270 * 16)

        # Arc
        if self._score > 0:
            pen2 = QPen(QColor(self._color), 7)
            pen2.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen2)
            span = int(-270 * 16 * self._score / 100.0)
            p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 225 * 16, span)

        # Value label
        p.setPen(QColor(_C_TEXT))
        font = QFont(self.font().family(), 14, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{int(self._score)}")
        p.end()


class _DefectPill(QLabel):
    """Compact pill showing a single defect with severity colour."""

    _SEVERITY_COLORS = {
        "low": ("#4A90D9", "rgba(74,144,217,0.14)", "rgba(74,144,217,0.30)"),
        "medium": ("#C8A84B", "rgba(200,168,75,0.14)", "rgba(200,168,75,0.30)"),
        "high": ("#B87A7A", "rgba(184,122,122,0.14)", "rgba(184,122,122,0.30)"),
    }

    def __init__(self, label: str, severity: str = "medium") -> None:
        super().__init__(label)
        fg, bg, border = self._SEVERITY_COLORS.get(severity, self._SEVERITY_COLORS["medium"])
        self.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {border};"
            " border-radius:8px; padding:2px 8px; font-size:8pt; font-weight:600;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


# ---------------------------------------------------------------------------
# DEFECT_LABELS: German user-readable names for limiting defects
# ---------------------------------------------------------------------------
_DEFECT_LABELS: dict[str, str] = {
    "extremes_rauschen": "Extremes Rauschen",
    "starkes_rauschen": "Starkes Rauschen",
    "leichtes_rauschen": "Leichtes Rauschen",
    "starkes_clipping": "Starke Übersteuerung",
    "leichtes_clipping": "Übersteuerung",
    "sehr_schmale_bandbreite": "Sehr schmale Bandbreite",
    "schmale_bandbreite": "Schmale Bandbreite",
    "starkes_crackle": "Starkes Knistern",
    "leichtes_crackle": "Leichtes Knistern",
    # DefectScanner keys
    "CRACKLE": "Knistern",
    "CLICKS": "Knackser",
    "CLIPPING": "Übersteuerung",
    "HUM": "Netzbrummen",
    "NOISE": "Rauschen",
    "DROPOUT": "Aussetzer",
    "WOW_FLUTTER": "Gleichlaufschwankungen",
    "REVERB": "Nachhall",
    "TAPE_HISS": "Bandrauschen",
    "DC_OFFSET": "Gleichspannung",
    # Extended DefectType enum labels
    "WOW": "Gleichlaufschwankung (Wow)",
    "FLUTTER": "Tonhöhenzittern (Flutter)",
    "ALIASING": "Aliasing-Verzerrung",
    "AZIMUTH_ERROR": "Azimuth-Abweichung",
    "BIAS_ERROR": "Vormagnetisierungsfehler",
    "BANDWIDTH_LOSS": "Bandbreitenverlust",
    "PHASE_ISSUES": "Phasenproblem",
    "STEREO_IMBALANCE": "Stereo-Ungleichgewicht",
    "PITCH_DRIFT": "Tonhöhendrift",
    "REVERB_EXCESS": "Übermäßiger Nachhall",
    "PRINT_THROUGH": "Übersprechen (Print-Through)",
    "QUANTIZATION_NOISE": "Quantisierungsrauschen",
    "JITTER_ARTIFACTS": "Jitter-Artefakte",
    "DYNAMIC_COMPRESSION_EXCESS": "Starke Dynamikkompression",
    "PRE_ECHO": "Vor-Echo",
    "TRANSIENT_SMEARING": "Transienten-Verschmierung",
    "SOFT_SATURATION": "Weiche Sättigung",
    "HEAD_WEAR": "Tonkopf-Verschleiß",
    "RIAA_CURVE_ERROR": "RIAA-Kurven-Abweichung",
    "TRANSPORT_BUMP": "Transportgeräusch",
    "VOCAL_HARSHNESS": "Stimmlich-Rauheit",
    "HISS": "Rauschen",
    "SURFACE_NOISE": "Oberflächenrauschen",
    "DISTORTION": "Verzerrung",
    "SATURATION": "Sättigung",
    "SIBILANCE": "Zischlaute",
}


def _defect_label(key: str) -> str:
    if key in _DEFECT_LABELS:
        return _DEFECT_LABELS[key]
    # Fallback: prettify
    return key.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Phase prognosis heuristics
# ---------------------------------------------------------------------------
_GRADE_PHASES: dict[str, tuple[int, int, str]] = {
    "excellent": (22, 28, "Leichte Behandlung"),
    "good": (28, 34, "Standard-Restaurierung"),
    "fair": (34, 40, "Intensive Restaurierung"),
    "poor": (38, 45, "Tiefgreifende Restaurierung"),
    "critical": (40, 50, "Vollständige Rekonstruktion"),
    "unknown": (28, 38, "Analyse läuft …"),
}

_GRADE_LABELS_DE: dict[str, tuple[str, str, str]] = {
    "excellent": (_C_GREEN, "Exzellent", "▶  Sehr gut restaurierbar"),
    "good": (_C_GREEN, "Gut", "▶  Gut restaurierbar"),
    "fair": (_C_AMBER, "Mäßig", "▶  Mäßig restaurierbar"),
    "poor": (_C_RED, "Schwierig", "▶  Schwierig restaurierbar"),
    "critical": (_C_RED, "Kritisch", "▶  Sehr stark beschädigt"),
    "unknown": (_C_MUTED, "—", "— wird analysiert …"),
}

# Material human-readable names
_MATERIAL_NAMES: dict[str, str] = {
    "wax_cylinder": "Wachswalze",
    "lacquer_disc": "Lackfolie",
    "shellac": "Schellack",
    "vinyl": "Vinyl",
    "wire_recording": "Drahtband",
    "reel_tape": "Spulenband",
    "tape": "Kassette (Band)",
    "cassette": "Kassette (Band)",
    "dat": "DAT",
    "cd_digital": "CD / Digital",
    "cd": "CD",
    "digital": "Digital",
    "minidisc": "MiniDisc",
    "mp3_low": "MP3 (niedrige Bitrate)",
    "mp3_high": "MP3",
    "damaged_mp3": "MP3 (beschädigt)",
    "aac": "AAC",
    "streaming": "Streaming-Format",
    "unknown": "Unbekannt",
}


# ---------------------------------------------------------------------------
# SongPrognoseWidget
# ---------------------------------------------------------------------------


class SongPrognoseWidget(QWidget):
    """
    Comprehensive pre-flight analysis tab shown after file open.

    Public update interface (all must be called from GUI thread):
      - reset()                       → new file loaded
      - update_material(key, conf)    → after MediumClassifier
      - update_era_genre(decade, genre) → after EraClassifier / GenreClassifier
      - update_restorability(result)  → after RestorabilityEstimator
      - update_defects(defects_dict)  → after DefectScanner (optional, detected)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grade: str = "unknown"
        self._score100: float = 0.0
        self._material: str = "unknown"
        self._decade: int | None = None
        self._genre: str = ""
        self._result_obj: Any = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:{_C_BG}; border:none; }}"
            " QScrollBar:vertical { background:rgba(20,24,48,0.7); width:8px; border-radius:4px; }"
            " QScrollBar::handle:vertical { background:rgba(102,126,234,0.4); border-radius:4px; }"
        )

        container = QWidget()
        container.setStyleSheet(f"background:{_C_BG};")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        self._header_lbl = QLabel("🔍  Song-Analyse")
        self._header_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:12pt; font-weight:bold; background:transparent;")
        hdr_row.addWidget(self._header_lbl)
        hdr_row.addStretch()
        self._status_lbl = QLabel("Datei öffnen, um die Analyse zu starten")
        self._status_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:9pt; background:transparent;")
        hdr_row.addWidget(self._status_lbl)
        main_layout.addLayout(hdr_row)

        # ── Row 1: Chancen-Score (dial) + Material/Ära/Genre ─────────
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        # Left: big score dial + grade
        dial_card = QFrame()
        dial_card.setStyleSheet(_card_style())
        dial_inner = QVBoxLayout(dial_card)
        dial_inner.setContentsMargins(12, 10, 12, 10)
        dial_inner.setSpacing(4)
        dial_inner.addWidget(_section_label("CHANCEN-SCORE"))
        dial_row = QHBoxLayout()
        self._dial = _ScoreDial()
        dial_row.addWidget(self._dial)
        dial_label_col = QVBoxLayout()
        self._grade_lbl = QLabel("—")
        self._grade_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_sub_lbl = QLabel("— wird analysiert …")
        self._grade_sub_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        self._grade_sub_lbl.setWordWrap(True)
        dial_label_col.addWidget(self._grade_lbl)
        dial_label_col.addWidget(self._grade_sub_lbl)
        dial_label_col.addStretch()
        dial_row.addLayout(dial_label_col)
        dial_inner.addLayout(dial_row)
        dial_inner.addStretch()
        row1.addWidget(dial_card, 0)

        # Right: meta info (material, era, genre, SNR)
        meta_card = QFrame()
        meta_card.setStyleSheet(_card_style())
        meta_inner = QVBoxLayout(meta_card)
        meta_inner.setContentsMargins(12, 10, 12, 10)
        meta_inner.setSpacing(6)
        meta_inner.addWidget(_section_label("AUFNAHME-MERKMALE"))

        self._meta_rows: dict[str, QLabel] = {}
        for key, placeholder in [
            ("material", "— wird erkannt …"),
            ("era", "— wird erkannt …"),
            ("genre", "— wird erkannt …"),
            ("snr", "— wird gemessen …"),
            ("bandwidth", "— wird gemessen …"),
        ]:
            row_w = QHBoxLayout()
            row_w.setSpacing(6)
            row_key = _section_label(key.upper())
            row_key.setFixedWidth(80)
            row_val = QLabel(placeholder)
            row_val.setStyleSheet(f"color:{_C_TEXT}; font-size:9pt; background:transparent;")
            self._meta_rows[key] = row_val
            row_w.addWidget(row_key)
            row_w.addWidget(row_val)
            row_w.addStretch()
            meta_inner.addLayout(row_w)

        meta_inner.addStretch()
        row1.addWidget(meta_card, 1)
        main_layout.addLayout(row1)

        # ── Row 2: MOS prediction bar ─────────────────────────────────
        mos_card = QFrame()
        mos_card.setStyleSheet(_card_style())
        mos_inner = QVBoxLayout(mos_card)
        mos_inner.setContentsMargins(12, 8, 12, 8)
        mos_inner.setSpacing(4)
        mos_inner.addWidget(_section_label("ERWARTETE QUALITÄT NACH RESTAURIERUNG (MOS 1–5)"))

        mos_bar_row = QHBoxLayout()
        mos_bar_row.setSpacing(10)
        self._mos_bar = QProgressBar()
        self._mos_bar.setRange(0, 500)
        self._mos_bar.setValue(0)
        self._mos_bar.setTextVisible(False)
        self._mos_bar.setFixedHeight(14)
        self._mos_bar.setStyleSheet(
            "QProgressBar { background:rgba(30,37,61,0.9); border-radius:7px; border:none; }"
            f"QProgressBar::chunk {{ background:{_C_BLUE}; border-radius:7px; }}"
        )
        self._mos_val_lbl = QLabel("—")
        self._mos_val_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:10pt; font-weight:bold; background:transparent;")
        self._mos_val_lbl.setFixedWidth(60)
        _lbl_lo = QLabel("1,0")
        _lbl_lo.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_bar_row.addWidget(_lbl_lo, 0)
        mos_bar_row.addWidget(self._mos_bar, 1)
        lbl5 = QLabel("5,0")
        lbl5.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_bar_row.addWidget(lbl5, 0)
        mos_bar_row.addWidget(self._mos_val_lbl, 0)
        mos_inner.addLayout(mos_bar_row)

        self._mos_range_lbl = QLabel("")
        self._mos_range_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:8pt; background:transparent;")
        mos_inner.addWidget(self._mos_range_lbl)
        main_layout.addWidget(mos_card)

        # ── Row 3: Defekte ────────────────────────────────────────────
        defect_card = QFrame()
        defect_card.setStyleSheet(_card_style())
        defect_inner = QVBoxLayout(defect_card)
        defect_inner.setContentsMargins(12, 8, 12, 8)
        defect_inner.setSpacing(6)
        defect_inner.addWidget(_section_label("ERKANNTE SCHÄDEN"))

        self._defect_pills_row = QHBoxLayout()
        self._defect_pills_row.setSpacing(6)
        self._defect_pills_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._defect_placeholder = QLabel("— Scan läuft nach dem Start …")
        self._defect_placeholder.setStyleSheet(f"color:{_C_MUTED}; font-size:9pt; background:transparent;")
        self._defect_pills_row.addWidget(self._defect_placeholder)
        defect_inner.addLayout(self._defect_pills_row)

        self._defect_recs_lbl = QLabel("")
        self._defect_recs_lbl.setStyleSheet(f"color:{_C_TEXT_DIM}; font-size:8pt; background:transparent;")
        self._defect_recs_lbl.setWordWrap(True)
        defect_inner.addWidget(self._defect_recs_lbl)
        main_layout.addWidget(defect_card)

        # ── Row 4: Phase prognosis ────────────────────────────────────
        phase_card = QFrame()
        phase_card.setStyleSheet(_card_style())
        phase_inner = QVBoxLayout(phase_card)
        phase_inner.setContentsMargins(12, 8, 12, 8)
        phase_inner.setSpacing(4)
        phase_inner.addWidget(_section_label("PHASENPROGNOSE"))

        phase_row = QHBoxLayout()
        phase_row.setSpacing(16)
        self._phase_count_lbl = QLabel("—")
        self._phase_count_lbl.setStyleSheet(
            f"color:{_C_TEXT}; font-size:12pt; font-weight:bold; background:transparent;"
        )
        phase_row.addWidget(self._phase_count_lbl)
        self._phase_desc_lbl = QLabel("—")
        self._phase_desc_lbl.setStyleSheet(f"color:{_C_TEXT_DIM}; font-size:9pt; background:transparent;")
        phase_row.addWidget(self._phase_desc_lbl)
        phase_row.addStretch()
        self._mode_rec_lbl = QLabel("")
        self._mode_rec_lbl.setStyleSheet(f"color:{_C_BLUE}; font-size:9pt; font-weight:bold; background:transparent;")
        phase_row.addWidget(self._mode_rec_lbl)
        phase_inner.addLayout(phase_row)
        main_layout.addWidget(phase_card)

        # ── Row 5: Recommendations ────────────────────────────────────
        rec_card = QFrame()
        rec_card.setStyleSheet(_card_style())
        rec_inner = QVBoxLayout(rec_card)
        rec_inner.setContentsMargins(12, 8, 12, 8)
        rec_inner.setSpacing(4)
        rec_inner.addWidget(_section_label("EMPFEHLUNGEN"))
        self._rec_lbl = QLabel("— Analyse läuft …")
        self._rec_lbl.setWordWrap(True)
        self._rec_lbl.setStyleSheet(f"color:{_C_TEXT}; font-size:9pt; background:transparent; line-height:160%;")
        rec_inner.addWidget(self._rec_lbl)
        main_layout.addWidget(rec_card)

        main_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public update interface (GUI thread only)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called when a new file is loaded — resets all fields."""
        self._grade = "unknown"
        self._score100 = 0.0
        self._material = "unknown"
        self._decade = None
        self._genre = ""
        self._result_obj = None

        self._dial.set_score(0.0, _C_MUTED)
        self._grade_lbl.setStyleSheet(f"color:{_C_MUTED}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_lbl.setText("—")
        self._grade_sub_lbl.setText("— wird analysiert …")
        self._status_lbl.setText("Analyse läuft …")

        for key in ("material", "era", "genre", "snr", "bandwidth"):
            self._meta_rows[key].setText("— wird erkannt …")

        self._mos_bar.setValue(0)
        self._mos_val_lbl.setText("—")
        self._mos_range_lbl.setText("")

        self._phase_count_lbl.setText("—")
        self._phase_desc_lbl.setText("—")
        self._mode_rec_lbl.setText("")

        self._rec_lbl.setText("— Analyse läuft …")
        self._defect_recs_lbl.setText("")
        self._clear_defect_pills()
        self._defect_placeholder.setText("— Scan läuft nach dem Start …")
        self._defect_placeholder.setVisible(True)

    def update_material(self, material_key: str, confidence: float) -> None:
        """Update material row. Call from GUI thread after MediumClassifier."""
        self._material = str(material_key or "unknown")
        name = _MATERIAL_NAMES.get(self._material, self._material)
        pct = int(round(confidence * 100))
        self._meta_rows["material"].setText(
            f"{name}  <span style='color:{_C_MUTED};font-size:8pt;'>({pct}\u202f% Sicherheit)</span>"
        )
        self._meta_rows["material"].setTextFormat(Qt.TextFormat.RichText)
        self._refresh_phase_prognosis()

    def update_era_genre(self, decade: int | None, genre: str | None) -> None:
        """Update era/genre rows. Call from GUI thread after EraClassifier."""
        self._decade = decade
        self._genre = str(genre or "")
        era_txt = f"{decade}er" if decade else "—"
        self._meta_rows["era"].setText(era_txt)
        self._meta_rows["genre"].setText(self._genre if self._genre else "—")

    def update_restorability(self, result: Any) -> None:
        """
        Update all score fields. Call from GUI thread after RestorabilityEstimator.
        result: RestorabilityResult
        """
        self._result_obj = result
        if result is None:
            return

        score100 = float(getattr(result, "restorability_score", 50.0))
        predicted_mos = float(getattr(result, "predicted_mos", 3.5))
        mos_range = getattr(result, "predicted_mos_range", (predicted_mos - 0.3, predicted_mos + 0.3))
        limiting = list(getattr(result, "limiting_defects", []))
        snr_db = float(getattr(result, "snr_db", 0.0))
        grade = str(getattr(result, "grade", "unknown"))
        recommendations = list(getattr(result, "recommendations", []))

        self._grade = grade
        self._score100 = score100

        color, grade_name, grade_line = _GRADE_LABELS_DE.get(grade, (_C_MUTED, grade, grade))

        # Dial
        self._dial.set_score(score100, color)
        self._grade_lbl.setText(grade_name)
        self._grade_lbl.setStyleSheet(f"color:{color}; font-size:10pt; font-weight:bold; background:transparent;")
        self._grade_sub_lbl.setText(grade_line)
        self._status_lbl.setText("Analyse abgeschlossen")

        # MOS bar
        self._mos_bar.setValue(int(round(predicted_mos * 100)))
        bar_color = _C_GREEN if predicted_mos >= 4.0 else (_C_AMBER if predicted_mos >= 3.0 else _C_RED)
        self._mos_bar.setStyleSheet(
            "QProgressBar { background:rgba(30,37,61,0.9); border-radius:7px; border:none; }"
            f"QProgressBar::chunk {{ background:{bar_color}; border-radius:7px; }}"
        )
        self._mos_val_lbl.setText(f"{predicted_mos:.1f} / 5")
        self._mos_val_lbl.setStyleSheet(f"color:{bar_color}; font-size:10pt; font-weight:bold; background:transparent;")
        lo = float(mos_range[0]) if mos_range else predicted_mos - 0.3
        hi = float(mos_range[1]) if len(mos_range) >= 2 else predicted_mos + 0.3  # type: ignore[arg-type]
        self._mos_range_lbl.setText(f"90\u202f%-Wahrscheinlichkeit: {lo:.1f} – {hi:.1f}")

        # SNR
        self._meta_rows["snr"].setText(
            f"{snr_db:.1f}\u202fdB" + ("  ✓" if snr_db >= 20 else ("  △" if snr_db >= 5 else "  ✗"))
        )

        # Defect pills from limiting_defects
        if limiting:
            self._clear_defect_pills()
            self._defect_placeholder.setVisible(False)
            for d in limiting[:6]:
                sev = "high" if any(s in str(d) for s in ("stark", "extrem", "critical")) else "medium"
                pill = _DefectPill(_defect_label(str(d)), sev)
                self._defect_pills_row.addWidget(pill)

        # Recommendations
        if recommendations:
            self._rec_lbl.setText("\n".join(f"• {r}" for r in recommendations))
        else:
            self._rec_lbl.setText("—")

        # Phase prognosis
        self._refresh_phase_prognosis()

        # Print terminal report
        _print_prognose_terminal(
            material=self._material,
            decade=self._decade,
            genre=self._genre,
            score100=score100,
            grade=grade,
            predicted_mos=predicted_mos,
            mos_range=(lo, hi),
            snr_db=snr_db,
            limiting_defects=limiting,
            recommendations=recommendations,
        )

    def update_defects(self, defects: dict) -> None:
        """
        Update defect pills from DefectScanner result (BatchProcessingThread).
        Only processes status == 'detected'.
        """
        if defects.get("status") != "detected":
            return

        # Collect significant defects (severity > 0.15)
        significant: list[tuple[str, float]] = []
        for k, v in defects.items():
            if k in ("status", "_no_anim"):
                continue
            if isinstance(v, (int, float)) and float(v) > 0.15:
                significant.append((str(k), float(v)))

        if not significant:
            return

        significant.sort(key=lambda x: x[1], reverse=True)
        self._clear_defect_pills()
        self._defect_placeholder.setVisible(False)

        for key, sev_val in significant[:8]:
            sev = "high" if sev_val > 0.6 else ("medium" if sev_val > 0.25 else "low")
            label = f"{_defect_label(key)} ({sev_val:.0%})"
            pill = _DefectPill(label, sev)
            self._defect_pills_row.addWidget(pill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_defect_pills(self) -> None:
        for i in reversed(range(self._defect_pills_row.count())):
            item = self._defect_pills_row.itemAt(i)
            if item and item.widget() is not self._defect_placeholder:
                w = item.widget()
                if w:
                    self._defect_pills_row.removeWidget(w)
                    w.deleteLater()

    def _refresh_phase_prognosis(self) -> None:
        lo, hi, desc = _GRADE_PHASES.get(self._grade, _GRADE_PHASES["unknown"])

        # Analog materials get more phases
        if self._material in ("shellac", "wax_cylinder", "lacquer_disc", "wire_recording"):
            lo += 4
            hi += 6
            desc += " (historisches Analog-Material)"
        elif self._material in ("vinyl", "reel_tape", "tape", "cassette"):
            lo += 2
            hi += 3

        self._phase_count_lbl.setText(f"~ {lo}–{hi}")
        self._phase_desc_lbl.setText(desc)

        # Mode recommendation
        if self._grade in ("excellent", "good"):
            self._mode_rec_lbl.setText("✦ Studio 2026 empfohlen")
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_GREEN}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        elif self._grade in ("fair",):
            self._mode_rec_lbl.setText("▶ Restoration empfohlen")
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_AMBER}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        elif self._grade in ("poor", "critical"):
            self._mode_rec_lbl.setText("▶ Restoration (maximale Verträglichkeit)")
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_RED}; font-size:9pt; font-weight:bold; background:transparent;"
            )

    def set_recommended_mode(self, mode: str) -> None:
        """Override the grade-based mode recommendation with the authoritative result
        from _recommend_mode_from_ui_context() (which considers material, defect severity,
        era, and genre in addition to the plain restorability grade).  Called by
        ModernMainWindow._apply_mode_recommendation_visuals() once pre-analysis is final."""
        if mode == "STUDIO_2026":
            self._mode_rec_lbl.setText("\u2746 Studio 2026 empfohlen")
            self._mode_rec_lbl.setStyleSheet(
                f"color:{_C_GREEN}; font-size:9pt; font-weight:bold; background:transparent;"
            )
        else:  # RESTORATION
            if self._grade in ("poor", "critical"):
                self._mode_rec_lbl.setText("\u25b6 Restoration (maximale Vertr\u00e4glichkeit)")
                self._mode_rec_lbl.setStyleSheet(
                    f"color:{_C_RED}; font-size:9pt; font-weight:bold; background:transparent;"
                )
            else:
                self._mode_rec_lbl.setText("\u25b6 Restoration empfohlen")
                self._mode_rec_lbl.setStyleSheet(
                    f"color:{_C_AMBER}; font-size:9pt; font-weight:bold; background:transparent;"
                )


# ---------------------------------------------------------------------------
# Terminal report (ANSI colour output)
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_GREEN = "\033[32m"
_ANSI_AMBER = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_BLUE = "\033[34;1m"
_ANSI_CYAN = "\033[36m"
_ANSI_DIM = "\033[2m"


def _color_grade(grade: str, text: str) -> str:
    c = {
        "excellent": _ANSI_GREEN,
        "good": _ANSI_GREEN,
        "fair": _ANSI_AMBER,
        "poor": _ANSI_RED,
        "critical": _ANSI_RED,
    }.get(grade, "")
    return f"{c}{_ANSI_BOLD}{text}{_ANSI_RESET}" if c else text


def _print_prognose_terminal(
    *,
    material: str,
    decade: int | None,
    genre: str,
    score100: float,
    grade: str,
    predicted_mos: float,
    mos_range: tuple[float, float],
    snr_db: float,
    limiting_defects: list[str],
    recommendations: list[str],
) -> None:
    """Print a colored prognosis report to the logger (INFO level)."""
    grade_de = {
        "excellent": "Exzellent",
        "good": "Gut",
        "fair": "Mäßig",
        "poor": "Schwierig",
        "critical": "Kritisch",
    }.get(grade, grade)
    mat_name = _MATERIAL_NAMES.get(material, material)
    lo, hi, _ = _GRADE_PHASES.get(grade, _GRADE_PHASES["unknown"])

    lines = [
        "",
        f"{_ANSI_BOLD}{_ANSI_BLUE}╔══════════════════════════════════════════════════════╗{_ANSI_RESET}",
        f"{_ANSI_BOLD}{_ANSI_BLUE}║        Aurik 9 — Song-Prognose (Pre-Flight)          ║{_ANSI_RESET}",
        f"{_ANSI_BOLD}{_ANSI_BLUE}╚══════════════════════════════════════════════════════╝{_ANSI_RESET}",
        f"  {_ANSI_DIM}Trägermedium{_ANSI_RESET}    : {mat_name}",
        f"  {_ANSI_DIM}Aufnahme-Ära{_ANSI_RESET}    : {f'{decade}er' if decade else '—'}",
        f"  {_ANSI_DIM}Genre{_ANSI_RESET}           : {genre if genre else '—'}",
        f"  {_ANSI_DIM}SNR{_ANSI_RESET}             : {snr_db:.1f} dB",
        "",
        f"  {_ANSI_BOLD}Chancen-Score{_ANSI_RESET}   : {_color_grade(grade, f'{score100:.0f} / 100  ({grade_de})')}",
        f"  {_ANSI_BOLD}MOS-Prognose{_ANSI_RESET}    : {predicted_mos:.2f}  [{mos_range[0]:.1f}–{mos_range[1]:.1f}]",
        f"  {_ANSI_BOLD}Phasen-Schätzung{_ANSI_RESET}: {lo}–{hi} Phasen",
        "",
    ]

    if limiting_defects:
        lines.append(f"  {_ANSI_AMBER}Hauptschäden{_ANSI_RESET}:")
        for d in limiting_defects:
            lines.append(f"    • {_defect_label(d)}")
        lines.append("")

    if recommendations:
        lines.append(f"  {_ANSI_CYAN}Empfehlungen{_ANSI_RESET}:")
        for r in recommendations:
            lines.append(f"    • {r}")
        lines.append("")

    lines.append(f"{_ANSI_DIM}  ─────────────────────────────────────────────────────{_ANSI_RESET}")

    report = "\n".join(lines)
    logger.info("PROGNOSE_REPORT:\n%s", report)
