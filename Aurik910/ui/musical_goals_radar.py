"""
Musical Goals Bar Chart Widget für Aurik 9
Zeigt alle 14 musikalischen Qualitätsziele als lesbares Balkendiagramm.

Rein PyQt5-basiert (kein Matplotlib) — CPU-leicht, sofort responsiv.
Vollständig laienfreundlich: Deutsche Labels, Farb-Kodierung, Prozentwerte, Tooltips.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import QSizePolicy, QToolTip, QWidget

# ─────────────────────────────────────────────────────────────────────────────
# Datmodell: Die 14 Musical Goals
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GoalEntry:
    """Ein einzelnes Musical Goal mit allen relevanten Metadaten."""

    key: str  # interner Schlüssel
    label: str  # Vollbezeichnung (Deutsch)
    threshold: float  # Pflicht-Schwellwert laut Spec
    score: float = 0.0  # aktueller Wert ∈ [0, 1]
    adaptive_threshold: float = -1.0  # adaptierter Schwellwert (–1 = nicht gesetzt)
    applicable: bool = True  # GoalApplicabilityFilter
    inapplicable_reason: str = ""  # Deutsch: Warum nicht messbar?
    synthesized: bool = False  # EraAuthenticPerceptualCompletion aktiv?
    adaptation_reason: str = ""  # Deutsch: Warum Schwellwert angepasst?


# Standard-Goals gemäß §1.2 der Spec — vollständige deutsche Bezeichnungen
DEFAULT_GOALS: list[GoalEntry] = [
    GoalEntry("brillanz", "Brillanz", 0.85),
    GoalEntry("waerme", "Wärme", 0.80),
    GoalEntry("natuerlichkeit", "Natürlichkeit", 0.90),
    GoalEntry("authentizitaet", "Authentizität", 0.88),
    GoalEntry("emotionalitaet", "Emotionalität", 0.87),
    GoalEntry("transparenz", "Transparenz", 0.89),
    GoalEntry("bass_kraft", "Bass-Kraft", 0.85),
    GoalEntry("groove", "Groove", 0.88),
    GoalEntry("spatial_depth", "Raumtiefe", 0.75),
    GoalEntry("timbre_authentizitaet", "Timbre", 0.87),
    GoalEntry("tonal_center", "Tonales Zentrum", 0.95),
    GoalEntry("micro_dynamics", "Mikro-Dynamik", 0.92),
    GoalEntry("separation_fidelity", "Separation", 0.82),
    GoalEntry("artikulation", "Artikulation", 0.85),
]

# Farb-Konstanten
COLOR_PASS = QColor(76, 175, 80, 220)  # Grün  — Ziel erfüllt
COLOR_WARN = QColor(255, 193, 7, 220)  # Gelb  — knapp am Limit (< +0.04)
COLOR_FAIL = QColor(244, 67, 54, 220)  # Rot   — Ziel unterschritten
COLOR_NA = QColor(100, 110, 130, 140)  # Grau  — nicht anwendbar
COLOR_SYNTH = QColor(220, 130, 240, 200)  # Lila  — era-authentisch ergänzt
COLOR_BAR_BG = QColor(30, 38, 58, 180)  # Balken-Hintergrund
COLOR_THRESH = QColor(255, 255, 255, 120)  # Schwellwert-Linie
COLOR_TEXT = QColor(190, 200, 218)  # Standard-Textfarbe
COLOR_DIM = QColor(120, 130, 150, 160)  # Gedimmter Text

# Vollständige Farbpalette für Bar BG je nach Status
_BAR_BG_PASS = QColor(76, 175, 80, 40)
_BAR_BG_WARN = QColor(255, 193, 7, 35)
_BAR_BG_FAIL = QColor(244, 67, 54, 35)
_BAR_BG_NA = QColor(60, 70, 90, 60)


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Widget — Balkendiagramm statt Radar
# ─────────────────────────────────────────────────────────────────────────────


class MusicalGoalsRadarWidget(QWidget):
    """
    Horizontales Balkendiagramm für die 14 Musical Goals.

    Zeigt pro Ziel:
    - Vollständiger deutscher Name
    - Farbiger Balken (Score-Länge) mit Schwellwert-Markierung
    - Prozentwert als Zahl (lesbar)
    - Status-Icon (✅ / ⚠ / ✗ / –)
    - Farb-Kodierung: Grün/Gelb/Rot/Grau

    Vor Restaurierung: freundliche Platzhaltermeldung.
    Hover-Tooltips: erklären das Ziel auf Deutsch.
    """

    # Geometry constants
    _LABEL_W = 98  # px width of the goal name column
    _SCORE_W = 34  # px width of the "87 %" score column
    _ICON_W = 18  # px width of status icon
    _ROW_H = 16  # px height per goal row
    _ROW_GAP = 2  # px gap between rows
    _PAD_X = 8  # horizontal outer padding
    _PAD_TOP = 28  # top padding (header)
    _PAD_BOT = 22  # bottom padding (legend)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

        import copy

        self._goals: list[GoalEntry] = copy.deepcopy(DEFAULT_GOALS)
        self._hovered_idx: int = -1
        self._has_data: bool = False  # False = vor erster Restaurierung

        # Animation
        self._anim_scores: dict[str, float] = {g.key: 0.0 for g in DEFAULT_GOALS}
        self._target_scores: dict[str, float] = {g.key: 0.0 for g in DEFAULT_GOALS}
        self._anim_timer: QTimer | None = None
        self._anim_step: int = 0
        self._ANIM_STEPS: int = 28  # 28 × 20 ms ≈ 560 ms

    # ──────────────────────────── Public API ──────────────────────────────

    def update_scores(
        self,
        scores: dict[str, float],
        adaptive_thresholds: dict[str, float] | None = None,
        applicable_goals: set[str] | None = None,
        inapplicable_reasons: dict[str, str] | None = None,
        synthesized_goals: set[str] | None = None,
        adaptation_reasons: dict[str, str] | None = None,
    ) -> None:
        """Aktualisiert alle Score-Werte und löst ein Neuzeichnen aus."""
        adaptive_thresholds = adaptive_thresholds or {}
        applicable = applicable_goals
        inapplicable_reasons = inapplicable_reasons or {}
        synthesized = synthesized_goals or set()
        adapt_reasons = adaptation_reasons or {}

        has_any = any(v > 0.001 for v in scores.values())
        self._has_data = has_any

        for g in self._goals:
            g.applicable = (applicable is None) or (g.key in applicable)
            g.inapplicable_reason = inapplicable_reasons.get(g.key, "")
            g.synthesized = g.key in synthesized
            g.adaptive_threshold = float(adaptive_thresholds[g.key]) if g.key in adaptive_thresholds else -1.0
            g.adaptation_reason = adapt_reasons.get(g.key, "")
            self._target_scores[g.key] = float(scores.get(g.key, 0.0))
            self._anim_scores[g.key] = g.score

        self._anim_step = 0
        if self._anim_timer is not None:
            self._anim_timer.stop()
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_timer.start(20)

    def _anim_tick(self) -> None:
        """EaseOutCubic animation step."""
        self._anim_step += 1
        t_norm = self._anim_step / self._ANIM_STEPS
        eased = 1.0 - (1.0 - min(t_norm, 1.0)) ** 3
        done = self._anim_step >= self._ANIM_STEPS
        for g in self._goals:
            start = self._anim_scores[g.key]
            target = self._target_scores[g.key]
            g.score = target if done else start + (target - start) * eased
        if done:
            self._anim_timer.stop()
            self._anim_timer = None
        self.update()

    def reset(self) -> None:
        """Setzt alle Scores zurück (vor erster Restaurierung)."""
        import copy

        self._goals = copy.deepcopy(DEFAULT_GOALS)
        self._has_data = False
        self.update()

    # ──────────────────────────── Vektor-Icons ────────────────────────────

    @staticmethod
    def _draw_icon(painter: QPainter, cx: float, cy: float, r: float, kind: str, color: QColor) -> None:
        """Draw a crisp vector status icon centered at (cx, cy) with radius r.

        kind: 'pass' | 'warn' | 'fail' | 'na' | 'synth'
        """
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if kind == "pass":
            # Filled circle + white checkmark
            bg = QColor(color)
            bg.setAlpha(210)
            painter.setBrush(QBrush(bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            pen = QPen(QColor(255, 255, 255, 240), max(1.2, r * 0.3))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Checkmark path  ✓
            painter.drawLine(
                QPointF(cx - r * 0.38, cy + r * 0.05),
                QPointF(cx - r * 0.08, cy + r * 0.38),
            )
            painter.drawLine(
                QPointF(cx - r * 0.08, cy + r * 0.38),
                QPointF(cx + r * 0.42, cy - r * 0.32),
            )

        elif kind == "warn":
            # Filled rounded triangle + white exclamation mark
            bg = QColor(color)
            bg.setAlpha(210)
            tri = QPolygonF(
                [
                    QPointF(cx, cy - r * 0.92),
                    QPointF(cx - r * 0.88, cy + r * 0.62),
                    QPointF(cx + r * 0.88, cy + r * 0.62),
                ]
            )
            painter.setBrush(QBrush(bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(tri)
            pen = QPen(QColor(255, 255, 255, 240), max(1.2, r * 0.28))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(cx, cy - r * 0.28),
                QPointF(cx, cy + r * 0.12),
            )
            painter.drawPoint(QPointF(cx, cy + r * 0.35))

        elif kind == "fail":
            # Filled circle + white X
            bg = QColor(color)
            bg.setAlpha(210)
            painter.setBrush(QBrush(bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            pen = QPen(QColor(255, 255, 255, 240), max(1.2, r * 0.3))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            d = r * 0.35
            painter.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
            painter.drawLine(QPointF(cx + d, cy - d), QPointF(cx - d, cy + d))

        elif kind == "synth":
            # Diamond shape (rotated square) with inner glow
            bg = QColor(color)
            bg.setAlpha(200)
            diamond = QPolygonF(
                [
                    QPointF(cx, cy - r),
                    QPointF(cx + r * 0.72, cy),
                    QPointF(cx, cy + r),
                    QPointF(cx - r * 0.72, cy),
                ]
            )
            painter.setBrush(QBrush(bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(diamond)
            # Inner star dot
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.drawEllipse(QPointF(cx, cy), r * 0.22, r * 0.22)

        else:  # na / default — grey circle with horizontal dash
            bg = QColor(color)
            bg.setAlpha(120)
            painter.setBrush(QBrush(bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), r, r)
            pen = QPen(QColor(255, 255, 255, 180), max(1.0, r * 0.28))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx - r * 0.4, cy), QPointF(cx + r * 0.4, cy))

        painter.restore()

    # ──────────────────────────── Geometrie ───────────────────────────────

    def _row_y(self, idx: int) -> float:
        """Y-Koordinate der Oberkante von Zeile idx."""
        return self._PAD_TOP + idx * (self._ROW_H + self._ROW_GAP)

    def _bar_rect(self, idx: int) -> QRectF:
        """Rechteck für den Balkenbereich (Hintergrund) von Zeile idx."""
        x = self._PAD_X + self._ICON_W + self._LABEL_W + 4
        bar_w = self.width() - x - self._SCORE_W - self._PAD_X
        y = self._row_y(idx) + 2
        return QRectF(x, y, max(bar_w, 40), self._ROW_H - 4)

    # ──────────────────────────── Zeichnen ───────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if not self._has_data:
            self._draw_placeholder(p)
        else:
            self._draw_header(p)
            self._draw_bars(p)
            self._draw_footer_legend(p)

        p.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Zeigt Platzhaltertext wenn noch keine Restaurierung stattgefunden hat."""
        w, h = float(self.width()), float(self.height())
        # Sanfte Linie als Rahmen-Hint
        painter.setPen(QPen(QColor(80, 100, 140, 60), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(4, 4, w - 8, h - 8), 8, 8)

        # Icon
        font_icon = QFont("Segoe UI", 20)
        painter.setFont(font_icon)
        painter.setPen(QPen(QColor(80, 100, 140, 100)))
        painter.drawText(QRectF(0, h / 2 - 54, w, 40), Qt.AlignmentFlag.AlignCenter, "🎵")

        # Haupttext
        font_main = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_main)
        painter.setPen(QPen(QColor(130, 150, 190, 180)))
        painter.drawText(
            QRectF(10, h / 2 - 14, w - 20, 22),
            Qt.AlignmentFlag.AlignCenter,
            "Noch nicht gemessen",
        )

        # Subtext
        font_sub = QFont("Segoe UI", 7)
        painter.setFont(font_sub)
        painter.setPen(QPen(QColor(100, 120, 160, 140)))
        painter.drawText(
            QRectF(10, h / 2 + 10, w - 20, 36),
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
            "Nach der Restaurierung werden hier\nalle 14 Klangziele bewertet.",
        )

        # Goal-Namen als gedimmte Vorschau
        font_prev = QFont("Segoe UI", 6)
        painter.setFont(font_prev)
        painter.setPen(QPen(QColor(80, 95, 130, 80)))
        names = [g.label for g in self._goals]
        half = len(names) // 2
        col_w = (w - 20) / 2
        for col, chunk in enumerate([names[:half], names[half:]]):
            for row, name in enumerate(chunk):
                x = 10 + col * col_w
                y = h / 2 + 48 + row * 10
                painter.drawText(
                    QRectF(x, y, col_w, 10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"· {name}"
                )

    def _draw_header(self, painter: QPainter) -> None:
        """Kopfzeile: Statusübersicht (✅ N  ⚠ N  ✗ N)."""
        n_pass = n_warn = n_fail = n_na = 0
        for g in self._goals:
            if not g.applicable:
                n_na += 1
                continue
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
            if g.score >= t + 0.04:
                n_pass += 1
            elif g.score >= t:
                n_warn += 1
            else:
                n_fail += 1

        w = float(self.width())
        font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font)

        parts: list[tuple[str, str, QColor]] = [
            (str(n_pass), "pass", COLOR_PASS),
            (str(n_warn), "warn", COLOR_WARN),
            (str(n_fail), "fail", COLOR_FAIL),
        ]
        if n_na > 0:
            parts.append((str(n_na), "na", COLOR_NA))

        x = float(self._PAD_X)
        y_top = 6.0
        h_row = 18.0
        _icon_r = 5.0
        for label, kind, col in parts:
            self._draw_icon(painter, x + _icon_r, y_top + h_row * 0.5, _icon_r, kind, col)
            painter.setPen(QPen(col))
            painter.drawText(
                QRectF(x + _icon_r * 2 + 3, y_top, 32, h_row),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            x += 48
            x += 52

        # Thin separator line
        painter.setPen(QPen(QColor(80, 100, 150, 60), 1))
        painter.drawLine(QPointF(self._PAD_X, self._PAD_TOP - 4), QPointF(w - self._PAD_X, self._PAD_TOP - 4))

    def _draw_bars(self, painter: QPainter) -> None:
        """Zeichnet alle 14 Goal-Balken."""
        font_label = QFont("Segoe UI", 6, QFont.Weight.Bold)
        font_score = QFont("Segoe UI", 6)

        for idx, g in enumerate(self._goals):
            y = self._row_y(idx)
            bar_rect = self._bar_rect(idx)
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold

            # Determine status color
            if not g.applicable:
                color = COLOR_NA
                bar_bg = _BAR_BG_NA
            elif g.synthesized:
                color = COLOR_SYNTH
                bar_bg = QColor(200, 100, 230, 35)
            elif g.score >= t + 0.04:
                color = COLOR_PASS
                bar_bg = _BAR_BG_PASS
            elif g.score >= t:
                color = COLOR_WARN
                bar_bg = _BAR_BG_WARN
            else:
                color = COLOR_FAIL
                bar_bg = _BAR_BG_FAIL

            # Hover highlight
            is_hovered = idx == self._hovered_idx
            if is_hovered:
                painter.setBrush(QBrush(QColor(255, 255, 255, 12)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(
                    QRectF(self._PAD_X, y - 1, self.width() - 2 * self._PAD_X, self._ROW_H + 2), 3, 3
                )

            # ── Status icon (vector) ──
            if g.synthesized:
                _ik = "synth"
            elif not g.applicable:
                _ik = "na"
            elif g.score >= t + 0.04:
                _ik = "pass"
            elif g.score >= t:
                _ik = "warn"
            else:
                _ik = "fail"
            _icx = self._PAD_X + self._ICON_W * 0.5
            _icy = y + self._ROW_H * 0.5
            _ir = min(self._ICON_W, self._ROW_H) * 0.38
            self._draw_icon(painter, _icx, _icy, _ir, _ik, color)

            # ── Goal name ──
            painter.setFont(font_label)
            label_color = color if g.applicable else COLOR_DIM
            painter.setPen(QPen(label_color))
            name_text = g.label if g.applicable else f"({g.label})"
            painter.drawText(
                QRectF(self._PAD_X + self._ICON_W, y, self._LABEL_W, self._ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                name_text,
            )

            # ── Bar background ──
            painter.setBrush(QBrush(bar_bg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 3, 3)

            # ── Bar fill ──
            if g.applicable and g.score > 0.001:
                fill_w = bar_rect.width() * min(1.0, g.score)
                fill_rect = QRectF(bar_rect.x(), bar_rect.y(), fill_w, bar_rect.height())
                fill_color = QColor(color)
                fill_color.setAlpha(180)
                painter.setBrush(QBrush(fill_color))
                painter.drawRoundedRect(fill_rect, 3, 3)

            # ── Threshold marker (vertical line) ──
            if g.applicable:
                thresh_x = bar_rect.x() + bar_rect.width() * min(1.0, t)
                painter.setPen(QPen(COLOR_THRESH, 1.5))
                painter.drawLine(
                    QPointF(thresh_x, bar_rect.y() - 1),
                    QPointF(thresh_x, bar_rect.y() + bar_rect.height() + 1),
                )

            # ── Score value ──
            painter.setFont(font_score)
            painter.setPen(QPen(color if g.applicable else COLOR_DIM))
            score_str = f"{int(g.score * 100)} %" if g.applicable else "—"
            score_x = bar_rect.x() + bar_rect.width() + 3
            painter.drawText(
                QRectF(score_x, y, self._SCORE_W, self._ROW_H),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                score_str,
            )

    def _draw_footer_legend(self, painter: QPainter) -> None:
        """Legende am unteren Rand."""
        y = float(self.height()) - self._PAD_BOT + 4
        font = QFont("Segoe UI", 6)
        painter.setFont(font)

        items = [
            (COLOR_PASS, "Erfüllt"),
            (COLOR_WARN, "Knapp"),
            (COLOR_FAIL, "Nicht erreicht"),
            (COLOR_THRESH, "Zielwert"),
        ]
        x = float(self._PAD_X)
        for color, text in items:
            # Color dot
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(x, y + 2, 7, 7))
            painter.setPen(QPen(QColor(160, 170, 190, 180)))
            painter.drawText(
                QRectF(x + 10, y, 58, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
            )
            x += 66

    # ──────────────────────────── Tooltip / Hover ─────────────────────────

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._has_data:
            super().mouseMoveEvent(event)
            return

        py = float(event.pos().y())
        found = -1
        for idx in range(len(self._goals)):
            row_y = self._row_y(idx)
            if row_y <= py <= row_y + self._ROW_H:
                found = idx
                break

        if found != self._hovered_idx:
            self._hovered_idx = found
            self.update()

        if found >= 0:
            QToolTip.showText(event.globalPos(), self._tooltip_text(self._goals[found]), self)
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered_idx = -1
        self.update()
        super().leaveEvent(event)

    @staticmethod
    def _tooltip_text(g: GoalEntry) -> str:
        """Laienverständlicher Tooltip-Text für ein Goal."""
        # Human-readable descriptions per goal
        _DESCRIPTIONS: dict[str, str] = {
            "brillanz": "Helligkeit und Klarheit in den Höhen (Obertöne, Luftigkeit).",
            "waerme": "Wärme und Fülle im Bassbereich — angenehmes Klangfundament.",
            "natuerlichkeit": "Klingt die Aufnahme natürlich und unverarbeitet?",
            "authentizitaet": "Entspricht der Klang dem Original-Charakter der Aufnahme?",
            "emotionalitaet": "Transportiert die Restaurierung die emotionale Intensität?",
            "transparenz": "Sind alle Instrumente klar voneinander trennbar?",
            "bass_kraft": "Kraft und Tiefe im Bass inkl. fehlende Grundtöne.",
            "groove": "Rhythmisches Timing — keine Verschmierung von Transienten.",
            "spatial_depth": "Stereobreite und Raumtiefe (nur bei Stereo-Aufnahmen).",
            "timbre_authentizitaet": "Klangfarbe der Instrumente — klingt die Oboe noch wie eine Oboe?",
            "tonal_center": "Tonart und Stimmung — kein Pitch-Shift durch die Restaurierung.",
            "micro_dynamics": "Lautstärke-Feinstruktur — Atemzüge, Pianissimo-Stellen.",
            "separation_fidelity": "Stimme und Instrumente bleiben getrennt (kein Matsch).",
            "artikulation": "Ansätze, Transienten und Konsonanten bleiben scharf.",
        }
        lines: list[str] = [f"<b>{g.label}</b>"]
        if _desc := _DESCRIPTIONS.get(g.key, ""):
            lines.append(f"<br><i>{_desc}</i>")

        if not g.applicable:
            lines.append("<br><br>⚪ <i>Nicht messbar für diese Aufnahme.</i>")
            if g.inapplicable_reason:
                lines.append(f"<br>{g.inapplicable_reason}")
            return "".join(lines)

        t_eff = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
        score_pct = int(g.score * 100)
        thresh_pct = int(t_eff * 100)

        if g.score >= t_eff + 0.04:
            status = "✅ Exzellent — deutlich über dem Zielwert"
            color = "#4CAF50"
        elif g.score >= t_eff:
            status = "🟡 Bestanden — knapp am Zielwert"
            color = "#FFC107"
        else:
            status = "❌ Unter Zielwert"
            color = "#F44336"

        lines.append(f'<br><br><span style="color:{color}"><b>{status}</b></span>')
        lines.append(f"<br>Erreicht: <b>{score_pct} %</b> &nbsp;|&nbsp; Zielwert: <b>{thresh_pct} %</b>")

        if g.adaptive_threshold >= 0 and abs(g.adaptive_threshold - g.threshold) > 0.005:
            orig_pct = int(g.threshold * 100)
            diff = int((g.adaptive_threshold - g.threshold) * 100)
            sign = "+" if diff >= 0 else ""
            lines.append(f"<br><i>Zielwert angepasst: {sign}{diff} % gegenüber Standard ({orig_pct} %)</i>")
        if g.adaptation_reason:
            lines.append(f"<br>{g.adaptation_reason}")
        if g.synthesized:
            lines.append("<br>✦ <i>Frequenzbereich wurde era-authentisch ergänzt.</i>")

        return "".join(lines)

    def sizeHint(self) -> QSize:
        total_h = self._PAD_TOP + len(DEFAULT_GOALS) * (self._ROW_H + self._ROW_GAP) + self._PAD_BOT
        return QSize(300, total_h)

    def minimumSizeHint(self) -> QSize:
        total_h = self._PAD_TOP + len(DEFAULT_GOALS) * (self._ROW_H + self._ROW_GAP) + self._PAD_BOT
        return QSize(220, total_h)


def apply_restoration_result(
    widget: MusicalGoalsRadarWidget,
    result: object,
) -> None:
    """
    Liest ein RestorationResult-Objekt aus und aktualisiert das Radar-Widget.
    Versteht alle in der Spec definierten Felder (graceful degradation wenn fehlen).
    """
    # 1. Musical Goals Scores
    scores: dict[str, float] = {}
    if hasattr(result, "musical_goals") and result.musical_goals:
        mg = result.musical_goals
        scores = mg if isinstance(mg, dict) else {}

    # 2. Adaptive Thresholds
    adaptive: dict[str, float] = {}
    if hasattr(result, "adaptive_thresholds") and result.adaptive_thresholds:
        at = result.adaptive_thresholds
        if hasattr(at, "thresholds"):
            adaptive = at.thresholds
        elif isinstance(at, dict):
            adaptive = at

    # 3. Goal Applicability
    applicable: set[str] | None = None
    inapplicable_reasons: dict[str, str] = {}
    if hasattr(result, "goal_applicability") and result.goal_applicability:
        ga = result.goal_applicability
        if hasattr(ga, "applicable"):
            applicable = set(ga.applicable)
        if hasattr(ga, "reasons"):
            inapplicable_reasons = ga.reasons or {}

    # 4. Synthesierte Ziele (EraAuthentic)
    synthesized: set[str] = set()
    if hasattr(result, "genealogy") and result.genealogy:
        gen = result.genealogy
        if hasattr(gen, "operations"):
            for op in gen.operations:
                if hasattr(op, "operation_type") and "synthesize" in str(op.operation_type):
                    # Brillanz ist die typische synthesierte Achse
                    synthesized.add("brillanz")

    # 5. Adaptation Reasons
    adapt_reasons: dict[str, str] = {}
    if hasattr(result, "adaptive_thresholds") and result.adaptive_thresholds:
        at = result.adaptive_thresholds
        if hasattr(at, "adaptations"):
            adapt_reasons = at.adaptations or {}

    widget.update_scores(
        scores=scores,
        adaptive_thresholds=adaptive if adaptive else None,
        applicable_goals=applicable,
        inapplicable_reasons=inapplicable_reasons,
        synthesized_goals=synthesized if synthesized else None,
        adaptation_reasons=adapt_reasons if adapt_reasons else None,
    )
