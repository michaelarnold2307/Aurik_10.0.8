"""
Musical Goals Radar Chart Widget für Aurik 9
Zeigt alle 14 musikalischen Qualitätsziele als Spinnen-/Radar-Diagramm.

Rein PyQt5-basiert (kein Matplotlib) — CPU-leicht, sofort responsiv.
Vollständig laienfreundlich: Deutsche Labels, Farb-Kodierung, Tooltips.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
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
    label: str  # Kurzbezeichnung (Deutsch, max. 12 Zeichen)
    threshold: float  # Pflicht-Schwellwert laut Spec
    score: float = 0.0  # aktueller Wert ∈ [0, 1]
    adaptive_threshold: float = -1.0  # adaptierter Schwellwert (–1 = nicht gesetzt)
    applicable: bool = True  # GoalApplicabilityFilter
    inapplicable_reason: str = ""  # Deutsch: Warum nicht messbar?
    synthesized: bool = False  # EraAuthenticPerceptualCompletion aktiv?
    adaptation_reason: str = ""  # Deutsch: Warum Schwellwert angepasst?


# Standard-Goals gemäß §1.2 der Spec
DEFAULT_GOALS: list[GoalEntry] = [
    GoalEntry("brillanz", "Brillanz", 0.85),
    GoalEntry("waerme", "Wärme", 0.80),
    GoalEntry("natuerlichkeit", "Natürl.", 0.90),
    GoalEntry("authentizitaet", "Authent.", 0.88),
    GoalEntry("emotionalitaet", "Emotion.", 0.87),
    GoalEntry("transparenz", "Transp.", 0.89),
    GoalEntry("bass_kraft", "Bass-Kraft", 0.85),
    GoalEntry("groove", "Groove", 0.88),
    GoalEntry("spatial_depth", "Raumtiefe", 0.75),
    GoalEntry("timbre_authentizitaet", "Timbre", 0.87),
    GoalEntry("tonal_center", "Ton.Zentrum", 0.95),
    GoalEntry("micro_dynamics", "Mikro-Dyn.", 0.92),
    GoalEntry("separation_fidelity", "Separation", 0.82),
    GoalEntry("artikulation", "Artikulat.", 0.85),
]

# Farb-Konstanten
COLOR_PASS = QColor(76, 175, 80, 220)  # Grün — Ziel erfüllt
COLOR_WARN = QColor(255, 193, 7, 220)  # Gelb — Ziel nah am Limit (< +0.04)
COLOR_FAIL = QColor(244, 67, 54, 220)  # Rot   — Ziel unterschritten
COLOR_NA = QColor(120, 130, 150, 120)  # Grau  — nicht anwendbar
COLOR_THRESHOLD = QColor(255, 255, 255, 70)  # Weiß-halbtransparent — Schwellwert-Ring
COLOR_FILL_PASS = QColor(76, 175, 80, 55)  # Füllung Haupt-Polygon
COLOR_FILL_FAIL = QColor(244, 67, 54, 40)  # Füllung wenn ein Ziel fehlt
COLOR_GRID = QColor(255, 255, 255, 18)  # Gitternetz
COLOR_BG = QColor(15, 18, 30, 0)  # Hintergrund transparent
COLOR_SYNTH = QColor(240, 147, 251, 200)  # Lila — synthesiert/ergänzt


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Widget
# ─────────────────────────────────────────────────────────────────────────────


class MusicalGoalsRadarWidget(QWidget):
    """
    14-Punkte Radar-Chart für Musical Goals.

    Zeigt:
    - Ausgefülltes Polygon der aktuellen Scores (grün/gelb/rot je nach Status)
    - Strichliertes Polygon der (adaptiven) Schwellwerte
    - Grau ausgeblendete Achsen für nicht-anwendbare Ziele
    - (✦) Markierung an der Brillanz-Achse wenn EraAuthenticPerceptualCompletion aktiv
    - Tooltips beim Hover über die Achsenpunkte (Deutsch, laienverständlich)
    - Konzentrische Gitternetz-Ringe (0.25 / 0.50 / 0.75 / 1.0)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

        # Arbeits-Kopie der Goals (wird per update_scores() befüllt)
        import copy

        self._goals: list[GoalEntry] = copy.deepcopy(DEFAULT_GOALS)
        self._hovered_idx: int = -1  # Index des gehoverten Goals
        self._show_legend: bool = True

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
        """
        Aktualisiert alle Score-Werte und löst ein Neuzeichnen aus.

        Args:
            scores:              goal_key → Score ∈ [0, 1]
            adaptive_thresholds: goal_key → adaptierter Schwellwert (optional)
            applicable_goals:    Menge aktiver Ziele (None = alle aktiv)
            inapplicable_reasons: goal_key → Deutsche Erklärung warum inaktiv
            synthesized_goals:   Set von goal_keys mit synthetisierten Daten (✦)
            adaptation_reasons:  goal_key → Deutsche Erklärung der Anpassung
        """
        adaptive_thresholds = adaptive_thresholds or {}
        applicable = applicable_goals  # None = alle anwendbar
        inapplicable_reasons = inapplicable_reasons or {}
        synthesized = synthesized_goals or set()
        adapt_reasons = adaptation_reasons or {}

        for g in self._goals:
            g.score = float(scores.get(g.key, 0.0))
            g.applicable = (applicable is None) or (g.key in applicable)
            g.inapplicable_reason = inapplicable_reasons.get(g.key, "")
            g.synthesized = g.key in synthesized
            if g.key in adaptive_thresholds:
                g.adaptive_threshold = float(adaptive_thresholds[g.key])
            else:
                g.adaptive_threshold = -1.0
            g.adaptation_reason = adapt_reasons.get(g.key, "")

        self.update()  # Neuzeichnen anfordern

    def reset(self) -> None:
        """Setzt alle Scores auf 0 zurück (vor erster Restaurierung)."""
        import copy

        self._goals = copy.deepcopy(DEFAULT_GOALS)
        self.update()

    # ──────────────────────────── Geometrie ───────────────────────────────

    def _center(self) -> QPointF:
        # Vertikaler Mittelpunkt lässt unten 72 px für die Legende frei
        legend_h = 72.0 if self._show_legend else 0.0
        usable_h = self.height() - legend_h
        return QPointF(self.width() / 2.0, usable_h / 2.0)

    def _radius(self) -> float:
        margin = 50.0  # Platz für Labels außen
        legend_h = 72.0 if self._show_legend else 0.0
        usable_h = self.height() - legend_h
        return min(self.width(), usable_h) / 2.0 - margin

    def _angle_for(self, idx: int) -> float:
        """Winkel (Grad) für Goal-Achse idx — Start oben, im Uhrzeigersinn."""
        n = len(self._goals)
        return -90.0 + idx * 360.0 / n

    def _point_on_ring(self, idx: int, value: float) -> QPointF:
        """Kartesischer Punkt auf Achse idx bei normalisiertem Wert value ∈ [0,1]."""
        cx, cy = self._center().x(), self._center().y()
        r = self._radius()
        angle_rad = math.radians(self._angle_for(idx))
        return QPointF(cx + value * r * math.cos(angle_rad), cy + value * r * math.sin(angle_rad))

    # ──────────────────────────── Zeichnen ───────────────────────────────

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        self._draw_grid(p)
        self._draw_threshold_ring(p)
        self._draw_score_polygon(p)
        self._draw_axes(p)
        self._draw_labels(p)
        if self._show_legend:
            self._draw_legend(p)

        p.end()

    def _draw_grid(self, painter: QPainter) -> None:
        """Konzentrische Gitter-Ringe bei 0.25, 0.50, 0.75, 1.00."""
        n = len(self._goals)
        cx, cy = self._center().x(), self._center().y()
        self._radius()

        pen = QPen(COLOR_GRID, 0.8, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for ring in (0.25, 0.50, 0.75, 1.00):
            polygon = QPolygonF()
            for i in range(n):
                polygon.append(self._point_on_ring(i, ring))
            painter.drawPolygon(polygon)

    def _draw_threshold_ring(self, painter: QPainter) -> None:
        """Gestrichelter Polygon-Ring für Schwellwerte (weiß, halbtransparent)."""
        pen = QPen(COLOR_THRESHOLD, 1.6, Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 3])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        polygon = QPolygonF()
        for i, g in enumerate(self._goals):
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
            t = max(0.0, min(1.0, t))
            polygon.append(self._point_on_ring(i, t))
        painter.drawPolygon(polygon)

    def _draw_score_polygon(self, painter: QPainter) -> None:
        """Ausgefülltes Polygon der aktuellen Scores mit Farbverlauf."""
        polygon = QPolygonF()
        all_pass = True
        for i, g in enumerate(self._goals):
            score = g.score if g.applicable else 0.0
            t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
            if g.applicable and score < t:
                all_pass = False
            polygon.append(self._point_on_ring(i, max(0.0, min(1.0, score))))

        fill = COLOR_FILL_PASS if all_pass else COLOR_FILL_FAIL
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(COLOR_PASS if all_pass else COLOR_FAIL, 1.8))
        painter.drawPolygon(polygon)

        # Einzelne Punkte auf den Achsen
        for i, g in enumerate(self._goals):
            if not g.applicable:
                c = COLOR_NA
            else:
                score = g.score
                t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
                if score >= t + 0.04:
                    c = COLOR_PASS
                elif score >= t:
                    c = COLOR_WARN
                else:
                    c = COLOR_FAIL
            # Synthesiert → lila übermalen
            if g.synthesized:
                c = COLOR_SYNTH

            pt = self._point_on_ring(i, max(0.0, min(1.0, g.score if g.applicable else 0.0)))
            dot_r = 5.0 if i != self._hovered_idx else 8.0
            painter.setBrush(QBrush(c))
            painter.setPen(QPen(c.lighter(140), 1.0))
            painter.drawEllipse(pt, dot_r, dot_r)

    def _draw_axes(self, painter: QPainter) -> None:
        """Achsenlinien vom Zentrum zu den Ecken."""
        cx, cy = self._center().x(), self._center().y()
        center = QPointF(cx, cy)

        for i, g in enumerate(self._goals):
            end = self._point_on_ring(i, 1.0)
            color = COLOR_NA if not g.applicable else QColor(255, 255, 255, 45)
            pen = QPen(color, 0.8, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(center, end)

    def _draw_labels(self, painter: QPainter) -> None:
        """Label jeder Achse außerhalb des Rings — mit Qualitäts-Farbkodierung."""
        r_label = self._radius() + 12.0

        font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(font)

        cx, cy = self._center().x(), self._center().y()
        w, h = float(self.width()), float(self.height())

        for i, g in enumerate(self._goals):
            angle_rad = math.radians(self._angle_for(i))
            lx = cx + r_label * math.cos(angle_rad)
            ly = cy + r_label * math.sin(angle_rad)

            # Farbwahl
            if not g.applicable:
                color = QColor(120, 130, 150, 160)
            elif g.synthesized:
                color = COLOR_SYNTH
            else:
                t = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
                if g.score >= t + 0.04:
                    color = COLOR_PASS
                elif g.score >= t:
                    color = COLOR_WARN
                else:
                    color = COLOR_FAIL

            painter.setPen(QPen(color))

            # Textausrichtung je nach Position
            text = g.label
            if g.synthesized:
                text = f"{g.label}✦"
            if not g.applicable:
                text = f"({g.label})"

            rect_w, rect_h = 76, 26
            # Links/rechts/oben/unten ausrichten
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            if cos_a < -0.3:
                tx = lx - rect_w
            elif cos_a > 0.3:
                tx = lx
            else:
                tx = lx - rect_w / 2.0
            if sin_a < -0.3:
                ty = ly - rect_h
            elif sin_a > 0.3:
                ty = ly
            else:
                ty = ly - rect_h / 2.0

            # Clamp an Widget-Grenzen (verhindert Abschneiden)
            margin = 2.0
            tx = max(margin, min(w - rect_w - margin, tx))
            ty = max(margin, min(h - rect_h - 14.0 - margin, ty))

            painter.drawText(QRectF(tx, ty, rect_w, rect_h), Qt.AlignmentFlag.AlignCenter, text)

            # Score-Wert klein darunter/darüber
            score_text = f"{g.score:.2f}" if g.applicable else "—"
            font_small = QFont("Segoe UI", 6)
            painter.setFont(font_small)
            painter.setPen(QPen(color.lighter(120)))
            painter.drawText(QRectF(tx, ty + 14, rect_w, 14), Qt.AlignmentFlag.AlignCenter, score_text)
            painter.setFont(font)  # zurücksetzen

    def _draw_legend(self, painter: QPainter) -> None:
        """Kleine Legende unten links: Farbkodierung."""
        x, y = 10.0, self.height() - 70.0
        font = QFont("Segoe UI", 7)
        painter.setFont(font)
        items = [
            (COLOR_PASS, "Ziel erfüllt"),
            (COLOR_WARN, "Knapp am Limit"),
            (COLOR_FAIL, "Ziel nicht erreicht"),
            (COLOR_NA, "Nicht messbar"),
            (COLOR_SYNTH, "Ergänzt (✦)"),
            (COLOR_THRESHOLD, "Schwellwert"),
        ]
        for color, text in items:
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(x, y, 8, 8))
            painter.setPen(QPen(QColor(180, 190, 210)))
            painter.drawText(
                QRectF(x + 12, y - 1, 130, 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text
            )
            y += 11.0

    # ──────────────────────────── Tooltip / Hover ─────────────────────────

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = QPointF(event.pos())
        found = -1
        best_dist = 25.0  # Pixel-Fangradius

        for i in range(len(self._goals)):
            score = self._goals[i].score if self._goals[i].applicable else 0.0
            pt = self._point_on_ring(i, max(0.0, min(1.0, score)))
            dx = pos.x() - pt.x()
            dy = pos.y() - pt.y()
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                found = i

        if found != self._hovered_idx:
            self._hovered_idx = found
            self.update()

        if found >= 0:
            g = self._goals[found]
            QToolTip.showText(event.globalPos(), self._tooltip_text(g), self)
        else:
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered_idx = -1
        self.update()
        super().leaveEvent(event)

    @staticmethod
    def _tooltip_text(g: GoalEntry) -> str:
        """Laienverständlicher Tooltip-Text für ein Goal."""
        lines: list[str] = []
        lines.append(f"<b>{g.label}</b>")

        if not g.applicable:
            lines.append("<br><i>⚪ Nicht messbar für diese Aufnahme</i>")
            if g.inapplicable_reason:
                lines.append(f"<br>{g.inapplicable_reason}")
            return "".join(lines)

        t_eff = g.adaptive_threshold if g.adaptive_threshold >= 0 else g.threshold
        score_pct = int(g.score * 100)
        thresh_pct = int(t_eff * 100)

        # Status
        if g.score >= t_eff + 0.04:
            status = "✅ Exzellent"
            color = "#4CAF50"
        elif g.score >= t_eff:
            status = "🟡 Knapp erfüllt"
            color = "#FFC107"
        else:
            status = "❌ Unter Ziel"
            color = "#F44336"

        lines.append(f'<br><span style="color:{color}"><b>{status}</b></span>')
        lines.append(f"<br>Score: <b>{score_pct} %</b>  |  Ziel: <b>{thresh_pct} %</b>")

        if g.adaptive_threshold >= 0 and abs(g.adaptive_threshold - g.threshold) > 0.005:
            orig_pct = int(g.threshold * 100)
            lines.append(f"<br><i>Ziel angepasst (Original: {orig_pct} %)</i>")
        if g.adaptation_reason:
            lines.append(f"<br>{g.adaptation_reason}")

        if g.synthesized:
            lines.append("<br>✦ <i>Frequenzbereich wurde era-authentisch ergänzt</i>")

        return "".join(lines)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(380, 380)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(280, 280)


# ─────────────────────────────────────────────────────────────────────────────
# Hilfs-Funktion: RestorationResult → GoalEntry-Updates
# ─────────────────────────────────────────────────────────────────────────────


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
