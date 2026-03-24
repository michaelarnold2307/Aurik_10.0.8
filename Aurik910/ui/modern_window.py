"""
AURIK Professional - Modern Frameless Window mit Premium Look & Feel
Basiert auf PyQt5 mit Custom-Styling, Glassmorphism und Animationen
Mit integrierter Audio-Verarbeitung (Backend V2)
"""

import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

# ── Bridge: einzige erlaubte Schnittstelle zu backend/core/ (§11 Spec 08) ──
# Frontend importiert Core-Module AUSSCHLIEßLICH über diese Bridge.
try:
    from backend.api.bridge import (
        cache_defect_result,
        cache_era_genre_result,
        cache_medium_result,
        cache_restorability_result,
        clear_defect_cache as _bridge_clear_defect_cache,
        clear_era_genre_cache as _bridge_clear_era_genre_cache,
        export_guard as _export_guard,
        get_audio_exporter_class as _bridge_get_audio_exporter_class,
        get_audio_file_validator as _bridge_get_audio_file_validator,
        get_aurik_denker_class as _bridge_get_aurik_denker_class,
        get_aurik_denker_instance as _bridge_get_aurik_denker_instance,
        get_cached_defect_result,
        get_cached_era_genre_result,
        get_cached_medium_result,
        get_cached_restorability_result,
        get_carrier_forensics_fn as _bridge_get_carrier_forensics_fn,
        get_cleanup_after_file_fn as _bridge_get_cleanup_after_file_fn,
        get_defect_scanner as _bridge_get_defect_scanner,
        get_defect_type as _bridge_get_defect_type,
        get_era_classifier_fn as _bridge_get_era_classifier_fn,
        get_genre_classifier_fn as _bridge_get_genre_classifier_fn,
        get_lyrics_guided_enhancement_fn as _bridge_get_lyrics_guided_enhancement,
        get_medium_classifier_fn as _bridge_get_medium_classifier_fn,
        get_restorability_estimator_class as _bridge_get_restorability_estimator_class,
        resolve_pipeline_fail_reason as _bridge_resolve_pipeline_fail_reason,
        validate_export_quality as _validate_export_quality,
        warmup_models_background as _warmup_models_background,
    )
    from backend.api.bridge import (
        normalize_pipeline_health_state as _bridge_normalize_pipeline_health_state,  # type: ignore[assignment]
    )

    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

    def _export_guard(audio):  # type: ignore[misc]
        import numpy as _np

        audio = _np.asarray(audio, dtype=_np.float32)
        audio = _np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return _np.clip(audio, -1.0, 1.0)

    def _validate_export_quality(result: object) -> tuple:  # type: ignore[misc]
        return True, []

    def cache_defect_result(file_path: str, result: object) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar, Defekt-Cache deaktiviert

    def get_cached_defect_result(file_path: str) -> object | None:  # type: ignore[misc]
        return None

    def cache_era_genre_result(
        file_path: str, era_result: object | None = None, genre_result: object | None = None
    ) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar

    def get_cached_era_genre_result(file_path: str) -> dict | None:  # type: ignore[misc]
        return None

    def cache_medium_result(file_path: str, result: object) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar

    def get_cached_medium_result(file_path: str) -> object | None:  # type: ignore[misc]
        return None

    def cache_restorability_result(file_path: str, result: object) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar

    def get_cached_restorability_result(file_path: str) -> object | None:  # type: ignore[misc]
        return None

    def _bridge_clear_era_genre_cache(file_path: str | None = None) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar

    def _warmup_models_background() -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar, kein Vorwärmen möglich

    # Bridge-Fallbacks: geben None zurück (§11.4 Bridge-Fallback)
    def _bridge_get_aurik_denker_class() -> type | None:  # type: ignore[misc]
        return None

    def _bridge_get_aurik_denker_instance() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_audio_file_validator() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_clear_defect_cache(file_path: str | None = None) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar, kein Cache zu leeren

    def _bridge_get_audio_exporter_class() -> type | None:  # type: ignore[misc]
        return None

    def _bridge_get_carrier_forensics_fn() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_defect_scanner() -> type | None:  # type: ignore[misc]
        return None

    def _bridge_get_defect_type() -> type | None:  # type: ignore[misc]
        return None

    def _bridge_get_era_classifier_fn() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_genre_classifier_fn() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_lyrics_guided_enhancement() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_medium_classifier_fn() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_cleanup_after_file_fn() -> object | None:  # type: ignore[misc]
        return None

    def _bridge_get_restorability_estimator_class() -> type | None:  # type: ignore[misc]
        return None

    def _bridge_normalize_pipeline_health_state(raw):  # type: ignore[misc,return-value]
        class _FallbackState:
            value = str(raw or "ok")

        return _FallbackState()  # type: ignore[return-value]

    def _bridge_resolve_pipeline_fail_reason(  # type: ignore[misc]
        *,
        typed_fail_reason=None,
        metadata=None,
        stage_notes=None,
        fail_reasons=None,
    ) -> str:
        _meta = metadata or {}
        _notes = stage_notes or {}
        _reason = typed_fail_reason or _meta.get("fail_reason", "") or _notes.get("fail_reason", "")
        if not _reason:
            _reasons = fail_reasons or _meta.get("fail_reasons") or []
            if isinstance(_reasons, list):
                for _entry in _reasons:
                    if not isinstance(_entry, dict):
                        continue
                    for _key in ("error_code", "exc_msg", "message"):
                        _cand = str(_entry.get(_key, "") or "").strip()
                        if _cand and _cand not in {"none", "None"}:
                            _reason = _cand
                            break
                    if _reason:
                        break
        _resolved = str(_reason or "").strip()
        if _resolved in {"", "none", "None"}:
            return ""
        return _resolved


import contextlib

from PyQt5.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRegion,
)
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import soundfile as sf

from Aurik910.i18n import get_language, set_language, t

# Optionaler Audio-Player (sounddevice) – Fallback falls nicht vorhanden
try:
    import sounddevice as _sd

    _SD_AVAILABLE = True
except Exception:
    _sd = None
    _SD_AVAILABLE = False

# Musical Goals Radar Chart (pure PyQt5, kein Matplotlib)
try:
    from Aurik910.ui.musical_goals_radar import MusicalGoalsRadarWidget, apply_restoration_result
except ImportError:
    try:
        from musical_goals_radar import MusicalGoalsRadarWidget, apply_restoration_result
    except ImportError:
        MusicalGoalsRadarWidget = None  # type: ignore[assignment]
        apply_restoration_result = None  # type: ignore[assignment]


# ── Quality Meter: circular arc gauge (MOS 0–5 → green/yellow/red) ──────────
class QualityMeterWidget(QWidget):
    """Horizontal VU-style quality bar displaying MOS quality (0–5 scale).

    Fills left→right with a red→yellow→green gradient.
    Always visible; shows placeholder text when not yet measured.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PyQt5.QtWidgets import QSizePolicy

        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mos: float = 0.0
        self._max_mos: float = 5.0
        self.setToolTip("Klangqualitäts-Anzeige (MOS 0–5)")

    def set_mos(self, mos: float) -> None:
        """Update meter to *mos* value (0–5) and repaint."""
        self._mos = float(max(0.0, min(self._max_mos, mos)))
        self.update()

    def reset(self) -> None:
        """Reset meter to empty state."""
        self._mos = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        m = 2

        # ── Container background ──────────────────────────────────────
        bg_rect = QRectF(m, m, w - 2 * m, h - 2 * m)
        p.setPen(QPen(QColor(50, 65, 100, 90), 1))
        p.setBrush(QBrush(QColor(16, 20, 38, 200)))
        p.drawRoundedRect(bg_rect, 3, 3)

        fraction = self._mos / self._max_mos  # 0.0–1.0

        # ── Gradient fill (left → right) ──────────────────────────────
        if fraction > 0.001:
            fill_w = (w - 2 * m) * fraction
            fill_rect = QRectF(m, m, fill_w, h - 2 * m)

            # Gradient spans full bar width so leading edge always shows color
            grad = QLinearGradient(m, 0, w - m, 0)
            grad.setColorAt(0.0, QColor(210, 55, 55, 210))
            grad.setColorAt(0.35, QColor(215, 175, 35, 210))
            grad.setColorAt(1.0, QColor(55, 195, 75, 210))

            clip_path = QPainterPath()
            clip_path.addRoundedRect(fill_rect, 3, 3)
            p.setClipPath(clip_path)
            p.fillRect(fill_rect, QBrush(grad))
            p.setClipping(False)

        # ── Tick marks at 1/5, 2/5, 3/5, 4/5 ────────────────────────
        p.setPen(QPen(QColor(40, 55, 90, 130), 1))
        inner_w = w - 2 * m
        for step in range(1, 5):
            tx = m + inner_w * step / 5
            p.drawLine(QPointF(tx, m + 2), QPointF(tx, h - m - 2))

        # ── Text overlay ──────────────────────────────────────────────
        font = p.font()
        font.setPointSize(7)
        if self._mos > 0.01:
            font.setBold(True)
            p.setFont(font)
            p.setPen(QPen(QColor(230, 235, 245)))
            p.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, f"Klang  {self._mos:.1f} / 5.0")
        else:
            font.setBold(False)
            p.setFont(font)
            p.setPen(QPen(QColor(75, 90, 115)))
            p.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, "Klangqualität: Noch nicht gemessen")

        p.end()


def _defect_analysis_to_display(scores: dict, status: str = "detected") -> dict:
    """Konvertiert DefectScanner.scan().scores (Dict[DefectType, DefectScore]) in Display-Dict.

    Skalierung Severity [0.0–1.0] → Widget-kompatible Typen:
        Integer-Felder (Zaehlwerte): clicks, crackle, pops, clipping, sibilance, dropout
        Float-Felder (physikalisch): hum [Hz], noise_level [dB], wow [%], flutter [%]
        Float-Felder (0–100 %): alle weiteren Defekttypen
    """
    DefectType = _bridge_get_defect_type()
    if DefectType is None:
        return {"status": status}

    def _sev(key) -> float:
        ds = scores.get(key)
        if ds is None:
            return 0.0
        return float(getattr(ds, "severity", 0.0) or 0.0)

    def _sev_opt(attr: str) -> float:
        dt = getattr(DefectType, attr, None)
        if dt is None:
            return 0.0
        return _sev(dt)

    sev_clicks = _sev(DefectType.CLICKS)
    sev_crackle = _sev(DefectType.CRACKLE)
    sev_clip = _sev(DefectType.CLIPPING)
    sev_hum = _sev(DefectType.HUM)
    sev_noise = _sev(DefectType.HIGH_FREQ_NOISE)
    sev_dropout = _sev(DefectType.DROPOUTS)
    sev_wow = _sev_opt("WOW")
    sev_flutter = _sev_opt("FLUTTER")
    sev_rumble = _sev(DefectType.LOW_FREQ_RUMBLE)
    sev_sibilance = _sev(DefectType.SIBILANCE)
    # Alle weiteren DefectTypes (mit _sev_opt für Rückwärtskompatibilität)
    sev_digital = _sev(DefectType.DIGITAL_ARTIFACTS)
    sev_compress = _sev(DefectType.COMPRESSION_ARTIFACTS)
    sev_stereo = _sev_opt("STEREO_IMBALANCE")
    sev_phase = _sev_opt("PHASE_ISSUES")
    sev_dc = _sev_opt("DC_OFFSET")
    sev_bw = _sev_opt("BANDWIDTH_LOSS")
    sev_pitch = _sev_opt("PITCH_DRIFT")
    sev_reverb = _sev_opt("REVERB_EXCESS")
    sev_print = _sev_opt("PRINT_THROUGH")
    sev_quant = _sev_opt("QUANTIZATION_NOISE")
    sev_jitter = _sev_opt("JITTER_ARTIFACTS")
    sev_dyncomp = _sev_opt("DYNAMIC_COMPRESSION_EXCESS")
    sev_pre_echo = _sev_opt("PRE_ECHO")
    sev_transient = _sev_opt("TRANSIENT_SMEARING")
    sev_head = _sev_opt("HEAD_WEAR")
    sev_riaa = _sev_opt("RIAA_CURVE_ERROR")
    sev_alias = _sev_opt("ALIASING")
    sev_bias = _sev_opt("BIAS_ERROR")
    sev_transport_bump = _sev_opt("TRANSPORT_BUMP")

    # Zeitpositionen (locations) pro Defekttyp für vertikale Marker in der Wellenform.
    # Format: {"clicks": [(t_start, t_end), ...], ...}
    def _locs(key) -> list:
        ds = scores.get(key)
        return list(ds.locations) if ds is not None and ds.locations else []

    def _locs_opt(attr: str) -> list:
        dt = getattr(DefectType, attr, None)
        if dt is None:
            return []
        return _locs(dt)

    def _ch_locs(key) -> dict:
        """Per-channel locations: {"L": [...], "R": [...]} if available."""
        ds = scores.get(key)
        if ds is None:
            return {}
        return dict(getattr(ds, "metadata", {}).get("channel_locations", {}))

    def _ch_locs_opt(attr: str) -> dict:
        dt = getattr(DefectType, attr, None)
        if dt is None:
            return {}
        return _ch_locs(dt)

    return {
        # Integer-Felder → skalierte Zaehlwerte (Widget: animate_int + {:,} Formatierung)
        "clicks": int(sev_clicks * 500),
        "crackle": int(sev_crackle * 500),
        "pops": int(sev_clip * 100),
        "clipping": int(sev_clip * 200),
        "sibilance": int(sev_sibilance * 300),
        "dropout": int(sev_dropout * 50),
        # Float-Felder → physikalisch skaliert (Widget: animate_float)
        "hum": round(sev_hum * 50.0, 2),  # 0–50 Hz
        "noise_level": round(sev_noise * 75.0, 2),  # 0–75 dB Rauschboden
        "wow": round(sev_wow * 3.0, 3),  # 0–3 % Tonhöhenschwankung (< 0.5 Hz)
        "flutter": round(sev_flutter * 3.0, 3),  # 0–3 % Tonhöhenschwankung (0.5–200 Hz)
        # Float-Felder (0–100 % Schwere) für defect_summary_label
        "rumble": round(sev_rumble * 100.0, 1),
        "digital_artifacts": round(sev_digital * 100.0, 1),
        "compression_artifacts": round(sev_compress * 100.0, 1),
        "stereo_imbalance": round(sev_stereo * 100.0, 1),
        "phase_issues": round(sev_phase * 100.0, 1),
        "dc_offset": round(sev_dc * 100.0, 1),
        "bandwidth_loss": round(sev_bw * 100.0, 1),
        "pitch_drift": round(sev_pitch * 100.0, 1),
        "reverb_excess": round(sev_reverb * 100.0, 1),
        "print_through": round(sev_print * 100.0, 1),
        "quantization_noise": round(sev_quant * 100.0, 1),
        "jitter_artifacts": round(sev_jitter * 100.0, 1),
        "dynamic_compression_excess": round(sev_dyncomp * 100.0, 1),
        "pre_echo": round(sev_pre_echo * 100.0, 1),
        "transient_smearing": round(sev_transient * 100.0, 1),
        "head_wear": round(sev_head * 100.0, 1),
        "riaa_curve_error": round(sev_riaa * 100.0, 1),
        "aliasing": round(sev_alias * 100.0, 1),
        "bias_error": round(sev_bias * 100.0, 1),
        "transport_bump": round(sev_transport_bump * 100.0, 1),
        # Zeitpositionen (Sekunden) für vertikale Wellenform-Marker
        "_locations": {
            "clicks": _locs(DefectType.CLICKS),
            "crackle": _locs(DefectType.CRACKLE),
            "clipping": _locs(DefectType.CLIPPING),
            "hum": _locs(DefectType.HUM),
            "noise": _locs(DefectType.HIGH_FREQ_NOISE),
            "dropout": _locs(DefectType.DROPOUTS),
            "wow": _locs_opt("WOW"),
            "flutter": _locs_opt("FLUTTER"),
            "rumble": _locs(DefectType.LOW_FREQ_RUMBLE),
            "sibilance": _locs(DefectType.SIBILANCE),
            "digital_artifacts": _locs(DefectType.DIGITAL_ARTIFACTS),
            "compression_artifacts": _locs(DefectType.COMPRESSION_ARTIFACTS),
            "stereo_imbalance": _locs_opt("STEREO_IMBALANCE"),
            "phase_issues": _locs_opt("PHASE_ISSUES"),
            "dc_offset": _locs_opt("DC_OFFSET"),
            "bandwidth_loss": _locs_opt("BANDWIDTH_LOSS"),
            "pitch_drift": _locs_opt("PITCH_DRIFT"),
            "reverb_excess": _locs_opt("REVERB_EXCESS"),
            "print_through": _locs_opt("PRINT_THROUGH"),
            "quantization_noise": _locs_opt("QUANTIZATION_NOISE"),
            "jitter_artifacts": _locs_opt("JITTER_ARTIFACTS"),
            "dynamic_compression_excess": _locs_opt("DYNAMIC_COMPRESSION_EXCESS"),
            "pre_echo": _locs_opt("PRE_ECHO"),
            "transient_smearing": _locs_opt("TRANSIENT_SMEARING"),
            "head_wear": _locs_opt("HEAD_WEAR"),
            "riaa_curve_error": _locs_opt("RIAA_CURVE_ERROR"),
            "aliasing": _locs_opt("ALIASING"),
            "bias_error": _locs_opt("BIAS_ERROR"),
            "transport_bump": _locs_opt("TRANSPORT_BUMP"),
        },
        # Per-channel locations (L/R) for stereo channel-accurate overlay.
        # Premium: alle lokalisierbaren Impuls-/Event-Defekte kanalgenau,
        # kein "über einen Kamm" für L+R.
        "_channel_locations": {
            "clicks": _ch_locs(DefectType.CLICKS),
            "dropout": _ch_locs(DefectType.DROPOUTS),
            "crackle": _ch_locs(DefectType.CRACKLE),
            "clipping": _ch_locs(DefectType.CLIPPING),
            "sibilance": _ch_locs(DefectType.SIBILANCE),
            "transport_bump": _ch_locs_opt("TRANSPORT_BUMP"),
        },
        "status": status,
    }


def _result_scores_to_display(defect_scores: dict, status: str = "completed") -> dict:
    """Konvertiert RestorationResult.defect_scores (Dict[DefectType, float]) in Display-Dict.

    defect_scores enthaelt Severity-Werte 0.0–1.0; Skalierung wie _defect_analysis_to_display.
    Nach Restaurierung liegen die Werte typisch nahe 0 → Zaehler gehen auf 0 zurueck.
    """
    DefectType = _bridge_get_defect_type()
    if DefectType is None:
        return {"status": status}

    def _f(key) -> float:
        v = defect_scores.get(key, 0.0)
        return float(v) if v is not None else 0.0

    def _f_opt(attr: str) -> float:
        dt = getattr(DefectType, attr, None)
        if dt is None:
            return 0.0
        return _f(dt)

    sev_clicks = _f(DefectType.CLICKS)
    sev_crackle = _f(DefectType.CRACKLE)
    sev_clip = _f(DefectType.CLIPPING)
    sev_hum = _f(DefectType.HUM)
    sev_noise = _f(DefectType.HIGH_FREQ_NOISE)
    sev_dropout = _f(DefectType.DROPOUTS)
    sev_wow = _f_opt("WOW")
    sev_flutter = _f_opt("FLUTTER")
    sev_rumble = _f(DefectType.LOW_FREQ_RUMBLE)
    sev_sibilance = _f(DefectType.SIBILANCE)
    sev_digital = _f(DefectType.DIGITAL_ARTIFACTS)
    sev_compress = _f(DefectType.COMPRESSION_ARTIFACTS)
    sev_stereo = _f_opt("STEREO_IMBALANCE")
    sev_phase = _f_opt("PHASE_ISSUES")
    sev_dc = _f_opt("DC_OFFSET")
    sev_bw = _f_opt("BANDWIDTH_LOSS")
    sev_pitch = _f_opt("PITCH_DRIFT")
    sev_reverb = _f_opt("REVERB_EXCESS")
    sev_print = _f_opt("PRINT_THROUGH")
    sev_quant = _f_opt("QUANTIZATION_NOISE")
    sev_jitter = _f_opt("JITTER_ARTIFACTS")
    sev_dyncomp = _f_opt("DYNAMIC_COMPRESSION_EXCESS")
    sev_pre_echo = _f_opt("PRE_ECHO")
    sev_transient = _f_opt("TRANSIENT_SMEARING")
    sev_head = _f_opt("HEAD_WEAR")
    sev_riaa = _f_opt("RIAA_CURVE_ERROR")
    sev_alias = _f_opt("ALIASING")
    sev_bias = _f_opt("BIAS_ERROR")
    sev_transport_bump = _f_opt("TRANSPORT_BUMP")

    return {
        "clicks": int(sev_clicks * 500),
        "crackle": int(sev_crackle * 500),
        "pops": int(sev_clip * 100),
        "clipping": int(sev_clip * 200),
        "sibilance": int(sev_sibilance * 300),
        "dropout": int(sev_dropout * 50),
        "hum": round(sev_hum * 50.0, 2),
        "noise_level": round(sev_noise * 75.0, 2),
        "wow": round(sev_wow * 3.0, 3),
        "flutter": round(sev_flutter * 3.0, 3),
        "rumble": round(sev_rumble * 100.0, 1),
        "digital_artifacts": round(sev_digital * 100.0, 1),
        "compression_artifacts": round(sev_compress * 100.0, 1),
        "stereo_imbalance": round(sev_stereo * 100.0, 1),
        "phase_issues": round(sev_phase * 100.0, 1),
        "dc_offset": round(sev_dc * 100.0, 1),
        "bandwidth_loss": round(sev_bw * 100.0, 1),
        "pitch_drift": round(sev_pitch * 100.0, 1),
        "reverb_excess": round(sev_reverb * 100.0, 1),
        "print_through": round(sev_print * 100.0, 1),
        "quantization_noise": round(sev_quant * 100.0, 1),
        "jitter_artifacts": round(sev_jitter * 100.0, 1),
        "dynamic_compression_excess": round(sev_dyncomp * 100.0, 1),
        "pre_echo": round(sev_pre_echo * 100.0, 1),
        "transient_smearing": round(sev_transient * 100.0, 1),
        "head_wear": round(sev_head * 100.0, 1),
        "riaa_curve_error": round(sev_riaa * 100.0, 1),
        "aliasing": round(sev_alias * 100.0, 1),
        "bias_error": round(sev_bias * 100.0, 1),
        "transport_bump": round(sev_transport_bump * 100.0, 1),
        "_locations": {},  # Nach Restaurierung keine Zeitpositionen verfügbar
        "status": status,
    }


# Formate die soundfile NICHT unterstützt → direkt zu pedalboard/librosa
_SF_UNSUPPORTED_EXT = frozenset(
    {
        ".mp3",
        ".mp2",
        ".mp1",
        ".m4a",
        ".m4b",
        ".m4p",
        ".aac",
        ".wma",
        ".asf",
        ".opus",
        ".webm",
        ".amr",
        ".3gp",
        ".3g2",
        ".ac3",
        ".dts",
    }
)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Stellt sicher: float32, shape=(samples,) mono oder (samples, 2) stereo, kein NaN/Inf."""
    audio = np.asarray(audio, dtype=np.float32)
    # (channels, samples) → (samples, channels)
    if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    # Mono squeezen
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    # > 2 Kanäle: gewichteter Downmix zu Stereo (L+R)
    if audio.ndim == 2 and audio.shape[1] > 2:
        audio = audio[:, :2]
    # NaN/Inf bereinigen
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Clipping auf [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def _load_audio_robust(file_path: str) -> tuple:
    """Lädt Audio mit 4-stufiger Fallback-Kaskade (Spec §11.4 Audio-Lade-Kaskade).

    Reihenfolge (immer soundfile zuerst, dann pedalboard):
      ① soundfile    – WAV/FLAC/OGG/AIFF (libsndfile, verlustfrei, Stufe 1)
      ② pedalboard   – MP3/M4A/AAC/WMA/OPUS (ffmpeg-Backend, Stufe 2)
      ③ pydub/ffmpeg – universeller Fallback via ffmpeg Subprocess
      ④ librosa      – letzter Fallback (audioread/GStreamer)

    Returns (audio: np.ndarray float32, sr: int).
    Raises RuntimeError wenn alle Stufen scheitern.
    """
    _errors: list[str] = []

    # ── Stufe 1: soundfile (WAV, FLAC, OGG, AIFF …) ─────────────────────────
    try:
        import soundfile as _sf

        _audio_sf, _sr_sf = _sf.read(file_path, dtype="float32", always_2d=False)
        return _normalize_audio(_audio_sf), int(_sr_sf)
    except Exception as _e1:
        _errors.append(f"soundfile: {_e1}")

    # ── Stufe 2: pedalboard (MP3/M4A/AAC/WMA/OPUS und universeller Fallback) ─
    try:
        from pedalboard.io import AudioFile as _PBAudioFile  # type: ignore

        with _PBAudioFile(file_path) as _f:
            _sr = int(_f.samplerate)
            _frames = _f.frames
            # Chunk-basiertes Lesen (verhindert OOM bei sehr langen Dateien)
            _chunk_size = _sr * 300  # 300 s Chunks
            _parts: list[np.ndarray] = []
            _read = 0
            while _read < _frames:
                _block = _f.read(min(_chunk_size, _frames - _read))  # (ch, samples)
                _parts.append(_block)
                _read += _block.shape[-1]
        _raw = np.concatenate(_parts, axis=1) if len(_parts) > 1 else _parts[0]
        return _normalize_audio(_raw), _sr
    except Exception as _e2:
        _errors.append(f"pedalboard: {_e2}")

    # ── Stufe 4: pydub via ffmpeg (universell, sehr robust) ──────────────────
    try:
        from pydub import AudioSegment as _AudioSeg  # type: ignore

        _seg = _AudioSeg.from_file(file_path)
        _sr_pd = _seg.frame_rate
        _samples_pd = np.array(_seg.get_array_of_samples(), dtype=np.float32)
        # Normalisieren auf [-1, 1] (pydub gibt int16-Werte zurück)
        _bit_depth = _seg.sample_width * 8
        _samples_pd /= float(2 ** (_bit_depth - 1))
        if _seg.channels > 1:
            _samples_pd = _samples_pd.reshape(-1, _seg.channels)
        return _normalize_audio(_samples_pd), int(_sr_pd)
    except Exception as _e4:
        _errors.append(f"pydub: {_e4}")

    # ── Stufe 5: librosa (audioread/GStreamer, letzter Ausweg) ────────────────
    try:
        import threading as _threading

        import librosa as _librosa  # type: ignore

        _result: list = []
        _err: list = []

        def _lb_load():
            try:
                _y, _s = _librosa.load(file_path, sr=None, mono=False)
                _result.append((_y, int(_s)))
            except Exception as _le:
                _err.append(_le)

        _t = _threading.Thread(target=_lb_load, daemon=True)
        _t.start()
        _t.join(timeout=120)
        if _t.is_alive():
            raise RuntimeError("Timeout nach 120 s (audioread/GStreamer hängt)")
        if _err:
            raise _err[0]
        _y_lb, _sr_lb = _result[0]
        return _normalize_audio(_y_lb), _sr_lb
    except Exception as _e5:
        _errors.append(f"librosa: {_e5}")

    # ── Alle Stufen gescheitert ───────────────────────────────────────────────
    _detail = "\n  • ".join(_errors)
    raise RuntimeError(
        f"'{Path(file_path).name}' konnte nicht geladen werden.\n\n"
        f"Mögliche Ursachen:\n"
        f"  • Datei ist beschädigt oder unvollständig\n"
        f"  • Dateiformat nicht unterstützt (unterstützt: MP3, WAV, FLAC, M4A, AAC, WMA, OGG, AIFF)\n"
        f"  • ffmpeg fehlt: sudo apt install ffmpeg\n\n"
        f"Technische Details:\n  • {_detail}"
    )


class SimpleBatchItem:
    """Simple batch queue item"""

    def __init__(self, item_id, input_file, output_file, settings):
        self.id = item_id
        self.input_file = input_file
        self.output_file = output_file
        self.settings = settings
        self.status = "pending"
        self.progress = 0
        self.error = None
        self.restoration_result = None  # RestorationResult nach Verarbeitung


class SimpleBatchQueue:
    """Simple batch queue manager"""

    def __init__(self):
        self.items = []
        self.next_id = 1

    def add_item(self, input_file, output_file, settings):
        """Add item to queue"""
        item = SimpleBatchItem(f"item_{self.next_id}", input_file, output_file, settings)
        self.items.append(item)
        self.next_id += 1
        return item

    def get_next_pending(self):
        """Get next pending item"""
        for item in self.items:
            if item.status == "pending":
                return item
        return None

    def get_item(self, item_id):
        """Get item by ID"""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_stats(self):
        """Get queue statistics"""
        return {
            "total": len(self.items),
            "pending": sum(1 for i in self.items if i.status == "pending"),
            "processing": sum(1 for i in self.items if i.status == "processing"),
            "completed": sum(1 for i in self.items if i.status == "completed"),
            "failed": sum(1 for i in self.items if i.status == "failed"),
        }

    def clear_completed(self):
        """Clear completed items"""
        self.items = [i for i in self.items if i.status not in ("completed", "failed")]


class BatchProcessingThread(QThread):
    """Background thread for batch queue processing with visualization"""

    item_started = pyqtSignal(str)  # item_id
    item_progress = pyqtSignal(str, int)  # item_id, progress
    item_finished = pyqtSignal(str)  # item_id
    item_finished_with_result = pyqtSignal(str, object)  # item_id, RestorationResult
    item_error = pyqtSignal(str, str)  # item_id, error_message
    all_finished = pyqtSignal()
    # Visualization signals
    waveform_data = pyqtSignal(np.ndarray, int)  # audio samples, sample_rate
    defect_update = pyqtSignal(dict)  # defect statistics
    phase_update = pyqtSignal(str)  # current processing phase
    # Resource/Mode signals
    mode_update = pyqtSignal(str)  # quality mode (FAST/BALANCED/QUALITY/MAXIMUM)
    ml_status_update = pyqtSignal(bool, list)  # ml_active, active_plugins
    # Enhanced real-time UX feedback (§11.4)
    phase_progress = pyqtSignal(int)  # sub-phase progress 0–100 within current step
    scan_progress = pyqtSignal(float)  # waveform scan-cursor fraction 0.0–1.0
    quality_update = pyqtSignal(float)  # live MOS estimate 0.0–5.0
    phase_step_update = pyqtSignal(int, int)  # (current_step, total_steps) — Stufe X von Y
    # §Live-Waveform: Audio-Daten nach jeder Phase aktualisiert (kein Zoom-Reset)
    waveform_phase_update = pyqtSignal(np.ndarray, int, str)  # audio, sr, phase_id

    def __init__(self, queue: SimpleBatchQueue):
        super().__init__()
        self.queue = queue
        self._stop_requested = False

    def run(self):
        """Process all items in queue with visualization updates"""
        try:
            # P1: Core-Imports AUSSCHLIEßLICH über Bridge (§11 Spec 08)
            # Singleton-Accessor: sichert Single-Orchestrator Ownership pro Prozess
            # (No-Competing-Instances-Protokoll — RELEASE_MUST).
            _denker_singleton = _bridge_get_aurik_denker_instance()
            if _denker_singleton is None:
                raise RuntimeError(
                    "AurikDenker-Singleton ist über die Bridge nicht verfügbar. "
                    "Frontend startet aus Sicherheitsgründen keine UV3-Direktrestaurierung."
                )
        except Exception as e:
            logger.error("Backend-Import fehlgeschlagen: %s", e)
            # Alle ausstehenden Einträge als fehlgeschlagen markieren
            for _pending in [i for i in self.queue.items if i.status == "pending"]:
                _pending.status = "failed"
                _pending.error = f"Backend konnte nicht geladen werden: {e}"
                self.item_error.emit(_pending.id, _pending.error)
            self.all_finished.emit()
            return

        while not self._stop_requested:
            # P0: Qt-Interrupt-Check (Escape / _cancel_processing)
            if self.isInterruptionRequested():
                logger.info("BatchProcessingThread: Abbruch durch Nutzer")
                break
            # Get next item
            item = self.queue.get_next_pending()
            if item is None:
                break

            try:
                # Mark as processing
                item.status = "processing"
                item.progress = 0
                self._last_phase_state = None  # reset live-time state for new item
                self.item_started.emit(item.id)
                self.phase_step_update.emit(1, 0)
                self.phase_update.emit(f"Restaurierung startet: {Path(item.input_file).name}")
                # Sofort 3 % zeigen — noch bevor Audio-Loading beginnt.
                # Ohne diesen Emit bleibt die Bar bei 0,00 % für die gesamte
                # Ladedauer (bei großen MP3s bis zu 30 s).
                item.progress = 3
                self.item_progress.emit(item.id, 3)

                # Load audio (MP3, WAV, FLAC, M4A, …) — resample to 48 kHz (Aurik internal SR)
                audio, sr = _load_audio_robust(item.input_file)
                if sr != 48_000:
                    from scipy.signal import resample_poly as _rp

                    _gcd = math.gcd(sr, 48_000)
                    audio = _rp(
                        audio,
                        48_000 // _gcd,
                        sr // _gcd,
                        axis=0 if audio.ndim > 1 else -1,
                    ).astype(np.float32)
                    sr = 48_000
                self.waveform_data.emit(audio, sr)  # Send to visualization
                # Track audio duration for realistic ETA (accessible from heartbeat)
                _audio_dur_s_local = float(audio.shape[0]) / sr
                self._audio_duration_s = _audio_dur_s_local
                self._ml_start_elapsed = -1.0  # set when UV3 phase begins (pct ≥ 35)
                item.progress = 6
                self.item_progress.emit(item.id, 6)

                # Map GUI modes to Denker modes (§2.2 Spec)
                mode = item.settings.get("mode", "RESTORATION")
                if mode == "STUDIO_2026":
                    ui_mode = "MAXIMUM"
                    _aurik_mode = "studio2026"
                else:  # RESTORATION
                    ui_mode = "QUALITY"
                    _aurik_mode = "restoration"

                # Emit mode update
                self.mode_update.emit(ui_mode)
                # Pre-compute expected processing time for user-facing ETA.
                # RT factors measured on reference desktop (CPU-only, no GPU):
                #   Restoration ≈12× RT (±2× by hardware), Studio 2026 ≈18× RT (±4×).
                _rt_lo_eta = 10.0 if _aurik_mode == "restoration" else 14.0
                _rt_hi_eta = 15.0 if _aurik_mode == "restoration" else 22.0
                _eta_ovhd = 90.0  # analysis + DefectScanner overhead
                self._expected_processing_s = _audio_dur_s_local * (_rt_lo_eta + _rt_hi_eta) / 2.0 + _eta_ovhd
                _eta_lo_m = max(1, int((_audio_dur_s_local * _rt_lo_eta + _eta_ovhd) / 60))
                _eta_hi_m = max(1, int((_audio_dur_s_local * _rt_hi_eta + _eta_ovhd) / 60) + 1)
                _aud_m, _aud_sr = divmod(int(_audio_dur_s_local), 60)
                self._eta_range_str = f"Erwartet: {_eta_lo_m}–{_eta_hi_m} min für {_aud_m}:{_aud_sr:02d} Audio"

                # ML-Plugins: Lokale ONNX-Modelle (kein Docker)
                self.ml_status_update.emit(False, [])

                item.progress = 8
                self.item_progress.emit(item.id, 8)

                # Defect analysis phase: Cache-First (kein Doppelscan, §9.4)
                self.phase_step_update.emit(2, 0)
                self.phase_update.emit("Schadensbewertung wird präzisiert …")

                _cached_scan = get_cached_defect_result(item.input_file)
                if _cached_scan is not None:
                    # Scan-Ergebnis aus dem Import-Cache übernehmen
                    _scan = _cached_scan
                    logger.debug("BatchProcessingThread: DefectScan aus Cache (%s)", item.input_file)
                else:
                    # Kein Cache-Eintrag → Scan einmalig hier durchführen
                    _DefectScanner = _bridge_get_defect_scanner()
                    if _DefectScanner is None:
                        raise RuntimeError("DefectScanner ist über die Bridge nicht verfügbar")
                    _scan = _DefectScanner().scan(audio, sr)
                    cache_defect_result(item.input_file, _scan)
                defects = _defect_analysis_to_display(_scan.scores, status="detected")
                self.defect_update.emit(defects)

                # P0: Interrupt-Check nach schwerem Scan
                if self.isInterruptionRequested():
                    item.status = "cancelled"
                    self.item_finished.emit(item.id)
                    break

                item.progress = 12
                self.item_progress.emit(item.id, 12)

                # Process
                self.phase_step_update.emit(3, 0)
                self.phase_update.emit("Musik wird restauriert …")

                # Phase 2: Correction starting
                defects_correcting = defects.copy()
                defects_correcting["status"] = "correcting"
                self.defect_update.emit(defects_correcting)

                # ML-Plugin-Status: Phasen-Namen → aktive Plugin-Schlüsselwörter
                # Two-tier matching:
                # Tier 1 — denker-level phase name keywords (plugin names appear literally, e.g. "deepfilternet")
                # Tier 2 — UV3 phase ID substrings (phase_XX_name style, mapped to their primary ML plugin)
                _ML_PHASE_MARKERS = {
                    # ── Tier 1: Plugin-Namen erscheinen direkt im Phasennamen ──
                    "deepfilternet": "DeepFilterNet",
                    "dfn": "DeepFilterNet",
                    "melbandroformer": "MelBandRoformer",
                    "bs_roformer": "MelBandRoformer",
                    "mdx23c": "MDX23C",
                    "sgmse": "SGMSE+",
                    "resemble": "Resemble-Enhance",
                    "apollo": "Apollo",
                    "rmvpe": "RMVPE",
                    "crepe": "CREPE",
                    "audiosr": "AudioSR",
                    "vocos": "Vocos",
                    "bigvgan": "BigVGAN",
                    "panns": "PANNs",
                    "beats": "BEATs",
                    "versa": "VERSA (Bewertung)",
                    "flow_matching": "Flow-Matching",
                    "cqtdiff": "CQTdiff+",
                    "fcpe": "FCPE",
                    # ── Tier 2: UV3 Phase-ID Teilstrings → primäres ML-Plugin ──
                    # UV3 phase names: "phase_03_denoise", "phase_29_tape_hiss_reduction", …
                    "denoise": "DeepFilterNet",
                    "tape_hiss": "DeepFilterNet",
                    "diffusion_inpainting": "Flow-Matching",
                    "dropout_repair": "AudioSR",
                    "frequency_restoration": "AudioSR",
                    "wow_flutter": "CREPE",
                    "vocal_enhancement": "VocalAI",
                    "ml_deesser": "ML-DeEsser",
                    "reverb_reduction": "WPE",
                    "semantic_audio": "PANNs",
                    "spectral_band_gap": "CQTdiff+",
                    # ── Tier 2b: Weitere Phasen → DSP-Modul / Algorithmus ──
                    "hum_removal": "DSP Notch-Filter",
                    "click_removal": "DSP Click-Repair",
                    "click_pop": "DSP Click-Repair",
                    "crackle_removal": "DSP Decrackle",
                    "rumble_filter": "DSP Hochpass",
                    "surface_noise": "DSP Noise-Profiling",
                    "noise_gate": "DSP Noise-Gate",
                    "eq_correction": "DSP EQ-Korrektur",
                    "harmonic_restoration": "DSP Harmonic-Synth",
                    "phase_correction": "DSP Phase-Align",
                    "speed_pitch": "CREPE",
                    "azimuth_correction": "DSP Azimuth",
                    "spectral_repair": "DSP PGHI",
                    "print_through": "DSP LMS-Adaptive",
                    "transient_preservation": "DSP Transient-Guard",
                    "transient_shaper": "DSP Transient-Shaper",
                    "de_esser": "DSP De-Esser",
                    "tape_saturation": "DSP Sättigungs-Emulation",
                    "compression": "DSP Kompressor",
                    "loudness_normalization": "DSP LUFS-Norm",
                    "truepeak_limiter": "DSP TruePeak",
                    "final_eq": "DSP Final-EQ",
                    "mastering_polish": "DSP Mastering",
                    "dc_offset": "DSP DC-Offset",
                    "advanced_dereverb": "WPE",
                    "declip": "DSP Declip",
                    "spectral_coherence": "DSP Spectral-Coherence",
                    "output_format": "DSP Export-Optimierung",
                }
                _live_ml_seen: set[str] = set()

                # ── Smooth progress interpolator (§11.4) ─────────────────────
                # Runs at ~30 fps in a daemon thread. Advances _sp["current"]
                # toward _sp["target"] with easing, plus a time-proportional creep
                # between phases — bar moves at a steady estimated rate, no jumps.
                #
                #   Creep strategy: advance (last_jump * 0.75) pts over avg_phase_dur
                #   seconds so the bar always moves proportionally to real phase speed.
                #   avg_phase_dur is a rolling average of actual inter-phase elapsed
                #   time, self-calibrating after the first two phase transitions.
                _SP_INTERVAL = 0.033  # ~30 fps
                _SP_MAX_STEP = 0.6  # max step per frame for catch-up easing
                _SP_MIN_VELOCITY = 0.012  # min pts/frame ≈ 0.36 pts/s — bar always moves
                _sp: dict = {
                    "current": 12.0,
                    "target": 12.0,
                    "alive": True,
                    "last_target_time": time.perf_counter(),
                    "last_jump": 3.0,  # initial estimate (display pts)
                    "avg_phase_dur": 4.0,  # initial estimate (seconds/phase) — self-calibrates
                }
                # Sub-progress bar state: 0–100 within the current phase step
                _sp2: dict = {"current": 0.0, "target": 0.0}
                _sp_lock = threading.Lock()

                def _smooth_progress_emitter(_item=item) -> None:
                    """30fps smooth progress emitter — continuous velocity, never stalls.

                    Uses a unified velocity model instead of a binary catch-up/creep
                    switch with a hard cap (which caused the bar to freeze for seconds
                    when a phase took longer than estimated).

                    Main bar:
                      - Behind target: proportional catch-up (eddy-current brake easing)
                      - At/past target: predictive creep that distributes expected
                        advancement over the estimated remaining time to next phase.
                      - Overtime (phase takes longer than avg): hyperbolic slowdown —
                        velocity = base_rate / (1 + overtime_fraction × 2). Never zero.
                      - Always ≥ _SP_MIN_VELOCITY → no freezes, no "ruck" pattern.

                    Sub-bar (_sp2): time-proportional fill 0→98 distributed over
                    estimated phase duration — no hard cap at 95.

                    Throttle: only emit signals when integer value actually changes.
                    """
                    _last_emit_val: int = -1
                    _last_phase_pct: int = -1
                    _last_scan_int: int = -1  # int(frac * 500) for ~0.2 % granularity
                    while True:
                        time.sleep(_SP_INTERVAL)
                        with _sp_lock:
                            if not _sp["alive"]:
                                return
                            cur = _sp["current"]
                            tgt = _sp["target"]
                            last_jump = _sp["last_jump"]
                            avg_dur = max(1.0, _sp["avg_phase_dur"])
                            last_tgt_time = _sp["last_target_time"]
                            sub_cur = _sp2["current"]
                            sub_tgt = _sp2["target"]

                        gap = tgt - cur
                        elapsed_since_tgt = time.perf_counter() - last_tgt_time

                        # ── Main bar: unified velocity model ──────────────────
                        if gap > 0.3:
                            # Behind target: exponential easing (eddy-current brake)
                            step = max(0.08, gap * 0.06)
                            step = min(step, _SP_MAX_STEP)
                            new_cur = cur + step
                        else:
                            # At/near/past target: predictive creep.
                            # How much we expect the bar to advance for the next phase.
                            expected_advance = max(1.5, last_jump * 0.85)

                            # How far we've already crept beyond the last reported target.
                            overshoot = max(0.0, cur - tgt)
                            fill_frac = min(0.92, overshoot / expected_advance) if expected_advance > 0 else 0.9

                            if elapsed_since_tgt < avg_dur:
                                # Within expected phase duration: steady advance.
                                # Distribute remaining budget over estimated remaining time.
                                est_remaining_t = max(0.3, avg_dur - elapsed_since_tgt)
                                remaining_pts = expected_advance * (1.0 - fill_frac)
                                velocity = remaining_pts * _SP_INTERVAL / est_remaining_t
                            else:
                                # Overtime: phase taking longer than avg_phase_dur.
                                # Hyperbolic slowdown: v = base / (1 + k * overtime_frac).
                                # Never reaches zero — bar always moves.
                                overtime_frac = (elapsed_since_tgt - avg_dur) / avg_dur
                                base_rate = expected_advance * _SP_INTERVAL / avg_dur
                                velocity = base_rate / (1.0 + overtime_frac * 2.0)

                            # Smooth blend from catch-up when gap is small-positive
                            if gap > 0:
                                velocity += gap * 0.05

                            # Floor: bar ALWAYS moves at minimum velocity (anti-stall)
                            velocity = max(_SP_MIN_VELOCITY, velocity)

                            # Soft ceiling near 97.5: asymptotic deceleration, not hard stop
                            headroom = max(0.01, 97.5 - cur)
                            if headroom < 2.0:
                                velocity = min(velocity, headroom * 0.015)
                                velocity = max(velocity, 0.003)

                            new_cur = cur + velocity
                            new_cur = min(new_cur, 97.5)

                        # ── Sub-bar: phase-completion-aware fill ──────────
                        # Sentinel: current == -1.0 means "flash 100 for one tick
                        # (previous phase completed), then start new phase from 0"
                        if sub_cur < -0.5:
                            sub_new = 100.0
                            with _sp_lock:
                                _sp2["current"] = 0.0
                        elif sub_tgt >= 99.5:
                            # Phase completed (✓ received) — accelerate to 100
                            sub_gap_c = max(0.01, 100.0 - sub_cur)
                            sub_new = sub_cur + max(1.5, sub_gap_c * 0.25)
                            sub_new = min(sub_new, 100.0)
                            with _sp_lock:
                                _sp2["current"] = sub_new
                        else:
                            sub_gap = sub_tgt - sub_cur
                            if sub_gap > 1.0:
                                # Catching up to sub-target: proportional easing
                                sub_step = max(0.8, sub_gap * 0.15)
                                sub_new = sub_cur + min(sub_step, 5.0)
                            else:
                                # Drift toward 90 over estimated remaining phase time
                                sub_headroom = max(0.01, 90.0 - sub_cur)
                                if elapsed_since_tgt < avg_dur and sub_headroom > 0.5:
                                    sub_remaining_t = max(0.3, avg_dur - elapsed_since_tgt)
                                    sub_vel = sub_headroom * _SP_INTERVAL / sub_remaining_t * 0.8
                                else:
                                    # Overtime drift: slow but never zero
                                    sub_vel = max(0.04, sub_headroom * 0.008)
                                sub_new = sub_cur + max(0.04, sub_vel)
                                sub_new = min(sub_new, 95.0)
                            with _sp_lock:
                                _sp2["current"] = sub_new

                        with _sp_lock:
                            _sp["current"] = new_cur
                        emit_val = min(98, int(new_cur))
                        _item.progress = emit_val
                        # Throttle: only emit when integer value changes (avoids 30 fps
                        # identical-value signals flooding the Qt event queue).
                        if emit_val != _last_emit_val:
                            _last_emit_val = emit_val
                            self.item_progress.emit(_item.id, emit_val)
                        _phase_pct = min(100, int(sub_new))
                        if _phase_pct != _last_phase_pct:
                            _last_phase_pct = _phase_pct
                            self.phase_progress.emit(_phase_pct)
                        # Scan-cursor: derived from the REAL reported-progress target (tgt).
                        # Using tgt directly ensures the cursor stays at the audio-start
                        # (frac=0.0) during analysis and only advances on real callbacks.
                        _scan_frac = max(0.0, min(1.0, (tgt - 12.0) / 85.0))
                        _scan_int = int(_scan_frac * 500)
                        if _scan_int != _last_scan_int:
                            _last_scan_int = _scan_int
                            self.scan_progress.emit(_scan_frac)

                _sp_thread = threading.Thread(
                    target=_smooth_progress_emitter, daemon=True, name="aurik-smooth-progress"
                )
                _sp_thread.start()

                # ── Closure state for real-time UX feedback ─────────────────────────────
                # Tracks phase label for sub-bar reset on phase transitions
                _last_phase_key: list[str] = [""]
                # Live defect scores captured from scan; reduced per phase for count-down animation
                _current_defect_scores: dict = {k: v for k, v in defects.items() if k != "status"}
                # Accumulated variant scores for ranking display
                _variant_scores: list[tuple[str, str]] = []
                # UV3 per-phase step tracking: each distinct [phase_id] becomes its own
                # numbered step so the user sees "Stufe 12 · Click Removal" etc.
                _uv3_seen: dict[str, int] = {}  # phase_id → absolute step number
                _uv3_count: list[int] = [0]  # mutable counter, increments per new phase
                _uv3_total_known: list[int] = [0]  # UV3 meldet Gesamtzahl vorab → fixer Total-Wert
                # Reset dynamic step names for this run
                self._dynamic_step_names: dict[int, str] = {}
                # Human-readable phase explanation appended to status line
                _PHASE_EXPL: dict[str, str] = {
                    "eingabe wird geprüft": "Datei wird überprüft",
                    "eingangsqualität wird analysiert": "Klangqualität wird gemessen",
                    "defekte und material werden erkannt": "Schäden und Aufnahme-Typ werden erkannt",
                    "kausale defektreihenfolge": "Reihenfolge der Korrekturen wird festgelegt",
                    "signalkette wird invertiert": "Originale Aufnahmekette wird analysiert",
                    "lücken und aussetzer werden erkannt": "Stille und Aussetzer werden kartiert",
                    "musikalische restaurierungsziele": "14 Klangziele werden kalibriert",
                    "restaurierungs-varianten werden geplant": "Beste Restaurierungsstrategie wird gewählt",
                    "multi-pass": "Klangqualität wird optimiert",
                    "qualitäts-gate": "Klangqualität wird geprüft",
                    "gewinner": "Bestes Ergebnis wird finalisiert",
                    "ergebnis wird gespeichert": "Endergebnis wird gesichert",
                    "tape_hiss": "Bandrauschen wird entfernt",
                    "denoise": "Rauschen wird unterdrückt",
                    "dropout": "Tonaussetzer werden repariert",
                    "click_repair": "Knackser werden repariert",
                    "click_removal": "Knackser werden entfernt",
                    "click_pop": "Knackser und Impulse werden entfernt",
                    "declick": "Knackser werden entfernt",
                    "wow_flutter": "Tonhöhenschwankungen werden korrigiert",
                    "reverb_reduction": "Nachhall wird reduziert",
                    "frequency_restoration": "Fehlende Frequenzen werden ergänzt",
                    "vocal_enhancement": "Gesang wird mit KI verbessert",
                    "diffusion_inpainting": "Fehlende Audiostellen werden rekonstruiert",
                    "semantic_audio": "Klangstruktur wird analysiert",
                    "hum_removal": "Netzbrummen wird entfernt",
                    "crackle_removal": "Knistern wird entfernt",
                    "rumble_filter": "Tieffrequenz-Rumpeln wird entfernt",
                    "surface_noise": "Oberflächengeräusche werden gemessen",
                    "noise_gate": "Stille-Abschnitte werden bereinigt",
                    "eq_correction": "Klangbalance wird korrigiert",
                    "harmonic_restoration": "Obertöne werden ergänzt",
                    "phase_correction": "Kanalausrichtung wird korrigiert",
                    "speed_pitch": "Geschwindigkeit und Tonhöhe werden angepasst",
                    "transport_bump": "Bandhopser werden repariert",
                    "azimuth_correction": "Bandkopf-Ausrichtung wird korrigiert",
                    "spectral_repair": "Frequenzspektrum wird repariert",
                    "print_through": "Bandübersprechen wird entfernt",
                    "transient_preservation": "Anschläge und Einsätze werden geschützt",
                    "transient_shaper": "Anschlagsdynamik wird geformt",
                    "de_esser": "Zischlaute werden gemindert",
                    "ml_deesser": "Zischlaute werden mit KI gemindert",
                    "tape_saturation": "Vintage-Charakter wird bewahrt",
                    "compression": "Lautstärkedynamik wird angeglichen",
                    "loudness_normalization": "Lautstärke wird normiert",
                    "truepeak_limiter": "Maximallautstärke wird gesichert",
                    "final_eq": "Klangfarbe wird abgestimmt",
                    "mastering_polish": "Klang wird endbearbeitet",
                    "dc_offset": "Gleichspannung wird entfernt",
                    "advanced_dereverb": "Raumhall wird rekonstruiert und entfernt",
                    "spectral_band_gap": "Frequenzlücken werden geschlossen",
                    "spectral_coherence": "Frequenzstabilität wird gesichert",
                    "output_format": "Export wird vorbereitet",
                    "declip": "Übersteuerungen werden repariert",
                    "resampling": "Abtastrate wird angepasst",
                    "restaurierbarkeit": "Aufnahme wird auf Bearbeitbarkeit geprüft",
                    "phasenauswahl": "Passende Korrekturen werden ausgewählt",
                    "pipeline startet": "Restaurierung wird gestartet",
                    "initialisierung": "Aurik wird vorbereitet",
                    "exzellenz": "Klang wird weiter verfeinert",
                    "versa": "Klangqualität wird bewertet",
                    "ram-management": "Arbeitsspeicher wird freigegeben",
                }
                # Phase keyword → defect keys to reduce (count-down mapping)
                _PHASE_REDUCES: dict[str, list[str]] = {
                    "tape_hiss": ["crackle", "noise_level", "noise"],
                    "denoise": ["noise_level", "noise", "hum"],
                    "dropout": ["dropout"],
                    "click_repair": ["clicks", "pops"],
                    "declick": ["clicks", "pops"],
                    "wow_flutter": ["wow", "flutter", "transport_bump"],
                    "transport_bump": ["transport_bump"],
                    "reverb_reduction": ["reverb_excess"],
                    "frequency_restoration": ["bandwidth_loss"],
                    "vocal": ["sibilance"],
                    "diffusion_inpainting": ["dropout", "bandwidth_loss"],
                    "hum_removal": ["hum"],
                    "rumble": ["rumble"],
                    "declip": ["clipping"],
                    "dc_offset": ["dc_offset"],
                    "quantization": ["quantization_noise"],
                    "compression_artifact": ["compression_artifacts"],
                    "transient": ["transient_smearing"],
                }

                def _on_batch_progress(pct: int, msg: str, elapsed_s: float = 0.0, _item=item) -> None:
                    # §11.4 Stufen-Vorab-Meldung: UV3 meldet Gesamtzahl der Phasen
                    # BEVOR die Pipeline startet → Total von Anfang an korrekt.
                    if msg.startswith("__total_uv3_phases__:"):
                        try:
                            _uv3_total_known[0] = int(msg.split(":")[1])
                        except (ValueError, IndexError):
                            pass
                        return  # Keine UI-Aktualisierung für Metadaten-Nachricht
                    # Track when UV3 ML-phase starts (pct ≥ 35) for sub-segment ETA
                    if elapsed_s > 0 and pct >= 35 and getattr(self, "_ml_start_elapsed", -1.0) < 0:
                        self._ml_start_elapsed = elapsed_s
                    # Direct passthrough: denker pct = UI bar target (no double-compression).
                    # Pre-steps set bar to 12; monotonic guard holds until denker pct > 12.
                    _new_tgt = float(min(98, pct))
                    _item.progress = int(_new_tgt)
                    # Update smooth-emitter target + calibrate phase-timing stats
                    # MONOTONIC: target darf nie sinken (verhindert Rücksprung bei
                    # verschachtelten UV3-Aufrufen in ARE multi-pass full-pass)
                    with _sp_lock:
                        _now = time.perf_counter()
                        _inter_s = _now - _sp["last_target_time"]
                        # Rolling average of actual inter-phase duration (skip rapid-fire < 0.5 s)
                        # 0.5/0.5 EMA adapts faster than old 0.7/0.3 — reaches realistic
                        # estimates within 2-3 phases instead of 5-7.
                        if _inter_s >= 0.5:
                            _sp["avg_phase_dur"] = 0.5 * _sp["avg_phase_dur"] + 0.5 * _inter_s
                        # Track upward jump size for creep velocity calibration
                        _delta = _new_tgt - _sp["target"]
                        if _delta > 0.5:
                            _sp["last_jump"] = _delta
                        _sp["last_target_time"] = _now
                        # Monotonic: nur aufwärts setzen
                        if _new_tgt > _sp["target"]:
                            _sp["target"] = _new_tgt
                    # Live ML-Plugin-Erkennung aus progress-Meldung (voller msg incl. [phase_id])
                    _msg_lower = msg.lower()
                    _msg_underscored = _msg_lower.replace(" ", "_")
                    _newly_active: list[str] = []
                    _current_plugin: str = ""
                    for _key, _label in _ML_PHASE_MARKERS.items():
                        if _key in _msg_lower or _key in _msg_underscored:
                            _current_plugin = _label  # letzter Treffer = spezifischster
                            if _label not in _live_ml_seen:
                                _live_ml_seen.add(_label)
                                _newly_active.append(_label)
                    if _newly_active:
                        self.ml_status_update.emit(True, list(_live_ml_seen))
                    # [phase_id]-Annotation für Anzeige entfernen
                    import re as _re

                    _display_msg = _re.sub(r"\s*\[[a-z0-9_]+\]\s*$", "", msg)
                    # Aktives ML-Plugin / DSP-Modul im Statustext anzeigen
                    if _current_plugin:
                        _display_msg = f"{_display_msg}  ‹{_current_plugin}›"
                    # Phasen-Erklärungstext (kontextueller Kurzhinweis in eckigen Klammern)
                    _expl = ""
                    for _epk, _epv in _PHASE_EXPL.items():
                        if _epk in _msg_lower and _epv:
                            _expl = f"  ·  {_epv}"
                            break
                    # Varianten-Score aus Backend-Meldung erkennen und Rangliste pflegen
                    _vm = _re.search(r"'([^']+)'\s*[→>]\s*(mos\s*[\d.]+|score\s*[\d.]+)", _msg_lower)
                    if _vm:
                        _vname = _vm.group(1)[:22]
                        _raw = _vm.group(2).upper()
                        _vscore = _raw.replace("MOS ", "").replace("SCORE ", "")
                        _variant_scores[:] = [(n, s) for n, s in _variant_scores if n != _vname]
                        _variant_scores.append((_vname, _vscore))

                        def _mos_to_stars(_mos_raw: str) -> str:
                            try:
                                _v = float(_mos_raw)
                                _f = min(5, max(1, round(_v)))
                                return "★" * _f + "☆" * (5 - _f)
                            except Exception:
                                return "★★★"

                        _vranking = "  ›  ".join(
                            f"✓ {n}  {_mos_to_stars(s)}" if i == 0 else f"  {n}  {_mos_to_stars(s)}"
                            for i, (n, s) in enumerate(_variant_scores)
                        )
                        _display_msg = f"Bestes Ergebnis: {_vranking}"
                        _expl = ""
                    # Sub-Fortschrittsbalken: Phase-Tracking mit ✓-Completion
                    # UV3 sends "✓ Phase [id]" when a phase completes → fill sub-bar to 100.
                    # New phase starts → flash previous to 100 (sentinel -1) then reset to 0.
                    _is_variant = "variante" in _msg_lower
                    _is_completion = _display_msg.startswith("✓")
                    if _is_variant:
                        with _sp_lock:
                            _sp2["target"] = min(100.0, max(0.0, float(pct)))
                    elif _is_completion:
                        # Phase completed → accelerate sub-bar to 100%
                        with _sp_lock:
                            _sp2["target"] = 100.0
                    else:
                        # Strip "✓ " prefix for consistent phase-key comparison
                        _phase_key = _display_msg.lstrip("✓ ").strip()[:45] if _display_msg else ""
                        if _phase_key != _last_phase_key[0]:
                            _last_phase_key[0] = _phase_key
                            with _sp_lock:
                                # Sentinel -1: emitter flashes 100 (prev phase done), then resets to 0
                                _sp2["current"] = -1.0
                                _sp2["target"] = 75.0
                        else:
                            with _sp_lock:
                                _sp2["target"] = min(100.0, _sp2["target"] + 12.0)
                    # Defekt-Abbau: Behobene Defekte sofort auf 0 setzen → verschwinden aus Anzeige
                    _defect_reduced = False
                    for _prk, _dlist in _PHASE_REDUCES.items():
                        if _prk in _msg_lower or _prk in _msg_underscored:
                            for _dk in _dlist:
                                if isinstance(_current_defect_scores.get(_dk), (int, float)):
                                    _current_defect_scores[_dk] = 0.0  # sofort entfernen
                            _defect_reduced = True
                    if _defect_reduced:
                        # Set active tool on waveform BEFORE emitting defect update
                        # so resolved locations get tagged with the correct tool name
                        if _current_plugin and hasattr(self, "waveform_widget"):
                            self.waveform_widget._active_tool = _current_plugin
                        self.defect_update.emit({**_current_defect_scores, "status": "correcting"})
                    # Scan-cursor is now driven by the 30 fps smooth-emitter (see below).
                    # Only update quality interpolation here.
                    # Live-Qualitätsschätzung: lineare Interpolation 2.5 → 4.2
                    self.quality_update.emit(2.5 + (pct / 100.0) * 1.7)
                    # Map denker pct to live pipeline step.
                    # PRE_STEPS (1–8) are fixed analysis macro-steps.
                    # Each UV3 phase gets its own sequential step number.
                    # Total = 0 until UV3 reports __total_uv3_phases__, then
                    # total = PRE_STEPS + N_uv3 + 2 (post-steps) stays fixed.
                    _PRE_STEPS = 8
                    _pid_m = _re.search(r"\[([a-z0-9_]+)\]", msg)
                    _cur_pid = _pid_m.group(1) if _pid_m else ""
                    if pct < 4:
                        _d_step = 4  # Tonträger (pct=2)
                    elif pct < 6:
                        _d_step = 5  # Kette (pct=4)
                    elif pct < 8:
                        _d_step = 6  # Defekte (pct=6)
                    elif pct < 12:
                        _d_step = 7  # Globalplan + Strategie (pct=8, 10)
                    elif pct < 35:
                        _d_step = 8  # Vorverarbeitung + UV3-Analyse (pct=12–34)
                    elif _cur_pid:
                        # Real UV3 phase with [phase_id] — assign sequential step
                        if _cur_pid not in _uv3_seen:
                            _uv3_count[0] += 1
                            _uv3_seen[_cur_pid] = _PRE_STEPS + _uv3_count[0]
                            _step_lbl = _re.sub(r"\s+‹[^›]+›\s*$", "", _display_msg)
                            _step_lbl = _step_lbl.lstrip("✓ ").split("  ·")[0].strip()
                            self._dynamic_step_names[_PRE_STEPS + _uv3_count[0]] = _step_lbl
                        _d_step = _uv3_seen[_cur_pid]
                    elif pct < 93:
                        _d_step = _PRE_STEPS + max(1, _uv3_count[0])
                    elif pct < 97:
                        _n_uv3 = _uv3_total_known[0] if _uv3_total_known[0] > 0 else _uv3_count[0]
                        _d_step = _PRE_STEPS + _n_uv3 + 1  # Qualitätsprüfung
                    else:
                        _n_uv3 = _uv3_total_known[0] if _uv3_total_known[0] > 0 else _uv3_count[0]
                        _d_step = _PRE_STEPS + _n_uv3 + 2  # Fertigstellung
                    # Total: 0 (unknown) until UV3 reports phase count, then fixed.
                    if _uv3_total_known[0] > 0:
                        _d_total = _PRE_STEPS + _uv3_total_known[0] + 2
                    else:
                        _d_total = 0  # not yet known → UI shows "Stufe X" without total
                    self.phase_step_update.emit(_d_step, _d_total)
                    # Phasennachricht ohne eingebettete Zeit emittieren —
                    # _tick_heartbeat zählt die Zeit jede 500 ms live herunter.
                    _base_text = f"{_display_msg}{_expl}"
                    # ETA-Restzeit HIER berechnen (wenn pct sich ändert), damit
                    # _tick_heartbeat nur noch runterzählt statt die Formel mit
                    # wachsendem _elapsed und eingefrorenem _pct erneut auszuwerten.
                    _remaining_s = -1.0  # kein ETA
                    _eta_range = getattr(self, "_eta_range_str", "")
                    if elapsed_s >= 2.0 and pct >= 5:
                        _exp_total = getattr(self, "_expected_processing_s", None)
                        _ml_start = getattr(self, "_ml_start_elapsed", -1.0)
                        if pct < 35 and _exp_total and _exp_total > 0:
                            _remaining_s = max(0.0, _exp_total - elapsed_s)
                        elif pct >= 35 and _ml_start >= 0:
                            _ml_elapsed_cb = max(0.1, elapsed_s - _ml_start)
                            _ml_frac_cb = max(0.002, (pct - 35) / 59.0)
                            _remaining_s = max(0.0, _ml_elapsed_cb / _ml_frac_cb * (1.0 - _ml_frac_cb) + 25.0)
                        else:
                            _remaining_s = max(0.0, elapsed_s / max(1, pct) * (100 - pct))
                    self._last_phase_state = {
                        "base": _base_text,
                        "pct": pct,
                        "elapsed_s": elapsed_s,
                        "wall_time": time.perf_counter(),
                        "remaining_s": _remaining_s,
                        "eta_range": _eta_range,
                    }
                    self.phase_update.emit(_base_text)

                # AurikDenker ist der verpflichtende Frontend-Einstiegspunkt.
                # Singleton (No-Competing-Instances-Protokoll): kein new instance pro Run.
                # Era/Genre-Cache: UI-Vorabanalyse wiederverwenden (kein Doppellauf in UV3)

                # §Live-Waveform: Callback für Audio-Updates nach jeder UV3-Phase
                def _audio_update_cb(phase_audio, phase_sr, phase_id_str):
                    """Emit updated audio to waveform widget after each phase."""
                    try:
                        self.waveform_phase_update.emit(phase_audio, int(phase_sr), str(phase_id_str))
                    except Exception:
                        pass

                _denke_kwargs: dict = {
                    "mode": _aurik_mode,
                    "progress_callback": _on_batch_progress,
                    "audio_update_callback": _audio_update_cb,
                }
                # DefectScan-Cache: gecachten Scan an AurikDenker weiterreichen
                # (kein Triple-Scan — §9.4 Cache-First-Invariante)
                if _scan is not None:
                    _denke_kwargs["cached_defect_result"] = _scan
                _cached_eg = get_cached_era_genre_result(item.input_file)
                if _cached_eg is not None:
                    if _cached_eg.get("era_result") is not None:
                        _denke_kwargs["cached_era_result"] = _cached_eg["era_result"]
                    if _cached_eg.get("genre_result") is not None:
                        _denke_kwargs["cached_genre_result"] = _cached_eg["genre_result"]
                    logger.debug("BatchProcessingThread: Era/Genre aus Cache (%s)", item.input_file)
                # Medium-Cache: UI-Vorabanalyse wiederverwenden (kein Doppellauf in UV3)
                _cached_medium = get_cached_medium_result(item.input_file)
                if _cached_medium is not None:
                    _denke_kwargs["cached_medium_result"] = _cached_medium
                    logger.debug("BatchProcessingThread: Medium aus Cache (%s)", item.input_file)
                # Restorability-Cache: UI-Vorabanalyse wiederverwenden (kein Doppellauf in UV3)
                _cached_restorability = get_cached_restorability_result(item.input_file)
                if _cached_restorability is not None:
                    _denke_kwargs["cached_restorability_result"] = _cached_restorability
                    logger.debug("BatchProcessingThread: Restorability aus Cache (%s)", item.input_file)

                result = _denker_singleton.denke(
                    audio,
                    sr,
                    **_denke_kwargs,
                )
                item.progress = 98
                # Stop smooth-emitter and let bar glide to 98 in one final step
                with _sp_lock:
                    _sp["target"] = 98.0
                    _sp["alive"] = False
                self.item_progress.emit(item.id, 98)

                # Phase 3: Post-Restore Defekt-Status + ML-Plugin-Anzeige aus RestorationResult
                _post_scores = result.defect_scores if hasattr(result, "defect_scores") else {}
                if _post_scores:
                    # Backend provided post-restoration scores: trust them.
                    # Defects whose score is near-zero have been removed; others remain.
                    _completed_display = _result_scores_to_display(_post_scores, status="completed")
                else:
                    # Backend returned no post-restoration defect data.
                    # Preserve the animated in-flight state so only defects that were
                    # explicitly reduced to near-zero by matched phases vanish —
                    # not everything.  Strip _locations so waveform position-markers
                    # are cleared (they can't be re-attributed without a re-scan).
                    _completed_display = {k: v for k, v in _current_defect_scores.items() if k != "_locations"}
                    _completed_display["_locations"] = {}
                    _completed_display["status"] = "completed"
                self.defect_update.emit(_completed_display)

                # ML-Plugin-Status: Live-Ergebnis aus _live_ml_seen (während denke() gesammelt)
                # + Nachberechnung aus phases_executed für Plugins die keine progress-Meldung gesendet haben
                _phases = list(getattr(result, "phases_executed", []))
                _active_ml: list[str] = list(_live_ml_seen)
                _seen: set[str] = set(_live_ml_seen)
                for _p in _phases:
                    _pl = _p.lower()
                    for _key, _label in _ML_PHASE_MARKERS.items():
                        if _key in _pl and _label not in _seen:
                            _active_ml.append(_label)
                            _seen.add(_label)

                # Pass-through detection: saubere Digitalquelle → nur VERSA-Messung
                _winning = getattr(result, "winning_variant", None)
                if _winning == "clean_digital_pass_through":
                    # Restore chain was skipped; VERSA only evaluated quality
                    _active_ml_passthrough = [l for l in _active_ml if "VERSA" in l]
                    if not _active_ml_passthrough:
                        _active_ml_passthrough = ["VERSA (Bewertung)"]
                    self.ml_status_update.emit(True, _active_ml_passthrough)
                    self.phase_update.emit("✅ Saubere Quelle erkannt — kein Eingriff nötig, Qualität bestätigt")
                else:
                    self.ml_status_update.emit(bool(_active_ml), _active_ml)

                # RestorationResult im Item speichern (Musical Goals, Genealogie, …)
                # → wird in _on_item_finished an _compute_and_show_quality weitergereicht
                item.restoration_result = result

                # Save: export_guard (NaN/Inf + Clip) + atomares Schreiben (.tmp → os.replace)
                _final_total = (
                    (8 + _uv3_total_known[0] + 2) if _uv3_total_known[0] > 0 else max(13, 8 + max(_uv3_count[0], 1) + 2)
                )
                self.phase_step_update.emit(_final_total, _final_total)
                self.phase_update.emit("Ergebnis wird gespeichert …")
                # Handle RestorationResult object
                if hasattr(result, "audio"):
                    restored_audio = result.audio
                else:
                    restored_audio = result  # Fallback
                restored_audio = _export_guard(restored_audio)

                # §G2: Export-Quality-Gate — prüft Chroma/LUFS/Goals vor Export
                _eq_passed, _eq_warnings = _validate_export_quality(result)
                if _eq_warnings:
                    for _eqw in _eq_warnings:
                        logger.warning("Export-Quality: %s", _eqw)
                if not _eq_passed:
                    logger.error(
                        "Export-Quality-Gate FAILED für %s — schwere Qualitätsverletzung. "
                        "Export wird trotzdem durchgeführt (best-effort). Ursachen: %s",
                        item.id,
                        "; ".join(_eq_warnings),
                    )

                # Ensure output directory exists (P1: output/ subfolder)
                os.makedirs(os.path.dirname(item.output_file), exist_ok=True)
                # Atomares Schreiben: .wav.tmp (soundfile erkennt .mp3.tmp nicht als Format)
                _tmp_path = item.output_file + ".wav.tmp"
                try:
                    sf.write(_tmp_path, restored_audio, sr, format="WAV")
                    os.replace(_tmp_path, item.output_file)
                finally:
                    if os.path.exists(_tmp_path):
                        with contextlib.suppress(OSError):
                            os.remove(_tmp_path)
                item.progress = 100
                self.item_progress.emit(item.id, 100)

                # Mark as completed
                item.status = "completed"
                self.item_finished.emit(item.id)
                self.item_finished_with_result.emit(item.id, result)

            except Exception as e:
                import traceback as _tb

                logger.error(
                    "BatchProcessingThread: Fehler bei %s: %s\n%s",
                    item.input_file,
                    e,
                    _tb.format_exc(),
                )
                error_msg = str(e)
                item.status = "failed"
                item.error = error_msg
                self.item_error.emit(item.id, error_msg)

            finally:
                # Inter-file RAM cleanup — release plugin memory between files
                try:
                    _cleanup_fn = _bridge_get_cleanup_after_file_fn()
                    if callable(_cleanup_fn):
                        _cleanup_fn()
                except Exception:
                    pass

        self.all_finished.emit()

    def stop(self):
        """Request stop"""
        self._stop_requested = True


class _Theme:
    """Central design tokens — single source of truth for colours, radii, fonts.

    Semantic state palette (4 states only — never use raw hex elsewhere):
      IDLE    muted steel-blue  — neutral/processing
      SUCCESS muted sage-green  — successful result
      CAUTION warm amber        — warnings / in-progress
      ALERT   muted brick-red   — errors / failures
    """

    PRIMARY = "#667eea"
    SECONDARY = "#5B9FE8"
    BG_DARK = "#080a18"
    BG_CARD = "rgba(14, 18, 36, 0.75)"
    TEXT_DIM = "#8894A8"
    BORDER = "rgba(102, 126, 234, 0.22)"
    FONT_UI = "Segoe UI"
    RADIUS_SM = 8
    RADIUS_MD = 10
    RADIUS_LG = 15

    # Semantic state — text colours
    IDLE_TEXT = "#7B93B8"  # muted steel-blue
    SUCCESS_TEXT = "#82B89A"  # muted sage-green
    CAUTION_TEXT = "#B8A068"  # warm aged amber
    ALERT_TEXT = "#B87A7A"  # muted brick-red

    # Semantic state — background / border
    SUCCESS_BG = "rgba(85, 155, 115, 0.10)"
    SUCCESS_BD = "rgba(100, 168, 130, 0.26)"
    CAUTION_BG = "rgba(150, 130, 68, 0.10)"
    CAUTION_BD = "rgba(150, 130, 68, 0.26)"
    ALERT_BG = "rgba(148, 82, 82, 0.10)"
    ALERT_BD = "rgba(152, 88, 88, 0.26)"


class WaveformWidget(QWidget):
    """Premium Professional Stereo Waveform Visualization

    Features:
    - Dual-channel stereo display (L/R separated)
    - Peak/RMS envelope rendering
    - Professional gradient fills
    - Time axis with markers
    - Amplitude scale in dB
    - High-quality antialiasing
    - Per-stage restoration visualization overlay
    """

    # ── Stage visualization: category → (color, icon, label) ──────────────
    _STAGE_VISUALS: dict[str, tuple[tuple[int, int, int], str, str]] = {
        # ── Early pipeline stages (BatchThread + AurikDenker Stufen 1–8) ──
        # Reihenfolge wichtig: spezifische Matches VOR generischen.
        # "kette" VOR "tonträger" weil "Tonträgerkette" beide enthält.
        # "musik wird restauriert" VOR "restaurierung" weil sonst Fehlmatch.
        "audio wird geladen": ((0, 188, 212), "⚙", "Audio wird geladen"),
        "restaurierung startet": ((0, 188, 212), "⚙", "Audio wird geladen"),
        "kette": ((0, 155, 180), "🔗", "Tonträgerkette"),
        "tonträger": ((0, 172, 193), "💿", "Tonträger-Erkennung"),
        "schadensbewertung": ((0, 140, 168), "🔎", "Schadensbewertung"),
        "defekte werden": ((220, 120, 50), "🔎", "Defekt-Analyse"),
        "globalplan": ((100, 140, 210), "📋", "Restaurierungsplan"),
        "restaurierungsplan": ((100, 140, 210), "📋", "Restaurierungsplan"),
        "strategie": ((90, 130, 200), "🧭", "Strategie"),
        "vorverarbeitung": ((130, 100, 200), "⚙", "Vorverarbeitung"),
        "dsp-reparatur": ((255, 152, 0), "⚡", "DSP-Reparatur"),
        "musik wird restauriert": ((150, 100, 220), "🎵", "Restaurierung"),
        # ── Analysis stages (UV3 internal) ──
        "analyse": ((0, 188, 212), "🔍", "Analyse"),
        "scan": ((0, 188, 212), "🔍", "Scan"),
        "restaurierbarkeit": ((0, 188, 212), "🔍", "Analyse"),
        "erkennung": ((0, 188, 212), "🔍", "Erkennung"),
        "erkannt": ((0, 172, 193), "🔍", "Erkennung"),
        "phasenauswahl": ((0, 188, 212), "🔍", "Planung"),
        "initialisierung": ((0, 188, 212), "⚙", "Vorbereitung"),
        # Repair stages
        "click": ((255, 152, 0), "⚡", "Knackser-Reparatur"),
        "declick": ((255, 152, 0), "⚡", "Knackser-Entfernung"),
        "crackle": ((255, 152, 0), "⚡", "Knistern-Entfernung"),
        "declip": ((255, 82, 82), "⚡", "Übersteuerungs-Reparatur"),
        "spectral_repair": ((255, 82, 82), "⚡", "Spektral-Reparatur"),
        "hum_removal": ((156, 39, 176), "〰", "Brumm-Entfernung"),
        "dc_offset": ((200, 200, 80), "〰", "DC-Korrektur"),
        "transport_bump": ((150, 80, 180), "⚡", "Bandhopser-Reparatur"),
        "dropout": ((233, 30, 99), "🔧", "Lücken-Rekonstruktion"),
        "diffusion_inpainting": ((233, 30, 99), "🔧", "KI-Rekonstruktion"),
        # Noise reduction
        "denoise": ((100, 181, 246), "🔇", "Rauschunterdrückung"),
        "tape_hiss": ((100, 181, 246), "🔇", "Bandrauschen-Entfernung"),
        "noise_gate": ((100, 181, 246), "🔇", "Noise-Gate"),
        "surface_noise": ((100, 181, 246), "🔇", "Oberflächenrauschen"),
        "rumble_filter": ((96, 125, 139), "🔇", "Rumpel-Filter"),
        "advanced_dereverb": ((63, 137, 199), "🔇", "Nachhall-Entfernung"),
        "reverb_reduction": ((63, 137, 199), "🔇", "Nachhall-Reduktion"),
        # Frequency / spectral restoration
        "frequency_restoration": ((171, 71, 188), "📶", "Frequenz-Restauration"),
        "spectral_band_gap": ((171, 71, 188), "📶", "Spektral-Lückenfüllung"),
        "spectral_coherence": ((121, 85, 196), "📶", "Spektral-Kohärenz"),
        "harmonic_restoration": ((171, 71, 188), "📶", "Oberton-Ergänzung"),
        "eq_correction": ((102, 126, 234), "🎛", "Klangbalance"),
        "final_eq": ((102, 126, 234), "🎛", "Final-EQ"),
        # Vocal / instrument enhancement
        "vocal": ((255, 193, 7), "🎤", "Gesangs-Verbesserung"),
        "de_esser": ((0, 188, 212), "🎤", "De-Esser"),
        "ml_deesser": ((0, 188, 212), "🎤", "KI-De-Esser"),
        # Timing / pitch
        "wow_flutter": ((76, 175, 80), "🎵", "Tonhöhen-Korrektur"),
        "speed_pitch": ((76, 175, 80), "🎵", "Geschwindigkeits-Korrektur"),
        "azimuth_correction": ((76, 175, 80), "🎵", "Azimut-Korrektur"),
        "phase_correction": ((0, 137, 123), "🎵", "Phasen-Korrektur"),
        # Dynamics / mastering
        "compression": ((192, 192, 210), "🎚", "Dynamik-Bearbeitung"),
        "transient": ((255, 167, 38), "🎚", "Transienten-Formung"),
        "mastering_polish": ((192, 192, 210), "🎚", "Mastering"),
        "loudness_normalization": ((192, 192, 210), "🎚", "Lautstärke-Normierung"),
        "truepeak_limiter": ((192, 192, 210), "🎚", "TruePeak-Limiter"),
        "tape_saturation": ((188, 170, 164), "🎚", "Vintage-Charakter"),
        # Quality assessment
        "exzellenz": ((80, 200, 120), "✓", "Qualitätsprüfung"),
        "versa": ((80, 200, 120), "✓", "Qualitäts-Bewertung"),
        "qualitäts-gate": ((80, 200, 120), "✓", "Qualitäts-Gate"),
        "ergebnis": ((80, 200, 120), "💾", "Export"),
    }

    # ── Stage effect category mapping (keyword → animation type) ──────────
    _STAGE_EFFECTS: dict[str, str] = {
        # Scan/Analysis
        "analyse": "scan",
        "scan": "scan",
        "restaurierbarkeit": "scan",
        "erkennung": "scan",
        "erkannt": "scan",
        "phasenauswahl": "scan",
        "schadensbewertung": "scan",
        "defekte werden": "scan",
        "tonträger": "scan",
        "kette": "scan",
        # Repair/Impact
        "click": "repair",
        "declick": "repair",
        "crackle": "repair",
        "declip": "repair",
        "transport_bump": "repair",
        "spectral_repair": "repair",
        "hum_removal": "repair",
        "dc_offset": "repair",
        # Noise reduction
        "denoise": "noise",
        "tape_hiss": "noise",
        "noise_gate": "noise",
        "surface_noise": "noise",
        "rumble_filter": "noise",
        "advanced_dereverb": "noise",
        "reverb_reduction": "noise",
        # Dropout/Reconstruction
        "dropout": "reconstruct",
        "diffusion_inpainting": "reconstruct",
        # Spectral/Frequency
        "frequency_restoration": "spectral",
        "spectral_band_gap": "spectral",
        "spectral_coherence": "spectral",
        "harmonic_restoration": "spectral",
        "eq_correction": "spectral",
        "final_eq": "spectral",
        # Vocal
        "vocal": "vocal",
        "de_esser": "vocal",
        "ml_deesser": "vocal",
        # Timing/Pitch
        "wow_flutter": "timing",
        "speed_pitch": "timing",
        "azimuth_correction": "timing",
        "phase_correction": "timing",
        # Dynamics/Mastering
        "compression": "dynamics",
        "transient": "dynamics",
        "mastering_polish": "dynamics",
        "loudness_normalization": "dynamics",
        "truepeak_limiter": "dynamics",
        "tape_saturation": "dynamics",
        # Quality
        "exzellenz": "quality",
        "versa": "quality",
        "qualitäts-gate": "quality",
        "ergebnis": "quality",
        # Pipeline/Processing
        "initialisierung": "process",
        "vorverarbeitung": "process",
        "dsp-reparatur": "repair",
        "restaurierung startet": "process",
        "audio wird geladen": "process",
        "globalplan": "process",
        "strategie": "process",
        "restaurierungsplan": "process",
        "musik wird restauriert": "process",
    }

    # ── DSP/ML Tool color palette: unique color + icon per tool ───────────
    # Each tool that Aurik uses for defect repair gets its own visual identity
    # so the user can see exactly which algorithm fixed which defect region.
    _TOOL_COLORS: dict[str, tuple[tuple[int, int, int], str]] = {
        # ML Models — vibrant, saturated colors
        "DeepFilterNet": ((0, 180, 255), "🧠"),  # electric cyan
        "MelBandRoformer": ((180, 80, 255), "🎛"),  # vivid purple
        "MDX23C": ((255, 120, 200), "🔬"),  # hot pink
        "SGMSE+": ((0, 230, 180), "🌊"),  # turquoise-green
        "Resemble-Enhance": ((255, 180, 50), "✨"),  # warm gold
        "Apollo": ((255, 100, 60), "🚀"),  # bright orange
        "RMVPE": ((200, 255, 80), "🎵"),  # lime green
        "CREPE": ((80, 220, 120), "🎶"),  # fresh green
        "FCPE": ((100, 240, 100), "🎼"),  # green
        "AudioSR": ((160, 100, 255), "📡"),  # deep violet
        "Vocos": ((255, 215, 0), "💎"),  # gold
        "BigVGAN": ((220, 160, 255), "🔊"),  # soft lavender
        "PANNs": ((0, 200, 200), "👂"),  # teal
        "BEATs": ((50, 180, 220), "🎧"),  # sky blue
        "VERSA (Bewertung)": ((80, 200, 120), "📊"),  # mint green
        "Flow-Matching": ((230, 100, 255), "🌀"),  # magenta-violet
        "CQTdiff+": ((255, 140, 180), "🔮"),  # rose
        "VocalAI": ((255, 200, 80), "🎤"),  # warm yellow
        "ML-DeEsser": ((100, 210, 230), "🔉"),  # light cyan
        "WPE": ((80, 160, 220), "🏛"),  # steel blue
        # DSP Algorithms — cooler, more muted but distinct colors
        "DSP Notch-Filter": ((140, 100, 200), "〰"),  # soft purple
        "DSP Click-Repair": ((255, 140, 60), "⚡"),  # amber
        "DSP Decrackle": ((240, 170, 80), "⚙"),  # warm amber
        "DSP Hochpass": ((100, 140, 180), "🔽"),  # slate blue
        "DSP Noise-Profiling": ((120, 170, 220), "📉"),  # light blue
        "DSP Noise-Gate": ((90, 150, 200), "🚪"),  # medium blue
        "DSP EQ-Korrektur": ((130, 120, 220), "🎚"),  # indigo
        "DSP Harmonic-Synth": ((200, 130, 220), "🎹"),  # orchid
        "DSP Phase-Align": ((80, 180, 160), "🔄"),  # sea green
        "DSP Azimuth": ((100, 190, 170), "🧭"),  # aqua green
        "DSP PGHI": ((180, 120, 200), "📶"),  # purple-pink
        "DSP LMS-Adaptive": ((170, 150, 130), "📐"),  # taupe
        "DSP Transient-Guard": ((220, 180, 100), "🛡"),  # bronze
        "DSP Transient-Shaper": ((240, 190, 80), "⚔"),  # gold
        "DSP De-Esser": ((80, 200, 210), "🔇"),  # soft cyan
        "DSP Sättigungs-Emulation": ((190, 170, 150), "🎛"),  # warm grey
        "DSP Kompressor": ((170, 170, 200), "🎚"),  # soft steel
        "DSP LUFS-Norm": ((160, 180, 200), "📏"),  # light steel
        "DSP TruePeak": ((180, 180, 210), "📐"),  # lavender steel
        "DSP Final-EQ": ((140, 130, 220), "🎛"),  # blue-indigo
        "DSP Mastering": ((200, 190, 220), "💿"),  # pale lavender
        "DSP DC-Offset": ((190, 190, 100), "➖"),  # olive
        "DSP Declip": ((240, 90, 70), "🔧"),  # warm red
        "DSP Spectral-Coherence": ((140, 110, 210), "🌈"),  # soft violet
        "DSP Export-Optimierung": ((170, 200, 180), "💾"),  # sage green
    }

    # ── Mapping: phase keyword → tool name (mirrors _ML_PHASE_MARKERS) ────
    _PHASE_TO_TOOL: dict[str, str] = {
        "deepfilternet": "DeepFilterNet",
        "dfn": "DeepFilterNet",
        "melbandroformer": "MelBandRoformer",
        "bs_roformer": "MelBandRoformer",
        "mdx23c": "MDX23C",
        "sgmse": "SGMSE+",
        "resemble": "Resemble-Enhance",
        "apollo": "Apollo",
        "rmvpe": "RMVPE",
        "crepe": "CREPE",
        "audiosr": "AudioSR",
        "vocos": "Vocos",
        "bigvgan": "BigVGAN",
        "panns": "PANNs",
        "beats": "BEATs",
        "versa": "VERSA (Bewertung)",
        "flow_matching": "Flow-Matching",
        "cqtdiff": "CQTdiff+",
        "fcpe": "FCPE",
        "vocal_enhancement": "VocalAI",
        "ml_deesser": "ML-DeEsser",
        "reverb_reduction": "WPE",
        "advanced_dereverb": "WPE",
        "denoise": "DeepFilterNet",
        "tape_hiss": "DeepFilterNet",
        "diffusion_inpainting": "Flow-Matching",
        "dropout_repair": "AudioSR",
        "dropout": "AudioSR",
        "frequency_restoration": "AudioSR",
        "wow_flutter": "CREPE",
        "speed_pitch": "CREPE",
        "spectral_band_gap": "CQTdiff+",
        "hum_removal": "DSP Notch-Filter",
        "click_removal": "DSP Click-Repair",
        "click_pop": "DSP Click-Repair",
        "click": "DSP Click-Repair",
        "declick": "DSP Click-Repair",
        "crackle": "DSP Decrackle",
        "crackle_removal": "DSP Decrackle",
        "rumble_filter": "DSP Hochpass",
        "rumble": "DSP Hochpass",
        "surface_noise": "DSP Noise-Profiling",
        "noise_gate": "DSP Noise-Gate",
        "eq_correction": "DSP EQ-Korrektur",
        "harmonic_restoration": "DSP Harmonic-Synth",
        "phase_correction": "DSP Phase-Align",
        "azimuth_correction": "DSP Azimuth",
        "spectral_repair": "DSP PGHI",
        "print_through": "DSP LMS-Adaptive",
        "transient_preservation": "DSP Transient-Guard",
        "transient": "DSP Transient-Shaper",
        "de_esser": "DSP De-Esser",
        "tape_saturation": "DSP Sättigungs-Emulation",
        "compression": "DSP Kompressor",
        "loudness_normalization": "DSP LUFS-Norm",
        "truepeak_limiter": "DSP TruePeak",
        "final_eq": "DSP Final-EQ",
        "mastering_polish": "DSP Mastering",
        "dc_offset": "DSP DC-Offset",
        "declip": "DSP Declip",
        "spectral_coherence": "DSP Spectral-Coherence",
    }

    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = 44100
        self.is_loading = False  # Lade-Zustandsanzeige
        self.defects: dict = {}  # Defekte für Waveform-Overlay

        # Zoom/Pan state (fractions 0.0–1.0 of total duration)
        self._view_start: float = 0.0
        self._view_end: float = 1.0
        self._pan_anchor: int | None = None  # x-pixel at pan-press
        self._pan_view_start_at_press: float = 0.0
        # Timed defect locations: {"clicks": [(t_start, t_end), ...], ...}
        self._defect_locations: dict = {}
        # Per-channel defect locations: {"clicks": {"L": [...], "R": [...]}, ...}
        self._channel_locations: dict = {}
        # Resolved/fixed defect locations — kept for tool-specific "fixed" overlay
        self._resolved_locations: dict = {}  # {defect_key: [(t0, t1), ...]}
        # Repair history: which tool fixed which defect regions
        # {defect_key: [(t0, t1, tool_name), ...]}
        self._repair_history: dict[str, list[tuple[float, float, str]]] = {}
        # Current active tool name (detected from phase keywords)
        self._active_tool: str = ""

        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.setStyleSheet("""
            background: rgba(20, 20, 30, 0.95);
            border: 2px solid rgba(102, 126, 234, 0.4);
            border-radius: 10px;
        """)
        # Playhead position (0.0–1.0 fraction of total duration, -1.0 = hidden)
        self._playhead_pos: float = -1.0
        # Processed-region fraction: 0.0 = nothing processed, 1.0 = fully done. -1.0 = hidden.
        self._scan_pos: float = -1.0
        # ── Stage visualization state ────────────────────────────────────────
        self._active_stage: str = ""  # current stage keyword (e.g., "denoise")
        self._stage_color: tuple[int, int, int] = (0, 0, 0)
        self._stage_icon: str = ""
        self._stage_label: str = ""
        self._stage_progress: float = 0.0  # 0.0–1.0 progress within current stage
        self._stage_start_time: float = 0.0  # monotonic clock when stage started
        self._stage_history: list[str] = []  # completed stage labels for trail
        # Breathing-border animation for empty/idle state (20 fps)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(lambda: self.update() if self.audio_data is None else None)
        self._pulse_timer.start(50)

    def update_waveform(self, audio, sr):
        """Update waveform data and reset view window."""
        self.audio_data = audio
        self.sample_rate = sr
        # Reset zoom/pan to show full file on new load
        self._view_start = 0.0
        self._view_end = 1.0
        # Reset repair tracking for new audio
        self._repair_history.clear()
        self._resolved_locations.clear()
        self._active_tool = ""
        self.update()

    def update_audio_live(self, audio, sr):
        """Update waveform with phase-processed audio WITHOUT resetting zoom/pan.

        Called after each UV3 phase completes to reflect the actual audio changes
        in real time. Preserves the current view window so the user keeps their
        zoom/scroll position during processing.
        """
        self.audio_data = audio
        self.sample_rate = sr
        self.update()

    def set_defects(self, defects: dict) -> None:
        """Speichert Defekte für farbiges Severity-Overlay in der Wellenform.

        Progressive Defekt-Entfernung: Wenn ein Defekttyp-Score auf 0 sinkt (via
        _PHASE_REDUCES), werden seine Locations in _resolved_locations verschoben
        und dort als tool-spezifische "behoben"-Marker angezeigt.
        Die _repair_history speichert welches Tool welchen Defekt behoben hat.
        """
        self.defects = defects or {}
        new_locs = self.defects.get("_locations", {})
        self._channel_locations = self.defects.get("_channel_locations", {})

        # Progressive removal: detect which defect types just went to 0
        for dk, old_segs in list(self._defect_locations.items()):
            if not old_segs:
                continue
            new_score = self.defects.get(dk, 0)
            if isinstance(new_score, (int, float)) and new_score <= 0.01:
                # Score dropped to ~0 → defect resolved, move to resolved list
                existing = self._resolved_locations.get(dk, [])
                existing.extend(old_segs)
                self._resolved_locations[dk] = existing
                # Record which tool fixed this defect
                _tool = self._active_tool or "DSP"
                history = self._repair_history.get(dk, [])
                for seg in old_segs:
                    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                        history.append((float(seg[0]), float(seg[1]), _tool))
                self._repair_history[dk] = history

        self._defect_locations = new_locs
        self.update()

    def set_scan_pos(self, frac: float) -> None:
        """Set restoration scan-cursor position (0.0–1.0 fraction). Pass -1.0 to hide."""
        new_pos = float(frac)
        # Guard: skip repaint if position effectively unchanged (avoids spurious repaints
        # when scan_progress signal arrives with the same value multiple times).
        if abs(new_pos - self._scan_pos) < 0.002:
            return
        self._scan_pos = new_pos
        self.update()

    def set_active_stage(self, phase_text: str) -> None:
        """Set active restoration stage for waveform visualization overlay.

        Matches phase_text against _STAGE_VISUALS keywords to determine the
        visual category (color, icon, label). Also detects the active DSP/ML
        tool from _PHASE_TO_TOOL for tool-specific repair visualizations.
        """
        phase_lower = phase_text.lower()
        matched = False
        for keyword, (color, icon, label) in self._STAGE_VISUALS.items():
            if keyword in phase_lower:
                if self._active_stage != keyword:
                    # Stage transition: archive previous stage label
                    if self._stage_label and self._stage_label not in self._stage_history:
                        self._stage_history.append(self._stage_label)
                        # Keep only last 5 completed stages
                        if len(self._stage_history) > 5:
                            self._stage_history = self._stage_history[-5:]
                    self._active_stage = keyword
                    self._stage_color = color
                    self._stage_icon = icon
                    self._stage_label = label
                    self._stage_progress = 0.0
                    self._stage_start_time = time.monotonic()
                matched = True
                break
        if not matched and phase_lower:
            # Generic fallback: keep current stage active but don't change identity
            pass
        # Detect active DSP/ML tool from phase keywords
        _phase_underscored = phase_lower.replace(" ", "_")
        _detected_tool = ""
        for _kw, _tn in self._PHASE_TO_TOOL.items():
            if _kw in phase_lower or _kw in _phase_underscored:
                _detected_tool = _tn
        if _detected_tool:
            self._active_tool = _detected_tool
        self.update()

    def set_stage_progress(self, progress: float) -> None:
        """Update the per-stage progress (0.0–1.0)."""
        self._stage_progress = max(0.0, min(1.0, float(progress)))
        self.update()

    def clear_stage(self) -> None:
        """Clear all stage visualization state (called when processing finishes)."""
        self._active_stage = ""
        self._stage_color = (0, 0, 0)
        self._stage_icon = ""
        self._stage_label = ""
        self._stage_progress = 0.0
        self._stage_history.clear()
        self._active_tool = ""
        self.update()

    def set_view(self, start: float, end: float) -> None:
        """Programmatic zoom/pan: set view window to [start, end] fraction of total audio.

        Clamps to [0.0, 1.0] and enforces minimum span (0.0005 ≈ ~1 ms at 48 kHz / 1 min),
        so inpainting zoom can frame individual gap regions accurately.
        Emits a repaint request.
        """
        _min_span = 0.0005
        start = max(0.0, float(start))
        end = min(1.0, float(end))
        if end - start < _min_span:
            # Center the view around the mid-point
            mid = (start + end) / 2.0
            start = max(0.0, mid - _min_span / 2)
            end = min(1.0, start + _min_span)
        self._view_start = start
        self._view_end = end
        self.update()

    # ── Zoom / Pan interactions ───────────────────────────────────────────────

    def wheelEvent(self, event):
        """Zoom in/out centered on the mouse X position."""
        if self.audio_data is None:
            return
        # Signal ModernMainWindow that user manually overrode any auto-zoom
        # so the inpainting tour doesn't fight the user's preferred view.
        if hasattr(self, "_inpainting_user_override_cb") and callable(self._inpainting_user_override_cb):
            self._inpainting_user_override_cb()
        margin_left = 50
        margin_right = 20
        plot_w = max(1, self.width() - margin_left - margin_right)
        delta = event.angleDelta().y()
        factor = 0.80 if delta > 0 else 1.0 / 0.80
        span = self._view_end - self._view_start
        new_span = max(0.005, min(1.0, span * factor))
        # fraction of plot width where the mouse is
        frac = max(0.0, min(1.0, (event.pos().x() - margin_left) / plot_w))
        center = self._view_start + frac * span
        new_start = max(0.0, center - frac * new_span)
        new_end = min(1.0, new_start + new_span)
        # Clamp start if end was clipped
        new_start = max(0.0, new_end - new_span)
        self._view_start = new_start
        self._view_end = new_end
        self.update()

    def mousePressEvent(self, event):
        """Start pan drag on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton and self.audio_data is not None:
            self._pan_anchor = event.pos().x()
            self._pan_view_start_at_press = self._view_start
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Pan view while dragging."""
        if self._pan_anchor is not None and self.audio_data is not None:
            margin_left = 50
            margin_right = 20
            plot_w = max(1, self.width() - margin_left - margin_right)
            dx = event.pos().x() - self._pan_anchor
            span = self._view_end - self._view_start
            shift = -dx / plot_w * span
            new_start = max(0.0, min(1.0 - span, self._pan_view_start_at_press + shift))
            self._view_start = new_start
            self._view_end = new_start + span
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End pan drag."""
        self._pan_anchor = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        """Draw premium stereo waveform"""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.audio_data is None:
            w, h = self.width(), self.height()

            # Lade-Modus: andere Nachricht anzeigen
            if self.is_loading:
                painter.setPen(QPen(QColor(255, 193, 7, 120), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(20, 20, w - 40, h - 40, 18, 18)
                painter.setPen(QColor(255, 220, 100))
                font_big = QFont("Segoe UI", 16, QFont.Weight.Bold)
                painter.setFont(font_big)
                painter.drawText(
                    self.rect().adjusted(0, -20, 0, 0),
                    Qt.AlignmentFlag.AlignCenter,
                    t("ui.waveform_loading_title"),
                )
                painter.setPen(QColor(180, 160, 80))
                font_small = QFont("Segoe UI", 10)
                painter.setFont(font_small)
                painter.drawText(
                    self.rect().adjusted(0, 32, 0, 0),
                    Qt.AlignmentFlag.AlignCenter,
                    t("ui.waveform_loading_sub"),
                )
                return

            # Willkommens-Screen: animierte Drop-Zone-Anleitung
            # Hintergrund-Rahmen (gestrichelt, Breathing-Animation via QTimer)
            _pulse_alpha = int(80 + 45 * math.sin(time.monotonic() * 1.8))
            painter.setPen(QPen(QColor(102, 126, 234, _pulse_alpha), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(20, 20, w - 40, h - 40, 18, 18)

            # Haupttext
            painter.setPen(QColor(180, 190, 220))
            font_big = QFont("Segoe UI", 16, QFont.Weight.Bold)
            painter.setFont(font_big)
            painter.drawText(
                self.rect().adjusted(0, -30, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                t("ui.waveform_drop_title"),
            )
            # Subtext
            painter.setPen(QColor(130, 150, 180))
            font_small = QFont("Segoe UI", 10)
            painter.setFont(font_small)
            painter.drawText(
                self.rect().adjusted(0, 38, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                t("ui.waveform_drop_sub"),
            )
            # Formate
            painter.setPen(QColor(90, 110, 140))
            font_tiny = QFont("Segoe UI", 8)
            painter.setFont(font_tiny)
            painter.drawText(
                self.rect().adjusted(0, 72, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                t("ui.waveform_formats"),
            )
            return

        # Get dimensions with margins for axes
        margin_left = 50
        margin_right = 20
        margin_top = 10
        margin_bottom = 30

        plot_width = self.width() - margin_left - margin_right
        plot_height = self.height() - margin_top - margin_bottom
        plot_x = margin_left
        plot_y = margin_top

        if plot_width <= 0 or plot_height <= 0:
            return

        # Prepare audio data — apply zoom window
        n_total = len(self.audio_data) if len(self.audio_data.shape) == 1 else self.audio_data.shape[0]
        s_start = int(self._view_start * n_total)
        s_end = max(s_start + 1, int(self._view_end * n_total))

        if len(self.audio_data.shape) > 1 and self.audio_data.shape[1] == 2:
            # Stereo - draw both channels
            left_channel = self.audio_data[s_start:s_end, 0]
            right_channel = self.audio_data[s_start:s_end, 1]

            # Draw stereo waveforms (split vertically)
            channel_height = plot_height // 2 - 5

            # Left channel (top)
            self._draw_channel(
                painter, left_channel, plot_x, plot_y, plot_width, channel_height, (102, 126, 234), (118, 75, 162), "L"
            )

            # Right channel (bottom)
            self._draw_channel(
                painter,
                right_channel,
                plot_x,
                plot_y + channel_height + 10,
                plot_width,
                channel_height,
                (234, 102, 126),
                (162, 75, 118),
                "R",
            )
        else:
            # Mono - draw single waveform
            if len(self.audio_data.shape) > 1:
                audio = np.mean(self.audio_data[s_start:s_end], axis=1)
            else:
                audio = self.audio_data[s_start:s_end]

            self._draw_channel(
                painter, audio, plot_x, plot_y, plot_width, plot_height, (102, 126, 234), (118, 75, 162), "M"
            )

        # Draw time axis
        self._draw_time_axis(painter, plot_x, plot_y + plot_height + 5, plot_width)

        # Defekt-Severity-Overlay (farbige Bänder + Badge)
        self._draw_defect_overlay(painter, plot_x, plot_y, plot_width, plot_height)

        # Stage-Visualization-Overlay (aktive Restaurierungsstufe)
        self._draw_stage_overlay(painter, plot_x, plot_y, plot_width, plot_height)

        # Lyrics-Timeline-Overlay (§2.36, Taste L) — nur wenn aktiv
        _lt = getattr(self, "_lyrics_transcription", None)
        if _lt is not None:
            try:
                _lge = _bridge_get_lyrics_guided_enhancement()
                if _lge is not None:
                    _dur = len(self.audio_data) / self.sample_rate if self.sample_rate > 0 else 0.0
                    _lge.get_timeline().render_overlay(painter, _lt, plot_width, _dur)
            except Exception:
                pass

        # ── Playhead (Spielkopf-Cursor während Wiedergabe) ────────────────────
        if 0.0 <= self._playhead_pos <= 1.0:
            _view_span = max(1e-9, self._view_end - self._view_start)
            _view_frac = (self._playhead_pos - self._view_start) / _view_span
            if 0.0 <= _view_frac <= 1.0:
                _ph_x = int(plot_x + _view_frac * plot_width)
                # Weißer vertikaler Spielkopf-Strich
                painter.setPen(QPen(QColor(255, 255, 255, 220), 2, Qt.PenStyle.SolidLine))
                painter.drawLine(_ph_x, plot_y, _ph_x, plot_y + plot_height)
                # Dreieck-Anfasser oben
                _tri = QPolygonF()
                _tri.append(QPointF(_ph_x - 6, plot_y))
                _tri.append(QPointF(_ph_x + 6, plot_y))
                _tri.append(QPointF(_ph_x, plot_y + 12))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
                painter.drawPolygon(_tri)
                # Zeitstempel neben dem Cursor
                if self.audio_data is not None and self.sample_rate > 0:
                    _n = self.audio_data.shape[0]
                    _t_s = self._playhead_pos * (_n / self.sample_rate)
                    _m, _s = divmod(int(_t_s), 60)
                    _t_str = f"{_m}:{_s:02d}"
                    _lx = _ph_x + 5 if _ph_x + 38 <= plot_x + plot_width else _ph_x - 36
                    painter.setPen(QColor(255, 255, 200, 210))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
                    painter.drawText(_lx, plot_y + 12, _t_str)

        # ── Scan-Cursor: only visible while DefectScanner is actively reading ──
        # Once status changes to "correcting" or "completed", phases process the
        # *entire* audio from t=0 — there is no spatially meaningful cursor
        # position during that time.  We hide the cursor to avoid implying that
        # "everything left of the cursor is already done".
        _scan_status = self.defects.get("status", "detected") if self.defects else "detected"
        _scan_active = _scan_status not in ("correcting", "completed")
        if _scan_active and 0.0 <= self._scan_pos <= 1.0:
            _view_span_sc = max(1e-9, self._view_end - self._view_start)
            _sc_vfrac = (self._scan_pos - self._view_start) / _view_span_sc
            if 0.0 <= _sc_vfrac <= 1.0:
                _sc_x = int(plot_x + _sc_vfrac * plot_width)
                # Glow (breit, halbtransparent)
                painter.setPen(QPen(QColor(255, 150, 30, 45), 12, Qt.PenStyle.SolidLine))
                painter.drawLine(_sc_x, plot_y, _sc_x, plot_y + plot_height)
                # Core-Linie (gestrichelt, opak)
                painter.setPen(QPen(QColor(255, 178, 55, 215), 2, Qt.PenStyle.DashLine))
                painter.drawLine(_sc_x, plot_y, _sc_x, plot_y + plot_height)

    def _draw_channel(
        self,
        painter,
        audio,
        x,
        y,
        width,
        height,
        color1: tuple[int, int, int],
        color2: tuple[int, int, int],
        label,
    ):
        """Draw a single audio channel with dynamic auto-scaling and RMS envelope.

        The waveform is normalized to the true signal peak so it always fills
        ~96 % of the available channel height, independent of the input level.
        A secondary RMS-envelope line is drawn on top of the peak fill for
        visual richness similar to professional metering tools.
        """
        center_y = y + height // 2

        # ── Dynamic auto-scale ───────────────────────────────────────────────
        peak_amplitude = float(np.max(np.abs(audio)))
        if peak_amplitude < 1e-6:
            # Silent channel – draw center line only
            painter.setPen(QPen(QColor(100, 100, 120, 80), 1, Qt.PenStyle.DashLine))
            painter.drawLine(x, center_y, x + width, center_y)
            painter.setPen(QColor(180, 180, 200))
            font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(
                x - 40, center_y - 10, 30, 20, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label
            )
            return

        # Scale so the true peak fills 48 % of half-height → 96 % of channel
        half_h = height * 0.48
        px_scale = half_h / peak_amplitude  # pixels per amplitude unit

        # ── Build per-pixel min/max and RMS envelope ─────────────────────────
        # Use floating-point step to avoid missing edge pixels
        n_samples = len(audio)
        step = n_samples / width  # samples per pixel (float)

        points_top: list[tuple[float, float]] = []
        points_bottom: list[tuple[float, float]] = []
        rms_top: list[tuple[float, float]] = []
        rms_bottom: list[tuple[float, float]] = []

        for pixel in range(width):
            s0 = int(pixel * step)
            s1 = min(int((pixel + 1) * step) + 1, n_samples)
            seg = audio[s0:s1]
            if len(seg) == 0:
                continue
            p_max = float(np.max(seg))
            p_min = float(np.min(seg))
            rms = float(np.sqrt(np.mean(seg * seg)))

            px = x + pixel
            yt = center_y - p_max * px_scale
            yb = center_y - p_min * px_scale
            yr_t = center_y - rms * px_scale
            yr_b = center_y + rms * px_scale

            # Clamp to channel bounds
            yt = max(y, min(y + height, yt))
            yb = max(y, min(y + height, yb))
            yr_t = max(y, min(y + height, yr_t))
            yr_b = max(y, min(y + height, yr_b))

            points_top.append((px, yt))
            points_bottom.insert(0, (px, yb))
            rms_top.append((px, yr_t))
            rms_bottom.append((px, yr_b))

        # ── Draw filled peak envelope ─────────────────────────────────────────
        if points_top and points_bottom:
            gradient = QLinearGradient(0, y, 0, y + height)
            gradient.setColorAt(0, QColor(*color1, 130))
            gradient.setColorAt(0.5, QColor(*color2, 90))
            gradient.setColorAt(1, QColor(*color1, 130))

            polygon = QPolygonF()
            for px, py in points_top:
                polygon.append(QPointF(px, py))
            for px, py in points_bottom:
                polygon.append(QPointF(px, py))

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(polygon)

            # Peak outline (top + bottom)
            outline_gradient = QLinearGradient(0, y, 0, y + height)
            outline_gradient.setColorAt(0, QColor(*color1, 230))
            outline_gradient.setColorAt(1, QColor(*color2, 230))
            painter.setPen(QPen(QBrush(outline_gradient), 1.5))

            path_top = QPainterPath()
            path_top.moveTo(points_top[0][0], points_top[0][1])
            for px, py in points_top[1:]:
                path_top.lineTo(px, py)
            painter.drawPath(path_top)

            path_bottom = QPainterPath()
            path_bottom.moveTo(points_bottom[-1][0], points_bottom[-1][1])
            for px, py in reversed(points_bottom[:-1]):
                path_bottom.lineTo(px, py)
            painter.drawPath(path_bottom)

        # ── RMS envelope overlay (brighter inner contour) ─────────────────────
        if rms_top:
            rms_color = QColor(*color1, 200)
            painter.setPen(QPen(rms_color, 1.0))
            path_rms_t = QPainterPath()
            path_rms_b = QPainterPath()
            path_rms_t.moveTo(rms_top[0][0], rms_top[0][1])
            path_rms_b.moveTo(rms_bottom[0][0], rms_bottom[0][1])
            for (px, yt), (_, yb) in zip(rms_top[1:], rms_bottom[1:]):
                path_rms_t.lineTo(px, yt)
                path_rms_b.lineTo(px, yb)
            painter.drawPath(path_rms_t)
            painter.drawPath(path_rms_b)

        # ── Center line ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(100, 100, 120, 80), 1, Qt.PenStyle.DashLine))
        painter.drawLine(x, center_y, x + width, center_y)

        # ── Channel label ─────────────────────────────────────────────────────
        painter.setPen(QColor(180, 180, 200))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            x - 40, center_y - 10, 30, 20, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label
        )

        # ── Adaptive dB scale (only ticks within visible range) ───────────────
        painter.setPen(QColor(150, 150, 170, 150))
        font_small = QFont("Segoe UI", 7)
        painter.setFont(font_small)

        for db in [0, -6, -12, -18, -24, -36, -48]:
            amp = 10 ** (db / 20.0)
            y_pos = center_y - amp * px_scale
            if y_pos < y or y_pos > y + height:
                continue  # outside visible area
            painter.setPen(QColor(150, 150, 170, 150))
            painter.drawLine(x - 5, int(y_pos), x, int(y_pos))
            painter.drawText(
                x - 45, int(y_pos) - 8, 40, 16, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{db}dB"
            )

        # ── Peak dBFS annotation (top-left of channel) ────────────────────────
        peak_db = 20.0 * np.log10(peak_amplitude)
        painter.setPen(QColor(200, 220, 255, 170))
        font_peak = QFont("Segoe UI", 7)
        painter.setFont(font_peak)
        painter.drawText(
            x + 4,
            y + 2,
            width - 8,
            14,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"↑ {peak_db:.1f} dBFS",
        )

    def _draw_time_axis(self, painter, x, y, width):
        """Draw time axis with markers — zoom/pan aware."""
        if self.audio_data is None or self.sample_rate == 0:
            return

        n_total = self.audio_data.shape[0]
        total_duration = n_total / self.sample_rate
        view_start_sec = self._view_start * total_duration
        view_end_sec = self._view_end * total_duration
        view_dur = max(1e-6, view_end_sec - view_start_sec)

        painter.setPen(QColor(150, 150, 170, 150))
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        num_markers = min(10, max(2, int(view_dur) + 1))
        for i in range(num_markers):
            frac = i / (num_markers - 1) if num_markers > 1 else 0
            time_sec = view_start_sec + frac * view_dur
            x_pos = x + frac * width

            painter.drawLine(int(x_pos), int(y), int(x_pos), int(y + 5))
            painter.drawText(int(x_pos - 30), int(y + 8), 60, 20, Qt.AlignmentFlag.AlignCenter, f"{time_sec:.1f}s")

        # Zoom indicator: show current zoom level if not at 100 %
        if self._view_end - self._view_start < 0.99:
            zoom_factor = 1.0 / max(0.001, self._view_end - self._view_start)
            painter.setPen(QColor(102, 126, 234, 180))
            font_z = QFont("Segoe UI", 7)
            painter.setFont(font_z)
            painter.drawText(
                int(x + width - 80),
                int(y + 8),
                78,
                16,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"🔍 ×{zoom_factor:.1f}",
            )

    def _draw_defect_overlay(self, painter, x, y, width, height):
        """Channel-accurate defect markers + resolved-overlay + bottom legend.

        Improvements over previous version:
        - Per-channel (L/R) markers drawn in the correct half of the stereo display
        - Resolved defects briefly shown as green "fixed" markers before fading
        - Processed-region tinted green instead of a confusing scan-cursor line

        The view window (_view_start / _view_end) is respected so markers remain
        accurate after zoom/pan.
        """
        if self.audio_data is None:
            return

        _DEFECT_COLORS = {
            "clicks": QColor(255, 82, 82),
            "crackle": QColor(255, 152, 0),
            "pops": QColor(255, 193, 7),
            "clipping": QColor(220, 50, 30),
            "hum": QColor(156, 39, 176),
            "noise_level": QColor(100, 181, 246),
            "noise": QColor(80, 160, 230),
            "sibilance": QColor(0, 188, 212),
            "dropout": QColor(233, 30, 99),
            "wow": QColor(76, 175, 80),
            "flutter": QColor(129, 199, 132),
            "rumble": QColor(96, 125, 139),
            "dc_offset": QColor(200, 200, 80),
            "digital_artifacts": QColor(255, 111, 0),
            "compression_artifacts": QColor(255, 87, 162),
            "stereo_imbalance": QColor(29, 233, 182),
            "phase_issues": QColor(0, 137, 123),
            "bandwidth_loss": QColor(121, 85, 196),
            "pitch_drift": QColor(255, 214, 0),
            "reverb_excess": QColor(63, 137, 199),
            "print_through": QColor(161, 136, 127),
            "quantization_noise": QColor(84, 110, 122),
            "jitter_artifacts": QColor(230, 238, 156),
            "dynamic_compression_excess": QColor(244, 67, 54),
            "pre_echo": QColor(240, 98, 146),
            "transient_smearing": QColor(255, 167, 38),
            "head_wear": QColor(188, 170, 164),
            "riaa_curve_error": QColor(77, 182, 172),
            "aliasing": QColor(171, 71, 188),
            "bias_error": QColor(255, 112, 67),
            "transport_bump": QColor(150, 80, 180),
        }
        _DEFECT_LABELS = {
            "clicks": "Knackser",
            "crackle": "Knistern",
            "pops": "Impulse",
            "clipping": "Übersteuerung",
            "hum": "Netzbrummen",
            "noise_level": "Rauschen",
            "noise": "Rauschen",
            "sibilance": "Zischlaute",
            "dropout": "Tonaussetzer",
            "wow": "Tonhöhenschw.",
            "flutter": "Tonhöhenzittern",
            "rumble": "Rumpeln",
            "dc_offset": "Gleichspannung",
            "digital_artifacts": "Digitale Artefakte",
            "compression_artifacts": "Kompressions-Art.",
            "stereo_imbalance": "Stereo-Balance",
            "phase_issues": "Phasenfehler",
            "bandwidth_loss": "Bandbreitenverlust",
            "pitch_drift": "Tonhöhendrift",
            "reverb_excess": "Überhall",
            "print_through": "Bandübersprechen",
            "quantization_noise": "Quant.-Rauschen",
            "jitter_artifacts": "Zeit-Flattern",
            "dynamic_compression_excess": "Lautstärkekompr.",
            "pre_echo": "Vorecho",
            "transient_smearing": "Transienten-Vers.",
            "head_wear": "Tonkopf-Abnutz.",
            "riaa_curve_error": "RIAA-Fehler",
            "aliasing": "Frequenz-Aliasing",
            "bias_error": "Vormagnetisierung",
            "transport_bump": "Bandhopser",
        }
        _SKIP_KEYS = {"status", "_locations", "_channel_locations", "_no_anim"}
        BAND_H = 5  # §11.4: 5-px indicator strip at bottom per active defect type

        n_total = self.audio_data.shape[0]
        total_dur = n_total / max(1, self.sample_rate)
        view_start_s = self._view_start * total_dur
        view_end_s = self._view_end * total_dur
        view_dur = max(1e-6, view_end_s - view_start_s)

        is_stereo = self.audio_data.ndim > 1 and self.audio_data.shape[1] == 2
        channel_height = height // 2 - 5 if is_stereo else height

        # ── Helper: time → pixel x ──────────────────────────────────────────
        def _t2px(t_s: float) -> int:
            return int(x + (t_s - view_start_s) / view_dur * width)

        # ── Active defect set — show ALL detected defects (score > 0) ───────
        # User requirement: "Es sollen alle erkannten Defekte angezeigt werden."
        # Minimal threshold 0.01 filters only numerical noise / true zeros.
        _DISPLAY_THRESHOLD = 0.01

        active_keys: list[str] = []
        if self.defects:
            for k, v in self.defects.items():
                if k in _SKIP_KEYS or not isinstance(v, (int, float)):
                    continue
                if v >= _DISPLAY_THRESHOLD:
                    active_keys.append(k)

        has_active = bool(active_keys)
        has_resolved = bool(self._resolved_locations)
        if not has_active and not has_resolved:
            return

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)

        # ── 1. Status-dependent spatial overlay ──────────────────────────────
        # IMPORTANT: Each restoration phase processes the ENTIRE audio from
        # t=0 to t=end. There is no "already done" region on the timeline
        # during phase execution — showing a partial green tint would be
        # misleading (it would imply the left portion is finished while later
        # phases will re-process it from the start).
        #
        # Visual contract:
        #   scan phase ("detected", _scan_pos 0→1): orange cursor in paintEvent
        #     shows scanner advancing — spatially meaningful.
        #   phase execution ("correcting"): NO spatial clip overlay.
        #     Defect-marker fadeout communicates per-defect progress instead.
        #   completed ("completed"): subtle full-width green tint — the entire
        #     song has been processed by all selected phases.
        _wf_status_overlay = self.defects.get("status", "detected") if self.defects else "detected"
        if _wf_status_overlay == "completed":
            painter.setBrush(QBrush(QColor(80, 200, 120, 14)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(x), int(y), int(width), int(height))
        # "correcting": no overlay — each phase starts from t=0 on the full audio

        # ── 2. Active defect markers — CHANNEL-AWARE (Premium) ──────────────
        # Premium-Konzept: Defekte werden kanalgenau auf L oder R gezeichnet,
        # NICHT als durchgehende Striche über beide Kanäle.
        # - Per-channel locations vorhanden → nur im betroffenen Kanal zeichnen
        # - Per-channel locations NICHT vorhanden aber Stereo → Audio-Energie
        #   pro Kanal an der Defekt-Position vergleichen → kanal-gewichtete
        #   Darstellung (höhere Opazität im stärker betroffenen Kanal)
        # - Mono → volle Höhe (einziger Kanal)

        # Stereo layout geometry
        _ch_L_y = y
        _ch_L_h = channel_height
        _ch_R_y = y + channel_height + 10
        _ch_R_h = channel_height
        _ch_gap = 10  # gap between L and R

        for defect_key, locations in self._defect_locations.items():
            if not locations:
                continue
            _curr_score = self.defects.get(defect_key, 0) if self.defects else 0
            if not isinstance(_curr_score, (int, float)) or _curr_score <= 0.01:
                continue
            base = _DEFECT_COLORS.get(defect_key, QColor(180, 180, 180))
            _br, _bg_c, _bb = base.red(), base.green(), base.blue()

            # Check for per-channel locations
            ch_data = self._channel_locations.get(defect_key, {})
            has_ch = bool(ch_data) and is_stereo

            def _draw_defect_marker(_px0, _px1, _my, _mh, _alpha_scale=1.0):
                """Draw a single defect marker with gradient + glow.

                _alpha_scale: 0.0–1.0 to modulate opacity for channel-weighted
                rendering (e.g. weaker channel gets lower alpha).
                """
                _mw = max(3, _px1 - _px0)
                _a65 = int(65 * _alpha_scale)
                _a35 = int(35 * _alpha_scale)
                _a60 = int(60 * _alpha_scale)
                # Gradient fill (top-to-bottom, stronger at edges)
                _fg = QLinearGradient(_px0, _my, _px0, _my + _mh)
                _fg.setColorAt(0.0, QColor(_br, _bg_c, _bb, _a65))
                _fg.setColorAt(0.5, QColor(_br, _bg_c, _bb, _a35))
                _fg.setColorAt(1.0, QColor(_br, _bg_c, _bb, _a60))
                painter.setBrush(QBrush(_fg))
                painter.drawRect(_px0, int(_my), _mw, int(_mh))
                # Left edge glow (bright)
                painter.setPen(QPen(QColor(_br, _bg_c, _bb, int(210 * _alpha_scale)), 1.5))
                painter.drawLine(_px0, int(_my), _px0, int(_my + _mh))
                # Subtle right edge
                if _mw > 6:
                    painter.setPen(QPen(QColor(_br, _bg_c, _bb, int(80 * _alpha_scale)), 0.8))
                    painter.drawLine(_px0 + _mw, int(_my), _px0 + _mw, int(_my + _mh))
                # Top/bottom edge highlight (1px)
                painter.setPen(QPen(QColor(_br, _bg_c, _bb, int(90 * _alpha_scale)), 0.6))
                painter.drawLine(_px0, int(_my), _px0 + _mw, int(_my))
                painter.drawLine(_px0, int(_my + _mh), _px0 + _mw, int(_my + _mh))
                painter.setPen(Qt.PenStyle.NoPen)

            if has_ch:
                # ── Premium: per-channel markers — only in the affected channel ──
                for ch_label, ch_y, ch_h in (("L", _ch_L_y, _ch_L_h), ("R", _ch_R_y, _ch_R_h)):
                    ch_locs = ch_data.get(ch_label, [])
                    for seg in ch_locs:
                        if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                            continue
                        t_start, t_end = float(seg[0]), float(seg[1])
                        if t_end < view_start_s or t_start > view_end_s:
                            continue
                        px0 = _t2px(max(t_start, view_start_s))
                        px1 = max(px0 + 3, _t2px(min(t_end, view_end_s)))
                        _draw_defect_marker(px0, px1, ch_y, ch_h)
            elif is_stereo and self.audio_data is not None and self.audio_data.ndim > 1:
                # ── Premium fallback: no per-channel data, but stereo audio ──
                # Compare energy per channel at each defect location to weight
                # the marker opacity per channel. This avoids the "über einen
                # Kamm" problem for defects without explicit L/R annotations.
                for seg in locations:
                    if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                        continue
                    t_start, t_end = float(seg[0]), float(seg[1])
                    if t_end < view_start_s or t_start > view_end_s:
                        continue
                    px0 = _t2px(max(t_start, view_start_s))
                    px1 = max(px0 + 3, _t2px(min(t_end, view_end_s)))
                    # Compute per-channel energy at defect location
                    s0 = max(0, int(t_start * self.sample_rate))
                    s1 = min(n_total, int(t_end * self.sample_rate))
                    if s1 <= s0:
                        s1 = min(n_total, s0 + 1)
                    _seg_L = self.audio_data[s0:s1, 0]
                    _seg_R = self.audio_data[s0:s1, 1]
                    _rms_L = float(np.sqrt(np.mean(_seg_L**2) + 1e-12))
                    _rms_R = float(np.sqrt(np.mean(_seg_R**2) + 1e-12))
                    _rms_max = max(_rms_L, _rms_R, 1e-10)
                    # Alpha scale: channel with higher energy → full opacity,
                    # other channel → proportionally reduced (min 0.15 so always visible)
                    _alpha_L = max(0.15, _rms_L / _rms_max)
                    _alpha_R = max(0.15, _rms_R / _rms_max)
                    _draw_defect_marker(px0, px1, _ch_L_y, _ch_L_h, _alpha_L)
                    _draw_defect_marker(px0, px1, _ch_R_y, _ch_R_h, _alpha_R)
            else:
                # Mono: full height
                for seg in locations:
                    if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                        continue
                    t_start, t_end = float(seg[0]), float(seg[1])
                    if t_end < view_start_s or t_start > view_end_s:
                        continue
                    px0 = _t2px(max(t_start, view_start_s))
                    px1 = max(px0 + 3, _t2px(min(t_end, view_end_s)))
                    _draw_defect_marker(px0, px1, y, height)

        # ── 2b. Global defect tints — CHANNEL-AWARE (Premium) ──────────────
        # Defects with severity > threshold but no time-specific locations (e.g.
        # hum, noise, dc_offset, bandwidth_loss, wow, flutter, phase_issues,
        # stereo_imbalance, etc.) are visualized as per-channel tints based on
        # actual signal characteristics — no "über einen Kamm" for stereo.
        for gk in active_keys:
            _gk_locs = self._defect_locations.get(gk, [])
            if _gk_locs:
                continue  # already drawn with precise markers above
            base_g = _DEFECT_COLORS.get(gk, QColor(180, 180, 180))
            _gr, _gg, _gb = base_g.red(), base_g.green(), base_g.blue()

            if is_stereo and self.audio_data is not None and self.audio_data.ndim > 1:
                # Per-channel global RMS → channel-specific tint alphas
                _g_rms_L = float(np.sqrt(np.mean(self.audio_data[:, 0] ** 2) + 1e-12))
                _g_rms_R = float(np.sqrt(np.mean(self.audio_data[:, 1] ** 2) + 1e-12))
                _g_rms_max = max(_g_rms_L, _g_rms_R, 1e-10)
                # For diffuse defects scale alpha by relative channel energy
                # (min 0.3 → both channels always show some tint)
                for _tch_y, _tch_h, _tch_rms in (
                    (_ch_L_y, _ch_L_h, _g_rms_L),
                    (_ch_R_y, _ch_R_h, _g_rms_R),
                ):
                    _tch_scale = max(0.3, _tch_rms / _g_rms_max)
                    _a_outer = int(18 * _tch_scale)
                    _a_inner = int(10 * _tch_scale)
                    _gt = QLinearGradient(int(x), int(_tch_y), int(x), int(_tch_y + _tch_h))
                    _gt.setColorAt(0.0, QColor(_gr, _gg, _gb, _a_outer))
                    _gt.setColorAt(0.35, QColor(_gr, _gg, _gb, _a_inner))
                    _gt.setColorAt(0.65, QColor(_gr, _gg, _gb, _a_inner))
                    _gt.setColorAt(1.0, QColor(_gr, _gg, _gb, _a_outer))
                    painter.setBrush(QBrush(_gt))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(int(x), int(_tch_y), int(width), int(_tch_h))
                    # Edge accent line per channel
                    painter.setPen(QPen(QColor(_gr, _gg, _gb, int(60 * _tch_scale)), 1.0))
                    painter.drawLine(int(x), int(_tch_y), int(x + width), int(_tch_y))
                    painter.setPen(Qt.PenStyle.NoPen)
            else:
                # Mono: full-width tint
                _gt = QLinearGradient(int(x), int(y), int(x), int(y + height))
                _gt.setColorAt(0.0, QColor(_gr, _gg, _gb, 18))
                _gt.setColorAt(0.35, QColor(_gr, _gg, _gb, 10))
                _gt.setColorAt(0.65, QColor(_gr, _gg, _gb, 10))
                _gt.setColorAt(1.0, QColor(_gr, _gg, _gb, 18))
                painter.setBrush(QBrush(_gt))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(int(x), int(y), int(width), int(height))
                painter.setPen(QPen(QColor(_gr, _gg, _gb, 60), 1.0))
                painter.drawLine(int(x), int(y), int(x + width), int(y))
                painter.setPen(Qt.PenStyle.NoPen)

        # ── 3. Resolved defect markers — CHANNEL-AWARE tool-colored overlays ──
        # Each repaired region is colored by the DSP/ML tool that fixed it.
        # Premium: In stereo mode, resolved markers are drawn per-channel using
        # energy-weighted opacity (same approach as active markers).

        def _draw_resolved_rect(_px0, _px1, _ry, _rh, _tr, _tg, _tb, _show_check=True):
            """Draw a single resolved-region marker in tool color."""
            _mw = _px1 - _px0
            _rg = QLinearGradient(_px0, int(_ry), _px0, int(_ry + _rh))
            _rg.setColorAt(0.0, QColor(_tr, _tg, _tb, 50))
            _rg.setColorAt(0.3, QColor(_tr, _tg, _tb, 25))
            _rg.setColorAt(0.7, QColor(_tr, _tg, _tb, 25))
            _rg.setColorAt(1.0, QColor(_tr, _tg, _tb, 45))
            painter.setBrush(QBrush(_rg))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(_px0, int(_ry), _mw, int(_rh))
            # Left edge glow
            painter.setPen(QPen(QColor(_tr, _tg, _tb, 180), 1.5))
            painter.drawLine(_px0, int(_ry), _px0, int(_ry + _rh))
            if _mw > 6:
                painter.setPen(QPen(QColor(_tr, _tg, _tb, 70), 0.8))
                painter.drawLine(_px1, int(_ry), _px1, int(_ry + _rh))
            painter.setPen(QPen(QColor(_tr, _tg, _tb, 100), 0.6))
            painter.drawLine(_px0, int(_ry), _px1, int(_ry))
            painter.drawLine(_px0, int(_ry + _rh), _px1, int(_ry + _rh))
            if _show_check and _mw > 10:
                painter.setPen(QColor(_tr, _tg, _tb, 200))
                painter.setFont(QFont("Segoe UI", 6))
                painter.drawText(_px0 + 2, int(_ry) + 9, "✓")
            painter.setPen(Qt.PenStyle.NoPen)

        if self._repair_history:
            for _rk, _r_entries in self._repair_history.items():
                for _entry in _r_entries:
                    if not (isinstance(_entry, (list, tuple)) and len(_entry) >= 3):
                        continue
                    t_start, t_end, tool_name = float(_entry[0]), float(_entry[1]), str(_entry[2])
                    if t_end < view_start_s or t_start > view_end_s:
                        continue
                    px0 = _t2px(max(t_start, view_start_s))
                    px1 = max(px0 + 3, _t2px(min(t_end, view_end_s)))
                    _tc = self._TOOL_COLORS.get(tool_name, ((80, 200, 120), "✓"))
                    _tr, _tg, _tb = _tc[0]
                    if is_stereo and self.audio_data is not None and self.audio_data.ndim > 1:
                        # Channel-aware: draw in each channel half
                        _draw_resolved_rect(px0, px1, _ch_L_y, _ch_L_h, _tr, _tg, _tb, True)
                        _draw_resolved_rect(px0, px1, _ch_R_y, _ch_R_h, _tr, _tg, _tb, False)
                    else:
                        _draw_resolved_rect(px0, px1, y, height, _tr, _tg, _tb, True)
        # Fallback: resolved locations without repair history (legacy path)
        elif self._resolved_locations:
            _resolved_color_L = QColor(80, 200, 120, 35)
            _resolved_edge_L = QColor(80, 200, 120, 140)
            for _rk, _r_segs in self._resolved_locations.items():
                for seg in _r_segs:
                    if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                        continue
                    t_start, t_end = float(seg[0]), float(seg[1])
                    if t_end < view_start_s or t_start > view_end_s:
                        continue
                    px0 = _t2px(max(t_start, view_start_s))
                    px1 = max(px0 + 3, _t2px(min(t_end, view_end_s)))
                    if is_stereo:
                        for _rch_y, _rch_h in ((_ch_L_y, _ch_L_h), (_ch_R_y, _ch_R_h)):
                            painter.setBrush(QBrush(_resolved_color_L))
                            painter.drawRect(px0, int(_rch_y), px1 - px0, int(_rch_h))
                            painter.setPen(QPen(_resolved_edge_L, 1.0))
                            painter.drawLine(px0, int(_rch_y), px0, int(_rch_y + _rch_h))
                            painter.setPen(Qt.PenStyle.NoPen)
                    else:
                        painter.setBrush(QBrush(_resolved_color_L))
                        painter.drawRect(px0, int(y), px1 - px0, int(height))
                        painter.setPen(QPen(_resolved_edge_L, 1.0))
                        painter.drawLine(px0, int(y), px0, int(y + height))
                        painter.setPen(Qt.PenStyle.NoPen)

        # ── 4. Status badge (rounded pill with dark background) ────────────
        n_active = len(active_keys)
        n_resolved = sum(len(v) for v in self._resolved_locations.values())
        n_repaired = sum(len(v) for v in self._repair_history.values())
        if n_active > 0 or n_resolved > 0 or n_repaired > 0:
            _wf_status = self.defects.get("status", "detected") if self.defects else "detected"
            if _wf_status == "correcting":
                _n_fixed = max(n_resolved, n_repaired)
                if _n_fixed > 0:
                    _badge = f"⚠ {n_active} aktiv · {_n_fixed} behoben"
                else:
                    _badge = f"⚠ {n_active} Defekte · wird bearbeitet"
                _badge_color = QColor(100, 200, 120, 230)
                _badge_bg = QColor(20, 50, 35, 180)
            elif _wf_status == "completed":
                _n_tools = len({e[2] for entries in self._repair_history.values() for e in entries if len(e) >= 3})
                _n_fixed = max(n_resolved, n_repaired)
                if _n_fixed and _n_tools:
                    _badge = f"✓ {_n_fixed} Defekte behoben · {_n_tools} Tools"
                elif _n_fixed:
                    _badge = f"✓ {_n_fixed} Defekte behoben"
                else:
                    _badge = f"⚠ {n_active} verblieben"
                _badge_color = QColor(130, 200, 154, 230)
                _badge_bg = QColor(20, 50, 35, 170)
            else:
                _suffix = "e" if n_active != 1 else ""
                _badge = f"⚠ {n_active} Defekt{_suffix} erkannt"
                _badge_color = QColor(255, 175, 70, 230)
                _badge_bg = QColor(50, 35, 10, 180)
            _badge_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
            painter.setFont(_badge_font)
            _bfm = painter.fontMetrics()
            _btw = _bfm.horizontalAdvance(_badge)
            _bx = int(x) + 6
            _by = int(y) + 5
            _bp = 5  # padding
            # Rounded pill background
            painter.setBrush(QBrush(_badge_bg))
            painter.setPen(QPen(QColor(_badge_color.red(), _badge_color.green(), _badge_color.blue(), 80), 0.8))
            painter.drawRoundedRect(_bx, _by, _btw + _bp * 2, _bfm.height() + _bp, 6, 6)
            # Text
            painter.setPen(_badge_color)
            painter.drawText(_bx + _bp, _by + _bp + _bfm.ascent(), _badge)

        # ── 4b. Active tool badge (right side, tool-colored pill) ──────────
        if self._active_tool and _wf_status_overlay == "correcting":
            _atc = self._TOOL_COLORS.get(self._active_tool, ((180, 180, 200), "⚙"))
            _at_r, _at_g, _at_b = _atc[0]
            _at_icon = _atc[1]
            _at_text = f"{_at_icon} {self._active_tool}"
            _at_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
            painter.setFont(_at_font)
            _at_fm = painter.fontMetrics()
            _at_tw = _at_fm.horizontalAdvance(_at_text)
            _at_x = int(x + width) - _at_tw - 18
            _at_y = int(y) + 5
            _at_p = 5
            # Pill background with tool color border
            painter.setBrush(QBrush(QColor(10, 12, 25, 200)))
            painter.setPen(QPen(QColor(_at_r, _at_g, _at_b, 140), 1.0))
            painter.drawRoundedRect(_at_x, _at_y, _at_tw + _at_p * 2, _at_fm.height() + _at_p, 6, 6)
            # Text in tool color
            painter.setPen(QColor(_at_r, _at_g, _at_b, 240))
            painter.drawText(_at_x + _at_p, _at_y + _at_p + _at_fm.ascent(), _at_text)

        # ── 5. Legend (defects + tools used) ─────────────────────────────────
        legend_h = 14
        swatch_s = 8
        swatch_gap = 4
        item_gap = 10
        legend_y = int(y + height) - legend_h - 1

        label_font = QFont("Segoe UI", 7)
        painter.setFont(label_font)
        _fm = painter.fontMetrics()

        items: list[tuple[str, str, QColor]] = []
        for k in active_keys:
            if k not in _DEFECT_LABELS and k not in _DEFECT_COLORS:
                continue
            items.append((k, _DEFECT_LABELS.get(k, k), _DEFECT_COLORS.get(k, QColor(180, 180, 180))))
        # Add tool-specific "behoben" legend items for each tool used in repairs
        _tools_used: dict[str, int] = {}
        for _rk, _r_entries in self._repair_history.items():
            for _entry in _r_entries:
                if isinstance(_entry, (list, tuple)) and len(_entry) >= 3:
                    _tn = str(_entry[2])
                    _tools_used[_tn] = _tools_used.get(_tn, 0) + 1
        if _tools_used:
            for _tn, _cnt in _tools_used.items():
                _tc = self._TOOL_COLORS.get(_tn, ((80, 200, 120), "✓"))
                _tr, _tg, _tb = _tc[0]
                items.append((_tn, f"✓ {_tn} ({_cnt})", QColor(_tr, _tg, _tb)))
        elif self._resolved_locations:
            items.append(("_resolved", "Behoben", QColor(80, 200, 120)))

        if not items:
            painter.restore()
            return

        item_widths = [swatch_s + swatch_gap + _fm.horizontalAdvance(lbl) for _, lbl, _ in items]
        total_legend_w = sum(item_widths) + item_gap * (len(items) - 1)
        # If legend is too wide, truncate items to fit
        _max_legend_w = int(width * 0.95)
        while total_legend_w > _max_legend_w and len(items) > 3:
            items.pop(-1)
            item_widths.pop(-1)
            total_legend_w = sum(item_widths) + item_gap * (len(items) - 1)
        cur_x = int(x + max(0, (width - total_legend_w) // 2))

        bg_padding = 4
        bg_rect_x = cur_x - bg_padding
        bg_rect_w = total_legend_w + bg_padding * 2
        bg_rect_y = legend_y - bg_padding
        bg_rect_h = legend_h + bg_padding * 2
        if bg_rect_h > 0 and bg_rect_w > 0:
            painter.setBrush(QBrush(QColor(10, 10, 20, 160)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h, 3, 3)

        for (k, lbl, color), iw in zip(items, item_widths):
            # For tool legend items: rounded swatch with slight glow
            _is_tool = k in self._TOOL_COLORS
            _swatch_a = 230 if _is_tool else 210
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), _swatch_a)))
            painter.setPen(Qt.PenStyle.NoPen)
            swatch_y = legend_y + (legend_h - swatch_s) // 2
            painter.drawRoundedRect(cur_x, swatch_y, swatch_s, swatch_s, 2, 2)
            # Subtle glow ring for tool swatches
            if _is_tool:
                painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 60), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(cur_x - 1, swatch_y - 1, swatch_s + 2, swatch_s + 2, 3, 3)
            painter.setPen(QColor(210, 215, 230, 220))
            painter.setFont(label_font)
            painter.drawText(
                cur_x + swatch_s + swatch_gap,
                legend_y,
                iw,
                legend_h,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                lbl,
            )
            cur_x += iw + item_gap

        # ── 6. Dual-row indicator bands (§11.4 enhanced) ────────────────────
        # Top row: active defect colors. Bottom row: tool colors for repairs.
        if active_keys or _tools_used:
            band_y = int(y + height) - BAND_H * 2
            # Row 1: Active defects (top)
            if active_keys:
                _dband_w = max(2, width // max(1, len(active_keys)))
                for bi, bk in enumerate(active_keys):
                    bcolor = _DEFECT_COLORS.get(bk, QColor(180, 180, 180))
                    painter.setBrush(QBrush(QColor(bcolor.red(), bcolor.green(), bcolor.blue(), 200)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(int(x) + bi * _dband_w, band_y, _dband_w, BAND_H)
            # Row 2: Tools used in repairs (bottom, tool-colored)
            _tool_list = list(_tools_used.keys())
            if _tool_list:
                _tband_w = max(2, width // max(1, len(_tool_list)))
                for ti, tn in enumerate(_tool_list):
                    _tc = self._TOOL_COLORS.get(tn, ((80, 200, 120), "✓"))
                    _tr, _tg, _tb = _tc[0]
                    painter.setBrush(QBrush(QColor(_tr, _tg, _tb, 200)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(int(x) + ti * _tband_w, band_y + BAND_H, _tband_w, BAND_H)

        painter.restore()

    def _draw_stage_overlay(self, painter, x, y, width, height):
        """Draw professional animated stage visualization (splash-screen quality).

        Visual design:
        - Category-specific animated overlays on the waveform showing
          what is actually happening to the audio signal
        - Translucent top info panel (30 px) with icon, label, mini-EQ bars,
          aurora gradient border, completed stage trail
        - Progress sweep with animated glow edge
        """
        if not self._active_stage or not self._stage_label:
            return

        painter.save()

        r, g, b = self._stage_color
        elapsed = time.monotonic() - self._stage_start_time if self._stage_start_time > 0 else 0.0
        progress = self._stage_progress

        # ── Resolve tool color: use tool-specific color if available ─────────
        _tr, _tg, _tb = r, g, b
        if self._active_tool and self._active_tool in self._TOOL_COLORS:
            _tr, _tg, _tb = self._TOOL_COLORS[self._active_tool][0]

        # ── Determine effect category ────────────────────────────────────────
        effect = "process"
        stage_lower = self._active_stage.lower()
        for kw, eff in self._STAGE_EFFECTS.items():
            if kw in stage_lower:
                effect = eff
                break

        # ── 1. Category-specific waveform animation (tool-colored) ──────────
        self._draw_stage_effect(painter, x, y, width, height, effect, _tr, _tg, _tb, elapsed, progress)

        # ── 2. Progress sweep (left → right, tool-colored) ──────────────────
        if progress > 0.01:
            sweep_w = int(width * progress)
            sweep_grad = QLinearGradient(x, y, x + sweep_w, y)
            sweep_grad.setColorAt(0.0, QColor(_tr, _tg, _tb, 6))
            sweep_grad.setColorAt(0.85, QColor(_tr, _tg, _tb, 14))
            sweep_grad.setColorAt(1.0, QColor(_tr, _tg, _tb, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(sweep_grad))
            painter.drawRect(int(x), int(y), sweep_w, int(height))
            # Animated glow at sweep edge
            edge_x = int(x) + sweep_w
            _pulse = 0.5 + 0.5 * math.sin(elapsed * 3.0)
            _edge_a = int(30 + 50 * _pulse)
            edge_grad = QLinearGradient(edge_x - 16, y, edge_x + 6, y)
            edge_grad.setColorAt(0.0, QColor(_tr, _tg, _tb, 0))
            edge_grad.setColorAt(0.55, QColor(_tr, _tg, _tb, _edge_a))
            edge_grad.setColorAt(1.0, QColor(_tr, _tg, _tb, 0))
            painter.setBrush(QBrush(edge_grad))
            painter.drawRect(max(int(x), edge_x - 16), int(y), 22, int(height))

        # ── 3. Top info panel (30 px) ────────────────────────────────────────
        panel_h = 30
        panel_y = int(y)
        # Dark background
        panel_grad = QLinearGradient(x, panel_y, x, panel_y + panel_h)
        panel_grad.setColorAt(0.0, QColor(7, 9, 22, 210))
        panel_grad.setColorAt(1.0, QColor(7, 9, 22, 150))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(panel_grad))
        painter.drawRect(int(x), panel_y, int(width), panel_h)

        # Aurora gradient bottom border (gold → tool-color → electric blue)
        border_y = panel_y + panel_h - 1
        _aur_a = int(80 + 60 * min(1.0, progress))
        aurora = QLinearGradient(x, 0, x + width, 0)
        aurora.setColorAt(0.0, QColor(212, 162, 50, _aur_a))
        aurora.setColorAt(0.4, QColor(_tr, _tg, _tb, min(255, _aur_a + 40)))
        aurora.setColorAt(1.0, QColor(18, 145, 255, _aur_a))
        painter.setPen(QPen(QBrush(aurora), 1.5))
        painter.drawLine(int(x), border_y, int(x + width), border_y)

        # Progress fill in border (2.5 px, tool-colored aurora gradient)
        if progress > 0.01:
            prog_w = int(width * progress)
            prog_g = QLinearGradient(x, 0, x + prog_w, 0)
            prog_g.setColorAt(0.0, QColor(212, 162, 50, 210))
            prog_g.setColorAt(0.5, QColor(_tr, _tg, _tb, 245))
            prog_g.setColorAt(1.0, QColor(18, 145, 255, 230))
            painter.setPen(QPen(QBrush(prog_g), 2.5))
            painter.drawLine(int(x), border_y, int(x) + prog_w, border_y)

        # ── 3a. Icon + Label ─────────────────────────────────────────────────
        text_y = panel_y + 3
        text_h = panel_h - 5

        painter.setPen(QColor(r, g, b, 235))
        painter.setFont(QFont("Segoe UI", 11))
        painter.drawText(int(x) + 8, text_y, 24, text_h, Qt.AlignmentFlag.AlignVCenter, self._stage_icon)

        _lbl_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(_lbl_font)
        _fm = painter.fontMetrics()
        _lbl_w = _fm.horizontalAdvance(self._stage_label)
        painter.setPen(QColor(r, g, b, 230))
        painter.drawText(
            int(x) + 34,
            text_y,
            _lbl_w + 10,
            text_h,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._stage_label,
        )

        # Progress %
        _pct_offset = 0
        if progress > 0.01:
            _pct_text = f"{int(progress * 100)} %"
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(r, g, b, 170))
            _pct_offset = 50
            painter.drawText(
                int(x) + 34 + _lbl_w + 12,
                text_y,
                _pct_offset,
                text_h,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                _pct_text,
            )

        # Active tool indicator (tool-colored label next to progress %)
        if self._active_tool:
            _atc = self._TOOL_COLORS.get(self._active_tool, ((180, 180, 200), "⚙"))
            _at_r, _at_g, _at_b = _atc[0]
            _at_icon = _atc[1]
            _tool_text = f"{_at_icon} {self._active_tool}"
            _tool_font = QFont("Segoe UI", 7)
            painter.setFont(_tool_font)
            _tfm = painter.fontMetrics()
            _ttw = _tfm.horizontalAdvance(_tool_text)
            _tool_x = int(x) + 34 + _lbl_w + 12 + _pct_offset + 6
            # Small rounded pill behind tool name
            _tp = 3
            painter.setBrush(QBrush(QColor(_at_r, _at_g, _at_b, 25)))
            painter.setPen(QPen(QColor(_at_r, _at_g, _at_b, 80), 0.7))
            painter.drawRoundedRect(_tool_x - _tp, text_y + 2, _ttw + _tp * 2, text_h - 4, 4, 4)
            painter.setPen(QColor(_at_r, _at_g, _at_b, 220))
            painter.drawText(
                _tool_x,
                text_y,
                _ttw + 10,
                text_h,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                _tool_text,
            )

        # ── 3b. Mini equalizer bars (5 bars, splash-style sine oscillators) ──
        n_eq = 5
        eq_bw, eq_gap = 3, 2
        eq_total = n_eq * eq_bw + (n_eq - 1) * eq_gap
        eq_x0 = int(x + width) - eq_total - 80
        eq_max_h = 14
        eq_base = panel_y + panel_h // 2 + eq_max_h // 2
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(n_eq):
            _ph = elapsed * 2.5 + i * 0.8
            _hf = 0.35 + 0.30 * math.sin(_ph) + 0.20 * math.sin(_ph * 1.7 + 1.1) + 0.15 * math.sin(_ph * 3.1 + 0.6)
            _hf = max(0.10, min(1.0, _hf))
            _bh = int(eq_max_h * _hf)
            _bx = eq_x0 + i * (eq_bw + eq_gap)
            _by = eq_base - _bh
            _bg = QLinearGradient(_bx, eq_base, _bx, eq_base - eq_max_h)
            _bg.setColorAt(0.0, QColor(_tr, _tg, _tb, 180))
            _bg.setColorAt(1.0, QColor(min(255, _tr + 60), min(255, _tg + 60), min(255, _tb + 60), 240))
            _bp = QPainterPath()
            _bp.addRoundedRect(QRectF(_bx, _by, eq_bw, _bh), 1, 1)
            painter.setBrush(QBrush(_bg))
            painter.drawPath(_bp)

        # ── 3c. Completed stage trail (right-aligned) ────────────────────────
        if self._stage_history:
            painter.setFont(QFont("Segoe UI", 7))
            _fm_t = painter.fontMetrics()
            _trail_x = int(x + width) - 8
            for i, _cl in enumerate(reversed(self._stage_history)):
                _a = max(50, 150 - i * 30)
                painter.setPen(QColor(80, 200, 120, _a))
                _tt = f"✓ {_cl}"
                _tw = _fm_t.horizontalAdvance(_tt) + 6
                _trail_x -= _tw
                if _trail_x < eq_x0 + eq_total + 10:
                    break
                painter.drawText(
                    _trail_x, text_y, _tw, text_h, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _tt
                )

        painter.restore()

    # ── Category-specific animated effects ────────────────────────────────

    def _draw_stage_effect(self, painter, x, y, width, height, effect, r, g, b, elapsed, progress):
        """Draw category-specific animated overlay visualizing audio processing.

        Each effect type renders a unique animation that communicates what
        is happening to the audio signal, not just generic progress.
        """
        painter.setPen(Qt.PenStyle.NoPen)

        if effect == "scan":
            # ── Sweeping scan line with trailing glow ────────────────────
            scan_phase = (elapsed * 0.4) % 1.0
            scan_x = int(x + scan_phase * width)
            # Trailing glow (wider → narrower)
            for _radius, _a in ((22, 8), (10, 18), (4, 35)):
                painter.setBrush(QBrush(QColor(r, g, b, _a)))
                painter.drawRect(max(int(x), scan_x - _radius), int(y), _radius * 2, int(height))
            # Bright center line
            painter.setPen(QPen(QColor(r, g, b, 130), 1.5))
            painter.drawLine(scan_x, int(y), scan_x, int(y + height))
            painter.setPen(Qt.PenStyle.NoPen)

        elif effect == "repair":
            # ── Spark/flash effects at pseudo-random positions ───────────
            n_sparks = 7
            for i in range(n_sparks):
                _ph = elapsed * 1.3 + i * 2.618
                _life = (_ph % 2.0) / 2.0
                if _life > 0.8:
                    continue
                _fade = 1.0 - _life / 0.8
                _sx = x + ((int(_ph * 137.508) % 997) / 997.0) * width
                _sy = y + ((int(_ph * 251.317) % 991) / 991.0) * height
                _alpha = int(160 * _fade)
                _sz = 3.0 + 5.0 * _fade
                # Outer glow
                painter.setBrush(QBrush(QColor(r, g, b, _alpha // 4)))
                painter.drawEllipse(QPointF(_sx, _sy), _sz * 2.5, _sz * 2.5)
                # Core
                painter.setBrush(QBrush(QColor(min(255, r + 80), min(255, g + 80), min(255, b + 80), _alpha)))
                painter.drawEllipse(QPointF(_sx, _sy), _sz, _sz)
                # White hot center
                painter.setBrush(QBrush(QColor(255, 255, 255, _alpha // 2)))
                painter.drawEllipse(QPointF(_sx, _sy), _sz * 0.35, _sz * 0.35)

        elif effect == "noise":
            # ── Animated noise floor line dropping with progress ─────────
            noise_level = 0.65 - progress * 0.45
            noise_y = int(y + height * (1.0 - noise_level))
            # Noise area (fading with progress)
            _na = max(3, int(14 - 11 * progress))
            painter.setBrush(QBrush(QColor(r, g, b, _na)))
            painter.drawRect(int(x), noise_y, int(width), int(y + height - noise_y))
            # Animated dashed noise floor
            n_dashes = 40
            _dw = width / n_dashes
            for i in range(n_dashes):
                _ph = elapsed * 2.0 + i * 0.3
                _jit = 3.5 * math.sin(_ph)
                _da = int(50 + 35 * math.sin(_ph * 0.7))
                painter.setPen(QPen(QColor(r, g, b, _da), 1.0))
                _dx = int(x + i * _dw)
                _dy = int(noise_y + _jit)
                painter.drawLine(_dx, _dy, int(_dx + _dw * 0.55), _dy)
            painter.setPen(Qt.PenStyle.NoPen)
            # Drifting noise particles (fade out with progress)
            if progress < 0.85:
                _np = int(10 * (1.0 - progress))
                for i in range(_np):
                    _ph = elapsed * 0.8 + i * 1.7
                    _px = x + ((int(_ph * 173.13) % 997) / 997.0) * width
                    _py = noise_y + (y + height - noise_y) * ((_ph % 3.0) / 3.0)
                    _drift = 2.5 * math.sin(_ph * 2.1)
                    _pa = int(35 * (1.0 - progress))
                    painter.setBrush(QBrush(QColor(r, g, b, _pa)))
                    painter.drawEllipse(QPointF(_px + _drift, _py), 1.5, 1.5)

        elif effect == "reconstruct":
            # ── Gap-filling bands growing from bottom ────────────────────
            n_bands = 7
            _bw = max(4, int(width / 55))
            for i in range(n_bands):
                _ph = elapsed * 0.6 + i * 0.9
                _life = (_ph % 3.0) / 3.0
                _bp = _life / 0.6 if _life < 0.6 else 1.0 - (_life - 0.6) / 0.4
                _bx = x + ((int(_ph * 197.37) % 991) / 991.0) * (width - _bw)
                _bh = int(height * 0.65 * _bp)
                _by = y + height - _bh
                _ba = int(45 * _bp)
                _bg = QLinearGradient(_bx, _by + _bh, _bx, _by)
                _bg.setColorAt(0.0, QColor(r, g, b, _ba // 2))
                _bg.setColorAt(1.0, QColor(r, g, b, _ba))
                painter.setBrush(QBrush(_bg))
                painter.drawRect(int(_bx), int(_by), _bw, _bh)

        elif effect == "spectral":
            # ── Rising spectral bars (mini-equalizer across waveform) ────
            n_bars = 18
            _bw = max(2, int(width / 75))
            _gap = int((width - n_bars * _bw) / max(1, n_bars))
            for i in range(n_bars):
                _ph = elapsed * 1.5 + i * 0.4
                _hf = 0.3 + 0.25 * math.sin(_ph) + 0.20 * math.sin(_ph * 1.7 + 1.0) + 0.10 * math.sin(_ph * 2.9)
                _hf = max(0.05, min(1.0, _hf))
                _bh = int(height * 0.45 * _hf)
                _bx = int(x + i * (_bw + _gap) + _gap / 2)
                _by = int(y + height - _bh)
                _ba = int(18 + 14 * _hf)
                _bg = QLinearGradient(_bx, _by + _bh, _bx, _by)
                _bg.setColorAt(0.0, QColor(r, g, b, _ba // 2))
                _bg.setColorAt(1.0, QColor(min(255, r + 40), min(255, g + 40), min(255, b + 40), _ba))
                painter.setBrush(QBrush(_bg))
                painter.drawRect(_bx, _by, _bw, _bh)

        elif effect == "vocal":
            # ── Animated vocal waveform with multi-harmonic modulation ────
            n_pts = 60
            _cy = y + height / 2.0
            _amp = height * 0.28
            painter.setPen(QPen(QColor(r, g, b, 55), 1.5))
            pts: list[QPointF] = []
            for i in range(n_pts + 1):
                _frac = i / n_pts
                _px = x + _frac * width
                _ph = elapsed * 2.0 + _frac * 12.0
                _wave = math.sin(_ph) * 0.5 + math.sin(_ph * 2.1 + 0.3) * 0.3 + math.sin(_ph * 0.4) * 0.2
                _env = math.sin(_frac * math.pi)
                _py = _cy + _wave * _amp * _env
                pts.append(QPointF(_px, _py))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            # Wider glow
            painter.setPen(QPen(QColor(r, g, b, 16), 5.0))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            painter.setPen(Qt.PenStyle.NoPen)

        elif effect == "timing":
            # ── Pitch-correction sinusoidal becoming straighter ──────────
            n_pts = 50
            _cy = y + height / 2.0
            _amp = height * 0.18
            _wobble = 1.0 - progress * 0.75  # reduces as correction succeeds
            painter.setPen(QPen(QColor(r, g, b, 50), 1.2))
            for i in range(n_pts):
                _frac = i / n_pts
                _px = x + _frac * width
                _ph = elapsed * 1.8 + _frac * 10.0
                _dev = math.sin(_ph) * _amp * _wobble
                _py = _cy + _dev
                _px2 = x + (i + 1) / n_pts * width
                _ph2 = elapsed * 1.8 + ((i + 1) / n_pts) * 10.0
                _dev2 = math.sin(_ph2) * _amp * _wobble
                painter.drawLine(QPointF(_px, _py), QPointF(_px2, _cy + _dev2))
            # Reference line (target = straight)
            painter.setPen(QPen(QColor(r, g, b, int(25 + 25 * progress)), 0.8, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, _cy), QPointF(x + width, _cy))
            painter.setPen(Qt.PenStyle.NoPen)

        elif effect == "dynamics":
            # ── Compression envelope visualization ───────────────────────
            n_pts = 40
            _cy = y + height * 0.5
            painter.setPen(QPen(QColor(r, g, b, 30), 1.0))
            for i in range(n_pts):
                _frac = i / n_pts
                _px = x + _frac * width
                _ph = elapsed * 1.5 + _frac * 8.0
                _env = 0.6 + 0.4 * math.sin(_ph) * math.sin(_ph * 0.3 + 1.2)
                _comp = _env * (0.8 + 0.2 * (1.0 - progress))
                _top = _cy - height * 0.35 * _env
                _c_top = _cy - height * 0.35 * _comp
                _c_bot = _cy + height * 0.35 * _comp
                if abs(_top - _c_top) > 1:
                    painter.setBrush(QBrush(QColor(r, g, b, 12)))
                    painter.drawRect(int(_px), int(_c_top), max(1, int(width / n_pts)), int(_c_bot - _c_top))
            painter.setPen(Qt.PenStyle.NoPen)

        elif effect == "quality":
            # ── Green quality sweep with sparkle ─────────────────────────
            if progress > 0.0:
                _sw = int(width * progress)
                painter.setBrush(QBrush(QColor(80, 200, 120, 8)))
                painter.drawRect(int(x), int(y), _sw, int(height))
                _ew = 10
                _ea = int(35 + 25 * math.sin(elapsed * 5.0))
                for _wo, _am in ((-_ew, 0.3), (-_ew // 2, 0.6), (0, 1.0)):
                    painter.setBrush(QBrush(QColor(80, 200, 120, int(_ea * _am))))
                    painter.drawRect(max(int(x), int(x) + _sw + _wo), int(y), _ew // 3, int(height))

        else:  # "process" — generic: orbiting dots
            _cx = x + width / 2.0
            _cy_pos = y + height / 2.0
            _orb = min(width, height) * 0.10
            for i in range(4):
                _angle = elapsed * 1.5 + i * (math.pi / 2.0)
                _dx = _cx + _orb * math.cos(_angle)
                _dy = _cy_pos + _orb * 0.4 * math.sin(_angle)
                _a = int(70 + 55 * math.sin(elapsed * 2.0 + i))
                _dr = 2.5 + 1.2 * math.sin(elapsed * 3.0 + i)
                # Glow
                painter.setBrush(QBrush(QColor(r, g, b, _a // 3)))
                painter.drawEllipse(QPointF(_dx, _dy), _dr * 2.5, _dr * 2.5)
                # Dot
                painter.setBrush(QBrush(QColor(r, g, b, _a)))
                painter.drawEllipse(QPointF(_dx, _dy), _dr, _dr)


class SpectrogramWidget(QWidget):
    """Premium Professional Spectrogram Visualization Widget

    Features:
    - High-resolution STFT with Hann windowing
    - Perceptual mel-frequency scaling
    - Professional dB scaling with optimal dynamic range
    - Inferno colormap for maximum detail visibility
    - Frequency axis labels (20Hz - 20kHz)
    - Time axis labels
    """

    def __init__(self):
        super().__init__()
        self.spectrogram_data = None
        self.frequencies = None
        self.times = None
        self.sample_rate = None
        self.setMinimumHeight(320)
        self.setStyleSheet("""
            background: rgba(20, 20, 30, 0.95);
            border: 2px solid rgba(102, 126, 234, 0.4);
            border-radius: 10px;
        """)

    def update_spectrogram(self, audio, sr):
        """Spektrogramm nicht-blockierend berechnen und anzeigen.

        Kann aus einem Hintergrundthread aufgerufen werden – das abschließende
        self.update() erfolgt über QTimer.singleShot im Haupt-Thread.
        """
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        self.sample_rate = sr

        # ── Audiodauer auf max. 60 s begrenzen (verhindert OOM + Minutenlang-Freeze) ──
        _max_samples = int(sr * 60)
        if len(audio) > _max_samples:
            # Repräsentativen Ausschnitt aus der Mitte nehmen
            _mid = len(audio) // 2
            audio = audio[max(0, _mid - _max_samples // 2) : _mid + _max_samples // 2]

        try:
            from scipy import signal

            # Kompakte STFT-Parameter: ausreichend für Visualisierung, CPU-schonend
            # nperseg=2048 → ~43 ms Fenster bei 48 kHz (guter Kompromiss)
            # noverlap=1024 → 50 % Overlap (statt 87,5 % → ~8× schneller)
            # nfft=2048 → kein Zero-Padding nötig für Anzeige
            nperseg = min(2048, len(audio) // 4)
            noverlap = nperseg // 2
            nfft = nperseg

            frequencies, times, Sxx = signal.spectrogram(
                audio,
                fs=sr,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                scaling="density",
                mode="magnitude",
            )

            Sxx_db = 10 * np.log10(Sxx + 1e-10)

            vmin = np.percentile(Sxx_db, 5)
            vmax = np.max(Sxx_db)
            Sxx_db_norm = np.clip((Sxx_db - vmin) / (vmax - vmin), 0.0, 1.0) if vmax > vmin else np.zeros_like(Sxx_db)

            # Auf audiblen Bereich beschränken (20 Hz – 20 kHz)
            freq_mask = (frequencies >= 20) & (frequencies <= 20000)
            self.frequencies = frequencies[freq_mask]
            self.times = times
            self.spectrogram_data = Sxx_db_norm[freq_mask, :]

            # Für die Anzeige auf max. 400 × 200 Bins reduzieren
            if self.spectrogram_data.shape[1] > 400:
                step = max(1, self.spectrogram_data.shape[1] // 400)
                self.spectrogram_data = self.spectrogram_data[:, ::step]
                self.times = self.times[::step]

            if self.spectrogram_data.shape[0] > 200:
                step = max(1, self.spectrogram_data.shape[0] // 200)
                self.spectrogram_data = self.spectrogram_data[::step, :]
                self.frequencies = self.frequencies[::step]

            # Hintergrundthread-sicher: update() nur im Haupt-Thread aufrufen
            QTimer.singleShot(0, self.update)

        except Exception as e:
            import logging as _log_sg

            _log_sg.getLogger(__name__).debug("Spectrogram-Berechnung fehlgeschlagen: %s", e)
            self.spectrogram_data = None

    def paintEvent(self, event):
        """Draw premium spectrogram with professional color mapping"""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.spectrogram_data is None:
            # Draw premium placeholder
            painter.setPen(QColor(180, 180, 200))
            font = QFont("Segoe UI", 11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "🎵 Spektrogramm wird berechnet...")
            return

        # Calculate drawing area (leave margins for axes)
        margin_left = 60
        margin_right = 20
        margin_top = 20
        margin_bottom = 40

        plot_x = margin_left
        plot_y = margin_top
        plot_width = self.width() - margin_left - margin_right
        plot_height = self.height() - margin_top - margin_bottom

        if plot_width <= 0 or plot_height <= 0:
            return

        # Spektrogramm-Daten als QImage rendern (Numpy-vektorisiert, ≫ Python-Schleife)
        n_freq, n_time = self.spectrogram_data.shape
        # Inferno-Lookup-Table (256 Einträge) – einmalig als uint8-Array erzeugen
        lut = self._get_inferno_lut()

        # Werte in 0-255 quantisieren und über LUT in RGBA wandeln
        idx = np.clip((self.spectrogram_data * 255).astype(np.uint8), 0, 255)
        # Frequenzachse umkehren (0 = unten im Plot)
        idx_flipped = np.flipud(idx)
        # RGBA-Array aufbauen (uint32 für QImage.Format_RGBX8888 / Format_RGB888)
        rgb = lut[idx_flipped]  # shape: (n_freq, n_time, 3)
        alpha = np.full((n_freq, n_time, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2)  # RGBA

        # QImage aus numpy-Array (keine Pixel-Schleife)
        img = QImage(
            rgba.tobytes(),
            n_time,
            n_freq,
            n_time * 4,
            QImage.Format.Format_RGBA8888,
        )
        # Auf Plot-Größe skalieren und zeichnen
        scaled = img.scaled(
            plot_width,
            plot_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(plot_x, plot_y, scaled)

        # Draw professional axes
        self._draw_axes(painter, plot_x, plot_y, plot_width, plot_height)

    @staticmethod
    def _get_inferno_lut() -> np.ndarray:
        """Gibt eine vorberechnete 256×3-uint8-Lookup-Table für den Inferno-Farbverlauf zurück.

        Wird beim ersten Aufruf berechnet und dann gecacht (Klassenattribut).
        Dark purple (0) → Red (0.5) → Yellow (0.8) → White (1)
        """
        if not hasattr(SpectrogramWidget, "_inferno_lut_cache"):
            lut = np.zeros((256, 3), dtype=np.uint8)
            for i in range(256):
                v = i / 255.0
                if v < 0.25:
                    t = v / 0.25
                    r, g, b = int(t * 60), int(t * 20), int(100 + t * 80)
                elif v < 0.5:
                    t = (v - 0.25) / 0.25
                    r, g, b = int(60 + t * 150), int(20 + t * 20), int(180 - t * 120)
                elif v < 0.75:
                    t = (v - 0.5) / 0.25
                    r, g, b = int(210 + t * 45), int(40 + t * 150), int(60 - t * 40)
                else:
                    t = (v - 0.75) / 0.25
                    r, g, b = 255, min(255, int(190 + t * 65)), min(255, int(20 + t * 235))
                lut[i] = [r, g, b]
            SpectrogramWidget._inferno_lut_cache = lut
        return SpectrogramWidget._inferno_lut_cache

    def _inferno_colormap(self, value: float) -> list[int]:  # type: ignore[override]
        """Einzelwert-Fallback (wird nur noch für _draw_axes-Hilfsfarben verwendet)."""
        value = np.clip(value, 0, 1)

        if value < 0.25:
            t = value / 0.25
            r = int(t * 60)
            g = int(t * 20)
            b = int(100 + t * 80)
        elif value < 0.5:
            t = (value - 0.25) / 0.25
            r = int(60 + t * 150)
            g = int(20 + t * 20)
            b = int(180 - t * 120)
        elif value < 0.75:
            t = (value - 0.5) / 0.25
            r = int(210 + t * 45)
            g = int(40 + t * 150)
            b = int(60 - t * 40)
        else:
            # Yellow to white
            t = (value - 0.75) / 0.25
            r = 255
            g = int(190 + t * 65)
            b = int(20 + t * 235)

        return QColor(r, g, b)

    def _draw_axes(self, painter, x, y, width, height):
        """Draw professional frequency and time axes"""
        painter.setPen(QColor(200, 200, 220, 200))
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        # Draw frequency axis (left side) - logarithmic spacing
        freq_labels = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for freq in freq_labels:
            if self.frequencies is not None and freq <= self.frequencies[-1]:
                # Find closest frequency bin
                idx = np.argmin(np.abs(self.frequencies - freq))
                if self.spectrogram_data is None:
                    continue
                y_pos = y + height - (idx / self.spectrogram_data.shape[0]) * height

                if y < y_pos < y + height:  # Check if within drawing area
                    painter.drawLine(int(x - 5), int(y_pos), int(x), int(y_pos))

                    # Format label
                    label = f"{freq // 1000}k" if freq >= 1000 else str(freq)

                    painter.drawText(
                        int(x - 55),
                        int(y_pos - 8),
                        50,
                        16,
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        label + " Hz",
                    )

        # Draw time axis (bottom)
        if self.times is not None:
            num_time_labels = 5
            for i in range(num_time_labels + 1):
                t_idx = int(i * len(self.times) / num_time_labels)
                if t_idx < len(self.times):
                    x_pos = x + (t_idx / len(self.times)) * width
                    time_val = self.times[t_idx]

                    painter.drawLine(int(x_pos), int(y + height), int(x_pos), int(y + height + 5))
                    painter.drawText(
                        int(x_pos - 30), int(y + height + 10), 60, 20, Qt.AlignmentFlag.AlignCenter, f"{time_val:.1f}s"
                    )


class ResourceStatusWidget(QWidget):
    """Verarbeitungs-Überprüfungsinstanz: zeigt aktiven DSP- oder KI-Analysemodus."""

    def __init__(self):
        super().__init__()
        self.quality_mode = "BALANCED"
        self.ml_mode_active = False
        self.active_ml_plugins: list[str] = []

        self.setMinimumHeight(70)
        self.setStyleSheet("""
            background: rgba(102, 126, 234, 0.10);
            border: 1px solid rgba(102, 126, 234, 0.22);
            border-radius: 8px;
        """)

        self._init_ui()

    def _init_ui(self):
        """Initialize UI layout — shows only the active analysis mode, not raw CPU/RAM."""
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(8, 6, 8, 6)

        # Single "Überprüfungsinstanz" label — replaces CPU/RAM indicators
        self.label_check_mode = QLabel(t("ui.resource_check", value=t("ui.resource_check_dsp")))
        self.label_check_mode.setStyleSheet("color: #B0C4DE; font-size: 8pt;")
        self.label_check_mode.setWordWrap(True)
        layout.addWidget(self.label_check_mode)

        # ML-Plugin-Status (separat, nur sichtbar wenn ML aktiv)
        self.label_ml_status = QLabel("")
        self.label_ml_status.setStyleSheet("color: #00FF7F; font-size: 8pt;")
        self.label_ml_status.setWordWrap(True)
        self.label_ml_status.setVisible(False)
        layout.addWidget(self.label_ml_status)

    def _update_resources(self):
        """No-op — CPU/RAM are no longer displayed."""

    def update_status(self, cpu=None, memory=None, mode=None, ml_active=None, ml_plugins=None, phase=None):
        """Update the active analysis-mode indicator (CPU/RAM parameters are ignored)."""
        if mode is not None:
            self.quality_mode = mode
        if ml_active is not None:
            self.ml_mode_active = ml_active
            if ml_active is False:
                self._phase_override: str | None = None
        if ml_plugins is not None:
            self.active_ml_plugins = ml_plugins
        if phase is not None:
            self._phase_override = phase

        _phase_override = getattr(self, "_phase_override", None)
        if _phase_override:
            # Show active restoration phase — overrides KI/DSP label
            self.label_check_mode.setText(_phase_override)
            self.label_check_mode.setStyleSheet("color: #E8C060; font-size: 8pt; font-weight: 600;")
            self.label_ml_status.setVisible(False)
        elif self.ml_mode_active and self.active_ml_plugins:
            # KI active: show number of active modules (no internal model names shown)
            n = len(self.active_ml_plugins)
            self.label_check_mode.setText(t("ui.resource_check", value=t("ui.resource_check_ml")))
            self.label_check_mode.setStyleSheet("color: #00FF7F; font-size: 8pt;")
            self.label_ml_status.setText(t("ui.resource_check_ml_detail", count=n, suffix=("e" if n != 1 else "")))
            self.label_ml_status.setVisible(True)
        else:
            self.label_check_mode.setText(t("ui.resource_check", value=t("ui.resource_check_dsp")))
            self.label_check_mode.setStyleSheet("color: #B0C4DE; font-size: 8pt;")
            self.label_ml_status.setVisible(False)


class DefectCounterWidget(QWidget):
    """Animated defect counter display with two-phase animation"""

    @staticmethod
    def _severity_word(val: float) -> str:
        """Return layman severity label for a float defect value."""
        if val >= 0.6:
            return "Kritisch"
        if val >= 0.3:
            return "Stark"
        if val >= 0.1:
            return "Mittel"
        return "Leicht"

    def __init__(self):
        super().__init__()
        self.defects = {
            "clicks": 0,
            "crackle": 0,
            "pops": 0,
            "clipping": 0,
            "hum": 0.0,
            "noise_level": 0.0,
            "sibilance": 0,
            "dropout": 0,
            "wow": 0.0,
            "flutter": 0.0,
        }
        self.target_defects = self.defects.copy()
        self.detected_values = self.defects.copy()  # Store detected values for phase 2
        self.phase = "detecting"  # 'detecting', 'correcting', 'completed'

        # Setup animation timer for "rattern" effect
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._animate_counters)

        self.setStyleSheet("""
            background: rgba(30, 30, 46, 0.5);
            border: 1px solid rgba(255, 165, 0, 0.3);
            border-radius: 8px;
            padding: 10px;
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI elements"""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QLabel(t("ui.defects_header"))
        header.setStyleSheet("color: #B8A068; font-size: 12pt; font-weight: bold;")
        layout.addWidget(header)

        # Counter labels
        _st = t("ui.defect_status_detect")
        self.label_clicks = QLabel(t("ui.defect_clicks", value=0, status=_st))
        self.label_crackle = QLabel(t("ui.defect_crackle", value=0, status=_st))
        self.label_pops = QLabel(t("ui.defect_pops", value=0, status=_st))
        self.label_clipping = QLabel(t("ui.defect_clipping", value=0, status=_st))
        self.label_hum = QLabel(t("ui.defect_hum", value=0.0, status=_st))
        self.label_noise = QLabel(t("ui.defect_noise", value=0.0, status=_st))
        self.label_sibilance = QLabel(t("ui.defect_sibilance", value=0, status=_st))
        self.label_dropout = QLabel(t("ui.defect_dropout", value=0, status=_st))
        self.label_wow = QLabel(t("ui.defect_wow", value=0.0, status=_st))
        self.label_flutter = QLabel(t("ui.defect_flutter", value=0.0, status=_st))

        for label in [
            self.label_clicks,
            self.label_crackle,
            self.label_pops,
            self.label_clipping,
            self.label_hum,
            self.label_noise,
            self.label_sibilance,
            self.label_dropout,
            self.label_wow,
            self.label_flutter,
        ]:
            label.setStyleSheet("color: #AAB8C6; font-family: 'Courier New'; font-size: 10pt;")
            layout.addWidget(label)

    def update_defects(self, defects):
        """Update defect counts with two-phase animation"""
        if "flutter" not in defects:
            defects = defects.copy()
            defects["flutter"] = 0.0

        status = defects.get("status", "detecting")

        if status == "detected":
            # Phase 1: Detection - ratter UP to detected values
            self.phase = "detecting"
            self.target_defects = defects.copy()
            self.detected_values = defects.copy()  # Store for phase 2
        elif status == "correcting":
            # Phase 2: Correction - ratter DOWN to zero
            self.phase = "correcting"
            # Set targets to zero for all counters
            self.target_defects = {
                "clicks": 0,
                "crackle": 0,
                "pops": 0,
                "clipping": 0,
                "hum": 0.0,
                "noise_level": 0.0,
                "sibilance": 0,
                "dropout": 0,
                "wow": 0.0,
                "flutter": 0.0,
            }
        elif status == "completed":
            # Phase 3: Completed - all at zero
            self.phase = "completed"
            self.target_defects = self.target_defects.copy()  # Keep zeros

        # Start animation
        if not self.anim_timer.isActive():
            self.anim_timer.start(50)  # Update every 50ms for smooth animation

    def _animate_counters(self):
        """Animate counter values (rattern effect) - UP during detection, DOWN during correction"""
        all_reached = True

        # Helper function to animate integer values (both up and down)
        def animate_int(current, target):
            if current < target:
                # Ratter UP
                return min(target, current + max(1, int((target - current) * 0.15)))
            elif current > target:
                # Ratter DOWN
                return max(target, current - max(1, int((current - target) * 0.15)))
            return current

        # Helper function to animate float values (both up and down)
        def animate_float(current, target, threshold=0.01):
            if abs(current - target) > threshold:
                diff = target - current
                return current + diff * 0.15
            return target

        # Animate all counters (supports both directions)
        new_clicks = animate_int(self.defects["clicks"], self.target_defects["clicks"])
        if new_clicks != self.defects["clicks"]:
            self.defects["clicks"] = new_clicks
            all_reached = False

        new_crackle = animate_int(self.defects["crackle"], self.target_defects["crackle"])
        if new_crackle != self.defects["crackle"]:
            self.defects["crackle"] = new_crackle
            all_reached = False

        new_pops = animate_int(self.defects["pops"], self.target_defects["pops"])
        if new_pops != self.defects["pops"]:
            self.defects["pops"] = new_pops
            all_reached = False

        new_clipping = animate_int(self.defects["clipping"], self.target_defects["clipping"])
        if new_clipping != self.defects["clipping"]:
            self.defects["clipping"] = new_clipping
            all_reached = False

        new_hum = animate_float(self.defects["hum"], self.target_defects["hum"])
        if abs(new_hum - self.defects["hum"]) > 0.01:
            self.defects["hum"] = new_hum
            all_reached = False

        new_noise = animate_float(self.defects["noise_level"], self.target_defects["noise_level"])
        if abs(new_noise - self.defects["noise_level"]) > 0.01:
            self.defects["noise_level"] = new_noise
            all_reached = False

        new_sibilance = animate_int(self.defects["sibilance"], self.target_defects["sibilance"])
        if new_sibilance != self.defects["sibilance"]:
            self.defects["sibilance"] = new_sibilance
            all_reached = False

        new_dropout = animate_int(self.defects["dropout"], self.target_defects["dropout"])
        if new_dropout != self.defects["dropout"]:
            self.defects["dropout"] = new_dropout
            all_reached = False

        new_wow = animate_float(self.defects["wow"], self.target_defects["wow"], 0.001)
        if abs(new_wow - self.defects["wow"]) > 0.001:
            self.defects["wow"] = new_wow
            all_reached = False

        new_flutter = animate_float(self.defects["flutter"], self.target_defects["flutter"], 0.001)
        if abs(new_flutter - self.defects["flutter"]) > 0.001:
            self.defects["flutter"] = new_flutter
            all_reached = False

        # Determine status icon and color based on phase
        if self.phase == "detecting":
            status_icon = t("ui.defect_status_detect")
            status_color = "#7B93B8"
        elif self.phase == "correcting":
            status_icon = t("ui.defect_status_correct")
            status_color = "#B8A068"
        elif self.phase == "completed":
            status_icon = t("ui.defect_status_done")
            status_color = "#82B89A"
        else:
            status_icon = t("ui.defect_status_detect")
            status_color = "#7B93B8"

        # Update labels
        self.label_clicks.setText(t("ui.defect_clicks", value=f"{self.defects['clicks']:,}", status=status_icon))
        self.label_crackle.setText(t("ui.defect_crackle", value=f"{self.defects['crackle']:,}", status=status_icon))
        self.label_pops.setText(t("ui.defect_pops", value=f"{self.defects['pops']:,}", status=status_icon))
        self.label_clipping.setText(t("ui.defect_clipping", value=f"{self.defects['clipping']:,}", status=status_icon))
        self.label_hum.setText(
            t("ui.defect_hum", severity=self._severity_word(self.defects["hum"]), status=status_icon)
        )
        self.label_noise.setText(
            t("ui.defect_noise", severity=self._severity_word(self.defects["noise_level"]), status=status_icon)
        )
        self.label_sibilance.setText(
            t("ui.defect_sibilance", value=f"{self.defects['sibilance']:,}", status=status_icon)
        )
        self.label_dropout.setText(t("ui.defect_dropout", value=f"{self.defects['dropout']:,}", status=status_icon))
        self.label_wow.setText(
            t("ui.defect_wow", severity=self._severity_word(self.defects["wow"]), status=status_icon)
        )
        self.label_flutter.setText(
            t("ui.defect_flutter", severity=self._severity_word(self.defects["flutter"]), status=status_icon)
        )

        # Update color based on phase
        for label in [
            self.label_clicks,
            self.label_crackle,
            self.label_pops,
            self.label_clipping,
            self.label_hum,
            self.label_noise,
            self.label_sibilance,
            self.label_dropout,
            self.label_wow,
            self.label_flutter,
        ]:
            if self.phase == "completed":
                label.setStyleSheet(
                    f"color: {status_color}; font-family: 'Courier New'; font-size: 10pt; font-weight: bold;"
                )
            else:
                label.setStyleSheet(f"color: {status_color}; font-family: 'Courier New'; font-size: 10pt;")

        # Stop animation when all values reached
        if all_reached:
            self.anim_timer.stop()


class ModernTitleBar(QWidget):
    """Custom Title Bar mit Drag-Support und Window Controls"""

    # Signals
    minimize_clicked = pyqtSignal()
    maximize_clicked = pyqtSignal()
    close_clicked = pyqtSignal()
    help_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.drag_position = None
        self.is_maximized = False

        # Setup UI
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Setup Title Bar UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 10, 0)
        layout.setSpacing(10)

        # Logo/Icon
        icon_label = QLabel("🎵")
        icon_label.setFont(QFont("Segoe UI", 20))
        layout.addWidget(icon_label)

        # App Title
        self.title_label = QLabel(t("ui.app_title"))
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.title_label)

        # Version Label
        self.version_label = QLabel("v9.10.57")
        self.version_label.setFont(QFont("Segoe UI", 8))
        self.version_label.setStyleSheet("color: rgba(255,255,255,0.38); padding: 0 2px;")
        layout.addWidget(self.version_label)

        # Stretch
        layout.addStretch()

        # Status Indicator (versteckt – Status wird in der unteren Leiste angezeigt)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)

        # Help / Shortcut-Übersicht
        self.btn_help = self._create_control_button("?", self.help_clicked)

        # Window Controls
        self.btn_minimize = self._create_control_button("−", self.minimize_clicked)
        self.btn_maximize = self._create_control_button("□", self.maximize_clicked)
        self.btn_close = self._create_control_button("×", self.close_clicked)

        layout.addWidget(self.btn_help)
        layout.addWidget(self.btn_minimize)
        layout.addWidget(self.btn_maximize)
        layout.addWidget(self.btn_close)

    def _create_control_button(self, text, signal):
        """Create window control button"""
        btn = QPushButton(text)
        btn.setFixedSize(40, 30)
        btn.setFont(QFont("Segoe UI", 14))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(signal)

        if text == "×":
            btn.setObjectName("closeButton")
        else:
            btn.setObjectName("controlButton")

        return btn

    def _apply_style(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            ModernTitleBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e,
                    stop:1 #16213e
                );
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            QPushButton#controlButton {
                background: transparent;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
            }
            QPushButton#controlButton:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            QPushButton#closeButton {
                background: transparent;
                color: #FFFFFF;
                border: none;
                border-radius: 5px;
            }
            QPushButton#closeButton:hover {
                background: #E74C3C;
                color: #FFFFFF;
            }
        """)

    def mousePressEvent(self, event):
        """Start window drag"""
        if event.button() == Qt.MouseButton.LeftButton:
            _w = self.window()
            if _w is None:
                return
            self.drag_position = event.globalPos() - _w.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle window drag"""
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position and not self.is_maximized:
            _w = self.window()
            if _w is None:
                return
            _w.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Toggle maximize on double-click"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.maximize_clicked.emit()
            event.accept()

    def set_status(self, text, color="#7B93B8"):
        """Update status indicator"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; padding: 5px 15px;")


class ModernButton(QPushButton):
    """Modern styled button with gradient and hover effects"""

    def __init__(self, text, icon=None, primary=False, parent=None):
        super().__init__(text, parent)
        self.primary = primary
        self.setMinimumHeight(45)
        self.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        self._apply_style()

    def _apply_style(self):
        """Apply modern gradient styling"""
        if self.primary:
            self.setStyleSheet("""
                ModernButton {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #667eea,
                        stop:1 #764ba2
                    );
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                }
                ModernButton:hover {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #7a8fff,
                        stop:1 #8a5cbd
                    );
                }
                ModernButton:pressed {
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:1,
                        stop:0 #5568d3,
                        stop:1 #653a8b
                    );
                }
            """)
        else:
            self.setStyleSheet("""
                ModernButton {
                    background: rgba(255, 255, 255, 0.05);
                    color: #FFFFFF;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 10px 20px;
                }
                ModernButton:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                ModernButton:pressed {
                    background: rgba(255, 255, 255, 0.15);
                }
            """)


class MagicImageButton(QPushButton):
    """Magic Button with image rendering, 3-D bevel and animated glow on hover.

    • Renders PNG full-size with object-fit:cover semantics (clipped, rounded).
    • 3-D bevel: top-left shimmer highlight + gradient rim (light top → dark bottom).
    • Hover glow: QGraphicsDropShadowEffect pulsed by a 60 fps QTimer.
    • Press: image shifts 2 px down-right, dark overlay, inverted rim.
    """

    def __init__(
        self,
        image_path: "Path | None" = None,
        hover_color: "tuple[int, int, int, int]" = (118, 75, 162, 191),
        pressed_color: "tuple[int, int, int, int]" = (80, 40, 120, 242),
        glow_color: "tuple[int, int, int]" = (118, 75, 162),
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Transparent background: prevents QGraphicsDropShadowEffect from filling
        # the bounding rect with palette background color (black corners artifact).
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self._pixmap: QPixmap | None = None
        if image_path is not None:
            px = QPixmap(str(image_path))
            self._pixmap = px if not px.isNull() else None
        self._hover_color = QColor(*hover_color)
        self._pressed_color = QColor(*pressed_color)
        self._glow_color_base = QColor(*glow_color)
        self._hovered = False
        self._btn_pressed = False
        self._glow_alpha = 0
        self._glow_dir = 1
        self.setAttribute(Qt.WA_Hover, True)
        self.setFlat(True)
        self.setStyleSheet("QPushButton { background: transparent; border: none; border-radius: 16px; }")
        # Glow drop-shadow, driven by _glow_timer
        self._glow_fx = QGraphicsDropShadowEffect(self)
        self._glow_fx.setBlurRadius(0)
        self._glow_fx.setOffset(0, 0)
        self._glow_fx.setColor(
            QColor(
                self._glow_color_base.red(),
                self._glow_color_base.green(),
                self._glow_color_base.blue(),
                0,
            )
        )
        self.setGraphicsEffect(self._glow_fx)

        self._glow_timer = QTimer(self)
        self._glow_timer.setInterval(16)  # ~60 fps
        self._glow_timer.timeout.connect(self._tick_glow)

    # ── Public helper ──────────────────────────────────────────────────────
    def reattach_glow(self) -> None:
        """Re-attach glow effect after an external setGraphicsEffect() cleared it."""
        self._glow_alpha = 0
        self._glow_fx.setBlurRadius(0)
        self._glow_fx.setColor(
            QColor(
                self._glow_color_base.red(),
                self._glow_color_base.green(),
                self._glow_color_base.blue(),
                0,
            )
        )
        self.setGraphicsEffect(self._glow_fx)

    # ── Hover / mouse events ───────────────────────────────────────────────
    def enterEvent(self, event):
        self._hovered = True
        self._glow_dir = 1
        self._glow_timer.start()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._btn_pressed = False
        self._glow_timer.stop()
        self._glow_alpha = 0
        self._apply_glow()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._btn_pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._btn_pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        """Keep image buttons visually circular to avoid square background artifacts."""
        super().resizeEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            self.clearMask()
            return

        diameter = max(0, min(self.width(), self.height()))
        if diameter <= 0:
            self.clearMask()
            return
        x = (self.width() - diameter) // 2
        y = (self.height() - diameter) // 2
        self.setMask(QRegion(x, y, diameter, diameter, QRegion.Ellipse))

    # ── Glow animation ─────────────────────────────────────────────────────
    def _tick_glow(self) -> None:
        self._glow_alpha = max(0, min(220, self._glow_alpha + 9 * self._glow_dir))
        if self._glow_alpha >= 220:
            self._glow_dir = -1
        elif self._glow_alpha <= 70 and self._glow_dir == -1:
            self._glow_dir = 1
        self._apply_glow()

    def _apply_glow(self) -> None:
        c = QColor(self._glow_color_base)
        c.setAlpha(self._glow_alpha)
        self._glow_fx.setColor(c)
        self._glow_fx.setBlurRadius(30 if self._glow_alpha > 0 else 0)

    # ── Paint ──────────────────────────────────────────────────────────────
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()
        diameter = max(0, min(rect.width(), rect.height()))
        draw_rect = QRect(
            rect.x() + (rect.width() - diameter) // 2,
            rect.y() + (rect.height() - diameter) // 2,
            diameter,
            diameter,
        )
        pressed = self._btn_pressed

        if not self._pixmap or self._pixmap.isNull():
            # No image: fall back to default QPushButton rendering (uses setStyleSheet)
            painter.end()
            super().paintEvent(event)
            return

        if draw_rect.width() <= 0 or draw_rect.height() <= 0:
            painter.end()
            return

        # ── 1. Image (object-fit:cover, circular clip) ─────────────────────
        scaled = self._pixmap.scaled(
            draw_rect.width(),
            draw_rect.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        ox = draw_rect.x() + (draw_rect.width() - scaled.width()) // 2
        oy = draw_rect.y() + (draw_rect.height() - scaled.height()) // 2
        painter.save()
        clip = QPainterPath()
        clip.addEllipse(QRectF(draw_rect))
        painter.setClipPath(clip)
        # Pressed: shift image 2 px down-right to simulate physical depth
        painter.drawPixmap(ox + (2 if pressed else 0), oy + (2 if pressed else 0), scaled)
        painter.restore()

        # ── 2. 3-D bevel overlay ───────────────────────────────────────────
        if not pressed:
            # Top shimmer highlight (top 28 % of height, white fade-out)
            tg = QLinearGradient(
                0.0,
                float(draw_rect.top()),
                0.0,
                float(draw_rect.top() + draw_rect.height() * 0.28),
            )
            tg.setColorAt(0.0, QColor(255, 255, 255, 62))
            tg.setColorAt(1.0, QColor(255, 255, 255, 0))
            tg_path = QPainterPath()
            tg_path.addEllipse(QRectF(draw_rect))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(tg))
            painter.drawPath(tg_path)

            # Rim border: light top → dark bottom  (raised-button illusion)
            rim = QLinearGradient(0.0, float(draw_rect.top()), 0.0, float(draw_rect.bottom()))
            rim.setColorAt(0.00, QColor(255, 255, 255, 115))
            rim.setColorAt(0.42, QColor(255, 255, 255, 20))
            rim.setColorAt(0.58, QColor(0, 0, 0, 20))
            rim.setColorAt(1.00, QColor(0, 0, 0, 135))
            painter.setPen(QPen(QBrush(rim), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(draw_rect.adjusted(1, 1, -1, -1))
        else:
            # Pressed: darken + inner top-shadow + inverted rim (sunken illusion)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 80)))
            painter.drawEllipse(draw_rect)

            ig = QLinearGradient(0.0, float(draw_rect.top()), 0.0, float(draw_rect.top() + 30))
            ig.setColorAt(0.0, QColor(0, 0, 0, 145))
            ig.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(ig))
            painter.drawEllipse(draw_rect)

            rim2 = QLinearGradient(0.0, float(draw_rect.top()), 0.0, float(draw_rect.bottom()))
            rim2.setColorAt(0.0, QColor(0, 0, 0, 115))
            rim2.setColorAt(1.0, QColor(255, 255, 255, 38))
            painter.setPen(QPen(QBrush(rim2), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(draw_rect.adjusted(1, 1, -1, -1))

        # ── 3. Disabled overlay ─────────────────────────────────────────────
        if not self.isEnabled():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 145)))
            painter.drawEllipse(draw_rect)

        painter.end()


class ModernCard(QFrame):
    """Modern card widget with glassmorphism effect"""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setObjectName("modernCard")

        # Shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)

        # Layout
        self._card_layout = QVBoxLayout(self)
        self._card_layout.setContentsMargins(20, 20, 20, 20)
        self._card_layout.setSpacing(15)

        # Title
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            title_label.setStyleSheet("color: #FFFFFF; margin-bottom: 10px;")
            self._card_layout.addWidget(title_label)

        self._apply_style()

    def _apply_style(self):
        """Apply glassmorphism style"""
        self.setStyleSheet("""
            QFrame#modernCard {
                background: rgba(30, 30, 46, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
            }
        """)

    def add_widget(self, widget):
        """Add widget to card"""
        self._card_layout.addWidget(widget)


class ModernProgressBar(QProgressBar):
    """Modern styled progress bar with gradient"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(30)

        self.setStyleSheet("""
            QProgressBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0   #0d1020,
                    stop:0.45 #080b18,
                    stop:0.55 #080b18,
                    stop:1   #0d1020);
                border: 1px solid rgba(102, 126, 234, 0.28);
                border-radius: 15px;
                text-align: center;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 10pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0.00 #1a8a3a,
                    stop:0.40 #22c55e,
                    stop:0.75 #39ff7a,
                    stop:1.00 #00ff66);
                border-radius: 13px;
                margin: 1px;
            }
        """)

    def setValue(self, value: int) -> None:
        """Setzt den Fortschrittswert und aktualisiert die Anzeige auf 0.1 % genau.

        Interner Bereich 0–10000 entspricht 0.00 %–100.00 % für flüssige Updates.
        Anzeige wird nur aktualisiert wenn Δ ≥ 10 (= 0.1 %), um Render-Last zu begrenzen.
        """
        prev = self.value()
        if abs(value - prev) < 10 and value != 0 and value != self.maximum():
            return  # Änderung < 0.1 % → kein Repaint
        super().setValue(value)
        mx = self.maximum()
        if mx > 0:
            pct = value * 100.0 / mx
            super().setFormat(f"{pct:.1f} %")
        else:
            super().setFormat("")


class GradientMainArea(QWidget):
    """Hauptbereich mit Wallpaper-Hintergrund (hintergrund.png).

    Skaliert das Hintergrundbild auf die aktuelle Widget-Größe und zeichnet
    es verlustfrei (AspectRatioMode.IgnoreAspectRatio für lückenlosen Fill).
    Fallback: dunkel-navy → pink Verlauf wenn die Datei nicht geladen werden kann.
    """

    _bg_pixmap: "QPixmap | None" = None  # Klassen-Cache: einmal laden, immer nutzen

    @classmethod
    def _load_bg(cls) -> "QPixmap | None":
        if cls._bg_pixmap is not None:
            return cls._bg_pixmap
        import os

        _here = os.path.dirname(os.path.abspath(__file__))
        _path = os.path.join(_here, "..", "resources", "hintergrund.png")
        _path = os.path.normpath(_path)
        if os.path.isfile(_path):
            pix = QPixmap(_path)
            if not pix.isNull():
                cls._bg_pixmap = pix
                return pix
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        pix = GradientMainArea._load_bg()

        if pix is not None:
            # Hintergrundbild skaliert auf Widget-Größe zeichnen
            painter.drawPixmap(self.rect(), pix, pix.rect())
        else:
            # Fallback: Verlauf dunkel-navy → Pink (wie bisher)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            gradient = QLinearGradient(0, 0, w * 0.85, h)
            gradient.setColorAt(0.00, QColor(10, 10, 30))
            gradient.setColorAt(0.40, QColor(50, 10, 70))
            gradient.setColorAt(0.72, QColor(130, 18, 110))
            gradient.setColorAt(1.00, QColor(200, 35, 155))
            painter.fillRect(self.rect(), gradient)

        super().paintEvent(event)


class ExportConfigDialog(QDialog):
    """Dialog für Exporteinstellungen: Speicherort, Dateiname, Format, Bitrate."""

    FORMATS: list[tuple[str, str, str]] = [
        ("FLAC 24-bit (verlustfrei, empfohlen)", "flac24", ".flac"),
        ("WAV 24-bit, 48 kHz (verlustfrei)", "wav24", ".wav"),
        ("WAV 16-bit, 44.1 kHz (CD-Qualität)", "wav16", ".wav"),
        ("AIFF 24-bit, 48 kHz (verlustfrei)", "aiff24", ".aiff"),
        ("MP3 CBR – 320 kbps (höchste MP3-Qualität)", "mp3_cbr_320", ".mp3"),
        ("MP3 CBR – 256 kbps", "mp3_cbr_256", ".mp3"),
        ("MP3 CBR – 192 kbps", "mp3_cbr_192", ".mp3"),
        ("MP3 VBR – V0 (~245 kbps, beste Qualität)", "mp3_vbr_v0", ".mp3"),
        ("MP3 VBR – V2 (~190 kbps)", "mp3_vbr_v2", ".mp3"),
        ("OGG Vorbis q9 (verlustbehaftet, offen)", "ogg9", ".ogg"),
    ]

    def __init__(self, source_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("ui.export_cfg_title"))
        self.setMinimumWidth(560)
        self.setModal(True)
        self._source_path = source_path
        self._build_ui(source_path)
        self.setStyleSheet("""
            QDialog { background: #1E1E2E; }
            QLabel  { color: #CCCCCC; font-size: 10pt; }
            QLineEdit {
                background: #2A2A3E; color: #FFFFFF;
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 6px; padding: 6px 10px; font-size: 10pt;
            }
            QLineEdit:focus { border-color: rgba(118,75,162,0.85); }

            /* ── ComboBox Hauptfeld ── */
            QComboBox {
                background: #2A2A3E;
                color: #FFFFFF;
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 6px;
                padding: 6px 32px 6px 10px;
                font-size: 10pt;
                min-height: 28px;
            }
            QComboBox:focus {
                border-color: rgba(118,75,162,0.85);
            }
            QComboBox:hover {
                border-color: rgba(180,140,220,0.55);
                background: #32324A;
            }
            /* Pfeil-Bereich */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid rgba(255,255,255,0.12);
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: rgba(118,75,162,0.30);
            }
            QComboBox::down-arrow {
                image: none;
                width: 0; height: 0;
                border-left:  5px solid transparent;
                border-right: 5px solid transparent;
                border-top:   6px solid #CCCCCC;
            }
            /* ── Popup-Liste ── */
            QComboBox QAbstractItemView {
                background: #22223A;
                color: #EEEEEE;
                border: 1px solid rgba(118,75,162,0.55);
                border-radius: 6px;
                padding: 4px 0;
                outline: none;
                font-size: 10pt;
            }
            QComboBox QAbstractItemView::item {
                background: transparent;
                color: #EEEEEE;
                padding: 7px 14px;
                min-height: 26px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(118,75,162,0.35);
                color: #FFFFFF;
            }
            QComboBox QAbstractItemView::item:selected {
                background: rgba(118,75,162,0.65);
                color: #FFFFFF;
            }

            QPushButton {
                background: rgba(118,75,162,0.75); color: white;
                border-radius: 8px; padding: 8px 16px;
                font-size: 10pt; font-weight: 600;
            }
            QPushButton:hover  { background: rgba(138,95,182,0.95); }
            QPushButton:pressed { background: rgba(90,50,140,0.95); }
        """)

    def _build_ui(self, source_path: str):
        src = Path(source_path)
        default_dir = str(src.parent)
        default_name = src.stem + "_restauriert"

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 22)

        # Titel
        title = QLabel(t("ui.export_cfg_pick_file"))
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #FFFFFF; margin-bottom: 4px;")
        layout.addWidget(title)

        sub = QLabel(t("ui.export_cfg_source", name=src.name))
        sub.setStyleSheet("color: #90A4AE; font-size: 9pt;")
        layout.addWidget(sub)

        # Trennlinie
        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setStyleSheet("color: rgba(255,255,255,0.12); margin-bottom: 4px;")
        layout.addWidget(sep0)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Speicherort
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(default_dir)
        self._dir_edit.setMinimumWidth(320)
        dir_btn = QPushButton("📂")
        dir_btn.setFixedWidth(38)
        dir_btn.setToolTip(t("ui.export_cfg_choose_folder"))
        dir_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(self._dir_edit)
        dir_row.addWidget(dir_btn)
        form.addRow(t("ui.export_cfg_storage"), dir_row)

        # Dateiname
        self._name_edit = QLineEdit(default_name)
        form.addRow(t("ui.export_cfg_filename"), self._name_edit)

        # Format
        self._fmt_combo = QComboBox()
        _fmt_tooltips = [
            "Beste Wahl für Archive und Weitergabe. Verlustfreie Kompression, "
            "24-Bit-Auflösung — volle Dynamik, kompaktes Format. Empfohlen für restaurierte Aufnahmen.",
            "Universell kompatibel, verlustfrei. Größere Dateien als FLAC, "
            "aber von JEDER Audiosoftware lesbar. Ideal für weitere Bearbeitung in DAWs.",
            "CD-Standard (44,1 kHz / 16 Bit). Geringere Dateigröße als WAV 24-Bit, "
            "kompatibel mit CD-Brennern. Etwas weniger Dynamikumfang als 24-Bit.",
            "Apple-Format, verlustfrei. Technisch gleichwertig zu WAV 24-Bit, "
            "bevorzugt in macOS/iOS-Workflows und Logic Pro.",
            "Höchste MP3-Qualität. Kompakt, universell kompatibel. "
            "Für Aufnahmen bis 1970er empfohlen wenn Speicher begrenzt — kaum hörbarer Qualitätsverlust.",
            "Guter MP3-Kompromiss. Für Podcasts, Demos oder Online-Sharing. "
            "Kleiner als 320 kbps, leichter Qualitätsverlust bei kritischem Hören.",
            "Kompakter MP3 für einfaches Streaming. Hörbarer Qualitätsverlust bei hochwertigen "
            "Restaurierungen — nur wenn Dateigröße entscheidend ist.",
            "Variable Bitrate, höchste MP3-Qualität (≈245 kbps Durchschnitt). "
            "Weniger Speicher als CBR 320 kbps bei vergleichbarer Qualität. Nicht alle Player unterstützen VBR.",
            "Variable Bitrate, guter Kompromiss (≈190 kbps). "
            "Für Alltagsgebrauch ausreichend. Deutlicher Qualitätsverlust im Hochtonbereich.",
            "Offenes Verlustformat, qualitativ besser als MP3 bei gleicher Bitrate. "
            "Ideal für Streaming-Plattformen und Open-Source-Workflows.",
        ]
        for (label, _, _), tooltip in zip(self.FORMATS, _fmt_tooltips):
            self._fmt_combo.addItem(label)
            self._fmt_combo.setItemData(
                self._fmt_combo.count() - 1,
                tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._fmt_combo.setToolTip(
            "Wählen Sie das Ausgabeformat. Für Archivierung: FLAC 24-Bit. "
            "Für Weitergabe: MP3 320 kbps oder höher. Fahren Sie über einen Eintrag für mehr Details."
        )
        form.addRow(t("ui.export_cfg_format"), self._fmt_combo)

        layout.addLayout(form)

        # Vorschau
        self._preview_lbl = QLabel()
        self._preview_lbl.setStyleSheet("color: #78909C; font-size: 9pt; padding: 6px 0 0 0;")
        self._preview_lbl.setWordWrap(True)
        layout.addWidget(self._preview_lbl)

        # Signale
        self._dir_edit.textChanged.connect(self._refresh)
        self._name_edit.textChanged.connect(self._refresh)
        self._fmt_combo.currentIndexChanged.connect(self._refresh)
        self._refresh()

        # Trennlinie
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.12); margin-top: 4px;")
        layout.addWidget(sep)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_cancel = QPushButton(t("ui.export_cfg_cancel"))
        self._btn_cancel.setFixedWidth(120)
        self._btn_cancel.setStyleSheet(
            "background: rgba(255,255,255,0.09); color: #AAAAAA;border-radius: 8px; padding: 8px 14px;"
        )
        self._btn_cancel.clicked.connect(self.reject)

        self._btn_ok = QPushButton(t("ui.export_cfg_continue"))
        self._btn_ok.setFixedWidth(220)
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(self._btn_cancel)
        btn_row.addSpacing(10)
        btn_row.addWidget(self._btn_ok)
        layout.addLayout(btn_row)

    def _browse_dir(self):
        start = self._dir_edit.text() or str(Path.home())
        d = ""

        # 1. Linux: zenity / kdialog
        if sys.platform.startswith("linux"):
            if shutil.which("zenity"):
                try:
                    proc = subprocess.run(
                        [
                            "zenity",
                            "--file-selection",
                            "--directory",
                            f"--title={t('ui.export_cfg_choose_storage')}",
                            f"--filename={start.rstrip('/')}/",
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        d = proc.stdout.strip()
                except Exception:
                    pass
            if not d and shutil.which("kdialog"):
                try:
                    proc = subprocess.run(
                        ["kdialog", "--getexistingdirectory", start, "--title", t("ui.export_cfg_choose_storage")],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout.strip():
                        d = proc.stdout.strip()
                except Exception:
                    pass

        # 2. Tkinter native (Win32 auf Windows, Tk-Fallback auf Linux)
        if not d:
            try:
                import tkinter as _tk
                from tkinter import filedialog as _fd

                _root = _tk.Tk()
                _root.withdraw()
                _root.wm_attributes("-topmost", True)
                try:
                    d = (
                        _fd.askdirectory(
                            title=t("ui.export_cfg_choose_storage"),
                            initialdir=start,
                            mustexist=False,
                            parent=_root,
                        )
                        or ""
                    )
                finally:
                    with contextlib.suppress(Exception):
                        _root.destroy()
            except Exception:
                pass

        # 3. Qt-Fallback
        if not d:
            dlg = QFileDialog(self, t("ui.export_cfg_choose_storage"), start)
            dlg.setFileMode(QFileDialog.FileMode.Directory)
            dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.setWindowFlag(Qt.WindowType.Dialog, True)
            dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            dlg.raise_()
            dlg.activateWindow()
            if dlg.exec_() == QDialog.DialogCode.Accepted:
                files = dlg.selectedFiles()
                if files:
                    d = files[0]

        if d:
            self._dir_edit.setText(d)

    def _refresh(self):
        idx = self._fmt_combo.currentIndex()
        ext = self.FORMATS[idx][2]
        name = (self._name_edit.text().strip() or "output") + ext
        full = str(Path(self._dir_edit.text().strip() or ".") / name)
        self._preview_lbl.setText(t("ui.export_cfg_output", path=full))

    def get_config(self) -> dict:
        """Gibt gewählte Einstellungen zurück."""
        idx = self._fmt_combo.currentIndex()
        _, fmt_key, ext = self.FORMATS[idx]
        name = (self._name_edit.text().strip() or "output") + ext
        out_dir = self._dir_edit.text().strip() or str(Path(self._source_path).parent)
        return {
            "output_dir": out_dir,
            "filename": name,
            "format_key": fmt_key,
            "output_path": str(Path(out_dir) / name),
        }


class ModernMainWindow(QMainWindow):
    """Modern Frameless Main Window mit Premium Design"""

    # Thread-sicheres Signal: Hintergrundthread → Fortschrittsbalken im GUI-Thread
    _load_progress = pyqtSignal(int)
    # Thread-sicherer Callable-Dispatch in den GUI-Thread
    _gui_dispatch = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        # Window flags for frameless
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowSystemMenuHint)

        # App icon (title bar, taskbar, alt-tab)
        _icon_path = Path(__file__).parent.parent / "resources" / "icon.png"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))

        # Linux compositors may show holes with translucent frameless windows.
        if sys.platform.startswith("linux"):
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Window properties
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            min_w = min(1280, max(960, available.width() - 40))
            min_h = min(900, max(640, available.height() - 40))
            self.setMinimumSize(min_w, min_h)
            self.resize(min(1500, available.width() - 20), min(1000, available.height() - 20))
        else:
            self.setMinimumSize(1280, 900)
            self.resize(1500, 1000)

        # State
        self.is_maximized = False
        self.old_position = None

        # Processing state
        self.current_file = None
        self.current_file_path = None
        self.processing_thread = None
        self.batch_thread = None
        self.batch_queue = SimpleBatchQueue()

        # A/B-Player Audio-State
        self._orig_audio: np.ndarray | None = None
        self._orig_sr: int = 48000
        self._rest_audio: np.ndarray | None = None
        self._rest_sr: int = 48000
        self._play_thread: threading.Thread | None = None
        # Playhead timing (updated by _playhead_timer)
        self._playback_start_time: float = 0.0
        self._playback_audio_duration: float = 0.0

        # Drag & Drop aktivieren
        self.setAcceptDrops(True)

        # Setup UI
        self._setup_ui()
        self._apply_theme()
        self._apply_i18n_texts()

        # Center window
        self._center_window()

        # Fade-in animation
        self._animate_fade_in()
        self._setup_shortcuts()
        # §9.7.4: Modell-Warmup 2 s nach App-Start im Daemon-Thread
        QTimer.singleShot(
            2000,
            lambda: threading.Thread(target=_warmup_models_background, daemon=True, name="AurikWarmup").start(),
        )
        # §11.4 Bridge-Fallback: Sichtbare Fehlermeldung wenn Backend nicht geladen
        if not _BRIDGE_AVAILABLE:
            QTimer.singleShot(300, self._show_bridge_unavailable_warning)

    def _show_bridge_unavailable_warning(self) -> None:
        """Zeigt eine deutliche Fehlermeldung wenn das Aurik-Backend nicht geladen werden konnte.

        Wird via QTimer 300 ms nach Fensterstart aufgerufen — sichert, dass das Fenster
        vollständig gerendert ist bevor der Dialog erscheint (§11.4 Bridge-Fallback).
        """
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Aurik — Startfehler")
        msg.setText("<b>⚠ Aurik konnte nicht vollständig starten.</b>")
        msg.setInformativeText(
            "Die Restaurierungs-Funktion ist nicht verfügbar.\n\n"
            "Was du jetzt tun kannst:\n"
            "  • Starte Aurik neu.\n"
            "  • Falls das Problem bleibt: Installiere Aurik erneut.\n"
            "  • Wende dich an den Support falls das Problem weiterhin auftritt.\n\n"
            "Die Klang-Analyse und Restaurierung sind bis zum Neustart deaktiviert."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        # Statusanzeige auch im Hauptfenster setzen
        if hasattr(self, "detected_medium_label"):
            self.detected_medium_label.setText(
                "⚠ Aurik konnte nicht vollständig starten.\n→ Starte die Anwendung neu. Bitte erneut installieren falls das Problem anhält."
            )
            self.detected_medium_label.setStyleSheet(
                "color: #B87A7A; font-size: 8pt; padding: 5px 8px;"
                "background: rgba(148, 82, 82, 0.10);"
                "border-radius: 8px; border: 1px solid rgba(152, 88, 88, 0.26);"
            )

    def _toggle_lyrics_overlay(self) -> None:
        """L-Shortcut: Lyrics-Timeline-Overlay ein-/ausblenden (§11.4 / §2.36 Spec 08).

        Transkribiert das geladene Audio per LyricsGuidedEnhancement (Whisper-Tiny ONNX
        oder DSP-Fallback) und zeichnet farbige Phonem-Bänder über den WaveformWidget-Canvas.

        Datenschutz-Pflicht (§2.36): Lyrics-Text wird NIEMALS geloggt oder angezeigt —
        nur Phonem-Typen (vowel_stressed, fricative_stressed, plosive, silence, …).
        """
        ov = getattr(self, "_lyrics_overlay_visible", False)
        self._lyrics_overlay_visible = not ov

        if not self._lyrics_overlay_visible:
            # Overlay ausblenden: WaveformWidget zurücksetzen
            if hasattr(self, "waveform_widget") and hasattr(self.waveform_widget, "_lyrics_transcription"):
                self.waveform_widget._lyrics_transcription = None
                self.waveform_widget.update()
            if hasattr(self, "status_text"):
                self.status_text.setText(t("status.lyrics_overlay_hidden"))
            return

        # Overlay einblenden: Transkription im Hintergrund starten
        if self._orig_audio is None:
            if hasattr(self, "status_text"):
                self.status_text.setText(t("status.lyrics_load_file_first"))
            self._lyrics_overlay_visible = False
            return

        if hasattr(self, "status_text"):
            self.status_text.setText(t("status.lyrics_transcribing"))

        _audio_ref = self._orig_audio
        _sr_ref = self._orig_sr

        def _transcribe_bg(_a=_audio_ref, _s=_sr_ref, _self=self) -> None:
            try:
                _lge = _bridge_get_lyrics_guided_enhancement()
                if _lge is None:
                    raise ImportError("LyricsGuidedEnhancement nicht verfügbar")
                # Mono für Transkription
                _mono = np.mean(_a, axis=1).astype(np.float32) if _a.ndim > 1 else _a.astype(np.float32)
                transcription = _lge._transcriber.transcribe(_mono, _s)  # interne Transkription

                def _apply():
                    if hasattr(_self, "waveform_widget"):
                        _self.waveform_widget._lyrics_transcription = transcription
                        _self.waveform_widget.update()
                    if hasattr(_self, "status_text"):
                        n = len(transcription.words) if not transcription.fallback_used else 0
                        src = "Signalanalyse" if transcription.fallback_used else "KI-Spracherkennung"
                        _self.status_text.setText(f"🎵 Lautanalyse eingeblendet  ·  {n} Abschnitte  ·  via {src}")

                _self._dispatch_to_gui(_apply)
            except Exception as _exc:
                logger.warning("LyricsGuided-Overlay: Transkription fehlgeschlagen: %s", _exc)

                def _err():
                    if hasattr(_self, "status_text"):
                        _self.status_text.setText(t("status.lyrics_unavailable"))
                    _self._lyrics_overlay_visible = False

                _self._dispatch_to_gui(_err)

        threading.Thread(target=_transcribe_bg, daemon=True, name="LyricsOverlay").start()

    def _setup_ui(self):
        """Layout: schmales linkes Panel + breiter Hauptbereich mit Gradient-Hintergrund.

        Struktur:
            QMainWindow
            └── main_container (QWidget, transparent)
                ├── title_bar  (ModernTitleBar)
                └── body (QHBoxLayout, kein Abstand)
                    ├── left_panel  (220 px, dunkel-navy)
                    └── main_area   (GradientMainArea, füllt Rest)
        """
        self.main_container = QWidget()
        self.main_container.setObjectName("mainContainer")
        self.main_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCentralWidget(self.main_container)

        outer = QVBoxLayout(self.main_container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Titelleiste ───────────────────────────────────────────────
        self.title_bar = ModernTitleBar(self)
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.maximize_clicked.connect(self._toggle_maximize)
        self.title_bar.close_clicked.connect(self.close)
        self.title_bar.help_clicked.connect(self._show_shortcut_help)
        outer.addWidget(self.title_bar)

        # ── Körper: linkes Panel ◀ | ▶ Hauptbereich ──────────────────
        body = QWidget()
        body.setObjectName("bodyWidget")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        body_layout.addWidget(self._create_left_panel())  # feste 220 px
        body_layout.addWidget(self._create_main_area(), 1)  # füllt Rest

        outer.addWidget(body, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # LINKES PANEL
    # ─────────────────────────────────────────────────────────────────────────
    def _create_left_panel(self) -> QWidget:
        """Schmales linkes Panel (220 px):
        [Audio-Datei öffnen]
        Erkannter Tonträger: ▸ detected_medium_label   (Carrier-Name + Konfidenz)
        Restaurierbarkeit:   ▸ restorability_banner     (Score 0–100 + MOS-Erwartung)
        erkannte Defekte:  ▸ defect_summary_label
        Musikalische Ziele: ▸ radar_widget + quality_score_label
        """
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setObjectName("leftPanel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        panel.setStyleSheet("""
            QWidget#leftPanel {
                background: #080a18;
                border-right: 1px solid rgba(102, 126, 234, 0.18);
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(10)

        # ── "Audio-Datei öffnen" Button ───────────────────────────────
        self.btn_import = QPushButton(t("action.open_file"))
        self.btn_import.setMinimumHeight(42)
        self.btn_import.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import.clicked.connect(self._open_file)
        self.btn_import.setFont(QFont("Segoe UI", 10))
        self.btn_import.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(200, 200, 220, 0.60);
                border-radius: 10px;
                color: #E8EAF6;
                font-size: 10pt;
                padding: 8px 14px;
            }
            QPushButton:hover  { background: rgba(255,255,255,0.07); }
            QPushButton:pressed { background: rgba(255,255,255,0.13); }
        """)
        layout.addWidget(self.btn_import)

        # ── Interne Hilfs-Funktion: Sektion mit Titel ─────────────────
        def _section(title: str, content: QWidget) -> QWidget:
            w = QWidget()
            vl = QVBoxLayout(w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(3)
            lbl = QLabel(title)
            lbl.setStyleSheet("color: #7080A0; font-size: 8pt; background: transparent;")
            vl.addWidget(lbl)
            vl.addWidget(content)
            return w

        # ── Aufnahme ─────────────────────────────────────────────────
        self.detected_medium_label = QLabel(t("ui.no_file_loaded"))
        self.detected_medium_label.setWordWrap(True)
        self.detected_medium_label.setTextFormat(Qt.TextFormat.RichText)
        self.detected_medium_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.detected_medium_label.setStyleSheet("""
            color: #B0C4DE; font-size: 8pt; padding: 5px 8px;
            background: rgba(102, 126, 234, 0.10);
            border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.22);
        """)
        self.detected_medium_label.setToolTip("<b>Erkanntes Quellformat</b><br>Vinyl, Kassette, CD, MP3 …")
        layout.addWidget(_section("Erkannter Tonträger:", self.detected_medium_label))

        # ── Restaurierbarkeit (separat vom Tonträger, §11.4 Spec 08) ────────
        self.restorability_banner = QLabel("—")
        self.restorability_banner.setWordWrap(True)
        self.restorability_banner.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.restorability_banner.setStyleSheet("""
            color: #B0C4DE; font-size: 8pt; padding: 5px 8px;
            background: rgba(102, 126, 234, 0.10);
            border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.22);
        """)
        layout.addWidget(_section("Restaurierbarkeit:", self.restorability_banner), 1)

        self.mode_recommendation_label = QLabel("—")
        self.mode_recommendation_label.setWordWrap(True)
        self.mode_recommendation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.mode_recommendation_label.setStyleSheet("""
            color: #B0C4DE; font-size: 8pt; padding: 5px 8px;
            background: rgba(102, 126, 234, 0.10);
            border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.22);
        """)
        self.mode_recommendation_label.setVisible(False)
        layout.addWidget(_section("Aurik-Empfehlung:", self.mode_recommendation_label), 1)

        # ── Erkannte Defekte + Echtzeit-Zähler ──────────────────────────────
        # Echtzeit-Zähler-Label: wird während des Scans live aktualisiert
        self.defect_count_live_label = QLabel("")
        self.defect_count_live_label.setWordWrap(False)
        self.defect_count_live_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.defect_count_live_label.setStyleSheet(
            "color: #90A4AE; font-size: 8pt; background: transparent; padding: 0 2px;"
        )
        self.defect_count_live_label.setVisible(False)

        # Header-Zeile: "erkannte Defekte" links, live-Zähler rechts
        _defect_header = QWidget()
        _dh_row = QHBoxLayout(_defect_header)
        _dh_row.setContentsMargins(0, 0, 0, 0)
        _dh_row.setSpacing(4)
        _dh_title_lbl = QLabel(t("ui.defects_detected_title"))
        _dh_title_lbl.setStyleSheet("color: #7080A0; font-size: 8pt; background: transparent;")
        _dh_row.addWidget(_dh_title_lbl)
        _dh_row.addStretch()
        _dh_row.addWidget(self.defect_count_live_label)

        self.defect_summary_label = QLabel(t("ui.no_analysis"))
        self.defect_summary_label.setWordWrap(True)
        self.defect_summary_label.setMinimumHeight(0)
        self.defect_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.defect_summary_label.setStyleSheet("""
            color: #B0C4DE; font-size: 8pt; padding: 5px 8px;
            background: rgba(102, 126, 234, 0.10);
            border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.22);
        """)

        self.defect_summary_scroll = QScrollArea()
        self.defect_summary_scroll.setWidgetResizable(True)
        self.defect_summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.defect_summary_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.defect_summary_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.defect_summary_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                width: 8px;
                background: rgba(20, 24, 36, 0.55);
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(120, 140, 180, 0.55);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.defect_summary_scroll.setWidget(self.defect_summary_label)
        _defect_sb = self.defect_summary_scroll.verticalScrollBar()
        if _defect_sb is not None:
            _defect_sb.rangeChanged.connect(
                lambda _min, _max: self.defect_summary_scroll.setVerticalScrollBarPolicy(
                    Qt.ScrollBarPolicy.ScrollBarAsNeeded if _max > 0 else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                )
            )

        _defect_container = QWidget()
        _dc_vbox = QVBoxLayout(_defect_container)
        _dc_vbox.setContentsMargins(0, 0, 0, 0)
        _dc_vbox.setSpacing(2)
        _dc_vbox.addWidget(_defect_header)
        _dc_vbox.addWidget(self.defect_summary_scroll)
        layout.addWidget(_defect_container, 7)

        # Interne Widgets (verborgen, nur für Datenverarbeitung)
        self.defect_counter_widget = DefectCounterWidget()
        self.defect_counter_widget.setVisible(False)
        self.resource_status_widget = ResourceStatusWidget()
        # ResourceStatusWidget kept as data source; no longer shown as panel

        # ── Musikalische Ziele ────────────────────────────────────────
        quality_frame = QFrame()
        quality_frame.setStyleSheet("""
            QFrame {
                background: rgba(16, 20, 38, 0.92);
                border: 1px solid rgba(102, 126, 234, 0.25);
                border-radius: 10px;
            }
        """)
        qi = QVBoxLayout(quality_frame)
        qi.setContentsMargins(6, 6, 6, 6)
        qi.setSpacing(4)

        # Quality Meter VU bar — always visible, above radar/placeholder
        self.quality_meter_widget = QualityMeterWidget()
        qi.addWidget(self.quality_meter_widget)

        if MusicalGoalsRadarWidget is not None:
            self.radar_widget = MusicalGoalsRadarWidget()
            self.radar_widget.setMinimumHeight(200)
            self.radar_widget.setMaximumHeight(280)
            qi.addWidget(self.radar_widget)
        else:
            self.radar_widget = None

        self.quality_score_label = QLabel("—")
        self.quality_score_label.setWordWrap(True)
        self.quality_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quality_score_label.setStyleSheet("color: #8894A8; font-size: 8pt; padding: 4px; background: transparent;")
        qi.addWidget(self.quality_score_label)

        self.info_banner = QLabel("")
        self.info_banner.setWordWrap(True)
        self.info_banner.setVisible(False)
        self.info_banner.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info_banner.setStyleSheet("color: #B0BEC5; font-size: 8pt; padding: 8px; background: transparent;")
        self.info_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        qi.addWidget(self.info_banner)

        layout.addWidget(_section("Musikalische Ziele:", quality_frame), 2)
        return panel

    # ─────────────────────────────────────────────────────────────────────────
    # HAUPTBEREICH (Gradient-Hintergrund)
    # ─────────────────────────────────────────────────────────────────────────
    def _create_main_area(self) -> QWidget:
        """Hauptbereich: Verlaufs-Hintergrund + Tabs (Wellenform/Spektrogramm)
        + A/B-Player + Magic Buttons + Status-Leiste.
        """
        # GradientMainArea malt Hintergrund + Kreise; Layout liegt drüber
        area = GradientMainArea()
        area.setObjectName("mainArea")
        area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(area)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(12)

        # ── Tab-Widget: Wellenform | Spektrogramm ────────────────────
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane {
                background: rgba(14, 18, 36, 0.88);
                border: 1px solid rgba(102, 126, 234, 0.28);
                border-radius: 12px;
                border-top-left-radius: 0px;
            }
            QTabBar::tab {
                background: rgba(22, 28, 50, 0.80);
                color: #8898BB;
                padding: 7px 20px;
                margin-right: 3px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 10pt;
            }
            QTabBar::tab:selected {
                background: rgba(102, 126, 234, 0.30);
                color: #FFFFFF;
                border-bottom: 2px solid #667eea;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(102, 126, 234, 0.14);
                color: #D0DCFF;
            }
        """)

        waveform_tab = QWidget()
        wf_layout = QVBoxLayout(waveform_tab)
        wf_layout.setContentsMargins(8, 8, 8, 8)
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.setMinimumHeight(400)
        wf_layout.addWidget(self.waveform_widget)
        self.viz_tabs.addTab(waveform_tab, t("ui.tab_waveform"))

        spectrogram_tab = QWidget()
        sp_layout = QVBoxLayout(spectrogram_tab)
        sp_layout.setContentsMargins(8, 8, 8, 8)
        self.spectrogram_widget = SpectrogramWidget()
        self.spectrogram_widget.setMinimumHeight(400)
        sp_layout.addWidget(self.spectrogram_widget)
        self.viz_tabs.addTab(spectrogram_tab, t("ui.tab_spectrogram"))

        # ── A/B-Vergleichs-Tab ─────────────────────────────────────────────
        _ab_tab = QWidget()
        _ab_tab_layout = QVBoxLayout(_ab_tab)
        _ab_tab_layout.setContentsMargins(8, 4, 8, 8)
        _ab_tab_layout.setSpacing(2)
        _ab_splitter = QWidget()
        _ab_split_layout = QHBoxLayout(_ab_splitter)
        _ab_split_layout.setContentsMargins(0, 0, 0, 0)
        _ab_split_layout.setSpacing(6)
        # Linke Spalte: Original
        _orig_col = QWidget()
        _orig_col_layout = QVBoxLayout(_orig_col)
        _orig_col_layout.setContentsMargins(0, 0, 0, 0)
        _orig_col_layout.setSpacing(4)
        _lbl_orig = QLabel("▶  Original")
        _lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _lbl_orig.setStyleSheet("color:rgba(100,150,255,0.9);font-weight:bold;font-size:9pt;padding:2px 0;")
        self.waveform_widget_orig_ab = WaveformWidget()
        self.waveform_widget_orig_ab.setMinimumHeight(370)
        _orig_col_layout.addWidget(_lbl_orig)
        _orig_col_layout.addWidget(self.waveform_widget_orig_ab, 1)
        # Rechte Spalte: Restauriert
        _rest_col = QWidget()
        _rest_col_layout = QVBoxLayout(_rest_col)
        _rest_col_layout.setContentsMargins(0, 0, 0, 0)
        _rest_col_layout.setSpacing(4)
        _lbl_rest = QLabel("✦  Restauriert")
        _lbl_rest.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _lbl_rest.setStyleSheet("color:rgba(80,220,100,0.9);font-weight:bold;font-size:9pt;padding:2px 0;")
        self.waveform_widget_rest_ab = WaveformWidget()
        self.waveform_widget_rest_ab.setMinimumHeight(370)
        _rest_col_layout.addWidget(_lbl_rest)
        _rest_col_layout.addWidget(self.waveform_widget_rest_ab, 1)
        _ab_split_layout.addWidget(_orig_col, 1)
        _ab_split_layout.addWidget(_rest_col, 1)
        _ab_tab_layout.addWidget(_ab_splitter, 1)
        self.viz_tabs.addTab(_ab_tab, "↔  A/B")

        layout.addWidget(self.viz_tabs, 3)

        # ── A/B Vor-/Nachher-Player ───────────────────────────────────
        ab_card = QFrame()
        ab_card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 36, 0.75);
                border: 1px solid rgba(102, 126, 234, 0.22);
                border-radius: 10px;
            }
        """)
        ab_inner = QVBoxLayout(ab_card)
        ab_inner.setContentsMargins(12, 4, 12, 4)
        ab_inner.setSpacing(3)

        self.ab_hdr = QLabel(t("ui.ab_compare"))
        self.ab_hdr.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.ab_hdr.setFixedHeight(16)
        self.ab_hdr.setStyleSheet("color: #B8CCEE; background: transparent;")
        ab_inner.addWidget(self.ab_hdr)

        ab_row = QWidget()
        # Vertical layout — more room for the larger Magic Buttons below
        ab_row_layout = QVBoxLayout(ab_row)
        ab_row_layout.setSpacing(5)
        ab_row_layout.setContentsMargins(0, 2, 0, 2)

        _ab_style_orig = (
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #1E88E5,stop:1 #1565C0);border:none;border-radius:7px;"
            "color:white;font-size:9pt;font-weight:bold;padding:7px 10px;}"
            "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #42A5F5,stop:1 #1976D2);}"
            "QPushButton:disabled{background:rgba(80,80,80,0.4);color:#666;}"
        )
        _ab_style_rest = (
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #3a6655,stop:1 #2a4f42);border:none;border-radius:7px;"
            "color:#d8eee4;font-size:9pt;font-weight:bold;padding:7px 10px;}"
            "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #4a8270,stop:1 #38625a);}"
            "QPushButton:disabled{background:rgba(80,80,80,0.4);color:#666;}"
        )
        _ab_style_stop = (
            "QPushButton{background:rgba(244,67,54,0.65);border:none;border-radius:7px;"
            "color:white;font-size:9pt;font-weight:bold;padding:7px 10px;}"
            "QPushButton:hover{background:rgba(244,67,54,0.90);}"
            "QPushButton:disabled{background:rgba(80,80,80,0.30);color:#555;}"
        )

        self.btn_play_original = ModernButton(f"▶  {t('action.listen_original')}")
        self.btn_play_original.setEnabled(False)
        self.btn_play_original.setFixedHeight(38)
        self.btn_play_original.setStyleSheet(_ab_style_orig)
        self.btn_play_original.clicked.connect(
            lambda: self._orig_audio is not None and self._play_audio(self._orig_audio, self._orig_sr)
        )
        ab_row_layout.addWidget(self.btn_play_original)

        self.btn_play_restored = ModernButton(f"▶  {t('action.listen_restored')}")
        self.btn_play_restored.setEnabled(False)
        self.btn_play_restored.setFixedHeight(38)
        self.btn_play_restored.setStyleSheet(_ab_style_rest)
        self.btn_play_restored.clicked.connect(
            lambda: self._rest_audio is not None and self._play_audio(self._rest_audio, self._rest_sr)
        )
        ab_row_layout.addWidget(self.btn_play_restored)

        self.btn_stop_playback = ModernButton(f"⏹  {t('action.stop')}")
        self.btn_stop_playback.setFixedHeight(34)
        self.btn_stop_playback.setStyleSheet(_ab_style_stop)
        self.btn_stop_playback.setEnabled(False)  # nur aktiv während Wiedergabe
        self.btn_stop_playback.clicked.connect(self._stop_playback)
        ab_row_layout.addWidget(self.btn_stop_playback)

        # Playback-Zeitanzeige (Elapsed / Total)
        self._playback_time_label = QLabel("– : – –")
        self._playback_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self._playback_time_label.setStyleSheet(
            "color: rgba(180, 200, 255, 0.70); font-size: 9pt; font-family: 'Courier New', monospace; padding: 0 6px;"
        )
        self._playback_time_label.setMinimumWidth(80)
        ab_row_layout.addWidget(self._playback_time_label)

        ab_inner.addWidget(ab_row)
        layout.addWidget(ab_card)

        # ── Magic Buttons ─────────────────────────────────────────────
        layout.addWidget(self._create_magic_buttons_section())

        # ── Audio-Info-Chip-Leiste ─────────────────────────────────────
        self._audio_info_bar = self._create_audio_info_bar()
        layout.addWidget(self._audio_info_bar)

        # ── Status-Leiste ─────────────────────────────────────────────
        layout.addWidget(self._create_status_bar())

        return area

    def _set_magic_buttons_enabled(self, enabled: bool) -> None:
        """Aktiviert/deaktiviert Magic Buttons inkl. optischer Verblassung."""
        for _name in ("btn_magic_restoration", "btn_magic_studio"):
            _btn = getattr(self, _name, None)
            if _btn is None:
                continue
            _btn.setEnabled(enabled)
            if enabled:
                # MagicImageButton handles disabled state via its own paintEvent;
                # just re-attach the glow effect that may have been cleared.
                if isinstance(_btn, MagicImageButton):
                    _btn.reattach_glow()
                else:
                    _btn.setGraphicsEffect(None)
            else:
                # MagicImageButton darkens itself in paintEvent — no opacity effect needed.
                if not isinstance(_btn, MagicImageButton):
                    from PyQt5.QtWidgets import QGraphicsOpacityEffect

                    _eff = QGraphicsOpacityEffect(_btn)
                    _eff.setOpacity(0.30)
                    _btn.setGraphicsEffect(_eff)

    def _recommend_mode_from_ui_context(self) -> tuple[str, str, str]:
        """Leitet eine konservative UI-Empfehlung aus Analysemetadaten ab."""
        medium = str(getattr(self, "_raw_medium_type", "unknown") or "unknown").strip().lower()
        restorability = float(getattr(self, "_restorability_score", 50.0) or 50.0)
        badge = str(getattr(self, "_era_genre_badge", "") or "").lower()

        historical_materials = {
            "wax_cylinder",
            "shellac",
            "lacquer_disc",
            "wire_recording",
            "vinyl",
            "tape",
            "reel_tape",
            "cassette",
        }
        modern_digital = {"cd_digital", "dat", "aac", "mp3_high", "streaming"}
        is_historical = medium in historical_materials or any(
            token in badge for token in ("190", "191", "192", "193", "194", "195", "196")
        )
        is_schlager = "schlager" in badge
        is_modern_digital = medium in modern_digital

        if is_historical or restorability < 65.0:
            return (
                "RESTORATION",
                f"💿 Empfehlung: RESTORATION — sicherer für {medium or 'dieses Material'} bei {restorability:.0f}/100 Restaurierbarkeit.",
                "color:#F6EFD9; font-size:8pt; font-weight:bold; padding:5px 8px;"
                "background: rgba(255, 193, 7, 0.10); border-radius:8px;"
                "border: 1px solid rgba(255, 193, 7, 0.28);",
            )
        if is_schlager and not is_modern_digital:
            return (
                "RESTORATION",
                f"💿 Empfehlung: RESTORATION — Schlager bleibt bei {medium or 'dieser Quelle'} musikalisch konservativ.",
                "color:#F6EFD9; font-size:8pt; font-weight:bold; padding:5px 8px;"
                "background: rgba(255, 193, 7, 0.10); border-radius:8px;"
                "border: 1px solid rgba(255, 193, 7, 0.28);",
            )
        if is_modern_digital and restorability >= 70.0:
            return (
                "STUDIO_2026",
                f"🎯 Empfehlung: STUDIO 2026 — stabiles Digitalmaterial mit {restorability:.0f}/100 Restaurierbarkeit.",
                "color:#DDEFE1; font-size:8pt; font-weight:bold; padding:5px 8px;"
                "background: rgba(76, 175, 80, 0.10); border-radius:8px;"
                "border: 1px solid rgba(76, 175, 80, 0.28);",
            )
        return (
            "RESTORATION",
            f"💿 Empfehlung: RESTORATION — konservativer Standard bei {restorability:.0f}/100 Restaurierbarkeit.",
            "color:#B0C4DE; font-size:8pt; font-weight:bold; padding:5px 8px;"
            "background: rgba(102, 126, 234, 0.10); border-radius:8px;"
            "border: 1px solid rgba(102, 126, 234, 0.22);",
        )

    def _apply_mode_recommendation_visuals(self) -> None:
        """Aktualisiert Banner und Tooltips, ohne die Moduswahl automatisch zu ändern."""
        if not hasattr(self, "mode_recommendation_label"):
            return

        recommended_mode, text, css = self._recommend_mode_from_ui_context()
        self._recommended_mode = recommended_mode
        self.mode_recommendation_label.setText(text)
        self.mode_recommendation_label.setStyleSheet(css)
        self.mode_recommendation_label.setVisible(bool(getattr(self, "current_file_path", None)))

        if hasattr(self, "btn_magic_restoration"):
            restoration_tip = (
                "<b>Originalgetreue Restauration</b><br>"
                "Erhält den historischen Klang, entfernt Artefakte ohne Klangveränderung."
            )
            if recommended_mode == "RESTORATION":
                restoration_tip += "<br><b>Empfohlen für diese Quelle.</b>"
            self.btn_magic_restoration.setToolTip(restoration_tip)

        if hasattr(self, "btn_magic_studio"):
            studio_tip = (
                "<b>Highend-Studio-Klang 2026</b><br>"
                "Modern, frisch, klar, kräftig — auf heutigen Referenzstandard gebracht."
            )
            if recommended_mode == "STUDIO_2026":
                studio_tip += "<br><b>Empfohlen für diese Quelle.</b>"
            self.btn_magic_studio.setToolTip(studio_tip)

    def _process_with_mode(self, mode):
        """Process current file with selected mode"""
        # Check if file is loaded
        if not hasattr(self, "current_file_path") or not self.current_file_path:
            QMessageBox.warning(self, t("dialog.no_file_title"), t("dialog.no_file_body"))
            return

        # Store selected mode
        self.selected_mode = mode

        try:
            # Now add file to queue with the selected mode
            self._add_to_queue_with_mode(self.current_file_path, mode)

            _recommended = getattr(self, "_recommended_mode", "")
            if _recommended and _recommended != mode and hasattr(self, "status_text"):
                self.status_text.setText(
                    f"Aurik-Empfehlung war {_recommended} — deine Auswahl {mode} wird trotzdem respektiert."
                )

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Buttons deaktivieren bis Verarbeitung abgeschlossen
            self._set_magic_buttons_enabled(False)

            # Start processing
            self._start_processing()
        except Exception as _exc:
            logger.exception("Fehler beim Starten der Verarbeitung")
            # Buttons wieder freigeben, damit der Nutzer es erneut versuchen kann
            self._set_magic_buttons_enabled(True)
            QMessageBox.critical(
                self,
                t("dialog.processing_error_title"),
                t("dialog.processing_error_body", error=_exc),
            )

    def _create_magic_buttons_section(self) -> QWidget:
        """Erstellt die zwei vollflächigen Magic-Image-Buttons (Restoration / Studio 2026).

        Die Bilder sind nahezu quadratisch (669×698 / 666×694, Verhältnis ≈ 1:1.04).
        Ein AspectRatioButton-Wrapper sorgt dafür, dass die Buttons immer mit korrektem
        Seitenverhältnis gerendert werden — unabhängig von der Fensterbreite.
        """
        _res_dir = Path(__file__).parent.parent / "resources"
        _img_r = _res_dir / "restoration.png"
        _img_s = _res_dir / "studio.png"
        # Aspect ratios: 1.0 = square (height == width)
        _ratio_r = 1.0
        _ratio_s = 1.0

        # ────────────────────────────────────────────────────────────────────
        # AspectRatioContainer: Wrapper-Widget das die Button-Höhe dynamisch
        # anpasst, sodass das Bild-Seitenverhältnis (Höhe/Breite) erhalten bleibt.
        # ────────────────────────────────────────────────────────────────────
        class _AspectContainer(QWidget):
            def __init__(self, btn: QPushButton, ratio: float, parent=None):
                super().__init__(parent)
                self._ratio = ratio
                self._btn = btn
                self.setStyleSheet("background: transparent;")
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
                self.setAutoFillBackground(False)
                inner = QVBoxLayout(self)
                inner.setContentsMargins(0, 0, 0, 0)
                inner.addWidget(btn)
                from PyQt5.QtWidgets import QSizePolicy as _QSP

                sp = _QSP(_QSP.Policy.Expanding, _QSP.Policy.Preferred)
                sp.setHeightForWidth(True)
                self.setSizePolicy(sp)

            def hasHeightForWidth(self) -> bool:
                return True

            def heightForWidth(self, w: int) -> int:
                return max(80, int(w * self._ratio))

            def resizeEvent(self, event) -> None:
                super().resizeEvent(event)
                target_h = self.heightForWidth(self.width())
                if self.height() != target_h:
                    self.setFixedHeight(target_h)

        # Äußerer Container mit HBox (zentriert, max. Breite begrenzt)
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        container.setAutoFillBackground(False)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(12)
        row.addStretch(1)

        # ── Restoration Button ──────────────────────────────────────────────
        self.btn_magic_restoration = MagicImageButton(
            image_path=_img_r if _img_r.exists() else None,
            hover_color=(118, 75, 162, 191),
            pressed_color=(80, 40, 120, 242),
            glow_color=(118, 75, 162),  # violet glow
        )
        self.btn_magic_restoration.setMinimumSize(140, 140)
        self.btn_magic_restoration.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_magic_restoration.setToolTip(
            "<b>Originalgetreue Restauration</b><br>"
            "Erhält den historischen Klang, entfernt Artefakte ohne Klangveränderung."
        )
        self.btn_magic_restoration.clicked.connect(lambda: self._process_with_mode("RESTORATION"))
        if not _img_r.exists():
            self.btn_magic_restoration.setText(f"💿  {t('action.restore_restoration')}")
            self.btn_magic_restoration.setStyleSheet(
                "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 #6a11cb,stop:1 #2575fc); color: white; border-radius: 16px;"
                "font-size: 13pt; font-weight: bold; }"
            )
        _ac_r = _AspectContainer(self.btn_magic_restoration, _ratio_r)
        _ac_r.setMaximumWidth(360)
        row.addWidget(_ac_r)

        # ── Studio 2026 Button ─────────────────────────────────────────────
        self.btn_magic_studio = MagicImageButton(
            image_path=_img_s if _img_s.exists() else None,
            hover_color=(255, 165, 0, 191),
            pressed_color=(180, 110, 0, 242),
            glow_color=(255, 165, 0),  # golden glow
        )
        self.btn_magic_studio.setMinimumSize(140, 140)
        self.btn_magic_studio.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_magic_studio.setToolTip(
            "<b>Highend-Studio-Klang 2026</b><br>"
            "Modern, frisch, klar, kräftig — auf heutigen Referenzstandard gebracht."
        )
        self.btn_magic_studio.clicked.connect(lambda: self._process_with_mode("STUDIO_2026"))
        if not _img_s.exists():
            self.btn_magic_studio.setText(f"🎯  {t('action.restore_studio')}")
            self.btn_magic_studio.setStyleSheet(
                "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 #f7971e,stop:1 #ffd200); color: #1a1a2e; border-radius: 16px;"
                "font-size: 13pt; font-weight: bold; }"
            )
        _ac_s = _AspectContainer(self.btn_magic_studio, _ratio_s)
        _ac_s.setMaximumWidth(360)
        row.addWidget(_ac_s)
        row.addStretch(1)

        # Initial deaktiviert — werden nach Defektanalyse aktiviert
        self._set_magic_buttons_enabled(False)
        self._apply_mode_recommendation_visuals()

        return container

    def _create_audio_info_bar(self) -> QWidget:
        """Compact horizontal chip strip showing detected audio properties.

        Chips appear one by one as background analysis completes:
        medium/carrier  |  era decade  |  genre  |  restorability score.
        Hidden until first chip is populated; reset on new file load.
        """
        bar = QWidget()
        bar.setObjectName("audioInfoBar")
        bar.setStyleSheet("background: transparent;")
        bar.setVisible(False)

        _chip_css = (
            "QLabel {{ "
            "color: {fg}; background: {bg}; border: 1px solid {border};"
            " border-radius: 10px; padding: 3px 10px;"
            " font-size: 9pt; font-weight: 600; "
            "}}"
        )

        def _make_chip(fg: str, bg: str, border: str) -> QLabel:
            lbl = QLabel()
            lbl.setVisible(False)
            lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            lbl.setStyleSheet(_chip_css.format(fg=fg, bg=bg, border=border))
            return lbl

        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(6, 4, 6, 4)
        hbox.setSpacing(8)

        # Chip: Tonträger (blaugrau)
        self.chip_medium = _make_chip(
            "#A8C0D8",
            "rgba(100, 140, 200, 0.12)",
            "rgba(100, 140, 200, 0.30)",
        )
        hbox.addWidget(self.chip_medium)

        # Chip: Ära / Dekade (violett)
        self.chip_era = _make_chip(
            "#BBA8D8",
            "rgba(140, 100, 200, 0.12)",
            "rgba(140, 100, 200, 0.30)",
        )
        hbox.addWidget(self.chip_era)

        # Chip: Genre (orange-gold)
        self.chip_genre = _make_chip(
            "#D8B870",
            "rgba(200, 160, 60, 0.12)",
            "rgba(200, 160, 60, 0.30)",
        )
        hbox.addWidget(self.chip_genre)

        # Chip: Restaurierbarkeit (dynamische Farbe, wird beim Setzen geändert)
        self.chip_restorability = _make_chip(
            "#82B89A",
            "rgba(85, 155, 115, 0.12)",
            "rgba(85, 155, 115, 0.30)",
        )
        hbox.addWidget(self.chip_restorability)

        hbox.addStretch()
        return bar

    # ── helper: show / update a single chip ───────────────────────────────
    def _show_chip(self, chip: QLabel, text: str, *, fg: str | None = None) -> None:
        """Set chip text, optionally change colour, make bar + chip visible."""
        if fg:
            current_css = chip.styleSheet()
            # Replace colour in existing CSS
            import re as _re_chip

            current_css = _re_chip.sub(r"color:\s*[^;]+;", f"color: {fg};", current_css, count=1)
            chip.setStyleSheet(current_css)
        chip.setText(text)
        chip.setVisible(True)
        # Make the bar visible when first chip appears
        if hasattr(self, "_audio_info_bar") and not self._audio_info_bar.isVisible():
            self._audio_info_bar.setVisible(True)

    def _create_status_bar(self):
        """Statusbereich: voller Fortschrittsbalken (oben) + transparente Textzeile (unten)."""
        wrapper = QWidget()
        wrapper.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(wrapper)
        vbox.setContentsMargins(0, 2, 0, 2)
        vbox.setSpacing(4)

        # ── Fortschrittsbalken – volle Breite, ausgeblendet bis Lade-/Verarbeitungsstart ──
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setFixedHeight(28)
        self.progress_bar.setRange(0, 10000)  # 1 Einheit = 0.01 % (0.1 %-Schritte angezeis
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        vbox.addWidget(self.progress_bar)
        # Thin sub-progress bar: current step within overall processing (5 px, no text)
        self.phase_progress_bar = QProgressBar()
        self.phase_progress_bar.setRange(0, 10000)
        self.phase_progress_bar.setFixedHeight(5)
        self.phase_progress_bar.setVisible(False)
        self.phase_progress_bar.setValue(0)
        self.phase_progress_bar.setTextVisible(False)
        self.phase_progress_bar.setStyleSheet(
            "QProgressBar { background: rgba(45,55,80,0.45); border-radius: 2px; border: none; }"
            " QProgressBar::chunk { background: qlineargradient("
            "x1:0,y1:0,x2:1,y2:0, stop:0 #7080D8, stop:1 #B878D0); border-radius: 2px; }"
        )
        vbox.addWidget(self.phase_progress_bar)
        # Compact stage counter label: "Stufe X / 12" — shown between sub-bar and status row
        self._phase_step_label = QLabel("")
        self._phase_step_label.setStyleSheet(
            "color: #7A90B0; font-size: 8pt; background: transparent; padding: 0px 2px;"
        )
        self._phase_step_label.setVisible(False)
        vbox.addWidget(self._phase_step_label)
        # Signal verbinden: _load_progress emittiert 0-100, Bar intern 0-10000
        self._load_progress.connect(lambda v: self.progress_bar.setValue(v * 100))
        # Callable-Dispatch-Signal verbinden
        self._gui_dispatch.connect(lambda fn: fn())

        # ── Statuszeile: Status-Text | Stretch | Queue-Stats ─────────
        status_row = QWidget()
        status_row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(status_row)
        row_layout.setContentsMargins(4, 0, 4, 0)
        row_layout.setSpacing(0)

        self.status_text = QLabel(t("status.ready"))
        self.status_text.setStyleSheet(
            "color: #8AABCC; font-size: 11pt; font-weight: 600; background: transparent; padding: 2px 0px;"
        )
        self.status_text.setToolTip(
            "<b>Aktueller Systemstatus</b><br>"
            "Zeigt an, was Aurik 9 gerade tut — z.\u202fB. Datei laden, "
            "Defekte analysieren, Restaurierung durchführen oder Ergebnis speichern.<br>"
            "<small>→ Farbe ändert sich: Blau = bereit, Orange = läuft, "
            "Grün = fertig.</small>"
        )
        row_layout.addWidget(self.status_text)
        row_layout.addStretch()

        self.stats_label = QLabel(t("status.stats", pending=0, completed=0, failed=0))
        self.stats_label.setStyleSheet("color: #AAB8C6; font-size: 9pt; background: transparent;")
        row_layout.addWidget(self.stats_label)

        vbox.addWidget(status_row)
        return wrapper

    def _apply_theme(self):
        """Apply dark premium theme for two-column layout."""
        main_bg = "#080a18" if sys.platform.startswith("linux") else "transparent"
        self.setStyleSheet(
            "QMainWindow {"
            f"background: {main_bg};"
            "}"
            "QWidget#mainContainer {"
            "background: #080a18;"
            "border-radius: 15px;"
            "}"
            "QWidget#bodyWidget {"
            "background: transparent;"
            "}"
            "QWidget#mainArea {"
            "background: transparent;"
            "}"
            "QLabel {"
            "color: #FFFFFF;"
            "}"
            "QSplitter::handle {"
            "background: rgba(255, 255, 255, 0.1);"
            "width: 2px;"
            "}"
        )

    def _center_window(self):
        """Center window on screen"""
        _screen = QApplication.primaryScreen()
        if _screen is None:
            return
        screen = _screen.availableGeometry()
        size = self.frameGeometry()
        x = screen.x() + max(0, (screen.width() - size.width()) // 2)
        y = screen.y() + max(0, (screen.height() - size.height()) // 2)
        self.move(x, y)

    def _animate_fade_in(self):
        """Animate window fade-in"""
        self.setWindowOpacity(0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()

    def _setup_shortcuts(self):
        """Keyboard-Shortcuts gemäß §11.4 der copilot-instructions (bindend).

        Tabelle:
            Leertaste   → Play / Pause
            A           → Original hören
            B           → Restauriert hören
            Ctrl+O      → Datei öffnen
            Ctrl+S      → Exportieren
            Ctrl+R      → RESTORATION starten
            Ctrl+Shift+R → STUDIO_2026 starten
            Escape      → Verarbeitung abbrechen
            Ctrl+Z      → Letzten Export-Pfad in Zwischenablage
        """
        from PyQt5.QtGui import QKeySequence
        from PyQt5.QtWidgets import QShortcut

        self._shortcuts = []

        def _bind_shortcut(seq, handler) -> None:
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

        _bind_shortcut(Qt.Key.Key_Space, self._toggle_playback)
        _bind_shortcut(
            Qt.Key.Key_A,
            lambda: self._play_audio(self._orig_audio, self._orig_sr) if self._orig_audio is not None else None,
        )
        _bind_shortcut(
            Qt.Key.Key_B,
            lambda: self._play_audio(self._rest_audio, self._rest_sr) if self._rest_audio is not None else None,
        )
        _bind_shortcut(QKeySequence.StandardKey.Open, self._open_file)
        _bind_shortcut(QKeySequence.StandardKey.Save, self._export_all)
        _bind_shortcut("Ctrl+R", lambda: self._process_with_mode("RESTORATION"))
        _bind_shortcut("Ctrl+Shift+R", lambda: self._process_with_mode("STUDIO_2026"))
        _bind_shortcut(Qt.Key.Key_Escape, self._cancel_processing)
        _bind_shortcut(QKeySequence.StandardKey.Undo, self._copy_last_output_to_clipboard)
        # L-Shortcut: Lyrics-Timeline-Overlay an/aus (§11.4 Spec 08)
        _bind_shortcut(Qt.Key.Key_L, self._toggle_lyrics_overlay)
        # F1: Shortcut-Übersicht
        _bind_shortcut(Qt.Key.Key_F1, self._show_shortcut_help)

    def _toggle_playback(self):
        """Leertaste: Play/Pause — Wiedergabe stoppen oder Original starten."""
        if not _SD_AVAILABLE:
            return
        if self._play_thread is not None and self._play_thread.is_alive():
            try:
                if not hasattr(self, "_sd_lock"):
                    self._sd_lock = threading.Lock()
                with self._sd_lock:
                    if _sd is not None:
                        _sd.stop()
            except Exception as exc:
                logger.warning("A/B playback toggle stop failed: %s", exc)
                if hasattr(self, "status_text"):
                    self.status_text.setStyleSheet("color: #B87A7A; font-size: 10pt;")
                    self.status_text.setText("⚠ Wiedergabe konnte nicht gestoppt werden.")
            return
        if self._orig_audio is not None:
            self._play_audio(self._orig_audio, self._orig_sr)

    def _show_shortcut_help(self) -> None:
        """F1 / ?: Zeigt Tastenkürzel-Übersicht als modalen Dialog."""
        from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("Aurik — Tastenkürzel")
        dlg.setFixedWidth(420)
        dlg.setStyleSheet(
            "QDialog { background: #0d0d1f; border: 1px solid rgba(102,126,234,0.5);"
            " border-radius: 10px; }"
            " QLabel { color: #d0d8ff; font-family: 'Segoe UI', sans-serif; }"
            " QPushButton { background: rgba(102,126,234,0.25); color: #fff;"
            " border: 1px solid rgba(102,126,234,0.6); border-radius: 6px;"
            " padding: 6px 20px; font-size: 10pt; }"
            " QPushButton:hover { background: rgba(102,126,234,0.50); }"
        )
        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)
        layout.setContentsMargins(20, 18, 20, 14)

        title_lbl = QLabel("<b style='font-size:13pt;'>⌨  Tastenkürzel</b>")
        title_lbl.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title_lbl)

        _shortcuts = [
            ("Space", "Wiedergabe Original starten / stoppen"),
            ("A", "Original abspielen"),
            ("B", "Restauriertes Audio abspielen"),
            ("Ctrl + O", "Datei öffnen"),
            ("Ctrl + S", "Exportieren"),
            ("Ctrl + R", "RESTORATION starten"),
            ("Ctrl + Shift + R", "STUDIO 2026 starten"),
            ("Esc", "Verarbeitung abbrechen"),
            ("Ctrl + Z", "Letzten Export-Pfad kopieren"),
            ("L", "Lyrics-Timeline-Overlay an/aus"),
            ("F1 / ?", "Diese Hilfe anzeigen"),
        ]
        for key, desc in _shortcuts:
            row = QLabel(
                f"<span style='font-size:9pt; color:#a0aaff; font-weight:600;"
                f" font-family:Courier New,monospace;'>{key}</span>"
                f"<span style='color:rgba(200,205,255,0.55);'>  —  </span>"
                f"<span style='font-size:9pt;'>{desc}</span>"
            )
            row.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(row)

        layout.addSpacing(6)
        btn_close = QPushButton("Schließen")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignCenter)
        dlg.exec_()

    def _cancel_processing(self):
        """Escape: Laufende Batch-Verarbeitung abbrechen."""
        if not (self.batch_thread and self.batch_thread.isRunning()):
            return
        if hasattr(self, "_watchdog_timer") and self._watchdog_timer.isActive():
            self._watchdog_timer.stop()
        self.batch_thread.requestInterruption()
        self.batch_thread.wait(3000)
        if self.batch_thread.isRunning():
            self.batch_thread.terminate()
            self.batch_thread.wait(2000)
        self._set_magic_buttons_enabled(True)
        if hasattr(self, "_heartbeat_timer") and self._heartbeat_timer.isActive():
            self._heartbeat_timer.stop()
        self.title_bar.set_status(t("status.cancelled"), "#B87A7A")
        self.status_text.setText(t("status.processing_cancelled"))

    def _copy_last_output_to_clipboard(self):
        """Ctrl+Z: Letzten Export-Pfad in die Zwischenablage kopieren."""
        last_output = None
        for item in reversed(self.batch_queue.items):
            if item.status == "completed" and item.output_file:
                last_output = item.output_file
                break
        if last_output:
            _clipboard = QApplication.clipboard()
            if _clipboard is not None:
                _clipboard.setText(str(last_output))
            self.status_text.setText(t("status.path_copied", file=Path(last_output).name))
        else:
            self.status_text.setText(t("status.no_export_path"))

    def _toggle_maximize(self):
        """Toggle window maximize/restore"""
        if self.is_maximized:
            self.showNormal()
            self.title_bar.is_maximized = False
            self.is_maximized = False
        else:
            self.showMaximized()
            self.title_bar.is_maximized = True
            self.is_maximized = True

    # Action Methods
    def dragEnterEvent(self, event):
        """Drag & Drop: Datei-Drop auf das Fenster akzeptieren."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".wma"}
            if any(Path(u.toLocalFile()).suffix.lower() in audio_exts for u in urls):
                event.acceptProposedAction()
                # Prominentes Drop-Ziel: gesamter Visualisierungsbereich leuchtet auf
                self.detected_medium_label.setStyleSheet("""
                    color: #8fa4e8; font-size: 12pt; font-weight: bold; padding: 15px;
                    background: rgba(102, 126, 234, 0.18);
                    border-radius: 12px;
                    border: 3px dashed rgba(102, 126, 234, 0.85);
                """)
                self.detected_medium_label.setText(t("status.drop_release_to_load"))
                # Waveform-Bereich ebenfalls hervorheben
                if hasattr(self, "waveform_widget"):
                    self.waveform_widget.setStyleSheet("border: 3px dashed rgba(76,175,80,0.85); border-radius: 12px;")
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Drag & Drop: Rahmen zurücksetzen wenn Datei das Fenster verlässt."""
        # Nur zurücksetzen wenn keine Datei geladen ist — sonst bleibt die erkannte
        # Carrier-Info sichtbar (dragEnterEvent hatte sie temporär überschrieben).
        if not getattr(self, "current_file_path", None):
            self.detected_medium_label.setText("")
            self.detected_medium_label.setStyleSheet("""
                color: #7B93B8; font-size: 11pt; padding: 15px;
                background: rgba(102, 126, 234, 0.10);
                border-radius: 10px;
                border: 2px solid rgba(102, 126, 234, 0.22);
            """)
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.setStyleSheet("")

    def dropEvent(self, event):
        """Drag & Drop: Abgelegte Audiodatei(en) laden."""
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".wma"}
        paths = [
            Path(u.toLocalFile()) for u in event.mimeData().urls() if Path(u.toLocalFile()).suffix.lower() in audio_exts
        ]
        if paths:
            self._load_file(str(paths[0]))
            for p in paths[1:]:
                self._add_to_queue(str(p))
        self.dragLeaveEvent(event)  # Rahmen zurücksetzen (inkl. waveform_widget)

    def _load_file(self, file_path: str):
        """Datei nicht-blockierend laden: sf.read + Carrier-Forensics im Hintergrundthread."""
        self.current_file_path = file_path

        # Cache-Invalidierung: veralteter DefectScan für diese Datei entfernen (§9.4).
        # Nötig wenn dieselbe Datei nach einer Änderung erneut geöffnet wird.
        _bridge_clear_defect_cache(file_path)
        _bridge_clear_era_genre_cache(file_path)
        # Lyrics-Overlay-Zustand zurücksetzen (neue Datei → kein altes Transkript)
        self._lyrics_overlay_visible = False
        if hasattr(self, "waveform_widget"):
            self.waveform_widget._lyrics_transcription = None

        # Sofortiges visuelles Feedback im Haupt-Thread (BEVOR der Hintergrundthread startet)
        self.status_text.setText(t("status.loading_file", file=Path(file_path).name))
        self.status_text.setStyleSheet("color: #B8A068; font-size: 10pt;")
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.is_loading = True
            self.waveform_widget.audio_data = None  # Drop-Overlay ersetzt durch Lade-Overlay
            self.waveform_widget.update()

        # Buttons sofort deaktivieren – bleiben disabled, bis Defektanalyse fertig ist
        for _btn_name in ("btn_magic_restoration", "btn_magic_studio"):
            if hasattr(self, _btn_name):
                _btn = getattr(self, _btn_name)
                _btn.setEnabled(False)
                _btn.update()
        if hasattr(self, "progress_bar"):
            self.progress_bar.setRange(0, 10000)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

        # Restaurierbarkeits-Banner zurücksetzen (neues File) – vor Thread-Start
        if hasattr(self, "restorability_banner"):
            self.restorability_banner.setVisible(False)
            self.restorability_banner.setText("")
        self._restorability_score = 50.0
        self._recommended_mode = "RESTORATION"
        if hasattr(self, "mode_recommendation_label"):
            self.mode_recommendation_label.setVisible(False)
            self.mode_recommendation_label.setText("—")
        self._apply_mode_recommendation_visuals()

        # ── PFLICHT-GATE §10.5: AudioFileValidator vor dem Laden ─────────────
        # Zugriff ausschließlich über Bridge (§11.4 — kein direkter core/-Import)
        try:
            _validator = _bridge_get_audio_file_validator()
            if _validator is None:
                raise RuntimeError(
                    "Backend-Bridge nicht verfügbar. Datei-Validierung ohne Bridge ist im Frontend deaktiviert."
                )
            _val_result = _validator.validate(Path(file_path))
            _warnings = getattr(_val_result, "warnings", []) or []
            if _warnings:
                for _w in _warnings:
                    logger.warning("AudioFileValidator: %s", _w)
        except Exception as _val_exc:
            # Nutzer-sichtbare Fehlermeldung auf Deutsch (AudioLoadError hat .message_user)
            _user_msg = getattr(_val_exc, "message_user", str(_val_exc))
            for _btn_name in ("btn_magic_restoration", "btn_magic_studio"):
                if hasattr(self, _btn_name):
                    getattr(self, _btn_name).setEnabled(True)
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
            # Spec §11.4: warning() + detected_medium_label (nicht critical/status_text)
            QMessageBox.warning(
                self,
                t("dialog.invalid_file_title"),
                t("dialog.invalid_file_body", error=_user_msg),
            )
            if hasattr(self, "detected_medium_label"):
                self.detected_medium_label.setText(t("status.invalid_file", file=Path(file_path).name))
            return

        # ── Schwere I/O + Carrier-Forensics in Hintergrundthread ──────────────
        def _bg_load():
            """Läuft komplett im Hintergrundthread – kein Qt-Widget-Aufruf hier.

            Audio-Lade-Kaskade (robuste Mehrstufen-Strategie):
              1. soundfile.SoundFile   – WAV, FLAC, OGG, AIFF (chunk-basiert, Prozent-Feedback)
              2. pedalboard.io.AudioFile – MP3, M4A, WMA, AAC (chunk-basiert)
              3. librosa.load()        – Letzter Fallback (audioread-Backend)
            """
            audio: np.ndarray | None = None
            sr: int = 48000
            _load_errors: list[str] = []

            def _set_progress(pct: int) -> None:
                """Fortschritt thread-sicher im Haupt-Thread setzen (via Signal)."""
                self._load_progress.emit(pct)

            # ── Stufe 1: soundfile (chunk-basiert, WAV / FLAC / OGG / AIFF) ──
            try:
                with sf.SoundFile(file_path) as _sf_file:
                    _total = len(_sf_file)
                    sr = _sf_file.samplerate
                    _chunk = max(1, _total // 50)
                    _chunks: list = []
                    _read = 0
                    while _read < _total:
                        _block = _sf_file.read(min(_chunk, _total - _read))
                        if len(_block) == 0:
                            break  # Unexpected EOF – avoid infinite loop
                        _chunks.append(_block)
                        _read += len(_block)
                        _pct = int(_read / _total * 100)
                        _set_progress(_pct)
                    logger.debug("soundfile loop done: _read=%d, _total=%d", _read, _total)
                    if not _chunks:
                        raise ValueError("soundfile: keine Frames gelesen (leere oder beschädigte Datei)")
                    audio = np.concatenate(_chunks, axis=0) if len(_chunks) > 1 else _chunks[0]
                    del _chunks  # Chunk-Liste sofort freigeben (OOM-Schutz)
            except Exception as _e1:
                _load_errors.append(f"soundfile: {_e1}")
                audio = None

            # ── Stufe 2: pedalboard (chunk-basiert, MP3 / M4A / WMA / AAC) ───
            if audio is None:
                try:
                    from pedalboard.io import AudioFile  # type: ignore

                    with AudioFile(file_path) as _f:
                        sr = int(_f.samplerate)
                        _frames = _f.frames
                        _chunk_pb = max(1, _frames // 50)
                        _clist: list = []
                        _read2 = 0
                        while _read2 < _frames:
                            _block2 = _f.read(min(_chunk_pb, _frames - _read2))  # (ch, samples)
                            _clist.append(_block2)
                            _read2 += _block2.shape[-1]
                            _set_progress(int(_read2 / _frames * 100))
                        _raw = np.concatenate(_clist, axis=1)  # (ch, total)
                        del _clist  # Chunk-Liste sofort freigeben (OOM-Schutz)
                        audio = np.ascontiguousarray(_raw.T)  # (total, ch) – zusammenhängend
                        del _raw  # Original sofort freigeben (OOM-Schutz)
                        if audio.ndim == 1:
                            pass
                        elif audio.shape[1] == 1:
                            audio = audio[:, 0]
                except Exception as _e2:
                    _load_errors.append(f"pedalboard: {_e2}")
                    audio = None

            # ── Stufe 3: librosa (letzter Fallback – audioread/GStreamer) ────
            # ACHTUNG: audioread/GStreamer kann auf Linux-Systemen (Zorin OS, Ubuntu)
            # unbegrenzt hängen → Timeout-Thread (90 s) verhindert Freeze + Absturz.
            if audio is None:
                try:
                    import librosa  # type: ignore

                    _lib_result: list = []
                    _lib_error: list = []

                    def _librosa_load_fn() -> None:
                        try:
                            _set_progress(20)
                            _y2, _sr2 = librosa.load(file_path, sr=None, mono=False)
                            _lib_result.append((_y2, int(_sr2)))
                        except Exception as _le:
                            _lib_error.append(_le)

                    _lib_t = threading.Thread(target=_librosa_load_fn, daemon=True)
                    _lib_t.start()
                    _lib_t.join(timeout=90)  # max. 90 s – GStreamer hängt nicht ewig
                    if _lib_t.is_alive():
                        _load_errors.append(
                            "librosa: Timeout nach 90 s – GStreamer/audioread hängt. "
                            "Bitte ffmpeg installieren: sudo apt install ffmpeg"
                        )
                    elif _lib_error:
                        _load_errors.append(f"librosa: {_lib_error[0]}")
                    else:
                        _y, _tmp_sr = _lib_result[0]
                        sr = _tmp_sr
                        _set_progress(90)
                        audio = _y.T if _y.ndim == 2 else _y
                        del _y  # Originalarray sofort freigeben (OOM-Schutz)
                except Exception as _e3:
                    _load_errors.append(f"librosa: {_e3}")

            # ── Alle Stufen gescheitert ────────────────────────────────────────
            if audio is None:
                _msg = " | ".join(_load_errors)[:200]

                def _err():
                    for _bn in ("btn_magic_restoration", "btn_magic_studio"):
                        if hasattr(self, _bn):
                            getattr(self, _bn).setEnabled(True)
                    if hasattr(self, "progress_bar"):
                        self.progress_bar.setRange(0, 10000)
                        self.progress_bar.setValue(0)
                        self.progress_bar.setVisible(False)
                    if hasattr(self, "detected_medium_label"):
                        self.detected_medium_label.setText(
                            t("status.load_failed_with_reason", file=Path(file_path).name, error=_msg)
                        )
                        self.detected_medium_label.setStyleSheet("""
                            color: #B87A7A; font-size: 11pt; padding: 12px;
                            background: rgba(148, 82, 82, 0.10);
                            border-radius: 8px; border: 2px solid rgba(152, 88, 88, 0.26);
                        """)
                    from PyQt5.QtWidgets import QMessageBox

                    QMessageBox.warning(
                        self,
                        t("dialog.import_failed_title"),
                        t("dialog.import_failed_body", file=Path(file_path).name, error=_msg[:300]),
                    )

                QTimer.singleShot(0, _err)
                return

            # Laden vollständig — Balken auf 100 % (letzte Iteration ist bereits 100 %,
            # explizites Setzen als Sicherheitsnetz für librosa-Pfad)
            _set_progress(100)
            logger.debug("_bg_load: progress 100 emitted, dispatching _on_file_loaded")

            audio = _normalize_audio(audio)

            # Resample to 48 kHz — Aurik internal SR (accepts any input SR)
            if sr != 48_000:
                from scipy.signal import resample_poly as _rp

                _gcd = math.gcd(int(sr), 48_000)
                audio = _rp(
                    audio,
                    48_000 // _gcd,
                    int(sr) // _gcd,
                    axis=0 if audio.ndim > 1 else -1,
                ).astype(np.float32)
                sr = 48_000
                audio = _normalize_audio(audio)

            # _on_file_loaded im GUI-Thread starten
            _audio_ref = audio
            _sr_ref = int(sr)
            # Carrier-Ergebnis: Platzhalter → wird asynchron nachgefüllt
            self._dispatch_to_gui(
                lambda: self._on_file_loaded(_audio_ref, _sr_ref, file_path, t("status.analyzing_wait"), 0)
            )

            # Carrier-Forensics läuft NACH dem GUI-Update in separatem Thread
            def _carrier_bg(_a=audio, _s=int(sr)):
                # PNG-Icon-Pfad (relativ zu dieser Datei)
                import os as _os

                _ICONS_DIR = _os.path.join(
                    _os.path.dirname(_os.path.dirname(__file__)),
                    "resources",
                    "carrier_icons",
                )

                def _html(icon_key: str, label: str) -> str:
                    _p = _os.path.join(_ICONS_DIR, f"{icon_key}.png")
                    return (
                        f'<img src="file:///{_p}" width="22" height="22" style="vertical-align:middle;">&nbsp;{label}'
                    )

                # (icon_key, Anzeige-Name) je Trägermedium
                _MEDIUM_DATA: dict[str, tuple[str, str]] = {
                    "wax_cylinder": ("wax_cylinder", "Wachswalze"),
                    "lacquer_disc": ("lacquer_disc", "Lackfolie"),
                    "shellac": ("shellac", "Schellack"),
                    "vinyl": ("vinyl", "Vinyl"),
                    "wire_recording": ("wire_recording", "Drahtband"),
                    "reel_tape": ("reel_tape", "Spulenband"),
                    "tape": ("tape", "Magnetband"),
                    "cassette": ("cassette", "Kassette"),
                    "dat": ("dat", "DAT"),
                    "cd_digital": ("cd_digital", "CD"),
                    "cd": ("cd", "CD"),
                    "digital": ("cd_digital", "Digital"),
                    "minidisc": ("minidisc", "MiniDisc"),
                    "mp3_low": ("mp3_low", "MP3 (schwach)"),
                    "mp3_high": ("mp3_high", "MP3"),
                    "damaged_mp3": ("damaged_mp3", "MP3 (defekt)"),
                    "aac": ("aac", "AAC"),
                    "streaming": ("streaming", "Streaming"),
                    "unknown": ("unknown", "Unbekannt"),
                }
                # (icon_key, Anzeige-Name) je Dateicontainer
                _EXT_DATA: dict[str, tuple[str, str]] = {
                    ".mp3": ("mp3_high", "MP3"),
                    ".m4a": ("aac", "M4A/AAC"),
                    ".aac": ("aac", "AAC"),
                    ".ogg": ("streaming", "OGG"),
                    ".opus": ("streaming", "Opus"),
                    ".wma": ("streaming", "WMA"),
                    ".flac": ("cd_digital", "FLAC"),
                    ".wav": ("cd_digital", "WAV"),
                    ".aiff": ("cd_digital", "AIFF"),
                    ".aif": ("cd_digital", "AIFF"),
                }
                # Analoge/physikalische Ursprungsmedien (Ära 0 + 1)
                _ANALOG_MEDIA = frozenset(
                    {
                        "wax_cylinder",
                        "lacquer_disc",
                        "shellac",
                        "vinyl",
                        "wire_recording",
                        "reel_tape",
                        "tape",
                        "cassette",
                    }
                )
                _raw_medium = "unknown"
                _score = 0
                _mc_result_obj = None
                try:
                    _classify_medium = _bridge_get_medium_classifier_fn()
                    if callable(_classify_medium):
                        _mono = np.mean(_a, axis=1) if _a.ndim > 1 else _a
                        _res = _classify_medium(_mono, _s)
                        _mc_result_obj = _res
                        _raw_medium = _res.material_type
                        _score = round(_res.confidence * 5)
                except Exception:
                    pass
                # MediumClassifier-Ergebnis im Bridge-Cache speichern, damit UV3
                # es wiederverwenden kann statt ML-Klassifikator erneut auszuführen.
                if _mc_result_obj is not None:
                    cache_medium_result(file_path, _mc_result_obj)
                # HTML-Icon für Ursprungsträger
                _orig_html = _html(*_MEDIUM_DATA.get(_raw_medium, ("unknown", _raw_medium)))
                # Kettenanzeige: analoger Ursprungsträger → Container-Icon
                _ext = Path(file_path).suffix.lower()
                _ext_entry = _EXT_DATA.get(_ext)
                if _ext_entry and _raw_medium in _ANALOG_MEDIA:
                    _lbl = f"{_orig_html}&nbsp;&nbsp;→&nbsp;&nbsp;{_html(*_ext_entry)}"
                else:
                    _lbl = _orig_html
                # Ergebnis vor GUI-Dispatch speichern (Race-Condition-Fix)
                # _continue_file_loaded liest diesen Wert falls es NACH dem
                # Dispatch ausgeführt wird und würde sonst "Wird analysiert…" anzeigen
                self._carrier_bg_label = _lbl
                self._carrier_bg_score = _score
                self._raw_medium_type = _raw_medium  # raw MaterialType key (e.g. "shellac", "vinyl")
                self._era_genre_badge = ""  # Badge bei neuem File zurücksetzen
                # Label im GUI-Thread aktualisieren
                self._dispatch_to_gui(lambda l=_lbl, sc=_score: self._update_carrier_display(l, sc, file_path))

            threading.Thread(target=_carrier_bg, daemon=True).start()

        threading.Thread(target=_bg_load, daemon=True).start()

    # ── Thread-sichere GUI-Dispatch-Helfer ─────────────────────────────────
    def _dispatch_to_gui(self, fn) -> None:
        """Ruft `fn()` thread-sicher im GUI-Thread auf (via pyqtSignal)."""
        self._gui_dispatch.emit(fn)

    def _update_carrier_display(self, carrier_label: str, carrier_score: int, file_path: str) -> None:
        """Aktualisiert den Carrier-Label im GUI-Thread (nach async Analyse)."""
        if hasattr(self, "detected_medium_label"):
            _stars = self._render_star_html(carrier_score)
            # Era/Genre-Badge mitnehmen, falls bereits berechnet (Race-Condition-Fix)
            _badge = getattr(self, "_era_genre_badge", "")
            self.detected_medium_label.setText(f"{carrier_label}   {_stars}{_badge}")
            self.detected_medium_label.setStyleSheet("""
                color: #82B89A; font-size: 11pt; padding: 12px;
                background: rgba(85, 155, 115, 0.10);
                border-radius: 8px; border: 2px solid rgba(100, 168, 130, 0.26);
                margin-top: 8px; font-weight: 600;
            """)
        if hasattr(self, "current_detected_carrier"):
            self.current_detected_carrier = carrier_label
        if hasattr(self, "current_carrier_confidence"):
            self.current_carrier_confidence = carrier_score
        self._apply_mode_recommendation_visuals()

    @staticmethod
    def _render_star_html(score: int, size: int = 18) -> str:
        """Render star rating as HTML <img> tags using premium star icons."""
        import os as _os

        _star_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "resources", "star_icons")
        _full = _os.path.join(_star_dir, "star_full.png")
        _empty = _os.path.join(_star_dir, "star_empty.png")
        parts = []
        for i in range(5):
            _p = _full if i < score else _empty
            parts.append(f'<img src="file:///{_p}" width="{size}" height="{size}" style="vertical-align:middle;">')
        return "".join(parts)

    def _on_file_loaded(self, audio: np.ndarray, sr: int, file_path: str, carrier_label: str, carrier_score: int):
        """Wird im Haupt-Thread aufgerufen, nachdem sf.read + Carrier-Forensics fertig sind."""
        logger.debug("_on_file_loaded called: audio.shape=%s sr=%d", audio.shape, sr)
        # Lade-Zustand des WaveformWidgets aufheben
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.is_loading = False
        # Fortschrittsbalken: 100 % sichtbar anzeigen
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(10000)
            self.progress_bar.setFormat(t("status.import_done"))

        # Export-Dialog mit kurzem Delay öffnen, damit 100 % kurz gerendert wird
        def _open_export_dialog(
            _audio=audio,
            _sr=sr,
            _fp=file_path,
            _cl=carrier_label,
            _cs=carrier_score,
        ):
            logger.debug("_open_export_dialog: opening ExportConfigDialog")
            _dlg = ExportConfigDialog(_fp, parent=self)
            logger.debug("ExportConfigDialog created, calling exec()")
            if _dlg.exec() != QDialog.DialogCode.Accepted:
                if hasattr(self, "progress_bar"):
                    self.progress_bar.setVisible(False)
                    self.progress_bar.setValue(0)
                if hasattr(self, "status_text"):
                    self.status_text.setText(t("status.import_cancelled"))
                return
            self._export_config = _dlg.get_config()
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
                self.progress_bar.setValue(0)
            self._continue_file_loaded(_audio, _sr, _fp, _cl, _cs)

        QTimer.singleShot(150, _open_export_dialog)

    def _continue_file_loaded(
        self,
        audio: "np.ndarray",
        sr: int,
        file_path: str,
        carrier_label: str,
        carrier_score: int,
    ):
        """Setzt _on_file_loaded nach dem Export-Dialog fort (UI-Block)."""
        try:  # pylint: disable=too-many-statements
            _audio = _normalize_audio(audio)
            _sr = int(sr)
            audio_mono = np.mean(_audio, axis=1) if len(_audio.shape) > 1 else _audio
            # A/B-Player: Original speichern
            self._orig_audio = _audio
            self._orig_sr = _sr
            self._rest_audio = None  # zurücksetzen nach neuem File

            # Restaurierbarkeits-Banner zurücksetzen (neues File)
            if hasattr(self, "restorability_banner"):
                self.restorability_banner.setVisible(False)
                self.restorability_banner.setText("")

            try:
                # Race-Condition-Fix: Falls _carrier_bg bereits ein Ergebnis
                # gespeichert hat (schneller als Dialog-Close), dieses verwenden
                # statt des Platzhalters "Wird analysiert …".
                detected_carrier = getattr(self, "_carrier_bg_label", carrier_label)
                confidence = getattr(self, "_carrier_bg_score", carrier_score)
                _stars = self._render_star_html(confidence)
                self.detected_medium_label.setText(f"{detected_carrier}   {_stars}")
                self.detected_medium_label.setStyleSheet("""
                    color: #82B89A; font-size: 11pt; padding: 12px;
                    background: rgba(85, 155, 115, 0.10);
                    border-radius: 8px; border: 2px solid rgba(100, 168, 130, 0.26);
                    margin-top: 8px; font-weight: 600;
                """)
                self.current_detected_carrier = detected_carrier
                self.current_carrier_confidence = confidence

                # ── Ära- und Genre-Erkennung im Hintergrund ────────────────
                # Kein .copy(): audio_mono wird von allen Hintergrundthreads nur gelesen
                _sr2 = int(_sr)
                _base_text = f"{detected_carrier}   {_stars}"
                # NOTE: _base_text used in _detect_era_genre_bg closure already contains star HTML

                def _detect_era_genre_bg(_a=audio_mono, _s=_sr2, _bt=_base_text, _self=self):
                    decade_label = ""
                    genre_label = ""
                    _era_result_obj = None
                    _genre_result_obj = None
                    # Analyse-Module (EraClassifier, GenreClassifier) sind per Spec SR-agnostisch
                    # (§2.37 / Performance-Budget). Resampling auf 48 kHz wird versucht, aber bei
                    # Fehler wird mit nativer SR weitergemacht — KEIN früher Abbruch.
                    _a_48, _s_48 = _a, _s
                    if _s != 48000:
                        try:
                            import librosa  # lazy — nur für Resampling

                            _a_48 = librosa.resample(_a, orig_sr=_s, target_sr=48000)
                            _s_48 = 48000
                        except Exception as _rs_exc:
                            logger.warning(
                                "Era/Genre: Resampling auf 48 kHz fehlgeschlagen, nutze native SR %d Hz (Ursache: %s)",
                                _s,
                                _rs_exc,
                            )
                            # Spec §2.37: Analyse-Module arbeiten bei nativer Import-SR —
                            # Klassifikation mit originalem SR fortsetzen statt abzubrechen.
                            _a_48, _s_48 = _a, _s
                    try:
                        _classify_era = _bridge_get_era_classifier_fn()
                        if callable(_classify_era):
                            er = _classify_era(_a_48, _s_48)
                            # §Medium-Floor-Constraint: Ära kann nicht vor Erfindung
                            # des erkannten Mediums liegen (z. B. Magnetband ≠ 1890er).
                            _raw_med = getattr(_self, "_raw_medium_type", None)
                            if _raw_med and _raw_med not in ("unknown", ""):
                                try:
                                    from backend.core.era_classifier import constrain_era_to_medium as _era_floor

                                    er = _era_floor(er, str(_raw_med))
                                except Exception:
                                    pass
                            _era_result_obj = er
                            dec = getattr(er, "decade", None) or (er.get("decade") if isinstance(er, dict) else None)
                            if dec:
                                decade_label = f"{dec}er"
                    except Exception as _era_exc:
                        logger.warning(
                            "Era-Erkennung im Frontend fehlgeschlagen (Ursache: %s). "
                            "Ära wird nach Restaurierung aus Ergebnis geladen.",
                            _era_exc,
                        )
                    try:
                        _classify_genre = _bridge_get_genre_classifier_fn()
                        if callable(_classify_genre):
                            gr = _classify_genre(_a_48, _s_48)
                            _genre_result_obj = gr
                            gl = getattr(gr, "genre_label", None) or (
                                gr.get("genre_label") if isinstance(gr, dict) else None
                            )
                            if gl and gl.lower() not in ("unbekannt", "unknown", ""):
                                genre_label = str(gl)
                    except Exception as _genre_exc:
                        logger.warning(
                            "Genre-Erkennung im Frontend fehlgeschlagen (Ursache: %s). "
                            "Genre wird nach Restaurierung aus Ergebnis geladen.",
                            _genre_exc,
                        )

                    # Era/Genre-Ergebnisse im Bridge-Cache speichern, damit UV3 sie
                    # wiederverwenden kann statt ML-Klassifikatoren erneut auszuführen.
                    _fp = getattr(_self, "current_file_path", None)
                    if _fp and (_era_result_obj is not None or _genre_result_obj is not None):
                        cache_era_genre_result(_fp, _era_result_obj, _genre_result_obj)
                    badge = ""
                    if decade_label:
                        badge = f"  │  ◷ {decade_label}"
                    if genre_label:
                        badge += f" · {genre_label}"
                    if not badge:
                        return  # nichts zu ergänzen
                    _carrier_name = detected_carrier
                    tip = (
                        f"<b>Träger-Forensik &amp; Aufnahme-Epoche</b><br>Erkanntes Medium: <b>{_carrier_name}</b><br>"
                    )
                    if decade_label:
                        tip += f"Aufnahme-Ära: <b>{decade_label}</b><br>"
                    if genre_label:
                        tip += f"Genre: <b>{genre_label}</b><br>"
                    tip += "<small>Die Ära-Erkennung passt alle Restaurierungs-Parameter historisch korrekt an.</small>"

                    def _upd(_badge=badge, _tip=tip, _decade=decade_label, _genre=genre_label):
                        if not hasattr(_self, "detected_medium_label"):
                            return
                        # Badge für _update_carrier_display merken (Race-Condition-Fix)
                        _self._era_genre_badge = _badge
                        # Aktuellen Carrier-Label lesen (kann inzwischen gesetzt worden sein)
                        _cur_lbl = getattr(_self, "_carrier_bg_label", _bt)
                        _cur_sc = getattr(_self, "_carrier_bg_score", 0)
                        _cur_stars = _self._render_star_html(_cur_sc)
                        _self.detected_medium_label.setText(f"{_cur_lbl}   {_cur_stars}{_badge}")
                        _self.detected_medium_label.setToolTip(_tip)
                        _self._apply_mode_recommendation_visuals()

                    _self._dispatch_to_gui(_upd)

                threading.Thread(target=_detect_era_genre_bg, daemon=True).start()
                # ── Ende Ära/Genre-Hintergrund ────────────────────────────

            except Exception:
                self.detected_medium_label.setText(t("status.analyzing_wait"))
                self.detected_medium_label.setStyleSheet("""
                    color: #7B93B8; font-size: 11pt; padding: 12px;
                    background: rgba(102, 126, 234, 0.10);
                    border-radius: 8px; border: 2px solid rgba(102, 126, 234, 0.22);
                """)

            self._update_waveform(_audio, _sr)
            # A/B Original-Waveform direkt beim Öffnen befüllen (nicht erst nach Restaurierung)
            if hasattr(self, "waveform_widget_orig_ab"):
                with contextlib.suppress(Exception):
                    self.waveform_widget_orig_ab.update_waveform(_audio, _sr)
            # A/B-Buttons aktivieren
            self._update_ab_player_state()

            # ── Restaurierbarkeit-Vorschau im Hintergrund ──────────────────
            # Nicht-blockierend (<3 s), aktualisiert das Banner über den
            # Magic Buttons sobald das Ergebnis vorliegt.
            # audio_mono wird nur gelesen – kein .copy() (OOM-Schutz)
            _sr_cap = int(_sr)

            def _estimate_restorability_bg(_a=audio_mono, _s=_sr_cap, _self=self):
                _restorability_result_obj = None

                # ── Schrittweise Fortschrittsanzeige im Banner ─────────────
                _step_css = (
                    "color: #90A4AE; font-size:10pt; font-weight:600;"
                    " padding:10px 18px; border-radius:10px;"
                    " background: rgba(144, 164, 174, 0.10);"
                    " border: 1px solid rgba(144, 164, 174, 0.25);"
                )
                _steps = [
                    "⏳  Signal wird analysiert …",
                    "⏳  Rauschabstand wird gemessen …",
                    "⏳  Clipping-Analyse …",
                    "⏳  Frequenzbandbreite wird geprüft …",
                    "⏳  Impuls-/Crackle-Dichte wird bestimmt …",
                    "⏳  Restaurierbarkeit wird berechnet …",
                ]

                def _show_step(msg: str) -> None:
                    def _do():
                        if hasattr(_self, "restorability_banner"):
                            _self.restorability_banner.setText(msg)
                            _self.restorability_banner.setStyleSheet(_step_css)
                            _self.restorability_banner.setVisible(True)

                    _self._dispatch_to_gui(_do)

                _show_step(_steps[0])

                try:
                    _RestorabilityEstimator = _bridge_get_restorability_estimator_class()
                    if _RestorabilityEstimator is None:
                        raise ImportError("RestorabilityEstimator nicht verfügbar")
                    _estimator = _RestorabilityEstimator()

                    # Schrittweise Analyse mit UI-Feedback zwischen den Stufen
                    _mono_e = np.mean(_a, axis=0).astype(np.float32) if _a.ndim == 2 else _a.astype(np.float32)
                    _mono_e = np.nan_to_num(_mono_e, nan=0.0, posinf=0.0, neginf=0.0)

                    _show_step(_steps[1])
                    _estimator._estimate_snr(_mono_e, _s)

                    _show_step(_steps[2])
                    _clip = float(np.mean(np.abs(_mono_e) >= 0.98))

                    _show_step(_steps[3])
                    _estimator._estimate_bandwidth(_mono_e, _s)

                    _show_step(_steps[4])
                    _estimator._estimate_crackle_rate(_mono_e, _s)

                    _show_step(_steps[5])
                    r = _estimator.estimate(_a, _s)
                    _restorability_result_obj = r
                    score100 = float(getattr(r, "restorability_score", 50.0))
                    predicted_mos = float(getattr(r, "predicted_mos", 3.5))
                    limiting = list(getattr(r, "limiting_defects", []))
                except Exception:
                    # DSP-Heuristik: SNR-basiert (läuft immer als Fallback)
                    _mono = _a.astype(np.float64)
                    rms = float(np.sqrt(np.mean(_mono**2))) + 1e-12
                    noise = float(np.percentile(np.abs(_mono), 5)) + 1e-12
                    snr_db = 20.0 * np.log10(rms / noise)
                    score100 = float(np.clip((snr_db - 8.0) / 35.0 * 100.0, 5.0, 98.0))
                    predicted_mos = round(1.0 + score100 / 100.0 * 4.0, 1)
                    limiting = []

                # Restorability-Ergebnis im Bridge-Cache speichern, damit UV3
                # es wiederverwenden kann statt Estimator erneut auszuführen.
                _fp = getattr(_self, "current_file_path", None)
                if _fp and _restorability_result_obj is not None:
                    cache_restorability_result(_fp, _restorability_result_obj)

                # Kategorie
                if score100 >= 70:
                    _s_color = "#82B89A"
                    bg = "rgba(85, 155, 115, 0.14)"
                    border = "rgba(100, 168, 130, 0.36)"
                    zeile1 = f"▶  Sehr gut restaurierbar  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Aurik kann diese Aufnahme auf exzellentem Niveau restaurieren."
                elif score100 >= 40:
                    _s_color = "#B8A068"
                    bg = "rgba(150, 130, 68, 0.14)"
                    border = "rgba(150, 130, 68, 0.36)"
                    zeile1 = f"▶  Mäßig restaurierbar  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Deutliche Verbesserung möglich – Restdefekte können bleiben."
                else:
                    _s_color = "#B87A7A"
                    bg = "rgba(148, 82, 82, 0.14)"
                    border = "rgba(152, 88, 88, 0.36)"
                    zeile1 = f"▶  Stark beschädigt  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Material ist sehr stark beschädigt – Aurik holt das physikalisch Mögliche heraus."

                mos_str = f"{predicted_mos:.1f}"
                banner_txt = (
                    f"{zeile1}    ·    Erw. Qualität nach Restaurierung:  {mos_str}\u202f/\u202f5,0 MOS\n{detail}"
                )
                if limiting:
                    banner_txt += f"    (Hauptdefekte: {', '.join(str(d) for d in limiting[:2])})"
                # Detaillierte Messwerte für Tooltip zusammenstellen
                _tip_details = ""
                if _restorability_result_obj is not None:
                    _snr_val = getattr(_restorability_result_obj, "snr_db", None)
                    _grade_val = getattr(_restorability_result_obj, "grade", None)
                    _time_est = getattr(_restorability_result_obj, "processing_time_estimate_s", None)
                    if _snr_val is not None:
                        _tip_details += f"SNR: <b>{_snr_val:.1f}\u202fdB</b><br>"
                    if _grade_val is not None:
                        _tip_details += f"Stufe: <b>{_grade_val}</b><br>"
                    if _time_est is not None:
                        _min = int(_time_est // 60)
                        _sec = int(_time_est % 60)
                        _tip_details += f"Gesch. Verarbeitungszeit: <b>~{_min}\u202fMin.\u202f{_sec}\u202fSek.</b><br>"
                tip = (
                    f"<b>Restaurierbarkeits-Vorschätzung</b><br>"
                    f"Wert: <b>{score100:.0f}\u202f/\u202f100</b><br>"
                    f"Erw. Qualität nach Restaurierung: <b>{mos_str} von 5,0 MOS</b><br>"
                    f"{_tip_details}"
                    f"<small>Schnelle Vorab-Analyse des Signals – "
                    f"vor dem eigentlichen Restaurierungsvorgang.</small>"
                )
                css = (
                    f"color:{_s_color}; font-size:10pt; font-weight:600;"
                    f" padding:10px 18px; border-radius:10px;"
                    f" background:{bg}; border:1px solid {border};"
                )

                def _update_ui():
                    _self._restorability_score = score100
                    if hasattr(_self, "restorability_banner"):
                        _self.restorability_banner.setText(banner_txt)
                        _self.restorability_banner.setStyleSheet(css)
                        _self.restorability_banner.setToolTip(tip)
                        _self.restorability_banner.setVisible(True)
                    _self._apply_mode_recommendation_visuals()

                _self._dispatch_to_gui(_update_ui)

            threading.Thread(target=_estimate_restorability_bg, daemon=True).start()
            # ── Ende Restaurierbarkeit-Hintergrund ─────────────────────────

            # ── Defekt-Analyse im Hintergrund nach Import ──────────────────
            # DefectScanner.scan() läuft nicht-blockierend (<3 s) und befüllt
            # sofort das Defekt-Panel — ohne dass der Nutzer den Magic Button
            # drücken muss (§2.1 DefectScanner, §9.5 Performance-Budget).

            # Sofortiges Feedback im Haupt-Thread → Nutzer sieht direkt, dass
            # die Analyse läuft (kein leeres Label bis der Thread fertig ist).
            if hasattr(self, "defect_summary_label"):
                self.defect_summary_label.setText(t("status.defects_summary_analyzing"))
                self.defect_summary_label.setStyleSheet("""
                    color: #90A4AE; font-size: 10pt; padding: 12px;
                    background: rgba(144, 164, 174, 0.10);
                    border-radius: 10px; border: 1px solid rgba(144, 164, 174, 0.25);
                """)
            # Live-Zähler-Label beim Scan-Start sichtbar schalten
            if hasattr(self, "defect_count_live_label"):
                self.defect_count_live_label.setText(t("status.analyzing_short"))
                self.defect_count_live_label.setStyleSheet(
                    "color: #90A4AE; font-size: 8pt; background: transparent; padding: 0 2px;"
                )
                self.defect_count_live_label.setVisible(True)

            # Pulsierender Fortschrittsbalken während Defektanalyse
            if hasattr(self, "progress_bar"):
                self.progress_bar.setRange(0, 10000)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat(t("status.defects_progress"))
                self.progress_bar.setVisible(True)
            if hasattr(self, "status_text"):
                self.status_text.setText(t("status.defects_analyzing"))
                self.status_text.setStyleSheet("color: #B8A068; font-size: 10pt;")

            # audio_mono wird nur gelesen – kein .copy() (OOM-Schutz)
            _sr_scan = int(_sr)

            def _run_defect_scan_bg(_a=audio_mono, _s=_sr_scan, _self=self, _fp=file_path):
                import logging as _log_

                _logger_ = _log_.getLogger(__name__)

                def _on_scan_progress(pct: int) -> None:
                    """Leitet Scan-Fortschritt thread-sicher in den GUI-Thread weiter."""
                    _self._load_progress.emit(int(pct))

                try:
                    _DS = _bridge_get_defect_scanner()
                    if _DS is None:
                        raise RuntimeError("DefectScanner über Bridge nicht verfügbar")
                    _scan = _DS().scan(_a, _s, progress_callback=_on_scan_progress)
                    # P1: Scan-Ergebnis in Bridge-Cache speichern (vermeidet Doppelscan in BatchThread)
                    cache_defect_result(_fp, _scan)
                    defects = _defect_analysis_to_display(_scan.scores, status="detected")
                except Exception as _exc:
                    # Fehler sichtbar machen (Debug-Log) + DSP-Fallback
                    _logger_.warning("DefectScanner nach Import fehlgeschlagen: %s", _exc, exc_info=True)
                    defects = {
                        "clicks": 0,
                        "crackle": 0,
                        "pops": 0,
                        "clipping": 0,
                        "sibilance": 0,
                        "dropout": 0,
                        "hum": 0.0,
                        "noise_level": 0.0,
                        "wow": 0.0,
                        "flutter": 0.0,
                        "rumble": 0.0,
                        "status": "detected",
                    }

                def _apply():
                    # Spec §11.4: Label beim Scan-Start (Datei-Öffnen-Pfad) sichtbar schalten
                    if hasattr(_self, "defect_count_live_label"):
                        _self.defect_count_live_label.setText(t("status.analyzing_short"))
                        _self.defect_count_live_label.setVisible(True)
                    if hasattr(_self, "_update_defects"):
                        _self._update_defects(defects)
                    # Nur UI zurücksetzen wenn KEINE Restaurierung läuft.
                    # Race Condition: _apply() kann über _gui_dispatch NACH batch_thread.start()
                    # in der Event-Queue landen → würde Progress-Bar und Buttons falsch setzen.
                    _batch_running = bool(_self.batch_thread and _self.batch_thread.isRunning())
                    if not _batch_running:
                        if hasattr(_self, "progress_bar"):
                            _self.progress_bar.setRange(0, 10000)
                            _self.progress_bar.setValue(10000)
                            _self.progress_bar.setFormat(t("status.defect_scan_done"))

                            def _reset_progress_if_idle():
                                if _self.batch_thread and _self.batch_thread.isRunning():
                                    return
                                _self.progress_bar.setVisible(False)
                                _self.progress_bar.setValue(0)

                            QTimer.singleShot(1500, _reset_progress_if_idle)
                        if hasattr(_self, "status_text"):
                            _self.status_text.setText(t("status.ready_to_restore"))
                            _self.status_text.setStyleSheet("color: #82B89A; font-size: 10pt;")
                        # ✔️ Defektanalyse fertig → Magic Buttons aktivieren
                        _self._set_magic_buttons_enabled(True)

                # Thread-sicher in GUI-Thread dispatchen (QTimer.singleShot
                # aus Background-Thread hat keine Event-Loop → _apply() würde nie laufen)
                _self._dispatch_to_gui(_apply)

            threading.Thread(target=_run_defect_scan_bg, daemon=True).start()
            # ── Ende Defekt-Analyse-Hintergrund ───────────────────────────

        except Exception as e:
            # Fehlerfall: Buttons sofort wieder freischalten (Defektanalyse startet nicht)
            for _btn_name in ("btn_magic_restoration", "btn_magic_studio"):
                if hasattr(self, _btn_name):
                    _btn = getattr(self, _btn_name)
                    _btn.setEnabled(True)
                    _btn.update()
            if hasattr(self, "progress_bar"):
                self.progress_bar.setRange(0, 10000)
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            self.detected_medium_label.setText(t("status.load_error_short", error=str(e)[:80]))
            self.detected_medium_label.setStyleSheet("""
                color: #B87A7A; font-size: 11pt; padding: 12px;
                background: rgba(148, 82, 82, 0.10);
                border-radius: 8px; border: 2px solid rgba(152, 88, 88, 0.26);
            """)
        self.title_bar.set_status(t("status.file_loaded"), "#82B89A")
        self.status_text.setText(t("status.loaded_and_analyzing", file=Path(file_path).name))
        # ── Ende _on_file_loaded ───────────────────────────────────────────

    def _auto_preview_restored(self) -> None:
        """Auto-play first 5 s of restored audio shortly after completion (Feature 8)."""
        if getattr(self, "_rest_audio", None) is None:
            return
        if not _SD_AVAILABLE:
            return
        _already = hasattr(self, "_play_thread") and self._play_thread is not None and self._play_thread.is_alive()
        if _already:
            return
        _ra = self._rest_audio
        assert _ra is not None  # guarded by check above
        _preview = np.ascontiguousarray(_ra[: min(len(_ra), 5 * 48000)], dtype=np.float32)
        if _preview.size > 0:
            self._play_audio(_preview, 48000)

    def _play_audio(self, audio: np.ndarray, sr: int):
        """Audiodaten asynchron über sounddevice abspielen."""
        if not _SD_AVAILABLE:
            QMessageBox.information(self, t("dialog.player_title"), t("dialog.player_body"))
            return
        prepared_audio = np.ascontiguousarray(_normalize_audio(audio), dtype=np.float32)
        if prepared_audio.size == 0:
            if hasattr(self, "status_text"):
                self.status_text.setStyleSheet("color: #B87A7A; font-size: 10pt;")
                self.status_text.setText(
                    "⚠ Wiedergabe nicht möglich: Die Audiodatei enthält keine abspielbaren Samples."
                )
            if hasattr(self, "title_bar"):
                self.title_bar.set_status("Wiedergabe-Fehler", "#B87A7A")
            return

        # Thread-sichere Wiedergabe: Lock verhindert Race-Condition bei stop/play/wait
        if not hasattr(self, "_sd_lock"):
            self._sd_lock = threading.Lock()

        # Laufende Wiedergabe stoppen (geschützt)
        try:
            with self._sd_lock:
                if _sd is not None:
                    _sd.stop()
        except Exception as exc:
            logger.warning("A/B playback stop failed: %s", exc)

        def _play():
            try:
                data = prepared_audio
                if data.max() > 1.0 or data.min() < -1.0:
                    data = data / (np.abs(data).max() + 1e-9)
                with self._sd_lock:
                    if _sd is not None:
                        _sd.play(data, samplerate=int(sr))
                # wait() outside lock so stop() from main thread can interrupt
                if _sd is not None:
                    _sd.wait()
            except Exception as exc:
                logger.warning("A/B playback failed: %s", exc)

                def _notify_playback_error(_exc: Exception = exc) -> None:
                    err = str(_exc)[:140]
                    if hasattr(self, "status_text"):
                        self.status_text.setStyleSheet("color: #B87A7A; font-size: 10pt;")
                        self.status_text.setText(
                            f"⚠ Wiedergabe fehlgeschlagen: {err}. Hinweis: Audio-Ausgabegerät prüfen."
                        )
                    if hasattr(self, "title_bar"):
                        self.title_bar.set_status("Wiedergabe-Fehler", "#B87A7A")

                self._dispatch_to_gui(_notify_playback_error)

        self._play_thread = threading.Thread(target=_play, daemon=True)
        self._play_thread.start()
        if hasattr(self, "btn_stop_playback"):
            self.btn_stop_playback.setEnabled(True)

        # ── Playhead-Timer (50 ms-Takt, läuft im Hauptthread) ────────────────
        self._playback_start_time = time.monotonic()
        self._playback_audio_duration = len(audio) / max(1, sr)
        if not hasattr(self, "_playhead_timer"):
            self._playhead_timer = QTimer(self)
            self._playhead_timer.timeout.connect(self._update_playhead)
        self._playhead_timer.start(50)

    def _update_playhead(self) -> None:
        """50-ms-Timer-Slot: aktualisiert Playhead-Position und Zeitanzeige im Hauptthread."""
        _thread_alive = hasattr(self, "_play_thread") and self._play_thread is not None and self._play_thread.is_alive()
        if not _thread_alive:
            # Wiedergabe beendet — alles zurücksetzen
            if hasattr(self, "_playhead_timer"):
                self._playhead_timer.stop()
            if hasattr(self, "waveform_widget"):
                self.waveform_widget._playhead_pos = -1.0
                self.waveform_widget.update()
            if hasattr(self, "_playback_time_label"):
                self._playback_time_label.setText("– : – –")
            if hasattr(self, "btn_stop_playback"):
                self.btn_stop_playback.setEnabled(False)
            return
        _elapsed = time.monotonic() - self._playback_start_time
        _dur = self._playback_audio_duration
        _pos = min(_elapsed / max(_dur, 1e-9), 1.0)
        if hasattr(self, "waveform_widget"):
            self.waveform_widget._playhead_pos = _pos
            self.waveform_widget.update()
        if hasattr(self, "_playback_time_label") and _dur > 0:
            _em, _es = divmod(int(_elapsed), 60)
            _dm, _ds = divmod(int(_dur), 60)
            self._playback_time_label.setText(f"{_em}:{_es:02d} / {_dm}:{_ds:02d}")

    def _update_ab_player_state(self):
        """A/B-Player Buttons je nach verfügbaren Audiodaten aktivieren."""
        if hasattr(self, "btn_play_original"):
            self.btn_play_original.setEnabled(self._orig_audio is not None)
        if hasattr(self, "btn_play_restored"):
            self.btn_play_restored.setEnabled(self._rest_audio is not None)
        # Stop-Button nur aktivierbar wenn Wiedergabe läuft — Grundzustand: deaktiviert
        if hasattr(self, "btn_stop_playback"):
            _playing = hasattr(self, "_play_thread") and self._play_thread is not None and self._play_thread.is_alive()
            self.btn_stop_playback.setEnabled(_playing)

    def _dialog_options(self, *, directory_only: bool = False) -> QFileDialog.Options:
        """Build QFileDialog options for the Qt fallback path.

        System dialogs (zenity/kdialog/tkinter) are tried first; this method
        only governs the last-resort Qt dialog, which uses native OS dialogs
        on all platforms.
        """
        opts = QFileDialog.Options()
        if directory_only:
            opts |= QFileDialog.Option.ShowDirsOnly
        return opts

    def _pick_with_system_dialog(
        self,
        *,
        title: str,
        start_dir: str,
        multiple: bool = False,
        directory: bool = False,
        save: bool = False,
        default_filename: str = "",
    ) -> list[str]:
        """Use native OS system dialogs for file/directory selection.

        Priority:
          1. Linux: zenity (GTK) or kdialog (KDE) — avoids Qt frameless-window issues
          2. All platforms: tkinter native dialog (Win32 on Windows, Tk on Linux fallback)
          Returns an empty list if the user cancels or no dialog is available.
        """
        start = start_dir or str(Path.home())
        _audio_filter = "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a *.wma *.aac"

        # ── 1. Linux: zenity (GTK native) ─────────────────────────────────
        if sys.platform.startswith("linux") and shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", f"--title={title}"]
            if directory:
                cmd.append("--directory")
            elif save:
                cmd.append("--save")
                fn = start.rstrip("/") + "/" + (default_filename or "")
                cmd.append(f"--filename={fn}")
            else:
                if multiple:
                    cmd.extend(["--multiple", "--separator=\n"])
                cmd.append(f"--filename={start.rstrip('/')}/")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0:
                    out = proc.stdout.strip()
                    if out:
                        return [p for p in out.splitlines() if p.strip()]
                    return []
            except Exception:
                pass

        # ── 2. Linux: kdialog (KDE native) ────────────────────────────────
        if sys.platform.startswith("linux") and shutil.which("kdialog"):
            if save:
                fn = f"{start}/{default_filename}" if default_filename else start
                cmd = ["kdialog", "--getsavefilename", fn, "--title", title]
            elif directory:
                cmd = ["kdialog", "--getexistingdirectory", start, "--title", title]
            elif multiple:
                cmd = [
                    "kdialog",
                    "--getopenfilename",
                    start,
                    _audio_filter,
                    "--multiple",
                    "--separate-output",
                    "--title",
                    title,
                ]
            else:
                cmd = ["kdialog", "--getopenfilename", start, _audio_filter, "--title", title]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0:
                    out = proc.stdout.strip()
                    if out:
                        return [p for p in out.splitlines() if p.strip()]
                    return []
            except Exception:
                pass

        # ── 3. Tkinter native dialog (Win32 on Windows, Tk fallback on Linux) ──
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd

            _root = _tk.Tk()
            _root.withdraw()
            _root.wm_attributes("-topmost", True)
            _audio_types = [
                ("Audio-Dateien", _audio_filter.replace(" ", " ")),
                ("Alle Dateien", "*"),
            ]
            try:
                if directory:
                    path = _fd.askdirectory(title=title, initialdir=start, mustexist=False, parent=_root)
                    return [path] if path else []
                elif save:
                    path = _fd.asksaveasfilename(
                        title=title,
                        initialdir=start,
                        initialfile=default_filename,
                        filetypes=[("Alle Dateien", "*")],
                        parent=_root,
                    )
                    return [path] if path else []
                elif multiple:
                    paths = _fd.askopenfilenames(title=title, initialdir=start, filetypes=_audio_types, parent=_root)
                    return list(paths) if paths else []
                else:
                    path = _fd.askopenfilename(title=title, initialdir=start, filetypes=_audio_types, parent=_root)
                    return [path] if path else []
            finally:
                with contextlib.suppress(Exception):
                    _root.destroy()
        except Exception:
            pass

        return []

    def _exec_file_dialog(self, dlg: QFileDialog) -> int:
        """Exec file dialog with explicit modality/focus to avoid hidden dialogs."""
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setWindowFlag(Qt.WindowType.Dialog, True)
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        # Dark-Theme für alle Dialog-Widgets (Combo-Box, Listen, Eingabefelder)
        dlg.setStyleSheet("""
            QFileDialog {
                background: #1a1a2e; color: #d0d4e0;
            }
            QWidget {
                background: #1a1a2e; color: #d0d4e0;
            }
            QComboBox {
                background: #1e1e36; color: #d0d4e0;
                border: 1px solid rgba(120, 140, 180, 0.35);
                border-radius: 4px; padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                background: #1e1e36; color: #d0d4e0;
                selection-background-color: rgba(102, 126, 234, 0.45);
                selection-color: #ffffff;
                border: 1px solid rgba(120, 140, 180, 0.45);
            }
            QComboBox::drop-down {
                border: none; width: 20px;
            }
            QTreeView, QListView, QTableView {
                background: #14142a; color: #d0d4e0;
                selection-background-color: rgba(102, 126, 234, 0.40);
                selection-color: #ffffff;
                border: 1px solid rgba(120, 140, 180, 0.25);
            }
            QLineEdit {
                background: #1e1e36; color: #d0d4e0;
                border: 1px solid rgba(120, 140, 180, 0.35);
                border-radius: 4px; padding: 4px 8px;
            }
            QLabel { color: #d0d4e0; background: transparent; }
            QHeaderView::section {
                background: #1a1a2e; color: #8890a8;
                border: 1px solid rgba(120, 140, 180, 0.20);
                padding: 4px;
            }
            QPushButton {
                background: rgba(102, 126, 234, 0.25); color: #d0d4e0;
                border: 1px solid rgba(102, 126, 234, 0.40);
                border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover {
                background: rgba(102, 126, 234, 0.40);
            }
            QToolButton {
                background: transparent; color: #d0d4e0;
                border: none;
            }
            QToolButton:hover {
                background: rgba(102, 126, 234, 0.25);
            }
            QSplitter::handle { background: rgba(120, 140, 180, 0.20); }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #14142a; width: 8px; height: 8px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: rgba(120, 140, 180, 0.40); border-radius: 4px;
            }
        """)
        dlg.raise_()
        dlg.activateWindow()
        return dlg.exec_()

    def _open_file(self):
        """Öffnet den Datei-Dialog und delegiert an _load_file."""
        _start_dir = getattr(self, "_last_open_dir", None) or str(
            next(
                (p for p in (Path.home() / "Music", Path.home() / "Musik", Path.home()) if p.exists()),
                Path.home(),
            )
        )
        # Prefer external Linux system dialogs to avoid broken Qt file dialogs
        # in frameless-window/compositor combinations.
        picked = self._pick_with_system_dialog(
            title=t("ui.open_audio_file"),
            start_dir=_start_dir,
            multiple=False,
            directory=False,
        )
        if picked:
            file_path = picked[0]
            self._last_open_dir = str(Path(file_path).parent)
            self._load_file(file_path)
            return

        dlg = QFileDialog(self, t("ui.open_audio_file"), _start_dir, t("ui.audio_filter"))
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setOptions(self._dialog_options())
        file_path = ""
        if self._exec_file_dialog(dlg) == QDialog.DialogCode.Accepted:
            files = dlg.selectedFiles()
            if files:
                file_path = files[0]
        if file_path:
            self._last_open_dir = str(Path(file_path).parent)
            self._load_file(file_path)

    def _batch_import(self):
        """Batch import multiple files"""
        _start_dir = getattr(self, "_last_open_dir", None) or str(
            next(
                (p for p in (Path.home() / "Music", Path.home() / "Musik", Path.home()) if p.exists()),
                Path.home(),
            )
        )
        picked = self._pick_with_system_dialog(
            title=t("ui.open_multiple_files"),
            start_dir=_start_dir,
            multiple=True,
            directory=False,
        )
        if picked:
            file_paths = picked
            self._last_open_dir = str(Path(file_paths[0]).parent)
            for path in file_paths:
                self._add_to_queue(path)
            self.title_bar.set_status(t("status.files_loaded", count=len(file_paths)), "#82B89A")
            self.status_text.setText(t("status.batch_import_files", count=len(file_paths)))
            return

        dlg = QFileDialog(self, t("ui.open_multiple_files"), _start_dir, t("ui.audio_filter_batch"))
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dlg.setOptions(self._dialog_options())
        file_paths = dlg.selectedFiles() if self._exec_file_dialog(dlg) == QDialog.DialogCode.Accepted else []
        if file_paths:
            self._last_open_dir = str(Path(file_paths[0]).parent)
        for path in file_paths:
            self._add_to_queue(path)

        if file_paths:
            self.title_bar.set_status(t("status.files_loaded", count=len(file_paths)), "#82B89A")
            self.status_text.setText(t("status.batch_import_files", count=len(file_paths)))

    def _album_import(self):
        """Ganzen Ordner / Album rekursiv importieren.

        Nutzt BatchProcessor.find_audio_files() falls verfügbar,
        sonst eigenes rglob als Fallback.
        """
        _album_start = getattr(self, "_last_open_dir", None) or str(
            next(
                (p for p in (Path.home() / "Music", Path.home() / "Musik", Path.home()) if p.exists()),
                Path.home(),
            )
        )
        picked = self._pick_with_system_dialog(
            title=t("ui.album_select_dir"),
            start_dir=_album_start,
            multiple=False,
            directory=True,
        )
        folder = picked[0] if picked else ""
        dlg = QFileDialog(self, t("ui.album_select_dir"), _album_start)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOptions(self._dialog_options(directory_only=True))
        if not folder and self._exec_file_dialog(dlg) == QDialog.DialogCode.Accepted:
            files = dlg.selectedFiles()
            if files:
                folder = files[0]
        if not folder:
            return

        folder_path = Path(folder)

        # Audio-Dateien finden — BatchProcessor.find_audio_files nutzen wenn möglich
        try:
            import tempfile

            from batch_processor import BatchProcessor

            bp = BatchProcessor(output_dir=Path(tempfile.gettempdir()))
            found = bp.find_audio_files([folder])
        except Exception:
            # Fallback: eigenes rekursives Suchen
            AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".m4a", ".wma"}
            found = [p for ext in AUDIO_EXTS for p in folder_path.rglob(f"*{ext}")]
            found = list(set(found))

        if not found:
            QMessageBox.information(
                self,
                t("dialog.album_import_title"),
                t("dialog.album_import_no_files", folder=folder_path.name),
            )
            return

        # Kurze Vorschau – Nutzer bestätigen lassen
        # Dateien nach Unterordner (= CD/LP-Seite) gruppiert anzeigen
        subdirs = sorted({p.parent.name for p in found})
        subdir_info = ", ".join(subdirs[:5])
        if len(subdirs) > 5:
            subdir_info += f" … (+{len(subdirs) - 5} weitere)"

        reply = QMessageBox.question(
            self,
            t("ui.album_import_confirm_title"),
            t("ui.album_import_confirm_body", count=len(found), subdirs=subdir_info or "–"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Modus abfragen
        mode_dialog = QDialog(self)
        mode_dialog.setWindowTitle(t("ui.album_mode_title"))
        mode_dialog.setMinimumWidth(380)
        d_layout = QVBoxLayout(mode_dialog)
        d_layout.setSpacing(15)
        d_layout.addWidget(QLabel(t("ui.album_mode_question")))

        btn_group = QButtonGroup(mode_dialog)
        rb_rest = QRadioButton(t("ui.album_mode_restoration"))
        rb_rest.setChecked(True)
        rb_studio = QRadioButton(t("ui.album_mode_studio"))
        btn_group.addButton(rb_rest)
        btn_group.addButton(rb_studio)
        d_layout.addWidget(rb_rest)
        d_layout.addWidget(rb_studio)

        buttons = QDialogButtonBox(parent=mode_dialog)
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(mode_dialog.accept)
        buttons.rejected.connect(mode_dialog.reject)
        d_layout.addWidget(buttons)

        if mode_dialog.exec_() != QDialog.DialogCode.Accepted:
            return

        chosen_mode = "RESTORATION" if rb_rest.isChecked() else "STUDIO_2026"

        # Alle Dateien zur Queue (sortiert nach Pfad = Track-Reihenfolge)
        found_sorted = sorted(found)
        added = 0
        for p in found_sorted:
            try:
                self._add_to_queue_with_mode(str(p), chosen_mode)
                added += 1
            except Exception:
                pass

        self.title_bar.set_status(t("status.album_tracks_loaded", count=added), "#82B89A")
        self.status_text.setText(t("status.album_import_ready", album=folder_path.name, count=added))

    def _add_to_queue(self, file_path):
        """Add file to processing queue with RESTORATION mode (batch-import Pfad).

        MaterialType wird von V3/AurikDenker automatisch erkannt — kein Pre-Scan nötig.
        """
        # Settings for V3: MaterialType is auto-detected by DefectScanner
        # Only need to specify the processing mode (default RESTORATION for batch)
        settings = {"mode": "RESTORATION"}  # Default mode for batch operations

        # Generate output filename — P1: immer in output/-Unterordner (Projektgrenze)
        input_path = Path(file_path)
        _out_dir = getattr(self, "_output_dir", None) or (input_path.parent / "output")
        _out_dir = Path(_out_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(_out_dir / f"{input_path.stem}_restored{input_path.suffix}")

        # Add to queue
        item = self.batch_queue.add_item(file_path, output_file, settings)

        # Add to list widget
        if hasattr(self, "queue_list"):
            list_item = QListWidgetItem(f"📄 {input_path.name}")
            list_item.setData(Qt.ItemDataRole.UserRole, item.id)
            self.queue_list.addItem(list_item)

        self._update_stats()

    def _add_to_queue_with_mode(self, file_path, mode):
        """Add file to processing queue with specified mode

        Args:
            file_path: Path to audio file
            mode: Processing mode ("RESTORATION" or "STUDIO_2026")
        """
        # Raw MaterialType key gespeichert von _carrier_bg (z. B. "shellac", "vinyl", "tape").
        # V3/AurikDenker erkennt MaterialType eigenständig — dieser Wert dient nur als
        # optionaler GP-Warmstart-Hint (z. B. decade ≤ 1940 → NR ~ N(0.90, 0.05)).
        detected_medium = getattr(self, "_raw_medium_type", "AUTO_DETECT") or "AUTO_DETECT"

        # Settings for V3: MaterialType is auto-detected by DefectScanner
        # We only need to specify the processing mode and medium hint
        settings = {"mode": mode, "medium_hint": detected_medium}  # RESTORATION or STUDIO_2026

        # Generate output filename — P1: immer in output/-Unterordner (Projektgrenze)
        input_path = Path(file_path)
        mode_suffix = "_restored" if mode == "RESTORATION" else "_studio2026"
        _out_dir = getattr(self, "_output_dir", None) or (input_path.parent / "output")
        _out_dir = Path(_out_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(_out_dir / f"{input_path.stem}{mode_suffix}{input_path.suffix}")

        # Add to queue
        item = self.batch_queue.add_item(file_path, output_file, settings)

        # Add to list widget (if it exists - might not be visible in new UI)
        if hasattr(self, "queue_list"):
            mode_icon = "💿" if mode == "RESTORATION" else "🎯"
            list_item = QListWidgetItem(f"{mode_icon} {input_path.name}")
            list_item.setData(Qt.ItemDataRole.UserRole, item.id)
            self.queue_list.addItem(list_item)

        self._update_stats()

    def _start_processing(self):
        """Start audio processing"""
        queue_len = self.queue_list.count() if hasattr(self, "queue_list") else self.batch_queue.get_stats()["pending"]
        if queue_len == 0:
            QMessageBox.warning(self, t("dialog.no_files_title"), t("dialog.no_files_body"))
            return

        if self.batch_thread and self.batch_thread.isRunning():
            QMessageBox.warning(self, t("dialog.processing_running_title"), t("dialog.processing_running_body"))
            return

        # Abgeschlossene/fehlgeschlagene Einträge vom vorherigen Lauf bereinigen.
        # Hier (Batch-Start) statt in _on_all_finished, damit Ctrl+S-Export nach
        # Abschluss noch funktioniert, bevor ein neuer Lauf gestartet wird.
        self.batch_queue.clear_completed()
        if hasattr(self, "queue_list"):
            for i in range(self.queue_list.count() - 1, -1, -1):
                list_item = self.queue_list.item(i)
                item_id = list_item.data(Qt.ItemDataRole.UserRole)
                item = self.batch_queue.get_item(item_id)
                if item is None:  # wurde durch clear_completed() entfernt
                    self.queue_list.takeItem(i)

        stats = self.batch_queue.get_stats()
        if stats["pending"] == 0:
            QMessageBox.information(self, t("dialog.no_pending_title"), t("dialog.no_pending_body"))
            return

        # Disable process button and Magic Buttons during processing
        if hasattr(self, "btn_process"):
            self.btn_process.setEnabled(False)
        if hasattr(self, "btn_magic_restoration"):
            self.btn_magic_restoration.setEnabled(False)
        if hasattr(self, "btn_magic_studio"):
            self.btn_magic_studio.setEnabled(False)

        # Update status
        self.title_bar.set_status(t("status.processing_running"), "#B8A068")
        self.status_text.setStyleSheet("color: #7B93B8; font-size: 10pt; background: transparent;")
        _n_pend = stats["pending"]
        self.status_text.setText(t("status.processing_files", count=_n_pend))

        # Heartbeat-Timer starten
        self._heartbeat_dots = 0
        if not hasattr(self, "_heartbeat_timer"):
            self._heartbeat_timer = QTimer(self)
            self._heartbeat_timer.timeout.connect(self._tick_heartbeat)
        if not self._heartbeat_timer.isActive():
            self._heartbeat_timer.start(500)

        # Start batch processing
        self.batch_thread = BatchProcessingThread(self.batch_queue)
        self.batch_thread.item_started.connect(self._on_item_started)
        self.batch_thread.item_progress.connect(self._on_item_progress)
        self.batch_thread.item_finished.connect(self._on_item_finished)
        self.batch_thread.item_finished_with_result.connect(self._on_item_finished_with_result)
        self.batch_thread.item_error.connect(self._on_item_error)
        self.batch_thread.all_finished.connect(self._on_all_finished)

        # Connect visualization signals
        self.batch_thread.waveform_data.connect(self._update_waveform)
        self.batch_thread.waveform_phase_update.connect(self._update_waveform_live)
        self.batch_thread.defect_update.connect(self._update_defects)
        self.batch_thread.phase_update.connect(self._update_phase)
        # Connect resource/mode signals
        self.batch_thread.mode_update.connect(self._update_mode)
        self.batch_thread.ml_status_update.connect(self._update_ml_status)
        # Enhanced real-time UX feedback signals (§11.4)
        if hasattr(self, "phase_progress_bar"):
            self.batch_thread.phase_progress.connect(lambda v: self.phase_progress_bar.setValue(v * 100))
        self.batch_thread.scan_progress.connect(self._on_scan_progress)
        if hasattr(self, "quality_meter_widget"):
            self.batch_thread.quality_update.connect(self.quality_meter_widget.set_mos)
        self.batch_thread.phase_step_update.connect(self._on_phase_step_update)

        # RAM-Sicherheitscheck: mindestens 6 GB verfügbarer Arbeitsspeicher erforderlich.
        # Verhindert OOM-Kills, die das gesamte System einfrieren (Swap nur 2 GB).
        try:
            import psutil as _psutil

            _avail_gb = _psutil.virtual_memory().available / 1024**3
            if _avail_gb < 6.0:
                QMessageBox.critical(
                    self,
                    t("dialog.low_ram_title"),
                    t("dialog.low_ram_body", avail=f"{_avail_gb:.1f}"),
                )
                self._set_magic_buttons_enabled(True)
                self.progress_bar.setVisible(False)
                self.batch_thread = None
                return
            logger.info("RAM-Check vor Restaurierung: %.1f GB verfügbar → OK", _avail_gb)
        except Exception:
            pass  # psutil nicht verfügbar → kein Check, weiterfahren

        self.progress_bar.setRange(0, 10000)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        if hasattr(self, "phase_progress_bar"):
            self.phase_progress_bar.setValue(0)
            self.phase_progress_bar.setVisible(True)
        if hasattr(self, "_phase_step_label"):
            self._phase_step_label.setText("")
            self._phase_step_label.setVisible(False)
        if hasattr(self, "quality_meter_widget"):
            self.quality_meter_widget.set_mos(2.5)

        # Watchdog-Timer: feuert wenn Verarbeitung zu lange hängt (z. B. blockierender ONNX-Call).
        # Budget muss großzügiger sein als PerformanceGuard (LIMIT_MAXIMUM=32×, plus
        # Post-Pipeline-Analytik, FeedbackChain 5×5 Iterationen, PMGG-Retries).
        # Formel: 32× Audiodauer + 30 Min Overhead, Minimum 90 Min (= MAX_ABSOLUTE_SECONDS).
        _audio_dur_s = 0.0
        _ww = getattr(self, "waveform_widget", None)
        if _ww is not None and getattr(_ww, "audio_data", None) is not None:
            _sr = max(1, getattr(_ww, "sample_rate", 48000))
            _audio_dur_s = _ww.audio_data.shape[0] / _sr
        _per_file_ms = max(5_400_000, int(_audio_dur_s * 32_000) + 1_800_000)
        _watchdog_ms = max(5_400_000, stats["pending"] * _per_file_ms)
        if not hasattr(self, "_watchdog_timer"):
            self._watchdog_timer = QTimer(self)
            self._watchdog_timer.setSingleShot(True)
            self._watchdog_timer.timeout.connect(self._on_watchdog_timeout)
        self._watchdog_timer.start(_watchdog_ms)
        logger.info(
            "Watchdog-Timer gestartet: %.0f s (Audio %.0f s, %d Dateien)",
            _watchdog_ms / 1000,
            _audio_dur_s,
            stats["pending"],
        )

        assert self.batch_thread is not None
        self.batch_thread.start()

    def _on_watchdog_timeout(self):
        """Watchdog feuert: Verarbeitung hat das Timeout überschritten — Thread wird zwangsbeendet."""
        if not (self.batch_thread and self.batch_thread.isRunning()):
            return  # normaler Abschluss — kein Handlungsbedarf
        logger.error("Watchdog ausgelöst: Verarbeitung hat Timeout überschritten — Thread wird beendet.")
        self.batch_thread.requestInterruption()
        self.batch_thread.wait(3000)
        if self.batch_thread.isRunning():
            self.batch_thread.terminate()
            self.batch_thread.wait(2000)
        if hasattr(self, "_heartbeat_timer") and self._heartbeat_timer.isActive():
            self._heartbeat_timer.stop()
        self._set_magic_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.title_bar.set_status(t("dialog.timeout_title"), "#B87A7A")
        _msg = (
            "⏰ Die Verarbeitung hat das Zeitlimit überschritten und wurde abgebrochen.\n"
            "Ursache: Ein Verarbeitungsschritt hat nicht reagiert (möglicher ONNX-Deadlock).\n"
            "→ Starten Sie Aurik neu und versuchen Sie es mit einer kürzeren Audiodatei."
        )
        if hasattr(self, "detected_medium_label"):
            self.detected_medium_label.setText(_msg)
        self.status_text.setText(t("dialog.timeout_title") + " — " + t("status.cancelled"))
        QMessageBox.warning(
            self,
            t("dialog.timeout_title"),
            t("dialog.timeout_body"),
        )

    def _tick_heartbeat(self):
        """Animierter Spinner + Progress-Polling alle 500 ms.

        Fallback für den Fall, dass item_progress-Signale aus irgendeinem Grund
        nicht zugestellt werden: wir lesen item.progress direkt aus der Queue.
        Außerdem: Zeitanzeige in status_text live herunterzählen, damit der Nutzer
        sieht, dass der Prozess läuft — auch wenn kein neuer progress_callback kommt.
        """
        self._heartbeat_dots = (self._heartbeat_dots + 1) % 4
        spinners = ["◐", "◓", "◑", "◒"]
        spin = spinners[self._heartbeat_dots]
        self.title_bar.set_status(t("status.processing_running_spinner", spin=spin), "#B8A068")

        # Progress-Bar über Queue-Status pollen — unabhängig von Signals.
        if self.batch_thread and self.batch_thread.isRunning():
            current_item = next(
                (i for i in self.batch_queue.items if i.status == "processing"),
                None,
            )
            if current_item is not None:
                polled = max(100, min(10000, current_item.progress * 100))
                # Nur aktualisieren wenn polled-Wert größer als aktueller Wert
                # (verhindert Rückschritt durch Race Condition)
                if polled > self.progress_bar.value():
                    self.progress_bar.setRange(0, 10000)
                    self.progress_bar.setValue(polled)
                    self.progress_bar.setVisible(True)

            # Zeitanzeige live herunterzählen — unabhängig vom Backend-Callback-Takt.
            _state = getattr(self.batch_thread, "_last_phase_state", None)
            if _state and hasattr(self, "status_text"):
                _elapsed = _state["elapsed_s"] + (time.perf_counter() - _state["wall_time"])
                _pct = _state["pct"]
                _base = _state["base"]
                if _elapsed >= 2.0:
                    _el = f"{int(_elapsed)}s" if _elapsed < 60 else f"{int(_elapsed // 60)}m{int(_elapsed % 60):02d}s"
                    if _pct >= 5:
                        # ETA wurde beim letzten Callback berechnet — hier nur runterzählen.
                        _remaining_at_cb = _state.get("remaining_s", -1.0)
                        _eta_range = _state.get("eta_range", "")
                        if _remaining_at_cb >= 0:
                            _time_since_cb = time.perf_counter() - _state["wall_time"]
                            _rem = max(0.0, _remaining_at_cb - _time_since_cb)
                            _eta_str = f"~{int(_rem // 60)}m{int(_rem % 60):02d}s" if _rem >= 60 else f"~{int(_rem)}s"
                            if _pct < 19 and _eta_range:
                                _full = f"{_base}  ·  {_el} · {_eta_range}  –  noch {_eta_str}"
                            else:
                                _full = f"{_base}  ·  {_el} · noch {_eta_str}"
                        else:
                            _full = f"{_base}  ·  {_el}"
                    else:
                        _full = f"{_base}  ·  {_el}"
                    self.status_text.setText(f"⚙️ {_full}")
                    self.status_text.setStyleSheet(
                        "color: #E8C060; font-size: 11pt; font-weight: 600;"
                        " background: rgba(200,160,40,0.07); border-radius: 6px;"
                        " padding: 2px 8px;"
                    )

    def _on_item_started(self, item_id):
        """Handle item processing start"""
        item = self.batch_queue.get_item(item_id)
        if item:
            logger.info("Verarbeitung gestartet: %s (id=%s)", Path(item.input_file).name, item_id)
            self.status_text.setText(t("status.processing_item", file=Path(item.input_file).name))

    def _on_item_progress(self, item_id, progress):
        """Handle item progress update — obere Bar zeigt GESAMT-Batch-Fortschritt."""
        # setRange sicherstellen: verhindert Marquee-Modus (range 0-0) der
        # QProgressBar, der setValue() wirkungslos macht.
        self.progress_bar.setRange(0, 10000)
        # Gesamtfortschritt: (abgeschlossene Dateien + Anteil der aktuellen Datei) / Gesamt
        stats = self.batch_queue.get_stats()
        _total = max(1, stats.get("total", 1))
        _done = stats.get("completed", 0) + stats.get("failed", 0)
        _overall_pct = (_done + progress / 100.0) / _total * 100.0
        val = max(100, min(10000, int(_overall_pct * 100)))
        self.progress_bar.setValue(val)
        self.progress_bar.setVisible(True)
        logger.debug("[progress] item=%s pct=%d overall=%.1f%% → bar=%d", item_id, progress, _overall_pct, val)

        # Waveform stage overlay: monotonen Gesamt-Fortschritt weiterleiten
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_stage_progress(progress / 100.0)

        # Update list item
        if hasattr(self, "queue_list"):
            for i in range(self.queue_list.count()):
                list_item = self.queue_list.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                    item = self.batch_queue.get_item(item_id)
                    if item:
                        list_item.setText(f"⏳ {Path(item.input_file).name} ({progress}%)")
                    break

    def _on_item_finished(self, item_id):
        """Handle item completion — Queue-Update + Stats.

        Qualitäts-Radar wird ausschließlich in _on_item_finished_with_result
        aktualisiert (folgt immer direkt danach bei Erfolg). Kein Doppel-Aufruf
        von _compute_and_show_quality hier.
        """
        if hasattr(self, "queue_list"):
            for i in range(self.queue_list.count()):
                list_item = self.queue_list.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                    item = self.batch_queue.get_item(item_id)
                    if item:
                        list_item.setText(f"✅ {Path(item.input_file).name}")
                    break

        # Export-Bestätigung: Ausgabepfad kurz in status_text anzeigen (5 s Auto-Clear)
        item = self.batch_queue.get_item(item_id)
        if item and item.output_file and hasattr(self, "status_text"):
            _out_name = Path(item.output_file).name
            self.status_text.setStyleSheet("color: #82B89A; font-size: 10pt;")
            self.status_text.setText(f"✅ Gespeichert: {_out_name}")
            QTimer.singleShot(
                5000,
                lambda: self.status_text.setText("") if self.status_text.text().startswith("✅ Gespeichert") else None,
            )

        self._update_stats()

    def _on_item_finished_with_result(self, item_id, restoration_result):
        """Handle item completion mit RestorationResult — aktualisiert Qualitäts-Radar."""
        # Sofort aus AurikErgebnis: A/B-Player aktivieren und Musical Goals anzeigen,
        # unabhängig vom Dateisystem-Zugriff (kein sf.read-Risiko hier).
        if restoration_result is not None and hasattr(restoration_result, "audio"):
            _ra = _normalize_audio(restoration_result.audio)
            if isinstance(_ra, np.ndarray) and _ra.size > 0:
                self._rest_audio = _ra
                self._rest_sr = 48000  # Aurik interne SR immer 48 kHz
                QTimer.singleShot(0, self._update_ab_player_state)
                # Feature 8: Auto-Vorschau — kurze Hörprobe sofort nach Fertigstellung
                QTimer.singleShot(1400, self._auto_preview_restored)
        # Musical Goals direkt aus AurikErgebnis im Radar anzeigen (kein sf.read nötig)
        if restoration_result is not None:
            _goals = getattr(restoration_result, "musical_goals", None)
            # Fallback: Scores aus metadata["musical_goals"]["scores"] wenn .musical_goals None
            if not (isinstance(_goals, dict) and _goals):
                _meta_mg = (getattr(restoration_result, "metadata", {}) or {}).get("musical_goals") or {}
                _meta_scores = _meta_mg.get("scores") or {}
                if isinstance(_meta_scores, dict) and _meta_scores:
                    _goals = _meta_scores
            if isinstance(_goals, dict) and _goals:
                _at = getattr(restoration_result, "adaptive_thresholds", None)
                _adaptive = _at if isinstance(_at, dict) else {}

                def _show_goals(_g=dict(_goals), _ath=dict(_adaptive)):
                    if self.radar_widget is not None:
                        try:
                            self.radar_widget.update_scores(
                                scores=_g,
                                adaptive_thresholds=_ath if _ath else None,
                            )
                        except Exception as _rw_exc:
                            logger.debug("radar_widget.update_scores fehlgeschlagen: %s", _rw_exc)

                QTimer.singleShot(0, _show_goals)
        # ── Tonträgerkette aus TontraegerketteDenker im Label anzeigen ────────
        # Die _carrier_bg-Voranalyse zeigt nur MediumClassifier (Primärträger +
        # Dateiendung). Der TontraegerketteDenker (läuft im Batch) kennt die
        # vollständige Kette — z.B. "Vinyl → Magnetband → MP3".
        # Falls der Ursprungsträger im aktuellen Label fehlt, Kette ergänzen.
        try:
            _chain_info = getattr(restoration_result, "chain_info", None)
            if isinstance(_chain_info, dict) and hasattr(self, "detected_medium_label"):
                _chain_string = str(_chain_info.get("chain_string", ""))
                _orig_medium = str(_chain_info.get("original_medium", "")).lower()
                _gen_count = int(_chain_info.get("generation_count", 1))
                # Analoge Quellträger die selten korrekt via _carrier_bg erkannt werden
                _ANALOG_SOURCES = {"vinyl", "shellac", "lacquer_disc", "wax_cylinder", "wire_recording"}
                _is_multi_analog = _gen_count >= 2 and _orig_medium in _ANALOG_SOURCES
                if _chain_string and _is_multi_analog:
                    _cur_text = self.detected_medium_label.text()
                    # Nur aktualisieren wenn die Kettendarstellung neue Info enthält
                    _chain_already_shown = _chain_string.split(" →")[0].lower() in _cur_text.lower()
                    if not _chain_already_shown:
                        _new_text = _cur_text + f"<br><small style='color:#8ab4d4;'>🔗&nbsp;{_chain_string}</small>"
                        self.detected_medium_label.setText(_new_text)
                        logger.debug("detected_medium_label: Kettenanalyse ergänzt — %s", _chain_string)
                # Vollständige Kette immer als Tooltip-Info
                if _chain_string:
                    _glieder = _chain_info.get("glieder", [])
                    _tip_lines = [f"<b>Tonträgerkette:</b> {_chain_string}"]
                    for _gl in _glieder:
                        if isinstance(_gl, dict):
                            _tip_lines.append(
                                f"&nbsp;• {_gl.get('label', _gl.get('medium', '?'))}: {_gl.get('degradation_type', '')}"
                            )
                    self.detected_medium_label.setToolTip("<br>".join(_tip_lines))
        except Exception as _chain_upd_exc:
            logger.debug("Kettenanzeige-Update fehlgeschlagen: %s", _chain_upd_exc)
        # ── Ära-Badge aus UV3-Ergebnis aktualisieren (autoritativ) ────────────
        # Die Voranalyse (_detect_era_genre_bg) kann abweichende Dekaden liefern,
        # da sie nur DSP-Heuristiken nutzt. UV3 hat GlobalPlan-Prior + vollständige
        # Analyse-Kette → era_decade im RestorationResult ist die verlässliche Quelle.
        try:
            _final_decade = getattr(restoration_result, "era_decade", None) if restoration_result is not None else None
            if _final_decade is not None and hasattr(self, "detected_medium_label"):
                _new_decade_label = f"{_final_decade}er"
                _old_badge = getattr(self, "_era_genre_badge", "")
                # Nur aktualisieren wenn sich die Dekade tatsächlich geändert hat
                if _new_decade_label not in _old_badge:
                    # Genre-Teil aus bestehendem Badge beibehalten
                    _genre_part = ""
                    if _old_badge and "·" in _old_badge:
                        _genre_part = _old_badge.split("·", 1)[1].strip()
                    _new_badge = f"  │  ◷ {_new_decade_label}"
                    if _genre_part:
                        _new_badge += f" · {_genre_part}"
                    self._era_genre_badge = _new_badge
                    # Carrier-Label neu aufbauen
                    _cur_lbl = getattr(self, "_carrier_bg_label", "")
                    _cur_sc = getattr(self, "_carrier_bg_score", 0)
                    _cur_stars = self._render_star_html(_cur_sc)
                    self.detected_medium_label.setText(f"{_cur_lbl}   {_cur_stars}{_new_badge}")
                    logger.info(
                        "Era-Badge aus UV3-Ergebnis aktualisiert: %s → %s",
                        _old_badge.strip(),
                        _new_badge.strip(),
                    )
        except Exception as _era_upd_exc:
            logger.debug("Era-Badge-Update aus Ergebnis fehlgeschlagen: %s", _era_upd_exc)
        # ──────────────────────────────────────────────────────────────────────
        item = self.batch_queue.get_item(item_id)
        if item and item.output_file and Path(item.output_file).exists():
            self._compute_and_show_quality(item.output_file, restoration_result=restoration_result)

    def _on_item_error(self, item_id, error_msg):
        """Handle item error — zeigt deutsche Fehlermeldung im UI (Spec §11.4)."""
        item = self.batch_queue.get_item(item_id)
        file_name = Path(item.input_file).name if item else t("status.unknown_file")

        # Update list item
        if hasattr(self, "queue_list"):
            for i in range(self.queue_list.count()):
                list_item = self.queue_list.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                    list_item.setText(f"❌ {file_name}")
                    break

        # Deutsche Fehlermeldung mit Ursache + Lösungshinweis (Spec: Fehlermeldungskonvention).
        # Bewusst NICHT in detected_medium_label: der erkannte Tonträger bleibt sichtbar.
        # Fehlermeldung erscheint im defect_summary_label (hat Wordwrap + ausreichend Platz)
        # und als Kurztext in status_text.
        _cause = str(error_msg)[:200] if error_msg else t("status.unknown_error")
        _file_label = f"\u201e{file_name}\u201c"
        if hasattr(self, "defect_summary_label"):
            self.defect_summary_label.setText(t("status.processing_error_detail", file=_file_label, cause=_cause))
            self.defect_summary_label.setStyleSheet("""
                color: #B87A7A; font-size: 9pt; padding: 12px;
                background: rgba(148, 82, 82, 0.10);
                border-radius: 10px; border: 1px solid rgba(152, 88, 88, 0.32);
            """)
        if hasattr(self, "status_text"):
            self.status_text.setStyleSheet("color: #B87A7A; font-size: 10pt;")
            self.status_text.setText(t("status.processing_error_short", file=file_name))
        logger.warning("Item-Fehler %s: %s", item_id, error_msg)

        self._update_stats()

    def _on_all_finished(self):
        """Handle all items finished"""
        _stats = self.batch_queue.get_stats() if hasattr(self, "batch_queue") else {}
        logger.info(
            "Batch abgeschlossen: %d erfolgreich, %d fehlgeschlagen, %d gesamt",
            _stats.get("completed", 0),
            _stats.get("failed", 0),
            _stats.get("total", 0),
        )
        if hasattr(self, "_watchdog_timer") and self._watchdog_timer.isActive():
            self._watchdog_timer.stop()
        if hasattr(self, "_heartbeat_timer") and self._heartbeat_timer.isActive():
            self._heartbeat_timer.stop()
        # End any active inpainting zoom tour and restore full overview
        self._end_inpainting_zoom(restore=True)
        # Phase-Overlay ausblenden — mit Fade-Out
        if hasattr(self, "_phase_overlay_label"):
            _ov = self._phase_overlay_label
            if _ov.isVisible():
                try:
                    from PyQt5.QtWidgets import QGraphicsOpacityEffect

                    _eff = getattr(_ov, "_opacity_eff", None)
                    if _eff is None:
                        _eff = QGraphicsOpacityEffect(_ov)
                        _ov.setGraphicsEffect(_eff)
                        _ov._opacity_eff = _eff
                    _fa = QPropertyAnimation(_eff, b"opacity", _ov)
                    _fa.setDuration(350)
                    _fa.setStartValue(1.0)
                    _fa.setEndValue(0.0)
                    _fa.setEasingCurve(QEasingCurve.Type.InCubic)
                    _fa.finished.connect(lambda: _ov.setVisible(False))
                    _fa.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
                    _ov._fade_anim = _fa
                except Exception:
                    _ov.setVisible(False)
            else:
                _ov.setVisible(False)
        # Stats VOR clear_completed() lesen — danach ist der Zähler 0!
        stats = self.batch_queue.get_stats()
        self._set_magic_buttons_enabled(True)
        self.progress_bar.setValue(10000)  # 100 % = 10000 Einheiten (0.01 %/Einheit)
        if hasattr(self, "phase_progress_bar"):
            self.phase_progress_bar.setValue(10000)
            self.phase_progress_bar.setVisible(False)
        if hasattr(self, "_phase_step_label"):
            self._phase_step_label.setVisible(False)
        if hasattr(self, "resource_status_widget"):
            self.resource_status_widget.update_status(phase=None, ml_active=False, ml_plugins=[])
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_scan_pos(-1.0)
            self.waveform_widget.clear_stage()
        if hasattr(self, "btn_process"):
            self.btn_process.setEnabled(True)

        n_ok = stats["completed"]
        n_fail = stats["failed"]

        if n_fail == 0:
            self.title_bar.set_status(t("status.completed"), "#82B89A")
            self.status_text.setStyleSheet("color: #82B89A; font-size: 10pt;")
            # Completion-Icon (Trophy) inline vor Erfolgsmeldung
            _compl_icon = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources",
                "completion_success.png",
            )
            _msg = t("status.completed_success_summary", count=n_ok)
            if os.path.isfile(_compl_icon):
                self.status_text.setText(
                    f'<img src="file:///{_compl_icon}" width="24" height="24"'
                    f' style="vertical-align: middle;"> '
                    f'<span style="color:#82B89A;">{_msg}</span>'
                )
            else:
                self.status_text.setText(_msg)
            # defect_summary_label NICHT überschreiben: Defektliste bleibt sichtbar
        else:
            self.title_bar.set_status(t("status.completed_with_errors"), "#B8A068")
            self.status_text.setStyleSheet("color: #B8A068; font-size: 10pt;")
            if n_ok > 0:
                self.status_text.setText(t("status.completed_mixed_summary", ok=n_ok, failed=n_fail))
            else:
                self.status_text.setText(t("status.completed_failed_summary", failed=n_fail))

    def _stop_playback(self):
        """Laufende Wiedergabe anhalten."""
        if _SD_AVAILABLE:
            try:
                if _sd is not None:
                    _sd.stop()
            except Exception:
                pass
        if hasattr(self, "btn_stop_playback"):
            self.btn_stop_playback.setEnabled(False)
        # Playhead-Timer anhalten und Cursor zurücksetzen
        if hasattr(self, "_playhead_timer"):
            self._playhead_timer.stop()
        if hasattr(self, "waveform_widget"):
            self.waveform_widget._playhead_pos = -1.0
            self.waveform_widget.update()
        if hasattr(self, "_playback_time_label"):
            self._playback_time_label.setText("– : – –")

    def _compute_and_show_quality(self, output_path: str, restoration_result=None):
        """Qualitätsscore im Hintergrund berechnen und Radar-Chart aktualisieren."""

        def _run():
            try:
                try:
                    rest_audio, rest_sr = sf.read(output_path)
                except Exception:
                    # Fallback: Audio direkt aus RestorationResult (bereits im Speicher)
                    if restoration_result is not None and hasattr(restoration_result, "audio"):
                        rest_audio = np.asarray(restoration_result.audio, dtype=np.float32)
                        rest_sr = 48000
                    else:
                        raise
                rest_audio = _normalize_audio(rest_audio)
                self._rest_audio = rest_audio
                self._rest_sr = int(rest_sr)

                # ── Schritt 1: Korrelations-MOS (Fallback / Basisschätzung) ──
                corr = 1.0
                if self._orig_audio is not None:
                    o_mono = np.mean(self._orig_audio, axis=1) if self._orig_audio.ndim > 1 else self._orig_audio
                    r_mono = np.mean(rest_audio, axis=1) if rest_audio.ndim > 1 else rest_audio
                    min_len = min(len(o_mono), len(r_mono))
                    o_s = o_mono[:min_len].astype(np.float64)
                    r_s = r_mono[:min_len].astype(np.float64)
                    if o_s.std() > 1e-9 and r_s.std() > 1e-9:
                        corr = float(np.corrcoef(o_s, r_s)[0, 1])
                        corr = max(0.0, min(1.0, corr))

                mos_est = 1.0 + 4.0 * corr

                # ── Schritt 2: Musical Goals aus RestorationResult (wenn vorhanden) ──
                musical_goals: dict = {}
                adaptive_thresholds: dict = {}
                applicable_goals = None
                inapplicable_reasons: dict = {}
                synthesized_goals: set = set()
                adaptation_reasons: dict = {}
                phase_gate_notes: list = []
                ceiling_reached: bool = False
                era_label: str = ""
                genre_label: str = ""

                # ── Transparenz-Variablen (erweiterter Backend-Report) ──
                rt_factor: float = 0.0
                total_time_s: float = 0.0
                phases_exec_count: int = 0
                phases_skip_count: int = 0
                pipeline_confidence: float = 0.0
                restorability_grade: str = ""
                restorability_mos_min: float = 0.0
                restorability_mos_max: float = 0.0
                temporal_coh_score: float = 0.0
                emotional_arc_score: float = 0.0
                top_causal_cause: str = ""
                causal_conf: float = 0.0
                era_label_full: str = ""
                era_conf: float = 0.0
                genre_bpm: float = 0.0
                genre_key: str = ""
                genre_accordion: float = 0.0
                genre_is_schlager: bool = False
                pipeline_tier: str = ""
                pipeline_hint: str = ""
                mushra_score: float = 0.0
                mushra_grade: str = ""
                mushra_itu: str = ""
                quality_before_score: float = 0.0
                quality_after_score: float = 0.0
                quality_delta: float = 0.0
                delta_snr: float = 0.0
                feedback_retries: int = 0
                feedback_chain_score: float = 0.0
                excellence_steps: list = []
                musical_violations: list = []
                fail_reason: str = ""
                degradation_status: str = "ok"
                primary_error_code: str = ""

                if restoration_result is not None:
                    r = restoration_result
                    # Musical Goals
                    if hasattr(r, "musical_goals") and isinstance(r.musical_goals, dict):
                        musical_goals = r.musical_goals
                    # Realer PQS-MOS
                    if hasattr(r, "pqs_result") and r.pqs_result is not None:
                        if hasattr(r.pqs_result, "mos"):
                            mos_est = float(r.pqs_result.mos)
                    # Adaptive Thresholds — RestorationResult.adaptive_thresholds is a plain
                    # dict[str, float] (the resolved goal thresholds used during processing).
                    _at = getattr(r, "adaptive_thresholds", None)
                    if isinstance(_at, dict) and _at:
                        adaptive_thresholds = _at
                    elif hasattr(_at, "thresholds"):
                        adaptive_thresholds = _at.thresholds or {}  # type: ignore[union-attr]
                    # Goal Applicability — RestorationResult.goal_applicability is a plain
                    # dict[str, bool] mapping goal_key → is_applicable.
                    _ga = getattr(r, "goal_applicability", None)
                    if isinstance(_ga, dict) and _ga:
                        applicable_goals = {k for k, v in _ga.items() if v}
                    elif hasattr(_ga, "applicable"):
                        applicable_goals = set(_ga.applicable)  # type: ignore[union-attr]
                        if hasattr(_ga, "reasons"):
                            inapplicable_reasons = _ga.reasons or {}  # type: ignore[union-attr]
                    # Synthesierte Ziele (EraAuthentic ✦)
                    if hasattr(r, "genealogy") and r.genealogy is not None:
                        gen = r.genealogy
                        if hasattr(gen, "operations"):
                            for op in gen.operations:
                                if hasattr(op, "operation_type") and "synthesize" in str(op.operation_type):
                                    synthesized_goals.add("brillanz")
                    # PMGG Phase-Gate-Log
                    if hasattr(r, "phase_gate_log") and r.phase_gate_log:
                        phase_gate_notes = list(r.phase_gate_log)
                    # Physical Ceiling
                    if hasattr(r, "physical_ceiling") and r.physical_ceiling is not None:
                        pc = r.physical_ceiling
                        if hasattr(pc, "further_optimization_worthwhile"):
                            ceiling_reached = not pc.further_optimization_worthwhile
                    # Ära & Genre (Grunddaten) — era_decade ist int|None, prüfe auf not None
                    if hasattr(r, "era_decade") and r.era_decade is not None:
                        era_label = str(r.era_decade)
                    if hasattr(r, "genre_label") and r.genre_label:
                        genre_label = str(r.genre_label)

                    # ── Transparenz-Extraktion ──
                    rt_factor = float(getattr(r, "rt_factor", 0.0))
                    total_time_s = float(getattr(r, "total_time_seconds", 0.0))
                    phases_exec = getattr(r, "phases_executed", []) or []
                    phases_skip = getattr(r, "phases_skipped", []) or []
                    phases_exec_count = len(phases_exec)
                    phases_skip_count = len(phases_skip)
                    pipeline_confidence = float(getattr(r, "confidence", 0.0))

                    # Restorability Grade
                    _rest = getattr(r, "restorability", None)
                    if _rest is not None:
                        restorability_grade = str(getattr(_rest, "grade", ""))
                        _mos_range = getattr(_rest, "predicted_mos_range", None)
                        if _mos_range and len(_mos_range) >= 2:
                            restorability_mos_min = float(_mos_range[0])
                            restorability_mos_max = float(_mos_range[1])

                    # Temporal Coherence & Emotional Arc
                    _tc = getattr(r, "temporal_coherence", None)
                    if _tc is not None:
                        temporal_coh_score = float(getattr(_tc, "score", 0.0))
                    _ea = getattr(r, "emotional_arc", None)
                    if _ea is not None:
                        emotional_arc_score = float(getattr(_ea, "score", 0.0))

                    # Metadata-Extraktion (robuste dict-Zugriffe)
                    _meta = getattr(r, "metadata", {}) or {}
                    _stage_notes = getattr(r, "stage_notes", {}) or {}

                    def _first_error_code(entries) -> str:
                        if not isinstance(entries, list):
                            return ""
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            code = str(entry.get("error_code") or "").strip()
                            if code and code not in {"None", "none"}:
                                return code
                        return ""

                    primary_error_code = _first_error_code(_meta.get("fail_reasons"))
                    if not primary_error_code:
                        primary_error_code = _first_error_code(_stage_notes.get("fail_reasons"))

                    # Kausal-Analyse
                    _cp = (_meta.get("defect_analysis") or {}).get("causal_plan") or {}
                    top_causal_cause = str(_cp.get("primary_cause") or "")
                    causal_conf = float(_cp.get("confidence") or 0.0)

                    # Ära-Details
                    _era = _meta.get("era") or {}
                    era_label_full = str(_era.get("era_label") or "")
                    era_conf = float(_era.get("confidence") or 0.0)
                    if era_label_full and not era_label:
                        era_label = era_label_full

                    # Genre/Schlager-Details
                    _genre = _meta.get("genre") or {}
                    if _genre:
                        genre_is_schlager = bool(_genre.get("is_schlager", False))
                        genre_bpm = float(_genre.get("bpm") or 0.0)
                        genre_key = str(_genre.get("key") or "")
                        genre_accordion = float(_genre.get("accordion_score") or 0.0)
                        if not genre_label:
                            genre_label = str(_genre.get("genre_label") or "")

                    # Pipeline-Konfidenz
                    _pc = _meta.get("pipeline_confidence") or {}
                    pipeline_tier = str(_pc.get("tier") or "")
                    pipeline_hint = str(_pc.get("user_hint") or "")

                    # MUSHRA
                    _mushra = _meta.get("mushra") or {}
                    if _mushra:
                        mushra_score = float(_mushra.get("mushra_score") or 0.0)
                        mushra_grade = str(_mushra.get("grade") or "")
                        mushra_itu = str(_mushra.get("itu_grade") or "")

                    # Qualitätsverbesserung
                    _qi = _meta.get("quality_improvement") or {}
                    _qbef = _qi.get("before") or {}
                    _qaft = _qi.get("after") or {}
                    if _qbef and _qaft:
                        quality_before_score = float(_qbef.get("overall_score") or 0.0)
                        quality_after_score = float(_qaft.get("overall_score") or 0.0)
                        quality_delta = float(_qi.get("delta_score") or 0.0)
                        delta_snr = float(_qi.get("delta_snr_db") or 0.0)

                    # Feedback-Chain
                    _fc = _meta.get("feedback_chain") or {}
                    if _fc:
                        feedback_retries = int(_fc.get("total_retries") or 0)
                        feedback_chain_score = float(_fc.get("overall_score") or 0.0)

                    # Excellence-Optimizer
                    _exc = _meta.get("excellence_optimizer") or {}
                    if _exc:
                        excellence_steps = list(_exc.get("applied_steps") or [])

                    # Musical-Goals-Verletzungen
                    _mg_meta = _meta.get("musical_goals") or {}
                    musical_violations = list(_mg_meta.get("violations") or [])
                    # Fail/degradation mapping prefers typed result fields.
                    fail_reason = _bridge_resolve_pipeline_fail_reason(
                        typed_fail_reason=getattr(r, "fail_reason", None),
                        metadata=_meta,
                        stage_notes=_stage_notes,
                        fail_reasons=_meta.get("fail_reasons") or _stage_notes.get("fail_reasons"),
                    )
                    degradation_status = _bridge_normalize_pipeline_health_state(
                        getattr(r, "degradation_status", None)
                        or _meta.get("degradation_status", "")
                        or _stage_notes.get("degradation_status", "")
                    ).value

                # ── Schritt 3: Synthetische Goal-Schätzung wenn keine echten Daten ──
                if not musical_goals:
                    # Bestmögliche Korrelation: falls kein Original geladen (z.B. Batch-Only),
                    # MOS-basierten Schätzwert verwenden (MOS 5 → corr 1.0, MOS 1 → corr 0.0).
                    _corr_synth = corr if self._orig_audio is not None else max(0.0, min(1.0, (mos_est - 1.0) / 4.0))
                    musical_goals = {
                        "brillanz": min(1.0, _corr_synth * 0.95 + 0.05),
                        "waerme": min(1.0, _corr_synth * 0.92 + 0.06),
                        "natuerlichkeit": min(1.0, _corr_synth * 0.97 + 0.02),
                        "authentizitaet": min(1.0, _corr_synth * 0.94 + 0.04),
                        "emotionalitaet": min(1.0, _corr_synth * 0.90 + 0.05),
                        "transparenz": min(1.0, _corr_synth * 0.93 + 0.04),
                        "bass_kraft": min(1.0, _corr_synth * 0.91 + 0.05),
                        "groove": min(1.0, _corr_synth * 0.96 + 0.02),
                        "spatial_depth": min(1.0, _corr_synth * 0.88 + 0.07),
                        "timbre_authentizitaet": min(1.0, _corr_synth * 0.93 + 0.04),
                        "tonal_center": min(1.0, _corr_synth * 0.98 + 0.01),
                        "micro_dynamics": min(1.0, _corr_synth * 0.94 + 0.04),
                        "separation_fidelity": min(1.0, _corr_synth * 0.89 + 0.06),
                        "artikulation": min(1.0, _corr_synth * 0.93 + 0.04),
                    }
                    synthesized_goals = set(musical_goals.keys())  # alle als geschätzt markieren

                # ── Schritt 4: GUI-Texte zusammenstellen ──
                stars = "⭐" * max(1, min(5, round(mos_est)))

                # --- Qualitätsscore-Label ---
                _score_lines = [f"{stars}  Qualitätsscore: {mos_est:.1f} / 5.0"]
                if restorability_grade:
                    _mos_range_str = (
                        f" ({restorability_mos_min:.1f}–{restorability_mos_max:.1f})"
                        if restorability_mos_max > 0
                        else ""
                    )
                    _score_lines.append(f"Restaurierbarkeit: Klasse {restorability_grade}{_mos_range_str}")
                if mushra_score > 0:
                    _mushra_str = f"MUSHRA: {mushra_score:.0f}"
                    if mushra_grade:
                        _mushra_str += f"  ({mushra_grade}"
                        if mushra_itu:
                            _mushra_str += f" · {mushra_itu}"
                        _mushra_str += ")"
                    _score_lines.append(_mushra_str)
                _era_str = era_label_full or (f"{era_label}er" if era_label else "")
                if _era_str:
                    _conf_str = f"  ({era_conf * 100:.0f}%)" if era_conf > 0 else ""
                    _score_lines.append(f"Ära: {_era_str}{_conf_str}")
                if genre_label and genre_label.lower() not in ("unknown", ""):
                    _genre_str = f"Genre: {genre_label}"
                    if genre_bpm > 0:
                        _genre_str += f" · {genre_bpm:.0f} BPM"
                    if genre_key:
                        _genre_str += f" · {genre_key}"
                    _score_lines.append(_genre_str)

                if pipeline_confidence > 0:
                    _score_lines.append(
                        f"Konfidenz: {pipeline_confidence * 100:.0f}%  ·  Datei: {Path(output_path).name}"
                    )
                else:
                    _score_lines.append(f"Datei: {Path(output_path).name}")
                mos_text = "\n".join(_score_lines)

                # --- Info-Banner (immer befüllt nach Verarbeitung) ---
                banner_sections: list[str] = []

                # Fail-Reason prominently first (No-Guess Export Gate — Spec §11.4)
                if fail_reason and fail_reason not in ("None", "none", ""):
                    banner_sections.append(f"🚫  Export blockiert: {fail_reason}")
                elif degradation_status in {"blocked", "critical_degraded", "degraded"}:
                    banner_sections.append(f"⚠️  Degradation-Status: {degradation_status}")
                if degradation_status in {"blocked", "critical_degraded", "degraded"} and primary_error_code:
                    if primary_error_code not in fail_reason:
                        banner_sections.append(f"🧩  Fehlercode: {primary_error_code}")

                # Pipeline-Stats
                if phases_exec_count > 0 or total_time_s > 0:
                    _stat_parts = []
                    if phases_exec_count > 0:
                        _stat_parts.append(f"{phases_exec_count} Phasen ausgeführt")
                    if phases_skip_count > 0:
                        _stat_parts.append(f"{phases_skip_count} übersprungen")
                    if total_time_s > 0:
                        _stat_parts.append(f"{total_time_s:.1f} s")
                    if rt_factor > 0:
                        _stat_parts.append(f"{rt_factor:.1f}× Echtzeit")
                    banner_sections.append("⚙️  Pipeline: " + "  ·  ".join(_stat_parts))

                # Kausal-Ursache
                if top_causal_cause and top_causal_cause not in ("None", "none", ""):
                    _cause_map = {
                        "vinyl_scratches": "Vinyl-Kratzer",
                        "surface_noise": "Oberflächenrauschen",
                        "mechanical_hum": "Mechanisches Brummen",
                        "tape_hiss": "Bandrauschen",
                        "electrical_noise": "Elektrisches Rauschen",
                        "clipping_distortion": "Übersteuerungsverzerrung",
                        "wow": "Wow",
                        "flutter": "Flutter",
                        "dropout": "Signalausfall",
                        "codec_artifacts": "Codec-Artefakte",
                        "room_resonance": "Raumresonanz",
                        "microphone_noise": "Mikrofon-Rauschen",
                        "dc_offset": "DC-Gleichspannungsversatz",
                    }
                    _cause_de = _cause_map.get(top_causal_cause, top_causal_cause)
                    _cause_str = f"🔍  Hauptursache: {_cause_de}"
                    if causal_conf > 0:
                        _cause_str += f"  ({causal_conf * 100:.0f}% Sicherheit)"
                    banner_sections.append(_cause_str)

                # Qualitätsverbesserung
                if quality_before_score > 0 and quality_after_score > 0:
                    _delta_str = f"+{quality_delta:.0f}" if quality_delta >= 0 else f"{quality_delta:.0f}"
                    _qi_str = (
                        f"📈  Qualität: {quality_before_score:.0f} → {quality_after_score:.0f} Pkte ({_delta_str})"
                    )
                    if delta_snr != 0:
                        _snr_sign = "+" if delta_snr >= 0 else ""
                        _qi_str += f"  ·  SNR: {_snr_sign}{delta_snr:.1f} dB"
                    banner_sections.append(_qi_str)

                # Temporale Kohärenz & Emotionaler Bogen
                _perc_parts = []
                if temporal_coh_score > 0:
                    _perc_parts.append(f"Temporale Kohärenz: {temporal_coh_score:.2f}")
                if emotional_arc_score > 0:
                    _perc_parts.append(f"Emotionaler Bogen: {emotional_arc_score:.2f}")
                if _perc_parts:
                    banner_sections.append("🎭  " + "  ·  ".join(_perc_parts))

                # Feedback-Chain & Excellence
                _opt_parts = []
                if feedback_retries > 0:
                    _opt_parts.append(f"Optimierung: {feedback_retries}× Anpassung")
                    if feedback_chain_score > 0:
                        _opt_parts.append(f"Score: {feedback_chain_score:.2f}")
                if excellence_steps:
                    _opt_parts.append(f"Excellence: {len(excellence_steps)} Schritte")
                if _opt_parts:
                    banner_sections.append("♻️  " + "  ·  ".join(_opt_parts))

                # Genre-Details (Schlager)
                if genre_is_schlager and (genre_accordion > 0 or genre_bpm > 0):
                    _g_parts = ["🪗  Schlager-Profil aktiv"]
                    if genre_bpm > 0:
                        _g_parts.append(f"{genre_bpm:.0f} BPM")
                    if genre_key:
                        _g_parts.append(genre_key)
                    if genre_accordion > 0:
                        _g_parts.append(f"Akkordeon: {genre_accordion * 100:.0f}%")
                    banner_sections.append("  ·  ".join(_g_parts))

                # Pipeline-Hinweis (wenn vorhanden)
                if pipeline_hint and pipeline_hint not in ("None", ""):
                    _hint_str = f"💡  {pipeline_hint}"
                    if pipeline_tier and pipeline_tier not in ("None", ""):
                        _hint_str += f"  [{pipeline_tier}]"
                    banner_sections.append(_hint_str)
                elif pipeline_tier and pipeline_tier not in ("None", ""):
                    banner_sections.append(f"💡  Pipeline-Tier: {pipeline_tier}")

                # Musical-Goals-Verletzungen
                if musical_violations:
                    _viol_map = {
                        "brillanz": "Brillanz",
                        "waerme": "Wärme",
                        "natuerlichkeit": "Natürlichkeit",
                        "authentizitaet": "Authentizität",
                        "emotionalitaet": "Emotionalität",
                        "transparenz": "Transparenz",
                        "bass_kraft": "Bass-Kraft",
                        "groove": "Groove",
                        "spatial_depth": "Raumtiefe",
                        "timbre_authentizitaet": "Timbre",
                        "tonal_center": "Tonales Zentrum",
                        "micro_dynamics": "Mikro-Dynamik",
                        "separation_fidelity": "Separation",
                        "artikulation": "Artikulation",
                    }
                    _viol_de = [str(_viol_map.get(v, v)) for v in musical_violations]
                    banner_sections.append(f"⚠️  Ziele unter Schwellwert: {', '.join(_viol_de)}")

                # PMGG-Warnungen & Ceiling
                if phase_gate_notes:
                    banner_sections.append(
                        "⚠️  Einige Verarbeitungsschritte wurden angepasst, um den Klang zu schützen."
                    )
                if ceiling_reached:
                    banner_sections.append(
                        "🏆  Das Beste aus dieser Aufnahme wurde herausgeholt — physikalische Grenzen erreicht."
                    )

                def _update_gui():
                    # Radar-Chart aktualisieren
                    if self.radar_widget is not None and musical_goals:
                        self.radar_widget.update_scores(
                            scores=musical_goals,
                            adaptive_thresholds=adaptive_thresholds if adaptive_thresholds else None,
                            applicable_goals=applicable_goals,
                            inapplicable_reasons=inapplicable_reasons if inapplicable_reasons else None,
                            synthesized_goals=synthesized_goals if synthesized_goals else None,
                            adaptation_reasons=adaptation_reasons if adaptation_reasons else None,
                        )
                    # Score-Label: bright flash on appearance, decay to normal
                    _qs_style_final = (
                        "color: #82B89A; font-size: 9pt; font-weight: bold;"
                        " padding: 10px; background: rgba(85, 155, 115, 0.08);"
                        " border-radius: 8px; border: 1px solid rgba(100, 168, 130, 0.26);"
                        " line-height: 150%;"
                    )
                    self.quality_score_label.setStyleSheet(
                        "color: #A8CCBA; font-size: 9pt; font-weight: bold;"
                        " padding: 10px; background: rgba(85, 155, 115, 0.14);"
                        " border-radius: 8px; border: 1px solid rgba(100, 168, 130, 0.42);"
                        " line-height: 150%;"
                    )
                    self.quality_score_label.setText(mos_text)
                    QTimer.singleShot(
                        380,
                        lambda _s=_qs_style_final: self.quality_score_label.setStyleSheet(_s),
                    )
                    # Quality Meter gauge — animated count-up
                    try:
                        import re as _re

                        _m = _re.search(r"MOS[:\s]+(\d+(?:[.,]\d+)?)", mos_text)
                        if _m:
                            self._animate_mos_gauge(float(_m.group(1).replace(",", ".")))
                    except Exception:
                        pass
                    # Info-Banner — immer befüllt wenn Daten vorhanden
                    if banner_sections:
                        self.info_banner.setText("\n".join(banner_sections))
                        self.info_banner.setStyleSheet("""
                            color: #B0BEC5; font-size: 8pt; padding: 10px;
                            background: rgba(30, 40, 55, 0.80);
                            border-radius: 8px; border: 1px solid rgba(96, 125, 139, 0.35);
                            line-height: 155%;
                        """)
                        self.info_banner.setVisible(True)
                    else:
                        self.info_banner.setVisible(False)

                    # ── Abschluss-Zusammenfassung im Defekt-Panel (links) ──
                    if hasattr(self, "defect_summary_label") and restoration_result is not None:
                        _sum: list[str] = []
                        _has_problem = degradation_status in {"blocked", "critical_degraded", "degraded"}
                        _winning_var = getattr(restoration_result, "winning_variant", None)
                        _is_passthrough = _winning_var == "clean_digital_pass_through"

                        if _is_passthrough:
                            # Clean digital source — pipeline skipped intentionally
                            _sum.append("✅  Saubere Quelle — kein Eingriff nötig")
                            _sum.append("🎵  Die Aufnahme war bereits in hervorragendem Zustand.")
                            _sum.append("   Aurik hat darauf verzichtet, unnötige Veränderungen")
                            _sum.append("   vorzunehmen (Overprocessing-Schutz).")
                            if mos_est > 0:
                                _sum.append(f"📊  Qualitätsmessung (VERSA): {mos_est:.1f} / 5.0 MOS")
                            elif quality_after_score > 0:
                                _sum.append(f"📊  Qualitätsscore: {quality_after_score:.0f} / 100")
                            _color_pt = "#82B89A"
                            _bg_pt = "rgba(85,155,115,0.09)"
                            _brd_pt = "rgba(100,168,130,0.24)"
                            self.defect_summary_label.setText("\n".join(_sum))
                            self.defect_summary_label.setStyleSheet(f"""
                                color: {_color_pt}; font-size: 9pt; padding: 12px;
                                background: {_bg_pt};
                                border-radius: 10px; border: 1px solid {_brd_pt};
                                line-height: 160%;
                            """)
                            if hasattr(self, "defect_count_live_label"):
                                self.defect_count_live_label.setText("✅ Fertig")
                                self.defect_count_live_label.setStyleSheet(
                                    "color: #82B89A; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                                )
                                self.defect_count_live_label.setVisible(True)
                        else:
                            # Normal restoration path
                            if _has_problem:
                                _sum.append(
                                    f"⚠️  Restaurierung mit Einschränkungen\n   {fail_reason or degradation_status}"
                                )
                            else:
                                _sum.append("✅  Restaurierung erfolgreich abgeschlossen")

                            # Was wurde behandelt — Hauptursache
                            _cause_map_de = {
                                "tape_hiss": "Bandrauschen",
                                "tape_dropout": "Bandsignalausfall",
                                "vinyl_crackle": "Vinyl-Knistern",
                                "vinyl_warp": "Vinyl-Verwölbung",
                                "electrical_hum": "Elektrisches Brummen",
                                "head_misalignment": "Bandkopf-Fehljustage",
                                "dc_offset": "DC-Gleichspannungsversatz",
                                "digital_clip": "Digitale Übersteuerung",
                                "soft_saturation": "Röhren-/Bandsättigung bewahrt",
                                "head_wear": "Bandkopf-Verschleiß",
                                "print_through": "Bandübersprechen",
                                "riaa_curve_error": "RIAA-Kurven-Fehler",
                                "aliasing": "Aliasing-Artefakte",
                                "bias_error": "Vormagnetisierungs-Fehler",
                                "vinyl_scratches": "Vinyl-Kratzer",
                                "surface_noise": "Oberflächenrauschen",
                                "mechanical_hum": "Mechanisches Brummen",
                                "clipping_distortion": "Übersteuerungsverzerrung",
                                "wow": "Wow-Schwankungen",
                                "flutter": "Flutter-Schwankungen",
                                "dropout": "Tonaussetzer",
                                "codec_artifacts": "Codec-Artefakte",
                                "room_resonance": "Raumresonanz",
                            }
                            if top_causal_cause and top_causal_cause not in ("None", "none", ""):
                                _cause_de = _cause_map_de.get(top_causal_cause, top_causal_cause)
                                _conf_s = f" ({causal_conf * 100:.0f}%)" if causal_conf > 0 else ""
                                _sum.append(f"🔍  Hauptproblem behandelt: {_cause_de}{_conf_s}")

                            # Qualitätsverbesserung
                            if quality_before_score > 0 and quality_after_score > 0:
                                _sign = "+" if quality_delta >= 0 else ""
                                _sum.append(
                                    f"📈  Qualität: {quality_before_score:.0f} → {quality_after_score:.0f}"
                                    f"  ({_sign}{quality_delta:.0f} Punkte)"
                                )
                                if delta_snr != 0:
                                    _snr_s = f"+{delta_snr:.1f}" if delta_snr >= 0 else f"{delta_snr:.1f}"
                                    _sum.append(f"   Rauschverbesserung: {_snr_s} dB")
                            elif mos_est > 0:
                                _sum.append(f"🎵  Klangqualität: {mos_est:.1f} / 5.0 MOS")

                            # Phasen
                            if phases_exec_count > 0:
                                _ph_s = f"⚙️  {phases_exec_count} Verarbeitungsschritte ausgeführt"
                                if phases_skip_count > 0:
                                    _ph_s += f"  ({phases_skip_count} nicht benötigt)"
                                _sum.append(_ph_s)

                            # Musical-Goals-Verletzungen
                            if musical_violations:
                                _vmap = {
                                    "brillanz": "Brillanz",
                                    "waerme": "Wärme",
                                    "natuerlichkeit": "Natürlichkeit",
                                    "authentizitaet": "Authentizität",
                                    "emotionalitaet": "Emotionalität",
                                    "transparenz": "Transparenz",
                                    "bass_kraft": "Bass-Kraft",
                                    "groove": "Groove",
                                    "spatial_depth": "Raumtiefe",
                                    "timbre_authentizitaet": "Timbre",
                                    "tonal_center": "Tonales Zentrum",
                                    "micro_dynamics": "Mikro-Dynamik",
                                    "separation_fidelity": "Separation",
                                    "artikulation": "Artikulation",
                                }
                                _vnames: list[str] = [str(_vmap.get(v, v)) for v in musical_violations]
                                _sum.append(f"⚠️  Ziele unter Schwellwert: {', '.join(_vnames)}")

                            # Ceiling / Optimum
                            if ceiling_reached:
                                _sum.append("🏆  Physikalisches Optimum dieser Aufnahme erreicht")
                            elif feedback_retries > 0:
                                _sum.append(f"♻️  {feedback_retries}× nachoptimiert für bestes Ergebnis")

                            _color = "#B8A068" if _has_problem else "#82B89A"
                            _bg = "rgba(150,130,68,0.09)" if _has_problem else "rgba(85,155,115,0.09)"
                            _brd = "rgba(150,130,68,0.24)" if _has_problem else "rgba(100,168,130,0.24)"
                            self.defect_summary_label.setText("\n".join(_sum))
                            self.defect_summary_label.setStyleSheet(f"""
                                color: {_color}; font-size: 9pt; padding: 12px;
                                background: {_bg};
                                border-radius: 10px; border: 1px solid {_brd};
                                line-height: 160%;
                            """)
                            # Live-Zähler auch aktualisieren
                            if hasattr(self, "defect_count_live_label"):
                                if _has_problem:
                                    self.defect_count_live_label.setText("⚠ Eingeschränkt")
                                    self.defect_count_live_label.setStyleSheet(
                                        "color: #B8A068; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                                    )
                                else:
                                    self.defect_count_live_label.setText("✅ Fertig")
                                    self.defect_count_live_label.setStyleSheet(
                                        "color: #82B89A; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                                    )
                                self.defect_count_live_label.setVisible(True)

                    if hasattr(self, "status_text") and degradation_status in {
                        "blocked",
                        "critical_degraded",
                        "degraded",
                    }:
                        _status_reason = fail_reason if fail_reason not in ("None", "none", "") else degradation_status
                        _status_code = f" · Code: {primary_error_code}" if primary_error_code else ""
                        self.status_text.setStyleSheet("color: #B8A068; font-size: 10pt;")
                        self.status_text.setText(f"⚠ Verarbeitung mit Einschränkungen: {_status_reason}{_status_code}")
                    self._update_ab_player_state()
                    self._update_waveform(self._rest_audio, self._rest_sr)

                QTimer.singleShot(0, _update_gui)

            except Exception as _ex:
                _ex_msg = str(_ex)

                def _show_err(_msg=_ex_msg):
                    self.quality_score_label.setText(f"⚠️ Score-Berechnung fehlgeschlagen: {_msg}")

                QTimer.singleShot(0, _show_err)

        threading.Thread(target=_run, daemon=True).start()

    def _animate_mos_gauge(self, target_mos: float) -> None:
        """Smooth EaseOutCubic count-up animation for the MOS quality meter gauge."""
        _STEPS = 28
        _INTERVAL_MS = 22  # ~600 ms total
        _start = getattr(self, "_last_mos_displayed", 1.0)
        _delta = target_mos - _start
        _step_ref = [0]
        _prev = getattr(self, "_mos_anim_timer", None)
        if _prev is not None:
            _prev.stop()
        _t = QTimer(self)
        self._mos_anim_timer = _t

        def _tick() -> None:
            _step_ref[0] += 1
            _t_norm = _step_ref[0] / _STEPS
            _eased = 1.0 - (1.0 - _t_norm) ** 3
            _current = min(_start + _delta * _eased, 5.0)
            if hasattr(self, "quality_meter_widget"):
                self.quality_meter_widget.set_mos(_current)
            if _step_ref[0] >= _STEPS:
                _t.stop()
                self._last_mos_displayed = target_mos
                if hasattr(self, "quality_meter_widget"):
                    self.quality_meter_widget.set_mos(target_mos)

        _t.timeout.connect(_tick)
        _t.start(_INTERVAL_MS)

    def _update_waveform(self, audio, sr):
        """Waveform im Haupt-Thread rendern; Spektrogramm im Hintergrundthread berechnen."""
        try:
            if hasattr(self, "waveform_widget"):
                self.waveform_widget.update_waveform(audio, sr)
            # A/B-Vergleich: Restauriert-Widget + Original-Widget befüllen
            if hasattr(self, "waveform_widget_rest_ab"):
                self.waveform_widget_rest_ab.update_waveform(audio, sr)
            if hasattr(self, "waveform_widget_orig_ab"):
                _orig = getattr(self, "_orig_audio", None)
                if _orig is not None:
                    self.waveform_widget_orig_ab.update_waveform(_orig, int(getattr(self, "_orig_sr", sr)))
            # Kein automatischer Tab-Wechsel — Wellenform bleibt Standardansicht
            if hasattr(self, "spectrogram_widget"):
                # Spektrogramm-Berechnung ist bei langen Dateien sehr aufwändig
                # → in Hintergrundthread auslagern (QTimer.singleShot in update_spectrogram
                #   stellt sicher, dass self.update() im Haupt-Thread aufgerufen wird)
                _widget = self.spectrogram_widget
                _audio_cp = audio  # kein .copy() nötig – nur gelesen
                _sr_cp = int(sr)
                threading.Thread(
                    target=_widget.update_spectrogram,
                    args=(_audio_cp, _sr_cp),
                    daemon=True,
                ).start()
        except Exception as _exc:
            logger.debug("Waveform-Update fehlgeschlagen: %s", _exc)

    def _update_waveform_live(self, audio, sr, phase_id):
        """Live-Waveform-Update nach jeder UV3-Phase — KEIN Zoom-Reset, KEIN Spektrogramm.

        Wird pro Phase (und bei Per-Channel-Phasen pro Kanal) aufgerufen.
        Zeigt den aktuellen Audio-Zustand in Echtzeit während der Restaurierung.
        """
        try:
            if hasattr(self, "waveform_widget"):
                self.waveform_widget.update_audio_live(audio, sr)
            # A/B-Vergleich: Restauriert-Widget aktualisieren (nicht das Original)
            if hasattr(self, "waveform_widget_rest_ab"):
                self.waveform_widget_rest_ab.update_audio_live(audio, sr)
        except Exception as _exc:
            logger.debug("Live-Waveform-Update fehlgeschlagen (%s): %s", phase_id, _exc)

    def _update_defects(self, defects):
        """Update defect counter display and human-readable summary label"""
        # Count-up animation when scan first completes (status=="detected")
        _status = defects.get("status", "")
        if _status == "detected" and not defects.get("_no_anim"):
            self._defect_anim_target = dict(defects)
            self._defect_anim_frame = 0
            if not hasattr(self, "_defect_anim_timer"):
                self._defect_anim_timer = QTimer(self)
                self._defect_anim_timer.timeout.connect(self._tick_defect_reveal)
            if not self._defect_anim_timer.isActive():
                self._defect_anim_timer.start(85)
            defects = {
                k: (0.001 if isinstance(v, (int, float)) and k not in ("status", "_no_anim") else v)
                for k, v in defects.items()
            }
        try:
            if hasattr(self, "defect_counter_widget"):
                self.defect_counter_widget.update_defects(defects)
        except Exception as _exc:
            logger.debug("Defekt-Update fehlgeschlagen: %s", _exc)
        # Update user-friendly summary for defect_summary_label
        if hasattr(self, "defect_summary_label"):
            # Mapping: interne Schlüssel → (Laienname, Schwellwerte [leicht, mittel, schwer])
            label_map = {
                # Analoge Defekte (skalierte Zählwerte)
                "clicks": ("Knackser", 0.5, 2.0),
                "crackle": ("Knistern", 0.1, 0.5),
                "pops": ("Pops", 0.5, 3.0),
                "clipping": ("Übersteuerung", 0.05, 0.3),
                "hum": ("Brummen", 0.05, 0.4),
                "noise_level": ("Rauschen", 0.1, 0.5),
                "noise": ("Rauschen", 0.1, 0.5),
                "sibilance": ("Zischlaute", 0.1, 0.5),
                "dropout": ("Tonaussetzer", 0.5, 3.0),
                "wow": ("Wow (<0.5 Hz)", 0.2, 0.8),
                "flutter": ("Flutter (0.5–200 Hz)", 0.2, 0.8),
                "rumble": ("Tieffrequenzrumpeln", 0.1, 0.5),
                # 0–100 % Skala (alle weiteren Defekttypen)
                "dc_offset": ("DC-Gleichspannungsversatz", 5.0, 30.0),
                "digital_artifacts": ("Digitale Artefakte", 5.0, 30.0),
                "compression_artifacts": ("Codec-Artefakte", 5.0, 30.0),
                "stereo_imbalance": ("Stereo-Imbalance", 5.0, 30.0),
                "phase_issues": ("Phasenfehler", 5.0, 30.0),
                "bandwidth_loss": ("Bandbreitenverlust", 5.0, 30.0),
                "pitch_drift": ("Tonhöhendrift", 5.0, 30.0),
                "reverb_excess": ("Übermäßiger Hall", 5.0, 30.0),
                "print_through": ("Bandübersprechen", 5.0, 30.0),
                "quantization_noise": ("Quantisierungsrauschen", 5.0, 30.0),
                "jitter_artifacts": ("Jitter-Artefakte", 5.0, 30.0),
                "dynamic_compression_excess": ("Loudness-Überkompression", 5.0, 30.0),
                "pre_echo": ("Pre-Echo (Codec)", 5.0, 30.0),
                "transient_smearing": ("Transienten-Verschmierung", 5.0, 30.0),
                "head_wear": ("Kopf-/Azimuth-Fehler", 5.0, 30.0),
                "riaa_curve_error": ("RIAA-Kurven-Fehler", 5.0, 30.0),
                "aliasing": ("Aliasing-Artefakte", 5.0, 30.0),
                "bias_error": ("Vormagnetisierungs-Fehler", 5.0, 30.0),
            }
            # Nur echte Defektfelder auswerten ("status"-Key ausschliessen)
            active = [
                (k, v)
                for k, v in defects.items()
                if k != "status" and isinstance(v, (int, float)) and v > 0.01 and k in label_map
            ]

            # ── Echtzeit-Defektzähler aktualisieren ─────────────────
            if hasattr(self, "defect_count_live_label"):
                n = len(active)
                if n > 0:
                    self.defect_count_live_label.setText(
                        t("status.defect_count", count=n, suffix=("e" if n != 1 else ""))
                    )
                    self.defect_count_live_label.setStyleSheet(
                        "color: #B8A068; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                    )
                else:
                    self.defect_count_live_label.setText(t("status.clean_short"))
                    self.defect_count_live_label.setStyleSheet(
                        "color: #82B89A; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                    )
                self.defect_count_live_label.setVisible(True)
            if not active:
                _rest_score = getattr(self, "_restorability_score", 100.0)
                _medium = getattr(self, "_raw_medium_type", "unknown")
                # Material-spezifische Defekt-Hinweise auch ohne Scanner-Treffer
                _MEDIUM_HINTS: dict[str, str] = {
                    "tape": "Leises Bandrauschen, mögliche Tonhöhenschwankungen (Wow/Flutter) und Dropout-Spuren.",
                    "reel_tape": "Spulenband-typisches Hiss, Print-Through-Echos und Azimuth-Drift möglich.",
                    "cassette": "Kassetten-Rauschen, schwache Frequenzbegrenzung (<15 kHz) und Bandsättigung.",
                    "vinyl": "Submikroskopische Knackser, subtiles Rumpeln und Pressungsverzerrung.",
                    "shellac": "Breitband-Schellackrauschen (>−30 dB) und stark begrenzte Bandbreite (<8 kHz).",
                    "wax_cylinder": "Extreme Bandbreiten-Limitierung (<5 kHz) und Oberflächenrauschen.",
                    "lacquer_disc": "Lackfoliendegradierung, subtile Klicks und Substrat-Rauschen.",
                    "wire_recording": "Drahtband-Jitter und partielle Frequenzausfälle.",
                    "mp3_low": "Codec-Artefakte (Prä-Echo, Hochton-Ringing) und Bandbreitenbeschneidung.",
                    "mp3_high": "Subtile Codec-Ringing-Artefakte und leichter Hochtonverlust ab 16 kHz.",
                    "aac": "AAC-Preprocessing-Spuren und subtile Transientenverschmierung.",
                    "streaming": "Streaming-Codec-Artefakte und variable Bitratenschwankungen.",
                    "dat": "DAT-Jitter und mögliche Word-Clock-Fehler bei alten Aufnahmen.",
                }
                _hint = _MEDIUM_HINTS.get(_medium, "")
                _status = defects.get("status", "detected")
                if _status == "completed":
                    # Restoration done — show success message in green
                    if _rest_score < 65 or _hint:
                        _lines = ["✅  Restaurierung erfolgreich abgeschlossen"]
                        if _hint:
                            _lines.append(f"   Subtile {_hint.split(',')[0].lower()} wurden behandelt.")
                        _lines.append("→ Alle erkannten Defekte wurden präzise korrigiert")
                    else:
                        _lines = ["✅  Aufnahme erfolgreich restauriert — keine messbaren Defekte verblieben"]
                    self.defect_summary_label.setText("\n".join(_lines))
                    self.defect_summary_label.setStyleSheet("""
                        color: #82B89A; font-size: 9pt; padding: 12px;
                        background: rgba(85, 155, 115, 0.09);
                        border-radius: 10px; border: 1px solid rgba(100, 168, 130, 0.24);
                    """)
                    if hasattr(self, "defect_count_live_label"):
                        self.defect_count_live_label.setText("✅ Behandelt")
                        self.defect_count_live_label.setStyleSheet(
                            "color: #82B89A; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                        )
                elif _rest_score < 65 or _hint:
                    # Pre-restoration: subtle degradation hinted by material type
                    _lines = ["⚠  Subtile Signal-Degradierung erkannt"]
                    if _hint:
                        _lines.append(f"   {_hint}")
                    if _rest_score < 65:
                        _lines.append(
                            f"   Restaurierbarkeit {_rest_score:.0f}/100 – Aurik wird die Qualität messbar verbessern."
                        )
                    else:
                        _lines.append("   Aurik kann die Aufnahme weiter verfeinern.")
                    _lines.append("→ Defekte werden beim Restaurieren präzise behandelt")
                    self.defect_summary_label.setText("\n".join(_lines))
                    self.defect_summary_label.setStyleSheet("""
                        color: #B8A068; font-size: 9pt; padding: 12px;
                        background: rgba(150, 130, 68, 0.09);
                        border-radius: 10px; border: 1px solid rgba(150, 130, 68, 0.24);
                    """)
                    if hasattr(self, "defect_count_live_label"):
                        self.defect_count_live_label.setText("⚠ Subtil")
                        self.defect_count_live_label.setStyleSheet(
                            "color: #B8A068; font-size: 8pt; background: transparent; font-weight: bold; padding: 0 2px;"
                        )
                else:
                    # Wirklich sauber: pristine digital source, hohe Restorabilität
                    self.defect_summary_label.setText(t("status.no_defects_detected"))
                    self.defect_summary_label.setStyleSheet("""
                        color: #82B89A; font-size: 10pt; padding: 12px;
                        background: rgba(85, 155, 115, 0.09);
                        border-radius: 10px; border: 1px solid rgba(100, 168, 130, 0.24);
                    """)
            else:
                # Alle Defekte als Einzelzeilen mit Premium-Icons und Schweregrad anzeigen
                import os as _os

                _DEF_ICONS = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "resources", "defect_icons")
                html_lines = []
                has_severe = False
                for k, v in sorted(active, key=lambda x: -x[1]):
                    name, thr_light, thr_heavy = label_map[k]
                    if v >= thr_heavy:
                        _sev_icon = "severity_high"
                        sev = "schwer"
                        _sev_color = "#E06060"
                        has_severe = True
                    elif v >= thr_light:
                        _sev_icon = "severity_medium"
                        sev = "mittel"
                        _sev_color = "#D4B040"
                    else:
                        _sev_icon = "severity_low"
                        sev = "leicht"
                        _sev_color = "#6CC090"
                    # Defect-type icon
                    _def_path = _os.path.join(_DEF_ICONS, f"{k}.png")
                    _sev_path = _os.path.join(_DEF_ICONS, f"{_sev_icon}.png")
                    _def_img = (
                        (f'<img src="file:///{_def_path}" width="20" height="20" style="vertical-align:middle;">')
                        if _os.path.exists(_def_path)
                        else ""
                    )
                    _sev_img = (
                        (f'<img src="file:///{_sev_path}" width="14" height="14" style="vertical-align:middle;">')
                        if _os.path.exists(_sev_path)
                        else ""
                    )
                    html_lines.append(
                        f"<tr><td>{_def_img}</td>"
                        f'<td style="padding-left:5px;">{name}</td>'
                        f'<td style="padding-left:8px;">{_sev_img}</td>'
                        f'<td style="padding-left:4px;color:{_sev_color};">{sev}</td></tr>'
                    )
                n = len(active)
                _s = defects.get("status", "detected")
                if _s == "detected":
                    header = f"⚠ {n} Defekt{'e' if n != 1 else ''} erkannt:"
                    action = "→ werden beim Restaurieren entfernt"
                elif _s == "correcting":
                    header = f"⚠ {n} Defekt{'e' if n != 1 else ''} werden bearbeitet:"
                    action = "→ werden gerade korrigiert …"
                else:
                    header = f"⚠ {n} Defekt{'e' if n != 1 else ''} verblieben:"
                    action = "→ konnten nicht vollständig behoben werden"
                _table = '<table cellspacing="2" cellpadding="1">' + "".join(html_lines) + "</table>"
                summary_html = f"{header}<br>{_table}<br>{action}"
                self.defect_summary_label.setTextFormat(Qt.TextFormat.RichText)
                self.defect_summary_label.setText(summary_html)
                color = "#B87A7A" if has_severe else "#B8A068"
                bg = "rgba(148,82,82,0.09)" if has_severe else "rgba(150,130,68,0.09)"
                brd = "rgba(152,88,88,0.24)" if has_severe else "rgba(150,130,68,0.24)"
                self.defect_summary_label.setStyleSheet(f"""
                    color: {color}; font-size: 9pt; padding: 10px;
                    background: {bg};
                    border-radius: 10px; border: 1px solid {brd};
                """)

        # Overlay nach initialem Scan ausblenden
        if hasattr(self, "_phase_overlay_label"):
            self._phase_overlay_label.setVisible(False)

        # Waveform-Overlay mit Defekt-Daten versorgen
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_defects(defects)

        # Nur wenn Batch-Restaurierung NICHT läuft: Buttons freischalten + UI zurücksetzen.
        # Während der Restaurierung sendet BatchProcessingThread ebenfalls defect_update-
        # Signale (status="correcting" / "completed") — diese dürfen den Fortschrittsbalken
        # NICHT auf 0 zurücksetzen und die Buttons NICHT vorzeitig freischalten.
        _batch_is_running = bool(self.batch_thread and self.batch_thread.isRunning())
        if not _batch_is_running:
            # Analyse abgeschlossen → Magic Buttons freischalten
            for _btn_name in ("btn_magic_restoration", "btn_magic_studio"):
                if hasattr(self, _btn_name):
                    _btn = getattr(self, _btn_name)
                    _btn.setEnabled(True)
                    _btn.update()
            if hasattr(self, "progress_bar"):
                self.progress_bar.setRange(0, 10000)
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            if hasattr(self, "status_text") and hasattr(self, "current_file_path") and self.current_file_path:
                self.status_text.setText(t("status.analysis_done_prompt", file=Path(self.current_file_path).name))

    def _tick_defect_reveal(self) -> None:
        """QTimer slot: animate defect scores counting up from 0 to scan values (22 frames × 85 ms)."""
        if not hasattr(self, "_defect_anim_target"):
            if hasattr(self, "_defect_anim_timer"):
                self._defect_anim_timer.stop()
            return
        self._defect_anim_frame = min(getattr(self, "_defect_anim_frame", 0) + 1, 22)
        _frac = self._defect_anim_frame / 22.0
        _partial: dict = {
            k: (float(v) * _frac if isinstance(v, (int, float)) and k not in ("status", "_no_anim") else v)
            for k, v in self._defect_anim_target.items()
        }
        _partial["_no_anim"] = True
        try:
            if hasattr(self, "defect_counter_widget"):
                self.defect_counter_widget.update_defects(_partial)
        except Exception:
            pass
        if _frac >= 1.0 and hasattr(self, "_defect_anim_timer"):
            self._defect_anim_timer.stop()

    def _on_scan_progress(self, frac: float) -> None:
        """Update waveform scan-cursor from batch restoration progress (0.0–1.0)."""
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_scan_pos(frac)

    def _update_phase(self, phase_text):
        """Update current processing phase in status bar and system status widget.

        Während der Batch-Verarbeitung (Heartbeat aktiv) wird status_text NICHT
        direkt geschrieben — _tick_heartbeat ist der einzige Writer und baut die
        vollständige Zeile inkl. Elapsed/ETA.  Sonst flattert die Anzeige, weil
        _update_phase die Zeitangaben entfernt und der nächste Heartbeat sie wieder
        hinzufügt (sichtbares An/Aus alle 500 ms).
        """
        _heartbeat_active = hasattr(self, "_heartbeat_timer") and self._heartbeat_timer.isActive()
        if not _heartbeat_active:
            self.status_text.setText(f"⚙️ {phase_text}")
            self.status_text.setStyleSheet(
                "color: #E8C060; font-size: 11pt; font-weight: 600;"
                " background: rgba(200,160,40,0.07); border-radius: 6px;"
                " padding: 2px 8px;"
            )
        else:
            # Heartbeat aktiv: sofort Statuszeile mit Zeitinfo aktualisieren,
            # damit der neue Phasentext ohne 500ms-Verzögerung sichtbar wird.
            self._tick_heartbeat()
        if hasattr(self, "resource_status_widget"):
            self.resource_status_widget.update_status(phase=phase_text)
        # Fortschrittsbalken: nur sicherstellen dass Range gesetzt ist.
        # Der prozentuale Fortschritt wird über _on_item_progress (Gesamt) bzw.
        # phase_progress-Signal (untere Bar) aktualisiert.
        if hasattr(self, "progress_bar"):
            pb = self.progress_bar
            if pb.maximum() == 0:
                pb.setRange(0, 10000)
            pb.setVisible(True)
        # Vorhandenes Overlay-Label ausblenden (falls aus früherer Session noch vorhanden)
        ov = getattr(self, "_phase_overlay_label", None)
        if ov is not None:
            ov.setVisible(False)
        # ── Inpainting auto-zoom trigger ──────────────────────────────────
        _INPAINTING_KEYWORDS = (
            "dropout",
            "inpainting",
            "lücken",
            "gap",
            "tonaussetzer",
            "phase_03",
            "phase_15",
            "phase_21",
            "rekonstruktion",
            "lückenrekonstruktion",
        )
        _phase_lower = phase_text.lower()
        _is_inpainting = any(kw in _phase_lower for kw in _INPAINTING_KEYWORDS)
        if _is_inpainting:
            self._start_inpainting_zoom()
        else:
            # A non-inpainting phase started — end the tour if active
            self._end_inpainting_zoom(restore=True)
        # ── Waveform stage visualization ──────────────────────────────────
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.set_active_stage(phase_text)

    # ── Inpainting Auto-Zoom ──────────────────────────────────────────────────

    def _start_inpainting_zoom(self) -> None:
        """Begin the inpainting-zoom tour: display each dropout gap in close-up."""
        if not hasattr(self, "waveform_widget"):
            return
        ww = self.waveform_widget
        # Collect dropout (and gap-fill) locations from the waveform widget
        _locs: list[tuple[float, float]] = list(ww._defect_locations.get("dropout", []))
        # Also include resolved dropouts still held in _resolved_locations
        _locs += list(ww._resolved_locations.get("dropout", []))
        # Deduplicate and sort by start time
        _seen: set[tuple[float, float]] = set()
        _sorted: list[tuple[float, float]] = []
        for seg in _locs:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                key = (round(float(seg[0]), 3), round(float(seg[1]), 3))
                if key not in _seen:
                    _seen.add(key)
                    _sorted.append((float(seg[0]), float(seg[1])))
        _sorted.sort(key=lambda s: s[0])

        if not _sorted:
            return  # No located dropouts — nothing to zoom to

        if ww.audio_data is None:
            return
        total_dur = float(ww.audio_data.shape[0]) / max(1, ww.sample_rate)
        if total_dur <= 0:
            return

        # Already active with the same locations → don't restart
        if getattr(self, "_inpainting_zoom_active", False) and getattr(self, "_inpainting_locations", []) == _sorted:
            return

        # Save current view for restoration after the tour
        if not getattr(self, "_inpainting_zoom_active", False):
            self._inpainting_saved_view: tuple[float, float] = (ww._view_start, ww._view_end)

        self._inpainting_zoom_active = True
        self._inpainting_locations = _sorted
        self._inpainting_location_idx = 0
        self._inpainting_user_override = False
        self._inpainting_total_dur = total_dur

        # Register override callback so wheel-events pause the auto-tour
        ww._inpainting_user_override_cb = self._on_inpainting_user_override

        # Set up cycling QTimer if not present
        if not hasattr(self, "_inpainting_zoom_timer"):
            self._inpainting_zoom_timer = QTimer(self)
            self._inpainting_zoom_timer.timeout.connect(self._inpainting_zoom_tick)
        self._inpainting_zoom_timer.stop()

        # Jump to first location immediately, then start the tick timer
        self._inpainting_zoom_to(0)
        # Dwell-time per gap: at least 4 s, scale with gap width (up to 10 s)
        _dwell_ms = self._inpainting_dwell_ms(0)
        self._inpainting_zoom_timer.start(_dwell_ms)

    def _inpainting_dwell_ms(self, idx: int) -> int:
        """Compute ms to dwell on gap idx before advancing. Scales with gap length."""
        locs = getattr(self, "_inpainting_locations", [])
        if idx >= len(locs):
            return 4000
        t0, t1 = locs[idx]
        gap_s = max(0.0, t1 - t0)
        # Short gap (< 0.5 s): 4 s; medium (0.5–2 s): 6 s; wide (> 2 s): 10 s
        if gap_s < 0.5:
            return 4000
        elif gap_s < 2.0:
            return int(4000 + gap_s * 2000)
        else:
            return 10000

    def _inpainting_zoom_to(self, idx: int) -> None:
        """Zoom the waveform widget to show gap at index idx with context margin."""
        ww = getattr(self, "waveform_widget", None)
        if ww is None or not getattr(self, "_inpainting_zoom_active", False):
            return
        locs = getattr(self, "_inpainting_locations", [])
        if idx >= len(locs):
            return
        t0, t1 = locs[idx]
        total_dur = getattr(self, "_inpainting_total_dur", 1.0)

        # Context: show 20× the gap width on each side, minimum 0.2 s, max 3 s
        gap_s = max(t1 - t0, 0.01)
        margin_s = max(0.2, min(3.0, gap_s * 20.0))

        view_start_s = max(0.0, t0 - margin_s)
        view_end_s = min(total_dur, t1 + margin_s)

        ww.set_view(view_start_s / total_dur, view_end_s / total_dur)

    def _inpainting_zoom_tick(self) -> None:
        """Advance to the next dropout gap in the inpainting tour."""
        if not getattr(self, "_inpainting_zoom_active", False):
            self._inpainting_zoom_timer.stop()
            return
        # User paused the tour via manual zoom — check if timeout-resume desired
        if getattr(self, "_inpainting_user_override", False):
            # Resume auto-tour after one extra dwell period (give user time to pan/zoom)
            self._inpainting_user_override = False
            return
        locs = getattr(self, "_inpainting_locations", [])
        self._inpainting_location_idx += 1
        if self._inpainting_location_idx >= len(locs):
            # All gaps toured — tour will end when phase changes (non-inpainting)
            # Restart from start to keep pulsing until backend signals phase-end
            self._inpainting_location_idx = 0
        self._inpainting_zoom_to(self._inpainting_location_idx)
        _dwell_ms = self._inpainting_dwell_ms(self._inpainting_location_idx)
        self._inpainting_zoom_timer.setInterval(_dwell_ms)

    def _on_inpainting_user_override(self) -> None:
        """Called when the user manually zooms/pans during inpainting — pauses the tour."""
        self._inpainting_user_override = True
        # Don't stop the timer; _inpainting_zoom_tick will skip one cycle, then resume

    def _end_inpainting_zoom(self, restore: bool = True) -> None:
        """End inpainting zoom tour and optionally restore the pre-tour view."""
        if not getattr(self, "_inpainting_zoom_active", False):
            return
        self._inpainting_zoom_active = False
        if hasattr(self, "_inpainting_zoom_timer"):
            self._inpainting_zoom_timer.stop()
        ww = getattr(self, "waveform_widget", None)
        if ww is not None:
            # Remove override callback
            ww._inpainting_user_override_cb = None
            if restore:
                saved = getattr(self, "_inpainting_saved_view", (0.0, 1.0))
                ww.set_view(saved[0], saved[1])

    def _on_phase_step_update(self, step: int, total: int) -> None:
        """Show 'Stufe X / Y — Name' counter below the sub-progress bar.

        total == 0 means the actual total is not yet known (early analysis
        steps before UV3 reports __total_uv3_phases__).  In that case the
        label shows 'Stufe X  ·  Name' without '/ Y'.
        """
        if not hasattr(self, "_phase_step_label"):
            return
        _STEP_NAMES: dict[int, str] = {
            1: "Audio wird geladen",
            2: "Schadensbewertung (Voranalyse)",
            3: "Restaurierung startet",
            4: "Tonträger wird erkannt",
            5: "Tonträgerkette wird analysiert",
            6: "Defekte werden bewertet",
            7: "Restaurierungsplan & Strategie",
            8: "Vorverarbeitung & Reparatur",
        }
        name = _STEP_NAMES.get(step, "") or getattr(self, "_dynamic_step_names", {}).get(step, "")
        if total > 0:
            label = f"Stufe {step} / {total}" + (f"  ·  {name}" if name else "")
        else:
            label = f"Stufe {step}" + (f"  ·  {name}" if name else "")
        self._phase_step_label.setText(label)
        self._phase_step_label.setVisible(True)

    def closeEvent(self, event) -> None:
        """Stop sounddevice playback before closing to avoid PortAudio mutex assertion on Linux."""
        if _SD_AVAILABLE and _sd is not None:
            try:
                _sd.stop()
            except Exception:
                pass
        # Wait briefly for the play thread to finish
        _play_thread = getattr(self, "_play_thread", None)
        if _play_thread is not None and _play_thread.is_alive():
            _play_thread.join(timeout=0.5)
        super().closeEvent(event)

    def _update_mode(self, mode):
        """Update processing mode in resource status widget"""
        if hasattr(self, "resource_status_widget"):
            self.resource_status_widget.update_status(mode=mode)

    def _update_ml_status(self, ml_active, ml_plugins):
        """Update ML plugin status in resource status widget"""
        if hasattr(self, "resource_status_widget"):
            self.resource_status_widget.update_status(ml_active=ml_active, ml_plugins=ml_plugins)

    def _clear_queue(self):
        """Clear processing queue"""
        if self.batch_thread and self.batch_thread.isRunning():
            QMessageBox.warning(self, t("dialog.processing_running_title"), t("dialog.queue_busy_body"))
            return

        self.batch_queue.clear_completed()

        # Update display - remove completed/failed items
        i = 0
        if not hasattr(self, "queue_list"):
            self._update_stats()
            return
        while i < self.queue_list.count():
            list_item = self.queue_list.item(i)
            item_id = list_item.data(Qt.ItemDataRole.UserRole)
            item = self.batch_queue.get_item(item_id)

            if item is None:  # Item was cleared from queue
                self.queue_list.takeItem(i)
            else:
                i += 1

        self.progress_bar.setValue(0)
        self.status_text.setText(t("status.queue_cleared"))
        self._update_stats()

    def _export_all(self):
        """Export-Dialog: Format, Bittiefe und Zielordner wählen, dann AudioExporter nutzen."""
        stats = self.batch_queue.get_stats()
        if stats["completed"] == 0:
            QMessageBox.information(
                self,
                t("dialog.no_processed_title"),
                t("dialog.no_processed_body"),
            )
            return

        # ── Export-Dialog ────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(t("ui.export_dialog_title"))
        dlg.setMinimumWidth(420)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setSpacing(18)

        dlg_layout.addWidget(QLabel(t("ui.export_dialog_intro", count=stats["completed"])))

        # Format-Auswahl
        fmt_group_label = QLabel(t("ui.export_format"))
        fmt_group_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        dlg_layout.addWidget(fmt_group_label)

        # Format-Konfig: fmt_key → (real_ext, bit_depth, extra_export_kwargs)
        # Spiegelt ExportConfigDialog's 10 Formate (§11.4 Spec 08).
        _FMT_CONFIGS: dict = {
            ".flac": (".flac", 24, {}),
            ".wav24": (".wav", 24, {}),
            ".wav16": (".wav", 16, {}),
            ".aiff24": (".aiff", 24, {}),
            ".mp3_320": (".mp3", 16, {"mp3_bitrate": 320}),
            ".mp3_256": (".mp3", 16, {"mp3_bitrate": 256}),
            ".mp3_192": (".mp3", 16, {"mp3_bitrate": 192}),
            ".mp3_v0": (".mp3", 16, {"mp3_vbr_quality": 0}),
            ".mp3_v2": (".mp3", 16, {"mp3_vbr_quality": 2}),
            ".ogg9": (".ogg", 16, {"ogg_quality": 9}),
        }
        fmt_bg = QButtonGroup(dlg)
        formats = [
            (".flac", "FLAC 24-bit     — verlustfrei, Archivqualität  ✅ (empfohlen)"),
            (".wav24", "WAV 24-bit      — verlustfrei, DAW-Kompatibel"),
            (".wav16", "WAV 16-bit      — CD-Qualität, kleinere Datei"),
            (".aiff24", "AIFF 24-bit     — verlustfrei, Apple/ProTools-Kompatibel"),
            (".mp3_320", "MP3 CBR 320 kbps — maximale Kompatibilität"),
            (".mp3_256", "MP3 CBR 256 kbps — gut ausbalanciert"),
            (".mp3_192", "MP3 CBR 192 kbps — kleine Dateien"),
            (".mp3_v0", "MP3 VBR V0       — höchste VBR-Qualität (~245 kbps)"),
            (".mp3_v2", "MP3 VBR V2       — empfohlene VBR-Qualität (~190 kbps)"),
            (".ogg9", "OGG Vorbis Q9   — Open-Source, streaming-optimiert"),
        ]
        rb_formats = []
        for i, (key, label) in enumerate(formats):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            rb.setProperty("fmt_ext", key)
            fmt_bg.addButton(rb)
            rb_formats.append(rb)
            dlg_layout.addWidget(rb)

        # Normalisierung
        chk_normalize = QCheckBox(t("ui.export_normalize"))
        chk_normalize.setChecked(True)
        dlg_layout.addWidget(chk_normalize)

        buttons = QDialogButtonBox(parent=dlg)
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        _ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if _ok_btn is not None:
            _ok_btn.setText(t("ui.export_pick_folder"))
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec_() != QDialog.DialogCode.Accepted:
            return

        # Gewähltes Format ermitteln
        _fmt_key = ".flac"
        for rb in rb_formats:
            if rb.isChecked():
                _fmt_key = rb.property("fmt_ext")
                break
        normalize = chk_normalize.isChecked()
        real_ext, bit_depth, _extra_kwargs = _FMT_CONFIGS.get(_fmt_key, (".flac", 24, {}))

        # Zielordner wählen — System-Dialog (Win32 / zenity / kdialog / tkinter)
        _last_export_dir = getattr(self, "_last_export_dir", "") or str(Path.home())
        _dir_picked = self._pick_with_system_dialog(
            title=t("ui.export_choose_dir"),
            start_dir=_last_export_dir,
            directory=True,
        )
        if not _dir_picked:
            # Qt-Fallback falls kein System-Dialog verfügbar
            _dir_picked = [
                QFileDialog.getExistingDirectory(
                    self,
                    t("ui.export_choose_dir"),
                    _last_export_dir,
                    self._dialog_options(directory_only=True),
                )
            ]
        output_dir = _dir_picked[0] if _dir_picked and _dir_picked[0] else ""
        if output_dir:
            self._last_export_dir = output_dir
        if not output_dir:
            return

        # ── Exportieren ──────────────────────────────────────────────────
        def _do_export():
            exporter_init_error = None
            try:
                _AudioExporter = _bridge_get_audio_exporter_class()
                exporter = _AudioExporter() if _AudioExporter is not None else None
            except Exception as ex:
                exporter_init_error = ex
                exporter = None  # Fallback: shutil.copy

            exported = 0
            errors = []
            for item in self.batch_queue.items:
                if item.status != "completed":
                    continue
                src = Path(item.output_file)
                if not src.exists():
                    errors.append(src.name)
                    continue
                dst = Path(output_dir) / (src.stem + real_ext)
                try:
                    if exporter is not None:
                        audio, sr = sf.read(str(src))
                        try:
                            exporter.export(
                                audio,
                                sr,
                                dst,
                                bit_depth=bit_depth,
                                quality="veryhigh",
                                normalize=normalize,
                                **_extra_kwargs,
                            )
                        except TypeError:
                            # Älterer Exporter ohne erweiterte Format-Parameter → Standardaufruf
                            exporter.export(
                                audio,
                                sr,
                                dst,
                                bit_depth=bit_depth,
                                quality="veryhigh",
                                normalize=normalize,
                            )
                    else:
                        import shutil

                        shutil.copy2(src, dst)
                    exported += 1
                except Exception as ex:
                    errors.append(f"{src.name}: {ex}")

            def _update():
                _extra_tag = ""
                if "mp3_bitrate" in _extra_kwargs:
                    _extra_tag = f" {_extra_kwargs['mp3_bitrate']} kbps CBR"
                elif "mp3_vbr_quality" in _extra_kwargs:
                    _extra_tag = f" VBR V{_extra_kwargs['mp3_vbr_quality']}"
                elif "ogg_quality" in _extra_kwargs:
                    _extra_tag = f" Q{_extra_kwargs['ogg_quality']}"
                fmt_nice = real_ext.upper().lstrip(".") + f" {bit_depth}-bit" + _extra_tag
                msg = t("dialog.export_done_body", exported=exported, fmt=fmt_nice, output_dir=output_dir)

                if exporter is None and exporter_init_error is not None:
                    msg += (
                        "\n\n⚠ Erweiterter Exporter nicht verfügbar. Es wurde ein einfacher Datei-Fallback verwendet."
                    )

                if errors:
                    msg += "\n\n" + t("dialog.export_done_errors", count=len(errors)) + "\n" + "\n".join(errors[:5])
                    msg += (
                        "\n\nHinweis: Bitte Schreibrechte im Zielordner prüfen "
                        "und offene Dateien im Zielprogramm schließen."
                    )

                if exported == 0 and errors:
                    self.status_text.setStyleSheet("color: #B87A7A; font-size: 10pt;")
                    self.status_text.setText("⚠ Export fehlgeschlagen")
                    self.title_bar.set_status("Export fehlgeschlagen", "#B87A7A")
                    QMessageBox.warning(self, "Export fehlgeschlagen", msg)
                    return

                self.status_text.setStyleSheet("color: #82B89A; font-size: 10pt;")
                self.status_text.setText(t("status.export_summary", count=exported, ext=real_ext.upper()))
                self.title_bar.set_status(t("status.export_finished"), "#82B89A")
                QMessageBox.information(self, t("status.export_finished"), msg)

            QTimer.singleShot(0, _update)

        threading.Thread(target=_do_export, daemon=True).start()
        self.status_text.setText(f"⏳ {t('status.exporting')}")

    def _show_settings(self):
        """Einstellungs-Dialog mit Output-Format-Voreinstellung."""
        dlg = QDialog(self)
        dlg.setWindowTitle(t("settings.title"))
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(14)

        layout.addWidget(QLabel(f"<b>{t('settings.default_export_format')}</b>"))

        fmt_bg = QButtonGroup(dlg)
        fmt_choices = [
            (".flac", "FLAC 24-bit  — Archivqualität (Standard)"),
            (".wav", "WAV 24-bit   — für DAW-Weiterverarbeitung"),
            (".mp3", "MP3 320 kbps — maximale Kompatibilität"),
        ]
        current_fmt = getattr(self, "_default_export_fmt", ".flac")
        rb_fmts = []
        for ext, label in fmt_choices:
            rb = QRadioButton(label)
            rb.setChecked(ext == current_fmt)
            rb.setProperty("fmt_ext", ext)
            fmt_bg.addButton(rb)
            layout.addWidget(rb)
            rb_fmts.append(rb)

        layout.addWidget(QLabel(f"<b>{t('settings.default_mode_batch_album')}</b>"))
        mode_bg = QButtonGroup(dlg)
        rb_rest = QRadioButton(t("ui.settings_mode_restoration"))
        rb_stu = QRadioButton(t("ui.settings_mode_studio"))
        rb_rest.setChecked(getattr(self, "_default_mode", "RESTORATION") == "RESTORATION")
        rb_stu.setChecked(getattr(self, "_default_mode", "RESTORATION") == "STUDIO_2026")
        mode_bg.addButton(rb_rest)
        mode_bg.addButton(rb_stu)
        layout.addWidget(rb_rest)
        layout.addWidget(rb_stu)

        # Sprache
        form = QFormLayout()
        lang_combo = QComboBox()
        lang_combo.addItem(t("settings.language_de"), "de")
        lang_combo.addItem(t("settings.language_en"), "en")
        current_lang = get_language()
        idx = max(0, lang_combo.findData(current_lang))
        lang_combo.setCurrentIndex(idx)
        form.addRow(t("settings.language"), lang_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox(parent=dlg)
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() == QDialog.DialogCode.Accepted:
            for rb in rb_fmts:
                if rb.isChecked():
                    self._default_export_fmt = rb.property("fmt_ext")
            self._default_mode = "RESTORATION" if rb_rest.isChecked() else "STUDIO_2026"
            set_language(str(lang_combo.currentData()))
            self._apply_i18n_texts()
            self.title_bar.set_status(t("status.settings_saved"), "#82B89A")

    def _apply_i18n_texts(self) -> None:
        """Refresh visible UI texts after language changes."""
        if hasattr(self, "btn_import"):
            self.btn_import.setText(t("action.open_file"))
        if hasattr(self, "title_bar") and hasattr(self.title_bar, "title_label"):
            self.title_bar.title_label.setText(t("ui.app_title"))
        if hasattr(self, "btn_play_original"):
            self.btn_play_original.setText(f"▶  {t('action.listen_original')}")
        if hasattr(self, "btn_play_restored"):
            self.btn_play_restored.setText(f"▶  {t('action.listen_restored')}")
        if hasattr(self, "btn_stop_playback"):
            self.btn_stop_playback.setText(f"⏹  {t('action.stop')}")

        # Tabs in main visualization area
        if hasattr(self, "viz_tabs") and self.viz_tabs.count() >= 2:
            self.viz_tabs.setTabText(0, t("ui.tab_waveform"))
            self.viz_tabs.setTabText(1, t("ui.tab_spectrogram"))
        if hasattr(self, "ab_hdr"):
            self.ab_hdr.setText(t("ui.ab_compare"))

        # Fallback labels when image assets are unavailable
        if hasattr(self, "btn_magic_restoration") and self.btn_magic_restoration.text().startswith("💿"):
            self.btn_magic_restoration.setText(f"💿  {t('action.restore_restoration')}")
        if hasattr(self, "btn_magic_studio") and self.btn_magic_studio.text().startswith("🎯"):
            self.btn_magic_studio.setText(f"🎯  {t('action.restore_studio')}")

        # Keep user-facing placeholders translated
        if hasattr(self, "detected_medium_label") and not self.current_file_path:
            self.detected_medium_label.setText(t("ui.no_file_loaded"))
        if hasattr(self, "defect_summary_label") and self.defect_summary_label.text() in {
            "Noch keine Analyse",
            "No analysis yet",
        }:
            self.defect_summary_label.setText(t("ui.no_analysis"))
        if hasattr(self, "status_text") and self.status_text.text() in {
            "Bereit für Verarbeitung",
            "Ready for processing",
        }:
            self.status_text.setText(t("status.ready"))

        if hasattr(self, "findChildren"):
            for _lbl in self.findChildren(QLabel):
                if _lbl.text() in {"erkannte Defekte:", "detected defects:"}:
                    _lbl.setText(t("ui.defects_detected_title"))

        self._update_stats()

    def _update_stats(self):
        """Update statistics display"""
        stats = self.batch_queue.get_stats()
        self.stats_label.setText(
            t(
                "status.stats",
                pending=stats["pending"],
                completed=stats["completed"],
                failed=stats["failed"],
            )
        )

    # Window resize events
    def mousePressEvent(self, event):
        """Handle window edge dragging for resize"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_position = event.globalPos()

    def mouseMoveEvent(self, event):
        """Handle window resize on edges"""
        if self.old_position and not self.is_maximized:
            delta = QPoint(event.globalPos() - self.old_position)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_position = event.globalPos()


def main():
    """Launch modern application"""
    app = QApplication(sys.argv)

    # Set app-wide font
    app.setFont(QFont("Segoe UI", 10))

    # Create window
    window = ModernMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
