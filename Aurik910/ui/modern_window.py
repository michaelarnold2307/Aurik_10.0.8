"""
AURIK Professional - Modern Frameless Window mit Premium Look & Feel
Basiert auf PyQt5 mit Custom-Styling, Glassmorphism und Animationen
Mit integrierter Audio-Verarbeitung (Backend V2)
"""

import logging
import math
import os
from pathlib import Path
import sys
import threading

logger = logging.getLogger(__name__)

# ── Bridge: einzige erlaubte Schnittstelle zu backend/core/ (§11 Spec 08) ──
# Frontend importiert Core-Module AUSSCHLIEßLICH über diese Bridge.
try:
    from backend.api.bridge import (
        cache_defect_result,
        clear_defect_cache as _bridge_clear_defect_cache,
        export_guard as _export_guard,
        get_audio_file_validator as _bridge_get_audio_file_validator,
        get_audio_exporter_class as _bridge_get_audio_exporter_class,
        get_cached_defect_result,
        get_carrier_forensics_fn as _bridge_get_carrier_forensics_fn,
        get_defect_scanner as _bridge_get_defect_scanner,
        get_defect_type as _bridge_get_defect_type,
        get_era_classifier_fn as _bridge_get_era_classifier_fn,
        get_genre_classifier_fn as _bridge_get_genre_classifier_fn,
        get_lyrics_guided_enhancement_fn as _bridge_get_lyrics_guided_enhancement,
        get_medium_classifier_fn as _bridge_get_medium_classifier_fn,
        get_cleanup_after_file_fn as _bridge_get_cleanup_after_file_fn,
        get_restorability_estimator_class as _bridge_get_restorability_estimator_class,
        get_aurik_denker_class as _bridge_get_aurik_denker_class,
        warmup_models_background as _warmup_models_background,
    )
    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False

    def _export_guard(audio):  # type: ignore[misc]
        import numpy as _np
        audio = _np.asarray(audio, dtype=_np.float32)
        audio = _np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return _np.clip(audio, -1.0, 1.0)

    def cache_defect_result(file_path: str, result: object) -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar, Defekt-Cache deaktiviert

    def get_cached_defect_result(file_path: str) -> object | None:  # type: ignore[misc]
        return None

    def _warmup_models_background() -> None:  # type: ignore[misc]
        return  # no-op: Bridge nicht verfügbar, kein Vorwärmen möglich

    # Bridge-Fallbacks: geben None zurück (§11.4 Bridge-Fallback)
    def _bridge_get_aurik_denker_class() -> type | None:  # type: ignore[misc]
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

from PyQt5.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
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
        "_locations": {},   # Nach Restaurierung keine Zeitpositionen verfügbar
        "status": status,
    }


# Formate die soundfile NICHT unterstützt → direkt zu pedalboard/librosa
_SF_UNSUPPORTED_EXT = frozenset({
    ".mp3", ".mp2", ".mp1",
    ".m4a", ".m4b", ".m4p", ".aac",
    ".wma", ".asf",
    ".opus", ".webm",
    ".amr", ".3gp", ".3g2",
    ".ac3", ".dts",
})


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
        import librosa as _librosa  # type: ignore
        import threading as _threading
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

    def __init__(self, queue: SimpleBatchQueue):
        super().__init__()
        self.queue = queue
        self._stop_requested = False

    def run(self):
        """Process all items in queue with visualization updates"""
        try:
            # P1: Core-Imports AUSSCHLIEßLICH über Bridge (§11 Spec 08)
            AurikDenkerClass = _bridge_get_aurik_denker_class()  # Primary restorer (§2.2)
            if AurikDenkerClass is None:
                raise RuntimeError(
                    "AurikDenker ist über die Bridge nicht verfügbar. "
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
                self.item_started.emit(item.id)
                self.phase_update.emit(f"Restaurierung startet: {Path(item.input_file).name}")
                # Sofort 3 % zeigen — noch bevor Audio-Loading beginnt.
                # Ohne diesen Emit bleibt die Bar bei 0,00 % für die gesamte
                # Ladedauer (bei großen MP3s bis zu 30 s).
                item.progress = 3
                self.item_progress.emit(item.id, 3)

                # Load audio (MP3, WAV, FLAC, M4A, …)
                audio, sr = _load_audio_robust(item.input_file)
                self.waveform_data.emit(audio, sr)  # Send to visualization
                item.progress = 20
                self.item_progress.emit(item.id, 20)

                # Map GUI modes to Denker modes (§2.2 Spec)
                mode = item.settings.get("mode", "RESTORATION")
                if mode == "STUDIO_2026":
                    ui_mode = "MAXIMUM"
                    _aurik_mode = "studio2026"
                else:  # RESTORATION
                    ui_mode = "QUALITY"
                    _aurik_mode = "quality"

                # Emit mode update
                self.mode_update.emit(ui_mode)

                # ML-Plugins: Lokale ONNX-Modelle (kein Docker)
                self.ml_status_update.emit(False, [])

                item.progress = 28
                self.item_progress.emit(item.id, 28)

                # Defect analysis phase: Cache-First (kein Doppelscan, §9.4)
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

                item.progress = 50
                self.item_progress.emit(item.id, 50)

                # Process
                self.phase_update.emit("Musik wird restauriert …")

                # Phase 2: Correction starting
                defects_correcting = defects.copy()
                defects_correcting["status"] = "correcting"
                self.defect_update.emit(defects_correcting)

                def _on_batch_progress(pct: int, msg: str, elapsed_s: float = 0.0, _item=item) -> None:
                    display = 25 + int(pct * 0.65)
                    _item.progress = min(90, display)
                    self.item_progress.emit(_item.id, min(90, display))
                    if pct > 5 and elapsed_s > 0:
                        remaining_s = max(0, elapsed_s / pct * (100 - pct))
                        eta_str = (
                            f"~{int(remaining_s)}s"
                            if remaining_s < 60
                            else f"~{int(remaining_s // 60)}m{int(remaining_s % 60)}s"
                        )
                        self.phase_update.emit(f"{msg}  ·  ETA {eta_str}")
                    else:
                        self.phase_update.emit(msg)

                # AurikDenker ist der verpflichtende Frontend-Einstiegspunkt.
                _denker = AurikDenkerClass()
                result = _denker.denke(
                    audio,
                    sr,
                    mode=_aurik_mode,
                    progress_callback=_on_batch_progress,
                )
                item.progress = 80
                self.item_progress.emit(item.id, 80)

                # Phase 3: Post-Restore Defekt-Status + ML-Plugin-Anzeige aus RestorationResult
                _post_scores = result.defect_scores if hasattr(result, "defect_scores") else {}
                self.defect_update.emit(_result_scores_to_display(_post_scores, status="completed"))

                # ML-Plugin-Status: Phasen-Namen → aktive Plugin-Schlüsselwörter
                # Mapping: Phasenname → Display-Label für resource_status_widget
                _ML_PHASE_MARKERS = {
                    "deepfilternet": "DeepFilterNet", "dfn": "DeepFilterNet",
                    "melbandroformer": "MelBandRoformer", "bs_roformer": "MelBandRoformer",
                    "mdx23c": "MDX23C", "mdx": "MDX23C",
                    "sgmse": "SGMSE+", "resemble": "Resemble-Enhance",
                    "apollo": "Apollo", "rmvpe": "RMVPE", "crepe": "CREPE",
                    "audiosr": "AudioSR", "vocos": "Vocos", "bigvgan": "BigVGAN",
                    "panns": "PANNs", "beats": "BEATs",
                    "versa": "VERSA", "flow_matching": "Flow-Matching",
                    "cqtdiff": "CQTdiff+", "fcpe": "FCPE",
                }
                _phases = list(getattr(result, "phases_executed", []))
                _active_ml: list[str] = []
                _seen: set[str] = set()
                for _p in _phases:
                    _pl = _p.lower()
                    for _key, _label in _ML_PHASE_MARKERS.items():
                        if _key in _pl and _label not in _seen:
                            _active_ml.append(_label)
                            _seen.add(_label)
                self.ml_status_update.emit(bool(_active_ml), _active_ml)

                # RestorationResult im Item speichern (Musical Goals, Genealogie, …)
                # → wird in _on_item_finished an _compute_and_show_quality weitergereicht
                item.restoration_result = result

                # Save: export_guard (NaN/Inf + Clip) + atomares Schreiben (.tmp → os.replace)
                self.phase_update.emit("Ergebnis wird gespeichert …")
                # Handle RestorationResult object
                if hasattr(result, "audio"):
                    restored_audio = result.audio
                else:
                    restored_audio = result  # Fallback
                restored_audio = _export_guard(restored_audio)
                # Ensure output directory exists (P1: output/ subfolder)
                os.makedirs(os.path.dirname(item.output_file), exist_ok=True)
                _tmp_path = item.output_file + ".tmp"
                try:
                    sf.write(_tmp_path, restored_audio, sr)
                    os.replace(_tmp_path, item.output_file)
                finally:
                    if os.path.exists(_tmp_path):
                        try:
                            os.remove(_tmp_path)
                        except OSError:
                            pass
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


class WaveformWidget(QWidget):
    """Premium Professional Stereo Waveform Visualization

    Features:
    - Dual-channel stereo display (L/R separated)
    - Peak/RMS envelope rendering
    - Professional gradient fills
    - Time axis with markers
    - Amplitude scale in dB
    - High-quality antialiasing
    """

    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = 44100
        self.is_loading = False  # Lade-Zustandsanzeige
        self.defects: dict = {}   # Defekte für Waveform-Overlay

        # Zoom/Pan state (fractions 0.0–1.0 of total duration)
        self._view_start: float = 0.0
        self._view_end:   float = 1.0
        self._pan_anchor: int | None = None       # x-pixel at pan-press
        self._pan_view_start_at_press: float = 0.0
        # Timed defect locations: {"clicks": [(t_start, t_end), ...], ...}
        self._defect_locations: dict = {}

        self.setMouseTracking(True)
        self.setMinimumHeight(320)
        self.setStyleSheet("""
            background: rgba(20, 20, 30, 0.95);
            border: 2px solid rgba(102, 126, 234, 0.4);
            border-radius: 10px;
        """)

    def update_waveform(self, audio, sr):
        """Update waveform data and reset view window."""
        self.audio_data = audio
        self.sample_rate = sr
        # Reset zoom/pan to show full file on new load
        self._view_start = 0.0
        self._view_end   = 1.0
        self.update()

    def set_defects(self, defects: dict) -> None:
        """Speichert Defekte für farbiges Severity-Overlay in der Wellenform."""
        self.defects = defects or {}
        self._defect_locations = self.defects.get("_locations", {})
        self.update()

    # ── Zoom / Pan interactions ───────────────────────────────────────────────

    def wheelEvent(self, event):
        """Zoom in/out centered on the mouse X position."""
        if self.audio_data is None:
            return
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
        new_end   = min(1.0, new_start + new_span)
        # Clamp start if end was clipped
        new_start = max(0.0, new_end - new_span)
        self._view_start = new_start
        self._view_end   = new_end
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
            self._view_end   = new_start + span
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
                    "📂  Datei wird geladen …",
                )
                painter.setPen(QColor(180, 160, 80))
                font_small = QFont("Segoe UI", 10)
                painter.setFont(font_small)
                painter.drawText(
                    self.rect().adjusted(0, 32, 0, 0),
                    Qt.AlignmentFlag.AlignCenter,
                    "Bitte warten – Audio und Defektanalyse werden vorbereitet",
                )
                return

            # Willkommens-Screen: animierte Drop-Zone-Anleitung
            # Hintergrund-Rahmen (gestrichelt, animiert wirkend)
            painter.setPen(QPen(QColor(102, 126, 234, 100), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(20, 20, w - 40, h - 40, 18, 18)

            # Haupttext
            painter.setPen(QColor(180, 190, 220))
            font_big = QFont("Segoe UI", 16, QFont.Weight.Bold)
            painter.setFont(font_big)
            painter.drawText(
                self.rect().adjusted(0, -30, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "🎵  Musikdatei hierher ziehen",
            )
            # Subtext
            painter.setPen(QColor(130, 150, 180))
            font_small = QFont("Segoe UI", 10)
            painter.setFont(font_small)
            painter.drawText(
                self.rect().adjusted(0, 38, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "oder oben auf  📂 Audio-Datei öffnen  klicken",
            )
            # Formate
            painter.setPen(QColor(90, 110, 140))
            font_tiny = QFont("Segoe UI", 8)
            painter.setFont(font_tiny)
            painter.drawText(
                self.rect().adjusted(0, 72, 0, 0),
                Qt.AlignmentFlag.AlignCenter,
                "MP3 · WAV · FLAC · AAC · OGG · AIFF · WMA",
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
        s_end   = max(s_start + 1, int(self._view_end * n_total))

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
            painter.drawText(x - 40, center_y - 10, 30, 20,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
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
        painter.drawText(x - 40, center_y - 10, 30, 20,
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

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
            painter.drawText(x - 45, int(y_pos) - 8, 40, 16,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{db}dB")

        # ── Peak dBFS annotation (top-left of channel) ────────────────────────
        peak_db = 20.0 * np.log10(peak_amplitude)
        painter.setPen(QColor(200, 220, 255, 170))
        font_peak = QFont("Segoe UI", 7)
        painter.setFont(font_peak)
        painter.drawText(x + 4, y + 2, width - 8, 14,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"↑ {peak_db:.1f} dBFS")

    def _draw_time_axis(self, painter, x, y, width):
        """Draw time axis with markers — zoom/pan aware."""
        if self.audio_data is None or self.sample_rate == 0:
            return

        n_total = self.audio_data.shape[0]
        total_duration = n_total / self.sample_rate
        view_start_sec = self._view_start * total_duration
        view_end_sec   = self._view_end   * total_duration
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
            painter.drawText(int(x_pos - 30), int(y + 8), 60, 20,
                             Qt.AlignmentFlag.AlignCenter, f"{time_sec:.1f}s")

        # Zoom indicator: show current zoom level if not at 100 %
        if self._view_end - self._view_start < 0.99:
            zoom_factor = 1.0 / max(0.001, self._view_end - self._view_start)
            painter.setPen(QColor(102, 126, 234, 180))
            font_z = QFont("Segoe UI", 7)
            painter.setFont(font_z)
            painter.drawText(int(x + width - 80), int(y + 8), 78, 16,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"🔍 ×{zoom_factor:.1f}")

    def _draw_defect_overlay(self, painter, x, y, width, height):
        """Vertical defect markers at exact time positions + bottom legend.

        Renders semi-transparent colored vertical spans for each localized defect
        event. For global/continuous defects (no timed locations) a subtle tinted
        background strip across the full waveform is drawn instead.  A compact
        legend with color swatches is shown at the bottom of the plot area.

        The view window (_view_start / _view_end) is respected so markers remain
        accurate after zoom/pan.
        """
        if not self.defects or self.audio_data is None:
            return

        _DEFECT_COLORS = {
            "clicks":                     QColor(255,  82,  82),
            "crackle":                    QColor(255, 152,   0),
            "pops":                       QColor(255, 193,   7),
            "clipping":                   QColor(220,  50,  30),
            "hum":                        QColor(156,  39, 176),
            "noise_level":                QColor(100, 181, 246),
            "noise":                      QColor( 80, 160, 230),
            "sibilance":                  QColor(  0, 188, 212),
            "dropout":                    QColor(233,  30,  99),
            "wow":                        QColor( 76, 175,  80),
            "flutter":                    QColor(129, 199, 132),
            "rumble":                     QColor( 96, 125, 139),
            "dc_offset":                  QColor(200, 200,  80),
            "digital_artifacts":          QColor(255, 111,   0),
            "compression_artifacts":      QColor(255,  87, 162),
            "stereo_imbalance":           QColor( 29, 233, 182),
            "phase_issues":               QColor(  0, 137, 123),
            "bandwidth_loss":             QColor(121,  85, 196),
            "pitch_drift":                QColor(255, 214,   0),
            "reverb_excess":              QColor( 63, 137, 199),
            "print_through":              QColor(161, 136, 127),
            "quantization_noise":         QColor( 84, 110, 122),
            "jitter_artifacts":           QColor(230, 238, 156),
            "dynamic_compression_excess": QColor(244,  67,  54),
            "pre_echo":                   QColor(240,  98, 146),
            "transient_smearing":         QColor(255, 167,  38),
            "head_wear":                  QColor(188, 170, 164),
            "riaa_curve_error":           QColor( 77, 182, 172),
            "aliasing":                   QColor(171,  71, 188),
            "bias_error":                 QColor(255, 112,  67),
        }
        _DEFECT_LABELS = {
            "clicks": "Knackser", "crackle": "Knacken", "pops": "Impulse",
            "clipping": "Übersteuerung", "hum": "Brummen",
            "noise_level": "Rauschen", "noise": "Rauschen",
            "sibilance": "Zischlaut", "dropout": "Aussetzer",
            "wow": "Wow", "flutter": "Flutter", "rumble": "Rumpeln",
            "dc_offset": "DC-Offset", "digital_artifacts": "Dig.Artefakt",
            "compression_artifacts": "Codec", "stereo_imbalance": "Stereo-Balance",
            "phase_issues": "Phase", "bandwidth_loss": "Bandbreite",
            "pitch_drift": "Tonhöhe", "reverb_excess": "Hall",
            "print_through": "Bandüberspr.", "quantization_noise": "Quantis.",
            "jitter_artifacts": "Jitter", "dynamic_compression_excess": "Kompression",
            "pre_echo": "Pre-Echo", "transient_smearing": "Transient",
            "head_wear": "Kopf-Fehler", "riaa_curve_error": "RIAA",
            "aliasing": "Aliasing", "bias_error": "Vormagnet.",
        }
        _SKIP_KEYS = {"status", "_locations"}

        # Compute view window in seconds
        n_total = self.audio_data.shape[0]
        total_dur = n_total / max(1, self.sample_rate)
        view_start_s = self._view_start * total_dur
        view_end_s   = self._view_end   * total_dur
        view_dur = max(1e-6, view_end_s - view_start_s)

        # ── Active defect set (above noise floor) ────────────────────────────
        _severity_thresholds = {
            "clicks": 0.5, "crackle": 0.1, "pops": 0.5, "clipping": 0.05,
            "hum": 0.05, "noise_level": 0.1, "noise": 0.1, "sibilance": 0.1,
            "dropout": 0.5, "wow": 0.2, "flutter": 0.2, "rumble": 0.1,
        }
        _DEFAULT_SEVERITY_THRESHOLD = 5.0  # for 0–100 % fields

        active_keys: list[str] = []
        for k, v in self.defects.items():
            if k in _SKIP_KEYS or not isinstance(v, (int, float)):
                continue
            thresh = _severity_thresholds.get(k, _DEFAULT_SEVERITY_THRESHOLD)
            if v >= thresh:
                active_keys.append(k)

        if not active_keys:
            return

        painter.save()

        # ── 1. Vertical timed markers (from _defect_locations) ───────────────
        painter.setPen(Qt.PenStyle.NoPen)
        for defect_key, locations in self._defect_locations.items():
            if not locations:
                continue
            base = _DEFECT_COLORS.get(defect_key, QColor(180, 180, 180))
            for seg in locations:
                if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                    continue
                t_start, t_end = float(seg[0]), float(seg[1])
                # Clip to view window
                if t_end < view_start_s or t_start > view_end_s:
                    continue
                t0 = max(t_start, view_start_s)
                t1 = min(t_end,   view_end_s)
                px0 = int(x + (t0 - view_start_s) / view_dur * width)
                px1 = int(x + (t1 - view_start_s) / view_dur * width)
                px1 = max(px0 + 2, px1)  # minimum 2 px visible width

                # Semi-transparent fill
                fill = QColor(base.red(), base.green(), base.blue(), 45)
                painter.setBrush(QBrush(fill))
                painter.drawRect(px0, int(y), px1 - px0, int(height))

                # Bright vertical edge line at start
                pen_col = QColor(base.red(), base.green(), base.blue(), 190)
                painter.setPen(QPen(pen_col, 1.5))
                painter.drawLine(px0, int(y), px0, int(y + height))
                painter.setPen(Qt.PenStyle.NoPen)

        # ── 2. Global/wide defects: subtle full-width tinted top strip ────────
        #    (shown for active defects that have NO timed locations — e.g. noise,
        #    rumble, hum — as a light background tint so the user knows it is
        #    present throughout the file)
        GLOBAL_DEFECT_KEYS = {
            "hum", "noise_level", "noise", "rumble", "dc_offset",
            "bandwidth_loss", "stereo_imbalance", "phase_issues",
        }
        strip_idx = 0
        for k in active_keys:
            if k not in GLOBAL_DEFECT_KEYS:
                continue
            locs = self._defect_locations.get(k, [])
            if locs:
                continue  # has timed markers — skip global strip
            base = _DEFECT_COLORS.get(k, QColor(180, 180, 180))
            strip_y = int(y) + strip_idx * 3
            if strip_y + 3 > int(y) + height:
                break
            tint = QColor(base.red(), base.green(), base.blue(), 28)
            painter.setBrush(QBrush(tint))
            painter.drawRect(int(x), strip_y, int(width), 3)
            strip_idx += 1

        # ── 3. Legend at bottom of plot area ─────────────────────────────────
        legend_h = 14        # total height of legend row
        swatch_s = 8         # color-swatch square size
        swatch_gap = 4       # gap between swatch and label text
        item_gap = 10        # gap between legend items
        legend_y = int(y + height) - legend_h - 1

        # Measure total legend width to center it
        label_font = QFont("Segoe UI", 7)
        _fm = painter.fontMetrics()
        painter.setFont(label_font)
        _fm = painter.fontMetrics()

        items: list[tuple[str, str, QColor]] = []
        for k in active_keys:
            if k not in _DEFECT_LABELS and k not in _DEFECT_COLORS:
                continue
            label_text = _DEFECT_LABELS.get(k, k)
            color = _DEFECT_COLORS.get(k, QColor(180, 180, 180))
            items.append((k, label_text, color))

        if not items:
            painter.restore()
            return

        # Compute item widths
        item_widths = [swatch_s + swatch_gap + _fm.horizontalAdvance(lbl) for _, lbl, _ in items]
        total_legend_w = sum(item_widths) + item_gap * (len(items) - 1)

        # Start drawing centered
        cur_x = int(x + max(0, (width - total_legend_w) // 2))

        # Semi-transparent legend background
        bg_padding = 4
        bg_rect_x = cur_x - bg_padding
        bg_rect_w = total_legend_w + bg_padding * 2
        bg_rect_y = legend_y - bg_padding
        bg_rect_h = legend_h + bg_padding * 2
        if bg_rect_h > 0 and bg_rect_w > 0:
            painter.setBrush(QBrush(QColor(10, 10, 20, 140)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bg_rect_x, bg_rect_y, bg_rect_w, bg_rect_h, 3, 3)

        for (k, lbl, color), iw in zip(items, item_widths):
            # Color swatch
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 210)))
            painter.setPen(Qt.PenStyle.NoPen)
            swatch_y = legend_y + (legend_h - swatch_s) // 2
            painter.drawRoundedRect(cur_x, swatch_y, swatch_s, swatch_s, 2, 2)

            # Label text
            painter.setPen(QColor(210, 215, 230, 220))
            painter.setFont(label_font)
            painter.drawText(cur_x + swatch_s + swatch_gap, legend_y, iw, legend_h,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             lbl)

            cur_x += iw + item_gap

        painter.restore()


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
            if vmax > vmin:
                Sxx_db_norm = np.clip((Sxx_db - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                Sxx_db_norm = np.zeros_like(Sxx_db)

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
                    if freq >= 1000:
                        label = f"{freq//1000}k"
                    else:
                        label = str(freq)

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
    """Real-time resource status display (CPU, Memory, Mode, ML/DSP Indicators)"""

    def __init__(self):
        super().__init__()
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.quality_mode = "BALANCED"
        self.ml_mode_active = False
        self.active_ml_plugins = []

        self.setMinimumHeight(140)
        self.setStyleSheet("""
            background: rgba(20, 20, 30, 0.95);
            border: 2px solid rgba(102, 126, 234, 0.4);
            border-radius: 10px;
        """)

        # Timer for updating resource stats
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_resources)
        self.update_timer.start(1000)  # Update every 1 second

        self._init_ui()

    def _init_ui(self):
        """Initialize UI layout"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QLabel("⚙️ System-Ressourcen & Verarbeitungs-Modus")
        header.setStyleSheet("color: #667eea; font-size: 12pt; font-weight: bold;")
        layout.addWidget(header)

        # Status labels
        self.label_cpu = QLabel("🖥️  CPU-Auslastung:      [0.0%]")
        self.label_memory = QLabel("💾 Speicher-Auslastung: [0.0%]")
        self.label_mode = QLabel("🎯 Verarbeitungs-Modus:  [BALANCED]")
        self.label_ml_status = QLabel("🤖 ML-Plugins:           [DSP-Modus]")

        for label in [self.label_cpu, self.label_memory, self.label_mode, self.label_ml_status]:
            label.setStyleSheet("color: #AAB8C6; font-family: 'Courier New'; font-size: 10pt;")
            layout.addWidget(label)

    def _update_resources(self):
        """Update resource information periodically"""
        try:
            import psutil

            self.cpu_usage = psutil.cpu_percent(interval=None)
            self.memory_usage = psutil.virtual_memory().percent
            self.update()
        except ImportError:
            pass  # psutil not available

    def update_status(self, cpu=None, memory=None, mode=None, ml_active=None, ml_plugins=None):
        """Update displayed status information"""
        if cpu is not None:
            self.cpu_usage = cpu
        if memory is not None:
            self.memory_usage = memory
        if mode is not None:
            self.quality_mode = mode
        if ml_active is not None:
            self.ml_mode_active = ml_active
        if ml_plugins is not None:
            self.active_ml_plugins = ml_plugins

        # Update labels
        cpu_color = "#00FF7F" if self.cpu_usage < 70 else ("#FFD700" if self.cpu_usage < 90 else "#FF4444")
        mem_color = "#00FF7F" if self.memory_usage < 70 else ("#FFD700" if self.memory_usage < 90 else "#FF4444")

        self.label_cpu.setText(f"🖥️  CPU-Auslastung:      [{self.cpu_usage:.1f}%]")
        self.label_cpu.setStyleSheet(f"color: {cpu_color}; font-family: 'Courier New'; font-size: 10pt;")

        self.label_memory.setText(f"💾 Speicher-Auslastung: [{self.memory_usage:.1f}%]")
        self.label_memory.setStyleSheet(f"color: {mem_color}; font-family: 'Courier New'; font-size: 10pt;")

        mode_icon = "⚡" if self.quality_mode == "FAST" else ("⚖️" if self.quality_mode == "BALANCED" else "💎")
        self.label_mode.setText(f"🎯 Verarbeitungs-Modus:  [{mode_icon} {self.quality_mode}]")

        if self.ml_mode_active and self.active_ml_plugins:
            plugins_str = ", ".join(self.active_ml_plugins[:2])  # Show first 2
            if len(self.active_ml_plugins) > 2:
                plugins_str += f" +{len(self.active_ml_plugins)-2}"
            self.label_ml_status.setText(f"🤖 ML-Plugins:           [{plugins_str}]")
            self.label_ml_status.setStyleSheet("color: #00FF7F; font-family: 'Courier New'; font-size: 10pt;")
        else:
            self.label_ml_status.setText("🤖 ML-Plugins:           [DSP-Modus]")
            self.label_ml_status.setStyleSheet("color: #AAB8C6; font-family: 'Courier New'; font-size: 10pt;")


class DefectCounterWidget(QWidget):
    """Animated defect counter display with two-phase animation"""

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
        header = QLabel("⚠️ Erkannte Defekte & Korrekturen")
        header.setStyleSheet("color: #FFA500; font-size: 12pt; font-weight: bold;")
        layout.addWidget(header)

        # Counter labels
        self.label_clicks = QLabel("⚡ Knackser:             [0] 🔍 ERKENNE")
        self.label_crackle = QLabel("🧻 Knistern:             [0] 🔍 ERKENNE")
        self.label_pops = QLabel("💥 Pops:                 [0] 🔍 ERKENNE")
        self.label_clipping = QLabel("🔊 Übersteuerung:        [0] 🔍 ERKENNE")
        self.label_hum = QLabel("🔌 Brummen:              [0.00Hz] 🔍 ERKENNE")
        self.label_noise = QLabel("🌀 Rauschen:             [0.00dB] 🔍 ERKENNE")
        self.label_sibilance = QLabel("🎤 Sibilanzen:           [0] 🔍 ERKENNE")
        self.label_dropout = QLabel("📍 Aussetzer:            [0] 🔍 ERKENNE")
        self.label_wow = QLabel("🎚️ Wow (<0.5 Hz):        [0.00%] 🔍 ERKENNE")
        self.label_flutter = QLabel("🎚️ Flutter (0.5–200 Hz): [0.00%] 🔍 ERKENNE")

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
            status_icon = "🔍 ERKENNE"
            status_color = "#88AAFF"
        elif self.phase == "correcting":
            status_icon = "⚙️ BEARBEITE"
            status_color = "#FFA500"
        elif self.phase == "completed":
            status_icon = "✓ BEREINIGT"
            status_color = "#4CAF50"
        else:
            status_icon = "🔍 ERKENNE"
            status_color = "#88AAFF"

        # Update labels
        self.label_clicks.setText(f"⚡ Knackser:             [{self.defects['clicks']:,}] {status_icon}")
        self.label_crackle.setText(f"🧻 Knistern:             [{self.defects['crackle']:,}] {status_icon}")
        self.label_pops.setText(f"💥 Pops:                 [{self.defects['pops']:,}] {status_icon}")
        self.label_clipping.setText(f"🔊 Übersteuerung:        [{self.defects['clipping']:,}] {status_icon}")
        self.label_hum.setText(f"🔌 Brummen:              [{self.defects['hum']:.2f}Hz] {status_icon}")
        self.label_noise.setText(f"🌀 Rauschen:             [-{self.defects['noise_level']:.2f}dB] {status_icon}")
        self.label_sibilance.setText(f"🎤 Sibilanzen:           [{self.defects['sibilance']:,}] {status_icon}")
        self.label_dropout.setText(f"📍 Aussetzer:            [{self.defects['dropout']:,}] {status_icon}")
        self.label_wow.setText(f"🎚️ Wow (<0.5 Hz):        [{self.defects['wow']:.2f}%] {status_icon}")
        self.label_flutter.setText(f"🎚️ Flutter (0.5–200 Hz): [{self.defects['flutter']:.2f}%] {status_icon}")

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
        self.title_label = QLabel("Aurik 9.10.51 für meinen lieben Freund Dieter Schönemann")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.title_label)

        # Stretch
        layout.addStretch()

        # Status Indicator (versteckt – Status wird in der unteren Leiste angezeigt)
        self.status_label = QLabel("")
        self.status_label.setVisible(False)

        # Window Controls
        self.btn_minimize = self._create_control_button("−", self.minimize_clicked)
        self.btn_maximize = self._create_control_button("□", self.maximize_clicked)
        self.btn_close = self._create_control_button("×", self.close_clicked)

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
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
            if not self.is_maximized:
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

    def set_status(self, text, color="#88AAFF"):
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
        self.setAutoFillBackground(False)
        self._pixmap: "QPixmap | None" = None
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
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 16px; }"
        )
        # Glow drop-shadow, driven by _glow_timer
        self._glow_fx = QGraphicsDropShadowEffect(self)
        self._glow_fx.setBlurRadius(0)
        self._glow_fx.setOffset(0, 0)
        self._glow_fx.setColor(QColor(
            self._glow_color_base.red(),
            self._glow_color_base.green(),
            self._glow_color_base.blue(), 0,
        ))
        self.setGraphicsEffect(self._glow_fx)

        self._glow_timer = QTimer(self)
        self._glow_timer.setInterval(16)  # ~60 fps
        self._glow_timer.timeout.connect(self._tick_glow)

    # ── Public helper ──────────────────────────────────────────────────────
    def reattach_glow(self) -> None:
        """Re-attach glow effect after an external setGraphicsEffect() cleared it."""
        self._glow_alpha = 0
        self._glow_fx.setBlurRadius(0)
        self._glow_fx.setColor(QColor(
            self._glow_color_base.red(),
            self._glow_color_base.green(),
            self._glow_color_base.blue(), 0,
        ))
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
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()
        radius = 16.0
        pressed = self._btn_pressed

        if not self._pixmap or self._pixmap.isNull():
            # No image: fall back to default QPushButton rendering (uses setStyleSheet)
            painter.end()
            super().paintEvent(event)
            return

        # ── 1. Image (object-fit:cover, rounded clip) ──────────────────────
        scaled = self._pixmap.scaled(
            rect.width(), rect.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        ox = (rect.width()  - scaled.width())  // 2
        oy = (rect.height() - scaled.height()) // 2
        painter.save()
        clip = QPainterPath()
        clip.addRoundedRect(
            float(rect.x()), float(rect.y()),
            float(rect.width()), float(rect.height()),
            radius, radius,
        )
        painter.setClipPath(clip)
        # Pressed: shift image 2 px down-right to simulate physical depth
        painter.drawPixmap(ox + (2 if pressed else 0), oy + (2 if pressed else 0), scaled)
        painter.restore()

        # ── 2. 3-D bevel overlay ───────────────────────────────────────────
        if not pressed:
            # Top shimmer highlight (top 28 % of height, white fade-out)
            tg = QLinearGradient(
                0.0, float(rect.top()),
                0.0, float(rect.top() + rect.height() * 0.28),
            )
            tg.setColorAt(0.0, QColor(255, 255, 255, 62))
            tg.setColorAt(1.0, QColor(255, 255, 255,  0))
            tg_path = QPainterPath()
            tg_path.addRoundedRect(
                float(rect.x()), float(rect.y()),
                float(rect.width()), float(rect.height() * 0.50),
                radius, radius,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(tg))
            painter.drawPath(tg_path)

            # Rim border: light top → dark bottom  (raised-button illusion)
            rim = QLinearGradient(0.0, float(rect.top()), 0.0, float(rect.bottom()))
            rim.setColorAt(0.00, QColor(255, 255, 255, 115))
            rim.setColorAt(0.42, QColor(255, 255, 255,  20))
            rim.setColorAt(0.58, QColor(  0,   0,   0,  20))
            rim.setColorAt(1.00, QColor(  0,   0,   0, 135))
            painter.setPen(QPen(QBrush(rim), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius - 1, radius - 1)
        else:
            # Pressed: darken + inner top-shadow + inverted rim (sunken illusion)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 80)))
            painter.drawRoundedRect(rect, radius, radius)

            ig = QLinearGradient(0.0, float(rect.top()), 0.0, float(rect.top() + 30))
            ig.setColorAt(0.0, QColor(0, 0, 0, 145))
            ig.setColorAt(1.0, QColor(0, 0, 0,   0))
            painter.setBrush(QBrush(ig))
            painter.drawRoundedRect(rect, radius, radius)

            rim2 = QLinearGradient(0.0, float(rect.top()), 0.0, float(rect.bottom()))
            rim2.setColorAt(0.0, QColor(  0,   0,   0, 115))
            rim2.setColorAt(1.0, QColor(255, 255, 255,  38))
            painter.setPen(QPen(QBrush(rim2), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius - 1, radius - 1)

        # ── 3. Disabled overlay ─────────────────────────────────────────────
        if not self.isEnabled():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 145)))
            painter.drawRoundedRect(rect, radius, radius)

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
                    stop:0   #1c2e1c,
                    stop:0.45 #101e10,
                    stop:0.55 #101e10,
                    stop:1   #1c2e1c);
                border: 2px solid qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8aaa8a, stop:0.5 #3a6a3a, stop:1 #8aaa8a);
                border-radius: 15px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 11pt;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0.00 #d4f7d4,
                    stop:0.08 #7ee87e,
                    stop:0.22 #44d044,
                    stop:0.45 #22b022,
                    stop:0.55 #1a981a,
                    stop:0.78 #0e760e,
                    stop:0.92 #0a5c0a,
                    stop:1.00 #083808);
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

    def paintEvent(self, event):  # noqa: N802
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
        self.setWindowTitle("💾  Exporteinstellungen")
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
        default_dir  = str(src.parent)
        default_name = src.stem + "_restauriert"

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 22)

        # Titel
        title = QLabel("💾  Exportdatei festlegen")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #FFFFFF; margin-bottom: 4px;")
        layout.addWidget(title)

        sub = QLabel(f"Quelldatei: {src.name}")
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
        dir_btn.setToolTip("Speicherordner wählen")
        dir_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(self._dir_edit)
        dir_row.addWidget(dir_btn)
        form.addRow("Speicherort:", dir_row)

        # Dateiname
        self._name_edit = QLineEdit(default_name)
        form.addRow("Dateiname:", self._name_edit)

        # Format
        self._fmt_combo = QComboBox()
        for label, _, _ in self.FORMATS:
            self._fmt_combo.addItem(label)
        form.addRow("Format:", self._fmt_combo)

        layout.addLayout(form)

        # Vorschau
        self._preview_lbl = QLabel()
        self._preview_lbl.setStyleSheet(
            "color: #78909C; font-size: 9pt; padding: 6px 0 0 0;"
        )
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
        self._btn_cancel = QPushButton("Abbrechen")
        self._btn_cancel.setFixedWidth(120)
        self._btn_cancel.setStyleSheet(
            "background: rgba(255,255,255,0.09); color: #AAAAAA;"
            "border-radius: 8px; padding: 8px 14px;"
        )
        self._btn_cancel.clicked.connect(self.reject)

        self._btn_ok = QPushButton("✅  Weiter zur Defektanalyse")
        self._btn_ok.setFixedWidth(220)
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(self._btn_cancel)
        btn_row.addSpacing(10)
        btn_row.addWidget(self._btn_ok)
        layout.addLayout(btn_row)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Speicherort wählen", self._dir_edit.text()
        )
        if d:
            self._dir_edit.setText(d)

    def _refresh(self):
        idx = self._fmt_combo.currentIndex()
        ext = self.FORMATS[idx][2]
        name = (self._name_edit.text().strip() or "output") + ext
        full = str(Path(self._dir_edit.text().strip() or ".") / name)
        self._preview_lbl.setText(f"➡️  Ausgabedatei: {full}")

    def get_config(self) -> dict:
        """Gibt gewählte Einstellungen zurück."""
        idx = self._fmt_combo.currentIndex()
        _, fmt_key, ext = self.FORMATS[idx]
        name = (self._name_edit.text().strip() or "output") + ext
        out_dir = self._dir_edit.text().strip() or str(Path(self._source_path).parent)
        return {
            "output_dir": out_dir,
            "filename":   name,
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
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Window properties
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
            lambda: threading.Thread(
                target=_warmup_models_background, daemon=True, name="AurikWarmup"
            ).start(),
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
                self.status_text.setText("🎵 Lyrics-Timeline-Overlay ausgeblendet")
            return

        # Overlay einblenden: Transkription im Hintergrund starten
        if self._orig_audio is None:
            if hasattr(self, "status_text"):
                self.status_text.setText("🎵 Lyrics-Timeline: Bitte zuerst eine Audiodatei laden.")
            self._lyrics_overlay_visible = False
            return

        if hasattr(self, "status_text"):
            self.status_text.setText("🎵 Lyrics-Timeline-Overlay: Transkription läuft …")

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
                        src = "DSP-Fallback" if transcription.fallback_used else "Whisper-Tiny"
                        _self.status_text.setText(
                            f"🎵 Lyrics-Timeline eingeblendet  ·  {n} Segmente  ·  Quelle: {src}"
                        )

                _self._dispatch_to_gui(_apply)
            except Exception as _exc:
                logger.warning("LyricsGuided-Overlay: Transkription fehlgeschlagen: %s", _exc)

                def _err():
                    if hasattr(_self, "status_text"):
                        _self.status_text.setText("🎵 Lyrics-Timeline: Transkription nicht verfügbar.")
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
        self.setCentralWidget(self.main_container)

        outer = QVBoxLayout(self.main_container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Titelleiste ───────────────────────────────────────────────
        self.title_bar = ModernTitleBar(self)
        self.title_bar.minimize_clicked.connect(self.showMinimized)
        self.title_bar.maximize_clicked.connect(self._toggle_maximize)
        self.title_bar.close_clicked.connect(self.close)
        outer.addWidget(self.title_bar)

        # ── Körper: linkes Panel ◀ | ▶ Hauptbereich ──────────────────
        body = QWidget()
        body.setObjectName("bodyWidget")
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
        Aufnahme:          ▸ detected_medium_label
        erkannte Tonträger: ▸ restorability_banner
        erkannte Defekte:  ▸ defect_summary_label
        Musikalische Ziele: ▸ radar_widget + quality_score_label
        """
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setObjectName("leftPanel")
        panel.setStyleSheet("""
            QWidget#leftPanel {
                background: rgba(8, 10, 24, 0.98);
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

        # ── Erkannte Defekte + Echtzeit-Zähler ──────────────────────────────
        # Echtzeit-Zähler-Label: wird während des Scans live aktualisiert
        self.defect_count_live_label = QLabel("")
        self.defect_count_live_label.setWordWrap(False)
        self.defect_count_live_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.defect_count_live_label.setStyleSheet(
            "color: #90A4AE; font-size: 8pt; background: transparent; padding: 0 2px;"
        )
        self.defect_count_live_label.setVisible(False)

        # Header-Zeile: "erkannte Defekte" links, live-Zähler rechts
        _defect_header = QWidget()
        _dh_row = QHBoxLayout(_defect_header)
        _dh_row.setContentsMargins(0, 0, 0, 0)
        _dh_row.setSpacing(4)
        _dh_title_lbl = QLabel("erkannte Defekte:")
        _dh_title_lbl.setStyleSheet("color: #7080A0; font-size: 8pt; background: transparent;")
        _dh_row.addWidget(_dh_title_lbl)
        _dh_row.addStretch()
        _dh_row.addWidget(self.defect_count_live_label)

        self.defect_summary_label = QLabel(t("ui.no_analysis"))
        self.defect_summary_label.setWordWrap(True)
        self.defect_summary_label.setMinimumHeight(180)
        self.defect_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.defect_summary_label.setStyleSheet("""
            color: #C8A84B; font-size: 9pt; padding: 8px 10px;
            background: rgba(255, 165, 0, 0.09);
            border-radius: 8px; border: 1px solid rgba(255, 165, 0, 0.22);
        """)
        _defect_container = QWidget()
        _dc_vbox = QVBoxLayout(_defect_container)
        _dc_vbox.setContentsMargins(0, 0, 0, 0)
        _dc_vbox.setSpacing(2)
        _dc_vbox.addWidget(_defect_header)
        _dc_vbox.addWidget(self.defect_summary_label)
        layout.addWidget(_defect_container, 3)

        # Interne Widgets (verborgen, nur für Datenverarbeitung)
        self.defect_counter_widget = DefectCounterWidget()
        self.defect_counter_widget.setVisible(False)
        self.resource_status_widget = ResourceStatusWidget()
        layout.addWidget(_section("Systemstatus:", self.resource_status_widget), 1)

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
            "stop:0 #43A047,stop:1 #2E7D32);border:none;border-radius:7px;"
            "color:white;font-size:9pt;font-weight:bold;padding:7px 10px;}"
            "QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #66BB6A,stop:1 #388E3C);}"
            "QPushButton:disabled{background:rgba(80,80,80,0.4);color:#666;}"
        )
        _ab_style_stop = (
            "QPushButton{background:rgba(244,67,54,0.65);border:none;border-radius:7px;"
            "color:white;font-size:9pt;font-weight:bold;padding:7px 10px;}"
            "QPushButton:hover{background:rgba(244,67,54,0.90);}"
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
        self.btn_stop_playback.clicked.connect(self._stop_playback)
        ab_row_layout.addWidget(self.btn_stop_playback)

        ab_inner.addWidget(ab_row)
        layout.addWidget(ab_card)

        # ── Magic Buttons ─────────────────────────────────────────────
        layout.addWidget(self._create_magic_buttons_section())

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
                    from PyQt5.QtWidgets import QGraphicsOpacityEffect  # noqa: PLC0415
                    _eff = QGraphicsOpacityEffect(_btn)
                    _eff.setOpacity(0.30)
                    _btn.setGraphicsEffect(_eff)

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
                inner = QVBoxLayout(self)
                inner.setContentsMargins(0, 0, 0, 0)
                inner.addWidget(btn)
                from PyQt5.QtWidgets import QSizePolicy as _QSP

                sp = _QSP(_QSP.Policy.Expanding, _QSP.Policy.Preferred)
                sp.setHeightForWidth(True)
                self.setSizePolicy(sp)

            def hasHeightForWidth(self) -> bool:  # noqa: N802
                return True

            def heightForWidth(self, w: int) -> int:  # noqa: N802
                return max(80, int(w * self._ratio))

            def resizeEvent(self, event) -> None:  # noqa: N802
                super().resizeEvent(event)
                target_h = self.heightForWidth(self.width())
                if self.height() != target_h:
                    self.setFixedHeight(target_h)

        # Äußerer Container mit HBox (zentriert, max. Breite begrenzt)
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(12)
        row.addStretch(1)

        # ── Restoration Button ──────────────────────────────────────────────
        self.btn_magic_restoration = MagicImageButton(
            image_path=_img_r if _img_r.exists() else None,
            hover_color=(118, 75, 162, 191),
            pressed_color=(80, 40, 120, 242),
            glow_color=(118, 75, 162),   # violet glow
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
            glow_color=(255, 165, 0),    # golden glow
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

        return container

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
        self.status_text.setStyleSheet("color: #88AAFF; font-size: 10pt; background: transparent;")
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
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
            }
            QWidget#mainContainer {
                background: #080a18;
                border-radius: 15px;
            }
            QWidget#bodyWidget {
                background: transparent;
            }
            QWidget#mainArea {
                background: transparent;
            }
            QLabel {
                color: #FFFFFF;
            }
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.1);
                width: 2px;
            }
        """)

    def _center_window(self):
        """Center window on screen"""
        _screen = QApplication.primaryScreen()
        if _screen is None:
            return
        screen = _screen.geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

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
        _bind_shortcut(Qt.Key.Key_A, lambda: self._play_audio(self._orig_audio, self._orig_sr) if self._orig_audio is not None else None)
        _bind_shortcut(Qt.Key.Key_B, lambda: self._play_audio(self._rest_audio, self._rest_sr) if self._rest_audio is not None else None)
        _bind_shortcut(QKeySequence.StandardKey.Open, self._open_file)
        _bind_shortcut(QKeySequence.StandardKey.Save, self._export_all)
        _bind_shortcut("Ctrl+R", lambda: self._process_with_mode("RESTORATION"))
        _bind_shortcut("Ctrl+Shift+R", lambda: self._process_with_mode("STUDIO_2026"))
        _bind_shortcut(Qt.Key.Key_Escape, self._cancel_processing)
        _bind_shortcut(QKeySequence.StandardKey.Undo, self._copy_last_output_to_clipboard)
        # L-Shortcut: Lyrics-Timeline-Overlay an/aus (§11.4 Spec 08)
        _bind_shortcut(Qt.Key.Key_L, self._toggle_lyrics_overlay)

    def _toggle_playback(self):
        """Leertaste: Play/Pause — Wiedergabe stoppen oder Original starten."""
        if not _SD_AVAILABLE:
            return
        if self._play_thread is not None and self._play_thread.is_alive():
            try:
                if _sd is not None:
                    _sd.stop()
            except Exception:
                pass
            return
        if self._orig_audio is not None:
            self._play_audio(self._orig_audio, self._orig_sr)

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
        self.title_bar.set_status("Verarbeitung abgebrochen", "#FF5252")
        self.status_text.setText("⏹ Verarbeitung wurde abgebrochen.")

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
            self.status_text.setText(f"📋 Pfad kopiert: {Path(last_output).name}")
        else:
            self.status_text.setText("⚠️ Noch kein Export-Pfad vorhanden.")

    def _update_mode_info(self):
        """Update mode description based on selection"""
        mode = self.mode_combo.currentText()

        if "Highend Studio 2026" in mode:
            self.mode_info_label.setText(
                "🎯 Modern & streaming-optimiert\n"
                "• Maximale Brillanz & Klarheit\n"
                "• Zeitgenössischer Sound\n"
                "• Streaming-ready"
            )
        elif "Restoration" in mode:
            self.mode_info_label.setText(
                "💿 Authentisch & behutsam\n"
                "• Erhalt des Original-Charakters\n"
                "• Moderate Bearbeitung\n"
                "• Archivierungs-geeignet"
            )

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
                    color: #4CAF50; font-size: 12pt; font-weight: bold; padding: 15px;
                    background: rgba(76, 175, 80, 0.25);
                    border-radius: 12px;
                    border: 3px dashed rgba(76, 175, 80, 1.0);
                """)
                self.detected_medium_label.setText("🎵  Datei loslassen, um sie zu laden …")
                # Waveform-Bereich ebenfalls hervorheben
                if hasattr(self, "waveform_widget"):
                    self.waveform_widget.setStyleSheet("border: 3px dashed rgba(76,175,80,0.85); border-radius: 12px;")
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Drag & Drop: Rahmen zurücksetzen wenn Datei das Fenster verlässt."""
        self.detected_medium_label.setText("")
        self.detected_medium_label.setStyleSheet("""
            color: #88AAFF; font-size: 11pt; padding: 15px;
            background: rgba(102, 126, 234, 0.15);
            border-radius: 10px;
            border: 2px solid rgba(102, 126, 234, 0.3);
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
        # Lyrics-Overlay-Zustand zurücksetzen (neue Datei → kein altes Transkript)
        self._lyrics_overlay_visible = False
        if hasattr(self, "waveform_widget"):
            self.waveform_widget._lyrics_transcription = None

        # Sofortiges visuelles Feedback im Haupt-Thread (BEVOR der Hintergrundthread startet)
        self.status_text.setText(f"📂  Wird geladen: {Path(file_path).name} …")
        self.status_text.setStyleSheet("color: #FFC107; font-size: 10pt;")
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

        # ── PFLICHT-GATE §10.5: AudioFileValidator vor dem Laden ─────────────
        # Zugriff ausschließlich über Bridge (§11.4 — kein direkter core/-Import)
        try:
            _validator = _bridge_get_audio_file_validator()
            if _validator is None:
                raise RuntimeError(
                    "Backend-Bridge nicht verfügbar. "
                    "Datei-Validierung ohne Bridge ist im Frontend deaktiviert."
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
                "Datei ungültig",
                f"Diese Datei kann nicht geladen werden:\n\n{_user_msg}",
            )
            if hasattr(self, "detected_medium_label"):
                self.detected_medium_label.setText(
                    f"⚠️ Ungültige Datei: {Path(file_path).name}"
                )
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
                            f"⚠️ Datei konnte nicht geladen werden: {Path(file_path).name}\n{_msg}"
                        )
                        self.detected_medium_label.setStyleSheet("""
                            color: #FF5252; font-size: 11pt; padding: 12px;
                            background: rgba(255, 82, 82, 0.15);
                            border-radius: 8px; border: 2px solid rgba(255, 82, 82, 0.3);
                        """)
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Import fehlgeschlagen",
                        f"Die Datei »{Path(file_path).name}« konnte nicht geladen werden.\n\n"
                        f"Unterstützte Formate: WAV, FLAC, OGG, AIFF, MP3, M4A, WMA, AAC\n\n"
                        f"Details: {_msg[:300]}",
                    )

                QTimer.singleShot(0, _err)
                return

            # Laden vollständig — Balken auf 100 % (letzte Iteration ist bereits 100 %,
            # explizites Setzen als Sicherheitsnetz für librosa-Pfad)
            _set_progress(100)
            logger.debug("_bg_load: progress 100 emitted, dispatching _on_file_loaded")

            # _on_file_loaded im GUI-Thread starten
            _audio_ref = audio
            _sr_ref = int(sr)
            # Carrier-Ergebnis: Platzhalter → wird asynchron nachgefüllt
            self._dispatch_to_gui(
                lambda: self._on_file_loaded(_audio_ref, _sr_ref, file_path, "Wird analysiert …", 0)
            )

            # Carrier-Forensics läuft NACH dem GUI-Update in separatem Thread
            def _carrier_bg(_a=audio, _s=int(sr)):
                # PNG-Icon-Pfad (relativ zu dieser Datei)
                import os as _os
                _ICONS_DIR = _os.path.join(
                    _os.path.dirname(_os.path.dirname(__file__)),
                    "resources", "carrier_icons",
                )

                def _html(icon_key: str, label: str) -> str:
                    _p = _os.path.join(_ICONS_DIR, f"{icon_key}.png")
                    return (
                        f'<img src="file:///{_p}" width="22" height="22" '
                        f'style="vertical-align:middle;">&nbsp;{label}'
                    )

                # (icon_key, Anzeige-Name) je Trägermedium
                _MEDIUM_DATA: dict[str, tuple[str, str]] = {
                    "wax_cylinder":  ("wax_cylinder",  "Wachswalze"),
                    "lacquer_disc":  ("lacquer_disc",   "Lackfolie"),
                    "shellac":       ("shellac",        "Schellack"),
                    "vinyl":         ("vinyl",          "Vinyl"),
                    "wire_recording": ("wire_recording", "Drahtband"),
                    "reel_tape":     ("reel_tape",      "Spulenband"),
                    "tape":          ("tape",           "Magnetband"),
                    "cassette":      ("cassette",       "Kassette"),
                    "dat":           ("dat",            "DAT"),
                    "cd_digital":    ("cd_digital",     "CD"),
                    "cd":            ("cd",             "CD"),
                    "digital":       ("cd_digital",     "Digital"),
                    "minidisc":      ("minidisc",       "MiniDisc"),
                    "mp3_low":       ("mp3_low",        "MP3 (schwach)"),
                    "mp3_high":      ("mp3_high",       "MP3"),
                    "damaged_mp3":   ("damaged_mp3",    "MP3 (defekt)"),
                    "aac":           ("aac",            "AAC"),
                    "streaming":     ("streaming",      "Streaming"),
                    "unknown":       ("unknown",        "Unbekannt"),
                }
                # (icon_key, Anzeige-Name) je Dateicontainer
                _EXT_DATA: dict[str, tuple[str, str]] = {
                    ".mp3":  ("mp3_high",  "MP3"),
                    ".m4a":  ("aac",       "M4A/AAC"),
                    ".aac":  ("aac",       "AAC"),
                    ".ogg":  ("streaming", "OGG"),
                    ".opus": ("streaming", "Opus"),
                    ".wma":  ("streaming", "WMA"),
                    ".flac": ("cd_digital", "FLAC"),
                    ".wav":  ("cd_digital", "WAV"),
                    ".aiff": ("cd_digital", "AIFF"),
                    ".aif":  ("cd_digital", "AIFF"),
                }
                # Analoge/physikalische Ursprungsmedien (Ära 0 + 1)
                _ANALOG_MEDIA = frozenset({
                    "wax_cylinder", "lacquer_disc", "shellac", "vinyl",
                    "wire_recording", "reel_tape", "tape", "cassette",
                })
                _raw_medium = "unknown"
                _score = 0
                try:
                    _classify_medium = _bridge_get_medium_classifier_fn()
                    if callable(_classify_medium):
                        _mono = np.mean(_a, axis=1) if _a.ndim > 1 else _a
                        _res = _classify_medium(_mono, _s)
                        _raw_medium = _res.material_type
                        _score = round(_res.confidence * 5)
                except Exception:
                    pass
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
                self._era_genre_badge = ""  # Badge bei neuem File zurücksetzen
                # Label im GUI-Thread aktualisieren
                self._dispatch_to_gui(
                    lambda l=_lbl, sc=_score: self._update_carrier_display(l, sc, file_path)
                )

            threading.Thread(target=_carrier_bg, daemon=True).start()

        threading.Thread(target=_bg_load, daemon=True).start()

    # ── Thread-sichere GUI-Dispatch-Helfer ─────────────────────────────────
    def _dispatch_to_gui(self, fn) -> None:
        """Ruft `fn()` thread-sicher im GUI-Thread auf (via pyqtSignal)."""
        self._gui_dispatch.emit(fn)

    def _update_carrier_display(self, carrier_label: str, carrier_score: int, file_path: str) -> None:
        """Aktualisiert den Carrier-Label im GUI-Thread (nach async Analyse)."""
        if hasattr(self, "detected_medium_label"):
            _stars = "★" * carrier_score + "☆" * (5 - carrier_score)
            # Era/Genre-Badge mitnehmen, falls bereits berechnet (Race-Condition-Fix)
            _badge = getattr(self, "_era_genre_badge", "")
            self.detected_medium_label.setText(
                f"{carrier_label}   {_stars}{_badge}"
            )
            self.detected_medium_label.setStyleSheet("""
                color: #4CAF50; font-size: 11pt; padding: 12px;
                background: rgba(76, 175, 80, 0.15);
                border-radius: 8px; border: 2px solid rgba(76, 175, 80, 0.3);
                margin-top: 8px; font-weight: 600;
            """)
        if hasattr(self, "current_detected_carrier"):
            self.current_detected_carrier = carrier_label
        if hasattr(self, "current_carrier_confidence"):
            self.current_carrier_confidence = carrier_score

    def _on_file_loaded(self, audio: np.ndarray, sr: int, file_path: str, carrier_label: str, carrier_score: int):
        """Wird im Haupt-Thread aufgerufen, nachdem sf.read + Carrier-Forensics fertig sind."""
        logger.debug("_on_file_loaded called: audio.shape=%s sr=%d", audio.shape, sr)
        # Lade-Zustand des WaveformWidgets aufheben
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.is_loading = False
        # Fortschrittsbalken: 100 % sichtbar anzeigen
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(10000)
            self.progress_bar.setFormat("✅  Import abgeschlossen")

        # Export-Dialog mit kurzem Delay öffnen, damit 100 % kurz gerendert wird
        def _open_export_dialog(
            _audio=audio, _sr=sr, _fp=file_path,
            _cl=carrier_label, _cs=carrier_score,
        ):
            logger.debug("_open_export_dialog: opening ExportConfigDialog")
            _dlg = ExportConfigDialog(_fp, parent=self)
            logger.debug("ExportConfigDialog created, calling exec()")
            if _dlg.exec() != QDialog.DialogCode.Accepted:
                if hasattr(self, "progress_bar"):
                    self.progress_bar.setVisible(False)
                    self.progress_bar.setValue(0)
                if hasattr(self, "status_text"):
                    self.status_text.setText("❌  Import abgebrochen.")
                return
            self._export_config = _dlg.get_config()
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
                self.progress_bar.setValue(0)
            self._continue_file_loaded(_audio, _sr, _fp, _cl, _cs)

        QTimer.singleShot(150, _open_export_dialog)

    def _continue_file_loaded(  # noqa: C901
        self,
        audio: "np.ndarray",
        sr: int,
        file_path: str,
        carrier_label: str,
        carrier_score: int,
    ):
        """Setzt _on_file_loaded nach dem Export-Dialog fort (UI-Block)."""
        _do_load_file_body = True
        _file_path_for_body = file_path  # noqa: F841
        try:  # pylint: disable=too-many-statements
            _audio = audio
            _sr = sr
            if len(_audio.shape) > 1:
                audio_mono = np.mean(_audio, axis=1)
            else:
                audio_mono = _audio
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
                _stars = "★" * confidence + "☆" * (5 - confidence)
                self.detected_medium_label.setText(
                    f"{detected_carrier}   {_stars}"
                )
                self.detected_medium_label.setStyleSheet("""
                    color: #4CAF50; font-size: 11pt; padding: 12px;
                    background: rgba(76, 175, 80, 0.15);
                    border-radius: 8px; border: 2px solid rgba(76, 175, 80, 0.3);
                    margin-top: 8px; font-weight: 600;
                """)
                self.current_detected_carrier = detected_carrier
                self.current_carrier_confidence = confidence

                # ── Ära- und Genre-Erkennung im Hintergrund ────────────────
                # Kein .copy(): audio_mono wird von allen Hintergrundthreads nur gelesen
                _sr2 = int(_sr)
                _base_text = (
                    f"{detected_carrier}   {_stars}"
                )

                def _detect_era_genre_bg(_a=audio_mono, _s=_sr2, _bt=_base_text, _self=self):
                    decade_label = ""
                    genre_label = ""
                    try:
                        _classify_era = _bridge_get_era_classifier_fn()
                        if callable(_classify_era):
                            er = _classify_era(_a, _s)
                            dec = getattr(er, "decade", None) or (er.get("decade") if isinstance(er, dict) else None)
                            if dec:
                                decade_label = f"{dec}er"
                    except Exception:
                        pass
                    try:
                        _classify_genre = _bridge_get_genre_classifier_fn()
                        if callable(_classify_genre):
                            gr = _classify_genre(_a, _s)
                            gl = getattr(gr, "genre_label", None) or (
                                gr.get("genre_label") if isinstance(gr, dict) else None
                            )
                            if gl and gl.lower() not in ("unbekannt", "unknown", ""):
                                genre_label = str(gl)
                    except Exception:
                        pass
                    badge = ""
                    if decade_label:
                        badge = f"  │  ◷ {decade_label}"
                    if genre_label:
                        badge += f" · {genre_label}"
                    if not badge:
                        return  # nichts zu ergänzen
                    _carrier_name = detected_carrier
                    tip = (
                        f"<b>Träger-Forensik &amp; Aufnahme-Epoche</b><br>"
                        f"Erkanntes Medium: <b>{_carrier_name}</b><br>"
                    )
                    if decade_label:
                        tip += f"Aufnahme-Ära: <b>{decade_label}</b><br>"
                    if genre_label:
                        tip += f"Genre: <b>{genre_label}</b><br>"
                    tip += (
                        "<small>Die Ära-Erkennung passt alle " "Restaurierungs-Parameter historisch korrekt an.</small>"
                    )

                    def _upd(_badge=badge, _tip=tip):
                        if not hasattr(_self, "detected_medium_label"):
                            return
                        # Badge für _update_carrier_display merken (Race-Condition-Fix)
                        _self._era_genre_badge = _badge
                        # Aktuellen Carrier-Label lesen (kann inzwischen gesetzt worden sein)
                        _cur_lbl = getattr(_self, "_carrier_bg_label", _bt)
                        _cur_sc  = getattr(_self, "_carrier_bg_score", 0)
                        _cur_stars = "★" * _cur_sc + "☆" * (5 - _cur_sc)
                        _self.detected_medium_label.setText(
                            f"{_cur_lbl}   {_cur_stars}{_badge}"
                        )
                        _self.detected_medium_label.setToolTip(_tip)

                    QTimer.singleShot(0, _upd)

                threading.Thread(target=_detect_era_genre_bg, daemon=True).start()
                # ── Ende Ära/Genre-Hintergrund ────────────────────────────

            except Exception:
                self.detected_medium_label.setText("Wird analysiert …")
                self.detected_medium_label.setStyleSheet("""
                    color: #88AAFF; font-size: 11pt; padding: 12px;
                    background: rgba(102, 126, 234, 0.15);
                    border-radius: 8px; border: 2px solid rgba(102, 126, 234, 0.3);
                """)

            self._update_waveform(_audio, _sr)
            # A/B-Buttons aktivieren
            self._update_ab_player_state()

            # ── Restaurierbarkeit-Vorschau im Hintergrund ──────────────────
            # Nicht-blockierend (<3 s), aktualisiert das Banner über den
            # Magic Buttons sobald das Ergebnis vorliegt.
            # audio_mono wird nur gelesen – kein .copy() (OOM-Schutz)
            _sr_cap = int(_sr)

            def _estimate_restorability_bg(_a=audio_mono, _s=_sr_cap, _self=self):
                try:
                    _RestorabilityEstimator = _bridge_get_restorability_estimator_class()
                    if _RestorabilityEstimator is None:
                        raise ImportError("RestorabilityEstimator nicht verfügbar")
                    r = _RestorabilityEstimator().estimate(_a, _s)
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

                # Kategorie
                if score100 >= 70:
                    bg = "rgba(76,175,80,0.22)"
                    border = "rgba(76,175,80,0.60)"
                    zeile1 = f"🟢  Sehr gut restaurierbar  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Aurik kann diese Aufnahme auf exzellentem Niveau restaurieren."
                elif score100 >= 40:
                    bg = "rgba(255,193,7,0.22)"
                    border = "rgba(255,193,7,0.60)"
                    zeile1 = f"🟡  Mäßig restaurierbar  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Deutliche Verbesserung möglich – Restdefekte können bleiben."
                else:
                    bg = "rgba(244,67,54,0.22)"
                    border = "rgba(244,67,54,0.60)"
                    zeile1 = f"🔴  Stark beschädigt  ({score100:.0f}\u202f/\u202f100)"
                    detail = "Material ist sehr stark beschädigt – " "Aurik holt das physikalisch Mögliche heraus."

                mos_str = f"{predicted_mos:.1f}"
                banner_txt = (
                    f"{zeile1}    ·    "
                    f"Erw. Qualität nach Restaurierung:  {mos_str}\u202f/\u202f5,0 MOS\n"
                    f"{detail}"
                )
                if limiting:
                    banner_txt += f"    (Hauptdefekte: " f"{', '.join(str(d) for d in limiting[:2])})"
                tip = (
                    f"<b>Restaurierbarkeits-Vorschätzung</b><br>"
                    f"Wert: <b>{score100:.0f}\u202f/\u202f100</b><br>"
                    f"Erw. Qualität nach Restaurierung: <b>{mos_str} von 5,0 MOS</b><br>"
                    f"<small>Schnelle Vorab-Analyse des Signals – "
                    f"vor dem eigentlichen Restaurierungsvorgang.</small>"
                )
                css = (
                    f"color:#FFFFFF; font-size:10pt; font-weight:bold;"
                    f" padding:10px 18px; border-radius:10px;"
                    f" background:{bg}; border:2px solid {border};"
                )

                def _update_ui():
                    if hasattr(_self, "restorability_banner"):
                        _self.restorability_banner.setText(banner_txt)
                        _self.restorability_banner.setStyleSheet(css)
                        _self.restorability_banner.setToolTip(tip)
                        _self.restorability_banner.setVisible(True)

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
                self.defect_summary_label.setText("🔄\u2002 Schäden werden analysiert \u2026")
                self.defect_summary_label.setStyleSheet("""
                    color: #90A4AE; font-size: 10pt; padding: 12px;
                    background: rgba(144, 164, 174, 0.10);
                    border-radius: 10px; border: 1px solid rgba(144, 164, 174, 0.25);
                """)
            # Live-Zähler-Label beim Scan-Start sichtbar schalten
            if hasattr(self, "defect_count_live_label"):
                self.defect_count_live_label.setText("🔍 Analysiere…")
                self.defect_count_live_label.setStyleSheet(
                    "color: #90A4AE; font-size: 8pt; background: transparent; padding: 0 2px;"
                )
                self.defect_count_live_label.setVisible(True)

            # Pulsierender Fortschrittsbalken während Defektanalyse
            if hasattr(self, "progress_bar"):
                self.progress_bar.setRange(0, 10000)
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("🔍  Schäden werden analysiert: %p %")
                self.progress_bar.setVisible(True)
            if hasattr(self, "status_text"):
                self.status_text.setText("🔍  Schäden werden analysiert …")
                self.status_text.setStyleSheet("color: #FFC107; font-size: 10pt;")

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
                        _self.defect_count_live_label.setText("🔍 Analysiere…")
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
                            _self.progress_bar.setFormat("✅  Schadensanalyse abgeschlossen")

                            def _reset_progress_if_idle():
                                if _self.batch_thread and _self.batch_thread.isRunning():
                                    return
                                _self.progress_bar.setVisible(False)
                                _self.progress_bar.setValue(0)

                            QTimer.singleShot(1500, _reset_progress_if_idle)
                        if hasattr(_self, "status_text"):
                            _self.status_text.setText("✅  Bereit zur Restaurierung")
                            _self.status_text.setStyleSheet("color: #4CAF50; font-size: 10pt;")
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
            self.detected_medium_label.setText(f"⚠️ Ladefehler: {str(e)[:80]}")
            self.detected_medium_label.setStyleSheet("""
                color: #FF5252; font-size: 11pt; padding: 12px;
                background: rgba(255, 82, 82, 0.15);
                border-radius: 8px; border: 2px solid rgba(255, 82, 82, 0.3);
            """)
        self.title_bar.set_status("Datei geladen", "#4CAF50")
        self.status_text.setText(
            f"✅ Geladen: {Path(file_path).name}  –  " f"🔍 Defekte werden analysiert … Buttons erscheinen gleich."
        )
        # ── Ende _on_file_loaded ───────────────────────────────────────────

    def _play_audio(self, audio: np.ndarray, sr: int):
        """Audiodaten asynchron über sounddevice abspielen."""
        if not _SD_AVAILABLE:
            QMessageBox.information(
                self, "Player", "Für die Vorschau bitte 'sounddevice' installieren:\n  pip install sounddevice"
            )
            return
        # Laufende Wiedergabe stoppen
        try:
            if _sd is not None:
                _sd.stop()
        except Exception:
            pass

        def _play():
            try:
                data = audio.astype(np.float32)
                if data.max() > 1.0 or data.min() < -1.0:
                    data = data / (np.abs(data).max() + 1e-9)
                if _sd is not None:
                    _sd.play(data, samplerate=sr)
                    _sd.wait()
            except Exception:
                pass

        self._play_thread = threading.Thread(target=_play, daemon=True)
        self._play_thread.start()

    def _update_ab_player_state(self):
        """A/B-Player Buttons je nach verfügbaren Audiodaten aktivieren."""
        if hasattr(self, "btn_play_original"):
            self.btn_play_original.setEnabled(self._orig_audio is not None)
        if hasattr(self, "btn_play_restored"):
            self.btn_play_restored.setEnabled(self._rest_audio is not None)

    def _open_file(self):
        """Öffnet den Datei-Dialog und delegiert an _load_file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Audio-Datei öffnen", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a);;All Files (*)"
        )
        if file_path:
            self._load_file(file_path)

    def _batch_import(self):
        """Batch import multiple files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Mehrere Dateien auswählen",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a *.wma);;All Files (*)",
        )
        for path in file_paths:
            self._add_to_queue(path)

        if file_paths:
            self.title_bar.set_status(f"{len(file_paths)} Dateien geladen", "#4CAF50")
            self.status_text.setText(f"Batch-Import: {len(file_paths)} Dateien")

    def _album_import(self):
        """Ganzen Ordner / Album rekursiv importieren.

        Nutzt BatchProcessor.find_audio_files() falls verfügbar,
        sonst eigenes rglob als Fallback.
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Album-Ordner auswählen (alle Unterordner werden eingeschlossen)", "", QFileDialog.Option.ShowDirsOnly
        )
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
                "Album-Import",
                f"Im Ordner '{folder_path.name}' wurden keine Audiodateien gefunden.\n"
                "Unterstützt: wav, mp3, flac, ogg, aiff, m4a, wma",
            )
            return

        # Kurze Vorschau – Nutzer bestätigen lassen
        # Dateien nach Unterordner (= CD/LP-Seite) gruppiert anzeigen
        subdirs = sorted({p.parent.name for p in found})
        subdir_info = ", ".join(subdirs[:5])
        if len(subdirs) > 5:
            subdir_info += f" … (+{len(subdirs)-5} weitere)"

        reply = QMessageBox.question(
            self,
            "Album importieren",
            f"Gefunden: {len(found)} Audiodatei(en)\n"
            f"Unterordner: {subdir_info or '–'}\n\n"
            "Alle Dateien zur Verarbeitungs-Queue hinzufügen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Modus abfragen
        mode_dialog = QDialog(self)
        mode_dialog.setWindowTitle("Verarbeitungsmodus für Album")
        mode_dialog.setMinimumWidth(380)
        d_layout = QVBoxLayout(mode_dialog)
        d_layout.setSpacing(15)
        d_layout.addWidget(QLabel("<b>Welchen Modus für alle Album-Tracks verwenden?</b>"))

        btn_group = QButtonGroup(mode_dialog)
        rb_rest = QRadioButton("💿  Restoration – original-getreu, behutsam")
        rb_rest.setChecked(True)
        rb_studio = QRadioButton("🎯  Studio 2026 – moderner Highend-Klang")
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

        self.title_bar.set_status(f"{added} Tracks aus Album geladen", "#4CAF50")
        self.status_text.setText(f"📀 Album '{folder_path.name}': {added} Dateien in Queue – Magic Button drücken!")

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
        # Map detected carrier to medium type
        carrier_to_medium = {
            "Schallplatte (Vinyl/Schellack)": "VINYL_33",
            "Kassette/Band": "CASSETTE",
            "CD": "CD",
            "MiniDisc": "MINIDISC",
            "DAT": "DAT",
            "Tonband/Reel": "REEL_TO_REEL",
            "Digital": "LOSSLESS",
            "Unbekannt": "AUTO_DETECT",
        }

        # Use already detected carrier info and pass it to settings
        # (V3 auto-detects MaterialType, so this is only used as a hint)
        if hasattr(self, "current_detected_carrier"):
            detected_medium = carrier_to_medium.get(self.current_detected_carrier, "AUTO_DETECT")
        else:
            detected_medium = "AUTO_DETECT"

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
            QMessageBox.warning(self, "Keine Dateien", "Bitte fügen Sie zuerst Dateien zur Queue hinzu.")
            return

        if self.batch_thread and self.batch_thread.isRunning():
            QMessageBox.warning(self, "Verarbeitung läuft", "Verarbeitung läuft bereits!")
            return

        stats = self.batch_queue.get_stats()
        if stats["pending"] == 0:
            QMessageBox.information(self, "Keine ausstehenden Dateien", "Alle Dateien wurden bereits verarbeitet.")
            return

        # Disable process button and Magic Buttons during processing
        if hasattr(self, "btn_process"):
            self.btn_process.setEnabled(False)
        if hasattr(self, "btn_magic_restoration"):
            self.btn_magic_restoration.setEnabled(False)
        if hasattr(self, "btn_magic_studio"):
            self.btn_magic_studio.setEnabled(False)

        # Update status
        self.title_bar.set_status("Verarbeitung läuft...", "#FFA500")
        self.status_text.setStyleSheet("color: #88AAFF; font-size: 10pt; background: transparent;")
        _n_pend = stats["pending"]
        self.status_text.setText(f"Verarbeite {'1 Datei' if _n_pend == 1 else str(_n_pend) + ' Dateien'} …")

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
        self.batch_thread.defect_update.connect(self._update_defects)
        self.batch_thread.phase_update.connect(self._update_phase)
        # Connect resource/mode signals
        self.batch_thread.mode_update.connect(self._update_mode)
        self.batch_thread.ml_status_update.connect(self._update_ml_status)

        # RAM-Sicherheitscheck: mindestens 6 GB verfügbarer Arbeitsspeicher erforderlich.
        # Verhindert OOM-Kills, die das gesamte System einfrieren (Swap nur 2 GB).
        try:
            import psutil as _psutil
            _avail_gb = _psutil.virtual_memory().available / 1024 ** 3
            if _avail_gb < 6.0:
                QMessageBox.critical(
                    self,
                    "Zu wenig Arbeitsspeicher",
                    f"Es stehen nur {_avail_gb:.1f} GB freier RAM zur Verfügung.\n\n"
                    "Aurik benötigt mindestens 6 GB freien RAM für die Restaurierung.\n\n"
                    "Bitte schließen Sie andere Programme (Browser, VS Code …) und "
                    "versuchen Sie es erneut.",
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

        # Watchdog-Timer: feuert wenn Verarbeitung zu lange hängt (z. B. blockierender ONNX-Call).
        # Budget: max(300 s, 600 s × Anzahl Dateien) — 10 min/Datei deckt Pipeline-Limit (120 s/min × 5 min).
        _watchdog_ms = max(300_000, stats["pending"] * 600_000)
        if not hasattr(self, "_watchdog_timer"):
            self._watchdog_timer = QTimer(self)
            self._watchdog_timer.setSingleShot(True)
            self._watchdog_timer.timeout.connect(self._on_watchdog_timeout)
        self._watchdog_timer.start(_watchdog_ms)
        logger.info("Watchdog-Timer gestartet: %.0f s", _watchdog_ms / 1000)

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
        self.title_bar.set_status("Zeitüberschreitung", "#FF5252")
        _msg = (
            "⏰ Die Verarbeitung hat das Zeitlimit überschritten und wurde abgebrochen.\n"
            "Ursache: Ein Verarbeitungsschritt hat nicht reagiert (möglicher ONNX-Deadlock).\n"
            "→ Starten Sie Aurik neu und versuchen Sie es mit einer kürzeren Audiodatei."
        )
        if hasattr(self, "detected_medium_label"):
            self.detected_medium_label.setText(_msg)
        self.status_text.setText("⏰ Zeitüberschreitung — Verarbeitung abgebrochen.")
        QMessageBox.warning(
            self,
            "Zeitüberschreitung",
            "Die Verarbeitung hat das Zeitlimit überschritten und wurde abgebrochen.\n\n"
            "Bitte starten Sie Aurik neu und versuchen Sie es erneut.",
        )

    def _tick_heartbeat(self):
        """Animierter Spinner + Progress-Polling alle 500 ms.

        Fallback für den Fall, dass item_progress-Signale aus irgendeinem Grund
        nicht zugestellt werden: wir lesen item.progress direkt aus der Queue.
        """
        self._heartbeat_dots = (self._heartbeat_dots + 1) % 4
        spinners = ["◐", "◓", "◑", "◒"]
        spin = spinners[self._heartbeat_dots]
        self.title_bar.set_status(f"Verarbeitung läuft  {spin}", "#FFA500")

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

    def _on_item_started(self, item_id):
        """Handle item processing start"""
        item = self.batch_queue.get_item(item_id)
        if item:
            self.status_text.setText(f"Verarbeite: {Path(item.input_file).name}")

    def _on_item_progress(self, item_id, progress):
        """Handle item progress update"""
        # setRange sicherstellen: verhindert Marquee-Modus (range 0-0) der
        # QProgressBar, der setValue() wirkungslos macht.
        self.progress_bar.setRange(0, 10000)
        val = max(100, min(10000, progress * 100))  # mind. 100 (= 1 %) damit Bar sichtbar wird
        self.progress_bar.setValue(val)
        self.progress_bar.setVisible(True)
        logger.debug("[progress] item=%s pct=%d → bar=%d", item_id, progress, val)

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

        self._update_stats()

    def _on_item_finished_with_result(self, item_id, restoration_result):
        """Handle item completion mit RestorationResult — aktualisiert Qualitäts-Radar."""
        item = self.batch_queue.get_item(item_id)
        if item and item.output_file and Path(item.output_file).exists():
            self._compute_and_show_quality(item.output_file, restoration_result=restoration_result)

    def _on_item_error(self, item_id, error_msg):
        """Handle item error — zeigt deutsche Fehlermeldung im UI (Spec §11.4)."""
        item = self.batch_queue.get_item(item_id)
        file_name = Path(item.input_file).name if item else "Datei unbekannt"

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
        _cause = str(error_msg)[:200] if error_msg else "Unbekannter Fehler"
        _file_label = f"\u201e{file_name}\u201c"
        if hasattr(self, "defect_summary_label"):
            self.defect_summary_label.setText(
                f"❌  Fehler bei {_file_label}:\n"
                f"Ursache: {_cause}\n"
                f"\u2192 Prüfen Sie das Protokoll oder versuchen Sie eine andere Datei."
            )
            self.defect_summary_label.setStyleSheet("""
                color: #EF5350; font-size: 9pt; padding: 12px;
                background: rgba(239, 83, 80, 0.12);
                border-radius: 10px; border: 2px solid rgba(239, 83, 80, 0.45);
            """)
        if hasattr(self, "status_text"):
            self.status_text.setStyleSheet("color: #EF5350; font-size: 10pt;")
            self.status_text.setText(f"❌  Verarbeitungsfehler – {file_name}")
        logger.warning("Item-Fehler %s: %s", item_id, error_msg)

        self._update_stats()

    def _on_all_finished(self):
        """Handle all items finished"""
        if hasattr(self, "_watchdog_timer") and self._watchdog_timer.isActive():
            self._watchdog_timer.stop()
        if hasattr(self, "_heartbeat_timer") and self._heartbeat_timer.isActive():
            self._heartbeat_timer.stop()
        # Phase-Overlay ausblenden
        if hasattr(self, "_phase_overlay_label"):
            self._phase_overlay_label.setVisible(False)
        # Stats VOR clear_completed() lesen — danach ist der Zähler 0!
        stats = self.batch_queue.get_stats()

        # Abgeschlossene/fehlgeschlagene Einträge aus der Queue entfernen.
        # Verhindert, dass sich completed-Zähler über mehrere Läufe ansammeln
        # und die Fortschrittsberechnung verfälschen.
        self.batch_queue.clear_completed()
        if hasattr(self, "queue_list"):
            for i in range(self.queue_list.count() - 1, -1, -1):
                list_item = self.queue_list.item(i)
                item_id = list_item.data(Qt.ItemDataRole.UserRole)
                item = self.batch_queue.get_item(item_id)
                if item is None:  # wurde durch clear_completed() entfernt
                    self.queue_list.takeItem(i)
        self._set_magic_buttons_enabled(True)

        self.progress_bar.setValue(10000)  # 100 % = 10000 Einheiten (0.01 %/Einheit)
        if hasattr(self, "btn_process"):
            self.btn_process.setEnabled(True)

        n_ok   = stats["completed"]
        n_fail = stats["failed"]
        _n_ok_str   = "1 Datei" if n_ok   == 1 else f"{n_ok} Dateien"
        _n_fail_str = "1 Datei" if n_fail == 1 else f"{n_fail} Dateien"

        if n_fail == 0:
            self.title_bar.set_status("Abgeschlossen", "#4CAF50")
            self.status_text.setStyleSheet("color: #66BB6A; font-size: 10pt;")
            self.status_text.setText(
                f"✅  {_n_ok_str} erfolgreich restauriert – ▶ Anhören oder 💾 Speichern"
            )
            # defect_summary_label NICHT überschreiben: Defektliste bleibt sichtbar
        else:
            self.title_bar.set_status("Abgeschlossen mit Fehlern", "#FFA500")
            self.status_text.setStyleSheet("color: #FFA500; font-size: 10pt;")
            if n_ok > 0:
                self.status_text.setText(
                    f"⚠️  {_n_ok_str} restauriert, {_n_fail_str} fehlgeschlagen – Protokoll prüfen"
                )
            else:
                self.status_text.setText(
                    f"❌  {_n_fail_str} konnten nicht verarbeitet werden – Protokoll prüfen"
                )

    def _stop_playback(self):
        """Laufende Wiedergabe anhalten."""
        if _SD_AVAILABLE:
            try:
                if _sd is not None:
                    _sd.stop()
            except Exception:
                pass

    def _compute_and_show_quality(self, output_path: str, restoration_result=None):
        """Qualitätsscore im Hintergrund berechnen und Radar-Chart aktualisieren."""

        def _run():
            try:
                rest_audio, rest_sr = sf.read(output_path)
                self._rest_audio = rest_audio
                self._rest_sr = rest_sr

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

                if restoration_result is not None:
                    r = restoration_result
                    # Musical Goals
                    if hasattr(r, "musical_goals") and isinstance(r.musical_goals, dict):
                        musical_goals = r.musical_goals
                    # Realer PQS-MOS
                    if hasattr(r, "pqs_result") and r.pqs_result is not None:
                        if hasattr(r.pqs_result, "mos"):
                            mos_est = float(r.pqs_result.mos)
                    # Adaptive Thresholds
                    if hasattr(r, "adaptive_thresholds") and r.adaptive_thresholds is not None:
                        at = r.adaptive_thresholds
                        if hasattr(at, "thresholds"):
                            adaptive_thresholds = at.thresholds or {}
                        if hasattr(at, "adaptations"):
                            adaptation_reasons = at.adaptations or {}
                    # Goal Applicability
                    if hasattr(r, "goal_applicability") and r.goal_applicability is not None:
                        ga = r.goal_applicability
                        if hasattr(ga, "applicable"):
                            applicable_goals = set(ga.applicable)
                        if hasattr(ga, "reasons"):
                            inapplicable_reasons = ga.reasons or {}
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
                    # Ära & Genre (Grunddaten)
                    if hasattr(r, "era_decade") and r.era_decade:
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

                # ── Schritt 3: Synthetische Goal-Schätzung wenn keine echten Daten ──
                if not musical_goals and self._orig_audio is not None:
                    # Einfache DSP-Heuristik als Platzhalter (besser als 0)
                    musical_goals = {
                        "brillanz": min(1.0, corr * 0.95 + 0.05),
                        "waerme": min(1.0, corr * 0.92 + 0.06),
                        "natuerlichkeit": min(1.0, corr * 0.97 + 0.02),
                        "authentizitaet": min(1.0, corr * 0.94 + 0.04),
                        "emotionalitaet": min(1.0, corr * 0.90 + 0.05),
                        "transparenz": min(1.0, corr * 0.93 + 0.04),
                        "bass_kraft": min(1.0, corr * 0.91 + 0.05),
                        "groove": min(1.0, corr * 0.96 + 0.02),
                        "spatial_depth": min(1.0, corr * 0.88 + 0.07),
                        "timbre_authentizitaet": min(1.0, corr * 0.93 + 0.04),
                        "tonal_center": min(1.0, corr * 0.98 + 0.01),
                        "micro_dynamics": min(1.0, corr * 0.94 + 0.04),
                        "separation_fidelity": min(1.0, corr * 0.89 + 0.06),
                        "artikulation": min(1.0, corr * 0.93 + 0.04),
                    }

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
                    _score_lines.append(f"Konfidenz: {pipeline_confidence * 100:.0f}%  ·  Datei: {Path(output_path).name}")
                else:
                    _score_lines.append(f"Datei: {Path(output_path).name}")
                mos_text = "\n".join(_score_lines)

                # --- Info-Banner (immer befüllt nach Verarbeitung) ---
                banner_sections: list[str] = []

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
                    _qi_str = f"📈  Qualität: {quality_before_score:.0f} → {quality_after_score:.0f} Pkte ({_delta_str})"
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
                    banner_sections.append(f"💡  {pipeline_hint}")

                # Musical-Goals-Verletzungen
                if musical_violations:
                    _viol_map = {
                        "brillanz": "Brillanz", "waerme": "Wärme",
                        "natuerlichkeit": "Natürlichkeit", "authentizitaet": "Authentizität",
                        "emotionalitaet": "Emotionalität", "transparenz": "Transparenz",
                        "bass_kraft": "Bass-Kraft", "groove": "Groove",
                        "spatial_depth": "Raumtiefe", "timbre_authentizitaet": "Timbre",
                        "tonal_center": "Tonales Zentrum", "micro_dynamics": "Mikro-Dynamik",
                        "separation_fidelity": "Separation", "artikulation": "Artikulation",
                    }
                    _viol_de = [str(_viol_map.get(v, v)) for v in musical_violations]
                    banner_sections.append(f"⚠️  Ziele unter Schwellwert: {', '.join(_viol_de)}")

                # PMGG-Warnungen & Ceiling
                if phase_gate_notes:
                    banner_sections.append("⚠️  Einige Verarbeitungsschritte wurden angepasst, um den Klang zu schützen.")
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
                    # Score-Label
                    self.quality_score_label.setText(mos_text)
                    self.quality_score_label.setStyleSheet("""
                        color: #4CAF50; font-size: 9pt; font-weight: bold;
                        padding: 10px; background: rgba(76, 175, 80, 0.10);
                        border-radius: 8px; border: 1px solid rgba(76, 175, 80, 0.35);
                        line-height: 150%;
                    """)
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
                    self._update_ab_player_state()
                    self._update_waveform(self._rest_audio, self._rest_sr)

                QTimer.singleShot(0, _update_gui)

            except Exception as _ex:
                _ex_msg = str(_ex)

                def _show_err(_msg=_ex_msg):
                    self.quality_score_label.setText(f"⚠️ Score-Berechnung fehlgeschlagen: {_msg}")

                QTimer.singleShot(0, _show_err)

        threading.Thread(target=_run, daemon=True).start()

    def _update_waveform(self, audio, sr):
        """Waveform im Haupt-Thread rendern; Spektrogramm im Hintergrundthread berechnen."""
        try:
            if hasattr(self, "waveform_widget"):
                self.waveform_widget.update_waveform(audio, sr)
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

    def _update_defects(self, defects):
        """Update defect counter display and human-readable summary label"""
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
                        f"⚠ {n} Defekt{'e' if n != 1 else ''}"
                    )
                    self.defect_count_live_label.setStyleSheet(
                        "color: #FFA040; font-size: 8pt; background: transparent;"
                        " font-weight: bold; padding: 0 2px;"
                    )
                else:
                    self.defect_count_live_label.setText("✅ Sauber")
                    self.defect_count_live_label.setStyleSheet(
                        "color: #66BB6A; font-size: 8pt; background: transparent;"
                        " font-weight: bold; padding: 0 2px;"
                    )
                self.defect_count_live_label.setVisible(True)
            if not active:
                self.defect_summary_label.setText("✅  Keine Schäden erkannt – gute Aufnahmequalität")
                self.defect_summary_label.setStyleSheet("""
                    color: #66BB6A; font-size: 10pt; padding: 12px;
                    background: rgba(76, 175, 80, 0.12);
                    border-radius: 10px; border: 1px solid rgba(76, 175, 80, 0.28);
                """)
            else:
                # Alle Defekte als Einzelzeilen mit Schweregrad anzeigen
                lines = []
                has_severe = False
                for k, v in sorted(active, key=lambda x: -x[1]):
                    name, thr_light, thr_heavy = label_map[k]
                    if v >= thr_heavy:
                        icon = "🔴"
                        sev = "schwer"
                        has_severe = True
                    elif v >= thr_light:
                        icon = "🟡"
                        sev = "mittel"
                    else:
                        icon = "🟢"
                        sev = "leicht"
                    lines.append(f"{icon} {name}  –  {sev}")
                n = len(active)
                header = f"⚠ {n} Defekt{'e' if n != 1 else ''} erkannt:"
                action = "werden entfernt" if defects.get("status") == "detected" else "wurden behandelt"
                summary = header + "\n" + "\n".join(lines) + f"\n→ {action}"
                self.defect_summary_label.setText(summary)
                color = "#FF5252" if has_severe else "#FFC107"
                bg = "rgba(255,82,82,0.12)" if has_severe else "rgba(255,165,0,0.12)"
                brd = "rgba(255,82,82,0.28)" if has_severe else "rgba(255,165,0,0.28)"
                self.defect_summary_label.setStyleSheet(f"""
                    color: {color}; font-size: 9pt; padding: 10px;
                    background: {bg};
                    border-radius: 10px; border: 1px solid {brd};
                    line-height: 160%;
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
                self.status_text.setText(
                    f"✅ Analyse abgeschlossen: {Path(self.current_file_path).name}  –  "
                    f"Jetzt einen der Buttons drücken!"
                )

    def _update_phase(self, phase_text):
        """Update current processing phase in status bar AND as overlay over waveform."""
        self.status_text.setText(f"⚙️ {phase_text}")
        # Fortschrittsbalken: nur Text-Format aktualisieren, KEIN Marquee-Modus.
        # Der prozentuale Fortschritt wird über _on_item_progress gesetzt —
        # setRange(0, 0) würde den Balken auf Marquee umschalten und den Prozentwert
        # unsichtbar machen.
        if hasattr(self, "progress_bar"):
            pb = self.progress_bar
            if pb.maximum() == 0:
                # Nur wenn noch kein prozentualer Bereich gesetzt → Basis setzen
                pb.setRange(0, 10000)
            pb.setVisible(True)
        # Kompaktes Overlay-Label über dem Waveform-Tab (bleibt im UI-Thread via QTimer)
        if hasattr(self, "waveform_widget"):
            ov = getattr(self, "_phase_overlay_label", None)
            if ov is None:
                from PyQt5.QtWidgets import QLabel  # already imported, but safe

                ov = QLabel(self.waveform_widget)
                ov.setAlignment(Qt.AlignmentFlag.AlignCenter)
                ov.setStyleSheet("""
                    color: #FFFFFF; font-size: 10pt; font-weight: bold;
                    background: rgba(30, 34, 60, 0.82);
                    border-radius: 8px; padding: 6px 18px;
                    border: 1px solid rgba(102,126,234,0.50);
                """)
                ov.setWordWrap(False)
                self._phase_overlay_label = ov
            ov.setText(f"⚙️  {phase_text}")
            ov.adjustSize()
            # Rechts oben im waveform_widget positionieren
            pw = self.waveform_widget
            ov.move(pw.width() - ov.width() - 12, 10)
            ov.setVisible(True)
            ov.raise_()

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
            QMessageBox.warning(
                self, "Verarbeitung läuft", "Queue kann nicht geleert werden während Verarbeitung läuft."
            )
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
        self.status_text.setText("Queue geleert")
        self._update_stats()

    def _export_all(self):
        """Export-Dialog: Format, Bittiefe und Zielordner wählen, dann AudioExporter nutzen."""
        stats = self.batch_queue.get_stats()
        if stats["completed"] == 0:
            QMessageBox.information(
                self,
                "Keine verarbeiteten Dateien",
                "Es wurden noch keine Dateien verarbeitet.\n" "Bitte zuerst eine Datei restaurieren, dann exportieren.",
            )
            return

        # ── Export-Dialog ────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("💾 Export-Einstellungen")
        dlg.setMinimumWidth(420)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setSpacing(18)

        dlg_layout.addWidget(
            QLabel(
                f"<b>{stats['completed']} Datei(en) werden exportiert.</b><br>"
                "<small>Wählen Sie Format und Qualität:</small>"
            )
        )

        # Format-Auswahl
        fmt_group_label = QLabel("Ausgabe-Format:")
        fmt_group_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        dlg_layout.addWidget(fmt_group_label)

        fmt_bg = QButtonGroup(dlg)
        formats = [
            (".flac", "FLAC 24-bit  — verlustfrei, Archivqualität  ✅ (empfohlen)"),
            (".wav", "WAV 24-bit   — verlustfrei, DAW-Kompatibel"),
            (".wav16", "WAV 16-bit   — CD-Qualität, kleinere Datei"),
            (".mp3", "MP3 320 kbps — verlustbehaftet, maximale Kompatibilität"),
            (".ogg", "OGG Vorbis   — verlustbehaftet, Open-Source"),
        ]
        rb_formats = []
        for i, (ext, label) in enumerate(formats):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            rb.setProperty("fmt_ext", ext)
            fmt_bg.addButton(rb)
            rb_formats.append(rb)
            dlg_layout.addWidget(rb)

        # Normalisierung
        chk_normalize = QCheckBox("Audio auf −0.1 dBFS normalisieren (True-Peak-sicher)")
        chk_normalize.setChecked(True)
        dlg_layout.addWidget(chk_normalize)

        buttons = QDialogButtonBox(parent=dlg)
        buttons.setStandardButtons(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        _ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if _ok_btn is not None:
            _ok_btn.setText("Zielordner wählen →")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec_() != QDialog.DialogCode.Accepted:
            return

        # Gewähltes Format ermitteln
        chosen_ext = ".flac"
        for rb in rb_formats:
            if rb.isChecked():
                chosen_ext = rb.property("fmt_ext")
                break
        normalize = chk_normalize.isChecked()

        # WAV 16-bit Sonderfall
        bit_depth = 24
        real_ext = chosen_ext
        if chosen_ext == ".wav16":
            real_ext = ".wav"
            bit_depth = 16

        # Zielordner wählen
        output_dir = QFileDialog.getExistingDirectory(self, "Zielordner für Export wählen")
        if not output_dir:
            return

        # ── Exportieren ──────────────────────────────────────────────────
        def _do_export():
            try:
                _AudioExporter = _bridge_get_audio_exporter_class()
                exporter = _AudioExporter() if _AudioExporter is not None else None
            except Exception:
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
                fmt_nice = real_ext.upper().lstrip(".") + f" {bit_depth}-bit"
                msg = f"✅ {exported} Datei(en) als {fmt_nice} nach\n{output_dir}\nexportiert."
                if errors:
                    msg += f"\n\n⚠️ {len(errors)} Fehler:\n" + "\n".join(errors[:5])
                self.status_text.setText(f"Export: {exported} Datei(en) → {real_ext.upper()}")
                self.title_bar.set_status(t("status.export_finished"), "#4CAF50")
                QMessageBox.information(self, "Export abgeschlossen", msg)

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
        rb_rest = QRadioButton("💿  Restoration — original-getreu")
        rb_stu = QRadioButton("🎯  Studio 2026 — moderner Highend-Klang")
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
            self.title_bar.set_status(t("status.settings_saved"), "#4CAF50")

    def _apply_i18n_texts(self) -> None:
        """Refresh visible UI texts after language changes."""
        if hasattr(self, "btn_import"):
            self.btn_import.setText(t("action.open_file"))
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
            "Noch keine Analyse", "No analysis yet"
        }:
            self.defect_summary_label.setText(t("ui.no_analysis"))
        if hasattr(self, "status_text") and self.status_text.text() in {
            "Bereit für Verarbeitung", "Ready for processing"
        }:
            self.status_text.setText(t("status.ready"))

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
