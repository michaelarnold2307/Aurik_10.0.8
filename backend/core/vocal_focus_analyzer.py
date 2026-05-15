"""
VocalFocusAnalyzer (VFA) — §0p [RELEASE_MUST] Aurik 9.12.1
===========================================================

Aggregiert alle vokalspezifischen Analysen zu einem einzelnen VFAResult,
der in ``_restoration_context["vfa_result"]`` injiziert wird und allen
nachgelagerten Phasen zur Verfügung steht.

Läuft nach SongCalibration, vor GoalApplicabilityFilter (§0p / §0g).
Aktivierung: PANNs Singing ≥ 0.25.

Komponenten:
  - PANNs-Singing-Konfidenz (aus UV3 übergeben)
  - VocalRegisterDetector → dominant_register + energy_bias_db
  - FrissonCandidateDetector → frisson_zones
  - LPC-Formant-Tracker → f1_mean, f2_mean, formant_stable

Singleton-Pattern (thread-safe double-checked locking).
Performance: ≤ 3 s für 30 s Stereo (48 kHz) ohne ML-Modelle.

Spec: copilot-instructions.md §0p (v9.12.1)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from backend.core.dsp.vocal_register_detector import detect_vocal_register as _detect_vocal_register_fn
except Exception:
    _detect_vocal_register_fn = None  # type: ignore[assignment]

try:
    from backend.core.frisson_candidate_detector import get_frisson_detector as _get_frisson_detector_fn
except Exception:
    _get_frisson_detector_fn = None  # type: ignore[assignment]

try:
    from backend.core.dsp.lpc_formant_tracker import get_lpc_formant_tracker as _get_lpc_formant_tracker_fn
except Exception:
    _get_lpc_formant_tracker_fn = None  # type: ignore[assignment]

try:
    import librosa as _librosa
except Exception:
    _librosa = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VFAResult — Ergebnis-Container
# ---------------------------------------------------------------------------


@dataclass
class VFAResult:
    """Gebündeltes Ergebnis der Vokal-Fokus-Analyse.

    Alle Felder sind serialisierbar (dict-kompatibel für ``_restoration_context``).
    """

    panns_singing: float = 0.0
    """PANNs Singing-Konfidenz [0, 1]."""

    vocal_present: bool = False
    """True wenn panns_singing ≥ 0.25."""

    dominant_register: str = "chest"
    """Vokalregister: "head", "chest", "fry_whisper", "unknown"."""

    energy_bias_db: float = -6.0
    """Register-adaptiver Energy-Bias für NR-Algorithmen (dB, negativ)."""

    frisson_zones: list[tuple[float, float]] = field(default_factory=list)
    """Liste von (start_s, end_s) Klimax/Gänsehaut-Passagen."""

    formant_f1_mean: float = 0.0
    """Mittlere F1-Formantfrequenz (Hz); 0.0 wenn nicht ermittelbar."""

    formant_f2_mean: float = 0.0
    """Mittlere F2-Formantfrequenz (Hz); 0.0 wenn nicht ermittelbar."""

    formant_stable: bool = True
    """True wenn F1-Varianz gering (Opera/Klassik); False bei stark fluktuierenden Formanten."""

    passaggio_zones: list[tuple[float, float]] = field(default_factory=list)
    """Register-Übergangszonen (Brust→Kopf). Leer wenn nicht erkannt."""

    vibrato_zones: list[tuple[float, float]] = field(default_factory=list)
    """Vibrato-geschützte Passagen (F0-Modulation 4–7 Hz). Leer wenn nicht erkannt."""

    vqi_gate_active: bool = False
    """True wenn panns_singing ≥ 0.35 → VQI-Gate in HolisticPerceptualGate aktiv."""

    analysis_duration_s: float = 0.0
    """Tatsächlich analysierte Segmentlänge (s)."""

    def to_dict(self) -> dict:
        """Serialisiert für ``_restoration_context``."""
        return {
            "panns_singing": self.panns_singing,
            "vocal_present": self.vocal_present,
            "dominant_register": self.dominant_register,
            "energy_bias_db": self.energy_bias_db,
            "frisson_zones": list(self.frisson_zones),
            "formant_f1_mean": self.formant_f1_mean,
            "formant_f2_mean": self.formant_f2_mean,
            "formant_stable": self.formant_stable,
            "passaggio_zones": list(self.passaggio_zones),
            "vibrato_zones": list(self.vibrato_zones),
            "vqi_gate_active": self.vqi_gate_active,
            "analysis_duration_s": self.analysis_duration_s,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: VocalFocusAnalyzer | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# VocalFocusAnalyzer
# ---------------------------------------------------------------------------


class VocalFocusAnalyzer:
    """Koordiniert alle Vokal-Analysen und gibt VFAResult zurück.

    Alle Teilanalysen sind non-blocking: Exceptions → sicherer Fallback.
    """

    # Analyse-Segment-Länge (s) — Balance zwischen Genauigkeit und Speed
    _ANALYSIS_MAX_S: float = 30.0
    _FORMANT_SEGMENT_S: float = 4.0

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing: float = 0.0,
    ) -> VFAResult:
        """Vollständige Vokal-Fokus-Analyse.

        Args:
            audio:         Mono oder Stereo float32 (SR = sr).
            sr:            Abtastrate.
            panns_singing: PANNs-Singing-Konfidenz aus UV3 (bereits berechnet).

        Returns:
            VFAResult mit allen Vokal-Analysedaten.
        """
        _t0 = time.perf_counter()

        result = VFAResult(
            panns_singing=float(panns_singing),
            vocal_present=panns_singing >= 0.25,
            vqi_gate_active=panns_singing >= 0.35,
        )

        # Mono-Segment für Analyse (Zentrum, max. _ANALYSIS_MAX_S)
        mono = self._to_mono(audio)
        n_max = int(self._ANALYSIS_MAX_S * sr)
        if len(mono) > n_max:
            _start = (len(mono) - n_max) // 2
            mono_seg = mono[_start : _start + n_max]
        else:
            mono_seg = mono
        result.analysis_duration_s = len(mono_seg) / max(sr, 1)

        if not result.vocal_present:
            # Kein Gesang → Defaults, keine Analyse nötig
            return result

        # 1. Vokalregister + Energy-Bias
        result.dominant_register, result.energy_bias_db = self._detect_register(mono_seg, sr, panns_singing)

        # 2. Frisson-Zonen (non-blocking, max. 150 ms)
        result.frisson_zones = self._detect_frisson(audio, sr)

        # 3. Formant-Analyse (non-blocking, max. 500 ms)
        f1_mean, f2_mean, stable = self._analyze_formants(mono_seg, sr)
        result.formant_f1_mean = f1_mean
        result.formant_f2_mean = f2_mean
        result.formant_stable = stable

        # 4. Passaggio-Zonen (F0-basiert, optional — leert sich bei Fehler)
        result.passaggio_zones = self._detect_passaggio(mono_seg, sr)

        # 5. Vibrato-Zonen (F0-Modulation 4–7 Hz, §0p Vibrato-Schutz)
        result.vibrato_zones = self._detect_vibrato(mono_seg, sr)

        elapsed = time.perf_counter() - _t0
        logger.info(
            "VocalFocusAnalyzer: panns_singing=%.2f register=%s energy_bias=%.1fdB "
            "frisson_zones=%d formant_f1=%.0fHz stable=%s passaggio=%d vibrato=%d zones in %.2fs",
            panns_singing,
            result.dominant_register,
            result.energy_bias_db,
            len(result.frisson_zones),
            result.formant_f1_mean,
            result.formant_stable,
            len(result.passaggio_zones),
            len(result.vibrato_zones),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Private Hilfsmethoden — alle non-blocking
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio.astype(np.float32)
        if audio.ndim == 2:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                return audio.mean(axis=0).astype(np.float32)
            if audio.shape[1] == 2:
                return audio.mean(axis=1).astype(np.float32)
            if audio.shape[0] == 1:
                return audio[0].astype(np.float32)
        return audio.flatten().astype(np.float32)

    @staticmethod
    def _detect_register(mono: np.ndarray, sr: int, panns_singing: float) -> tuple[str, float]:
        """VocalRegisterDetector → (register, energy_bias_db). Fallback: ("chest", -6.0)."""
        try:
            register, bias = _detect_vocal_register_fn(mono, sr, panns_singing)  # type: ignore[misc]
            return register, bias
        except Exception as exc:
            logger.debug("VFA._detect_register fallback: %s", exc)
            return "chest", -6.0

    @staticmethod
    def _detect_frisson(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """FrissonCandidateDetector → [(start_s, end_s)]. Non-blocking."""
        try:
            zones_raw = _get_frisson_detector_fn().detect(audio, sr)  # type: ignore[misc]
            # Normalisiere auf (start_s, end_s) — FrissonZone hat start_s / end_s
            result = []
            for z in zones_raw:
                if hasattr(z, "start_s") and hasattr(z, "end_s"):
                    result.append((float(z.start_s), float(z.end_s)))
                elif isinstance(z, (tuple, list)) and len(z) >= 2:
                    result.append((float(z[0]), float(z[1])))
            return result
        except Exception as exc:
            logger.debug("VFA._detect_frisson fallback: %s", exc)
            return []

    def _analyze_formants(self, mono: np.ndarray, sr: int) -> tuple[float, float, bool]:
        """LPC-Formant-Analyse → (f1_mean_hz, f2_mean_hz, is_stable).

        Verwendet 4 s Segment aus der Mitte. Fallback: (0.0, 0.0, True).
        """
        try:
            # 4s Analyse-Segment (Zentrum)
            n4 = int(self._FORMANT_SEGMENT_S * sr)
            if len(mono) > n4:
                start4 = (len(mono) - n4) // 2
                seg = mono[start4 : start4 + n4]
            else:
                seg = mono

            tracker = _get_lpc_formant_tracker_fn()  # type: ignore[misc]
            result = tracker.track(seg.astype(np.float32), sr)

            f1_vals: list[float] = []
            f2_vals: list[float] = []
            for frame in result.formant_tracks:
                freqs = frame.frequencies if hasattr(frame, "frequencies") else []
                confs = frame.confidences if hasattr(frame, "confidences") else [1.0] * len(freqs)
                if len(freqs) >= 1 and len(confs) >= 1 and float(confs[0]) > 0.25:
                    f1_vals.append(float(freqs[0]))
                if len(freqs) >= 2 and len(confs) >= 2 and float(confs[1]) > 0.25:
                    f2_vals.append(float(freqs[1]))

            f1_mean = float(np.mean(f1_vals)) if f1_vals else 0.0
            f2_mean = float(np.mean(f2_vals)) if f2_vals else 0.0
            # Stabilität: F1-Standardabweichung < 15 % des Mittelwerts → stabil (Opera)
            f1_std = float(np.std(f1_vals)) if len(f1_vals) >= 4 else 0.0
            is_stable = (f1_std < f1_mean * 0.15) if f1_mean > 0 else True
            return f1_mean, f2_mean, is_stable
        except Exception as exc:
            logger.debug("VFA._analyze_formants fallback: %s", exc)
            return 0.0, 0.0, True

    @staticmethod
    def _detect_passaggio(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Registerübergangszonen (Brust→Kopf) via F0-Sprünge.

        Gibt [(start_s, end_s)] zurück. Non-blocking.
        Passaggio-Kriterium: F0-Sprung > 2 Halbtöne innerhalb 200 ms.
        """
        try:
            hop = 512
            f0, voiced_flag, _ = _librosa.pyin(  # type: ignore[union-attr]
                mono.astype(np.float32),
                fmin=_librosa.note_to_hz("C2"),  # type: ignore[union-attr]
                fmax=_librosa.note_to_hz("C7"),  # type: ignore[union-attr]
                sr=sr,
                frame_length=2048,
                hop_length=hop,
            )
            if f0 is None or voiced_flag is None:
                return []

            zones: list[tuple[float, float]] = []
            times = _librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)  # type: ignore[union-attr]
            # Halbtöne-Differenz zwischen aufeinanderfolgenden voiced Frames
            prev_f0: float | None = None
            prev_t: float = 0.0
            in_passaggio = False
            zone_start: float = 0.0
            for i, (f, v) in enumerate(zip(f0, voiced_flag)):
                t = float(times[i])
                if not v or not np.isfinite(f) or f <= 0:
                    if in_passaggio:
                        zones.append((zone_start, t))
                        in_passaggio = False
                    prev_f0 = None
                    continue
                f = float(f)
                if prev_f0 is not None and prev_f0 > 0:
                    semitones = abs(12.0 * np.log2(f / prev_f0 + 1e-12))
                    if semitones >= 2.0 and (t - prev_t) <= 0.25:
                        if not in_passaggio:
                            zone_start = max(0.0, prev_t - 0.05)
                            in_passaggio = True
                    elif in_passaggio and semitones < 1.0:
                        zones.append((zone_start, t + 0.05))
                        in_passaggio = False
                prev_f0 = f
                prev_t = t
            if in_passaggio:
                zones.append((zone_start, float(times[-1])))
            return zones
        except Exception as exc:
            logger.debug("VFA._detect_passaggio fallback: %s", exc)
            return []

    @staticmethod
    def _detect_vibrato(mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Vibrato-Passagen (F0-Modulation 4–7 Hz) für §0p Strength-Cap.

        Nutzt ``natural_performance_detector._detect_vibrato_zones`` als Backend.
        Konvertiert Sample-basierte VibratoZone-Objekte in (start_s, end_s)-Tupel.
        Non-blocking: Fehler → [].
        """
        try:
            from backend.core.dsp.natural_performance_detector import (
                detect_vibrato_zones as _dvz,
            )

            raw_zones = _dvz(mono, sr)
            result: list[tuple[float, float]] = []
            for z in raw_zones:
                if hasattr(z, "start_sample") and hasattr(z, "end_sample"):
                    start_s = float(z.start_sample) / max(sr, 1)
                    end_s = float(z.end_sample) / max(sr, 1)
                    result.append((start_s, end_s))
                elif isinstance(z, (tuple, list)) and len(z) >= 2:
                    result.append((float(z[0]), float(z[1])))
            return result
        except Exception as exc:
            logger.debug("VFA._detect_vibrato fallback: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Singleton-Zugriff
# ---------------------------------------------------------------------------


def get_vocal_focus_analyzer() -> VocalFocusAnalyzer:
    """Thread-sicherer Singleton-Zugriff."""
    # pylint: disable-next=global-statement
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VocalFocusAnalyzer()
                logger.info("VocalFocusAnalyzer initialized (§0p)")
    return _instance
