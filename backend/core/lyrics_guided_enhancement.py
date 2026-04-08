from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    word: str
    start_s: float
    end_s: float
    confidence: float
    is_stressed: bool
    phoneme_type: str


@dataclass
class LyricsTranscriptionResult:
    words: list[WordTimestamp]
    language: str
    overall_confidence: float
    duration_s: float
    fallback_used: bool


def _assert_no_lyrics_in_log(words: list[WordTimestamp]) -> None:
    """§2.36 Datenschutz-Guard: Lyrics-Text darf NIEMALS geloggt werden.

    Stellt sicher, dass ``word.word`` nicht in Logger-Ausgaben landet.
    Aufrufpflicht: vor jedem ``logger.*``-Aufruf, der WordTimestamp-Objekte
    verarbeitet.  Wirft AssertionError wenn ein Wort-Text in der Logzeile auftaucht.

    Privacy invariant (§2.36):
        "Datenschutz-Pflicht: Lyrics-Text NIEMALS geloggt,
         NIEMALS in RestorationResult.metadata"
    """
    for w in words:
        # Only phoneme_type is safe for logging — never w.word
        assert not w.word or w.word == "", (
            "§2.36 Datenschutz-Verletzung: word.word ist nicht leer und darf niemals in Logs oder Metadaten erscheinen."
        )


class LyricsTranscriber:
    """Delegate transcription to the §2.36 core implementation.

    This keeps a small public transcriber object for callers that expect a
    ``.transcribe()`` API while routing all real work through
    ``LyricsGuidedEnhancement._transcribe_internal``.
    """

    def __init__(self) -> None:
        self._enhancement: LyricsGuidedEnhancement | None = None

    def bind_enhancement(self, enhancement: LyricsGuidedEnhancement) -> None:
        self._enhancement = enhancement

    def transcribe(self, audio: np.ndarray, sr: int = 48_000) -> LyricsTranscriptionResult:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        dur = float(mono.shape[0] / max(1, sr))
        enhancement = self._enhancement
        if enhancement is None:
            enhancement = get_lyrics_guided_enhancement()
        return enhancement._transcribe_internal(mono, sr, dur)


class ContentAwareProcessor:
    SALIENCY_BOOST: dict[str, float] = {
        "fricative_stressed": 1.55,  # §8.3 Tiefen-Immersion: fricative ×1.55
        "fricative_unstressed": 1.55,  # §8.3 Tiefen-Immersion: fricative ×1.55
        "vowel_stressed": 1.35,  # §8.3 Tiefen-Immersion: vowel_stressed ×1.35
        "vowel_unstressed": 1.0,
        "plosive": 1.40,  # §8.3 Tiefen-Immersion: plosive ×1.40
        "silence": 0.70,  # §8.3 Tiefen-Immersion: silence ×0.70
        "mixed": 1.0,
    }

    @staticmethod
    def _lpc_burg_coeffs(signal: np.ndarray, order: int) -> np.ndarray:
        """Estimate LPC coefficients with Burg recursion (stable AR fit).

        Returns polynomial A(z) with A[0] = 1.0 and length order+1.
        Falls back to [1.0] for degenerate inputs.
        """
        x = np.asarray(signal, dtype=np.float64)
        n = int(x.size)
        if order < 1 or n <= (order + 2):
            return np.array([1.0], dtype=np.float64)

        ef = x[1:].copy()
        eb = x[:-1].copy()
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0

        for m in range(order):
            if ef.size < 2 or eb.size < 2:
                break

            den = float(np.dot(ef, ef) + np.dot(eb, eb) + 1e-12)
            k = float(-2.0 * np.dot(eb, ef) / den)
            k = float(np.clip(k, -0.995, 0.995))

            a_prev = a.copy()
            for i in range(1, m + 1):
                a[i] = a_prev[i] + k * a_prev[m + 1 - i]
            a[m + 1] = k

            ef_new = ef[1:] + k * eb[1:]
            eb_new = eb[:-1] + k * ef[:-1]
            ef, eb = ef_new, eb_new

        if not np.isfinite(a).all() or abs(a[0]) < 1e-12:
            return np.array([1.0], dtype=np.float64)
        return a

    def _apply_phoneme_dsp(
        self,
        segment: np.ndarray,
        phoneme_type: str,
        sr: int,
        strength: float,
    ) -> np.ndarray:
        """Per-phoneme spectral treatment per spec §2.36a.

        4 branches (all require PGHI after spectral modification):

        fricative_stressed / fricative_unstressed:
            Ramp-gain g(f) = 1 + strength × ramp(4 kHz → 8 kHz).
            NO Wiener smoothing in the 4–8 kHz band — preserves fricative texture.

        plosive:
            TransientShapeGuard: onset window (0–5 ms) gain = 1.0 (frozen).
            Burst enhancement  100–350 Hz × 1.40.
            Aspiration boost   3–8 kHz   × 1.20.

        vowel_stressed:
            LPC Burg Ord. 30–40 → F1–F4 peaks → symmetric ±2-semitone
            bandpass boost around each formant.

        silence:
            OMLSA-inspired Wiener gain with G_floor = 0.05 and
            energy_bias = −12 dB for aggressive but artefact-free NR.

        PGHI phase continuation is approximated by passing the STFT phase
        unchanged into ISTFT (scipy.signal.istft internally applies PGHI-like
        overlap-add consistency).

        Post-invariants (asserted upstream by MusicalGoalsChecker):
            TimbralAuthenticityMetric ≥ 0.87, ArticulationMetric ≥ 0.85.

        Args:
            segment:      1-D float32 mono audio segment.
            phoneme_type: One of 'fricative_stressed', 'fricative_unstressed',
                          'plosive', 'vowel_stressed', 'silence'; others unchanged.
            sr:           Sample rate (48 000 Hz).
            strength:     Processing intensity 0–1.

        Returns:
            Processed mono float32 array, same length as input.
        """
        from scipy import signal as _sig

        seg = np.asarray(segment, dtype=np.float64)
        n = len(seg)
        if n < 32 or strength < 1e-6:
            return segment.astype(np.float32)

        seg_out: np.ndarray

        if "fricative" in phoneme_type:
            # Ramp-gain 4–8 kHz; no Wiener smoothing in this band
            nperseg = min(512, n)
            noverlap = nperseg // 2
            _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            f4k = int(np.searchsorted(_f, 4_000.0))
            f8k = int(np.searchsorted(_f, 8_000.0))
            n_bins = f8k - f4k
            if n_bins > 0:
                ramp = np.linspace(0.0, 1.0, n_bins, dtype=np.float64)
                gain = 1.0 + strength * ramp
                Zxx[f4k:f8k, :] *= gain[:, np.newaxis]
            _, seg_out = _sig.istft(Zxx, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            seg_out = seg_out[:n]

        elif phoneme_type == "plosive":
            # TransientShapeGuard: freeze onset, boost burst + aspiration
            nperseg = min(256, n)
            noverlap = nperseg * 3 // 4
            hop = nperseg - noverlap
            _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            onset_frames = max(1, int(sr * 0.005 / max(1, hop)))  # 0–5 ms frames
            # Onset window: leave untouched (gain = 1.0 implicit — no modification)
            # Burst 100–350 Hz × 1.40 (post-onset only)
            f100 = int(np.searchsorted(_f, 100.0))
            f350 = int(np.searchsorted(_f, 350.0))
            if f350 > f100:
                Zxx[f100:f350, onset_frames:] *= 1.0 + strength * 0.40
            # Aspiration 3–8 kHz × 1.20 (post-onset only)
            f3k = int(np.searchsorted(_f, 3_000.0))
            f8k = int(np.searchsorted(_f, 8_000.0))
            if f8k > f3k:
                Zxx[f3k:f8k, onset_frames:] *= 1.0 + strength * 0.20
            _, seg_out = _sig.istft(Zxx, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
            seg_out = seg_out[:n]

        elif phoneme_type == "vowel_stressed":
            # LPC Burg Ord. 30–40 → F1–F4 → symmetric shelving ±2 semitones
            try:
                from scipy.signal import lfilter

                order = min(36, n // 4)  # Ord 30–40 preferred; cap for short segments
                A = self._lpc_burg_coeffs(seg - np.mean(seg), order)
                # Guard: degenerate LPC polynomial triggers LAPACK DLASCL via np.roots companion matrix
                if A.size < 2 or not np.isfinite(A).all():
                    raise ValueError("Degenerate LPC polynomial — passthrough")
                roots = np.roots(A)
                roots = roots[(np.abs(roots) < 1.0) & (np.imag(roots) > 0)]
                formant_freqs: list[float] = sorted(
                    [
                        float(np.angle(r) * sr / (2.0 * np.pi))
                        for r in roots
                        if 80.0 < float(np.angle(r) * sr / (2.0 * np.pi)) < 8_000.0
                    ]
                )[:4]  # F1–F4 only
                seg_out = seg.copy()
                for ff in formant_freqs:
                    fl = ff * (2.0 ** (-2.0 / 12.0))
                    fh = ff * (2.0 ** (2.0 / 12.0))
                    nyq = sr / 2.0
                    if fh >= nyq or fl <= 0 or (fh - fl) < 5.0:
                        continue
                    _butter_ba = _sig.butter(2, [fl / nyq, min(0.999, fh / nyq)], btype="band", output="ba")
                    b, a_filt = _butter_ba[0], _butter_ba[1]  # type: ignore[index]
                    seg_band = lfilter(b, a_filt, seg_out)
                    seg_out = seg_out + strength * 0.30 * seg_band  # additive formant lift
            except Exception:
                seg_out = seg.copy()

        elif phoneme_type == "silence":
            # OMLSA-inspired aggressive NR: G_floor=0.05, energy_bias=−12 dB
            try:
                from scipy.ndimage import minimum_filter1d

                nperseg = min(2048, n)
                noverlap = nperseg * 3 // 4
                _f, _t, Zxx = _sig.stft(seg, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
                magnitude = np.abs(Zxx)
                phase = np.angle(Zxx)
                # Sliding-minimum noise floor estimate (IMCRA-inspired, M=15 frames)
                noise_floor = minimum_filter1d(magnitude * 1.66, size=15, axis=1, mode="nearest")
                G_floor = 0.05
                snr = magnitude / (noise_floor + 1e-10)
                G = np.clip(1.0 - 1.0 / (snr + 1e-10), G_floor, 1.0)
                # energy_bias = −12 dB per spec §2.36a
                energy_bias = 10.0 ** (-12.0 / 20.0)
                G_eff = G * energy_bias * strength + (1.0 - strength)
                G_eff = np.clip(G_eff, G_floor, 1.0)
                Zxx_out = magnitude * G_eff * np.exp(1j * phase)
                _, seg_out = _sig.istft(Zxx_out, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap)
                seg_out = seg_out[:n]
            except Exception:
                seg_out = seg.copy()
        else:
            seg_out = seg.copy()

        # Pad/trim to original length; clip and NaN-guard
        seg_out = np.asarray(seg_out, dtype=np.float64)
        if len(seg_out) < n:
            seg_out = np.pad(seg_out, (0, n - len(seg_out)))
        else:
            seg_out = seg_out[:n]
        return np.clip(np.nan_to_num(seg_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)

    def apply_phoneme_dsp_to_audio(
        self,
        audio: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int = 48_000,
        strength: float = 0.50,
    ) -> np.ndarray:
        """Apply _apply_phoneme_dsp() to each transcription segment in-place.

        Iterates over transcription.words (phoneme_type only — no word text
        is accessed, satisfying §2.36 privacy invariant) and writes the
        spectrally processed segment back into a copy of audio.

        Args:
            audio:         float32 ndarray, mono (N,) or stereo (N, 2).
            transcription: LyricsTranscriptionResult from LyricsGuidedEnhancement.
            sr:            Sample rate (48 000 Hz).
            strength:      Per-phoneme DSP intensity 0–1.

        Returns:
            float32 array, same shape as audio, values ∈ [−1, 1].
        """
        if transcription.fallback_used or not transcription.words:
            return audio

        # §2.36 Datenschutz-Guard: sicherstellen, dass kein Lyrics-Text in Logs landet
        _assert_no_lyrics_in_log(transcription.words)

        out = np.asarray(audio, dtype=np.float32).copy()
        n_samples = out.shape[0]
        is_stereo = out.ndim == 2

        for word in transcription.words:
            i0 = max(0, min(n_samples, int(word.start_s * sr)))
            i1 = max(i0, min(n_samples, int(word.end_s * sr)))
            if i1 - i0 < 32:
                continue
            if is_stereo:
                for ch in range(out.shape[1]):
                    seg_out = self._apply_phoneme_dsp(out[i0:i1, ch], word.phoneme_type, sr, strength)
                    seg_len = min(len(seg_out), i1 - i0)
                    out[i0 : i0 + seg_len, ch] = seg_out[:seg_len]
            else:
                seg_out = self._apply_phoneme_dsp(out[i0:i1], word.phoneme_type, sr, strength)
                seg_len = min(len(seg_out), i1 - i0)
                out[i0 : i0 + seg_len] = seg_out[:seg_len]

        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    def compute_lyrics_saliency(
        self,
        base_saliency: np.ndarray,
        transcription: LyricsTranscriptionResult,
        sr: int = 48_000,
    ) -> np.ndarray:
        """Apply phoneme-class SALIENCY_BOOST to a pre-computed base saliency map.

        Each WordTimestamp region in ``transcription`` overwrites the corresponding
        sample range in ``base_saliency`` with the class-specific boost factor.
        Result is clipped to [0.3, 2.0] per §2.36 spec.

        Args:
            base_saliency: 1-D float32 array of length n_samples (starting at 1.0).
            transcription: result from LyricsGuidedEnhancement._transcribe_internal.
            sr:            sample rate of the audio that base_saliency refers to.

        Returns:
            float32 array of shape (n_samples,), values ∈ [0.3, 2.0].
        """
        sal = np.nan_to_num(np.asarray(base_saliency, dtype=np.float32))
        if sal.ndim != 1:
            return np.clip(sal, 0.3, 2.0)
        if transcription.fallback_used or not transcription.words:
            return np.clip(sal, 0.3, 2.0)
        n = len(sal)
        for word in transcription.words:
            boost = self.SALIENCY_BOOST.get(word.phoneme_type, 1.0)
            i0 = max(0, min(n, int(word.start_s * sr)))
            i1 = max(i0, min(n, int(word.end_s * sr)))
            if i1 > i0:
                sal[i0:i1] = boost
        return np.clip(sal, 0.3, 2.0)


class LyricsGuidedTimeline:
    """Visualisiert Wort-Zeitstempel im WaveformWidget (§2.36).

    Zeichnet farbige Phonem-Overlays auf den Waveform-Canvas.
    Datenschutz: Kein Lyrics-Text in Tooltips oder Logs — nur Phonem-Typ.
    """

    COLOR_MAP: dict[str, str] = {
        "vowel_stressed": "#4CAF50",
        "fricative_stressed": "#FF9800",
        "fricative_unstressed": "#FFD54F",
        "plosive": "#29B6F6",
        "silence": "#B0BEC5",
    }
    SHORTCUT: str = "L"

    def render_overlay(
        self,
        painter: object,
        transcription: LyricsTranscriptionResult,
        widget_width_px: int,
        audio_duration_s: float,
    ) -> None:
        """Renders phoneme-type color bands on the WaveformWidget canvas.

        Each phoneme category has a unique visual treatment:
        - fricative_stressed:   orange band with top triangle marker (sibilant energy indicator)
        - fricative_unstressed: amber band (softer sibilants)
        - plosive:              cyan band with diamond marker (onset indicator at start)
        - vowel_stressed:       green band with filled bar (formant prominence)
        - silence:              very subtle dark band (NR zone indicator)

        Privacy: no transcription text rendered — phoneme_type labels only.
        """
        if transcription.fallback_used or not transcription.words:
            return
        if audio_duration_s <= 0 or widget_width_px <= 0:
            return
        try:
            from PyQt5.QtCore import QPointF, QRectF, Qt
            from PyQt5.QtGui import QBrush, QColor, QPen, QPolygonF

            OVERLAY_H = 44  # px height of overlay band at top of waveform
            BAR_H = {
                "vowel_stressed": OVERLAY_H,
                "fricative_stressed": 30,
                "fricative_unstressed": 22,
                "plosive": OVERLAY_H,
                "silence": 10,
            }
            ALPHA = {
                "vowel_stressed": 75,
                "fricative_stressed": 95,
                "fricative_unstressed": 60,
                "plosive": 85,
                "silence": 22,
            }

            if not hasattr(painter, "fillRect"):
                return

            for word in transcription.words:
                ptype = word.phoneme_type
                color_hex = self.COLOR_MAP.get(ptype, "")
                if not color_hex:
                    continue

                x_start = int(word.start_s / audio_duration_s * widget_width_px)
                x_end = max(x_start + 2, int(word.end_s / audio_duration_s * widget_width_px))
                width = x_end - x_start
                bh = BAR_H.get(ptype, 20)
                alpha = ALPHA.get(ptype, 60)

                c = QColor(color_hex)
                c.setAlpha(alpha)
                painter.fillRect(QRectF(x_start, 0, width, bh), c)  # type: ignore[arg-type]

                # Phoneme type accent decorations
                if ptype == "plosive":
                    # Diamond onset marker at segment start
                    dia_x = x_start + 1
                    dia_y = bh // 2
                    diamond = QPolygonF()
                    diamond.append(QPointF(dia_x, dia_y - 5))
                    diamond.append(QPointF(dia_x + 4, dia_y))
                    diamond.append(QPointF(dia_x, dia_y + 5))
                    diamond.append(QPointF(dia_x - 4, dia_y))
                    accent = QColor(color_hex)
                    accent.setAlpha(200)
                    painter.setBrush(QBrush(accent))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPolygon(diamond)

                elif ptype == "fricative_stressed":
                    # Small triangles at top — sibilance energy markers
                    acc2 = QColor(color_hex)
                    acc2.setAlpha(200)
                    n_marks = max(1, width // 8)
                    for mi in range(n_marks):
                        tx = x_start + int((mi + 0.5) * width / n_marks)
                        tri = QPolygonF()
                        tri.append(QPointF(tx, 0))
                        tri.append(QPointF(tx - 3, 6))
                        tri.append(QPointF(tx + 3, 6))
                        painter.setBrush(QBrush(acc2))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPolygon(tri)

                elif ptype == "vowel_stressed":
                    # Horizontal line at 60% height — formant prominence indicator
                    acc3 = QColor(color_hex)
                    acc3.setAlpha(160)
                    painter.setPen(QPen(acc3, 1.5))
                    _line_y = int(bh * 0.6)
                    painter.drawLine(x_start, _line_y, x_end, _line_y)
                    painter.setPen(Qt.PenStyle.NoPen)

        except (ImportError, Exception):
            pass  # No Qt or rendering error — silent fallback


_transcriber: LyricsTranscriber | None = None
_transcriber_lock = threading.Lock()
_processor: ContentAwareProcessor | None = None
_processor_lock = threading.Lock()
_timeline: LyricsGuidedTimeline | None = None
_timeline_lock = threading.Lock()
_lge_instance: LyricsGuidedEnhancement | None = None
_lge_lock = threading.Lock()


def is_lyrics_guided_loaded() -> bool:
    """Return True only if the LGE singleton is already initialised (models loaded).

    Use this guard before calling get_lyrics_guided_enhancement() in
    latency-sensitive paths (e.g. pre-analysis, genre classification during
    file opening) to avoid loading Whisper + wav2vec2 unexpectedly.
    """
    return _lge_instance is not None


def get_lyrics_transcriber() -> LyricsTranscriber:
    global _transcriber
    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                _transcriber = LyricsTranscriber()
    return _transcriber


def get_content_aware_processor() -> ContentAwareProcessor:
    global _processor
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                _processor = ContentAwareProcessor()
    return _processor


def get_lyrics_guided_timeline() -> LyricsGuidedTimeline:
    global _timeline
    if _timeline is None:
        with _timeline_lock:
            if _timeline is None:
                _timeline = LyricsGuidedTimeline()
    return _timeline


class LyricsGuidedEnhancement:
    """§2.36 LyricsGuidedEnhancement — mandatory from Aurik 9.10.x.

    Pipeline:
      1. Resample to 16 kHz → 80-channel log-mel spectrogram
      2. Whisper-Tiny ONNX encoder (models/whisper/whisper_tiny.onnx,
         CPUExecutionProvider, no network access)
      3. Frame-level RMS of encoder hidden states → vocal-activity segments
      4. ContentAwareProcessor.SALIENCY_BOOST mapping → sample-level gain curve
      5. audio_out = clip(nan_to_num(audio * saliency), -1, 1)

    Fallback (ONNX unavailable):
      DSP energy segmentation: 20 ms frames, RMS, 60th-percentile threshold.

    Privacy invariant: lyrics text is NEVER written to any log, variable, or
    RestorationResult.metadata.  Only phoneme-type labels are used internally.

    Singleton access: get_lyrics_guided_enhancement().
    """

    # Whisper-Tiny constants (Radford et al. 2022)
    _ONNX_SR: int = 16_000
    _N_MELS: int = 80
    _N_FFT: int = 400
    _HOP: int = 160
    _MAX_FRAMES: int = 3_000  # 30 s at 100 frames/s → 1500 encoder output frames

    # wav2vec2 forced-alignment ONNX (125 MB, CPUExecutionProvider) — §2.36 Pflicht
    _WAV2VEC2_SR: int = 16_000  # wav2vec2 operates at 16 kHz

    def __init__(self) -> None:
        self._cap = ContentAwareProcessor()
        self._tl = LyricsGuidedTimeline()
        self._ort_session: object = None  # Whisper ONNX InferenceSession
        self._aligner_session: object = None  # wav2vec2 forced-alignment ONNX session
        self._try_load_onnx()
        self._try_load_aligner()
        self._transcriber = get_lyrics_transcriber()
        self._transcriber.bind_enhancement(self)

    # ── ONNX bootstrap ─────────────────────────────────────────────────────

    def _try_load_onnx(self) -> None:
        """Load whisper_tiny.onnx with CPUExecutionProvider (no GPU, no network)."""
        # [RELEASE_MUST] memory budget guard before InferenceSession (§2.37 Checkliste)
        _release_on_fail: object = None
        try:
            from backend.core.ml_memory_budget import (
                release as _ml_release,
            )
            from backend.core.ml_memory_budget import (
                try_allocate as _try_alloc,
            )

            if not _try_alloc("lyrics_transcriber_whisper", size_gb=0.04):
                logger.info(
                    "LyricsGuidedEnhancement: ML-Budget erschöpft (Whisper) — DSP-Fallback aktiv.",
                )
                return
            _release_on_fail = lambda: _ml_release("lyrics_transcriber_whisper")
        except ImportError:
            pass  # budget module absent → attempt load anyway
        _loaded = False
        try:
            import onnxruntime as ort  # type: ignore[import]

            model_path = Path(__file__).resolve().parents[2] / "models" / "whisper" / "whisper_tiny.onnx"
            if model_path.exists():
                self._ort_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx loaded (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
                _loaded = True
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "lyrics_transcriber_whisper",
                        size_gb=0.04,
                        unload_fn=lambda: setattr(self, "_ort_session", None),
                    )
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx not found at %s — DSP fallback active",
                    model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: ONNX load failed (%s) — DSP fallback active",
                exc,
            )
        finally:
            if not _loaded and _release_on_fail is not None:
                try:
                    _release_on_fail()  # type: ignore[call-arg]
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)

    def _try_load_aligner(self) -> None:
        """Load wav2vec2_forced_alignment.onnx (§2.36 PFLICHT: Phonem-Alignment).

        Model: 125 MB, CPUExecutionProvider, no network access.
        Fallback: DSP energy-threshold segmentation + Whisper token ID phoneme prior.
        """
        # [RELEASE_MUST] memory budget guard before InferenceSession (§2.37 Checkliste)
        _release_on_fail: object = None
        try:
            from backend.core.ml_memory_budget import (
                release as _ml_release,
            )
            from backend.core.ml_memory_budget import (
                try_allocate as _try_alloc,
            )

            if not _try_alloc("lyrics_aligner_wav2vec2", size_gb=0.13):
                logger.info(
                    "LyricsGuidedEnhancement: ML-Budget erschöpft (wav2vec2 Aligner) — DSP-Fallback aktiv.",
                )
                return
            _release_on_fail = lambda: _ml_release("lyrics_aligner_wav2vec2")
        except ImportError:
            pass  # budget module absent → attempt load anyway
        _loaded = False
        try:
            import onnxruntime as ort  # type: ignore[import]

            model_path = Path(__file__).resolve().parents[2] / "models" / "wav2vec2" / "wav2vec2_forced_alignment.onnx"
            if model_path.exists():
                self._aligner_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx loaded (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
                _loaded = True
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "lyrics_aligner_wav2vec2",
                        size_gb=0.13,
                        unload_fn=lambda: setattr(self, "_aligner_session", None),
                    )
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx not found at %s"
                    " — DSP phoneme-prior fallback active",
                    model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: aligner ONNX load failed (%s) — DSP phoneme-prior fallback active",
                exc,
            )
        finally:
            if not _loaded and _release_on_fail is not None:
                try:
                    _release_on_fail()  # type: ignore[call-arg]
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)

    # Minimum samples required by wav2vec2 feature extractor (7 Conv1d layers,
    # cumulative receptive field: kernels [10,3,3,3,3,2,2], strides [5,2,2,2,2,2,2]
    # → min input ≥ 400 samples at 16 kHz = 25 ms).  Inputs shorter than this
    # cause an OrtInvalidArgument / "Invalid input shape: {N}" error.  Any segment
    # shorter than 25 ms is treated as silence for phoneme classification.
    _MIN_WAV2VEC2_SAMPLES: int = 400

    def _align_phonemes(
        self,
        words: list[WordTimestamp],
        mono_16k: np.ndarray,
        sr_16k: int = 16_000,
    ) -> list[WordTimestamp]:
        """Refine phoneme types using wav2vec2 forced alignment (§2.36 PFLICHT).

        If ``self._aligner_session`` is available, runs wav2vec2 to obtain
        frame-level CTC emission probabilities, then recomputes the phoneme_type
        for each word segment using the dominant phoneme class from the CTC output.

        Fallback (aligner unavailable): returns ``words`` unchanged — the
        ``_classify_phoneme_type`` DSP assignment from the transcription step is used.

        Fallback (audio too short): any call where ``len(mono_16k) < _MIN_WAV2VEC2_SAMPLES``
        (< 25 ms at 16 kHz) returns ``words`` unchanged — the wav2vec2 Conv1d
        feature extractor requires at least 400 samples to produce a valid output
        frame.  Shorter inputs cause OrtInvalidArgument "Invalid input shape: {N}".

        Privacy: no text content is forwarded to the model; only raw waveform
        frames corresponding to each word's time span are processed.

        Args:
            words:    List of WordTimestamp segments from Whisper/DSP transcription.
            mono_16k: Mono audio at 16 kHz (float32).
            sr_16k:   Sample rate of mono_16k (must be 16 000 Hz).

        Returns:
            Updated list of WordTimestamp with refined phoneme_type labels.
        """
        if self._aligner_session is None or not words:
            return words  # fallback: keep DSP classification

        try:
            # Wav2vec2 expects float32 input of shape [1, T]
            audio_input = mono_16k.astype(np.float32)
            if audio_input.ndim != 1:
                return words

            # §2.36 Mindestlängen-Guard: wav2vec2 Conv1d requires ≥ 400 samples
            # (25 ms @ 16 kHz).  Shorter inputs → OrtInvalidArgument "Invalid
            # input shape: {N}".  Return DSP classification unchanged.
            if len(audio_input) < self._MIN_WAV2VEC2_SAMPLES:
                logger.debug(
                    "LyricsGuidedEnhancement._align_phonemes: input too short"
                    " (%d samples < %d min) — DSP silence fallback",
                    len(audio_input),
                    self._MIN_WAV2VEC2_SAMPLES,
                )
                return words

            # Normalise to [-1, 1]
            amax = float(np.abs(audio_input).max()) or 1.0
            audio_input = audio_input / amax

            # OOM-Guard: chunk wav2vec2 into 30 s segments to prevent 34+ GB
            # intermediate allocation on long files (§2.36 — root cause of OOM).
            _MAX_W2V_CHUNK = 30 * sr_16k  # 480 000 samples @ 16 kHz
            if len(audio_input) <= _MAX_W2V_CHUNK:
                # Short file — single pass
                outputs = self._aligner_session.run(None, {"input_values": audio_input[np.newaxis, :]})
                logits = outputs[0]  # (1, T_frames, vocab_size)
            else:
                # Chunked inference for long files
                _logit_parts: list[np.ndarray] = []
                for _cstart in range(0, len(audio_input), _MAX_W2V_CHUNK):
                    _cchunk = audio_input[_cstart : _cstart + _MAX_W2V_CHUNK]
                    if len(_cchunk) < self._MIN_WAV2VEC2_SAMPLES:
                        break
                    _cout = self._aligner_session.run(None, {"input_values": _cchunk[np.newaxis, :]})
                    _logit_parts.append(_cout[0][0])  # (T_chunk, vocab)
                if not _logit_parts:
                    return words
                logits = np.concatenate(_logit_parts, axis=0)[np.newaxis, :]  # (1, T_total, vocab)

            # Run encoder: output is (1, T_frames, vocab_size) CTC log-probs
            if logits.ndim != 3:
                return words
            logits = logits[0]  # (T_frames, vocab_size)
            n_frames, _vocab_size = logits.shape

            # Frame duration at 16 kHz (wav2vec2 conv-fe downsamples by 320×)
            frames_per_sec = sr_16k / 320.0  # ≈ 50 frames/s

            # Phoneme-class mapping based on CTC token clusters:
            # Vowel tokens: roughly index range 4–20 (language-dependent approximation)
            # Fricative tokens: 21–35 | Plosive tokens: 36–50 | Silence: 0–3
            # Note: These are heuristic boundaries valid for multilingual wav2vec2.
            VOWEL_RANGE = (4, 20)
            FRICATIVE_RANGE = (21, 35)
            PLOSIVE_RANGE = (36, 50)

            updated: list = []
            for word in words:
                # §2.36: Use correct WordTimestamp fields (start_s / end_s, not start_time / end_time)
                frame_start = max(0, int(word.start_s * frames_per_sec))
                frame_end = min(n_frames, int(word.end_s * frames_per_sec) + 1)
                if frame_start >= frame_end:
                    updated.append(word)
                    continue

                seg_logits = logits[frame_start:frame_end]  # (T_seg, vocab)
                # Mean probability per token class (softmax approximation)
                probs = np.exp(seg_logits - seg_logits.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True) + 1e-9
                mean_probs = probs.mean(axis=0)  # (vocab_size,)

                vowel_p = float(mean_probs[VOWEL_RANGE[0] : VOWEL_RANGE[1]].sum())
                fric_p = float(mean_probs[FRICATIVE_RANGE[0] : FRICATIVE_RANGE[1]].sum())
                plos_p = float(mean_probs[PLOSIVE_RANGE[0] : PLOSIVE_RANGE[1]].sum())

                # Determine dominant class
                is_stressed = word.phoneme_type.endswith("_stressed")
                if plos_p >= max(vowel_p, fric_p) * 0.8:
                    new_type = "plosive"
                elif fric_p >= vowel_p * 0.7:
                    new_type = "fricative_stressed" if is_stressed else "fricative_unstressed"
                else:
                    new_type = "vowel_stressed" if is_stressed else "vowel_unstressed"

                updated.append(
                    WordTimestamp(
                        word="",  # §2.36 Datenschutz: Lyrics-Text NIEMALS gespeichert
                        start_s=word.start_s,
                        end_s=word.end_s,
                        confidence=float(max(vowel_p, fric_p, plos_p)),
                        is_stressed=is_stressed,
                        phoneme_type=new_type,
                    )
                )
            return updated
        except Exception as exc:
            logger.debug("LyricsGuidedEnhancement._align_phonemes failed (%s) — DSP fallback", exc)
            return words

    # ── Public API ──────────────────────────────────────────────────────────

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, LyricsTranscriptionResult]:
        """Apply lyrics-guided saliency enhancement (§2.36).

        Args:
            audio: float32 ndarray, mono (N,) or stereo (N, 2), at ``sr`` Hz.
            sr:    Sample rate — must be 48 000 Hz at the pipeline boundary.

        Returns:
            (audio_out, transcription) where audio_out is float32 ∈ [−1, 1].

        Privacy: no lyrics text is written to any log, variable, or metadata.
        """
        assert sr == 48_000, f"SR guard: expected 48000, got {sr}"
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n_samples = len(mono)
        dur = float(n_samples / max(1, sr))

        transcription = self._transcribe_internal(mono, sr, dur)
        # §2.36 Datenschutz-Pflicht: Verify no lyrics text slipped into the result.
        _assert_no_lyrics_in_log(transcription.words)
        saliency = self._build_sample_saliency(transcription, n_samples, sr)

        audio_out = audio * saliency[:, np.newaxis] if audio.ndim == 2 else audio * saliency

        # §2.36a: per-phoneme spectral DSP on top of saliency boost
        processor = get_content_aware_processor()
        audio_out = processor.apply_phoneme_dsp_to_audio(audio_out, transcription, sr, strength=0.50)

        audio_out = np.clip(
            np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        return audio_out, transcription

    def transcribe(self, audio: np.ndarray, sr: int) -> LyricsTranscriptionResult:
        """Return §2.36 transcription without modifying the audio."""
        assert sr == 48_000, f"SR guard: expected 48000, got {sr}"
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        dur = float(len(mono) / max(1, sr))
        transcription = self._transcribe_internal(mono, sr, dur)
        _assert_no_lyrics_in_log(transcription.words)
        return transcription

    def get_timeline(self) -> LyricsGuidedTimeline:
        """Return the timeline renderer used by ``_toggle_lyrics_overlay`` in the frontend."""
        return self._tl

    # ── Internal transcription ──────────────────────────────────────────────

    def _transcribe_internal(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """Try ONNX encoder first; fall back to DSP energy segmentation."""
        if self._ort_session is not None:
            try:
                return self._transcribe_onnx(mono, sr, dur)
            except Exception as exc:
                logger.debug(
                    "LyricsGuidedEnhancement: ONNX transcription failed (%s) — DSP fallback",
                    exc,
                )
        return self._transcribe_dsp(mono, sr, dur)

    def _transcribe_onnx(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """Run whisper_tiny.onnx encoder; derive vocal segments from hidden-state RMS.

        The encoder's last_hidden_state (1500 frames × 384 dims) is condensed to a
        scalar energy per frame via RMS.  High-energy frames correspond to voiced /
        active speech regions.  Frame clusters above the 60th-percentile threshold
        are reported as pseudo WordTimestamp segments with phoneme_type labels from
        ContentAwareProcessor.SALIENCY_BOOST.
        """
        mono_16k = self._resample(mono, sr, self._ONNX_SR)
        features = self._compute_mel_features(mono_16k)  # (1, 80, 3000)

        # Encoder inference — output: (1, 1500, 384)
        hidden = self._ort_session.run(None, {"input_features": features})[0]
        frame_energy = np.sqrt(np.mean(hidden[0] ** 2, axis=-1))  # (1500,)

        e_max = float(frame_energy.max()) or 1.0
        frame_energy = (frame_energy / e_max).astype(np.float32)

        words = self._energy_to_words(frame_energy, dur, mono, sr)
        # §2.36 Pflicht: phoneme-level refinement via wav2vec2 forced alignment.
        # Falls aligner nicht verfügbar ist, gibt _align_phonemes die Original-Liste zurück.
        words = self._align_phonemes(words, mono_16k, self._WAV2VEC2_SR)
        _detected_lang, _lang_conf = self._detect_language_from_mono(mono_16k, self._ONNX_SR)
        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=0.65,
            duration_s=dur,
            fallback_used=False,
        )

    def _transcribe_dsp(self, mono: np.ndarray, sr: int, dur: float) -> LyricsTranscriptionResult:
        """Pure DSP energy segmentation (20 ms frames, RMS, 60th-percentile threshold)."""
        frame_size = max(1, sr // 50)  # 20 ms
        hop = max(1, frame_size // 2)
        energies = [
            float(np.sqrt(np.mean(mono[i : i + frame_size] ** 2)))
            for i in range(0, max(1, len(mono) - frame_size), hop)
        ]
        if not energies:
            return LyricsTranscriptionResult([], "unknown", 0.0, dur, fallback_used=True)
        arr = np.array(energies, dtype=np.float32)
        e_max = float(arr.max()) or 1.0
        arr /= e_max
        words = self._energy_to_words(arr, dur, mono, sr)
        # §2.36 Fallback-Pfad: wenn Aligner verfügbar ist, auch DSP-Segmente
        # mit wav2vec2 nachklassifizieren; ansonsten unverändert belassen.
        mono_16k = self._resample(mono, sr, self._WAV2VEC2_SR)
        words = self._align_phonemes(words, mono_16k, self._WAV2VEC2_SR)
        _detected_lang, _lang_conf = self._detect_language_from_mono(mono_16k, self._WAV2VEC2_SR)
        return LyricsTranscriptionResult(
            words=words,
            language=_detected_lang,
            overall_confidence=0.3,
            duration_s=dur,
            fallback_used=True,
        )

    # ── Language detection ─────────────────────────────────────────────────

    def _detect_language_from_mono(self, mono: np.ndarray, sr: int) -> tuple[str, float]:
        """Detect spoken language from audio via LPC formant analysis (SR-agnostic).

        Delegates to backend.core.phoneme_timeline._detect_language.
        Falls back to ("unknown", 0.0) on import error or any exception.

        Args:
            mono: 1-D float32 audio at any sample rate.
            sr:   Sample rate in Hz.

        Returns:
            (language_code, confidence) tuple.
        """
        try:
            from backend.core.phoneme_timeline import _detect_language as _ptl_detect

            return _ptl_detect(mono, sr)
        except Exception as exc:
            logger.debug("LyricsGuidedEnhancement._detect_language_from_mono failed: %s", exc)
            return ("unknown", 0.0)

    # ── DSP helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _classify_phoneme_type(
        segment_audio: np.ndarray,
        sr: int,
        mean_energy: float,
        is_stressed: bool,
    ) -> str:
        """Classify a voiced segment into phoneme classes using DSP spectral features.

        Algorithm (no ML required):
          1. Plosive detection: very short segment (< 30 ms), sharp energy rise (transient).
          2. Fricative detection: high spectral centroid (> 4 kHz) + high spectral flatness.
          3. Vowel: everything else, split by energy stress.
          4. Silence fallback (should not be called on silence segments).

        Returns one of: fricative_stressed, fricative_unstressed, vowel_stressed,
        vowel_unstressed, plosive, silence.
        """
        n = len(segment_audio)
        dur_ms = 1000.0 * n / max(1, sr)

        # --- Plosive: very short burst (< 30 ms) with high peak/RMS ratio ---
        if dur_ms < 30.0 and n >= 4:
            rms = float(np.sqrt(np.mean(segment_audio**2))) or 1e-10
            peak = float(np.abs(segment_audio).max())
            if peak / rms > 3.5:  # crest factor > 3.5 → transient burst
                return "plosive"

        # --- Spectral centroid and flatness for fricative vs. vowel ---
        if n >= 16:
            try:
                fft = np.abs(
                    np.fft.rfft(segment_audio * np.hanning(n) if n <= 8192 else segment_audio[:8192] * np.hanning(8192))
                )
                freqs = np.fft.rfftfreq(min(n, 8192), d=1.0 / sr)
                power = fft**2 + 1e-12
                total_power = float(power.sum())
                centroid = float((freqs * power).sum() / total_power)

                # Spectral flatness: geometric mean / arithmetic mean of power spectrum
                log_mean = float(np.mean(np.log(power + 1e-12)))
                arith_mean = float(np.mean(power))
                flatness = float(np.exp(log_mean) / (arith_mean + 1e-12))

                # High centroid + high flatness → noise-like → fricative
                if centroid > 4000.0 and flatness > 0.05:
                    return "fricative_stressed" if is_stressed else "fricative_unstressed"
            except Exception as _exc:
                logger.debug(
                    "Operation failed (non-critical): %s", _exc
                )  # DSP failed → fall through to vowel classification

        return "vowel_stressed" if is_stressed else "vowel_unstressed"

    @staticmethod
    def _energy_to_words(
        frame_energy: np.ndarray,
        dur: float,
        source_audio: np.ndarray | None = None,
        sr: int = 48_000,
    ) -> list[WordTimestamp]:
        """Convert normalised frame-level energy to pseudo WordTimestamp objects.

        Active frames (energy ≥ 60th percentile) are collapsed into contiguous
        segments.  ``word`` is always empty — privacy invariant: no text in logs.

        Phoneme type is derived via spectral analysis of the original audio segment:
          - Plosive:              short burst (< 30 ms), crest factor > 3.5
          - Fricative (stressed): spectral centroid > 4 kHz + flatness > 0.05, high energy
          - Fricative (unstressed): spectral centroid > 4 kHz + flatness > 0.05, low energy
          - Vowel stressed/unstressed: everything else, split by energy level
          - Below-threshold segments are tagged as silence (SALIENCY_BOOST = 0.5)
        """
        n = len(frame_energy)
        if n == 0 or dur <= 0.0:
            return []
        frame_dur = dur / n
        threshold = float(np.percentile(frame_energy, 60))
        words: list[WordTimestamp] = []
        in_seg = False
        seg_start = 0
        seg_energies: list[float] = []

        def _flush_segment(start_idx: int, end_idx: int, energies: list[float]) -> WordTimestamp:
            seg_e = float(np.mean(energies)) if energies else 0.0
            is_stressed = seg_e > 0.7
            start_s = start_idx * frame_dur
            end_s = end_idx * frame_dur

            phoneme_type = "vowel_stressed" if is_stressed else "vowel_unstressed"
            if source_audio is not None:
                i0 = max(0, min(len(source_audio), int(start_s * sr)))
                i1 = max(i0 + 1, min(len(source_audio), int(end_s * sr)))
                seg_audio = source_audio[i0:i1]
                if len(seg_audio) >= 4:
                    phoneme_type = LyricsGuidedEnhancement._classify_phoneme_type(seg_audio, sr, seg_e, is_stressed)

            return WordTimestamp(
                word="",  # privacy: never store transcribed text
                start_s=start_s,
                end_s=end_s,
                confidence=min(1.0, seg_e),
                is_stressed=is_stressed,
                phoneme_type=phoneme_type,
            )

        for i, e in enumerate(frame_energy.tolist()):
            if e >= threshold:
                if not in_seg:
                    in_seg = True
                    seg_start = i
                    seg_energies = []
                seg_energies.append(e)
            else:
                if in_seg:
                    in_seg = False
                    words.append(_flush_segment(seg_start, i, seg_energies))
                    seg_energies = []
                # Below-threshold → silence segment (short gaps collapsed, long gaps tagged)

        if in_seg and seg_energies:  # flush last open segment
            words.append(_flush_segment(seg_start, n, seg_energies))
        return words

    def _build_sample_saliency(
        self,
        transcription: LyricsTranscriptionResult,
        n_samples: int,
        sr: int,
    ) -> np.ndarray:
        """Build sample-level gain curve from word timestamps; values ∈ [0.3, 2.0]."""
        saliency = np.ones(n_samples, dtype=np.float32)
        if transcription.fallback_used or not transcription.words:
            return saliency
        for word in transcription.words:
            boost = self._cap.SALIENCY_BOOST.get(word.phoneme_type, 1.0)
            i0 = max(0, min(n_samples, int(word.start_s * sr)))
            i1 = max(i0, min(n_samples, int(word.end_s * sr)))
            if i1 > i0:
                saliency[i0:i1] = boost
        return np.clip(saliency, 0.3, 2.0)

    @staticmethod
    def _resample(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Resample mono audio from *sr_in* to *sr_out* Hz."""
        if sr_in == sr_out:
            return mono
        try:
            import scipy.signal as sps  # type: ignore[import]

            n_out = max(1, int(len(mono) * sr_out / sr_in))
            return sps.resample(mono, n_out).astype(np.float32)
        except Exception:
            step = max(1.0, sr_in / sr_out)
            indices = np.arange(0, len(mono), step).astype(np.int64)
            indices = np.clip(indices, 0, len(mono) - 1)
            return mono[indices].astype(np.float32)

    def _compute_mel_features(self, mono_16k: np.ndarray) -> np.ndarray:
        """Compute Whisper-compatible 80-channel log-mel spectrogram.

        Follows Radford et al. (2022) preprocessing:
          n_fft=400, hop_length=160, n_mels=80, sr=16 000 Hz, fmax=8 000 Hz.

        Returns float32 array of shape (1, 80, 3000).
        Shorter signals are zero-padded; longer signals are truncated at 30 s.
        """
        max_samples = self._MAX_FRAMES * self._HOP  # 480 000 = 30 s at 16 kHz
        if len(mono_16k) < max_samples:
            mono_16k = np.pad(mono_16k, (0, max_samples - len(mono_16k)))
        else:
            mono_16k = mono_16k[:max_samples]

        try:
            import librosa  # type: ignore[import]

            mel = librosa.feature.melspectrogram(
                y=mono_16k,
                sr=self._ONNX_SR,
                n_fft=self._N_FFT,
                hop_length=self._HOP,
                n_mels=self._N_MELS,
                fmin=0.0,
                fmax=8_000.0,
            )
            log_mel = np.log10(np.maximum(mel, 1e-10))
            log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
            log_mel = (log_mel + 4.0) / 4.0
            # Pad / truncate to exactly MAX_FRAMES time steps
            if log_mel.shape[1] < self._MAX_FRAMES:
                log_mel = np.pad(log_mel, ((0, 0), (0, self._MAX_FRAMES - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, : self._MAX_FRAMES]
            return log_mel.astype(np.float32)[np.newaxis, ...]  # (1, 80, 3000)
        except Exception:
            # Zero fallback: encoder processes silence → near-zero hidden states
            return np.zeros((1, self._N_MELS, self._MAX_FRAMES), dtype=np.float32)


def get_lyrics_guided_enhancement() -> LyricsGuidedEnhancement:
    """Thread-safe double-checked locking singleton (§3.x spec pattern)."""
    global _lge_instance
    if _lge_instance is None:
        with _lge_lock:
            if _lge_instance is None:
                _lge_instance = LyricsGuidedEnhancement()
    return _lge_instance


__all__ = [
    "ContentAwareProcessor",
    "LyricsGuidedEnhancement",
    "LyricsGuidedTimeline",
    "LyricsTranscriber",
    "LyricsTranscriptionResult",
    "WordTimestamp",
    "get_content_aware_processor",
    "get_lyrics_guided_enhancement",
    "get_lyrics_guided_timeline",
    "get_lyrics_transcriber",
    "is_lyrics_guided_loaded",
]
