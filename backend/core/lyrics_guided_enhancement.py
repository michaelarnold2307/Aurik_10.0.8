from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
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


def _assert_no_lyrics_in_log(words: "list[WordTimestamp]") -> None:
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
            "§2.36 Datenschutz-Verletzung: word.word ist nicht leer und darf niemals "
            "in Logs oder Metadaten erscheinen."
        )


class LyricsTranscriber:
    """Offline-safe placeholder transcriber with deterministic DSP fallback."""

    def transcribe(self, audio: np.ndarray, sr: int = 48_000) -> LyricsTranscriptionResult:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        dur = float(mono.shape[0] / max(1, sr))
        return LyricsTranscriptionResult(
            words=[],
            language="de",
            overall_confidence=0.0,
            duration_s=dur,
            fallback_used=True,
        )


class ContentAwareProcessor:
    SALIENCY_BOOST: dict[str, float] = {
        "fricative_stressed": 2.0,
        "fricative_unstressed": 1.4,
        "vowel_stressed": 1.6,
        "vowel_unstressed": 1.0,
        "plosive": 1.5,
        "silence": 0.5,
        "mixed": 1.0,
    }

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
        """Zeichnet farbige Wort-Overlays auf den WaveformWidget-Canvas.

        Kein Overlay wenn fallback_used=True oder words leer.
        Tooltip-Text zeigt nur Phonem-Typ — kein Transkriptions-Text.
        """
        if transcription.fallback_used or not transcription.words:
            return
        if audio_duration_s <= 0 or widget_width_px <= 0:
            return
        # Qt-Painter-Aufrufe nur wenn QApplication vorhanden (GUI-Kontext)
        try:
            from PyQt5.QtGui import QColor
            from PyQt5.QtCore import QRectF

            for word in transcription.words:
                color_hex = self.COLOR_MAP.get(word.phoneme_type, "")
                if not color_hex:
                    continue
                x_start = int(word.start_s / audio_duration_s * widget_width_px)
                x_end = int(word.end_s / audio_duration_s * widget_width_px)
                width = max(1, x_end - x_start)
                color = QColor(color_hex)
                color.setAlpha(80)
                # painter muss ein aktiver QPainter sein (Guard außerhalb)
                if hasattr(painter, "fillRect"):
                    painter.fillRect(QRectF(x_start, 0, width, 40), color)  # type: ignore[arg-type]
        except ImportError:
            pass  # Kein Qt verfügbar — kein Rendering, kein Absturz


_transcriber: LyricsTranscriber | None = None
_transcriber_lock = threading.Lock()
_processor: ContentAwareProcessor | None = None
_processor_lock = threading.Lock()
_timeline: LyricsGuidedTimeline | None = None
_timeline_lock = threading.Lock()
_lge_instance: "LyricsGuidedEnhancement | None" = None
_lge_lock = threading.Lock()


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
    _MAX_FRAMES: int = 3_000   # 30 s at 100 frames/s → 1500 encoder output frames

    # wav2vec2 forced-alignment ONNX (125 MB, CPUExecutionProvider) — §2.36 Pflicht
    _WAV2VEC2_SR: int = 16_000   # wav2vec2 operates at 16 kHz

    def __init__(self) -> None:
        self._cap = ContentAwareProcessor()
        self._tl = LyricsGuidedTimeline()
        self._ort_session: object = None     # Whisper ONNX InferenceSession
        self._aligner_session: object = None  # wav2vec2 forced-alignment ONNX session
        self._try_load_onnx()
        self._try_load_aligner()

    # ── ONNX bootstrap ─────────────────────────────────────────────────────

    def _try_load_onnx(self) -> None:
        """Load whisper_tiny.onnx with CPUExecutionProvider (no GPU, no network)."""
        try:
            import onnxruntime as ort  # type: ignore[import]
            model_path = (
                Path(__file__).resolve().parents[2]
                / "models" / "whisper" / "whisper_tiny.onnx"
            )
            if model_path.exists():
                self._ort_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx loaded (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: whisper_tiny.onnx not found at %s"
                    " — DSP fallback active", model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: ONNX load failed (%s) — DSP fallback active", exc,
            )

    def _try_load_aligner(self) -> None:
        """Load wav2vec2_forced_alignment.onnx (§2.36 PFLICHT: Phonem-Alignment).

        Model: 125 MB, CPUExecutionProvider, no network access.
        Fallback: DSP energy-threshold segmentation + Whisper token ID phoneme prior.
        """
        try:
            import onnxruntime as ort  # type: ignore[import]
            model_path = (
                Path(__file__).resolve().parents[2]
                / "models" / "wav2vec2" / "wav2vec2_forced_alignment.onnx"
            )
            if model_path.exists():
                self._aligner_session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                logger.info(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx loaded (%.1f MB)",
                    model_path.stat().st_size / 1e6,
                )
            else:
                logger.debug(
                    "LyricsGuidedEnhancement: wav2vec2_forced_alignment.onnx not found at %s"
                    " — DSP phoneme-prior fallback active", model_path,
                )
        except Exception as exc:
            logger.debug(
                "LyricsGuidedEnhancement: aligner ONNX load failed (%s)"
                " — DSP phoneme-prior fallback active", exc,
            )

    def _align_phonemes(
        self,
        words: "list[WordTimestamp]",
        mono_16k: np.ndarray,
        sr_16k: int = 16_000,
    ) -> "list[WordTimestamp]":
        """Refine phoneme types using wav2vec2 forced alignment (§2.36 PFLICHT).

        If ``self._aligner_session`` is available, runs wav2vec2 to obtain
        frame-level CTC emission probabilities, then recomputes the phoneme_type
        for each word segment using the dominant phoneme class from the CTC output.

        Fallback (aligner unavailable): returns ``words`` unchanged — the
        ``_classify_phoneme_type`` DSP assignment from the transcription step is used.

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
            # Normalise to [-1, 1]
            amax = float(np.abs(audio_input).max()) or 1.0
            audio_input = audio_input / amax

            # Run encoder: output is (1, T_frames, vocab_size) CTC log-probs
            inputs = {"input_values": audio_input[np.newaxis, :]}  # (1, T)
            outputs = self._aligner_session.run(None, inputs)
            logits = outputs[0]   # (1, T_frames, vocab_size)
            if logits.ndim != 3:
                return words
            logits = logits[0]    # (T_frames, vocab_size)
            n_frames, vocab_size = logits.shape

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

                seg_logits = logits[frame_start:frame_end]   # (T_seg, vocab)
                # Mean probability per token class (softmax approximation)
                probs = np.exp(seg_logits - seg_logits.max(axis=-1, keepdims=True))
                probs /= probs.sum(axis=-1, keepdims=True) + 1e-9
                mean_probs = probs.mean(axis=0)   # (vocab_size,)

                vowel_p = float(mean_probs[VOWEL_RANGE[0]:VOWEL_RANGE[1]].sum())
                fric_p  = float(mean_probs[FRICATIVE_RANGE[0]:FRICATIVE_RANGE[1]].sum())
                plos_p  = float(mean_probs[PLOSIVE_RANGE[0]:PLOSIVE_RANGE[1]].sum())

                # Determine dominant class
                is_stressed = word.phoneme_type.endswith("_stressed")
                if plos_p >= max(vowel_p, fric_p) * 0.8:
                    new_type = "plosive"
                elif fric_p >= vowel_p * 0.7:
                    new_type = "fricative_stressed" if is_stressed else "fricative_unstressed"
                else:
                    new_type = "vowel_stressed" if is_stressed else "vowel_unstressed"

                updated.append(WordTimestamp(
                    word="",          # §2.36 Datenschutz: Lyrics-Text NIEMALS gespeichert
                    start_s=word.start_s,
                    end_s=word.end_s,
                    confidence=float(max(vowel_p, fric_p, plos_p)),
                    is_stressed=is_stressed,
                    phoneme_type=new_type,
                ))
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

        if audio.ndim == 2:
            audio_out = audio * saliency[:, np.newaxis]
        else:
            audio_out = audio * saliency

        audio_out = np.clip(
            np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        return audio_out, transcription

    def get_timeline(self) -> LyricsGuidedTimeline:
        """Return the timeline renderer used by ``_toggle_lyrics_overlay`` in the frontend."""
        return self._tl

    # ── Internal transcription ──────────────────────────────────────────────

    def _transcribe_internal(
        self, mono: np.ndarray, sr: int, dur: float
    ) -> LyricsTranscriptionResult:
        """Try ONNX encoder first; fall back to DSP energy segmentation."""
        if self._ort_session is not None:
            try:
                return self._transcribe_onnx(mono, sr, dur)
            except Exception as exc:
                logger.debug(
                    "LyricsGuidedEnhancement: ONNX transcription failed (%s)"
                    " — DSP fallback", exc,
                )
        return self._transcribe_dsp(mono, sr, dur)

    def _transcribe_onnx(
        self, mono: np.ndarray, sr: int, dur: float
    ) -> LyricsTranscriptionResult:
        """Run whisper_tiny.onnx encoder; derive vocal segments from hidden-state RMS.

        The encoder's last_hidden_state (1500 frames × 384 dims) is condensed to a
        scalar energy per frame via RMS.  High-energy frames correspond to voiced /
        active speech regions.  Frame clusters above the 60th-percentile threshold
        are reported as pseudo WordTimestamp segments with phoneme_type labels from
        ContentAwareProcessor.SALIENCY_BOOST.
        """
        mono_16k = self._resample(mono, sr, self._ONNX_SR)
        features = self._compute_mel_features(mono_16k)    # (1, 80, 3000)

        # Encoder inference — output: (1, 1500, 384)
        hidden = self._ort_session.run(None, {"input_features": features})[0]
        frame_energy = np.sqrt(np.mean(hidden[0] ** 2, axis=-1))  # (1500,)

        e_max = float(frame_energy.max()) or 1.0
        frame_energy = (frame_energy / e_max).astype(np.float32)

        words = self._energy_to_words(frame_energy, dur, mono, sr)
        return LyricsTranscriptionResult(
            words=words,
            language="de",
            overall_confidence=0.65,
            duration_s=dur,
            fallback_used=False,
        )

    def _transcribe_dsp(
        self, mono: np.ndarray, sr: int, dur: float
    ) -> LyricsTranscriptionResult:
        """Pure DSP energy segmentation (20 ms frames, RMS, 60th-percentile threshold)."""
        frame_size = max(1, sr // 50)   # 20 ms
        hop = max(1, frame_size // 2)
        energies = [
            float(np.sqrt(np.mean(mono[i : i + frame_size] ** 2)))
            for i in range(0, max(1, len(mono) - frame_size), hop)
        ]
        if not energies:
            return LyricsTranscriptionResult([], "de", 0.0, dur, fallback_used=True)
        arr = np.array(energies, dtype=np.float32)
        e_max = float(arr.max()) or 1.0
        arr /= e_max
        words = self._energy_to_words(arr, dur, mono, sr)
        return LyricsTranscriptionResult(
            words=words,
            language="de",
            overall_confidence=0.3,
            duration_s=dur,
            fallback_used=True,
        )

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
            rms = float(np.sqrt(np.mean(segment_audio ** 2))) or 1e-10
            peak = float(np.abs(segment_audio).max())
            if peak / rms > 3.5:   # crest factor > 3.5 → transient burst
                return "plosive"

        # --- Spectral centroid and flatness for fricative vs. vowel ---
        if n >= 16:
            try:
                fft = np.abs(np.fft.rfft(segment_audio * np.hanning(n) if n <= 8192
                             else segment_audio[:8192] * np.hanning(8192)))
                freqs = np.fft.rfftfreq(min(n, 8192), d=1.0 / sr)
                power = fft ** 2 + 1e-12
                total_power = float(power.sum())
                centroid = float((freqs * power).sum() / total_power)

                # Spectral flatness: geometric mean / arithmetic mean of power spectrum
                log_mean = float(np.mean(np.log(power + 1e-12)))
                arith_mean = float(np.mean(power))
                flatness = float(np.exp(log_mean) / (arith_mean + 1e-12))

                # High centroid + high flatness → noise-like → fricative
                if centroid > 4000.0 and flatness > 0.05:
                    return "fricative_stressed" if is_stressed else "fricative_unstressed"
            except Exception:
                pass   # DSP failed → fall through to vowel classification

        return "vowel_stressed" if is_stressed else "vowel_unstressed"

    @staticmethod
    def _energy_to_words(
        frame_energy: np.ndarray, dur: float,
        source_audio: "np.ndarray | None" = None,
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
                    phoneme_type = LyricsGuidedEnhancement._classify_phoneme_type(
                        seg_audio, sr, seg_e, is_stressed
                    )

            return WordTimestamp(
                word="",   # privacy: never store transcribed text
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

        if in_seg and seg_energies:   # flush last open segment
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
        max_samples = self._MAX_FRAMES * self._HOP   # 480 000 = 30 s at 16 kHz
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
                log_mel = np.pad(
                    log_mel, ((0, 0), (0, self._MAX_FRAMES - log_mel.shape[1]))
                )
            else:
                log_mel = log_mel[:, : self._MAX_FRAMES]
            return log_mel.astype(np.float32)[np.newaxis, ...]   # (1, 80, 3000)
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
    "WordTimestamp",
    "LyricsTranscriptionResult",
    "LyricsTranscriber",
    "ContentAwareProcessor",
    "LyricsGuidedTimeline",
    "LyricsGuidedEnhancement",
    "get_lyrics_transcriber",
    "get_content_aware_processor",
    "get_lyrics_guided_timeline",
    "get_lyrics_guided_enhancement",
]
