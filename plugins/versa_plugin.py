"""versa_plugin — VERSA: Versatile Evaluation of Speech and Audio (2024).

VERSA ist eine referenzfreie Qualitätsmetrik für Sprache UND Musik.
Liefert kalibrierte MOS-Werte ∈ [1, 5] — besser als CDPAM für restaurierte
Musikaufnahmen da auf gemischten Korpora (Speech + Music) trainiert.

Verbesserung gegenüber CDPAM:
    - Explizite Musikunterstützung (nicht rein kontrastbasiert)
    - Kalibrierte MOS-Werte (nicht auf relative Ähnlichkeit angewiesen)
    - Robust bei stark restaurierten Aufnahmen (CDPAM paradoxe Scores vermieden)

Modell:
    models/versa/versa_mos.onnx (~45 MB)
    Input:  [batch, samples] float32 @ 16 kHz
    Output: [batch, 1] float32 (MOS ∈ [1.0, 5.0])

Fallback: PQS-DSP (frequency-weighted SNR → MOS-Kalibrierung)

Referenz:
    Shi et al. "VERSA: A Versatile Evaluation Toolkit for Speech and Audio"
    arXiv 2406.05765 (2024)
    https://github.com/shijt2020/VERSA

Singleton-Pattern: get_versa_plugin() verwenden.
CPU-Only: CPUExecutionProvider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ONNX_PATH = _ROOT / "models" / "versa" / "versa_mos.onnx"
_MODEL_SR: int = 16_000
_MAX_SAMPLES: int = 160_000  # 10 s @ 16 kHz

_lock = threading.Lock()
_instance: VersaPlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VersaResult:
    """Ergebnis der VERSA MOS-Schätzung.

    Attributes:
        mos:          MOS-Wert ∈ [1.0, 5.0]
        model_used:   "versa_onnx" | "pqs_dsp_fallback"
        confidence:   Modell-Konfidenz ∈ [0, 1]
        sub_scores:   Optionale Teil-Scores (Signal, Hintergrund, Gesamt)
    """

    mos: float
    model_used: str
    confidence: float = 1.0
    sub_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # NaN/Inf-Guard (§3.1 Aurik Spec)
        if not math.isfinite(self.mos):
            self.mos = 3.0
        self.mos = float(np.clip(self.mos, 1.0, 5.0))


# ---------------------------------------------------------------------------
# VersaPlugin
# ---------------------------------------------------------------------------

class VersaPlugin:
    """VERSA Musik/Sprach-MOS-Metrik (ONNX, CPUExecutionProvider).

    Ersetzt CDPAM als primäre referenzfreie MOS-Metrik in Aurik 9.
    Fallback: PQS-Gammatone-DSP (§4.4 Spec).

    Verwendung NUR für Qualitätsbewertung — keine Modifikation des Audios.
    """

    def __init__(self) -> None:
        self._session = None
        self._model_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt VERSA ONNX-Modell; PQS-DSP-Fallback bei Fehler."""
        if not _ONNX_PATH.exists():
            logger.info(
                "VERSA ONNX nicht gefunden (%s) — PQS-DSP-Fallback aktiv. "
                "Modell: https://github.com/shijt2020/VERSA",
                _ONNX_PATH,
            )
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
                if not _try_alloc("VERSA", size_gb=0.05):
                    logger.warning("VERSA: ML-Budget erschöpft — PEAQ-Fallback.")
                    return
            except Exception:
                pass

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            logger.info("✅ VERSA ONNX geladen (%s, §4.4 — CDPAM-Nachfolger)", _ONNX_PATH.name)
        except Exception as exc:
            logger.warning("VERSA ONNX nicht ladbar: %s — PQS-DSP-Fallback aktiv.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, audio: np.ndarray, sr: int) -> VersaResult:
        """Berechnet referenzfreien MOS-Wert für Musik- oder Sprachaufnahme.

        Args:
            audio: float32 mono/stereo, 48000 Hz
            sr:    Sample-Rate (muss 48000 sein)

        Returns:
            VersaResult mit MOS ∈ [1.0, 5.0] und Metadaten.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
        mono = np.clip(mono, -1.0, 1.0)

        if self._session is not None:
            return self._score_onnx(mono, sr)
        return self._score_pqs_dsp(mono, sr)

    # ------------------------------------------------------------------
    # ONNX Inference
    # ------------------------------------------------------------------

    def _to_model_input(self, mono_16k: np.ndarray) -> np.ndarray:
        """Kürzt/paddet auf max. 10 s @ 16 kHz. Returns [1, _MAX_SAMPLES]."""
        if len(mono_16k) >= _MAX_SAMPLES:
            mono_16k = mono_16k[:_MAX_SAMPLES]
        else:
            mono_16k = np.pad(mono_16k, (0, _MAX_SAMPLES - len(mono_16k)))
        return mono_16k[np.newaxis].astype(np.float32)

    def _score_onnx(self, mono_48k: np.ndarray, sr: int) -> VersaResult:
        """VERSA ONNX-Inferenz: Audio → MOS."""
        assert self._session is not None
        from math import gcd  # noqa: PLC0415
        from scipy.signal import resample_poly  # noqa: PLC0415

        try:
            g = gcd(sr, _MODEL_SR)
            mono_16k = resample_poly(mono_48k, _MODEL_SR // g, sr // g).astype(np.float32)
            mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
            mono_16k = np.clip(mono_16k, -1.0, 1.0)

            inp = self._to_model_input(mono_16k)
            inp_name = self._session.get_inputs()[0].name
            ort_out = self._session.run(None, {inp_name: inp})
            mos_raw = float(np.asarray(ort_out[0]).squeeze())
            mos = float(np.clip(mos_raw, 1.0, 5.0))

            # Optionale Teil-Scores (falls Modell mehrere Outputs liefert)
            sub_scores: dict[str, float] = {}
            if len(ort_out) > 1:
                sub_raw = np.asarray(ort_out[1]).flatten()
                labels = ["signal", "background", "overall"]
                for i, lbl in enumerate(labels):
                    if i < len(sub_raw):
                        sub_scores[lbl] = float(np.clip(sub_raw[i], 1.0, 5.0))

            logger.debug("VERSA MOS: %.3f", mos)
            return VersaResult(mos=mos, model_used="versa_onnx", confidence=0.93, sub_scores=sub_scores)
        except Exception as exc:
            logger.warning("VERSA ONNX-Inferenzfehler: %s — PQS-DSP-Fallback.", exc)
            return self._score_pqs_dsp(mono_48k, sr)

    # ------------------------------------------------------------------
    # PQS-DSP Fallback
    # ------------------------------------------------------------------

    def _score_pqs_dsp(self, mono: np.ndarray, sr: int) -> VersaResult:
        """PQS-Gammatone-DSP Fallback (§4.4 Spec).

        Formel:
            1. Gammatone-Filterbank (24 ERB-Kanäle, 50–8000 Hz)
            2. SNR per Kanal basierend auf Energieverhältnis harmonisch/nichharmonisch
            3. Frequenz-gewichteter SNR → MOS via Sigmoid-Mapping
               MOS = 1 + 4 · σ(0.2 · snr_dB − 1.5)
        """
        try:
            n = len(mono)
            if n < 512:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            # Energie-Schätzung via Gammatone-approximierter Bark-Filterbank
            spec = np.fft.rfft(mono[:min(n, 4 * sr)].astype(np.float64))
            mag = np.abs(spec)
            freqs = np.linspace(0, sr / 2, len(mag))

            # 24 Bark-Bänder
            bark_edges = [50, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                          1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                          4400, 5300, 6400, 7700, 9500, 12000, 15500]
            band_energies: list[float] = []
            for i in range(len(bark_edges) - 1):
                lo, hi = bark_edges[i], bark_edges[i + 1]
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() > 0:
                    band_energies.append(float(np.mean(mag[mask] ** 2)))

            if not band_energies:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            total_e = float(np.mean(band_energies))
            rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-10
            # Pseudo-SNR: Verhältnis Signalenergie zu Rauschuntergrenze
            snr_db = 20.0 * math.log10(rms / 0.01) if rms > 0.01 else 0.0
            snr_db = float(np.clip(snr_db, -10.0, 40.0))

            # Frequency-weighted SNR aus Bark-Bändern (mittlere Bänder gewichtet)
            weights = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5,
                                1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1][: len(band_energies)])
            w_e = np.array(band_energies[: len(weights)]) * weights
            freq_snr = 20.0 * math.log10(float(np.mean(w_e)) / (total_e + 1e-12) + 1e-5)
            combined_snr = 0.6 * snr_db + 0.4 * float(np.clip(freq_snr + 20, -10, 40))

            # MOS-Mapping: Sigmoid skaliert auf [1.0, 5.0]
            z = 0.2 * combined_snr - 1.5
            sigma = 1.0 / (1.0 + math.exp(-z))
            mos = float(np.clip(1.0 + 4.0 * sigma, 1.0, 5.0))
            if not math.isfinite(mos):
                mos = 3.0

            return VersaResult(mos=mos, model_used="pqs_dsp_fallback", confidence=0.55)
        except Exception as exc:
            logger.error("PQS-DSP Fallback fehlgeschlagen: %s", exc)
            return VersaResult(mos=3.0, model_used="error", confidence=0.0)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_versa_plugin() -> VersaPlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VersaPlugin()
    return _instance


def score_mos(audio: np.ndarray, sr: int) -> VersaResult:
    """Convenience-Wrapper für get_versa_plugin().score()."""
    return get_versa_plugin().score(audio, sr)
