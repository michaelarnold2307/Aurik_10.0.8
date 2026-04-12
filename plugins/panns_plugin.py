"""
PANNs Plugin — Direkte ONNX-Inferenz auf Audio-Array (§4.4 Aurik Spec v9.10.45).

Dieses Plugin ersetzt die dateibasierte panns_inference.AudioTagging durch
vollständige ONNX-basierte Inferenz auf np.ndarray-Signalen (kein Datei-I/O
zur Laufzeit, keine Container-Abhängigkeit).

Modell: models/panns/panns_wavegram_logmel_cnn14.onnx
  - Input:  audio [1, 320000] float32 (10 s @ 32 kHz)
  - Output: clipwise_output [1, 527] float32 (AudioSet-527-Klassen, Sigmoid-Scores)

Spec-Referenzen:
  §4.4: PANNs CNN14 — Audio-Tagging / Instrument-Erkennung
        Top-K=10 Tags; Konfidenz-Schwellen: Vocals ≥ 0.40, Drums ≥ 0.50,
        Instruments ≥ 0.50; steuert Phasen-Aktivierungsmatrix
  §2.9: Aktivierungsmatrix (PANNs-Tag → Phase)
  §3.2: Singleton + Convenience-Pattern (Double-Checked Locking, thread-sicher)
  §3.1: NaN/Inf-Guard am Ausgang
  §3.7: Type-Annotations (PEP 484)
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# §9.7.1 SHA256-Ergebnis-Cache für PANNs-Inferenz (§3.8)
# ---------------------------------------------------------------------------
_tags_cache: dict[str, dict[str, float]] = {}
_tags_cache_lock = threading.Lock()
_TAGS_CACHE_MAX = 128  # FIFO-Trim


def _audio_tags_cache_key(audio: np.ndarray, sr: int) -> str:
    """Deterministischer Cache-Key für PANNsPlugin.get_tags()."""
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    return f"panns:{h.hexdigest()[:16]}"


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AudioSet-527-Klassen — Tag-Name → Indizes in clipwise_output[527]
# Quelle: models/clap/class_labels/audioset_class_labels_indices.json
# Schlüsselnamen entsprechen exakt den Bezeichnern im Orchestrator
# (unified_restorer_v3.py, _select_phases()).
# ---------------------------------------------------------------------------
_TAG_INDEX_MAP: dict[str, list[int]] = {
    # Gesang & Stimme (§2.9: Threshold ≥ 0.40 für Vocals/Speech)
    "Singing voice": [27, 32, 33, 34, 254, 255, 266],
    # Idx 27=Singing, 32=Male singing, 33=Female singing, 34=Child singing,
    #     254=Vocal music, 255=A capella, 266=Song
    "Vocals": [27, 32, 33, 34],
    "Speech": [0, 1, 2, 3],
    # Idx 0=Speech, 1=Male speech, 2=Female speech, 3=Child speech
    # Saiteninstrumente (§2.9: Threshold ≥ 0.50)
    "Guitar": [140, 143, 144, 146],
    # Idx 140=Guitar, 143=Acoustic guitar, 144=Steel guitar, 146=Strum
    "Electric guitar": [141],
    "Bass guitar": [142],
    # Tasteninstrumente (§2.9: Threshold ≥ 0.50)
    "Piano": [153, 154],  # Piano, Electric piano
    "Keyboard (musical)": [152, 155, 157, 158],
    # Idx 152=Keyboard, 155=Organ, 157=Hammond organ, 158=Synthesizer
    # Blasinstrumente (§2.9: Threshold ≥ 0.50)
    "Brass instrument": [185, 186, 188],  # Brass, French horn, Trombone
    "Trumpet": [187],
    "Saxophone": [197],
    # Perkussion (§2.9: Threshold ≥ 0.50)
    "Drum": [164, 162, 163, 165, 168],
    # Idx 164=Drum, 162=Drum kit, 163=Drum machine, 165=Snare, 168=Bass drum
    "Percussion": [161, 174, 176, 179, 180, 181],
    # Idx 161=Percussion, 174=Tambourine, 176=Maraca, 179=Mallet, 180=Marimba,
    #     181=Glockenspiel
    # Streicher
    "Bowed string instrument": [189, 191, 193],  # Bowed string, Violin, Cello
    # Akkordeon (Schlager-Erkennung §2.19.2)
    "Accordion": [209],
    # Genre-Überblick
    "Music": [137, 138],  # Music, Musical instrument
    "Classical music": [237],
    "Jazz": [235],
    "Rock music": [219],
    "Electronic music": [239],
}


# ---------------------------------------------------------------------------
# Singleton-Lock (§3.2 Aurik Spec — Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: PANNsPlugin | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Haupt-Plugin-Klasse
# ---------------------------------------------------------------------------


class PANNsPlugin:
    """PANNs CNN14 — Audio-Tagging via lokaler ONNX-Inferenz.

    Erkennt Instrumente, Gesang und Genre aus np.ndarray-Signalen.
    Kein Datei-I/O, kein Netzwerk-Zugriff, kein Docker.

    Algorithmus:
        1. Audio → mono, resample auf 32 000 Hz (Modell-SR)
        2. Pad / Truncate auf genau 320 000 Samples (10 s @ 32 kHz)
        3. ONNX-Inferenz → clipwise_output [1, 527] float32 (Sigmoid-Scores ∈ [0,1])
        4. Pro Tag: max(scores[indices]) → confidence ∈ [0, 1]

    Invarianten (§3.1):
        - Alle Ausgabe-Scores ∈ [0, 1], NaN/Inf → 0.0
        - Fallback auf leeres Dict wenn ONNX nicht verfügbar (kein Absturz)
        - Thread-safe Singleton via Double-Checked Locking (§3.2)
    """

    _MODEL_SR: int = 32_000
    _MODEL_SAMPLES: int = 320_000  # 10 s × 32 000 Hz
    _ONNX_PATH: Path = Path(__file__).parent.parent / "models" / "panns" / "panns_wavegram_logmel_cnn14.onnx"

    def __init__(self) -> None:
        self._session: object | None = None
        self._load_onnx()

    # ------------------------------------------------------------------
    # ONNX-Laden
    # ------------------------------------------------------------------

    def _load_onnx(self) -> None:
        """Lädt ONNX-Session einmalig lazy; warnt bei Fehler, kein Absturz."""
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("PANNs", size_gb=0.66):
                    logger.warning("PANNs: ML-Budget erschöpft — Spektral-Fallback.")
                    return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            try:
                from backend.core.ml_device_manager import get_ort_providers_fp16 as _get_prov

                _providers = _get_prov("PANNs")
            except Exception:
                _providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(self._ONNX_PATH),
                providers=_providers,
            )
            logger.info(
                "panns_plugin: CNN14 ONNX model loaded (%s, §4.4 primary genre/tagging)",
                self._ONNX_PATH.name,
            )
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("PANNs", size_gb=0.66, unload_fn=lambda s=self: setattr(s, "_session", None))
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning(
                "PANNs ONNX nicht verfügbar — Instrument-Gate inaktiv (alle Phasen sind aktiv): %s",
                exc,
            )
            self._session = None
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("PANNs")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Audio-Aufbereitung
    # ------------------------------------------------------------------

    def _to_model_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Konvertiert Audio-Array auf [1, 320000] float32 @ 32 kHz.

        Algorithmus:
            1. Stereo → Mono (arithmetisches Mittel)
            2. NaN/Inf bereinigen
            3. Resample auf 32 000 Hz (scipy.signal.resample_poly, Fallback: linspace-interp)
            4. Amplituden-Normalisierung auf Spitze ≈ 0.9
            5. Mittlere 10 s extrahieren (repräsentativer als Anfang)
            6. Pad/Truncate auf genau 320 000 Samples

        Args:
            audio: Mono oder Stereo, beliebige Sample-Rate, float32/64.
            sr:    Quell-Sample-Rate in Hz.

        Returns:
            ndarray [1, 320000] float32, NaN/Inf-frei.
        """
        # Stereo → Mono
        if audio.ndim == 2:
            # [channels, samples] oder [samples, channels]
            audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=1)

        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Resample auf Modell-SR
        if sr != self._MODEL_SR:
            try:
                from scipy.signal import resample_poly

                g = math.gcd(self._MODEL_SR, sr)
                audio = resample_poly(
                    audio,
                    self._MODEL_SR // g,
                    sr // g,
                ).astype(np.float32)
            except Exception:
                n_out = max(1, round(len(audio) * self._MODEL_SR / sr))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

        # Amplituden-Normalisierung
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-7:
            audio = (audio / peak * 0.9).astype(np.float32)

        # Pad oder mittlere 10 s extrahieren
        n = len(audio)
        if n < self._MODEL_SAMPLES:
            audio = np.pad(audio, (0, self._MODEL_SAMPLES - n))
        elif n > self._MODEL_SAMPLES:
            start = max(0, (n - self._MODEL_SAMPLES) // 2)
            audio = audio[start : start + self._MODEL_SAMPLES]

        return audio[np.newaxis, :].astype(np.float32)  # [1, 320000]

    # ------------------------------------------------------------------
    # Haupt-Inferenz-Methode
    # ------------------------------------------------------------------

    def get_tags(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Gibt Konfidenz-Dict zurück (Tag-Name → Score ∈ [0, 1]).

        Schlüssel entsprechen den Bezeichnern im Orchestrator
        (unified_restorer_v3.py, _select_phases()). §4.4 Konfidenz-Schwellen:
            - "Singing voice" / "Vocals" / "Speech": threshold ≥ 0.40
            - "Drum" / "Percussion":                 threshold ≥ 0.50
            - alle Instrument-Tags:                  threshold ≥ 0.50

        Args:
            audio: Audio-Signal als np.ndarray (mono oder stereo), float32/64.
            sr:    Sample-Rate in Hz.

        Returns:
            Dict[str, float] oder leeres Dict wenn ONNX nicht verfügbar.
        """
        if self._session is None:
            return {}

        # §9.7.1 SHA256-Cache — teure ONNX-Inferenz nur bei neuem Signal
        _cache_key = _audio_tags_cache_key(audio, sr)
        with _tags_cache_lock:
            if _cache_key in _tags_cache:
                logger.debug("PANNs Cache-Hit: %s", _cache_key)
                return _tags_cache[_cache_key]

        try:
            model_input = self._to_model_input(audio, sr)
            ort_out = self._session.run(
                None,
                {self._session.get_inputs()[0].name: model_input},
            )
            scores: np.ndarray = ort_out[0][0]  # [527] float32

            result: dict[str, float] = {}
            for tag, indices in _TAG_INDEX_MAP.items():
                raw = [float(scores[i]) for i in indices if 0 <= i < len(scores)]
                val = max(raw) if raw else 0.0
                result[tag] = max(0.0, min(1.0, val if math.isfinite(val) else 0.0))

            logger.debug(
                "PANNs-Tags: Vocals=%.2f Speech=%.2f Guitar=%.2f Drum=%.2f Piano=%.2f Brass=%.2f Accordion=%.2f",
                result.get("Singing voice", 0.0),
                result.get("Speech", 0.0),
                result.get("Guitar", 0.0),
                result.get("Drum", 0.0),
                result.get("Piano", 0.0),
                result.get("Brass instrument", 0.0),
                result.get("Accordion", 0.0),
            )
            # §9.7.1 Cache-Write — teure Inferenz nur einmal pro Audio
            with _tags_cache_lock:
                if len(_tags_cache) >= _TAGS_CACHE_MAX:
                    _tags_cache.pop(next(iter(_tags_cache)))  # FIFO-Trim
                _tags_cache[_cache_key] = result
            return result

        except Exception as exc:
            logger.debug("PANNs-Inferenz fehlgeschlagen — leeres Tag-Dict: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Backward-Kompatibilität: dateibasierter Aufruf (Legacy-API)
    # ------------------------------------------------------------------

    def tag(
        self,
        input_wav: str,
        output_json: str | None = None,
    ) -> dict[str, float]:
        """Legacy-Methode: Lädt Audio-Datei und delegiert an get_tags().

        Args:
            input_wav:   Pfad zur Audio-Datei (beliebiges soundfile-Format).
            output_json: Optionaler Pfad für JSON-Ausgabe (wird erstellt falls nötig).

        Returns:
            Selbes Format wie get_tags().
        """
        import json

        try:
            from backend.file_import import load_audio_file

            _res = load_audio_file(str(input_wav), do_carrier_analysis=False)
            audio = np.asarray(_res["audio"], dtype=np.float32)
            sr = int(_res["sr"])
        except Exception as exc:
            logger.error("PANNsPlugin.tag: Datei nicht lesbar '%s': %s", input_wav, exc)
            return {}

        result = self.get_tags(audio, sr)

        if output_json:
            out_path = Path(output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

        return result


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 Aurik Spec)
# ---------------------------------------------------------------------------


def get_panns_plugin() -> PANNsPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PANNsPlugin()
    return _instance


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------


def classify_audio(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Convenience-Wrapper für get_panns_plugin().get_tags()."""
    return get_panns_plugin().get_tags(audio, sr)


# ---------------------------------------------------------------------------
# Backward-Kompatibilität: alter Klassenname PANNSPlugin (alle Großbuchstaben)
# Verwendet von: backend/adaptive_pipeline.py und Legacy-Code
# ---------------------------------------------------------------------------
PANNSPlugin = PANNsPlugin


# ---------------------------------------------------------------------------
# CLI-Nutzung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.debug("Verwendung: panns_plugin.py <audio_datei> [output.json]")
        sys.exit(1)

    from backend.file_import import load_audio_file

    _res = load_audio_file(sys.argv[1])
    _audio, _sr = _res["audio"], int(_res["sr"])
    _tags = classify_audio(_audio, _sr)
    logger.debug("PANNs CNN14 — %s", sys.argv[1])
    for _tag, _score in sorted(_tags.items(), key=lambda x: -x[1]):
        bar = "█" * int(_score * 20)
        logger.debug("  %s  %.4f  %s", _tag, _score, bar)

    if len(sys.argv) > 2:
        import json

        with open(sys.argv[2], "w", encoding="utf-8") as _f:
            json.dump(_tags, _f, indent=2)
        logger.debug("→ JSON gespeichert: %s", sys.argv[2])
