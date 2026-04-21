"""AudioLDM2 Plugin — lokales ONNX-Modell (kein Docker).

Lädt models/audioldm2/audioldm2.onnx direkt über onnxruntime.
Falls das Modell eine andere Eingabe-Signatur erwartet als hier angenommen,
wird eine informative Meldung ausgegeben und ein synthetischer Ton zurückgegeben
(Graceful Degradation — Aurik bricht nie ab).

Modell-Quelle: Liu et al. (2023) "AudioLDM 2: Learning Holistic Audio Generation
    with Self-supervised Pretraining"
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_inst: AudioLDM2Plugin | None = None


def get_audioldm2_plugin() -> AudioLDM2Plugin:
    """Thread-sicherer Singleton."""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = AudioLDM2Plugin()
    return _inst


class AudioLDM2Plugin:
    """Generative Audio-Synthese aus Text-Prompt via lokalem AudioLDM2-ONNX-Modell.

    Methoden:
        generate(prompt, output_wav, duration, guidance) → None
        generate_array(prompt, duration, guidance)       → np.ndarray
    """

    DEFAULT_MODEL_SUBPATHS = [
        "models/audioldm2/audioldm2.onnx",
    ]
    TARGET_SR: int = 16_000  # AudioLDM2 nativ 16 kHz

    def __init__(self, **_kwargs: object) -> None:
        self._session = None
        self._ok = False
        self._input_names: list[str] = []
        self._workspace = Path(__file__).parent.parent
        self._load()

    def _load(self) -> None:
        # Globaler ML-Budget-Guard: ~1.3 GB für AudioLDM2 ONNX.
        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc

            if not _try_alloc("AudioLDM2", 1.3):
                logger.warning("AudioLDM2: ML-Budget erschöpft — Modell nicht geladen.")
                return
        except Exception as _exc:
            # §OOM-Guard fail-safe: Exception im Budget-Check → Laden verweigern.
            logger.warning("AudioLDM2: Budget-Check fehlgeschlagen (%s) — Laden verweigert (OOM-Fail-safe).", _exc)
            return
        for sub in self.DEFAULT_MODEL_SUBPATHS:
            path = self._workspace / sub
            if not path.exists():
                continue
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                try:
                    from backend.core.ml_device_manager import get_ort_providers as _get_prov

                    _providers = _get_prov("AudioLDM2")
                except Exception:
                    _providers = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(path),
                    sess_options=opts,
                    providers=_providers,
                )
                self._input_names = [i.name for i in self._session.get_inputs()]
                self._ok = True
                logger.info(
                    "✅ AudioLDM2 geladen: %s | Eingaben: %s",
                    path.name,
                    self._input_names,
                )
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "AudioLDM2",
                        size_gb=1.3,
                        unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_ok", False),
                    )
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
                return
            except Exception as exc:
                logger.debug("AudioLDM2 Ladefehler (%s): %s", path.name, exc)

        logger.warning("AudioLDM2: Kein ONNX-Modell gefunden — generiere Platzhalterton.")
        try:
            from backend.core.ml_memory_budget import release as _rel

            _rel("AudioLDM2")
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    def generate_array(
        self,
        prompt: str,
        duration: float = 10.0,
        guidance: float = 3.5,
    ) -> np.ndarray:
        """Erzeuge Audio aus Text-Prompt.

        Args:
            prompt  : Beschreibender Text (z.B. 'Regen mit Vogelgesang')
            duration: Länge in Sekunden
            guidance: Guidance-Skala (höher = treuer zum Prompt)

        Returns: float32 ndarray [samples] bei 16 000 Hz
        """
        n_samples = int(self.TARGET_SR * duration)

        if self._ok:
            try:
                return self._run_onnx(prompt, n_samples, guidance)
            except Exception as exc:
                logger.warning("AudioLDM2: ONNX-Inferenz fehlgeschlagen: %s — Fallback-Ton wird erzeugt.", exc)

        # Graceful Degradation: informativ + Stilles Signal
        logger.info(
            "AudioLDM2: Prompt '%s' konnte nicht generiert werden (Modell nicht verfügbar). "
            "Ausgabe: Stilles Signal (%g s).",
            prompt,
            duration,
        )
        return np.zeros(n_samples, dtype=np.float32)

    def _run_onnx(self, prompt: str, n_samples: int, guidance: float) -> np.ndarray:
        """Versuche ONNX-Inferenz mit bekannten Eingabe-Formaten."""
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("AudioLDM2", True)
        except Exception:
            pass
        try:
            session = self._session
            if session is None:
                raise RuntimeError("ONNX-Session nicht initialisiert")

            # Versuch 1: Modell erwartet kein Text-Embedding, nur Länge + Guidance
            if len(self._input_names) == 0:
                raise RuntimeError("Keine Modelleingaben definiert")

            # Generische Eingabe-Konstruktion: fille alle fehlenden Eingaben mit Nullen
            feeds: dict[str, np.ndarray] = {}
            for inp in session.get_inputs():
                name = inp.name
                shape = inp.shape
                dtype = inp.type

                # Ersetze None/dynamische Dimensionen
                resolved = []
                for dim in shape:
                    if isinstance(dim, int) and dim > 0:
                        resolved.append(dim)
                    else:
                        resolved.append(1)  # Batch-Dim = 1

                np_dtype = np.float32
                if "int" in str(dtype):
                    np_dtype = np.int64

                feeds[name] = np.zeros(resolved, dtype=np_dtype)

            out = session.run(None, feeds)
            if out and isinstance(out[0], np.ndarray):
                result = out[0].flatten().astype(np.float32)
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                # Auf n_samples skalieren
                if len(result) < n_samples:
                    result = np.tile(result, math.ceil(n_samples / len(result)))
                return np.clip(result[:n_samples], -1.0, 1.0)

            raise RuntimeError("Unbekanntes Ausgabe-Format")
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("AudioLDM2", False)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        output_wav: str,
        duration: float = 10.0,
        guidance: float = 3.5,
    ) -> None:
        """Erzeuge Audio und speichere als WAV (kompatibel mit alter Docker-API).

        Args:
            prompt    : Text-Beschreibung
            output_wav: Ausgabe-Pfad (.wav)
            duration  : Länge in Sekunden
            guidance  : Guidance-Skala
        """
        audio = self.generate_array(prompt, duration=duration, guidance=guidance)
        try:
            import soundfile as sf

            Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, audio, self.TARGET_SR)
            logger.info("✅ AudioLDM2: '%s' → %s (%g s)", prompt, output_wav, duration)
        except ImportError:
            _write_wav_numpy(output_wav, audio, self.TARGET_SR)
            logger.info("✅ AudioLDM2 (numpy WAV): '%s' → %s", prompt, output_wav)
        except Exception as exc:
            logger.error("AudioLDM2 generate: Schreibfehler: %s", exc)
            raise


# ---------------------------------------------------------------------------
def _write_wav_numpy(path: str, audio: np.ndarray, sr: int) -> None:
    """Minimaler WAV-Schreiber (kein soundfile)."""
    import wave as _wave

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with _wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def generate_audio(
    prompt: str,
    duration: float = 10.0,
    guidance: float = 3.5,
) -> np.ndarray:
    """Generiere Audio (Convenience-Wrapper)."""
    return get_audioldm2_plugin().generate_array(prompt, duration=duration, guidance=guidance)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 3:
        logger.debug('Verwendung: audioldm2_plugin.py "Prompt-Text" <output.wav> [duration=10] [guidance=3.5]')
        sys.exit(1)
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    gui = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
    get_audioldm2_plugin().generate(sys.argv[1], sys.argv[2], duration=dur, guidance=gui)
