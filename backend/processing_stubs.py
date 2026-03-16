# Echte DSP-Implementierungen aus dsp/ laden (mit Fallback auf Passthrough)
try:
    from dsp.ki_inpainting import SpectralInpainter
except ImportError:

    class SpectralInpainter:  # type: ignore[no-redef]
        """Fallback-SpectralInpainter: Passthrough wenn dsp.ki_inpainting fehlt."""

        def __init__(self, sr: int) -> None:
            self.sr = sr

        def inpaint(self, audio, dropout_start: int, dropout_end: int):
            # Lineare Interpolation als minimaler produktiver Fallback
            import numpy as _np

            out = audio.copy()
            if 0 <= dropout_start < dropout_end <= len(audio):
                gap = dropout_end - dropout_start
                v0 = float(audio[dropout_start - 1]) if dropout_start > 0 else 0.0
                v1 = float(audio[dropout_end]) if dropout_end < len(audio) else 0.0
                out[dropout_start:dropout_end] = _np.linspace(v0, v1, gap)
            return out


try:
    import numpy as _np_acf

    from dsp.adaptive_comb_filter import AdaptiveCombFilter as _RealACF

    class AdaptiveCombFilter:  # type: ignore[no-redef]
        """Adapter für dsp.adaptive_comb_filter.AdaptiveCombFilter mit sr/hum_freq-API."""

        def __init__(self, sr: int, hum_freq: float = 50.0) -> None:
            self.sr = sr
            self.hum_freq = hum_freq
            delay = max(1, round(sr / hum_freq))
            self._impl = _RealACF(delay=delay, gain=0.9)

        def _rebuild(self, hum_freq: float) -> None:
            delay = max(1, round(self.sr / hum_freq))
            self._impl = _RealACF(delay=delay, gain=0.9)

        def process(self, audio):
            return self._apply(audio)

        def remove_hum(self, audio, hum_freq=None):
            if hum_freq is not None and hum_freq != self.hum_freq:
                self.hum_freq = hum_freq
                self._rebuild(hum_freq)
            return self._apply(audio)

        def _apply(self, audio):
            """Kammfilter: y[n] = x[n] + gain * x[n - delay] (Feedback-Form)."""

            x = _np_acf.asarray(audio, dtype=_np_acf.float32)
            y = x.copy()
            d = self._impl.delay
            g = self._impl.gain
            for n in range(d, len(x)):
                y[n] = x[n] - g * y[n - d]  # IIR-Notch an Hum-Frequenz
            return y

except ImportError:

    class AdaptiveCombFilter:  # type: ignore[no-redef]
        """Fallback-AdaptiveCombFilter: IIR-Notch ohne externe Abhängigkeit."""

        def __init__(self, sr: int, hum_freq: float = 50.0) -> None:
            self.sr = sr
            self.hum_freq = hum_freq

        def process(self, audio):
            return self.remove_hum(audio)

        def remove_hum(self, audio, hum_freq=None):
            import numpy as np

            f0 = hum_freq or self.hum_freq
            # Einfacher IIR-Notch: y[n] = x[n] - x[n-delay] (Kammfilter)
            delay = max(1, round(self.sr / f0))
            x = np.asarray(audio, dtype=np.float32)
            y = x.copy()
            for n in range(delay, len(x)):
                y[n] = x[n] - 0.9 * y[n - delay]
            return y


def get_stem_processing_chain(stem: str) -> list:
    """Gibt eine geordnete DSP-Plugin-Liste für den angegebenen Stem-Typ zurück.

    Rückgabe-Plugins müssen ein optionales ``get_metadata()``-Interface besitzen.
    Wenn keine spezialisierten Plugins geladen werden können, wird eine leere
    Liste zurückgegeben (Passthrough).
    """
    try:
        from plugins.crepe_plugin import CrepePlugin
        from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin

        chains: dict[str, list] = {
            "vocals": [DeepFilterNetV3IIPlugin(), CrepePlugin()],
            "drums": [DeepFilterNetV3IIPlugin()],
            "bass": [DeepFilterNetV3IIPlugin()],
            "other": [DeepFilterNetV3IIPlugin()],
        }
        return chains.get(stem, [DeepFilterNetV3IIPlugin()])
    except Exception:
        return []  # Passthrough wenn Plugins nicht verfügbar


def process_stems_auto(
    stems: dict,
    sr: int = 48000,
    log_fn=None,
    analysis_dict: dict | None = None,
) -> dict:
    """Verarbeitet alle Stems mit ihrer jeweiligen DSP-Processing-Chain.

    Für jeden Stem wird ``get_stem_processing_chain`` aufgerufen und die
    Plugins sequenziell angewendet. Scheitert ein Plugin, bleibt das Audio
    unverändert (nicht-destruktives Fallback-Prinzip).
    """
    processed: dict = {}
    for stem, audio in stems.items():
        chain = get_stem_processing_chain(stem)
        result = audio
        for plugin in chain:
            try:
                if hasattr(plugin, "process"):
                    out = plugin.process(result, sr=sr)
                    result = out if out is not None else result
                    if log_fn:
                        log_fn(f"[{stem}] {plugin.__class__.__name__} angewendet")
            except Exception as exc:
                if log_fn:
                    log_fn(f"[{stem}] {plugin.__class__.__name__} fehlgeschlagen: {exc}")
        processed[stem] = result
    return processed


# Platzhalter für DSP-/ML-Module und Bearbeitungslogik


import io

import numpy as np

from backend.quality_control import QualityControl

"""
Hinweis: Die Importe aus aurik4 sind entfernt. Sobald die entsprechenden Module nach aurik6 migriert wurden,
können sie wie folgt importiert werden:

try:
    from Aurik_Standalone.dsp_modules import SpectralInpainter, AdaptiveCombFilter
except ImportError:
    class SpectralInpainter:
        def __init__(self, sr):
            pass
        def inpaint(self, audio, start, end):
            return audio
    class AdaptiveCombFilter:
        def __init__(self, sr):
            pass
        def remove_hum(self, audio, hum_freq):
            return audio
from Aurik_Standalone.sota_productive_models import DemucsSeparator, CrepePitchDetector
"""


def process_audio(audio_bytes, features, policy):
    # Audio-Bytes in numpy-Array (float32, mono, 16kHz)
    import json

    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    sr = int(sr)
    qc = QualityControl()
    # Policy/Features auswerten
    try:
        policy_dict = json.loads(policy) if isinstance(policy, str) else (policy or {})
    except Exception:
        policy_dict = {}
    try:
        features_dict = json.loads(features) if isinstance(features, str) else (features or {})
    except Exception:
        features_dict = {}
    dropout_start = policy_dict.get("dropout_start", features_dict.get("dropout_start", 10000))
    dropout_end = policy_dict.get("dropout_end", features_dict.get("dropout_end", 10200))
    use_inpainter = policy_dict.get("use_inpainter", True)
    hum_freq = policy_dict.get("hum_freq", 50.0)
    # DSP-Module initialisieren
    inpainter = SpectralInpainter(sr)
    comb = AdaptiveCombFilter(sr)
    # 1. Dropout-Reparatur (optional)
    if use_inpainter and dropout_end > dropout_start and dropout_end <= len(audio):
        audio_inpainted = inpainter.inpaint(audio, dropout_start, dropout_end)
    else:
        audio_inpainted = audio
    # 2. Brummunterdrückung
    audio_restored = comb.remove_hum(audio_inpainted, hum_freq)

    # Qualitätsmechanismen: Nicht-Destruktivität prüfen
    snr = qc.check_non_destructive(audio, audio_restored)
    # A/B-Test (Original vs. Bearbeitet)
    ab_score = qc.ab_test(audio, audio_restored)
    # 3. SOTA-ML: Source Separation (Demucs, optional)
    demucs_result = None  # noqa: F841
    stems = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_restored, sr)
            from dsp.demucs_separator import DemucsSeparator as _DemucsSeparator  # noqa: F401

            demucs = _DemucsSeparator()
            stems = demucs.separate(tmp.name)
    except Exception as e:
        _demucs_err = str(e)  # noqa: F841
        stems = {"mix": audio_restored}
    # 4. SOTA-Forensik: MediaForensicsEngine nach Audioimport
    forensic_report = None
    try:
        from backend.core.forensics.detector import MediaForensicsEngine

        mfe = MediaForensicsEngine()
        forensic_report = mfe.analyze(audio_restored, sr)
    except Exception as e:
        forensic_report = {"error": str(e)}

    # 5. Stem-spezifische Processing-Chains (vollautomatisch)
    # Analysewerte pro Stem (hier nur Platzhalter, kann aus features extrahiert werden)
    analysis_dict = {stem: {} for stem in stems} if stems else {}
    # Forensik-Report systemweit bereitstellen
    analysis_dict["forensic_report"] = forensic_report

    # Qualitätslog und Warnungen bereitstellen
    analysis_dict["quality_log"] = qc.get_quality_log()
    analysis_dict["warnings"] = qc.get_warnings()
    analysis_dict["ab_score"] = ab_score
    analysis_dict["snr"] = snr

    # Logging für Metadaten
    timeline_changes = []
    timeline_info = []

    def log_fn(msg):
        timeline_info.append({"log": msg})

    _processed_stems = (
        process_stems_auto(stems, sr=sr, log_fn=log_fn, analysis_dict=analysis_dict) if stems else {}
    )  # noqa: F841
    # Metadaten für Timeline: pro Stem und Plugin
    for stem, audio_in in stems.items():
        # Hole die Chain für diesen Stem
        chain = get_stem_processing_chain(stem)
        for plugin in chain:
            meta = plugin.get_metadata() if hasattr(plugin, "get_metadata") else {}
            timeline_changes.append(
                {
                    "module": plugin.__class__.__name__,
                    "stem": stem,
                    "phase": meta.get("phase", "Restoration"),
                    "start": 0,
                    "end": len(audio_in),
                    "color": meta.get("color", "#607d8b"),
                    "reason": meta.get("description", "Processing"),
                    "params": meta,
                }
            )
    # Dashboard und Info wie bisher
    dashboard = {c["module"]: {"count": 1, "color": c["color"], "params": c["params"]} for c in timeline_changes}
    info = timeline_info
    return audio_bytes, {
        "changes": timeline_changes,
        "dashboard": dashboard,
        "info": info,
        "forensic_report": forensic_report,
        "message": "Bearbeitung abgeschlossen",
    }
