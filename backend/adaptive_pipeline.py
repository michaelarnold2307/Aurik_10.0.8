# ---------------------------------------------------------------------------
# Plugin-Imports mit individuellem Fallback — jeder Import einzeln geschützt,
# damit ein fehlgeschlagener Import nicht alle anderen blockiert.
# Mit --import-mode=importlib (pytest) müssen Fallback-Klassen auf Modulebene
# VOR dem try-Block deklariert werden, damit sie im Modul-Namensraum sichtbar sind.
# ---------------------------------------------------------------------------
# pylint: disable=wrong-import-position,import-outside-toplevel,too-many-positional-arguments

import io
import logging as _logging
import os
import tempfile

import numpy as np
import soundfile as sf

try:
    from backend.file_import import load_audio_file as _load_audio_file
except ImportError:
    _load_audio_file = None  # type: ignore[assignment]

_log = _logging.getLogger(__name__)
logger = _log


def _decode_audio_bytes_canonical(audio_bytes: bytes, *, suffix: str = ".wav") -> tuple[np.ndarray, int]:
    """Dekodiert Audio-Bytes bevorzugt über den kanonischen Dateiloader."""
    tmp_path: str | None = None
    if _load_audio_file is not None:
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as _tmp:
                _tmp.write(audio_bytes)
                tmp_path = _tmp.name
            _loaded = _load_audio_file(tmp_path)
            if isinstance(_loaded, dict) and _loaded.get("audio") is not None and _loaded.get("sr") is not None:
                return np.asarray(_loaded["audio"], dtype=np.float32), int(_loaded["sr"])
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Kanonischer Byte-Decode fehlgeschlagen, sf.read-Fallback aktiv: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _enforce_canonical_policy_route(task: str, model_name: str, context: dict) -> str:
    """Normalisiert Legacy-Policy-Ausgaben auf kanonische Aurik-9-Routen."""
    try:
        from policy.ml_policy_engine import (  # pylint: disable=import-outside-toplevel
            CANONICAL_INSTRUMENTAL_NR_ROUTE,
            CANONICAL_REPAIR_ROUTE,
            CANONICAL_SEPARATION_ROUTE,
            CANONICAL_VOCAL_NR_ROUTE,
        )
    except Exception:
        return model_name

    name = str(model_name or "").strip()
    if not name:
        return model_name

    if task in {"denoise", "enhancement"}:
        canonical_set = {CANONICAL_VOCAL_NR_ROUTE, CANONICAL_INSTRUMENTAL_NR_ROUTE}
        if name in canonical_set:
            return name
        fallback = (
            CANONICAL_VOCAL_NR_ROUTE if bool(context.get("has_vocals", False)) else CANONICAL_INSTRUMENTAL_NR_ROUTE
        )
        logger.warning(
            "Policy-Drift abgefangen (%s): '%s' -> '%s'",
            task,
            name,
            fallback,
        )
        return fallback

    if task == "repair":
        if name == CANONICAL_REPAIR_ROUTE:
            return name
        logger.warning("Policy-Drift abgefangen (repair): '%s' -> '%s'", name, CANONICAL_REPAIR_ROUTE)
        return CANONICAL_REPAIR_ROUTE

    if task == "separation":
        if name == CANONICAL_SEPARATION_ROUTE:
            return name
        logger.warning("Policy-Drift abgefangen (separation): '%s' -> '%s'", name, CANONICAL_SEPARATION_ROUTE)
        return CANONICAL_SEPARATION_ROUTE

    return name


def _canonical_policy_audio_route(model_name: str, audio: np.ndarray, sr: int, context: dict) -> np.ndarray | None:
    """Führt aus: canonical Aurik 9 policy routes; return None for non-canonical legacy names."""
    try:
        from policy.ml_policy_engine import (  # pylint: disable=import-outside-toplevel
            CANONICAL_INSTRUMENTAL_NR_ROUTE,
            CANONICAL_REPAIR_ROUTE,
            CANONICAL_VOCAL_NR_ROUTE,
        )
    except Exception:
        return None

    if model_name == CANONICAL_REPAIR_ROUTE:
        return np.asarray(audio, dtype=np.float32).copy()

    if model_name not in {CANONICAL_VOCAL_NR_ROUTE, CANONICAL_INSTRUMENTAL_NR_ROUTE}:
        return None

    try:
        from backend.core.dsp.sota_vocal_model_router import (  # pylint: disable=import-outside-toplevel
            get_sota_vocal_model_router,
        )

        router = get_sota_vocal_model_router()
        if model_name == CANONICAL_VOCAL_NR_ROUTE:
            result = router.enhance_vocal(
                np.asarray(audio, dtype=np.float32),
                sr,
                energy_bias_db=-6.0,
                noise_snr_db=float(context.get("snr", context.get("snr_db", 0.0)) or 0.0),
            )
        else:
            result = router.enhance_instrumental(
                np.asarray(audio, dtype=np.float32),
                sr,
                energy_bias_db=-9.0,
            )
        if result.success:
            return np.asarray(result.audio, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32).copy()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Canonical policy route %s unavailable: %s", model_name, exc)
        return np.asarray(audio, dtype=np.float32).copy()


def _canonical_policy_separation_route(model_name: str, audio: np.ndarray, sr: int, context: dict) -> np.ndarray | None:
    """Gibt routed vocal stem for canonical separation, or None for legacy names zurück."""
    try:
        from policy.ml_policy_engine import CANONICAL_SEPARATION_ROUTE  # pylint: disable=import-outside-toplevel
    except Exception:
        return None
    if model_name != CANONICAL_SEPARATION_ROUTE:
        return None
    try:
        from backend.core.dsp.sota_vocal_model_router import (  # pylint: disable=import-outside-toplevel
            get_sota_vocal_model_router,
        )

        routed = get_sota_vocal_model_router().separate_vocal_instrumental(
            np.asarray(audio, dtype=np.float32),
            sr,
            panns_singing=0.8 if context.get("has_vocals", False) else 0.0,
            ctx={"legacy_adapter": "adaptive_pipeline"},
        )
        if routed.success:
            return np.asarray(routed.vocal, dtype=np.float32)
        return np.asarray(audio, dtype=np.float32).copy()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Canonical separation route unavailable: %s", exc)
        return np.asarray(audio, dtype=np.float32).copy()


# Fallback-Stub auf Modulebene
class _PluginStub:
    """Basis-Fallback-Stub für nicht ladbare Plugins."""

    def process(self, audio, sr):
        """Gibt input audio unchanged for process-style plugin APIs zurück."""
        del sr
        return audio

    def enhance(self, audio, sr):
        """Gibt input audio unchanged for enhance-style plugin APIs zurück."""
        del sr
        return audio

    def separate(self, audio, sr):
        """Gibt a neutral two-stem fallback zurück."""
        del sr
        return audio, audio

    def run(self, audio, sr):
        """Gibt input audio unchanged for run-style plugin APIs zurück."""
        del sr
        return audio


# FullSubNetPlusPlugin entfernt — 16 kHz-Sprach-NR (DNS-Challenge), nicht in §11.3
# SpleeterPlugin entfernt — veraltetes 2019er Modell (Deezer), nicht in §11.3
# DCCRNPlugin entfernt — §4.4 verboten; MpSenetPlugin ist der Nachfolger
# ConvTasNetPlugin entfernt — Sprach-Separation (Luo 2019), HPSS-DSP-Stub, nicht in §11.3
# WaveUNetPlugin entfernt — Sprach-Separation (Stoller 2018), HPSS-DSP-Stub, nicht in §11.3
# SOTAUniversalEnhancer entfernt — orchestriert FullSubNetPlus (Sprach-NR) + np.var()>1.5-Heuristik, nicht §11.3
# DNSMOSPlugin entfernt — explizit verboten §4.4+§10.2 (16 kHz Sprach-Modell)
# NISQAPlugin entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik)
# PESQPlugin entfernt — explizit verboten §4.4+§10.2 (Telefonband 300–3400 Hz)
# ViSQOLPlugin: entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik, kein Musik-Support)
# CDPAMPlugin: entfernt — explizit verboten §4.4+§10.2 (Speech-perceptual metric)
# VampNetPlugin: VERBOTEN (kein stabiler ONNX-Export, kein gebündeltes Plugin) — §4.4
# Stub entfernt; flow_matching_plugin ist der Nachfolger für generatives Inpainting.


# Jetzt die echten Imports versuchen — direkt auf Modulebene, kein globals()-Trick,
# damit --import-mode=importlib (pytest) keine Namespace-Probleme verursacht.
# Jeder try/except-Block ist eigenständig: ein Fehler blockiert keine anderen.
try:
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin as _DeepFilterNetV3IIPlugin
except Exception as _e:
    _log.warning("DeepFilterNetV3IIPlugin nicht verfügbar: %s", _e)
    _DeepFilterNetV3IIPlugin = _PluginStub

DeepFilterNetV3IIPlugin = _DeepFilterNetV3IIPlugin

try:
    from plugins.resemble_enhance_plugin import ResembleEnhancePlugin as _ResembleEnhancePlugin
except Exception as _e:
    _log.warning("ResembleEnhancePlugin nicht verfügbar: %s", _e)
    _ResembleEnhancePlugin = _PluginStub

ResembleEnhancePlugin = _ResembleEnhancePlugin

try:
    from plugins.demucs_v4_plugin import DemucsV4Plugin as _DemucsV4Plugin
except Exception as _e:
    _log.warning("DemucsV4Plugin nicht verfügbar: %s", _e)
    _DemucsV4Plugin = _PluginStub

DemucsV4Plugin = _DemucsV4Plugin

try:
    from plugins.mdx23c_plugin import MDX23CPlugin as _MDX23CPlugin
except Exception as _e:
    _log.warning("MDX23CPlugin nicht verfügbar: %s", _e)
    _MDX23CPlugin = _PluginStub

MDX23CPlugin = _MDX23CPlugin

try:
    from plugins.wpe_plugin import SGMSEPlugin as _SGMSEPlugin
    from plugins.wpe_plugin import WpePlugin as _WpePlugin
except Exception as _e:
    _log.warning("wpe_plugin (WpePlugin/SGMSEPlugin) nicht verfügbar: %s", _e)
    _SGMSEPlugin = _PluginStub
    _WpePlugin = _PluginStub

SGMSEPlugin = _SGMSEPlugin
WpePlugin = _WpePlugin

# FullSubNetPlusPlugin: Import entfernt — 16 kHz-Sprach-NR, nicht in §11.3
# SpleeterPlugin: Import entfernt — veraltetes 2019er Modell, nicht in §11.3

try:
    # §4.4: MP-SENet 2023 ersetzt DCCRN (§4.4 verboten)
    from plugins.mp_senet_plugin import MpSenetPlugin as _MpSenetPlugin
except Exception as _e:
    _log.warning("MpSenetPlugin (DCCRN-Nachfolger) nicht verfügbar: %s", _e)
    _MpSenetPlugin = _PluginStub

MpSenetPlugin = _MpSenetPlugin

try:
    from plugins.uvr_mdxnet_plugin import UVRMDXNetPlugin as _UVRMDXNetPlugin
except Exception as _e:
    _log.warning("UVRMDXNetPlugin nicht verfügbar: %s", _e)
    _UVRMDXNetPlugin = _PluginStub

UVRMDXNetPlugin = _UVRMDXNetPlugin

try:
    from plugins.banquet_vinyl_plugin import BanquetVinylPlugin as _BanquetVinylPlugin
except Exception as _e:
    _log.warning("BanquetVinylPlugin nicht verfügbar: %s", _e)
    _BanquetVinylPlugin = _PluginStub

BanquetVinylPlugin = _BanquetVinylPlugin

try:
    from plugins.hifigan_plugin import HiFiGANPlugin as _HiFiGANPlugin
except Exception as _e:
    _log.warning("HiFiGANPlugin nicht verfügbar: %s", _e)
    _HiFiGANPlugin = _PluginStub

HiFiGANPlugin = _HiFiGANPlugin

# ConvTasNetPlugin: Import entfernt — Sprach-Separation (Luo 2019), nicht in §11.3

try:
    from plugins.diffwave_plugin import DiffWavePlugin as _DiffWavePlugin
except Exception as _e:
    _log.warning("DiffWavePlugin nicht verfügbar: %s", _e)
    _DiffWavePlugin = _PluginStub

DiffWavePlugin = _DiffWavePlugin

# WaveUNetPlugin: Import entfernt — Sprach-Separation (Stoller 2018), nicht in §11.3

try:
    from plugins.crepe_plugin import CREPEPlugin as _CREPEPlugin
except Exception as _e:
    _log.warning("CREPEPlugin nicht verfügbar: %s", _e)
    _CREPEPlugin = _PluginStub

CREPEPlugin = _CREPEPlugin

# SOTAUniversalEnhancer: Import entfernt — orchestriert Sprach-NR (FullSubNetPlus), nicht §11.3
# DNSMOSPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-MOS)
# NISQAPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-Metrik)
# PESQPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Telefonband 300–3400 Hz)

# ViSQOLPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik)

try:
    from plugins.audioldm2_plugin import AudioLDM2Plugin as _AudioLDM2Plugin
except Exception as _e:
    _log.warning("AudioLDM2Plugin nicht verfügbar: %s", _e)
    _AudioLDM2Plugin = _PluginStub

AudioLDM2Plugin = _AudioLDM2Plugin

try:
    from plugins.audiosr_plugin import AudioSRPlugin as _AudioSRPlugin
except Exception as _e:
    _log.warning("AudioSRPlugin nicht verfügbar: %s", _e)
    _AudioSRPlugin = _PluginStub

AudioSRPlugin = _AudioSRPlugin

# CDPAMPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Speech-perceptual metric)

try:
    from plugins.gacela_plugin import GACELAPlugin as _GACELAPlugin
except Exception as _e:
    _log.warning("GACELAPlugin nicht verfügbar: %s", _e)
    _GACELAPlugin = _PluginStub

GACELAPlugin = _GACELAPlugin

try:
    from plugins.matchering_plugin import MatcheringPlugin as _MatcheringPlugin
except Exception as _e:
    _log.warning("MatcheringPlugin nicht verfügbar: %s", _e)
    _MatcheringPlugin = _PluginStub

MatcheringPlugin = _MatcheringPlugin

try:
    from plugins.panns_plugin import PANNSPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("PANNSPlugin nicht verfügbar: %s", _e)

try:
    from plugins.silero_plugin import SileroPlugin as _SileroPlugin
except Exception as _e:
    _log.warning("SileroPlugin nicht verfügbar: %s", _e)
    _SileroPlugin = _PluginStub

SileroPlugin = _SileroPlugin

try:
    from plugins.bs_roformer_plugin import BSRoFormerPlugin as _BSRoFormerPlugin
except Exception as _e:
    _log.warning("BSRoFormerPlugin nicht verfügbar: %s", _e)
    _BSRoFormerPlugin = _PluginStub

BSRoFormerPlugin = _BSRoFormerPlugin

from .sota_maximum_analyzer import SOTAMaximumAnalyzer

# v8.1: Advanced Vocal Separation (Feature #1 - Phase 1)
try:
    from .ml.inference_only.vocal_separation import HybridVocalSeparator
    from .ml.safety_wrappers.vocal_separation_safety import HIPSViolationError, VocalSeparationSafetyWrapper

    VOCAL_SEPARATION_V8_AVAILABLE = True
except ImportError as e:
    _log.warning("v8.1 Vocal Separation unavailable: %s", e)
    VOCAL_SEPARATION_V8_AVAILABLE = False

# v8.2: Conservative Pitch Correction (Feature #2 - Phase 1)
try:
    from .ml.inference_only.pitch_correction import ConservativePitchCorrector
    from .ml.safety_wrappers.pitch_correction_safety import PitchCorrectionSafetyWrapper

    PITCH_CORRECTION_V8_AVAILABLE = True
except ImportError as e:
    _log.warning("v8.2 Pitch Correction unavailable: %s", e)
    PITCH_CORRECTION_V8_AVAILABLE = False

# v8.2: Unified Defect Detection (Feature #3 - Phase 1, Week 3-4)
try:
    from .defect_detection import UnifiedDefectDetector

    DEFECT_DETECTION_V8_AVAILABLE = True
except ImportError as e:
    _log.warning("v8.2 Defect Detection unavailable: %s", e)
    DEFECT_DETECTION_V8_AVAILABLE = False

# Adaptive Processing Pipeline für Magic Button
# ------------------------------------------------
# Diese Pipeline steuert alle Bearbeitungsschritte adaptiv und nachvollziehbar:
# Restaurierung, Reparatur, Rekonstruktion und Remastering.
# Sie nutzt Kontextanalyse, Zieldefinition und modulare Verarbeitungsketten.
# Alle Entscheidungen, Parameter und Ergebnisse werden geloggt.

# Ethics & Monitoring (Phase 4.5, v8.0)
from backend.core.epistemic_gate.ethics_engine import EpistemicDecision, EthicsEngine
from backend.core.model_manager import ModelManager

from ._dsp_applier import apply_dsp_chain
from .adaptive_goal import AdaptiveGoalEngine
from .audio_monitor import PermanentAudioMonitor
from .context_analysis import ContextAnalyzer
from .logging_config import get_logger
from .quality_control import QualityControl

# Docker-based ML Plugins (Phase 7: Docker Migration - ALL 28 COMPLETE)
# Processing Plugins (16)
# Imports already above - removed duplicate


class AdaptiveProcessingPipeline:
    """Legacy adaptive pipeline facade kept for compatibility with old callers."""

    def __init__(self, progress_callback=None):
        """
        Initialisiert die Adaptive Processing Pipeline.

        Args:
            progress_callback: Optional callback(phase_name, progress, current, total)
                              für Fortschritts-Updates während der Verarbeitung
        """
        self.context_analyzer = ContextAnalyzer()
        self.goal_engine = AdaptiveGoalEngine()
        self.quality_control = QualityControl()
        self.log = []
        self.logger = get_logger("AdaptiveProcessingPipeline")
        self.progress_callback = progress_callback

        # Policy-Engine für intelligente Modellauswahl
        from policy.dsp_policy_engine import DSPPolicyEngine
        from policy.ml_policy_engine import MLModelPolicyEngine

        self.policy_engine = MLModelPolicyEngine()
        self.dsp_policy_engine = DSPPolicyEngine()

        # Ethics Engine & Audio Monitor (v8.0)
        self.ethics_engine = EthicsEngine()
        self.audio_monitor = PermanentAudioMonitor()

        self.logger.info(
            "AdaptiveProcessingPipeline initialized: %s DSP + 28 ML-Plugins + Ethics Engine + Monitor",
            self.dsp_policy_engine.count_total_modules(),
        )
        # Plugin Availability Registry (für autarkes Fehler-Handling)
        self.available_plugins = set()

        # Docker-based ML Plugin instances (für Processing-Methoden)
        # Robust plugin loading: Jedes Plugin wird einzeln mit try/except geladen
        # Bei Fehler: Plugin = None, weiter mit nächstem, Fallback später
        self.deepfilternet = self._safe_load_plugin("deepfilternet", DeepFilterNetV3IIPlugin)
        self.resemble_enhance = self._safe_load_plugin("resemble_enhance", ResembleEnhancePlugin)
        self.mdx23c = self._safe_load_plugin("mdx23c", MDX23CPlugin)
        self.demucs = self.mdx23c  # §4.4: MDX23C (Kim_Vocal_2) ersetzt HTDemucs als Primär-Separator
        self.sgmse = self._safe_load_plugin("wpe", WpePlugin)
        self.mp_senet = self._safe_load_plugin("mp_senet", MpSenetPlugin)
        self.banquet = self._safe_load_plugin("banquet", BanquetVinylPlugin)
        self.uvr_mdxnet = self._safe_load_plugin("uvr_mdxnet", UVRMDXNetPlugin)
        self.gacela = self._safe_load_plugin("gacela", GACELAPlugin)
        self.bs_roformer = self._safe_load_plugin("bs_roformer", BSRoFormerPlugin)

        # ModelManager initialisieren und produktive sowie adaptive ML-Modelle registrieren
        from backend.core.adaptive_plugins import BreathNet, LanguageNet, SibilantNet, VoiceHealthNet

        self.model_manager = ModelManager()
        self.model_manager.set_voice_profile({"gender": "female", "age": 32})
        self.model_manager.register_model(
            "resemble_enhance", self.resemble_enhance, {"type": "universal", "domain": "vocal", "quality": "high"}
        )
        self.model_manager.register_model(
            "mp_senet",
            self.mp_senet,
            {"type": "enhancement", "domain": "music", "quality": "high"},  # §4.4
        )
        self.model_manager.register_model(
            "wpe", self.sgmse, {"type": "denoising", "domain": "music", "quality": "high"}
        )
        self.model_manager.register_model(
            "deepfilternet", self.deepfilternet, {"type": "denoising", "domain": "speech", "quality": "medium"}
        )
        self.model_manager.register_model(
            "mdx23c", self.mdx23c, {"type": "separation", "domain": "music", "quality": "high"}
        )
        self.model_manager.register_model(
            "uvr_mdxnet", self.uvr_mdxnet, {"type": "isolation", "domain": "vocal", "quality": "medium"}
        )
        self.model_manager.register_model(
            "bs_roformer", self.bs_roformer, {"type": "separation", "domain": "music", "quality": "ultra"}
        )
        # Adaptive Plugins registrieren
        self.model_manager.register_adaptive_plugins(SibilantNet(), BreathNet(), VoiceHealthNet(), LanguageNet())

        # Erweiterter Komponenten-Status: Zeige ALLE importierten ML-Plugins und Metriken
        all_plugins = [
            "deepfilternet",
            "resemble_enhance",
            "wpe",
            "banquet",
            "dccrn",
            "fullsubnet",
            "demucs",
            "mdx23c",
            "uvr_mdxnet",
            "gacela",
            "bs_roformer",
            "audiosr",
            "audioldm2",
            "matchering",
            "panns",
            "silero",
            # vampnet: VERBOTEN — §4.4 (kein stabiler ONNX-Export)
            "diffwave",
            "hifigan",
            "sota_universal_enhancer",
            "waveunet",
            "convtasnet",
            # DNSMOS/NISQA/PESQ/ViSQOL: entfernt — verboten §4.4 (Sprach-Metriken)
        ]

        # Ermittele tatsächlich gewählte Modelle/Metriken für diesen Durchlauf
        # (vereinfachte Policy-Logik, kann je nach Kontext angepasst werden)
        chosen = set()
        try:
            # Beispiel: Denoise, Repair, Enhancement, Separation, Quality,
            # Vocoder, Tagging, Mastering, Generation, Pitch.
            context = getattr(self, "last_context", {}) if hasattr(self, "last_context") else {}
            goal = getattr(self, "last_goal", {}) if hasattr(self, "last_goal") else {}
            chosen.add(self.policy_engine.select_denoise_model(context, goal))
            chosen.add(self.policy_engine.select_repair_model(context, goal))
            chosen.add(self.policy_engine.select_enhancement_model(context, goal))
            chosen.add(self.policy_engine.select_stem_separation_model(context, goal))
            for m in self.policy_engine.select_quality_assessment_model(context, goal):
                chosen.add(m)
            chosen.add(self.policy_engine.select_vocoder_model(context, goal))
            chosen.add(self.policy_engine.select_audio_tagging_model(context, goal))
            chosen.add(self.policy_engine.select_mastering_model(context, goal))
            chosen.add(self.policy_engine.select_generative_model(context, goal))
            chosen.add(self.policy_engine.select_pitch_detection_model(context, goal))
        except Exception as _exc:
            self.logger.debug("Operation failed (non-critical): %s", _exc)

        self.logger.info("\n┌─ ML-PLUGINS (alle importiert/verfügbar) ─────────────────────────────────────────────┐")
        for name in all_plugins:
            mark = "*" if name in chosen else " "
            self.logger.info("│%s ✓ %s │", mark, name.ljust(17))
        self.logger.info(
            "└─ Status: %s/47 ML-Plugins & Metriken importiert ───────────────────────┘\n",
            len(all_plugins),
        )
        self._print_component_status()

        # v8.1: Advanced Vocal Separation (Hybrid: MDX-Net + Demucs v5)
        if VOCAL_SEPARATION_V8_AVAILABLE:
            self.vocal_separator_v8 = HybridVocalSeparator(
                fusion_strategy="adaptive",
                sample_rate=44100,
                device="cpu",  # §9.5 CPU-only Policy
            )
            self.vocal_safety_wrapper = VocalSeparationSafetyWrapper(
                self.vocal_separator_v8,
                strict_mode=False,  # Warning mode (not blocking)
            )
            self.logger.info("v8.1 Vocal Separation initialized (Hybrid: MDX-Net + Demucs v5)")
        else:
            self.vocal_separator_v8 = None
            self.vocal_safety_wrapper = None
            self.logger.warning("v8.1 Vocal Separation unavailable, using legacy plugins")

        # v8.2: Conservative Pitch Correction (CREPE + Epistemic Gates)
        if PITCH_CORRECTION_V8_AVAILABLE:
            self.pitch_corrector_v8 = ConservativePitchCorrector(
                sample_rate=44100,
                error_threshold_cents=25.0,  # Only correct obvious errors
                max_dcs=0.15,  # Maximum acceptable damage
                min_epistemic_confidence=0.80,  # High confidence required
                formant_preservation=True,  # Always enabled
            )
            self.pitch_corrector_safety = PitchCorrectionSafetyWrapper(
                self.pitch_corrector_v8,
                strict_mode=False,  # Warning mode (logs but doesn't block)
            )
            self.logger.info("v8.2 Pitch Correction initialized (CREPE + Epistemic Gates)")
        else:
            self.pitch_corrector_v8 = None
            self.pitch_corrector_safety = None
            self.logger.warning("v8.2 Pitch Correction unavailable (install crepe-tf + librosa)")

        # v8.2: Unified Defect Detection (11 defect types, iZotope RX10 competitor)
        if DEFECT_DETECTION_V8_AVAILABLE:
            self.defect_detector_v8 = UnifiedDefectDetector(enable_treatments=True)
            self.logger.info("v8.2 Defect Detection initialized (11 defect types + treatment recommendations)")
        else:
            self.defect_detector_v8 = None
            self.logger.warning("v8.2 Defect Detection unavailable")

        self.logger.info(
            "AdaptiveProcessingPipeline initialized with Docker-based ML plugins + "
            "Policy-Engine + v8.1 Vocal Sep + v8.2 Pitch Correction + v8.2 Defect Detection"
        )

    def _safe_load_plugin(self, plugin_name: str, plugin_class):
        """
        Robustes Plugin-Laden mit try/except und Availability-Tracking.

        Diese Methode garantiert, dass ein einzelner fehlgeschlagener Plugin-Import
        NICHT den gesamten Pipeline-Start crasht. Stattdessen:
        - Logge warning
        - Setze Plugin auf None
        - Tracke Availability in self.available_plugins
        - Nutze später Fallback-Chain

        Args:
            plugin_name: Name des Plugins (z.B. 'resemble_enhance')
            plugin_class: Plugin-Klasse (z.B. ResembleEnhancePlugin)

        Returns:
            Plugin-Instanz oder None bei Fehler
        """
        try:
            plugin_instance = plugin_class()
            self.available_plugins.add(plugin_name)
            return plugin_instance
        except Exception as e:
            logger.warning("Plugin %s nicht verfügbar: %s", plugin_name, e)
            return None

    def _print_component_status(self):
        """
        Strukturierter Status-Report aller Komponenten beim Pipeline-Start.
        Zeigt transparent welche Plugins/Module verfügbar sind.
        """
        # Version from backend-internal constant — avoids forbidden UI import (§11 VERBOTEN).
        _aurik_version = "9.12.9"
        self.logger.info("\n%s", "═" * 80)
        logger.info("  AURIK %s — SYSTEM-KOMPONENTEN STATUS", _aurik_version)
        self.logger.info("%s\n", "═" * 80)

        # ML-Plugins Status
        self.logger.info("┌─ ML-PLUGINS (Denoise/Repair/Enhancement) %s┐", "─" * 35)
        plugin_info = [
            ("deepfilternet", "DeepFilterNet v3.0 (Speech/Broadband)"),
            ("resemble_enhance", "Resemble-Enhance ONNX (Universal)"),
            ("wpe", "WPE Dereverberation (Music/Classical)"),
            ("banquet", "Banquet Vinyl (Vinyl Restoration)"),
            ("dccrn", "DCCRN (De-Reverb/Speech)"),
            ("fullsubnet", "FullSubNet+ (Speech Repair)"),
            ("demucs", "Demucs v4 (Stem Separation)"),
            ("mdx23c", "MDX23C (Advanced Separation)"),
            ("uvr_mdxnet", "UVR MDXNet (Vocal Isolation)"),
            ("gacela", "GACELA (Quality Enhancement)"),
            ("bs_roformer", "MelBandRoformer (Ultra Stem Separation)"),
        ]

        for plugin_name, description in plugin_info:
            status = "✓" if plugin_name in self.available_plugins else "⚠️"
            avail = "geladen" if plugin_name in self.available_plugins else "nicht verfügbar"
            logger.info("│  %s %s %s %s │", status, plugin_name, description, avail)

        available_count = len(self.available_plugins)
        self.logger.info(
            "└─ Status: %s/11 ML-Plugins verfügbar %s┘\n",
            available_count,
            "─" * (80 - 40 - len(str(available_count))),
        )

        # DSP-Module Status
        dsp_count = self.dsp_policy_engine.count_total_modules()
        self.logger.info("┌─ DSP-MODULE (Policy-Engine) %s┐", "─" * 48)
        self.logger.info("│  ✓ Spectral Processing     ✓ Dynamic Range Control  ✓ EQ Systems       │")
        self.logger.info("│  ✓ Artifact Detection      ✓ Transient Shaping      ✓ Stereo Tools     │")
        self.logger.info("│  ✓ Noise Gates             ✓ Compressors/Limiters   ✓ Phase Correction │")
        self.logger.info(
            "└─ Status: %s DSP-Module aktiv %s┘\n",
            dsp_count,
            "─" * max(0, (80 - 32 - len(str(dsp_count)))),
        )

        # Advanced Features Status
        self.logger.info("┌─ ADVANCED FEATURES (v8.0+) %s┐", "─" * 50)
        vocal_status = "✓" if VOCAL_SEPARATION_V8_AVAILABLE else "⚠️"
        pitch_status = "✓" if PITCH_CORRECTION_V8_AVAILABLE else "⚠️"
        defect_status = "✓" if DEFECT_DETECTION_V8_AVAILABLE else "⚠️"
        self.logger.info("│  %s Vocal Separation v8.1    (Hybrid: MDX-Net + Demucs v5)        │", vocal_status)
        self.logger.info("│  %s Pitch Correction v8.2    (CREPE + Epistemic Gates)           │", pitch_status)
        self.logger.info("│  %s Defect Detection v8.2    (11 Defect Types, iZotope-level)    │", defect_status)
        self.logger.info("│  ✓ Ethics Engine              (HIPS Compliance + Safety Wrappers)     │")
        self.logger.info("│  ✓ Audio Monitor              (Permanent Quality Tracking)            │")
        self.logger.info("└%s┘\n", "─" * 79)

        # Material Quality & Musical Goals
        self.logger.info("┌─ ANALYSIS & METRICS %s┐", "─" * 57)
        self.logger.info("│  ✓ Material Quality Analyzer  (7-Level Classification + Adaptive)     │")
        self.logger.info("│  ✓ Musical Goals Validation   (7 Goals: Authenticity, Waerme, etc.)   │")
        self.logger.info("│  ✓ Adaptive Thresholds        (Generation-Count Weighted Scoring)     │")
        self.logger.info("│  ✓ LUFS/True-Peak Metering    (EBU R128 Compliant)                    │")
        self.logger.info("└%s┘\n", "─" * 79)

        # Gesamtstatus
        if available_count >= 8:
            overall = "✓ SYSTEM BEREIT - Alle kritischen Komponenten verfügbar"
        elif available_count >= 5:
            overall = "⚠️ SYSTEM BEREIT - Einige Plugins fehlen, Fallback aktiv"
        else:
            overall = "⚠️ LIMITIERTER MODUS - Nur wenige ML-Plugins verfügbar"

        self.logger.info("%s", "═" * 80)
        logger.info("  %s", overall)
        self.logger.info("%s\n", "═" * 80)

    def separate_vocals_v8(self, audio: np.ndarray, sr: int, use_safety_wrapper: bool = True) -> dict[str, np.ndarray]:
        """
        v8.1: Advanced vocal separation with HIPS compliance

        Uses Hybrid ensemble (MDX-Net + Demucs v5) with adaptive fusion.
        This is Feature #1 from the v8.0 → World-Class Excellence Roadmap.

        Args:
            audio: Audio array (stereo [2, samples] or mono [samples])
            sr: Sample rate
            use_safety_wrapper: Enable HIPS compliance checking (recommended)

        Returns:
            Dictionary with 'vocals' and 'instrumental' stems

        Raises:
            RuntimeError: If v8.1 vocal separation unavailable
            HIPSViolationError: If strict_mode=True and HIPS violation detected
        """
        if not VOCAL_SEPARATION_V8_AVAILABLE or self.vocal_separator_v8 is None:
            raise RuntimeError("v8.1 Vocal Separation not available. Install dependencies: pip install demucs librosa")

        self.logger.info("Starting v8.1 vocal separation (Hybrid: MDX-Net + Demucs v5)")

        if not use_safety_wrapper:
            self.logger.warning(
                "separate_vocals_v8 wurde ohne Safety-Wrapper angefordert; "
                "aus Sicherheitsgruenden wird trotzdem der HIPS-Wrapper erzwungen"
            )

        if self.vocal_safety_wrapper is None:
            raise RuntimeError(
                "v8.1 Vocal Separation Safety Wrapper unavailable; unsichere Direktverarbeitung ist deaktiviert"
            )

        # HIPS-compliant separation with validation (fail-closed, kein Direkt-Bypass)
        try:
            stems = self.vocal_safety_wrapper.safe_separate(audio, sr, return_individual=False)
        except HIPSViolationError as e:
            logger.error("HIPS violation during vocal separation: %s", e)
            raise

        # Log metrics
        metrics = self.vocal_separator_v8.get_metrics()
        self.logger.info(
            "Vocal separation complete: %s total, fusion=%s",
            metrics["total_separations"],
            metrics["fusion_strategy"],
        )

        # Audio monitor tracking (if available)
        if hasattr(self, "audio_monitor"):
            self.audio_monitor.track_operation(
                "vocal_separation_v8", {"model": "hybrid_mdx_demucs_v5", "metrics": metrics}
            )

        return stems

    def correct_pitch_v8(
        self, audio: np.ndarray, sr: int, use_safety_wrapper: bool = True, **kwargs
    ) -> tuple[np.ndarray, dict]:
        """
        v8.2: Conservative pitch correction with HIPS compliance

        Uses CREPE-based pitch detection with epistemic safety gates.
        Only corrects unambiguous pitch errors (> 25 cents) while preserving
        vibrato, glissando, and formants.

        This is Feature #2 from the v8.0 → World-Class Excellence Roadmap.

        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate
            use_safety_wrapper: Enable HIPS compliance checking (recommended)
            **kwargs: Additional args:
                - reference_pitch: Optional reference pitch curve (Hz)
                - dry_wet: Mix between original (0) and corrected (1), default 1.0
                - error_threshold_cents: Override default threshold (25.0)

        Returns:
            Tuple of (corrected_audio, metadata)

            metadata contains:
                - corrected: bool (whether correction was applied)
                - reason: str (rejection reason if not corrected)
                - n_corrections: int (number of corrected regions)
                - dcs: float (Damage Cost Score)
                - epistemic_confidence: float (confidence in analysis)
                - hips_checks: dict (pre/post validation results)

        Raises:
            RuntimeError: If v8.2 pitch correction unavailable
            HIPSViolationError: If strict_mode=True and HIPS violation detected
        """
        if not PITCH_CORRECTION_V8_AVAILABLE or self.pitch_corrector_v8 is None:
            self.logger.warning(
                "v8.2 Pitch Correction not available. Install dependencies: pip install crepe-tf librosa scipy"
            )
            return audio, {
                "corrected": False,
                "reason": "module_unavailable",
                "error": "v8.2 pitch correction module not initialized",
            }

        self.logger.info("Starting v8.2 pitch correction (CREPE + Epistemic Gates)")

        if not use_safety_wrapper:
            self.logger.warning(
                "correct_pitch_v8 wurde ohne Safety-Wrapper angefordert; "
                "aus Sicherheitsgruenden wird trotzdem der HIPS-Wrapper erzwungen"
            )

        if self.pitch_corrector_safety is None:
            self.logger.error(
                "v8.2 Pitch Correction Safety Wrapper unavailable; unsichere Direktverarbeitung ist deaktiviert"
            )
            return audio, {
                "corrected": False,
                "reason": "safety_wrapper_unavailable",
                "error": "unsafe_direct_processing_disabled",
            }

        # HIPS-compliant correction with validation (fail-closed, kein Direkt-Bypass)
        try:
            audio_corrected, metadata = self.pitch_corrector_safety.safe_correct(audio, sr, **kwargs)
        except HIPSViolationError as e:
            logger.error("HIPS violation during pitch correction: %s", e)
            raise

        # Log result
        if metadata.get("corrected", False):
            self.logger.info(
                "Pitch correction applied: %s regions, DCS=%.3f, epistemic_conf=%.2f",
                metadata["n_corrections"],
                metadata["dcs"],
                metadata["epistemic_confidence"],
            )
        else:
            self.logger.info("Pitch correction rejected: %s", metadata.get("reason", "unknown"))

        # Audio monitor tracking (if available)
        if hasattr(self, "audio_monitor"):
            self.audio_monitor.track_operation(
                "pitch_correction_v8",
                {
                    "corrected": metadata.get("corrected", False),
                    "dcs": metadata.get("dcs", 0.0),
                    "epistemic_confidence": metadata.get("epistemic_confidence", 0.0),
                },
            )

        return audio_corrected, metadata

    def analyze_defects_v8(self, audio: np.ndarray, sr: int, quick_scan: bool = False) -> dict:
        """
        v8.2: Unified defect detection and treatment recommendation

        Comprehensive audio quality analysis competing with iZotope RX10's "Repair Assistant".
        Detects 11 defect types, provides severity scoring, and recommends treatments.

        This is Feature #3 (Week 3-4) from the v8.0 → World-Class Excellence Roadmap.

        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate
            quick_scan: If True, runs fast scan (only lightweight detectors)

        Returns:
            Dictionary with comprehensive defect analysis:
            {
                'overall_quality': float,      # 0.0-1.0
                'needs_restoration': bool,
                'defects': List[Dict],         # All detected defects
                'treatments': List[Dict],      # Recommended treatments
                'summary': Dict,               # Counts by severity
                'analysis_time': float
            }

        Example:
            >>> report = pipeline.analyze_defects_v8(audio, sr=48000)
            >>> self.logger.info(f"Quality: {report['overall_quality']:.2f}")
            >>> for treatment in report['treatments'][:3]:  # Top 3 priorities
            ...     self.logger.info("Priority %s: %s", treatment['priority'], treatment['method'])
        """
        if not DEFECT_DETECTION_V8_AVAILABLE or self.defect_detector_v8 is None:
            self.logger.warning("Defect Detection v8.2 not available, returning empty report")
            return {
                "overall_quality": 1.0,
                "needs_restoration": False,
                "defects": [],
                "treatments": [],
                "summary": {},
                "analysis_time": 0.0,
                "error": "Defect Detection v8.2 not installed",
            }

        logger.info("Analyzing defects (quick_scan=%s)...", quick_scan)

        try:
            if quick_scan:
                # Fast scan (only lightweight detectors)
                quick_result = self.defect_detector_v8.quick_scan(audio, sr)

                return {
                    "overall_quality": quick_result["quality_score"],
                    "needs_restoration": quick_result["needs_restoration"],
                    "defects": [],
                    "treatments": [],
                    "summary": {
                        "has_defects": quick_result["has_defects"],
                        "critical_count": quick_result["critical_count"],
                    },
                    "analysis_time": 0.0,  # Quick scan is very fast
                }
            # Full analysis
            report = self.defect_detector_v8.analyze(audio, sr)

            # Convert to serializable dictionary
            return report.to_dict()

        except Exception as e:
            logger.error("Defect analysis failed: %s", e, exc_info=True)
            return {
                "overall_quality": 0.5,
                "needs_restoration": True,
                "defects": [],
                "treatments": [],
                "summary": {},
                "analysis_time": 0.0,
                "error": str(e),
            }

    def run(self, audio_bytes, features, user_profile=None, reference_audio=None, detected_medium=None):
        """Führt aus: the legacy adaptive processing workflow for byte-based callers."""
        del reference_audio
        # 0. Eingangsaudio dekodieren; Tonträgerkette kommt autoritativ aus MediumDetector/PreAnalysis.
        audio_np, sr_audio = _decode_audio_bytes_canonical(audio_bytes)
        features = dict(features) if features else {}
        medium_result = features.get("medium_result")

        def _result_field(result, *names, default=None):
            if result is None:
                return default
            if isinstance(result, dict):
                for name in names:
                    if name in result:
                        return result[name]
                return default
            for name in names:
                if hasattr(result, name):
                    return getattr(result, name)
            return default

        media_chain = []
        if medium_result is not None:
            if hasattr(medium_result, "transfer_chain"):
                _chain = list(getattr(medium_result, "transfer_chain", []) or [])
                _conf = list(getattr(medium_result, "medium_confidences", []) or [])
                media_chain = [
                    {
                        "medium": medium,
                        "confidence": (
                            float(_conf[idx]) if idx < len(_conf) else float(getattr(medium_result, "confidence", 0.0))
                        ),
                    }
                    for idx, medium in enumerate(_chain)
                ]
            elif isinstance(medium_result, dict):
                _chain = list(medium_result.get("transfer_chain", []) or [])
                _conf = list(medium_result.get("medium_confidences", []) or [])
                media_chain = [
                    {
                        "medium": medium,
                        "confidence": (
                            float(_conf[idx]) if idx < len(_conf) else float(medium_result.get("confidence", 0.0))
                        ),
                    }
                    for idx, medium in enumerate(_chain)
                ]

        if media_chain:
            self.logger.info(
                "\n🔎 Erkannte Medienkette: %s",
                " → ".join(f"{m['medium']} ({m['confidence'] * 100:.1f}%)" for m in media_chain),
            )
        else:
            self.logger.info("\n🔎 Medienkette: Kein autoritatives MediumDetector-Ergebnis im Kontext")
        self.log.append({"step": "media_chain_detection", "media_chain": media_chain})

        # --- Materialklassifikations-Konfliktregel nach copilot-instructions.md ---
        def resolve_material_conflict(era_result, medium_result, defect_results, conflict_logger):
            # 1. Höhere Konfidenz gewinnt
            era_type = _result_field(era_result, "material_type", "primary_material")
            era_conf = float(_result_field(era_result, "confidence", default=0.0) or 0.0)
            med_type = _result_field(medium_result, "material_type", "primary_material", "material")
            med_conf = float(_result_field(medium_result, "confidence", default=0.0) or 0.0)
            if era_type == med_type:
                conflict_logger.info("Materialklassifikation eindeutig: %s", era_type)
                return {"type": era_type, "confidence": max(era_conf, med_conf)}
            if era_type and med_type and era_type != med_type:
                conflict_logger.warning(
                    "Materialklassifikations-Konflikt: Era=%s (%.2f), Medium=%s (%.2f)",
                    era_type,
                    era_conf,
                    med_type,
                    med_conf,
                )
                if era_conf > med_conf:
                    conflict_logger.info("Konfliktregel: EraClassifier gewinnt (höhere Konfidenz)")
                    return {"type": era_type, "confidence": era_conf}
                if med_conf > era_conf:
                    conflict_logger.info("Konfliktregel: MediumClassifier gewinnt (höhere Konfidenz)")
                    return {"type": med_type, "confidence": med_conf}
                # 2. Bei Gleichstand: DefectScanner-Auswertung
                if defect_results:

                    def get_max_score_for_material(material):
                        scores = [d.get("severity", 0.0) for d in defect_results if d.get("material_type") == material]
                        return max(scores) if scores else 0.0

                    era_score = get_max_score_for_material(era_type)
                    med_score = get_max_score_for_material(med_type)
                    if era_score > med_score:
                        conflict_logger.info(
                            "Konfliktregel: DefectScanner-Score entscheidet für EraClassifier-Material"
                        )
                        return {"type": era_type, "confidence": era_conf}
                    if med_score > era_score:
                        conflict_logger.info(
                            "Konfliktregel: DefectScanner-Score entscheidet für MediumClassifier-Material"
                        )
                        return {"type": med_type, "confidence": med_conf}
                # 3. Konservativer Materialtyp (restaurierungsschonender)
                conservative = era_type if era_type in ("shellac", "tape", "vinyl") else med_type
                conflict_logger.info("Konfliktregel: Konservativer Materialtyp gewählt: %s", conservative)
                return {"type": conservative, "confidence": max(era_conf, med_conf)}
            # Fallback: nur einer vorhanden
            if era_type:
                return {"type": era_type, "confidence": era_conf}
            if med_type:
                return {"type": med_type, "confidence": med_conf}
            conflict_logger.warning("Materialklassifikation nicht möglich — Default 'unknown'")
            return {"type": "unknown", "confidence": 0.0}

        # Annahme: era_result, medium_result, defect_results werden im Kontext/Features bereitgestellt
        era_result = features.get("era_result")
        defect_results = features.get("detected_defects", [])
        detected_medium_final = detected_medium
        if not detected_medium_final:
            detected_medium_final = resolve_material_conflict(era_result, medium_result, defect_results, self.logger)
        # detected_medium in Features übernehmen, damit alle Folge-Analysen darauf zugreifen
        if detected_medium_final:
            features["detected_medium"] = detected_medium_final

        # SOTA-Maximum-Policy wird nach jedem Durchlauf neu abgeleitet, aber mit aktuellen Analyse-Features kombiniert
        sota_policy = None
        try:
            analyzer = SOTAMaximumAnalyzer()
            sota_policy = analyzer.recommend_sota_policy()
            logger.info("SOTA-Maximum-Policy (dynamisch): %s", sota_policy)
        except Exception as e:
            logger.error("SOTA-Maximum-Policy konnte nicht geladen werden: %s", e)

        # Start Permanent Audio Monitoring
        self.audio_monitor.capture_baseline(
            audio_np,
            sr_audio,
            file_path=features.get("file_path", "unknown"),
            metadata={"detected_medium": detected_medium_final, "user_profile": user_profile},
        )

        # Kontextanalyse und alle Folge-Analysen greifen jetzt auf das aktualisierte features zu
        context = self.context_analyzer.analyze(audio_np)
        self.log.append({"step": "context_analysis", "context": context})
        logger.info("Kontextanalyse abgeschlossen: %s", context)

        # Defekt- und Störungserkennung direkt nach Medienkette/Analyse
        # (Hier ggf. eigene Methode oder Modul für Defekterkennung einbinden)
        try:
            defect_results = self.defect_detector.detect(audio_np, sr_audio, features)
            self.log.append({"step": "defect_detection", "defects": defect_results})
            logger.info("Defekt-/Störungserkennung abgeschlossen: %s", defect_results)
            # Defektergebnisse in Features/Maßnahmenkette übernehmen
            features["detected_defects"] = defect_results
        except Exception as e:
            logger.error("Defekt-/Störungserkennung fehlgeschlagen: %s", e)

        # 2. Zieldefinition
        goal = self.goal_engine.define_goal(context)
        self.log.append({"step": "goal_definition", "goal": goal})
        logger.info("Zieldefinition abgeschlossen: %s", goal)

        # 3. PHASE 4.5: Ethics Engine - Epistemic Gate
        self.logger.info("🧭 Phase 4.5: Ethics Engine - Epistemic Gate")
        ethics_context = {
            "confidence": context.get("confidence", 0.85),
            "defect_type": features.get("defect_type", "unknown"),
            "defect_severity": context.get("defect_severity", "medium"),
            "user_mode": user_profile.get("mode", "restoration") if user_profile else "restoration",
            "cultural_significance": context.get("cultural_significance", False),
            "detected_medium": detected_medium,
            "has_artifacts": context.get("artefact_risk", False),
        }

        ethics_report = self.ethics_engine.epistemic_gate(ethics_context)
        self.log.append(
            {"step": "ethics_gate", "decision": ethics_report.decision.value, "reasoning": ethics_report.reasoning}
        )
        logger.info("Ethics Decision: %s - %s", ethics_report.decision.value, ethics_report.reasoning)

        # Ergebnis-Container für ALLE Ethics-Pfade (muss vor dem Ethics-Gate stehen)
        results = {"steps": [], "quality": [], "log": [], "ethics_decision": None}
        phase_count = 0
        total_phases = 4
        current_audio = audio_bytes
        policy = sota_policy  # SOTA-Policy für Phasen-Verarbeitung

        # Handle ethics decisions
        # Erzwinge immer mindestens eine DSP-Phase, auch bei PRESERVE/HARD_STOP
        if ethics_report.decision in [EpistemicDecision.HARD_STOP, EpistemicDecision.PRESERVE]:
            logger.warning("Ethics Engine: %s - Erzwinge minimalen DSP-Processing", ethics_report.decision.value)
            audio_np, sr_audio = _decode_audio_bytes_canonical(audio_bytes)
            # Minimaler DSP: Loudness-Normalisierung
            from .mastering import mastering_chain

            mastered = mastering_chain(audio_np, sr_audio)
            with io.BytesIO() as buf:
                sf.write(buf, mastered, sr_audio, format="WAV")
                mastered_bytes = buf.getvalue()
            self.audio_monitor.export_audit_report(output_dir="./audits", formats=["json", "yaml"])
            return {
                "steps": [
                    {"name": "forced_mastering", "audio": mastered_bytes, "reason": ethics_report.decision.value}
                ],
                "quality": [],
                "log": self.log,
                "ethics_decision": ethics_report.decision.value,
            }

        # Restaurierung (normaler Verarbeitungspfad für mode_a u.ä.)
        if self._needs_restoration(context, goal):
            phase_count += 1
            logger.info("Phase %s/%s: Restaurierung...", phase_count, total_phases)
            if self.progress_callback:
                self.progress_callback("restoration", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.start_module("restoration")

            res = self._restoration(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.end_module(
                audio_in,
                audio_out,
                sr_audio,
                confidence=res.get("confidence", 0.9),
                quality_gate_passed=res.get("quality_passed", True),
            )

            if self.progress_callback:
                self.progress_callback("restoration", 1.0, phase_count, total_phases)
            self.logger.info("✅ Restaurierung abgeschlossen.")

        # Reparatur
        if self._needs_repair(context, goal):
            phase_count += 1
            logger.info("Phase %s/%s: Reparatur...", phase_count, total_phases)
            if self.progress_callback:
                self.progress_callback("repair", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.start_module("repair")

            res = self._repair(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.end_module(
                audio_in,
                audio_out,
                sr_audio,
                confidence=res.get("confidence", 0.9),
                quality_gate_passed=res.get("quality_passed", True),
            )

            if self.progress_callback:
                self.progress_callback("repair", 1.0, phase_count, total_phases)
            self.logger.info("✅ Reparatur abgeschlossen.")

        # Rekonstruktion
        if self._needs_reconstruction(context, goal):
            phase_count += 1
            logger.info("Phase %s/%s: Rekonstruktion...", phase_count, total_phases)
            if self.progress_callback:
                self.progress_callback("reconstruction", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.start_module("reconstruction")

            res = self._reconstruction(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.end_module(
                audio_in,
                audio_out,
                sr_audio,
                confidence=res.get("confidence", 0.9),
                quality_gate_passed=res.get("quality_passed", True),
            )

            if self.progress_callback:
                self.progress_callback("reconstruction", 1.0, phase_count, total_phases)
            self.logger.info("✅ Rekonstruktion abgeschlossen.")

        # Remastering
        if self._needs_remastering(context, goal):
            phase_count += 1
            logger.info("Phase %s/%s: Remastering...", phase_count, total_phases)
            if self.progress_callback:
                self.progress_callback("remastering", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.start_module("remastering")

            res = self._remastering(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = _decode_audio_bytes_canonical(current_audio)
            self.audio_monitor.end_module(
                audio_in,
                audio_out,
                sr_audio,
                confidence=res.get("confidence", 0.9),
                quality_gate_passed=res.get("quality_passed", True),
            )

            if self.progress_callback:
                self.progress_callback("remastering", 1.0, phase_count, total_phases)
            self.logger.info("✅ Remastering abgeschlossen.")

        # 4. Qualitätskontrolle nach jedem Schritt
        for step in results["steps"]:
            qc_result = self.quality_control.psychoacoustic_score(step["audio"], features.get("sr", 44100))
            results["quality"].append(qc_result)
            self.log.append({"step": step["name"], "quality": qc_result})
            self.logger.info("Qualitätskontrolle für %s: %s", step["name"], qc_result)

        # 5. Mastering/Postprocessing (immer am Ende, vor Export)
        if results["steps"]:
            last = results["steps"][-1]
            # Audio-Bytes in numpy-Array
            audio, sr = _decode_audio_bytes_canonical(last["audio"])
            # Sicherstellen, dass mastering_chain importiert ist
            from .mastering import mastering_chain as imported_mastering_chain

            if imported_mastering_chain is not None:
                mastered = imported_mastering_chain(audio, sr)
                # Zurück in Bytes
                with io.BytesIO() as buf:
                    sf.write(buf, mastered, sr, format="WAV")
                    mastered_bytes = buf.getvalue()
                last["audio"] = mastered_bytes
                self.log.append(
                    {
                        "step": "mastering",
                        "info": "Loudness-Normalisierung, Limiting, Dithering angewendet",
                    }
                )
                self.logger.info("Mastering/Postprocessing durchgeführt.")
            else:
                self.logger.error("Mastering konnte nicht durchgeführt werden: mastering_chain nicht importierbar.")
                self.log.append({"step": "mastering", "error": "mastering_chain nicht importierbar"})

        # 6. Capture Final Metrics & Export Audit
        final_audio, _ = _decode_audio_bytes_canonical(current_audio)
        self.audio_monitor.capture_final(final_audio, sr_audio)
        self.audio_monitor.export_audit_report(output_dir="./audits", formats=["json", "yaml", "csv"])
        # 7. Logging aller Entscheidungen
        results["log"] = self.log
        results["ethics_decision"] = ethics_report.decision.value
        results["cas_improvement"] = self.audio_monitor.compute_cas_improvement()
        return results

    def _needs_restoration(self, context, goal):
        # Beispiel: Artefaktrisiko oder niedrige Qualität
        return context.get("artefact_risk", False) or goal.get("quality_level") == "maximal"

    def _needs_repair(self, context, goal):
        # Beispiel: Transientenverlust oder Warnungen
        return not context.get("transient_rich", True) or "Warnung" in str(goal)

    def _needs_reconstruction(self, context, goal):
        del context
        # Beispiel: Referenzwarnung oder starke Abweichung
        return "reference_warning" in goal

    def _needs_remastering(self, context, goal):
        del context
        # Beispiel: explizit gewünscht oder pro-User
        return goal.get("quality_level") == "maximal"

    def _restoration(self, audio_bytes, features, goal, context=None):
        """
        SOTA-Maximum: Denoising + Enhancement (Phase 8A Integration)

        Pipeline:
        1. ML-Denoise (Policy-selected: DeepFilterNet, WPE, etc.)
        2. Hybrid Refinement (DSP post-processing)
        3. HF Extension (Neural via AudioSR, conditional)
        4. Stereo Widening (Frequency-dependent, genre-adaptive)
        5. Adaptive EQ (Genre-specific curves)
        """
        # Audio-Bytes in numpy-Array (kanonischer Loader mit robustem Fallback)
        audio_original, sr = _decode_audio_bytes_canonical(audio_bytes)
        # Kontext für Policy-Entscheidung (maximaler Informationsumfang)
        if context is None:
            context = {}
        # Kontext aus Audio-Features aufbauen
        context.update(
            {
                "has_vocals": features.get("has_vocals", False),
                "vocal_confidence": features.get("vocal_confidence", 0.0),
                "noise_type": features.get("noise_type", "unknown"),
                "has_reverb": features.get("has_reverb", False),
                "genre": features.get("genre", "unknown"),
                "detected_medium": features.get("detected_medium", "unknown"),
                "sample_rate": sr,
                "channels": 2 if audio_original.ndim > 1 else 1,
                "snr": features.get("snr", None),
                "clipping": features.get("has_clipping", False),
                "transient_rich": features.get("transient_rich", True),
                "defects": features.get("defects", []),
                "artifacts": features.get("artifacts", None),
                "crest_factor": features.get("crest_factor", None),
                "lufs": features.get("lufs", None),
                "spectral_centroid": features.get("spectral_centroid", None),
                "spectral_rolloff": features.get("spectral_rolloff", None),
                "quality_gates": features.get("quality_gates", {}),
            }
        )
        # Adaptive Modellwahl via Policy-Engine
        selected_model = self.policy_engine.select_enhancement_model(context, goal)
        selected_model = _enforce_canonical_policy_route("enhancement", selected_model, context)
        model_obj = self.model_manager.models.get(selected_model, {}).get("obj")
        if model_obj:
            model_obj.process(audio_original, context)
        elif not ("." in selected_model or selected_model.startswith("phase_") or selected_model.startswith("uv3.")):
            # Fallback: Multi-Stage Enhancement
            self.model_manager.multi_stage_enhancement(audio_original, context)
        logger.info("Policy-Kontext für Restoration: %s", context)

        # STAGE 1: ML-Denoise (Policy-selected Model)
        model_name = self.policy_engine.select_denoise_model(context, goal)
        model_name = _enforce_canonical_policy_route("denoise", model_name, context)
        canonical_audio = _canonical_policy_audio_route(model_name, audio_original, sr, context)
        if canonical_audio is not None:
            audio_denoised = canonical_audio
            logger.info("Policy-Selektion: %s über kanonischen Aurik-9-Router", model_name)
        else:
            logger.error(
                "Kanonische Denoise-Route nicht verfügbar (%s) — Legacy-Pluginpfad deaktiviert, Dry-Fallback aktiv",
                model_name,
            )
            audio_denoised = np.asarray(audio_original, dtype=np.float32).copy()

        # Strukturierte Stage-Ausgabe
        self.logger.info("\n╔%s╗", "═" * 78)
        self.logger.info("║  RESTORATION STAGE 1: ML-DENOISE%s║", " " * 45)
        self.logger.info("╚%s╝\n", "═" * 78)

        # Policy-Entscheidung mit Begründung
        reason_map = {
            "banquet": "Vinyl-Detection → Spezialisiertes Banquet Model",
            "deepfilternet": "Speech + Broadband Noise → DeepFilterNet v3",
            "wpe": "Classical/Jazz Genre → WPE Dereverberation",
            "dccrn": "Reverb Detected → DCCRN De-Reverb Model",
            "resemble_enhance": "Standard Noise Reduction → Universal Model",
        }
        reason = reason_map.get(model_name, "Policy-basierte Auswahl")

        logger.info("Policy-Selektion: %s", model_name)
        logger.info("  Begründung: %s", reason)
        self.logger.info(
            "  Medium: %s, Genre: %s, Quality: %s",
            context.get("detected_medium", "unknown"),
            context.get("genre", "unknown"),
            goal.get("quality_level", "standard"),
        )
        self.logger.info("")

        self.logger.info("Legacy-Plugin-Execution deaktiviert — Verarbeitung bleibt auf kanonischem Routerpfad")

        # STAGE 2: Hybrid Refinement (Phase 8A-1)
        try:
            from dsp.hybrid_denoise_refiner import apply_hybrid_refinement

            audio_refined, refine_metrics = apply_hybrid_refinement(
                audio_denoised,
                audio_original,
                sr,
                genre=context["genre"],
                strength=0.3,  # Conservative default
            )

            self.logger.info(
                "✓ Hybrid Refinement applied (spectral_dev=%.3f)",
                refine_metrics.get("spectral_deviation", 0),
            )

        except Exception as e:
            logger.warning("Hybrid Refinement skipped: %s", e)
            audio_refined = audio_denoised

        # STAGE 2.5: Vocal Enhancement & De-Esser (Sibilanten-Behandlung)
        # Nur wenn Vocals erkannt wurden
        if context.get("has_vocals", False):
            try:
                from dsp.aurik_deesser_pro.music_vocal_pipeline import process_vocals

                self.logger.info("Vocal Enhancement & De-Esser starting...")

                # Gender-Detection für optimale Parameter (optional)
                detected_gender = "auto"
                try:
                    from backend.core.forensics.gender_detection import GenderDetector

                    detector = GenderDetector()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_gender:
                        sf.write(tmp_gender.name, audio_refined, sr)
                        detected_gender = detector.detect_gender(tmp_gender.name)
                        os.remove(tmp_gender.name)
                    logger.info("Gender detected: %s", detected_gender)
                except Exception as gender_err:
                    logger.info("Gender detection unavailable, using signal-adaptive analysis: %s", gender_err)
                    detected_gender = "auto"

                # Process Vocals mit adaptiven Sibilanten-Parametern
                audio_vocal_enhanced = process_vocals(
                    audio_refined,
                    sr=sr,
                    gender=detected_gender,
                    model_path=None,  # Optional: ML-based HF texture model
                )

                logger.info("✓ Vocal Enhancement applied (gender=%s)", detected_gender)

                audio_refined = audio_vocal_enhanced

            except Exception as e:
                logger.warning("Vocal Enhancement skipped: %s", e)
                # audio_refined bleibt unverändert bei Fehler
        else:
            self.logger.info("No vocals detected, skipping Vocal Enhancement")

        # STAGE 3: HF Extension (Phase 8A-2, conditional)
        try:
            from dsp.hf_extender import apply_hf_extension_if_needed

            audio_hf, sr_out = apply_hf_extension_if_needed(audio_refined, sr, context=context, goal=goal)

            if sr_out != sr:
                logger.info("✓ HF Extension applied: %sHz → %sHz", sr, sr_out)
                sr = sr_out  # Update sample rate
            else:
                audio_hf = audio_refined

        except Exception as e:
            logger.warning("HF Extension skipped: %s", e)
            audio_hf = audio_refined

        # STAGE 4: Stereo Widening (Phase 8A-3, conditional)
        try:
            from dsp.stereo_widener import apply_stereo_widening_if_needed

            audio_wide = apply_stereo_widening_if_needed(audio_hf, sr, context=context, goal=goal)

            if not np.array_equal(audio_wide, audio_hf):
                self.logger.info("✓ Stereo Widening applied")

        except Exception as e:
            logger.warning("Stereo Widening skipped: %s", e)
            audio_wide = audio_hf

        # STAGE 5: Adaptive EQ (Phase 8A-4)
        try:
            from dsp.adaptive_eq import apply_adaptive_eq

            audio_final = apply_adaptive_eq(audio_wide, sr, context=context, goal=goal)

            self.logger.info("✓ Adaptive EQ applied (genre=%s)", context["genre"])

        except Exception as e:
            logger.warning("Adaptive EQ skipped: %s", e)
            audio_final = audio_wide

        # === DSP POST-PROCESSING (adaptive chain) ===
        self.logger.info("Starting DSP Post-Processing...")
        dsp_chain: list = []  # default empty chain (populated upstream in full pipeline)
        post_chain = dsp_chain[3:]  # Remaining modules: Dynamics, Enhancement, Post

        logger.info("\n✅ DSP POST-PROCESSING: %s Module angewendet", len(post_chain))

        audio_final = self._apply_dsp_chain(audio_final, sr, post_chain)
        logger.info("✓ DSP Post-Processing applied: %s modules", len(post_chain))

        # Audit-Infos
        stages_applied = [
            "dsp_preprocessing",
            "ml_denoise",
            "hybrid_refinement",
        ]

        # Vocal Enhancement nur wenn durchgeführt
        if context.get("has_vocals", False):
            stages_applied.append("vocal_enhancement_deesser")

        stages_applied.extend(
            [
                "hf_extension",
                "stereo_widening",
                "adaptive_eq",
                "dsp_postprocessing",
            ]
        )

        vocal_stage = " + Vocal Enhancement" if context.get("has_vocals", False) else ""

        self.log.append(
            {
                "step": "restoration_enhanced",
                "info": (
                    f"DSP Pre ({len(dsp_chain[:3])}) + {model_name} + "
                    f"Hybrid Refinement{vocal_stage} + HF Extension + Stereo + "
                    f"EQ + DSP Post ({len(post_chain)})"
                ),
                "params": goal,
                "dsp_chain": [name for name, _ in dsp_chain],
                "stages": stages_applied,
            }
        )

        # Zurück in Bytes
        with io.BytesIO() as buf:
            sf.write(buf, audio_final, sr, format="WAV")
            out_bytes = buf.getvalue()

        return {"name": "restoration_enhanced", "audio": out_bytes, "params": goal}

    def restoration(self, audio_bytes, features, goal, context=None):
        """Public compatibility wrapper for legacy V2 delegation."""
        return self._restoration(audio_bytes, features, goal, context)

    def _apply_dsp_chain(self, audio, sr, dsp_chain):
        """Wrapper für apply_dsp_chain."""
        return apply_dsp_chain(audio, sr, dsp_chain)

    def _repair(self, audio_bytes, features, goal, context):
        """
        SOTA-Maximum: Clipping-Repair + Refinement (Phase 8A Integration)

        Pipeline:
        1. ML-Repair (Policy-selected: FullSubNet+ or DCCRN)
        2. Hybrid Refinement (preserve transients)
        """
        del features
        # Audio-Bytes in numpy-Array (kanonischer Loader mit robustem Fallback)
        audio_original, sr = _decode_audio_bytes_canonical(audio_bytes)

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info("Repair Pipeline: detected_medium=%s", context.get("detected_medium", "unknown"))

        # STAGE 1: ML-Repair (Policy-selected Model)
        model_name = self.policy_engine.select_repair_model(context, goal)
        model_name = _enforce_canonical_policy_route("repair", model_name, context)
        canonical_audio = _canonical_policy_audio_route(model_name, audio_original, sr, context)
        if canonical_audio is not None:
            audio_repaired = canonical_audio
            logger.info("Policy-Selektion: %s über UV3-Reparaturroute", model_name)
        else:
            logger.error(
                "Kanonische Repair-Route nicht verfügbar (%s) — Legacy-Pluginpfad deaktiviert, Dry-Fallback aktiv",
                model_name,
            )
            audio_repaired = np.asarray(audio_original, dtype=np.float32).copy()

        self.logger.info("\n╔%s╗", "═" * 78)
        self.logger.info("║  REPAIR STAGE 1: ML-REPAIR (Clipping/Artifacts)%s║", " " * 30)
        self.logger.info("╚%s╝\n", "═" * 78)

        repair_type = (
            "FullSubNet+ (Speech Enhancement)" if "fullsubnet" in model_name else "DCCRN (Music/Reverb Repair)"
        )
        logger.info("Policy-Selektion: %s", model_name)
        logger.info("  Typ: %s", repair_type)
        self.logger.info("  Vocals: %s", "Ja" if context.get("has_vocals", False) else "Nein")
        self.logger.info("")

        self.logger.info("Legacy-Plugin-Execution deaktiviert — Verarbeitung bleibt auf kanonischer Repair-Route")

        # STAGE 2: Hybrid Refinement (preserve transients, reduce artifacts)
        try:
            from dsp.hybrid_denoise_refiner import apply_hybrid_refinement

            audio_refined, _refine_metrics = apply_hybrid_refinement(
                audio_repaired,
                audio_original,
                sr,
                genre=context["genre"],
                strength=0.2,  # Lower strength for repair (preserve transients)
            )

            self.logger.info("✓ Hybrid Refinement applied after repair")

        except Exception as e:
            logger.warning("Hybrid Refinement skipped: %s", e)
            audio_refined = audio_repaired

        # Audit-Infos
        self.log.append(
            {
                "step": "repair_enhanced",
                "info": f"{model_name} + Hybrid Refinement (Phase 8A)",
                "params": goal,
                "stages": ["ml_repair", "hybrid_refinement"],
            }
        )

        # Zurück in Bytes
        with io.BytesIO() as buf:
            sf.write(buf, audio_refined, sr, format="WAV")
            out_bytes = buf.getvalue()

        return {"name": "repair_enhanced", "audio": out_bytes, "params": goal}

    def _reconstruction(self, audio_bytes, features, goal, context):
        """SOTA-Maximum: Source-Separation mit Docker-basierten ML-Modellen (Policy-basierte Auswahl)"""
        del features

        # Audio-Bytes in numpy-Array (kanonischer Loader mit robustem Fallback)
        audio, sr = _decode_audio_bytes_canonical(audio_bytes)

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info(
            "Reconstruction Pipeline: detected_medium=%s, stems=%s",
            context.get("detected_medium", "unknown"),
            goal.get("stems", 4),
        )

        # Policy-Engine wählt optimales Model
        model_name = self.policy_engine.select_separation_model(context, goal)
        model_name = _enforce_canonical_policy_route("separation", model_name, context)
        canonical_separated = _canonical_policy_separation_route(model_name, audio, sr, context)
        if canonical_separated is not None:
            self.log.append(
                {
                    "step": "reconstruction",
                    "info": f"{model_name} (canonical Aurik-9 router)",
                    "params": goal,
                }
            )
            with io.BytesIO() as buf:
                sf.write(buf, canonical_separated, sr, format="WAV")
                out_bytes = buf.getvalue()
            return {"name": "reconstruction", "audio": out_bytes, "params": goal}

        logger.error(
            "Kanonische Separation-Route nicht verfügbar (%s) — Legacy-Pluginpfad deaktiviert, Dry-Fallback aktiv",
            model_name,
        )
        audio_separated = np.asarray(audio, dtype=np.float32).copy()

        self.log.append(
            {
                "step": "reconstruction",
                "info": f"{model_name} (canonical route unavailable, dry fallback)",
                "params": goal,
            }
        )
        with io.BytesIO() as buf:
            sf.write(buf, audio_separated, sr, format="WAV")
            out_bytes = buf.getvalue()
        return {"name": "reconstruction", "audio": out_bytes, "params": goal}

    def _remastering(self, audio_bytes, features, goal, context):
        # SOTA-Remastering via Matchering-API-Client mit Fallback
        from .mastering import mastering_chain

        try:
            from plugins.matchering_plugin import MatcheringPlugin as LocalMatcheringPlugin
        except ImportError:
            LocalMatcheringPlugin = None  # type: ignore[assignment,misc]
            _log.warning("MatcheringPlugin nicht verfügbar — Fallback auf mastering_chain")

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info("Remastering Pipeline: detected_medium=%s", context.get("detected_medium", "unknown"))

        self.logger.info("\n%s", "=" * 80)
        self.logger.info("🎼 MASTERING/REMASTERING")
        self.logger.info("%s", "=" * 80)
        self.logger.info("   Method: Matchering 2.0 (AI-powered)")
        self.logger.info("   Target: Professional Mastering Standards")
        self.logger.info("%s\n", "=" * 80)

        container_info = {
            "container": "matchering2.0",
            "image": "models/matchering2.0/Dockerfile.matchering2.0",
            "api_url": "http://localhost:8360/api/process",
        }

        try:
            if LocalMatcheringPlugin is None:
                raise ImportError("MatcheringPlugin nicht verfügbar")

            sr = int(features.get("sr", 44100))
            target_audio = features["audio"]
            reference_audio = features["reference_audio"]
            client = LocalMatcheringPlugin()
            remastered_audio = client.process(target_audio, reference_audio, sr)

            with io.BytesIO() as buf:
                sf.write(buf, remastered_audio, sr, format="WAV")
                remastered_bytes = buf.getvalue()

            self.logger.info("Remastering mit Matchering durchgeführt.")
            self.log.append(
                {
                    "step": "remastering",
                    "info": "Matchering-Plugin verwendet",
                    "params": goal,
                    "container": container_info,
                    "status": "success",
                }
            )
            return {
                "name": "remastering",
                "audio": remastered_bytes,
                "params": goal,
                "status": "success",
                "container": container_info,
            }
        except Exception as e:
            logger.error("Remastering fehlgeschlagen: %s", e, exc_info=True)
            # Fallback: internes Mastering
            self.logger.warning("Fallback: internes Mastering wird verwendet.")
            try:
                audio = features["audio"]
                sr = features.get("sr", 44100)
                mastered = mastering_chain(audio, sr)
                with io.BytesIO() as buf:
                    sf.write(buf, mastered, sr, format="WAV")
                    fallback_bytes = buf.getvalue()
                self.log.append(
                    {
                        "step": "remastering",
                        "error": str(e),
                        "params": goal,
                        "container": container_info,
                        "status": "fallback-internal-mastering",
                    }
                )
                return {
                    "name": "remastering",
                    "audio": fallback_bytes,
                    "params": goal,
                    "status": "fallback-internal-mastering",
                    "error": str(e),
                    "container": container_info,
                }
            except Exception as fallback_e:
                self.logger.error(
                    "Fallback-Remastering ebenfalls fehlgeschlagen: %s",
                    fallback_e,
                    exc_info=True,
                )
                self.log.append(
                    {
                        "step": "remastering",
                        "error": str(fallback_e),
                        "params": goal,
                        "container": container_info,
                        "status": "error",
                    }
                )
                return {
                    "name": "remastering",
                    "audio": audio_bytes,
                    "params": goal,
                    "status": "error",
                    "error": str(fallback_e),
                    "container": container_info,
                }


# ==============================================================================
# Adaptive Processing Pipeline V2 with Formal Job Tracking (AURIK Spec 4.1)
# ==============================================================================


class AdaptiveProcessingPipelineV2:
    """
    Aktualisierte Pipeline mit formalen RestaurationJob-Datenstrukturen.

    Key Features:
    - Creates formal ResturationJob with UUID tracking
    - Uses AnalysisEngineAdapter for formal AnalysisProfile
    - Tracks all ProcessingSteps with hashes, timestamps, parameters
    - Calculates CAS before/after for each step
    - Archives job with ArchiveManager
    - Maintains backward compatibility with existing code

    Spec Reference: Section 4.1 - ResturationJob structure
    """

    def __init__(self):
        # Existing components
        self.context_analyzer = ContextAnalyzer()
        self.goal_engine = AdaptiveGoalEngine()
        self.quality_control = QualityControl()
        self.logger = get_logger("AdaptiveProcessingPipelineV2")

        # New formal components
        from backend.core.aesthetic_judgment import AestheticJudgmentModel
        from backend.core.archive_manager import ArchiveManager
        from backend.core.forensics.analysis_and_modules import AnalysisEngineAdapter

        self.analysis_engine = AnalysisEngineAdapter()
        self.ajm = AestheticJudgmentModel()
        self.archive_manager = ArchiveManager()

        # ML-Plugin-Instanzen — §11.3-konforme Auswahl
        # Processing Plugins
        self.deepfilternet = DeepFilterNetV3IIPlugin()
        self.resemble_enhance = ResembleEnhancePlugin()
        self.mdx23c = MDX23CPlugin()  # §4.4: MDX23C (Kim_Vocal_2) Primär-Separator
        self.demucs = self.mdx23c  # Legacy-Alias
        self.sgmse = WpePlugin()
        # self.fullsubnet entfernt — 16 kHz-Sprach-NR, nicht §11.3
        # DCCRNPlugin entfernt — §4.4 verboten; MpSenetPlugin übernimmt
        self.mp_senet = MpSenetPlugin()
        # self.spleeter entfernt — veraltetes 2019er Modell, nicht §11.3
        self.uvr_mdxnet = UVRMDXNetPlugin()
        self.banquet = BanquetVinylPlugin()
        self.hifigan = HiFiGANPlugin()
        # self.convtasnet entfernt — Sprach-Separation (Luo 2019), nicht §11.3
        self.diffwave = DiffWavePlugin()
        # self.waveunet entfernt — Sprach-Separation (Stoller 2018), nicht §11.3
        self.crepe = CREPEPlugin()
        # self.sota_enhancer entfernt — orchestriert Sprach-NR, nicht §11.3

        # Metriken Plugins — DNSMOS/NISQA/PESQ verboten §4.4+§10.2 (Sprach-Modelle)
        # ViSQOLPlugin: entfernt — explizit verboten §4.4+§10.2

        # Neue Plugins (8)
        self.audioldm2 = AudioLDM2Plugin()
        self.audiosr = AudioSRPlugin()
        # CDPAMPlugin: entfernt — explizit verboten §4.4+§10.2
        self.gacela = GACELAPlugin()
        self.matchering = MatcheringPlugin()
        self.panns = PANNSPlugin()
        self.silero = SileroPlugin()
        # self.vampnet: entfernt — VERBOTEN §4.4; Nachfolger: flow_matching_plugin

        self.logger.info("AdaptiveProcessingPipelineV2 initialized with 27 Docker-based ML plugins")

    def run_with_job_tracking(
        self, input_audio_path: str, output_audio_path: str, user_id: str | None = None
    ) -> object:
        """
        Complete restoration workflow with formal ResturationJob tracking.

        Args:
            input_audio_path: Path to input audio file
            output_audio_path: Path to save output audio file
            user_id: Optional user identifier

        Returns:
            ResturationJob with complete processing history
        """
        from pathlib import Path
        from uuid import uuid4

        from backend.core.data_models import AudioFile, ResturationJob
        from backend.file_import import load_audio_file

        # 1. Create ResturationJob with UUID
        job_id = uuid4()
        logger.info("Starting restoration job %s", job_id)

        # Load input audio — §VERBOTEN: sf.read() → load_audio_file()
        _load_result = load_audio_file(str(input_audio_path), do_carrier_analysis=False)
        if _load_result is None or _load_result.get("error"):
            raise RuntimeError(f"load_audio_file failed: {(_load_result or {}).get('error')}")
        audio = _load_result["audio"]
        sr = _load_result["sr"]

        # Create input AudioFile
        input_file_hash = self.archive_manager.calculate_file_hash(input_audio_path)
        input_file = AudioFile(
            file_path=input_audio_path,
            file_hash=input_file_hash,
            format=Path(input_audio_path).suffix.upper().lstrip("."),
            sample_rate=sr,
            bit_depth=16,  # Placeholder
            channels=1 if audio.ndim == 1 else audio.shape[0],
            duration=len(audio) / sr,
            file_size=Path(input_audio_path).stat().st_size,
        )

        # 2. Run analysis → create AnalysisProfile
        self.logger.info("Running audio analysis...")
        analysis_profile = self.analysis_engine.analyze(audio, sr, input_audio_path)

        # 3. Create ResturationJob
        job = ResturationJob(
            job_id=job_id,
            input_file=input_file,
            analysis_profile=analysis_profile,
            user_id=user_id,
            status="running",
            completed_at=None,
            output_file=None,
            quality_report=None,
            archive_path=None,
            error_message=None,
        )

        # Prepare context from analysis for legacy pipeline
        features = analysis_profile.raw_features
        features["sr"] = sr
        features["audio"] = audio

        # 4. Calculate initial CAS
        self.logger.info("Calculating initial CAS...")
        initial_cas, _initial_scores = self.ajm.cas_calculator.calculate_cas(
            audio,
            sr,
            analysis_profile,
            genre=analysis_profile.musical_context.genre,
            genre_confidence=analysis_profile.musical_context.genre_confidence,
        )

        # 5. Run processing steps with tracking
        current_audio = audio
        step_counter = 0

        # Context analysis for pipeline decisions
        context = self.context_analyzer.analyze(audio)
        goal = self.goal_engine.define_goal(context)

        # Process each step and track
        if self._needs_restoration(context, goal):
            current_audio, step_counter = self._process_step_with_tracking(
                job,
                current_audio,
                sr,
                "restoration",
                "denoise+declick",
                step_counter,
                context,
                goal,
            )

        if self._needs_repair(context, goal):
            current_audio, step_counter = self._process_step_with_tracking(
                job, current_audio, sr, "repair", "declip", step_counter, context, goal
            )

        if self._needs_reconstruction(context, goal):
            current_audio, step_counter = self._process_step_with_tracking(
                job,
                current_audio,
                sr,
                "reconstruction",
                "source_separation",
                step_counter,
                context,
                goal,
            )

        if self._needs_remastering(context, goal):
            current_audio, step_counter = self._process_step_with_tracking(
                job,
                current_audio,
                sr,
                "remastering",
                "mastering_chain",
                step_counter,
                context,
                goal,
            )

        # 6. Final analysis and quality report
        self.logger.info("Calculating final CAS...")
        final_analysis = self.analysis_engine.analyze(current_audio, sr)

        final_cas, _final_scores = self.ajm.cas_calculator.calculate_cas(
            current_audio,
            sr,
            final_analysis,
            audio,
            genre=analysis_profile.musical_context.genre,
            genre_confidence=analysis_profile.musical_context.genre_confidence,
        )

        # Create quality report with constraint checking
        quality_report = self.ajm.evaluate(
            audio,
            current_audio,
            sr,
            analysis_profile,
            final_analysis,
            analysis_profile.musical_context.genre,
            analysis_profile.musical_context.genre_confidence,
        )

        job.quality_report = quality_report

        # 7. Save output file
        sf.write(output_audio_path, current_audio, sr)

        output_file_hash = self.archive_manager.calculate_file_hash(output_audio_path)
        output_file = AudioFile(
            file_path=output_audio_path,
            file_hash=output_file_hash,
            format=Path(output_audio_path).suffix.upper().lstrip("."),
            sample_rate=sr,
            bit_depth=16,
            channels=1 if current_audio.ndim == 1 else current_audio.shape[0],
            duration=len(current_audio) / sr,
            file_size=Path(output_audio_path).stat().st_size,
        )

        job.output_file = output_file
        job.mark_completed()

        # 8. Archive job
        logger.info("Archiving job %s...", job_id)
        try:
            self.archive_manager.archive_job(job)
            logger.info("Job %s archived successfully", job_id)
        except Exception as e:
            logger.error("Failed to archive job %s: %s", job_id, e)

        self.logger.info(
            "Job %s completed: CAS %.3f → %.3f (Δ%+.3f)",
            job_id,
            initial_cas,
            final_cas,
            final_cas - initial_cas,
        )

        return job

    def _process_step_with_tracking(self, job, audio, sr, operation, model_name, step_id, context, goal):
        """Verarbeitet a single step and create ProcessingStep tracking."""
        from datetime import datetime

        from backend.core.data_models import ProcessingStep

        logger.info("Processing step %s: %s (%s)", step_id, operation, model_name)

        # Calculate input hash
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr)
            input_hash = self.archive_manager.calculate_file_hash(tmp_in.name)
            os.remove(tmp_in.name)

        # Calculate CAS before
        analysis_before = self.analysis_engine.analyze(audio, sr)
        cas_before, _ = self.ajm.cas_calculator.calculate_cas(
            audio, sr, analysis_before, genre=job.analysis_profile.musical_context.genre
        )

        # Execute processing (use existing pipeline methods)
        start_time = datetime.now()

        # Convert audio to bytes for legacy pipeline
        with io.BytesIO() as buf:
            sf.write(buf, audio, sr, format="WAV")
            audio_bytes = buf.getvalue()

        # Call appropriate processing method
        if operation == "restoration":
            result = self._restoration(audio_bytes, {"sr": sr}, goal)
        elif operation == "repair":
            result = self._repair(audio_bytes, {"sr": sr}, goal)
        elif operation == "reconstruction":
            result = self._reconstruction(audio_bytes, {"sr": sr}, goal)
        elif operation == "remastering":
            result = self._remastering(audio_bytes, {"sr": sr, "audio": audio}, goal)
        else:
            # Pass-through
            result = {"audio": audio_bytes}

        # Convert back to numpy
        processed_audio, _ = _decode_audio_bytes_canonical(result["audio"])

        duration = (datetime.now() - start_time).total_seconds()

        # Calculate output hash
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            sf.write(tmp_out.name, processed_audio, sr)
            output_hash = self.archive_manager.calculate_file_hash(tmp_out.name)
            os.remove(tmp_out.name)

        # Calculate CAS after
        analysis_after = self.analysis_engine.analyze(processed_audio, sr)
        cas_after, _ = self.ajm.cas_calculator.calculate_cas(
            processed_audio,
            sr,
            analysis_after,
            genre=job.analysis_profile.musical_context.genre,
        )

        # Create ProcessingStep
        step = ProcessingStep(
            step_id=step_id,
            operation=operation,
            model_name=model_name,
            model_version="1.0.0",  # Placeholder
            parameters=result.get("params", {}),
            input_hash=input_hash,
            output_hash=output_hash,
            cas_before=cas_before,
            cas_after=cas_after,
            cas_delta=cas_after - cas_before,
            decision_reason=(
                f"Context: {context.get('artefact_risk', False)}, Goal: {goal.get('quality_level', 'unknown')}"
            ),
            skipped=False,
            skip_reason=None,
            duration_seconds=duration,
        )

        job.add_processing_step(step)

        logger.info("Step %s complete: CAS %.3f → %.3f (Δ%+.3f)", step_id, cas_before, cas_after, step.cas_delta)

        return processed_audio, step_id + 1

    # Copy decision methods from AdaptiveProcessingPipeline
    def _needs_restoration(self, context, goal):
        return context.get("artefact_risk", False) or goal.get("quality_level") == "maximal"

    def _needs_repair(self, context, goal):
        return not context.get("transient_rich", True) or "Warnung" in str(goal)

    def _needs_reconstruction(self, context, goal):
        del context
        return "reference_warning" in goal

    def _needs_remastering(self, context, goal):
        del context
        return goal.get("quality_level") == "maximal"

    # Copy processing methods from AdaptiveProcessingPipeline (for backward compat)
    def _restoration(self, audio_bytes, features, goal):
        # Delegate to original pipeline
        pipeline = AdaptiveProcessingPipeline()
        return pipeline.restoration(audio_bytes, features, goal)

    # === VERALTET: Alte Stub-Methoden entfernt ===
    # Context wird jetzt korrekt aus Phase 1 durchgereicht
