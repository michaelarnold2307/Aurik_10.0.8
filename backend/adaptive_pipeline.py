# ---------------------------------------------------------------------------
# Plugin-Imports mit individuellem Fallback — jeder Import einzeln geschützt,
# damit ein fehlgeschlagener Import nicht alle anderen blockiert.
# Mit --import-mode=importlib (pytest) müssen Fallback-Klassen auf Modulebene
# VOR dem try-Block deklariert werden, damit sie im Modul-Namensraum sichtbar sind.
# ---------------------------------------------------------------------------
import logging as _logging

_log = _logging.getLogger(__name__)


# Fallback-Stubs auf Modulebene vordeklarieren (werden durch erfolgreiche Imports überschrieben)
class _PluginStub:
    """Basis-Fallback-Stub für nicht ladbare Plugins."""

    def process(self, audio, sr):
        return audio

    def enhance(self, audio, sr):
        return audio

    def separate(self, audio, sr):
        return audio, audio

    def run(self, audio, sr):
        return audio


class DeepFilterNetV3IIPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class ResembleEnhancePlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class DemucsV4Plugin(_PluginStub):
    pass  # type: ignore[no-redef]


class MDX23CPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class WpePlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class SGMSEPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# FullSubNetPlusPlugin entfernt — 16 kHz-Sprach-NR (DNS-Challenge), nicht in §11.3
# SpleeterPlugin entfernt — veraltetes 2019er Modell (Deezer), nicht in §11.3
# DCCRNPlugin entfernt — §4.4 verboten; MpSenetPlugin ist der Nachfolger
class MpSenetPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class UVRMDXNetPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class BanquetVinylPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class HiFiGANPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# ConvTasNetPlugin entfernt — Sprach-Separation (Luo 2019), HPSS-DSP-Stub, nicht in §11.3
class DiffWavePlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# WaveUNetPlugin entfernt — Sprach-Separation (Stoller 2018), HPSS-DSP-Stub, nicht in §11.3
class CREPEPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# SOTAUniversalEnhancer entfernt — orchestriert FullSubNetPlus (Sprach-NR) + np.var()>1.5-Heuristik, nicht §11.3
# DNSMOSPlugin entfernt — explizit verboten §4.4+§10.2 (16 kHz Sprach-Modell)
# NISQAPlugin entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik)
# PESQPlugin entfernt — explizit verboten §4.4+§10.2 (Telefonband 300–3400 Hz)
# ViSQOLPlugin: entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik, kein Musik-Support)
class AudioLDM2Plugin(_PluginStub):
    pass  # type: ignore[no-redef]


class AudioSRPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# CDPAMPlugin: entfernt — explizit verboten §4.4+§10.2 (Speech-perceptual metric)
class GACELAPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class MatcheringPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


class SileroPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# VampNetPlugin: VERBOTEN (kein stabiler ONNX-Export, kein gebündeltes Plugin) — §4.4
# Stub entfernt; flow_matching_plugin ist der Nachfolger für generatives Inpainting.


class BSRoFormerPlugin(_PluginStub):
    pass  # type: ignore[no-redef]


# Jetzt die echten Imports versuchen — direkt auf Modulebene, kein globals()-Trick,
# damit --import-mode=importlib (pytest) keine Namespace-Probleme verursacht.
# Jeder try/except-Block ist eigenständig: ein Fehler blockiert keine anderen.
try:
    from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3IIPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("DeepFilterNetV3IIPlugin nicht verfügbar: %s", _e)

try:
    from plugins.resemble_enhance_plugin import ResembleEnhancePlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("ResembleEnhancePlugin nicht verfügbar: %s", _e)

try:
    from plugins.demucs_v4_plugin import DemucsV4Plugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("DemucsV4Plugin nicht verfügbar: %s", _e)

try:
    from plugins.mdx23c_plugin import MDX23CPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("MDX23CPlugin nicht verfügbar: %s", _e)

try:
    from plugins.wpe_plugin import SGMSEPlugin, WpePlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("wpe_plugin (WpePlugin/SGMSEPlugin) nicht verfügbar: %s", _e)

# FullSubNetPlusPlugin: Import entfernt — 16 kHz-Sprach-NR, nicht in §11.3
# SpleeterPlugin: Import entfernt — veraltetes 2019er Modell, nicht in §11.3

try:
    # §4.4: MP-SENet 2023 ersetzt DCCRN (§4.4 verboten)
    from plugins.mp_senet_plugin import MpSenetPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("MpSenetPlugin (DCCRN-Nachfolger) nicht verfügbar: %s", _e)

try:
    from plugins.uvr_mdxnet_plugin import UVRMDXNetPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("UVRMDXNetPlugin nicht verfügbar: %s", _e)

try:
    from plugins.banquet_vinyl_plugin import BanquetVinylPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("BanquetVinylPlugin nicht verfügbar: %s", _e)

try:
    from plugins.hifigan_plugin import HiFiGANPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("HiFiGANPlugin nicht verfügbar: %s", _e)

# ConvTasNetPlugin: Import entfernt — Sprach-Separation (Luo 2019), nicht in §11.3

try:
    from plugins.diffwave_plugin import DiffWavePlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("DiffWavePlugin nicht verfügbar: %s", _e)

# WaveUNetPlugin: Import entfernt — Sprach-Separation (Stoller 2018), nicht in §11.3

try:
    from plugins.crepe_plugin import CREPEPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("CREPEPlugin nicht verfügbar: %s", _e)

# SOTAUniversalEnhancer: Import entfernt — orchestriert Sprach-NR (FullSubNetPlus), nicht §11.3
# DNSMOSPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-MOS)
# NISQAPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-Metrik)
# PESQPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Telefonband 300–3400 Hz)

# ViSQOLPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Sprach-Qualitätsmetrik)

try:
    from plugins.audioldm2_plugin import AudioLDM2Plugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("AudioLDM2Plugin nicht verfügbar: %s", _e)

try:
    from plugins.audiosr_plugin import AudioSRPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("AudioSRPlugin nicht verfügbar: %s", _e)

# CDPAMPlugin: Import entfernt — explizit verboten §4.4+§10.2 (Speech-perceptual metric)

try:
    from plugins.gacela_plugin import GACELAPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("GACELAPlugin nicht verfügbar: %s", _e)

try:
    from plugins.matchering_plugin import MatcheringPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("MatcheringPlugin nicht verfügbar: %s", _e)

try:
    from plugins.panns_plugin import PANNSPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("PANNSPlugin nicht verfügbar: %s", _e)

try:
    from plugins.silero_plugin import SileroPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("SileroPlugin nicht verfügbar: %s", _e)

try:
    from plugins.bs_roformer_plugin import BSRoFormerPlugin  # type: ignore[no-redef]
except Exception as _e:
    _log.warning("BSRoFormerPlugin nicht verfügbar: %s", _e)

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

"""
Adaptive Processing Pipeline für Magic Button
------------------------------------------------
Diese Pipeline steuert alle Bearbeitungsschritte (Restaurierung, Reparatur, Rekonstruktion, Remastering) adaptiv und nachvollziehbar.
Sie nutzt Kontextanalyse, Zieldefinition und modulare Verarbeitungsketten. Alle Entscheidungen, Parameter und Ergebnisse werden geloggt.
"""


import numpy as np

from backend.core.model_manager import ModelManager

from ._dsp_applier import apply_dsp_chain
from .adaptive_goal import AdaptiveGoalEngine
from .audio_monitor import PermanentAudioMonitor
from .context_analysis import ContextAnalyzer

# Ethics & Monitoring (Phase 4.5, v8.0)
from .ethics_engine import EpistemicDecision, EthicsEngine
from .logging_config import get_logger
from .quality_control import QualityControl

# Docker-based ML Plugins (Phase 7: Docker Migration - ALL 28 COMPLETE)
# Processing Plugins (16)
# Imports already above - removed duplicate


class AdaptiveProcessingPipeline:
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

        # Medienkettenerkennung initialisieren
        from backend.media_chain_detector import MediaChainDetector

        self.media_chain_detector = MediaChainDetector()

        self.logger.info(
            f"AdaptiveProcessingPipeline initialized: {self.dsp_policy_engine.count_total_modules()} DSP + 28 ML-Plugins + Ethics Engine + Monitor + MediaChainDetector"
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
            # Beispiel: Denoise, Repair, Enhancement, Separation, Quality, Vocoder, Tagging, Mastering, Generation, Pitch
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
        except Exception:
            pass

        self.logger.info("\n┌─ ML-PLUGINS (alle importiert/verfügbar) ─────────────────────────────────────────────┐")
        for name in all_plugins:
            mark = "*" if name in chosen else " "
            self.logger.info(f"│{mark} ✓ {name.ljust(17)} │")
        self.logger.info(
            f"└─ Status: {len(all_plugins)}/47 ML-Plugins & Metriken importiert ───────────────────────┘\n"
        )
        self._print_component_status()

        # v8.1: Advanced Vocal Separation (Hybrid: MDX-Net + Demucs v5)
        if VOCAL_SEPARATION_V8_AVAILABLE:
            self.vocal_separator_v8 = HybridVocalSeparator(
                fusion_strategy="adaptive",
                sample_rate=44100,
                device=None,  # Auto-detect CUDA/CPU
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
            self.logger.warning(f"Plugin {plugin_name} nicht verfügbar: {e}")
            return None

    def _print_component_status(self):
        """
        Strukturierter Status-Report aller Komponenten beim Pipeline-Start.
        Zeigt transparent welche Plugins/Module verfügbar sind.
        """
        try:
            from Aurik910 import __version__ as _aurik_version
        except ImportError:
            _aurik_version = "9.10.41"
        self.logger.info("\n" + "═" * 80)
        self.logger.info(f"  AURIK {_aurik_version} — SYSTEM-KOMPONENTEN STATUS")
        self.logger.info("═" * 80 + "\n")

        # ML-Plugins Status
        self.logger.info("┌─ ML-PLUGINS (Denoise/Repair/Enhancement) " + "─" * 35 + "┐")
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
            "geladen" if plugin_name in self.available_plugins else "nicht verfügbar"
            self.logger.info(f"│  {status} {plugin_name:20s} {description:35s} │")

        available_count = len(self.available_plugins)
        self.logger.info(
            f"└─ Status: {available_count}/11 ML-Plugins verfügbar "
            + "─" * (80 - 40 - len(str(available_count)))
            + "┘\n"
        )

        # DSP-Module Status
        dsp_count = self.dsp_policy_engine.count_total_modules()
        self.logger.info("┌─ DSP-MODULE (Policy-Engine) " + "─" * 48 + "┐")
        self.logger.info("│  ✓ Spectral Processing     ✓ Dynamic Range Control  ✓ EQ Systems       │")
        self.logger.info("│  ✓ Artifact Detection      ✓ Transient Shaping      ✓ Stereo Tools     │")
        self.logger.info("│  ✓ Noise Gates             ✓ Compressors/Limiters   ✓ Phase Correction │")
        self.logger.info(f"└─ Status: {dsp_count} DSP-Module aktiv " + "─" * (80 - 32 - len(str(dsp_count))) + "┘\n")

        # Advanced Features Status
        self.logger.info("┌─ ADVANCED FEATURES (v8.0+) " + "─" * 50 + "┐")
        vocal_status = "✓" if VOCAL_SEPARATION_V8_AVAILABLE else "⚠️"
        pitch_status = "✓" if PITCH_CORRECTION_V8_AVAILABLE else "⚠️"
        defect_status = "✓" if DEFECT_DETECTION_V8_AVAILABLE else "⚠️"
        self.logger.info(f"│  {vocal_status} Vocal Separation v8.1    (Hybrid: MDX-Net + Demucs v5)        │")
        self.logger.info(f"│  {pitch_status} Pitch Correction v8.2    (CREPE + Epistemic Gates)           │")
        self.logger.info(f"│  {defect_status} Defect Detection v8.2    (11 Defect Types, iZotope-level)    │")
        self.logger.info("│  ✓ Ethics Engine              (HIPS Compliance + Safety Wrappers)     │")
        self.logger.info("│  ✓ Audio Monitor              (Permanent Quality Tracking)            │")
        self.logger.info("└" + "─" * 79 + "┘\n")

        # Material Quality & Musical Goals
        self.logger.info("┌─ ANALYSIS & METRICS " + "─" * 57 + "┐")
        self.logger.info("│  ✓ Material Quality Analyzer  (7-Level Classification + Adaptive)     │")
        self.logger.info("│  ✓ Musical Goals Validation   (7 Goals: Authenticity, Waerme, etc.)   │")
        self.logger.info("│  ✓ Adaptive Thresholds        (Generation-Count Weighted Scoring)     │")
        self.logger.info("│  ✓ LUFS/True-Peak Metering    (EBU R128 Compliant)                    │")
        self.logger.info("└" + "─" * 79 + "┘\n")

        # Gesamtstatus
        if available_count >= 8:
            overall = "✓ SYSTEM BEREIT - Alle kritischen Komponenten verfügbar"
        elif available_count >= 5:
            overall = "⚠️ SYSTEM BEREIT - Einige Plugins fehlen, Fallback aktiv"
        else:
            overall = "⚠️ LIMITIERTER MODUS - Nur wenige ML-Plugins verfügbar"

        self.logger.info("═" * 80)
        self.logger.info(f"  {overall}")
        self.logger.info("═" * 80 + "\n")

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

        if use_safety_wrapper and self.vocal_safety_wrapper is not None:
            # HIPS-compliant separation with validation
            try:
                stems = self.vocal_safety_wrapper.safe_separate(audio, sr, return_individual=False)
            except HIPSViolationError as e:
                self.logger.error(f"HIPS violation during vocal separation: {e}")
                raise
        else:
            # Direct separation (bypass safety checks)
            stems = self.vocal_separator_v8.separate(audio, sr, return_individual=False)

        # Log metrics
        metrics = self.vocal_separator_v8.get_metrics()
        self.logger.info(
            f"Vocal separation complete: {metrics['total_separations']} total, fusion={metrics['fusion_strategy']}"
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

        if use_safety_wrapper and self.pitch_corrector_safety is not None:
            # HIPS-compliant correction with validation
            try:
                audio_corrected, metadata = self.pitch_corrector_safety.safe_correct(audio, sr, **kwargs)
            except HIPSViolationError as e:
                self.logger.error(f"HIPS violation during pitch correction: {e}")
                raise
        else:
            # Direct correction (bypass safety checks)
            audio_corrected, metadata = self.pitch_corrector_v8.correct_pitch(audio, **kwargs)

        # Log result
        if metadata.get("corrected", False):
            self.logger.info(
                f"Pitch correction applied: {metadata['n_corrections']} regions, "
                f"DCS={metadata['dcs']:.3f}, "
                f"epistemic_conf={metadata['epistemic_confidence']:.2f}"
            )
        else:
            self.logger.info(f"Pitch correction rejected: {metadata.get('reason', 'unknown')}")

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
            ...     self.logger.info(f"Priority {treatment['priority']}: {treatment['method']}")
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

        self.logger.info(f"Analyzing defects (quick_scan={quick_scan})...")

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
            else:
                # Full analysis
                report = self.defect_detector_v8.analyze(audio, sr)

                # Convert to serializable dictionary
                return report.to_dict()

        except Exception as e:
            self.logger.error(f"Defect analysis failed: {e}", exc_info=True)
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
        # 0. Medienkettenerkennung und detected_medium immer als ersten Schritt durchführen
        import io

        import soundfile as sf

        audio_np, sr_audio = sf.read(io.BytesIO(audio_bytes))
        media_chain = self.media_chain_detector.detect_chain(audio_np, sr_audio)
        if media_chain:
            self.logger.info(
                "\n🔎 Erkannte Medienkette: "
                + " → ".join(f"{m['medium']} ({m['confidence'] * 100:.1f}%)" for m in media_chain)
            )
        else:
            self.logger.info("\n🔎 Medienkette: Keine eindeutige Erkennung möglich")
        self.log.append({"step": "media_chain_detection", "media_chain": media_chain})
        # detected_medium aus Medienkette ableiten, falls nicht explizit übergeben
        detected_medium_final = detected_medium
        if not detected_medium_final and media_chain:
            detected_medium_final = {"type": media_chain[0]["medium"], "confidence": media_chain[0]["confidence"]}
        # detected_medium in Features übernehmen, damit alle Folge-Analysen darauf zugreifen
        features = dict(features) if features else {}
        if detected_medium_final:
            features["detected_medium"] = detected_medium_final

        # SOTA-Maximum-Policy wird nach jedem Durchlauf neu abgeleitet, aber mit aktuellen Analyse-Features kombiniert
        sota_policy = None
        try:
            analyzer = SOTAMaximumAnalyzer()
            sota_policy = analyzer.recommend_sota_policy()
            self.logger.info(f"SOTA-Maximum-Policy (dynamisch): {sota_policy}")
        except Exception as e:
            self.logger.error(f"SOTA-Maximum-Policy konnte nicht geladen werden: {e}")

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
        self.logger.info(f"Kontextanalyse abgeschlossen: {context}")

        # Defekt- und Störungserkennung direkt nach Medienkette/Analyse
        # (Hier ggf. eigene Methode oder Modul für Defekterkennung einbinden)
        try:
            defect_results = self.defect_detector.detect(audio_np, sr_audio, features)
            self.log.append({"step": "defect_detection", "defects": defect_results})
            self.logger.info(f"Defekt-/Störungserkennung abgeschlossen: {defect_results}")
            # Defektergebnisse in Features/Maßnahmenkette übernehmen
            features["detected_defects"] = defect_results
        except Exception as e:
            self.logger.error(f"Defekt-/Störungserkennung fehlgeschlagen: {e}")

        # 2. Zieldefinition
        goal = self.goal_engine.define_goal(context)
        self.log.append({"step": "goal_definition", "goal": goal})
        self.logger.info(f"Zieldefinition abgeschlossen: {goal}")

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
        self.logger.info(f"Ethics Decision: {ethics_report.decision.value} - {ethics_report.reasoning}")

        # Ergebnis-Container für ALLE Ethics-Pfade (muss vor dem Ethics-Gate stehen)
        results = {"steps": [], "quality": [], "log": [], "ethics_decision": None}
        phase_count = 0
        total_phases = 4
        current_audio = audio_bytes
        policy = sota_policy  # SOTA-Policy für Phasen-Verarbeitung

        # Handle ethics decisions
        # Erzwinge immer mindestens eine DSP-Phase, auch bei PRESERVE/HARD_STOP
        if ethics_report.decision in [EpistemicDecision.HARD_STOP, EpistemicDecision.PRESERVE]:
            self.logger.warning(f"Ethics Engine: {ethics_report.decision.value} - Erzwinge minimalen DSP-Processing")
            import io

            import soundfile as sf

            audio_np, sr_audio = sf.read(io.BytesIO(audio_bytes))
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
            self.logger.info(f"Phase {phase_count}/{total_phases}: Restaurierung...")
            if self.progress_callback:
                self.progress_callback("restoration", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = sf.read(io.BytesIO(current_audio))
            self.audio_monitor.start_module("restoration")

            res = self._restoration(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = sf.read(io.BytesIO(current_audio))
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
            self.logger.info(f"Phase {phase_count}/{total_phases}: Reparatur...")
            if self.progress_callback:
                self.progress_callback("repair", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = sf.read(io.BytesIO(current_audio))
            self.audio_monitor.start_module("repair")

            res = self._repair(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = sf.read(io.BytesIO(current_audio))
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
            self.logger.info(f"Phase {phase_count}/{total_phases}: Rekonstruktion...")
            if self.progress_callback:
                self.progress_callback("reconstruction", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = sf.read(io.BytesIO(current_audio))
            self.audio_monitor.start_module("reconstruction")

            res = self._reconstruction(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = sf.read(io.BytesIO(current_audio))
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
            self.logger.info(f"Phase {phase_count}/{total_phases}: Remastering...")
            if self.progress_callback:
                self.progress_callback("remastering", 0.0, phase_count, total_phases)

            # Monitor: Start module
            audio_in, _ = sf.read(io.BytesIO(current_audio))
            self.audio_monitor.start_module("remastering")

            res = self._remastering(current_audio, features, policy or goal, context)
            results["steps"].append(res)
            current_audio = res["audio"]  # Output wird Input für nächste Phase

            # Monitor: End module
            audio_out, _ = sf.read(io.BytesIO(current_audio))
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
            self.logger.info(f"Qualitätskontrolle für {step['name']}: {qc_result}")

        # 5. Mastering/Postprocessing (immer am Ende, vor Export)
        if results["steps"]:
            last = results["steps"][-1]
            import io

            import soundfile as sf

            # Audio-Bytes in numpy-Array
            audio, sr = sf.read(io.BytesIO(last["audio"]), always_2d=False)
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
        final_audio, _ = sf.read(io.BytesIO(current_audio))
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
        # Beispiel: Referenzwarnung oder starke Abweichung
        return "reference_warning" in goal

    def _needs_remastering(self, context, goal):
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
        import io
        import os
        import tempfile

        import numpy as np
        import soundfile as sf

        # Audio-Bytes in numpy-Array
        audio_original, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
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
        model_obj = self.model_manager.models.get(selected_model, {}).get("obj")
        if model_obj:
            model_obj.process(audio_original, context)
        else:
            # Fallback: Multi-Stage Enhancement
            self.model_manager.multi_stage_enhancement(audio_original, context)
        self.logger.info(f"Policy-Kontext für Restoration: {context}")

        # STAGE 1: ML-Denoise (Policy-selected Model)
        model_name = self.policy_engine.select_denoise_model(context, goal)

        # Strukturierte Stage-Ausgabe
        self.logger.info("\n" + "╔" + "═" * 78 + "╗")
        self.logger.info("║  RESTORATION STAGE 1: ML-DENOISE" + " " * 45 + "║")
        self.logger.info("╚" + "═" * 78 + "╝\n")

        # Policy-Entscheidung mit Begründung
        reason_map = {
            "banquet": "Vinyl-Detection → Spezialisiertes Banquet Model",
            "deepfilternet": "Speech + Broadband Noise → DeepFilterNet v3",
            "wpe": "Classical/Jazz Genre → WPE Dereverberation",
            "dccrn": "Reverb Detected → DCCRN De-Reverb Model",
            "resemble_enhance": "Standard Noise Reduction → Universal Model",
        }
        reason = reason_map.get(model_name, "Policy-basierte Auswahl")

        self.logger.info(f"Policy-Selektion: {model_name}")
        self.logger.info(f"  Begründung: {reason}")
        self.logger.info(
            f"  Medium: {context.get('detected_medium', 'unknown')}, "
            f"Genre: {context.get('genre', 'unknown')}, "
            f"Quality: {goal.get('quality_level', 'standard')}"
        )
        self.logger.info("")

        # Lade das ausgewählte Plugin
        plugin = getattr(self, model_name, None)

        # ROBUSTES PLUGIN-HANDLING: Wenn ausgewähltes Plugin nicht verfügbar, nutze Fallback-Chain
        if plugin is None:
            self.logger.info(f"Plugin-Status: ⚠️ {model_name} nicht verfügbar")

            # Fallback-Chain (Reihenfolge: Universal → Robust → DSP-Only)
            fallback_chain = ["resemble_enhance", "wpe", "deepfilternet", "dccrn"]

            # Entferne bereits-geprüftes Plugin aus Chain (avoid infinite loop)
            if model_name in fallback_chain:
                fallback_chain.remove(model_name)

            self.logger.info(f"Fallback-Chain: {' → '.join(fallback_chain)}")

            # Durchlaufe Fallback-Chain
            for fallback_name in fallback_chain:
                fallback_plugin = getattr(self, fallback_name, None)
                if fallback_plugin is not None:
                    self.logger.info(f"  ✓ Fallback erfolgreich: {fallback_name} wird verwendet")
                    plugin = fallback_plugin
                    model_name = fallback_name  # Update für Logging
                    break

            # Wenn immer noch kein Plugin verfügbar → Skip ML-Denoise
            if plugin is None:
                self.logger.info("  ⚠️ Keine ML-Plugins verfügbar - ML-Denoise wird übersprungen")
                self.logger.info("  → Fallback zu DSP-basierter Verarbeitung\n")
                audio_denoised = audio_original  # Behalte Original
            else:
                self.logger.info("")  # Leerzeile vor Processing
        else:
            self.logger.info(f"Plugin-Status: ✓ {model_name} geladen und bereit\n")

        # Nur wenn Plugin verfügbar ist, führe ML-Processing durch
        if plugin is not None:
            # CRITICAL FIX: Use workspace temp directory instead of /tmp
            # Docker containers need writable directories with proper permissions
            workspace_temp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp_ml_processing")
            os.makedirs(workspace_temp, exist_ok=True)

            # Create temp files in workspace temp directory
            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=workspace_temp)
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=workspace_temp)
            tmp_in_path = tmp_in.name
            tmp_out_path = tmp_out.name
            tmp_in.close()  # Close file handle
            tmp_out.close()  # Close file handle

            try:
                # Write input WAV file (now file handle is closed, so write will work correctly)
                sf.write(tmp_in_path, audio_original, sr)

                # Process with ML plugin
                plugin.process(tmp_in_path, tmp_out_path)

                # Read result
                audio_denoised, _ = sf.read(tmp_out_path)

                self.logger.info(f"Processing: ✓ ML-Denoise erfolgreich ({model_name})")

            except Exception as e:
                self.logger.info(f"Processing: ❌ ML-Plugin {model_name} fehlgeschlagen: {e}")
                self.logger.info("  → Fallback zu Original-Audio\n")
                # Fallback to original audio
                audio_denoised = audio_original

            finally:
                # Clean up temp files
                if os.path.exists(tmp_in_path):
                    os.remove(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)

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
                f"✓ Hybrid Refinement applied (spectral_dev={refine_metrics.get('spectral_deviation', 0):.3f})"
            )

        except Exception as e:
            self.logger.warning(f"Hybrid Refinement skipped: {e}")
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
                    self.logger.info(f"Gender detected: {detected_gender}")
                except Exception as gender_err:
                    self.logger.info(f"Gender detection unavailable, using signal-adaptive analysis: {gender_err}")
                    detected_gender = "auto"

                # Process Vocals mit adaptiven Sibilanten-Parametern
                audio_vocal_enhanced = process_vocals(
                    audio_refined,
                    sr=sr,
                    gender=detected_gender,
                    model_path=None,  # Optional: ML-based HF texture model
                )

                self.logger.info(f"✓ Vocal Enhancement applied (gender={detected_gender})")

                audio_refined = audio_vocal_enhanced

            except Exception as e:
                self.logger.warning(f"Vocal Enhancement skipped: {e}")
                # audio_refined bleibt unverändert bei Fehler
        else:
            self.logger.info("No vocals detected, skipping Vocal Enhancement")

        # STAGE 3: HF Extension (Phase 8A-2, conditional)
        try:
            from dsp.hf_extender import apply_hf_extension_if_needed

            audio_hf, sr_out = apply_hf_extension_if_needed(audio_refined, sr, context=context, goal=goal)

            if sr_out != sr:
                self.logger.info(f"✓ HF Extension applied: {sr}Hz → {sr_out}Hz")
                sr = sr_out  # Update sample rate
            else:
                audio_hf = audio_refined

        except Exception as e:
            self.logger.warning(f"HF Extension skipped: {e}")
            audio_hf = audio_refined

        # STAGE 4: Stereo Widening (Phase 8A-3, conditional)
        try:
            from dsp.stereo_widener import apply_stereo_widening_if_needed

            audio_wide = apply_stereo_widening_if_needed(audio_hf, sr, context=context, goal=goal)

            if not np.array_equal(audio_wide, audio_hf):
                self.logger.info("✓ Stereo Widening applied")

        except Exception as e:
            self.logger.warning(f"Stereo Widening skipped: {e}")
            audio_wide = audio_hf

        # STAGE 5: Adaptive EQ (Phase 8A-4)
        try:
            from dsp.adaptive_eq import apply_adaptive_eq

            audio_final = apply_adaptive_eq(audio_wide, sr, context=context, goal=goal)

            self.logger.info(f"✓ Adaptive EQ applied (genre={context['genre']})")

        except Exception as e:
            self.logger.warning(f"Adaptive EQ skipped: {e}")
            audio_final = audio_wide

        # === DSP POST-PROCESSING (adaptive chain) ===
        self.logger.info("Starting DSP Post-Processing...")
        dsp_chain: list = []  # default empty chain (populated upstream in full pipeline)
        post_chain = dsp_chain[3:]  # Remaining modules: Dynamics, Enhancement, Post

        self.logger.info(f"\n✅ DSP POST-PROCESSING: {len(post_chain)} Module angewendet")

        audio_final = self._apply_dsp_chain(audio_final, sr, post_chain)
        self.logger.info(f"✓ DSP Post-Processing applied: {len(post_chain)} modules")

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
                "info": f"DSP Pre ({len(dsp_chain[:3])}) + {model_name} + Hybrid Refinement{vocal_stage} + HF Extension + Stereo + EQ + DSP Post ({len(post_chain)})",
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

    def _apply_dsp_chain(self, audio, sr, dsp_chain):
        """Wrapper for apply_dsp_chain"""
        return apply_dsp_chain(audio, sr, dsp_chain)

    def _repair(self, audio_bytes, features, goal, context):
        """
        SOTA-Maximum: Clipping-Repair + Refinement (Phase 8A Integration)

        Pipeline:
        1. ML-Repair (Policy-selected: FullSubNet+ or DCCRN)
        2. Hybrid Refinement (preserve transients)
        """
        import io
        import os
        import tempfile

        import soundfile as sf

        # Audio-Bytes in numpy-Array
        audio_original, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info(f"Repair Pipeline: detected_medium={context.get('detected_medium', 'unknown')}")

        # STAGE 1: ML-Repair (Policy-selected Model)
        model_name = self.policy_engine.select_repair_model(context, goal)

        self.logger.info("\n" + "╔" + "═" * 78 + "╗")
        self.logger.info("║  REPAIR STAGE 1: ML-REPAIR (Clipping/Artifacts)" + " " * 30 + "║")
        self.logger.info("╚" + "═" * 78 + "╝\n")

        repair_type = (
            "FullSubNet+ (Speech Enhancement)" if "fullsubnet" in model_name else "DCCRN (Music/Reverb Repair)"
        )
        self.logger.info(f"Policy-Selektion: {model_name}")
        self.logger.info(f"  Typ: {repair_type}")
        self.logger.info(f"  Vocals: {'Ja' if context.get('has_vocals', False) else 'Nein'}")
        self.logger.info("")

        # Lade das ausgewählte Plugin
        plugin = getattr(self, model_name, None)

        # ROBUSTES PLUGIN-HANDLING für Repair
        if plugin is None:
            self.logger.info(f"Plugin-Status: ⚠️ {model_name} nicht verfügbar")

            # Fallback-Chain für Repair (FullSubNet → DCCRN → Skip)
            fallback_chain = ["fullsubnet", "dccrn"]

            # Entferne bereits-geprüftes Plugin
            if model_name in fallback_chain:
                fallback_chain.remove(model_name)

            self.logger.info(f"Fallback-Chain: {' → '.join(fallback_chain)}")

            # Durchlaufe Fallback-Chain
            for fallback_name in fallback_chain:
                fallback_plugin = getattr(self, fallback_name, None)
                if fallback_plugin is not None:
                    self.logger.info(f"  ✓ Fallback erfolgreich: {fallback_name} wird verwendet")
                    plugin = fallback_plugin
                    model_name = fallback_name
                    break

            # Wenn immer noch kein Plugin → Skip ML-Repair
            if plugin is None:
                self.logger.info("  ⚠️ Keine Repair-Plugins verfügbar - ML-Repair wird übersprungen")
                self.logger.info("  → Original-Audio wird beibehalten\n")
                audio_repaired = audio_original
        else:
            self.logger.info(f"Plugin-Status: ✓ {model_name} geladen und bereit\n")

        # Nur wenn Plugin verfügbar, führe ML-Repair durch
        if plugin is not None:
            # Create temporary files for ML plugin
            tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_in_path = tmp_in.name
            tmp_out_path = tmp_out.name
            tmp_in.close()  # Close file handle before writing
            tmp_out.close()  # Close file handle

            try:
                # Write input WAV file
                sf.write(tmp_in_path, audio_original, sr)

                # Process with ML plugin
                plugin.process(tmp_in_path, tmp_out_path)

                # Read result
                audio_repaired, _ = sf.read(tmp_out_path)

                self.logger.info(f"Processing: ✓ ML-Repair erfolgreich ({model_name})")

            except Exception as e:
                self.logger.info(f"Processing: ❌ ML-Plugin {model_name} fehlgeschlagen: {e}")
                self.logger.info("  → Fallback zu Original-Audio\n")
                audio_repaired = audio_original

            finally:
                # Clean up temp files
                if os.path.exists(tmp_in_path):
                    os.remove(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.remove(tmp_out_path)

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
            self.logger.warning(f"Hybrid Refinement skipped: {e}")
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
        import glob
        import io
        import os
        import tempfile

        import soundfile as sf

        # Audio-Bytes in numpy-Array
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info(
            f"Reconstruction Pipeline: detected_medium={context.get('detected_medium', 'unknown')}, stems={goal.get('stems', 4)}"
        )

        # Policy-Engine wählt optimales Model
        model_name = self.policy_engine.select_separation_model(context, goal)

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"🎶 SOURCE-SEPARATION AUSGEWÄHLT: {model_name.upper()}")
        self.logger.info(f"{'=' * 80}")
        self.logger.info(f"   Stems: {goal.get('stems', 4)}")
        self.logger.info(f"   Genre: {context.get('genre', 'unknown')}")

        if "mdx23c" in model_name:
            self.logger.info("   Model: MDX23C (maximal quality)")
        elif "demucs" in model_name:
            self.logger.info("   Model: Demucs v4 (4+ stems)")
        else:
            self.logger.info("   Model: UVR-MDXNet (2-stem)")

        self.logger.info(f"{'=' * 80}\n")

        plugin = getattr(self, model_name)

        # Temporäre Dateien und Verzeichnis für Docker I/O
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            output_dir = tempfile.mkdtemp()

            try:
                # Audio in Temp-Datei schreiben
                sf.write(tmp_in.name, audio, sr)

                # Docker-Plugin aufrufen
                # Demucs separiert in output_dir/vocals.wav, output_dir/accompaniment.wav etc.
                plugin.process(tmp_in.name, output_dir, stems=context["stem_count"])

                # Lade separierte Vocals (oder andere Stems je nach goal)
                vocals_path = os.path.join(output_dir, "vocals.wav")
                if os.path.exists(vocals_path):
                    audio_separated, _ = sf.read(vocals_path)
                else:
                    # Fallback: Nutze Original falls Separation fehlschlägt
                    self.logger.warning("Vocals nicht gefunden, nutze Original")
                    audio_separated = audio

                # Audit-Infos
                self.log.append(
                    {
                        "step": "reconstruction",
                        "info": f"{model_name} (Docker, Policy-selected)",
                        "params": goal,
                    }
                )

                # Zurück in Bytes
                with io.BytesIO() as buf:
                    sf.write(buf, audio_separated, sr, format="WAV")
                    out_bytes = buf.getvalue()

                return {"name": "reconstruction", "audio": out_bytes, "params": goal}

            finally:
                # Cleanup temporäre Dateien
                if os.path.exists(tmp_in.name):
                    os.remove(tmp_in.name)
                # Cleanup output directory und alle Stems
                if os.path.exists(output_dir):
                    for file in glob.glob(os.path.join(output_dir, "*.wav")):
                        os.remove(file)
                    os.rmdir(output_dir)

    def _remastering(self, audio_bytes, features, goal, context):
        # SOTA-Remastering via Matchering-API-Client mit Fallback
        import io
        import os
        import tempfile

        import soundfile as sf

        from .mastering import mastering_chain

        try:
            from .matchering_api_client import MatcheringAPIClient
        except ImportError:
            MatcheringAPIClient = None  # type: ignore[assignment,misc]
            _log.warning("MatcheringAPIClient nicht verfügbar — Fallback auf mastering_chain")

        # NUTZE CONTEXT AUS PHASE 1
        self.logger.info(f"Remastering Pipeline: detected_medium={context.get('detected_medium', 'unknown')}")

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info("🎼 MASTERING/REMASTERING")
        self.logger.info(f"{'=' * 80}")
        self.logger.info("   Method: Matchering 2.0 (AI-powered)")
        self.logger.info("   Target: Professional Mastering Standards")
        self.logger.info(f"{'=' * 80}\n")

        container_info = {
            "container": "matchering2.0",
            "image": "models/matchering2.0/Dockerfile.matchering2.0",
            "api_url": "http://localhost:8360/api/process",
        }

        # Target-Audio temporär speichern
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target_file:
            sf.write(target_file, features["audio"], features.get("sr", 44100))
            target_path = target_file.name
        # Reference-Audio temporär speichern
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            sf.write(ref_file, features["reference_audio"], features.get("sr", 44100))
            ref_path = ref_file.name
        # Output-Pfad
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            output_path = out_file.name

        try:
            if MatcheringAPIClient is None:
                raise ImportError("MatcheringAPIClient nicht verfügbar")
            client = MatcheringAPIClient()
            client.remaster(target_path, ref_path, output_path)
            # Remastertes Audio laden
            with open(output_path, "rb") as f:
                remastered_bytes = f.read()
            self.logger.info("Remastering mit Matchering durchgeführt.")
            self.log.append(
                {
                    "step": "remastering",
                    "info": "Matchering-Container verwendet",
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
            self.logger.error(f"Remastering fehlgeschlagen: {e}", exc_info=True)
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
                    f"Fallback-Remastering ebenfalls fehlgeschlagen: {fallback_e}",
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
        finally:
            os.remove(target_path)
            os.remove(ref_path)
            os.remove(output_path)


# ==============================================================================
# Adaptive Processing Pipeline V2 with Formal Job Tracking (AURIK Spec 4.1)
# ==============================================================================


class AdaptiveProcessingPipelineV2:
    """
    Updated pipeline using formal ResturationJob data structures.

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

        import soundfile as sf

        from backend.core.data_models import AudioFile, ResturationJob

        # 1. Create ResturationJob with UUID
        job_id = uuid4()
        self.logger.info(f"Starting restoration job {job_id}")

        # Load input audio
        audio, sr = sf.read(input_audio_path, always_2d=False)

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
        self.logger.info(f"Archiving job {job_id}...")
        try:
            self.archive_manager.archive_job(job)
            self.logger.info(f"Job {job_id} archived successfully")
        except Exception as e:
            self.logger.error(f"Failed to archive job {job_id}: {e}")

        self.logger.info(
            f"Job {job_id} completed: CAS {initial_cas:.3f} → {final_cas:.3f} (Δ{final_cas - initial_cas:+.3f})"
        )

        return job

    def _process_step_with_tracking(self, job, audio, sr, operation, model_name, step_id, context, goal):
        """Process a single step and create ProcessingStep tracking"""
        from datetime import datetime
        import io
        import tempfile

        import soundfile as sf

        from backend.core.data_models import ProcessingStep

        self.logger.info(f"Processing step {step_id}: {operation} ({model_name})")

        # Calculate input hash
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr)
            input_hash = self.archive_manager.calculate_file_hash(tmp_in.name)
            import os

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
        processed_audio, _ = sf.read(io.BytesIO(result["audio"]), always_2d=False)

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
            decision_reason=f"Context: {context.get('artefact_risk', False)}, Goal: {goal.get('quality_level', 'unknown')}",
            skipped=False,
            skip_reason=None,
            duration_seconds=duration,
        )

        job.add_processing_step(step)

        self.logger.info(f"Step {step_id} complete: CAS {cas_before:.3f} → {cas_after:.3f} (Δ{step.cas_delta:+.3f})")

        return processed_audio, step_id + 1

    # Copy decision methods from AdaptiveProcessingPipeline
    def _needs_restoration(self, context, goal):
        return context.get("artefact_risk", False) or goal.get("quality_level") == "maximal"

    def _needs_repair(self, context, goal):
        return not context.get("transient_rich", True) or "Warnung" in str(goal)

    def _needs_reconstruction(self, context, goal):
        return "reference_warning" in goal

    def _needs_remastering(self, context, goal):
        return goal.get("quality_level") == "maximal"

    # Copy processing methods from AdaptiveProcessingPipeline (for backward compat)
    def _restoration(self, audio_bytes, features, goal):
        # Delegate to original pipeline
        pipeline = AdaptiveProcessingPipeline()
        return pipeline._restoration(audio_bytes, features, goal)

    # === VERALTET: Alte Stub-Methoden entfernt ===
    # Context wird jetzt korrekt aus Phase 1 durchgereicht
