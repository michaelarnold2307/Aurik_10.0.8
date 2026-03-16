"""
ML Model Policy Engine - AURIK 6.0
===================================

Intelligente Modellauswahl basierend auf Audio-Kontext.
Wählt optimales SOTA-Modell für jede Restaurierungs-Aufgabe.

Verwendung:
    policy = MLModelPolicyEngine()
    model_name = policy.select_denoise_model(context, goal)
    plugin = getattr(pipeline, model_name)
    plugin.process(input, output)
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MLModelPolicyEngine:
    """
    Policy-Engine für automatische SOTA-Modellauswahl.

    Wählt basierend auf Audio-Kontext das optimale ML-Modell:
    - Kontext: detected_medium, genre, has_vocals, noise_type, etc.
    - Goal: quality_level, specific requirements
    """

    def __init__(self):
        """Initialisiert Policy-Engine."""
        self.logger = logging.getLogger("MLModelPolicyEngine")
        self.logger.info("Policy-Engine initialisiert")

    def select_denoise_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Denoise/Restoration-Modell basierend auf semantischem Audio-Verständnis.

        Args:
            context: Policy-Kontext mit semantischen Feldern:
                - detected_medium, has_vocals, noise_type
                - dominant_instrument, content_character, processing_strategy (🔬 Innovation #3)
                - preserve_transients, enhance_clarity, reduce_harshness
                - has_drums, has_guitar, has_keys, has_ambient
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name: 'resemble_enhance', 'deepfilternet', 'wpe', 'dccrn', 'banquet'

        Entscheidungslogik (🎯 Semantic-Aware):
        1. Vinyl Medium → Banquet (vinyl-specialized, hat Vorrang vor Quality-Override)
        2. Quality-Override / schlechte SNR → DeepFilterNet (erzwungen)
        3. Vocals/Speech → Resemble Enhance (SOTA for voice)
        4. Drums/Transient-Rich → DCCRN (preserves transients)
        5. Ambient/Sustained → DeepFilterNet (aggressive smoothing)
        6. General → Resemble Enhance (balanced, high quality)
        """
        # === 🔬 SEMANTIC-AWARE MODEL SELECTION (Innovation #3) ===

        # Priority 1: Medium-specific models (Vorrang vor Quality-Override)
        detected_medium = context.get("detected_medium", "unknown")
        if detected_medium == "vinyl":
            self.logger.info("🎵 Vinyl erkannt → Banquet (vinyl-specialized)")
            return "banquet"

        # Priority 2: Quality-Override / schlechte SNR → DeepFilterNet erzwungen
        snr = context.get("snr", None)
        quality_level = goal.get("quality_level", "standard")
        if (
            quality_level == "maximal"
            or detected_medium in ["mp3", "aac", "minidisc"]
            or (snr is not None and snr < 15)
        ):
            self.logger.info("🔴 Schlechte Qualität erkannt → ML-Modell wird erzwungen (deepfilternet)")
            return "deepfilternet"

        # Priority 2: Vocal/Speech content (semantic detection, not F0 heuristic)
        has_vocals = context.get("has_vocals", False)
        if has_vocals:
            self.logger.info("🎤 Vocals/Speech erkannt → Resemble Enhance (voice-optimized)")
            return "resemble_enhance"

        # Priority 3: Transient-rich content (drums, percussion)
        # DCCRN excels at preserving sharp transients while removing noise
        has_drums = context.get("has_drums", False)
        content_character = context.get("content_character", "BALANCED")

        if has_drums or content_character in ["HIGHLY_TRANSIENT", "TRANSIENT"]:
            self.logger.info(f"⚡ Transient-rich content ({content_character}) → DCCRN (transient-preserving)")
            return "dccrn"

        # Priority 4: Sustained/Ambient content
        # DeepFilterNet provides aggressive smoothing without harming sustained tones
        # BUT: Only for true ambient/drone content, not for sustained instruments (piano, strings)
        has_ambient = context.get("has_ambient", False)
        dominant_instrument = context.get("dominant_instrument", "unknown")

        if has_ambient or (content_character == "HIGHLY_SUSTAINED" and dominant_instrument == "AMBIENT"):
            self.logger.info(f"🌊 Ambient content ({content_character}) → DeepFilterNet (aggressive smoothing)")
            return "deepfilternet"

        # Priority 5: Processing strategy override
        # If semantic analysis recommends specific strategy, honor it
        processing_strategy = context.get("processing_strategy", "BALANCED_PROCESSING")

        if processing_strategy == "PRESERVE_TRANSIENTS":
            self.logger.info("🎯 Strategy: PRESERVE_TRANSIENTS → DCCRN")
            return "dccrn"
        elif processing_strategy == "AGGRESSIVE_SMOOTHING":
            self.logger.info("🎯 Strategy: AGGRESSIVE_SMOOTHING → DeepFilterNet")
            return "deepfilternet"

        # Priority 6: Dominant instrument guidance
        dominant_instrument = context.get("dominant_instrument", "unknown")

        instrument_model_map = {
            "DRUMS": "dccrn",  # Preserve attack/transients
            "PERCUSSION": "dccrn",  # Preserve attack/transients
            "VOCALS": "resemble_enhance",  # Voice optimization
            "SPEECH": "resemble_enhance",  # Voice optimization
            "GUITAR": "resemble_enhance",  # Balanced, preserves harmonics
            "BASS": "deepfilternet",  # Low-freq optimization
            "KEYS": "resemble_enhance",  # Balanced
            "SYNTH": "deepfilternet",  # Broadband, synthetic
            "AMBIENT": "deepfilternet",  # Aggressive smoothing OK
            "STRINGS": "resemble_enhance",  # Balanced, preserve harmonics
            "BRASS": "dccrn",  # Preserve attack
        }

        if dominant_instrument in instrument_model_map:
            selected = instrument_model_map[dominant_instrument]
            self.logger.info(f"🎼 Dominant: {dominant_instrument} → {selected}")
            return selected

        # Fallback: Resemble Enhance (general-purpose SOTA)
        self.logger.info("🎯 Balanced content → Resemble Enhance (general-purpose SOTA)")
        return "resemble_enhance"

    def select_repair_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Repair/Declipping-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name: 'dccrn', 'fullsubnet'

        Entscheidungslogik:
        - Speech → fullsubnet (ONNX-optimiert, SOTA für Speech)
        - Music → dccrn (Complex Network für Music)
        """
        # Speech-optimiert
        if context.get("has_vocals", False):
            self.logger.info("Speech erkannt → FullSubNet+ (Speech-optimiert)")
            return "fullsubnet"

        # Music-optimiert
        self.logger.info("Music → DCCRN (Music-optimiert)")
        return "dccrn"

    def select_stem_separation_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Stem Separation-Modell.

        Args:
            context: Audio-Kontext
            goal: Separation-Ziel (vocals, instruments, stems)

        Returns:
            Plugin-Name: 'mdx23c', 'demucs', 'uvr_mdxnet', 'convtasnet'

        Entscheidungslogik:
        - Vocals/Instrument Split → MDX23C (SOTA quality)
        - 6-Stem Separation → Demucs v4 (most stems)
        - HQ Mastering → UVR MDX-Net HQ4 (best quality)
        - Speech Separation → Conv-TasNet (speech-optimized)
        """
        num_stems = goal.get("num_stems", 2)
        quality_level = goal.get("quality_level", "high")

        # Speech Separation
        if context.get("has_vocals", False) and num_stems == 2:
            self.logger.info("Speech Separation → Conv-TasNet (speech-optimized)")
            return "convtasnet"

        # Fast processing
        if quality_level == "fast":
            self.logger.info("Fast stem separation → MDX23C (SOTA, schnell)")
            return "mdx23c"

        # Ultra-HQ Mastering
        if quality_level == "ultra":
            self.logger.info("Ultra-HQ mastering → UVR MDX-Net HQ4 (best quality)")
            return "uvr_mdxnet"

        # 6-Stem Separation
        if num_stems >= 6:
            self.logger.info("6-Stem separation → Demucs v4 (most stems)")
            return "demucs"

        # Default: MDX23C (SOTA)
        self.logger.info("Standard separation → MDX23C (SOTA)")
        return "mdx23c"

    def select_enhancement_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Enhancement/Upsampling-Modell.

        Args:
            context: Audio-Kontext
            goal: Enhancement-Ziel

        Returns:
            Plugin-Name: 'resemble_enhance', 'audiosr', 'wpe', 'gacela'

        Entscheidungslogik:
        - Speech → Resemble Enhance (voice clarity)
        - Super-Resolution → AudioSR (16/24 kHz → 48 kHz)
        - Diffusion Enhancement → WPE Dereverberation (Nakatani 2010)
        - General Enhancement → GACELA (audio enhancement)
        """
        enhancement_type = goal.get("enhancement_type", "general")

        # Speech Enhancement
        if context.get("has_vocals", False) or enhancement_type == "speech":
            self.logger.info("Speech Enhancement → Resemble Enhance (voice clarity)")
            return "resemble_enhance"

        # Super-Resolution (Upsampling)
        if enhancement_type == "super_resolution":
            self.logger.info("Super-Resolution → AudioSR (16/24 kHz → 48 kHz)")
            return "audiosr"

        # Diffusion-based Enhancement
        if enhancement_type == "diffusion":
            self.logger.info("Diffusion Enhancement → WPE Dereverberation (Nakatani 2010)")
            return "wpe"

        # General Enhancement
        self.logger.info("General Enhancement → GACELA")
        return "gacela"

    def select_quality_assessment_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> List[str]:
        """
        Wählt Quality Assessment-Modelle für Musikqualität.

        Args:
            context: Audio-Kontext
            goal: Assessment-Ziel

        Returns:
            List[Plugin-Name]: ['cdpam', 'visqol', 'peaq', 'fad']

        Entscheidungslogik (§4.4/§10.2 — Verbotene Metriken berücksichtigen):
        - Immer: CDPAM (Musik-Wahrnehmungsqualität ohne Referenz)
        - Mit Referenz: +ViSQOL v3 (zwingend --audio Mode, kein Speech-Default)
        - Vollständig: CDPAM + ViSQOL + PEAQ + FAD (erweiterte Metriken, nur Reporting)

        VERBOTEN (§4.4/§10.2/§11.3):
        - DNSMOS: trainiert auf 16 kHz DNS-Challenge-Sprachkorpus, bewertet Musik falsch
        - NISQA: Sprachqualitäts-CNN, keine Musik-Trainingsdaten
        - PESQ: Telefonband 300–3400 Hz, strukturell ungeeignet für Vollband-Musik
        - STOI: Sprachverständlichkeit, sinnlos für Instrumentalmusik
        """
        has_reference = goal.get("has_reference", False)
        assessment_type = goal.get("assessment_type", "full")

        # Basis: CDPAM (musik-spezifisch, keine Referenz nötig)
        models = ["cdpam"]

        # Mit Referenz: ViSQOL v3 (--audio Mode zwingend per Spec)
        if has_reference:
            models.append("visqol")

        # Vollständig: Erweiterte Metriken (nur für Reporting, kein Quality-Gate)
        if assessment_type == "full" and has_reference:
            models = ["cdpam", "visqol", "peaq", "fad"]

        self.logger.info(f"Quality Assessment (Musik-Metriken §4.4) → {models}")
        return models

    def select_vocoder_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Vocoder/Synthesis-Modell.

        Args:
            context: Audio-Kontext
            goal: Synthesis-Ziel

        Returns:
            Plugin-Name: 'hifigan', 'diffwave'

        Entscheidungslogik:
        - Fast/Real-time → HiFi-GAN (GAN-based, fast)
        - High-Quality → DiffWave (Diffusion-based, slower but better)
        """
        quality_level = goal.get("quality_level", "high")

        # Vocos ist Primär-Vocoder für alle Qualitätsstufen (§4.5 Spec):
        # Vocos 0.1.0 ist 8× schneller als BigVGAN-v2 auf CPU und erzeugt höhere MOS.
        # Fallback-Kaskade (im Plugin selbst): Vocos ONNX → PyPI → HiFi-GAN → PGHI-ISTFT
        # DiffWave ist für Inpainting (Dropout-Lücken), kein Vocoder.
        if quality_level == "fast":
            self.logger.info("Fast Vocoding → Vocos (ConvNeXt-iSTFT, 8× schneller als BigVGAN-v2)")
            return "vocos"
        else:
            self.logger.info("High-Quality Vocoding → Vocos 0.1.0 (Primär-Vocoder §4.5)")
            return "vocos"

    def select_audio_tagging_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt Audio Tagging/Classification-Modell.

        Args:
            context: Audio-Kontext
            goal: Tagging-Ziel

        Returns:
            Plugin-Name: 'panns' (527 AudioSet classes)
        """
        self.logger.info("Audio Tagging → PANNS (527 AudioSet classes)")
        return "panns"

    def select_mastering_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt Automated Mastering-Modell.

        Args:
            context: Audio-Kontext
            goal: Mastering-Ziel (with reference track)

        Returns:
            Plugin-Name: 'matchering'
        """
        self.logger.info("Automated Mastering → Matchering 2.0 (reference-based)")
        return "matchering"

    def select_generative_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt Generative Audio-Modell.

        Args:
            context: Audio-Kontext
            goal: Generation-Ziel

        Returns:
            Plugin-Name: 'audioldm2', 'vampnet'

        Entscheidungslogik:
        - Text-to-Audio → AudioLDM2 (text prompt)
        - Music Generation → VampNet (generative music)
        """
        generation_type = goal.get("generation_type", "text_to_audio")

        if generation_type == "music":
            self.logger.info("Music Generation → VampNet")
            return "vampnet"
        else:
            self.logger.info("Text-to-Audio → AudioLDM2 (text prompt)")
            return "audioldm2"

    def select_pitch_detection_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt Pitch Detection-Modell.

        Args:
            context: Audio-Kontext
            goal: Pitch detection-Ziel

        Returns:
            Plugin-Name: 'crepe' (monophonic pitch tracking)
        """
        model_size = goal.get("model_size", "large")
        self.logger.info(f"Pitch Detection → CREPE ({model_size} model)")
        return "crepe"

    def select_medium_specific_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt Medium-spezifisches Restoration-Modell.

        Args:
            context: Audio-Kontext
            goal: Medium-specific restoration

        Returns:
            Plugin-Name: 'banquet' (vinyl-specialized)
        """
        detected_medium = context.get("detected_medium", "unknown")

        if detected_medium == "vinyl":
            self.logger.info("Vinyl Restoration → Banquet (vinyl-specialized)")
            return "banquet"
        else:
            self.logger.info(f"No medium-specific model for {detected_medium}, using general denoise model")
            return self.select_denoise_model(context, goal)

    def select_all_models(self, context: Dict[str, Any], tasks: List[str]) -> Dict[str, Any]:
        """
        Intelligente Auswahl aller benötigten Modelle basierend auf Tasks.

        Args:
            context: Audio-Kontext
            tasks: Liste von Tasks ['denoise', 'separation', 'enhancement', 'quality']

        Returns:
            Dict mit gewählten Modellen pro Task
        """
        selected_models: Dict[str, Any] = {}

        for task in tasks:
            if task == "denoise":
                selected_models["denoise"] = self.select_denoise_model(context, {})
            elif task == "repair":
                selected_models["repair"] = self.select_repair_model(context, {})
            elif task == "separation":
                selected_models["separation"] = self.select_stem_separation_model(context, {})
            elif task == "enhancement":
                selected_models["enhancement"] = self.select_enhancement_model(context, {})
            elif task == "quality":
                selected_models["quality"] = self.select_quality_assessment_model(context, {})
            elif task == "vocoder":
                selected_models["vocoder"] = self.select_vocoder_model(context, {})
            elif task == "tagging":
                selected_models["tagging"] = self.select_audio_tagging_model(context, {})
            elif task == "mastering":
                selected_models["mastering"] = self.select_mastering_model(context, {})
            elif task == "generation":
                selected_models["generation"] = self.select_generative_model(context, {})
            elif task == "pitch":
                selected_models["pitch"] = self.select_pitch_detection_model(context, {})
            elif task == "medium_specific":
                selected_models["medium_specific"] = self.select_medium_specific_model(context, {})

        self.logger.info(f"Selected models for tasks {tasks}: {selected_models}")
        return selected_models

    def select_separation_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Source-Separation-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name: 'demucs', 'mdx23c', 'uvr_mdxnet'

        Entscheidungslogik:
        - Maximal Quality → mdx23c (SOTA)
        - 4+ Stems needed → demucs (6-Stem: vocals, drums, bass, other, piano, guitar)
        - Fast 2-Stem → uvr_mdxnet (HQ4 für Qualität, HQ1 für Speed)
        """
        # Priorität 1: Maximal Quality explizit gewünscht
        if goal.get("quality_level") == "maximal":
            self.logger.info("Maximal Quality → MDX23C (SOTA)")
            return "mdx23c"

        # Priorität 2: Viele Stems benötigt
        if context.get("stem_count", 2) >= 4 or goal.get("stems", 2) >= 4:
            self.logger.info("4+ Stems benötigt → Demucs v4 (6-Stem)")
            return "demucs"

        # Priorität 3: Fast 2-Stem (vocals/accompaniment)
        if context.get("stem_count", 2) == 2:
            # HQ4 für Qualität, HQ1 für Speed
            model = "HQ4" if goal.get("quality_level") in ["high", "maximal"] else "HQ1"
            self.logger.info(f"2-Stem → UVR MDX-Net ({model})")
            return "uvr_mdxnet"

        # Standard: Demucs (vielseitig, gute Qualität)
        self.logger.info("Standard → Demucs v4 (4-Stem)")
        return "demucs"

    def select_enhancement_model_alt(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes allgemeines Enhancement-Modell.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Plugin-Name für allgemeines Enhancement

        Entscheidungslogik:
        - Speech → resemble_enhance (SOTA Speech Enhancement)
        - Classical → wpe (Dereverberation)
        - General → waveunet oder gacela
        """
        if context.get("has_vocals", False):
            return "resemble_enhance"
        elif context.get("genre") in ["classical", "jazz"]:
            return "wpe"
        else:
            return "gacela"  # oder 'waveunet'

    def select_super_resolution_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:
        """
        Wählt bestes Super-Resolution-Modell.

        Returns:
            'audiosr' - Aktuell nur ein SOTA-Modell verfügbar
        """
        self.logger.info("Super-Resolution → AudioSR (Diffusion-basiert, 48kHz)")
        return "audiosr"

    def select_mastering_model(self, context: Dict[str, Any], goal: Dict[str, Any]) -> str:  # noqa: F811
        """
        Wählt bestes Mastering-Modell.

        Returns:
            'matchering' - Aktuell nur ein SOTA-Modell verfügbar
        """
        self.logger.info("Mastering → Matchering 2.0")
        return "matchering"

    def select_quality_metrics(self, context: Dict[str, Any], goal: Dict[str, Any]) -> list:
        """
        Wählt relevante Quality-Metriken.

        Args:
            context: Audio-Kontext
            goal: Restaurierungs-Ziel

        Returns:
            Liste von Metrik-Plugin-Namen

        Metrik-Auswahl:
        - Speech → dnsmos, nisqa (non-intrusive)
        - Maximal Quality → alle Metriken (dnsmos, nisqa, pesq, visqol, cdpam)
        - Standard → dnsmos, nisqa
        """
        metrics = []  # Initialisierung
        if context.get("has_vocals", False):
            metrics.extend(["pesq", "visqol", "cdpam"])
        elif context.get("genre") == "classical":
            metrics.append("pesq")
        # Bei maximal quality: Alle Metriken
        if goal.get("quality_level") == "maximal":
            metrics.extend(["pesq", "visqol", "cdpam"])
            self.logger.info("Maximal Quality → ALLE Metriken (5)")

        # Bei Speech: PESQ hinzufügen
        elif context.get("has_vocals", False):
            metrics.append("pesq")
            self.logger.info("Speech → DNSMOS + NISQA + PESQ")

        else:
            self.logger.info("Standard → DNSMOS + NISQA")

        return metrics

    def should_use_diffusion_models(self, context: Dict[str, Any], goal: Dict[str, Any]) -> bool:
        """
        Entscheidet ob Diffusion-basierte Modelle genutzt werden sollen.

        Diffusion-Modelle (WPE, DiffWave, AudioSR, AudioLDM2):
        - Sehr hohe Qualität
        - Langsamer (600s Timeout)
        - Besonders gut für Musik/Classical

        Returns:
            True wenn Diffusion-Modelle empfohlen werden
        """
        # Maximal quality explizit gewünscht
        if goal.get("quality_level") == "maximal":
            return True

        # Classical/Jazz/Acoustic genres profitieren besonders
        if context.get("genre") in ["classical", "jazz", "acoustic"]:
            return True

        # High-End Context (hohe Sample Rate, low noise floor)
        if context.get("sample_rate", 0) >= 48000:
            return True

        return False


# ===== CONVENIENCE FUNCTIONS =====


def get_recommended_models(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick helper: Get recommended models for common restoration workflow.

    Args:
        context: Audio context dictionary

    Returns:
        Dict with recommended models:
        - denoise: Best denoise model
        - quality: List of quality assessment models
        - separation: Best stem separation model (if needed)
    """
    policy = MLModelPolicyEngine()

    recommendations = {
        "denoise": policy.select_denoise_model(context, {}),
        "quality": policy.select_quality_assessment_model(context, {"has_reference": False}),
    }

    # Add separation if multi-track content detected
    if context.get("has_vocals", False):
        recommendations["separation"] = policy.select_stem_separation_model(context, {"num_stems": 2})

    return recommendations


# Convenience-Funktionen für direkte Nutzung
def get_optimal_denoise_plugin(context: Dict[str, Any], goal: Dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Denoise-Plugin zurück."""
    engine = MLModelPolicyEngine()
    return engine.select_denoise_model(context, goal)


def get_optimal_separation_plugin(context: Dict[str, Any], goal: Dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Separation-Plugin zurück."""
    engine = MLModelPolicyEngine()
    return engine.select_stem_separation_model(context, goal)


def get_optimal_repair_plugin(context: Dict[str, Any], goal: Dict[str, Any]) -> str:
    """Shortcut: Gibt bestes Repair-Plugin zurück."""
    engine = MLModelPolicyEngine()
    return engine.select_repair_model(context, goal)


if __name__ == "__main__":
    # Test-Beispiele
    logging.basicConfig(level=logging.INFO)

    engine = MLModelPolicyEngine()

    # Test 1: Speech Restoration
    context1: Dict[str, Any] = {"has_vocals": True, "noise_type": "broadband", "genre": "speech"}
    goal1 = {"quality_level": "high"}
    print(f"Speech Restoration: {engine.select_denoise_model(context1, goal1)}")

    # Test 2: Vinyl Restoration
    context2: Dict[str, Any] = {"detected_medium": "vinyl", "genre": "jazz"}
    goal2 = {"quality_level": "maximal"}
    print(f"Vinyl Restoration: {engine.select_denoise_model(context2, goal2)}")

    # Test 3: Classical Music Enhancement
    context3: Dict[str, Any] = {"genre": "classical", "has_vocals": False}
    goal3 = {"quality_level": "maximal"}
    print(f"Classical Enhancement: {engine.select_denoise_model(context3, goal3)}")

    # Test 4: Source Separation
    context4: Dict[str, Any] = {"stem_count": 4}
    goal4 = {"quality_level": "maximal"}
    print(f"Source Separation: {engine.select_stem_separation_model(context4, goal4)}")
