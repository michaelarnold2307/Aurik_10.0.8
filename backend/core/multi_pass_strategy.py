"""
Multi-Pass Processing Strategy & Adaptive Selection

Implements GAP #1: Multi-Pass Strategy für vollautomatisches Audio-Processing.
Ein weltklasse-automatisches System sollte mehrere Ansätze testen und automatisch
den besten wählen basierend auf objektiven Metriken.

Architecture:
1. ProcessingVariant - Beschreibt eine Processing-Strategie mit Parametern
2. ObjectiveScorer - Bewertet Resultate via CDPAM, DNSMOS, Musical Goals
3. MultiPassEngine - Führt Audio durch mehrere Varianten, wählt beste
4. ConfidenceCalculator - Berechnet Confidence basierend auf variance

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

from dataclasses import asdict, dataclass
from enum import Enum
import logging
from typing import Any, Callable

import numpy as np

from backend.core.intrinsic_audio_quality_scorer import IntrinsicAudioQualityScorer
from backend.core.processing_modes import ProcessingConfig, ProcessingMode

logger = logging.getLogger(__name__)

# Singleton IAQS-Instanz (kein State, daher Thread-safe)
_iaqs = IntrinsicAudioQualityScorer()


class VariantStrategy(Enum):
    """Strategien für ProcessingVarianten."""

    CONSERVATIVE = "conservative"
    """Conservative: Minimal processing, max preservation."""

    BALANCED = "balanced"
    """Balanced: Moderate processing, good quality/authenticity trade-off."""

    AGGRESSIVE = "aggressive"
    """Aggressive: Strong processing, maximum quality improvement."""

    GENTLE_DENOISE = "gentle_denoise"
    """Focus: Ultra-gentle denoising, preserve everything else."""

    STRONG_DYNAMICS = "strong_dynamics"
    """Focus: Strong dynamics control, moderate other processing."""

    NATURALNESS_FIRST = "naturalness_first"
    """Focus: Maximale Natürlichkeit — minimales Processing, volle Authentizität."""


@dataclass
class ProcessingVariant:
    """
    Beschreibt eine Processing-Variante mit spezifischen Parametern.

    Jede Variante ist eine andere "Strategie" das Material zu restaurieren.
    Multi-Pass Engine testet 3-5 Varianten und wählt die beste.
    """

    name: str
    """Eindeutiger Name (z.B. 'conservative', 'balanced', 'aggressive')."""

    strategy: VariantStrategy
    """High-level Strategie-Typ."""

    config: ProcessingConfig
    """Konkrete Processing-Parameter."""

    description: str = ""
    """Optionale Beschreibung der Variante."""

    weight: float = 1.0
    """Gewichtung im Scoring (default: 1.0 = equal weight)."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary für Serialisierung."""
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "config": asdict(self.config),
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def create_conservative(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Conservative Variante: Minimal processing.

        Gut für:
        - Material mit wenig Defekten
        - Maximale Authentizität gefragt
        - Archival/Forensic use cases
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Ultra-conservative adjustments
        config.denoise_strength *= 0.5
        config.declip_strength *= 0.7
        config.click_removal_sensitivity *= 0.6
        config.compression_ratio = min(config.compression_ratio, 2.5)
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="conservative",
            strategy=VariantStrategy.CONSERVATIVE,
            config=config,
            description="Minimal processing, maximum preservation",
            weight=1.0,
        )

    @classmethod
    def create_balanced(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Balanced Variante: Standard processing.

        Gut für:
        - Typisches Material
        - Balance zwischen Quality und Authenticity
        - Default Ansatz
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Balanced = use base config as-is

        return cls(
            name="balanced",
            strategy=VariantStrategy.BALANCED,
            config=config,
            description="Balanced processing, good quality/authenticity trade-off",
            weight=1.0,
        )

    @classmethod
    def create_aggressive(cls, base_mode: ProcessingMode = ProcessingMode.STUDIO_2026) -> "ProcessingVariant":
        """
        Erstelle Aggressive Variante: Strong processing.

        Gut für:
        - Stark defektes Material
        - Maximum Quality improvement gefragt
        - Competitive sound
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Aggressive adjustments
        config.denoise_strength *= 1.3
        config.declip_strength *= 1.2
        config.click_removal_sensitivity *= 1.2
        config.compression_ratio *= 1.3
        config.enhancement_strength = min(config.enhancement_strength * 1.3, 1.0)

        # Clamp to valid ranges
        config.denoise_strength = min(config.denoise_strength, 1.0)
        config.declip_strength = min(config.declip_strength, 1.0)
        config.click_removal_sensitivity = min(config.click_removal_sensitivity, 1.0)

        return cls(
            name="aggressive",
            strategy=VariantStrategy.AGGRESSIVE,
            config=config,
            description="Strong processing, maximum quality improvement",
            weight=1.0,
        )

    @classmethod
    def create_gentle_denoise(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Gentle Denoise Variante: Ultra-gentle noise reduction only.

        Gut für:
        - Leichtes Background Noise
        - Preservation kritisch
        - Vintage recordings
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Focus on gentle denoising
        config.denoise_strength = 0.15  # Ultra-gentle
        config.declip_strength = 0.3  # Minimal
        config.click_removal_sensitivity = 0.3  # Minimal
        config.compression_ratio = 1.5  # Very light compression
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="gentle_denoise",
            strategy=VariantStrategy.GENTLE_DENOISE,
            config=config,
            description="Ultra-gentle denoising, preserve everything else",
            weight=1.0,
        )

    @classmethod
    def create_strong_dynamics(cls, base_mode: ProcessingMode = ProcessingMode.STUDIO_2026) -> "ProcessingVariant":
        """
        Erstelle Strong Dynamics Variante: Focus on dynamics control.

        Gut für:
        - Extreme dynamic range
        - Competitive loudness
        - Modern streaming
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Focus on dynamics
        config.compression_ratio = 6.0  # Strong compression
        config.compression_threshold_db = -30.0  # Lower threshold
        config.target_lufs = -14.0  # Streaming standard
        config.denoise_strength = 0.4  # Moderate
        config.preserve_breaths = False  # Allow breath reduction for loudness

        return cls(
            name="strong_dynamics",
            strategy=VariantStrategy.STRONG_DYNAMICS,
            config=config,
            description="Strong dynamics control, moderate other processing",
            weight=1.0,
        )

    @classmethod
    def create_naturalness_first(cls, base_mode: ProcessingMode = ProcessingMode.RESTORATION) -> "ProcessingVariant":
        """
        Erstelle Naturalness-First Variante: Maximale Natürlichkeit.

        Gut für:
        - Vintage-Aufnahmen mit Charakter
        - Archivierung (originaler Charakter erhalten)
        - Material wo Over-Processing schlimmer wäre als Rest-Artefakte

        Strategie:
        - Enhancement auf 0% (keine künstlichen Verbesserungen)
        - Nur defekte Stellen korrigieren (Klicks, Clips) — sehr sanft
        - Raum-Ton, Atem, Analogcharakter vollständig erhalten
        - Kompression fast abgeschaltet (natürliche Dynamik)
        """
        import copy

        from backend.core.processing_modes import get_processing_config

        config = copy.deepcopy(get_processing_config(base_mode))

        # Naturalness: so wenig Eingriff wie möglich
        config.denoise_strength = 0.08  # Fast kein Rauschen entfernen
        config.declip_strength = 0.25  # Nur harte Clips korrigieren
        config.click_removal_sensitivity = 0.20  # Nur deutliche Klicks
        config.compression_ratio = 1.2  # Annähernd keine Kompression
        config.enhancement_strength = 0.0  # Keine künstliche Verbesserung
        config.preserve_breaths = True
        config.preserve_room_tone = True
        config.preserve_analog_character = True

        return cls(
            name="naturalness_first",
            strategy=VariantStrategy.NATURALNESS_FIRST,
            config=config,
            description="Maximale Natürlichkeit: minimales Processing, volle Authentizität",
            weight=1.05,  # Leicht bevorzugt bei RESTORATION
        )


@dataclass
class ObjectiveScore:
    """
    Objektive Quality Scores für ein verarbeitetes Audio.

    Kombiniert:
    - CDPAM (Perceptual Similarity, <0.3 = sehr ähnlich zu Reference)
    - DNSMOS (Speech Quality, 3.5-5.0 = sehr gut)
    - Musical Goals (0.0-1.0, >0.7 = excellent)
    - Signal Stats (SNR, THD, etc.)
    """

    # === Perceptual Metrics ===
    cdpam_score: float = 0.0
    """CDPAM Score (0.0-1.0, lower=better, <0.3=excellent)."""

    dnsmos_score: float = 0.0
    """DNSMOS P.835 (1.0-5.0, higher=better, >3.5=good)."""

    # === Musical Goals ===
    musical_goals_avg: float = 0.0
    """Average Musical Goals Score (0.0-1.0, higher=better)."""

    musical_goals_min: float = 0.0
    """Minimum Musical Goals Score (detectiert worst-case goal)."""

    # === Signal Statistics ===
    snr_db: float = 0.0
    """Signal-to-Noise Ratio in dB (higher=better)."""

    thd_percent: float = 0.0
    """Total Harmonic Distortion in % (lower=better)."""

    # === IAQS Holistic Score ===
    iaqs_total: float = 0.0
    """Intrinsic Audio Quality Score gesamt (0.0-1.0, higher=better)."""

    # === Flags: welche Metriken tatsächlich berechnet wurden ===
    iaqs_active: bool = False
    """True wenn IAQS tatsächlich berechnet (nicht Default-Wert)."""

    cdpam_active: bool = False
    """True wenn CDPAM tatsächlich berechnet (nicht Default-Wert)."""

    dnsmos_active: bool = False
    """True wenn DNSMOS tatsächlich berechnet (nicht Default-Wert)."""

    # === Composite Scores ===
    composite_score: float = 0.0
    """Weighted composite score (0.0-1.0, higher=better)."""

    confidence: float = 0.0
    """Confidence in this score (0.0-1.0, based on consistency)."""

    # === Metadata ===
    variant_name: str = ""
    """Name der Variante die diesen Score produziert hat."""

    processing_time_sec: float = 0.0
    """Processing time in Sekunden."""

    def to_dict(self) -> dict[str, Any]:
        """Konvertiere zu Dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ObjectiveScore(variant='{self.variant_name}', "
            f"composite={self.composite_score:.3f}, "
            f"confidence={self.confidence:.2f}, "
            f"CDPAM={self.cdpam_score:.3f}, "
            f"DNSMOS={self.dnsmos_score:.2f}, "
            f"MG_avg={self.musical_goals_avg:.2f}, "
            f"SNR={self.snr_db:.1f}dB)"
        )


class ObjectiveScorer:
    """
    Bewertet Audio via objektive Metriken.

    Integriert:
    - CDPAM Plugin (wenn availableund Reference vorhanden)
    - DNSMOS Plugin (wenn available)
    - Musical Goals Checker
    - Enhanced Metrics (SNR, THD, etc.)
    """

    def __init__(self, enable_cdpam: bool = True, enable_dnsmos: bool = False, enable_musical_goals: bool = True):
        """
        Initialize ObjectiveScorer.

        Args:
            enable_cdpam: Enable CDPAM scoring (requires reference)
            enable_dnsmos: PERMANENT FALSE — DNSMOS P.835 ist auf Sprachkorpus trainiert
                           (16 kHz DNS-Challenge) und ist VERBOTEN als Musik-Metrik (§10.2).
                           Parameter bleibt aus Rückwärtskompatibilität erhalten, ist aber wirkungslos.
            enable_musical_goals: Enable Musical Goals scoring
        """
        self.enable_cdpam = enable_cdpam
        self.enable_dnsmos = enable_dnsmos
        self.enable_musical_goals = enable_musical_goals

        # Try to import plugins (may fail if not installed)
        # §4.4: VERSA 2024 ersetzt CDPAM als non-reference MOS-Metrik
        self.versa_plugin = None
        self.dnsmos_plugin = None

        if enable_cdpam:
            try:
                from plugins.versa_plugin import get_versa_plugin  # noqa: PLC0415

                self.versa_plugin = get_versa_plugin()
                logger.info("✓ VERSA Plugin loaded (§4.4, non-reference MOS)")
            except Exception as e:
                logger.warning(f"⚠ VERSA Plugin not available: {e}")

        # §10.2: DNSMOS-Plugin wird nicht geladen — DNSMOS P.835 ist auf Sprachkorpus
        # trainiert (16 kHz DNS-Challenge) und ist VERBOTEN als Musik-Qualitätsmetrik.

    def score(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variant_name: str,
        reference_audio: np.ndarray | None = None,
        processing_time_sec: float = 0.0,
    ) -> ObjectiveScore:
        """
        Score audio via objektive Metriken.

        Args:
            audio: Processed audio array
            sample_rate: Sample rate
            variant_name: Name der Variante
            reference_audio: Optional reference für CDPAM
            processing_time_sec: Processing time

        Returns:
            ObjectiveScore mit allen Metriken
        """
        score = ObjectiveScore(variant_name=variant_name, processing_time_sec=processing_time_sec)

        # === 1. VERSA: non-reference MOS (§4.4 CDPAM-Nachfolger) ===
        if self.enable_cdpam and self.versa_plugin is not None:
            try:
                import numpy as _np  # noqa: PLC0415

                proc_arr = _np.asarray(audio, dtype=_np.float32)
                if proc_arr.ndim == 2:
                    proc_arr = proc_arr.mean(axis=1)
                proc_arr = _np.nan_to_num(proc_arr, nan=0.0, posinf=0.0, neginf=0.0)
                versa_result = self.versa_plugin.score(proc_arr, sample_rate)
                # MOS [1,5] → [0,1] skaliert (Feld heißt weiterhin cdpam_score für Compat)
                score.cdpam_score = float(_np.clip((versa_result.mos - 1.0) / 4.0, 0.0, 1.0))
                score.cdpam_active = True

            except Exception as e:
                logger.warning(f"VERSA scoring failed: {e}")
                score.cdpam_score = 0.5  # Neutral default, cdpam_active bleibt False

        # §10.2: DNSMOS-Berechnung deaktiviert — Sprach-Metrik verboten für Musikrestaurierung.
        # score.dnsmos_active bleibt False; score.dnsmos_score bleibt 0.0.

        # === 3. Musical Goals ===
        if self.enable_musical_goals:
            _mg_loaded = False
            try:
                from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

                checker = MusicalGoalsChecker()
                goals_scores = checker.measure_all(audio, sample_rate)

                if goals_scores:
                    values = list(goals_scores.values())
                    score.musical_goals_avg = float(np.mean(values))
                    score.musical_goals_min = float(np.min(values))
                    _mg_loaded = True

            except Exception as e:
                logger.debug(f"MusicalGoalsChecker nicht verfügbar ({e}) — IAQS-Fallback")

            if not _mg_loaded:
                # Intrinsischer Fallback: IAQS liefert psychoakustisch fundierte Scores
                try:
                    iaqs = _iaqs.score(audio, sample_rate)
                    # harmonicity + bark_balance + spectral_regularity ≈ musikalische Güte
                    mg_approx = (iaqs.harmonicity + iaqs.bark_balance + iaqs.spectral_regularity) / 3.0
                    score.musical_goals_avg = float(np.clip(mg_approx, 0.0, 1.0))
                    score.musical_goals_min = float(min(iaqs.harmonicity, iaqs.bark_balance, iaqs.spectral_regularity))
                    logger.debug(f"IAQS Musical-Goals-Fallback: avg={score.musical_goals_avg:.3f}")
                except Exception as e2:
                    logger.warning(f"IAQS-Fallback fehlgeschlagen: {e2}")
                    score.musical_goals_avg = 0.5
                    score.musical_goals_min = 0.5

        # === 4. Signal Statistics + IAQS Holistic (primär, keine externen Deps) ===
        try:
            iaqs_stats = _iaqs.score(audio, sample_rate)
            score.snr_db = float(iaqs_stats.snr_estimate)
            score.thd_percent = float(iaqs_stats.thd_estimate_pct)
            # IAQS gesamt Score direkt aus bereits berechnetem Objekt (kein Doppel-Call)
            score.iaqs_total = float(np.clip(iaqs_stats.overall, 0.0, 1.0))
            score.iaqs_active = True
            logger.debug(
                f"IAQS: SNR={score.snr_db:.1f} dB, THD={score.thd_percent:.2f}%, " f"Total={score.iaqs_total:.3f}"
            )
        except Exception as e:
            logger.warning(f"IAQS Signal statistics failed: {e}")
            # Letzter Fallback: EnhancedMetrics Backend
            try:
                from backend.metrics.enhanced_metrics import EnhancedMetrics

                metrics = EnhancedMetrics()
                try:
                    score.snr_db = metrics.compute_snr(audio, sample_rate)
                except Exception:
                    score.snr_db = 20.0
                try:
                    score.thd_percent = metrics.compute_thd(audio, sample_rate)
                except Exception:
                    score.thd_percent = 1.0
            except Exception as e2:
                logger.warning(f"EnhancedMetrics auch nicht verfügbar: {e2}")
                score.snr_db = 20.0
                score.thd_percent = 1.0

        # === 5. Composite Score (weighted) ===
        score.composite_score = self._calculate_composite(score)

        # === 6. Confidence (placeholder - wird später von ConfidenceCalculator gesetzt) ===
        score.confidence = 0.8  # Default medium confidence

        return score

    def _calculate_composite(self, score: ObjectiveScore) -> float:
        """
        Berechne weighted composite score aus allen Metriken.

        Adaptive Gewichtung:
        - IAQS-Gesamt (immer): 35%
        - Musical Goals Average: 25%
        - DNSMOS (normalized, wenn aktiv): 20% — sonst 0%
        - CDPAM (inverted, wenn aktiv): 10% — sonst 0%
        - SNR (normalized): 7%
        - THD (inverted, normalized): 3%

        Wenn DNSMOS/CDPAM deaktiviert: ihr Gewicht wird
        anteilig auf IAQS und Musical Goals umverteilt,
        damit die Summe immer 100% ergibt.

        Result: 0.0-1.0, higher=better
        """
        # Basis: immer verfügbare Metriken
        base_iaqs = 0.35
        base_mg = 0.25
        base_snr = 0.07
        base_thd = 0.03
        # §10.2: pool_dnsmos entfernt — DNSMOS P.835 verboten als Musik-Qualitätsmetrik
        # Das ehemals 20%-Gewicht wird vollständig auf CDPAM-Pool (10%) und freie
        # Umverteilung (restliche 10%) aufgeteilt, damit die Summe 100% ergibt.
        pool_cdpam = 0.30  # erhöht von 0.10 auf 0.30 (kompensiert entfallenes DNSMOS-Gewicht)
        pool_iaqs = base_iaqs  # Umverteilungspool wenn IAQS nicht aktiv

        # Nicht genutzte Pools umverteilen
        freed = 0.0
        if not score.cdpam_active:
            freed += pool_cdpam
        if not score.iaqs_active:
            freed += pool_iaqs

        # Freed weight → 60% auf Musical Goals, 30% auf SNR, 10% auf THD
        w_iaqs = base_iaqs if score.iaqs_active else 0.0
        w_mg = base_mg + freed * 0.60
        w_snr = base_snr + freed * 0.30
        w_thd = base_thd + freed * 0.10

        composite = 0.0

        # IAQS Gesamt (0.0-1.0) — nur wenn aktiv
        if score.iaqs_active:
            composite += w_iaqs * score.iaqs_total

        # Musical Goals (0.0-1.0)
        composite += w_mg * score.musical_goals_avg

        # §10.2: DNSMOS-Block entfernt — Sprach-Metrik verboten für Musikrestaurierung

        # CDPAM (0.0-1.0, lower=better → invert) — nur wenn aktiv
        if score.cdpam_active:
            cdpam_norm = 1.0 - min(score.cdpam_score, 1.0)
            composite += pool_cdpam * cdpam_norm

        # SNR (typ. 10-40 dB → 0.0-1.0) — Gewicht inkl. umverteilter Anteile
        snr_norm = min(max((score.snr_db - 10.0) / 30.0, 0.0), 1.0)
        composite += w_snr * snr_norm

        # THD (0-10% → 0.0-1.0, lower=better → invert)
        thd_norm = 1.0 - min(score.thd_percent / 10.0, 1.0)
        composite += w_thd * thd_norm

        return min(max(composite, 0.0), 1.0)


class ConfidenceCalculator:
    """
    Berechnet Confidence für Multi-Pass Selection.

    Confidence basiert auf:
    - Variance zwischen top-2 Varianten (geringe variance = high confidence)
    - Consistency across Metriken (alle Metriken zeigen gleiche Variante = high)
    - Absolute composite score (sehr hoher Score = high confidence)
    """

    @staticmethod
    def calculate_confidence(scores: list[ObjectiveScore], best_score: ObjectiveScore) -> float:
        """
        Berechne Confidence für best_score Selection.

        Args:
            scores: Liste aller Scores (sorted by composite, descending)
            best_score: Der beste Score

        Returns:
            Confidence (0.0-1.0), 1.0 = very confident, 0.0 = guessing
        """
        if len(scores) < 2:
            return 0.5  # Not enough data for confidence

        # === 1. Top-2 Variance ===
        # Je größer der Gap zwischen #1 und #2, desto confidenter
        top1_composite = scores[0].composite_score
        top2_composite = scores[1].composite_score

        gap = top1_composite - top2_composite

        # Gap > 0.15 → very confident
        # Gap < 0.05 → not confident
        variance_confidence = min(gap / 0.15, 1.0)

        # === 2. Absolute Quality ===
        # Sehr hoher composite score = zusätzliche confidence
        absolute_confidence = best_score.composite_score

        # === 3. Metric Consistency ===
        # How much do individual metrics agree on the ranking?
        # High agreement (best variant leads on MG avg AND worst-case goal) → high consistency.
        mg_lead = scores[0].musical_goals_avg - scores[1].musical_goals_avg
        mg_min_lead = scores[0].musical_goals_min - scores[1].musical_goals_min
        # Map ±0.2-range to [0, 1] with centre 0.5
        consistency = min(max(0.5 + (mg_lead + mg_min_lead) * 2.5, 0.0), 1.0)

        # === Final Confidence (weighted) ===
        confidence = 0.50 * variance_confidence + 0.30 * absolute_confidence + 0.20 * consistency
        # NaN/Inf-Guard (§3.1)
        confidence = np.nan_to_num(confidence, nan=0.5, posinf=1.0, neginf=0.0)
        return float(np.clip(confidence, 0.0, 1.0))


class MultiPassEngine:
    """
    Multi-Pass Processing Engine.

    Führt Audio durch mehrere ProcessingVarianten,
    scored jede Variante via objektive Metriken,
    wählt automatisch die beste.

    Usage:
        engine = MultiPassEngine()
        result = engine.process_with_variants(
            audio, sample_rate,
            variants=[
                ProcessingVariant.create_conservative(),
                ProcessingVariant.create_balanced(),
                ProcessingVariant.create_aggressive()
            ]
        )

        best_audio = result["audio"]
        best_variant_name = result["variant_name"]
        confidence = result["confidence"]
    """

    def __init__(self, scorer: ObjectiveScorer | None = None):
        """
        Initialize MultiPassEngine.

        Args:
            scorer: ObjectiveScorer instance (creates default if None)
        """
        self.scorer = scorer or ObjectiveScorer()
        self.confidence_calc = ConfidenceCalculator()
        self._restorer = None  # gecachte UnifiedRestorerV3-Instanz (einmalig laden)

    def process_with_variants(
        self,
        audio: np.ndarray,
        sample_rate: int,
        variants: list[ProcessingVariant],
        reference_audio: np.ndarray | None = None,
        process_func: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Process audio mit mehreren Varianten, wähle beste.

        Args:
            audio: Input audio array
            sample_rate: Sample rate
            variants: Liste von ProcessingVarianten zu testen
            reference_audio: Optional reference für CDPAM scoring
            process_func: Custom processing function (audio, config) -> processed_audio
                         Falls None, wird UnifiedRestorerV3 verwendet

        Returns:
            Dict mit:
            - "audio": Best processed audio
            - "variant_name": Name der besten Variante
            - "variant_strategy": Strategy enum
            - "confidence": Confidence (0.0-1.0)
            - "composite_score": Composite score
            - "all_scores": Liste aller ObjectiveScores
            - "processing_times": Dict variant_name -> time_sec
        """
        import time

        if len(variants) == 0:
            raise ValueError("Mindestens 1 ProcessingVariant erforderlich")

        logger.info(f"🎯 Multi-Pass Processing: testing {len(variants)} variants...")

        # Default processing function
        if process_func is None:
            process_func = self._default_process_func

        # Process mit jeder Variante
        results = []
        processing_times = {}

        for variant in variants:
            try:
                logger.info(f"  Processing with '{variant.name}' ({variant.strategy.value})...")
                logger.debug(f"[MPASS] Starte Variante '{variant.name}' …", flush=True)

                start_time = time.time()
                processed_audio = process_func(audio, sample_rate, variant.config)
                proc_time = time.time() - start_time
                logger.debug(f"[MPASS] Variante '{variant.name}' fertig in {proc_time:.1f}s", flush=True)

                processing_times[variant.name] = proc_time

                # Score result
                score = self.scorer.score(
                    audio=processed_audio,
                    sample_rate=sample_rate,
                    variant_name=variant.name,
                    reference_audio=reference_audio,
                    processing_time_sec=proc_time,
                )

                results.append({"audio": processed_audio, "variant": variant, "score": score})

                logger.info(f"    → {score}")

            except Exception as e:
                logger.error(f"  ❌ Variant '{variant.name}' failed: {e}")
                continue

        if len(results) == 0:
            raise RuntimeError("Alle Varianten sind fehlgeschlagen")

        # Sort by composite score (descending)
        results.sort(key=lambda x: x["score"].composite_score, reverse=True)

        # Best result
        best = results[0]
        best_variant = best["variant"]
        best_score = best["score"]

        # Calculate confidence
        all_scores = [r["score"] for r in results]
        confidence = self.confidence_calc.calculate_confidence(all_scores, best_score)
        best_score.confidence = confidence

        logger.info(
            f"✅ Best variant: '{best_variant.name}' "
            f"(score={best_score.composite_score:.3f}, confidence={confidence:.2f})"
        )

        return {
            "audio": best["audio"],
            "best_audio": best["audio"],  # Alias für Rückwärtskompatibilität
            "success": True,
            "variant_name": best_variant.name,
            "variant_strategy": best_variant.strategy,
            "confidence": confidence,
            "composite_score": best_score.composite_score,
            "best_score": best_score,
            "all_scores": all_scores,
            "processing_times": processing_times,
        }

    def _default_process_func(self, audio: np.ndarray, sample_rate: int, config: ProcessingConfig) -> np.ndarray:
        """
        Default processing function using UnifiedRestorerV3.
        UnifiedRestorerV3 wird einmalig instanziiert und gecacht (ML-Modelle nur 1× laden).

        Args:
            audio: Input audio
            sample_rate: Sample rate
            config: ProcessingConfig

        Returns:
            Processed audio (np.ndarray)
        """

        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3

            # Einmalig laden — alle Varianten nutzen dieselbe Instanz
            if self._restorer is None:
                logger.info("Initialisiere UnifiedRestorerV3 (einmalig)...")
                self._restorer = UnifiedRestorerV3()

            # Für Varianten-Bewertung: max. 10 s Audio (Geschwindigkeit)
            _max_eval = sample_rate * 10
            _eval_audio = audio[:_max_eval] if len(audio) > _max_eval else audio
            logger.debug(f"[MPASS] restore() auf {len(_eval_audio)/sample_rate:.0f}s Excerpt …", flush=True)
            result = self._restorer.restore(
                audio=_eval_audio,
                sample_rate=sample_rate,
                mode="restoration",
            )

            # V3 gibt RestorationResult zurück — audio-Array extrahieren
            if hasattr(result, "audio") and result.audio is not None:
                return result.audio
            return audio

        except Exception as e:
            logger.error(f"Default processing failed: {e}")
            # Fallback: return original
            return audio


def create_default_variants(
    base_mode: ProcessingMode = ProcessingMode.RESTORATION, num_variants: int = 3
) -> list[ProcessingVariant]:
    """
    Erstelle default Varianten für Multi-Pass Processing.

    Args:
        base_mode: Base ProcessingMode
        num_variants: Anzahl Varianten (3 or 5)

    Returns:
        Liste von ProcessingVarianten
    """
    if num_variants == 3:
        # Conservative, Balanced, Aggressive
        return [
            ProcessingVariant.create_conservative(base_mode),
            ProcessingVariant.create_balanced(base_mode),
            ProcessingVariant.create_aggressive(base_mode),
        ]
    elif num_variants == 5:
        # Full spectrum
        return [
            ProcessingVariant.create_conservative(base_mode),
            ProcessingVariant.create_gentle_denoise(base_mode),
            ProcessingVariant.create_balanced(base_mode),
            ProcessingVariant.create_aggressive(base_mode),
            ProcessingVariant.create_strong_dynamics(ProcessingMode.STUDIO_2026),
        ]
    else:
        raise ValueError(f"num_variants muss 3 oder 5 sein, nicht {num_variants}")


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    # Load test audio
    audio, sr = sf.read("test_audio/test_input.wav")

    # Create variants
    variants = create_default_variants(base_mode=ProcessingMode.RESTORATION, num_variants=3)

    # Run Multi-Pass
    engine = MultiPassEngine()
    result = engine.process_with_variants(audio=audio, sample_rate=sr, variants=variants)

    # Save best result
    sf.write("test_output/multipass_best.wav", result["audio"], sr)

    logger.debug(f"\n✅ Best Variant: {result['variant_name']}")
    logger.debug(f"   Composite Score: {result['composite_score']:.3f}")
    logger.debug(f"   Confidence: {result['confidence']:.2f}")
    logger.debug("\n📊 All Scores:")
    for score in result["all_scores"]:
        logger.debug(f"   {score}")
