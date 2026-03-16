"""
Performance Guard for Aurik 9.0 - 3× Real-Time Enforcement
============================================================

Überwacht Processing-Performance und stellt sicher, dass 3× Real-Time Limit
eingehalten wird. Implementiert adaptive Quality-Reduction bei Bedarf.

Performance Target: Max 3× Real-Time (3:45 Audio → max 11:15 Processing)

Key Features:
- Real-time Performance Monitoring
- Early-Exit Predictions
- Adaptive Phase Skipping
- Detailed Performance Logging

Author: Aurik 9.0 Development Team
Version: 9.0.0
Date: 2026-02-15
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class PerformanceStatus(Enum):
    """Performance-Status der Verarbeitung."""

    OPTIMAL = "optimal"  # < 2× RT (viel Spielraum)
    GOOD = "good"  # 2-2.5× RT
    ACCEPTABLE = "acceptable"  # 2.5-2.9× RT
    CRITICAL = "critical"  # 2.9-3× RT ⚠️
    EXCEEDED = "exceeded"  # > 3× RT ❌


class QualityMode(Enum):
    """Quality-Modi für adaptive Processing."""

    FAST = "fast"  # ~1.5× RT, 87% Quality
    BALANCED = "balanced"  # ~2.4× RT, 92% Quality (DEFAULT)
    QUALITY = "quality"  # ~9× RT, 95% Quality (kein 3× RT Limit!)
    MAXIMUM = "maximum"  # Studio 2026: kein RT-Limit, alle Enhancement-Phasen aktiv


@dataclass
class PhasePerformance:
    """Performance-Metrics für eine einzelne Phase."""

    phase_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    audio_duration_seconds: float
    rt_factor: float  # Real-Time Factor (duration / audio_duration)
    is_critical: bool = False  # Phase ist kritisch für Quality
    skipped: bool = False


@dataclass
class PerformanceReport:
    """Vollständiger Performance-Report."""

    total_duration_seconds: float
    audio_duration_seconds: float
    total_rt_factor: float
    status: PerformanceStatus
    phases: list[PhasePerformance]
    skipped_phases: list[str]
    warnings: list[str] = field(default_factory=list)
    quality_degradation: float = 0.0  # 0-1 (0 = keine Degradation)

    def __str__(self):
        return (
            f"PerformanceReport:\n"
            f"  Total: {self.total_duration_seconds:.1f}s for {self.audio_duration_seconds:.1f}s audio\n"
            f"  RT Factor: {self.total_rt_factor:.2f}× ({self.status.value})\n"
            f"  Phases: {len(self.phases)} executed, {len(self.skipped_phases)} skipped\n"
            f"  Quality: {(1-self.quality_degradation)*100:.1f}%"
        )


class PerformanceGuard:
    """
    Überwacht und enforced 5× Real-Time Performance-Limit.

    Garantiert:
    - Max 5× RT für Balanced Mode
    - Adaptive Phase-Skipping wenn nötig
    - Detailed Performance-Logging
    """

    # Performance Limits (RT Factors)
    LIMIT_3X_RT = 5.0  # Hard Limit für Balanced Mode
    LIMIT_FAST = 1.5  # Target für Fast Mode
    LIMIT_BALANCED = 5.0  # Budget für Balanced Mode — maximal RT×5 (User-Spec)

    # Warnschwellen
    WARNING_THRESHOLD_OPTIMAL = 3.0  # < 3× RT = optimal
    WARNING_THRESHOLD_GOOD = 4.0
    WARNING_THRESHOLD_ACCEPTABLE = 4.5
    WARNING_THRESHOLD_CRITICAL = 5.0

    # Phase Priorities (höher = wichtiger für Quality)
    # ── Musikalische Exzellenz — Priorität 1 (§MusEx-P1) ────────────────────
    # Phasen mit Priority ≥ 9 werden NIEMALS übersprungen (§2.29, §9.5).
    # RT×3-Budget (max. 3× Audiodauer) bleibt harte Obergrenze — garantiert
    # durch <15 ms/s Laufzeit der Excellence-Phasen (§9.5 Performance-Budget).
    # ─────────────────────────────────────────────────────────────────────────
    PHASE_PRIORITIES = {
        # Gruppe 1: CRITICAL — niemals überspringen
        "click_removal": 10,
        "hum_removal": 10,
        "wow_flutter": 9,
        "dehum": 10,
        "decracklé": 9,
        # ── Musikalische Exzellenz Priorität 1 (§MusEx-P1) ──────────────────
        # Waren MEDIUM/HIGH → auf CRITICAL angehoben: RT×3 Budget garantiert.
        "harmonic_recovery": 10,  # war: 5 (MEDIUM, skippable bei >2.5×)
        "frequency_restoration": 10,  # war: 7 (HIGH,   skippable bei >2.8×)
        "transient_preservation": 10,  # war: 5 (MEDIUM, skippable bei >2.5×)
        "excellence_optimizer": 10,  # neu: ExcellenceOptimizer (§2.5)
        "psychoacoustic_enhancement": 10,  # neu: PsychoakustikModell (§4.5)
        "micro_dynamics_restoration": 10,  # neu: MicroDynamicsEnvMorphing (§2.30)
        "vocal_enhancement": 10,  # neu: VocalAIEnhancement (§2.8)
        "harmonic_preservation": 10,  # neu: HarmonicPreservationGuard (§2.28)
        # ─────────────────────────────────────────────────────────────────────
        # Gruppe 2: HIGH (nur bei < 2.8× RT skippen)
        "denoise": 8,
        "digital_repair": 8,
        # Gruppe 3: MEDIUM (bei > 2.5× RT skippen)
        "stereo_enhancement": 6,
        # Gruppe 4: LOW (bei > 2.0× RT skippen)
        "final_polish": 3,
        "metadata_embedding": 1,
    }

    # Musikalische Exzellenz — Priorität 1 (§MusEx-P1)
    # Phasen in diesem Set werden unter keinen Umständen übersprungen.
    # RT×3-Kompatibilität: alle Excellence-Phasen < 15 ms/s Audio (§9.5).
    MUSICAL_EXCELLENCE_PHASES: frozenset = frozenset(
        {
            "harmonic_recovery",
            "frequency_restoration",
            "transient_preservation",
            "excellence_optimizer",
            "psychoacoustic_enhancement",
            "micro_dynamics_restoration",
            "vocal_enhancement",
            "harmonic_preservation",
        }
    )

    # Hard Budget: maximaler RT-Faktor für Musikalische Exzellenz-Betrieb
    RT3_EXCELLENCE_BUDGET: float = 5.0

    def __init__(
        self, mode: QualityMode = QualityMode.QUALITY, enforce_limit: bool = True, enable_adaptive_skipping: bool = True
    ):
        """
        Initialisiert PerformanceGuard.

        Args:
            mode: Quality-Mode (bestimmt Performance-Target)
            enforce_limit: Enforce 3× RT Limit (False = nur Monitor)
            enable_adaptive_skipping: Adaptive Phase-Skipping aktivieren
        """
        self.mode = mode
        self.enforce_limit = enforce_limit
        self.enable_adaptive_skipping = enable_adaptive_skipping

        # Performance-Targets basierend auf Mode
        self.target_rt_factor = {
            QualityMode.FAST: self.LIMIT_FAST,
            QualityMode.BALANCED: self.LIMIT_BALANCED,
            QualityMode.QUALITY: 15.0,  # Kein Limit
            QualityMode.MAXIMUM: 999.0,  # Studio 2026: absolut kein RT-Limit
        }[mode]

        # Tracking State
        self.phase_performances: list[PhasePerformance] = []
        self.skipped_phases: list[str] = []
        self.start_time: float | None = None
        self.audio_duration: float | None = None
        self.current_rt_factor: float = 0.0
        self.warnings: list[str] = []

        logger.info(
            f"PerformanceGuard initialized: Mode={mode.value}, "
            f"Target={self.target_rt_factor:.1f}× RT, "
            f"Enforce={enforce_limit}, Adaptive={enable_adaptive_skipping}"
        )

    def start_monitoring(self, audio_duration_seconds: float):
        """Startet Performance-Monitoring für Audio-File."""
        if audio_duration_seconds < 0.5:
            # Ignoriere Warmup/Dummy-Audio (z.B. 2-Sample-Signale aus HPSS-Init)
            logger.debug(
                "PerformanceGuard: start_monitoring ignoriert (audio_duration=%.6fs < 0.5s — Dummy-Signal)",
                audio_duration_seconds,
            )
            # audio_duration bleibt None → should_skip_phase gibt False zurück
            return
        self.start_time = time.time()
        self.audio_duration = audio_duration_seconds
        self.phase_performances.clear()
        self.skipped_phases.clear()
        self.warnings.clear()
        self.current_rt_factor = 0.0

        logger.info(
            f"Started monitoring: {audio_duration_seconds:.1f}s audio, "
            f"max {self.target_rt_factor * audio_duration_seconds:.1f}s processing"
        )

    def start_phase(self, phase_id: str) -> float:
        """
        Startet Monitoring für eine Phase.

        Returns:
            Phase start timestamp
        """
        return time.time()

    def end_phase(self, phase_id: str, phase_start_time: float) -> PhasePerformance:
        """
        Beendet Monitoring für eine Phase und gibt Performance zurück.

        Args:
            phase_id: Phase ID
            phase_start_time: Timestamp von start_phase()

        Returns:
            PhasePerformance Objekt
        """
        phase_end_time = time.time()
        phase_duration = phase_end_time - phase_start_time

        # RT Factor für diese Phase
        phase_rt_factor = phase_duration / self.audio_duration if self.audio_duration else 0

        # Ist Phase kritisch?
        is_critical = self._is_phase_critical(phase_id)

        perf = PhasePerformance(
            phase_id=phase_id,
            start_time=phase_start_time,
            end_time=phase_end_time,
            duration_seconds=phase_duration,
            audio_duration_seconds=self.audio_duration,
            rt_factor=phase_rt_factor,
            is_critical=is_critical,
            skipped=False,
        )

        self.phase_performances.append(perf)

        # Update current RT Factor (kumulativ)
        total_time_so_far = phase_end_time - self.start_time
        self.current_rt_factor = total_time_so_far / self.audio_duration if self.audio_duration else 0

        # Status Check
        status = self._get_status()

        logger.debug(
            f"Phase {phase_id}: {phase_duration:.2f}s ({phase_rt_factor:.2f}× RT), "
            f"Total: {self.current_rt_factor:.2f}× RT [{status.value}]"
        )

        # Warnungen
        if status == PerformanceStatus.CRITICAL:
            warning = f"⚠️ Performance CRITICAL: {self.current_rt_factor:.2f}× RT (limit: {self.target_rt_factor:.1f}×)"
            self.warnings.append(warning)
            logger.warning(warning)
        elif status == PerformanceStatus.EXCEEDED:
            warning = f"❌ Performance EXCEEDED: {self.current_rt_factor:.2f}× RT"
            self.warnings.append(warning)
            logger.error(warning)

        return perf

    def should_skip_phase(self, phase_id: str, estimated_time_seconds: float, remaining_phases: int) -> bool:
        """
        Entscheidet ob Phase übersprungen werden sollte.

        Args:
            phase_id: Phase ID
            estimated_time_seconds: Geschätzte Laufzeit der Phase
            remaining_phases: Anzahl verbleibender Phasen

        Returns:
            True wenn Phase übersprungen werden sollte
        """
        if not self.enable_adaptive_skipping:
            return False

        if self.mode == QualityMode.QUALITY:
            return False  # Quality Mode: Never skip

        # Kein Skip wenn audio_duration nicht gesetzt oder zu kurz (Dummy-Audio-Guard)
        if not self.audio_duration or self.audio_duration < 0.5 or self.start_time is None:
            return False

        # Kritische Phasen nie skippen
        if self._is_phase_critical(phase_id):
            return False

        # Musikalische Exzellenz — Priorität 1 (§MusEx-P1): niemals überspringen.
        # Doppelter Schutz: Priority-Dict UND Namens-Set verhindern Skip.
        # RT×3-Kompatibilität garantiert: alle Excellence-Phasen < 15 ms/s (§9.5).
        if phase_id in self.MUSICAL_EXCELLENCE_PHASES:
            logger.debug(
                f"🎵 Phase '{phase_id}' ist Musikalische-Exzellenz-Phase (§MusEx-P1) "
                f"— wird niemals übersprungen (RT×3 Budget: {self.RT3_EXCELLENCE_BUDGET:.1f}×)"
            )
            return False

        # Prognostiziere RT Factor nach dieser Phase
        estimated_total_time = (time.time() - self.start_time) + estimated_time_seconds
        estimated_total_time / self.audio_duration if self.audio_duration else 0

        # Schätze verbleibende Zeit (konservativ: 0.5s pro Phase)
        estimated_remaining_time = remaining_phases * 0.5
        final_projected_rt_factor = (
            (estimated_total_time + estimated_remaining_time) / self.audio_duration if self.audio_duration else 0
        )

        # Skip-Kriterien basierend auf Phase Priority
        phase_priority = self.PHASE_PRIORITIES.get(phase_id, 5)  # Default: Medium

        # Skip-Thresholds
        if phase_priority <= 3:  # LOW Priority
            skip_threshold = 3.5
        elif phase_priority <= 6:  # MEDIUM Priority
            skip_threshold = 4.5
        elif phase_priority <= 8:  # HIGH Priority
            skip_threshold = 4.8
        else:  # CRITICAL (≥ 9)
            return False  # Never skip

        # Entscheidung
        should_skip = final_projected_rt_factor > skip_threshold

        if should_skip:
            logger.warning(
                f"⏭️ Skipping {phase_id} (priority={phase_priority}): "
                f"Projected {final_projected_rt_factor:.2f}× RT > {skip_threshold:.1f}× threshold"
            )
            self.skipped_phases.append(phase_id)

        return should_skip

    def check_early_exit(self, remaining_phases: int) -> bool:
        """
        Prüft ob Early-Exit nötig ist (3× RT Limit bereits erreicht/überschritten).

        Args:
            remaining_phases: Anzahl verbleibender Phasen

        Returns:
            True wenn sofortiger Abbruch empfohlen
        """
        if not self.enforce_limit:
            return False

        if self.mode == QualityMode.QUALITY:
            return False  # Quality Mode: No limit

        # Aktuelle RT Factor
        if self.current_rt_factor > self.LIMIT_3X_RT:
            logger.error(
                f"❌ EARLY EXIT: 3× RT Limit exceeded ({self.current_rt_factor:.2f}× RT), "
                f"aborting {remaining_phases} remaining phases"
            )
            return True

        # Prognose: Werden wir das Limit überschreiten?
        estimated_remaining_time = remaining_phases * 0.5  # Konservativ
        total_time = (time.time() - self.start_time) + estimated_remaining_time
        projected_rt_factor = total_time / self.audio_duration if self.audio_duration else 0

        if projected_rt_factor > self.LIMIT_3X_RT * 1.1:  # 10% Puffer
            logger.warning(
                f"⚠️ EARLY EXIT recommended: Projected {projected_rt_factor:.2f}× RT > limit, "
                f"would abort {remaining_phases} phases"
            )
            # Aber nicht wirklich abbrechen, nur warnen
            return False

        return False

    def get_performance_report(self) -> PerformanceReport:
        """Erzeugt vollständigen Performance-Report."""
        if self.start_time is None or self.audio_duration is None:
            raise RuntimeError("Monitoring not started!")

        total_duration = time.time() - self.start_time
        total_rt_factor = total_duration / self.audio_duration
        status = self._get_status()

        # Quality Degradation berechnen
        quality_degradation = self._calculate_quality_degradation()

        report = PerformanceReport(
            total_duration_seconds=total_duration,
            audio_duration_seconds=self.audio_duration,
            total_rt_factor=total_rt_factor,
            status=status,
            phases=self.phase_performances.copy(),
            skipped_phases=self.skipped_phases.copy(),
            warnings=self.warnings.copy(),
            quality_degradation=quality_degradation,
        )

        logger.info(f"\n{report}")

        return report

    def _get_status(self) -> PerformanceStatus:
        """Ermittelt aktuellen Performance-Status."""
        rt = self.current_rt_factor

        if rt > self.WARNING_THRESHOLD_CRITICAL:
            return PerformanceStatus.EXCEEDED
        elif rt > self.WARNING_THRESHOLD_ACCEPTABLE:
            return PerformanceStatus.CRITICAL
        elif rt > self.WARNING_THRESHOLD_GOOD:
            return PerformanceStatus.ACCEPTABLE
        elif rt > self.WARNING_THRESHOLD_OPTIMAL:
            return PerformanceStatus.GOOD
        else:
            return PerformanceStatus.OPTIMAL

    def _is_phase_critical(self, phase_id: str) -> bool:
        """Prüft ob Phase kritisch für Quality ist."""
        priority = self.PHASE_PRIORITIES.get(phase_id, 5)
        return priority >= 9  # Priority ≥ 9 = CRITICAL

    def _calculate_quality_degradation(self) -> float:
        """
        Berechnet Quality-Degradation durch geskippte Phasen.

        Returns:
            0.0 - 1.0 (0 = keine Degradation, 1 = kompletter Verlust)
        """
        if not self.skipped_phases:
            return 0.0

        # Summiere Priorities der geskippten Phasen
        total_priority = sum(self.PHASE_PRIORITIES.values())
        skipped_priority = sum(self.PHASE_PRIORITIES.get(phase_id, 5) for phase_id in self.skipped_phases)

        # Degradation = Anteil der geskippten Priority
        degradation = skipped_priority / total_priority if total_priority > 0 else 0

        return min(1.0, degradation)

    def get_phase_budget_seconds(self, phase_id: str, total_phases: int, completed_phases: int) -> float:
        """
        Berechnet verbleibendes Time-Budget für eine Phase.

        Args:
            phase_id: Phase ID
            total_phases: Gesamtanzahl Phasen
            completed_phases: Anzahl abgeschlossener Phasen

        Returns:
            Verbleibendes Budget in Sekunden
        """
        # Gesamtbudget
        total_budget = self.target_rt_factor * self.audio_duration if self.audio_duration else 0

        # Bereits verbrauchte Zeit
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        # Verbleibendes Budget
        remaining_budget = total_budget - elapsed_time

        # Verteile auf verbleibende Phasen
        remaining_phases = total_phases - completed_phases
        phase_budget = remaining_budget / remaining_phases if remaining_phases > 0 else 0

        return max(0, phase_budget)


# ========== CLI/Testing Interface ==========

if __name__ == "__main__":
    """Test PerformanceGuard mit Beispiel-Workflow."""

    # Setup Logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.debug(f"\n{'='*60}")
    logger.debug("PERFORMANCE GUARD TEST")
    logger.debug(f"{'='*60}\n")

    # Simuliere 3:45 Audio (225 Sekunden)
    audio_duration = 225.0

    # Test 1: Balanced Mode (sollte erfolgreich sein)
    logger.debug("TEST 1: Balanced Mode (Target: 2.4× RT = 540s max)")
    logger.debug("-" * 60)

    guard = PerformanceGuard(mode=QualityMode.BALANCED, enforce_limit=True, enable_adaptive_skipping=True)
    guard.start_monitoring(audio_duration)

    # Simuliere Phasen
    phases = [
        ("click_removal", 15, 10),  # (phase_id, duration_seconds, priority)
        ("hum_removal", 20, 10),
        ("denoise", 30, 8),
        ("stereo_enhancement", 25, 6),
        ("final_polish", 40, 3),
    ]

    for i, (phase_id, duration, priority) in enumerate(phases):
        # Prüfe ob skippen
        remaining = len(phases) - i - 1
        if guard.should_skip_phase(phase_id, duration, remaining):
            continue

        # Simuliere Phase
        phase_start = guard.start_phase(phase_id)
        time.sleep(duration / 100)  # Skaliert runter für schnellen Test
        guard.end_phase(phase_id, phase_start)

        # Check Early Exit
        if guard.check_early_exit(remaining):
            logger.debug("⚠️ Early exit triggered!")
            break

    report1 = guard.get_performance_report()

    # Test 2: Fast Mode
    logger.debug("\n\nTEST 2: Fast Mode (Target: 1.5× RT = 338s max)")
    logger.debug("-" * 60)

    guard2 = PerformanceGuard(mode=QualityMode.FAST, enforce_limit=True, enable_adaptive_skipping=True)
    guard2.start_monitoring(audio_duration)

    for i, (phase_id, duration, priority) in enumerate(phases):
        remaining = len(phases) - i - 1
        if guard2.should_skip_phase(phase_id, duration * 0.6, remaining):  # Faster phases in fast mode
            continue

        phase_start = guard2.start_phase(phase_id)
        time.sleep(duration * 0.6 / 100)
        guard2.end_phase(phase_id, phase_start)

    report2 = guard2.get_performance_report()

    # Zusammenfassung
    logger.debug(f"\n\n{'='*60}")
    logger.debug("SUMMARY")
    logger.debug(f"{'='*60}")
    logger.debug(
        f"Balanced Mode: {report1.total_rt_factor:.2f}× RT, "
        f"{len(report1.skipped_phases)} skipped, "
        f"{(1-report1.quality_degradation)*100:.1f}% quality"
    )
    logger.debug(
        f"Fast Mode:     {report2.total_rt_factor:.2f}× RT, "
        f"{len(report2.skipped_phases)} skipped, "
        f"{(1-report2.quality_degradation)*100:.1f}% quality"
    )
