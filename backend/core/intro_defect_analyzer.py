"""
§2.59 Intro Defect Zone Detector (2026-07-09)

Erkennt transient Bandfehler, die NUR am Song-Anfang auftreten
(Leader-Tape, Anlauf-Störungen, Kopf-Kontakt-Probleme).
Markiert diese Zonen für fokussierte, hochpräzise Reparatur.

Prinzip: Nicht das ganze Lied gleich behandeln.
Nur die kranken Stellen operieren. Der Rest bleibt unberührt.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class IntroDefectZone:
    """Eine Zone mit konzentrierten Bandfehlern am Song-Anfang."""

    def __init__(
        self,
        start_s: float,
        end_s: float,
        defect_type: str,
        severity: float,
    ) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.defect_type = defect_type
        self.severity = severity


class IntroDefectAnalyzer:
    """Analysiert die ersten 30 Sekunden auf konzentrierte Bandfehler."""

    INTRO_DURATION_S: float = 30.0
    MIN_SEVERITY: float = 0.3

    def analyze(
        self,
        defect_scores: dict[str, float],
        transport_bump_count: int = 0,
        audio_duration_s: float = 0.0,
    ) -> list[IntroDefectZone]:
        """Findet Bandfehler-Zonen, die auf den Song-Anfang konzentriert sind.

        Returns:
            Liste von IntroDefectZone, sortiert nach Startzeit.
        """
        zones: list[IntroDefectZone] = []

        # Bandfehler, die typischerweise am Anfang konzentriert sind
        intro_defect_types = {
            "wow": "Geschwindigkeitsschwankung beim Anlauf",
            "flutter": "Flutter beim Bandstart",
            "transport_bump": "Transport-Störung",
            "modulation_noise": "Band-Anlauf-Rauschen",
            "dropouts": "Kopf-Kontakt-Aussetzer",
        }

        for defect_type, description in intro_defect_types.items():
            sev = defect_scores.get(defect_type, 0.0)
            if sev < self.MIN_SEVERITY:
                continue

            # Zone: erste INTRO_DURATION_S Sekunden
            end_s = min(self.INTRO_DURATION_S, audio_duration_s if audio_duration_s > 0 else 30.0)
            zones.append(IntroDefectZone(0.0, end_s, defect_type, sev))

        # Transport-Bump ist besonders stark am Anfang
        if transport_bump_count > 50:
            zones.append(
                IntroDefectZone(0.0, min(15.0, audio_duration_s),
                                "transport_bump", 0.70)
            )

        if zones:
            logger.info(
                "IntroDefectAnalyzer: %d Bandfehler-Zone(n) am Song-Anfang "
                "(0–%.0fs) für fokussierte Reparatur markiert",
                len(zones),
                self.INTRO_DURATION_S,
            )
            for z in zones:
                logger.debug("  Zone: %.1f–%.1fs %s (sev=%.2f)",
                             z.start_s, z.end_s, z.defect_type, z.severity)

        return zones


# ── Universelle Defekt-Klassifikation ───────────────────────────────────

# §2.59: 24 von 66 Defekten sind ZEITLICH LOKALISIERT.
# Sie sollten per-Instance chirurgisch behandelt werden, nicht global.
SURGICAL_DEFECT_TYPES: frozenset[str] = frozenset({
    # Transienten (ms-Bereich)
    "clicks", "crackle", "click_pop",
    # Dropouts (ms-s)
    "dropouts", "dropout_oxide", "dropout_head_contact", "dropout_splice",
    # Transport (ereignisbasiert)
    "transport_bump", "tape_splice_artifact",
    # Zeitliche Artefakte
    "pre_echo", "print_through", "groove_echo", "mpeg_frame_loss",
    # Kopf/Kontakt (transient)
    "tape_head_clog", "sticky_shed_residue",
    # Lokalisierte Bandfehler (Sektionen)
    "wow", "flutter", "scrape_flutter", "multiband_wow_flutter",
    "modulation_noise",
    # Positionsabhängig
    "inner_groove_distortion", "dc_offset",
    "motor_interference", "sibilance", "transient_smearing",
})

# Defekte die GLOBAL behandelt werden (39 Typen):
# bandwidth_loss, high_freq_noise, quantization_noise, compression_artifacts,
# clipping, dynamic_compression_excess, reverb_excess, stereo_imbalance,
# soft_saturation, digital_artifacts, aliasing, jitter_artifacts, bias_error,
# dolby_nr_mismatch, hf_remanence_loss, generation_loss, nr_breathing_artifact,
# phase_issues, stereo_field_collapse, phase_rotation, pitch_drift,
# speed_calibration_error, hum, low_freq_rumble, overload_distortion,
# intermodulation_distortion, crosstalk, vocal_harshness, riaa_curve_error,
# azimuth_error, head_wear, tape_head_level_dip, amplitude_drift,
# proximity_effect_excess, room_mode_resonance, lacquer_disc_degradation,
# stylus_damage, flutter_spectral_sidebands
# → Diese werden weiterhin global durch die Phasen behandelt.
