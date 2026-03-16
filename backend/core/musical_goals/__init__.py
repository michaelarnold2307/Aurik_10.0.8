"""
AURIK v9.9 Musical Goals Package
==================================

Messbare Metriken für alle 9 musikalischen Qualitätsziele:

1.  Bass-Kraft     – Kraftvolle Basswiedergabe (20–250 Hz)     ≥ 0.85
2.  Brillanz       – HF-Klarheit (8–20 kHz)                    ≥ 0.85
3.  Wärme          – Mittentiefe (200–2000 Hz)                  ≥ 0.80
4.  Natürlichkeit  – Gesamtklang ohne Artefakte                 ≥ 0.90
5.  Authentizität  – Klang-Fingerabdruck & Stimme               ≥ 0.88
6.  Emotionalität  – Dynamik & Ausdruck                         ≥ 0.87
7.  Transparenz    – Klarheit & Trennung                        ≥ 0.89
8.  Groove         – Mikro-Timing, Swing, DTW ≤ 8 ms RMS       ≥ 0.88  (v9.9)
9.  Spatial Depth  – Räumliche Tiefe & Stereo-Bild              ≥ 0.75  (v9.9)

Version: 9.9.0
"""

from .musical_goals_metrics import (
    AuthentizitaetMetric,
    BassKraftMetric,
    BrillanzMetric,
    EmotionalitaetMetric,
    GoalMeasurement,
    GrooveMetric,
    MusicalGoalsChecker,
    NatuerlichkeitMetric,
    SpatialDepthMetric,
    TransparenzMetric,
    WaermeMetric,
    get_checker,
    measure_all,
)
from .musical_goals_monitor import MonitoringCheckpoint, MonitoringReport, MusicalGoalsMonitor, PreValidationResult

__all__ = [
    # Metriken (einzeln)
    "BassKraftMetric",
    "BrillanzMetric",
    "WaermeMetric",
    "NatuerlichkeitMetric",
    "AuthentizitaetMetric",
    "EmotionalitaetMetric",
    "TransparenzMetric",
    "GrooveMetric",  # v9.9 — 8. Ziel
    "SpatialDepthMetric",  # v9.9 — 9. Ziel
    # Checker
    "MusicalGoalsChecker",
    "GoalMeasurement",
    # Singletons & Convenience
    "get_checker",
    "measure_all",
    # Monitor
    "MusicalGoalsMonitor",
    "PreValidationResult",
    "MonitoringCheckpoint",
    "MonitoringReport",
]

__version__ = "9.9.0"
