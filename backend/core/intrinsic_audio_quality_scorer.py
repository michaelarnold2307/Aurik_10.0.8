"""
core/intrinsic_audio_quality_scorer.py
Intrinsic Audio Quality Scorer (IAQS)
=======================================

Psychoakustisch fundierter Qualitätsscorer — vollständig ohne externe
Abhängigkeiten (kein CDPAM, kein DNSMOS, kein PyTorch).

Basiert auf messbaren Signal-Eigenschaften, die stark mit wahrgenommener
Qualität korrelieren:

  A) Spektrale Güte
     - SNR (blind, via Minimum-Statistics-Schätzung)
     - Spektrale Regularität (Spitzen-zu-Tal-Verhältnis)
     - Bandbreiteneffizienz (genutzte Bandbreite vs. erwartete)
     - Bark-Band-Energie-Verteilung (Psychoakustisches Modell)

  B) Zeitbereichs-Güte
     - Transientenklarheit (Attack-Erkennung im Zeitsignal)
     - Dynamikumfang (EBU R128 Loudness Range näherungsweise)
     - Klirrfaktor-Schätzung (THD via Harmonics)

  C) Musikalische Güte
     - Harmonizität (Verhältnis harmonische zu inharmonische Energie)
     - Stimmungsklarheit (Pitch-Konsistenz über Zeit)
     - Authentizitätsindikator (Vintage vs. Digital-Überprägung)

  D) Artefakt-Detektion
     - Klick-Energie-Residuen (hohe Kurzzeitpegel)
     - Digitale Clipping-Indikatoren (Flat-Top-Samples)
     - Codec-Blockartefakte (periodische Spektralmodulation)

Alle Metriken sind:
  - schnell (< 0.5× Echtzeit für typische Längen)
  - robust (kein NaN/Inf)
  - skaliert auf [0.0, 1.0] (1.0 = perfekt)

Verwendung in MultiPassEngine als fallback wenn Plugins fehlen,
und als primärer Scorer in AutonomousRestorationEngine.

Author: Aurik Development Team
Version: 1.0.0 "Perceptual Precision"
Date: 2026-02-17
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field


from backend.core.multi_pass_strategy import IntrinsicAudioQualityScorer  # canonical (§dedup)

logger = logging.getLogger(__name__)

# Bark-Band-Grenzen in Hz (25 Bänder nach Zwicker 1961)
_BARK_BANDS_HZ: tuple[float, ...] = (
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
    20000,
)


# ---------------------------------------------------------------------------
# Ergebnis-Datenstruktur
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicQualityScore:
    """Vollständiges intrinsisches Qualitätsergebnis."""

    # === Zusammenfassung ===
    overall: float = 0.0
    """Gewichteter Gesamtscore (0–1, 1 = perfekt)."""

    # === Spektral ===
    snr_estimate: float = 0.0
    """Blind-SNR-Schätzung in dB."""

    snr_score: float = 0.0
    """SNR normiert (0–1)."""

    spectral_regularity: float = 0.0
    """Spektrale Glätte (0–1, 1 = glatt)."""

    bandwidth_score: float = 0.0
    """Bandbreiteneffizienz (0–1)."""

    bark_balance: float = 0.0
    """Bark-Band-Balance (0–1, 1 = ideal)."""

    # === Zeitbereich ===
    dynamic_range_score: float = 0.0
    """Dynamikumfang-Score (0–1)."""

    transient_clarity: float = 0.0
    """Transientenklarheit (0–1)."""

    thd_estimate_pct: float = 0.0
    """THD-Schätzung in % (kleiner = besser)."""

    thd_score: float = 0.0
    """THD normiert (0–1, 1 = kein Klirr)."""

    # === Musikalisch ===
    harmonicity: float = 0.0
    """Harmonizität (0–1, 1 = rein harmonisch)."""

    pitch_consistency: float = 0.0
    """Pitch-Konsistenz (0–1, 1 = stabile Intonation)."""

    # === Artefakte ===
    click_residual: float = 0.0
    """Klick-Residual-Score (1 = keine Klicks, 0 = viele)."""

    clipping_score: float = 0.0
    """Clipping-Score (1 = kein Clipping, 0 = geclippt)."""

    codec_artifact_score: float = 0.0
    """Codec-Artefakt-Score (1 = keine, 0 = stark)."""

    # === Metadaten ===
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    is_stereo: bool = False
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------
