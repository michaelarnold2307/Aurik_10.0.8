"""
Aurik 9 — forensics/medium_detector.py  (§6.7, bindend ab v9.10.97)
=====================================================================
Vereinheitlichte Tonträgerketten-Erkennung: kombiniert die forensische
Spektralfingerabdruck-Analyse mit Bayesian-Material-Scoring für
weltklasse-Präzision bei komplexen Mehrstufenketten.

Ab v9.10.97 ist dies das **einzige** autoritative Material-Erkennungssystem.
MediumClassifier-Features (Rotation, Infrasonic, Codec MDCT) sind direkt
integriert — kein zweiter Klassifikator nötig.

Pflicht-Spektralfingerabdruck (§6.7.1):
    1. Rolloff 95 %  — diagnostiziert Bandbreitenbegrenzung
    2. Wow/Flutter-Index — Pitch-Instabilität via pYIN-Ableitung
    3. HF-Energie > 16 kHz — MP3/Kassettenkette
    4. Rauschpegel (Percentile-5 PSD)  — Bandrauschen
    5. Effektive Bandbreite — physikalische Signalbandbreite

Erweiterte Features (§6.7.3, NEU v9.10.97):
    6. Rotation-Periodizität     — Vinyl-Plattentellerfrequenz (0.3–2 Hz)
    7. Infraschall-RMS (< 20 Hz) — Vinyl-Lager-Rumble-Diskriminator
    8. MDCT-Codec-Artefakt-Score — MP3/AAC-Quantisierungsfingerabdruck
    9. Codec-Typ-Code            — MP3/AAC/lossy-Differenzierung

Kettenerkennung (§6.7.2):
    - Bayesian-Primär-Material-Scoring (Gaussian-Likelihood, 16 Materialien)
    - Sekundäre Codec-Schicht via Artefakt-Score
    - 3+-Layer-Ketten möglich (z. B. vinyl → tape → mp3_low)
    - is_multi_generation=True → kombinierte Phasen aller Materialien
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFingerprint:
    """Pflicht-Spektralfingerabdruck (§6.7.1 + §6.7.3) aus Rohsignal-Vorabanalyse."""

    rolloff_95_hz: float = 0.0  # Spectral Rolloff 95 % — Median
    wow_flutter_index: float = 0.0  # Pitch-Varianz [Hz std] über 100-ms-Fenster
    hf_energy_above_16k: float = 0.0  # Anteil Energie > 16 kHz an Gesamt
    noise_floor_db: float = -60.0  # 5. Perzentil der Frame-Energien [dBFS]
    effective_bandwidth_hz: float = 0.0  # HF-Rolloff −60 dBFS

    # --- §6.7.3 Erweiterte Features (v9.10.97) ---
    rotation_hz: float = 0.0  # Vinyl-Plattentellerfrequenz [Hz]; 0 = keine Rotation
    rotation_strength: float = 0.0  # Normierter Peak-SNR des Rotationssignals [0, 1]
    infrasonic_rms: float = 0.0  # Sub-20 Hz normierter RMS (Vinyl-Rumble)
    codec_artifact_score: float = 0.0  # MDCT-Quantisierungs-Artefakt-Score [0, 1]
    codec_type_code: float = 0.0  # 0=clean, 1=mp3, 2=aac, 3=lossy
    crackle_density: float = 0.0  # Anteil Samples > 4σ (Vinyl-Knackser)
    snr_db: float = 0.0  # Signal-Rausch-Abstand [dB]
    noise_color: float = 1.0  # Spektrale Neigung (β): 0=weiß, 2=braun

    # --- Alias-Properties für Test-Kompatibilität (§6.7.1) ---
    @property
    def rolloff_95_percent_hz(self) -> float:
        """Alias für rolloff_95_hz — Rückwärtskompatibilität."""
        return self.rolloff_95_hz

    @property
    def hf_energy_above_16khz_percent(self) -> float:
        """Alias für hf_energy_above_16k als Prozentwert (0–100)."""
        return float(self.hf_energy_above_16k * 100.0)

    def __contains__(self, item: object) -> bool:
        """Unterstützt 'key in fingerprint'-Syntax für Tests."""
        return item in (
            "rolloff_95_hz",
            "rolloff_95_percent_hz",
            "wow_flutter_index",
            "hf_energy_above_16k",
            "hf_energy_above_16khz_percent",
            "noise_floor_db",
            "effective_bandwidth_hz",
            "rotation_hz",
            "rotation_strength",
            "infrasonic_rms",
            "codec_artifact_score",
            "codec_type_code",
            "crackle_density",
            "snr_db",
            "noise_color",
        )

    def as_dict(self) -> dict:
        return {
            "rolloff_95_hz": self.rolloff_95_hz,
            "rolloff_95_percent_hz": self.rolloff_95_hz,
            "wow_flutter_index": self.wow_flutter_index,
            "hf_energy_above_16k_fraction": self.hf_energy_above_16k,
            "hf_energy_above_16khz_percent": self.hf_energy_above_16k * 100.0,
            "noise_floor_db": self.noise_floor_db,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "rotation_hz": self.rotation_hz,
            "rotation_strength": self.rotation_strength,
            "infrasonic_rms": self.infrasonic_rms,
            "codec_artifact_score": self.codec_artifact_score,
            "codec_type_code": self.codec_type_code,
            "crackle_density": self.crackle_density,
            "snr_db": self.snr_db,
            "noise_color": self.noise_color,
        }


@dataclass
class TransferChain:
    """Erkannte Medien-Transferkette."""

    chain: list[str] = field(default_factory=list)
    """Kette von MediaType-Strings, z. B. ['tape', 'mp3_low']."""

    is_multi_generation: bool = False
    """True wenn ≥ 2 verschiedene Medienstufen erkannt wurden."""

    primary_material: str = "unknown"
    """Letzter analoger Träger = primärer MaterialType-Prior."""

    confidence: float = 0.0
    """Gesamtkonfidenz der Ketten-Schätzung ∈ [0, 1]."""

    reasoning: str = ""

    def __len__(self) -> int:
        return len(self.chain)


@dataclass
class MediumDetectionResult:
    """Vollständiges Ergebnis der Tonträgerketten-Erkennung."""

    transfer_chain: list[str]
    """Kette wie ['vinyl', 'tape', 'mp3_low'] — ursprünglicher Träger zuerst."""

    is_multi_generation: bool
    primary_material: str
    confidence: float
    spectral_fingerprint: SpectralFingerprint
    evidence: list[str] = field(default_factory=list)
    """Laienverständliche Diagnose-Begründungen."""
    medium_confidences: list[float] = field(default_factory=list)
    """Per-Link-Konfidenz — gleiche Länge wie transfer_chain."""

    # §6.7.3 (v9.10.97): Bayesian-Scoring-Ergebnis für Durchreichung
    bayesian_scores: dict[str, float] = field(default_factory=dict)
    """Posterior-Wahrscheinlichkeiten aller Materialtypen."""

    classification_result: object | None = None
    """ClassificationResult aus MediumClassifier (für cached_medium_result)."""

    dolby_nr_type: str = "none"
    """Erkannter Dolby/DBX NR-Typ: 'dolby_b'|'dolby_c'|'dolby_s'|'dbx_i'|'dbx_ii'|'none'."""

    dolby_nr_confidence: float = 0.0
    """Konfidenz der Dolby-NR-Erkennung ∈ [0, 1]."""

    @property
    def chain_label(self) -> str:
        return " → ".join(self.transfer_chain) if self.transfer_chain else "unknown"

    def as_dict(self) -> dict:
        return {
            "transfer_chain": self.transfer_chain,
            "medium_confidences": self.medium_confidences,
            "is_multi_generation": self.is_multi_generation,
            "primary_material": self.primary_material,
            "confidence": self.confidence,
            "chain_label": self.chain_label,
            "spectral_fingerprint": self.spectral_fingerprint.as_dict(),
            "evidence": self.evidence,
            "bayesian_scores": self.bayesian_scores,
            "dolby_nr_type": self.dolby_nr_type,
            "dolby_nr_confidence": self.dolby_nr_confidence,
        }


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class MediumDetector:
    """Vereinheitlichte forensische Tonträgerketten-Erkennung (§6.7, v9.10.97).

    Kombiniert:
      - Spektralfingerabdruck (5 Basis-Features, §6.7.1)
      - Erweiterte physikalische Features (Rotation, Infrasonic, Codec, §6.7.3)
      - Bayesian Gaussian-Likelihood Material-Scoring (aus MediumClassifier)
      - Mehrstufige Ketten-Inferenz (3+ Layer: vinyl → tape → mp3_low)

    Ersetzt die alte sequenzielle if/elif-Heuristik durch probabilistisches
    Scoring über 16 Materialtypen.

    Singleton-Zugang: ``get_medium_detector()``
    Convenience:      ``detect_medium_chain(audio, sr)``
    """

    # ── Schwellwerte ──────────────────────────────────────────────────
    HF_ENERGY_THRESHOLD_FRACTION: float = 0.001  # < 0.1 % → kein HF
    _CODEC_ARTIFACT_THRESHOLD: float = 0.15  # ab hier: Codec-Layer erkannt
    _ANALOG_POSTERIOR_MIN: float = 0.08  # Mindest-Posterior für Analog-Layer
    _SECONDARY_ANALOG_MIN: float = 0.08  # Mindest-Posterior für 2. Analog-Stufe
    _MAX_ANALOG_CHAIN_DEPTH: int = 4  # inkl. Primärquelle; erlaubt 3+ Transfer-Layer
    _SAME_ORDER_ANALOG_MIN: float = 0.22  # konservativer Guard für gleichrangige Analog-Stufen

    # Analog-Materialtypen — können als Primärquelle in Ketten vorkommen
    _ANALOG_MATERIALS: frozenset[str] = frozenset(
        {
            "shellac",
            "wax_cylinder",
            "vinyl",
            "wire_recording",
            "lacquer_disc",
            "tape",
            "reel_tape",
            "cassette",
        }
    )

    # Digitale Container-Formate — niemals Primärquelle
    _CODEC_MATERIALS: frozenset[str] = frozenset(
        {
            "mp3_low",
            "mp3_high",
            "aac",
            "streaming",
            "minidisc",
        }
    )

    # Digitale verlustfreie Formate
    _DIGITAL_LOSSLESS: frozenset[str] = frozenset(
        {
            "cd_digital",
            "dat",
        }
    )

    # Zeitliche Ordnung für Kettensortierung (niedrig = früher)
    _MEDIUM_ORDER: dict[str, int] = {
        "wax_cylinder": 0,
        "lacquer_disc": 0,
        "shellac": 0,
        "vinyl": 0,
        "wire_recording": 0,
        "reel_tape": 1,
        "tape": 1,
        "cassette": 1,
        "dat": 2,
        "cd_digital": 2,
        "mp3_low": 3,
        "mp3_high": 3,
        "aac": 3,
        "streaming": 3,
        "minidisc": 3,
    }

    # ── Bayesian Material-Modelle (Gaussian μ, σ) ────────────────────
    # Identisch mit MediumClassifier._MATERIAL_MODELS — kanonische Quelle.
    # Features: bandwidth_hz, snr_db, noise_color, crackle_density,
    #           wow_depth, block_artifact, pre_echo_ms,
    #           rotation_strength, infrasonic_rms, codec_type_code
    _MATERIAL_MODELS: dict[str, dict[str, tuple[float, float]]] = {
        "shellac": {
            "bandwidth_hz": (5500.0, 1500.0),
            "snr_db": (10.0, 5.0),
            "noise_color": (2.2, 0.5),
            "crackle_density": (0.02, 0.02),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.40, 0.20),
            "infrasonic_rms": (0.06, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        "wax_cylinder": {
            "bandwidth_hz": (3500.0, 1200.0),
            "snr_db": (6.0, 4.0),
            "noise_color": (2.8, 0.6),
            "crackle_density": (0.04, 0.03),
            "wow_depth": (1.0, 0.8),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.02, 0.03),
            "codec_type_code": (0.0, 0.3),
        },
        "vinyl": {
            "bandwidth_hz": (14000.0, 4000.0),
            "snr_db": (30.0, 10.0),
            "noise_color": (1.5, 0.4),
            "crackle_density": (0.004, 0.005),
            "wow_depth": (0.15, 0.15),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.45, 0.20),
            "infrasonic_rms": (0.08, 0.05),
            "codec_type_code": (0.0, 0.3),
        },
        "tape": {
            "bandwidth_hz": (12000.0, 3000.0),
            "snr_db": (25.0, 8.0),
            "noise_color": (1.6, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (1.2, 0.8),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "reel_tape": {
            "bandwidth_hz": (15000.0, 3000.0),
            "snr_db": (28.0, 7.0),
            "noise_color": (1.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.3, 0.3),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "wire_recording": {
            "bandwidth_hz": (5000.0, 1500.0),
            "snr_db": (12.0, 5.0),
            "noise_color": (2.0, 0.5),
            "crackle_density": (0.0001, 0.0002),
            "wow_depth": (3.0, 1.5),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.10),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "lacquer_disc": {
            "bandwidth_hz": (9000.0, 2500.0),
            "snr_db": (18.0, 6.0),
            "noise_color": (1.7, 0.4),
            "crackle_density": (0.008, 0.008),
            "wow_depth": (0.2, 0.2),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.30, 0.20),
            "infrasonic_rms": (0.04, 0.04),
            "codec_type_code": (0.0, 0.3),
        },
        "cassette": {
            "bandwidth_hz": (10000.0, 3000.0),
            "snr_db": (22.0, 7.0),
            "noise_color": (1.5, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (1.5, 1.0),
            "block_artifact": (0.0, 0.05),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.08),
            "infrasonic_rms": (0.01, 0.02),
            "codec_type_code": (0.0, 0.3),
        },
        "dat": {
            "bandwidth_hz": (20000.0, 2000.0),
            "snr_db": (50.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.08, 0.06),
            "pre_echo_ms": (0.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.5),
        },
        "cd_digital": {
            "bandwidth_hz": (21000.0, 1500.0),
            "snr_db": (60.0, 8.0),
            "noise_color": (0.2, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.05),
            "block_artifact": (0.0, 0.03),
            "pre_echo_ms": (0.0, 1.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (0.0, 0.3),
        },
        "mp3_low": {
            "bandwidth_hz": (11000.0, 2500.0),
            "snr_db": (35.0, 8.0),
            "noise_color": (0.5, 0.4),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.40, 0.15),
            "pre_echo_ms": (12.0, 6.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "mp3_high": {
            "bandwidth_hz": (17000.0, 2000.0),
            "snr_db": (42.0, 7.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.15, 0.10),
            "pre_echo_ms": (6.0, 4.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.0, 0.3),
        },
        "aac": {
            "bandwidth_hz": (19000.0, 1500.0),
            "snr_db": (48.0, 7.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.10, 0.08),
            "pre_echo_ms": (1.5, 1.5),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (2.0, 0.3),
        },
        "minidisc": {
            "bandwidth_hz": (14000.0, 2000.0),
            "snr_db": (40.0, 6.0),
            "noise_color": (0.4, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.18, 0.10),
            "pre_echo_ms": (3.0, 3.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (3.0, 0.5),
        },
        "streaming": {
            "bandwidth_hz": (18000.0, 2000.0),
            "snr_db": (45.0, 8.0),
            "noise_color": (0.3, 0.3),
            "crackle_density": (0.0, 0.001),
            "wow_depth": (0.0, 0.1),
            "block_artifact": (0.06, 0.05),
            "pre_echo_ms": (2.0, 2.0),
            "rotation_strength": (0.0, 0.05),
            "infrasonic_rms": (0.0, 0.01),
            "codec_type_code": (1.5, 0.8),
        },
        "unknown": {
            "bandwidth_hz": (12000.0, 8000.0),
            "snr_db": (25.0, 15.0),
            "noise_color": (1.0, 1.0),
            "crackle_density": (0.005, 0.01),
            "wow_depth": (0.5, 1.0),
            "block_artifact": (0.05, 0.10),
            "pre_echo_ms": (2.0, 5.0),
            "rotation_strength": (0.05, 0.15),
            "infrasonic_rms": (0.02, 0.05),
            "codec_type_code": (1.0, 1.5),
        },
    }

    _FEATURE_KEYS: list[str] = [
        "bandwidth_hz",
        "snr_db",
        "noise_color",
        "crackle_density",
        "wow_depth",
        "block_artifact",
        "pre_echo_ms",
        "rotation_strength",
        "infrasonic_rms",
        "codec_type_code",
    ]

    def _infer_analog_source_from_fingerprint(self, fp: SpectralFingerprint) -> list[tuple[str, float]]:
        """Infer analog source materials from physical fingerprint features.

        Called when file_ext zeroes Bayesian posteriors so that the Bayesian
        scorer alone cannot identify the original analog source.  Physical cues
        (vinyl rumble, cassette/tape transport flutter, shellac crackle) survive
        recording chain transfers and remain detectable even through codec encoding.

        Detection thresholds calibrated from MediumClassifier Gaussian models and
        empirical test corpus (see §6.7, Pohlmann 2010, IEC 60386).

        Returns:
            List of (material_key, confidence) sorted by MEDIUM_ORDER
            (original source first).
        """
        sources: list[tuple[str, float]] = []

        # ── Vinyl ─────────────────────────────────────────────────────
        # Physical evidence: infrasonic platter-bearing rumble + groove crackle
        # + optional turntable rotation periodicity.
        # Rumble: infrasonic_rms > 0.030 (vinyl model μ=0.08, σ=0.05)
        # Crackle: crackle_density > 0.004 (vinyl model μ=0.004, σ=0.005)
        vinyl_conf = 0.0
        if fp.infrasonic_rms > 0.030:
            vinyl_conf = max(vinyl_conf, float(min((fp.infrasonic_rms - 0.030) / 0.080, 1.0)))
        if fp.crackle_density > 0.004:
            vinyl_conf = max(vinyl_conf, float(min((fp.crackle_density - 0.004) / 0.025, 1.0)))
        if fp.rotation_strength > 0.08:
            vinyl_conf = max(vinyl_conf, float(fp.rotation_strength))
        if vinyl_conf >= 0.20:
            sources.append(("vinyl", float(np.clip(vinyl_conf, 0.20, 0.85))))

        # ── Shellac ───────────────────────────────────────────────────
        # Heavy crackle + strong infrasonic → beats vinyl classification.
        shellac_conf = 0.0
        if fp.crackle_density > 0.015 and fp.infrasonic_rms > 0.040:
            shellac_conf = float(min(fp.crackle_density / 0.040, 1.0))
        if shellac_conf >= 0.35:
            sources = [(m, c) for m, c in sources if m != "vinyl"]
            sources.append(("shellac", float(np.clip(shellac_conf, 0.35, 0.85))))

        # ── Cassette ─────────────────────────────────────────────────
        # Capstan/pinch-roller transport flutter: wow_flutter_index 0.30–2.5
        # Calibration from Bayesian model: cassette μ=1.5, σ=1.0.
        # Vinyl-only wow_depth μ=0.15, σ=0.15 → 99th-percentile vinyl flutter ≈ 0.30.
        # When a disc source is already in the chain the combined vinyl+cassette flutter
        # starts at ~0.18 even for high-quality decks (Nakamichi flutter spec 0.04 % WRMS,
        # vinyl pitch-drift adds 0.10–0.15). Use a lower threshold in that case.
        has_disc = any(m in ("vinyl", "shellac", "lacquer_disc") for m, _ in sources)
        _cass_flutter_thresh = 0.18 if has_disc else 0.30
        cassette_conf = 0.0
        if fp.wow_flutter_index > _cass_flutter_thresh:
            cassette_conf = float(min((fp.wow_flutter_index - _cass_flutter_thresh) / 1.20, 1.0))
        # Bandwidth evidence: cassette tape limits HF to ~14.5–15.5 kHz due to azimuth
        # misalignment and tape-formula roll-off.  Only use when codec artifacts are low
        # (MP3 also limits HF; we must not conflate codec roll-off with cassette roll-off).
        if has_disc and 5_000 < fp.effective_bandwidth_hz < 15_500 and fp.codec_artifact_score < 0.30:
            _bw_conf = float(np.clip((15_500 - fp.effective_bandwidth_hz) / 6_000, 0.15, 0.55))
            cassette_conf = max(cassette_conf, _bw_conf)
        _cassette_min = 0.15 if has_disc else 0.25
        if cassette_conf >= _cassette_min:
            sources.append(("cassette", float(np.clip(cassette_conf, _cassette_min, 0.85))))

        # ── Reel tape ────────────────────────────────────────────────
        # Direct reel-tape recording (no disc source): moderate flutter, no rotation.
        # Reel-tape Bayesian model: wow_depth μ=0.3, σ=0.3 (lower than cassette).
        if not has_disc and fp.wow_flutter_index > 0.20 and fp.rotation_strength < 0.05:
            tape_conf = float(min((fp.wow_flutter_index - 0.20) / 1.00, 1.0))
            if tape_conf >= 0.20:
                sources.append(("reel_tape", float(np.clip(tape_conf, 0.20, 0.85))))

        # Sort by signal-chain order: disc (0) before tape (1) before codec (2).
        sources.sort(key=lambda x: self._MEDIUM_ORDER.get(x[0], 5))
        return sources

    @staticmethod
    def _is_benign_codec_source(audio: np.ndarray, sr: int, fp: SpectralFingerprint) -> bool:
        """Heuristic guard to prevent false analog-chain inference on clean digital sources."""
        mono = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if mono.ndim == 2:
            mono = mono.mean(axis=0) if mono.shape[0] <= mono.shape[1] else mono.mean(axis=1)
        if mono.size < 4096 or sr <= 0:
            return False

        abs_mono = np.abs(mono)
        hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
        near_clip_ratio = float(np.mean(abs_mono >= 0.98))

        dyn_window = max(1, int(sr * 0.4))
        dyn_frames = mono.size // dyn_window
        if dyn_frames >= 2:
            dyn_blocks = mono[: dyn_frames * dyn_window].reshape(dyn_frames, dyn_window)
            dyn_rms = np.sqrt(np.mean(dyn_blocks**2, axis=1) + 1e-12)
            dyn_std_db = float(np.std(20.0 * np.log10(dyn_rms + 1e-12)))
        else:
            dyn_std_db = 0.0

        n_fft = 4096
        hop = 1024
        window = np.hanning(n_fft).astype(np.float32)
        flatness_values: list[float] = []
        for start in range(0, mono.size - n_fft + 1, hop):
            frame = mono[start : start + n_fft] * window
            mag = np.abs(np.fft.rfft(frame)).astype(np.float64) + 1e-12
            flatness_values.append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))

        if not flatness_values:
            return False

        flatness_median = float(np.median(flatness_values))
        return (
            hard_clip_ratio <= 1e-5
            and near_clip_ratio <= 1e-4
            and dyn_std_db >= 3.5
            and flatness_median <= 1e-2
            and fp.noise_floor_db <= -38.0
            and fp.wow_flutter_index < 0.02
            and fp.effective_bandwidth_hz >= 8_000.0
        )

    def _compute_fingerprint(self, audio: np.ndarray, sr: int) -> SpectralFingerprint:
        """Berechnet den vollständigen Spektralfingerabdruck (§6.7.1 + §6.7.3).

        Umfasst 5 Basis-Features + 8 erweiterte Features für Bayesian Scoring.
        NaN/Inf-sicher; alle Felder werden immer befüllt.
        """
        mono = self._to_mono(audio)
        n = len(mono)
        if n == 0:
            return SpectralFingerprint()

        hop = max(1, n // 200)
        win = min(2048, n)

        # ── 1. Rolloff 95 % ────────────────────────────────────────────
        try:
            frames = [mono[i : i + win] for i in range(0, n - win, hop)]
            rolloffs = []
            for frame in frames[:100]:
                spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
                cum = np.cumsum(spec**2)
                total = cum[-1]
                if total > 0:
                    idx = int(np.searchsorted(cum, 0.95 * total))
                    rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))
            rolloff_95 = float(np.median(rolloffs)) if rolloffs else 0.0
        except Exception:
            rolloff_95 = 0.0

        # ── 2. Wow/Flutter-Index ────────────────────────────────────────
        try:
            from scipy.signal import hilbert

            frame_size = int(0.1 * sr)  # 100 ms
            pitches = []
            for start in range(0, n - frame_size, frame_size):
                frame = mono[start : start + frame_size].astype(np.float64)
                analytic: np.ndarray = hilbert(frame)  # type: ignore[assignment]
                env = np.abs(analytic)
                mean_e = float(np.mean(env))
                if mean_e > 1e-6:
                    pitches.append(mean_e)
            wow_flutter = float(np.std(np.diff(pitches))) if len(pitches) > 2 else 0.0
        except Exception:
            wow_flutter = 0.0

        # ── 3. HF-Energie > 16 kHz ─────────────────────────────────────
        try:
            spec_full = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_full = np.fft.rfftfreq(65536, 1.0 / sr)
            mask_hf = freqs_full > 16_000
            total_e = float(np.sum(spec_full**2))
            hf_e = float(np.sum(spec_full[mask_hf] ** 2))
            hf_fraction = hf_e / max(total_e, 1e-12)
        except Exception:
            hf_fraction = 0.0

        # ── 4. Rauschpegel (5. Perzentil PSD) ──────────────────────────
        try:
            frame_energies = []
            for start in range(0, n - win, hop):
                e = float(np.mean(mono[start : start + win] ** 2))
                if e > 0:
                    frame_energies.append(10 * math.log10(e))
            noise_floor = float(np.percentile(frame_energies, 5)) if frame_energies else -60.0
            noise_floor = max(-120.0, min(0.0, noise_floor))
        except Exception:
            noise_floor = -60.0

        # ── 5. Effektive Bandbreite (Rolloff −60 dBFS, Multi-Segment) ──────────────
        # BUG FIX (v9.10.98): Using only the first 65536 samples (≈1.4 s at 48 kHz)
        # caused false wax_cylinder detection for tracks with a quiet intro. A silent
        # or near-silent intro yields effective_bandwidth_hz ≈ 0, which matches the
        # wax_cylinder Gaussian model (μ=3500 Hz, σ=1200 Hz) far better than any
        # digital format. Fix: compute bandwidth on 5 energy-distributed segments
        # spanning the full audio and take the 80th-percentile result, ensuring that
        # the representative musical content (not a silent intro) determines the
        # material classification.
        try:
            n_fft_bw = 65536
            freqs_bw = np.fft.rfftfreq(n_fft_bw, 1.0 / sr)
            seg_len = n_fft_bw
            # Select 5 segments distributed across the audio; prefer energy-rich sections
            rms_win = max(1, int(sr * 0.5))  # 0.5 s RMS windows for energy ranking
            n_rms = max(1, n // rms_win)
            rms_vals = np.array([float(np.mean(mono[i * rms_win : (i + 1) * rms_win] ** 2)) for i in range(n_rms)])
            # Pick 5 energy-ranked segments; fallback to equally spaced if audio short
            if n >= seg_len * 2:
                # Sort by energy descending; pick top-5 diverse segments (≥2 s apart)
                sorted_idx = np.argsort(rms_vals)[::-1]
                chosen_starts: list[int] = []
                min_gap_frames = max(1, int(sr * 2.0 / rms_win))
                for idx in sorted_idx:
                    start_s = idx * rms_win
                    if start_s + seg_len > n:
                        continue
                    # Enforce minimum gap between segments to avoid highly correlated windows
                    if all(abs(idx - c // rms_win) >= min_gap_frames for c in chosen_starts):
                        chosen_starts.append(start_s)
                    if len(chosen_starts) >= 5:
                        break
                # Fallback: equally-spaced segments if diversity selection yielded too few
                if not chosen_starts:
                    step = max(1, (n - seg_len) // 5)
                    chosen_starts = [min(i * step, n - seg_len) for i in range(5)]
            else:
                chosen_starts = [0]

            bw_candidates: list[float] = []
            for ss in chosen_starts:
                seg = mono[ss : ss + seg_len]
                if len(seg) < seg_len:
                    seg = np.pad(seg, (0, seg_len - len(seg)))
                spec_seg = np.abs(np.fft.rfft(seg.astype(np.float64), n=n_fft_bw))
                peak = float(spec_seg.max())
                if peak < 1e-12:
                    continue  # silent segment — skip
                spec_db_seg = 20.0 * np.log10(np.clip(spec_seg / peak, 1e-15, np.inf))
                above = freqs_bw[spec_db_seg > -60.0]
                if len(above) > 0:
                    bw_candidates.append(float(above.max()))

            if bw_candidates:
                # 80th-percentile: robust against outlier-quiet segments while not
                # being fooled by a single segment with anomalous HF boost.
                eff_bw = float(np.percentile(bw_candidates, 80))
            else:
                eff_bw = 0.0
        except Exception:
            eff_bw = 0.0

        # ── 6. Erweiterte Features (§6.7.3) ────────────────────────────
        rotation_hz = 0.0
        rotation_strength = 0.0
        infrasonic_rms = 0.0
        codec_artifact_score = 0.0
        codec_type_code = 0.0
        crackle_density = 0.0
        snr_db = 0.0
        noise_color = 0.0

        try:
            rotation_hz, rotation_strength = self._rotation_periodicity(mono, sr)
        except Exception:
            pass

        try:
            infrasonic_rms = self._infrasonic_rms(mono, sr)
        except Exception:
            pass

        try:
            codec_artifact_score, codec_type_code = self._codec_artifact_score(mono, sr)
        except Exception:
            pass

        try:
            crackle_density = self._crackle_density(mono, sr)
        except Exception:
            pass

        try:
            snr_db = self._snr(mono, sr)
        except Exception:
            pass

        try:
            noise_color = self._noise_color(mono, sr)
        except Exception:
            pass

        return SpectralFingerprint(
            rolloff_95_hz=float(np.nan_to_num(rolloff_95)),
            wow_flutter_index=float(np.nan_to_num(wow_flutter)),
            hf_energy_above_16k=float(np.nan_to_num(hf_fraction)),
            noise_floor_db=float(np.nan_to_num(noise_floor, nan=-60.0)),
            effective_bandwidth_hz=float(np.nan_to_num(eff_bw)),
            rotation_hz=float(np.nan_to_num(rotation_hz)),
            rotation_strength=float(np.nan_to_num(rotation_strength)),
            infrasonic_rms=float(np.nan_to_num(infrasonic_rms)),
            codec_artifact_score=float(np.nan_to_num(codec_artifact_score)),
            codec_type_code=float(np.nan_to_num(codec_type_code)),
            crackle_density=float(np.nan_to_num(crackle_density)),
            snr_db=float(np.nan_to_num(snr_db)),
            noise_color=float(np.nan_to_num(noise_color)),
        )

    # ── Erweiterte Feature-Extraktion (aus MediumClassifier portiert) ───

    @staticmethod
    def _rotation_periodicity(mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Detect turntable/disc rotation periodicity via autocorrelation.

        Returns (rotation_hz, rotation_strength).
        Vinyl: 33⅓ RPM → 0.556 Hz, 45 RPM → 0.750 Hz, 78 RPM → 1.300 Hz.
        """
        n = len(mono)
        if n < sr * 4:  # need ≥ 4 s for sub-Hz periodicity
            return 0.0, 0.0

        from scipy.signal import decimate

        # Envelope via RMS in 50 ms windows
        win_samples = max(1, int(sr * 0.05))
        n_frames = n // win_samples
        if n_frames < 20:
            return 0.0, 0.0
        rms = np.sqrt(np.mean(mono[: n_frames * win_samples].reshape(n_frames, win_samples) ** 2, axis=1) + 1e-12)
        env_sr = sr / win_samples

        # Downsample envelope to ~20 Hz
        dec_factor = max(1, int(env_sr / 20))
        if dec_factor > 1 and len(rms) > dec_factor * 10:
            rms_dec = decimate(rms.astype(np.float64), dec_factor)
            dec_sr = env_sr / dec_factor
        else:
            rms_dec = rms.astype(np.float64)
            dec_sr = env_sr

        # Autocorrelation
        rms_dec = rms_dec - np.mean(rms_dec)
        n_dec = len(rms_dec)
        if n_dec < 10:
            return 0.0, 0.0
        acf = np.correlate(rms_dec, rms_dec, mode="full")
        acf = acf[n_dec - 1 :]  # positive lags only
        acf_norm = acf / (acf[0] + 1e-12)

        # Search for peaks in 0.4–1.5 Hz range (turntable rotation)
        min_lag = max(1, int(dec_sr / 1.5))
        max_lag = min(n_dec - 1, int(dec_sr / 0.4))
        if max_lag <= min_lag:
            return 0.0, 0.0

        search = acf_norm[min_lag : max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_val = float(search[peak_idx])
        rot_hz = float(dec_sr / (min_lag + peak_idx)) if (min_lag + peak_idx) > 0 else 0.0

        return rot_hz, max(0.0, peak_val)

    @staticmethod
    def _infrasonic_rms(mono: np.ndarray, sr: int) -> float:
        """Measure infrasonic energy (< 20 Hz), strong in vinyl due to warp/eccentricity."""
        n = len(mono)
        fft_n = min(n, 131072)
        spec = np.abs(np.fft.rfft(mono[:fft_n], n=fft_n))
        freqs = np.fft.rfftfreq(fft_n, 1.0 / sr)
        mask_infra = freqs < 20.0
        mask_total = freqs < sr / 2
        e_infra = float(np.sum(spec[mask_infra] ** 2))
        e_total = float(np.sum(spec[mask_total] ** 2))
        return float(np.sqrt(e_infra / max(e_total, 1e-12)))

    @staticmethod
    def _codec_artifact_score(mono: np.ndarray, sr: int) -> tuple[float, float]:
        """Detect lossy codec artifacts (block artifacts, pre-echo).

        Returns (artifact_score, codec_type_code).
        codec_type_code: 0=none, 1=MP3, 2=AAC, 3=MiniDisc/ATRAC.
        """
        n = len(mono)
        block_sizes = [576, 1152, 1024, 512]  # MP3 short/long, AAC, ATRAC
        best_score = 0.0
        best_block = 0

        for bs in block_sizes:
            if n < bs * 10:
                continue
            n_blocks = n // bs
            blocks = mono[: n_blocks * bs].reshape(n_blocks, bs)
            boundary_e = np.mean(np.abs(np.diff(blocks, axis=0)[:, :4]) ** 2)
            mid_e = np.mean(np.abs(np.diff(blocks, axis=0)[:, bs // 2 : bs // 2 + 4]) ** 2)
            ratio = float(boundary_e / max(mid_e, 1e-12))
            if ratio > best_score:
                best_score = ratio
                best_block = bs

        artifact_score = float(np.clip((best_score - 1.0) / 2.0, 0.0, 1.0))

        # Pre-echo detection
        pre_echo_score = 0.0
        frame_ms = 10
        frame_n = max(1, int(sr * frame_ms / 1000))
        n_frames_pe = n // frame_n
        if n_frames_pe > 20:
            frame_e = np.sqrt(np.mean(mono[: n_frames_pe * frame_n].reshape(n_frames_pe, frame_n) ** 2, axis=1) + 1e-12)
            diff_e = np.diff(frame_e)
            jumps = np.where(diff_e > np.std(diff_e) * 2.0)[0]
            if len(jumps) > 2:
                pre_rises = 0
                for j in jumps:
                    if j >= 3 and all(diff_e[j - k] > 0 for k in range(1, min(4, j + 1))):
                        pre_rises += 1
                pre_echo_score = float(np.clip(pre_rises / max(len(jumps), 1), 0.0, 1.0))

        combined = float(np.clip(0.6 * artifact_score + 0.4 * pre_echo_score, 0.0, 1.0))

        codec_type = 0.0
        if combined > 0.15:
            if best_block in (576, 1152):
                codec_type = 1.0  # MP3
            elif best_block == 1024:
                codec_type = 2.0  # AAC
            elif best_block == 512:
                codec_type = 3.0  # MiniDisc/ATRAC

        return combined, codec_type

    @staticmethod
    def _crackle_density(mono: np.ndarray, sr: int) -> float:
        """Count impulsive crackle events per second (vinyl/shellac indicator)."""
        n = len(mono)
        duration_s = n / max(sr, 1)
        if duration_s < 0.5:
            return 0.0

        from scipy.signal import butter, sosfilt

        sos = butter(4, 2000.0, btype="high", fs=sr, output="sos")
        hp = sosfilt(sos, mono.astype(np.float64))

        threshold = 6.0 * float(np.median(np.abs(hp)) + 1e-12)
        impulses = np.abs(hp) > threshold

        min_gap = max(1, int(sr * 0.005))
        events = 0
        last_event = -min_gap
        for i in np.where(impulses)[0]:
            if i - last_event >= min_gap:
                events += 1
                last_event = i

        return events / max(duration_s, 0.1)

    @staticmethod
    def _snr(mono: np.ndarray, sr: int) -> float:
        """Estimate SNR in dB via signal vs noise-floor percentile."""
        n = len(mono)
        win = min(2048, n)
        hop = max(1, n // 200)
        frame_energies = []
        for start in range(0, n - win, hop):
            e = float(np.mean(mono[start : start + win] ** 2))
            if e > 0:
                frame_energies.append(e)
        if len(frame_energies) < 5:
            return 0.0
        arr = np.array(frame_energies)
        noise_e = float(np.percentile(arr, 5))
        signal_e = float(np.percentile(arr, 95))
        if noise_e <= 0:
            return 60.0
        return float(np.clip(10 * math.log10(signal_e / noise_e), 0.0, 80.0))

    @staticmethod
    def _noise_color(mono: np.ndarray, sr: int) -> float:
        """Estimate noise colour as spectral tilt (0=white, 1=pink, 2=brown/red).

        Uses 5th-percentile frames (noise floor) spectral slope.
        """
        n = len(mono)
        fft_n = min(n, 16384)
        win = np.hanning(fft_n).astype(np.float32)
        hop_nc = fft_n
        slopes: list[float] = []

        for start in range(0, n - fft_n, hop_nc):
            frame = mono[start : start + fft_n] * win
            mag = np.abs(np.fft.rfft(frame)) + 1e-12
            freqs = np.fft.rfftfreq(fft_n, 1.0 / sr)
            freqs = freqs[1:]
            mag = mag[1:]
            if len(freqs) < 10:
                continue
            log_f = np.log10(freqs + 1e-6)
            log_m = np.log10(mag)
            slope = float(np.polyfit(log_f, log_m, 1)[0])
            slopes.append(slope)

        if not slopes:
            return 1.0  # default pink
        slopes_sorted = sorted(slopes)
        n_quiet = max(1, len(slopes_sorted) // 5)
        avg_slope = float(np.mean(slopes_sorted[:n_quiet]))
        return float(np.clip(-avg_slope * 2.0, 0.0, 4.0))

    # ── Bayesian Material Scoring ──────────────────────────────────────

    def _bayesian_score(self, fp: SpectralFingerprint) -> dict[str, float]:
        """Compute posterior probabilities for all 16 material types via Gaussian log-likelihood.

        Returns dict[material_name → posterior_probability], sorted descending.
        """
        feature_vals = {
            "bandwidth_hz": fp.effective_bandwidth_hz,
            "snr_db": fp.snr_db,
            "noise_color": fp.noise_color,
            "crackle_density": fp.crackle_density,
            "wow_depth": fp.wow_flutter_index,
            "block_artifact": fp.codec_artifact_score,
            "pre_echo_ms": 0.0,
            "rotation_strength": fp.rotation_strength,
            "infrasonic_rms": fp.infrasonic_rms,
            "codec_type_code": fp.codec_type_code,
        }

        log_likes: dict[str, float] = {}
        for mat, params in self._MATERIAL_MODELS.items():
            ll = 0.0
            for feat_key in self._FEATURE_KEYS:
                mu, sigma = params[feat_key]
                sigma = max(sigma, 1e-6)
                x = feature_vals.get(feat_key, 0.0)
                ll -= 0.5 * ((x - mu) / sigma) ** 2 + math.log(sigma)
            log_likes[mat] = ll

        # Softmax normalization → posterior probabilities
        max_ll = max(log_likes.values())
        exp_vals = {k: math.exp(v - max_ll) for k, v in log_likes.items()}
        total = sum(exp_vals.values()) + 1e-12
        posteriors = {k: v / total for k, v in exp_vals.items()}

        return dict(sorted(posteriors.items(), key=lambda x: x[1], reverse=True))

    def detect(self, audio: np.ndarray, sr: int, *, file_ext: str = "") -> MediumDetectionResult:
        """Erkennt die Tonträgerkette forensisch via Bayesian-Fusion (§6.7 v9.10.97).

        Ablauf:
        1. Vollständiger Spektralfingerabdruck (13 Features)
        2. Bayesian Scoring über 16 Materialtypen
        3. §6.7b File-Extension Prior: digital formats → analog posteriors zeroed
        4. Primärmaterial = höchster Analog-Posterior (oder Digital)
        5. Codec-Layer-Erkennung (Block-Artefakte, Pre-Echo)
        6. Multi-Layer-Ketten-Inferenz (3+ Stufen möglich)
        7. ClassificationResult-Kompatibilitäts-Objekt für Passthrough

        Args:
            audio:    Input audio, float32/64, mono or stereo.
            sr:       Sample rate in Hz.
            file_ext: File extension of the source file (e.g. '.mp3', '.flac').
                      When provided, constrains the Bayesian prior so that analog
                      materials can never win for known-digital formats.

        Returns:
            MediumDetectionResult mit transfer_chain, bayesian_scores,
            classification_result für Downstream-Passthrough.
        """
        if sr != 48000:
            logger.debug("MediumDetector: SR=%d (erwartet 48000), arbeite trotzdem weiter", sr)

        fp = self._compute_fingerprint(audio, sr)
        posteriors = self._bayesian_score(fp)

        # §6.7b File-Extension Prior: digital file formats cannot originate from
        # analog physical media.  A .mp3 file was encoded digitally at capture time —
        # any analog artefacts are from the recording chain, not mechanical transport.
        # Zero out all analog material posteriors and renormalise before chain inference.
        _DIGITAL_FILE_EXTS: frozenset[str] = frozenset(
            {
                ".mp3",
                ".mp2",
                ".aac",
                ".m4a",
                ".ogg",
                ".opus",
                ".flac",
                ".wav",
                ".aiff",
                ".aif",
                ".wv",
                ".mpc",
                ".wma",
            }
        )
        _ext_lower = file_ext.lower()
        if _ext_lower in _DIGITAL_FILE_EXTS:
            _adjusted: dict[str, float] = {
                mat: (0.0 if mat in self._ANALOG_MATERIALS else score) for mat, score in posteriors.items()
            }
            _total = sum(_adjusted.values()) + 1e-12
            posteriors = dict(
                sorted(
                    {k: v / _total for k, v in _adjusted.items()}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            logger.info(
                "MediumDetector: file_ext=%s → analog posteriors zeroed; top-3 adjusted: %s",
                _ext_lower,
                ", ".join(f"{m}={s:.3f}" for m, s in list(posteriors.items())[:3]),
            )

        # Flag: analog Bayesian posteriors wurden durch file_ext genullt
        _analog_zeroed: bool = _ext_lower in _DIGITAL_FILE_EXTS

        chain: list[str] = []
        evidence: list[str] = []
        chain_confidences: list[float] = []

        # ── Phase 1: Primärmaterial bestimmen ──────────────────────────
        top_materials = list(posteriors.items())

        # Best analog material (physical source)
        best_analog: str | None = None
        best_analog_score = 0.0
        for mat, score in top_materials:
            if mat in self._ANALOG_MATERIALS and score >= self._ANALOG_POSTERIOR_MIN:
                best_analog = mat
                best_analog_score = score
                break

        # ── Phase 1b: Physical-feature analog inference ───────────────
        # Wenn file_ext die Bayesian-Analog-Posteriors genullt hat (z. B. *.mp3),
        # kann der Bayesian-Scorer keinen analogen Ursprungsträger mehr erkennen.
        # Physikalische Merkmale (Vinyl-Rumble, Kassetten-Wow/Flutter, Schellack-
        # Krackeln) überleben die gesamte Aufnahmekette und sind auch in Codec-
        # Dateien noch nachweisbar.  Diese Features werden genutzt, um den analogen
        # Ursprungsträger unabhängig vom Bayesian-Scoring zu rekonstruieren.
        _physical_analog_sources: list[tuple[str, float]] = []
        if _analog_zeroed and best_analog is None:
            _physical_analog_sources = self._infer_analog_source_from_fingerprint(fp)
            if _physical_analog_sources:
                best_analog, best_analog_score = _physical_analog_sources[0]
                logger.info(
                    "MediumDetector: physical-feature analog inference — "
                    "primary=%s (conf=%.3f) chain=%s "
                    "[crackle=%.4f infrasonic=%.4f rotation=%.3f wow_flutter=%.3f]",
                    best_analog,
                    best_analog_score,
                    [m for m, _ in _physical_analog_sources],
                    fp.crackle_density,
                    fp.infrasonic_rms,
                    fp.rotation_strength,
                    fp.wow_flutter_index,
                )

        # Best digital lossless
        best_digital: str | None = None
        best_digital_score = 0.0
        for mat, score in top_materials:
            if mat in self._DIGITAL_LOSSLESS and score >= self._ANALOG_POSTERIOR_MIN:
                best_digital = mat
                best_digital_score = score
                break

        # Best codec (lossy)
        best_codec: str | None = None
        best_codec_score = 0.0
        for mat, score in top_materials:
            if mat in self._CODEC_MATERIALS and score >= self._ANALOG_POSTERIOR_MIN:
                best_codec = mat
                best_codec_score = score
                break

        # Guard: check if source is clean digital (no analog chain)
        benign_codec = self._is_benign_codec_source(audio, sr, fp)

        # ── Codec-Hard-Gate: suppress implausible analog materials ────
        # If codec_type_code is clearly digital (≥ 0.5) AND no physical
        # rotation signature AND low crackle → analog classification is
        # implausible.  Override to benign_codec path.
        if (
            not benign_codec
            and fp.codec_type_code >= 0.5
            and fp.rotation_strength < 0.05
            and fp.crackle_density < 0.005
            and fp.wow_flutter_index < 0.02
        ):
            logger.info(
                "MediumDetector: codec_hard_gate override — codec=%.2f, "
                "rotation=%.3f, crackle=%.4f → treating as digital source",
                fp.codec_type_code,
                fp.rotation_strength,
                fp.crackle_density,
            )
            benign_codec = True

        # ── Phase 2: Kette aufbauen ───────────────────────────────────
        has_codec_artifacts = fp.codec_artifact_score > self._CODEC_ARTIFACT_THRESHOLD

        if benign_codec:
            if best_digital and best_digital_score > best_codec_score:
                chain.append(best_digital)
                chain_confidences.append(best_digital_score)
                evidence.append(f"Digitaler Träger: {best_digital} (posterior={best_digital_score:.3f})")
            elif best_codec:
                chain.append(best_codec)
                chain_confidences.append(best_codec_score)
                evidence.append(f"Codec-Quelle: {best_codec} (posterior={best_codec_score:.3f})")
            else:
                chain.append("cd_digital")
                chain_confidences.append(0.50)
                evidence.append("Clean digital — cd_digital default")
        else:
            if best_analog:
                chain.append(best_analog)
                chain_confidences.append(best_analog_score)
                evidence.append(
                    f"Primärquelle: {best_analog} "
                    f"(posterior={best_analog_score:.3f}, "
                    f"rotation={fp.rotation_strength:.3f}, "
                    f"infrasonic={fp.infrasonic_rms:.4f}, "
                    f"crackle={fp.crackle_density:.4f})"
                )

                # Build a deeper analog chain from Bayesian + physical cues.
                # This allows multi-generation chains beyond one secondary layer.
                _candidate_scores: dict[str, float] = {}
                _candidate_sources: dict[str, str] = {}
                for mat2, score2 in top_materials:
                    if mat2 == best_analog or mat2 not in self._ANALOG_MATERIALS or score2 < self._SECONDARY_ANALOG_MIN:
                        continue
                    _candidate_scores[mat2] = float(score2)
                    _candidate_sources[mat2] = "posterior"

                for phys_mat, phys_conf in _physical_analog_sources[1:]:
                    if phys_mat == best_analog:
                        continue
                    _prev = _candidate_scores.get(phys_mat)
                    if _prev is None or phys_conf > _prev:
                        _candidate_scores[phys_mat] = float(phys_conf)
                        _candidate_sources[phys_mat] = "physical"
                    else:
                        _candidate_sources[phys_mat] = _candidate_sources.get(phys_mat, "posterior") + "+physical"

                _analog_depth = 1
                _last_order = self._MEDIUM_ORDER.get(best_analog, 0)
                for mat2 in sorted(
                    _candidate_scores,
                    key=lambda _m: (self._MEDIUM_ORDER.get(_m, 99), -_candidate_scores[_m]),
                ):
                    if _analog_depth >= self._MAX_ANALOG_CHAIN_DEPTH:
                        break
                    if mat2 in chain:
                        continue

                    _order2 = self._MEDIUM_ORDER.get(mat2, 99)
                    _score2 = float(_candidate_scores[mat2])
                    _src2 = _candidate_sources.get(mat2, "posterior")

                    # Keep chain order causal: no backward jumps.
                    if _order2 < _last_order:
                        continue

                    # Same-order links are only accepted with stronger evidence.
                    if _order2 == _last_order and _score2 < self._SAME_ORDER_ANALOG_MIN:
                        continue

                    chain.append(mat2)
                    chain_confidences.append(_score2)
                    evidence.append(f"Sekundäre Analog-Stufe ({_src2}): {mat2} (conf={_score2:.3f})")
                    _analog_depth += 1
                    _last_order = _order2

                # Optional digital lossless intermediate (e.g. vinyl→tape→cd_digital→mp3).
                if (
                    best_digital
                    and best_digital not in chain
                    and self._MEDIUM_ORDER.get(best_digital, 99) >= _last_order
                    and best_digital_score >= self._ANALOG_POSTERIOR_MIN
                ):
                    chain.append(best_digital)
                    chain_confidences.append(best_digital_score)
                    evidence.append(f"Digitale Zwischenstufe: {best_digital} (posterior={best_digital_score:.3f})")
                    _last_order = self._MEDIUM_ORDER.get(best_digital, _last_order)

            # ── Codec-Layer anhängen ──────────────────────────────────
            if has_codec_artifacts or (
                fp.hf_energy_above_16k < self.HF_ENERGY_THRESHOLD_FRACTION and fp.effective_bandwidth_hz < 17_500
            ):
                if best_codec:
                    codec_name = best_codec
                    codec_conf = best_codec_score
                elif fp.effective_bandwidth_hz < 14_000:
                    codec_name = "mp3_low"
                    codec_conf = 0.40
                else:
                    codec_name = "mp3_high"
                    codec_conf = 0.35
                if codec_name not in chain:
                    chain.append(codec_name)
                    chain_confidences.append(codec_conf)
                    evidence.append(
                        f"Codec-Stufe: {codec_name} "
                        f"(artifact={fp.codec_artifact_score:.3f}, "
                        f"eff_bw={fp.effective_bandwidth_hz:.0f} Hz)"
                    )

        # ── Fallback ─────────────────────────────────────────────────
        if not chain:
            top_mat, top_score = top_materials[0] if top_materials else ("unknown", 0.30)
            if top_mat == "unknown" or top_score < 0.05:
                chain = ["unknown"]
                chain_confidences = [0.30]
                evidence.append("Träger unbekannt — Bayesian-Scores zu niedrig")
            else:
                chain = [top_mat]
                chain_confidences = [top_score]
                evidence.append(f"Bayesian-Fallback: {top_mat} (posterior={top_score:.3f})")

        # §6.1 [RELEASE_MUST] Material-Key-Normalisierung (v9.10.101):
        # MediumDetector interne Bayesian-Schlüssel → SUPPORTED_MATERIALS-konforme Schlüssel.
        # Betrifft alle Elemente der transfer_chain, nicht nur primary.
        _normalized_chain: list[str] = []
        _normalized_confidences: list[float] = []
        for _mat, _conf in zip(chain, chain_confidences):
            _norm = self._normalize_material_key(_mat)
            _safe_conf = float(np.clip(_conf, 0.0, 1.0))
            if _normalized_chain and _normalized_chain[-1] == _norm:
                _normalized_confidences[-1] = max(_normalized_confidences[-1], _safe_conf)
                continue
            _normalized_chain.append(_norm)
            _normalized_confidences.append(_safe_conf)
        chain = _normalized_chain
        chain_confidences = _normalized_confidences

        primary = chain[0]
        is_multi = len(chain) > 1
        # Weakest-link principle: a transfer chain is only as confident as its
        # least certain component.  Using sum() caused multi-link chains (e.g.
        # vinyl → mp3_low with 0.65 + 0.40 = 1.05) to always clip at 1.0 → 5 stars.
        confidence = float(np.clip(min(chain_confidences) if chain_confidences else 0.0, 0.0, 1.0))

        # ── ClassificationResult für Passthrough bauen ───────────────
        try:
            from backend.core.medium_classifier import ClassificationResult

            classification_result = ClassificationResult(
                material=primary,
                confidence=confidence,
                bandwidth_hz=fp.effective_bandwidth_hz,
                snr_db=fp.snr_db,
                noise_color=fp.noise_color,
                crackle_density=fp.crackle_density,
                wow_flutter_hz=fp.wow_flutter_index,
                block_artifact=fp.codec_artifact_score,
                rotation_hz=fp.rotation_hz,
                rotation_strength=fp.rotation_strength,
                infrasonic_rms=fp.infrasonic_rms,
                codec_type="mp3" if fp.codec_type_code >= 0.5 else "clean",
                classifier_source="bayesian_fusion",
            )
        except (ImportError, TypeError):
            classification_result = None

        logger.info(
            "MediumDetector: Kette=%s, primär=%s, multi=%s, Konfidenz=%.2f, Top-3 Bayesian: %s",
            " → ".join(chain),
            primary,
            is_multi,
            confidence,
            ", ".join(f"{m}={s:.3f}" for m, s in list(posteriors.items())[:3]),
        )

        result = MediumDetectionResult(
            transfer_chain=chain,
            is_multi_generation=is_multi,
            primary_material=primary,
            confidence=confidence,
            spectral_fingerprint=fp,
            evidence=evidence,
            medium_confidences=chain_confidences,
            bayesian_scores=posteriors,
            classification_result=classification_result,
        )

        # §6.7 Dolby / DBX NR detection for tape-chain material
        _tape_types = {"tape", "reel_tape", "wire_recording"}
        if primary in _tape_types or any(m in _tape_types for m in chain):
            try:
                from backend.core.dolby_nr_detector import get_dolby_nr_detector as _get_dolby

                _dolby_det = _get_dolby().detect(audio, sr, material_type=primary)
                if _dolby_det.detected:
                    result.dolby_nr_type = _dolby_det.nr_type
                    result.dolby_nr_confidence = _dolby_det.confidence
                    result.evidence.append(
                        f"Dolby/DBX NR detected: {_dolby_det.nr_type} "
                        f"({_dolby_det.confidence:.0%} konfident, "
                        f"HF-Überschuss {_dolby_det.hf_excess_db:.1f} dB)"
                    )
                    logger.info(
                        "MediumDetector: Dolby NR detected type=%s conf=%.2f hf_excess=%.1f dB",
                        _dolby_det.nr_type,
                        _dolby_det.confidence,
                        _dolby_det.hf_excess_db,
                    )
            except Exception as exc:
                logger.debug("MediumDetector: Dolby NR detection skipped (%s)", exc)

        return result

    # ── Hilfsmethoden ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_material_key(key: str) -> str:
        """Map internal Bayesian-Scorer keys to canonical SUPPORTED_MATERIALS keys (§6.1).

        The Bayesian scorer uses some internal identifiers that differ from the
        SUPPORTED_MATERIALS list consumed by UnifiedRestorerV3 / DefectScanner.
        This method ensures all keys leaving detect() are spec-compliant.

        Mapping (spec §6.1 kanonisch):
            cassette         → tape           (Compact Cassette Typ I/II/IV)
            reel_wire        → wire_recording (Drahtband 1940–1955)
            cassette_digital → dat            (Digitalkassette; DAT-Pfad)
            vhs_audio        → tape           (VHS-Tonspur)
            composite        → composite (unchanged — caller must use transfer_chain[0])
        """
        _KEY_MAP: dict[str, str] = {
            "cassette": "tape",
            "reel_wire": "wire_recording",
            "cassette_digital": "dat",
            "vhs_audio": "tape",
        }
        normalized = _KEY_MAP.get(key, key)
        if normalized != key:
            logger.debug(
                "MediumDetector._normalize_material_key: '%s' → '%s'",
                key,
                normalized,
            )
        return normalized

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Wandelt beliebiges Audio in mono float32 um."""
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
        mono = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mono, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: MediumDetector | None = None
_lock = threading.Lock()


def get_medium_detector() -> MediumDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MediumDetector()
    return _instance


def detect_medium_chain(audio: np.ndarray, sr: int) -> MediumDetectionResult:
    """Convenience-Wrapper: erkennt die Tonträgerkette eines Audio-Signals."""
    return get_medium_detector().detect(audio, sr)
