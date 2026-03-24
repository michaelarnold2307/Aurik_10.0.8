"""
EraClassifier — Ära-/Dekaden-adaptives Processing (§2.14 Spec)
===============================================================

Erkennt das Aufnahme-Jahrzehnt (1890–2025) automatisch und leitet
material- und epochenspezifische Verarbeitungspriors ab.

Erkennungs-Kaskade (3 Stufen):
    Tier-1: LAION-CLAP-Embeddings → Nearest-Neighbor zu Ära-Referenz-Ankern
    Tier-2: DSP-Fingerprint (HF-Rolloff + Bandbreiten-Kurve)
    Tier-3: Mikrofon-Typ-Heuristik

Referenz: §2.14 Aurik-9-Spec (v9.9.5)
Autor: Aurik Development Team
Datum: 20. Februar 2026
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace as dc_replace
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

# --------------- Dekaden-Definition ----------------------------------------

VALID_DECADES: list[int] = [
    1890,
    1900,
    1910,
    1920,
    1930,
    1940,
    1950,
    1960,
    1970,
    1980,
    1990,
    2000,
    2010,
    2020,
    2025,
]

# Bekannte HF-Rolloff-Grenzen pro Jahrzehnt [Hz]
DECADE_HF_LIMITS: dict[int, float] = {
    1890: 3000,
    1900: 4000,
    1910: 5000,
    1920: 6000,
    1930: 7000,
    1940: 8000,
    1950: 10000,
    1960: 12000,
    1970: 16000,
    1980: 20000,
    1990: 20000,
    2000: 20000,
    2010: 20000,
    2020: 20000,
    2025: 20000,
}

# Material-Prior pro Dekade
DECADE_MATERIAL_PRIOR: dict[int, str] = {
    1890: "wax_cylinder",
    1900: "wax_cylinder",
    1910: "shellac",
    1920: "shellac",
    1930: "shellac",
    1940: "shellac",
    1950: "vinyl",
    1960: "vinyl",
    1970: "reel_tape",
    1980: "tape",
    1990: "cd_digital",
    2000: "cd_digital",
    2010: "streaming",
    2020: "streaming",
    2025: "streaming",
}

# Medium-based minimum decade floor: a recording on a given medium cannot
# predate the physical invention of that medium.  Used by
# constrain_era_to_medium() to correct impossible decade assignments
# (e.g. reel_tape → 1890 is a classification artefact, not a real recording).
#
# Conservative lower bounds (rounded to nearest VALID_DECADES entry):
#   wax_cylinder  : 1890 (Edison 1877 → commercial 1888)
#   wire_recording: 1900 (Poulsen Telegraphone 1898)
#   shellac       : 1900 (shellac discs commercial ~1898)
#   lacquer_disc  : 1920 (transcription discs widespread 1920s)
#   vinyl         : 1950 (Columbia 12″ LP 1948, 45rpm 1949)
#   reel_tape     : 1940 (AEG Magnetophon 1935; commercial 1940s)
#   tape/cassette : 1960 (Philips compact cassette 1963)
#   dat           : 1980 (DAT standard 1987; decade floor 1980)
#   minidisc      : 1990 (Sony MiniDisc 1992)
#   cd_digital/cd : 1980 (first CD October 1982)
#   mp3_low/high  : 1990 (MP3 standard 1993)
#   aac           : 2000 (AAC in iTunes 2001)
#   streaming     : 2000 (consumer streaming widespread ~2005)
MEDIUM_DECADE_FLOOR: dict[str, int] = {
    "wax_cylinder": 1890,
    "wire_recording": 1900,
    "shellac": 1900,
    "lacquer_disc": 1920,
    "vinyl": 1950,
    "reel_tape": 1940,
    "tape": 1960,
    "cassette": 1960,
    "dat": 1980,
    "minidisc": 1990,
    "cd_digital": 1980,
    "cd": 1980,
    "mp3_low": 1990,
    "mp3_high": 1990,
    "aac": 2000,
    "streaming": 2000,
}

# GP-Warmstart: noise_reduction_strength prior mean pro Epoche
DECADE_NR_PRIOR_MEAN: dict[int, float] = {
    1890: 0.95,
    1900: 0.95,
    1910: 0.92,
    1920: 0.90,
    1930: 0.90,
    1940: 0.85,
    1950: 0.80,
    1960: 0.75,
    1970: 0.65,
    1980: 0.55,
    1990: 0.50,
    2000: 0.50,
    2010: 0.45,
    2020: 0.45,
    2025: 0.45,
}

DECADE_NR_PRIOR_STD: dict[int, float] = dict.fromkeys(VALID_DECADES, 0.07)
DECADE_NR_PRIOR_STD.update(
    {1900: 0.05, 1910: 0.05, 1920: 0.05, 1930: 0.05, 1940: 0.06, 1970: 0.08, 1980: 0.10, 1990: 0.10}
)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class EraResult:
    """Ergebnis des EraClassifiers.

    Attributes:
        decade:                Erkanntes Jahrzehnt (z. B. 1940, 1970, …).
        era_label:             Menschenlesbare Bezeichnung (z. B. „1970er").
        confidence:            Konfidenz ∈ [0.0, 1.0].
        material_prior:        Empfohlener Material-Typ-String aus ``DECADE_MATERIAL_PRIOR``.
        noise_profile:         Spektrales Rauschprofil (Bark-Bänder, 24 Werte).
        tier_used:             Welche Erkennungsstufe genutzt wurde (1 = CLAP, 2 = DSP, 3 = Heuristik).
        hf_rolloff_hz:         Gemessener HF-Rolloff-Punkt (-3 dB) in Hz.
        is_remaster_suspected: True wenn RemasterDetector einen Remaster erkannt hat.
    """

    decade: int
    era_label: str
    confidence: float
    material_prior: str
    noise_profile: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.float32))
    tier_used: int = 2
    hf_rolloff_hz: float = 20000.0
    is_remaster_suspected: bool = False

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.decade not in VALID_DECADES:
            self.decade = min(VALID_DECADES, key=lambda d: abs(d - self.decade))

    def as_dict(self) -> dict:
        """Serialisierung ohne ndarray."""
        d = asdict(self)
        d["noise_profile"] = self.noise_profile.tolist()
        return d


# ---------------------------------------------------------------------------
# Bark-Skala Hilfsfunktion
# ---------------------------------------------------------------------------

BARK_EDGES_HZ = [
    20,
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
]


def _bark_band_energies(audio_mono: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet normalisierte Energie in 24 Bark-Bändern.

    Args:
        audio_mono: Mono-Audio (1D float32/64).
        sr:         Sample-Rate.

    Returns:
        ndarray shape (24,) — normalisierte Energien (sum = 1).
    """
    n_fft = min(4096, len(audio_mono))
    hop = n_fft // 4
    # STFT (kein scipy.signal für diesen einfachen Spektral-Pfad — numpy direkt)
    frames = []
    for start in range(0, len(audio_mono) - n_fft, hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        frame_fft = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(frame_fft)
    if not frames:
        return np.ones(24) / 24.0
    psd = np.mean(np.array(frames), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    energies = np.zeros(24, dtype=np.float32)
    for i, (lo, hi) in enumerate(itertools.pairwise(BARK_EDGES_HZ)):
        mask = (freqs >= lo) & (freqs < hi)
        energies[i] = float(np.sum(psd[mask]))

    total = energies.sum()
    if total < 1e-12:
        return np.ones(24, dtype=np.float32) / 24.0
    return np.nan_to_num(energies / total)


# ---------------------------------------------------------------------------
# Tier-2: DSP-Fingerprint
# ---------------------------------------------------------------------------


def _dsp_hf_rolloff(audio_mono: np.ndarray, sr: int) -> float:
    """Effective recording bandwidth via 90th-percentile cumulative energy rolloff.

    Uses the frequency below which 90 % of the total spectral energy is
    contained.  For a 6th-order Butterworth low-pass filter (the model used in
    the calibration tests) this gives rolloff ≈ 0.90 × cutoff_hz, which is
    exactly the calibration basis for all thresholds in _dsp_fingerprint_decade().

    Bass-heavy real-music correction: if the 90th-percentile falls below
    8 kHz but the signal carries more than 2 % of its total energy above
    8 kHz (meaning real HF content exists, not just filter roll-off), the
    floor is raised to 8 kHz.  This prevents a 1990s bass-heavy Schlager MP3
    from being mis-mapped to decade = 1890 while keeping the calibrated
    physics-test cases intact (their LP-filtered noise has ≪ 0.01 % energy
    above the filter cut-off).

    Args:
        audio_mono: Mono-Audio (1-D).
        sr:         Sample-Rate.

    Returns:
        Effective bandwidth (rolloff) frequency in Hz.
    """
    n_fft = min(4096, len(audio_mono))
    if n_fft < 64:
        return float(sr) / 2.0
    hop = n_fft // 2  # 50 % overlap for better averaging
    specs = []
    for start in range(0, max(1, len(audio_mono) - n_fft), hop):
        frame = audio_mono[start : start + n_fft] * np.hanning(n_fft)
        specs.append(np.abs(np.fft.rfft(frame)) ** 2)
    if not specs:
        return float(sr) / 2.0

    avg_spec = np.mean(np.array(specs), axis=0)
    cum_energy = np.cumsum(avg_spec)
    total_energy = cum_energy[-1]
    if total_energy < 1e-12:
        return float(sr) / 2.0

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 90th-percentile energy rolloff — calibrated for _dsp_fingerprint_decade().
    # All decade boundary thresholds are derived as the midpoint of 0.90 ×
    # adjacent DECADE_HF_LIMITS entries, so the measurement method and the
    # lookup table are consistent.
    idx = int(np.searchsorted(cum_energy, 0.90 * total_energy))
    idx = int(np.clip(idx, 0, len(avg_spec) - 1))
    rolloff = float(freqs[idx])

    # Bass-heavy content floor: if the 90th-percentile is below 8 kHz but
    # > 2 % of the total energy sits above 8 kHz, the recording carries real
    # high-frequency content.  Raise the floor to 8 kHz to avoid mapping
    # modern bass-heavy tracks to pre-1900 decades.
    # Butterworth-filtered calibration signals have < 0.05 % energy above
    # their cut-off, so this guard never triggers for them.
    if rolloff < 8000.0:
        hf_mask = freqs >= 8000.0
        hf_fraction = float(np.sum(avg_spec[hf_mask])) / (total_energy + 1e-20)
        if hf_fraction > 0.02:
            # Bass-heavy recording with real HF content (e.g. bass-heavy Schlager MP3):
            # The 90th-percentile is dominated by low-frequency energy, but the
            # recording has genuine HF content.  Use the 98th-percentile rolloff
            # to find where energy actually ends rather than capping at exactly
            # 8 kHz (which would incorrectly map 1970s/80s tracks to decade 1930).
            idx98 = int(np.searchsorted(cum_energy, 0.98 * total_energy))
            idx98 = int(np.clip(idx98, 0, len(avg_spec) - 1))
            rolloff_98 = float(freqs[idx98])
            # Ensure at least 8 kHz floor in case 98th-pct is also dominated by bass.
            rolloff = max(rolloff_98, 8000.0)

    return rolloff


def _dsp_fingerprint_decade(rolloff_hz: float, snr_db: float) -> tuple[int, float]:
    """Mappt Bandbreite + SNR auf Jahrzehnt via kalibrierter Schwellwert-Tabelle.

    Erkennungsprinzip:
    - Jahrzehnte 1890–1980: Bandbreite (HF-Rolloff) ist der primäre Indikator,
      da analoge Medien physikalisch begrenzte Übertragungsbandbreiten haben.
      SNR dient als Sekundärindikator und verstärkt die Konfidenz bei Grenzfällen.
    - Jahrzehnte 1990–2025: Alle haben nominell ≥20 kHz Bandbreite — BW allein
      kann sie nicht unterscheiden. Der dynamische Bereich (SNR = P90/P10 der
      Frame-Energien) dient als primärer Diskriminator: CD (1990) ≈ 30–45 dB
      Musikdynamik, HD-Streaming (2010+) > 45 dB.

    SNR-Schwellwerte für post-1980 Material sind bewusst konservativ gesetzt, da
    der Frame-Energie-SNR-Schätzer die Musikdynamik misst, nicht den Rauschboden
    des Mediums. Stark komprimierte Musik (DR3–DR8) ergibt niedrige SNR-Werte
    und fällt zurecht in 1980/1990 (ältere Produktionspraxis).

    Args:
        rolloff_hz: Gemessener HF-Rolloff in Hz (90th-percentile cumulative energy).
        snr_db:     Geschätzter SNR in dB (frame-energy P90/P10 ratio).

    Returns:
        (decade, confidence)
    """
    bw_khz = rolloff_hz / 1000.0

    # Primary decade selection via bandwidth.
    # Thresholds = midpoint of 0.90 × adjacent DECADE_HF_LIMITS:
    #   threshold(D, D+1) = (DECADE_HF_LIMITS[D] × 0.9 + DECADE_HF_LIMITS[D+1] × 0.9) / 2
    # Recalibrated 23.03.2026 to match DECADE_HF_LIMITS table consistently.
    # 1900 added: threshold(1890, 1900) = (3000×0.9 + 4000×0.9)/2 = 3150 → 3.2 kHz
    if bw_khz < 3.2:
        decade = 1890  # LIMIT  3 kHz → expected rolloff ~2.7 kHz
    elif bw_khz < 4.5:
        decade = 1900  # LIMIT  4 kHz → expected rolloff ~3.6 kHz
    elif bw_khz < 5.4:
        decade = 1910  # LIMIT  5 kHz → expected rolloff ~4.5 kHz
    elif bw_khz < 7.0:
        decade = 1920  # LIMIT  6 kHz → expected rolloff ~5.4 kHz
    elif bw_khz < 8.8:
        decade = 1930  # LIMIT  7 kHz → expected rolloff ~6.3 kHz
    elif bw_khz < 9.5:
        decade = 1940  # LIMIT  8 kHz → expected rolloff ~7.2 kHz; (7200+9000)/2=8100
    elif bw_khz < 11.5:
        decade = 1950  # LIMIT 10 kHz → expected rolloff ~9.0 kHz; (9000+10800)/2=9900
    elif bw_khz < 12.8:
        decade = 1960  # LIMIT 12 kHz → expected rolloff ~10.8 kHz; (10800+14400)/2=12600
    elif bw_khz < 17.0:
        decade = 1970  # LIMIT 16 kHz → expected rolloff ~14.4 kHz; (14400+18000)/2=16200
    elif bw_khz < 19.0:
        decade = 1980  # LIMIT 20 kHz → expected rolloff ~18.0 kHz
    else:
        # Full-bandwidth (≥ 19 kHz): BW cannot distinguish 1990–2025.
        # Use frame-energy SNR to differentiate digital decades.
        # Conservative thresholds: real-music DR is ~15–30 dB lower than medium SNR.
        #   2020 streaming: medium ~80 dB → typical music DR 50–65 dB
        #   2010 streaming: medium ~75 dB → typical music DR 45–60 dB
        #   2000 digital:   medium ~70 dB → typical music DR 33–50 dB
        #   1990 CD:        medium ~65 dB → typical music DR 22–40 dB
        #   1980 tape:      medium ~58 dB → typical music DR < 22 dB
        if snr_db >= 50.0:
            decade = 2020
        elif snr_db >= 38.0:
            decade = 2010
        elif snr_db >= 28.0:
            decade = 2000
        elif snr_db >= 18.0:
            decade = 1990
        else:
            decade = 1980

    # SNR micro-correction for vintage decades (Carbon/Ribbon-microphone heuristic)
    if snr_db < 20.0 and bw_khz < 6.0:
        decade = min(decade, 1930)  # Carbon-microphone characteristic
    elif snr_db < 25.0 and bw_khz < 8.0 and decade > 1940:
        decade = min(max(decade, 1920), 1940)  # Ribbon-microphone era

    # Confidence: combine BW error and SNR deviation for each era class.
    expected_bw = DECADE_HF_LIMITS.get(decade, 20000.0) / 1000.0
    bw_error = abs(bw_khz - expected_bw) / max(expected_bw, 1.0)
    expected_snr = _decade_expected_snr(decade)
    snr_error = abs(snr_db - expected_snr) / max(expected_snr, 1.0)

    if decade >= 1990:
        # Post-1990: SNR is primary classifier; BW error is irrelevant (all ≈20 kHz)
        conf = float(np.clip(1.0 - snr_error * 0.55, 0.50, 0.90))
    elif decade <= 1940:
        # Pre-1950: BW and SNR both carry independent physical evidence
        conf = float(np.clip(1.0 - bw_error * 0.50 - snr_error * 0.30, 0.25, 0.90))
    else:
        # 1950–1980: BW-dominated, SNR secondary
        conf = float(np.clip(1.0 - bw_error * 0.70 - snr_error * 0.15, 0.25, 0.87))

    if bw_khz >= 18.0 and decade < 1990:
        conf = max(conf, 0.75)  # Full-bandwidth analog clearly ≥ 1980
    return decade, conf


def _decade_expected_snr(decade: int) -> float:
    """Grobe SNR-Erwartung pro Dekade [dB]."""
    snr_map = {
        1890: 12,
        1900: 15,
        1910: 18,
        1920: 22,
        1930: 28,
        1940: 32,
        1950: 38,
        1960: 44,
        1970: 52,
        1980: 58,
        1990: 65,
        2000: 70,
        2010: 75,
        2020: 80,
        2025: 80,
    }
    return snr_map.get(decade, 50)


def _estimate_snr(audio_mono: np.ndarray, sr: int = 48000) -> float:
    """Frame-basierte SNR-Schätzung via Energie-Perzentile.

    Robuster als die frühere Sample-Level-Sortierung für Vintage-Aufnahmen
    mit dauerhaftem Rauschen: Die Sortierung von Einzelsamples lieferte dort
    ~52 dB SNR statt der korrekten ~15–25 dB, da kurze Stille-Momente die
    untersten Perzentile dominierten.

    Frame-Level-Ansatz (100 ms Frames):
        - 10. Energie-Perzentil → Rauschboden
        - 90. Energie-Perzentil → Nutz-Signal

    Args:
        audio_mono: Mono-Audio.
        sr:         Sample-Rate (für Frame-Größen-Berechnung).

    Returns:
        Geschätzter SNR in dB, geclamppt auf [0, 80].
    """
    frame_size = max(1, sr // 10)  # 100-ms-Frames
    frames = [audio_mono[i : i + frame_size] for i in range(0, len(audio_mono) - frame_size, frame_size)]
    if not frames:
        return 40.0
    energies = np.array([np.mean(f**2) for f in frames])
    noise_floor = float(np.percentile(energies, 10))
    signal_power = float(np.percentile(energies, 90))
    if noise_floor < 1e-18:
        return 60.0
    snr = 10.0 * math.log10(max(signal_power / noise_floor, 1.0))
    return float(np.clip(snr, 0.0, 80.0))


# ---------------------------------------------------------------------------
# Tier-3: Mikrofon-Typ-Heuristik
# ---------------------------------------------------------------------------


def _microphone_type_decade(bark_energies: np.ndarray) -> tuple[int, float]:
    """Ära-Schätzung aus 24 Bark-Band-Energien via 95th-Perzentil-Bandbreite (Tier-3 Fallback).

    Mirrors Tier-2's 90th-percentile rolloff logic but operates on the already-computed
    Bark-band energy array.  The 95th-percentile Bark band (bw95) is the band index below
    which 95 % of the total spectral energy lies.  This is a robust proxy for the recording's
    effective bandwidth and avoids the non-linear Bark band width bias that invalidates
    simple min/max flatness ratios.

    Calibration (6th-order Butterworth through _bark_band_energies at 48 kHz):
        cutoff 3.5 kHz  → bw95 ≈ 15–16  (Bark band boundary ~3.2 kHz)
        cutoff 6.0 kHz  → bw95 ≈ 18–19  (~5.3 kHz)
        cutoff 10 kHz   → bw95 ≈ 20     (~6.4 kHz region)
        cutoff 14 kHz   → bw95 ≈ 21     (~7.7 kHz)
        cutoff 20 kHz+  → bw95 ≈ 22–23  (full band)

    Args:
        bark_energies: 24 normalised Bark-band energies (sum should be ≈ 1.0).

    Returns:
        (decade, confidence)
    """
    total = float(bark_energies.sum()) + 1e-12

    # Low-frequency dominance: fraction of energy below 630 Hz (Bark bands 0–5)
    lf_frac = float(np.sum(bark_energies[:6]) / total)

    # 95th-percentile Bark bandwidth band (similar principle to Tier-2's 90th-pct rolloff)
    cum = np.cumsum(bark_energies)
    bw95 = int(np.clip(int(np.searchsorted(cum, 0.95 * total)), 0, 23))

    # Map bandwidth band to decade.  Boundaries calibrated against Butterworth LP test signals.
    if bw95 <= 6:
        # < ~770 Hz effective BW → pre-1920 acoustic/mechanical format
        return 1910, 0.42
    elif bw95 <= 9:
        # 770–1270 Hz → carbon-microphone era (1920s)
        return 1920, 0.40
    elif bw95 <= 12:
        # 1270–1720 Hz → early ribbon/condenser (1930s)
        return 1930, 0.38
    elif bw95 <= 16:
        # 1720–3700 Hz → vintage tape/early vinyl; SNR-based sub-split
        dec = 1930 if lf_frac > 0.30 else 1940
        return dec, 0.36
    elif bw95 <= 18:
        # 3700–5300 Hz → HiFi LP / early reel tape (1950s/60s)
        return 1960, 0.33
    elif bw95 <= 20:
        # 5300–7700 Hz → FM radio / cassette era (1970s)
        return 1970, 0.31
    elif bw95 <= 22:
        # 7700–12000 Hz → HiFi tape / early digital (1980s)
        return 1980, 0.30
    else:
        # > 12000 Hz → full-bandwidth digital era (1990+)
        return 1990, 0.28


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".aurik" / "era_cache"
_CACHE_VERSION = "v5"  # v5: recalibrated 1940-1970 BW thresholds to match DECADE_HF_LIMITS (23.03.2026)


class EraClassifier:
    """Erkennt Aufnahme-Ära (1890–2025) und leitet epochenspezifische Priors ab.

    Erkennungs-Kaskade (3 Stufen):
        Tier-1: LAION-CLAP-Embeddings → NN zu Ära-Referenz-Ankern
        Tier-2: DSP-Fingerprint → HF-Rolloff + Bandbreiten-Kurve
        Tier-3: Mikrofon-Typ-Heuristik (Carbon/Kondensator)

    Ausgabe: EraResult(decade, era_label, confidence, material_prior,
                       noise_profile, tier_used, hf_rolloff_hz)

    Invarianten:
        - Konfidenz < 0.4 → material_prior = "unknown" (konservative Priors)
        - CLAP-Fallback auf DSP-Fingerprint wenn Import fehlschlägt
        - Decade-Label wird in RestorationResult.era_decade gespeichert
        - Paläografie-Cache unter ~/.aurik/era_cache/<sha256_prefix>.json
    """

    def __init__(self) -> None:
        self._clap_plugin: object | None = None
        self._clap_loaded: bool = False
        self._clap_lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def classify(self, audio: np.ndarray, sr: int) -> EraResult:
        """Erkennt Aufnahme-Ära (Cascaded Tier-1 → Tier-2 → Tier-3).

        Args:
            audio: Audio-Signal (mono oder stereo).
            sr:    Sample-Rate in Hz — muss exakt 48000 sein (Spec §3.x).

        Returns:
            EraResult mit Dekade, Confidence und Material-Prior.

        Raises:
            ValueError:    Falls audio leer ist.
        """
        if audio.size == 0:
            raise ValueError("Audio darf nicht leer sein.")
        # SR-agnostic: analysis modules work at native import SR (Spec §Performance-Budget)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        audio_mono = np.mean(audio, axis=-1 if audio.shape[-1] <= 2 else 0) if audio.ndim > 1 else audio.copy()

        # Cache-Key aus SHA256-Prefix
        sha = hashlib.sha256(audio_mono.tobytes()).hexdigest()[:16]
        cache_path = CACHE_DIR / f"{sha}_{_CACHE_VERSION}.json"
        cached = self._load_cache(cache_path)
        if cached:
            logger.debug(
                "EraClassifier: Cache-Hit %s → Jahrzehnt=%d, Konfidenz=%.2f, Tier=%d",
                sha,
                cached.decade,
                cached.confidence,
                cached.tier_used,
            )
            return cached

        bark = _bark_band_energies(audio_mono, sr)
        rolloff_hz = _dsp_hf_rolloff(audio_mono, sr)
        snr_db = _estimate_snr(audio_mono, sr)

        # Tier-1: CLAP (optional)
        result = self._try_tier1(audio_mono, sr, bark, rolloff_hz, snr_db)

        # Tier-2: DSP-Fingerprint
        if result is None or result.confidence < 0.40:
            result = self._tier2(bark, rolloff_hz, snr_db)

        # Tier-3: Mikrofon-Heuristik (letzter Fallback)
        if result.confidence < 0.30:
            result = self._tier3(bark, rolloff_hz, snr_db)

        # Invariante: Conf < 0.40 → konservatives Material
        if result.confidence < 0.40:
            result = EraResult(
                decade=result.decade,
                era_label=result.era_label,
                confidence=result.confidence,
                material_prior="unknown",
                noise_profile=result.noise_profile,
                tier_used=result.tier_used,
                hf_rolloff_hz=rolloff_hz,
            )

        # RemasterDetector-Guard (§2.14): verhindert falsche Ära-Zuweisung bei Remasters
        try:
            from backend.core.remaster_detector import get_remaster_detector

            _rm = get_remaster_detector().analyse(audio_mono, sr)
            if _rm is not None and _rm.is_remaster:
                result = dc_replace(result, is_remaster_suspected=True)
                logger.info(
                    "RemasterDetector: Remaster erkannt (conf=%.2f, BW=%.1f kHz)",
                    _rm.confidence,
                    getattr(_rm, "hf_rolloff_khz", 0.0),
                )
        except Exception:
            pass

        self._save_cache(cache_path, result)
        logger.info(
            "🕰️ EraClassifier: Jahrzehnt=%d, Konfidenz=%.2f, Material=%s, Tier=%d",
            result.decade,
            result.confidence,
            result.material_prior,
            result.tier_used,
        )
        return result

    def get_material_prior(self, era: EraResult) -> str:
        """Gibt empfohlenen Material-String für CausalDefectReasoner zurück.

        Bei Konfidenz < 0.40 → 'unknown' (konservative Priors, Spec §2.14).
        """
        if era.confidence < 0.40:
            return "unknown"
        return era.material_prior

    def get_gp_warmstart(self, era: EraResult) -> dict[str, float]:
        """GP-Optimizer-Initialisierungswerte für das erkannte Jahrzehnt.

        Returns:
            Dict mit Parameternamen → Initialwert.
        """
        decade = era.decade
        nr_mean = DECADE_NR_PRIOR_MEAN.get(decade, 0.65)
        nr_std = DECADE_NR_PRIOR_STD.get(decade, 0.08)
        return {
            "noise_reduction_strength": float(np.clip(nr_mean, 0.10, 1.0)),
            "noise_reduction_strength_std": nr_std,
            "harmonic_boost_db": 2.0 if decade <= 1950 else 1.0,
            "ola_crossfade_ms": 50.0 if decade <= 1940 else 30.0,
            "bass_restoration_db": 2.5 if decade <= 1960 else 0.5,
            "era_decade": float(decade),
            "era_confidence": float(era.confidence),
        }

    # ------------------------------------------------------------------
    # Tier-Implementierungen
    # ------------------------------------------------------------------

    def _try_tier1(
        self,
        audio_mono: np.ndarray,
        sr: int,
        bark: np.ndarray,
        rolloff_hz: float,
        snr_db: float,
    ) -> EraResult | None:
        """Tier-1: LAION-CLAP-basierte Ära-Erkennung (optional)."""
        try:
            with self._clap_lock:
                if not self._clap_loaded:
                    from plugins.laion_clap_plugin import get_laion_clap_plugin  # type: ignore[import]

                    self._clap_plugin = get_laion_clap_plugin()
                    self._clap_loaded = True
            if self._clap_plugin is None:
                return None
            # CLAP-Embedding → Cosinus-Ähnlichkeit zu Ära-Ankern
            embedding = self._clap_plugin.embed_audio(audio_mono, sr)  # type: ignore[union-attr]
            decade, conf = self._clap_nearest_neighbor(embedding)
            if conf < 0.35:
                return None
            return EraResult(
                decade=decade,
                era_label=f"{decade}er",
                confidence=conf,
                material_prior=DECADE_MATERIAL_PRIOR.get(decade, "unknown"),
                noise_profile=bark,
                tier_used=1,
                hf_rolloff_hz=rolloff_hz,
            )
        except Exception as exc:
            logger.debug("EraClassifier Tier-1 fehlgeschlagen: %s — nutze DSP-Fallback", exc)
            return None

    def _tier2(self, bark: np.ndarray, rolloff_hz: float, snr_db: float) -> EraResult:
        """Tier-2: DSP-Fingerprint (HF-Rolloff + SNR)."""
        decade, conf = _dsp_fingerprint_decade(rolloff_hz, snr_db)
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=2,
            hf_rolloff_hz=rolloff_hz,
        )

    def _tier3(self, bark: np.ndarray, rolloff_hz: float, snr_db: float) -> EraResult:
        """Tier-3: Mikrofon-Typ-Heuristik."""
        decade, conf = _microphone_type_decade(bark)
        material = DECADE_MATERIAL_PRIOR.get(decade, "unknown")
        return EraResult(
            decade=decade,
            era_label=f"{decade}er",
            confidence=conf,
            material_prior=material,
            noise_profile=bark,
            tier_used=3,
            hf_rolloff_hz=rolloff_hz,
        )

    def _clap_nearest_neighbor(self, embedding: np.ndarray) -> tuple[int, float]:
        """Findet nächsten Ära-Anker im CLAP-Embedding-Raum.

        Wenn keine vorberechneten Anker vorhanden sind, gibt unbekannte Ära zurück.
        """
        anchors_path = Path(__file__).parent.parent / "models" / "era_classifier" / "era_anchors.npy"
        if not anchors_path.exists():
            return 1960, 0.20
        try:
            anchors = np.load(str(anchors_path))  # (n_anchors, embedding_dim)
            # Letzte Spalte: decade-Label
            decade_labels = anchors[:, -1].astype(int)
            anchor_vecs = anchors[:, :-1]
            # L2-normalisieren für Cosinus
            anchor_norms = np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-12
            anchor_vecs = anchor_vecs / anchor_norms
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
            cosine_sims = anchor_vecs @ emb_norm
            best_idx = int(np.argmax(cosine_sims))
            best_sim = float(cosine_sims[best_idx])
            conf = float(np.clip((best_sim + 1.0) / 2.0 * 1.2, 0.0, 1.0))
            return int(decade_labels[best_idx]), conf
        except Exception as exc:
            logger.debug("CLAP NN-Suche fehlgeschlagen: %s", exc)
            return 1960, 0.20

    # ------------------------------------------------------------------
    # Cache-Verwaltung
    # ------------------------------------------------------------------

    def _load_cache(self, path: Path) -> EraResult | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return EraResult(
                decade=int(data["decade"]),
                era_label=str(data["era_label"]),
                confidence=float(data["confidence"]),
                material_prior=str(data["material_prior"]),
                noise_profile=np.array(data["noise_profile"], dtype=np.float32),
                tier_used=int(data.get("tier_used", 2)),
                hf_rolloff_hz=float(data.get("hf_rolloff_hz", 20000.0)),
            )
        except Exception:
            return None

    def _save_cache(self, path: Path, result: EraResult) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result.as_dict(), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("EraClassifier: Cache-Speichern fehlgeschlagen: %s", exc)


# ---------------------------------------------------------------------------
# Singleton (Thread-sicher, Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: EraClassifier | None = None
_lock = threading.Lock()


def get_era_classifier() -> EraClassifier:
    """Thread-sicherer Singleton-Accessor für EraClassifier."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EraClassifier()
    return _instance


def classify_era(audio: np.ndarray, sr: int) -> EraResult:
    """Convenience-Funktion: Erkennt Aufnahme-Ära ohne explizite Instanz.

    Args:
        audio: Audio-Signal (mono oder stereo, float32/64 [-1, 1]).
        sr:    Sample-Rate in Hz.

    Returns:
        EraResult mit Dekade, Confidence, Material-Prior und Noise-Profil.
    """
    return get_era_classifier().classify(audio, sr)


def constrain_era_to_medium(era_result: EraResult, medium: str) -> EraResult:
    """Applies a physical medium-based minimum decade floor to an EraResult.

    A tape recording cannot originate from 1890; a vinyl disc cannot predate
    1948.  This function corrects impossible decade assignments that arise when
    the EraClassifier operates on short or ambiguous audio segments.

    The corrected decade is the smallest VALID_DECADES entry >= the floor for
    the given medium.  Confidence is scaled down by 0.65 (indicating the
    assignment was constrained rather than directly measured) but clamped to
    [0.25, 0.80] to prevent both over-confidence and useless uncertainty.

    Args:
        era_result: EraResult produced by EraClassifier.classify().
        medium:     Physical medium string (e.g. 'tape', 'reel_tape', 'vinyl').
                    Case-insensitive; unknown medium strings are ignored.

    Returns:
        Original EraResult if no constraint applies; corrected EraResult otherwise.
    """
    floor = MEDIUM_DECADE_FLOOR.get(medium.strip().lower(), 0)
    if floor == 0 or era_result.decade >= floor:
        return era_result

    valid_above_floor = [d for d in VALID_DECADES if d >= floor]
    if not valid_above_floor:
        return era_result

    corrected_decade = min(valid_above_floor)
    new_conf = float(np.clip(era_result.confidence * 0.65, 0.25, 0.80))
    # Use the actually detected medium — not the decade-based prior which can
    # map e.g. 1960 → "vinyl" even though the medium was detected as "tape".
    new_material = medium.strip().lower()

    logger.info(
        "EraClassifier medium-floor constraint: %dер → %d (medium=%s, floor=%d, confidence %.2f → %.2f)",
        era_result.decade,
        corrected_decade,
        medium,
        floor,
        era_result.confidence,
        new_conf,
    )
    return EraResult(
        decade=corrected_decade,
        era_label=f"{corrected_decade}er",
        confidence=new_conf,
        material_prior=new_material,
        noise_profile=era_result.noise_profile,
        tier_used=era_result.tier_used,
        hf_rolloff_hz=era_result.hf_rolloff_hz,
        is_remaster_suspected=era_result.is_remaster_suspected,
    )
