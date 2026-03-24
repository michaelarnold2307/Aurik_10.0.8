"""
Defect Scanner for Aurik 9.0 - Defect-First Architecture
=========================================================

Erkennt und bewertet 15 Defekttypen in Audio-Materialien mit material-adaptiven
Thresholds. Ersetzt das Medium-First Detection System aus v8.0.

Performance-Budget: Max 5% der Gesamtzeit (< 30s für 3:45 Audio)

Author: Aurik 9.0 Development Team
Version: 9.0.0
Date: 2026-02-15
"""

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import threading
from typing import Optional

import numpy as np
import scipy.fft as fft
import scipy.signal as signal

# §6.3 CLIPPING vs SOFT_SATURATION discrimination via THD analysis (lazy import)
try:
    from backend.core.clipping_detection import ClippingType as _ClippingType, classify_clipping as _classify_clipping

    _CLIPPING_DETECTION_AVAILABLE = True
except ImportError:
    _CLIPPING_DETECTION_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §9.7.1 SHA256-Ergebnis-Cache — verhindert redundante Berechnungen bei Batch
# ---------------------------------------------------------------------------
_scan_cache: dict[str, object] = {}
_scan_cache_lock = threading.Lock()
_SCAN_CACHE_MAX = 128  # FIFO-Trim bei Überschreitung


def _audio_scan_cache_key(audio: np.ndarray, sr: int, material: object | None) -> str:
    """Deterministischer Cache-Key für DefectScanner.scan()."""
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    if material is not None:
        h.update(str(material).encode())
    return f"scan:{h.hexdigest()[:16]}"


class DefectType(Enum):
    """28 Defekttypen für weltklasse Audio-Restauration.

    Kern-Defekte (alle analogen/digitalen Quellen):
      CLIPPING        — Amplituden-Übersteuerung (Hard/Soft Clipping)
      DC_OFFSET       — Gleichspannungsversatz (Null-Linien-Verschiebung)
      BANDWIDTH_LOSS  — Hochfrequenz-Verlust (Shellac <7kHz, Kassette <14kHz, UKW/LP)
      PITCH_DRIFT     — Konstanter Geschwindigkeitsfehler (Tape-Stretch, Motorfehler)
      REVERB_EXCESS   — Unerwarteter/übermäßiger Raumhall (fehlerhafte Aufnahmeakustik)
      PRINT_THROUGH   — Magnetisches Übersprechen auf Tape (Pre-Echo 100–300 ms vor Einsatz)

    Analoge Tonträger:
      CLICKS, CRACKLE, HUM, WOW, FLUTTER, LOW_FREQ_RUMBLE, DROPOUTS

    Stereo / Kanal:
      STEREO_IMBALANCE, PHASE_ISSUES

    Digital / Codec:
      DIGITAL_ARTIFACTS, COMPRESSION_ARTIFACTS, HIGH_FREQ_NOISE
    """

    # --- Ursprüngliche 11 ---
    CLICKS = "clicks"
    CRACKLE = "crackle"
    HUM = "hum"
    WOW = "wow"  # Tonhöhenschwankung < 0.5 Hz (IEC 60386 — Motorexzentrizität, Plattenteller-Gleichlaufschwankung)
    FLUTTER = "flutter"  # Tonhöhenschwankung 0.5–200 Hz (IEC 60386 — mechanische Vibration, Führungsrolle, Bandantrieb)
    STEREO_IMBALANCE = "stereo_imbalance"
    DIGITAL_ARTIFACTS = "digital_artifacts"
    LOW_FREQ_RUMBLE = "low_freq_rumble"
    HIGH_FREQ_NOISE = "high_freq_noise"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    PHASE_ISSUES = "phase_issues"
    DROPOUTS = "dropouts"
    # --- Weltklasse-Erweiterung Runde 1 ---
    CLIPPING = "clipping"  # Amplituden-Übersteuerung (Hard/Soft Clip)
    DC_OFFSET = "dc_offset"  # Gleichspannungsversatz
    BANDWIDTH_LOSS = "bandwidth_loss"  # HF-Rolloff / Bandbreitenbegrenzung
    PITCH_DRIFT = "pitch_drift"  # Konstanter Tonhöhen-/Geschwindigkeitsfehler
    # --- Weltklasse-Erweiterung Runde 2 ---
    REVERB_EXCESS = "reverb_excess"  # Übermäßiger Raumhall (Akustik-Artefakt)
    PRINT_THROUGH = "print_through"  # Magnetisches Übersprechen bei Tape (Pre-Echo)
    # --- Weltklasse-Erweiterung Runde 3 ---
    QUANTIZATION_NOISE = "quantization_noise"  # Quantisierungsrauschen (niedrige Bit-Tiefe / Resampling)
    JITTER_ARTIFACTS = "jitter_artifacts"  # Zeitgitter-Fehler bei D/A-Wandlung (CD, DAT, Streaming)
    DYNAMIC_COMPRESSION_EXCESS = "dynamic_compression_excess"  # Übermäßige Dynamikkompression (Loudness War)
    # --- Spec §6.3: fehlende DefectTypes für 24-Wert-Katalog ---
    SOFT_SATURATION = "soft_saturation"  # Tube-/Tape-Sättigung (gerade Obertöne) — BEWAHREN! (§2.1, §6.3)
    HEAD_WEAR = "head_wear"  # Kopf-/Azimuth-Fehler, komplette Frequenzband-Auslöschung → phase_56 (§4.5, §7.2)
    AZIMUTH_ERROR = "azimuth_error"  # Kopf-Schrägstellung → HF-Phasen-Slope L/R > 20°/kHz (IEC 60386) → phase_56
    TRANSIENT_SMEARING = "transient_smearing"  # Ansatz-Verschmierung durch Kompression/Limiter — GrooveMetric-relevant
    PRE_ECHO = "pre_echo"  # MP3/AAC Temporal-Masking-Artefakt: Rauschen *vor* Transienten durch lange Codec-Analysefenster (§6.3)
    # --- Spec §6.3 v9.10.46c: 3 neue DefectTypes → 27 Gesamtanzahl ---
    RIAA_CURVE_ERROR = "riaa_curve_error"  # Falsche Disc-Entzerrungskurve (Shellac/früher Vinyl: AES/NAB/FFRR/Columbia) → phase_04 + phase_06
    ALIASING = (
        "aliasing"  # Spiegelfrequenzen durch unzureichenden AA-Filter bei ADC-Digitalisierung → phase_03 + phase_23
    )
    BIAS_ERROR = "bias_error"  # Falscher Vormagnetisierungsstrom bei Bandaufnahme → phase_04 + phase_03 + phase_29
    # --- Spec §6.3 v9.10.57: Sibilanten-Überbetonung (ergibt 28 DefectTypes) ---
    SIBILANCE = "sibilance"  # Zischlautüberbetonung (> 6 kHz) — De-Esser-Trigger (phase_19 + phase_43)
    # --- v9.10.57b: Transport-Bump (ergibt 29 DefectTypes) ---
    TRANSPORT_BUMP = (
        "transport_bump"  # Impulsartige Mikro-Geschwindigkeitssprünge 50–300 ms (Kassette/Tape-Holpern) → phase_12
    )


class MaterialType(Enum):
    """Material-Typen für adaptive Thresholds."""

    SHELLAC = "shellac"
    VINYL = "vinyl"
    TAPE = "tape"  # Cassette Tape
    REEL_TAPE = "reel_tape"  # Professional reel-to-reel (Studio)
    DAT = "dat"  # Digital Audio Tape (Professional)
    CD_DIGITAL = "cd_digital"
    MP3_LOW = "mp3_low"  # MP3 <128 kbps (heavy compression artifacts)
    MP3_HIGH = "mp3_high"  # MP3 ≥128 kbps (moderate artifacts)
    AAC = "aac"  # AAC/M4A (modern compressed)
    MINIDISC = "minidisc"  # ATRAC codec (90s/2000s)
    STREAMING = "streaming"
    WAX_CYLINDER = "wax_cylinder"  # Phonograph-Wachswalze (1890–1930), HF ≤ 5 kHz
    WIRE_RECORDING = "wire_recording"  # Drahtband (1940–1955), Jitter, Frequenzgang-Einbrüche
    LACQUER_DISC = "lacquer_disc"  # Acetat-Lackfolien (1930–1950), Risse, Substrat-Rauschen
    UNKNOWN = "unknown"


@dataclass
class DefectScore:
    """Score-Objekt für einen einzelnen Defekt."""

    defect_type: DefectType
    severity: float  # 0.0 - 1.0 (0 = keine Defekte, 1 = schwere Defekte)
    confidence: float  # 0.0 - 1.0 (Konfidenz der Detection)
    locations: list[tuple[float, float]] = field(default_factory=list)  # (start_time, end_time) in Sekunden
    metadata: dict = field(default_factory=dict)  # Zusätzliche Informationen

    def __repr__(self) -> str:
        return f"DefectScore({self.defect_type.value}: {self.severity:.3f}, conf={self.confidence:.2f}, {len(self.locations)} events)"


@dataclass
class DefectAnalysisResult:
    """Vollständiges Ergebnis der Defekt-Analyse."""

    material_type: MaterialType
    scores: dict[DefectType, DefectScore]
    analysis_time_seconds: float
    sample_rate: int
    duration_seconds: float
    # Forensische Tonträgerkettenerkennung (befüllt von DefectScanner.scan())
    transfer_chain_raw: dict = field(default_factory=dict)  # MediumDetector.detect()-Ausgabe
    is_multi_generation: bool = False  # Mehrstufige Überspielungskette
    transfer_chain_str: str = ""  # Lesbare Kette, z.B. "cassette → mp3"
    # §6.6.1 Pflicht-Spektralfingerabdruck — 5 normierte Messgrößen (immer befüllt)
    spectral_fingerprint: dict[str, float] = field(default_factory=dict)
    # spectral_fingerprint enthält:
    #   rolloff_95_hz       — Rolloff-Frequenz 95% der Spektralenergie [Hz]
    #   wow_flutter_index   — Pitch-Varianz-Index [0..∞, >1.5 = Kassette]
    #   hf_energy_above_16k — Anteil Energie > 16 kHz an Gesamtenergie [0..1]
    #   noise_floor_p5_db   — 5. Perzentil PSD als Rauschboden [dBFS]
    #   effective_bandwidth_hz — HF-Rolloff bei −60 dBFS [Hz]
    #   material_detected   — auto-erkanntes Material (auch wenn Hint übergeben)

    def get_top_defects(self, n: int = 5) -> list[DefectScore]:
        """Gibt die Top-N Defekte nach Severity zurück."""
        return sorted(self.scores.values(), key=lambda x: x.severity, reverse=True)[:n]

    def get_total_severity(self) -> float:
        """Gesamtschwere aller Defekte (gewichtet)."""
        return sum(score.severity * score.confidence for score in self.scores.values()) / len(self.scores)


class DefectScanner:
    """
    Hauptklasse für Defekt-Erkennung in Aurik 9.0.

    Ersetzt Medium-First Detection durch Defect-First Approach:
    - Scannt alle Audio-Daten einmal vollständig
    - Erkennt 28 Defekttypen sequenziell (11 Kern + 17 Weltklasse-Erweiterung)
    - Nutzt material-adaptive Thresholds für alle MaterialTypes
    - Performance-optimiert (max 5% overhead)
    """

    # Material-adaptive Sensitivity-Thresholds
    MATERIAL_SENSITIVITY = {
        MaterialType.SHELLAC: {
            DefectType.CLICKS: 0.3,  # Sehr empfindlich (Shellac hat viele Clicks)
            DefectType.CRACKLE: 0.3,
            DefectType.HUM: 0.6,
            DefectType.WOW: 0.40,  # Plattenteller-Gleichlaufschwankung (< 0.5 Hz)
            DefectType.FLUTTER: 0.50,  # Nadelresonanz, Abtastarm-Vibration (0.5–200 Hz)
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A für Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.5,
            DefectType.HIGH_FREQ_NOISE: 0.6,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.7,
            DefectType.CLIPPING: 0.5,  # Analoge Übersteuerung bei Schnittlack-Aufnahmen
            DefectType.DC_OFFSET: 0.4,  # Alte Vorverstärker mit DC-Drift
            DefectType.BANDWIDTH_LOSS: 0.2,  # Shellac: stark ausgeprägte HF-Rolloff < 7 kHz
            DefectType.PITCH_DRIFT: 0.3,  # 78 rpm Motorfehler sehr häufig
            DefectType.REVERB_EXCESS: 0.6,  # Shellac: meist trocken aufgenommen, aber möglich
            DefectType.PRINT_THROUGH: 1.0,  # N/A: Shellac ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Shellac ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Shellac ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.9,  # Shellac-Ära: keine Loudness-War-Kompression
            DefectType.SOFT_SATURATION: 0.3,  # Röhren-Mikrofon-Sättigung, Aufnahmetrichter — bewahren
            DefectType.HEAD_WEAR: 1.0,  # N/A: Shellac ist kein Magnetband (kein Magnetkopf-Verschleiß)
            DefectType.PRE_ECHO: 1.0,  # N/A: Shellac analog — kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.5,  # Mechanischer Trichter und schwere Nadel begrenzen Transientenbereich
            DefectType.RIAA_CURVE_ERROR: 0.2,  # AES/NAB/Columbia-Kurve vor RIAA-Standard — Entzerrungs-Fehler häufig
            DefectType.ALIASING: 0.5,  # Archiv-Digitalisierung oft mit unzureichendem AA-Filter
            DefectType.BIAS_ERROR: 1.0,  # N/A: Shellac ist kein Magnetband — kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Shellac ist Disc-Format — kein Magnetkopf-Azimuth
            DefectType.SIBILANCE: 0.6,  # Schwere Nadel + begrenzter HF → Zischlaut-Verzerrung bei Hochpegel-Passagen
            DefectType.TRANSPORT_BUMP: 0.5,  # Plattenteller-Transport: mechanisches Holpern bei 78 rpm
        },
        MaterialType.VINYL: {
            DefectType.CLICKS: 0.4,
            DefectType.CRACKLE: 0.5,
            DefectType.HUM: 0.5,  # 50Hz/60Hz hum häufig
            DefectType.WOW: 0.50,  # Plattenspieler-Motor-Gleichlauf (< 0.5 Hz)
            DefectType.FLUTTER: 0.55,  # Arm-/Stylusresonanz, Riemengetriebe-Vibration (0.5–200 Hz)
            DefectType.STEREO_IMBALANCE: 0.6,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.4,  # Turntable rumble
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.6,  # Schneidlackübersteuerung möglich
            DefectType.DC_OFFSET: 0.5,  # Phono-Vorverstärker DC-Bias
            DefectType.BANDWIDTH_LOSS: 0.4,  # RIAA-Entzerrung, HF generell begrenzt
            DefectType.PITCH_DRIFT: 0.4,  # Plattenspieler-Motorabweichungen
            DefectType.REVERB_EXCESS: 0.6,  # Vinyl: Aufnahme-Akustik selten Restaurationsproblem
            DefectType.PRINT_THROUGH: 1.0,  # N/A: Vinyl ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Vinyl ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Vinyl ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.6,  # Moderne Vinyl-Pressings manchmal überkomprimiert
            DefectType.SOFT_SATURATION: 0.4,  # Schneidlack-Sättigung möglich — bewahren wenn authentisch
            DefectType.HEAD_WEAR: 1.0,  # N/A: Vinyl ist kein Magnetband (kein Magnetkopf-Verschleiß)
            DefectType.PRE_ECHO: 1.0,  # N/A: Vinyl analog — kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.5,  # Schneidlack-Übertragungsfunktion: leichte Transientenverzerrung
            DefectType.RIAA_CURVE_ERROR: 0.3,  # Früh-Vinyl (vor 1954) nutzte verschiedene Kurven (AES, FFRR, Columbia)
            DefectType.ALIASING: 0.5,  # Digitalisierungsqualität variiert stark — AA-Filter oft suboptimal
            DefectType.BIAS_ERROR: 1.0,  # N/A: Schallplatte ist kein Magnetband — kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Vinyl ist Disc-Format — kein Magnetkopf-Azimuth
            DefectType.SIBILANCE: 0.7,  # Sehr häufig: Tonabnehmer-Sibilanz + Phono-Stufe; De-Esser-Pflicht
            DefectType.TRANSPORT_BUMP: 0.4,  # Plattenspieler-Transport: mechanisches Holpern möglich
        },
        MaterialType.TAPE: {
            DefectType.CLICKS: 0.7,
            DefectType.CRACKLE: 0.8,
            DefectType.HUM: 0.4,  # AC hum häufig bei Tape
            DefectType.WOW: 0.30,  # Capstan-Gleichlaufschwankung (< 0.5 Hz), sehr häufig!
            DefectType.FLUTTER: 0.25,  # Andruckrolle, Führungsrollen-Vibration (0.5–200 Hz)
            DefectType.STEREO_IMBALANCE: 0.5,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.6,
            DefectType.HIGH_FREQ_NOISE: 0.5,  # Tape hiss
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.4,  # Tape dropouts häufig
            DefectType.CLIPPING: 0.4,  # Tape-Sättigung durch Übersteuerung häufig
            DefectType.DC_OFFSET: 0.4,  # Kassettendecks mit DC-Bias im Signalweg
            DefectType.BANDWIDTH_LOSS: 0.3,  # Kassette: Rolloff bei 12–14 kHz typisch
            DefectType.PITCH_DRIFT: 0.2,  # Tape-Stretch & Motorfehler sehr häufig
            DefectType.REVERB_EXCESS: 0.4,  # Kassette: Aufnahmeraum oft hörbar
            DefectType.PRINT_THROUGH: 0.2,  # Kassette: magnetisches Übersprechen möglich
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Kassette ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Kassette ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.8,  # Kassette: Dolby-Kompander aber selten exzessiv
            DefectType.SOFT_SATURATION: 0.4,  # Bandsättigung (gerade Obertöne H2/H4) — BEWAHREN vs. Clipping
            DefectType.HEAD_WEAR: 0.6,  # Kassettenköpfe durch Abnutzung → Hochton-Auslöschung häufig
            DefectType.PRE_ECHO: 1.0,  # N/A: Kassette analog — kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.4,  # Dolby-Rauschreduktion und Bandsättigung können Transienten verschmieren
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Magnetband nutzt keine RIAA-Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Digitalisierungs-AA variiert; Resampling in der Verarbeitungskette
            DefectType.BIAS_ERROR: 0.3,  # SEHR HÄUFIG: falscher Bias für Bandsorte (Chromdioxid/Normallage)
            DefectType.AZIMUTH_ERROR: 0.30,  # Häufig! Kassettenköpfe neigen zu Azimuth-Drift zwischen verschiedenen Decks
            DefectType.SIBILANCE: 0.5,  # Kassettenkopf-HF-Sättigung → Zischlaut-Betonung bei Hochfrequenz-Peaking
            DefectType.TRANSPORT_BUMP: 0.3,  # Kassetten-Transport: Capstan/Andruckrolle-Holpern sehr häufig
        },
        MaterialType.CD_DIGITAL: {
            DefectType.CLICKS: 0.8,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.9,
            DefectType.WOW: 0.90,  # CD: Kristalloszillator — kein WOW (N/A)
            DefectType.FLUTTER: 0.90,  # CD: kristallstabil — kein FLUTTER (N/A)
            DefectType.STEREO_IMBALANCE: 0.7,
            DefectType.DIGITAL_ARTIFACTS: 0.3,  # Häufig bei CD!
            DefectType.LOW_FREQ_RUMBLE: 0.8,
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 0.4,  # MP3-artige Artifacts
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.5,  # Digital dropouts
            DefectType.CLIPPING: 0.4,  # Loudness-War CD-Mastering-Clipping häufig
            DefectType.DC_OFFSET: 0.7,  # Selten bei CD
            DefectType.BANDWIDTH_LOSS: 0.7,  # CD: 22 kHz Nyquist, volle Bandbreite
            DefectType.PITCH_DRIFT: 0.9,  # CD: Crystal-Takt, kein Pitch-Drift
            DefectType.REVERB_EXCESS: 0.7,  # CD: Reverb meist bewusste Produktionsentscheidung
            DefectType.PRINT_THROUGH: 1.0,  # N/A: CD ist kein Magnetband
            DefectType.QUANTIZATION_NOISE: 0.7,  # CD 16-bit: selten, aber bei schlecht gemasterter CD möglich
            DefectType.JITTER_ARTIFACTS: 0.3,  # CD-Player/Laufwerk-Jitter sehr häufig!
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.25,  # Loudness War bei CD sehr häufig
            DefectType.SOFT_SATURATION: 0.9,  # Digitale Quelldatei: Soft-Saturation selten, aber möglich
            DefectType.HEAD_WEAR: 1.0,  # N/A: CD digital — kein Magnetkopf
            DefectType.PRE_ECHO: 0.3,  # Loudness-War-Mastering: Pre-Echo als Transient-Vorlauf möglich
            DefectType.TRANSIENT_SMEARING: 0.2,  # Loudness-War-Kompression → Transient-Smearing sehr häufig
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: CD digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # CD digital-nativ; Resampling-Artefakte bei Formatketten möglich
            DefectType.BIAS_ERROR: 1.0,  # N/A: CD digital — kein Wechselstrom-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: CD digital — kein Magnetkopf
            DefectType.SIBILANCE: 0.3,  # CD: geringe Sibilanz-Gefahr; De-Emphasis-Fehler bei frühen CDs möglich
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: CD digital — kein mechanischer Transport
        },
        MaterialType.REEL_TAPE: {
            DefectType.CLICKS: 0.8,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.5,  # AC hum in professional gear
            DefectType.WOW: 0.40,  # Profi-Capstan-Gleichlauf (< 0.5 Hz)
            DefectType.FLUTTER: 0.35,  # Profi-Führungsrollen-Vibration (0.5–200 Hz)
            DefectType.STEREO_IMBALANCE: 0.6,
            DefectType.DIGITAL_ARTIFACTS: 1.0,
            DefectType.LOW_FREQ_RUMBLE: 0.7,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # Less hiss than cassette (better quality)
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.5,  # Professional tape more reliable
            DefectType.CLIPPING: 0.5,  # Studio Tape-Sättigung / Übersteuerung möglich
            DefectType.DC_OFFSET: 0.5,  # Studio-Vorverstärker können DC einbringen
            DefectType.BANDWIDTH_LOSS: 0.4,  # Reel-Tape: HF besser als Kassette, aber begrenzt
            DefectType.PITCH_DRIFT: 0.4,  # Professionelles Tape: besser aber nicht perfekt
            DefectType.REVERB_EXCESS: 0.3,  # Reel-Tape: Live-Raumakustik sehr häufig!
            DefectType.PRINT_THROUGH: 0.15,  # Reel-Tape: Print-Through klassisches Problem!
            DefectType.QUANTIZATION_NOISE: 0.9,  # N/A: Reel-Tape ist analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: Reel-Tape ist analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.8,  # Professionelles Reel-Tape: Kompression selten exzessiv
            DefectType.SOFT_SATURATION: 0.6,  # Profi-Bandsättigung (Röhren-Mischpult/Bandmaschine) — bewahren
            DefectType.HEAD_WEAR: 0.7,  # Profi-Banddeck Kopfverschleiß: breite Frequenzband-Auslöschung möglich
            DefectType.PRE_ECHO: 1.0,  # N/A: Spulenband analog — kein Codec-Pre-Echo
            DefectType.TRANSIENT_SMEARING: 0.4,  # Studio-Rauschreduktion (Dolby A/SR) → leichtes Transient-Smearing
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Spulenband nutzt keine RIAA-Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # Professionelle Digitalisierung meist gut — AA-Filter vorhanden
            DefectType.BIAS_ERROR: 0.3,  # Häufig: gemischte Bandsorten → falscher Bias-Strom beim Schnitt
            DefectType.AZIMUTH_ERROR: 0.25,  # Sehr häufig: Profi-Bandmaschinen mit verschiedenen Schnittköpfen
            DefectType.SIBILANCE: 0.4,  # Profi-Spulenband: HF-Sättigung bei hohem Bandfluss → Zischlaut-Überbetonung
            DefectType.TRANSPORT_BUMP: 0.2,  # Profi-Bandmaschine: Transport stabiler als Kassette
        },
        MaterialType.DAT: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,  # Digital, no crackle
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # DAT: digital, Crystal-Clock — kein WOW
            DefectType.FLUTTER: 1.0,  # DAT: digital, Crystal-Clock — kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.7,
            DefectType.DIGITAL_ARTIFACTS: 0.4,  # Some digital artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,
            DefectType.COMPRESSION_ARTIFACTS: 0.8,  # Lossless, minimal compression
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.4,  # Occasional digital dropouts
            DefectType.CLIPPING: 0.5,  # Digitales Clipping bei DAT exakt erkennbar
            DefectType.DC_OFFSET: 0.8,  # DAT digital: DC-Offset selten
            DefectType.BANDWIDTH_LOSS: 0.8,  # DAT 48 kHz: volle Bandbreite
            DefectType.PITCH_DRIFT: 0.9,  # DAT: Crystal-Takt, kein Drift
            DefectType.REVERB_EXCESS: 0.6,  # DAT: digitaler Feldrekorder, Aufnahmeakustik
            DefectType.PRINT_THROUGH: 1.0,  # N/A: DAT digital, kein magnetisches Übersprechen
            DefectType.QUANTIZATION_NOISE: 0.45,  # DAT 16-bit: Quantisierungsrauschen möglich
            DefectType.JITTER_ARTIFACTS: 0.35,  # DAT-Jitter bekanntes Problem!
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.6,  # DAT: oft schon komprimierter Quell-Content
            DefectType.SOFT_SATURATION: 0.9,  # DAT: Soft-Saturation aus analogem Eingang möglich
            DefectType.HEAD_WEAR: 1.0,  # DAT-Rotationskopf: Verschleiß → Hochton-Dropout möglich
            DefectType.PRE_ECHO: 0.3,  # DAT digital: Pre-Echo aus Quell-Codec-Material möglich
            DefectType.TRANSIENT_SMEARING: 0.2,  # DAT digital: seltenes Transient-Smearing aus Quell-Kompression
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: DAT digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # DAT digital-nativ; Resampling bei Weiterverarbeitung möglich
            DefectType.BIAS_ERROR: 1.0,  # N/A: DAT digital — kein analoger Bias erforderlich
            DefectType.AZIMUTH_ERROR: 0.50,  # DAT-Rotationskopf: Azimuth kann durch Kopfverschleiß driften
            DefectType.SIBILANCE: 0.2,  # DAT digital — geringe Sibilanz-Gefahr (selten bei Profi-DAT-Aufnahmen)
            DefectType.TRANSPORT_BUMP: 0.6,  # DAT-Laufwerk: Rotationskopf-Transport kann holpern
        },
        MaterialType.MP3_LOW: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MP3 digital — kein WOW
            DefectType.FLUTTER: 1.0,  # MP3 digital — kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.3,  # Heavy codec artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # HF loss typical for low bitrate
            DefectType.COMPRESSION_ARTIFACTS: 0.15,  # VERY common!
            DefectType.PHASE_ISSUES: 0.5,  # Stereo imaging issues
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.5,  # Clipping im Quellmaterial vor Kodierung
            DefectType.DC_OFFSET: 0.9,  # MP3: DC im digitalen Originalmaterial
            DefectType.BANDWIDTH_LOSS: 0.1,  # MP3 low: starker HF-Cutoff bei 10–14 kHz!
            DefectType.PITCH_DRIFT: 0.9,  # MP3: digital, kein Pitch-Drift
            DefectType.REVERB_EXCESS: 0.8,  # MP3: Reverb aus dem Quellmaterial, nicht Kodierung
            DefectType.PRINT_THROUGH: 1.0,  # N/A: MP3 digital
            DefectType.QUANTIZATION_NOISE: 0.3,  # MP3 low: aggressive Requantisierung sehr häufig!
            DefectType.JITTER_ARTIFACTS: 0.8,  # MP3: digital, Jitter selten
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.3,  # MP3 low: oft überkomprimierter Quell-Content
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MP3 kennt keine Soft-Saturation (digital)
            DefectType.HEAD_WEAR: 1.0,  # N/A: MP3 digital — kein Magnetkopf
            DefectType.PRE_ECHO: 0.2,  # MP3 Temporal-Masking → Pre-Echo vor Transienten sehr häufig!
            DefectType.TRANSIENT_SMEARING: 0.2,  # MP3 Temporal-Masking → starkes Attack-Smearing bei Perkussion
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MP3 digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Resampling-Kette im Codec erzeugt Aliasing-ähnliche Artefakte
            DefectType.BIAS_ERROR: 1.0,  # N/A: MP3 digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: MP3 digital — kein Magnetkopf
            DefectType.SIBILANCE: 0.6,  # MP3 128 kbps: psychoakustitsche Maskierung → starke Sibilanzverzerrung typisch
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: MP3 digital — kein mechanischer Transport
        },
        MaterialType.MP3_HIGH: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MP3 digital — kein WOW
            DefectType.FLUTTER: 1.0,  # MP3 digital — kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,  # Moderate artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.7,
            DefectType.COMPRESSION_ARTIFACTS: 0.3,  # Common but less severe
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.9,
            DefectType.CLIPPING: 0.5,  # Clipping im Quellmaterial
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.3,  # MP3 high: HF-Cutoff bei 16–18 kHz
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.5,  # MP3 high: weniger Requantisierung
            DefectType.JITTER_ARTIFACTS: 0.8,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.3,  # MP3 high: Quell-Content oft überkomprimiert
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MP3 kennt keine Soft-Saturation (digital)
            DefectType.HEAD_WEAR: 1.0,  # N/A: MP3 digital — kein Magnetkopf
            DefectType.PRE_ECHO: 0.3,  # MP3 (höheres Bitrate) → Pre-Echo weniger stark als Low
            DefectType.TRANSIENT_SMEARING: 0.3,  # MP3 high: Transient-Smearing weniger stark als Low
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MP3 digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # Höheres Bitrate — weniger Resampling-Artefakte als MP3-Low
            DefectType.BIAS_ERROR: 1.0,  # N/A: MP3 digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: MP3 digital — kein Magnetkopf
            DefectType.SIBILANCE: 0.3,  # MP3 ≥ 192 kbps: deutlich weniger Sibilanz-Artefakte als Low-Bitrate
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: MP3 digital — kein mechanischer Transport
        },
        MaterialType.AAC: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # AAC digital — kein WOW
            DefectType.FLUTTER: 1.0,  # AAC digital — kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,  # Better HF than MP3
            DefectType.COMPRESSION_ARTIFACTS: 0.25,  # More efficient than MP3
            DefectType.PHASE_ISSUES: 0.7,
            DefectType.DROPOUTS: 0.9,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.4,  # AAC: bessere HF als MP3 bei gleichem Bitrate
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.5,  # AAC: effizienter als MP3, weniger Quantisierungsfehler
            DefectType.JITTER_ARTIFACTS: 0.8,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.35,  # AAC: Source-Content oft überkomprimiert
            DefectType.SOFT_SATURATION: 1.0,  # N/A: AAC digital — Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: AAC digital — kein Magnetkopf
            DefectType.PRE_ECHO: 0.35,  # AAC: Temporal-Masking → Pre-Echo vor Transienten möglich
            DefectType.TRANSIENT_SMEARING: 0.3,  # AAC: Transient-Smearing bei hoher Kompression
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: AAC digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.3,  # AAC-Codec: Resampling-Artefakte in der Transkodierkette
            DefectType.BIAS_ERROR: 1.0,  # N/A: AAC digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: AAC digital — kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # AAC-Codec kann bei mittleren Bitraten Zischlaut-Artefakte einführen
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: AAC digital — kein mechanischer Transport
        },
        MaterialType.MINIDISC: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 1.0,
            DefectType.HUM: 0.9,
            DefectType.WOW: 1.0,  # MiniDisc digital ATRAC — kein WOW
            DefectType.FLUTTER: 1.0,  # MiniDisc digital ATRAC — kein FLUTTER
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.35,  # ATRAC specific artifacts
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.6,  # ATRAC artifacts at HF
            DefectType.COMPRESSION_ARTIFACTS: 0.2,  # ATRAC aggressive
            DefectType.PHASE_ISSUES: 0.6,  # Joint stereo issues
            DefectType.DROPOUTS: 0.8,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.2,  # ATRAC: starke HF-Artefakte über 14 kHz
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.7,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.3,  # ATRAC: aggressive Quantisierung sehr häufig!
            DefectType.JITTER_ARTIFACTS: 0.5,  # MiniDisc: ATRAC Buffer-Timing-Pröbleme
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.5,  # MiniDisc: Aufnahmen variieren stark
            DefectType.SOFT_SATURATION: 1.0,  # N/A: MiniDisc digital — Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: MiniDisc digital — kein Magnetkopf im klassischen Sinne
            DefectType.PRE_ECHO: 0.35,  # ATRAC: aggressive Temporal-Masking → Pre-Echo häufig
            DefectType.TRANSIENT_SMEARING: 0.2,  # ATRAC: starkes Transient-Smearing — GrooveMetric-relevant
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: MiniDisc digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # ATRAC-Codec: Resampling-Artefakte und Aliasing-Stufigkeit
            DefectType.BIAS_ERROR: 1.0,  # N/A: MiniDisc ATRAC digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 0.60,  # MiniDisc Rotationskopf: Azimuth-Drift bei Alterung möglich
            DefectType.SIBILANCE: 0.5,  # ATRAC-Codec (MiniDisc): Sibilanz-Artefakte charakteristisch bei 132 kbps
            DefectType.TRANSPORT_BUMP: 0.7,  # MiniDisc-Laufwerk: Rotationstransport kann holpern
        },
        MaterialType.STREAMING: {
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 0.9,
            DefectType.HUM: 0.9,
            DefectType.WOW: 0.90,  # Streaming digital — WOW N/A
            DefectType.FLUTTER: 0.90,  # Streaming digital — FLUTTER N/A
            DefectType.STEREO_IMBALANCE: 0.8,
            DefectType.DIGITAL_ARTIFACTS: 0.4,
            DefectType.LOW_FREQ_RUMBLE: 0.9,
            DefectType.HIGH_FREQ_NOISE: 0.8,
            DefectType.COMPRESSION_ARTIFACTS: 0.2,  # Sehr häufig!
            DefectType.PHASE_ISSUES: 0.6,
            DefectType.DROPOUTS: 0.3,  # Streaming dropouts
            DefectType.CLIPPING: 0.4,  # Loudness-War Streaming Masters
            DefectType.DC_OFFSET: 0.9,
            DefectType.BANDWIDTH_LOSS: 0.4,
            DefectType.PITCH_DRIFT: 0.9,
            DefectType.REVERB_EXCESS: 0.8,
            DefectType.PRINT_THROUGH: 1.0,
            DefectType.QUANTIZATION_NOISE: 0.4,  # Streaming: Transkodierungskette erzeugt Requantisierung
            DefectType.JITTER_ARTIFACTS: 0.4,  # Netzwerk-Jitter → Puffer-Underruns / Artefakte
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.2,  # Streaming-Normalisierung → Loudness-War besonders sichtbar!
            DefectType.SOFT_SATURATION: 1.0,  # N/A: Streaming digital — Soft-Saturation nur aus Quellmaterial
            DefectType.HEAD_WEAR: 1.0,  # N/A: Streaming digital — kein Magnetkopf
            DefectType.PRE_ECHO: 0.4,  # Streaming-Codec: Pre-Echo aus Transkodierkette möglich
            DefectType.TRANSIENT_SMEARING: 0.3,  # Streaming-Normalisierung + Codec → Transient-Smearing häufig
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Streaming digital — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.4,  # Mehrfache Transkodierkette erzeugt kumulative Aliasing-Artefakte
            DefectType.BIAS_ERROR: 1.0,  # N/A: Streaming digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Streaming digital — kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # Streaming-Codec (Opus/AAC): variable Bitraten → Sibilanz-Artefakte möglich
            DefectType.TRANSPORT_BUMP: 1.0,  # N/A: Streaming digital — kein mechanischer Transport
        },
        MaterialType.UNKNOWN: dict.fromkeys(DefectType, 0.6),
        MaterialType.WAX_CYLINDER: {
            # Phonograph-Wachswalze (1890–1930): extremer Rauschboden, HF ≤ 5 kHz
            DefectType.CLICKS: 0.2,  # Sehr häufig durch Zylinderoberfläche
            DefectType.CRACKLE: 0.2,  # Wachswalzen-Abrieb → starkes Crackle
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.30,  # Wachswalzen-Laufungenauigkeit < 0.5 Hz
            DefectType.FLUTTER: 0.40,  # Trichter-/Mechanik-Vibration 0.5–200 Hz
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: immer Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.4,  # Mechanisches Rumpeln der Walze
            DefectType.HIGH_FREQ_NOISE: 0.2,  # Extremes Oberflächenrauschen
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.4,  # Wachs-Fehler → Dropout
            DefectType.CLIPPING: 0.5,  # Analoge Groß-Signal-Verzerrung
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.1,  # HF extrem begrenzt (≤ 5 kHz)
            DefectType.PITCH_DRIFT: 0.2,  # Häufige Drehzahl-Ungleichmäßigkeit
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 1.0,  # N/A: kein Magnetband
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 1.0,
            DefectType.SOFT_SATURATION: 0.3,  # Akustische Sättigung durch Aufnahmetrichter
            DefectType.HEAD_WEAR: 1.0,  # N/A: kein Magnetkopf
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec auf Wachswalze
            DefectType.TRANSIENT_SMEARING: 0.5,  # Trägheit des Trichters begrenzt Transienten
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Wachswalze mechanisch — keine elektrische EQ-Kurve
            DefectType.ALIASING: 0.6,  # Sehr alte Digitalisierungen mit primitiven AA-Filtern
            DefectType.BIAS_ERROR: 1.0,  # N/A: Wachswalze mechanisch — kein Magnetband-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: mechanische Abtastung — kein Magnetkopf
            DefectType.SIBILANCE: 1.0,  # N/A: HF ≤ 5 kHz, Zischlautbereich physikalisch nicht erreichbar
            DefectType.TRANSPORT_BUMP: 0.5,  # Walzen-Transportholpern bei mechanischer Abtastung
        },
        MaterialType.WIRE_RECORDING: {
            # Drahtbandaufnahme (1940–1955): Jitter, Frequenzgang-Einbrüche, Magnetisierungs-Dropout
            DefectType.CLICKS: 0.5,
            DefectType.CRACKLE: 0.4,
            DefectType.HUM: 0.4,  # Magnetfeld-Überkopplung
            DefectType.WOW: 0.20,  # Drahtjitter-Gleichlauf (< 0.5 Hz), sehr charakteristisch
            DefectType.FLUTTER: 0.25,  # Drahtführungs-Vibration (0.5–200 Hz)
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: immer Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.5,
            DefectType.HIGH_FREQ_NOISE: 0.3,
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.3,  # Magnetisierungs-Dropout sehr häufig
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.2,  # HF begrenzt (≤ 8 kHz)
            DefectType.PITCH_DRIFT: 0.2,  # Drahtjitter → Pitch-Instabilität
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 0.5,  # Magnetdraht kann Print-Through entwickeln
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 0.2,  # Draht-Transportjitter sehr spezifisch
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 1.0,
            DefectType.SOFT_SATURATION: 0.4,  # Magnetische Sättigung des Drahts möglich
            DefectType.HEAD_WEAR: 0.3,  # Drahtkopfverschleiß typisch
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec
            DefectType.TRANSIENT_SMEARING: 0.4,  # Begrenzte Bandbreite → Transientenverzerrung
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Drahtbandaufnahme — kein Disc-Format
            DefectType.ALIASING: 0.5,  # Frühe Digitalisierungen mit suboptimalen AA-Filtern
            DefectType.BIAS_ERROR: 0.4,  # AC-Bias oft falsch justiert bei primitiven Drahtbandgeräten
            DefectType.AZIMUTH_ERROR: 0.40,  # Draht-Magnetkopf: Azimuth-Fehler durch primitive Justierung häufig
            DefectType.SIBILANCE: 0.5,  # Drahtband: begrenzter HF-Frequenzgang → De-Esser teilweise relevant
            DefectType.TRANSPORT_BUMP: 0.3,  # Drahtbandfuehrung: Transportrueckeln durch primitive Mechanik
        },
        MaterialType.LACQUER_DISC: {
            # Acetat-Lackfolien-Heimaufnahme (1930–1950): Risse, Substrat-Rauschen, Rille-Ermüdung
            DefectType.CLICKS: 0.2,  # Rissbildung → sehr häufige Klicks
            DefectType.CRACKLE: 0.3,  # Rille-Ermüdung → Crackle
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.30,  # Acetat-Verformung → Plattenteller-Gleichlauf < 0.5 Hz
            DefectType.FLUTTER: 0.40,  # Heim-Abtastarm-Vibration 0.5–200 Hz
            DefectType.STEREO_IMBALANCE: 1.0,  # N/A: meist Mono
            DefectType.DIGITAL_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.LOW_FREQ_RUMBLE: 0.4,
            DefectType.HIGH_FREQ_NOISE: 0.3,  # Substrat-Rauschen
            DefectType.COMPRESSION_ARTIFACTS: 1.0,
            DefectType.PHASE_ISSUES: 1.0,
            DefectType.DROPOUTS: 0.4,  # Risse erzeugen schmale Dropout-Ereignisse
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.4,
            DefectType.BANDWIDTH_LOSS: 0.2,  # Schmalbandige Hausaufnahme-Technik
            DefectType.PITCH_DRIFT: 0.3,  # Heimplattenspieler mit Motorungenauigkeiten
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 1.0,  # N/A: kein Magnetband
            DefectType.QUANTIZATION_NOISE: 1.0,  # N/A: analog
            DefectType.JITTER_ARTIFACTS: 1.0,  # N/A: analog
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.9,
            DefectType.SOFT_SATURATION: 0.3,  # Rillenverzerrung bei Heimaufnahmen möglich
            DefectType.HEAD_WEAR: 1.0,  # N/A: kein Magnetkopf
            DefectType.PRE_ECHO: 1.0,  # N/A: kein digitaler Codec
            DefectType.TRANSIENT_SMEARING: 0.4,  # Limitierte Heimaufnahme-Technik
            DefectType.RIAA_CURVE_ERROR: 0.2,  # Heimaufnahmen: verschiedene EQ-Kurven (AES/NAB/FFRR) häufig falsch
            DefectType.ALIASING: 0.5,  # Heim-Digitalisierung variiert — AA-Filter fehlt oft
            DefectType.BIAS_ERROR: 1.0,  # N/A: Lackfolie ist kein Magnetband — kein Aufnahme-Bias
            DefectType.AZIMUTH_ERROR: 1.0,  # N/A: Lackfolie ist mechanisches Disc-Format — kein Magnetkopf
            DefectType.SIBILANCE: 0.4,  # Heimaufnahme-Nadel: Zischlaute bei HF-Überbetonung möglich
            DefectType.TRANSPORT_BUMP: 0.5,  # Plattenteller-Holpern bei Heimaufnahme-Technik
        },
    }

    def __init__(self, sample_rate: int = 44100, material_type: MaterialType | None = None):
        """
        Initialisiert DefectScanner.

        Args:
            sample_rate: Audio sample rate (Standard: 44100 Hz)
            material_type: Material-Typ (wird auto-detected wenn None)
        """
        self.sample_rate = sample_rate
        self.material_type = material_type
        # Always set thresholds (use UNKNOWN if material_type is None for auto-detection)
        self.thresholds = self.MATERIAL_SENSITIVITY.get(
            material_type if material_type else MaterialType.UNKNOWN, self.MATERIAL_SENSITIVITY[MaterialType.UNKNOWN]
        )

        logger.info(f"DefectScanner initialisiert: SR={sample_rate}, Material={material_type}")

    def scan(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        material_type: MaterialType | None = None,
        progress_callback: Optional["Callable[[int, str], None]"] = None,
    ) -> DefectAnalysisResult:
        """
        Hauptmethode: Scannt Audio-Daten und erkennt alle 20 Defekttypen.

        Args:
            audio: Audio-Daten (mono: shape=(n_samples,), stereo: shape=(n_samples, 2))
            sample_rate: Sample rate (falls nicht im Constructor gesetzt)
            material_type: Override für Material-Typ (falls nicht im Constructor gesetzt)

        Returns:
            DefectAnalysisResult mit allen Scores
        """
        import time

        # §9.7.1 SHA256-Cache — bei identischem Eingangssignal sofort zurückgeben
        _sr_for_key = sample_rate if sample_rate is not None else self.sample_rate
        _cache_key = _audio_scan_cache_key(audio, _sr_for_key, material_type)
        with _scan_cache_lock:
            if _cache_key in _scan_cache:
                logger.debug("DefectScanner Cache-Hit: %s", _cache_key)
                return _scan_cache[_cache_key]  # type: ignore[return-value]

        start_time = time.time()

        # Sample rate verwenden (Parameter > Instance > Default)
        sr = sample_rate if sample_rate is not None else self.sample_rate

        # §6.6.1 Pflicht-Spektralfingerabdruck: IMMER berechnen, unabhängig vom material_type-Hint
        _sf: dict[str, float] = {}
        try:
            _fp_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
            _fp_freqs, _fp_psd = signal.welch(_fp_mono, sr, nperseg=min(4096, len(_fp_mono)))
            _fp_total = max(float(np.sum(_fp_psd)), 1e-12)
            # Rolloff 95 %
            _fp_cumsum = np.cumsum(_fp_psd)
            _fp_rolloff_mask = _fp_cumsum >= 0.95 * _fp_total
            _sf["rolloff_95_hz"] = (
                float(_fp_freqs[_fp_rolloff_mask][0]) if _fp_rolloff_mask.any() else float(_fp_freqs[-1])
            )
            # Wow/Flutter-Index (Pitch-Varianz über 100ms-Fenster)
            _wf_hop = max(1, int(sr * 0.1))
            _wf_frames = [_fp_mono[i : i + _wf_hop] for i in range(0, len(_fp_mono) - _wf_hop, _wf_hop)]
            if len(_wf_frames) > 2:
                _wf_rms = np.array([float(np.sqrt(np.mean(f**2) + 1e-12)) for f in _wf_frames])
                _sf["wow_flutter_index"] = float(np.std(_wf_rms) / (np.mean(_wf_rms) + 1e-12) * sr / 100.0)
            else:
                _sf["wow_flutter_index"] = 0.0
            # HF-Energie > 16 kHz
            _sf["hf_energy_above_16k"] = float(np.sum(_fp_psd[_fp_freqs > 16000]) / _fp_total)
            # Rauschboden: 5. Perzentil PSD in dBFS
            _fp_p5 = float(np.percentile(_fp_psd, 5))
            _sf["noise_floor_p5_db"] = float(10.0 * np.log10(max(_fp_p5, 1e-20)))
            # Effektive Bandbreite: Rolloff bei −60 dBFS
            _fp_db = 10.0 * np.log10(np.maximum(_fp_psd, 1e-20))
            _fp_max_db = float(np.max(_fp_db))
            _fp_bw_mask = _fp_db >= (_fp_max_db - 60.0)
            _sf["effective_bandwidth_hz"] = float(_fp_freqs[_fp_bw_mask][-1]) if _fp_bw_mask.any() else 0.0
            logger.debug(
                "§6.6.1 SpektralFingerabdruck: rolloff95=%.0f Hz, wow=%.2f, hf16k=%.3f, bw=%.0f Hz, rausch=%.1f dBFS",
                _sf["rolloff_95_hz"],
                _sf["wow_flutter_index"],
                _sf["hf_energy_above_16k"],
                _sf["effective_bandwidth_hz"],
                _sf["noise_floor_p5_db"],
            )
        except Exception as _fp_err:
            logger.debug("SpektralFingerabdruck-Berechnung Fehler: %s", _fp_err)

        # Material-Typ: Hint als Prior, aber auto-detect für spectral_fingerprint immer ausführen
        _material_hint = material_type  # Hint des Aufrufers (kann None sein)
        _auto_material = self._auto_detect_material(audio)  # §6.6.1: immer berechnen
        _sf["material_detected"] = float(list(MaterialType).index(_auto_material))
        if material_type is None:
            material_type = self.material_type or _auto_material

        # §9.7.5a — Update instance attribute so _detect_dropouts() can access
        # the resolved material type for adaptive thresholds.
        self.material_type = material_type

        self.thresholds = self.MATERIAL_SENSITIVITY[material_type]

        # Audio normalisieren für konsistente Detection
        if audio.ndim == 2:
            is_stereo = True
            audio_mono = np.mean(audio, axis=1)
        else:
            is_stereo = False
            audio_mono = audio

        # §9.7.5 Audio-Cap for 28 detectors.
        # Defects are stationary over a 60 s sample (spec: ≤ 2 s/min audio).
        # Reduces scan time from ~22 s (225 s track) to ~6 s.
        # The spectral fingerprint and _auto_detect_material() above intentionally
        # run on the full audio for a representative signal characterisation.
        _DETECTOR_CAP_S = 60
        _detector_cap_n = _DETECTOR_CAP_S * sr
        _location_offset_s = 0.0  # seconds to add to all detector locations
        # §9.7.5a — Full-audio reference for non-stationary defects (dropouts).
        # Dropouts can occur anywhere (intro, outro, tape leader) — the 60 s
        # center-crop would miss them.  Other detectors (noise, hum, flutter)
        # are stationary and still use the cropped audio for performance.
        _audio_mono_full = audio_mono  # kept for dropout detection
        if len(audio_mono) > _detector_cap_n:
            _dc_mid = len(audio_mono) // 2
            _dc_half = _detector_cap_n // 2
            _location_offset_s = (_dc_mid - _dc_half) / sr
            audio_mono = audio_mono[_dc_mid - _dc_half : _dc_mid + _dc_half]
            logger.debug(
                "DefectScanner: audio_mono capped to %d s (%d samples) for detection pass, offset=%.1f s",
                _DETECTOR_CAP_S,
                _detector_cap_n,
                _location_offset_s,
            )

        def _prog(pct: int, name: str = "") -> None:
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    try:
                        progress_callback(pct, name)
                    except TypeError:
                        progress_callback(pct)  # type: ignore[call-arg]

        # Alle 28 Defekttypen sequentiell — nach jedem Schritt Fortschritt melden
        scores = {}

        _prog(5, "Klicks")
        scores[DefectType.CLICKS] = self._detect_clicks(audio_mono if not is_stereo else audio)
        _prog(10, "Knistern")
        scores[DefectType.CRACKLE] = self._detect_crackle(audio_mono)
        _prog(16, "Brummen")
        scores[DefectType.HUM] = self._detect_hum(audio_mono)
        _prog(22, "Tonhöhenschwankung")
        scores[DefectType.WOW] = self._detect_wow(audio_mono)  # IEC 60386 < 0.5 Hz
        scores[DefectType.FLUTTER] = self._detect_flutter(audio_mono)  # IEC 60386 0.5–200 Hz
        scores[DefectType.AZIMUTH_ERROR] = self._detect_azimuth_error(audio)  # PHD-Slope L/R
        _prog(28, "Stereo-Ungleichgewicht")
        scores[DefectType.STEREO_IMBALANCE] = (
            self._detect_stereo_imbalance(audio) if is_stereo else DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)
        )
        _prog(34, "Digitalartefakte")
        scores[DefectType.DIGITAL_ARTIFACTS] = self._detect_digital_artifacts(audio_mono)
        _prog(40, "Tieffrequenz-Rumpeln")
        scores[DefectType.LOW_FREQ_RUMBLE] = self._detect_low_freq_rumble(audio_mono)
        _prog(46, "Hochfrequenzrauschen")
        scores[DefectType.HIGH_FREQ_NOISE] = self._detect_high_freq_noise(audio_mono)
        _prog(52, "Kompressions-Artefakte")
        scores[DefectType.COMPRESSION_ARTIFACTS] = self._detect_compression_artifacts(audio_mono)
        _prog(58, "Phasenfehler")
        scores[DefectType.PHASE_ISSUES] = (
            self._detect_phase_issues(audio) if is_stereo else DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)
        )
        _prog(64, "Aussetzer")
        # §9.7.5a — Dropout detection runs on FULL audio (not center-cropped).
        # Tape dropouts occur anywhere (intro, leader, splice points).
        scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)
        _prog(70, "Übersteuerung")
        # --- Weltklasse-Erweiterung: 4 neue Defekttypen ---
        scores[DefectType.CLIPPING] = self._detect_clipping(audio_mono)
        _prog(75, "DC-Versatz")
        scores[DefectType.DC_OFFSET] = self._detect_dc_offset(audio_mono)
        _prog(80, "Bandbreitenverlust")
        scores[DefectType.BANDWIDTH_LOSS] = self._detect_bandwidth_loss(audio_mono)
        _prog(84, "Tonhöhendrift")
        scores[DefectType.PITCH_DRIFT] = self._detect_pitch_drift(audio_mono)
        _prog(88, "Übermäßiger Hall")
        scores[DefectType.REVERB_EXCESS] = self._detect_reverb_excess(audio_mono)
        _prog(91, "Durchkopieren")
        scores[DefectType.PRINT_THROUGH] = self._detect_print_through(audio_mono)
        _prog(94, "Quantisierungsrauschen")
        # --- Weltklasse-Erweiterung Runde 3 ---
        scores[DefectType.QUANTIZATION_NOISE] = self._detect_quantization_noise(audio_mono)
        _prog(96, "Jitter-Artefakte")
        scores[DefectType.JITTER_ARTIFACTS] = self._detect_jitter_artifacts(audio_mono)
        _prog(98, "Überkompression")
        scores[DefectType.DYNAMIC_COMPRESSION_EXCESS] = self._detect_dynamic_compression_excess(audio_mono)
        # --- Vollständige 28-Typen-Erweiterung (F-7) ---
        scores[DefectType.SOFT_SATURATION] = self._detect_soft_saturation(audio_mono)
        scores[DefectType.SIBILANCE] = self._detect_sibilance(audio_mono)
        scores[DefectType.BIAS_ERROR] = self._detect_bias_error(audio_mono)
        scores[DefectType.RIAA_CURVE_ERROR] = self._detect_riaa_curve_error(audio_mono)
        scores[DefectType.HEAD_WEAR] = self._detect_head_wear(audio_mono)
        scores[DefectType.TRANSIENT_SMEARING] = self._detect_transient_smearing(audio_mono)
        scores[DefectType.PRE_ECHO] = self._detect_pre_echo(audio_mono)
        scores[DefectType.ALIASING] = self._detect_aliasing(audio_mono)
        scores[DefectType.TRANSPORT_BUMP] = self._detect_transport_bump(audio_mono)

        # ── §9.1b Intro-Salienz-Gewichtung ──────────────────────────────────────
        # Psychoacoustic research (Zacharov & Koivuniemi 2001, Bech & Zacharov 2006):
        # the first 3-5 seconds define the listener's overall quality judgment.
        # Defects in the intro region receive a severity boost so that the pipeline
        # prioritizes their repair.  This is especially critical for tape media
        # where leader artifacts and run-in fluctuations cluster at the beginning.
        _INTRO_SECONDS = 5.0
        _INTRO_SEVERITY_BOOST = 1.5  # 50% boost for intro defects
        _total_duration_s = len(_audio_mono_full) / sr
        if _total_duration_s > _INTRO_SECONDS * 2:  # only for non-trivial audio
            for _dt, _ds in scores.items():
                if not _ds.locations:
                    continue
                _intro_events = [(t0, t1) for t0, t1 in _ds.locations if t0 < _INTRO_SECONDS]
                if _intro_events and _ds.severity > 0.0:
                    _intro_fraction = len(_intro_events) / max(len(_ds.locations), 1)
                    # Boost severity proportional to intro defect concentration
                    _boost = 1.0 + (_INTRO_SEVERITY_BOOST - 1.0) * _intro_fraction
                    _old_sev = _ds.severity
                    _ds.severity = float(
                        np.nan_to_num(
                            min(1.0, _ds.severity * _boost),
                            nan=0.0,
                        )
                    )
                    if _ds.severity > _old_sev:
                        _ds.metadata["intro_boost_applied"] = True
                        _ds.metadata["intro_boost_factor"] = round(_boost, 3)
                        _ds.metadata["intro_events"] = len(_intro_events)
                        logger.debug(
                            "Intro-saliency boost: %s severity %.3f → %.3f (%d/%d events in first %.0fs)",
                            _dt.value,
                            _old_sev,
                            _ds.severity,
                            len(_intro_events),
                            len(_ds.locations),
                            _INTRO_SECONDS,
                        )

        # ── Location-Offset korrigieren ─────────────────────────────────────────
        # Detektoren, die auf dem 60 s-Mitte-Clip (audio_mono) laufen, erzeugen
        # Locations relativ zum Clip-Start.  Der Offset muss addiert werden, damit
        # die Marker an der korrekten Stelle im Gesamt-Audio erscheinen.
        # _detect_clicks() bei Stereo nutzt das volle Audio → kein Offset nötig.
        if _location_offset_s > 0.0:
            _FULL_AUDIO_DETECTORS = {DefectType.DROPOUTS}  # runs on full audio, no offset needed
            if is_stereo:
                _FULL_AUDIO_DETECTORS.add(DefectType.CLICKS)
            for _dt, _ds in scores.items():
                if _dt in _FULL_AUDIO_DETECTORS:
                    continue
                if _ds.locations:
                    _ds.locations = [(t0 + _location_offset_s, t1 + _location_offset_s) for t0, t1 in _ds.locations]

        # ── Per-Channel Location-Tags (L/R) für Stereo ─────────────────────────
        # Alle lokalisierbaren Impuls-/Event-Defekte werden pro Kanal detektiert,
        # damit die Wellenform-Visualisierung kanalgenau darstellen kann.
        # Premium-Anforderung: kein "über einen Kamm scheren" beider Kanäle.
        if is_stereo and audio.ndim == 2 and audio.shape[1] >= 2:
            _per_channel_locs: dict = {}
            _DT_TO_KEY = {
                DefectType.CLICKS: "clicks",
                DefectType.DROPOUTS: "dropout",
                DefectType.CRACKLE: "crackle",
                DefectType.CLIPPING: "clipping",
                DefectType.SIBILANCE: "sibilance",
                DefectType.TRANSPORT_BUMP: "transport_bump",
            }
            for _ch_idx, _ch_label in ((0, "L"), (1, "R")):
                _ch_audio = audio[:, _ch_idx]
                # Clicks per channel
                _ch_clicks = self._detect_clicks(_ch_audio)
                if _ch_clicks.locations:
                    _per_channel_locs.setdefault("clicks", {})[_ch_label] = list(_ch_clicks.locations)
                # Dropout per channel
                _ch_drops = self._detect_dropouts(_ch_audio)
                if _ch_drops.locations:
                    _per_channel_locs.setdefault("dropout", {})[_ch_label] = list(_ch_drops.locations)
                # Crackle per channel
                try:
                    _ch_crackle = self._detect_crackle(_ch_audio)
                    if _ch_crackle.locations:
                        _per_channel_locs.setdefault("crackle", {})[_ch_label] = list(_ch_crackle.locations)
                except Exception:
                    pass
                # Clipping per channel
                try:
                    _ch_clip = self._detect_clipping(_ch_audio)
                    if _ch_clip.locations:
                        _per_channel_locs.setdefault("clipping", {})[_ch_label] = list(_ch_clip.locations)
                except Exception:
                    pass
                # Sibilance per channel
                try:
                    _ch_sib = self._detect_sibilance(_ch_audio)
                    if _ch_sib.locations:
                        _per_channel_locs.setdefault("sibilance", {})[_ch_label] = list(_ch_sib.locations)
                except Exception:
                    pass
                # Transport bumps per channel
                try:
                    _ch_bump = self._detect_transport_bump(_ch_audio)
                    if _ch_bump.locations:
                        _per_channel_locs.setdefault("transport_bump", {})[_ch_label] = list(_ch_bump.locations)
                except Exception:
                    pass
            # Store per-channel info as metadata on the combined scores
            for _dk, _ch_dict in _per_channel_locs.items():
                _dt_key = {v: k for k, v in _DT_TO_KEY.items()}.get(_dk)
                if _dt_key and _dt_key in scores:
                    scores[_dt_key].metadata["channel_locations"] = _ch_dict

        # ── §6.3 Medium-Gate: physikalisch unmögliche Defekte unterdrücken ──────
        # RIAA-Entzerrungsfehler sind NUR auf Disc-Medien möglich (Vinyl, Shellac,
        # Lacquer-Disc, Wax-Cylinder).  Auf Magnetband / Digital / Codec-Quellen
        # ist ein Bass/Mid-Ungleichgewicht KEIN RIAA-Fehler, sondern ggf. ein
        # EQ-Problem oder Aufnahme-Charakteristik — severity wird auf 0 gesetzt.
        _DISC_MEDIA = {
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.LACQUER_DISC,
            MaterialType.WAX_CYLINDER,
            MaterialType.UNKNOWN,  # unbekannt: nicht unterdrücken
        }
        if material_type not in _DISC_MEDIA:
            _riaa_orig = scores[DefectType.RIAA_CURVE_ERROR].severity
            if _riaa_orig > 0.0:
                logger.info(
                    "Medium-Gate: RIAA_CURVE_ERROR suppressed (material=%s, was=%.3f) "
                    "— RIAA is disc-only, not applicable to %s",
                    material_type.value,
                    _riaa_orig,
                    material_type.value,
                )
                scores[DefectType.RIAA_CURVE_ERROR] = DefectScore(
                    defect_type=DefectType.RIAA_CURVE_ERROR,
                    severity=0.0,
                    confidence=scores[DefectType.RIAA_CURVE_ERROR].confidence,
                    locations=[],
                    metadata={
                        **scores[DefectType.RIAA_CURVE_ERROR].metadata,
                        "medium_gated": True,
                        "original_severity": _riaa_orig,
                    },
                )

        # ── §9.1c Perceptual Salience — psychoacoustic masking annotation ───────
        # Annotates each defect with a 'perceptual_salience' score (0.0–1.0).
        # Masked defects (in loud passages) get reduced severity; exposed defects
        # (in quiet passages) keep full severity.  This prevents unnecessary repairs
        # on inaudible defects and focuses the pipeline on what the ear can detect.
        try:
            from backend.core.perceptual_salience import get_perceptual_salience_estimator

            _pse = get_perceptual_salience_estimator()
            _pse_audio = _audio_mono_full if _audio_mono_full is not None else audio_mono
            _pse_result = _pse.annotate_defect_scores(
                _pse_audio,
                sr,
                DefectAnalysisResult(
                    material_type=material_type,
                    scores=scores,
                    analysis_time_seconds=0.0,
                    sample_rate=sr,
                    duration_seconds=len(_pse_audio) / sr,
                ),
            )
            scores = _pse_result.scores
        except Exception as _pse_err:
            logger.warning("PerceptualSalienceEstimator failed (non-critical): %s", _pse_err)

        _prog(99)

        analysis_time = time.time() - start_time
        duration = len(audio_mono) / sr

        logger.info(
            f"DefectScan completed: {analysis_time:.2f}s für {duration:.1f}s Audio ({analysis_time / duration * 100:.1f}% overhead)"
        )

        # Forensische Tonträgerkettenerkennung (MediumDetector, DSP-basiert)
        _fmd_result: object = {}
        try:
            from backend.core.forensics.medium_detector import MediumDetector as _ForensicMD

            # Für lange Dateien: nur erste 30 s analysieren (Geschwindigkeit)
            _max_forensic = sr * 30
            _audio_forensic = audio_mono[:_max_forensic] if len(audio_mono) > _max_forensic else audio_mono
            logger.debug(f"[SCAN] MediumDetector.detect() → {len(_audio_forensic) / sr:.1f}s Audio …")
            _fmd_result = _ForensicMD().detect(_audio_forensic, sr)
            # MediumDetectionResult ist ein Dataclass — getattr statt .get()
            _multi = getattr(_fmd_result, "is_multi_generation", None)
            _chain_val = getattr(_fmd_result, "transfer_chain", None) or getattr(_fmd_result, "chain", "?")
            logger.debug(
                f"[SCAN] MediumDetector OK: multi={_multi}, chain={_chain_val}",
            )
        except Exception as _fmd_err:
            logger.debug(f"[SCAN] MediumDetector FEHLER: {_fmd_err}")
            logger.debug("ForensicMediumDetector nicht verfügbar: %s", _fmd_err)

        _prog(100)
        _scan_result = DefectAnalysisResult(
            material_type=material_type,
            scores=scores,
            analysis_time_seconds=analysis_time,
            sample_rate=sr,
            duration_seconds=duration,
            transfer_chain_raw=_fmd_result,
            is_multi_generation=bool(getattr(_fmd_result, "is_multi_generation", False)),
            transfer_chain_str=str(getattr(_fmd_result, "transfer_chain", "") or getattr(_fmd_result, "chain", "")),
            spectral_fingerprint=_sf,  # §6.6.1 Pflicht-Fingerabdruck — 5 Messgrößen immer befüllt
        )

        # §9.7.1 Cache-Write — Ergebnis für künftige identische Aufrufe sichern
        with _scan_cache_lock:
            if len(_scan_cache) >= _SCAN_CACHE_MAX:
                _scan_cache.pop(next(iter(_scan_cache)))  # FIFO-Trim
            _scan_cache[_cache_key] = _scan_result
        return _scan_result

    # ========== MATERIAL AUTO-DETECTION ==========

    def _auto_detect_material(self, audio: np.ndarray) -> MaterialType:
        """
        Auto-Detect Material-Typ basierend auf Audio-Charakteristiken.

        Heuristiken:
        - Shellac: Hohe Click-Rate, Mono, niedrige Frequenz-Range
        - Vinyl: Moderate Click-Rate, Stereo oder Mono, 50/60Hz hum, rumble
        - Tape: High-freq noise (hiss), wow/flutter, stereo oder mono
        - CD/Digital: Sehr sauber, evtl. digital artifacts, stereo
        - Streaming: Compression artifacts, stereo
        """
        if audio.ndim == 1:
            # Mono → Shellac, Vinyl (Mono) oder Tape (Mono)
            return self._detect_mono_material(audio)
        else:
            # Stereo → Vinyl, Tape, CD oder Streaming
            return self._detect_stereo_material(audio)

    def _detect_mono_material(self, audio: np.ndarray) -> MaterialType:
        """
        Unterscheidet Shellac vs Vinyl (Mono) vs Tape (Mono).

        Verwendet Scoring-System für robustere Detektion.
        """
        # === Feature-Extraction ===

        # Quick click detection
        diff = np.abs(np.diff(audio))
        click_rate = np.sum(diff > np.percentile(diff, 99.9)) / len(audio) * self.sample_rate

        # Spectral analysis
        freqs, psd = signal.welch(audio, self.sample_rate, nperseg=2048)
        _psd_total = np.maximum(np.sum(psd), 1e-12)  # §3.1: Zero-Division-Guard bei Stille

        # High-freq energy (8kHz+)
        high_freq_energy = np.sum(psd[freqs > 8000]) / _psd_total

        # Rumble/Low-freq content
        rumble_energy = np.sum(psd[freqs < 60]) / _psd_total

        # Mid-low freq (50-60Hz hum typical for vinyl)
        _mid_low_energy = np.sum(psd[(freqs >= 50) & (freqs <= 70)]) / _psd_total  # noqa: F841

        # Crackle detection
        crackle_score = self._detect_crackle(audio).severity

        # Wow/Flutter (beide analog — IEC 60386)
        wow_flutter_score = max(
            self._detect_wow(audio if audio.ndim == 1 else audio.mean(axis=1)).severity,
            self._detect_flutter(audio if audio.ndim == 1 else audio.mean(axis=1)).severity,
        )

        # === Material Scoring ===

        # SHELLAC Score (Mono) - sehr alt, viele defekte, wenig HF
        # Penalized heavily since modern materials are more common
        shellac_score = 0.0
        shellac_score += click_rate * 0.3  # Clicks weniger gewichtet
        shellac_score += crackle_score * 2.0  # Crackle reduziert
        shellac_score += (1.0 - high_freq_energy) * 1.0  # HF-Mangel weniger stark
        shellac_score += wow_flutter_score * 0.3  # Wow/Flutter reduziert
        shellac_score -= rumble_energy * 20.0  # Shellac sollte fast kein Rumble haben
        shellac_score -= 10.0  # Starke Baseline-Penalty (Shellac ist selten)

        # VINYL Score (Mono) - Basiert auf empirischen Daten
        # Vinyl (1950s scratched): HF=0.035, rumble=0.0002, crackle=1.0, clicks=48
        vinyl_score = 0.0
        vinyl_score += high_freq_energy * 150.0  # Höhere HF als Tape (Hauptmerkmal!)
        vinyl_score += crackle_score * 3.0  # Starkes Crackle typisch
        vinyl_score += click_rate * 0.3  # Moderate Clicks
        vinyl_score += wow_flutter_score * 2.0  # Speed-Variation
        vinyl_score -= rumble_energy * 1500.0  # SEHR wenig Rumble - stark gewichtet!

        # TAPE Score (Mono) - Basiert auf empirischen Daten
        # Tape (1980s cassette): HF=0.024, rumble=0.0010, crackle=1.0, clicks=48
        tape_score = 0.0
        tape_score += wow_flutter_score * 6.0  # Sehr typisch (Tape-Stretch)
        tape_score += rumble_energy * 2500.0  # Viel mehr Rumble - stark erhöht!
        tape_score -= high_freq_energy * 80.0  # Niedrigere HF als Vinyl (Hauptmerkmal!)
        tape_score += crackle_score * 1.0  # Crackle schwach positiv (beide haben es)
        tape_score += click_rate * 0.1  # Clicks schwach positiv
        tape_score += 10.0  # Baseline-Bonus erhöht

        logger.debug(
            f"Mono material scores: shellac={shellac_score:.2f}, vinyl={vinyl_score:.2f}, tape={tape_score:.2f}"
        )

        # Select best match (minimum threshold 0.5)
        scores = {MaterialType.SHELLAC: shellac_score, MaterialType.VINYL: vinyl_score, MaterialType.TAPE: tape_score}

        best_material = max(scores, key=lambda k: scores[k])
        best_score = scores[best_material]

        if best_score > 0.5:
            logger.info(f"Detected mono material: {best_material.value} (score={best_score:.2f})")
            return best_material
        else:
            logger.warning(f"Mono material unclear (best score={best_score:.2f}), using UNKNOWN")
            return MaterialType.UNKNOWN

    def _detect_stereo_material(self, audio: np.ndarray) -> MaterialType:
        """
        Unterscheidet alle Stereo-Material-Typen (Vinyl, Tape, Reel-Tape, DAT, CD, MP3, AAC, MiniDisc, Streaming).

        Verwendet Scoring-System statt binärer Thresholds für robustere Detektion.
        """
        audio_mono = np.mean(audio, axis=1)

        # === Feature-Extraction ===

        # 1. Rumble-Detection (Vinyl-typisch)
        freqs, psd = signal.welch(audio_mono, self.sample_rate, nperseg=4096)
        _psd_tot = np.maximum(np.sum(psd), 1e-12)  # §3.1: Zero-Division-Guard bei Stille
        rumble_energy = np.sum(psd[freqs < 60]) / _psd_tot

        # 2. Compression artifacts (Streaming/MP3/AAC)
        compression_score = self._detect_compression_artifacts(audio_mono).severity

        # 3. Digital artifacts (CD/DAT)
        digital_score = self._detect_digital_artifacts(audio_mono).severity

        # 4. High-freq noise / Tape hiss
        hf_noise_score = self._detect_high_freq_noise(audio_mono).severity

        # 5. High-freq energy loss (MP3/AAC low-pass effect)
        hf_energy = np.sum(psd[freqs > 15000]) / _psd_tot if self.sample_rate >= 44100 else 0.0
        hf_loss_indicator = 1.0 - (hf_energy / 0.05)  # Normalized: 0.05 = full HF, 0.0 = no HF
        hf_loss_indicator = np.clip(hf_loss_indicator, 0.0, 1.0)

        # 6. Crackle (Vinyl-typisch)
        crackle_score = self._detect_crackle(audio_mono).severity

        # 7. Wow/Flutter (analog media — IEC 60386)
        wow_flutter_score = max(
            self._detect_wow(audio_mono).severity,
            self._detect_flutter(audio_mono).severity,
        )

        # 8. Clicks (alle analog, aber unterschiedliche Pattern)
        click_score = self._detect_clicks(audio_mono).severity

        # === Material Scoring ===
        scores = {}

        # VINYL Score
        vinyl_score = 0.0
        vinyl_score += crackle_score * 3.0  # Crackle ist SEHR typisch für Vinyl
        vinyl_score += rumble_energy * 10.0  # Rumble ist Vinyl-spezifisch
        vinyl_score += wow_flutter_score * 1.5  # Analog-Medium
        vinyl_score += click_score * 1.0  # Moderate Clicks
        vinyl_score -= compression_score * 2.0  # Vinyl hat keine Compression
        vinyl_score -= digital_score * 2.0  # Vinyl ist analog
        scores[MaterialType.VINYL] = max(0, vinyl_score)

        # TAPE (Cassette) Score
        tape_score = 0.0
        tape_score += hf_noise_score * 4.0  # Tape hiss sehr charakteristisch
        tape_score += wow_flutter_score * 2.0  # Noch typischer für Tape
        tape_score += click_score * 0.5  # Wenige Clicks
        tape_score -= crackle_score * 2.0  # Tape hat kein Crackle
        tape_score -= rumble_energy * 5.0  # Tape hat keinen Rumble
        tape_score -= compression_score * 2.0  # Tape analog
        scores[MaterialType.TAPE] = max(0, tape_score)

        # REEL_TAPE (Professional) Score
        reel_tape_score = 0.0
        reel_tape_score += hf_noise_score * 3.0  # Less hiss than cassette (better quality)
        reel_tape_score += wow_flutter_score * 1.0  # Less than cassette, but present
        reel_tape_score += click_score * 0.3  # Very few clicks
        reel_tape_score -= crackle_score * 2.0  # No crackle
        reel_tape_score -= rumble_energy * 5.0  # No rumble
        reel_tape_score -= compression_score * 2.0  # Analog
        reel_tape_score -= digital_score * 1.5  # Analog
        # Boost if hiss present but better quality than cassette
        if hf_noise_score > 0.2 and hf_noise_score < 0.5:
            reel_tape_score += 1.0  # Professional tape sweet spot
        scores[MaterialType.REEL_TAPE] = max(0, reel_tape_score)

        # DAT (Digital Audio Tape) Score
        dat_score = 0.0
        dat_score += digital_score * 2.0  # Some digital artifacts
        dat_score -= crackle_score * 3.0  # No crackle
        dat_score -= rumble_energy * 10.0  # No rumble
        dat_score -= wow_flutter_score * 3.0  # Digital, no wow/flutter
        dat_score -= hf_noise_score * 2.0  # No tape hiss (digital)
        dat_score -= compression_score * 1.5  # Lossless, minimal compression
        # DAT is cleaner than CD but still digital
        if digital_score > 0.1 and compression_score < 0.2:
            dat_score += 1.5  # Clean digital indicator
        scores[MaterialType.DAT] = max(0, dat_score)

        # CD_DIGITAL Score
        cd_score = 0.0
        cd_score += digital_score * 3.0  # Digital artifacts typisch
        cd_score -= crackle_score * 3.0  # CD hat kein Crackle
        cd_score -= rumble_energy * 10.0  # CD hat keinen Rumble
        cd_score -= wow_flutter_score * 2.0  # CD hat kein Wow/Flutter
        cd_score -= hf_noise_score * 2.0  # CD hat keinen Tape-Hiss
        cd_score -= compression_score * 1.0  # CD meist lossless
        scores[MaterialType.CD_DIGITAL] = max(0, cd_score)

        # MP3_LOW (<128 kbps) Score
        mp3_low_score = 0.0
        mp3_low_score += compression_score * 4.5  # Heavy compression
        mp3_low_score += hf_loss_indicator * 3.0  # Significant HF loss
        mp3_low_score += digital_score * 1.0  # Codec artifacts
        mp3_low_score -= crackle_score * 3.0  # No crackle
        mp3_low_score -= rumble_energy * 10.0  # No rumble
        mp3_low_score -= wow_flutter_score * 3.0  # No wow/flutter
        # Heavy compression + HF loss indicator
        if compression_score > 0.6 and hf_loss_indicator > 0.5:
            mp3_low_score += 2.0  # Strong MP3 low bitrate indicator
        scores[MaterialType.MP3_LOW] = max(0, mp3_low_score)

        # MP3_HIGH (≥128 kbps) Score
        mp3_high_score = 0.0
        mp3_high_score += compression_score * 3.5  # Moderate compression
        mp3_high_score += hf_loss_indicator * 1.5  # Some HF loss
        mp3_high_score += digital_score * 0.5  # Fewer artifacts than low bitrate
        mp3_high_score -= crackle_score * 3.0  # No crackle
        mp3_high_score -= rumble_energy * 10.0  # No rumble
        mp3_high_score -= wow_flutter_score * 3.0  # No wow/flutter
        # Moderate compression, less HF loss than low bitrate
        if compression_score > 0.3 and compression_score < 0.6 and hf_loss_indicator < 0.5:
            mp3_high_score += 1.5  # MP3 high bitrate sweet spot
        scores[MaterialType.MP3_HIGH] = max(0, mp3_high_score)

        # AAC (M4A) Score
        aac_score = 0.0
        aac_score += compression_score * 3.0  # Efficient compression
        aac_score += hf_loss_indicator * 0.8  # Better HF preservation than MP3
        aac_score += digital_score * 0.3  # Minimal artifacts
        aac_score -= crackle_score * 3.0  # No crackle
        aac_score -= rumble_energy * 10.0  # No rumble
        aac_score -= wow_flutter_score * 3.0  # No wow/flutter
        # AAC is better quality than MP3 at same bitrate
        if compression_score > 0.2 and compression_score < 0.5 and hf_loss_indicator < 0.3:
            aac_score += 1.8  # AAC efficiency indicator
        scores[MaterialType.AAC] = max(0, aac_score)

        # MINIDISC (ATRAC) Score
        minidisc_score = 0.0
        minidisc_score += compression_score * 4.0  # ATRAC aggressive compression
        minidisc_score += hf_loss_indicator * 2.5  # Significant HF artifacts
        minidisc_score += digital_score * 1.5  # ATRAC-specific artifacts
        minidisc_score -= crackle_score * 3.0  # No crackle
        minidisc_score -= rumble_energy * 10.0  # No rumble
        minidisc_score -= wow_flutter_score * 3.0  # Digital, no wow/flutter
        # ATRAC has specific artifacts pattern (aggressive + HF issues)
        if compression_score > 0.5 and hf_loss_indicator > 0.4 and digital_score > 0.3:
            minidisc_score += 1.5  # ATRAC signature
        scores[MaterialType.MINIDISC] = max(0, minidisc_score)

        # STREAMING Score
        streaming_score = 0.0
        streaming_score += compression_score * 4.0  # Hauptindikator
        streaming_score -= crackle_score * 3.0  # Kein Crackle
        streaming_score -= rumble_energy * 10.0  # Kein Rumble
        streaming_score -= wow_flutter_score * 2.0  # Kein Wow/Flutter
        streaming_score -= digital_score * 1.0  # Compression ≠ digital artifacts
        scores[MaterialType.STREAMING] = max(0, streaming_score)

        # SHELLAC Score (aus Stereo unwahrscheinlich, aber möglich bei Remaster)
        shellac_score = 0.0
        shellac_score += click_score * 3.0  # Sehr viele Clicks
        shellac_score += crackle_score * 2.0  # Auch Crackle
        shellac_score -= hf_noise_score * 2.0  # Wenig HF (alt)
        shellac_score -= compression_score * 2.0  # Analog
        # Shellac aus Stereo sehr unwahrscheinlich
        shellac_score *= 0.3  # Penalty für Stereo-Context
        scores[MaterialType.SHELLAC] = max(0, shellac_score)

        # Wähle Material mit höchstem Score
        if not scores or max(scores.values()) < 0.5:  # Minimaler Confidence-Threshold
            return MaterialType.UNKNOWN

        best_material = max(scores.items(), key=lambda x: x[1])
        logger.debug(f"Material scores: {scores}")
        logger.info(f"Detected material: {best_material[0].value} (score: {best_material[1]:.2f})")

        return best_material[0]

    # ========== DEFECT DETECTORS (11 Typen) ==========

    @staticmethod
    def _sample_locations_evenly(locations: list, max_n: int) -> list:
        """Return up to max_n locations sampled evenly across the full duration.

        Instead of truncating with [:max_n] (which biases markers to the start of
        the recording), this method picks representatives at uniform stride so that
        the returned positions span the entire audio length.
        """
        if len(locations) <= max_n:
            return locations
        step = len(locations) / max_n
        return [locations[int(i * step)] for i in range(max_n)]

    def _detect_clicks(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Clicks (kurze, impulsive Störungen).

        Anti-FP guards:
        1. Outlier-robust threshold: requires diff values to be ≥ 5× the median
           diff AND above the 99.9th percentile (scaled by material sensitivity).
           This prevents tonal signals (sine, strings) from triggering.
        2. Transient-width discrimination: only events spanning ≤ MAX_CLICK_WIDTH
           samples are accepted.  Musical transients (drums, plucks) are wider
           than true clicks (≤ 0.15 ms).
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Inter-sample difference (classical click detection)
        diff = np.abs(np.diff(audio))
        local_median = float(np.median(diff)) + 1e-10

        # --- Anti-FP: outlier-robust threshold ---
        # Clicks are extreme outliers (>> 5× median diff).  Using only
        # percentile(99.5) × factor fails for pure tones where the 99.5th
        # percentile is close to the normal diff maximum.
        min_outlier_factor = 5.0
        base_threshold = max(
            float(np.percentile(diff, 99.9)),
            local_median * min_outlier_factor,
        )
        threshold_dynamic = self.thresholds[DefectType.CLICKS] * base_threshold

        # Find candidate click samples
        click_mask = diff > threshold_dynamic
        click_indices = np.where(click_mask)[0]

        if len(click_indices) == 0:
            return DefectScore(DefectType.CLICKS, 0.0, 1.0)

        # Group nearby exceedances (within 1 ms = one click event)
        group_window = max(1, int(0.001 * self.sample_rate))
        click_groups = []
        current_group = [click_indices[0]]
        for idx in click_indices[1:]:
            if idx - current_group[-1] < group_window:
                current_group.append(idx)
            else:
                click_groups.append(current_group)
                current_group = [idx]
        click_groups.append(current_group)

        # --- Anti-FP: transient width discrimination ---
        # True clicks are extremely narrow (≤ 0.15 ms ≈ 7 samples @ 48 kHz).
        # Musical transients (snare, pluck, piano hammer) are wider (> 20 samples).
        max_click_width = max(5, int(0.00015 * self.sample_rate))  # 0.15 ms
        verified_groups = []
        for group in click_groups:
            group_width = group[-1] - group[0] + 1
            if group_width <= max_click_width:
                verified_groups.append(group)

        if len(verified_groups) == 0:
            return DefectScore(
                DefectType.CLICKS,
                0.0,
                0.95,
                metadata={"click_rate": 0.0, "total_clicks": 0, "rejected_as_transients": len(click_groups)},
            )

        # Convert to timestamps — sample evenly across full duration (not just first 50)
        MAX_LOCATIONS = 50
        all_locations = [(group[0] / self.sample_rate, group[-1] / self.sample_rate) for group in verified_groups]
        locations = self._sample_locations_evenly(all_locations, MAX_LOCATIONS)

        # Severity: click-rate per second (uses full count, not capped)
        duration = len(audio) / self.sample_rate
        click_rate = len(verified_groups) / duration
        severity = min(1.0, click_rate / 20)  # 20 clicks/sec = severity 1.0

        return DefectScore(
            defect_type=DefectType.CLICKS,
            severity=severity,
            confidence=0.9,
            locations=locations,
            metadata={
                "click_rate": click_rate,
                "total_clicks": len(verified_groups),
                "rejected_as_transients": len(click_groups) - len(verified_groups),
            },
        )

    def _detect_crackle(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Crackle (kontinuierliches leises Knistern, z.B. Vinyl-Surface-Noise).

        Anti-FP guard: brilliant recordings (cymbals, synthesizers, sibilants) have
        naturally high HF energy that can trigger the HP-filtered envelope detector.
        We compute a broadband HF-energy ratio and apply a discount when the HF
        content is smoothly distributed (tonal HF) rather than impulsive (crackle).
        True crackle has high kurtosis in the HP-filtered signal; tonal HF does not.
        """
        # Crackle = High-Pass filtered + Envelope Detection
        sos = signal.butter(4, 3000, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Envelope via Hilbert-Transform
        analytic_signal: np.ndarray = signal.hilbert(audio_hp)  # type: ignore[assignment]
        envelope = np.abs(analytic_signal)

        # Smoothed envelope
        window_size = int(0.05 * self.sample_rate)  # 50ms window
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Crackle detection: regions with elevated noise-floor
        threshold = self.thresholds[DefectType.CRACKLE] * np.percentile(envelope_smooth, 95)
        crackle_mask = envelope_smooth > threshold

        severity_raw = float(np.sum(crackle_mask)) / len(audio)

        # --- Anti-FP: kurtosis check on HP-filtered signal ---
        # Crackle is impulsive → high kurtosis (>> 3.0 for Gaussian).
        # Tonal HF (cymbals, synths, sibilants) has near-sinusoidal distribution
        # → kurtosis ≈ 1.5 (pure sine) to 3.0 (Gaussian noise).
        hp_kurtosis = 3.0  # Gaussian default (safe fallback)
        hp_std = float(np.std(audio_hp))
        if hp_std > 1e-8:
            hp_mean = float(np.mean(audio_hp))
            hp_kurtosis = float(np.mean(((audio_hp - hp_mean) / hp_std) ** 4))

        kurtosis_discount = 1.0
        if hp_kurtosis < 4.0:
            # Clearly tonal or Gaussian HF — NOT impulsive crackle.
            # Hard cap severity to near-zero regardless of energy ratio.
            kurtosis_discount = 0.0
        elif hp_kurtosis < 6.0:
            # Borderline zone — scale linearly
            kurtosis_discount = max(0.1, (hp_kurtosis - 4.0) / 2.0)
        # Kurtosis > 6 → impulsive, keep full weight

        severity = min(1.0, severity_raw * 10 * kurtosis_discount)

        # Find connected regions
        from scipy.ndimage import label

        labeled_array, num_features = label(crackle_mask)  # type: ignore[misc]
        locations = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) > int(0.01 * self.sample_rate):  # At least 10ms
                start = indices[0] / self.sample_rate
                end = indices[-1] / self.sample_rate
                locations.append((start, end))

        confidence = 0.8
        if hp_kurtosis < 4.0:
            confidence = 0.3  # Very low confidence when HF is clearly tonal

        return DefectScore(
            defect_type=DefectType.CRACKLE,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, 50),
            metadata={
                "crackle_percentage": severity_raw * 100,
                "hp_kurtosis": hp_kurtosis,
                "kurtosis_discount": kurtosis_discount,
            },
        )

    def _detect_hum(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hum (50Hz/60Hz AC-Brummen und Harmonische)."""
        # FFT für Frequenz-Analyse
        fft_size = min(len(audio), int(4 * self.sample_rate))  # 4 Sekunden oder weniger
        freqs = fft.rfftfreq(fft_size, 1 / self.sample_rate)
        spectrum = np.abs(fft.rfft(audio[:fft_size]))  # type: ignore[call-overload]

        # Suche nach 50Hz und 60Hz + Harmonische
        hum_freqs_50 = [50, 100, 150, 200, 250, 300]
        hum_freqs_60 = [60, 120, 180, 240, 300, 360]

        def measure_hum(hum_freqs):
            hum_energy = 0
            for f in hum_freqs:
                idx = np.argmin(np.abs(freqs - f))
                # Energie in ±2Hz Band
                band_start = max(0, idx - 2)
                band_end = min(len(spectrum), idx + 3)
                hum_energy += np.sum(spectrum[band_start:band_end] ** 2)
            return hum_energy

        hum_50 = measure_hum(hum_freqs_50)
        hum_60 = measure_hum(hum_freqs_60)
        total_energy = np.sum(spectrum**2)

        hum_ratio = max(hum_50, hum_60) / (total_energy + 1e-10)

        # Severity basierend auf Hum-Stärke
        threshold = self.thresholds[DefectType.HUM]
        severity = min(1.0, hum_ratio / (threshold * 0.001))  # 0.1% = max severity

        hum_freq = 50 if hum_50 > hum_60 else 60

        return DefectScore(
            defect_type=DefectType.HUM,
            severity=severity,
            confidence=0.95,
            locations=[],  # Hum ist typischerweise durchgehend
            metadata={"hum_frequency": hum_freq, "hum_ratio": hum_ratio},
        )

    def _detect_wow(self, audio: np.ndarray) -> DefectScore:
        """Detect WOW: slow pitch modulation < 0.5 Hz (IEC 60386, Blum 1984).

        WOW arises from motor eccentricity, platter warps, capstan slip (< 0.5 Hz).
        Algorithm: pYIN-style energy-RMS on 500 ms windows → FFT of RMS time series
        → energy below 0.5 Hz indicates WOW.
        """
        n = len(audio)
        if n < self.sample_rate:  # less than 1 s — too short
            return DefectScore(DefectType.WOW, 0.0, 0.3)

        # 500 ms windows, 250 ms hop to track slow speed variations
        win_len = max(1, int(0.5 * self.sample_rate))
        hop = max(1, win_len // 2)
        rms_series = np.array(
            [float(np.sqrt(np.mean(audio[i : i + win_len] ** 2) + 1e-12)) for i in range(0, n - win_len, hop)],
            dtype=np.float32,
        )

        if len(rms_series) < 4:
            return DefectScore(DefectType.WOW, 0.0, 0.3)

        # Normalize and check spectral content below 0.5 Hz
        rms_series = rms_series / (rms_series.mean() + 1e-12)
        frame_rate = self.sample_rate / hop  # frames per second of rms_series
        fft_rms = np.abs(np.fft.rfft(rms_series - rms_series.mean()))
        freqs_rms = np.fft.rfftfreq(len(rms_series), d=1.0 / frame_rate)

        wow_mask = freqs_rms < 0.5
        total_energy = float(fft_rms.sum() + 1e-12)
        wow_energy = float(fft_rms[wow_mask].sum()) if wow_mask.any() else 0.0
        wow_ratio = wow_energy / total_energy

        threshold = self.thresholds.get(DefectType.WOW, 0.5)
        severity = float(np.clip(wow_ratio / (threshold * 0.3 + 1e-12), 0.0, 1.0))

        return DefectScore(
            defect_type=DefectType.WOW,
            severity=severity,
            confidence=0.75,
            locations=[],
            metadata={"wow_energy_ratio": wow_ratio, "frame_rate_hz": float(frame_rate)},
        )

    def _detect_transport_bump(self, audio: np.ndarray) -> DefectScore:
        """Detect TRANSPORT_BUMP: impulsive micro-speed jumps (50–300 ms) from tape transport shocks.

        These are distinct from continuous wow/flutter: they manifest as a sudden,
        abrupt pitch excursion (±1–5 %) with simultaneous amplitude perturbation,
        typically caused by mechanical jolts (vibration, capstan slip, tape-guide bump).

        Algorithm:
            1. Short-window (20 ms hop) RMS envelope → detect amplitude transients
            2. Short-window zero-crossing-rate (ZCR) → detect pitch-rate transients
            3. Combined: both RMS and ZCR show transient within ±150 ms → transport bump
            4. Minimum duration 50 ms, maximum 300 ms → filter by event width
            5. Severity = max(event_magnitudes); confidence from event count + material

        Returns:
            DefectScore with locations [(start_s, end_s), ...] for each bump event.
        """
        n = len(audio)
        sr = self.sample_rate
        min_dur_s = 0.05  # 50 ms minimum bump width
        max_dur_s = 0.30  # 300 ms maximum bump width
        # Minimum 2 s of audio to analyse
        if n < sr * 2:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        # Parameters
        hop_s = 0.020  # 20 ms hop
        win_s = 0.040  # 40 ms window
        hop = max(1, int(hop_s * sr))
        win = max(1, int(win_s * sr))
        n_frames = (n - win) // hop

        if n_frames < 10:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        # 1. Compute short-time RMS envelope
        rms_env = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            rms_env[i] = float(np.sqrt(np.mean(audio[start : start + win] ** 2) + 1e-12))

        # 2. Compute short-time ZCR (zero-crossing rate) — proxy for instantaneous spectral centroid
        zcr_env = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + win]
            zcr_env[i] = float(np.sum(np.abs(np.diff(np.signbit(frame))))) / max(1.0, float(win))

        # 3. Compute first-order derivatives (rate of change)
        rms_diff = np.abs(np.diff(rms_env))
        zcr_diff = np.abs(np.diff(zcr_env))

        if len(rms_diff) < 4:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        # 4. Threshold: adaptive (median + 4 × MAD for both)
        rms_med = float(np.median(rms_diff))
        rms_mad = float(np.median(np.abs(rms_diff - rms_med))) + 1e-12
        zcr_med = float(np.median(zcr_diff))
        zcr_mad = float(np.median(np.abs(zcr_diff - zcr_med))) + 1e-12

        rms_thr = rms_med + 4.0 * rms_mad
        zcr_thr = zcr_med + 4.0 * zcr_mad

        # 5. Find frames where BOTH rms_diff AND zcr_diff exceed their thresholds
        #    within a ±3 frame (±60 ms) window → simultaneous perturbation
        rms_peaks = rms_diff > rms_thr
        zcr_peaks = zcr_diff > zcr_thr

        # Dilate zcr_peaks by ±3 frames for temporal alignment tolerance
        dilated_zcr = np.zeros_like(zcr_peaks)
        for shift in range(-3, 4):
            shifted = np.roll(zcr_peaks, shift)
            if shift < 0:
                shifted[shift:] = False
            elif shift > 0:
                shifted[:shift] = False
            dilated_zcr |= shifted

        candidates = rms_peaks & dilated_zcr

        # 6. Group consecutive candidate frames into events, filter by duration
        min_frames = max(1, int(min_dur_s / hop_s))
        max_frames = max(1, int(max_dur_s / hop_s))

        locations: list[tuple[float, float]] = []
        magnitudes: list[float] = []

        i = 0
        while i < len(candidates):
            if candidates[i]:
                start_frame = i
                while i < len(candidates) and candidates[i]:
                    i += 1
                end_frame = i
                event_len = end_frame - start_frame
                if min_frames <= event_len <= max_frames:
                    t_start = float(start_frame * hop) / sr
                    t_end = float(min(end_frame * hop + win, n)) / sr
                    locations.append((t_start, t_end))
                    # Magnitude: max RMS delta within event
                    mag = float(np.max(rms_diff[start_frame:end_frame]))
                    magnitudes.append(mag)
            else:
                i += 1

        n_bumps = len(locations)
        if n_bumps == 0:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.6)

        # 7. Severity: based on magnitude and frequency of events
        duration_s = n / sr
        bump_density = n_bumps / max(1.0, duration_s / 60.0)  # bumps per minute
        max_mag = float(max(magnitudes)) if magnitudes else 0.0
        # Normalize magnitude relative to global RMS
        global_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        rel_mag = max_mag / (global_rms + 1e-12)

        severity = float(
            np.clip(
                0.3 * min(1.0, bump_density / 10.0) + 0.7 * min(1.0, rel_mag / 5.0),
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(0.5 + 0.05 * n_bumps, 0.5, 0.95))

        logger.info(
            "transport_bump detection: n_bumps=%d, density=%.1f/min, max_rel_mag=%.2f, severity=%.3f",
            n_bumps,
            bump_density,
            rel_mag,
            severity,
        )

        return DefectScore(
            defect_type=DefectType.TRANSPORT_BUMP,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "n_bumps": n_bumps,
                "bump_density_per_min": bump_density,
                "max_relative_magnitude": rel_mag,
                "magnitudes": [float(m) for m in magnitudes[:20]],  # limit for metadata size
            },
        )

    def _detect_flutter(self, audio: np.ndarray) -> DefectScore:
        """Detect FLUTTER: rapid pitch modulation 0.5–200 Hz (IEC 60386).

        FLUTTER arises from mechanical vibration, guide rollers, pinch roller
        irregularities in the 0.5–200 Hz range.
        Algorithm: ZCR on 50 ms windows → FFT of ZCR series → energy in [0.5, 200] Hz.
        """
        n = len(audio)
        if n < self.sample_rate // 2:  # less than 500 ms — too short
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        # 50 ms windows, 25 ms hop to track faster mechanical vibrations
        win_len = max(1, int(0.05 * self.sample_rate))
        hop = max(1, win_len // 2)
        zcr_series = np.array(
            [float(np.sum(np.abs(np.diff(np.signbit(audio[i : i + win_len]))))) for i in range(0, n - win_len, hop)],
            dtype=np.float32,
        )

        if len(zcr_series) < 8:
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        frame_rate = self.sample_rate / hop
        fft_zcr = np.abs(np.fft.rfft(zcr_series - zcr_series.mean()))
        freqs_zcr = np.fft.rfftfreq(len(zcr_series), d=1.0 / frame_rate)

        flutter_mask = (freqs_zcr >= 0.5) & (freqs_zcr <= 200.0)
        total_energy = float(fft_zcr.sum() + 1e-12)
        flutter_energy = float(fft_zcr[flutter_mask].sum()) if flutter_mask.any() else 0.0
        flutter_ratio = flutter_energy / total_energy

        threshold = self.thresholds.get(DefectType.FLUTTER, 0.5)
        severity = float(np.clip(flutter_ratio / (threshold * 0.4 + 1e-12), 0.0, 1.0))

        return DefectScore(
            defect_type=DefectType.FLUTTER,
            severity=severity,
            confidence=0.70,
            locations=[],
            metadata={"flutter_energy_ratio": flutter_ratio, "frame_rate_hz": float(frame_rate)},
        )

    def _detect_wow_flutter(self, audio: np.ndarray) -> DefectScore:
        """Combined WOW+FLUTTER score — max of both sub-detectors.

        Convenience wrapper combining _detect_wow and _detect_flutter into a
        single DefectScore (worst-case severity) for legacy callers.
        """
        wow = self._detect_wow(
            audio if audio.ndim == 1 else audio.mean(axis=1 if audio.shape[1] > audio.shape[0] else 0)
        )
        flutter = self._detect_flutter(
            audio if audio.ndim == 1 else audio.mean(axis=1 if audio.shape[1] > audio.shape[0] else 0)
        )
        if wow.severity >= flutter.severity:
            return wow
        return flutter

    def _detect_azimuth_error(self, audio: np.ndarray) -> DefectScore:
        """Detect AZIMUTH_ERROR: playback-head tilt causing HF L/R phase slope > 20°/kHz.

        Azimuth errors occur when the recording head angle differs from the playback
        head angle, causing a phase difference between L and R that grows linearly
        with frequency (PHD-Slope criterion, IEC 60386).

        For mono input: returns severity 0.0 with confidence 0.0 (not applicable).
        For stereo: computes L-R cross-spectrum phase difference, fits a linear model
        vs. frequency, and reports the slope in °/kHz.
        """
        if audio.ndim != 2 or audio.shape[0] < 2 or audio.shape[1] < 2:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.0)

        # Determine stereo layout: expect (N, 2) samples-first
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            left = audio[0]
            right = audio[1]
        else:
            left = audio[:, 0]
            right = audio[:, 1]

        n = min(len(left), len(right))
        if n < 512:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        # Limit to first 10 s for speed
        n_use = min(n, 10 * self.sample_rate)
        left = left[:n_use]
        right = right[:n_use]

        # Cross-spectrum via Welch-style averaged FFT over 4096-sample frames
        fft_n = 4096
        hop = fft_n // 2
        phase_diffs = []
        freqs_hz = np.fft.rfftfreq(fft_n, d=1.0 / self.sample_rate)

        for i in range(0, n_use - fft_n, hop):
            L = np.fft.rfft(left[i : i + fft_n] * np.hanning(fft_n))
            R = np.fft.rfft(right[i : i + fft_n] * np.hanning(fft_n))
            cross = L * np.conj(R)
            phase_diffs.append(np.angle(cross))  # radians, per bin

        if not phase_diffs:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        mean_phase = np.mean(np.array(phase_diffs), axis=0)  # (N_fft//2+1,)
        mean_phase_deg = np.degrees(np.unwrap(mean_phase))

        # Linear fit in 1–8 kHz range (avoids DC and near-Nyquist noise)
        fit_mask = (freqs_hz >= 1000.0) & (freqs_hz <= 8000.0)
        if fit_mask.sum() < 4:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        x = freqs_hz[fit_mask] / 1000.0  # kHz
        y = mean_phase_deg[fit_mask]
        # Least-squares slope (°/kHz)
        slope = float(np.polyfit(x, y, 1)[0])
        phd_slope_abs = abs(slope)

        # PHD-Slope threshold: > 20°/kHz indicates significant azimuth error
        threshold = self.thresholds.get(DefectType.AZIMUTH_ERROR, 0.5)
        severity = float(np.clip((phd_slope_abs - 5.0) / (20.0 + 1e-12), 0.0, 1.0))
        # Apply material sensitivity: higher threshold → lower sensitivity
        severity = float(np.clip(severity / (threshold + 1e-12), 0.0, 1.0)) if threshold < 1.0 else 0.0

        confidence = 0.80 if n_use >= 3 * self.sample_rate else 0.50

        return DefectScore(
            defect_type=DefectType.AZIMUTH_ERROR,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={"phd_slope_deg_per_khz": phd_slope_abs},
        )

    def _detect_stereo_imbalance(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Stereo-Imbalance (L/R Kanal-Unterschiede)."""
        if audio.ndim != 2:
            return DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)

        left, right = audio[:, 0], audio[:, 1]

        # RMS-Level Vergleich
        rms_left = np.sqrt(np.mean(left**2))
        rms_right = np.sqrt(np.mean(right**2))

        if rms_left < 1e-10 or rms_right < 1e-10:  # Stille
            return DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.5)

        balance_ratio = min(rms_left, rms_right) / max(rms_left, rms_right)
        imbalance = 1.0 - balance_ratio

        # Severity
        threshold = self.thresholds[DefectType.STEREO_IMBALANCE]
        severity = min(1.0, imbalance / (1 - threshold))  # > 40% Imbalance = max

        return DefectScore(
            defect_type=DefectType.STEREO_IMBALANCE,
            severity=severity,
            confidence=0.9,
            locations=[],
            metadata={"rms_left": rms_left, "rms_right": rms_right, "balance_ratio": balance_ratio},
        )

    def _detect_digital_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Digital-Artifacts (Quantisierungs-Rauschen, Clipping, Aliasing)."""
        # Clipping-Detection
        clipping_count = np.sum(np.abs(audio) > 0.99)
        clipping_ratio = clipping_count / len(audio)

        # Quantisierungs-Rauschen: Analyse der LSBs
        # Bei 16-bit Audio sollten die untersten Bits zufällig sein
        audio_int = (audio * 32767).astype(np.int16)
        lsb = np.abs(audio_int % 2)
        lsb_randomness = np.std(lsb)  # Sollte ~0.5 sein für echte Aufnahmen

        quantization_artifact = 1.0 - min(1.0, lsb_randomness / 0.5)

        # Kombinierte Severity
        severity = max(clipping_ratio * 10, quantization_artifact)
        threshold = self.thresholds[DefectType.DIGITAL_ARTIFACTS]
        severity = min(1.0, severity / threshold)

        return DefectScore(
            defect_type=DefectType.DIGITAL_ARTIFACTS,
            severity=severity,
            confidence=0.8,
            locations=[],
            metadata={"clipping_ratio": clipping_ratio, "quantization_score": quantization_artifact},
        )

    def _detect_low_freq_rumble(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Low-Frequency Rumble (< 60 Hz, z.B. Turntable-Rumble)."""
        # Low-Pass Filter
        sos = signal.butter(4, 60, btype="low", fs=self.sample_rate, output="sos")
        audio_lp = signal.sosfilt(sos, audio)

        # Energie-Vergleich
        rumble_energy = np.sum(audio_lp**2)
        total_energy = np.sum(audio**2)

        rumble_ratio = rumble_energy / (total_energy + 1e-10)

        threshold = self.thresholds[DefectType.LOW_FREQ_RUMBLE]
        severity = min(1.0, rumble_ratio / (threshold * 0.1))  # 10% = max

        return DefectScore(
            defect_type=DefectType.LOW_FREQ_RUMBLE,
            severity=severity,
            confidence=0.9,
            locations=[],
            metadata={"rumble_ratio": rumble_ratio},
        )

    def _detect_high_freq_noise(self, audio: np.ndarray) -> DefectScore:
        """Erkennt High-Frequency Noise (> 8 kHz, z.B. Tape-Hiss)."""
        # High-Pass Filter
        sos = signal.butter(4, 8000, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Energie-Vergleich
        hf_energy = np.sum(audio_hp**2)
        total_energy = np.sum(audio**2)

        hf_ratio = hf_energy / (total_energy + 1e-10)

        threshold = self.thresholds[DefectType.HIGH_FREQ_NOISE]
        severity = min(1.0, hf_ratio / (threshold * 0.05))  # 5% = max

        return DefectScore(
            defect_type=DefectType.HIGH_FREQ_NOISE,
            severity=severity,
            confidence=0.85,
            locations=[],
            metadata={"hf_ratio": hf_ratio},
        )

    def _detect_compression_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Compression-Artifacts (MP3/AAC Pre-Echo, Ringing).

        Anti-FP guards:
        1. HF bandwidth cross-validation: real codecs cut/attenuate HF > 14-16 kHz.
           If full HF bandwidth present → likely tonal, not compressed.
        2. SFM temporal variance: codecs produce uniformly low SFM; natural tonal
           content has high SFM variance across frames.
        3. Spectral concentration: narrowband signals (sine, bass guitar) naturally
           have low SFM without being compressed.  If > 80% of energy is in < 5%
           of frequency bins → tonal narrowband → heavy discount.
        """
        # STFT for Time-Frequency Analysis
        _f, _t, Zxx = signal.stft(audio, self.sample_rate, nperseg=1024)
        spectrogram = np.abs(Zxx)

        # Spectral Flatness Measure (SFM)
        geometric_mean = np.exp(np.mean(np.log(spectrogram + 1e-10), axis=0))
        arithmetic_mean = np.mean(spectrogram, axis=0)
        sfm = geometric_mean / (arithmetic_mean + 1e-10)

        compression_score = 1.0 - float(np.mean(sfm))  # Low SFM → high score

        # --- Anti-FP: spectral concentration check ---
        # Narrowband signals (pure tones, bass) have low SFM naturally.
        # If most energy is concentrated in a few bins → not compression.
        avg_spectrum = np.mean(spectrogram**2, axis=1)  # average power per freq bin
        total_power = float(np.sum(avg_spectrum)) + 1e-12
        sorted_power = np.sort(avg_spectrum)[::-1]
        n_bins = len(avg_spectrum)
        top_5pct = max(1, int(0.05 * n_bins))
        top_5pct_power = float(np.sum(sorted_power[:top_5pct]))
        spectral_concentration = top_5pct_power / total_power

        concentration_discount = 1.0
        if spectral_concentration > 0.80:
            # > 80% of energy in < 5% of bins → narrowband tonal signal
            concentration_discount = max(0.05, 1.0 - (spectral_concentration - 0.80) * 5.0)

        # --- Anti-FP: HF bandwidth cross-validation ---
        # Real MP3/AAC always cuts or heavily attenuates HF above ~15 kHz.
        # If full HF bandwidth is present → signal is tonal, not compressed.
        hf_penalty = 1.0  # 1.0 = no penalty, < 1.0 = reduce severity
        if self.sample_rate >= 32000:
            freqs_stft = np.asarray(_f)
            hf_cutoff = min(15000.0, self.sample_rate * 0.45)
            hf_mask = freqs_stft >= hf_cutoff
            if hf_mask.any():
                hf_energy = float(np.mean(spectrogram[hf_mask, :] ** 2))
                total_energy_stft = float(np.mean(spectrogram**2)) + 1e-12
                hf_ratio = hf_energy / total_energy_stft
                # Full HF present (> 1% of total energy) → likely not lossy-coded
                if hf_ratio > 0.01:
                    hf_penalty = max(0.15, 1.0 - hf_ratio * 8.0)

        # --- Anti-FP: temporal SFM variance check ---
        # Codec artifacts produce consistently low SFM across ALL frames.
        # Tonal music has high SFM variance (tonal vs. transient frames).
        sfm_std = float(np.std(sfm))
        tonality_discount = 1.0
        if sfm_std > 0.12:  # high variance → natural tonal content, not codec
            tonality_discount = max(0.2, 1.0 - (sfm_std - 0.12) * 3.0)

        compression_score *= hf_penalty * tonality_discount * concentration_discount

        threshold = self.thresholds[DefectType.COMPRESSION_ARTIFACTS]
        severity = min(1.0, compression_score / threshold)

        confidence = 0.7
        if spectral_concentration > 0.80:
            confidence = 0.3  # Narrowband signal — compression unlikely
        elif hf_penalty < 0.5 and sfm_std < 0.08:
            confidence = 0.5  # HF present + low variance → uncertain
        elif hf_penalty >= 0.9 and sfm_std < 0.06:
            confidence = 0.85  # HF loss confirmed + stable SFM → confident

        return DefectScore(
            defect_type=DefectType.COMPRESSION_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "spectral_flatness_mean": float(np.mean(sfm)),
                "sfm_std": sfm_std,
                "hf_penalty": hf_penalty,
                "tonality_discount": tonality_discount,
                "spectral_concentration": spectral_concentration,
                "concentration_discount": concentration_discount,
            },
        )

    def _detect_phase_issues(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Phase-Issues (L/R Phase-Differenzen, Mono-Kompatibilität)."""
        if audio.ndim != 2:
            return DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)

        left, right = audio[:, 0], audio[:, 1]

        # Mid/Side Analysis
        mid = (left + right) / 2
        side = (left - right) / 2

        # Phase-Issues: Zu viel Side-Energie (schlechte Mono-Compatibility)
        mid_energy = np.sum(mid**2)
        side_energy = np.sum(side**2)
        total_energy = mid_energy + side_energy

        if total_energy < 1e-10:
            return DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.5)

        side_ratio = side_energy / total_energy

        # Auch: Cross-Correlation zwischen L und R (sollte hoch sein)
        correlation = np.corrcoef(left, right)[0, 1]

        # Severity
        phase_score = max(0, side_ratio - 0.3)  # > 30% Side = problematisch
        corr_score = max(0, 0.5 - correlation)  # Correlation < 0.5 = problematisch

        severity = min(1.0, (phase_score + corr_score) * 2)

        return DefectScore(
            defect_type=DefectType.PHASE_ISSUES,
            severity=severity,
            confidence=0.8,
            locations=[],
            metadata={"side_ratio": side_ratio, "stereo_correlation": correlation},
        )

    def _detect_dropouts(self, audio: np.ndarray) -> DefectScore:
        """Detect dropouts (short silence / low-energy segments).

        Algorithm (v9.10.73 — material-adaptive):
        1. RMS in 5 ms windows (was 10 ms): catches short tape dropouts
        2. Material-adaptive threshold: tape/vinyl 20% median-RMS, digital 10%
        3. Minimum 1 window (>= 2.5 ms) — was 2 windows (10-20 ms)
        4. Severity = total_dropout_duration / audio_duration (was: count-based)
           -> even few but long dropouts are correctly weighted

        Typical tape dropout signatures:
        - Short level fluctuations (2-50 ms): oxide loss, tape guide errors
        - Level drops at tape start/end: tape leader, reel geometry
        """
        # 5 ms windows for finer detection (was 10 ms)
        window_size = max(1, int(0.005 * self.sample_rate))
        hop_size = window_size // 2

        rms_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i : i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)

        if not rms_values:
            return DefectScore(DefectType.DROPOUTS, 0.0, 0.5)

        rms_values = np.array(rms_values)

        # Material-adaptive threshold: analog media are more sensitive
        _DROPOUT_ANALOG_MATERIALS = {
            MaterialType.TAPE,
            MaterialType.REEL_TAPE,
            MaterialType.VINYL,
            MaterialType.SHELLAC,
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
            MaterialType.DAT,
        }
        _mat = getattr(self, "material_type", None)
        # Analog/DAT: 20% median-RMS (catches gradual level fades); digital: 10%
        _threshold_ratio = 0.20 if _mat in _DROPOUT_ANALOG_MATERIALS else 0.10
        median_rms = float(np.median(rms_values))
        dropout_threshold = _threshold_ratio * median_rms

        dropout_mask = rms_values < dropout_threshold

        # Find dropout events via connected-component labelling
        from scipy.ndimage import label

        labeled_array, num_dropouts = label(dropout_mask)  # type: ignore[misc]

        locations = []
        total_dropout_s = 0.0
        for i in range(1, num_dropouts + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) >= 1:  # 1 window suffices (>= 2.5 ms) — was 2
                start = indices[0] * hop_size / self.sample_rate
                end = (indices[-1] + 1) * hop_size / self.sample_rate
                locations.append((start, end))
                total_dropout_s += end - start

        # Severity: based on total dropout duration (not just count)
        # 1% dropout fraction -> severity 0.5; 2% -> 1.0
        duration = len(audio) / self.sample_rate
        dropout_fraction = total_dropout_s / max(duration, 1e-6)
        severity = float(np.nan_to_num(min(1.0, dropout_fraction / 0.02), nan=0.0))

        return DefectScore(
            defect_type=DefectType.DROPOUTS,
            severity=severity,
            confidence=0.85,
            locations=self._sample_locations_evenly(locations, 50),
            metadata={
                "dropout_count": len(locations),
                "dropout_rate": len(locations) / max(duration, 1e-6),
                "total_dropout_s": round(total_dropout_s, 4),
                "threshold_ratio": _threshold_ratio,
            },
        )

    # ========== NEU: Weltklasse-Detektoren ==========

    def _detect_clipping(self, audio: np.ndarray) -> DefectScore:
        """Detects Hard Clipping vs. SOFT_SATURATION via THD analysis (§6.3).

        Uses classify_clipping() from clipping_detection module (THD-based) when
        available.  SOFT_SATURATION (even harmonics — tube/tape character) → zero
        severity DefectScore with SOFT_SATURATION type; pipeline skips repair.
        CLIPPING (odd harmonics dominant + flat-tops > 0.1 %) → severity derived
        from flat_top ratio, DefectType.CLIPPING returned for repair.

        Falls back to amplitude-only detection when clipping_detection unavailable.
        """
        if len(audio) == 0:
            return DefectScore(DefectType.CLIPPING, 0.0, 0.0)

        peak = float(np.max(np.abs(audio)))
        if peak < 1e-6:
            return DefectScore(DefectType.CLIPPING, 0.0, 0.5)

        # §6.3 THD-based discrimination: CLIPPING → odd harmonics; SOFT_SATURATION → even harmonics
        if _CLIPPING_DETECTION_AVAILABLE:
            try:
                _clip_type = _classify_clipping(audio, self.sample_rate)
                if _clip_type == _ClippingType.SOFT_SATURATION:
                    # Tube/tape character — preserve, do NOT repair
                    logger.debug("§6.3 _detect_clipping: SOFT_SATURATION erkannt (even-harmonic profile) — kein Repair")
                    return DefectScore(
                        defect_type=DefectType.SOFT_SATURATION,
                        severity=0.0,
                        confidence=0.90,
                        locations=[],
                        metadata={"clipping_type": "SOFT_SATURATION", "thd_discriminated": True},
                    )
                # CLIPPING confirmed by THD analysis — compute severity from flat-tops
                audio_norm = audio / peak
                hard_clip_mask = np.abs(audio_norm) >= 0.999
                hard_clip_ratio = float(np.sum(hard_clip_mask)) / len(audio)
                threshold_factor = float(self.thresholds.get(DefectType.CLIPPING, 0.5))
                severity = min(1.0, (hard_clip_ratio * 10) / max(threshold_factor, 1e-6))
                clip_indices = np.where(hard_clip_mask)[0]
                locations: list[tuple[float, float]] = []
                if len(clip_indices) > 0:
                    groups = np.split(
                        clip_indices,
                        np.where(np.diff(clip_indices) > int(0.005 * self.sample_rate))[0] + 1,
                    )
                    for g in groups[:50]:
                        locations.append((float(g[0]) / self.sample_rate, float(g[-1]) / self.sample_rate))
                logger.debug(
                    "§6.3 _detect_clipping: CLIPPING erkannt (odd-harmonic profile) — severity=%.3f flat_tops=%.4f",
                    severity,
                    hard_clip_ratio,
                )
                return DefectScore(
                    defect_type=DefectType.CLIPPING,
                    severity=severity,
                    confidence=0.95,
                    locations=locations,
                    metadata={
                        "hard_clip_ratio": hard_clip_ratio,
                        "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
                        "thd_discriminated": True,
                        "clipping_type": "CLIPPING",
                    },
                )
            except Exception as _thd_exc:
                logger.debug("§6.3 THD-Diskriminierung fehlgeschlagen, Amplitude-Fallback: %s", _thd_exc)

        # Amplitude-only fallback (wenn clipping_detection nicht verfügbar)
        audio_norm = audio / peak
        hard_clip_mask = np.abs(audio_norm) >= 0.995
        hard_clip_ratio = float(np.sum(hard_clip_mask)) / len(audio)
        window_ms = int(0.001 * self.sample_rate)
        soft_clip_events = 0
        total_windows = 0
        for i in range(0, len(audio_norm) - window_ms, window_ms):
            window = audio_norm[i : i + window_ms]
            peak_w = float(np.max(np.abs(window)))
            if peak_w > 0.80:
                above_threshold = int(np.sum(np.abs(window) > 0.85 * peak_w))
                if above_threshold / len(window) > 0.25:
                    soft_clip_events += 1
            total_windows += 1
        soft_clip_ratio = soft_clip_events / max(total_windows, 1)
        threshold_factor = float(self.thresholds.get(DefectType.CLIPPING, 0.5))
        combined = hard_clip_ratio * 10 + soft_clip_ratio * 2
        severity = min(1.0, combined / max(threshold_factor, 1e-6))
        clip_indices = np.where(hard_clip_mask)[0]
        locations = []
        if len(clip_indices) > 0:
            groups = np.split(
                clip_indices,
                np.where(np.diff(clip_indices) > int(0.005 * self.sample_rate))[0] + 1,
            )
            for g in groups[:50]:
                locations.append((float(g[0]) / self.sample_rate, float(g[-1]) / self.sample_rate))
        return DefectScore(
            defect_type=DefectType.CLIPPING,
            severity=severity,
            confidence=0.92,
            locations=locations,
            metadata={
                "hard_clip_ratio": hard_clip_ratio,
                "soft_clip_ratio": soft_clip_ratio,
                "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
                "thd_discriminated": False,
            },
        )

    def _detect_dc_offset(self, audio: np.ndarray) -> DefectScore:
        """Erkennt DC-Offset (Gleichspannungsversatz am Nullpunkt).

        Methodik:
          - Misst den Mittelwert des Signals (=DC-Komponente)
          - Normiert auf den Spitzenpegel → relativer DC-Anteil
          - Severity: 0 = kein DC, 1.0 = starker DC-Versatz (≥ 10% des Spitzenpegels)
        """
        if len(audio) == 0:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.0)

        dc_value = float(np.mean(audio))
        peak = float(np.max(np.abs(audio)))

        if peak < 1e-6:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.5)

        dc_ratio = abs(dc_value) / peak  # 0.0 (kein DC) bis 1.0 (alles DC)

        # Threshold-skalierter Severity-Score: 10% DC relativ zum Peak = Severity 1.0
        threshold_factor = self.thresholds.get(DefectType.DC_OFFSET, 0.6)
        severity = min(1.0, dc_ratio / (0.05 * threshold_factor))

        dc_mv = dc_value  # In normalisierten Einheiten (kein echtes mV ohne Pegel-Referenz)

        return DefectScore(
            defect_type=DefectType.DC_OFFSET,
            severity=severity,
            confidence=0.98,  # Sehr zuverlässige Detektion
            locations=[],  # DC ist global, keine Zeitstempel
            metadata={
                "dc_value": dc_mv,
                "dc_ratio_percent": dc_ratio * 100,
                "dc_dbfs": 20 * np.log10(abs(dc_value)) if abs(dc_value) > 1e-10 else -120.0,
            },
        )

    def _detect_bandwidth_loss(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hochfrequenz-Verlust / Bandbreitenbegrenzung (HF-Rolloff).

        Typische Fälle:
          - Shellac: Stark begrenzt < 7 kHz
          - Kassette: Rolloff bei 12–14 kHz
          - LP-Vinyl: Obertöne über 16 kHz selten
          - MP3-Low: Hochpass-Abschnitt bei 10–14 kHz
          - ATRAC (MiniDisc): HF-Artefakte > 14 kHz

        Methodik:
          - Spektrum via Welch-Methode
          - Normierter HF-Energie-Anteil (Frequenz über 8 kHz) vs. Gesamtenergie
          - Vergleich mit mittlerem Breitband-Audio-Referenzwert (~8–12% über 8 kHz)
          - Severity: 0 = volle Bandbreite vorhanden, 1.0 = HF stark beschnitten
        """
        if len(audio) < 2048:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)

        if self.sample_rate < 16000:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)

        freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, len(audio) // 4))

        total_energy = np.sum(psd) + 1e-12
        # HF-Energie oberhalb von 8 kHz als Indikator für volle Bandbreite
        hf_cutoff = min(8000.0, self.sample_rate * 0.45)
        hf_energy = np.sum(psd[freqs >= hf_cutoff])
        hf_ratio = hf_energy / total_energy

        # Referenzwert: gesundes Breitband-Audio hat ca. 5–15% Energie über 8 kHz
        # Niedrige hf_ratio = hoher Bandbreitenverlust
        reference_hf_ratio = 0.08  # 8% Referenz
        loss_factor = max(0.0, 1.0 - (hf_ratio / reference_hf_ratio))

        # Material-Threshold skaliert die Empfindlichkeit
        threshold_factor = self.thresholds.get(DefectType.BANDWIDTH_LOSS, 0.5)
        # Hohe threshold_factor (z.B. 0.8 für CD) → weniger sensitiv → niedrigere Severity
        severity = min(1.0, loss_factor * (1.0 / max(threshold_factor, 0.1)))
        severity = min(1.0, severity)

        # Schätze effektive Bandbreite: letzte Frequenz mit > 1% der mittleren spektralen Dichte
        mean_psd_density = np.mean(psd)
        meaningful = np.where(psd > 0.01 * mean_psd_density)[0]
        effective_bw = float(freqs[meaningful[-1]]) if len(meaningful) > 0 else 0.0

        return DefectScore(
            defect_type=DefectType.BANDWIDTH_LOSS,
            severity=severity,
            confidence=0.85,
            locations=[],  # Bandbreitenverlust ist global
            metadata={
                "hf_ratio_percent": hf_ratio * 100,
                "effective_bandwidth_hz": effective_bw,
                "reference_hf_ratio_percent": reference_hf_ratio * 100,
            },
        )

    def _detect_pitch_drift(self, audio: np.ndarray) -> DefectScore:
        """Erkennt konstanten Pitch-Drift / Geschwindigkeitsfehler (≠ WOW/FLUTTER).

        Unterschied zu WOW/FLUTTER:
          - WOW/FLUTTER: *periodische* Tonhöhenschwankung (< 10 Hz Modulation)
          - PITCH_DRIFT: *monotoner / konstanter* Geschwindigkeitsfehler
            (z.B. Tape spielt 1–3% zu langsam oder zu schnell)

                Methodik:
                    - Teilt Audio in Langzeit-Segmente (z.B. 10 s)
                    - Schätzt die dominante Grundfrequenz via Autokorrelation
                    - Vergleicht Grundfrequenz zwischen frühem und spätem Segment
                    - Großer monotoner Drift → hohe Severity; periodisches → WOW/FLUTTER zuständig
        """
        min_len = int(5 * self.sample_rate)  # Mindestens 5 Sekunden für sinnvolle Analyse
        if len(audio) < min_len:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)

        # High-Pass filter für Grundfrequenz-Analyse (unter 50 Hz ignorieren)
        sos = signal.butter(4, 50, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        segment_len = int(min(10.0, len(audio) / (self.sample_rate * 3)) * self.sample_rate)
        segment_len = max(segment_len, int(2 * self.sample_rate))  # Mindestens 2s

        def estimate_fundamental(seg: np.ndarray) -> float:
            """Grundfrequenz-Schätzung über Autokorrelation (FFT-basiert, O(n log n))."""
            seg = seg / (np.max(np.abs(seg)) + 1e-8)
            # Suche im Bereich 80–2000 Hz
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 80)
            if max_lag >= len(seg):
                return 0.0
            # FFT-basierte Autokorrelation: O(n log n) statt O(n²) bei np.correlate
            # Zero-Padding auf 2n verhindert zirkuläre Aliasing-Artefakte
            n = len(seg)
            fft_seg = np.fft.rfft(seg, n=2 * n)
            corr_full = np.fft.irfft(fft_seg * np.conj(fft_seg))
            corr = np.real(corr_full[:n])  # Nur positive Lags (äquivalent zu mode='full'[n-1:])
            corr = corr[min_lag:max_lag]
            if len(corr) == 0:
                return 0.0
            peak_lag = np.argmax(corr) + min_lag
            if peak_lag == 0:
                return 0.0
            return float(self.sample_rate / peak_lag)

        # Früh- und spätsegment vergleichen
        early_seg = audio_hp[:segment_len]
        late_seg = audio_hp[-segment_len:]

        f_early = estimate_fundamental(early_seg)
        f_late = estimate_fundamental(late_seg)

        if f_early < 60 or f_late < 60:
            # Keine sinnvolle Grundfrequenz detektierbar (z.B. Perkussion oder Stille)
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.2)

        # Relativer Frequenzunterschied (Pitch-Drift in Cent)
        drift_ratio = abs(f_late - f_early) / f_early
        drift_cents = 1200 * np.log2(max(f_late, f_early) / min(f_late, f_early))

        # Severity: 10 Cent Drift = 0.1, 100 Cent (1 Halbton) = 1.0
        # WOW/FLUTTER ist für periodische Schwankungen verantwortlich;
        # hier nur monotoner Trend über lange Dauer.
        threshold_factor = self.thresholds.get(DefectType.PITCH_DRIFT, 0.6)
        severity = min(1.0, (drift_cents / 50.0) / threshold_factor)

        return DefectScore(
            defect_type=DefectType.PITCH_DRIFT,
            severity=severity,
            confidence=0.65,  # Schwieriger zu detektieren als DC-Offset
            locations=[],
            metadata={
                "f_early_hz": f_early,
                "f_late_hz": f_late,
                "drift_cents": drift_cents,
                "drift_ratio_percent": drift_ratio * 100,
            },
        )

    def _detect_reverb_excess(self, audio: np.ndarray) -> DefectScore:
        """Erkennt übermäßigen / unerwünschten Raumhall (Reverb Excess).

        Methodik (Schroeder-Integrationsverfahren, RT60-Schätzung):
          1. Signal in kurze Segmente aufteilen (2–4 s)
          2. Energie-Abklingkurve pro Segment via Backward-Integration
          3. RT60 aus dem Abfall von -5 dB auf -65 dB schätzen (Sabine/ISO 3382)
          4. Material-unabhängige Grenzwerte: RT60 > 0.8 s = problematisch,
             RT60 > 1.5 s = schwerer Defekt
          5. Zusätzlich: Tail/Direct-Ratio im Spektrum (Diffusfeld-Anteil)

        Severity: 0 = kein nennenswerter Hall, 1.0 = T60 > 1.5 s
        """
        if len(audio) < int(2 * self.sample_rate):
            return DefectScore(DefectType.REVERB_EXCESS, 0.0, 0.3)

        # Segmentlänge für RT60-Schätzung: 2–4 Sekunden
        seg_len = int(min(4.0, len(audio) / self.sample_rate * 0.5) * self.sample_rate)
        seg_len = max(seg_len, int(1.5 * self.sample_rate))
        if seg_len > len(audio):
            seg_len = len(audio)

        seg = audio[-seg_len:]  # Letztes Segment: enthält am meisten Nachhall

        # Schritte 1–3: Energie-Abklingkurve (Schroeder-Rückwärtsintegration)
        seg_sq = seg**2
        # Rückwärts kumulatives Integral
        decay_curve = np.cumsum(seg_sq[::-1])[::-1]
        decay_curve = decay_curve / (decay_curve[0] + 1e-12)  # Normieren auf 1.0
        decay_db = 10 * np.log10(decay_curve + 1e-12)

        # RT60 schätzen: Zeit von -5 dB bis -35 dB (× 2 = T60)
        # Verwende EDT (Early Decay Time) → robuster bei verhalletem Material
        try:
            idx_5db = np.argmax(decay_db <= -5.0)
            idx_35db = np.argmax(decay_db <= -35.0)
            if idx_5db == 0 or idx_35db == 0 or idx_35db <= idx_5db:
                rt60 = 0.0
            else:
                # T30 × 2 = T60 Approximation
                rt60 = 2.0 * (idx_35db - idx_5db) / self.sample_rate
        except Exception:
            rt60 = 0.0

        # Tail-to-Direct-Ratio: Energie im Schwanz vs. erster 100ms
        direct_energy = np.sum(audio[: int(0.1 * self.sample_rate)] ** 2) + 1e-12
        tail_start = int(0.3 * self.sample_rate)  # Nach 300ms = Nachhall
        tail_energy = (
            np.sum(audio[tail_start : tail_start + int(0.5 * self.sample_rate)] ** 2)
            if len(audio) > tail_start + int(0.5 * self.sample_rate)
            else 0.0
        )
        tdr = tail_energy / direct_energy  # Hoch = viel Nachhall

        # Severity aus RT60 + TDR kombinieren
        # RT60 > 1.5s = severity 1.0; alles unter 0.4s = 0.0
        rt60_severity = min(1.0, max(0.0, (rt60 - 0.4) / 1.1))
        tdr_severity = min(1.0, max(0.0, (tdr - 0.5) / 4.0))

        threshold_factor = self.thresholds.get(DefectType.REVERB_EXCESS, 0.6)
        combined = (rt60_severity * 0.65 + tdr_severity * 0.35) / max(threshold_factor, 0.1)
        severity = min(1.0, combined)

        return DefectScore(
            defect_type=DefectType.REVERB_EXCESS,
            severity=severity,
            confidence=0.72,  # RT60-Schätzung bei Musik (kein Impulssignal) nur mäßig präzise
            locations=[],
            metadata={
                "rt60_seconds": rt60,
                "tail_to_direct_ratio": tdr,
                "rt60_severity": rt60_severity,
                "tdr_severity": tdr_severity,
            },
        )

    def _detect_print_through(self, audio: np.ndarray) -> DefectScore:
        """Detect bidirectional magnetic print-through (pre-echo and post-echo) on tape.

        Print-Through arises from magnetic flux transfer between adjacent tape layers.
        It produces:
          - **Pre-echo  (alpha_pre)**: ghost signal 80–400 ms BEFORE a loud onset
            (winding/storage: lower layer magnetizes through to the current layer).
          - **Post-echo (alpha_post)**: ghost signal 80–400 ms AFTER a loud onset
            (playback head: current layer partially magnetized from the preceding layer
            during playback).

        Both directions are modelled as damped copies of the main signal:
          pre_echo  ≈ alpha_pre  × x[n + lag]   → audible as ghost before onset
          post_echo ≈ alpha_post × x[n - lag]   → audible as ghost after onset

        IEC 60094-3 / DIN 45 513: Pre-echo typically –20 to –35 dB relative to program.

        Note: Only applicable for TAPE/REEL_TAPE (threshold = 1.0 for all other materials).
        """
        min_len = int(0.5 * self.sample_rate)
        if len(audio) < min_len:
            return DefectScore(DefectType.PRINT_THROUGH, 0.0, 0.2)

        # Kurzzeit-RMS für Einsatz-Detektion
        win = int(0.010 * self.sample_rate)  # 10ms
        hop = win // 2
        n_frames = (len(audio) - win) // hop
        rms = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * hop
            rms[i] = np.sqrt(np.mean(audio[s : s + win] ** 2))

        rms_db = 20 * np.log10(rms + 1e-10)
        median_rms_db = np.median(rms_db)

        # Einsätze: Stellen mit > 20 dB über Median
        onset_frames = np.where(rms_db > median_rms_db + 20)[0]
        if len(onset_frames) == 0:
            return DefectScore(DefectType.PRINT_THROUGH, 0.0, 0.5)

        # Bidirektionale Print-Through-Detektion: Pre-Echo UND Post-Echo
        # search window: 80–400 ms in both directions (IEC 60094-3)
        echo_delays_ms = [80, 120, 160, 200, 250, 320, 400]
        print_through_events = 0
        total_magnitude = 0.0
        alpha_pre_list: list = []  # Gemessene Pre-Echo-Amplituden
        alpha_post_list: list = []  # Gemessene Post-Echo-Amplituden
        locations: list[tuple[float, float]] = []

        for onset_f in onset_frames[:30]:  # Max. erste 30 Einsätze prüfen
            onset_db = rms_db[onset_f]
            for delay_ms in echo_delays_ms:
                delay_f = int(delay_ms / (hop / self.sample_rate * 1000))

                # --- Pre-echo: Ghost BEFORE onset ---
                pre_f = onset_f - delay_f
                if pre_f >= 0:
                    pre_db = rms_db[pre_f]
                    db_below = onset_db - pre_db
                    if 18 <= db_below <= 48 and pre_db > median_rms_db + 8:
                        print_through_events += 1
                        alpha_pre = 10 ** (-(db_below) / 20)
                        alpha_pre_list.append(alpha_pre)
                        total_magnitude += alpha_pre
                        t_pre = float(pre_f * hop / self.sample_rate)
                        t_end = t_pre + win / self.sample_rate
                        if not any(abs(loc[0] - t_pre) < 0.020 for loc in locations):
                            locations.append((t_pre, float(t_end)))

                # --- Post-echo: Ghost AFTER onset (alpha_post) ---
                post_f = onset_f + delay_f
                if post_f < len(rms_db):
                    post_db = rms_db[post_f]
                    db_below = onset_db - post_db
                    if 18 <= db_below <= 48 and post_db > median_rms_db + 8:
                        print_through_events += 1
                        alpha_post = 10 ** (-(db_below) / 20)
                        alpha_post_list.append(alpha_post)
                        total_magnitude += alpha_post
                        t_post = float(post_f * hop / self.sample_rate)
                        t_end = t_post + win / self.sample_rate
                        if not any(abs(loc[0] - t_post) < 0.020 for loc in locations):
                            locations.append((t_post, float(t_end)))

        # Severity: > 5 Events = schweres Print-Through
        event_severity = min(1.0, print_through_events / 8.0)
        mag_severity = min(1.0, total_magnitude * 20)

        threshold_factor = self.thresholds.get(DefectType.PRINT_THROUGH, 0.6)
        severity = min(1.0, (event_severity * 0.6 + mag_severity * 0.4) / max(threshold_factor, 0.05))

        avg_alpha_pre = float(np.mean(alpha_pre_list)) if alpha_pre_list else 0.0
        avg_alpha_post = float(np.mean(alpha_post_list)) if alpha_post_list else 0.0

        return DefectScore(
            defect_type=DefectType.PRINT_THROUGH,
            severity=severity,
            confidence=0.60,  # Schwierig ohne Ground-Truth; Pre-Echo ≠ immer Print-Through
            locations=locations[:50],
            metadata={
                "pre_echo_events": len(alpha_pre_list),
                "post_echo_events": len(alpha_post_list),
                "avg_alpha_pre": avg_alpha_pre,
                "avg_alpha_post": avg_alpha_post,
                "avg_magnitude": total_magnitude / max(print_through_events, 1),
                "onsets_checked": min(len(onset_frames), 30),
            },
        )

    # ------------------------------------------------------------------
    # Detektoren — Runde 3: QUANTIZATION_NOISE, JITTER_ARTIFACTS, DYNAMIC_COMPRESSION_EXCESS
    # ------------------------------------------------------------------

    def _detect_quantization_noise(self, audio: np.ndarray) -> DefectScore:
        """
        Erkennt Quantisierungsrauschen durch Analyse der Amplitudenverteilung.

        Niedrige Bit-Tiefe (8-Bit, 12-Bit, aggressive Lossy-Kodierung) erzeugt:
        - Diskrete Amplitudenstufen → Histogramm-Clustering (wenige belegte Bins)
        - Granulationsrauschen in leisen Passagen (harmonisch unkorrelliertes Rauschen)
        - Effektive Bit-Tiefe < 14 Bit → Severity > 0

        Methode: Histogramm-Füllgrad + Granularitäts-Index leiser Passagen.
        """
        audio_norm = audio - np.mean(audio)
        max_amp = np.max(np.abs(audio_norm))
        if max_amp < 1e-8:
            return DefectScore(DefectType.QUANTIZATION_NOISE, 0.0, 0.0)
        audio_norm = audio_norm / max_amp

        # Methode 1: Histogramm-Füllgrad
        hist, _ = np.histogram(audio_norm, bins=512)
        n_populated = int(np.sum(hist > 0))
        # 16-bit → ~500 belegte Bins; 8-bit → ~100–200 belegte Bins
        fill_ratio = n_populated / 512.0

        # Methode 2: Granularität in leisen Passagen (< -40 dBFS)
        quiet_mask = np.abs(audio_norm) < 0.01
        if quiet_mask.sum() > 256:
            quiet_audio = audio_norm[quiet_mask]
            diff = np.diff(quiet_audio)
            # Viele ±LSB-Sprünge = charakteristisch für Quantisierungsrauschen
            granularity = float(np.sum(np.abs(diff) > 0.003) / len(diff))
        else:
            granularity = 0.0

        # Severity: fill_ratio < 0.5 → beginnt zu skalieren
        severity_fill = max(0.0, 0.5 - fill_ratio) * 2.0
        severity = min(1.0, severity_fill * 0.6 + granularity * 0.4)

        threshold = self.thresholds.get(DefectType.QUANTIZATION_NOISE, 0.6)
        if severity < threshold * 0.5:
            severity = 0.0

        return DefectScore(
            defect_type=DefectType.QUANTIZATION_NOISE,
            severity=severity,
            confidence=0.65,
            locations=[],
            metadata={
                "fill_ratio": fill_ratio,
                "n_populated_bins": n_populated,
                "granularity_index": granularity,
            },
        )

    def _detect_jitter_artifacts(self, audio: np.ndarray) -> DefectScore:
        """
        Erkennt Jitter-Artefakte durch Analyse der Hochfrequenz-Varianz.

        D/A-Wandler-Jitter (Zeitgitter-Fehler) erzeugt:
        - Frequenzmodulations-Seitenbänder um reine Sinustöne (FM-Seitenbänder)
        - Zeitlich inkonsistente Energie bei hohen Frequenzen (> 8 kHz)
        - Erhöhtes Spitzen-zu-Rauschen-Verhältnis im HF-Band

        Methode: Segmentweise HF-Spektrum-Varianz + Seitenbandindikator.
        """
        n = len(audio)
        if n < 2048:
            return DefectScore(DefectType.JITTER_ARTIFACTS, 0.0, 0.0)

        seg_len = min(8192, n // 4)
        n_segs = min(8, n // seg_len)
        if n_segs < 2:
            return DefectScore(DefectType.JITTER_ARTIFACTS, 0.0, 0.0)

        win = np.hanning(seg_len)
        spectra = []
        for i in range(n_segs):
            s = i * seg_len
            spectrum = np.abs(np.fft.rfft(audio[s : s + seg_len] * win))
            spectra.append(spectrum)

        mean_spectrum = np.mean(spectra, axis=0)
        freqs = np.fft.rfftfreq(seg_len, 1.0 / self.sample_rate)

        # Hochfrequenzbereich > 8 kHz analysieren
        hf_mask = freqs > 8000
        if hf_mask.sum() < 4:
            return DefectScore(DefectType.JITTER_ARTIFACTS, 0.0, 0.0)

        hf_spectra = np.array([s[hf_mask] for s in spectra])

        # Zeitliche Inkonsistenz der HF-Energie (Jitter → schwankende HF)
        hf_mean_per_band = np.mean(hf_spectra, axis=0) + 1e-10
        hf_var_ratio = float(np.mean(np.std(hf_spectra, axis=0) / hf_mean_per_band))

        # Seitenband-Indikator: Peak-zu-Median im HF-Band
        hf_mean = mean_spectrum[hf_mask]
        sorted_hf = np.sort(hf_mean)[::-1]
        top_n = max(1, len(sorted_hf) // 100)
        sideband_ratio = float(np.mean(sorted_hf[:top_n]) / (np.median(hf_mean) + 1e-10))

        severity_var = min(1.0, max(0.0, (hf_var_ratio - 0.2) / 0.6))
        severity_sb = min(1.0, max(0.0, (sideband_ratio - 3.0) / 10.0))
        severity = severity_var * 0.55 + severity_sb * 0.45

        threshold = self.thresholds.get(DefectType.JITTER_ARTIFACTS, 0.6)
        if severity < threshold * 0.5:
            severity = 0.0

        return DefectScore(
            defect_type=DefectType.JITTER_ARTIFACTS,
            severity=severity,
            confidence=0.55,  # Jitter schwer von anderen HF-Phänomenen zu trennen
            locations=[],
            metadata={
                "hf_var_ratio": hf_var_ratio,
                "sideband_ratio": sideband_ratio,
                "n_segments": n_segs,
            },
        )

    def _detect_dynamic_compression_excess(self, audio: np.ndarray) -> DefectScore:
        """
        Erkennt übermäßige Dynamikkompression ('Loudness War', DR-Wert < 6 dB).

        Charakteristika stark komprimierter Aufnahmen:
        - Niedriger Crest Factor: Peak/RMS < 6 dB (ideal: 12–20 dB für Musik)
        - Histogramm-Clustering nahe ±1.0 (viele Samples nahe Maximum)
        - Geringe LRA (Loudness Range, EBU R128): LRA < 3 LU = exzessiv

        Referenz: DR-Database (Dynamic Range Database), EBU R128, AES17.
        """
        n = len(audio)
        if n < 512:
            return DefectScore(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.0, 0.0)

        # Schritt 1: Crest Factor
        max_amp = np.max(np.abs(audio))
        if max_amp < 1e-8:
            return DefectScore(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.0, 0.0)
        audio_norm = audio / max_amp
        rms = float(np.sqrt(np.mean(audio_norm**2)))
        crest_db = -20.0 * np.log10(rms + 1e-10)  # dB über RMS → höher = mehr Dynamik

        # Schritt 2: Histogramm-Analyse
        hist, _ = np.histogram(np.abs(audio_norm), bins=100, range=(0.0, 1.0))
        # Samples nahe Maximum (> -1 dBFS ≈ 0.891 linear) → typisch für Loudness War
        high_amp_ratio = float(hist[-10:].sum() / (n + 1))

        # Schritt 3: Vereinfachte LRA-Schätzung über 400ms-Fenster-RMS
        win_s = max(1, int(0.4 * self.sample_rate))
        hop_s = max(1, win_s // 2)
        n_wins = max(1, (n - win_s) // hop_s)
        rms_values = []
        for i in range(n_wins):
            s = i * hop_s
            w = audio_norm[s : s + win_s]
            r = float(np.sqrt(np.mean(w**2)))
            if r > 1e-6:
                rms_values.append(20.0 * np.log10(r))

        if len(rms_values) > 4:
            rms_arr = np.array(rms_values)
            lra = float(np.percentile(rms_arr, 95) - np.percentile(rms_arr, 10))
        else:
            lra = 20.0  # Konservativer Default: keine Aussage möglich

        # Severity-Berechnung
        # Crest < 6 dB → severity=1.0; > 14 dB → severity=0.0
        severity_crest = max(0.0, min(1.0, (12.0 - crest_db) / 6.0))
        # high_amp_ratio > 30 % → stark überkomprimiert
        severity_hist = max(0.0, min(1.0, (high_amp_ratio - 0.05) / 0.25))
        # LRA < 3 LU → severity=1.0; > 12 LU → severity=0.0
        severity_lra = max(0.0, min(1.0, (8.0 - lra) / 5.0))

        severity = severity_crest * 0.35 + severity_hist * 0.35 + severity_lra * 0.30

        threshold = self.thresholds.get(DefectType.DYNAMIC_COMPRESSION_EXCESS, 0.6)
        if severity < threshold * 0.5:
            severity = 0.0

        return DefectScore(
            defect_type=DefectType.DYNAMIC_COMPRESSION_EXCESS,
            severity=severity,
            confidence=0.72,  # Crest Factor + LRA sind gut kalibrierte Metriken
            locations=[],
            metadata={
                "crest_factor_db": crest_db,
                "high_amplitude_ratio": high_amp_ratio,
                "lra_db": lra,
                "rms_db": 20.0 * np.log10(rms + 1e-10),
            },
        )

    # ------------------------------------------------------------------
    # Detektoren für Typen 21-28 (F-7 Ergänzung)
    # ------------------------------------------------------------------

    def _detect_soft_saturation(self, audio: np.ndarray) -> DefectScore:
        """Detect SOFT_SATURATION: tube/tape even-harmonic distortion — preserve, do not repair.

        Per Spec §6.3: flat_tops < 0.1 % AND even harmonics (H2, H4) dominate odd (H3, H5).
        Returns severity > 0 only when soft-saturation profile is clearly present AND
        hard clipping is absent — ensuring phase_23 (clipping repair) is NOT triggered.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

        # Flat-top clipping check — if hard clipping present, this is CLIPPING, not SOFT_SATURATION
        flat_top_threshold = 0.99
        flat_tops = float(np.sum(np.abs(audio) >= flat_top_threshold)) / max(n, 1)
        if flat_tops >= 0.001:  # ≥ 0.1 % flat-tops → hard clipping, not soft saturation
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.80)

        try:
            max_amp = float(np.max(np.abs(audio)))
            if max_amp < 0.05:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.5)
            audio_norm = audio / max_amp

            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(audio_norm[:n_fft])) ** 2
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

            # Find fundamental (80–1000 Hz highest-energy bin)
            fund_mask = (freqs >= 80.0) & (freqs <= 1000.0)
            if not fund_mask.any():
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)
            fund_idx = int(np.argmax(spec * fund_mask.astype(float)))
            fund_hz = float(freqs[fund_idx])
            if fund_hz <= 0:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

            # Harmonic energy helper (±2 Hz window per harmonic)
            bin_w = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            half_bin = max(1, int(2.0 / (bin_w + 1e-10)))

            def _harm(h: int) -> float:
                target_hz = fund_hz * h
                if target_hz >= self.sample_rate / 2.0:
                    return 0.0
                tidx = round(target_hz / (self.sample_rate / n_fft))
                lo = max(0, tidx - half_bin)
                hi = min(len(spec), tidx + half_bin + 1)
                return float(np.sum(spec[lo:hi]))

            even_energy = _harm(2) + _harm(4)
            odd_energy = _harm(3) + _harm(5)
            total_harm = even_energy + odd_energy
            if total_harm < 1e-20:
                return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

            even_ratio = even_energy / total_harm
            # > 0.55 = even-dominant (tube/tape character); 0.5 = neutral
            severity = float(np.clip((even_ratio - 0.55) / 0.35, 0.0, 1.0))
            confidence = 0.65 if severity > 0.1 else 0.40

            return DefectScore(
                defect_type=DefectType.SOFT_SATURATION,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={"even_ratio": even_ratio, "flat_top_ratio": flat_tops},
            )
        except Exception:
            return DefectScore(DefectType.SOFT_SATURATION, 0.0, 0.3)

    def _detect_sibilance(self, audio: np.ndarray) -> DefectScore:
        """Detect SIBILANCE: excessive energy in 4–12 kHz zone (harsh 's'/'sh' consonants).

        Measures the fraction of total spectral power in the sibilance range.
        High ratio (> 0.40) indicates problematic sibilance requiring de-essing.
        Windowed analysis provides time-locations of sibilant events.
        """
        n = len(audio)
        if n < 1024:
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

        try:
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(2048, n))
            total_power = float(np.sum(psd) + 1e-20)
            sib_mask = (freqs >= 4000.0) & (freqs <= 12000.0)
            sib_frac = float(np.sum(psd[sib_mask]) / total_power)

            # Expected sibilance zone fraction for natural music: ~0.08–0.15
            # Problematic: > 0.25 (de-essing needed)
            severity = float(np.clip((sib_frac - 0.12) / 0.20, 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.SIBILANCE, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            # Windowed sibilance location detection (50 ms windows, 25 ms hop)
            locations: list[tuple[float, float]] = []
            if severity > 0.0:
                win_sib = max(1, int(0.050 * self.sample_rate))
                hop_sib = max(1, win_sib // 2)
                sib_threshold = 0.20  # fraction above which a window is sibilant
                in_event = False
                ev_start = 0.0
                for i in range(0, n - win_sib, hop_sib):
                    chunk = audio[i : i + win_sib]
                    cf = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
                    cs = np.abs(np.fft.rfft(chunk)) ** 2
                    c_total = float(np.sum(cs) + 1e-20)
                    c_sib = float(np.sum(cs[(cf >= 4000.0) & (cf <= 12000.0)]))
                    frac = c_sib / c_total
                    t_s = i / self.sample_rate
                    if frac >= sib_threshold:
                        if not in_event:
                            ev_start = t_s
                            in_event = True
                    else:
                        if in_event:
                            locations.append((ev_start, t_s + win_sib / self.sample_rate))
                            in_event = False
                if in_event:
                    locations.append((ev_start, n / self.sample_rate))
                locations = self._sample_locations_evenly(locations, 50)

            return DefectScore(
                defect_type=DefectType.SIBILANCE,
                severity=severity,
                confidence=0.68,
                locations=locations,
                metadata={"sibilance_power_fraction": sib_frac},
            )
        except Exception:
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

    def _detect_bias_error(self, audio: np.ndarray) -> DefectScore:
        """Detect BIAS_ERROR: wrong AC-bias level on magnetic tape recording.

        Over-bias → HF rolloff earlier than expected (upper-mid/mid ratio < 0.2).
        Under-bias → elevated HF noise floor (upper-mid/mid ratio > 1.2).
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

        try:
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
            mid_e = float(np.sum(psd[(freqs >= 2000.0) & (freqs < 4000.0)]) + 1e-20)
            um_e = float(np.sum(psd[(freqs >= 6000.0) & (freqs <= 12000.0)]) + 1e-20)
            ratio = um_e / mid_e

            over_bias_sev = float(np.clip((0.3 - ratio) / 0.25, 0.0, 1.0))  # HF cut
            under_bias_sev = float(np.clip((ratio - 1.0) / 0.5, 0.0, 1.0))  # HF elevated
            severity = float(np.clip(max(over_bias_sev, under_bias_sev), 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.BIAS_ERROR, 0.5)
            if severity < threshold * 0.20:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.BIAS_ERROR,
                severity=severity,
                confidence=0.60,
                locations=[],
                metadata={"mid_uppermid_ratio": ratio, "over_bias": over_bias_sev, "under_bias": under_bias_sev},
            )
        except Exception:
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

    def _detect_riaa_curve_error(self, audio: np.ndarray) -> DefectScore:
        """Detect RIAA_CURVE_ERROR: wrong EQ curve applied during vinyl/disc digitization.

        RIAA not applied → massive bass excess (bass/mid ratio >> 5).
        Double-RIAA or wrong curve applied → extreme bass cut (ratio << 0.1).
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.RIAA_CURVE_ERROR, 0.0, 0.3)

        try:
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
            bass_e = float(np.sum(psd[freqs < 300.0]) + 1e-20)
            mid_e = float(np.sum(psd[(freqs >= 1000.0) & (freqs < 4000.0)]) + 1e-20)
            ratio = bass_e / mid_e

            riaa_missing = float(np.clip((ratio - 5.0) / 10.0, 0.0, 1.0))
            riaa_double = float(np.clip((0.1 - ratio) / 0.10, 0.0, 1.0))
            severity = float(np.clip(max(riaa_missing, riaa_double), 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.RIAA_CURVE_ERROR, 0.5)
            if severity < threshold * 0.3:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.RIAA_CURVE_ERROR,
                severity=severity,
                confidence=0.58,
                locations=[],
                metadata={"bass_mid_ratio": ratio, "riaa_missing": riaa_missing, "riaa_double": riaa_double},
            )
        except Exception:
            return DefectScore(DefectType.RIAA_CURVE_ERROR, 0.0, 0.3)

    def _detect_head_wear(self, audio: np.ndarray) -> DefectScore:
        """Detect HEAD_WEAR: magnetic head degradation causing progressive HF rolloff.

        Worn head → energy above 8 kHz severely reduced relative to 2–4 kHz reference.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

        try:
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
            ref_e = float(np.sum(psd[(freqs >= 2000.0) & (freqs < 4000.0)]) + 1e-20)
            hf_e = float(np.sum(psd[freqs >= 8000.0]))
            hf_ratio = hf_e / ref_e

            # Fresh head: hf_ratio ~ 0.3–1.0 for normal music
            # Moderately worn: hf_ratio 0.10–0.20; severely worn: < 0.05
            severity = float(np.clip((0.20 - hf_ratio) / 0.08, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.HEAD_WEAR, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.HEAD_WEAR,
                severity=severity,
                confidence=0.58,
                locations=[],
                metadata={"hf_to_reference_ratio": hf_ratio},
            )
        except Exception:
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

    def _detect_transient_smearing(self, audio: np.ndarray) -> DefectScore:
        """Detect TRANSIENT_SMEARING: broadened attack rise-times from limiting/saturation.

        Measures 10%–90% rise-time at up to 20 onset events.
        Smeared: mean rise-time > 15 ms (normal music: 1–10 ms).
        """
        n = len(audio)
        if n < self.sample_rate // 2:
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

        try:
            # High-pass envelope to find transients
            hp_sos = signal.butter(
                2, float(np.clip(200.0 / (self.sample_rate / 2.0), 1e-6, 0.999)), btype="high", output="sos"
            )
            hp_audio = signal.sosfilt(hp_sos, audio)
            win = max(1, int(0.005 * self.sample_rate))
            envelope = np.convolve(np.abs(hp_audio), np.ones(win) / win, mode="same")
            envelope = np.nan_to_num(envelope, nan=0.0)

            threshold_level = float(np.percentile(envelope, 90))
            if threshold_level < 1e-8:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # Onset candidates: rising edges above 50% of threshold
            diff_env = np.diff(envelope)
            onset_idxs = np.where((envelope[1:] > threshold_level * 0.5) & (diff_env > 0))[0]
            if len(onset_idxs) == 0:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            rise_times_ms: list[float] = []
            smear_locations: list[tuple[float, float]] = []
            for idx in onset_idxs[:20]:
                peak_val = float(envelope[idx])
                level_10 = peak_val * 0.10
                level_90 = peak_val * 0.90
                window_back = max(0, idx - int(0.030 * self.sample_rate))
                pre_env = envelope[window_back:idx]
                idx_10 = idx_90 = None
                for k, v in enumerate(pre_env):
                    if v >= level_10 and idx_10 is None:
                        idx_10 = window_back + k
                    if v >= level_90 and idx_90 is None:
                        idx_90 = window_back + k
                        break
                if idx_10 is not None and idx_90 is not None and idx_90 > idx_10:
                    rt = (idx_90 - idx_10) / self.sample_rate * 1000.0
                    if 0.5 < rt < 100.0:
                        rise_times_ms.append(rt)
                        if rt > 8.0:  # only mark actually smeared onsets
                            t_s = window_back / self.sample_rate
                            t_e = (idx + int(0.010 * self.sample_rate)) / self.sample_rate
                            smear_locations.append((t_s, t_e))

            if not rise_times_ms:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            mean_rt = float(np.mean(rise_times_ms))
            severity = float(np.clip((mean_rt - 8.0) / 20.0, 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.TRANSIENT_SMEARING, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.TRANSIENT_SMEARING,
                severity=severity,
                confidence=0.60,
                locations=smear_locations if severity > 0.0 else [],
                metadata={"mean_rise_time_ms": mean_rt, "n_onsets": len(rise_times_ms)},
            )
        except Exception:
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

    def _detect_pre_echo(self, audio: np.ndarray) -> DefectScore:
        """Detect PRE_ECHO: energy preceding a major transient.

        Codec temporal masking → pre-echo 5–35 ms before transient.
        Tape print-through → ghost signal 100–600 ms before main signal.
        Measures ratio of pre-onset energy to steady-state baseline.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

        try:
            win = max(1, int(0.010 * self.sample_rate))
            envelope = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
            envelope = np.nan_to_num(envelope)

            peak_env = float(np.max(envelope))
            if peak_env < 1e-8:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            transient_thresh = float(np.percentile(envelope, 80))
            diff_env = np.diff(envelope)
            transient_idxs = np.where((envelope[1:] > transient_thresh) & (diff_env > transient_thresh * 0.1))[0]

            if len(transient_idxs) == 0:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            pre_window_s = int(0.030 * self.sample_rate)
            baseline_window_s = int(0.100 * self.sample_rate)
            pre_echo_ratios: list[float] = []
            pre_echo_locations: list[tuple[float, float]] = []

            for idx in transient_idxs[:15]:
                pre_start = max(0, idx - pre_window_s - int(0.005 * self.sample_rate))
                pre_end = max(0, idx - int(0.005 * self.sample_rate))
                if pre_end <= pre_start:
                    continue
                pre_energy = float(np.mean(envelope[pre_start:pre_end] ** 2))

                baseline_start = max(0, idx - pre_window_s - baseline_window_s)
                baseline_end = max(0, idx - pre_window_s)
                if baseline_end <= baseline_start:
                    continue
                baseline_energy = float(np.mean(envelope[baseline_start:baseline_end] ** 2)) + 1e-20

                ratio = pre_energy / baseline_energy
                if ratio > 1.0:
                    pre_echo_ratios.append(ratio)
                    t_start = pre_start / self.sample_rate
                    t_end = idx / self.sample_rate
                    pre_echo_locations.append((t_start, t_end))

            if not pre_echo_ratios:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            mean_ratio = float(np.mean(pre_echo_ratios))
            severity = float(np.clip((mean_ratio - 1.5) / 4.5, 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.PRE_ECHO, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.PRE_ECHO,
                severity=severity,
                confidence=0.62,
                locations=pre_echo_locations if severity > 0.0 else [],
                metadata={"pre_echo_mean_ratio": mean_ratio, "n_transients": len(transient_idxs[:15])},
            )
        except Exception:
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

    def _detect_aliasing(self, audio: np.ndarray) -> DefectScore:
        """Detect ALIASING: spurious near-Nyquist energy without natural musical source.

        Anti-aliasing filter failure during digitization → elevated spectral floor
        in the 85–97 % Nyquist region that exceeds the expected HF rolloff.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.ALIASING, 0.0, 0.3)

        try:
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
            nyquist = self.sample_rate / 2.0

            near_nyq_mask = (freqs >= nyquist * 0.85) & (freqs <= nyquist * 0.97)
            mid_hf_mask = (freqs >= 10000.0) & (freqs < nyquist * 0.85)

            near_e = float(np.sum(psd[near_nyq_mask]) + 1e-20)
            mid_e = float(np.sum(psd[mid_hf_mask]) + 1e-20)

            near_nyq_ratio = near_e / mid_e
            # Normal: HF monotonically decreases → near-Nyquist < 0.5× mid
            # Aliasing: plateau or rise near Nyquist
            severity = float(np.clip((near_nyq_ratio - 0.6) / 0.8, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.ALIASING, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            return DefectScore(
                defect_type=DefectType.ALIASING,
                severity=severity,
                confidence=0.55,
                locations=[],
                metadata={"near_nyquist_ratio": near_nyq_ratio, "nyquist_hz": nyquist},
            )
        except Exception:
            return DefectScore(DefectType.ALIASING, 0.0, 0.3)


# ========== CLI/Testing Interface ==========

if __name__ == "__main__":
    """Test DefectScanner mit Beispiel-Audio."""

    # Generiere Test-Audio mit künstlichen Defekten
    duration = 10  # Sekunden
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))

    # Basis-Signal: 440 Hz Sinus
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Füge Defekte hinzu
    # 1. Clicks (5 pro Sekunde)
    for i in range(50):
        pos = int(np.random.rand() * len(audio))
        audio[pos : pos + 10] += 0.3 * np.random.randn(10)

    # 2. 60Hz Hum
    audio += 0.05 * np.sin(2 * np.pi * 60 * t)

    # 3. High-freq Noise (Tape Hiss)
    audio += 0.02 * np.random.randn(len(audio))

    # 4. Rumble (20 Hz)
    audio += 0.08 * np.sin(2 * np.pi * 20 * t)

    # Test Scanner
    scanner = DefectScanner(sample_rate=sr)
    result = scanner.scan(audio, material_type=MaterialType.VINYL)

    logger.debug(f"\n{'=' * 60}")
    logger.debug("DEFECT SCAN RESULTS")
    logger.debug(f"{'=' * 60}")
    logger.debug(f"Material: {result.material_type.value}")
    logger.debug(f"Duration: {result.duration_seconds:.1f}s")
    logger.debug(
        f"Analysis Time: {result.analysis_time_seconds:.3f}s ({result.analysis_time_seconds / result.duration_seconds * 100:.1f}% overhead)"
    )
    logger.debug("\nTop 5 Defects:")
    for i, score in enumerate(result.get_top_defects(5), 1):
        logger.debug(f"  {i}. {score}")

    logger.debug(f"\nTotal Severity: {result.get_total_severity():.3f}")
