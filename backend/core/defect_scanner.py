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

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.fft as fft
import scipy.signal as signal

# §6.3 CLIPPING vs SOFT_SATURATION discrimination via THD analysis (lazy import)
try:
    from backend.core.clipping_detection import classify_clipping as _classify_clipping, ClippingType as _ClippingType
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


def _audio_scan_cache_key(audio: np.ndarray, sr: int, material: Optional[object]) -> str:
    """Deterministischer Cache-Key für DefectScanner.scan()."""
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    if material is not None:
        h.update(str(material).encode())
    return f"scan:{h.hexdigest()[:16]}"


class DefectType(Enum):
    """30 Defekttypen für weltklasse Audio-Restauration.

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
    WOW = "wow"     # Tonhöhenschwankung < 0.5 Hz (IEC 60386 — Motorexzentrizität, Plattenteller-Gleichlaufschwankung)
    WOW_FLUTTER = "wow"  # Backward-compatible alias: legacy combined defect now maps to WOW/Flutter phase path
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
    QUADROPHONY = "quadrophony"  # 4-Kanal-Quadrofonie (1970–1978), SQ/QS/CD-4-Matrix
    AMBISONIC = "ambisonic"  # Ambisonic B-Format (W/X/Y/Z), ITU-R BS.2076
    UNKNOWN = "unknown"


@dataclass
class DefectScore:
    """Score-Objekt für einen einzelnen Defekt."""

    defect_type: DefectType
    severity: float  # 0.0 - 1.0 (0 = keine Defekte, 1 = schwere Defekte)
    confidence: float  # 0.0 - 1.0 (Konfidenz der Detection)
    locations: List[Tuple[float, float]] = field(default_factory=list)  # (start_time, end_time) in Sekunden
    metadata: Dict = field(default_factory=dict)  # Zusätzliche Informationen

    def __repr__(self) -> str:
        return f"DefectScore({self.defect_type.value}: {self.severity:.3f}, conf={self.confidence:.2f}, {len(self.locations)} events)"


@dataclass
class DefectAnalysisResult:
    """Vollständiges Ergebnis der Defekt-Analyse."""

    material_type: MaterialType
    scores: Dict[DefectType, DefectScore]
    analysis_time_seconds: float
    sample_rate: int
    duration_seconds: float
    # Forensische Tonträgerkettenerkennung (befüllt von DefectScanner.scan())
    transfer_chain_raw: Dict = field(default_factory=dict)  # MediumDetector.detect()-Ausgabe
    is_multi_generation: bool = False  # Mehrstufige Überspielungskette
    transfer_chain_str: str = ""  # Lesbare Kette, z.B. "cassette → mp3"
    # §6.6.1 Pflicht-Spektralfingerabdruck — 5 normierte Messgrößen (immer befüllt)
    spectral_fingerprint: Dict[str, float] = field(default_factory=dict)
    # spectral_fingerprint enthält:
    #   rolloff_95_hz       — Rolloff-Frequenz 95% der Spektralenergie [Hz]
    #   wow_flutter_index   — Pitch-Varianz-Index [0..∞, >1.5 = Kassette]
    #   hf_energy_above_16k — Anteil Energie > 16 kHz an Gesamtenergie [0..1]
    #   noise_floor_p5_db   — 5. Perzentil PSD als Rauschboden [dBFS]
    #   effective_bandwidth_hz — HF-Rolloff bei −60 dBFS [Hz]
    #   material_detected   — auto-erkanntes Material (auch wenn Hint übergeben)

    def get_top_defects(self, n: int = 5) -> List[DefectScore]:
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
    - Erkennt 20 Defekttypen parallel (11 Kern + 9 Weltklasse-Erweiterung)
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
        },
        MaterialType.UNKNOWN: {
            # Konservative Defaults (hohe Thresholds = weniger falsch-positive)
            dt: 0.6
            for dt in DefectType
        },
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
        },
        MaterialType.QUADROPHONY: {
            # 4-Kanal-Quadrofonie (1970–1978): CD-4, SQ/QS-Matrix — Kanalübersprechen & Phasenfehler
            DefectType.CLICKS: 0.5,
            DefectType.CRACKLE: 0.4,  # Vinyl-Trägermedium häufig
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.30,  # Vinyl-Träger: Plattenteller-Gleichlauf < 0.5 Hz
            DefectType.FLUTTER: 0.35,  # Abtastarm-Resonanz 0.5–200 Hz im Quad-Betrieb
            DefectType.STEREO_IMBALANCE: 0.3,  # Kanalungleichgewicht in 4-Kanal-Matrix
            DefectType.DIGITAL_ARTIFACTS: 0.8,  # Meist analoger Träger
            DefectType.LOW_FREQ_RUMBLE: 0.4,
            DefectType.HIGH_FREQ_NOISE: 0.4,
            DefectType.COMPRESSION_ARTIFACTS: 0.5,  # CD-4 FM-Träger-Verzerrung möglich
            DefectType.PHASE_ISSUES: 0.2,  # Matrix-Dekodierung → Phasenfehler typisch
            DefectType.DROPOUTS: 0.4,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.4,
            DefectType.PITCH_DRIFT: 0.4,
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 0.6,
            DefectType.QUANTIZATION_NOISE: 0.8,  # Analoger Träger
            DefectType.JITTER_ARTIFACTS: 0.7,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.5,
            DefectType.SOFT_SATURATION: 0.4,
            DefectType.HEAD_WEAR: 0.6,
            DefectType.PRE_ECHO: 0.3,  # Quad-Disc: ATRAC- oder MP3-Downkonvertierung → Pre-Echo möglich
            DefectType.TRANSIENT_SMEARING: 0.4,
            DefectType.RIAA_CURVE_ERROR: 0.4,  # Quad-Disc: SQ/QS-Matrix + EQ-Kurven-Fehler möglich
            DefectType.ALIASING: 0.4,  # Matrix-Dekodierung und Downmix erzeugen Aliasing-Artefakte
            DefectType.BIAS_ERROR: 0.3,  # 4-Kanal-Spulenband: Bias-Fehler bei gemischten Bandsorten häufig
            DefectType.AZIMUTH_ERROR: 0.30,  # Quad-Spulenband (ggf.): Azimuth-Fehler zwischen 2- und 4-Spur-Köpfen
            DefectType.SIBILANCE: 0.5,  # Vinyl/Bandträger: Tonabnehmer-Sibilanz + Matrizierungsartefakte (CD-4/SQ) möglich
        },
        MaterialType.AMBISONIC: {
            # Ambisonic B-Format (W/X/Y/Z): Kugelmikrofon-Mehrkanal, ITU-R BS.2076
            DefectType.CLICKS: 0.6,
            DefectType.CRACKLE: 0.7,
            DefectType.HUM: 0.5,
            DefectType.WOW: 0.70,  # Ambisonic: WOW vom Aufzeichnungsmedium abhängig
            DefectType.FLUTTER: 0.70,  # Ambisonic: FLUTTER vom Aufzeichnungsmedium abhängig
            DefectType.STEREO_IMBALANCE: 0.5,  # Multi-Kanal: kein klassisches L/R-Ungleichgewicht
            DefectType.DIGITAL_ARTIFACTS: 0.5,  # Digitale Aufzeichnung üblich
            DefectType.LOW_FREQ_RUMBLE: 0.5,
            DefectType.HIGH_FREQ_NOISE: 0.5,
            DefectType.COMPRESSION_ARTIFACTS: 0.5,
            DefectType.PHASE_ISSUES: 0.2,  # B-Format Phasenkohärenz W/X/Y/Z kritisch
            DefectType.DROPOUTS: 0.5,
            DefectType.CLIPPING: 0.5,
            DefectType.DC_OFFSET: 0.5,
            DefectType.BANDWIDTH_LOSS: 0.5,
            DefectType.PITCH_DRIFT: 0.6,
            DefectType.REVERB_EXCESS: 0.5,
            DefectType.PRINT_THROUGH: 0.8,
            DefectType.QUANTIZATION_NOISE: 0.5,
            DefectType.JITTER_ARTIFACTS: 0.5,
            DefectType.DYNAMIC_COMPRESSION_EXCESS: 0.5,
            DefectType.SOFT_SATURATION: 0.6,
            DefectType.HEAD_WEAR: 0.7,
            DefectType.PRE_ECHO: 0.2,  # N/A: Ambisonics digital — Pre-Echo nur aus Quell-Codec möglich
            DefectType.TRANSIENT_SMEARING: 0.4,  # Phasenfehler können Transienten verbreitern
            DefectType.RIAA_CURVE_ERROR: 1.0,  # N/A: Ambisonics modernes Format — keine Disc-Entzerrungskurve
            DefectType.ALIASING: 0.2,  # Modernes Format — professionelle Handhabung, AA meist korrekt
            DefectType.BIAS_ERROR: 1.0,  # N/A: Ambisonics digital — kein analoger Bias
            DefectType.AZIMUTH_ERROR: 0.50,  # Ambisonic: Magnetbandaufnahme möglich — Azimuth-Fehler wirkt B-Format-Inkohärenz
            DefectType.SIBILANCE: 0.3,  # Ambisonic: Hochqualitatives Kugelmikrofon — Sibilanz selten, B-Format-Summierung kann es verstärken
        },
    }

    def __init__(self, sample_rate: int = 44100, material_type: Optional[MaterialType] = None):
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
        sample_rate: Optional[int] = None,
        material_type: Optional[MaterialType] = None,
        progress_callback: Optional["Callable[[int], None]"] = None,
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
        _sf: Dict[str, float] = {}
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
        _material_hint = material_type  # Hint des Aufrufers (kann None sein)  # noqa: F841
        _auto_material = self._auto_detect_material(audio)  # §6.6.1: immer berechnen
        _sf["material_detected"] = float(list(MaterialType).index(_auto_material))
        if material_type is None:
            material_type = self.material_type or _auto_material

        self.thresholds = self.MATERIAL_SENSITIVITY[material_type]

        # Audio normalisieren für konsistente Detection
        if audio.ndim == 2:
            is_stereo = True
            audio_mono = np.mean(audio, axis=1)
        else:
            is_stereo = False
            audio_mono = audio

        def _prog(pct: int) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(pct)
                except Exception:
                    pass

        # Alle 20 Defekt-Detektoren sequentiell – nach jedem Schritt Fortschritt melden
        scores = {}

        _prog(5)
        scores[DefectType.CLICKS] = self._detect_clicks(audio_mono if not is_stereo else audio)
        _prog(10)
        scores[DefectType.CRACKLE] = self._detect_crackle(audio_mono)
        _prog(16)
        scores[DefectType.HUM] = self._detect_hum(audio_mono)
        _prog(22)
        scores[DefectType.WOW] = self._detect_wow(audio_mono)              # IEC 60386 < 0.5 Hz
        scores[DefectType.FLUTTER] = self._detect_flutter(audio_mono)          # IEC 60386 0.5–200 Hz
        scores[DefectType.AZIMUTH_ERROR] = self._detect_azimuth_error(audio)  # PHD-Slope L/R
        _prog(28)
        scores[DefectType.STEREO_IMBALANCE] = (
            self._detect_stereo_imbalance(audio) if is_stereo else DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)
        )
        _prog(34)
        scores[DefectType.DIGITAL_ARTIFACTS] = self._detect_digital_artifacts(audio_mono)
        _prog(40)
        scores[DefectType.LOW_FREQ_RUMBLE] = self._detect_low_freq_rumble(audio_mono)
        _prog(46)
        scores[DefectType.HIGH_FREQ_NOISE] = self._detect_high_freq_noise(audio_mono)
        _prog(52)
        scores[DefectType.COMPRESSION_ARTIFACTS] = self._detect_compression_artifacts(audio_mono)
        _prog(58)
        scores[DefectType.PHASE_ISSUES] = (
            self._detect_phase_issues(audio) if is_stereo else DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)
        )
        _prog(64)
        scores[DefectType.DROPOUTS] = self._detect_dropouts(audio_mono)
        _prog(70)
        # --- Weltklasse-Erweiterung: 4 neue Defekttypen ---
        scores[DefectType.CLIPPING] = self._detect_clipping(audio_mono)
        _prog(75)
        scores[DefectType.DC_OFFSET] = self._detect_dc_offset(audio_mono)
        _prog(80)
        scores[DefectType.BANDWIDTH_LOSS] = self._detect_bandwidth_loss(audio_mono)
        _prog(84)
        scores[DefectType.PITCH_DRIFT] = self._detect_pitch_drift(audio_mono)
        _prog(88)
        scores[DefectType.REVERB_EXCESS] = self._detect_reverb_excess(audio_mono)
        _prog(91)
        scores[DefectType.PRINT_THROUGH] = self._detect_print_through(audio_mono)
        _prog(94)
        # --- Weltklasse-Erweiterung Runde 3 ---
        scores[DefectType.QUANTIZATION_NOISE] = self._detect_quantization_noise(audio_mono)
        _prog(96)
        scores[DefectType.JITTER_ARTIFACTS] = self._detect_jitter_artifacts(audio_mono)
        _prog(98)
        scores[DefectType.DYNAMIC_COMPRESSION_EXCESS] = self._detect_dynamic_compression_excess(audio_mono)
        _prog(99)

        analysis_time = time.time() - start_time
        duration = len(audio_mono) / sr

        logger.info(
            f"DefectScan completed: {analysis_time:.2f}s für {duration:.1f}s Audio ({analysis_time/duration*100:.1f}% overhead)"
        )

        # Forensische Tonträgerkettenerkennung (MediumDetector, DSP-basiert)
        _fmd_result: Dict = {}
        try:
            from backend.core.forensics.medium_detector import MediumDetector as _ForensicMD

            # Für lange Dateien: nur erste 30 s analysieren (Geschwindigkeit)
            _max_forensic = sr * 30
            _audio_forensic = audio_mono[:_max_forensic] if len(audio_mono) > _max_forensic else audio_mono
            logger.debug(f"[SCAN] MediumDetector.detect() → {len(_audio_forensic)/sr:.1f}s Audio …", flush=True)
            _fmd_result = _ForensicMD().detect(_audio_forensic, sr)
            # MediumDetectionResult ist ein Dataclass — getattr statt .get()
            _multi = getattr(_fmd_result, "is_multi_generation", None)
            _chain_val = getattr(_fmd_result, "transfer_chain", None) or getattr(_fmd_result, "chain", "?")
            logger.debug(
                f"[SCAN] MediumDetector OK: multi={_multi}, chain={_chain_val}",
                flush=True,
            )
        except Exception as _fmd_err:
            logger.debug(f"[SCAN] MediumDetector FEHLER: {_fmd_err}", flush=True)
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
        hum_energy = np.sum(psd[(freqs >= 50) & (freqs <= 70)]) / _psd_total  # noqa: F841

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
            f"Mono material scores: " f"shellac={shellac_score:.2f}, vinyl={vinyl_score:.2f}, tape={tape_score:.2f}"
        )

        # Select best match (minimum threshold 0.5)
        scores = {MaterialType.SHELLAC: shellac_score, MaterialType.VINYL: vinyl_score, MaterialType.TAPE: tape_score}

        best_material = max(scores, key=scores.get)
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

    def _detect_clicks(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Clicks (kurze, impulsive Störungen)."""
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Inter-sample difference (klassischer Click-Detection Ansatz)
        diff = np.abs(np.diff(audio))
        threshold_dynamic = self.thresholds[DefectType.CLICKS] * np.percentile(diff, 99.5)

        # Finde Click-Events
        click_mask = diff > threshold_dynamic
        click_indices = np.where(click_mask)[0]

        if len(click_indices) == 0:
            return DefectScore(DefectType.CLICKS, 0.0, 1.0)

        # Gruppiere Clicks (innerhalb 0.01s = zusammengehörig)
        click_groups = []
        current_group = [click_indices[0]]
        for idx in click_indices[1:]:
            if idx - current_group[-1] < int(0.01 * self.sample_rate):
                current_group.append(idx)
            else:
                click_groups.append(current_group)
                current_group = [idx]
        click_groups.append(current_group)

        # Konvertiere zu Zeitstempeln
        locations = [(group[0] / self.sample_rate, group[-1] / self.sample_rate) for group in click_groups]

        # Severity: Click-Rate pro Sekunde
        duration = len(audio) / self.sample_rate
        click_rate = len(click_groups) / duration
        severity = min(1.0, click_rate / 20)  # 20 clicks/sec = severity 1.0

        return DefectScore(
            defect_type=DefectType.CLICKS,
            severity=severity,
            confidence=0.9,
            locations=locations,
            metadata={"click_rate": click_rate, "total_clicks": len(click_groups)},
        )

    def _detect_crackle(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Crackle (kontinuierliches leises Knistern, z.B. Vinyl-Surface-Noise)."""
        # Crackle = High-Pass gefiltert + Envelope-Detection
        sos = signal.butter(4, 3000, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # Envelope via Hilbert-Transform
        analytic_signal = signal.hilbert(audio_hp)
        envelope = np.abs(analytic_signal)

        # Smoothed envelope
        window_size = int(0.05 * self.sample_rate)  # 50ms window
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Crackle-Detektion: Regions mit erhöhtem Noise-Floor
        threshold = self.thresholds[DefectType.CRACKLE] * np.percentile(envelope_smooth, 95)
        crackle_mask = envelope_smooth > threshold

        severity = np.sum(crackle_mask) / len(audio)  # Anteil der Samples mit Crackle

        # Finde zusammenhängende Regionen
        from scipy.ndimage import label

        labeled_array, num_features = label(crackle_mask)
        locations = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) > int(0.01 * self.sample_rate):  # Mindestens 10ms
                start = indices[0] / self.sample_rate
                end = indices[-1] / self.sample_rate
                locations.append((start, end))

        return DefectScore(
            defect_type=DefectType.CRACKLE,
            severity=min(1.0, severity * 10),  # 10% Crackle = severity 1.0
            confidence=0.8,
            locations=locations[:50],  # Maximal 50 Locations speichern
            metadata={"crackle_percentage": severity * 100},
        )

    def _detect_hum(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hum (50Hz/60Hz AC-Brummen und Harmonische)."""
        # FFT für Frequenz-Analyse
        fft_size = min(len(audio), int(4 * self.sample_rate))  # 4 Sekunden oder weniger
        freqs = fft.rfftfreq(fft_size, 1 / self.sample_rate)
        spectrum = np.abs(fft.rfft(audio[:fft_size]))

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
        rms_series = np.array([
            float(np.sqrt(np.mean(audio[i : i + win_len] ** 2) + 1e-12))
            for i in range(0, n - win_len, hop)
        ], dtype=np.float32)

        if len(rms_series) < 4:
            return DefectScore(DefectType.WOW, 0.0, 0.3)

        # Normalize and check spectral content below 0.5 Hz
        rms_series = rms_series / (rms_series.mean() + 1e-12)
        frame_rate = self.sample_rate / hop   # frames per second of rms_series
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
        zcr_series = np.array([
            float(np.sum(np.abs(np.diff(np.signbit(audio[i : i + win_len])))))
            for i in range(0, n - win_len, hop)
        ], dtype=np.float32)

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
            phase_diffs.append(np.angle(cross))   # radians, per bin

        if not phase_diffs:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        mean_phase = np.mean(np.array(phase_diffs), axis=0)  # (N_fft//2+1,)
        mean_phase_deg = np.degrees(np.unwrap(mean_phase))

        # Linear fit in 1–8 kHz range (avoids DC and near-Nyquist noise)
        fit_mask = (freqs_hz >= 1000.0) & (freqs_hz <= 8000.0)
        if fit_mask.sum() < 4:
            return DefectScore(DefectType.AZIMUTH_ERROR, 0.0, 0.2)

        x = freqs_hz[fit_mask] / 1000.0   # kHz
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
        """Erkennt Compression-Artifacts (MP3/AAC Pre-Echo, Ringing)."""
        # Analyse der spektralen Kontraste (MP3 reduziert diese)
        # STFT für Time-Frequency Analysis
        f, t, Zxx = signal.stft(audio, self.sample_rate, nperseg=1024)
        spectrogram = np.abs(Zxx)

        # Spectral Flatness Measure (SFM)
        # Hohe SFM = noise-like (bei MP3-Kompression reduziert)
        geometric_mean = np.exp(np.mean(np.log(spectrogram + 1e-10), axis=0))
        arithmetic_mean = np.mean(spectrogram, axis=0)
        sfm = geometric_mean / (arithmetic_mean + 1e-10)

        compression_score = 1.0 - np.mean(sfm)  # Niedrige SFM = Compression

        threshold = self.thresholds[DefectType.COMPRESSION_ARTIFACTS]
        severity = min(1.0, compression_score / threshold)

        return DefectScore(
            defect_type=DefectType.COMPRESSION_ARTIFACTS,
            severity=severity,
            confidence=0.7,  # Schwierig zu detektieren
            locations=[],
            metadata={"spectral_flatness_mean": np.mean(sfm)},
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
        """Erkennt Dropouts (kurze Stille/niedrig-Energie Segmente)."""
        # RMS in kurzen Fenstern
        window_size = int(0.01 * self.sample_rate)  # 10ms windows
        hop_size = window_size // 2

        rms_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i : i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)

        # Threshold für Dropout (10% der median RMS)
        median_rms = np.median(rms_values)
        dropout_threshold = 0.1 * median_rms

        dropout_mask = rms_values < dropout_threshold

        # Finde Dropout-Events
        from scipy.ndimage import label

        labeled_array, num_dropouts = label(dropout_mask)

        locations = []
        for i in range(1, num_dropouts + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) >= 2:  # Mindestens 2 Windows = 10-20ms
                start = indices[0] * hop_size / self.sample_rate
                end = indices[-1] * hop_size / self.sample_rate
                locations.append((start, end))

        # Severity
        duration = len(audio) / self.sample_rate
        dropout_rate = len(locations) / duration
        severity = min(1.0, dropout_rate / 1.0)  # 1 dropout/sec = max

        return DefectScore(
            defect_type=DefectType.DROPOUTS,
            severity=severity,
            confidence=0.85,
            locations=locations[:50],
            metadata={"dropout_count": len(locations), "dropout_rate": dropout_rate},
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
                    logger.debug(
                        "§6.3 _detect_clipping: SOFT_SATURATION erkannt (even-harmonic profile) — kein Repair"
                    )
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
                locations: List[Tuple[float, float]] = []
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
        """Erkennt konstanten Pitch-Drift / Geschwindigkeitsfehler (≠ WOW_FLUTTER).

        Unterschied zu WOW_FLUTTER:
          - WOW_FLUTTER: *periodische* Tonhöhenschwankung (< 10 Hz Modulation)
          - PITCH_DRIFT: *monotoner / konstanter* Geschwindigkeitsfehler
            (z.B. Tape spielt 1–3% zu langsam oder zu schnell)

        Methodik:
          - Teilt Audio in Langzeit-Segmente (z.B. 10 s)
          - Schätzt die dominante Grundfrequenz via Autokorrelation
          - Vergleicht Grundfrequenz zwischen frühem und spätem Segment
          - Großer monotoner Drift → hohe Severity; periodisches → WOW_FLUTTER zuständig
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
        # WOW_FLUTTER wird für periodische Schwankungen verantwortlich sein;
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
        alpha_pre_list: list = []   # Gemessene Pre-Echo-Amplituden
        alpha_post_list: list = []  # Gemessene Post-Echo-Amplituden
        locations: List[Tuple[float, float]] = []

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

    logger.debug(f"\n{'='*60}")
    logger.debug(f"DEFECT SCAN RESULTS")
    logger.debug(f"{'='*60}")
    logger.debug(f"Material: {result.material_type.value}")
    logger.debug(f"Duration: {result.duration_seconds:.1f}s")
    logger.debug(
        f"Analysis Time: {result.analysis_time_seconds:.3f}s ({result.analysis_time_seconds/result.duration_seconds*100:.1f}% overhead)"
    )
    logger.debug(f"\nTop 5 Defects:")
    for i, score in enumerate(result.get_top_defects(5), 1):
        logger.debug(f"  {i}. {score}")

    logger.debug(f"\nTotal Severity: {result.get_total_severity():.3f}")
