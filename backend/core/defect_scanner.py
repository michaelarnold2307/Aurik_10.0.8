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

import contextlib
import hashlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import scipy.fft as fft
import scipy.signal as signal

from backend.core.material_canonical import canonical_material_key

# §6.3 CLIPPING vs SOFT_SATURATION discrimination via THD analysis (lazy import)
try:
    from backend.core.clipping_detection import ClippingType as _ClippingType
    from backend.core.clipping_detection import classify_clipping as _classify_clipping

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
    """32 Defekttypen für weltklasse Audio-Restauration.

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
    # --- v9.10.77: Vocal-Harshness (ergibt 30 DefectTypes) ---
    VOCAL_HARSHNESS = "vocal_harshness"  # Vokale Härte/Übersteuerung/Kratzigkeit im 2–6 kHz Band → phase_42 + phase_19
    # --- v9.10.x: Dolby NR Mismatch (ergibt 31 DefectTypes) ---
    DOLBY_NR_MISMATCH = (
        "dolby_nr_mismatch"  # Dolby B/C/S encode ohne passende Dekodierung → +6–20 dB HF-Anhebung → phase_04 + phase_14
    )
    # --- v9.10.x: Tape Head Level Dip (ergibt 32 DefectTypes) ---
    TAPE_HEAD_LEVEL_DIP = "tape_head_level_dip"  # Graduelle Pegeleinbrüche durch Bandkopf-Kontaktdruckvariation / Capstan-Unregelmäßigkeit → phase_12
    # --- v9.10.98: 12 neue DefectTypes → 44 Gesamtanzahl (SOTA-Erweiterung) ---
    # Echte Lücken (6 neue Defekttypen):
    MODULATION_NOISE = "modulation_noise"  # Signal-abhängiges Rauschen bei Bandaufnahmen (moduliert mit Signalpegel) → phase_59 — Esquef & Biscainho 2006
    INNER_GROOVE_DISTORTION = "inner_groove_distortion"  # Vinyl-IGD: Abtastverzerrung nimmt zum Platteninneren zu (geringere Rillengeschwindigkeit) → phase_60
    GROOVE_ECHO = (
        "groove_echo"  # Vinyl-Rillen-Pre-Echo durch Deformation benachbarter Rillen (~1.8 s Vorecho) → phase_61
    )
    CROSSTALK = "crosstalk"  # Kanalübersprechen in frühen Stereo-Aufnahmen (Kanaltrennung < 20 dB) → phase_62
    INTERMODULATION_DISTORTION = "intermodulation_distortion"  # IMD: Summen-/Differenzfrequenzen durch nichtlineare Verstärkerketten (Volterra) → phase_63
    TAPE_SPLICE_ARTIFACT = "tape_splice_artifact"  # Bandschnitt-Artefakte: Klick + Pegelsprung + Phasendiskontinuität an Klebestellen → phase_64
    # SOTA-Upgrades bestehender Defekte (6 neue Sub-Typen):
    HF_REMANENCE_LOSS = "hf_remanence_loss"  # Magnetische Remanenz-Degradation: HF-Verlust durch Alterung (anders als nie aufgenommenes HF) → phase_06 + age-model
    STYLUS_DAMAGE = "stylus_damage"  # Nadelbeschädigung/-abnutzung: asymmetrische Abtastverzerrung (anders als generisches Crackle) → phase_09 + phase_23
    STICKY_SHED_RESIDUE = "sticky_shed_residue"  # Binder-Hydrolyse-Residuen: moduliertes Rauschen + Pegeleinbrüche nach Backen → phase_24 + phase_29
    MULTIBAND_WOW_FLUTTER = "multiband_wow_flutter"  # Frequenzabhängiger Wow/Flutter (Kopfspalt-Geometrie) — Czyzewski 2023 → phase_12 (multi-band)
    GENERATION_LOSS = (
        "generation_loss"  # Kumulativer Generationsverlust durch Tape-Dubbing → ganzheitliches Degradationsmodell
    )
    MOTOR_INTERFERENCE = "motor_interference"  # Plattenspieler-Motorinterferenz: harmonische Obertöne 80–300 Hz (nicht nur Rumble) → phase_02 + phase_05


class MaterialType(Enum):
    """Material-Typen für adaptive Thresholds."""

    SHELLAC = "shellac"
    VINYL = "vinyl"
    VINYL_STANDARD = "vinyl"  # alias → canonical key "vinyl" (material_canonical.py)
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
            DefectType.VOCAL_HARSHNESS: 0.5,  # Schwere Nadel + Trichter: Vokal-Verzerrung im Mitteltonbereich häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Shellac-Ära vor Dolby NR (Dolby 1966) — kein Dolby-Mismatch möglich
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Schellack hat keine Tape-Kopf-Mechanik
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Shellac mechanisch — kein Magnetband-Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 0.2,  # SEHR HÄUFIG: Mechanische Abtastung, schwere Nadel → IGD extrem ausgeprägt
            DefectType.GROOVE_ECHO: 0.3,  # Weiche Schellackmasse → starke Rillenverformung → Vorecho häufig
            DefectType.CROSSTALK: 1.0,  # N/A: Shellac immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.3,  # Trichter-Aufnahme nichtlinear → IMD durch mechanische Kopplung
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Shellac ist kein Band — kein Schnitt
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Shellac mechanisch — keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 0.2,  # SEHR HÄUFIG: Schwere Stahlnadeln zerstören Rillen bei Wiederholung
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Shellac hat keinen Binder wie Magnetband
            DefectType.MULTIBAND_WOW_FLUTTER: 0.5,  # Mechanischer Antrieb: frequenzunabhängiger Wow/Flutter
            DefectType.GENERATION_LOSS: 0.6,  # Matrizen-Pressung: jede Generation verliert Detail
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Grammophon-Motor: Federwerk/Elektro → harmonische Störungen
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
            DefectType.VOCAL_HARSHNESS: 0.4,  # Tonabnehmer-Verzerrung + Phono-Stufe → Vokal-Übersteuerung häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Vinyl-Heimaufnahmen selten mit Dolby NR — Schallplatten nutzen kein Dolby
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Vinyl hat keine Tape-Kopf-Mechanik
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Vinyl mechanisch — kein Magnetband-Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 0.15,  # EXTREM HÄUFIG: Abtastverzerrung zum Platteninneren — Schlüsseldefekt!
            DefectType.GROOVE_ECHO: 0.2,  # HÄUFIG: Laute Passagen deformieren Nachbarrille → Pre-Echo ~1.8 s
            DefectType.CROSSTALK: 0.5,  # Frühe Stereo-Vinyl: Kanaltrennung oft nur 15–20 dB
            DefectType.INTERMODULATION_DISTORTION: 0.4,  # Schneidlack-Nichtlinearität → IMD bei Hochpegel
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Vinyl hat keine Bandschnitte
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Vinyl mechanisch — keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 0.3,  # Abgenutzte Nadel → asymmetrische Verzerrung, häufig bei gebrauchten Platten
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Vinyl hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 0.6,  # Plattenspieler: frequenzunabhängig
            DefectType.GENERATION_LOSS: 0.7,  # Pressung: marginal (Master→Stamper)
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Plattenspieler-Motor: Gleichstrom-/Synchron-Störungen 80–300 Hz
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
            DefectType.VOCAL_HARSHNESS: 0.4,  # Bandsättigung + HF-Peaking → Vokal-Härte bei Hochpegel-Passagen
            DefectType.DOLBY_NR_MISMATCH: 0.25,  # SEHR HÄUFIG: Dolby B/C bei Heimkassetten 1975–2000; Playback ohne Dolby-Dekoder → HF-Anhebung +6–20 dB
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.20,  # SEHR HÄUFIG: Kompaktkassetten-Transport verursacht Kopf-Kontakt-Druckvariation durch Capstan/Andruckrolle
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.15,  # EXTREM HÄUFIG: Signal-abhängiges Rauschen bei JEDER Bandaufnahme — Esquef 2006
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.CROSSTALK: 0.4,  # Kassette: Spur-Übersprechen bei schmalen 4-Spur-Kassetten
            DefectType.INTERMODULATION_DISTORTION: 0.4,  # Bandkopf-/Verstärker-Nichtlinearität → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.2,  # HÄUFIG: Bandschnitte bei Heim- und Profi-Kassetten
            DefectType.HF_REMANENCE_LOSS: 0.15,  # SEHR HÄUFIG: Alterung → HF-Verlust über Jahrzehnte
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Tape hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 0.2,  # HÄUFIG: Binder-Hydrolyse bei alten Kassetten (1970er–90er)
            DefectType.MULTIBAND_WOW_FLUTTER: 0.2,  # HÄUFIG: Kopfspalt + Bandkontakt → frequenzabhängiges Flutter
            DefectType.GENERATION_LOSS: 0.2,  # HÄUFIG: Kassetten-Dubbing (Band→Band-Kopien häufig)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Kassettenmotor-Störung → über HUM/FLUTTER abgedeckt
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
            DefectType.VOCAL_HARSHNESS: 0.25,  # Loudness-War-Mastering → Vokal-Übersteuerung/Harshness SEHR häufig bei CD
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: CD ist digital, kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: CD ist digital, kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: CD digital — kein analoges Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: CD hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: CD hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # CD digital: Crosstalk nur in extremen Fällen (L/R-Bleed bei schlechtem Mastering)
            DefectType.INTERMODULATION_DISTORTION: 0.7,  # CD: IMD nur bei analogem Mastering-Signalpfad
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: CD hat keine Bandschnitte
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: CD digital — keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: CD hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: CD hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: CD Crystal-Clock
            DefectType.GENERATION_LOSS: 0.7,  # Selten: Mehrfach-Transcode in digitaler Kette
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: CD-Laufwerk digital
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
            DefectType.VOCAL_HARSHNESS: 0.4,  # Profi-Bandsättigung bei hohem Bandfluss → Vokal-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 0.6,  # Möglich: Profi-Spulenband mit Dolby A/SR — Broadcast-Dekoder fehlt oft bei Archivierung
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.40,  # Möglich: Spulenbandtransport stabiler als Kassette, aber Kopfverschleiß/Alignmentfehler möglich
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.12,  # EXTREM HÄUFIG: Profi-Spulenband bei hohem Bandfluss — Signal-abhängig
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Tape hat keine Rillen
            DefectType.CROSSTALK: 0.3,  # Frühe Stereo-Spulenbänder: Spur-Übersprechen bei Halbspur-Stereo
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Profi-Röhrenverstärker + Schneidkopf → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.15,  # SEHR HÄUFIG: Professionelle Spulenbänder mit vielen Klebestellen
            DefectType.HF_REMANENCE_LOSS: 0.12,  # SEHR HÄUFIG: Profi-Spulenband altert → HF-Verlust
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Tape hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 0.1,  # EXTREM HÄUFIG: Polyester-Urethan-Bänder (Ampex 456, Scotch 226) — Sticky-Shed-Syndrom
            DefectType.MULTIBAND_WOW_FLUTTER: 0.2,  # Profi-Kopfspalt + Bandkontakt → frequenzabhängig
            DefectType.GENERATION_LOSS: 0.15,  # SEHR HÄUFIG: Studio-Dubbing (Mix → Master → Copy)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Profi-Tape-Motor → über WOW/FLUTTER abgedeckt
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
            DefectType.VOCAL_HARSHNESS: 0.3,  # DAT: digitale Übersteuerung + Quell-Material-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: DAT ist digital — kein analoger Dolby-NR-Kompander
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: DAT ist digital mit Drehtrommel — keine analoge Kopf-Kontaktdruckvariation
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: DAT digital — kein analoges Modulationsrauschen
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: DAT hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: DAT hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # DAT digital: minimalsts Crosstalk
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # DAT: IMD nur bei analogem Eingang
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: DAT digital — kein physischer Schnitt
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: DAT digital — keine magnetische Remanenz
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: DAT hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: DAT-Kassette anderes Bindemittel
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: DAT Crystal-Clock
            DefectType.GENERATION_LOSS: 0.7,  # Selten: DAT-zu-DAT-Kopie möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: DAT digital
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
            DefectType.VOCAL_HARSHNESS: 0.3,  # MP3-Low: Codec-Artefakte + Quell-Clipping → Vokal-Härte häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MP3 digital — kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MP3 digital — kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MP3 digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: MP3 digital (minimalster L/R-Bleed)
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MP3: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MP3 digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MP3 digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MP3 digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MP3 digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MP3 digital
            DefectType.GENERATION_LOSS: 0.2,  # SEHR HÄUFIG: Mehrfach-Transkodierung (MP3→WAV→MP3)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MP3 digital
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
            DefectType.VOCAL_HARSHNESS: 0.3,  # MP3-High: Quell-Mastering-Clipping → Vokal-Übersteuerung möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MP3 digital — kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MP3 digital — kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MP3 digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MP3 hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: MP3 digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MP3: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MP3 digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MP3 digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MP3 digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MP3 digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MP3 digital
            DefectType.GENERATION_LOSS: 0.3,  # HÄUFIG: Mehrfach-Transkodierung (weniger aggressiv bei 192+ kbps)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MP3 digital
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
            DefectType.VOCAL_HARSHNESS: 0.3,  # AAC-Codec: Quell-Mastering-Härte + Codec-Artefakte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: AAC digital — kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: AAC digital — kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: AAC digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: AAC hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: AAC hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: AAC digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # AAC: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: AAC digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: AAC digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: AAC digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: AAC digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: AAC digital
            DefectType.GENERATION_LOSS: 0.3,  # Mehrfach-Transkodierung möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: AAC digital
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
            DefectType.VOCAL_HARSHNESS: 0.4,  # ATRAC-Codec: Vokal-Artefakte bei 132 kbps → Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: MiniDisc ist ATRAC-digital — kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: MiniDisc digital — kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: MiniDisc ATRAC digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: MiniDisc hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: MiniDisc hat keine Rillen
            DefectType.CROSSTALK: 0.8,  # MiniDisc: Joint-Stereo → minimaler L/R-Bleed
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # MiniDisc: IMD nur aus analogem Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: MiniDisc digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: MiniDisc digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: MiniDisc digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: MiniDisc digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: MiniDisc Crystal-Clock
            DefectType.GENERATION_LOSS: 0.3,  # ATRAC-Transkodierung möglich
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: MiniDisc digital
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
            DefectType.VOCAL_HARSHNESS: 0.3,  # Streaming: Quell-Mastering-Übersteuerung + Codec-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Streaming digital — kein Dolby-Analogband-NR
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Streaming digital — kein Magnetband-Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Streaming digital
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Streaming hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Streaming hat keine Rillen
            DefectType.CROSSTALK: 0.9,  # N/A: Streaming digital
            DefectType.INTERMODULATION_DISTORTION: 0.8,  # Streaming: IMD nur aus Quellmaterial
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Streaming digital
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Streaming digital
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Streaming digital
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Streaming digital
            DefectType.MULTIBAND_WOW_FLUTTER: 1.0,  # N/A: Streaming digital
            DefectType.GENERATION_LOSS: 0.2,  # SEHR HÄUFIG: Streaming-Transkodierung (YouTube, Spotify)
            DefectType.MOTOR_INTERFERENCE: 1.0,  # N/A: Streaming digital
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
            DefectType.VOCAL_HARSHNESS: 0.6,  # Wachswalze: Mittelton-Verzerrung durch Trichter-Resonanz
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Wachswalze (1890–1930) — Dolby NR erst 1966 erfunden
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Wachswalze hat kein Band/Kopf
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Wachswalze mechanisch
            DefectType.INNER_GROOVE_DISTORTION: 0.15,  # EXTREM HÄUFIG: Wachswalze mit primitiver Nadel
            DefectType.GROOVE_ECHO: 0.3,  # Weiche Wachsmasse → Rillenverformung
            DefectType.CROSSTALK: 1.0,  # N/A: immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.3,  # Trichter-Aufnahme nichtlinear
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Wachswalze
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: Wachswalze mechanisch
            DefectType.STYLUS_DAMAGE: 0.15,  # EXTREM HÄUFIG: Stahlnadeln zerstören Wachsrillen
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Wachswalze
            DefectType.MULTIBAND_WOW_FLUTTER: 0.4,  # Mechanischer Antrieb
            DefectType.GENERATION_LOSS: 0.5,  # Duplikat-Walzen: Qualitätsverlust durch Kopierverfahren
            DefectType.MOTOR_INTERFERENCE: 0.3,  # Federwerk/Elektromotor-Störungen
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
            DefectType.VOCAL_HARSHNESS: 0.5,  # Drahtband: Magnetisierungs-Verzerrung → Vokal-Härte möglich
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Drahtband (1940–1955) — Dolby NR erst 1966 erfunden
            DefectType.TAPE_HEAD_LEVEL_DIP: 0.35,  # Möglich: Drahtband-Kopf-Kontakt instabil
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 0.2,  # Drahtband: magnetisches Modulationsrauschen vorhanden
            DefectType.INNER_GROOVE_DISTORTION: 1.0,  # N/A: Draht hat keine Rillen
            DefectType.GROOVE_ECHO: 1.0,  # N/A: Draht hat keine Rillen
            DefectType.CROSSTALK: 1.0,  # N/A: immer Mono
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Primitive Verstärker → IMD
            DefectType.TAPE_SPLICE_ARTIFACT: 0.3,  # Drahtschweißstellen-Artefakte möglich
            DefectType.HF_REMANENCE_LOSS: 0.2,  # Draht altert → HF-Verlust
            DefectType.STYLUS_DAMAGE: 1.0,  # N/A: Draht hat keine Nadel
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Draht hat keinen Binder
            DefectType.MULTIBAND_WOW_FLUTTER: 0.25,  # Primitive Drahtführung → fl-abhängig
            DefectType.GENERATION_LOSS: 0.3,  # Draht-Dubbing möglich
            DefectType.MOTOR_INTERFERENCE: 0.4,  # Primitive Motoren
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
            DefectType.VOCAL_HARSHNESS: 0.5,  # Heimaufnahme: primitive Mikrofone → Vokal-Verzerrung häufig
            DefectType.DOLBY_NR_MISMATCH: 1.0,  # N/A: Lacquer Disc ist kein Magnetband — kein Dolby-Kompander
            DefectType.TAPE_HEAD_LEVEL_DIP: 1.0,  # N/A: Lacquer Disc hat kein Magnetband
            # v9.10.98: 12 neue SOTA-DefectTypes
            DefectType.MODULATION_NOISE: 1.0,  # N/A: Lacquer mechanisch
            DefectType.INNER_GROOVE_DISTORTION: 0.2,  # HÄUFIG: Heimschneidmaschine → IGD
            DefectType.GROOVE_ECHO: 0.25,  # Weiche Acetat-Folie → Rillenverformung
            DefectType.CROSSTALK: 1.0,  # N/A: meist Mono
            DefectType.INTERMODULATION_DISTORTION: 0.35,  # Heimverstärker nichtlinear
            DefectType.TAPE_SPLICE_ARTIFACT: 1.0,  # N/A: Disc
            DefectType.HF_REMANENCE_LOSS: 1.0,  # N/A: mechanisch
            DefectType.STYLUS_DAMAGE: 0.2,  # Heimnadeln zerstören weiche Acetat-Rillen
            DefectType.STICKY_SHED_RESIDUE: 1.0,  # N/A: Disc
            DefectType.MULTIBAND_WOW_FLUTTER: 0.4,  # Heim-Plattenspieler
            DefectType.GENERATION_LOSS: 0.5,  # Dubbing: Acetat → Pressmatrize
            DefectType.MOTOR_INTERFERENCE: 0.35,  # Heim-Plattenspieler Motor
        },
    }

    # Location caps are disabled (0 = uncapped) to avoid losing valid events
    # on long recordings with very dense defect activity.
    _LOCATION_CAP_UNCAPPED = 0

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

        logger.info("DefectScanner initialisiert: SR=%s, Material=%s", sample_rate, material_type)

    def scan(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
        material_type: MaterialType | None = None,
        progress_callback: Optional["Callable[[float, str], None]"] = None,
        file_ext: str = "",
        forensic_medium_result: object | None = None,
    ) -> DefectAnalysisResult:
        """
        Hauptmethode: Scannt Audio-Daten und erkennt alle 20 Defekttypen.

        Args:
            audio: Audio-Daten (mono: shape=(n_samples,), stereo: shape=(n_samples, 2))
            sample_rate: Sample rate (falls nicht im Constructor gesetzt)
            material_type: Override für Material-Typ (falls nicht im Constructor gesetzt)
            file_ext: Dateiendung der Quelldatei (z.B. '.mp3') — wird an ForensicMediumDetector
                      weitergegeben, um Analog-Posterior-Zeroing anzuwenden (Bug-15-Fix).

        Returns:
            DefectAnalysisResult mit allen Scores
        """
        import time

        # Bug-15-Fix: material_type als String (z.B. 'mp3_low') → MaterialType-Enum normalisieren.
        # ClassificationResult.material kann ein String sein (forensics/medium_detector.py L1026),
        # aber _DIGITAL_NO_BUMP enthält MaterialType-Enums → String-Vergleich schlägt immer fehl
        # → TRANSPORT_BUMP wird nie übersprungen → 150+ falsche Bumps auf MP3-Dateien.
        if isinstance(material_type, str):
            try:
                material_type = MaterialType(canonical_material_key(material_type))
            except ValueError:
                logger.debug(
                    "DefectScanner.scan(): material_type='%s' unbekannt — auf None gesetzt.",
                    material_type,
                )
                material_type = None

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
            # Vectorized: non-overlapping frames via reshape (replaces list + loop)
            _n_wf = (len(_fp_mono) - 1) // _wf_hop
            if _n_wf > 2:
                _wf_mat = _fp_mono[: _n_wf * _wf_hop].reshape(_n_wf, _wf_hop)
                _wf_rms = np.sqrt(np.mean(_wf_mat**2, axis=1) + 1e-12)
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

        def _prog(pct: float, name: str = "") -> None:
            _pct = float(np.clip(pct, 0.0, 100.0))
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    try:
                        progress_callback(_pct, name)
                    except TypeError:
                        progress_callback(_pct)  # type: ignore[call-arg]

        _tail_steps_done = 0
        # Fixed tail steps: 24 defect-type ticks always run.
        # Full-audio recheck path (files > 60 s): up to 8 event-detector re-runs
        # + up to 9 severity-recheck re-runs = 17 extra ticks.  Without the
        # dynamic budget the counter reaches min(1.0, 34/34) = 1.0 too early,
        # freezing the progress bar at 99.9 % while ~40 s of work remains.
        _FIXED_TAIL = 24
        _EXTRA_TAIL = 17 if _location_offset_s > 0.0 else 0
        _TAIL_STEP_BUDGET = _FIXED_TAIL + _EXTRA_TAIL  # 24 (short) or 41 (long files)
        _DIGITAL_FAST_TAIL: frozenset[MaterialType] = frozenset(
            {
                MaterialType.CD_DIGITAL,
                MaterialType.MP3_LOW,
                MaterialType.MP3_HIGH,
                MaterialType.AAC,
                MaterialType.STREAMING,
                MaterialType.DAT,
                MaterialType.MINIDISC,
            }
        )
        _ANALOG_DEEP_TAIL: frozenset[MaterialType] = frozenset(
            {
                MaterialType.WAX_CYLINDER,
                MaterialType.LACQUER_DISC,
                MaterialType.SHELLAC,
                MaterialType.VINYL,
                MaterialType.WIRE_RECORDING,
                MaterialType.REEL_TAPE,
                MaterialType.TAPE,
            }
        )
        if material_type in _DIGITAL_FAST_TAIL:
            _tail_start = 93.0
        elif material_type in _ANALOG_DEEP_TAIL:
            _tail_start = 88.0
        else:
            _tail_start = 90.0
        _tail_span = 99.9 - _tail_start

        def _lead_pct(base_pct: float) -> float:
            """Map legacy lead progress (5..90) into the material-adaptive lead range."""
            _b = float(np.clip(base_pct, 5.0, 90.0))
            _ratio = (_b - 5.0) / 85.0
            return 5.0 + _ratio * (_tail_start - 5.0)

        def _tail_tick(name: str = "") -> None:
            """Fine-grained progress updates for heavy post-90% scan blocks."""
            nonlocal _tail_steps_done
            _tail_steps_done += 1
            _frac = min(1.0, _tail_steps_done / float(_TAIL_STEP_BUDGET))
            _prog(_tail_start + _tail_span * _frac, name)

        # Alle 28 Defekttypen sequentiell — nach jedem Schritt Fortschritt melden
        scores = {}

        _prog(_lead_pct(5), "Klicks")
        scores[DefectType.CLICKS] = self._detect_clicks(audio_mono if not is_stereo else audio)
        _prog(_lead_pct(9), "Knistern")
        scores[DefectType.CRACKLE] = self._detect_crackle(audio_mono)
        _prog(_lead_pct(15), "Brummen")
        scores[DefectType.HUM] = self._detect_hum(audio_mono)
        _prog(_lead_pct(20), "Tonhöhenschwankung")
        scores[DefectType.WOW] = self._detect_wow(audio_mono)  # IEC 60386 < 0.5 Hz
        scores[DefectType.FLUTTER] = self._detect_flutter(audio_mono)  # IEC 60386 0.5–200 Hz
        # v9.10.97 §9.1b-ext: Tape-Intro-Supplement for WOW/FLUTTER.
        # Center-cropped 60 s misses cassette motor startup instability (first 0–20 s).
        # For tape material: re-run WOW/FLUTTER on the first 20 s of full audio and
        # take max(center_crop_severity, intro_severity).  This catches capstan run-up
        # and head-engagement speed irregularities that are non-stationary in the intro.
        # Scientific basis: IEC 60386:1972 §4.2.1 — short-term speed variations must
        # be measured at representative positions including start-of-tape.
        if self.material_type in (MaterialType.TAPE, MaterialType.REEL_TAPE) and len(_audio_mono_full) > sr * 4:
            _tape_intro_n = min(int(20.0 * sr), len(_audio_mono_full))
            _tape_intro_audio = _audio_mono_full[:_tape_intro_n]
            _wow_intro = self._detect_wow(_tape_intro_audio)
            _flutter_intro = self._detect_flutter(_tape_intro_audio)
            if _wow_intro.severity > scores[DefectType.WOW].severity:
                logger.info(
                    "WOW intro-supplement: severity %.3f > center-crop %.3f (tape head startup)",
                    _wow_intro.severity,
                    scores[DefectType.WOW].severity,
                )
                scores[DefectType.WOW] = _wow_intro
                scores[DefectType.WOW].metadata["intro_supplement"] = True
            if _flutter_intro.severity > scores[DefectType.FLUTTER].severity:
                logger.info(
                    "FLUTTER intro-supplement: severity %.3f > center-crop %.3f (tape head startup)",
                    _flutter_intro.severity,
                    scores[DefectType.FLUTTER].severity,
                )
                scores[DefectType.FLUTTER] = _flutter_intro
                scores[DefectType.FLUTTER].metadata["intro_supplement"] = True
        scores[DefectType.AZIMUTH_ERROR] = self._detect_azimuth_error(audio)  # PHD-Slope L/R
        _prog(_lead_pct(26), "Stereo-Ungleichgewicht")
        scores[DefectType.STEREO_IMBALANCE] = (
            self._detect_stereo_imbalance(audio) if is_stereo else DefectScore(DefectType.STEREO_IMBALANCE, 0.0, 0.0)
        )
        _prog(_lead_pct(31), "Digitalartefakte")
        scores[DefectType.DIGITAL_ARTIFACTS] = self._detect_digital_artifacts(audio_mono)
        _prog(_lead_pct(37), "Tieffrequenz-Rumpeln")
        scores[DefectType.LOW_FREQ_RUMBLE] = self._detect_low_freq_rumble(audio_mono)
        _prog(_lead_pct(42), "Hochfrequenzrauschen")
        scores[DefectType.HIGH_FREQ_NOISE] = self._detect_high_freq_noise(audio_mono)
        _prog(_lead_pct(48), "Kompressions-Artefakte")
        scores[DefectType.COMPRESSION_ARTIFACTS] = self._detect_compression_artifacts(audio_mono)
        _prog(_lead_pct(53), "Phasenfehler")
        scores[DefectType.PHASE_ISSUES] = (
            self._detect_phase_issues(audio) if is_stereo else DefectScore(DefectType.PHASE_ISSUES, 0.0, 0.0)
        )
        _prog(_lead_pct(59), "Aussetzer")
        # §9.7.5a — Dropout detection runs on FULL audio (not center-cropped).
        # Tape dropouts occur anywhere (intro, leader, splice points).
        scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)
        _prog(_lead_pct(64), "Übersteuerung")
        # --- Weltklasse-Erweiterung: 4 neue Defekttypen ---
        scores[DefectType.CLIPPING] = self._detect_clipping(audio_mono)
        _prog(_lead_pct(69), "DC-Versatz")
        scores[DefectType.DC_OFFSET] = self._detect_dc_offset(audio_mono)
        _prog(_lead_pct(74), "Bandbreitenverlust")
        scores[DefectType.BANDWIDTH_LOSS] = self._detect_bandwidth_loss(audio_mono)
        _prog(_lead_pct(78), "Tonhöhendrift")
        scores[DefectType.PITCH_DRIFT] = self._detect_pitch_drift(audio_mono)
        _prog(_lead_pct(82), "Übermäßiger Hall")
        scores[DefectType.REVERB_EXCESS] = self._detect_reverb_excess(audio_mono)
        _prog(_lead_pct(84), "Durchkopieren")
        scores[DefectType.PRINT_THROUGH] = self._detect_print_through(audio_mono)
        _prog(_lead_pct(87), "Quantisierungsrauschen")
        # --- Weltklasse-Erweiterung Runde 3 ---
        scores[DefectType.QUANTIZATION_NOISE] = self._detect_quantization_noise(audio_mono)
        _prog(_lead_pct(89), "Jitter-Artefakte")
        scores[DefectType.JITTER_ARTIFACTS] = self._detect_jitter_artifacts(audio_mono)
        _prog(_lead_pct(90), "Überkompression")
        scores[DefectType.DYNAMIC_COMPRESSION_EXCESS] = self._detect_dynamic_compression_excess(audio_mono)
        # --- Vollständige 28-Typen-Erweiterung (F-7) ---
        scores[DefectType.SOFT_SATURATION] = self._detect_soft_saturation(audio_mono)
        _tail_tick("Soft-Sättigung")
        scores[DefectType.SIBILANCE] = self._detect_sibilance(audio_mono)
        _tail_tick("Sibilanz")
        scores[DefectType.VOCAL_HARSHNESS] = self._detect_vocal_harshness(audio_mono)
        _tail_tick("Vokalhärte")
        scores[DefectType.BIAS_ERROR] = self._detect_bias_error(audio_mono)
        _tail_tick("Bias-Fehler")
        scores[DefectType.DOLBY_NR_MISMATCH] = self._detect_dolby_nr_mismatch(audio_mono)
        _tail_tick("Dolby-NR-Mismatch")
        scores[DefectType.RIAA_CURVE_ERROR] = self._detect_riaa_curve_error(audio_mono)
        _tail_tick("RIAA-Kurve")
        scores[DefectType.HEAD_WEAR] = self._detect_head_wear(audio_mono)
        _tail_tick("Kopfverschleiß")
        scores[DefectType.TRANSIENT_SMEARING] = self._detect_transient_smearing(audio_mono)
        _tail_tick("Transienten-Schmierung")
        scores[DefectType.PRE_ECHO] = self._detect_pre_echo(audio_mono)
        _tail_tick("Pre-Echo")
        scores[DefectType.ALIASING] = self._detect_aliasing(audio_mono)
        _tail_tick("Aliasing")

        # ── v9.10.98: 12 newly added defect detectors ─────────────────────────
        scores[DefectType.MODULATION_NOISE] = self._detect_modulation_noise(audio_mono)
        _tail_tick("Modulationsrauschen")
        scores[DefectType.INNER_GROOVE_DISTORTION] = self._detect_inner_groove_distortion(audio_mono)
        _tail_tick("Innenrillen-Verzerrung")
        scores[DefectType.GROOVE_ECHO] = self._detect_groove_echo(audio_mono)
        _tail_tick("Rillenecho")
        # Crosstalk needs stereo input
        if is_stereo:
            scores[DefectType.CROSSTALK] = self._detect_crosstalk(audio)
        else:
            scores[DefectType.CROSSTALK] = DefectScore(DefectType.CROSSTALK, 0.0, 0.5, metadata={"reason": "mono"})
        _tail_tick("Übersprechen")
        scores[DefectType.INTERMODULATION_DISTORTION] = self._detect_intermodulation_distortion(audio_mono)
        _tail_tick("Intermodulation")
        scores[DefectType.TAPE_SPLICE_ARTIFACT] = self._detect_tape_splice_artifact(audio_mono)
        _tail_tick("Band-Spleiß")
        scores[DefectType.HF_REMANENCE_LOSS] = self._detect_hf_remanence_loss(audio_mono)
        _tail_tick("HF-Remanenz")
        scores[DefectType.STYLUS_DAMAGE] = self._detect_stylus_damage(audio_mono)
        _tail_tick("Nadelschaden")
        scores[DefectType.STICKY_SHED_RESIDUE] = self._detect_sticky_shed_residue(audio_mono)
        _tail_tick("Sticky-Shed")
        scores[DefectType.MULTIBAND_WOW_FLUTTER] = self._detect_multiband_wow_flutter(audio_mono)
        _tail_tick("Multiband Wow/Flutter")
        scores[DefectType.GENERATION_LOSS] = self._detect_generation_loss(audio_mono)
        _tail_tick("Generationsverlust")
        scores[DefectType.MOTOR_INTERFERENCE] = self._detect_motor_interference(audio_mono)
        _tail_tick("Motor-Interferenz")

        # §9.1a — TRANSPORT_BUMP is non-stationary (impulsive micro-speed jumps).
        # MUST run on FULL audio, same as DROPOUTS.
        # §9.4a — Digital-only materials have no mechanical transport mechanism.
        # Their MATERIAL_SENSITIVITY threshold is 1.0 (never triggers). Skip the
        # expensive full-audio frame analysis entirely to save ~15–40 s on long files.
        _DIGITAL_NO_BUMP: frozenset[MaterialType] = frozenset(
            {
                MaterialType.CD_DIGITAL,
                MaterialType.MP3_LOW,
                MaterialType.MP3_HIGH,
                MaterialType.AAC,
                MaterialType.STREAMING,
            }
        )
        if material_type in _DIGITAL_NO_BUMP:
            scores[DefectType.TRANSPORT_BUMP] = DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.95)
            logger.info(
                "DefectScanner: TRANSPORT_BUMP skipped (digital material=%s, §9.4a)",
                material_type,
            )
        else:
            scores[DefectType.TRANSPORT_BUMP] = self._detect_transport_bump(_audio_mono_full)
        _tail_tick("Transport-Bump")

        # §9.1a — TAPE_HEAD_LEVEL_DIP is non-stationary (gradual level dips from
        # head-contact pressure variation).  MUST run on FULL audio.
        # Only relevant for tape-based materials with magnetic head mechanism.
        _TAPE_HEAD_DIP_MATERIALS: frozenset[MaterialType] = frozenset(
            {
                MaterialType.TAPE,
                MaterialType.REEL_TAPE,
                MaterialType.WIRE_RECORDING,
            }
        )
        _tape_head_dip_score = self._detect_tape_head_level_dips(_audio_mono_full)
        if material_type in _TAPE_HEAD_DIP_MATERIALS:
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = _tape_head_dip_score
        elif self._should_keep_cross_material_tape_head_level_dip(_tape_head_dip_score):
            _tape_head_dip_score.severity = float(np.clip(_tape_head_dip_score.severity * 0.75, 0.0, 1.0))
            _tape_head_dip_score.confidence = float(np.clip(min(_tape_head_dip_score.confidence, 0.72), 0.05, 0.95))
            _tape_head_dip_score.metadata["cross_material_fallback"] = True
            _tape_head_dip_score.metadata["fallback_material_gate_bypassed"] = material_type.value
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = _tape_head_dip_score
        else:
            scores[DefectType.TAPE_HEAD_LEVEL_DIP] = DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.95)
        _tail_tick("Tape-Head-Level-Dip")

        # ── Full-Audio Location Re-Detection ───────────────────────────────────
        # The center-crop analysis (60 s) produces locations ONLY in the middle
        # section of the song.  For event-based defects (clicks, crackle, etc.)
        # this is misleading — a 4-minute vinyl track shows markers only between
        # ~90 s and ~150 s while defects actually occur throughout.
        #
        # Fix: Re-run the event-based detectors on the FULL mono audio to get
        # accurate full-range locations.  Severity values from the center-crop
        # are retained (they are statistically representative).
        # Semi-stationary defects (print_through, transient_smearing, pre_echo)
        # have their locations cleared — they show as full-width tints instead.
        if _location_offset_s > 0.0:
            # §9.7.5b Performance: analyze only non-center sections for location re-detection.
            # Center 60 s is already covered by the primary pass (locations offset-corrected below).
            # Event-detectors (location-critical): use FULL intro + FULL outro to ensure 100 % coverage.
            # Severity-rechecks (statistical sample only): capped at 30 s per side — severity is not
            # location-dependent; a representative sample is sufficient.
            _center_begin = _dc_mid - _dc_half  # sample index where center-crop starts
            _center_end = _dc_mid + _dc_half  # sample index where center-crop ends
            _nc_intro_audio = _audio_mono_full[:_center_begin]  # full intro (all before center)
            _nc_intro_offset_s = 0.0
            _nc_outro_audio = _audio_mono_full[_center_end:]  # full outro (all after center)
            _nc_outro_offset_s = float(_center_end) / sr
            # 30 s samples for severity-only rechecks (statistically representative)
            _sev_cap = _detector_cap_n // 2  # 30 s
            _sev_intro = _audio_mono_full[: min(_sev_cap, _center_begin)]
            _sev_intro_off = 0.0
            _sev_outro = _audio_mono_full[_center_end : _center_end + _sev_cap]
            _sev_outro_off = _nc_outro_offset_s

            # Event-based detectors: re-detect on non-center sections for locations only
            _EVENT_DETECTORS: list[tuple] = [
                (DefectType.CLICKS, self._detect_clicks),
                (DefectType.CRACKLE, self._detect_crackle),
                (DefectType.CLIPPING, self._detect_clipping),
                (DefectType.SIBILANCE, self._detect_sibilance),
                (DefectType.VOCAL_HARSHNESS, self._detect_vocal_harshness),
                (DefectType.TAPE_SPLICE_ARTIFACT, self._detect_tape_splice_artifact),
                (DefectType.STICKY_SHED_RESIDUE, self._detect_sticky_shed_residue),
                (DefectType.GROOVE_ECHO, self._detect_groove_echo),
            ]
            # Long-audio center-crop misses are most critical for cheap impulse
            # detectors; always re-check them on non-center sections to avoid blind spots.
            _FORCE_FULL_REDETECT = {
                DefectType.CLICKS,
                DefectType.CRACKLE,
                DefectType.CLIPPING,
                DefectType.SIBILANCE,
                DefectType.TAPE_SPLICE_ARTIFACT,
                DefectType.STICKY_SHED_RESIDUE,
                DefectType.GROOVE_ECHO,
            }
            for _edt, _edet_fn in _EVENT_DETECTORS:
                if _edt not in scores:
                    continue
                _need_redetect = (_edt in _FORCE_FULL_REDETECT) or (scores[_edt].severity > 0.01)
                if not _need_redetect:
                    continue
                _prev_sev = float(scores[_edt].severity)
                _prev_conf = float(scores[_edt].confidence)
                # Start with center-crop locations already offset-corrected to absolute time.
                _abs_locations: list[tuple[float, float]] = [
                    (t0 + _location_offset_s, t1 + _location_offset_s) for t0, t1 in scores[_edt].locations
                ]
                _nc_sev = _prev_sev
                _nc_conf = _prev_conf
                for _seg, _seg_off in ((_nc_intro_audio, _nc_intro_offset_s), (_nc_outro_audio, _nc_outro_offset_s)):
                    if len(_seg) < sr // 4:
                        continue
                    try:
                        _sub = _edet_fn(_seg)
                        _nc_sev = max(_nc_sev, float(_sub.severity))
                        _nc_conf = max(_nc_conf, float(_sub.confidence))
                        _abs_locations.extend((t0 + _seg_off, t1 + _seg_off) for t0, t1 in _sub.locations)
                    except Exception:
                        pass
                scores[_edt].severity = float(np.clip(_nc_sev, 0.0, 1.0))
                scores[_edt].confidence = float(np.clip(_nc_conf, 0.0, 1.0))
                if _abs_locations:
                    scores[_edt].locations = sorted(_abs_locations, key=lambda _loc: _loc[0])
                    if _prev_sev <= 0.01 and _nc_sev > 0.01:
                        scores[_edt].metadata["outside_center_crop_detected"] = True
                        scores[_edt].metadata["outside_center_crop_recheck"] = "non_center_sections"
                else:
                    scores[_edt].locations = []
                _tail_tick("Vollaudio-Recheck (Events)")

            # Severity rescue for center-crop-sensitive defects.
            # Severity is a statistical / global property — a 30 s intro + 30 s outro sample
            # is sufficient to catch intro/outro-concentrated defects without running the full
            # duration (e.g. 225 s) through expensive detectors like pitch_drift or reverb_excess.
            # §9.7.5b: use _sev_intro / _sev_outro (both ≤ 30 s) instead of _audio_mono_full.
            _FULL_AUDIO_SEVERITY_RECHECK: list[tuple] = [
                (DefectType.PITCH_DRIFT, self._detect_pitch_drift),
                (DefectType.REVERB_EXCESS, self._detect_reverb_excess),
                (DefectType.DYNAMIC_COMPRESSION_EXCESS, self._detect_dynamic_compression_excess),
                (DefectType.SOFT_SATURATION, self._detect_soft_saturation),
                (DefectType.HEAD_WEAR, self._detect_head_wear),
                (DefectType.TRANSIENT_SMEARING, self._detect_transient_smearing),
                (DefectType.PRE_ECHO, self._detect_pre_echo),
                (DefectType.INNER_GROOVE_DISTORTION, self._detect_inner_groove_distortion),
                (DefectType.STYLUS_DAMAGE, self._detect_stylus_damage),
            ]
            for _sdt, _sdet_fn in _FULL_AUDIO_SEVERITY_RECHECK:
                if _sdt not in scores:
                    continue
                _prev_sev = float(scores[_sdt].severity)
                _prev_conf = float(scores[_sdt].confidence)
                # Re-check only if the center-crop score is not already meaningful.
                # Threshold 0.10: any real center-crop detection is statistically representative;
                # running the expensive detector again on 225 s is not justified when center-crop
                # already found evidence (severity > 0.10 is a reliable detection).
                if _prev_sev >= 0.10:
                    continue
                _sev_merged = _prev_sev
                _conf_merged = _prev_conf
                for _seg, _seg_off in ((_sev_intro, _sev_intro_off), (_sev_outro, _sev_outro_off)):
                    if len(_seg) < sr // 4:
                        continue
                    try:
                        _sub = _sdet_fn(_seg)
                        _sev_merged = max(_sev_merged, float(_sub.severity))
                        _conf_merged = max(_conf_merged, float(_sub.confidence))
                        if _sub.locations:
                            scores[_sdt].locations = [(t0 + _seg_off, t1 + _seg_off) for t0, t1 in _sub.locations]
                    except Exception as _exc:
                        logger.debug("Non-stationary recheck failed for %s: %s", _sdt, _exc)
                scores[_sdt].severity = float(np.clip(_sev_merged, 0.0, 1.0))
                scores[_sdt].confidence = float(np.clip(_conf_merged, 0.0, 1.0))
                if _sev_merged > (_prev_sev + 0.02):
                    scores[_sdt].metadata["outside_center_crop_detected"] = True
                    scores[_sdt].metadata["outside_center_crop_recheck"] = "non_center_sample_severity"
                _tail_tick("Vollaudio-Recheck (Severity)")

            # DROPOUTS + TRANSPORT_BUMP + TAPE_HEAD_LEVEL_DIP already use full audio — no correction needed.
            # Event detectors re-detected above — no correction needed.
            _SKIP_OFFSET = {
                DefectType.DROPOUTS,
                DefectType.TRANSPORT_BUMP,
                DefectType.TAPE_HEAD_LEVEL_DIP,
                DefectType.CLICKS,
                DefectType.CRACKLE,
                DefectType.CLIPPING,
                DefectType.SIBILANCE,
                DefectType.VOCAL_HARSHNESS,
                DefectType.GROOVE_ECHO,
                # §9.7.5b: Event re-detection produces absolute-time locations for these too;
                # must skip the offset correction that would double-correct them.
                DefectType.TAPE_SPLICE_ARTIFACT,
                DefectType.STICKY_SHED_RESIDUE,
            }
            if is_stereo:
                _SKIP_OFFSET.add(DefectType.CLICKS)  # redundant but explicit

            # Semi-stationary defects: clear locations → full-width tint display
            _CLEAR_LOCATIONS = {DefectType.PRINT_THROUGH, DefectType.TRANSIENT_SMEARING, DefectType.PRE_ECHO}
            for _cdt in _CLEAR_LOCATIONS:
                if _cdt in scores and scores[_cdt].locations:
                    scores[_cdt].locations = []

            # Remaining detectors with locations: apply offset correction
            for _dt, _ds in scores.items():
                if _dt in _SKIP_OFFSET or _dt in _CLEAR_LOCATIONS:
                    continue
                if _ds.locations:
                    _ds.locations = [(t0 + _location_offset_s, t1 + _location_offset_s) for t0, t1 in _ds.locations]

        # ── §9.1b Intro-Salienz-Gewichtung (v9.10.97) ────────────────────────────
        # Psychoacoustic research (Zacharov & Koivuniemi 2001, Bech & Zacharov 2006):
        # the first 3-5 seconds define the listener's overall quality judgment.
        # Defects in the intro region receive a severity boost so that the pipeline
        # prioritizes their repair.  This is especially critical for tape media
        # where leader artifacts and run-in fluctuations cluster at the beginning.
        #
        # v9.10.97: Two-tier intro boost for cassette/tape head pickup errors.
        # Tape head engagement & motor stabilization cause defects up to 20 s:
        #   Tier 1 (0–5 s):  ×1.50 — perceptually critical (Zacharov 2001)
        #   Tier 2 (5–20 s): ×1.25 — cassette head settling region (tape-only)
        # Scientific basis: McKnight (1969) "Tape Reproducer Response Measurements
        # with a Reproducer Test Tape", AES Convention 36; Camras (1988) Ch. 7
        # "Transport Mechanisms — Start Transients".
        #
        # IMPORTANT: This MUST run AFTER the full-audio location re-detection and
        # offset correction above, so that t0 values are in absolute song time.
        # Running it before would check intro membership against crop-relative
        # timestamps (center of the song), not the actual intro.
        _INTRO_SECONDS = 5.0
        _INTRO_SEVERITY_BOOST = 1.5  # 50% boost for intro defects
        _TAPE_EXTENDED_INTRO_S = 20.0  # extended intro region for tape head settling
        _TAPE_EXTENDED_BOOST = 1.25  # 25% boost for tape extended intro (5–20 s)
        _is_tape_material = (
            self.material_type
            in (
                MaterialType.TAPE,
                MaterialType.REEL_TAPE,
            )
            if self.material_type is not None
            else False
        )
        _total_duration_s = len(_audio_mono_full) / sr
        if _total_duration_s > _INTRO_SECONDS * 2:  # only for non-trivial audio
            for _dt, _ds in scores.items():
                if not _ds.locations:
                    continue
                _intro_events = [(t0, t1) for t0, t1 in _ds.locations if t0 < _INTRO_SECONDS]
                # v9.10.97: tape-extended intro region (5–20 s) for head-settling defects
                _ext_intro_events = []
                if _is_tape_material:
                    _ext_intro_events = [
                        (t0, t1) for t0, t1 in _ds.locations if _INTRO_SECONDS <= t0 < _TAPE_EXTENDED_INTRO_S
                    ]
                if (_intro_events or _ext_intro_events) and _ds.severity > 0.0:
                    _n_total = max(len(_ds.locations), 1)
                    # Tier 1: standard intro boost (0–5 s)
                    _intro_fraction = len(_intro_events) / _n_total
                    _tier1_boost = 1.0 + (_INTRO_SEVERITY_BOOST - 1.0) * _intro_fraction
                    # Tier 2: tape-extended intro boost (5–20 s)
                    _ext_fraction = len(_ext_intro_events) / _n_total
                    _tier2_boost = 1.0 + (_TAPE_EXTENDED_BOOST - 1.0) * _ext_fraction
                    # Combined: multiplicative (both tiers can contribute)
                    _boost = _tier1_boost * _tier2_boost
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
                        if _ext_intro_events:
                            _ds.metadata["tape_extended_intro_events"] = len(_ext_intro_events)
                            _ds.metadata["tape_extended_intro_boost"] = round(_tier2_boost, 3)
                        logger.debug(
                            "Intro-saliency boost: %s severity %.3f → %.3f "
                            "(%d/%d events in first %.0fs, %d ext-intro events)",
                            _dt.value,
                            _old_sev,
                            _ds.severity,
                            len(_intro_events),
                            len(_ds.locations),
                            _INTRO_SECONDS,
                            len(_ext_intro_events),
                        )

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
                DefectType.VOCAL_HARSHNESS: "vocal_harshness",
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
                except Exception as _exc:
                    logger.debug("Per-channel crackle detection failed (%s): %s", _ch_label, _exc)
                try:
                    _ch_clip = self._detect_clipping(_ch_audio)
                    if _ch_clip.locations:
                        _per_channel_locs.setdefault("clipping", {})[_ch_label] = list(_ch_clip.locations)
                except Exception as _exc:
                    logger.debug("Per-channel clipping detection failed (%s): %s", _ch_label, _exc)
                # Sibilance per channel
                try:
                    _ch_sib = self._detect_sibilance(_ch_audio)
                    if _ch_sib.locations:
                        _per_channel_locs.setdefault("sibilance", {})[_ch_label] = list(_ch_sib.locations)
                except Exception as _exc:
                    logger.debug("Per-channel sibilance detection failed (%s): %s", _ch_label, _exc)
                # Transport bumps per channel — skip for pure digital material
                # (no tape/disc transport mechanism, §9.4a — same guard as main call)
                if material_type not in _DIGITAL_NO_BUMP:
                    try:
                        _ch_bump = self._detect_transport_bump(_ch_audio)
                        if _ch_bump.locations:
                            _per_channel_locs.setdefault("transport_bump", {})[_ch_label] = list(_ch_bump.locations)
                    except Exception as _exc:
                        logger.debug("Per-channel transport_bump detection failed (%s): %s", _ch_label, _exc)
                # Vocal harshness per channel
                try:
                    _ch_harsh = self._detect_vocal_harshness(_ch_audio)
                    if _ch_harsh.locations:
                        _per_channel_locs.setdefault("vocal_harshness", {})[_ch_label] = list(_ch_harsh.locations)
                except Exception as _exc:
                    logger.debug("Per-channel vocal_harshness detection failed (%s): %s", _ch_label, _exc)
                _tail_tick(f"Kanal-Analyse {_ch_label}")
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
            _tail_tick("Medium-Gate")

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
        _tail_tick("Perceptual Salience")

        # Post-calibration: material-aware confidence stabilization and
        # transparent locality limits for long center-crop scans.
        self._post_calibrate_scores(
            scores=scores,
            material_type=material_type,
            used_center_crop=(_location_offset_s > 0.0),
        )
        _tail_tick("Post-Kalibrierung")

        _prog(99)

        analysis_time = time.time() - start_time
        duration = len(audio_mono) / sr

        logger.info(
            f"DefectScan completed: {analysis_time:.2f}s für {duration:.1f}s Audio ({analysis_time / duration * 100:.1f}% overhead)"
        )

        # Forensische Tonträgerkettenerkennung (MediumDetector, DSP-basiert).
        # Wenn UV3 bereits ein vollständiges MediumDetectionResult aus dem Bridge-Cache
        # übergeben hat (forensic_medium_result), wird KEIN zweiter Detector-Aufruf
        # ausgeführt — Redundanz-Fix (Erkennung einmalig nach Import, nicht nochmals beim
        # Klick auf Restoration / Studio 2026).
        _fmd_result: object = {}
        if forensic_medium_result is not None and hasattr(forensic_medium_result, "transfer_chain"):
            _fmd_result = forensic_medium_result
            logger.debug(
                "[SCAN] ForensicMediumDetector: gecachtes Ergebnis verwendet — kein erneuter Detect-Aufruf "
                "(primary_material=%s, chain=%s)",
                getattr(_fmd_result, "primary_material", "?"),
                getattr(_fmd_result, "transfer_chain", "?"),
            )
        else:
            try:
                from backend.core.forensics.medium_detector import MediumDetector as _ForensicMD

                # Für lange Dateien: nur erste 30 s analysieren (Geschwindigkeit)
                _max_forensic = sr * 30
                _audio_forensic = audio_mono[:_max_forensic] if len(audio_mono) > _max_forensic else audio_mono
                logger.debug("[SCAN] MediumDetector.detect() → %.1fs Audio …", len(_audio_forensic) / sr)
                _fmd_result = _ForensicMD().detect(_audio_forensic, sr, file_ext=file_ext)
                # MediumDetectionResult ist ein Dataclass — getattr statt .get()
                _multi = getattr(_fmd_result, "is_multi_generation", None)
                _chain_val = getattr(_fmd_result, "transfer_chain", None) or getattr(_fmd_result, "chain", "?")
                logger.debug(
                    f"[SCAN] MediumDetector OK: multi={_multi}, chain={_chain_val}",
                )
            except Exception as _fmd_err:
                logger.debug("[SCAN] MediumDetector FEHLER: %s", _fmd_err)
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

    def _post_calibrate_scores(
        self,
        *,
        scores: dict[DefectType, DefectScore],
        material_type: MaterialType,
        used_center_crop: bool,
    ) -> None:
        """Apply conservative confidence calibration and crop-locality annotations.

        Calibration strategy:
        - Increase confidence slightly when measured severity clearly exceeds
          material-adaptive sensitivity.
        - Reduce confidence slightly for weak detections near threshold.
        - Mark long-audio center-crop locality limits for non-rechecked types.
        """
        _FULL_AUDIO_RECHECKED = {
            DefectType.DROPOUTS,
            DefectType.TRANSPORT_BUMP,
            DefectType.CLICKS,
            DefectType.CRACKLE,
            DefectType.CLIPPING,
            DefectType.SIBILANCE,
            DefectType.VOCAL_HARSHNESS,
            DefectType.PITCH_DRIFT,
            DefectType.REVERB_EXCESS,
            DefectType.DYNAMIC_COMPRESSION_EXCESS,
            DefectType.SOFT_SATURATION,
            DefectType.HEAD_WEAR,
            DefectType.TRANSIENT_SMEARING,
            DefectType.PRE_ECHO,
        }

        for defect_type, score in scores.items():
            old_conf = float(np.clip(score.confidence, 0.0, 1.0))
            severity = float(np.clip(score.severity, 0.0, 1.0))
            sensitivity = float(np.clip(self.thresholds.get(defect_type, 0.5), 1e-3, 1.0))

            # Evidence ratio > 1 means stronger-than-threshold defect evidence.
            evidence_ratio = float(severity / sensitivity)
            calib_scale = float(np.clip(0.88 + 0.20 * evidence_ratio, 0.78, 1.14))
            new_conf = float(np.clip(old_conf * calib_scale, 0.05, 0.99))

            # Long-audio center-crop uncertainty: keep severity but report slightly
            # lower confidence if no full-audio recheck path exists.
            if used_center_crop and defect_type not in _FULL_AUDIO_RECHECKED:
                has_locations = bool(score.locations)
                already_rechecked = bool(score.metadata.get("outside_center_crop_recheck"))
                if has_locations and not already_rechecked:
                    new_conf = float(np.clip(new_conf * 0.92, 0.05, 0.99))
                    score.metadata["crop_locality_limited"] = True
                    score.metadata["crop_locality_note"] = "center_crop_locations_with_offset"

            score.confidence = new_conf
            score.metadata["confidence_calibrated"] = True
            score.metadata["confidence_before_calibration"] = old_conf
            score.metadata["confidence_after_calibration"] = new_conf
            score.metadata["confidence_evidence_ratio"] = round(evidence_ratio, 4)
            score.metadata["confidence_material"] = material_type.value

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
        _mid_low_energy = np.sum(psd[(freqs >= 50) & (freqs <= 70)]) / _psd_total

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
        vinyl_score += crackle_score * 2.0  # Crackle typisch, aber nicht exklusiv (§3 Material-Mismatch-Fix)
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
            logger.info("Detected mono material: %s (score=%.2f)", best_material.value, best_score)
            return best_material
        else:
            logger.warning("Mono material unclear (best score=%.2f), using UNKNOWN", best_score)
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
        vinyl_score += crackle_score * 2.0  # Crackle typisch für Vinyl, aber nicht exklusiv (§3 Material-Mismatch-Fix)
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
        tape_score -= crackle_score * 0.5  # Gealtertes Tape kann Crackle haben (Oxidflaking) — leichte Penalty
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
        logger.debug("Material scores: %s", scores)
        logger.info("Detected material: %s (score: %.2f)", best_material[0].value, best_material[1])

        return best_material[0]

    # ========== DEFECT DETECTORS (11 Typen) ==========

    @staticmethod
    def _sample_locations_evenly(locations: list, max_n: int) -> list:
        """Return evenly sampled locations, optionally uncapped.

        If max_n <= 0, return all locations (no cap).
        Otherwise, pick representatives at uniform stride so returned positions
        span the full audio length without start-bias.
        """
        if max_n <= 0:
            return locations
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
        MAX_LOCATIONS = self._LOCATION_CAP_UNCAPPED
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
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "crackle_percentage": severity_raw * 100,
                "hp_kurtosis": hp_kurtosis,
                "kurtosis_discount": kurtosis_discount,
            },
        )

    def _detect_hum(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hum (50Hz/60Hz AC-Brummen und Harmonische).

        Multi-Window + Peak-Sharpness + Harmonic-Decay-Validation (v9.10.77b).
        Literature: Vaseghi 2008 'Advanced DSP and Noise Reduction' Ch.13.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.HUM, 0.0, 0.3)

        # --- Multi-segment windowed analysis (catch intermittent hum) ---
        seg_dur = min(4.0, n / self.sample_rate)
        seg_len = int(seg_dur * self.sample_rate)
        n_segs = max(1, min(8, n // seg_len))
        hum_freqs_50 = [50 * (k + 1) for k in range(8)]  # 8 harmonics
        hum_freqs_60 = [60 * (k + 1) for k in range(8)]
        # Harmonic weight: fundamental strongest, geometric decay
        harm_weights = np.array([1.0 / (k + 1) for k in range(8)])
        harm_weights /= harm_weights.sum()

        best_ratio = 0.0
        best_freq = 50
        best_sharpness = 0.0
        seg_ratios: list[float] = []

        for si in range(n_segs):
            start = si * (n - seg_len) // max(1, n_segs - 1) if n_segs > 1 else 0
            seg = audio[start : start + seg_len]
            # Hanning window to reduce spectral leakage
            win = np.hanning(len(seg))
            seg_win = np.asarray(seg * win, dtype=np.float64)
            spectrum = np.abs(np.fft.rfft(seg_win))
            freqs = fft.rfftfreq(len(seg), 1.0 / self.sample_rate)
            total_e = float(np.sum(spectrum**2) + 1e-12)
            bin_width = freqs[1] if len(freqs) > 1 else 1.0
            # ±2 Hz band in bins
            half_band = max(1, int(2.0 / bin_width))

            def _measure_hum_weighted(hum_f_list):
                energy = 0.0
                sharpness_sum = 0.0
                for ki, f in enumerate(hum_f_list):
                    if f >= self.sample_rate / 2:
                        break
                    idx = int(round(f / bin_width))
                    if idx >= len(spectrum):
                        break
                    b0 = max(0, idx - half_band)
                    b1 = min(len(spectrum), idx + half_band + 1)
                    peak_e = float(np.sum(spectrum[b0:b1] ** 2))
                    energy += harm_weights[ki] * peak_e
                    # Peak sharpness: ratio of peak bin to band mean
                    band_mean = float(np.mean(spectrum[b0:b1] ** 2) + 1e-12)
                    peak_val = float(spectrum[idx] ** 2)
                    sharpness_sum += peak_val / band_mean
                return energy, sharpness_sum / max(1, len(hum_f_list))

            e50, sharp50 = _measure_hum_weighted(hum_freqs_50)
            e60, sharp60 = _measure_hum_weighted(hum_freqs_60)

            if e50 > e60:
                ratio = e50 / total_e
                sharp = sharp50
                freq = 50
            else:
                ratio = e60 / total_e
                sharp = sharp60
                freq = 60

            seg_ratios.append(ratio)
            if ratio > best_ratio:
                best_ratio = ratio
                best_freq = freq
                best_sharpness = sharp

        # --- Anti-FP: Peak sharpness guard ---
        # Hum has SHARP peaks (Q > 50). Broadband bass content has low sharpness.
        # Sharpness < 1.5 means energy is spread, NOT hum.
        if best_sharpness < 1.5:
            best_ratio *= 0.15  # Massively discount — likely bass content

        # --- Anti-FP: Sub-50 Hz rumble confusion guard ---
        # If the dominant low-frequency energy is below 40 Hz, this is LOW_FREQ_RUMBLE,
        # not HUM. HUM is at 50/60 Hz (mains) with integer harmonics. Sub-40 Hz content
        # is transport rumble, acoustic feedback, or building vibration.
        # Strategy: measure energy in <40 Hz band vs 40–80 Hz band; if sub-40 Hz
        # dominates AND the sharpness at 50/60 Hz is low → suppress.
        if len(audio) > self.sample_rate:
            try:
                _sub40_sos = signal.butter(
                    4, float(np.clip(40.0 / (self.sample_rate / 2.0), 1e-6, 0.999)), btype="low", output="sos"
                )
                _sub40_audio = signal.sosfilt(_sub40_sos, audio[: min(len(audio), 4 * self.sample_rate)])
                _sub40_e = float(np.mean(_sub40_audio**2) + 1e-20)
                _hum_band_e = best_ratio * (float(np.mean(audio[: min(len(audio), 4 * self.sample_rate)] ** 2)) + 1e-20)
                if _sub40_e > _hum_band_e * 3.0 and best_sharpness < 3.0:
                    best_ratio *= 0.25  # Sub-40 Hz dominates → likely rumble, not hum
            except Exception as _exc:
                logger.debug("Hum sub-40 Hz rumble guard failed: %s", _exc)

        # --- Anti-FP: Harmonic decay validation ---
        # Real hum has monotonically decaying harmonics. Random bass doesn't.
        # (already handled by harm_weights, but sharpness is the main guard)

        threshold = self.thresholds[DefectType.HUM]
        severity = float(np.clip(best_ratio / (threshold * 0.001), 0.0, 1.0))

        # Confidence based on sharpness and multi-seg consistency
        seg_std = float(np.std(seg_ratios)) if len(seg_ratios) > 1 else 0.0
        confidence = float(
            np.clip(
                0.60 + 0.30 * min(1.0, best_sharpness / 3.0) - 0.20 * min(1.0, seg_std / (best_ratio + 1e-12)),
                0.3,
                0.98,
            )
        )

        # --- Locations: segments where hum is present ---
        locations: list[tuple[float, float]] = []
        if severity > 0.0 and len(seg_ratios) > 1:
            mean_r = float(np.mean(seg_ratios))
            for si, r in enumerate(seg_ratios):
                if r > mean_r * 0.5:
                    t0 = si * (n - seg_len) / max(1, n_segs - 1) / self.sample_rate if n_segs > 1 else 0.0
                    t1 = t0 + seg_dur
                    locations.append((t0, min(t1, n / self.sample_rate)))
            locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

        return DefectScore(
            defect_type=DefectType.HUM,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "hum_frequency": best_freq,
                "hum_ratio": best_ratio,
                "peak_sharpness": best_sharpness,
                "n_segments_analyzed": n_segs,
                "segment_consistency_std": seg_std,
            },
        )

    def _detect_wow(self, audio: np.ndarray) -> DefectScore:
        """Detect WOW: slow pitch modulation < 0.5 Hz (IEC 60386, Blum 1984).

        Upgraded v9.10.77b: Dual-track detection using both RMS-envelope AND
        instantaneous-frequency (Hilbert analytic signal) for pitch-based WOW.
        RMS catches amplitude-WOW; IF catches pure pitch-WOW missed by RMS.
        """
        n = len(audio)
        if n < self.sample_rate * 2:  # need at least 2s for sub-0.5Hz
            return DefectScore(DefectType.WOW, 0.0, 0.3)

        # --- Track 1: RMS envelope modulation (amplitude WOW) ---
        win_len = max(1, int(0.5 * self.sample_rate))
        hop = max(1, win_len // 2)
        # Vectorized RMS
        n_frames = (n - win_len) // hop
        if n_frames < 4:
            return DefectScore(DefectType.WOW, 0.0, 0.3)
        indices = np.arange(n_frames) * hop
        rms_series = np.array(
            [float(np.sqrt(np.mean(audio[i : i + win_len] ** 2) + 1e-12)) for i in indices],
            dtype=np.float64,
        )
        rms_series /= rms_series.mean() + 1e-12

        frame_rate = self.sample_rate / hop
        # Power spectrum (squared magnitude) for proper energy metric
        rms_centered = rms_series - rms_series.mean()
        win_h = np.hanning(len(rms_centered))
        fft_rms = np.abs(np.fft.rfft(rms_centered * win_h)) ** 2
        freqs_rms = np.fft.rfftfreq(len(rms_centered), d=1.0 / frame_rate)

        wow_mask = (freqs_rms > 0.02) & (freqs_rms < 0.5)  # skip DC
        total_power = float(np.sum(fft_rms[1:]) + 1e-12)  # skip DC bin
        wow_power_rms = float(np.sum(fft_rms[wow_mask])) if wow_mask.any() else 0.0
        wow_ratio_rms = wow_power_rms / total_power

        # --- Track 2: Instantaneous frequency modulation (pitch WOW) ---
        # Bandpass 80–4000 Hz to focus on pitched content
        try:
            bp_sos = signal.butter(
                3,
                [80.0 / (self.sample_rate / 2), min(4000.0, self.sample_rate * 0.45) / (self.sample_rate / 2)],
                btype="band",
                output="sos",
            )
            bp_audio = signal.sosfilt(bp_sos, audio)
            # Hilbert transform for analytic signal
            analytic = signal.hilbert(bp_audio[: min(n, self.sample_rate * 30)])  # cap at 30s
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            inst_phase = np.unwrap(np.angle(analytic_arr))
            inst_freq = np.diff(inst_phase) * self.sample_rate / (2 * np.pi)
            # Smooth to 50ms windows
            if_win = max(1, int(0.05 * self.sample_rate))
            if len(inst_freq) > if_win * 4:
                if_smooth = np.convolve(inst_freq, np.ones(if_win) / if_win, mode="valid")
                if_smooth = if_smooth / (np.median(if_smooth) + 1e-6)  # normalize
                if_centered = if_smooth - if_smooth.mean()
                if_rate = self.sample_rate / if_win
                win_if = np.hanning(len(if_centered))
                fft_if = np.abs(np.fft.rfft(if_centered * win_if)) ** 2
                freqs_if = np.fft.rfftfreq(len(if_centered), d=1.0 / if_rate)
                wow_mask_if = (freqs_if > 0.02) & (freqs_if < 0.5)
                total_if = float(np.sum(fft_if[1:]) + 1e-12)
                wow_power_if = float(np.sum(fft_if[wow_mask_if])) if wow_mask_if.any() else 0.0
                wow_ratio_if = wow_power_if / total_if
            else:
                wow_ratio_if = 0.0
        except Exception:
            wow_ratio_if = 0.0

        # --- Combine: max of both tracks ---
        wow_ratio = max(wow_ratio_rms, wow_ratio_if * 0.8)

        # --- Periodicity check (anti-FP): real WOW is quasi-periodic ---
        # Find dominant modulation frequency
        if wow_mask.any() and wow_power_rms > 0:
            dom_idx = np.argmax(fft_rms[wow_mask])
            dom_freq = float(freqs_rms[wow_mask][dom_idx])
            dom_peak = float(fft_rms[wow_mask][dom_idx])
            dom_ratio = dom_peak / (wow_power_rms + 1e-12)
        else:
            dom_freq = 0.0
            dom_ratio = 0.0

        # Periodic WOW has concentrated energy at one frequency
        periodicity_bonus = 1.0 + 0.5 * max(0.0, dom_ratio - 0.3)  # boost if periodic

        threshold = self.thresholds.get(DefectType.WOW, 0.5)
        severity = float(np.clip(wow_ratio * periodicity_bonus / (threshold * 0.3 + 1e-12), 0.0, 1.0))

        confidence = float(np.clip(0.55 + 0.30 * min(1.0, dom_ratio), 0.3, 0.92))

        return DefectScore(
            defect_type=DefectType.WOW,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "wow_energy_ratio_rms": wow_ratio_rms,
                "wow_energy_ratio_if": wow_ratio_if,
                "dominant_mod_freq_hz": dom_freq,
                "periodicity_ratio": dom_ratio,
                "frame_rate_hz": float(frame_rate),
            },
        )

    def _detect_transport_bump(self, audio: np.ndarray) -> DefectScore:
        """Detect TRANSPORT_BUMP: impulsive micro-speed jumps from tape transport shocks.

        Multi-feature algorithm (v2) that distinguishes real transport bumps from
        normal musical transients by requiring co-occurrence of 5 features:
          1. Sudden energy anomaly (RMS envelope drop > 55% or spike > 2.5×)
          2. Low-frequency thump (< 80 Hz energy surge ≥ 2.5× local context)
          3. Spectral centroid disruption (> 40% shift from local baseline)
          4. Spectral flux spike (≥ 3.5× local median — rapid timbral change)
          5. Pitch instability (ZCR derivative spike — proxy for pitch jump)

        Each candidate must satisfy:
          - Feature 1 (energy anomaly) MUST be present (mandatory)
          - Plus ≥ 2 of features 2–5 (total ≥ 3 features)
        Musical transients (drum hits, note onsets) rarely show energy anomaly
        AND spectral centroid disruption AND LF thump simultaneously.

        Scientific basis:
          - Godsill & Rayner (1998): Digital Audio Restoration — transport bump model
          - Esquef et al. (2002): Frequency-domain analysis of mechanical artifacts

        Returns:
            DefectScore with locations [(start_s, end_s), ...] for each bump event.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        hop_s = 0.010  # 10 ms hop
        win_s = 0.030  # 30 ms window
        hop = max(1, int(hop_s * sr))
        win = max(1, int(win_s * sr))
        n_frames = (n - win) // hop

        if n_frames < 20:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.3)

        hann = np.hanning(win).astype(np.float64)

        # --- Feature extraction ---
        rms_env = np.empty(n_frames, dtype=np.float64)
        lf_ratio = np.empty(n_frames, dtype=np.float64)
        sc_env = np.empty(n_frames, dtype=np.float64)
        zcr_env = np.empty(n_frames, dtype=np.float64)

        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        lf_mask = freqs < 80.0
        mf_mask = (freqs >= 80.0) & (freqs < 4000.0)

        prev_mag: np.ndarray | None = None
        flux_list: list[float] = []

        for i in range(n_frames):
            s = i * hop
            frame = audio[s : s + win].astype(np.float64)
            rms_env[i] = float(np.sqrt(np.mean(frame**2) + 1e-12))
            zcr_env[i] = float(np.sum(np.abs(np.diff(np.signbit(frame))))) / max(1.0, float(win))

            windowed = frame * hann
            mag = np.abs(np.fft.rfft(windowed))

            lf_e = float(np.sum(mag[lf_mask] ** 2))
            mf_e = float(np.sum(mag[mf_mask] ** 2)) + 1e-12
            lf_ratio[i] = lf_e / mf_e

            total_mag = float(np.sum(mag)) + 1e-12
            sc_env[i] = float(np.sum(freqs * mag)) / total_mag

            if prev_mag is not None:
                flux_list.append(float(np.sqrt(np.sum((mag - prev_mag) ** 2))))
            prev_mag = mag.copy()

        flux_env = np.array(flux_list, dtype=np.float64) if flux_list else np.zeros(1)

        # --- Adaptive local baselines (rolling median, ~1 s context) ---
        from scipy.ndimage import median_filter as _medfilt

        ctx = max(3, int(1.0 / hop_s))  # 1 s context for stable baseline
        if ctx % 2 == 0:
            ctx += 1

        rms_baseline = _medfilt(rms_env, size=ctx)
        lf_baseline = _medfilt(lf_ratio, size=ctx)
        sc_baseline = _medfilt(sc_env, size=ctx)
        flux_baseline = _medfilt(flux_env, size=min(ctx, len(flux_env)))
        _medfilt(zcr_env, size=ctx)

        # Numerical stability: clamp rms_baseline to avoid rms_ratio blowing up
        # during silent passages (fade-ins, intros, near-silence frames).
        # Without this guard, rms_baseline ≈ 0 → rms_ratio ≈ 307× → false positives.
        _rms_floor = max(float(np.percentile(rms_env[rms_env > 0], 5)) * 0.1 if np.any(rms_env > 0) else 1e-5, 1e-5)
        rms_baseline = np.maximum(rms_baseline, _rms_floor)

        # --- Feature 1 (MANDATORY): Energy anomaly ---
        rms_ratio = rms_env / (rms_baseline + 1e-12)
        feat_energy = (rms_ratio < 0.45) | (rms_ratio > 2.5)

        # --- Feature 2: Low-frequency thump ---
        lf_factor = lf_ratio / (lf_baseline + 1e-12)
        feat_lf = lf_factor > 2.5

        # --- Feature 3: Spectral centroid disruption ---
        sc_ratio = sc_env / (sc_baseline + 1e-12)
        feat_sc = (sc_ratio < 0.60) | (sc_ratio > 1.6)

        # --- Feature 4: Spectral flux spike ---
        flux_ratio = flux_env / (flux_baseline + 1e-12)
        feat_flux = np.zeros(n_frames, dtype=bool)
        feat_flux[: len(flux_ratio)] = flux_ratio > 3.5

        # --- Feature 5: Pitch instability (ZCR derivative spike) ---
        zcr_diff = np.abs(np.diff(zcr_env))
        zcr_diff_baseline = _medfilt(zcr_diff, size=min(ctx, len(zcr_diff)))
        zcr_diff_ratio = zcr_diff / (zcr_diff_baseline + 1e-12)
        feat_pitch = np.zeros(n_frames, dtype=bool)
        feat_pitch[: len(zcr_diff_ratio)] = zcr_diff_ratio > 3.0

        # --- Multi-feature scoring with MANDATORY energy requirement ---
        # Energy must be present; then count additional features
        score_arr = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            lo = max(0, i - 2)
            hi = min(n_frames, i + 3)
            # Mandatory: energy feature must be triggered in vicinity
            if not np.any(feat_energy[lo:hi]):
                continue
            hits = 1.0  # energy counts as 1
            if np.any(feat_lf[lo:hi]):
                hits += 1.0
            if np.any(feat_sc[lo:hi]):
                hits += 1.0
            if np.any(feat_flux[lo:hi]):
                hits += 1.0
            if np.any(feat_pitch[lo:hi]):
                hits += 1.0
            score_arr[i] = hits

        # Require energy + 2 more = total ≥ 3 features
        candidates = score_arr >= 3.0

        # --- Group into events (30 ms – 500 ms), merge nearby (gap ≤ 50 ms) ---
        min_frames_evt = max(1, int(0.030 / hop_s))
        max_frames_evt = max(1, int(0.500 / hop_s))
        merge_gap = max(1, int(0.050 / hop_s))

        locations: list[tuple[float, float]] = []
        magnitudes: list[float] = []
        bump_scores: list[float] = []
        suppressed_head_dip_like = 0

        i = 0
        while i < n_frames:
            if candidates[i]:
                start_frame = i
                while i < n_frames and candidates[i]:
                    i += 1
                end_frame = i
                # Merge nearby events
                while i < n_frames and i - end_frame <= merge_gap:
                    if candidates[i]:
                        while i < n_frames and candidates[i]:
                            i += 1
                        end_frame = i
                    else:
                        i += 1
                event_len = end_frame - start_frame
                if min_frames_evt <= event_len <= max_frames_evt:
                    evt_ratios = rms_ratio[start_frame:end_frame]

                    # Tape-head level dips are gradual 80-400 ms envelope drops.
                    # The crucial discriminator vs. transport bumps is morphology:
                    # they stay below the local level for most of the event and do
                    # not show a positive in-event recovery/thump.
                    _looks_like_head_level_dip = (
                        event_len * hop_s >= 0.120
                        and float(np.mean(evt_ratios < 0.70)) >= 0.80
                        and float(np.max(evt_ratios)) < 0.90
                    )
                    if _looks_like_head_level_dip:
                        suppressed_head_dip_like += 1
                        continue

                    t_start = max(0.0, float(start_frame * hop) / sr - 0.015)
                    t_end = min(float(n) / sr, float(end_frame * hop + win) / sr + 0.015)
                    locations.append((t_start, t_end))
                    magnitudes.append(float(np.max(np.abs(1.0 - evt_ratios))))
                    bump_scores.append(float(np.mean(score_arr[start_frame:end_frame])))
            else:
                i += 1

        n_bumps = len(locations)
        if n_bumps == 0:
            return DefectScore(DefectType.TRANSPORT_BUMP, 0.0, 0.6)

        duration_s = n / sr
        bump_density = n_bumps / max(1.0, duration_s / 60.0)
        max_mag = float(max(magnitudes)) if magnitudes else 0.0
        mean_score = float(np.mean(bump_scores)) if bump_scores else 3.0

        severity = float(
            np.clip(
                0.20 * min(1.0, bump_density / 8.0)
                + 0.50 * min(1.0, max_mag / 2.0)
                + 0.30 * min(1.0, (mean_score - 2.5) / 1.5),
                0.0,
                1.0,
            )
        )
        confidence = float(np.clip(0.60 + 0.08 * n_bumps, 0.60, 0.95))

        logger.info(
            "transport_bump detection: n_bumps=%d, density=%.1f/min, max_mag=%.3f, mean_score=%.2f, severity=%.3f, suppressed_head_dip_like=%d",
            n_bumps,
            bump_density,
            max_mag,
            mean_score,
            severity,
            suppressed_head_dip_like,
        )

        return DefectScore(
            defect_type=DefectType.TRANSPORT_BUMP,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "n_bumps": n_bumps,
                "bump_density_per_min": round(bump_density, 2),
                "max_magnitude": round(max_mag, 4),
                "mean_multi_feature_score": round(mean_score, 2),
                "suppressed_head_dip_like": suppressed_head_dip_like,
                "magnitudes": [round(float(m), 4) for m in magnitudes[:30]],
            },
        )

    def _detect_flutter(self, audio: np.ndarray) -> DefectScore:
        """Detect FLUTTER: rapid pitch modulation 0.5–200 Hz (IEC 60386).

        Upgraded v9.10.77b: Spectral-centroid modulation analysis replaces
        primitive ZCR. Flutter causes rapid spectral centroid oscillation
        at characteristic mechanical frequencies (guide roller 4-8 Hz,
        capstan 2-4 Hz, tape resonance 10-30 Hz).
        """
        n = len(audio)
        if n < self.sample_rate // 2:
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        # --- Short-time spectral centroid series ---
        # 20ms windows, 10ms hop (captures up to 50 Hz modulation = Nyquist)
        win_len = max(256, int(0.020 * self.sample_rate))
        hop = max(1, win_len // 2)
        n_frames = (n - win_len) // hop
        if n_frames < 16:
            return DefectScore(DefectType.FLUTTER, 0.0, 0.3)

        # Vectorized spectral centroid computation
        centroid_series = np.empty(n_frames, dtype=np.float64)
        freqs_stft = np.fft.rfftfreq(win_len, 1.0 / self.sample_rate)
        win_h = np.hanning(win_len)

        for fi in range(n_frames):
            start = fi * hop
            seg = audio[start : start + win_len] * win_h
            mag = np.abs(np.fft.rfft(seg))
            total_mag = float(np.sum(mag) + 1e-12)
            centroid_series[fi] = float(np.sum(freqs_stft * mag) / total_mag)

        # Normalize centroid to fractional deviation
        mean_c = float(np.mean(centroid_series) + 1e-6)
        centroid_norm = (centroid_series - mean_c) / mean_c

        # --- Modulation spectrum of centroid ---
        frame_rate = self.sample_rate / hop
        nyquist_mod = frame_rate / 2.0
        centroid_win = np.hanning(len(centroid_norm))
        fft_c = np.abs(np.fft.rfft(centroid_norm * centroid_win)) ** 2
        freqs_mod = np.fft.rfftfreq(len(centroid_norm), d=1.0 / frame_rate)

        # Flutter band: 0.5–min(200, Nyquist) Hz
        flutter_hi = min(200.0, nyquist_mod * 0.95)
        flutter_mask = (freqs_mod >= 0.5) & (freqs_mod <= flutter_hi)
        total_power = float(np.sum(fft_c[1:]) + 1e-12)
        flutter_power = float(np.sum(fft_c[flutter_mask])) if flutter_mask.any() else 0.0
        flutter_ratio = flutter_power / total_power

        # --- Sub-band analysis: identify dominant flutter source ---
        # Guide roller: 4-8 Hz; Capstan: 2-4 Hz; Tape resonance: 10-30 Hz
        sub_bands = {
            "capstan": (2.0, 4.0),
            "guide_roller": (4.0, 8.0),
            "tape_resonance": (10.0, 30.0),
        }
        dom_source = "unknown"
        dom_power = 0.0
        for src, (lo, hi) in sub_bands.items():
            mask = (freqs_mod >= lo) & (freqs_mod <= min(hi, flutter_hi))
            if mask.any():
                p = float(np.sum(fft_c[mask]))
                if p > dom_power:
                    dom_power = p
                    dom_source = src

        # --- Periodicity check ---
        if flutter_mask.any() and flutter_power > 0:
            dom_idx = np.argmax(fft_c[flutter_mask])
            dom_freq = float(freqs_mod[flutter_mask][dom_idx])
            dom_peak = float(fft_c[flutter_mask][dom_idx])
            periodicity = dom_peak / (flutter_power + 1e-12)
        else:
            dom_freq = 0.0
            periodicity = 0.0

        # Periodic flutter gets a boost (mechanical sources are quasi-periodic)
        periodicity_bonus = 1.0 + 0.4 * max(0.0, periodicity - 0.2)

        # --- Anti-FP: centroid variability in non-flutter range ---
        # Musical content has centroid variation too — check if flutter-band
        # is significantly MORE energetic than the rest
        non_flutter_mask = (freqs_mod >= 0.5) & (freqs_mod <= flutter_hi)
        non_flutter_mask = ~flutter_mask & (freqs_mod > 0.02)
        non_flutter_power = float(np.sum(fft_c[non_flutter_mask])) if non_flutter_mask.any() else 1e-12
        selectivity = flutter_power / (non_flutter_power + 1e-12)
        # If flutter is less than 1.5× the other modulation, probably music
        if selectivity < 1.5:
            flutter_ratio *= 0.3

        threshold = self.thresholds.get(DefectType.FLUTTER, 0.5)
        severity = float(np.clip(flutter_ratio * periodicity_bonus / (threshold * 0.4 + 1e-12), 0.0, 1.0))

        confidence = float(np.clip(0.50 + 0.35 * min(1.0, periodicity), 0.3, 0.90))

        return DefectScore(
            defect_type=DefectType.FLUTTER,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "flutter_energy_ratio": flutter_ratio,
                "dominant_source": dom_source,
                "dominant_mod_freq_hz": dom_freq,
                "periodicity": periodicity,
                "selectivity": selectivity,
                "frame_rate_hz": float(frame_rate),
            },
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

        polarity_inverted = bool(correlation <= -0.9)

        return DefectScore(
            defect_type=DefectType.PHASE_ISSUES,
            severity=severity,
            confidence=0.8,
            locations=[],
            metadata={
                "side_ratio": side_ratio,
                "stereo_correlation": correlation,
                "polarity_inverted": polarity_inverted,
            },
        )

    def _detect_dropouts(self, audio: np.ndarray) -> DefectScore:
        """Detect dropouts (short silence / level dip segments).

        Upgraded v9.10.77b: Multi-indicator dropout detection:
        1. RMS in 5 ms windows with vectorized computation
        2. Material-adaptive threshold: analog 20% median-RMS, digital 10%
        3. Spectral dropout analysis: partial (single-band) vs total (broadband)
        4. Local-context thresholding (sliding median) for dynamic-range-aware detection
        5. Perceptual minimum duration: >= 2.5 ms (1 window)
        6. Severity combines duration fraction + event count + spectral analysis
        """
        # 5 ms windows for finer detection
        window_size = max(1, int(0.005 * self.sample_rate))
        hop_size = window_size // 2

        # Vectorized RMS computation
        n_frames = max(0, (len(audio) - window_size) // hop_size)
        if n_frames < 4:
            return DefectScore(DefectType.DROPOUTS, 0.0, 0.5)

        rms_values = np.array(
            [np.sqrt(np.mean(audio[i * hop_size : i * hop_size + window_size] ** 2)) for i in range(n_frames)],
            dtype=np.float64,
        )

        # Material-adaptive threshold
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
        _threshold_ratio = 0.20 if _mat in _DROPOUT_ANALOG_MATERIALS else 0.10
        median_rms = float(np.median(rms_values))
        global_threshold = _threshold_ratio * median_rms

        # --- Local-context adaptive threshold (sliding median, 1s window) ---
        local_win = max(3, int(1.0 * self.sample_rate / hop_size))  # ~1s in frames
        from scipy.ndimage import median_filter

        local_median = median_filter(rms_values, size=local_win, mode="reflect")
        local_threshold = _threshold_ratio * local_median
        # Combined threshold: stricter of global and local
        combined_threshold = np.maximum(global_threshold, local_threshold)

        dropout_mask = rms_values < combined_threshold

        # Connected-component labelling
        from scipy.ndimage import label

        label_result = label(dropout_mask)
        if isinstance(label_result, tuple):
            labeled_array, num_dropouts = label_result
        else:
            labeled_array = label_result
            num_dropouts = int(np.max(labeled_array))

        locations = []
        total_dropout_s = 0.0
        partial_dropouts = 0
        total_dropouts = 0

        for i in range(1, num_dropouts + 1):
            indices = np.where(labeled_array == i)[0]
            if len(indices) < 1:
                continue
            start_s = indices[0] * hop_size / self.sample_rate
            end_s = (indices[-1] + 1) * hop_size / self.sample_rate
            dur = end_s - start_s
            locations.append((start_s, end_s))
            total_dropout_s += dur

            # --- Spectral dropout classification ---
            # Check if dropout is broadband (total) or partial (single-band loss)
            start_samp = int(indices[0] * hop_size)
            end_samp = min(int((indices[-1] + 1) * hop_size + window_size), len(audio))
            if end_samp - start_samp >= 256:
                seg = audio[start_samp:end_samp]
                spec = np.abs(np.fft.rfft(seg))
                if len(spec) > 8:
                    lo_e = float(np.sum(spec[: len(spec) // 4]))
                    hi_e = float(np.sum(spec[len(spec) // 4 :]))
                    total_e = lo_e + hi_e + 1e-20
                    # If one band has vastly less energy → partial dropout
                    if min(lo_e, hi_e) / total_e < 0.05:
                        partial_dropouts += 1
                    else:
                        total_dropouts += 1
                else:
                    total_dropouts += 1
            else:
                total_dropouts += 1

        # --- Severity ---
        duration = len(audio) / self.sample_rate
        dropout_fraction = total_dropout_s / max(duration, 1e-6)
        # Duration-based: 1% dropout = severity 0.5; 2% = 1.0
        sev_duration = float(np.clip(dropout_fraction / 0.02, 0.0, 1.0))
        # Event density: many short dropouts (tape) rated higher than single long one
        event_rate = len(locations) / max(duration, 1e-6)
        sev_events = float(np.clip(event_rate / 5.0, 0.0, 0.5))  # max 0.5 from events
        # Total dropouts weighted more heavily than partial
        total_ratio = total_dropouts / max(total_dropouts + partial_dropouts, 1)
        severity = float(np.clip(sev_duration * (0.6 + 0.4 * total_ratio) + sev_events * 0.3, 0.0, 1.0))

        confidence = float(np.clip(0.80 + 0.15 * min(1.0, sev_duration), 0.65, 0.95))

        return DefectScore(
            defect_type=DefectType.DROPOUTS,
            severity=severity,
            confidence=confidence,
            locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
            metadata={
                "dropout_count": len(locations),
                "locations_returned": len(locations),
                "dropout_rate": round(event_rate, 3),
                "total_dropout_s": round(total_dropout_s, 4),
                "threshold_ratio": _threshold_ratio,
                "partial_dropouts": partial_dropouts,
                "total_dropouts": total_dropouts,
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
                    for g in groups:
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
            for g in groups:
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
        """Erkennt DC-Offset (Gleichspannungsversatz) inkl. zeitvarianter Drift.

        Upgraded v9.10.77b: Segmentierte Analyse erkennt auch DC-Drift
        (Potentiometer-Alterung, Kondensator-Leckstrom). Absoluter DC-Schwellwert
        zusätzlich zu relativem — leise Aufnahmen werden nicht fehlerkannt.
        """
        if len(audio) == 0:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.0)

        n = len(audio)
        dc_global = float(np.mean(audio))
        peak = float(np.max(np.abs(audio)))

        if peak < 1e-6:
            return DefectScore(DefectType.DC_OFFSET, 0.0, 0.5)

        # --- Global DC ---
        dc_ratio = abs(dc_global) / peak
        # Absolute threshold: DC < 0.001 (-60 dBFS) is inaudible regardless
        abs_dc = abs(dc_global)

        # --- Segmented DC drift detection ---
        n_segs = min(16, max(2, n // self.sample_rate))
        seg_len = n // n_segs
        seg_means = np.array([float(np.mean(audio[i * seg_len : (i + 1) * seg_len])) for i in range(n_segs)])

        # DC drift: max deviation between segments
        dc_drift = float(np.max(seg_means) - np.min(seg_means))
        drift_ratio = dc_drift / (peak + 1e-12)

        # Worst-segment DC
        worst_seg_dc = float(np.max(np.abs(seg_means)))
        worst_seg_ratio = worst_seg_dc / (peak + 1e-12)

        # Combined severity: global + drift + worst segment
        sev_global = dc_ratio / 0.05  # 5% = severity 1.0
        sev_drift = drift_ratio / 0.03  # 3% drift = severity 1.0
        sev_worst = worst_seg_ratio / 0.05
        raw_severity = max(sev_global, sev_drift * 0.8, sev_worst * 0.7)

        # Absolute guard: if DC is < 0.0005 in amplitude, cap severity
        if abs_dc < 0.0005 and worst_seg_dc < 0.001:
            raw_severity = min(raw_severity, 0.15)  # inaudible

        threshold_factor = self.thresholds.get(DefectType.DC_OFFSET, 0.6)
        severity = float(np.clip(raw_severity / threshold_factor, 0.0, 1.0))

        # Confidence: higher if consistent across segments
        seg_std = float(np.std(seg_means))
        confidence = float(
            np.clip(0.80 + 0.15 * min(1.0, abs_dc / 0.01) - 0.10 * min(1.0, seg_std / (abs_dc + 1e-12)), 0.5, 0.98)
        )

        return DefectScore(
            defect_type=DefectType.DC_OFFSET,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "dc_value": dc_global,
                "dc_ratio_percent": dc_ratio * 100,
                "dc_drift_percent": drift_ratio * 100,
                "worst_segment_dc": worst_seg_dc,
                "n_segments": n_segs,
                "dc_dbfs": 20 * np.log10(abs(dc_global)) if abs(dc_global) > 1e-10 else -120.0,
            },
        )

    def _detect_bandwidth_loss(self, audio: np.ndarray) -> DefectScore:
        """Erkennt Hochfrequenz-Verlust / Bandbreitenbegrenzung (HF-Rolloff).

        Upgraded v9.10.77b: Multi-band rolloff detection with -3dB and -6dB
        cutoff estimation. Material-adaptive reference values.
        Genre-aware anti-FP (warm jazz/classical != bandwidth loss).
        """
        if len(audio) < 2048:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)
        if self.sample_rate < 16000:
            return DefectScore(DefectType.BANDWIDTH_LOSS, 0.0, 0.3)

        nperseg = min(8192, len(audio) // 4)
        if nperseg < 512:
            nperseg = 512
        freqs, psd = signal.welch(audio, self.sample_rate, nperseg=nperseg)
        psd_db = 10 * np.log10(psd + 1e-20)

        # --- Material-adaptive HF reference ---
        _mat = getattr(self, "material_type", None)
        _MATERIAL_HF_REF = {
            "shellac": 0.02,
            "wax_cylinder": 0.01,
            "wire_recording": 0.015,
            "lacquer_disc": 0.03,
            "tape": 0.06,
            "reel_tape": 0.07,
            "cassette": 0.05,
            "vinyl": 0.08,
            "cd_digital": 0.10,
            "dat": 0.10,
            "mp3_low": 0.04,
            "mp3_high": 0.08,
            "aac": 0.08,
            "streaming": 0.09,
            "minidisc": 0.06,
        }
        if _mat is None:
            mat_name = ""
        elif isinstance(_mat, Enum):
            mat_name = str(_mat.value)
        else:
            mat_name = str(_mat)
        reference_hf_ratio = _MATERIAL_HF_REF.get(mat_name, 0.08)

        total_energy = float(np.sum(psd) + 1e-12)
        hf_cutoff = min(8000.0, self.sample_rate * 0.45)
        hf_energy = float(np.sum(psd[freqs >= hf_cutoff]))
        hf_ratio = hf_energy / total_energy

        loss_factor = max(0.0, 1.0 - (hf_ratio / (reference_hf_ratio + 1e-12)))

        # --- Estimate -3dB and -6dB rolloff frequencies ---
        # Find the frequency where PSD drops below peak_mid - 3dB / -6dB
        mid_mask = (freqs >= 500) & (freqs <= 4000)
        if mid_mask.any():
            peak_mid_db = float(np.max(psd_db[mid_mask]))
        else:
            peak_mid_db = float(np.max(psd_db))

        rolloff_3db = float(self.sample_rate / 2)
        rolloff_6db = float(self.sample_rate / 2)
        for fi in range(len(freqs) - 1, 0, -1):
            if psd_db[fi] >= peak_mid_db - 3.0:
                rolloff_3db = float(freqs[fi])
                break
        for fi in range(len(freqs) - 1, 0, -1):
            if psd_db[fi] >= peak_mid_db - 6.0:
                rolloff_6db = float(freqs[fi])
                break

        # --- Multi-band energy ratios ---
        bands = [(4000, 8000), (8000, 12000), (12000, 16000), (16000, 22000)]
        band_ratios = {}
        for lo, hi in bands:
            if lo < self.sample_rate / 2:
                hi_clip = min(hi, self.sample_rate * 0.45)
                mask = (freqs >= lo) & (freqs < hi_clip)
                band_ratios[f"{lo // 1000}-{hi // 1000}kHz"] = (
                    float(np.sum(psd[mask]) / total_energy) if mask.any() else 0.0
                )

        # --- Rolloff-based severity (more informative than simple ratio) ---
        # Rolloff at 4 kHz = severe; at 8 kHz = moderate; at 16 kHz = mild
        sev_rolloff = 0.0
        if rolloff_3db < 20000:
            sev_rolloff = float(np.clip((16000 - rolloff_3db) / 12000, 0.0, 1.0))

        # Combined severity: ratio-based + rolloff-based
        raw_severity = 0.5 * loss_factor + 0.5 * sev_rolloff

        # --- Anti-FP: if material naturally has low HF, reduce severity ---
        if mat_name in ("shellac", "wax_cylinder", "wire_recording", "lacquer_disc"):
            # Historic media: low HF is expected, only flag extreme cases
            raw_severity *= 0.5

        threshold_factor = self.thresholds.get(DefectType.BANDWIDTH_LOSS, 0.5)
        severity = float(np.clip(raw_severity / max(threshold_factor, 0.1), 0.0, 1.0))

        # Effective bandwidth: last frequency with > 1% mean PSD
        mean_psd_density = float(np.mean(psd))
        meaningful = np.where(psd > 0.01 * mean_psd_density)[0]
        effective_bw = float(freqs[meaningful[-1]]) if len(meaningful) > 0 else 0.0

        confidence = float(np.clip(0.70 + 0.20 * min(1.0, loss_factor), 0.5, 0.95))

        return DefectScore(
            defect_type=DefectType.BANDWIDTH_LOSS,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "hf_ratio_percent": hf_ratio * 100,
                "effective_bandwidth_hz": effective_bw,
                "rolloff_3db_hz": rolloff_3db,
                "rolloff_6db_hz": rolloff_6db,
                "reference_hf_ratio_percent": reference_hf_ratio * 100,
                "band_ratios": band_ratios,
                "material_ref": mat_name,
            },
        )

    def _detect_pitch_drift(self, audio: np.ndarray) -> DefectScore:
        """Detect constant pitch drift / speed error (≠ WOW/FLUTTER).

        Upgraded v9.10.77b: Multi-segment trend analysis (8+ segments),
        linear regression with monotonicity check, trend significance
        filtering, robust fundamental estimation, anti-FP for key changes.
        """
        min_len = int(5 * self.sample_rate)
        if len(audio) < min_len:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)

        sos = signal.butter(4, 50, btype="high", fs=self.sample_rate, output="sos")
        audio_hp = signal.sosfilt(sos, audio)

        # --- Multi-segment analysis (8 segments minimum) ---
        n_segments = min(16, max(4, len(audio) // (int(3 * self.sample_rate))))
        segment_len = len(audio_hp) // n_segments
        if segment_len < int(2 * self.sample_rate):
            segment_len = int(2 * self.sample_rate)
            n_segments = max(2, len(audio_hp) // segment_len)

        def estimate_fundamental(seg: np.ndarray) -> float:
            """Fundamental frequency via FFT-based autocorrelation."""
            seg = seg / (np.percentile(np.abs(seg), 99.9) + 1e-8)
            min_lag = int(self.sample_rate / 2000)
            max_lag = int(self.sample_rate / 80)
            if max_lag >= len(seg):
                return 0.0
            n_fft = len(seg)
            fft_seg = np.fft.rfft(seg, n=2 * n_fft)
            corr_full = np.fft.irfft(fft_seg * np.conj(fft_seg))
            corr = np.real(corr_full[:n_fft])
            corr = corr[min_lag:max_lag]
            if len(corr) == 0:
                return 0.0
            peak_lag = np.argmax(corr) + min_lag
            if peak_lag == 0:
                return 0.0
            # Confidence check: peak should be significantly above neighbours
            peak_val = corr[peak_lag - min_lag]
            if peak_val < 0.1 * corr[0] if len(corr) > 0 else True:
                return 0.0
            return float(self.sample_rate / peak_lag)

        frequencies = []
        time_centers = []
        for i in range(n_segments):
            start = i * segment_len
            seg = audio_hp[start : start + segment_len]
            if len(seg) < int(1.5 * self.sample_rate):
                continue
            f0 = estimate_fundamental(seg)
            if 60 < f0 < 2000:
                frequencies.append(f0)
                time_centers.append((start + segment_len / 2) / self.sample_rate)

        if len(frequencies) < 3:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)

        frequencies = np.array(frequencies)
        time_centers = np.array(time_centers)

        # --- Linear regression to find monotonic drift trend ---
        # Normalize time to [0, 1]
        t_norm = (time_centers - time_centers[0]) / (time_centers[-1] - time_centers[0] + 1e-6)
        # Linear fit: f(t) = a*t + b
        coeffs = np.polyfit(t_norm, frequencies, 1)
        slope = coeffs[0]  # Hz change over full duration
        intercept = coeffs[1]

        # Predicted values and residuals
        f_predicted = np.polyval(coeffs, t_norm)
        residuals = frequencies - f_predicted
        r_squared = 1.0 - float(np.sum(residuals**2) / (np.sum((frequencies - np.mean(frequencies)) ** 2) + 1e-12))

        # --- Drift in cents ---
        f_start = float(intercept)
        f_end = float(intercept + slope)
        if f_start < 30 or f_end < 30:
            return DefectScore(DefectType.PITCH_DRIFT, 0.0, 0.3)
        drift_cents = abs(1200 * np.log2(max(f_end, f_start) / min(f_end, f_start)))

        # --- Monotonicity check: is drift consistently in one direction? ---
        diffs = np.diff(frequencies)
        if len(diffs) > 0:
            n_positive = np.sum(diffs > 0)
            n_negative = np.sum(diffs < 0)
            monotonicity = abs(n_positive - n_negative) / max(len(diffs), 1)
        else:
            monotonicity = 0.0

        # --- Anti-FP: key changes produce sudden jumps, not gradual drift ---
        max_jump_cents = 0.0
        for i in range(len(frequencies) - 1):
            jump = abs(
                1200 * np.log2(max(frequencies[i + 1], frequencies[i]) / min(frequencies[i + 1], frequencies[i]) + 1e-8)
            )
            max_jump_cents = max(max_jump_cents, jump)
        # If largest single jump is > 50% of total drift → likely key change, not drift
        key_change_penalty = 0.0
        if max_jump_cents > drift_cents * 0.5 and drift_cents > 10:
            key_change_penalty = 0.5

        # --- Combined severity ---
        # 10 cents = mild; 50 cents = moderate; 100 cents = severe
        sev_drift = float(np.clip(drift_cents / 60.0, 0.0, 1.0))
        # R² indicates how well a linear drift model fits (high = true drift)
        sev_fit = float(np.clip(r_squared, 0.0, 1.0))
        # Monotonicity bonus: consistent direction → more likely real drift
        raw_severity = sev_drift * (0.5 + 0.3 * sev_fit + 0.2 * monotonicity) - key_change_penalty

        threshold_factor = self.thresholds.get(DefectType.PITCH_DRIFT, 0.6)
        raw_severity /= max(threshold_factor, 0.1)
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        # Confidence scales with number of segments and fit quality
        confidence = float(np.clip(0.55 + 0.25 * r_squared + 0.10 * monotonicity, 0.40, 0.85))

        return DefectScore(
            defect_type=DefectType.PITCH_DRIFT,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "drift_cents": round(drift_cents, 2),
                "r_squared": round(r_squared, 3),
                "monotonicity": round(monotonicity, 3),
                "slope_hz_per_norm": round(float(slope), 3),
                "max_jump_cents": round(max_jump_cents, 2),
                "n_segments_used": len(frequencies),
                "f_start_hz": round(f_start, 1),
                "f_end_hz": round(f_end, 1),
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

        # --- Asymmetry confirmation (IEC 60094-3 §4.2) ---
        # True print-through is BIDIRECTIONAL but ASYMMETRIC:
        #   alpha_pre ≠ alpha_post due to different magnetisation transfer mechanisms.
        # Equal pre/post → likely reverb bleed or room ambience, not magnetic print-through.
        # Strong asymmetry → confirms genuine magnetic print-through → raise confidence.
        asymmetry_ratio = 0.0
        asymmetry_confidence_bonus = 0.0
        if avg_alpha_pre > 0 and avg_alpha_post > 0:
            asymmetry_ratio = float(max(avg_alpha_pre, avg_alpha_post) / (min(avg_alpha_pre, avg_alpha_post) + 1e-8))
            if asymmetry_ratio > 1.3:
                asymmetry_confidence_bonus = float(np.clip((asymmetry_ratio - 1.3) / 2.0, 0.0, 0.20))
        elif avg_alpha_pre > 0 or avg_alpha_post > 0:
            asymmetry_confidence_bonus = 0.08

        confidence = float(np.clip(0.60 + asymmetry_confidence_bonus, 0.55, 0.80))

        return DefectScore(
            defect_type=DefectType.PRINT_THROUGH,
            severity=severity,
            confidence=confidence,
            locations=locations,
            metadata={
                "pre_echo_events": len(alpha_pre_list),
                "post_echo_events": len(alpha_post_list),
                "avg_alpha_pre": round(avg_alpha_pre, 5),
                "avg_alpha_post": round(avg_alpha_post, 5),
                "asymmetry_ratio": round(asymmetry_ratio, 3),
                "avg_magnitude": total_magnitude / max(print_through_events, 1),
                "onsets_checked": min(len(onset_frames), 30),
            },
        )

    # ------------------------------------------------------------------
    # Detektoren — Runde 3: QUANTIZATION_NOISE, JITTER_ARTIFACTS, DYNAMIC_COMPRESSION_EXCESS
    # ------------------------------------------------------------------

    def _detect_quantization_noise(self, audio: np.ndarray) -> DefectScore:
        """Detect quantization noise via ENOB estimation + spectral flatness.

        Upgraded v9.10.77b: Effective Number of Bits (ENOB) estimation,
        step-size detection in quiet passages, spectral flatness of noise
        floor (quantization noise is spectrally flat), and dynamic SNR
        analysis across passages of varying amplitude.
        """
        audio_norm = audio - np.mean(audio)
        max_amp = float(np.percentile(np.abs(audio_norm), 99.9))
        if max_amp < 1e-8:
            return DefectScore(DefectType.QUANTIZATION_NOISE, 0.0, 0.0)
        audio_norm = audio_norm / max_amp

        # --- 1. Histogram fill ratio (coarse bit-depth indicator) ---
        n_bins = 1024
        hist, _ = np.histogram(audio_norm, bins=n_bins)
        n_populated = int(np.sum(hist > 0))
        fill_ratio = n_populated / float(n_bins)

        # --- 2. ENOB estimation from noise floor ---
        # Quiet passages: < -40 dBFS (0.01 amplitude)
        quiet_mask = np.abs(audio_norm) < 0.01
        enob = 16.0  # assume best case
        granularity = 0.0
        spectral_flatness_quiet = 0.0
        step_size_est = 0.0

        if quiet_mask.sum() > 512:
            quiet_audio = audio_norm[quiet_mask]

            # Granularity: fraction of ±LSB jumps
            diff = np.diff(quiet_audio)
            abs_diff = np.abs(diff)
            nonzero_diffs = abs_diff[abs_diff > 1e-10]

            if len(nonzero_diffs) > 32:
                # Step-size estimation: mode of small differences
                # Quantized signals have recurring step sizes
                diff_hist, diff_edges = np.histogram(nonzero_diffs, bins=200, range=(0, 0.02))
                if diff_hist.max() > 0:
                    mode_idx = np.argmax(diff_hist)
                    step_size_est = float((diff_edges[mode_idx] + diff_edges[mode_idx + 1]) / 2)

                    # ENOB from step size: step ≈ 2/(2^N) → N ≈ log2(2/step)
                    if step_size_est > 1e-8:
                        enob = float(np.clip(np.log2(2.0 / step_size_est), 4.0, 24.0))

                # Granularity: fraction of diffs clustered near the step size
                if step_size_est > 1e-8:
                    near_step = np.abs(abs_diff - step_size_est) < step_size_est * 0.3
                    granularity = float(np.sum(near_step) / max(len(abs_diff), 1))
                else:
                    granularity = float(np.sum(abs_diff > 0.003) / max(len(abs_diff), 1))

            # Spectral flatness of quiet passages (quantization noise → flat spectrum)
            if len(quiet_audio) >= 512:
                spec_q = np.abs(np.fft.rfft(quiet_audio[: min(4096, len(quiet_audio))]))
                spec_q = spec_q[1:]  # remove DC
                if len(spec_q) > 4 and float(np.mean(spec_q)) > 1e-12:
                    geo_mean = float(np.exp(np.mean(np.log(spec_q + 1e-20))))
                    arith_mean = float(np.mean(spec_q))
                    spectral_flatness_quiet = geo_mean / (arith_mean + 1e-12)
                    # Flat spectrum (≈ 1.0) = characteristic of quantization noise

        # --- 3. Dynamic SNR: compare noise in quiet vs loud passages ---
        loud_mask = np.abs(audio_norm) > 0.3
        snr_indicator = 0.0
        if loud_mask.sum() > 256 and quiet_mask.sum() > 256:
            rms_loud = float(np.sqrt(np.mean(audio_norm[loud_mask] ** 2)))
            rms_quiet = float(np.sqrt(np.mean(audio_norm[quiet_mask] ** 2)))
            if rms_quiet > 1e-10:
                dynamic_snr_db = 20.0 * np.log10(rms_loud / rms_quiet)
                # Low ENOB → low dynamic SNR. 16-bit ≈ 96 dB, 8-bit ≈ 48 dB
                snr_indicator = float(np.clip((80.0 - dynamic_snr_db) / 40.0, 0.0, 1.0))

        # --- Combined severity ---
        # ENOB < 12 → noticeable; < 10 → severe; < 8 → extreme
        sev_enob = float(np.clip((14.0 - enob) / 6.0, 0.0, 1.0))
        sev_fill = float(np.clip((0.5 - fill_ratio) * 2.0, 0.0, 1.0))
        sev_flat = spectral_flatness_quiet * granularity  # both high = characteristic

        raw_severity = 0.35 * sev_enob + 0.25 * sev_fill + 0.20 * sev_flat + 0.20 * snr_indicator

        threshold = self.thresholds.get(DefectType.QUANTIZATION_NOISE, 0.6)
        if raw_severity < threshold * 0.4:
            raw_severity = 0.0
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        confidence = float(np.clip(0.60 + 0.25 * min(1.0, sev_enob + granularity), 0.5, 0.90))

        return DefectScore(
            defect_type=DefectType.QUANTIZATION_NOISE,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "fill_ratio": fill_ratio,
                "n_populated_bins": n_populated,
                "enob_estimate": round(enob, 1),
                "step_size_estimate": step_size_est,
                "granularity_index": granularity,
                "spectral_flatness_quiet": round(spectral_flatness_quiet, 3),
                "snr_indicator": round(snr_indicator, 3),
            },
        )

    def _detect_jitter_artifacts(self, audio: np.ndarray) -> DefectScore:
        """Detect jitter artifacts: D/A clock instability → FM sidebands + phase incoherence.

        Upgraded v9.10.77b: Multi-indicator jitter detection:
        1. Zero-crossing regularity deviation (jitter disturbs sample timing)
        2. Instantaneous-frequency variance from analytic signal (phase jitter)
        3. Spectral tone purity degradation (jitter smears pure tones into sidebands)
        4. Segment-wise HF energy variance (temporal inconsistency)
        Material-aware: digital media more susceptible than analog.
        """
        n = len(audio)
        if n < 4096:
            return DefectScore(DefectType.JITTER_ARTIFACTS, 0.0, 0.0)

        # --- 1. Zero-crossing regularity ---
        # Jitter displaces samples → irregular zero-crossing intervals
        zc_indices = np.where(np.diff(np.sign(audio)))[0]
        zc_regularity = 0.0
        if len(zc_indices) > 20:
            zc_intervals = np.diff(zc_indices).astype(float)
            zc_mean = float(np.mean(zc_intervals))
            if zc_mean > 1.0:
                zc_cv = float(np.std(zc_intervals) / zc_mean)  # coefficient of variation
                # Natural music: CV ~0.5–1.5; jitter adds extra irregularity
                zc_regularity = float(np.clip((zc_cv - 1.0) / 1.5, 0.0, 1.0))

        # --- 2. Instantaneous frequency variance (analytic signal) ---
        if_variance = 0.0
        try:
            from scipy.signal import hilbert as _hilbert

            # Use center 32k samples for efficiency
            center = max(0, n // 2 - 16384)
            seg = audio[center : center + min(32768, n)]
            analytic = _hilbert(seg)
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            inst_phase = np.unwrap(np.angle(analytic_arr))
            inst_freq = np.diff(inst_phase) * self.sample_rate / (2 * np.pi)
            # Only consider positive frequencies in plausible range
            valid = (inst_freq > 20) & (inst_freq < self.sample_rate / 2)
            if valid.sum() > 100:
                median_if = float(np.median(inst_freq[valid]))
                if median_if > 50:
                    if_std = float(np.std(inst_freq[valid]))
                    if_variance = float(np.clip(if_std / median_if - 0.05, 0.0, 1.0))
        except (ImportError, ValueError) as _exc:
            logger.debug("Instantaneous frequency variance failed: %s", _exc)

        # --- 3. HF spectral variance across segments ---
        seg_len = min(8192, n // 4)
        n_segs = min(12, n // seg_len)
        hf_var_ratio = 0.0
        sideband_ratio = 0.0
        if n_segs >= 3:
            win = np.hanning(seg_len)
            spectra = []
            for i in range(n_segs):
                s = i * seg_len
                spectrum = np.abs(np.fft.rfft(audio[s : s + seg_len] * win))
                spectra.append(spectrum)

            freqs = np.fft.rfftfreq(seg_len, 1.0 / self.sample_rate)
            hf_mask = freqs > 8000
            if hf_mask.sum() >= 4:
                hf_spectra = np.array([s[hf_mask] for s in spectra])
                hf_mean_per_band = np.mean(hf_spectra, axis=0) + 1e-10
                hf_var_ratio = float(np.mean(np.std(hf_spectra, axis=0) / hf_mean_per_band))

                # Sideband indicator: peak-to-median in HF band
                mean_spectrum = np.mean(spectra, axis=0)
                hf_mean = mean_spectrum[hf_mask]
                sorted_hf = np.sort(hf_mean)[::-1]
                top_n = max(1, len(sorted_hf) // 100)
                sideband_ratio = float(np.mean(sorted_hf[:top_n]) / (np.median(hf_mean) + 1e-10))

        sev_zc = zc_regularity
        sev_if = if_variance
        sev_hf = float(np.clip((hf_var_ratio - 0.15) / 0.5, 0.0, 1.0))
        sev_sb = float(np.clip((sideband_ratio - 3.0) / 10.0, 0.0, 1.0))

        raw_severity = 0.25 * sev_zc + 0.30 * sev_if + 0.25 * sev_hf + 0.20 * sev_sb

        # Material-aware: digital sources more likely to have jitter
        _mat = getattr(self, "material_type", None)
        _DIGITAL_MATS = {"cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        if _mat is None:
            mat_name = ""
        elif isinstance(_mat, Enum):
            mat_name = str(_mat.value)
        else:
            mat_name = str(_mat)
        if mat_name in _DIGITAL_MATS:
            raw_severity *= 1.2  # boost for digital media

        threshold = self.thresholds.get(DefectType.JITTER_ARTIFACTS, 0.6)
        if raw_severity < threshold * 0.4:
            raw_severity = 0.0
        severity = float(np.clip(raw_severity, 0.0, 1.0))

        confidence = float(np.clip(0.55 + 0.25 * min(1.0, sev_if + sev_hf), 0.45, 0.85))

        return DefectScore(
            defect_type=DefectType.JITTER_ARTIFACTS,
            severity=severity,
            confidence=confidence,
            locations=[],
            metadata={
                "zc_regularity": round(zc_regularity, 3),
                "if_variance": round(if_variance, 3),
                "hf_var_ratio": round(hf_var_ratio, 3),
                "sideband_ratio": round(sideband_ratio, 3),
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
        max_amp = np.percentile(np.abs(audio), 99.9)
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
            max_amp = float(np.percentile(np.abs(audio), 99.9))
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
        """Detect SIBILANCE: excessive sibilant energy (harsh 's'/'sh' consonants).

        Upgraded v9.10.77b: Psychoacoustic sharpness model (Zwicker & Fastl),
        spectral centroid in sibilance band for gender-adaptive detection,
        anti-FP for bright instruments (cymbals, hi-hats vs true sibilance),
        vocal-gate via sibilance-to-broadband energy ratio, temporal burst
        pattern analysis (sibilance = short bursts, not sustained brightness).
        """
        n = len(audio)
        if n < 2048:
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

        try:
            nperseg = min(4096, n)
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=nperseg)
            total_power = float(np.sum(psd) + 1e-20)

            # --- Sibilance band analysis ---
            sib_lo, sib_hi = 4000.0, 12000.0
            sib_mask = (freqs >= sib_lo) & (freqs <= min(sib_hi, self.sample_rate * 0.45))
            sib_frac = float(np.sum(psd[sib_mask]) / total_power)

            # Spectral centroid within sibilance band → identifies center frequency
            if sib_mask.any():
                sib_psd = psd[sib_mask]
                sib_freqs = freqs[sib_mask]
                sib_centroid = float(np.sum(sib_freqs * sib_psd) / (np.sum(sib_psd) + 1e-20))
            else:
                sib_centroid = 7000.0

            # --- Anti-FP: distinguish sibilance from general brightness ---
            # True sibilance: concentrated 5–9 kHz bursts. Cymbals: broadband > 8 kHz
            very_hf_mask = freqs > 12000.0
            very_hf_frac = float(np.sum(psd[very_hf_mask]) / total_power) if very_hf_mask.any() else 0.0
            # If very_hf energy is comparable to sibilance → it's broadband brightness, not sibilance
            brightness_ratio = very_hf_frac / max(sib_frac, 1e-6)
            if brightness_ratio > 0.6:
                # Broadband bright signal — reduce sibilance severity
                sib_frac *= 0.4

            # --- Psychoacoustic sharpness approximation (Zwicker model simplified) ---
            # Weight the sibilance band by critical-band rate (Bark scale)
            # Sibilance zone 4–12 kHz ≈ Bark 17–24 → high Bark = high sharpness contribution
            sharpness = 0.0
            if sib_mask.any():
                # Bark-weighted energy
                bark = 13 * np.arctan(0.00076 * sib_freqs) + 3.5 * np.arctan((sib_freqs / 7500) ** 2)
                bark_weight = bark / 24.0  # normalize to [0, 1]
                sharpness = float(np.sum(bark_weight * sib_psd) / (np.sum(psd) + 1e-20))

            # --- Temporal burst pattern (windowed analysis) ---
            win_sib = max(1, int(0.040 * self.sample_rate))  # 40 ms windows
            hop_sib = max(1, win_sib // 2)
            sib_bursts = 0
            n_windows = 0
            max_frac_w = 0.0
            locations: list[tuple[float, float]] = []
            in_event = False
            ev_start = 0.0
            # --- Adaptive burst threshold ---
            # A bright pop recording with lots of air (>8 kHz) has a naturally higher
            # sibilance fraction → the fixed 0.18 would flag it as sibilant.
            # Adapt the threshold upward relative to the signal's mean HF balance:
            # if the signal already has >15% energy in 4–12 kHz, raise the threshold.
            _sib_adapt_base = 0.18
            if total_power > 1e-20:
                _mean_sib_frac = float(np.sum(psd[sib_mask]) / total_power)
                # If the average sibilance fraction is naturally high (bright recording),
                # require a higher burst fraction to flag as sibilance problem.
                _sib_adapt_delta = float(np.clip((_mean_sib_frac - 0.12) * 0.6, 0.0, 0.12))
                _sib_adapt_base = _sib_adapt_base + _sib_adapt_delta
            sib_burst_threshold = _sib_adapt_base

            for i in range(0, n - win_sib, hop_sib):
                chunk = audio[i : i + win_sib]
                cf = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
                cs = np.abs(np.fft.rfft(chunk)) ** 2
                c_total = float(np.sum(cs) + 1e-20)
                c_sib = float(np.sum(cs[(cf >= sib_lo) & (cf <= sib_hi)]))
                frac_w = c_sib / c_total
                max_frac_w = max(max_frac_w, frac_w)
                n_windows += 1

                t_s = i / self.sample_rate
                if frac_w >= sib_burst_threshold:
                    sib_bursts += 1
                    if not in_event:
                        ev_start = t_s
                        in_event = True
                else:
                    if in_event:
                        locations.append((ev_start, t_s + win_sib / self.sample_rate))
                        in_event = False
            if in_event:
                locations.append((ev_start, n / self.sample_rate))

            # Burst ratio: persistent brightness (>60% windows) = instrument, not sibilance
            burst_ratio = sib_bursts / max(n_windows, 1)
            if burst_ratio > 0.60:
                sib_frac *= 0.3  # sustained brightness penalty

            # Peak-window salience avoids long-file dilution of short, harsh bursts.
            sev_peak = float(np.clip((max_frac_w - (sib_burst_threshold + 0.03)) / 0.35, 0.0, 1.0))
            if burst_ratio > 0.60:
                sev_peak *= 0.2

            # --- Combined severity ---
            sev_frac = float(np.clip((sib_frac - 0.10) / 0.22, 0.0, 1.0))
            sev_sharp = float(np.clip((sharpness - 0.08) / 0.15, 0.0, 1.0))
            raw_severity = 0.35 * sev_frac + 0.25 * sev_sharp + 0.15 * burst_ratio + 0.25 * sev_peak

            threshold = self.thresholds.get(DefectType.SIBILANCE, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(np.clip(0.65 + 0.20 * min(1.0, sev_frac), 0.50, 0.88))
            locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

            return DefectScore(
                defect_type=DefectType.SIBILANCE,
                severity=severity,
                confidence=confidence,
                locations=locations,
                metadata={
                    "sibilance_power_fraction": round(sib_frac, 4),
                    "sibilance_centroid_hz": round(sib_centroid, 1),
                    "sharpness_index": round(sharpness, 4),
                    "burst_ratio": round(burst_ratio, 3),
                    "brightness_ratio": round(brightness_ratio, 3),
                },
            )
        except Exception:
            return DefectScore(DefectType.SIBILANCE, 0.0, 0.3)

    def _detect_vocal_harshness(self, audio: np.ndarray) -> DefectScore:
        """Detect VOCAL_HARSHNESS: excessive energy, roughness, or distortion in 2–6 kHz vocal presence zone.

        Multi-indicator harshness detection combining:
        1. Crest factor analysis in harmonic band (low crest = compressed/distorted)
        2. Spectral flux roughness in 2–6 kHz (high flux = harsh transient content)
        3. Odd-harmonic dominance in presence band (clipping signature on vocals)
        4. Peak-to-average ratio in presence zone vs. rest (harshness concentrates energy)

        Psychoacoustic basis: Harshness perception peaks at 2–6 kHz (Zwicker & Fastl 2007,
        "sharpness" metric). Distorted vocals concentrate odd-harmonic energy in this zone,
        creating audible roughness even when flat-top clipping is absent.

        Thresholds calibrated for common vocal defects:
        - Microphone preamp saturation (no flat-tops but audible distortion)
        - Over-compressed vocals (loudness war, low crest factor)
        - Scratchy/harsh recordings (excessive spectral flux in presence band)
        """
        n = len(audio)
        if n < 4096:
            return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.3)

        try:
            # Energy guard: silence/very quiet signals cannot be harsh
            rms_total = float(np.sqrt(np.mean(audio**2)))
            if rms_total < 0.005:
                return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.5)

            # --- Indicator 1: Crest Factor on FULL signal ---
            # Low full-signal crest factor indicates heavy compression/limiting
            # which often accompanies vocal harshness.
            # Computed on full signal (not just presence band) because narrowband
            # crest is misleading for tonal content.
            crest_db = 20.0 * np.log10((float(np.max(np.abs(audio))) + 1e-12) / (rms_total + 1e-12))
            # Very compressed: <5 dB; normal: 8–15 dB
            crest_score = float(np.clip((6.0 - crest_db) / 4.0, 0.0, 1.0))

            # Presence band extraction (used for flux and location detection)
            sos_bp = signal.butter(4, [2000.0, 6000.0], btype="band", fs=self.sample_rate, output="sos")
            presence = signal.sosfilt(sos_bp, audio)
            rms_pres = float(np.sqrt(np.mean(presence**2)) + 1e-12)

            # Guard: if presence band energy is negligible, no harshness possible
            if rms_pres < 0.002:
                return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.5)

            # --- Indicator 2: Spectral flux roughness in 2–6 kHz ---
            # High frame-to-frame spectral variation = harsh/scratchy texture
            hop = max(1, int(0.020 * self.sample_rate))  # 20 ms hop
            win = max(256, int(0.040 * self.sample_rate))  # 40 ms window
            n_frames = max(1, (n - win) // hop)
            n_frames = min(n_frames, 200)  # cap for performance

            freq_lo_bin = int(2000.0 * win / self.sample_rate)
            freq_hi_bin = int(6000.0 * win / self.sample_rate)
            freq_hi_bin = min(freq_hi_bin, win // 2)

            prev_spec = None
            flux_values = []
            for k in range(n_frames):
                start = k * hop
                frame = audio[start : start + win]
                if len(frame) < win:
                    break
                spec = np.abs(np.fft.rfft(frame * np.hanning(win)))
                pres_spec = spec[freq_lo_bin:freq_hi_bin]
                if prev_spec is not None:
                    # Only count positive changes (onset harshness)
                    diff = np.maximum(pres_spec - prev_spec, 0.0)
                    flux = float(np.sum(diff**2))
                    flux_values.append(flux)
                prev_spec = pres_spec

            if flux_values:
                mean_flux = float(np.mean(flux_values))
                # Normalize by total energy to get relative roughness
                total_energy = float(np.mean(audio**2) + 1e-12)
                rel_flux = mean_flux / (total_energy * win)
                flux_score = float(np.clip(rel_flux / 2.0, 0.0, 1.0))
            else:
                flux_score = 0.0

            # --- Indicator 3: Odd-harmonic excess in presence band ---
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
            pres_mask = (freqs >= 2000.0) & (freqs <= 6000.0)
            below_mask = (freqs >= 200.0) & (freqs < 2000.0)
            pres_energy = float(np.sum(psd[pres_mask]) + 1e-20)
            below_energy = float(np.sum(psd[below_mask]) + 1e-20)
            # Harshness: presence band dominates relative to fundamentals
            pres_ratio = pres_energy / below_energy
            # Normal vocals: ratio ~0.02–0.10; harsh: >0.25
            ratio_score = float(np.clip((pres_ratio - 0.10) / 0.30, 0.0, 1.0))

            # --- Indicator 4: Peak energy concentration in presence ---
            total_psd = float(np.sum(psd) + 1e-20)
            pres_fraction = pres_energy / total_psd
            # Normal: ~0.02–0.08; harsh: >0.15
            concentration_score = float(np.clip((pres_fraction - 0.08) / 0.25, 0.0, 1.0))

            # --- Combined severity (weighted) ---
            # Ratio and concentration are the primary discriminators,
            # flux captures temporal roughness, crest is supplementary.
            severity = float(0.15 * crest_score + 0.30 * flux_score + 0.30 * ratio_score + 0.25 * concentration_score)
            severity = float(np.clip(severity, 0.0, 1.0))

            # Apply material threshold
            threshold = self.thresholds.get(DefectType.VOCAL_HARSHNESS, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            # --- Windowed location detection (100 ms windows, 50 ms hop) ---
            locations: list[tuple[float, float]] = []
            if severity > 0.0:
                win_h = max(1, int(0.100 * self.sample_rate))
                hop_h = max(1, win_h // 2)
                harsh_threshold = 0.35
                in_event = False
                ev_start = 0.0
                for i in range(0, n - win_h, hop_h):
                    chunk = audio[i : i + win_h]
                    chunk_pres = signal.sosfilt(sos_bp, chunk)
                    chunk_rms = float(np.sqrt(np.mean(chunk_pres**2)) + 1e-12)
                    chunk_peak = float(np.max(np.abs(chunk_pres)) + 1e-12)
                    chunk_crest = 20.0 * np.log10(chunk_peak / chunk_rms)
                    # Local harshness: low crest + high presence energy
                    chunk_energy_ratio = float(np.mean(chunk_pres**2)) / (float(np.mean(chunk**2)) + 1e-12)
                    local_harsh = (chunk_crest < 7.0) and (chunk_energy_ratio > harsh_threshold)
                    t_s = i / self.sample_rate
                    if local_harsh:
                        if not in_event:
                            ev_start = t_s
                            in_event = True
                    else:
                        if in_event:
                            locations.append((ev_start, t_s + win_h / self.sample_rate))
                            in_event = False
                if in_event:
                    locations.append((ev_start, n / self.sample_rate))
                locations = self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED)

            confidence = 0.70 if severity > 0.3 else 0.55

            return DefectScore(
                defect_type=DefectType.VOCAL_HARSHNESS,
                severity=severity,
                confidence=confidence,
                locations=locations,
                metadata={
                    "crest_factor_db": round(crest_db, 2),
                    "crest_score": round(crest_score, 3),
                    "flux_score": round(flux_score, 3),
                    "presence_ratio_score": round(ratio_score, 3),
                    "presence_concentration_score": round(concentration_score, 3),
                    "presence_fraction": round(pres_fraction, 4),
                },
            )
        except Exception:
            return DefectScore(DefectType.VOCAL_HARSHNESS, 0.0, 0.3)

    def _detect_bias_error(self, audio: np.ndarray) -> DefectScore:
        """Detect BIAS_ERROR: wrong AC-bias level on magnetic tape recording.

        Over-bias → HF rolloff earlier than expected (spectral slope too steep
        above the bias frequency, typically 8–12 kHz on 38 cm/s tape).
        Under-bias → elevated uncorrelated HF noise floor (SNR collapse above 6 kHz).

        Upgraded v9.10.77c: Multi-band spectral slope measurement instead of
        simple 2-band ratio.  Slope is estimated from four logarithmically-spaced
        bands spanning 2–14 kHz; linear regression in dB/octave identifies
        pathological rolloff patterns.  Under-bias confirmed via noise-correlation
        check (uncorrelated segment pairs above 6 kHz).
        Literature: Lindsey & Levy 1978, IEC 60094-1, Ampex/Studer bias specs.
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        # "cassette" is a MediumDetector alias that maps to MaterialType.TAPE="tape" — include both
        tape_materials = {"tape", "reel_tape", "wire_recording", "cassette"}
        if material_name not in tape_materials:
            return DefectScore(
                defect_type=DefectType.BIAS_ERROR,
                severity=0.0,
                confidence=0.85,
                locations=[],
                metadata={"medium_gated": True},
            )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=nperseg)

            # --- Multi-band energy measurement (log-spaced 2–14 kHz) ---
            # Bands: 2-3 kHz, 3-5 kHz, 5-8 kHz, 8-14 kHz
            band_def = [(2000.0, 3000.0), (3000.0, 5000.0), (5000.0, 8000.0), (8000.0, 14000.0)]
            band_centers_oct = [np.log2((lo + hi) / 2.0) for lo, hi in band_def]
            band_energies_db: list[float] = []
            for lo, hi in band_def:
                mask = (freqs >= lo) & (freqs < hi)
                e = float(np.sum(psd[mask]) + 1e-20)
                band_energies_db.append(float(10.0 * np.log10(e)))

            # --- Spectral slope (linear regression in dB/octave) ---
            x = np.array(band_centers_oct)
            y = np.array(band_energies_db)
            slope_db_per_oct = float(np.polyfit(x - x.mean(), y - y.mean(), 1)[0])
            # Well-biased tape: slope ≈ -6 to -12 dB/oct in this range (natural music + tape EQ).
            # Over-bias: slope steeper than -16 dB/oct (HF content suppressed at bias frequency).
            # Under-bias: slope flatter/positive above 6 kHz (unmasked HF noise).
            over_bias_sev = float(np.clip((-slope_db_per_oct - 16.0) / 10.0, 0.0, 1.0))
            under_bias_sev = float(np.clip((slope_db_per_oct + 6.0) / 8.0, 0.0, 1.0))

            # --- Anti-FP: warm jazz / orchestral has naturally steep HF rolloff ---
            # Check if there is ANY musical HF content (>10 kHz), if the 4th band has
            # some content the signal is probably not over-biased at tape level.
            hf_broad_mask = freqs >= 10000.0
            hf_fraction = float(np.sum(psd[hf_broad_mask]) / (np.sum(psd[freqs > 100.0]) + 1e-20))
            # If >10 kHz has >2% of the energy, the source has HF content → over-bias less likely
            if hf_fraction > 0.02:
                over_bias_sev *= max(0.2, 1.0 - (hf_fraction - 0.02) / 0.05)

            # --- Under-bias noise confirmation: segment-to-segment HF variance ---
            # Under-bias: HF noise is uncorrelated between segments → high variance.
            # Music: HF varies but tracks envelope → moderate variance.
            hf_noise_var_score = 0.0
            seg_len_n = min(n, int(0.5 * self.sample_rate))
            n_segs = max(1, min(6, n // seg_len_n))
            if n_segs > 2:
                hf_seg_energies = []
                for si in range(n_segs):
                    s = si * max(1, (n - seg_len_n) // (n_segs - 1)) if n_segs > 1 else 0
                    seg = audio[s : s + seg_len_n]
                    fq, ps = signal.welch(seg, self.sample_rate, nperseg=min(1024, len(seg)))
                    hf_e = float(np.sum(ps[fq >= 6000.0]) + 1e-20)
                    mid_e_seg = float(np.sum(ps[(fq >= 1000.0) & (fq < 4000.0)]) + 1e-20)
                    hf_seg_energies.append(hf_e / mid_e_seg)
                hf_var = float(np.std(hf_seg_energies) / (np.mean(hf_seg_energies) + 1e-8))
                # High HF variance relative to level = uncorrelated noise
                hf_noise_var_score = float(np.clip((hf_var - 0.3) / 0.5, 0.0, 1.0))
            under_bias_sev = float(np.clip(0.5 * under_bias_sev + 0.5 * hf_noise_var_score, 0.0, 1.0))

            severity = float(np.clip(max(over_bias_sev, under_bias_sev), 0.0, 1.0))
            threshold = self.thresholds.get(DefectType.BIAS_ERROR, 0.5)
            if severity < threshold * 0.15:
                severity = 0.0

            bias_mode = "over_bias" if over_bias_sev >= under_bias_sev else "under_bias"
            confidence = float(np.clip(0.62 + 0.15 * min(1.0, abs(slope_db_per_oct) / 8.0), 0.55, 0.82))
            return DefectScore(
                defect_type=DefectType.BIAS_ERROR,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={
                    "bias_direction": bias_mode,
                    "hf_slope": round(slope_db_per_oct, 2),
                    "slope_db_per_oct": round(slope_db_per_oct, 2),
                    "over_bias_sev": round(over_bias_sev, 3),
                    "under_bias_sev": round(under_bias_sev, 3),
                    "bias_mode": bias_mode,
                    "hf_fraction": round(hf_fraction, 4),
                    "hf_noise_variance": round(hf_noise_var_score, 3),
                },
            )
        except Exception:
            return DefectScore(DefectType.BIAS_ERROR, 0.0, 0.3)

    def _detect_dolby_nr_mismatch(self, audio: np.ndarray) -> DefectScore:
        """Detect DOLBY_NR_MISMATCH: Dolby B/C/S encode played back without decoding.

        Symptom: Dolby-encoded tape has a pre-emphasis of +6..+20 dB above 1 kHz
        (Dolby B: ~+6 dB at HF; Dolby C: ~+20 dB; Dolby S: ~+14 dB). When played
        back without the matching expander, high frequencies are severely elevated
        relative to mid frequencies — the inverse of the expected tape-hiss-masking
        curve. This is distinct from HEAD_WEAR (which attenuates HF) and BIAS_ERROR
        (which shifts the energy above the bias frequency).

        Detection (literature: Dolby Laboratories 1968–1995, Nakajima & Odaka 1983):
        - Compute the energy ratio: E(2–16 kHz) / E(300 Hz–2 kHz)
        - In normal music: ratio ≈ 0.1–0.5 (mid / presence / air balanced)
        - Dolby-B mismatch (mild):  ratio > 0.8
        - Dolby-C mismatch (severe): ratio > 1.5
        - Anti-FP: speech/bright sources naturally have high HF — check that
          the HF excess is flat (Dolby-NR shapes a shelf), not peaky (instrument).

        Only triggered on tape materials (cassette/reel_tape).  All digital and
        disc-based materials return severity=0.0 immediately.
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        tape_materials = {"tape", "reel_tape", "wire_recording", "cassette"}
        if material_name not in tape_materials:
            return DefectScore(
                defect_type=DefectType.DOLBY_NR_MISMATCH,
                severity=0.0,
                confidence=0.9,
                locations=[],
                metadata={"medium_gated": True},
            )

        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.DOLBY_NR_MISMATCH, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=nperseg)

            # --- Energy in three bands ---
            def _band_energy(lo: float, hi: float) -> float:
                mask = (freqs >= lo) & (freqs < hi)
                return float(np.sum(psd[mask]) + 1e-20)

            e_mid = _band_energy(300.0, 2000.0)  # Mid: speech fundamentals
            e_presence = _band_energy(2000.0, 8000.0)  # Presence: Dolby-NR pre-emphasis zone
            e_air = _band_energy(8000.0, 16000.0)  # Air: upper Dolby-B/C zone

            hf_to_mid = (e_presence + e_air) / e_mid

            # --- Shelf-flatness check: Dolby pre-emp is a shelf, not a peak ---
            # Divide presence into 4 sub-bands; Dolby-NR raises all uniformly.
            presence_bands = [
                _band_energy(2000.0, 3000.0),
                _band_energy(3000.0, 4500.0),
                _band_energy(4500.0, 6500.0),
                _band_energy(6500.0, 8000.0),
            ]
            pb_norm = np.array(presence_bands) / (np.max(presence_bands) + 1e-20)
            # Shelf: all sub-bands elevated uniformly → low variance, all > 0.3
            shelf_uniformity = float(1.0 - np.std(pb_norm))  # 0..1; high = shelf-like

            # --- Severity mapping ---
            # Dolby-B threshold (mild): hf_to_mid > 0.8
            # Dolby-C threshold (severe): hf_to_mid > 1.5
            if hf_to_mid < 0.8:
                sev = 0.0
            elif hf_to_mid < 1.5:
                # Mild Dolby-B range: 0.0 → 0.55
                sev = float(np.clip((hf_to_mid - 0.8) / 0.7, 0.0, 1.0)) * 0.55
            else:
                # Dolby-C range: 0.55 → 1.0
                sev = 0.55 + float(np.clip((hf_to_mid - 1.5) / 1.5, 0.0, 1.0)) * 0.45

            # Weight by shelf-uniformity (pure tone peaks are not Dolby-NR)
            sev *= float(np.clip(shelf_uniformity, 0.3, 1.0))

            sev = float(np.clip(sev, 0.0, 1.0))
            confidence = 0.65 if sev > 0.05 else 0.85

            # §6.7 Dolby NR type inference from HF-to-mid ratio
            if hf_to_mid >= 1.5:
                _dolby_type = "dolby_c"
            elif hf_to_mid >= 0.8:
                _dolby_type = "dolby_b"
            else:
                _dolby_type = "none"

            return DefectScore(
                defect_type=DefectType.DOLBY_NR_MISMATCH,
                severity=sev,
                confidence=confidence,
                locations=[],
                metadata={
                    "hf_to_mid_ratio": round(hf_to_mid, 3),
                    "shelf_uniformity": round(shelf_uniformity, 3),
                    "dolby_nr_type": _dolby_type,
                },
            )
        except Exception:
            return DefectScore(DefectType.DOLBY_NR_MISMATCH, 0.0, 0.3)

    def _detect_tape_head_level_dips(self, audio: np.ndarray) -> DefectScore:
        """Detect TAPE_HEAD_LEVEL_DIP: gradual envelope dips from head-contact pressure variation.

        Cassette / reel tape transports can exhibit periodic or irregular
        level dips caused by:
        - Capstan irregularity or worn pinch roller (periodic, ~1-3 s cycle)
        - Tape tension variation (non-periodic)
        - Head-tape spacing changes (oxide shedding, wrinkled tape)

        Morphology (observed in real cassette recordings):
        - Gradual onset: 60-100 ms ramp down
        - Minimum: 10-25 dB below context level
        - Sharp recovery: < 25 ms snap back to normal
        - Duration: 100-400 ms per event
        - Rate: 0.3-1.5 events per second

        Detection algorithm:
        1. RMS envelope (20 ms windows, 10 ms hop)
        2. Percentile-75 local reference (500 ms filter)
        3. Connected-component dip labelling (threshold: 3 dB below ref)
        4. Filter: skip genuine silence (< -55 dBFS) and very short events (< 30 ms)
        5. Severity from event rate + mean dip depth

        Scientific basis: Camras (1988) Magnetic Recording Handbook;
        McKnight (1969) 'Tape Reproducer Response Measurements with a
        Reproducer Test Tape'.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr:
            return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.5)

        try:
            # Parameters
            env_win = max(1, int(0.020 * sr))  # 20 ms
            env_hop = max(1, int(0.010 * sr))  # 10 ms
            ref_win_s = 0.500  # 500 ms
            dip_thresh_db = 3.0
            min_dip_frames = 3  # 30 ms minimum

            n_frames = max(0, (n - env_win) // env_hop)
            if n_frames < 10:
                return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.5)

            # Vectorized RMS computation
            rms_env = np.array(
                [np.sqrt(np.mean(audio[i * env_hop : i * env_hop + env_win] ** 2) + 1e-15) for i in range(n_frames)],
                dtype=np.float64,
            )
            rms_db = 20.0 * np.log10(rms_env + 1e-15)

            # Local reference (p75, robust to dips)
            ref_frames = max(3, int(ref_win_s / 0.010))
            if ref_frames % 2 == 0:
                ref_frames += 1
            from scipy.ndimage import percentile_filter

            ref_db = percentile_filter(rms_db, percentile=75, size=ref_frames, mode="reflect")

            # Dip mask
            dip_mask = rms_db < (ref_db - dip_thresh_db)

            # Connected-component labelling
            from scipy.ndimage import label as nd_label

            _label_result: tuple[np.ndarray, int] = nd_label(dip_mask)  # type: ignore[assignment]
            labeled, n_dips_raw = _label_result

            locations = []
            dip_depths = []
            for i in range(1, n_dips_raw + 1):
                frames = np.where(labeled == i)[0]
                if len(frames) < min_dip_frames:
                    continue
                # Skip genuine silence
                if np.mean(rms_db[frames]) < -55.0:
                    continue
                start_s = float(frames[0] * env_hop / sr)
                end_s = float((frames[-1] + 1) * env_hop / sr)
                depth = float(np.max(ref_db[frames] - rms_db[frames]))
                locations.append((start_s, end_s))
                dip_depths.append(depth)

            n_dips = len(locations)
            if n_dips == 0:
                return DefectScore(
                    DefectType.TAPE_HEAD_LEVEL_DIP,
                    0.0,
                    0.85,
                    locations=[],
                    metadata={"dip_count": 0},
                )

            duration_s = n / sr
            event_rate = n_dips / max(duration_s, 1e-6)
            mean_depth = float(np.mean(dip_depths))

            # Severity: event rate + depth
            sev_rate = float(np.clip(event_rate / 2.0, 0.0, 0.6))  # 2/s → sev 0.6
            sev_depth = float(np.clip((mean_depth - 3.0) / 15.0, 0.0, 0.4))  # 18 dB → sev 0.4
            severity = float(np.clip(sev_rate + sev_depth, 0.0, 1.0))

            confidence = float(np.clip(0.70 + 0.20 * min(1.0, severity), 0.70, 0.95))

            # ── Periodicity bonus (capstan-irregularity signature) ──────────
            # Real capstan/pinch-roller dips recur at a stable interval (0.5–3.5 s).
            # Musical dynamics also cause level dips but are NOT periodic.
            # A low coefficient-of-variation in inter-event intervals → confidence bonus.
            is_periodic = False
            median_interval_s = 0.0
            if n_dips >= 3:
                event_centres_s = [(loc[0] + loc[1]) / 2.0 for loc in locations]
                intervals = np.diff(event_centres_s)
                if len(intervals) >= 2:
                    median_interval_s = float(np.median(intervals))
                    cv = float(np.std(intervals)) / max(median_interval_s, 1e-9)
                    # Capstan cycle: 0.5–3.5 s, coefficient-of-variation < 0.35
                    if cv < 0.35 and 0.5 <= median_interval_s <= 3.5:
                        confidence = float(np.clip(confidence + 0.08, 0.70, 0.99))
                        is_periodic = True

            return DefectScore(
                defect_type=DefectType.TAPE_HEAD_LEVEL_DIP,
                severity=severity,
                confidence=confidence,
                locations=self._sample_locations_evenly(locations, self._LOCATION_CAP_UNCAPPED),
                metadata={
                    "dip_count": n_dips,
                    "locations_returned": n_dips,
                    "event_rate_per_s": round(event_rate, 3),
                    "mean_depth_db": round(mean_depth, 2),
                    "max_depth_db": round(float(np.max(dip_depths)), 2) if dip_depths else 0.0,
                    "is_periodic_capstan": is_periodic,
                    "median_interval_s": round(median_interval_s, 3),
                },
            )
        except Exception:
            return DefectScore(DefectType.TAPE_HEAD_LEVEL_DIP, 0.0, 0.3)

    @staticmethod
    def _should_keep_cross_material_tape_head_level_dip(score: "DefectScore") -> bool:
        """Keep strong head-level-dip evidence even when material classification is wrong.

        Real-world archive imports can contain tape transfers mislabelled as vinyl/CD.
        When the dip morphology is strong enough (high event rate, deep dips), hard
        material gating hides a real defect and prevents the Tape Level Stabilizer (Phase 12)
        from ever activating.  Thresholds are deliberately conservative to avoid false
        positives on legitimate amplitude-dynamic music.

        Criteria (all must hold):
          - severity >= 0.12   (not just scanner noise)
          - dip_count >= 2     (at least two periodic dip events)
          - mean_depth_db >= 6.0  (audible head-contact dip depth)
          - event_rate_per_s >= 0.15  (recurrent, not a single transient)
        """
        severity = float(np.clip(score.severity, 0.0, 1.0))
        dip_count = int(score.metadata.get("dip_count", 0))
        mean_depth_db = float(score.metadata.get("mean_depth_db", 0.0))
        event_rate = float(score.metadata.get("event_rate_per_s", 0.0))
        return severity >= 0.12 and dip_count >= 2 and mean_depth_db >= 6.0 and event_rate >= 0.15

    def _detect_riaa_curve_error(self, audio: np.ndarray) -> DefectScore:
        """Detect RIAA_CURVE_ERROR: wrong EQ curve applied during vinyl/disc digitization.

        RIAA not applied → massive bass excess (bass/mid ratio >> 5).
        Double-RIAA or wrong curve applied → extreme bass cut (ratio << 0.1).
        """
        material_name = str(getattr(self.material_type, "value", self.material_type)).lower()
        riaa_materials = {"vinyl", "shellac", "lacquer_disc"}
        if material_name not in riaa_materials:
            # Preserve raw RIAA evidence for diagnostics even when the defect is
            # physically impossible for the current medium and therefore gated.
            _orig_sev = 0.0
            _ratio = 0.0
            _riaa_missing = 0.0
            _riaa_double = 0.0
            try:
                n = len(audio)
                if n >= self.sample_rate:
                    freqs, psd = signal.welch(audio, self.sample_rate, nperseg=min(4096, n))
                    bass_e = float(np.sum(psd[freqs < 300.0]) + 1e-20)
                    mid_e = float(np.sum(psd[(freqs >= 1000.0) & (freqs < 4000.0)]) + 1e-20)
                    _ratio = bass_e / mid_e
                    _riaa_missing = float(np.clip((_ratio - 5.0) / 10.0, 0.0, 1.0))
                    _riaa_double = float(np.clip((0.1 - _ratio) / 0.10, 0.0, 1.0))
                    _orig_sev = float(np.clip(max(_riaa_missing, _riaa_double), 0.0, 1.0))
            except Exception:
                _orig_sev = 0.0

            return DefectScore(
                defect_type=DefectType.RIAA_CURVE_ERROR,
                severity=0.0,
                confidence=0.85,
                locations=[],
                metadata={
                    "medium_gated": True,
                    "original_severity": _orig_sev,
                    "bass_mid_ratio": _ratio,
                    "riaa_missing": _riaa_missing,
                    "riaa_double": _riaa_double,
                    "riaa_missing_score": _riaa_missing,
                    "best_matching_curve": "RIAA",
                },
            )

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
                metadata={
                    "bass_mid_ratio": ratio,
                    "riaa_missing": riaa_missing,
                    "riaa_double": riaa_double,
                    "riaa_missing_score": riaa_missing,
                    "best_matching_curve": "RIAA",
                },
            )
        except Exception:
            return DefectScore(DefectType.RIAA_CURVE_ERROR, 0.0, 0.3)

    def _detect_head_wear(self, audio: np.ndarray) -> DefectScore:
        """Detect HEAD_WEAR: magnetic head degradation causing progressive HF rolloff.

        Worn head → progressive rolloff above the head-gap frequency (typically
        10–16 kHz for 1/4" tape at 38 cm/s).  Characteristic signature:
          - Monotonically decreasing band energies from 4 kHz → 8 kHz → 12 kHz
          - dB/octave slope steeper than -18 dB/oct above 4 kHz
          - Smooth rolloff (no EQ bumps), distinguishable from intentional vintage EQ

        Upgraded v9.10.77c: Multi-band progressive rolloff fit + monotonicity check.
        Anti-FP: vintage/orchestral recordings naturally lack HF; confirmed only when
        rolloff is MONOTONIC and starts from mid range (4 kHz), not just absent HF.
        Literature: Bohn 1987 'Magnetic Recording Head Design', Ampex service manuals.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

        try:
            nperseg = min(8192, n)
            freqs, psd = signal.welch(audio, self.sample_rate, nperseg=nperseg)

            # --- Multi-band energy: 5 bands from 2 kHz to 16 kHz (log-spaced) ---
            hw_bands = [
                (2000.0, 3500.0),  # 0: Reference mid
                (3500.0, 5500.0),  # 1: Upper-mid
                (5500.0, 8000.0),  # 2: Presence
                (8000.0, 12000.0),  # 3: Brilliance
                (12000.0, 16000.0),  # 4: Air (most affected by head wear)
            ]
            band_e_db = []
            for lo, hi in hw_bands:
                mask = (freqs >= lo) & (freqs < min(hi, self.sample_rate * 0.48))
                e = float(np.sum(psd[mask]) + 1e-20)
                band_e_db.append(float(10.0 * np.log10(e)))

            ref_db = band_e_db[0]  # 2–3.5 kHz reference
            relative_db = [e - ref_db for e in band_e_db[1:]]
            # relative_db[0] = upper-mid, [1] = presence, [2] = brilliance, [3] = air

            # --- Monotonicity check: each band should be lower than the previous ---
            # Head wear → strictly monotonic decline.
            # Creative EQ or recording → may have bumps or non-monotonic pattern.
            monotonic_penalties = 0
            for i in range(len(relative_db) - 1):
                if relative_db[i + 1] > relative_db[i] + 2.0:
                    monotonic_penalties += 1
            # Allow 1 minor excursion (mastering EQ bump), but 2+ = not head wear
            monotonicity_ok = monotonic_penalties <= 1

            # --- Spectral slope from 4 kHz to 16 kHz (dB/octave) ---
            slope_bands = hw_bands[1:]
            xs = np.array([np.log2((lo + hi) / 2.0) for lo, hi in slope_bands])
            ys = np.array(band_e_db[1:])
            slope = float(np.polyfit(xs - xs.mean(), ys - ys.mean(), 1)[0])
            # Normal tape: -6 to -14 dB/oct.  Head wear: steeper than -18 dB/oct.
            slope_sev = float(np.clip((-slope - 18.0) / 10.0, 0.0, 1.0))

            # --- Absolute HF level check (band 3 + 4 vs. reference) ---
            mean_hf_relative_db = float(np.mean(relative_db[2:]))  # brilliance + air
            # Worn head: >30 dB below reference causes audible extension loss
            level_sev = float(np.clip((-mean_hf_relative_db - 30.0) / 15.0, 0.0, 1.0))

            severity = 0.0
            if monotonicity_ok:
                severity = float(np.clip(0.55 * slope_sev + 0.45 * level_sev, 0.0, 1.0))
            else:
                # Non-monotonic → likely EQ, not head wear; heavily discount
                severity = float(np.clip(0.15 * slope_sev + 0.10 * level_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.HEAD_WEAR, 0.5)
            if severity < threshold * 0.12:
                severity = 0.0

            confidence = float(np.clip(0.58 + 0.18 * float(monotonicity_ok) - 0.10 * monotonic_penalties, 0.40, 0.80))

            return DefectScore(
                defect_type=DefectType.HEAD_WEAR,
                severity=severity,
                confidence=confidence,
                locations=[],
                metadata={
                    "slope_db_per_oct": round(slope, 2),
                    "slope_severity": round(slope_sev, 3),
                    "level_severity": round(level_sev, 3),
                    "monotonicity_ok": monotonicity_ok,
                    "monotonic_penalties": monotonic_penalties,
                    "relative_db_bands": [round(x, 1) for x in relative_db],
                },
            )
        except Exception:
            return DefectScore(DefectType.HEAD_WEAR, 0.0, 0.3)

    def _detect_transient_smearing(self, audio: np.ndarray) -> DefectScore:
        """Detect TRANSIENT_SMEARING: broadened attack rise-times.

        Upgraded v9.10.77b: Hilbert-envelope for smoother analysis,
        strongest-onset selection (by peak amplitude, not just first N),
        reduced self-smoothing (2ms window), median rise-time for outlier
        robustness, spectral transient sharpness indicator.
        """
        n = len(audio)
        if n < self.sample_rate // 2:
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

        try:
            # --- Hilbert envelope (no self-smoothing bias) ---
            from scipy.signal import hilbert as _hilbert

            hp_sos = signal.butter(
                2, float(np.clip(200.0 / (self.sample_rate / 2.0), 1e-6, 0.999)), btype="high", output="sos"
            )
            hp_audio = signal.sosfilt(hp_sos, audio)

            # Hilbert envelope + light smoothing (2ms instead of 5ms)
            analytic = _hilbert(hp_audio[: min(n, 480000)])  # limit to 10s at 48kHz
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            envelope = np.abs(analytic_arr)
            smooth_win = max(1, int(0.002 * self.sample_rate))
            envelope = np.convolve(envelope, np.ones(smooth_win) / smooth_win, mode="same")
            envelope = np.nan_to_num(envelope, nan=0.0)

            threshold_level = float(np.percentile(envelope, 90))
            if threshold_level < 1e-8:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # --- Find ALL onset candidates ---
            diff_env = np.diff(envelope)
            onset_mask = (envelope[1:] > threshold_level * 0.5) & (diff_env > 0)
            onset_idxs = np.where(onset_mask)[0]
            if len(onset_idxs) == 0:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # --- Select strongest onsets by peak amplitude ---
            # Group nearby onsets (within 50ms) and pick the strongest from each group
            min_gap = int(0.050 * self.sample_rate)
            selected_onsets = []
            last_idx = -min_gap - 1
            group_best = None
            group_best_val = 0.0
            for idx in onset_idxs:
                if idx - last_idx > min_gap:
                    if group_best is not None:
                        selected_onsets.append(group_best)
                    group_best = idx
                    group_best_val = float(envelope[idx])
                else:
                    if float(envelope[idx]) > group_best_val:
                        group_best = idx
                        group_best_val = float(envelope[idx])
                last_idx = idx
            if group_best is not None:
                selected_onsets.append(group_best)

            # Sort by amplitude (strongest first), take up to 40
            selected_onsets.sort(key=lambda x: float(envelope[x]), reverse=True)
            selected_onsets = selected_onsets[:40]

            # --- Measure 10%–90% rise time at each onset ---
            rise_times_ms: list[float] = []
            smear_locations: list[tuple[float, float]] = []
            for idx in selected_onsets:
                peak_val = float(envelope[idx])
                if peak_val < threshold_level * 0.3:
                    continue
                level_10 = peak_val * 0.10
                level_90 = peak_val * 0.90
                window_back = max(0, idx - int(0.040 * self.sample_rate))
                pre_env = envelope[window_back : idx + 1]
                idx_10 = idx_90 = None
                for k in range(len(pre_env)):
                    if pre_env[k] >= level_10 and idx_10 is None:
                        idx_10 = window_back + k
                    if pre_env[k] >= level_90 and idx_90 is None:
                        idx_90 = window_back + k
                        break
                if idx_10 is not None and idx_90 is not None and idx_90 > idx_10:
                    rt = (idx_90 - idx_10) / self.sample_rate * 1000.0
                    if 0.3 < rt < 80.0:
                        rise_times_ms.append(rt)
                        if rt > 6.0:
                            t_s = window_back / self.sample_rate
                            t_e = (idx + int(0.010 * self.sample_rate)) / self.sample_rate
                            smear_locations.append((t_s, t_e))

            if not rise_times_ms:
                return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

            # Use median for robustness against outliers
            median_rt = float(np.median(rise_times_ms))
            mean_rt = float(np.mean(rise_times_ms))
            p90_rt = float(np.percentile(rise_times_ms, 90))

            # --- Spectral transient sharpness: HF-to-LF ratio at onset ---
            spectral_sharpness_values = []
            for idx in selected_onsets[:15]:
                onset_start = max(0, idx - int(0.005 * self.sample_rate))
                onset_end = min(len(hp_audio), idx + int(0.005 * self.sample_rate))
                if onset_end - onset_start >= 256:
                    seg = hp_audio[onset_start:onset_end]
                    spec = np.abs(np.fft.rfft(seg))
                    half = len(spec) // 2
                    if half > 2:
                        hf_e = float(np.sum(spec[half:]))
                        lf_e = float(np.sum(spec[:half]) + 1e-12)
                        spectral_sharpness_values.append(hf_e / lf_e)

            spectral_sharpness = float(np.median(spectral_sharpness_values)) if spectral_sharpness_values else 0.5

            # --- Severity ---
            # Smeared: median rise > 10ms; severe: > 20ms
            sev_rise = float(np.clip((median_rt - 6.0) / 18.0, 0.0, 1.0))
            # Low spectral sharpness at onsets = smeared transients
            sev_spec = float(np.clip(1.0 - spectral_sharpness / 1.5, 0.0, 0.5))
            # Fraction of smeared onsets
            n_smeared = sum(1 for rt in rise_times_ms if rt > 8.0)
            smear_fraction = n_smeared / max(len(rise_times_ms), 1)

            raw_severity = 0.50 * sev_rise + 0.25 * smear_fraction + 0.25 * sev_spec
            threshold = self.thresholds.get(DefectType.TRANSIENT_SMEARING, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(np.clip(0.60 + 0.20 * min(1.0, len(rise_times_ms) / 15.0), 0.50, 0.85))

            return DefectScore(
                defect_type=DefectType.TRANSIENT_SMEARING,
                severity=severity,
                confidence=confidence,
                locations=smear_locations if severity > 0.0 else [],
                metadata={
                    "median_rise_time_ms": round(median_rt, 2),
                    "mean_rise_time_ms": round(mean_rt, 2),
                    "p90_rise_time_ms": round(p90_rt, 2),
                    "n_onsets": len(rise_times_ms),
                    "smear_fraction": round(smear_fraction, 3),
                    "spectral_sharpness": round(spectral_sharpness, 3),
                },
            )
        except Exception:
            return DefectScore(DefectType.TRANSIENT_SMEARING, 0.0, 0.3)

    def _detect_pre_echo(self, audio: np.ndarray) -> DefectScore:
        """Detect PRE_ECHO: energy preceding a major transient.

        Upgraded v9.10.77b: Dual time-scale detection:
        1. Short pre-echo (5–35 ms): codec temporal masking artifacts
        2. Long pre-echo (100–600 ms): tape print-through ghost signals
        Spectral similarity check (does pre-echo share spectrum with transient?),
        material-aware routing (tape→print-through, digital→codec pre-echo),
        robust baseline estimation with percentile-based floor.
        """
        n = len(audio)
        if n < self.sample_rate:
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

        try:
            win = max(1, int(0.008 * self.sample_rate))  # 8ms envelope smoothing
            envelope = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
            envelope = np.nan_to_num(envelope)

            peak_env = float(np.max(envelope))
            if peak_env < 1e-8:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            # Find strong transients (top 10% of envelope, rising edge)
            transient_thresh = float(np.percentile(envelope, 85))
            diff_env = np.diff(envelope)
            transient_idxs = np.where((envelope[1:] > transient_thresh) & (diff_env > transient_thresh * 0.1))[0]

            if len(transient_idxs) == 0:
                return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

            # --- Deduplicate transients (min 100ms apart) ---
            min_gap = int(0.100 * self.sample_rate)
            deduped = [transient_idxs[0]]
            for idx in transient_idxs[1:]:
                if idx - deduped[-1] > min_gap:
                    deduped.append(idx)
            transient_idxs = deduped

            # Material check for analysis routing
            _mat = getattr(self, "material_type", None)
            if _mat is None:
                mat_name = ""
            elif isinstance(_mat, Enum):
                mat_name = str(_mat.value)
            else:
                mat_name = str(_mat)
            _TAPE_MATS = {"tape", "reel_tape", "cassette"}
            is_tape = mat_name in _TAPE_MATS

            # --- Short pre-echo analysis (codec, 5–35 ms before transient) ---
            short_pre_ms = int(0.035 * self.sample_rate)
            short_gap_ms = int(0.003 * self.sample_rate)  # 3ms gap from transient onset
            short_ratios: list[float] = []
            # --- Long pre-echo analysis (print-through, 100–600 ms before transient) ---
            long_pre_start_ms = int(0.600 * self.sample_rate)
            long_pre_end_ms = int(0.100 * self.sample_rate)
            long_ratios: list[float] = []
            spectral_similarities: list[float] = []
            pre_echo_locations: list[tuple[float, float]] = []

            # Noise floor baseline (10th percentile of envelope)
            baseline_energy = float(np.percentile(envelope, 10) ** 2) + 1e-20

            for idx in transient_idxs:
                # -- Short pre-echo (codec) --
                pre_end = max(0, idx - short_gap_ms)
                pre_start = max(0, pre_end - short_pre_ms)
                if pre_end > pre_start + 32:
                    pre_energy = float(np.mean(envelope[pre_start:pre_end] ** 2))
                    ratio_short = pre_energy / baseline_energy
                    if ratio_short > 1.5:
                        short_ratios.append(ratio_short)
                        t_start = pre_start / self.sample_rate
                        t_end = idx / self.sample_rate
                        pre_echo_locations.append((t_start, t_end))

                # -- Long pre-echo (print-through) --
                if is_tape and idx > long_pre_start_ms:
                    lp_start = max(0, idx - long_pre_start_ms)
                    lp_end = max(0, idx - long_pre_end_ms)
                    if lp_end > lp_start + 128:
                        long_pre_energy = float(np.mean(envelope[lp_start:lp_end] ** 2))
                        ratio_long = long_pre_energy / baseline_energy
                        if ratio_long > 1.3:
                            long_ratios.append(ratio_long)

                            # Spectral similarity: does the ghost match the transient?
                            trans_start = idx
                            trans_end = min(n, idx + int(0.030 * self.sample_rate))
                            if trans_end - trans_start >= 256 and lp_end - lp_start >= 256:
                                spec_trans = np.abs(np.fft.rfft(audio[trans_start:trans_end]))
                                spec_ghost = np.abs(np.fft.rfft(audio[lp_start:lp_end][: trans_end - trans_start]))
                                min_len_s = min(len(spec_trans), len(spec_ghost))
                                if min_len_s > 4:
                                    corr = float(np.corrcoef(spec_trans[:min_len_s], spec_ghost[:min_len_s])[0, 1])
                                    if not np.isnan(corr):
                                        spectral_similarities.append(max(0.0, corr))

            # --- Severity calculation ---
            sev_short = 0.0
            if short_ratios:
                mean_short = float(np.mean(short_ratios))
                sev_short = float(np.clip((mean_short - 2.0) / 5.0, 0.0, 1.0))

            sev_long = 0.0
            if long_ratios:
                mean_long = float(np.mean(long_ratios))
                sev_long = float(np.clip((mean_long - 1.5) / 4.0, 0.0, 1.0))
                # Bonus if spectrally similar (= confirmed print-through)
                if spectral_similarities:
                    mean_sim = float(np.mean(spectral_similarities))
                    sev_long *= 0.5 + 0.5 * mean_sim

            # Weighted combination: tape → prioritize long; digital → prioritize short
            if is_tape:
                raw_severity = 0.40 * sev_short + 0.60 * sev_long
            else:
                raw_severity = 0.80 * sev_short + 0.20 * sev_long

            # Event count bonus: many pre-echoes = systematic problem
            n_events = len(short_ratios) + len(long_ratios)
            event_bonus = float(np.clip(n_events / 10.0, 0.0, 0.3))
            raw_severity += event_bonus

            threshold = self.thresholds.get(DefectType.PRE_ECHO, 0.5)
            if raw_severity < threshold * 0.15:
                raw_severity = 0.0
            severity = float(np.clip(raw_severity, 0.0, 1.0))

            confidence = float(np.clip(0.58 + 0.22 * min(1.0, n_events / 8.0), 0.45, 0.85))

            return DefectScore(
                defect_type=DefectType.PRE_ECHO,
                severity=severity,
                confidence=confidence,
                locations=pre_echo_locations if severity > 0.0 else [],
                metadata={
                    "short_pre_echo_mean_ratio": round(float(np.mean(short_ratios)), 3) if short_ratios else 0.0,
                    "long_pre_echo_mean_ratio": round(float(np.mean(long_ratios)), 3) if long_ratios else 0.0,
                    "spectral_similarity": (
                        round(float(np.mean(spectral_similarities)), 3) if spectral_similarities else 0.0
                    ),
                    "n_short_events": len(short_ratios),
                    "n_long_events": len(long_ratios),
                    "is_tape": is_tape,
                },
            )
        except Exception:
            return DefectScore(DefectType.PRE_ECHO, 0.0, 0.3)

    def _detect_aliasing(self, audio: np.ndarray) -> DefectScore:
        """Detect ALIASING: spurious near-Nyquist energy without natural musical source.

        Anti-aliasing filter failure during digitization → elevated spectral floor
        in the 85–97 % Nyquist region that exceeds the expected HF rolloff.
        """
        # Digital sources (cd_digital, dat, mp3_*, aac, streaming, minidisc) have proper
        # anti-aliasing filters by design — aliasing is not physically possible.
        _DIGITAL_MATS = {"cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        _mat = getattr(self, "material_type", None)
        if _mat is not None:
            _mat_name = str(_mat.value) if isinstance(_mat, Enum) else str(_mat)
            if _mat_name in _DIGITAL_MATS:
                return DefectScore(
                    defect_type=DefectType.ALIASING,
                    severity=0.0,
                    confidence=0.9,
                    locations=[],
                    metadata={"medium_gated": True},
                )

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

    # ========== v9.10.98: 12 neue SOTA-Detektoren ==========

    def _detect_modulation_noise(self, audio: np.ndarray) -> DefectScore:
        """Detect signal-dependent modulation noise (tape recording artifact).

        Modulation noise is noise whose level varies proportionally to the signal
        level — fundamentally different from stationary tape hiss.  Present in every
        analog tape recording.

        Algorithm (Esquef & Biscainho 2006):
        1. Compute short-time RMS envelope (10 ms frames)
        2. Compute short-time noise variance estimate (difference of adjacent frames)
        3. Correlate noise variance with signal envelope — high positive correlation
           indicates modulation noise (noise tracks signal level)
        4. Severity derived from correlation coefficient and noise/signal power ratio

        Scientific basis: Esquef & Biscainho (2006), "An Improved Model for
        Tape Modulation Noise"; Czyzewski et al. (2020), "Signal-Dependent Noise
        Models with MMSE Estimation for Audio Restoration".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)
        try:
            # Short-time RMS envelope (10 ms frames, 5 ms hop)
            frame_len = max(1, int(0.010 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 10:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

            # Noise estimate: difference between adjacent frames (stationarity assumption)
            noise_var = np.abs(np.diff(rms_env))
            signal_env = rms_env[:-1]

            # Only consider frames where signal is above noise floor
            signal_threshold = float(np.percentile(rms_env, 20))
            mask = signal_env > signal_threshold
            if np.sum(mask) < 20:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.4)

            # Pearson correlation between signal level and noise variance
            corr = float(np.corrcoef(signal_env[mask], noise_var[mask])[0, 1])
            if np.isnan(corr):
                corr = 0.0

            # Power ratio: noise variance relative to signal
            mean_noise = float(np.mean(noise_var[mask]))
            mean_signal = float(np.mean(signal_env[mask]))
            ratio = mean_noise / (mean_signal + 1e-12)

            # Severity: high correlation AND significant ratio
            raw_sev = float(np.clip(max(0.0, corr) * 1.5, 0.0, 1.0))
            raw_sev *= float(np.clip(ratio * 10.0, 0.3, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.MODULATION_NOISE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.5)
            confidence = float(np.clip(0.5 + corr * 0.3, 0.3, 0.95))
            return DefectScore(
                DefectType.MODULATION_NOISE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"correlation": round(corr, 4), "noise_signal_ratio": round(ratio, 6)},
            )
        except Exception:
            return DefectScore(DefectType.MODULATION_NOISE, 0.0, 0.3)

    def _detect_inner_groove_distortion(self, audio: np.ndarray) -> DefectScore:
        """Detect Inner Groove Distortion (IGD) on vinyl/shellac recordings.

        IGD increases towards the center of the disc due to decreasing linear
        velocity.  Algorithm:
        1. Split audio into 4 equal quarters (simulating disc radius progression)
        2. Measure THD in 2–8 kHz range per quarter
        3. If THD increases monotonically from Q1→Q4 → IGD pattern detected
        4. Severity from THD slope and absolute Q4 distortion level

        Scientific basis: Vinyl stylus tracking geometry; Kates (1981) "A Model of
        Record Tracing Distortion".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 8:
            return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.3)
        try:
            quarter = n // 4
            thd_values = []
            for q in range(4):
                segment = audio[q * quarter : (q + 1) * quarter]
                n_fft = min(4096, len(segment))
                freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
                spec = np.abs(np.fft.rfft(segment[:n_fft]))
                # THD in 2–8 kHz range (where IGD is most audible)
                hf_mask = (freqs >= 2000) & (freqs <= 8000)
                total_mask = freqs <= 8000
                hf_energy = float(np.sum(spec[hf_mask] ** 2))
                total_energy = float(np.sum(spec[total_mask] ** 2)) + 1e-12
                thd = hf_energy / total_energy
                thd_values.append(thd)

            # Check for monotonic increase Q1→Q4 (IGD pattern)
            diffs = [thd_values[i + 1] - thd_values[i] for i in range(3)]
            n_increasing = sum(1 for d in diffs if d > 0)
            slope = (thd_values[3] - thd_values[0]) / (thd_values[0] + 1e-12)

            if n_increasing < 2:
                return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.5)

            raw_sev = float(np.clip(slope * 2.0, 0.0, 1.0))
            # Absolute Q4 THD bonus
            raw_sev += float(np.clip(thd_values[3] * 5.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.INNER_GROOVE_DISTORTION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.5)
            confidence = float(np.clip(0.4 + 0.15 * n_increasing, 0.3, 0.90))
            return DefectScore(
                DefectType.INNER_GROOVE_DISTORTION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"thd_per_quarter": [round(v, 6) for v in thd_values], "slope": round(slope, 4)},
            )
        except Exception:
            return DefectScore(DefectType.INNER_GROOVE_DISTORTION, 0.0, 0.3)

    def _detect_groove_echo(self, audio: np.ndarray) -> DefectScore:
        """Detect groove echo (pre-echo from adjacent groove deformation on vinyl).

        Groove echo occurs ~1.8 s before loud passages (one revolution at 33⅓ rpm).
        Algorithm:
        1. Find strong transients (top 10% envelope peaks)
        2. Search for correlated ghost signal at approximately one revolution delay
        3. Compute spectral similarity between ghost and transient
        4. Severity from ghost-to-noise ratio and spectral match

        DISTINCT from codec pre-echo (PRE_ECHO: 5–35 ms) — groove echo is 1.5–2.2 s.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 5:
            return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)
        try:
            # Revolution delays for different speeds
            delays_s = [1.8, 1.35, 0.77]  # 33⅓, 45, 78 rpm
            delay_samples = [int(d * sr) for d in delays_s]

            win = max(1, int(0.010 * sr))
            envelope = np.convolve(np.abs(audio), np.ones(win) / win, mode="same")
            peak_thresh = float(np.percentile(envelope, 90))
            peaks = np.where(envelope > peak_thresh)[0]
            if len(peaks) == 0:
                return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)

            # Deduplicate peaks (>500 ms apart)
            min_gap = int(0.5 * sr)
            deduped = [peaks[0]]
            for p in peaks[1:]:
                if p - deduped[-1] > min_gap:
                    deduped.append(p)
            peaks = deduped

            best_ratios = []
            best_corrs = []
            echo_locations: list[tuple[float, float]] = []
            for delay_s_val in delay_samples:
                for idx in peaks:
                    ghost_start = max(0, idx - delay_s_val - int(0.05 * sr))
                    ghost_end = max(0, idx - delay_s_val + int(0.05 * sr))
                    if ghost_start >= ghost_end or ghost_end >= n:
                        continue
                    ghost_energy = float(np.mean(envelope[ghost_start:ghost_end] ** 2))
                    baseline = float(np.percentile(envelope, 10) ** 2) + 1e-20
                    ratio = ghost_energy / baseline
                    if ratio > 1.5:
                        best_ratios.append(ratio)
                        echo_locations.append((ghost_start / sr, ghost_end / sr))
                        # Spectral similarity
                        trans_seg = audio[idx : min(n, idx + int(0.03 * sr))]
                        ghost_seg = audio[ghost_start : ghost_start + len(trans_seg)]
                        if len(trans_seg) >= 64 and len(ghost_seg) >= 64:
                            min_l = min(len(trans_seg), len(ghost_seg))
                            s1 = np.abs(np.fft.rfft(trans_seg[:min_l]))
                            s2 = np.abs(np.fft.rfft(ghost_seg[:min_l]))
                            if len(s1) > 4:
                                c = float(np.corrcoef(s1, s2)[0, 1])
                                if not np.isnan(c):
                                    best_corrs.append(max(0.0, c))

            if not best_ratios:
                return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.5)

            mean_ratio = float(np.mean(best_ratios))
            mean_corr = float(np.mean(best_corrs)) if best_corrs else 0.0
            raw_sev = float(np.clip((mean_ratio - 1.5) / 5.0, 0.0, 0.7))
            raw_sev += float(np.clip(mean_corr * 0.3, 0.0, 0.3))
            raw_sev *= float(np.clip(len(best_ratios) / 5.0, 0.5, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.GROOVE_ECHO, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.5)
            confidence = float(np.clip(0.4 + mean_corr * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.GROOVE_ECHO,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=self._sample_locations_evenly(echo_locations, self._LOCATION_CAP_UNCAPPED),
                metadata={
                    "mean_ratio": round(mean_ratio, 4),
                    "spectral_corr": round(mean_corr, 4),
                    "n_echoes": len(best_ratios),
                },
            )
        except Exception:
            return DefectScore(DefectType.GROOVE_ECHO, 0.0, 0.3)

    def _detect_crosstalk(self, audio: np.ndarray) -> DefectScore:
        """Detect channel crosstalk in early stereo recordings.

        Crosstalk appears as ghost images in the stereo field (channel separation < 20 dB).
        Algorithm (BSS-based):
        1. Compute frequency-dependent channel separation L→R and R→L
        2. Low separation (<20 dB) in specific frequency bands = crosstalk
        3. Cross-correlation analysis for delayed crosstalk components

        Scientific basis: Blauert (1997) "Spatial Hearing"; BSS/ICA methods.
        """
        if audio.ndim != 2 or (audio.ndim == 2 and min(audio.shape) < 2):
            return DefectScore(DefectType.CROSSTALK, 0.0, 0.5, metadata={"reason": "mono_or_invalid"})
        try:
            # Ensure [samples, channels] format
            if audio.shape[0] <= 2:
                left, right = audio[0], audio[1]
            else:
                left, right = audio[:, 0], audio[:, 1]

            sr = self.sample_rate
            n_fft = min(4096, len(left))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            # Compute cross-spectral density
            spec_l = np.fft.rfft(left[:n_fft])
            spec_r = np.fft.rfft(right[:n_fft])
            cross_spec = spec_l * np.conj(spec_r)
            auto_l = np.abs(spec_l) ** 2 + 1e-12
            auto_r = np.abs(spec_r) ** 2 + 1e-12

            # Channel separation in bands (250–4000 Hz where crosstalk is most problematic)
            band_mask = (freqs >= 250) & (freqs <= 4000)
            if not band_mask.any():
                return DefectScore(DefectType.CROSSTALK, 0.0, 0.4)

            coherence = np.abs(cross_spec[band_mask]) ** 2 / (auto_l[band_mask] * auto_r[band_mask] + 1e-12)
            mean_coherence = float(np.mean(coherence))

            # Very high coherence in mid-band = poor channel separation = crosstalk
            # Normal stereo: coherence 0.3–0.7; Crosstalk: >0.85
            raw_sev = float(np.clip((mean_coherence - 0.75) / 0.20, 0.0, 1.0))

            # Time-domain cross-correlation peak (delayed crosstalk)
            max_delay = int(0.002 * sr)  # Max 2ms delay for mechanical crosstalk
            xcorr = np.correlate(left[:n_fft], right[:n_fft], mode="full")
            center = len(xcorr) // 2
            side_peak = float(np.max(np.abs(xcorr[center + 1 : center + max_delay + 1])))
            center_peak = float(np.abs(xcorr[center])) + 1e-12
            delay_ratio = side_peak / center_peak
            raw_sev += float(np.clip(delay_ratio * 0.3, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.CROSSTALK, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.CROSSTALK, 0.0, 0.5)
            confidence = float(np.clip(0.4 + mean_coherence * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.CROSSTALK,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"mean_coherence": round(mean_coherence, 4), "delay_ratio": round(delay_ratio, 4)},
            )
        except Exception:
            return DefectScore(DefectType.CROSSTALK, 0.0, 0.3)

    def _detect_intermodulation_distortion(self, audio: np.ndarray) -> DefectScore:
        """Detect intermodulation distortion (IMD) from nonlinear amplifier chains.

        IMD creates sum/difference frequencies (NOT harmonics!) from nonlinear
        signal path.  Algorithm:
        1. Find spectral peaks (strong tonal components)
        2. Check for energy at sum/difference frequencies of peak pairs
        3. IMD products at f1±f2 that are NOT harmonics of either = IMD evidence

        Scientific basis: Volterra series models; SMPTE/DIN IMD measurement standards.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            noise_floor = float(np.percentile(spec_db, 20))

            # Find prominent spectral peaks (>20 dB above noise floor)
            peak_mask = spec_db > (noise_floor + 20.0)
            peak_indices = np.where(peak_mask)[0]
            if len(peak_indices) < 2:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.4)

            # Cluster peaks (within 5 Hz = same component)
            freq_resolution = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            clusters = []
            current_cluster = [peak_indices[0]]
            for idx in peak_indices[1:]:
                if (idx - current_cluster[-1]) * freq_resolution < 5.0:
                    current_cluster.append(idx)
                else:
                    clusters.append(int(np.mean(current_cluster)))
                    current_cluster = [idx]
            clusters.append(int(np.mean(current_cluster)))
            clusters = clusters[:8]  # Limit computation

            # Check for IMD products at f1±f2
            imd_evidence = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    f1 = freqs[clusters[i]]
                    f2 = freqs[clusters[j]]
                    # Sum and difference frequencies
                    for target_f in [abs(f1 - f2), f1 + f2]:
                        if target_f < 50 or target_f > sr / 2 - 100:
                            continue
                        # Is target a harmonic of f1 or f2? If so, skip (THD not IMD)
                        is_harmonic = False
                        for base_f in [f1, f2]:
                            for h in range(1, 8):
                                if abs(target_f - h * base_f) < freq_resolution * 3:
                                    is_harmonic = True
                                    break
                            if is_harmonic:
                                break
                        if is_harmonic:
                            continue
                        # Check for energy at IMD frequency
                        target_idx = int(target_f / freq_resolution)
                        if 0 < target_idx < len(spec_db):
                            imd_level = spec_db[target_idx] - noise_floor
                            if imd_level > 5.0:
                                imd_evidence.append(imd_level)

            if not imd_evidence:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.5)

            mean_imd = float(np.mean(imd_evidence))
            raw_sev = float(np.clip(mean_imd / 30.0, 0.0, 1.0))
            raw_sev *= float(np.clip(len(imd_evidence) / 5.0, 0.5, 1.5))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.INTERMODULATION_DISTORTION, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.5)
            confidence = float(np.clip(0.4 + len(imd_evidence) * 0.05, 0.3, 0.90))
            return DefectScore(
                DefectType.INTERMODULATION_DISTORTION,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"n_imd_products": len(imd_evidence), "mean_imd_level_db": round(mean_imd, 2)},
            )
        except Exception:
            return DefectScore(DefectType.INTERMODULATION_DISTORTION, 0.0, 0.3)

    def _detect_tape_splice_artifact(self, audio: np.ndarray) -> DefectScore:
        """Detect tape splice artifacts (click + level jump + phase discontinuity).

        Splice artifacts are distinct from normal clicks: they combine an impulsive
        transient with a simultaneous level change and often a phase jump.

        Algorithm:
        1. Detect simultaneous level-jump AND impulse (co-occurrence)
        2. Check for phase discontinuity at the event boundary
        3. Level jump persistence (>50 ms) distinguishes splice from click
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)
        try:
            # RMS envelope (10 ms frames)
            frame_len = max(1, int(0.010 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 10:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

            # Level jumps: sudden RMS change > 6 dB in one frame
            rms_db = 20.0 * np.log10(rms_env + 1e-12)
            level_diffs = np.abs(np.diff(rms_db))
            jump_threshold = 6.0  # dB
            jump_indices = np.where(level_diffs > jump_threshold)[0]

            if len(jump_indices) == 0:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)

            # For each jump: check for impulse (high-frequency energy burst)
            splice_events = []
            locations = []
            for ji in jump_indices:
                sample_idx = ji * hop
                if sample_idx < 64 or sample_idx > n - 64:
                    continue
                # High-frequency impulse at junction
                junction = audio[sample_idx - 32 : sample_idx + 32]
                hf_spec = np.abs(np.fft.rfft(junction))
                hf_energy = float(np.sum(hf_spec[len(hf_spec) // 2 :] ** 2))
                total_energy = float(np.sum(hf_spec**2)) + 1e-12
                hf_ratio = hf_energy / total_energy

                # Check level persistence (must persist > 50 ms after jump)
                persist_frames = min(5, n_frames - ji - 1)
                if persist_frames > 2:
                    post_jump_rms = rms_db[ji + 1 : ji + 1 + persist_frames]
                    pre_jump_rms = rms_db[max(0, ji - persist_frames) : ji]
                    if len(post_jump_rms) > 0 and len(pre_jump_rms) > 0:
                        level_diff_persist = abs(float(np.mean(post_jump_rms)) - float(np.mean(pre_jump_rms)))
                        if hf_ratio > 0.2 and level_diff_persist > 3.0:
                            splice_events.append(level_diffs[ji])
                            t = sample_idx / sr
                            locations.append((max(0.0, t - 0.02), t + 0.02))

            if not splice_events:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)

            mean_jump = float(np.mean(splice_events))
            raw_sev = float(np.clip(mean_jump / 20.0, 0.0, 0.7))
            raw_sev += float(np.clip(len(splice_events) / 8.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.TAPE_SPLICE_ARTIFACT, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.5)
            confidence = float(np.clip(0.5 + len(splice_events) * 0.05, 0.3, 0.90))
            return DefectScore(
                DefectType.TAPE_SPLICE_ARTIFACT,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=locations,
                metadata={"n_splices": len(splice_events), "mean_jump_db": round(mean_jump, 2)},
            )
        except Exception:
            return DefectScore(DefectType.TAPE_SPLICE_ARTIFACT, 0.0, 0.3)

    def _detect_hf_remanence_loss(self, audio: np.ndarray) -> DefectScore:
        """Detect high-frequency remanence loss from magnetic tape aging.

        Different from BANDWIDTH_LOSS: HF was originally recorded but has faded
        over decades due to magnetic particle demagnetization.  The spectral envelope
        shows gradual HF roll-off WITH residual harmonic structure (ghost harmonics),
        unlike bandwidth limitation where HF was never present.

        Algorithm:
        1. Compute spectral envelope (smoothed magnitude spectrum)
        2. Detect roll-off rate above 4 kHz
        3. Check for 'ghost harmonics': residual harmonic peaks in rolled-off region
        4. Age-dependent severity model (steeper rolloff + ghost harmonics = aging)

        Scientific basis: Bertram (1994) "Theory of Magnetic Recording";
        Jorgensen (1996) "The Complete Handbook of Magnetic Recording".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            # Spectral envelope roll-off rate above 4 kHz
            hf_mask = (freqs >= 4000) & (freqs <= sr / 2 - 100)
            if not hf_mask.any() or np.sum(hf_mask) < 10:
                return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)

            hf_freqs = freqs[hf_mask]
            hf_db = spec_db[hf_mask]
            float(np.percentile(spec_db, 5))

            # Linear regression for roll-off slope (dB/octave)
            log_freqs = np.log2(hf_freqs + 1.0)
            if len(log_freqs) > 5:
                coeffs = np.polyfit(log_freqs, hf_db, 1)
                slope_db_per_octave = float(coeffs[0])
            else:
                slope_db_per_octave = 0.0

            # Ghost harmonics: peaks that rise above smoothed envelope in HF region
            smooth_kernel = max(5, len(hf_db) // 20)
            smoothed = np.convolve(hf_db, np.ones(smooth_kernel) / smooth_kernel, mode="same")
            residual_peaks = hf_db - smoothed
            n_ghost = int(np.sum(residual_peaks > 3.0))

            # Severity: steep negative slope + ghost harmonics = remanence loss
            raw_sev = float(np.clip(abs(min(0.0, slope_db_per_octave)) / 15.0, 0.0, 0.7))
            if n_ghost > 3:
                raw_sev += float(np.clip(n_ghost / 15.0, 0.0, 0.3))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.HF_REMANENCE_LOSS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.5)
            confidence = float(np.clip(0.4 + 0.1 * min(n_ghost, 5), 0.3, 0.90))
            return DefectScore(
                DefectType.HF_REMANENCE_LOSS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"slope_db_octave": round(slope_db_per_octave, 2), "n_ghost_harmonics": n_ghost},
            )
        except Exception:
            return DefectScore(DefectType.HF_REMANENCE_LOSS, 0.0, 0.3)

    def _detect_stylus_damage(self, audio: np.ndarray) -> DefectScore:
        """Detect stylus damage / mistracking distortion on vinyl/shellac.

        Damaged/worn styli produce asymmetric distortion: odd harmonics dominate
        (unlike soft saturation which produces even harmonics).  Also: consistent
        distortion pattern that tracks groove modulation.

        Algorithm:
        1. Compute short-time distortion (odd vs even harmonic ratio)
        2. Detect persistent asymmetric waveform clipping pattern
        3. Track distortion consistency across the recording

        Scientific basis: Kates (1979) "Turntable Playback Distortion".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)
        try:
            n_fft = min(4096, n)
            spec = np.abs(np.fft.rfft(audio[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            # Find fundamental frequency (strongest peak 80–1000 Hz)
            fund_mask = (freqs >= 80) & (freqs <= 1000)
            if not fund_mask.any():
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)
            fund_idx = int(np.argmax(spec[fund_mask])) + int(np.argmax(fund_mask))
            f0 = freqs[fund_idx]
            if f0 < 50:
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)

            # Odd vs even harmonic energy
            odd_energy = 0.0
            even_energy = 0.0
            for h in range(2, 10):
                h_idx = int(h * f0 / freq_res)
                if h_idx >= len(spec):
                    break
                h_energy = float(spec[max(0, h_idx - 1) : min(len(spec), h_idx + 2)].max() ** 2)
                if h % 2 == 1:
                    odd_energy += h_energy
                else:
                    even_energy += h_energy

            # Asymmetry: stylus damage → odd harmonics dominate
            total_h = odd_energy + even_energy + 1e-12
            odd_ratio = odd_energy / total_h

            # Waveform asymmetry (positive vs negative peak ratio)
            pos_peak = float(np.percentile(audio, 99))
            neg_peak = float(abs(np.percentile(audio, 1)))
            asym = abs(pos_peak - neg_peak) / (max(pos_peak, neg_peak) + 1e-12)

            raw_sev = float(np.clip((odd_ratio - 0.55) * 3.0, 0.0, 0.6))
            raw_sev += float(np.clip(asym * 0.8, 0.0, 0.4))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.STYLUS_DAMAGE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.5)
            confidence = float(np.clip(0.4 + odd_ratio * 0.3, 0.3, 0.90))
            return DefectScore(
                DefectType.STYLUS_DAMAGE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"odd_ratio": round(odd_ratio, 4), "waveform_asymmetry": round(asym, 4)},
            )
        except Exception:
            return DefectScore(DefectType.STYLUS_DAMAGE, 0.0, 0.3)

    def _detect_sticky_shed_residue(self, audio: np.ndarray) -> DefectScore:
        """Detect sticky-shed residue artifacts (post-baking tape degradation).

        After baking, polyester-urethane tapes show: short level dips, modulated
        noise bursts, and specific harmonic distortion patterns.

        Algorithm:
        1. Detect short-duration level dips (10–100 ms, characteristic of shed events)
        2. Measure noise modulation at dip boundaries
        3. Check for periodicity (shed events often correlate with tape wrap period)

        Scientific basis: Hess (2008) "Tape Degradation Factors and
        Predicting Tape Life Expectancy"; Schüller (2005) "Tape Baking".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)
        try:
            # RMS envelope (5 ms frames)
            frame_len = max(1, int(0.005 * sr))
            hop = max(1, frame_len // 2)
            n_frames = max(1, (n - frame_len) // hop)
            if n_frames < 20:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)

            frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(n_frames, frame_len),
                strides=(audio.strides[0] * hop, audio.strides[0]),
            ).copy()
            rms_env = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
            rms_db = 20.0 * np.log10(rms_env + 1e-12)

            # Detect short dips (10–100 ms duration, >4 dB depth)
            median_rms = float(np.median(rms_db))
            dip_threshold = median_rms - 4.0
            in_dip = rms_db < dip_threshold
            dip_events = []
            dip_start = None
            for i, is_dipped in enumerate(in_dip):
                if is_dipped and dip_start is None:
                    dip_start = i
                elif not is_dipped and dip_start is not None:
                    dip_len_ms = (i - dip_start) * hop * 1000.0 / sr
                    if 10.0 <= dip_len_ms <= 100.0:
                        depth = median_rms - float(np.min(rms_db[dip_start:i]))
                        dip_events.append((dip_start, i, depth, dip_len_ms))
                    dip_start = None

            if len(dip_events) < 2:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.5)

            mean_depth = float(np.mean([d[2] for d in dip_events]))
            mean_duration = float(np.mean([d[3] for d in dip_events]))
            event_rate = len(dip_events) / (n / sr)  # events per second

            raw_sev = float(np.clip(event_rate * 2.0, 0.0, 0.5))
            raw_sev += float(np.clip(mean_depth / 15.0, 0.0, 0.3))
            raw_sev += float(np.clip(len(dip_events) / 20.0, 0.0, 0.2))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.STICKY_SHED_RESIDUE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.5)
            locations = [(d[0] * hop / sr, d[1] * hop / sr) for d in dip_events]
            confidence = float(np.clip(0.4 + event_rate * 0.1, 0.3, 0.90))
            return DefectScore(
                DefectType.STICKY_SHED_RESIDUE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                locations=locations,
                metadata={
                    "n_events": len(dip_events),
                    "mean_depth_db": round(mean_depth, 2),
                    "mean_duration_ms": round(mean_duration, 1),
                    "events_per_second": round(event_rate, 3),
                },
            )
        except Exception:
            return DefectScore(DefectType.STICKY_SHED_RESIDUE, 0.0, 0.3)

    def _detect_multiband_wow_flutter(self, audio: np.ndarray) -> DefectScore:
        """Detect frequency-dependent wow/flutter (head gap geometry artifact).

        In some tape machines, wow/flutter varies across the frequency spectrum
        due to head gap geometry and tape contact characteristics.

        Algorithm (Czyzewski 2023):
        1. Split spectrum into 4 octave bands (250, 500, 1000, 2000 Hz centers)
        2. Measure pitch stability independently per band
        3. If pitch instability differs significantly across bands → multiband flutter

        Scientific basis: Czyzewski et al. (2023), "Frequency-Dependent Speed
        Fluctuation Analysis in Audio Tape Recordings".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 3:
            return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.3)
        try:
            band_centers = [250, 500, 1000, 2000]
            band_instabilities = []

            for fc in band_centers:
                # Bandpass filter (±0.5 octave)
                f_low = fc / 1.414
                f_high = min(fc * 1.414, sr / 2 - 1)
                if f_high <= f_low:
                    continue
                sos = signal.butter(4, [f_low, f_high], btype="bandpass", fs=sr, output="sos")
                filtered = signal.sosfilt(sos, audio)

                # Measure pitch instability in this band via zero-crossing rate variance
                hop = max(1, int(0.050 * sr))  # 50 ms frames
                n_f = max(1, (len(filtered) - hop) // hop)
                zcr_per_frame = []
                for i in range(n_f):
                    frame = filtered[i * hop : (i + 1) * hop]
                    zcr = float(np.sum(np.abs(np.diff(np.sign(frame))) > 0)) / len(frame)
                    zcr_per_frame.append(zcr)

                if len(zcr_per_frame) > 5:
                    instability = float(np.std(zcr_per_frame) / (np.mean(zcr_per_frame) + 1e-12))
                    band_instabilities.append(instability)

            if len(band_instabilities) < 3:
                return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.4)

            # Frequency-dependence: coefficient of variation across bands
            mean_inst = float(np.mean(band_instabilities))
            cv = float(np.std(band_instabilities) / (mean_inst + 1e-12))

            # High CV = different instability per band = multiband flutter
            raw_sev = float(np.clip(cv * 2.0, 0.0, 0.6))
            raw_sev += float(np.clip(mean_inst * 3.0, 0.0, 0.4))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.MULTIBAND_WOW_FLUTTER, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.5)
            confidence = float(np.clip(0.4 + cv * 0.2, 0.3, 0.90))
            return DefectScore(
                DefectType.MULTIBAND_WOW_FLUTTER,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "band_instabilities": [round(b, 4) for b in band_instabilities],
                    "cv_across_bands": round(cv, 4),
                },
            )
        except Exception:
            return DefectScore(DefectType.MULTIBAND_WOW_FLUTTER, 0.0, 0.3)

    def _detect_generation_loss(self, audio: np.ndarray) -> DefectScore:
        """Detect cumulative generation loss from tape dubbing / transcoding.

        Each copy generation adds noise, bandwidth loss, and phase distortion with
        a specific cumulative signature different from individual defects.

        Algorithm:
        1. Measure noise floor shape (generation loss has uniform elevation)
        2. Detect cumulative bandwidth narrowing (steeper than single-source rolloff)
        3. Check spectral coherence degradation (phase randomization)
        4. Combine into multi-generational degradation model

        Scientific basis: McKnight & Giddings (1981) AES; Cabot (1977)
        "Generation Loss in Magnetic Tape Recording".
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)

            # 1. Noise floor uniformity (multi-gen: elevated and flat)
            noise_floor_db = float(np.percentile(spec_db, 10))
            lf_floor = float(np.percentile(spec_db[(freqs >= 100) & (freqs <= 500)], 10))
            hf_floor = float(np.percentile(spec_db[(freqs >= 2000) & (freqs <= 8000)], 10))
            floor_uniforms = abs(lf_floor - hf_floor)  # Small = uniform = multi-gen

            # 2. Bandwidth: effective -20 dB rolloff point
            peak_db = float(np.max(spec_db))
            bw_mask = spec_db > (peak_db - 20.0)
            if bw_mask.any():
                effective_bw = float(freqs[bw_mask][-1])
            else:
                effective_bw = float(freqs[-1])

            # 3. Phase coherence (spectral flatness as proxy)
            spectral_flatness = float(np.exp(np.mean(np.log(spec + 1e-20))) / (np.mean(spec) + 1e-20))

            # Severity model: high noise floor + narrow bandwidth + phase degradation
            noise_sev = float(np.clip((noise_floor_db + 40.0) / 30.0, 0.0, 0.4))
            bw_sev = float(np.clip((12000.0 - effective_bw) / 10000.0, 0.0, 0.3))
            phase_sev = float(np.clip(spectral_flatness * 2.0, 0.0, 0.3))
            raw_sev = noise_sev + bw_sev + phase_sev
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.GENERATION_LOSS, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.5)
            confidence = float(np.clip(0.35 + raw_sev * 0.3, 0.3, 0.85))
            return DefectScore(
                DefectType.GENERATION_LOSS,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={
                    "noise_floor_db": round(noise_floor_db, 2),
                    "effective_bw_hz": round(effective_bw, 0),
                    "spectral_flatness": round(spectral_flatness, 4),
                    "floor_uniformity": round(floor_uniforms, 2),
                },
            )
        except Exception:
            return DefectScore(DefectType.GENERATION_LOSS, 0.0, 0.3)

    def _detect_motor_interference(self, audio: np.ndarray) -> DefectScore:
        """Detect turntable/tape motor interference (harmonics 80–300 Hz).

        Different from LOW_FREQ_RUMBLE (sub-bass <80 Hz) — motor interference
        produces harmonics in the 80–300 Hz range from DC and synchronous motors.

        Algorithm:
        1. Analyze spectral peaks at motor-related frequencies (50, 60, 80, 100, 120 Hz etc.)
        2. Distinguish from electrical hum (motor harmonics are broader, less sharp)
        3. Check modulation pattern (motor speed variation → FM sidebands)

        Scientific basis: Tremaine (1969) "Audio Cyclopedia"; Fletcher & Rossing.
        """
        n = len(audio)
        sr = self.sample_rate
        if n < sr * 2:
            return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.3)
        try:
            n_fft = min(8192, n)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            spec = np.abs(np.fft.rfft(audio[:n_fft])) ** 2
            spec_db = 10.0 * np.log10(spec + 1e-20)
            float(np.percentile(spec_db, 20))
            freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

            # Motor frequencies: 33⅓ rpm = 0.556 Hz, 45 rpm = 0.75 Hz, 78 rpm = 1.3 Hz
            # Higher harmonics land in 80–300 Hz range
            motor_freqs = []
            for base in [0.556, 0.75, 1.3]:
                for h in range(50, 400):
                    f = base * h
                    if 60 < f < 350:
                        motor_freqs.append(f)

            # Combine with generic motor harmonic series
            motor_freqs.extend([80, 100, 120, 133, 150, 160, 180, 200, 240, 250, 267, 300])
            motor_freqs = sorted({int(f) for f in motor_freqs if 60 < f < 350})

            # Check for peaks at motor frequencies (broad peaks, not sharp peaks like hum)
            motor_peaks = []
            for f in motor_freqs:
                idx = int(f / freq_res)
                if 0 < idx < len(spec_db) - 2:
                    # Broad peak: average over ±3 bins
                    local_level = float(np.mean(spec_db[max(0, idx - 3) : idx + 4]))
                    # Compare with surrounding 50-bin region
                    surround = spec_db[max(0, idx - 50) : min(len(spec_db), idx + 50)]
                    local_median = float(np.median(surround))
                    prominence = local_level - local_median
                    if prominence > 3.0:  # Broader threshold than hum (which uses sharper peaks)
                        motor_peaks.append(prominence)

            if len(motor_peaks) < 3:
                return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.5)

            mean_prominence = float(np.mean(motor_peaks))
            raw_sev = float(np.clip(mean_prominence / 15.0, 0.0, 0.6))
            raw_sev += float(np.clip(len(motor_peaks) / 15.0, 0.0, 0.4))
            raw_sev = float(np.clip(raw_sev, 0.0, 1.0))

            threshold = self.thresholds.get(DefectType.MOTOR_INTERFERENCE, 0.5)
            if raw_sev < threshold * 0.1:
                return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.5)
            confidence = float(np.clip(0.4 + len(motor_peaks) * 0.03, 0.3, 0.90))
            return DefectScore(
                DefectType.MOTOR_INTERFERENCE,
                float(np.clip(raw_sev, 0.0, 1.0)),
                confidence,
                metadata={"n_motor_peaks": len(motor_peaks), "mean_prominence_db": round(mean_prominence, 2)},
            )
        except Exception:
            return DefectScore(DefectType.MOTOR_INTERFERENCE, 0.0, 0.3)


# ========== Singleton ==========

_defect_scanner_instance: "DefectScanner | None" = None
_defect_scanner_lock = threading.Lock()


def get_defect_scanner(sample_rate: int = 48000) -> "DefectScanner":
    """Return the module-level DefectScanner singleton (thread-safe, double-checked locking)."""
    global _defect_scanner_instance
    if _defect_scanner_instance is None:
        with _defect_scanner_lock:
            if _defect_scanner_instance is None:
                _defect_scanner_instance = DefectScanner(sample_rate=sample_rate)
    return _defect_scanner_instance


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

    logger.debug("\n%s", "=" * 60)
    logger.debug("DEFECT SCAN RESULTS")
    logger.debug("%s", "=" * 60)
    logger.debug("Material: %s", result.material_type.value)
    logger.debug("Duration: %.1fs", result.duration_seconds)
    logger.debug(
        f"Analysis Time: {result.analysis_time_seconds:.3f}s ({result.analysis_time_seconds / result.duration_seconds * 100:.1f}% overhead)"
    )
    logger.debug("\nTop 5 Defects:")
    for i, score in enumerate(result.get_top_defects(5), 1):
        logger.debug("  %s. %s", i, score)

    logger.debug("\nTotal Severity: %.3f", result.get_total_severity())
