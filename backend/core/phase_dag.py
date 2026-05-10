"""
Phase DAG — §7.5a [RELEASE_MUST]
=================================

Formaler Abhängigkeitsgraph der Aurik-Pipeline-Phasen.
Definiert HARD_BEFORE-Constraints, INDEPENDENT-Gruppen und CONFLICT-Paare.

Spec: 06_phases_system.md §7.5a (v9.12.0)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseConstraint:
    """A muss vor B ausgeführt worden sein (HARD_BEFORE)."""

    before: str  # Phase-ID die zuerst laufen muss
    after: str  # Phase-ID die danach läuft
    reason: str  # Normative Begründung


@dataclass(frozen=True)
class ConflictPair:
    """Zwei Phasen dürfen nicht gleichzeitig aktiv sein."""

    phase_a: str
    phase_b: str
    reason: str


# ---------------------------------------------------------------------------
# Normative HARD_BEFORE-Constraints (§7.5a)
# ---------------------------------------------------------------------------

HARD_BEFORE_CONSTRAINTS: list[PhaseConstraint] = [
    # phase_01 muss IMMER zuerst laufen (DC-Offset → alle anderen)
    PhaseConstraint("phase_01_click_removal", "phase_03_denoise", "DC/Click vor ML-NR"),
    PhaseConstraint("phase_01_click_removal", "phase_06_frequency_restoration", "DC-Offset vor BW-Extension"),
    PhaseConstraint("phase_01_click_removal", "phase_07_harmonic_restoration", "DC vor Harmonik"),
    PhaseConstraint("phase_01_click_removal", "phase_09_crackle_removal", "DC vor Crackle"),
    PhaseConstraint("phase_01_click_removal", "phase_12_wow_flutter_fix", "DC vor Wow/Flutter"),
    PhaseConstraint("phase_01_click_removal", "phase_18_noise_gate", "DC vor Noise-Gate"),
    PhaseConstraint("phase_01_click_removal", "phase_29_tape_hiss_reduction", "DC vor Band-NR"),
    # NR vor Harmonik (Phase_03 → Phase_06/07)
    PhaseConstraint(
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "NR vor BW-Extension (kein Rauschen in Extended-Harmonics)",
    ),
    PhaseConstraint(
        "phase_03_denoise", "phase_07_harmonic_restoration", "NR vor Harmonik-Enhancement (§2.46 Stufe 4 vor 5)"
    ),
    # Carrier-Chain-Reihenfolge (§2.46 Stufe 4 → Stufe 5)
    PhaseConstraint(
        "phase_29_tape_hiss_reduction",
        "phase_07_harmonic_restoration",
        "Band-NR vor Harmonik (Rauschen nicht als Harmonik rekonstruieren)",
    ),
    PhaseConstraint(
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
        "BW-Extension vor Harmonik (Stufe 5 intern geordnet)",
    ),
    # Crackle vor Noise-Gate (Phase_09 → Phase_18)
    PhaseConstraint(
        "phase_09_crackle_removal",
        "phase_18_noise_gate",
        "Crackle vor NR/Gate (Crackle-Residuen nicht als Rauschen gaten)",
    ),
    # Wow/Flutter vor Azimuth (Phase_12 → Phase_25)
    PhaseConstraint(
        "phase_12_wow_flutter_fix",
        "phase_25_azimuth_correction",
        "Wow/Flutter-Korrektur vor Azimuth (Zeit-Alignement zuerst)",
    ),
    # Dropout vor BW-Extension (Phase_24 → Phase_06)
    PhaseConstraint(
        "phase_24_dropout_repair",
        "phase_06_frequency_restoration",
        "Dropout-Reparatur vor BW-Extension (keine Lücken in erweitertem Spektrum)",
    ),
    # ADC-Artefakt-Reihenfolge: die frühere Quantisierungs-NR-Phase existiert in UV3 nicht mehr.
    # Daher hier kein phase_31-Constraint mehr — sonst entstehen False-Positives mit
    # phase_31_speed_pitch_correction, das semantisch ein anderer Verarbeitungsschritt ist.
]

# Kurzform-Map: "phase_XX" → Vollname (für validate_phase_order)
_SHORT_TO_FULL: dict[str, str] = {
    "phase_01": "phase_01_click_removal",
    "phase_03": "phase_03_denoise",
    "phase_06": "phase_06_frequency_restoration",
    "phase_07": "phase_07_harmonic_restoration",
    "phase_09": "phase_09_crackle_removal",
    "phase_12": "phase_12_wow_flutter_fix",
    "phase_18": "phase_18_noise_gate",
    "phase_24": "phase_24_dropout_repair",
    "phase_25": "phase_25_azimuth_correction",
    "phase_29": "phase_29_tape_hiss_reduction",
    "phase_30": "phase_30_dc_offset_removal",
}


# ---------------------------------------------------------------------------
# CONFLICT-Paare (aus §2.29e CONFLICT_REGISTRY)
# ---------------------------------------------------------------------------

CONFLICT_PAIRS: list[ConflictPair] = [
    ConflictPair(
        "phase_06_bw_extension",
        "phase_23_audio_sr_upsampling",
        "Beide erweitern Bandbreite — nur eine aktiv (Phase_06 hat Vorrang bei Tape/Shellac)",
    ),
    ConflictPair(
        "phase_21_exciter",
        "phase_07_harmonic_enhancement",
        "§0a: phase_21 (Exciter) ist in Restoration VERBOTEN — nie gleichzeitig mit phase_07",
    ),
]


# ---------------------------------------------------------------------------
# Parallelisierungs-Klassen (§7.5a)
# ---------------------------------------------------------------------------

INDEPENDENT_CLASS_A = frozenset(
    {
        "phase_14_stereo_width",
        "phase_15_stereo_field_repair",
        "phase_25_azimuth_correction",
    }
)
# Klasse A: Stereo/Phase — parallel ausführbar nach phase_12.

INDEPENDENT_CLASS_B = frozenset(
    {
        "phase_09_crackle_removal",
        "phase_24_dropout_repair",
    }
)
# Klasse B: Lokale Defekte — parallel nach phase_01; Defekttypen überlappen nicht.

INDEPENDENT_CLASS_C = frozenset(
    {
        "phase_05_hum_removal",
        "phase_11_spectral_repair",
    }
)
# Klasse C: Analyse/leichtgewichtig — kann parallel zu anderen laufen.


# ---------------------------------------------------------------------------
# Validierungs-API (§7.5a)
# ---------------------------------------------------------------------------


def _normalize_phase_id(phase_id: str) -> str:
    """Normiert Kurzform (phase_03) auf Langform (phase_03_denoise) wenn möglich."""
    phase_id = phase_id.strip().lower()
    # Bereits in Kurzform ohne Suffix?
    for short, full in _SHORT_TO_FULL.items():
        if phase_id == short:
            return full
        if phase_id.startswith(short + "_"):
            return phase_id  # bereits Langform
    return phase_id


def validate_phase_order(phase_list: list[str]) -> list[str]:
    """Prüft eine geordnete Phase-Liste gegen HARD_BEFORE-Constraints.

    Args:
        phase_list: Geordnete Liste von Phase-IDs (kurz- oder langform).

    Returns:
        Liste von Constraint-Verletzungen (leer = korrekt).
        Format: "phase_07_harmonic_enhancement kommt vor phase_03_denoise (NR vor Harmonik)"
    """
    normalized = [_normalize_phase_id(p) for p in phase_list]
    violations = []

    for constraint in HARD_BEFORE_CONSTRAINTS:
        b = _normalize_phase_id(constraint.before)
        a = _normalize_phase_id(constraint.after)

        # Nur prüfen wenn beide in der Liste sind
        b_indices = [i for i, p in enumerate(normalized) if p == b]
        a_indices = [i for i, p in enumerate(normalized) if p == a]

        if not b_indices or not a_indices:
            continue  # eine der Phasen nicht aktiv → Constraint irrelevant

        b_idx = min(b_indices)
        a_idx = min(a_indices)
        if a_idx < b_idx:
            violations.append(f"{a} kommt vor {b} — Verletzung: {constraint.reason}")

    return violations


def check_conflict(phase_a: str, phase_b: str) -> str | None:
    """Prüft ob zwei Phasen im Konflikt stehen.

    Returns:
        Konflikt-Beschreibung wenn vorhanden, sonst None.
    """
    a = _normalize_phase_id(phase_a)
    b = _normalize_phase_id(phase_b)

    for pair in CONFLICT_PAIRS:
        pa = _normalize_phase_id(pair.phase_a)
        pb = _normalize_phase_id(pair.phase_b)
        if (a.startswith(pa[:8]) and b.startswith(pb[:8])) or (a.startswith(pb[:8]) and b.startswith(pa[:8])):
            return pair.reason

    return None


def get_parallel_class(phase_id: str) -> str | None:
    """Gibt die Parallelisierungs-Klasse zurück ('A', 'B', 'C') oder None."""
    p = _normalize_phase_id(phase_id)
    for cls_name, cls_set in [("A", INDEPENDENT_CLASS_A), ("B", INDEPENDENT_CLASS_B), ("C", INDEPENDENT_CLASS_C)]:
        for member in cls_set:
            if p.startswith(member[:11]):  # "phase_XX_" prefix
                return cls_name
    return None
