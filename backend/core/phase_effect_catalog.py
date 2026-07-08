"""§2.60 Denker-Intelligenz: PhaseEffectProfile — Wissen was Phasen bewirken.

Damit der PhaseInteractionDenker Intensitäten PROAKTIV kalibrieren kann
(statt nur PMGG-rollback REACTIV), braucht er ein Modell jeder Phase:
  - Welche Musical Goals werden beeinflusst?
  - In welche Richtung? (boost/dampen)
  - Wie stark ist der typische Effekt?
  - Welche Risiken gibt es? (vocal_distortion, transient_smearing, etc.)
  - Welche Vorbedingungen braucht die Phase?

Die Profile werden vom Denker mit dem aktuellen Audio-Zustand (SNR, Panns,
Bandbreite, Defekte) kombiniert und ergeben eine kalibrierte Intensität.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseEffectProfile:
    """Beschreibt die Wirkung einer Phase auf Musical Goals und ihre Risiken."""

    phase_id: str
    # Goal-Impact: goal_name → typical_delta (positiv = verbessert, negativ = verschlechtert)
    goal_impact: dict[str, float] = field(default_factory=dict)
    # Risiko-Typen (wenn diese Bedingungen vorliegen, Intensität reduzieren)
    risks: list[str] = field(default_factory=list)
    # Vorbedingungen für optimale Wirkung
    preconditions: dict[str, Any] = field(default_factory=dict)
    # Maximal sichere Stärke (0-1) für verschiedene Materialien
    max_strength_by_material: dict[str, float] = field(default_factory=dict)
    # Zeitaufwand-Kategorie: "fast" (<5s), "medium" (5-30s), "slow" (30-120s), "heavy" (>120s)
    time_profile: str = "medium"
    # Minimale Defekt-Schwere damit Phase sinnvoll ist (0-1)
    min_severity: float = 0.0
    # Kommentar für Debugging
    note: str = ""


# ── §2.60 Phase-Wissensbasis ──────────────────────────────────────────────

PHASE_EFFECT_CATALOG: dict[str, PhaseEffectProfile] = {
    # ── Defekt-Entfernung ──────────────────────────────────────────
    "phase_01_click_removal": PhaseEffectProfile(
        phase_id="phase_01_click_removal",
        goal_impact={
            "transparenz": +0.03,
            "artikulation": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["transient_smearing"],  # Zu aggressiv → Ansätze verschmiert
        preconditions={"click_density": "> 100/s"},
        max_strength_by_material={"shellac": 1.0, "vinyl": 0.9, "tape": 0.6, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.2,
        note="Median-Filter; bei zu hoher Stärke werden Transienten verschmiert",
    ),
    "phase_02_hum_removal": PhaseEffectProfile(
        phase_id="phase_02_hum_removal",
        goal_impact={
            "transparenz": +0.02,
            "natuerlichkeit": +0.02,
            "waerme": -0.01,  # Notch kann Wärme minimal reduzieren
        },
        risks=["bass_loss"],
        preconditions={"hum_energy_db": "> -50"},
        max_strength_by_material={"vinyl": 1.0, "tape": 0.8, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.1,
        note="IIR-Notch 50/60Hz+Harmonische; sehr gezielt, kaum Kollateralschaden",
    ),
    # ── Rauschunterdrückung ──────────────────────────────────────
    "phase_03_denoise": PhaseEffectProfile(
        phase_id="phase_03_denoise",
        goal_impact={
            "transparenz": +0.06,
            "artikulation": +0.04,
            "waerme": -0.03,  # ML kann Wärme reduzieren
            "natuerlichkeit": -0.02,  # ML kann künstlich klingen
            "emotionalitaet": -0.02,
        },
        risks=["vocal_distortion", "ml_artifact", "energy_loss"],
        preconditions={"snr_db": "< 20", "bypass_if": "snr_unknown AND vocal_heavy"},
        max_strength_by_material={"vinyl": 0.85, "tape": 0.90, "shellac": 0.95, "cd_digital": 0.40},
        time_profile="heavy",  # BS-RoFormer + MIIPHER + Resemble = 9+ Minuten!
        min_severity=0.3,
        note="Schwerste ML-Phase; SNR<10 + vocal → MIIPHER-Sigma konservativ (0.25-0.40)",
    ),
    # ── Frequenz-Entzerrung ──────────────────────────────────────
    "phase_04_eq_correction": PhaseEffectProfile(
        phase_id="phase_04_eq_correction",
        goal_impact={
            "brillanz": +0.03,
            "waerme": +0.02,
            "natuerlichkeit": +0.01,
        },
        risks=["over_brightening"],
        preconditions={"bandwidth_hz": "< 15000"},
        max_strength_by_material={"vinyl": 0.8, "tape": 0.7, "shellac": 0.9},
        time_profile="fast",
        min_severity=0.2,
        note="Material-adaptive EQ; unkritisch, nur bei Bandbreitenverlust stark",
    ),
    # ── Wow/Flutter ─────────────────────────────────────────────
    "phase_12_wow_flutter_fix": PhaseEffectProfile(
        phase_id="phase_12_wow_flutter_fix",
        goal_impact={
            "tonal_center": +0.04,
            "emotionalitaet": +0.03,
            "waerme": +0.01,
        },
        risks=["pitch_artifact", "phase_distortion"],
        preconditions={"wow_severity": "> 0.3"},
        max_strength_by_material={"vinyl": 0.7, "tape": 0.9, "cassette": 1.0},
        time_profile="medium",
        min_severity=0.3,
        note="Polyphonic-Speed-Korrektur; bei vinyl konservativ (mechanisch, nicht elektrisch)",
    ),
    # ── Vocal/De-Esser ──────────────────────────────────────────
    "phase_19_de_esser": PhaseEffectProfile(
        phase_id="phase_19_de_esser",
        goal_impact={
            "artikulation": +0.04,
            "natuerlichkeit": +0.01,
            "brillanz": -0.01,  # kann HF marginal dämpfen
        },
        risks=["vocal_dulling", "gender_mismatch"],
        preconditions={"panns_singing": "> 0.20", "gender_detected": "valid"},
        max_strength_by_material={"vinyl": 0.85, "tape": 0.80, "cd_digital": 0.60},
        time_profile="medium",
        min_severity=0.1,
        note="Gender-abhängige Sibilanz-Bänder; female→6-10kHz, male→4-8kHz",
    ),
    # ── Reverb/Dereverb ──────────────────────────────────────────
    "phase_20_reverb_reduction": PhaseEffectProfile(
        phase_id="phase_20_reverb_reduction",
        goal_impact={
            "transparenz": +0.04,
            "artikulation": +0.03,
            "waerme": -0.02,  # Hall-Entfernung reduziert Wärme
            "spatial_depth": -0.03,  # Weniger Hall = weniger Raumtiefe
        },
        risks=["over_drying", "vocal_thinning"],
        preconditions={"rt60_s": "> 0.5"},
        max_strength_by_material={"vinyl": 0.6, "tape": 0.7, "cd_digital": 0.5},
        time_profile="medium",
        min_severity=0.2,
        note="DSP+DNN-Hybrid; bei church/broadcast cap durch RoomAcoustics",
    ),
    # ── Präsenz-Boost ───────────────────────────────────────────
    "phase_38_presence_boost": PhaseEffectProfile(
        phase_id="phase_38_presence_boost",
        goal_impact={
            "brillanz": +0.05,
            "artikulation": +0.03,
            "waerme": -0.01,
            "natuerlichkeit": -0.02,  # kann künstlich wirken
        },
        risks=["over_brightening", "vocal_harshness"],
        preconditions={"bandwidth_loss": "present"},
        max_strength_by_material={"vinyl": 0.7, "tape": 0.6, "cd_digital": 0.3},
        time_profile="fast",
        min_severity=0.3,
        note="HF-Anhebung; nur bei echten Bandbreitenverlust, nicht als Default-Enhancement",
    ),
}


# ── §2.60 Kalibrierungs-Logik ─────────────────────────────────────────────

def calibrate_phase_intensity(
    phase_id: str,
    base_strength: float,
    *,
    defect_severity: float = 0.0,
    material: str = "vinyl",
    panns_singing: float = 0.0,
    snr_db: float | None = None,
    rt60_s: float = 0.5,
    bandwidth_hz: float = 20000,
    era_decade: int = 1980,
    genre_is_schlager: bool = False,
    soft_saturation_preserve: bool = False,
) -> float:
    """§2.60: Kalibriert die Phasen-Intensität proaktiv.

    Nutzt das PhaseEffectProfile + Audio-Zustand um die optimale
    Intensität VOR der Ausführung zu berechnen. PMGG validiert danach.

    Returns: kalibrierte Stärke in [0.0, 1.0]
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return base_strength  # Kein Profil → Original-Stärke

    strength = float(base_strength)

    # 1. Defekt-Schwere-Skalierung
    if profile.min_severity > 0 and defect_severity < profile.min_severity:
        strength *= max(0.1, defect_severity / max(profile.min_severity, 0.01))

    # 2. Material-Cap
    mat_cap = profile.max_strength_by_material.get(material, 0.9)
    strength = min(strength, mat_cap)

    # 3. Risiko-basierte Reduktion
    for risk in profile.risks:
        if risk == "vocal_distortion" and panns_singing > 0.25:
            # Gesang im Signal → ML-Phasen konservativer
            vocal_factor = 1.0 - (panns_singing - 0.25) * 0.8  # 0.25→1.0, 0.35→0.92, 0.50→0.80
            strength *= max(0.4, vocal_factor)
        if risk == "energy_loss" and snr_db is not None and snr_db < 8:
            strength *= 0.6  # Sehr niedriger SNR → Energie-Verlust-Risiko
        if risk == "over_brightening" and soft_saturation_preserve:
            strength *= 0.5  # Sättigung erhalten → nicht zusätzlich aufhellen
        if risk == "over_drying" and rt60_s > 2.0:
            strength *= 0.7  # Sehr hallig → nicht zu viel Hall entfernen (war Aufnahme-Charakter)
        if risk == "vocal_dulling" and panns_singing > 0.3:
            strength *= 0.85  # De-Esser bei starkem Gesang etwas zurückhaltender
        if risk == "ml_artifact" and era_decade < 1980:
            strength *= 0.7  # Vintage-Material → ML-Artefakte wahrscheinlicher

    # 4. Genre-spezifische Anpassung
    if genre_is_schlager:
        if "waerme" in profile.goal_impact and profile.goal_impact.get("waerme", 0) < 0:
            strength *= 0.75  # Schlager braucht Wärme → Phasen die Wärme reduzieren drosseln
        if phase_id == "phase_20_reverb_reduction":
            strength *= 0.6  # Schlager-HALL ist erwünscht!

    # 5. SNR-Abhängigkeit für ML-Phasen
    if snr_db is not None and "ml_artifact" in profile.risks:
        if snr_db < 5:
            strength *= 0.5  # Sehr niedriger SNR → ML tut mehr Schaden als Nutzen
        elif snr_db > 15:
            strength = min(strength, 0.4)  # Hoher SNR → ML kaum nötig

    return max(0.0, min(1.0, strength))


def get_phase_risk_level(phase_id: str, **audio_state) -> str:
    """Bewertet das Risiko-Level einer Phase im aktuellen Audio-Kontext.

    Returns: "low" | "medium" | "high" | "critical"
    """
    profile = PHASE_EFFECT_CATALOG.get(phase_id)
    if profile is None:
        return "low"

    risk_score = 0.0
    panns = float(audio_state.get("panns_singing", 0))
    snr = audio_state.get("snr_db")
    era = int(audio_state.get("era_decade", 1980))

    if "vocal_distortion" in profile.risks and panns > 0.25:
        risk_score += (panns - 0.25) * 3.0
    if "ml_artifact" in profile.risks and era < 1980:
        risk_score += 0.5
    if "ml_artifact" in profile.risks and (snr is None or snr < 8):
        risk_score += 1.0

    if risk_score > 2.0:
        return "critical"
    elif risk_score > 1.0:
        return "high"
    elif risk_score > 0.3:
        return "medium"
    return "low"
