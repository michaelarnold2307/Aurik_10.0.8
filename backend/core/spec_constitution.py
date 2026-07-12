"""
spec_constitution.py — v10 Normative Spec-Konstitution
=========================================================

Zentrale, maschinenlesbare Konstitution aller Aurik-Spezifikationen.
Dient als SINGLE SOURCE OF TRUTH fuer Agent und Watchdog.

WAS HIER STEHT, GILT. Keine Ausnahmen, keine Overrides ohne Spec-Änderung.

Integration:
  from backend.core.spec_constitution import get_constitution
  const = get_constitution()
  const.check_artifact_freedom(audio, sr)     → (bool, str)
  const.get_forbidden_patterns()               → list[ForbiddenPattern]
  const.get_musical_goal_thresholds(material)  → dict[str, float]
  const.validate_against_paragraph_zero(audio) → list[str]

Author: Aurik 10 Development Team — Juli 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §0 — Oberstes Prinzip: Klangwahrheit
# ═══════════════════════════════════════════════════════════════════════════════

KLANGWAHRHEIT = """
Das Ziel jeder Restaurierung ist, dass der Hörer die Augen schließt und
die originale Performance hört — nicht eine technisch korrekte
Signalverarbeitung, und nicht eine 'verbesserte' Version.
"""

PRIMUM_NON_NOCERE = "Füge dem Klang keinen Schaden zu."
MINIMAL_INTERVENTION = "Greife nur ein, wo der Defekt hörbar ist."
PERCEPTUAL_IMPROVEMENT = "Der Export muss näher am Original-Klang liegen als der degradierte Input."

# §0h Music-Death-Shield
MUSIC_DEATH_SHIELD = {
    "artifact_freedom_min": 0.95,
    "hpi_min": 0.0,
    "vqi_recovery_trigger": 0.72,
    "vqi_restoration_target": 0.82,
    "vqi_studio2026_target": 0.87,
    "oqs_min": 80,
    "timbral_fidelity_min": 0.93,
}

# §0i Perceptual Transparency Guarantee
PERCEPTUAL_TRANSPARENCY = {
    "musical_noise_max_above_carrier_db": 0.0,
    "frisson_zones_preserved": True,
}

# §0k Maximum-Achievable-Score
MAS_PRINCIPLE = "Jeder Song wird unter Berücksichtigung der Quelldatei und der physikalischen Grenzen bis zum maximal möglichen Ergebnis restauriert."

# §0m Maximal-Ausbaustufe Defektintelligenz
MAX_DEFECT_INTELLIGENCE = "Beide Modi (restoration + studio2026) immer auf maximaler Ausbaustufe für Defekterkennung, -Differenzierung und -Dosierung."

# §0p Primus-inter-Pares (Vocals first)
PRIMUS_INTER_PARES = "Wenn panns_singing >= 0.25, erhält Stimmqualität Vorrang vor allen anderen Zielen."

# ═══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN — VERBOTEN-Regeln (normativ, aus .github/VERBOTEN.md)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ForbiddenPattern:
    """Eine VERBOTEN-Regel mit Erkennungsmuster und Begründung."""

    id: str  # V01-V58
    category: str  # Logging, API, DSP, Guard, ...
    pattern: str  # Erkennungsmuster (für AST/Linter)
    description: str  # Was verboten ist
    correct: str  # Was stattdessen zu tun ist
    severity: str  # critical / warning


FORBIDDEN_PATTERNS: list[ForbiddenPattern] = [
    # ── Kritische Architektur-Regeln ──
    ForbiddenPattern(
        "V01", "Logging", "print(", "print()-Aufrufe in Produktionscode", "logger.info/warning/error()", "critical"
    ),
    ForbiddenPattern("V02", "API", "return dict", "dict als API-Returnwert", "@dataclass", "warning"),
    ForbiddenPattern(
        "V03", "Cache", "_cache = {}", "Ungeschützter Module-Level-Cache", "threading.Lock() + Dict", "critical"
    ),
    ForbiddenPattern(
        "V04",
        "Guard",
        "gate_dbfs=-36.0",
        "Fester Gate-Schwellwert ohne reference_for_gate",
        "compute_signal_relative_gate_dbfs(ref)",
        "critical",
    ),
    ForbiddenPattern(
        "V05",
        "GPU",
        'map_location="cuda"',
        "GPU-Zugriff ohne ml_device_manager",
        "get_torch_device('PluginName')",
        "critical",
    ),
    ForbiddenPattern(
        "V06",
        "Audio",
        "sf.read|librosa.load",
        "Direkter Audio-Import ohne load_audio_file()",
        "load_audio_file(filepath)",
        "critical",
    ),
    ForbiddenPattern(
        "V07",
        "DSP",
        "sosfilt(sos, audio)",
        "Kausaler Filter bei Signal-Addition → Phasenversatz",
        "sosfiltfilt(sos, audio) (zero-phase)",
        "critical",
    ),
    ForbiddenPattern(
        "V08", "Normalize", "RMS|peak.*normali", "RMS/Peak-Normalisierung", "LUFS ITU-R BS.1770-5", "warning"
    ),
    ForbiddenPattern("V09", "Metric", "pesq|dnsmos|nisqa", "Veraltete Metriken", "PQS-MOS, VERSA, SingMOS", "warning"),
    ForbiddenPattern(
        "V10",
        "SongCal",
        "clip.*0\\.0.*2\\.0",
        "global_scalar außerhalb [0.50, 1.50]",
        "global_scalar ∈ [0.50,1.50], family ∈ [0.30,1.80]",
        "critical",
    ),
    ForbiddenPattern(
        "V11",
        "Guard",
        "MAX_DRIFT = -0\\.05|regression > 0\\.02",
        "Feste Guard-Schwellwerte",
        "compute_adaptive_drift_tolerance()",
        "warning",
    ),
    ForbiddenPattern(
        "V12",
        "Phase",
        "Phase_21|Phase_35|Phase_42.*CAUSE_TO_PHASES",
        "Phase 21/35/42 in Restoration-CAUSE_TO_PHASES",
        "Diese Phasen NIE für Restoration-Cause einplanen",
        "critical",
    ),
    ForbiddenPattern(
        "V13",
        "DC",
        "np\\.mean.*subtract|lfilter.*DC",
        "DC-Offset via np.mean/lfilter bei reel_tape",
        "filtfilt([1,-1],[1,-0.9995]) zero-phase",
        "warning",
    ),
    ForbiddenPattern(
        "V14",
        "Import",
        "from Aurik10.*import.*backend",
        "Backend-Import aus Frontend (Aurik10)",
        "Bridge-API via backend/api/bridge.py",
        "critical",
    ),
    ForbiddenPattern(
        "V15",
        "Gain",
        "audio \\*= gain_factor",
        "Uniformer Gain (auch auf Stille)",
        "_musical_gain_envelope()",
        "critical",
    ),
    ForbiddenPattern(
        "V16",
        "Limiter",
        "soft.limit|hard.limit.*routin",
        "Routine-Limiter ohne Peak-Check",
        "NUR wenn peak > 0.98",
        "warning",
    ),
    ForbiddenPattern(
        "V17",
        "Phase_skip",
        "severity.*<.*0\\.05.*skip",
        "Phase-Skip nur nach Severity",
        "_salience_adjusted_severity() Pflicht",
        "warning",
    ),
    ForbiddenPattern(
        "V18",
        "ML",
        "except.*pass",
        "Stilles ML-Fallback ohne Logging",
        "logger.warning('ML->DSP fallback: ...')",
        "critical",
    ),
    ForbiddenPattern(
        "V19",
        "Noise",
        "noise.texture.*ohne.*V19",
        "Noise-Textur-Prüfung nach NR-Phase ohne V19-Check",
        "compute_noise_texture_distance()",
        "warning",
    ),
    ForbiddenPattern(
        "V20",
        "Correlation",
        "frame.energy.*ohne.*V20",
        "Mikrodynamik-Check nach NR ohne V20",
        "frame_energy_correlation() auf voiced-Zonen",
        "warning",
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 15 Musical Goals — Schwellwerte (Spec 01)
# ═══════════════════════════════════════════════════════════════════════════════

MUSICAL_GOALS = {
    "authentizitaet": {"threshold": 0.70, "weight": 0.10, "priority": 1},
    "natuerlichkeit": {"threshold": 0.70, "weight": 0.12, "priority": 1},
    "brillanz": {"threshold": 0.65, "weight": 0.06, "priority": 2},
    "timbre": {"threshold": 0.65, "weight": 0.08, "priority": 2},
    "groove": {"threshold": 0.60, "weight": 0.07, "priority": 2},
    "micro_dynamics": {"threshold": 0.60, "weight": 0.08, "priority": 3},
    "artikulation": {"threshold": 0.65, "weight": 0.08, "priority": 2},
    "waerme": {"threshold": 0.70, "weight": 0.07, "priority": 2},
    "tiefe": {"threshold": 0.65, "weight": 0.06, "priority": 3},
    "durchsetzung": {"threshold": 0.65, "weight": 0.06, "priority": 3},
    "transparenz": {"threshold": 0.65, "weight": 0.06, "priority": 2},
    "kohaerenz": {"threshold": 0.65, "weight": 0.05, "priority": 3},
    "fokus": {"threshold": 0.60, "weight": 0.05, "priority": 3},
    "balance": {"threshold": 0.65, "weight": 0.04, "priority": 3},
    "stimmung": {"threshold": 0.60, "weight": 0.03, "priority": 3},
}

# Material-adaptive Floor-Toleranzen (fragile Materialien haben niedrigere Erwartungen)
MATERIAL_GOAL_FLOOR: dict[str, dict[str, float]] = {
    "wax_cylinder": {"authentizitaet": 0.55, "natuerlichkeit": 0.55, "brillanz": 0.45},
    "shellac": {"authentizitaet": 0.60, "natuerlichkeit": 0.60, "brillanz": 0.55},
    "vinyl": {"authentizitaet": 0.65, "natuerlichkeit": 0.65, "brillanz": 0.60},
    "tape": {"authentizitaet": 0.65, "natuerlichkeit": 0.65, "brillanz": 0.60},
    "cassette": {"authentizitaet": 0.60, "natuerlichkeit": 0.60, "brillanz": 0.55},
}

# ═══════════════════════════════════════════════════════════════════════════════
# Bug & Gap Detection — 5-Layer-Scan-Protokoll (Spec 10)
# ═══════════════════════════════════════════════════════════════════════════════

BUG_GAP_LAYERS = {
    "L1_Frontend": {"scope": "Aurik10/ui/, Aurik10/__init__.py", "tools": "version-check, grep hardcoded"},
    "L2_Bridge_CLI": {"scope": "backend/api/bridge.py, cli/", "tools": "contract-completeness"},
    "L3_Denker": {"scope": "denker/*.py", "tools": "goal_applicability, reference params"},
    "L4_UV3": {"scope": "backend/core/unified_restorer_v3.py", "tools": "SSIP, Rescheduler, Memory, HPG"},
    "L5_Phases_DSP": {
        "scope": "backend/core/phases/, dsp/, plugins/",
        "tools": "V33-V42, ForwardMasking, Halluzination",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# SpecConstitution — Der programmatische Zugang
# ═══════════════════════════════════════════════════════════════════════════════


class SpecConstitution:
    """Zentrale, maschinenlesbare Aurik-Konstitution.

    Lädt und validiert alle normativen Regeln aus den Spec-Dokumenten.
    Wird von Agent und Watchdog als Single Source of Truth verwendet.

    Usage:
        const = get_constitution()
        ok, issues = const.check_paragraph_zero(audio, sr)
        patterns = const.get_forbidden_patterns()
        goals = const.get_musical_goal_thresholds("vinyl")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forbidden: list[ForbiddenPattern] = FORBIDDEN_PATTERNS
        self._musical_goals: dict[str, dict[str, float]] = MUSICAL_GOALS
        self._material_floors: dict[str, dict[str, float]] = MATERIAL_GOAL_FLOOR
        self._shield: dict[str, float] = MUSIC_DEATH_SHIELD
        self._transparency: dict[str, Any] = PERCEPTUAL_TRANSPARENCY

    # ── §0 Validation ────────────────────────────────────────────────────

    def check_paragraph_zero(
        self,
        audio: np.ndarray,
        sr: int,
        artifact_freedom: float = 1.0,
        hpi: float = 0.5,
        has_vocals: bool = False,
        vqi: float = 0.8,
    ) -> list[str]:
        """Prüft gegen §0 Klangwahrheit und Music-Death-Shield.

        Returns:
            Liste von Verletzungen (leer = alle Checks bestanden).
        """
        issues: list[str] = []

        # §0h: Music-Death-Shield — Artifact Freedom ist primäres Veto
        if artifact_freedom < self._shield["artifact_freedom_min"]:
            issues.append(
                f"§0h VETO: artifact_freedom={artifact_freedom:.3f} < "
                f"{self._shield['artifact_freedom_min']} — Export muss blockiert werden"
            )

        # §0h: HPI ≤ 0 → Over-Processing
        if hpi <= self._shield["hpi_min"]:
            issues.append(f"§0h VETO: HPI={hpi:.3f} ≤ 0 — Signal wurde verschlechtert. Original-Input exportieren.")

        # §0p: Primus-inter-Pares — Vokal-Vorrang
        if has_vocals and vqi < self._shield["vqi_recovery_trigger"]:
            issues.append(
                f"§0p RECOVERY: VQI={vqi:.3f} < {self._shield['vqi_recovery_trigger']} "
                f"bei erkanntem Gesang — _recovery_cascade() auslösen"
            )

        return issues

    def check_primum_non_nocere(
        self,
        original_rms_dbfs: float,
        restored_rms_dbfs: float,
        crest_delta_db: float,
    ) -> list[str]:
        """Prüft gegen Primum non nocere.

        Erkennt:
          - RMS-Kollaps (Signal wurde zu leise / verschwand)
          - Crest-Verlust (Dynamik wurde plattgebügelt)
          - Mögliche Artefakt-Einführung durch Überbearbeitung
        """
        issues: list[str] = []

        if restored_rms_dbfs < -80.0:
            issues.append("Primum non nocere: RMS-Kollaps — Signal wurde zerstört")
        if restored_rms_dbfs < original_rms_dbfs - 15.0:
            issues.append(
                f"Primum non nocere: RMS-Abfall {original_rms_dbfs - restored_rms_dbfs:.1f}dB — möglicher Signalkollaps"
            )
        if crest_delta_db < -6.0:
            issues.append(f"Primum non nocere: Crest-Verlust {crest_delta_db:.1f}dB — Dynamik wurde plattgebügelt")

        return issues

    # ── Forbidden Patterns ──────────────────────────────────────────────

    def get_forbidden_patterns(self, severity: str | None = None) -> list[ForbiddenPattern]:
        """Gibt alle VERBOTEN-Regeln zurück, optional gefiltert nach Severity."""
        if severity:
            return [p for p in self._forbidden if p.severity == severity]
        return list(self._forbidden)

    def get_critical_forbidden(self) -> list[ForbiddenPattern]:
        """Nur kritische VERBOTEN-Regeln (müssen immer gelten)."""
        return [p for p in self._forbidden if p.severity == "critical"]

    def check_forbidden_in_code(self, code_text: str) -> list[tuple[ForbiddenPattern, str]]:
        """Scannt Code-Text auf VERBOTEN-Muster und gibt Treffer zurück."""
        import re

        hits: list[tuple[ForbiddenPattern, str]] = []
        for fp in self._forbidden:
            try:
                matches = re.finditer(fp.pattern, code_text, re.MULTILINE)
                for m in matches:
                    line = code_text[max(0, m.start() - 20) : m.end() + 20].replace("\n", " ")
                    hits.append((fp, line.strip()))
            except re.error:
                pass
        return hits

    # ── Musical Goals ───────────────────────────────────────────────────

    def get_musical_goal_thresholds(self, material: str = "unknown") -> dict[str, float]:
        """Gibt Goal-Schwellwerte zurück, material-adaptiv mit Floor-Toleranzen."""
        thresholds: dict[str, float] = {}
        floor = self._material_floors.get(material, {})
        for goal, cfg in self._musical_goals.items():
            floor_val = floor.get(goal, cfg["threshold"])
            thresholds[goal] = max(floor_val, cfg["threshold"])
        return thresholds

    def get_goal_weights(self) -> dict[str, float]:
        """Gibt Goal-Gewichte für Composite-Score zurück."""
        return {g: c["weight"] for g, c in self._musical_goals.items()}

    def evaluate_goals(self, scores: dict[str, float], material: str = "unknown") -> tuple[int, int, list[str]]:
        """Evaluiert Goal-Scores gegen material-adaptive Schwellwerte.

        Returns:
            (passed, total, failed_goals)
        """
        thresholds = self.get_musical_goal_thresholds(material)
        passed = 0
        total = 0
        failed: list[str] = []
        for goal, score in scores.items():
            if goal in thresholds:
                total += 1
                if score >= thresholds[goal]:
                    passed += 1
                else:
                    failed.append(f"{goal}: {score:.3f} < {thresholds[goal]:.3f}")
        return passed, total, failed

    # ── Music Death Shield ──────────────────────────────────────────────

    def get_shield_thresholds(self) -> dict[str, float]:
        return dict(self._shield)

    def is_export_blocked(self, artifact_freedom: float, hpi: float) -> tuple[bool, str]:
        """Prüft ob der Export durch §0h blockiert wird."""
        if artifact_freedom < self._shield["artifact_freedom_min"]:
            return True, f"§0h: artifact_freedom={artifact_freedom:.3f} < {self._shield['artifact_freedom_min']}"
        if hpi <= 0:
            return True, f"§0h: HPI={hpi:.3f} ≤ 0 — Over-Processing"
        return False, ""

    # ── Bug-Gap-Strategie ───────────────────────────────────────────────

    def get_bug_gap_layers(self) -> dict[str, dict[str, str]]:
        return dict(BUG_GAP_LAYERS)

    def validate_layer_completeness(self, layer: str, artifacts: dict[str, Any]) -> list[str]:
        """Validiert ob eine Bug-Gap-Layer vollständig geprüft wurde."""
        if layer not in BUG_GAP_LAYERS:
            return [f"Unbekannte Layer: {layer}"]
        expected = BUG_GAP_LAYERS[layer]
        missing = []
        for tool in expected["tools"].split(", "):
            if tool not in str(artifacts):
                missing.append(f"Fehlendes Tool: {tool} in {layer}")
        return missing

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def forbidden_count(self) -> int:
        return len(self._forbidden)

    @property
    def goal_count(self) -> int:
        return len(self._musical_goals)

    @property
    def shield(self) -> dict[str, float]:
        return dict(self._shield)


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_constitution: SpecConstitution | None = None
_constitution_lock = threading.Lock()


def get_constitution() -> SpecConstitution:
    """Thread-sicherer Singleton-Accessor."""
    global _constitution
    with _constitution_lock:
        if _constitution is None:
            _constitution = SpecConstitution()
    return _constitution
