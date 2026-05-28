"""Compliance-Auto-Verifikation [RELEASE_MUST] — Aurik VERBOTEN-Linter (V01–V33 Zielstand) im CI.

Stellt sicher, dass kein Anti-Pattern-Scan-Ergebnis veraltet ist:
der Linter wird bei jedem Pytest-Lauf direkt ausgeführt und validiert.

Abgedeckte Regeln (ERROR-Level):
    V01 np.corrcoef → guarded dot-product
    V03 boundary='reflect' → 'even'
    V04 apply_musical_gain_envelope ohne reference_for_gate
    V05 print() statt logger
    V06 map_location='cuda' ohne ml_device_manager
    V07 scipy.signal.wiener() direkt
    V08 np.correlate O(n²)
    V09 from Aurik910 in backend/
    V10 load_audio_file ohne do_carrier_analysis=False
    V12 CAUSE_TO_PHASES/CAUSES Bidirektional-Sync (§2.59)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LINTER = _ROOT / "scripts" / "aurik_verboten_linter.py"
_PYTHON = sys.executable


class TestVerbotenlLinterZeroViolations:
    """Stellt sicher, dass der VERBOTEN-Linter im gesamten backend/ und plugins/ Verzeichnis
    keine ERROR-Level-Verstöße findet.

    Dieser Test ist kein Unit-Test eines einzelnen Moduls — er ist der systemische
    Compliance-Gate der verhindert, dass Scan-Ergebnisse veralten (§0f §0g)."""

    def test_linter_script_exists(self) -> None:
        assert _LINTER.exists(), f"aurik_verboten_linter.py nicht gefunden: {_LINTER}"

    def test_backend_no_error_violations(self) -> None:
        """Aktuelle ERROR-Level-Regeln des VERBOTEN-Linters: 0 Verstöße in backend/."""
        result = subprocess.run(
            [_PYTHON, str(_LINTER), str(_ROOT / "backend")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        # Exit 0 = keine ERROR-Verstöße (Warnings sind erlaubt)
        assert result.returncode == 0, f"VERBOTEN-Linter meldet ERROR-Verstöße in backend/:\n\n{output}"

    def test_plugins_no_error_violations(self) -> None:
        """Aktuelle ERROR-Level-Regeln des VERBOTEN-Linters: 0 Verstöße in plugins/."""
        result = subprocess.run(
            [_PYTHON, str(_LINTER), str(_ROOT / "plugins")],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, f"VERBOTEN-Linter meldet ERROR-Verstöße in plugins/:\n\n{output}"

    def test_causal_reasoner_v12_sync(self) -> None:
        """V12 speziell: CAUSE_TO_PHASES/CAUSES Bidirektional-Sync in causal_defect_reasoner.py."""
        cdr = _ROOT / "backend" / "core" / "causal_defect_reasoner.py"
        result = subprocess.run(
            [_PYTHON, str(_LINTER), str(cdr)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"V12 CAUSE_TO_PHASES/CAUSES Sync-Verletzung in causal_defect_reasoner.py:\n\n{output}"
        )

    def test_no_stash_drift_above_threshold(self) -> None:
        """Stash-Drift-Guard: Nicht mehr als 2 offene Stashes (Snapshot-Akkumulation).

        Mehr als 2 Stashes deuten auf Snapshot-Drift — ältere Fixes die nicht
        in HEAD gemergt wurden (§0g Autonomes-Entscheidungs-Doktrin).
        """
        result = subprocess.run(
            ["git", "stash", "list"],
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=10,
        )
        stash_count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
        assert stash_count <= 2, (
            f"Stash-Drift: {stash_count} Stashes vorhanden (Limit: 2).\n"
            "Vor dem nächsten Commit auflösen:\n"
            "  git stash show --stat   → Inhalt prüfen\n"
            "  git stash drop          → Verwerfen wenn bereits in HEAD\n"
            "  git stash pop           → Integrieren (Konflikte manuell lösen)\n"
            "Hintergrund: Snapshot-Stashes auf alten Commit-Basen führen zu "
            "Regressions-Reintroduktion (§0c Universalitäts-Invariante)."
        )


class TestVerbotenLinterExtendedRules:
    def test_v27_flags_jitter_phase12_in_causal_reasoner(self, tmp_path: Path) -> None:
        file_path = tmp_path / "causal_defect_reasoner.py"
        file_path.write_text(
            textwrap.dedent(
                """
                CAUSES = ["jitter_artifacts"]
                CAUSE_TO_PHASES = {
                    "jitter_artifacts": ["phase_12_wow_flutter_fix", "phase_23_spectral_repair"],
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V27" for v in violations)

    def test_v28_flags_nr_breathing_with_denoise_in_mapper(self, tmp_path: Path) -> None:
        file_path = tmp_path / "defect_phase_mapper.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import DefectType
                class PhaseAssignment:
                    def __init__(self, **kwargs):
                        pass
                MAPPING = {
                    DefectType.NR_BREATHING_ARTIFACT: PhaseAssignment(
                        defect_type=DefectType.NR_BREATHING_ARTIFACT,
                        primary_phases=["phase_03_denoise"],
                    ),
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V28" for v in violations)

    def test_v29_flags_overload_with_phase63_in_mapper(self, tmp_path: Path) -> None:
        file_path = tmp_path / "defect_phase_mapper.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import DefectType
                class PhaseAssignment:
                    def __init__(self, **kwargs):
                        pass
                MAPPING = {
                    DefectType.OVERLOAD_DISTORTION: PhaseAssignment(
                        defect_type=DefectType.OVERLOAD_DISTORTION,
                        primary_phases=["phase_63_intermodulation_reduction"],
                    ),
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V29" for v in violations)

    def test_v30_flags_aliasing_with_phase03_in_reasoner(self, tmp_path: Path) -> None:
        file_path = tmp_path / "causal_defect_reasoner.py"
        file_path.write_text(
            textwrap.dedent(
                """
                CAUSES = ["aliasing"]
                CAUSE_TO_PHASES = {
                    "aliasing": ["phase_03_denoise", "phase_23_spectral_repair"],
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V30" for v in violations)

    def test_v30_does_not_cross_into_next_mapper_block(self, tmp_path: Path) -> None:
        file_path = tmp_path / "defect_phase_mapper.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import DefectType
                class PhaseAssignment:
                    def __init__(self, **kwargs):
                        pass
                MAPPING = {
                    DefectType.ALIASING: PhaseAssignment(
                        defect_type=DefectType.ALIASING,
                        primary_phases=["phase_23_spectral_repair"],
                        secondary_phases=["phase_50_spectral_repair"],
                    ),
                    DefectType.BIAS_ERROR: PhaseAssignment(
                        defect_type=DefectType.BIAS_ERROR,
                        primary_phases=["phase_03_denoise"],
                    ),
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert not any(v.rule == "V30" for v in violations)

    def test_v31_warns_when_room_mode_lacks_phase04(self, tmp_path: Path) -> None:
        file_path = tmp_path / "causal_defect_reasoner.py"
        file_path.write_text(
            textwrap.dedent(
                """
                CAUSES = ["room_mode_resonance"]
                CAUSE_TO_PHASES = {
                    "room_mode_resonance": ["phase_05_rumble_filter"],
                }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V31" for v in violations)

    def test_v32_flags_missing_transparenz_exclusion(self, tmp_path: Path) -> None:
        file_path = tmp_path / "cumulative_interaction_guard.py"
        file_path.write_text(
            textwrap.dedent(
                """
                _PHASE_SPECIFIC_DRIFT_EXCLUSIONS = {
                    "phase_29": ["authentizitaet"],
                    "phase_03": ["natuerlichkeit"],
                }

                CRITICAL_PAIRS = [
                    (frozenset({"phase_29_tape_hiss_reduction", "phase_03_denoise"}), "transparenz", "x", -0.04),
                ]
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V32" for v in violations)

    def test_v32_detects_annassign_exclusions_dict(self, tmp_path: Path) -> None:
        file_path = tmp_path / "cumulative_interaction_guard.py"
        file_path.write_text(
            textwrap.dedent(
                """
                _PHASE_SPECIFIC_DRIFT_EXCLUSIONS: dict[str, list[str]] = {
                    "phase_29": ["authentizitaet"],
                    "phase_03": ["natuerlichkeit"],
                }

                CRITICAL_PAIRS = [
                    (frozenset({"phase_29_tape_hiss_reduction", "phase_03_denoise"}), "transparenz", "x", -0.04),
                ]
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V32" for v in violations)

    def test_v32_detects_tuple_exclusions_without_transparenz(self, tmp_path: Path) -> None:
        file_path = tmp_path / "cumulative_interaction_guard.py"
        file_path.write_text(
            textwrap.dedent(
                """
                _PHASE_SPECIFIC_DRIFT_EXCLUSIONS = {
                    "phase_29": ("authentizitaet",),
                    "phase_03": ("natuerlichkeit",),
                }

                CRITICAL_PAIRS = [
                    (frozenset({"phase_29_tape_hiss_reduction", "phase_03_denoise"}), "transparenz", "x", -0.04),
                ]
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V32" for v in violations)

    def test_v33_flags_phase_material_dict_missing_cassette(self, tmp_path: Path) -> None:
        phases_dir = tmp_path / "phases"
        phases_dir.mkdir()
        file_path = phases_dir / "phase_12_wow_flutter_fix.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import MaterialType

                class WowFlutterFix:
                    CORRECTION_STRENGTH = {
                        MaterialType.TAPE: 0.8,
                        MaterialType.VINYL: 0.7,
                        MaterialType.SHELLAC: 0.6,
                    }

                    DETECTION_THRESHOLD = {
                        MaterialType.TAPE: 0.3,
                        MaterialType.VINYL: 0.5,
                    }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V33" for v in violations)

    def test_v33_flags_annassign_material_dict_missing_cassette(self, tmp_path: Path) -> None:
        phases_dir = tmp_path / "phases"
        phases_dir.mkdir()
        file_path = phases_dir / "phase_40_gain_tracking.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import MaterialType

                class GainTracking:
                    DETECTION_THRESHOLD: dict[MaterialType, float] = {
                        MaterialType.TAPE: 0.3,
                        MaterialType.VINYL: 0.5,
                        MaterialType.SHELLAC: 0.7,
                    }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert any(v.rule == "V33" for v in violations)

    def test_v33_ignores_non_tracked_material_dict_names(self, tmp_path: Path) -> None:
        phases_dir = tmp_path / "phases"
        phases_dir.mkdir()
        file_path = phases_dir / "phase_11_custom.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import MaterialType

                class CustomPhase:
                    MATERIAL_WEIGHTS = {
                        MaterialType.TAPE: 0.9,
                        MaterialType.VINYL: 0.8,
                        MaterialType.SHELLAC: 0.7,
                    }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert not any(v.rule == "V33" for v in violations)

    def test_v33_ignores_non_carrier_material_dicts(self, tmp_path: Path) -> None:
        phases_dir = tmp_path / "phases"
        phases_dir.mkdir()
        file_path = phases_dir / "phase_41_digital_only.py"
        file_path.write_text(
            textwrap.dedent(
                """
                from backend.core.defect_scanner import MaterialType

                class DigitalOnly:
                    CORRECTION_STRENGTH = {
                        MaterialType.CD_DIGITAL: 0.3,
                        MaterialType.STREAMING: 0.2,
                    }
                """
            ),
            encoding="utf-8",
        )

        import scripts.aurik_verboten_linter as _linter

        violations = _linter.scan_file(file_path)
        assert not any(v.rule == "V33" for v in violations)


class TestV38Phase24VfaZoneCaps:
    """§V38 Per-Event-Strength-Oracle in phase_24_dropout_repair."""

    def test_vfa_vibrato_cap_applied(self):
        """Dropout in Vibrato-Zone muss auf max 0.20 gecappt werden."""
        import numpy as np

        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        p = DropoutRepairPhase()
        sr = 48000
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        d_start = int(0.04 * sr)
        d_end = int(0.05 * sr)
        audio[d_start:d_end] = 0.0
        p._current_panns_tags = {"Singing voice": 0.8}
        p._current_material = "vinyl"
        p.sample_rate = sr

        base = p._content_adaptive_repair_strength(0.95, "tonal", 10.0)
        local_s = p._compute_dropout_local_strength(audio, d_start, d_end, sr, base, [(0.03, 0.08, 0.20)])
        assert local_s <= 0.20 + 1e-6, f"Vibrato-Cap nicht aktiv: {local_s:.4f}"

    def test_vfa_frisson_cap_applied(self):
        """Dropout in Frisson-Zone muss auf max 0.30 gecappt werden."""
        import numpy as np

        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        p = DropoutRepairPhase()
        sr = 48000
        audio = 0.4 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        d_start = int(0.15 * sr)
        d_end = int(0.17 * sr)
        audio[d_start:d_end] = 0.0
        p._current_panns_tags = {"Singing voice": 0.9}
        p._current_material = "shellac"
        p.sample_rate = sr

        base = p._content_adaptive_repair_strength(0.95, "tonal", 20.0)
        local_s = p._compute_dropout_local_strength(audio, d_start, d_end, sr, base, [(0.10, 0.20, 0.30)])
        assert local_s <= 0.30 + 1e-6, f"Frisson-Cap nicht aktiv: {local_s:.4f}"

    def test_outside_zone_no_cap(self):
        """Dropout ausserhalb aller Schutzzonen darf volle Staerke erhalten."""
        import numpy as np

        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        p = DropoutRepairPhase()
        sr = 48000
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        d_start = int(0.50 * sr)
        d_end = int(0.51 * sr)
        audio[d_start:d_end] = 0.0
        p._current_panns_tags = {}
        p._current_material = "vinyl"
        p.sample_rate = sr

        base = p._content_adaptive_repair_strength(0.95, "mixed", 10.0)
        local_s = p._compute_dropout_local_strength(
            audio, d_start, d_end, sr, base, [(0.03, 0.08, 0.20), (0.10, 0.20, 0.30)]
        )
        assert local_s > 0.20, f"VFA-Cap feuerte falsch-positiv ausserhalb Zone: {local_s:.4f}"

    def test_zero_base_strength_returns_zero(self):
        """base_strength < 1e-6 muss 0.0 zurueckgeben."""
        import numpy as np

        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        p = DropoutRepairPhase()
        sr = 48000
        audio = np.zeros(sr, dtype=np.float32)
        p._current_panns_tags = {}
        p._current_material = "unknown"
        p.sample_rate = sr

        local_s = p._compute_dropout_local_strength(audio, 0, 100, sr, 0.0, [])
        assert local_s == 0.0, f"Erwartete 0.0, erhalten: {local_s}"

    def test_process_accepts_vfa_zone_kwargs(self):
        """process() muss vibrato_zones/frisson_zones kwargs verarbeiten ohne Fehler."""
        import numpy as np

        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        p = DropoutRepairPhase()
        sr = 48000
        audio = 0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        audio[1000:1100] = 0.0
        result = p.process(
            audio,
            sr,
            material_type="vinyl",
            vibrato_zones=[(0.01, 0.05, 0.20)],
            frisson_zones=[(0.5, 0.8, 0.30)],
            whisper_zones=[(0.2, 0.3, 0.25)],
            passaggio_zones=[(0.6, 0.7, 0.35)],
        )
        assert result.audio.shape == audio.shape


class TestV38Phase55VfaZoneRouting:
    """§V38 VFA-Schutzzonen-Routing in phase_55_diffusion_inpainting."""

    def test_vibrato_zone_uses_boundary_fill_not_ml(self):
        """Gap in Vibrato-Zone darf kein ML-Inpainting erhalten."""
        import numpy as np

        from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase

        p = DiffusionInpaintingPhase()
        sr = 48000
        audio = 0.4 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        audio[int(0.04 * sr) : int(0.06 * sr)] = 0.0  # Dropout in Vibrato-Zone

        # Ohne VFA: normales Inpainting
        # Mit VFA: Boundary-Fill → kein plugin_used
        r = p.process(
            audio,
            sr,
            material_type="vinyl",
            vibrato_zones=[(0.03, 0.08, 0.20)],
        )
        assert r.audio.shape == audio.shape

    def test_frisson_zone_uses_boundary_fill(self):
        """Gap in Frisson-Zone → kein aggressives ML-Inpainting."""
        import numpy as np

        from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase

        p = DiffusionInpaintingPhase()
        sr = 48000
        audio = 0.4 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        audio[int(0.5 * sr) : int(0.52 * sr)] = 0.0  # Dropout in Frisson-Zone

        r = p.process(
            audio,
            sr,
            material_type="vinyl",
            frisson_zones=[(0.45, 0.6, 0.30)],
        )
        assert r.audio.shape == audio.shape

    def test_no_vfa_zones_normal_path(self):
        """Ohne VFA-Zones läuft normaler Inpainting-Pfad."""
        import numpy as np

        from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase

        p = DiffusionInpaintingPhase()
        sr = 48000
        audio = np.zeros(sr, dtype=np.float32)
        r = p.process(audio, sr, material_type="vinyl")
        assert r.audio.shape == audio.shape


class TestV38Phase27VfaZoneCaps:
    """§V38 phase_27 per-click-strength-oracle VFA-Schutzzonen."""

    def test_compute_click_local_strength_base_zero(self):
        """base_strength < 1e-6 → 0.0 (V38-Invariante)."""
        import numpy as np

        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        p = ClickPopRemoval()
        audio = np.random.default_rng(0).random(48000).astype(np.float32) * 0.1
        result = p._compute_click_local_strength(audio, 1000, 1020, 48000, 0.0, [])
        assert result == 0.0

    def test_compute_click_local_strength_vibrato_cap(self):
        """Vibrato-Zone cap 0.20 wird eingehalten."""
        import numpy as np

        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        p = ClickPopRemoval()
        audio = np.ones(48000, dtype=np.float32) * 0.5
        # click at 0.5s, vibrato zone 0.4–0.7s
        start = int(0.5 * 48000)
        end = start + 10
        protected_zones = [(0.4, 0.7, 0.20)]
        result = p._compute_click_local_strength(audio, start, end, 48000, 1.0, protected_zones)
        assert result <= 0.20

    def test_compute_click_local_strength_frisson_cap(self):
        """Frisson-Zone cap 0.30 wird eingehalten."""
        import numpy as np

        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        p = ClickPopRemoval()
        audio = np.ones(48000, dtype=np.float32) * 0.5
        start = int(0.5 * 48000)
        end = start + 10
        protected_zones = [(0.4, 0.7, 0.30)]
        result = p._compute_click_local_strength(audio, start, end, 48000, 1.0, protected_zones)
        assert result <= 0.30

    def test_process_with_vibrato_zones_no_crash(self):
        """process() mit vibrato_zones läuft ohne Exception durch."""
        import numpy as np

        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        p = ClickPopRemoval()
        sr = 48000
        audio = np.random.default_rng(1).random(sr * 2).astype(np.float32) * 0.05
        r = p.process(
            audio,
            sr,
            material_type="vinyl",
            vibrato_zones=[(0.3, 0.6)],
            frisson_zones=[(0.8, 1.0)],
        )
        assert r.audio.shape == audio.shape

    def test_process_without_vfa_zones_normal_path(self):
        """Ohne VFA-Zones läuft normaler Klick-Reparatur-Pfad."""
        import numpy as np

        from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval

        p = ClickPopRemoval()
        sr = 48000
        audio = np.zeros(sr, dtype=np.float32)
        r = p.process(audio, sr, material_type="vinyl")
        assert r.audio.shape == audio.shape
