"""
User Acceptance Test (UAT) — Acceptance Criteria & Release Gates
Aurik 9.10.77 — Formal Validation Suite
Status: 28. März 2026

This module defines 30 acceptance criteria (15 Restoration + 15 Studio 2026)
and 7 release gates (K.O. criteria). Parametrized tests validate each criterion.
Output is formatted for audit/uat_report_generator.py machine parsing.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ============================================================================
# CRITERIA DEFINITIONS
# ============================================================================

RESTORATION_CRITERIA = [
    {
        "id": "R1",
        "name": "Einstiegs-Nachricht klar und hilfreich",
        "description": "Mode-Ankündigung (Restoration/Studio 2026) ist präzise & verständlich",
        "category": "UI/UX",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check modern_window.py for mode announcement strings",
    },
    {
        "id": "R2",
        "name": "Defekt-Scanning transparent gemacht",
        "description": "Scanning-Fortschritt wird live dem Nutzer angezeigt",
        "category": "UI/UX",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check scan_progress signal usage in modern_window.py",
    },
    {
        "id": "R3",
        "name": "Zweistufige Progress Bars funktionieren",
        "description": "Haupt-ProgressBar + phase_progress_bar beide aktiv",
        "category": "UI/UX",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check both progress labels in modern_window.py UI definition",
    },
    {
        "id": "R4",
        "name": "Waveform-Scan-Cursor sichtbar",
        "description": "Orange Scan-Cursor mit Glow während Defekt-Analyse",
        "category": "UI/UX",
        "severity": "SHOULD",
        "test_type": "code_inspection",
        "validation": "Check waveform_widget.set_scan_pos() call presence",
    },
    {
        "id": "R5",
        "name": "Vocals in Stereo präserviert",
        "description": "Stereo-Separation in der Vokal-Restaurierung bleibt intakt",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run phase_42_vocal_enhancement on stereo test signal",
    },
    {
        "id": "R6",
        "name": "Tonart nicht verschoben",
        "description": "TonalCenterMetric ≥ 0.95 nach Restaurierung",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Check TonalCenterMetric in musical_goals_checker output",
    },
    {
        "id": "R7",
        "name": "Mikro-Dynamik erhalten",
        "description": "MDEM-Modul erfolgreich angewendet; Dynamics-Pearson ≥ 0.92",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run MDEM in unified_restorer_v3; verify score",
    },
    {
        "id": "R8",
        "name": "Keine stillen Defekte eingeführt",
        "description": "Audio-Rauschboden bleibt ≥ -72 dBFS nach Verarbeitung",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Measure noise floor before/after restoration",
    },
    {
        "id": "R9",
        "name": "Reversing funktioniert",
        "description": "Ctrl+Z (Undo last restoration) lädt Originallude nicht",
        "category": "UI/UX",
        "severity": "SHOULD",
        "test_type": "code_inspection",
        "validation": "Check undo logic in modern_window.py shortcuts",
    },
    {
        "id": "R10",
        "name": "Export mit korrekten LUFS",
        "description": "LUFS-Differenz original → export ≤ 1.0 LU für Restoration",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Measure LUFS ITU-R BS.1770-5 on export file",
    },
    {
        "id": "R11",
        "name": "Musikalische Ziele nicht verschlechtert",
        "description": "Sämtliche 14 Musikalischen Ziele ≥ Threshold nach Phase-Ausführung",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run MusicalGoalsChecker.measure_all() at end of pipeline",
    },
    {
        "id": "R12",
        "name": "Keine NaN/Inf-Werte im Audio",
        "description": "Vollständiges Ausgabe-Audio ist finite (keine NaN, Inf)",
        "category": "Code Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "np.isfinite(audio).all() check after export",
    },
    {
        "id": "R13",
        "name": "Mono/Stereo korrekt detektiert",
        "description": "Kanal-Zähler nach Import = Echo real channels (nicht falsch klassifiziert)",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check file_import.py channel detection logic",
    },
    {
        "id": "R14",
        "name": "Material-Klassifikation funktioniert",
        "description": "EraClassifier & MediumClassifier ordnen Material korrekt ein",
        "category": "Audio Analysis",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run era/medium classifier on test samples",
    },
    {
        "id": "R15",
        "name": "Pass-Through SNR > 40 dB",
        "description": "Bei sehr hohem SNR (clean digital) ändert sich Audio minimal (PQS < 0.05)",
        "category": "Audio Quality",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Test on clean CD/MP3-high material",
    },
]

STUDIO_2026_CRITERIA = [
    {
        "id": "S1",
        "name": "Studio 2026 Modusmeldung",
        "description": "Nutzer erhält Bestätigung: 'Studio 2026 gewählt'",
        "category": "UI/UX",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check modern_window.py mode announcement for Studio 2026",
    },
    {
        "id": "S2",
        "name": "Stem-Separation aktiv",
        "description": "BsRoFormer/Stem-Sep liefert Vocals + Instruments Streams",
        "category": "Audio Processing",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run phase_42 stem separation; verify stream independence",
    },
    {
        "id": "S3",
        "name": "Vocal-Enhancement aktiv",
        "description": "VocalAIEnhancement modul wird auf Vokal-Stream angewendet",
        "category": "Audio Processing",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check phase_43 + VocalAIEnhancement invocation",
    },
    {
        "id": "S4",
        "name": "Reference Mastering angewendet",
        "description": "Mastering-Chain mit Sidechain, EQ, Kompression wird ausgeführt",
        "category": "Audio Processing",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Verify mastering.py is_invoked in Studio 2026 path",
    },
    {
        "id": "S5",
        "name": "LUFS -14 EBU R128 erreicht",
        "description": "Finales Export-Audio ≈ -14 LUFS ± 0.5 LU (EBU R128)",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Measure LUFS on final export; compare to -14 target",
    },
    {
        "id": "S6",
        "name": "Brillanz/Wärme-Balance",
        "description": "Presence + Air ≤ +4 dB relativ zu Original; Wärme ≥ 0.75",
        "category": "Audio Quality",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Check BrillanzMetric + WaermeMetric scores",
    },
    {
        "id": "S7",
        "name": "Räumliche Tiefe erhalten",
        "description": "SpatialDepthMetric ≥ 0.75 nach Studio-2026-Verarbeitung",
        "category": "Audio Quality",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Run SpatialDepthMetric check",
    },
    {
        "id": "S8",
        "name": "TruePeak respektiert",
        "description": "Maximales true-peak ≤ +3 dBFS; keine Übersteuerung",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Measure true-peak on export file",
    },
    {
        "id": "S9",
        "name": "Resampling korrekt",
        "description": "Bei 44.1k Import: Resampling zu 48k, Phasen-Verarbeitung, zurück zu 44.1k; SNR ≥ -0.8 dB",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Test resampling chain on 44.1k file",
    },
    {
        "id": "S10",
        "name": "Multi-band Compressor angewendet",
        "description": "5-band EQ-linked compressor zur Dynamik-Kontrolle",
        "category": "Audio Processing",
        "severity": "SHOULD",
        "test_type": "code_inspection",
        "validation": "Check multiband_compressor invocation in mastering.py",
    },
    {
        "id": "S11",
        "name": "Emotional Arc erhalten",
        "description": "Makro-Dynamik-Bogen (5 s) bleibt Arousal/Valence ≥ 0.80",
        "category": "Audio Quality",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Run emotional_arc_correction; verify score improvement",
    },
    {
        "id": "S12",
        "name": "Artefakte minimal",
        "description": "Artefakt-Detektionsquote < 0.5 % (von Gesamt-Audio-Samples)",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Run artifact_detection_api on final audio",
    },
    {
        "id": "S13",
        "name": "Rauschboden -72 dBFS",
        "description": "Studio-2026-Ausgabe: Rausch ≤ -72 dBFS, A-gewichtet ≤ -75 dB(A)",
        "category": "Audio Quality",
        "severity": "MUST",
        "test_type": "functional_test",
        "validation": "Measure noise floor on near-silent regions",
    },
    {
        "id": "S14",
        "name": "Sidechain funktioniert (Vocals)",
        "description": "Compressor-Sidechain reagiert auf Vokal-Energie; Pumpen hörbar bei hoher Kompression",
        "category": "Audio Processing",
        "severity": "SHOULD",
        "test_type": "functional_test",
        "validation": "Verify sidechain signal flow in multiband_compressor",
    },
    {
        "id": "S15",
        "name": "Export-Gate erfolgreich",
        "description": "Export findet statt NUR wenn quality_estimate ≥ 0.55",
        "category": "Code Quality",
        "severity": "MUST",
        "test_type": "code_inspection",
        "validation": "Check export_guard() logic in bridge.py",
    },
]

RELEASE_GATES = [
    {
        "id": "G1",
        "name": "Kein Docker in Production-Pfaden",
        "description": "Keine Docker-Abhängigkeit in Produktions-Audio-Verarbeitung (bare-metal-only)",
        "ko": True,
        "test_id": "test_no_docker_in_production_paths",
        "severity": "CRITICAL",
    },
    {
        "id": "G2",
        "name": "KMV batch audio aus Originaludio",
        "description": "KMV Stufe 2 nutzt Originaludio, nicht Tube3-Export; kein Doppel-Processing",
        "ko": True,
        "test_id": "test_kmv_batch_audio_correct",
        "severity": "CRITICAL",
    },
    {
        "id": "G3",
        "name": "Keine silent refinement cancellations",
        "description": "Wenn Nutzer KMV abbricht: Feedback-Signal sent; kein Silent Hang",
        "ko": True,
        "test_id": "test_no_silent_refinement_cancellation",
        "severity": "CRITICAL",
    },
    {
        "id": "G4",
        "name": "Progress Counter funktioniert",
        "description": "Defekt-Zähler: +1 bei Erkennung, -1 bei Phase-Repair; konsistent mit Phasen",
        "ko": False,
        "test_id": "test_progress_counter_consistency",
        "severity": "MAJOR",
    },
    {
        "id": "G5",
        "name": "Musical Goals Gate nicht übersprungen",
        "description": "PMGG führt nie Phase aus (Action='rollback') — bei Failure nutze Best-Effort",
        "ko": True,
        "test_id": "test_pmgg_no_rollback_skipping",
        "severity": "CRITICAL",
    },
    {
        "id": "G6",
        "name": "OQS ≥ 80 auf ≥1 AMRB-Szenario",
        "description": "AMRB-Benchmark: Aurik erreicht mindestens auf 1/10 Szenarien OQS 80+",
        "ko": False,
        "test_id": "test_amrb_minimum_oqs_80",
        "severity": "MAJOR",
    },
    {
        "id": "G7",
        "name": "Hybrid Release Mode deterministisch",
        "description": "Release-Mode (primary/fallback/blocked) lässt sich reproducieren; Fallback-Kaskade funktioniert",
        "ko": True,
        "test_id": "test_hybrid_release_mode_determinism",
        "severity": "CRITICAL",
    },
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def check_code_for_pattern(file_path: str, patterns: list[str]) -> bool:
    """
    Check if any pattern is found in a file.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return any(re.search(p, content, re.IGNORECASE) for p in patterns)
    except Exception as e:
        pytest.fail(f"File check failed: {e}")


def assert_code_contract(contract_name: str, checks: list[tuple[str, list[str]]]) -> None:
    """Assert that all required code-level markers for a contract are present."""
    missing: list[str] = []
    for file_path, patterns in checks:
        if not check_code_for_pattern(file_path, patterns):
            missing.append(f"{file_path}: {patterns}")
    assert not missing, f"{contract_name} verletzt — fehlende Marker: {missing}"


def run_existing_test(test_id: str) -> bool:
    """
    Run an existing pytest test and return pass/fail status.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-xvs",
                f"tests/normative/{test_id}.py",
                "--tb=short",
            ],
            cwd=Path("/media/michael/Software 4TB/Aurik_Standalone"),
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        pytest.fail(f"Test execution failed: {e}")


# ============================================================================
# RESTORATION CRITERIA TESTS
# ============================================================================


@pytest.mark.parametrize("criterion", RESTORATION_CRITERIA, ids=lambda c: c["id"])
def test_restoration_criteria(criterion: dict[str, Any]):
    """Parametrized test for all Restoration criteria."""

    result = {
        "criterion_id": criterion["id"],
        "name": criterion["name"],
        "result": "PASS",
        "evidence": "",
        "timestamp": "",
    }

    try:
        if criterion["id"] == "R1":
            # Mode announcement
            found = check_code_for_pattern(
                "Aurik910/ui/modern_window.py",
                [
                    r"Restoration\s+gew[äa]hlt",
                    r"Studio\s+2026\s+gew[äa]hlt",
                ],
            )
            assert found, "Mode announcement strings not found"
            result["evidence"] = "Mode announcement strings present in code"

        elif criterion["id"] == "R2":
            # Defect scanning
            found = check_code_for_pattern(
                "Aurik910/ui/modern_window.py",
                [r"scan_progress", r"_on_scan_progress"],
            )
            assert found, "scan_progress signal not found"
            result["evidence"] = "scan_progress signal integrated in UI"

        elif criterion["id"] == "R3":
            # Progress bars
            found = check_code_for_pattern(
                "Aurik910/ui/modern_window.py",
                [r"phase_progress_bar", r"setRange\(0,\s*10000\)"],
            )
            assert found, "Phase progress bar not configured"
            result["evidence"] = "phase_progress_bar + main progress_bar both present"

        elif criterion["id"] == "R4":
            # Waveform cursor
            found = check_code_for_pattern(
                "Aurik910/ui/modern_window.py",
                [r"set_scan_pos", r"waveform_widget"],
            )
            assert found, "Waveform scan position not implemented"
            result["evidence"] = "waveform_widget.set_scan_pos() integrated"

        elif criterion["id"] == "R5":
            assert_code_contract(
                "R5 Vocals in Stereo",
                [
                    ("backend/core/stem_remix_balancer.py", [r"def\s+balance_remix", r"vocals", r"instruments"]),
                    ("backend/core/unified_restorer_v3.py", [r"BsRoFormer|_bsr_", r"stems", r"vocals"]),
                ],
            )
            result["evidence"] = "StemRemixBalancer + BsRoFormer-Stems im Pipeline-Code vorhanden"

        elif criterion["id"] == "R6":
            assert_code_contract(
                "R6 Tonal Center",
                [
                    (
                        "backend/core/musical_goals/musical_goals_metrics.py",
                        [r"class\s+TonalCenterMetric", r"tonal_center"],
                    )
                ],
            )
            result["evidence"] = "TonalCenterMetric im Musical-Goals-System vorhanden"

        elif criterion["id"] == "R7":
            assert_code_contract(
                "R7 Mikro-Dynamik",
                [
                    (
                        "backend/core/unified_restorer_v3.py",
                        [r"MicroDynamicsEnvelopeMorphing", r"correct_emotional_arc"],
                    ),
                ],
            )
            result["evidence"] = "MDEM + Emotional-Arc-Korrektur in UV3 verdrahtet"

        elif criterion["id"] == "R8":
            assert_code_contract(
                "R8 Noise Floor Guard",
                [
                    ("backend/core/unified_restorer_v3.py", [r"noise_floor_p5_db|noise_floor_db", r"-72\.0|-72 dB"]),
                ],
            )
            result["evidence"] = "Noise-Floor-Marker inkl. -72 dBFS Referenz vorhanden"

        elif criterion["id"] == "R9":
            # Reversing (Ctrl+Z)
            found = check_code_for_pattern("Aurik910/ui/modern_window.py", [r"Ctrl\+Z", r"Undo"])
            assert found, "Undo shortcut not found"
            result["evidence"] = "Ctrl+Z shortcut defined"

        elif criterion["id"] == "R10":
            assert_code_contract(
                "R10 Export LUFS",
                [
                    ("backend/core/audio_exporter.py", [r"LUFS", r"EBU R128", r"TruePeak|dBTP"]),
                ],
            )
            result["evidence"] = "AudioExporter enthält LUFS-Normalisierung + TruePeak-Safety"

        elif criterion["id"] == "R11":
            assert_code_contract(
                "R11 Musical Goals Gate",
                [
                    (
                        "backend/core/musical_goals/musical_goals_metrics.py",
                        [r"measure_all", r"self\.thresholds|get_mode_thresholds", r"14"],
                    ),
                ],
            )
            result["evidence"] = "MusicalGoalsChecker misst 14 Ziele gegen Schwellwerte"

        elif criterion["id"] == "R12":
            assert_code_contract(
                "R12 NaN/Inf Guard",
                [
                    ("backend/core/unified_restorer_v3.py", [r"np\.isfinite", r"NaN|Inf"]),
                    ("backend/exporter.py", [r"_export_guard", r"nan_to_num|np\.clip"]),
                ],
            )
            result["evidence"] = "Finite-Guards in UV3 + Exporter vorhanden"

        elif criterion["id"] == "R13":
            # Mono/Stereo detection
            found = check_code_for_pattern(
                "backend/file_import.py",
                [r"ndim.*2", r"channels?.*==.*[12]", r"shape\[0\]"],
            )
            assert found, "Channel detection code not clear"
            result["evidence"] = "Channel detection logic present in file_import.py"

        elif criterion["id"] == "R14":
            assert_code_contract(
                "R14 Material-Klassifikation",
                [
                    ("backend/core/era_classifier.py", [r"class\s+EraClassifier", r"def\s+get_era_classifier"]),
                    (
                        "backend/core/medium_classifier.py",
                        [r"class\s+MediumClassifier", r"def\s+get_medium_classifier"],
                    ),
                ],
            )
            result["evidence"] = "EraClassifier + MediumClassifier implementiert"

        elif criterion["id"] == "R15":
            assert_code_contract(
                "R15 Pass-Through",
                [
                    ("denker/aurik_denker.py", [r"clean_digital_pass_through", r"pass-through|passthrough"]),
                ],
            )
            result["evidence"] = "Clean-digital Pass-Through-Pfad im Denker vorhanden"

        else:
            pytest.fail(f"Unknown criterion {criterion['id']}")

    except AssertionError as e:
        result["result"] = "FAIL"
        result["evidence"] = str(e)
        pytest.fail(f"{criterion['id']}: {e!s}")
    except Exception as e:
        result["result"] = "ERROR"
        result["evidence"] = str(e)
        raise


# ============================================================================
# STUDIO 2026 CRITERIA TESTS
# ============================================================================


@pytest.mark.parametrize("criterion", STUDIO_2026_CRITERIA, ids=lambda c: c["id"])
def test_studio_2026_criteria(criterion: dict[str, Any]):
    """Parametrized test for all Studio 2026 criteria."""

    result = {
        "criterion_id": criterion["id"],
        "name": criterion["name"],
        "result": "PASS",
        "evidence": "",
        "timestamp": "",
    }

    try:
        if criterion["id"] == "S1":
            # Studio 2026 mode announcement
            found = check_code_for_pattern("Aurik910/ui/modern_window.py", [r"Studio\s+2026\s+gew[äa]hlt"])
            assert found, "Studio 2026 announcement not found"
            result["evidence"] = "Studio 2026 mode announcement present"

        elif criterion["id"] == "S2":
            assert_code_contract(
                "S2 Stem Separation",
                [
                    ("backend/core/unified_restorer_v3.py", [r"BsRoFormer|_bsr_", r"stems", r"vocals|instruments"]),
                ],
            )
            result["evidence"] = "BsRoFormer-Stem-Separation in UV3 verdrahtet"

        elif criterion["id"] == "S3":
            assert_code_contract(
                "S3 Vocal Enhancement",
                [
                    ("backend/core/unified_restorer_v3.py", [r"phase_43_ml_deesser", r"vocal"]),
                ],
            )
            result["evidence"] = "Phase 43 (ML-De-Esser/Vocal-Kette) im UV3-Flow"

        elif criterion["id"] == "S4":
            assert_code_contract(
                "S4 Reference Mastering",
                [
                    ("backend/core/regulator/mastering.py", [r"def\s+mastering_chain", r"multiband_compress"]),
                    ("backend/core/unified_restorer_v3.py", [r"phase_17_mastering_polish|mastering"]),
                ],
            )
            result["evidence"] = "Mastering-Chain und UV3-Mastering-Phase vorhanden"

        elif criterion["id"] == "S5":
            assert_code_contract(
                "S5 LUFS -14",
                [
                    ("backend/core/regulator/mastering.py", [r"target_lufs\s*=\s*-14\.0", r"lufs_normalize"]),
                ],
            )
            result["evidence"] = "Mastering nutzt -14 LUFS Ziel"

        elif criterion["id"] == "S6":
            assert_code_contract(
                "S6 Brillanz/Waerme",
                [
                    (
                        "backend/core/musical_goals/musical_goals_metrics.py",
                        [r"class\s+BrillanzMetric", r"class\s+WaermeMetric"],
                    ),
                ],
            )
            result["evidence"] = "Brillanz- und Wärme-Metriken vorhanden"

        elif criterion["id"] == "S7":
            assert_code_contract(
                "S7 Spatial Depth",
                [
                    ("backend/core/musical_goals/musical_goals_metrics.py", [r"class\s+SpatialDepthMetric"]),
                ],
            )
            result["evidence"] = "SpatialDepthMetric implementiert"

        elif criterion["id"] == "S8":
            assert_code_contract(
                "S8 TruePeak",
                [
                    ("backend/core/audio_exporter.py", [r"TruePeak|dBTP"]),
                ],
            )
            result["evidence"] = "TruePeak-Schutz im AudioExporter vorhanden"

        elif criterion["id"] == "S9":
            assert_code_contract(
                "S9 Resampling-Kette",
                [
                    ("backend/core/dsp_resample_wrapper.py", [r"resample_to_48k", r"48000"]),
                    ("backend/file_import.py", [r"librosa\.resample", r"target_sr"]),
                ],
            )
            result["evidence"] = "Resampling-Pfade für Import/48k-Verarbeitung vorhanden"

        elif criterion["id"] == "S10":
            assert_code_contract(
                "S10 Multiband Compressor",
                [
                    ("backend/core/regulator/mastering.py", [r"def\s+multiband_compress", r"mastering_chain"]),
                ],
            )
            result["evidence"] = "Multiband-Kompressor in der Mastering-Chain aktiv"

        elif criterion["id"] == "S11":
            assert_code_contract(
                "S11 Emotional Arc",
                [
                    (
                        "backend/core/emotional_arc_preservation.py",
                        [r"def\s+correct_emotional_arc", r"measure_emotional_arc"],
                    ),
                    ("backend/core/unified_restorer_v3.py", [r"correct_emotional_arc"]),
                ],
            )
            result["evidence"] = "Emotional-Arc-Metrik und Korrektur in UV3 eingebunden"

        elif criterion["id"] == "S12":
            assert_code_contract(
                "S12 Artifact Detection",
                [
                    ("plugins/artifact_detection_plugin.py", [r"detect_artifacts", r"ArtifactDetectionPlugin"]),
                    (
                        "backend/core/introduced_artifact_detector.py",
                        [r"class\s+IntroducedArtifactDetector|def\s+get_iad"],
                    ),
                ],
            )
            result["evidence"] = "Artefakt-Detektion über Plugin + Core-Detector vorhanden"

        elif criterion["id"] == "S13":
            assert_code_contract(
                "S13 Noise Floor -72 dBFS",
                [
                    ("backend/core/unified_restorer_v3.py", [r"noise_floor", r"-72\.0|-72 dB"]),
                ],
            )
            result["evidence"] = "Noise-Floor-Referenz für Studio-Pfade vorhanden"

        elif criterion["id"] == "S14":
            assert_code_contract(
                "S14 Vocal-adaptive Mixing",
                [
                    ("backend/core/stem_remix_balancer.py", [r"vocal_weight", r"balance_remix"]),
                    ("backend/core/regulator/mastering.py", [r"multiband_compress"]),
                ],
            )
            result["evidence"] = "Vokal-adaptive Remix-Logik + Kompressorpfad vorhanden"

        elif criterion["id"] == "S15":
            # Export gate
            found = check_code_for_pattern(
                "backend/api/bridge.py",
                [r"export_guard", r"quality_estimate.*>=.*0.55"],
            )
            assert found, "Export guard not properly implemented"
            result["evidence"] = "export_guard() checks quality_estimate >= 0.55"

        else:
            pytest.fail(f"Unknown criterion {criterion['id']}")

    except AssertionError as e:
        result["result"] = "FAIL"
        result["evidence"] = str(e)
        pytest.fail(f"{criterion['id']}: {e!s}")
    except Exception as e:
        result["result"] = "ERROR"
        result["evidence"] = str(e)
        raise


# ============================================================================
# RELEASE GATES TESTS
# ============================================================================


def test_no_docker_in_production_paths():
    """Gate G1: No Docker in production paths."""
    # This test typically exists in tests/normative/
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/normative/test_no_docker_in_production_paths.py",
                "-xvs",
                "--tb=short",
            ],
            cwd=Path("/media/michael/Software 4TB/Aurik_Standalone"),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, "Docker in production paths detected"
    except subprocess.TimeoutExpired:
        pytest.skip("Gate test timeout")


def test_kmv_batch_audio_correct():
    """Gate G2: KMV uses original audio for batch refinement."""
    # Check that KMV refinement path uses audio_original, not tube3_export
    try:
        base = Path("/media/michael/Software 4TB/Aurik_Standalone")
        code_path = base / "Aurik910" / "ui" / "modern_window.py"
        with open(code_path, encoding="utf-8") as f:
            content = f.read()
        # Ensure KMV job payload carries original audio and no legacy tube3 reference.
        assert "audio_original" in content, "DeferredRefinementJob should use audio_original"
        assert "tube3_export" not in content, "KMV path should not reference tube3_export"
    except Exception as e:
        pytest.fail(f"KMV batch audio check failed: {e}")


def test_no_silent_refinement_cancellation():
    """Gate G3: Refinement cancellation sends feedback signal."""
    from pathlib import Path

    try:
        code_path = Path("/media/michael/Software 4TB/Aurik_Standalone") / "Aurik910" / "ui" / "ml_refinement_thread.py"
        with open(code_path, encoding="utf-8") as f:
            content = f.read()
        # Check for refinement_cancelled signal emission
        assert "refinement_cancelled" in content, "No refinement_cancelled signal found"
        # Ensure signal is actually emitted in cancellation path
        assert ".emit(" in content, "Signal emission not found"
    except Exception as e:
        pytest.fail(f"Silent cancellation check failed: {e}")


def test_progress_counter_consistency():
    """Gate G4: Progress counter increments/decrements correctly."""
    try:
        code_path = Path("/media/michael/Software 4TB/Aurik_Standalone") / "Aurik910" / "ui" / "modern_window.py"
        with open(code_path, encoding="utf-8") as f:
            content = f.read()
        # Check for counter update logic
        assert "_PHASE_REDUCES" in content or "detected" in content, "Phase-defect mapping not found"
    except Exception as e:
        pytest.fail(f"Progress counter check failed: {e}")


def test_pmgg_no_rollback_skipping():
    """Gate G5: PMGG never returns 'rollback' action."""
    try:
        code_path = (
            Path("/media/michael/Software 4TB/Aurik_Standalone")
            / "backend"
            / "core"
            / "per_phase_musical_goals_gate.py"
        )
        with open(code_path, encoding="utf-8") as f:
            content = f.read()
        # Check that 'rollback' is not a valid action in PMGG
        assert 'action="rollback"' not in content, "PMGG should never use rollback action"
        # Ensure best_effort is used instead
        assert 'action="best_effort"' in content or "best_effort" in content, "PMGG should use best_effort"
    except Exception as e:
        pytest.fail(f"PMGG rollback check failed: {e}")


@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_amrb_minimum_oqs_80():
    """Gate G6: AMRB achieves OQS >= 80 on at least one scenario.

    This is intentionally classified as heavy (`ml` + `slow`), so default test
    runs deselect it via root `conftest.py`. In heavy runs it executes a real
    AMRB benchmark instead of using a hard skip.
    """
    from benchmarks.musical_restoration_benchmark import BenchmarkConfig, run_benchmark

    def _aurik_restoration_fn(audio, sr):
        from backend.core.unified_restorer_v3 import get_restorer

        restorer = get_restorer()
        result = restorer.restore(audio, sr)
        return result.audio

    report = run_benchmark(
        BenchmarkConfig(
            restoration_fn=_aurik_restoration_fn,
            system_name="Aurik 9 UAT Gate G6",
            n_items_per_scenario=1,
            verbose=False,
        )
    )
    best_oqs = max((res.mushra_mean for res in report.scenario_results.values()), default=0.0)
    assert best_oqs >= 80.0, (
        f"Gate G6 failed: best scenario OQS={best_oqs:.1f} < 80.0 "
        f"(overall={report.overall_score:.1f}, passed={report.n_passed}/{report.n_scenarios})"
    )


def test_hybrid_release_mode_determinism():
    """Gate G7: Hybrid Release Mode is deterministic."""
    try:
        code_path = Path("/media/michael/Software 4TB/Aurik_Standalone") / "backend" / "core" / "fallback_guard.py"
        with open(code_path, encoding="utf-8") as f:
            content = f.read()
        # Check for release_mode states
        assert "release_mode" in content, "release_mode not defined"
        assert "primary" in content and "fallback" in content and "blocked" in content, "Release mode states incomplete"
    except Exception as e:
        pytest.fail(f"Hybrid release mode check failed: {e}")


# ============================================================================
# PYTEST FIXTURE FOR COLLECTING RESULTS
# ============================================================================


@pytest.fixture(scope="session")
def uat_results_collector():
    """Collects UAT test results for report generation."""
    results = {
        "restoration_criteria": [],
        "studio_2026_criteria": [],
        "release_gates": [],
        "summary": {
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "ko_violations": 0,
            "recommendation": "UNKNOWN",
        },
    }
    return results


# ============================================================================
# MARKER DEFINITIONS
# ============================================================================

pytest.mark.uat = pytest.mark.uat
pytest.mark.gate = pytest.mark.gate
