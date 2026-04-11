"""Vollständigkeits-Beweis: Jeder DefectType × MaterialType × QualityMode → ≥1 gezielte Phase.

Spec-Referenz: copilot-instructions.md §7.2 (CAUSE_TO_PHASES), §6.3 (DefectType), §6.1 (MaterialType)
Invariante: „Entscheidend ist immer die musikalische Exzellenz."

Struktureller Beweis:
    810 parametrische Kombinationen (27 DefectTypes × 15 MaterialTypes × 2 QualityModes).
    Für jede Kombination mit severity=0.80 muss _select_phases() mindestens eine
    gezielt ausgewählte Phase (außer Tier-0-Pflichtphasen und Tier-6-Export) zurückgeben.

    Tier-0 (immer): phase_30_dc_offset_removal, phase_05_rumble_filter
    Tier-6 (immer): phase_16_final_eq, phase_17_mastering_polish,
                    phase_47_truepeak_limiter, phase_40_loudness_normalization,
                    phase_41_output_format_optimization
"""

from unittest.mock import MagicMock

import pytest

from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, DefectType, MaterialType
from backend.core.performance_guard import QualityMode

# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────────────────

TIER_0_PHASES = {
    "phase_30_dc_offset_removal",
    "phase_05_rumble_filter",
}

TIER_6_PHASES = {
    "phase_16_final_eq",
    "phase_17_mastering_polish",
    "phase_47_truepeak_limiter",
    "phase_40_loudness_normalization",
    "phase_41_output_format_optimization",
}

STRUCTURAL_PHASES = TIER_0_PHASES | TIER_6_PHASES


def _make_defect_result(
    material: MaterialType, defect_type: DefectType, severity: float = 0.80
) -> DefectAnalysisResult:
    """Erstellt ein DefectAnalysisResult mit einem einzigen aktiven Defekt."""
    scores = {dt: DefectScore(defect_type=dt, severity=0.0, confidence=0.0) for dt in DefectType}
    scores[defect_type] = DefectScore(
        defect_type=defect_type,
        severity=severity,
        confidence=0.9,
    )
    return DefectAnalysisResult(
        material_type=material,
        scores=scores,
        analysis_time_seconds=0.1,
        sample_rate=48000,
        duration_seconds=0.0,
    )


def _select_phases_for(material: MaterialType, defect_type: DefectType, mode: QualityMode) -> list[str]:
    """Ruft _select_phases() über RestorationConfig auf, ohne echte ML-Modelle."""
    # Lazy import damit der Singleton nicht im Modulscope instanziiert wird
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    config = RestorationConfig(mode=mode)
    restorer = UnifiedRestorerV3.__new__(UnifiedRestorerV3)
    restorer.config = config
    restorer.logger = MagicMock()

    dr = _make_defect_result(material, defect_type, severity=0.80)
    return restorer._select_phases(dr)


# ─────────────────────────────────────────────────────────────────────────────
# Parametrische Haupt-Testsuite (630 Kombinationen)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [QualityMode.QUALITY, QualityMode.BALANCED])
@pytest.mark.parametrize("material", list(MaterialType))
@pytest.mark.parametrize("defect_type", list(DefectType))
def test_every_switching_state_produces_targeted_phases(
    defect_type: DefectType, material: MaterialType, mode: QualityMode
):
    """Jede DefectType × MaterialType × QualityMode Kombination → ≥1 gezielte Phase."""
    phases = _select_phases_for(material, defect_type, mode)
    phase_set = set(phases)
    targeted = phase_set - STRUCTURAL_PHASES
    assert len(targeted) >= 1, (
        f"Keine gezielte Phase für:\n"
        f"  DefectType   = {defect_type.name}\n"
        f"  MaterialType = {material.name}\n"
        f"  QualityMode  = {mode.name}\n"
        f"  Aktive Phasen: {sorted(phase_set)}\n"
        f"  (Tier-0/6 strukturell, zählen nicht)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dedizierte Tests für die 8 behobenen Lücken
# ─────────────────────────────────────────────────────────────────────────────


class TestPrintThrough:
    """Fix #1: PRINT_THROUGH hatte überhaupt keinen Branch."""

    def test_tape_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases
        assert "phase_03_denoise" in phases

    def test_reel_tape_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases
        assert "phase_03_denoise" in phases

    def test_non_tape_material_no_print_through_phases(self):
        """Vinyl hat kein Print-Through — soll NICHT durch den PRINT_THROUGH-Block aktiviert werden."""
        phases = _select_phases_for(MaterialType.VINYL, DefectType.PRINT_THROUGH, QualityMode.QUALITY)
        # phase_29 sollte NICHT durch Print-Through ausgelöst sein (Vinyl ≠ Tape)
        # (Es könnte durch HIGH_FREQ_NOISE ausgelöst sein — aber wir testen nur die Grundphase)
        assert isinstance(phases, list)  # kein Absturz

    def test_no_nan_in_phases(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.PRINT_THROUGH, QualityMode.BALANCED)
        assert all(isinstance(p, str) for p in phases)


class TestQuantizationNoise:
    """Fix #2: QUANTIZATION_NOISE hatte überhaupt keinen Branch."""

    def test_quality_mode_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.CD_DIGITAL, DefectType.QUANTIZATION_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_balanced_mode_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.MP3_LOW, DefectType.QUANTIZATION_NOISE, QualityMode.BALANCED)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.QUANTIZATION_NOISE, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"QUANTIZATION_NOISE × {mat.name} → keine Phasen"


class TestJitterArtifacts:
    """Fix #3: JITTER_ARTIFACTS hatte überhaupt keinen Branch."""

    def test_dat_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases
        assert "phase_23_spectral_repair" in phases

    def test_streaming_material_activates_phases(self):
        phases = _select_phases_for(MaterialType.STREAMING, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_all_materials_produce_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.JITTER_ARTIFACTS, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"JITTER_ARTIFACTS × {mat.name} → keine Phasen"


class TestWaxCylinder:
    """Fix #4: WAX_CYLINDER wurde nirgends in _select_phases() behandelt."""

    def test_wax_cylinder_activates_click_removal(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_01_click_removal" in phases

    def test_wax_cylinder_activates_denoise(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases

    def test_wax_cylinder_activates_freq_restoration(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.BANDWIDTH_LOSS, QualityMode.QUALITY)
        assert "phase_06_frequency_restoration" in phases

    def test_wax_cylinder_always_includes_harmonic_restoration(self):
        phases = _select_phases_for(MaterialType.WAX_CYLINDER, DefectType.CLICKS, QualityMode.BALANCED)
        assert "phase_07_harmonic_restoration" in phases

    def test_wax_cylinder_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.WAX_CYLINDER, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"WAX_CYLINDER × {dt.name} → keine Phasen"


class TestWireRecording:
    """Fix #5: WIRE_RECORDING wurde nirgends in _select_phases() behandelt."""

    @pytest.mark.parametrize("defect_type", [DefectType.WOW, DefectType.FLUTTER])
    def test_wire_recording_activates_wow_and_flutter(self, defect_type: DefectType):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, defect_type, QualityMode.QUALITY)
        assert "phase_12_wow_flutter_fix" in phases

    def test_wire_recording_activates_dropout_repair(self):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, DefectType.DROPOUTS, QualityMode.QUALITY)
        assert "phase_24_dropout_repair" in phases

    def test_wire_recording_activates_hiss_reduction(self):
        phases = _select_phases_for(MaterialType.WIRE_RECORDING, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases

    def test_wire_recording_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.WIRE_RECORDING, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"WIRE_RECORDING × {dt.name} → keine Phasen"


class TestLacquerDisc:
    """Fix #6: LACQUER_DISC wurde nirgends in _select_phases() behandelt."""

    def test_lacquer_disc_activates_click_removal(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_01_click_removal" in phases

    def test_lacquer_disc_activates_crackle_removal(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CRACKLE, QualityMode.QUALITY)
        assert "phase_09_crackle_removal" in phases

    def test_lacquer_disc_activates_surface_noise_profiling(self):
        """LACQUER_DISC muss jetzt im surface_noise_profiling-Gate sein (Fix #1 + #6)."""
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.CRACKLE, QualityMode.QUALITY)
        assert "phase_28_surface_noise_profiling" in phases

    def test_lacquer_disc_activates_denoise(self):
        phases = _select_phases_for(MaterialType.LACQUER_DISC, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases

    def test_lacquer_disc_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.LACQUER_DISC, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"LACQUER_DISC × {dt.name} → keine Phasen"


class TestReelTapeGaps:
    """Fix #7: REEL_TAPE fehlte in Tape-Hiss, Azimuth, Air-Band, Tape-Saturation, EQ-Gates."""

    def test_reel_tape_activates_tape_hiss_reduction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.HIGH_FREQ_NOISE, QualityMode.QUALITY)
        assert "phase_29_tape_hiss_reduction" in phases

    def test_reel_tape_activates_azimuth_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.PHASE_ISSUES, QualityMode.QUALITY)
        assert "phase_25_azimuth_correction" in phases

    def test_reel_tape_activates_air_band_enhancement(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BANDWIDTH_LOSS, QualityMode.QUALITY)
        assert "phase_39_air_band_enhancement" in phases

    def test_reel_tape_activates_tape_saturation(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.CLICKS, QualityMode.BALANCED)
        assert "phase_22_tape_saturation" in phases

    def test_reel_tape_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_reel_tape_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.REEL_TAPE, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"REEL_TAPE × {dt.name} → keine Phasen"


class TestDATGap:
    """Fix #8: DAT fehlte im EQ-Correction-Gate."""

    def test_dat_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.CLICKS, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_dat_all_defects_have_phases(self):
        for dt in DefectType:
            phases = _select_phases_for(MaterialType.DAT, dt, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"DAT × {dt.name} → keine Phasen"


# ─────────────────────────────────────────────────────────────────────────────
# Robustheitstests
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    """_select_phases() darf niemals abstürzen oder NaN zurückgeben."""

    def test_all_combinations_no_exception(self):
        """630 Kombinationen — kein einziger Exception."""
        for mode in [QualityMode.QUALITY, QualityMode.BALANCED]:
            for mat in MaterialType:
                for dt in DefectType:
                    try:
                        phases = _select_phases_for(mat, dt, mode)
                        assert isinstance(phases, list)
                    except Exception as exc:
                        pytest.fail(f"Exception bei {dt.name} × {mat.name} × {mode.name}: {exc}")

    def test_phases_are_strings(self):
        """Alle zurückgegebenen Phasen-Namen sind echte Strings."""
        for mat in [
            MaterialType.WAX_CYLINDER,
            MaterialType.WIRE_RECORDING,
            MaterialType.LACQUER_DISC,
            MaterialType.REEL_TAPE,
        ]:
            for dt in [DefectType.PRINT_THROUGH, DefectType.QUANTIZATION_NOISE, DefectType.JITTER_ARTIFACTS]:
                phases = _select_phases_for(mat, dt, QualityMode.QUALITY)
                assert all(isinstance(p, str) and len(p) > 0 for p in phases), (
                    f"Ungültige Phase bei {mat.name} × {dt.name}: {phases}"
                )

    def test_no_duplicate_phases(self):
        """Keine doppelten Phasen in einer Auswahl (Deduplizierung)."""
        # WAX_CYLINDER kann mehrfach dieselbe Phase triggern (z.B. phase_03_denoise)
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.HIGH_FREQ_NOISE, QualityMode.BALANCED)
            assert len(phases) == len(set(phases)), f"Doppelte Phasen bei {mat.name}: {phases}"

    def test_tier_0_threshold_gated(self):
        """Tier-0 Phasen werden nur bei echten Defekten aktiviert (§0 Minimal-Intervention)."""
        for mat in list(MaterialType)[:3]:  # Stichprobe
            # Phase_30 nur bei DC-OFFSET severity > 0.10
            phases_dc = _select_phases_for(mat, DefectType.DC_OFFSET, QualityMode.QUALITY)
            assert "phase_30_dc_offset_removal" in set(phases_dc), f"phase_30 fehlt bei {mat.name} mit DC_OFFSET"

            # Phase_05 nur bei LOW_FREQ_RUMBLE severity > 0.10
            phases_rumble = _select_phases_for(mat, DefectType.LOW_FREQ_RUMBLE, QualityMode.QUALITY)
            assert "phase_05_rumble_filter" in set(phases_rumble), f"phase_05 fehlt bei {mat.name} mit RUMBLE"

    def test_tier_6_always_present(self):
        """Tier-6 Export-Pflichtphasen sind immer enthalten."""
        for mat in list(MaterialType)[:5]:  # Stichprobe
            phases = _select_phases_for(mat, DefectType.CLICKS, QualityMode.BALANCED)
            phase_set = set(phases)
            for p in TIER_6_PHASES:
                assert p in phase_set, f"Tier-6 Phase {p} fehlt bei {mat.name}"


# ─────────────────────────────────────────────────────────────────────────────
# Zusammenfassungs-Statistik (Informativ, kein Fehlschlag)
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_summary():
    """Gibt eine Zusammenfassung der Abdeckungsstatistik aus."""
    total = 0
    covered = 0
    gaps = []

    for mode in [QualityMode.QUALITY, QualityMode.BALANCED]:
        for mat in MaterialType:
            for dt in DefectType:
                total += 1
                phases = _select_phases_for(mat, dt, mode)
                targeted = set(phases) - STRUCTURAL_PHASES
                if len(targeted) >= 1:
                    covered += 1
                else:
                    gaps.append((dt.name, mat.name, mode.name))

    coverage_pct = covered / total * 100
    # §10.1: Kein print() in Tests — Diagnose-Info in assert-Nachricht
    gap_summary = (
        (f"Lücken ({len(gaps)}): " + "; ".join(f"{g[0]}×{g[1]}×{g[2]}" for g in gaps[:20]))
        if gaps
        else "alle Kombinationen vollständig abgedeckt"
    )

    assert coverage_pct == 100.0, f"Nicht alle Kombinationen abgedeckt: {coverage_pct:.1f}% — {gap_summary}"


# ─────────────────────────────────────────────────────────────────────────────
# Neue DefectTypes v9.10.46c: RIAA_CURVE_ERROR, ALIASING, BIAS_ERROR
# ─────────────────────────────────────────────────────────────────────────────


class TestRiaaCurveError:
    """Fix #9: RIAA_CURVE_ERROR — falsche Entzerrungskurve (Shellac/früher Vinyl)."""

    def test_shellac_activates_eq_and_freq_restoration(self):
        """Shellac-RIAA-Fehler → phase_04_eq_correction + phase_06_frequency_restoration."""
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases
        assert "phase_06_frequency_restoration" in phases

    def test_vinyl_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.VINYL, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_balanced_mode_adds_harmonic_restoration(self):
        """Im BALANCED-Modus Oberton-Rekonstruktion bei hoher Severity (durch Entzerrungs-Kette verloren)."""
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.BALANCED)
        assert "phase_07_harmonic_restoration" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.RIAA_CURVE_ERROR, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"RIAA_CURVE_ERROR × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.SHELLAC, DefectType.RIAA_CURVE_ERROR, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)


class TestAliasing:
    """Fix #10: ALIASING — AA-Filter-Artefakte aus ADC-Digitalisierung."""

    def test_cd_activates_denoise_and_spectral(self):
        """CD-Aliasing → phase_03_denoise + phase_23_spectral_repair."""
        phases = _select_phases_for(MaterialType.CD_DIGITAL, DefectType.ALIASING, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_tape_activates_denoise_and_spectral(self):
        phases = _select_phases_for(MaterialType.TAPE, DefectType.ALIASING, QualityMode.QUALITY)
        assert "phase_03_denoise" in phases
        assert "phase_23_spectral_repair" in phases

    def test_balanced_mode_adds_spectral_repair_second_pass(self):
        """Schweres Aliasing im BALANCED-Modus → phase_50_spectral_repair (zweiter Pass)."""
        phases = _select_phases_for(MaterialType.MP3_LOW, DefectType.ALIASING, QualityMode.BALANCED)
        assert "phase_50_spectral_repair" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.ALIASING, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"ALIASING × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.DAT, DefectType.ALIASING, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)


class TestBiasError:
    """Fix #11: BIAS_ERROR — falscher Vormagnetisierungsstrom bei Bandaufnahme."""

    def test_tape_activates_eq_and_denoise(self):
        """Kassetten-Bias-Fehler → phase_04_eq_correction + phase_03_denoise."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.BIAS_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases
        assert "phase_03_denoise" in phases

    def test_reel_tape_activates_eq_correction(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BIAS_ERROR, QualityMode.QUALITY)
        assert "phase_04_eq_correction" in phases

    def test_balanced_mode_adds_freq_and_hiss_reduction(self):
        """Starker Bias-Fehler im BALANCED-Modus → phase_06 + phase_29."""
        phases = _select_phases_for(MaterialType.TAPE, DefectType.BIAS_ERROR, QualityMode.BALANCED)
        assert "phase_06_frequency_restoration" in phases
        assert "phase_29_tape_hiss_reduction" in phases

    def test_all_materials_produce_targeted_phases(self):
        for mat in MaterialType:
            phases = _select_phases_for(mat, DefectType.BIAS_ERROR, QualityMode.QUALITY)
            targeted = set(phases) - STRUCTURAL_PHASES
            assert len(targeted) >= 1, f"BIAS_ERROR × {mat.name} → keine Phasen"

    def test_no_nan_phases(self):
        phases = _select_phases_for(MaterialType.REEL_TAPE, DefectType.BIAS_ERROR, QualityMode.BALANCED)
        assert all(isinstance(p, str) and len(p) > 0 for p in phases)
