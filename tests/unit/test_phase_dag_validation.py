"""
Tests für PhaseDAG / validate_phase_order (§7.5a).
"""


class TestPhaseDagImport:
    def test_import_ok(self):
        from backend.core.phase_dag import HARD_BEFORE_CONSTRAINTS, validate_phase_order

        assert callable(validate_phase_order)
        assert isinstance(HARD_BEFORE_CONSTRAINTS, list)
        assert len(HARD_BEFORE_CONSTRAINTS) > 0


class TestValidatePhaseOrder:
    def test_empty_list_no_violations(self):
        from backend.core.phase_dag import validate_phase_order

        violations = validate_phase_order([])
        assert violations == []

    def test_correct_order_no_violations(self):
        """Korrekte Reihenfolge: phase_03 vor phase_07 → keine Verletzung."""
        from backend.core.phase_dag import validate_phase_order

        phases = [
            "phase_01_click_removal",
            "phase_03_denoise",
            "phase_06_bw_extension",
            "phase_07_harmonic_restoration",
        ]
        violations = validate_phase_order(phases)
        assert violations == [], f"Unerwartete Verletzungen: {violations}"

    def test_wrong_order_detected(self):
        """phase_07 vor phase_03 muss als Verletzung erkannt werden."""
        from backend.core.phase_dag import validate_phase_order

        phases = [
            "phase_07_harmonic_restoration",
            "phase_03_denoise",
        ]
        violations = validate_phase_order(phases)
        assert len(violations) > 0, "Reihenfolge-Verletzung sollte erkannt werden"
        assert any("phase_07" in v or "phase_03" in v for v in violations)

    def test_phase_29_before_phase_07(self):
        """phase_29 (tape hiss) MUSS vor phase_07 sein (§7.5a kritische Kette)."""
        from backend.core.phase_dag import validate_phase_order

        # Korrekte Reihenfolge
        correct = ["phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"]
        assert validate_phase_order(correct) == []

        # Falsche Reihenfolge
        wrong = ["phase_07_harmonic_restoration", "phase_29_tape_hiss_reduction"]
        violations = validate_phase_order(wrong)
        assert len(violations) > 0

    def test_only_one_phase_no_violations(self):
        """Einzelne Phase → keine Constraint-Verletzung möglich."""
        from backend.core.phase_dag import validate_phase_order

        violations = validate_phase_order(["phase_07_harmonic_restoration"])
        assert violations == []

    def test_short_form_phase_ids(self):
        """Kurzform phase_03 wird korrekt normiert."""
        from backend.core.phase_dag import validate_phase_order

        phases = ["phase_03", "phase_07"]
        violations = validate_phase_order(phases)
        # Korrekte Reihenfolge → keine Verletzung
        assert violations == []

    def test_returns_list_of_strings(self):
        from backend.core.phase_dag import validate_phase_order

        result = validate_phase_order(["phase_07_harmonic_restoration", "phase_03_denoise"])
        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, str)


class TestCheckConflict:
    def test_no_conflict_for_unrelated_phases(self):
        from backend.core.phase_dag import check_conflict

        result = check_conflict("phase_01_click_removal", "phase_09_crackle_removal")
        assert result is None

    def test_known_conflict_detected(self):
        """phase_06 und phase_23 sind im CONFLICT."""
        from backend.core.phase_dag import check_conflict

        result = check_conflict("phase_06_bw_extension", "phase_23_audio_sr_upsampling")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_conflict_bidirectional(self):
        """Konflikt gilt in beide Richtungen."""
        from backend.core.phase_dag import check_conflict

        a_b = check_conflict("phase_06_bw_extension", "phase_23_audio_sr_upsampling")
        b_a = check_conflict("phase_23_audio_sr_upsampling", "phase_06_bw_extension")
        assert a_b is not None
        assert b_a is not None


class TestParallelClass:
    def test_stereo_phases_class_a(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_14_stereo_width") == "A"
        assert get_parallel_class("phase_15_stereo_field_repair") == "A"
        assert get_parallel_class("phase_25_azimuth_correction") == "A"

    def test_local_defect_phases_class_b(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_09_crackle_removal") == "B"
        assert get_parallel_class("phase_24_dropout_repair") == "B"

    def test_unknown_phase_none(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_99_unknown") is None
