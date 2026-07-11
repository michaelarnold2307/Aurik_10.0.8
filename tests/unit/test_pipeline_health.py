"""
test_pipeline_health.py — Pre-Pipeline Health Check Tests
==========================================================

Verifiziert, dass run_health_checks() funktioniert und alle C1-C5 Checks durchführt.
"""

from __future__ import annotations



class TestPipelineHealthCheck:
    """Pre-Pipeline Health Verification."""

    def test_01_health_check_runs_without_crash(self):
        """run_health_checks() läuft ohne Exception."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks(audio_duration_s=60.0)
        assert report is not None
        assert len(report.checks) >= 4, f"Nur {len(report.checks)} Checks, erwartet >=4"

    def test_02_all_checks_have_results(self):
        """Jeder Check hat name, passed, duration_ms."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks()
        for check in report.checks:
            assert check.name, "Check ohne Namen"
            assert check.duration_ms >= 0, f"{check.name}: duration_ms negativ"
            assert isinstance(check.passed, bool), f"{check.name}: passed kein bool"

    def test_03_summary_includes_all_checks(self):
        """Summary enthält alle Check-Namen."""
        from backend.core.pipeline_health_check import run_health_checks

        report = run_health_checks()
        summary = report.summary()
        for check in report.checks:
            assert check.name in summary, f"{check.name} fehlt in Summary"

    def test_04_numpy_scipy_available(self):
        """C1: numpy und scipy sind verfügbar."""
        import numpy as np
        from scipy import signal

        assert np is not None
        assert signal is not None

    def test_05_dsp_modules_importable(self):
        """C2: Kritische DSP-Module sind importierbar."""
        from backend.core.audio_utils import compute_gated_rms_linear

        assert callable(compute_gated_rms_linear)

    def test_06_configuration_files_exist(self):
        """C4: Erforderliche Konfigurationsdateien existieren."""
        import os

        assert os.path.exists("pytest.ini"), "pytest.ini fehlt"
        assert os.path.exists(".github/specs/01_musical_goals.md"), "Spec 01 fehlt"
