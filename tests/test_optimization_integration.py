#!/usr/bin/env python3
"""Smoke tests for the canonical CLI entrypoint replacing the legacy orchestrator script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_optimization_integration():
    """Canonical CLI exists and exposes the current AurikDenker-based interface."""
    cli_path = Path("cli/aurik_cli.py")
    assert cli_path.exists(), "Kanonischer CLI-Einstieg cli/aurik_cli.py fehlt"
    assert not Path("orchestrator_and_cli.py").exists(), (
        "Legacy-Skript orchestrator_and_cli.py sollte nicht mehr verwendet werden"
    )

    result = subprocess.run(
        [sys.executable, "-m", "cli.aurik_cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI --help schlug fehl: {result.stderr or result.stdout}"
    help_text = (result.stdout or "") + (result.stderr or "")
    assert "Restoration" in help_text
    assert "Studio 2026" in help_text

    source = cli_path.read_text(encoding="utf-8")
    assert "AurikDenker" in source or "get_aurik_denker" in source
    assert "orchestrator_and_cli.py" not in source
