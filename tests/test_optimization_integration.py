#!/usr/bin/env python3
"""Smoke tests for the canonical CLI entrypoint replacing the legacy orchestrator script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
    assert "run_pre_analysis(" in source
    assert "denker.denke(" in source


def test_cli_uses_frontend_export_contract():
    """CLI muss denselben Bridge-Exportvertrag wie das Frontend nutzen."""
    source = Path("cli/aurik_cli.py").read_text(encoding="utf-8")
    assert "export_guard(" in source
    assert "validate_export_quality(" in source
    assert "build_export_quality_gate_payload(" in source
    assert "get_audio_exporter_class(" in source
    assert "os.replace(tmp_path, out_path)" in source
    assert "sf.write(output_path" not in source


def test_cli_export_helper_uses_audio_exporter_and_quality_payload(monkeypatch, tmp_path):
    """CLI-Export muss AudioExporter, Export-Guard und Gate-Payload wie das Frontend nutzen."""
    from cli import aurik_cli

    calls: dict[str, object] = {}

    class FakeAudioExporter:
        FORMATS = {".wav": {}}

        def export(self, audio, sr, output_path, **kwargs):
            calls["audio"] = np.asarray(audio)
            calls["sr"] = sr
            calls["output_path"] = output_path
            calls["kwargs"] = kwargs
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

    def fake_export_guard(audio):
        calls["guard_count"] = int(calls.get("guard_count", 0)) + 1
        return np.clip(np.nan_to_num(np.asarray(audio, dtype=np.float32)), -1.0, 1.0)

    monkeypatch.setattr(aurik_cli, "get_audio_exporter_class", lambda: FakeAudioExporter)
    monkeypatch.setattr(aurik_cli, "export_guard", fake_export_guard)
    monkeypatch.setattr(aurik_cli, "validate_export_quality", lambda result: (False, ["gate-warning"]))
    monkeypatch.setattr(
        aurik_cli,
        "build_export_quality_gate_payload",
        lambda result: {
            "passed": False,
            "degradation_status": "degraded",
            "fail_reason": "gate-warning",
            "recovery_attempted": True,
            "best_possible_reached": True,
            "fallback_quality_floor": {"status": "recovered"},
        },
    )

    result = SimpleNamespace(audio=np.zeros((2, 8), dtype=np.float32), metadata={})
    passed, warnings, payload = aurik_cli._export_audio_frontend_parity(
        result,
        str(tmp_path / "out.wav"),
        np.array([[np.nan, 2.0, -2.0, 0.0], [0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        aurik_cli.logging.getLogger("test_cli_export"),
    )

    assert passed is False
    assert warnings == ["gate-warning"]
    assert payload["degradation_status"] == "degraded"
    assert calls["guard_count"] == 2
    assert calls["sr"] == 48_000
    assert calls["audio"].shape == (4, 2)
    kwargs = calls["kwargs"]
    assert kwargs["bit_depth"] == 24
    assert kwargs["quality"] == "veryhigh"
    assert kwargs["normalize"] is False
    assert kwargs["metadata"]["quality_gate_degradation_status"] == "degraded"
    assert result.metadata["export_quality_gate_failed"] is True
