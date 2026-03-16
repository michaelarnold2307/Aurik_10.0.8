"""
Aurik 6.0 – Testintegration und Golden Sample Validierung
Automatisierte Validierung der Pipeline gegen Golden Samples aus der Testdatenbank.
"""

import os

import soundfile as sf

from backend.core.export_workflow import export_audio, export_audit_log
from backend.core.pipeline_main import AurikMainPipeline


def run_golden_sample_test(
    audio_path: str,
    reference_path: str,
    sample_rate: int,
    policy_template=None,
    config=None,
):
    # Audio und Referenz laden
    audio, sr = sf.read(audio_path)
    reference, sr_ref = sf.read(reference_path)
    assert sr == sr_ref == sample_rate, "Sample-Rates stimmen nicht überein!"
    # Pipeline ausführen
    pipeline = AurikMainPipeline(policy_template=policy_template, config=config)
    restored, audit_log = pipeline.process(audio, sr, reference)
    # Export
    result_name = f"restored_{os.path.basename(audio_path)}"
    audit_name = f"audit_{os.path.splitext(os.path.basename(audio_path))[0]}.ndjson"
    export_audio(restored, sr, result_name)
    export_audit_log(audit_log, audit_name)
    # Ergebnis zurückgeben
    return restored, audit_log


# Beispiel für die Nutzung:
# run_golden_sample_test("tests/golden_sample_01.wav", "tests/golden_sample_01_ref.wav", 44100)
