"""
Aurik 6.0 – SOTA-Batch-Audioverarbeitung

Dieses Skript verarbeitet Audiodateien im Batch nach SOTA-Standards, nutzt Policy-Engine, Audit-Logging und konsistente Begriffe gemäß Dokumentation.
Alle Workflows, Begriffe und Logs sind dokumentationskonform.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf

from backend.batch_run import run_batch


def process_audio_file(
    job_id,
    file_path,
    sr,
    dsp_only,
    qa_only,
    no_parallel,
    result_dir,
    policy_yaml=None,
    config_path=None,
):
    logger.info(f"[SOTA] Starte Job {job_id} für Datei {file_path}")
    logger.debug("[DEBUG] Vor Audio-Laden")
    try:
        from backend.file_import import load_audio_file

        info = load_audio_file(file_path) or {}  # None-safe für mypy union-attr
        if info.get("error"):
            logger.warning(f"[SOTA][Fehler] {info.get('error')}")
            return
        x = info.get("audio")
        file_sr = info.get("sr")
        logger.info(
            f"[SOTA][Medium] Format: {info.get('format')}, Kanäle: {info.get('channels')}, Sample-Rate: {file_sr}, Dauer: {info.get('duration', 0):.2f}s"
        )
        logger.info(f"[SOTA][Tonträger] Heuristik: {info.get('carrier_heuristic', 'Unbekannt')}")
        logger.info(
            f"[SOTA][Tonträger] Forensik: {info.get('carrier_forensic', 'Unbekannt')} (Score: {info.get('carrier_forensic_score', 0)})"
        )
        logger.info(
            f"[SOTA][Tonträger] ML: {info.get('carrier_ml', 'Unbekannt')} (Confidence: {info.get('carrier_ml_confidence', 0.0):.2f})"
        )
        if info.get("carrier_ml_probas"):
            logger.info(f"[SOTA][ML-Probas] {info.get('carrier_ml_probas')}")
        if info.get("carrier_ml_explain"):
            logger.info(f"[SOTA][ML-Explain] {info.get('carrier_ml_explain')}")
        if info.get("carrier_forensic_features"):
            logger.info(f"[SOTA][Forensik-Features] {info.get('carrier_forensic_features')}")
        if info.get("meta"):
            logger.info(f"[SOTA][Metadaten] {info.get('meta')}")
    except Exception as e:
        logger.warning(f"[SOTA][Fehler] Audio-Import fehlgeschlagen: {e}")
        return
    logger.debug(f"[DEBUG] Nach Audio-Laden, file_sr={file_sr}")
    if file_sr != sr:
        logger.warning(f"[SOTA][Warnung] Resampling von {file_sr} auf {sr} nicht implementiert, benutze Original-SR.")
    logger.debug("[DEBUG] Vor Konfig-Laden")
    config = None
    if config_path:
        with open(config_path) as f:
            try:
                config = json.loads(f.read())
                logger.debug(f"[DEBUG] Konfig geladen: {config}")
            except Exception as ex:
                logger.debug(f"[DEBUG] Fehler beim Laden der Konfig: {ex}")
                config = None
    logger.debug("[DEBUG] Vor Ergebnissammlung")
    result = {
        "job_id": job_id,
        "file": file_path,
        "sr": file_sr,
        "dsp_only": dsp_only,
        "qa_only": qa_only,
        "no_parallel": no_parallel,
        "config": config,
    }
    try:
        logger.debug("[DEBUG] Vor run_batch")
        logs = run_batch(
            input_length=len(x),
            sr=file_sr,
            dsp_only=dsp_only,
            qa_only=qa_only,
            no_parallel=no_parallel,
            policy_yaml=policy_yaml,
            log_policy=True,
            log_qa=True,
        )
        logger.debug("[DEBUG] Nach run_batch")
        result["status"] = "ok"
        # Policy- und QA-Logs speichern
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"policylog_{job_id}.json"), "w") as f:
            json.dump(logs.get("policy_log", []), f)
        with open(os.path.join(result_dir, f"qalog_{job_id}.json"), "w") as f:
            json.dump(logs.get("qa_log", []), f)
    except Exception as e:
        logger.debug(f"[DEBUG] Fehler in run_batch oder Log-Speicherung: {e}")
        result["status"] = "error"
        result["error"] = str(e)
    logger.debug("[DEBUG] Vor Ergebnis speichern")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"result_{job_id}.json")
    logger.debug(f"[DEBUG] Schreibe Ergebnisdatei: {result_path}")
    if os.path.exists(result_path):
        os.remove(result_path)
    with open(result_path, "w") as f:
        json.dump(result, f)
    logger.debug(f"[DEBUG] Ergebnisdatei geschrieben: {result_path}")
    logger.info(f"[SOTA] Job {job_id} abgeschlossen.")


def main():
    # Beispiel: Liste von Audiodateien (hier Dummy-Dateien anlegen, falls nicht vorhanden)
    audio_files = ["test1.wav", "test2.wav"]
    for fname in audio_files:
        if not os.path.exists(fname):
            sf.write(fname, np.random.randn(10000), 16000)
    # Policy- und Config-Dateien pro Job
    policy_path = "policy_test.yaml"
    config1 = {"dsp": {"n_fft": 2048, "hop": 512}, "qa": {"threshold": 0.95}}
    config2 = {"dsp": {"n_fft": 4096, "hop": 1024}, "qa": {"threshold": 0.90}}
    config_paths = ["config_test1.json", "config_test2.json"]
    # Schreibe die Konfigurationsdateien exakt per Python
    with open("config_test1.json", "w") as f:
        import json

        f.write(json.dumps(config1, separators=(",", ":")))
    with open("config_test2.json", "w") as f:
        f.write(json.dumps(config2, separators=(",", ":")))
    jobs = [
        {
            "job_id": i + 1,
            "file_path": fname,
            "sr": 16000,
            "dsp_only": False,
            "qa_only": False,
            "no_parallel": False,
            "result_dir": "results_sota",
            "policy_yaml": policy_path,
            "config_path": config_paths[i],
        }
        for i, fname in enumerate(audio_files)
    ]
    # Debug: Nur einen Job direkt ausführen, um Fehler zu sehen
    process_audio_file(**jobs[0])
    logger.info("[SOTA] Einzelner Audio-Job abgeschlossen.")


if __name__ == "__main__":
    main()
