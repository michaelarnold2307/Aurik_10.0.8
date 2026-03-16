import logging
"""
Minimal-API für die Integration des SOTA-Batch-Workflows in ein Frontend (z.B. Web-UI, Desktop-GUI)
Bietet REST-Endpunkte für Start, Status, Ergebnis und Audit-Report.
"""

import os
import threading
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify
import soundfile as sf

from Aurik_Standalone.dsp.dsp_decision_logic import DSPDecisionLogic

logger = logging.getLogger(__name__)

app = Flask(__name__)

AUDIO_IN_DIR = "input_audio/"
AUDIO_OUT_DIR = "output_audio/"
CONFIG = "config_dsp_chain_example.yaml"
BATCH_STATUS: Dict[str, Any] = {
    "running": False,
    "progress": 0,
    "total": 0,
    "last_file": None,
}


def batch_worker() -> None:
    logic = DSPDecisionLogic(config_path=CONFIG)
    files = [f for f in os.listdir(AUDIO_IN_DIR) if f.lower().endswith((".wav", ".flac"))]
    BATCH_STATUS["total"] = len(files)
    BATCH_STATUS["running"] = True
    for idx, fname in enumerate(files):
        in_path = os.path.join(AUDIO_IN_DIR, fname)
        out_path = os.path.join(AUDIO_OUT_DIR, fname)
        BATCH_STATUS["last_file"] = fname
        try:
            audio, sr = sf.read(in_path)
            logic.output_path_hint = out_path
            result = logic.process(audio, sr)
            sf.write(out_path, result, sr)
        except Exception as e:
            logger.error(f"[BatchAPI] Fehler bei {in_path}: {e}")
        BATCH_STATUS["progress"] = idx + 1
    BATCH_STATUS["running"] = False


@app.route("/batch/start", methods=["POST"])
def start_batch():
    if BATCH_STATUS["running"]:
        return jsonify({"status": "already running"}), 409
    threading.Thread(target=batch_worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/batch/status", methods=["GET"])
def batch_status():
    return jsonify(BATCH_STATUS)


@app.route("/batch/result", methods=["GET"])
def batch_result():
    files = [f for f in os.listdir(AUDIO_OUT_DIR) if f.lower().endswith((".wav", ".flac"))]
    return jsonify({"files": files})


@app.route("/batch/audit", methods=["GET"])
def batch_audit():
    audits = [f for f in os.listdir(AUDIO_OUT_DIR) if f.endswith("_audit.json")]
    return jsonify({"audits": audits})


if __name__ == "__main__":
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
