#!/usr/bin/env python3
"""
Full-DEBUG restoration run — logs every event to console + file.
Usage: .venv_aurik/bin/python _debug_restoration_run.py
"""

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Enable full DEBUG logging BEFORE any aurik import ─────────────────────
_LOG_FILE = PROJECT_ROOT / "logs" / "debug_restoration_run.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s:%(lineno)d  %(message)s")

fh = logging.FileHandler(str(_LOG_FILE), mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
root.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)
root.addHandler(ch)

# Suppress extremely chatty 3rd-party loggers
for noisy in ("matplotlib", "PIL", "urllib3", "numba", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("debug_run")

_input_candidates = [
    os.environ.get("AURIK_DEBUG_INPUT", "").strip(),
    str(PROJECT_ROOT / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"),
    str(PROJECT_ROOT / "temp_repro" / "repro_input.mp3"),
]
INPUT_FILE = next((p for p in _input_candidates if p and os.path.exists(p)), _input_candidates[-1])
OUTPUT_FILE = str(PROJECT_ROOT / "output" / "Elke_Best_debug_restored.wav")
MODE = "Restoration"

logger.info("=" * 80)
logger.info("AURIK DEBUG RESTORATION RUN")
logger.info("  Input : %s", INPUT_FILE)
logger.info("  Output: %s", OUTPUT_FILE)
logger.info("  Mode  : %s", MODE)
logger.info("=" * 80)

# ── Import & run ──────────────────────────────────────────────────────────
from cli.aurik_cli import _TARGET_SR, _load_audio, _resample_to_48k
from denker.aurik_denker import get_aurik_denker

t0 = time.perf_counter()

audio_raw, sr_raw = _load_audio(INPUT_FILE)
logger.info("Loaded: shape=%s sr=%d dtype=%s", audio_raw.shape, sr_raw, audio_raw.dtype)

audio_48k = _resample_to_48k(audio_raw, sr_raw)
logger.info("Resampled to 48 kHz: shape=%s", audio_48k.shape)


def progress_cb(pct: int, msg: str, elapsed_s: float = 0.0):
    logger.info("PROGRESS %3d%% [%.1fs] %s", pct, elapsed_s, msg)


denker = get_aurik_denker()
logger.info("AurikDenker singleton obtained")

result = denker.denke(
    audio_48k,
    sr=_TARGET_SR,
    mode=MODE,
    no_rt_limit=True,
    input_path=os.path.abspath(INPUT_FILE),
    output_path=os.path.abspath(OUTPUT_FILE),
    progress_callback=progress_cb,
)

elapsed = time.perf_counter() - t0

logger.info("=" * 80)
logger.info("RESULT SUMMARY")
logger.info("  Material        : %s", result.material)
logger.info("  Quality Estimate: %.4f", result.quality_estimate)
logger.info("  RT Factor       : %.2f×", result.rt_factor)
logger.info("  Goals Passed    : %d / 14", result.goals_passed)
logger.info("  Phases Executed : %s", result.phases_executed)
logger.info("  Warnings        : %s", result.warnings)
logger.info("  Processing Note : %s", result.processing_note)

if hasattr(result, "musical_goals") and result.musical_goals:
    logger.info("  Musical Goals:")
    for g, v in sorted(result.musical_goals.items()):
        logger.info("    %-25s = %.4f", g, v)

if hasattr(result, "metadata") and result.metadata:
    for k, v in sorted(result.metadata.items()):
        logger.info("  metadata[%-30s] = %s", k + "]", repr(v)[:200])

logger.info("  Wall-clock time : %.1f s", elapsed)
logger.info("=" * 80)

# ── Save ──────────────────────────────────────────────────────────────────
import soundfile as sf

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
sf.write(OUTPUT_FILE, result.audio, _TARGET_SR, subtype="PCM_24")
logger.info("Saved: %s", OUTPUT_FILE)
logger.info("Full debug log: %s", str(_LOG_FILE.resolve()))
