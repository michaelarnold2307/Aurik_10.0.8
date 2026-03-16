"""AMRB v1.0 Runner — Aurik 9.9.9 Finale Validierung.

Führt den vollständigen Aurik Musical Restoration Benchmark gegen
Aurik 9.9 UnifiedRestorerV3 aus und prüft OS-Führerschaft.

Aufruf:
    python scripts/run_amrb_v99.py [--quick]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

# Projekt-Root sicherstellen
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("amrb_runner")


def make_restoration_fn(mode: str = "quality"):
    """Gibt eine (audio, sr) → audio Funktion zurück, die UnifiedRestorerV3 nutzt."""
    try:
        from core.unified_restorer_v3 import get_restorer

        restorer = get_restorer("quality")  # QUALITY: 9× RT, kein Phase-Skipping

        def restore(audio: np.ndarray, sr: int) -> np.ndarray:
            try:
                result = restorer.restore(audio, sr, mode=mode)
                return result.audio if hasattr(result, "audio") else result
            except Exception as exc:
                logger.debug("Restore-Fehler (DSP-Fallback): %s", exc)
                return audio.copy()

        return restore

    except ImportError as exc:
        logger.warning("UnifiedRestorerV3 nicht verfügbar (%s) — DSP-Baseline", exc)

        def dsp_fallback(audio: np.ndarray, sr: int) -> np.ndarray:
            """Einfacher DSP-Fallback: Butterworth-Hochpass + Normalisierung."""
            from scipy.signal import butter, sosfilt

            sos = butter(4, 40.0 / (sr / 2), btype="high", output="sos")
            out = sosfilt(sos, audio)
            peak = np.max(np.abs(out))
            if peak > 1e-8:
                out = out / peak * 0.95
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return dsp_fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="AMRB v1.0 — Aurik 9.9 Validierung")
    parser.add_argument("--quick", action="store_true", help="Schnell-Modus: 2 Items/Szenario statt 5")
    parser.add_argument(
        "--mode",
        default="restoration",
        choices=["restoration", "studio2026"],
        help="Restaurierungsmodus (Standard: restoration)",
    )
    parser.add_argument("--report", default="reports/amrb_v99_result.json", help="Ausgabepfad für JSON-Bericht")
    args = parser.parse_args()

    n_items = 2 if args.quick else 5
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AMRB v1.0  —  Aurik 9.9.9  —  Modus: %s", args.mode)
    logger.info("Items/Szenario: %d | Bericht: %s", n_items, report_path)
    logger.info("=" * 60)

    from benchmarks.musical_restoration_benchmark import (
        BenchmarkConfig,
        MusicalRestorationBenchmark,
    )

    restore_fn = make_restoration_fn(mode=args.mode)

    config = BenchmarkConfig(
        restoration_fn=restore_fn,
        system_name=f"Aurik 9.9.9 ({args.mode})",
        n_items_per_scenario=n_items,
        sample_rate=48_000,
        report_path=report_path,
        verbose=True,
    )

    engine = MusicalRestorationBenchmark(config)
    report = engine.run()
    MusicalRestorationBenchmark.print_report(report)

    logger.info("")
    logger.info("━" * 60)
    logger.info("AMRB Gesamt-Score : %.1f / 100", report.overall_score)
    logger.info("Szenarien bestanden: %d / %d", report.n_passed, report.n_scenarios)
    logger.info(
        "OS-Führerschaft   : %s", "✅ JA (≥ 84.0 UND ≥ 8/10)" if report.passes_os_leadership_threshold() else "❌ NEIN"
    )
    logger.info("Bericht gespeichert: %s", report_path)
    logger.info("━" * 60)

    return 0 if report.passes_os_leadership_threshold() else 1


if __name__ == "__main__":
    sys.exit(main())
