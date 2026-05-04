#!/usr/bin/env python3
"""
Aurik 9 — Kontinuierliche Tiefenanalyse & Qualitäts-Monitoring
=================================================================

Führt eine vollständige Restaurierung mit Phase-weisen Qualitäts-Checkpoints durch,
monitort Musical Goals, HPI, Artefakt-Freiheit und erkennt Anomalien automatisch.

Wird als Echtzeit-Dashboard mit Alerts ausgeführt.

Usage:
    python scripts/continuous_deep_analysis.py [--audio <path>] [--mode restoration|studio] [--realtime]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Setup paths
_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

# Import Pegelexplosion-Detektor
try:
    from scripts.pegelexplosion_detector import PegelexplosionDetector
except ImportError:
    PegelexplosionDetector = None

# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class PhaseCheckpoint:
    """Checkpoint-Daten nach einer Phase."""

    phase_id: str
    wall_time_s: float
    musical_goals: dict[str, float]
    hpi_score: float | None
    artifact_freedom: float | None
    carrier_recovery_ratio: float | None
    noise_floor_db: float | None
    defects_remaining: int | None
    anomalies: list[str]
    pegelexplosion_detected: bool = False
    pegelexplosion_severity: str = "none"
    pegelexplosion_cause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["musical_goals"] = {
            k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in self.musical_goals.items()
        }
        return d


# ============================================================================
# ANALYZER
# ============================================================================


class ContinuousDeepAnalyzer:
    """Führt Tiefenanalyse während der Restaurierung durch."""

    def __init__(self, realtime: bool = True):
        self.realtime = realtime
        self.logger = logging.getLogger(__name__)
        self.checkpoints: list[PhaseCheckpoint] = []
        self.anomalies_detected: list[str] = []
        self.pegelexplosion_detector = PegelexplosionDetector() if PegelexplosionDetector else None
        self._last_phase_audio = None

    def run_analysis(
        self,
        audio_path: str,
        sr: int = 48000,
        mode: str = "restoration",
        output_dir: str = "analysis_results",
    ) -> dict[str, Any]:
        """
        Führt kontinuierliche Tiefenanalyse durch.

        Args:
            audio_path: Pfad zur Audio-Datei
            sr: Sample Rate (Standard: 48000 Hz)
            mode: "restoration" oder "studio_2026"
            output_dir: Verzeichnis für Analyse-Ergebnisse

        Returns:
            Ergebnis-Dictionary mit allen Checkpoints und Anomalien
        """
        _start_t = time.monotonic()
        self.logger.info("=" * 80)
        self.logger.info("AURIK 9 — KONTINUIERLICHE TIEFENANALYSE")
        self.logger.info(f"Audio: {audio_path}")
        self.logger.info(f"Mode: {mode}")
        self.logger.info(f"SR: {sr} Hz")
        self.logger.info("=" * 80)

        # 1. Audio laden
        try:
            from backend.file_import import load_audio_file

            result = load_audio_file(audio_path)
            if result is None or result.get("error"):
                self.logger.error(f"✗ Audio-Import fehlgeschlagen: {result.get('error') if result else 'None'}")
                return {"error": str(result.get("error") if result else "Unknown"), "checkpoints": []}
            audio = result["audio"]
            sr_imported = result["sr"]
            self.logger.info(f"✓ Audio geladen: {len(audio) / sr_imported:.1f}s @ {sr_imported} Hz")
        except Exception as e:
            self.logger.error(f"✗ Audio-Import fehlgeschlagen: {e}")
            return {"error": str(e), "checkpoints": []}

        # 2. Pre-Analyse durchführen
        try:
            from backend.core.pre_analysis import run_pre_analysis

            pre_result = run_pre_analysis(
                audio,
                sr_imported,
                file_path=str(Path(audio_path).resolve()),
                store_in_bridge_cache=False,
            )
            _medium_label = getattr(pre_result.medium, "primary_material", None) or getattr(
                pre_result.medium, "chain_label", None
            )
            _era_label = getattr(pre_result.era, "decade", None) or getattr(pre_result.era, "year_estimate", None)
            _rest_score = getattr(pre_result.restorability, "restorability_score", None)
            _defect_count = len(getattr(pre_result.defects, "scores", {}) or {}) if pre_result.defects else 0
            self.logger.info("✓ Pre-Analyse komplett:")
            self.logger.info(f"  - Material: {_medium_label if _medium_label is not None else 'unknown'}")
            self.logger.info(f"  - Era: {_era_label if _era_label is not None else 'unknown'}")
            self.logger.info(
                f"  - Restorability: {_rest_score:.1f}%"
                if isinstance(_rest_score, (int, float))
                else "  - Restorability: unknown"
            )
            self.logger.info(f"  - Defekte gefunden: {_defect_count}")
        except Exception as e:
            self.logger.warning(f"Pre-Analyse fehlgeschlagen (nicht kritisch): {e}")
            pre_result = None

        # 3. Restaurierung mit Monitoring durchführen
        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3

            restorer = UnifiedRestorerV3(quality_mode="maximum", monitor_phases=True)

            # Restaurierung durchführen
            is_studio_2026 = mode == "studio_2026"
            restoration_result = restorer.restore(
                audio=audio,
                sample_rate=sr_imported,
                is_studio_2026=is_studio_2026,
                pre_analysis_result=pre_result,
                enable_debug_trace=True,
            )

            self._collect_checkpoints_from_restore_result(restoration_result, pre_result)

            self.logger.info(f"✓ Restaurierung komplett (Dauer: {time.monotonic() - _start_t:.1f}s)")
            self.logger.info(f"  - HPI: {restoration_result.metadata.get('hpi_score', 'N/A')}")
            self.logger.info(
                f"  - Artefakt-Freiheit: {restoration_result.metadata.get('artifact_freedom_score', 'N/A')}"
            )

        except Exception as e:
            self.logger.error(f"✗ Restaurierung fehlgeschlagen: {e}", exc_info=True)
            self.anomalies_detected.append(f"Restoration failed: {e}")

        # 4. Ergebnisse speichern
        os.makedirs(output_dir, exist_ok=True)
        result_dict = {
            "wall_time_s": time.monotonic() - _start_t,
            "audio_path": audio_path,
            "mode": mode,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "anomalies": self.anomalies_detected,
            "summary": self._generate_summary(),
        }

        result_file = Path(output_dir) / f"analysis_{Path(audio_path).stem}_{mode}_{int(time.time())}.json"
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        self.logger.info(f"✓ Ergebnisse gespeichert: {result_file}")

        return result_dict

    def _collect_checkpoints_from_restore_result(self, restoration_result: Any, pre_result: Any) -> None:
        """Build checkpoints from UV3 debug-trace entries after restoration."""
        metadata = getattr(restoration_result, "metadata", {}) or {}
        entries = metadata.get("pmgg_log_entries") or []
        if not isinstance(entries, list):
            return

        final_hpi = self._metric_to_float(metadata.get("hpi_score"), ("hpi", "hpi_score", "score", "value"))
        final_afg = self._metric_to_float(
            metadata.get("artifact_freedom")
            if metadata.get("artifact_freedom") is not None
            else metadata.get("artifact_freedom_score"),
            ("artifact_freedom", "artifact_freedom_score", "score", "value"),
        )
        final_ccr = self._metric_to_float(
            metadata.get("carrier_chain_recovery_ratio"),
            ("carrier_chain_recovery_ratio", "score", "value"),
        )

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            phase_id = str(entry.get("phase_id") or "unknown_phase")
            scores_before = entry.get("scores_before") if isinstance(entry.get("scores_before"), dict) else {}
            scores_after = entry.get("scores_after") if isinstance(entry.get("scores_after"), dict) else {}
            anomalies = self._check_anomalies_from_scores(
                phase_id,
                scores_before,
                scores_after,
                final_hpi,
                final_afg,
                pre_result,
                str(entry.get("action") or ""),
            )
            cp = PhaseCheckpoint(
                phase_id=phase_id,
                wall_time_s=float(entry.get("timestamp") or time.time()),
                musical_goals={k: float(v) for k, v in scores_after.items() if isinstance(v, (int, float))},
                hpi_score=final_hpi,
                artifact_freedom=final_afg,
                carrier_recovery_ratio=final_ccr,
                noise_floor_db=None,
                defects_remaining=None,
                anomalies=anomalies,
            )
            self.checkpoints.append(cp)

    def _check_anomalies_from_scores(
        self,
        phase_id: str,
        scores_before: dict[str, Any],
        scores_after: dict[str, Any],
        hpi: Any,
        afg: Any,
        pre_result: Any,
        action: str,
    ) -> list[str]:
        """Detect anomalies from PMGG debug-trace score snapshots."""
        anomalies: list[str] = []
        hpi_value = self._metric_to_float(hpi, ("hpi", "hpi_score", "score", "value"))
        afg_value = self._metric_to_float(afg, ("artifact_freedom", "artifact_freedom_score", "score", "value"))
        for goal, after in scores_after.items():
            before = scores_before.get(goal)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                delta = float(after) - float(before)
                if delta < -0.10:
                    anomalies.append(f"{goal}: {float(before):.2f} → {float(after):.2f} (Δ={delta:.2f})")

        if str(action).startswith("best_effort"):
            anomalies.append(f"PMGG best_effort: {action}")

        if hpi_value is not None and hpi_value < 0.1:
            anomalies.append(f"HPI kritisch niedrig: {hpi_value:.3f}")

        if afg_value is not None and afg_value < 0.85:
            anomalies.append(f"Artefakt-Freiheit kritisch: {afg_value:.3f}")

        if pre_result and hasattr(pre_result, "noise_floor_estimate"):
            if pre_result.noise_floor_estimate > -50.0:
                anomalies.append(f"Rauschboden hoch: {pre_result.noise_floor_estimate:.1f} dBFS")

        return anomalies

    @staticmethod
    def _metric_to_float(metric: Any, preferred_keys: tuple[str, ...] = ("score", "value")) -> float | None:
        """Coerce scalar-like metric values (including dict payloads) to float."""
        if isinstance(metric, (int, float)):
            return float(metric)
        if isinstance(metric, dict):
            for key in preferred_keys:
                value = metric.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for value in metric.values():
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def _create_checkpoint(self, phase_id: str, restorer: Any, pre_result: Any) -> PhaseCheckpoint:
        """Erstellt einen Checkpoint nach einer Phase."""
        # Musical Goals auslesen
        goals: dict[str, float] = {}
        if hasattr(restorer, "_musical_goals_results"):
            goals = dict(restorer._musical_goals_results)

        # HPI, AFG, etc. auslesen
        hpi = None
        afg = None
        ccr = None
        noise_db = None
        defects_left = None

        if hasattr(restorer, "_hpi_score"):
            hpi = float(restorer._hpi_score)
        if hasattr(restorer, "_artifact_freedom_score"):
            afg = float(restorer._artifact_freedom_score)
        if hasattr(restorer, "_carrier_chain_recovery_ratio"):
            ccr = float(restorer._carrier_chain_recovery_ratio)

        # Anomalien prüfen
        anomalies = self._check_anomalies(phase_id, goals, hpi, afg, pre_result)

        return PhaseCheckpoint(
            phase_id=phase_id,
            wall_time_s=time.monotonic(),
            musical_goals=goals,
            hpi_score=hpi,
            artifact_freedom=afg,
            carrier_recovery_ratio=ccr,
            noise_floor_db=noise_db,
            defects_remaining=defects_left,
            anomalies=anomalies,
        )

    def _check_anomalies(
        self, phase_id: str, goals: dict[str, float], hpi: float | None, afg: float | None, pre_result: Any
    ) -> list[str]:
        """Erkennt Anomalien in Phase-Ergebnissen."""
        anomalies: list[str] = []

        # Musical Goals Regressions
        if self.checkpoints:
            last_cp = self.checkpoints[-1]
            for goal, score in goals.items():
                last_score = last_cp.musical_goals.get(goal, 1.0)
                delta = score - last_score
                if delta < -0.10:  # > 10% Regression
                    anomalies.append(f"{goal}: {last_score:.2f} → {score:.2f} (Δ={delta:.2f})")

        # HPI Drop
        if hpi is not None and hpi < 0.1:
            anomalies.append(f"HPI kritisch niedrig: {hpi:.3f}")

        # Artefakt-Explosion
        if afg is not None and afg < 0.85:
            anomalies.append(f"Artefakt-Freiheit kritisch: {afg:.3f}")

        # Rauschboden steigende
        if pre_result and hasattr(pre_result, "noise_floor_estimate"):
            if pre_result.noise_floor_estimate > -50.0:
                anomalies.append(f"Rauschboden hoch: {pre_result.noise_floor_estimate:.1f} dBFS")

        return anomalies

    def _print_checkpoint_summary(self, cp: PhaseCheckpoint) -> None:
        """Druckt Checkpoint als Tabelle."""
        self.logger.info(f"\n  Phase: {cp.phase_id}")
        self.logger.info("  - Musical Goals: ", extra={"no_newline": True})
        for goal, score in list(cp.musical_goals.items())[:3]:
            status = "✓" if score >= 0.80 else "✗" if score < 0.50 else "~"
            self.logger.info(f" {status}{goal}={score:.2f}", extra={"no_newline": True})
        self.logger.info("")
        if cp.hpi_score is not None:
            self.logger.info(f"  - HPI: {cp.hpi_score:.3f}")
        if cp.artifact_freedom is not None:
            self.logger.info(f"  - AFG: {cp.artifact_freedom:.3f}")
        if cp.anomalies:
            for anom in cp.anomalies:
                self.logger.info(f"  ⚠ ANOMALIE: {anom}")

    def _generate_summary(self) -> dict[str, Any]:
        """Generiert eine Zusammenfassung."""
        if not self.checkpoints:
            return {"status": "no_checkpoints"}

        last_cp = self.checkpoints[-1]
        p1_goals = ["natuerlichkeit", "authentizitaet"]
        p1_scores = [last_cp.musical_goals.get(g, 0.0) for g in p1_goals]
        p1_avg = np.mean(p1_scores) if p1_scores else 0.0

        return {
            "total_phases": len(self.checkpoints),
            "total_anomalies": len(self.anomalies_detected),
            "final_hpi": last_cp.hpi_score,
            "final_artifact_freedom": last_cp.artifact_freedom,
            "p1_avg_score": float(p1_avg),
            "quality_status": "EXCELLENT" if p1_avg >= 0.90 else "GOOD" if p1_avg >= 0.80 else "NEEDS_REVIEW",
        }


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Aurik 9 — Kontinuierliche Tiefenanalyse")
    parser.add_argument("--audio", type=str, default=None, help="Audio-Datei-Pfad")
    parser.add_argument(
        "--mode",
        type=str,
        default="restoration",
        choices=["restoration", "studio_2026"],
        help="Restaurierungs-Modus",
    )
    parser.add_argument("--realtime", action="store_true", help="Echtzeit-Dashboard aktivieren")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output-Verzeichnis")

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("analysis_runtime.log"),
        ],
    )

    # Standard-Audio wenn nicht angegeben
    if not args.audio:
        default_audio = (
            Path("test_audio") / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
        )
        if default_audio.exists():
            args.audio = str(default_audio)
        else:
            print("✗ Keine Audio-Datei angegeben und keine Standard-Datei gefunden")
            sys.exit(1)

    # Analyzer ausführen
    analyzer = ContinuousDeepAnalyzer(realtime=args.realtime)
    result = analyzer.run_analysis(
        audio_path=args.audio,
        mode=args.mode,
        output_dir=args.output_dir,
    )

    # Finale Ausgabe
    summary = result.get("summary", {})
    print("\n" + "=" * 80)
    print("ANALYSE ABSCHLUSS")
    print("=" * 80)
    print(f"Status: {summary.get('quality_status', 'UNKNOWN')}")
    print(f"P1 Durchschnitt: {summary.get('p1_avg_score', 0.0):.2f}")
    print(f"Anomalien erkannt: {len(result.get('anomalies', []))}")
    if result.get("anomalies"):
        print("\nAnomalien:")
        for anom in result["anomalies"]:
            print(f"  - {anom}")

    return 0 if summary.get("quality_status") == "EXCELLENT" else 1


if __name__ == "__main__":
    sys.exit(main())
