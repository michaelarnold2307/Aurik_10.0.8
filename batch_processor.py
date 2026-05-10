#!/usr/bin/env python3
"""
Batch Processing System for Aurik 9.0
===============================================

Process multiple audio files in parallel with progress tracking.

Features:
- Multi-threaded processing
- Progress bar with ETA
- Memory-efficient streaming
- Resume capability
- Detailed logging

Author: Aurik 9.0 Team
Date: February 15, 2026
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None  # type: ignore[assignment]

from tqdm import tqdm

try:
    from backend.file_import import load_audio_file as _load_audio_file
except Exception:
    _load_audio_file = None  # type: ignore[assignment]

try:
    from backend.api.bridge import get_medium_type_enum as _get_medium_type_enum

    MaterialType = _get_medium_type_enum()
except Exception:
    MaterialType = None  # type: ignore[assignment,misc]
try:
    from denker.aurik_denker import get_aurik_denker as _get_aurik_denker
except ImportError:
    _get_aurik_denker = None  # type: ignore[assignment,misc]

try:
    from backend.core.album_consistency import get_album_consistency_pass
except Exception:
    get_album_consistency_pass = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("batch_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch audio processing with multi-threading."""

    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def __init__(self, output_dir: Path, workers: int = 4, resume: bool = False):
        """
        Initialize batch processor.

        Args:
            output_dir: Output directory for processed files
            workers: Number of parallel workers
            resume: Resume from previous batch (skip completed files)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.resume = resume

        # Resume state
        self.state_file = self.output_dir / ".batch_state.json"
        self.completed = self._load_state() if resume else set()

        logger.info("BatchProcessor initialized: %d workers, output: %s", workers, output_dir)
        if resume and self.completed:
            logger.info("Resume mode: %d files already processed", len(self.completed))

    def _load_state(self) -> set:
        """Load completed files from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state = json.load(f)
                return set(state.get("completed", []))
            except Exception:
                return set()
        return set()

    def _save_state(self):
        """Save completed files to state file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"completed": list(self.completed)}, f)

    def find_audio_files(self, input_paths: list[str]) -> list[Path]:
        """
        Find all audio files from input paths (files or directories).

        Args:
            input_paths: List of file paths or directories

        Returns:
            List of audio file paths
        """
        files = []

        for input_path_str in input_paths:
            input_path = Path(input_path_str)

            if input_path.is_file():
                if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    files.append(input_path)
            elif input_path.is_dir():
                # Recursively find audio files
                for ext in self.SUPPORTED_FORMATS:
                    files.extend(input_path.rglob(f"*{ext}"))

        # Remove duplicates
        files = list(set(files))

        # Filter out already completed (if resume)
        if self.resume:
            files = [f for f in files if str(f) not in self.completed]

        logger.info("Found %d audio files to process", len(files))
        return files

    def process_file(self, input_file: Path, config: dict | None = None) -> dict:
        """
        Process a single audio file.

        Args:
            input_file: Input audio file path
            config: Custom configuration

        Returns:
            Processing result dict
        """
        start_time = time.time()

        try:
            # Output filename
            output_file = self.output_dir / f"{input_file.stem}_restored{input_file.suffix}"

            # Process via AurikDenker (Pflicht-Einstiegspunkt §2.2)
            logger.info("Processing: %s", input_file.name)
            if _load_audio_file is None:
                raise RuntimeError(
                    "Audio-Importmodul nicht verfügbar. "
                    "Ursache: backend.file_import.load_audio_file konnte nicht geladen werden. "
                    "Lösung: Installation prüfen und Anwendung neu starten."
                )

            _res = _load_audio_file(str(input_file), do_carrier_analysis=False)
            if _res is None or _res.get("error"):
                raise RuntimeError(
                    f"Audio-Import fehlgeschlagen: {(_res or {}).get('error', 'Unbekannter Fehler')}. "
                    "Datei prüfen oder als WAV/FLAC neu exportieren."
                )
            audio_data = _res["audio"]
            file_sr = _res["sr"]

            if _get_aurik_denker is None:
                raise RuntimeError(
                    "AurikDenker nicht verfügbar — Backend-Import fehlgeschlagen. "
                    "Installation prüfen: denker/aurik_denker.py muss ladbar sein."
                )
            # §2.2 Canonical Singleton-Einstiegspunkt (No-Competing-Instances-Protokoll)
            denker = _get_aurik_denker()
            mode = (config or {}).pop("mode", "restoration")
            denker_result = denker.denke(
                audio_data,
                file_sr,
                mode=mode,
                no_rt_limit=True,
                input_path=str(input_file),
            )
            restorer_result = denker_result
            restored_audio = denker_result.audio

            # Export-Quality-Gate §8.1 + §0c RELEASE_MUST
            # §0c: Bei fehlgeschlagenem Gate → Export mit degraded-Status, kein Hardstop.
            _export_degraded = False
            _degraded_reasons: list[str] = []

            _qe = getattr(denker_result, "quality_estimate", None)
            if _qe is not None and _qe < 0.55:
                _export_degraded = True
                _degraded_reasons.append(f"quality_estimate={_qe:.3f}<0.55 (§8.1)")
                logger.warning("§0c: quality_estimate=%.3f < 0.55 — Export mit Status 'degraded' (§0c Pflicht).", _qe)
            _P1_P2_THRESHOLDS = {
                "natuerlichkeit": 0.90,
                "authentizitaet": 0.88,
                "tonal_center": 0.95,
                "timbre_authentizitaet": 0.87,
                "artikulation": 0.85,
            }
            _goals = getattr(denker_result, "musical_goals_scores", None) or {}
            _failed_goals = [
                f"{g}={_goals[g]:.3f}<{t}" for g, t in _P1_P2_THRESHOLDS.items() if g in _goals and _goals[g] < t
            ]
            if _failed_goals:
                _export_degraded = True
                _degraded_reasons.append(f"P1/P2-Goals: {', '.join(_failed_goals)}")
                logger.warning(
                    "§0c: P1/P2-Goals verfehlt (%s) — Export mit Status 'degraded'.",
                    ", ".join(_failed_goals),
                )

            if sf is None:
                raise RuntimeError(
                    "Audio-Export fehlgeschlagen: soundfile konnte nicht geladen werden. "
                    "Lösung: Installation prüfen und erneut starten."
                )

            # Aurik-internes Format: (channels, samples) → normalisieren auf (samples, channels)
            if isinstance(restored_audio, np.ndarray):
                if restored_audio.ndim == 2 and restored_audio.shape[0] < restored_audio.shape[1]:
                    restored_audio = np.ascontiguousarray(restored_audio.T)
                restored_audio = np.asarray(restored_audio, dtype=np.float32)

            sf.write(str(output_file), restored_audio, file_sr)

            elapsed = time.time() - start_time

            # Mark as completed
            self.completed.add(str(input_file))
            self._save_state()

            _result_dict: dict = {
                "file": str(input_file),
                "success": True,
                "output": str(output_file),
                "elapsed": elapsed,
                "total_time": restorer_result.total_time_seconds,
            }
            if _export_degraded:
                _result_dict["degraded"] = True
                _result_dict["degraded_reasons"] = _degraded_reasons
            return _result_dict

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Failed to process %s: %s", input_file.name, e)

            return {"file": str(input_file), "success": False, "error": str(e), "elapsed": elapsed}

    def process_batch(self, input_files: list[Path], config: dict | None = None) -> list[dict]:
        """
        Process multiple audio files in parallel.

        Args:
            input_files: List of input audio files
            config: Custom configuration

        Returns:
            List of processing results
        """
        results = []

        # Progress bar
        with tqdm(total=len(input_files), desc="Processing", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(self.process_file, f, config): f for f in input_files}

                # Process as completed
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)

                    # Update progress
                    if result["success"]:
                        pbar.set_postfix({"Success": result["file"]})
                    else:
                        pbar.set_postfix({"Failed": result["file"]})
                    pbar.update(1)

        return results

    def print_summary(self, results: list[dict]):
        """Log batch processing summary."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        total_time = sum(r["elapsed"] for r in results)
        avg_time = total_time / len(results) if results else 0
        total = len(results)
        success_pct = (len(successful) / total * 100.0) if total else 0.0
        failed_pct = (len(failed) / total * 100.0) if total else 0.0

        logger.info("=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info("Total Files:   %d", total)
        logger.info("Successful:    %d (%.1f%%)", len(successful), success_pct)
        logger.info("Failed:        %d (%.1f%%)", len(failed), failed_pct)
        logger.info("Total Time:    %.1fs", total_time)
        logger.info("Average Time:  %.1fs per file", avg_time)

        if failed:
            logger.info("Failed Files:")
            for r in failed:
                logger.info("  - %s: %s", os.path.basename(r["file"]), r["error"])

        logger.info("=" * 80)


def main():
    """Parse CLI arguments and run batch processing."""
    parser = argparse.ArgumentParser(description="Batch process audio files with Aurik 9.0")
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-m",
        "--material",
        default="vinyl",
        choices=["shellac", "vinyl", "tape", "cd", "streaming"],
        help="Material type (default: vinyl)",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume from previous batch")
    parser.add_argument(
        "--no-album-consistency",
        action="store_true",
        default=False,
        help="Skip post-batch album consistency pass (LUFS + spectral tilt alignment)",
    )
    parser.add_argument(
        "--album-consistency-dry-run",
        action="store_true",
        default=False,
        help="Analyze album consistency without writing corrected files (report only)",
    )

    args = parser.parse_args()

    # Material mapping
    if MaterialType is None:
        logger.error("MaterialType not available - backend module not imported")
        return

    material_map = {
        "shellac": MaterialType.SHELLAC,
        "vinyl": MaterialType.VINYL,
        "tape": MaterialType.TAPE,
        "cd": MaterialType.CD_DIGITAL,
        "streaming": MaterialType.STREAMING,
    }
    material = material_map[args.material]

    # Initialize processor
    processor = BatchProcessor(output_dir=Path(args.output), workers=args.workers, resume=args.resume)

    # Find files
    files = processor.find_audio_files(args.inputs)

    if not files:
        logger.warning("No audio files found")
        return

    # Process
    logger.info("Starting batch processing: %d files, %d workers", len(files), args.workers)
    results = processor.process_batch(files, material)

    # Album Consistency Pass (§ Album-Konsistenz-Pass)
    # Runs after all songs have been individually restored.  Aligns LUFS and
    # spectral tilt across the batch using gentle, bounded corrections so the
    # whole album sounds coherent without touching songs already within the
    # album median (§0 Primum non nocere).
    if not args.no_album_consistency:
        _output_files = [r["output"] for r in results if r.get("success") and r.get("output")]
        if len(_output_files) >= 3:
            try:
                if get_album_consistency_pass is None:
                    raise RuntimeError("Album-Konsistenz-Modul nicht verfügbar")

                _album_pass = get_album_consistency_pass()
                _report = _album_pass.process_output_files(
                    _output_files,
                    sr=48000,
                    dry_run=getattr(args, "album_consistency_dry_run", False),
                )
                if not _report.skipped_insufficient_songs:
                    logger.info(
                        "Album-Konsistenz-Pass: %d/%d Korrekturen angewendet "
                        "(LUFS-Median=%.1f LU, Tilt-Median=%.2f dB/oct, %.1fs)",
                        _report.corrections_applied,
                        _report.n_songs,
                        _report.album_lufs_median,
                        _report.album_tilt_median,
                        _report.elapsed_seconds,
                    )
            except Exception as _exc:
                logger.warning("Album-Konsistenz-Pass fehlgeschlagen (non-critical): %s", _exc)
        else:
            logger.info(
                "Album-Konsistenz-Pass: übersprungen (%d erfolgreiche Songs < 3 Minimum).",
                len(_output_files),
            )

    # Summary
    processor.print_summary(results)


if __name__ == "__main__":
    main()
