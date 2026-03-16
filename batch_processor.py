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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

import soundfile as sf
from tqdm import tqdm

from core.defect_scanner import MaterialType
from core.unified_restorer_v3 import UnifiedRestorerV3

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

        logger.info(f"BatchProcessor initialized: {workers} workers, output: {output_dir}")
        if resume and self.completed:
            logger.info(f"Resume mode: {len(self.completed)} files already processed")

    def _load_state(self) -> set:
        """Load completed files from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                return set(state.get("completed", []))
            except Exception:
                return set()
        return set()

    def _save_state(self):
        """Save completed files to state file."""
        with open(self.state_file, "w") as f:
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

        logger.info(f"Found {len(files)} audio files to process")
        return files

    def process_file(self, input_file: Path, material: MaterialType, config: dict | None = None) -> dict:
        """
        Process a single audio file.

        Args:
            input_file: Input audio file path
            material: Material type
            config: Custom configuration

        Returns:
            Processing result dict
        """
        start_time = time.time()

        try:
            # Output filename
            output_file = self.output_dir / f"{input_file.stem}_restored{input_file.suffix}"

            # Initialize restorer
            restorer = UnifiedRestorerV3()

            # Process
            logger.info(f"Processing: {input_file.name}")
            audio_data, file_sr = sf.read(str(input_file))
            restorer_result = restorer.restore(
                audio_data, sample_rate=file_sr, material=material, **(config if config else {})
            )
            sf.write(str(output_file), restorer_result.audio, file_sr)

            elapsed = time.time() - start_time

            # Mark as completed
            self.completed.add(str(input_file))
            self._save_state()

            return {
                "file": str(input_file),
                "success": True,
                "output": str(output_file),
                "elapsed": elapsed,
                "total_time": restorer_result.total_time_seconds,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to process {input_file.name}: {e}")

            return {"file": str(input_file), "success": False, "error": str(e), "elapsed": elapsed}

    def process_batch(self, input_files: list[Path], material: MaterialType, config: dict | None = None) -> list[dict]:
        """
        Process multiple audio files in parallel.

        Args:
            input_files: List of input audio files
            material: Material type
            config: Custom configuration

        Returns:
            List of processing results
        """
        results = []

        # Progress bar
        with tqdm(total=len(input_files), desc="Processing", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(self.process_file, f, material, config): f for f in input_files}

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
        """Print batch processing summary."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        total_time = sum(r["elapsed"] for r in results)
        avg_time = total_time / len(results) if results else 0

        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total Files:   {len(results)}")
        print(f"Successful:    {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed:        {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"Total Time:    {total_time:.1f}s")
        print(f"Average Time:  {avg_time:.1f}s per file")
        print()

        if failed:
            print("Failed Files:")
            for r in failed:
                print(f"  - {os.path.basename(r['file'])}: {r['error']}")
            print()

        print("=" * 80)


def main():
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

    args = parser.parse_args()

    # Material mapping
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
    logger.info(f"Starting batch processing: {len(files)} files, {args.workers} workers")
    results = processor.process_batch(files, material)

    # Summary
    processor.print_summary(results)


if __name__ == "__main__":
    main()
