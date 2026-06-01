#!/usr/bin/env python3
"""
Batch Processing System for Aurik 9.12.10
========================================

Process multiple audio files in parallel with progress tracking.

Features:
- Multi-threaded processing
- Progress bar with ETA
- Memory-efficient streaming
- Resume capability
- Detailed logging

Author: Aurik Team
Date: May 2026
"""

import argparse
import importlib
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

_bridge = importlib.import_module("backend.api.bridge")
_build_export_quality_gate_payload = _bridge.build_export_quality_gate_payload
_export_guard = _bridge.export_guard
_get_audio_exporter_class = _bridge.get_audio_exporter_class
_get_aurik_denker_instance = _bridge.get_aurik_denker_instance
_get_load_audio_fn = _bridge.get_load_audio_fn
_normalize_user_mode = _bridge.normalize_user_mode
_run_pre_analysis = _bridge.run_pre_analysis
_validate_export_quality = _bridge.validate_export_quality

try:
    _load_audio_file = _get_load_audio_fn()
except Exception:
    _load_audio_file = None  # type: ignore[assignment]

_get_aurik_denker = _get_aurik_denker_instance

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

_TARGET_SR = 48_000


def _coerce_audio_array(audio: object) -> np.ndarray:
    """Normalisiert geladene Audio-Daten auf float32 in (samples, channels)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _load_audio_pair(path: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Lädt Audio kanonisch via Bridge als native + 48-kHz-Version."""
    load_audio_file = _load_audio_file
    if load_audio_file is None:
        load_audio_file = _get_load_audio_fn()
    loaded_native = load_audio_file(path, target_sr=None, mono=False, do_carrier_analysis=False)
    if not isinstance(loaded_native, dict) or loaded_native.get("audio") is None or loaded_native.get("sr") is None:
        raise RuntimeError(str((loaded_native or {}).get("error") or "Unbekannter Ladefehler"))

    audio_native = _coerce_audio_array(loaded_native["audio"])
    sr_native = int(loaded_native["sr"])
    if sr_native == _TARGET_SR:
        return audio_native, sr_native, audio_native

    loaded_48k = load_audio_file(path, target_sr=_TARGET_SR, mono=False, do_carrier_analysis=False)
    if not isinstance(loaded_48k, dict) or loaded_48k.get("audio") is None:
        raise RuntimeError(str((loaded_48k or {}).get("error") or "48-kHz-Ladefehler"))
    audio_48k = _coerce_audio_array(loaded_48k["audio"])
    return audio_native, sr_native, audio_48k


def _export_processed_audio(
    output_file: Path,
    restored_audio: object,
    reference_audio: np.ndarray,
    export_metadata: dict[str, object],
) -> None:
    """Exportiert Batch-Audio über export_guard + AudioExporter/Fallback-WAV."""
    write_audio = _export_guard(restored_audio)
    if isinstance(write_audio, np.ndarray) and write_audio.ndim == 2 and write_audio.shape[0] < write_audio.shape[1]:
        write_audio = np.ascontiguousarray(write_audio.T)

    audio_exporter_cls = _get_audio_exporter_class()
    if audio_exporter_cls is not None and output_file.suffix.lower() in audio_exporter_cls.FORMATS:
        exporter = audio_exporter_cls()
        exporter.export(
            write_audio,
            _TARGET_SR,
            output_file,
            bit_depth=24,
            quality="veryhigh",
            metadata=export_metadata,
            normalize=False,
            reference_audio=reference_audio,
        )
        return

    if sf is None:
        raise RuntimeError(
            "Audio-Export fehlgeschlagen: soundfile konnte nicht geladen werden. "
            "Lösung: Installation prüfen und erneut starten."
        )

    tmp_path = str(output_file) + ".wav.tmp"
    try:
        sf.write(tmp_path, write_audio, _TARGET_SR, format="WAV", subtype="PCM_24")
        os.replace(tmp_path, output_file)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Temporäre Exportdatei konnte nicht entfernt werden: %s", tmp_path)


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
            audio_native, file_sr, audio_48k = _load_audio_pair(str(input_file))
            pre_analysis = _run_pre_analysis(
                audio_native=audio_native,
                sr_native=file_sr,
                audio_48k=audio_48k,
                file_path=str(input_file),
                store_in_bridge_cache=True,
            )

            # §2.2 Canonical Singleton-Einstiegspunkt (No-Competing-Instances-Protokoll)
            denker = _get_aurik_denker()
            mode = _normalize_user_mode(str((config or {}).get("mode", "restoration")))
            denker_result = denker.denke(
                audio_48k,
                _TARGET_SR,
                mode=mode,
                no_rt_limit=True,
                input_path=str(input_file),
                output_path=str(output_file),
                pre_analysis_result=pre_analysis,
            )
            restorer_result = denker_result
            eq_passed, eq_warnings = _validate_export_quality(denker_result)
            eq_payload = _build_export_quality_gate_payload(denker_result)

            _meta = getattr(denker_result, "metadata", {})
            if not isinstance(_meta, dict):
                _meta = {}
            _wcs_raw = _meta.get("worldclass_composite_gate", {})
            _wcs_gate = _wcs_raw if isinstance(_wcs_raw, dict) else {}
            _hybrid_engineer_vector_json = json.dumps(
                _meta.get("hybrid_engineer_vector", {}),
                sort_keys=True,
                ensure_ascii=True,
            )

            export_metadata: dict[str, object] = {
                "quality_gate_passed": str(bool(eq_payload.get("passed", eq_passed))),
                "quality_gate_degradation_status": str(eq_payload.get("degradation_status", "ok")),
                "quality_gate_fail_reason": str(eq_payload.get("fail_reason", "")),
                "quality_gate_recovery_attempted": str(bool(eq_payload.get("recovery_attempted", False))),
                "quality_gate_best_possible_reached": str(bool(eq_payload.get("best_possible_reached", False))),
                "quality_gate_profile": str(eq_payload.get("profile", "")),
                "quality_gate_material": str(eq_payload.get("material", "")),
                "quality_gate_preserve_signal": str(float(eq_payload.get("preserve_signal", 0.0) or 0.0)),
                "quality_gate_worldclass_score": str(float(_wcs_gate.get("wcs", 0.0) or 0.0)),
                "quality_gate_hybrid_engineer_vector": _hybrid_engineer_vector_json,
            }
            if eq_warnings:
                export_metadata["quality_gate_warnings"] = json.dumps(list(eq_warnings), ensure_ascii=True)

            _export_processed_audio(output_file, denker_result.audio, audio_48k, export_metadata)

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
                "metadata": dict(export_metadata),
            }
            _result_dict["quality_gate_payload"] = eq_payload
            if not eq_passed:
                _result_dict["degraded"] = True
                _result_dict["degraded_reasons"] = list(eq_warnings) or [str(eq_payload.get("fail_reason", ""))]
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
    parser = argparse.ArgumentParser(description="Batch process audio files with Aurik 9.12.10")
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-m",
        "--material",
        default="vinyl",
        choices=["shellac", "vinyl", "tape", "cd", "streaming"],
        help="Legacy-Kompatibilitaetsoption. Das Medium wird produktiv automatisch erkannt.",
    )
    parser.add_argument(
        "--mode",
        default="Restoration",
        choices=["Restoration", "Studio 2026"],
        help="Verarbeitungsmodus (default: Restoration)",
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

    # Initialize processor
    processor = BatchProcessor(output_dir=Path(args.output), workers=args.workers, resume=args.resume)

    # Find files
    files = processor.find_audio_files(args.inputs)

    if not files:
        logger.warning("No audio files found")
        return

    mode = "studio2026" if args.mode == "Studio 2026" else "restoration"
    logger.info(
        "Hinweis: --material=%s bleibt aus Kompatibilitaetsgruenden erhalten; das Medium wird automatisch erkannt.",
        args.material,
    )

    # Process
    logger.info("Starting batch processing: %d files, %d workers", len(files), args.workers)
    results = processor.process_batch(files, {"mode": mode})

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
