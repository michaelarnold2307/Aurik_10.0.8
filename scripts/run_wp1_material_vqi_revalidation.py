#!/usr/bin/env python3
"""WP1-Runner fuer material_vqi_floor-Revalidierung.

Verarbeitet nur Planzeilen mit `workpackage == wp1_vqi_floor` und schreibt Ergebnisse
in `result_template.csv` des jeweiligen Run-Ordners.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("workpackage", ""),
        row.get("variant", ""),
        row.get("case_id", ""),
        row.get("param_name", ""),
        row.get("param_value", ""),
    )


def _safe_result_metric(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    return None


def _prepare_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)

    if sr != 48_000:
        import librosa  # pylint: disable=import-outside-toplevel

        arr = librosa.resample(arr.T, orig_sr=sr, target_sr=48_000).T.astype(np.float32)
        sr = 48_000

    return arr, sr


def _execute_wp1_case(
    row: dict[str, str],
    repo_root: Path,
    max_seconds: float,
    ml_runtime_budget_s: float,
) -> dict[str, str]:
    # Importe nur im Execute-Pfad, damit dry-run schnell bleibt.
    from backend.core.musical_goals import vocal_quality_index as _vqi_mod  # pylint: disable=import-outside-toplevel
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3  # pylint: disable=import-outside-toplevel
    from backend.file_import import load_audio_file  # pylint: disable=import-outside-toplevel

    compute_vqi = _vqi_mod.compute_vqi
    _get_floor = getattr(_vqi_mod, "get_vqi_material_floor", None)

    out = dict(row)
    audio_rel = row.get("audio_path", "")
    audio_path = (repo_root / audio_rel).resolve()

    if not audio_path.exists():
        out["status"] = "skipped_missing_audio"
        out["fail_reason"] = "audio_not_found"
        out["notes"] = f"Datei fehlt: {audio_rel}"
        return out

    loaded = load_audio_file(str(audio_path), target_sr=None, mono=False, do_carrier_analysis=False)
    if loaded is None or "audio" not in loaded:
        out["status"] = "failed_load"
        out["fail_reason"] = "audio_load_failed"
        out["notes"] = "load_audio_file lieferte kein Audio"
        return out

    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or 48_000)
    audio, sr = _prepare_audio(audio, sr)

    if max_seconds > 0:
        max_n = int(sr * max_seconds)
        if audio.shape[0] > max_n:
            audio = audio[:max_n]

    restorer = UnifiedRestorerV3()
    result = restorer.restore(
        audio.T,
        sample_rate=sr,
        mode="restoration",
        ml_runtime_budget_s=float(max(1.0, ml_runtime_budget_s)),
    )

    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] in (1, 2) and restored.shape[1] > restored.shape[0]:
        restored = restored.T
    if restored.ndim == 1:
        restored = np.stack([restored, restored], axis=1)

    n = min(audio.shape[0], restored.shape[0])
    orig_mono = np.mean(audio[:n], axis=1)
    rest_mono = np.mean(restored[:n], axis=1)

    vqi_result = compute_vqi(orig_mono.astype(np.float32), rest_mono.astype(np.float32), sr)
    vqi = float(vqi_result.get("vqi", 0.0))

    material = str(row.get("material", "unknown"))
    base_floor = float(_get_floor(material) if callable(_get_floor) else 0.72)
    delta = _to_float(row.get("param_value", "0")) or 0.0
    effective_floor = float(np.clip(base_floor + delta, 0.50, 0.95))

    metadata = getattr(result, "metadata", {}) or {}

    out["artifact_freedom"] = str(_safe_result_metric(metadata, "artifact_freedom") or "")
    out["hpi"] = str(
        _safe_result_metric(
            metadata,
            "holistic_perceptual_index",
            "hpi",
            "final_hpi",
        )
        or ""
    )
    out["vqi"] = f"{vqi:.6f}"
    out["mert_similarity"] = str(_safe_result_metric(metadata, "mert_similarity") or "")
    out["timbral_fidelity"] = str(_safe_result_metric(metadata, "timbral_fidelity") or "")

    out["status"] = "recovered" if vqi >= effective_floor else "degraded"
    out["fail_reason"] = "vqi_below_effective_floor" if vqi < effective_floor else ""
    out["notes"] = (
        f"base_floor={base_floor:.3f}; delta={delta:+.3f}; effective_floor={effective_floor:.3f}; vqi={vqi:.3f}"
    )

    sid = vqi_result.get("singer_identity_cosine")
    if isinstance(sid, (float, int)):
        out["singer_identity_cosine"] = f"{float(sid):.6f}"

    return out


def main() -> None:
    """CLI-Einstiegspunkt: verarbeitet WP1-Planzeilen und aktualisiert result_template.csv."""
    parser = argparse.ArgumentParser(description="WP1 material_vqi_floor Runner")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run-Ordner mit plan.csv und result_template.csv",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Fuehrt echte Restoration + VQI-Berechnung aus. Ohne Flag nur Plan/Dry-Run-Markierung.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optionales Limit fuer die Anzahl verarbeiteter WP1-Faelle (0 = kein Limit).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=8.0,
        help="Maximale Audiodauer pro Fall fuer Execute-Lauf (0 = volle Datei).",
    )
    parser.add_argument(
        "--ml-runtime-budget-s",
        type=float,
        default=20.0,
        help="ML-Runtime-Budget pro Restore-Aufruf in Sekunden.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    plan_csv = run_dir / "plan.csv"
    result_csv = run_dir / "result_template.csv"

    if not plan_csv.exists() or not result_csv.exists():
        raise SystemExit("plan.csv oder result_template.csv fehlt im Run-Ordner.")

    plan_rows = _read_csv(plan_csv)
    result_rows = _read_csv(result_csv)

    wp1_plan = [r for r in plan_rows if r.get("workpackage") == "wp1_vqi_floor"]
    if args.max_cases > 0:
        wp1_plan = wp1_plan[: args.max_cases]

    updates: dict[tuple[str, str, str, str, str], dict[str, str]] = {}

    for row in wp1_plan:
        out = dict(row)
        if args.execute:
            out = _execute_wp1_case(
                row,
                repo_root,
                max_seconds=float(args.max_seconds),
                ml_runtime_budget_s=float(args.ml_runtime_budget_s),
            )
        else:
            audio_rel = row.get("audio_path", "")
            audio_exists = (repo_root / audio_rel).exists()
            out["status"] = "planned_wp1" if audio_exists else "skipped_missing_audio"
            out["fail_reason"] = "" if audio_exists else "audio_not_found"
            out["notes"] = "dry_run_only" if audio_exists else f"Datei fehlt: {audio_rel}"

        updates[_row_key(row)] = out

    merged: list[dict[str, str]] = []
    for row in result_rows:
        key = _row_key(row)
        if key in updates:
            updated = dict(row)
            updated.update(updates[key])
            merged.append(updated)
        else:
            merged.append(row)

    _write_csv(result_csv, merged)

    print(f"WP1-Zeilen im Plan: {len([r for r in plan_rows if r.get('workpackage') == 'wp1_vqi_floor'])}")
    print(f"WP1-Zeilen verarbeitet: {len(updates)}")
    print(f"Execute-Modus: {'ja' if args.execute else 'nein (dry-run)'}")
    print(f"Aktualisiert: {result_csv}")


if __name__ == "__main__":
    main()
