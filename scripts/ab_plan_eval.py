"""ab_plan_eval.py — Plan-A-vs-Plan-B-Mess-Harness (EVAL_NON_RELEASE).

Rendert DENSELBEN Song mit zwei verschiedenen Phasen-Plänen über denselben
``UnifiedRestorerV3``-Pfad und stellt die perzeptuellen Kennzahlen (HPI, VQI,
artifact_freedom, OQS/MUSHRA) gegenüber. Die restaurierten Audios werden
geschrieben, damit der Nutzer **gegenhören** kann — die Zahlen sind nur ein
Proxy, das Ohr entscheidet (§0 Klangwahrheit, §0g kein Metric-Gaming).

WICHTIG — Statuskennzeichnung:
    Dies ist eine reine **Eval-/Mess-Harness**, KEIN Release-Pfad. Sie ruft
    ``UnifiedRestorerV3.restore(..., precomputed_phase_plan=[...])`` direkt auf,
    weil das der einzige offizielle Weg ist, einen FIXEN Phasenplan zu
    injizieren (``AurikDenker.denke()`` reicht ``precomputed_phase_plan`` nicht
    durch — §2.53b). Der direkte UV3-Aufruf ist laut "Canonical Contract Drift
    Gate" in **Release**-Pfaden verboten; für diese klar gekennzeichnete
    Nicht-Release-Eval ist er der korrekte Weg, um zwei Pläne fair zu vergleichen.

Fairness-Garantie:
    Beide Läufe teilen sich EINE Voranalyse (``run_pre_analysis``) — identisches
    Material/Era/Genre/Defects/Restorability. Der EINZIGE Unterschied zwischen
    Lauf A und Lauf B ist die Phasenliste. Jeder Lauf nutzt eine frische
    UV3-Instanz (kein Singleton-Stale-State zwischen A und B).

Beispiel:
    .venv_aurik/bin/python scripts/ab_plan_eval.py \\
        --input "test_audio/song.mp3" \\
        --mode restoration \\
        --plan-a auto \\
        --plan-b "phase_03_denoise,phase_29_tape_hiss_reduction,phase_06_frequency_restoration" \\
        --label-a baseline --label-b kandidat

    ``--plan-a auto`` ⇒ UV3 selektiert autonom (Baseline). ``--plan-b`` ⇒
    fixierter Kandidatenplan. Beide werden gerendert und verglichen.
"""

# pylint: disable=wrong-import-position
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ab_plan_eval")

import numpy as np
import soundfile as sf

TARGET_SR = 48000


def _parse_plan(raw: str | None, plan_file: str | None) -> tuple[list[str] | None, str]:
    """Parst eine Plan-Angabe zu einer Phasenliste oder ``None`` (= autonome Selektion).

    Rückgabe: (plan_or_None, beschreibung). ``None`` bedeutet UV3 selektiert
    autonom ("auto"-Baseline). Eine JSON-Datei hat Vorrang vor ``raw``.
    """
    if plan_file:
        data = json.loads(Path(plan_file).read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
            raise ValueError(f"Plan-Datei {plan_file} muss eine JSON-Liste von Phasen-Strings sein.")
        return list(data), f"file:{plan_file} ({len(data)} Phasen)"
    if raw is None:
        return None, "auto (UV3 autonom)"
    token = raw.strip()
    if token.lower() in {"auto", "none", ""}:
        return None, "auto (UV3 autonom)"
    phases = [p.strip() for p in token.split(",") if p.strip()]
    return phases, f"{len(phases)} Phasen (CLI)"


def _validate_phase_ids(phases: list[str] | None, repo_root: Path) -> list[str]:
    """Warnt vor Phasen-IDs ohne korrespondierende ``backend/core/phases/<id>.py``.

    Reine Diagnose — UV3 würde unbekannte Phasen still überspringen, was eine
    A/B-Messung verfälschen könnte. Rückgabe: Liste unbekannter IDs.
    """
    if not phases:
        return []
    phase_dir = repo_root / "backend" / "core" / "phases"
    unknown: list[str] = []
    for pid in phases:
        if not (phase_dir / f"{pid}.py").exists():
            unknown.append(pid)
    if unknown:
        logger.warning(
            "Unbekannte Phasen-IDs (keine Datei in backend/core/phases/): %s — "
            "UV3 überspringt diese still; Vergleich könnte verfälscht sein.",
            unknown,
        )
    return unknown


def _load_audio_48k(input_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Lädt Audio über den kanonischen Importpfad und resampled auf 48 kHz.

    Rückgabe: (audio_native, audio_48k, sr_native).
    """
    from backend.file_import import load_audio_file

    res = load_audio_file(input_path, do_carrier_analysis=False)
    if res is None or "audio" not in res or "sr" not in res:
        raise RuntimeError(f"Audio-Import fehlgeschlagen für: {input_path}")
    audio_native = res["audio"]
    sr_native = int(res["sr"])
    logger.info("Geladen: shape=%s sr=%d", audio_native.shape, sr_native)
    if sr_native != TARGET_SR:
        import resampy

        axis = 0 if audio_native.ndim == 2 else -1
        audio_48k = resampy.resample(audio_native, sr_native, TARGET_SR, axis=axis)
        logger.info("Resampled auf %d Hz, shape=%s", TARGET_SR, audio_48k.shape)
    else:
        audio_48k = audio_native.copy()
    return audio_native, audio_48k, sr_native


def _extract_metrics(result: Any) -> dict[str, Any]:
    """Liest die perzeptuellen Kennzahlen aus dem RestorationResult-Metadata.

    Robust gegen fehlende/``None``-Felder. OQS liegt als MUSHRA-Score unter
    ``metadata["mushra"]["mushra_score"]``; VQI ist nur bei Gesang gesetzt.
    """
    md = getattr(result, "metadata", None) or {}
    hpg = md.get("holistic_perceptual_gate") or {}
    mushra = md.get("mushra")
    oqs = mushra.get("mushra_score") if isinstance(mushra, dict) else None
    vqi = hpg.get("vqi")
    if vqi is None:
        vqi = md.get("vqi")
    return {
        "hpi": hpg.get("hpi"),
        "artifact_freedom": hpg.get("artifact_freedom"),
        "vqi": vqi,
        "oqs_mushra": oqs,
        "timbral_fidelity": hpg.get("timbral_fidelity"),
        "mert_similarity": hpg.get("mert_similarity"),
        "hpg_passed": hpg.get("passed"),
        "quality_estimate": getattr(result, "quality_estimate", None),
        "phases_executed": list(getattr(result, "phases_executed", []) or []),
        "n_phases_executed": len(getattr(result, "phases_executed", []) or []),
    }


def _run_one(
    *,
    audio_48k: np.ndarray,
    pre_result: Any,
    mode: str,
    plan: list[str] | None,
    no_rt_limit: bool,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Führt EINEN Restaurierungslauf mit fixiertem (oder autonomem) Plan aus.

    Frische UV3-Instanz pro Lauf → kein Stale-State zwischen A und B.
    Rückgabe: (restauriertes_audio, metriken, laufzeit_s).
    """
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    kwargs: dict[str, Any] = {
        "mode": mode,
        "pre_analysis_result": pre_result,
        "no_rt_limit": no_rt_limit,
    }
    if plan:  # leere Liste / None → UV3 selektiert autonom
        kwargs["precomputed_phase_plan"] = plan
    t0 = time.time()
    result = uv3.restore(audio_48k, sample_rate=TARGET_SR, **kwargs)
    elapsed = time.time() - t0
    return getattr(result, "audio", audio_48k), _extract_metrics(result), elapsed


def _fmt(v: Any) -> str:
    """Formatiert einen Metrikwert für die Tabelle (None → 'n/a')."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(b: Any, a: Any) -> str:
    """Formatiert die Differenz B−A für numerische Metriken."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = float(b) - float(a)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan-A-vs-Plan-B-Mess-Harness (EVAL_NON_RELEASE) — rendert beide Pläne und vergleicht.",
    )
    parser.add_argument("--input", required=True, help="Pfad zur Audio-Importdatei.")
    parser.add_argument(
        "--mode",
        default="restoration",
        choices=["restoration", "studio2026", "studio_2026"],
        help="Verarbeitungsmodus (default: restoration).",
    )
    parser.add_argument("--plan-a", default="auto", help="Plan A: Komma-Liste von Phasen-IDs oder 'auto'.")
    parser.add_argument("--plan-b", default=None, help="Plan B: Komma-Liste von Phasen-IDs oder 'auto'.")
    parser.add_argument("--plan-a-file", default=None, help="Plan A als JSON-Liste (hat Vorrang vor --plan-a).")
    parser.add_argument("--plan-b-file", default=None, help="Plan B als JSON-Liste (hat Vorrang vor --plan-b).")
    parser.add_argument("--label-a", default="A", help="Label für Plan A (Dateiname/Report).")
    parser.add_argument("--label-b", default="B", help="Label für Plan B (Dateiname/Report).")
    parser.add_argument("--out-dir", default="output_audio/ab_eval", help="Ausgabeverzeichnis.")
    parser.add_argument(
        "--rt-limit",
        action="store_true",
        help="Runtime-Limit aktivieren (default: aus → maximale Qualität, wie run30).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    no_rt_limit = not args.rt_limit

    plan_a, desc_a = _parse_plan(args.plan_a, args.plan_a_file)
    plan_b, desc_b = _parse_plan(args.plan_b, args.plan_b_file)
    if plan_a is None and plan_b is None:
        logger.error("Mindestens einer von Plan A / Plan B muss ein FIXER Plan sein (nicht beide 'auto').")
        return 2

    _validate_phase_ids(plan_a, repo_root)
    _validate_phase_ids(plan_b, repo_root)

    logger.info("Plan A [%s]: %s", args.label_a, desc_a)
    logger.info("Plan B [%s]: %s", args.label_b, desc_b)

    # --- Audio laden + 48 kHz ---
    audio_native, audio_48k, sr_native = _load_audio_48k(args.input)

    # --- Voranalyse EINMALIG (fair: beide Läufe teilen sich identische Analyse) ---
    from backend.core.pre_analysis import run_pre_analysis

    logger.info("Starte Voranalyse (Medium, Era, Genre, Defects, Restorability)…")
    t_pre = time.time()
    pre_result = run_pre_analysis(
        audio_native=audio_native,
        sr_native=sr_native,
        audio_48k=audio_48k,
        file_path=os.path.abspath(args.input),
        store_in_bridge_cache=True,
    )
    logger.info(
        "Voranalyse fertig in %.1fs | medium=%s era=%s restorability=%s",
        time.time() - t_pre,
        getattr(getattr(pre_result, "medium", None), "primary_material", "?"),
        getattr(getattr(pre_result, "era", None), "decade", "?"),
        getattr(getattr(pre_result, "restorability", None), "restorability_score", "?"),
    )
    if getattr(pre_result, "errors", None):
        logger.warning("Voranalyse-Fehler: %s", pre_result.errors)

    # --- Lauf A ---
    logger.info("=" * 60)
    logger.info("RENDER Plan A [%s] …", args.label_a)
    audio_a, metrics_a, rt_a = _run_one(
        audio_48k=audio_48k, pre_result=pre_result, mode=args.mode, plan=plan_a, no_rt_limit=no_rt_limit
    )
    logger.info("Plan A fertig in %.1fs", rt_a)

    # --- Lauf B ---
    logger.info("=" * 60)
    logger.info("RENDER Plan B [%s] …", args.label_b)
    audio_b, metrics_b, rt_b = _run_one(
        audio_48k=audio_48k, pre_result=pre_result, mode=args.mode, plan=plan_b, no_rt_limit=no_rt_limit
    )
    logger.info("Plan B fertig in %.1fs", rt_b)

    # --- Audio-Artefakte schreiben (zum Gegenhören) ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    path_a = out_dir / f"{stem}__{args.label_a}.wav"
    path_b = out_dir / f"{stem}__{args.label_b}.wav"
    sf.write(str(path_a), np.asarray(audio_a), TARGET_SR)
    sf.write(str(path_b), np.asarray(audio_b), TARGET_SR)
    logger.info("Geschrieben: %s", path_a)
    logger.info("Geschrieben: %s", path_b)

    # --- Vergleichstabelle ---
    metric_keys = [
        "hpi",
        "artifact_freedom",
        "vqi",
        "oqs_mushra",
        "timbral_fidelity",
        "mert_similarity",
        "quality_estimate",
        "n_phases_executed",
    ]
    logger.info("=" * 60)
    logger.info("VERGLEICH  (Δ = B − A)")
    logger.info("%-20s %12s %12s %12s", "Metrik", args.label_a, args.label_b, "Δ")
    for k in metric_keys:
        logger.info(
            "%-20s %12s %12s %12s",
            k,
            _fmt(metrics_a.get(k)),
            _fmt(metrics_b.get(k)),
            _delta(metrics_b.get(k), metrics_a.get(k)),
        )
    logger.info("%-20s %12.1f %12.1f %12s", "runtime_s", rt_a, rt_b, _delta(rt_b, rt_a))

    # --- JSON-Report ---
    report = {
        "eval_kind": "EVAL_NON_RELEASE_ab_plan",
        "input": os.path.abspath(args.input),
        "mode": args.mode,
        "no_rt_limit": no_rt_limit,
        "plan_a": {"label": args.label_a, "desc": desc_a, "phases": plan_a, "runtime_s": rt_a, "metrics": metrics_a},
        "plan_b": {"label": args.label_b, "desc": desc_b, "phases": plan_b, "runtime_s": rt_b, "metrics": metrics_b},
        "audio_a": str(path_a),
        "audio_b": str(path_b),
    }
    report_path = out_dir / f"{stem}__ab_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report: %s", report_path)
    logger.info(
        "Hinweis (§0g): Die Zahlen sind ein Proxy. Höre A und B gegen, bevor du einen Plan als 'besser' wertest."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
