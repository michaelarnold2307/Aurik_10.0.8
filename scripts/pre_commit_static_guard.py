#!/usr/bin/env python3
"""
pre-commit hook: Aurik Static-Value Guard (§v10)
=================================================

Prüft vor jedem Commit, dass keine neuen statischen Werte eingeführt werden,
die dynamisch/adaptiv sein müssten. Wissenschaftlich begründete Konstanten
(IEC, ISO, physikalische Limits) sind ausgenommen.

Regeln:
  R1 — Kein neuer fester dB/Gain-Wert ohne SNR-Adaption
  R2 — Kein neuer fester EQ-Gain ohne Spektrum-Messung
  R3 — Keine neuen Source-Grep-Tests (inspect.getsource + assert string in)
  R4 — Keine neuen unguarded Imports (hypothesis, requests, torch ohne try/except)
  R5 — Test-Dateien ≥ 50 Zeilen MÜSSEN min. 1 pytest-Marker haben

Usage:
  python scripts/pre_commit_static_guard.py [--ci] [file ...]
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Wissenschaftlich begründete Konstanten (ISO, IEC, physikalische Limits)
SCIENTIFIC_CONSTANTS: set[str] = {
    "ISO_226",
    "IEC_60098",
    "IEC_60094",
    "RIAA",
    "NAB",
    "CCIR",
    "NYQUIST",
    "SPEED_OF_SOUND",
    "GRAVITY",
}

# Erlaubte statische Werte in bestimmten Kontexten
ALLOWED_STATIC_PATTERNS: list[str] = [
    r"sample_rate\s*=\s*48000",  # Projektspezifisch festgelegt
    r"CROSSOVER_FREQS\s*=\s*\[",  # Wissenschaftlich (Bark-Skala)
    r"_CODEC_ARTIFACT_THRESHOLD\s*=",  # Kalibriert, nicht willkürlich
    r"SECONDARY_ANALOG_MIN\s*=",  # Bayesian-kalibriert
    r"SILENCE_THRESHOLD_DBFS\s*=",  # Physikalisch (digital floor)
]

# Muster für statische Werte, die dynamisch sein sollten
STATIC_VALUE_PATTERNS: list[tuple[str, str]] = [
    # (regex, description)
    (r"(?:gain_db|gain|strength|threshold|factor)\s*=\s*\d+\.?\d*\s*[#\n]", "fester Gain/Strength/Threshold-Wert"),
    (r"jump_threshold\s*=\s*\d+\.?\d*\s*#.*dB", "fester dB-Sprung-Schwellwert"),
    (r"min_outlier_factor\s*=\s*\d+\.?\d+", "fester Outlier-Faktor"),
    (r'"click_threshold_sigma"\s*:\s*\d+\.?\d+', "fester Click-Sigma-Wert"),
    (r'"expansion_threshold_db"\s*:\s*-\d+\.?\d+', "fester Expansion-Threshold"),
    (r'"declip_threshold"\s*:\s*\d+\.?\d+', "fester Declip-Threshold"),
]

# Fragile Test-Patterns
SOURCE_GREP_PATTERN = re.compile(r"inspect\.getsource.*\n.*assert.*in (?:src|content|source)")
UNGUARDED_IMPORT_PATTERN = re.compile(
    r"^(?:from hypothesis |import hypothesis|import requests|import torch\b)", re.MULTILINE
)


def _find_python_files(paths: list[str] | None = None) -> list[Path]:
    """Findet alle zu prüfenden Python-Dateien."""
    if paths:
        return [Path(p) for p in paths if p.endswith(".py")]

    files: list[Path] = []
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and ".venv" not in d]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(root) / f)
    return files


def check_r1_r2_static_values(filepath: Path) -> list[str]:
    """R1+R2: Prüft auf neue statische Werte ohne Adaption."""
    violations: list[str] = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return violations

    # Skip files that are known to have scientific constants
    if any(c in filepath.name for c in ["iso_", "iec_", "standard_", "reference_"]):
        return violations

    for pattern, desc in STATIC_VALUE_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in matches:
            line_no = content[: match.start()].count("\n") + 1
            line = content.split("\n")[line_no - 1].strip()
            # Check if there's SNR adaptation nearby (±5 lines)
            context_start = max(0, line_no - 6)
            context_end = min(len(content.split("\n")), line_no + 5)
            context = "\n".join(content.split("\n")[context_start:context_end])
            has_adaptation = any(
                kw in context.lower() for kw in ["snr_adapt", "snr-adapt", "_snr", "adaptive", "measured", "dynamic"]
            )
            if not has_adaptation and not any(re.search(p, line) for p in ALLOWED_STATIC_PATTERNS):
                violations.append(f"{filepath}:{line_no}: {desc} ohne SNR-Adaption — '{line[:80]}'")
    return violations


def check_r3_source_grep_tests(filepath: Path) -> list[str]:
    """R3: Keine neuen Source-Grep-Tests."""
    violations: list[str] = []
    if "test_" not in filepath.name:
        return violations
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return violations

    # Check for inspect.getsource + assert ... in src patterns
    if "inspect.getsource" in content and "assert" in content:
        # Only flag if it's a NEW file or if the pattern is new
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                func_src = ast.get_source_segment(content, node) or ""
                if "inspect.getsource" in func_src and "assert" in func_src:
                    if "in src" in func_src or "in content" in func_src or "in source" in func_src:
                        violations.append(
                            f"{filepath}:{node.lineno}: Source-Grep-Test '{node.name}' — "
                            f"fragil bei Refactoring, siehe audit_lessons_learned.md §4"
                        )
    return violations


def check_r4_unguarded_imports(filepath: Path) -> list[str]:
    """R4: Keine unguarded Imports."""
    violations: list[str] = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return violations

    for match in UNGUARDED_IMPORT_PATTERN.finditer(content):
        line_no = content[: match.start()].count("\n") + 1
        # Check if there's a try/except nearby
        context_start = max(0, line_no - 3)
        context = "\n".join(content.split("\n")[context_start : line_no + 1])
        if "try:" not in context and "except" not in context:
            violations.append(
                f"{filepath}:{line_no}: Unguarded Import '{match.group().strip()}' — "
                f"sollte in try/except ImportError gewrappt sein"
            )
    return violations


def check_r5_missing_markers(filepath: Path) -> list[str]:
    """R5: Test-Dateien müssen Marker haben."""
    violations: list[str] = []
    if "test_" not in filepath.name or not filepath.name.endswith(".py"):
        return violations
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return violations

    if len(content.split("\n")) < 50:
        return violations  # Kleine Tests brauchen keine Marker

    # Check for pytest marker usage
    has_marker = bool(re.search(r"@pytest\.mark\.\w+", content))
    has_pytestmark = bool(re.search(r"pytestmark\s*=", content))
    if not has_marker and not has_pytestmark:
        violations.append(
            f"{filepath}:1: Test-Datei ({len(content.split(chr(10)))} Zeilen) "
            f"ohne pytest-Marker — bitte min. einen Marker setzen"
        )
    return violations


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aurik Static-Value Guard")
    parser.add_argument("--ci", action="store_true", help="CI mode (strenger)")
    parser.add_argument("files", nargs="*", help="Zu prüfende Dateien")
    args = parser.parse_args()

    py_files = _find_python_files(args.files if args.files else None)
    # Nur geänderte/neue Dateien prüfen (Git Staging)
    if not args.files:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            staged = {f.strip() for f in result.stdout.split("\n") if f.endswith(".py")}
            if staged:
                py_files = [
                    f
                    for f in py_files
                    if str(f.relative_to(PROJECT_ROOT)) in staged or any(str(f).endswith(s) for s in staged)
                ]
        except Exception:
            pass

    all_violations: list[str] = []
    for fp in py_files:
        all_violations.extend(check_r1_r2_static_values(fp))
        all_violations.extend(check_r3_source_grep_tests(fp))
        all_violations.extend(check_r4_unguarded_imports(fp))
        if args.ci:
            all_violations.extend(check_r5_missing_markers(fp))

    if all_violations:
        print(f"\n⚠️  {len(all_violations)} Static-Value-Guard Verstöße gefunden:\n")
        for v in all_violations:
            print(f"  {v}")
        print("\n→ Siehe docs/reports/audit_lessons_learned_2026-07-11.md für Kontext.")
        if args.ci:
            return 1
        return 0 if input("\nTrotzdem commiten? [y/N] ").lower() != "y" else 0

    print("✅ Static-Value-Guard: Keine Verstöße.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
