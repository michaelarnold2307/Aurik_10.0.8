#!/usr/bin/env python3
"""V01–V50 VERBOTEN-Linter v4 — vollständig an VERBOTEN.md angepasst.

Abdeckung: 15 von 26 regex-detectable Regeln (AST/Runtime-Regeln separat).
Referenz: .github/VERBOTEN.md — Linter-Referenz-Tabelle.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════════════════
# Rule definitions: (pattern, description, skip-patterns, severity)
# ═══════════════════════════════════════════════════════════════════════════
RULES: dict[str, dict] = {
    # ── V01: print() in production code (§Teil-A Logging) ──────────────
    "V01": {
        "p": r"\bprint\s*\(",
        "d": "print() statt logger — Produktionscode",
        "skip": {
            "test_",
            "conftest",
            "__init__",
            "scripts/",
            "setup.py",
            "examples/",
            "audit/",
            "Aurik10/",
            "benchmarks/",
            "cli/",
            "golden_samples/",
            "usability/",
            "policy/",
            "tests/",
        },
        "sev": "ERROR",
    },
    # ── V-BRIDGE (Bridge-Bypass): from backend.core import ─────────────
    "V-BRIDGE": {
        "p": r"from backend\.core import|import backend\.core\.",
        "d": "Bridge-Bypass-Verbot",
        "skip": {
            "bridge",
            "__init__",
            "conftest",
            "denker/",
            "api/",
            "policy/",
            "Aurik10/",
            "tests/",
            "plugins/",
            "scripts/",
        },
        "sev": "ERROR",
    },
    # ── V02: sf.read() / librosa.load() statt load_audio_file() ────────
    "V02": {
        "p": r"\b(sf\.read\s*\(|librosa\.load\s*\()",
        "d": "sf.read/librosa.load statt load_audio_file()",
        "skip": {
            "test_",
            "exporter.py",
            "audio_exporter.py",
            "file_import",
            "generate_dummy",
            "conftest",
            "audit/",
            "benchmarks/",
            "scripts/",
            "Aurik10/",
            "golden_samples",
            "tests/",
        },
        "sev": "ERROR",
    },
    # ── V03: map_location="cuda" ohne ml_device_manager ────────────────
    "V03": {
        "p": r'map_location\s*=\s*["\']cuda["\']',
        "d": "map_location=cuda ohne ml_device_manager",
        "skip": {"ml_device_manager", "test_"},
        "sev": "ERROR",
    },
    # ── V05: griffinlim() als Endschritt ───────────────────────────────
    "V05": {
        "p": r"\bgriffinlim\s*\(",
        "d": "griffinlim() — PGHI/Vocos verwenden",
        "skip": {"test_", "VERBOTEN", "docs", "spec", "scripts/"},
        "sev": "ERROR",
    },
    # ── V08: np.max(np.abs(audio)) in Gain-Pfad ────────────────────────
    "V08": {
        "p": r"np\.max\s*\(\s*np\.abs\s*\(\s*audio\s*\)\s*\)",
        "d": "np.max(abs(audio)) — np.percentile(99.9) verwenden",
        "skip": {
            "test_",
            "exporter.py",
            "export_guard",
            "audio_exporter.py",
            "peak_guard",
            "dsp/",
            "forensics/",
            "safety_wrappers/",
            "golden_samples",
            "scripts/",
            "metrics",
            "classifier",
            "detector",
            "monitor",
            "generator",
            "persistence/",
            "quality_prediction",
            "naturalness",
            "emotional_resonance",
        },
        "sev": "WARNING",
    },
    # ── V09: consecutive_rollbacks += in Carrier-Repair ────────────────
    "V09": {
        "p": r"consecutive_rollbacks\s*\+=",
        "d": "consecutive_rollbacks in Carrier-Repair-Phase",
        "skip": {"test_", "feedback_chain", "cumulative_interaction_guard"},
        "sev": "ERROR",
    },
    # ── V11: sosfilt() ohne sosfiltfilt() im selben File ───────────────
    "V11": {
        "p": r"\bsosfilt\s*\(",
        "negate": r"\bsosfiltfilt\s*\(",
        "d": "sosfilt ohne sosfiltfilt — destruktive Interferenz möglich",
        "skip": {
            "test_",
            "exporter.py",
            "_export_nuance",
            "dsp/",
            "forensics/",
            "safety_wrappers/",
            "denker/",
            "metrics",
            "classifier",
            "detector",
            "feature_",
            "genre_",
            "mushra",
            "bark_",
            "allpass",
            "psychoacoustics",
            "benchmarks",
            "processing/",
            "adaptive_plugins",
            "consonant_enhancement",
            "defect_analysis",
            "defect_scanner",
            "emergency_restoration",
            "ki_hearing_model",
            "perceptual_export_optimizer",
            "perceptual_intensity",
            "production_enhancements",
            "sub_stem_processor",
            "phase_03_denoise",
            "phase_09_crackle",
            "phase_34_mid_side",
            "phase_40_loudness",
            "run_amrb",
            "scripts/",
            "verify_all",
        },
        "sev": "WARNING",
    },
    # ── V14: Speech-Metrik (PESQ/STOI/DNSMOS/NISQA) ───────────────────
    "V14": {
        "p": r"(?:^|\s)(?:PESQ|pesq|SI[.-]SDR|si_sdr|STOI|stoi|DNSMOS|NISQA|VISQOL.*Speech)\b",
        "d": "Speech-Metrik — PQS-MOS/VERSA verwenden",
        "skip": {
            "test_",
            "VERBOTEN",
            "forbidden",
            "benchmark",
            "sota_eval",
            "spec",
            "docs",
            "quality_",
            "dsp/",
            "config/",
            "policy/",
            "audit/",
            "multi_pass",
            "musical_goals",
            "hyperparameter",
        },
        "sev": "ERROR",
    },
    # ── V21: Truncation ohne Dither ────────────────────────────────────
    "V21": {
        "p": r"int\s*\(\s*audio.*\)\s*(?:#.*(?:ohne|without|kein).*(?:dither|noise.shape))",
        "d": "Truncation ohne Dither",
        "skip": {"exporter.py", "test_", "dither"},
        "sev": "ERROR",
    },
    # ── V27: JITTER_ARTIFACTS → phase_12 (falsch) ─────────────────────
    "V27": {
        "p": r"JITTER_ARTIFACTS.*phase_12_wow_flutter",
        "d": "JITTER mit phase_12 — phase_14+23 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V28: NR_BREATHING_ARTIFACT → phase_03/29 (falsch) ─────────────
    "V28": {
        "p": r"NR_BREATHING_ARTIFACT.*phase_(?:0[3]|29)",
        "d": "NR_ATEMARTEFAKT mit NR-Phase — phase_54 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V29: OVERLOAD_DISTORTION → phase_63 (falsch) ───────────────────
    "V29": {
        "p": r"OVERLOAD_DISTORTION.*phase_63",
        "d": "OVERLOAD mit phase_63 — phase_09+23 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V30: ALIASING → phase_03 (falsch) ─────────────────────────────
    "V30": {
        "p": r"ALIASING.*phase_03",
        "d": "ALIASING mit phase_03 — phase_23+50 korrekt",
        "skip": {"test_", "VERBOTEN", "docs", "scripts/"},
        "sev": "ERROR",
    },
    # ── V39: §0a-forbidden Phasen für Restoration ─────────────────────
    "V39": {
        "p": r"Restoration.*(?:phase_21_exciter|phase_35_multiband|phase_42_vocal)",
        "d": "§0a-verbotene Phase für Restoration-Cause",
        "skip": {"test_", "VERBOTEN", "docs", "causal_defect_reasoner", "scripts/", "cumulative_interaction_guard"},
        "sev": "ERROR",
    },
    # ── V44: IACC ohne Stereo-Guard ────────────────────────────────────
    "V44": {
        "p": r"IACC\s*[<>]\s*0\.[0-9]+",
        "d": "IACC ohne Stereo-Guard",
        "skip": {"stereo_guard", "test_", "musical_goals_metrics", "perceptual_salience"},
        "sev": "INFO",
    },
    # ── V46: dBFS mit linearem Skalierungsfaktor ───────────────────────
    "V46": {
        "p": r"level_db\s*\*\s*strength|dbFS\s*\*\s*(?:0\.\d+|strength)",
        "d": "dBFS linear skaliert — 10^(dB/20) verwenden",
        "skip": {"test_", "dsp/"},
        "sev": "ERROR",
    },
}

SKIP_DIRS = {".venv", ".venv_aurik", "__pycache__", "node_modules", ".git", "models/", "temp_repro/"}

# ═══════════════════════════════════════════════════════════════════════════
# File-scope skip: certain file patterns are globally excluded per rule
# ═══════════════════════════════════════════════════════════════════════════
GLOBAL_FILE_SKIP: dict[str, set[str]] = {
    # Skip production-only rules for test/script/doc files globally
    "prod_only": {"test_", "/tests/", "setup.py", "conftest.py", ".github/", "docs/", "scripts/"},
    # Skip for plugins/sdk abstract base classes
    "sdk": {"plugins/sdk/"},
}


def _should_skip_rule(fp: Path, rid: str) -> bool:
    """Check if file should be skipped for a given rule."""
    r = str(fp)
    if any(s in r for s in SKIP_DIRS):
        return True
    rule = RULES.get(rid)
    if rule is None:
        return True
    return any(s in r for s in rule.get("skip", set()))


def scan(fp: Path) -> list[str]:
    """Scan one Python file for VERBOTEN violations."""
    if fp.suffix != ".py":
        return []
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    # Strip docstrings and comments (keep code only for regex matching)
    code_lines: list[str] = []
    in_ds = False
    for line in lines:
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_ds = not in_ds
            continue
        if in_ds:
            continue
        if s.startswith("#"):
            continue
        code_lines.append(line)

    code_text = "\n".join(code_lines)
    rel = fp.relative_to(_PROJECT_ROOT)

    issues: list[str] = []
    for rid, rule in RULES.items():
        if _should_skip_rule(fp, rid):
            continue
        if re.search(rule["p"], code_text, re.IGNORECASE | re.MULTILINE):
            # Check negate pattern: if present, only flag when negate does NOT match
            negate_pat = rule.get("negate")
            if negate_pat and re.search(negate_pat, code_text, re.IGNORECASE | re.MULTILINE):
                continue  # negate pattern found → file uses correct form, skip
            sev = rule.get("sev", "WARNING")
            issues.append(f"{rid}[{sev}]: {rule['d']} — {rel}")

    return issues


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VERBOTEN-Linter v4")
    parser.add_argument("--ci", action="store_true", help="CI mode: exit 1 on issues")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--errors-only", action="store_true", help="Show only ERROR severity")
    args = parser.parse_args()

    all_issues: dict[str, list[str]] = {}
    py_files = list(_PROJECT_ROOT.rglob("*.py"))
    for pf in py_files:
        iss = scan(pf)
        if iss:
            all_issues[str(pf.relative_to(_PROJECT_ROOT))] = iss

    total = sum(len(v) for v in all_issues.values())

    if args.json:
        errors = sum(1 for v in all_issues.values() for i in v if "ERROR" in i)
        warnings = total - errors
        print(
            json.dumps(
                {
                    "clean": total == 0,
                    "issues": total,
                    "errors": errors,
                    "warnings": warnings,
                    "files_scanned": len(py_files),
                    "rules_active": len(RULES),
                }
            )
        )
        return 0

    if total:
        if args.errors_only:
            err_issues = {f: [i for i in iss if "[ERROR]" in i] for f, iss in all_issues.items()}
            err_issues = {f: iss for f, iss in err_issues.items() if iss}
            if err_issues:
                err_total = sum(len(v) for v in err_issues.values())
                print(f"\n{err_total} ERRORS in {len(err_issues)} files:")
                for f, iss in sorted(err_issues.items()):
                    for i in iss:
                        print(f"  {i}")
            else:
                print("VERBOTEN-Linter: no errors")
            return 0
        print(f"\n{total} issues in {len(all_issues)} files:")
        for f, iss in sorted(all_issues.items()):
            for i in iss:
                print(f"  {i}")
        return 1 if args.ci else 0
    else:
        print("VERBOTEN-Linter: clean")
        return 0


if __name__ == "__main__":
    sys.exit(main())
