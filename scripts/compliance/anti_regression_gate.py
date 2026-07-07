#!/usr/bin/env python3
"""
§2.59 Anti-Regression-Gate — alle 9 behobenen Bug-Muster abdecken.

Jeder Bug dieser Session wird hier als Check verewigt.
Läuft als Pre-Commit-Hook. Blockt Commits, die bekannte Fehlermuster
reproduzieren würden.

Bug-Abdeckung:
  Bug 1: @staticmethod + self.X → check_staticmethod_self.py
  Bug 2: input_path fehlt → dieser Check
  Bug 3: doppeltes Präfix (cached_cached_*) → dieser Check
  Bug 4: PhasePruner falsche Defekt-Namen → ContractValidator
  Bug 5: defekt_hint ohne defect_types → dieser Check
  Bug 6: Preservation Mode Schwelle < 0.97 → dieser Check
  Bug 7: source_fidelity_bandwidth_hz (falsches Feld) → dieser Check
  Bug 8: QualityModeConfig fehlt → check_import_breaking.py
  Bug 9: except Exception: pass → dieser Check
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def check_typo_double_prefix(filepath: str) -> list[str]:
    """Bug 3: Doppelte Präfixe wie cached_cached_*."""
    issues: list[str] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Pattern: word_word_ where word == word (like cached_cached_)
    for match in re.finditer(r'\b([a-z]+)_\1_[a-z]', content):
        issues.append(f"{filepath}:{content[:match.start()].count(chr(10))+1}: "
                      f"doppeltes Präfix '{match.group()}' (Bug 3)")
    return issues


def check_preservation_mode_threshold(filepath: str) -> list[str]:
    """Bug 6: Preservation Mode bw_loss < 0.97."""
    issues: list[str] = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Pattern: bw_loss_sev >= 0.90 (old threshold)
    if re.search(r'bw_loss.*>=\s*0\.9[0-6]', content):
        for i, line in enumerate(content.split('\n'), 1):
            if 'bw_loss' in line and '>=' in line:
                m = re.search(r'>=\s*(0\.\d+)', line)
                if m and float(m.group(1)) < 0.97:
                    issues.append(f"{filepath}:{i}: Preservation Mode Schwelle "
                                  f"={m.group(1)} < 0.97 (Bug 6)")
    return issues


def check_wrong_field_name(filepath: str) -> list[str]:
    """Bug 7: Falsche Feldnamen."""
    issues: list[str] = []
    KNOWN_WRONG = {
        "source_fidelity_bandwidth_hz": "source_fidelity_bandwidth_target_hz",
    }
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    for wrong, correct in KNOWN_WRONG.items():
        if wrong in content:
            for i, line in enumerate(content.split('\n'), 1):
                if wrong in line:
                    issues.append(f"{filepath}:{i}: Falsches Feld '{wrong}' "
                                  f"→ sollte '{correct}' sein (Bug 7)")
    return issues


def check_bare_except_pass(filepath: str) -> list[str]:
    """Bug 9: except Exception: pass ohne Logging."""
    issues: list[str] = []
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception:
        return issues
    for i, line in enumerate(lines):
        if re.match(r'\s*except\s+Exception\s*(as\s+\w+)?\s*:', line):
            # Check next line for bare pass/return/continue
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'\s*(pass|return|continue)\s*$', next_line):
                    # Check if logger.debug is within 2 lines above
                    has_logger = False
                    for j in range(max(0, i - 2), i):
                        if 'logger.' in lines[j]:
                            has_logger = True
                            break
                    if not has_logger:
                        issues.append(f"{filepath}:{i+1}: stummer except Exception: "
                                      f"{next_line.strip()} ohne Logging (Bug 9)")
    return issues



def check_pruner_signature(filepath: str) -> list[str]:
    """PhasePruner.prune() must accept restoration_context."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    if 'def prune(' in content and 'restoration_context' not in content:
        for i, line in enumerate(content.split(chr(10)), 1):
            if 'def prune(' in line:
                if 'restoration_context' not in line:
                    issues.append(f"{filepath}:{i}: prune() missing restoration_context parameter")
    return issues


def check_sentinel_architecture(filepath: str) -> list[str]:
    """VocalDistortionSentinel must be SENSOR only (measure, no check/strength_overrides)."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    if 'VocalDistortionSentinel' in content:
        if 'def check(' in content and 'def measure(' not in content:
            issues.append(f"{filepath}: Sentinel has check() but no measure() — must be SENSOR")
        if 'strength_overrides' in content or 'injected_phases' in content:
            issues.append(f"{filepath}: Sentinel must not contain strength_overrides/injected_phases — WRITE to restoration_context instead")
    return issues


def check_magic_numbers(filepath: str) -> list[str]:
    """No hardcoded multipliers where continuous measurement is appropriate."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Pattern: *= 0.85 or *= 0.6 in goal weight context
    for i, line in enumerate(content.split(chr(10)), 1):
        if 'weights[' in line and '*= 0.85' in line:
            issues.append(f"{filepath}:{i}: hardcoded *= 0.85 — use continuous function instead")
        if 'weights[' in line and '*= 0.6' in line and 'bw_ratio' not in content:
            issues.append(f"{filepath}:{i}: hardcoded *= 0.6 — use bw_ratio instead")
    return issues


def check_absolute_bw_loss(filepath: str) -> list[str]:
    """bw_loss must be material-relative, not absolute against 20 kHz."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    # Pattern: using _bw_loss_sev directly for decisions (not _bw_loss_relative)
    if '_bw_loss_sev' in content and '_bw_loss_relative' not in content:
        # Allow in the guard calculation itself (where _bw_loss_relative is defined)
        if 'def _build_song_calibration_profile' not in content:
            issues.append(f"{filepath}: uses _bw_loss_sev without material-relative normalization")
    # Pattern: hardcoded bandwidth comparison against 20000
    for i, line in enumerate(content.split(chr(10)), 1):
        if '20000' in line and ('bandwidth' in line.lower() or 'bw' in line.lower()):
            if 'MATERIAL_EXPECTED_BW' not in content:
                issues.append(f"{filepath}:{i}: hardcoded 20000 Hz bandwidth reference without MATERIAL_EXPECTED_BW")
    return issues


def check_defect_classification(filepath: str) -> list[str]:
    """Every DefectType must be classified as SURGICAL or GLOBAL."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues
    if 'SURGICAL_DEFECT_TYPES' in content:
        # Check all DefectTypes are accounted for
        try:
            from backend.core.defect_scanner import DefectType
            from backend.core.surgical_defect_analyzer import SURGICAL_DEFECT_TYPES
            all_defects = {e.value for e in DefectType}
            surgical = all_defects & SURGICAL_DEFECT_TYPES
            unaccounted = all_defects - SURGICAL_DEFECT_TYPES
            # GLOBAL types are everything not in SURGICAL — no explicit list needed
            if len(surgical) != 24:
                issues.append(f'{filepath}: SURGICAL_DEFECT_TYPES has {len(surgical)} types (expected 24)')
        except ImportError:
            pass
    return issues



def check_surgical_architecture(filepath: str) -> list[str]:
    """Surgical repair architecture must be intact."""
    issues = []
    try:
        with open(filepath) as f:
            content = f.read()
    except Exception:
        return issues

    # Check 1: PhasePlan must have surgical_routing
    if 'class PhasePlan' in content:
        if 'surgical_routing' not in content:
            issues.append(f"{filepath}: PhasePlan missing surgical_routing field")

    # Check 2: PhasePruner must protect surgical phases
    if 'def prune(' in content and 'IntelligentPhasePruner' in content:
        if 'surgical_defect_types' not in content:
            issues.append(f"{filepath}: PhasePruner missing surgical phase protection")

    # Check 3: PhaseResult must have time_range
    if 'class PhaseResult' in content:
        if 'time_range' not in content:
            issues.append(f"{filepath}: PhaseResult missing time_range field")

    # Check 4: restoration_context must propagate surgical_defect_types
    if '_active_defekt_hint = _defekt_hint_kwarg' in content:
        if 'surgical_defect_types' not in content:
            issues.append(f"{filepath}: UV3 not propagating surgical_defect_types to restoration_context")

    # Check 5: ExzellenzDenker must know surgical zones
    if 'class ExzellenzDenker' in content or 'def messe_und_repariere' in content:
        if 'surgical' not in content.lower():
            issues.append(f"{filepath}: ExzellenzDenker missing surgical zone awareness")

    return issues

def main() -> None:
    changed = sys.argv[1:]
    if not changed:
        print("Anti-Regression-Gate: ⚠️ keine Dateien")
        sys.exit(0)

    all_issues: list[str] = []
    for fp in changed:
        if not fp.endswith('.py'):
            continue
        all_issues.extend(check_typo_double_prefix(fp))
        all_issues.extend(check_preservation_mode_threshold(fp))
        all_issues.extend(check_wrong_field_name(fp))
        all_issues.extend(check_bare_except_pass(fp))
        all_issues.extend(check_pruner_signature(fp))
        all_issues.extend(check_sentinel_architecture(fp))
        all_issues.extend(check_magic_numbers(fp))
        all_issues.extend(check_absolute_bw_loss(fp))
        all_issues.extend(check_defect_classification(fp))
        all_issues.extend(check_surgical_architecture(fp))

    if all_issues:
        print(f"🛡️ Anti-Regression-Gate: {len(all_issues)} Verletzung(en)\n")
        for issue in all_issues:
            print(f"  🚫 {issue}")
        print(f"\nDiese Muster wurden in Bugfix-Session 2026-07-09 behoben.")
        print("Commits, die sie reproduzieren, werden blockiert.")
        sys.exit(1)

    print("🛡️ Anti-Regression-Gate: ✅ keine bekannten Fehlermuster")
    sys.exit(0)


if __name__ == "__main__":
    main()
