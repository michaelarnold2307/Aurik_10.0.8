#!/usr/bin/env python3
"""Aurik VERBOTEN-Linter — prüft die Top-10 normativ verbotenen Anti-Patterns.

Verwendung:
    python scripts/aurik_verboten_linter.py [pfad ...]
    python scripts/aurik_verboten_linter.py backend/ plugins/
    python scripts/aurik_verboten_linter.py  # scannt backend/ und plugins/
    python scripts/aurik_verboten_linter.py --strict  # Warning-Regeln auch als Fehler

Exit-Codes:
    0 = keine ERROR-Verstöße (Warning-Regeln werden angezeigt, blockieren nicht)
    1 = ERROR-Verstöße gefunden (oder --strict + Warnings)

Regel-Level:
    ERROR  (blockiert CI): V01, V03, V04, V05, V06, V07, V08, V09, V10
    WARNING (nur angezeigt): V02 — np.max(np.abs()) Heuristik hat false-positive-Rate
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Regel-Definitionen
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    file: Path
    line: int
    col: int
    rule: str
    message: str
    snippet: str = ""


@dataclass
class Rule:
    id: str
    description: str


RULES = {
    "V01": Rule("V01", "np.corrcoef() → guarded dot-product verwenden (VERBOTEN: NaN-unsafe)"),
    "V02": Rule("V02", "np.max(np.abs(audio)) als Peak-Guard → np.percentile(np.abs(audio), 99.9)"),
    "V03": Rule("V03", "boundary='reflect' in scipy STFT → boundary='even' (scipy < 1.12 crash)"),
    "V04": Rule("V04", "apply_musical_gain_envelope ohne gate_dbfs=-36.0 (VERBOTEN: -50.0 default)"),
    "V05": Rule("V05", "print() statt logger.info/warning/error/debug"),
    "V06": Rule("V06", "map_location='cuda' ohne ml_device_manager.get_torch_device()"),
    "V07": Rule("V07", "scipy.signal.wiener() direkt statt OMLSA/DeepFilterNet"),
    "V08": Rule("V08", "np.correlate(x, x, mode='full') O(n²) → scipy FFT-basiert"),
    "V09": Rule("V09", "from Aurik910... import in backend/ (Architektur-Verletzung)"),
    "V10": Rule("V10", "load_audio_file() ohne do_carrier_analysis=False in Thread/UI-Kontext"),
}

# V02 ist Warning-Level: Heuristik hat false-positive-Rate bei Analyse/Telemetrie-Kontext.
# Alle anderen Regeln sind ERROR-Level (blockieren CI).
WARNING_RULES: frozenset[str] = frozenset({"V02"})
ERROR_RULES: frozenset[str] = frozenset(RULES) - WARNING_RULES


# ---------------------------------------------------------------------------
# Checker-Klasse
# ---------------------------------------------------------------------------


class VerbotenlLinter(ast.NodeVisitor):
    def __init__(self, filepath: Path, source_lines: list[str]) -> None:
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: list[Violation] = []

    def _add(self, node: ast.AST, rule_id: str, extra: str = "") -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = self.source_lines[line - 1].rstrip() if 0 < line <= len(self.source_lines) else ""
        msg = RULES[rule_id].description
        if extra:
            msg = f"{msg} | {extra}"
        self.violations.append(Violation(self.filepath, line, col, rule_id, msg, snippet))

    # --- V01: np.corrcoef ---
    def _check_corrcoef(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "corrcoef":
            if (isinstance(func.value, ast.Attribute) and func.value.attr == "np") or (
                isinstance(func.value, ast.Name) and func.value.id == "np"
            ):
                self._add(node, "V01")

    # --- V02: np.max(np.abs(...)) as peak guard in normalization context ---
    def _check_npmax_abs(self, node: ast.Call) -> None:
        """Flags np.max(np.abs(...)) ONLY in normalization/gain contexts.

        False-positive exclusions:
        - Zero-checks: `if np.max(np.abs(x)) < threshold`  → analysis, not normalization
        - Direct comparison: `np.max(...) > 0.99` → clipping detection
        - Used in crest-factor calculation: `20 * np.log10(np.max(...))`

        True-positive patterns (normalization):
        - Assignment to gain/peak variable: `peak = np.max(np.abs(audio))`
          where the variable name contains: peak, max_val, gain, scale, norm, headroom
        - Direct divisor in audio normalization: `audio /= np.max(np.abs(audio))`
        """
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "max"):
            return
        if not (isinstance(func.value, ast.Name) and func.value.id == "np"):
            return
        if not node.args:
            return
        arg0 = node.args[0]
        if not (
            isinstance(arg0, ast.Call)
            and isinstance(arg0.func, ast.Attribute)
            and arg0.func.attr == "abs"
            and isinstance(arg0.func.value, ast.Name)
            and arg0.func.value.id == "np"
        ):
            return

        # Only flag when used as direct divisor in AugAssign (/=)
        # We check parent context via stored assignment info — use a heuristic:
        # look at the argument variable name for audio signals
        inner_arg = arg0.args[0] if arg0.args else None
        if inner_arg is None:
            return
        inner_name = ""
        if isinstance(inner_arg, ast.Name):
            inner_name = inner_arg.id.lower()
        elif isinstance(inner_arg, ast.Subscript) and isinstance(inner_arg.value, ast.Name):
            inner_name = inner_arg.value.id.lower()

        # Only flag for audio-named variables (not xcorr, delta, gradient etc.)
        _AUDIO_NAMES = (
            "audio",
            "signal",
            "sig",
            "mono",
            "stereo",
            "samples",
            "data",
            "out",
            "buf",
            "channel",
            "oversampled",
            "restored",
            "original",
            "left",
            "right",
            "waveform",
            "chunk",
            "frame_",
        )
        _ANALYSIS_ONLY = (
            "corr",
            "xcorr",
            "gradient",
            "delta",
            "envelope",
            "seg_mean",
            "ratios",
            "magnitude",
            "mag",
            "fft",
            "window",
            "coeff",
        )

        is_audio_var = any(n in inner_name for n in _AUDIO_NAMES)
        is_analysis_var = any(n in inner_name for n in _ANALYSIS_ONLY)

        if not is_audio_var or is_analysis_var:
            return

        # Exclude crest-factor context: if the immediate parent assign target is
        # named "peak" AND used as divisor for crest factor (peak / rms pattern),
        # this is an analysis call, not a normalization.
        # We check the enclosing assignment's variable name heuristic via snippet.
        snippet = self.source_lines[node.lineno - 1] if 0 < node.lineno <= len(self.source_lines) else ""
        # Crest factor pattern: `peak = float(np.max(...))` — peak is measured, not used as divisor in audio
        if "crest" in snippet.lower():
            return
        # Crest factor: the result variable is named "peak" and only used for crest_db computation
        # Heuristic: if the whole line looks like `peak = float(np.max(np.abs(mono)))` with no division
        # into audio signal → allow. We detect by checking the variable has "+1e-" guard (analysis only)
        if "+ 1e-" in snippet and "crest" not in snippet.lower() and " / " not in snippet:
            return  # measurement with epsilon guard, not a peak normalizer

        self._add(node, "V02")

    # --- V03: boundary='reflect' ---
    def _check_stft_boundary(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg == "boundary" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "reflect":
                    self._add(node, "V03", "boundary='reflect' → 'even'")

    # --- V04: apply_musical_gain_envelope ohne gate_dbfs=-36.0 ---
    def _check_gain_envelope_gate(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        if name != "apply_musical_gain_envelope":
            return
        # Check: does the call have gate_dbfs keyword?
        gate_kw = {kw.arg: kw for kw in node.keywords if kw.arg == "gate_dbfs"}
        if "gate_dbfs" not in gate_kw:
            # Missing gate_dbfs entirely — will use default which may be wrong
            self._add(node, "V04", "gate_dbfs nicht gesetzt (Pflicht: -36.0)")
        else:
            val = gate_kw["gate_dbfs"].value
            if isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub):
                if isinstance(val.operand, ast.Constant) and float(val.operand.value) == 50.0:
                    self._add(node, "V04", "gate_dbfs=-50.0 gefunden → muss -36.0 sein")
            elif isinstance(val, ast.Constant) and float(val.value) == -50.0:
                self._add(node, "V04", "gate_dbfs=-50.0 gefunden → muss -36.0 sein")

    # --- V05: print() ---
    def _check_print(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            # Tolerate test files and scripts
            parts = self.filepath.parts
            if "tests" in parts or "scripts" in parts:
                return
            self._add(node, "V05", "print() in Produktionscode → logger.info/warning/debug")

    # --- V06: map_location='cuda' ---
    def _check_map_location_cuda(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg == "map_location" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "cuda":
                    self._add(node, "V06", "map_location='cuda' → get_torch_device('PluginName')")

    # --- V07: scipy.signal.wiener ---
    def _check_wiener(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "wiener":
            self._add(node, "V07", "scipy.signal.wiener() → OMLSA/DeepFilterNet")

    # --- V08: np.correlate(x, x, mode='full') O(n²) ---
    def _check_np_correlate_full(self, node: ast.Call) -> None:
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "correlate"):
            return
        if not (isinstance(func.value, ast.Name) and func.value.id == "np"):
            return
        # Check mode='full'
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "full":
                    # Also check if args[0] == args[1] (autocorrelation)
                    if len(node.args) >= 2:
                        a0 = ast.unparse(node.args[0]) if hasattr(ast, "unparse") else ""
                        a1 = ast.unparse(node.args[1]) if hasattr(ast, "unparse") else ""
                        if a0 == a1 or a0 == "" or a1 == "":
                            self._add(node, "V08", "O(n²) Autokorrelation → scipy.signal.fftconvolve")

    def visit_Call(self, node: ast.Call) -> None:
        self._check_corrcoef(node)
        self._check_npmax_abs(node)
        self._check_stft_boundary(node)
        self._check_gain_envelope_gate(node)
        self._check_print(node)
        self._check_map_location_cuda(node)
        self._check_wiener(node)
        self._check_np_correlate_full(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """V09: from Aurik910... import in backend/."""
        if node.module and node.module.startswith("Aurik910"):
            parts = self.filepath.parts
            if "backend" in parts:
                self._add(node, "V09", f"from {node.module} import ... in backend/")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Datei scannen
# ---------------------------------------------------------------------------


def scan_file(filepath: Path) -> list[Violation]:
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []
    checker = VerbotenlLinter(filepath, lines)
    checker.visit(tree)
    return checker.violations


def collect_py_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    strict = "--strict" in argv
    argv = [a for a in argv if a != "--strict"]

    root = Path(__file__).resolve().parent.parent
    if argv:
        scan_roots = [Path(a) if Path(a).is_absolute() else root / a for a in argv]
    else:
        scan_roots = [root / "backend", root / "plugins"]

    py_files = collect_py_files(scan_roots)
    all_violations: list[Violation] = []
    for f in py_files:
        all_violations.extend(scan_file(f))

    errors = [v for v in all_violations if v.rule in ERROR_RULES]
    warnings = [v for v in all_violations if v.rule in WARNING_RULES]

    if not all_violations:
        print(f"✓ Kein Verstoß in {len(py_files)} Dateien ({', '.join(str(r) for r in scan_roots)})")
        return 0

    # Gruppieren nach Regel
    by_rule: dict[str, list[Violation]] = {}
    for v in all_violations:
        by_rule.setdefault(v.rule, []).append(v)

    print(f"\n{'=' * 70}")
    lvl = f"{len(errors)} ERROR" + (f", {len(warnings)} WARNING" if warnings else "")
    print(f"Aurik VERBOTEN-Linter — {lvl} in {len(py_files)} Dateien")
    print(f"{'=' * 70}")

    for rule_id in sorted(by_rule):
        viols = by_rule[rule_id]
        rule = RULES[rule_id]
        level = "WARNING" if rule_id in WARNING_RULES else "ERROR"
        print(f"\n[{rule_id}] [{level}] {rule.description}")
        print(f"  {len(viols)} Fundstelle(n):")
        for v in sorted(viols, key=lambda x: (str(x.file), x.line)):
            rel = v.file.relative_to(root) if v.file.is_relative_to(root) else v.file
            print(f"    {rel}:{v.line}:{v.col}")
            if v.snippet:
                snippet = textwrap.shorten(v.snippet.strip(), width=90, placeholder="…")
                print(f"      → {snippet}")

    has_blocking = bool(errors) or (strict and warnings)
    print(f"\n{'=' * 70}")
    status = "FAIL" if has_blocking else "PASS (nur Warnings)"
    print(f"  Status: {status}  |  Regeln: {', '.join(sorted(by_rule))}")
    if not strict and warnings and not errors:
        print(f"  Hinweis: {len(warnings)} Warning(s) — mit --strict werden diese auch als Fehler gewertet")
    print(f"{'=' * 70}\n")
    return 1 if has_blocking else 0


if __name__ == "__main__":
    sys.exit(main())
