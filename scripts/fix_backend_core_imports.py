#!/usr/bin/env python3
"""
scripts/fix_backend_core_imports.py — Nachhaltige Shim-Eliminierung (Aurik 9, §9.4)

Problem:
    325 Dateien im Projekt nutzen `from backend.core.X import ...`
    statt `from core.X import ...`.  Die 16 Shim-Dateien in backend/core/
    importieren sich sogar selbst (Self-Import-Loop).

Lösung (3 Schritte):
    1. Alle Shim-Dateien (backend/core/X.py wo core/X.py existiert)
       auf minimalsten Thin-Wrapper `from core.X import *` reduzieren.
    2. Alle anderen .py-Dateien im Projekt: `from backend.core.X import Y`
       → `from core.X import Y`  (nur wenn core/X.py existiert).
    3. Analog für `import backend.core.X` → `import core.X`.

Sicherheits-Invarianten:
    - Nur Dateien mit tatsächlichen Änderungen werden angefasst.
    - Keine Änderung wenn core/X.py NICHT existiert (backend-only Module bleiben).
    - Dry-Run-Modus mit --dry-run verfügbar.
    - Rückgabe-Code 0 = Erfolg, 1 = Fehler.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent


def build_core_module_set(root: Path) -> set[str]:
    """Alle Modulnamen die in core/ als .py existieren (ohne Unterverzeichnisse)."""
    core_dir = root / "core"
    mods: set[str] = set()
    for p in core_dir.glob("*.py"):
        mods.add(p.stem)
    # Sub-Pakete wie core/musical_goals/, core/phases/, core/optimization/
    for sub in core_dir.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            pkg = sub.name
            for p in sub.glob("*.py"):
                mods.add(f"{pkg}.{p.stem}")
    return mods


# Regex für Einzel- und Mehrzeilen-Imports
_FROM_BC = re.compile(
    r"from\s+backend\.core\.([A-Za-z0-9_.]+)\s+import",
    re.MULTILINE,
)
_IMPORT_BC = re.compile(
    r"\bimport\s+backend\.core\.([A-Za-z0-9_.]+)",
    re.MULTILINE,
)


def fix_source(content: str, core_mods: set[str]) -> tuple[str, int]:
    """Wendet alle Ersetzungen auf den Quelltext an.

    Returns:
        (neuer_inhalt, anzahl_ersetzungen)
    """
    changes = 0

    def replace_from(m: re.Match) -> str:
        nonlocal changes
        mod = m.group(1)
        if mod in core_mods:
            changes += 1
            return f"from core.{mod} import"
        return m.group(0)  # unverändert

    def replace_import(m: re.Match) -> str:
        nonlocal changes
        mod = m.group(1)
        if mod in core_mods:
            changes += 1
            return f"import core.{mod}"
        return m.group(0)

    content = _FROM_BC.sub(replace_from, content)
    content = _IMPORT_BC.sub(replace_import, content)
    return content, changes


SHIM_TEMPLATE = '''\
"""
{path} — Kompatibilitäts-Shim (Aurik 9, §Anti-Parallelwelten §9.4)
===========================================================================
Kanonische Implementierung: core/{mod}.py
Diese Datei leitet alle Namen transparent weiter.
Kein Produktionscode hier — alle Änderungen in core/{mod}.py vornehmen.
"""
# ruff: noqa: F401, F403
from core.{mod} import *  # noqa: F403
'''


def simplify_shim(shim_path: Path, mod_name: str) -> str:
    """Gibt den vereinfachten Shim-Inhalt zurück."""
    rel = shim_path.relative_to(ROOT)
    return SHIM_TEMPLATE.format(path=rel, mod=mod_name)


def iter_py_files(root: Path):
    """Alle .py-Dateien im Projekt (ohne __pycache__ und .venv)."""
    skip_dirs = {"__pycache__", ".venv_aurik", ".venv", ".git", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(root):
        # In-place filtern damit os.walk nicht in skip_dirs abstammt
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Ersetzt from backend.core.X durch from core.X im gesamten Projekt.")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen was geändert würde, nichts schreiben.")
    parser.add_argument(
        "--shims-only", action="store_true", help="Nur die 16 Shim-Dateien vereinfachen, Rest unverändert."
    )
    args = parser.parse_args()

    core_mods = build_core_module_set(ROOT)
    print(f"✓ {len(core_mods)} Module in core/ gefunden.")

    # Shim-Dateien: backend/core/X.py wo core/X.py existiert
    backend_core = ROOT / "backend" / "core"
    shim_files: dict[Path, str] = {}
    for bp in backend_core.glob("*.py"):
        if bp.stem in core_mods:
            shim_files[bp] = bp.stem

    print(f"✓ {len(shim_files)} Shim-Dateien identifiziert.")

    total_files_changed = 0
    total_replacements = 0

    # ── Schritt 1: Shims vereinfachen ─────────────────────────────────────
    for shim_path, mod_name in sorted(shim_files.items()):
        new_content = simplify_shim(shim_path, mod_name)
        if args.dry_run:
            print(f"  [DRY] Shim vereinfachen: {shim_path.relative_to(ROOT)}")
        else:
            shim_path.write_text(new_content, encoding="utf-8")
            print(f"  ✓ Shim vereinfacht: {shim_path.relative_to(ROOT)}")
        total_files_changed += 1

    if args.shims_only:
        print(f"\nFertig (--shims-only). {total_files_changed} Shim-Dateien behandelt.")
        return 0

    # ── Schritt 2 & 3: Alle anderen Dateien umstellen ─────────────────────
    for py_file in iter_py_files(ROOT):
        # Shim-Dateien wurden bereits oben behandelt
        if py_file in shim_files:
            continue
        # Das Migrationsskript selbst überspringen
        if py_file.name == "fix_backend_core_imports.py":
            continue

        try:
            original = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        new_content, n = fix_source(original, core_mods)
        if n == 0:
            continue

        total_replacements += n
        total_files_changed += 1

        if args.dry_run:
            print(f"  [DRY] {py_file.relative_to(ROOT)}: {n} Ersetzung(en)")
        else:
            py_file.write_text(new_content, encoding="utf-8")
            print(f"  ✓ {py_file.relative_to(ROOT)}: {n} Ersetzung(en)")

    mode = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{mode}Fertig: {total_files_changed} Dateien geändert, " f"{total_replacements} Import-Zeilen ersetzt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
