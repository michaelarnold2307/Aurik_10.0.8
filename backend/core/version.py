"""Zentrale Versions-Hilfsfunktion für Backend-Layer.

Liefert die Aurik-Version ohne Abhängigkeit auf Aurik10-Frontend-Module
(Architektur-Invariante V09: Backend darf nicht aus Aurik10 importieren).
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def get_aurik_version() -> str:
    """Gibt die Aurik-Versionsnummer zurück.

    Priorität:
    1. importlib.metadata (wenn als editierbares Paket installiert)
    2. pyproject.toml (immer im Repository-Root vorhanden)
    3. "unknown" als letzter Fallback
    """
    try:
        from importlib.metadata import PackageNotFoundError  # pylint: disable=import-outside-toplevel
        from importlib.metadata import version as _pkg_version  # pylint: disable=import-outside-toplevel

        # Try multiple possible package names (aurik9, aurik10, Aurik10)
        for _pkg_name in ("aurik9", "aurik10", "Aurik10"):
            try:
                return _pkg_version(_pkg_name)
            except PackageNotFoundError:
                continue
        # All package names failed — fall through to pyproject.toml
    except Exception as e:
        logger.debug("version.py::get_aurik_version pkg fallback: %s", e)
    try:
        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        content = _pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception as _exc:
        logger.debug("get_aurik_version pyproject.toml Fallback fehlgeschlagen: %s", _exc)
    return "unknown"
