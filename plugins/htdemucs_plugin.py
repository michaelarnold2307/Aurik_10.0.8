"""
HTDemucs Plugin Facade — Routes to MDX23CPlugin (Kim_Vocal_2).

Provides ``get_htdemucs_plugin()`` as bridge-compatible accessor (§9.7.4).
Delegates to ``plugins.mdx23c_plugin.get_mdx23c_plugin()``.
§4.4: MDX23C (Kim_Vocal_2) ersetzt HTDemucs 6s als Primär-Separator.

Author: Aurik Development Team
Version: 9.10.57
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.mdx23c_plugin import MDX23CPlugin

logger = logging.getLogger(__name__)

try:
    from plugins.mdx23c_plugin import get_mdx23c_plugin as _get_mdx23c

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.debug("htdemucs_plugin: MDX23CPlugin not available")


def get_htdemucs_plugin() -> MDX23CPlugin | None:
    """Return MDX23C plugin (Kim_Vocal_2) as stem separator, or None."""
    if not _AVAILABLE:
        return None
    return _get_mdx23c()
