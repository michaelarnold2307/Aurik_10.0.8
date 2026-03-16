"""
AdaptiveChainRouter: Wählt und konfiguriert Verarbeitungsketten basierend auf Forensik-Analyse und Confidence.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


_instance: Optional["AdaptiveChainRouter"] = None
_lock = threading.Lock()


def get_adaptive_chain_router(templates: dict[str, list[str]] | None = None) -> "AdaptiveChainRouter":
    """Get or create AdaptiveChainRouter singleton.

    Args:
        templates: Chain templates dict (only used on first call)

    Returns:
        AdaptiveChainRouter singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AdaptiveChainRouter(templates or CHAIN_TEMPLATES)
    return _instance


class AdaptiveChainRouter:
    def __init__(self, templates: dict[str, list[str]]) -> None:
        self.templates = templates
        logger.info(f"AdaptiveChainRouter initialized with {len(templates)} templates")

    def select_chain(self, forensic_report: dict[str, str], confidence: float) -> list[str]:
        material = forensic_report.get("medium_type", "GENERIC").upper()
        chain = self.templates.get(material, self.templates.get("GENERIC", []))
        # Optional: Anpassung der Kette je nach Confidence
        if confidence < 0.6:
            chain = [m for m in chain if m != "Enhancement"]
        return chain

    def configure_modules(self, chain: list[str], forensic_report: dict[str, str]) -> dict[str, dict[str, float | str | bool]]:
        config = {}
        for module in chain:
            # Beispiel: Material- und Defekt-spezifische Parameter
            params = {}
            if module == "Denoiser" and forensic_report.get("defects_detected") == "NOISE_BURST":
                params["strength"] = 0.9
            config[module] = params
        return config


# Beispiel-Templates
CHAIN_TEMPLATES = {
    "VINYL": [
        "DCBlocker",
        "ClickRemover",
        "RumbleFilter",
        "HumRemover",
        "WowFlutterCorrection",
        "Denoiser",
        "Enhancement",
    ],
    "TAPE": [
        "DCBlocker",
        "TapeAzimuthCorrector",
        "PrintThroughRemover",
        "WowFlutterCorrection",
        "HFRestoration",
        "Denoiser",
    ],
    "CD": ["DCBlocker", "CodecArtifactRemover", "PreEchoSuppressor", "SpectralHoleFiller", "Dynamics", "Enhancement"],
    "DIGITAL": ["DCBlocker", "CodecArtifactRemover", "SpectralHoleFiller", "Dynamics", "Enhancement"],
    "GENERIC": ["DCBlocker", "Denoiser", "Enhancement"],
}
