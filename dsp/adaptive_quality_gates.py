import logging
import numpy as np

logger = logging.getLogger(__name__)
"""
Aurik 6.0 - Adaptive Quality Gates (Vorlage)
Ermöglicht dynamische Anpassung der Prüfgrenzen je nach Musikstil, Zielvorgabe oder Nutzerfeedback.
"""


def adaptive_hf_gate(hf_ratio: float, style: str = "default") -> bool:
    # Beispiel: Dynamische Grenzwerte je nach Stil
    thresholds = {
        "default": (0.15, 0.35),
        "klassik": (0.10, 0.25),
        "pop": (0.18, 0.40),
        "jazz": (0.12, 0.30),
    }
    low, high = thresholds.get(style, thresholds["default"])
    return low <= hf_ratio <= high


def adaptive_corr_gate(corr: float, min_corr: float = 0.98) -> bool:
    # Beispiel: Anpassbarer Mindestwert
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    return corr >= min_corr


# Beispiel für Integration in die Pipeline
if __name__ == "__main__":
    # Beispielwerte für manuellen Test
    hf = 0.22
    corr = 0.99
    logger.info("HF-Gate (Pop): %s", adaptive_hf_gate(hf, style="pop"))
    logger.info("Korrelation-Gate: %s", adaptive_corr_gate(corr, min_corr=0.97))
