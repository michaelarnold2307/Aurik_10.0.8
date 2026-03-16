"""
PerceptualQualityGates: Integration von ViSQOL (--audio), CDPAM (§4.4-konforme Musik-Metriken).
# NISQA/DNSMOS entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
Automatisches Reprocessing und Multi-Pass-Optimierung.
"""

from collections.abc import Callable
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class PerceptualQualityGates:
    """Perceptual Quality Gates für Musik-Restaurierung (§4.4-konform).

    Integriert:
    - ViSQOL v3 (--audio mode, MOS 1.0–5.0)
    - CDPAM (Cross-Domain Perceptual Audio Metrics)

    Verboten:
    - NISQA, DNSMOS, PESQ, STOI (§10.2: Sprach-Metriken)
    """

    def __init__(self, reprocess_callback: Callable[[], None]) -> None:
        """Initialisiert Perceptual Quality Gates.

        Args:
            reprocess_callback: Callback-Funktion für Reprocessing bei Failure.
        """
        # NISQA/DNSMOS entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
        self.thresholds: Dict[str, float] = {"ViSQOL": 3.5, "CDPAM": 0.7}
        self.reprocess_callback = reprocess_callback
        logger.info("PerceptualQualityGates initialized with thresholds: %s", self.thresholds)

    def evaluate(self, metrics: Dict[str, float]) -> bool:
        """Evaluiert Metriken gegen Schwellwerte.

        Args:
            metrics: Dictionary mit Metrik-Namen und Werten.

        Returns:
            True wenn alle Schwellwerte erfüllt, sonst False.
        """
        # NaN-Guard (§3.1)
        metrics_clean = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in metrics.items()}

        failed = [k for k, v in metrics_clean.items() if k in self.thresholds and v < self.thresholds[k]]
        if failed:
            logger.warning("Quality Gate FAILED: %s", failed)
            self.reprocess_callback()
            return False
        logger.info("Quality Gate PASSED")
        return True
