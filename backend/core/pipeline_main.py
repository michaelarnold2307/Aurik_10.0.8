"""
Aurik – Hauptpipeline
======================

Zwei Einstiegspunkte:

1. AurikAutonomousPipeline (PRIMÄR / empfohlen ab v9.0)
   Vollautomatisch. Nutzer wählt NUR den Modus:
     - ProcessingMode.RESTORATION  → Authentizität, Originalcharakter erhalten
     - ProcessingMode.STUDIO_2026  → Moderner Streaming-Sound, maximale Brillanz
   Alles andere (Material, Defekte, Kette, Qualitätskontrolle) läuft intern.

2. AurikMainPipeline (Legacy / rückwärtskompatibel)
   Verknüpft ImportPipeline + UnifiedRestorerV2 wie in Aurik 6.0.

Author: Aurik Development Team
Version: 9.0.0 "Zero-Intervention Excellence"
"""

import json
import logging
import os

import numpy as np

from backend.core.autonomous_restoration_engine import (
    AutonomousRestorationEngine,
    AutonomousRestorationResult,
)
from backend.core.core_utils import log_message
from backend.core.processing_modes import ProcessingMode

logger = logging.getLogger(__name__)

_AUDIT_LOG_PATH = "logs/pipeline_audit_log.ndjson"


# ---------------------------------------------------------------------------
# PRIMÄRE PIPELINE (ab Aurik 9.0)
# ---------------------------------------------------------------------------


class AurikAutonomousPipeline:
    """
    Vollautomatische Aurik-Hauptpipeline.

    Einzige Nutzereingabe: Modus (RESTORATION | STUDIO_2026).
    Alles andere geschieht intern und transparent.

    Verwendung:
        pipeline = AurikAutonomousPipeline(mode=ProcessingMode.RESTORATION)
        result = pipeline.process(audio, sample_rate=44100)
        restored_audio = result.audio
    """

    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.RESTORATION,
        enable_self_learning: bool = True,
    ):
        """
        Args:
            mode: RESTORATION (Authentizität) oder STUDIO_2026 (Modern/Streaming).
                  Das ist die EINZIGE Entscheidung, die der Nutzer trifft.
            enable_self_learning: Lernt aus jedem Ergebnis für zukünftige Sessions.
        """
        self.mode = mode
        self._engine = AutonomousRestorationEngine(
            mode=mode,
            enable_self_learning=enable_self_learning,
        )
        self._session_results: list[AutonomousRestorationResult] = []
        os.makedirs("logs", exist_ok=True)
        logger.info("AurikAutonomousPipeline bereit | Modus: %s", mode.value)

    def process(self, audio: np.ndarray, sample_rate: int) -> AutonomousRestorationResult:
        """
        Vollautomatische Restaurierung.

        Args:
            audio:       Eingabe-Audio (float32 numpy-Array, mono oder stereo).
            sample_rate: Abtastrate in Hz.

        Returns:
            AutonomousRestorationResult mit restauriertem Audio, vollständigem
            Protokoll und allen Qualitätsmetriken.
        """
        result = self._engine.process(audio, sample_rate)
        self._session_results.append(result)
        self._append_audit(result)

        log_message(
            f"[Aurik] Modus={result.mode.value} | Material={result.material_type.value} "
            f"| Variante={result.winning_variant} "
            f"| Q: {result.quality_before:.1f}→{result.quality_after:.1f} "
            f"| SNR Δ={result.improvement_db:+.2f} dB "
            f"| Rollback={'JA' if result.rollback_triggered else 'NEIN'}"
        )
        return result

    def get_session_summary(self) -> dict:
        """Kurzübersicht aller bisherigen Ergebnisse dieser Session."""
        if not self._session_results:
            return {"session_results": 0}
        deltas = [r.improvement_db for r in self._session_results]
        return {
            "session_results": len(self._session_results),
            "mode": self.mode.value,
            "avg_snr_improvement_db": round(sum(deltas) / len(deltas), 2),
            "rollbacks": sum(1 for r in self._session_results if r.rollback_triggered),
            "materials_seen": list({r.material_type.value for r in self._session_results}),
        }

    def _append_audit(self, result: AutonomousRestorationResult) -> None:
        """Schreibt das Audit-Trail in die NDJSON-Logdatei."""
        try:
            entry = {
                "mode": result.mode.value,
                "material": result.material_type.value,
                "winning_variant": result.winning_variant,
                "quality_before": result.quality_before,
                "quality_after": result.quality_after,
                "improvement_db": result.improvement_db,
                "rollback": result.rollback_triggered,
                "processing_time_s": result.processing_time_seconds,
                "passes": result.passes_executed,
            }
            with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Audit-Log konnte nicht geschrieben werden: %s", exc)


# ---------------------------------------------------------------------------
# LEGACY PIPELINE (Aurik 6.0 / 8.0 — rückwärtskompatibel)
# ---------------------------------------------------------------------------


class AurikMainPipeline:
    """
    Legacy-Pipeline (Aurik 6.0/8.0).

    Für neue Projekte bitte AurikAutonomousPipeline verwenden.
    """

    def __init__(self, policy_template=None, config=None):
        from backend.core.import_pipeline import ImportPipeline
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        self.importer = ImportPipeline(policy_template=policy_template)
        self.restorer = UnifiedRestorerV3()
        self.config = config or {}
        self.audit_log = []

    def process(self, audio: np.ndarray, sr: int, reference: np.ndarray = None):
        features, policy = self.importer.import_audio(audio, sr, reference)
        self.audit_log.append({"step": "import", "features": features, "policy": policy})
        log_message(f"MainPipeline: Import abgeschlossen {features}")
        result = self.restorer.restore(audio, sample_rate=sr, mode="restoration")
        restored = result.audio if hasattr(result, "audio") and result.audio is not None else audio
        self.audit_log.append({"step": "restoration", "policy": policy})
        log_message("MainPipeline: Restaurierung abgeschlossen")
        os.makedirs("logs", exist_ok=True)
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            for entry in self.audit_log:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        return restored, self.audit_log
