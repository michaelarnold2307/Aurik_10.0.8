"""
Quality Metrics Manager - Aurik 9.x
====================================

Zentrale Integration aller §4.4-konformen Audio-Qualitätsmetriken für Musik:
- CDPAM: Contrastive Deep Perceptual Audio Metric (Vollband-Musik, 22 050 Hz)
- ViSQOL v3 --audio: Reference-based Perceptual Quality (ITU-Standard, Musik-Modus zwingend)
- PQS-DSP: Gammatone-NSIM+MCD+LUFS (Aurik-eigener Musik-Scorer, §2.6)

Ausdrücklich VERBOTEN (§4.4+§10.2): DNSMOS, NISQA, PESQ, STOI, POLQA.
Diese Sprach-Metriken funktionieren systematisch falsch auf Musik.

Verwendung:
    manager = QualityMetricsManager()

    # Non-reference metrics (nur degraded audio benötigt)
    scores = manager.assess_quality("output.wav")

    # Reference-based metric (clean + degraded audio)
    visqol = manager.assess_visqol("reference.wav", "degraded.wav")
"""

import json
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

# Import Plugins (nur spec-konforme Musik-Metriken — §10.2)
# VERSA 2024 ersetzt CDPAM als non-reference MOS-Metrik (§4.4)
try:
    from plugins.versa_plugin import get_versa_plugin as _get_versa_plugin_fn
    _VERSA_IMPORT_OK = True
except ImportError:  # Fallback: PQS-DSP
    _VERSA_IMPORT_OK = False
    _get_versa_plugin_fn = None  # type: ignore[assignment]
from plugins.visqol_plugin import ViSQOLPlugin

# PQS-DSP: musik-orientierter Gammatone-NSIM+MCD+LUFS-Scorer (§2.6)
try:
    from backend.core.perceptual_quality_scorer import score_audio_absolute as _pqs_score

    _PQS_AVAILABLE = True
except Exception:
    _PQS_AVAILABLE = False
    _pqs_score = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class QualityMetricsManager:
    """
    Zentraler Manager für alle objektiven Quality-Metriken.

    # DNSMOS/NISQA entfernt — verboten §4.4+§10.2 (Sprach-Metriken)
    # CDPAM ersetzt durch VERSA 2024 (referenzfreie MOS, §4.4)
    Orchestriert VERSA und ViSQOL (--audio) für musik-konforme Audio-Qualitätsbewertung.
    """

    def __init__(self, enable_all: bool = True):
        """
        Initialize Quality Metrics Manager.

        Args:
            enable_all: Wenn False, Plugins lazy initialisieren
        """
        self._versa = None
        self._visqol = None

        if enable_all:
            self._init_plugins()

    def _init_plugins(self):
        """Initialize all plugins."""
        try:
            if _VERSA_IMPORT_OK and _get_versa_plugin_fn is not None:
                self._versa = _get_versa_plugin_fn()
                logger.info("✓ VERSA Plugin loaded (§4.4 CDPAM-Nachfolger)")
        except Exception as e:
            logger.warning("VERSA Plugin nicht verfügbar: %s — PQS-DSP Fallback.", e)

        try:
            self._visqol = ViSQOLPlugin()
            logger.info("✓ ViSQOL Plugin loaded")
        except Exception as e:
            logger.warning("ViSQOL Plugin nicht verfügbar: %s", e)

        if _PQS_AVAILABLE:
            logger.info("✓ PQS-DSP (Gammatone-NSIM+MCD) verfügbar")
        else:
            logger.debug("PQS-DSP nicht verfügbar — VERSA als non-reference Metrik aktiv.")

    @property
    def versa(self):
        """Lazy-load VERSA plugin (§4.4 CDPAM-Nachfolger)."""
        if self._versa is None and _VERSA_IMPORT_OK and _get_versa_plugin_fn is not None:
            self._versa = _get_versa_plugin_fn()
        return self._versa

    # Backward-Kompatibilität
    @property
    def cdpam(self):
        """Alias auf versa (§4.4: CDPAM durch VERSA ersetzt)."""
        return self.versa

    @property
    def visqol(self):
        """Lazy-load ViSQOL plugin (--audio mode zwingend, §4.4)."""
        if self._visqol is None:
            self._visqol = ViSQOLPlugin()
        return self._visqol

    def assess_quality(
        self,
        audio_file: str,
        output_dir: str | None = None,
        metrics: list | None = None,
    ) -> dict[str, Any]:
        """Qualitätsbewertung mit musik-spezifischen non-reference Metriken (§4.4, §10.2).

        Verwendet ausschließlich für Musik zertifizierte Metriken:
          - CDPAM (Vollband-Musik-Wahrnehmungsqualität, Vollband 22 050 Hz)
          - PQS-DSP (Gammatone-NSIM+MCD+LUFS, aurikeigener Musik-Scorer, §2.6)

        DNSMOS, NISQA, PESQ und STOI sind für Musik verboten (§10.2) und werden
        hier weder geladen noch aufgerufen.

        Args:
            audio_file: Pfad zur Audio-Datei
            output_dir: Ausgabeverzeichnis für JSON-Dateien (optional)
            metrics:    Liste der zu verwendenden Metriken ['cdpam', 'pqs'].
                        Wenn None → beide musik-konformen Metriken verwenden.

        Returns:
            Dict mit allen Scores und aggregiertem Qualitäts-Score.
        """
        if metrics is None:
            metrics = ["cdpam", "pqs"]  # Nur musik-konforme non-reference Metriken!

        results = {"audio_file": Path(audio_file).name, "metrics": {}, "aggregate": {}}

        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. VERSA — referenzfreie MOS-Metrik (§4.4, CDPAM-Nachfolger)
        #    Ergebnis erscheint weiterhin unter Key "cdpam" für Backward-Kompatibilität.
        if "cdpam" in metrics:
            try:
                import soundfile as _sf  # noqa: PLC0415

                audio_np_v, sr_v = _sf.read(audio_file, always_2d=False)
                audio_np_v = audio_np_v.astype(np.float32)
                if audio_np_v.ndim == 2:
                    audio_np_v = audio_np_v.mean(axis=1)
                audio_np_v = np.nan_to_num(audio_np_v, nan=0.0, posinf=0.0, neginf=0.0)

                versa_plugin = self.versa
                if versa_plugin is not None:
                    versa_result = versa_plugin.score(audio_np_v, sr_v)
                    mos_raw = float(np.nan_to_num(versa_result.mos, nan=1.0, posinf=5.0, neginf=1.0))
                else:
                    # PQS-Gammatone DSP-Fallback
                    mos_raw = 3.0  # Neutral default
                # MOS [1,5] → Score [0,100] für Backward-Kompatibilität
                score_raw = float(np.clip((mos_raw - 1.0) / 4.0 * 100.0, 0.0, 100.0))
                score_norm = float(np.clip(score_raw / 100.0, 0.0, 1.0))
                results["metrics"]["cdpam"] = {
                    "score": score_raw,
                    "normalized": score_norm,
                    "mos": mos_raw,
                    "model_used": getattr(versa_result, "model_used", "dsp_fallback") if versa_plugin else "dsp_fallback",
                    "rating": self._rate_cdpam(score_raw),
                }
                logger.info("✓ VERSA-MOS: %.2f/5 → CDPAM-kompatibel: %.1f/100", mos_raw, score_raw)
            except Exception as e:
                logger.warning("VERSA assessment fehlgeschlagen: %s", e)
                results["metrics"]["cdpam"] = {"error": str(e)}

        # 2. PQS-DSP — Gammatone-NSIM+MCD+LUFS (Aurik-eigener Musik-Scorer, §2.6)
        #    score_audio_absolute() benötigt kein Referenzsignal.
        if "pqs" in metrics and _PQS_AVAILABLE and _pqs_score is not None:
            try:
                import soundfile as _sf

                audio_np, sr = _sf.read(audio_file, always_2d=False)
                if audio_np.ndim == 2:
                    audio_np = audio_np.mean(axis=1)
                audio_np = audio_np.astype(np.float32)
                # NaN/Inf-Guard am Eingang (§3.1)
                audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
                pqs_result = _pqs_score(audio_np, sr)
                mos = float(getattr(pqs_result, "mos", 0.0))
                nsim = float(getattr(pqs_result, "nsim", 0.0))
                mcd = float(getattr(pqs_result, "mcd_db", 99.0))
                coh = float(getattr(pqs_result, "spectral_coherence", 0.0))
                # NaN/Inf-Guards (§3.1)
                mos = float(np.nan_to_num(mos, nan=0.0, posinf=5.0, neginf=0.0))
                nsim = float(np.nan_to_num(nsim, nan=0.0, posinf=1.0, neginf=0.0))
                mcd = float(np.nan_to_num(mcd, nan=99.0, posinf=99.0, neginf=0.0))
                coh = float(np.nan_to_num(coh, nan=0.0, posinf=1.0, neginf=0.0))
                # Normierung: MOS 1–5 → 0–1; MCD ≤ 3 dB → 1.0, ≥ 8 dB → 0.0
                mcd_norm = float(np.clip((8.0 - mcd) / 5.0, 0.0, 1.0))
                pqs_norm = 0.40 * nsim + 0.30 * mcd_norm + 0.15 * coh + 0.15 * ((mos - 1.0) / 4.0)
                pqs_norm = float(np.clip(pqs_norm, 0.0, 1.0))
                results["metrics"]["pqs"] = {
                    "mos": mos,
                    "nsim": nsim,
                    "mcd_db": mcd,
                    "spectral_coherence": coh,
                    "normalized": pqs_norm,
                    "rating": self._rate_mos(mos),
                }
                logger.info("✓ PQS-DSP: MOS=%.2f NSIM=%.3f MCD=%.1f dB", mos, nsim, mcd)
            except Exception as e:
                logger.warning("PQS-DSP assessment fehlgeschlagen: %s", e)
                results["metrics"]["pqs"] = {"error": str(e)}

        # Aggregate Scores
        results["aggregate"] = self._compute_aggregate(results["metrics"])

        return results

    def assess_visqol(
        self,
        reference_file: str,
        degraded_file: str,
        mode: str = "audio",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        ViSQOL Reference-based Quality Assessment.

        Args:
            reference_file: Pfad zum sauberen Reference-Audio
            degraded_file: Pfad zum zu bewertenden Audio
            mode: "audio" (48kHz wideband) oder "speech" (8kHz narrowband)
            output_dir: Ausgabeverzeichnis für JSON-Dateien (optional)

        Returns:
            Dict mit ViSQOL MOS-LQO Score
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            visqol_json = output_path / "visqol_scores.json"
            visqol_scores = self.visqol.calculate(reference_file, degraded_file, str(visqol_json), mode=mode)

            mos_raw = visqol_scores.get("ViSQOL_MOS", 0.0)
            # NaN/Inf-Guard (§3.1)
            mos_raw = float(np.nan_to_num(mos_raw, nan=0.0, posinf=5.0, neginf=0.0))
            mos_norm = float(np.clip(mos_raw / 5.0, 0.0, 1.0))
            result = {
                "ViSQOL_MOS": mos_raw,
                "normalized": mos_norm,
                "rating": self._rate_mos(mos_raw),
                "mode": mode,
                "reference_file": Path(reference_file).name,
                "degraded_file": Path(degraded_file).name,
            }

            logger.info(f"✓ ViSQOL: MOS-LQO={mos_raw:.2f}/5.0")
            return result

        except Exception as e:
            logger.warning(f"ViSQOL assessment failed: {e}")
            return {"error": str(e)}

    def assess_comprehensive(
        self,
        audio_file: str,
        reference_file: str | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Umfassende Qualitätsbewertung mit allen verfügbaren Metriken.

        Args:
            audio_file: Pfad zur Audio-Datei
            reference_file: Optional - für ViSQOL reference-based assessment
            output_dir: Ausgabeverzeichnis für JSON-Dateien

        Returns:
            Dict mit allen Scores und Gesamt-Qualitätsrating
        """
        # Non-reference metrics
        results = self.assess_quality(audio_file, output_dir)

        # Reference-based metric (wenn verfügbar)
        if reference_file is not None:
            try:
                visqol_result = self.assess_visqol(reference_file, audio_file, output_dir=output_dir)
                results["metrics"]["visqol"] = visqol_result

                # Update aggregate mit ViSQOL
                results["aggregate"] = self._compute_aggregate(results["metrics"], include_visqol=True)
            except Exception as e:
                logger.warning(f"ViSQOL nicht verfügbar: {e}")

        return results

    def _compute_aggregate(self, metrics: dict[str, Any], include_visqol: bool = False) -> dict[str, Any]:
        """
        Berechne aggregierte Qualitäts-Scores.

        Kombiniert alle verfügbaren Metriken zu einem Gesamt-Score.
        """
        scores = []
        weights = {}

        # CDPAM (Gewicht: 0.50) — primäre Musik-Wahrnehmungsmetrik (§4.4)
        if "cdpam" in metrics and "normalized" in metrics["cdpam"]:
            scores.append(metrics["cdpam"]["normalized"])
            weights["cdpam"] = 0.50

        # PQS-DSP (Gewicht: 0.50) — Gammatone-NSIM+MCD+LUFS-Musik-Proxy (§2.6)
        if "pqs" in metrics and "normalized" in metrics["pqs"]:
            scores.append(metrics["pqs"]["normalized"])
            weights["pqs"] = 0.50

        # ViSQOL (wenn verfügbar, Gewichte 3-wege aufteilen)
        if include_visqol and "visqol" in metrics and "normalized" in metrics["visqol"]:
            scores.append(metrics["visqol"]["normalized"])
            weights = {
                "cdpam": 0.35,
                "pqs": 0.35,
                "visqol": 0.30,
            }

        # Normalisiere Gewichte
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Gewichteter Durchschnitt
        if scores:
            weighted_scores = []
            for metric_name, weight in weights.items():
                if metric_name in metrics and "normalized" in metrics[metric_name]:
                    weighted_scores.append(metrics[metric_name]["normalized"] * weight)

            aggregate_score = sum(weighted_scores)
            # NaN/Inf-Guard (§3.1)
            aggregate_score = float(np.nan_to_num(aggregate_score, nan=0.0, posinf=1.0, neginf=0.0))
            aggregate_score = float(np.clip(aggregate_score, 0.0, 1.0))
        else:
            aggregate_score = 0.0

        return {
            "overall_score": aggregate_score,
            "overall_rating": self._rate_aggregate(aggregate_score),
            "weights": weights,
            "num_metrics": len(scores),
        }

    def _rate_cdpam(self, score: float) -> str:
        """Rate CDPAM score (0-100)."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Poor"

    def _rate_mos(self, score: float) -> str:
        """Rate MOS score (1-5)."""
        if score >= 4.5:
            return "Excellent"
        elif score >= 4.0:
            return "Very Good"
        elif score >= 3.5:
            return "Good"
        elif score >= 3.0:
            return "Fair"
        else:
            return "Poor"

    def _rate_aggregate(self, score: float) -> str:
        """Rate aggregate score (0-1)."""
        if score >= 0.9:
            return "Weltklasse"
        elif score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Very Good"
        elif score >= 0.6:
            return "Good"
        else:
            return "Needs Improvement"

    def generate_report(self, results: dict[str, Any], output_file: str | None = None) -> str:
        """
        Generiere formatierten Quality Report.

        Args:
            results: Ergebnisse von assess_quality() oder assess_comprehensive()
            output_file: Optional - speichere Report als JSON

        Returns:
            Formatierter Report als String
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QUALITY METRICS REPORT — Aurik 9.x (§4.4 musik-konforme Metriken)")
        report_lines.append("=" * 80)
        report_lines.append(f"\nAudio File: {results['audio_file']}")
        report_lines.append("\n" + "-" * 80)

        # Individual Metrics
        metrics = results.get("metrics", {})

        if "cdpam" in metrics and "score" in metrics["cdpam"]:
            cdpam = metrics["cdpam"]
            report_lines.append("\n1. CDPAM (Perceptual Quality)")
            report_lines.append(f"   Score: {cdpam['score']:.2f}/100")
            report_lines.append(f"   Rating: {cdpam['rating']}")

        if "pqs" in metrics and "mos" in metrics["pqs"]:
            pqs = metrics["pqs"]
            report_lines.append("\n2. PQS-DSP (Gammatone-NSIM+MCD+LUFS — Musik-Proxy, §2.6)")
            report_lines.append(f"   MOS:                {pqs['mos']:.2f}/5.0")
            report_lines.append(f"   NSIM:               {pqs['nsim']:.3f}")
            report_lines.append(f"   MCD:                {pqs['mcd_db']:.1f} dB")
            report_lines.append(f"   Spectral Coherence: {pqs['spectral_coherence']:.3f}")
            report_lines.append(f"   Rating: {pqs['rating']}")

        if "visqol" in metrics and "ViSQOL_MOS" in metrics["visqol"]:
            visqol = metrics["visqol"]
            report_lines.append("\n3. ViSQOL v3 --audio (Reference-based Perceptual)")
            report_lines.append(f"   MOS-LQO: {visqol['ViSQOL_MOS']:.2f}/5.0")
            report_lines.append(f"   Mode: {visqol.get('mode', 'audio')}")
            report_lines.append(f"   Rating: {visqol['rating']}")

        # Aggregate Score
        aggregate = results.get("aggregate", {})
        if "overall_score" in aggregate:
            report_lines.append("\n" + "-" * 80)
            report_lines.append("\nAGGREGATE QUALITY SCORE")
            report_lines.append(
                f"   Overall: {aggregate['overall_score']:.3f}/1.0 ({aggregate['overall_score']*100:.1f}%)"
            )
            report_lines.append(f"   Rating: {aggregate['overall_rating']}")
            report_lines.append(f"   Based on: {aggregate['num_metrics']} metrics")

        report_lines.append("\n" + "=" * 80)

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"✓ Report saved: {output_file}")

        return report


# ==============================================================================
# Convenience Functions
# ==============================================================================


def assess_audio_quality(audio_file: str, output_dir: str | None = None) -> dict[str, Any]:
    """
    Quick assessment mit allen non-reference Metriken.

    Args:
        audio_file: Pfad zur Audio-Datei
        output_dir: Optional - Ausgabeverzeichnis

    Returns:
        Quality metrics dict
    """
    manager = QualityMetricsManager()
    return manager.assess_quality(audio_file, output_dir)


def assess_with_reference(reference_file: str, degraded_file: str, output_dir: str | None = None) -> dict[str, Any]:
    """
    Assessment mit ViSQOL reference-based metric.

    Args:
        reference_file: Sauberes Reference-Audio
        degraded_file: Zu bewertendes Audio
        output_dir: Optional - Ausgabeverzeichnis

    Returns:
        ViSQOL scores dict
    """
    manager = QualityMetricsManager()
    return manager.assess_visqol(reference_file, degraded_file, output_dir=output_dir)


def comprehensive_assessment(audio_file: str, reference_file: str | None = None, output_dir: str | None = None) -> None:
    """
    Vollständiges Assessment mit Report-Ausgabe.

    Args:
        audio_file: Pfad zur Audio-Datei
        reference_file: Optional - für ViSQOL
        output_dir: Optional - Ausgabeverzeichnis
    """
    manager = QualityMetricsManager()
    results = manager.assess_comprehensive(audio_file, reference_file, output_dir)
    report = manager.generate_report(results)
    logger.info("Quality report:\n%s", report)

    return results


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(str("\n" + "=" * 80))
    logger.info("QUALITY METRICS MANAGER TEST")
    logger.info("=" * 80 + "\n")

    # Check if audio file provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        reference_file = sys.argv[2] if len(sys.argv) > 2 else None

        comprehensive_assessment(audio_file, reference_file)
    else:
        logger.info("Usage: python quality_metrics_manager.py <audio_file> [reference_file]")
        logger.info("\nExample:")
        logger.info("  python quality_metrics_manager.py output.wav")
        logger.info("  python quality_metrics_manager.py output.wav reference.wav")
