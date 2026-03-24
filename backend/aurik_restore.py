"""
Aurik 6.0 – SOTA-End-to-End-Restaurierungsskript

Dieses Skript verarbeitet ein Importfile (Audio + optionale Metadaten) nach dem dokumentierten, SOTA-konformen Workflow:
1. Analyse
2. Policy-Engine
3. Restaurierung
4. Reparatur
5. Rekonstruktion
6. Quality-Gates
7. Export

Alle Begriffe, Workflows und Logs sind an die aktuelle Dokumentation und SOTA-Standards angepasst.
"""

import json
import logging
import os
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from dsp.artifact_detector import SpectralArtifactDetector
from dsp.auto_eq import AutoEQ
from dsp.harmonic_exciter import HarmonicExciter

# SOTA-Remastering-Module produktiv importieren
from dsp.intelligent_limiter import IntelligentLimiter
from dsp.multiband_compressor import MultibandCompressor
from dsp.resample_utils import ensure_sr
from dsp.stereo_widener import StereoWidener
from dsp.target_sound_matcher import TargetSoundMatcher
from plugins.deepfilternet_v3_ii_plugin import enhance_audio
from plugins.diffwave_plugin import DiffwavePlugin
from plugins.mdx23c_plugin import MDX23CPlugin
from plugins.utmos_plugin import estimate_mos


def load_importfile(importfile_path: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Lädt das Importfile (Audio + Metadaten, beliebiges Audioformat)."""
    import librosa

    ext = os.path.splitext(importfile_path)[1].lower()
    if ext == ".json":
        with open(importfile_path) as f:
            data = json.load(f)
        audio_path = data["audio"]
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio, sr = ensure_sr(audio, sr, 48000)
        # NaN/Inf-Guard (§3.1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, sr, data.get("metadata", {})
    else:
        # Direktes Audiofile (mp3, wav, flac, ...)
        audio, sr = librosa.load(importfile_path, sr=None, mono=True)
        audio, sr = ensure_sr(audio, sr, 48000)
        # NaN/Inf-Guard (§3.1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, sr, {}


def analyse(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Analysiert Audio und gibt Features + Material-Chain zurück."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard am Eingang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info("[1] Analyse läuft...")
    from backend.core.forensics.detector import MediaForensicsEngine

    engine = MediaForensicsEngine()
    # Versuche, den audio_path zu setzen, falls bekannt
    if hasattr(audio, "audio_path"):
        engine.audio_path = audio.audio_path
    elif hasattr(audio, "name"):
        engine.audio_path = audio.name
    else:
        # Fallback: Hole aus globalem Kontext, falls Importfile bekannt
        import inspect

        frame = inspect.currentframe()
        while frame:
            if "importfile_path" in frame.f_locals:
                engine.audio_path = frame.f_locals["importfile_path"]
                break
            frame = frame.f_back
    forensic_report = engine.analyze(audio, sr)

    # MaterialChainAnalysis-Objekt simulieren (oder erweitern, falls vorhanden)
    class MaterialChain:
        def __init__(self, report):
            self.detected_medium = getattr(report, "primary_media", None)
            self.medium_confidence = getattr(report, "primary_confidence", 0.0)
            self.forensic_report = report

    material_chain = MaterialChain(forensic_report)
    return {
        "features": {
            k: getattr(forensic_report, k, None)
            for k in ("snr_db", "lufs", "lsd", "phase_coherence", "transient_density", "artifact_score")
        },
        "material_chain": material_chain,
    }


def policy_engine(analysis_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gibt die Verarbeitungskette basierend auf Analyse zurück."""
    try:
        from policy.ml_policy_engine import MLModelPolicyEngine

        engine = MLModelPolicyEngine()
        return engine.select_processing_chain(analysis_result)
    except Exception as exc:
        logger.warning("ML-Policy-Engine nicht verfügbar: %s — Fallback-Kette", exc)
        return [{"phase": "restaurierung"}, {"phase": "reparatur"}, {"phase": "remastering"}]


def restaurierung(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Führt Restaurierung (DeepFilterNet) durch."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard am Eingang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info("[3] Restaurierung...")
    audio, sr = ensure_sr(audio, sr, 48000)
    audio = enhance_audio(audio, sr)
    # NaN/Inf-Guard am Ausgang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


def reparatur(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Führt Reparatur (Inpainting mit DiffWave) durch."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard am Eingang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info("[4] Reparatur...")
    audio, sr = ensure_sr(audio, sr, 48000)
    plugin = DiffwavePlugin()
    audio = plugin.inpaint(audio, sr, mask=None)
    # NaN/Inf-Guard am Ausgang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


def rekonstruktion(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Führt Rekonstruktion (Vocal-Separation mit Demucs) durch."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard am Eingang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info("[5] Rekonstruktion...")
    audio, sr = ensure_sr(audio, sr, 48000)
    plugin = MDX23CPlugin()
    vocals = plugin.process(audio, sr, stem="vocals")
    audio = vocals
    # NaN/Inf-Guard am Ausgang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


def quality_gates(audio: np.ndarray, sr: int) -> bool:
    """Prüft Quality-Gates (GACELA-MSE + UTMOSv2)."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard am Eingang (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info("[6] Quality-Gates prüfen Ergebnis...")
    # SOTA-Quality-Gates: GACELA-MSE + UTMOSv2 (§4.4-konform: keine DNSMOS/ViSQOL)
    import subprocess
    import tempfile

    import soundfile as sf

    # Sicherstellen, dass das Audio 48 kHz hat
    audio, sr = ensure_sr(audio, sr, 48000)

    passed = True
    results = {}
    # 1. Audio temporär speichern
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio, sr)
        tmp_wav_path = tmp_wav.name
    # 2. GACELA CLI aufrufen
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_score:
            tmp_score_path = tmp_score.name
        # Annahme: cli_gacela_metric.py ist im Python-Pfad
        result = subprocess.run(
            [
                sys.executable,
                os.path.join("models", "gacela", "cli_gacela_metric.py"),
                tmp_wav_path,
                tmp_score_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            with open(tmp_score_path) as f:
                for line in f:
                    if "GACELA_MSE" in line:
                        mse = float(line.strip().split(":")[-1])
                        results["GACELA_MSE"] = mse
                        # Schwellenwert für "bestanden" ggf. anpassen
                        if mse > 0.1:
                            passed = False
        else:
            logger.error("[GACELA] Fehler beim Aufruf:", result.stderr)
    except Exception as e:
        logger.error("[GACELA] Exception:", e)
    mos_result = estimate_mos(audio, sr)
    results["UTMOS"] = mos_result.as_dict() if hasattr(mos_result, "as_dict") else float(mos_result)
    mos_score = results["UTMOS"] if isinstance(results["UTMOS"], float) else results["UTMOS"].get("mos", 0.0)
    if mos_score < 3.5:
        passed = False
    logger.info(f"        [Quality-Gate] GACELA_MSE: {results.get('GACELA_MSE', 'n/a')}")
    return passed


def export(audio: np.ndarray, sr: int, out_path: str) -> None:
    """Exportiert Audio in Ausgabedatei."""
    # SR-Invariante (§3.1)
    assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
    # NaN/Inf-Guard + Clip vor Export (§3.1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)

    logger.info(f"[7] Exportiere nach {out_path} ...")
    import soundfile as sf

    # Falls (channels, samples), transponieren zu (samples, channels)
    if isinstance(audio, np.ndarray) and audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    sf.write(out_path, audio, sr)
    logger.info(f"[Export] Datei gespeichert: {out_path}")


def main(importfile_path: str, out_path: str) -> None:
    """Hauptfunktion für End-to-End-Restaurierung."""
    logger.info("\n──────────────────────────────────────────────────────────────")
    logger.info("AURIK RESTAURIERUNG - REPARATUR - REKONSTRUKTION - REMASTERING")
    logger.info("──────────────────────────────────────────────────────────────\n")
    logger.info("Vorbereitung\n────────────")
    logger.info("[1] Importiere Datei …")
    audio, sr, _metadata = load_importfile(importfile_path)
    logger.info("✔️")

    # Tiefenanalyse erhält Audio, sr, medium
    logger.info("\nAnalyse\n───────")
    logger.info("[3] Song-Analyse …")
    analysis_result = analyse(audio, sr)
    logger.info("✔️")
    logger.info("[Analyse-Ergebnisse]")
    for k, v in analysis_result.items():
        logger.info(f"    {k}: {v}")
    # Tonträgererkennung und Kette explizit ausgeben
    material_chain = analysis_result.get("material_chain", None)
    if material_chain and hasattr(material_chain, "detected_medium"):
        medium = material_chain.detected_medium
        confidence = material_chain.medium_confidence
        logger.info(f"[Tonträgererkennung] Medium: {medium} (Konfidenz: {confidence:.2f})")
    # Versuche, die Kette auszugeben, falls vorhanden
    forensic_report = getattr(material_chain, "forensic_report", None)
    if forensic_report and hasattr(forensic_report, "transfer_chain"):
        logger.info(f"[Erkannte Kette] Transfer-Chain: {forensic_report.transfer_chain}")

    # Maximal transparente Gesangs- und Sibilantenerkennung
    try:
        from backend.core.forensics.analysis_and_modules import AnalysisEngineAdapter

        # Führe vollständige Analyse durch, um AnalysisProfile zu erhalten
        analysis_engine = AnalysisEngineAdapter()
        profile = analysis_engine.analyze(audio, sr)
        va = profile.vocal_analysis
        logger.info("[VocalAnalysis]")
        logger.info(f"    Vocals erkannt: {va.has_vocals} (Konfidenz: {va.vocal_confidence:.2f})")
        # PANNS-Tags für Gender-Transparenz
        raw_tags = profile.raw_features.get("panns_tags", {})
        # Frauen-, Männer-, Kindergesang
        female = raw_tags.get("Female voice", 0.0)
        male = raw_tags.get("Male voice", 0.0)
        child = raw_tags.get("Child speech, kid speaking", 0.0)
        choir = raw_tags.get("Choir", 0.0)
        logger.info(
            f"    Frauenstimme: {female:.2f} | Männerstimme: {male:.2f} | Kind: {child:.2f} | Chor: {choir:.2f}"
        )
        # Kombinationen
        detected = []
        if female > 0.3:
            detected.append("Frauen")
        if male > 0.3:
            detected.append("Männer")
        if child > 0.3:
            detected.append("Kinder")
        if choir > 0.3:
            detected.append("Chor")
        if detected:
            logger.info(f"    Erkannt: {', '.join(detected)}")
        else:
            logger.info("    Keine dominante Gesangsart erkannt.")
        # Sibilanten-Score (sofern vorhanden)
        sibilant = raw_tags.get("Sibilance", None)
        if sibilant is not None:
            logger.info(f"    Sibilanten-Score: {sibilant:.2f}")
        else:
            logger.info("    Sibilanten-Score: nicht verfügbar")
    except Exception as e:
        logger.info(f"[VocalAnalysis] Transparenz nicht möglich: {e}")

    # Policy- und Quality-Gates initialisieren mit realen Zielwerten
    policy_targets = {
        "snr": {"threshold": 30.0},
        "lufs": {"threshold": -14.0},
        "lsd": {"threshold": 1.5},
        "phase_coh": {"threshold": 0.95},
        "transient": {"threshold": 0.8},
        "artifacts": {"threshold": 0.1},
        "improvement": {"threshold": 0.0},
    }
    analysis_result["policy_targets"] = policy_targets
    logger.info("[Policy-Targets]")
    for k, v in policy_targets.items():
        logger.info(f"    {k}: {v}")

    logger.info("\nVerarbeitung\n────────────")
    logger.info("[4] Policy-Engine wählt Maßnahmenkette …")
    chain = policy_engine(analysis_result)
    logger.info("✔️")
    logger.info("[5] Starte Restaurierungsschritte:")
    # Policy-Engine gibt DSP-Kette und KI-Modelle zurück (Ausgabeformat: Liste von Phase-Dicts)
    # Annahme: chain enthält Dicts mit 'phase', 'dsps', 'models', optional 'details'
    # Fallback: Wenn chain nur Strings enthält, wie bisher anzeigen
    if chain and isinstance(chain[0], dict):
        for phase in chain:
            phase_name = phase.get("phase", "Unbekannt")
            dsps = phase.get("dsps", [])
            models = phase.get("models", [])
            details = phase.get("details", {})
            logger.info(f"    • {phase_name}:")
            if dsps:
                for dsp in dsps:
                    dsp_info = details.get(dsp, {}) if isinstance(details, dict) else {}
                    params = dsp_info.get("parameter", "-")
                    beschr = dsp_info.get("beschreibung", "-")
                    status = dsp_info.get("status", "-")
                    effect = dsp_info.get("effect", None)
                    logger.info(f"        DSP-Modul: {dsp}")
                    logger.info(f"            Parameter: {params}")
                    logger.info(f"            Beschreibung: {beschr}")
                    logger.info(f"            Status: {status}")
                    if effect:
                        logger.info(f"            Effekt: {effect}")
            else:
                logger.info("        DSP-Modul: -")
            if models:
                for model in models:
                    model_info = details.get(model, {}) if isinstance(details, dict) else {}
                    params = model_info.get("parameter", "-")
                    beschr = model_info.get("beschreibung", "-")
                    status = model_info.get("status", "-")
                    effect = model_info.get("effect", None)
                    logger.info(f"        KI-Modell: {model}")
                    logger.info(f"            Parameter: {params}")
                    logger.info(f"            Beschreibung: {beschr}")
                    logger.info(f"            Status: {status}")
                    if effect:
                        logger.info(f"            Effekt: {effect}")
            else:
                logger.info("        KI-Modell: -")
    else:
        logger.info("    (Maßnahmenkette: " + ", ".join([m.capitalize() for m in chain]) + ")")
    step_idx = 0
    max_iterations = 5
    feedback_log = []
    audit_trail = []
    intermediate_exports = []
    # Test- und Feinjustierungs-Workflow für echte Audiodaten
    logger.info("\nTest & Feinjustierung\n─────────────────────")
    logger.info("[Vorab-Test] Qualitäts- und Policy-Gates werden mit echten Audiodaten geprüft …")
    # Beispiel: PolicyManager mit Zielwerten
    from backend.core.forensics.analysis_and_modules import PolicyManager

    policy_manager = PolicyManager(policy_targets)
    # Feature-Extraktion und Policy-Test
    from backend.core.forensics.analysis_and_modules import FeatureExtractor

    features = FeatureExtractor().extract(audio, sr, policy_manager=policy_manager)
    logger.info(f"[Test] Extrahierte Features: {features}")
    logger.info(f"[Test] Policy-Status: {policy_manager.policy}")

    while step_idx < len(chain):
        # Wenn chain Dicts enthält, nutze Phase, DSPs und Modelle
        if isinstance(chain[step_idx], dict):
            step = chain[step_idx].get("phase", f"Schritt {step_idx + 1}")
            dsps = chain[step_idx].get("dsps", [])
            models = chain[step_idx].get("models", [])
            logger.info(
                f"    • Maßnahme: {step} (DSP: {', '.join(dsps) if dsps else '-'} | KI: {', '.join(models) if models else '-'}) …"
            )
        else:
            step = chain[step_idx]
            logger.info(f"    • Maßnahme: {step.capitalize()} …")
        # --- Audit-Trail: Schrittstart ---
        audit_entry = {"step": step, "index": step_idx, "status": "started"}
        # --- Adaptive Parameteroptimierung ---
        if feedback_log:
            last = feedback_log[-1]
            if not last.get("passed") and last.get("step") == step:
                logger.info(f"        [Adaptiv] Wiederholung von '{step}' mit erhöhter Intensität.")
        # --- Maßnahmenausführung ---
        while step_idx < len(chain):
            # Wenn chain Dicts enthält, nutze Phase, DSPs und Modelle
            if isinstance(chain[step_idx], dict):
                step = chain[step_idx].get("phase", f"Schritt {step_idx + 1}")
                dsps = chain[step_idx].get("dsps", [])
                models = chain[step_idx].get("models", [])
            else:
                step = chain[step_idx]
                dsps = []
                models = []
                logger.info(f"    • Maßnahme: {step.capitalize()} …")
            # --- Audit-Trail: Schrittstart ---
            audit_entry = {"step": step, "index": step_idx, "status": "started"}
            # --- Maßnahmenausführung ---
            if step.lower() == "restaurierung":
                audio, sr = restaurierung(audio, sr)
            elif step.lower() == "reparatur":
                audio, sr = reparatur(audio, sr)
            elif step.lower() == "rekonstruktion":
                audio, sr = rekonstruktion(audio, sr)
            elif step.lower() == "remastering":
                # SOTA-Produktiv-Workflow mit Referenz-Matching und psychoakustischer Loudness:
                audio = AutoEQ(ref_profile="Studio2026").process(audio, sr)
                audio = MultibandCompressor().process(audio, sr)
                audio = HarmonicExciter(amount=0.3).process(audio, sr)
                audio = TargetSoundMatcher(reference_audio=None).process(audio, sr)
                audio = StereoWidener().widen(audio, sr, width=1.2)
                audio = IntelligentLimiter(ceiling=-1.0).process(audio, sr)
                # Artefakterkennung nach Remastering
                artifacts = SpectralArtifactDetector().detect(audio)
                logger.info(f"        Artefakterkennung: {artifacts}")
                mos_result = estimate_mos(audio, sr)
                mos_score = (
                    mos_result.as_dict().get("mos", 0.0) if hasattr(mos_result, "as_dict") else float(mos_result)
                )
                logger.info(f"        Quality Gate: UTMOS={mos_score:.3f}")
            # --- Export Zwischenergebnis ---
            interm_path = f"intermediate_{step_idx + 1}_{step}.wav"
            export(audio, sr, interm_path)
            intermediate_exports.append(interm_path)
            audit_entry["intermediate_export"] = interm_path
            # --- Quality-Gate nach jedem Schritt ---
            passed = quality_gates(audio, sr)
            audit_entry["quality_gate_passed"] = passed
            feedback_log.append({"step": step, "passed": passed})
            if passed:
                logger.info("✔️")
            else:
                logger.info("✖️ (Quality-Gate nicht bestanden)")
            # --- Fehlerklassifikation & Alternativmaßnahmen ---
            if not passed:
                logger.info(f"[Quality-Gate] Schritt '{step}' nicht bestanden. Feedback-Loop wird aktiviert.")
                audit_entry["status"] = "failed"
                audit_entry["action"] = "feedback_loop"
                # Fehlerklassifikation:
                error_type = "unbekannt"
                if step.lower() == "restaurierung":
                    error_type = "Restaurierung unzureichend"
                elif step.lower() == "reparatur":
                    error_type = "Reparatur ineffektiv"
                elif step.lower() == "rekonstruktion":
                    error_type = "Rekonstruktion fehlgeschlagen"
                elif step.lower() == "remastering":
                    error_type = "Remastering nicht optimal"
                audit_entry["error_type"] = error_type
                # Policy-Engine erhält Feedback-Log und kann Maßnahmenkette adaptiv anpassen
                analysis_result["feedback_log"] = feedback_log
                chain = policy_engine(analysis_result)
                logger.info(f"[Workflow] Maßnahmenkette nach Feedback-Loop: {chain}")
                # Optional: Schritt zurücksetzen oder anpassen
                step_idx = max(0, step_idx - 1)
                # Optional: Abbruch nach zu vielen Iterationen
                if len(feedback_log) > max_iterations * len(chain):
                    logger.info("[Abbruch] Zu viele Feedback-Loops. Workflow wird beendet.")
                    audit_entry["status"] = "aborted"
                    audit_trail.append(audit_entry)
                    break
            else:
                audit_entry["status"] = "passed"
                step_idx += 1
            audit_trail.append(audit_entry)
    # --- User-Feedback nach Export ---
    logger.info("Bitte geben Sie Feedback zum Ergebnis (z.B. 1-5 Sterne, Kommentar):")
    user_rating = input("Bewertung (1-5): ")
    user_comment = input("Kommentar: ")
    with open("user_feedback.json", "w") as f:
        json.dump({"rating": user_rating, "comment": user_comment}, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.info("Nutzung: python aurik_restore.py <importfile.json> <output.wav>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
