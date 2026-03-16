import logging
import os
import sys

import soundfile as sf

from backend.adaptive_pipeline import AdaptiveProcessingPipeline


def process_audio(input_path, output_path, verbose=True):
    logger = logging.getLogger("aurik_cli")
    # Demucs YAML — lokale Prüfung (kein Docker erforderlich)
    demucs_yaml_host = os.path.join("models", "demucs", "htdemucs.yaml")
    if not os.path.exists(demucs_yaml_host):
        logger.warning(
            "Demucs-Konfiguration nicht gefunden: %s — " "Stem-Separation nutzt internen DSP-Fallback.",
            demucs_yaml_host,
        )

    # MDX-Net Modell — lokale Prüfung (kein Docker erforderlich)
    mdx_model_host = os.path.join("models", "mdx_net", "mdx_net_vocal_v2.onnx")
    if not os.path.exists(mdx_model_host):
        logger.warning(
            "MDX-Net Modell nicht gefunden: %s — "
            "Stem-Separation nutzt Kim_Vocal_2/Kim_Inst ONNX oder HPSS-Fallback.",
            mdx_model_host,
        )
    logger = logging.getLogger("aurik_cli")
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    # Audio als Bytes laden (wie bei /magic_button API)
    if not os.path.exists(input_path):
        logger.error(f"Input-Datei nicht gefunden: {input_path}")
        sys.exit(2)
    try:
        with open(input_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        logger.error(f"Fehler beim Laden der Datei: {e}")
        sys.exit(3)

    # Automatisiere detected_medium und Quality-Kontext für MP3
    features = {}
    if input_path.lower().endswith(".mp3"):
        features["detected_medium"] = {"type": "mp3"}
        features["snr"] = 10
        features["quality_gates"] = {"overall": "bad"}
        features["has_vocals"] = True
        features["genre"] = "pop"
        features["has_reverb"] = True
        features["has_clipping"] = True
        features["transient_rich"] = False
        features["defects"] = ["compression_artifacts", "hf_loss"]
        features["artifacts"] = "mp3_lowpass"
        features["lufs"] = -18
        features["quality_level"] = "maximal"

    if verbose:
        logger.info(f"Größe: {len(audio_bytes) / 1024 / 1024:.2f} MB")
        logger.info("🔧 Starte Adaptive Processing Pipeline...")
        logger.info("   • Defekterkennung (11 Typen)")
        logger.info("   • Tonträgerketten-Analyse")
        logger.info("   • ML-Modell-Auswahl")
        logger.info("   • Ethics Engine")
        logger.info("   • Quality Monitoring")

    # Pipeline initialisieren und verarbeiten
    try:
        pipeline = AdaptiveProcessingPipeline()
        # detected_medium und Quality-Kontext explizit übergeben
        result = pipeline.run(
            audio_bytes,
            features=features,
            user_profile={},
            reference_audio=None,
            detected_medium=features.get("detected_medium", None),
        )
    except Exception as e:
        logger.error(f"Fehler in der Pipeline: {e}")
        sys.exit(4)

    if verbose:
        logger.info("✅ Verarbeitung abgeschlossen")

    # Ergebnis speichern — Pipeline liefert WAV-Bytes in result["steps"][-1]["audio"]
    import io

    processed_audio = None
    processed_sr = None
    steps_list = (result or {}).get("steps") or []
    if steps_list:
        raw = steps_list[-1].get("audio")
        if raw:
            try:
                processed_audio, processed_sr = sf.read(io.BytesIO(raw))
            except Exception:
                processed_audio = None

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if processed_audio is not None:
            sf.write(output_path, processed_audio, processed_sr)
        else:
            # Fallback: Original verwenden
            if verbose:
                logger.warning("Kein processed_audio - verwende Original")
            audio_orig, sr = sf.read(input_path)
            sf.write(output_path, audio_orig, sr)
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Audiodatei: {e}")
        sys.exit(5)

    if verbose:
        logger.info(f"💾 Gespeichert: {output_path}")
        # Stats anzeigen
        steps = result.get("steps", [])
        quality = result.get("quality", [])
        log = result.get("log", [])
        logger.info("📊 Verarbeitungs-Details:")
        logger.info(f"   • Steps: {len(steps)}")
        logger.info(f"   • Quality Checks: {len(quality)}")
        logger.info(f"   • Log Entries: {len(log)}")
        # Zeige angewandte Steps
        if steps:
            logger.info("   Angewandte Processing-Steps:")
            for step in steps[:5]:  # Max 5 anzeigen
                module = step.get("module", "unknown")
                logger.info(f"      - {module}")
            if len(steps) > 5:
                logger.info(f"      ... und {len(steps) - 5} weitere")
        # Erweiterte Log-Ausgabe
        if log:
            logger.info("📋 Detailliertes Log:")
            for entry in log:
                step = entry.get("step", "?")
                info = entry.get("info", "")
                params = entry.get("params", {})
                stages = entry.get("stages", [])
                logger.info(f"   • Step: {step}")
                logger.info(f"     Info: {info}")
                if stages:
                    logger.info(f"     Stages: {', '.join(stages)}")
                if params:
                    logger.info(f"     Params: {params}")
        # Zusätzlicher Hinweis bei Modellproblemen
        if result.get("warnings"):
            logger.warning("Modell-/Plugin-Warnungen:")
            for w in result["warnings"]:
                logger.warning(f"   - {w}")
    return result


def print_usage():
    print("\nOptionen:")
    print("  -q, --quiet     Keine Fortschritts-Ausgaben")
    print("  -h, --help      Diese Hilfe anzeigen")
    print()


def main():
    args = sys.argv[1:]
    verbose = True
    if "-q" in args or "--quiet" in args:
        verbose = False
        args = [a for a in args if a not in ["-q", "--quiet"]]

    if "-h" in args or "--help" in args:
        print_usage()
        sys.exit(0)

    input_file = None
    output_file = None
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--input_audio", "--input"):
            if i + 1 < len(args):
                input_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--input_audio", "--input"):
            input_file = arg.split("=", 1)[1]
        elif arg in ("--output_audio", "--output"):
            if i + 1 < len(args):
                output_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--output_audio", "--output"):
            output_file = arg.split("=", 1)[1]
        elif arg in ("--mode",):
            skip_next = True  # Verbrauche naechsten Wert, Modus wird ignoriert

    # Positional Fallback: nur Nicht-Flag-Argumente verwenden
    positional = [a for a in args if not a.startswith("-")]
    if input_file is None and len(positional) >= 1:
        input_file = positional[0]
    if output_file is None and len(positional) >= 2:
        output_file = positional[1]

    if not input_file or not output_file:
        print("❌ Fehler: Zu wenig oder ungültige Argumente\n")
        print_usage()
        sys.exit(1)

    process_audio(input_file, output_file, verbose=verbose)


if __name__ == "__main__":
    main()
