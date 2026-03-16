"""
update_yaml_with_dsp_modules.py

Dieses Skript liest die sota_dsp_module_list.yaml ein, durchsucht das DSP-Modulverzeichnis nach Python-Klassen,
und ergänzt für jeden Eintrag die Felder modul_file und modul_class, sofern ein passendes Modul gefunden wird.

Voraussetzung: pyyaml ist installiert (pip install pyyaml)
"""

import logging
import os
import re

import yaml
logger = logging.getLogger(__name__)

YAML_PATH = "tests/sota/sota_dsp_module_list.yaml"
DSP_PATH = "Aurik_Standalone/dsp/"

# Lade YAML
with open(YAML_PATH, encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Erzeuge Mapping: Name -> (Datei, Klasse)
module_map = {}
for fname in os.listdir(DSP_PATH):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(DSP_PATH, fname)
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    # Suche nach Klassen
    for match in re.finditer(r"class (\w+)", content):
        cls = match.group(1)
        # Mapping nach Namensähnlichkeit (vereinfachte Heuristik)
        key = (
            cls.replace("_", " ")
            .replace("AI", "Ai")
            .replace("SOTA", "Sota")
            .replace("Adaptive", "Adaptive ")
            .replace("Remover", " Remover")
            .replace("Enhancer", " Enhancer")
            .replace("Separator", " Separator")
            .replace("Exciter", " Exciter")
            .replace("Limiter", " Limiter")
            .replace("Compressor", " Compressor")
            .replace("Gate", " Gate")
            .replace("Shaper", " Shaper")
            .replace("Preservation", " Preservation")
            .replace("Detection", " Detection")
            .replace("Correction", " Correction")
            .replace("Matrix", " Matrix")
            .replace("Widener", " Widener")
            .replace("Expander", " Expander")
            .replace("Declipper", " Declipper")
            .replace("Denoiser", " Denoiser")
            .replace("Equalizer", " Equalizer")
            .replace("Analyzer", " Analyzer")
            .replace("Evaluator", " Evaluator")
            .replace("Estimator", " Estimator")
            .replace("Profile", " Profile")
            .replace("Filter", " Filter")
            .replace("Synthesizer", " Synthesizer")
            .replace("Regenerator", " Regenerator")
            .replace("Filler", " Filler")
            .replace("Inpainting", " Inpainting")
            .replace("SuperRes", " SuperRes")
            .replace("SuperResolution", " SuperResolution")
            .replace("Normalizer", " Normalizer")
            .replace("Normalizer", " Normalizer")
            .replace("Balancer", " Balancer")
            .replace("Ducker", " Ducker")
            .replace("Panner", " Panner")
            .replace("Analyzer", " Analyzer")
            .replace("Remover", " Remover")
            .replace("Artifact", " Artifact")
            .replace("Artifact", " Artifact")
            .replace("TruePeak", " TruePeak")
            .replace("Oversampler", " Oversampler")
            .replace("AGC", " AGC")
            .replace("Leveler", " Leveler")
            .replace("Formant", " Formant")
            .replace("Preserver", " Preserver")
            .replace("Isolator", " Isolator")
            .replace("Confidence", " Confidence")
            .replace("Weighting", " Weighting")
            .replace("Hole", " Hole")
            .replace("Segment", " Segment")
            .replace("Genre", " Genre")
            .replace("Key", " Key")
            .replace("Beat", " Beat")
            .replace("Onset", " Onset")
            .replace("Sibilance", " Sibilance")
            .replace("Breath", " Breath")
            .replace("Dropout", " Dropout")
            .replace("Crackle", " Crackle")
            .replace("Bias", " Bias")
            .replace("NAB", " NAB")
            .replace("RIAA", " RIAA")
            .replace("PrintThrough", " PrintThrough")
            .replace("Azimuth", " Azimuth")
            .replace("Bark", " Bark")
            .replace("ISO", " ISO")
            .replace("BS", " BS")
            .replace("Loudness", " Loudness")
            .replace("Range", " Range")
            .replace("Fade", " Fade")
            .replace("SampleRate", " Sample Rate")
            .replace("Dithering", " Dithering")
            .replace("NoiseShaping", " Noise Shaping")
            .replace("Intersample", " Intersample")
            .replace("TruePeak", " True Peak")
            .replace("LRA", " LRA")
            .replace("AGC", " AGC")
            .replace("Leveler", " Leveler")
            .replace("Multiband", " Multiband")
            .replace("MCRA", " MCRA")
            .replace("IMCRA", " IMCRA")
            .replace("MMSE", " MMSE")
            .replace("OMLSA", " OMLSA")
            .replace("PSOLA", " PSOLA")
            .replace("WSOLA", " WSOLA")
            .replace("CQT", " CQT")
            .replace("STFT", " STFT")
            .replace("ISTFT", " ISTFT")
            .replace("RMS", " RMS")
            .replace("SNR", " SNR")
            .replace("LSD", " LSD")
            .replace("MOS", " MOS")
            .replace("SDR", " SDR")
            .replace("SI", " SI")
            .replace("DNSMOS", " DNSMOS")
            .replace("NISQA", " NISQA")
            .replace("POLQA", " POLQA")
            .replace("PESQ", " PESQ")
            .replace("ViSQOL", " ViSQOL")
            .replace("STOI", " STOI")
            .replace("LRA", " LRA")
            .replace("Loudness", " Loudness")
            .replace("Range", " Range")
            .replace("Fade", " Fade")
            .replace("SampleRate", " Sample Rate")
            .replace("Dithering", " Dithering")
            .replace("NoiseShaping", " Noise Shaping")
            .replace("Intersample", " Intersample")
            .replace("TruePeak", " True Peak")
            .replace("LRA", " LRA")
            .replace("AGC", " AGC")
            .replace("Leveler", " Leveler")
            .replace("Multiband", " Multiband")
            .replace("MCRA", " MCRA")
            .replace("IMCRA", " IMCRA")
            .replace("MMSE", " MMSE")
            .replace("OMLSA", " OMLSA")
            .replace("PSOLA", " PSOLA")
            .replace("WSOLA", " WSOLA")
            .replace("CQT", " CQT")
            .replace("STFT", " STFT")
            .replace("ISTFT", " ISTFT")
            .replace("RMS", " RMS")
            .replace("SNR", " SNR")
            .replace("LSD", " LSD")
            .replace("MOS", " MOS")
            .replace("SDR", " SDR")
            .replace("SI", " SI")
            .replace("DNSMOS", " DNSMOS")
            .replace("NISQA", " NISQA")
            .replace("POLQA", " POLQA")
            .replace("PESQ", " PESQ")
            .replace("ViSQOL", " ViSQOL")
            .replace("STOI", " STOI")
        )
        module_map[key.lower()] = (fname, cls)

# Ergänze YAML-Einträge
for entry in data["dsp_modules"]:
    key = entry["name"].lower()
    if key in module_map:
        entry["modul_file"] = module_map[key][0]
        entry["modul_class"] = module_map[key][1]
    else:
        entry["modul_file"] = None
        entry["modul_class"] = None

# Schreibe YAML zurück
with open(YAML_PATH, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True, sort_keys=False)

logger.info("YAML-Liste wurde mit modul_file und modul_class ergänzt.")
