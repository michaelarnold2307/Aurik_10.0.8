# Minimal-CLI für Wave-U-Net-Inferenz im Docker-Container (angepasst für aktuelle Wave-U-Net-Struktur)
import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.append("/workspace/Wave-U-Net")
import Config

try:
    from Evaluate import predict as _evaluate_predict

    class _Evaluate:
        @staticmethod
        def predict(*args, **kwargs):
            return _evaluate_predict(*args, **kwargs)

    Evaluate = _Evaluate
except ImportError:

    class Evaluate:  # type: ignore[no-redef]
        @staticmethod
        def predict(track, model_config, load_model_fn):  # noqa: F811
            raise RuntimeError("Wave-U-Net Evaluate nicht verfügbar")


def main():
    parser = argparse.ArgumentParser(description="Wave-U-Net Inferenz (angepasst)")
    parser.add_argument("--input", required=True, help="Eingabedatei (WAV)")
    parser.add_argument("--output", required=True, help="Ausgabedatei (WAV)")
    parser.add_argument("--model", default=None, help="Optional: Modellpfad (Checkpoint)")
    args = parser.parse_args()

    # Modell-Konfiguration laden
    model_config = Config.cfg()["model_config"]
    if args.model:
        model_path = args.model
    else:
        # Default: wie in Predict.py
        model_path = os.path.join("/workspace/Wave-U-Net/checkpoints", "full_44KHz", "full_44KHz-236118")

    # Audio laden
    audio, sr = librosa.load(args.input, sr=None, mono=False)

    # Für Evaluate.predict wird ein Track-Objekt mit .audio und .rate benötigt
    class SimpleTrack:
        pass

    track = SimpleTrack()
    # librosa gibt (channels, samples), wir brauchen (samples, channels)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    track.audio = audio.T  # (samples, channels)
    track.rate = sr

    # Inferenz
    # Achtung: Evaluate.predict erwartet weitere Parameter, ggf. anpassen
    # Hier: Dummy für load_model, results_dir nicht genutzt
    def dummy_load_model():
        return model_path

    sources = Evaluate.predict(track, model_config, dummy_load_model)

    # Speichern: Wir nehmen die erste Quelle (z.B. vocals) als Beispiel
    # (samples, channels) -> (channels, samples) für soundfile
    if isinstance(sources, dict):
        # Nimm die erste Quelle im Dict
        first_key = list(sources.keys())[0]
        enhanced = sources[first_key]
    else:
        enhanced = sources
    sf.write(args.output, enhanced, sr)


if __name__ == "__main__":
    main()
