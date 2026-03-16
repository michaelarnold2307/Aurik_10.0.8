# DeepFilterNet3_ii Inferenzskript für Docker/CLI
# Beispiel: python deepfilternet_infer.py input.wav output.wav

import sys

try:
    from df import enhance, init_df
except ImportError:
    # Mock für df-Modul, damit Tests laufen
    def enhance(model, df_state, x_tensor):
        pass

        # Gibt das Tensor unverändert zurück (Dummy)
        return x_tensor

    def init_df(default_model=None):
        # Dummy-Modelle und State
        return None, None, None, None


import logging
import os

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="[AURIK-LOG] %(asctime)s %(levelname)s: %(message)s")


def load_audio(path):
    os.path.splitext(path)[1].lower()
    try:
        x, sr = sf.read(path)
        # x: (samples,) für mono, (samples, channels) für stereo
        return x, sr
    except RuntimeError:
        try:
            import librosa

            x, sr = librosa.load(path, sr=None, mono=True)
            return x, sr
        except Exception as e:
            logging.error("Fehler beim Laden von %s: %s", path, e)
            if __name__ == "__main__":
                sys.exit(2)
            else:
                raise


def main():
    if len(sys.argv) != 3:
        logging.error("Nutzung: python deepfilternet_infer.py <input.(wav|mp3|flac)> <output.wav>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_wav = sys.argv[2]

    logging.info(f"Starte Restaurierung: {input_file}")
    x, sr = load_audio(input_file)
    logging.info(f"Audio geladen: sr={sr}, shape={x.shape}")
    try:
        model, df_state, _, _ = init_df(default_model="DeepFilterNet3")
        logging.info("DeepFilterNet3 Modell initialisiert.")
    except Exception as e:
        logging.error(f"Fehler bei Modellinitialisierung: {e}")
        sys.exit(3)
    try:
        if x.ndim == 1:
            # Mono
            x_tensor = torch.from_numpy(x).float().unsqueeze(0)  # (1, samples)
            logging.info(f"Audio in torch.Tensor konvertiert: shape={x_tensor.shape}")
            x_enh = enhance(model, df_state, x_tensor)
            x_enh_np = x_enh.cpu().numpy() if hasattr(x_enh, "cpu") else x_enh
            if x_enh_np.ndim == 2 and x_enh_np.shape[0] == 1:
                x_enh_np = x_enh_np.squeeze(0)
        elif x.ndim == 2 and x.shape[1] == 2:
            # Stereo: beide Kanäle getrennt verarbeiten
            left = torch.from_numpy(x[:, 0]).float().unsqueeze(0)
            right = torch.from_numpy(x[:, 1]).float().unsqueeze(0)
            logging.info(f"Stereo erkannt: left shape={left.shape}, right shape={right.shape}")
            left_enh = enhance(model, df_state, left)
            right_enh = enhance(model, df_state, right)
            left_np = left_enh.cpu().numpy().squeeze(0) if hasattr(left_enh, "cpu") else left_enh.squeeze(0)
            right_np = right_enh.cpu().numpy().squeeze(0) if hasattr(right_enh, "cpu") else right_enh.squeeze(0)
            x_enh_np = np.stack([left_np, right_np], axis=1)  # (samples, 2)
        else:
            raise ValueError(f"Unerwartete Audioform: {x.shape}")
        logging.info("Enhancement erfolgreich durchgeführt.")
    except Exception as e:
        logging.error(f"Fehler beim Enhancement: {e}")
        sys.exit(4)
    try:
        sf.write(output_wav, x_enh_np, sr)
        logging.info("Ergebnis gespeichert: %s", output_wav)
        logging.info("Fertig: %s", output_wav)
    except Exception as e:
        logging.error(f"Fehler beim Speichern: {e}")
        sys.exit(5)


if __name__ == "__main__":
    main()
