import io
import logging
import math

import numpy as np
import soundfile as sf

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

logger = logging.getLogger(__name__)

# True-Peak threshold: > -0.5 dBTP triggers a warning, > 0 dBTP triggers clipping guard
_TRUE_PEAK_WARN_DBTP = -0.5
_TRUE_PEAK_LIMIT = 1.0


def _export_guard(audio: np.ndarray) -> np.ndarray:
    """
    Numerische Robustheitspr\xfcfung unmittelbar vor dem Schreiben.

    Entfernt NaN/Inf, clampt auf [-1.0, 1.0] und warnt bei True-Peak-\xdcberschreitung.
    Entspricht der Pflicht-Invariante aus den Copilot-Instructions (§ Numerische Robustheit).
    """
    # 1. NaN / Inf bereinigen
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. True-Peak pr\xfcfen (Spitzenwert vor Hard-Clip)
    peak = float(np.max(np.abs(audio)))
    if peak > _TRUE_PEAK_LIMIT:
        logger.warning(
            "Export-Guard: True-Peak %.4f dBTP \xfcberschreitet 0 dBTP — wird begrenzt.",
            20 * math.log10(peak) if peak > 0 else -math.inf,
        )
    elif peak > 10 ** (_TRUE_PEAK_WARN_DBTP / 20):
        logger.warning(
            "Export-Guard: True-Peak %.2f dBFS n\xe4hert sich 0 dBTP.",
            20 * math.log10(peak) if peak > 0 else -math.inf,
        )

    # 3. Hard-Clip auf [-1.0, 1.0] (Pflicht-Invariante)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def export_audio(audio_bytes, export_path, format="wav"):
    # Audio-Bytes in numpy-Array laden
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Fehler beim Lesen der Audiodaten: {e}")

    # Export Guard: NaN/Inf-Bereinigung + True-Peak-Schutz
    audio = _export_guard(audio)

    # WAV, FLAC, OGG, AIFF direkt mit soundfile
    if format.lower() in ["wav", "flac", "ogg", "aiff", "aif", "alac", "caf"]:
        try:
            sf.write(export_path, audio, sr, format=format.upper())
            return True
        except Exception as e:
            raise RuntimeError(f"Fehler beim Export als {format}: {e}")
    # MP3, AAC, M4A, OPUS nur mit ffmpeg
    elif format.lower() in ["mp3", "aac", "m4a", "opus"]:
        if ffmpeg is None:
            raise RuntimeError("ffmpeg-python nicht installiert. Export nicht möglich.")
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Export Guard already applied above; write guarded audio to temp WAV
            sf.write(tmp.name, audio, sr, format="WAV")
            try:
                out_args = {"format": format.lower()}
                if format.lower() == "mp3":
                    out_args["audio_bitrate"] = "320k"
                (ffmpeg.input(tmp.name).output(export_path, **out_args).run(overwrite_output=True, quiet=True))
                return True
            except Exception as e:
                raise RuntimeError(f"Fehler beim {format.upper()}-Export: {e}")
    else:
        raise ValueError(f"Nicht unterstütztes Exportformat: {format}")
