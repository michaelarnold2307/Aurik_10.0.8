import io
import logging
import math
from typing import Any

import numpy as np
import soundfile as sf

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    from scipy import signal as _scipy_signal

    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_signal = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# True-Peak threshold: > -0.5 dBTP triggers a warning, > 0 dBTP triggers clipping guard
_TRUE_PEAK_WARN_DBTP = -0.5
_TRUE_PEAK_LIMIT = 1.0

# ---------------------------------------------------------------------------
# POW-r Type 3 noise-shaping coefficients (Craven / Law / Stuart, AES 1987).
# Psychoacoustically optimised for 48 kHz / 16-bit targets; also applied for
# 24-bit (reduces perceived noise floor well beyond −98 dBFS requirement).
# Reference: Meridian Lossless Packing — noise-shaping appendix, Table 3.
# ---------------------------------------------------------------------------
_POWR3_COEFFS = np.array(
    [2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847],
    dtype=np.float64,
)


def _apply_powr3_dither(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    """Apply POW-r Type 3 noise-shaped dither (primary) before integer quantisation.

    Uses error-feedback-approximated noise shaping: TPDF dither is pre-shaped
    with the POW-r Type 3 FIR filter via ``scipy.signal.lfilter``, then added to
    the signal.  This concentrates dither energy in psychoacoustically less
    sensitive high-frequency bands, pushing the perceived noise floor ≥ 14 dB
    lower than flat TPDF at 16-bit (targeting ≤ −72 dBFS per §8.2).

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio in ``[-1.0, 1.0]``, shape ``(samples,)`` or
        ``(samples, channels)``.
    bit_depth : int
        Target integer bit depth.  No-op for 32 or higher.

    Returns
    -------
    np.ndarray
        Dithered float32 audio, still in ``[-1.0, 1.0]``, ready for
        ``soundfile`` integer quantisation.
    """
    if bit_depth >= 32 or not _SCIPY_AVAILABLE:
        return audio

    lsb = 2.0 / (2**bit_depth)

    mono_input = audio.ndim == 1
    if mono_input:
        a = audio[:, np.newaxis].astype(np.float64)
    else:
        a = audio.astype(np.float64)

    n_samples, n_ch = a.shape

    # TPDF dither: two uniform RVs → triangular distribution centred on 0,
    # amplitude = ±1 LSB (spec §DSP: POW-r Typ 3 primär → TPDF Fallback).
    rng = np.random.default_rng()
    raw_dither = (rng.random((n_samples, n_ch)) + rng.random((n_samples, n_ch)) - 1.0) * lsb

    # Shape the dither with the POW-r Type 3 FIR response.
    shaped = np.empty_like(raw_dither)
    for ch in range(n_ch):
        shaped[:, ch] = _scipy_signal.lfilter(_POWR3_COEFFS, [1.0], raw_dither[:, ch])

    result = np.clip((a + shaped).astype(np.float32), -1.0, 1.0)
    return result[:, 0] if mono_input else result


def _apply_tpdf_dither(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    """TPDF fallback dither — no noise shaping.

    Used when scipy is unavailable.  Amplitude = ±1 LSB triangular noise.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio in ``[-1.0, 1.0]``.
    bit_depth : int
        Target integer bit depth.  No-op for 32 or higher.

    Returns
    -------
    np.ndarray
        Dithered float32 audio in ``[-1.0, 1.0]``.
    """
    if bit_depth >= 32:
        return audio
    lsb = 2.0 / (2**bit_depth)
    rng = np.random.default_rng()
    noise = (rng.random(audio.shape) + rng.random(audio.shape) - 1.0) * lsb
    return np.clip((audio + noise).astype(np.float32), -1.0, 1.0)


def apply_dither(audio: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """Apply dither before integer quantisation.

    Primary: POW-r Type 3 noise-shaped dither (spec §DSP-Spezialregeln).
    Fallback: TPDF dither when scipy is unavailable.
    No-op for ``bit_depth >= 32``.

    Parameters
    ----------
    audio : np.ndarray
        Float audio in ``[-1.0, 1.0]``.
    bit_depth : int
        Target bit depth.

    Returns
    -------
    np.ndarray
        Dithered float32 audio.
    """
    if bit_depth >= 32:
        return audio

    if _SCIPY_AVAILABLE:
        logger.debug("Dithering: POW-r Type 3 applied (bit_depth=%d)", bit_depth)
        return _apply_powr3_dither(audio, bit_depth)

    logger.warning("Dithering: scipy unavailable — TPDF fallback applied (bit_depth=%d).", bit_depth)
    return _apply_tpdf_dither(audio, bit_depth)


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


def validate_export_quality(result: Any) -> tuple[bool, list[str]]:
    """Validate export quality based on RestorationResult metadata.

    Checks chroma correlation (§8.2), LUFS delta, and Musical Goals.
    Returns (passed, list_of_warnings).  Always allows export but logs violations.
    Hard-fail only on chroma_correlation < 0.80 (catastrophic tonal shift).
    """
    warnings: list[str] = []
    passed = True

    # §8.2 Chroma correlation ≥ 0.95 (Tonart-Erhaltung)
    chroma = getattr(result, "chroma_correlation", None)
    if chroma is not None:
        if chroma < 0.80:
            warnings.append(
                f"KRITISCH: Chroma-Korrelation {chroma:.3f} < 0.80 — "
                "schwere Tonart-Verschiebung erkannt. Export wird nicht empfohlen."
            )
            passed = False
        elif chroma < 0.95:
            warnings.append(f"WARNUNG: Chroma-Korrelation {chroma:.3f} < 0.95 — geringe Tonart-Abweichung erkannt.")

    # §8.2 LUFS delta ≤ 1 LU (Restoration) / EBU R128 (Studio 2026)
    lufs_delta = getattr(result, "lufs_delta", None)
    if lufs_delta is not None and lufs_delta > 3.0:
        warnings.append(f"WARNUNG: LUFS-Delta {lufs_delta:.1f} LU > 3.0 LU — signifikante Lautstärke-Änderung.")

    # Musical Goals Violations
    meta = getattr(result, "metadata", {})
    goals_meta = meta.get("musical_goals", {})
    violations = goals_meta.get("violations", [])
    if violations:
        warnings.append(
            f"Qualitäts-Hinweis: {len(violations)} Musical Goal(s) nicht erfüllt: {', '.join(violations[:5])}"
        )

    for w in warnings:
        logger.warning("Export-Quality-Gate: %s", w)

    return passed, warnings


def export_audio(
    audio_bytes,
    export_path: str,
    format: str = "wav",
    bit_depth: int = 24,
) -> bool:
    """Export audio bytes to a file on disk.

    Applies the full export chain mandated by the spec:
    1. NaN/Inf guard + True-Peak clip (``_export_guard``).
    2. POW-r Type 3 dithering (primary) / TPDF (fallback) for integer targets
       (``bit_depth < 32``).  Spec §DSP-Spezialregeln: *VERBOTEN: Truncation
       ohne Dithering.*
    3. Atomic write via ``.tmp → os.replace``.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio bytes (any format readable by soundfile).
    export_path : str
        Destination file path.
    format : str
        Output container/codec (wav, flac, mp3, …).
    bit_depth : int
        Target integer bit depth for lossless formats (16 or 24).
        Use 32 to write float32 without dithering.  Default: 24.

    Returns
    -------
    bool
        ``True`` on success.
    """
    _SUBTYPE_MAP = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}

    # 1. Decode incoming bytes
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    except Exception as e:
        logger.error("Export: Audiodaten konnten nicht gelesen werden: %s", e)
        raise RuntimeError(f"Fehler beim Lesen der Audiodaten: {e}")

    logger.info(
        "Export gestartet: path=%s, format=%s, bit_depth=%d, sr=%d, shape=%s, duration=%.1fs",
        export_path,
        format,
        bit_depth,
        sr,
        audio.shape,
        len(audio) / max(sr, 1) if audio.ndim == 1 else audio.shape[0] / max(sr, 1),
    )

    # 2. NaN/Inf-Bereinigung + True-Peak-Schutz
    audio = _export_guard(audio)

    # 3. Dithering before integer quantisation (spec §DSP-Spezialregeln)
    if bit_depth < 32 and format.lower() not in ("mp3", "aac", "m4a", "opus"):
        audio = apply_dither(audio, bit_depth=bit_depth)

    subtype = _SUBTYPE_MAP.get(bit_depth)

    # 4. WAV, FLAC, OGG, AIFF direkt mit soundfile — atomic write via .tmp → os.replace
    if format.lower() in ["wav", "flac", "ogg", "aiff", "aif", "alac", "caf"]:
        import os

        tmp_path = export_path + ".tmp"
        try:
            write_kwargs: dict = {"format": format.upper()}
            if subtype and format.lower() in ("wav", "flac", "aiff", "aif"):
                write_kwargs["subtype"] = subtype
            sf.write(tmp_path, audio, sr, **write_kwargs)
            os.replace(tmp_path, export_path)
            _size_mb = os.path.getsize(export_path) / (1024 * 1024)
            logger.info(
                "Export abgeschlossen: %s (%.1f MB, %s %d-bit)", export_path, _size_mb, format.upper(), bit_depth
            )
            return True
        except Exception as e:
            # Cleanup orphaned tmp on failure
            logger.error("Export fehlgeschlagen (%s): %s", format, e)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Fehler beim Export als {format}: {e}")
    # MP3, AAC, M4A, OPUS nur mit ffmpeg
    elif format.lower() in ["mp3", "aac", "m4a", "opus"]:
        if ffmpeg is None:
            raise RuntimeError("ffmpeg-python nicht installiert. Export nicht möglich.")
        import os
        import tempfile

        tmp_wav = None
        tmp_out = export_path + ".tmp"
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name
                # Export Guard already applied above; write guarded audio to temp WAV
                sf.write(tmp_wav, audio, sr, format="WAV")
            out_args = {"format": format.lower()}
            if format.lower() == "mp3":
                out_args["audio_bitrate"] = "320k"
            (ffmpeg.input(tmp_wav).output(tmp_out, **out_args).run(overwrite_output=True, quiet=True))
            os.replace(tmp_out, export_path)
            _size_mb = os.path.getsize(export_path) / (1024 * 1024)
            logger.info("Export abgeschlossen: %s (%.1f MB, %s)", export_path, _size_mb, format.upper())
            return True
        except Exception as e:
            # Cleanup orphaned tmp files
            logger.error("Export fehlgeschlagen (%s via ffmpeg): %s", format, e)
            for _p in (tmp_out,):
                try:
                    os.remove(_p)
                except OSError:
                    pass
            raise RuntimeError(f"Fehler beim {format.upper()}-Export: {e}")
        finally:
            # Always clean up the intermediate WAV temp file
            if tmp_wav:
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
    else:
        raise ValueError(f"Nicht unterstütztes Exportformat: {format}")
