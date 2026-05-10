import logging
import os
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Formats that libsndfile cannot decode — route directly to pedalboard (FFmpeg backend)
# to avoid a guaranteed exception + log noise on every import of these files.
_SF_UNSUPPORTED_EXT: frozenset[str] = frozenset(
    {
        ".mp3",
        ".mp2",
        ".mp1",
        ".m4a",
        ".m4b",
        ".m4p",
        ".aac",
        ".wma",
        ".asf",
        ".opus",
        ".webm",
        ".amr",
        ".3gp",
        ".3g2",
        ".ac3",
        ".dts",
    }
)


def _estimate_interchannel_lag_samples(audio: np.ndarray, sr: int, max_seconds: float = 5.0) -> int:
    """Estimate L/R lag (samples) using GCC-PHAT on a bounded window.

    Returns 0 for non-stereo inputs or on analysis failure.
    """
    try:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim != 2:
            return 0
        if arr.shape[1] == 2 and arr.shape[0] > 2:
            l = arr[:, 0]
            r = arr[:, 1]
        elif arr.shape[0] == 2 and arr.shape[1] > 2:
            l = arr[0]
            r = arr[1]
        else:
            return 0

        n = min(len(l), len(r), int(float(sr) * max_seconds))
        if n < max(1024, sr // 10):
            return 0

        x = l[:n].astype(np.float64)
        y = r[:n].astype(np.float64)
        n_fft = 1
        while n_fft < 2 * n:
            n_fft <<= 1

        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)
        cross = X * np.conj(Y)
        gcc = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n_fft)

        max_delay = min(int(sr * 0.2), n - 1)  # ±200 ms
        if max_delay <= 0:
            return 0
        search = np.concatenate([gcc[n_fft - max_delay :], gcc[: max_delay + 1]])
        return int(np.argmax(np.abs(search))) - max_delay
    except Exception:
        return 0


def _lazy_get_carrier_tools():
    from .carrier_forensics import analyze_carrier_forensics
    from .carrier_ml_classifier import classify_carrier_ml

    return analyze_carrier_forensics, classify_carrier_ml


def detect_carrier(filepath: str, meta: dict[str, Any] | None = None) -> str:
    """
    Versucht, den Tonträger (Schallplatte, Kassette, CD, Band, etc.) anhand von Dateiname, Metadaten oder User-Tag zu erkennen.
    """
    name = os.path.basename(filepath).lower()
    meta = meta or {}
    # Heuristik: Dateiname
    if any(
        x in name
        for x in [
            "vinyl",
            "lp",
            "platte",
            "schallplatte",
            "33rpm",
            "45rpm",
            "78rpm",
            "shellac",
            "schellack",
        ]
    ):
        return "Schallplatte (Vinyl/Schellack)"
    if any(x in name for x in ["cassette", "mc", "kassette", "tape"]):
        return "Kassette/Band"
    if any(x in name for x in ["cd", "compactdisc", "compact_disc"]):
        return "CD"
    if any(x in name for x in ["minidisc", "md"]):
        return "MiniDisc"
    if any(x in name for x in ["dat", "digitalaudiotape"]):
        return "DAT"
    if any(x in name for x in ["reel", "tonband", "spule"]):
        return "Tonband/Reel"
    # Heuristik: Metadaten
    for k, v in meta.items() if meta else []:
        vstr = str(v).lower()
        if "vinyl" in vstr or "lp" in vstr or "platte" in vstr or "schellack" in vstr:
            return "Schallplatte (Vinyl/Schellack)"
        if "cassette" in vstr or "mc" in vstr or "kassette" in vstr or "tape" in vstr:
            return "Kassette/Band"
        if "cd" in vstr:
            return "CD"
        if "minidisc" in vstr or "md" in vstr:
            return "MiniDisc"
        if "dat" in vstr:
            return "DAT"
        if "reel" in vstr or "tonband" in vstr or "spule" in vstr:
            return "Tonband/Reel"
    return "Unbekannt"


def is_supported_audio_file(filename: str) -> bool:
    """Prüft ob Datei ein unterstütztes Audioformat ist."""
    # Unterstützte Formate: WAV, MP3, FLAC, OGG, AAC, AIFF (erweiterbar)
    return filename.lower().endswith(
        (
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".aac",
            ".aiff",
            ".aif",
            ".wma",
            ".opus",
            ".m4a",
            ".alac",
            ".caf",
        )
    )


def load_audio_file(
    filepath: str,
    target_sr: int | None = None,
    mono: bool = False,
    do_carrier_analysis: bool = True,
) -> dict[str, Any] | None:
    """Lädt Audiodatei robust und gibt Dict mit allen Metadaten zurück.

    Args:
        filepath: Pfad zur Audiodatei
        target_sr: Ziel-Sample-Rate (optional, Resampling wenn nötig)
        mono: Wenn True, konvertiere zu Mono

    Returns:
        Dict mit 'audio', 'sr', 'channels', 'format', 'duration', 'meta', 'error'
    """
    result: dict[str, Any] = {
        "audio": None,
        "sr": None,
        "channels": None,
        "format": None,
        "duration": None,
        "meta": {},
        "carrier": None,
        "error": None,
    }
    try:
        if not os.path.isfile(filepath):
            result["error"] = "File not found"
            return result
        if not is_supported_audio_file(filepath):
            result["error"] = "Unsupported file type"
            return result

        _ext = os.path.splitext(filepath)[1].lower()
        _sf_unsupported = _ext in _SF_UNSUPPORTED_EXT

        # Opportunistic capability probe: some deployments support MP3/AAC via
        # libsndfile plugins even when extension-only heuristics mark them unsupported.
        # If SoundFile can read header metadata, prefer the SoundFile decode path
        # because it preserves inter-channel alignment more reliably than FFmpeg fallbacks.
        if _sf_unsupported:
            try:
                sf.info(filepath)
                _sf_unsupported = False
                logger.debug("load_audio_file: soundfile capability probe passed for %s", _ext)
            except Exception:
                pass

        # ── Metadata ─────────────────────────────────────────────────────────
        # soundfile.info() is fast and reliable for lossless formats.
        # For MP3/AAC/WMA etc. it always fails — skip it and use extension only.
        if not _sf_unsupported:
            try:
                _info = sf.info(filepath)
                result["format"] = _info.format
                result["channels"] = _info.channels
                result["sr"] = _info.samplerate
                result["duration"] = _info.duration
                result["meta"] = dict(getattr(_info, "extra_info", {}) or {})
            except Exception:
                result["format"] = _ext[1:].upper()
        else:
            result["format"] = _ext[1:].upper()

        # ── Audio decode ──────────────────────────────────────────────────────
        # Route: soundfile (lossless) → pedalboard/FFmpeg (lossy + universal fallback)
        audio: np.ndarray | None = None
        sr: int = 0

        if not _sf_unsupported:
            # Stufe 1: soundfile — WAV, FLAC, OGG, AIFF, ALAC, CAF …
            try:
                audio, sr = sf.read(filepath, always_2d=False)
                logger.debug("load_audio_file: soundfile OK (%s)", filepath)
            except Exception as _e1:
                logger.debug("load_audio_file: soundfile failed (%s) — trying pedalboard", _e1)

        if audio is None and _sf_unsupported:
            # Stufe 2 (lossy primary): pedalboard/FFmpeg — stabiler als pydub für in-process.
            # pydub.from_file() kann via libavcodec SIGABRT auslösen (auch ohne malloc_trim),
            # weil FFmpeg-Allokierungsfehler unter Speicherdruck als abort() propagieren.
            # Pedalboard nutzt denselben FFmpeg-Stack, aber ohne audioop C-Extension.
            try:
                from pedalboard.io import AudioFile as _PBAudioFile  # type: ignore

                with _PBAudioFile(filepath) as _f:  # pylint: disable=not-context-manager
                    sr = int(_f.samplerate)
                    _frames = _f.frames
                    _chunk = sr * 300  # 300 s chunks to avoid OOM on long files
                    _parts: list[np.ndarray] = []
                    _read = 0
                    while _read < _frames:
                        _block = _f.read(min(_chunk, _frames - _read))
                        if _block.shape[-1] == 0:
                            break  # EOF — pedalboard.frames can overcount for VBR
                        _parts.append(_block)
                        _read += _block.shape[-1]
                _raw = np.concatenate(_parts, axis=1) if len(_parts) > 1 else _parts[0]
                # pedalboard returns (channels, samples) — transpose to (samples, channels)
                if _raw.ndim == 2:
                    _raw = _raw.T
                audio = _raw.astype(np.float32)
                logger.debug("load_audio_file: pedalboard/FFmpeg OK for lossy (%s)", filepath)
            except Exception as _e_pb_lossy:
                logger.debug(
                    "load_audio_file: pedalboard failed for lossy (%s) — pydub subprocess fallback", _e_pb_lossy
                )

        if audio is None and _sf_unsupported:
            # Stufe 3 (lossy fallback): pydub — in subprocess, um in-process SIGABRT zu vermeiden.
            # Subprocess-Isolation: SIGABRT des Child-Prozesses wird als Exception behandelt.
            try:
                import json
                import subprocess
                import sys
                import tempfile

                _script = (
                    "import sys, json, numpy as np\n"
                    "from pydub import AudioSegment\n"
                    f"seg = AudioSegment.from_file({filepath!r})\n"
                    "sr = seg.frame_rate\n"
                    "bw = seg.sample_width\n"
                    "raw = bytes(seg.raw_data)\n"
                    "dt = np.uint8 if bw==1 else (np.int16 if bw==2 else np.int32)\n"
                    "s = np.frombuffer(raw, dtype=dt).astype(np.float32)\n"
                    "if bw == 1:\n"
                    "    s = (s - 128.0) / 128.0\n"
                    "else:\n"
                    "    s /= float(2**(bw*8-1))\n"
                    "if seg.channels > 1:\n"
                    "    s = s.reshape(-1, seg.channels)\n"
                    "result = {'sr': sr, 'channels': seg.channels, 'shape': list(s.shape)}\n"
                    "with open(sys.argv[1], 'wb') as f:\n"
                    "    np.save(f, s)\n"
                    "print(json.dumps(result))\n"
                )
                _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
                _tmp.close()
                _r = subprocess.run(
                    [sys.executable, "-c", _script, _tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if _r.returncode == 0 and _r.stdout.strip():
                    _d = json.loads(_r.stdout.strip())
                    audio = np.load(_tmp.name).astype(np.float32)
                    sr = int(_d["sr"])
                    logger.debug("load_audio_file: pydub subprocess OK (%s)", filepath)
                else:
                    raise RuntimeError(_r.stderr[:500] or f"returncode={_r.returncode}")
            except Exception as _e2:
                result["error"] = f"Audio read error: pedalboard + pydub subprocess failed. Last: {_e2}"
                return result
            finally:
                try:
                    import os as _os_tmp

                    _os_tmp.unlink(_tmp.name)
                except Exception:
                    pass

        if audio is None and not _sf_unsupported:
            # Stufe 2/3: pedalboard (FFmpeg backend) — preferred for lossless fallback,
            # for lossless fallback only.
            try:
                from pedalboard.io import AudioFile as _PBAudioFile  # type: ignore

                with _PBAudioFile(filepath) as _f:  # pylint: disable=not-context-manager
                    sr = int(_f.samplerate)
                    _frames = _f.frames
                    _chunk = sr * 300  # 300 s chunks to avoid OOM on long files
                    _parts: list[np.ndarray] = []
                    _read = 0
                    while _read < _frames:
                        _block = _f.read(min(_chunk, _frames - _read))
                        if _block.shape[-1] == 0:
                            break  # EOF — pedalboard.frames can overcount for VBR MP3
                        _parts.append(_block)
                        _read += _block.shape[-1]
                _raw = np.concatenate(_parts, axis=1) if len(_parts) > 1 else _parts[0]
                # pedalboard returns (channels, samples) — transpose to (samples, channels)
                if _raw.ndim == 2:
                    _raw = _raw.T
                audio = _raw.astype(np.float32)
                logger.debug("load_audio_file: pedalboard/FFmpeg OK (%s)", filepath)
            except Exception as _e3:
                result["error"] = f"Audio read error: soundfile + pedalboard failed. Last: {_e3}"
                return result

        # ── Post-processing ──────────────────────────────────────────────────
        # Spec §2.47: nur Mono und Stereo unterstützt. > 2 Kanäle → gewichteter Downmix.
        # WICHTIG: Downmix VOR Resampling — so wird resampy/soxr nur auf 1-2 Kanäle angewendet
        # (statt 3+ Kanäle), was den soxr-np.apply_along_axis-Hänger bei 3-Kanal-MP3 verhindert.
        # PANNs-Plugin noch nicht im Import-Pfad verfügbar → einfacher Energie-Downmix.
        # Der Downmix ergibt Stereo (L=avg(L+odd), R=avg(R+even)) für 4-Kanal-Material
        # und Mono für alle anderen Kanalzahlen (> 2).
        if audio.ndim == 2 and audio.shape[-1] > 2:
            n_ch = audio.shape[-1]
            logger.warning(
                "load_audio_file: %d Kanäle erkannt (nur Mono/Stereo unterstützt) — "
                "Downmix auf Stereo L/R (Energie-gewichtet).",
                n_ch,
            )
            # Energie-gewichteter Downmix: höhere Energie → höherer Beitrag pro Kanal.
            _ch_rms = np.sqrt(np.mean(audio**2, axis=0)) + 1e-9  # (n_ch,)
            _weights = _ch_rms / _ch_rms.sum()
            if n_ch >= 4:
                # L = Summe ungerade Kanäle, R = Summe gerade Kanäle (häufiges LRLS-RS-Schema)
                _l_idx = list(range(0, n_ch, 2))
                _r_idx = list(range(1, n_ch, 2))
                ch_l = float(np.sum(_weights[_l_idx])) + 1e-9
                ch_r = float(np.sum(_weights[_r_idx])) + 1e-9
                _l = np.average(audio[:, _l_idx], axis=1, weights=_weights[_l_idx] / ch_l)
                _r = np.average(audio[:, _r_idx], axis=1, weights=_weights[_r_idx] / ch_r)
                audio = np.stack([_l, _r], axis=-1).astype(np.float32)
            else:
                # 3 o.ä. → Mono-Downmix
                audio = np.average(audio, axis=-1, weights=_weights).astype(np.float32)
            logger.info(
                "load_audio_file: Downmix %d→%d-Kanal abgeschlossen.",
                n_ch,
                1 if audio.ndim == 1 else audio.shape[-1],
            )
        if mono and audio.ndim > 1:
            audio = audio.mean(axis=-1)
        # Resampling NACH Downmix (1-2 Kanäle) — verhindert soxr-Hänger bei 3-Kanal-MP3.
        if target_sr and sr != target_sr:
            import resampy

            if audio.ndim == 1:
                audio = resampy.resample(audio, sr, target_sr)
            else:
                audio = resampy.resample(audio.T, sr, target_sr, axis=-1).T
            sr = target_sr
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Import-stage stereo blind-spot guard: detect unexpected L/R lag early.
        # If critical and sr=48k, run STCG correction immediately.
        _import_lag_before = _estimate_interchannel_lag_samples(audio, sr)
        _import_lag_after = _import_lag_before
        if abs(_import_lag_before) > 64:
            logger.warning(
                "load_audio_file: detected interchannel lag=%d samples (%.1f ms) before pipeline",
                _import_lag_before,
                (_import_lag_before / float(sr)) * 1000.0,
            )
            if sr == 48000:
                try:
                    from backend.core.stereo_temporal_coherence_guard import (
                        get_stereo_temporal_coherence_guard,
                    )

                    audio = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                        audio,
                        sr,
                        phase_id="import_pipeline",
                    )
                    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                    audio = np.clip(audio, -1.0, 1.0)
                    _import_lag_after = _estimate_interchannel_lag_samples(audio, sr)
                    logger.info(
                        "load_audio_file: import lag corrected %d -> %d samples",
                        _import_lag_before,
                        _import_lag_after,
                    )
                except Exception as _lag_fix_exc:
                    logger.debug("load_audio_file: import lag correction skipped: %s", _lag_fix_exc)

        result["audio"] = audio
        result["sr"] = sr
        result["channels"] = 1 if audio.ndim == 1 else audio.shape[-1]
        result["format"] = result["format"] or _ext[1:].upper()
        result["duration"] = result["duration"] or (len(audio) / sr)
        result["interchannel_lag_samples_before"] = int(_import_lag_before)
        result["interchannel_lag_samples_after"] = int(_import_lag_after)
        # Carrier detection (heuristic + forensics)
        # Skipped when do_carrier_analysis=False (e.g. BatchProcessingThread, AudioPlayer,
        # RecoveryCheckpoint) — carrier analysis is done once in _carrier_bg and cached.
        # Running MediumClassifier on a full 225 s audio blocks the load-ticker for 6+ min.
        result["carrier_heuristic"] = detect_carrier(filepath, result["meta"]) if do_carrier_analysis else "unknown"
        if do_carrier_analysis:
            try:
                analyze_carrier_forensics, classify_carrier_ml = _lazy_get_carrier_tools()
                forensic = analyze_carrier_forensics(audio, sr)
                result["carrier_forensic"] = forensic["carrier_forensic"]
                result["carrier_forensic_score"] = forensic["score"]
                result["carrier_forensic_features"] = forensic["features"]
                # ML-based carrier classification
                ml = classify_carrier_ml(forensic["features"])
                result["carrier_ml"] = ml.get("carrier_ml", "Unbekannt")
                result["carrier_ml_confidence"] = ml.get("confidence", 0.0)
                result["carrier_ml_probas"] = ml.get("probas", {})
                result["carrier_ml_explain"] = ml.get("explain", "")
            except Exception as _e_carrier:
                result["carrier_forensic"] = "Unbekannt"
                result["carrier_forensic_score"] = 0
                result["carrier_forensic_features"] = {}
                result["carrier_ml"] = "Unbekannt"
                result["carrier_ml_confidence"] = 0.0
                result["carrier_ml_probas"] = {}
                result["carrier_ml_explain"] = str(_e_carrier)
    except Exception as e:
        result["error"] = f"Unknown error: {e}"
    return result
