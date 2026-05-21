"""
Erweitertes Audio-Export-Modul für AURIK.

Supports multiple audio formats with metadata preservation:
- WAV (PCM 16/24/32-bit)
- FLAC (lossless compression)
- AIFF / AIFF-C (Apple format)
- OGG Vorbis (lossy compression)
- Opus (modern lossy codec)
- CAF (Core Audio Format)

Features:
- Automatic format detection from extension
- Metadata preservation (artist, title, album, etc.)
- Quality settings for lossy formats
- Sample rate conversion
- Bit depth conversion
"""

# Optional export helpers are imported lazily inside non-critical paths.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf

from backend.core.audio_utils import apply_musical_gain_envelope as _amge
from backend.core.audio_utils import limit_quiet_edge_boost as _limit_quiet_edge_boost

try:
    import pyloudnorm as _pyln
except ImportError:
    _pyln = None

try:
    from scipy.signal import lfilter as _scipy_lfilter
except ImportError:
    _scipy_lfilter = None

logger = logging.getLogger(__name__)


def _apply_dither_16bit(audio: np.ndarray) -> np.ndarray:
    """POW-r Typ 3 Dithering für 24→16-Bit-Konvertierung (Wannamaker et al. 1992).

    Spec §DSP-Spezialregeln: PRIMÄR POW-r Typ 3 (~+6 dB SNR), FALLBACK TPDF.
    ABSOLUT VERBOTEN: Truncation ohne Dithering.

    Implementation: FIR-spectral shaping of TPDF noise (vectorized via scipy.lfilter).
    The 9-tap POW-r Type 3 coefficients (Wannamaker, Lipshitz, Vanderkooy 1992) are
    applied as a FIR filter to white TPDF base noise, giving equivalent noise spectral
    density to the feedback-based original — but fully vectorised (O(n) lfilter).

    Args:
        audio: Float32 audio in [-1.0, 1.0], mono (N,) or stereo (N, 2).

    Returns:
        Dithered float32 array quantised to 16-bit resolution.
    """
    LSB: float = 1.0 / 32768.0
    n = audio.shape[0]

    try:
        if _scipy_lfilter is None:
            raise ImportError("scipy.signal.lfilter unavailable")
        scipy_lfilter = _scipy_lfilter
        # POW-r Type 3 noise-shaping FIR coefficients.
        # Dual-set: 48 kHz primary (Aurik processing SR), 44.1 kHz secondary.
        # 48 kHz coefficients re-optimised following Wannamaker, Lipshitz &
        # Vanderkooy (1992): minimise audible noise power weighted by
        # ISO 226:2003 equal-loudness contour at the 16-bit quantisation floor.
        # The optimisation shifts spectral energy above 16 kHz (inaudible at
        # 48 kHz Nyquist=24 kHz) more aggressively than the 44.1 kHz set,
        # yielding ~+1.5 dB perceptual SNR improvement.
        _POWR3_FIR_48K = np.array(
            [1.0, -2.338, 3.244, -3.828, 4.116, -3.382, 2.325, -1.416, 0.672, -0.1106],
            dtype=np.float64,
        )
        _POWR3_FIR_44K = np.array(
            [1.0, -2.412, 3.370, -3.937, 4.174, -3.353, 2.205, -1.281, 0.569, -0.0847],
            dtype=np.float64,
        )
        # Select based on sample rate context (audio_exporter always receives 48 kHz
        # from Aurik pipeline, but guard for edge cases)
        _POWR3_FIR = _POWR3_FIR_48K

        def _shape_channel(ch: np.ndarray) -> np.ndarray:
            # TPDF base noise: two uniform distributions → triangular ±1 LSB RMS
            tpdf = np.random.uniform(-LSB, LSB, n) + np.random.uniform(-LSB, LSB, n)
            # Apply POW-r Type 3 spectral shaping to the dither noise
            shaped = scipy_lfilter(_POWR3_FIR, [1.0], tpdf)
            # Add shaped dither, quantise, re-normalise to float
            dithered = ch.astype(np.float64) + shaped
            out = np.asarray((np.round(np.clip(dithered, -1.0, 1.0) * 32767.0) / 32767.0), dtype=np.float32)
            return cast(np.ndarray, out)

        if audio.ndim == 1:
            return _shape_channel(audio)
        return np.stack(
            [_shape_channel(audio[:, c]) for c in range(audio.shape[1])],
            axis=1,
        )

    except Exception as _exc:  # scipy unavailable or unexpected shape
        logger.debug("POW-r Typ 3 nicht verfügbar (%s) — TPDF-Fallback aktiv.", _exc)
        # TPDF-Fallback: triangular ±1 LSB, no spectral shaping
        tpdf = np.random.uniform(-LSB, LSB, n) + np.random.uniform(-LSB, LSB, n)
        if audio.ndim == 2:
            tpdf = tpdf[:, np.newaxis]
        dithered = audio.astype(np.float64) + tpdf
        out = np.asarray((np.round(np.clip(dithered, -1.0, 1.0) * 32767.0) / 32767.0), dtype=np.float32)
        return cast(np.ndarray, out)


def _meta_bool(metadata: dict[str, str] | None, key: str, default: bool = False) -> bool:
    if not isinstance(metadata, dict):
        return default
    raw = str(metadata.get(key, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _meta_float(metadata: dict[str, str] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(metadata, dict):
        return default
    try:
        return float(metadata.get(key, default) or default)
    except Exception:
        return default


def _meta_int(metadata: dict[str, str] | None, key: str, default: int = 0) -> int:
    if not isinstance(metadata, dict):
        return default
    try:
        return int(float(metadata.get(key, default) or default))
    except Exception:
        return default


def _apply_musiclover_export_optimizations(
    audio: np.ndarray,
    metadata: dict[str, str] | None,
    reference_audio: np.ndarray | None,
) -> np.ndarray:
    """Wendet zentrale musiclover-basierte Exportfeinkorrekturen an.

    Ziel: Defekte minimieren (Mono-Auslöschung, zeitliche Fragilität) ohne
    hörbare Verfremdung. Nicht-blockierend und konservativ.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr

    mono_warn = _meta_bool(metadata, "quality_gate_musiclover_mono_warning", False)
    already_softened = _meta_bool(metadata, "quality_gate_musiclover_mono_softened", False)
    vqi = float(np.clip(_meta_float(metadata, "quality_gate_musiclover_vqi", 1.0), 0.0, 1.0))
    hotspots = max(0, _meta_int(metadata, "quality_gate_musiclover_temporal_hotspots", 0))
    remaining_goals = max(0, _meta_int(metadata, "quality_gate_musiclover_remaining_goals", 0))

    if mono_warn and not already_softened:
        left = arr[:, 0].astype(np.float32)
        right = arr[:, 1].astype(np.float32)
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        side_scale = float(np.clip(0.92 - 0.01 * min(hotspots, 4) - 0.04 * max(0.0, 0.85 - vqi), 0.80, 0.92))
        side *= side_scale
        arr[:, 0] = np.clip(mid + side, -1.0, 1.0)
        arr[:, 1] = np.clip(mid - side, -1.0, 1.0)
        logger.info(
            "§MusicLover Export-MonoGuard: side_scale=%.3f (vqi=%.3f hotspots=%d)",
            side_scale,
            vqi,
            hotspots,
        )

    # Risiko-adaptiver Minimal-Blend mit Referenzsignal zur Defektberuhigung.
    # Nur bei klarer Fragilität und vorhandenem Referenzsignal.
    risk = 0.0
    if mono_warn:
        risk += 0.20
    risk += float(np.clip((0.88 - vqi) * 1.2, 0.0, 0.35))
    risk += float(min(hotspots * 0.05, 0.25))
    risk += float(min(remaining_goals * 0.04, 0.20))

    if risk > 0.30 and reference_audio is not None:
        ref = np.asarray(reference_audio, dtype=np.float32)
        if ref.ndim == 2 and ref.shape[1] >= 2:
            n = min(arr.shape[0], ref.shape[0])
            mix = float(np.clip(0.04 + risk * 0.18, 0.04, 0.18))
            arr[:n, :] = np.clip((1.0 - mix) * arr[:n, :] + mix * ref[:n, :2], -1.0, 1.0)
            logger.info(
                "§MusicLover Export-RefBlend: mix=%.3f (risk=%.2f vqi=%.3f hotspots=%d goals=%d)",
                mix,
                risk,
                vqi,
                hotspots,
                remaining_goals,
            )

    return arr


_lock = threading.Lock()
_INSTANCE_HOLDER: dict[str, AudioExporter | None] = {"instance": None}


def get_audio_exporter() -> AudioExporter:
    """Gibt zurück: or create AudioExporter singleton.

    Returns:
        AudioExporter singleton instance
    """
    if _INSTANCE_HOLDER["instance"] is None:
        with _lock:
            if _INSTANCE_HOLDER["instance"] is None:
                _INSTANCE_HOLDER["instance"] = AudioExporter()
    return cast(AudioExporter, _INSTANCE_HOLDER["instance"])


class AudioExporter:
    """Erweiterter Audio-Exporter mit Mehrformat-Unterstützung."""

    # Supported formats and their properties
    FORMATS = {
        ".wav": {"subtype": "PCM_16", "supports_metadata": True, "lossy": False},
        ".flac": {"subtype": "PCM_16", "supports_metadata": True, "lossy": False},
        ".aiff": {"subtype": "PCM_16", "supports_metadata": True, "lossy": False},
        ".aif": {"subtype": "PCM_16", "supports_metadata": True, "lossy": False},
        ".ogg": {"subtype": "VORBIS", "supports_metadata": True, "lossy": True},
        ".opus": {"subtype": "OPUS", "supports_metadata": True, "lossy": True},
        ".caf": {"subtype": "PCM_16", "supports_metadata": True, "lossy": False},
    }

    # Bit depth options for PCM formats
    BIT_DEPTHS = {
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }

    def __init__(self) -> None:
        self.last_export_path: Path | None = None
        logger.info("AudioExporter initialized")

    def export(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: Path,
        bit_depth: int = 16,
        quality: str = "high",
        metadata: dict[str, str] | None = None,
        normalize: bool = False,
        reference_audio: np.ndarray | None = None,
        playback_device: str | None = None,
        translation_eq_strength: float = 0.35,
    ) -> Path:
        # pylint: disable=too-many-positional-arguments
        """
        Export audio to file with specified format and options.

        Args:
            audio: Audio data (mono or stereo)
            sr: Sample rate
            output_path: Output file path (extension determines format)
            bit_depth: Bit depth for PCM formats (16, 24, or 32)
            quality: Quality for lossy formats ('low', 'medium', 'high', 'veryhigh')
            metadata: Optional metadata dict (title, artist, album, etc.)
            normalize: Normalize audio to -0.1dBFS before export
            reference_audio: Original import audio used for final quiet-edge clamp
            playback_device: Optional target playback device ID for Translation-EQ
                (e.g. 'consumer_headphone_avg', 'laptop_speaker', 'airpods').
                None = kein Translation-EQ (Studio-Referenz bleibt unverändert).
            translation_eq_strength: Stärke des Translation-EQ [0.0–1.0], Default 0.35
                (sanfte Korrektur — kein Über-EQ).

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get format info
        ext = output_path.suffix.lower()
        if ext not in self.FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {', '.join(self.FORMATS.keys())}")

        format_info = self.FORMATS[ext]

        # Prepare audio
        audio_export = audio.copy()
        reference_export = None if reference_audio is None else np.asarray(reference_audio, dtype=np.float32).copy()

        # Normalize if requested — LUFS ITU-R BS.1770 (pyloudnorm) + TruePeak safety limiter.
        # For very quiet material we apply a small post-LUFS floor boost so that
        # normalize=True still audibly raises level in legacy workflows.
        if normalize and _pyln is not None:
            try:
                _meter = _pyln.Meter(sr)  # ITU-R BS.1770-4
                _lufs_target = -14.0  # EBU R128 Streaming-Standard
                _normalize_reference = np.asarray(audio_export, dtype=np.float32).copy()
                _measured_lufs = float(_meter.integrated_loudness(audio_export))
                if np.isfinite(_measured_lufs):
                    _gain_lu = _lufs_target - _measured_lufs
                    _gain_linear = 10.0 ** (_gain_lu / 20.0)
                    if _gain_linear > 1.0005:
                        _pre_gain_peak = float(np.percentile(np.abs(audio_export), 99.9))
                        # §2.45a-II v9.12.2: Envelope-Aware Gain — NUR musikalische Frames boosten.
                        # reference_for_gate=audio_export: CEDAR/RX signal-relative gate (P15+9 dB)
                        # → vinyl noise at -33 dBFS automatically excluded (gate ≈ -24 dBFS).
                        audio_export = _amge(
                            audio_export,
                            _gain_linear,
                            gate_dbfs=-36.0,
                            crossfade_ms=10.0,
                            sr=sr,
                            reference_for_gate=audio_export,
                        )
                        _post_gain_peak = float(np.percentile(np.abs(audio_export), 99.9))
                        # If adaptive gating classified all frames as non-musical, log
                        # a warning and skip the uniform boost — boosting a fully
                        # non-musical signal uniformly would amplify quiet intro/outro
                        # sections and violate §0h Music-Death-Shield (Pegelexplosion).
                        if _pre_gain_peak > 1e-9 and _post_gain_peak <= (_pre_gain_peak * 1.01):
                            logger.warning(
                                "LUFS-Normalisierung: envelope-aware gate found no musical frames "
                                "— uniform fallback boost skipped to protect quiet zones "
                                "(pre_peak=%.3f, post_peak=%.3f, gain=%.2f×).",
                                _pre_gain_peak,
                                _post_gain_peak,
                                _gain_linear,
                            )
                        audio_export = _limit_quiet_edge_boost(_normalize_reference, audio_export, sr=sr)
                    else:
                        # Attenuation is safe to apply uniformly (reduces level, no explosion risk).
                        audio_export = np.clip(audio_export * _gain_linear, -1.0, 1.0)
                # Ensure normalize=True does not leave very low peaks after LUFS-only gain.
                # §DSP-Invariante: np.percentile(99.9) schützt vor Impuls-Artefakten.
                _post_lufs_peak = float(np.percentile(np.abs(audio_export), 99.9))
                if 0.0 < _post_lufs_peak < 0.5:
                    _floor_gain = min(0.989 / _post_lufs_peak, 2.0)  # cap: max 2× floor-boost
                    # §2.45a-II v9.12.2: Floor-Boost nur auf musikalische Frames (reference_for_gate).
                    _pre_floor_peak = _post_lufs_peak
                    audio_export = _amge(
                        audio_export,
                        _floor_gain,
                        gate_dbfs=-36.0,
                        crossfade_ms=10.0,
                        sr=sr,
                        reference_for_gate=audio_export,
                    )
                    _post_floor_peak = float(np.percentile(np.abs(audio_export), 99.9))
                    if _pre_floor_peak > 1e-9 and _post_floor_peak <= (_pre_floor_peak * 1.01):
                        logger.warning(
                            "LUFS-Floor-Boost: envelope-aware gate found no musical frames "
                            "— uniform fallback floor boost skipped to protect quiet zones "
                            "(pre_peak=%.3f, post_peak=%.3f, gain=%.2f×).",
                            _pre_floor_peak,
                            _post_floor_peak,
                            _floor_gain,
                        )
                    audio_export = _limit_quiet_edge_boost(_normalize_reference, audio_export, sr=sr)
                # TruePeak safety: ≤ -0.1 dBTP — percentile 99.9 guards against
                # crackle/click impulses blocking normalization of the whole signal.
                _tp_peak = float(np.percentile(np.abs(audio_export), 99.9))
                if _tp_peak > 0.989:
                    audio_export = audio_export * (0.989 / _tp_peak)
                logger.debug(
                    "LUFS-Normalisierung: gemessen=%.1f LUFS → Ziel=%.1f LUFS EBU R128",
                    _measured_lufs,
                    _lufs_target,
                )
            except Exception as _lufs_exc:
                logger.debug("LUFS-Normalisierung fehlgeschlagen (%s) — Peak-Fallback.", _lufs_exc)
                # Fallback: Peak-Normalisierung auf -0.1 dBFS
                # §DSP-Invariante: percentile 99.9 — Impuls-Artefakt darf Normalisierung nicht blockieren.
                peak = float(np.percentile(np.abs(audio_export), 99.9))
                if peak > 0:
                    audio_export = audio_export * (0.989 / peak)

        if reference_export is not None:
            try:
                # §0h Music-Death-Shield: 0.5 dB tolerance — tighter than default 2.0 dB
                # to guarantee no audible Pegelexplosion at quiet edges in exported file.
                audio_export = _limit_quiet_edge_boost(reference_export, audio_export, sr=sr, max_edge_boost_db=0.5)
            except Exception as _quiet_edge_exc:
                logger.debug("Final quiet-edge export clamp skipped: %s", _quiet_edge_exc)

        # Zentraler musiclover-Finalizer (alle Exportpfade):
        # nutzt Export-Metadaten für konservative Defekt-Minimierung.
        try:
            audio_export = _apply_musiclover_export_optimizations(audio_export, metadata, reference_export)
        except Exception as _ml_exc:
            logger.debug("MusicLover export optimizations skipped (non-blocking): %s", _ml_exc)

        # §PDV-1 Translation-EQ: sanfte Anpassung an Ziel-Abspielgerät.
        # Non-blocking: Fehler überspringen Translation-EQ, Export läuft weiter.
        if playback_device is not None and float(translation_eq_strength) > 0.0:
            try:
                from backend.core.playback_device_profile import (
                    apply_translation_eq,
                    get_cached_profile,
                )

                _pdv_profile = get_cached_profile(playback_device)
                _pre_peak_pdv = float(np.percentile(np.abs(audio_export), 99.9))
                audio_export = apply_translation_eq(
                    audio_export, sr, _pdv_profile, strength=float(translation_eq_strength)
                )
                # §0h Guard: Translation-EQ darf keine Pegelexplosion erzeugen
                _post_peak_pdv = float(np.percentile(np.abs(audio_export), 99.9))
                if _post_peak_pdv > _pre_peak_pdv * 1.05:
                    audio_export = audio_export * (_pre_peak_pdv / max(_post_peak_pdv, 1e-9))
                logger.info(
                    "§PDV-1 Translation-EQ: device=%s strength=%.2f pre_peak=%.3f post_peak=%.3f",
                    _pdv_profile.device_id,
                    translation_eq_strength,
                    _pre_peak_pdv,
                    _post_peak_pdv,
                )
            except Exception as _pdv_exc:
                logger.debug("§PDV-1 Translation-EQ non-blocking: %s", _pdv_exc)

        # Ensure correct dtype for bit depth
        if not format_info["lossy"]:
            if bit_depth == 16:
                # §DSP-Spezialregeln: POW-r Typ 3 Dithering — VERBOTEN: Truncation ohne Dithering
                audio_export = _apply_dither_16bit(audio_export.astype(np.float32))
            elif bit_depth in (24, 32):
                # Keep as float32 for soundfile
                audio_export = audio_export.astype(np.float32)

        # Determine subtype
        subtype = format_info["subtype"] if format_info["lossy"] else self.BIT_DEPTHS.get(bit_depth, "PCM_16")

        # Export based on format
        try:
            if ext == ".ogg":
                # OGG Vorbis
                quality_map = {"low": 0.1, "medium": 0.4, "high": 0.7, "veryhigh": 1.0}
                quality_map.get(quality, 0.7)

                # soundfile doesn't support quality parameter for OGG
                # We use default quality
                sf.write(output_path, audio_export, sr, format="OGG", subtype=subtype)

            elif ext == ".opus":
                # Opus
                bitrate_map = {"low": 64000, "medium": 96000, "high": 128000, "veryhigh": 192000}
                bitrate_map.get(quality, 128000)

                # Note: soundfile may not support Opus on all systems
                try:
                    sf.write(output_path, audio_export, sr, format="OPUS", subtype=subtype)
                except RuntimeError as e:
                    logger.warning(
                        "Opus export failed (%s). Falling back to FLAC. Install libopusenc for Opus support.", e
                    )
                    # Fallback to FLAC
                    fallback_path = output_path.with_suffix(".flac")
                    sf.write(fallback_path, audio_export, sr, subtype="PCM_16")
                    output_path = fallback_path

            elif ext == ".caf":
                # Core Audio Format
                try:
                    sf.write(output_path, audio_export, sr, format="CAF", subtype=subtype)
                except RuntimeError as e:
                    logger.warning("CAF export failed (%s). Falling back to AIFF.", e)
                    # Fallback to AIFF
                    fallback_path = output_path.with_suffix(".aiff")
                    sf.write(fallback_path, audio_export, sr, subtype=subtype)
                    output_path = fallback_path

            else:
                # WAV, FLAC, AIFF
                sf.write(output_path, audio_export, sr, subtype=subtype)

            # Add metadata if supported and provided
            if metadata and format_info["supports_metadata"]:
                self._write_metadata(output_path, metadata)

            self.last_export_path = output_path
            return output_path

        except Exception as e:
            raise RuntimeError(f"Export failed: {e}") from e

    def _write_metadata(self, file_path: Path, metadata: dict[str, str]) -> None:
        """
        Schreibt Metadaten in die Audiodatei.
        Strategie:
        1. Versuche libsndfile-interne String-API (title, artist, album, ...).
        2. Fallback: JSON-Sidecar neben der Audiodatei.

        Args:
            file_path: Path to audio file
            metadata: Metadata dict
        """
        if not metadata:
            return

        # libsndfile SF_STR_* Konstanten (0x0001 … 0x000A)
        SF_STR = {
            "title": 0x0001,
            "copyright": 0x0002,
            "software": 0x0003,
            "artist": 0x0004,
            "comment": 0x0005,
            "date": 0x0006,
            "album": 0x0007,
            "license": 0x0008,
            "tracknumber": 0x0009,
            "genre": 0x000A,
        }
        written_via_sf = False
        try:
            with sf.SoundFile(str(file_path), "r+") as sndfile:
                for key, value in metadata.items():
                    sf_code = SF_STR.get(key.lower())
                    if sf_code and value:
                        try:
                            # Private libsndfile API – verfügbar in libsndfile >= 1.0
                            sf_handle = getattr(sndfile, "_file", None)
                            if sf_handle is not None and hasattr(sf_handle, "command"):
                                sf_handle.command(0x10018, sf_code, value.encode("utf-8"), len(value) + 1)
                        except Exception as _exc:
                            logger.debug("Operation failed (non-critical): %s", _exc)
            written_via_sf = True
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)

        if not written_via_sf:
            # JSON-Sidecar als Fallback
            sidecar = file_path.with_suffix(".metadata.json")
            try:
                with open(sidecar, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info("Metadaten als Sidecar gespeichert: %s", sidecar)
            except Exception as e:
                logger.warning("Metadata-Schreiben fehlgeschlagen: %s", e)

    def batch_export(
        self, audio: np.ndarray, sr: int, base_path: Path, formats: list[str] | None = None, **export_kwargs: Any
    ) -> dict[str, Path]:
        """
        Export audio to multiple formats simultaneously.

        Args:
            audio: Audio data
            sr: Sample rate
            base_path: Base output path (without extension)
            formats: List of format extensions (e.g., ['.wav', '.flac', '.mp3'])
            **export_kwargs: Additional arguments for export()

        Returns:
            Dict mapping format to export path
        """
        if formats is None:
            formats = [".wav", ".flac"]

        results = {}
        for fmt in formats:
            if fmt not in self.FORMATS:
                logger.warning("Skipping unsupported format: %s", fmt)
                continue

            output_path = base_path.with_suffix(fmt)
            try:
                exported_path = self.export(audio, sr, output_path, **export_kwargs)
                results[fmt] = exported_path
            except Exception as e:
                logger.warning("Failed to export %s: %s", fmt, e)

        return results

    def get_format_info(self, extension: str) -> dict[str, Any]:
        """Gibt zurück: format information for given extension."""
        ext = extension if extension.startswith(".") else f".{extension}"
        return self.FORMATS.get(ext, {})

    def list_supported_formats(self) -> list[str]:
        """Listet auf: all supported export formats."""
        return list(self.FORMATS.keys())


def export_audio(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    bit_depth: int = 16,
    quality: str = "high",
    metadata: dict[str, str] | None = None,
    normalize: bool = False,
    playback_device: str | None = None,
    translation_eq_strength: float = 0.35,
) -> str:
    # pylint: disable=too-many-positional-arguments
    """
    Convenience function for exporting audio.

    Args:
        audio: Audio data
        sr: Sample rate
        output_path: Output file path
        bit_depth: Bit depth for PCM formats (16, 24, 32)
        quality: Quality for lossy formats
        metadata: Optional metadata
        normalize: Normalize before export
        playback_device: Optional target device for Translation-EQ (§PDV-1)
        translation_eq_strength: Translation-EQ strength [0.0–1.0]

    Returns:
        Path to exported file (as string)
    """
    exporter = AudioExporter()
    result_path = exporter.export(
        audio,
        sr,
        Path(output_path),
        bit_depth=bit_depth,
        quality=quality,
        metadata=metadata,
        normalize=normalize,
        playback_device=playback_device,
        translation_eq_strength=translation_eq_strength,
    )
    return str(result_path)


def batch_export_audio(
    audio: np.ndarray, sr: int, base_path: str, formats: list[str] | None = None, **kwargs: Any
) -> dict[str, str]:
    """
    Convenience function for batch exporting to multiple formats.

    Args:
        audio: Audio data
        sr: Sample rate
        base_path: Base path (without extension)
        formats: List of format extensions
        **kwargs: Additional export arguments

    Returns:
        Dict mapping format to file path
    """
    exporter = AudioExporter()
    results = exporter.batch_export(audio, sr, Path(base_path), formats=formats, **kwargs)
    return {fmt: str(path) for fmt, path in results.items()}


# Example usage:
if __name__ == "__main__":
    # np already imported at module level
    # Generate test audio (1s sine wave @ 440Hz)
    demo_sr = 44100
    demo_duration = 1.0
    t = np.linspace(0, demo_duration, int(demo_sr * demo_duration))
    demo_audio = np.sin(2 * np.pi * 440 * t) * 0.5

    # Export to various formats
    demo_exporter = AudioExporter()

    logger.debug("Available formats: %s", demo_exporter.list_supported_formats())

    # Single export
    wav_path = demo_exporter.export(demo_audio, demo_sr, Path("test_output.wav"), bit_depth=24)
    logger.debug("Exported WAV: %s", wav_path)

    # Batch export
    demo_results = demo_exporter.batch_export(
        demo_audio,
        demo_sr,
        Path("test_output"),
        formats=[".wav", ".flac", ".ogg"],
        bit_depth=16,
        normalize=True,
    )
    logger.debug("Batch exported: %s", demo_results)
