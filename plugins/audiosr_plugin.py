"""AudioSR Plugin -- Bandbreiten-Erweiterung via ML + DSP-Fallback.

ML-Pfad (Primaer, Lazy-Load bei eingeschraenkter Bandbreite):
    AudioSR (Liu et al. 2023) mit lokalem Safetensors-Modell.
    Modell: models/audiosr/audiosr_basic_fp16.safetensors (2.9 GB FP16-on-disk, FP32-inference, CPU)
    Aktivierung: wenn effektive Bandbreite (Spektral-Rolloff 90 %) < 15 kHz
    ddim_steps=50 fuer Desktop-CPU-Kompatibilitaet (Standard 200 zu langsam)

DSP-Kaskade (Fallback bei fehlendem Modell oder ML-Fehler):
    1. Polyphase-Resampling (scipy resample_poly, Lanczos-aequivalent)
    2. Spektrale HF-Erweiterung durch harmonische Oberton-Synthese
    3. Sekundärfallback: Spectral Band Replication (SBR)
    4. PGHI-konsistente Phasenrekonstruktion
    4. Psychoakustisch-optimiertes HF-Shelving (> 8 kHz)

Referenz:
    Liu et al. (2023): "AudioSR: Versatile Audio Super-Resolution at Scale"
"""

from __future__ import annotations

# pylint: disable=import-outside-toplevel,reimported
import gc
import importlib
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lokaler AudioSR-Modell-Pfad (kein HuggingFace-Download erforderlich)
# ---------------------------------------------------------------------------
_AUDIOSR_ROOT = Path(__file__).parent.parent / "models" / "audiosr"
_MODEL_SAFETENSORS = _AUDIOSR_ROOT / "audiosr_basic_fp16.safetensors"

# GPU policy: ml_device_manager decides whether to use ROCm/DirectML.
# CUDA_VISIBLE_DEVICES is no longer suppressed unconditionally.
# Lazy-geladenes ML-Modell (einmal pro Prozess, Thread-gesichert)
_ml_model = None
_ml_model_failed: bool = False  # Sentinel: True → kein Retry nach Fehlschlag (§3.2)
_ml_model_lock = threading.Lock()

logger = logging.getLogger(__name__)

_lock: threading.Lock = threading.Lock()
_instance: AudioSRPlugin | None = None


def _detect_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Schaetze effektive Signalbandbreite via Spektral-Rolloff (90 %).

    Returns: Rolloff-Frequenz in Hz.
    """
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
    else:
        mono = audio
    n_fft = min(len(mono), sr)  # max. 1 Sekunde fuer Geschwindigkeit
    if n_fft < 64:
        return float(sr / 2)
    fft_mag = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    cumpow = np.cumsum(fft_mag**2)
    total = cumpow[-1]
    if total < 1e-12:
        return float(sr / 2)
    idx = int(np.searchsorted(cumpow, 0.90 * total))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


# AudioSR FP16-on-disk (cast to FP32 at load): ~2.9 GB on disk → ~7 GB peak RSS in RAM (FP32 inference).
_AUDIOSR_BUDGET_GB: float = 5.75  # 2.9 GB FP16 on-disk → ~5.8 GB FP32 steady-state RAM
# NOTE: ml_memory_budget._preflight_system_memory applies an additional 1.6× peak
# factor for torch.load() overhead.  _AUDIOSR_BUDGET_GB must therefore represent
# the *steady-state* runtime RAM, NOT the inflated peak.  safetensors loading is
# memory-mapped (no double-buffer peak), so ~5.8 GB is the correct upper bound.
# 5.75 GB × 1.6 + oomd_safe ≈ 13.3 GB required vs. 14+ GB available on 32 GB system.


def _get_ml_model() -> object | None:
    """Lazy-Load des AudioSR-ML-Modells (Double-Checked Locking + Sentinel).

    Gibt None zurueck wenn Modell-Datei fehlt, zu wenig RAM vorhanden ist,
    oder Import fehlschlaegt.
    Nach dem ersten Fehlschlag wird _ml_model_failed=True gesetzt und jeder
    weitere Aufruf kehrt sofort zurueck -- kein Retry, kein CUDA-Glob-Timeout
    (Sentinel-Pattern §3.2 copilot-instructions).
    """
    global _ml_model, _ml_model_failed  # pylint: disable=global-statement
    # Schnellpfad ohne Lock (§3.2 Double-Checked Locking)
    if _ml_model is not None:
        return _ml_model  # type: ignore[no-any-return]
    if _ml_model_failed:
        return None  # Kein Retry nach erstem Fehlschlag
    with _ml_model_lock:
        if _ml_model is not None:
            return _ml_model  # type: ignore[no-any-return]
        if _ml_model_failed:
            return None
        if not _MODEL_SAFETENSORS.exists():
            logger.debug(
                "AudioSR: Modell-Datei nicht gefunden (%s) -- DSP-Fallback aktiv.",
                _MODEL_SAFETENSORS,
            )
            _ml_model_failed = True
            return None
        # Globaler ML-Budget-Guard: verhindert kumulative OOM über alle Plugins.
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate as _try_alloc

            if not _try_alloc("AudioSR", _AUDIOSR_BUDGET_GB):
                try:
                    _release("AudioSR")
                except Exception:  # nosec B110
                    pass
                if not _try_alloc("AudioSR", _AUDIOSR_BUDGET_GB):
                    return None  # Budget erschöpft → DSP-Fallback
        except Exception as _exc:
            # §OOM-Guard fail-safe: Exception im Budget-Check → Laden verweigern, nicht zulassen.
            # Fail-open (weiter laden) wäre bei 5.75 GB ein OOM-Risiko.
            logger.warning("AudioSR: Budget-Check fehlgeschlagen (%s) — Laden verweigert (OOM-Fail-safe).", _exc)
            return None
        try:
            # AudioSR-Paket liegt lokal in models/audiosr/ (kein pip install noetig)
            audiosr_pkg = str(_AUDIOSR_ROOT)
            if audiosr_pkg not in sys.path:
                sys.path.insert(0, audiosr_pkg)
            pipeline_module = importlib.import_module("audiosr.pipeline")
            build_model = getattr(pipeline_module, "build_model", None)
            if build_model is None:
                raise ImportError("audiosr.pipeline.build_model nicht gefunden")

            logger.info(
                "AudioSR: Lade ML-Modell von %s (nur einmalig)...",
                _MODEL_SAFETENSORS.name,
            )
            try:
                from backend.core.ml_device_manager import get_torch_device as _get_dev

                _dev = _get_dev("AudioSR")
            except Exception:
                _dev = "cpu"
            # §SOTA: AudioSR-Modell vollständig auf CPU laden.
            # ROCm (AMD GPU) produziert NaN im HiFi-GAN-Vocoder (first_stage_model)
            # aufgrund von transposed-convolution-Bugs im ROCm-Treiber.
            # Der alte Fix (mixed device: DDIM auf GPU, Vocoder auf CPU) führte zu
            # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
            # weil sub-modules inkonsistent auf CPU/GPU verteilt waren.
            # Fix: gesamtes Modell auf CPU. Für 225s Audio mit 50 DDIM-Steps ≈ 15 min,
            # akzeptabel für Offline-Restoration.
            model = build_model(model_name="basic", device="cpu")
            # weight_norm (torch<2.0 compat) kann NaN/Inf in Parametern erzeugen
            # wenn FP16→FP32-Konversion weights mit Norm≈0 produziert.
            # Clean ALL parameters once after loading.
            import torch as _clean_torch
            for _p in model.parameters():
                if not _clean_torch.isfinite(_p).all():
                    _p.data = _clean_torch.nan_to_num(_p.data, nan=0.0, posinf=0.0, neginf=0.0)
            _ml_model = model
            _actual_device = "cpu"
            if hasattr(model, "parameters"):
                try:
                    _actual_device = str(next(model.parameters()).device)
                except (StopIteration, RuntimeError):
                    _actual_device = "cpu"
            logger.info("AudioSR: ML-Modell bereit (device=%s, ddim_steps=50).", _actual_device)
            # PLM-Registrierung für LRU-basierte Auto-Eviction
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("AudioSR", size_gb=_AUDIOSR_BUDGET_GB, unload_fn=unload_audiosr)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            return _ml_model  # type: ignore[return-value,no-any-return]
        except Exception as exc:
            _ml_model_failed = True  # Sentinel setzen -- kein Retry (§3.2)
            # Budget-Slot freigeben bei Ladefehler
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("AudioSR")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            logger.info("AudioSR: ML-Modell-Load fehlgeschlagen (%s) -- DSP-Fallback aktiv.", exc)
            return None


def _run_audiosr_ml(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """ML-Inferenz via AudioSR (Liu et al. 2023).

    Zone-by-zone processing: max _AUDIOSR_ZONE_SECONDS pro Aufruf, damit DDIM-Inferenz-Buffer
    (50 Schritte x Latent-Tensoren) das System nicht in OOM treibt.
    Gibt None zurueck bei Fehler (dann greift DSP-Fallback).
    """
    # Max Zone-Laenge: AudioSR warnt ab 10.24 s; 10 s pro Zone begrenzt DDIM-Peak-RAM
    # auf ~2-3 GB/Zone (statt 10-15 GB fuer 225 s in einem Aufruf).
    _AUDIOSR_ZONE_SECONDS: int = 10
    _AUDIOSR_ZONE_SAMPLES: int = _AUDIOSR_ZONE_SECONDS * sr  # 480 000 @ 48 kHz
    _AUDIOSR_FADE_SAMPLES: int = max(1, sr // 200)  # 5 ms boundary fade to suppress clicks
    # §4.6b PLM active-guard: must be initialised before try-block so except-clause can access it
    _plm_asr = None
    try:
        # Zuerst Modell laden (injiziert sys.path fuer audiosr-Paket)
        model = _get_ml_model()
        if model is None:
            return None

        # §4.6b: set plugin active to block Emergency-Eviction during inference
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_asr_fn

            _plm_asr = _get_plm_asr_fn()
            _plm_asr.set_active("AudioSR", True)
        except Exception as _exc:
            logger.debug("AudioSR: PLM set_active failed: %s", _exc)

        import soundfile as sf

        pipeline_module = importlib.import_module("audiosr.pipeline")
        super_resolution = getattr(pipeline_module, "super_resolution", None)
        if super_resolution is None:
            raise ImportError("audiosr.pipeline.super_resolution nicht gefunden")
        # §audiosr-nowav: make_batch_for_super_resolution direkt importieren um
        # WAV-Datei-Roundtrip zu vermeiden. Alle AudioSR-Versionen die wir verwenden
        # unterstützen den waveform-Parameter (Funktion in audiosr/pipeline.py vorhanden).
        _make_batch_fn = getattr(pipeline_module, "make_batch_for_super_resolution", None)

        # Sicherstellen: float32, immer samples-first (N, ch) für Zone-Loop
        audio_f32 = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio_f32.ndim == 2:
            # Shape-adaptive: Phase_06 übergibt (N, 2) nach audio.T, andere Aufrufer ggf. (2, N).
            # Normalisierung auf samples-first (N, 2):
            #   shape[0] < shape[1] → channels-first (2, N) → .T erforderlich
            #   shape[0] > shape[1] → samples-first (N, 2) → kein .T
            if audio_f32.shape[0] < audio_f32.shape[1]:
                wav_data = audio_f32.T  # channels-first (2, N) → (N, 2)
            else:
                wav_data = audio_f32  # samples-first (N, 2), bereits korrekt
        else:
            wav_data = audio_f32  # shape (N,)

        n_total: int = wav_data.shape[0]

        # --- Zone-Slices berechnen (non-overlapping, Achse 0 = samples) ---
        zone_slices: list[tuple[int, int]] = []
        pos = 0
        while pos < n_total:
            zone_slices.append((pos, min(pos + _AUDIOSR_ZONE_SAMPLES, n_total)))
            pos += _AUDIOSR_ZONE_SAMPLES

        n_zones = len(zone_slices)
        logger.info(
            "AudioSR: %d zone(s) a max %d s (total %.1f s)",
            n_zones,
            _AUDIOSR_ZONE_SECONDS,
            n_total / max(1, sr),
        )

        # §2.45a Wall-time-Budget: adaptiv nach GPU-Verfügbarkeit.
        # GPU: 900 s (15 min). CPU-only: 3600 s (60 min, DDIM ist CPU-intensiv).
        try:
            from backend.core.ml_device_manager import get_device_manager

            _is_gpu = get_device_manager().gpu_available
        except Exception:
            _is_gpu = False
        _AUDIOSR_WALL_BUDGET_S: float = 900.0 if _is_gpu else 3600.0
        _audiosr_ddim_steps: int = 50 if _is_gpu else 25
        _audiosr_t0: float = time.monotonic()

        zone_results: list[np.ndarray] = []
        for z_idx, (z_start, z_end) in enumerate(zone_slices):
            # Wall-time-Budget prüfen: restliche Zonen per Passthrough abschließen
            if z_idx > 0 and (time.monotonic() - _audiosr_t0) > _AUDIOSR_WALL_BUDGET_S:
                logger.warning(
                    "AudioSR: Wall-time budget %.0f s überschritten nach Zone %d/%d "
                    "(%.1f s verbraucht) — Passthrough für restliche %d Zonen",
                    _AUDIOSR_WALL_BUDGET_S,
                    z_idx,
                    n_zones,
                    time.monotonic() - _audiosr_t0,
                    n_zones - z_idx,
                )
                for z_start_rem, z_end_rem in zone_slices[z_idx:]:
                    rem_zone = wav_data[z_start_rem:z_end_rem].copy()
                    zone_results.append(rem_zone.astype(np.float32))
                break
            zone_wav = wav_data[z_start:z_end]

            z_result_raw = None
            if _make_batch_fn is not None:
                # §audiosr-nowav: Direkt-Waveform-Pfad — kein WAV-Datei-Roundtrip.
                # AudioSR erwartet (1, N) mono float32 numpy-Array bei 48000 Hz.
                # Stereo-Input: Kanal-Mittelung (M/S-äquivalent, verlustlos für HF-Rekonstruktion).
                import torch as _asr_torch

                _audiosr_utils = importlib.import_module("audiosr.utils")
                _seed_everything = getattr(_audiosr_utils, "seed_everything", None)
                if _seed_everything is not None:
                    _seed_everything(42)
                # Stereo → Mono (arithmetisches Mittel der Kanäle)
                if zone_wav.ndim > 1:
                    zone_mono = np.mean(zone_wav, axis=1, dtype=np.float32)  # (N, ch) → (N,)
                else:
                    zone_mono = zone_wav.astype(np.float32)
                zone_mono_1d = np.ascontiguousarray(zone_mono[np.newaxis, :])  # (1, N)
                # §2.62 Safety: NaN/Inf im Signal → AudioSR lowpass() schlägt fehl (librosa valid_audio)
                zone_mono_1d = np.nan_to_num(zone_mono_1d, nan=0.0, posinf=0.0, neginf=0.0)
                # Zusätzlich: Clip auf [-1.0, 1.0] (Übersteuerung verhindert Instabilität in sosfiltfilt)
                zone_mono_1d = np.clip(zone_mono_1d, -1.0, 1.0)
                try:
                    batch, _asr_duration = _make_batch_fn(input_file=None, waveform=zone_mono_1d)
                    # Modell ist vollständig auf CPU (ROCm: keine GPU-Mixed-Devices)
                    with _asr_torch.no_grad():
                        z_result_raw = model.generate_batch(
                            batch,
                            unconditional_guidance_scale=3.5,
                            ddim_steps=_audiosr_ddim_steps,
                            duration=_asr_duration,
                        )
                    # generate_batch kann NaN in der Waveform-Rekonstruktion
                    # produzieren. Clean vor Weiterverarbeitung.
                    if hasattr(z_result_raw, "detach"):
                        _z_tmp = z_result_raw.detach().cpu().numpy()
                    else:
                        _z_tmp = np.asarray(z_result_raw, dtype=np.float32)
                    _z_tmp = np.nan_to_num(_z_tmp, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.isfinite(_z_tmp).all():
                        logger.debug("AudioSR Zone %d: NaN/Inf nach generate_batch, cleaned", z_idx + 1)
                    logger.debug(
                        "AudioSR: Zone %d/%d direkt-Waveform-Inferenz (%.1f s)",
                        z_idx + 1,
                        n_zones,
                        zone_mono.shape[0] / max(1, sr),
                    )
                except Exception as _direct_exc:
                    logger.warning(
                        "AudioSR CPU-DDIM fehlgeschlagen: %.100s — Recovery: SBR-DSP",
                        str(_direct_exc),
                    )
                    z_result_raw = None  # Fällt durch zu SBR-DSP-Fallback
            else:
                # Fallback: WAV-Datei-Pfad (Legacy, falls make_batch_for_super_resolution fehlt)
                _asr_tmp_dir: str | None = "/tmp" if os.access("/tmp", os.W_OK) else None  # nosec B108
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=_asr_tmp_dir)
                os.close(tmp_fd)
                try:
                    sf.write(tmp_path, np.ascontiguousarray(zone_wav), sr)
                    _wav_bytes = os.path.getsize(tmp_path)
                    if _wav_bytes < 44:
                        raise OSError(f"AudioSR WAV zu klein: {_wav_bytes} B")
                    z_result_raw = super_resolution(
                        model, tmp_path, seed=42, ddim_steps=_audiosr_ddim_steps, guidance_scale=3.5
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            del zone_wav  # Freigabe vor GC

            # §AUDIOSR-FALLBACK: Wenn generate_batch fehlschlägt (NaN/OOM/Timeout),
            # SBR (Spectral Band Replication) statt primitivem tanh()-Exciter.
            # Kopiert Energie aus dem Quellband (2–6 kHz), transponiert sie um
            # eine Oktave nach oben (4–12 kHz) und blended mit Quell-Spektrum.
            if z_result_raw is None:
                _fallback_zone = wav_data[z_start:z_end].astype(np.float32)
                try:
                    _fb_mono = _fallback_zone.mean(axis=0) if _fallback_zone.ndim > 1 else _fallback_zone
                    from backend.core.dsp.sbr_extend import _sbr_extend

                    _fb_result = _sbr_extend(_fb_mono, sr)
                    if _fallback_zone.ndim > 1:
                        _fb_stereo = np.stack([_fb_result, _fb_result], axis=0)
                        zone_results.append(np.ascontiguousarray(_fb_stereo.astype(np.float32)))
                    else:
                        zone_results.append(np.ascontiguousarray(_fb_result.astype(np.float32)))
                    logger.debug(
                        "AudioSR Zone %d: SBR-DSP-Fallback (%.1f s)", z_idx + 1, len(_fb_mono) / max(1, sr)
                    )
                except Exception:
                    zone_results.append(np.ascontiguousarray(_fallback_zone))
                del _fallback_zone
                continue

            # Tensor -> numpy
            if hasattr(z_result_raw, "detach"):
                z_result_raw = z_result_raw.detach().cpu().numpy()
            z_arr = np.squeeze(np.array(z_result_raw, dtype=np.float32))
            z_arr = np.nan_to_num(z_arr, nan=0.0, posinf=0.0, neginf=0.0)
            z_arr = np.clip(z_arr, -1.0, 1.0)
            del z_result_raw

            # 5 ms Fade-in/-out am Zonenrand (kein Knacken an Zonennaehten)
            fade_n = min(_AUDIOSR_FADE_SAMPLES, z_arr.shape[0])
            fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
            if z_idx > 0:  # Fade-in am Zonenstart (ausser erster Zone)
                if z_arr.ndim > 1:
                    z_arr[:fade_n] *= fade_in[:, np.newaxis]
                else:
                    z_arr[:fade_n] *= fade_in
            if z_idx < n_zones - 1:  # Fade-out am Zonenende (ausser letzter Zone)
                if z_arr.ndim > 1:
                    z_arr[-fade_n:] *= fade_out[:, np.newaxis]
                else:
                    z_arr[-fade_n:] *= fade_out

            zone_results.append(z_arr)
            del z_arr

            # Inferenz-Buffer zwischen Zonen aggressiv freigeben
            gc.collect()
            try:
                import ctypes as _ct

                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception:  # nosec B110
                pass

        # --- Zonen zusammenfuegen ---
        if len(zone_results) == 1:
            result = zone_results[0]
        else:
            result = np.concatenate(zone_results, axis=0)
        del zone_results

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if _plm_asr is not None:
                _plm_asr.set_active("AudioSR", False)
        except Exception as _exc:
            logger.debug("AudioSR: PLM unset_active failed: %s", _exc)
        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.warning("AudioSR ML-Inferenz fehlgeschlagen: %s", exc)
        try:
            if _plm_asr is not None:
                _plm_asr.set_active("AudioSR", False)
        except Exception:  # nosec B110
            pass
        return None


def allow_reset_ml_model_failed() -> None:
    """§Punkt 2: Reset AudioSR Sentinel zur Erlaubung von Wiederholungsversuchen.

    Normaler Pfad: _ml_model_failed wird nur am Phasen-Ende zurückgesetzt (unload_audiosr).
    Neu: Diese Funktion ermöglicht per-Phase Retry nach transienten Fehlern.
    Verwendung: Hinter-PMGG-Retry in Phase mit AudioSR-Nutzung.
    """
    global _ml_model_failed  # pylint: disable=global-statement
    with _ml_model_lock:
        if _ml_model_failed:
            logger.debug("AudioSR: Sentinel reset für Wiederholungsversuch")
            _ml_model_failed = False


def unload_audiosr() -> None:
    """Entlädt das AudioSR-ML-Modell aus dem RAM und gibt das Budget frei.

    Nach dem Entladen fällt jeder nachfolgende Aufruf auf DSP-Fallback zurück.
    Aufruf: direkt nach der letzten AudioSR-Phase in der Pipeline.
    """
    global _ml_model, _ml_model_failed  # pylint: disable=global-statement

    with _ml_model_lock:
        if _ml_model is not None:
            _ml_model = None
            _ml_model_failed = False  # Reset: ermöglicht erneutes Laden bei Bedarf
            gc.collect()
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("AudioSR")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            logger.info("AudioSR: Modell entladen, ~7 GB RAM freigegeben.")


def get_audiosr_plugin() -> AudioSRPlugin:
    """Thread-sicherer Singleton."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AudioSRPlugin()
    return _instance


def has_audiosr_ml_failed() -> bool:
    """Fast sentinel: True when a previous ML load attempt failed (no I/O, thread-safe).

    Callers can use this to skip ML-thread creation entirely and avoid
    blocking join() timeouts (up to 600 s) when the model is unavailable.
    """
    return bool(_ml_model_failed) and _ml_model is None


class AudioSRPlugin:
    """Bandbreiten-Erweiterung: AudioSR ML (primaer) + DSP-Kaskade (Fallback).

    ML-Pfad wird aktiviert wenn:
        - Effektive Bandbreite (Rolloff 90 %) < BW_THRESHOLD Hz, UND
        - Modell-Datei models/audiosr/audiosr_basic_fp16.safetensors existiert

    Methoden:
        process(audio, sr, target_sr)  -> ndarray
        process_files(in_wav, out_wav) -> None
    """

    TARGET_SR: int = 48_000
    BW_THRESHOLD: float = 15_000.0  # Hz -- unter diesem Wert: ML aktiviert

    def __init__(self, **_kwargs: object) -> None:
        logger.debug(
            "AudioSRPlugin initialisiert (ML-Primaer wenn BW < %.0f Hz, DSP-Fallback)",
            self.BW_THRESHOLD,
        )

    # ------------------------------------------------------------------
    def process(
        self,
        audio: np.ndarray,
        sr: int,
        target_sr: int = TARGET_SR,
    ) -> np.ndarray:
        """Bandbreiten-Erweiterung und Resampling.

        ML-Pfad (primaer): AudioSR wenn Quell-Bandbreite eingeschraenkt.
        DSP-Fallback: harmonische Oberton-Synthese + SBR + PGHI.

        Args:
            audio    : float32 [samples] ODER [channels, samples]
            sr       : Eingangs-Sample-Rate in Hz
            target_sr: Ziel-Sample-Rate in Hz (Standard 48 000 Hz)

        Returns: float32 ndarray, selbe Kanalanzahl wie Eingabe
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono_in = audio.ndim == 1

        # Bandbreiten-Erkennung (einmal auf Gesamt-Signal)
        bw_hz = _detect_bandwidth(audio, sr)
        needs_hf = bw_hz < self.BW_THRESHOLD
        logger.debug("AudioSR: erkannte Bandbreite %.0f Hz (Schwelle %.0f Hz).", bw_hz, self.BW_THRESHOLD)

        # ---- ML-Pfad (Primaer) -----------------------------------------------
        if needs_hf and _MODEL_SAFETENSORS.exists():
            logger.info(
                "AudioSR: ML-Bandbreitenerweiterung aktiviert (BW=%.0f Hz < %.0f Hz).",
                bw_hz,
                self.BW_THRESHOLD,
            )
            ml_result = _run_audiosr_ml(audio, sr)
            if ml_result is not None:
                # Ausgabe-SR von AudioSR ist immer 48 kHz
                out_sr = 48_000
                if out_sr != target_sr:
                    ml_result = self._resample(ml_result, out_sr, target_sr)
                if mono_in and ml_result.ndim > 1:
                    ml_result = ml_result.mean(axis=-1) if ml_result.shape[-1] <= 2 else ml_result[: ml_result.shape[0]]
                return np.clip(np.nan_to_num(ml_result.astype(np.float32), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
            logger.debug(
                "AudioSR: ML fehlgeschlagen -- DSP-Kaskade aktiv."
            )  # Erwartet, DSP-Fallback ist designed-in (§3.5)

        # ---- DSP-Fallback -------------------------------------------------------
        if mono_in:
            audio = audio[np.newaxis, :]  # [1, samples]

        channels, _ = audio.shape
        out_ch = []
        for c in range(channels):
            ch = audio[c]
            if needs_hf:
                ch = self._hf_extend(ch, sr)
            if sr != target_sr:
                ch = self._resample(ch, sr, target_sr)
            out_ch.append(ch)

        result = np.stack(out_ch)  # [channels, samples]
        if mono_in:
            result = result[0]
        return np.clip(np.nan_to_num(result.astype(np.float32), nan=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def _hf_extend(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Primärer DSP-Fallback, sekundär Spectral Band Replication."""
        try:
            return self._spectral_exciter(x, sr)
        except Exception as exc:
            logger.debug("HF-Erweiterung fehlgeschlagen: %s — SBR-Fallback aktiv", exc)
            return self._spectral_band_replication(x, sr)

    def _spectral_band_replication(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Sekundärfallback: konservative Spectral Band Replication mit Originalphase."""
        n_fft = 2048
        hop = 256
        win = np.hanning(n_fft).astype(np.float32)
        n_frames = (len(x) - n_fft) // hop + 1
        if n_frames <= 0:
            return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

        specs = []
        for i in range(n_frames):
            seg = x[i * hop : i * hop + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            specs.append(np.fft.rfft(seg * win))

        S = np.stack(specs, axis=1).astype(np.complex64)
        mag = np.abs(S)
        phase = np.angle(S)
        freq_bins = mag.shape[0]
        bin_hz = (sr / 2.0) / max(freq_bins - 1, 1)
        cutoff_b = int(8000 / bin_hz) if bin_hz > 0 else freq_bins
        cutoff_b = int(np.clip(cutoff_b, 8, freq_bins - 1))

        mag_ext = mag.copy()
        src_band = mag[max(1, cutoff_b // 2) : cutoff_b]
        dst_len = freq_bins - cutoff_b
        if src_band.size > 0 and dst_len > 0:
            src_idx = np.linspace(0, src_band.shape[0] - 1, dst_len)
            for t in range(mag.shape[1]):
                replicated = np.interp(src_idx, np.arange(src_band.shape[0]), src_band[:, t])
                tilt = np.linspace(0.85, 0.45, dst_len, dtype=np.float32)
                mag_ext[cutoff_b:, t] = np.maximum(mag_ext[cutoff_b:, t], replicated * tilt)

        S_ext = mag_ext * np.exp(1j * phase)
        try:
            from scipy.signal import istft as _istft_fn

            _, out = _istft_fn(
                S_ext,
                fs=sr,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window=win,
            )
        except Exception as exc:
            logger.debug("SBR-Fallback ISTFT fehlgeschlagen: %s", exc)
            out = x
        out = np.asarray(out, dtype=np.float32)
        if len(out) > len(x):
            out = out[: len(x)]
        elif len(out) < len(x):
            out = np.pad(out, (0, len(x) - len(out)))
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]

    def _spectral_exciter(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Spektraler Exciter via Oberton-Synthese."""
        n_fft = 2048
        hop = 256
        win = np.hanning(n_fft).astype(np.float32)
        n_frames = (len(x) - n_fft) // hop + 1
        if n_frames <= 0:
            return x

        # STFT
        specs = []
        for i in range(n_frames):
            seg = x[i * hop : i * hop + n_fft]
            if len(seg) < n_fft:
                seg = np.pad(seg, (0, n_fft - len(seg)))
            specs.append(np.fft.rfft(seg * win))

        S = np.stack(specs, axis=1).astype(np.complex64)  # [freq, time]
        mag = np.abs(S)
        phase = np.angle(S)

        freq_bins = mag.shape[0]
        src_nyq = sr / 2.0
        bin_hz = src_nyq / (freq_bins - 1)
        cutoff_b = int(8000 / bin_hz) if bin_hz > 0 else freq_bins

        # 2. Harmonische: Energie des [0..cutoff_b/2]-Bereichs verdoppeln
        half = cutoff_b // 2
        harm2_mag = np.zeros_like(mag)
        if half > 0 and 2 * half < freq_bins:
            harm2_mag[half : 2 * half] = mag[:half] * 0.25  # gedaempft
        # Derive harmonic phase from original signal phase — no random (§2.40 determinism)
        harm2_phase = np.zeros_like(phase)
        if half > 0 and 2 * half < freq_bins:
            harm2_phase[half : 2 * half] = phase[:half]  # fold original phase into harmonic range

        S_ext = S + harm2_mag * np.exp(1j * harm2_phase)

        # HF-Shelving: sanfter Boost > 8 kHz
        shelf = np.ones(freq_bins, dtype=np.float32)
        boost_start = min(cutoff_b, freq_bins - 1)
        shelf[boost_start:] = np.linspace(1.0, 1.4, freq_bins - boost_start)
        S_ext *= shelf[:, np.newaxis]

        # iSTFT via PGHI (§4.5 Pflicht: kein Griffin-Lim nach Spektralmodifikation)
        try:
            from dsp.pghi import pghi_reconstruct_from_stft as _pghi_fn

            out = _pghi_fn(S_ext, n_fft, hop, sr)
        except Exception:
            # Fallback: iSTFT mit Originalphase — NIE Griffin-Lim (§4.5)
            from scipy.signal import istft as _istft_fn

            mag_ext = np.abs(S_ext)
            phase_ext = np.angle(S_ext)
            _, out = _istft_fn(
                mag_ext * np.exp(1j * phase_ext),
                fs=sr,
                nperseg=n_fft,
                noverlap=n_fft - hop,
                window=win,
            )
        out = np.asarray(out, dtype=np.float32)
        if len(out) > len(x):
            out = out[: len(x)]
        elif len(out) < len(x):
            out = np.pad(out, (0, len(x) - len(out)))
        return np.clip(out, -1.0, 1.0)  # type: ignore[no-any-return]

    @staticmethod
    def _griffin_lim(
        S: np.ndarray,
        n_fft: int,
        hop: int,
        win: np.ndarray,
        n_iter: int = 32,
        orig_len: int = 0,
    ) -> np.ndarray:
        """Griffin-Lim Phasen-Rekonstruktion (>= 32 Iterationen, PGHI-konsistent)."""
        mag = np.abs(S)
        phase = np.angle(S)
        n_fr = S.shape[1]
        out_len = orig_len if orig_len > 0 else (n_fr - 1) * hop + n_fft

        for _ in range(n_iter):
            sig = np.zeros(out_len + n_fft, dtype=np.float32)
            norm = np.zeros_like(sig)
            for i in range(n_fr):
                frame = np.fft.irfft(mag[:, i] * np.exp(1j * phase[:, i]), n=n_fft).real  # type: ignore[index]
                s = i * hop
                sig[s : s + n_fft] += frame.astype(np.float32) * win
                norm[s : s + n_fft] += win**2
            sig[:out_len] /= np.where(norm[:out_len] < 1e-8, 1.0, norm[:out_len])

            for i in range(n_fr):
                seg = sig[i * hop : i * hop + n_fft]
                if len(seg) < n_fft:
                    seg = np.pad(seg, (0, n_fft - len(seg)))
                phase[:, i] = np.angle(np.fft.rfft(seg * win))  # type: ignore[index]

        return sig[:orig_len]  # type: ignore[no-any-return]

    @staticmethod
    def _resample(x: np.ndarray, src: int, tgt: int) -> np.ndarray:
        """Polyphase-Resampling (scipy resample_poly, Lanczos-aequivalent)."""
        if src == tgt:
            return x
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(tgt, src)
            return resample_poly(x, tgt // g, src // g).astype(np.float32)  # type: ignore[no-any-return]
        except ImportError:
            ratio = tgt / src
            new_n = round(len(x) * ratio)
            idx = np.linspace(0, len(x) - 1, new_n)
            return np.interp(idx, np.arange(len(x)), x).astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def process_files(
        self,
        input_wav: str,
        output_wav: str,
        target_sr: int = TARGET_SR,
    ) -> None:
        """Verarbeite WAV-Datei.

        Args:
            input_wav : Pfad zur Eingabedatei
            output_wav: Pfad zur Ausgabedatei
            target_sr : Ziel-Sample-Rate (Standard 48 000 Hz)
        """
        try:
            import soundfile as sf

            from backend.file_import import load_audio_file

            _res = load_audio_file(input_wav, do_carrier_analysis=False)
            if _res is None:
                raise RuntimeError(f"AudioSR: Datei konnte nicht geladen werden: {input_wav}")
            audio = np.asarray(_res["audio"], dtype=np.float32)
            sr = int(_res["sr"])
            audio = audio[np.newaxis, :] if audio.ndim == 1 else audio.T  # [channels, samples]
            result = self.process(audio, sr, target_sr=target_sr)
            Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_wav,
                result.T if result.ndim == 2 else result,
                target_sr,
            )
            logger.info("AudioSR: %s -> %s (%d Hz)", input_wav, output_wav, target_sr)
        except Exception as exc:
            logger.error("AudioSR process_files fehlgeschlagen: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def upsample_audio(
    audio: np.ndarray,
    sr: int,
    target_sr: int = AudioSRPlugin.TARGET_SR,
) -> np.ndarray:
    """Bandbreiten-Erweiterung (Convenience-Wrapper)."""
    return get_audiosr_plugin().process(audio, sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(_sys.argv) < 3:
        logger.debug("Verwendung: audiosr_plugin.py <input.wav> <output.wav> [target_sr=48000]")
        _sys.exit(1)
    _tsr = int(_sys.argv[3]) if len(_sys.argv) > 3 else 48_000
    get_audiosr_plugin().process_files(_sys.argv[1], _sys.argv[2], target_sr=_tsr)
