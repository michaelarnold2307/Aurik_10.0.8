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

import importlib
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lokaler AudioSR-Modell-Pfad (kein HuggingFace-Download erforderlich)
# ---------------------------------------------------------------------------
_AUDIOSR_ROOT = Path(__file__).parent.parent / "models" / "audiosr"
_MODEL_SAFETENSORS = _AUDIOSR_ROOT / "audiosr_basic_fp16.safetensors"

# CPU-only-Pflicht (§9.5 copilot-instructions): kein CUDA, kein GPU
# Vor dem ersten torch-Import setzen, damit CUDA-Initialisierung übersprungen wird.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
    mono = (
        audio if audio.ndim == 1 else (audio.mean(axis=0) if audio.shape[0] <= audio.shape[-1] else audio.mean(axis=1))
    )
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
_AUDIOSR_BUDGET_GB: float = 7.0


def _get_ml_model() -> object | None:
    """Lazy-Load des AudioSR-ML-Modells (Double-Checked Locking + Sentinel).

    Gibt None zurueck wenn Modell-Datei fehlt, zu wenig RAM vorhanden ist,
    oder Import fehlschlaegt.
    Nach dem ersten Fehlschlag wird _ml_model_failed=True gesetzt und jeder
    weitere Aufruf kehrt sofort zurueck -- kein Retry, kein CUDA-Glob-Timeout
    (Sentinel-Pattern §3.2 copilot-instructions).
    """
    global _ml_model, _ml_model_failed
    # Schnellpfad ohne Lock (§3.2 Double-Checked Locking)
    if _ml_model is not None:
        return _ml_model
    if _ml_model_failed:
        return None  # Kein Retry nach erstem Fehlschlag
    with _ml_model_lock:
        if _ml_model is not None:
            return _ml_model
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
                except Exception:
                    pass
                if not _try_alloc("AudioSR", _AUDIOSR_BUDGET_GB):
                    return None  # Budget erschöpft → DSP-Fallback
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)  # Budget-Modul nicht verfügbar — weiter
        try:
            # CUDA_VISIBLE_DEVICES="" wurde bereits am Modulstart gesetzt (§9.5)
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
            # §2.40 Determinismus-Invariante: AudioSR läuft auf CPU (device="cpu").
            # Falls audiosr-Internals ONNX-Sub-Sessions nutzen, müssen diese
            # providers=["CPUExecutionProvider"] setzen (kein CUDA, kein GPU-Dispatch).
            model = build_model(model_name="basic", device="cpu")
            _ml_model = model
            logger.info("AudioSR: ML-Modell bereit (CPU, ddim_steps=50).")
            # PLM-Registrierung für LRU-basierte Auto-Eviction
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("AudioSR", size_gb=_AUDIOSR_BUDGET_GB, unload_fn=unload_audiosr)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
            return _ml_model
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

    Schreibt Audio in Temp-WAV, ruft super_resolution() auf, gibt float32-Array zurueck.
    Gibt None zurueck bei Fehler (dann greift DSP-Fallback).
    """
    try:
        # Zuerst Modell laden (injiziert sys.path fuer audiosr-Paket)
        model = _get_ml_model()
        if model is None:
            return None

        import soundfile as sf

        pipeline_module = importlib.import_module("audiosr.pipeline")
        super_resolution = getattr(pipeline_module, "super_resolution", None)
        if super_resolution is None:
            raise ImportError("audiosr.pipeline.super_resolution nicht gefunden")

        # Sicherstellen: float32, 1D oder [samples, ch] fuer soundfile
        audio_f32 = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio_f32.ndim == 2:
            wav_data = audio_f32.T  # [samples, channels] fuer soundfile
        else:
            wav_data = audio_f32  # [samples]

        # Temp-WAV schreiben
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            sf.write(tmp_path, wav_data, sr)
            result = super_resolution(
                model,
                tmp_path,
                seed=42,
                ddim_steps=50,  # Desktop-CPU-Budget
                guidance_scale=3.5,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Ergebnis normalisieren
        if hasattr(result, "detach"):  # torch.Tensor
            result = result.detach().cpu().numpy()
        result = np.squeeze(np.array(result, dtype=np.float32))
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0)

    except Exception as exc:
        logger.warning("AudioSR ML-Inferenz fehlgeschlagen: %s", exc)
        return None


def allow_reset_ml_model_failed() -> None:
    """§Punkt 2: Reset AudioSR Sentinel zur Erlaubung von Wiederholungsversuchen.

    Normaler Pfad: _ml_model_failed wird nur am Phasen-Ende zurückgesetzt (unload_audiosr).
    Neu: Diese Funktion ermöglicht per-Phase Retry nach transienten Fehlern.
    Verwendung: Hinter-PMGG-Retry in Phase mit AudioSR-Nutzung.
    """
    global _ml_model_failed
    with _ml_model_lock:
        if _ml_model_failed:
            logger.debug("AudioSR: Sentinel reset für Wiederholungsversuch")
            _ml_model_failed = False


def unload_audiosr() -> None:
    """Entlädt das AudioSR-ML-Modell aus dem RAM und gibt das Budget frei.

    Nach dem Entladen fällt jeder nachfolgende Aufruf auf DSP-Fallback zurück.
    Aufruf: direkt nach der letzten AudioSR-Phase in der Pipeline.
    """
    global _ml_model, _ml_model_failed
    import gc

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
    global _instance
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
                return np.clip(np.nan_to_num(ml_result.astype(np.float32), nan=0.0), -1.0, 1.0)
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
        return np.clip(np.nan_to_num(result.astype(np.float32), nan=0.0), -1.0, 1.0)

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
            return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

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
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

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
        return np.clip(out, -1.0, 1.0)

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
                frame = np.fft.irfft(mag[:, i] * np.exp(1j * phase[:, i]), n=n_fft).real
                s = i * hop
                sig[s : s + n_fft] += frame.astype(np.float32) * win
                norm[s : s + n_fft] += win**2
            sig[:out_len] /= np.where(norm[:out_len] < 1e-8, 1.0, norm[:out_len])

            for i in range(n_fr):
                seg = sig[i * hop : i * hop + n_fft]
                if len(seg) < n_fft:
                    seg = np.pad(seg, (0, n_fft - len(seg)))
                phase[:, i] = np.angle(np.fft.rfft(seg * win))

        return sig[:orig_len]

    @staticmethod
    def _resample(x: np.ndarray, src: int, tgt: int) -> np.ndarray:
        """Polyphase-Resampling (scipy resample_poly, Lanczos-aequivalent)."""
        if src == tgt:
            return x
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(tgt, src)
            return resample_poly(x, tgt // g, src // g).astype(np.float32)
        except ImportError:
            ratio = tgt / src
            new_n = round(len(x) * ratio)
            idx = np.linspace(0, len(x) - 1, new_n)
            return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

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
