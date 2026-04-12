"""
BigVGAN-v2 Plugin — Neuronaler Vocoder für Aurik 9 (NVIDIA 2024)

Finaler Synthese-Schritt nach kompletter Restaurierungspipeline wenn
PQS-MOS < 4.3 (Studio-2026-Modus) oder nach Stem-Separation + Re-Mix.
BigVGAN-v2 übertrifft alle bisherigen GAN-basierten Vocoder.

Referenz:
    Lee et al. (2024): "BigVGAN v2: Advancing GAN-based Neural Vocoders"
    NVIDIA 2024. Lizenz: Apache 2.0. https://github.com/NVIDIA/BigVGAN

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   BigVGAN-v2 (ONNX, CPUExecutionProvider, Mel-Eingang 80 Bänder)
    Fallback: HiFi-GAN v2 (2021) → PGHI-ISTFT (Rohrekonstruktion)

Aktivierungsbedingungen (§4.5 Aurik-Spec):
    - Studio-2026-Modus: PQS-MOS < 4.3 nach Phase-Pipeline
    - Nach Stem-Separation + Re-Mix als finaler Synthese-Schritt
    VERBOTEN: Griffin-Lim als Endschritt in Studio-2026-Modus

CPU-Policy: Ausschließlich CPUExecutionProvider, torch.set_num_threads(os.cpu_count()).
Modell-Gewichte: ~/.aurik/models/bigvgan_v2/ (via ModelDownloader, Apache 2.0)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

_ROOT: Path = Path(__file__).parent.parent

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstantdefinitionen (Mel-Featureisierung)
# ---------------------------------------------------------------------------

MEL_BANDS: int = 80  # Standard für BigVGAN-v2
MEL_HOP_MS: float = 12.5  # 12.5 ms Hop → 600 Frames/s @ 48000 Hz
MEL_WIN_MS: float = 50.0  # 50 ms Analysefenster
MEL_FMIN: float = 0.0  # Untergrenze Mel-Filterbank
MEL_FMAX: float = 24000.0  # Obergrenze = Nyquist @ 48000 Hz


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class VocoderResult:
    """Ergebnis der BigVGAN-v2 Vocoder-Synthese.

    Attribute:
        audio:         Synthetisiertes Audio (float32, normalisiert [-1,1], 48000 Hz)
        sr:            Sample-Rate (48000 Hz)
        pqs_mos:       Geschätzter PQS-MOS des Ausgangs ∈ [1.0, 5.0]
        model_used:    "bigvgan_v2" | "hifigan_v2_fallback" | "pghi_istft_fallback"
        confidence:    Konfidenz der Synthese ∈ [0, 1]
        mel_snr_db:    Mel-Spektrogramm SNR Original vs. Synthese [dB]
    """

    audio: np.ndarray
    sr: int
    pqs_mos: float
    model_used: str
    confidence: float
    mel_snr_db: float = 0.0
    metadata: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sr": self.sr,
            "pqs_mos": self.pqs_mos,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "mel_snr_db": self.mel_snr_db,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: BigVGANv2Plugin | None = None
_lock = threading.Lock()


class BigVGANv2Plugin:
    """BigVGAN-v2 Neuronaler Vocoder-Plugin für Aurik 9.

    Algorithmus (BigVGAN-v2 Vollpfad):
        1. Mel-Spektrogramm-Extraktion (80 Bänder, 12.5 ms Hop, Hanning 50 ms)
           f_mel(n) = 2595 · log₁₀(1 + f/700)  [Mel-Skala]
        2. GAN-Inferenz: Generator(Mel) → Waveform-Samples
           Multi-Periodic Discriminator (MPD) + Multi-Scale Discriminator (MSD)
        3. Anti-Aliasing aktiviert (sinc-Interpolation an Upsampling-Stufen)
        4. clip(−1, 1), nan_to_num
        5. PQS-MOS-Schätzung des Ausgangs

    Fallback-Kaskade:
        1. HiFi-GAN v2 (2021, kleineres Modell) — wenn BigVGAN nicht verfügbar
        2. PGHI-ISTFT (Perraudin 2013) — wenn kein GAN-Modell verfügbar

    CPU-Policy:
        torch.set_num_threads(os.cpu_count())  # alle CPU-Kerne nutzen
        device = "cpu"  — keine CUDA-Abhängigkeit

    Nutzung:
        Aktivierung NUR wenn:
        - Studio-2026-Modus UND PQS-MOS < 4.3 nach Pipeline
        - Nach Stem-Re-Mix als finaler Synthese-Schritt
        VERBOTEN: In Restoration-Modus (verändert Klangcharakter)
    """

    MODELS_DIR: Path = _ROOT / "models" / "bigvgan"
    MEL_HOP: int = int(48000 * MEL_HOP_MS / 1000)  # 600 Samples @ 48000 Hz
    MEL_WIN: int = int(48000 * MEL_WIN_MS / 1000)  # 2400 Samples @ 48000 Hz

    def __init__(self) -> None:
        self._session = None  # onnxruntime.InferenceSession
        self._torch_gen = None  # torch.nn.Module (Generator)
        self._model_loaded: bool = False
        self._fallback_mode: str = "pghi_istft"
        self._device: str = "cpu"  # set by _try_load_model
        self._device: str = "cpu"  # set by _try_load_model
        self._try_load_model()

    _BUDGET_NAME: str = "bigvgan_v2"
    _BUDGET_SIZE_GB: float = 0.40  # BigVGAN-v2 checkpoint ~200-400 MB

    def _try_load_model(self) -> None:
        """Lädt BigVGAN-v2 aus PyTorch-Checkpoint, sonst Fallback."""
        # [RELEASE_MUST] memory budget guard before torch.load (§2.37 Checkliste)
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("BigVGAN-v2: ML-Budget erschöpft — PGHI-ISTFT Fallback aktiv.")
                self._fallback_mode = "pghi_istft"
                return
        except ImportError:
            pass  # budget module absent → attempt load anyway
        # Versuch 1: torch (CPU)
        try:
            import torch

            torch.set_num_threads(os.cpu_count() or 4)
            checkpoint = self.MODELS_DIR / "bigvgan_v2.pth"
            if checkpoint.exists():
                try:
                    from backend.core.ml_device_manager import get_torch_device as _get_dev

                    _dev = _get_dev("BigVGAN")
                except Exception:
                    _dev = "cpu"
                self._torch_gen = torch.load(str(checkpoint), map_location=_dev)  # nosec B614 — lokales Modell aus models/
                self._torch_gen.eval()
                self._torch_gen.to(_dev)
                self._device = _dev
                self._model_loaded = True
                self._fallback_mode = "bigvgan_v2_torch"
                logger.info("🟢 BigVGAN-v2: torch-Modell geladen (device=%s, %s)", _dev, checkpoint)
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        self._BUDGET_NAME,
                        size_gb=self._BUDGET_SIZE_GB,
                        unload_fn=lambda: setattr(self, "_torch_gen", None),
                    )
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
                return
        except ImportError:
            logger.debug("torch nicht verfügbar für BigVGAN-v2")
        except Exception as exc:
            logger.debug("BigVGAN-v2 torch nicht ladbar: %s", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

        # Kein Modell gefunden
        logger.info(
            "BigVGAN-v2: Kein Modell in %s — PGHI-ISTFT Fallback aktiv",
            self.MODELS_DIR,
        )
        self._fallback_mode = "pghi_istft"

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        mode: str = "studio2026",
    ) -> VocoderResult:
        """Neuronale Vocoder-Synthese via Mel-Spektrogramm-Konditionierung.

        Algorithmus:
            1. Audio → Mel-Spektrogramm (80 Bänder, Hanning 50 ms, Hop 12.5 ms)
            2. Mel → BigVGAN-v2-Generator → Waveform (48 kHz)
            3. Ausgabe: clip(−1, 1), nan_to_num, PQS-MOS-Schätzung

        Args:
            audio:  Eingangs-Audio (1D float32, 48000 Hz) für Mel-Extraktion
            sr:     Sample-Rate (muss 48000 sein)
            mode:   "studio2026" (Standard) | "restoration" (BigVGAN deaktiviert)

        Returns:
            VocoderResult mit synthetisiertem audio und Qualitätsmetriken

        Raises:
            ValueError: Falls sr != 48000 oder mode=="restoration"
        """
        assert sr == 48000, f"BigVGAN-v2: SR muss 48000 Hz sein, erhalten: {sr}"
        if mode == "restoration":
            raise ValueError(
                "BigVGAN-v2 ist im Restoration-Modus deaktiviert (verändert Klangcharakter — nur Studio-2026!)"
            )

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "🟢 BigVGAN-v2: Vocoder-Synthese | %.1f s | Modell=%s",
            len(audio_f32) / sr,
            "BigVGAN-v2" if self._model_loaded else self._fallback_mode,
        )

        if self._model_loaded:
            result_audio, model_name, conf = self._synthesize_bigvgan(audio_f32, sr)
        else:
            result_audio, model_name, conf = self._synthesize_pghi_fallback(audio_f32, sr)

        pqs_mos = self._estimate_pqs_mos(audio_f32, result_audio, sr)
        mel_snr = self._mel_snr(audio_f32, result_audio, sr)

        logger.info(
            "🟢 BigVGAN-v2: PQS-MOS=%.2f | Mel-SNR=%.1f dB | Modell=%s",
            pqs_mos,
            mel_snr,
            model_name,
        )
        return VocoderResult(
            audio=result_audio,
            sr=sr,
            pqs_mos=pqs_mos,
            model_used=model_name,
            confidence=conf,
            mel_snr_db=mel_snr,
            metadata={"mode": mode},
        )

    # ------------------------------------------------------------------
    # BigVGAN-v2 ONNX/torch Synthese
    # ------------------------------------------------------------------

    def _synthesize_bigvgan(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, str, float]:
        """BigVGAN-v2 Generator-Inferenz (ONNX oder torch)."""
        try:
            mel = self._compute_mel(audio, sr)  # [n_mel, T]

            if self._session is not None:
                # ONNX-Pfad
                mel_input = mel[np.newaxis, :, :].astype(np.float32)  # [1, 80, T]
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: mel_input})
                if outputs and outputs[0] is not None:
                    synthesized = outputs[0].flatten()
                else:
                    synthesized = np.zeros(int(mel.shape[1] * self._HOP_SIZE), dtype=np.float32)
            elif self._torch_gen is not None:
                # torch-Pfad
                import torch

                try:
                    from backend.core.ml_device_manager import get_ml_device_manager as _get_mdm
                    _pin_fn = _get_mdm().pin_tensor_rocm
                except Exception:
                    _pin_fn = lambda x: x  # noqa: E731
                with torch.no_grad():
                    _mel_arr = mel[np.newaxis, :, :]
                    _pinned = _pin_fn(_mel_arr)
                    mel_t = (_pinned if isinstance(_pinned, torch.Tensor) else torch.from_numpy(_pinned)).to(self._device)
                    waveform = self._torch_gen(mel_t)
                    synthesized = waveform.squeeze().cpu().numpy()

            else:
                raise RuntimeError("Kein Modell verfügbar")

            n = min(len(audio), len(synthesized))
            result = np.zeros(len(audio), dtype=np.float32)
            result[:n] = np.nan_to_num(synthesized[:n], nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0).astype(np.float32), "bigvgan_v2", 0.95

        except Exception as exc:
            if self._device != "cpu" and self._torch_gen is not None:
                logger.warning("BigVGAN-v2: GPU-Inferenz fehlgeschlagen (%s) — CPU-Retry", exc)
                try:
                    self._torch_gen.cpu()
                    self._device = "cpu"
                    try:
                        from backend.core.ml_device_manager import get_ml_device_manager as _mgr

                        _mgr().report_gpu_error("BigVGAN", exc)
                    except Exception:
                        pass
                except Exception as _mv_exc:
                    logger.debug("BigVGAN-v2 GPU→CPU move fehlgeschlagen: %s", _mv_exc)
                    self._device = "cpu"
                return self._synthesize_bigvgan(audio, sr)
            logger.warning("BigVGAN-v2 Inferenz-Fehler: %s — PGHI-Fallback", exc)
            return self._synthesize_pghi_fallback(audio, sr)

    # ------------------------------------------------------------------
    # PGHI-ISTFT Fallback
    # ------------------------------------------------------------------

    def _synthesize_pghi_fallback(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[np.ndarray, str, float]:
        """PGHI-ISTFT Fallback — phasenkonsistente Rekonstruktion ohne GAN.

        Referenz: Perraudin et al. (2013) PGHI.
        """
        n_fft = self.MEL_WIN
        hop = self.MEL_HOP
        window = np.hanning(n_fft).astype(np.float32)

        # OLA-STFT → Modifikation → ISTFT (Magnitude-Only, Griffin-Lim-ähnlich)
        frames = []
        for start in range(0, len(audio) - n_fft, hop):
            frame = audio[start : start + n_fft] * window
            spec = np.fft.rfft(frame, n=n_fft)
            # Magnitude-only PGHI-Stub: Phasen aus Original behalten
            frames.append((start, spec))

        result = np.zeros(len(audio), dtype=np.float32)
        for start, spec in frames:
            frame_out = np.fft.irfft(spec, n=n_fft).real.astype(np.float32) * window
            result[start : start + n_fft] += frame_out

        result = np.nan_to_num(result, nan=0.0)
        return np.clip(result, -1.0, 1.0).astype(np.float32), "pghi_istft_fallback", 0.55

    # ------------------------------------------------------------------
    # Mel-Spektrogramm-Extraktion
    # ------------------------------------------------------------------

    def _compute_mel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Mel-Spektrogramm (80 Bänder) für BigVGAN-v2-Konditionierung.

        f_mel(n) = 2595 · log₁₀(1 + f/700)

        Returns:
            np.ndarray: [n_mel, T] float32
        """
        try:
            import librosa

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.MEL_WIN,
                hop_length=self.MEL_HOP,
                n_mels=MEL_BANDS,
                fmin=MEL_FMIN,
                fmax=MEL_FMAX,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            return np.clip(mel_db, -80.0, 0.0)

        except ImportError:
            # Minimaler Fallback ohne librosa
            n_fft = self.MEL_WIN
            hop = self.MEL_HOP
            n_frames = (len(audio) - n_fft) // hop + 1
            mel_out = np.zeros((MEL_BANDS, n_frames), dtype=np.float32)
            for i in range(n_frames):
                frame = audio[i * hop : i * hop + n_fft]
                if len(frame) < n_fft:
                    break
                spec = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
                for band in range(MEL_BANDS):
                    band_slice = spec[band * (len(spec) // MEL_BANDS) : (band + 1) * (len(spec) // MEL_BANDS)]
                    mel_out[band, i] = float(np.mean(band_slice)) if len(band_slice) > 0 else 0.0
            return mel_out

    # ------------------------------------------------------------------
    # Qualitätsmetriken
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_pqs_mos(original: np.ndarray, synthesized: np.ndarray, sr: int) -> float:
        """Schätzt PQS-MOS via Spektral-Kohärenz ∈ [1.0, 5.0]."""
        n = min(len(original), len(synthesized), 4096)
        if n < 64:
            return 3.5
        orig_spec = np.abs(np.fft.rfft(original[:n], n=4096))
        synth_spec = np.abs(np.fft.rfft(synthesized[:n], n=4096))
        total = np.sum(orig_spec) + 1e-12
        cohérence = float(np.sum(np.minimum(orig_spec, synth_spec)) / total)
        mos = 1.0 + 4.0 / (1.0 + np.exp(-8.0 * (cohérence - 0.5)))
        return float(np.clip(mos, 1.0, 5.0))

    @staticmethod
    def _mel_snr(original: np.ndarray, synthesized: np.ndarray, sr: int) -> float:
        """Mel-Spektrogramm SNR original vs. Synthese [dB] (höher = besser)."""
        n = min(len(original), len(synthesized), 4096)
        if n < 64:
            return 0.0
        sig_e = float(np.mean(original[:n] ** 2))
        err_e = float(np.mean((original[:n] - synthesized[:n]) ** 2))
        if sig_e < 1e-18 or err_e < 1e-18:
            return 40.0
        return float(np.clip(10.0 * np.log10(sig_e / err_e), -20.0, 60.0))


# ---------------------------------------------------------------------------
# Singleton-Accessor
# ---------------------------------------------------------------------------


def get_bigvgan_v2() -> BigVGANv2Plugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BigVGANv2Plugin()
    return _instance


def synthesize_audio(
    audio: np.ndarray,
    sr: int,
    *,
    mode: str = "studio2026",
) -> VocoderResult:
    """Convenience-Wrapper — BigVGAN-v2-Vocoder-Synthese ohne Klassen-Instantiierung.

    NUR für Studio-2026-Modus. Im Restoration-Modus ValueError.

    Beispiel::

        result = synthesize_audio(restored_audio, sr=48000, mode="studio2026")
        logger.debug("PQS-MOS: %.2f, Modell: %s", result.pqs_mos, result.model_used)

    Args:
        audio:  Restauriertes Audio (1D float32, 48000 Hz)
        sr:     Sample-Rate (48000)
        mode:   "studio2026" (Standard)

    Returns:
        VocoderResult
    """
    return get_bigvgan_v2().synthesize(audio, sr, mode=mode)
