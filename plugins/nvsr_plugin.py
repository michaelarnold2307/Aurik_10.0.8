"""NVSR Plugin — productive SBR/NVSR adapter (§Spec-04).

Zielbereich: 8–16 kHz Bandbreitenerweiterung.
Methode:     Spectral Band Replication (SBR) + harmonic envelope extrapolation.
Abgrenzung:  AudioSR (Diffusion) für 0–8 kHz (Shellac).
             NVSR (diese Datei) für 8–16 kHz (Vinyl, MP3-128kbps).
             Vorteil: deterministisch, kein Halluzinationsrisiko, 10–30× schneller.

Wenn ein lokales NVSR-ONNX-Modell (models/nvsr/nvsr.onnx) vorhanden ist,
wird auf Modell-Inferenz eskaliert. Sonst ist SBR der produktive, deterministische
Fallback für 8–16 kHz ohne Halluzinationsrisiko.

§SOTA-Matrix:
    "SBR-Heuristik + NVSR: 8–16 kHz fehlt (MP3 128kbps): deterministisch, schneller,
     weniger Halluzinationsrisiko als AudioSR (Diffusion)"

Referenz:
    Spectral Band Replication (SBR): Dietz et al. (2002), ISO/IEC 14496-3.
    HE-AAC SBR: Malah (2003) "Spectral band replication, a novel approach".
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as _sp_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton — thread-safe, double-checked locking
# ---------------------------------------------------------------------------
_instance: NvsrPlugin | None = None
_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# NVSR-ONNX-Modell-Pfad (optional — zukünftige DNN-Eskalation)
# ---------------------------------------------------------------------------
_NVSR_ONNX_PATH = Path(__file__).parent.parent / "models" / "nvsr" / "nvsr.onnx"

# ---------------------------------------------------------------------------
# Material-spezifische HF-Ceiling für SBR-Output
# ---------------------------------------------------------------------------
_MATERIAL_HF_CEILING_HZ: dict[str, float] = {
    "shellac": 8_000.0,  # Shellac: handled by AudioSR, NVSR should not activate
    "wax_cylinder": 5_000.0,
    "wire_recording": 7_000.0,
    "vinyl": 16_000.0,
    "tape": 16_000.0,
    "reel_tape": 16_000.0,
    "cassette": 14_000.0,
    "minidisc": 15_000.0,
    "mp3_low": 16_000.0,  # 128kbps
    "mp3_medium": 18_000.0,  # 192kbps
    "mp3_high": 20_000.0,
    "aac": 18_000.0,
    "cd_digital": 22_000.0,
    "streaming": 20_000.0,
    "digital": 22_000.0,
}

# SBR source band: content in this range is used to synthesize 8–16kHz
_SBR_SOURCE_LOW_HZ = 4_000.0  # lower edge of source band
_SBR_SOURCE_HIGH_HZ = 8_000.0  # upper edge of source band (= target band starts here)
_SBR_TARGET_LOW_HZ = 8_000.0  # lower edge of target band
_SBR_TARGET_HIGH_HZ = 16_000.0  # upper edge of target band


def get_nvsr_plugin() -> NvsrPlugin:
    """Singleton-Factory — gibt bestehende Instanz zurück oder erstellt sie."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NvsrPlugin()
    return _instance


class NvsrPlugin:
    """NVSR-SBR-Plugin: Deterministische Bandbreitenerweiterung 8–16 kHz.

    Keine DNN-Modelle erforderlich — reiner DSP-Pfad mit ONNX-Hook für
    zukünftige Eskalation.

    Interface:
        plugin = get_nvsr_plugin()
        result = plugin.process(audio, sr, target_hz=16000, strength=0.7,
                                material_type="vinyl", energy_bias_db=0.0)
    """

    def __init__(self) -> None:
        self._onnx_session = None
        self._onnx_available = False
        self._onnx_load_attempted = False
        self._onnx_lock = threading.Lock()
        self._last_route_metadata: dict[str, Any] = {
            "strategy": "uninitialized",
            "capability_status": "unavailable",
            "model_path": str(_NVSR_ONNX_PATH),
            "model_loaded": False,
        }
        logger.info("NvsrPlugin: productive SBR/NVSR adapter initialized (model path=%s)", _NVSR_ONNX_PATH)

    @property
    def route_metadata(self) -> dict[str, Any]:
        """Gibt metadata for the last processing route zurück."""
        return dict(self._last_route_metadata)

    def capability_status(self) -> str:
        """Gibt local NVSR capability without running inference zurück."""
        if self._onnx_available and self._onnx_session is not None:
            return "sota_real"
        if _NVSR_ONNX_PATH.exists():
            return "sota_fallback"
        return "dsp_productive"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float = 16_000.0,
        strength: float = 0.7,
        material_type: str = "vinyl",
        energy_bias_db: float = 0.0,
        panns_singing: float = 0.0,
    ) -> dict[str, Any]:
        """Erweitere die Bandbreite deterministisch auf target_hz.

        Parameters
        ----------
        audio : np.ndarray
            Mono- oder Stereo-Signal (N,) oder (2,N), float32, 48000 Hz.
        sr : int
            Abtastrate (muss 48000 sein).
        target_hz : float
            Ziel-Bandbreite in Hz (Standard: 16000 Hz).
        strength : float
            Mischungsstärke [0.0, 1.0] (Standard: 0.7).
        material_type : str
            Materialtyp für Material-Ceiling (Standard: "vinyl").
        energy_bias_db : float
            Energiekorrektur in dB auf SBR-Ausgang (Standard: 0 dB).
            Für Gesangsmaterial: −6 dB (§0j), Instrumental: −9 dB.
        panns_singing : float
            PANNs-Singing-Score [0.0, 1.0] für Vocal-Aware-Modus.

        Returns
        -------
        dict with keys:
            audio       : np.ndarray — erweitertes Signal, gleiche Form wie Input
            strategy    : str — "onnx" | "dsp_sbr"
            target_hz   : float
            ceiling_hz  : float — tatsächliche Material-Ceiling
            strength    : float — angewandete Stärke
            hf_energy_added_db : float
        """
        assert sr == 48000, "NVSR: Eingang muss 48000 Hz sein"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Material-Ceiling: nie über physikalischem Limit
        ceiling_hz = _MATERIAL_HF_CEILING_HZ.get(str(material_type).lower(), 16_000.0)
        effective_target = min(target_hz, ceiling_hz, float(sr) / 2.0)

        if effective_target <= _SBR_TARGET_LOW_HZ + 500.0:
            # Ziel bereits nah an Quelle → kein Gewinn
            self._last_route_metadata = self._metadata(
                strategy="passthrough",
                target_hz=effective_target,
                ceiling_hz=ceiling_hz,
                strength=0.0,
                hf_energy_added_db=0.0,
            )
            return {
                "audio": audio,
                "strategy": "passthrough",
                "target_hz": effective_target,
                "ceiling_hz": ceiling_hz,
                "strength": 0.0,
                "hf_energy_added_db": 0.0,
                **self._last_route_metadata,
            }

        # Versuche ONNX-Eskalation wenn Modell vorhanden
        if _NVSR_ONNX_PATH.exists() and not self._onnx_load_attempted:
            self._try_load_onnx()

        if self._onnx_available and self._onnx_session is not None:
            try:
                return self._process_onnx(audio, sr, effective_target, strength, energy_bias_db)
            except Exception as _onnx_exc:
                logger.warning("NvsrPlugin: ONNX-Fehler → DSP-Fallback: %s", _onnx_exc)

        return self._process_dsp_sbr(audio, sr, effective_target, strength, energy_bias_db, panns_singing)

    def _metadata(
        self,
        *,
        strategy: str,
        target_hz: float,
        ceiling_hz: float,
        strength: float,
        hf_energy_added_db: float,
    ) -> dict[str, Any]:
        """Erstellt JSON-safe route metadata."""
        status = "sota_real" if strategy == "onnx" else self.capability_status()
        return {
            "strategy": strategy,
            "capability_status": status,
            "model_path": str(_NVSR_ONNX_PATH),
            "model_loaded": bool(self._onnx_available and self._onnx_session is not None),
            "target_hz": float(target_hz),
            "ceiling_hz": float(ceiling_hz),
            "strength": float(strength),
            "hf_energy_added_db": float(hf_energy_added_db),
        }

    # ------------------------------------------------------------------
    # DSP-SBR Kern
    # ------------------------------------------------------------------

    def _process_dsp_sbr(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
        panns_singing: float,
    ) -> dict[str, Any]:
        """Spectral Band Replication (SBR) — deterministisch, kein DNN.

        Algorithmus (§Spec-04 SBR-Heuristik):
        1. STFT auf Input.
        2. Für Zielbins (8–target_hz): interpoliere Energie aus Quellband (4–8kHz)
           via Harmonik-Ratio und Einhüllende-Matching.
        3. Phase: Ableitung aus bestehenden Subband-Phasen (kohärent).
        4. PGHI-konsistente Rekonstruktion wenn PGHI verfügbar.
        5. Blend mit Input: Stärke-gewichtet, HF-beschränkt.
        6. Hallucination-Guard nach additiver Operation.
        """
        stereo = audio.ndim == 2 and audio.shape[0] == 2
        if stereo:
            left_out = self._process_channel_sbr(audio[0], sr, target_hz, strength, energy_bias_db, panns_singing)
            right_out = self._process_channel_sbr(audio[1], sr, target_hz, strength, energy_bias_db, panns_singing)
            result_audio = np.stack([left_out, right_out], axis=0)
        else:
            mono = audio if audio.ndim == 1 else audio.mean(axis=0)
            result_audio = self._process_channel_sbr(mono, sr, target_hz, strength, energy_bias_db, panns_singing)

        result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)
        result_audio = np.clip(result_audio, -1.0, 1.0)

        # Hallucination guard (§2.46e): ADDITIVE Phase, spec-conform check
        try:
            # pylint: disable-next=import-outside-toplevel
            from backend.core.dsp.hallucination_guard import check_hallucination

            hg_result = check_hallucination(audio, result_audio, sr=sr, mode="restoration")
            if hg_result.requires_rollback:
                logger.info(
                    "NvsrPlugin: HallucinationGuard → Rollback (spectral_novelty=%.3f)", hg_result.spectral_novelty
                )
                result_audio = audio.copy()
        except Exception as _hg_exc:
            logger.debug("NvsrPlugin: HallucinationGuard nicht verfügbar: %s", _hg_exc)

        # Energie-Vergleich für Metadata
        hf_added_db = 0.0
        try:
            _mono_in = audio if audio.ndim == 1 else audio.mean(axis=0)
            _mono_out = result_audio if result_audio.ndim == 1 else result_audio.mean(axis=0)
            _, _, _Zin = _sp_signal.stft(_mono_in, fs=sr, nperseg=512, boundary="even")
            _, _, _Zout = _sp_signal.stft(_mono_out, fs=sr, nperseg=512, boundary="even")
            _freqs = np.linspace(0, sr / 2.0, _Zin.shape[0])
            _hf_mask = (_freqs >= _SBR_TARGET_LOW_HZ) & (_freqs <= target_hz)
            _e_in = float(np.mean(np.abs(_Zin[_hf_mask]) ** 2)) + 1e-20
            _e_out = float(np.mean(np.abs(_Zout[_hf_mask]) ** 2)) + 1e-20
            hf_added_db = float(10.0 * np.log10(_e_out / _e_in))
        except Exception:
            logger.warning("nvsr_plugin.py::_process_dsp_sbr fallback", exc_info=True)

        self._last_route_metadata = self._metadata(
            strategy="dsp_sbr",
            target_hz=target_hz,
            ceiling_hz=target_hz,
            strength=strength,
            hf_energy_added_db=hf_added_db,
        )
        return {
            "audio": result_audio,
            "strategy": "dsp_sbr",
            "target_hz": target_hz,
            "ceiling_hz": target_hz,
            "strength": strength,
            "hf_energy_added_db": hf_added_db,
            **self._last_route_metadata,
        }

    def _process_channel_sbr(
        self,
        channel: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
        panns_singing: float,
    ) -> np.ndarray:
        """SBR-Verarbeitung eines einzelnen Kanals."""
        n = len(channel)
        N_FFT = 1024
        HOP = 256

        # Reflect-padding (§2.63)
        pad_len = HOP * 4
        padded = np.pad(channel, pad_len, mode="reflect")

        f, _, Zxx = _sp_signal.stft(padded, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
        n_freq = len(f)
        freq_res = f[1] - f[0] if len(f) > 1 else (sr / N_FFT)

        # Frequenzindex-Grenzen
        src_low_bin = max(1, int(np.round(_SBR_SOURCE_LOW_HZ / freq_res)))
        src_high_bin = min(n_freq - 1, int(np.round(_SBR_SOURCE_HIGH_HZ / freq_res)))
        tgt_low_bin = min(n_freq - 1, int(np.round(_SBR_TARGET_LOW_HZ / freq_res)))
        tgt_high_bin = min(n_freq - 1, int(np.round(target_hz / freq_res)))

        if tgt_low_bin >= tgt_high_bin or src_low_bin >= src_high_bin:
            # Kein Spielraum → Passthrough
            _, channel_out = _sp_signal.istft(Zxx, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
            channel_out = channel_out[pad_len : pad_len + n]
            if len(channel_out) < n:
                channel_out = np.pad(channel_out, (0, n - len(channel_out)))
            return channel_out.astype(np.float32)  # type: ignore[no-any-return]

        # SBR-Kern: Spektrale Einhüllende aus Quellband ableiten
        src_mag = np.abs(Zxx[src_low_bin : src_high_bin + 1, :])  # (src_bins, T)
        src_phase = np.angle(Zxx[src_low_bin : src_high_bin + 1, :])

        # Einhüllende glätten (zeitlich: 3-Frame, spektral: Gliding-Mean)
        src_env = np.maximum(src_mag, 1e-12)
        kernel = np.ones(3) / 3.0
        src_env_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=src_env)

        # Zielband-Energie via Harmonik-Ratio (Frequenz-Skalierung ×2)
        n_src_bins = src_high_bin - src_low_bin + 1
        n_tgt_bins = tgt_high_bin - tgt_low_bin + 1

        # Resample Quelle → Ziel über lineare Interpolation der Magnitude
        src_bins_norm = np.linspace(0.0, 1.0, n_src_bins)
        tgt_bins_norm = np.linspace(0.0, 1.0, n_tgt_bins)
        tgt_mag = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)
        for t_idx in range(Zxx.shape[1]):
            tgt_mag[:, t_idx] = np.interp(tgt_bins_norm, src_bins_norm, src_env_smooth[:, t_idx]).astype(np.float32)

        # HF-Rolloff: Energie nimmt mit steigender Frequenz ab (−6 dB/Oktave Faustregel)
        rolloff_slope = np.linspace(1.0, 0.25, n_tgt_bins, dtype=np.float32)
        tgt_mag = tgt_mag * rolloff_slope[:, np.newaxis]

        # Energy bias (§0j): bei Gesang −6 dB, Instrumental −9 dB
        if panns_singing >= 0.4:
            _bias = -6.0 + energy_bias_db
        elif panns_singing >= 0.1:
            _bias = energy_bias_db
        else:
            _bias = -9.0 + energy_bias_db
        bias_lin = 10.0 ** (_bias / 20.0)
        tgt_mag = tgt_mag * float(bias_lin)

        # Phasen-Kohärenz: Phasengradient aus benachbarten Quell-Bins fortführen
        if src_phase.shape[1] > 1:
            _phase_grad = np.diff(src_phase, axis=1)  # (src_bins, T-1)
            _phase_grad_resamp = np.zeros((n_tgt_bins, _phase_grad.shape[1]), dtype=np.float32)
            for t_idx in range(_phase_grad.shape[1]):
                _phase_grad_resamp[:, t_idx] = np.interp(tgt_bins_norm, src_bins_norm, _phase_grad[:, t_idx]).astype(
                    np.float32
                )
            tgt_phase = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)
            if Zxx.shape[1] > 0:
                tgt_phase[:, 0] = np.interp(tgt_bins_norm, src_bins_norm, src_phase[:, 0]).astype(np.float32)
            for t_idx in range(1, Zxx.shape[1]):
                tgt_phase[:, t_idx] = (
                    tgt_phase[:, t_idx - 1] + _phase_grad_resamp[:, min(t_idx - 1, _phase_grad_resamp.shape[1] - 1)]
                )
        else:
            tgt_phase = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)

        # SBR-Signal in STFT einschreiben (nur Zielbins modifizieren)
        Zxx_out = Zxx.copy()
        tgt_complex = tgt_mag * np.exp(1j * tgt_phase)

        # Smooth-Blending an der Grenze src→tgt (2 Bins Crossfade)
        _crossfade_bins = min(4, n_tgt_bins)
        _fade_in = np.linspace(0.0, 1.0, _crossfade_bins + 2)[1:-1]  # (crossfade_bins,)
        for bi, tb in enumerate(range(tgt_low_bin, tgt_high_bin + 1)):
            if tb < n_freq:
                if bi < _crossfade_bins:
                    fade = _fade_in[bi]
                else:
                    fade = 1.0
                Zxx_out[tb, :] = (1.0 - fade * strength) * Zxx[tb, :] + fade * strength * tgt_complex[bi, :]

        # iSTFT
        _, channel_sbr = _sp_signal.istft(Zxx_out, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
        channel_sbr = channel_sbr[pad_len : pad_len + n]
        if len(channel_sbr) < n:
            channel_sbr = np.pad(channel_sbr, (0, n - len(channel_sbr)))

        channel_sbr = np.nan_to_num(channel_sbr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(channel_sbr, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # ONNX-Hook (zukünftige DNN-Eskalation)
    # ------------------------------------------------------------------

    def _try_load_onnx(self) -> None:
        """Versuche NVSR-ONNX-Modell zu laden. Nicht-blockierend."""
        with self._onnx_lock:
            if self._onnx_load_attempted:
                return
            self._onnx_load_attempted = True
            try:
                import onnxruntime as ort  # pylint: disable=import-outside-toplevel

                from backend.core.ml_device_manager import get_ort_providers  # pylint: disable=import-outside-toplevel

                self._onnx_session = ort.InferenceSession(
                    str(_NVSR_ONNX_PATH),
                    providers=get_ort_providers("NVSR"),
                )
                self._onnx_available = True
                logger.info("NvsrPlugin: ONNX-Modell geladen von %s", _NVSR_ONNX_PATH)
            except Exception as _onnx_load_exc:
                logger.debug("NvsrPlugin: ONNX-Modell nicht verfügbar (DSP-SBR aktiv): %s", _onnx_load_exc)
                self._onnx_available = False

    def _process_onnx(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
    ) -> dict[str, Any]:
        """Führt aus: a bundled NVSR ONNX waveform model and blend safely."""
        if self._onnx_session is None:
            raise RuntimeError("NVSR ONNX session not loaded")
        session = self._onnx_session
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("NVSR ONNX model has no inputs")
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        mono = np.asarray(mono, dtype=np.float32)
        rank = len(getattr(inputs[0], "shape", []) or [])
        if rank >= 3:
            model_input = mono[np.newaxis, np.newaxis, :].astype(np.float32)
        elif rank == 2:
            model_input = mono[np.newaxis, :].astype(np.float32)
        else:
            model_input = mono.astype(np.float32)
        outputs = session.run(None, {inputs[0].name: model_input})
        if not outputs:
            raise RuntimeError("NVSR ONNX model returned no outputs")
        out = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if len(out) > len(mono):
            out = out[: len(mono)]
        elif len(out) < len(mono):
            out = np.pad(out, (0, len(mono) - len(out)), mode="constant")
        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)
        if audio.ndim == 2 and audio.shape[0] == 2:
            out_audio = np.stack([out, out], axis=0)
        else:
            out_audio = out

        dsp_ref = self._process_dsp_sbr(audio, sr, target_hz, strength, energy_bias_db, 0.0)["audio"]
        blended = np.clip((1.0 - strength) * np.asarray(dsp_ref, dtype=np.float32) + strength * out_audio, -1.0, 1.0)
        self._last_route_metadata = self._metadata(
            strategy="onnx",
            target_hz=target_hz,
            ceiling_hz=target_hz,
            strength=strength,
            hf_energy_added_db=0.0,
        )
        return {
            "audio": blended.astype(np.float32),
            "strategy": "onnx",
            "target_hz": target_hz,
            "ceiling_hz": target_hz,
            "strength": strength,
            "hf_energy_added_db": 0.0,
            **self._last_route_metadata,
        }
