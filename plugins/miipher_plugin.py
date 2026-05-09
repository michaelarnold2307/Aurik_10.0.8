"""
MIIPHER Plugin — Last-Resort-Entrauscher für SNR < 10 dB (v9.12.1)

§4.4 SOTA-Matrix 2026: MIIPHER (Zhang et al. 2023, Google) ist das SOTA-Modell
für extrem starke Rauschumgebungen (SNR < 10 dB), Vocal-Restaurierung von
stark degradiertem Gesangsmaterial. Basiert auf W2v-BERT als Conditioning.

Modell-Status: Stub — Modell noch nicht gebündelt.
Fallback-Kaskade: MIIPHER → DeepFilterNet v3.II (energy_bias=-6dB).

Aktivierung: Nur wenn DefectScanner `noise_snr_db < 10.0` UND
`panns_singing_confidence ≥ 0.35`.

§0h Invariante: Kein Output wird akzeptiert wenn artifact_freedom < 0.95.
§0j KI-Modell-Limitation: MIIPHER halluziniert potentiell Harmonics für
unbekannte Singstimmen → hallucination_guard.py nach Anwendung Pflicht.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Modell-Pfad (nach Integration in AppImage)
_MIIPHER_ONNX_PATH = None  # TODO: "models/miipher/miipher.onnx" nach Modell-Integration

# SNR-Schwellwert für MIIPHER-Aktivierung (dB)
MIIPHER_SNR_THRESHOLD_DB = 10.0

# Minimum PANNs Gesangskonfidenz für MIIPHER
MIIPHER_SINGING_CONFIDENCE_MIN = 0.35

# DeepFilterNet Fallback energy_bias bei Gesang (§0j)
_DFN_FALLBACK_ENERGY_BIAS_DB = -6.0

# Singleton
_instance: MiipherPlugin | None = None
_lock = threading.Lock()


class MiipherPlugin:
    """
    MIIPHER Last-Resort-Entrauscher für stark degradiertes Gesangsmaterial.

    Primary: MIIPHER ONNX (wenn Modell vorhanden).
    Fallback: DeepFilterNet v3.II mit Gesang-optimiertem energy_bias.

    Verwendung:
        plugin = get_miipher_plugin()
        if plugin.should_activate(noise_snr_db, panns_singing):
            result = plugin.enhance(audio, sr)
    """

    def __init__(self) -> None:
        self._model_loaded = False
        self._model_session = None
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Versucht MIIPHER-Modell zu laden. Stub wenn Modell nicht verfügbar."""
        if _MIIPHER_ONNX_PATH is None:
            logger.info(
                "MIIPHER: Modell nicht gebündelt (Stub-Modus) — "
                "DeepFilterNet v3.II wird als Fallback verwendet. "
                "§4.4 Roadmap: MIIPHER.onnx in models/miipher/ ablegen."
            )
            return

        try:
            import onnxruntime as ort  # type: ignore[import]

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            self._model_session = ort.InferenceSession(
                str(_MIIPHER_ONNX_PATH),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True
            logger.info("✅ MIIPHER ONNX geladen — §4.4 Last-Resort NR für SNR < 10 dB.")
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg

                _reg(
                    "MIIPHER",
                    size_gb=0.8,
                    unload_fn=lambda s=self: setattr(s, "_model_session", None) or setattr(s, "_model_loaded", False),
                )
            except Exception as _exc:
                logger.debug("PLM-Registrierung MIIPHER (non-critical): %s", _exc)
        except Exception as exc:
            logger.debug("MIIPHER ONNX nicht ladbar: %s — DeepFilterNet-Fallback aktiv.", exc)

    def should_activate(self, noise_snr_db: float, panns_singing: float) -> bool:
        """
        Prüft ob MIIPHER für dieses Material sinnvoll ist.

        Args:
            noise_snr_db:   Geschätzter SNR in dB (aus DefectScanner)
            panns_singing:  PANNs Gesangskonfidenz [0,1]

        Returns:
            True wenn MIIPHER (oder sein Fallback) aktiviert werden soll.
        """
        return noise_snr_db < MIIPHER_SNR_THRESHOLD_DB and panns_singing >= MIIPHER_SINGING_CONFIDENCE_MIN

    def enhance(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        noise_snr_db: float = 0.0,  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.float32]:
        """
        Entrauscht stark degradiertes Gesangsmaterial.

        Primary: MIIPHER ONNX (wenn geladen).
        Fallback: DeepFilterNet v3.II (energy_bias=-6 dB für Gesang).
        Last-Resort: Wiener-Filter als stets verfügbarer DSP-Fallback.

        §0h: artifact_freedom-Check NACH Anwendung in UV3 (nicht hier —
        Vermeidung von Doppel-Checks). Hier: nur NaN/Clip-Guard.

        Args:
            audio:         float32 Audio (mono/stereo, 48000 Hz)
            sr:            Abtastrate (muss 48000 Hz sein)
            noise_snr_db:  Geschätzter Input-SNR (für Logging)

        Returns:
            Prozessiertes float32 Audio, gleiche Form wie Input.
        """
        assert sr == 48000, f"MIIPHER: SR muss 48000 Hz sein, erhalten: {sr}"

        if self._model_loaded and self._model_session is not None:
            try:
                return self._enhance_miipher(audio, sr)
            except Exception as exc:
                logger.warning("MIIPHER-Modell Fehler: %s — DeepFilterNet-Fallback.", exc)

        # Fallback: DeepFilterNet v3.II
        try:
            return self._enhance_dfn_fallback(audio, sr)
        except Exception as exc:
            logger.warning("DeepFilterNet-Fallback Fehler: %s — Wiener-Filter-Fallback.", exc)

        # Last-Resort: DSP Wiener-Filter
        return self._enhance_wiener_fallback(audio, sr)

    def _enhance_miipher(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """
        MIIPHER ONNX-Inferenz.

        TODO: Nach Modell-Integration implementieren.
        W2v-BERT Conditioning benötigt 16kHz Input → Resampling → Inferenz → Upsample.
        """
        raise NotImplementedError("MIIPHER ONNX-Inferenz: TODO nach Modell-Integration")

    def _enhance_dfn_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """DeepFilterNet v3.II Fallback mit Gesang-optimiertem energy_bias (-6 dB)."""
        from plugins.deepfilternet_v3_ii_plugin import get_deepfilternet_plugin  # type: ignore[import]

        dfn = get_deepfilternet_plugin()
        result_raw = dfn.enhance(audio, sr=sr, energy_bias_db=_DFN_FALLBACK_ENERGY_BIAS_DB)
        if result_raw is not None and np.isfinite(np.asarray(result_raw)).all():
            logger.debug(
                "MIIPHER-Stub: DeepFilterNet-Fallback erfolgreich (energy_bias=%.1f dB)", _DFN_FALLBACK_ENERGY_BIAS_DB
            )
            out_f32: npt.NDArray[np.float32] = np.clip(np.asarray(result_raw, dtype=np.float32), -1.0, 1.0)
            return out_f32
        raise RuntimeError("DeepFilterNet-Fallback: ungültiges Ergebnis")

    def _enhance_wiener_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.float32]:
        """
        Einfacher spektraler Wiener-Filter als Last-Resort DSP-Fallback.

        Schätzt Rauschspektrum aus lautestem 10%-Segment (umgekehrt: Signal dominiert dort),
        dann klassisches Wiener-Spektral-Subtraktionsfilter.
        """
        mono: np.ndarray
        is_stereo = audio.ndim == 2
        if is_stereo:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                mono = np.mean(audio, axis=0).astype(np.float64)
                is_channels_first = True
            else:
                mono = np.mean(audio, axis=1).astype(np.float64)
                is_channels_first = False
        else:
            mono = audio.astype(np.float64)
            is_channels_first = False

        n_fft = 2048
        hop = 512
        window = np.hanning(n_fft)

        frames = []
        for i in range(0, len(mono) - n_fft, hop):
            frame = mono[i : i + n_fft] * window
            frames.append(np.fft.rfft(frame))

        if not frames:
            return audio

        spectra = np.array(frames)  # (T, F)
        mag = np.abs(spectra)

        # Schätze Rauschprofil aus lautestem 10% der Frames invertiert
        # (im lauten Signal sieht man das Rausch-Minimum besser)
        frame_rms = np.sqrt(np.mean(mag**2, axis=1))
        quiet_thresh = np.percentile(frame_rms, 20)
        quiet_mask = frame_rms <= quiet_thresh
        if np.any(quiet_mask):
            noise_est = np.percentile(mag[quiet_mask], 50, axis=0)
        else:
            noise_est = np.percentile(mag, 10, axis=0)

        # Wiener-Gain G(f) = max(0.1, 1 - noise_est / (mag + ε))
        gain = np.clip(1.0 - noise_est[None, :] / np.clip(mag, 1e-10, None), 0.10, 1.0)
        spectra_filtered = spectra * gain

        # iSTFT via OLA
        out = np.zeros(len(mono))
        norm = np.zeros(len(mono))
        for i, (frame_filt, i_start) in enumerate(zip(spectra_filtered, range(0, len(mono) - n_fft, hop))):
            frame_t = np.fft.irfft(frame_filt).real[:n_fft] * window
            out[i_start : i_start + n_fft] += frame_t
            norm[i_start : i_start + n_fft] += window**2

        norm = np.where(norm > 1e-8, norm, 1.0)
        out /= norm
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        logger.debug("MIIPHER-Stub: Wiener-Filter DSP-Fallback angewendet")

        if is_stereo:
            # Signal-Ratio Stereo-Rekonstruktion
            if is_channels_first:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[0]):
                    result[ch] = np.clip(audio[ch] * ratio, -1.0, 1.0)
            else:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[1]):
                    result[:, ch] = np.clip(audio[:, ch] * ratio, -1.0, 1.0)
            return result
        out_f32_w: npt.NDArray[np.float32] = out.astype(np.float32)
        return out_f32_w


def get_miipher_plugin() -> MiipherPlugin:
    """Thread-safe Singleton (Double-Checked Locking, §3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MiipherPlugin()
    return _instance
