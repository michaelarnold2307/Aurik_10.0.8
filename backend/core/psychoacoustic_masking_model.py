"""
backend/core/psychoacoustic_masking_model.py — Psychoakustisches Masking-Modell (Aurik 9 §4.5)
===========================================================================
ISO 11172-3 Simultane + Temporale Maskierung als OMLSA-Gain-Modifier.
Stille-Segmente (<= SILENCE_DB) erhalten Gain ≤ SILENCE_GAIN_MAX = 0.30.
SR-Invariante: assert sr == 48000 in compute_threshold().
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Öffentliche Konstanten (§4.5, §8.2)
# ---------------------------------------------------------------------------
SILENCE_DB: float = -40.0  # Schwelle Stille (≤ -30 dBFS, §8.2)
GAIN_FLOOR: float = 0.10  # Mindest-Gain (G_floor, §4.5)
N_BARK: int = 24  # Anzahl Bark-Bänder
SILENCE_GAIN_MAX: float = 0.30  # Maximaler Gain in Stille-Frames
POST_MASK_MS: float = 100.0  # Temporale Post-Masking-Dauer in ms

# ISO 11172-3 Masking-Slopes: alpha_b ≈ 14.5 + b [dB/Bark] — monoton steigend
_MASKING_SLOPE_DB: list[float] = [14.5 + float(b) for b in range(N_BARK)]


# ---------------------------------------------------------------------------
# MaskingResult
# ---------------------------------------------------------------------------
@dataclass
class MaskingResult:
    """Rückgabe-Container von PsychoacousticMaskingModel.compute_threshold()."""

    gain_modifier: np.ndarray  # [n_frames × 24] float32 ∈ [GAIN_FLOOR, 1.0]
    masking_threshold: np.ndarray  # [n_frames × 24] float32 ≥ 0
    silence_frames: np.ndarray  # [n_frames] bool — True = Stille (<= SILENCE_DB)
    post_mask_frames: np.ndarray  # [n_frames] bool — True = temporale Maskierung aktiv
    n_frames: int
    n_bark_bands: int = N_BARK

    # Rückwärtskomp. Alias
    @property
    def gain_mask(self) -> np.ndarray:
        return self.gain_modifier

    @property
    def threshold_bark(self) -> np.ndarray:
        return self.masking_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_bark_bands": self.n_bark_bands,
            "silence_fraction": float(np.mean(self.silence_frames)),
            "post_mask_fraction": float(np.mean(self.post_mask_frames)),
            "gain_modifier_mean": float(np.mean(self.gain_modifier)),
        }


# ---------------------------------------------------------------------------
# PsychoacousticMaskingModel
# ---------------------------------------------------------------------------
class PsychoacousticMaskingModel:
    """Psychoakustisches Masking-Modell (ISO 11172-3, §4.5).

    Gleichzeitige Maskierung: pro Bark-Band b:
        MT_b = Signal_b · 10^(-alpha_b / 10)
    Stille-Segmente: Gain ≤ SILENCE_GAIN_MAX = 0.30
    Perkussive Transienten: Post-Masking 50–200 ms
    """

    def __init__(self, g_floor: float = GAIN_FLOOR) -> None:
        self.g_floor = float(np.clip(g_floor, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Interne Hilfsmethode
    # ------------------------------------------------------------------
    def _build_bark_bins(self, sr: int) -> np.ndarray:
        """Gibt N_BARK+1 Kanten der Bark-Bänder als float32-Array zurück.

        24 Bänder = 25 Kantenpositionen in Hz.
        Basiert auf Zwicker & Fastl (1990): BARK_EDGES_HZ.
        """
        bark_edges_hz = np.array(
            [
                20,
                100,
                200,
                300,
                400,
                510,
                630,
                770,
                920,
                1080,
                1270,
                1480,
                1720,
                2000,
                2320,
                2700,
                3150,
                3700,
                4400,
                5300,
                6400,
                7700,
                9500,
                12000,
                15500,
            ],
            dtype=np.float32,
        )
        return bark_edges_hz  # 25 Elemente = N_BARK + 1

    # ------------------------------------------------------------------
    def compute_threshold(self, audio: np.ndarray, sr: int) -> MaskingResult:
        """Berechnet Masking-Schwelle und Gain-Modifier.

        Args:
            audio: float32/64 mono oder stereo
            sr:    Sample-Rate — MUSS 48000 sein (§6.6 SR-Invariante)

        Returns:
            MaskingResult mit gain_modifier, masking_threshold,
            silence_frames, post_mask_frames
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr

        if mono.size == 0:
            empty = np.zeros((1, N_BARK), dtype=np.float32)
            return MaskingResult(
                gain_modifier=np.full((1, N_BARK), self.g_floor, dtype=np.float32),
                masking_threshold=empty,
                silence_frames=np.array([True], dtype=bool),
                post_mask_frames=np.array([False], dtype=bool),
                n_frames=1,
            )

        frame = max(256, sr // 100)  # 480 samples @ 48 kHz
        hop = frame // 2
        n_frames = max(1, (mono.size - frame) // hop + 1)

        gain_mod = np.zeros((n_frames, N_BARK), dtype=np.float32)
        mask_thr = np.zeros((n_frames, N_BARK), dtype=np.float32)
        sil_frms = np.zeros(n_frames, dtype=bool)
        post_frms = np.zeros(n_frames, dtype=bool)

        bark_edges = self._build_bark_bins(sr)
        fft_freqs = np.fft.rfftfreq(frame, d=1.0 / sr).astype(np.float32)

        prev_loud_frame = -999
        post_mask_frames_count = max(1, int(POST_MASK_MS * 1e-3 * sr / hop))

        for i in range(n_frames):
            s = i * hop
            e = min(mono.size, s + frame)
            seg = mono[s:e]
            if seg.size < 2:
                continue

            # Segment-Energie
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
            rms_db = 20.0 * math.log10(rms + 1e-12)

            is_silent = rms_db <= SILENCE_DB
            sil_frms[i] = is_silent
            if not is_silent:
                prev_loud_frame = i

            # Temporale Post-Masking
            in_post_mask = is_silent and prev_loud_frame >= 0 and (i - prev_loud_frame) <= post_mask_frames_count
            post_frms[i] = in_post_mask

            # Spektrale Analyse fürs Masking
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            seg = np.clip(seg, -1.0, 1.0)
            spec = np.abs(np.fft.rfft(seg, n=frame)).astype(np.float32)
            spec_sq = spec**2

            # Bark-Band-Energie
            band_energy = np.zeros(N_BARK, dtype=np.float32)
            for b in range(N_BARK):
                low_hz = float(bark_edges[b])
                high_hz = float(bark_edges[b + 1])
                mask_b = (fft_freqs >= low_hz) & (fft_freqs < high_hz)
                if mask_b.any():
                    band_energy[b] = float(np.mean(spec_sq[mask_b]))

            # Masking-Schwelle pro Band: ISO 11172-3 Typ 1
            for b in range(N_BARK):
                slope_db = _MASKING_SLOPE_DB[b]
                mask_val = band_energy[b] * 10.0 ** (-slope_db / 10.0)
                mask_thr[i, b] = max(0.0, float(mask_val))

            # Gain-Modifier: vollständige Maskierung → volle Stärke
            total_e = float(np.sum(band_energy) + 1e-12)
            for b in range(N_BARK):
                rel = band_energy[b] / total_e
                if is_silent:
                    # Stille: max. SILENCE_GAIN_MAX = 0.30
                    g = SILENCE_GAIN_MAX * rel * N_BARK
                    gain_mod[i, b] = float(np.clip(g, self.g_floor, SILENCE_GAIN_MAX))
                elif in_post_mask:
                    # Post-Masking: bis zu voller Stärke erlaubt
                    g = 0.5 + 0.5 * rel * N_BARK
                    gain_mod[i, b] = float(np.clip(g, self.g_floor, 1.0))
                else:
                    # Normale Maskierung
                    g = 0.3 + 0.7 * rel * N_BARK
                    gain_mod[i, b] = float(np.clip(g, self.g_floor, 1.0))

        gain_mod = np.nan_to_num(gain_mod)
        mask_thr = np.nan_to_num(mask_thr)
        np.clip(gain_mod, self.g_floor, 1.0, out=gain_mod)
        np.clip(mask_thr, 0.0, None, out=mask_thr)

        return MaskingResult(
            gain_modifier=gain_mod,
            masking_threshold=mask_thr,
            silence_frames=sil_frms,
            post_mask_frames=post_frms,
            n_frames=n_frames,
        )

    def apply_adaptive_gain(
        self,
        gain_mask: np.ndarray,
        masking_result: "MaskingResult | np.ndarray",
    ) -> np.ndarray:
        """Skaliert gain_mask mit dem Masking-Modifier.

        Args:
            gain_mask:       [n_frames × 24] Eingangs-Gain
            masking_result:  MaskingResult oder np.ndarray (Rückwärtskomp.)

        Returns:
            Gain-Maske geclippt auf [G_floor, 1.0], NaN-frei
        """
        g = np.nan_to_num(np.asarray(gain_mask, dtype=np.float32))
        if g.ndim != 2:
            return np.clip(g, self.g_floor, 1.0)

        # MaskingResult oder raw ndarray als Masking-Modifier
        if isinstance(masking_result, MaskingResult):
            t = masking_result.gain_modifier
        else:
            t = np.nan_to_num(np.asarray(masking_result, dtype=np.float32))

        rows = min(g.shape[0], t.shape[0]) if t.ndim >= 1 else g.shape[0]
        cols = min(g.shape[1], t.shape[1]) if t.ndim == 2 else g.shape[1]
        out = g.copy()
        if t.ndim == 2 and rows > 0 and cols > 0:
            scale = np.clip(t[:rows, :cols], 0.3, 1.0)
            out[:rows, :cols] *= scale
        return np.clip(np.nan_to_num(out), self.g_floor, 1.0)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: PsychoacousticMaskingModel | None = None
_lock = threading.Lock()


def get_masking_model() -> PsychoacousticMaskingModel:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PsychoacousticMaskingModel()
    return _instance


# Alias für Rückwärtskompatibilität
def get_psychoacoustic_masking_model() -> PsychoacousticMaskingModel:
    return get_masking_model()


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------
def compute_masking_threshold(audio: np.ndarray, sr: int) -> MaskingResult:
    """Convenience-Wrapper: Masking-Schwelle berechnen."""
    return get_masking_model().compute_threshold(audio, sr)


__all__ = [
    "PsychoacousticMaskingModel",
    "MaskingResult",
    "get_masking_model",
    "get_psychoacoustic_masking_model",
    "compute_masking_threshold",
    "SILENCE_DB",
    "GAIN_FLOOR",
    "N_BARK",
    "SILENCE_GAIN_MAX",
    "_MASKING_SLOPE_DB",
]
