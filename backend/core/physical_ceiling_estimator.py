"""
backend/core/physical_ceiling_estimator.py
Aurik 9 -- Spec §2.33: PhysicalCeilingEstimator

Schaetzt informationstheoretische Qualitaets-Obergrenzen
aus Quell-SNR und Bandbreite.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

HEADROOM_THRESHOLD: float = 0.03  # < 3 % -> keine weiteren Iterationen


@dataclass
class PhysicalCeilingResult:
    """Spec §2.33: Informationstheoretische Qualitaets-Obergrenzen."""

    ceiling: Dict[str, float]
    snr_profile_db: np.ndarray
    effective_bandwidth_hz: float
    headroom_per_goal: Dict[str, float]
    further_optimization_worthwhile: bool

    def as_dict(self) -> Dict[str, Any]:
        """Serialisierung fuer Logging und API-Antworten (§2.22, §3.6)."""
        return {
            "ceiling": dict(self.ceiling),
            "effective_bandwidth_hz": float(self.effective_bandwidth_hz),
            "further_optimization_worthwhile": bool(self.further_optimization_worthwhile),
            "headroom_per_goal": dict(self.headroom_per_goal),
            "snr_profile_db": self.snr_profile_db.tolist(),
        }


class PhysicalCeilingEstimator:
    """Spec §2.33: Schaetzt informationstheoretische Qualitaetsgrenzen.

    Algorithmus:
        1. SNR pro Bark-Band (IMCRA-Minima-Schaetzung, 24 Baender)
        2. Effektive Bandbreite: hoechstes Bark-Band mit SNR >= -10 dB
        3. Musical-Goal-Ceiling-Mapping
        4. further_optimization_worthwhile Flag
    """

    HEADROOM_THRESHOLD: float = HEADROOM_THRESHOLD

    BARK_EDGES_HZ = [
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
    ]

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        current_goal_scores: Dict[str, float],
        material: str = "unknown",
    ) -> PhysicalCeilingResult:
        """Schaetzt physikalische Qualitaetsdecke.

        Args:
            audio: float32 ndarray
            sr: Sample-Rate in Hz
            current_goal_scores: Aktuelle Musical-Goal-Scores
            material: MaterialType-Label

        Returns:
            PhysicalCeilingResult
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if audio.ndim == 2:
            mono = audio.mean(axis=0)
        else:
            mono = audio.copy()

        # 1. SNR pro Bark-Band
        snr_profile = self._compute_bark_snr(mono, sr)

        # 2. Effektive Bandbreite
        bw_hz = self._effective_bandwidth(mono, sr, snr_profile)

        # 3. Ceiling-Mapping
        mean_snr = float(np.mean(snr_profile))
        ceiling = self._compute_ceiling(mean_snr, bw_hz, mono, sr, material)

        # NaN-Guard
        for g in ceiling:
            if not math.isfinite(ceiling[g]):
                ceiling[g] = 0.98

        # 4. Headroom + Flag
        headroom: Dict[str, float] = {}
        for g, ceil_v in ceiling.items():
            cur = current_goal_scores.get(g, 0.0)
            if not math.isfinite(cur):
                cur = 0.0
            headroom[g] = max(0.0, ceil_v - cur)

        worthwhile = any(h >= self.HEADROOM_THRESHOLD for h in headroom.values())

        return PhysicalCeilingResult(
            ceiling=ceiling,
            snr_profile_db=snr_profile,
            effective_bandwidth_hz=bw_hz,
            headroom_per_goal=headroom,
            further_optimization_worthwhile=worthwhile,
        )

    def _compute_bark_snr(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """SNR pro Bark-Band (24 Baender), float32 [24]."""
        n = min(len(mono), 65536)
        if n < 512:
            return np.full(24, 20.0, dtype=np.float32)

        spec = np.abs(np.fft.rfft(mono[:n])) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        bark_snr = np.zeros(24, dtype=np.float32)
        edges = self.BARK_EDGES_HZ

        for b in range(24):
            lo, hi = edges[b], edges[b + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                bark_snr[b] = -30.0
                continue
            band_energy = spec[mask]
            band_energy_sorted = np.sort(band_energy)
            n_floor = max(1, len(band_energy_sorted) // 10)
            noise_e = float(np.mean(band_energy_sorted[:n_floor])) + 1e-20
            signal_e = float(np.mean(band_energy_sorted[-n_floor:])) + 1e-20
            snr = 10.0 * math.log10(signal_e / noise_e)
            bark_snr[b] = float(np.clip(snr, -60.0, 80.0))

        return bark_snr

    def _effective_bandwidth(self, mono: np.ndarray, sr: int, snr_profile: np.ndarray) -> float:
        """Effektive Bandbreite: hoechstes Bark-Band mit SNR >= -10 dB."""
        edges = self.BARK_EDGES_HZ
        bw = edges[1]  # Minimum
        for b in range(24):
            if snr_profile[b] >= -10.0:
                bw = float(edges[b + 1])
        return bw

    def _compute_ceiling(
        self,
        mean_snr: float,
        bw_hz: float,
        mono: np.ndarray,
        sr: int,
        material: str,
    ) -> Dict[str, float]:
        """Musical-Goal-Ceiling-Mapping (Spec §2.33)."""

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        # Natuerlichkeit: sigmoid((mean_snr - 5) / 5) * 0.97 + 0.03
        nat = sigmoid((mean_snr - 5.0) / 5.0) * 0.97 + 0.03

        # Brillanz: sigmoid((bw_hz - 8000) / 2000) * 0.95
        brill = sigmoid((bw_hz - 8000.0) / 2000.0) * 0.95

        # Stereo-Dekorrelation (vereinfacht)
        if mono.ndim == 1 or len(mono) < 100:
            stereo_decor = 0.0
        else:
            # Proxy: Spektral-Flatness als Stereo-Proxy
            spec = np.abs(np.fft.rfft(mono[: min(len(mono), 8192)])) ** 2 + 1e-15
            gm = math.exp(float(np.mean(np.log(spec))))
            am = float(np.mean(spec))
            stereo_decor = float(np.clip(gm / (am + 1e-10), 0.0, 1.0))

        spatial = sigmoid(stereo_decor * 10.0) * 0.92

        # Wow/Flutter -> GrooveMetric
        groove = 0.97  # Standard-Decke

        # TonalCenter -> SNR-abhaengig
        tonal_center = sigmoid(mean_snr * 2.0) * 0.98

        ceiling = {
            "natuerlichkeit": min(0.98, nat),
            "brillanz": min(0.95, brill),
            "spatial_depth": min(0.92, spatial),
            "groove": min(0.97, groove),
            "tonal_center": min(0.98, tonal_center),
            # Alle anderen: konservativ 0.98
            "waerme": 0.98,
            "authentizitaet": 0.98,
            "emotionalitaet": 0.98,
            "transparenz": 0.98,
            "bass_kraft": 0.98,
            "timbre_authentizitaet": 0.98,
            "micro_dynamics": 0.98,
            "separation_fidelity": 0.98,
            "artikulation": 0.98,
        }

        # Shellac/Wachswalze: kuenstliche Deckel
        if material in {"shellac", "wax_cylinder"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.75)
            ceiling["natuerlichkeit"] = min(ceiling["natuerlichkeit"], 0.85)

        # MP3 (Verlustbehaftet): Codec-Artefakt-bedingter Brillanz-Deckel
        if material in {"mp3_low", "mp3_high", "mp3"}:
            ceiling["brillanz"] = min(ceiling["brillanz"], 0.82)

        return ceiling


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: Optional[PhysicalCeilingEstimator] = None
_lock = threading.Lock()


def get_physical_ceiling_estimator() -> PhysicalCeilingEstimator:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PhysicalCeilingEstimator()
    return _instance


def estimate_physical_ceiling(
    audio: np.ndarray,
    sr: int,
    current_goal_scores: Dict[str, float],
    material: str = "unknown",
) -> PhysicalCeilingResult:
    """Convenience-Wrapper."""
    return get_physical_ceiling_estimator().estimate(audio, sr, current_goal_scores, material)


__all__ = [
    "PhysicalCeilingEstimator",
    "PhysicalCeilingResult",
    "get_physical_ceiling_estimator",
    "estimate_physical_ceiling",
    "HEADROOM_THRESHOLD",
]
