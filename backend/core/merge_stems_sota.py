from __future__ import annotations

from aurik6.analysis.analysis_and_modules import FeatureExtractor, PolicyManager
import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


_instance: Optional["MergeStemsSOTA"] = None
_lock = threading.Lock()


def get_stem_merger(spectral_weight: float = 0.7, phase_align: bool = True, loudness_match: bool = True) -> "MergeStemsSOTA":
    """Get or create MergeStemsSOTA singleton.

    Args:
        spectral_weight: Spectral weighting factor (only used on first call)
        phase_align: Enable phase alignment (only used on first call)
        loudness_match: Enable loudness matching (only used on first call)

    Returns:
        MergeStemsSOTA singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MergeStemsSOTA(spectral_weight, phase_align, loudness_match)
    return _instance


class MergeStemsSOTA:
    """
    SOTA-konformes intelligentes Stem-Merging:
    - Spektrale Gewichtung
    - Phasenkohärenz
    - Lautheitsabgleich
    """

    def __init__(self, spectral_weight: float = 0.7, phase_align: bool = True, loudness_match: bool = True) -> None:
        self.spectral_weight = spectral_weight
        self.phase_align = phase_align
        self.loudness_match = loudness_match
        logger.info(f"MergeStemsSOTA initialized: spectral_weight={spectral_weight}, phase_align={phase_align}")

    def merge(
        self,
        stems: list[np.ndarray],
        sr: int,
        reference: np.ndarray | None = None,
        policy: dict | None = None,
    ) -> np.ndarray:
        if policy is None:
            policy = {}
        policy_manager = PolicyManager(policy)
        extractor = FeatureExtractor()
        # 1. Optional: Phasenkohärenz herstellen
        stems_proc = self._phase_align_stems(stems) if self.phase_align else stems
        features1 = extractor.extract(  # noqa: F841
            np.stack(stems_proc).sum(axis=0),
            sr,
            reference,
            policy_manager=policy_manager,
        )
        if any(v.get("action") == "bypass" for v in policy.values() if isinstance(v, dict)):
            return np.asarray(np.stack(stems_proc).sum(axis=0).astype(stems[0].dtype))
        # 2. Optional: Lautheitsabgleich
        stems_proc = self._loudness_match(stems_proc) if self.loudness_match else stems_proc
        features2 = extractor.extract(  # noqa: F841
            np.stack(stems_proc).sum(axis=0),
            sr,
            reference,
            policy_manager=policy_manager,
        )
        if any(v.get("action") == "bypass" for v in policy.values() if isinstance(v, dict)):
            return np.asarray(np.stack(stems_proc).sum(axis=0).astype(stems[0].dtype))
        # 3. Spektrale Gewichtung und Summierung
        merged = self._spectral_weighted_sum(stems_proc)
        features3 = extractor.extract(merged, sr, reference, policy_manager=policy_manager)  # noqa: F841
        if any(v.get("action") == "bypass" for v in policy.values() if isinstance(v, dict)):
            return np.asarray(merged.astype(stems[0].dtype))
        # 4. Clipping vermeiden
        maxval = np.max(np.abs(merged))
        if maxval > 1.0:
            merged = merged / maxval * 0.999
        merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(np.asarray(merged.astype(stems[0].dtype)), -1.0, 1.0)

    def _phase_align_stems(self, stems: list[np.ndarray]) -> list[np.ndarray]:
        # Einfache Phasenausrichtung: Maximale Kreuzkorrelation
        ref = stems[0]
        aligned = [ref]
        for s in stems[1:]:
            shift = np.argmax(np.correlate(ref, s, mode="full")) - len(s) + 1
            if shift > 0:
                s_aligned = np.pad(s, (shift, 0), mode="constant")[: len(ref)]
            elif shift < 0:
                s_aligned = np.pad(s, (0, -shift), mode="constant")[-shift : len(ref) - shift]
            else:
                s_aligned = s[: len(ref)]
            aligned.append(s_aligned)
        return aligned

    def _loudness_match(self, stems: list[np.ndarray]) -> list[np.ndarray]:
        # RMS-Normalisierung
        rms_ref = np.sqrt(np.mean(stems[0] ** 2))
        matched = [stems[0]]
        for s in stems[1:]:
            rms = np.sqrt(np.mean(s**2)) + 1e-8
            s_matched = s * (rms_ref / rms)
            matched.append(s_matched)
        return matched

    def _spectral_weighted_sum(self, stems: list[np.ndarray]) -> np.ndarray:
        # Gewichtung: z.B. Hauptstimme stärker, Begleitstimmen schwächer
        n = len(stems)
        weights = np.linspace(self.spectral_weight, 1.0 - self.spectral_weight, n)
        weights = weights / np.sum(weights)
        stacked = np.stack(stems, axis=0)
        merged = np.sum(stacked * weights[:, None], axis=0)
        return merged
