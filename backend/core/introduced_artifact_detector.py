"""
Aurik 9 — IntroducedArtifactDetector (IAD) §2.23
==================================================
Erkennt durch Restaurierung neu eingebrachte Artefakte im restaurierten Audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArtifactRegion:
    """Zeitbereich mit erkanntem Artefakt."""

    artifact_type: str
    start_sample: int
    end_sample: int
    severity: float
    confidence: float
    description: str = ""


@dataclass
class IADResult:
    """Ergebnis des IntroducedArtifactDetectors."""

    has_artifacts: bool
    artifacts: list[ArtifactRegion] = field(default_factory=list)
    n_ml_hallucinations: int = 0
    n_nmf_clicks: int = 0
    n_pvoc_smearing: int = 0
    n_musical_noise: int = 0
    artifact_mask: Optional[np.ndarray] = None
    total_contaminated_fraction: float = 0.0
    confidence: float = 1.0

    @property
    def artifact_types(self) -> list[str]:
        return sorted({a.artifact_type for a in self.artifacts}) if self.artifacts else []


# Rückwärtskompatibel
IADRegion = ArtifactRegion


class IntroducedArtifactDetector:
    """Erkennt durch Restaurierung neu eingebrachte Artefakte (§2.23)."""

    IAD_ARTIFACT_TYPES: list[str] = [
        "ml_hallucination",
        "nmf_residual_click",
        "phase_vocoder_smearing",
        "musical_noise",
    ]

    CLICK_THRESHOLD_DB: float = 12.0
    CLICK_MAX_DURATION_MS: float = 5.0
    PVOC_SMEAR_THRESHOLD_MS: float = 10.0
    MUSICAL_NOISE_THRESHOLD_DB: float = 3.0
    SILENCE_THRESHOLD_DBFS: float = -40.0
    HARMONICITY_THRESHOLD: float = 0.70

    def detect(self, original: np.ndarray, restored: np.ndarray, sr: int) -> IADResult:
        """Erkennt durch Restaurierung eingebrachte Artefakte."""
        original = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.nan_to_num(np.asarray(restored, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        original = np.clip(original, -1.0, 1.0)
        restored = np.clip(restored, -1.0, 1.0)
        orig_mono = original if original.ndim == 1 else original.mean(axis=0)
        rest_mono = restored if restored.ndim == 1 else restored.mean(axis=0)
        min_len = min(len(orig_mono), len(rest_mono))
        if min_len == 0:
            return IADResult(has_artifacts=False, confidence=1.0)
        orig_mono = orig_mono[:min_len]
        rest_mono = rest_mono[:min_len]
        n_samples = min_len
        artifact_mask = np.zeros(n_samples, dtype=bool)
        artifacts: list[ArtifactRegion] = []
        residuum = np.nan_to_num(rest_mono - orig_mono)

        for a in self._detect_nmf_clicks(orig_mono, residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_musical_noise(orig_mono, residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_ml_hallucinations(residuum, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True
        for a in self._detect_pvoc_smearing(orig_mono, rest_mono, sr):
            artifacts.append(a)
            artifact_mask[a.start_sample : a.end_sample] = True

        frac = float(np.sum(artifact_mask)) / n_samples
        return IADResult(
            has_artifacts=len(artifacts) > 0,
            artifacts=artifacts,
            n_ml_hallucinations=sum(1 for a in artifacts if a.artifact_type == "ml_hallucination"),
            n_nmf_clicks=sum(1 for a in artifacts if a.artifact_type == "nmf_residual_click"),
            n_pvoc_smearing=sum(1 for a in artifacts if a.artifact_type == "phase_vocoder_smearing"),
            n_musical_noise=sum(1 for a in artifacts if a.artifact_type == "musical_noise"),
            artifact_mask=artifact_mask,
            total_contaminated_fraction=frac,
            confidence=float(np.clip(1.0 - frac, 0.0, 1.0)),
        )

    def get_artifact_mask(self, iad_result: IADResult, n_samples: int) -> np.ndarray:
        if iad_result.artifact_mask is None:
            return np.zeros(n_samples, dtype=bool)
        mask = iad_result.artifact_mask
        if len(mask) < n_samples:
            return np.pad(mask, (0, n_samples - len(mask)))
        return mask[:n_samples]

    def _detect_nmf_clicks(self, orig: np.ndarray, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        click_len = max(1, int(self.CLICK_MAX_DURATION_MS / 1000.0 * sr))
        kernel = np.ones(click_len) / click_len
        energy_orig = np.sqrt(np.convolve(orig**2, kernel, mode="same") + 1e-12)
        energy_res = np.sqrt(np.convolve(residuum**2, kernel, mode="same") + 1e-12)
        threshold_ratio = 10.0 ** (self.CLICK_THRESHOLD_DB / 20.0)
        click_mask = (energy_res / energy_orig) > threshold_ratio
        artifacts: list[ArtifactRegion] = []
        in_click = False
        start = 0
        for i, v in enumerate(click_mask):
            if v and not in_click:
                in_click = True
                start = i
            elif not v and in_click:
                in_click = False
                if (i - start) <= click_len * 2:
                    ratio = energy_res[start:i] / energy_orig[start:i]
                    sev = float(np.clip(float(np.max(ratio)) / threshold_ratio, 0.0, 1.0))
                    artifacts.append(
                        ArtifactRegion(
                            "nmf_residual_click", max(0, start - click_len), min(len(orig), i + click_len), sev, 0.75
                        )
                    )
        return artifacts

    def _detect_musical_noise(self, orig: np.ndarray, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        win = max(1, int(0.10 * sr))
        hop = max(1, int(0.05 * sr))
        artifacts: list[ArtifactRegion] = []
        for s in range(0, len(orig) - win, hop):
            e = s + win
            db_o = 20.0 * np.log10(max(float(np.sqrt(np.mean(orig[s:e] ** 2))), 1e-10))
            if db_o < self.SILENCE_THRESHOLD_DBFS:
                db_r = 20.0 * np.log10(max(float(np.sqrt(np.mean(residuum[s:e] ** 2))), 1e-10))
                if db_r > db_o + self.MUSICAL_NOISE_THRESHOLD_DB:
                    sev = float(np.clip((db_r - db_o) / 20.0, 0.0, 1.0))
                    artifacts.append(ArtifactRegion("musical_noise", s, e, sev, 0.70))
        return artifacts

    def _detect_ml_hallucinations(self, residuum: np.ndarray, sr: int) -> list[ArtifactRegion]:
        win = max(1, int(2.0 * sr))
        hop = max(1, int(1.0 * sr))
        artifacts: list[ArtifactRegion] = []
        for s in range(0, len(residuum) - win, hop):
            h = self._harmonicity(residuum[s : s + win], sr)
            if h > self.HARMONICITY_THRESHOLD:
                artifacts.append(ArtifactRegion("ml_hallucination", s, s + win, float(np.clip(h, 0.0, 1.0)), float(h)))
        return artifacts

    def _detect_pvoc_smearing(self, orig: np.ndarray, rest: np.ndarray, sr: int) -> list[ArtifactRegion]:
        hop = 512
        smear = max(1, int(self.PVOC_SMEAR_THRESHOLD_MS / 1000.0 * sr))

        def frames(sig: np.ndarray) -> np.ndarray:
            return np.array([float(np.sqrt(np.mean(sig[i : i + hop] ** 2))) for i in range(0, len(sig) - hop, hop)])

        eo = frames(orig)
        er = frames(rest)
        n = min(len(eo), len(er))
        if n < 2:
            return []
        do = np.diff(eo[:n])
        dr = np.diff(er[:n])
        artifacts: list[ArtifactRegion] = []
        for i in range(1, min(n - 1, len(do))):
            if do[i - 1] > 0.1:
                bj, bv = i, -1.0
                for j in range(max(0, i - 20), min(n - 1, i + 20)):
                    if j < len(dr) and dr[j] > bv:
                        bv = dr[j]
                        bj = j
                if abs(bj - i) * hop > smear:
                    sev = float(np.clip(abs(bj - i) * hop / max(smear * 5, 1), 0.0, 1.0))
                    artifacts.append(
                        ArtifactRegion("phase_vocoder_smearing", i * hop, min(len(orig), (i + 5) * hop), sev, 0.65)
                    )
        return artifacts

    def _harmonicity(self, frame: np.ndarray, sr: int) -> float:
        if len(frame) < sr // 10:
            return 0.0
        n = len(frame)
        nf = 1
        while nf < 2 * n:
            nf <<= 1
        F = np.fft.rfft(frame, n=nf)
        ac = np.fft.irfft(F * np.conj(F), n=nf)[:n]
        if ac[0] < 1e-10:
            return 0.0
        acn = ac / ac[0]
        lo = max(1, int(sr / 255))
        hi = min(n - 1, int(sr / 85))
        if lo >= hi:
            return 0.0
        return float(np.clip(np.nan_to_num(float(np.max(acn[lo:hi]))), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (§3.2)
# ---------------------------------------------------------------------------
_instance: Optional[IntroducedArtifactDetector] = None
_lock = threading.Lock()


def get_iad() -> IntroducedArtifactDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = IntroducedArtifactDetector()
    return _instance


get_introduced_artifact_detector = get_iad


def detect_introduced_artifacts(original: np.ndarray, restored: np.ndarray, sr: int) -> IADResult:
    """Convenience-Wrapper für Artefakt-Erkennung."""
    if sr != 48000:
        raise ValueError(f"SR muss 48000 Hz sein, erhalten: {sr}")
    return get_iad().detect(original, restored, sr)


__all__ = [
    "ArtifactRegion",
    "IADRegion",
    "IADResult",
    "IntroducedArtifactDetector",
    "get_iad",
    "get_introduced_artifact_detector",
    "detect_introduced_artifacts",
]
