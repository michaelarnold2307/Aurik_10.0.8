"""
backend/core/medium_classifier.py  — Aurik 9 Spec §2.1
Automatische Trägermedien-Erkennung (17 Materialtypen).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import math
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_material_type():
    try:
        from backend.core.defect_scanner import MaterialType  # noqa: PLC0415

        return MaterialType
    except Exception:
        return None


@dataclass
class MaterialEvidence:
    material: Any
    confidence: float
    features_matched: List[str] = field(default_factory=list)
    features_against: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        mt = self.material
        return {
            "material": mt.value if hasattr(mt, "value") else str(mt),
            "confidence": self.confidence,
            "features_matched": self.features_matched,
            "features_against": self.features_against,
        }


@dataclass
class ClassificationResult:
    material: Any
    confidence: float
    evidence: List[MaterialEvidence] = field(default_factory=list)
    bandwidth_hz: float = 0.0
    snr_db: float = 0.0
    noise_color: float = 1.0
    crackle_density: float = 0.0
    wow_flutter_hz: float = 0.0
    block_artifact: float = 0.0
    pre_echo_ms: float = 0.0
    classifier_source: str = "dsp"

    @property
    def material_type(self) -> str:
        mt = self.material
        if hasattr(mt, "value"):
            return str(mt.value)
        return str(mt)

    def as_dict(self) -> Dict[str, Any]:
        mt = self.material
        return {
            "material": mt.value if hasattr(mt, "value") else str(mt),
            "confidence": self.confidence,
            "bandwidth_hz": self.bandwidth_hz,
            "snr_db": self.snr_db,
            "noise_color": self.noise_color,
            "crackle_density": self.crackle_density,
            "wow_flutter_hz": self.wow_flutter_hz,
            "block_artifact": self.block_artifact,
            "pre_echo_ms": self.pre_echo_ms,
            "classifier_source": self.classifier_source,
            "n_evidence": len(self.evidence),
        }


class _SpectralFingerprinter:
    _FRAME_SIZE = 1024
    _HOP_SIZE = 512

    def extract(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        mono = self._to_mono(audio)
        if mono.size < self._FRAME_SIZE:
            return self._null_features()
        f: Dict[str, float] = {}
        f["bandwidth_hz"] = self._bandwidth(mono, sr)
        f["snr_db"] = self._snr(mono)
        f["noise_color"] = self._noise_color(mono, sr)
        f["crackle_density"] = self._crackle_density(mono)
        f["wow_flutter_hz"] = self._wow_flutter(mono, sr)
        f["block_artifact"] = self._block_artifact(mono)
        f["pre_echo_ms"] = self._pre_echo(mono, sr)
        for k, v in f.items():
            if not math.isfinite(v):
                f[k] = 0.0
        return f

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, -1.0, 1.0)
        return arr.mean(axis=0) if arr.ndim == 2 else arr

    @staticmethod
    def _null_features() -> Dict[str, float]:
        return {
            "bandwidth_hz": 0.0,
            "snr_db": 0.0,
            "noise_color": 1.0,
            "crackle_density": 0.0,
            "wow_flutter_hz": 0.0,
            "block_artifact": 0.0,
            "pre_echo_ms": 0.0,
        }

    def _bandwidth(self, mono: np.ndarray, sr: int) -> float:
        n = min(mono.size, 16384)
        spec = np.abs(np.fft.rfft(mono[:n], n=n))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        cumsum = np.cumsum(spec)
        if cumsum[-1] < 1e-12:
            return 0.0
        idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1]))
        return float(freqs[min(idx, len(freqs) - 1)])

    def _snr(self, mono: np.ndarray) -> float:
        frame = self._FRAME_SIZE
        n_frames = max(1, len(mono) // frame)
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        energies = np.sqrt(np.mean(frames**2, axis=1))
        noise = float(np.percentile(energies, 5)) + 1e-10
        signal = float(np.percentile(energies, 95))
        if signal <= noise:
            return 0.0
        return float(np.clip(20.0 * math.log10(signal / noise), 0.0, 90.0))

    def _noise_color(self, mono: np.ndarray, sr: int = 48000) -> float:
        n = min(mono.size, 8192)
        spec = np.abs(np.fft.rfft(mono[:n], n=n)) + 1e-10
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        mask = freqs > 50.0
        if not mask.any():
            return 1.0
        log_f = np.log10(freqs[mask] + 1e-10)
        log_s = np.log10(spec[mask])
        if log_f.std() < 1e-8:
            return 1.0
        beta = -float(np.polyfit(log_f, log_s, 1)[0])
        return float(np.clip(beta, 0.0, 4.0))

    def _crackle_density(self, mono: np.ndarray) -> float:
        if mono.size < 512:
            return 0.0
        sigma = float(np.std(mono)) + 1e-10
        return float(np.clip((np.abs(mono) > 4.0 * sigma).mean() * 100.0, 0.0, 1.0))

    def _wow_flutter(self, mono: np.ndarray, sr: int) -> float:
        frame = max(self._FRAME_SIZE, int(0.02 * sr))
        n_frames = max(1, len(mono) // frame)
        if n_frames < 2:
            return 0.0
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        zcr = np.mean(np.abs(np.diff(np.sign(frames))) / 2.0, axis=1)
        zcr_hz = zcr * sr / 2.0
        denom = max(float(np.mean(zcr_hz)), 1.0)
        return float(np.clip(float(np.std(zcr_hz)) / denom * 10.0, 0.0, 20.0))

    def _block_artifact(self, mono: np.ndarray) -> float:
        block = 576
        n_blocks = len(mono) // block
        if n_blocks < 2:
            return 0.0
        frames = mono[: n_blocks * block].reshape(n_blocks, block)
        energies = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
        deltas = np.abs(np.diff(np.log(energies)))
        return float(np.clip((deltas > 1.0).mean(), 0.0, 1.0))

    def _pre_echo(self, mono: np.ndarray, sr: int) -> float:
        frame = self._FRAME_SIZE
        n_frames = len(mono) // frame
        if n_frames < 4:
            return 0.0
        frames = mono[: n_frames * frame].reshape(n_frames, frame)
        energies = np.sqrt(np.mean(frames**2, axis=1))
        if energies.max() < 1e-8:
            return 0.0
        max_idx = int(np.argmax(energies))
        if max_idx == 0:
            return 0.0
        ratio = float(energies[max_idx - 1]) / (float(energies[max_idx]) + 1e-10)
        ms_per_frame = frame / sr * 1000.0
        return float(np.clip(ratio * ms_per_frame * 2.0, 0.0, 50.0))


class _MaterialScorer:
    def score(self, features: Dict[str, float], MaterialType: Any) -> "ClassificationResult":
        bw = features.get("bandwidth_hz", 0.0)
        snr = features.get("snr_db", 0.0)
        nc = features.get("noise_color", 1.0)
        cd = features.get("crackle_density", 0.0)
        wf = features.get("wow_flutter_hz", 0.0)
        ba = features.get("block_artifact", 0.0)
        pe = features.get("pre_echo_ms", 0.0)

        scores = {
            "shellac": self._s(nc > 1.8, snr < 15.0, bw < 8000.0),
            "wax_cylinder": self._s(nc > 2.2, snr < 10.0, bw < 5500.0),
            "vinyl": self._s(cd > 0.0005, nc > 1.2, bw > 8000.0),
            "tape": self._s(nc > 1.3, wf > 0.5, snr < 35.0, bw < 16000.0),
            "reel_tape": self._s(nc > 1.1, wf < 0.5, snr < 30.0),
            "wire_recording": self._s(wf > 1.5, snr < 20.0),
            "lacquer_disc": self._s(cd > 0.001, snr < 25.0, bw < 12000.0),
            "dat": self._s(bw > 18000.0, ba > 0.05, snr > 40.0),
            "cd_digital": self._s(bw > 18000.0, snr > 50.0, ba < 0.05, wf < 0.3),
            "mp3_low": self._s(ba > 0.2, pe > 5.0, bw < 16000.0),
            "mp3_high": self._s(ba > 0.05, bw > 14000.0, snr > 30.0),
            "aac": self._s(ba > 0.03, bw > 17000.0, pe < 3.0),
            "minidisc": self._s(ba > 0.1, bw < 16000.0, snr > 35.0),
            "streaming": self._s(ba > 0.02, bw > 16000.0),
            "unknown": 0.10,
        }

        best_name, best_score = max(scores.items(), key=lambda x: x[1])
        total = max(1.0, sum(scores.values()))
        confidence = float(np.clip(best_score / total, 0.0, 1.0))
        material = self._find_enum(MaterialType, best_name)

        evidence = [
            MaterialEvidence(
                material=self._find_enum(MaterialType, n),
                confidence=float(np.clip(s, 0.0, 1.0)),
                features_matched=["snr_db", "bandwidth_hz", "crackle_density"],
                features_against=[],
            )
            for n, s in sorted(scores.items(), key=lambda x: -x[1])[:5]
        ]

        return ClassificationResult(
            material=material,
            confidence=confidence,
            evidence=evidence,
            bandwidth_hz=features.get("bandwidth_hz", 0.0),
            snr_db=features.get("snr_db", 0.0),
            noise_color=features.get("noise_color", 1.0),
            crackle_density=features.get("crackle_density", 0.0),
            wow_flutter_hz=features.get("wow_flutter_hz", 0.0),
            block_artifact=features.get("block_artifact", 0.0),
            pre_echo_ms=features.get("pre_echo_ms", 0.0),
            classifier_source="dsp",
        )

    @staticmethod
    def _s(*conds: bool) -> float:
        return float(sum(1.0 for c in conds if c))

    @staticmethod
    def _find_enum(MaterialType: Any, name: str) -> Any:
        if MaterialType is None:
            return name
        for m in MaterialType:
            if m.value == name or m.name.lower() == name.lower():
                return m
        return name


_sha_cache: Dict[str, ClassificationResult] = {}
_sha_cache_lock = threading.Lock()
_MAX_CACHE = 64


class MediumClassifier:
    """Automatische Trägermedien-Erkennung (Aurik Spec §2.1)."""

    def __init__(self) -> None:
        self._fp = _SpectralFingerprinter()
        self._sc = _MaterialScorer()

    def classify_medium(self, audio: np.ndarray, sr: int) -> "ClassificationResult":
        return self.classify(audio, sr, use_ml=False)

    def classify(self, audio: np.ndarray, sr: int, use_ml: bool = True) -> "ClassificationResult":
        key = self._cache_key(audio, sr)
        with _sha_cache_lock:
            if key in _sha_cache:
                return _sha_cache[key]
        if use_ml:
            r = self._try_clap_classification(audio, sr)
            if r is not None:
                self._cache_put(key, r)
                return r
        r = self._dsp_classify(audio, sr)
        self._cache_put(key, r)
        return r

    def _dsp_classify(self, audio: np.ndarray, sr: int) -> "ClassificationResult":
        MT = _get_material_type()
        if audio.size == 0:
            mat = MT.UNKNOWN if MT is not None else "unknown"
            return ClassificationResult(
                material=mat, confidence=0.0, evidence=[MaterialEvidence(mat, 0.0)], classifier_source="unknown"
            )
        return self._sc.score(self._fp.extract(audio, sr), MT)

    def _try_clap_classification(self, audio: np.ndarray, sr: int) -> Optional["ClassificationResult"]:
        try:
            from plugins.laion_clap_plugin import get_laion_clap_plugin  # noqa: PLC0415

            plugin = get_laion_clap_plugin()
            r = plugin.classify_medium(audio, sr)
            if r is not None and r.confidence >= 0.35:
                r.classifier_source = "clap_ml"
                return r
        except Exception:
            pass
        return None

    @staticmethod
    def _cache_key(audio: np.ndarray, sr: int) -> str:
        h = hashlib.sha256()
        h.update(audio.ravel().view(np.uint8)[:65536])
        h.update(sr.to_bytes(4, "little"))
        return h.hexdigest()[:16]

    @staticmethod
    def _cache_put(key: str, result: "ClassificationResult") -> None:
        with _sha_cache_lock:
            if len(_sha_cache) >= _MAX_CACHE:
                del _sha_cache[next(iter(_sha_cache))]
            _sha_cache[key] = result


_instance: Optional[MediumClassifier] = None
_lock = threading.Lock()


def get_medium_classifier() -> MediumClassifier:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MediumClassifier()
    return _instance


def classify_medium(audio: np.ndarray, sr: int, use_ml: bool = True) -> ClassificationResult:
    return get_medium_classifier().classify(audio, sr, use_ml=use_ml)


__all__ = ["ClassificationResult", "MaterialEvidence", "MediumClassifier", "get_medium_classifier", "classify_medium"]
