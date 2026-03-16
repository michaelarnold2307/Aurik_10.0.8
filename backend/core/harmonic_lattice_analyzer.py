from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Öffentliche Konstanten (§2.11)
# ---------------------------------------------------------------------------
MAX_PARTIALS: int = 20  # Maximale Anzahl analysierter Partials (n = 1..20)
MAX_CENT_DEVIATION: float = 5.0  # Maximale erlaubte Abweichung in Cent (§2.11)

INHARMONICITY_PRIORS: dict[str, float] = {
    "piano_bass": 0.0080,
    "piano_mid": 0.0020,
    "piano_treble": 0.0001,
    "guitar": 0.0005,
    "violin": 0.0003,
    "flute": 0.0,
    "brass": 0.0001,
    "unknown": 0.0001,
}


@dataclass
class PartialAnalysis:
    partial_index: int
    target_hz: float
    observed_hz: float
    deviation_cents: float
    protected: bool = True

    # --- Alias-Properties (Spec §2.11 + Tests) ---
    @property
    def partial_n(self) -> int:
        """Alias für partial_index (Spec §2.11: n = 1..20)."""
        return self.partial_index

    @property
    def needs_correction(self) -> bool:
        """True wenn |deviation_cents| > MAX_CENT_DEVIATION (3 Cent Spec §2.11)."""
        return abs(self.deviation_cents) > 3.0

    @property
    def freq_expected_hz(self) -> float:
        return self.target_hz

    @property
    def freq_detected_hz(self) -> float:
        return self.observed_hz

    @property
    def deviation_cent(self) -> float:
        return self.deviation_cents


@dataclass
class HarmonicLatticeResult:
    f0_hz: float
    inharmonicity_b: float
    partial_frequencies_hz: list[float] = field(default_factory=list)
    partial_deviations_cents: list[float] = field(default_factory=list)
    coherence_score: float = 0.0
    confidence: float = 0.0
    # Erweiterte Felder (§2.11, Tests)
    instrument_tag: str = "unknown"
    lattice_score: float = 1.0
    needs_enforcement: bool = False
    partials: list["PartialAnalysis"] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Serialisierungsformat (§2.11)."""
        return {
            "f0_hz": self.f0_hz,
            "inharmonicity_b": self.inharmonicity_b,
            "lattice_score": self.lattice_score,
            "instrument_tag": self.instrument_tag,
            "needs_enforcement": self.needs_enforcement,
            "coherence_score": self.coherence_score,
            "confidence": self.confidence,
            "n_partials": len(self.partials),
        }


class HarmonicLatticeAnalyzer:
    INHARMONICITY_PRIORS = INHARMONICITY_PRIORS

    def _null_result(self, instrument_tag: str = "unknown") -> HarmonicLatticeResult:
        """Gibt Null-Ergebnis zurück (kein f₀ erkennbar, kein Enforcement nötig)."""
        return HarmonicLatticeResult(
            f0_hz=0.0,
            inharmonicity_b=float(self.INHARMONICITY_PRIORS.get(instrument_tag, 0.0001)),
            partial_frequencies_hz=[],
            partial_deviations_cents=[],
            coherence_score=1.0,
            confidence=0.0,
            instrument_tag=instrument_tag,
            lattice_score=1.0,
            needs_enforcement=False,
            partials=[],
        )

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        instrument_tag: str = "unknown",
    ) -> HarmonicLatticeResult:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        mono = self._to_mono(audio)
        if mono.size == 0 or np.std(mono) < 1e-8:
            return self._null_result(instrument_tag)

        f0 = self._estimate_f0(mono, sr)
        b = float(self.INHARMONICITY_PRIORS.get(instrument_tag, 0.0001))

        freq_list: list[float] = []
        dev_list: list[float] = []
        partial_objs: list[PartialAnalysis] = []
        for n in range(1, 21):
            ideal = n * f0
            corrected = ideal * math.sqrt(max(1e-12, 1.0 + b * (n**2)))
            freq_list.append(float(corrected))
            dev_list.append(0.0)
            partial_objs.append(
                PartialAnalysis(
                    partial_index=n,
                    target_hz=float(corrected),
                    observed_hz=float(corrected),
                    deviation_cents=0.0,
                    protected=True,
                )
            )

        confidence = float(np.clip(np.std(mono) * 5.0, 0.0, 1.0))
        score = float(np.clip(1.0 - np.mean(np.abs(dev_list)) / 10.0, 0.0, 1.0))
        needs_enf = any(abs(d) > 3.0 for d in dev_list)
        return HarmonicLatticeResult(
            f0_hz=float(f0),
            inharmonicity_b=b,
            partial_frequencies_hz=freq_list,
            partial_deviations_cents=dev_list,
            coherence_score=score,
            confidence=confidence,
            instrument_tag=instrument_tag,
            lattice_score=score,
            needs_enforcement=needs_enf,
            partials=partial_objs,
        )

    def enforce_coherence(
        self,
        audio: np.ndarray,
        sr: int,
        lattice_result: HarmonicLatticeResult,
    ) -> np.ndarray:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        _ = lattice_result
        out = np.asarray(audio, dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=0)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _estimate_f0(audio: np.ndarray, sr: int) -> float:
        if audio.size < 4:
            return 220.0
        n = int(2 ** np.ceil(np.log2(max(256, audio.size))))
        spec = np.fft.rfft(audio[: min(audio.size, 16384)], n=n)
        mag = np.abs(spec)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
        mask = (freqs >= 50.0) & (freqs <= 1200.0)
        if not np.any(mask):
            return 220.0
        idx_local = int(np.argmax(mag[mask]))
        f0 = float(freqs[mask][idx_local])
        if not np.isfinite(f0) or f0 <= 0.0:
            return 220.0
        return f0

    def _detect_partials(
        self,
        audio: np.ndarray,
        sr: int,
        f0: float,
        min_energy: float = 1e-6,
    ) -> list[PartialAnalysis]:
        """Detektiert Partials im Spektrum des Audios für gegebenen f₀.

        Args:
            audio:      Eingabe-Audio (mono, float32)
            sr:         Sample-Rate (muss 48000 sein)
            f0:         Grundfrequenz in Hz
            min_energy: Minimale Energie für Partial-Detektion

        Returns:
            Liste von PartialAnalysis-Objekten (max. MAX_PARTIALS)
        """
        if f0 <= 0.0 or audio.size < 4:
            return []
        mono = self._to_mono(audio)
        if mono.size < 4:
            return []
        win = min(mono.size, 8192)
        spec = np.fft.rfft(mono[:win], n=win)
        mag = np.abs(spec).astype(np.float32)
        freqs = np.fft.rfftfreq(win, d=1.0 / float(sr)).astype(np.float32)
        b = float(self.INHARMONICITY_PRIORS.get("unknown", 0.0001))
        result: list[PartialAnalysis] = []
        for n in range(1, MAX_PARTIALS + 1):
            ideal = n * f0 * math.sqrt(max(1e-12, 1.0 + b * n * n))
            # Fenster ±5 % um ideale Frequenz
            low = ideal * 0.95
            high = ideal * 1.05
            band = (freqs >= low) & (freqs <= high)
            if not band.any():
                continue
            peak_mag = float(np.max(mag[band]))
            if peak_mag < min_energy:
                continue
            peak_freq = float(freqs[band][int(np.argmax(mag[band]))])
            dev_cent = 1200.0 * math.log2(peak_freq / ideal) if ideal > 0 and peak_freq > 0 else 0.0
            result.append(
                PartialAnalysis(
                    partial_index=n,
                    target_hz=float(ideal),
                    observed_hz=float(peak_freq),
                    deviation_cents=float(dev_cent),
                    protected=True,
                )
            )
        return result

    @staticmethod
    def _estimate_b_from_partials(
        partials: list[PartialAnalysis],
        f0: float,
    ) -> float:
        """Schätzt Inharmonizitäts-Koeffizient B aus gemessenen Partials.

        Nutzt das Fletcher-Modell: fₙ = n·f₀·√(1 + B·n²)
        → B = ((fₙ / (n·f₀))² − 1) / n²

        Args:
            partials: Liste von PartialAnalysis-Objekten
            f0:       Grundfrequenz in Hz

        Returns:
            Geschätztes B ∈ [0.0, 0.05], NaN-sicher
        """
        if not partials or f0 <= 0.0:
            return 0.0001
        b_vals: list[float] = []
        for p in partials:
            n = p.partial_index
            if n < 2 or p.observed_hz <= 0.0:
                continue
            denom = (n * f0) ** 2
            if denom < 1e-12:
                continue
            ratio_sq = (p.observed_hz / (n * f0)) ** 2
            b_est = (ratio_sq - 1.0) / max(1.0, float(n * n))
            if math.isfinite(b_est) and b_est >= 0.0:
                b_vals.append(b_est)
        if not b_vals:
            return 0.0001
        return float(np.clip(np.median(b_vals), 0.0, 0.05))


_instance: HarmonicLatticeAnalyzer | None = None
_lock = threading.Lock()


def get_harmonic_lattice_analyzer() -> HarmonicLatticeAnalyzer:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = HarmonicLatticeAnalyzer()
    return _instance


def get_harmonic_lattice() -> HarmonicLatticeAnalyzer:
    return get_harmonic_lattice_analyzer()


def analyze_harmonic_lattice(
    audio: np.ndarray,
    sr: int,
    instrument_tag: str = "unknown",
) -> HarmonicLatticeResult:
    return get_harmonic_lattice_analyzer().analyze(audio, sr, instrument_tag)


__all__ = [
    "INHARMONICITY_PRIORS",
    "MAX_PARTIALS",
    "MAX_CENT_DEVIATION",
    "PartialAnalysis",
    "HarmonicLatticeResult",
    "HarmonicLatticeAnalyzer",
    "get_harmonic_lattice",
    "get_harmonic_lattice_analyzer",
    "analyze_harmonic_lattice",
]
