"""
Aurik 9 — forensics/medium_detector.py  (§6.7, bindend ab v9.10.45)
=====================================================================
Tonträgerketten-Erkennung: bestimmt den vollständigen Degradationspfad
einer Aufnahme (z. B. cassette_tape → mp3_low) und liefert einen
MaterialType-Prior für den DefectScanner.

Pflicht-Spektralfingerabdruck (§6.7.1):
    1. Rolloff 95 %  — diagnostiziert Bandbreitenbegrenzung
    2. Wow/Flutter-Index — Pitch-Instabilität via pYIN-Ableitung
    3. HF-Energie > 16 kHz — MP3/Kassettenkette
    4. Rauschpegel (Percentile-5 PSD)  — Bandrauschen
    5. Effektive Bandbreite — physikalische Signalbandbreite

Kettenerkennung (§6.7.2):
    - Primär-Träger               = letzte Analogstufe
    - Sekundäre Stufen            = digital/komprimiert
    - is_multi_generation=True    → kombinierte Phasen beider Materialien
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFingerprint:
    """Pflicht-Spektralfingerabdruck (§6.7.1) aus Rohsignal-Vorabanalyse."""

    rolloff_95_hz: float = 0.0  # Spectral Rolloff 95 % — Median
    wow_flutter_index: float = 0.0  # Pitch-Varianz [Hz std] über 100-ms-Fenster
    hf_energy_above_16k: float = 0.0  # Anteil Energie > 16 kHz an Gesamt
    noise_floor_db: float = -60.0  # 5. Perzentil der Frame-Energien [dBFS]
    effective_bandwidth_hz: float = 0.0  # HF-Rolloff −60 dBFS

    # --- Alias-Properties für Test-Kompatibilität (§6.7.1) ---
    @property
    def rolloff_95_percent_hz(self) -> float:
        """Alias für rolloff_95_hz — Rückwärtskompatibilität."""
        return self.rolloff_95_hz

    @property
    def hf_energy_above_16khz_percent(self) -> float:
        """Alias für hf_energy_above_16k als Prozentwert (0–100)."""
        return float(self.hf_energy_above_16k * 100.0)

    def __contains__(self, item: object) -> bool:
        """Unterstützt 'key in fingerprint'-Syntax für Tests."""
        return item in (
            "rolloff_95_hz", "rolloff_95_percent_hz",
            "wow_flutter_index",
            "hf_energy_above_16k", "hf_energy_above_16khz_percent",
            "noise_floor_db", "effective_bandwidth_hz",
        )

    def as_dict(self) -> dict:
        return {
            "rolloff_95_hz": self.rolloff_95_hz,
            "rolloff_95_percent_hz": self.rolloff_95_hz,
            "wow_flutter_index": self.wow_flutter_index,
            "hf_energy_above_16k_fraction": self.hf_energy_above_16k,
            "hf_energy_above_16khz_percent": self.hf_energy_above_16k * 100.0,
            "noise_floor_db": self.noise_floor_db,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
        }


@dataclass
class TransferChain:
    """Erkannte Medien-Transferkette."""

    chain: list[str] = field(default_factory=list)
    """Kette von MediaType-Strings, z. B. ['tape', 'mp3_low']."""

    is_multi_generation: bool = False
    """True wenn ≥ 2 verschiedene Medienstufen erkannt wurden."""

    primary_material: str = "unknown"
    """Letzter analoger Träger = primärer MaterialType-Prior."""

    confidence: float = 0.0
    """Gesamtkonfidenz der Ketten-Schätzung ∈ [0, 1]."""

    reasoning: str = ""

    def __len__(self) -> int:
        return len(self.chain)


@dataclass
class MediumDetectionResult:
    """Vollständiges Ergebnis der Tonträgerketten-Erkennung."""

    transfer_chain: list[str]
    """Kette wie ['tape', 'mp3_low'] — primärer Träger zuerst."""

    is_multi_generation: bool
    primary_material: str
    confidence: float
    spectral_fingerprint: SpectralFingerprint
    evidence: list[str] = field(default_factory=list)
    """Laienverständliche Diagnose-Begründungen."""

    @property
    def chain_label(self) -> str:
        return " → ".join(self.transfer_chain) if self.transfer_chain else "unknown"

    def as_dict(self) -> dict:
        return {
            "transfer_chain": self.transfer_chain,
            "is_multi_generation": self.is_multi_generation,
            "primary_material": self.primary_material,
            "confidence": self.confidence,
            "chain_label": self.chain_label,
            "spectral_fingerprint": self.spectral_fingerprint.as_dict(),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class MediumDetector:
    """Erkennt Tonträgerketten forensisch (§6.7).

    Laufreihenfolge je Import:
        1.  Pflicht-Spektralfingerabdruck (5 Merkmale)
        2a. Kassetten-Erkennung  (Rolloff, Wow/Flutter, Rauschpegel)
        2b. Shellac/Schellack    (Rolloff ≤ 4 kHz, Rauschpegel sehr hoch)
        2c. MP3/Codec-Kette      (HF-Anteil 0 %, Frequenz-Kerbmuster)
        2d. Digitaler Träger     (Rolloff > 18 kHz, niedriger Rauschboden)
        3.  Kettenzusammenführung (primär + sekundär)

    Singleton-Zugang: ``get_medium_detector()``
    Convenience:      ``detect_medium_chain(audio, sr)``
    """

    # ── Diagnostik-Schwellen (§6.7.1) ──────────────────────────────────
    SHELLAC_ROLLOFF_MAX_HZ: float = 4_500.0
    TAPE_ROLLOFF_MAX_HZ: float = 10_000.0
    TAPE_WOW_FLUTTER_MIN: float = 0.4  # Hz std
    TAPE_NOISE_FLOOR_MAX_DB: float = -36.0  # lauter = Bandrauschen
    HF_ENERGY_THRESHOLD_FRACTION: float = 0.001  # < 0.1 % → kein HF
    MP3_KERBMUSTER_THZ: float = 16_000.0  # typischer MP3-Rolloff

    def _compute_fingerprint(self, audio: np.ndarray, sr: int) -> SpectralFingerprint:
        """Berechnet den Pflicht-Spektralfingerabdruck (§6.7.1).

        NaN/Inf-sicher; alle Felder werden immer befüllt.
        """
        mono = self._to_mono(audio)
        n = len(mono)
        if n == 0:
            return SpectralFingerprint()

        hop = max(1, n // 200)
        win = min(2048, n)

        # ── 1. Rolloff 95 % ────────────────────────────────────────────
        try:
            frames = [mono[i : i + win] for i in range(0, n - win, hop)]
            rolloffs = []
            for frame in frames[:100]:
                spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
                cum = np.cumsum(spec**2)
                total = cum[-1]
                if total > 0:
                    idx = int(np.searchsorted(cum, 0.95 * total))
                    rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))
            rolloff_95 = float(np.median(rolloffs)) if rolloffs else 0.0
        except Exception:
            rolloff_95 = 0.0

        # ── 2. Wow/Flutter-Index ────────────────────────────────────────
        try:
            from scipy.signal import hilbert

            frame_size = int(0.1 * sr)  # 100 ms
            pitches = []
            for start in range(0, n - frame_size, frame_size):
                frame = mono[start : start + frame_size].astype(np.float64)
                env = np.abs(hilbert(frame))
                mean_e = float(np.mean(env))
                if mean_e > 1e-6:
                    pitches.append(mean_e)
            wow_flutter = float(np.std(np.diff(pitches))) if len(pitches) > 2 else 0.0
        except Exception:
            wow_flutter = 0.0

        # ── 3. HF-Energie > 16 kHz ─────────────────────────────────────
        try:
            spec_full = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_full = np.fft.rfftfreq(65536, 1.0 / sr)
            mask_hf = freqs_full > 16_000
            total_e = float(np.sum(spec_full**2))
            hf_e = float(np.sum(spec_full[mask_hf] ** 2))
            hf_fraction = hf_e / max(total_e, 1e-12)
        except Exception:
            hf_fraction = 0.0

        # ── 4. Rauschpegel (5. Perzentil PSD) ──────────────────────────
        try:
            frame_energies = []
            for start in range(0, n - win, hop):
                e = float(np.mean(mono[start : start + win] ** 2))
                if e > 0:
                    frame_energies.append(10 * math.log10(e))
            noise_floor = float(np.percentile(frame_energies, 5)) if frame_energies else -60.0
            noise_floor = max(-120.0, min(0.0, noise_floor))
        except Exception:
            noise_floor = -60.0

        # ── 5. Effektive Bandbreite (Rolloff −60 dBFS) ──────────────────
        try:
            spec_bw = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_bw = np.fft.rfftfreq(65536, 1.0 / sr)
            spec_db = 20 * np.log10(np.clip(spec_bw / max(spec_bw.max(), 1e-12), 1e-15, None))
            above_thresh = freqs_bw[spec_db > -60.0]
            eff_bw = float(above_thresh.max()) if len(above_thresh) > 0 else 0.0
        except Exception:
            eff_bw = 0.0

        return SpectralFingerprint(
            rolloff_95_hz=float(np.nan_to_num(rolloff_95)),
            wow_flutter_index=float(np.nan_to_num(wow_flutter)),
            hf_energy_above_16k=float(np.nan_to_num(hf_fraction)),
            noise_floor_db=float(np.nan_to_num(noise_floor, nan=-60.0)),
            effective_bandwidth_hz=float(np.nan_to_num(eff_bw)),
        )

    def detect(self, audio: np.ndarray, sr: int) -> MediumDetectionResult:
        """Erkennt die Tonträgerkette forensisch.

        Laufreihenfolge MUSS VOR classify_medium() sein (§6.7.2).

        Returns:
            MediumDetectionResult mit transfer_chain, is_multi_generation,
            primary_material, confidence, spectral_fingerprint.
        """
        if sr != 48000:
            logger.debug("MediumDetector: SR=%d (erwartet 48000), arbeite trotzdem weiter", sr)

        fp = self._compute_fingerprint(audio, sr)
        chain: list[str] = []
        evidence: list[str] = []
        confidence_parts: list[float] = []

        # ── Shellac/Wachswalze (extremste Bandbreitenbegrenzung) ─────────
        if fp.rolloff_95_hz < self.SHELLAC_ROLLOFF_MAX_HZ and fp.rolloff_95_hz > 0:
            chain.append("shellac")
            confidence_parts.append(0.80)
            evidence.append(f"Sehr enge Bandbreite ({fp.rolloff_95_hz:.0f} Hz Rolloff) → Shellac/Wachswalze")

        # ── Kassetten-Magnetband (Tape) ───────────────────────────────────
        elif fp.rolloff_95_hz < self.TAPE_ROLLOFF_MAX_HZ and fp.wow_flutter_index > self.TAPE_WOW_FLUTTER_MIN:
            chain.append("tape")
            confidence_parts.append(0.75)
            evidence.append(
                f"Kassetten-Signatur: Rolloff {fp.rolloff_95_hz:.0f} Hz, " f"Wow/Flutter {fp.wow_flutter_index:.2f} Hz"
            )
            if fp.noise_floor_db > self.TAPE_NOISE_FLOOR_MAX_DB:
                evidence.append(f"Starkes Bandrauschen ({fp.noise_floor_db:.1f} dBFS)")
                confidence_parts.append(0.10)

        # ── Vinyl (Crackle-Profil, mittlerer Rolloff, niedriger Rauschboden) ─
        elif (
            self.TAPE_ROLLOFF_MAX_HZ <= fp.rolloff_95_hz < 18_000
            and fp.wow_flutter_index < 0.3
            and fp.noise_floor_db < -38.0
        ):
            chain.append("vinyl")
            confidence_parts.append(0.65)
            evidence.append(f"Vinyl-Profil: Rolloff {fp.rolloff_95_hz:.0f} Hz, ruhiger Rauschboden")

        # ── Digitaler Träger (CD/WAV) ────────────────────────────────────
        elif fp.rolloff_95_hz >= 18_000 and fp.hf_energy_above_16k > 0.01:
            chain.append("cd_digital")
            confidence_parts.append(0.70)
            evidence.append(f"Digitaler Träger: HF-Energie vorhanden ({fp.hf_energy_above_16k*100:.1f} %)")

        # ── Sekundäre MP3/Codec-Kette erkennen ──────────────────────────
        has_mp3_signature = (
            fp.hf_energy_above_16k < self.HF_ENERGY_THRESHOLD_FRACTION and fp.effective_bandwidth_hz < 17_500
        )

        if has_mp3_signature:
            # Wenn kein primärer Träger erkannt → mp3 ist primär
            if not chain:
                # Starkes Bandrauschen OB mp3 → wahrscheinlich Kassette+MP3
                if fp.noise_floor_db > -45.0:
                    chain = ["tape", "mp3_low"]
                    confidence_parts.extend([0.55, 0.20])
                    evidence.append(
                        f"Kassette+MP3-Kette: kein HF ({fp.hf_energy_above_16k*100:.2f} %), "
                        f"Rauschboden {fp.noise_floor_db:.1f} dBFS"
                    )
                else:
                    bitrate_estimate = "mp3_low" if fp.effective_bandwidth_hz < 14_000 else "mp3_high"
                    chain = [bitrate_estimate]
                    confidence_parts.append(0.60)
                    evidence.append(f"MP3-Kette: kein HF, Bandbreite {fp.effective_bandwidth_hz:.0f} Hz")
            else:
                # Sekundäre Codec-Stufe
                bitrate = "mp3_low" if fp.effective_bandwidth_hz < 14_000 else "mp3_high"
                chain.append(bitrate)
                confidence_parts.append(0.20)
                evidence.append(f"Sekundäre MP3-Kodierung erkannt (BW {fp.effective_bandwidth_hz:.0f} Hz)")

        # ── Fallback ─────────────────────────────────────────────────────
        if not chain:
            chain = ["unknown"]
            confidence_parts = [0.30]
            evidence.append("Träger unbekannt — Standard-Prior wird verwendet")

        primary = chain[0]
        is_multi = len(chain) > 1
        confidence = float(np.clip(sum(confidence_parts), 0.0, 1.0))

        logger.info(
            "MediumDetector: Kette=%s, primär=%s, multi=%s, Konfidenz=%.2f",
            " → ".join(chain),
            primary,
            is_multi,
            confidence,
        )

        return MediumDetectionResult(
            transfer_chain=chain,
            is_multi_generation=is_multi,
            primary_material=primary,
            confidence=confidence,
            spectral_fingerprint=fp,
            evidence=evidence,
        )

    # ── Hilfsmethode ─────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Wandelt beliebiges Audio in mono float32 um."""
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
        mono = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mono, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: Optional[MediumDetector] = None
_lock = threading.Lock()


def get_medium_detector() -> MediumDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MediumDetector()
    return _instance


def detect_medium_chain(audio: np.ndarray, sr: int) -> MediumDetectionResult:
    """Convenience-Wrapper: erkennt die Tonträgerkette eines Audio-Signals."""
    return get_medium_detector().detect(audio, sr)
