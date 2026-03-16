"""
Phase 43: DSP De-Esser v2.1 — Stimmtyp-adaptiver Sidechain-De-Esser
=====================================================================

Vollständige DSP-Implementierung ohne aurik_ml.
Stimmtyp-adaptive Frequenzauswahl gemäß §2.8 (Vocal-Restaurierungskette).

ALGORITHMUS — Split-Band De-Esser:
  1. Sibilantenband extrahieren: Butterworth-Bandpass 4. Ordnung, gender-adaptiv
  2. Hüllkurve des Sibilantenbands via RMS-Fenster (5 ms)
  3. Gain Reduction: wenn Hüllkurve > threshold, Kompression 1:4
     GR = (threshold / envelope)^((ratio-1)/ratio)  → logarithmisch glatt
  4. Smooth Gain per Sample: Attack 2 ms, Release 80 ms
  5. Strength-Cap: GR >= strength_cap (verhindert Überdämpfung bei Schlager-Modus)
  6. Gefilterte Band × Gain → vom Original subtrahieren
  7. Funktioniert Mono + Stereo (channelweise)

PARAMETER (kwargs):
  threshold_db  (float, default -20.0)  — Detektionsschwelle in dBFS
  ratio         (float, default 4.0)    — Kompressionsverhältnis (1:ratio)
  attack_ms     (float, default 2.0)    — Gain-Attack in ms
  release_ms    (float, default 80.0)   — Gain-Release in ms
  freq_low      (float, optional)       — Untere Sibilanzgrenze Hz (überschreibt gender)
  freq_high     (float, optional)       — Obere Sibilanzgrenze Hz (überschreibt gender)
  gender        (str, default "unknown") — Stimmtyp: "male"|"female"|"child"|"unknown"
  strength_cap  (float, default 1.0)    — Max. GR-Stärke 0.0–1.0 (§2.19.3 Schlager: 0.45)

STIMMTYP-ADAPTIVE FREQUENZEN (§2.8):
  male:    5 000 – 10 000 Hz
  female:  6 000 – 12 000 Hz
  child:   7 000 – 14 000 Hz
  unknown: 5 000 –  9 000 Hz  (konservativ)

Author: Aurik Development Team
Version: 2.1.0
"""

from __future__ import annotations

import logging
import time

import numpy as np
import scipy.signal as sig

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stimmtyp-adaptive Sibilanz-Frequenzgrenzen (§2.8 Vocal-Restaurierungskette)
# ---------------------------------------------------------------------------
#  MALE:    5 –10 kHz  (tiefer Grundton, breitere Konsonanten)
#  FEMALE:  6 –12 kHz  (höherer Grundton, scharfe Frikative)
#  CHILD:   7 –14 kHz  (höchster Grundton, sehr hohe Sibilanz)
#  unknown: 5 – 9 kHz  (konservativer Fallback)
GENDER_FREQ_MAP: dict[str, tuple[float, float]] = {
    "male":    (5_000.0, 10_000.0),
    "female":  (6_000.0, 12_000.0),
    "child":   (7_000.0, 14_000.0),
    "unknown": (5_000.0,  9_000.0),
}

_DEFAULT_THRESHOLD_DB = -20.0
_DEFAULT_RATIO = 4.0
_DEFAULT_ATTACK_MS = 2.0
_DEFAULT_RELEASE_MS = 80.0
_DEFAULT_GENDER = "unknown"
_DEFAULT_STRENGTH_CAP = 1.0  # kein Cap; §2.19.3 Schlager-Modus: 0.45


def _rms_envelope(signal: np.ndarray, sr: int, window_ms: float = 5.0) -> np.ndarray:
    """RMS-Hüllkurve mit gleitendem Fenster."""
    win = max(2, int(window_ms / 1000.0 * sr))
    sq = signal**2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-12)
    return rms


def _smooth_gain(gain_lin: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Exponentielles Glättung der Gain-Kurve (Attack + Release)."""
    att = np.exp(-1.0 / (attack_ms / 1000.0 * sr + 1e-6))
    rel = np.exp(-1.0 / (release_ms / 1000.0 * sr + 1e-6))
    smoothed = np.ones_like(gain_lin)
    for i in range(1, len(gain_lin)):
        if gain_lin[i] < smoothed[i - 1]:
            coef = att  # Gain fällt (mehr Dämpfung) → schnell
        else:
            coef = rel  # Gain steigt (Erholung) → langsam
        smoothed[i] = coef * smoothed[i - 1] + (1.0 - coef) * gain_lin[i]
    return smoothed


def _estimate_breathiness(audio: np.ndarray, sr: int) -> float:
    """Schätzt Breathiness anhand des Spektralabfalls (0.0 = klar, 1.0 = sehr hauchig).

    Proxy: Spektraler Slope (dB/Oktave) im Vokalbereich 500–6000 Hz.
    Slope ≈ −6 dB/Okt → normalsprachlich (breathiness=0.0)
    Slope < −18 dB/Okt → stark hauchig (breathiness→1.0)
    """
    mono = audio if audio.ndim == 1 else (audio[:, 0] if audio.ndim == 2 else audio.mean(axis=0))
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    n_fft = min(4096, len(mono))
    if n_fft < 64:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= 500.0) & (freqs <= 6000.0) & (spectrum > 1e-10)
    if mask.sum() < 4:
        return 0.0
    log_freqs = np.log2(freqs[mask] + 1e-6)
    log_amps = 20.0 * np.log10(spectrum[mask] + 1e-10)
    slope = float(np.polyfit(log_freqs, log_amps, 1)[0])  # dB/Oktave
    # slope ≈ −6 dB/Okt → Schwellwert; slope < −18 dB/Okt → Maximalhauchigkeit
    return float(np.clip((-slope - 6.0) / 12.0, 0.0, 1.0))


def _deess_channel(
    ch: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    freq_low: float,
    freq_high: float,
    strength_cap: float = 1.0,
) -> tuple[np.ndarray, float]:
    """De-Esser auf einem Mono-Kanal. Gibt (processed, avg_gain_reduction_db) zurück.

    Sibilantenband-Extraktion via sosfiltfilt (Zero-Phase / §4.5) — vermeidet
    Phasenversatz zwischen Original und Bandpass-Signal.
    """
    # 1. Sibilantenband — Zero-Phase-Filter (sosfiltfilt, offline-verarbeitung)
    sos = sig.butter(4, [freq_low, freq_high], btype="band", fs=sr, output="sos")
    try:
        sib_band = sig.sosfiltfilt(sos, ch)
    except ValueError:
        # Fallback für sehr kurze Signale (< filter-Transiente)
        sib_band = sig.sosfilt(sos, ch)

    # 2. Hüllkurve
    envelope = _rms_envelope(sib_band, sr, 5.0)

    # 3. Gain Reduction (linker Arm: über Schwelle → komprimieren)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    gr = np.where(
        envelope > threshold_lin,
        (threshold_lin / (envelope + 1e-12)) ** ((ratio - 1.0) / ratio),
        1.0,
    )

    # 4. Smooth
    gr_smooth = _smooth_gain(gr, sr, attack_ms, release_ms)

    # 5. Strength-Cap (§2.19.3): GR darf nicht stärker als strength_cap
    #    strength_cap = 1.0 → kein Cap; = 0.45 → max. 55 % GR
    if strength_cap < 1.0:
        gr_smooth = np.maximum(gr_smooth, strength_cap)

    # 6. Anwenden: Sibilantenband dämpfen, zum Restsignal addieren
    processed = ch - sib_band + sib_band * gr_smooth

    avg_gr_db = float(np.mean(20.0 * np.log10(gr_smooth + 1e-12)))
    return processed.astype(ch.dtype), avg_gr_db


class MLDeEsserPhase(PhaseInterface):
    """Stimmtyp-adaptiver Sidechain-De-Esser (DSP, kein aurik_ml, §2.8)."""

    phase_id = "phase_43_ml_deesser"
    name = "De-Esser (Sidechain DSP, stimmtyp-adaptiv)"
    description = (
        "Split-Band De-Esser mit Butterworth-Bandpass gender-adaptiver Frequenzauswahl "
        "(§2.8: MALE 5–10 kHz / FEMALE 6–12 kHz / CHILD 7–14 kHz). "
        "RMS-Hüllkurve, Gain-Reduction 1:4, Attack 2 ms / Release 80 ms, Strength-Cap. "
        "Kein aurik_ml."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.phase_id,
            name=self.name,
            category=PhaseCategory.ENHANCEMENT,
            priority=6,
            version="2.1.0",
            dependencies=[],
            estimated_time_factor=0.04,
            memory_requirement_mb=50,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.88,
            description=self.description,
        )

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult:
        """
        De-Essing: Sibilanten reduzieren (stimmtyp-adaptiv, §2.8).

        Args:
            audio:        Mono oder Stereo float32/64
            sample_rate:  Hz (muss 48 000 sein)
            **kwargs:
                gender        (str)   "male"|"female"|"child"|"unknown"
                threshold_db  (float) Detektionsschwelle dBFS, Default -20.0
                ratio         (float) Kompressionsverhältnis, Default 4.0
                attack_ms     (float) Attack in ms, Default 2.0
                release_ms    (float) Release in ms, Default 80.0
                strength_cap  (float) Max. GR-Stärke 0–1: §2.19.3 Schlager 0.45
                freq_low      (float) Überschreibt gender-Auswahl (Hz)
                freq_high     (float) Überschreibt gender-Auswahl (Hz)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        t0 = time.time()

        # Parameter
        gender: str = str(kwargs.get("gender", _DEFAULT_GENDER)).lower()
        threshold_db: float = float(kwargs.get("threshold_db", _DEFAULT_THRESHOLD_DB))
        ratio: float = float(kwargs.get("ratio", _DEFAULT_RATIO))
        attack_ms: float = float(kwargs.get("attack_ms", _DEFAULT_ATTACK_MS))
        release_ms: float = float(kwargs.get("release_ms", _DEFAULT_RELEASE_MS))
        strength_cap: float = float(kwargs.get("strength_cap", _DEFAULT_STRENGTH_CAP))
        strength_cap = float(np.clip(strength_cap, 0.0, 1.0))

        # Stimmtyp-adaptive Frequenzauswahl (§2.8); explizite freq_low/freq_high überschreiben
        default_low, default_high = GENDER_FREQ_MAP.get(gender, GENDER_FREQ_MAP["unknown"])
        freq_low: float = float(kwargs.get("freq_low", default_low))
        freq_high: float = float(kwargs.get("freq_high", default_high))

        # Nyquist-Sicherung
        nyquist = sample_rate / 2.0
        freq_high = min(freq_high, nyquist * 0.98)
        freq_low = min(freq_low, freq_high * 0.90)

        x = audio.astype(np.float64)

        # Breathiness-Guard (§2.8): Hauchige Stimmen (spectral slope < −18 dB/Okt)
        # dürfen nicht über-de-esst werden. Bei breathiness > 0.4 wird strength_cap
        # auf max. 0.50–0.60 begrenzt, um hauchige Vokale natürlich zu erhalten.
        _breathiness = _estimate_breathiness(x, sample_rate)
        if _breathiness > 0.4:
            _breath_cap = float(np.clip(0.60 - (_breathiness - 0.4) * 0.10, 0.50, 1.0))
            strength_cap = max(strength_cap, _breath_cap)
            logger.info(
                "Phase 43 Breathiness-Guard: breathiness=%.2f → strength_cap angepasst auf %.2f",
                _breathiness, strength_cap,
            )

        gr_dbs: list[float] = []

        if x.ndim == 1:
            processed_ch, gr_db = _deess_channel(
                x, sample_rate, threshold_db, ratio, attack_ms, release_ms,
                freq_low, freq_high, strength_cap,
            )
            processed = processed_ch
            gr_dbs.append(gr_db)
        else:
            channels = []
            for ch in range(x.shape[1]):
                processed_ch, gr_db = _deess_channel(
                    x[:, ch], sample_rate, threshold_db, ratio, attack_ms, release_ms,
                    freq_low, freq_high, strength_cap,
                )
                channels.append(processed_ch)
                gr_dbs.append(gr_db)
            processed = np.column_stack(channels)

        processed = np.clip(processed, -1.0, 1.0).astype(audio.dtype)
        avg_gr = float(np.mean(gr_dbs))

        logger.info(
            "Phase 43 DeEsser: gender=%s freq=[%.0f–%.0f Hz] "
            "threshold=%.1f dB ratio=%.1f strength_cap=%.2f avg_GR=%.2f dB",
            gender, freq_low, freq_high, threshold_db, ratio, strength_cap, avg_gr,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed,
            execution_time_seconds=time.time() - t0,
            metadata={
                "gender": gender,
                "threshold_db": threshold_db,
                "ratio": ratio,
                "attack_ms": attack_ms,
                "release_ms": release_ms,
                "freq_low_hz": freq_low,
                "freq_high_hz": freq_high,
                "strength_cap": strength_cap,
            },
            metrics={"avg_gain_reduction_db": avg_gr},
        )
