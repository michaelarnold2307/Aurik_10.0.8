"""Carrier Transfer Characteristics — Single Source of Truth (§4.8, §6.2b, §6.2c).

Kanonische Tabelle aller 16 Materialtypen mit physikalischen Grenzwerten.
Alle Module importieren aus dieser Datei — niemals duplizieren.

Version: 9.12.0 — instructions_version 9.0
Changelog: tape DR-Ceiling 68→62 dB (Bug #4/#5): MediumDetector normiert cassette→tape;
  tape repräsentiert Kompaktkassette (Typ I ~55 dB, Typ II ~65 dB, Mittel ≈62 dB);
  nicht Reel-Tape (reel_tape = 72 dB). Inkonsistenz mit §0a Kassette-Ceiling behoben.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch as _welch

# (bw_ceiling_hz, snr_floor_db, generation_loss_db_per_gen, dr_ceiling_db)
CARRIER_TRANSFER_CHARACTERISTICS: dict[str, tuple[int, int, float, int]] = {
    "wax_cylinder": (5000, -25, -6.0, 35),
    "shellac": (8000, -30, -5.0, 45),
    "lacquer_disc": (8000, -32, -4.5, 50),
    "wire_recording": (6000, -28, -5.5, 40),
    "vinyl": (16000, -55, -2.0, 70),
    "tape": (
        15000,
        -50,
        -3.0,
        62,
    ),  # Kassette: DR Typ I ~55 dB, Typ II ~65 dB, Mittel ≈62 dB; cassette→tape-Alias (§0a, §Bug#4)
    "reel_tape": (18000, -60, -1.5, 72),
    "cassette": (14000, -48, -3.5, 60),
    "dat": (22000, -90, -0.2, 92),
    "minidisc": (20000, -85, -0.5, 88),
    "cd_digital": (22050, -96, -0.1, 96),
    "mp3_low": (16000, -70, -1.5, 90),
    "mp3_high": (20000, -80, -0.5, 93),
    "aac": (20000, -82, -0.4, 93),
    "streaming": (20000, -78, -0.8, 90),
    "unknown": (20000, -50, -2.0, 70),
}

# §6.2c — Material-Bandwidth-Ceiling (abgeleitet aus Spalte 0)
MATERIAL_BW_CEILING_HZ: dict[str, int] = {k: v[0] for k, v in CARRIER_TRANSFER_CHARACTERISTICS.items()}

# §6.2b — Material-Dynamic-Range-Ceiling (abgeleitet aus Spalte 3)
MATERIAL_DR_CEILING_DB: dict[str, int] = {k: v[3] for k, v in CARRIER_TRANSFER_CHARACTERISTICS.items()}

# §4.8 — SNR-Floor (abgeleitet aus Spalte 1)
MATERIAL_SNR_FLOOR_DB: dict[str, int] = {k: v[1] for k, v in CARRIER_TRANSFER_CHARACTERISTICS.items()}


def compute_cumulative_generation_loss(transfer_chain: list[str]) -> float:
    """§4.8 — Berechnet den kumulativen HF-Verlust einer Tonträgerkette.

    Args:
        transfer_chain: Materialkette, z.B. ['shellac', 'reel_tape', 'cd_digital'].

    Returns:
        Kumulativer HF-Verlust in dB (negativer Wert = Verlust).
    """
    total_loss = 0.0
    for material in transfer_chain:
        chars = CARRIER_TRANSFER_CHARACTERISTICS.get(material)
        if chars is not None:
            total_loss += chars[2]  # generation_loss_db_per_gen
    return total_loss


def get_bw_ceiling_hz(material_type: str) -> int:
    """§6.2c — Gibt das BW-Ceiling für ein Material zurück (Default: 20000 Hz)."""
    return MATERIAL_BW_CEILING_HZ.get(material_type, 20000)


def get_chain_bw_ceiling_hz(transfer_chain: list[str]) -> int:
    """§6.2c — Minimum-BW-Ceiling über die gesamte Trägerkette.

    Verhindert, dass AudioSR/Phase_06 über den schwächsten Kettenstufen-Ceiling
    hinaus Obertöne halluziniert. Beispiel: vinyl→mp3_low → min(16000, 11000) = 11000 Hz.
    """
    if not transfer_chain:
        return 20000
    return min(MATERIAL_BW_CEILING_HZ.get(m, 20000) for m in transfer_chain)


def get_dr_ceiling_db(material_type: str) -> int:
    """§6.2b — Gibt das DR-Ceiling für ein Material zurück (Default: 70 dB)."""
    return MATERIAL_DR_CEILING_DB.get(material_type, 70)


def spectral_correlation(audio1: np.ndarray, audio2: np.ndarray, sr: int = 48000) -> float:
    """§0d — Spectral correlation ∈ [0, 1] between two audio signals.

    Uses log-PSD (Welch) cosine-similarity. Returns 1.0 for identical spectra,
    0.0 for orthogonal, and values near 0 for heavily carrier-inverted signals.

    carrier_chain_recovery_ratio = 1.0 - spectral_correlation(pre, post)
    """

    def _to_mono(a: np.ndarray) -> np.ndarray:
        if a.ndim == 2:
            if a.shape[0] == 2:
                return np.asarray((a[0] + a[1]) / 2.0, dtype=np.float64)
            if a.shape[-1] == 2:
                return np.asarray((a[:, 0] + a[:, 1]) / 2.0, dtype=np.float64)
        return a.ravel()

    m1 = np.asarray(_to_mono(audio1), dtype=np.float64)
    m2 = np.asarray(_to_mono(audio2), dtype=np.float64)

    min_len = min(len(m1), len(m2))
    if min_len < 2048:
        return 1.0  # too short to evaluate

    m1 = m1[:min_len]
    m2 = m2[:min_len]

    nperseg = min(4096, min_len)
    _, psd1 = _welch(m1, fs=sr, nperseg=nperseg)
    _, psd2 = _welch(m2, fs=sr, nperseg=nperseg)

    log1 = np.log1p(psd1)
    log2 = np.log1p(psd2)

    # Guarded dot-product correlation (NaN-safe, no np.corrcoef)
    dot = float(np.dot(log1, log2))
    norm1 = float(np.sqrt(np.dot(log1, log1)))
    norm2 = float(np.sqrt(np.dot(log2, log2)))
    denom = norm1 * norm2 + 1e-12
    corr = dot / denom

    return float(np.clip(corr, 0.0, 1.0))


def compute_carrier_recovery_ratio(audio_pre: np.ndarray, audio_post: np.ndarray, sr: int = 48000) -> float:
    """§0d — NR-aware carrier-chain recovery ratio.

    Kombiniert zwei Messgrössen:

    1. **Spectral-shape ratio** (1 − spectral_correlation):
       Sensitiv für grosse Spektralveränderungen: BW-Extension (phase_06/07),
       RIAA-Inversion (phase_04), Heavy-EQ (phase_16).

    2. **Noise-floor-delta ratio**:
       ``spectral_correlation`` nutzt log-PSD — Musikinhalt dominiert absolut.
       Hiss-Entfernung (phase_03/29, cassette/tape) ändert die Spektralform
       kaum (Hiss-Leistung ≈ 10⁻⁶ relativ zu Musikpegel → corr ≈ 0.997).
       Das Noise-Floor-Delta detektiert NR-Recovery, die ``spectral_correlation``
       ignoriert:
       - 5. Perzentil der Welch-PSD ≈ Rauschboden-Schätzung
       - SNR-Verbesserung ≥ 12 dB → ratio ≥ 0.15 → §0d aktiviert

    Kalibrierung (§0d-Schwelle 0.15):
       * 12 dB NR-Improvement → _nr_ratio = 0.15 → §0d aktiviert
       * 20 dB NR-Improvement → _nr_ratio = 0.25 → §0d sicher aktiv
       * < 6 dB Änderung → kein §0d-Trigger

    Returns:
        ratio in [0.0, 1.0]: > 0.15 → §0d carrier-reference shift aktiviert.
    """
    # Component 1: spectral-shape change
    _spec_ratio = float(np.clip(1.0 - spectral_correlation(audio_pre, audio_post, sr=sr), 0.0, 1.0))

    # Component 2: noise-floor change (NR-sensitive)
    def _to_mono(a: np.ndarray) -> np.ndarray:
        if a.ndim == 2:
            if a.shape[0] == 2:
                return np.asarray((a[0] + a[1]) / 2.0, dtype=np.float64)
            return np.asarray((a[:, 0] + a[:, 1]) / 2.0, dtype=np.float64)
        return a.ravel()

    m_pre = np.asarray(_to_mono(audio_pre), dtype=np.float64)
    m_post = np.asarray(_to_mono(audio_post), dtype=np.float64)
    min_len = min(len(m_pre), len(m_post))

    _nr_ratio = 0.0
    if min_len >= 4096:
        try:
            nperseg = min(4096, min_len)
            _, psd_pre = _welch(m_pre[:min_len], fs=sr, nperseg=nperseg)
            _, psd_post = _welch(m_post[:min_len], fs=sr, nperseg=nperseg)
            # 5th-percentile PSD ≈ noise floor estimate (below musical peaks)
            _floor_pre = float(np.percentile(psd_pre + 1e-30, 5))
            _floor_post = float(np.percentile(psd_post + 1e-30, 5))
            if _floor_pre > _floor_post * 1.5:  # floor dropped ≥ 3.5 dB
                # Scale: 12 dB NR → ratio 0.15 (§0d threshold); 80 dB → 1.0 (capped 0.40)
                _nr_db = float(10.0 * np.log10(_floor_pre / (_floor_post + 1e-30)))
                _nr_ratio = float(np.clip(_nr_db / 80.0, 0.0, 0.40))
        except Exception:
            pass  # non-blocking: spectral_correlation fallback is sufficient

    return float(np.clip(_spec_ratio + _nr_ratio, 0.0, 1.0))
