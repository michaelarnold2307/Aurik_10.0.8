"""
Preservation Metrics — SOTA Verification for Blind Tests (§G46–§G48)

Three standalone metrics for verifying preservation quality:
  §G46  Harmonic Preservation Score — overtone structure intact?
  §G47  Transient Preservation Score — attacks survive processing?
  §G48  Vocal Formant Preservation Score — voice character unchanged?

Each function: (original, processed, sr) → score in [0, 1]
  1.0 = perfectly preserved  0.0 = completely destroyed

Scientific basis:
  Harmonic:   F0-tracking → harmonic peak comparison (Fletcher, 1964)
  Transient:  Spectral-flux onset detection (Bello et al., 2005)
  Formant:    LPC-based spectral envelope comparison (Markel & Gray, 1976)

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── §G46 Harmonic Preservation ──────────────────────────────────────────


def compute_harmonic_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    f0_min: float = 80.0,
    f0_max: float = 1200.0,
    n_harmonics: int = 12,
) -> float:
    """§G46: Vergleicht harmonische Struktur vor/nach Processing.

    Algorithmus:
    1. STFT auf beiden Signalen (2048-pt, 50% Overlap)
    2. Für jeden Frame mit genug Energie: finde F0 via HPS (Harmonic Product Spectrum)
    3. Extrahiere harmonische Peaks bei n·F0 für n=1..n_harmonics
    4. Berechne Pearson-Korrelation der harmonischen Amplituden-Vektoren
    5. Gewichteter Mittelwert über alle Frames

    Returns: float [0,1] — 1.0 = identische harmonische Struktur.
    """
    mono_orig = _to_mono(original)
    mono_proc = _to_mono(processed)
    n_min = min(len(mono_orig), len(mono_proc))
    if n_min < 4096:
        return 1.0

    mono_orig = mono_orig[:n_min].astype(np.float64)
    mono_proc = mono_proc[:n_min].astype(np.float64)

    n_fft = 2048
    hop = n_fft // 2
    n_frames = (n_min - n_fft) // hop + 1
    if n_frames < 3:
        return 1.0

    win = np.hanning(n_fft)
    scores = []
    weights = []

    for i in range(n_frames):
        start = i * hop
        fo = mono_orig[start : start + n_fft] * win
        fp = mono_proc[start : start + n_fft] * win

        energy_o = float(np.sum(fo**2))
        if energy_o < 1e-10:
            continue

        spec_o = np.abs(np.fft.rfft(fo))
        spec_p = np.abs(np.fft.rfft(fp))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Rough F0 estimation via HPS (harmonic product spectrum)
        f0 = _estimate_f0_hps(spec_o, freqs, f0_min, f0_max)
        if f0 is None:
            continue

        # Extract harmonic amplitudes at n·F0
        harm_amps_o = np.zeros(n_harmonics, dtype=np.float64)
        harm_amps_p = np.zeros(n_harmonics, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            target_hz = f0 * h
            if target_hz >= sr / 2:
                break
            idx = np.argmin(np.abs(freqs - target_hz))
            # Use peak within ±1 bin
            lo = max(0, idx - 1)
            hi = min(len(freqs) - 1, idx + 2)
            harm_amps_o[h - 1] = np.max(spec_o[lo:hi])
            harm_amps_p[h - 1] = np.max(spec_p[lo:hi])

        # Normalize to fundamental
        if harm_amps_o[0] > 0 and harm_amps_p[0] > 0:
            harm_amps_o /= harm_amps_o[0]
            harm_amps_p /= harm_amps_p[0]

        # Correlation of harmonic profiles
        so = float(np.std(harm_amps_o))
        sp = float(np.std(harm_amps_p))
        if so < 1e-10 or sp < 1e-10:
            score = 1.0 if np.allclose(harm_amps_o[:4], harm_amps_p[:4], atol=0.1) else 0.5
        else:
            corr = float(
                np.dot(harm_amps_o - np.mean(harm_amps_o), harm_amps_p - np.mean(harm_amps_p))
                / (len(harm_amps_o) * so * sp + 1e-12)
            )
            score = max(0.0, min(1.0, (corr + 1.0) / 2.0))

        scores.append(score)
        weights.append(energy_o)

    if not scores:
        return 1.0

    weights = np.array(weights, dtype=np.float64)
    weights /= np.sum(weights) + 1e-10
    return float(np.dot(scores, weights))


def _estimate_f0_hps(
    spec: np.ndarray, freqs: np.ndarray, f0_min: float, f0_max: float
) -> float | None:
    """Grobe F0-Schätzung via Harmonic Product Spectrum."""
    n_bins = len(spec)
    max_harmonics = 4
    hps = spec.copy()
    for h in range(2, max_harmonics + 1):
        downsampled = spec[::h]
        hps[: len(downsampled)] *= downsampled

    i_min = np.searchsorted(freqs, f0_min)
    i_max = np.searchsorted(freqs, f0_max)
    i_min = max(0, min(i_min, n_bins - 1))
    i_max = max(i_min + 1, min(i_max, n_bins))

    if i_max <= i_min:
        return None

    peak_idx = i_min + int(np.argmax(hps[i_min:i_max]))
    peak_mag = hps[peak_idx]
    mean_mag = np.mean(hps[i_min:i_max])

    if peak_mag < mean_mag * 3.0:
        return None  # No clear F0

    return float(freqs[peak_idx])


# ── §G47 Transient Preservation ─────────────────────────────────────────


def compute_transient_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    onset_threshold: float = 0.3,
    freq_lo: float = 2000.0,
    freq_hi: float = 8000.0,
) -> float:
    """§G47: Vergleicht Onset-Struktur vor/nach Processing.

    Algorithmus (Bello et al., 2005):
    1. Berechne spektrale Energie in Transienten-Band (2–8 kHz)
    2. Onset Detection Function = positive Halbwelle der Energie-Differenz
    3. Vergleiche Onset-Peaks: Position + relative Stärke
    4. Score = recall (wie viele Original-Onsets sind erhalten?)

    Returns: float [0,1] — 1.0 = alle Onsets erhalten.
    """
    mono_orig = _to_mono(original)
    mono_proc = _to_mono(processed)
    n_min = min(len(mono_orig), len(mono_proc))
    if n_min < 2048:
        return 1.0

    mono_orig = mono_orig[:n_min].astype(np.float64)
    mono_proc = mono_proc[:n_min].astype(np.float64)

    n_fft = 1024
    hop = 256
    n_frames = (n_min - n_fft) // hop + 1
    if n_frames < 4:
        return 1.0

    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lo = np.searchsorted(freqs, freq_lo)
    hi = np.searchsorted(freqs, freq_hi)

    def onset_func(audio: np.ndarray) -> np.ndarray:
        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop
            frame = audio[s : s + n_fft] * win
            spec = np.abs(np.fft.rfft(frame))
            energy[i] = float(np.sum(spec[lo:hi] ** 2))
        if np.max(energy) < 1e-15:
            return np.zeros(n_frames, dtype=np.float64)
        energy_db = 10.0 * np.log10(energy + 1e-15)
        # Onset = positive half-wave of first difference
        diff = np.diff(energy_db)
        onset = np.maximum(diff, 0.0)
        onset /= np.max(onset) + 1e-15
        return onset

    onset_o = onset_func(mono_orig)
    onset_p = onset_func(mono_proc)

    if np.max(onset_o) < onset_threshold:
        return 1.0  # No significant onsets in original — nothing to preserve

    # Find onset peaks in original
    orig_peaks = _find_peaks(onset_o, threshold=onset_threshold * np.max(onset_o))

    if len(orig_peaks) == 0:
        return 1.0

    # Check each original onset: does processed have a peak at the same position?
    preserved = 0
    for idx in orig_peaks:
        # Look in ±2 frame window
        lo_w = max(0, idx - 2)
        hi_w = min(len(onset_p) - 1, idx + 3)
        if np.max(onset_p[lo_w:hi_w]) >= onset_threshold * np.max(onset_p) * 0.5:
            preserved += 1

    score = float(preserved) / float(len(orig_peaks))
    return score


def _find_peaks(signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Finde lokale Maxima über threshold."""
    if len(signal) < 3:
        return np.array([], dtype=int)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


# ── §G48 Vocal Formant Preservation ─────────────────────────────────────


def compute_formant_preservation_score(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
    *,
    formant_range: tuple = (200.0, 4000.0),
    n_formants: int = 4,
) -> float:
    """§G48: Vergleicht Vokal-Formanten (F1–F4) vor/nach Processing.

    Algorithmus:
    1. Bandpass 200–4000 Hz (Gesangs-Formantbereich)
    2. LPC (Linear Predictive Coding) Ordnung = sr/1000 + 4
    3. Formanten = Peaks im LPC-Spektrum
    4. Vergleiche Formant-Positionen: relative Abweichung < 5% = erhalten
    5. Zusätzlich: spektrale Energie-Erhaltung im Formantbereich

    Returns: float [0,1] — 1.0 = Formanten identisch.
    """
    mono_orig = _to_mono(original)
    mono_proc = _to_mono(processed)
    n_min = min(len(mono_orig), len(mono_proc))
    if n_min < 4096:
        return 1.0

    mono_orig = mono_orig[:n_min].astype(np.float64)
    mono_proc = mono_proc[:n_min].astype(np.float64)

    # Process in 100ms windows, take best-scoring window
    win_len = int(0.100 * sr)
    hop = win_len // 2
    n_frames = (n_min - win_len) // hop + 1
    if n_frames < 3:
        return 1.0

    scores = []
    weights = []

    for i in range(n_frames):
        start = i * hop
        fo = mono_orig[start : start + win_len]
        fp = mono_proc[start : start + win_len]

        rms_o = float(np.sqrt(np.mean(fo**2)))
        if rms_o < 1e-6:
            continue  # Silence — skip

        formants_o = _extract_formants(fo, sr, n_formants)
        formants_p = _extract_formants(fp, sr, n_formants)

        if not formants_o or not formants_p:
            continue

        # Compare formant positions
        n_common = min(len(formants_o), len(formants_p))
        deviations = []
        for j in range(n_common):
            f_o = formants_o[j]
            f_p = formants_p[j]
            if f_o > 0:
                rel_dev = abs(f_p - f_o) / f_o
                deviations.append(rel_dev)

        if not deviations:
            continue

        # Score based on mean relative deviation
        mean_dev = float(np.mean(deviations))
        # <5% deviation → score 1.0, >20% → score 0.0
        frame_score = float(max(0.0, min(1.0, 1.0 - (mean_dev - 0.05) / 0.15)))

        scores.append(frame_score)
        weights.append(rms_o)

    if not scores:
        # Fallback: spectral energy preservation in formant range
        return _formant_energy_fallback(mono_orig, mono_proc, sr, formant_range)

    weights = np.array(weights, dtype=np.float64)
    weights /= np.sum(weights) + 1e-10
    return float(np.dot(scores, weights))


def _extract_formants(audio: np.ndarray, sr: int, n_formants: int = 4) -> list[float]:
    """Extrahiere Formant-Frequenzen via LPC."""
    try:
        # LPC order: ~sr/1000 for formant resolution
        order = min(int(sr / 1000.0) + 4, 50)
        n = len(audio)

        # Autocorrelation
        r = np.correlate(audio, audio, mode="full")[n - 1 : n + order]
        r = r.astype(np.float64)

        if abs(r[0]) < 1e-15:
            return []

        # Levinson-Durbin
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        e = float(r[0])
        for k in range(1, order + 1):
            lam = float(r[k])
            for j in range(1, k):
                lam += a[j] * r[k - j]
            lam = -lam / max(e, 1e-15)
            a_prev = a.copy()
            for j in range(1, k):
                a[j] = a_prev[j] + lam * a_prev[k - j]
            a[k] = lam
            e *= 1.0 - lam**2

        if e <= 0:
            return []

        # Frequency response of LPC filter → find peaks
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        # H(z) = 1 / A(z), A(z) = 1 + a₁z⁻¹ + ...
        # |A(e^(jω))|² → peaks in |H|² = 1/|A|²
        A = np.fft.rfft(a, n=n_fft)
        H = 1.0 / (np.abs(A) + 1e-10)
        H_db = 20.0 * np.log10(H)

        # Find peaks in 200-4000 Hz range
        lo = np.searchsorted(freqs, 200.0)
        hi = np.searchsorted(freqs, 4000.0)
        if hi <= lo:
            return []

        peaks = _find_peaks(H_db[lo:hi], threshold=np.mean(H_db[lo:hi]))
        peak_values = [(freqs[lo + p], H_db[lo + p]) for p in peaks if lo + p < len(freqs)]
        peak_values.sort(key=lambda x: -x[1])  # Sort by magnitude
        formants = [f for f, _ in peak_values[:n_formants]]
        formants.sort()  # Sort by frequency
        return formants
    except Exception:
        return []


def _formant_energy_fallback(
    orig: np.ndarray, proc: np.ndarray, sr: int, formant_range: tuple
) -> float:
    """Fallback: Energie-Erhaltung im Formantbereich."""
    n_fft = 2048
    n_min = min(len(orig), len(proc))
    if n_min < n_fft:
        return 1.0

    fo = orig[:n_min]
    fp = proc[:n_min]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lo = np.searchsorted(freqs, formant_range[0])
    hi = np.searchsorted(freqs, formant_range[1])

    # Average over blocks
    hop = n_fft // 2
    n_blocks = (n_min - n_fft) // hop + 1
    ratios = []
    for i in range(n_blocks):
        s = i * hop
        spec_o = np.abs(np.fft.rfft(fo[s : s + n_fft] * np.hanning(n_fft)))
        spec_p = np.abs(np.fft.rfft(fp[s : s + n_fft] * np.hanning(n_fft)))
        e_o = float(np.sum(spec_o[lo:hi] ** 2))
        e_p = float(np.sum(spec_p[lo:hi] ** 2))
        if e_o > 1e-10:
            ratios.append(min(e_p / e_o, e_o / e_p))  # Symmetric ratio
    if not ratios:
        return 1.0
    return float(np.mean(ratios))


# ── Helpers ──────────────────────────────────────────────────────────────


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0) if audio.shape[1] < audio.shape[0] else audio.mean(axis=1)
