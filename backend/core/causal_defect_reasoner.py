"""
Causal Defect Reasoner — Aurik 9.7
=====================================
Bayesianische Ursachendiagnose für mangelhafte Tonträger-Aufnahmen.
Gegeben eine Menge von Fehlermerkmalen (DefectResult + akustische Signalmerkmale)
und dem Trägermaterial berechnet dieses Modul die posteriore Wahrscheinlichkeit
über 8 mögliche Wurzelursachen und erstellt einen priorisierten Restaurierungsplan.

Das System ist ein diskretisiertes Bayesnetz:
    P(cause | observations, material) ∝ P(observations | cause) · P(cause | material)

Ursachen (11) — Spec §2.4:
    tape_dropout        — Magnetband-Aussetzer (Dropout)
    tape_hiss           — Bandrauschen (thermisches & Partikelrauschen)
    vinyl_crackle       — Vinyl-Knistern (Oberflächendefekte)
    vinyl_warp          — Plattenwellung (Pitch-Instabilität + Intermodulation)
    electrical_hum      — Netzbrumm 50/60 Hz + Obertöne
    head_misalignment   — Tonkopf-Fehlausrichtung (Azimut-Fehler, HF-Verlust)
    dc_offset           — DC-Versatz (einseitiger Betriebspunkt)
    digital_clip        — Digitales Clipping (Harddist, Integer-Overflow)
    soft_saturation     — Röhren-/Tape-Sättigung (gerade Obertöne) — kein Eingriff (BEWAHREN)
    head_wear           — Komplette Frequenzband-Auslöschung durch Kopfverschleiß (→ phase_56)
    print_through       — Magnetisches Tape-Übersprechen / Vorecho (Adaptive Temporal Subtraction)

Ausgabe:
    RestorationPlan mit geordneten Restaurierungsphasen + Parametern

Referenzen:
    - Pearl, Causality (2000) — Bayesnetze
    - Maher, Audio Restoration (IEEE, 1993)
    - Lahat et al., Temporal and Spectral Audio Forensics (2013)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typ-Definitionen
# ---------------------------------------------------------------------------

# 11 Kausal-Ursachen (Spec §2.4) — konzeptuell getrennt von den 24 DefectType-Scan-Werten
CAUSES = [
    "tape_dropout",
    "tape_hiss",
    "vinyl_crackle",
    "vinyl_warp",
    "electrical_hum",
    "head_misalignment",
    "dc_offset",
    "digital_clip",
    "soft_saturation",  # Tube-/Tape-Sättigung — BEWAHREN, P(phases) = leer
    "head_wear",  # Frequenzband-Auslöschung → phase_56_spectral_band_gap_repair
    "print_through",  # Magnetisches Vorecho → Adaptive Temporal Subtraction
]

# Material-Typen
MATERIAL_PRIORS: Dict[str, Dict[str, float]] = {
    # Priors normiert auf 1.0 pro Materialtyp (3 neue Ursachen mit kleinen Priors)
    "tape": {
        "tape_dropout": 0.27,
        "tape_hiss": 0.27,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.13,
        "head_misalignment": 0.10,
        "dc_offset": 0.05,
        "digital_clip": 0.05,
        "soft_saturation": 0.05,  # Röhrenverstärker-Tape häufig
        "head_wear": 0.04,  # Kopfverschleiß realistisch bei altem Tape
        "print_through": 0.02,  # Magnetisches Übersprechen
    },
    "vinyl": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.38,
        "vinyl_warp": 0.18,
        "electrical_hum": 0.09,
        "head_misalignment": 0.03,
        "dc_offset": 0.05,
        "digital_clip": 0.13,
        "soft_saturation": 0.05,  # Röhrenschneidköpfe
        "head_wear": 0.02,
        "print_through": 0.01,
    },
    "shellac": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.09,
        "vinyl_crackle": 0.42,
        "vinyl_warp": 0.14,
        "electrical_hum": 0.11,
        "head_misalignment": 0.05,
        "dc_offset": 0.04,
        "digital_clip": 0.07,
        "soft_saturation": 0.04,
        "head_wear": 0.02,
        "print_through": 0.01,
    },
    "digital": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.02,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.18,
        "head_misalignment": 0.02,
        "dc_offset": 0.09,
        "digital_clip": 0.50,
        "soft_saturation": 0.07,  # Soft-Limiting in DAWs
        "head_wear": 0.01,
        "print_through": 0.04,  # Print-Through in digitalen Kopien hist. Tape
    },
    "unknown": {
        "tape_dropout": 0.10,
        "tape_hiss": 0.10,
        "vinyl_crackle": 0.10,
        "vinyl_warp": 0.07,
        "electrical_hum": 0.13,
        "head_misalignment": 0.09,
        "dc_offset": 0.09,
        "digital_clip": 0.17,
        "soft_saturation": 0.07,
        "head_wear": 0.05,
        "print_through": 0.03,
    },
    # ── Digitale / Codec-Quellen (kein Magnetband-Dropout möglich) ──────────
    "mp3_low": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.50,
        "soft_saturation": 0.07,
        "head_wear": 0.01,
        "print_through": 0.25,  # MP3 Print-Through = Pre-Echo-Artefakt
    },
    "mp3_high": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.09,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.36,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.39,
    },
    "aac": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.09,
        "head_misalignment": 0.01,
        "dc_offset": 0.04,
        "digital_clip": 0.36,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.39,
    },
    "cd_digital": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.11,
        "head_misalignment": 0.01,
        "dc_offset": 0.07,
        "digital_clip": 0.41,
        "soft_saturation": 0.06,
        "head_wear": 0.01,
        "print_through": 0.28,
    },
    "streaming": {
        "tape_dropout": 0.01,
        "tape_hiss": 0.02,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.03,
        "digital_clip": 0.32,
        "soft_saturation": 0.05,
        "head_wear": 0.01,
        "print_through": 0.46,
    },
    "dat": {
        "tape_dropout": 0.07,
        "tape_hiss": 0.04,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.11,
        "head_misalignment": 0.02,
        "dc_offset": 0.04,
        "digital_clip": 0.27,
        "soft_saturation": 0.03,
        "head_wear": 0.05,  # DAT-Köpfe verschleißen stark
        "print_through": 0.35,
    },
    "minidisc": {
        "tape_dropout": 0.04,
        "tape_hiss": 0.03,
        "vinyl_crackle": 0.01,
        "vinyl_warp": 0.01,
        "electrical_hum": 0.07,
        "head_misalignment": 0.01,
        "dc_offset": 0.03,
        "digital_clip": 0.41,
        "soft_saturation": 0.04,
        "head_wear": 0.02,
        "print_through": 0.33,
    },
    # ── Historische Medien ───────────────────────────────────────────────────
    "wax_cylinder": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.23,
        "vinyl_crackle": 0.37,
        "vinyl_warp": 0.09,
        "electrical_hum": 0.07,
        "head_misalignment": 0.05,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.02,
        "head_wear": 0.08,  # Phonograph-Nadel-Verschleiß
        "print_through": 0.01,
    },
    "lacquer_disc": {
        "tape_dropout": 0.02,
        "tape_hiss": 0.13,
        "vinyl_crackle": 0.41,
        "vinyl_warp": 0.11,
        "electrical_hum": 0.09,
        "head_misalignment": 0.04,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.03,
        "head_wear": 0.09,  # Ritznadel-Ermüdung
        "print_through": 0.02,
    },
    "wire_recording": {
        "tape_dropout": 0.18,
        "tape_hiss": 0.22,
        "vinyl_crackle": 0.04,
        "vinyl_warp": 0.13,
        "electrical_hum": 0.13,
        "head_misalignment": 0.09,
        "dc_offset": 0.04,
        "digital_clip": 0.02,
        "soft_saturation": 0.02,
        "head_wear": 0.10,  # Magnetdraht-Kopfverschleiß
        "print_through": 0.03,
    },
}

# Phase-Empfehlungen pro Ursache (kanonische phase_id = Dateiname ohne .py)
CAUSE_TO_PHASES: Dict[str, List[str]] = {
    # ── Magnetband ────────────────────────────────────────────────────────────
    "tape_dropout": [
        "phase_24_dropout_repair",
        "phase_55_diffusion_inpainting",  # §7.2 DiffWave-Inpainting
        "phase_01_click_removal",
        "phase_03_denoise",
    ],
    "tape_hiss": [
        "phase_29_tape_hiss_reduction",
        "phase_03_denoise",
        "phase_04_eq_correction",
        "phase_40_loudness_normalization",
    ],
    # ── Vinyl ────────────────────────────────────────────────────────────────
    "vinyl_crackle": [
        "phase_09_crackle_removal",
        "phase_01_click_removal",
        "phase_28_surface_noise_profiling",
        "phase_03_denoise",
    ],
    "vinyl_warp": [
        "phase_12_wow_flutter_fix",
        "phase_31_speed_pitch_correction",
        "phase_04_eq_correction",
        "phase_03_denoise",
    ],
    # ── Elektrik / Mechanik ──────────────────────────────────────────────────
    "electrical_hum": ["phase_02_hum_removal", "phase_03_denoise", "phase_04_eq_correction"],
    "head_misalignment": [
        "phase_06_frequency_restoration",
        "phase_04_eq_correction",
        "phase_14_phase_correction",
        "phase_25_azimuth_correction",
        "phase_03_denoise",
    ],
    "dc_offset": ["phase_30_dc_offset_removal", "phase_40_loudness_normalization"],
    "digital_clip": ["phase_23_spectral_repair", "phase_06_frequency_restoration", "phase_40_loudness_normalization"],
    # ── Spektrale Defekte ────────────────────────────────────────────────────
    "bandwidth_loss": [
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
        "phase_39_air_band_enhancement",
    ],
    "high_freq_noise": ["phase_29_tape_hiss_reduction", "phase_03_denoise", "phase_18_noise_gate"],
    # ── Stereo / Phase ───────────────────────────────────────────────────────
    "stereo_imbalance": ["phase_15_stereo_balance", "phase_33_stereo_width_limiter", "phase_34_mid_side_processing"],
    "phase_issues": ["phase_14_phase_correction", "phase_25_azimuth_correction"],
    # ── Pitch / Dynamik ──────────────────────────────────────────────────────
    "pitch_drift": ["phase_31_speed_pitch_correction", "phase_12_wow_flutter_fix"],
    "reverb_excess": ["phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
    "print_through": [
        "phase_29_tape_hiss_reduction",
        "phase_24_dropout_repair",  # §4.5+§7.2
        "phase_03_denoise",
        "phase_23_spectral_repair",
    ],
    # ── Digital / Codec ──────────────────────────────────────────────────────
    "digital_artifacts": ["phase_23_spectral_repair", "phase_50_spectral_repair", "phase_06_frequency_restoration"],
    "compression_artifacts": [
        "phase_23_spectral_repair",
        "phase_50_spectral_repair",  # §7.2 Apollo primär
        "phase_26_dynamic_range_expansion",
        "phase_06_frequency_restoration",
        "phase_54_transparent_dynamics",
    ],
    "quantization_noise": ["phase_23_spectral_repair", "phase_03_denoise", "phase_06_frequency_restoration"],
    "jitter_artifacts": ["phase_23_spectral_repair", "phase_12_wow_flutter_fix"],
    "dynamic_compression_excess": [
        "phase_26_dynamic_range_expansion",
        "phase_54_transparent_dynamics",
        "phase_35_multiband_compression",
    ],
    "head_wear": [
        "phase_56_spectral_band_gap_repair",  # §4.5/§7.2
        "phase_14_phase_correction",
        "phase_06_frequency_restoration",
    ],
    "soft_saturation": [],  # §2.1/§6.3 — BEWAHREN, kein destruktiver Eingriff
    # ── MP3/AAC-Codec-Pre-Echo ───────────────────────────────────────────────
    "pre_echo": [
        "phase_23_spectral_repair",  # §6.3 — Codec-Pre-Echo vor Transienten
        "phase_50_spectral_repair",
        "phase_08_transient_preservation",
    ],
    # ── Tieffrequenz / Transienten / Clipping ────────────────────────────────
    "low_freq_rumble": [
        "phase_05_rumble_filter",  # §7.2 — Subsonic-/Rumble-Filter
        "phase_03_denoise",
        "phase_04_eq_correction",
    ],
    "transient_smearing": [
        "phase_08_transient_preservation",  # §7.2 — Transienten-Restaurierung
        "phase_36_transient_shaper",
        "phase_23_spectral_repair",
    ],
    "clipping": ["phase_23_spectral_repair", "phase_06_frequency_restoration"],  # §7.2 — Alias für DefectType.CLIPPING
    # ── Entzerrungs- & Digitalisierungsfehler (§6.3, §7.2 v9.10.46) ─────────
    "riaa_curve_error": [
        "phase_04_eq_correction",           # RIAA/AES/NAB/FFRR-Entzerrungs-Fehler
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
    ],
    "aliasing": [
        "phase_03_denoise",                 # AA-Filter-Artefakte aus Digitalisierung
        "phase_23_spectral_repair",
        "phase_50_spectral_repair",
    ],
    "bias_error": [
        "phase_04_eq_correction",           # Falscher Vormagnetisierungsstrom (Bandaufnahme)
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_29_tape_hiss_reduction",
    ],
}

# Empfohlene Parameter pro Ursache
CAUSE_PARAMS: Dict[str, Dict[str, Any]] = {
    "tape_dropout": {
        "noise_reduction_strength": 0.55,
        "ar_order": 64,
        "ola_crossfade_ms": 20.0,
        "inpaint_context_ms": 50.0,
    },
    "tape_hiss": {
        "noise_reduction_strength": 0.70,
        "ar_order": 32,
        "hpf_cutoff_hz": 60.0,
        "nr_smoothing_ms": 80.0,
    },
    "vinyl_crackle": {
        "click_threshold_sigma": 4.5,
        "noise_reduction_strength": 0.40,
        "ar_order": 48,
        "declicker_window_ms": 5.0,
    },
    "vinyl_warp": {
        "pitch_correction_semitones": 0.5,
        "wow_flutter_filter_hz": 0.5,
        "noise_reduction_strength": 0.25,
    },
    "electrical_hum": {
        "hum_fundamental_hz": 50.0,  # wird auto-erkannt
        "hum_harmonics": 5,
        "hum_notch_q": 20.0,
        "noise_reduction_strength": 0.20,
    },
    "head_misalignment": {
        "azimuth_correction_deg": 0.0,  # wird optimiert
        "hf_boost_db": 3.0,
        "eq_high_shelf_hz": 8000.0,
    },
    "dc_offset": {
        "hpf_cutoff_hz": 5.0,
        "normalization_lufs": -23.0,
    },
    "digital_clip": {
        "declip_threshold": 0.98,
        "harmonic_boost_db": 1.5,
        "noise_reduction_strength": 0.10,
    },
}


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFeatures:
    """Kompakte Signalmerkmale für Bayesianische Diagnose."""

    rms: float = 0.0
    peak: float = 0.0
    dc_offset: float = 0.0
    crest_factor_db: float = 0.0
    spectral_rolloff_hz: float = 5000.0
    hf_energy_ratio: float = 0.5  # Energie oberhalb 4 kHz / Gesamt
    lf_energy_ratio: float = 0.2  # Energie unterhalb 200 Hz / Gesamt
    hum_score: float = 0.0  # Stärke der harmonischen Linien bei 50/60 Hz
    click_density: float = 0.0  # Clicks pro Sekunde
    dropout_density: float = 0.0  # Dropouts pro Sekunde
    pitch_instability: float = 0.0  # σ(F0-Varianz)
    stereo_correlation: float = 1.0  # [-1, 1]
    clip_fraction: float = 0.0  # Anteil geclippter Samples


@dataclass
class RestorationPlan:
    """Priorisierter Restaurierungsplan des CausalDefectReasoner."""

    primary_cause: str
    cause_probabilities: Dict[str, float]  # normierte Posterioren
    ranked_causes: List[Tuple[str, float]]  # absteigend nach Prob.
    recommended_phases: List[str]  # geordnet
    phase_parameters: Dict[str, Any]  # {phase_id: {param: value}}
    confidence: float  # max. Posterior-Wert
    reasoning: str  # menschenlesbare Erklärung
    material: str


# ---------------------------------------------------------------------------
# Merkmals-Extraktion
# ---------------------------------------------------------------------------


def extract_spectral_features(audio: np.ndarray, sample_rate: int) -> SpectralFeatures:
    """
    Extrahiert kompakte Signalmerkmale für das Bayesnetz.

    Args:
        audio:       np.ndarray, mono oder stereo
        sample_rate: Abtastrate
    Returns:
        SpectralFeatures
    """
    if audio.ndim == 2:
        # Stereo: Kanäle trennen für Korrelation
        if audio.shape[0] <= 2:
            ch_l, ch_r = audio[0], audio[1]
            mono = 0.5 * (ch_l + ch_r)
        else:
            mono = np.mean(audio, axis=1)
            ch_l = ch_r = mono
    else:
        mono = audio.astype(np.float64)
        ch_l = ch_r = mono

    mono = mono.astype(np.float64)
    n = len(mono)
    if n < 32:
        return SpectralFeatures()

    sr = sample_rate

    # Basismerkmale
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    dc = float(np.mean(mono))
    cf = float(20 * math.log10(peak / (rms + 1e-9) + 1e-9))

    # FFT-Merkmale
    n_fft = min(4096, _next_pow2(n))
    hop = n_fft // 4
    # Mittleres Powerspektrum
    n_frames = max(1, (n - n_fft) // hop)
    ps = np.zeros(n_fft // 2 + 1)
    for i in range(n_frames):
        seg = mono[i * hop : i * hop + n_fft]
        ps += np.abs(np.fft.rfft(seg * np.hanning(n_fft))) ** 2
    ps /= n_frames + 1e-12

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total_energy = np.sum(ps) + 1e-12

    # Spektrale Rolloff
    cumsum = np.cumsum(ps)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # HF / LF Energieverhältnis
    hf_mask = freqs >= 4000.0
    lf_mask = freqs <= 200.0
    hf_ratio = float(np.sum(ps[hf_mask]) / total_energy)
    lf_ratio = float(np.sum(ps[lf_mask]) / total_energy)

    # Brumm-Score: harmonische Linien bei 50/60 Hz
    hum_score = _compute_hum_score(ps, freqs)

    # Click-Dichte (Impuls-Detektor via Z-Score)
    click_density = _compute_click_density(mono, sr)

    # Dropout-Dichte (kurze Stille-Intervalle)
    dropout_density = _compute_dropout_density(mono, sr)

    # Pitch-Instabilität (näherungsweise via AutoKorrelation)
    pitch_instability = _compute_pitch_instability(mono, sr)

    # Stereo-Korrelation
    if ch_l is not ch_r:
        corr = float(np.corrcoef(ch_l[: min(len(ch_l), len(ch_r))], ch_r[: min(len(ch_l), len(ch_r))])[0, 1])
        stereo_corr = float(np.clip(corr, -1.0, 1.0))
    else:
        stereo_corr = 1.0

    # Clipping-Anteil
    clip_thr = 0.97
    clip_frac = float(np.mean(np.abs(mono) >= clip_thr))

    return SpectralFeatures(
        rms=rms,
        peak=peak,
        dc_offset=dc,
        crest_factor_db=cf,
        spectral_rolloff_hz=rolloff_hz,
        hf_energy_ratio=hf_ratio,
        lf_energy_ratio=lf_ratio,
        hum_score=hum_score,
        click_density=click_density,
        dropout_density=dropout_density,
        pitch_instability=pitch_instability,
        stereo_correlation=stereo_corr,
        clip_fraction=clip_frac,
    )


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _compute_hum_score(ps: np.ndarray, freqs: np.ndarray) -> float:
    """Stärke harmonischer Linien bei 50 oder 60 Hz."""
    score_50 = _harmonic_line_score(ps, freqs, 50.0, n_harmonics=5)
    score_60 = _harmonic_line_score(ps, freqs, 60.0, n_harmonics=5)
    return float(max(score_50, score_60))


def _harmonic_line_score(ps: np.ndarray, freqs: np.ndarray, f0: float, n_harmonics: int = 5) -> float:
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    total = np.sum(ps) + 1e-12  # noqa: F841
    score = 0.0
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        idx = int(round(fk / df))
        idx = min(max(idx, 0), len(ps) - 1)
        # Schmales Fenster ± 2 Bins
        window = ps[max(0, idx - 2) : idx + 3]
        line_energy = np.max(window) if len(window) > 0 else 0.0
        # Vergleich mit lokal umgebendem Hintergrund
        bg_lo = ps[max(0, idx - 10) : max(0, idx - 3)]
        bg_hi = ps[min(len(ps) - 1, idx + 3) : min(len(ps), idx + 10)]
        bg = np.mean(np.concatenate([bg_lo, bg_hi])) + 1e-12
        score += min(line_energy / bg, 10.0)  # SNR der Linie
    return score / (n_harmonics * 10.0)  # normiert auf [0, 1]


def _compute_click_density(mono: np.ndarray, sr: int) -> float:
    """Clicks pro Sekunde mittels robustem Z-Score."""
    diff = np.diff(mono)
    med = np.median(np.abs(diff))  # noqa: F841
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-9
    z = np.abs(diff - np.median(diff)) / (1.4826 * mad)
    clicks = float(np.sum(z > 8.0))
    dur_s = len(mono) / (sr + 1e-9)
    return clicks / (dur_s + 1e-9)


def _compute_dropout_density(mono: np.ndarray, sr: int) -> float:
    """Dropouts pro Sekunde: kurze Intervalle mit sehr kleiner Energie."""
    frame_len = int(sr * 0.005)  # 5 ms Frames
    if frame_len < 1 or len(mono) < frame_len:
        return 0.0
    n_frames = len(mono) // frame_len
    frame_rms = np.array([np.sqrt(np.mean(mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)])
    global_rms = np.mean(frame_rms) + 1e-9
    dropout_frames = np.sum(frame_rms < 0.05 * global_rms)
    dur_s = len(mono) / (sr + 1e-9)
    return float(dropout_frames * 0.005 / dur_s)


def _compute_pitch_instability(mono: np.ndarray, sr: int) -> float:
    """Grobe Pitch-Instabilität mittels autokorrelationsbasierter F0-Verfolgung."""
    frame_len = int(sr * 0.050)  # 50 ms Frames
    hop = frame_len // 2
    if frame_len < 16 or len(mono) < frame_len:
        return 0.0
    f0_values = []
    for i in range(0, len(mono) - frame_len, hop):
        seg = mono[i : i + frame_len]
        # Autokorrelation
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        ac = ac / (ac[0] + 1e-9)
        # Suche Peak zwischen 2 ms und 20 ms (50 Hz – 500 Hz)
        lo = max(1, int(sr * 0.002))
        hi = min(len(ac) - 1, int(sr * 0.020))
        if lo >= hi:
            continue
        peak_idx = int(np.argmax(ac[lo:hi])) + lo
        if ac[peak_idx] > 0.3:
            f0 = sr / (peak_idx + 1)
            f0_values.append(f0)
    if len(f0_values) < 2:
        return 0.0
    f0_arr = np.array(f0_values)
    return float(np.std(f0_arr) / (np.mean(f0_arr) + 1e-9))


# ---------------------------------------------------------------------------
# Likelihood-Funktionen P(Merkmale | Ursache)
# ---------------------------------------------------------------------------


def _likelihood_tape_dropout(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    """Bedingte Wahrscheinlichkeit für Tape-Dropout."""
    p = 0.0
    p += _gaussian_score(sf.dropout_density, mu=0.5, sigma=0.3) * 0.40
    p += _gaussian_score(sf.click_density, mu=0.5, sigma=0.5) * 0.15
    p += _sigmoid_score(defect_scores.get("dropout_severity", 0.0), k=8, x0=0.4) * 0.30
    p += _sigmoid_score(defect_scores.get("silence_ratio", 0.0), k=6, x0=0.2) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_tape_hiss(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.35, sigma=0.15) * 0.40
    p += _sigmoid_score(defect_scores.get("noise_floor_db", -60.0) + 60.0, k=0.1, x0=30.0) * 0.35
    p += (1.0 - sf.hum_score) * 0.15
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vinyl_crackle(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(sf.click_density, k=1.0, x0=2.0) * 0.45
    p += _sigmoid_score(defect_scores.get("click_severity", 0.0), k=8, x0=0.35) * 0.35
    p += (1.0 - abs(sf.dc_offset)) * 0.10
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_vinyl_warp(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(sf.pitch_instability, k=20, x0=0.02) * 0.50
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.30, sigma=0.15) * 0.30
    p += _gaussian_score(float(defect_scores.get("wow_flutter", 0.0)), mu=0.4, sigma=0.2) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_electrical_hum(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += sf.hum_score * 0.60
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.35, sigma=0.15) * 0.25
    p += sf.stereo_correlation * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_head_misalignment(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    # HF-Verlust: rolloff deutlich unter 10 kHz
    hf_loss = max(0.0, 1.0 - sf.spectral_rolloff_hz / 10000.0)
    p += hf_loss * 0.45
    p += (1.0 - sf.hf_energy_ratio) * 0.30
    p += _sigmoid_score(float(defect_scores.get("azimuth_error", 0.0)), k=10, x0=0.3) * 0.25
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_dc_offset(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(abs(sf.dc_offset), k=20, x0=0.03) * 0.70
    p += _gaussian_score(sf.lf_energy_ratio, mu=0.40, sigma=0.15) * 0.20
    p += (1.0 - sf.clip_fraction) * 0.10
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_digital_clip(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    p = 0.0
    p += _sigmoid_score(sf.clip_fraction, k=30, x0=0.02) * 0.55
    p += _sigmoid_score(sf.crest_factor_db, k=0.1, x0=0.0) * 0.25  # niedriger Crestfaktor
    p += _sigmoid_score(float(defect_scores.get("clip_severity", 0.0)), k=8, x0=0.3) * 0.20
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_soft_saturation(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    """P(Merkmale | soft_saturation) — Röhren-/Tape-Sättigung (gerade Obertöne).

    Soft-Saturation erzeugt gerade Harmonische (H2, H4) und runde Wellenformen
    ohne Flat-Tops. Spectral Flatness im Clip-Bereich ist niedrig (< 0.3).
    Spec §6.3: BEWAHREN — Phasen-Liste bleibt leer, aber die Ursache muss
    im Bayes-Posterior korrekt erscheinen, damit keine destruktiven Phasen
    (z.B. phase_23_spectral_repair als Clipping-Fix) fälschlicherweise aktiviert werden.
    """
    p = 0.0
    # Hinweis auf Sättigung: Clip-Anteil niedrig, Crest-Faktor moderat
    p += _gaussian_score(sf.clip_fraction, mu=0.005, sigma=0.01) * 0.40  # kaum Flat-Tops
    p += _gaussian_score(sf.crest_factor_db, mu=8.0, sigma=4.0) * 0.30  # typischer Crest
    p += (1.0 - sf.clip_fraction) * 0.30
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_head_wear(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    """P(Merkmale | head_wear) — Kopfverschleiß / Frequenzband-Auslöschung.

    Charakteristisch: vollständiger HF-Energie-Verlust (> 3 kHz) über die
    gesamte Dateilänge, sehr niedriger Spectral-Rolloff (§4.5, §6.3).
    """
    p = 0.0
    # Niedriger HF-Anteil — Hauptmerkmal
    hf_loss = max(0.0, 1.0 - sf.hf_energy_ratio / 0.05)  # < 5 % HF-Energie
    p += hf_loss * 0.50
    # Rolloff deutlich unter 5 kHz
    rolloff_loss = max(0.0, 1.0 - sf.spectral_rolloff_hz / 5000.0)
    p += rolloff_loss * 0.35
    p += _sigmoid_score(float(defect_scores.get("azimuth_error", 0.0)), k=8, x0=0.5) * 0.15
    return float(np.clip(p, 0.0, 1.0))


def _likelihood_print_through(sf: SpectralFeatures, defect_scores: Dict[str, float]) -> float:
    """P(Merkmale | print_through) — Magnetisches Vorecho / Nachecho (Tape).

    Print-Through äußert sich als sehr schwaches Geister-Signal (typisch −20 bis
    −30 dB) kurz vor/nach dem Hauptsignal. Als Proxy: hoher Dropout-Score bei
    gleichzeitig vorhandenem Tape-Hiss-Profil (Reel-Tape-Context).
    """
    p = 0.0
    p += _gaussian_score(sf.hf_energy_ratio, mu=0.30, sigma=0.12) * 0.35  # Tape-HF-Profil
    p += _sigmoid_score(defect_scores.get("noise_floor_db", -60.0) + 60.0, k=0.08, x0=25.0) * 0.35
    p += _gaussian_score(sf.dropout_density, mu=0.1, sigma=0.15) * 0.30
    return float(np.clip(p, 0.0, 1.0))


LIKELIHOOD_FNS = {
    "tape_dropout": _likelihood_tape_dropout,
    "tape_hiss": _likelihood_tape_hiss,
    "vinyl_crackle": _likelihood_vinyl_crackle,
    "vinyl_warp": _likelihood_vinyl_warp,
    "electrical_hum": _likelihood_electrical_hum,
    "head_misalignment": _likelihood_head_misalignment,
    "dc_offset": _likelihood_dc_offset,
    "digital_clip": _likelihood_digital_clip,
    "soft_saturation": _likelihood_soft_saturation,
    "head_wear": _likelihood_head_wear,
    "print_through": _likelihood_print_through,
}


def _gaussian_score(x: float, mu: float, sigma: float) -> float:
    """Gaussianischer Ähnlichkeitsscore ∈ [0, 1]."""
    return float(math.exp(-0.5 * ((x - mu) / (sigma + 1e-9)) ** 2))


def _sigmoid_score(x: float, k: float = 5.0, x0: float = 0.5) -> float:
    """Sigmoidaler Score ∈ (0, 1). Höheres x → höhere Wahrscheinlichkeit."""
    return float(1.0 / (1.0 + math.exp(-k * (x - x0))))


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class CausalDefectReasoner:
    """
    Bayesianische Ursachendiagnose für Tonträger-Fehler.

    Verwendung::

        reasoner = CausalDefectReasoner()
        plan = reasoner.reason(
            defect_scores={"dropout_severity": 0.7, "noise_floor_db": -45.0},
            material="tape",
            audio=waveform,
            sample_rate=44100,
        )
        logger.debug(plan.primary_cause)
        logger.debug(plan.recommended_phases)
    """

    def __init__(self, detect_hum_hz: Optional[float] = None):
        """
        Args:
            detect_hum_hz: Bekannte Netzfrequenz (50 oder 60 Hz). Wenn None,
                           wird automatisch erkannt.
        """
        self._known_hum_hz = detect_hum_hz

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def reason(
        self,
        defect_scores: Dict[str, float],
        material: str = "unknown",
        audio: Optional[np.ndarray] = None,
        sample_rate: int = 48000,
        sr: Optional[int] = None,
    ) -> RestorationPlan:
        """
        Berechnet den Restaurierungsplan für die gegebene Aufnahme.

        Args:
            defect_scores: Fehlermerkmale {"dropout_severity": 0.7, ...}
            material:      Trägermaterial ("tape"|"vinyl"|"shellac"|"digital"|"unknown")
            audio:         Optional – rohes Audio-Array für Signal-Merkmale
            sample_rate:   Abtastrate des Audio-Arrays
        Returns:
            RestorationPlan
        """
        if sr is not None:
            sample_rate = sr
        if audio is not None and len(audio) > 0:
            assert sample_rate == 48000, f"CausalDefectReasoner.reason() erwartet SR=48000, erhalten: {sample_rate}"
        material = material.lower().strip()
        if material not in MATERIAL_PRIORS:
            material = "unknown"

        # Signal-Merkmale extrahieren
        if audio is not None and len(audio) > 0:
            sf = extract_spectral_features(audio, sample_rate)
        else:
            sf = SpectralFeatures()

        return self._infer(defect_scores, sf, material)

    # ------------------------------------------------------------------
    # Bayes-Inferenz
    # ------------------------------------------------------------------

    def _infer(
        self,
        defect_scores: Dict[str, float],
        sf: SpectralFeatures,
        material: str,
    ) -> RestorationPlan:
        priors = MATERIAL_PRIORS[material]
        posteriors: Dict[str, float] = {}

        for cause in CAUSES:
            prior = priors.get(cause, 1.0 / len(CAUSES))
            likelihood = LIKELIHOOD_FNS[cause](sf, defect_scores)
            posteriors[cause] = prior * likelihood

        # Normierung
        total = sum(posteriors.values()) + 1e-12
        posteriors = {c: v / total for c, v in posteriors.items()}

        # Sortierung absteigend
        ranked = sorted(posteriors.items(), key=lambda kv: kv[1], reverse=True)
        primary_cause = ranked[0][0]
        confidence = ranked[0][1]

        # Fusions-Plan: Phasen der Top-3 Ursachen zusammenführen
        seen_phases: set = set()
        ordered_phases: List[str] = []
        merged_params: Dict[str, Any] = {}

        for cause, prob in ranked[:3]:
            if prob < 0.05:
                break
            for phase in CAUSE_TO_PHASES.get(cause, []):
                if phase not in seen_phases:
                    ordered_phases.append(phase)
                    seen_phases.add(phase)
            for param, val in CAUSE_PARAMS.get(cause, {}).items():
                if param not in merged_params:
                    merged_params[param] = val

        reasoning = self._build_reasoning(primary_cause, ranked, sf, material)

        return RestorationPlan(
            primary_cause=primary_cause,
            cause_probabilities=posteriors,
            ranked_causes=ranked,
            recommended_phases=ordered_phases,
            phase_parameters=merged_params,
            confidence=confidence,
            reasoning=reasoning,
            material=material,
        )

    def _build_reasoning(
        self,
        primary_cause: str,
        ranked: List[Tuple[str, float]],
        sf: SpectralFeatures,
        material: str,
    ) -> str:
        lines = [
            f"Trägermaterial: {material}",
            f"Primäre Ursache: {primary_cause} (Posterior={ranked[0][1]:.3f})",
        ]
        if len(ranked) > 1 and ranked[1][1] > 0.10:
            lines.append(f"Zweitwahrscheinlichste: {ranked[1][0]} ({ranked[1][1]:.3f})")
        lines.append(f"Klick-Dichte: {sf.click_density:.2f}/s")
        lines.append(f"Dropout-Dichte: {sf.dropout_density:.3f}")
        lines.append(f"Brumm-Score: {sf.hum_score:.3f}")
        lines.append(f"Clipping-Anteil: {sf.clip_fraction:.4f}")
        lines.append(f"Spektraler Rolloff: {sf.spectral_rolloff_hz:.0f} Hz")
        return " | ".join(lines)


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------

_reasoner: Optional[CausalDefectReasoner] = None
_reasoner_lock = threading.Lock()


def get_reasoner() -> CausalDefectReasoner:
    """Globaler Singleton-Reasoner."""
    global _reasoner
    if _reasoner is None:
        with _reasoner_lock:
            if _reasoner is None:
                _reasoner = CausalDefectReasoner()
    return _reasoner


def reason_about_defects(
    defect_scores: Dict[str, float],
    material: str = "unknown",
    audio: Optional[np.ndarray] = None,
    sample_rate: int = 44100,
    sr: Optional[int] = None,
) -> RestorationPlan:
    """Convenience-Funktion für direkten Aufruf."""
    return get_reasoner().reason(defect_scores, material, audio, sample_rate, sr=sr)


# Spec §3.2 / §2.4: kanonischer Fabrik-Name
get_causal_reasoner = get_reasoner
