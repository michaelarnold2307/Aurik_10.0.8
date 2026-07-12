"""Human-readable phase name mapping for log messages.

Usage:
    from backend.core.phase_names import phase_human_name
    name = phase_human_name("phase_03_denoise")  # → "Entrauschen"
"""

from __future__ import annotations

_PHASE_NAMES: dict[str, str] = {
    "phase_01_click_removal":             "Knackser-Entfernung",
    "phase_02_hum_removal":               "Brumm-Entfernung",
    "phase_03_denoise":                   "Entrauschen",
    "phase_04_eq_correction":             "EQ-Korrektur",
    "phase_05_rumble_filter":             "Rumpel-Filter",
    "phase_06_frequency_restoration":     "Frequenz-Restaurierung",
    "phase_07_harmonic_restoration":      "Harmonische-Restaurierung",
    "phase_08_transient_preservation":    "Transienten-Erhalt",
    "phase_09_crackle_removal":           "Knackser-Entfernung (Vinyl)",
    "phase_10_compression":               "Kompression",
    "phase_11_limiting":                  "Limiting",
    "phase_12_wow_flutter_fix":           "Gleichlauf-Korrektur",
    "phase_13_stereo_enhancement":        "Stereo-Anreicherung",
    "phase_14_phase_correction":          "Phasen-Korrektur",
    "phase_15_stereo_balance":            "Stereo-Balance",
    "phase_16_final_eq":                  "Final-EQ",
    "phase_17_mastering_polish":          "Mastering-Politur",
    "phase_18_noise_gate":                "Rauschsperre",
    "phase_19_de_esser":                  "De-Esser",
    "phase_20_reverb_reduction":          "Hall-Reduktion",
    "phase_21_exciter":                   "Exciter",
    "phase_22_tape_saturation":           "Band-Saettigung",
    "phase_23_spectral_repair":           "Spektrale-Reparatur",
    "phase_24_dropout_repair":            "Dropout-Reparatur",
    "phase_25_azimuth_correction":        "Azimuth-Korrektur",
    "phase_26_dynamic_range_expansion":   "Dynamik-Erweiterung",
    "phase_27_click_pop_removal":         "Knackser-Entfernung",
    "phase_28_surface_noise_profiling":   "Oberflaechenrauschen",
    "phase_29_tape_hiss_reduction":       "Band-Rausch-Unterdrueckung",
    "phase_31_speed_pitch_correction":    "Geschwindigkeits-Korrektur",
    "phase_37_bass_enhancement":          "Bass-Anhebung",
    "phase_38_presence_boost":            "Praesenz-Anhebung",
    "phase_39_air_band_enhancement":      "Luftband-Anhebung",
    "phase_40_loudness_normalization":    "Lautheits-Normalisierung",
    "phase_42_vocal_enhancement":         "Vokal-Verbesserung",
    "phase_43_ml_deesser":                "ML-De-Esser",
    "phase_44_guitar_enhancement":        "Gitarren-Verbesserung",
    "phase_46_spatial_enhancement":       "Raeumlichkeit",
    "phase_49_advanced_dereverb":         "Erweitertes-Dereverb",
    "phase_52_piano_restoration":         "Klavier-Restaurierung",
    "phase_55_diffusion_inpainting":      "Diffusions-Inpainting",
    "phase_58_lyrics_guided_enhancement": "Textgefuehrte-Verbesserung",
    "phase_65_vocal_naturalness_restoration": "Vokal-Natuerlichkeit",
    "phase_66_stem_targeted_nr":          "Stem-NR",
}


def phase_human_name(phase_id: str) -> str:
    """Return a human-readable name for a phase_id, or the id itself."""
    # Strip common prefixes
    for prefix in ("backend/core/phases/", "phases/"):
        if phase_id.startswith(prefix):
            phase_id = phase_id[len(prefix):]
    if phase_id.endswith(".py"):
        phase_id = phase_id[:-3]
    return _PHASE_NAMES.get(phase_id, phase_id)
