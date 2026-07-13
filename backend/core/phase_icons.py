"""
Phase Icon Registry — Unicode Icons für alle 68 Phasen + System-Stages

Jede Phase und jeder System-Prozess bekommt ein intuitives Icon,
das in Log-Ausgaben vorangestellt wird.

Kategorien:
  🔍 Analyse       🧹 Reinigung      🎚️ Dynamik
  🎤 Gesang        🔊 Stereo         🎛️ EQ/Frequenz
  🔧 Reparatur     ⚡ Speed/Pitch    🛡️ Sicherheit
  🎸 Instrument    💿 Export/Master  📊 Qualität
  🔗 Assembly      🔌 Interface      🤖 AI/ML
  🎵 Musik         ⚠️ Warnung        ❌ Fehler

Author: Aurik Development Team
Version: 10.0.7
Date: 2026-07-13
"""

# ── Kategorie-Icons ─────────────────────────────────────────────────────

CATEGORY_ICONS: dict[str, str] = {
    "analyse": "🔍",
    "reinigung": "🧹",
    "dynamik": "🎚️",
    "gesang": "🎤",
    "stereo": "🔊",
    "frequenz": "🎛️",
    "reparatur": "🔧",
    "speed": "⚡",
    "sicherheit": "🛡️",
    "instrument": "🎸",
    "export": "💿",
    "qualitaet": "📊",
    "assembly": "🔗",
    "interface": "🔌",
    "ai": "🤖",
    "musik": "🎵",
    "warnung": "⚠️",
    "fehler": "❌",
    "erfolg": "✅",
    "start": "🚀",
    "ende": "🏁",
    "cd_noise": "💿",
    "parallel": "⚡⚡",
}

# ── Phasen-Icon-Mapping (alle 60+ Phasen) ───────────────────────────────

PHASE_ICONS: dict[str, str] = {
    # ── Analyse / Forensik ──
    "phase_01_click_removal": "🔍🧹",
    "phase_02_hum_removal": "🔍🧹",
    "phase_correction": "🔍",
    "phase_interface": "🔌",
    "phase_glue_stage": "🔗",

    # ── Reinigung / Denoising ──
    "phase_03_denoise": "🧹",
    "phase_05_rumble_filter": "🧹",
    "phase_09_crackle_removal": "🧹",
    "phase_23_spectral_repair": "🧹🔧",
    "phase_27_click_pop_removal": "🧹",
    "phase_28_surface_noise_profiling": "🔍🧹",
    "phase_29_tape_hiss_reduction": "🧹",
    "phase_30_dc_offset_removal": "🧹",
    "phase_50_spectral_repair": "🧹🔧",
    "phase_57_print_through_reduction": "🧹",
    "phase_59_modulation_noise_reduction": "🧹",
    "phase_62_crosstalk_cancellation": "🧹🔊",
    "phase_63_intermodulation_reduction": "🧹",
    "phase_66_stem_targeted_nr": "🧹🎤",

    # ── EQ / Frequenz ──
    "phase_04_eq_correction": "🎛️",
    "phase_06_frequency_restoration": "🎛️🔧",
    "phase_16_final_eq": "🎛️",
    "phase_37_bass_enhancement": "🎛️",
    "phase_38_presence_boost": "🎛️",
    "phase_39_air_band_enhancement": "🎛️",
    "phase_56_spectral_band_gap_repair": "🎛️🔧",

    # ── Harmonik / Restoration ──
    "phase_07_harmonic_restoration": "🎵🔧",
    "phase_22_tape_saturation": "🎵",
    "phase_24_dropout_repair": "🔧",

    # ── Transienten / Dynamik ──
    "phase_08_transient_preservation": "🎚️",
    "phase_10_compression": "🎚️",
    "phase_11_limiting": "🎚️",
    "phase_18_noise_gate": "🎚️",
    "phase_26_dynamic_range_expansion": "🎚️",
    "phase_35_multiband_compression": "🎚️",
    "phase_36_transient_shaper": "🎚️",
    "phase_47_truepeak_limiter": "🎚️🛡️",
    "phase_54_transparent_dynamics": "🎚️",

    # ── Stereo / Räumlich ──
    "phase_13_stereo_enhancement": "🔊",
    "phase_15_stereo_balance": "🔊",
    "phase_25_azimuth_correction": "🔊",
    "phase_32_mono_to_stereo": "🔊",
    "phase_33_stereo_width_limiter": "🔊",
    "phase_34_mid_side_processing": "🔊",
    "phase_46_spatial_enhancement": "🔊",
    "phase_48_stereo_width_enhancer": "🔊",

    # ── Speed / Pitch ──
    "phase_12_wow_flutter_fix": "⚡",
    "phase_31_speed_pitch_correction": "⚡",

    # ── Gesang ──
    "phase_19_de_esser": "🎤",
    "phase_42_vocal_enhancement": "🎤",
    "phase_43_ml_deesser": "🎤🤖",
    "phase_58_lyrics_guided_enhancement": "🎤🤖",
    "phase_65_vocal_naturalness_restoration": "🎤🔧",

    # ── Instrumente ──
    "phase_44_guitar_enhancement": "🎸",
    "phase_45_brass_enhancement": "🎸",
    "phase_51_drums_enhancement": "🎸",
    "phase_52_piano_restoration": "🎸",

    # ── Effekte / Hall ──
    "phase_20_reverb_reduction": "🔊",
    "phase_21_exciter": "🎛️",
    "phase_49_advanced_dereverb": "🔊",

    # ── Mastering / Export ──
    "phase_17_mastering_polish": "💿",
    "phase_40_loudness_normalization": "💿",
    "phase_41_output_format_optimization": "💿",

    # ── AI / ML ──
    "phase_53_semantic_audio": "🤖🔍",
    "phase_55_diffusion_inpainting": "🤖🔧",

    # ── Spezial-Reparatur ──
    "phase_60_inner_groove_distortion_repair": "🔧",
    "phase_61_groove_echo_cancellation": "🔧",
    "phase_64_tape_splice_repair": "🔧",
}

# ── System-Prozess-Icons ────────────────────────────────────────────────

SYSTEM_ICONS: dict[str, str] = {
    "restoration_start": "🚀",
    "restoration_end": "🏁",
    "song_calibration": "📊",
    "defect_scan": "🔍",
    "era_classification": "🔍",
    "medium_detection": "🔍",
    "export_pipeline": "💿",
    "cd_noise_profile": "💿",
    "dither": "💿",
    "quality_report": "📊",
    "mushra_score": "📊",
    "abx_test": "📊",
    "vocal_repair": "🎤🔧",
    "stem_separation": "🎤",
    "stcg_pre": "🔊🛡️",
    "stcg_post": "🔊🛡️",
    "stcg_chunk": "🔊🛡️",
    "lag_probe": "🔊",
    "graceful_stop": "🛡️",
    "guard_bypass": "⚠️",
    "guard_enforce": "🛡️",
    "feedback_chain": "🔗",
    "parallel_group": "⚡⚡",
    "error_recovery": "⚠️🔧",
}


# ── API ──────────────────────────────────────────────────────────────────


def phase_icon(phase_id: str) -> str:
    """Gibt das Icon für eine Phase zurück.

    Args:
        phase_id: z.B. 'phase_42_vocal_enhancement' oder 'phase_03_denoise'

    Returns:
        Icon-String oder '🎵' als Fallback.
    """
    # Exakter Match
    if phase_id in PHASE_ICONS:
        return PHASE_ICONS[phase_id]
    # Präfix-Match (z.B. 'phase_42' matcht 'phase_42_vocal_enhancement')
    for key, icon in PHASE_ICONS.items():
        if phase_id.startswith(key.split("_")[0] + "_" + key.split("_")[1]) and len(key.split("_")) >= 3:
            if phase_id.replace("_", "").startswith(key.split("_")[0] + key.split("_")[1]):
                return icon
    return "🎵"


def system_icon(process: str) -> str:
    """Gibt das Icon für einen System-Prozess zurück."""
    return SYSTEM_ICONS.get(process, "⚙️")


def icon_log(phase_id: str, message: str, is_system: bool = False) -> str:
    """Formatiert eine Log-Nachricht mit vorangestelltem Icon.

    Args:
        phase_id: Phase-ID oder System-Prozess-Name.
        message: Die eigentliche Log-Nachricht.
        is_system: True für System-Prozesse, False für Phasen.

    Returns:
        Formatierte Nachricht: '🔍 message'
    """
    icon = system_icon(phase_id) if is_system else phase_icon(phase_id)
    return f"{icon} {message}"
