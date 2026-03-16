# Aurik 9 — Spec 06: Phasen-System

> Vollständige Phase-Liste (Phase 01–56), CAUSE_TO_PHASES-Mapping,
> PhaseInterface-Pattern, Namenskonvention für neue Phasen.

---

## §7.1 Vollständige Phase-Liste (exakte Dateinamen in `backend/core/phases/`)

```
phase_01_click_removal.py           Clicks/Impulse (Median-Detektion)
phase_02_hum_removal.py             Brumm 50/60 Hz + Obertöne (Kammfilter)
phase_03_denoise.py                 Breitrauschen (OMLSA + DeepFilterNet)
phase_04_eq_correction.py           Frequenzgang-Korrektur (parametrisch)
phase_05_rumble_filter.py           Tieffrequenzrumpeln (< 20 Hz Hochpass)
phase_06_frequency_restoration.py   Bandbreitenerweiterung / Shelving-EQ
phase_07_harmonic_restoration.py    Oberton-Rekonstruktion
phase_08_transient_preservation.py  Transientenerhalt (Attack-Schutz)
phase_09_crackle_removal.py         Vinyl-Crackle (Impulsanteil)
phase_10_compression.py             Dynamikkompression
phase_11_limiting.py                Limiting (True-Peak-Schutz)
phase_12_wow_flutter_fix.py         Wow/Flutter-Korrektur (Pitch-Instabilität)
phase_13_stereo_enhancement.py      Stereo-Erweiterung
phase_14_phase_correction.py        Phasen-/Azimuth-Korrektur
phase_15_stereo_balance.py          Stereo-Balance L/R
phase_16_final_eq.py                Finales EQ-Trimming
phase_17_mastering_polish.py        Mastering-Politur
phase_18_noise_gate.py              Noise-Gate (Stille-Segmente)
phase_19_de_esser.py                De-Esser (Sibilanten-Reduktion)
phase_20_reverb_reduction.py        Dereverb / Nachhall-Reduktion
phase_21_exciter.py                 Harmonischer Exciter
phase_22_tape_saturation.py         Tape-Sättigungs-Emulation
phase_23_spectral_repair.py         Spektrale Lücken-Reparatur (Apollo primär)
phase_24_dropout_repair.py          Dropout-Interpolation
phase_25_azimuth_correction.py      Azimuth-Korrektur (Bandlaufwerk)
phase_26_dynamic_range_expansion.py Dynamikbereich-Expansion
phase_27_click_pop_removal.py       Click/Pop-Entfernung (2. Pass)
phase_28_surface_noise_profiling.py Oberflächenrauschen-Profil (Vinyl)
phase_29_tape_hiss_reduction.py     Tape-Hiss + Print-Through (LMS-Adaptivfilter)
phase_30_dc_offset_removal.py       DC-Offset-Entfernung (Hochpass 5 Hz)
phase_31_speed_pitch_correction.py  Geschwindigkeit/Pitch-Korrektur
phase_32_mono_to_stereo.py          Mono→Stereo-Aufweitung
phase_33_stereo_width_limiter.py    Stereo-Breiten-Begrenzer
phase_34_mid_side_processing.py     Mid/Side-Verarbeitung
phase_35_multiband_compression.py   Multibandkompression (transparent)
phase_36_transient_shaper.py        Transient-Shaper
phase_37_bass_enhancement.py        Bass-Fundament-Anhebung
phase_38_presence_boost.py          Präsenz-Boost (2–6 kHz)
phase_39_air_band_enhancement.py    Air-Band-Enhancement (> 12 kHz)
phase_40_loudness_normalization.py  LUFS-Normierung (ITU-R BS.1770-5, −14 LUFS)
phase_41_output_format_optimization.py  Ausgabe-Format-Optimierung
phase_42_vocal_enhancement.py       Gesangs-Enhancement
phase_43_ml_deesser.py              ML-gestützter De-Esser
phase_44_guitar_enhancement.py      Gitarren-Enhancement (PANNs conf ≥ 0.6)
phase_45_brass_enhancement.py       Blechbläser-Enhancement (PANNs conf ≥ 0.6)
phase_46_spatial_enhancement.py     Spatial-Enhancement (Raumklang)
phase_47_truepeak_limiter.py        True-Peak-Limiter (EBU R128, −1.0 dBTP)
phase_48_stereo_width_enhancer.py   Stereo-Breiten-Enhancer (MS, Blumlein, Schroeder)
phase_49_advanced_dereverb.py       Advanced Dereverb (Blind-RIR → WPE)
phase_50_spectral_repair.py         Spektrale Gesamt-Reparatur
phase_51_drums_enhancement.py       Schlagzeug-Enhancement (PANNs conf ≥ 0.5)
phase_52_piano_restoration.py       Klavier-Restaurierung
phase_53_semantic_audio.py          Semantische Audio-Analyse
phase_54_transparent_dynamics.py    Transparente Dynamik-Verarbeitung
phase_55_diffusion_inpainting.py    DiffWave / CQTdiff+ / FlowMatching Inpainting
phase_56_spectral_band_gap_repair.py HEAD_WEAR: Frequenzband-Lücken-Reparatur
                                    (SpectralBandGapRepair — nur bei conf ≥ 0.55)
```

---

## §7.2 CAUSE_TO_PHASES-Mapping (CausalDefectReasoner)

```python
CAUSE_TO_PHASES = {
    "tape_dropout":              ["phase_24_dropout_repair", "phase_55_diffusion_inpainting",
                                  "phase_01_click_removal", "phase_03_denoise"],
    "tape_hiss":                 ["phase_29_tape_hiss_reduction", "phase_03_denoise",
                                  "phase_04_eq_correction", "phase_40_loudness_normalization"],
    "vinyl_crackle":             ["phase_09_crackle_removal", "phase_01_click_removal",
                                  "phase_28_surface_noise_profiling", "phase_03_denoise"],
    "wow_flutter":                ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_04_eq_correction", "phase_03_denoise"],
    "wow":                        ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_04_eq_correction"],
    "flutter":                    ["phase_12_wow_flutter_fix", "phase_08_transient_preservation",
                                  "phase_31_speed_pitch_correction", "phase_03_denoise"],
    "electrical_hum":            ["phase_02_hum_removal", "phase_03_denoise", "phase_04_eq_correction"],
    "head_misalignment":         ["phase_06_frequency_restoration", "phase_04_eq_correction",
                                  "phase_14_phase_correction", "phase_25_azimuth_correction",
                                  "phase_03_denoise"],
    "dc_offset":                 ["phase_30_dc_offset_removal", "phase_40_loudness_normalization"],
    "digital_clip":              ["phase_23_spectral_repair", "phase_06_frequency_restoration",
                                  "phase_40_loudness_normalization"],
    "bandwidth_loss":            ["phase_06_frequency_restoration", "phase_07_harmonic_restoration",
                                  "phase_39_air_band_enhancement"],
    "high_freq_noise":           ["phase_29_tape_hiss_reduction", "phase_03_denoise",
                                  "phase_18_noise_gate"],
    "stereo_imbalance":          ["phase_15_stereo_balance", "phase_33_stereo_width_limiter",
                                  "phase_34_mid_side_processing"],
    "phase_issues":              ["phase_14_phase_correction", "phase_25_azimuth_correction"],
    "pitch_drift":               ["phase_31_speed_pitch_correction", "phase_12_wow_flutter_fix"],
    "reverb_excess":             ["phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
    "print_through":             ["phase_29_tape_hiss_reduction", "phase_24_dropout_repair",
                                  "phase_03_denoise", "phase_23_spectral_repair"],
    "digital_artifacts":         ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_06_frequency_restoration"],
    "compression_artifacts":     ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_26_dynamic_range_expansion", "phase_06_frequency_restoration",
                                  "phase_54_transparent_dynamics"],
    "quantization_noise":        ["phase_23_spectral_repair", "phase_03_denoise",
                                  "phase_06_frequency_restoration"],
    "jitter_artifacts":          ["phase_23_spectral_repair", "phase_12_wow_flutter_fix"],
    "dynamic_compression_excess":["phase_26_dynamic_range_expansion", "phase_54_transparent_dynamics",
                                  "phase_35_multiband_compression"],
    "head_wear":                 ["phase_56_spectral_band_gap_repair", "phase_14_phase_correction",
                                  "phase_06_frequency_restoration"],    "azimuth_error":             ["phase_14_phase_correction", "phase_25_azimuth_correction",
                                  "phase_06_frequency_restoration", "phase_34_mid_side_processing"],
                                  # Azimuth-Korrekturreihenfolge: phase_14 (Phasenkonsistenz L/R)
                                  # dann phase_25 (Kopf-Ausrichtungs-EQ-Kompensation)
                                  # dann phase_34 (M/S zur Restfehler-Kontrolle)    "soft_saturation":           [],  # BEWAHREN — kein destruktiver Eingriff
    "pre_echo":                  ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_08_transient_preservation"],
    "low_freq_rumble":           ["phase_05_rumble_filter", "phase_03_denoise",
                                  "phase_04_eq_correction"],
    "transient_smearing":        ["phase_08_transient_preservation", "phase_36_transient_shaper",
                                  "phase_23_spectral_repair"],
    "clipping":                  ["phase_23_spectral_repair", "phase_06_frequency_restoration"],
    # Neu v9.10.46:
    "riaa_curve_error":          ["phase_04_eq_correction", "phase_06_frequency_restoration",
                                  "phase_07_harmonic_restoration"],
    "aliasing":                  ["phase_03_denoise", "phase_23_spectral_repair",
                                  "phase_50_spectral_repair"],
    "bias_error":                ["phase_04_eq_correction", "phase_03_denoise",
                                  "phase_06_frequency_restoration", "phase_29_tape_hiss_reduction"],
}
# PFLICHT: Jede neue Ursache → Eintrag hier UND in allen Material-Prior-Tabellen des DefectScanners.
```

---

## §7.3 Instrument-Kontexterkennung (PANNs-Aktivierungsmatrix)

| PANNs-Kategorie | Aktivierte Phase | Confidence-Schwelle |
|---|---|---|
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.6 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.6 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.5 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.6 |
| Singing / Vocals / Speech | `phase_19_de_esser` + `phase_42_vocal_enhancement` + `phase_43_ml_deesser` + VocalAIEnhancement | ≥ 0.40 (Speech: ≥ 0.35) |

**Regel**: Instrument-Phasen IMMER nach Defektkorrektur, VOR Mastering.

---

## §7.4 PhaseInterface — Pflicht-Pattern für neue Phasen

Jede neue Phase **muss**:
- In `backend/core/phases/phase_NN_<beschreibung>.py` angelegt werden
- `PhaseInterface` aus `backend/core/phases/phase_interface.py` implementieren
- `process(audio: np.ndarray, **kwargs) -> PhaseResult` bereitstellen
- In `backend/core/phases/__init__.py` exportiert werden
- Ausgang: immer `np.clip(result.audio, -1.0, 1.0)` vor `PhaseResult`-Erzeugung
- SR-Invariante: `assert sample_rate == 48000`

```python
# Checkliste neue Phase:
# □ PhaseInterface implementiert
# □ process() → PhaseResult (kein raw ndarray)
# □ NaN/Inf-Guard: np.clip(audio, -1.0, 1.0), nan_to_num
# □ assert sample_rate == 48000
# □ Export in __init__.py
# □ Eintrag in CAUSE_TO_PHASES (wenn neuer Defekt-Typ)
# □ ≥ 3 Unit-Tests (Erkennung, False-Positive-Rate, Material-Prior)
```

---

## §7.5 Parallelisierungs-Invariante (Pipeline-Tiers)

```
TIER 0 + TIER 1: IMMER sequenziell (TransientDecoupledProcessing, Click-Removal)
TIER 2–4: Dürfen parallelisieren; Merge via np.mean NUR wenn gleiche Frequenzzone
TIER 6:   IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)
```

---

## §7.6 Adaptive Chunk-Verarbeitung (ab 5 Minuten Dateilänge)

| Defektdichte (lokal) | Chunk-Größe | Begründung |
|---|---|---|
| Hoch (severity ≥ 0.6) | **5 s** | Feingranulare Kontrolle |
| Mittel (0.3 ≤ severity < 0.6) | **15 s** | Balance Qualität/Rechenzeit |
| Niedrig (severity < 0.3) | **60 s** | Kontextkohärenz |
| Stille-Segmente | **120 s** | Passthrough ohne DSP |

```python
def adaptive_chunk_size(defect_severity: float, segment_type: str) -> float:
    if segment_type == "silence":
        return 120.0
    if defect_severity >= 0.6:
        return 5.0
    if defect_severity >= 0.3:
        return 15.0
    return 60.0
# Minimum: 2 s | Maximum: 120 s
# Crossfade: Hanning-Fenster, ola_crossfade_ms aus GP-Optimizer
# Segment-Grenzen (SegmentAdaptiveProcessor) haben Vorrang vor Chunk-Grenzen
```
