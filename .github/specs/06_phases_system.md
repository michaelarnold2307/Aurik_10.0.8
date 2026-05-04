# Aurik 9 — Spec 06: Phasen-System

> Vollständige Phase-Liste (Phase 01–64), CAUSE_TO_PHASES-Mapping,
> PhaseInterface-Pattern, Namenskonvention für neue Phasen.

---

## §7.1 Vollständige Phase-Liste (exakte Dateinamen in `backend/core/phases/`)

```text
phase_01_click_removal.py           Clicks/Impulse (Median-Detektion)
phase_02_hum_removal.py             Brumm 50/60 Hz + Obertöne (Kammfilter)
phase_03_denoise.py                 Breitrauschen (OMLSA + DeepFilterNet)
phase_04_eq_correction.py           Frequenzgang-Korrektur (parametrisch)
phase_05_rumble_filter.py           Tieffrequenzrumpeln (< 20 Hz Hochpass)
phase_06_frequency_restoration.py   Bandbreitenerweiterung / Shelving-EQ + SourceFidelityEQ (§2.42)
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
phase_38_presence_boost.py          Präsenz-Boost (Ära-bewusst: 2–6 kHz, SourceFidelity Mic-Center §2.42)
phase_39_air_band_enhancement.py    Air-Band-Enhancement (Ära-bewusst, BW-Cap §2.42, > 12 kHz)
                                    **[BUG-FIX v9.12.0] RESTORATION-EINSCHRÄNKUNG**: Phase_39 ist im
                                    Restoration-Modus für **alle analogen Materialien VERBOTEN**:
                                    vinyl, shellac, wax_cylinder, wire_recording, tape, reel_tape,
                                    cassette, lacquer_disc. Grund: Air-Band-Erweiterung über das
                                    physikalische BW-Ceiling (§0a §6.2c) des Originals erzeugt
                                    Halluzinationen (+18 dB Air-Band-Energie bei Vinyl > 16 kHz).
                                    Erlaubt: Studio 2026 (beide Modi), digitale Quellen (cd_digital,
                                    mp3_low, mp3_high, dat, md) in Restoration wenn BW-Ceiling passt.
phase_40_loudness_normalization.py  LUFS-Normierung (ITU-R BS.1770-5, −14 LUFS)
phase_41_output_format_optimization.py  Ausgabe-Format-Optimierung
phase_42_vocal_enhancement.py       Gesangs-Enhancement
phase_43_ml_deesser.py              ML-gestützter De-Esser
phase_44_guitar_enhancement.py      Gitarren-Enhancement (PANNs conf ≥ 0.50)
phase_45_brass_enhancement.py       Blechbläser-Enhancement (PANNs conf ≥ 0.50)
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
                                    (adaptiver AR-Order: AR(64) < 50 ms, AR(192) ≥ 50 ms)
phase_56_spectral_band_gap_repair.py HEAD_WEAR: Frequenzband-Lücken-Reparatur
                                    (SpectralBandGapRepair — nur bei conf ≥ 0.55)
phase_57_print_through_reduction.py  Print-Through-Reduktion
                                    (bidirektionale LMS: Pre-/Post-Echo getrennt)
phase_58_lyrics_guided_enhancement.py Lyrics-gestütztes Enhancement
                                    (Whisper-Tiny ONNX → wav2vec2-Phonem-Alignment →
                                     ContentAwareProcessor; Latenz ≤ 8 s/min;
                                     produktiver Pfad ausschließlich über
                                     backend/core/lyrics_guided_enhancement.py;
                                     aktiviert wenn Vocals erkannt; §2.36 PFLICHT)
phase_59_modulation_noise_reduction.py Modulationsrauschen-Reduktion
                                    (signalabhängige Rauschminderung bei Bandaufnahmen)
phase_60_inner_groove_distortion_repair.py Inner-Groove-Distortion-Reparatur
                                    (positionsadaptive THD/Asymmetrie-Korrektur)
phase_61_groove_echo_cancellation.py Groove-Echo-Kompensation
                                    (template-basierte Vinyl-Vorecho-Unterdrückung)
phase_62_crosstalk_cancellation.py   Crosstalk-Kompensation
                                    (BSS-basierte Kanalentflechtung)
phase_63_intermodulation_reduction.py Intermodulations-Reduktion
                                    (Volterra-basierte IMD-Tilgung)
phase_64_tape_splice_repair.py       Tape-Splice-Reparatur
                                    (Klick-, Pegel- und Phasendiskontinuität an Klebestellen)
```

**Phase-58-Datenvertrag (bindend ab v9.10.100):**

- Persistiert oder geloggt werden dürfen nur Segmentzeiten, `phoneme_type`, Konfidenzen, Fallback-Flags und aggregierte Zähler.
- Verboten sind Worttext, Transkript, Voll-Lyrics und Roh-Alignment-Tokens in `RestorationResult.metadata`, Checkpoints, Logger-Ausgaben und Debug-UI.
- Legacy-/Forschungsimplementierungen unter `backend/lyrics_guided/` sind nicht Teil des produktiven Phase-58-Vertrags.

---

## §7.1b [RELEASE_MUST] Phase 12 — Tape-Head-Level-Stabilizer v2 (v9.11.2)

`phase_12_wow_flutter_fix` enthält neben Wow/Flutter-Korrektur einen dedizierten
`TAPE_HEAD_LEVEL_DIP`-Pfad mit frequenzabhängiger Kompensation.

### Pflichtverhalten

1. Dip-Erkennung auf RMS-Hüllkurve (20 ms Fenster / 10 ms Hop) mit lokaler p75-Referenz.
2. Spektrale Korrektur in STFT-Domain:
    - Broadband-Gain aus Dip-Defizit
    - zusätzlicher HF-Tilt aus Kontext-vs.-Dip-Spektralverlust
3. SNR-Guard pro Frequenzbin: Bins nahe Noise-Floor dürfen nicht geboostet werden.
4. Asymmetrische Gain-Hüllkurve je Dip:
    - langsamer Onset (~30 %)
    - schnelle Recovery (~10 %)
5. §7.1a/§2.51 Stereo-Kohärenz: linked Stereo-Maske (identische Gain-Maske auf L/R).

### Sicherheitsinvarianten

- `strength < 0.01` → passthrough.
- `max_gain_db <= 15 dB`.
- Dips in quasi-Stille (`< -55 dBFS`) werden nicht repariert.
- NaN/Inf-Guard + hartes Clipping auf `[-1, 1]` am Phasenende.

### Detektor-Kopplung

Dieser Pfad ist primär an `tape_head_contact_instability` gebunden und wird bei
`TAPE_HEAD_LEVEL_DIP`-Severity automatisch aktiviert; siehe Spec 05 §6.4b für
Cross-Material-Fallback und Periodizitäts-Marker.

---

## §7.1a [RELEASE_MUST] Stereo-Kohärenz-Pflicht für Phasen (v9.10.127)

Phasen, die auf Stereo-Audio operieren, dürfen L und R **nicht unabhängig** mit signal-modifizierendem DSP verarbeiten (separates Gate, separater Kompressor, separate Spektralreparatur). Dies erzeugt anti-phasige Transient-Artefakte in 2–3 Frame-Grenzen, die §2.49 korrekt als Phase-Cancellation flaggt und zurückrollt — mit direkter OQS-Auswirkung.

**Verbindliche Implementierungsstrategie pro Phase** (detailliert in §2.51 Spec 02):

| Phase | Pflicht-Strategie | Verbot |
| --- | --- | --- |
| `phase_07_harmonic_restoration` | **M/S-Domain**: Harmonics nur auf Mid, Side unverändert | Separate Oberton-Synthese für L und R |
| `phase_18_noise_gate` | **Linked Stereo**: Gate öffnet wenn `max(L_rms, R_rms) > threshold` | Unabhängige Gate-Entscheidung pro Kanal |
| `phase_23_spectral_repair` | **M/S-Domain**: Spektralreparatur auf Mid; Side minimal/nicht bearbeiten | Separate Lückenfüllung für L und R |
| `phase_24_dropout_repair` | **Linked Stereo**: Dropout-Grenze erkannt wenn BEIDE Kanäle unterfallen; Füllung kohärent | Unabhängige L/R-Dropout-Erkennung |
| `phase_35_multiband_compression` | **Linked Stereo**: Gain auf `√(L²+R²)/√2`; gleicher Gain für L und R | Separate GR-Berechnung pro Kanal |

**Zusatzpflicht für `phase_23_spectral_repair` (ML-Pfad, RELEASE_MUST):**

- AudioSR/Apollo-Inference auf Stereo MUSS Mid-zentriert erfolgen (M/S-Domain), Side bleibt unverändert oder nur minimal skaliert.
- **VERBOTEN**: Unabhängige ML-Inferenz auf L und R mit separater Nachkorrektur der Längen.
- **VERBOTEN**: Kanalweises Resampling als primäre Längenkorrektur nach ML-Inferenz.

**Code-Pattern für M/S-Domain** (Referenz-Implementierung):

```python
def _process_ms_domain(audio: np.ndarray, process_fn, side_strength: float = 0.0) -> np.ndarray:
    """Process stereo audio in M/S domain. process_fn anwenden auf Mid;
    Side mit side_strength skaliert (0.0 = unverändert, 1.0 = voll verarbeitet)."""
    assert audio.shape[0] == 2  # channel-first (2, N)
    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0
    mid_processed = process_fn(mid)
    if side_strength > 0.0:
        side_processed = process_fn(side) * side_strength + side * (1.0 - side_strength)
    else:
        side_processed = side
    l = np.clip(mid_processed + side_processed, -1.0, 1.0)
    r = np.clip(mid_processed - side_processed, -1.0, 1.0)
    return np.stack([l, r], axis=0)
```

**Code-Pattern für Linked Stereo** (Referenz-Implementierung):

```python
def _compute_linked_gain(l_audio, r_audio, gain_fn) -> np.ndarray:
    """Gain-Kurve auf kombiniertem RMS berechnen; identisch auf L und R anwenden."""
    combined_rms = np.sqrt((l_audio ** 2 + r_audio ** 2) / 2.0 + 1e-12)
    gain_curve = gain_fn(combined_rms)  # z.B. Gate-Öffnen, Kompressor-GR
    l_out = np.clip(l_audio * gain_curve, -1.0, 1.0)
    r_out = np.clip(r_audio * gain_curve, -1.0, 1.0)
    return np.stack([l_out, r_out], axis=0)
```

---

## §7.1c [RELEASE_MUST] Phasen-Familien-Taxonomie — ADDITIVE / SUBTRAKTIVE / DYNAMICS (v9.11.14)

Jede Phase gehört zu **genau einer** Phasen-Familie. Die Zuordnung steuert:

- `_post_additive_bw_guard()` (§6.2c): BW-Ceiling-Guard NUR nach ADDITIVE-Block
- `NoiseTextureCoherenceGuard` (§4.7): nur nach SUBTRAKTIVE-Phasen
- `ArtifactFreedomGate` Roughness/Sharpness-Penalty (§2.49c): nur DYNAMICS/ADDITIVE/ENHANCEMENT

| Familie | Phase-IDs | Beschreibung |
| --- | --- | --- |
| **SUBTRAKTIVE** | phase_01, phase_02, phase_03, phase_05, phase_09, phase_10, phase_17, phase_18, phase_19, phase_20, phase_22, phase_24, phase_27, phase_28, phase_29, phase_30, phase_31, phase_49, phase_50, phase_60, phase_63 | Entfernt Energie (Rauschen, Knistern, Klicks, Reverb, DC-Offset) |
| **ADDITIVE** | phase_06, phase_07, phase_21, phase_23, phase_37, phase_38, phase_39, phase_55, phase_56 | Fügt Frequenzinhalt hinzu (BW-Extension, Harmonik, Inpainting) |
| **DYNAMICS** | phase_11, phase_26, phase_33, phase_34, phase_35, phase_36, phase_40 | Verändert Dynamik/Lautstärke (Kompression, Expansion, Normalisierung) |
| **CORRECTION** | phase_04, phase_08, phase_12, phase_13, phase_14, phase_15, phase_25, phase_32, phase_41 | Korrigiert technische Parameter (EQ, Speed, Phase, Azimuth, Stereo) |
| **ENHANCEMENT** | phase_42, phase_43, phase_44, phase_45, phase_46, phase_47, phase_48, phase_51, phase_52, phase_53, phase_54, phase_57, phase_58, phase_59, phase_61, phase_62, phase_64 | Instrument-spezifisch, Stem-Enhancement, Lyrics-guided |

**Invariante (§6.2c)**: UV3 `_post_additive_bw_guard()` MUSS nach der letzten ADDITIVE-Phase im Pipeline-Block laufen. Neue Phasen MÜSSEN hier zugeordnet werden — fehlende Zuordnung ist ein RELEASE-Blocker.

**[RELEASE_MUST] Ceiling-Pflicht für ADDITIVE- und DYNAMICS-Phasen**:

Jede ADDITIVE-Phase MUSS `_MATERIAL_BW_CEILING_HZ` (Spec 05 §6.2c) respektieren:

| ADDITIVE-Phase | Ceiling-Referenz | Guard |
| --- | --- | --- |
| `phase_06` (BW-Extension) | `_MATERIAL_BW_CEILING_HZ[material]` | Hard-Cap auf generiertes HF-Band |
| `phase_07` (Harmonik) | `_MATERIAL_BW_CEILING_HZ[material]` | Obertöne oberhalb Ceiling unterdrücken |
| `phase_23` (Spectral Repair) | `_MATERIAL_BW_CEILING_HZ[material]` | Apollo/AudioSR-Output BW-begrenzen |
| `phase_39` (Air-Band) | `_MATERIAL_BW_CEILING_HZ[material]` | Air-Band nur bis Ceiling |
| `phase_55` (Inpainting) | `_MATERIAL_BW_CEILING_HZ[material]` | Diffusion-Output BW-begrenzen |
| `phase_56` (Band-Gap) | `_MATERIAL_BW_CEILING_HZ[material]` | Gap-Fill nur bis Ceiling |

Jede DYNAMICS-Phase mit Expansion MUSS `_MATERIAL_DR_CEILING_DB` (Spec 05 §6.2b) respektieren:

| DYNAMICS-Phase | Ceiling-Referenz | Guard |
| --- | --- | --- |
| `phase_26` (DR-Expansion) | `_MATERIAL_DR_CEILING_DB[material]` | Expansion bis Ceiling, nicht darüber |
| `phase_54` (Transparent Dynamics) | `_MATERIAL_DR_CEILING_DB[material]` | DR-Change begrenzt auf Ceiling |

**Invariante**: Fehlender Ceiling-Check in einer ADDITIVE/DYNAMICS-Phase ist ein §0a-Verstoß (Material-Ceiling-Überschreitung = Artefakt).

**Implementierung**: `backend/core/phase_ontology.py` — `PHASE_FAMILY_MAP: dict[str, str]`

---

## §7.2 CAUSE_TO_PHASES-Mapping (CausalDefectReasoner)

**Merge-Regel mit `_MATERIAL_PRIORITY_PHASES` (§6.2a)**:

Beide Dicts liefern unabhängig voneinander Phase-Listen. Die resultierende Phase-Selektion
ist die **Vereinigung (Union)** beider Quellen:

```
selected = set(MATERIAL_PRIORITY_PHASES[material]) | set()
for cause in detected_causes:
    selected |= set(CAUSE_TO_PHASES[cause])
```

- Material-Phasen sind **immer aktiv** (§6.2a Invariante), auch wenn kein Defect sie triggert.
- Cause-Phasen fügen kontextspezifische Reparatur hinzu (z. B. `vocal_harshness` → `phase_43`).
- Duplikate werden durch die Vereinigung automatisch entfernt.
- `GoalApplicabilityFilter` darf einzelne Phasen nachträglich deaktivieren (z. B. phase_48 bei Mono).

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
    "print_through":             ["phase_57_print_through_reduction", "phase_29_tape_hiss_reduction",
                                  "phase_24_dropout_repair", "phase_03_denoise", "phase_23_spectral_repair"],
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
    # Sibilanten (§6.3 v9.10.57):
    "sibilance":                 ["phase_19_de_esser", "phase_43_ml_deesser",
                                  "phase_42_vocal_enhancement"],
    # Transport-Bump (v9.10.57b — Kassetten-Holpern):
    "transport_bump":            ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_31_speed_pitch_correction"],
    # Vocal-Harshness (v9.10.77 — Vokal-Härte/Übersteuerung/Kratzigkeit):
    "vocal_harshness":           ["phase_42_vocal_enhancement", "phase_19_de_esser",
                                  "phase_43_ml_deesser", "phase_23_spectral_repair"],
    # Neu v9.10.97/98:
    "tape_start_instability":    ["phase_12_wow_flutter_fix", "phase_25_azimuth_correction",
                                  "phase_31_speed_pitch_correction", "phase_14_phase_correction",
                                  "phase_24_dropout_repair"],
    "tape_head_contact_instability": ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_26_dynamic_range_expansion"],
    "modulation_noise":          ["phase_59_modulation_noise_reduction", "phase_03_denoise",
                                  "phase_29_tape_hiss_reduction"],
    "inner_groove_distortion":   ["phase_60_inner_groove_distortion_repair", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "groove_echo":               ["phase_61_groove_echo_cancellation", "phase_20_reverb_reduction",
                                  "phase_03_denoise"],
    "crosstalk":                 ["phase_62_crosstalk_cancellation", "phase_14_phase_correction",
                                  "phase_34_mid_side_processing"],
    "intermodulation_distortion": ["phase_63_intermodulation_reduction", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "tape_splice_artifact":      ["phase_64_tape_splice_repair", "phase_01_click_removal",
                                  "phase_24_dropout_repair"],
    "hf_remanence_loss":         ["phase_06_frequency_restoration", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "stylus_damage":             ["phase_09_crackle_removal", "phase_23_spectral_repair",
                                  "phase_60_inner_groove_distortion_repair"],
    "sticky_shed_residue":       ["phase_24_dropout_repair", "phase_29_tape_hiss_reduction",
                                  "phase_03_denoise"],
    "multiband_wow_flutter":     ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_08_transient_preservation"],
    "generation_loss":           ["phase_06_frequency_restoration", "phase_03_denoise",
                                  "phase_23_spectral_repair", "phase_04_eq_correction"],
    "motor_interference":        ["phase_02_hum_removal", "phase_03_denoise",
                                  "phase_29_tape_hiss_reduction", "phase_04_eq_correction"],
}
# PFLICHT: Jede neue Ursache → Eintrag hier UND in allen Material-Prior-Tabellen des DefectScanners.

# [RELEASE_MUST] §7.2a Severity-Weighted Phase-Reorder bei ≥3 Simultandefekten (v9.10.100+):

### §6.9b [RELEASE_MUST] Phase-50 Team-Kohärenz + CONFLICT_REGISTRY (v9.11.5, erweitert v9.11.7)

`phase_50_spectral_repair` ist nach `phase_06_frequency_restoration`,
`phase_07_harmonic_restoration` oder `phase_23_spectral_repair` als
**kooperative Folgephase** zu behandeln.

**Normative Invarianten**:

- UV3 muss `prior_phase_context` an Folgephasen durchreichen.
- UV3 muss für jede erfolgreiche Phase den Ontologie-Typ in den Kontext schreiben
  (`last_phase_type`, Typ-Counter, angewendete Typ-Familien).
- PMGG muss diesen Kontext **für alle Phasen** über eine zentrale Übergangs-Policy
  auswerten (Spec 02 §2.29e).
    wenn die erkannte Regression dem Team-Policy-Grund
    `phase50_after_hf_restoration` entspricht.

**CONFLICT_REGISTRY (v9.11.7)** — `backend/core/phase_ontology.py`:

Für alle aktiven Phasen gilt: UV3 prüft vor jeder Phasen-Ausführung, ob im
`CONFLICT_REGISTRY` ein Eintrag existiert, der die aktuelle Phase als potentiell
neutralisierend einstuft. Falls ja, erhält die Phase `conflict_with_prior_phases: list[str]`.

Die Phase selbst entscheidet, wie sie mit `conflict_with_prior_phases` umgeht —
typischerweise: konservativere Threshold, Schutz bestimmter Frequenzbereiche.
`phase_50` nutzt bereits `hf_protected_bin_start` (v9.11.4) für genau diesen Zweck.

**Team-Telemetrie (v9.11.7)**:

UV3 schreibt `metadata["team_coordination"]` nach jeder Pipeline:
- `event_count`: Anzahl Phasen, bei denen Team-Policy aktiv war
- `events`: Liste mit `phase_id`, `action`, `reason`, `excluded_goals`, `threshold_mult`, `strength_cap`
- `phase_type_summary`: Häufigkeiten der Phase-Operationstypen (SUBTRACTIVE, ADDITIVE, etc.)

**Rationale**: Die Phasenkette arbeitet als Team; spätere Reparaturphasen
dürfen frühere restaurative Interventionen nicht indirekt neutralisieren.
# Wenn CausalDefectReasoner für ein Audio-Segment ≥3 Ursachen mit
# max_severity ≥ 0.70 identifiziert, DARF die kanonische CAUSE_TO_PHASES-Reihenfolge
# durch folgende Regel überschrieben werden:
#
#   phase_order = sorted(all_activated_phases,
#       key=lambda p: (severity_for_phase(p), -phase_number(p)), reverse=True)
#
# Rationale: Ein schwerer Dropout (severity=0.92) generiert nach phase_03_denoise mehr
# Information (Stille statt Rauschen) als wenn Hum (severity=0.45) zuerst bearbeitet wird.
# Die Defektinteraktion ist nicht linearkumulativ — frühe Schlüsselphasen erhöhen die
# Qualität nachfolgender Phasen messbar.
#
# Implementierung in `CausalDefectReasoner.build_restoration_plan()`:
#   if len([d for d in defects if d.severity >= 0.70]) >= 3:
#       plan.phases = _severity_reorder(plan.phases, defect_severity_map)
# Dabei gilt: PMGG-Exclusions und Phase-Validity-Regeln bleiben unverändert.
# VERBOTEN: Reorder bei < 3 Hochseverity-Defekten (destabilisiert gut getestete Reihenfolge).
```

---

## §7.3 Instrument-Kontexterkennung (PANNs-Aktivierungsmatrix)

| PANNs-Kategorie | Aktivierte Phase | Confidence-Schwelle |
| --- | --- | --- |
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.50 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.50 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.50 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.50 |
| Singing / Vocals | `phase_19_de_esser` + `phase_42_vocal_enhancement` + `phase_43_ml_deesser` + VocalAIEnhancement | ≥ 0.40 (Soft 0.35–0.40: 50 % Strength) |

> **Invariante** (v9.10.83): Instrument-Schwelle ist einheitlich **0.50** für alle Instrumente. Höherer Wert (z.B. 0.60) blockiert Enhancement bei Ensemble-Aufnahmen mit mehreren gleichzeitigen Instrumenten. Änderungen hier → immer auch `backend/core/unified_restorer_v3.py` L≈5822 + `plugins/panns_plugin.py` Docstring anpassen.

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
# □ Stereo-Kohärenz (§2.51 Spec 02 / §7.1a Spec 06):
#     Wenn Stereo-Audio verarbeitet wird: M/S-Domain ODER Linked-Stereo —
#     KEIN unabhängiges L/R-Processing mit gain- oder zeitvarianter Operation
```

---

## §7.5 Parallelisierungs-Invariante (Pipeline-Tiers)

```text
TIER 0 + TIER 1: IMMER sequenziell (TransientDecoupledProcessing, Click-Removal)
TIER 2–4: Dürfen parallelisieren; Merge via np.mean NUR wenn gleiche Frequenzzone
TIER 6:   IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)
```

---

## §7.6 Adaptive Chunk-Verarbeitung (ab 5 Minuten Dateilänge)

| Defektdichte (lokal) | Chunk-Größe | Begründung |
| --- | --- | --- |
| Hoch (severity ≥ 0.6) | **5 s** | Feingranulare Kontrolle |
| Mittel (0.3 ≤ severity < 0.6) | **15 s** | Balance Qualität/Rechenzeit |
| Niedrig (severity < 0.3) | **60 s** | Kontextkohärenz |
| Stille-Segmente | **120 s** | Passthrough ohne DSP |

**Implementierung**: `backend/core/adaptive_chunk_processor.py`

```python
from backend.core.adaptive_chunk_processor import process_in_adaptive_chunks

# Opt-in für Phasen, die von feinerer Chunk-Verarbeitung profitieren:
result = process_in_adaptive_chunks(
    phase_fn=my_phase_process,
    audio=audio,
    sr=48000,
    max_severity=defect_severity,
    phase_kwargs={"material_type": material, ...},
)
# Crossfade: Hanning-Fenster 10 ms @ 50% Overlap — COLA-konform (Lücke-E-Fix v9.10.100)
#   COLA-Bedingung (Constant Overlap-Add): sum(w[n-H], w[n]) = 1.0 für alle n
#   → Hanning + 50 % Overlap (Hop = fsize/2) ist COLA-konform — kein Amplitudeneinbruch
#   → Konkret: fsize = 480 Samples (10 ms @ 48 kHz), hop = 240 Samples (5 ms)
#   → Jeder Chunk-Rand-Sample ist genau einmal von Hanning = 1.0-Summe abgedeckt
#   VERBOTEN: Hanning mit Hop > fsize/2 (erzeugt periodische −0.5–1.5 dB Dips @ jeder
#             Chunk-Grenze — besonders hörbar bei 5-s-Chunks mit hoher Defektdichte)
# Minimum: 2 s | Maximum: 120 s
# Segment-Grenzen (SegmentAdaptiveProcessor) haben Vorrang vor Chunk-Grenzen
#
# [RELEASE_MUST] §7.6a Chunk-Boundary-Transient-Guard (v9.10.100+):
# Liegt ein Transient (onset_strength > 0.35, aus librosa.onset.onset_strength) innerhalb
# von ±20 ms einer geplanten Chunk-Grenze, MUSS die Grenze um +25 ms nach vorne verschoben
# werden (weg vom Transient).
# Begründung: Ein Drum-Hit oder Plosiv exakt auf der Crossfade-Grenze führt selbst bei
# COLA-konformem Hanning zu hörbarem Transient-Smearing (±0.3–0.8 dB Amplitudenabweichung
# im ersten Attack-Frame, ca. 1–22 ms). Beträgt der Shift mehr als halbe Chunk-Größe,
# Grenze stattdessen um −25 ms nach hinten verschieben.
# Implementierung: `adaptive_chunk_processor._find_safe_boundary(pos_samples, audio, sr)`
```

**Defect-Locations-Flow** (v9.10.75): `_execute_pipeline` extrahiert `defect_locations` (dict[str, list[tuple[float,float]]]) und `max_defect_severity` (float) aus DefectScanner-Ergebnissen und übergibt sie als kwargs an jede Phase. Phasen können Locations als Hints für gezieltere Verarbeitung nutzen (opt-in), erkennen Defekte weiterhin auch eigenständig intern (Redundanz-Prinzip).

**[RELEASE_MUST] Location-Completeness-Invariante**:

- Im Core-Analyse- und Reparaturpfad sind harte Obergrenzen auf `defect_locations` unzulässig.
- Bei hoher Ereignisdichte (auch >1000 Events) muss die vollständige Eventliste für Phasenrouting, PMGG und Nachverarbeitung erhalten bleiben.
- Marker-Reduktion ist nur im UI-Layer erlaubt und darf nie die intern verwendete Defektliste verändern.

---

## §7.7 [RELEASE_MUST] PMGG Inference-Caching bei Retries (§2.29a — ML-deterministische Phasen)

**Kernprinzip**: ML-Modelle sind deterministisch (gleicher Input → gleicher Output). Bei PMGG-Retries wird ML-Inferenz **NICHT** wiederholt. Erster Aufruf mit `strength=1.0` (volle Inferenz) → Cache `audio_full`. Retries variieren ausschließlich Wet/Dry-Blending: `audio_retry = dry + strength × (audio_full − dry)`.

### ML-deterministische Phasen (gecachte Inferenz, nur Wet/Dry-Reblend bei Retry)

| Phase | ML-Modell | Begründung |
| --- | --- | --- |
| `phase_03_denoise` | SGMSE+ (Tier-0, Vokal) / ResembleEnhance / DeepFilterNetV3 / OMLSA (DSP) | ML-Hybrid: Inferenz-Output identisch bei gleichem Input |
| `phase_06_frequency_restoration` | AudioSR | Neurale Bandwidth-Extension deterministisch |
| `phase_09_crackle_removal` | BANQUET ONNX | Blind-Denoising deterministisch |
| `phase_12_wow_flutter_fix` | FCPE/CREPE/pYIN | f₀-Schätzung deterministisch (Timing-Phase: kein Wet/Dry) |
| `phase_18_noise_gate` | Silero VAD | Binary-Mask deterministisch |
| `phase_20_reverb_reduction` | SGMSE+ (Primärpfad) | Reverb-Speech-Separation deterministisch (WPE-DSP-Fallback: muss re-run) |
| `phase_23_spectral_repair` | Apollo (primär) + AudioSR (Fallback) | Spektral-Lückenfüllung deterministisch |
| `phase_24_dropout_repair` | AudioSR | Audio-Generierung deterministisch |
| `phase_29_tape_hiss_reduction` | DeepFilterNet v3 II | HF-Denoising deterministisch (OMLSA-DSP <2 kHz: muss re-run) |
| `phase_42_vocal_enhancement` | BSRoFormer | Stem-Separation deterministisch |
| `phase_55_diffusion_inpainting` | CQTdiff/FlowMatching | Diffusions-Inpainting deterministisch |
| `phase_56_spectral_band_gap` | FCPE/CREPE + Synthese | Noten-Synthese deterministisch |

### Strength-abhängige DSP-Phasen (MÜSSEN bei jedem Retry neu ausgeführt werden)

Alle übrigen Phasen, bei denen `strength` Algorithmus-Parameter steuert (z. B. Filterfrequenz, Kompressionsratio, Sättigungsgrad). Beispiele: `phase_01`, `phase_02`, `phase_04`, `phase_10`, `phase_14`, `phase_17`, `phase_19`, `phase_22`, `phase_25`–`phase_28`, `phase_31`–`phase_41`, `phase_43`–`phase_54`.

**Implementierung**: `PerPhaseMusicalGoalsGate._run_with_retry()` führt `_run_phase(phase, audio, 1.0, kwargs)` genau einmal aus. Retries nutzen `_wet_dry_blend(audio, audio_full, strength, phase)`.

```python
# Wet/Dry-Blend bei Retry (PMGG §2.29a Referenzimplementierung):
def _wet_dry_blend(dry: np.ndarray, wet: np.ndarray, strength: float) -> np.ndarray:
    return np.clip(dry + strength * (wet - dry), -1.0, 1.0)

# Erster Aufruf (strength=1.0, gecacht):
audio_full = _run_phase(phase, audio, strength=1.0, kwargs)  # einmal
# Retries (nur Blend, kein erneuter ML-Run):
audio_retry = _wet_dry_blend(audio, audio_full, retry_strength)
```

> **Invariante**: `phase_58_lyrics_guided_enhancement` ist NICHT ML-deterministisch im PMGG-Sinne — sie operiert als Post-Processing-Modul nach der PMGG-Kette und unterliegt eigenen Retry-Regeln (§2.36).

---

## §7.8 [RELEASE_MUST] Phase-50 HF-Spike-Schutz nach Vorphasen-Restauration (v9.11.4)

Pass-1 Spike-Detektor (11-Bin-Fenster, Threshold-Factor 3.0–4.5) darf durch `phase_07`/`phase_06`
restaurierte Harmoniken **nicht** als Codec-Spikes flaggen.

**Invariante**: `_repair_channel` erhält `hf_protected_bin_start` aus Material-Rolloff-Tabelle
(nur analoge Materialtypen). Bins ≥ `hf_protected_bin_start` sind in Pass-1 ausgeschlossen.
Pass-2 (Frame-Energy-Dropout) bleibt global aktiv.

**VERBOTEN**: Spike-Detektion ohne HF-Schutzzone für analoge Materialien nach Harmonik-Restauration.

> Vollständige Invariante: Spec 02 §2.57a — Algorithmus: Spec 04 §4.7a (Lookup-Tabelle)

## §7.9 [RELEASE_MUST] Phase-09 LPC/AR-Lücken-Interpolation (v9.11.13)

`_interpolate_hybrid()` ist eine **vollständige LPC/AR-Vorhersage** — kein Stub.

**Invariante**: Vorwärts-AR aus Pre-Gap + Rückwärts-AR aus Post-Gap, linear überblendet.
Pol-Stabilisierung (|z| ≥ 0.995 → 0.994). 5 ms Boundary-Crossfade.

**VERBOTEN**: `_interpolate_hybrid()` als Alias für `_interpolate_linear()`.

> Vollständiger Algorithmus: Spec 04 §4.7a

---

## §7.3a Phase-Implementierung — Patterns, Caching & Checkliste (konsolidiert aus Skill new-phase)

### Phasen-Interface (Pflicht für jede Phase)

Jede Phase liegt in `backend/core/phases/phase_XX_<name>.py` und MUSS:

- `def execute(audio, sr, strength=1.0, **kwargs) -> tuple[np.ndarray, dict]` exportieren
- `assert sr == 48000` am Eingang
- `audio = np.clip(audio, -1.0, 1.0)` am Ausgang
- `np.nan_to_num(result)` vor Return
- PhaseResult-dict mit mindestens: `phase_id`, `applied`, `strength`, `metadata`

### §2.29a Inference-Caching bei ML-Phasen

ML-deterministische Phasen: Erster Aufruf mit `strength=1.0` → Cache `audio_full`. Retries nur Wet/Dry-Blend:
`audio_retry = dry + strength × (audio_full − dry)`

**ML-deterministische Phasen** (gecachte Inferenz):
`phase_03`, `phase_06`, `phase_09`, `phase_12`, `phase_18`, `phase_20`,
`phase_23`, `phase_24`, `phase_29`, `phase_42`, `phase_55`, `phase_56`

**Strength-abhängige DSP-Phasen** (bei Retry neu ausführen):
`phase_01`, `phase_02`, `phase_04`, `phase_10`, `phase_14`, `phase_17`, `phase_19`,
`phase_22`, `phase_25`–`phase_28`, `phase_31`–`phase_41`, `phase_43`–`phase_54`

### §2.43 Phase-Preserved Wet/Dry-Blend

STFT-Bereich: `M_blend = (1−α)·M_dry + α·M_wet`, Phase vom Wet-Signal.
Verhindert Phase-Cancellation bei Kopfhörer.

### PHASE_GOAL_EXCLUSIONS — kanonische Tabelle (v9.10.96)

| Phase | Ausgeschlossene Goals | Begründung |
| --- | --- | --- |
| `phase_02` | bass_kraft, authentizitaet, natuerlichkeit, transparenz, groove, timbre_authentizitaet | Kammfilter Hum-Removal |
| `phase_03` | natuerlichkeit, artikulation, authentizitaet, tonal_center, timbre_authentizitaet | CREPE-Load-State + shaped NR |
| `phase_04` | transparenz, brillanz, waerme, authentizitaet, natuerlichkeit, timbre_authentizitaet | EQ |
| `phase_08` | micro_dynamics, artikulation | TDP/HPSS |
| `phase_12` | tonal_center, timbre_authentizitaet | K-S volatile nach Pitch-Korrektur |
| `phase_18` | micro_dynamics, authentizitaet, emotionalitaet, groove | Noise Gate |
| `phase_20` | authentizitaet, natuerlichkeit | SGMSE+ Reverb-Reduction |
| `phase_23` | natuerlichkeit, brillanz, authentizitaet, artikulation, timbre_authentizitaet | AudioSR synthetisiert |
| `phase_24` | natuerlichkeit, brillanz, authentizitaet, artikulation, timbre_authentizitaet | Dropout |
| `phase_29` | artikulation, authentizitaet, natuerlichkeit, tonal_center, timbre_authentizitaet | DeepFilterNet Tape-Hiss |
| `phase_49` | authentizitaet | Dereverb |

**Material-adaptive Relaxation**: `cd_digital`/`dat` → phase_03/phase_29 reduziert auf `{"natuerlichkeit", "artikulation"}`.

### §2.31b Song-Kalibrierungs-Integration (7 PMGG-Schnittstellen)

1. **Threshold**: `global_scalar < 0.85` → ×0.85; `> 1.20` → ×1.15. Begrenzt [0.015, 0.070]
2. **Retry-Leiter**: `initial_strength < 0.90` → Ankerpunkte `[0.80, 0.65, 0.50, 0.35, 0.20]`
3. **Stagnation**: `max(0.002, threshold × 0.15)`
4. **P3-Budget**: tier="good" → 3 Retries; tier="poor" → 1
5. **FeedbackChain target**: Base 0.72/0.78 ±0.035 nach restorability. Begrenzt [0.60, 0.85]
6. **Catastrophic**: `max(0.08, 4.0 × adaptive_threshold)`
7. **Material-adaptive Exclusions**: cd_digital/dat → reduzierter Satz

### PANNs Instrument-Aktivierungsmatrix

| PANNs-Kategorie | Phase | Schwellwert |
| --- | --- | --- |
| Vocals / Singing | phase_19 + phase_42 + phase_43 | ≥ 0.40 / ≥ 0.35 |
| Guitar | phase_44 | ≥ 0.50 |
| Brass / Saxophone | phase_45 | ≥ 0.50 |
| Drum / Percussion | phase_51 | ≥ 0.50 |
| Piano / Keyboard | phase_52 | ≥ 0.50 |

### Vocal-Restaurierungskette (§2.8) — API-Falle

`enhanced, report = self.breath_intelligence.process(audio, sr)` — **KEIN `events`-Argument!**

### §2.36a Phonem-DSP-Klassen

| Klasse | Algorithmus |
| --- | --- |
| fricative | Ramp-Gain 4–8 kHz, KEIN Wiener |
| plosive | TransientShapeGuard (onset gain=1.0), Burst ×1.40 |
| vowel_stressed | LPC Burg → F1–F4 Shelving |
| silence | OMLSA G_floor=0.05 |

### Checkliste neue Phase

```
□ backend/core/phases/phase_XX_<name>.py
□ execute(audio, sr, strength=1.0, **kwargs) → (ndarray, dict)
□ assert sr == 48000; NaN/Inf-Guard + Clip [-1, 1]
□ PMGG-Exclusions festlegen + begründen (§2.55 bidirektional mit CIG!)
□ ML-deterministisch oder strength-abhängig? → Caching-Strategie
□ DSP-Fallback für optionale ML-Imports
□ ml_memory_budget.try_allocate() VOR Modell-Laden
□ defect_locations kwargs opt-in nutzen (§9.1)
□ §2.51 Stereo: M/S-Domain oder Linked-Stereo (kein unabhängiges L/R)
□ ≥ 35 Unit-Tests (Shape, NaN, Bounds, Edge, Mono, Stereo, MG, Groove-DTW, SOFT_SAT, Pass-Through, quality_est)
□ OQS ≥ 80 nachweisbar
□ CHANGELOG.md Eintrag
```
