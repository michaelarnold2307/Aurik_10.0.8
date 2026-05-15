---
applyTo: "backend/core/phases/phase_*.py"
---

# Phasen-Regeln (normativ, Aurik 9.12.x)

## Pflicht-Checkliste bei jeder neuen Phase

```
1. process()-Signatur: (audio, sr, material_type, strength, **kwargs) → np.ndarray
2. assert sr == 48000 am Eingang
3. audio = np.clip(audio, -1.0, 1.0) am Ausgang
4. result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) vor Return
5. logger.info("phase=%s score=%.2f", phase_id, score) — kein print()
6. CAUSE_TO_PHASES + CAUSES bidirektional ergänzen (V12)
7. Neue HPF/Notch-Phase: 4-stufige Checkliste (s. unten)
8. Wenn panns_singing ≥ 0.25: apply_hnr_blend() nach jeder ML-NR-Phase (§0p)
9. Wenn panns_singing ≥ 0.25: Formant-Delta via lpc_formant_tracker prüfen (§0p)
   → ΔF1–F4 > ±2 dB: sofortiger Rollback auf Phase-Input — keine Ausnahme
10. Vibrato-Zonen (4–7 Hz F0): strength = min(strength, 0.20) (§0p)
    → detect_performance_artifacts() VOR der Phase ausführen
```

## §2.46 Carrier-Chain-Inversion — Stufenreihenfolge (HARD)

```
Stufe 1: ADC-Artefakte     → phase_30 (DC), phase_31 (Quantisierung)
Stufe 2: Playback          → phase_04 (RIAA), phase_25 (Azimuth), phase_12 (Wow/Flutter)
         (VERBOTEN: phase_06 hier — phase_06 = BW-Erweiterung → gehört zu Stufe 5!)
         RIAA-Entzerrkurve: IEC 60098 / RIAA-Standard:
           Zeitkonstanten: 3180 μs (50 Hz Pol) | 318 μs (500 Hz Pol/Null) | 75 μs (2122 Hz Null)
           Boost LF: +20 dB bei 50 Hz; Cut HF: -20 dB @ 20 kHz (vor Entzerrung)
           Entzerrung: invers (LF anheben, HF bedämpfen)
           VERBOTEN: Näherungen ohne alle 3 Zeitkonstanten
Stufe 3: Alterungsschäden  → phase_09 (Knistern), phase_24 (Dropout)
Stufe 4: Carrier subtraktiv→ phase_29 (Bandrauschen), phase_03 (Surface Noise)
Stufe 5: Carrier additiv   → phase_06/23 (BW-Erweiterung), phase_07 (Harmonik)
                              ↑ IMMER nach Stufe 4 — sonst werden rekonstruierte
                                Obertöne sofort entrauscht
Stufe 6: Mixer/Preamp      → BEWAHREN (Recording-Chain-Signatur = Original)

Digitale Carrier-Kette (MP3/AAC/OGG) — Reihenfolge HARD:
  Phase A: phase_50 (Psycho-Masking-Residue, Pre-Echo) — VOR analoger Stufe 1-4
  Phase B: phase_23 (BW-Erweiterung auf Codec-Cutoff) — nach Stufe 4
  Phase C: wenn analog+digital kombiniert: analog Stufe 1-4 → digital Phase A-B
  MP3-Pre-Echo (HARD): siehe unten §MP3
```

## §2.46e Hallucination-Guard — ADDITIVE Phasen (phase_37/38/48/32)

```python
# PFLICHT nach jeder additiven Operation:
from backend.core.dsp.hallucination_guard import check_hallucination

guard = check_hallucination(pre_audio, post_audio, sr, mode)
if guard.requires_rollback:
    return pre_audio  # spectral_novelty > 0.15 in Restoration
if guard.score_penalty > 0:
    phase_score -= 0.3  # spectral_novelty > 0.08

# Drei verbotene Halluzinations-Kategorien in Restoration:
# 1. Harmonik über BW-Ceiling des Materials
# 2. Raumklang/Reverb der nicht im Signal nachweisbar ist
# 3. ML-generierte Spektral-Texturen ohne physikalisches Gegenstück
```

## §2.46f Natural-Performance-Artifacts-Guard

```python
# Diese drei Kategorien sind KEINE Defekte — niemals entfernen:
# 1. Atemgeräusche: -55 bis -40 dBFS, 50-500ms, spectral_flatness > 0.4
# 2. Vibrato/Portamento: F0-Modulation 4-7 Hz, Amplitude ≤ ±50 Cent
# 3. Early Reflections: 0-50ms nach Onset → Dereverb wet_mix cap = 0.35

from backend.core.dsp.natural_performance_detector import detect_performance_artifacts
protected_segments = detect_performance_artifacts(audio, sr)
# Phasen müssen protected_segments respektieren
```

## §0a — Crossfire-Modus-Invariante (absolut)

```python
# VERBOTEN in Restoration — diese Phasen NIE aktivieren:
_RESTORATION_FORBIDDEN = {
    "phase_21_exciter",           # §0a: kein künstlicher Harmonik-Zusatz
    "phase_35_multiband_compression",  # §0a: nur Studio 2026
    "phase_42_vocal_enhancement", # §0a: nur Studio 2026
}
# Diese Phasen dürfen auch nicht in CAUSE_TO_PHASES für Restoration-Causes stehen
```

## Material-Ceiling-Pflicht bei ADDITIVEN Phasen

```python
from backend.core.dsp.physical_ceiling import _MATERIAL_BW_CEILING_HZ, _MATERIAL_DR_CEILING_DB

# BW-Erweiterung (phase_06/07/23):
max_freq = _MATERIAL_BW_CEILING_HZ[material]  # Shellac ≤ 8kHz, Vinyl ≤ 16kHz
# Keine Harmonik/Energie über max_freq hinzufügen

# DR-Expansion (phase_26):
max_dr = _MATERIAL_DR_CEILING_DB[material]  # Vinyl ≤ 70dB, Shellac ≤ 45dB
# Expansion über Ceiling = Artefakt → sofortiger Rollback
```

## §2.63 Boundary-Mechanismus (STFT/ML-Phasen)

```python
# KANONISCH — Reflect-Padding VOR STFT:
_pad_len = hop_length * 4
audio_padded = np.pad(audio, _pad_len, mode="reflect")
# ... STFT-Verarbeitung ...
audio_out = audio_out[_pad_len: _pad_len + n_original]  # deterministischer Strip

# VERBOTEN: np.pad(..., mode="constant") NACH STFT als primäre Längenkorrektur
# Stereo-Lag-Invariante: L + R MÜSSEN identischen _pad_len und Strip-Offset haben

# Overlap-Pflicht (Konsistenz mit dsp.instructions.md):
# hop_length = n_fft // 4  (75 % Overlap) — für Analyse UND Synthese
# Ausnahme: ML-Phasen (DFN, SGMSE+) nutzen interne Fensterung — nicht überschreiben
```

## HPF/Notch-Phase — 4-stufige Checkliste

```
1. KEIN Loudness-Guard in der Phase-Datei selbst
2. enable_loudness=False in _phase_overrides setzen
3. Phase-ID in _HPF_NOTCH_CUM_RESET_PHASES eintragen
4. _update_positive_makeup_authority aufrufen
```

## §2.63 Stereo-Lag-Invariante

```python
# Wenn L/R separat verarbeitet:
# VERBOTEN: Per-Channel-Resampling als Längenkorrektur
# RICHTIG: identische Kontextlänge, identischer Strip-Offset, identische Zielsamplezahl
# VERBOTEN: assert (durch python -O deaktivierbar — see copilot-instructions.md V01)
if not (len(audio_L_out) == len(audio_R_out) == n_original):
    logger.error("stereo_lag_invariant phase=%s L=%d R=%d expected=%d — cropping",
                 phase_id, len(audio_L_out), len(audio_R_out), n_original)
    audio_L_out = audio_L_out[:n_original]
    audio_R_out = audio_R_out[:n_original]  # UV3 §2.61 fängt verbliebene Abweichungen ab
```

## §MP3/AAC-Artefakte — Spezialfälle (phase_50)

```python
# MP3-Artefakte (MDCT-basierter Codec, ISO/IEC 11172-3):
# 1. Pre-Echo: Schmiervorgabe vor Transienten durch lange MDCT-Fenster (576 Samples)
#    → Symptom: Vorgeräusch 5–20 ms vor Attack (Becken, Plänkselgeräusche, S-Laute)
#    → Korrektur: Transient-gestützte fensterwechsel-adaptive Unterdrückung
#    → VERBOTEN: allgemeine NR auf Pre-Echo (zerstört Original-Transient)
# 2. Psychoakustisches Masking-Residue: Artefakt-Energie in maskierten Banden
#    sichtbar sobald Maskierung abläuft (temporal unmasking)
#    → Korrektur: zeitabhängiger Masking-Floor in diesen Banden höher halten
# 3. HF-Rolloff: bei 128 kbps typisch bei 16 kHz, bei 64 kbps bei 11 kHz
#    → Korrektur: AudioSR/NVSR für BW-Erweiterung VOR RIAA (phase_06/23 nach phase_50)
# 4. Granule-Seams: Block-Grenzen alle 576 Samples (ca. 12 ms @ 48kHz)
#    → Symptom: subtile Amplitudensprünge an Block-Grenzen
#    → Korrektur: Überlapp-Glättung über Grenzen

# AAC-Artefakte (ISO/IEC 13818-7, MPEG-4 AAC):
# 1. Stereo-Downmix-Artefakte (Joint Stereo): ähnlich MP3, aber mit Intensity-Stereo
#    und Mid-Side-Coding → bei tiefen Bitraten Stereobreite kollabiert
# 2. TNS (Temporal Noise Shaping): kann bei Vokalen Präsenz-Region dämpfen
```

## §BW-Erweiterung — Algorithmuswahl (phase_06/07/23)

```python
# Entscheidung nach Frequenzlücke:
# 0–8 kHz fehlt (Shellac/Telefon): AudioSR (Liu et al. 2023, SpeechFlow-Diffusion)
#   → resampelt intern auf 16 kHz, generiert bis 48 kHz mittels Diffusion
#   → VERBOTEN bei Studio 2026 ohne MUSHRA ≥ 3.5 (Hallucination-Guard aktiv)
# 8–16 kHz fehlt (MP3 128kbps): SBR-Heuristik (Spectral Band Replication) oder NVSR
#   NVSR: nicht-DNN-basiert, schneller, weniger Halluzinationsrisiko
# 16–22 kHz fehlt (Vinyl, Tape): harmonische Extrapolation (phase_07) mit G_max-Ceiling
# Material-Ceiling IMMER einhalten (_MATERIAL_BW_CEILING_HZ)
```

## §2.36 LyricsGuidedEnhancement — Offline-Pfad (Lücke 3)

```python
# Aurik ist 100 % offline — Lyrics-Quelle: integriertes Whisper-Tiny ONNX (39 MB)
# Prioritätskaskade (vollständig, keine ad-hoc-Alternative erlaubt):
try:
    # Stufe 1: Whisper-Tiny ONNX (primär, offline, CPUExecutionProvider)
    lge = LyricsGuidedEnhancement()
    transcript = lge.transcribe(audio, sr)  # kein Netzwerk
    if transcript.confidence >= 0.60:
        phoneme_mask = lge.get_phoneme_mask(audio, sr, transcript=transcript)
    else:
        # Stufe 2: DSP-Fallback bei Whisper-Confidence < 0.60
        from backend.core.dsp.phoneme_boundary_detector import detect_phoneme_boundaries_dsp
        phoneme_mask = detect_phoneme_boundaries_dsp(audio, sr)
        # Algorithmus: Energie-Differenz (25 ms Fenster) + ZCR-Schwelle
        # voiced: ZCR < 0.10 | unvoiced: ZCR > 0.15 | plosive: Energie-Spike
        metadata["lyrics_fallback"] = "dsp_phoneme_boundary"
except Exception as e:
    # Stufe 3: Beide fehlgeschlagen — phoneme_mask deaktiviert, kein Absturz
    logger.warning("lge_offline: whisper+dsp failed — phoneme_mask=None: %s", e)
    phoneme_mask = None
    metadata["lyrics_fallback"] = "disabled"

# phoneme_mask[frame] = True → NR-Bypass (Konsonanten-Burst geschützt)
# VERBOTEN: Lyrics-Text loggen oder in metadata schreiben (§2.36 Datenschutz)
```

## Frisson-Schutz in Phasen mit Dynamik-Eingriff

```python
from backend.core.frisson_candidate_detector import get_frisson_detector

frisson_zones = get_frisson_detector().detect(original_audio, sr)
# Klimax-Segmente: NR-Strength × 0.85
# Strophen: NR-Strength × 1.15
# MDEM: Zwei-Stufen pre-SG + post-SG, Floor -1.0 LU
```

## VQI-Gate für Vokal-Phasen

```python
# PFLICHT nach vokal-beeinflussenden Phasen:
from backend.core.musical_goals.vocal_quality_index import compute_vqi

if panns_singing >= 0.35:  # kanonischer Name: panns_singing (≠ panns_singing_confidence)
    # audio_orig=audio_in: Phase-Eingang als Referenz (per-Phase-Degradierungs-Check)
    # 1.0 = keine Degradierung durch Phase; < 0.95 = > 5 % VQI-Regression → Rollback
    result_after = compute_vqi(audio_orig=audio_in, audio_restored=audio_out, sr=sr)
    vqi_after = result_after["vqi"]
    if vqi_after < 0.95:  # VERBOTEN: vqi_before verwenden — ist in Phase-Scope nicht definiert
        return audio_in  # Rollback (per-Phase-VQI-Regression ≥ 0.05)
```

## Passaggio-Glättung

```python
# PFLICHT in Pitch/Register-Phasen:
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal

register_sequence = detect_vocal_register_temporal(audio, sr)
# glättet Brust→Kopf-Übergänge ±5 Frames linear
# verhindert Timbre-Knick bei Passaggio-Sprüngen
```
