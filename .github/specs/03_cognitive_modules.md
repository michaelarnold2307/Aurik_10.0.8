# Aurik 9 — Spec 03: Kognitive Module

> Alle Modul-Specs §2.1–§2.36 + Plugin-Richtlinie.
> Verzeichnis-Konvention: `core/` = physisch `backend/core/`.

---

## §2.1 Pflicht-Kernmodule

| Modul | Datei | Zweck |
|---|---|---|
| `PerceptualEmbedder` | `backend/core/perceptual_embedder.py` | 256-dim L2-normalisierter Einbettungsraum |
| `CausalDefectReasoner` | `backend/core/causal_defect_reasoner.py` | Bayesianisch: 28 DefectTypes → 14 Kausal-Ursachen |
| `GPParameterOptimizer` | `backend/core/gp_parameter_optimizer.py` | RBF-GP + UCB + MOO Pareto-Front |
| `PerceptualQualityScorer` | `backend/core/perceptual_quality_scorer.py` | Gammatone-NSIM+MCD+LUFS+MOS |
| `MusicalGoalsChecker` | `backend/core/musical_goals/musical_goals_metrics.py` | 14 Qualitätsziele |
| `MediumClassifier` | `backend/core/medium_classifier.py` | 15 Materialtypen + 2 Multichannel (CLAP-ML + DSP) |
| `DefectScanner` | `backend/core/defect_scanner.py` | 28 DefectType-Werte |
| `VocalAIEnhancement` | `backend/core/vocal_ai_enhancement.py` | VoiceGender (MALE/FEMALE/CHILD/ANDROGYNOUS) |
| `FeedbackChain` | `backend/core/feedback_chain.py` | Iterative PQS-Qualitätsschleife |
| `ExcellenceOptimizer` | `backend/core/excellence_optimizer.py` | GP-Params + MOO |
| `UnifiedRestorerV3` | `backend/core/unified_restorer_v3.py` | Defect-First-Pipeline-Orchestrator |
| `TransientDecoupledProcessing` | `backend/core/transient_decoupled_processor.py` | HPSS-Trennung allererster Schritt |
| `HarmonicPreservationGuard` | `backend/core/harmonic_preservation_guard.py` | G_floor=0.85 an Harmonik-Bins |
| `MusikalischerGlobalplanDienst` | `backend/core/musikalischer_globalplan.py` | Cross-Phase-Globalplan: 13 Ära-Profile × Genre-Modifikatoren, 17 Phase-Adjustments |
| `PerPhaseMusicalGoalsGate` | `backend/core/per_phase_musical_goals_gate.py` | Rollback pro Phase |
| `EraClassifier` | `plugins/era_classifier_plugin.py` | Ära 1890–2025 |
| `GermanSchlagerClassifier` | `backend/core/genre_classifier.py` | 6-Schicht Zero-Shot |
| `RestorabilityEstimator` | `backend/core/restorability_estimator.py` | < 5 s Vor-Assessment |
| `IntroducedArtifactDetector` | `backend/core/introduced_artifact_detector.py` | Post-Restaurierungs-Artefakte |
| `MicroDynamicsEnvelopeMorphing` | `backend/core/micro_dynamics_envelope_morphing.py` | Letzter Schritt vor Export |
| `MertPlugin` | `plugins/mert_plugin.py` | Music Understanding + Naturalness |
| `DiffWavePlugin` | `plugins/diffwave_plugin.py` | AR-Inpainting für Dropout-Lücken |
| `CrepePlugin` | `plugins/crepe_plugin.py` | Pitch-Tracking f₀, CNN-basiert |
| `FormantTracker` | `plugins/formant_tracker.py` | LPC-Formanten F1–F4 |

---

## §2.3 PerceptualEmbedder

```python
# 256-dim Embedding aus 5 psychoakustischen Kanälen:
# A (96 dim): Multi-Skala STFT (FFT 256/1024/4096)
# B (48 dim): Bark-Skala spezifische Lautheit (Zwicker, 24 Bänder)
# C (36 dim): CQT-Chroma (12 Tonklassen × 3 Zeitfenster)
# D (32 dim): AM/FM-Modulation (8 Träger × 4 Statistiken)
# E (44 dim): HPSS tonisch/perkussiv + Spektralkontrast

embedding = embedder.embed(audio, sr)   # → AudioEmbedding
sim = embedding.cosine_similarity(other)  # ∈ [-1, 1]
# Invariante: ‖embedding.vector‖₂ = 1.0 (immer L2-normalisiert)
```

---

## §2.4 CausalDefectReasoner

```python
# 14 Kausal-Ursachen (≠ 28 DefectTypes des DefectScanners):
#   tape_dropout, tape_hiss, vinyl_crackle, vinyl_warp,
#   electrical_hum, head_misalignment, dc_offset, digital_clip,
#   soft_saturation, head_wear, print_through,
#   riaa_curve_error, aliasing, bias_error

plan = reasoner.reason(defect_scores, material="tape", audio=audio, sr=sr)
# plan.primary_cause     → str
# plan.confidence        → float ∈ [0, 1]
# plan.recommended_phases → List[str] (niemals leer; Fallback: ["phase_03_denoise"])
# plan.phase_parameters  → Dict[str, Dict[str, float]]
# plan.reasoning         → str (Begründung)
# Invariante: sum(cause_probabilities.values()) ≈ 1.0
```

---

## §2.5 GPParameterOptimizer

```python
PARAMETER_SPACE: Dict[str, Tuple[float, float, str]] = {
    "noise_reduction_strength": (0.05, 0.95, "float"),
    "harmonic_boost_db":        (0.0,  6.0,  "float"),
    "ola_crossfade_ms":         (5.0,  60.0, "float"),
    "compression_ratio":        (1.05, 5.0,  "log"),
    "eq_high_shelf_db":         (-6.0, 6.0,  "float"),
    "ar_order":                 (16.0, 128.0,"int"),
    "click_threshold_sigma":    (3.0,  8.0,  "float"),
    "hpf_cutoff_hz":            (10.0, 120.0,"log"),
    "nr_smoothing_ms":          (20.0, 200.0,"log"),
    "declip_threshold":         (0.90, 0.99, "float"),
}
# Gedächtnis-Persistenz: ~/.aurik/gp_memory/<material>.json
# Ab v9.x.x: propose_pareto() (MOO, 14 Objectives) ersetzt propose() als primären Aufruf
```

**MOO Pareto-Front:**

```python
PARETO_OBJECTIVES = [
    "brillanz", "waerme", "natuerlichkeit", "authentizitaet",
    "emotionalitaet", "transparenz", "bass_kraft", "groove",
    "spatial_depth", "tonal_center", "micro_dynamics",
    "timbre_authentizitaet", "separation_fidelity", "artikulation",
]
# propose_pareto() → List[ParameterProposal] (max 5 Pareto-Kandidaten)
```

---

## §2.8 Stimmtyp-Adaptierung (VoiceGender-System)

```python
# VoiceGender-Enum:
class VoiceGender:
    MALE       # F₀ 85–180 Hz, De-Essing 5–10 kHz
    FEMALE     # F₀ 165–255 Hz, De-Essing 6–12 kHz
    CHILD      # F₀ 200–500 Hz, De-Essing 7–14 kHz
    ANDROGYNOUS  # auto-detect
    UNKNOWN    # → FEMALE-Fallback
```

**Vocal-Restaurierungskette (Reihenfolge zwingend):**

```
1. GenderDetector.detect() → VoiceCharacteristics (F₀, Formanten, Breathiness)
2. FCPEPlugin (f₀) → CrepePlugin → pYIN-Fallback
3. FormantTracker (LPC F1–F4) + WORLD-Vocoder-Quervalidierung
   (LPC↔WORLD Abweichung > 15% → WORLD-Wert bevorzugt)
4. BreathDetector → breathiness ratio (Erhalt ±0.05)
5. PhonemeDetector + ConsonantDetector (ZCR > 0.3, Energie 4–16 kHz dominant)
5c. ConsonantEnhancement: HF-Anhebung ≤ +6 dB, SNR_frikativ +3 dB mind.
6. De-Esser (phase_19) + ML-De-Esser (phase_43) stimmtyp-spezifisch
7. VocalAIEnhancement.enhance()
8. Formant-Prüfung: Pearson(F1_before, F1_after) ≥ 0.95
9. Emotionalität: emotion_preservation_score ≥ 0.87
```

**Pflicht-PSOLA**: bei Gesang (PANNs Vocals ≥ 0.4) bei Pitch-Korrektur > ±2 Halbton.

---

## §2.9 Instrument-Phasen-Aktivierungsmatrix

| PANNs-Kategorie | Phase | Schwellwert |
|---|---|---|
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.5 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.5 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.5 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.5 |
| Singing voice / Vocals | `phase_19` + `phase_42` + `phase_43` + VocalAIEnhancement | ≥ 0.40 |

---

## §2.14 EraClassifier

```python
# Erkennungs-Kaskade:
# Tier-1: LAION-CLAP → Nearest-Neighbor zu Ära-Referenz-Ankern
# Tier-2: DSP-Fingerprint → HF-Rolloff + Bandbreiten-Kurve
# Tier-3: Mikrofon-Typ-Heuristik

# decade-Werte: 1890, 1900, ..., 2025 (10-Jahres-Blöcke)
# GP-Optimizer Warmstart:
#   decade ≤ 1940: noise_reduction_strength ~ N(0.90, 0.05)
#   decade ≤ 1960: noise_reduction_strength ~ N(0.75, 0.08)
#   decade ≥ 1970: noise_reduction_strength ~ N(0.50, 0.10)

# EraResult hat ab v9.10.45 is_remaster_suspected: bool
# RemasterDetector: floor_score + bw_score → confidence ≥ 0.35 → is_remaster_suspected=True
```

---

## §2.19 GermanSchlagerClassifier — 6-Schicht Zero-Shot

**Erkennungs-Kaskade (kein vortrainiertes Schlager-Modell nötig):**

| Tier | Methode | Schwellwert |
|---|---|---|
| 1: LAION-CLAP | 7 gewichtete Text-Prompts + 5 negative Prompts | clap_score ≥ 0.26 |
| 2: Akkordeon-AM | Hilbert → Hüllkurven-FFT → Reed-Beating [5–15] Hz + Tremolo [4–8] Hz | accordion_score ≥ 0.60 |
| 3: HSI | CQT-Chroma → Quintenkreis-Übergänge ≤ 2 Schritte → fraction ≥ 0.82 | hsi ≥ 0.82 |
| 4: Rhythmus | madmom RNN → BPM + Metrum (Schunkel/Walzer/Marsch/Disco) | rhythm_score ≥ 0.65 |
| 5: Vokal-Prior | LPC-Formanten F1/F2 → Overlap mit Deutschen Vokal-Polygonen (ä/ö/ü) | Tie-Breaker |
| 6: Melodie-Rep. | MFCC-SSM, Kosinus ≥ 0.85, Mindestabstand 8 s | melodic_rep ≥ 0.42 |

**Ensemble:** ≥ 3 von 5 DSP-Schichten (Tier 2–6) über Schwellwert UND Gesamt-Konfidenz ≥ 0.52 → `is_schlager=True`

```python
SCHLAGER_RESTORATION_PROFILE: dict[str, object] = {
    "soft_saturation_preserve": True,
    "tonal_center_threshold": 0.97,     # verschärft
    "phase_21_exciter_enabled": False,
    "groove_dtw_max_ms": 5.0,           # schärfer als Standard 8.0
    "deessing_target_hz": 6500,
    "deessing_strength_cap": 0.45,
    "brillanz_target": 0.82,            # warm, nicht crisp
    "waerme_target": 0.88,              # erhöht
    "gp_memory_key": "schlager",
}
```

**Laufzeit**: ≤ 4 s/Minute Audio. **Recall**: ≥ 90 % (mit CLAP), ≥ 75 % (nur DSP).

Nutzer-Meldung: „Deutscher Schlager erkannt — Akkordeon-Klangcharakter und Schunkelrhythmus werden sorgfältig bewahrt."

---

## §2.20 Genre-Restaurierungsprofile

```python
JAZZ_RESTORATION_PROFILE = {
    "groove_dtw_max_ms": 4.0,      # Jazz-Timing heilig
    "tonal_center_threshold": 0.92,
    "harmonic_exciter_enabled": False,
    "dereverb_strength_cap": 0.30,
    "compression_ratio_cap": 1.8,   # Jazz lebt von Dynamik
    "gp_memory_key": "jazz",
}

KLASSIK_RESTORATION_PROFILE = {
    "phase_20_dereverb_enabled": False,
    "phase_49_dereverb_enabled": False,  # Konzertsaal-RT60 heilig
    "transient_preservation_strength": 1.0,
    "compression_ratio_cap": 1.3,
    "spatial_depth_threshold": 0.82,
    "gp_memory_key": "orchestral",
}

ROCK_RESTORATION_PROFILE = {
    "transient_preservation_strength": 1.0,
    "brillanz_target": 0.90,
    "soft_saturation_preserve": True,
    "compression_ratio_cap": 2.5,
    "gp_memory_key": "rock",
}

OPER_RESTORATION_PROFILE = {
    "deessing_target_hz": 7000,
    "deessing_strength_cap": 0.35,
    "formant_pearson_threshold": 0.97,
    "phase_20_dereverb_enabled": False,
    "vibrato_rate_tolerance_hz": 0.20,
    "de_esser_voice_adaptive": True,
    "gp_memory_key": "opera",
}
```

---

## §2.27 TransientDecoupledProcessing (TDP)

```python
HPSS_HARMONIC_KERNEL: int = 31    # Frames (Frequenzachse)
HPSS_PERCUSSIVE_KERNEL: int = 31  # Frames (Zeitachse)
PERCUSSIVE_ONLY_PHASES: list[str] = [
    "phase_01_click_removal", "phase_27_click_pop_removal",
]
# Rekombination: audio_out = audio_p_processed + audio_h_processed
# via OLA-Crossfade (Hanning, 10 ms)
# Safety-Net: falls DTW > 8 ms RMS → audio_p_original direkt übernehmen
# Laufzeit: ≤ 0.8 s / Minute Audio
```

---

## §2.28 HarmonicPreservationGuard (HPG)

```python
G_FLOOR_HARMONIC: float = 0.85   # Protected bins (an f₀-Partials)
G_FLOOR_DEFAULT:  float = 0.10   # Alle anderen Bins
MAX_GAIN_CORRECTION: float = 2.0  # Niemals mehr als ×2 anheben
VOICING_CONFIDENCE_MIN: float = 0.60

# Algorithmus:
# 1. CREPE (CPU, full) → f₀(t) mit Voicing-Konfidenz ≥ 0.6
# 2. Harmonisches Gitter: fₙ = n·f₀·√(1+B·n²), n=1..20
# 3. STFT-Bins innerhalb ±3 Cent → protected_bins = True
# 4. Nach NR: |STFT(restored)| < 0.85·H_ref → gain ∈ [1.0, 2.0] + PGHI
```

---

## §2.30 MicroDynamicsEnvelopeMorphing (MDEM)

```python
MAX_GAIN_LU: float = 3.0          # (Restoration-Modus: 2.0 LU)
FRAME_SIZE_SAMPLES: int = 19200   # 400 ms @ 48000 Hz
HOP_SIZE_SAMPLES: int = 9600      # 200 ms (50 % Überlappung)
PEARSON_TARGET: float = 0.93
MIN_LEVEL_LUFS: float = -60.0     # Stille-Segmente: G[k] = 0

# Position: NACH phase_47_truepeak_limiter, LETZTER Schritt vor Export
# Glättung: Savitzky-Golay(G, window=7, polyorder=2)
# True-Peak-Prüfung nach Morphing: −1.0 dBTP zwingend
```

---

## §2.26 RestorabilityEstimator

```python
SCORE_THRESHOLDS = {
    "excellent": 90.0,  # "Exzellent restaurierbar — fast wie Neuaufnahme erwartet."
    "good": 70.0,       # "Gut restaurierbar — deutliche Verbesserung erwartet."
    "fair": 50.0,       # "Mäßig restaurierbar — Restdefekte werden bleiben."
    "poor": 30.0,       # "Schwierig restaurierbar — begrenzt."
}
# < 30: "Sehr schwer restaurierbar — das Material ist stark beschädigt."
# Laufzeit ≤ 5 s (nur DSP-Schnellanalyse, kein ML)
# CLI: --pre-assess Flag
```

---

## §2.36 LyricsGuidedEnhancement (ab 9.10.x)

```python
# LyricsTranscriber: Whisper-Tiny ONNX (39 MB, CPUExecutionProvider, kein Netzwerk)
# Fallback bei Whisper nicht verfügbar: Energie-Segmentierung (DSP)

# ContentAwareProcessor — Salienz-Boosts:
SALIENCY_BOOST = {
    "fricative_stressed":   2.0,   # G_floor = 0.90
    "fricative_unstressed": 1.4,
    "vowel_stressed":       1.6,
    "vowel_unstressed":     1.0,
    "plosive":              1.5,
    "silence":              0.5,
}

# LyricsGuidedTimeline — Shortcut L (Overlay an/aus)
COLOR_MAP = {
    "vowel_stressed":       "#4CAF50",
    "fricative_stressed":   "#FF9800",
    "plosive":              "#29B6F6",
    "silence":              "#B0BEC5",
}
# Datenschutz: Lyrics-Text NIEMALS geloggt, NIEMALS in RestorationResult.metadata
```

---

## §11.6 Plugin-Richtlinie (vollständige Liste)

**Pflicht: Erst diese Liste prüfen, DANN neu schreiben.**

```
# ✅ = lokal gebündelt, kein Download, out-of-the-box

# Vocoder & Synthese
plugins/vocos_plugin.py              → ✅ PRIMÄR (Vocos 24kHz ONNX, 52 MB)
plugins/hifigan_plugin.py            → ✅ Tertiär-Fallback (3,6 MB ONNX)

# Stem-Separation
plugins/mdx23c_plugin.py              → ✅ MDX23C Kim_Vocal_2/Kim_Inst (2×64 MB) PRIÄR
plugins/demucs_v4_plugin.py          → ✅ HTDemucs 6s (Legacy-Fallback, experimental)
plugins/uvr_mdxnet_plugin.py         → ✅ UVR HQ 1–4 (56–64 MB je)
plugins/bs_roformer_plugin.py        → ✅ BS-RoFormer + Mel-RoFormer (SOTA)

# Rauschunterdrückung & Dereverb
plugins/deepfilternet_v3_ii_plugin.py → ✅ PRIMÄR NR (37 MB: enc+dec+erb_dec)
plugins/sgmse_plugin.py              → ✅ Dereverb/Enhancement PRIMÄR (sgmse_plus.ts, 251 MB) — SGMSE+ 2022
plugins/mp_senet_plugin.py           → ✅ Speech/Music Enhancement (mp_senet.onnx, 35 MB) — MP-SENet 2023
plugins/wpe_plugin.py                → ✅ WPE Dereverb (rein DSP, kein Checkpoint)
# VERBOTEN: dccrn_plugin (deprecated — ersetzt durch mp_senet_plugin §4.4)

# Codec-Artefakte
plugins/apollo_plugin.py             → ✅ PRIMÄR Codec-Korrektur (65 MB ONNX)
plugins/resemble_enhance_plugin.py   → ✅ Fallback Apollo (41 MB ONNX)

# Inpainting
plugins/flow_matching_plugin.py      → ✅ Generatives Inpainting PRIMÄR (SOTA, Flow Matching)
plugins/cqtdiff_plus_plugin.py       → ✅ Inpainting ≥ 50 ms (CQTdiff+ ONNX)
plugins/diffwave_plugin.py           → ✅ Inpainting Fallback (552 KB ONNX)
plugins/banquet_vinyl_plugin.py      → ✅ Vinyl-spezifisch (Graph 1,4 MB + Data 90,5 MB)

# Audio-Tagging & MOS
plugins/beats_plugin.py              → ✅ Audio-Tagging PRIMÄR (beats_iter3.onnx, 90 MB) — +10.7 % mAP
plugins/panns_plugin.py              → ✅ Audio-Tagging Fallback (81 KB ONNX)
plugins/versa_plugin.py              → ✅ MOS-Bewertung PRIMÄR (SingMOS-Checkpoint .pth im hub_cache) — VERSA 2024
plugins/visqol_plugin.py             → ViSQOL v3 (PFLICHT: --audio Mode)

# Pitch, Formanten, Stimme
plugins/crepe_plugin.py              → ✅ Pitch-Tracking (85 MB ONNX)
plugins/formant_tracker.py           → LPC F1–F4 (DSP, kein Modell)

# Großmodelle (lazy load)
plugins/rmvpe_plugin.py              → ✅ Pitch-Tracking PRIMÄR (rmvpe.onnx, 26 MB) — RMVPE 2023
plugins/fcpe_plugin.py               → ✅ Pitch-Tracking Fallback (FCPE ONNX)
plugins/mert_plugin.py               → MERT-v1-330M (3,9 GB, lazy load)
plugins/audiosr_plugin.py            → AudioSR BW-Erweiterung (5,9 GB, lazy load)
plugins/matchering_plugin.py         → ✅ Reference Mastering (matchering==2.0.6) — nur Studio 2026

# Ära & Genre
plugins/era_classifier_plugin.py     → EraClassifier (1890–2025)
# core/genre_classifier.py          → GermanSchlagerClassifier (kein Download)
```

**DSP-Fallback PFLICHT für jeden Plugin-Import:**

```python
try:
    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
except (ImportError, FileNotFoundError):
    session = None  # DSP-Fallback aktiv
```

---

## §11.7 denker/ — Kognitive Orchestrierungsschicht

```python
# 10 Sub-Denker koordinieren alle 29 Kernmodule:
from denker import get_aurik_denker, restauriere

denker = get_aurik_denker()                    # Singleton, Thread-sicher
ergebnis = denker.restauriere_komplett(audio, sr=48_000)

# Convenience-Wrapper:
ergebnis = restauriere(audio, sr=48_000)

# Pflicht-Assertions auf AurikErgebnis:
assert np.isfinite(ergebnis.audio).all()
assert np.max(np.abs(ergebnis.audio)) <= 1.0
assert ergebnis.qualitaet >= 0.55  # PQS-MOS-basiert
```

Jeder Sub-Denker folgt §3.2 Singleton-Pattern. SR-Invariante `assert sample_rate == 48000` in jedem. `§6.6-Bindung`: TontraegerketteDenker läuft VOR DefektDenker.

### §11.7a [RELEASE_MUST] Denker-Rollendifferenzierung (v9.10.74)

Die drei Ausführungs-Denker (Stufen 6–8 in `AurikDenker._orchestriere()`) haben **disjunkte Verantwortungen**. Jeder darf **ausschließlich** seine Domäne bearbeiten.

| Stufe | Denker | Domäne | Zweck | Verboten |
|---|---|---|---|---|
| 6 | **ReparaturDenker** | Defekt-Beseitigung | Gezielte DSP-Eingriffe an **bekannten Defekten** (Clicks, Hum, Clipping). Entfernt Störungen, ohne den musikalischen Inhalt zu verändern. | Rekonstruktion, Enhancement, Klangveränderung |
| 7 | **RekonstruktionsDenker** | Rekonstruktion | **Erschafft, was fehlt** — füllt Lücken im Audio-Signal (Dropouts, Silence-Gaps, Tape-Aussetzer). Stellt verloren gegangene Signalanteile wieder her. Erzeugt `ReconstructionContext` mit Hinweisen für UV3. | Klangverbesserung, Defekt-Beseitigung |
| 8 | **RestaurierDenker** | Restaurierung/Erhaltung | **Bewahrt und veredelt, was vorhanden ist** — orchestriert UV3 für die vollständige Restaurierungskette. Schützt den gewollten Klangcharakter (Vintage-Ästhetik, Raumeigenschaften, Dynamik). | Lücken-Füllung, gezielte Defekt-Reparatur |

**Kontextfluss (Pflicht)**:

```
DefektDenker (Stufe 3) → defect_result
    ↓
ReparaturDenker (Stufe 6) — nutzt defect_result für gezielte Reparaturen
    ↓ (repariertes Audio)
RekonstruktionsDenker (Stufe 7) — nutzt defect_result + material_hint
    ↓ (rekonstruiertes Audio + ReconstructionContext)
RestaurierDenker (Stufe 8) — nutzt alle Caches + reconstruction_context
```

**`ReconstructionContext`** (Pflicht-Felder):

```python
@dataclass
class ReconstructionContext:
    gaps_found: int               # Anzahl erkannter Lücken
    gaps_repaired: int            # Anzahl erfolgreich gefüllter Lücken
    total_repaired_ms: float      # Gesamte reparierte Zeitdauer
    bandwidth_limited: bool       # True wenn BANDWIDTH_LOSS erkannt
    estimated_original_bandwidth_hz: float  # Geschätzte Original-Bandbreite
    reconstruction_quality: float # Qualität der Rekonstruktion [0, 1]
```

**Invarianten**:

- RekonstruktionsDenker MUSS `defect_result` akzeptieren (optional, für DROPOUT-Severity)
- RekonstruktionsDenker MUSS `ReconstructionContext` zurückgeben
- RestaurierDenker MUSS `reconstruction_context` akzeptieren und an UV3 weitergeben
- AurikDenker._run_rest() MUSS den Kontext zwischen den Stufen durchreichen
