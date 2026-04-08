# Aurik 9 — Spec 03: Kognitive Module

> Alle Modul-Specs §2.1–§2.43 + Plugin-Richtlinie.
> Verzeichnis-Konvention: `core/` = physisch `backend/core/`.

---

## §2.1 Pflicht-Kernmodule

| Modul | Datei | Zweck |
| --- | --- | --- |
| `PerceptualEmbedder` | `backend/core/perceptual_embedder.py` | 256-dim L2-normalisierter Einbettungsraum |
| `CausalDefectReasoner` | `backend/core/causal_defect_reasoner.py` | Bayesianisch: 32 DefectTypes → 35 Kausal-Ursachen |
| `GPParameterOptimizer` | `backend/core/gp_parameter_optimizer.py` | RBF-GP + UCB + MOO Pareto-Front |
| `PerceptualQualityScorer` | `backend/core/perceptual_quality_scorer.py` | Gammatone-NSIM+MCD+LUFS+MOS |
| `MusicalGoalsChecker` | `backend/core/musical_goals/musical_goals_metrics.py` | 14 Qualitätsziele |
| `MediumClassifier` | `backend/core/medium_classifier.py` | 15 Materialtypen + 2 Multichannel (CLAP-ML + DSP) |
| `DefectScanner` | `backend/core/defect_scanner.py` | 32 DefectType-Werte |
| `VocalAIEnhancement` | `backend/core/vocal_ai_enhancement.py` | VoiceGender (MALE/FEMALE/CHILD/ANDROGYNOUS) |
| `FeedbackChain` | `backend/core/feedback_chain.py` | Iterative PQS-Qualitätsschleife |
| `ExcellenceOptimizer` | `backend/core/excellence_optimizer.py` | GP-Params + MOO |
| `UnifiedRestorerV3` | `backend/core/unified_restorer_v3.py` | Defect-First-Pipeline-Orchestrator |
| `TransientDecoupledProcessing` | `backend/core/transient_decoupled_processor.py` | HPSS-Trennung allererster Schritt |
| `HarmonicPreservationGuard` | `backend/core/harmonic_preservation_guard.py` | G_floor=0.85 an Harmonik-Bins |
| `MusikalischerGlobalplanDienst` | `backend/core/musikalischer_globalplan.py` | Cross-Phase-Globalplan: 13 Ära-Profile × Genre-Modifikatoren, 17 Phase-Adjustments |
| `PerPhaseMusicalGoalsGate` | `backend/core/per_phase_musical_goals_gate.py` | Rollback pro Phase |
| `SongCalibrationProfile` | `backend/core/song_calibration.py` | §2.31a: materialadaptives Kalibrierungsprofil (global_scalar + family_scalars) vor Phasenkette |
| `EraAuthenticPerceptualCompletion` | `backend/core/era_authentic_completion.py` | Ära-authentische Wahrnehmungs-Ergänzung (Quell-BW < 10 kHz); Studio-2026-Kette Schritt 8 |
| `LyricsGuidedEnhancement` | `backend/core/lyrics_guided_enhancement.py` | §2.36 RELEASE_MUST: Whisper-Tiny ONNX → Phonem-Alignment → ContentAwareProcessor |
| `EraClassifier` | `plugins/era_classifier_plugin.py` | Ära 1890–2025 |
| `GermanSchlagerClassifier` | `backend/core/genre_classifier.py` | 6-Schicht Zero-Shot |
| `RestorabilityEstimator` | `backend/core/restorability_estimator.py` | < 5 s Vor-Assessment |
| `IntroducedArtifactDetector` | `backend/core/introduced_artifact_detector.py` | Post-Restaurierungs-Artefakte |
| `MicroDynamicsEnvelopeMorphing` | `backend/core/micro_dynamics_envelope_morphing.py` | Letzter Schritt vor Export |
| `MertPlugin` | `plugins/mert_plugin.py` | Music Understanding + Naturalness |
| `SourceFidelityReconstructor` | `backend/core/source_fidelity_reconstructor.py` | §2.42: Generationsverlust-Kompensation, Ära-BW, FIR-EQ |
| `PerceptualSalienceEstimator` | `backend/core/perceptual_salience.py` | §9.1c: Psychoakustische Salienz-Annotation, Maskierungs-Scoring |
| `MediumDetector` | `forensics/medium_detector.py` | §6.7: Zweiphasige Bayesianische Tonträger-Ketten-Erkennung |
| `DiffWavePlugin` | `plugins/diffwave_plugin.py` | AR-Inpainting für Dropout-Lücken |
| `CrepePlugin` | `plugins/crepe_plugin.py` | Pitch-Tracking f₀, CNN-basiert |
| `FormantTracker` | `plugins/formant_tracker.py` | LPC-Formanten F1–F4 |

### §2.1a [RELEASE_MUST] Exzellenz-API-Kompatibilitätsvertrag (v9.11.1)

Der Orchestrator (`AurikDenker`) MUSS mit beiden Exzellenz-Schnittstellen kompatibel bleiben:

1. Primärpfad: `messe_und_repariere(audio, sr, ...) -> (audio_out, goals_dict)`
2. Legacy-Fallback: `messe_ziele(audio, sr, ...)` (goals-only oder tuple-kompatibel)

**Invarianten:**

- Kein harter Methoden-Bind auf nur eine API.
- Fallback-Pfad muss Stage-Notes eindeutig markieren (`Legacy-Goal-Messpfad`).
- Bei fehlender Primärmethode darf die Pipeline nicht abbrechen, solange Legacy-Pfad verfügbar ist.

### §2.36b Lyrics-Produktivpfad und Datenschutzvertrag (bindend ab v9.10.100)

- Autoritatives Kernmodul für Lyrics-gestützte Verarbeitung ist ausschließlich `backend/core/lyrics_guided_enhancement.py`.
- `backend/lyrics_guided/` ist Legacy-/Forschungsbestand. Diese Module dürfen ohne explizite Freigabe nicht als Produktionspfad, nicht als Referenzimplementierung und nicht als normative Architekturquelle behandelt werden.
- Erlaubte strukturierte Ausgaben aus dem Lyrics-Pfad: `phoneme_type`, Start-/Endzeit, Konfidenz, `fallback_used`, aggregierte Segment-Counts, modellbezogene Statusflags.
- Verbotene Datenflüsse: Lyrics-Worttext, Volltranskript, Roh-Alignment, wortscharfe Tokens in Logs, `RestorationResult.metadata`, Checkpoints, Crash-Dumps oder UI-Debug-Anzeigen.
- Privacy-Invariante: Logging darf ausschließlich phonemische Klassen oder aggregierte Statistiken enthalten. Jede Codeänderung am Lyrics-Pfad muss diese Invariante explizit erhalten.

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
# 35 Kausal-Ursachen (≠ 32 DefectTypes des DefectScanners):
# Hinweis: transport_bump (v9.10.57b) und vocal_harshness (v9.10.77) als
# eigenständige Ursachen ergänzt; Gruppe Pitch/Dynamik dadurch 4→5.
#
# ── Analoge Magnetband-Ursachen (10) ─────────────────────────────────────
#   tape_dropout, tape_hiss, transport_bump, print_through,
#   head_wear, head_misalignment, bias_error,
#   wow, flutter, wow_flutter
#
# ── Vinyl-/Schellack-Ursachen (4) ────────────────────────────────────────
#   vinyl_crackle, vinyl_warp, riaa_curve_error, low_freq_rumble
#
# ── Elektrik / Mechanik (2) ──────────────────────────────────────────────
#   electrical_hum, dc_offset
#
# ── Digital / Codec (8) ──────────────────────────────────────────────────
#   digital_clip, clipping, digital_artifacts, compression_artifacts,
#   quantization_noise, jitter_artifacts, pre_echo, aliasing,
#   dynamic_compression_excess
#
# ── Spektrale Ursachen (2) ───────────────────────────────────────────────
#   bandwidth_loss, high_freq_noise
#
# ── Stereo / Phase (2) ──────────────────────────────────────────────────
#   stereo_imbalance, phase_issues
#
# ── Pitch / Dynamik / Vokal (5) ──────────────────────────────────────────
#   pitch_drift, reverb_excess, transient_smearing, sibilance,
#   vocal_harshness  (v9.10.77 — Vokal-Härte/Übersteuerung/Kratzigkeit 2–6 kHz)
#
# ── Vintage (Schutz) (1) ────────────────────────────────────────────────
#   soft_saturation  (BEWAHREN — P(phases) = leer)

plan = reasoner.reason(defect_scores, material="tape", audio=audio, sr=sr)
# plan.primary_cause     → str
# plan.confidence        → float ∈ [0, 1]
# plan.recommended_phases → List[str] (niemals leer; Fallback: ["phase_03_denoise"])
# plan.phase_parameters  → Dict[str, Dict[str, float]]
# plan.reasoning         → str (Begründung)
# Invariante: sum(cause_probabilities.values()) ≈ 1.0
```

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

```text
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
| --- | --- | --- |
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.5 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.5 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.5 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.5 |
| Singing voice / Vocals | `phase_19` + `phase_42` + `phase_43` + VocalAIEnhancement | Singing ≥ **0.40** (Soft 0.35–0.40: 50 % Strength) |

**[RELEASE_MUST] §2.9a PANNs Soft-Activation-Regel (v9.10.100+):**
Vokale PANNs-Konfidenz im Bereich **0.35–0.40** führt zu 50 % Vocal-Enhancement-Strength
(kein Hard-Block). Hintergrund: Ensemble-Aufnahmen mit schwächem Vocalkanal liefern
typischerweise 0.36–0.39 — ein harter Block bei 0.40 unterbindet Enhancement dort, wo
es klanglich wichtigsten Einfluss hätte.

```python
# Implementierung in UV3 vor Phase-Aktivierung:
vocal_prob = panns_result.get("Singing voice", 0.0)
if vocal_prob >= 0.40:
    vocal_strength_scale = 1.0        # Volle Aktivierung
elif vocal_prob >= 0.35:
    vocal_strength_scale = 0.5        # Soft-Aktivierung (50 % Strength)
else:
    vocal_strength_scale = 0.0        # Keine Vocal-Phasen aktiviert
# vocal_strength_scale wird als kwargs an phase_19/42/43/VocalAIEnhancement übergeben.
```

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
| --- | --- | --- |
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

### §2.19.2 17-Genre-System (Non-Schlager, ab v9.10.103)

`_compute_non_schlager_scores()` berechnet für alle 16 Non-Schlager-Genres einen Wert ∈ [0, 1]
aus 5 akustischen Parametern:

| Parameter | Quelle | Bedeutung |
| --- | --- | --- |
| `centroid_hz` | Spektraler Centroid (Librosa) | Helligkeit / Frequenzschwerpunkt |
| `onset_rate` | Onset-Dichte (Librosa) | Anschlags- / Rhythmusdichte |
| `hsi` | Harmonische Simplizität (CQT) | 1 = sehr schlicht; 0 = sehr komplex |
| `dr_db` | Dynamikumfang (Crest-Faktor, dB) | Kompressionsgrad |
| `bpm` | Tempo (Rhythmus-Tier) | Schläge pro Minute |

**Best-Genre-Auswahl:**
`best = max(scores)`. Genre wird nur vergeben wenn:

- `best_score >= _NON_SCHLAGER_MIN_SCORE` (0.35) UND
- `best_score - second_score >= _OPEN_SET_MARGIN` (0.08)

Andernfalls: `genre_label="Unbekannt"`, `open_set_unknown=True`.

### §2.19.3 Vollständige Genre-Liste (v9.10.103)

| Label | gp_memory_key | Methode |
| --- | --- | --- |
| Schlager | `schlager` | `is_schlager=True` (§2.19) |
| Rock | `rock` | `_score_rock` |
| Jazz | `jazz` | `_score_jazz` |
| Klassik | `orchestral` | `_score_classical` |
| Oper | `opera` | `_score_oper` |
| Pop | `pop` | `_score_pop` |
| Blues | `blues` | `_score_blues` |
| Soul/R&B | `soul_rnb` | `_score_soul_rnb` |
| Country | `country` | `_score_country` |
| Folk | `folk` | `_score_folk` |
| Funk | `funk` | `_score_funk` |
| Electronic | `electronic` | `_score_electronic` |
| Hip-Hop | `hiphop` | `_score_hiphop` |
| Metal | `metal` | `_score_metal` |
| Latin | `latin` | `_score_latin` |
| Gospel | `gospel` | `_score_gospel` |
| Reggae | `reggae` | `_score_reggae` |

### §2.19.4 [RELEASE_MUST] Genre-Diskriminierungs-Regeln (Anti-Falsifikations-Matrix, v9.10.103)

**Zweck:** Verhindert, dass akustisch ähnliche Genres sich gegenseitig verdrücken.
Alle Regeln sind kanonisch im Code-Kommentar jeder `_score_*`-Methode dokumentiert.

#### Harte Centroid-Gates (früher Return 0.0)

| Methode | Gate-Bedingung | Begründung |
| --- | --- | --- |
| `_score_latin` | `centroid_hz < 1800` → `return 0.0` | Kein Blechblas-Anteil → kein Latin |
| `_score_electronic` | `centroid_hz < 2200` → `return 0.0` | Synthese ist inhärent hell; dunkler Centroid + niedriger DR = komprimiertes Akustik-Material, nicht Synthese |
| `_score_hiphop` | `centroid_hz < 1400` → `return 0.0` | Keine Vokal-/Sample-Präsenz im Mid-Range |
| `_score_reggae` | `bpm > 100` → `return 0.0` | Electronic/Hip-Hop bei 120+ BPM explizit ausgeschlossen |

#### Centroid-Fenstergrenzen (Disambiguation Rock/Metal vs. Wärme-Genres)

| Methode | Centroid-Regel | Rock/Metal-Ausschluss |
| --- | --- | --- |
| `_score_funk` | Centroid-Bonus NUR bei `1800 < centroid < 2800` Hz | centroid ≥ 2800 Hz = Rock/Metal-Distortion → kein Funk-Centroid-Bonus |
| `_score_blues` | Pentatonischer Bereich `hsi ∈ [0.38, 0.65]` | Blues hat moderate Harmonie; Jazz-Komplexität (hsi < 0.38) oder Schlager-Simplizität (hsi > 0.65) schließen Blues aus |
| `_score_soul_rnb` | hsi ∈ [0.38, 0.70] | Soul verwendet Gospel-Akkorde; zu simpel (> 0.70) = Schlager, zu komplex (< 0.38) = Jazz |

#### BPM-Kontext-Abhängige Centroid-Bonus-Logik (Latin)

**Problem:** Bossa Nova (bpm 80–130) hat dunklen Streicher-Centroid (~1800–2400 Hz).
Salsa (bpm > 150) hat helle Blechbläser (~2200–4000 Hz). Rock-Signale (bpm 90–150 + centroid 2800+)
würden ohne Kontextprüfung gleichwertig behandelt.

**Regel in `_score_latin`:**

```python
# Salsa (bpm > 150): heller Blechblas-Spectral-Content erwartet
if bpm > 150 and centroid_hz > 2200:
    score += 0.25   # Salsa / Merengue Blechbläser
# Bossa nova / Cumbia (bpm <= 150): dunkleres Streifen-Spektrum
elif bpm <= 150 and 1800 < centroid_hz < 2500:
    score += 0.25   # Bossa nova / Cumbia
# centroid >= 2500 bei bpm <= 150 → marginal; Rock-Gebiet
elif bpm <= 150 and centroid_hz >= 2500:
    score += 0.05
```

**Ergebnis:** Latin(centroid=3200, bpm=120) = 0.80 vs. Rock(centroid=3200, bpm=120) = 0.90 → Margin 0.10 ✓

#### Anti-Jazz-Guard (hsi-Gate)

`_score_jazz`: wenn `hsi >= 0.58` → `return 0.0`.
**Begründung:** Jazz-Harmonik (chromatische Akkorde, Tritond-Substitution) erzeugt niedrigen hsi.
Ein Signal mit harmonisch einfachem Charakter (hsi ≥ 0.58) kann nicht Jazz sein.

#### Folk-Klassik-Guard (DR-Penalty)

`_score_folk`: wenn `dr_db > 40` → `score -= 0.25`.
**Begründung:** Orchestermaterial hat sehr hohe DR (> 40 dB). Folk ist intim und kompakter.
Ohne diese Regel klassifiziert der Scorer Streicherquartette als Folk.

#### Unit-Test-Isolierungs-Invariante

**[RELEASE_MUST]** Alle Unit-Tests, die gezielt ein bestimmtes Genre überprüfen, MÜSSEN
alle anderen 12+ Score-Methoden über `monkeypatch.setattr` auf einen Neutralwert (z. B. `0.10`)
patchen. Andernfalls können Audio-Synthesis-Artefakte (z. B. hohe Onset-Rate bei reinen
Sinustönen durch Onset-Detektor-Störungen) benachbarte Scores hochziehen und die Margin-Prüfung
zum offenen-Set kollabieren lassen.

```python
# Pflicht-Pattern für Isolation-Tests:
for _method in ("_score_pop", "_score_blues", "_score_soul_rnb", "_score_country",
                "_score_folk", "_score_funk", "_score_electronic", "_score_hiphop",
                "_score_metal", "_score_latin", "_score_gospel", "_score_reggae"):
    monkeypatch.setattr(clf, _method, lambda *_a, **_k: 0.10)
```

---

## §2.20 Genre-Restaurierungsprofile

### [RELEASE_MUST] Aktivierungsregeln (v9.10.102)

Jedes Genre-Profil wird NUR aktiviert wenn die zugehörige PANNs-Kategorie den Schwellwert
überschreitet. Die Erkennung läuft parallel im `_run_genre_classifier`-Thread (§9.7.2 Spec 08).

| Genre-Profil | Aktivierungsbedingung | Priorität |
| --- | --- | --- |
| `SCHLAGER_RESTORATION_PROFILE` | `GermanSchlagerClassifier.is_schlager=True` (§2.19) | 1 (höchste) |
| `OPER_RESTORATION_PROFILE` | PANNs `"Opera"` ≥ 0.45 **oder** `"Singing"` ≥ 0.50 + `formant_pearson ≥ 0.90` | 2 |
| `KLASSIK_RESTORATION_PROFILE` | PANNs `"Orchestra"` ≥ 0.45 **oder** `"Classical music"` ≥ 0.40 | 3 |
| `JAZZ_RESTORATION_PROFILE` | PANNs `"Jazz"` ≥ 0.40 **oder** `"Blues"` ≥ 0.40 | 4 |
| `ROCK_RESTORATION_PROFILE` | PANNs `"Rock music"` ≥ 0.40 **oder** `"Electric guitar"` + `"Drum"` beide ≥ 0.35 | 5 (niedrigste) |

**Kollisionregel:** Nur ein Profil ist aktiv (höchste Priorität gewinnt). `SCHLAGER` schlägt immer alle anderen.

**Kein-Profil-Fallback:** Kein PANNs-Score über Schwellwert → `DEFAULT_RESTORATION_PROFILE` (alle Felder auf Standardwerte).

```python
DEFAULT_RESTORATION_PROFILE = {
    # Neutrale Werte — keine genre-spezifischen Einschränkungen
    "groove_dtw_max_ms": 8.0,          # Allgemein-tolerant (§8.2 Standard)
    "tonal_center_threshold": 0.95,    # Standard-PMGG
    "harmonic_exciter_enabled": True,  # Kein Genre-Override
    "dereverb_strength_cap": 0.70,     # Standard — nicht genre-eingeschränkt
    "compression_ratio_cap": 3.0,      # Großzügig
    "soft_saturation_preserve": False, # Kein pauschal geschütztes Sättigungs-Profil
    "gp_memory_key": "default",        # Allgemeiner GP-Speicher
    # Erkennt die 12 nicht-profilierten Genres (Pop, Blues, Soul/R&B, Country,
    # Folk, Funk, Electronic, Hip-Hop, Metal, Latin, Gospel, Reggae) und
    # nutzt deren gp_memory_key für genre-spezifische GP-Konvergenz —
    # aber ohne harte Restaurierungs-Restriktionen.
}
```

**Hinweis zu den 12 Genres ohne eigenes Profil**: Pop, Blues, Soul/R&B, Country, Folk, Funk, Electronic, Hip-Hop, Metal, Latin, Gospel und Reggae werden zwar klassifiziert (17-Genre-System §2.19.2) und erhalten eigene `gp_memory_key`-Einträge (→ GP lernt genre-spezifisch), verwenden aber `DEFAULT_RESTORATION_PROFILE` als Basis. Das ist **absichtlich konservativ**: ohne validierte akustische Constraints riskiert ein zu spezifisches Profil Artefakte. Sobald > 50 GP-Beobachtungen pro Genre vorliegen, können genre-spezifische Profile nachgerüstet werden.

### [RELEASE_MUST] Genre-Profil-Override-Invariante gegenüber CausalDefectReasoner (v9.10.102)

**Problem:** CausalDefectReasoner aktiviert `phase_20`/`phase_49` bei erkanntem `REVERB_EXCESS`.
`KLASSIK_RESTORATION_PROFILE` setzt `phase_20_dereverb_enabled: False` — aber ohne Durchsetzungsregel
überschreibt der Reasoner das Profil und vernichtet den Konzertsaal-RT60.

**Bindende Durchsetzungsregel:**
UV3 MUSS unmittelbar nach `CausalDefectReasoner`-Auswertung, vor der Phasenausführung,
alle `*_enabled: False`-Keys des aktiven Genre-Profils als **harten Override** auf den
Phasenplan anwenden. Genre-Profil hat Priorität über CausalDefectReasoner für diese Phasen.

```python
# Pflicht-Pattern in UV3._execute_pipeline() (nach causal_result, vor Phase-Exec):
if active_genre_profile:
    for key, val in active_genre_profile.items():
        if key.endswith("_enabled") and val is False:
            phase_id = key.replace("_enabled", "")   # z. B. "phase_20_dereverb"
            if phase_id in planned_phases:
                planned_phases.remove(phase_id)
                logger.info(
                    "genre_profile_override: phase=%s disabled by genre=%s",
                    phase_id, active_genre_profile["gp_memory_key"]
                )
```

**Invariante:** `*_enabled: False` im Genre-Profil = absolutes Verbot dieser Phase, unabhängig
vom Defekt-Score. Ausnahme: Wenn ein Defekt-Score > 0.85 (kritisch) vorliegt, darf UV3
eine einmalige Warn-Meldung ins Log schreiben und die Phase trotzdem überspringen — die
Entscheidung des Genre-Profils ist final. Nur ein manueller Studio-2026-Modus darf diesen
Override aufheben (kein automatischer Bypass).

```python
JAZZ_RESTORATION_PROFILE = {
    "groove_dtw_max_ms": 4.0,      # Jazz-Timing heilig
    "tonal_center_threshold": 0.92, # PMGG-intern (phase-level Regression Guard) — KEIN Export-Gate-Override!
                                    # Musikalisch korrekt für Jazz: Blue Notes (♭3/♭5/♭7), Tritond-Substitution,
                                    # modale Harmonik und chromatische Stimmführung lösen K-S-Detektions-
                                    # änderungen aus, die KEINE echten Regressionen sind.
                                    # INVARIANTE: MusicalGoalsChecker erzwingt immer Restoration ≥ 0.95 /
                                    #   Studio 2026 ≥ 0.97 — dieser Wert (0.92) begrenzt NUR den
                                    #   PMGG-Retry-Auslöser während der Phasenausführung.
    "harmonic_exciter_enabled": False,
    "dereverb_strength_cap": 0.30,
    "compression_ratio_cap": 1.8,   # Jazz lebt von Dynamik
    "gp_memory_key": "jazz",
}

KLASSIK_RESTORATION_PROFILE = {
    "phase_20_dereverb_enabled": False,
    "phase_49_dereverb_enabled": False,  # Konzertsaal-RT60 heilig — Genre-Override aktiv (s. o.)
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
    "phase_20_dereverb_enabled": False,  # Opernhall-Raum heilig — Genre-Override aktiv (s. o.)
    "vibrato_rate_tolerance_hz": 0.20,
    "de_esser_voice_adaptive": True,
    "gp_memory_key": "opera",
}
```

---

## §2.27 TransientDecoupledProcessing (TDP)

```python
HPSS_HARMONIC_KERNEL: int = 17    # Frames (Frequenzachse) — v9.10.119, Fitzgerald 2010
HPSS_PERCUSSIVE_KERNEL: int = 13  # Frames (Zeitachse) — v9.10.119, perkussive Schärfe
PERCUSSIVE_ONLY_PHASES: list[str] = [
    "phase_01_click_removal", "phase_27_click_pop_removal",
]
# Rekombination: audio_out = audio_p_processed + audio_h_processed
# via OLA-Crossfade (Hanning 480 Samples = 10 ms @ 48 kHz, Hop = 240 Samples)
# COLA-Invariante: Hop = fsize/2 = 240 Samples (VERBOTEN: Hop > 240 Samples)
# → Hanning-Fenster: w[n]² + w[n+hop]² = 1.0 für alle n — kein Amplitudendip
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
MAX_GAIN_LU_RESTORATION: float = 4.0   # Restoration-Modus: ±4.0 LU (konservativ, Originalcharakter)
MAX_GAIN_LU_STUDIO:      float = 6.0   # Studio-2026-Modus: ±6.0 LU (modern, stärker, frischer)
# Normative Quelle: copilot-instructions §8.3 + Spec 07 §8.3 Zwei-Skalen-Dynamik-Schutz.
# VERBOTEN: einheitliches MAX_GAIN_LU = 3.0 / 2.0 LU — ignoriert die bewusste
#   Modus-Differenzierung (Studio 2026 soll mehr Dynamik-Spielraum als Restoration haben).
FRAME_SIZE_SAMPLES: int = 19200   # 400 ms @ 48000 Hz
HOP_SIZE_SAMPLES: int = 9600      # 200 ms (50 % Überlappung)
PEARSON_TARGET: float = 0.93
MIN_LEVEL_LUFS: float = -60.0     # Stille-Segmente: G[k] = 0

# Position: NACH phase_47_truepeak_limiter, LETZTER Schritt vor Export
# Glättung: Savitzky-Golay(G, window=7, polyorder=2)
# True-Peak-Prüfung nach Morphing: −1.0 dBTP zwingend
```

**[RELEASE_MUST] MDEM tail-gap-Invariante (v9.10.100):**
`gain_envelope` MUSS lückenlos alle Samples bis `n-1` abdecken. Die Standard-Frame-Formel
`(n - fsize) // hop + 1` lässt bis zu `hop - 1 = 9599` Samples (⋜200 ms) am Songschluss
unbehandelt (gain = 1.0). Fix:

```python
# Nach dem gain_envelope-Loop: letzten Gain-Wert bis Signalende fortschreiben
_last_covered = min((n_frames - 1) * hop + fsize, n)
if _last_covered < n and _last_covered > 0:
    gain_envelope[_last_covered:] = gain_envelope[_last_covered - 1]
```

Begründung: Die letzten ~200 ms eines Songs (Fade-out, pp-Ausklang) dürfen kein abruptes
Ende des LUFS-Morphings erfahren. `gain_envelope[…] = 1.0` ist ein impliziter Pegel-Sprung.

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

# ContentAwareProcessor — Salienz-Boosts (§8.3 Tiefen-Immersion, kanonisch):
SALIENCY_BOOST = {
    "fricative_stressed":   1.55,  # §8.3: fricative ×1.55
    "fricative_unstressed": 1.55,  # §8.3: fricative ×1.55
    "vowel_stressed":       1.35,  # §8.3: vowel_stressed ×1.35
    "vowel_unstressed":     1.0,
    "plosive":              1.40,  # §8.3: plosive ×1.40
    "silence":              0.70,  # §8.3: silence ×0.70
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

### §2.36a Phonem-spezifische DSP-Algorithmen (v9.10.90, [RELEASE_MUST])

Einheitlicher SALIENCY_BOOST-Gain reicht nicht — Phonemklassen erfordern
unterschiedliche Spektral-/Zeit-Behandlung.

```python
# ContentAwareProcessor._apply_phoneme_dsp(audio_segment, phoneme_type, sr, strength)
#
# FRIKATIVE ("fricative_stressed", "fricative_unstressed"):
#   Charakteristik: breitbandiges Rauschen 4–8 kHz (turbulente Strömung)
#   Anforderung:    Rauschen-Textur ERHALTEN, kein Smoothing!
#   Algorithmus:    Spektrale Verstärkung mit frequenzabhängigem Gain
#                   g(f) = 1.0 + strength × ramp(f; 4000 Hz, 8000 Hz)  # lineares Ramp-Gain
#                   KEIN: NR-Glättung, kein Wiener-Filter im 4–8 kHz Band
#                   Clip-Schutz: g(f) ≤ 2.5 × base_gain
#
# PLOSIVE ("plosive"):
#   Charakteristik: explosiver Burst (1–5 ms) gefolgt von Stille + Vokal
#                   Frequenzinhalte: 100–350 Hz Burst-Energie + 3–8 kHz Aspiration
#   Anforderung:    Attack-Transient-Shape ERHALTEN, kein Gain-Smoothing am Onset
#   Algorithmus:    TransientShapeGuard (onset_window = 5 ms):
#                   1. Onset-Detektion via Energie-Differenz: Δt < 2 ms → Onset
#                   2. Pre-Onset (−3 ms): kein Gain-Eingriff (Ruhephase)
#                   3. Onset-Fenster (0–5 ms): gain = 1.0 (unveränderlich)
#                   4. Post-Onset (5–40 ms, Burst): gain = strength × 1.40 im 100–350 Hz Band
#                   5. Aspiration (40–150 ms): gain = strength × 1.20 im 3–8 kHz Band
#   KEIN: Kompressionsglättung im Onset-Fenster, kein Fade-in
#
# VOWEL_STRESSED ("vowel_stressed"):
#   Charakteristik: periodisch, F1–F4-Formanten, stimmhaft
#   Anforderung:    Formant-Amplituden proportional anheben (nicht shiften!)
#   Algorithmus:    LPC-Formanten (Burg Ord. 30–40) → peaks F1..F4 identifizieren
#                   Gain: symmetrisches Shelving ±2 Halbton um jeden Formantpeak
#                   g(F_k) = strength × 1.35   (k = 1..4)
#
# SILENCE ("silence"):
#   Anforderung:    Aggressivere NR (OMLSA mit G_floor = 0.05 statt 0.10)
#   Algorithmus:    OMLSA gain_floor = 0.05, DeepFilterNet energy_bias = −12 dB
#                   Ziel: Atemgeräusche/Raumrauschen in Pausen entfernen
#                   KEIN: Stille-Gate (Hard-Muting der Pause zerstört Raumtiefe)
#
# ORCHESTRIERUNG in ContentAwareProcessor.process():
#   1. _build_sample_saliency() → saliency-Karte
#   2. Für jedes WordTimestamp-Segment: _apply_phoneme_dsp(segment, word.phoneme_type)
#   3. SALIENCY_BOOST × Phonem-DSP-Ergebnis (Multiplikativ — NUR wenn boost ≠ 1.0)
#   4. PGHI nach jeder Spektral-Modifikation (PFLICHT)
#
# INVARIANTE: TimbralAuthenticityMetric nach phase_57 ≥ 0.87
#             ArticulationMetric nach phase_57 ≥ 0.85 (Transient-Shape-Korrelation)
```

---

## §11.6 Plugin-Richtlinie (vollständige Liste)

**Pflicht: Erst diese Liste prüfen, DANN neu schreiben.**

```text
# ✅ = lokal gebündelt, kein Download, out-of-the-box

# Vocoder & Synthese
plugins/vocos_plugin.py              → ✅ PRIMÄR (Vocos 48 kHz nativ, Kaskade: 48k→44.1k→24k)
plugins/bigvgan_v2_plugin.py         → ✅ SEKUNDÄR (BigVGAN-v2, 0,4 GB ONNX/PyTorch, Studio-2026, CPU-only)
plugins/hifigan_plugin.py            → ✅ Tertiär-Fallback (3,6 MB ONNX)

# Stem-Separation
plugins/mdx23c_plugin.py              → ✅ MDX23C Kim_Vocal_2/Kim_Inst (2×64 MB) PRIÄR
plugins/demucs_v4_plugin.py          → ✅ HTDemucs 6s (Legacy-Fallback, experimental)
plugins/uvr_mdxnet_plugin.py         → ✅ UVR HQ 1–4 (56–64 MB je)
plugins/bs_roformer_plugin.py        → ✅ BS-RoFormer + Mel-RoFormer (SOTA)

# Rauschunterdrückung & Dereverb
plugins/deepfilternet_v3_ii_plugin.py → ✅ PRIMÄR NR (37 MB: enc+dec+erb_dec)
plugins/sgmse_plugin.py              → ✅ Dereverb/Enhancement PRIMÄR (sgmse_plus.ts, 251 MB) — SGMSE+ 2022
plugins/mp_senet_plugin.py           → ✅ Music/Vocal Enhancement (mp_senet.onnx, 35 MB) — MP-SENet 2023
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
| --- | --- | --- | --- | --- |
| 6 | **ReparaturDenker** | Defekt-Beseitigung | Gezielte DSP-Eingriffe an **bekannten Defekten** (Clicks, Hum, Clipping). Entfernt Störungen, ohne den musikalischen Inhalt zu verändern. | Rekonstruktion, Enhancement, Klangveränderung |
| 7 | **RekonstruktionsDenker** | Rekonstruktion | **Erschafft, was fehlt** — füllt Lücken im Audio-Signal (Dropouts, Silence-Gaps, Tape-Aussetzer). Stellt verloren gegangene Signalanteile wieder her. Erzeugt `ReconstructionContext` mit Hinweisen für UV3. | Klangverbesserung, Defekt-Beseitigung |
| 8 | **RestaurierDenker** | Restaurierung/Erhaltung | **Bewahrt und veredelt, was vorhanden ist** — orchestriert UV3 für die vollständige Restaurierungskette. Schützt den gewollten Klangcharakter (Vintage-Ästhetik, Raumeigenschaften, Dynamik). | Lücken-Füllung, gezielte Defekt-Reparatur |

**Kontextfluss (Pflicht)**:

```text
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
