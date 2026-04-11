---
name: pipeline-debug
description: "Debuggt und versteht die Aurik-9-Pipeline (UV3, Denker, SongCalibration, KMV, FeedbackChain). Use when: UV3, UnifiedRestorerV3, Denker, AurikDenker, SongCalibration, KMV, FeedbackChain, RestaurierDenker, ReparaturDenker, RekonstruktionsDenker, pipeline, _execute_pipeline, restore(), defect_result, PhysicalCeilingEstimator, PerformanceGuard, deferred_phases."
argument-hint: "Was debuggen? (z.B. 'SongCalibration-Profil falsch', 'KMV Stufe 2 startet nicht')"
---

# Aurik 9 — Pipeline debuggen / verstehen

## Kanonischer Pipeline-Ablauf (vollständig)

**PFLICHT-EINSTIEG**: `AurikDenker.denke(audio, sr, mode, progress_callback)`

### Denker-Rollendifferenzierung (§11.7a)

| Stufe | Denker | Domäne | Kurzregel |
|---|---|---|---|
| 6 | `ReparaturDenker` | Defekt-Beseitigung | Entfernt Clicks, Hum, Clipping |
| 7 | `RekonstruktionsDenker` | Rekonstruktion | Füllt Lücken, annotiert BW-Verlust |
| 8 | `RestaurierDenker` | Restaurierung | Orchestriert UV3, schützt Klangcharakter |

**Kontextfluss**: `defect_result → ReparaturDenker → RekonstruktionsDenker(+defect_result) → RestaurierDenker(+reconstruction_context) → UV3`

### §2.41 Denker-Vollkontext (v9.10.117)

- **ReparaturDenker**: 12 Material-Profile (click_iqr, click_kernel_ms, clip_threshold, hum_detect_db). Shellac IQR=4.0 → CD IQR=9.0. Era-adaptive Hum (≤1940: ≥−42 dB).
- **RekonstruktionsDenker**: 6 Material-Konfigurationen für GapReconstructor (Shellac: max 200 ms, Tape: bis 2000 ms).
- **AurikDenker**: Leitet defect_scores, defect_locations, era_decade, material an alle Stufen weiter.

### UV3-Kernreihenfolge

```
DCOffset → TDP (HPSS) → RestorabilityEstimator →
SongCalibrationProfile → EraClassifier ∥ GermanSchlager ∥ MediumDetector →
GoalApplicabilityFilter → AdaptiveGoalThresholds →
DefectScanner (32 Defekte) → CausalDefectReasoner →
GPParameterOptimizer → HarmonicPreservationGuard →
Phasen (01–64) [mit Minimal-Intervention §2.45] →
FeedbackChain → PhysicalCeilingEstimator →
MusicalGoalsChecker → MDEM →
**HolisticPerceptualGate (§2.44)** → RestorationResult
```

**Parallelisierung**: Tier 0+1 sequenziell; Era+Schlager+Medium parallel (ThreadPoolExecutor max_workers=3); Tier 6 sequenziell.

### §2.44 Holistic Perceptual Gate — HPI (v9.10.123)

Letztes Gate vor Export. Fragt: „Klingt der Output **als Ganzes** näher am Original UND ist er **artefaktfrei**?"

**Referenz-Paradoxon**: Das Studio-Original ist unbekannt. Bei Restorability > 70 → Input als Referenz. Bei Restorability ≤ 50 → MERT-Referenz-Vektor aus GP-Memory (genre × material × ära).

**Restoration**: `HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation`

- `timbral_fidelity` dominant: strukturelle Klangkohärenz, nicht bloße Input-Ähnlichkeit
- `artifact_freedom` (§2.49): Musical Noise, Pre-Echo, Spectral Holes = 0
- RestorabilityEstimator > 0.85 → strengeres Gate

**Studio 2026**: `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`

- PQS dominant: Qualität steigern > Original-Treue
- `studio_quality_gain`: Abstand zu Referenz-Studioniveau

### §2.46 Carrier-Chain-Inversion (v9.10.122)

**Restoration**: Nicht Einzel-Defekte reparieren, sondern **Tonträgerkette invertieren**:

1. ADC-Artefakte → 2. Playback-Verzerrungen → 3. Alterung → 4. Carrier-Encoding invertieren
5. Mixer/Preamp-Charakter BEWAHREN | 6. Studio-Raumklang BEWAHREN

**Studio 2026**: Carrier-Chain-Inversion + Enhancement-Kette (§1.5 Spec 02). Mixer/Preamp darf modernisiert werden.

> Der Rauschboden ist modus-differenziert (§0a): Restoration = material-adaptiv; Studio 2026 = ≤ −72 dBFS

Beide: `HPI > 0` → Export | `HPI ≤ 0` → Rollback

### §2.45 Minimal-Intervention-Prinzip (v9.10.122)

**Restoration**: `perceptual_delta ≤ 0` → Phase Skip. So wenige Phasen wie nötig.
**Studio 2026**: Volle Enhancement-Kette aktiv, aber `perceptual_delta > 0` bleibt Pflicht. Auch Enhancement-Phasen müssen Klanggewinn nachweisen. Kein Skip wegen Input-Ähnlichkeit.

### §2.29d Differenziertes Regressions-Regime (v9.10.122)

- P1/P2: **Hart** — keine Phase darf diese verschlechtern
- P3–P5: **Pipeline-Netto-Budget** — Zwischenregressionen erlaubt, MusicalGoalsChecker prüft am Kettenende

### §2.47 Adaptive-Intelligence-Prinzip (v9.10.123)

Adaptions-Kaskade (Reihenfolge = Informationsgewinn):

1. Material-Erkennung → 2. Ära-Klassifikation → 3. Genre-Klassifikation (17 Genres)
4. Restorability-Schätzung → 5. Defekt-Analyse (32 Typen) → 6. Song-Kalibrierung → 7. GPOptimizer

**Edge-Cases**: < 10 s → Groove/MikroDyn/EmotionalArc deaktivieren; > 60 min → Segment-Verarbeitung; Restorability < 20 + Shellac → scale_factor 0.65.
**Prior-Konflikte**: Material-Prior hat Vorrang bei physikalischen Grenzen; Ära-Prior bei ästhetischen Entscheidungen.
**GP-Cross-Material**: < 10 Beobachtungen → Priors vom nächstverwandten Material initialisieren.
**ML-Failure-Kaskade**: DeepFilterNet→OMLSA→Spectral-Gating; MDX23C→NMF→Bypass; MertPlugin→MFCC-Similarity→Bypass. Kein ML-Failure darf die Pipeline abbrechen.

### §2.48 Kumulative-Phasen-Interaktions-Guard (v9.10.123)

Nach jeder Phase: kumulative Drift der P1/P2-Goals messen (Natürlichkeit, Authentizität, TonalCenter, Timbre, Artikulation). `cumulative_drift = goals_now - goals_pre_pipeline`. Drift < −0.05 → Rollback auf `best_checkpoint`.

**Kritische Paare**: De-Hiss + De-Reverb → Raumklang; NR + De-Hiss → Over-Denoising; Multiband-Comp + LUFS → Dynamik-Verlust; Harmonic-Restoration + Vocal-AI → Frequenz-Doppelung.
**Checkpoints**: best_checkpoint + best_artifact_free_checkpoint. Max 2 aufeinanderfolgende Rollbacks → Pipeline-Stop auf best_checkpoint.

### §2.49 Artefakt-Freiheits-Gate (v9.10.123)

Dediziertes Gate — unabhängig von Musical Goals. Prüft nach jeder Phase UND als Export-Gate:

- Musical Noise (isolierte Tonale in Stille, Spectral-Variance) = 0 Events
- Pre-Echo (Energie vor Attack ≤ −40 dB) = 0 Events
- Spectral Holes (Lücken > 200 Hz im Passband) = 0 Holes
- Phase-Cancellation (M/S-Korrelation ≥ 0.3) = ≥ 0.3
- Metallic Ringing (CQT-Peaks > 6 dB über Nachbarn) = 0 Events

`artifact_freedom = 1.0 - (weighted_artifact_count / max_tolerance)`, clipped [0, 1]. Fließt als Multiplikator in HPI (§2.44) ein. `artifact_freedom < 0.95` → Phase-Rollback, kein Export.

## §2.31a Song-Selbstkalibrierung (v9.10.83)

Pflichtprofil `song_calibration_profile`:
`material`, `mode`, `restorability_score`, `input_snr_db`, `max_defect_severity`,
`pipeline_confidence`, `global_scalar`, `family_scalars`

**family_scalars** (mind. 8): `denoise`, `reverb`, `reconstruction`, `dynamics_eq`, `transient`, `vocal`, `instrument`, `general`

### Kalibrier-Berechnungsblöcke (Reihenfolge in `_build_song_calibration_profile`):

1. Era-GP-Warmstart: ≤1940 → ×1.10; ≤1960 → ×1.00; ≥1970 → ×0.88
2. Material-Multiplikatoren (6 Materialien)
3. Per-Defekt-Family-Boost: 28 DefectTypes → 6 Familien, max +12 %
4. Spektral-Fingerprint: rolloff→reconstruction, noise_floor→denoise, wow_flutter→dynamics
5. SOFT_SATURATION-Guard: severity ≥ 0.25 → denoise −12 %, transient −7 %
6. Schlager-Profil: vocal +10 %, transient +5 %, dynamics +5 %, reconstruction ×0.95
7. Diversity-Penalty: ≥8 Defekte → global −1 % je Extra, max −6 %
8. PANNs: vocal_prob/inst_prob → Familien-Skalierung
9. Modus-Post: studio → reconstruction ×1.08, transient/vocal/instrument ×1.05

## §2.42 SourceFidelityReconstructor (v9.10.115–116)

Modell: `backend/core/source_fidelity_reconstructor.py`
Tabellen: `_ERA_BANDWIDTH_HZ`, `_MATERIAL_GENERATION_COUNT`, `_GENERATION_LOSS_DB_PER_GEN` (13 Klassen)
`compute_correction_curve_db()`: Frequenz-abhängig, Cap `_MAX_CORRECTION_DB = 12.0`
UV3-Felder in SongCal: `source_fidelity_bandwidth_target_hz`, `reconstruction_strength`, `confidence`, etc.

## §2.38 KMV — Kontinuierliche ML-Veredelung

| Stufe | Thread | RT-Limit | Ergebnis |
|---|---|---|---|
| 1 | BatchProcessingThread | 32× RT | Sofort-Export (DSP wo RT überschritten) |
| 2 | MLRefinementThread | ∞ | Auto-Overwrite mit voller ML-Qualität |

**Stufe-2-Start**: `deferred_phases > 0` AND `RAM ≥ 4 GB` AND kein anderer Refinement aktiv.

### Quality-First Hauptlauf (v9.10.80)

GUI/CLI/Batch MÜSSEN `denke(..., no_rt_limit=True)` nutzen — keine Qualitätsreduktion im Normalbetrieb.

### DeferredRefinementJob-Felder

`output_path`, `audio_original`, `sr`, `mode`, `deferred_phase_ids`, `cached_defect_result`, `cached_era_result`, `cached_medium_result`, `stufe1_quality`

### RestorationResult-Pflicht-Felder (§2.38)

```python
deferred_phases: list[str]            # Phasen für Stufe 2
refinement_complete: bool = False     # True nach ML-Veredelung
stufe2_quality_estimate: Optional[float] = None
```

## PerformanceGuard — RT-Budget (v9.10.72)

- `LIMIT_BALANCED = LIMIT_QUALITY = LIMIT_MAXIMUM = 32.0`
- `MAX_ABSOLUTE_SECONDS = 5400.0` (90 Min.)
- `LIMIT_BACKGROUND = float("inf")` — nur MLRefinementThread
- Überschreitung → DSP-Fallback + `deferred_phases` eintragen

## FeedbackChain

`target_score`: Base 0.72/0.78 (Restoration/Studio) ±0.035 nach restorability.
Rollback bei |MOS_neu − MOS_alt| > 0.05.

## §2.39 OOM-Recovery-Checkpoint

Modul: `backend/core/recovery_checkpoint.py`

Lifecycle: `_execute_pipeline` MemoryError → `save_checkpoint()` → sessions/STEM_oom_checkpoint.json + _oom_audio.wav → ModernMainWindow Dialog → Resume mit Original-Audio

**Invarianten**: FLOAT WAV, 7 Tage Ablauf, Thread-safe (.tmp → os.replace), Lyrics NICHT im Checkpoint.

## §2.40 Vollpipeline-Determinismus

Gleiche Eingabe + Umgebung + Modus → bitnahe Ausgabe.
Toleranzen: `max_abs_err ≤ 1e-6`, `rms_err ≤ 1e-7`, identische `phases_executed`.

## Dual-SR-Routing (Pflicht)

| Pfad | SR | Module |
|---|---|---|
| `analysis_audio/analysis_sr` | Native Import-SR | DefectScanner, EraClassifier, MediumDetector, RestorabilityEstimator |
| `processing_audio/processing_sr` | 48000 Hz | Alle Phasen (01–64), alle Plugins |

Fail-fast: `processing_sr != 48000` und Resampling unmöglich → strukturierter Abbruch.

## Defect-Locations-Flow (§9.1)

`_execute_pipeline` extrahiert `defect_locations: dict[str, list[tuple[float, float]]]` + `max_defect_severity: float` aus `defect_result.scores` → als kwargs an jede Phase.

**Invarianten**:

- Keine Caps auf defect_locations (vollständige Liste)
- §9.1a: DROPOUTS/TRANSPORT_BUMP auf vollständigem Audio (kein 60s Crop)
- §9.1b: Intro ≤ 5 s → Severity ×1.5
- §9.1c: Maskierte Defekte → `severity * (0.3 + 0.7 * salience)`

## Psychoakustik & Tiefen-Immersion (§8.3)

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Prinzip | Modul |
|---|---|
| Transient-Punch (~40 %) | TDP |
| Mikro-Dynamik (~25 %) | MDEM (400 ms) + EmotionalArcCorrection (5 s) |
| Klarheit (~20 %) | SGMSE+ / OMLSA |
| Vokal-Präsenz (~10 %) | Phase 42/43 + VocalAI |
| Neurale Synthese (~5 %) | Vocos 48k (Studio, MOS < 4.3) |

> Vollständige Pipeline-Spezifikation: `.github/specs/02_pipeline_architecture.md`
> Denker-Details: `denker/aurik_denker.py`
