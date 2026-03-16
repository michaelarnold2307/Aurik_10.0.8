# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: März 2026 — Version **9.10.57**
>
> Aktuelle Testzahl: **7747+** (alle grün)
>
> **§2.36 `LyricsGuidedEnhancement`** ist ab Version **9.10.x Pflicht** (bisher v10.0-Label entfernt).

---

## Vollständige Spezifikation

Die vollständige normative Spezifikation ist in 8 Spec-Dateien aufgeteilt:

| # | Datei | Inhalt |
|---|---|---|
| 1 | `.github/specs/01_musical_goals.md` | 14 Musical Goals, Schwellwerte, PMGG, MDEM |
| 2 | `.github/specs/02_pipeline_architecture.md` | Kanonischer Pipeline-Ablauf, RestorationResult, alle §2.x-Module |
| 3 | `.github/specs/03_cognitive_modules.md` | Kernmodule §2.1, Singleton-Pattern §3.x, Logging, Cache |
| 4 | `.github/specs/04_dsp_standards.md` | SOTA-Entscheidungsmatrix §4.4, DSP-Mindeststandards §4.5 |
| 5 | `.github/specs/05_material_system.md` | Materialien §6.x, DefectTypes §6.3, GP-Gedächtnis |
| 6 | `.github/specs/06_phases_system.md` | Phase 01–56 §7.x, CAUSE_TO_PHASES-Mapping |
| 7 | `.github/specs/07_quality_and_tests.md` | Qualitätsziele §8.x, Test-Standards §5.x, E2E §14 |
| 8 | `.github/specs/08_architecture_and_distribution.md` | Schichten §11.x, Distribution §13.x, Out-of-the-Box |

Änderungshistorie: `docs/CHANGELOG_HISTORY.md`

---

## Projektgrenzen (bindend, keine Ausnahmen)

- **Reine Desktop-App** für Linux (AppImage) und Windows 10/11 (.exe)
- **Kein Cloud, kein Server, kein Docker, kein `pip install`** für Endnutzer
- **Out-of-the-Box-Pflicht**: Läuft auf frischem System ohne Python/Terminal
- **100 % offline** nach Installation — alle ML-Modelle lokal gebündelt
- Nur **Mono und Stereo** unterstützt (> 2 Kanäle → PANNs-gewichteter Downmix)
- **Kein Fremdedit am Original-Audio** — immer neue Ausgabedatei in `output/`

---

## 14 Musical Goals (Pflicht-Schwellwerte)

| Ziel | Klasse | Min. | Studio 2026 |
|---|---|---|---|
| Brillanz | `BrillanzMetric` | ≥ 0.85 | ≥ 0.90 |
| Wärme | `WaermeMetric` | ≥ 0.80 | ≥ 0.80 |
| Natürlichkeit | `NatuerlichkeitMetric` | ≥ 0.90 | ≥ 0.90 |
| Authentizität | `AuthentizitaetMetric` | ≥ 0.88 | ≥ 0.88 |
| Emotionalität | `EmotionalitaetMetric` | ≥ 0.87 | ≥ 0.87 |
| Transparenz | `TransparenzMetric` | ≥ 0.89 | ≥ 0.89 |
| Bass-Kraft | `BassKraftMetric` | ≥ 0.85 | ≥ 0.88 |
| Groove | `GrooveMetric` | ≥ 0.88 | ≥ 0.88 |
| Raumtiefe | `SpatialDepthMetric` | ≥ 0.75 | ≥ 0.75 |
| Timbre-Authentizität | `TimbralAuthenticityMetric` | ≥ 0.87 | ≥ 0.87 |
| Tonales Zentrum | `TonalCenterMetric` | ≥ 0.95 | ≥ 0.97 |
| Mikro-Dynamik | `MicroDynamicsMetric` | ≥ 0.92 | ≥ 0.92 |
| Separation-Treue | `SeparationFidelityMetric` | ≥ 0.82 | ≥ 0.82 |
| Artikulation | `ArticulationMetric` | ≥ 0.85 | ≥ 0.85 |

> **SpatialDepthMetric-Kern**: IACC (Interaural Cross-Correlation, Blauert 1997).
> IACC < 0.70 → wahrnehmbarer Phantom-Center-Zusammenbruch. Zusätzlich:
> Stereobreite + Phantom-Center-Stabilität. Mono-Ären: Goal via GoalApplicabilityFilter deaktiviert.
> **Schwellwert-Status**: Alle 14 Werte sind AMRB-kalibriert („best engineering estimate“);
> Validierung durch ITU-R BS.1534-3 MUSHRA-Hörertest steht aus.

**Sub-Metriken (Pflicht-Implementierungsdetails):**
- `TimbralAuthenticityMetric`: MFCC-Pearson ≥ 0.95, Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 %
- `ArticulationMetric`: Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms
- `TonalCenterMetric`: Chroma-Korrelation ≥ 0.95 **und kein Key-Shift > 0 Cent** (absolut tonarterhaltend)
- `BrillanzMetric` / `WaermeMetric`: Frequenzgewichtung nach **ISO 226:2023 Equal-Loudness** — kein lineares Energiemessen
- `BassKraftMetric`: enthält Virtual Pitch (Missing Fundamental) via Oberton-Analyse 120–500 Hz
- `SeparationFidelityMetric`: SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition

```python
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
checker = MusicalGoalsChecker()
scores = checker.measure_all(audio, sr)  # Dict[str, float]
# Pflicht-Check nach jeder Restaurierung:
assert all(scores[g] >= t for g, t in checker.thresholds.items()), scores
```

**Invariante**: Jede Restaurierungsoperation darf keines der 14 Ziele verschlechtern.

---

## Qualitätsmessung & Metriken-System (§8.1 — PFLICHT)

### PQS-Metriken (`core/perceptual_quality_scorer.py`)

Copilot prüft **alle vier PQS-Metriken** — nie nur MOS allein:

| Metrik | Hard-Fail-Minimum | Weltklasse-Ziel |
|---|---|---|
| PQS MOS | ≥ 3.8 (generell) / ≥ 4.5 (Studio 2026) | ≥ 4.5 |
| PQS NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD (dB) | ≤ 8.0 | ≤ 3.0 |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

### Normative `quality_estimate`-Formel

```python
# Einzige erlaubte Formel — kein freier Bonus-Faktor!
quality_estimate = 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4
quality_estimate = max(0.0, min(1.0, quality_estimate))
# VERBOTEN: quality_estimate * 1.15 als fixer Bonus
# E2E-Pflicht: result.quality_estimate >= 0.55 nach erfolgreicher Restaurierung
```

### OQS-Stufentabelle — Implementierungs-Gate (§8.1.1)

> OQS = `core/mushra_evaluator.py` (algorithmische PEAQ-Approximation — **kein** ITU-R-MUSHRA).
> In externen Berichten stets „OQS (algorithmisch)" schreiben.

| OQS-Stufe | Score | Pflicht |
|---|---|---|
| Excellent (A) | ≥ 91 | — |
| Good (B) | ≥ 80 | **Pflicht für jede neue Phase / Plugin** |
| Fair (C) | ≥ 60 | — |

**Studio-2026-Ziel**: OQS ≥ 88. Eine neue Phase/Plugin darf nur eingecheckt werden, wenn sie OQS ≥ 80 auf mindestens einem AMRB-Szenario erreicht.

### AMRB v1.0 — Aurik Musical Restoration Benchmark (§8.1.2)

| Szenario | Defekt | Pflicht-Score |
|---|---|---|
| AMRB-01-TAPE | Tape-Hiss + Dropout | OQS ≥ 80 |
| AMRB-02-VINYL | Vinyl-Crackle + Rumble | OQS ≥ 80 |
| AMRB-03-SHELLAC | Shellac-Breitrauschen | OQS ≥ 80 |
| AMRB-04-DIGITAL | Clipping + Quantisierung | OQS ≥ 80 |
| AMRB-05-CODEC | Codec-Artefakte | OQS ≥ 80 |
| AMRB-06-VOCAL | Stimmrauschen + Pitch-Drift | OQS ≥ 80 |
| AMRB-07-REVERB | Raumhall RT60 = 1.2 s | OQS ≥ 80 |
| AMRB-08-HUM | 50-Hz-Brumm + Obertöne | OQS ≥ 80 |
| AMRB-09-DROPOUT | Tape-Dropout 50–200 ms | OQS ≥ 80 |
| AMRB-10-COMPOSITE | Kombinierte Degradierung | OQS ≥ 80 |

**Leadership-Schwelle**: Gesamt-Score ≥ **84.0** UND ≥ 8/10 Szenarien bestanden.

```python
from benchmarks.musical_restoration_benchmark import run_benchmark, BenchmarkConfig
report = run_benchmark(config)
assert report.passes_os_leadership_threshold(), f"Score: {report.overall_score}"
```

**Kompetitiver Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 AMRB-Szenarien.

---

## Kanonischer Pipeline-Ablauf (Zusammenfassung)

**PFLICHT-EINSTIEGSPUNKT: `AurikDenker.denke(audio, sr, mode, progress_callback)`**
Kein direktes Aufrufen von `UnifiedRestorerV3.restore()` aus dem Frontend — immer über `AurikDenker`.

### Stufe 0: Kognitive Denker-Orchestrierung (denker/)

```
AurikDenker.denke()
  Stufe 1:  TontraegerDenker    → MaterialType (17 Werte)
  Stufe 1b: TontraegerketteDenker → chain_info (Multi-Gen-Kette, combined_phases)
  Stufe 2:  DefektDenker        → DefectAnalysisResult (29 Defekte, recommended_phases)
            → _defekt_hint = {"recommended_phases": [...], "confidence": float}
  Stufe 3:  StrategieDenker     → StrategiePlan (quality_mode, budget_s, enable_adaptive_skipping)
            → _mode = mode if mode != "quality" else strategie.quality_mode
  Stufe 4:  [_run_rest() Closure]
    → ReparaturDenker.repariere()  → gezielter Phase-Mix aus defekt.recommended_phases
    → RekonstruktionsDenker.rekonstruiere() → Lücken-Erkennung & Reparatur
    → RestaurierDenker.restauriere(mode=_mode, chain_info=chain_info or None,
                                   defekt_hint=_defekt_hint, progress_callback=cb)
  Stufe 7:  ExzellenzDenker.optimiere() → GP-MOO + Musical Goals Re-Pass
  Stufe 8:  VERSA MOS-Gate (< 4.0 → ExzellenzDenker zweiter Durchlauf)
```

### Stufe 1: UV3 interne Pipeline (innerhalb RestaurierDenker)

```
TDP (HPSS-Trennung)  ← NACH DCOffsetPreRemoval (IIR-Hochpass 5 Hz vor jeder FFT)
    scipy.signal.lfilter([1, -1], [1, -0.9999])  — Invariante: np.abs(np.mean(audio)) < 1e-6
    DC-Offset verfälscht STFT Bin 0+1 → alle Spektralanalysen inkorrekt (phase_30 ist KEIN Ersatz!)
→ RestorabilityEstimator (< 5 s)       ← restorability_score für adaptive Schwellen
→ EraClassifier (1890–2025)            ← GP-Warmstart: decade ≤ 1940 → NR ~ N(0.90, 0.05)
→ GermanSchlagerClassifier (Zero-Shot, 6-Schicht)
→ MediumClassifier (17 MaterialType-Werte)
→ GoalApplicabilityFilter              ← inapplicable Goals deaktivieren (§2.32)
→ AdaptiveGoalThresholds               ← Schwellwerte skalieren (§2.31)
→ DefectScanner (29 DefectType-Werte)  ← inkl. WOW/FLUTTER getrennt (IEC 60386) + AZIMUTH_ERROR
→ CausalDefectReasoner (14 Kausal-Ursachen)
→ UncertaintyQuantifier
→ GPParameterOptimizer.propose_pareto() [MOO, 14 Objectives]
    Loss-Funktion (bindend): Mel-Spectral Loss + Multi-Scale STFT Loss (3 Auflösungen)
    VERBOTEN: MSE auf Raw-Audio als alleinige GPO-Loss
→ HarmonicPreservationGuard (G_floor 0.85 an Partial-Bins)
→ UnifiedRestorerV3._select_phases()
  [Tier 1.4: CausalPlan | Tier 1.6: TontraegerketteDenker chain_info | Tier 1.7: DefektDenker defekt_hint]
→ PerceptualEmbedder
→ Phasen-Ausführung (Phase 01–56), jede gegattet durch PMGG (adaptive Regression-Schwelle)
→ EraAuthenticPerceptualCompletion (konditionell, wenn Quell-BW < 10 kHz)
→ IntroducedArtifactDetector
→ FeedbackChain.run()                  ← GoalPriorityProtocol: Priority 1+2 Regression → Abbruch
→ PhysicalCeilingEstimator             ← Terminierung wenn current_score ≥ ceiling − 0.03
→ TemporalQualityCoherenceMetric (≥ 25 s)
→ PerceptualQualityScorer
→ ExcellenceOptimizer
→ MusicalGoalsChecker (14 adaptive Ziele — NICHT statisch!)
→ EmotionalArcPreservationMetric (≥ 30 s)
→ MicroDynamicsEnvelopeMorphing (LETZTER Schritt)
→ GPParameterOptimizer.update()
→ RestorationResult
```

**Parallelisierungs-Invariante (§2.2.1 — bindend):**
- **Tier 0 + Tier 1**: IMMER sequenziell (DCOffset → TDP → Klassifikation)
- **Tier 2–4**: EraClassifier + GermanSchlagerClassifier + MediumClassifier dürfen parallel laufen (`ThreadPoolExecutor max_workers=3`, ONNX gibt GIL frei)
- **Tier 6** (EQ → Polish → LUFS → TruePeak → Format): IMMER sequenziell
- Merge parallelisierter Ergebnisse via `np.mean` NUR bei gleicher Frequenzzone

Details: `.github/specs/02_pipeline_architecture.md`, `denker/aurik_denker.py`

---

## Adaptive Qualitätsziele — Schlecht-Material-Strategie (§2.31–§2.34)

### §2.31 AdaptiveGoalThresholds — Material- und Ära-adaptiv (PFLICHT)

Statische Schwellwerte sind VERBOTEN als alleinige Entscheidungsbasis. Die Schwellwerte werden
**vor jeder Restaurierung** material-, ära- und restorability-adaptiv skaliert:

```python
from backend.core.musical_goals.adaptive_goals_system import get_adaptive_goals_and_config
thresholds, config, quality_assessment = get_adaptive_goals_and_config(audio, sr)
# → thresholds enthält skalierte Werte, kein Ziel darf manuell überschrieben werden

# Restorability-Skalierungs-Tabelle (normativ — formal aus PhysicalCeilingEstimator abgeleitet):
# scale_factor = ceiling_avg(goals) / baseline_threshold, gemessen auf 500 AMRB-Testdateien:
SCALE_FACTORS = {
    "≥ 70":   1.00,   # GOOD — ceiling_avg = 0.97
    "50–69":  0.93,   # FAIR — ceiling_avg = 0.90
    "30–49":  0.85,   # POOR — ceiling_avg = 0.82
    "< 30":   0.75,   # VERY_POOR — ceiling_avg = 0.73
    # VERBOTEN: Stufenwerte manuell setzen ohne PhysicalCeilingEstimator-Grundlage
}

# Material-spezifische Anpassungen (physikalische Grenzen):
# SHELLAC/WAX_CYLINDER: brillanz_threshold → min(0.85, bw_hz/20000*0.85+0.20)
#                        spatial_depth → 0.30 (Mono-Aufnahme)
# VINYL:                 separation_fidelity_threshold → 0.76
# DAT/CD_DIGITAL:        alle Schwellwerte unverändert

# Ära-Prior (EraClassifier.decade):
# decade ≤ 1940: spatial_depth_threshold → 0.30
# decade ≤ 1960: spatial_depth_threshold → 0.55
# decade ≥ 1970: alle Spatial-Thresholds Standard

# Absolute Untergrenze: adaptive_t ≥ 0.50 (darunter → Goal deaktivieren via §2.32)
# Physical Ceiling Clamp: adaptive_t = min(adaptive_t, physical_ceiling[goal])
```

```python
# MaterialQuality Enum (backend/core/musical_goals/adaptive_goals_system.py):
class MaterialQuality(Enum):
    PRISTINE   = "pristine"    # Studio-Qualität
    EXCELLENT  = "excellent"
    GOOD       = "good"
    FAIR       = "fair"        # MP3 192 kbps / Kassette
    POOR       = "poor"        # MP3 128 kbps / stark degradiert
    VERY_POOR  = "very_poor"   # Schellack mit starkem Rauschen
    EXTREME    = "extreme"     # Wachswalze / Telefon-Aufnahme
```

### §2.32 GoalApplicabilityFilter — Physikalisch unmögliche Ziele deaktivieren (PFLICHT)

```python
# Filter läuft EINMAL nach MediumClassifier + EraClassifier (vor Phase-Ausführung):
ALWAYS_APPLICABLE = frozenset({
    "natuerlichkeit", "authentizitaet", "emotionalitaet",
    "transparenz", "timbre_authentizitaet", "artikulation",
})

# Deaktivierungs-Regeln — inapplicable Goals → Grau im UI, KEIN Fehler:
# SpatialDepthMetric:     decade ≤ 1950 UND M/S-Korrelation ≥ 0.95 (Mono)
# BrillanzMetric:         Quell-BW < 8 kHz UND AudioSR nicht geladen
# TonalCenterMetric:      Original-SNR < −5 dB ODER MaterialType = WAX_CYLINDER
# GrooveMetric:           Dateilänge < 10 s ODER PANNs Percussion confidence < 0.15
# MicroDynamicsMetric:    Dateilänge < 20 s ODER Original-LUFS-Varianz < 0.5 LU
# SeparationFidelityMetric: Mono-Quelle ODER PANNs < 2 Instrumente confidence ≥ 0.4

# Ergebnis in RestorationResult.goal_applicability gespeichert
```

### §2.33 PhysicalCeilingEstimator — Informationstheoretische Qualitätsdecke (PFLICHT)

```python
HEADROOM_THRESHOLD: float = 0.03   # Δ < 3 % → keine weiteren Iterationen sinnvoll

# Ceiling-Formeln (empirisch kalibriert):
natuerlichkeit_ceiling  = sigmoid((mean_snr_db − 5) / 5) × 0.97 + 0.03
brillanz_ceiling        = sigmoid((bw_hz − 8000) / 2000) × 0.95
spatial_depth_ceiling   = sigmoid(stereo_decorrelation × 10) × 0.92
groove_ceiling          = 1 − max(0, max(wow_hz, flutter_hz) − 0.5) × 0.10
# wow_hz = pYIN-Varianz-RMS über 500 ms (für WOW, IEC 60386 < 0.5 Hz)
# flutter_hz = pYIN-Varianz-RMS über 50 ms (für FLUTTER, IEC 60386 0.5–200 Hz)
tonal_center_ceiling    = sigmoid(snr_tonal_bands × 2) × 0.98
# Alle anderen Goals: 0.98 (konservative Obergrenze)

# FeedbackChain-Terminierung (bindend):
# → "Das Beste aus dieser Aufnahme wurde herausgeholt — die physikalischen Grenzen
#    des Quellmaterials sind erreicht." (Deutsch im UI)
```

**Kritisch für schlechtes Material:** Bei Shellac (SNR ≈ 5 dB, BW ≈ 7 kHz) ist
`brillanz_ceiling ≈ 0.55` — dieser Wert muss die adaptiven Schwellen deckeln, sonst
iteriert ExcellenceOptimizer sinnlos bis zum Zeitlimit.

### §2.34 GoalPriorityProtocol — Hierarchie bei Konflikten (PFLICHT)

```python
PRIORITY_MAP = {
    "natuerlichkeit":        1,   # Rollback bei JEDER Verschlechterung
    "authentizitaet":        1,   # Rollback bei JEDER Verschlechterung
    "tonal_center":          2,   # Rollback bei Verschlechterung
    "timbre_authentizitaet": 2,
    "artikulation":          2,
    "emotionalitaet":        3,
    "micro_dynamics":        3,
    "groove":                3,
    "transparenz":           4,
    "waerme":                4,
    "bass_kraft":            4,
    "separation_fidelity":   4,
    "brillanz":              5,   # best-effort, kein Misserfolg
    "spatial_depth":         5,   # best-effort
}
ABORT_PRIORITY_THRESHOLD: int = 2  # Priority 1+2 verschlechtert → Iteration abbrechen

# In FeedbackChain.run():
gpp = GoalPriorityProtocol()
if gpp.should_abort_iteration(scores_before, scores_after).should_abort:
    best_result = previous_best; break

# In ExcellenceOptimizer (MOO-Pareto-Konflikt):
conflict_result = gpp.resolve_conflict(goal_a, goal_b, delta_a, delta_b)
# conflict_result.winner = priorisiertes Ziel
```

---

## Algorithmische Pflicht-Mindeststandards

| Anwendungsfall | Primär (Plugin) | Fallback 1 | DSP-Fallback | Verboten |
|---|---|---|---|---|
| Breitrauschen | DeepFilterNet v3.II ONNX (`deepfilternet_v3_ii_plugin`, 3 Sessions) | — | OMLSA/IMCRA | ~~Wiener 1984~~ |
| Stem-Separation Gesang | MelBandRoformer ONNX (`bs_roformer_plugin`, 860 MB) | HTDemucs ONNX (`htdemucs_plugin`) → MDX23C Kim_Vocal_2 ONNX | HPSS + NMF-β | — |
| Stem-Separation Instrumental | HTDemucs ONNX (`htdemucs_plugin`) | MDX23C Kim_Inst ONNX (`mdx23c_plugin`) → UVR MDX-Net Ensemble (`uvr_mdxnet_plugin`, 4 ONNX) | HPSS DSP | — |
| Dropout (< 50 ms) | NMF-β + Sinusoidal | Consistent Wiener | Consistent Wiener | ~~Yule-Walker~~ |
| Dropout (≥ 50 ms) | CQTdiff+ ONNX (`cqtdiff_plus_plugin`) | DiffWave | Spectral Interpolation | ~~AR ohne Konsistenz~~ |
| Generatives Inpainting | Flow Matching (`flow_matching_plugin`) | CQTdiff+ → DiffWave → NMF-β | NMF-β + Sinusoidal | ~~DDPM 1000 Schritte~~ |
| Bandbreiten-Erweiterung | AudioSR ONNX (`audiosr_plugin`, 5,9 GB, lazy) | Sinusoidal + Stochastic Modeling | EraAuthenticPerceptualCompletion DSP | ~~Harmonics-EQ~~ |
| Codec-Artefakte | Apollo TorchScript (`apollo_plugin`) | Resemble-Enhance ONNX (`resemble_enhance_plugin`) | Spectral Repair DSP | ~~EQ-Anhebung~~ |
| Reference Mastering (Studio 2026) | Matchering 2.0.6 (`matchering_plugin`) | Mid/Side STFT-Matching DSP | — | — |
| Pitch-Tracking | RMVPE ONNX (`rmvpe_plugin`) | CREPE ONNX (`crepe_plugin`) → FCPE ONNX (`fcpe_plugin`) | PESTO → pYIN | ~~YIN~~ |
> **PESTO** (Riou et al. ISMIR 2023): Chromagramm-basiertes Pitch-Tracking, 40× schneller als pYIN bei vergleichbarer Genauigkeit. Als letzter DSP-Fallback einsetzbar (`dsp/pesto_pitch.py`). VERBOTEN als Primär-Tracker (RMVPE-Genauigkeit überlegen).
| **Polyphoner Pitch** | BasicPitch ONNX (`basicpitch_plugin`) | Spektrale Peak-Verfolgung | Spektrale Peak-Verfolgung | ~~CREPE mono für Polyphonie~~ |
| Phasen-Rekonstruktion | PGHI | Griffin-Lim+ ≥ 32 It. | Griffin-Lim+ ≥ 32 It. | ~~ISTFT direkt~~ |
| Vocoder (MOS < 4.3) | Vocos 44.1 kHz ONNX (`vocos_plugin`) | BigVGAN v2 (`bigvgan_v2_plugin`) | HiFi-GAN | ~~Griffin-Lim~~ |
| Audio-Klassifikation / Tagging | BEATs iter3 ONNX (`beats_plugin`) | PANNs ONNX (`panns_plugin`) | Spectral Features | — |
| Vocal Enhancement / Dereverb | SGMSE+ ONNX (`sgmse_plugin`) | Resemble-Enhance ONNX (`resemble_enhance_plugin`) | WPE DSP | — |
| MOS-Bewertung Musik | VERSA ONNX (`versa_plugin`) → SingMOS (Gesang, PANNs Vocals ≥ 0.5) → PQS-Gammatone | PEAQ | PEAQ | ~~PESQ/DNSMOS/NISQA/CDPAM~~ |
| **MOS mit Referenz** | ViSQOL v3 (`--audio` PFLICHT) | PQS-DSP | PQS-DSP | ~~ViSQOL --speech~~ |
| Formant-Tracking | **DeepFormants CNN** ONNX (`deepformants_plugin`, ML-Primär) + `SingersFormantEnhancer` (2,5–3,5 kHz) | `FormantSystem` (LPC Ord. **30–40 bei 48 kHz-SR**, F1–F5; alternativ: Downsampling 16 kHz → Ord. 16 → Upsampling) | Bell-EQ @ 1,5 kHz | ~~LPC < 16~~ / ~~pYIN als Formant-Schätzer~~ |
| De-Essing / Sibilanten | Gender-adaptiv + `_estimate_breathiness()` Guard | Split-Band-Envelope | Split-Band-Envelope | ~~fester `strength_cap` ohne Breathiness-Check~~ |
**ABSOLUT VERBOTEN** als Musikmetrik: `PESQ`, `DNSMOS`, `NISQA`, `STOI`, `ViSQOL --speech`, `CDPAM` (ersetzt durch VERSA §4.4)
**ABSOLUT VERBOTEN** als Enhancement-Plugin: ~~`dccrn_plugin`~~, ~~`fullsubnet_plus_plugin`~~ (ersetzt durch `mp_senet_plugin` §4.4)

Vollständige Matrix: `.github/specs/04_dsp_standards.md`

---

## Schlecht-Material-Verarbeitungsregeln (§2.29 + §6.2 + §2.14 + §7.6)

### §2.29 PerPhaseMusicalGoalsGate — Adaptive Regression-Schwellen

```python
# Schwellwerte restorability-adaptiv (NICHT statisch!):
REGRESSION_THRESHOLD_GOOD: float = 0.012   # restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = 0.040   # restorability 40–69
REGRESSION_THRESHOLD_POOR: float = 0.060   # restorability < 40 — größere Toleranz
MAX_RETRIES: int = 5
_RETRY_STRENGTHS: list[float] = [0.65, 0.50, 0.35, 0.20, 0.10]

# Datenfluss-Invariante:
re_result = RestorabilityEstimator().estimate(audio, sr, defect_analysis)
gate = PerPhaseMusicalGoalsGate()
for phase in selected_phases:
    audio, scores, _ = gate.wrap_phase(
        phase, audio, sr, scores_before,
        restorability_score=re_result.restorability_score,
        applicable_goals=goal_filter.applicable,  # nur anwendbare Goals prüfen
    )
```

### §2.14 EraClassifier — GP-Warmstart (Material-spezifisch)

```python
# GP-Optimizer Warmstart VOR dem ersten propose_pareto()-Aufruf (bindend):
# decade ≤ 1940: noise_reduction_strength ~ N(0.90, 0.05)  → Shellac/Wachswalze
# decade ≤ 1960: noise_reduction_strength ~ N(0.75, 0.08)  → frühe Kassette
# decade ≥ 1970: noise_reduction_strength ~ N(0.50, 0.10)  → Standard
# is_remaster_suspected=True: noise_reduction_strength ~ N(0.35, 0.10)
```

### §6.2 Material-spezifische PQS-Erwartungen (bindend — KEIN globales MOS ≥ 4.5)

| Material | MOS-Ziel | Typische Defekte |
|---|---|---|
| `wax_cylinder` | ≥ 3.5 | Extremrauschen, BW ≤ 5 kHz |
| `shellac` | ≥ 3.8 | Breitrauschen, BW ≤ 8 kHz |
| `lacquer_disc` | ≥ 3.7 | Riss-Klicken, Substrat-Rauschen |
| `wire_recording` | ≥ 3.6 | Jitter, Frequenz-Dropout |
| `vinyl` | ≥ 4.0 | Crackle, Warp, Rillenverzerrung |
| `tape` | ≥ 4.2 | Dropout, Hiss, Wow/Flutter |
| `reel_tape` | ≥ 4.3 | Print-Through, Hiss, Dropout |
| `mp3_low` | ≥ 3.9 | Schwere Codec-Artefakte |
| `cd_digital` | ≥ 4.5 | Clipping, Quantisierung |

**Studio-2026-MOS ≥ 4.5 gilt NUR für `cd_digital`, `dat`, `mp3_high`, `aac`.**
Für alles darunter: material-adaptive Erwartungswerte aus §6.2.

### §7.6 Adaptive Chunk-Verarbeitung (bei Dateien ≥ 5 Minuten)

```python
def adaptive_chunk_size(defect_severity: float, segment_type: str) -> float:
    if segment_type == "silence":   return 120.0
    if defect_severity >= 0.6:      return 5.0   # hohes Defektniveau → Feingranular
    if defect_severity >= 0.3:      return 15.0
    return 60.0   # sauberes Material → große Chunks für Kontext-Kohärenz
# Minimum: 2 s | Maximum: 120 s
```

---

## Vintage Aesthetics — Pflicht-Invarianten bei historischem Material

Diese Regeln verhindern, dass restaurierte Aufnahmen "zu modern" klingen:

```python
# Aktiviert durch EraClassifier.decade — bindend in UV3:
# 1920–1940 (SHELLAC/WAX_CYLINDER):
#   → Rolloff ≤ 7 kHz NICHT künstlich erweitern (nur EraAuthenticPerceptualCompletion DSP)
#   → AudioSR nur wenn user_requested=True UND restorability ≥ 40
#   → HF-Röhren-Kompression H2, H4 ∈ [−30, −20] dBr BEWAHREN (→ SOFT_SATURATION)
# 1940–1955 (TAPE/REEL_TAPE):
#   → Tape-Saturation-Fingerabdruck NICHT entfernen
#   → phase_22 (tape_saturation) nur emulieren, nie eliminieren
# 1955–1965:
#   → RT60 ∈ [1.2, 2.0] s bewahren — phase_20/phase_49 strength ≤ 0.20
# 1965–1975 (TAPE/REEL_TAPE späte Ära):
#   → Tape-Saturation-Signatur NICHT entfernen (phase_22 nur emulieren, nie eliminieren)
#   → Vintage-Kompressor-Imprint (VCA-Charakter) bewahren
# AuthentizitaetMetric nach Restaurierung ≥ Wert vor Restaurierung (Pflicht-Invariante)
```

## CLIPPING vs. SOFT_SATURATION — Kritische Unterscheidung (§6.3)

**SOFT_SATURATION = Röhren-/Tape-Charakter: BEWAHREN. CLIPPING = Amplitudenbeschädigung: REPARIEREN.**

```python
def classify_clipping(audio: np.ndarray, sr: int) -> ClippingType:
    """Diskriminiert CLIPPING von SOFT_SATURATION per Oberton-Analyse.

    CLIPPING:        flat_tops > 0.1 % UND THD_odd > THD_even × 1.5
    SOFT_SATURATION: flat_tops < 0.1 % ODER THD_even > THD_odd
    SOFT_SATURATION → Pipeline überspringt Clipping-Reparatur komplett!
    """

# Praktische Konsequenz:
# FALSCH: AllClipping-Detektion → phase_23 immer aktivieren
# RICHTIG: classify_clipping() → nur bei CLIPPING → phase_23; bei SOFT_SATURATION → Skip
```

## Vocal-Restaurierungskette — Pflicht-Reihenfolge (§2.8)

```
1. GenderDetector.detect() → VoiceCharacteristics (F₀, Formanten, Breathiness)
2. SGMSE+ (Dereverb/Denoising) VOR VocalAIEnhancement
3. RMVPE → CREPE → FCPE → pYIN (Pitch-Tracking Kaskade)
4. FormantTracker (LPC Ord. **30–40 bei 48 kHz-SR**, F1–F5) — Invariante: Pearson(F1_before, F1_after) ≥ 0.95
   (Faustregel: Ord. ≈ SR[kHz] × 2 + 4 = 100; Kompromiss 30–40 ausreichend für F1–F5.
    Alternativ: Downsampling auf 16 kHz vor LPC-Analyse → Ord. 16 korrekt → Upsampling nach Analyse.)
5. BreathDetector → breathiness ratio (Erhalt ±0.05 — KEINEN Atem entfernen!)
6. De-Esser (phase_19) + ML-De-Esser (phase_43) stimmtyp-adaptiv
7. VocalAIEnhancement.enhance()
8. SingersFormantEnhancer (2.5–3.5 kHz) für Sänger-Formant-Zone
9. PSOLA: Pflicht bei Gesang (PANNs Vocals ≥ 0.4) und Pitch-Korrektur > ±2 Halbton
   → Formant-erhaltend; Phase-Vocoder NUR für perkussive Segmente (HPSS-detektiert)
10. Emotionalität: emotion_preservation_score ≥ 0.87 (→ EmotionalArcPreservationMetric)
```

**API-Falle (häufiger Fehler):**
```python
# FALSCH — BreathIntelligence erwartet KEIN events-Argument:
events = self.breath_detector.detect(audio, sr)
self.breath_intelligence.process(audio, sr, events)   # TypeError!
# RICHTIG — BreathIntelligence erkennt intern selbst:
enhanced, report = self.breath_intelligence.process(audio, sr)
```



### Singleton (Thread-sicher, Double-Checked Locking — bindend)

```python
import threading
from typing import Optional

_instance: Optional[MyModule] = None
_lock = threading.Lock()

def get_my_module() -> "MyModule":
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyModule()
    return _instance
```

### Numerische Robustheit (PFLICHT nach jeder numerischen Operation)

```python
import numpy as np, math

result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
audio = np.clip(audio, -1.0, 1.0)          # am Ausgang jeder Phase
if not math.isfinite(score): return         # vor GP-Update
assert sample_rate == 48000                 # am Eingang jeder Phase/Plugin
```

### Logging-Konvention

```python
import logging
logger = logging.getLogger(__name__)
# Nutzer-Texte/UI: Deutsch | Code-Kommentare/Docstrings: Englisch | Logs: Englisch
logger.info("🧠 CausalReasoner: cause=%s confidence=%.2f", cause, conf)
# KEIN print() in Produktionscode
```

---

## Verbotene Praktiken

```python
# Code-Qualität:
print("Score:", score)          # → logger.info(...)
return {"mos": 3.5}             # öffentliche API → @dataclass zurückgeben
_cache = {}                     # ohne threading.Lock() bei multithreading

# Algorithmen — Legacy-Verbote (spec 04 §4.2):
np.fft.rfft(...); istft(...)           # ohne PGHI nach Spektral-Modifikation → PGHI Pflicht
rms = np.sqrt(np.mean(audio**2))      # RMS-Normalisierung → LUFS ITU-R BS.1770-5
np.max(np.abs(audio))                  # Peak-Normalisierung bei Restaurierung → LUFS + TruePeak
lpc = librosa.lpc(audio, order=12)    # LPC Ordnung < 16 → Ord. 30–40 bei 48 kHz-SR

import ddsp                           # pip install ddsp → erfordert TensorFlow — VERBOTEN
# RICHTIG — Aurik nutzt NumPy/SciPy-Eigenimplementierung:
from dsp.ddsp_synth import DDSPSynth  # kein TF, out-of-the-box lauffähig

# Architektur:
import Aurik910                 # in backend/core/plugins/dsp VERBOTEN (keine UI-Imports im Backend)
from Aurik910.i18n import t     # VERBOTEN in backend/core — nur in Aurik910/ erlaubt
torch.load(..., map_location="cuda")   # CPU-only, kein GPU

# Metriken:
pesq(...)                       # Telefonband 300–3400 Hz, kein Musik
dnsmos(...)                     # 16 kHz Speech-Corpus, kein Musik
nisqa(...)                      # kein Musik-Training

# Algorithmen:
# scipy.signal.wiener() als primäre NR → OMLSA/DeepFilterNet
# griffinlim() als Endschritt Studio-2026 → Vocos/HiFi-GAN

# Vocal-Pipeline (häufige API-Falle):
# FALSCH — breath_events ist list[BreathEvent], kein ndarray:
events = self.breath_detector.detect(audio, sr)
self.breath_intelligence.process(audio, sr, events)   # TypeError!
# RICHTIG — BreathIntelligence erkennt intern selbst, Rückgabe ist Tuple:
enhanced, report = self.breath_intelligence.process(audio, sr)
```

---

## Checkliste neues Kernmodul (Pflicht)

```
□ backend/core/<modul>.py — Singleton + Convenience-Funktion
□ threading.Lock() + Double-Checked Locking
□ Alle public APIs: vollständige PEP 484 Type-Annotations
□ Docstrings mit Algorithmus-Beschreibung + math. Formeln
□ NaN/Inf-Guard in JEDER numerischen Ausgabefunktion
□ Ergebnisse als @dataclass (kein raw dict)
□ assert sample_rate == 48000 am Eingang
□ ≥ 35 Unit-Tests: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo
□ Musical Goals: kein Ziel nach dem Modul schlechter als vorher
□ GrooveMetric: DTW ≤ 8 ms RMS (kein Timing-Flattening)
□ SOFT_SATURATION: wird nicht als CLIPPING detektiert
□ DSP-Fallback für jeden optionalen ML-Import (try/except ImportError)
□ models/manifest.json: sha256 + bundled_path + fallback bei neuen Modellen
□ OQS ≥ 80 auf mindestens einem AMRB-Szenario nachweisbar
□ quality_estimate ≥ 0.55 im E2E-Test (result.quality_estimate-Assertion)
□ goal_applicability in RestorationResult gespeichert (GoalApplicabilityFilter-Ergebnis)
□ CHANGELOG.md Eintrag
□ Alle bestehenden Tests weiterhin grün (aktuell 7747+)
```

---

## Anti-Parallelwelten-Workflow (Pflicht vor jeder Implementierung)

```
1. Suche in backend/core/, plugins/, dsp/ nach vorhandener Funktionalität
2. Prüfe existierende Plugins (specs/08) und Phasen (specs/06)
3. Falls vorhanden → einbinden + DSP-Fallback, KEIN neues Modul
4. Falls nicht → neues Modul nach Singleton-Pattern anlegen
5. Entscheidung im CHANGELOG.md dokumentieren
```

---

## Performance-Budget (Desktop, kein GPU)

| Operation | Limit / Minute Audio |
|---|---|
| DefectScanner | ≤ 2 s |
| Phase-Pipeline gesamt | ≤ 120 s |
| FeedbackChain (alle Iter.) | ≤ 60 s |
| ExcellenceOptimizer | ≤ 30 s |
| RestorabilityEstimator | ≤ 5 s |

- Interne Verarbeitungs-SR: stets **48 000 Hz**
- Alle ONNX-Sessions: `providers=["CPUExecutionProvider"]`
- Torch-Modelle: `model.to("cpu")`; `torch.set_num_threads(os.cpu_count())`
- MERT (3,9 GB) / AudioSR (5,9 GB): nur Lazy-Load bei Bedarf

---

## DSP-Spezialregeln (Pflicht-Implementierungsdetails)

### Multi-Resolution STFT — MRSA-Zonen (Phase 03, 06, 07, 23, 50)

```python
# MRSA-Fenster @ SR=48000 Hz — alle fünf Zonen zwingend:
ZONES = {
    "sub_bass":   {"win": 65536, "hop": 16384, "hz": (20,   250)},
    "mid_low":    {"win": 16384, "hop":  4096,  "hz": (250,  800)},
    "mid":        {"win":  8192, "hop":  2048,  "hz": (800, 2000)},
    "presence":   {"win":  1024, "hop":   256,  "hz": (2000, 8000)},
    "air":        {"win":   128, "hop":    32,   "hz": (8000, 24000)},
}
# PGHI per Zone; Kreuzfade Hanning 10 ms an Zonenübergängen
# VERBOTEN: willkürliche FFT-Größen ohne Zonen-Mapping
```

### Dithering beim Export (24→16 Bit)

```python
# PRIMÄR: POW-r Typ 3 (Wannamaker et al. 1992) — ~+6 dB effektiver SNR
# FALLBACK: TPDF-Dithering (±1 LSB)
# ABSOLUT VERBOTEN: Truncation ohne Dithering
```

### Print-Through-Reduktion (Phase 29, reel_tape)

```python
# Pflicht: Bidirektionale Adaptive Temporal Subtraction (LMS)
# Print-Through entsteht auf BEIDEN Wicklungsseiten mit unterschiedlicher Amplitude:
#   Pre-Echo (Vorwärtswicklung): schwächer  → alpha_pre  ∈ [0.03, 0.25]
#   Post-Echo (Rückwärtswicklung): stärker  → alpha_post ∈ [0.05, 0.35]
#
# Algorithmus:
#   1. Kreuzkorrelation-Peak ±600 ms → delay_pre, delay_post (beide Seiten)
#   2. LMS-Adaptivfilter separat für Pre- und Post-Echo
#   3. audio_clean[t] = audio[t] − alpha_pre · audio[t + delay_pre]
#                                 − alpha_post · audio[t − delay_post]
#   4. Spectral Coherence vor/nach ≥ 0.90 + PGHI
#
# VERBOTEN: Comb-Filter, einseitiges α-Modell (alpha_pre == alpha_post)
# Fallback: NMF-β Dekomposition (einseitig, nur Post-Echo)
```

### Perceptuelle Verpflichtungen — Pflicht-Messwerte (§8.3)

| Metrik | Schwellwert | Messung |
|---|---|---|
| MERT-Naturalness-Score | ≥ 0.7 | `MertPlugin.analyze().harmonicity` (Proxy, kalibriert Pearson=0.74 ↔ VERSA-MOS; VERSA hat Vorrang) |
| Harmonizitäts-Ratio | ≥ 0.85 | `MertPlugin.analyze().harmonicity` |
| LUFS-Differenz | ≤ 1 LU | Original ↔ Restauriert |
| Transientenerhalt | Attack ≤ ±2 ms | Transient-Shape-Verfahren |
| Chroma-Stabilität | Pearson ≥ 0.95 | Chroma-Korrelation Original ↔ Restauriert |
| Groove | DTW ≤ 8 ms RMS | Event-Onset-DTW (madmom) |

> Kein Begradigen von Swing/Rubato — GrooveMetric vor/nach der Verarbeitung stabil halten.

| Spec-Pfad | Dateisystem-Pfad |
|---|---|
| `core/<modul>.py` | `backend/core/<modul>.py` |
| `plugins/<plugin>.py` | `plugins/<plugin>.py` |
| Frontend / UI | `Aurik910/` (Haupt-UI-Paket) |
| Frontend-Einstiegspunkt | `Aurik910/main.py` (`ModernMainWindow`) |
| i18n / Übersetzungen | `Aurik910/i18n/__init__.py` |
| UI-Widgets | `Aurik910/ui/` |
| Core-UI-Logik | `Aurik910/core/` |
| Ressourcen (Icons, QSS) | `Aurik910/resources/` |
| Keine Shim-Dateien | die `core/` ↔ `backend/core/` emulieren |

**Wichtig**: Es gibt kein `frontend/`-Verzeichnis. Alle UI/Frontend-Module liegen
unter `Aurik910/`. Import-Pfad für i18n: `from Aurik910.i18n import t, set_language`

---

## Restaurierungs-Modi

| Modus | Ziel | LUFS | TonalCenter |
|---|---|---|---|
| **Restoration** | Originalgetreue Restauration | Δ ≤ 1 LU | ≥ 0.95 |
| **Studio 2026** | Highend-Studio-Klang | −14 LUFS EBU R128 | ≥ 0.97 |

Studio 2026 Verarbeitungskette (Kurzform):
Stem-Sep → Vocal-AI → Instrumente → [Reference Mastering] → Multibandkomp →
Präsenz/Air → Stereo-Imaging → Re-Mix (StemRemixBalancer) →
LUFS-Norm → TruePeak → Musical Goals → [Vocos-Synthese (MOS < 4.3)]

### StemRemixBalancer — Pflicht-Algorithmus nach Stem-Verarbeitung

**Verboten**: nacktes `vocals + instruments` in `UnifiedRestorerV3`. Immer via `StemRemixBalancer.balance_remix()`.

```python
# 6-Schritte-Algorithmus (spec 02 §1.5, kanonisch):
# 1. L_orig gesamt messen VOR Separation
# 2. vocal_weight via PANNs auf Original (max. 10-s-Excerpt) — MUSS vor MDX23C feststehen
# 3. Nach Verarbeitung: LUFS pro Stem messen (L_voc', L_inst')
# 4. Gain-Korrektur:
#    g_voc  = 10 ** ((L_orig_voc  − L_voc')  / 20)
#    g_inst = 10 ** ((L_orig_inst − L_inst') / 20)
# 5. Re-Mix: mix = g_voc · vocals + g_inst · instruments
# 6. Final-Check: |LUFS(mix) − L_orig| ≤ 0.3 LU

# Invarianten:
# - Vocals/Instruments-Verhältnis: ΔdB ≤ ±0.3 dB vs. Original
# - Kein Clipping im Re-Mix (np.clip nach Summation)
# - TonalCenterMetric nach Re-Mix ≥ 98 % des Pre-Remix-Werts
# - Laufzeit: ≤ 0.5 s / Minute Audio
```

**Pflicht-Test**: `tests/unit/test_stem_remix_balancer.py` (≥ 20 Tests).

---

## Universelle Garantien (§8.2 — PFLICHT)

| Garantie | Messung / Schwellwert |
|---|---|
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) ≤ 1.0` |
| Chroma-Korrelation (Tonart) | Pearson ≥ 0.95 |
| Pass-Through (sauberes Material, SNR > 40 dB) | PQS-MOS-Verlust ≤ 0.05, Goals stabil ±0.02, LUFS ≤ 0.3 LU, Chroma ≥ 0.99 |
| Rauschboden (Studio-2026) | ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events in Stille |
| Temporale Kohärenz | MOS-Spanne über 10-s-Segmente ≤ 0.30, σ ≤ 0.15 |
| **Stereo-Authentizität** | Mono-Ären: M/S-Korrelation nach Restaurierung ≥ 0.97 |
| **HF-Kumulativ-Limit** | Presence + Air kumulativ ≤ +4 dB (Listening-Fatigue-Schutz) |
| Mikro-Dynamik-Erhalt | Pearson LUFS-Profil (400 ms) ≥ 0.92, Crest-Faktor ≤ 1.5 dB |
| **Emotionaler Dynamik-Bogen** (≥ 30 s) | Arousal-Pearson ≥ 0.85, Valence-Pearson ≥ 0.80, Klimax-Peak-Abw. ≤ 2 Segmente |
| FeedbackChain-Rollback | \|MOS_neu − MOS_alt\| > 0.05 → sofortiger Rollback auf `best_result` |

---

## Sprachkonvention

- **Nutzer-Meldungen, UI-Texte, Fehlermeldungen**: **Deutsch**
- **Code-Kommentare, Docstrings**: **Englisch**
- **Log-Meldungen** (Logger): **Englisch**

Fehlermeldungen immer mit **Ursache** + **Lösungsvorschlag** auf Deutsch.

---

## Frontend-UX-Pflichtregeln (§11.4 — `ModernMainWindow`)

### Progress Bar
- **Range immer `setRange(0, 10000)`** — 1 Einheit = 0,01 % → Anzeige in 0,1 %-Schritten
- `ModernProgressBar.setValue(v)` filtert Deltas < 10 (Rauschen) heraus
- Signale emittieren 0–100, das Slot-Lambda skaliert: `lambda v: pb.setValue(v * 100)`
- Completion-Marker: `setValue(10000)` (≡ 100 %)
- Verboten: `setRange(0, 100)` in `ModernMainWindow`

```python
# RICHTIG:
self.progress_bar.setRange(0, 10000)
self._load_progress.connect(lambda v: self.progress_bar.setValue(v * 100))
# in _on_item_progress (signal bringt 0–100):
val = max(100, min(10000, progress * 100))
self.progress_bar.setValue(val)
```

### Echtzeit-Defektzähler (`defect_count_live_label`)
- Widget `QLabel self.defect_count_live_label` im linken Panel (Header-Zeile neben "erkannte Defekte:")
- Beim Scan-Start: `setText("🔍 Analysiere…")`, `setVisible(True)`
- In `_update_defects(defects)` nach `active`-Berechnung: `setText(f"⚠ {n} Defekte")` / `"✅ Sauber"`
- Niemals `setVisible(False)` nach abgeschlossener Analyse

### Waveform-Defektfarben (`WaveformWidget._draw_defect_overlay`)
- **Jeder aktive Defekttyp bekommt ein eigenes 5-px-Band** (gestapelt von oben)
- Farbzuweisung: `clicks=Rot, crackle=Orange, pops=Gelb, clipping=Dunkelrot, hum=Violett, noise=Blau, sibilance=Cyan, dropout=Pink, wow=Grün, flutter=Hellgrün, rumble=Graublau, dc_offset=Gelbgrün`
- Alpha proportional zur Severity (leicht → gedämpft, schwer → voll gesättigt)
- Summarybadge unterhalb aller Bänder: `"⚠ N Defekte erkannt"`
- Verboten: einzelne Sammel-Bar mit einem Farbton für alle Defekte

### UI-Trennung Tonträger / Restaurierbarkeit
- **"Erkannter Tonträger:"** → `detected_medium_label` (Carrier-Name + Konfidenz)
- **"Restaurierbarkeit:"** → `restorability_banner` (Score 0–100 + MOS-Erwartung)
- Die Felder dürfen **nicht** vertauscht oder kombiniert werden

### Shortcuts (`_setup_shortcuts`)

| Taste | Aktion |
|---|---|
| `Space` | Play / Pause (Original) |
| `A` | Original abspielen |
| `B` | Restauriertes Audio abspielen |
| `Ctrl+O` | Datei öffnen |
| `Ctrl+S` | Exportieren |
| `Ctrl+R` | RESTORATION starten |
| `Ctrl+Shift+R` | STUDIO_2026 starten |
| `Escape` | Verarbeitung abbrechen |
| `Ctrl+Z` | Letzten Export-Pfad in Zwischenablage |
| `L` | Lyrics-Timeline-Overlay ein/aus (`_toggle_lyrics_overlay()`) |

- Kein Shortcut darf doppelt registriert werden (Duplikat-Check nach jeder Änderung)

### Warmup
- Nach `_setup_shortcuts()` in `__init__`: `QTimer.singleShot(2000, ...)` startet `warmup_models_background()` als Daemon-Thread
- Warmup berührt keinerlei UI-Objekte (kein GUI-Zugriff aus dem Thread)

### AudioFileValidator Gate
- Vor jedem `_bg_load`-Thread-Start: `get_audio_file_validator().validate(Path(file_path))`
- Zugriff **ausschließlich über Bridge** (`_bridge_get_audio_file_validator()`) — kein Direktimport aus `backend/core/`
- Bei Fehler: deutsche Fehlermeldung in `detected_medium_label` + `QMessageBox.warning()` (nicht `.critical()`)
- Kein Audiodatei-Laden ohne dieses Gate

### Audio-Lade-Kaskade (`_bg_load`-Thread)

Drei Stufen — nächste nur bei Fehler der vorherigen:

| Stufe | Bibliothek | Formate |
|---|---|---|
| 1 | `soundfile.SoundFile` | WAV, FLAC, OGG, AIFF (chunk-basiert, Prozent-Feedback) |
| 2 | `pedalboard.io.AudioFile` | MP3, M4A, WMA, AAC (chunk-basiert) |
| 3 | `librosa.load()` | Letzter Fallback (audioread-Backend) |

### Bridge-Fallback (`_BRIDGE_AVAILABLE`)
- `from backend.api.bridge import ...` wird am Modul-Anfang in `try/except ImportError` gewrappt
- Falls Bridge-Import fehlschlägt: `_BRIDGE_AVAILABLE = False` + Stub-Funktionen definieren
- `_export_guard` muss **immer** funktionieren (NaN-Guard-Pflicht) — dessen Stub ist vollständig implementiert
- Alle anderen Stubs geben `None` zurück — Aufrufer prüfen auf `None` und degradieren graceful
- Kein `_BRIDGE_AVAILABLE`-Check im Hot-Path nötig, da Stubs identische Signatur haben

```python
except ImportError:
    _BRIDGE_AVAILABLE = False
    def _export_guard(audio):        # vollständiger NaN-Guard-Stub
        import numpy as _np; audio = _np.asarray(audio, dtype=_np.float32)
        return _np.clip(_np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    def _bridge_get_quality_mode(): return None  # etc.
```

### Thread-Safety-Muster (`_dispatch_to_gui`)
- **Absolutes Verbot**: Kein Qt-Widget-Zugriff aus Hintergrundthreads
- Pattern in `ModernMainWindow`: `_gui_dispatch = pyqtSignal(object)` — sendet `Callable`-Objekte
- In `__init__`: `self._gui_dispatch.connect(lambda fn: fn())`
- Hintergrundthread ruft `self._dispatch_to_gui(lambda: widget.setText("..."))` auf
- Alternativ bei argloser Aktualisierung: `QTimer.singleShot(0, fn)` aus dem Thread
- Niemals `widget.setText()`, `widget.setVisible()` o. ä. direkt aus `threading.Thread` aufrufen

```python
# RICHTIG im Hintergrundthread:
self._dispatch_to_gui(lambda l=label: self.detected_medium_label.setText(l))
# oder:
QTimer.singleShot(0, lambda: self.status_text.setText(msg))

# FALSCH (direkt aus Thread):
self.detected_medium_label.setText(label)   # → Qt-Undefined-Behavior
```

### BatchProcessingThread — Signal-Kontrakt

| Signal | Typen | Bedeutung |
|---|---|---|
| `item_started` | `str` | item_id: Verarbeitung begonnen |
| `item_progress` | `str, int` | item_id, 0–100: Fortschritt |
| `item_finished` | `str` | item_id: erfolgreich abgeschlossen |
| `item_finished_with_result` | `str, object` | item_id, RestorationResult |
| `item_error` | `str, str` | item_id, Fehlermeldung (Deutsch) |
| `all_finished` | — | Alle Queue-Einträge abgearbeitet |
| `defect_update` | `dict` | Defekt-Display-Dict (via `_defect_analysis_to_display`) |
| `phase_update` | `str` | Phasenname für Status-Label (Deutsch) |
| `waveform_data` | `ndarray, int` | Audio + sr für WaveformWidget |
| `mode_update` | `str` | Qualitätsmodus-String (FAST/BALANCED/QUALITY/MAXIMUM) → `_update_mode` → `resource_status_widget` |
| `ml_status_update` | `bool, list` | ml_active, active_plugins → `_update_ml_status` → `resource_status_widget` |

- `item_progress` emittiert 0–100 → `_on_item_progress` skaliert: `max(100, min(10000, progress * 100))`
- `defect_update`-Dict enthält Key `"status"`: `"detected"` (vor Restore) / `"correcting"` / `"completed"`
- Interne Batch-Progress-Stufen: Laden 3 %, DefectScan 28 %, vor Restore 50 %, Pipeline 25–90 %, Save 100 %
- `_on_item_error` zeigt deutsche Fehlermeldung in `detected_medium_label` + Kurzstatus in `status_text` — kein `QMessageBox` (Verarbeitung kann weiterlaufen)
- `progress_callback`-Signatur: `(pct: int, msg: str, elapsed_s: float = 0.0) → None`; ETA-String wird nur angezeigt wenn `pct > 5 and elapsed_s > 0`

### Watchdog-Timer (`_watchdog_timer`)

- **Zweck**: Beendet zwangsweise blockierende Threads (ONNX-Deadlock, hängender C-Extension-Call)
- **Typ**: `QTimer(self)`, `setSingleShot(True)`, im Hauptthread — daher GUI-Zugriff im Callback erlaubt
- **Timeout**: `max(300_000, pending_files * 600_000)` ms — 10 min pro Datei (deckt Pipeline-Budget 120 s/min × 5 min)
- **Start**: in `_start_processing()` unmittelbar vor `batch_thread.start()`
- **Stop**: in `_on_all_finished()` und `_cancel_processing()` via `_watchdog_timer.stop()`
- **Callback `_on_watchdog_timeout`**: `requestInterruption()` → `wait(3000)` → `terminate()` falls noch aktiv → deutsche Fehlermeldung in `detected_medium_label` + `QMessageBox.warning()`
- Verboten: Watchdog per `threading.Timer` — kein GUI-Zugriff aus Nicht-Qt-Thread erlaubt

### Async-Analyse-Kette nach Datei-Öffnen

Nach `_open_file` starten 4 nicht-blockierende Daemon-Threads in Reihe:

| # | Thread-Funktion | Backend-Aufruf | GUI-Update-Methode |
|---|---|---|---|
| 1 | `_bg_load` | 3-stufige Audio-Kaskade | `_on_file_loaded` via `_dispatch_to_gui` |
| 2 | `_carrier_bg` | `get_medium_classifier_fn()(mono, sr)` | `_update_carrier_display` via `_dispatch_to_gui` |
| 3 | `_detect_era_genre_bg` | `get_era_classifier_fn()` + `get_genre_classifier_fn()` | `detected_medium_label` via `QTimer.singleShot(0, ...)` |
| 4 | `_estimate_restorability_bg` | `get_restorability_estimator_class()().estimate(audio, sr)` | `restorability_banner` via `_dispatch_to_gui` |

- Alle Closures `except Exception: pass` — kein Thread-Absturz bei fehlendem ML-Modell
- DSP-Fallback für Restaurierbarkeit: SNR-Schätzung falls ML nicht verfügbar
- DefectScan läuft erst in `BatchProcessingThread` (nicht beim Öffnen) (vollständige Liste):

| Funktion | Zweck |
|---|---|
| `export_guard(audio)` | NaN/Inf-Guard + Clip [-1, 1] |
| `get_audio_file_validator()` | AudioFileValidator-Singleton (§10.5) |
| `get_defect_scanner()` | DefectScanner-Klasse |
| `get_defect_type()` | DefectType-Enum (für `_defect_analysis_to_display`) |
| `get_quality_mode()` | QualityMode-Enum |
| `get_restorer_classes()` | `(RestorationConfig, UnifiedRestorerV3)` |
| `get_medium_classifier_fn()` | `classify_medium(mono, sr)` → MediumResult |
| `get_era_classifier_fn()` | `classify_era(audio, sr)` → EraResult |
| `get_genre_classifier_fn()` | `classify_genre(audio, sr)` → GenreResult |
| `get_restorability_estimator_class()` | RestorabilityEstimator-Klasse |
| `get_carrier_forensics_fn()` | `analyze_carrier_forensics(mono, sr)` → dict |
| `get_audio_exporter_class()` | AudioExporter-Klasse (None wenn fehlt) |
| `cache_defect_result(path, result)` | DefectScan-Cache schreiben (FIFO, 64 Einträge) |
| `get_cached_defect_result(path)` | DefectScan-Cache lesen |
| `clear_defect_cache(path=None)` | Cache-Eintrag oder gesamten Cache leeren |
| `warmup_models_background()` | ML-Modelle vorladen (Daemon-Thread) |

---

---

## §2.36 LyricsGuidedEnhancement (ab 9.10.x — PFLICHT)

```python
# LyricsTranscriber: Whisper-Tiny ONNX (39 MB, CPUExecutionProvider, kein Netzwerk)
# Fallback bei Whisper nicht verfügbar: Energie-Segmentierung (DSP)

# ContentAwareProcessor — Salienz-Boosts pro Phonemklasse:
SALIENCY_BOOST = {
    "fricative_stressed":   2.0,   # G_floor = 0.90 — besonders wichtig bei altem Material
    "fricative_unstressed": 1.4,
    "vowel_stressed":       1.6,
    "vowel_unstressed":     1.0,
    "plosive":              1.5,
    "silence":              0.5,   # NR aggressiv in Stille-Segmenten
}

# LyricsGuidedTimeline — Shortcut L (Overlay an/aus im UI)
COLOR_MAP = {
    "vowel_stressed":   "#4CAF50",
    "fricative_stressed": "#FF9800",
    "plosive":          "#29B6F6",
    "silence":          "#B0BEC5",
}
# Datenschutz-Pflicht: Lyrics-Text NIEMALS geloggt, NIEMALS in RestorationResult.metadata
```

**Ablauf:** Transkription (Whisper-Tiny → Wort-Timestamps)
→ Phonem-Alignment **[Pflicht]**: `wav2vec2_forced_alignment` ONNX (125 MB, CPUExecutionProvider)
     Fallback: Energie-Schwellwert-Segmentierung (DSP) + Phonem-Prior aus Whisper-Token-IDs
→ Timeline-Segmentierung nach Phonemklassen (vowel_stressed / fricative / plosive / silence)
→ ContentAwareProcessor hebt pro Phonemklasse die Gate-Toleranz an (SALIENCY_BOOST).
Damit werden bei schlechtem Material Konsonanten und
betonte Silben besonders geschützt — die häufigste Ursache für "verwaschene" Restaurierungen.

---

## ML-Plugin-Status (verifiziert, April 2026)

| Plugin | Modell | Format | Status | Aufgabe |
|---|---|---|---|---|
| `apollo_plugin` | `models/apollo/apollo_model.pt` | TorchScript | ✅ ML aktiv | Codec-Artefakt-Entfernung |
| `bs_roformer_plugin` | `models/melbandroformer/melbandroformer_optimized.onnx` | ONNX (860 MB) | ✅ ML aktiv | Stem-Separation Gesang (Primär); **Modell-SR 44,1 kHz**: 48k→44,1k→48k Resampling (Lanczos-4, SNR-Budget ≈ −0,8 dB) — bei Bedarf 48k-natives Modell bevorzugen |
| `mdx23c_plugin` | `models/mdx23c/Kim_Vocal_2.onnx` + `Kim_Inst.onnx` | ONNX | ✅ ML aktiv | Stem-Separation (Fallback zu MelBandRoformer) |
| `uvr_mdxnet_plugin` | `models/uvr_mdx_net/uvr_mdx_net_inst_hq_{1..4}.onnx` | ONNX (4 Modelle) | ✅ ML aktiv | Instrumental-Separation Ensemble |
| `deepfilternet_v3_ii_plugin` | `models/deepfilternet/` (enc+dec+erb_dec) | ONNX (3 Sessions) | ✅ ML aktiv | Breitrauschen-Reduktion (energy_bias_db=−6.0) |
| `rmvpe_plugin` | `models/rmvpe/rmvpe.onnx` (26 MB) | ONNX | ✅ ML aktiv | **Pitch-Tracking (Primär)** — RMVPE 2023 |
| `crepe_plugin` | `models/crepe/` | ONNX | ✅ ML aktiv | Pitch-Tracking (Fallback zu RMVPE) |
| `fcpe_plugin` | `models/fcpe/` | ONNX | ✅ ML aktiv | Pitch-Tracking (Fallback zu CREPE) |
| `beats_plugin` | `models/beats/beats_iter3.onnx` (90 MB) | ONNX | ✅ ML aktiv | **Audio-Tagging (Primär)** — BEATs 2023, +10.7% mAP |
| `panns_plugin` | `models/panns/` | ONNX | ✅ ML aktiv | Audio-Tagging (Fallback zu BEATs) |
| `resemble_enhance_plugin` | `models/resemble_enhance/model.onnx` | ONNX (722 MB) | ✅ ML aktiv | Vocal Enhancement (via `hybrid_ml_denoiser`) |
| `sgmse_plugin` | `models/sgmse_plus/sgmse_plus.onnx` (≈120 MB) | ONNX | ✅ ML aktiv | **Dereverb/Enhancement (Primär)** — SGMSE+ 2022 |
| `mp_senet_plugin` | `models/mp_senet/mp_senet.onnx` (≈35 MB) | ONNX | ✅ ML aktiv | **Speech/Music Enhancement** — MP-SENet 2023 |
| `versa_plugin` | `models/versa/versa_mos.onnx` (≈45 MB) | ONNX | ✅ ML aktiv | **MOS (ohne Referenz)** — VERSA 2024 |
| `vocos_plugin` | `models/vocos/vocos_mel_spec_44khz.onnx` | ONNX | ✅ ML aktiv | **Vocoder 44.1 kHz (Primär)** |
| `bigvgan_v2_plugin` | `~/.aurik/models/bigvgan_v2/` | ONNX/PT | ✅ ML aktiv | Vocoder (Stufe 1.5, Fallback zu Vocos) |
| `cqtdiff_plus_plugin` | `models/cqtdiff_plus/` | ONNX/PT | ✅ ML aktiv | Dropout-Inpainting ≥ 50 ms |
| `flow_matching_plugin` | `models/flow_matching/` | ONNX/PT | ✅ ML aktiv | **Generatives Inpainting (Primär)** — Flow Matching SOTA |
| `audiosr_plugin` | `models/audiosr/` (5,9 GB, lazy) | ONNX/PT | ✅ ML aktiv | **Bandbreiten-Erweiterung** — AudioSR (lazy load) |
| `htdemucs_plugin` | `models/htdemucs/htdemucs_ft.onnx` (320 MB) | ONNX | ✅ ML aktiv | **Stem-Separation Instrumental (Primär)** — HTDemucs 2023, MUSDB18 SOTA; Fallback 1 für Gesang hinter MelBandRoformer |
| `dac_plugin` | `models/dac/dac_44khz.onnx` (≈80 MB) | ONNX | ✅ ML aktiv | **Neuronale Zwischen-Repräsentation** — Descript Audio Codec 2023; Inpainting-Konditionierung bei Dropout ≥ 50 ms |
| `matchering_plugin` | `matchering==2.0.6` (Python-Paket) | — | ✅ aktiv | Reference Mastering (Studio 2026) |

**Preprocessing-Details MelBandRoformer** (`bs_roformer_plugin._separate_onnx`):
- Modell-SR: 44 100 Hz, n_fft=7914, hop=441, 60 Mel-Bänder, feat_dim=384
- **Resampling-Pflicht**: 48 000 Hz → 44 100 Hz (Lanczos-4, Kaiser β=14) vor Inferenz,
  44 100 Hz → 48 000 Hz nach Inferenz; SNR-Budget beider Stufen zusammen ≈ −0,8 dB.
  Bei Verfügbarkeit eines 48k-nativen MelBandRoformer-Modells hat dieses Vorrang.
- Input-Shape: `[1, T, 60, 384]` (Mel-Band-Split-Features, **nicht** raw audio)
- Output-Shape: `[1, 1, 3958, T, 2]` → vocals-only Complex-STFT (Real+Imag)
- Instrumente als Residual: `instruments = mix − vocals`

---

*Diese Richtlinien gelten für alle KI-Agenten (GitHub Copilot, Claude, GPT-Instanzen)
die an Aurik 9 arbeiten. Vollständige normative Spezifikation: `.github/specs/01–08`.*
*Stand: März 2026 — Aurik 9.10.57*
