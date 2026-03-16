# Aurik 9 — Spec 02: Pipeline-Architektur

> Kanonischer Pipeline-Ablauf, RestorationResult-Spec, Restaurierungs-Modi,
> StemRemixBalancer, Studio-2026-Verarbeitungskette.

---

## §1.4 Restaurierungs-Modi

| Modus | Ziel | Charakteristik |
|---|---|---|
| **`restoration`** | Originalgetreue Restauration | Erhalt des historischen Klangs, minimaler Eingriff, LUFS-Diff ≤ 1 LU, kein Harmonic-Exciter, GP `mode="restoration"` konservativ |
| **`studio2026`** | Highend-Studio-Klang | Modern, kräftig — PQS MOS ≥ 4.5, Brillanz ≥ 0.90, Bass-Kraft ≥ 0.88, GP `mode="studio2026"` aggressiv |

**Restoration-Modus Pflicht-Invarianten:**
- Chroma-Korrelation Original↔Restauriert ≥ 0.95
- LUFS-Differenz ≤ 1 LU
- Kein hinzugefügtes Harmonic-Exciter-Material

**Studio-2026-Modus Pflicht-Invarianten:**
- PQS MOS ≥ 4.5 (Weltklasse)
- Brillanz-Score ≥ 0.90 (verschärft)
- Bass-Kraft ≥ 0.88 (verschärft)

---

## §1.5 Studio-2026-Verarbeitungskette (kanonische Reihenfolge nach Defektkorrektur)

```
1.  Stem-Separation (MDX23C lokal, Kim_Vocal_2/Kim_Inst)
2.  Vocals: VocalAIEnhancement (stimmtyp-adaptiv) + ConsonantEnhancement (Frikative adaptiv)
3.  Sub-Mix-Instrumente: genre-adaptiv (guitar/brass/piano/drums nach PANNs)
4.  Reference Mastering (optional): OT-Spektral-Matching, Chroma-Korrelation ≥ 0.92
5.  Multiband-Dynamik: phase_35_multiband_compression
6.  Präsenz & Air: phase_38 + phase_39 (> 12 kHz)
7.  Stereo-Imaging: phase_48 + phase_46
8.  EraAuthenticPerceptualCompletion (wenn Quell-BW < 10 kHz)
9.  Re-Stem-Mix: StemRemixBalancer.balance_remix() — KEIN nacktes vocals + instruments
    Invariante: |LUFS(mix) − L_orig| ≤ 0.3 LU guaranteed
10. Lautheit: phase_40 (−14 LUFS EBU R128)
11. True-Peak-Begrenzung: phase_47 (−1.0 dBTP)
12. Musical Goals: alle 14 Ziele prüfen (verschärfte Studio-Schwellen)
13. Vocos-Synthese (konditionell): wenn PQS-MOS < 4.3
    → vocos_mel_spec_24khz.onnx → HiFi-GAN → PGHI-ISTFT
```

### StemRemixBalancer (Pflicht nach getrennter Stem-Verarbeitung)

```python
class StemRemixBalancer:
    """Gain-korrigierter Re-Mix nach getrennter Stem-Verarbeitung.

    Algorithmus:
        1. Vor Separation: L_orig gesamt messen
        2. Vor Separation: vocal_weight via PANNs auf Original (max. 10-s-Excerpt)
           → vocal_weight MUSS vollständig feststehen BEVOR MDX23C startet
        3. Nach Verarbeitung: LUFS pro Stem messen (L_voc', L_inst')
        4. Gain-Korrektur:
           g_voc  = 10 ** ((L_orig_voc  − L_voc')  / 20)
           g_inst = 10 ** ((L_orig_inst − L_inst') / 20)
        5. Re-Mix: mix = g_voc · vocals + g_inst · instruments
        6. Final-Check: |LUFS(mix) − L_orig| ≤ 0.3 LU

    Invarianten:
        - Vocals/Instruments-Verhältnis: ΔdB ≤ ±0.3 dB vs. Original
        - Kein Clipping im Re-Mix (np.clip nach Summation)
        - TonalCenterMetric nach Re-Mix ≥ 98 % des Pre-Remix-Werts
        - Laufzeit: ≤ 0.5 s / Minute Audio
    """
    def balance_remix(self, vocals, instruments, original, sr, vocal_weight=0.5): ...
```

**Pflicht**: Kein nacktes `vocals + instruments` in `UnifiedRestorerV3`.
**Pflicht-Test**: `tests/unit/test_stem_remix_balancer.py` (≥ 20 Tests).

---

## §2.2 Pipeline-Ablauf (kanonisch, Code-genau)

### §2.2.1 Parallelisierungs-Invariante
- TIER 0 und TIER 1: IMMER sequenziell
- TIER 2–4: dürfen parallelisieren; Merge via `np.mean` NUR wenn gleiche Frequenzzone
- TIER 6: IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)

```
Audio-Eingang (mono/stereo, beliebige SR)
    ↓
[DCOffsetPreRemoval]  ← PFLICHT-VORSTUFE vor jeder FFT-Analyse (kein phase_30!)
    │ scipy.signal.lfilter([1, -1], [1, -0.9999]) — Hochpass-IIR 5 Hz
    │ Invariante: np.abs(np.mean(audio)) < 1e-6 nach Entfernung
    │ Begründung: DC-Offset verfälscht STFT Bin 0+1 und damit alle
    │   Spektralanalysen (OMLSA-Profil, DefectScanner, HarmonicPreservationGuard).
    │   phase_30 bleibt für Post-Kettenausgleich erhalten, ist aber KEIN Ersatz.
    ↓
[TransientDecoupledProcessing]  ← ZWEITER Schritt (nach DC-Entfernung)
    │ separate(audio, sr) → (audio_percussive, audio_harmonic)
    │ audio_percussive → NUR phase_01 + phase_27 (kein NR, kein EQ!)
    │ audio_harmonic → volle Pipeline
    ↓
[RestorabilityEstimator]  (< 5 s, optional)
    ↓
[EraClassifier]  → EraResult (decade, material_prior, confidence)
    ↓
[GermanSchlagerClassifier]  → SchlagerClassificationResult
    │ → aktiviert SCHLAGER_RESTORATION_PROFILE bei is_schlager=True
    ↓
[MediumClassifier]  → ClassificationResult (MaterialType, confidence)

  ⚡ PARALLEL (ThreadPoolExecutor max_workers=3):
    EraClassifier + GermanSchlagerClassifier + MediumClassifier gleichzeitig
    (ONNX gibt GIL frei → echte Parallelität)

    ↓
[DefectScanner]  → DefectAnalysisResult (27 DefectTypes)
    ↓
[CausalDefectReasoner]  → RestorationPlan (14 Kausal-Ursachen)
    ↓
[UncertaintyQuantifier]  → confidence → GP-Bounds adj.
    ↓
[GPParameterOptimizer]  → propose_pareto() → ParameterProposal (Pareto-Front)
    ↓
[HarmonicPreservationGuard]  ← NACH TDP, VOR phase_03/phase_29
    │ extract_harmonic_mask(audio_harmonic, sr) → protected_bins[t,f]
    │ G_floor = 0.85 an Harmonik-Bins, 0.10 sonst
    ↓
[UnifiedRestorerV3._select_phases()]
    ↓
[PerceptualEmbedder]  → AudioEmbedding (256-dim L2, Pre-Fingerprint)
    ↓
[Phasen-Ausführung]  ← jede Phase gewrapped durch PerPhaseMusicalGoalsGate
    │ 5-s-Sample → measure_quick(6 Ziele) → Rollback bei Δ > REGRESSION_THRESHOLD
    │ MAX_RETRIES = 5; STRENGTHS = [0.65, 0.50, 0.35, 0.20, 0.10]
    ↓
[EraAuthenticPerceptualCompletion]  (wenn Quell-BW < 10 kHz)
    ↓
[IntroducedArtifactDetector]  → ML_HALLUCINATION / NMF_RESIDUAL_CLICK / etc.
    ↓
[FeedbackChain.run()]  → iteriert bis PQS-MOS konvergiert || max_iterations
    ↓
[TemporalQualityCoherenceMetric]  (bei Dateien ≥ 25 s)
    ↓
[PerceptualQualityScorer]  → PQSResult (.mos, .nsim, .mcd_db, .spectral_coherence)
    ↓
[ExcellenceOptimizer]  → ExcellenceResult (GP-Params)
    ↓
[MusicalGoalsChecker]  → Dict[str, float] (alle 14 Ziele)
    ↓
[EmotionalArcPreservationMetric]  (bei Dateien ≥ 30 s)
    ↓
[MicroDynamicsEnvelopeMorphing]  ← LETZTER Schritt vor Export
    ↓
[GPParameterOptimizer.update()]  ← persistiert Lernerfolg
    ↓
Audio-Ausgang + RestorationResult
```

---

## Kanonische RestorationResult-Definition

```python
@dataclass
class RestorationResult:
    # ── Pflichtfelder ────────────────────────────────────────
    audio:                np.ndarray
    config:               "RestorationConfig"
    material_type:        "MaterialType"
    defect_scores:        dict["DefectType", float]
    phases_executed:      list[str]
    phases_skipped:       list[str]
    total_time_seconds:   float
    rt_factor:            float
    quality_estimate:     float   # = 0.40·(1−defect_severity) + 0.60·(pqs_mos−1)/4
    warnings:             list[str]
    metadata:             dict[str, Any]
    # ── Optionale Felder ─────────────────────────────────────
    pqs_result:           Optional[Any] = None    # .mos, .nsim, .mcd_db, .spectral_coherence
    musical_goals:        Optional[dict[str, float]] = None   # 14 Ziele → Score
    excellence:           Optional[Any] = None
    temporal_coherence:   Optional[Any] = None    # MOS-Spanne ≤ 0.30
    emotional_arc:        Optional[Any] = None    # Arousal/Valence Pearson
    restorability:        Optional[Any] = None    # 0–100
    confidence:           float = 1.0
    genealogy:            Optional[Any] = None
    harmonic_fingerprint: Optional[Any] = None    # 256-dim L2 Post-Fingerprint
    phase_gate_log:       Optional[list[str]] = None
    adaptive_thresholds:  dict[str, float] = field(default_factory=dict)
    physical_ceiling:     dict[str, float] = field(default_factory=dict)
    goal_applicability:   dict[str, bool] = field(default_factory=dict)
    goal_priority_log:    list[str] = field(default_factory=list)
    preview_mos:          Optional[float] = None
    era_decade:           Optional[int] = None
```

**quality_estimate-Formel (normativ):**
```python
quality_estimate = 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4
# VERBOTEN: * 1.15 Bonus; quality_estimate aus defect_severity allein
# Clip: max(0.0, min(1.0, quality_estimate))
```

**Serialisierungsregeln:**
- `audio`-Feld wird NICHT in JSON serialisiert
- NaN/Inf-Werte → `null` (via `clean_nans()`)
- `genealogy` → separates `<sha256_prefix>_genealogy.json`
- Neue Felder: immer mit Default `null`

---

## §2.29 PerPhaseMusicalGoalsGate — Adaptive Regression-Schwellen

```python
# Schwellwerte restorability-adaptiv:
REGRESSION_THRESHOLD_GOOD: float = 0.012   # restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = 0.040   # restorability 40–69
REGRESSION_THRESHOLD_POOR: float = 0.060   # restorability < 40
SAMPLE_DURATION_S: float = 5.0
MAX_RETRIES: int = 5
_RETRY_STRENGTHS: list[float] = [0.65, 0.50, 0.35, 0.20, 0.10]

# Schnell-Ziele (≤ 200 ms Gesamtcheck):
FAST_GOALS_SUBSET = [
    "brillanz", "waerme", "groove",
    "tonal_center", "natuerlichkeit_mfcc_proxy", "timbre_authentizitaet",
]
# Phasen-adaptive Sample-Dauer (§9.7.3):
PHASE_SAMPLE_DURATIONS = {
    "phase_30": 1.5,  "phase_05": 1.5,  "phase_02": 2.0,
    "phase_15": 1.5,  "phase_11": 1.5,  "phase_18": 2.0,
}

# Datenfluss-Invariante: restorability_score MUSS aus RestorabilityEstimator stammen:
re_result = RestorabilityEstimator().estimate(audio, sr, defect_analysis)
gate = PerPhaseMusicalGoalsGate()
for phase in selected_phases:
    audio, scores, _ = gate.wrap_phase(
        phase, audio, sr, scores_before,
        restorability_score=re_result.restorability_score,
        applicable_goals=goal_filter.applicable,
    )
```
