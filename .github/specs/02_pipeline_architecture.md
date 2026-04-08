# Aurik 9 — Spec 02: Pipeline-Architektur

> Kanonischer Pipeline-Ablauf, RestorationResult-Spec, Restaurierungs-Modi,
> StemRemixBalancer, Studio-2026-Verarbeitungskette.

---

## §1.4 Restaurierungs-Modi

| Modus | Ziel | Charakteristik |
| --- | --- | --- |
| **`restoration`** | Originalgetreue Restauration — Tonträgerkette invertieren (§2.46) | Erhalt des historischen Klangs, minimaler Eingriff, LUFS-Diff ≤ 1 LU, kein Harmonic-Exciter, GP `mode="restoration"` konservativ |
| **`studio2026`** | Highend-Studio-Klang — Carrier-Chain-Inversion + Enhancement | Modern, kräftig — PQS MOS ≥ 4.5, Brillanz ≥ 0.90, Bass-Kraft ≥ 0.88, GP `mode="studio2026"` aggressiv |

**Restoration-Modus Pflicht-Invarianten:**

- Chroma-Korrelation Original↔Restauriert ≥ 0.95
- LUFS-Differenz ≤ 1 LU
- Kein hinzugefügtes Harmonic-Exciter-Material
- Rauschboden: material-adaptiv (Shellac ≤ −45, Vinyl ≤ −55, Tape ≤ −60, Digital ≤ −72 dBFS) — Studio-Ambience bewahren (§0a)
- HPI-Gate: `timbral_fidelity` dominant (§2.44) — akustisch nicht unterscheidbar vom Original

**Studio-2026-Modus Pflicht-Invarianten:**

- PQS MOS ≥ 4.5 (Weltklasse)
- Brillanz-Score ≥ 0.90 (verschärft)
- Bass-Kraft ≥ 0.88 (verschärft)
- Rauschboden ≤ −72 dBFS (§0a)
- HPI-Gate: PQS-Improvement dominant (§2.44)

### §1.4a [RELEASE_MUST] Fail-Fast-Kontrakt für kritische Qualitätsmodule (v9.10.130)

Kritische Qualitätsmodule dürfen in `restoration` und `studio2026` nicht unbemerkt in
qualitativ schwache Platzhalterpfade fallen.

**Kritische Module:**

- `PerceptualQualityScorer` / PQS
- `HolisticPerceptualGate`
- `ArtifactFreedomGate`
- `MusicalGoalsChecker` (P1/P2)

**Pflichtregeln:**

1. Fällt ein kritisches Modul zur Laufzeit aus, MUSS ein strukturierter
    `fail_reason`-Eintrag erzeugt werden (`severity=failed`, `component`, `error_code`).
2. Für `studio2026` ist bei Ausfall von PQS oder HPI kein stiller Positiv-Proxy erlaubt.
    Der Run MUSS in einen kontrollierten Safe-Mode mit striktem End-Gate wechseln
    oder fail-fast abbrechen.
3. Für `restoration` gilt: Primum non nocere hat Vorrang. Bei unklarer Qualitätslage
    MUSS auf das beste artefaktfreie Checkpoint-Audio oder Input zurückgerollt werden.
4. VERBOTEN: Konstante positive Platzhalter (`pqs_improvement=0.1` o. ä.) als
    dauerhafter Ersatz für echte Qualitätsmessung im finalen Exportpfad.

**Invariante:** Ein Export darf nie allein deshalb passieren, weil ein kritischer
Qualitätsdetektor nicht verfügbar war.

---

## §1.5 Studio-2026-Verarbeitungskette (kanonische Reihenfolge nach Defektkorrektur)

```text
1.  Stem-Separation (MDX23C lokal, Kim_Vocal_2/Kim_Inst)
2.  Vocals: VocalAIEnhancement (stimmtyp-adaptiv) + ConsonantEnhancement (Frikative adaptiv)
    + Vocal-Intimitäts-Gate (Pre/Post-Check; Rescue bei Delta < -0.04)
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

### §2.2.0 Sample-Rate-Vertrag (Dual-SR, [RELEASE_MUST])

- `analysis_sr = import_sr` (native): DefectScanner, RestorabilityEstimator, EraClassifier, MediumDetector, classify_clipping/analyse_clipping.
- `processing_sr = 48000`: alle Verarbeitungsphasen (01–64), PMGG, ML-Plugins, Export-Gates.
- Es müssen zwei getrennte Datenpfade geführt werden: `analysis_audio` (native SR) und `processing_audio` (48 kHz).
- Wenn die Normierung `import_sr -> 48000` fehlschlägt, MUSS die Verarbeitung fail-fast abbrechen; ein Weiterlauf der Phasen auf Nicht-48k ist unzulässig.
- Resampling darf nur `processing_audio` betreffen; `analysis_audio` bleibt unverändert in nativer SR.

### §2.2.1 Parallelisierungs-Invariante

- TIER 0 und TIER 1: IMMER sequenziell

### §2.2.2 SCHLAGER_RESTORATION_PROFILE — Definition (GermanSchlagerClassifier)

Wird aktiviert wenn `GermanSchlagerClassifier.is_schlager == True` (Gesamt-Konfidenz ≥ 0.52, gem. §2.19 Spec 03).
**Invariante**: Aktivierungsschwelle ist **0.52** — kein abweichender Wert darf im Code verwendet werden.
Enthält adjustierte GP-Priors und aktivierte Pflicht-Phasen für das Genre.

```python
SCHLAGER_RESTORATION_PROFILE = {
    # GP-Priors (überschreiben die Era-basierten Defaults aus §2.14)
    "gp_priors": {
        "noise_reduction_strength":  {"mean": 0.60, "std": 0.08},   # moderater als 1940er (0.90)
        "reverb_reduction_strength": {"mean": 0.55, "std": 0.10},   # typisch: Hallplatten-Echo
        "eq_correction_strength":    {"mean": 0.50, "std": 0.08},   # Mid-Boost bewahren
        "harmonic_preservation":     {"mean": 0.90, "std": 0.05},   # hohe Harmoniebewahrungs-Prio
        "transient_strength":        {"mean": 0.45, "std": 0.08},   # Schlagzeug-Transienten sanft
    },
    # Pflicht-Aktivierte Phasen (unabhängig von DefectScanner-Ergebnis)
    "forced_phases": [
        "phase_42_vocal_enhancement",    # Gesang ist Haupt-Träger im Schlager
        "phase_19_de_esser",             # Vintage-Mikrofon → Sibilanten-Spitzen
        "phase_07_harmonic_restoration", # Harmonie-Authentizität (H2/H4-Bewahren)
        "phase_08_transient_preservation",  # Orchester-Attacken
    ],
    # Family-Scalars für SongCalibrationProfile (überschreiben material-basierte Defaults)
    "family_scalars_override": {
        "denoise":        0.65,   # sanfter als Shellac/pre-war (weniger aggressiv)
        "reverb":         0.60,   # Hallplatten sind Stilmerkmal — nicht vollständig entfernen
        "reconstruction": 0.70,
        "dynamics_eq":    0.55,
        "transient":      0.45,
        "general":        0.60,
    },
    # Vokal-Intimität besonders schützen (§2.36 / §8.3 Tiefen-Immersion)
    "vocal_intimacy_guard": True,
    # TonalCenter-Pflicht: Schlager streng tonal — kein Key-Shift toleriert
    "tonal_center_strict": True,
    # Typisches Erscheinungsbild: Analog-Tape (1950–1980)
    "expected_material_range": ["tape_standard", "tape_studio", "vinyl_standard"],
    "expected_era_range": (1950, 1985),
}
```

**Invariante**: `SCHLAGER_RESTORATION_PROFILE["family_scalars_override"]` überschreibt SongCalibrationProfile-Defaults, wird aber durch denselben `global_scalar`-Bound begrenzt (Anti-Overfitting). `SCHLAGER_RESTORATION_PROFILE` wird in `RestorationResult.metadata["schlager_profile_active"]` als `True` protokolliert.

> **Kreuzreferenz Spec 03 §2.19**: Die obige strukturierte Definition (GP-Priors, forced_phases, family_scalars_override, vocal_intimacy_guard) ist die autoritative Spec-02-Vollform. Spec 03 §2.19 ergänzt flache Zielwerte (`groove_dtw_max_ms`, `deessing_strength_cap`, `waerme_target`, `brillanz_target`) — diese sind additive Qualitätsziele, kein Ersatz für GP-Priors und forced_phases. **Implementierungen MÜSSEN beide Spec-Abschnitte konsultieren.** Konflikte: Spec 02 hat Vorrang bei strukturellen Feldern (forced_phases, family_scalars_override); Spec 03 §2.19 bei metrischen Zielwerten.

- TIER 2–4: dürfen parallelisieren; Merge via `np.mean` NUR wenn gleiche Frequenzzone
- TIER 6: IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)

```text
Audio-Eingang (mono/stereo, beliebige SR)
    ↓
[Dual-SR-Split]
    │ analysis_audio @ import_sr (unveraendert)
    │ processing_audio @ 48000 Hz (resampled)
    │ Invariante: Kein Processing auf Nicht-48k
    ↓
[DCOffsetPreRemoval]  ← PFLICHT-VORSTUFE vor jeder FFT-Analyse (kein phase_30!)
    │ Standard (alle Materialien): scipy.signal.lfilter([1, -1], [1, -0.9999])
    │   → Hochpass-IIR 1. Ordnung, Pol bei z=0.9999, fc ≈ 0.76 Hz @ 48 kHz
    │   → Sicher für BassKraftMetric: Cutoff << 20 Hz, kein Energieverlust im Bassband
    │ Material-Sonderfall reel_tape (Lücke-H-Fix v9.10.100):
    │   Tape-Transport erzeugt DYNAMISCHEN DC-Drift (Geschwindigkeitsschwankungen
    │   → Pitch-/Amplitudenmodulation → langsame Basislinienwanderung 0.1–2 Hz).
    │   Für material_type == "reel_tape" MUSS segmentweise DC-Entfernung erfolgen:
    │   scipy.signal.lfilter([1, -1], [1, -0.9995])  — aggressiverer Pol (fc ≈ 3.8 Hz)
    │   ODER: scipy.signal.filtfilt([1, -1], [1, -0.9995]) — zero-phase (bevorzugt)
    │   Begründung: causales lfilter erzeugt Phasendrehung < 10 Hz → verfälscht Onset-
    │   Zeitstempel in WowFlutter-Erkennung; filtfilt vermeidet das.
    │   VERBOTEN bei Tape: globale Mittelwert-Subtraktion (np.mean) — erfasst keinen Drift.
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
[SongCalibrationProfile]  (§2.31a, Pflicht)
    │ Input: material_type, mode, restorability_score, input_snr_db,
    │        max_defect_severity, pipeline_confidence
    │ Output: global_scalar + family_scalars
    │ Familien: denoise | reverb | reconstruction | dynamics_eq | transient | general
    │ Invariante: bounded scalars (anti-overfitting) + deterministische Berechnung
    │
    │ [RELEASE_MUST] Bounds (Lücke-G-Fix v9.10.100):
    │   global_scalar       ∈ [0.50, 1.50]  — kein Wert < 0.50 (neutralisiert alle Phasen)
    │                                          kein Wert > 1.50 (Soft-Saturation-Guard umgangen)
    │   family_scalars[*]   ∈ [0.30, 1.80]  — Untergrenze schützt vor Komplettunterdrückung
    │                                          einer Familie; Obergrenze verhindert Überamplitude
    │   VERBOTEN: np.clip(scalar, 0.0, 2.0) — zu weite Grenzen; nur enge Clipping erlaubt
    │   Pflicht: assert 0.50 <= global_scalar <= 1.50 vor Phasen-Ausführung
    ↓
[EraClassifier]  → EraResult (decade, material_prior, confidence)
    ↓
[GermanSchlagerClassifier]  → SchlagerClassificationResult
    │ → aktiviert SCHLAGER_RESTORATION_PROFILE bei is_schlager=True
    ↓
[MediumDetectorResult]  → transfer_chain, primary_material, confidence (aus PreAnalysis-Handover)

    ⚡ PARALLEL (ThreadPoolExecutor max_workers=3):
        EraClassifier + GermanSchlagerClassifier + RestorabilityEstimator gleichzeitig
    (ONNX gibt GIL frei → echte Parallelität)

    ↓
[MusikalischerGlobalplanDienst]  ← Stufe 4 (Cross-Phase-Reasoning)
    │ erstelle_globalplan(audio, sr, use_ml_classifiers=False)  [DSP-only]
    │ 13 Ära-Profile × 7 Genre-Modifikatoren → 17 Per-Phase-Adjustments
    │ Enrichment nach Stufe 8 mit era_decade (→ RestorationConfig.global_plan)
    ↓
[DefectScanner]  → DefectAnalysisResult (32 DefectTypes)
    ↓
[CausalDefectReasoner]  → RestorationPlan (34 Kausal-Ursachen)
    ↓
[UncertaintyQuantifier]  → confidence → GP-Bounds adj.
    ↓
[GPParameterOptimizer]  → propose_pareto() → ParameterProposal (Pareto-Front)
    ↓
[HarmonicPreservationGuard]  ← NACH TDP, VOR phase_03/phase_29
    │ extract_harmonic_mask(audio_harmonic, sr) → protected_bins[t,f]
    │ G_floor = 0.85 an Harmonik-Bins, 0.10 sonst
    │
    │ [RELEASE_MUST] Mask-Gültigkeit (Fix L, v9.10.100):
    │ Die Maske ist gültig für phase_03 (Denoise) und phase_29 (Tape-Hiss).
    │ Für alle übrigen Phasen (EQ, Pitch, Stem-Sep, Dereverb etc.) darf die
    │ initiale Maske NICHT unverändert wiederverwendet werden — das harmonische
    │ Spektrum verschiebt sich nach Pitch-Korrektur, EQ und Stem-Separation.
    │ Regel:
    │   (a) phase_03: initiale Maske (berechnet aus audio_harmonic, prä-Denoise).
    │   (a.1) phase_29 (Tape-Hiss): wenn UV3 nach phase_03 einen SNR-Gewinn
    │         > 12 dB misst (snr_after_03 − snr_before_03 > 12.0 dB), MUSS die
    │         Maske VOR phase_29 neu berechnet werden (rauschverdeckte Transienten
    │         sind nach Denoise freigelegt; alte Maske schützt Rauschartefakte
    │         statt echter Harmonik). Übergabe: `recompute_harmonic_mask=True`.
    │         Bei SNR-Gewinn ≤ 12 dB: initiale Maske weiterverwendbar.
    │   (b) phase_42/43 (Vocal), phase_44–45 (Instrument): Maske NEU aus
    │       dem zum Zeitpunkt der Phase aktuellen audio berechnen
    │       (Übergabe als `recompute_harmonic_mask=True` an HPG).
    │   (c) alle übrigen Phasen: kein HPG-Eingriff (Verarbeitungs-Semantik
    │       der Phase definiert selbst ihren Amplituden-Schutz).
    │ VERBOTEN: Globale Maske ohne Ggültigkeit über alle 64 Phasen propagieren.
    ↓
[UnifiedRestorerV3._select_phases()]
    ↓
[PerceptualEmbedder]  → AudioEmbedding (256-dim L2, Pre-Fingerprint)
    ↓
[Phasen-Ausführung]  ← jede Phase gewrapped durch PerPhaseMusicalGoalsGate
    │ 5-s-Sample → measure_quick(6 Ziele) → Rollback bei Δ > REGRESSION_THRESHOLD
    │ SongCalibrationProfile skaliert phasenfamilien-basiert strength/wet-dry
    │ (psychoakustisch priorisiert: P1/P2-Stabilität, Maskierung, Transienten)
    │ MAX_RETRIES = 5; STRENGTHS = [0.65, 0.50, 0.35, 0.25, 0.15]   # kanonisch gem. §2.29 _RETRY_STRENGTHS
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
[HolisticPerceptualGate]  → HPI-Score (inkl. artifact_freedom §2.49)
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
    # ── §2.38 KMV-Felder ─────────────────────────────────────
    deferred_phases:      list[str] = field(default_factory=list)   # Phasen die Stufe 2 benötigen
    refinement_complete:  bool = False                               # True nach ML-Veredelung
    stufe2_quality_estimate: Optional[float] = None                  # quality nach vollständigem ML-Pass
```

### §2.2.3 [RELEASE_MUST] Experience-Telemetrie-Vertrag (v9.11.1)

Für die Produktions-Closed-Loop-Steuerung müssen folgende Felder in `RestorationResult.metadata`
normativ vorhanden sein (fehlertolerant, aber schema-stabil):

- `song_calibration.cluster_key: str`
- `song_calibration.cluster_policy: dict`
- `joy_runtime_index: {joy_index: float, fatigue_index: float, components: dict}`
- `auto_improvement_recommendations: {count: int, recommendations: list[dict]}`

**Invarianten:**

1. Alle numerischen Werte sind finite und auf [0,1] bzw. plausible Bereiche begrenzt.
2. Fehlende Upstream-Teile führen zu leeren Strukturen (`{}`, `[]`), nicht zu Schema-Bruch.
3. Die Telemetrie ist advisory-only: sie darf Pipeline/Export nicht blockieren.

**Rationale:** Ohne explizite Freude-/Ermüdungs- und Root-Cause-Telemetrie bleibt die
geschlossene Nachbesserung intransparent und kann in UI/Orchestrator nicht stabil genutzt werden.

### §2.38a ML-Guard-Fallback-Metadaten (PFLICHT)

Wenn eine heavy ML-Stufe wegen RAM-Headroom-Guard nicht gestartet wird, MUESSEN strukturierte Metadaten geschrieben werden.

```python
metadata.setdefault("ml_guard_events", []).append(
    {
        "phase_id": "phase_20_reverb_reduction",
        "model": "SGMSE+",
        "reason": "insufficient_physical_ram_headroom",
        "required_gb": 9.0,
        "available_gb": 6.8,
        "channels": 2,
        "duration_s": 245.3,
        "fallback": "wpe_dsp",
    }
)
```

**Invarianten:**

- Kein Rollback auf Original-Audio als Guard-Reaktion.
- Phase bleibt ausgefuehrt (DSP/Fallback-Pfad) und wird in `phases_executed` gefuehrt.
- Betroffene Phase MUSS in `deferred_phases` eingetragen werden (Stufe-2-KMV-Nachzug).
- `metadata["ml_guard_events"]` ist JSON-serialisierbar und NaN/Inf-frei.

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

**[RELEASE_MUST] PMGG darf Phasen NIEMALS überspringen (kein Rollback auf Original-Audio).**
CausalDefectReasoner hat die Phase als notwendig bestimmt — sie MUSS angewendet
werden, ggf. mit reduzierter Stärke (best-effort). Nach max. Retries wird der Versuch
mit der geringsten Musical-Goal-Regression angewendet (action=`best_effort`).

VERBOTEN: `return audio, scores_before, "rollback", 0.0` — Rückgabe von
unverändertem Original-Audio gleichbedeutend mit Phasen-Skip.

```python
# Schwellwerte restorability-adaptiv:
REGRESSION_THRESHOLD_GOOD: float = 0.020   # restorability ≥ 70 (v9.10.77: §9.7.5 Reference-Aware)
REGRESSION_THRESHOLD_FAIR: float = 0.035   # restorability 40–69
REGRESSION_THRESHOLD_POOR: float = 0.055   # restorability < 40
SAMPLE_DURATION_S: float = 5.0

# Priority-Aware Retry-Budget (v9.10.79 + §2.31b v9.10.85):
_RETRY_STRENGTHS: list[float] = [0.65, 0.50, 0.35, 0.25, 0.15]   # 5 Stufen, Floor 0.15 (Last-Resort)
# §2.31b: initial_strength < 0.90 (SongCal vorreduziert) → Ankerpunkte [0.80, 0.65, 0.50, 0.35, 0.20]
_PRIORITY_MAX_RETRIES: dict[int, int] = {1: 4, 2: 4, 3: 2, 4: 0, 5: 0}
_PRIORITY_THRESHOLD_FACTOR: dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.5, 4: 99.0, 5: 99.0}
# P1/P2: volle Kaskade (4 Retries + Emergency)
# Catastrophic-Threshold: max(0.08, 4.0 × adaptive_threshold) statt fest 0.20 (§2.31b)
# P3: max 2 Retries, 1.5× Regression-Toleranz
#   §2.31b: restorability_tier="good" → 3 Retries; tier="poor" → 1 Retry
# P4/P5: kein Retry — nur Logging (action="passed_p4p5_tolerated")
# Stagnation-Abbruch: max(0.002, threshold × 0.15) (§2.31b proportional)

# Schnell-Ziele (≤ 200 ms Gesamtcheck):
FAST_GOALS_SUBSET = [
    "natuerlichkeit", "authentizitaet", "tonal_center",
    "timbre_authentizitaet", "artikulation", "emotionalitaet",
    "micro_dynamics", "groove", "transparenz", "waerme",
    "bass_kraft", "separation_fidelity", "brillanz", "spatial_depth",
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
# action ∈ {"passed", "retry1"..., "best_effort", "best_effort_rN", "passed_p4p5_tolerated"}
```

### §9.7.7 [RELEASE_MUST] PMGG Stable-Metric-Invariante (v9.10.79)

Metriken mit ML-zustandsabhängigem Gewicht **DÜRFEN NICHT** in `_PRECISE_METRICS` für PMGG-Delta-Checks stehen.

**Root-Cause `NatuerlichkeitMetric`**: CREPE-Load-State verändert die internen Gewichte zwischen
`scores_before` (CREPE nicht geladen → `w_crepe=0.0`) und `scores_after` (CREPE geladen → `w_crepe=0.18`).
Das erzeugt Pseudo-Regression Δ ≈ 0.15–0.28 auf unverändertem Audio, triggert die vollständige
P1-Retry-Kaskade (4 Retries + 2 Emergency) und erzwingt Phase_03 best-effort bei strength=0.056.

**Auswirkung auf Gänsehaut-Erlebnis**: Phase_03 bei 5.6 % Wet-Mix erreicht Noise Floor −55 dBFS
statt −72 dBFS. Der Air-Layer (8–20 kHz) und der Vokal-Intimität-Layer (4–8 kHz) bleiben unter
dem Rauschteppich verdeckt → kein „Ohr-in-die-Musik-Legen", keine Tiefen-Immersion.

**Invarianten**:

- `NatuerlichkeitMetric` läuft ausschließlich in `MusicalGoalsChecker` (Export-Gate), nie im PMGG-Delta.
- Neue Metriken vor `_PRECISE_METRICS`-Aufnahme: Eigenrauschen ≤ 0.02 auf identischen Audio-Paaren Pflicht.
- `_PRECISE_OVERRIDE_WARN_MS = 200.0` (angehoben von 120.0).

### §2.29c [RELEASE_MUST] PMGG Restorative-Phase-Baseline-Capping (v9.10.96)

**Problem**: In restorativen Phasen (Denoise, Dereverb, Declip, etc.) misst `scores_before` auf
defekt-belastetem Audio. Bestimmte Defekte **inflationieren** Metriken künstlich:

- Breitbandrauschen hebt `transparenz` (Spectral Crest) und `brillanz` (HF-Energie)
- Hall-Nachhall hebt `waerme` (LF-Energie-Ratio) und verdeckt `authentizitaet`-Verluste
- Dropout-Lücken verfälschen `groove` (Autokorrelation) und `micro_dynamics` (RMS-Envelope)

Nach der Restaurierung sinken die Werte auf **physikalisch korrekte Levels** → PMGG meldet
Falsch-Regression → Retry-Kaskade → best-effort bei minimaler Wet-Strength → Defekte bleiben.

**Lösung**: `_RESTORATIVE_PHASES` + `_CANONICAL_THRESHOLDS` + `effective_scores_before`:

```python
_RESTORATIVE_PHASES: frozenset[str] = frozenset({
    "phase_02", "phase_03", "phase_09", "phase_18",
    "phase_20", "phase_23", "phase_24", "phase_29", "phase_49",
})

_CANONICAL_THRESHOLDS: dict[str, float] = {
    "natuerlichkeit": 0.90, "authentizitaet": 0.88, "tonal_center": 0.95,
    "timbre_authentizitaet": 0.87, "artikulation": 0.85, "emotionalitaet": 0.82,
    "micro_dynamics": 0.88, "groove": 0.83, "transparenz": 0.82,
    "waerme": 0.75, "bass_kraft": 0.78, "separation_fidelity": 0.78,
    "brillanz": 0.78, "spatial_depth": 0.70,
}
```

**Algorithmus** in `_run_with_retry()`:

```python
# §2.29c Restorative-Phase-Baseline-Capping
if phase_id in _RESTORATIVE_PHASES:
    effective_scores_before = {}
    for goal, measured in scores_before.items():
        canonical = _CANONICAL_THRESHOLDS.get(goal, 0.80)
        effective_scores_before[goal] = min(measured, canonical + 0.05)
else:
    effective_scores_before = scores_before
# Delta-Check: scores_after[g] - effective_scores_before[g]
```

**Invarianten**:

- `_CANONICAL_THRESHOLDS` = Restoration-Mode-Schwellwerte + 0.05 Headroom
- Capping greift nur in `_RESTORATIVE_PHASES` — Enhancement-Phasen nutzen echte `scores_before`
- Defekt-inflationierte Baselines über Canonical+5% werden gedeckelt → kein false Regression-Trigger
- Deterministisch: kein Zufall, keine ML-Abhängigkeit

**Aktualisierte `PHASE_GOAL_EXCLUSIONS`** (v9.10.96 — kanonische Quelle: `backend/core/per_phase_musical_goals_gate.py`):

```python
PHASE_GOAL_EXCLUSIONS: dict[str, set[str]] = {
    # Broadband denoise: CREPE-Load-State + transient-shape mismatch +
    # K-S NOT invariant for shaped NR §9.7.11 ext (non-uniform NR reshapes
    # chroma-bin balance → key-label flip) + MFCC-Pearson/Centroid-CV
    # disturbed by spectral-envelope change after NR.
    # §2.31b material-adaptive: cd_digital/dat → reduce to {"natuerlichkeit", "artikulation"}.
    "phase_03": {"natuerlichkeit", "artikulation", "authentizitaet", "tonal_center", "timbre_authentizitaet"},
    # DeepFilterNet tape-hiss: same root-causes as phase_03.
    "phase_29": {"artikulation", "authentizitaet", "natuerlichkeit", "tonal_center", "timbre_authentizitaet"},
    # Comb-filter hum removal: G1/G2/G3 notches cause false regressions:
    #   - groove: §9.7.10 rms_env variance-normalisation artefact (50 Hz modulation)
    #   - timbre_authentizitaet: MFCC-Pearson/centroid disturbed by LF notches → false P2
    "phase_02": {"bass_kraft", "authentizitaet", "natuerlichkeit", "transparenz",
                 "groove", "timbre_authentizitaet"},
    # EQ / tonal shaping: broadband frequency shifts invalidate timbre comparisons.
    "phase_04": {"transparenz", "brillanz", "waerme", "authentizitaet", "natuerlichkeit", "timbre_authentizitaet"},
    # TDP/HPSS: Transient-Shaping.
    "phase_08": {"micro_dynamics", "artikulation"},
    # Wow/Flutter: K-S volatile after pitch-/speed-correction + Centroid-CV disturbed.
    "phase_12": {"tonal_center", "timbre_authentizitaet"},
    # Noise gate: VAD mask applies binary gains → micro-dynamics artifacts.
    "phase_18": {"micro_dynamics", "authentizitaet", "emotionalitaet", "groove"},
    # SGMSE+ reverb reduction: SGMSE+ spectral deconvolution disturbs
    # CREPE pitch confidence → natuerlichkeit false P1.
    "phase_20": {"authentizitaet", "natuerlichkeit"},
    # AudioSR spectral inpainting: synthesised gap-fill has no valid reference —
    # same mechanism as phase_24 (Dropout Repair).
    # timbre_authentizitaet: MFCC-Pearson/Centroid-CV disturbed by synthesis.
    "phase_23": {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"},
    # Dropout repair: synthesised gap-fill; same root-causes as phase_23.
    "phase_24": {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"},
    # Dereverb: authentizitaet regression from RT60 removal.
    "phase_49": {"authentizitaet"},
    # Vocal processing: NMF separation shifts NatuerlichkeitMetric sub-weights.
    "phase_19": {"natuerlichkeit", "timbre_authentizitaet", "micro_dynamics"},
    # Dithering / noise-shaping: micro-dynamics by design.
    "phase_17": {"micro_dynamics", "natuerlichkeit"},
    # Diffusion inpainting: synthesised content; artikulation reference absent.
    "phase_55": {"artikulation", "micro_dynamics"},
    # Bandwidth extension (AudioSR): adds HF content → brillanz intentionally rises.
    "phase_06": {"brillanz"}, "phase_07": {"brillanz"},
    # Transient / time-domain: micro-dynamics re-shaping alters onset metric.
    "phase_26": {"micro_dynamics", "artikulation"}, "phase_36": {"micro_dynamics", "artikulation"},
    # Passthrough / analysis-only phases: no musical scoring required.
    "phase_28": set(), "phase_05": set(), "phase_30": set(),
    # Click removal (phase_01, phase_27): impulse transients + spectral interpolation.
    #   - artikulation: clicks appear as transients → removal reduces onset-count correlation.
    #   - natuerlichkeit: spectral interpolation at click locations creates MFCC-smoothness
    #     discontinuities (transition from reconstructed frames to undamaged context). CREPE-
    #     based NatuerlichkeitMetric flags these as unnatural → false P1 regression (0.267
    #     confirmed in real-run, PMGG dithered to strength=0.17). Same mechanism as phase_02.
    "phase_01": {"artikulation", "natuerlichkeit"},  # click impulses + interpolation → false P2/P1
    "phase_27": {"artikulation", "natuerlichkeit"},  # click/pop removal — identical to phase_01
    # BANQUET blind denoising (phase_09): full-band neural spectral modification.
    #   - natuerlichkeit: MFCC-smoothness proxy disturbed by full-band NR (same as phase_03/29).
    #   - groove: crackle events appear as periodic impulsive onsets. GrooveMetric onset-based
    #     DTW proxy registers the change in LF onset density as rhythmic disruption. Real-run
    #     confirmed: regression=0.291 (P1), stagnation across all retries, strength=0.15.
    #     Same mechanism as phase_02 groove exclusion.
    #   - authentizitaet: crackle fills log-spectrum valleys (roughness low before BANQUET);
    #     after processing valleys reappear → roughness rises → false P1. Identical to phase_03.
    #   - timbre_authentizitaet: MFCC-Pearson/centroid-CV disturbed (same as phase_29).
    "phase_09": {"natuerlichkeit", "groove", "authentizitaet", "timbre_authentizitaet"},
    # LyricsGuidedEnhancement (phase_58): Fricative-Ramp-Gain (4–8 kHz) verändert Spektralenveloppe
    # wie shaped NR → K-S-Key-Label-Flip möglich (tonal_center).
    # Vowel-LPC-Shelving und Plosive-Burst ändern MFCC-Pearson/Centroid-CV (timbre_authentizitaet).
    # HINWEIS: Key muss "phase_58_lyrics_guided_enhancement" lauten — NICHT "phase_57"
    # (würde via startswith-Präfix-Matching phase_57_print_through_reduction treffen).
    "phase_58_lyrics_guided_enhancement": {"tonal_center", "timbre_authentizitaet", "artikulation", "emotionalitaet"},
}
```

**Änderungen v9.10.90 → v9.10.96**:

- phase_03/29: brillanz/transparenz entfernt (§9.7.12/13 SNR-robust); tonal_center + timbre_authentizitaet eingefügt (§9.7.11 ext: K-S NOT invariant to shaped NR; Centroid-CV-Disturbance).
- phase_12: **NEU** — K-S volatile nach Pitch-/Speed-Korrektur + Centroid-CV.
- phase_02: tonal_center entfernt (K-S stabil bei Kammfilter).
- phase_18: brillanz/transparenz/tonal_center entfernt; groove hinzugefügt.
- phase_20: brillanz/waerme/transparenz entfernt (§9.7.12/13/14 reverb-invariant).
- phase_23/24: timbre_authentizitaet hinzugefügt (MFCC-Pearson/Centroid-CV gestört durch Synthese).
- phase_49: brillanz/waerme/transparenz entfernt (§9.7.12/13/14 reverb-invariant).
- phase_08: aus Passthrough-Gruppe in eigenen Eintrag verschoben.

### §9.7.8 [RELEASE_MUST] Precise-Metric Audio-Cap (v9.10.79)

`_apply_precise_metric_overrides` kappt Audio auf **max. 2.5 s** vor dem Metric-Loop.

- Alle 7 verbleibenden präzisen Metriken (Brillanz, Wärme, TonalCenter, MicroDynamics,
  Artikulation, SeparationFidelity, Transparenz) sind spektral-stationär über kurze Fenster.
- Ohne Cap: `ArticulationMetric` (Short-Frame 5 ms Hop) und `SeparationFidelityMetric`
  (NMF) benötigen > 2 s/Call auf 60-s-Material → kumulative PMGG-Latenz 4+ s pro Phase.
- Mit 2.5 s Cap: alle 7 Metriken < 200 ms gesamt.

### §9.7.9 [RELEASE_MUST] Material-adaptive PHASE_GOAL_EXCLUSIONS (v9.10.85)

Für hochwertige digitale Quellen (`cd_digital`, `dat`) entfallen Rausch-bedingte Ausschlüsse
bei `phase_03` (Breitband-Denoise) und `phase_29` (DeepFilterNet Tape-Hiss):

**Root-cause**: Die Ausschlüsse für `brillanz`, `authentizitaet`, `transparenz` und `tonal_center`
entstehen durch HF-Rauschminderung auf analogen Medien — Tape-Hiss und Vinyl-Hiss verschieben
spektrale Flatness, ZCR und Rolloff. Digitale Quellen haben kein Breitbandrauschen → diese
Falsch-Regressions-Ursachen treten nicht auf.

**Stabile Ausschlüsse (bleiben für alle Materialien)**:

- `natuerlichkeit`: CREPE-Load-State ändert interne Gewichte material-unabhängig
- `artikulation`: Transient-shape mismatch bei leichter Filterung bleibt relevant
- `tonal_center`: K-S ist bei shaped/HF-selektiver NR **nicht** invariant (§9.7.11 ext v9.10.95) — nicht-uniformes NR verändert Chroma-Bin-Balance → Key-Label-Flip
- `timbre_authentizitaet`: MFCC-Pearson/Centroid-CV gestört durch Spektral-Hüllkurvenänderung nach NR

**Implementierung** in `wrap_phase()` nach dem `PHASE_GOAL_EXCLUSIONS`-Loop:

```python
# §2.31b Material-adaptive exclusion relaxation (v9.10.85, akt. v9.10.96)
if _excluded_goals:
    _mat_str = ... # aus phase_kwargs["material_type"] oder ["material"]
    if _mat_str in {"cd_digital", "dat"} and (
        phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
    ):
        _excluded_goals &= {"natuerlichkeit", "artikulation"}
```

**Qualitätswirkung**: Für digitale Quellen werden `authentizitaet`, `tonal_center` und
`timbre_authentizitaet` jetzt im PMGG-Delta aktiv gemessen → Regressions-Schutz greift für
digitale Pfade wo bisher Falsch-Ausschlüsse standen. brillanz/transparenz/waerme sind seit
§9.7.12/13/14 bei **allen** Materialtypen SNR-robust und nicht mehr ausgeschlossen.

### §9.7.10 [RELEASE_MUST] Groove-Proxy LF-Robustheit (v9.10.90)

**Problem**: `_measure_quick` berechnet die Groove-Metrik via Autokorrelation einer 10 ms-Hop
RMS-Energiehüllkurve `rms_env`. Die Normierungsbasis `autocorr[0]` ist gleich der Gesamtvarianz
von `rms_env`. 50/100 Hz-Hum erzeugt innerhalb jedes 10 ms-Frames (≈ 0.5–1 Hum-Perioden/Frame)
Frame-zu-Frame-Schwankungen, die `autocorr[0]` erhöhen, ohne die 500 ms-Rhythmusperiodizität
zu verändern. Ergebnis: `autocorr[lag_05]` / `autocorr[0]` hängt von der Hum-Stärke ab →
false groove-Delta bei `phase_02_hum_removal`, obwohl der echte Rhythmus unverändert bleibt.
Stagnation Δ=0.000000 entsteht, weil das Artefakt rein normierungsbedingt ist und sich mit der
Filter-Stärke nicht ändert.

**Fix**: 5-Frame Moving-Average (= 50 ms) auf `rms_env` **vor** `np.correlate()`:

```python
# §9.7.10 LF-Robustheit: 5-Frame-MA filtert 50/100 Hz-Hum-Modulation aus rms_env.
# Hum-Periode 10–20 ms → stark gedämpft; Groove-Periode 120–500 ms → nahezu unverändert.
_sw = min(5, len(rms_env) // 4)
if _sw >= 2:
    rms_env = np.convolve(rms_env, np.ones(_sw) / float(_sw), mode="valid")
autocorr = np.correlate(rms_env, rms_env, mode="full")
autocorr = autocorr[len(rms_env) - 1:]
autocorr /= autocorr[0] + 1e-12
```

**Invarianten**:

- `_sw = min(5, len(rms_env) // 4)` → keine Überglättung bei kurzen Clips (< 0.2 s, ≈ 12 Frames → `_sw=3`)
- `_sw < 2` → kein Smoothing (Edge Case: < 8 Frames = < 80 ms Audio)
- Groove-Score bleibt deterministisch (kein stochastischer Anteil)
- `autocorr[0]` nach MA repräsentiert ausschließlich rhythmische Energievarianz

**Tests**: `TestGrooveProxyLFRobustness` (4 Tests, test_74–test_77) in
`tests/unit/test_per_phase_musical_goals_gate.py`.

---

### §9.7.11 [RELEASE_MUST] Krumhansl-Schmuckler tonal_center Proxy (v9.10.91)

**Problem**: Der bisherige `tonal_center`-Proxy maß **Chroma-Konzentrations-Entropie**
(`1 − entropy/log(12)`). Das ist SNR-abhängig: Rauschen/Nachhall/EQ-Filter verteilen
Energie gleichmäßig über alle 12 Chroma-Bins → hohe Konzentration `scores_before`;
nach Denoise/Dereverb sichtbare Spektralpeaks → niedrigere Konzentration `scores_after`
→ false P2-Regression auf **jedem rauschreduzierenden Phase bei beliebiger Stärke**.
Δ≈0 Stagnation bestätigt globale Stärke-Unabhängigkeit = strukturelle Proxy-Invalidität.
Beobachtete Katastrophen in Produktionslogs (2026-03-30):

| Phase | Regression | Δ-Stagnation | Root-Cause |
| --- | --- | --- | --- |
| phase_49_advanced_dereverb | 0.5312 | 0.000010 | Nachhall füllt Chroma-Bins diffus |
| phase_08_transient_preservation | 0.5612 | 0.000025 | HPSS verschiebt harmonisch/perkussiv-Balance |
| phase_04_eq_correction | 0.0753 | 0.000600 | EQ-Notch/Shelf verschiebt Chroma-Bin-Amplituden |
| phase_18_noise_gate | 0.1721 (groove) | 0.002226 | VAD-Gating → Chroma-Sparsität |

**Lösung**: Krumhansl-Schmuckler (1990) Key Detection — SNR-invariant, weil gleichmäßiges
Rauschen alle 24 KS-Scores gleichmäßig hebt → argmax unverändert.

**Algorithmus**:

1. Chroma-Vektor aus FFT-Magnitude (Hann-Fenster, n=4096) über Frequenz > 27.5 Hz
2. Korrelere gegen 24 KS-Dur/Moll-Profile (alle 12 Root-Transpositionen)
3. `key_before = argmax` im Referenzsignal, `key_after = argmax` im verarbeiteten Signal
4. Zirkuläre Semitondistanz `d = min(|k_a − k_b| mod 12, 12 − ...) ∈ [0, 6]`
5. Moduswechsel (Dur ↔ Moll) = +1 Semiton-Äquivalent, max 6
6. `tonal_center = 1 − d/6` → 0 = Tritonus/maximale Verschiebung, 1 = gleiche Tonart

```python
# §9.7.11 Krumhansl-Schmuckler key detection (SNR-invariant)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Both normalized to zero-mean unit-variance for Pearson equivalence via np.dot

def _ks_key(signal_mono, n_fft=4096, sr=48000) -> int:
    spec = np.abs(np.fft.rfft(signal_mono * np.hanning(len(signal_mono)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    chroma = np.zeros(12); bins = np.where((freqs > 27.5) & (freqs < 4186))[0]
    np.add.at(chroma, np.round(12*np.log2(freqs[bins]/440+1e-12)).astype(int)%12, spec[bins])
    chroma -= chroma.mean(); chroma /= chroma.std() + 1e-12
    best_r, best_k = -np.inf, 0
    for root in range(12):
        r_maj = np.dot(chroma, np.roll(_ks_maj_n, root))   # _ks_maj_n = normalised
        r_min = np.dot(chroma, np.roll(_ks_min_n, root))
        if r_maj > best_r: best_r, best_k = r_maj, root
        if r_min > best_r: best_r, best_k = r_min, root+12
    return best_k

# Delta score (reference available):
d = min((k_proc % 12 - k_ref % 12) % 12, 12 - ...)   # circular
mode_penalty = 0 if same_mode else 1
tonal_center = 1.0 - min(6, d + mode_penalty) / 6.0
```

**Invarianten**:

- Fallback bei Stille / sehr kurzem Signal → `0.5`
- KS-Profile: Krumhansl & Schmuckler 1990 Table 1 (kanonisch, unveränderlich)
- Pearson-Äquivalenz: Profile werden zu `zero-mean, unit-variance` normiert → `np.dot = n × pearson`
- Kein `assert sr == 48000` nötig (sr-agnostisch durch `rfftfreq(n, d=1/sr)`)
- Deterministisch: kein Zufall in der Berechnung

**PHASE_GOAL_EXCLUSIONS nach §9.7.11** (tonal_center in folgenden Phasen **nicht** mehr ausgeschlossen):
`phase_02`, `phase_04`, `phase_08`, `phase_18`, `phase_49`

These exclusions were removed because the old entropy proxy was SNR-dependent. K-S is key-label-based
and does not react to spectral energy redistribution that doesn't cause a genuine pitch transposition.

**§9.7.11 Extension (v9.10.95/96)**: K-S ist bei **shaped/HF-selektiver NR** (phase_03 OMLSA+ResembleEnhance,
phase_29 DeepFilterNet) **nicht** invariant. Nicht-uniformes NR verändert Chroma-Bin-Balance selektiv
→ Key-Label-Flip möglich. Daher bleiben `tonal_center`-Ausschlüsse für phase_03 und phase_29 bestehen.
Phase_12 (Wow/Flutter) erhält ebenfalls tonal_center-Ausschluss: Pitch-/Speed-Korrektur verschiebt
fundamentale Frequenzen → K-S volatile.

**Tests**: `TestKrumhanslSchmucklerTonalCenter` (24 Tests, test_78–test_101) in
`tests/unit/test_per_phase_musical_goals_gate.py`. Enthält auch §9.7.12/13/14 Proxy-Tests
(brillanz HF Crest, transparenz Multi-Band Crest, waerme Sub-Band-Ratio).

---

## §2.37 [RELEASE_MUST] Frontend-Backend-PreAnalysis-Handover-Architektur (v9.10.127)

### Kernprinzip

Pre-Analyseergebnisse werden **einmalig** bei Import berechnet (`run_pre_analysis()`) und als **direkte Objektreferenz** (nicht Cache-Keys) weitergereicht. Cache-basierte Rekonstruktion in asynchronen Batch-Threads erzeugt Racebedingungen.

### Datenfluss: Import → Analysis → Queue → Batch → Denker

```
UI: _load_file(path)
  │
  ├─→ [A] Hard Cache Clear: _bridge_clear_cache_for_path(old_path)
  │       └─ Alte Caches (defect, era/genre, medium, restorability) aktiv löschen
  │
  ├─→ [B] _pre_analysis_bg() → run_pre_analysis(audio_native, sr_native, ...)
  │       └─ MediumDetector.detect() aufgerufen GENAU 1x (native SR)
  │       └─ Alle 5 Analysen parallel: Medium, Era, Genre, Defect, Restorability
  │       └─ Ergebnisse in Bridge-Cache speichern (LRU, content-addressed)
  │
  ├─→ [C] Frontend speichert: _latest_pre_analysis_result = PreAnalysisResult(...)
  │       └─ Complete object reference (nicht nur Cache-Keys)
  │
  └─→ [D] Mode-Click (Restoration / Studio 2026)
          │
          ├─→ _add_to_queue_with_mode()
          │   └─ queue_item.settings["pre_analysis_result"] = _latest_pre_analysis_result
          │   └─ falls vorhanden: queue_item.settings["cached_defect_result"] = pre_analysis_result.defects
          │
          └─→ BatchProcessingThread.run()
              │
              ├─→ [E] Check queue_item.settings.get("pre_analysis_result"):
              │       IF present: pre_result = settings["pre_analysis_result"]
              │       ELSE: Rekonstruiere von Bridge-Caches (Fallback)
              │       Zusätzlich: konkret verwendetes Defect-Result immer als
              │       `cached_defect_result` an denke()/UV3 weiterreichen
              │
              └─→ [F] AurikDenker.denke(pre_analysis_result=pre_result, ...)
                  │
                  └─→ UV3.restore(cached_medium_kwarg=..., ...)
                      └─ MediumDetector.detect() NICHT aufgerufen (bereits 1x in pre_analysis)
```

### Invarianten (RELEASE_MUST)

| Invariante | Ort | Status |
| --- | --- | --- |
| Hard Cache Clear bei neuem Import | `Aurik910/ui/modern_window.py` line ~11920 | ✅ |
| PreAnalysisResult Storage | `Aurik910/ui/modern_window.py` line ~12691 | ✅ |
| Queue-Handover | `Aurik910/ui/modern_window.py` line ~13939 | ✅ |
| Batch-Prioritization | `Aurik910/ui/modern_window.py` line ~2117 | ✅ |
| Defect-Handover-Absicherung | `Aurik910/ui/modern_window.py` line ~2107 | ✅ |
| Test: Exactly 1 detect() call | `tests/unit/test_pre_analysis_handover_no_double_detect.py` | ✅ |

**Kritische Invariante**: `MediumDetector.detect()` wird **GENAU 1x** aufgerufen (von `run_pre_analysis()`), nie 2x oder 3x.

**Zusätzliche Invariante**: Das für den Run tatsächlich verwendete `DefectAnalysisResult` MUSS `AurikDenker.denke()` und UV3 immer als `cached_defect_result` erreichen. Ein unvollständiges `PreAnalysisResult` darf keinen zweiten Defect-Scan erzwingen, solange bereits ein konkretes Defect-Result im Queue-Kontext vorliegt.

### Fallback-Hierarchie

Falls `queue_item.settings["pre_analysis_result"]` ist `None` (shouldn't happen):

1. Bridge-Cache Rekonstruktion bei einzelnen Caches
2. Wenn Cache incomplete: UV3 führt fehlende Analysen eigenständig aus
3. Monitoring: `metadata["pre_analysis_handover"]` dokumentiert Fallback-Nutzung

### Rationale: Warum nicht Bridge-Cache?

**Problem**: Zeitfenster zwischen Frontend und Batch erlaubt Racebedingungen

```python
# ❌ RACE CONDITION
# Thread 1 (Frontend):
bridge.cache_medium_result(path, medium)
bridge.cache_defect_result(path, defect)

# Fenster (ms) — Batch-Thread könnte stale Cache lesen
# Old cache von vorrigem File könnte persistent sein

# Thread 2 (Batch):
medium = bridge.get_cached_medium_result(path)  # Original oder degradiert?
defect = bridge.get_cached_defect_result(path)  # Aus alter Datei gelesen?
```

**Lösung**: Direct Object Reference (Frozen nach Frontend-Capture, keine Parallelität)

```python
# ✓ DETERMINISTIC
pre_result = queue_item.settings["pre_analysis_result"]  # Complete object
# Immutable nach Frontend-Capture → keine Racebedingungen
```

---

## §2.38 Kontinuierliche ML-Veredelung (KMV) — [RELEASE_MUST]

> **Kernprinzip**: Der PerformanceGuard verwirft überschrittene Phasen nie endgültig — er _deferriert_ sie.
> RT-Limit-Überschreitung führt zu DSP-Fallback für Sofort-Export **plus** automatischer Hintergrund-Veredelung.
>
> **Quality-First Ergänzung (v9.10.80)**: In den nutzerseitigen Standardpfaden
> (GUI/CLI/Batch) wird `no_rt_limit=True` gesetzt. Dadurch darf der Hauptlauf
> Qualität nicht zugunsten von RT reduzieren; `deferred_phases` entstehen dort
> primär durch Ressourcen-/Stabilitäts-Fallbacks (OOM, Headroom, Inference-Timeout),
> nicht durch RT-Budget-Cuts.

### Zweistufiger Export-Ablauf

```text
Stufe 1 (Sofort-Export, Quality-first im Standardpfad)
    │  Standard: no_rt_limit=True (GUI/CLI/Batch)
    │  Optionaler RT-limitierter Pfad: Deferral bei should_skip_phase
    │  Phasen die RT-Limit überschreiten: DSP-Fallback + in deferred_phases eingetragen
    │  Pipeline finalisiert; Qualitäts-Gate bestanden?
    │   └─ Nein → Stufe 1 abgebrochen (Fail-Reason in metadata)
    │   └─ Ja  → Atomischer Export (immediately listenable)
    │              Wenn len(deferred_phases) > 0:
    ↓
Stufe 2 (Hintergrund-ML-Veredelung, LIMIT_BACKGROUND = ∞)
    │  MLRefinementThread startet automatisch nach Stufe-1-Export
    │  Gecachte Analyse-Ergebnisse aus Stufe 1 (kein Neustart von DefectScanner,
    │    EraClassifier, MediumClassifier, GPParameterOptimizer)
    │  Vollständige UV3-Pipeline ohne RT-Limit (no_rt_limit=True)
    │  QThread.LowPriority + os.nice(10) auf Linux
    │  isInterruptionRequested() zwischen jeder Phase prüfen
    │  Qualitätsinvariante: quality(v2) ≥ quality(v1) → sonst alten Export behalten
    └→ Atomischer Export-Overwrite: result_v2.tmp → os.replace(output_path)
       signal: refinement_complete(output_path, final_RestorationResult)
```

### RAM-Guard (Stufe 2 Startbedingung)

```python
import psutil
avail_gb = psutil.virtual_memory().available / 1024**3
if avail_gb < 4.0:
    logger.warning("KMV Stufe 2 übersprungen: nur %.1f GB RAM verfügbar (< 4 GB)", avail_gb)
    return  # Stufe-1-Export bleibt permanent
```

### DeferredRefinementJob (Pflicht-Dataclass)

```python
@dataclass
class DeferredRefinementJob:
    """Queued job for background ML refinement (§2.38)."""
    output_path:          str                       # Pfad der Stufe-1-Exportdatei
    audio_original:       np.ndarray                # Original-Audio (unkomprimiert, pre-pipeline)
    sr:                   int                       # Sample-Rate (48000)
    mode:                 str                       # "restoration" | "studio_2026"
    deferred_phase_ids:   list[str]                 # Phasen die in Stufe 1 deferriert wurden
    cached_defect_result: Any                       # DefectAnalysisResult aus Stufe 1
    cached_era_result:    Any                       # EraResult aus Stufe 1
    cached_medium_result: Any                       # ClassificationResult aus Stufe 1
    stufe1_quality:       float                     # quality_estimate Stufe 1 (Mindest-Benchmark)
    created_at:           float = field(default_factory=time.time)
```

### MLRefinementThread — Signal-Kontrakt

```python
class MLRefinementThread(QThread):
    refinement_started    = pyqtSignal(str, int)    # output_path, n_deferred_phases
    refinement_phase_done = pyqtSignal(str, float)  # phase_id, quality_improvement_delta
    refinement_progress   = pyqtSignal(int, str)    # pct 0–100, phase_name
    refinement_complete   = pyqtSignal(str, object) # output_path, final_RestorationResult
    refinement_cancelled  = pyqtSignal(str)         # output_path → Stufe-1-Export bleibt
```

### Invarianten

- `LIMIT_BACKGROUND = float("inf")` ist ausschließlich für `MLRefinementThread` — niemals für BatchProcessingThread
- Atomisches Schreiben: `output_path.tmp` → `os.replace(output_path)` nach vollständigem Pass
- Kein Downgrade: `if stufe2_result.quality_estimate < job.stufe1_quality: skip_overwrite()`
- Single active refinement: Pro Prozess höchstens ein aktiver `MLRefinementThread`
- Escape-Abbruch: `requestInterruption()` → Stufe-1-Export bleibt unverändert erhalten
- `DeferredRefinementJob.audio_original` registriert in `ml_memory_budget` (Budget-Guard); freigegeben unmittelbar nach Stufe-2-Export oder Abbruch

## §2.38b [RELEASE_MUST] Deferred-Phases vs. Phase-Skip — Formale Abgrenzung

| Konzept | Definition | Erlaubt | Mechanismus |
| --- | --- | --- | --- |
| **Phase-Skip** | Phase wird **permanent** nicht ausgeführt — Original-Audio wird unverändert weitergereicht | **VERBOTEN** für P1/P2-Phasen (§2.29) | — |
| **Phase-Defer** | Phase wird jetzt mit DSP-Fallback ausgeführt, volle ML-Qualität in Stufe 2 nachgeholt | **ERLAUBT** | `deferred_phases.append(phase_id)` + KMV Stufe 2 |

**Invariante**: RT-Limit-Überschreitung → **immer Defer, nie Skip**. Der PerformanceGuard darf `should_skip_phase()` im Quality-First-Pfad (`no_rt_limit=True`) nie zurückgeben, wenn das die einzige Restaurierungsmethode für eine P1/P2-Ursache ist.

```python
# RICHTIG: Phase deferrieren (Stufe 2 holt nach)
result.deferred_phases.append(phase_id)
phase_result = _run_phase_dsp_fallback(phase_id, audio, kwargs)  # temporärer DSP-Fallback

# VERBOTEN: Phase-Skip auf Original-Audio
# return audio, scores_before, "rollback", 0.0  ← nicht erlaubt gemäß §2.29
```

**Deferred-Phases-Priorisierung in Stufe 2**:

1. Phasen mit P1/P2-Zielbezug (höchste Priorität)
2. Phasen mit P3-Zielbezug
3. Alle übrigen (P4/P5 best-effort)

Innerhalb jeder Prioritätsgruppe entscheidet die Reihenfolge im ursprünglichen Pipeline-Plan. Bei erneutem Ressourcenmangel: Phase für nächsten Anlauf vormerken, nicht dauerhaft ausführen.

**Endlosschleifen-Prävention**: Nach 3 fehlgeschlagenen Deferred-Aufholversuchen wird die Phase als `"non_recoverable"` markiert. `RestorationResult.metadata["deferred_failed"]` wird befüllt. Weitere automatische Versuche unterbleiben bis zu einem manuellen Neustart.

## §2.39 OOM-Recovery-Checkpoint-System — [RELEASE_MUST]

**Kernprinzip**: `systemd-oomd`-Kill oder `MemoryError` führen nie zu Totalverlust. Pipeline-Zwischenstand wird atomar auf Disk persistiert und beim nächsten Start automatisch zur Wiederaufnahme angeboten.

### Checkpoint-Lifecycle

| Schritt | Komponente | Aktion |
| --- | --- | --- |
| 1 | `_execute_pipeline()` MemoryError-Handler | `save_checkpoint()` → `sessions/<stem>_oom_checkpoint.json` + `_oom_audio.wav` |
| 2 | `ModernMainWindow.__init__` (1,5 s QTimer) | `find_pending_checkpoints()` → Dialog "Restaurierung fortsetzen?" |
| 3 | Nutzer bestätigt | `_resume_from_checkpoint()` → Original laden → normale Restaurierung |
| 4 | Erfolgreicher Abschluss | `delete_checkpoint()` → Cleanup |

### Modul: `backend/core/recovery_checkpoint.py`

```python
@dataclass
class RecoveryCheckpoint:
    input_path: str
    output_path: str
    phases_executed: list[str]
    phases_remaining: list[str]
    mode: str                              # "restoration" | "studio_2026"
    material_type: str                     # MaterialType.value
    era_decade: int | None
    defect_scores: dict[str, float]        # {defect_type: severity}
    defect_scores_full: dict[str, dict]    # Full DefectScore with locations
    restorability_score: float | None
    spectral_fingerprint: dict[str, float]
    quality_estimate_at_failure: float
    musical_goals_at_failure: dict[str, float]
    audio_wav_path: str                    # FLOAT WAV (verlustfrei)
    sample_rate: int
    original_input_path: str
    timestamp: float
    aurik_version: str = "9.10.57"
    failure_phase: str = ""
    failure_reason: str = "MemoryError"
```

### Pfad-Durchleitung

```text
BatchProcessingThread
  → denke(input_path=, output_path=)
    → restauriere()
      → _orchestriere()
        → RestaurierDenker.restauriere()
          → UV3 restore(input_path=, output_path=)
            → self._recovery_ctx
              → _execute_pipeline MemoryError-Handler
                → save_checkpoint()
```

## §2.40 Vollpipeline-Determinismus (PFLICHT)

Die komplette UV3-Kette muss fuer identische Eingaben deterministisch reproduzierbar sein.

```python
# Determinismus-Vertrag (normativ)
assert max_abs_err <= 1e-6
assert rms_err <= 1e-7
assert result_a.phases_executed == result_b.phases_executed
```

Pflichtregeln:

- Alle Seeds zentral setzen und im Result-Metadata dokumentieren.
- Keine unseeded Zufallsfunktionen in Produktionspfaden.
- Vergleichslaeufe mit identischen Prozessparametern (Threads, Mode, Config).

## §2.41 Structured Fail-Reason Taxonomie (PFLICHT)

`RestorationResult.metadata["fail_reasons"]` ist eine Liste strukturierter Eintraege.

Pflichtfelder pro Eintrag:

- `phase_id`
- `reason_code` (z. B. `ml_guard_low_ram`, `goal_regression_p1`, `quality_gate_fail`)
- `severity` (`info|warning|error`)
- `action` (`fallback|retry|best_effort|blocked`)
- `details` (JSON-serialisierbar, NaN/Inf-frei)

**Invariante:** Kein freier String-only Fehlerpfad ohne reason_code in Kernmodulen.

## §2.42 [RELEASE_MUST] Pipeline-Stabilitäts-Kontrakt (v9.10.81)

Zusammenfassung aller Stabilitäts-Invarianten. Jede Verletzung einer dieser Regeln ist ein Release-Blocker.

| ID | Mechanismus | Spezifikation | Schutz gegen |
| --- | --- | --- | --- |
| S-01 | Per-Phase-Inference-Timeout | §3.9.1 spec 08 | BLAS-Deadlock, korruptes Modell |
| S-02 | SIGTERM-Handler + Emergency-Checkpoint | §3.9.2 spec 08 | Graceful OS-Shutdown ohne Datenverlust |
| S-03 | Phase-Output-Guard (`@phase_output_guard`) | §3.9.3 spec 08 | NaN/Inf-Propagation aus ML-Ausgaben |
| S-04 | ThreadPoolExecutor-Lifecycle (shutdown) | §3.9.4 spec 08 | Zombie-Threads, Ressourcen-Leaks |
| S-05 | ml_memory_budget Startup-Reconciliation | §3.9.5 spec 08 | Stale-Allokation nach SIGKILL |
| S-06 | Structured Exception Logging | §3.9.6 spec 08 | Stille Fehler, leer `fail_reasons` |
| S-07 | Audio-Buffer-RAM-Guard | §3.9.7 spec 08 | OOM durch sehr große Audio-Dateien |
| S-08 | Lock-Acquisition-Order (ARM→PLM→MLBudget) | §3.9.8 spec 08 | Deadlock zwischen ARM und PLM |
| S-09 | MLRefinementThread Buffer-Release in finally | §3.9.9 spec 08 | RAM-Leak bei KMV-Abbruch |
| S-10 | watchdog + requestInterruption → terminate() | §11.4 spec 08 | Freeze > 90 min (Desktop-Watchdog) |
| S-11 | OOM-Recovery-Checkpoint (MemoryError-Pfad) | §2.39 | Python MemoryError → kein Totalverlust |
| S-12 | §2.38 KMV Stufe 2 mit 4 GB RAM-Guard | §2.38 | OOM bei Hintergrund-ML-Veredelung |
| S-13 | §2.38a ML-Headroom-Guard vor ML-Load | §2.38a | OOM während Modell-Laden |
| S-14 | Hybrid-Release-Mode (primary/fallback/blocked) | §13 spec 08 | Crash durch quarantänisierte Modelle |
| S-15 | Atomic File Writes (.tmp → os.replace) | §3.1 spec 08 | Korrupte Ausgabedatei bei Abbruch |

### Stabilitäts-Priorisierung

- **S-01 bis S-09**: Neue Invarianten aus Tiefenanalyse v9.10.81 — RELEASE_MUST.
- **S-10 bis S-15**: Bestehende Invarianten — bereits implementiert, hier zur Referenz.

### Für jedes neue Kernmodul / jede neue Phase gilt zusätzlich (§9.1 Checkliste):

- `try`/`except` mit §2.41-konformem `fail_reasons`-Eintrag (S-06).
- `@phase_output_guard` oder äquivalente manuelle Absicherung (S-03).
- `ml_memory_budget.try_allocate()` vor ML-Load mit `release()` in Fehler-Pfad (S-13).
- Kein `ThreadPoolExecutor` ohne Shutdown in Cleanup (S-04).
- `_check_audio_buffer_size()` bei direktem `soundfile.read()` (S-07).
- **[RELEASE_MUST] Längen-Invariante**: `len(phase_output) == len(phase_input)` — Phasen dürfen die Signallänge nicht verändern. `_execute_pipeline()` korrigiert akkumulierten Längendrift am Ausgang (Trim bei Überlänge, Zero-Pad bei Unterlänge). Dies betrifft insbesondere PGHI-basierte Phasen mit `padded=False` (letztes unvollständiges Fenster wird weggelassen) — Abhilfe: `n_samples=len(audio_in)` immer an `pghi_reconstruct_from_stft()` übergeben.

### Invarianten

- Checkpoint-Audio als `FLOAT` WAV — verlustfrei, kein Encoding-Verlust
- Ablauf: 7 Tage (`_MAX_CHECKPOINT_AGE_S`) — danach automatische Bereinigung
- Thread-safe: Alle Writes über `.tmp` + `os.replace` (POSIX-atomar)
- Datenschutz: Lyrics-Text NICHT im Checkpoint (§2.36 Pflicht)
- Wiederaufnahme nutzt das **Original-Audio** (nicht das Checkpoint-Audio) für volle Qualität
- Checkpoint-Audio dient als Fallback wenn Original fehlt
- **VERBOTEN**: Checkpoint-Audio als Primärquelle für Re-Restaurierung (Doppelverarbeitung degradiert Qualität)

---

## §2.44 [RELEASE_MUST] Holistic Perceptual Gate (v9.10.123)

Letztes Gate vor Export. Misst **Gesamt-Hörverbesserung** statt nur Einzel-Goals.

### Referenz-Paradoxon (Restoration)

Das Ziel ist Nähe zum **unbekannten Studio-Original**, aber wir besitzen nur den **degradierten Input**. Je erfolgreicher die Restaurierung, desto unähnlicher wird der Output dem degradierten Input. Deshalb misst `timbral_fidelity` nicht bloße Ähnlichkeit zum Input, sondern **strukturelle akustische Kohärenz**:

- **Spectral-Envelope-Kontinuität**: Keine unnatürlichen Lücken oder Spitzen im Frequenzspektrum
- **Crest-Factor-Konsistenz**: Dynamik-Verhältnis bleibt physikalisch plausibel
- **MFCC-Stabilität**: Klangfarben-Koeffizienten zeigen keine abrupten Sprünge

**Referenz-Anker-Strategie** (Restorability-abhängig):

- **Restorability > 70** (leichte Degradation): Input ist gute Annäherung ans Original → `timbral_fidelity` gegen Input
- **Restorability 50–70** (mittlere Degradation): Gewichtete Mischung aus Input-Referenz (60 %) und MERT-Referenz-Vektor aus GP-Memory (40 %)
- **Restorability ≤ 50** (schwere Degradation): Input zu weit vom Original entfernt → MERT-Referenz-Vektor aus GP-Memory (genre × material × ära) als primärer Anker (70 %), Input nur noch für musikalische Identität (30 %)

### MERT-Referenz-Embedding-Aufbau (v9.10.123)

Die GP-Memory-Referenz-Vektoren werden **automatisch** aus dem Verarbeitungsverlauf aufgebaut — kein manuelles Kuratieren nötig:

**Bootstrap (Cold-Start)**:

- Beim ersten Start: 12 Genre-Prototypen aus vortrainierten MERT-Embeddings (im Modell-Bundle enthalten, ~2 MB)
- Abdeckung: je 1 Prototyp pro Genre-Cluster (Schlager, Oper, Klassik, Jazz, Rock, Pop, Blues, Soul, Electronic, Latin, Folk, Metal)
- Ära-Differenzierung: 3 Ära-Bins (pre-1960, 1960–1990, post-1990) × 12 Genres = 36 Basis-Vektoren

**Inkrementeller Aufbau**:

- Nach jeder **erfolgreichen** Restaurierung (HPI > 0.5 UND artifact_freedom ≥ 0.95 UND alle P1/P2-Goals bestanden):
  - MERT-Embedding des Outputs wird in GP-Memory unter `genre × material × ära_bin` gespeichert
  - Exponential Moving Average (α = 0.15) mit bestehendem Referenz-Vektor → konvergiert ohne Ausreißer
- **Qualitäts-Gate für Referenz-Updates**: Nur Outputs mit HPI > 0.5 fließen ein — verhindert, dass mittelmäßige Restaurierungen die Referenz verschlechtern
- **Mindest-Observationen**: Referenz-Vektor wird erst ab 3 Beobachtungen als "kalibriert" markiert; davor: Bootstrap-Prototyp mit erhöhter Unsicherheit (GP-Lengthscale × 1.5)

**Fallback-Kaskade** (wenn kein passender Referenz-Vektor existiert):

1. Gleiche Genre-Familie + nächstliegende Ära → GP-Memory
2. Gleiche Ära + nächstliegendes Genre → GP-Memory
3. Bootstrap-Prototyp für Genre-Cluster
4. Genre-agnostischer Ära-Median (alle Genres der Ära gemittelt)
5. Kein Referenz-Vektor → `timbral_fidelity` rein gegen Input (Restorability-unabhängig)

### HPI-Formeln

**Restoration**: `HPI = MERT_similarity(input, output) × timbral_fidelity(input, output) × artifact_freedom × emotional_arc_preservation`

- `timbral_fidelity` dominant: strukturelle Klangkohärenz (nicht bloße Input-Ähnlichkeit)
- `artifact_freedom` (§2.49): Artefakt-Freiheit — Musical Noise, Pre-Echo, Spectral Holes = 0
- MERT_similarity: musikalische Identität bewahren (Melodie, Harmonie, Rhythmus)
- `emotional_arc_preservation`: Arousal/Valence-Bogen + **Makrodynamik** (Vers-/Refrain-/Bridge-Pegelrelationen bleiben erhalten) + Lyrics-Salienz (§2.36: Phonem-Boost-Verhältnisse im Output konsistent mit Enhanced-Zielwerten)
- RestorabilityEstimator > 0.85 → strengeres Gate

**Studio 2026**: `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`

- PQS-Improvement dominant (Qualität steigern > Original-Treue)
- `studio_quality_gain`: Abstand zu Referenz-Studioniveau (−14 LUFS, Noise ≤ −72 dBFS)
- `artifact_freedom` (§2.49): auch Enhancement darf keine Artefakte erzeugen
- MERT-Ähnlichkeit fließt mit reduziertem Gewicht ein (musikalische Identität bewahren, nicht Klangfarbe)

**Beide Modi**: `HPI > 0` → Export | `HPI ≤ 0` → Rollback auf weniger aggressive Variante

### HPI-Gewichtungs-Semantik

Die HPI-Multiplikation ist **nicht** gleichgewichtet — die Faktoren operieren auf unterschiedlichen Wertebereichen:

| Faktor | Wertebereich | Rolle |
| --- | --- | --- |
| `timbral_fidelity` | [0.8, 1.0] | Geringe Varianz — dominiert durch **Sensitivität**: kleine Abweichung → großer HPI-Einbruch |
| `artifact_freedom` | [0.0, 1.0] | **Veto-Faktor**: < 0.95 → Gate-Fail (Primum non nocere) |
| `MERT_similarity` | [0.5, 1.0] | Musikalische Identität — verhindert, dass Restaurierung das Stück verändert |
| `emotional_arc` | [0.7, 1.0] | Dynamik-Bogen + Makrodynamik — Narrative Struktur erhalten |

Ein Artefakt (`artifact_freedom` = 0.5) killt den HPI härter als eine leichte Timbre-Abweichung (`timbral_fidelity` = 0.95) — das ist beabsichtigt.

## §2.45 [RELEASE_MUST] Minimal-Intervention-Prinzip (v9.10.122)

**Restoration**: Phasen ohne hörbare Verbesserung werden NICHT angewendet:

- `perceptual_delta > 0` nachweisen (MERT-Embedding-Distanz oder timbral_fidelity-Delta)
- `perceptual_delta ≤ 0` → Skip

**Studio 2026**: Volle Enhancement-Kette aktiv, aber jede Phase muss Klanggewinn nachweisen:

- `perceptual_delta > 0` Pflicht — auch Enhancement-Phasen müssen messbaren Nutzen zeigen
- Phasen ohne messbaren Klanggewinn → Skip

## §2.45a [RELEASE_MUST] Mid-Pipeline-Loudness-Drift-Guard (v9.10.128)

### Problem

Die finale LUFS-Invariante (`LUFS-Differenz ≤ 1 LU`) schützt den Export, aber nicht zwingend frühe, hörbare Pegelkollapse innerhalb der subtraktiven Phasenkette.

### Pflicht-Invarianten

- Für breitbandig/subtraktive Phasen MUSS ein material-adaptiver per-Phase-RMS-Drift-Guard aktiv sein.
- Ein Guard darf die Phase nicht trivialisieren (`strength=0`/Bypass als Standardreaktion ist unzulässig).
- Bei Überschreitung des material-adaptiven RMS-Drift-Limits gilt: primär Dry/Wet-Rescue (mehr Dry-Anteil), sekundär sichere Makeup-Gain-Kompensation.
- Gain-Limits müssen den DSP-Peak-Guard nutzen: `np.percentile(np.abs(audio), 99.9)`.
- Phase-Metadaten müssen `rms_drop_db` und `loudness_makeup_db` ausweisen.
- Pipeline-Metadaten müssen stärkste Pegelabfälle separat ausweisen (z. B. `phase_regression_top_drops`).

### Normativer Scope (typische Kandidaten)

- Denoise / Hiss / Surface-Noise Reduction
- Noise-Gate
- Dereverb

### Rationale

Schützt §0 (Primum non nocere), §2.45 (Minimal-Intervention) und P1/P2-Hartregeln gegen frühe Klangausdünnung, ohne die Defektkorrekturwirkung zu verlieren.

## §2.46 [RELEASE_MUST] Carrier-Chain-Inversion (v9.10.122)

**Restoration-Modus**: Ziel = **gesamte Tonträgerkette invertieren**, nicht Einzel-Defekte reparieren.

**Signalkette** (vorwärts): `Studio-Monitor → Mic/Line → Preamp → Mixer → Carrier-Encoding (Tape/Vinyl/Shellac/Digital) → Alterung → Playback → ADC → Digital-File`

**Restaurierung** (invers, Reihenfolge beachten):

1. ADC-Artefakte entfernen (DC-Offset, Quantisierungsrauschen)
2. Playback-Verzerrungen invertieren (RIAA-Inverse, Azimuth-Korrektur, Wow/Flutter)
3. Alterungsschäden reparieren (Knistern, Dropout, Oxidation)
4. Carrier-Encoding invertieren (Bandrauschen, Vinyl-Groove-Distortion, Shellac-Rauschen)
5. Mixer/Preamp-Charakter: **bewahren** (Recording-Chain-Signatur = Teil des Originals)
6. Studio-Raumklang: **bewahren** (nicht über-entrauschen — Rauschboden material-adaptiv §0a)

**Studio 2026**: Carrier-Chain-Inversion + Enhancement-Kette (§1.5). Mixer/Preamp-Charakter darf modernisiert werden.

> Kreuzreferenz: Slim Core §2.46, Spec 01 §8.2 Rauschboden modus-differenziert

## §2.47 [RELEASE_MUST] Adaptive-Intelligence-Prinzip (v9.10.123)

Aurik verarbeitet **kein generisches Audio** — jede Eingabe ist ein einzigartiges Musikstück. Das System muss sich **vor Beginn der Verarbeitung** vollständig an das konkrete Material anpassen.

### Adaptions-Kaskade (kanonische Reihenfolge)

```text
1. MediumDetector.detect()      → transfer_chain, primary_material, composite flag
2. EraClassifier.classify()     → decade, era_profile, vintage_aesthetics
3. GenreClassifier              → genre_label, RESTORATION_PROFILE (5 definierte + DEFAULT)
4. RestorabilityEstimator       → 0–100, tier (GOOD/FAIR/POOR/EXTREME), scale_factor
5. DefectScanner.scan_all()     → 32 defect_types × severity × locations
6. CausalDefectReasoner         → 35 Ursachen → Phase-Selektion (CAUSE_TO_PHASES)
7. SongCalibrationProfile       → family_scalars [0.30–1.80] + global_scalar [0.50–1.50]
8. GPOptimizer.propose()        → Pareto-optimale Hyperparameter (14-D MOO)
```

**Resultat**: Dieselbe Pipeline verarbeitet Schellack 1928 (SNR 15 dB, BW 7 kHz, Mono) fundamental anders als CD 2005 (SNR 60 dB, BW 20 kHz, Stereo) — ohne manuellen Eingriff.

### GP-Wissenstransfer (v9.10.123)

- GPOptimizer persistiert Beobachtungen pro `gp_memory_key` (Genre × Material)
- **Cross-Material-Generalisierung**: Bei < 10 Beobachtungen für ein neues Material werden Hyperparameter-Priors (Kernel-Lengthscale, Signal-Varianz) aus dem nächstverwandten Material initialisiert gemäß Material-Ähnlichkeitsmatrix (siehe unten)
- **Anti-Overfitting**: `global_scalar ∈ [0.30, 1.80]` begrenzt GP-Vorschläge; Extreme führen zu Conservative-Fallback
- **Batch-Konvergenz**: Bei sequenzieller Verarbeitung mehrerer Dateien gleichen Materials konvergieren GP-Priors → spätere Dateien profitieren von früheren Ergebnissen

### Material-Ähnlichkeitsmatrix (v9.10.123)

Definiert die Transferierbarkeit von GP-Priors zwischen Materialien. Wert = Ähnlichkeit [0, 1]. Bei < 10 GP-Beobachtungen wird der Prior vom Material mit höchstem Ähnlichkeitswert übernommen.

```text
                  shellac  wax_cyl  vinyl_78  vinyl_std  tape_std  tape_stu  cassette  digital  mp3_lossy
shellac             1.00    0.85     0.75      0.40       0.15      0.10     0.10      0.05     0.05
wax_cylinder        0.85    1.00     0.70      0.35       0.10      0.10     0.08      0.05     0.05
vinyl_78rpm         0.75    0.70     1.00      0.65       0.20      0.15     0.15      0.08     0.08
vinyl_standard      0.40    0.35     0.65      1.00       0.45      0.40     0.35      0.15     0.12
tape_standard       0.15    0.10     0.20      0.45       1.00      0.85     0.70      0.25     0.20
tape_studio         0.10    0.10     0.15      0.40       0.85      1.00     0.60      0.35     0.25
cassette            0.10    0.08     0.15      0.35       0.70      0.60     1.00      0.20     0.18
digital_pcm         0.05    0.05     0.08      0.15       0.25      0.35     0.20      1.00     0.55
mp3_lossy           0.05    0.05     0.08      0.12       0.20      0.25     0.18      0.55     1.00
```

**Nutzung bei Cross-Material-Init**:

1. Sortiere Materialien nach Ähnlichkeit absteigend
2. Wähle das ähnlichste Material mit ≥ 10 GP-Beobachtungen
3. Übernimm dessen Kernel-Lengthscale × `(1 / similarity)` (= höhere Unsicherheit bei geringerer Ähnlichkeit)
4. Übernimm Signal-Varianz × `similarity` (= gedämpfter Prior bei geringerer Ähnlichkeit)
5. Bei `similarity < 0.3` → kein Transfer, nur GP-Default-Priors (uninformativ)

### ML-Failure-Degradations-Kaskade (v9.10.123)

Wenn ein ML-Plugin nicht geladen werden kann (OOM, korruptes Modell, ONNX-Fehler), **muss** die Pipeline graceful degradieren statt abzubrechen:

| Failure | Primär-Fallback | Sekundär-Fallback |
| --- | --- | --- |
| DeepFilterNet OOM | OMLSA/IMCRA (§4.5 Spec 04) | Spectral-Gating (Dry-Signal wenn SNR > 35 dB) |
| MDX23C Stem-Sep OOM | NMF-β-Separation (sklearn, β=Itakura-Saito; sdB ≥ 5 Proxy-SDR-Check) | HPSS (librosa.effects.hpss, tertiärer Fallback) |
| AudioSR OOM | Harmonische Oberton-Synthese + PGHI-Phasenrekonstruktion | Spectral-Band-Replication (SBR) |
| MP-SENet OOM (phase_43, ML-De-Esser-Kontext) | OMLSA/IMCRA DSP (Cohen & Berdugo 2002; §4.4) | Bypass (phase_43 Phase-Skip) |
| CREPE Pitch-Track | pYIN (Mauch & Dixon 2014) | YIN (de Cheveigné & Kawahara 2002) |
| MertPlugin OOM | DSP-Analyse: F0+Harmonizität+SpektralFlux-Kohärenz (besser als MFCC) | Bypass (HPI ohne MERT-Anteil) |

**Invariante**: Kein ML-Failure darf die Pipeline vollständig abbrechen. Jede Phase **muss** einen DSP-Fallback haben (§4.4 Spec 04). Der Fallback wird in `RestorationResult.metadata["ml_fallbacks_used"]` protokolliert.

### §2.47 Adaptions-Erweiterungen (v9.11.0)

Vier neue Intelligence-Hebel ergänzen die Adaptions-Kaskade. Sie sind **nach** dem GP-Optimizer aktiv und erhöhen die perceptuelle Präzision ohne neue ML-Modelle.

#### Hebel 1 — Salience-aware PhaseSkipping (`_salience_adjusted_severity`)

`_apply_phase_skipping` in UV3 liest **keine rohe `DefectScore.severity`** mehr direkt, sondern ruft `_salience_adjusted_severity(defect_type)` auf:

```python
def _salience_adjusted_severity(defect_type: str) -> float:
    ds = defect_result.scores.get(defect_type)
    if ds is None:
        return 0.0
    sev = float(ds.severity)               # ERB-adjustiert durch PerceptualSalienceEstimator
    meta = getattr(ds, "metadata", {}) or {}
    n_masked   = int(meta.get("n_masked_events", 0))
    n_salient  = int(meta.get("n_salient_events", 0))
    if n_masked >= 3 and n_salient == 0:   # vollständig ERB-maskiert
        sev *= 0.5                          # zusätzlich -50 % → Phase meist inaktiv
    return sev
```

**Rationale**: Defekte, die durch simultane Maskierung unhörbar sind (ERB-Maskierungskurve), sollen keine Phase einschalten — §0 Minimal-Intervention. Das ERA-Flag `n_masked_events`/`n_salient_events` im `DefectScore.metadata` stammt aus dem `PerceptualSalienceEstimator` (§2.47 Schritt 5).

**Invariante**: Eine Phase, die durch reine Severity-Kalkulation aktiviert würde, aber nur vollständig maskierte Defekte adressiert, wird übersprungen (`_skip_phase()`) — kein Klangschaden durch unnötige Verarbeitung.

#### Hebel 2 — SGMSE+ Tier-0 in `phase_03_denoise` (Richter et al. 2022)

Score-based Diffusion-Denoising als **erster** Processing-Pfad vor dem bisherigen ML-Hybrid-Pfad:

```text
Tier 0  SGMSE+ (diffusion)    — Bedingungen: quality_mode ∈ {quality, maximum}
                                              + (vocal_genre OR panns_singing_confidence ≥ 0.30)
                                              + NOT digital (cd_digital, dat, minidisc)
                                              + NOT use_lightweight
Tier 1  ML-Hybrid (DeepFilterNet/MP-SENet + OMLSA)  — bisheriger Hauptpfad
Tier 2  OMLSA/IMCRA                                 — DSP-Fallback
Tier 3  Spectral-Gating                             — letzter Ausweg
```

**Tier-0-Auslöser**: Vokalmusik profitiert überproportional von Diffusion-Denoising, weil SGMSE+ die Lernverteilung natürlicher Sprachlaute als implizites Prior nutzt — Formanttreue bleibt erhalten. Bei nicht-vokalen Genres oder Digital-Material überwiegt das ML-Hybrid-Verfahren (deterministischer, kein Over-Smoothing).

**Metadata-Markierung**: `phase_result.metadata["sgmse_plus_tier0_applied"] = True` bei Tier-0-Nutzung.

#### Hebel 3 — PhaseConductor (inter-phase adaptive Strength)

Vollständiger Workflow und Invarianten: siehe §2.52 dieses Dokuments.

**Einbettungspunkt**: UV3 `_execute_pipeline`, sequentiell nach §2.31a MidCalibrate-Block.

#### Hebel 4 — Carrier-Formant-Decay-Inversion in `phase_42` (Stage 0.5)

Analoge Tonträger dämpfen charakteristisch den Formantbereich durch mechanische und magnetische Transfer-Verluste:

```python
def _restore_carrier_formant_decay(audio, sr, material_type):
    """Stage 0.5: Invertiert träger-spezifische F1–F4-Unterdrückung via zero-phase Bell-EQ."""
```

Trägertypische Bell-EQ-Profile (Gain in dB, Zentrum-Hz, Q):

| Material | F1-Boost | F2-Boost | F3-Boost | F4-Boost |
| --- | --- | --- | --- | --- |
| vinyl | +0.8 dB @ 800 Hz, Q=2.0 | +1.2 dB @ 1800 Hz, Q=2.5 | +0.6 dB @ 3200 Hz, Q=3.0 | +0.4 dB @ 4500 Hz, Q=3.5 |
| reel_tape | +1.0 dB @ 750 Hz, Q=1.8 | +1.5 dB @ 1700 Hz, Q=2.2 | +0.8 dB @ 3000 Hz, Q=2.8 | — |
| tape | +0.6 dB @ 800 Hz, Q=2.0 | +1.0 dB @ 1800 Hz, Q=2.5 | +0.5 dB @ 3200 Hz, Q=3.0 | — |
| shellac | +2.0 dB @ 600 Hz, Q=1.5 | +3.0 dB @ 1500 Hz, Q=2.0 | +1.5 dB @ 2800 Hz, Q=2.5 | — |
| minidisc | +0.4 dB @ 850 Hz, Q=2.5 | +0.8 dB @ 1900 Hz, Q=3.0 | +0.3 dB @ 3400 Hz, Q=3.5 | — |
| cd_digital | passthrough (kein Formant-Decay) | — | — | — |

**Implementierung**: `scipy.signal.filtfilt` (zero-phase, IIR-Biquad-Peaking) pro Formant. Kein Phasen-Artefakt, kein Pre-Ringing. Stage 0.5 läuft **vor** Stage 1 (Pitch-Korrektur) in `_enhance_channel(audio, sr, material_type=material)`.

> Kreuzreferenz: §2.52 (PhaseConductor), §2.46 (Carrier-Chain-Inversion), Spec 06 §7.4

## §2.48 [RELEASE_MUST] Kumulative-Phasen-Interaktions-Guard (v9.10.123)

Einzelne Phasen können isoliert korrekt arbeiten, aber in Kombination destruktive Effekte erzeugen (z.B. De-Noise + De-Reverb entfernen gemeinsam mehr Raumklang als beabsichtigt).

### Kumulative P1/P2-Drift-Messung

Nach jeder Phase wird die **kumulative** Gesamt-Regression der P1/P2-Goals (Natürlichkeit, Authentizität, TonalCenter, Timbre, Artikulation) gemessen — nicht nur die Delta-Regression der Einzelphase.

```python
# In _execute_pipeline(), nach jeder Phase:
goals_now = musical_goals_checker.evaluate(current_audio, sr)
cumulative_drift = {g: goals_now[g] - goals_pre_pipeline[g] for g in P1_P2_GOALS}
if any(drift < -0.05 for drift in cumulative_drift.values()):
    current_audio = best_checkpoint_audio  # Rollback
    logger.warning("phase=%s cumulative_drift=%s → rollback", phase_id, cumulative_drift)
```

### Kritische Interaktions-Paare (bekannte destruktive Kombinationen)

| Paar | Risiko | Guard |
| --- | --- | --- |
| `phase_03 (De-Hiss) + phase_20/49 (De-Reverb)` | Kumulative Raumklang-Entfernung | Nach De-Reverb: Natürlichkeit ≥ pre_pipeline − 0.03 |
| `phase_29 (NR) + phase_03 (De-Hiss)` | Over-Denoising | Nach zweiter NR-Phase: Rauschboden ≥ Material-Ziel (§0a) |
| `phase_35 (Multiband-Compression) + phase_40 (LUFS-Norm.)` | Dynamik-Verlust | Nach LUFS: MikroDynamik ≥ pre_pipeline − 0.04 |
| `phase_07 (Harmonic-Restoration) + phase_42 (Vocal-AI)` | Frequenz-Doppelung | Nach Vocal-AI: Spectral-Flatness-Check |
| `phase_23/24 (Super-Resolution) + phase_03 (De-Hiss)` | Künstliche Obertöne entrauscht | Super-Res immer VOR De-Hiss (Reihenfolge-Invariante) |

### Kumulative STFT-Phasenkohärenz

Mehrfache STFT→Modifikation→ISTFT erzeugt akkumulierte Phasenfehler (Gruppenlaufzeit-Deviation, Phase-Smearing bei Transienten). Dies ist kein Goal-messbarer Effekt, sondern ein rein technischer Fehler.

**Prüfung**: Nach ≥ 3 STFT-basierten Phasen in Folge:

- `group_delay_deviation = max(|τ_current(f) - τ_original(f)|)` über alle Frequenz-Bins
- Schwellwert: ≤ 5 ms (entspricht ~240 Samples bei 48 kHz)
  - Begründung v9.10.127: 2 ms war unrealistisch. Standard-2048-Punkt-STFT bei 48 kHz hat bereits 42,6 ms Fensterlänge (10,7 ms Hop). Spektralsubtraktions-Filter verschieben pro-Bin-Phase lokal 3–8 ms ohne hörbare Artefakte. Ab 5 ms liegt ein echtes Phase-Distorsions-Problem vor (typisch: unabhängige L/R-IIR-Filter oder falsch kaskadierte STFT-Ketten).
- Überschreitung → letzte STFT-Phase rollback, Alternative ohne STFT versuchen (z.B. PGHI statt GriffinLim, Zero-Phase-Filterung statt STFT-Modifikation)

**Betroffene Phasen** (STFT-basiert): phase_03 (De-Hiss), phase_07 (Harmonic), phase_20/49 (De-Reverb), phase_23/24 (Super-Resolution), phase_29 (NR), phase_35 (Multiband-Comp)

### Checkpoint-Verwaltung

- `best_checkpoint`: Audio-Snapshot + Goal-Scores nach der bisherigen besten Phase
- Bei Rollback: Phase-Skip protokollieren in `RestorationResult.metadata["interaction_rollbacks"]`
- Nach Rollback: nächste Phase erhält `best_checkpoint`-Audio, nicht das degradierte
- Max 2 aufeinanderfolgende Rollbacks → Pipeline-Stop, Export auf `best_checkpoint`

### Phasen-Reihenfolge-Optimierung

CAUSE_TO_PHASES wählt **welche** Phasen aktiv sind. Die **Reihenfolge** der aktiven Phasen folgt der **Carrier-Chain-Inversions-Logik** (§2.46):

1. **ADC-Stufe**: DC-Offset, Quantisierungs-Artefakte (phase_01, phase_31)
2. **Playback-Stufe**: RIAA-Inverse, Azimuth, Wow/Flutter, Speed-Korrektur (phase_06, phase_09, phase_10)
3. **Alterungs-Stufe**: Click/Pop, Dropout, Knistern (phase_02, phase_04, phase_05, phase_11)
4. **Carrier-Encoding-Stufe (subtraktiv)**: NR, De-Hiss, De-Reverb (phase_03, phase_29, phase_20/49)
5. **Carrier-Encoding-Stufe (additiv)**: Super-Resolution, Harmonic-Restoration, Bandwidth-Extension (phase_23, phase_24, phase_07)
6. **Enhancement-Stufe**: Vocal-AI, Stem-Sep, Dynamics, EQ, LUFS (phase_42, phase_35, phase_40)

**Invariante**: Subtraktive Phasen VOR additiven — sonst werden rekonstruierte Obertöne sofort wieder entrauscht.

> Kreuzreferenz: §2.29d (P1/P2 = hart), §2.45 (perceptual_delta), §2.44 (HPI)

## §2.48a [RELEASE_MUST] Phase-Typ-Ontologie — Architektur-Inversion (v9.11.0)

### Prinzip

**Guards dürfen nur feuern, wenn ihre Messvoraussetzung strukturell erfüllt ist** — abgeleitet aus dem intrinsischen Operationstyp der Phase, nicht aus Ausnahmelisten.

Das bisherige Muster (Ausnahmeliste) ist nicht skalierbar: Jede neue Phase braucht manuellen Eintrag in `_RESTORATIVE_PHASE_IDS`, `STFT_PHASES`, `PHASE_GOAL_EXCLUSIONS`. Fehlt ein Eintrag, feuert der Guard falsch → Rollback auf verbessertes Audio.

**Lösung**: `backend/core/phase_ontology.py` definiert `PhaseOperationType` als Enum. Jeder Guard konsultiert den Typ und entscheidet strukturell, ob seine Messung valide ist.

### Phase-Operationstypen (normativ)

| Typ | Beschreibung | Beispiele |
| --- | --- | --- |
| `SUBTRACTIVE` | Entfernt Rauschen/Artefakte | phase_03, phase_09, phase_18, phase_20, phase_27, phase_28, phase_29, phase_49, phase_50 |
| `ADDITIVE` | Fügt neue Signalkomponenten hinzu | phase_06, phase_07, phase_21, phase_22, phase_37, phase_38, phase_39 |
| `CORRECTIVE` | Korrigiert spektrale/zeitliche Eigenschaften | phase_04, phase_12, phase_14, phase_25, phase_30, phase_31, phase_41 |
| `ML_GENERATIVE` | ML-Diffusion/Flow-Matching (kein STFT-kohärenter Ausgang) | phase_42, phase_55, phase_36, phase_64 |
| `DYNAMICS` | Hüllkurven-Verarbeitung | phase_08, phase_10, phase_11, phase_17, phase_19, phase_26, phase_35, phase_40, phase_47 |
| `ANALYSIS_ONLY` | Kein Audio-Output | phase_53 |
| `ENHANCEMENT` | Mix/nicht eindeutig | phase_13, phase_32, phase_46, phase_48, phase_58 |

### Guard-Applicability-Matrix (normativ)

| Guard | Valide für Typen | Invalide für Typen | Wissenschaftliche Grundlage |
| --- | --- | --- | --- |
| **Noise-Texture-Check** (§2.49) | `SUBTRACTIVE` | alle anderen | Schwarz & Grill 2004: BW-Erweiterung verändert Spektral-Tilt intentional |
| **Pre-Echo-Detektor** (§2.49) | `DYNAMICS`, `ENHANCEMENT` | `SUBTRACTIVE`, `ADDITIVE`, `CORRECTIVE`, `ML_GENERATIVE` | Brandenburg & Johnston 1994: Pre-Echo ist ausschließlich Transform-Coding-Artefakt; Residual subtraktiver Phasen ≠ Prä-Transient-Energie |
| **GDD-Check** (§2.48) | `SUBTRACTIVE`, `DYNAMICS`, `ENHANCEMENT` | `ML_GENERATIVE`, `ADDITIVE` | Richter et al. (SGMSE+, TASLP 2022): Diffusionsausgang nicht STFT-phasenkohärent; Synthese erzeugt neue Bins mit eigener Phase |
| **Baseline-Capping** §2.29c | `SUBTRACTIVE` | alle anderen | ITU-R BS.1387 §4.2: Rauschresidual ist kein Artefakt; defekt-inflationierte Baseline ist strukturelles Merkmal subtraktiver Phasen |
| **P1/P2-Drift-Check** (§2.48) | alle außer `ANALYSIS_ONLY` | `ANALYSIS_ONLY` | Audio unverändert → Drift trivial 0.0 |

### Implementierung

```python
# backend/core/phase_ontology.py — normatives Register
from backend.core.phase_ontology import get_phase_type, GDD_VALID_TYPES, NOISE_TEXTURE_VALID_TYPES, PRE_ECHO_VALID_TYPES

# Guard-Entscheidung (Inversion):
phase_type = get_phase_type(phase_id)
if phase_type in NOISE_TEXTURE_VALID_TYPES:      # nur SUBTRACTIVE
    check_noise_texture(...)
if phase_type in PRE_ECHO_VALID_TYPES:           # nur DYNAMICS + ENHANCEMENT
    check_pre_echo(...)
if phase_type in GDD_VALID_TYPES:                # nicht ML_GENERATIVE, nicht ADDITIVE
    check_group_delay(...)
```

**Invariante**: `phase_ontology.py` IST die Wahrheit. Alle Guards leiten ihre Exemptionen ab — keine doppelte Pflege von Ausnahmelisten.

> Implementierung: `backend/core/phase_ontology.py` — `PhaseOperationType`, `get_phase_type()`, Guard-Applicability-Sets.
> Konsumenten: `artifact_freedom_gate.py`, `cumulative_interaction_guard.py`, `per_phase_musical_goals_gate.py`.

## §2.49 [RELEASE_MUST] Artefakt-Freiheits-Gate (v9.10.123)

Dediziertes Gate für **Artefakt-Erkennung** — unabhängig von den 14 Musical Goals. Eine Phase kann alle Goals bestehen und trotzdem hörbare Artefakte erzeugen.

### Geprüfte Artefakte

| Artefakt | Erkennungsmethode | Schwellwert |
| --- | --- | --- |
| Musical Noise | Spectral-Variance in Stille-Segmenten: isolierte tonale Peaks (> 12 dB über Nachbarn) in Stille/Pausen | 0 Events |
| Pre-Echo | Transient-Onset-Analyse: Energie in 5-ms-Fenster vor Attack ≤ −40 dB relativ zum Attack-Peak | 0 Events |
| Spectral Holes | Bandbreiten-Kontinuitäts-Check: keine Energielücken > 200 Hz im erwarteten Passband (SourceFidelity BW) | 0 Holes |
| Phase-Cancellation | M/S-Korrelation nach Stereo-Processing: `correlation(M, S) ≥ 0.3` (Mono-Kompatibilität) | ≥ 0.3 |
| Metallic Ringing | CQT-Peak-Detection: isolierte resonante Peaks > 6 dB über Nachbar-Bins, Dauer > 50 ms | 0 Events |

### Material-adaptive Schwellwert-Skalierung (v9.10.123)

Feste Schwellwerte führen zu Fehlalarmen bei historischem Material (z.B. Schellack-Oberflächen-Rauschen als "Musical Noise" fehlklassifiziert) oder zu Durchlassfehlern bei Digital-Material. Deshalb werden die Artefakt-Schwellwerte **material-adaptiv** skaliert:

| Artefakt | Digital/CD | Tape | Vinyl | Shellac/Wax |
| --- | --- | --- | --- | --- |
| Musical Noise (Peak-dB) | > 12 dB | > 15 dB | > 18 dB | > 22 dB |
| Pre-Echo (Rel. Attack) | ≤ −40 dB | ≤ −35 dB | ≤ −30 dB | ≤ −25 dB |
| Spectral Holes (Lücke) | > 200 Hz | > 300 Hz | > 400 Hz | > 600 Hz |
| Phase-Cancellation (mono_compat) | ≥ 0.30 | ≥ 0.20 | ≥ 0.20 | ≥ 0.15 |
| Metallic Ringing (Peak-dB) | > 6 dB | > 8 dB | > 10 dB | > 14 dB |

**Logik**: Historische Träger haben inhärent höhere Artefakt-Pegel im Eingangssignal. Was bei einer CD ein klarer Verarbeitungsfehler ist (Musical-Noise-Peak +12 dB), ist bei Shellac Teil des Trägerprofils. Die Erkennung muss nur **neue, durch Verarbeitung eingeführte** Artefakte finden — nicht die vorhandenen des Trägers.

**Direktionalitätspflicht für Musical-Noise-Detektor** (v9.10.125): Subtractive Phasen (Surface-Noise-Profiling, Denoise, Click-Removal) erzeugen ein Residual `restored − orig` dessen Spektrum die **entfernten** Artefakte spiegelt — nicht neu hinzugefügte. Die Spektralpeaks im Residual sind korrekte Entfernungen, keine Artefakte. Implementierungspflicht:

```python
# Nur flaggen wenn restored_spectrum[j] > orig_spectrum[j] × 1.05
# (Energie wurde ADDIERT, nicht subtrahiert)
if rest_spectrum[j] <= orig_spectrum[j] * 1.05:
    continue  # subtractive action — correct removal, not an artefact
```

Ohne diese Prüfung: Surface-Noise-Profiling erzeugt 50 False-Positive-Artefakte → `artifact_freedom=0.000` → Rollback-Loop → Pipeline-Blockade.

**Phase-Cancellation Detektor — Präzisierungen (v9.10.127)**:

Der Phase-Cancellation-Detektor vergleicht im per-phase-Modus die Stereo-Metrik **vor und nach** der Phase (Delta-Check). Folgende Regeln sind **normativ verbindlich**:

1. **Anti-Korrelation-Schwelle**: `lr_corr < −0.20` (nicht `< 0.0`). Werte zwischen 0 und −0.20 entstehen durch STFT-Window-Misalignment, Gate-Transient-Asymmetrie und normale Verarbeitungsunterschiede — sie sind **nicht hörbar** und dürfen nicht als Phase-Cancellation gezählt werden.

2. **Delta-Guard**: Eine Phase wird nur geflaggt, wenn `orig_compat − restored_compat > 0.10`. Kleinere Asymmetrien (< 0.10) durch DSP-Implementierungsdetails (Filter-Rounding, Overlap-Grenzen) sind technische Artefakte, keine perceptuell relevanten Stereo-Probleme.

3. **Near-Mono-Guard**: Wenn das Quellmaterial quasi-mono ist (`orig_compat > 0.65`) UND die verarbeitete Version noch moderat mono-kompatibel ist (`restored_compat > 0.40`), ist die Abweichung durch unabhängige Kanalverarbeitung (Noise-Gate Transient, Dropout-Füllung) **nicht hörbar** — skip. Ausnahme: Echter Stereo-Kollaps (`restored_compat ≤ 0.40`) wird trotzdem geflaggt.

4. **Stereo-Collapse-Guard**: Wenn ein Kanal einen RMS-Abfall > 40 dB gegenüber dem Original-Input verzeichnet (z. B. R-Kanal von −18 dBFS auf −∞), wird **ein Artefakt** erzeugt und der Frame-Loop wird übersprungen (globaler Kollaps überwiegt Frame-Level-Analyse). Voraussetzung: Originales Signal hatte RMS > 1e-4 (kein stiller Quellkanal).

**Implementierung**: `artifact_freedom_gate.py → _detect_phase_cancellation()`

**Implementierung**: `artifact_thresholds = BASE_THRESHOLDS × material_tolerance_factor[material]`. Der `material_tolerance_factor` kommt aus dem MediumDetector-Ergebnis (§2.47 Adaptions-Kaskade Schritt 1).

**Selbstkalibrierung**: Bei den ersten 3 Verarbeitungen eines neuen Material-Typs werden Artefakt-Schwellwerte konservativ (= strenger) angesetzt. Nach 3 erfolgreichen Verarbeitungen (artifact_freedom ≥ 0.98): Schwellwerte auf material-adaptive Normalwerte entspannen.

### Rauschtextur-Kohärenz (Restoration-Modus)

Unabhängig von den 5 Artefakttypen: Die **spektrale Form** des Restrauschens (Noise-Floor-Shape) muss dem originalen Trägerprofil entsprechen. Aggressive Denoising hinterlässt oft ein Restrauschen mit falscher spektraler Färbung.

**Messung**: In Stille-Segmenten (≥ 200 ms, RMS < −50 dBFS):

1. Input-Noise-Profile: Spectral-Tilt (lineare Regression über Log-Magnitude-Spektrum)
2. Output-Noise-Profile: gleiche Berechnung
3. `tilt_deviation = |tilt_output - tilt_input|` in dB/Oktave

**Schwellwerte**:

| Abweichung | Aktion |
| --- | --- |
| ≤ 3 dB/Oktave | OK — Restrauschen hat natürliche Textur |
| 3–6 dB/Oktave | Warnung — `artifact_freedom` −0.05 Penalty |
| > 6 dB/Oktave | Rollback auf letzte NR-Phase — unnatürliche Rauschtextur |

**Typische Fehlerbilder**:

- Vinyl-Denoising → weißes Rauschen (statt rosa-Tilt ≈ −3 dB/Oktave): Over-Denoising der tiefen Frequenzen
- Tape-NR → tonales Rauschen (isolierte NR-Residuen): Musical-Noise-Variante
- Shellac → zu "sauberes" Restrauschen: Ambient-Charakter verloren

### Score-Berechnung

```python
artifact_freedom = 1.0 - (weighted_artifact_count / max_tolerance)
artifact_freedom = np.clip(artifact_freedom, 0.0, 1.0)
```

Gewichtung: Musical Noise = 1.0, Pre-Echo = 0.8, Spectral Holes = 0.6, Phase-Cancellation = 1.0, Metallic Ringing = 0.9

**Perzeptuelle Salienz-Gewichtung**: Die obigen Gewichte werden zusätzlich nach perzeptueller Salienz skaliert:

- **Frequenz**: Artefakte im Bereich 200–5000 Hz (höchste Hörempfindlichkeit, ISO 226) erhalten Faktor 1.0; unter 200 Hz oder über 5000 Hz → Faktor 0.5; über 12 kHz → Faktor 0.2
- **Kontext**: Artefakte in Stille/Pausen-Segmenten (RMS < −40 dBFS) erhalten Faktor 1.5 (stärker hörbar); in Tutti-Passagen (RMS > −20 dBFS) → Faktor 0.5 (maskiert)
- **Dauer**: Artefakte > 100 ms erhalten Faktor 1.5; < 20 ms → Faktor 0.5
- Effektiver Score: `salience_weighted_artifact_count = Σ(type_weight × freq_factor × context_factor × duration_factor)`

### Integration

- **Im HPI**: `artifact_freedom` fließt als Multiplikator in beide HPI-Formeln ein (§2.44)
- **Phase-Level**: Nach jeder Phase prüfen — bei `artifact_freedom < 0.95` → Rollback auf `best_artifact_free_checkpoint`
- **Export-Gate**: `artifact_freedom < 0.95` → kein Export, auch wenn alle 14 Goals bestanden
- **Protokollierung**: `RestorationResult.metadata["artifact_freedom"]` = Score + Detail-Report (detected_artifacts: list)

### §2.49 Finaler Score — Berechnungsregel (v9.10.126)

**`_artifact_freedom_score` = Minimum aller per-Phase-Scores aller akzeptierten Phasen.**

FALSCH (und verboten): `artifact_gate.evaluate(pre_pipeline_audio, pipeline_output)` — jede echte Restaurierung erzeugt dadurch zwangsläufig `artifact_freedom=0.000`, weil intentionale Signalveränderungen (Rauschen entfernen, Bandbreite erweitern) im Vollvergleich als Artefakte erscheinen.

RICHTIG: Per-Phase-Minimum über alle Phasen, bei denen der Gate-Check durchgeführt wurde (`_min_per_phase_afg_score`). Phasen, die ge-rollt-back wurden, fließen nicht ein.

### §2.49b [RELEASE_MUST] Post-Pipeline Kumulativer Stereo-Collapse-Guard (v9.10.126)

Per-Phase-δ-Guards fangen nur Single-Phase-Kollapsen (> 40 dB in einer Phase). Kumulativer Stereo-Drift — bei dem 4 Stereo-Phasen jeweils 6–8 dB beitragen — bleibt unsichtbar. Lösung: Post-Pipeline-Vergleich gegen Pre-Pipeline-Baseline.

**Invariante** (direkt nach Phase-Loop, vor `_pmgg_log_entries`-Zuweisung):

```python
if current_audio.ndim == 2 and current_audio.shape[0] == 2:
    cu_imb = abs(L/R_dB(current_audio))      # Imbalance Pipeline-Ausgang
    pp_imb = abs(L/R_dB(afg_pre_pipeline))   # Imbalance Pipeline-Eingang
    if cu_imb > 20.0 and pp_imb < 6.0:       # kumulativer Kollaps
        # Rollback-Kaskade:
        # 1. best_clean_checkpoint — sofern selbst nicht kollabiert (> 20 dB prüfen)
        # 2. afg_pre_pipeline_audio (Primum non nocere)
        current_audio = recovery
```

Schwellwerte: Ausgang-Imbalance > 20 dB; Eingang-Imbalance < 6 dB (Kollaps neu durch Pipeline eingeführt).

### §2.44/§2.49 HPI-Rollback-Checkpoint Stereo-Health-Validation (v9.10.126)

Bevor `_hpi_best_rollback_audio` als Rollback-Ziel verwendet wird: L/R-Imbalance prüfen.

- Checkpoint-Imbalance > 20 dB UND Input war ausgeglichen (< 6 dB) → Checkpoint verwerfen
- Fallback: `original_audio_for_goals` (Primum non nocere)

Ohne diese Prüfung restauriert der HPI-Rollback ein stereo-zerstörtes Signal.

> Kreuzreferenz: §2.44 HPI (artifact_freedom als Multiplikator), §2.48 (Interaktions-Guard), §2.45 (perceptual_delta)

---

## §2.51 [RELEASE_MUST] Stereo-Kohärenz-Invariante für Phasen (v9.10.127)

### Motivation

Phasen, die L- und R-Kanal **unabhängig** verarbeiten (je Kanal eigener Denoiser, Gate, Kompressor, spektrale Reparatur), können in 2–3 Frames pro Phase `mono_compat < 0.20` erzeugen. Ursache: Minimale Unterschiede in Filterauflösung, Gate-Timing oder Spektralschätzung zwischen den Kanälen. Das §2.49-Gate flaggt diese Frames zu Recht — die Phasen verstoßen gegen §0 (Primum non nocere), weil sie Stereo-Kompatibilität verschlechtern.

Die Lösung ist **nicht** weitere Gate-Relaxation, sondern korrekte Implementierung der betroffenen Phasen.

### Normative Anforderung

Jede Phase, die auf Stereo-Audio operiert und den Signalpegel modifiziert, **MUSS** eine der folgenden zwei Verarbeitungsstrategien verwenden:

**Option A — M/S-Domain (bevorzugt für spektrale Operationen)**:

```
Mid = (L + R) / 2          # Summen-Kanal: Mono-kompatibler Inhalt
Side = (L - R) / 2         # Differenz-Kanal: Stereo-Breite

→  Verarbeite Mid mit voller Algorithmus-Stärke
→  Verarbeite Side mit reduzierter oder keiner Stärke (bewahre Stereo-Breite)
→  Rekonstruiere: L = Mid + Side,  R = Mid - Side
→  Clip: L = np.clip(L, -1.0, 1.0),  R = np.clip(R, -1.0, 1.0)
```

**Wann A**: Harmonische Restaurierung, spektrale Reparatur, Sprach-Enhancement, Dehum, EQ, Sättigungseffekte — immer wenn die Phasen-Verarbeitung tonal auf dem Informations-Inhalt arbeitet.

**Option B — Linked Stereo (für dynamische Verarbeitung)**:

```
signal_level = max(RMS(L), RMS(R))   # oder: np.sqrt(RMS(L)² + RMS(R)²)
gain = compute_gain(signal_level)     # Gain-Kurve einmalig berechnen
L_out = apply_gain(L, gain)           # Gleiches Gain für beide Kanäle
R_out = apply_gain(R, gain)
```

**Wann B**: Noise-Gate (Gate öffnet wenn L ODER R über Threshold), Dropout-Repair (synchrone Erkennung + kohärente Füllung), Multiband-Kompression, Transient-Shaper — immer wenn die Entscheidung (öffnen/schließen, verstärken/dämpfen) von der gemeinsamen Energie-Hüllkurve abhängt.

### Betroffene Phasen (Pflicht-Umsetzung)

| Phase | Problem | Strategie |
| --- | --- | --- |
| `phase_07_harmonic_restoration` | Harmonics separat auf L/R → Anti-Phase-Transients in 2–3 Frames | **Option A** (M/S) — Harmonics auf Mid, Side unverändert |
| `phase_18_noise_gate` | Gate öffnet/schließt für L und R unabhängig → Anti-Phase-Gate-Transients | **Option B** (Linked) — `max(L_rms, R_rms) > threshold → both open` |
| `phase_23_spectral_repair` | Spektrale Lücken auf L/R separat erzeugt minimale Anti-Phasigkeit | **Option A** (M/S) — Reparatur auf Mid, Side minimal bearbeiten |
| `phase_24_dropout_repair` | L/R-Dropouts erkannt und gefüllt unabhängig | **Option B** (Linked) — Dropout-Grenze ist der Eintritt BEIDER Kanäle unter Schwelle; Füllung kohärent |
| `phase_35_multiband_compression` | Kompressor berechnet Gain für L und R separat → L/R-Gain-Differenz in Transienten | **Option B** (Linked) — Gain-Berechnung auf Summen-RMS (`√(L²+R²)/√2`), gleicher Gain auf beide |

### Downstream-Auswirkungen auf Metriken

| Metrik | Auswirkung | Korrekturbedarf |
| --- | --- | --- |
| **Brillanz** | M/S in `phase_07`: Harmonics nur auf Mid → weniger HF-Energie im Side-Kanal. Brillanz-Schwellwert ≥ 0.78 unverändert, aber `BrillanzMetric` muss Stereo-Mid nicht Side-Anteil messen | Kein Schwellwert-Änderungsbedarf; Metrik misst bereits Gesamtspektrum |
| **Raumtiefe** | Linked Stereo in `phase_35`: Einheitlicher Gain erhält Side-kanal besser → Raumtiefe kann leicht steigen | Kein Korrekturbedarf (positive Auswirkung) |
| **SepFidelity** | Kohärente L/R-Füllung in `phase_24`: Dropout-Füllung ist konsistenter mit Stereo-Bild → SepFidelity tendenziell verbessert | Kein Korrekturbedarf |
| **Groove** | Linked Gate in `phase_18`: Transiente Energie wird kohärent erhalten (kein halbes Gate-Öffnen) → Groove-Presenz besser | Kein Korrekturbedarf (positive Auswirkung) |
| **§2.49 Phase-Cancellation** | Nach Implementierung: 5 Phasen passieren Gate ohne Rollback → `_min_per_phase_afg_score` bleibt 1.0 | Kein Korrekturbedarf; Gate-Schwellwerte unverändert |
| **PMGG Wärme §9.7.14** | Wärme nutzt harmonische Oberton-Ratio. M/S ändert Side-Obertöne nicht → Wärme-Proxy stabil | Kein Korrekturbedarf |

### Invariante

Kein Accept-Checkpoint darf `mono_compat < 0.20` in mehr als 5 % der Frames haben (außer das Quellmaterial hatte bereits diese Mono-Inkompatibilität — §2.50 SourceMaterialBaseline).

**Implementierungsprüfung**: `_detect_phase_cancellation()` im §2.49-Gate ist der objective Prüfer. Nach Umsetzung der obigen Phasen dürfen phase_07, phase_18, phase_23, phase_24, phase_35 keine §2.49-Rollbacks mehr auslösen.

> Kreuzreferenz: §2.49 (ArtifactFreedomGate), §2.50 (SourceMaterialBaseline), §7.4 Spec06 (PhaseInterface)

---

## §2.52 [RELEASE_MUST] PhaseConductor — Inter-Phase Adaptive Feedback (v9.11.0)

### Überblick

`PhaseConductor` ist ein **rein DSP-basierter inter-phase Feedback-Controller**. Er misst nach jeder Phase den verbleibenden Signal-Zustand und leitet daraus eine adaptive `strength`-Empfehlung für die **nächste** Phase ab. Kein ML, kein Netzwerkzugriff, kein I/O.

- **Singleton**: `get_phase_conductor()` in `backend/core/phase_conductor.py`; thread-safe (double-checked locking)
- **Session-Scope**: `conductor.reset()` zu Beginn jedes Songs in UV3 `_execute_pipeline`
- **Advisory-only**: PMGG-Strength hat immer Vorrang; alle Empfehlungen sind Hinweise, keine Befehle

### 4D State-Vector

| Dimension | Beschreibung | Normierung | Messzeit (48 kHz, 3 min) |
| --- | --- | --- | --- |
| `noise_floor_db` | 5. Perzentil der Leistungsdichteschätzung (PSD) | dBFS ≤ 0 | < 5 ms |
| `hf_energy_ratio` | Energie 8–24 kHz / Breitband (0–24 kHz) | [0, 1] | < 5 ms |
| `transient_density` | Onset-Rate (librosa.onset.onset_detect) [Events/s] | roh; as_vec() → /20 | < 20 ms |
| `harmonic_coherence` | Autocorrelation-Peak-Ratio auf Mid-Kanal | [0, 1] | < 15 ms |

Gesamt `measure_state()` < **50 ms** pro Aufruf auch für 3 Minuten Audio.

### Referenzgitter und Nearest-Neighbor-Empfehlung

Pro Material gibt es ein vorberechnetes Referenzgitter aus (state_4d → optimal_strength)-Paaren:

```python
# Beispiel-Grid (werden zur Laufzeit nicht trainiert, sind hardcoded DSP-Messungen):
_REFERENCE_GRIDS: dict[str, list[tuple[PhaseState, float]]] = {
    "vinyl":    [...],   # 12 Referenzpunkte × (state_vec, optimal_strength)
    "reel_tape":[...],
    "tape":     [...],
    "shellac":  [...],
    "minidisc": [...],
    "cd_digital":[...],
}
```

**Nearest-Neighbor**: L2-Distanz auf normiertem State-Vektor (noise/−90, hf/1, transient/20, coherence/1). Bei `distance > 0.8` → Fallback auf `_DEFAULT_STRENGTH[phase_id]` (kein Over-Extrapolation).

### Workflow in `_execute_pipeline` (UV3)

```python
# 1. Init vor Phase-Loop
_conductor = get_phase_conductor()
_conductor.reset()
_conductor_strength_hints: dict[str, float] = {}

# 2. Nach jeder erfolgreichen Phase
_conductor.measure_state(current_audio, sr, phase_id=current_phase_id)

# 3. Look-Ahead für nächste Phase
if next_phase_id:
    rec = _conductor.recommend(next_phase_id, state=_conductor.last_state, material_type=material_type)
    if rec.strength_hint is not None:
        _conductor_strength_hints[next_phase_id] = rec.strength_hint

# 4. In _profiled_phase_call: hint injizieren (nur wenn strength nicht explizit gesetzt)
if phase_id in _conductor_strength_hints and "strength" not in explicit_kwargs:
    kwargs["strength"] = _conductor_strength_hints[phase_id]
```

### Invarianten

| Invariante | Beschreibung |
| --- | --- |
| `_NEVER_SKIP` | `frozenset({"phase_01", "phase_09", "phase_12", "phase_14", "phase_15"})` — diese Phasen erhalten nie `skip_recommended=True`, egal wie der State-Vektor aussieht |
| `_MIN_STRENGTH` | `{"phase_03": 0.35, "phase_09": 0.50, ...}` — Untergrenze für kritische Phasen; `recommended_strength = max(rec, _MIN_STRENGTH.get(phase_id, 0.0))` |
| Exception-Sicherheit | Jede Exception in `measure_state` oder `recommend` → `logger.debug(exc)`, Pipeline läuft **unverändert** weiter (kein Abbruch, kein Fehler-Propagation) |
| Kein §0-Verstoß | Wenn `recommended_strength < explicit_strength`: PMGG-Wert gewinnt; wenn `recommended_strength > explicit_strength`: PMGG-Wert gewinnt — ConductorHint beeinflusst nur **nicht explizit gesetzten** Strength |
| Keine ML-Abhängigkeit | Rein DSP, nur numpy + scipy; kein torch, kein ONNX, kein Remote-Call |

### Zusammenspiel mit §2.47 PhaseSkipper (Hebel 1 + Hebel 3 Synergie)

```text
DefectScanner → _salience_adjusted_severity() (Hebel 1)
     ↓ severity (ERB-gewichtet)
_apply_phase_skipping → Phase aktiv/inaktiv?
     ↓ wenn aktiv:
_conductor.recommend(next_phase_id, …) (Hebel 3)
     ↓ strength_hint
_profiled_phase_call → Phase läuft mit adaptiver Wetness
```

Hebel 1 entscheidet **ob** eine Phase läuft; Hebel 3 entscheidet **wie stark** sie läuft. Beide zusammen vermeiden Über- und Unter-Processing.

> Implementierung: `backend/core/phase_conductor.py`
> UV3-Integration: `backend/core/unified_restorer_v3.py` — `_execute_pipeline`, `_profiled_phase_call`
> Tests: `tests/unit/test_hebel_intelligence_levers.py` (Hebel 3: Tests 17–26, 32/32 grün)
