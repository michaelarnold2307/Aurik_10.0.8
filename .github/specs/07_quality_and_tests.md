# Aurik 9 — Spec 07: Qualitätsziele & Tests

> PQS-Metriken, AMRB-Benchmark, universelle Garantien, Test-Standards,
> E2E-Assertions, Performance-Budget.

---

## §8.1 Numerische Qualitätsgrenzen

### PQS-Metriken (`core/perceptual_quality_scorer.py`)

| Metrik | Hard-Fail-Minimum | Weltklasse-Ziel |
|---|---|---|
| PQS MOS | ≥ 3.8 | ≥ 4.5 |
| PQS NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD (dB) | ≤ 8.0 | ≤ 3.0 |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

### quality_estimate — normative Formel

```python
quality_estimate = 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4
# Clip: max(0.0, min(1.0, quality_estimate))
# VERBOTEN: quality_estimate * 1.15 als fixer Bonus
# E2E-Pflicht: result.quality_estimate >= 0.55 nach erfolgreicher Restaurierung
```

### Verbotene Metriken für Musikqualitätsbewertung

```
PESQ     # Telefonband 300–3400 Hz — strukturell ungeeignet für Vollband-Musik
DNSMOS   # 16 kHz DNS-Challenge-Sprachkorpus
NISQA    # Sprach-CNN, keine Musik-Trainingsdaten
STOI     # Sprachverständlichkeit 150–5000 Hz
ViSQOL --speech (Default) # Musikspektren systematisch falsch bewertet
```

Erlaubte Musikmetriken: **PEAQ, FAD, PQS-MOS, ViSQOL v3 (`--audio` Mode), Musical Goals**

> **CDPAM** ist als Musik-Metrik verboten (Sprachkorpus-Training, kein Vollband-Musik-Bezug).
> Ersatz: VERSA ONNX (`versa_plugin`) für blinde MOS-Bewertung. Aurik restauriert Musik und Gesang — keine Sprachmetriken.

---

## §8.1.1 OQS — Objektiver Qualitäts-Score

> ⚠ **Wichtig**: `core/mushra_evaluator.py` ist eine algorithmische Approximation
> (PEAQ-ähnlich). Es ist **kein** ITU-R BS.1534-3-konformer MUSHRA-Hörertest.
> In externen Berichten: „OQS (algorithmisch)" — niemals „MUSHRA-Score".> Die 14 Musical-Goal-Schwellwerte sind aus AMRB-Daten hergeleitet („best engineering estimate“).
> Externe Validierung durch subjektiven Hörertest (ITU-R BS.1534-3) steht aus.
> Änderungen an Schwellwerten erfordern dokumentierten Hörertest als Präzedenz.
| OQS-Stufe | Score | Pflicht |
|---|---|---|
| Excellent (A) | ≥ 91 | — |
| Good (B) | ≥ 80 | **Pflicht für jede neue Phase / Plugin** |
| Fair (C) | ≥ 60 | — |

Studio-2026-Modus-Ziel: OQS ≥ 88.

---

## §8.1.2 AMRB v1.0 — Aurik Musical Restoration Benchmark

| Szenario | Defekt | AMRB-Pflicht-Score |
|---|---|---|
| AMRB-01-TAPE | Tape-Hiss + Dropout | OQS ≥ 80 |
| AMRB-02-VINYL | Vinyl-Crackle + Rumble | OQS ≥ 80 |
| AMRB-03-SHELLAC | Shellac-Breitrauschen | OQS ≥ 80 |
| AMRB-04-DIGITAL | Clipping + Quantisierung | OQS ≥ 80 |
| AMRB-05-CODEC | Codec-Artefakte | OQS ≥ 80 |
| AMRB-06-VOCAL | Stimmrauschen + Pitch-Drift | OQS ≥ 80 |
| AMRB-07-REVERB | Raumhall RT60=1.2s | OQS ≥ 80 |
| AMRB-08-HUM | 50-Hz-Brumm + Obertöne | OQS ≥ 80 |
| AMRB-09-DROPOUT | Tape-Dropout 50–200 ms | OQS ≥ 80 |
| AMRB-10-COMPOSITE | Kombinierte Degradierung | OQS ≥ 80 |

**OS-Führerschaft-Schwelle**: Gesamt-Score ≥ **84.0** UND ≥ 8/10 Szenarien bestanden.

```python
from benchmarks.musical_restoration_benchmark import run_benchmark, BenchmarkConfig
report = run_benchmark(config)
assert report.passes_os_leadership_threshold(), f"Score: {report.overall_score}"
```

---

## §8.2 Universelle Garantien

| Garantie | Messung |
|---|---|
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) ≤ 1.0` |
| Chroma-Korrelation (Tonart) | Pearson ≥ 0.95 |
| **Pass-Through (sauberes Material)** | PQS-MOS-Verlust ≤ 0.05, Goals stabil ± 0.02 |
| **Rauschboden (Studio-2026)** | ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events |
| **Temporale Kohärenz** | MOS-Spanne über 10-s-Segmente ≤ 0.30, σ ≤ 0.15 |
| **Stereo-Authentizität** | Mono-Ären M/S-Korrelation nach Restaur. ≥ 0.97 |
| **HF-Kumulativ-Limit** | Presence + Air kumulativ ≤ +4 dB (Listening-Fatigue) |
| Mikro-Dynamik-Erhalt | Pearson LUFS-Profil (400 ms) ≥ 0.92, Crest-Faktor ≤ 1.5 dB |
| Tests grün | Alle bestehenden Pytest-IDs (CI: `pytest --collect-only -q \| tail -1`) |

---

## §8.3 Perceptuelle Verpflichtungen

1. **Natürlichkeit**: MERT-Naturalness-Score ≥ 0.7
   (MERT-harmonicity ist Proxy-Score, kalibriert gegen VERSA-MOS: Pearson = 0.74, n=312;
    bei verfügbarem VERSA-MOS hat dieser Vorrang — MERT nur als Schnellprüfung verwenden)
2. **Harmonische Kohärenz**: Harmonizitäts-Ratio ≥ 0.85 (`MertPlugin.analyze().harmonicity`)
3. **Dynamik-Erhalt**: LUFS-Diff Original ↔ Restauriert ≤ 1 LU
4. **Transientenerhalt**: Attack-Zeiten ≤ 2 ms Änderung
5. **Tonale Stabilität**: Chroma-Korrelation ≥ 0.95
6. **Groove**: Event-Onset-DTW ≤ 8 ms RMS — kein Begradigen von Swing/Rubato
7. **Pass-Through-Invariante** (SNR > 40 dB): PQS-Verlust ≤ 0.05, Goals ≤ ±0.02, LUFS ≤ 0.3 LU
8. **Rauschboden**: Stille-Segmente ≤ −72 dBFS / ≤ −75 dB(A), 0 Musical-Noise-Ereignisse
9. **Mikro-Dynamik**: Pearson LUFS-Profil (400 ms) ≥ 0.92, Crest-Faktor-Abw. ≤ 1.5 dB
10. **Vintage Aesthetics**: Epochen-typische Klang-Charakteristika werden bewahrt (AuthentizitätMetric ≥ vor Restaurierung)

---

## §9 Performance-Budget (Desktop-Hardware, kein GPU)

### §9.1a [RELEASE_MUST] Stationäre vs. nicht-stationäre Defekt-Analyse (v9.10.73)

Der DefectScanner verwendet einen 60 s Center-Crop (`_DETECTOR_CAP_S`) für Performance-Optimierung. Nicht-stationäre Defekttypen MÜSSEN jedoch auf dem **vollständigen Audio** analysiert werden:

| Kategorie | Defekttypen | Audio-Scope | Begründung |
|---|---|---|---|
| **Nicht-stationär** | `DROPOUTS`, `TRANSPORT_BUMP` | Vollständiges Audio | Treten lokal auf (Intro, Outro, Splice-Punkte, Bandanfang) |
| **Stationär** | Alle anderen (Rauschen, Brummen, Flutter, …) | 60 s Center-Crop | Statistisch repräsentativ über kurzen Ausschnitt |

**Invariante**:

```python
_audio_mono_full = audio_mono  # MUSS vor Center-Crop gesichert werden
# ... Center-Crop ...
scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)  # volles Audio
```

**Location-Offset**: Nicht-stationäre Detektoren erzeugen absolute Positionen → `_FULL_AUDIO_DETECTORS`-Set in der Offset-Korrektur ausschließen.

### §9.1b [RELEASE_MUST] Intro-Salienz-Gewichtung (v9.10.73)

Die ersten 5 Sekunden eines Audiosignals bestimmen das Gesamtqualitätsurteil des Hörers (Zacharov & Koivuniemi 2001, Bech & Zacharov 2006). Defekte in dieser „Intro-Zone" MÜSSEN stärker gewichtet werden.

**Algorithmus**:

- Für jeden Defekttyp mit Locations: Anteil der Events in den ersten 5 s berechnen
- Severity-Boost: `severity *= 1.0 + 0.5 * intro_fraction` (max. 1.5×, gedeckelt auf 1.0)
- Metadata: `intro_boost_applied`, `intro_boost_factor`, `intro_events`

**Begründung**: Tape-Leader-Artefakte, Einlaufschwankungen und Splice-Dropouts treten gehäuft in den ersten Sekunden auf. Ohne Intro-Gewichtung kann die Pipeline diese als „statistisch irrelevant" einstufen, obwohl sie den ersten Höreindruck zerstören.

### §9.1c [RELEASE_MUST] Perceptual-Salience-Annotation (v9.10.74)

Jeder erkannte Defekt wird mit einem **psychoakustischen Salienz-Score** (0.0–1.0) annotiert. Defekte, die durch lautere Umgebung **maskiert** werden (Fastl & Zwicker 2007 „Psychoacoustics: Facts and Models"), erhalten einen niedrigeren Severity-Wert; exponierte (hörbare) Defekte behalten volle Severity.

**Wissenschaftliche Basis**:

- Simultane Maskierung: Kontext ≥ 12 dB über Defekt-Lautstärke → maskiert
- Temporale Vorwärts-Maskierung: Lautes Signal innerhalb 200 ms vor dem Defekt → teilmaskiert (Schwelle 8 dB)
- Temporale Rückwärts-Maskierung: Lautes Signal innerhalb 20 ms nach dem Defekt → teilmaskiert (Schwelle 6 dB)
- Loudness-Modell: ITU-R BS.1770-5 momentary loudness (400 ms Fenster, 100 ms Hop)

**Modul**: `backend/core/perceptual_salience.py` — `PerceptualSalienceEstimator` (Singleton, §3.2)

**API**: `annotate_defect_scores(audio, sr, defect_result)` → modifizierte `DefectAnalysisResult` mit:

- `metadata["perceptual_salience"]`: Mittelwert der Salienz aller Events (0.0–1.0)
- `metadata["n_salient_events"]`: Events mit Salienz ≥ 0.5
- `metadata["n_masked_events"]`: Events mit Salienz < 0.3

**Severity-Skalierung**: `severity = severity * (0.3 + 0.7 * mean_salience)` — bewahrt 30 % Basis-Severity auch für vollständig maskierte Defekte.

**Integration**: Wird im DefectScanner.scan() nach §9.1b (Intro-Boost) und vor Location-Offset aufgerufen.

**Pipeline-Nutzen**:  

1. Hörbare Defekte werden priorisiert repariert  
2. Unhörbare Defekte werden geschont (weniger Artefaktrisiko durch unnötige Reparatur)  
3. UX-Reporting: „3 hörbare Defekte behoben, 12 unterhalb der Hörschwelle"

**Invarianten**:

- Analyse-Modul → arbeitet bei nativer Import-SR (kein `assert sr == 48000`)
- NaN/Inf-Guard auf allen numerischen Ausgaben
- Thread-safe Singleton (double-checked locking)

| Schritt | Limit pro Minute Audio |
|---|---|
| DefectScanner | ≤ 2 s |
| Phase-Pipeline gesamt | ≤ 120 s |
| FeedbackChain (alle Iterationen) | ≤ 60 s |
| ExcellenceOptimizer | ≤ 30 s |
| Export (FLAC 24-bit) | ≤ 10 s |

**FeedbackChain-Abbruch:**

```python
MAX_ITERATIONS = 5
CONVERGENCE_DELTA = 0.02
# Regression: |MOS_neu - MOS_alt| > 0.05 → sofortiger Rollback auf best_result
```

### §8.3 [RELEASE_MUST] Psychoakustik & Gänsehaut-Prinzipien (v9.10.75)

**Oberstes Ziel**: Das Restaurierungsergebnis muss beim Hörer **emotionale Wirkung** erzeugen — nicht nur technische Korrektheit. Die Psychoakustik steht über der technischen Perfektion.

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Komponente | Verantwortliches Modul | Anteil |
|---|---|---|
| **Transient-Punch** | TDP (Transient Decoupled Processing) | ~40 % |
| **Mikro-Dynamik-Erhalt** | MDEM (400 ms LUFS-Morphing) + EmotionalArcCorrection (5 s Makro-Bogen) | ~25 % |
| **Rauschbefreiung/Klarheit** | SGMSE+ / OMLSA/IMCRA | ~20 % |
| **Vokal-Präsenz** | Phase 42 + Phase 43 + VocalAIEnhancement | ~10 % |
| **Neurale Synthese** | Vocos 48 kHz (nur Studio 2026, MOS < 4.3) | ~5 % |

**Zwei-Skalen-Dynamik-Schutz** (Pflicht — beide Ebenen im UV3 implementiert):

- **Mikro-Ebene (400 ms)**: `MicroDynamicsEnvelopeMorphing.morph()` — LUFS-Profil-Rückgewinnung (§2.30)
- **Makro-Ebene (5 s)**: `correct_emotional_arc()` — post-MDEM Gain-Korrektur bei Bogen-Abflachung.
  Algorithmus: Per-Segment RMS-Gain (orig/rest), ±6 dB max, 70 % Dämpfung, Savitzky-Golay-geglättet,
  Sicherheits-Revert wenn Arousal-Pearson sich verschlechtert.
  Modul: `backend/core/emotional_arc_preservation.py` — `correct_emotional_arc(original, restored, sr)`

**UV3-Pipeline-Integration** (nach MDEM, vor GP-Update):

1. MDEM korrigiert Mikro-Dynamik (400 ms Fenster)
2. EmotionalArc Post-MDEM-Messung: `measure_emotional_arc(original, restored, sr)`
3. Falls `not arc_preserved`: `correct_emotional_arc()` mit Makro-Gain-Hüllkurve
4. Safety-Revert: wenn Korrektur Arousal-Pearson um > 0.02 verschlechtert → Revert

**RAM-Budget:**

- Audio-Buffer max.: 4 GB
- ML-Modelle aktiv max.: 16 GB gesamt
- Großmodelle (MERT 3,9 GB / AudioSR 5,9 GB): nur bei Bedarf (lazy load)

**CPU-Policy:**

```python
# AUSSCHLIESSLICH CPU — keine GPU-Beschleunigung in Aurik 9.9
providers = ["CPUExecutionProvider"]   # ONNX-Runtime immer
model = model.to("cpu")               # PyTorch immer
torch.set_num_threads(os.cpu_count())  # alle CPU-Kerne nutzen
```

---

## §5 Test-Standards

### Mindestanforderungen pro neuem Modul

| Anforderung | Zielwert |
|---|---|
| Unit-Tests pro Kernmodul | ≥ 35 |
| Shape/Dtype-Tests | ✅ Pflicht |
| NaN/Inf-Tests | ✅ Pflicht |
| Bounds-Tests | ✅ Pflicht (alle metrischen Ausgaben) |
| Edge Cases | ✅ Pflicht (Stille, Rauschen, Dirac-Impuls) |
| Mono + Stereo | ✅ Pflicht für Audio-Eingaben |
| Konsistenz | ✅ Pflicht (selbe Eingabe → selbe Ausgabe) |
| Integration-Tests | ≥ 5 |

### Namenskonvention

```
tests/unit/test_v<VERSION>_<feature_name>.py
# Beispiele:
tests/unit/test_v97_cognitive_layer.py
tests/unit/test_v99_genre_schlager.py       # ≥ 35 Tests
tests/unit/test_v910_remaster_detector.py
```

### pytest-Konfiguration (`pytest.ini`)

```ini
[pytest]
addopts = --timeout=30 --import-mode=importlib -p no:warnings
```

- Kein Test darf > 30 s dauern
- Alle Tests mit **synthetischen Signalen** (`np.random.seed(42)`)
- Keine realen Audio-Dateien in Tests

---

## §14 E2E-Test-Spezifikation (Pflicht ab v9.10.42)

### §14.1 Pflicht-Assertions (`_validate_restoration_result`)

```python
assert len(set(result.phases_executed) & TIER_1_PHASES) >= 2
assert result.quality_estimate >= 0.55              # Formel: §8.1
assert result.material_type is not None
assert result.metadata.get("era") is not None
assert result.metadata.get("panns_tags") is not None
assert np.isfinite(audio).all()
assert np.max(np.abs(audio)) <= _TP_LIMIT + 1e-4
assert len(set(result.phases_executed) & TIER_6_PHASES) >= 3
```

### §14.2 Musik-Qualitäts-Assertions

```python
assert pqs_result.mos >= 4.0   # QUALITY; >= 4.5 für MAXIMUM
assert all(scores[g] >= checker.thresholds[g] for g in applicable_goals)
# Schlager-spezifisch:
assert scores["tonal_center"] >= 0.97
assert scores["waerme"] >= 0.88
```

### §14.3 Pipeline-Integrität

```python
assert config.enable_performance_guard is True
assert config.enable_phase_gate is True
assert len(result.phase_gate_log or []) <= 4   # max. 4 best-effort Phasen gesamt (PMGG darf nie skippen)
```
