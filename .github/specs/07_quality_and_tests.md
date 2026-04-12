# Aurik 9 — Spec 07: Qualitätsziele & Tests

> PQS-Metriken, AMRB-Benchmark, universelle Garantien, Test-Standards,
> E2E-Assertions, Performance-Budget.

---

## §8.1 Numerische Qualitätsgrenzen

### PQS-Metriken (`core/perceptual_quality_scorer.py`)

| Metrik | Hard-Fail-Minimum | Weltklasse-Ziel |
| --- | --- | --- |
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

```text
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
| --- | --- | --- |
| Excellent (A) | ≥ 91 | Exzellenz-Label — kein harter Gate-Wert |
| Good (B) | ≥ 80 | **[RELEASE_MUST] Pflicht für jede neue Phase / Plugin** |
| Fair (C) | ≥ 60 | — |

### §8.1.1a [RELEASE_MUST] Studio-2026 OQS-Gate (v9.10.130)

Für `mode="studio2026"` gilt ein dediziertes End-Gate:

- OQS ≥ **88** ist verpflichtend (vormals TARGET_2026).
- Bei OQS < 88 darf kein finaler Studio-2026-Export freigegeben werden.
- Fallback-Verhalten: kontrollierter Rollback auf bestes artefaktfreies Checkpoint-Audio
    oder Modus-Rückfall auf konservative Qualitätskette mit dokumentiertem `fail_reason`.

**Rationale:** Studio-2026 ist ein Qualitätsversprechen („modern, frisch, kräftig").
Ein optionales Roadmap-Ziel ist dafür normativ zu weich.

---

## §8.1.2 AMRB v1.0 — Aurik Musical Restoration Benchmark

| Szenario | Defekt | AMRB-Pflicht-Score |
| --- | --- | --- |
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

**[RELEASE_MUST] Fragment-Mindestlänge**: Jedes AMRB-Stimulusfragment MUSS **≥ 30 s** lang sein.
Fragmente < 30 s erzeugen OQS-Varianz von ±8 Punkten — ausreichend um einen 80-Punkt-Pass-Fail-Schwellwert
unzuverlässig zu machen. `run_amrb_baseline.py` erzwingt diesen Guard automatisch (`_MIN_AMRB_FRAGMENT_S = 30.0`)
und korrigiert kürzere `--duration`-Angaben mit einem Warn-Log. `n_items ≥ 5` bleibt Pflicht (Nightly-Config).

**OS-Führerschaft-Schwelle**: Gesamt-Score ≥ **84.0** UND ≥ 8/10 Szenarien bestanden.

```python
from benchmarks.musical_restoration_benchmark import run_benchmark, BenchmarkConfig
report = run_benchmark(config)
assert report.passes_os_leadership_threshold(), f"Score: {report.overall_score}"
```

### [RELEASE_MUST] AMRB Seeding-Invariante (v9.10.80 — deterministisch, MD5)

`item_seed` für AMRB-Items MUSS via `_sid_offset(sid)` berechnet werden — **niemals** `hash(sid)`.
Python's eingebautes `hash()` ist pro-Prozess randomisiert (PYTHONHASHSEED) → nicht reproduzierbar.

```python
import hashlib

def _sid_offset(sid: str) -> int:
    """Deterministisches Item-Seeding für AMRB via MD5.
    RELEASE_MUST: Kein hash(sid) — Python hash() ist PYTHONHASHSEED-abhängig.
    """
    return int(hashlib.md5(sid.encode()).hexdigest(), 16) % (2**31)

# Verwendung in MusicalRestorationBenchmark.run():
item_seed = _sid_offset(scenario_id)  # RICHTIG
# item_seed = hash(scenario_id)       # VERBOTEN — nicht reproduzierbar
```

**Nightly-Konfiguration**: `n_items ≥ 5`; Baseline-Key: `"iZotope RX 11 (commercial)"` (OQS 71.0).

---

## §8.2 Universelle Garantien

| Garantie | Messung |
| --- | --- |
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

## §8.2a [RELEASE_MUST] Universal-Fidelity-Gate (All Imports)

Maximale Klangtreue ist nur erfüllt, wenn die Qualitätsziele über **alle** Importklassen
robust gelten, nicht nur für Einzelbeispiele.

**Pflicht-Gates:**

1. **Matrix-Gate**: Material × Qualitätsmodus × Kontext (Ära/Genre) muss grün sein.
2. **Shape/SR-Invarianten-Gate**: keine verdeckten Layout-/Samplerate-Verletzungen.
3. **Fallback-Konsistenz-Gate**: OOM/ML-Ausfälle dürfen Qualität degradieren, aber nie
    den Signalvertrag (Länge, Kanäle, Headroom, End-Gates) brechen.

Ein Feature gilt nur dann als produktionsreif, wenn diese Gates ohne song-spezifische
Sonderbehandlung bestehen.

## §8.2b [RELEASE_MUST] Maximal-umsetzbare Recovery bei Gate-Fail (kein Blind-Hardstop)

Ziel ist das bestmögliche reale Ergebnis pro Song, nicht ein formaler Abbruch.
Ein finaler Quality-Gate-Fail triggert deshalb eine verpflichtende Recovery-Kaskade.

**Verbindlich:**

1. Bei `quality_gate.passed=False` MUSS Aurik eine Recovery-Suche durchführen:
    adaptive Strength-Reduktion, Checkpoint-Rollback, alternative sichere Kette,
    material-/restorability-adaptive Parameter.
2. Export ist zulässig, wenn kein besseres sicheres Ergebnis mehr erreichbar ist,
    aber nur mit explizitem Status (`recovered` oder `degraded`) und vollständiger Fail-Ursache.
3. Verboten sind beide Extreme:
    a) blinder Hardstop ohne Recovery-Suche,
    b) stilles „best-effort success" ohne transparente Degradationskennzeichnung.

Diese Regel stellt sicher, dass Aurik immer das maximal umsetzbare Ergebnis liefert,
ohne Qualitätsversagen zu verschleiern.

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

### §8.3.1 Song-Selbstkalibrierung (phasenübergreifend, psychoakustisch priorisiert)

- Für jeden Song MUSS ein Kalibrierungsprofil (`song_calibration_profile`) erzeugt und im Ergebnis abgelegt werden.
- Profile müssen bounded sein (keine ungebremste Verstärkung), um Song-Overfitting zu verhindern.
- Phasenfamilien-Skalierung MUSS P1/P2-Hörtreue priorisieren; P3–P5 dürfen nie zu Lasten von Natürlichkeit/Authentizität erzwungen werden.
- Variationen in Tonträgerketten/Material/Defektbildern müssen zu unterschiedlichen, aber deterministischen Profilen führen.
- Pflichtartefakt für Tests/Analysen: `RestorationResult.metadata["song_calibration"]`.

### §8.3.2 [RELEASE_MUST] Experience-Propagation-Testpflicht (v9.11.1)

Zusätzlich zur Song-Selbstkalibrierung sind folgende Tests verpflichtend:

1. **Metadata-Contract Test**:
    - `RestorationResult.metadata["joy_runtime_index"]` vorhanden und finite.
    - `RestorationResult.metadata["auto_improvement_recommendations"]` schema-stabil.
    - `RestorationResult.metadata["song_calibration"]["cluster_key"]` vorhanden.
2. **Bridge-Contract Test**:
    - `backend.api.bridge.get_experience_insights()` liefert stabile Rückgabe auch bei fehlenden Feldern.
3. **Orchestrator-Propagation Test**:
    - `AurikDenker` propagiert `RestaurierErgebnis.metadata` bis `AurikErgebnis.metadata` unverändert (bis auf defensive Defaults).
4. **PMGG-CIG-Synchronisations-Test (§2.55)**:
    - `tests/unit/test_pmgg_cig_sync.py` muss bidirektional grün sein
      (PMGG→CIG und CIG→PMGG für alle P1/P2-Exclusions).

**Invariante:** Änderungen an Bridge/Denker/UI/Goal-Gates, die Experience-
Telemetrie oder PMGG/CIG-Exclusions betreffen, dürfen ohne diese Testklassen
nicht als release-fähig gelten.

---

## §9 Performance-Budget (Desktop-Hardware, GPU-Mixed-Mode optional)

> **Kanonische Quelle für RT-Limits:** copilot-instructions.md §2.37 (LIMIT_BALANCED = 32.0× RT).
> Die nachstehenden Werte gelten **pro Minute Audio** und sind mit PerformanceGuard-Toleranzen kalibriert.
> VERBOTEN: niedrigere Limits aus Vorgängerversionen (DefectScanner ≤ 2 s, Pipeline ≤ 120 s) verwenden —
> diese wurden mit v9.10.80 (Quality-First-Hauptlauf) auf die untenstehenden Werte angehoben.
> **GPU-Beschleunigung** (ROCm/DirectML) reduziert Heavy-Plugin-Inferenz erheblich; die Limits gelten für CPU-only als Worst Case.

| Operation | Limit / Minute Audio |
| --- | --- |
| DefectScanner | ≤ **4 s** |
| Phase-Pipeline gesamt | ≤ **240 s** |
| FeedbackChain (alle Iterationen) | ≤ **120 s** |
| ExcellenceOptimizer | ≤ **60 s** |
| RestorabilityEstimator | ≤ **5 s** |
| Export (FLAC 24-bit) | ≤ 10 s |

**Absolutes Zeitlimit Stufe 1:** `MAX_ABSOLUTE_SECONDS = 5400.0` (90 Minuten).
Nach Überschreitung: KMV Stufe 2 (`MLRefinementThread`) übernimmt automatisch.

**FeedbackChain-Abbruch (Fix M, v9.10.100 — MOS-Metrik präzisiert):**

```python
MAX_ITERATIONS = 5
CONVERGENCE_DELTA = 0.02
# Regression-Trigger: PQS-MOS (PerceptualQualityScorer.score().mos) zwischen
# aufeinanderfolgenden Iterationen: |mos_iter_n - mos_iter_n_minus_1| > 0.05
# → sofortiger Rollback auf best_result (höchster PQS-MOS bisher)
#
# NICHT: Musical Goals Ø-Score (der wird separat über GoalPriorityProtocol überwacht)
# NICHT: MERT-harmonicity-Proxy (zu niedrige Sampling-Frequenz für Iterations-Vergleich)
# NICHT: Vergleich mit Baseline (nur Iteration-zu-Iteration, um Overshoot zu detektieren)
#
# Konvergenz-Abbruch (kein Rollback): |mos_iter_n - mos_iter_n_minus_1| < CONVERGENCE_DELTA
# → weitere Iteration würde Qualität nicht verbessern → frühzeitiger Exit
```

### §9.1a [RELEASE_MUST] Stationäre vs. nicht-stationäre Defekt-Analyse (v9.10.73)

Der DefectScanner verwendet einen 60 s Center-Crop (`_DETECTOR_CAP_S`) für Performance-Optimierung. Nicht-stationäre Defekttypen MÜSSEN jedoch auf dem **vollständigen Audio** analysiert werden:

| Kategorie | Defekttypen | Audio-Scope | Begründung |
| --- | --- | --- | --- |
| **Nicht-stationär** | `DROPOUTS`, `TRANSPORT_BUMP` | Vollständiges Audio | Treten lokal auf (Intro, Outro, Splice-Punkte, Bandanfang) |
| **Stationär** | Alle anderen (Rauschen, Brummen, Flutter, …) | 60 s Center-Crop | Statistisch repräsentativ über kurzen Ausschnitt |

**Invariante**:

```python
_audio_mono_full = audio_mono  # MUSS vor Center-Crop gesichert werden
# ... Center-Crop ...
scores[DefectType.DROPOUTS] = self._detect_dropouts(_audio_mono_full)  # volles Audio
```

- Für synthetische Langsignale mit >50 nicht-stationären Events (z. B. Dropouts/Tape-Head-Level-Dips) MUSS `len(score.locations) > 50` gelten.
- Eine künstliche Kappung der Core-Defektliste auf feste Grenzen (z. B. 50/100/256) ist als Regression zu werten.
- Optional verdichtete Anzeige-Listen müssen in separaten UI-Tests validiert werden und dürfen die Core-Liste nicht beeinflussen.

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

> **Hinweis**: Die normativen Performance-Limits sind ausschließlich in §9 (oben) definiert.
> VERBOTEN: Alte Limits (DefectScanner ≤ 2 s, Pipeline ≤ 120 s, FeedbackChain ≤ 60 s) —
> diese wurden mit v9.10.80 (Quality-First-Hauptlauf, Fix I) angehoben und gelten nicht mehr.

### §8.3 [RELEASE_MUST] Psychoakustik & Gänsehaut-Prinzipien (v9.10.75)

**Oberstes Ziel**: Das Restaurierungsergebnis muss beim Hörer **emotionale Wirkung** erzeugen — nicht nur technische Korrektheit. Die Psychoakustik steht über der technischen Perfektion.

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Komponente | Verantwortliches Modul | Anteil |
| --- | --- | --- |
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
   `arc_preserved = True` genau dann, wenn **beide** Bedingungen erfüllt:
   (a) `arousal_pearson_global ≥ 0.85` (5-s-Fenster, gesamte Datei)
   (b) kein einzelnes 5-s-Segment hat `Δarousal_local < −0.08`
       (`Δarousal_local[i] = rms_restored[i] / rms_original[i] − 1`;
        Normiert auf Original-RMS; Segment ≤ 3 s am Dateiende ignoriert).
   Rationale: Globaler Pearson ≥ 0.85 kann trotz eines eingebrochenen Segments
   (z.B. pp vor Refrain wird durch Kompression flachgelegt) bestehen. Die lokale
   Schwelle −0.08 fängt Spannungspunkte, die Schauer auslösen würden.
3. Falls `not arc_preserved`: `correct_emotional_arc()` mit Makro-Gain-Hüllkurve
4. Safety-Revert: wenn Korrektur Arousal-Pearson um > 0.02 verschlechtert → Revert

**RAM-Budget:**

- Audio-Buffer max.: 4 GB
- ML-Modelle aktiv max.: 16 GB gesamt
- Großmodelle (MERT 3,9 GB / AudioSR 5,9 GB): nur bei Bedarf (lazy load)

**Device-Policy (§GPU-Mixed-Mode, v9.11.10):**

```python
# Heavy ML Plugins (>200 MB): GPU wenn verfügbar, CPU-Fallback transparent
from backend.core.ml_device_manager import get_ort_providers, get_torch_device
providers = get_ort_providers("PluginName")  # ONNX-Runtime
device = get_torch_device("PluginName")      # PyTorch
# Leichtgewichtige Plugins (<200 MB), DSP, Analyse: immer CPU
providers = ["CPUExecutionProvider"]          # ONNX-Runtime
model = model.to("cpu")                       # PyTorch
torch.set_num_threads(os.cpu_count())         # alle CPU-Kerne nutzen
```

---

### §8.3.1 [RELEASE_MUST] Tiefen-Immersions-Prinzip — „Ohr in die Musik legen" (v9.10.79)

**Konzept**: Das Restaurierungsergebnis muss dem Hörer ermöglichen, in die Musik hineinzutauchen —
nicht nur zuzuhören, sondern sich von der Aufführung umgeben zu fühlen. Gänsehaut entsteht,
wenn alle akustischen Tiefenschichten gleichzeitig hörbar sind und die Spannungsdynamik
einer Aufführung authentisch erlebt wird.

#### Akustische Tiefenschichten (von außen nach innen)

| Schicht | Frequenzbereich | Enthält | Technische Bedingung |
| --- | --- | --- | --- |
| **Raumluft / Air** | 8–20 kHz | Saiten-Obertöne, Becken-Shimmer, Gesangs-Luft | Noise Floor < −72 dBFS; Phase_06 SBR; Phase_39 Air |
| **Vokal-Intimität** | 4–8 kHz | Frikative /s/ /f/ /ʃ/, Plosive /p/ /t/, Atem | LyricsGuidedEnhancement: `fricative ×1.55`, `plosive ×1.40` |
| **Instrument-Körper** | 200 Hz–4 kHz | Note-Sustain, Saitenresonanz, Bogen-Kratzen | TDP-Transient-Erhalt + MDEM 400 ms LUFS-Morphing |
| **Fundament** | 20–200 Hz | Kick-Punch, Bassresonanz, Raummode | BassKraftMetric + Virtual-Pitch (Missing Fundamental) |
| **Raumtiefe** | Diffus (MS) | Raumluft, Phantom-Center, Tiefenstaffelung | SpatialDepthMetric IACC ≥ 0.70, M/S-Korr. ≥ 0.97 |

#### Physikalische Kausalkette: PMGG-Stabilität → Tiefen-Immersion

```text
Phase_03_denoise bei GP-optimalem strength (§9.7.7 / §2.29b aktiv)
    → Noise Floor < −72 dBFS
    → Air-Layer (8–20 kHz) frei: Saitenobertöne, Atemgeräusch, Becken-Shimmer hörbar
    → Vokal-Intimität-Layer (4–8 kHz) frei: Frikative, Plosive, Atem wahrnehmbar
    → emotionaler Authentizitäts-Schock: „das klingt wie live im Raum"
    → Gänsehaut

Phase_03_denoise @ best-effort strength=0.056 (false P1-Regression, §9.7.7 FEHLT)
    → Noise Floor −55 dBFS (+17 dB über Ziel)
    → Mikrodetails unter Rauschteppich verdeckt
    → Studio-Distanz-Effekt; der Hörer bleibt „außen"
    → kein emotionaler Sog, keine Gänsehaut
```

**E2E-Pflicht-Assertions** (Tiefen-Immersions-Gate):

```python
assert noise_floor_silence_dbfs(restored, sr)                 <= -72.0  # Air-Layer frei
assert pearson_lufs_profile_400ms(original, restored, sr)     >= 0.92   # MDEM Mikro-Dynamik
assert emotional_arc_arousal_pearson(original, restored, sr)  >= 0.85   # Makro-Bogen
assert spatial_depth_iacc(restored, sr)                       >= 0.70   # Raumtiefe
```

#### Vokal-Intimität — Physik der akustischen Nähe

Der Hörer empfindet eine Stimme als „körperlich nah" durch drei physikalische Phänomene:

1. **Konsonanten-Transient** (Plosive /p/ /t/ /k/, Frikative /s/ /f/ /ʃ/): Der kurze Druckstoß
   simuliert das akustische Nahfeld < 50 cm. Over-Denoising verschleift diese Transienten → Stimme
   verliert physische Glaubwürdigkeit. LyricsGuidedEnhancement schützt diese Segmente mit den
   Boost-Faktoren anstatt sie zu dämpfen.

2. **Atemgeräusche** (150–800 Hz + 2–5 kHz): Das Einatmen zwischen Phrasen, das leichte Räuspern,
   das Öffnen der Lippen — diese unlyrischen Geräusche sind die stärksten Proximitäts-Signale.
   Standard-NR entfernt sie als „Rauschen". Aurik schützt sie in `silence`-Segmenten mit
   reduziertem NR-Boost (`×0.70`).

3. **Early Reflections** (1–30 ms): Die ersten Raum-Echos bestimmen die wahrgenommene
   Quellen-Distanz. Phase_20 (Reverb Reduction) darf nur Spät-Hall (> 80 ms RT60-Anteil)
   entfernen — Early Reflections müssen erhalten bleiben, sonst kollabiert die wahrgenommene
   Quellen-Distanz auf „unendlich weit".

#### Spektrale Phasenkohärenz — Tiefenstaffelung durch IPD

Nach **jeder** Spektral-Modifikation: **PGHI-ISTFT** (Phase Gradient Heap Integration, Virtanen 2018).

Das menschliche Gehirn nutzt interaurale Phasendifferenzen (IPD, < 0.7 ms Laufzeitunterschied)
für horizontale Richtungslokalisierung (Blauert 1997, §3.1 Precedence Effect). Griffin-Lim
randomisiert die Phase → alle Instrumente erscheinen in derselben Tiefenebene → räumliche
Staffelung kollabiert → „flaches Stereo-Bild" ohne Tiefe.

Mit PGHI: Phasenlaufzeiten erhalten → Gitarre links-vorne, Klavier rechts-mitte, Gesang mittig-nah.
**VERBOTEN** als Studio-2026-Endschritt: `griffinlim()`.

#### Emotionaler Atemzug — Spannungsdynamik als Gänsehaut-Mechanismus

Der stärkste Gänsehaut-Auslöser ist nicht die lauteste Stelle einer Aufnahme, sondern der
Moment **kurz davor**:

- Ein `pp`, das den Hörer in die Stille zieht, bevor das `fff` trifft
- Eine Fermate, die die Zeit anhält
- Ein Diminuendo, das den nächsten Akkord unausweichlich macht

Diese Spannungsmechanik setzt voraus:

- `pearson_lufs_400ms ≥ 0.92` (MDEM): Mikro-Energiewellen auf Sub-Phrasen-Ebene erhalten
- `arousal_pearson_5s ≥ 0.85` (EmotionalArc): Intensitätsbogen über Phrasen-Ebene erhalten
- Keine künstliche Dynamik-Kompression zwischen piano und forte Stellen

Wenn alle Stellen gleich laut klingen, ist die Spannungs-Mechanik zerstört und Gänsehaut unmöglich.

---

## §5 Test-Standards

### §5.4 [RELEASE_MUST] ML-Headroom-Guard + KMV-Rueckgewinnung

Folgende Testfaelle sind fuer heavy ML-Phasen verpflichtend (z. B. phase_03, phase_06, phase_20, phase_23, phase_24, phase_55):

1. **Low-RAM Completion Test**: Unter simuliert knappem RAM darf die Restaurierung nicht crashen; Ergebnis muss erfolgreich exportierbar sein.
2. **Guard Event Contract Test**: Bei Guard-Trigger muss `metadata["ml_guard_events"]` gesetzt sein und alle Pflichtfelder enthalten.
3. **Deferred-Phase Contract Test**: Guard-betroffene Phase muss in `deferred_phases` eingetragen werden.
4. **KMV Recovery Test**: Stufe 2 fuehrt die deferred Phase ohne RT-Limit nach; Overwrite nur bei `stufe2_quality_estimate >= stufe1_quality`.
5. **No-Original-Rollback Test**: Guard-Trigger darf nie zu `action="rollback"` auf Original-Audio fuehren.

**Regression-Fokus:** Diese Tests muessen sowohl mono als auch stereo sowie kurz (<60 s) und lang (>=180 s) abdecken.

### §5.5 [RELEASE_MUST] Vollpipeline-Determinismus-Gate

Bei identischer Eingabe, Umgebung und Konfiguration muessen zwei Vollpipeline-Runs bitnah identisch sein.

Pflichtkriterien:

1. `max_abs_err <= 1e-6`
2. `rms_err <= 1e-7`
3. identische `phases_executed`
4. identische `release_mode`-Entscheidung

### §5.6 [RELEASE_MUST] Stratifiziertes Konkurrenz-Gate

Konkurrenzvergleich wird nicht nur als Gesamtmittel, sondern pro Zelle einer Material-Defekt-Matrix bewertet.

Pflicht-Matrix:

- Materialien: `tape`, `vinyl`, `shellac`, `digital`, `vocal`
- Defektklassen: `hiss`, `crackle`, `dropout`, `reverb`, `hum`, `codec`

Release-Logik:

- Fail bei regressiver Einzelzelle gegen Referenz, auch wenn Gesamtmittel besteht.
- Bericht muss Delta pro Zelle enthalten.

### §5.7 [RELEASE_MUST] Externes Mini-MUSHRA-Artefakt

Bei Kern-aenderungen (Kernphasen, PMGG, DefectScanner, heavy ML-Fallbacks) ist ein externer Mini-MUSHRA-Bericht Pflicht.

Pflichtanforderungen:

1. mindestens 6 Szenarien, davon mindestens 2 Vocal-Szenarien
2. mindestens 8 Hoerer
3. Szenario-Score, Konfidenzintervall, Delta zur Vorversion
4. Bericht als Release-Artefakt versioniert abgelegt

### §5.7a [RELEASE_MUST] Modusgetrennte Hörvalidierungs-Checkliste (v9.10.130)

Die externe Hörvalidierung MUSS beide Modi getrennt ausweisen. Ein kombinierter
Gesamtwert ohne Modus-Trennung ist unzulässig.

**Pflicht-Checkliste Restoration (`mode=restoration`):**

1. Blindvergleich `Input` vs `Restored` vs `Reference/Needledrop-Best-Available` dokumentiert
2. Bewertungsachsen enthalten mindestens:
    `Natürlichkeit`, `Authentizität`, `Artefaktfreiheit`, `Tonale Treue`
3. Keine hörbare Verschlechterung bei P1/P2-Achsen gegenüber Input
4. Carrier-Chain-Inversion ist nachvollziehbar pro Szenario dokumentiert
5. **Szenarien-Pflicht**: Alle 3 mandatory Szenarien (`docs/reports/hearing_test_scenarios_restoration.yaml`) müssen als Validierungsbasis durchlaufen werden:
    - `RESTORATION_SCENARIO_1`: Vinyl Wear + Surface Noise (Rock Vocal)
    - `RESTORATION_SCENARIO_2`: Tape Hiss + Oxide Dropout (Jazz Vocal)
    - `RESTORATION_SCENARIO_3`: Shellac Brittleness + Click Storm (Classical Vocal)
6. Jedes Szenario muss die dokumentierten `mandatory_validation_points` erfüllen
7. GO/NO-GO-Entscheidung folgt Restoration-Entscheidungslogik in `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (Phase 1–3)

**Pflicht-Checkliste Studio 2026 (`mode=studio2026`):**

1. Blindvergleich `Input` vs `Restored` vs `Modern Reference` dokumentiert
2. Bewertungsachsen enthalten mindestens:
    `Frische/Presence`, `Punch/Bass-Kraft`, `Klarheit`, `Artefaktfreiheit`
3. OQS-Gate (§8.1.1a) und Hörurteil dürfen sich nicht widersprechen
4. Bei Widerspruch gilt Hörurteil als Release-Blocker bis Root-Cause-Analyse vorliegt
5. **Szenarien-Pflicht**: Alle 3 mandatory Szenarien (`docs/reports/hearing_test_scenarios_studio2026.yaml`) müssen als Validierungsbasis durchlaufen werden:
    - `STUDIO2026_SCENARIO_1`: Compressed Pop Mix + Thin Vocal (Pop/Dance Vocal)
    - `STUDIO2026_SCENARIO_2`: Lo-Fi Hip-Hop Muddy Mix + Weak Vocal (Hip-Hop Vocal)
    - `STUDIO2026_SCENARIO_3`: Acoustic Folk Thin + Narrow Stereo (Folk Vocal)
6. Jedes Szenario muss die dokumentierten `mandatory_validation_points` erfüllen (OQS ≥ 86–88, PQS MOS ≥ 4.3–4.5)
7. GO/NO-GO-Entscheidung folgt Studio-2026-Entscheidungslogik in `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md` (Phase 1–4)

**PR-Artefakt-Pflicht:**

- Pro Kernänderung muss ein ausgefülltes Template abgelegt werden:
  `docs/guides/PR_HOERVALIDIERUNG_TEMPLATE.md`
- Pflichtfelder: Szenarien, Hörerzahl, Ergebnis je Modus, Blocker/Entscheidung,
  Link auf Rohdaten/Anhang.

**Verfahren für großflächige Validierung (6-Szenario-Suite):**

1. **Input**: Hörvalidierung durchlaufen mit ≥ 8 Hörern pro Szenario (total 48 Hörer × Scenario-Datum)
2. **Prozess**: Folge dem **Structured GO/NO-GO Decision Protocol** (`docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`)
3. **Decision Flow**:
    - Restoration-Modi: Pre-Review Checks → Aggregate Gates → Per-Scenario Thresholds → Summary Decision
    - Studio2026-Modi: Pre-Review Checks → Objective Metric Gates (OQS, PQS, Artifacts) → Listener MOS → Mode-Goals → Per-Scenario Fine-Grained → Summary Decision
4. **Output**: Go/No-Go Entscheidung mit vollständiger Audit-Dokumentation in PR
5. **Escalation**: Bei NO-GO oder Conditional-GO siehe Protocol §VI (Remediation & Re-Test)

**Verboten:**

- rein algorithmische Freigabe ohne externes Hörvalidierungsartefakt
- Vermischung von Restoration- und Studio-2026-Ergebnissen in einer Einzelnote
- Verwendung von Hörer-Scores aus der falschen Szenario-Spur (Restoration × Studio2026 nicht mischen)

### Mindestanforderungen pro neuem Modul

| Anforderung | Zielwert |
| --- | --- |
| Unit-Tests pro Kernmodul | ≥ 35 |
| Shape/Dtype-Tests | ✅ Pflicht |
| NaN/Inf-Tests | ✅ Pflicht |
| Bounds-Tests | ✅ Pflicht (alle metrischen Ausgaben) |
| Edge Cases | ✅ Pflicht (Stille, Rauschen, Dirac-Impuls) |
| Mono + Stereo | ✅ Pflicht für Audio-Eingaben |
| Konsistenz | ✅ Pflicht (selbe Eingabe → selbe Ausgabe) |
| Integration-Tests | ≥ 5 |

### Namenskonvention

```text
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

### §5.8 [RELEASE_MUST] Pytest-Teardown-Stabilität für große Suiten (v9.10.129)

**Problem**: Unbedingtes `gc.collect()` nach jedem Test kann in Suiten mit tausenden Tests sporadische `pytest-timeout`-Fehler im Teardown auslösen. Zusätzlich können lang lebige Monitor-Threads (z. B. Plugin-/RAM-Manager) nach Testende weiterlaufen und den Teardown verfälschen.

**Pflichtregeln:**

1. Per-Test-Teardown nutzt standardmäßig **inkrementellen/leichten GC** (`gc.collect(0)` oder äquivalent), nicht volles `gc.collect()`.
2. Volles `gc.collect()` ist nur **cadence-gesteuert**, dateibasiert oder sessionbasiert erlaubt.
3. Diagnose-Kadenz darf optional über Env-Flag gesteuert werden; Standardverhalten muss für lokale Unit-Suiten timeout-stabil bleiben.
4. Hintergrund-Manager mit eigenem Thread müssen in `pytest_sessionfinish`, Fixture-Finalizern oder äquivalentem Session-Cleanup best-effort gestoppt werden.
5. Singleton-basierte Manager dürfen nach Session-Cleanup auf `None` zurückgesetzt werden, damit Folge-Sessions keinen stale Thread-/State-Rest sehen.

**Referenz-Pattern:**

```python
_GC_FULL_INTERVAL = int(os.environ.get("AURIK_TEST_FULL_GC_INTERVAL", "0"))
_gc_teardown_counter = 0

@pytest.fixture(autouse=True)
def _gc_after_test():
    yield
    global _gc_teardown_counter
    _gc_teardown_counter += 1
    gc.collect(0)
    if _GC_FULL_INTERVAL > 0 and (_gc_teardown_counter % _GC_FULL_INTERVAL) == 0:
        gc.collect()

def pytest_sessionfinish(session, exitstatus):
    manager.shutdown()   # best-effort, non-blocking
    singleton_module._instance = None
```

**VERBOTEN:**

- Unbedingtes Full-GC in jedem Test-Teardown der Standard-Unit-Suite.
- `join()` ohne Timeout in Test-Cleanup-Pfaden.
- Verlassen auf `daemon=True` als einziges Lebenszyklus-Modell für Hintergrund-Threads.

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
# Keine starre Obergrenze für Recovery-Schritte: entscheidend ist, dass
# best_effort-Ereignisse transparent protokolliert und nicht als stiller
# Success gewertet werden.
assert all("action" in e for e in (result.phase_gate_log or []))
```
