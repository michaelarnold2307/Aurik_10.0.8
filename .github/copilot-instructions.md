# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: März 2026 — Version **9.10.57**
>
> **instructions_version: 2.9** — komprimiert 24.03.2026
>
> Bump-Regel: neue RELEASE_MUST-Zeile, neues Gate oder §-Änderung → `instructions_version` inkrementieren + `docs/CHANGELOG_HISTORY.md` Eintrag.
>
> Aktuelle Testzahl: **~7750+ Pytest-IDs** (inkl. parametrisierter Tests; `def test_`-Funktionen ≈ 7252; alle grün)
>
> Stand: 21. März 2026 — Vocos 48 kHz nativ bevorzugt (§2.37); laion_clap ONNX-Format; Hybrid-Release-Mode RELEASE_MUST.
>
> **§2.36 `LyricsGuidedEnhancement`** ist ab Version **9.10.x Pflicht** (bisher v10.0-Label entfernt).

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

## Normative Priorisierung (P1-1 MUST/TARGET-Split)

`[RELEASE_MUST]` = Harte Release-Bedingung; CI-Gates nicht skippbar. `[TARGET_2026]` = Roadmap 2026, kein Release-Blocker außer explizit als Gate markiert. (Pflicht/bindend/verboten → RELEASE_MUST; Studio 2026/Weltklasse → TARGET_2026.)

Gate-zuordnung (aktuell implementierte CI-Guards):

| Bereich | Einstufung | Test-ID / Gate |
|---|---|---|
| Docker-/Offline-Produktionspfade | `[RELEASE_MUST]` | `tests/normative/test_no_docker_in_production_paths.py` |
| Wettbewerber-Gate (nicht skippbar) | `[RELEASE_MUST]` | `tests/normative/test_competitive_ci_gate.py` |
| Performance-Budget-Konfigurationsinvarianten | `[RELEASE_MUST]` | `tests/normative/test_performance_budget_ci_gate.py` |
| 14-Goal-Adaptive-Anbindung in UV3 | `[RELEASE_MUST]` | `tests/unit/test_unified_restorer_v3.py` (`test_44`/`test_45`) |
| Structured Fail Reasons in `RestorationResult.metadata` | `[RELEASE_MUST]` | `tests/unit/test_unified_restorer_v3.py` (`test_46`-`test_51`) |
| Memory-Budget `try_allocate()` für alle ML-Plugins | `[RELEASE_MUST]` | `tests/normative/test_combined_ml_memory_budget.py` + Architektur-Invariante: `ml_memory_budget.try_allocate()` vor jedem Modell-Laden (nie `PluginLifecycleManager.try_allocate()`) |
| Heavy-Tests isoliert (ML/slow kein Standard-Run) | `[RELEASE_MUST]` | `conftest.py` auto-markiert `ml/slow`; Freigabe nur mit `--run-heavy-tests` |
| AMRB-Seeding-Invariante (deterministisch, MD5) | `[RELEASE_MUST]` | `benchmarks/musical_restoration_benchmark.py` `item_seed` via `_sid_offset(sid)` (MD5-basiert, kein `hash(sid)`) |
| Hybrid-Release-Mode (release_mode + all_runtime_ready) | `[RELEASE_MUST]` | `tests/normative/test_hybrid_release_mode.py` — 14 Tests; prüft `release_mode` (primary\|fallback\|blocked) + Fallback-Kaskaden |
| Kombiniertes ML-Stack-Budget (Lazy + Core ≤ 12 GB) | `[RELEASE_MUST]` | `tests/normative/test_combined_ml_memory_budget.py` — 12 Tests; prüft Budget-Formel, Lazy-Klassifikation, thread-sichere Allokation |
| §2.36 LyricsGuidedEnhancement aktiv + Modellpfade definiert | `[RELEASE_MUST]` | `tests/normative/test_lyrics_guided_enhancement_gate.py` |
| OQS ≥ 88 / Weltklasse-Ziele | `[TARGET_2026]` | Roadmap/Benchmark-Ziel, kein harter Release-Blocker |
| AMRB-basierter MUSHRA-Hörertest (ITU-R BS.1534-3) | `[TARGET_2026]` | Extern validiert 14 Goal-Schwellwerte; ersetzt „best engineering estimate"-Status; geplant nach OQS ≥ 84.0-Erreichung |

## [RELEASE_MUST] Projektgrenzen (bindend, keine Ausnahmen)

- **Reine Desktop-App** für Linux (AppImage) und Windows 10/11 (.exe)
- **Kein Cloud, kein Server, kein Docker, kein `pip install`** für Endnutzer
- **Out-of-the-Box-Pflicht**: Läuft auf frischem System ohne Python/Terminal
- **100 % offline** nach Installation — alle ML-Modelle lokal gebündelt
- Nur **Mono und Stereo** unterstützt (> 2 Kanäle → PANNs-gewichteter Downmix)
- **Kein Fremdedit am Original-Audio** — immer neue Ausgabedatei in `output/`

## [RELEASE_MUST] 14 Musical Goals (Pflicht-Schwellwerte)

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

> **SpatialDepthMetric**: IACC (Blauert 1997), < 0.70 → Phantom-Center-Zusammenbruch. Mono-Ären: via GoalApplicabilityFilter deaktiviert. Alle 14 Schwellwerte AMRB-kalibriert; MUSHRA-Test ausstehend.

**Sub-Metriken (Pflicht-Implementierungsdetails):**
- `TimbralAuthenticityMetric`: MFCC-Pearson ≥ 0.95, Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 %
- `ArticulationMetric`: Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms
- `TonalCenterMetric`: Chroma-Korrelation ≥ 0.95 **und kein Key-Shift > 0 Cent** (absolut tonarterhaltend)
- `BrillanzMetric` / `WaermeMetric`: Frequenzgewichtung nach **ISO 226:2023 Equal-Loudness** — kein lineares Energiemessen
- `BassKraftMetric`: enthält Virtual Pitch (Missing Fundamental) via Oberton-Analyse 120–500 Hz
- `SeparationFidelityMetric`: SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition

**Invariante**: Jede Restaurierungsoperation darf keines der 14 Ziele verschlechtern. Pflicht-Check: `MusicalGoalsChecker().measure_all(audio, sr)` → `assert all(scores[g] >= t for g, t in checker.thresholds.items())`.

## [RELEASE_MUST] Qualitätsmessung & Metriken-System (§8.1 — PFLICHT)

| Metrik | Hard-Fail-Minimum | Weltklasse-Ziel |
|---|---|---|
| PQS MOS | ≥ 3.8 (generell) / ≥ 4.5 nur `cd_digital`/`dat`/`mp3_high`/`aac` | ≥ 4.5 (nur digitale) |
| PQS NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD (dB) | ≤ 8.0 | ≤ 3.0 |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

> MOS ≥ 4.5 gilt **NUR** für digitale Hochqualitäts-Quellen. Shellac ≥ 3.8, Vinyl ≥ 4.0, Tape ≥ 4.2. `assert mos >= 4.5` ohne Materialkontext ist ein Programmierfehler.

**Normative `quality_estimate`-Formel** (einzige erlaubte):
`quality_estimate = max(0.0, min(1.0, 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4))`
VERBOTEN: `quality_estimate * 1.15`. E2E-Pflicht: `result.quality_estimate >= 0.55`.

OQS = `core/mushra_evaluator.py` (algorithmische PEAQ-Approximation — **kein** ITU-R-MUSHRA). In externen Berichten: „OQS (algorithmisch)".

| Stufe | Score | Pflicht |
|---|---|---|
| Good (B) | ≥ 80 | **Pflicht für jede neue Phase / Plugin** |
| Excellent (A) | ≥ 91 | [TARGET_2026] Studio-2026-Ziel: OQS ≥ 88 |

10 Szenarien AMRB-01-TAPE … AMRB-10-COMPOSITE: alle **OQS ≥ 80** (Pflicht).
[TARGET_2026] Leadership: Gesamt-Score ≥ **84.0** und ≥ 8/10 Szenarien. Aurik ≥ iZotope RX 11 in ≥ 7/10.
Seeding: `_sid_offset(sid)` (MD5-basiert) — **kein** `hash(sid)`. Nightly: `n_items ≥ 5`.
Baseline-Key: `iZotope RX 11 (commercial)` (OQS 71.0). RX 10-Key als Legacy-Alias.

## [RELEASE_MUST] Autonomer Magic-Button-Betrieb + Profi-Highlights

### Vollautarker Bedienvertrag (bindend)

- Nutzerinteraktion ist auf **genau eine Entscheidung** begrenzt: `Restoration` oder `Studio 2026`.
- Danach läuft die gesamte Kette ohne manuelle Parameter, ohne Modul-Slider, ohne Nachjustage bis zum Export.
- Pflicht-Einstieg: `AurikDenker.denke(audio, sr, mode, progress_callback)`; kein UI-Bypass direkt in UV3.
- Export erfolgt nur nach bestandenem Qualitäts-Gate (Musical Goals + PQS + OQS/AMRB-Kontext + Safety-Invarianten).

### Profi-/Studio-Qualitätsprinzipien

| Referenz | Verbindliche Übersetzung in Aurik |
|---|---|
| iZotope RX 11 | CausalDefectReasoner + adaptive Phase-Selektion; feste Global-Presets verboten |
| Spectral-Editor-Klasse | Stem/HPSS/NMF-Auftrennung VOR aggressiven Einzelreparaturen |
| CEDAR-/Cambridge-Klasse | Natürlichkeit/Authentizität nie für stärkere NR opfern |
| Mastering-DAW-Klasse | LUFS + TruePeak + Chroma-/TonalCenter-Checks als Export-Voraussetzung |
| Adaptive-Workflow-Klasse | FeedbackChain-Rollback + PhysicalCeilingEstimator |

**Autopilot-Optimierungsregeln**: ≥ 2 Kandidatenpfade (MOO-Score); P1/P2-Regression → verworfen; Vintage-Invarianten > Brillanz; Fail-Reason in `RestorationResult.metadata` (kein best-effort Export).

### [RELEASE_MUST] No-Competing-Instances-Protokoll

- **Single-Orchestrator**: Pro Prozess nur **eine aktive** Aurik-Orchestrierung. `get_aurik_denker()` als Singleton-Zugriff.
- **Single Active Batch Thread**: Startversuche bei `isRunning()==True` blockieren.
- **Watchdog + Interrupt**: `requestInterruption()` → `wait(3000)` → `terminate()` — keine Zombie-Threads.
- **UI-Gating**: Magic-Buttons während aktiver Verarbeitung deaktiviert.
- **Atomisches Schreiben**: `.tmp` → `os.replace` für Export-Dateien.

**[RELEASE_MUST] Härtungsfahrplan**: (1) Denker-Einstieg erzwungen • (2) Konkurrenzfreiheit • (3) AMRB-Determinismus • (4) ML-Failure-Fallbacks (`release_mode = primary|fallback|blocked`) • (5) Export-Gate • (6) 32×RT + OOM-Budget • (7) Kein manueller Bypass.

## Kanonischer Pipeline-Ablauf (Zusammenfassung)

**PFLICHT-EINSTIEGSPUNKT: `AurikDenker.denke(audio, sr, mode, progress_callback)`**
Kein direktes Aufrufen von `UnifiedRestorerV3.restore()` aus dem Frontend — immer über `AurikDenker`.

### §11.7a Denker-Rollendifferenzierung (Pflicht, v9.10.74)

| Stufe | Denker | Domäne | Kurzregel |
|---|---|---|---|
| 6 | ReparaturDenker | Defekt-Beseitigung | Entfernt bekannte Störungen (Clicks, Hum, Clipping) |
| 7 | RekonstruktionsDenker | Rekonstruktion | **Erschafft, was fehlt** — füllt Lücken, annotiert Bandwidth-Verlust |
| 8 | RestaurierDenker | Restaurierung | **Bewahrt, was da ist** — orchestriert UV3, schützt Klangcharakter |

Kontextfluss: `defect_result → ReparaturDenker → RekonstruktionsDenker(+defect_result) → RestaurierDenker(+reconstruction_context) → UV3`

UV3-Kernreihenfolge: DCOffset-Removal → TDP (HPSS) → RestorabilityEstimator → EraClassifier → MediumClassifier → GoalApplicabilityFilter → AdaptiveGoalThresholds → DefectScanner (28 Defekte) → CausalDefectReasoner (14 Ursachen) → GPParameterOptimizer → HarmonicPreservationGuard → Phasen-Ausführung (01–56) → FeedbackChain → PhysicalCeilingEstimator → MusicalGoalsChecker → MicroDynamicsEnvelopeMorphing → RestorationResult

**Parallelisierungs-Invariante**: Tier 0+1 sequenziell; EraClassifier+GermanSchlager+MediumClassifier parallel (`ThreadPoolExecutor max_workers=3`); Tier 6 sequenziell.

> Details: `.github/specs/02_pipeline_architecture.md`, `denker/aurik_denker.py`

## Adaptive Qualitätsziele — Schlecht-Material-Strategie (§2.31–§2.34)

**Statische Schwellwerte VERBOTEN.** Vor jeder Restaurierung material-, ära- und restorability-adaptiv skalieren:
`get_adaptive_goals_and_config(audio, sr)` → skalierte Schwellwerte, GoalApplicabilityFilter, PhysicalCeilingEstimator.
Priority-Hierarchie: P1 (natürlichkeit, authentizität) → P2 (tonal_center, timbre, artikulation) → P3–5 best-effort.
P1+P2-Regression → `FeedbackChain`-Rollback; Ceiling Δ < 3 % → Terminierung.

## Algorithmische Pflicht-Mindeststandards

Primär → Fallback1 → DSP-Fallback-Kaskade für alle Anwendungsfälle.
**VERBOTEN** als Musikmetrik: `PESQ`, `DNSMOS`, `NISQA`, `STOI`, `ViSQOL --speech`, `CDPAM`.
**VERBOTEN** als Enhancement-Plugin: ~~`dccrn_plugin`~~, ~~`fullsubnet_plus_plugin`~~.

> Vollständige SOTA-Entscheidungsmatrix: `.github/specs/04_dsp_standards.md` §4.4

### [RELEASE_MUST] Hybrid-Release-Mode für Kernmodelle

- `release_mode` ∈ `primary|fallback|blocked` — Statusskripte greifen auf JSON zurück.
- Fallback-Kaskaden: `sgmse_plus.ts → wpe_dsp`, `versa → pqs_dsp`, `flow_matching → cqtdiff → diffwave`.
- Quarantänisierte Crash-Kandidaten (z. B. RMVPE) dürfen **nicht** als Primärpfad registriert werden.

## Schlecht-Material-Verarbeitungsregeln

**Adaptive Schwellen §2.29**: `REGRESSION_THRESHOLD` restorability-abhängig (GOOD 0.012 / FAIR 0.040 / POOR 0.060), max. 5 Retries.

**[RELEASE_MUST] §2.29 PMGG Phase-Skip-Verbot (v9.10.64)**:
- PMGG darf Phasen **NIEMALS** überspringen (kein Rollback auf Original-Audio).
- CausalDefectReasoner hat die Phase als notwendig bestimmt — sie **MUSS** angewendet werden.
- Nach 5 gescheiterten Retries: **Best-Effort** — der Versuch mit der **geringsten Musical-Goal-Regression** wird angewendet (`action="best_effort"`).
- **VERBOTEN**: `return audio, scores_before, "rollback", 0.0` — Rückgabe von unverändertem Original-Audio = Phasen-Skip.
- Gültige PMGG-Actions: `"passed"` | `"retry1"` … `"retry5"` | `"best_effort"` | `"best_effort_rN"`
**Era-GP-Warmstart §2.14**: decade ≤ 1940 → `noise_reduction_strength ~ N(0.90, 0.05)`; ≤ 1960 → N(0.75, 0.08); ≥ 1970 → N(0.50, 0.10).
**Material-MOS §6.2**: MOS ≥ 4.5 NUR für `cd_digital/dat/mp3_high/aac`; Shellac ≥ 3.8, Vinyl ≥ 4.0, Tape ≥ 4.2.
**Chunk-Größe §7.6**: Silence 120 s, Severity ≥ 0.6 → 5 s, ≥ 0.3 → 15 s, sonst 60 s (Min. 2 s / Max. 120 s).
Implementierung: `backend/core/adaptive_chunk_processor.py` — `process_in_adaptive_chunks(phase_fn, audio, sr, max_severity)`. Crossfade: Hanning 10 ms. Phasen können das Modul opt-in nutzen.

## Vintage Aesthetics (§5 — bindend)

**SOFT_SATURATION** (Röhren/Tape-Charakter) = **BEWAHREN**. **CLIPPING** (Amplitudenbeschädigung) = **REPARIEREN**.
`classify_clipping()`: flat_tops > 0.1 % UND THD_odd > THD_even×1.5 → CLIPPING; sonst → SOFT_SATURATION → phase_23 überspringen.

> **Allgemeiner Grundsatz SR-Agnostik in Analyse-Modulen** (autoritativ: Performance-Budget §2.37):
> Alle Analyse-/Scan-/Klassifikations-Module (DefectScanner, `classify_clipping`, `analyse_clipping`, RestorabilityEstimator, EraClassifier, MediumClassifier) arbeiten bei **nativer Import-SR**. THD-Berechnungen nutzen `sr` nur für Frequenz-Bin-Zuordnung — die Mathematik ist SR-agnostisch. **VERBOTEN**: `assert sr == 48000` in diesen Modulen. `assert sr == 48000` gilt **ausschließlich** für Verarbeitungs-Phasen (01–56) und Plugins.

1920–1940: Rolloff ≤ 7 kHz nicht erweitern, H2/H4 bewahren. 1940–1975: `phase_22` nur emulieren, nie eliminieren.

## §2.9 PANNs Instrument-Phasen-Aktivierungsmatrix (Pflicht-Schwellwerte)

| PANNs-Kategorie | Phase | Schwellwert |
|---|---|---|
| Singing voice / Vocals / Speech | `phase_19` + `phase_42` + `phase_43` + VocalAIEnhancement | ≥ 0.40 / ≥ 0.35 |
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.50 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.50 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.50 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.50 |

> **Invariante**: Instrument-Schwelle ist einheitlich **0.50** (nicht 0.60). Höherer Wert blockiert Enhancement bei Ensemble-Aufnahmen. Änderungen hier → immer auch `backend/core/unified_restorer_v3.py` L≈5822 + `plugins/panns_plugin.py` Docstring anpassen.

## Vocal-Restaurierungskette (§2.8)

Reihenfolge: `GenderDetector` → SGMSE+ → FCPE/CREPE/pYIN → FormantTracker (LPC Ord. 30–40 @ 48 kHz) → BreathDetector (±0.05) → De-Esser → `VocalAIEnhancement` → PSOLA (Pitch-Korrektur > ±2 HT).
**API-Falle**: `enhanced, report = self.breath_intelligence.process(audio, sr)` — kein `events`-Argument!

```python
# Singleton pattern — thread-safe, double-checked locking (bindend für alle Kernmodule)
import threading
_instance = None
_lock = threading.Lock()
def get_my_module():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyModule()
    return _instance
```

```python
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
audio = np.clip(audio, -1.0, 1.0)    # am Ausgang jeder Phase
assert sample_rate == 48000           # am Eingang jeder Phase/Plugin (NICHT in Analyse-Modulen — siehe Performance-Budget)
logger.info("phase=%s score=%.2f", phase, score)  # kein print(); Logs auf Englisch
```

## Verbotene Praktiken

**Code-Qualität — verboten:**
- `print(...)` → `logger.info(...)`; öffentliche APIs `return dict` → `@dataclass`; `_cache = {}` ohne `threading.Lock()`

**Algorithmen — verboten:**
- `np.fft.rfft/istft` ohne PGHI nach Spektral-Modifikation; RMS- / Peak-Normalisierung → LUFS ITU-R BS.1770-5
- LPC Ordnung < 16 → Ord. 30–40 bei 48 kHz; `scipy.signal.wiener()` primär → OMLSA/DeepFilterNet
- `griffinlim()` als Studio-2026-Endschritt → Vocos/HiFi-GAN; `import ddsp` (TF-Dependency) → `from dsp.ddsp_synth import DDSPSynth`

**Architektur — verboten:**
- `from Aurik910... import` in `backend/core/` (keine UI-Importe im Backend)
- `torch.load(..., map_location="cuda")` (CPU-only)

**Metriken — verboten:** `pesq()`, `dnsmos()`, `nisqa()` (kein Musik-Training)

**Vocal-Pipeline — API-Falle:**
- FALSCH: `self.breath_intelligence.process(audio, sr, events)` → TypeError
- RICHTIG: `enhanced, report = self.breath_intelligence.process(audio, sr)`

## Checkliste neues Kernmodul (Pflicht)

```
□ backend/core/<modul>.py — Singleton + Convenience-Funktion
□ threading.Lock() + Double-Checked Locking
□ Alle public APIs: vollständige PEP 484 Type-Annotations
□ Docstrings mit Algorithmus-Beschreibung + math. Formeln
□ NaN/Inf-Guard in JEDER numerischen Ausgabefunktion
□ Ergebnisse als @dataclass (kein raw dict)
□ assert sample_rate == 48000 am Eingang (nur Phasen/Plugins; Analyse-Module arbeiten bei nativer Import-SR)
□ ≥ 35 Unit-Tests: Shape, NaN, Bounds, Edge-Cases, Mono, Stereo
□ Musical Goals: kein Ziel nach dem Modul schlechter als vorher
□ GrooveMetric: DTW ≤ 8 ms RMS (kein Timing-Flattening)
□ SOFT_SATURATION: wird nicht als CLIPPING detektiert
□ DSP-Fallback für jeden optionalen ML-Import (try/except ImportError)
□ `ml_memory_budget.try_allocate(name, size_gb)` aus `backend/core/ml_memory_budget` vor jedem ML-Modell-Laden — Budget-Überschreitung → `return None` + DSP-Fallback (KEIN `plm.try_allocate()` — existiert nicht)
□ `ml_memory_budget.release(name)` in allen Fehler-Paths nach fehlgeschlagenem Load
□ `PluginLifecycleManager.register(name, size_gb, unload_fn)` unmittelbar nach erfolgreichem Modell-Laden (LRU-Tracking)
□ models/manifest.json: sha256 + bundled_path + size_gb + fallback bei neuen Modellen
□ Tests als `ml` / `slow` markieren wenn Timeout ≥ 30 s oder Heavy-ML — blockiert nicht Standard-Suite
□ OQS ≥ 80 auf mindestens einem AMRB-Szenario nachweisbar
□ quality_estimate ≥ 0.55 im E2E-Test (result.quality_estimate-Assertion)
□ goal_applicability in RestorationResult gespeichert (GoalApplicabilityFilter-Ergebnis)
□ deferred_phases in RestorationResult vorhanden (list[str], default=[]) — §2.38 KMV
□ CHANGELOG.md Eintrag
□ Alle bestehenden Tests weiterhin grün (aktuell ~7750+ Pytest-IDs)
```

## Anti-Parallelwelten-Workflow (Pflicht vor jeder Implementierung)

```
1. Suche in backend/core/, plugins/, dsp/ nach vorhandener Funktionalität
2. Prüfe existierende Plugins (specs/08) und Phasen (specs/06)
3. Falls vorhanden → einbinden + DSP-Fallback, KEIN neues Modul
4. Falls nicht → neues Modul nach Singleton-Pattern anlegen
5. Entscheidung im CHANGELOG.md dokumentieren
```

## Performance-Budget (Desktop, kein GPU)

| Operation | Limit / Minute Audio |
|---|---|
| DefectScanner | ≤ 4 s |
| Phase-Pipeline gesamt | ≤ 240 s |
| FeedbackChain (alle Iter.) | ≤ 120 s |
| ExcellenceOptimizer | ≤ 60 s |
| RestorabilityEstimator | ≤ 5 s |

- Interne Verarbeitungs-SR (Phasen 01–56, Plugins): stets **48 000 Hz**
- **Analyse-Module** (DefectScanner, classify_clipping, RestorabilityEstimator, EraClassifier, MediumClassifier): arbeiten bei **nativer Import-SR** — kein Resampling vor Analyse, kein `assert sr == 48000`
- **§9.1a Nicht-stationäre Defekte** (DROPOUTS, TRANSPORT_BUMP): MÜSSEN auf **vollständigem Audio** analysiert werden — kein 60 s Center-Crop. Stationäre Defekte (Rauschen, Brummen, Flutter) dürfen Center-Crop nutzen.
- **§6.2a Material-Prioritäts-Phasen**: Phasen aus Spec 05 §6.2 MÜSSEN bei erkanntem Material **unbedingt aktiviert** werden — unabhängig vom Severity-Score. Phase entscheidet intern über Reparaturbedarf.
- **§6.4a Material-adaptive Schwellen**: DefectScanner-Erkennungsschwellen material-abhängig (Analog 20 % median-RMS für Dropouts, Digital 10 %).
- **§9.1b Intro-Salienz-Gewichtung**: Defekte in den ersten 5 s erhalten Severity-Boost (×1.5) — erste Sekunden bestimmen Hörerurteil (Zacharov & Koivuniemi 2001).
- **§9.1c Perceptual-Salience-Annotation**: Jeder Defekt wird mit psychoakustischem Salienz-Score annotiert (Fastl & Zwicker 2007). Maskierte Defekte erhalten reduzierte Severity: `severity * (0.3 + 0.7 * mean_salience)`. Modul: `backend/core/perceptual_salience.py`.
- Alle ONNX-Sessions: `providers=["CPUExecutionProvider"]`
- Torch-Modelle: `model.to("cpu")`; `torch.set_num_threads(os.cpu_count())`
- MERT (3,9 GB) / AudioSR (5,9 GB): nur Lazy-Load bei Bedarf

### [RELEASE_MUST] PerformanceGuard — RT-Budget-System (v9.10.72)

- `LIMIT_BALANCED = 32.0` (32× Echtzeit, Standard-Modus), `LIMIT_QUALITY = 32.0` (Restoration), `LIMIT_MAXIMUM = 32.0` (Studio 2026 / volle ML-Chain)
- Absolutes Zeitlimit Stufe 1: **90 Minuten** (`MAX_ABSOLUTE_SECONDS = 5400.0`). Danach: KMV Stufe 2 übernimmt automatisch.
- Begründung der Anhebung: 20-min Vinyl × SGMSE+(5× RT) + BsRoformer(3× RT) + 25 weitere Phasen → Ziel-RT ca. 28–32×; altes Limit 15–20× führte zu exzessiver Deferral auch bei normalen Songs.
- Überschreitung in Stufe 1 → DSP-Fallback **+ Phase in `deferred_phases` eintragen** (kein endgültiger Abbruch)
- `LIMIT_BACKGROUND = float("inf")` — ausschließlich für `MLRefinementThread` (§2.38 KMV Stufe 2)
- `RT8_EXCELLENCE_BUDGET = 32.0` (Benchmark-Gate-Referenz, RT-Kappe für reported rt_factor im Output). Kein Silence-Padding.

### [RELEASE_MUST] §2.38 Kontinuierliche ML-Veredelung (KMV) — Vollqualitäts-Garantie

**Kernprinzip**: RT-Limit-Überschreitung führt nie zu dauerhaftem Qualitätsverlust im Export.

| Stufe | Thread | RT-Limit | Ergebnis |
|---|---|---|---|
| **Stufe 1** | `BatchProcessingThread` | `LIMIT_BALANCED/QUALITY/MAXIMUM` | Sofort-Export (DSP-Fallback wo RT überschritten), listenable |
| **Stufe 2** | `MLRefinementThread` | `LIMIT_BACKGROUND = ∞` | Automatischer Overwrite mit voller ML-Qualität |

**Stufe-2-Startbedingungen** (alle müssen erfüllt sein):
- `len(result.deferred_phases) > 0` (Stufe 1 hatte RT-überschrittene Phasen)
- `psutil.virtual_memory().available / 1024**3 ≥ 4.0` (RAM-Guard)
- Kein anderer `MLRefinementThread` aktiv (Single-active-Invariante)

**Ablauf Stufe 2**:
1. `DeferredRefinementJob` erstellen (gecachte Analyse-Ergebnisse aus Stufe 1, `no_rt_limit=True`)
2. `MLRefinementThread` starten: `Priority = QThread.LowPriority`, `os.nice(10)` auf Linux
3. Vollständige UV3-Pipeline (kein Neustart DefectScanner/EraClassifier/MediumClassifier — gecacht)
4. `isInterruptionRequested()` zwischen jeder Phase prüfen (Escape-Abbruch möglich)
5. Qualitätsinvariante: `stufe2_quality_estimate ≥ stufe1_quality` → sonst Overwrite verweigern
6. Atomischer Overwrite: `output.tmp` → `os.replace(output_path)`

**`DeferredRefinementJob`-Pflicht-Felder**: `output_path`, `audio_original`, `sr`, `mode`, `deferred_phase_ids`, `cached_defect_result`, `cached_era_result`, `cached_medium_result`, `stufe1_quality`

**`MLRefinementThread` Pflicht-Signale**:
```python
refinement_started(str, int)      # output_path, n_deferred_phases
refinement_phase_done(str, float) # phase_id, quality_improvement_delta
refinement_progress(int, str)     # pct 0–100, phase_name
refinement_complete(str, object)  # output_path, final_RestorationResult
refinement_cancelled(str)         # output_path → Stufe-1-Export bleibt
```

**UI (§11.4 Ergänzung)**:
- Stufe-2-Indikator: `refinement_progress_bar` (3 px, türkis `#00BCD4`, unter `phase_progress_bar`), sichtbar nur während Stufe 2
- Status-Text: `"ML-Veredelung: 3/5 Phasen verbessert..."`, nach Fertigstellung: `"Export vollständig restauriert ✓ — ML-Qualität"` (Notification, 5 s)
- Escape-Handling: `requestInterruption()` trifft **sowohl** `BatchProcessingThread` **als auch** aktiven `MLRefinementThread`

**RestorationResult-Pflicht-Felder** (§2.38):
```python
deferred_phases:         list[str] = field(default_factory=list)  # Phasen für Stufe 2
refinement_complete:     bool = False                              # True nach ML-Veredelung
stufe2_quality_estimate: Optional[float] = None                   # quality nach vollst. ML-Pass
```

**Memory**: `DeferredRefinementJob.audio_original` via `ml_memory_budget.try_allocate("kmv_job", size_gb)` registrieren; nach Stufe-2-Export oder Abbruch `release("kmv_job")`.
**VERBOTEN**: `LIMIT_BACKGROUND` für `BatchProcessingThread` verwenden.

### [RELEASE_MUST] `ml_memory_budget` — Zentrale OOM-Schutzschicht

`backend/core/ml_memory_budget.py`: `try_allocate(name, gb)` VOR jedem `InferenceSession`/`torch.load()`.
Auto-Budget: `max(4.0, min(12.0, RAM_GB/3))`. Überschreitung → DSP-Fallback + `release(name)`.
> **Warnung `psutil`-Ausfall**: Fehlt `psutil`, keine physischen RAM-Checks. `psutil` MUSS im AppImage gebündelt sein.

### §2.37 [TARGET_2026] CPU-Aware Pipeline Scheduling

- `torch.set_num_threads(os.cpu_count())`, ONNX: `intra_op_num_threads=os.cpu_count(), ORT_ENABLE_ALL`.
- OOM-Schutz Schicht 1: `ml_memory_budget.try_allocate()` / Schicht 2: `PluginLifecycleManager.register()` (LRU, >82% RAM).
- `VERBOTEN`: `plm.try_allocate()` — existiert nicht. Nur `ml_memory_budget.try_allocate()`.

## DSP-Spezialregeln

- **MRSA-Zonen (5 Pflicht-Zonen)**: sub_bass win=65536 / mid_low 16384 / mid 8192 / presence 1024 / air 128 — PGHI per Zone, Kreuzfade Hanning 10 ms. VERBOTEN: willkürliche FFT-Größen.
- **Dithering Export**: POW-r Typ 3 (primär) → TPDF (fallback). VERBOTEN: Truncation ohne Dithering.
- **Print-Through (Phase 29, reel_tape)**: Bidirektionale LMS — alpha_pre ≠ alpha_post. VERBOTEN: Comb-Filter oder symmetrisches Modell.
- **§2.12 PolyphonicSpeedCurveEstimator** (`quality_mode=maximum`): BasicPitch ONNX → Konfidenz-gewichteter Median ≥ 2 Voices → Savitzky-Golay. try_allocate("BasicPitch", 0.12) Pflicht. GrooveMetric-DTW ≤ 8 ms RMS nach Korrektur.
- **Perceptuelle Pflicht-Messwerte**: LUFS-Diff ≤ 1 LU | Chroma Pearson ≥ 0.95 | Groove DTW ≤ 8 ms RMS | Transient Attack ≤ ±2 ms | MERT-Harmonizität ≥ 0.85.

**Pfad-Mapping**: `core/<modul>.py` → `backend/core/`, `plugins/<plugin>.py` → `plugins/`, Frontend/UI → `Aurik910/` (kein `frontend/`-Verzeichnis!). Import: `from Aurik910.i18n import t, set_language`.

## Restaurierungs-Modi

| Modus | Ziel | LUFS | TonalCenter |
|---|---|---|---|
| **Restoration** | Originalgetreue Restauration | Δ ≤ 1 LU | ≥ 0.95 |
| **Studio 2026** | Highend-Studio-Klang | −14 LUFS EBU R128 | ≥ 0.97 |

Studio 2026: Stem-Sep → Vocal-AI → Instrumente → [Reference Mastering] → Multibandkomp → Präsenz/Air → Stereo-Imaging → Re-Mix (StemRemixBalancer) → LUFS-Norm → TruePeak → Musical Goals → [Vocos-Synthese (MOS < 4.3)]

**StemRemixBalancer** — Pflicht nach Stem-Verarbeitung: Verboten: nacktes `vocals + instruments` in UV3. Algorithmus (6 Schritte): L_orig messen → vocal_weight via PANNs → LUFS pro Stem → Gain-Korrektur → Re-Mix → Final-Check |LUFS(mix) − L_orig| ≤ 0.3 LU. Pflicht-Test: `tests/unit/test_stem_remix_balancer.py` (≥ 20 Tests).

## Universelle Garantien (§8.2 — PFLICHT)

| Garantie | Schwellwert |
|---|---|
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) ≤ 1.0` |
| Chroma-Korrelation (Tonart) | Pearson ≥ 0.95 |
| Pass-Through (SNR > 40 dB) | PQS-MOS-Verlust ≤ 0.05, LUFS ≤ 0.3 LU, Chroma ≥ 0.99 |
| Rauschboden (Studio-2026) | ≤ −72 dBFS, A-gew. ≤ −75 dB(A) |
| Temporale Kohärenz | MOS-Spanne ≤ 0.30, σ ≤ 0.15 |
| Stereo-Authentizität | Mono-Ären: M/S-Korrelation ≥ 0.97 |
| HF-Kumulativ-Limit | Presence + Air ≤ +4 dB |
| Mikro-Dynamik | Pearson LUFS-Profil (400 ms) ≥ 0.92 |
| Emotionaler Dynamik-Bogen (≥ 30 s) | Arousal-Pearson ≥ 0.85, Valence-Pearson ≥ 0.80 |
| Emotionaler-Bogen-Korrektur (post-MDEM) | `correct_emotional_arc()` — Makro-Gain-Korrektur bei Bogen-Degradation |
| FeedbackChain-Rollback | |MOS_neu − MOS_alt| > 0.05 → sofortiger Rollback |

## Psychoakustik & Gänsehaut-Prinzipien (§8.3 — bindend)

**Oberstes Ziel**: Das Restaurierungsergebnis muss beim Hörer **emotionale Wirkung** erzeugen — nicht nur technische Korrektheit. Die Psychoakustik steht über der technischen Perfektion.

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Komponente | Verantwortliches Modul | Anteil |
|---|---|---|
| **Transient-Punch** | TDP (Transient Decoupled Processing) | ~40 % |
| **Mikro-Dynamik-Erhalt** | MDEM (400 ms LUFS-Morphing) + EmotionalArcCorrection (5 s Makro-Bogen) | ~25 % |
| **Rauschbefreiung/Klarheit** | SGMSE+ / OMLSA/IMCRA | ~20 % |
| **Vokal-Präsenz** | Phase 42 + Phase 43 + VocalAIEnhancement | ~10 % |
| **Neurale Synthese** | Vocos 48 kHz (nur Studio 2026, MOS < 4.3) | ~5 % |

**Zwei-Skalen-Dynamik-Schutz** (Pflicht):
- **Mikro-Ebene (400 ms)**: MDEM — `morph(restored, original, sr)` — LUFS-Profil-Rückgewinnung
- **Makro-Ebene (5 s)**: `correct_emotional_arc(original, restored, sr)` — post-MDEM, wenn Arousal/Valence-Bogen abgeflacht. Algorithmus: RMS-basierte Gain-Hüllkurve, ±6 dB, 70 % Dämpfung, Savitzky-Golay-geglättet. Safety-Revert wenn Korrektur verschlechtert.

**Defect-Locations-Flow** (§9.1 Ergänzung):
- `_execute_pipeline` extrahiert `defect_locations: dict[str, list[tuple[float, float]]]` + `max_defect_severity: float` aus `defect_result.scores`
- Beide werden als `kwargs` an jede Phase übergeben (`defect_locations=`, `max_defect_severity=`)
- Phasen können Location-Hints für gezieltere Verarbeitung nutzen (opt-in)
- Phasen erkennen Defekte weiterhin auch eigenständig intern (Redundanz-Prinzip)

**Psychoakustische Pflicht-Invarianten**:
- Intro-Salienz (§9.1b): Defekte in den ersten 5 s → Severity ×1.5
- Perceptual Masking (§9.1c): Maskierte Defekte → `severity * (0.3 + 0.7 * salience)`
- Emotionaler Bogen: Messung + aktive Korrektur (nicht nur Logging)
- Vintage-Harmonische: H2/H4 bewahren, Soft-Saturation ≠ Clipping

## Sprachkonvention

- **Nutzer-Meldungen, UI-Texte, Fehlermeldungen**: **Deutsch**
- **Code-Kommentare, Docstrings**: **Englisch**
- **Log-Meldungen** (Logger): **Englisch**

Fehlermeldungen immer mit **Ursache** + **Lösungsvorschlag** auf Deutsch.

## Frontend-UX-Pflichtregeln (§11.4 — `ModernMainWindow`)

### Progress Bar
- **`setRange(0, 10000)`** immer — 1 Einheit = 0,01 %; Signale: 0–100, Slot skaliert `v * 100`
- `setValue(10000)` = Completion. VERBOTEN: `setRange(0, 100)` in `ModernMainWindow`

### Thread-Safety (absolutes Verbot)
- **Kein Qt-Widget-Zugriff aus Hintergrundthreads.** Pattern: `_gui_dispatch = pyqtSignal(object)`, connect `lambda fn: fn()`.
- Hintergrund: `self._dispatch_to_gui(lambda: widget.setText(...))` oder `QTimer.singleShot(0, fn)`

### Shortcuts (`_setup_shortcuts`)
`Space` Play/Pause | `A` Original | `B` Restauriert | `Ctrl+O` Öffnen | `Ctrl+S` Export | `Ctrl+R` Restoration | `Ctrl+Shift+R` Studio 2026 | `Escape` Abbruch | `Ctrl+Z` Pfad-Clipboard | `L` Lyrics-Overlay

### Watchdog-Timer
`QTimer(self)`, `setSingleShot(True)`. Timeout-Formel: `_per_file_ms = max(5_400_000, int(audio_dur_s * 32_000) + 1_800_000)` → `_watchdog_ms = max(5_400_000, n_files * _per_file_ms)` (Minimum **90 Min.**). Start vor `batch_thread.start()`. Callback: `requestInterruption()` → `wait(3000)` → `terminate()`.

### Bridge-Fallback (`_BRIDGE_AVAILABLE`)
`from backend.api.bridge import ...` in `try/except ImportError` wrappen. Bei Fehler: `_BRIDGE_AVAILABLE = False` + Stubs. `_export_guard`-Stub vollständig implementieren (NaN-Guard + Clip). Alle anderen Stubs: `return None`.

### BatchProcessingThread — Signal-Kontrakt (Kurzform)
`item_started(str)` | `item_progress(str, int 0–100)` | `item_finished(str)` | `item_finished_with_result(str, object)` | `item_error(str, str)` | `all_finished()` | `defect_update(dict)` | `phase_update(str)` | `waveform_data(ndarray, int)` | `mode_update(str)` | `ml_status_update(bool, list)` | `phase_progress(int 0–100)` | `scan_progress(float 0.0–1.0)` | `quality_update(float 0.0–5.0)`
- `progress_callback`-Signatur: `(pct: int, msg: str, elapsed_s: float = 0.0) → None`
- `phase_progress` → `phase_progress_bar.setValue(v * 100)` (5 px, lila Gradient, unter Hauptleiste)
- `scan_progress` → `waveform_widget.set_scan_pos(frac)` (oranger gestrichelter Cursor mit Glow)
- `quality_update` → `quality_meter_widget.set_mos(mos)` (steigt 2.5 → 4.2 während Verarbeitung)

### §11.4a Echtzeit-UX-Features (ab 9.10.57)

| # | Feature | Implementierung |
|---|---|---|
| 1 | **Zweistufiger Fortschrittsbalken** | `phase_progress_bar` (5 px, `QProgressBar`, lila Gradient) unter `progress_bar`; Sichtbarkeit: bei Batch-Start ein, bei `_on_all_finished` aus |
| 2 | **Defekte hochzählen/herunterzählen** | `_update_defects`: bei `status=="detected"` Count-up-Animation (QTimer, 22 Frames × 85 ms via `_tick_defect_reveal`); `_PHASE_REDUCES`-Mapping senkt Defekt-Scores × 0.3 bei passenden Phasen-Keywords |
| 3 | **Varianten-Wettkampf** | `multi_pass_strategy.process_with_variants()` emittiert nach jeder Variante `"Variante X/N: 'name' → MOS 4.12 ✓"`; `_on_batch_progress` baut Rangliste `★name_1 (4.12) › name_2 (3.87)` |
| 4 | **Musical-Goals-Meter live** | `quality_update`-Signal verbunden mit `quality_meter_widget.set_mos()`; startet bei 2.5, steigt proportional zum Fortschritt auf 4.2 |
| 5 | **Phasen-Erklärungstext** | `_PHASE_EXPL`-Dict (22 Einträge) mappt Phasen-Keywords auf Kurzbeschreibungen; wird als `[Kontext]` in Statuszeile angehängt |
| 6 | **Waveform-Scan-Cursor** | `WaveformWidget._scan_pos`; oranger Cursor (12 px Glow rgba(255,150,30,45) + 2 px DashLine rgba(255,178,55,215)); `set_scan_pos(-1.0)` blendet aus; Reset in `_on_all_finished` |
| 7 | **Live-Qualitätszahl** | `quality_meter_widget` wird bei Batch-Start sichtbar (`set_mos(2.5)`); steigt mit `scan_progress` |
| 8 | **Vorab-Hörprobe** | `QTimer.singleShot(1400, self._auto_preview_restored)` in `_on_item_finished_with_result`; spielt erste 5 s (= 5×48000 Samples) des restaurierten Audios; nur wenn kein anderer Playback läuft |

### Async-Analyse-Kette nach Datei-Öffnen
4 Daemon-Threads: `_bg_load` (3-stufige Audio-Kaskade) → `_carrier_bg` → `_detect_era_genre_bg` → `_estimate_restorability_bg`. Alle via `_dispatch_to_gui` oder `QTimer.singleShot(0, ...)`. DefectScan erst in `BatchProcessingThread`.

### Bridge-Funktionen (vollständige Liste)
`export_guard` | `get_audio_file_validator` | `get_defect_scanner` | `get_defect_type` | `get_quality_mode` | `get_restorer_classes` | `get_medium_classifier_fn` | `get_era_classifier_fn` | `get_genre_classifier_fn` | `get_restorability_estimator_class` | `get_carrier_forensics_fn` | `get_audio_exporter_class` | `cache_defect_result` | `get_cached_defect_result` | `clear_defect_cache` | `warmup_models_background`

## §2.36 LyricsGuidedEnhancement (ab 9.10.x — PFLICHT)

Whisper-Tiny ONNX (39 MB) → Phonem-Alignment via wav2vec2_forced_alignment.onnx (125 MB) → Timeline-Segmentierung (vowel_stressed / fricative / plosive / silence) → ContentAwareProcessor (Salienz-Boosts pro Phonemklasse, Stille → aggressivere NR). Latenz ≤ 8 s/min Audio. Shortcut L (Overlay). **Datenschutz-Pflicht**: Lyrics-Text niemals in Logs oder `RestorationResult.metadata`.

## ML-Plugin-Status (verifiziert, März 2026)

> Vollständige Plugin-Matrix (28 Plugins, Modellpfade, Format, Aufgabe, Fallback-Kaskaden): `.github/specs/08_architecture_and_distribution.md`

**Pflicht-Invarianten für alle ML-Plugins**:
- `ml_memory_budget.try_allocate(name, size_gb)` VOR jedem `InferenceSession`/`torch.load()` — bei Fehler DSP-Fallback
- `ml_memory_budget.release(name)` in allen Fehler-Pfaden nach fehlgeschlagenem Load
- `PluginLifecycleManager.register(name, size_gb, unload_fn)` nach erfolgreichem Load
- VERBOTEN: `plm.try_allocate()` (existiert nicht)

**Lazy-Load-Pflicht** (Budget überschreitet 4 GB allein): AudioSR (5,9 GB), MERT-v1-330M (3,9 GB).
**MelBandRoformer** (860 MB, ONNX): 48k→44.1k→48k Resampling (Lanczos-4, SNR ≈ −0.8 dB) — bei 48k-nativem Modell dieses bevorzugen.
*Diese Richtlinien gelten für alle KI-Agenten (GitHub Copilot, Claude, GPT-Instanzen) die an Aurik 9 arbeiten. Vollständige normative Spezifikation: `.github/specs/01–08`.*
*Stand: März 2026 — Aurik 9.10.57*
