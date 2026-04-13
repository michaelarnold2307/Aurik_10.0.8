# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.11.8 ist ein erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: April 2026 — Version **9.10.121**
>
> **instructions_version: 5.0** — komprimiert 06.04.2026
>
> Bump-Regel: neue RELEASE_MUST-Zeile, neues Gate oder §-Änderung → `instructions_version` inkrementieren + `docs/CHANGELOG_HISTORY.md` Eintrag.
>
> Aktuelle Testzahl: **~9990+ `def test_`-Funktionen** (375 Testdateien; alle grün)
>
> Stand: 6. April 2026 — §2.41 Denker-Vollkontext + §2.42 SourceFidelityReconstructor + §2.43 Phase-Preserved Wet/Dry-Blend + §9.7.15 Musical-Goals-Metriken-Recalibration (v9.10.113–121).
>
> **§2.36 `LyricsGuidedEnhancement`** ist ab Version **9.10.x Pflicht** (bisher v10.0-Label entfernt).

## Vollständige Spezifikation

Die vollständige normative Spezifikation ist in 8 Spec-Dateien aufgeteilt:

| # | Datei | Inhalt |
| --- | --- | --- |
| 1 | `.github/specs/01_musical_goals.md` | 14 Musical Goals, Schwellwerte, PMGG, MDEM |
| 2 | `.github/specs/02_pipeline_architecture.md` | Kanonischer Pipeline-Ablauf, RestorationResult, alle §2.x-Module |
| 3 | `.github/specs/03_cognitive_modules.md` | Kernmodule §2.1, Singleton-Pattern §3.x, Logging, Cache |
| 4 | `.github/specs/04_dsp_standards.md` | SOTA-Entscheidungsmatrix §4.4, DSP-Mindeststandards §4.5 |
| 5 | `.github/specs/05_material_system.md` | Materialien §6.x, DefectTypes §6.3, GP-Gedächtnis |
| 6 | `.github/specs/06_phases_system.md` | Phase 01–64 §7.x, CAUSE_TO_PHASES-Mapping |
| 7 | `.github/specs/07_quality_and_tests.md` | Qualitätsziele §8.x, Test-Standards §5.x, E2E §14 |
| 8 | `.github/specs/08_architecture_and_distribution.md` | Schichten §11.x, Distribution §13.x, Out-of-the-Box |

Änderungshistorie: `docs/CHANGELOG_HISTORY.md`

## Normative Priorisierung (P1-1 MUST/TARGET-Split)

`[RELEASE_MUST]` = Harte Release-Bedingung; CI-Gates nicht skippbar. `[TARGET_2026]` = Roadmap 2026, kein Release-Blocker außer explizit als Gate markiert. (Pflicht/bindend/verboten → RELEASE_MUST; Studio 2026/Weltklasse → TARGET_2026.)

Gate-zuordnung (aktuell implementierte CI-Guards):

| Bereich | Einstufung | Test-ID / Gate |
| --- | --- | --- |
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
| §2.39 OOM-Recovery-Checkpoint (Checkpoint-Save + Startup-Resume) | `[RELEASE_MUST]` | `tests/unit/test_recovery_checkpoint.py` |
| Vollpipeline-Determinismus (bitnahe Reproduzierbarkeit) | `[RELEASE_MUST]` | `tests/normative/test_full_pipeline_determinism.py` |
| Stratifiziertes Konkurrenz-Gate (Material x Defektklasse) | `[RELEASE_MUST]` | `tests/normative/test_competitive_stratified_gate.py` |
| Externer Mini-MUSHRA bei Kern-aenderungen | `[RELEASE_MUST]` | `tests/normative/test_external_mushra_artifact_contract.py` |
| §2.29b PMGG Stable-Metric-Invariante (`NatuerlichkeitMetric` nie in `_PRECISE_METRICS`) | `[RELEASE_MUST]` | `tests/unit/test_per_phase_musical_goals_gate.py` (122 Tests — `PHASE_GOAL_EXCLUSIONS` + Audio-Cap 2.5 s + Groove-Proxy §9.7.10 + K-S tonal_center §9.7.11 + Restorative-Baseline §2.29c) |
| §2.29c PMGG Restorative-Phase-Baseline-Capping (Defekt-inflationierte Baselines gedeckelt) | `[RELEASE_MUST]` | `tests/unit/test_per_phase_musical_goals_gate.py` — `_RESTORATIVE_PHASES` + `_CANONICAL_THRESHOLDS` + `effective_scores_before` |
| §3.9 Stabilitäts-Invarianten (9 Punkte: Inference-Timeout, SIGTERM, Phase-Output-Guard, Executor-Lifecycle, Budget-Reconciliation, Exception-Logging, Buffer-RAM-Guard, Lock-Order, KMV-Buffer-Release) | `[RELEASE_MUST]` | `tests/normative/test_stability_invariants.py` — je Invariante mind. 3 Tests |
| OQS ≥ 88 / Weltklasse-Ziele | `[TARGET_2026]` | Roadmap/Benchmark-Ziel, kein harter Release-Blocker |
| AMRB-basierter MUSHRA-Hörertest (ITU-R BS.1534-3) | `[TARGET_2026]` | Extern validiert 14 Goal-Schwellwerte; ersetzt „best engineering estimate"-Status; geplant nach OQS ≥ 84.0-Erreichung |

## [RELEASE_MUST] Projektgrenzen (bindend, keine Ausnahmen)

- **Reine Desktop-App** für Linux (AppImage) und Windows 10/11 (.exe)
- **Kein Cloud, kein Server, kein Docker, kein `pip install`** für Endnutzer
- **Out-of-the-Box-Pflicht**: Läuft auf frischem System ohne Python/Terminal
- **100 % offline** nach Installation — alle ML-Modelle lokal gebündelt
- Nur **Mono und Stereo** unterstützt (> 2 Kanäle → PANNs-gewichteter Downmix)
- **Kein Fremdedit am Original-Audio** — immer neue Ausgabedatei in `output/`

### [ERGÄNZUNG] OOM-Checkpoint-Ausnahme

Falls das Original-Audio nach einem OOM-Checkpoint nicht mehr verfügbar oder lesbar ist (z.B. Hardwarefehler), darf ausnahmsweise das im Checkpoint gespeicherte Audio als Quelle für die Wiederaufnahme verwendet werden. Dies ist explizit als Ausnahme zur Regel „kein Phase-Skip auf Original-Audio“ zu verstehen und MUSS im Log als Notfall dokumentiert werden. Qualitätsverluste sind in diesem Fall zulässig, aber zu minimieren.

## [RELEASE_MUST] 14 Musical Goals (Mode-differenzierte Schwellwerte, v9.10.77)

| Ziel | Klasse | Prio | Restoration | Studio 2026 |
| --- | --- | --- | --- | --- |
| Natürlichkeit | `NatuerlichkeitMetric` | P1 | ≥ 0.90 | ≥ 0.90 |
| Authentizität | `AuthentizitaetMetric` | P1 | ≥ 0.88 | ≥ 0.88 |
| Tonales Zentrum | `TonalCenterMetric` | P2 | ≥ 0.95 | ≥ 0.97 |
| Timbre-Authentizität | `TimbralAuthenticityMetric` | P2 | ≥ 0.87 | ≥ 0.87 |
| Artikulation | `ArticulationMetric` | P2 | ≥ 0.85 | ≥ 0.85 |
| Emotionalität | `EmotionalitaetMetric` | P3 | ≥ 0.82 | ≥ 0.87 |
| Mikro-Dynamik | `MicroDynamicsMetric` | P3 | ≥ 0.88 | ≥ 0.92 |
| Groove | `GrooveMetric` | P3 | ≥ 0.83 | ≥ 0.88 |
| Transparenz | `TransparenzMetric` | P4 | ≥ 0.82 | ≥ 0.89 |
| Wärme | `WaermeMetric` | P4 | ≥ 0.75 | ≥ 0.80 |
| Bass-Kraft | `BassKraftMetric` | P4 | ≥ 0.78 | ≥ 0.85 |
| Separation-Treue | `SeparationFidelityMetric` | P4 | ≥ 0.78 | ≥ 0.82 |
| Brillanz | `BrillanzMetric` | P5 | ≥ 0.78 | ≥ 0.85 |
| Raumtiefe | `SpatialDepthMetric` | P5 | ≥ 0.70 | ≥ 0.75 |

### [ERGÄNZUNG] Materialklassifikations-Konflikte

Falls EraClassifier und MediumClassifier widersprüchliche Materialtypen liefern (z.B. Tape vs. Vinyl), gilt folgende Konfliktregel: Priorität hat der Klassifikator mit höherer Konfidenz. Bei Gleichstand entscheidet die DefectScanner-Auswertung (höchster Defekt-Score für ein Material). Ist auch dies unklar, wird der konservativere (restaurierungsschonendere) Materialtyp gewählt. Der Entscheidungsweg MUSS im Log dokumentiert werden.

> **v9.10.77 Pareto-Differenzierung**: Restoration-Modus senkt P3–P5-Schwellwerte auf physikalisch erreichbare Werte (Pareto-Konflikte: Bass↔Transparenz [0.7], Brillanz↔Wärme [0.6]). P1/P2 bleiben identisch. Studio 2026 behält ambitionierte Ziele.
> **SpatialDepthMetric**: IACC (Blauert 1997), < 0.70 → Phantom-Center-Zusammenbruch. Mono-Ären: via GoalApplicabilityFilter deaktiviert. Alle 14 Schwellwerte AMRB-kalibriert; MUSHRA-Test ausstehend.

**Sub-Metriken (Pflicht-Implementierungsdetails):**

- `TimbralAuthenticityMetric`: MFCC-Pearson ≥ 0.95,Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 %
- `ArticulationMetric`: Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms
- `TonalCenterMetric`: Chroma-Korrelation ≥ 0.95 **und kein Key-Shift > 0 Cent** (absolut tonarterhaltend)
- `BrillanzMetric` / `WaermeMetric`: Frequenzgewichtung nach **ISO 226:2023 Equal-Loudness** — kein lineares Energiemessen
- `BassKraftMetric`: enthält Virtual Pitch (Missing Fundamental) via Oberton-Analyse 120–500 Hz
- `SeparationFidelityMetric`: SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition

**Invariante**: Jede Restaurierungsoperation darf keines der 14 Ziele verschlechtern. Pflicht-Check: `MusicalGoalsChecker(mode=mode).measure_all(audio, sr)` → `assert all(scores[g] >= t for g, t in checker.thresholds.items())`.

**PMGG Priority-Aware Retries (§2.29 v9.10.77 + §2.31b v9.10.85)**:

- P1/P2-Regression: Volle Retry-Kaskade (4 Retries + Emergency); Catastrophic-Threshold = `max(0.08, 4.0 × adaptive_threshold)`
- P3-Regression: Basis 2 Retries, 1.5× Toleranz; `restorability_tier="good"` → 3 Retries; `tier="poor"` → 1 Retry
- P4/P5-Regression: Kein Retry — nur Logging (`passed_p4p5_tolerated`)
- `initial_strength < 0.90` (SongCal-reduziert) → Retry-Ankerpunkte `[0.80, 0.65, 0.50, 0.35, 0.20]`
- Stagnation-Abbruch: `max(0.002, threshold × 0.15)` (proportional)

## [RELEASE_MUST] Qualitätsmessung & Metriken-System (§8.1 — PFLICHT)

| Metrik | Hard-Fail-Minimum | Weltklasse-Ziel |
| --- | --- | --- |
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
| --- | --- | --- |
| Good (B) | ≥ 80 | **[RELEASE_MUST] Pflicht für jede neue Phase / Plugin** |
| Excellent (A) | ≥ 91 | Exzellenz-Label — kein harter Gate-Wert |

**[TARGET_2026]** Studio-2026-Ziel: OQS ≥ **88** (zwischen Good und Excellent). Kein Release-Blocker, aber Roadmap-Zielvorgabe für 2026.

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

### [ERGÄNZUNG] Endlosschleifen-Prävention bei Deferred-Phases

Falls Deferred-Phases (Stufe 2) auch nach 3 Wiederholungsversuchen wegen RT-Limit oder RAM-Mangel nicht erfolgreich abgeschlossen werden können, wird die Phase endgültig als „nicht nachholbar“ markiert und im RestorationResult protokolliert. Der Nutzer erhält eine Benachrichtigung mit Hinweis auf die betroffene Phase und den Grund. Weitere automatische Versuche unterbleiben, bis ein manueller Neustart erfolgt.

### Profi-/Studio-Qualitätsprinzipien

| Referenz | Verbindliche Übersetzung in Aurik |
| --- | --- |
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
| --- | --- | --- | --- |
| 6 | ReparaturDenker | Defekt-Beseitigung | Entfernt bekannte Störungen (Clicks, Hum, Clipping) |
| 7 | RekonstruktionsDenker | Rekonstruktion | **Erschafft, was fehlt** — füllt Lücken, annotiert Bandwidth-Verlust |
| 8 | RestaurierDenker | Restaurierung | **Bewahrt, was da ist** — orchestriert UV3, schützt Klangcharakter |

Kontextfluss: `defect_result → ReparaturDenker → RekonstruktionsDenker(+defect_result) → RestaurierDenker(+reconstruction_context) → UV3`

UV3-Kernreihenfolge: DCOffset-Removal → TDP (HPSS) → RestorabilityEstimator → SongCalibrationProfile → EraClassifier → GermanSchlagerClassifier → MediumClassifier → GoalApplicabilityFilter → AdaptiveGoalThresholds → DefectScanner (32 Defekte) → CausalDefectReasoner (material- und defektadaptive Kausal-Ursachen) → GPParameterOptimizer → HarmonicPreservationGuard → Phasen-Ausführung (01–64) → FeedbackChain → PhysicalCeilingEstimator → MusicalGoalsChecker → MicroDynamicsEnvelopeMorphing → RestorationResult

**Parallelisierungs-Invariante**: Tier 0+1 sequenziell; EraClassifier+GermanSchlager+MediumClassifier parallel (`ThreadPoolExecutor max_workers=3`); Tier 6 sequenziell.

> Details: `.github/specs/02_pipeline_architecture.md`, `denker/aurik_denker.py`

## Adaptive Qualitätsziele — Schlecht-Material-Strategie (§2.31–§2.34)

**Statische Schwellwerte VERBOTEN.** Vor jeder Restaurierung material-, ära- und restorability-adaptiv skalieren:
`get_adaptive_goals_and_config(audio, sr)` → skalierte Schwellwerte, GoalApplicabilityFilter, PhysicalCeilingEstimator.
Priority-Hierarchie: P1 (natürlichkeit, authentizität) → P2 (tonal_center, timbre, artikulation) → P3–5 best-effort.
P1+P2-Regression → `FeedbackChain`-Rollback; Ceiling Δ < 3 % → Terminierung.

### [RELEASE_MUST] §2.31a Ganzheitliche Song-Selbstkalibrierung (v9.10.83)

**Kernprinzip**: Keine globalen Fix-Strengths. Jede Restaurierung MUSS vor der Phasenkette ein Song-Kalibrierungsprofil berechnen und phasenübergreifend anwenden.

- Pflichtprofil `song_calibration_profile` mit mindestens: `material`, `mode`, `restorability_score`, `input_snr_db`, `max_defect_severity`, `pipeline_confidence`, `global_scalar`, `family_scalars`.
- `family_scalars` müssen mindestens Familien enthalten: `denoise`, `reverb`, `reconstruction`, `dynamics_eq`, `transient`, `vocal`, `instrument`, `general`.
- Kalibrierung MUSS begrenzt sein (`global_scalar` bounded), um Overfitting auf Einzelsongs zu vermeiden.
- Kalibrierung MUSS psychoakustisch priorisieren: P1/P2-Stabilität und Hörbarkeit (Maskierung/Transient-Integrität) vor aggressiver Defektentfernung.
- Kalibrierung MUSS Tonträgerkette/Material/Defektlage berücksichtigen; statische One-Size-Fits-All-Konfigurationen sind für Produktionspfade unzulässig.
- `RestorationResult.metadata["song_calibration"]` ist Pflicht zur späteren songs-übergreifenden Auswertung.

**Invariante**: Bei identischer Eingangsumgebung ist das Kalibrierungsprofil deterministisch reproduzierbar; unterschiedliche Songs dürfen bewusst unterschiedliche Skalare erhalten.

**Kalibrier-Berechnungsblöcke** (kanonische Reihenfolge in `_build_song_calibration_profile`):

1. Era-GP-Warmstart: ≤1940 →`_era_denoise_scale` ×1.10; ≤1960 →×1.00; ≥1970 →×0.88
2. Material-Multiplikatoren: shellac/vinyl/tape/cd_digital/mp3/cassette (6 Werte)
3. Per-Defekt-Family-Boost: 28 DefectTypes → 6 Familien, max +12 % je Familie
4. Spektral-Fingerprint: `rolloff_95_hz` → reconstruction; `noise_floor_p5_db` → denoise; `wow_flutter_index` → dynamics
5. SOFT_SATURATION-Vintage-Guard: sat_severity ≥ 0.25 → denoise −12 %, transient −7 %
6. Schlager-Profil: vocal +10 %, transient +5 %, dynamics +5 %, reconstruction ×0.95
7. Diversity-Penalty: ≥8 aktive Defekte (severity ≥ 0.18) → global −1 % je Extra-Defekt, max −6 %
8. PANNs: vocal_prob < 0.10 → vocal ×0.80; ≥0.35 → vocal ×0.97–1.10; inst_prob ≥ 0.35 → inst ×0.97–1.10
9. Modus-Post-Skalierung: studio/maximum → reconstruction ×1.08, transient ×1.05, vocal ×1.05, instrument ×1.05

### [RELEASE_MUST] §2.31b PMGG Song-Kalibrierungs-Integration (v9.10.85)

Alle 7 PMGG-Schnittstellen MÜSSEN das `song_calibration_profile` aktiv nutzen:

1. **Kalibrierungs-adaptiver Threshold** (`wrap_phase`): `global_scalar < 0.85` → Threshold ×0.85 (engerer Schutz); `global_scalar > 1.20` → Threshold ×1.15 (reduziert Retry-Verschwendung). Begrenzt [0.015, 0.070].
2. **Sanftere Retry-Leiter** (`_run_with_retry`): `initial_strength < 0.90` (SongCal hat vorreduziert) → Ankerpunkte `[0.80, 0.65, 0.50, 0.35, 0.20]` statt `[0.65, 0.50, …]`. Verhindert destruktive Doppelreduktion.
3. **Proportionale Stagnation-Schwelle** (`_run_with_retry`): `max(0.002, threshold × 0.15)` — GOOD: 0.003 (geduldiger); POOR: 0.008 (bricht früher ab).
4. **P3-Retry-Budget nach `restorability_tier`** (`_run_with_retry`): `tier="good"` → P3 2→3 Retries; `tier="poor"` → P3 2→1. P1/P2/P4/P5 unverändert.
5. **FeedbackChain `target_score`** (`_fc_compute_target_score`): Base 0.72/0.78 (Restoration/Studio) ±0.035 nach `restorability_score`. Begrenzt [0.60, 0.85].
6. **[RELEASE_MUST] Dynamischer Catastrophic-Threshold** (`_run_with_retry`): `max(0.08, 4.0 × adaptive_threshold)`. GOOD (0.020) → 0.08 (Emergency früher, mehr Qualitätsschutz); POOR (0.055) → 0.22 (wie bisher).
7. **[RELEASE_MUST] Material-adaptive PHASE_GOAL_EXCLUSIONS** (`wrap_phase`): `cd_digital`/`dat` haben kein Breitbandrauschen → HF-bedingte Falsch-Regressionen treten bei phase_03/phase_29 nicht auf. Reduktion auf `{"natuerlichkeit", "artikulation"}`. Statische Ausschlüsse (CREPE-Load-State, transient-shape mismatch, timbre_authentizitaet) bleiben materialunabhängig.

## Algorithmische Pflicht-Mindeststandards

Primär → Fallback1 → DSP-Fallback-Kaskade für alle Anwendungsfälle.
**VERBOTEN** als Musikmetrik: `PESQ`, `DNSMOS`, `NISQA`, `STOI`, `ViSQOL --speech`, `CDPAM`.
**VERBOTEN** als Enhancement-Plugin: ~~`dccrn_plugin`~~, ~~`fullsubnet_plus_plugin`~~.

### [ERGÄNZUNG] Deferred-Phases-Priorisierung

Wenn mehrere Deferred-Phases in Stufe 2 nachgeholt werden müssen, erfolgt die Abarbeitung in folgender Reihenfolge: (1) Phasen mit P1/P2-Zielbezug, (2) Phasen mit P3-Zielbezug, (3) alle übrigen. Innerhalb jeder Gruppe entscheidet die Reihenfolge des Auftretens im ursprünglichen Pipeline-Plan. Bei erneutem Ressourcenmangel werden die verbleibenden Deferred-Phases für den nächsten Durchlauf vorgemerkt.

> Vollständige SOTA-Entscheidungsmatrix: `.github/specs/04_dsp_standards.md` §4.4

### [RELEASE_MUST] Hybrid-Release-Mode für Kernmodelle

- `release_mode` ∈ `primary|fallback|blocked` — Statusskripte greifen auf JSON zurück.
- Fallback-Kaskaden: `sgmse_plus.ts → wpe_dsp`, `versa → pqs_dsp`, `flow_matching → cqtdiff → diffwave`.
- Quarantänisierte Crash-Kandidaten (z. B. RMVPE) dürfen **nicht** als Primärpfad registriert werden.

## Schlecht-Material-Verarbeitungsregeln

**Adaptive Schwellen §2.29**: `REGRESSION_THRESHOLD` restorability-abhängig (GOOD 0.020 / FAIR 0.035 / POOR 0.055), max. 5 Retries.

**[RELEASE_MUST] §2.29 PMGG Phase-Skip-Verbot (v9.10.64)**:

- PMGG darf Phasen **NIEMALS** überspringen (kein Rollback auf Original-Audio).
- CausalDefectReasoner hat die Phase als notwendig bestimmt — sie **MUSS** angewendet werden.
- Nach 5 gescheiterten Retries: **Best-Effort** — der Versuch mit der **geringsten Musical-Goal-Regression** wird angewendet (`action="best_effort"`).
- **VERBOTEN**: `return audio, scores_before, "rollback", 0.0` — Rückgabe von unverändertem Original-Audio = Phasen-Skip.
- Gültige PMGG-Actions: `"passed"` | `"retry1"` … `"retry5"` | `"best_effort"` | `"best_effort_rN"`

**[RELEASE_MUST] §2.29a PMGG Inference-Caching bei Retries (v9.10.75)**:
ML-Modelle sind deterministisch (gleicher Input → gleicher Output). Bei PMGG-Retries wird die ML-Inferenz **NICHT** wiederholt. Stattdessen: Erster Aufruf mit `strength=1.0` (volle Inferenz) → Cache `audio_full`. Retries variieren ausschließlich Wet/Dry-Blending: `audio_retry = dry + strength × (audio_full − dry)`.

**ML-deterministische Phasen** (gecachte Inferenz, nur Wet/Dry-Reblend bei Retry):

| Phase | ML-Modell | Begründung |
| --- | --- | --- |
| `phase_03_denoise` | OMLSA + ResembleEnhance | ML-Hybrid: Inferenz-Output identisch bei gleichem Input |
| `phase_06_frequency_restoration` | AudioSR | Neurale Bandwidth-Extension deterministisch |
| `phase_09_crackle_removal` | BANQUET ONNX | Blind-Denoising deterministisch |
| `phase_12_wow_flutter_fix` | FCPE/CREPE/pYIN | f₀-Schätzung deterministisch (Timing-Phase: kein Wet/Dry) |
| `phase_18_noise_gate` | Silero VAD | Binary-Mask deterministisch |
| `phase_20_reverb_reduction` | SGMSE+ (Primärpfad) | Reverb-Speech-Separation deterministisch (WPE-DSP-Fallback: muss re-run) |
| `phase_23_spectral_repair` | AudioSR Inpainting | Spektral-Lückenfüllung deterministisch |
| `phase_24_dropout_repair` | AudioSR | Audio-Generierung deterministisch |
| `phase_29_tape_hiss_reduction` | DeepFilterNet v3 II | HF-Denoising deterministisch (OMLSA-DSP <2 kHz: muss re-run) |
| `phase_42_vocal_enhancement` | BSRoFormer | Stem-Separation deterministisch |
| `phase_55_diffusion_inpainting` | CQTdiff/FlowMatching | Diffusions-Inpainting deterministisch |
| `phase_56_spectral_band_gap` | FCPE/CREPE + Synthese | Noten-Synthese deterministisch |

**Strength-abhängige DSP-Phasen** (MÜSSEN bei jedem Retry neu ausgeführt werden):
Alle übrigen Phasen, bei denen `strength` Algorithmus-Parameter steuert (z.B. Filterfrequenz, Kompressionsratio, Sättigungsgrad). Beispiele: `phase_01`, `phase_02`, `phase_04`, `phase_10`, `phase_14`, `phase_17`, `phase_19`, `phase_22`, `phase_25`–`phase_28`, `phase_31`–`phase_41`, `phase_43`–`phase_54`.

**Implementierung**: `PerPhaseMusicalGoalsGate._run_with_retry()` führt `_run_phase(phase, audio, 1.0, kwargs)` genau einmal aus. Retries nutzen `_wet_dry_blend(audio, audio_full, strength, phase)`.

**[RELEASE_MUST] §9.7.11 K-S tonal_center Proxy** (v9.10.91):
`tonal_center` verwendet ab jetzt Krumhansl-Schmuckler-Key-Detection — SNR-invariant. Alle früheren tonal_center-Exclusions (phase_02/03/04/08/18/29/49) wurden entfernt. Wiss. Basis: Krumhansl & Schmuckler 1990, Temperley 2001.

**[RELEASE_MUST] §9.7.12/13/14 SNR-robuste PMGG-Proxy-Fixes für brillanz/transparenz/waerme** (v9.10.92):
Falsche P3–P5-Regressionen in Denoise-/Dereverb-Phasen durch SNR-sensitive Quick-Proxies und pathologische Precise-Override-Mechanismen eliminiert. brillanz, waerme, transparenz aus `_PRECISE_METRICS` entfernt; 14 Exclusion-Einträge über 7 Phasen gelöscht.

- **§9.7.12 brillanz — HF Spectral Crest Factor (2–16 kHz)**:

HF-Energie-Ratio ersetzt durch p95/p50-Crest-Factor. Noise hebt p50-Median; musikalische Peaks dominieren p95 → Crest nach Denoise steigt, nie false drop. `BrillanzMetric`-Preservation-Penalty war kontraproduktiv. Wiss. Basis: Fastl & Zwicker 2007 §8.3.

- **§9.7.13 transparenz — Multi-Band Spectral Crest (5 Oktavbänder 250 Hz–8 kHz)**: 75%-Rolloff-Proxy (SNR-sensitiv) ersetzt durch 5-Oktavband-Crest-Mittelwert. Zudem hatte `TransparenzMetric.measure()` **kein** `reference=`-Parameter → TypeError in Precise-Override → stille Fallback auf fehlerhaften Proxy. Wiss. Basis: Moore & Glasberg 1983; ITU-T P.862.
- **§9.7.14 waerme — Warmth Ratio E(200–800 Hz)/E(800–3000 Hz)**: ISO-226 mid/total-Energie-Ratio reverb-sensitiv → false P4-Regression nach Dereverb. Ersetzt durch reverb-invariantes Sub-Band-Verhältnis (Nachhall addiert Energie proportional in beiden Bändern → Ratio stabil). Wiss. Basis: Fletcher & Rossing; Moore & Glasberg 1983.

**[RELEASE_MUST] §2.29b PMGG Stable-Metric-Invariante (v9.10.79)**:
Metriken mit ML-zustandsabhängigem Gewicht **NIEMALS** in `_PRECISE_METRICS` für PMGG-Delta-Checks aufnehmen.

- **Root-Cause `NatuerlichkeitMetric`**:
CREPE-Load-State ändert Gewichte (w_crepe 0.0 → 0.18) zwischen `scores_before` (CREPE nicht geladen) und `scores_after` (CREPE nun geladen) → Pseudo-Regression Δ ≈ 0.15–0.28 auf unverändertem Audio → false P1-Kaskade → Phase_03 best-effort @ 5.6 % Wet → Noise-Floor −55 dBFS statt −72 dBFS → Mikrodetails verdeckt → **Tiefen-Immersion zerstört**.
- `NatuerlichkeitMetric` läuft **ausschließlich** im Export-Gate (`MusicalGoalsChecker`) — nie in PMGG-Delta-Checks. Schwellwert ≥ 0.90 bleibt dort unverändert normativ.
- **§9.7.8 Audio-Cap**: `_apply_precise_metric_overrides` kappt auf **2.5 s** — ausreichend für stationäre Spektral-/Chroma-/Transient-Metriken; verhindert NMF/Onset-Runs auf Langaudio (> 2 s/Call auf 60 s-Material).
- Neue Metriken vor `_PRECISE_METRICS`-Aufnahme: Eigenrauschen ≤ 0.02 auf identischen Audio-Paaren nachweisen.
- **Aktuelle `PHASE_GOAL_EXCLUSIONS`** (v9.10.96, kanonisch im Code): `phase_03: {"natuerlichkeit", "artikulation", "authentizitaet", "tonal_center", "timbre_authentizitaet"}` (Breitband-Denoise: CREPE-Load-State + transient-shape + K-S NOT invariant for shaped NR §9.7.11 ext + MFCC-Pearson/Centroid-CV disturbed by spectral-envelope change after NR); `phase_29: {"artikulation", "authentizitaet", "natuerlichkeit", "tonal_center", "timbre_authentizitaet"}` (DeepFilterNet Tape-Hiss — gleiche Root-Causes wie phase_03); `phase_02: {"bass_kraft", "authentizitaet", "natuerlichkeit", "transparenz", "groove", "timbre_authentizitaet"}` (Kammfilter Hum-Removal); `phase_04: {"transparenz", "brillanz", "waerme", "authentizitaet", "natuerlichkeit", "timbre_authentizitaet"}` (EQ); `phase_08: {"micro_dynamics", "artikulation"}` (TDP/HPSS); `phase_12: {"tonal_center", "timbre_authentizitaet"}` (Wow/Flutter: K-S volatile nach Pitch-/Speed-Korrektur + Centroid-CV disturbed); `phase_18: {"micro_dynamics", "authentizitaet", "emotionalitaet", "groove"}` (Noise Gate); `phase_20: {"authentizitaet", "natuerlichkeit"}` (SGMSE+ Reverb-Reduction); `phase_23: {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"}` (AudioSR Spectral Inpainting: synthetisierter Inhalt); `phase_24: {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"}` (Dropout); `phase_49: {"authentizitaet"}` (Dereverb); weitere kleinere Phasen. Material-adaptive Erweiterungen: `cd_digital`/`dat` → phase_03/phase_29 auf `{"natuerlichkeit", "artikulation"}` reduziert.
- **§2.31b Material-adaptive Relaxation** (v9.10.85, akt. v9.10.96): Für `cd_digital`/`dat` werden bei `phase_03` und `phase_29` die meisten Ausschlüsse aufgehoben — nur `{"natuerlichkeit", "artikulation"}` bleiben. brillanz/transparenz sind seit §9.7.12/13 bei **allen** Materialtypen SNR-robust — nicht mehr ausgeschlossen. tonal_center und timbre_authentizitaet bleiben statisch ausgeschlossen (materialunabhängig: shaped NR und Centroid-CV-Disturbance).

**Era-GP-Warmstart §2.14**: decade ≤ 1940 → `noise_reduction_strength ~ N(0.90, 0.05)`; ≤ 1960 → N(0.75, 0.08); ≥ 1970 → N(0.50, 0.10).
**Material-MOS §6.2**: MOS ≥ 4.5 NUR für `cd_digital/dat/mp3_high/aac`; Shellac ≥ 3.8, Vinyl ≥ 4.0, Tape ≥ 4.2.
**Chunk-Größe §7.6**: Silence 120 s, Severity ≥ 0.6 → 5 s, ≥ 0.3 → 15 s, sonst 60 s (Min. 2 s / Max. 120 s).
Implementierung: `backend/core/adaptive_chunk_processor.py` — `process_in_adaptive_chunks(phase_fn, audio, sr, max_severity)`. Crossfade: Hanning 10 ms. Phasen können das Modul opt-in nutzen.

## Vintage Aesthetics (§5 — bindend)

**SOFT_SATURATION** (Röhren/Tape-Charakter) = **BEWAHREN**. **CLIPPING** (Amplitudenbeschädigung) = **REPARIEREN**.
`classify_clipping()`: flat_tops > 0.1 % UND THD_odd > THD_even×1.5 → CLIPPING; sonst → SOFT_SATURATION → phase_23 überspringen.

> **Allgemeiner Grundsatz SR-Agnostik in Analyse-Modulen** (autoritativ: Performance-Budget §2.37):
> Alle Analyse-/Scan-/Klassifikations-Module (DefectScanner, `classify_clipping`, `analyse_clipping`, RestorabilityEstimator, EraClassifier, MediumClassifier) arbeiten bei **nativer Import-SR**. THD-Berechnungen nutzen `sr` nur für Frequenz-Bin-Zuordnung — die Mathematik ist SR-agnostisch. **VERBOTEN**: `assert sr == 48000` in diesen Modulen. `assert sr == 48000` gilt **ausschließlich** für Verarbeitungs-Phasen (01–64) und Plugins.

1920–1940: Rolloff ≤ 7 kHz nicht erweitern, H2/H4 bewahren. 1940–1975: `phase_22` nur emulieren, nie eliminieren.

## §2.9 PANNs Instrument-Phasen-Aktivierungsmatrix (Pflicht-Schwellwerte)

| PANNs-Kategorie | Phase | Schwellwert |
| --- | --- | --- |
| Singing voice / Vocals / Speech | `phase_19` + `phase_42` + `phase_43` + VocalAIEnhancement | ≥ 0.40 / ≥ 0.35 |
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.50 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.50 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.50 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.50 |

> **Invariante**: Instrument-Schwelle ist einheitlich **0.50** (nicht 0.60). Höherer Wert blockiert Enhancement bei Ensemble-Aufnahmen. Änderungen hier → immer auch `backend/core/unified_restorer_v3.py` L≈5822 + `plugins/panns_plugin.py` Docstring anpassen.

## Vocal-Restaurierungskette (§2.8)

Reihenfolge: `GenderDetector` → SGMSE+ → FCPE/CREPE/pYIN → FormantTracker (LPC Ord. 30–40 @ 48 kHz) → BreathDetector (±0.05) → De-Esser → `VocalAIEnhancement` → PSOLA (Pitch-Korrektur > ±2 HT).
**API-Falle**: `enhanced, report = self.breath_intelligence.process(audio, sr)` — kein `events`-Argument!

**[RELEASE_MUST] Vocal-Intimitäts-Gate (Phase 42)**:

- Vor/nach `phase_42_vocal_enhancement` wird eine Intimitäts-Metrik (Fricative 4–8 kHz + Plosive-Transienten 120–350 Hz) gemessen.
- Wenn `vocal_intimacy_delta < -0.04`, MUSS ein Safety-Rescue-Blending mit Dry-Signal greifen.
- Pflicht-Metadaten im `PhaseResult.metadata`: `vocal_intimacy_pre`, `vocal_intimacy_post`, `vocal_intimacy_delta`, `vocal_intimacy_gate_triggered`, `vocal_intimacy_rescue_mix`.

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
- `sf.read(path)` / `librosa.load(path)` direkt für beliebige Audio-Formate → immer `load_audio_file(filepath)` aus `backend.file_import` (Kaskade: soundfile → pedalboard/FFmpeg → pydub). `sf.read(io.BytesIO(...))` auf interne PCM-Puffer bleibt erlaubt.

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
| --- | --- |
| DefectScanner | ≤ 4 s |
| Phase-Pipeline gesamt | ≤ 240 s |
| FeedbackChain (alle Iter.) | ≤ 120 s |
| ExcellenceOptimizer | ≤ 60 s |
| RestorabilityEstimator | ≤ 5 s |

- Interne Verarbeitungs-SR (Phasen 01–64, Plugins): stets **48 000 Hz**
- **Analyse-Module** (DefectScanner, classify_clipping, RestorabilityEstimator, EraClassifier, MediumClassifier): arbeiten bei **nativer Import-SR** — kein Resampling vor Analyse, kein `assert sr == 48000`
- **Dual-SR-Routing (Pflicht)**: Zwei getrennte Pfade führen — `analysis_audio/analysis_sr` (native Import-SR) für Analyse/Klassifikation, `processing_audio/processing_sr=48000` für alle Verarbeitungsphasen/Plugins.
- **Fail-fast bei 48-kHz-Normierung (Pflicht)**: Wenn `processing_sr != 48000` und Resampling nicht möglich ist, MUSS der Lauf mit strukturierter Fehlermeldung abbrechen; ein Weiterlauf der Phasen auf Nicht-48k ist verboten.
- **Resampling-Scope-Invariante**: Resampling darf nur den Verarbeitungspfad verändern; Analysepfad bleibt unverändert in nativer SR für material-/ära-/defektrobuste Entscheidungen.
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

### [RELEASE_MUST] Quality-First Hauptlauf (v9.10.80)

- In nutzerseitigen Standardpfaden (GUI `BatchProcessingThread`, CLI `aurik_cli.py`, Batch `batch_processor.py`) MUSS `AurikDenker.denke(..., no_rt_limit=True)` verwendet werden.
- Ziel: Keine Qualitätsreduktion zugunsten von RT im Hauptlauf (kein RT-bedingtes Phase-Skip/Deferral im Normalbetrieb).
- PerformanceGuard bleibt aktiv für Telemetrie, Stabilitäts-Gates und strukturierte Schutzentscheidungen; RT-Limits dürfen Qualität nur in expliziten Nicht-Standardpfaden begrenzen.
- `deferred_phases` bleibt verpflichtend für echte Laufzeit-/Ressourcen-Fallbacks (z. B. OOM/Headroom/Inference-Timeout), damit KMV Stufe 2 Vollqualität nachzieht.

### [RELEASE_MUST] §2.38 Kontinuierliche ML-Veredelung (KMV) — Vollqualitäts-Garantie

**Kernprinzip**: RT-Limit-Überschreitung führt nie zu dauerhaftem Qualitätsverlust im Export.

| Stufe | Thread | RT-Limit | Ergebnis |
| --- | --- | --- | --- |
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

### [RELEASE_MUST] §2.38a ML-Headroom-Guard + Structured Fallback (v9.10.78)

**Kernprinzip**: Schwere ML-Stufen duerfen nur starten, wenn vor dem Modell-Load und direkt vor der Inferenz ausreichend physischer RAM-Headroom bestaetigt ist. Sonst: kontrollierter DSP-Fallback statt OOM/Hard-Crash.

**Pflicht fuer heavy ML-Pfade** (z. B. SGMSE+, ResembleEnhance, AudioSR, CQTdiff/FlowMatching):

- Vor `AudioSRPlugin()` / `InferenceSession()` / `torch.load()` muss ein lokaler Headroom-Guard laufen.
- Guard muss dauer- und kanalbewusst sein (mono/stereo, Dateilaenge) und darf bei knappem RAM proaktiv `evict_stale_plugins()` + `gc.collect()` + `malloc_trim(0)` versuchen.
- Wenn Guard triggert: **kein Phase-Skip auf Original-Audio**, sondern DSP/leichter Fallback innerhalb derselben Phase.

**Structured Fallback-Metadaten (Pflicht)**:

- `RestorationResult.metadata["ml_guard_events"]` als `list[dict]`.
- Pflichtfelder je Event: `phase_id`, `model`, `reason`, `required_gb`, `available_gb`, `channels`, `duration_s`, `fallback`.
- `deferred_phases` muss die betroffene Phase enthalten, damit KMV Stufe 2 sie spaeter ohne RT-Limit nachzieht.

**Qualitaetsinvariante**:

- Guard-getriggerter Fallback ist nur als temporaere Laufzeitstrategie zulaessig.
- Vollqualitaet muss ueber §2.38 Stufe 2 wiederhergestellt werden (Overwrite nur wenn `stufe2_quality_estimate >= stufe1_quality`).

### [RELEASE_MUST] §2.40 Vollpipeline-Determinismus + Konkurrenten-Stratifizierung (v9.10.79)

**Determinismus-Invariante**:

- Gleiche Eingabe + gleiche Umgebung + gleicher Modus => bitnahe Ausgabe.
- Toleranzen fuer Vergleichstests: `max_abs_err <= 1e-6`, `rms_err <= 1e-7`, identische `phases_executed`.
- Pflicht: alle stochastischen Komponenten mit dokumentiertem Seed; kein unseeded Zufall in Produktionspfaden.

**Konkurrenz-Gate (stratifiziert)**:

- Aurik muss gegen Referenzsystem pro Material und pro Defektklasse bestehen, nicht nur im Gesamtmittel.
- Pflicht-Matrix: `tape/vinyl/shellac/digital/vocal` x `hiss/crackle/dropout/reverb/hum/codec`.
- Release failt bei regressiver Zelle auch dann, wenn der Gesamt-OQS noch besteht.

### [RELEASE_MUST] §2.41 Denker-Vollkontext — Material-adaptive DSP-Reparatur (v9.10.117)

**Kernprinzip**: ReparaturDenker und RekonstruktionsDenker arbeiten kontextbewusst mit Material, Ära und DefectScanner-Locations.

- **ReparaturDenker**: 12 Material-Profile mit je 4 Schwellwerten (`click_iqr`, `click_kernel_ms`, `clip_threshold`, `hum_detect_db`). Hierarchie: Shellac (IQR=4.0) → Vinyl (5.0) → Tape (7.0) → CD (9.0). Era-adaptive Hum-Sensitivität (≤1940: ≥−42 dB, ≤1960: ≥−47 dB). Chirurgische Click-Reparatur mit DefectScanner-Locations (IQR-Maske auf Defekt-Positionen eingeschränkt).
- **RekonstruktionsDenker**: 6 Material-Konfigurationen für GapReconstructor (Shellac: max 200 ms, Tape: bis 2000 ms).
- **AurikDenker**: Leitet `defect_scores`, `defect_locations`, `era_decade`, `material` an alle audio-modifizierenden Denker-Stufen weiter.
- Wissenschaftliche Basis: Copeland 2008, Katz 2007.

### [RELEASE_MUST] §2.42 SourceFidelityReconstructor — Generationsverlust-Kompensation (v9.10.115–116)

**Kernprinzip**: Aurik modelliert den Unterschied zwischen dem erhaltenen Signal und dem Original am Aufnahmetag.

- **Modul**: `backend/core/source_fidelity_reconstructor.py` — `SourceFidelityTarget` + `SourceFidelityReconstructor`.
- **Tabellen**: `_ERA_BANDWIDTH_HZ`, `_ERA_DYNAMIC_RANGE_DB`, `_ERA_HARMONIC_DENSITY`, `_MATERIAL_GENERATION_COUNT`, `_ERA_MIC_TYPE`, `_MIC_PRESENCE_CENTER_HZ`, `_GENERATION_LOSS_DB_PER_GEN` (13 Materialklassen).
- **`compute_correction_curve_db()`**: Frequenz-abhängige dB-Korrekturkurve. Kompensiert `extra_gens × loss_per_gen[freq]`. Rolloff über Original-Ären-Bandbreite. Cap: `_MAX_CORRECTION_DB = 12.0`.
- **`SourceFidelityEQProcessor`**: Linear-Phase FIR-Filter (257 Taps, `firwin2`, boosts only). Singleton.
- **UV3-Integration**: `_build_song_calibration_profile()` befüllt `source_fidelity_*`-Felder: `bandwidth_target_hz`, `reconstruction_strength`, `confidence`, `generation_count`, `hf_loss_db`, `harmonic_density`, `era_mic_type`, `presence_hz_lower`, `presence_hz_upper`.
- **Phase 06**: SourceFidelityEQ nach ML-Hybrid-Restaurierung (`sqrt(recon_strength × confidence)`, cap 0.70).
- **Phase 38**: Ära-bewusste Presence-Center (1920s Carbon: 2200/3500 Hz → 1970s+ Modern: 4000/6500 Hz).
- **Phase 39**: Ära-bewusste Air-Band-Deckelung (Shelf-Frequenz ≤ `bandwidth_target_hz × 0.85`). HF-Loss-Kompensation.
- Wissenschaftliche Basis: Eargle 2004, Huber & Runstein 2009, Copeland 2008, IEC 60094.

### [RELEASE_MUST] §2.43 Phase-Preserved Wet/Dry-Blend (v9.10.118)

**Kernprinzip**: STFT-Wet/Dry-Blend bewahrt Phaseninformation statt naivem Sample-Blend.

- Magnitude-Interpolation im STFT-Bereich: `M_blend = (1−α)·M_dry + α·M_wet`, Phase vom Wet-Signal.
- Verhindert Phase-Cancellation-Artefakte bei Kopfhörer-Wiedergabe.
- **Diminishing-Returns-Moderation**: Kumulative Stärke-Moderation nach vielen aufeinanderfolgenden Phasen.
- **Datei**: `backend/core/unified_restorer_v3.py`

### [RELEASE_MUST] §9.7.15 Musical-Goals-Metriken-Recalibration (v9.10.120)

Alle Musical-Goals-Metriken auf psychoakustisch korrekte Divisoren/Multiplikatoren kalibriert:

| Metrik | Änderung | Wissenschaftliche Basis |
| --- | --- | --- |
| **Brillanz** | HF Crest-Divisor 13.5 → **10.5** | Fastl & Zwicker 2007 §8.3 |
| **Transparenz** | 5-Band-Crest-Divisor 8.8 → **7.0** | Moore & Glasberg 1983 |
| **Wärme** | H2/H4 Even-Harmonic-Divisor 9.0 → **5.0** | Fletcher & Rossing |
| **Natürlichkeit** | Flatness ×2→×2.5, ZCR-Var ×100→×60, Contrast ÷30→÷25, Onset ÷10→÷8 | Johnston 1988 |
| **Emotionalität** | LUFS Pre-Normalization auf −14 LUFS vor Dynamics-Berechnung | Loudness-invariant |
| **PQS NSIM** | Pearson → ERB-gewichtete Korrelation (300–4000 Hz) | Patterson et al. 1992 |
| **PQS MCD** | Pseudo-RMS → echte Mel-Cepstral Distortion (13 MFCCs) | Kubichek 1993 |

### Weitere Fixes v9.10.113–121

- **Phase 09**: Severity-adaptiver Dry-Blend bei Crackle (`texture_preserve` nach Defekt-Severity skaliert).
- **Phase 29**: Verschärftes G_floor für Presence/Air-Zonen im OMLSA-DSP-Fallback.
- **Phase 40**: Studio-2026 erzwingt −14 LUFS für alle Materialien (kein Material-Target im Studio-Modus).
- **Phase 42**: Stereo-Wiener-Masking statt Mono-Duplikation; Multi-Formant Bell-EQ (F1/F2/F3/Singer's Formant).
- **Phase 55**: Adaptiver AR-Order für lange Dropout-Gaps (AR(192) bei Gaps > 50 ms).
- **HPSS Kernel**: `HARMONIC_KERNEL: 31→17`, `PERCUSSIVE_KERNEL: 31→13` (Fitzgerald 2010).
- **ExcellenceOptimizer**: PGHI-Rekonstruktion nach jeder Magnitude-Modifikation.
- **Crossfade**: Float64-Zwischenpräzision in `adaptive_chunk_processor.py`.
- **MDEM**: Tail-Auslauf `np.linspace(last_gain, 1.0, tail_len)`.
- **EmotionalArc**: Spectral-Centroid statt ZCR; Per-Song Centroid-Median-Normalisierung.
- **OMLSA**: Energiebasierter G_floor für Stille-Segmente.
- **De-Esser**: Ära-abhängige Sibilanz-Schwellwerte.
- **Phase 12**: Wow/Flutter PITCH_HOP_FACTOR 2→4, STFT_WINDOW_SIZE 1024→2048.
- **`load_audio_file()`**: Parameter `do_carrier_analysis=False` im BatchProcessingThread (Fortschrittsbalken-Fix v9.10.121).

### [RELEASE_MUST] §8.4 Externes Mini-MUSHRA-Protokoll fuer Kern-aenderungen (v9.10.79)

Bei aenderungen an Kernphasen, PMGG, DefectScanner oder heavy ML-Fallbacks ist ein externer Mini-MUSHRA-Nachweis Pflicht.

- Feste Szenarien: mindestens 6 (darunter 2 Vocal-Szenarien).
- Feste Panelgroesse: mindestens 8 Hoerer.
- Pflichtbericht als Artefakt mit Szenario-Scores, Konfidenzen und Delta zur Vorversion.
- Kein Release ohne gueltiges Bericht-Artefakt.

### [RELEASE_MUST] §2.39 OOM-Recovery-Checkpoint-System — Nahtlose Wiederaufnahme

**Kernprinzip**: systemd-oomd-Kill oder MemoryError führen nie zu Totalverlust. Pipeline-Zwischenstand wird atomar auf Disk persistiert und beim nächsten Start automatisch zur Wiederaufnahme angeboten.

**Checkpoint-Lifecycle**:

| Schritt | Komponente | Aktion |
| --- | --- | --- |
| 1 | `_execute_pipeline()` MemoryError-Handler | `save_checkpoint()` → `sessions/<stem>_oom_checkpoint.json` + `_oom_audio.wav` (atomisch: `.tmp` → `os.replace`) |
| 2 | `ModernMainWindow.__init__` (1,5 s QTimer) | `find_pending_checkpoints()` → Dialog "Restaurierung fortsetzen?" |
| 3 | Nutzer bestätigt | `_resume_from_checkpoint()` → Originaldatei laden → normale Restaurierung |
| 4 | Erfolgreicher Abschluss | `delete_checkpoint()` → Cleanup |

**Modul**: `backend/core/recovery_checkpoint.py`

**`RecoveryCheckpoint`-Pflicht-Felder**: `input_path`, `output_path`, `phases_executed`, `phases_remaining`, `mode`, `material_type`, `era_decade`, `defect_scores`, `defect_scores_full`, `restorability_score`, `spectral_fingerprint`, `quality_estimate_at_failure`, `musical_goals_at_failure`, `audio_wav_path`, `sample_rate`, `original_input_path`, `timestamp`, `aurik_version`, `failure_phase`, `failure_reason`

**Pfad-Durchleitung**: `BatchProcessingThread` → `denke(input_path=, output_path=)` → `restauriere()` → `_orchestriere()` → `RestaurierDenker.restauriere()` → UV3 `restore(input_path=, output_path=)` → `self._recovery_ctx` → `_execute_pipeline` MemoryError-Handler → `save_checkpoint()`

**Invarianten**:

- Checkpoint-Audio als `FLOAT` WAV (verlustfrei, kein Encoding-Verlust)
- Ablauf: 7 Tage (`_MAX_CHECKPOINT_AGE_S`) — danach automatische Bereinigung
- Thread-safe: Alle Writes über `.tmp` + `os.replace` (POSIX-atomar)
- Datenschutz: Lyrics-Text NICHT im Checkpoint (§2.36 Pflicht)
- Wiederaufnahme nutzt das **Original-Audio** (nicht das Checkpoint-Audio) für volle Qualität
- Checkpoint-Audio dient als Fallback wenn Original fehlt

**VERBOTEN**: Checkpoint-Audio als Primärquelle für Re-Restaurierung (Doppelverarbeitung degradiert Qualität).

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
- **MP3-Export**: LAME VBR V0 (`ffmpeg q:a=0`, bis 320 kbps adaptiv, ≈245 kbps Ø). VERBOTEN: CBR für Restaurierungsausgaben — CBR erzeugt Pre-Echo auf restaurierten Transienten (TDP/MDEM §8.3). GUI zeigt „bis 320 kbps, VBR".
- **Print-Through (Phase 29, reel_tape)**: Bidirektionale LMS — alpha_pre ≠ alpha_post. VERBOTEN: Comb-Filter oder symmetrisches Modell.
- **§2.12 PolyphonicSpeedCurveEstimator** (`quality_mode=maximum`): BasicPitch ONNX → Konfidenz-gewichteter Median ≥ 2 Voices → Savitzky-Golay. try_allocate("BasicPitch", 0.12) Pflicht. GrooveMetric-DTW ≤ 8 ms RMS nach Korrektur.
- **Perceptuelle Pflicht-Messwerte**: LUFS-Diff ≤ 1 LU | Chroma Pearson ≥ 0.95 | Groove DTW ≤ 8 ms RMS | Transient Attack ≤ ±2 ms | MERT-Harmonizität ≥ 0.85.

**Pfad-Mapping**: `core/<modul>.py` → `backend/core/`, `plugins/<plugin>.py` → `plugins/`, Frontend/UI → `Aurik910/` (kein `frontend/`-Verzeichnis!). Import: `from Aurik910.i18n import t, set_language`.

## Restaurierungs-Modi

| Modus | Ziel | LUFS | TonalCenter |
| --- | --- | --- | --- |
| **Restoration** | Originalgetreue Restauration | Δ ≤ 1 LU | ≥ 0.95 |
| **Studio 2026** | Highend-Studio-Klang | −14 LUFS EBU R128 | ≥ 0.97 |

Studio 2026: Stem-Sep → Vocal-AI → Instrumente → [Reference Mastering] → Multibandkomp → Präsenz/Air → Stereo-Imaging → Re-Mix (StemRemixBalancer) → LUFS-Norm → TruePeak → Musical Goals → [Vocos-Synthese (MOS < 4.3)]

**StemRemixBalancer** — Pflicht nach Stem-Verarbeitung: Verboten: nacktes `vocals + instruments` in UV3. Algorithmus (6 Schritte): L_orig messen → vocal_weight via PANNs → LUFS pro Stem → Gain-Korrektur → Re-Mix → Final-Check |LUFS(mix) − L_orig| ≤ 0.3 LU. Pflicht-Test: `tests/unit/test_stem_remix_balancer.py` (≥ 20 Tests).

## Universelle Garantien (§8.2 — PFLICHT)

| Garantie | Schwellwert |
| --- | --- |
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
| FeedbackChain-Rollback | \|MOS_neu − MOS_alt\| > 0.05 → sofortiger Rollback |

## Psychoakustik & Gänsehaut-Prinzipien (§8.3 — bindend)

**Oberstes Ziel**: Das Restaurierungsergebnis muss beim Hörer **emotionale Wirkung** erzeugen — nicht nur technische Korrektheit. Die Psychoakustik steht über der technischen Perfektion.

**Gänsehaut-Formel**: `(TransientIntegrity × MicroDynamik × Klarheit × Authentizität) − Artefakte`

| Komponente | Verantwortliches Modul | Anteil |
| --- | --- | --- |
| **Transient-Punch** | TDP (Transient Decoupled Processing) | ~40 % |
| **Mikro-Dynamik-Erhalt** | MDEM (400 ms LUFS-Morphing) + EmotionalArcCorrection (5 s Makro-Bogen) | ~25 % |
| **Rauschbefreiung/Klarheit** | SGMSE+ / OMLSA/IMCRA | ~20 % |
| **Vokal-Präsenz** | Phase 42 + Phase 43 + VocalAIEnhancement | ~10 % |
| **Neurale Synthese** | Vocos 48 kHz (nur Studio 2026, MOS < 4.3) | ~5 % |

**Zwei-Skalen-Dynamik-Schutz** (Pflicht):

- **Mikro-Ebene (400 ms)**: MDEM — `morph(restored, original, sr)` — LUFS-Profil-Rückgewinnung; mode-adaptives Gain-Limit: Restoration 4.0 dB, Studio 2026 6.0 dB
- **Makro-Ebene (5 s)**: `correct_emotional_arc(original, restored, sr)` — post-MDEM, wenn Arousal/Valence-Bogen abgeflacht. Algorithmus: RMS-basierte Gain-Hüllkurve, ±6 dB, 70 % Dämpfung, Savitzky-Golay-geglättet. Safety-Revert wenn Korrektur verschlechtert.

**Defect-Locations-Flow** (§9.1 Ergänzung):

- `_execute_pipeline` extrahiert `defect_locations: dict[str, list[tuple[float, float]]]` + `max_defect_severity: float` aus `defect_result.scores`
- Beide werden als `kwargs` an jede Phase übergeben (`defect_locations=`, `max_defect_severity=`)
- Phasen können Location-Hints für gezieltere Verarbeitung nutzen (opt-in)
- Phasen erkennen Defekte weiterhin auch eigenständig intern (Redundanz-Prinzip)
- **Location-Completeness-Invariante**: In Analyse- und Reparaturpfaden sind harte Caps auf `defect_locations` verboten. Auch bei sehr vielen Events (hundert bis tausend) muss die vollständige Ereignisliste erhalten bleiben.
- UI/Visualisierung darf Marker verdichten oder sampeln, aber ausschließlich als Anzeige-Optimierung. Die zugrunde liegende Defektliste für PMGG/Phasenrouting bleibt unverändert vollständig.

**Psychoakustische Pflicht-Invarianten**:

- Intro-Salienz (§9.1b): Defekte in den ersten 5 s → Severity ×1.5
- Perceptual Masking (§9.1c): Maskierte Defekte → `severity * (0.3 + 0.7 * salience)`
- Emotionaler Bogen: Messung + aktive Korrektur (nicht nur Logging)
- Vintage-Harmonische: H2/H4 bewahren, Soft-Saturation ≠ Clipping

**[RELEASE_MUST] Tiefen-Immersions-Prinzip — „Ohr in die Musik legen" (v9.10.79)**:
Ziel ist keine technische Sauberkeit, sondern akustische Tiefe: Der Hörer taucht in die Musik ein.

| Tiefenschicht | Frequenzbereich | Enthält | Technische Bedingung |
| --- | --- | --- | --- |
| Air & Präsenz | 8–20 kHz | Saiten-Obertöne, Gesangs-Luft, Stick-Attack | Noise Floor < −72 dBFS, Phase_06 SBR, Phase_39 Air |
| Vokal-Intimität | 4–8 kHz | Frikative /s/ /f/, Plosive /p/ /t/, Atem | LyricsGuidedEnhancement segmentspezifisch |
| Instrument-Körper | 200 Hz–4 kHz | Note-Sustain, Saitenresonanz, Bogen-Kratzen | TDP-Transient + MDEM-Mikrohüllkurve |
| Fundament | 20–200 Hz | Kick-Punch, Bassresonanz | BassKraftMetric + Virtual-Pitch-Analyse |
| Raum | Diffus (MS) | Raumluft, Phantom-Center, Tiefenstaffelung | SpatialDepthMetric IACC ≥ 0.70, M/S ≥ 0.97 |

- **PMGG-Stabilitäts-Kausalkette**: Phase_03 @ optimalem `strength` → Noise Floor < −72 dBFS → Air-Layer + Vokal-Intimität-Layer frei → Hörer taucht ein. Phase_03 @ best-effort 5.6 % (false P1-Regression) → −55 dBFS → Mikrodetails verdeckt → Studio-Distanz-Effekt, keine Gänsehaut. **§2.29b ist direkte Voraussetzung für Tiefen-Immersion.**
- **Vokal-Intimität**: Phonem-Boost-Faktoren (§2.36): `fricative ×1.55`, `plosive ×1.40`, `vowel_stressed ×1.35`, `silence ×0.70`. Kein uniformes Enhancement — Konsonanten-Transient-Integrität erzeugt physische Nähe der Stimme.
- **Raumtiefe durch Phasenkohärenz**: PGHI nach jeder STFT-Modifikation bewahrt interaurale Phasendifferenzen (IPD) → Blauert Precedence Effect → Hörer lokalisiert Instrumente imaginär im Raum. Griffin-Lim randomisiert Phasen → Raumtiefe kollabiert (VERBOTEN als Endschritt).
- **Emotionaler Atemzug — Arousal-Bogen**: `arousal_pearson ≥ 0.85` — die Energie-Kurve einer Phrase muss erhalten bleiben. Ein pp vor dem fff, eine Pause vor dem Choreinschluss: diese Spannungsmechanik erzeugt Gänsehaut. Komprimierter Bogen → Musik klingt tot.
- **Harmonische Vollständigkeit**: Grundton + alle Partialtöne bis Air (> 12 kHz). Abgeschnittene Oberton-Reihen lassen Instrumente „kartonig" klingen. Phase_06 und Phase_39 sind keine optionalen Phasen — sie sind Immersions-Voraussetzung.

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

`from backend.api.bridge import ...` in `try/except ImportError` wrappen. Bei Fehler: `_BRIDGE_AVAILABLE = False` + Stubs. `_export_guard`-Stub vollständig implementieren (NaN-Guard + Clip). Alle anderen Stubs: `return None`.

### BatchProcessingThread — Signal-Kontrakt (Kurzform)

`item_started(str)` | `item_progress(str, int 0–100)` | `item_finished(str)` | `item_finished_with_result(str, object)` | `item_error(str, str)` | `all_finished()` | `defect_update(dict)` | `phase_update(str)` | `waveform_data(ndarray, int)` | `mode_update(str)` | `ml_status_update(bool, list)` | `phase_progress(int 0–100)` | `scan_progress(float 0.0–1.0)` | `quality_update(float 0.0–5.0)`

- `progress_callback`-Signatur: `(pct: int, msg: str, elapsed_s:
float = 0.0) → None`
- `phase_progress` → `phase_progress_bar.setValue(v * 100)` (5 px, lila Gradient, unter Hauptleiste)
- `scan_progress` → `waveform_widget.set_scan_pos(frac)` (oranger gestrichelter Cursor mit Glow)
- `quality_update` → `quality_meter_widget.set_mos(mos)` (steigt 2.5 → 4.2 während Verarbeitung)

### §11.4a Echtzeit-UX-Features (ab 9.10.57)

| # | Feature | Implementierung |
| --- | --- | --- |
| 1 | **Zweistufiger Fortschrittsbalken** | `phase_progress_bar` (5 px, `QProgressBar`, lila Gradient) unter `progress_bar`; Sichtbarkeit: bei Batch-Start ein, bei `_on_all_finished` aus |
| 2 | **Defekte hochzählen/herunterzählen** | `_update_defects`: bei `status=="detected"` Count-up-Animation (QTimer, 22 Frames × 85 ms via `_tick_defect_reveal`); `_PHASE_REDUCES`-Mapping senkt Defekt-Scores × 0.3 bei passenden Phasen-Keywords |
| 3 | **Varianten-Wettkampf** | `multi_pass_strategy.process_with_variants()` emittiert nach jeder Variante `"Variante X/N: 'name' → MOS 4.12 ✓"`; `_on_batch_progress` baut Rangliste `★name_1 (4.12) › name_2 (3.87)` |
| 4 | **Musical-Goals-Meter live** | `quality_update`-Signal verbunden mit `quality_meter_widget.set_mos()`; startet bei 2.5, steigt proportional zum Fortschritt auf 4.2 |
| 5 | **Phasen-Erklärungstext** | `_PHASE_EXPL`-Dict (22 Einträge) mappt Phasen-Keywords auf Kurzbeschreibungen; wird als `[Kontext]` in Statuszeile angehängt |
| 6 | **Waveform-Scan-Cursor** | `WaveformWidget._scan_pos`; oranger Cursor (12 px Glow rgba(255,150,30,45) + 2 px DashLine rgba(255,178,55,215)); `set_scan_pos(-1.0)` blendet aus; Reset in `_on_all_finished` |
| 7 | **Live-Qualitätszahl** | `quality_meter_widget` wird bei Batch-Start sichtbar (`set_mos(2.5)`); steigt mit `scan_progress` |
| 8 | **Vorab-Hörprobe** | `QTimer.singleShot(1400, self._auto_preview_restored)` in `_on_item_finished_with_result`; spielt erste 5 s (= 5×48000 Samples) des restaurierten Audios; nur wenn kein anderer Playback läuft |

### Async-Analyse-Kette nach Datei-Öffnen

5 Daemon-Threads beim Datei-Öffnen: `_bg_load` (3-stufige Audio-Kaskade) → `_carrier_bg` → `_detect_era_genre_bg` → `_estimate_restorability_bg` → `_run_defect_scan_bg`. Alle via `_dispatch_to_gui` oder `QTimer.singleShot(0, ...)`.

**Magic-Button-Synchronisations-Gate (v9.10.x):** Die Magic Buttons bleiben deaktiviert, bis **beide** Pre-Analyse-Threads abgeschlossen haben: `_run_defect_scan_bg` (Defektanalyse) **und** `_detect_era_genre_bg` (Ära/Genre). Erst wenn beide signalisiert haben, werden die Buttons via `_finalize_preanalysis()` freigegeben. Timeout-Fallback: `QTimer.singleShot(15_000, _preanalysis_timeout)` — danach Freigabe unabhängig vom Era/Genre-Status. Begründung: UV3 nutzt `_era_denoise_scale` (±10–12 % NR-Stärke) und Genre-`family_scalars` (vocal +10 %, transient +5 %) — die Empfehlung, die der Nutzer sieht, muss alle vier Signale enthalten.

**Sync-Mechanismus in `_continue_file_loaded`:**

- `_finalize_preanalysis()`: Buttons freigeben, Progress 100 %, Empfehlungstext setzen; Double-Fire-Guard via `_preanalysis_finalized_for`
- `_try_signal_preanalysis_done(flag)`: Flag `"defect_scan"` | `"era_genre"` setzen; bei beiden → `_finalize_preanalysis()`; bei nur `"defect_scan"` → indeterminate Spinner `"⏳ Ära & Genre werden erkannt …"`
- State-Reset in `_load_file`: `_preanalysis_flags: set[str] = set()`, `_preanalysis_timeout_fired = False`, `_preanalysis_finalized_for = ""`
- `_run_defect_scan_bg._apply()`: kein direktes `_set_magic_buttons_enabled(True)` mehr — ruft `_try_signal_preanalysis_done("defect_scan")`
- `_upd` in `_detect_era_genre_bg`: ruft `_try_signal_preanalysis_done("era_genre")` am Ende; No-badge-Pfad dispatcht `_signal_era_done_no_badge` statt sofort zu returnen

**VERBOTEN:** `_set_magic_buttons_enabled(True)` direkt in `_run_defect_scan_bg._apply()` oder `_detect_era_genre_bg._upd()` — immer über `_try_signal_preanalysis_done()`.

**`_carrier_bg` Pflicht-Invarianten (v9.10.97)**:

- MUSS `get_medium_detector().detect(audio, sr, file_ext=Path(file_path).suffix)` nutzen — **NICHT** `medium_classifier.classify_medium()` (kein file_ext-Kontext → gibt bei codec-enkodiertem Analog-Material "unknown" zurück).
- `MediumDetectionResult.transfer_chain` → HTML-Label mit `&nbsp;→&nbsp;` als Trennzeichen für `detected_medium_label`.
- `chip_era` Widget: **NIEMALS** `_show_chip(self.chip_era, ...)` aufrufen — die Ära steht bereits im `detected_medium_label` als `◷ 1970er` HTML-Segment. `chip_era` bleibt dauerhaft unsichtbar.
- In `_detect_era_genre_bg` und `_on_item_finished_with_result`: kein `_show_chip(self.chip_era, ...)`.

### Bridge-Funktionen (vollständige Liste)

`export_guard` | `get_audio_file_validator` | `get_defect_scanner` | `get_defect_type` | `get_quality_mode` | `get_restorer_classes` | `get_medium_classifier_fn` | `get_era_classifier_fn` | `get_genre_classifier_fn` | `get_restorability_estimator_class` | `get_carrier_forensics_fn` | `get_audio_exporter_class` | `cache_defect_result` | `get_cached_defect_result` | `clear_defect_cache` | `warmup_models_background`

## §2.19 Genre-Classifier-Härtung (17 Genres, [RELEASE_MUST])

`GermanSchlagerClassifier` MUSS im Non-Schlager-Zweig 16 Genres parallel scoren:
`Rock`, `Jazz`, `Klassik`, `Oper`, `Pop`, `Blues`, `Soul/R&B`, `Country`, `Folk`,
`Funk`, `Electronic`, `Hip-Hop`, `Metal`, `Latin`, `Gospel`, `Reggae`.

**Open-Set-Invarianten (bindend):**

- `best_score < _NON_SCHLAGER_MIN_SCORE` → `genre_label = "Unbekannt"`
- `best_score - second_score < _OPEN_SET_MARGIN` → `genre_label = "Unbekannt"`
- Bei Tests mit gezielten Top-Genre-Assertions müssen alle nicht getesteten neuen Scorer
    auf Neutralwert (z. B. `0.10`) gepatcht werden, damit Onset-/Centroid-Artefakte keine
    Margin-Kollisionen erzeugen.

**Disambiguation-Gates (Pflicht):**

- `Funk`: Centroid-Bonus nur in warmem Fenster `1800 < centroid_hz < 2800` (kein Rock/Metal-Bright-Bonus).
- `Latin`: Hartes Gate `centroid_hz >= 1800`; BPM-kontextierte Centroid-Bewertung (`bpm>150` hell, `bpm<=150` dunkler).
- `Electronic`: Bei `centroid_hz < 2200` zwingend `score=0.0`.
- `Hip-Hop`: Bei `centroid_hz < 1400` zwingend `score=0.0`.
- `Reggae`: Bei `bpm > 100` zwingend `score=0.0`.
- `Folk`: DR-Guard bei `dr_db > 40` (Penalty gegen Klassik-Fehlzuordnung).
- `Jazz`: Anti-Schlager-Guard `hsi >= 0.58` → `score=0.0`.

**Nicht verhandelbar:**

- Rock-Referenzprofil (`centroid≈3200`, `onset≈4.5`, `hsi≈0.58`, `dr≈20`, `bpm≈120`) darf nicht
    durch `Funk`/`Latin` zu `open_set_unknown=True` degenerieren.
- Jazz-Veto-Guard MUSS stabil bleiben (`n_active>=1` + `alt_genre="Jazz"` + `lang_de>=0.30`).

## §2.36 LyricsGuidedEnhancement (ab 9.10.x — PFLICHT)

Whisper-Tiny ONNX (39 MB) → Phonem-Alignment via wav2vec2_forced_alignment.onnx (125 MB) → Timeline-Segmentierung (vowel_stressed / fricative / plosive / silence) → ContentAwareProcessor (Salienz-Boosts pro Phonemklasse, Stille → aggressivere NR). Latenz ≤ 8 s/min Audio. Shortcut L (Overlay). **Datenschutz-Pflicht**: Lyrics-Text niemals in Logs oder `RestorationResult.metadata`.

**Produktionspfad (v9.10.100, bindend):**

- Autoritatives Produktionsmodul ist ausschließlich `backend/core/lyrics_guided_enhancement.py`.
- Legacy-/Forschungs-Module unter `backend/lyrics_guided/` sind keine normative Referenz für Produktionsänderungen, Bugfixes oder Agenten-Implementierungen, solange sie nicht explizit per Spec/CHANGELOG freigegeben wurden.
- Verboten als Produktionsreferenz: Docker-/MFA-/Python-Whisper-Altpfade aus `backend/lyrics_guided/lyrics_aligner.py` und `backend/lyrics_guided/content_aware_processor.py`.
- Zulässige Persistenz aus Phase 57: Segmentgrenzen, `phoneme_type`, aggregierte Counts, Konfidenzen, Fallback-Flags. Unzulässig: Worttext, Transkript, vollständige Lyrics, Roh-Alignment in Logs, `RestorationResult.metadata`, Checkpoints oder UI-Debug-Strings.
- Datenschutz-Guard ist vor jedem Logging lyrics-bezogener Segmentobjekte verpflichtend; geloggt werden dürfen nur phonemische Klassen und aggregierte Statistik, niemals `word.word`.

**§2.36a Phonem-spezifische DSP-Algorithmen ([RELEASE_MUST], v9.10.90):**
Einheitlicher Gain-Boost reicht nicht — jede Phonemklasse erfordert separate Spektral-Behandlung:

| Phonemklasse | DSP-Anforderung | Kernalgorithmus |
| --- | --- | --- |
| `fricative_stressed/unstressed` | Rauschtextur 4–8 kHz ERHALTEN | Ramp-Gain `g(f) = 1 + str × ramp(4k→8k Hz)`; KEIN Wiener-Smoothing in diesem Band |
| `plosive` | Onset-Transient unveränderlich (0–5 ms) | TransientShapeGuard: onset-Fenster gain=1.0; Burst 100–350 Hz ×1.40; Aspiration 3–8 kHz ×1.20 |
| `vowel_stressed` | Formant-Amplituden proportional heben | LPC Burg Ord.30–40 → F1–F4 → symmetrisches Shelving ±2 HT um jeden Peak |
| `silence` | Aggressivere NR ohne Hard-Gate | OMLSA G_floor=0.05, DeepFilterNet energy_bias=−12 dB |

PGHI nach jeder Spektral-Modifikation. TimbralAuthenticityMetric ≥ 0.87, ArticulationMetric ≥ 0.85 nach phase_57. Vollständige Spezifikation: `spec 03 §2.36a`.

## ML-Plugin-Status (verifiziert, März 2026)

> Vollständige Plugin-Matrix (28 Plugins, Modellpfade, Format, Aufgabe, Fallback-Kaskaden): `.github/specs/08_architecture_and_distribution.md`

**Pflicht-Invarianten für alle ML-Plugins**:

- `ml_memory_budget.try_allocate(name, size_gb)` VOR jedem `InferenceSession`/`torch.load()` — bei Fehler DSP-Fallback
- `ml_memory_budget.release(name)` in allen Fehler-Pfaden nach fehlgeschlagenem Load
- `PluginLifecycleManager.register(name, size_gb, unload_fn)` nach erfolgreichem Load
- VERBOTEN: `plm.try_allocate()` (existiert nicht)

**Lazy-Load-Pflicht** (Budget überschreitet 4 GB allein): AudioSR (5,9 GB), MERT-v1-330M (3,9 GB).
**MelBandRoformer** (860 MB, ONNX): 48k→44.1k→48k Resampling (Lanczos-4, SNR ≈ −0.8 dB) — bei 48k-nativem Modell dieses bevorzugen.
***Diese Richtlinien gelten für alle KI-Agenten (GitHub Copilot, Claude, GPT-Instanzen) die an Aurik 9 arbeiten. Vollständige normative Spezifikation: `.github/specs/01–08`.**
**Stand: März 2026 — Aurik 9.10.57**
