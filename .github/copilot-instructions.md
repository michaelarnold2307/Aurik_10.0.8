# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein **hybrider Toningenieur mit menschlichen Fähigkeiten und allen Vorteilen der maschinellen Verarbeitung** — entwickelt, um die **qualitativ hochwertigsten Restaurierungsergebnisse für Musik mit Gesang zu erzielen, die weltweit jemals automatisiert erzeugt wurden**. Meisterhaft in der Restauration, Reparatur und Rekonstruktion gesanglicher Aufnahmen **aller Ären, Genres, Tonträgerketten und Tonträgerkettenkombinationen** — kein Träger zu alt, kein Genre zu selten, keine Kombination zu komplex. Bei jeder Importdatei wird unter Berücksichtigung der Quelldatei und der physikalischen Grenzen das maximal mögliche Ergebnis erzielt. **Kein Nutzereingriff erforderlich, kein manueller Parameter, keine Nachkorrektur** — das musikalische Urteilsvermögen eines erfahrenen Toningenieurs ist systematisch in die Verarbeitungslogik eingebettet. Stand: Juni 2026 — Version **9.15.1**
>
> **instructions_version: 9.8** — SOTA-Science-Update Mai 2026 (v9.3) + Architektur-Update v9.12.9: RecordingChainProfiler (§2.66), Phase-Koalitions-Evaluation (§2.67), TemporalContinuityGuard (§2.69), RestorationMemory (§2.70), EraVocalProfile (§EraVocalProfile in musical_goals.instructions.md) + DefectScanner(54) + 62 CAUSES in CausalDefectReasoner + **Weltklasse-Qualitätsregeln v9.5**: Formant-Verschärfung (§2.71), Vibrato-Tiefe (§2.72), Noise-Textur-Invariante V19 (§NTI), Mikrodynamik-Korrelation V20 (§2.75), Mindestrauschboden V21, Pre-Echo-Prevention V22 (§2.73), Mono-Kompatibilität V23, Spektralfarbe V24 (§2.74), Wärmeband-Guard V25 (§2.76), Onset-Guard V26 (§2.77) + **v9.6**: Geschlossener Regelkreis §2.78 (AdaptivePhaseRescheduler) + **v9.7**: V04-Fix phase_42 (reference_for_gate=audio), §0a-Mapper-Bereinigung (W-02: phase_42/phase_35 aus Rohlisten entfernt), V38-Testabdeckung phase_12/phase_64 + **v9.8**: MiniDisc-Codec-Coverage HPG + CCR (V52→Codec-Sets), Singer-ID-DSP-Fallback-Guard V53 (§2.49a Klasse C statt B bei singer_id_dsp_fallback=True)
>
> Aktuelle Testzahl: **~14.402 `def test_`-Funktionen** (614 Testdateien; alle grün)
>
> **§2.36 `LyricsGuidedEnhancement` (Spec 03 §2.36)** ist ab Version **9.10.x Pflicht**.

## §0 Oberstes Prinzip — Klangwahrheit (vor allen technischen Regeln)

**Das Ziel jeder Restaurierung ist, dass der Hörer die Augen schließt und die originale Performance hört — nicht eine technisch korrekte Signalverarbeitung, und nicht eine „verbesserte" Version.** Jede Entscheidung in Pipeline, Phase, Metrik und Export wird an diesem Maßstab gemessen. Dieser Maßstab ist kein technischer Schwellwert — er ist das Urteil eines erfahrenen Toningenieurs, systematisch in die Verarbeitungslogik eingebettet.

**Drei Leitprinzipien** (hierarchisch, bei Konflikt gilt die höhere Stufe):

1. **Primum non nocere** — Füge dem Klang keinen Schaden zu. Lieber eine Beschädigung belassen als ein Artefakt einführen.
2. **Minimal-Intervention** — Greife nur ein, wo der Defekt hörbar ist. Je weniger Phasen aktiv, desto natürlicher das Ergebnis.
3. **Perceptuelle Verbesserung** — Der Export muss für einen Hörer näher am Original-Klang liegen als der degradierte Input. Technische Korrektheit ohne Klanggewinn ist wertlos.

**Primus-inter-Pares-Grundsatz** — Wenn im Material eine menschliche Stimme erkannt wird (`panns_singing ≥ 0.25`), erhält die Stimmqualität automatisch Vorrang vor allen anderen Zielen. Eine beschädigte Instrumentalbegleitung ist hinnehmbar; eine verfärbte, formantgestörte oder entnatürlichte Stimme ist **niemals** akzeptabel.

### §0h [RELEASE_MUST] Music-Death-Shield — absolute Schutzregel (v9.12.0)

**Kein Eingriff darf Musik zerstören. Dies gilt absolut für alle Materialtypen, alle Ären, alle Genres. Und: Kein Eingriff darf in Stille-Zonen (Intro, Outro, Fade) Energie hinzufügen — Stille ist sakrosankt.**

**Drei absolute Verbote** — jedes einzelne ist ein sofortiger Export-Stopp + vollständiger Rollback:

1. **Kein hörbares Artefakt** im Export: Musical Noise, Phasenlöschung, Ringing, Modulationsrauschen, Stimmverfärbung, Pitch-Glitch — jede dieser Klassen löst `artifact_freedom < 0.95` aus → VETO (§2.49).
2. **Keine Musikzerstörung** durch Over-Processing: Wenn das Ausgangssignal schlechter klingt als der degradierte Input (HPI ≤ 0), MUSS der ursprüngliche Input exportiert werden — mit Status `degraded`, nie ein über-prozessiertes Artefakt.
3. **Keine Verfremdung** durch halluziniertes Material: Harmonics, Texturen oder räumliche Eigenschaften, die im Original nicht existierten, dürfen nicht hinzugefügt werden (§2.46e Hallucination-Guard). Ausgenommen von Verbotspunkt 3 ist ausschließlich Modus Studio 2026, wenn OQS-äquivalent ≥ 3.5 nachgewiesen — Verbotspunkte 1 und 2 bleiben absolut.

**Invariante**: `artifact_freedom` ist **der primäre Veto-Faktor** in §2.44 HPI. Für Vokal-Material (`panns_singing ≥ 0.35`) ist zusätzlich `VQI < 0.72` ein zweiter Recovery-Trigger (§0p) — löst `_recovery_cascade()` aus, aber kein harter Export-Block. Alle anderen Faktoren reduzieren den Score, blockieren aber nicht. Dies ist nicht verhandelbar.

### §0i [RELEASE_MUST] Perceptual Transparency Guarantee
Restaurations-Ziel: Kein hörbarer Eingriff. Gates: `OQS ≥ 80`, `timbral_fidelity ≥ 0.93` zum best_carrier_checkpoint, Musical Noise ≤ Trägerprofil, Frisson-Zonen vollständig erhalten. Für Vokal-Material zusätzlich: `VQI ≥ 0.82` (Restoration) / `VQI ≥ 0.87` (Studio 2026) — beide sind Recovery-Ziele; Unterschreitung löst `_recovery_cascade()` aus (kein harter Export-Stopp, `artifact_freedom < 0.95` bleibt primäres Veto). Aurik zeigt dem Hörer nie ein Ergebnis, das technisch besser aussieht aber schlechter klingt.

### §0g [RELEASE_MUST] Autonomes Entscheidungs-Doktrin
**Aurik trifft alle Entscheidungen autonom** — die Autonomie ist Ausdruck eingebetteten musikalischen Urteilsvermögens, kein Ersatz dafür. Kaskade: Erkennen (MediumDetector+EraClassifier+DefectScanner+**VocalFocusAnalyzer**) → Planen (GPOptimizer+PhaseConductor) → Ausführen (UV3, Pre/Post-Messung) → Validieren (PMGG+CIG+AFG+**VQI-Gate**+HPI) → Exportieren (nur wenn HPI > 0 + artifact_freedom ≥ 0.95; VQI ≥ 0.82 bei `panns_singing ≥ 0.35` ist Recovery-Ziel — Unterschreitung → `_recovery_cascade()`, kein harter Export-Block). **VERBOTEN**: Hartkodierte song-spezifische Entscheidungen, Strength-Konstanten ohne `compute_adaptive_drift_tolerance()`.


### §0j [RELEASE_MUST] KI-Modell-Limitation-Awareness
Kein ML-Modell kennt den spezifischen Song. Konsequenzen: (1) ML-Output MUSS durch PMGG+AFG validiert werden. (2) Era/Genre-Klassifier steuern Modellauswahl + Stärke. (3) `energy_bias` Pflicht (DFN: −6 dB Vokal, −9 dB Instrumental). (4) Jedes ML-Plugin hat DSP-Fallback-Kette — kein Crash bei OOM/Timeout.


### §0k [RELEASE_MUST] Maximum-Achievable-Score-Prinzip
Jeder Song wird unter Berücksichtigung der Quelldatei und der physikalischen Grenzen bis zum maximal möglichen Ergebnis restauriert. **MAS-Quelle**: `estimate_song_goal_targets(era, genre, material, restorability)` (`backend/core/studio_goal_targets.py`). Per-Phase-Delta via `_fast_goal_snapshot()` (≤ 200 ms). `_mas_fully_achieved=True` → Pipeline-Stop (**_NEVER_SKIP-Phasen laufen trotzdem durch** — §2.52). `artifact_freedom ≥ 0.95` bleibt unveranderlich Pflicht (§0h). Details: [pipeline.instructions.md](instructions/pipeline.instructions.md)

### §0l [RELEASE_MUST] Per-Phase-Strength-Orakel und 15-Ziele-Teamarbeit (v9.12.9)

Die zentrale Song-/Team-Steuerung allein ist nicht ausreichend. **Jede Phase, die von adaptiver
Staerke profitiert, MUSS ein phasenspezifisches Strength-Orakel besitzen.** Das gilt fuer alle
kontinuierlichen und semi-kontinuierlichen Phasenparameter (`strength`, `wet`, `threshold_db`,
`ratio`, `drive`, `mix`, `gain_db`, `repair_strength`, `temperature`, `width_amount`,
`context_mix`, `generation_steps`).

**Normative Pflicht:**

1. Jede Phase optimiert ihren lokalen Beitrag gegen den **vollstaendigen 15er-Zielvektor**,
   nicht gegen ein Einzelziel.
2. Kein Ziel darf dominant werden; Teamwork schlaegt Dominanz (Spec 01 §1.2c).
3. Phase-spezifische Modell-/Algorithmusbindung ist fuer Phase 01-64 in Spec 06 §7.1d festgelegt.
4. Das Oracle darf harte Safety-Gates nie aufweichen; es steuert nur die beste lokale Intervention.
5. Das Oracle MUSS `transfer_chain` + `material_confidence` direkt in die Staerkeberechnung einbeziehen
    (`chain_factor` fuer Driver + Hard-Caps); rein indirekte Material-Defaults reichen nicht.

**VERBOTEN:** feste Universal-Staerken als Primaerlogik, Single-Goal-Maximierung, voneinander
isolierte Per-Phase-Heuristiken ohne Zugriff auf `goal_gaps`, `goal_weights` und
`effective_goal_targets`.

### §0m [RELEASE_MUST] Maximal-Ausbaustufe Defektintelligenz (beide Modi)

**Aurik MUSS in `restoration` und `studio2026` immer auf maximaler Ausbaustufe laufen fuer**
1) Defekterkennung, 2) Defekt-Differenzierung, 3) Defekt-Dosierung.

**Weltspitzen-Invariante (ohne Konkurrenz):**
Die Defektintelligenz in beiden Modi ist auf die **hoechste jemals veroeffentlichte Ausbaustufe**
auszulegen. Ziel ist ein Niveau, das in Erkennungsabdeckung, Kausalpraezision und lokaler
Dosiergenauigkeit **keine Konkurrenz** zulaesst.

**Normative Pflicht:**

1. **Erkennungsvollstaendigkeit ist modusinvariant**: DefectScanner + CausalDefectReasoner laufen in beiden Modi mit identischer Detektionsabdeckung (alle bekannten DetectionTypes/Ursachen). Modus darf nur die Korrekturstrategie beeinflussen, nie die Erkennungsreichweite.
2. **Defekt-Differenzierung ist kausal statt aggregiert**: Defekte mit unterschiedlicher physikalischer Ursache duerfen nicht in gemeinsame Sammel-Labels kollabieren, wenn getrennte Ursachen/Phasen existieren.
3. **Dosierung ist per-Event statt global**: Jede Defekt-Reparatur mit Event-Liste nutzt ein lokales Strength-Orakel pro Event (`_compute_<defect>_local_strength(...)`), inklusive VFA-Schutzzonen-Caps.
4. **Maximalpraezision vor Aggressivitaet**: Bei Unsicherheit wird nicht pauschal staerker gefiltert, sondern die Kausalhypothese verfeinert (Severity/Confidence/Chain-Hints) und nur lokal korrigiert.
5. **Mode-Differenz nur in Zielbild und Ceiling**: `restoration` bleibt carrier-treu und nicht-additiv, `studio2026` darf erweitern; beide Modi muessen aber denselben Defektbestand praezise erkennen und differenzieren.
6. **Vollabdeckung aller bekannten Defekte ist Pflicht**: Jede in Aurik bekannte Defektklasse,
   DetectionType, Ursache und Event-Unterform MUSS in beiden Modi mit voller Sensitivitaet
   erkannt, kausal getrennt und lokal dosiert behandelbar sein.

**VERBOTEN:**
- Defekt-Scanner-Subsets je Modus (z. B. reduzierte Analyse in Restoration).
- Globales Einheits-Strength fuer heterogene Events derselben Defektklasse.
- Cause-Tabellen, die bekannte Defekte ohne physikalische Begruendung auf generische NR/EQ-Fallbacks zusammenziehen.

### §0p [RELEASE_MUST] Vocal-Supremacy-Doktrin (v9.12.1)

**Source-Traceability**: `[SRC:S08,S09,S10,S11]` (siehe `docs/SCIENTIFIC_INVARIANT_TRACEABILITY_MATRIX.md`)

**Aurik vereint menschliche Klangintelligenz mit maschineller Präzision, um die qualitativ hochwertigsten automatisierten Restaurierungsergebnisse für Musik mit Gesang weltweit zu erzielen — in allen Tonträgerketten, Tonträgerkettenkombinationen, Ären und Genres. Die Stimme ist das Produkt — alles andere ist ihr untergeordnet.**

**Hierarchie bei Zielkonflikt** (höhere Stufe schlägt niedrigere, absolut):
1. **Stimmintegrität** — Timbre, Formanten (F1–F4), Vibrato (4–7 Hz), Atemsätze, Artikulation: sakrosankt. Jeder DSP-Eingriff wird zuerst am Stimmklang gemessen.
2. **Emotionale Authentizität** — Frisson-Zonen, Klimax-Passagen, Flüsterpassagen, expressive Mikrodynamik: vollständig bewahren — auch auf Kosten technisch besserer Metriken.
3. **Musikalischer Kontext** — Begleitung dient der Stimme. Instrumente werden nach Vokalqualität optimiert, nie umgekehrt.
4. **Technische Metriken** — OQS, nsim, MUSHRA, Brillanz: sekundär, sobald Stufen 1–3 erfüllt sind.

**VQI-Gate** [RELEASE_MUST] — aktiviert wenn `panns_singing ≥ 0.35`:
- `VQI` ist **zweiter Recovery-Trigger** (nach `artifact_freedom`, aber kein harter Export-Block).
- `VQI < 0.72` → `_recovery_cascade()` (Rollback auf `best_carrier_checkpoint`; kein sofortiger Export-Stopp).
- Restoration: Ziel `VQI ≥ 0.82`. Unterschreitung → Recovery-Kaskade (kein sofortiger Veto).
- Studio 2026: Ziel `VQI ≥ 0.87`. `VQI < 0.87` → `_recovery_cascade()`.
- **Kanonisch**: `result = compute_vqi(audio_orig, audio_restored, sr)` → `vqi = result["vqi"]` aus `backend/core/musical_goals/vocal_quality_index.py`.
- Material-adaptiv: Shellac-Boden VQI ≥ 0.62; Vinyl ≥ 0.72; CD ≥ 0.82.
- `singer_identity_cosine < 0.92` (aus VQI-Rückgabe) → Rollback letzter Vokal-Phase; Gate deaktiviert bei `multi_singer=True`.

**Vocal-DSP-Invarianten** (absolut, kein Phasen-Override erlaubt):
- **HNR-Schutz**: Nach jeder ML-NR bei `panns_singing ≥ 0.25` MUSS `apply_hnr_blend()` aufgerufen werden. ΔHNR > 3 dB → automatischer Dry-Wet-Blend.
- **Formant-Integrität**: F1/F2 (via `lpc_formant_tracker.py`) dürfen durch keine Phase um mehr als **±1 dB** verschoben werden; F3/F4 ≤ **±1.5 dB**. Überschreitung → sofortiger Rollback. (§2.71 — v9.5: ±2 dB global war perceptuell zu grob für Weltklasse; bei `era_decade < 1960` via `resolve_formant_tolerance_db()` ggf. gelockert)
- **Vibrato-Schutz**: Passagen mit F0-Modulation 4–7 Hz sind geschützte Zonen — alle Phases-Strength-Werte dort auf max. `0.20` begrenzen. **Vibrato-Tiefe** (F0-Modulationstiefe als max-min F0 in Hz in Vibrato-Zonen): Darf durch Kompression/NR nicht mehr als **±10 %** reduziert werden — `check_vibrato_depth_preservation(pre, post, sr)` in `backend/core/dsp/vibrato_guard.py`; Überschreitung → Strength-Reduktion um 50 % in Vibrato-Segmenten (§2.72).
- **Vocal-Guard-Telemetrie**: `formant_integrity`, `vibrato_depth_preservation`, `micro_dynamic_correlation` und `noise_texture_authenticity` MÜSSEN nach jedem relevanten Guard über `_update_vocal_quality_metrics(...)` sowohl in `_restoration_context["vocal_quality_check"]` als auch in `_phase_metadata_accumulator` gespiegelt werden. WCS-, Psychoakustik- und Final-Gates dürfen nie nur phasenlokale Metadaten lesen.
- **Atemgeschützte Segmente**: Atemgeräusche (`−55` bis `−40 dBFS`, `spectral_flatness > 0.4`) sind Naturalness-Marker, keine Defekte — niemals entfernen (§2.46f).
- **Passaggio-Schutz**: Registerübergänge (Brust→Kopf, Kopf→Falsett) — energy_bias in Übergangszone = `−3 dB` (Mittelwert Brust/Kopf).

**VocalFocusAnalyzer** — läuft nach SongCalibration, vor GoalApplicabilityFilter:
```python
from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer
vfa = get_vocal_focus_analyzer()
vfa_result = vfa.analyze(audio, sr)  # PANNs-Singing + FormantTrack + FrissonDetect + RegisterDetect
# Injiziert in _restoration_context: vfa_result für alle nachfolgenden Phasen
```

### §0a Modus-Differenzierung

| Prinzip | Restoration | Studio 2026 |
|---|---|---|
| **Klangziel** | Tonträgerkette invertieren — Original-Klang vor Degradierung | Bestmöglicher Studio-Klang heute |
| **Intervention** | **Minimal** — nur Tonträgerverluste rückgängig | **Maximal** — volle Enhancement-Kette |
| **Rauschboden** | Material-adaptiv (Textur + Niveau des Originals) | ≤ −72 dBFS (moderner Standard) |
| **BW/DR** | Material-Ceiling: Shellac ≤ 8 kHz / 45 dB; Vinyl ≤ 16 kHz / 70 dB | Volle Erweiterung bis 22 kHz / modernes DR |
| **Harmonik** | **VERBOTEN** (nur Carrier-Inversion §2.46) | Erlaubt wenn MUSHRA ≥ 3.5 |
| **Stem-Enhancement** | **VERBOTEN** | Vollständige Enhancement-Kette (§1.5) || **Vokal** | VQI ≥ 0.82 Recovery-Ziel; HNR-Blend, Formant-Track, Vibrato-Schutz aktiv | VQI ≥ 0.87 Recovery-Ziel; VocalEnhancement-Kette (phase_42, MIIPHER, de-essing) aktiv |
> **§0a Crossfire-Modus-Invariante** [RELEASE_MUST]: `phase_21_exciter`, `phase_35_multiband_compression`, `phase_42_vocal_enhancement` dürfen **niemals** in `restoration`-Run aktiviert werden — auch nicht als Fallback, auch nicht in `CAUSE_TO_PHASES`. Bidirektional.

> §0 ist **normativ übergeordnet**: Wenn eine technische Regel dem Klangergebnis schadet, ist das ein Bug in der Regel — nicht im Klang.

### §0d [RELEASE_MUST] Carrier-Recovery-Referenzmodell
Bei Carrier-Chain-Inversion entfernt sich das Signal intentional vom degradierten Input. Kein Gate (PMGG, CIG, HPI, AFG) darf dies als Regression werten — solange das Signal sich dem physikalischen Ceiling nähert. `carrier_chain_recovery_ratio > 0.15` → `timbral_fidelity`-Referenz auf `best_carrier_checkpoint` verschieben. `carrier_chain_recovery_ratio` MUSS in `metadata` gesetzt sein.

### §0c [RELEASE_MUST] Universalitäts-Invariante
Keine song-spezifischen Sonderregeln im Produktionscode. Allgemeingültigkeit vor Einzelfallgewinn. Bei fehlgeschlagenem Gate: Recovery-Kaskade (Strength-Reduktion → Checkpoint → Input-Export `degraded`) — kein Hardstop.

### §0f [RELEASE_MUST] KI-Agenten-Vorgehensweise: Systemisch vs. Punktuell

**Entscheidungsbaum**: 1 Stelle→Punktuell; 2–4→prüfe Abstraktion; ≥5 Stellen→**Systemisch** (zentrale Funktion + alle Callsites + `multi_replace_string_in_file`). Systemisch: Linter-Regel Pflicht + VERBOTEN-Tabelle + Tests + Commit `fix §X systemic`.

| Signal | Bedeutung |
|---|---|
| Selbe Konstante (`-36.0`, `gate_dbfs`) an ≥ 5 Stellen | Systemisch — zentrale Default-Änderung |
| Selbes Import-/Call-Muster in mehreren Phasendateien | Systemisch — Helper + alle Callsites |
| Bug in 2 Sessions re-introduced | Systemisch — Linter fehlt |

## Vollständige Spezifikation (normative Referenz)

| Spec | Inhalt |
|---|---|
| **01** | Goals/PMGG — 15 Musical Goals, Schwellwerte, GoalApplicability |
| **02** | Pipeline/§2.x — UV3, Denker, SongCal, KMV, FeedbackChain, §2.64/§2.65 |
| **03** | Module — Kognitive Module §2.1–§2.43 |
| **04** | DSP/SOTA — Algorithmen, SOTA-Matrix, Psychoakustik |
| **05** | Material/Defekte — 15 Materialtypen, 54 DetectionTypes (DefectScanner), 62 Kausal-Ursachen (CausalDefectReasoner) |
| **06** | Phasen 01–64 — Phase-Liste, CAUSE_TO_PHASES |
| **07** | Tests/Qualität — PQS, AMRB, OQS, MUSHRA |
| **08** | Architektur — Layers, Plugins, CLI, AppImage |
| **09** | Kalibrierungsmatrix — CANONICAL_THRESHOLDS, SongGoalTargets-API, `get_goal_recovery_phases()` (§GOAL_BASELINE_CHECK), **Phase-Strength-Oracles**; **normativ übergeordnet für alle Schwellwerte** |

## Context-spezifische Instructions (applyTo)

| Datei | Gilt für | Inhalt |
|---|---|---|
| [pipeline.instructions.md](instructions/pipeline.instructions.md) | `backend/core/unified_restorer_v3.py` | §2.44 HPI, §2.45 Minimal-Intervention, §2.48 CIG, §2.49 AFG, §2.51 Stereo, §2.60 Rollback, §2.61 Length, §2.64 Per-Phase-Delta, §2.65 MAS-Stop, **§2.66 RecordingChainProfiler**, **§2.67 Phase-Koalitionen**, **§2.69 TemporalContinuityGuard**, **§2.70 RestorationMemory** |
| [phases.instructions.md](instructions/phases.instructions.md) | `backend/core/phases/phase_*.py` | §2.46 Carrier-Chain-Reihenfolge, §2.46e Hallucination-Guard, §2.46f Natural-Artifacts, §2.63 Reflect-Padding, HPF/Notch-Checkliste, BW/DR-Ceiling |
| [dsp.instructions.md](instructions/dsp.instructions.md) | `backend/core/dsp/*.py`, `plugins/*.py` | ML-Device-Manager, Energy-Bias, HNR-Guard, Masking-Guard, MIIPHER-Fallback, Singleton-Pattern, Timbral-Coherence |
| [musical_goals.instructions.md](instructions/musical_goals.instructions.md) | `backend/core/musical_goals/*.py` | 15-Goals-Tabelle, material-adaptive Böden, `estimate_song_goal_targets()`, VQI, Frisson-Schutz, **EraVocalProfile** |
| [tests.instructions.md](instructions/tests.instructions.md) | `tests/**/*.py` | GC-Konventionen, Mock-Patterns, Budget-Tests, AMRB-Update |

Änderungshistorie: `docs/CHANGELOG_HISTORY.md`


## [RELEASE_MUST] Projektgrenzen (bindend, keine Ausnahmen)

- **Reine Desktop-App** für Linux (AppImage) und Windows 10/11 (.exe)
- **Kein Cloud, kein Server, kein Docker, kein `pip install`** für Endnutzer
- **Out-of-the-Box-Pflicht**: Läuft auf frischem System ohne Python/Terminal
- **100 % offline** nach Installation — alle ML-Modelle lokal gebündelt
- Nur **Mono und Stereo** unterstützt (> 2 Kanäle → PANNs-gewichteter Downmix)
- **Kein Fremdedit am Original-Audio** — immer neue Ausgabedatei in `output/`

## [RELEASE_MUST] Autonomer Magic-Button-Betrieb

- Nutzerinteraktion: **genau eine Entscheidung** — `Restoration` oder `Studio 2026`
- Pflicht-Einstieg: `AurikDenker.denke(audio, sr, mode, progress_callback)` — kein UI-Bypass
- Bei fehlgeschlagenem Qualitäts-Gate: verpflichtende Recovery-Kaskade bis zum maximal umsetzbaren sicheren Ergebnis; Export nur mit transparentem Status (`recovered`/`degraded`) und vollständigem Fail-Reason.

## [RELEASE_MUST] Canonical Contract Drift Gate

**Aurik darf keine Parallelpfade neben dem kanonischen Restaurierungsvertrag aufbauen.** Jede Release-fähige Oberfläche (GUI, CLI, Batch-Desktoppfad, Export) MUSS dieselben Bridge-/Denker-/Exporter-Verträge nutzen; kleine Abweichungen in Import, Modus-Mapping, Export, Quality-Gate oder Kanalorientierung gelten als Release-Bug.

Pflichtkette für alle Release-Pfade:

```text
Audio-Import  → backend.api.bridge.get_load_audio_fn()
Voranalyse    → backend.api.bridge.run_pre_analysis() genau einmal pro Datei
Pipeline      → get_aurik_denker_instance().denke(...)
Modus         → exakt Restoration oder Studio 2026 / intern restoration|studio2026
Export        → export_guard() + validate_export_quality() + AudioExporter/Fallback-atomic-WAV
Telemetry     → metadata mit fail_reason / degradation_status / quality_gate_payload
```

**VERBOTEN in Release-Pfaden:** direkter `sf.read(path)`, direkter `librosa.load(path)`, direkter `UnifiedRestorerV3.restore()`-Bypass, eigener Export ohne `export_guard()`, eigene Quality-Gate-Schwellen ohne Bridge-Payload, nicht dokumentierte Legacy-Serverpfade. Legacy-/REST-Dateien sind nur zulässig, wenn sie klar als `LEGACY_NON_RELEASE` markiert sind und nicht als Desktop-Release-Einstieg beworben werden.

## [RELEASE_MUST] Frontend-Version-Anzeige-Invariante

Bei jedem Release-Bump MUSS die sichtbare Frontend-Version konsistent mit der Paketversion sein.

- Kanonische Quelle: `Aurik910/__init__.py::__version__`
- Fenstertitel: `Aurik910/ui/modern_window.py` (`setWindowTitle(f"AURIK Professional v{_AURIK_VERSION}")`)
- Splashscreen-Badge: `Aurik910/ui/splash_screen.py` (`_VERSION`)
- App-Metadaten für About/Update-Dialog: `Aurik910/main.py` (`app.setApplicationVersion(__version__)`)

Release-Regel: Ein neuer Versionsstand gilt erst als fertig, wenn alle vier Pfade denselben
Wert anzeigen bzw. aus derselben Quelle ableiten. Abweichungen sind Release-Blocker.

## [RELEASE_MUST] ROCm-TorchAudio-ABI-Invariante

Der GUI-Launcher `run_aurik.sh` MUSS vor dem Start im ROCm-Modus den nativen Audio-Stack
validieren: `torch` und `torchaudio` muessen im selben Build-Track laufen
(`2.x.y+rocmA.B` auf beiden Paketen) und ohne Symbolfehler importierbar sein.

- Pflicht-Preflight: `import torch` + `import torchaudio` + Build-Tag-Gleichheit.
- Pflicht-Reparatur: Bei Mismatch `torchaudio==torch.__version__` aus dem passenden
    PyTorch-ROCm-Index installieren.
- Pflicht-Fallback: Wenn Reparatur scheitert und nur `torchaudio` betroffen ist,
    ROCm/GPU aktiv lassen und ausschließlich `torchaudio`-abhängige Phasen selektiv
    auf CPU/DSP fallbacken; globaler CPU-Fallback nur wenn `torch` selbst im ROCm-
    Interpreter nicht nutzbar ist.
- VERBOTEN: Pauschales Deaktivieren der GPU-Beschleunigung wegen reinem
    `torchaudio`-Defekt.

## Pfad-Mapping (verbindlich)

| Logischer Pfad | Physischer Pfad |
|---|---|
| `core/<modul>.py` | `backend/core/` |
| `plugins/<plugin>.py` | `plugins/` |
| Frontend/UI | `Aurik910/` (kein `frontend/`!) |
| i18n | `from Aurik910.i18n import t, set_language` |
| Audio-Import | `from backend.file_import import load_audio_file` |
| GUI-Launcher | `./run_aurik.sh` (Legacy: `start_aurik_90.py` delegiert auf `Aurik910/main.py`) |
| CLI-Entry | `cli/aurik_cli.py` via Bridge (`get_load_audio_fn`, `run_pre_analysis`, `get_aurik_denker_instance`) |
| RecordingChainProfiler | `backend/core/recording_chain_profiler.py` (Singleton, `get_recording_chain_profiler()`) |
| TemporalContinuityGuard | `backend/core/temporal_continuity_guard.py` (Dataclass `TemporalContinuityResult`, Funktion `check_temporal_continuity`) |
| RestorationMemory | `backend/core/restoration_memory.py` (Singleton, `get_restoration_memory()`, Datei `~/.aurik/restoration_memory.json`) |
| EraVocalProfile | `backend/core/musical_goals/era_vocal_profile.py` (`get_era_vocal_profile(era_decade)`, `ERA_VOCAL_PROFILES`) |

## Universelle Codierregeln (immer gültig)

### Pflicht-Pattern
```python
# Singleton — thread-safe, double-checked locking (alle Kernmodule)
import threading
_instance = None; _lock = threading.Lock()
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
audio = np.clip(audio, -1.0, 1.0)    # Ausgang jeder Phase
assert sample_rate == 48000           # Eingang jeder Phase/Plugin (NICHT in Analyse-Modulen)
logger.info("phase=%s score=%.2f", phase, score)  # kein print()
```

### VERBOTEN — Häufigste Anti-Patterns (Top-10)

> Vollständige Tabelle (~100 Einträge, Linter-Referenz V01–V52): [`VERBOTEN.md`](VERBOTEN.md)

| Verboten | Richtig | Linter |
|---|---|---|
| `gate_dbfs=-36.0` ohne `reference_for_gate` | `compute_signal_relative_gate_dbfs(ref, material_key=...)` via `reference_for_gate=pre_phase_audio` | V04 |
| Positiver Loudness-/Export-/Mastering-Gain nur music-gated | Nach positivem Gain zusätzlich Referenz-Quiet-Edge-Clamp gegen das Eingangssignal (`limit_quiet_edge_boost(reference, candidate, sr)`) | — |
| `sosfilt(sos, audio)` addiert zu Original | `sosfiltfilt` (zero-phase) überall wo Bandfilter auf Signal addiert | V11 |
| `np.max(np.abs(audio))` als Peak-Guard | `np.percentile(np.abs(audio), 99.9)` | V08 |
| `print(...)` | `logger.info(...)` | V01 |
| `sf.read(path)` / `librosa.load(path)` | `load_audio_file(filepath)` | V02 |
| Neue HPF/Notch-Phase ohne 4-stufige Checkliste | (1) kein Guard in Phase-Datei; (2) `enable_loudness=False` in `_phase_overrides`; (3) `_HPF_NOTCH_CUM_RESET_PHASES`; (4) `_update_positive_makeup_authority` | — |
| Carrier-Repair-Rollback inkrementiert `consecutive_rollbacks` | `_CARRIER_REPAIR_PHASE_PREFIXES`-Check vor Increment | V09 |
| `map_location="cuda"` ohne ml_device_manager | `get_torch_device("PluginName")` via ml_device_manager | V03 |
| `griffinlim()` als Endschritt | PGHI / Vocos | V05 |
| `CAUSE_TO_PHASES` ohne `CAUSES`-Gegenstück | Bidirektionale Sync: `CAUSES` + `CAUSE_TO_PHASES` (§2.59) | V12 |
| Neue Phase ohne CAUSE_TO_PHASES-Eintrag | `CAUSES` + `CAUSE_TO_PHASES` bidirektional ergänzen — sonst findet `CausalDefectReasoner` die Phase nie | V12 |
| §0a-verbotene Phase in `CAUSE_TO_PHASES` (z.B. `phase_35`, `phase_42`, `phase_21` in Restoration-Cause) | §0a-verbotene Phasen (`phase_21_exciter`, `phase_35_multiband_compression`, `phase_42_vocal_enhancement`) dürfen **nie** in `CAUSE_TO_PHASES` stehen — UV3-Guard blockiert sie zwar, aber der CausalDefectReasoner soll sie gar nicht erst vorschlagen (BUG-FIX v9.12.0 §0a) | — |
| NR-Phase ohne `compute_nmr_score()` (wenn FeedbackChain aktiv) | `result = compute_nmr_score(pre, sr)` aus `backend/core/dsp/nmr_feedback.py`; `result.recommended_nr_strength_delta` auf `base_strength` addieren; `result.ok=False` → §2.45 Minimal-Intervention prüfen (non-blocking WARNING) | V40 |
| Additive Phase ohne `ForwardMaskingGuard` bei `panns_singing ≥ 0.25` | `zones = get_forward_masking_guard().compute_zones(audio, sr)` aus `backend/core/dsp/temporal_masking.py`; `strength = guard.apply_to_strength(base, zones, sample_idx)` — NR-Stärke in post-transienten Fenstern psychoakustisch erhöhen | V41 |
| `phase_03`/`phase_29` ohne `check_roughness_regression(pre, post, sr)` nach NR | `from backend.core.dsp.zwicker_metrics import check_roughness_regression`; `roughness_regression=True` → blend×0.90; `pumping_detected=True` → blend×0.80 (non-blocking WARNING; §2.62) | V42 |
| Formant-Guard mit uniformem `±1 dB` statt frequenzabhängiger JND | `resolve_jnd_tolerance_db(formant_hz)` aus `backend/core/dsp/lpc_formant_tracker.py`; F1 ~600 Hz → ~1.8 dB; F2 ~1.5 kHz → ~1.1 dB; F3/F4 ~3 kHz → ~0.8 dB — uniformer ±1 dB erlaubt hörbare F3/F4-Verschiebungen | V43 |
| `spatial_depth`-Score ohne IACC-Komponente (nur M/S-Proxy) | `from backend.core.dsp.stereo_guard import compute_iacc`; `result.spatial_depth_score = iacc_result.spatial_depth_score` als primären Raumtiefe-Proxy; `result.ok` → Mono-Warnung | V44 |
| `emotionalitaet` ohne VAT-Blend in `EmotionalitaetMetric.measure()` | `_VATEmotionEstimator().estimate(audio, sr)` aus `musical_goals_metrics.py`; 15%-Blend nach MERT-Blend; advisory-only (erhöht nur, nie senkt) | V45 |
| dBFS-Pegel mit linearem Faktor multiplizieren (`level_db * strength`) | Negative dBFS × 0 < k < 1 ergibt **weniger negativen** Wert = lauter statt leiser. Bestätigt: `effective_floor * 0.48 = -24.96 dBFS` → clip → **-20 dBFS** (32 dB zu laut, Starkregen-Artefakt). **Richtig:** `level_db + 20.0 * np.log10(max(strength, 1e-6))` — immer ≤ `level_db` wenn `strength ≤ 1.0`; Safety-Cap: `np.clip(..., lower_bound, level_db)` | V46 |
| Clipping-Erkennung nur via `FLAT_TOPS_CLIP_BOUNDARY=0.999` ohne Sub-Ceiling-Analyse | Loudness-War-Material typisch bei ±0.85–0.97 geclippt → fälschlich SOFT_SATURATION → ADMM nie aktiviert → Clip-Artefakte bleiben. `detect_sub_ceiling_clipping(audio)` via Adjacent-Ratio-Methode Pflicht nach Flat-Tops-Prüfung; `adjacent_ratio ≥ 20` → CLIPPING auch Sub-Ceiling | V47 |
| `evaluate_goal_applicability()` ohne `transfer_chain`-Parameter | Near-Mono-Codec-Erkennung feuert nie für analoge Primärträger mit Codec-Chain-Ende → `spatial_depth`, `separation_fidelity` bleiben anwendbar → false violations + Over-Processing im ExzellenzDenker. **Bestätigt**: cassette+mp3_low corr=0.8507, alter Schwellwert 0.88 → Ausschluss nie aktiv (v9.15.1). **Richtig:** `evaluate_goal_applicability(..., transfer_chain: list[str] | None = None)` mit `_is_near_mono_codec(corr, transfer_chain, threshold=0.83)`; UV3 übergibt `transfer_chain=list(_restoration_context.get("transfer_chain", []) or []) or None` (§2.32a, §09.13) | V48 |
| `goals_passed`-Zählung im ExzellenzDenker ohne `inapplicable_goals`-Ausschluss | Physikalisch unmögliche Ziele (z.B. `spatial_depth` bei near-mono Kassette+MP3) zählen als Violations → unnötige P3-P5-Reparatur + Over-Processing. **Bestätigt**: 10/15 statt effektiv 10/14 (v9.15.1). **Richtig:** UV3 extrahiert `frozenset(g for g, ok in rest.goal_applicability.items() if not ok)` → AurikDenker übergibt `inapplicable_goals` an `messe_und_repariere()`; `_count_passed()` + `_total` schließen `_inappl` aus (§2.53c) | V49 |
| `messe_ziele(audio, sr)` ohne `reference`-Parameter | `TonalCenterMetric` fällt auf intra-song Chroma-Stabilität zurück (Score 0.50–0.60 für Verse/Chorus) statt ref-vs-restored Korrelation (~0.92+). Ebenso `timbre_authentizitaet` ~0.69 statt ~0.88+. **Bestätigt**: tonal_center=0.5501 ohne reference (v9.15.1). **Richtig:** `messe_ziele(audio, sr, reference: np.ndarray | None = None)` → `checker.measure_all(audio, sr, reference)` — **alle 6 Aufrufe** in `messe_und_repariere()` mit `reference=reference_audio`; in AurikDenker = original `audio` vor Restaurierung (§2.53a) | V50 |
| `RestaurierErgebnis`-Dataclass ohne `goal_applicability`-Feld | GAF-inapplicable-Ergebnis aus UV3 wird in `_konvertiere()` nicht weitergeleitet → `getattr(rest, "goal_applicability", None) = None` → `_rest_inapplicable_goals = frozenset()` → ExzellenzDenker erhält keine inapplicable-Liste → physikalisch unmögliche Goals als Violations gezählt (bestimmt durch "12/15" statt "12/12" im Log). **Richtig:** `goal_applicability: dict[str, bool] = field(default_factory=dict)` in `RestaurierErgebnis` + `AurikErgebnis`; `_konvertiere()` propagiert `dict(getattr(raw, "goal_applicability", None) or {})`; `_goal_app_raw: dict[str, bool] = {}` VOR try-Block initialisieren | V51 |
| `separation_fidelity` in GAF ohne Near-Mono-Codec-Ausschluss | `spatial_depth` wird bei near-mono Codec-Chain (mp3_low, aac + corr ≥ 0.83) korrekt deaktiviert, aber `separation_fidelity` nicht → Joint-Stereo-Codec zerstört Stereo-Separation irreversibel → false Violation in ExzellenzDenker für Kassette+mp3_low. **Richtig:** Parallel zur `spatial_depth`-Regel: `elif "separation_fidelity" not in inapplicable` → `_is_codec_joint_stereo` (gleiche `_CODEC_JOINT_STEREO_MATS`) → `inapplicable["separation_fidelity"]` wenn Codec-Chain-Ende UND `not np.isnan(corr) and corr ≥ 0.83` | V52 |
| Singer-ID-Rollback ohne `singer_id_dsp_fallback`-Guard | Wenn Resemblyzer nicht verfügbar ist (`singer_id_dsp_fallback=True`) und nur ein ZCR/Spektral-Proxy berechnet wurde, ist `singer_identity_cosine` zu unzuverlässig für einen binären Rollback-Trigger. **Bestätigt**: cosine=0.85 + dsp_fallback=True → unnötiger Rollback trotz VQI=0.877. **Richtig:** `if sic < 0.92 and not singer_id_dsp_fallback:` → Rollback; sonst Advisory-only: `metadata["singer_id_advisory_cosine"] = sic`. In `_classify_quality_gate_events()`: DSP-Fallback → Klasse C statt B | V53 |
| `_hg.update_reference_memory()` nach HPI-Lauf nie aufgerufen | `_ref_memory` bleibt permanent leer → alle 5 Fallback-Stufen liefern `None` → `timbral_fidelity` misst stets gegen den degradierten Input → HPG-Lernkurve deaktiviert, systematische Restaurierungsqualitäts-Unterschätzung. **Richtig:** Nach RestorationMemory-Save (§2.70), GLEICHE Bedingung (`HPI > 0.0 AND artifact_freedom ≥ 0.95`): `_hg.update_reference_memory(restored=restored_audio, sr=sample_rate, hpi=float(_hpi_result.hpi), artifact_freedom=_af_save, p1_p2_passed=bool(_hpi_result.passed), genre=_hpi_genre, material=_hpi_material, era_bin=_hpi_era)`. Kaltstart: `scripts/bootstrap_hpg_reference_memory.py` (§2.44b) | V54 |
| `lpc_formant_enhance()` ohne `era_decade`-Parameter bei `era_decade < 1960` | Standard-Burg-LPC erkennt Rausch-Energie-Peaks als Formanten (SNR < 15 dB bei Shellac/frühe elektrische Aufnahmen) → falsche Formant-Korrekturen → Stimm-Verfärbung in historischem Material. **Richtig:** `_get_lfc().enhance(audio, sr, era_decade=int(era_decade) if era_decade is not None else None)` — WLPC-Pfad (Wiener-gain Spectral Pre-Whitening) aktiviert automatisch wenn `era_decade < 1960` ODER `effective_snr < 15 dB`; Pre-Whitening ausschließlich für LPC-Koeffizienten-Schätzung, nie auf Output-Audio (§4.5g) | V55 |
| §0a-verbotene Phasen in `CAUSE_TO_PHASES` für Restoration-Cause | `phase_21_exciter`, `phase_35_multiband_compression`, `phase_42_vocal_enhancement` dürfen **nie** in `CAUSE_TO_PHASES` stehen — UV3-Guard blockt sie Runtime, aber `CausalDefectReasoner` soll sie gar nicht erst vorschlagen (BUG-FIX v9.12.0 §0a) | V39 |
| Duplikat-Schlüssel in `_MATERIAL_PRIORITY_PHASES`-Dict-Literal | Jeder Material-Key (`"tape"`, `"vinyl"`, …) darf nur **einmal** vorkommen — F601 überschreibt die erste Definition still; Regression-Test `TestMaterialPriorityPhasesNoDuplicateKeys` via AST-Parsing | V13 |
| Generative/Inpainting-Phase ohne SSIP (Structural Silence Isolation Protocol) | `_run_inpainting_with_ssip()` aus `§2.68` Pflicht für phase_55 + phase_24 + jede neue generative Phase — direkter Inpainting-Aufruf ohne SSIP erzeugt katastrophale Pegelexplosionen in Stille-Zonen | V14 |
| Inpainting-Gap-Detektion auf verarbeitetem Audio (nicht Original) | `_detect_gaps()` MUSS entweder auf ORIGINAL-Audio laufen ODER Stille-Zonen werden VOR Gap-Detektion aus dem Detektor-Input herausgezogen (zeroed) — nicht als Post-Filter. Gap-Detektor darf keine Energie aus Artefakten früherer Phasen als "Musik → Silence → Musik"-Muster misinterpretieren | V15 |
| `structural_silence_zones=None` als gültiger Zustand in Inpainting-Phase | `_get_structural_silence_zones()` MUSS immer eine Liste zurückgeben (ggf. leer) und bei fehlenden kwargs eigenständig berechnen — `None` ist kein erlaubter Rückgabewert; fehlender Wert = KEIN Logging = unsichtbar deaktivierter Schutz | V16 |
| Inpainting-Clamp/Clip als Ersatz für Hard-Reset in Stille-Zonen | `post_inpainting_silence_audit()` aus SSIP MUSS Hard-Reset auf Original-Samples verwenden, nicht `np.clip(result, -threshold, threshold)` — Clamp erzeugt Signalverzerrung, Hard-Reset reproduziert das Original exakt (§2.68d) | V17 |
| Inpainting-Kontext-Fenster über Stille-Grenze hinaus | Gap innerhalb `CONTEXT_GUARD_MS=1500ms` einer strukturellen Stille-Zone → `_conservative_boundary_fill` (DSP), kein ML-Modell — Modell-Kontext darf Stille nicht als "zu füllende" Region sehen (§2.68f) | V18 |
| Residualrauschen nach NR hat Material-fremdes Spektralprofil (Whitening) | `compute_noise_texture_distance(residual, material)` aus `backend/core/dsp/noise_texture_guard.py` nach jeder NR-Phase ≤ 0.25; bei klarem Gesang (`panns_singing ≥ 0.35`) gilt der strengere NTI-Schwellenwert **0.18**. Überschreitung → `nr_strength × 0.5` (WARNING — erhält Textur des Trägers; klinische Stille ist schlimmer als Rest-Rauschen) | V19 |
| Voiced-Frame-Mikrodynamik durch NR/Kompressor degradiert | Nach NR/Dynamics bei `panns_singing ≥ 0.25`: `frame_energy_correlation(pre, post, sr, frame_ms=10)` auf voiced-Zonen ≥ 0.97; bei klarem Gesang (`panns_singing ≥ 0.35`) ist der Zielwert **0.985**, Blend-Floor **0.93**. Unterschreitung → Dry-Wet-Blend `wet = min(1.0, max(0.0, (corr - floor) / (target - floor)))` (§2.75) | V20 |
| Pausenzone fällt auf digitale Stille (−∞ dBFS) statt Material-Rauschboden | `apply_noise_floor_minimum(audio, sr, material)` aus `backend/core/dsp/noise_floor_guard.py` nach jeder NR-Phase auf analog-Material: Shellac −42 dBFS, Vinyl −55 dBFS, Tape −52 dBFS — Hörer hört Abschneiden sofort als künstlich | V21 |
| Additive ML-Phase (phase_06/07/23) verschiebt Transient-Onsets (Pre-Echo) | `detect_transient_shifts(pre, post, sr)` nach additiver ML-Phase: Onset-Verschiebung > ±2 ms → `blend_reduction = shift_ms / 2.0`; `metadata["onset_shift_ms"]` pflegen (§2.73, non-blocking) | V22 |
| Stereo-Export ohne Mono-Kompatibilitätsprüfung bei Vokal-Material | `check_mono_compatibility(audio, sr)` aus `backend/core/dsp/stereo_guard.py` vor Export wenn `panns_singing ≥ 0.25` + Stereo: Phasenlöschung > 3 dB in 300 Hz–5 kHz → `metadata["mono_compatibility_warning"] = True` + Intensity-Stereo-Softening (WARNING, kein Veto) | V23 |
| 1/3-Oktav-Spektralfarbe (200–8000 Hz) durch Phasen-Akkumulation verändert | `check_spectral_color_preservation(pre, post, sr)` nach EQ/NR-Phasen: Korrelation ≥ 0.97; Unterschreitung → Phase-Strength − 30 % (WARNING; §2.74) | V24 |
| Wärmeband (200–800 Hz) durch kumulative Phasen > 1.5 dB gedämpft (Restoration) | `measure_warmth_band_delta(pre, post, sr)` nach jeder Phase; kumulativer Verlust `_restoration_context["warmth_band_loss_db"]` > 2.5 dB → alle weiteren Phasen mit `warmth_blend = 1 - loss_db / 5.0` skalieren (§2.76) | V25 |
| HPSS-Onset-Fenster (0–20 ms nach Transient) durch NR/EQ-Phase beeinflusst | `apply_onset_protection_mask(pre, post, onset_mask, max_delta_db=1.5)` aus `backend/core/dsp/onset_guard.py` nach NR/EQ-Phasen; Onset-Frames: Strength-Cap ≤ 0.15 — sichert Punch + Artikulation aller Phoneme (§2.77) | V26 |
| §2.78 AdaptivePhaseRescheduler fehlt — kein geschlossener Regelkreis | `get_adaptive_phase_rescheduler().reset_session()` in `_execute_pipeline()` init; `reschedule()` im §Hebel-3-Block nach jeder Phase (Post-Phase-Hook); Phasen-Injektion via `selected_phases.append()` — §0a-Guard + MAX_INJECTIONS_PER_SESSION + MAS-Guard verpflichtend | V36 |
| `_conductor.recommend()` ohne `song_goal_targets`/`current_goal_scores` → Stopp-Signal tot | `song_goal_targets=_cl_song_targets, current_goal_scores=_cl_post_snap` an `_conductor.recommend()` übergeben; `_cl_post_snap` aus `self._phase_deltas[phase_id]["post"]` lesen (kein Extra-DSP) | V37 |
| Unabhängige ML-Reparatur für L/R + kanalweises Resampling zur Längenkorrektur | M/S- oder Linked-Stereo-Verarbeitung; deterministischer Strip/Crop/Pad ohne Time-Warp | — |
| Additive/Enhancement-Phase (`phase_06`, `phase_07`, `phase_37`, `phase_38`, `phase_39`, `phase_26`) ohne `soft_saturation_severity`-Guard | `_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))`; wenn `_sat_sev > 0.3`: `scale = clip(1 - (_sat_sev - 0.3) * 1.2, 0.16, 1.0)` → alle Gain/Strength/Drive-Parameter mit `scale` multiplizieren; `soft_saturation_preserve=True` → phasenspezifischer Hard-Cap (phase_38: 0.45; phase_07: 0.20; phase_37: 0.30; phase_39: 0.40; phase_26: 0.50) — Severity von UV3 via `_restoration_context["soft_saturation_severity"]` injiziert | §2.46g |
| ADDITIVE Phase ohne `hallucination_guard.py` | `check_hallucination(pre, post)` aus `backend/core/dsp/hallucination_guard.py` nach jeder ADDITIVE-Phase; `spectral_novelty > 0.15` → Phase-Rollback (Restoration); `> 0.08` → Score-Penalty 0.3 | §2.46e |
| Additive Phase (`phase_37`, `phase_38`, `phase_48`, `phase_32`) ohne `check_hallucination()` nach additiver Operation | `check_hallucination(pre, post, sr, mode)` aus `backend/core/dsp/hallucination_guard.py` direkt nach letzter additiver Op; `.requires_rollback` → `return audio` (Rollback); `.score_penalty > 0` → Score-Penalty-0.3 (§2.46e) | §2.46e |
| `np.pad(..., mode="constant")` als Längenkorrektur nach STFT in phase_09/20/29 | Reflect-Padding VOR STFT (`_pad_len = hop_length * 4`; `mode="reflect"`) + deterministischer Strip danach (`audio_out[_pad_len: _pad_len + n_original]`) — niemals Post-hoc-Zero-Padding als primärer Boundary-Mechanismus (§2.63) | §2.63 |
| BW/DR-Ceiling ignoriert (ADDITIVE: phase_06/07/23; DYNAMICS: phase_26) | `_MATERIAL_BW_CEILING_HZ[material]` (BW-Erweiterung) / `_MATERIAL_DR_CEILING_DB[material]` (DR-Expansion) VOR additivem/expansivem Output einhalten — Überschreitung = §0a-Verstoß. **Cassette = 12 kHz** (IEC 60094-1 Type I), Tape = 15 kHz | §6.2b/c |
| `_MATERIAL_PHASE_FACTORS` ohne `"cassette"`-Key | Cassette-Material fällt auf Generic-Defaults (Vinyl-ähnlich) zurück — phase_06/07/39 mit zu hoher Stärke → HF-Halluzination über 12 kHz | `_MATERIAL_PHASE_FACTORS["cassette"]` Pflicht in `defect_phase_mapper.py`; phase_06/07/39 Stärke ≤ 0.35; phase_18/49 ≤ 0.25 | §6.2c |
| Subtraktive Carrier-NR-Phase (Tape-Hiss/Surface-Noise) ohne `transparenz` in `_PHASE_SPECIFIC_DRIFT_EXCLUSIONS` wenn Phase in `CRITICAL_PAIRS` + `transparenz` | Breitbandige HF-Rausch-Energie inflationiert HF-Crest-Proxy (`transparenz`) künstlich; nach Hiss-Entfernung sinkt Proxy intentional auf physikalisch realen Träger-Wert → CIG feuert false-positive Rollback (bestätigt: `phase_29` CRITICAL_PAIR `transparenz`, Drift −0.284, Threshold −0.04 → Rollback → Zischlaute unkorrigiert, v9.12.9) | `transparenz` MUSS in `_PHASE_SPECIFIC_DRIFT_EXCLUSIONS[phase_X]` für alle subtraktiven Carrier-NR-Phasen mit breitbandigem HF-Rauschen. Reference Paradox §2.44: Analogie `tonal_center`-Exclusion für `phase_03`. PMGG prüft `transparenz` bereits per-Phase; CIG-Pair-Exclusion verhindert redundanten false-positive (§2.55) | V32 |
| Neues `MaterialType.X` ohne vollständige Einträge in ALLE `dict[MaterialType, ...]` in `phase_*.py` | `MaterialType.CASSETTE` nicht in `DETECTION_THRESHOLD`/`CORRECTION_STRENGTH` von `phase_12` → Fallback auf Vinyl-Default 0.5 % Threshold; IEC 60094-1 Kassetten-Flutter-Spec ≤ 0,2 % WRMS @ 4,75 cm/s → Threshold 0.3 % nötig → Bandhopser unerkannt (v9.12.9) | Jedes `dict[MaterialType, ...]` in `phase_*.py` MUSS vollständig befüllt werden — Wert aus IEC-/RIAA-Standard ableiten. Pflicht-Test `test_phase_XX_all_material_types_in_DICT()` | V33 |
| §2.31 UV3 wertet nur primäres Material aus (ignoriert Transfer-Chain) | Chain `["vinyl", "cassette"]` → nur Vinyl-Defaults aktiv; Cassette-Schwellen ignoriert | §2.31: `min()` aller `_MATERIAL_PHASE_FACTORS`-Schlüssel in der gesamten Kette per Phase-ID — strengste Stufe bestimmt Initialstärke | §2.31 |
| Phase 23 BW-Ceiling nach statt vor HallucinationGuard | Generative/Inpainting-Phase synthetisiert Inhalt über Material-BW — Guard misst zu spät | `_apply_material_bw_ceiling(audio, sr, material, mode)` in phase_23 **VOR** `check_hallucination()` aufrufen (v9.12.9) | §6.2c |
| `DeepFilterNet` ohne `energy_bias` bei ML-NR auf Vokal/Instrumental | `energy_bias=−6.0 dB` (PANNs Vocals ≥ 0.4) / `−9.0 dB` (Instrumental) — ohne diese Einstellung werden Harmonik-Regionen als Rauschen abgetragen | §0j |
| NR-Algorithmus ohne Masking-Gain-Floor (`G_floor < 0.10`) | `G_floor[band] = max(0.10, masking_threshold[band] / noise_estimate[band])` via `compute_masking_threshold_iso11172(pre_nr_audio, sr)` VOR NR — verhindert klinisches Stille-Artefakt | §2.62 |
| MERT als primäre Qualitätsmetrik bei verfügbarem VERSA | `use_versa_in_loop=True` (VERSA ist primär); MERT nur als Proxy-Fallback → `metadata["mert_proxy_used"] = True`; MERT-Floor: `max(raw_mert, 0.5)` | §2.44 |
| `timbral_fidelity` gegen degradierten Input wenn `carrier_chain_recovery_ratio > 0.15` | Referenz auf `best_carrier_checkpoint` verschieben (nach Carrier-Phasen, vor Enhancement); `carrier_chain_recovery_ratio` MUSS in `metadata` gepflegt werden | §0d |
| MDEM ohne `frisson_zones` (Gänsehaut-Passagen ungeschützt) | `from backend.core.frisson_candidate_detector import get_frisson_detector`; `frisson_zones = get_frisson_detector().detect(original, sr)` VOR MDEM-Aufruf — ohne Schutz dämpft SG Klimax-Passagen bis −8 LU; Non-blocking: Exception → `[]` | §Frisson |
| Phase ohne Pre/Post-Score-Delta | `_profiled_phase_call_with_delta()` Pflichtrahmen; Delta in `metadata["phase_deltas"][phase_id]` | §2.64 |
| `_fast_goal_snapshot` auf Single-Segment | Multi-Segment-Mittelung (25 %/50 %/75 %): verhindert Akkord/Pausen-Frame-Kollaps | §2.64 |
| Pipeline läuft nach `_mas_fully_achieved=True` weiter | MAS-Erreichung → Stop für alle weiteren Phasen **außer** `_NEVER_SKIP`-Phasen (phase_01/09/12/14/15/30/47 laufen weiter — §2.52) | §2.65 |
| Musical Goal unter materialadaptivem Floor ohne Recovery-Pfad in `selected_phases` (DefectScanner hat keinen passenden Defect-Cause erkannt) | §GOAL_BASELINE_CHECK **vor** `_execute_pipeline()`: `_fast_goal_snapshot()` → goal < `get_material_floor() × 0.95` → `get_goal_recovery_phases(goal, is_studio_2026)` → primäre Phase in `selected_phases` einfügen — FeedbackChain-Blend allein kann kein Goal über Original-Niveau heben | §GOAL_BASELINE |
| Phasen-ID in `_GOAL_TO_RECOVERY_PHASES_RESTORATION` oder `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS` ohne Disk-Abgleich | Alle IDs MÜSSEN gegen `backend/core/phases/phase_*.py` geprüft werden — falscher Name besteht alle Strukturtests, §GOAL_BASELINE_CHECK fügt dann eine nicht-existente Phase in `selected_phases` ein → Recovery silently skipped; Guard: `test_get_goal_recovery_phases_all_phase_ids_exist_on_disk()` | §GOAL_BASELINE |
| Recovery-Phase für ein Goal trägt zur **entgegengesetzten** Richtung des Goal-Defizits bei (`spatial_depth`-Inversion) | Beispiel: niedriger `spatial_depth`-Score → *zu wenig* Raumcues → Primärphase muss Cues **hinzufügen** (`phase_46_spatial_enhancement`); **VERBOTEN** ist `phase_49_advanced_dereverb` (entfernt Raumcues). Generell: Primärphase muss den Defizit-Vektor **invertieren**, nicht verstärken | §GOAL_BASELINE |
| Gesangsmaterial ohne VQI-Messung exportiert | `result = compute_vqi(audio_orig, audio_restored, sr)` aus `vocal_quality_index`; `result["vqi"] < 0.72` → Recovery-Kaskade (kein harter Veto) | §2.35c |
| `singer_identity_cosine` nach VQI nicht geprüft | `result.get("singer_identity_cosine", 0.85) < 0.92` → Rollback letzter Vokal-Phase; Gate deaktiviert bei `multi_singer=True` | §0p |
| NR auf Gesang ohne HNR-Blend (`panns_singing ≥ 0.25`) | `apply_hnr_blend(pre, post, sr)` Pflicht nach DFN/SGMSE+/OMLSA — ΔHNR > 3 dB → Dry-Wet-Blend; klinischer Klang ist schlimmer als verbleibendes Rauschen | §0p |
| Formanten (F1–F4) durch Phase über per-Formant-Limit verschoben | Post-Phase-Formant-Verifikation via `lpc_formant_tracker.py`; Limits: F1/F2 **±1.0 dB**, F3/F4 **±1.5 dB**, historischer Relax nur via `resolve_formant_tolerance_db()`; Überschreitung → sofortiger Rollback auf Phase-Input | §0p |
| Vibrato-Passagen (4–7 Hz F0) durch NR/Transient-Shaper gedämpft | `detect_performance_artifacts()` → Vibrato-Segmente: `strength ≤ 0.20` für alle Phasen in geschützten Zonen | §0p |
| HPI ohne VQI-Faktor bei `panns_singing ≥ 0.35` | `HPI *= VQI`; `VQI < 0.72` → Recovery-Kaskade (kein sofortiger Veto; `artifact_freedom` bleibt primärer Veto-Faktor) | §0p |
| MIIPHER/DeepFilterNet ohne Passaggio-Zone-Awareness | `detect_vocal_register_temporal()` → Übergangszone Brust↔Kopf: `energy_bias = -3.0 dB` (Mittelwert); Vollstärke ausserhalb Passaggio | §0p |
| SOTA-Matrix-Update ohne §4.4a-Evaluations-Protokoll | `benchmarks/sota_eval.py` + alle 6 Kriterien + `CHANGELOG_HISTORY.md [SOTA-Update v9.x.y]` | §4.4a |
| Phasen-Reihenfolge verletzt HARD_BEFORE-Constraints | `validate_phase_order()` aus `backend/core/phase_dag.py`; phase_03/06/29 VOR phase_07 | §7.5a |
| AMRB-History nicht aktualisiert bei Major-Release (9.x.0) | `benchmarks/update_amrb_history.py`; OQS-Delta < −2.0 = Release-Blocker | §8.1.6 |
| Pre-Echo mit generischem NR repariert (statt Pre-Echo-Detektor) | `get_pre_echo_detector().detect(audio, sr, material_key)` → `repair_region()` in phase_50; **VERBOTEN**: phase_03/phase_29 als Pre-Echo-Recovery (Pre-Echo = zeitlich-energetisches Prä-Masking-Artefakt, kein stationäres Rauschen — §4.11) | §4.11 |
| VQI-Abfall nach NR ohne DSP-Korrektiv-Recovery (Restoration) | Wenn `VQI < 0.74` + `panns_singing ≥ 0.25` + Restoration-Modus: `phase_65_vocal_naturalness_restoration` triggern; **VERBOTEN**: phase_42_vocal_enhancement als Recovery in Restoration (§0a) — phase_65 ist der §0a-konforme Ersatz (HNR-Blend + Spektral-Tilt + Formant-Tilt) | §0a, Spec 06 §7.10 |
| `get_material_floor()` in §GOAL_BASELINE_CHECK ohne Restorability-Skalierung | `get_effective_material_floor(material_type, goal_name, restorability_score)` verwenden (§09.12); `restorability < 30` → `metadata["degraded_restorability"] = True`; `get_material_floor()` bleibt für PMGG + UI + Tests unverändert | §09.12 |
| `TransientEnergyMetric` fehlt nach subraktiver Phase (`transient_energie`-Goal) | `get_transient_energy_metric().measure_transient_energy(input, restored, sr)`; bei Score < `material_floor`: `_blend_onset_regions()` (Onset-selektiv, 5 ms Fenster); PHASE_GOAL_EXCLUSIONS für phase_18 + phase_26 ergänzen (§1.4.6) | §1.4.6 |
| `CausalDefectReasoner.reason()` ohne nachgelagerten `RecordingChainProfiler` (≥3 aktive Causes) | `RecordingChainProfiler().profile_chain(causes, material, era)` → `chain_hint` an GPOptimizer übergeben; ohne Profiler werden 8 Ursachen derselben physikalischen Kette als unabhängige Phasen-Cluster aktiviert → Over-Processing (§2.66) | §2.66 |
| Phase-Gruppe aus physikalisch gekoppelten Defekten einzeln per `perceptual_delta` gegated | Koalitions-Phasen via `_PHASE_COALITIONS` gruppieren; `perceptual_delta` erst nach der gesamten Koalition messen; §0a-verbotene Phasen dürfen nie Koalitions-Mitglied sein (§2.67) | §2.67 |
| `_profiled_phase_call_with_delta()` ohne `TemporalContinuityGuard`-Hook | `check_temporal_continuity(pre, post, phase_id, sr)` am Ende jeder Phase; `variance_ratio > 2.5` → WARNING; **`gain_step_db > 1.5`** (abrupter Gain-Sprung an Phase-Grenze → Mikro-Klick) → WARNING in `metadata["temporal_continuity"]`; kein Veto (§2.69) | §2.69 |
| GPOptimizer startet ohne Prior aus erfolgreichem Vorrun | `RestorationMemory().get_prior((era, material, defect_cluster_hash))` vor GPOptimizer-Aufruf; nach HPI > 0 + artifact_freedom ≥ 0.95: `save_result(...)` — kein Lernen aus schlechten Läufen (§2.70) | §2.70 |
| `compute_vqi()` auf historischem Material (era_decade < 1960) ohne `era_profile` | `get_era_vocal_profile(era_decade)` → `era_profile` an `compute_vqi()` übergeben; fixes ±2 dB ist falsch für historische Vokalstile → falsch-negative VQI-Scores + unnötige Recovery-Kaskaden (§EraVocalProfile) | §EraVocalProfile |
| `JITTER_ARTIFACTS` → `phase_12_wow_flutter_fix` in UV3 Tier-1 | D/A-Jitter erzeugt phasenmodulierte Intermodulationsprodukte (kein mechanisches Wow/Flutter) — `phase_12` (PSOLA) fügt Pitch-Artefakte hinzu statt zu helfen | `phase_14_phase_correction` + `phase_23_spectral_repair` (§4.11 Lumping-Invariante) | V27 |
| `NR_BREATHING_ARTIFACT` → `phase_03_denoise` oder `phase_29` als Primary | NR-Pumpen/Atmen entsteht durch übermäßige NR-Verarbeitung; weiteres NR auf diesem Artefakt verstärkt das Pumpen (bestätigt §4.11) | `phase_54_transparent_dynamics` (Envelope-Re-Smoothing, `gain_smooth_ms=200`) + `phase_08_transient_preservation`; KEIN `phase_03`/`phase_29` | V28 |
| `OVERLOAD_DISTORTION` → `phase_63_intermodulation_reduction` als Primary | Analoge Übersteuerung erzeugt Harmonische (H2/H3/H5); `phase_63` ist Volterra-IMD-Reduktion für Intermodulationsprodukte (f₁±f₂) — physikalisch falsche Algorithmus-Familie | `phase_09_crackle_removal` + `phase_23_spectral_repair`; KEIN `phase_63` für Harmonic Distortion | V29 |
| `ALIASING` → `phase_03_denoise` in Phasen-Selektion (UV3 oder Mapper) | Alias-Spiegelfrequenzen sind kohärente Signalspiegelungen (deterministisch); NR behandelt sie als Rauschen → entfernt Musikinhalt in der Alias-Zone, lässt Spiegelfrequenzen stehen | `phase_23_spectral_repair` + `phase_50_spectral_repair` für spektrale Chirurgie; KEIN `phase_03` | V30 |
| `ROOM_MODE_RESONANCE` → `phase_05_rumble_filter` als alleiniger Primary ohne Notch-EQ | Raumresonanzen (40–200 Hz) brauchen schmalbandige parametrische Notch-Filter (Q=12); `phase_05` ist Hochpass-Rolloff — `notch_q`/`notch_depth_db`-Config aus CausalReasoner erreicht `phase_05` nie | `phase_04_eq_correction` (Notch-EQ) als Primary; `phase_16_final_eq` als Sekundär; `phase_05` nur als Tertiär (Sub-Bass) | V31 |
| Strict-Conflict nutzt nur Event-Counts (ohne Reason/Priority/Vocal-Schwere) | Gleichbehandlung von P1/P2 und P4/P5 sowie vokalkritischen Fällen verwässert psychoakustische Prioritäten; führt zu falsch-kalibriertem ConflictScore/Decay | §2.58a: `reason_decay_weight` + `reason_cap_tighten_weight` + dynamische Schwere aus `goal_regressions` (GoalPriorityProtocol) + Vocal-Supremacy (`panns_singing`/`vocal_presence`) + Runtime-Severity-Telemetrie (`severity_score`, `severity_bucket`, `severity_fingerprint`) + Report-Felder (`regressive_weight_sum`, `runtime_severity_sum`, `runtime_severity_max`) | V34 |
| `pmgg_best_effort` wird bei hoher Reason-Diversität immer gleich hart bestraft (ohne Unsicherheitsdämpfung) | Inkonsistente Konfliktsignale (mehrere unterschiedliche weiche Gründe) werden überinterpretiert und provozieren Over-Damping; Folge: unnötige Strength-Reduktion trotz unsicherer Kausalität | §2.58a Disagreement-Guard: jüngste Family-Reason-Diversität (`>=3`) → `disagreement_brake_applied`; bei jüngstem Hard-Reason (`artifact_freedom_rollback`, `noise_texture_rollback`, `vocal_no_harm_rollback`, `hf_hallucination_rescue`) Guard zwingend aus | V35 |
| Phase iteriert über mehrere Defekt-Events desselben Typs (Schleife über `bump_locations`, `splice_points`, Dropout-Segmente …) mit **einheitlicher** `strength` für alle Events | Einheitliche Stärke ist falsch: leichte Events werden über-prozessiert; schwere Events in VFA-Schutzzonen (Vibrato, Frisson, Flüster, Passaggio) werden nicht beschränkt. Das menschliche Ohr registriert feinen Stärke-Mismatch zuverlässig. Bestätigt: `phase_12` (145 Kassetten-Bumps), `phase_64` (Mai 2026) | Per-Event-Strength-Oracle Pflicht: `_compute_<defect>_local_strength(mono_ref, start, end, sr, base_strength, protected_zones)` mit 250 ms Kontext-RMS-Proxy + VFA-Schutzzonen-Cap `[(start_s, end_s, max_cap)]` (Vibrato 0.20, Frisson 0.30, Flüster 0.25, Passaggio 0.35); `base_strength < 1e-6` → 0.0 | V38 |

> DSP-/Phase-spezifische VERBOTEN-Regeln (energy_bias, HNR-Guard, LPC-Ordnung, Passaggio, Timbral Coherence etc.): → [dsp.instructions.md](instructions/dsp.instructions.md) / [phases.instructions.md](instructions/phases.instructions.md)

### Sprachkonvention

- **UI-Texte, Fehlermeldungen**: Deutsch (Ursache + Lösungsvorschlag)
- **Docstrings**: Deutsch
- **Code-Kommentare, Log-Meldungen**: Deutsch

### Test-Infrastruktur
> Details: [tests.instructions.md](instructions/tests.instructions.md) — GC-Konventionen, Mock-Patterns, Budget-Tests, AMRB-Update, NaN-safe Correlation.


> **§0e** Regression-Fixes-Archiv (v9.11.2–v9.12.0, 23 Fixes): [`docs/CHANGELOG_HISTORY.md`](../docs/CHANGELOG_HISTORY.md). VERBOTEN-Tabelle (oben) schützt vor Reintroduktion.

## SOTA-Modell-Referenz (Mai 2026 — normativ für Plugin-Auswahl)

> Vollständige §4.4a-Evaluationsmatrix: `specs/04_dsp_sota.md`. Diese Tabelle ist die schnelle Referenz für Implementierungsentscheidungen.

| Aufgabe | Primäres Modell | Fallback | Anmerkung |
|---|---|---|---|
| **Gesangs-NR** (SNR < 10 dB) | SGMSE+ v2 | DeepFilterNet v3 → MIIPHER¹ | SGMSE+: bewährte Vokalqualität; ¹MIIPHER proprietär, nicht öffentlich verfügbar |
| **Musik-NR allgemein** | DeepFilterNet v3 + OMLSA | Apollo | DFN v3: beste Effizienz; OMLSA-Restglätter danach |
| **Musik-Separation** | BS-RoFormer (Wang et al. 2023) | HTDemucs v4 (Hybrid) | BS-RoFormer: beste SDR auf MusDB18; HTDemucs: schneller |
| **BW-Erweiterung 0–8 kHz** | AudioSR (SpeechFlow-Diffusion) | NVSR (DSP-basiert) | AudioSR: 16k→48k; NVSR: schneller, kein Halluzinationsrisiko |
| **BW-Erweiterung 8–16 kHz** | SBR-Heuristik + NVSR | Phase_07 Harmonik-Extrapolation | SBR: deterministisch; neuronale Methode erst nach Hallucination-Guard |
| **Musik-Restaurierung** | Apollo (Liu et al. 2024) | Aero (2024) | Apollo: Magnitude+Phase getrennt; ideal für recording-Ären |
| **Pitch-Estimation** | CREPE (Kim et al. 2018) | FCPE 2023 | FCPE: 5× schneller, akzeptable Genauigkeit |
| **Phasenrekonstruktion** | PGHI (Prusa et al. 2017) | Vocos (Siuzdak 2023) | PGHI für DSP-Magnitude; Vocos für ML-Magnitude |
| **Qualitäts-Messung** | VERSA (2024, multi-dim) | DNSMOS P.835 | VERSA für Restaurierung; DNSMOS für Sprache/Gesang |
| **Gesangs-MOS** | SingMOS (2023) | DNSMOS P.835 | SingMOS spezifisch für Gesangsmaterial |
| **MERT-Similarity** | MERT-95M (Li et al. 2023) | — | Nur Fallback wenn VERSA OOM; nie als Primärmetrik |

> **VERBOTEN**: Neues Plugin ohne Vergleich gegen Primärmodell in dieser Tabelle. Update erfordert §4.4a-Evaluationsprotokoll.

## Anti-Parallelwelten-Workflow (vor jeder Implementierung)

```
1. Suche in backend/core/, plugins/, dsp/ nach vorhandener Funktionalität
2. Prüfe existierende Plugins (specs/08) und Phasen (specs/06)
3. Falls vorhanden → einbinden, KEIN neues Modul
4. Falls nicht → Singleton-Pattern + DSP-Fallback
5. CHANGELOG.md dokumentieren
```

## Normative Priorisierung

`[RELEASE_MUST]` = Harte Release-Bedingung; CI-Gates nicht skippbar.
`[TARGET_2026]` = Roadmap 2026, kein Release-Blocker.

## Performance-Budget (Desktop, GPU-Mixed-Mode optional)

| Operation | Limit / Minute Audio |
|---|---|
| DefectScanner | ≤ 4 s |
| Phase-Pipeline gesamt | ≤ 240 s |
| FeedbackChain | ≤ 120 s |
| RestorabilityEstimator | ≤ 5 s |

- Verarbeitungs-SR: **48 000 Hz** (Phasen/Plugins)
- Analyse-Module: **native Import-SR** (kein `assert sr == 48000`)
- ONNX (heavy plugins): `providers=get_ort_providers("PluginName")` via `ml_device_manager`
- ONNX (light plugins): `providers=["CPUExecutionProvider"]`
- Torch (heavy plugins): `model.to(get_torch_device("PluginName"))` via `ml_device_manager`
- Torch (light plugins/DSP): `model.to("cpu")`; `torch.set_num_threads(os.cpu_count())`
- GPU-Fallback: Jedes Plugin MUSS bei GPU-Fehler transparent auf CPU fallen (§2.47)

### [RELEASE_MUST] AMD-GPU-Beschleunigung (v9.11.14)

Mixed-Mode: Heavy-Plugins → GPU, DSP/Light → CPU. Linux: ROCm 6.x; Windows: DirectML; Fallback: CPU. Tier-System (VRAM-abhängig): Tier 1 (≥16 GB, RDNA3): alle Plugins, fp16, 85 %; Tier 2 (8–15 GB): meiste, 80 %; Tier 3 (4–7 GB): selektiv (kein AudioSR), 70 %; Tier 4 (<4 GB): CPU-only.

**VERBOTEN**: `model.to("cuda")` / `CUDAExecutionProvider` direkt. **Richtig**: `get_torch_device("PluginName")` / `get_ort_providers("PluginName")` — aktiviert fp16 automatisch. Tier-adaptive Ratio via `_TIER_VRAM_PARAMS[gpu_tier]`.

## Kanonischer Pipeline-Ablauf (Kurzfassung)

`AurikDenker.denke()` → ReparaturDenker → RekonstruktionsDenker → RestaurierDenker → UV3:
DCOffset → TDP(HPSS) → RestorabilityEstimator → SongCalibration → Era/Genre/Medium-Classifier →
**VocalFocusAnalyzer** (PANNs-Singing ≥ 0.25 → VQI-Gate-Init + FormantTrack + FrissonDetect + RegisterDetect) →
GoalApplicabilityFilter → DefectScanner(54) → CausalDefectReasoner →
**RecordingChainProfiler** (§2.66: Causes → Ketten-Cluster → `chain_hint` für GPOptimizer; ≥3 aktive Causes; non-blocking) →
**RestorationMemory-Load** (§2.70: `(era, material, defect_cluster_hash)` → `X_init/Y_init` Prior für GPOptimizer) →
GPOptimizer →
**§GOAL_BASELINE_CHECK** (`_fast_goal_snapshot` auf Input → goal < `get_material_floor()` × 0.95 → `get_goal_recovery_phases()` → `selected_phases` ergänzt; ≤200ms DSP-Proxy; §0a-guard; §2.45 minimal-intervention; non-blocking) →
**§2.68 SSIP** (`detect_structural_silence_zones()` aus ORIGINAL-Audio → `_restoration_context["structural_silence_zones"]`; alle generativen Phasen erhalten Zones via kwargs; V14–V18 Linter-Guards) →
Phasen(01–64) [mit §2.48 Interaktions-Guard + **§0p Vocal-Invarianten** + **§2.67 Phase-Koalitionen** + **§2.69 TemporalContinuityGuard** als Post-Phase-Hook + **§2.78 AdaptivePhaseRescheduler** als Post-Phase-Hook (Closed-Loop: Goal-Gap → Recovery-Phasen-Injection; non-blocking; max 3/Session; §0a-Guard; MAS-Guard)] →
FeedbackChain → PhysicalCeiling → MusicalGoalsChecker → MDEM →
**HolisticPerceptualGate** (inkl. artifact_freedom §2.49 + **VQI-Gate §0p**) →
**RestorationMemory-Save** (§2.70: HPI > 0 AND artifact_freedom ≥ 0.95 → Priors persistieren) →
RestorationResult
> **Pipeline-Details** (§2.44–§2.78): [pipeline.instructions.md](instructions/pipeline.instructions.md)

### §2.44 HPI — Formeln
**Source-Traceability**: `[SRC:S06,S07,S08,S09,S10,S11]`
**Restoration** (Instrumental): `HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation`  
**Restoration** (Vokal — `panns_singing ≥ 0.35`): `HPI = MERT_similarity × timbral_fidelity × VQI × artifact_freedom × emotional_arc_preservation`  
**Studio 2026** (Instrumental): `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`  
**Studio 2026** (Vokal — `panns_singing ≥ 0.35`): `HPI = studio_quality_gain × PQS_improvement × VQI × artifact_freedom × emotional_arc_preservation`  
`artifact_freedom < 0.95` → Gate-Fail (primärer Veto-Faktor). `VQI < 0.72` bei `panns_singing ≥ 0.35` → Recovery-Kaskade (kein sofortiger Veto; §0p). VERSA primär (`use_versa_in_loop=True`); MERT nur Fallback; `MERT_floor = max(raw_mert, 0.5)`.

### §2.45/§2.45b Minimal-Intervention
`perceptual_delta > 0` Pflicht je Phase in Restoration — **Ausnahme: Phase-Koalitionen** (§2.67): zusammengehörige Phasen werden als Gruppe evaluiert, `perceptual_delta` gilt für die gesamte Koalition nach dem letzten Koalitions-Schritt. `restorability > 80 AND SNR > 40 dB` → Near-Passthrough (Strength ≤ 0.30, `metadata["high_restorability_gate"] = True`).

### §2.46 Carrier-Chain-Stufen (1→6)
Subtraktive Stufe 4 (NR) **vor** additiver Stufe 5 (Harmonik/BW-Erweiterung). → [phases.instructions.md](instructions/phases.instructions.md)

## 15 Musical Goals (Kurzreferenz)

| Prio | Restoration (Böden) | Studio 2026 (Böden) |
|---|---|---|
| **P0** ⚠️ Vokal | **VocalQuality ≥ 0.85**, **FormantFidelity ≥ 0.88** — nur wenn `panns_singing ≥ 0.35`; `VQI < material_vqi_floor` → Recovery-Kaskade (§0p) | **VocalQuality ≥ 0.90**, **FormantFidelity ≥ 0.92**; `VQI < material_vqi_floor` → Recovery-Kaskade |
| **P1** | Natürlichkeit ≥ 0.90, Authentizität ≥ 0.88 | Natürlichkeit ≥ 0.92, Authentizität ≥ 0.90 |
| **P2** | TonalCenter ≥ 0.95, Timbre ≥ 0.87, Artikulation ≥ 0.88 | TonalCenter ≥ 0.96, Timbre ≥ 0.89, Artikulation ≥ 0.90 |
| **P3** | Emotionalität ≥ 0.84, MikroDynamik ≥ 0.88, Groove ≥ 0.83, TransientEnergie ≥ 0.80 | Emotionalität ≥ 0.87, MikroDynamik ≥ 0.90, Groove ≥ 0.85, TransientEnergie ≥ 0.83 |
| **P4** | Transparenz ≥ 0.82, Wärme ≥ 0.77, BassKraft ≥ 0.78, SepFidelity ≥ 0.80 | Transparenz ≥ 0.85, Wärme ≥ 0.78, BassKraft ≥ 0.80, SepFidelity ≥ 0.83 |
| **P5** | Brillanz ≥ 0.78, Raumtiefe ≥ 0.70 | Brillanz ≥ 0.82, Raumtiefe ≥ 0.74 |

> Böden **immer** via `calibration_matrix.get_material_floor(material_type, goal)` — nie hardcodiert. Shellac ~0.72, Vinyl ~0.82, CD ~0.90. → [musical_goals.instructions.md](instructions/musical_goals.instructions.md)

## Vintage Aesthetics + Era-Verarbeitungsrichtlinien

**SOFT_SATURATION** = BEWAHREN. **CLIPPING** = REPARIEREN.

| Ära | Träger | Typische Defekte | Primäre Phasen | Verboten |
|---|---|---|---|---|
| 1900–1925 | Akustische Aufnahme (Trichter) | BW ≤ 3 kHz, hohes Oberflächenrauschen, kein Bass | phase_03, phase_06 (max 3 kHz!) | phase_07 (keine Harmonik-Ergänzung) |
| 1925–1945 | Elektrische Aufnahme + Shellac | BW ≤ 7 kHz, SNR ~15 dB, H2/H4-Sättigung; AGC-Schaltung → Amplitude-Drift | phase_03, phase_06 (max 7 kHz), phase_09, **phase_40** (wenn AMPLITUDE_DRIFT ≥ 0.30 UND Drift-Slope ≥ 1.5 dB/min) | Rolloff ≤ 7 kHz **nicht** erweitern; H2/H4 bewahren |
| 1945–1960 | Vinyl 78rpm/LP, Mono/frühes Stereo | RIAA-Entzerrung, Knistern, Wow/Flutter | phase_04, phase_09, phase_12 | Stereo-Enhancement (Mono-Quelle) |
| 1960–1975 | Vinyl LP, Analogband, frühes Stereo | Bandrauschen, Azimuth-Fehler, Wow/Flutter; Bandoxid-Drift (Temperatur/Feuchte) → Amplitude-Drift | phase_29, phase_25, phase_12, **phase_40** (wenn AMPLITUDE_DRIFT ≥ 0.30 UND Drift-Slope ≥ 1.5 dB/min) | Overdrive-NR (verwischt Recording-Ambience) |
| 1975–1990 | Vinyl, Cassette (Dolby B/C), Analogband | Dolby-Sättigungsrauschen, Dropouts, HF-Verlust; Dolby-AGC-Interaktion → Amplitude-Drift | phase_29, phase_24, phase_03, **phase_40** (wenn AMPLITUDE_DRIFT ≥ 0.30 UND Drift-Slope ≥ 1.5 dB/min) | Cassette: **BW-Ceiling 12 kHz** (Type I) — keine Frequenzsynthese darüber; Dolby nur invertieren wenn NR-Schaltung aktiv war; `_MATERIAL_PHASE_FACTORS["cassette"]` Pflicht in `defect_phase_mapper.py` |
| 1985–2000 | CD, DAT, Early Digital | Quantisierungsrauschen (16 bit), Jitter | phase_31, phase_30 | EQ-Erweiterung (über CD-BW = Artefakt) |
| 2000–2015 | MP3 (64–192 kbps), AAC | Pre-Echo, HF-Rolloff, psychoakust. Residue | phase_50, phase_23 | Aggressives NR auf MP3-Artefakte (Pre-Echo = kein Rauschen!) |
| 2015+ | FLAC, MP3 320, Streaming | Selten Artefakte; ggf. Loudness-War-Clipping | phase_01, phase_47 | Fast Passthrough; keine Carrier-Inversion nötig |

> Era-Erkennung: `EraClassifier.classify(audio, sr)` → era_decade, genre_label, carrier_chain

*Stand: Mai 2026 — Aurik 9.12.10 — instructions_version 9.6*
