# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: April 2026 — Version **9.11.14**
>
> **instructions_version: 7.9** — §0a DR/BW/Rauschtextur-Ceiling + §0d Carrier-Recovery-Referenzmodell + §1.2a/§4.7/§6.2b normiert 13.04.2026; §09.2 PMGG-Blend-Invariante + §2.54 Headroom-Scalar + MDEM Quiet-Zone normiert 23.04.2026; Wall-Time-Mismatch-Pegelexplosion + waerme-Proxy-Sättigung + MUSHRA-CCR-Referenz-Kontamination normiert 24.04.2026; §2.30b Per-Sample-Guard correct_arc() −36 dBFS + Musical-Goals-Kalibrierungsfehler (tonal_center KEY_SHIFT_PENALTY_DEFAULT / authentizitaet formant_threshold / phase_18 PMGG-CIG-Sync / adaptive_thresholds Material-Ceiling) normiert 25.04.2026; TonalCenter Bypass-Guard 0.70→0.60 (Denoising-Seiteneffekt) + GrooveMetric bidirektionaler DTW-Onset-Ratio-Guard normiert 26.04.2026; §2.30c WaveformPlausibilityGuard + §2.45a Music-Gated-LUFS-Delta Song-Boundary-Pegelexplosion normiert 28.04.2026; §Frisson Gänsehaut-Schutz-Pipeline (MDEM Zwei-Stufen-Invariante Pre+Post-SG) + §C10 Active Listener Calibration Bayesian EMA normiert 28.04.2026
>
> Aktuelle Testzahl: **~11598 `def test_`-Funktionen** (436 Testdateien; alle grün)
>
> **§2.36 `LyricsGuidedEnhancement` (Spec 03 §2.36)** ist ab Version **9.10.x Pflicht**.

## §0 Oberstes Prinzip — Klangwahrheit (vor allen technischen Regeln)

**Das Ziel jeder Restaurierung ist, dass der Hörer die Augen schließt und die originale Performance hört — nicht eine technisch korrekte Signalverarbeitung, und nicht eine „verbesserte“ Version.** Jede Entscheidung in Pipeline, Phase, Metrik und Export wird an diesem Maßstab gemessen.

**Drei Leitprinzipien** (hierarchisch, bei Konflikt gilt die höhere Stufe):

1. **Primum non nocere** — Füge dem Klang keinen Schaden zu. Lieber eine Beschädigung belassen als ein Artefakt einführen.
2. **Minimal-Intervention** — Greife nur ein, wo der Defekt hörbar ist. Je weniger Phasen aktiv, desto natürlicher das Ergebnis.
3. **Perceptuelle Verbesserung** — Der Export muss für einen Hörer näher am Original-Klang liegen als der degradierte Input. Technische Korrektheit ohne Klanggewinn ist wertlos.

### §0a Modus-Differenzierung

| Prinzip | Restoration | Studio 2026 |
|---|---|---|
| **Klangziel** | **Tonträgerkette invertieren** — den Klang wiederherstellen, der im Studio aus den Monitoren kam, bevor Trägermedium, Kopien und Alterung ihn degradiert haben | Bestmöglicher Studio-Klang — als wäre das Stück heute in einem Weltklasse-Studio aufgenommen |
| **Primum non nocere** | Unverändert — kein Artefakt, kein Klangschaden | Unverändert — auch Enhancement darf keine Artefakte erzeugen |
| **Intervention** | **Minimal** — nur Tonträgerverluste rückgängig machen, nichts hinzufügen | **Maximal-zielgerichtet** — volle Enhancement-Kette (Stem-Sep, Vocal-AI, Reference-Mastering, Stereo-Imaging), aber jede Phase muss messbaren Klanggewinn bringen |
| **Natürlichkeit** | Original-Charakter: Studio-Raumakustik, Ära-Klang, Recording-Chain-Signatur bewahren | Studio-Natürlichkeit: Klingt wie eine echte, professionelle Aufnahme — nicht synthetisch, nicht überbearbeitet |
| **Authentizität** | Zum Original — akustisch nicht unterscheidbar vom Studio-Master | Zum Künstler — musikalische Intention bewahren, Klangqualität modernisieren |
| **Rauschboden** | **Material-adaptiv** — Rauschboden-*Niveau* UND -*Textur* des originalen Aufnahmemediums anstreben, nicht aggressiver (Studio-Ambience bewahren). Spektrale Form des Restrauschens muss dem Trägerprofil entsprechen (kein weißes Rauschen nach Vinyl-Denoising) | **≤ −72 dBFS** — moderner Studio-Standard |
| **Qualitätsmaß** | Nähe zum Original (timbral fidelity) | Verbesserung gegenüber Input (PQS improvement) |
| **Dynamic Range** | **Material-Ceiling** — DR darf physikalisches Medium-Maximum nicht überschreiten (Spec 05 §6.2b DR_CEILING): Vinyl ≤ 70 dB, Shellac ≤ 45 dB, Tape ≤ 68 dB, CD ≤ 96 dB. Expansion über Ceiling = Artefakt. | DR-Erweiterung bis moderne Studio-Standards erlaubt, aber ≤ src_ceiling × 1.5 |
| **Bandbreite** | **Material-Ceiling** — Output-BW darf physikalisches Maximum des Quellmediums nicht überschreiten (Spec 05 §6.2c BW_CEILING): Shellac ≤ 8 kHz, Vinyl ≤ 16 kHz, WaxCyl ≤ 5 kHz. Additiv-Phasen müssen BW-Hard-Cap respektieren. | Volle BW-Erweiterung bis 22 kHz, erfordert aber MUSHRA ≥ 3.5 für Extension-Band |
| **Rauschtextur** | **Kohärent zum Trägerprofil** — spektrale Form des Restrauschens muss dem Trägermedium entsprechen (Spec 04 §4.7). Vinyl: rosa; Tape: Brown+HF-Hiss; CD: Weiß/Flat. Kohärenz-Score ≥ 0.80 Pflicht. | Minimaler Rauschboden; Textur-Kohärenz nicht erzwungen |
| **TFS (Temporal Fine Structure)** | **Strikt erhalten** — Hilbert-Phasen-Extraktion via `tfs_preservation_guard.py`; ERB-Band-Energie-Gate (nur Voiced-Frames > 3 aktiv); ΔPhase-Grenzwert material-adaptiv. Vintage-Ära-Charakter = Original-TFS. | **Flexibel** — TFS-Modifikation in Enhancement-Phasen erlaubt, wenn MUSHRA ≥ 3.5 im betreffenden ERB-Band; kein TFS-Rollback-Trigger in Studio 2026 |

> §0 ist **normativ übergeordnet**: Wenn eine technische Regel (PMGG-Threshold, Metrik-Schwellwert, Phase-Pflicht) dem Klangergebnis schadet, ist das ein Bug in der Regel — nicht im Klang.

### §0d [RELEASE_MUST] Carrier-Recovery-Referenzmodell (v9.11.14)

**Das fundamentale Messproblem**: Bei Carrier-Chain-Inversion verändert sich das Signal intentional stark gegenüber dem degradierten Input. Musical-Goals-Messungen, die den degradierten Input als Referenz verwenden, bestrafen korrekte Restaurierung als „Regression". Dieses Paradoxon muss auf **allen** Ebenen aufgelöst werden.

**Dreischichtiges Referenzmodell (normativ)**:

| Ebene | Referenzquelle | Anwendung |
|---|---|---|
| **1. Per-Phase PMGG** | §2.29c Baseline-Capping: `effective_before = min(measured, threshold + 0.05)` für restorative Phasen | Verhindert, dass verrauschter/defekter Input überhöhte Baseline liefert |
| **2. End-of-Pipeline Goals** | §1.2a Carrier-Recovery-Referenz: bei `carrier_chain_recovery_ratio > 0.15` wird die End-Referenz für timbral_fidelity/MFCC/Centroid auf das **best_carrier_checkpoint** (nach Carrier-Phasen, vor Enhancement) verschoben | Verhindert, dass starke Carrier-Inversion als Timbre-Regression gewertet wird |
| **3. HPI Export-Gate** | §2.44 Referenz-Paradoxon: restorability-abhängiger Referenz-Anker (GP-Memory MERT bei Restorability ≤ 50) | Finale Gesamt-Bewertung unabhängig vom degradierten Input |

**carrier_chain_recovery_ratio** (Pflichtfeld in `metadata`):
```python
recovery_ratio = 1.0 - spectral_correlation(pre_carrier_audio, post_carrier_audio)
```

**Invariante**: Kein Gate (PMGG, CIG, HPI, AFG) darf eine Phase oder den Export blockieren, weil das Signal sich vom **degradierten** Input entfernt hat — solange es sich dem **physikalischen Ceiling** des erkannten Materials nähert.

### §0c [RELEASE_MUST] Universalitäts-Invariante — alle Importsongs

Der Qualitätsanspruch gilt **universell für jede Importdatei** (Genre, Ära, Material, Länge, Defektprofil) und darf
niemals auf einen einzelnen Referenzsong optimiert werden.

**Verbindliche Invarianten:**

1. **Keine song-spezifischen Sonderregeln** im Produktionscode (z. B. Dateiname, Artist, statischer Song-Key,
  einmalige „Fixes" für ein einzelnes Beispiel).
2. **SOTA-übergreifende Robustheit**: Jede Optimierung muss in material-/kontextübergreifenden Gates
  nachweisen, dass sie auf einer repräsentativen Matrix (Material × Modus × Kontext) stabil ist.
3. **Maximal-umsetzbare Recovery statt Hardstop**: Bei fehlgeschlagenem End-Gate MUSS Aurik das bestmögliche sichere Ergebnis suchen (Strength-Reduktion, alternatives Checkpoint, Recovery-Kaskade). Früher Hardstop ohne Recovery-Suche ist unzulässig; Status: transparent (`degraded`/`recovered`).
4. **Allgemeingültigkeit vor Einzelfallgewinn**: Eine Änderung, die einen Song verbessert, aber die
  mittlere Qualität über die Import-Matrix verschlechtert, ist normativ unzulässig.

### §0f [RELEASE_MUST] KI-Agenten-Vorgehensweise: Systemisch vs. Punktuell

Jeder KI-Agent (GitHub Copilot, Claude, GPT), der an Aurik arbeitet, **muss zuerst erkennen**, ob ein Problem
punktuell (Einzelfall) oder systemisch (Muster über mehrere Stellen) ist — und dann die **korrekte Vorgehensweise** wählen.

**Entscheidungsbaum (vor jeder Änderung):** (1) Anzahl Stellen: 1→Punktuell; 2–4→prüfe Abstraktion; ≥5→IMMER Systemisch (zentrale Funktion + alle Callsites + multi_replace_string_in_file). (2) Wiedereinführbar?→Linter-Regel Pflicht. (3) Systemisch: Linter + VERBOTEN-Tabelle + Tests + Commit `fix §X systemic`.

**Verbindliche Invarianten für systemische Lösungen:**

1. **Keine halben Fixes**: Alle bekannten Callsites werden in **einem** Commit geschlossen — kein "fix die wichtigsten und den Rest später".
2. **Linter-First**: Jedes systemische Anti-Pattern MUSS eine Linter-Regel bekommen, bevor der Commit erfolgt. Sonst ist die nächste Regression garantiert.
3. **VERBOTEN-Tabelle-Pflicht**: Jede systemische Linter-Regel muss auch in der VERBOTEN-Tabelle (`copilot-instructions.md`) stehen, damit KI-Agenten sie in neuen Sessions kennen.
4. **Kein Over-Scope**: Systemische Lösung bedeutet "alle Stellen desselben Musters", nicht "refactore das gesamte Subsystem". Nur das betroffene Muster wird geschlossen.

**Erkennungsmerkmale systemischer Probleme (Checkliste):**

| Signal | Bedeutung |
|---|---|
| Selbe Konstante (`-36.0`, `-50.0`, `gate_dbfs`) an ≥ 5 Stellen | Systemisch — zentrale Default-Änderung nötig |
| Selbes Import-/Call-Muster in mehreren Phasendateien | Systemisch — Helper-Funktion + alle Callsites |
| Bug tritt in "2 Sessions re-introduced" auf | Systemisch — Linter fehlt |


## Vollständige Spezifikation (normative Referenz)

**9 Specs** in `.github/specs/`:

| Spec | Inhalt | Konsolidierte Praxis-Abschnitte |
|---|---|---|
| **01** Goals/PMGG | 14 Musical Goals, Schwellwerte, GoalApplicability | §1.3 Metric Recalibration & Debugging |
| **02** Pipeline/§2.x | UV3, Denker, SongCal, KMV, FeedbackChain | §2.2c Denker-Orchestrierung & Hänger-Patterns |
| **03** Module/§3.x | Kognitive Module §2.1–§2.43 | — |
| **04** DSP/SOTA | Algorithmen, SOTA-Matrix, Psychoakustik | §4.6a ML-Plugin-Workflow & ONNX-Chunking, §4.6b Material×Defect-Entscheidungsbaum |
| **05** Material/Defekte | 15 Materialtypen, 46 DefectTypes | — |
| **06** Phasen 01–64 | Phase-Liste, CAUSE_TO_PHASES | §7.3a Phase-Implementierung, Caching & Checkliste |
| **07** Tests/Qualität | PQS, AMRB, OQS, MUSHRA | §8.4a Test-Patterns, 6 Pitfalls & CI-Gates |
| **08** Architektur/Distribution | Layers, Plugins, CLI, AppImage | §11.4c UI-State-Machines & Thread-Safety, §11.5a Mermaid-Visualisierung |
| **09** Kalibrierungsmatrix | CANONICAL_THRESHOLDS (Restoration + Studio 2026), Material-/Ära-/Genre-Bias, SongGoalTargets-API | §09.2 Zwei-Ebenen-API (Pipeline vs. Convenience) — **normativ übergeordnet für alle Schwellwerte** |

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

> Vollständige Tabelle (~100 Einträge, Linter-Referenz V01–V11): [`.github/VERBOTEN.md`](.github/VERBOTEN.md)

| Verboten | Richtig | Linter |
|---|---|---|
| `gate_dbfs=-36.0` ohne `reference_for_gate` | `compute_signal_relative_gate_dbfs(ref, material_key=...)` via `reference_for_gate=pre_phase_audio` | V04 |
| `sosfilt(sos, audio)` addiert zu Original | `sosfiltfilt` (zero-phase) überall wo Bandfilter auf Signal addiert | V11 |
| `np.max(np.abs(audio))` als Peak-Guard | `np.percentile(np.abs(audio), 99.9)` | V08 |
| `print(...)` | `logger.info(...)` | V01 |
| `sf.read(path)` / `librosa.load(path)` | `load_audio_file(filepath)` | V02 |
| Neue HPF/Notch-Phase ohne 4-stufige Checkliste | (1) kein Guard in Phase-Datei; (2) `enable_loudness=False` in `_phase_overrides`; (3) `_HPF_NOTCH_CUM_RESET_PHASES`; (4) `_update_positive_makeup_authority` | — |
| Carrier-Repair-Rollback inkrementiert `consecutive_rollbacks` | `_CARRIER_REPAIR_PHASE_PREFIXES`-Check vor Increment | V09 |
| `map_location="cuda"` ohne ml_device_manager | `get_torch_device("PluginName")` via ml_device_manager | V03 |
| `griffinlim()` als Endschritt | PGHI / Vocos | V05 |
| `CAUSE_TO_PHASES` ohne `CAUSES`-Gegenstück | Bidirektionale Sync: `CAUSES` + `CAUSE_TO_PHASES` (§2.59) | — |



### Sprachkonvention
- **UI-Texte, Fehlermeldungen**: Deutsch (Ursache + Lösungsvorschlag)
- **Code-Kommentare, Docstrings, Log-Meldungen**: Englisch

### Test-Infrastruktur
- `pytest`/`conftest`-Teardowns für große Suiten bevorzugen **leichten inkrementellen GC** (`gc.collect(0)` o. ä.); volles `gc.collect()` nur cadence-gesteuert oder an Datei-/Session-Grenzen.
- Lang lebige Hintergrund-Manager (z. B. Monitor-Threads) brauchen einen **idempotenten `shutdown()`-Kontrakt** mit `Event.set()` + `join(timeout=...)`; Test-Sessions räumen diese Manager in `pytest_sessionfinish` oder Finalizern ab.
- **VERBOTEN**: unbedingtes Full-GC nach jedem Test in großen Suiten und das Verlassen auf `daemon=True` als einziges Cleanup-Modell.
- Unit-Tests, die Budget-Logik (`try_allocate`, `release`) prüfen, **müssen** `is_system_thrashing` auf `False` mocken — andernfalls versagen Tests auf Hosts mit hoher Swap-Auslastung (environment-dependent flakiness).
- Resampling-Bibliotheken (`resampy`, `librosa`) können `pkg_resources`-Warnings auslösen, die unter `-W error::Warning` Testabbrüche verursachen. Immer auf aktueller Version (`resampy >= 0.4.3`) halten.
- Teuere Analytic-Transforms (Hilbert, `filtfilt`, STFT) **IMMER nach** dem günstigsten Admissibility-Gate platzieren. Muster: Frame-Energie-Check → Voiced-Gate → dann `filtfilt` + Hilbert (vgl. TFS-Guard).
- ML-Budget-Tests und Phase-Tests, die `np.corrcoef` auf near-constant Signalen aufrufen, brauchen guarded correlation (`dot(a,b)/(||a||·||b||+ε)`) — kein Warning, NaN-safe.

> **§0e** Regression-Fixes-Archiv (v9.11.2–v9.12.0, 23 Fixes): [`docs/CHANGELOG_HISTORY.md`](../docs/CHANGELOG_HISTORY.md). VERBOTEN-Tabelle (oben) schützt vor Reintroduktion.

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
GoalApplicabilityFilter → DefectScanner(46) → CausalDefectReasoner → GPOptimizer →
Phasen(01–64) [mit §2.48 Interaktions-Guard] → FeedbackChain → PhysicalCeiling → MusicalGoalsChecker → MDEM →
**HolisticPerceptualGate** (inkl. artifact_freedom §2.49) → RestorationResult

### [RELEASE_MUST] §2.44 Holistic Perceptual Gate (v9.10.123)
Letztes Gate vor Export. Misst **Gesamt-Hörverbesserung** statt nur Einzel-Goals.

**Restoration**: `HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation`
**Studio 2026**: `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`

- `timbral_fidelity`: Strukturelle Klangkohärenz (nicht Input-Ähnlichkeit); Restorability-abhängiger Referenz-Anker
- `artifact_freedom`: Veto-Faktor (§2.49) — < 0.95 → Gate-Fail (Primum non nocere)
- `emotional_arc_preservation`: Arousal/Valence + Makrodynamik + Lyrics-Salienz (§2.36)
- **HPI > 0** → Export | **HPI ≤ 0** → Rollback

> Details (Referenz-Paradoxon, MERT-Aufbau, Gewichtungs-Semantik, Wertebereiche): Spec 02 §2.44 + Skill `fix-metric`

### [RELEASE_MUST] §2.45 Minimal-Intervention-Prinzip (v9.10.122)

**Restoration**: `perceptual_delta > 0` Pflicht für jede Phase; ≤ 0 → Skip. So wenige Phasen wie nötig.
**Studio 2026**: Volle Enhancement-Kette, aber `perceptual_delta > 0` bleibt Pflicht — kein Over-Processing.

### [RELEASE_MUST] §2.45a Mid-Pipeline-Loudness-Drift-Guard

Breitbandig-subtraktive Phasen (Denoise/Noise-Gate/Dereverb): Gated-RMS-Guard → envelope-aware Makeup-Gain (nur Musik-Frames, `gate_dbfs=-36.0` + `reference_for_gate=pre_phase_audio`, V04) → Soft-Limiter NUR wenn `peak > 0.98`. HPF/LPF/Notch/Bandpass: **kein** per-Phase-Guard (4-stufige Checkliste; vgl. VERBOTEN-Tabelle). Finale Fangschicht §2.30c: `apply_waveform_plausibility_guard(original, restored, sr, mode, material_type, restorability_score)` nach MDEM in UV3 — NIE Boost, non-blocking.

> Invarianten §2.45a-I bis -VII: Spec 02 §2.45a + Spec 04 §4.6

## 14 Musical Goals (Kurzreferenz)

| Prio | Restoration (Böden) | Studio 2026 (Böden) |
|---|---|---|
| **P1** | Natürlichkeit ≥ 0.90, Authentizität ≥ 0.88 | Natürlichkeit ≥ 0.92, Authentizität ≥ 0.90 |
| **P2** | TonalCenter ≥ 0.95, Timbre ≥ 0.87, Artikulation ≥ 0.85 | TonalCenter ≥ 0.96, Timbre ≥ 0.89, Artikulation ≥ 0.87 |
| **P3** | Emotionalität ≥ 0.82, MikroDynamik ≥ 0.88, Groove ≥ 0.83 | Emotionalität ≥ 0.84, MikroDynamik ≥ 0.90, Groove ≥ 0.85 |
| **P4** | Transparenz ≥ 0.82, Wärme ≥ 0.75, BassKraft ≥ 0.78, SepFidelity ≥ 0.78 | Transparenz ≥ 0.85, Wärme ≥ 0.78, BassKraft ≥ 0.80, SepFidelity ≥ 0.80 |
| **P5** | Brillanz ≥ 0.78, Raumtiefe ≥ 0.70 | Brillanz ≥ 0.82, Raumtiefe ≥ 0.74 |

> Alle Werte = **kanonische Böden** (Spec 09 / `calibration_matrix.py`). Song-spezifische Ziele berechnet die adaptive Schicht §2.31 + §09.2 + §2.56 aus Material, Ära, Genre und Restorability.

**Regressions-Regime** (differenziert — §2.29d, aktualisiert §2.54):
- **P1/P2** (Natürlichkeit, Authentizität, Tonal, Timbre, Artikulation): **Pipeline-Ende-Pflicht** — am Ende der gesamten Kette müssen alle P1/P2-Goals ≥ Schwellwert liegen. Einzelphasen dürfen vorübergehend P1/P2-Proxy-Werte senken, wenn Carrier-Repair (§2.44 Referenz-Paradoxon) oder restorative Defektentfernung (§2.29c Baseline-Capping) der Grund ist. Der CumulativeInteractionGuard (§2.48) ist die materialadaptive **Notbremse** (§2.54), nicht die Routine-Steuerung.
- **P3–P5**: **Pipeline-Netto-Budget** — Einzelphasen dürfen vorübergehend verschlechtern, wenn am Ende der Kette alle Goals ≥ Schwellwert. PMGG loggt Zwischenregressionen, blockiert aber nicht.

Details: Skill `fix-metric`

### [RELEASE_MUST] §2.56 / §2.56a / §C10 / §Frisson — Per-Song-Gewichtung + Frisson-Schutz

`estimate_goal_importance()` → 5-stufiges individuelles Gewichtungsprofil (Label/Audio/Psychoakustik/Vokal-Harmonik/Interactions, [0.30, 2.00]). P1/P2-Floor ≥ 0.70. §2.56a: `_compute_harmonic_adaptation_scalar()` advisory-only in UV3 `_profiled_phase_call` ([0.72, 1.18]), explizite PMGG-Strength hat Vorrang. §C10: Bayesian-EMA-Blend 15 % aus `SongGoalFeedbackStore.get_nudges()` nach Stufe 7. §Frisson: `get_frisson_detector().detect(original, sr)` → `frisson_zones` VOR MDEM-Aufruf (Non-blocking: Exception → `[]`); **Zwei-Stufen-Invariante**: Pre-SG + Post-SG Frisson-Floor −1.0 LU (SG verteilt sonst Dämpfung zurück). MDEM würde Klimax-Passagen sonst bis −8 LU dämpfen.

> Details: Spec 02 §2.56, §2.56a, §C10, §Frisson; `backend/core/song_goal_importance.py`, `frisson_candidate_detector.py`

### [RELEASE_MUST] §2.46 Carrier-Chain-Inversion (v9.10.122)

**Restoration**: **Gesamte Tonträgerkette invertieren** (invers: ADC → Playback → Alterung → Carrier-Encoding); Mixer/Preamp + Studio-Raumklang **bewahren**.
**Studio 2026**: Carrier-Chain-Inversion + Enhancement-Kette; Mixer-Charakter darf modernisiert werden.

**6 Inversions-Stufen (normative Reihenfolge für Phasen-Sortierung):**
1. **ADC-Artefakte** entfernen: DC-Offset (phase_30), Quantisierungsrauschen (phase_31)
2. **Playback-Verzerrungen** invertieren: RIAA-Inverse (phase_04/phase_06), Azimuth-Korrektur (phase_25), Wow/Flutter (phase_12)
3. **Alterungsschäden** reparieren: Knistern/Crackle (phase_09), Dropout (phase_24), Oxidation
4. **Carrier-Encoding (subtraktiv)**: Bandrauschen (phase_29), Surface Noise (phase_03), Shellac-Rauschen
5. **Carrier-Encoding (additiv)**: Bandbreiten-Erweiterung (phase_06/phase_23), Harmonik (phase_07) — **IMMER nach Stufe 4!**
6. **Mixer/Preamp + Studio-Raumklang**: BEWAHREN (Recording-Chain-Signatur = Original)

**Invariante**: Subtraktive Phasen (Stufe 4) VOR additiven (Stufe 5) — sonst werden rekonstruierte Obertöne sofort entrauscht.

### [RELEASE_MUST] §2.46a / §2.46b — Transfer-Chain-Vollständigkeit + Dateicontainer-Invariante

`transfer_chain` modelliert alle Stufen vollständig (§2.46a — keine Verkürzung auf Primär+1). `file_ext` bestimmt **nur** die letzte Kettenstufe (§2.46b): Fallback-Gate `rotation_strength ≥ 0.30 AND conf ≥ 0.20 → vinyl akzeptiert` trotz `.mp3`. Studio-Tape: `wow < 0.06 WRMS → reel_tape; ≥ 0.06 WRMS → cassette` (IEC 60386:1987). **Produktions-Invariante**: `["vinyl","reel_tape","mp3_low"]` bei `rotation=0.371, file_ext=.mp3`.

> Details: Spec 05 §6.7; Test: `test_vinyl_tape_mp3_chain_detection.py::test_vinyl_reel_tape_mp3_full_chain_production_case`

### [RELEASE_MUST] §2.46b Dateicontainer-Invariante — file_ext ≠ Quellanalyse (v9.11.14)

`file_ext` unterdrückt nie physikalische Fingerabdruck-Evidenz. Fallback-Gate Vinyl: `rotation_strength ≥ 0.30 AND conf ≥ 0.20` auch bei `.mp3`. Studio-Tape: `_thresh_rt = max(0.010, 0.025*(1−0.55*codec_contamination))`. Cassette vs. Reel: `wow < 0.06 WRMS → reel_tape`. VERBOTEN: bei `file_ext=.mp3 AND rotation=0.371` Einzelergebnis `mp3_low` — RELEASE_MUST-Verstoß.

> Spec 05 §6.7 Phase 1b; Test: `test_vinyl_tape_mp3_chain_detection.py`

### [RELEASE_MUST] §2.47 Adaptive-Intelligence-Prinzip (9-Schritt-Kaskade)

`MediumDetector.detect()` → `EraClassifier` (+ERB-Salience) → `GenreClassifier` → `RestorabilityEstimator` → `DefectScanner` (46) → `CausalDefectReasoner` (49) → `SongCalibration` → `SongGoalImportance` → `GPOptimizer`. SGMSE+ Tier-0 Vokal (phase_03). Carrier-Formant-Decay-Inversion phase_42 Stage 0.5. ML-OOM-Fallback Pflicht (`metadata["ml_fallbacks_used"]`). Edge: < 10 s → Groove/MicroDyn off; Restorability < 20 + Shellac → Scale 0.65.

> Details + ML-Fallback-Tabelle: Spec 02 §2.47 + Spec 05

### [RELEASE_MUST] §2.52–§2.53b PhaseConductor + Experience-Loop + Denker-Determinismus

**§2.52**: `PhaseConductor` (`get_phase_conductor()`) misst 4D-State nach jeder Phase (noise_floor_db, hf_energy_ratio, transient_density, harmonic_coherence) → advisory `strength`-Hint; PMGG-Strength hat Vorrang. `_NEVER_SKIP`: phase_01/09/12/14/15. **§2.53**: `joy_runtime_index` + `auto_improvement_recommendations` in `metadata`; `bridge.get_experience_insights()` → Frontend (non-blocking); `frisson_index` advisory. **§2.53a**: ExzellenzDenker — `messe_und_repariere()` primär, `messe_ziele()` Legacy. **§2.53b**: `precomputed_phase_plan` = Source of Truth — UV3 überspringt `_select_phases()` + `_optimize_phase_plan_intelligence()`.

> Spec 02 §2.52, §2.53, §2.53a, §2.53b

### §0b Konfliktauflösung / Anti-Widerspruch (v6.9)

Kollisions-Hierarchie: (1) §0 + RELEASE_MUST-Invarianten; (2) neuere versionsmarkierte Abschnitte (höhere `v9.x`); (3) spezifische Feld-/Kontrakt-Regeln vor generischen Stilregeln.

### [RELEASE_MUST] §2.47a Frontend-Backend-PreAnalysis-Handover

`run_pre_analysis()` GENAU 1× nach Import; `PreAnalysisResult` direkt (kein Cache-Rebuild) in Queue-Item Settings → `BatchProcessingThread` → `AurikDenker.denke()`. `MediumDetector.detect()` GENAU 1×. Neuer File-Import → Cache HARD gelöscht (verhindert Race Conditions in async Threads).

> Spec 02 §2.37; Test: `test_pre_analysis_handover_no_double_detect.py`

### [RELEASE_MUST] §6.2a Material-Pflicht-Phasen

`_MATERIAL_PRIORITY_PHASES` (UV3): vinyl→phase_09/12/05; tape/reel_tape→phase_29/24/03; shellac→phase_03/06/01; cassette→phase_29/12/06/24/03; mp3_low→phase_23/03/50 — aktiviert **unabhängig** vom DefectScanner-Severity-Score (DefectScanner statistisch; einzelne schwere Defekte können unter Schwelle liegen). Alle 15 Materialtypen in UV3 definiert.

### [RELEASE_MUST] §2.29c PMGG Restorative-Baseline-Capping

`_RESTORATIVE_PHASES` (phase_02/03/09/18/20/23/24/29/49): `effective_before[g] = min(measured, canonical_threshold[g] + 0.05)`. Enhancement-Phasen nutzen echte `scores_before`. Verhindert, dass defektbehafteter Input falsche Regression auslöst.

> Details (CANONICAL_THRESHOLDS): Spec 02 §2.29c

### [RELEASE_MUST] §2.29e PMGG Team-Koordination (Vorphasen-Kontext)

UV3 schreibt `prior_phase_context` nach jeder Phase fort. PMGG ruft `_resolve_team_context_policy()` — verhindert, dass Folgephasen intentionale Vorphasen-Reparaturen rükgängig machen. `CONFLICT_REGISTRY` in `phase_ontology.py` (`get_conflict_phases()`). Phase_50 nach HF-Restaurationskette (phase_06/07/23): Goal-Excl. brillanz/transparenz/timbre + Emergency-Retries unterdrückt. UV3 baut `metadata["team_coordination"]`.

> Details: Spec 02 §2.29e; Spec 06 §6.9b

### [RELEASE_MUST] §2.55 PMGG-CIG-Synchronisations-Invariante

`CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2 ↔ PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2` bidirektional synchron. Neue Phase = **beide Tabellen** aktualisieren. CI-Test: `test_pmgg_cig_sync.py`.

> `backend/core/per_phase_musical_goals_gate.py` + `cumulative_interaction_guard.py`

---

### [RELEASE_MUST] §2.48 Kumulative-Phasen-Interaktions-Guard (Notbremse)

`compute_adaptive_drift_tolerance(restorability, material, severity, n_phases)` — NICHT als Konstante. Carrier-Repair-Phasen (`_CARRIER_REPAIR_PHASE_PREFIXES`) inkrementieren `consecutive_rollbacks` NICHT. Pipeline-Stopp: `max(5, n_carrier_phases + 2)`. STFT-Kohärenz: nach ≥ 3 STFT-Phasen ≤ 5 ms Deviation.

> Details: Spec 02 §2.48

### [RELEASE_MUST] §2.49 Artefakt-Freiheits-Gate / §2.49b Stereo-Collapse / §2.51 Stereo-Kohärenz

`artifact_freedom = min(per-phase-scores)` (kein Pipeline-in/out-Delta). Musical-Noise: nur Bins wo `restored > orig × 1.05`. Phase-Cancellation: Delta-basiert + `original_stereo` Referenz; Frames, die im Input bereits anti-phasig waren, nicht flaggen. §2.49b: Post-Pipeline L/R-Imbalance > 20 dB + Input < 6 dB → Rollback-Kaskade. §2.51: Jede Phase mit Stereo → M/S-Domain oder Linked-Stereo (VERBOTEN: unabhängiges L/R-Processing). §2.51a Hard-Fail: Interchannel-Delay > 1 ms, L/R-Imbalance > 6 dB, True-Peak > −1 dBTP.

> Details: Spec 02 §2.49, §2.49b; Spec 08 §2.51, §2.51a

### [RELEASE_MUST] §2.50 Material-Adaptive Gate Baseline

`measure_source_baseline(audio, sr, material_type)` VOR Pipeline: `phase_cancellation_ratio`, `stereo_mono_compat_mean`, `has_critical_stereo_issue`, `has_anti_phase_region`, `hf_loss_db`. Notfall-Injection: `has_critical_stereo_issue=True` → phase_14 + phase_15 als Pflicht-Phasen; `has_anti_phase_region=True` → phase_14. Gate darf keine Eigenschaft bestrafen, die im Input bereits in gleicher Ausprägung vorhanden war (§0 Primum non nocere).

> Implementierung: `backend/core/artifact_freedom_gate.py`

### [RELEASE_MUST] §2.54 Adaptives Phasen-Optimum (Messen-Handeln-Validieren)

Feste Schwellwerte sind **Notbremsen**, nicht die Routine-Steuerung. Jede Phase: Messen→Handeln→Validieren-Zyklus. Guards nutzen `compute_adaptive_drift_tolerance()` statt Konstanten. Checkpoint = perceptuell bestes Ergebnis über alle Iterationen. Carrier-Repair-Phasen dürfen Signal intentional vom degradierten Input entfernen (§0d Referenz-Paradoxon).

> Details + ANTI-PATTERN-Tabelle: Spec 02 §2.54

## Vintage Aesthetics

**SOFT_SATURATION** = BEWAHREN. **CLIPPING** = REPARIEREN.
1920–1940: Rolloff ≤ 7 kHz nicht erweitern, H2/H4 bewahren.

*Diese Richtlinien gelten für alle KI-Agenten (GitHub Copilot, Claude, GPT) die an Aurik 9 arbeiten.*
*Vollständige normative Spezifikation: `.github/specs/01–09`.*
*Stand: April 2026 — Aurik 9.11.16 — instructions_version 8.0*
