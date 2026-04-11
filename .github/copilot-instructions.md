# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: April 2026 — Version **9.11.0**
>
> **instructions_version: 7.1** — §2.55 PMGG-CIG-Synchronisations-Invariante normiert 10.04.2026
>
> Aktuelle Testzahl: **~11084 `def test_`-Funktionen** (424 Testdateien; alle grün)
>
> **§2.36 `LyricsGuidedEnhancement`** ist ab Version **9.10.x Pflicht**.

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

> §0 ist **normativ übergeordnet**: Wenn eine technische Regel (PMGG-Threshold, Metrik-Schwellwert, Phase-Pflicht) dem Klangergebnis schadet, ist das ein Bug in der Regel — nicht im Klang.

## Architektur dieser Richtlinien

Diese Datei ist der **Slim Core** (~250 Zeilen) — wird in **jeder** Konversation geladen.
Detailwissen liegt in **aufgabenspezifischen Skills** unter `.github/skills/*/SKILL.md`, die nur bei Bedarf geladen werden.

| Aufgabe | Skill | Trigger-Phrases |
|---|---|---|
| Neue Phase implementieren | `new-phase` | phase, PMGG, Exclusions, PhaseResult |
| Musical-Goal-Metrik ändern | `fix-metric` | Metrik, Goal, Schwellwert, Kalibrierung, Divisor |
| DSP-Algorithmus wählen/bauen | `aurik-dsp-decision` | SOTA, Modell, Fallback, OMLSA, PGHI, STFT |
| ML-Plugin integrieren | `ml-plugin` | Plugin, ONNX, try_allocate, Headroom, Fallback |
| Pipeline debuggen/verstehen | `pipeline-debug` | UV3, Denker, SongCal, KMV, FeedbackChain |
| UI/Frontend-Feature | `ui-feature` | Qt, Signal, Widget, Progress, Thread-Safety |
| Tests schreiben | `test-writing` | Test, pytest, Marker, Heavy, conftest |
| Qualität/Benchmark messen | `quality-benchmark` | OQS, AMRB, PQS, MOS, MUSHRA |
| Architektur visualisieren | `aurik-architecture-diagram` | Diagramm, Mermaid, Flowchart, Übersicht |

## Vollständige Spezifikation (normative Referenz)

**8 Specs** in `.github/specs/`: **01** Goals/PMGG, **02** Pipeline/§2.x, **03** Module/§3.x, **04** DSP/SOTA, **05** Material/Defekte, **06** Phasen 01–64, **07** Tests/Qualität, **08** Architektur/Distribution — Änderungshistorie: `docs/CHANGELOG_HISTORY.md`

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
- Export nur nach bestandenem Qualitäts-Gate (Musical Goals + PQS + Safety-Invarianten)

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

### VERBOTEN (Kurzliste — vollständig in Skills)

| Kategorie | Verboten | Richtig |
|---|---|---|
| Logging | `print(...)` | `logger.info(...)` |
| API-Return | `return dict` | `@dataclass` |
| Cache | `_cache = {}` | `threading.Lock()` + Dict |
| Phase-Rekonstruktion | `griffinlim()` als Endschritt | PGHI / Vocos |
| Normalisierung | RMS / Peak | LUFS ITU-R BS.1770-5 |
| Audio-Import | `sf.read(path)` / `librosa.load(path)` | `load_audio_file(filepath)` |
| Backend-Import | `from Aurik910... import` in `backend/` | Architektur-Trennung |
| GPU | `map_location="cuda"` ohne ml_device_manager | `get_torch_device("PluginName")` via ml_device_manager |
| Musikmetriken | `pesq()`, `dnsmos()`, `nisqa()` | PQS-MOS, VERSA, SingMOS |
| Wiener-Filter | `scipy.signal.wiener()` primär | OMLSA / DeepFilterNet |
| LPC | Ordnung < 16 | Ord. 30–40 @ 48 kHz |
| ML-Budget | `plm.try_allocate()` | `ml_memory_budget.try_allocate()` |
| Tonträgerkette | `MediumClassifier.classify_medium()` | `MediumDetector.detect(audio, sr, file_ext=...)` (§6.7) |
| DC-Offset reel_tape | `np.mean`-Subtraktion / `lfilter` | `scipy.signal.filtfilt([1,-1],[1,-0.9995])` zero-phase |
| SongCal-Bounds | `np.clip(scalar, 0.0, 2.0)` | `global_scalar∈[0.50,1.50]`, `family_scalar∈[0.30,1.80]` |
| MDX23C-Fallback | `HPSS` direkt | NMF-β-Separation (sdB ≥ 5) → HPSS als tertiärer Fallback |
| Pflicht-Phasen | DefectScanner allein entscheidet | Material-Pflicht-Phasen (§6.2a) immer aktivieren |
| Peak-Guard (Gain) | `np.max(np.abs(audio))` | `np.percentile(np.abs(audio), 99.9)` — Impuls-Artefakt darf Normalisierung nicht blockieren |
| Dolby NR Inversion | Statische globale HF-Absenkung ohne Typ-Erkennung | `DolbyNRDetector.detect()` → `phase_04(dolby_nr_type=..., dolby_nr_confidence=...)` (§6.7 Phase 1c) |
| Head-Bump Tape | Kein LF-Kerbfilter bei Tape-Material | `phase_04(tape_speed_ips=X)` → HEAD_BUMP_PROFILES[nearest_speed] parametrischer Dip |
| Inpainting HF-Halluzination | AR/Diffusion ohne BW-Begrenzung | `_MATERIAL_BW_CAP_HZ` in phase_55 — wax_cylinder ≤ 5kHz, wire_recording ≤ 6kHz (§0) |
| Phase_63 Stereo IMD | Unabhängiges L/R-IMD-Notch | M/S-Domain: Notch-Maske aus Mid berechnen, symmetrisch auf Mid+Side anwenden (§2.51) |
| Phase-Wetness ohne Feedback | Feste `strength` ohne Mess-Feedback zwischen Phasen | `PhaseConductor.recommend()` (§2.52) — 4D-State-Vektor → adaptiver Strength-Hint |
| Feste Guard-Schwellwerte | `MAX_DRIFT = -0.05` / `regression > 0.02` als Konstanten | `compute_adaptive_drift_tolerance()` aus Material/Restorability/Defects (§2.54) |
| PhaseSkipper rohe Severity | `defect_score.severity` direkt ohne Salience-Gewichtung | `_salience_adjusted_severity()` (§2.47) — ERB-maskierte Severity; fully-masked (n_masked≥3, n_salient=0) → -50 % |
| Carrier-Formant-Inversion | Phase 42 ohne Material-Kontext in `_enhance_channel` | `_restore_carrier_formant_decay(audio, sr, material_type)` Stage 0.5 (§2.52, Hebel 4) |
| Loudness-Guard RMS | `np.mean(audio**2)` (globaler RMS in Guards) | `_rms_dbfs_gated()` — Frame-basiert, nur Frames > −50 dBFS, Stille ignoriert (§2.45a-I) |
| Loudness-Guard Gain | `audio *= gain_factor` (uniformer Gain) | `_musical_gain_envelope()` — Gain nur auf Musik-Frames, Stille unverändert (§2.45a-II) |
| Loudness-Guard Limiter | Unbedingter Soft-Limiter nach Makeup-Gain | Soft-Limiter NUR wenn `peak > 0.98` — keine Routine-Dynamik-Kompression (§2.45a-III) |
| FeedbackChain feste Schwellen | `_prune_threshold = -0.05` / `if history[-1] < history[-2] - 0.05` als Konstanten | `_compute_adaptive_prune_threshold(is_restorative, material, rest, severity)` — z. B. shellac 3×, vinyl 2×, clamped [-0.30, base] |
| Carrier-Repair consecutive_rollbacks | Carrier-Repair-Rollback inkrementiert `consecutive_rollbacks` | `_CARRIER_REPAIR_PHASE_PREFIXES`-Check vor Increment: Carrier-Repair-Rollbacks niemals zählen (§2.48 v9.11.3) |
| Spectral-Tilt-Drift in ADDITIVE-Phasen | HF-Extension ohne Tilt-Check (phase_06, phase_39) — Ära-Charakter wird zerstört ohne Goal-Verstoß | `era_result.spectral_tilt` in `kwargs` prüfen; Post-Tilt via `_estimate_spectral_tilt_quick()`; Cap wenn Deviation > material_tolerance (±1.5–±3.0 dB/oct je Material) (§2.46b) |
| Roughness/Sharpness Anstieg ungeprüft | DYNAMICS/ADDITIVE-Phasen (phase_35, phase_39) erhöhen psychoakustische Lästigkeit ohne Gate | `ArtifactFreedomGate._compute_roughness_zwicker()` + `_compute_sharpness_bismarck()` für `DYNAMICS/ADDITIVE/ENHANCEMENT`-Phasen; Penalty -0.05 bzw. -0.10 (§2.49c) |
| MERT als primäre Qualitätsmetrik | `MertPlugin.score()` direkt als HPI-Haupt-Koeffizient wenn VERSA verfügbar | VERSA primär; MERT als Proxy-Fallback wenn VERSA fehlschlägt; `metadata["mert_proxy_used"]` setzen (§2.44) |
| Lautheitsmessung ohne ISO 532-1 | `np.mean(audio**2)` oder LUFS-only nach Rumble/Multiband-Phasen | `compute_specific_loudness_zwicker(audio, sr)` → ΔN > 2.0 sone = FAIL, Dry/Wet-Rescue (§4.1b) |
| JND-blinde PMGG-Phase-Akzeptanz | Phase mit allen Deltas > 0 und < JND wird identisch zu signifikant positiver Phase behandelt | `JND_MIN_DELTA` Dict in `_run_with_retry()`: wenn alle Deltas ≥ 0 UND alle < JND → `sub_threshold`, kein Retry, `metadata["sub_threshold_phases"]` (§2.47b) |
| Uniforme Goal-Gewichtung | Alle 14 Goals gleich gewichtet via Minimax (`_max_regression` ohne Weights) | `estimate_goal_importance()` → Per-Song-Profil → `goal_weights` in PMGG/CIG/GPP/FC (§2.56) |

### Sprachkonvention
- **UI-Texte, Fehlermeldungen**: Deutsch (Ursache + Lösungsvorschlag)
- **Code-Kommentare, Docstrings, Log-Meldungen**: Englisch

### Test-Infrastruktur
- `pytest`/`conftest`-Teardowns für große Suiten bevorzugen **leichten inkrementellen GC** (`gc.collect(0)` o. ä.); volles `gc.collect()` nur cadence-gesteuert oder an Datei-/Session-Grenzen.
- Lang lebige Hintergrund-Manager (z. B. Monitor-Threads) brauchen einen **idempotenten `shutdown()`-Kontrakt** mit `Event.set()` + `join(timeout=...)`; Test-Sessions räumen diese Manager in `pytest_sessionfinish` oder Finalizern ab.
- **VERBOTEN**: unbedingtes Full-GC nach jedem Test in großen Suiten und das Verlassen auf `daemon=True` als einziges Cleanup-Modell.

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

## Kanonischer Pipeline-Ablauf (Kurzfassung)

`AurikDenker.denke()` → ReparaturDenker → RekonstruktionsDenker → RestaurierDenker → UV3:
DCOffset → TDP(HPSS) → RestorabilityEstimator → SongCalibration → Era/Genre/Medium-Classifier →
GoalApplicabilityFilter → DefectScanner(32) → CausalDefectReasoner → GPOptimizer →
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

### [RELEASE_MUST] §2.45a Mid-Pipeline-Loudness-Drift-Guard (v9.10.128, erweitert v9.11.5)

Frühe subtraktive Phasen dürfen den wahrgenommenen Musikpegel nicht kollabieren lassen.

- Betroffene Phasenklasse: breitbandig/subtraktiv (z. B. Denoise, Noise-Gate, Dereverb, Surface/Hiss-Reduction)
- Invariante: material-adaptiver per-Phase-RMS-Drift-Guard muss aktiv sein
- **Gated-RMS Pflicht** (§2.45a-I): RMS-Messungen frame-basiert, nur Frames > −50 dBFS; Stille-Frames ignorieren; Stereo→Mono-Downmix vor Framing
- **Envelope-Aware Gain** (§2.45a-II): Makeup-Gain NUR auf musikalische Frames; Stille-Frames unverändert (Gain=1.0); 10 ms Crossfade
- **Soft-Limiter bedingt** (§2.45a-III): tanh-Shaping bei 0.92 NUR wenn peak > 0.98 (echtes Clipping-Risiko)
- **Dreistufige Kaskade** (§2.45a-IV): Per-Phase → Mid-Pipeline (kumulativ) → End-of-Pipeline (final); jede Stufe nutzt Gated-RMS + Envelope-Gain
- Guard-Reaktion: **nicht** Phase wirkungslos machen; stattdessen begrenzte Dry/Wet-Rescue oder sichere Makeup-Gain-Kompensation
- Peak-Guard Pflicht: Gain-Limits mit `np.percentile(np.abs(audio), 99.9)` (kein `np.max()`)
- Telemetriepflicht: pro Phase `rms_drop_db` und `loudness_makeup_db` in Phase-Metadata; Pipeline-Metadaten führen Top-Drops

**Normativer Zweck**: Defektentfernung bleibt wirksam, ohne musikalische Substanz oder Natürlichkeit bereits in der Phase-Kette hörbar zu verlieren (§0, §2.45, P1/P2-Hartregeln).

> Pipeline-Details: Spec 02 §2.45a + Spec 04 §4.6 + Skill `pipeline-debug`

## 14 Musical Goals (Kurzreferenz)

| Prio | Ziele |
|---|---|
| **P1** | Natürlichkeit ≥ 0.90, Authentizität ≥ 0.88 |
| **P2** | TonalCenter ≥ 0.95, Timbre ≥ 0.87, Artikulation ≥ 0.85 |
| **P3** | Emotionalität ≥ 0.82, MikroDynamik ≥ 0.88, Groove ≥ 0.83 |
| **P4** | Transparenz ≥ 0.82, Wärme ≥ 0.75, BassKraft ≥ 0.78, SepFidelity ≥ 0.78 |
| **P5** | Brillanz ≥ 0.78, Raumtiefe ≥ 0.70 |

**Regressions-Regime** (differenziert — §2.29d, aktualisiert §2.54):
- **P1/P2** (Natürlichkeit, Authentizität, Tonal, Timbre, Artikulation): **Pipeline-Ende-Pflicht** — am Ende der gesamten Kette müssen alle P1/P2-Goals ≥ Schwellwert liegen. Einzelphasen dürfen vorübergehend P1/P2-Proxy-Werte senken, wenn Carrier-Repair (§2.44 Referenz-Paradoxon) oder restorative Defektentfernung (§2.29c Baseline-Capping) der Grund ist. Der CumulativeInteractionGuard (§2.48) ist die materialadaptive **Notbremse** (§2.54), nicht die Routine-Steuerung.
- **P3–P5**: **Pipeline-Netto-Budget** — Einzelphasen dürfen vorübergehend verschlechtern, wenn am Ende der Kette alle Goals ≥ Schwellwert. PMGG loggt Zwischenregressionen, blockiert aber nicht.

Details: Skill `fix-metric`

### [RELEASE_MUST] §2.56 Song-Goal-Importance — Per-Song Goal Weighting (v9.12.0)

Die 14 Goals bilden eine **Pareto-Front**: nicht alle gleichzeitig maximierbar. `estimate_goal_importance()` berechnet ein individuelles Gewichtungsprofil in **5 Stufen**:

| Stufe | Inhalt | Parameter |
|---|---|---|
| **1. Label** | Genre (16 Profile) → Era → Material → Vocal → Restorability → Studio 2026 | `genre_label`, `era_decade`, `material_type`, `vocal_detected/confidence`, `restorability_score`, `is_studio_2026` |
| **2. Audio** | SNR, Bandbreite, Dynamik, Stereo, BPM, Defekt-Schwere, Spektraler Tilt + Carrier-Chain (§2.46a: `transfer_generation_count`, `cumulative_hf_loss_db`, `source_fidelity_confidence`) | 10 optionale float/dict/int Parameter |
| **3. Psychoakustik** | Roughness (Zwicker), Sharpness (Bismarck), Spectral Flatness, Tonality, Frequenzbalance, Masked-Ratio (Sanity: ≥0.95/≤0.01 → ignoriert), Centroid (Bark) | 7 optionale float/dict Parameter |
| **4. Vokal/Harmonik** | HNR (Pitch-Period AC, [−10,+20] dB), Harmonic Coherence (STFT), Crest Factor (99.9-Pctl, [6,16] dB), Transient Density (STFT+50ms, /s) | 4 optionale float Parameter |
| **5. Interactions** | 6 superadditive Cross-Feature-Effekte (Rough×Noisy, HNR×Vocal, BW×Dark, Coh×Tonal, Dyn×Trans, Chain×Noisy) | Kombinationen aus Stufe 2–4 |

**Soft-Cap**: `w > 1.5 → 1.5 + excess/(1+3·excess)` (Asymptote 1.83); `w < 0.5` analog (0.17). Danach P1/P2-Floor ≥ 0.70, Hard-Bounds [0.30, 2.00].

**Integration**: PMGG (`weighted_reg = reg × weight`), CIG (gewichtete Drift), GoalPriorityProtocol (gewichtete Conflict-Resolution + Abort), FeedbackChain (gewichtete GPP-Prüfung).

**Messverfahren-VERBOTEN**: Lag-1-HNR (`_compute_hnr`), 46ms-Coherence (`_estimate_harmonic_coherence`), `np.max()`-Crest, RMS-Flux-Transients. → Spec 01 §2.56e.

**Implementierung**: `backend/core/song_goal_importance.py` — Feature-Extraktion in UV3 `restore()`, durchgereicht als `goal_weights`.

**Invarianten**: Einmalige Berechnung pro Song; alle Audio-Features optional (None → Skip); Fehler → Uniform-Fallback (1.0); Pipeline-Blockade verboten.

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

### [RELEASE_MUST] §2.46a Deep-Transfer-Chain-Pflicht (v9.10.124)

Importsongs mit **3+ Tonträgerstufen** sind vollständig zu modellieren; die Kette darf nicht auf Primärträger + 1 Sekundärstufe verkürzt werden.

- `transfer_chain` muss reale Mehrfachkopien abbilden (z. B. `shellac -> reel_tape -> cassette -> cd_digital -> mp3_low`).
- Zwischenstufen sind Pflicht, wenn Evidenz vorliegt: `cd_digital`/`dat` vor lossy Codec darf nicht weggelassen werden.
- Keine Rückwärtssprünge in der Kette: Reihenfolge bleibt kausal gemäß `_MEDIUM_ORDER`.
- Nach Material-Normalisierung sind Duplikate zu konsolidieren (Konfidenz = `max`), damit `source_fidelity_generation_count` nicht künstlich aufgebläht wird.

**Testpflicht**:
- Mindestens ein Unit-Test für 4-stufige Kette mit digitaler Zwischenstufe.
- Mindestens ein Unit-Test für `file_ext=.mp3` mit physikalischer Inferenz und 4-stufigem Ergebnis.

**Invariante**: Eine erkannte Mehrfachkette muss vollständig bis SongCalibration/SourceFidelity/Export-Metadata propagieren.

> Details (Signalkette, Phasen-Ordering in UV3): Spec 02 §2.46 + Skill `pipeline-debug`

### [RELEASE_MUST] §2.47 Adaptive-Intelligence-Prinzip (v9.10.123)

Jede Eingabe ist ein einzigartiges Musikstück. Das System passt sich **vor** der Verarbeitung an das konkrete Material an.

**Adaptions-Kaskade (9 Schritte, kanonische Reihenfolge):**
1. `MediumDetector.detect(audio, sr, file_ext=...)` → transfer_chain, primary_material (§6.7)
2. `EraClassifier.classify()` → decade, era_profile **+ ERB-Salience-Annotation** (v9.11.0)
3. `GenreClassifier.classify()` → genre_label, genre_profile
4. `RestorabilityEstimator.estimate()` → restorability_score, tier
5. `DefectScanner.scan()` → 46 DefectTypes × Severity × Locations
6. `CausalDefectReasoner` → 49 Ursachen → Phase-Selektion
7. `SongCalibrationProfile` → `global_scalar∈[0.50,1.50]`, `family_scalars[*]∈[0.30,1.80]`
8. `SongGoalImportance` (§2.56) → 14 Per-Song-Gewichte [0.3–2.0] (5 Stufen: Label/Audio/Psychoakustik/Vokal-Harmonik/Interactions)
9. `GPOptimizer.propose()` → Pareto-optimale Hyperparameter

**Adaptions-Erweiterungen (v9.11.0):**
- **Salience-aware Phase-Skipping**: `_apply_phase_skipping` liest ERB-adjustierte `DefectScore.severity`; fully-masked Defekte (n_masked≥3, n_salient=0) erhalten zusätzlich 50 % Reduktion — vermeidet Phasenaktivierung für unhörbare Schäden ohne §0-Verletzung.
- **PhaseConductor** (§2.52): misst nach jeder Phase einen 4D-State-Vektor und gibt adaptive `strength`-Empfehlung für die nächste Phase.
- **SGMSE+ Tier-0** in `phase_03_denoise`: Diffusionsbasiertes Denoising (Richter et al. 2022) als erster Pfad für Vokal-Material bei `quality_mode in (quality, maximum)` — vor ML-Hybrid-Pfad.
- **Carrier-Formant-Decay-Inversion** in `phase_42`: `_restore_carrier_formant_decay(audio, sr, material_type)` als Stage 0.5 invertiert trägertypische F1–F4-Unterdrückung (vinyl/reel_tape/tape/shellac/minidisc) via zero-phase Bell-EQ.

**Edge-Cases**: < 10 s → Groove/MicroDyn off | > 60 min → segmentweise | Restorability < 20 + Shellac → Scale 0.65, P3–P5 ≥ 0.50

**ML-Failure-Degradationskaskade (Fallback-Pflicht für jedes ML-Plugin):**

| Failure | Primär-Fallback | Sekundär-Fallback |
|---|---|---|
| DeepFilterNet OOM | OMLSA/IMCRA | Spectral-Gating (Dry-Signal wenn SNR > 35 dB) |
| MDX23C Stem-Sep OOM | NMF-β-Separation (sdB ≥ 5) | HPSS (Medianfilter-Trennung) |
| AudioSR OOM | Harmonische Oberton-Synthese + PGHI | Spectral-Band-Replication |
| MP-SENet OOM | OMLSA/IMCRA DSP (§4.4) | Bypass (phase_43 Phase-Skip) |
| CREPE Pitch-Track | pYIN (Mauch & Dixon 2014) | YIN (de Cheveigné & Kawahara 2002) |
| MertPlugin OOM | DSP-Analyse (F0+Harmonizität+FluxKohärenz) | Bypass (HPI ohne MERT-Anteil) |

**Invariante**: Kein ML-Failure darf Pipeline abbrechen. Fallback in `metadata["ml_fallbacks_used"]` protokollieren.

**GP-Wissenstransfer**: Pro `gp_memory_key` (Genre × Material); Cross-Material via Ähnlichkeitsmatrix bei < 10 Beobachtungen.

> Details (Kaskade, Ähnlichkeitsmatrix, Edge-Cases): Spec 02 §2.47 + Spec 05

### [RELEASE_MUST] §2.52 PhaseConductor — Inter-Phase Adaptive Feedback (v9.11.0)

Nach jeder Phase misst `PhaseConductor` (Singleton `get_phase_conductor()` in `backend/core/phase_conductor.py`) einen **4D-State-Vektor** und empfiehlt die optimale `strength` für die nächste Phase:

| Dimension | Beschreibung | Normierung |
|---|---|---|
| `noise_floor_db` | 5. Perzentil PSD (Rauschboden) | dBFS, ≤ 0 |
| `hf_energy_ratio` | Energie 8 kHz–Nyquist / Breitband | [0, 1] |
| `transient_density` | Onset-Rate [Events/s] | roh; as_vec() → /20 |
| `harmonic_coherence` | Autocorrelation-Peak-Ratio | [0, 1] |

**Workflow in `_execute_pipeline`:**
1. Vor Phase-Loop: `_conductor = get_phase_conductor(); _conductor.reset()`
2. Nach jeder erfolgreichen Phase: `_conductor.measure_state(current_audio, sr, phase_id)`
3. Look-Ahead: `_conductor.recommend(next_phase_id, state, material_type)` → `_conductor_strength_hints[next_pid]`
4. `_profiled_phase_call`: injiziert hint als `strength` kwarg (nur wenn `strength` nicht explizit gesetzt)

**Invarianten:**
- Advisory-only: PMGG-Strength hat immer Vorrang (explizit gesetzt = explizit gewinnt)
- `_NEVER_SKIP` frozenset (phase_01, phase_09, phase_12, phase_14, phase_15) — nie `skip_recommended=True`
- `_MIN_STRENGTH` Dict: Untergrenzen je kritischer Phase (z. B. phase_03 ≥ 0.35)
- Jede Exception → `logger.debug`, Pipeline läuft unverändert weiter
- Rein DSP, kein ML, < 50 ms pro `measure_state()` für 1 min Audio

> Implementierung: `backend/core/phase_conductor.py` — `PhaseConductor`, `get_phase_conductor()`
> Aufruf: `backend/core/unified_restorer_v3.py` — sequentieller Phase-Loop, nach §2.31a MidCalibrate-Block

### [RELEASE_MUST] §2.53 Experience-Closed-Loop + Bridge/UI-Propagation (v9.11.1)

Neue Zielpriorität: **maximales Hörerlebnis** wird im Produktionslauf als explizite Laufzeit-Telemetrie geführt und bis in die UI propagiert.

**Normative Invarianten:**

1. `UnifiedRestorerV3.restore()` MUSS folgende Felder in `RestorationResult.metadata` befüllen:
    - `song_calibration.cluster_key`
    - `song_calibration.cluster_policy`
    - `joy_runtime_index` (`joy_index`, `fatigue_index`, `components`)
    - `auto_improvement_recommendations` (`count`, `recommendations[*].focus/action/reason`)
2. `backend/api/bridge.py` MUSS `get_experience_insights(result)` bereitstellen.
    - Rückgabe ist frontend-sicher, NaN/Inf-frei und fehlertolerant.
3. `RestaurierDenker` und `AurikDenker` MÜSSEN `metadata` end-to-end propagieren.
    - VERBOTEN: Metadaten beim Konvertieren nach `AurikErgebnis` verwerfen.
4. Frontend (`Aurik910/ui/modern_window.py`) MUSS die Runtime-Signale sichtbar machen:
    - Statuszeile: Freude-/Ermüdungsindex.
    - Info-Banner: Cluster-Policy + Top Auto-Improve-Empfehlungen.
5. Fehlerverhalten ist **non-blocking**: fehlende Experience-Telemetrie darf den Export nicht stoppen,
    MUSS aber als degrade-hinweis protokolliert werden (kein stilles Ignorieren).

### [RELEASE_MUST] §2.53a Exzellenz-API-Kompatibilitätsvertrag (v9.11.1)

`AurikDenker` MUSS mit beiden Exzellenz-APIs kompatibel sein:

- Primär: `ExzellenzDenker.messe_und_repariere(audio, sr, ...) -> (audio, goals)`
- Legacy-Fallback: `ExzellenzDenker.messe_ziele(audio, sr, ...)`

**Verboten:** Harte Annahme, dass nur eine der beiden Methoden existiert.
Bei Fallback MUSS ein konsistenter Stage-Note-Eintrag (`Legacy-Goal-Messpfad`) gesetzt werden.

### [RELEASE_MUST] §2.53b Denker-Plan-Determinismus in UV3 (v9.11.2)

Wenn `UnifiedRestorerV3.restore(..., precomputed_phase_plan=[...])` aufgerufen wird,
gilt der Denker-Plan als **Source of Truth**.

**Invarianten:**
- UV3 MUSS autonome Planungsblöcke `_select_phases()` und `_optimize_phase_plan_intelligence()` überspringen.
- UV3 MUSS `selected_phases` direkt aus `precomputed_phase_plan` ableiten.
- UV3 MUSS `phase skipping` in diesem Pfad deaktivieren (kein nachträgliches Ausdünnen des Denker-Plans).
- Zulässig sind nur normative Sicherheitsinjektionen (z. B. §2.50 Stereo-Notfall-Remediation).
- Stale-Plan-Zustand aus vorherigen Läufen ist verboten (`_last_material_priority_phases` darf nicht implizit weiterwirken).

**Rationale**: Verhindert Doppel-Orchestrierung und driftende Entscheidungen zwischen Denker und UV3.

### §0b Konfliktauflösung / Anti-Widerspruch (v6.9)

Wenn Vorgaben kollidieren, gilt strikt:

1. `§0` Klangwahrheit + RELEASE_MUST-Invarianten
2. Neuere versionsmarkierte Abschnitte (höhere `v9.x` / `instructions_version`)
3. Spezifische Feld-/Kontrakt-Regeln vor generischen Stilregeln

Damit darf älterer Text nie die neuen Experience-/Propagation-Invarianten außer Kraft setzen.

### [RELEASE_MUST] §2.47a Frontend-Backend-PreAnalysis-Handover (v9.10.127)

Pre-Analyseergebnisse (Medium, Era, Genre, Defect, Restorability) werden **EINMALIG** während Import berechnet. Sie MÜSSEN als **direktes Übergabeobjekt** an die Restaurierungs-Pipeline weitergereicht werden — **NICHT mehrfach aus Cache rekonstruiert** in asynchronen Batch-Threads.

**Invarianten (Violation = RELEASE-Blocker)**:
- `run_pre_analysis()` läuft **GENAU 1x** nach Import (native SR, alle 5 Analysen parallel)
- Frontend speichert komplette `PreAnalysisResult` in `_latest_pre_analysis_result`
- Mode-Click → Queue-Item trägt `PreAnalysisResult` in `queue_settings` (nicht nur Cache-Keys)
- Falls `PreAnalysisResult.defects` vorhanden ist, trägt das Queue-Item zusätzlich `cached_defect_result` als direkte Defect-Referenz
- `BatchProcessingThread` prioritiert direktes Result **vor** Bridge-Cache-Lookup
- `BatchProcessingThread` reicht das konkret verwendete Defect-Result **immer** als `cached_defect_result` an `AurikDenker.denke()` weiter, auch wenn `PreAnalysisResult` unvollständig ist
- `AurikDenker.denke()` empfängt `pre_analysis_result` kwarg oder unpacked `cached_*` kwargs
- Auf **neuem File-Import** wird vorheriger Cache **HARD gelöscht** (verhindert Staleness)
- **Invariante: `MediumDetector.detect()` wird GENAU 1x aufgerufen**, nie 2–3x

**Rationale**: Cache-basierte Rekonstruktion erzeugt **Race Conditions** in asynchronen Threads. Single Direct Handover = deterministisch + thread-safe.

**Implementierungs-Checkliste**:
- [✅] UI speichert `PreAnalysisResult` nach erfolgreicher `_pre_analysis_bg()`
- [✅] Mode-Click injiziert `pre_analysis_result` in Queue-Item Settings
- [✅] Mode-Click injiziert bei vorhandenem Scan zusätzlich `cached_defect_result` in Queue-Item Settings
- [✅] BatchProcessingThread prüft Queue-Settings first
- [✅] BatchProcessingThread ergänzt `PreAnalysisResult.defects` aus `cached_defect_result`, falls nötig
- [✅] AurikDenker.denke() erhält `pre_analysis_result` kwarg
- [✅] Test: `tests/unit/test_pre_analysis_handover_no_double_detect.py` (2/2 passing)

> Details: Spec 02 §2.37 + Skill `pipeline-debug`

### [RELEASE_MUST] §6.2a Material-Pflicht-Phasen (v9.10.73)

Prioritäts-Phasen eines Materials MÜSSEN aktiviert werden, **unabhängig vom DefectScanner-Severity-Score**.
Begründung: DefectScanner arbeitet statistisch auf limitiertem Ausschnitt; einzelne schwere Defekte können unter Schwelle liegen.

```python
# Beispiel-Pflicht-Phasen (vollständig: backend/core/unified_restorer_v3.py _MATERIAL_PRIORITY_PHASES)
MATERIAL_PRIORITY_PHASES = {
    "vinyl":    ["phase_09_crackle_removal", "phase_12_wow_flutter_fix", "phase_05_rumble_filter"],
    "tape":     ["phase_29_tape_hiss_reduction", "phase_24_dropout_repair"],
    "reel_tape":["phase_29_tape_hiss_reduction", "phase_03_denoise", "phase_24_dropout_repair"],
    "shellac":  ["phase_03_denoise", "phase_06_frequency_restoration", "phase_01_click_removal"],
    "mp3_low":  ["phase_23_spectral_repair", "phase_03_denoise", "phase_50"],
    # … alle 15 Materialtypen
}
# Ausnahme: GoalApplicabilityFilter kann Material-Phasen für spezifisches Material deaktivieren
# (z.B. phase_48 Stereo-Imaging bei Mono-Quellen)
```

**Invariante**: Kein Import-Song wird ohne seine materialspezifischen Reparatur-Phasen verarbeitet.

### [RELEASE_MUST] §2.29c PMGG Restorative-Baseline-Capping (v9.10.96)

Restorative Phasen (Denoise, Dereverb, Declip) messen `scores_before` auf **defektbelastetem** Audio.
Breitbandrauschen inflationiert `transparenz`/`brillanz`; Dropout verfälscht `groove`. → falsche Regression → Defekte bleiben.

**Lösung**: Bei `_RESTORATIVE_PHASES` wird `scores_before` auf `canonical_threshold + 0.05` gedeckelt:

```python
_RESTORATIVE_PHASES = frozenset({"phase_02","phase_03","phase_09","phase_18","phase_20","phase_23","phase_24","phase_29","phase_49"})
# effective_scores_before[g] = min(measured, canonical_threshold[g] + 0.05)
# Delta-Check: scores_after[g] - effective_scores_before[g]
```

**Invariante**: Enhancement-Phasen nutzen echte `scores_before` (kein Capping).

> Details (CANONICAL_THRESHOLDS, Implementierung): Spec 02 §2.29c

### [RELEASE_MUST] §2.29e PMGG Team-Koordination über Vorphasen-Kontext (v9.11.5, erweitert v9.11.7)

PMGG darf Folgephasen nicht dazu drängen, intentionale Reparaturen der Vorphasen rückgängig zu machen.
Deshalb wird `prior_phase_context` aus UV3 in PMGG berücksichtigt.

**Normative Regel für alle Module/Phasen (zentral, ontologiebasiert):**

- UV3 schreibt pro erfolgreicher Phase `prior_phase_context` fort
  (inkl. `last_phase_type`, Typ-Counter, Semantik-Flags).
- PMGG muss team-policy-bewusst arbeiten (`_resolve_team_context_policy(...)`) und
  die Übergangs-Policy aus Vorphasen-Typ → Aktueller-Phasen-Typ ableiten.
- Team-Policy darf Goal-Exclusions, `threshold_multiplier` und `strength_cap`
  für die aktuelle Phase setzen (advisory-only).
- `PhaseGateLogEntry.metadata` erhält `team_policy_reason`, `team_excluded_goals`,
  `team_threshold_mult`, `team_strength_cap` wenn Team-Policy aktiv war.
- UV3 baut `metadata["team_coordination"]` (§2.53 RELEASE_MUST):
  `event_count`, `events`, `phase_type_summary`.
- `bridge.get_experience_insights()` propagiert `team_coordination` ins Frontend.

**CONFLICT_REGISTRY** (`backend/core/phase_ontology.py`):

- `CONFLICT_REGISTRY: dict[str, frozenset[str]]` — explizite Paare, bei denen Phase B
  die Arbeit von Phase A nicht neutralisieren darf.
- `get_conflict_phases(completed_phase_id)` — startswith-Matching, frozenset-Rückgabe.
- UV3 `_profiled_phase_call` injiziert `conflict_with_prior_phases: list[str]`
  wenn CONFLICT_REGISTRY Treffer liefert. Phase entscheidet selbst, wie sie reagiert.

**Normative Spezialregel für Phase 50 nach HF-Restaurationskette**
(`phase_06_frequency_restoration`, `phase_07_harmonic_restoration`, `phase_23_spectral_repair`):

- Goal-Exclusions sind um `brillanz`, `transparenz`, `timbre_authentizitaet` zu erweitern.
- Regressions-Threshold darf moderat gelockert werden (`threshold_multiplier`, capped).
- Initial-Strength darf konservativ gedeckelt werden (`strength_cap`).
- Catastrophic/Emergency-Retry-Pfad muss dieselbe Team-Policy anwenden;
    für `phase_50` mit Grund `phase50_after_hf_restoration` sind Emergency-Retries zu unterdrücken.

**Invariante**: Diese Koordination ist *advisory-only* für PMGG-Retry/Strength.
Export-Gates (§2.44/§2.49) bleiben unverändert hart.

> Details: Spec 02 §2.29e und Spec 06 §6.9b

### [RELEASE_MUST] §2.55 PMGG-CIG-Synchronisations-Invariante (v9.11.3)

**Strukturelle Invariante**: `CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[phase]` und `PMGG.PHASE_GOAL_EXCLUSIONS[phase]` sind **bidirektional synchron** für alle P1/P2-Goals.

**Formal**: Für jede Phase `p` gilt:
- `CIG_excl(p) ∩ P1P2 ⊇ PMGG_excl(p) ∩ P1P2` (PMGG→CIG: was PMGG von der Goal-Messung ausschließt, muss CIG aus dem Drift-Check ausschließen — sonst wird Falsch-Drift akkumuliert und triggert Rollback an späterer Phase)
- `PMGG_excl(p) ∩ P1P2 ⊇ CIG_excl(p) ∩ P1P2` (CIG→PMGG: was CIG nicht als Drift zählt, darf PMGG nicht als Regression blockieren)

**Warum**: PMGG erlaubt Phase bei voller Stärke (excl. Goal X) → CIG zählt Goal-X-Delta dennoch zum kumulativen Drift → bei späterer Phase (nicht excl.) übersteigt Drift die Toleranz → CIG-Rollback an falscher Stelle → pipeline-weite Stärke-Kaskade → Defekte bleiben unrepariert.

**VERBOTEN**: Neue Phase implementieren und nur eine der beiden Tabellen aktualisieren.

**Implementierung**: `backend/core/per_phase_musical_goals_gate.py` → `PHASE_GOAL_EXCLUSIONS`; `backend/core/cumulative_interaction_guard.py` → `_PHASE_SPECIFIC_DRIFT_EXCLUSIONS`

**Testpflicht**: CI-Regression-Test `tests/unit/test_pmgg_cig_sync.py` prüft die bidirektionale Synchronisation für alle Phasen automatisch.

---

### [RELEASE_MUST] §2.48 Kumulative-Phasen-Interaktions-Guard (v9.10.123, aktualisiert v9.11.2)

Phasen können isoliert korrekt, in Kombination destruktiv wirken (z.B. De-Noise + De-Reverb → Over-Denoising).

> **§2.54 ist übergeordnet**: Der Guard ist eine **Notbremse**, nicht die Steuerung. Drift-Toleranz wird
> **berechnet** aus Material/Restorability/Defect-Severity — nicht als Konstante definiert.

- **Kumulative P1/P2-Drift**: Nach jeder Phase messen; Drift-Toleranz materialadaptiv (§2.54 `compute_adaptive_drift_tolerance()`). Überschreitung → Rollback auf `best_checkpoint`.
- **Carrier-Repair-Phasen-Ausnahmen**: Phasen, die Tonträgerschäden invertieren — vgl. `_PHASE_SPECIFIC_DRIFT_EXCLUSIONS` — dürfen P1/P2-Goals vorübergehend senken; das ist Referenz-Paradoxon (§2.44, §2.55). Drift-Check für diese Goals ist bei diesen Phasen ausgesetzt (§2.55-Synchronisationspflicht).
- **STFT-Phasenkohärenz**: Nach ≥ 3 STFT-Phasen: Gruppenlaufzeit-Deviation ≤ 5 ms, sonst Rollback
- **Phasen-Reihenfolge**: Carrier-Chain-Inversions-Logik (§2.46); subtraktive vor additiven Phasen
- **Pipeline-Stopp adaptiv**: `max_consecutive_rollbacks = max(5, n_carrier_phases + 2)` — Mehrgenerations-Material benötigt mehr Carrier-Phasen, die einzeln rollback-anfällig sind.
- **Carrier-Repair-Phasen inkrementieren `consecutive_rollbacks` NICHT** (v9.11.3): Wenn eine Carrier-Repair-Phase (definiert in `_CARRIER_REPAIR_PHASE_PREFIXES`) rollback-bedingt zurückgesetzt wird, darf der `consecutive_rollbacks`-Zähler **nicht** erhöht werden. Andernfalls würde der Pipeline-Stopp nach `max_consecutive_rollbacks` fälschlicherweise durch restorative Phasen ausgelöst (VERBOTEN: preventable pipeline abort). Gilt sowohl für P1/P2-Drift-Rollback als auch Critical-Pair-Rollback-Pfad. Implementierung: `_CARRIER_REPAIR_PHASE_PREFIXES` tuple in `backend/core/cumulative_interaction_guard.py`.

> Details (Interaktions-Paare, Checkpoint-Management): Spec 02 §2.48 + Skill `pipeline-debug`

### [RELEASE_MUST] §2.49 Artefakt-Freiheits-Gate (v9.10.123)

Dediziertes Gate — unabhängig von Musical Goals. 5 Artefakttypen (Musical Noise, Pre-Echo, Spectral Holes, Phase-Cancellation, Metallic Ringing) + Rauschtextur-Kohärenz. Schwellwerte **material-adaptiv**.

- `artifact_freedom = 1.0 - (salience_weighted_artifact_count / max_tolerance)` — perzeptuell gewichtet (Frequenz, Kontext, Dauer)
- `artifact_freedom < 0.95` → **Per-Phase**: Rollback auf `_afg_phase_input`. **Finales Export-Gate**: `restored_audio` auf `_hpi_best_rollback_audio` → `original_audio_for_goals` zurücksetzen (Rollback-Kaskade, identisch §2.44). Nur Logging ohne Audio-Rollback ist eine RELEASE_MUST-Verletzung.
- **Per-Phase-Modus**: Alle Detektoren messen das **Delta** vor/nach der Phase — keine absoluten Eigenschaften des Ausgangssignals.
- **Finaler `_artifact_freedom_score`** (v9.10.126): = **Minimum aller per-Phase-Scores** aller akzeptierten Phasen (`_min_per_phase_afg_score`). VERBOTEN: `artifact_gate.evaluate(pre_pipeline_audio, pipeline_output)` — jede echte Restaurierung liefert zwangsläufig 0.000, weil intentionale Signaländerungen (Entrauschen, Bandbreite) als Artefakte erscheinen.
- **Musical-Noise-Direktionalität** (v9.10.125): `_detect_musical_noise` darf nur Bins flaggen, bei denen `restored_spectrum[j] > orig_spectrum[j] × 1.05` — d.h. Energie wurde **addiert**. Subtractive Phasen (phase_28, phase_03, phase_29, phase_01) erzeugen Residual = entfernter Schaden → False-Positive-Kaskade wenn Direktionalität fehlt (`artifact_freedom=0.000`, 50 Artefakte, Rollback-Loop).
- **Phase-Cancellation-Detektor**: Erhält immer `original_stereo` (Audio vor der Phase / vor der Pipeline) und überspringt Frames, die bereits im Input anti-phasig/mono-inkompatibel waren (§2.50). Nur Frames flaggen, die **diese Phase** neu verschlechtert hat (Delta > 0.05 in mono_compat). Zusätzlich: Absoluter Imbalance-Guard — wenn Input L/R-Imbalance < 6 dB aber Output > 20 dB → Artefakt (fängt Single-Phase-Kollapsen auch ohne per-Frame-Delta).

**§2.49b [RELEASE_MUST] Post-Pipeline Kumulativer Stereo-Collapse-Guard (v9.10.126)**

Per-Phase-Guards sind blind für kumulativen Drift: 4 Stereo-Phasen à 6–8 dB = −111 dBFS R-Kanal, jede Phase besteht ihren δ-Guard. Fix: Direkt nach Phase-Loop, vor `_pmgg_log_entries`:
```python
if current_audio.ndim == 2 and afg_pre_pipeline_audio.ndim == 2:
    cu_imb = abs(L/R_dB(current_audio))     # > 20 dB = Kollaps
    pp_imb = abs(L/R_dB(pre_pipeline))      # < 6 dB = war ausgeglichen
    if cu_imb > 20.0 and pp_imb < 6.0:
        # Kaskade: best_clean_checkpoint (falls selbst < 20 dB Imbalance) → pre_pipeline
        current_audio = validated_recovery
```

**§2.44/§2.49 HPI-Rollback-Checkpoint Stereo-Health-Validation (v9.10.126)**

Vor Verwendung von `_hpi_best_rollback_audio` als Rollback-Ziel: L/R-Imbalance prüfen. Checkpoint-Imbalance > 20 dB → verwerfen, Fallback auf `original_audio_for_goals`. Ohne diese Prüfung restauriert der HPI-Rollback ein stereo-zerstörtes Signal.

> Details (Schwellwert-Tabellen, Rauschtextur-Messung, Salienz-Faktoren): Spec 02 §2.49, §2.49b

**[RELEASE_MUST] §2.51 Stereo-Kohärenz-Invariante für Phasen (v9.10.127)**

Jede Phase mit Stereo-Audio **MUSS** M/S-Domain oder Linked-Stereo verwenden. Verboten: unabhängiges L/R-Processing mit gain- oder zeitvarianter Operation.

| Phase | Strategie |
|---|---|
| `phase_07` Harmonic Restoration | **M/S**: Obertöne nur auf Mid |
| `phase_18` Noise Gate | **Linked**: Gate öffnet wenn `max(L_rms, R_rms) > threshold` |
| `phase_21` Harmonic Exciter | **M/S**: Excitation nur auf Mid, Side unverändert |
| `phase_23` Spectral Repair | **M/S**: Reparatur auf Mid; Side minimal |
| `phase_24` Dropout Repair | **Linked**: kohärente L/R-Grenze + Füllung |
| `phase_27` Click/Pop Removal | **Linked**: Detektion auf Mono-Mix `(L+R)/2`, Repair synchronisiert auf L+R |
| `phase_29` Tape Hiss Reduction | **Linked**: OMLSA-Gain aus Mid-Sidechain `(L+R)/√2`, identisch auf L+R |
| `phase_35` Multiband Compression | **Linked**: Gain auf `√(L²+R²)/√2` |
| `phase_50` Spectral Repair (Inpainting) | **M/S**: Reparatur auf Mid voll, Side konservativ (×2 Threshold) |

Verletzung → §2.49 flaggt 2–5 Phase-Cancellation-Artefakte → Rollback → OQS-Einbuße.

> Details + Code-Patterns: Spec 08 — instructions_version 6.9

**DSP-Invariante (Peak-Guards)**: Gain-Berechnungen, die einen Peak-Schwellwert schützen (Headroom-Guard, True-Peak-Limiter-Vorberechnung), müssen `np.percentile(np.abs(audio), 99.9)` statt `np.max()` verwenden. Ein einzelnes Impuls-Artefakt (Crackle, Click) darf die Normalisierung des gesamten Musiksignals nicht blockieren.

### [RELEASE_MUST] §2.50 Material-Adaptive Gate Baseline (v9.10.125)

Vor Pipeline-Beginn misst das System die Artefakt-Charakteristik des degradierten Inputs als **Quellmaterial-Baseline**. Ein Gate darf **niemals** eine Eigenschaft bestrafen, die im Input bereits in gleicher oder stärkerer Ausprägung vorhanden war — das wäre eine Verletzung von §0 (Primum non nocere) und §2.47 (Adaptive Intelligence).

**`ArtifactFreedomGate.measure_source_baseline(audio, sr, material_type)` → `SourceMaterialBaseline`:**
- `phase_cancellation_ratio` — Anteil der 100-ms-Frames mit Stereo-Feld-Problem im Input
- `stereo_mono_compat_mean` — mittlere Mono-Kompatibilität des Quellmaterials
- `has_critical_stereo_issue` — True wenn > 20 % Frames mono-inkompatibel (Trägerkettendefekt)
- `has_anti_phase_region` — True wenn ein Frame `lr_corr < 0` hat
- `hf_loss_db` — geschätzter Hochfrequenz-Verlust vs. Breitband-Referenz (Trägerkettenanzeiger)

**Autonome Remediation (in `restore()`, nach `_select_phases`, vor Phase-Skipping):**
1. `has_critical_stereo_issue = True` → `phase_14_phase_correction` + `phase_15_stereo_balance` als Notfall-Pflicht-Phasen injizieren
2. `has_anti_phase_region = True` → `phase_14_phase_correction` injizieren
3. Logging: `§2.50 Stereo-Notfall-Remediation: ratio=0.82, mean_compat=0.41 → ['phase_14...', 'phase_15...']`

**Gate-Paradoxon-Invariante**: Feuert §2.49 bei > 50 % der Phasen mit `artifact_freedom = 0.0`, ist das ein **Implementierungsfehler** — kein Rollback-Grund. Der Detektor wird falsch kalibriert (Delta fehlt, `original_stereo` fehlt). Dieser Zustand darf nie zur Pipeline-Blockade führen.

**Baseline in Metadaten**: `RestorationResult.metadata["source_material_baseline"]` enthält alle 5 Felder für Audit und Diagnose.

> Implementierung: `backend/core/artifact_freedom_gate.py` — `SourceMaterialBaseline`, `measure_source_baseline()`.
> Aufruf: `backend/core/unified_restorer_v3.py` — §2.50-Block in `restore()` nach `_select_phases`.

### [RELEASE_MUST] §2.54 Adaptives Phasen-Optimum — Messen-Handeln-Validieren (v9.11.2)

> **Dieses Paradigma ist normativ übergeordnet gegenüber allen festen Schwellwerten in §2.48, §2.29d, §2.45.**
> Feste Schwellwerte sind **Notbremsen** (letztes Sicherheitsnetz), nicht die Steuerung.
> Die Steuerung ist das iterative Herantasten jeder Phase an ihr Optimum.

**Grundprinzip**: Jeder Song ist einzigartig — anderes Genre, andere Ära, andere Tonträgerkette, andere Defekte.
Feste Schwellwerte können diese Vielfalt nicht abbilden. Stattdessen gilt:

**Jede Phase durchläuft einen Messen→Handeln→Validieren-Zyklus:**

```
1. MESSEN     — Zustand vor der Phase: Klangtreue, Defekt-Schwere, Energie-Profil
2. HANDELN    — Phase mit materialadaptiver Stärke ausführen (SongCal × PhaseConductor)
3. VALIDIEREN — Zustand nach der Phase messen: Hat sich der Klang verbessert?
4. ENTSCHEIDEN:
   a) Verbesserung klar hörbar       → Phase akzeptieren, weiter
   b) Verbesserung marginal          → Stärke anpassen, erneut (max 3 Iterationen)
   c) Verschlechterung               → Stärke reduzieren oder Phase überspringen
   d) Katastrophale Beschädigung     → Rollback (Notbremse)
5. BESTES ERGEBNIS BEHALTEN — Über alle Iterationen das perceptuell beste Resultat wählen
```

**Wer steuert diesen Zyklus?**

| Komponente | Rolle | NICHT die Rolle |
|---|---|---|
| **Denker** (Reparatur/Rekonstruktion/Restaurier) | Plant Phase-Reihenfolge + Initialkonfiguration basierend auf Material/Era/Genre/Defekte | Feste Schwellwerte setzen |
| **PhaseConductor** (§2.52) | Misst 4D-Zustand nach jeder Phase, empfiehlt `strength` für nächste Phase | Starres Pass/Fail |
| **PMGG** (§2.29) | Misst Musical-Goals-Delta pro Phase, steuert Stärke-Iteration | Blocken mit `regression > 0.02` |
| **SongCalibration** (§2.47) | Skaliert alle Stärken material-/song-adaptiv | Universelle Konstante |
| **CumulativeInteractionGuard** (§2.48) | **Nur Notbremse**: Fängt katastrophale kumulative Drift | Routine-Steuerung der Pipeline |
| **GPOptimizer** | Lernt Pareto-optimale Hyperparameter aus vorherigen Songs | Erstmalige Parameterwahl |

**Adaptions-Kette (Material → Drift-Toleranz):**

Die Drift-Toleranz des CumulativeInteractionGuard wird **berechnet**, nicht fest vorgegeben:

```python
# §2.54 Adaptive Drift-Toleranz (ersetzt MAX_CUMULATIVE_DRIFT = -0.05)
adaptive_drift_tolerance = compute_adaptive_drift_tolerance(
    restorability_score,     # 0–100: wie stark degradiert? → mehr Spielraum
    material_type,           # vinyl/shellac brauchen mehr als cd_digital
    defect_severity_mean,    # hohe mittlere Severity → mehr Toleranz nötig
    n_active_phases,         # mehr Phasen → mehr kumulative Drift normal
)
# Ergebnis: z.B. -0.03 (CD, leicht) bis -0.25 (Shellac-4-Gen, schwer degradiert)
```

PMGG-Threshold analog: Regressions-Budget pro Phase hängt vom Material und der Defekt-Schwere ab —
ein stark verrauschter Vinyl-Song akzeptiert mehr Chroma-Drift nach Denoise als ein CD-Rip.

**Invarianten (bindend für ALLE Guards):**

1. **Kein fester Schwellwert darf eine restorative Phase blockieren**, wenn das Material den Eingriff braucht
   und die Phase den Defekt messbar reduziert — selbst wenn ein Proxy-Metrik-Score dabei sinkt.
2. **Checkpoint-Selektion**: Der Guard wählt immer das perceptuell *beste* Audio als Checkpoint
   (höchster gewichteter P1–P5-Score), nicht das *letzte nicht-gerollte*.
3. **Pipeline-Stopp nur bei echtem Schaden**: `should_stop` nur wenn die Notbremse-Schwelle NACH
   materialadaptiver Berechnung gerissen wird UND keine bessere Stärke gefunden wurde.
4. **Referenz-Paradoxon** (§2.44) ist immer zu berücksichtigen: Carrier-Repair-Phasen verändern das
   Signal *intentional* gegenüber dem beschädigten Checkpoint. Ein Metrik-Drop gegenüber dem
   beschädigten Referenzpunkt bedeutet nicht „Verschlechterung", sondern „das Signal entfernt sich
   vom Defekt" — genau das ist das Ziel.

**VERBOTEN (harter Anti-Pattern-Katalog):**

| Anti-Pattern | Warum falsch | Richtig |
|---|---|---|
| `if drift < -0.05: rollback` (feste Konstante) | Jeder Song braucht andere Toleranz | `compute_adaptive_drift_tolerance()` |
| `if regression > 0.020: retry` (feste Konstante) | Vinyl-Denoise ≠ CD-Kompressor | Material-/Defekt-adaptive Schwelle |
| 3 Rollbacks → Pipeline-Stopp (fest) | Vinyl→Tape→MP3 braucht >3 Carrier-Phasen | Adaptiv: `max(5, n_carrier_phases + 2)` |
| Guard blockt Phase → Phase wird übersprungen | Defekt bleibt im Audio | Stärke iterativ reduzieren bis Optimum |
| best_checkpoint = letztes nicht-gerolltes | Kann schlechter sein als Iteration 2/4 | Bestes perceptuelles Ergebnis über alle |
| Alle Goals gleichgewichtig in Drift-Check | tonal_center Drop nach Denoise ≠ Schaden | §2.29c Baseline-Capping + Phase-Exclusions |

> **Zusammenfassung für KI-Agenten**: Wenn du einen festen Schwellwert implementierst, der für Audio-Qualität
> entscheidet, ist das fast immer falsch. Berechne den Schwellwert aus dem konkreten Song-Kontext
> (Material, Era, Restorability, Defect-Severity). Feste Werte existieren nur als Notbremse für
> katastrophale Fälle — sie dürfen die Routine-Pipeline nie blockieren.

## Vintage Aesthetics

**SOFT_SATURATION** = BEWAHREN. **CLIPPING** = REPARIEREN.
1920–1940: Rolloff ≤ 7 kHz nicht erweitern, H2/H4 bewahren.

*Diese Richtlinien gelten für alle KI-Agenten (GitHub Copilot, Claude, GPT) die an Aurik 9 arbeiten.*
*Vollständige normative Spezifikation: `.github/specs/01–08`.*
*Stand: April 2026 — Aurik 9.11.0 — instructions_version 7.1*
