# Aurik 10 — Vollständige GEBOTEN-Tabelle

> **Normative Quelle** für verpflichtende Patterns.
> Ergänzt `.github/VERBOTEN.md` und `.github/copilot-instructions.md`.
> Generiert aus der Betriebssicherheits-Analyse vom 12. Juli 2026.

---

## Kategorie A: Kontrakt-Validierung

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G01** | Jedes Registrierungssystem (Plugin-Checks, Provider, Phase-Mapper) MUSS beim Modul-Import einen Selbsttest durchführen und Fehler als CRITICAL loggen | `PANNs._model_loaded` fehlte → 6 Monate unbemerkter DSP-Fallback | `ml_model_readiness._validate_all_checks()` |
| **G02** | Jedes ML-Plugin MUSS von `MLPluginBase` erben und `_model_loaded` als Property implementieren | Ohne Basisklasse kein Compile-Zeit-Schutz vor fehlenden Attributen | `backend/core/plugin_base.py` |
| **G03** | Jede Funktion/Variable, die in Log-Statements referenziert wird, MUSS importiert oder im Scope definiert sein | `phase_human_name` → NameError → Pipeline-Crash → Fallback-Pfad | `_execute_pipeline` L33303 |

## Kategorie B: Konfiguration & Kalibrierung

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G04** | ALLE hardware-abhängigen Schwellwerte MÜSSEN beim Modul-Import aus `psutil.virtual_memory().total` kalibriert werden | `_HEAVY_MODEL_PREEMPTIVE_AVAIL_GB_MAX = 6.0` galt für 8 GB wie 64 GB | `ml_memory_budget._calibrate_guard_thresholds()` |
| **G05** | Jeder Guard, der freien RAM prüft, MUSS modellgrößenabhängig sein: `safe_gb = f(model_size_gb)`, nicht `safe_gb = 6.0` | 1.1 GB Modell wurde mit derselben 6-GB-Schwelle blockiert wie 7 GB AudioSR | `_estimate_load_peak_factor()` |
| **G06** | Beim Startup MUSS ein vollständiges Systemprofil geloggt werden (RAM total/available, Swap total/%, CPU) | Post-mortem-Diagnose ohne Systemkontext ist wertlos | `_log_system_profile()` |

## Kategorie C: Resilience

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G07** | Jeder negative Cache MUSS eine TTL haben (empfohlen: 30 s) | `_FAILURE_CACHE` war permanent → Modell einmal als "nicht verfügbar" markiert, für immer blockiert | `_FAILURE_CACHE_TTL_S = 30.0` |
| **G08** | Jede Schleife, die einen externen ML-Schätzer aufruft, MUSS einen Circuit-Breaker haben (empfohlen: 3 konsekutive Fehler → deaktiviert) | Phase 12 rief PolyphonicSpeedCurveEstimator ~80× mit `consensus=0` auf | `_POLYPHONIC_CB_MAX_FAILURES = 3` |
| **G09** | Erfolgreiche Modell-Allokation MUSS den Readiness-Cache invalidieren | TTL ist passiv; aktive Invalidierung schließt die Lücke sofort | `try_allocate()` → `invalidate_ml_readiness()` |

## Kategorie D: Observability

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G10** | C-Level-Bibliotheken (ONNX, CUDA, ROCm) MÜSSEN beim frühestmöglichen Import auf ERROR-Log-Level gesetzt werden | 40+ MIOpen-Epsilon-Warnungen pro Modell-Ladung | `ort.set_default_logger_severity(3)` |
| **G11** | stderr MUSS beim Startup mit einem Deduplizierungs-Wrapper versehen werden | `os.environ` und `warnings.filterwarnings` greifen nicht bei C-Level-stderr | `stderr_dedup.install_stderr_dedup()` |
| **G12** | Singleton-Zugriffe MÜSSEN auf DEBUG loggen; nur die ERSTE Instantiierung auf INFO | "BasicPitch geladen" erschien bei jedem Chunk | `hybrid_wow_flutter._init_basicpitch()` |

## Kategorie E: Stereo & Kanalintegrität

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G13** | Globaler Interchannel-Lag MUSS vor Phase 1 erkannt und als VORVERARBEITUNG korrigiert werden – nicht nur pro Chunk | ~183 ms Versatz war von `LAG_PROBE 0B` bis `load_audio_file` durchgängig präsent | Run-Log: lag=-8900 → -8064 samples |
| **G14** | Nach Abschluss ALLER Phasen MUSS ein finaler Stereo-Lag-Check erfolgen und ggf. korrigieren | STCG korrigiert pro Chunk, aber keine Phase persistiert die Korrektur global | `load_audio_file: interchannel lag=-8064 samples before pipeline` |
| **G15** | Stereo-Kreuzkorrelation MUSS nach der Lag-Korrektur verifiziert werden (`corr > 0.7`) | `mean_corr=0.025` bei 183 ms Versatz → brauchbare Werte erst nach Korrektur | `ArtifactFreedomGate: ratio=0.00, mean_corr=0.025` |

## Kategorie F: Pipeline-Integrität

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G16** | Jede Phase, die chunk-weise arbeitet, MUSS ihre chunk-übergreifenden Zustände (Stereo-Lag, Gain, Phase) in einem persistenten State-Objekt speichern | STCG speichert pro-Chunk-Korrektur nicht global → nächster Chunk beginnt bei Null | Phase 12 STCG: `correcting R channel` pro Chunk, keine Persistenz |
| **G17** | Pipeline-Fallback-Pfade MÜSSEN dieselbe API-Signatur wie der Hauptpfad verwenden | `phase_human_name`-Fehler wurde gefangen, aber der gesamte Direkt-Pfad crasht → Fallback ist langsamer/schlechter | `restaurier_denker.py:567` |

---

## GEBOTEN-Linter-Referenz

Jede G-Regel kann durch einen Linter automatisiert geprüft werden:

| ID | Scope | Prüfung |
|----|-------|---------|
| G01 | `backend/core/*.py` mit `register_*`-Funktion | Modul MUSS `_validate_*()`-Aufruf nach Registrierung enthalten |
| G02 | `plugins/*plugin*.py` | `class *Plugin(MLPluginBase)` oder Linter-Warning |
| G03 | Alle `.py`-Dateien | `logger.info/logger.warning(f"...{name}...")` → `name` muss im Scope sein |
| G04 | `backend/core/*.py` | `= 6.0` / `= 3072` nach `GB`/`MB`-Kommentar → WARNING |
| G07 | `backend/core/*.py` | `_CACHE: dict` ohne `_TTL`- oder `_TIMESTAMPS`-Gegenstück → ERROR |
| G08 | `backend/core/phases/phase_*.py` | `for chunk in ...` + `external_call()` ohne `_MAX_CONSECUTIVE_FAILURES` → WARNING |
| G12 | Alle `.py` | `logger.info("... geladen")` in `__init__`/Getter ohne `_was_loaded`-Check → INFO |
| G13 | `backend/core/unified_restorer_v3.py` | Kein `_detect_and_correct_global_interchannel_lag()` vor `_execute_pipeline()` → ERROR |
| G14 | `backend/core/unified_restorer_v3.py` | Kein `_verify_stereo_integrity()` nach `_execute_pipeline()` → ERROR |

## Kategorie G: Architektur-Evolution (2026-07-12)

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G18** | Bei Refactoring eines Subsystems MÜSSEN ALLE Konsumenten der alten Signale auditiert werden. `grep` über alle Call-Sites VOR und NACH dem Umbau | Gate-Race: `defect_scan`-Flag wurde im neuen Unified-Pfad nur konditional gesetzt → Buttons gesperrt | `_try_signal_preanalysis_done` |
| **G19** | Nur EIN Progress-State-Objekt pro UI-Phase. Alle Callbacks (`scan_progress`, `progress_callback`, `emit_load_progress`) MÜSSEN in dasselbe Objekt schreiben | Balken-Wert und Schritt-Text waren entkoppelt → Balken stand bei 100% während Text lief | `_on_preanalysis_step` / `emit_load_progress` |
| **G20** | Jeder Timeout MUSS ein UI-Event emittieren. Degradierte Ergebnisse MÜSSEN als "Analyse nicht verfügbar" markiert werden | Era/Genre-Timeout → leere Prognose-Felder ohne Erklärung | `_SUBSTEP_TIMEOUT_S` |
| **G21** | Jeder Stateful-Prozess MUSS eine `_reset_*()`-Methode haben, die ALLE Guard-Flags zurücksetzt. Aufruf am Anfang jedes neuen Durchlaufs | `_preanalysis_finalized_for` blockte Wiederholung | `_finalize_preanalysis` Double-Fire-Guard |
| **G22** | Kein stiller `except: pass`. Jeder except-Block MUSS loggen (`logger.debug` mindestens) ODER einen Kommentar enthalten, warum Stille korrekt ist | PANNs 6 Monate DSP-Fallback, `phase_human_name`-Crash im Fallback-Pfad | Diverse |

## Kategorie H: GPU/Threading-Architektur (2026-07-12)

| ID | Gebot | Begründung | Fundstelle |
|----|-------|-----------|------------|
| **G23** | GPU-Modelle (PyTorch, ONNX mit CUDA/ROCm) dürfen NUR im Haupt-Thread oder in Threads geladen werden, die explizit via `ml_device_manager` GPU-initialisiert wurden. `ThreadPoolExecutor`-Worker haben KEINEN GPU-Kontext | CLAP-Loading im Pool-Thread: ROCm-Neuinitialisierung → 200+s. Restorability im Pool-Thread: 1s (weil CPU-only) | `_run_clap_chain` → Timeout |
| **G24** | Jede Architektur-Änderung, die Threading-Modelle verschiebt, MUSS einen Integrationstest haben, der den VOLLSTÄNDIGEN Flow (nicht nur gemockte Komponenten) durchläuft | Era-Timeout wurde erst nach 15+ Runs sichtbar. Unit-Tests mit Mocks liefen alle durch | `run_pre_analysis` → Era async |
| **G25** | Das Threading-Modell MUSS als Architektur-Diagramm dokumentiert sein: Welcher Thread hat GPU-Zugriff? Welche Threads teilen welche Singletons? Welche Threads dispatchen zur GUI? | 4× Threading-Refactoring (sequentiell → parallel → clap_chain → async) weil das Modell nicht explizit war | `_pre_analysis_bg`, Pool, Daemon |
