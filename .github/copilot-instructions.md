# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes,
> kontextbewusstes Musik- und Gesangs-Restaurations-, Reparatur- und
> Rekonstruktions-Denkersystem.* Stand: April 2026 — Version **9.11.14**
>
> **instructions_version: 7.6** — §0a DR/BW/Rauschtextur-Ceiling + §0d Carrier-Recovery-Referenzmodell + §1.2a/§4.7/§6.2b normiert 13.04.2026; §09.2 PMGG-Blend-Invariante + §2.54 Headroom-Scalar + MDEM Quiet-Zone normiert 23.04.2026; Wall-Time-Mismatch-Pegelexplosion + waerme-Proxy-Sättigung + MUSHRA-CCR-Referenz-Kontamination normiert 24.04.2026; §2.30b Per-Sample-Guard correct_arc() −36 dBFS + Musical-Goals-Kalibrierungsfehler (tonal_center KEY_SHIFT_PENALTY_DEFAULT / authentizitaet formant_threshold / phase_18 PMGG-CIG-Sync / adaptive_thresholds Material-Ceiling) normiert 25.04.2026
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
# > 0.15 = signifikante Carrier-Inversion → Referenz-Shift aktiv
# > 0.35 = massive Inversion (Shellac, Multi-Gen) → voller MERT-Referenz-Anker
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
3. **Maximal-umsetzbare Recovery statt Hardstop**: Bei fehlgeschlagenem End-Gate MUSS Aurik das
  bestmögliche sichere Ergebnis suchen (Strength-Reduktion, alternatives Checkpoint,
  material-adaptive Recovery-Kaskade). Ein früher Hardstop ohne Recovery-Suche ist unzulässig.
  Ergebnisstatus bleibt transparent (degraded/recovered), darf aber nicht als stiller Erfolg maskiert werden.
4. **Allgemeingültigkeit vor Einzelfallgewinn**: Eine Änderung, die einen Song verbessert, aber die
  mittlere Qualität über die Import-Matrix verschlechtert, ist normativ unzulässig.

**Rationale:** Aurik ist ein universelles Restaurationssystem. "Maximale Klangtreue" bedeutet
"maximale Treue pro Songklasse im gesamten Importraum", nicht "bestes Ergebnis für den aktuell geladenen Song".

## Architektur dieser Richtlinien

Diese Datei ist der **Slim Core** (~250 Zeilen) — wird in **jeder** Konversation geladen.
Detailwissen liegt in den **8 normativen Specs** unter `.github/specs/` — Single Source of Truth.

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
| Analogquelle in digitalem Dateicontainer supprimieren | `file_ext=.mp3` → alle Analog-Posteriors auf 0 → vinyl/reel_tape dauerhaft unerkannt — korrekte Kette `vinyl → reel_tape → mp3_low` kollabiert zu `mp3_low` (Datenmüll — falsche Phasen, falsche Ziele) | Fallback-Gate Pflicht: `rotation_strength ≥ 0.30 AND conf ≥ 0.20 → vinyl akzeptiert`. `file_ext` bestimmt **ausschließlich die letzte Kettenstufe**, NICHT die Quellanalyse (§2.46b) |
| reel_tape vs. cassette ohne Disambiguation | Universelle `wow_flutter`-Schwelle ohne Disc-Kontext → Studio-Bandmaschine wird als Kassette klassifiziert → falsche Phasen (phase_12 zu stark, phase_29-Profil falsch) | Studio-Pfad wenn `has_disc AND codec_contamination > 0.5`: Schwelle `max(0.010, 0.025×(1−0.55×cc))` (IEC 60386:1987); Disambiguation: `wow < 0.06 WRMS → reel_tape; wow ≥ 0.06 WRMS → cassette` (§2.46b) |
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
| Spectral-Tilt-Drift in ADDITIVE-Phasen | HF-Extension ohne Tilt-Check (phase_06, phase_39) — Ära-Charakter wird zerstört ohne Goal-Verstoß | `era_result.spectral_tilt` in `kwargs` prüfen; Post-Tilt via `_estimate_spectral_tilt_quick()`; Cap wenn Deviation > material_tolerance (±1.5–±3.0 dB/oct je Material) (Spec 04 §4.7) |
| Roughness/Sharpness Anstieg ungeprüft | DYNAMICS/ADDITIVE-Phasen (phase_35, phase_39) erhöhen psychoakustische Lästigkeit ohne Gate | `ArtifactFreedomGate._compute_roughness_zwicker()` + `_compute_sharpness_bismarck()` für `DYNAMICS/ADDITIVE/ENHANCEMENT`-Phasen; Penalty -0.05 bzw. -0.10 (§2.49c) |
| MERT als primäre Qualitätsmetrik | `MertPlugin.score()` direkt als HPI-Haupt-Koeffizient wenn VERSA verfügbar | VERSA primär; MERT als Proxy-Fallback wenn VERSA fehlschlägt; `metadata["mert_proxy_used"]` setzen (§2.44) |
| VERSA auf RESEARCH-Modus beschränkt | `use_versa_in_loop=deployment_mode == RESEARCH` im FeedbackChain-Aufruf | `use_versa_in_loop=True` — VERSA ist produktionsstabil und muss immer aktiv sein (§2.44 VERBOTEN: MERT darf nicht primary sein wenn VERSA verfügbar) |
| DR-Expansion ohne Ceiling | `phase_26` expandiert ungeprüft über Material-Limit | `_MATERIAL_DR_CEILING_DB` Dict respektieren (§6.2b): Vinyl ≤ 70 dB, Shellac ≤ 45 dB, Tape ≤ 68 dB, CD ≤ 96 dB |
| BW-Extension ohne Ceiling | Phase_06/07/23/39 erzeugen Frequenzinhalt über Material-BW-Limit | `_MATERIAL_BW_CEILING_HZ` Hard-Cap in UV3 `_post_additive_bw_guard()` (§6.2c): Shellac ≤ 8 kHz, Vinyl ≤ 16 kHz etc. |
| Rauschtextur-Check fehlt | Denoising erzeugt weiße/flache Rauschtextur statt Carrier-kohärentem Profil | `NoiseTextureCoherenceGuard` (§4.7): spektrale Form des Restrauschens vs. Trägerprofil; Kohärenz ≥ 0.80 in Restoration |
| Goals gegen degradierten Input am Pipeline-Ende | `MusicalGoalsChecker.measure_all(restored, sr, original=degraded_input)` | Bei `carrier_chain_recovery_ratio > 0.15`: End-Referenz = `best_carrier_checkpoint` (§0d Ebene 2, §1.2a) |
| carrier_chain_recovery_ratio fehlt | Metadata ohne Carrier-Recovery-Signal | UV3 Pflichtfeld `metadata["carrier_chain_recovery_ratio"]` — berechnet nach letzter Carrier-Phase (§0d) |
| ML-Inferenz ohne PLM-Active-Guard | `session.run()` / `model(input)` ohne `plm.set_active()` — Emergency-Eviction entlädt Plugin während aktiver Inferenz → Crash → OOM | `plm.set_active("model", True)` VOR Inferenz, `plm.set_active("model", False)` in `finally`-Block (§4.6b) |
| `_PHASE_REQUIRED_MODELS` unvollständig | Phase listet nur Primärmodell, nicht Fallback-Modelle — PLM evictiert Fallback-Modell bei `evict_for_phase()` | Alle ML-Modelle (primär + Fallback) in `_PHASE_REQUIRED_MODELS` listen; bidirektionale Sync mit `try_allocate()`-Aufrufen (§4.6c) |
| Pitch-Kaskade ohne RMVPE | FCPE → CREPE → PESTO → pYIN (RMVPE übersprungen) | FCPE → RMVPE → PESTO → pYIN: `get_rmvpe_plugin()` als Tier-2 in HPG `_estimate_f0_track`, `hybrid_wow_flutter._init_crepe`, `hybrid_speed_pitch_ml._init_crepe` (§4.4 — 30 % geringere Pitch-Fehlerrate bei Gesang, Wei et al. ICASSP 2023) |
| Lautheitsmessung ohne ISO 532-1 | `np.mean(audio**2)` oder LUFS-only nach Rumble/Multiband-Phasen | `compute_specific_loudness_zwicker(audio, sr)` → ΔN > 2.0 sone = FAIL, Dry/Wet-Rescue (§4.1b) |
| JND-blinde PMGG-Phase-Akzeptanz | Phase mit allen Deltas > 0 und < JND wird identisch zu signifikant positiver Phase behandelt | `JND_MIN_DELTA` Dict in `_run_with_retry()`: wenn alle Deltas ≥ 0 UND alle < JND → `sub_threshold`, kein Retry, `metadata["sub_threshold_phases"]` (§2.47b) |
| Uniforme Goal-Gewichtung | Alle 14 Goals gleich gewichtet via Minimax (`_max_regression` ohne Weights) | `estimate_goal_importance()` → Per-Song-Profil → `goal_weights` in PMGG/CIG/GPP/FC (§2.56) |
| §2.56 nur in Gates nutzen | `goal_weights` ausschließlich für PMGG/CIG verwenden; Phasen laufen mit statischer Strength/Wetness | Globale all-phase Kopplung in UV3 `_profiled_phase_call`: `_compute_harmonic_adaptation_scalar(...)` (advisory-only), wirkt auf implizite `strength` + wet/dry, explizite PMGG-Strength bleibt führend (§2.56a) |
| Phase-50 Spike-Detection ohne HF-Guard | `_repair_channel()` flaggt restaurierte analoge Harmoniken als Codec-Spikes und inpaintet sie (zerstört Vorphasen-Restaurierung) | `_hf_protected_bin_start = material_rolloff × 0.85 / bin_hz` für alle analogen Materialtypen; Bins ≥ Start aus Pass-1 ausschließen; Pass-2 (Frame-Energy) bleibt global aktiv (§2.57) |
| PMGG Goal-Scoring bei Passthrough-Audio | Phase gibt unverändert Audio zurück, trotzdem werden 3× CREPE/pYIN-Retries und StrictConflictDecay ausgelöst | `np.array_equal(input, output)` → kein Scoring, kein Retry, kein Decay; ~51 s Laufzeit-Einsparung bei confidence=0.0 (§2.58) |
| Phase-09 Stub-Interpolation | `_interpolate_hybrid()` ruft intern `_interpolate_linear()` auf — kein AR-Verhalten trotz Bezeichnung | Vollständige LPC/AR-Vorhersage: Vorwärts-AR aus Pre-Gap + Rückwärts-AR aus Post-Gap, linear übergeblendet; Pol-Stabilisierung via Einheitskreis-Spiegelung; 5 ms Boundary-Crossfade (Rabiner & Schafer 1978) (§2.57) |
| Phase-50 lineare Zeit-Interpolation | Dropout-Frames werden einmalig linear interpoliert — keine Nutzung der STFT-Redundanz | Iterative STFT-Konsistenz-Projektion (5 Iterationen, POCS-Schema): Initialisierung → ISTFT → STFT → undamaged Frames re-ankern → wiederholen (Siedenburg & Dörfler 2013) (§2.57) |
| CausalDefectReasoner einseitige Tabellen | Neue Ursache (z.B. `vocal_harshness`) nur in `CAUSE_TO_PHASES` eingetragen, nicht in `CAUSES`/`LIKELIHOOD_FNS` | `CAUSES` und `CAUSE_TO_PHASES` sind bidirektional konsistent: Einträge in `CAUSE_TO_PHASES` ohne korrespondierendes `CAUSES`-Feld sind dead code — Bayes-Loop iteriert ausschließlich `CAUSES` (§2.59) |
| QualityGate SNR/STFT vor Musical-Goal-Check | `check_dsp/check_ml` führt `_check_audio_array` (STFT/SNR) durch, bevor Musical-Goals-Failures ausgewertet sind | `_check_musical_goals()` zuerst; bei Failure sofort `return` — teure STFT-Analyse nur wenn Goals bestanden |
| TFS-Guard Hilbert vor Voiced-Gate | `tfs_preservation_guard.py` berechnet Hilbert-Phasenextraktion und Band-Filterung für alle 12 ERB-Bänder vor dem Voiced-Energy-Gate | Frame-Energie der Original-Bänder zuerst prüfen; Bänder mit < 3 Voiced-Frames überspringen vor `filtfilt` + Hilbert — Muster: teure Analytic-Transforms nach dem günstigsten Admissibility-Gate |
| AudioSR ohne Wall-Time-Budget | AudioSR-Zonen-Schleife läuft zeitlich unbegrenzt → Hänger bei extremen Songstrukturen | `_AUDIOSR_WALL_BUDGET_S = 900.0` (15 min) vor Zonen-Schleife; Zonen jenseits Budget als Passthrough (Original-Audio) abschließen |
| Stereo-Kanal-Slicing `a[0]` | `a[0]` wenn `a.ndim == 2` (Shape: samples×channels) gibt nur 2 Samples zurück (erste Zeitzeile), nicht Kanal 0 — Score-Funktionen liefern immer 0.0 → PlateauStop, Gate-Logik, Proxy-Metriken blind für alle Songs | `a[:, 0]` für Kanal-0; bei skalenunklarer Orientation erst `_normalize_audio()` verwenden, die (channels, samples) → (samples, channels) transponiert |
| ONNX Fixed-Shape-Input ohne Chunking | `session.run()` mit Audio-Länge ≠ `inp.shape[1]` (z. B. 4968577 statt 43844) → INVALID_ARGUMENT, DSP-Fallback, Pitch-Schätzung degradiert | Vor `session.run()`: `inp.shape[1]` lesen; `isinstance(dim, int) and dim > 0` → Chunking-Loop mit Zero-Padding für letzten Chunk; gilt für alle ONNX-Plugins mit variabel langem Audio-Input (§ml-plugin-SKILL) |
| PlateauStop mit festen Konstanten | `_PLATEAU_THRESHOLD = 0.005` und `_PLATEAU_DAMPEN = 0.40` als universelle Konstanten (§2.54-Verletzung): Shellac hat 0.002-Verbesserungen pro Phase → fälschlich gedämpft; CD hat 0.010-Sprünge → Plateau zu früh | `_compute_plateau_params(material_type)` → material-adaptive (threshold, dampen); Shellac 0.002/0.55, Reel-Tape 0.003/0.50, Cassette 0.004/0.45, MP3 0.008/0.40, CD 0.010/0.35; zusätzlich: `restorability < 40` → dampen_floor = 0.60 |
| O(n²)-Autokorrelation im DSP-Fallback | `np.correlate(signal, signal, mode="full")` für AR-Koeffizientenberechnung — bei langen Signalen (10 M+ Samples) multi-stündiger Hänger | `np.array([np.dot(s[:n-k], s[k:]) for k in range(AR_ORDER+1)])` — O(n·order); nur benötigte Lag-Werte berechnen |
| `np.corrcoef` auf nahezu-konstanten Signalen | `np.corrcoef(a, b)` bei near-constant (Stille, DC-Offset) → RuntimeWarning: invalid value in true_divide; mit `-W error` Testabbruch | Guarded correlation: `dot(a,b) / (||a||·||b|| + ε)` — NaN-safe, kein Warning |
| `scipy.signal.stft` boundary='reflect' | `boundary='reflect'` → `ValueError: Unknown boundary condition` in scipy < 1.12 | `boundary='even'` für reflect-ähnliches Verhalten — einzige universell unterstützte spiegelnde Randbedingung |
| `load_audio_file()` mit synchroner Carrier-Analyse | `load_audio_file()` ruft intern `analyze_carrier_forensics()` → `classify_medium()` auf vollem Audio (225s = 10 M+ Samples) → 6+ Minuten synchroner Block im BatchProcessingThread, Progress stuck bei 2 % | `load_audio_file(path, do_carrier_analysis=False)` in allen UI/Thread-Aufrufen; Carrier-Analyse läuft separat in `_carrier_bg`-Thread |
| QualityMode-Vergleich als roher String | `if quality_mode in ("restoration", "balanced")` statt Enum — überprüft literalen String; `QualityMode.QUALITY.value == "quality"` feuert nie → Gate deaktiviert | `if quality_mode in (QualityMode.RESTORATION, QualityMode.QUALITY, ...)` oder Enum-Value explizit mappen |
| ML-Budget-Größe als MB statt GB | `try_allocate("Plugin", 630)` wenn 630 MB gemeint — aber Einheit ist GB → `required 630000 MB` → alle Allokations-Checks schlagen für alle weiteren Plugins fehl | Einheitenpräzision in `try_allocate()`: Argument ist immer GB (float); `630 MB → 0.63` |
| Unit-Tests Budget-Logik ohne `is_system_thrashing`-Mock | `try_allocate()` prüft `is_system_thrashing()` vor Budget-Preflight; Tests ohne diesen Mock schlagen auf Hosts mit hoher Swap-Auslastung fehl | `monkeypatch(budget.is_system_thrashing, lambda: False)` in allen Unit-Tests die Budget-Logik testen — hostseitige Druckheuristiken isolieren |
| Tonträgerketten-Mapping inline duplizieren | `_MEDIUM_DATA`, `_CI_MEDIUM_DATA` oder gleichwertiger Dict lokal innerhalb von Methoden/Callbacks in `modern_window.py` definieren — bei neuem Medium vergisst man Kopie 2/3 → Divergenz, falscher Labeltext, kein Test-Alarm | Ausschließlich `_CARRIER_MEDIUM_DISPLAY` (Modul-Level-Konstante) referenzieren; HTML-Rendering über `_render_carrier_html(icon_stem, label)`; Kettenkombination über `_build_carrier_chain_html(chain_keys)` — beides in `Aurik910/ui/modern_window.py` definiert |
| `chain_info.get("transfer_chain")` als ersten Key versuchen | `kette.as_dict()` liefert **ausschließlich** den Key `"chain"` (nicht `"transfer_chain"`) — blindes Fallback-`get` verdeckt Tippfehler und übersteht Reviews ohne Fehler | `_chain_info.get("chain")` direkt; `"transfer_chain"` existiert in `KettenErgebnis.as_dict()` nicht (§UI-CARRIER-DISPLAY-INVARIANT) |
| `detected_medium_label.setText(...)` ohne `_carrier_bg_label`-Sync | Era-Badge-Block liest `self._carrier_bg_label` und schreibt `detected_medium_label` neu → wenn `_carrier_bg_label` ≠ aktuelle Ketten-HTML → Era-Badge-Update überschreibt die Kettenanzeige mit altem/leerem Content (Silent Data Loss) | Jedes `detected_medium_label.setText(html)` MUSS unmittelbar gefolgt von `self._carrier_bg_label = html` sein — keine Ausnahme |
| Kettenanzeige-Update bei len < 2 ohne Logging | `if len(chain_keys) < 2: return` — stilles Überspringen; kein Debug-Log; Anzeige bleibt bei Voranalyse-Wert obwohl TontraegerketteDenker geringere/andere Kette meldet | Immer `logger.debug("Kettenanzeige übersprungen – len=%d < 2 (chain=%s)", len(chain_keys), chain_keys)` wenn Guard feuert |
| Icon-HTML ohne Plaintext-Fallback | `f'<img src="file:///{path}"...'` ohne `if path is None: return label` — fehlendes Icon → kaputtes Bild-Tag im Label; besonders in Background-Threads ohne Exception-Handling fatal | `_render_carrier_html()` nutzen: prüft `_svg` → `_png` → `return label` wenn kein Icon gefunden; immer `except (OSError, TypeError, ValueError): return label` |
| MDEM Quiet-Zone mit fester Amplitude | `if _tail_rms < 0.003` / `MIN_LEVEL_LUFS = -60.0` — Vinyl/Shellac-Fadeout bei −40 dBFS wird als Musik-Frame klassifiziert → positiver Gain auf Rauschboden → Pegelexplosion bei 15 % Progress | Threshold immer in **dBFS**: `if _tail_rms_dbfs < -36.0` und `if lr < -36.0` → clamp G[k] ≤ 0 (kein positiver Boost im Quiet-Zone). Ausbreitungsregel: Quiet-Zone-Grenze = −36 dBFS, gilt für Frame-Level und Tail-Guard in `micro_dynamics_envelope_morphing.py` |
| `apply_musical_gain_envelope` mit `gate_dbfs=-50.0` | `gate_dbfs=-50.0` in jedem Aufruf von `apply_musical_gain_envelope` — Vinyl/Shellac-Oberflächenrauschen (~−40 dBFS) liegt oberhalb des Gates → erhält Makeup-Gain → Pegelexplosion in Intro/Outro/Fadeout. Bestätigt in `phase_05_rumble_filter` (2026-04-25): stille Bereiche zu Beginn und Ende des Songs werden ge-boosted | `apply_musical_gain_envelope(..., gate_dbfs=-36.0, ...)` — **immer −36 dBFS, niemals −50 dBFS** als Gain-Gate-Argument. Der RMS-Mess-Gate (`_rms_dbfs_gated`) darf −50 dBFS behalten — er ist ein anderer Schwellwert mit anderer Rolle (§2.45a-V) |
| Makeup-Gain-Guard in Hochpassfilter-Phasen | Per-Phase-RMS-Guard in subtraktiven HP-Filter-Phasen (phase_05 Rumble, phase_02 Hum) vergleicht `_rms_in_db_ref` (enthält Rumpel-/Hum-Energie sub-30 Hz) mit `_rms_out_db` (HP-gefiltert, Energie entfernt) → scheinbarer RMS-Drop → Guard feuert → `apply_musical_gain_envelope` boosted Fadeout/Intro-Frames → Pegelexplosion. Bestätigt `phase_05_rumble_filter` (2026-04-25): `-36 dBFS`-Fix reichte nicht — die Referenzmessung war falsch, nicht das Gate. Dies ist ein **logischer Widerspruch**: HP-Filter entfernt Energie absichtlich; Makeup-Gain kämpft dagegen → Endlosschleife aus Fixes | Kein per-Phase-Makeup-Gain-Guard in subtraktiven Filtertypen (HPF, LPF, Notch, Bandpass). Nur breitbandig-subtraktive Phasen (Denoise, Dereverb, Surface-Noise) dürfen per-Phase-Guards haben. HPF-Energieverlust = beabsichtigte Carrier-Inversion. UV3-Cumulative-Guard (§2.45a-IV) überwacht den Gesamtpegel auf Pipeline-Ebene — kein doppelter Guard in der Phase selbst |
| Gain-Morphing ohne Post-Smoothing-Quiet-Zone-Clamp | Pre-Smoothing Guard setzt stille/Fadeout-Segmente korrekt auf 0 dB — aber Savitzky-Golay (window=7, z. B. 17,5 s Reichweite bei HOP_S=2.5 s) verteilt positiven Gain aus Musik-Segmenten zurück → `correct_emotional_arc` boost­et denoised Fadeout → Pegelexplosion trotz korrektem MDEM. `np.interp` erzeugt zusätzlich positiven Übergangs-Boost zwischen Musik- und Stille-Segmenten | **Drei-Stufen-Invariante (§2.30b)**: (1) Pre-Smoothing Guard; (2) Smoother; **(3) Post-Smoothing Guard — MUSS erneut angewendet werden**; (4) `np.interp`; **(5) Per-Sample Quiet-Zone Guard auf interpoliertem Gain**. Gilt für ALLE Gain-Morphing-Funktionen: `morph()` in `micro_dynamics_envelope_morphing.py` UND `correct_arc()` in `emotional_arc_preservation.py`. Regressions-Test: `test_36_no_pegelexplosion_in_denoised_fadeout` — **Schwellwert zweistufig**: 5-s-Segment-Guards (Pre+Post-Smooth) = −42 dBFS + 6-dB-Differenz-Bedingung (Segmente enthalten Mischung aus Musik und Stille → −36 dBFS wäre aggressiv); **Per-Sample-Guard (Stufe 5) = −36 dBFS** — Vinyl/Shellac-Oberflächenrauschen liegt bei −35 bis −42 dBFS und passiert einen −42-dBFS-Guard → Pegelexplosion im Intro/Outro durch Arousal-Boost auf Rauschen (bestätigt Produktion 2026-04-25). Invariante: Per-Sample-Guard in `correct_arc()` MUSS −36 dBFS verwenden, nicht `_quiet_rms_thresh` (−42 dBFS) |
| PMGG fixer 60/40-Blend canonical/SGT | `_blended_thr = 0.60 * _cur_thr + 0.40 * float(_sgt_val)` global für alle Ziele und alle Materialien — Shellac `brillanz`: 0.60×0.78+0.40×0.51=0.71 obwohl physikalisches Ceiling 0.51 → 5 Retries bei 15 % Stärke → degradierte Restaurierung | **Delta-adaptiver Blend** (§09.2, §2.54): `delta = canonical − SGT`; `delta > 0.10` → SGT direkt (Ceiling-Fall); `delta > 0.04` → 40 % canonical + 60 % SGT; sonst 60/40. **Pre-Pipeline**: `_pmgg_ceiling_capped_targets = min(SGT, PhysicalCeiling)` VOR Phase-Loop berechnen und als `adaptive_goal_thresholds` an PMGG + Pipeline-Ende übergeben |
| Headroom-Scalar mit relativer Normierung | `hr_range = ceil − 0.30; hr_ratio = 1 − (curr − 0.30) / hr_range` — CD `brillanz=0.60, ceil=0.99`: scalar=0.57 obwohl 0.39 Headroom (Over-Dampening); Vinyl `waerme=0.65, ceil=0.85`: scalar=0.40 trotz 0.20 Headroom | **Absoluter Headroom** (§2.54 Psychoakustik): `headroom = ceil − curr; hr_ratio = min(1.0, headroom / 0.25)` — bei ≥ 0.25 Abstand zur Decke → Scalar=1.0 (volle Stärke); linear bis Scalar=0.40 an der Decke. Nur für additive/Enhancement-Familien in `_PHASE_INTERVENTION_CLASS`; Restorative/Pflicht-Phasen ausgenommen |
| Wall-Time-Referenz Mismatch im Phase-Budget | `time.monotonic()` als Accumulator-Zeitbasis, aber `start_phase()` liefert intern `time.perf_counter()` — beide geben nominell Sekunden zurück, aber ihre Epochen divergieren → Differenz bis 1 × 10¹² s → **alle** Phasen sofort als Wall-Time-Budget-Überschreitung klassifiziert → Pipeline wird komplett übersprungen → LUFS-Alignment sieht Δ bis 14 LU zwischen Original und unverändertem Audio → 12 dB Gain auf das degradierte Signal → **Pegelexplosion bei ~15 % Progress** (bestätigt in Produktion, 2026-04-23) | Ausschließlich `time.monotonic()` für **beide** Seiten des Wall-Time-Akkumulators: `_wall_phase_start = time.monotonic()` VOR dem Phase-Call; `_pipeline_non_exempt_elapsed_s += time.monotonic() - _wall_phase_start` danach. `start_phase()`-Rückgabewert DARF NICHT als Zeitreferenz für den Wall-Time-Akkumulator verwendet werden — er dient ausschließlich dem internen Phasen-Profiling |
| PMGG waerme Proxy-Sättigung durch falsche Normierung | `scores["waerme"] = np.clip(_e_low_mid / _e_upper_mid / 1.5, 0, 1)` — ungewichtetes FFT-Energieverhältnis E(200–800 Hz)/E(800–3000 Hz) liegt für warme Musik typischerweise bei 3–5 (Bass/untere Mitten dominant); Division durch 1.5 → Wert >> 1 → Clip auf 1.0 bei nahezu jeder Phase → PMGG permanent blind für waerme-Regressions durch Denoise/Dereverb/Derumble → `WaermeMetric._measure_absolute()` nutzt ISO 226:2003-Gewichtung (800–3000 Hz stärker gewichtet → reale Scores 0.70–0.90) → vollständige Kalibrierungsdivergenz → waerme fehlt am Pipeline-Ende trotz korrekter Vollmetrik (bestätigt in Produktion: before=1.0000/after=1.0000/delta=+0.0000 über alle Phasen, 2026-04-24) | Normierungskonstante so wählen, dass typisches warmes Musik-Ratio (ca. 3.0–4.0) einen Proxy-Score von 0.75–1.0 ergibt: `/ 4.0` statt `/ 1.5`. Kalibrierungsprüfung beim Implementieren: Warm-Signal (dominante 200–800 Hz) → Proxy 0.75–1.0; Neutral-Signal (ausgewogen) → 0.25–0.50; dann mit `WaermeMetric._measure_absolute()` abgleichen |
| MUSHRA-Referenz bei aktivem CCR Reference-Shift kontaminiert | `_mushra_ref_src = original_audio_for_goals` — wenn §0d CCR Reference-Shift aktiv ist (`carrier_chain_recovery_ratio > 0.05`), wurde `original_audio_for_goals` auf das `best_carrier_checkpoint` (Post-Carrier-Phase, z. B. post-phase_23) verschoben. MUSHRA misst dann finales Audio vs. Carrier-Checkpoint: das finale Audio entfernt sich intentional vom Checkpoint durch Enhancement-Phasen (FeedbackChain, ExzellenzDenker) → MUSHRA bewertet genau diese Verbesserung als Degradation → OQS=47.9 < anchor=57.6 (3.5-kHz-LP) trotz objektiv guter Restaurierung (bestätigt in Produktion 2026-04-24) | `_mushra_ref_src = audio if (original_audio_for_goals is not audio) else original_audio_for_goals` — bei aktivem CCR-Shift das ursprüngliche degradierte `audio` als MUSHRA-Referenz verwenden. Der LUFS-Unterschied zwischen `audio` und `restored_audio` wird durch `lufs_score` ([0,1]) in der MushraEvaluator-Gewichtungsmatrix bereits abgebildet und ist kein Ausschlusskriterium. Invariante: MUSHRA misst **immer** Qualität des restaurierten Audios relativ zum **degradierten Input**, nie relativ zu einem Zwischenstand der Pipeline |
| `TonalCenterMetric._KEY_SHIFT_PENALTY_DEFAULT = 0.0` | Default-Penalty=0.0 für unbekannte Pitch-Shift-Distanzen (> 3 Halbtöne) → `tonal_center = 0.000` nach phase_23 (Spectral Repair) / phase_06 (BW-Extension): spektrale Energie-Umverteilung verschiebt dominante Tonhöhe ≥ 2 Halbtöne ohne echten Tonartwechsel → PMGG lehnt jede dieser Phasen ab → Retries → Defekte bleiben (bestätigt Produktion 2026-04-25) | `_KEY_SHIFT_PENALTY_DEFAULT = 0.20`; Penalty-Dict `{0: 1.0, 1: 0.75, 2: 0.50, 3: 0.30}`; Bypass-Guard von `corr_score >= 0.85` auf `>= 0.70` senken — BW-Extension / Spectral Repair erhöht tonale Kohärenz ohne Tonartwechsel |
| `AuthentizitaetMetric._formant_threshold = max(500, ref*0.5)` | Bei mittlerem Spektral-Centroid 1200 Hz: Formant-Threshold = 600 Hz. BW-Extension (phase_06/07) hebt Centroid um +600–800 Hz → neuer Centroid 1800 Hz. Formant-Stability = max(0, 1 − |1800−1200|/600) = max(0, 0.0) = 0.0 → `authentizitaet = 0.0` nach jeder BW-Extension → PMGG-Rollback aller Carrier-Inversion-Phasen → falsche Restaurierung | `_formant_threshold = max(1200.0, mean_ref_centroid * 1.5)` — erlaubt ±150 % Centroid-Drift als korrekte Träger-Inversion; Phase-Centroid-Shift durch BW-Extension ist kein Authentizitätsverlust, sondern intentionale Carrier-Chain-Inversion (§2.46) |
| `phase_18_noise_gate` ohne PMGG/CIG-Exclusion für `artikulation` | Noise Gate schneidet Note-Attacks (Attack-Transienten) — das senkt den `artikulation`-Proxy-Score um 0.29 (katastrophal). Ohne Exclusion: PMGG löst Rollback aus, `consecutive_rollbacks` steigt, Phase-Skip → Rauschen verbleibt | `{"artikulation", "groove"}` in BEIDEN Tabellen eintragen: `PMGG.PHASE_GOAL_EXCLUSIONS["phase_18_noise_gate"]` UND `CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS["phase_18_noise_gate"]` — bidirektionale Sync-Pflicht (§2.55). Note-Attack-Unterdrückung durch Noise-Gate ist kein Artikulations-Verlust, sondern Betriebsdesign |
| `adaptive_thresholds` ohne Material-Ceiling in Export-Gate | PMGG-blended Thresholds können 0.90 für Vinyl-Natürlichkeit ergeben (canonical=0.90, SGT=0.88, blend=0.89). UV3 schreibt diese in `metadata["adaptive_goal_thresholds"]`. Export Gate liest adaptive_thresholds zuerst → lehnt physikalisch korrektes Vinyl-Restaurierungsergebnis (score=0.655) bei threshold=0.90 ab → statt Material-Floor 0.72 | `_ADAPTIVE_THR_MATERIAL_CEILING` Dict in UV3 nach PhysicalCeiling-Block: `min(adaptive_thr, ceiling_per_material_per_goal)`. Ceiling-Werte: Vinyl natuerlichkeit≤0.82, authentizitaet≤0.79, tonal_center≤0.84, Shellac≤0.68/0.65/0.70, Tape≤0.78, mp3_low≤0.76. Invariante: adaptive_thresholds darf nie das physikalische Material-Ceiling überschreiten |

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

Aurik unterstützt **AMD-GPU-Beschleunigung** als Mixed-Mode (Heavy-Plugins → GPU, DSP/Light → CPU).
Alle GPU-Operationen laufen über den kanonischen `ml_device_manager`-Singleton.

**Unterstützte Backends:**

| Plattform | Backend | Provider | Erkennung |
|---|---|---|---|
| **Linux** | ROCm 6.x | `ROCMExecutionProvider` (ONNX) + `torch.cuda` (PyTorch) | Automatisch via `torch.cuda.is_available()` |
| **Windows** | DirectML | `DmlExecutionProvider` (ONNX) + optional `torch-directml` (PyTorch) | Automatisch via `onnxruntime.get_available_providers()` |
| **Beide** | CPU-only | `CPUExecutionProvider` | Fallback wenn kein GPU-Backend erkannt |

**AMD-GPU-Architektur-Erkennung & Tier-System:**

| Tier | Architektur | VRAM | Verhalten |
|---|---|---|---|
| **Tier 1** | RDNA3 (RX 7000), RDNA2 (≥16 GB), CDNA (≥8 GB) | ≥16 GB | Alle Plugins GPU, fp16 auto, max_usage 85 % |
| **Tier 2** | RDNA2 (8–15 GB), RDNA1 (≥8 GB), CDNA (<8 GB) | 8–15 GB | Meiste Plugins GPU, fp16 auto, max_usage 80 % |
| **Tier 3** | RDNA2 (4–7 GB), RDNA1 (4–7 GB), GCN5 (≥8 GB) | 4–7 GB | Selektive GPU (AudioSR/AudioLDM2 ausgeschlossen), max_usage 70 % |
| **Tier 4** | GCN4, <4 GB VRAM | <4 GB | CPU-only empfohlen, VRAM-Budget zu klein |

**fp16-Auto-Aktivierung**: Auf ROCm erhalten fp16-eligible Plugins (BSRoFormer, MDXNet, DemucsV4, MPSENet, ResembleEnhance, PANNs, LaionCLAP, BanquetVinyl, DeepFilterNetV3, BigVGAN, MDX23C) automatisch fp16-Provider — halbiert VRAM, verdoppelt Throughput.

**VERBOTEN-Erweiterung:**

| Kategorie | Verboten | Richtig |
|---|---|---|
| GPU-Architektur | Feste VRAM-Ratio `0.85` für alle GPUs | Tier-adaptive Ratio via `_TIER_VRAM_PARAMS[gpu_tier]` |
| fp16-Aktivierung | Plugin ruft manuell `get_ort_providers_fp16()` | `get_ort_providers()` aktiviert fp16 automatisch wenn eligible+Tier erlaubt |
| GPU-Dispatch | `model.to("cuda")` / `CUDAExecutionProvider` direkt | `get_torch_device("PluginName")` / `get_ort_providers("PluginName")` |
| VRAM-Größe | Alle Plugins auf jeder GPU | Tier-basierte Exclusion für VRAM-hungrige Plugins (AudioSR, MERT-fairseq) |

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

**Integration**: PMGG (`weighted_reg = reg × weight`), CIG (gewichtete Drift), GoalPriorityProtocol (gewichtete Conflict-Resolution + Abort), FeedbackChain (gewichtete GPP-Prüfung), UV3 all-phase Kopplung über `_compute_harmonic_adaptation_scalar(...)` in `_profiled_phase_call` (advisory-only: explizite PMGG-Strength gewinnt).

### [RELEASE_MUST] §2.56a Global All-Phase Harmonic Adaptation (v9.11.12)

Alle 64 Phasen müssen harmonisch auf denselben Song-Kontext reagieren, ohne Einzelphasen-Hardcoding.

- Ort: `backend/core/unified_restorer_v3.py` (`_profiled_phase_call`)
- Pflicht-Funktion: `_compute_harmonic_adaptation_scalar(phase_id, phase_family, goal_weights, restorability_score, material_key)`
- Wirkung: multiplikative, bounded Anpassung auf implizite `strength` und `wet/dry`
- Bound: `harmonic_adaptation_scalar ∈ [0.72, 1.18]` (mit Pullback von Randwerten)
- Advisory-only: wenn `strength` explizit gesetzt wurde (PMGG/Team-Policy/Hard-Cap), darf §2.56a diesen Wert nicht überschreiben
- Fehlertoleranz: Ausnahme im Adaptionspfad darf Pipeline nicht blockieren (debug-log + neutraler Skalar 1.0)

**Normativer Zweck**: weniger False-Positive-Rollbacks bei gleichbleibend harten Safety-Gates (§2.44, §2.48, §2.49).

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

### [RELEASE_MUST] §2.46b Dateicontainer-Invariante — file_ext ≠ Quellanalyse (v9.11.14)

**Fundamentalregel**: Der Dateicontainer (`.mp3`, `.wav`, `.flac`) bestimmt **ausschließlich die letzte Stufe** der Tonträgerkette. Er darf physikalische Fingerabdruck-Evidenz **niemals vollständig unterdrücken** — sonst ist jede Restaurierung wertlos, weil falsche Phasen mit falschen Zielen auf falschem Material arbeiten.

**Zweistufige Erkennungsarchitektur (normativ)**:

1. **Bayesian-Scorer**: `file_ext ∈ DIGITAL_FILE_EXTS` → Analog-Posterior = 0 (korrekt, verhindert falsche Primärquelle)
2. **Physikalischer Fallback** (`_infer_analog_source_from_fingerprint`, Pflicht): prüft ob Fingerprint-Evidenz eine Analogquelle beweist

**Fallback-Gate Vinyl** (Pflicht, darf nicht entfernt werden):
```python
# Primär-Gate: conf >= codec-adaptiver Schwellwert (kann vinyl mit conf=0.25 zu Unrecht ablehnen)
# Fallback-Gate: rotation_strength >= 0.30 ist eindeutige Plattenspieler-Periodizität
_strong_physical_analog = (
    (_cand_conf >= _pa_conf_thresh and _feature_ok)
    or (_cand_conf >= 0.20 and fp.rotation_strength >= 0.30)  # Fallback — NICHT entfernen
)
```

**Studio-Tape-Pfad** (IEC 60386:1987, Pflicht bei `has_disc AND codec_contamination > 0.5`):
```python
# Professionelle Bandmaschine: 0.010–0.030 WRMS wow/flutter
# Alte Universalschwelle 0.20 ist für Studio-Reel blind!
_thresh_rt = max(0.010, 0.025 * (1.0 - 0.55 * _codec_contamination))
# Rotation-Guard entfernt: Vinyl-Drehzahl ist ERWARTET, kein Ausschlusskriterium
```

**Cassette vs. reel_tape Disambiguation** (IEC 60386:1987, normativ):
```python
if _has_cassette and _has_reel_tape:
    if fp.wow_flutter_index < 0.06:
        sources = [(m, c) for m, c in sources if m != "cassette"]   # Studio-Bandmaschine
    else:
        sources = [(m, c) for m, c in sources if m != "reel_tape"]  # Consumer-Kassette
```

**Produktions-Invariante** (Backend-Log 2026-04-21 — darf nie regressieren):
```python
# rotation=0.371, wow=0.034, codec_artifact=0.40, file_ext=".mp3"
result = MediumDetector().detect(audio, sr, file_ext=".mp3")
assert result.transfer_chain == ["vinyl", "reel_tape", "mp3_low"]
# Test: tests/unit/test_vinyl_tape_mp3_chain_detection.py::test_vinyl_reel_tape_mp3_full_chain_production_case
```

**VERBOTEN**: Jede Implementierung die bei `file_ext=.mp3 AND rotation_strength=0.371` das Einzelergebnis `mp3_low` zurückgibt. Das ist ein RELEASE_MUST-Verstoß — alle Phasen arbeiten dann auf dem falschen Material und das Ergebnis ist klanglich wertlos.

> Details: Spec 05 §6.7 Phase 1b — vollständige Schwellwerte, Kalibrierung, Referenzfall

### [RELEASE_MUST] §2.47 Adaptive-Intelligence-Prinzip (v9.10.123)

Jede Eingabe ist ein einzigartiges Musikstück. Das System passt sich **vor** der Verarbeitung an das konkrete Material an.

**Adaptions-Kaskade (9 Schritte, kanonische Reihenfolge):**
1. `MediumDetector.detect(audio, sr, file_ext=...)` → transfer_chain, primary_material (§6.7)
2. `EraClassifier.classify()` → decade, era_profile **+ ERB-Salience-Annotation** (v9.11.0)
3. `GenreClassifier.classify()` → genre_label, genre_profile
4. `RestorabilityEstimator.estimate()` → restorability_score, tier
5. `DefectScanner.scan()` → 46 DefectTypes × Severity × Locations

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
| CREPE Pitch-Track | RMVPE → pYIN (Mauch & Dixon 2014) | YIN (de Cheveigné & Kawahara 2002) — direkter pYIN-Fallback ohne RMVPE-Stufe |
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
6. `joy_runtime_index.components` MUSS folgende Advisory-Metriken enthalten (v9.11.14 Literature-Based):
    - `frisson_index` (0..1): musikalische Gänsehaut-Propensity aus Erwartungsbogen, Dynamik, Articulation, Raumtiefe (Blood & Zatorre 2001, Grewe 2007, Harrison & Loui 2014)
    - Alle Sub-Komponenten NaN/Inf-frei, clipped [0, 1]
  - **Mode-Policy**: Restoration = advisory-only (kein Audio-Impact); Studio 2026 = konservative bounded Mikro-Kopplung auf implizite Strength/Wet-Dry erlaubt
  - Keine Kopplung auf harte Gates (PMGG/CIG/AFG/HPI) und keine Überschreibung expliziter PMGG-Strength
  - UI zeigt "Gaensehaut X%" neben Freude/Ermüdung im Status und Ergebnis-Banner

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
    "cassette": ["phase_29_tape_hiss_reduction", "phase_12_wow_flutter_fix", "phase_06_frequency_restoration", "phase_24_dropout_repair", "phase_03_denoise"],
    "mp3_low":  ["phase_23_spectral_repair", "phase_03_denoise", "phase_50"],
    # … alle 15 Materialtypen (vollständig in backend/core/unified_restorer_v3.py)
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
*Stand: April 2026 — Aurik 9.11.14 — instructions_version 7.4*
