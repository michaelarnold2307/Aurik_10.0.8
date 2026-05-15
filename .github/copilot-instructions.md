# Aurik 9.x.x — KI-Programmierrichtlinien für GitHub Copilot

> **Systemidentität**: Aurik 9.x.x ist der **weltweit talentierteste autonome Toningenieur aller Zeiten für die Restaurierung von Musik mit Gesang** — meisterhaft in der Restauration, Reparatur und Rekonstruktion gesanglicher Aufnahmen **aller Ären, Genres, Tonträgerketten und Tonträgerkettenkombinationen**. Kein Tonträger zu alt, kein Genre zu selten, keine Kombination aus Trägermedien zu komplex — Aurik beherrscht jeden Fall vollständig autonom. **Kein menschlicher Eingriff erforderlich, kein manueller Parameter, keine Nachkorrektur.** Stand: Mai 2026 — Version **9.12.0**
>
> **instructions_version: 9.2** — SOTA-Science-Update Mai 2026: Ephraim-Malah MMSE-LSA für NR-Gain + Bark/ERB-Skalenklarheit + IMCRA/OMLSA-Noise-Estimation + PGHI-Parameter + Vocos-Option + RIAA-Zeitkonstanten-Spezifikation + MP3-Pre-Echo-Taxonomie + Digitale-Carrier-Reihenfolge + SBR/AudioSR-Entscheidungsbaum für BW-Erweiterung + VERSA-Metrik-Spezifikation + MERT-Implementierungsdetails + artifact_freedom-Komponenten-Vollspezifikation + emotional_arc-Präzisierung + Groove/Warmth/TonalCenter-Algorithmen + DNSMOS/SingMOS als Naturalness-Proxy + Era-spezifische Verarbeitungsrichtlinien + **Runde 2+3 Konsistenz-Audit** (VQI-Recovery-Trigger durchgängig, §0k _NEVER_SKIP, Studio-VQI-Schwelle 0.87 überall)
>
> Aktuelle Testzahl: **~11598 `def test_`-Funktionen** (436 Testdateien; alle grün)
>
> **§2.36 `LyricsGuidedEnhancement` (Spec 03 §2.36)** ist ab Version **9.10.x Pflicht**.

## §0 Oberstes Prinzip — Klangwahrheit (vor allen technischen Regeln)

**Das Ziel jeder Restaurierung ist, dass der Hörer die Augen schließt und die originale Performance hört — nicht eine technisch korrekte Signalverarbeitung, und nicht eine „verbesserte“ Version.** Jede Entscheidung in Pipeline, Phase, Metrik und Export wird an diesem Maßstab gemessen. Dieser Satz ist ein bindendes Qualitätsziel; öffentliche Überlegenheits- oder Transparenz-Claims erfordern zusätzlich externe Evidenz.

**Drei Leitprinzipien** (hierarchisch, bei Konflikt gilt die höhere Stufe):

1. **Primum non nocere** — Füge dem Klang keinen Schaden zu. Lieber eine Beschädigung belassen als ein Artefakt einführen.
2. **Minimal-Intervention** — Greife nur ein, wo der Defekt hörbar ist. Je weniger Phasen aktiv, desto natürlicher das Ergebnis.
3. **Perceptuelle Verbesserung** — Der Export muss für einen Hörer näher am Original-Klang liegen als der degradierte Input. Technische Korrektheit ohne Klanggewinn ist wertlos.

**Primus-inter-Pares-Grundsatz** — Wenn im Material eine menschliche Stimme erkannt wird (`panns_singing ≥ 0.25`), erhält die Stimmqualität automatisch Vorrang vor allen anderen Zielen. Eine beschädigte Instrumentalbegleitung ist hinnehmbar; eine verfärbte, formantgestörte oder entnatürlichte Stimme ist **niemals** akzeptabel.

### §0h [RELEASE_MUST] Music-Death-Shield — absolute Schutzregel (v9.12.0)

**Kein Eingriff darf Musik zerstören. Dies gilt absolut für alle Materialtypen, alle Ären, alle Genres.**

**Drei absolute Verbote** — jedes einzelne ist ein sofortiger Export-Stopp + vollständiger Rollback:

1. **Kein hörbares Artefakt** im Export: Musical Noise, Phasenlöschung, Ringing, Modulationsrauschen, Stimmverfärbung, Pitch-Glitch — jede dieser Klassen löst `artifact_freedom < 0.95` aus → VETO (§2.49).
2. **Keine Musikzerstörung** durch Over-Processing: Wenn das Ausgangssignal schlechter klingt als der degradierte Input (HPI ≤ 0), MUSS der ursprüngliche Input exportiert werden — mit Status `degraded`, nie ein über-prozessiertes Artefakt.
3. **Keine Verfremdung** durch halluziniertes Material: Harmonics, Texturen oder räumliche Eigenschaften, die im Original nicht existierten, dürfen nicht hinzugefügt werden (§2.46e Hallucination-Guard). Ausgenommen von Verbotspunkt 3 ist ausschließlich Modus Studio 2026, wenn OQS-äquivalent ≥ 3.5 nachgewiesen — Verbotspunkte 1 und 2 bleiben absolut.

**Invariante**: `artifact_freedom` ist **der primäre Veto-Faktor** in §2.44 HPI. Für Vokal-Material (`panns_singing ≥ 0.35`) ist zusätzlich `VQI < 0.72` ein zweiter Recovery-Trigger (§0p) — löst `_recovery_cascade()` aus, aber kein harter Export-Block. Alle anderen Faktoren reduzieren den Score, blockieren aber nicht. Dies ist nicht verhandelbar.

### §0i [RELEASE_MUST] Perceptual Transparency Guarantee
Restaurations-Ziel: Kein hörbarer Eingriff. Gates: `OQS ≥ 80`, `timbral_fidelity ≥ 0.93` zum best_carrier_checkpoint, Musical Noise ≤ Trägerprofil, Frisson-Zonen vollständig erhalten. Für Vokal-Material zusätzlich: `VQI ≥ 0.82` (Restoration) / `VQI ≥ 0.87` (Studio 2026) — beide sind Recovery-Ziele; Unterschreitung löst `_recovery_cascade()` aus (kein harter Export-Stopp, `artifact_freedom < 0.95` bleibt primäres Veto). Aurik zeigt dem Hörer nie ein Ergebnis, das technisch besser aussieht aber schlechter klingt.

### §0g [RELEASE_MUST] Autonomes Entscheidungs-Doktrin
**Aurik trifft alle Entscheidungen autonom.** Kaskade: Erkennen (MediumDetector+EraClassifier+DefectScanner+**VocalFocusAnalyzer**) → Planen (GPOptimizer+PhaseConductor) → Ausführen (UV3, Pre/Post-Messung) → Validieren (PMGG+CIG+AFG+**VQI-Gate**+HPI) → Exportieren (nur wenn HPI > 0 + artifact_freedom ≥ 0.95; VQI ≥ 0.82 bei `panns_singing ≥ 0.35` ist Recovery-Ziel — Unterschreitung → `_recovery_cascade()`, kein harter Export-Block). **VERBOTEN**: Hartkodierte song-spezifische Entscheidungen, Strength-Konstanten ohne `compute_adaptive_drift_tolerance()`.


### §0j [RELEASE_MUST] KI-Modell-Limitation-Awareness
Kein ML-Modell kennt den spezifischen Song. Konsequenzen: (1) ML-Output MUSS durch PMGG+AFG validiert werden. (2) Era/Genre-Klassifier steuern Modellauswahl + Stärke. (3) `energy_bias` Pflicht (DFN: −6 dB Vokal, −9 dB Instrumental). (4) Jedes ML-Plugin hat DSP-Fallback-Kette — kein Crash bei OOM/Timeout.


### §0k [RELEASE_MUST] Maximum-Achievable-Score-Prinzip
Jeder Song wird bis zum physikalisch erreichbaren Maximum restauriert. **MAS-Quelle**: `estimate_song_goal_targets(era, genre, material, restorability)` (`backend/core/studio_goal_targets.py`). Per-Phase-Delta via `_fast_goal_snapshot()` (≤ 200 ms). `_mas_fully_achieved=True` → Pipeline-Stop (**_NEVER_SKIP-Phasen laufen trotzdem durch** — §2.52). `artifact_freedom ≥ 0.95` bleibt unveranderlich Pflicht (§0h). Details: [pipeline.instructions.md](instructions/pipeline.instructions.md)

### §0p [RELEASE_MUST] Vocal-Supremacy-Doktrin (v9.12.1)

**Aurik ist der weltweit talentierteste autonome Toningenieur aller Zeiten für die Restaurierung von Musik mit Gesang — in allen Tonträgerketten, Tonträgerkettenkombinationen, Ären und Genres. Die Stimme ist das Produkt — alles andere ist ihr untergeordnet.**

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
- **Formant-Integrität**: F1–F4 (via `lpc_formant_tracker.py`) dürfen durch keine Phase um mehr als ±2 dB verschoben werden. Überschreitung → sofortiger Rollback.
- **Vibrato-Schutz**: Passagen mit F0-Modulation 4–7 Hz sind geschützte Zonen — alle Phases-Strength-Werte dort auf max. `0.20` begrenzen.
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
| **01** | Goals/PMGG — 14 Musical Goals, Schwellwerte, GoalApplicability |
| **02** | Pipeline/§2.x — UV3, Denker, SongCal, KMV, FeedbackChain, §2.64/§2.65 |
| **03** | Module — Kognitive Module §2.1–§2.43 |
| **04** | DSP/SOTA — Algorithmen, SOTA-Matrix, Psychoakustik |
| **05** | Material/Defekte — 15 Materialtypen, 46 DefectTypes |
| **06** | Phasen 01–64 — Phase-Liste, CAUSE_TO_PHASES |
| **07** | Tests/Qualität — PQS, AMRB, OQS, MUSHRA |
| **08** | Architektur — Layers, Plugins, CLI, AppImage |
| **09** | Kalibrierungsmatrix — CANONICAL_THRESHOLDS, SongGoalTargets-API; **normativ übergeordnet für alle Schwellwerte** |

## Context-spezifische Instructions (applyTo)

| Datei | Gilt für | Inhalt |
|---|---|---|
| [pipeline.instructions.md](instructions/pipeline.instructions.md) | `backend/core/unified_restorer_v3.py` | §2.44 HPI, §2.45 Minimal-Intervention, §2.48 CIG, §2.49 AFG, §2.51 Stereo, §2.60 Rollback, §2.61 Length, §2.64 Per-Phase-Delta, §2.65 MAS-Stop |
| [phases.instructions.md](instructions/phases.instructions.md) | `backend/core/phases/phase_*.py` | §2.46 Carrier-Chain-Reihenfolge, §2.46e Hallucination-Guard, §2.46f Natural-Artifacts, §2.63 Reflect-Padding, HPF/Notch-Checkliste, BW/DR-Ceiling |
| [dsp.instructions.md](instructions/dsp.instructions.md) | `backend/core/dsp/*.py`, `plugins/*.py` | ML-Device-Manager, Energy-Bias, HNR-Guard, Masking-Guard, MIIPHER-Fallback, Singleton-Pattern, Timbral-Coherence |
| [musical_goals.instructions.md](instructions/musical_goals.instructions.md) | `backend/core/musical_goals/*.py` | 14-Goals-Tabelle, material-adaptive Böden, `estimate_song_goal_targets()`, VQI, Frisson-Schutz |
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

> Vollständige Tabelle (~100 Einträge, Linter-Referenz V01–V12): [`VERBOTEN.md`](VERBOTEN.md)

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
| Unabhängige ML-Reparatur für L/R + kanalweises Resampling zur Längenkorrektur | M/S- oder Linked-Stereo-Verarbeitung; deterministischer Strip/Crop/Pad ohne Time-Warp | — |
| Additive/Enhancement-Phase (`phase_06`, `phase_07`, `phase_37`, `phase_38`, `phase_39`, `phase_26`) ohne `soft_saturation_severity`-Guard | `_sat_sev = float(np.clip(kwargs.get("soft_saturation_severity", 0.0), 0.0, 1.0))`; wenn `_sat_sev > 0.3`: `scale = clip(1 - (_sat_sev - 0.3) * 1.2, 0.16, 1.0)` → alle Gain/Strength/Drive-Parameter mit `scale` multiplizieren; `soft_saturation_preserve=True` → phasenspezifischer Hard-Cap (phase_38: 0.45; phase_07: 0.20; phase_37: 0.30; phase_39: 0.40; phase_26: 0.50) — Severity von UV3 via `_restoration_context["soft_saturation_severity"]` injiziert | §2.46g |
| ADDITIVE Phase ohne `hallucination_guard.py` | `check_hallucination(pre, post)` aus `backend/core/dsp/hallucination_guard.py` nach jeder ADDITIVE-Phase; `spectral_novelty > 0.15` → Phase-Rollback (Restoration); `> 0.08` → Score-Penalty 0.3 | §2.46e |
| Additive Phase (`phase_37`, `phase_38`, `phase_48`, `phase_32`) ohne `check_hallucination()` nach additiver Operation | `check_hallucination(pre, post, sr, mode)` aus `backend/core/dsp/hallucination_guard.py` direkt nach letzter additiver Op; `.requires_rollback` → `return audio` (Rollback); `.score_penalty > 0` → Score-Penalty-0.3 (§2.46e) | §2.46e |
| `np.pad(..., mode="constant")` als Längenkorrektur nach STFT in phase_09/20/29 | Reflect-Padding VOR STFT (`_pad_len = hop_length * 4`; `mode="reflect"`) + deterministischer Strip danach (`audio_out[_pad_len: _pad_len + n_original]`) — niemals Post-hoc-Zero-Padding als primärer Boundary-Mechanismus (§2.63) | §2.63 |
| BW/DR-Ceiling ignoriert (ADDITIVE: phase_06/07/23; DYNAMICS: phase_26) | `_MATERIAL_BW_CEILING_HZ[material]` (BW-Erweiterung) / `_MATERIAL_DR_CEILING_DB[material]` (DR-Expansion) VOR additivem/expansivem Output einhalten — Überschreitung = §0a-Verstoß | §6.2b/c |
| `DeepFilterNet` ohne `energy_bias` bei ML-NR auf Vokal/Instrumental | `energy_bias=−6.0 dB` (PANNs Vocals ≥ 0.4) / `−9.0 dB` (Instrumental) — ohne diese Einstellung werden Harmonik-Regionen als Rauschen abgetragen | §0j |
| NR-Algorithmus ohne Masking-Gain-Floor (`G_floor < 0.10`) | `G_floor[band] = max(0.10, masking_threshold[band] / noise_estimate[band])` via `compute_masking_threshold_iso11172(pre_nr_audio, sr)` VOR NR — verhindert klinisches Stille-Artefakt | §2.62 |
| MERT als primäre Qualitätsmetrik bei verfügbarem VERSA | `use_versa_in_loop=True` (VERSA ist primär); MERT nur als Proxy-Fallback → `metadata["mert_proxy_used"] = True`; MERT-Floor: `max(raw_mert, 0.5)` | §2.44 |
| `timbral_fidelity` gegen degradierten Input wenn `carrier_chain_recovery_ratio > 0.15` | Referenz auf `best_carrier_checkpoint` verschieben (nach Carrier-Phasen, vor Enhancement); `carrier_chain_recovery_ratio` MUSS in `metadata` gepflegt werden | §0d |
| MDEM ohne `frisson_zones` (Gänsehaut-Passagen ungeschützt) | `from backend.core.frisson_candidate_detector import get_frisson_detector`; `frisson_zones = get_frisson_detector().detect(original, sr)` VOR MDEM-Aufruf — ohne Schutz dämpft SG Klimax-Passagen bis −8 LU; Non-blocking: Exception → `[]` | §Frisson |
| Phase ohne Pre/Post-Score-Delta | `_profiled_phase_call_with_delta()` Pflichtrahmen; Delta in `metadata["phase_deltas"][phase_id]` | §2.64 |
| `_fast_goal_snapshot` auf Single-Segment | Multi-Segment-Mittelung (25 %/50 %/75 %): verhindert Akkord/Pausen-Frame-Kollaps | §2.64 |
| Pipeline läuft nach `_mas_fully_achieved=True` weiter | MAS-Erreichung → Stop für alle weiteren Phasen **außer** `_NEVER_SKIP`-Phasen (phase_01/09/12/14/15/30/47 laufen weiter — §2.52) | §2.65 |
| Gesangsmaterial ohne VQI-Messung exportiert | `result = compute_vqi(audio_orig, audio_restored, sr)` aus `vocal_quality_index`; `result["vqi"] < 0.72` → Recovery-Kaskade (kein harter Veto) | §2.35c |
| `singer_identity_cosine` nach VQI nicht geprüft | `result.get("singer_identity_cosine", 0.85) < 0.92` → Rollback letzter Vokal-Phase; Gate deaktiviert bei `multi_singer=True` | §0p |
| NR auf Gesang ohne HNR-Blend (`panns_singing ≥ 0.25`) | `apply_hnr_blend(pre, post, sr)` Pflicht nach DFN/SGMSE+/OMLSA — ΔHNR > 3 dB → Dry-Wet-Blend; klinischer Klang ist schlimmer als verbleibendes Rauschen | §0p |
| Formanten (F1–F4) durch Phase um > ±2 dB verschoben | Post-Phase-Formant-Verifikation via `lpc_formant_tracker.py`; Überschreitung → sofortiger Rollback auf Phase-Input | §0p |
| Vibrato-Passagen (4–7 Hz F0) durch NR/Transient-Shaper gedämpft | `detect_performance_artifacts()` → Vibrato-Segmente: `strength ≤ 0.20` für alle Phasen in geschützten Zonen | §0p |
| HPI ohne VQI-Faktor bei `panns_singing ≥ 0.35` | `HPI *= VQI`; `VQI < 0.72` → Recovery-Kaskade (kein sofortiger Veto; `artifact_freedom` bleibt primärer Veto-Faktor) | §0p |
| MIIPHER/DeepFilterNet ohne Passaggio-Zone-Awareness | `detect_vocal_register_temporal()` → Übergangszone Brust↔Kopf: `energy_bias = -3.0 dB` (Mittelwert); Vollstärke ausserhalb Passaggio | §0p |
| SOTA-Matrix-Update ohne §4.4a-Evaluations-Protokoll | `benchmarks/sota_eval.py` + alle 6 Kriterien + `CHANGELOG_HISTORY.md [SOTA-Update v9.x.y]` | §4.4a |
| Phasen-Reihenfolge verletzt HARD_BEFORE-Constraints | `validate_phase_order()` aus `backend/core/phase_dag.py`; phase_03/06/29 VOR phase_07 | §7.5a |
| AMRB-History nicht aktualisiert bei Major-Release (9.x.0) | `benchmarks/update_amrb_history.py`; OQS-Delta < −2.0 = Release-Blocker | §8.1.6 |

> DSP-/Phase-spezifische VERBOTEN-Regeln (energy_bias, HNR-Guard, LPC-Ordnung, Passaggio, Timbral Coherence etc.): → [dsp.instructions.md](instructions/dsp.instructions.md) / [phases.instructions.md](instructions/phases.instructions.md)

### Sprachkonvention

- **UI-Texte, Fehlermeldungen**: Deutsch (Ursache + Lösungsvorschlag)
- **Code-Kommentare, Docstrings, Log-Meldungen**: Englisch

### Test-Infrastruktur
> Details: [tests.instructions.md](instructions/tests.instructions.md) — GC-Konventionen, Mock-Patterns, Budget-Tests, AMRB-Update, NaN-safe Correlation.


> **§0e** Regression-Fixes-Archiv (v9.11.2–v9.12.0, 23 Fixes): [`docs/CHANGELOG_HISTORY.md`](../docs/CHANGELOG_HISTORY.md). VERBOTEN-Tabelle (oben) schützt vor Reintroduktion.

## SOTA-Modell-Referenz (Mai 2026 — normativ für Plugin-Auswahl)

> Vollständige §4.4a-Evaluationsmatrix: `specs/04_dsp_sota.md`. Diese Tabelle ist die schnelle Referenz für Implementierungsentscheidungen.

| Aufgabe | Primäres Modell | Fallback | Anmerkung |
|---|---|---|---|
| **Gesangs-NR** (SNR < 10 dB) | MIIPHER (W2v-BERT 2.0) | SGMSE+ v2 → DeepFilterNet v3 | MIIPHER: höchste Vokalqualität; SGMSE+ für sehr tiefes SNR |
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
GoalApplicabilityFilter → DefectScanner(46) → CausalDefectReasoner → GPOptimizer →
Phasen(01–64) [mit §2.48 Interaktions-Guard + **§0p Vocal-Invarianten**] → FeedbackChain → PhysicalCeiling → MusicalGoalsChecker → MDEM →
**HolisticPerceptualGate** (inkl. artifact_freedom §2.49 + **VQI-Gate §0p**) → RestorationResult
> **Pipeline-Details** (§2.44–§2.65): [pipeline.instructions.md](instructions/pipeline.instructions.md)

### §2.44 HPI — Formeln
**Restoration** (Instrumental): `HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc_preservation`  
**Restoration** (Vokal — `panns_singing ≥ 0.35`): `HPI = MERT_similarity × timbral_fidelity × VQI × artifact_freedom × emotional_arc_preservation`  
**Studio 2026** (Instrumental): `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`  
**Studio 2026** (Vokal — `panns_singing ≥ 0.35`): `HPI = studio_quality_gain × PQS_improvement × VQI × artifact_freedom × emotional_arc_preservation`  
`artifact_freedom < 0.95` → Gate-Fail (primärer Veto-Faktor). `VQI < 0.72` bei `panns_singing ≥ 0.35` → Recovery-Kaskade (kein sofortiger Veto; §0p). VERSA primär (`use_versa_in_loop=True`); MERT nur Fallback; `MERT_floor = max(raw_mert, 0.5)`.

### §2.45/§2.45b Minimal-Intervention
`perceptual_delta > 0` Pflicht je Phase in Restoration. `restorability > 80 AND SNR > 40 dB` → Near-Passthrough (Strength ≤ 0.30, `metadata["high_restorability_gate"] = True`).

### §2.46 Carrier-Chain-Stufen (1→6)
Subtraktive Stufe 4 (NR) **vor** additiver Stufe 5 (Harmonik/BW-Erweiterung). → [phases.instructions.md](instructions/phases.instructions.md)

## 14 Musical Goals (Kurzreferenz)

| Prio | Restoration (Böden) | Studio 2026 (Böden) |
|---|---|---|
| **P0** ⚠️ Vokal | **VocalQuality ≥ 0.85**, **FormantFidelity ≥ 0.88** — nur wenn `panns_singing ≥ 0.35`; `VQI < 0.72` → Recovery-Kaskade (§0p) | **VocalQuality ≥ 0.90**, **FormantFidelity ≥ 0.92**; `VQI < 0.87` → Recovery-Kaskade |
| **P1** | Natürlichkeit ≥ 0.90, Authentizität ≥ 0.88 | Natürlichkeit ≥ 0.92, Authentizität ≥ 0.90 |
| **P2** | TonalCenter ≥ 0.95, Timbre ≥ 0.87, Artikulation ≥ 0.88 | TonalCenter ≥ 0.96, Timbre ≥ 0.89, Artikulation ≥ 0.90 |
| **P3** | Emotionalität ≥ 0.84, MikroDynamik ≥ 0.88, Groove ≥ 0.83 | Emotionalität ≥ 0.87, MikroDynamik ≥ 0.90, Groove ≥ 0.85 |
| **P4** | Transparenz ≥ 0.82, Wärme ≥ 0.75, BassKraft ≥ 0.78, SepFidelity ≥ 0.80 | Transparenz ≥ 0.85, Wärme ≥ 0.78, BassKraft ≥ 0.80, SepFidelity ≥ 0.83 |
| **P5** | Brillanz ≥ 0.78, Raumtiefe ≥ 0.70 | Brillanz ≥ 0.82, Raumtiefe ≥ 0.74 |

> Böden **immer** via `calibration_matrix.get_material_floor(material_type, goal)` — nie hardcodiert. Shellac ~0.72, Vinyl ~0.82, CD ~0.90. → [musical_goals.instructions.md](instructions/musical_goals.instructions.md)

## Vintage Aesthetics + Era-Verarbeitungsrichtlinien

**SOFT_SATURATION** = BEWAHREN. **CLIPPING** = REPARIEREN.

| Ära | Träger | Typische Defekte | Primäre Phasen | Verboten |
|---|---|---|---|---|
| 1900–1925 | Akustische Aufnahme (Trichter) | BW ≤ 3 kHz, hohes Oberflächenrauschen, kein Bass | phase_03, phase_06 (max 3 kHz!) | phase_07 (keine Harmonik-Ergänzung) |
| 1925–1945 | Elektrische Aufnahme + Shellac | BW ≤ 7 kHz, SNR ~15 dB, H2/H4-Sättigung | phase_03, phase_06 (max 7 kHz), phase_09 | Rolloff ≤ 7 kHz **nicht** erweitern; H2/H4 bewahren |
| 1945–1960 | Vinyl 78rpm/LP, Mono/frühes Stereo | RIAA-Entzerrung, Knistern, Wow/Flutter | phase_04, phase_09, phase_12 | Stereo-Enhancement (Mono-Quelle) |
| 1960–1975 | Vinyl LP, Analogband, frühes Stereo | Bandrauschen, Azimuth-Fehler, Wow/Flutter | phase_29, phase_25, phase_12 | Overdrive-NR (verwischt Recording-Ambience) |
| 1975–1990 | Vinyl, Cassette (Dolby B/C), Analogband | Dolby-Sättigungsrauschen, Dropouts, HF-Verlust | phase_29, phase_24, phase_03 | Cassette: Dolby nur invertieren wenn NR-Schaltung aktiv war |
| 1985–2000 | CD, DAT, Early Digital | Quantisierungsrauschen (16 bit), Jitter | phase_31, phase_30 | EQ-Erweiterung (über CD-BW = Artefakt) |
| 2000–2015 | MP3 (64–192 kbps), AAC | Pre-Echo, HF-Rolloff, psychoakust. Residue | phase_50, phase_23 | Aggressives NR auf MP3-Artefakte (Pre-Echo = kein Rauschen!) |
| 2015+ | FLAC, MP3 320, Streaming | Selten Artefakte; ggf. Loudness-War-Clipping | phase_01, phase_47 | Fast Passthrough; keine Carrier-Inversion nötig |

> Era-Erkennung: `EraClassifier.classify(audio, sr)` → era_decade, genre_label, carrier_chain

*Stand: Mai 2026 — Aurik 9.12.0 — instructions_version 9.2*
