# Spec 10 — Bug & Gap Detection Strategy (v9.19.0)

> **Scope**: Systematische Erkennung und Eliminierung aller Bugs und Gaps über alle Ebenen —
> Frontend, Bridge/CLI, Denker, UV3-Pipeline, Phasen/DSP/Plugins, Tests.
> Diese Spec ist normativ für alle KI-Agenten-Sessions und CI-Gates.

---

## §10.1 Layer-Scan-Protokoll

Jede Bug-Erkennungs-Session MUSS alle 5 Ebenen systematisch durchlaufen:

| Ebene | Scope | Primäre Scan-Tools |
| --- | --- | --- |
| **L1 Frontend** | `Aurik910/ui/`, `Aurik910/__init__.py` | Version-Konsistenz-Check, `grep -rn "fallback\|hardcoded"` |
| **L2 Bridge/CLI** | `backend/api/bridge.py`, `cli/aurik_cli.py` | Contract-Vollständigkeit: `get_load_audio_fn`, `run_pre_analysis`, `export_guard` |
| **L3 Denker** | `denker/*.py` | `inapplicable_goals`-Propagation, `reference`-Parameter in `messe_ziele()`, `goal_applicability` in Dataclasses |
| **L4 UV3-Pipeline** | `backend/core/unified_restorer_v3.py` | SSIP-Zonen-Übergabe, AdaptivePhaseRescheduler, RestorationMemory, HPG-Update |
| **L5 Phasen/DSP** | `backend/core/phases/`, `backend/core/dsp/`, `plugins/` | V33 MaterialType-Dict-Vollständigkeit, V38 per-Event-Strength, V41 ForwardMaskingGuard |

### §10.1a L5 Phasen-Scan-Checkliste (pro Phase)

Für jede `phase_*.py` mit additiver oder NR-Funktion prüfen:

```
[ ] V38: per-Event-Strength-Oracle (_compute_<defect>_local_strength) bei Event-Schleifen
[x] V41: ForwardMaskingGuard bei additiven Phasen mit panns_singing >= 0.25  ← ERLEDIGT commit 0c9a069
[ ] V33: Alle dict[MaterialType, ...] enthalten CASSETTE-Key
[ ] V42: check_roughness_regression() nach NR in phase_03/phase_29
[ ] V40: compute_nmr_score() wenn FeedbackChain aktiv
[ ] §2.63: Reflect-Padding VOR STFT, deterministischer Strip danach
[ ] §0a: phase_21/35/42 nie in CAUSE_TO_PHASES für Restoration-Cause
[ ] §2.46e: check_hallucination() nach jeder additiven Operation
[ ] V19: compute_noise_texture_distance() nach NR-Phase
[ ] V20: frame_energy_correlation() auf voiced-Zonen nach NR/Kompressor
```

---

## §10.2 Automatische Erkennungs-Werkzeuge

| Werkzeug | Scope | Ausführung |
| --- | --- | --- |
| `scripts/aurik_verboten_linter.py` | V01–V58 (AST-basiert) | `pre-commit`, `pytest --co -q` |
| `mypy backend/core/ --ignore-missing-imports` | Type-Safety | Weekly, vor Release |
| `pytest tests/unit/ -x` | Unit-Regression | Bei jedem Commit |
| `pytest tests/integration/` | Integration | Täglich, CI |
| `pytest tests/normative/` | Spec-Gates (AMRB, MUSHRA) | Release-Gate |
| `scripts/worldclass_kpi_dashboard.py` | KPI-Übersicht | Wöchentlich |
| `scripts/worldclass_release_gate.py` | Release-Blocker-Check | Vor Release |

### §10.2a VERBOTEN-Linter-Abdeckung (V01–V58)

| Block | Regeln | Status |
| --- | --- | --- |
| V01–V10 | print(), sf.read, librosa.load, V03 ml_device, V04 gate_dbfs, V05 griffinlim, V08 percentile, V09 carrier_rollback, V11 sosfilt | ✅ Linter aktiv |
| V11–V20 | sosfilt, V12 CAUSE_TO_PHASES, V13 dict-Duplizierung, V14–V18 SSIP, V19 noise_texture, V20 mikrodynamik | ✅ Linter aktiv |
| V21–V30 | V21 Rauschboden, V22 Pre-Echo, V23 Mono-Kompatibilität, V24 Spektralfarbe, V25 Wärmeband, V26 Onset-Guard, V27–V31 Cause-Mapping-Inversionen | ✅ Linter aktiv |
| V31–V40 | V32 PMGG-Exclusion, V33 MaterialType-Vollständigkeit, V34–V35 Strict-Conflict, V36–V37 AdaptiveRescheduler, V38 per-Event-Oracle, V39/V40 NMR/FeedbackChain | ✅ Linter aktiv |
| V41–V50 | V41 ForwardMasking, V42 Roughness, V43 JND, V44 IACC, V45 VAT, V46 dBFS-Multiplikation, V47 Sub-Ceiling, V48–V50 GAF/Denker-Kette | ✅ Linter aktiv |
| V51–V58 | V51 goal_applicability-Propagation, V52 separation_fidelity-Codec, V53 Singer-ID-DSP-Fallback, V54 HPG-update, V55 WLPC-era, V56 Frontend-Version, V57 ForwardMasking-Pflicht, V58 no-any-return | ✅ neu v9.19.0 |

---

## §10.3 Bug-Taxonomie

| Klasse | Definition | Priorität | Beispiel |
| --- | --- | --- | --- |
| **R-BLOCKER** | Korrektheit-kritisch: Export-Fehler, Crash, Daten-Verlust, Clipping-Artefakt | Sofort (P0) | SSIP None-Return, pegelexplosion |
| **AUDIO-QUALITY** | Hörbar schlechteres Ergebnis als möglich | Nächster Commit (P1) | V41 ForwardMaskingGuard fehlt, V38 per-Event-Strength fehlt |
| **SPEC-GAP** | Spec-Anforderung implementiert aber nicht getestet oder nur partiell | P2 | V33 CASSETTE-Key fehlt in dict |
| **TYPE-SAFETY** | mypy-Fehler (no-any-return, override, etc.) | P3 | UV3 ndarray Returns ohne cast |
| **TEST-DESIGN** | Test-Assertion bricht durch nichtlineare Guards (deterministisch) | P2 | test_phase_65 HallucinationGuard-Rollback |

---

## §10.4 Priorisierungs-Schema

```
P0 (sofort): R-BLOCKER — Export-Crash, Pegelexplosion, Stille-Zone-Verletzung
P1 (nächster Commit): AUDIO-QUALITY-Lücke, Spec-RELEASE_MUST-Verletzung
P2 (nächste 3 Commits): SPEC-GAP, TEST-DESIGN-Fix
P3 (Backlog): TYPE-SAFETY, Linter-Coverage, Dokumentation
```

**§10.4a Eskalations-Trigger** — P0-Upgrade wenn:

- `artifact_freedom < 0.95` durch Bug reproduzierbar
- `VQI < 0.72` durch Bug reproduzierbar
- Stille-Zone-Energie durch Bug eingeführt
- Export enthält Over-processed Audio statt Fallback

---

## §10.5 Behebungs-Workflow (§0f-konform)

```
1. ERKENNEN: Layer-Scan (§10.1) → Bug-Klasse (§10.3) → Priorität (§10.4)
2. ROOT-CAUSE: grep + read_file → Minimal-Reproduktion in Test
3. FIX: Punkt-Fix (1-4 Stellen) oder Systemisch (≥5 Stellen, §0f)
4. TEST: Unit-Test + ggf. Integrations-Test
5. VERBOTEN.md: Neue VERBOTEN-Regel wenn wiederholbares Anti-Pattern
6. SPEC-UPDATE: Betroffene Spec-Datei (01–10) + copilot-instructions.md
7. COMMIT: `fix §X systemic: ...` (systemisch) oder `fix(phase_XX): ...` (punktuell)
```

### §10.5a Systemisch vs. Punktuell (§0f-Regel)

| Signal | Vorgehen |
| --- | --- |
| 1 Stelle betroffen | Punktuell — direkter Fix |
| 2–4 Stellen | Prüfe Abstraktion: zentraler Helper wenn Overhead gering |
| ≥5 Stellen | **Systemisch**: zentrale Funktion + alle Callsites + VERBOTEN-Regel + Linter |
| Bug in ≥2 Sessions re-introduced | Systemisch — Linter-Regel fehlt |

---

## §10.6 Bekannte offene Gaps (Stand v9.19.0 / Update v9.19.1)

### §10.6a P2: mypy-Cleanup-Backlog

| Datei | Fehler-Anzahl | Typ |
| --- | --- | --- |
| `backend/core/unified_restorer_v3.py` (isoliert) | ~46 | `no-any-return` für ndarray |
| `backend/core/authenticity_metrics_extended.py` | ~27 | Mixed |
| `backend/core/regulator/_dsp_applier.py` | ~24 | Mixed |
| `backend/core/optimization/uncertainty_quantification.py` | ~23 | Mixed |

Fix-Pattern: `cast(np.ndarray, result)` oder `# type: ignore[no-any-return]` (V58).

### §10.6b ~~P2: SSIP phase_55/24 intern (V14–V18)~~ — ERLEDIGT

`phase_55_diffusion_inpainting.py` (Z.1347–1577) und `phase_24_dropout_repair.py` (Z.958–1226)
haben `post_inpainting_silence_audit()` vollständig implementiert. Gap war ein Inventar-Fehler.

### §10.6c ~~P3: V33 MaterialType CASSETTE-Keys~~ — ERLEDIGT

Scan (2026-06-26): Alle Phasen mit `dict[MaterialType, ...]` enthalten CASSETTE-Key.
Gap war ein Inventar-Fehler.

### §10.6d ~~P1: V41 ForwardMaskingGuard additiv~~ — ERLEDIGT (v9.19.1, commit 0c9a069)

14 additive Phasen (phase_13,21,24,42,44,45,46,50,51,55,56,58,60,64) hatten keinen
ForwardMaskingGuard-Block. Standard-Block integriert: `panns_singing >= 0.25` →
`zone_frac * 0.15` Boost, non-blocking try/except. V41-Gap-Scan: 0 verbleibend.

---

## §10.7 Session-Start-Protokoll (KI-Agent)

Zu Beginn jeder Session, bevor Implementierung beginnt:

```bash
# 1. Failing Tests prüfen
pytest tests/unit/ -x --tb=line -q --timeout=30 2>&1 | tail -5

# 2. Aktuelle Branches / staged Files
git status --short && git log --oneline -3

# 3. Bekannte offene Gaps (diese Datei) lesen
# → specs/10_bug_gap_strategy.md §10.6

# 4. VERBOTEN.md auf neue Anti-Patterns prüfen
tail -20 .github/VERBOTEN.md
```

---

## §10.8 Kontinuierlicher Scan-Rhythmus

| Rhythmus | Aktion |
| --- | --- |
| Jeder Commit | Unit-Tests + VERBOTEN-Linter + mypy (staged files) |
| Täglich | Integration-Tests + Layer-Scan L1–L3 |
| Wöchentlich | mypy-Vollscan + Layer-Scan L4–L5 + KPI-Dashboard |
| Vor Release | Vollsuite + AMRB-Gate + Competitive-Gate + Worldclass-Release-Gate |

---

_Stand: v9.19.0, Juni 2026 — automatisch gepflegt, Änderungen per Commit §10-konform_
