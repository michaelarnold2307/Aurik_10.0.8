# Aurik 10 — Weltklasse-Audio-Restaurierung

**Ziel:** Intelligente Musikwiederherstellung mit psychoakustischer Präzision und deterministischer Reproduzierbarkeit.



## 🚀 v10 Invarianten

- **Bridge-Bypass-Verbot**: Kein Frontend-Code importiert `backend/core/` direkt. Nur über `backend/api/bridge.py`.
- **Soft-Knee-Gate**: `apply_musical_gain_envelope()` arbeitet mit Sigmoid-Soft-Knee (6dB), 200ms Hanning-Crossfade. KEIN Hard-Clamp.
- **PIM-first**: Vor jedem Phasen-Loop wird die PIM-Intensitäts-Map berechnet und in `restoration_context` gespeichert.
- **RLP-last**: Nach jedem Phasen-Loop wird der RLP ausgeführt. Korrekturen werden nur bei objektiver Verbesserung übernommen.
- **ML-Fallback-Logging**: JEDER ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden. Silent-Failures sind VERBOTEN.
- **Artistic Intent vor Defect-Scan**: `get_artistic_intent()` wird VOR dem Defect-Scan aufgerufen.
- **Glue Stage immer**: Die Glue-Stage läuft in ALLEN Modi als vorletzte Phase.
- **62 DefectTypes**: Keine willkürlichen neuen DefectTypes ohne Phase-Mapping und Material-Sensitivity.

## 🎯 Kernwerte

- **Präzision über Geschwindigkeit** — Akustische Wahrheit zuerst
- **Transparenz** — Jede Entscheidung ist nachvollziehbar
- **Konsistenz** — Derselbe Input → derselbe Output (immer)
- **Wissenschaftlichkeit** — SOTA-Modelle, verifiable Metriken

## 📊 Architektur-Ebenen

```
CLI (denker/aurik_cli.py)
  ↓
Bridge API (backend/api/bridge.py) [Mode-Normalisierung]
  ↓
Adaptive Pipeline (backend/adaptive_pipeline.py)
  ↓
Kernmodule [Psychoakustik + DSP]
  ├─ Defekt-Scanner (backend/core/defect_scanner.py)
  ├─ Phasen-Engine (backend/core/phases/)
  ├─ Musikalische Ziele (backend/core/musical_goals/)
  ├─ Excellence Optimizer (backend/core/excellence_optimizer.py)
  ├─ DSP/Formanten (backend/core/dsp/lpc_formant_tracker.py)
  └─ ML-Bridges (backend/ml/inference_only/)
  ↓
Export (backend/exporter.py)
```

## ✅ Qualitäts-Gating

### 1. **Tests** (Ziel: 80%+ Abdeckung)

- Unit: schnell, isoliert, ≤1s
- Integration: echte Daten, ≤5s  
- E2E: golden samples, ≤30s
- Contract: Bridge↔API Invarianten

### 2. **Typisierung** (mypy strict in Backend)

- Alle Backend-Funktionen: `def func(arg: Type) -> Type:`
- ML-Plugins: `ignore_errors = true` (extern)
- DSP-Core: Type-Hints für öffentliche APIs

### 3. **Linting** (ruff select=[E,W,F,I,N,UP,B,C4,SIM,RUF])

- Automatische Fixes: `ruff check --fix`
- Per-File-Ignores nur für bewusste DSP-Konventionen
- Pre-Commit: ruff, black, isort

### 4. **Konsistenz**

- **Imports:** isort (Black-kompatibel) → `from module import name`
- **Formatierung:** Black 120er Zeilenlänge
- **Namensgebung:** snake_case (PEP8), Math-Variablen (N, X, sr)

## 🔄 CI/CD-Pipeline (Aurik 9.20.3)

**Status:** aktiver 9.20.3-Qualitaetsbranch

- DSP-Verbesserungen (LPC, NMR, Formanten-Tracking)
- Backend-Optimierungen (Phasenmapper, Kalibrierung)
- Compliance-Gates (competitive-ci-gate, quality-gate)

**Nächste Schritte:**

1. ✅ Abhängigkeiten installieren (pydantic, etc.)
2. 🔴 Tests grün kriegen (61 Tests, 1 Error: Missing pydantic)
3. 🔴 Type-Checking durchlaufen
4. 🔴 Linting-Check (ruff check)
5. ✅ Spec-Review (vs. /github/specs/)

## 📝 Dateien-Struktur

```
backend/
  ├─ core/
  │  ├─ defect_scanner.py          # Audio-Defekt-Klassifizierung
  │  ├─ phases/                    # Phase-Engine
  │  ├─ musical_goals/             # Qualitäts-Metriken
  │  ├─ dsp/                       # DSP-Algorithmen
  │  │  ├─ lpc_formant_tracker.py
  │  │  ├─ nmr_feedback.py
  │  │  ├─ temporal_masking.py
  │  │  └─ zwicker_metrics.py
  │  └─ unified_restorer_v3.py     # Haupt-Orchestrator
  ├─ api/bridge.py                 # Mode-Normalisierung (CRITICAL)
  ├─ adaptive_pipeline.py          # Phasen-Orchestrierung
  └─ exporter.py                   # Output-Normalisierung

cli/
  └─ aurik_cli.py                  # CLI-Interface

tests/
  ├─ unit/                         # Schnelle Tests
  ├─ integration/                  # Mit echten Daten
  ├─ normative/                    # Contract-Tests
  └─ musical_goals/                # Quality-Gate-Tests
```

## 🚫 VERBOTEN

Siehe `.github/VERBOTEN.md` — nicht verhandelbarer Sicherheits- & Qualitäts-Katalog.

## 🔗 Externe Ressourcen

- Spezifikationen: `.github/specs/`
- Instruktionen: `.github/instructions/`
- Copilot-Verhaltensrichtlinien: `.github/copilot-instructions.md`
