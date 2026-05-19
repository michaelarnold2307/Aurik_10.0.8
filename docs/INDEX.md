
# 📚 Aurik 9.x.x — Projektdokumentation

Offizielle Dokumentation von **Aurik 9.12.8** — einem intelligenten,
kontextbewussten Musik- und Gesangs-Restaurierungs-, Reparatur- und
Rekonstruktions-Denkersystem. Alle Inhalte sind an die KI-Programmierrichtlinien
(`.github/copilot-instructions.md`) ausgerichtet.

**Version:** 9.12.8 | **Phasen:** 64 | **Musical Goals:** 14 | **DefectTypes:** 47 | **Tests:** ~13.662

> Hinweis: Verbindlicher Wahrheitsstand ist die Spezifikation in `.github/specs/01-08` plus `docs/CHANGELOG_HISTORY.md`. Wo Zahlen abweichen, gelten Specs/Changelog.
> Evidenzhinweis: Interne Qualitätsmetriken, Competitive-Benchmarks und Studienpakete in dieser
> Dokumentation dienen der technischen Steuerung und Validierung. Öffentliche Superlative oder
> formale Hörtest-Äquivalenz sind erst mit unabhängiger externer Evidenz belastbar.

---

## 📖 Quick Navigation

### Für Anwender

- **[Installation Guide](guides/INSTALLATION.md)** – Systemvoraussetzungen & Installation (Linux / Windows)
- **[User Guide](guides/USER_GUIDE.md)** – Vollständiges Benutzerhandbuch
- **[Configuration Guide](guides/CONFIGURATION.md)** – Modi (Restoration / Studio 2026) & Parameter
- **[Troubleshooting Guide](guides/TROUBLESHOOTING.md)** – Problemlösung & FAQ
- **[Phoneme Processing Guide](guides/PHONEME_PROCESSING_GUIDE.md)** – §2.36 LyricsGuidedEnhancement + PhonemeTimeline
- **[Processing Logger Usage](guides/PROCESSING_LOGGER_USAGE.md)** – Logging-System

### Für Entwickler

- **[KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)** – Regeln für KI-Agenten **(Pflicht!)**
- **[KI-Programmierrichtlinien](../.github/copilot-instructions.md)** – Bindende Systemregeln **(Pflicht!)**
- **[Python API Reference](api/PYTHON_API.md)** – API-Dokumentation
- **[Architecture Overview](architecture/ARCHITECTURE.md)** – Systemarchitektur (4 Schichten)
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** – 64-Phasen-Pipeline (Defect-First, Phase 01–64)
- **[Pipeline Flow](architecture/PIPELINE_FLOW_ANALYSIS.md)** – Ablauf & Datenfluss
- **[Contributing Guide](development/CONTRIBUTING.md)** – Beitrag leisten
- **[Testing Guide](development/TESTING.md)** – Teststrategie & Best Practices
- **[Performance Guard Spec](PERFORMANCE_GUARD_SPEC.md)** – 3×-Echtzeit-Budgetregeln
- **[MUSHRA Studienprotokoll](guides/MUSHRA_STUDIENPROTOKOLL.md)** – Externe Hoervalidierung nach ITU-R BS.1534-3
- **[Statistische Kalibrierungsrichtlinie](guides/STATISTISCHE_KALIBRIERUNGSRICHTLINIE.md)** – Datenbasierte Schwellwert-Aenderungen
- **[CI Determinismus Checkliste](guides/CI_DETERMINISMUS_CHECKLISTE.md)** – Reproduzierbarkeits-Gates fuer CI/Release
- **[Freude-Gate](guides/FREUDE_GATE.md)** – Psychoakustisches PASS/FAIL-Release-Gate fuer Hoerfreude
- **[Phase Harmonization Audit](../scripts/phase_harmonization_audit.py)** – Modulweise 100%-Checkliste fuer produktive Strength/Locality-Harmonisierung
- **[Spec Evidenzblock Template](guides/SPEC_EVIDENZBLOCK_TEMPLATE.md)** – Pflichtnachweis fuer normative Spec-Aenderungen
- **[PR Hoervalidierung Template](guides/PR_HOERVALIDIERUNG_TEMPLATE.md)** – Review-Template fuer Blindtest/Mini-MUSHRA
- **[Externe MUSHRA-Studie 2026 Q2](reports/studies/mushra_2026q2/README.md)** – Studienpaket mit Praeregistration und Szenarienmatrix
- **[Spec Evidence Reports](reports/spec_evidence/README.md)** – Pflichtberichte bei Spec-Aenderungen auf main (Solo-Release-Gate)

### Status & Fortschritt

- **[Project Status Report](PROJECT_STATUS.md)** – Projektstatus (normativer Stand via Specs/Changelog)
- **[Musical Excellence Analysis](musical_excellence_next_steps.md)** – Qualitätsanalyse & Roadmap
- **[Roadmap](aurik9_roadmap.md)** – Zukunftspläne (Studio 2026+)

---

## 📁 Dokumentationsstruktur

```text
docs/
├── guides/                     # Anwender-Guides
│   ├── INSTALLATION.md        # Installation (Linux AppImage & Windows 10/11)
│   ├── CONFIGURATION.md       # Modi (Restoration / Studio 2026), Parameter
│   ├── TROUBLESHOOTING.md     # Problemlösung & FAQ
│   ├── USER_GUIDE.md          # Vollständiges Benutzerhandbuch
│   ├── QUICKSTART_SUPPORT.md  # Schnelleinstieg
│   ├── LOCAL_APP_DEPLOYMENT.md # Desktop-Deployment
│   └── PHONEME_PROCESSING_GUIDE.md  # §2.36 LyricsGuidedEnhancement
│
├── architecture/               # Architektur-Dokumentation
│   ├── ARCHITECTURE.md        # Systemarchitektur (4 Schichten)
│   ├── PHASES_OVERVIEW.md     # 64-Phasen-Pipeline (Defect-First, Phase 01–64)
│   └── PIPELINE_FLOW_ANALYSIS.md
│
├── api/                        # API-Dokumentation
│   └── PYTHON_API.md          # Python-API-Referenz
│
├── reports/                    # Statusberichte
│   ├── current/               # Aktuelle Berichte (2026)
│   └── phase_completion/      # Phasenabschlussberichte
│
├── development/                # Entwickler-Dokumentation
│   ├── CONTRIBUTING.md        # Beitrag leisten
│   ├── TESTING.md             # Teststrategie
│   ├── TESTING_BEST_PRACTICES.md
│   ├── DECISION_NO_VST3_PLUGIN.md  # Architektur-Entscheidung
│   └── Packaging_Documentation.md  # AppImage / Windows-Build
│
├── archive/                    # Historische Dokumente (26 Dateien, nicht mehr aktiv)
│   └── README.md              # Archiv-Index
│
├── INDEX.md                    # ⭐ Diese Datei
├── README.md                   # Kurzübersicht → verweist auf INDEX.md
├── KI-AGENT-INTEGRATION-GUIDE.md  # ⚠️ Pflichtlektüre für KI-Agenten
├── PROJECT_STATUS.md          # Projektstatus (Living Document)
├── AURIK_9.x.x_ARCHITEKTUR.md # Kognitive Pipeline-Übersicht (Detail)
├── DEFECT_SCANNER_SPEC.md     # DefectScanner-Spezifikation (aktuelle DefectTypes siehe Specs)
├── VOCAL_AI_ENHANCEMENT.md    # Vocal-Pipeline §2.8 (Formanten, Breathiness)
├── PERFORMANCE_GUARD_SPEC.md  # 3×-Echtzeit-Budget [RELEASE_MUST]
├── RESOURCE_AWARE_FALLBACK.md # PLM / RAM-Budget
├── UNIFIED_RESTORER_V3_SPEC.md # UV3-Spezifikation
├── MODULAR_PHASES_API.md      # Phasen-API
├── COMPREHENSIVE_METRICS.md   # PQS-Metriken & OQS
├── CI_CD.md                   # CI/CD-Pipeline
├── aurik9_roadmap.md          # Roadmap
├── musical_excellence_next_steps.md  # Qualitätsziele 2026
├── natural_sound_improvement_analysis.md
└── tier2_ml_hybrid_analysis.md
```

---

## 🧠 Normkonformität & KI-Richtlinien

**Bindende Regeln:** `.github/copilot-instructions.md`

Schlüsselbindungen (Auszug):

- Interne SR: immer **48 000 Hz** — vor und nach jedem DSP-Schritt
- **CPU-only**: keine GPU/CUDA — `providers=["CPUExecutionProvider"]`
- **14 Musical Goals**: nach jeder Restaurierung zu prüfen, Regression = Feature ungültig
- **64 Phasen** (Phase 01–64, Defect-First) in `backend/core/phases/`
- Material-adaptive Verarbeitung via `MediumDetector.detect(audio, sr, file_ext=...)`
- **47 DefectTypes** — vollständiger Defektkatalog in `core/defect_scanner.py`
- **Desktop-only (Linux AppImage + Windows 10/11)**: keine Cloud, kein Docker, kein pip für Endnutzer
- **§2.36 LyricsGuidedEnhancement**: Whisper-Tiny ONNX + wav2vec2 Forced Alignment (Phase 58)
- **§2.39 OOM-Recovery-Checkpoint**: Nahtlose Pipeline-Wiederaufnahme nach OOM-Kill
- **ml_memory_budget.try_allocate() Pflicht** vor jedem ML-Modell-Laden
- **AMRB-Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 Szenarien [RELEASE_MUST]

---

### Erste Schritte

1. [Installation Guide](guides/INSTALLATION.md)
2. [User Guide](guides/USER_GUIDE.md)
3. [Configuration Guide](guides/CONFIGURATION.md)
4. [Troubleshooting Guide](guides/TROUBLESHOOTING.md)

### KI-Agenten (Pflicht)

- [KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)
- [KI-Programmierrichtlinien](../.github/copilot-instructions.md)

### Architektur & Entwicklung

- [Python API Reference](api/PYTHON_API.md)
- [Contributing Guide](development/CONTRIBUTING.md)
- [Testing Guide](development/TESTING.md)
- [Phases Overview](architecture/PHASES_OVERVIEW.md)
- [Pipeline Flow Analysis](architecture/PIPELINE_FLOW_ANALYSIS.md)
- [Architecture Overview](architecture/ARCHITECTURE.md)

### Projekt-Status

- [Project Status Report](PROJECT_STATUS.md)
- [Musical Excellence Analysis](musical_excellence_next_steps.md)
- [Roadmap](aurik9_roadmap.md)

---

## 📅 Aktuelle Updates

**Mai 2026 (v9.12.8):**

- Genre-Phase-1 (GenreClassifier: Family+Top-k+Open-Set, Lyrics-Fusion, UI-Badge) integriert.
- Normative Nachschärfung Lyrics-Produktivpfad (§2.36): Einziger Produktionspfad ist `backend/core/lyrics_guided_enhancement.py`.
- Mode-differenzierte Musical-Goals-Härtung und Priority-Aware PMGG dokumentiert.
- OOM-Recovery-Checkpoint-System (§2.39) und KMV (§2.38) als normative Pipeline-Bestandteile konsolidiert.
- DefectScanner-/Kausalpfad auf aktuellen DefectType- und Ursachenumfang nachgezogen.

Historische Detailstände sind in `docs/CHANGELOG_HISTORY.md` aufgeführt.

---
**Aurik 9.12.8 Projektdokumentation** | Letzte Aktualisierung: Mai 2026

[🏠 Zurück zur README](../README.md)
