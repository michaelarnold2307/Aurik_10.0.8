
# 📚 Aurik 9.x.x — Projektdokumentation

Offizielle Dokumentation von **Aurik 9.10.51** — dem weltweit ersten intelligenten,
kontextbewussten Musik- und Gesangs-Restaurierungs-, Reparatur- und
Rekonstruktions-Denkersystem. Alle Inhalte sind an die KI-Programmierrichtlinien
(`.github/copilot-instructions.md`) ausgerichtet.

**Version:** 9.10.51 | **Phasen:** 56 | **Tests:** 6312 | **Materialien:** 15 | **Musical Goals:** 14

---

## 📖 Quick Navigation

### Für Anwender
- **[Installation Guide](guides/INSTALLATION.md)** – Systemvoraussetzungen & Installation
- **[User Guide](guides/USER_GUIDE.md)** – Vollständiges Benutzerhandbuch
- **[Configuration Guide](guides/CONFIGURATION.md)** – Modi (Restoration / Studio 2026) & Parameter
- **[Troubleshooting Guide](guides/TROUBLESHOOTING.md)** – Problemlösung & FAQ
- **[Quickstart Guide](guides/QUICKSTART_SUPPORT.md)** – Schnelleinstieg

### Für Entwickler
- **[KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)** – Regeln für KI-Agenten **(Pflicht!)**
- **[KI-Programmierrichtlinien](../.github/copilot-instructions.md)** – Bindende Systemregeln **(Pflicht!)**
- **[Python API Reference](api/PYTHON_API.md)** – API-Dokumentation
- **[Architecture Overview](architecture/ARCHITECTURE.md)** – Systemarchitektur
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** – 56-Phasen-Pipeline
- **[Pipeline Flow](architecture/PIPELINE_FLOW_ANALYSIS.md)** – Ablauf & Datenfluss
- **[Contributing Guide](development/CONTRIBUTING.md)** – Beitrag leisten
- **[Testing Guide](development/TESTING.md)** – Teststrategie & Best Practices (6312 Tests)

### Status & Fortschritt
- **[Project Status Report](PROJECT_STATUS.md)** – Aktueller Entwicklungsstand v9.10.51
- **[Musical Excellence Analysis](musical_excellence_next_steps.md)** – Qualitätsanalyse & Roadmap
- **[Roadmap](aurik9_roadmap.md)** – Zukunftspläne (v10.0+)

---


---

## 📁 Dokumentationsstruktur

```
docs/
├── guides/                     # Anwender-Guides
│   ├── INSTALLATION.md        # Installation (Linux & Windows 10/11)
│   ├── CONFIGURATION.md       # Modi (Restoration / Studio 2026), Parameter
│   ├── TROUBLESHOOTING.md     # Problemlösung & FAQ
│   ├── USER_GUIDE.md          # Vollständiges Benutzerhandbuch
│   └── QUICKSTART_SUPPORT.md  # Schnelleinstieg
│
├── architecture/               # Architektur-Dokumentation
│   ├── ARCHITECTURE.md        # Systemarchitektur (5 Schichten)
│   ├── PHASES_OVERVIEW.md     # 56-Phasen-Pipeline (Defect-First)
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
│   ├── CONTRIBUTING.md
│   ├── TESTING.md             # Teststrategie (6312 Tests)
│   └── PROCESSING_LOGGER_SPECIFICATION.md
│
├── archive/                    # Historische Dokumentation
│
├── INDEX.md                    # ⭐ Diese Datei
├── KI-AGENT-INTEGRATION-GUIDE.md  # ⚠️ Pflichtlektüre für KI-Agenten
├── PROJECT_STATUS.md          # Aktueller Projektstand v9.10.51
├── DEFECT_SCANNER_SPEC.md     # 27 DefectTypes, 15 MaterialTypes
└── VOCAL_AI_ENHANCEMENT.md    # VoiceGender-System, Formanten
```
│   ├── old_roadmaps/          # Previous roadmaps (archived)
│   ├── old_summaries/         # Previous summaries (archived)
│   ├── old_status/            # Previous status docs (archived)
│   └── obsolete/              # Obsolete documentation
│
├── 00_normative/              # Legacy: Normative documents
├── 01_architecture/           # Legacy: Old architecture docs
├── 02_processing/             # Legacy: Old processing docs
└── 03_ui_spec/                # Legacy: Old UI specs
```

---

## 🧠 Normkonformität & KI-Richtlinien

Alle Module, Phasen, DSP/ML-Komponenten und Metriken sind normkonform dokumentiert.

**Bindende Regeln:** `.github/copilot-instructions.md`

Schlußssselbindungen (Auszug):
- Interne SR: immer **48 000 Hz** — vor und nach jedem DSP-Schritt, jeder ML-Inferenz, jeder Metrik
- **CPU-only**: keine GPU/CUDA — `providers=["CPUExecutionProvider"]`
- **7 Musical Goals**: nach jeder Restaurierung zu prüfen, Regression = Feature ungültig
- **55 Phasen** (Phase 01–55) — nur echte Dateinamen in `core/phases/`
- **12 Materialien** — auto-erkannt durch `DefectScanner`
- **21 DefectTypes** — vollständiger Defektkatalog in `core/defect_scanner.py`
- **Anti-Parallelwelten**: vor jeder Implementierung bestehende Module prüfen
- **Desktop-only**: keine Cloud-, Server- oder Netzwerk-Abhängigkeiten

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

**März 2026 (v9.8.0):**
- Über-SOTA DSP-Algorithmen implementiert: OMLSA/IMCRA, pYIN, NMF-β, PGHI
- Dokumentation vollständig auf v9.8.0 ausgerichtet, 222 Tests grün

**19. Februar 2026 (v9.7.0):**
- Dokumentation vollständig auf v9.7.0 ausgerichtet
- README.md, INDEX.md, PROJECT_STATUS.md, KI-AGENT-INTEGRATION-GUIDE.md aktualisiert
- 55 Phasen (nicht 42), 12 Materialien (nicht 5), 206 Tests (nicht 9)
- 7 Musical Goals mit Pflicht-Schwellwerten dokumentiert
- Magic-Button-Implementierung (restoration.png / studio.png) dokumentiert

**17. Februar 2026:**
- v9.7.0: Kognitive Architektur (PerceptualEmbedder, CausalDefectReasoner,
  GPParameterOptimizer, PerceptualQualityScorer, MusicalGoalsChecker)
- 40 neue Tests (Gesamt: 206)

---

<div align="center">

**Aurik 9.8.0 Projektdokumentation** | Letzte Aktualisierung: März 2026

[🏠 Zurück zur README](../README.md)

</div>
