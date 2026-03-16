# Aurik 9 — Intelligentes Musik-Restaurierungssystem

**Version:** 9.10.51 | **Status:** ✅ Produktionsbereit — Weltführend

**Datum:** März 2026

---

> Aurik 9.x.x ist ein *weltweit erstmaliges intelligentes, kontextbewusstes
> Musik- und Gesangs-Restaurierungs-, Reparatur- und Rekonstruktions-Denkersystem*.
> Es kombiniert psychoakustisch fundierte DSP, Bayesianische Kausalinferenz,
> Gaussianische Prozess-Optimierung und perceptuelle Qualitätsbewertung zu einer
> kognitiven Restaurierungs-Intelligenz.

---

## 📖 Quick Links

### Für Anwender
- **[Installations-Guide](guides/INSTALLATION.md)** — AppImage (Linux) / Installer (Windows)
- **[Benutzerhandbuch](guides/USER_GUIDE.md)** — Vollständige Bedienungsanleitung
- **[Konfigurations-Guide](guides/CONFIGURATION.md)** — Modi (Restoration / Studio 2026) & Parameter
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** — Problemlösung

### Für Entwickler
- **[KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)** — Regeln für KI-Agenten **(Pflicht!)**
- **[KI-Programmierrichtlinien](../.github/copilot-instructions.md)** — Bindende Systemregeln **(Pflicht!)**
- **[Python API Reference](api/PYTHON_API.md)** — API-Dokumentation
- **[Architecture Overview](architecture/ARCHITECTURE.md)** — Systemarchitektur (5 Schichten)
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** — 56-Phasen-Pipeline
- **[Contributing Guide](development/CONTRIBUTING.md)** — Beitrag leisten
- **[Testing Guide](development/TESTING.md)** — Teststrategie (6312 Tests)

### Status & Fortschritt
- **[Project Status Report](PROJECT_STATUS.md)** — Aktueller Entwicklungsstand v9.10.51
- **[Roadmap](aurik9_roadmap.md)** — Zukunftspläne (v10.0+)

📚 **[Vollständiger Dokumentations-Index](INDEX.md)**

---

## 🎯 Aurik 9.10.51 — Highlights

### ✅ Produktionsbereit (März 2026)

**Kennzahlen:**
- ✅ **6.312 Tests** — alle grün ✅
- ✅ **14 Musical Goals** — alle psychoakustisch geprüft
- ✅ **56 Phasen** (Phase 01–56, Defect-First)
- ✅ **15 auto-erkannte Materialtypen**
- ✅ **27 DefectTypes** erkannt und behandelt
- ✅ **CPU-only** (Desktop-Hardware, kein GPU erforderlich)
- ✅ **100 % offline** nach Installation (keine Cloud)

**Schlüssel-Module (v9.10.51):**

1. **14 Musical Goals** — Brillanz, Wärme, Natürlichkeit, Authentizität, Emotionalität, Transparenz, Bass-Kraft, Groove, Raumtiefe, Timbre-Authentizität, Tonales Zentrum, Mikro-Dynamik, Separation-Treue, Artikulation
2. **Transient Decoupled Processing (TDP)** — Transienten separat durch Pipeline führen; GrooveMetric +0.03–0.06
3. **HarmonicPreservationGuard (HPG)** — G_floor 0.85 an harmonischen Partials; Natürlichkeit +0.03–0.07
4. **PerPhaseMusicalGoalsGate (PMGG)** — kein kumulativer Qualitätsverlust; max. 5 Retries pro Phase
5. **MicroDynamicsEnvelopeMorphing (MDEM)** — originales Mikro-Dynamik-Profil wiederherstellen
6. **GermanSchlagerClassifier** — 6-Schicht Zero-Shot: Akkordeon-Reed-Beating + HSI + Rhythmus + Formant-Prior
7. **EraClassifier** — Aufnahme-Ära 1890–2025 erkennen; GP-Warmstart pro Epoche
8. **RestorabilityEstimator** — Vor-Assessment < 5 s; Score 0–100 + predicted MOS
9. **StemRemixBalancer** — LUFS-korrekter Re-Mix nach getrennter Stem-Verarbeitung
10. **AudioFileValidator** — Sicherheitsprüfung vor jeder DSP-Verarbeitung (OWASP A03)

---

## ⚡ API-Beispiel

```python
from denker import restauriere
import soundfile as sf

audio, sr = sf.read("vinyl_recording.wav")
ergebnis = restauriere(audio, sr=48_000)

print(f"Material: {ergebnis.material_type}")
print(f"Qualität: {ergebnis.qualitaet:.2f}")
print(f"PQS MOS: {ergebnis.pqs_result.mos:.2f}")
print(f"Echtzeit-Faktor: {ergebnis.rt_factor:.2f}×")

sf.write("output_restored.flac", ergebnis.audio, 48_000)
```

---

## 🎛️ Zwei Restaurierungs-Modi

| Modus | Ziel | Strength |
|---|---|---|
| **Restoration** | Originalgetreu, minimal-invasiv, historisch authentisch | Konservativ |
| **Studio 2026** | Modern, klar, kraeftig — heutiger Referenzstandard | Aggressiv |

Beide Modi via Magic Buttons in der GUI erreichbar.

---

## 📦 Distribution

| Plattform | Format | Status |
|---|---|---|
| **Linux** | AppImage (`.AppImage`) | ✅ |
| **Windows 10/11** | NSIS-Installer (`.exe`) | ✅ |

Keine Python-Kenntnisse, kein Terminal, kein `pip install` erforderlich.
