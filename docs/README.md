# Aurik 9 — Intelligentes Musik-Restaurierungssystem

**Version:** 9.11.53 | **Status:** ✅ Produktionsbereit — auf Spitzenqualität ausgerichtet

**Datum:** April 2026

> Hinweis: Verbindliche Detailstände liegen in `.github/specs/01-08` und `docs/CHANGELOG_HISTORY.md`.

---

> Aurik 9.x.x ist ein _intelligentes, kontextbewusstes
> Musik- und Gesangs-Restaurierungs-, Reparatur- und Rekonstruktions-Denkersystem_.
> Es kombiniert psychoakustisch fundierte DSP, Bayesianische Kausalinferenz,
> Gaussianische Prozess-Optimierung und perceptuelle Qualitätsbewertung zu einer
> kognitiven Restaurierungs-Intelligenz.
> Evidenzhinweis: Öffentliche Superlative und formale Hörtest-Äquivalenz werden
> erst durch unabhängige, verblindete Hörtests und reproduzierbare externe
> Wettbewerbsvergleiche belastbar.

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
- **[Architecture Overview](architecture/ARCHITECTURE.md)** — Systemarchitektur (4 Schichten)
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** — 64-Phasen-Pipeline
- **[Contributing Guide](development/CONTRIBUTING.md)** — Beitrag leisten
- **[Testing Guide](development/TESTING.md)** — Teststrategie und Qualitätssicherung

### Status & Fortschritt

- **[Project Status Report](PROJECT_STATUS.md)** — Projektstatus (Living Document)
- **[Roadmap](aurik9_roadmap.md)** — Zukunftspläne (Studio 2026+)

📚 **[Vollständiger Dokumentations-Index](INDEX.md)**

---

## 🎯 Aurik 9.11.53 — Highlights

### ✅ Produktionsbereit (April 2026)

**Kennzahlen:**

- ✅ **~11.598 Tests** — Unit/Normative/Integration grün
- ✅ **14 Musical Goals** — alle psychoakustisch geprüft, mode-differenziert (Restoration/Studio 2026)
- ✅ **64 Phasen** (Phase 01–64, Defect-First)
- ✅ **Material-adaptive Verarbeitung** über Medium-/Era-Kontext
- ✅ **46 DefectTypes** erkannt und behandelt
- ✅ **CPU/GPU (AMD ROCm/DirectML)** — Mixed-Mode, GPU optional
- ✅ **100 % offline** nach Installation (keine Cloud)

**Schlüssel-Module (v9.11.53):**

1. **14 Musical Goals** — Brillanz, Wärme, Natürlichkeit, Authentizität, Emotionalität, Transparenz, Bass-Kraft, Groove, Raumtiefe, Timbre-Authentizität, Tonales Zentrum, Mikro-Dynamik, Separation-Treue, Artikulation
2. **Transient Decoupled Processing (TDP)** — Transienten separat durch Pipeline führen; GrooveMetric +0.03–0.06
3. **HarmonicPreservationGuard (HPG)** — G_floor 0.85 an harmonischen Partials; Natürlichkeit +0.03–0.07
4. **PerPhaseMusicalGoalsGate (PMGG)** — kein kumulativer Qualitätsverlust; max. 5 Retries pro Phase
5. **MicroDynamicsEnvelopeMorphing (MDEM)** — originales Mikro-Dynamik-Profil wiederherstellen
6. **LyricsGuidedEnhancement** (§2.36) — Whisper-Tiny ONNX + Phonem-Alignment; Phase 58
7. **GermanSchlagerClassifier** — 6-Schicht Zero-Shot: Akkordeon-Reed-Beating + HSI + Rhythmus
8. **GenreClassifier** (Genre-Phase-1) — Family+Top-k+Open-Set, SongCal-Fusion, UI-Badge
9. **OOM-Recovery-Checkpoint** (§2.39) — nahtlose Pipeline-Wiederaufnahme nach OOM-Kill
10. **RestorabilityEstimator** — Vor-Assessment < 5 s; Score 0–100 + predicted MOS

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
| --- | --- | --- |
| **Restoration** | Originalgetreu, minimal-invasiv, historisch authentisch | Konservativ |
| **Studio 2026** | Modern, klar, kraeftig — heutiger Referenzstandard | Aggressiv |

Beide Modi via Magic Buttons in der GUI erreichbar.

---

## 📦 Distribution

| Plattform | Format | Status |
| --- | --- | --- |
| **Linux** | AppImage (`.AppImage`) | ✅ |
| **Windows 10/11** | NSIS-Installer (`.exe`) | ✅ |

Keine Python-Kenntnisse, kein Terminal, kein `pip install` erforderlich.
