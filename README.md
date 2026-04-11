# 🎵 Aurik 9.10.77c — Intelligentes Musik-Restaurierungs- und Rekonstruktionssystem

**Version:** 9.10.77c | **Status:** ✅ Produktionsbereit | **Stand:** März 2026

> Normativer Ist-Stand: `.github/specs/01-08` und `docs/CHANGELOG_HISTORY.md`.

![Tests](https://img.shields.io/badge/tests-6571%2B%20passing-brightgreen)
![Musical Goals](https://img.shields.io/badge/Musical%20Goals-14%2F14-brightgreen)
![Quality MOS](https://img.shields.io/badge/MOS-%E2%89%A54.5%20Weltklasse-brightgreen)
![Materials](https://img.shields.io/badge/Materialien-17%20Typen-blue)
![Phases](https://img.shields.io/badge/Phasen-56-blue)
![DefectTypes](https://img.shields.io/badge/DefectTypes-32-blue)
![CPU-only](https://img.shields.io/badge/Hardware-CPU--only-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

---

## 🎯 Was ist Aurik 9.x.x?

Aurik 9.x.x ist ein **weltweit erstmaliges intelligentes, kontextbewusstes Musik- und
Gesangs-Restaurations-, Reparatur- und Rekonstruktions-Denkersystem**.

Es kombiniert psychoakustisch fundierte DSP, Bayesianische Kausalinferenz,
Gaussianische Prozess-Optimierung und perceptuelle Qualitätsbewertung zu einer
kognitiven Restaurierungs-Intelligenz — für Desktop (Linux & Windows 10/11),
vollständig offline, ohne Cloud- oder Netzwerkabhängigkeiten.

**Aktuelle Ergebnisse (v9.10.77c):**

- ✅ **6571+ Unit-Tests** — grün (zzgl. weitere Test-Suites)
- ✅ **56 Phasen** — Defect-First-Pipeline inkl. SpectralBandGapRepair
- ✅ **17 Materialien** — auto-erkannt (tape, vinyl, shellac, wax_cylinder, wire_recording, lacquer_disc, dat, cd_digital, mp3_low, mp3_high, aac, minidisc, streaming, unknown, …)
- ✅ **14 Musical Goals** — psychoakustisch fundiert, alle Schwellwerte erreicht
- ✅ **PQS MOS ≥ 4.5** — Weltklasse-Qualität
- ✅ **CPU-only** — keine GPU-Pflicht, läuft auf Standard-Desktop-Hardware
- ✅ **GP-Lerngedächtnis** — optimiert sich dauerhaft pro Material und Ära
- ✅ **Zero-Shot-Genre-Erkennung** — Deutscher Schlager ohne vortrainiertes Modell

**Über-SOTA DSP-Algorithmen (v9.x.x — vollständig implementiert):**

| Phase | Legacy (verboten) | Über-SOTA (aktiv) | Referenz |
| --- | --- | --- | --- |
| Phase 03 Denoise | ~~Wiener 1984~~ | **OMLSA + IMCRA** + HarmonicPreservationGuard | Cohen 2002/2003 |
| Phase 09 Crackle | ~~Medianfilter~~ | **RBME + Sparse Bayes** | Cemgil 2006, Bando 2019 |
| Phase 12 Wow/Flutter | ~~YIN~~ | **pYIN probabilistisch** + DTW | Mauch & Dixon 2014 |
| Phase 24 Dropout | ~~AR-Spline~~ | **CQTdiff+ / NMF-β + PGHI** | Moliner 2023, Févotte 2011 |
| Phase 55 Inpainting | ~~Griffin-Lim~~ | **Flow Matching / DiffWave** | Lipman 2023, Bai 2024 |
| Phase 56 BandGap | — | **SpectralBandGapRepair** (HEAD_WEAR) | Roebel 2010 |

**Kognitive Module (v9.x.x — 38 Kernmodule):**

| Modul | Zweck |
| --- | --- |
| `PerceptualEmbedder` | 256-dim psychoakustischer Einbettungsraum (L2-normalisiert) |
| `CausalDefectReasoner` | Bayesianische Kausalinferenz, **34 Kausal-Ursachen** |
| `GPParameterOptimizer` | RBF-GP + UCB + **MOO Pareto-Front** (14 Objectives) |
| `PerceptualQualityScorer` | Gammatone-NSIM + MCD + LUFS + MOS |
| `MusicalGoalsChecker` | **14 musikalische Qualitätsziele** |
| `MediumDetector` | File-ext-aware Tonträgerketten-Erkennung, autoritatives Materialsystem |
| `DefectScanner` | 32 DefectTypes, material-adaptive Material-Priors |
| `TransientDecoupledProcessing` | HPSS-Trennung — Groove-Schutz vor jeder NR |
| `HarmonicPreservationGuard` | CREPE/pYIN → G_floor 0.85 an Harmonik-Bins |
| `PerPhaseMusicalGoalsGate` | Rollback bei kumulativer Degradation (56 Phasen) |
| `EraClassifier` | Ära-Erkennung 1890–2025, GP-Warmstart pro Dekade |
| `GermanSchlagerClassifier` | Zero-Shot 6-Schicht-Ensemble (kein Schlager-Training nötig) |
| `ArtistSignatureStore` | Longitudinaler Klang-Fingerabdruck pro Künstler/Session |
| `MusicalStructureAnalyzer` | SSM-Novelty, Chorus als Inpainting-Referenz |
| `MusicalPhraseContextExtractor` | Beat-Tracking → Phrasen-Kontext für Dropout-Inpainting |
| `UnifiedRestorerV3` | **56-Phasen-Orchestrator** (Defect-First) |
| `FeedbackChain` | Iterative PQS-Qualitätsschleife, max. 5 Iter. |
| `ExcellenceOptimizer` | GP-Pareto-Optimierung, `ExcellenceResult` |
| `EnsembleProcessor` | 3 parallele Ketten (CONSERVATIVE/BALANCED/AGGRESSIVE) |
| `RestorabilityEstimator` | < 5 s Vor-Assessment, Predicted MOS + Score 0–100 |
| `UncertaintyQuantifier` | Konfidenz-Schwellen (0.80/0.50), GP-Rückhaltung |
| `TemporalQualityCoherenceMetric` | MOS-Spanne ≤ 0.30, σ ≤ 0.15 über Zeitachse |
| `AdaptiveGoalThresholds` | Material- und ära-adaptive Schwellwerte pro Restaurierung |
| `GoalApplicabilityFilter` | Deaktiviert physikalisch unmessbare Goals (Mono/Bandbreite) |
| `PhysicalCeilingEstimator` | Shannon-Grenze pro Goal, frühe Terminierung |
| `GoalPriorityProtocol` | 5-stufige Vorranghierarchie bei Pareto-Konflikten |
| `MicroDynamicsEnvelopeMorphing` | 400 ms LUFS-Profil-Korrektur, Savitzky-Golay |
| `EmotionalArcPreservationMetric` | Arousal/Valence Pearson ≥ 0.85/0.80, Klimax-Erhalt |
| `IntroducedArtifactDetector` | ML_HALLUCINATION / NMF_CLICK / SMEARING-Detektion |
| `StemRemixBalancer` | LUFS-korrekter Re-Mix nach getrennter Stem-Verarbeitung |
| `MusikalischerGlobalplanDienst` | Cross-Phase-Globalplan: 13 Ära-Profile × Genre-Modifikatoren, 17 Phase-Adjustments (v9.10.50) |
| `PerceptualAttentionModel` | Salienz-Karte [n_frames × 24 Bark-Bänder] ∈ [0.3, 2.0] |
| `BatchSessionLearner` | GP-Warm-Start von Datei zu Datei (SHA256-Session-ID) |
| `ReferenceAnchorSynthesizer` | 270 MUSDB18-HQ-Ankerpunkte (Ära × Genre × Material) |
| `VocalAIEnhancement` | Stimmtyp-adaptiv (MALE/FEMALE/CHILD/ANDROGYNOUS) |
| `HarmonicLatticeAnalyzer` | Fletcher-Modell, B-Koeff., Partial-Abw. ≤ 3 Cent |
| `StereoAuthenticityInvariant` | Mono-Ära M/S ≥ 0.97, Decca-Wide ∈ [0.25, 0.65] |
| `LyricsGuidedEnhancement` | Wort-zeitgenaue Klangverbesserung via Transkription (§2.36, Pflicht ab v9.10.x); Stimmtyp- und Phonem-adaptiv |

---

## 🧠 Kognitive Orchestrierungsschicht (`denker/`)

`denker/` koordiniert alle 38 Kernmodule als Hochsprachen-Orchestrierungsschicht
und produziert das vollständige `AurikErgebnis` (17 Felder, `@dataclass`).

| Denker | Zuständigkeit |
| --- | --- |
| `TontraegerDenker` | Trägermedium-Erkennung (Vinyl / Tape / CD / Digital) |
| `TontraegerketteDenker` | §6.6-Ketten-Erkennung (bindend ab v9.10.45) |
| `DefektDenker` | Defektanalyse via `CausalDefectReasoner` |
| `StrategieDenker` | Phasenstrategie + RT-Guard (`_3X_RT_LIMIT = 8.0`) |
| `RestaurierDenker` | Vollrestaurierung via `UnifiedRestorerV3` |
| `ReparaturDenker` | Self-contained scipy-Direktreparatur |
| `RekonstruktionsDenker` | Lückenfüllung / Inpainting via `GapReconstructor` |
| `ExzellenzDenker` | 14 Musical Goals + `ExcellenceOptimizer` |

**Entry-Point:** `from denker import restauriere` ·
**Tests:** `tests/unit/test_denker/` (10 Dateien) ·
**Doku:** [`denker/README.md`](denker/README.md)

---

## 🚀 Quick Start

### 🎵 Für Einsteiger — Aurik in 3 Schritten starten

> **Kein Python, kein Terminal notwendig.** Aurik läuft direkt auf Ihrem Desktop.

| Schritt | Aktion | Was passiert |
| --- | --- | --- |
| **1** | **Datei öffnen** — Doppelklick auf `AURIK910.AppImage` (Linux) oder `AURIK910.exe` (Windows) | Das Programm startet. Alle KI-Modelle sind bereits enthalten — keine Internetverbindung nötig. |
| **2** | **Aufnahme laden** — Klick auf **📂 Datei öffnen** oder die Audiodatei ins Fenster ziehen | Aurik erkennt automatisch den Tonträger (Vinyl, Kassette, Shellac …) und analysiert alle Defekte. |
| **3** | **Restaurieren** — Klick auf **📀 Restoration** | Die restaurierte Datei wird im Ordner `output/` neben der Originaldatei gespeichert. |

**Unterstützte Formate:** WAV, FLAC, MP3, AIFF, OGG, M4A, WMA, AAC — Mono & Stereo

**Tastenkurzbefehle:** `A` = Original anhören, `B` = Restauriert anhören, `Leertaste` = Play/Pause

---

### Installation (Entwickler)

```bash
# Clone Repository
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone

# Setup Virtual Environment
python3 -m venv .venv_aurik
source .venv_aurik/bin/activate  # Linux/macOS
# .venv_aurik\Scripts\activate  # Windows

# Install Dependencies
pip install -r requirements/requirements.txt

# Optional: Install ML Plugins (für ML-Hybrid Modes)
bash scripts/install_ml_plugins.sh
```

### GUI starten

```bash
./run_aurik.sh
# alternativ (Legacy-Kompatibilitaet):
python start_aurik_90.py
```

Datei laden → **Magic Button** wählen:

- **💿 Restoration** — originalgetreue Restaurierung
- **🎯 Studio 2026** — Highend-Studio-Sound

### CLI-Nutzung

```bash
# Restoration-Modus
PYTHONPATH=. ./.venv_aurik/bin/python cli/aurik_cli.py \
  --input aufnahme.wav --output restauriert.wav --mode Restoration

# Studio 2026-Modus
PYTHONPATH=. ./.venv_aurik/bin/python cli/aurik_cli.py \
  --input aufnahme.wav --output studio.wav --mode "Studio 2026"

# Optionale Parameter: -q/--quiet
```

**Exit-Codes (CLI):**
0 = Erfolg · 1 = Argumentfehler · 2 = Input fehlt · 3 = Importfehler · 4 = Pipelinefehler ·
5 = Exportfehler · 6 = Resamplingfehler · 7 = Quality-Gate · 8 = P1/P2-Gate ·
9 = Pegelabfall > 2.5 dB · 10 = Pre-Analysis-Fehler

### Python API

```python
from core.unified_restorer_v3 import UnifiedRestorerV3
from core.restoration_config import RestorationConfig, QualityMode, MaterialType
import soundfile as sf

# Load Audio
audio, sr = sf.read('input.wav')

# Configure Processing
config = RestorationConfig(
    quality_mode=QualityMode.BALANCED,
    material_type=MaterialType.VINYL,  # or None for auto-detection
    ml_enabled=True  # Enable ML-Hybrid phases
)

# Initialize Restorer
restorer = UnifiedRestorerV3()

# Process Audio
result = restorer.process(audio, sr, config)

# Save Result
sf.write('output.wav', result.audio, sr)

# Check Quality Metrics
print(f"Quality: {result.quality_score:.2f}")
print(f"Processing Time: {result.processing_time_seconds:.1f}s")
print(f"RT Factor: {result.rt_factor:.2f}×")
```

---

## 📋 Features

### 🎼 Restaurierungs-Pipeline (56 Phasen)

**Pipeline-Reihenfolge (v9.10.45 — kanonisch):**

```text
TransientDecoupledProcessing → RestorabilityEstimator → EraClassifier
→ GermanSchlagerClassifier → MediumDetector → DefectScanner
→ CausalDefectReasoner → UncertaintyQuantifier → GPParameterOptimizer
→ HarmonicPreservationGuard → Phase 01–56 (mit PerPhaseMusicalGoalsGate)
→ IntroducedArtifactDetector → FeedbackChain → TemporalQualityCoherenceMetric
→ PerceptualQualityScorer → ExcellenceOptimizer → MusicalGoalsChecker
→ EmotionalArcPreservationMetric → MicroDynamicsEnvelopeMorphing
→ GPParameterOptimizer.update() → RestorationResult
```

**Defektkorrektur (Phase 01–30):**

- Phase 01: Click Removal · Phase 02: Hum Removal · Phase 03: Denoise (OMLSA)
- Phase 09: Crackle Removal (RBME) · Phase 12: Wow/Flutter Fix (pYIN)
- Phase 24: Dropout Repair (NMF-β+PGHI) · Phase 29: Tape Hiss (OMLSA)
- Phase 30: DC-Offset Removal · + weitere 22 Defekt-Phasen

**Enhancement & Mastering (Phase 31–55):**

- Phase 38: Presence Boost · Phase 39: Air-Band Enhancement (> 12 kHz)
- Phase 40: Loudness-Normierung (EBU R128) · Phase 47: True-Peak-Limiter (−1 dBTP)
- Phase 48: Stereo-Width · Phase 49: Advanced Dereverb (Blind-RIR)
- Phase 55: DiffWave/Flow-Matching-Inpainting · + Instrumental- und Vocal-Phasen

**Neue Phase 56 (v9.10.45):**

- Phase 56: SpectralBandGapRepair — HEAD_WEAR-Defekt, Frequenzband-Lücken

**Instrument-adaptive Phasen (PANNs-aktiviert):**

- Guitar → Phase 44 · Brass → Phase 45 · Drums → Phase 51 · Piano → Phase 52 · Vocals → Phase 42

### 🤖 ML-Plugin-Architektur

**Prinzip:** DSP als Fundament, ML als Erweiterung — immer mit DSP-Fallback:

| Situation | ML-Plugin (primär) | Fallback |
| --- | --- | --- |
| Breites Rauschen | DeepFilterNet v3.II (ONNX, 37 MB) | OMLSA+IMCRA (DSP) |
| Raumrauschen / Reverb | WPE (Nakatani 2010) | nara_wpe → OMLSA (DSP) |
| Stem-Separation Vocals | MDX23C Kim_Vocal_2 (64 MB) | NMF-β |
| Stem-Separation Instrumente | MDX23C Kim_Inst (64 MB) | Energy-Masking |
| Codec-Artefakte | **Apollo** (65 MB ONNX) | Resemble-Enhance |
| Dropout < 50 ms | NMF-β + Sinusoidal (DSP) | Consistent Wiener |
| Dropout 50–999 ms | CQTdiff+ / **Flow Matching** | DiffWave ONNX |
| Pitch-Tracking mono | CREPE full (85 MB) | pYIN (DSP) |
| Pitch-Tracking polyphon | BasicPitch (ONNX) | pYIN Multi-Pitch |
| Audio-Tagging / Genre | PANNs CNN14 (81 KB) | DSP Spectral Fingerprint |
| Bandbreiten-Erweiterung | AudioSR (5,9 GB, lazy) | Sinusoidal+Stoch. |
| Vocos-Vocoder (Synthese) | **Vocos 24 kHz** (52 MB) | HiFi-GAN → PGHI-ISTFT |
| MOS Musik (ohne Referenz) | **CDPAM** (102 MB) | PQS-DSP (Gammatone) |
| MOS Musik (mit Referenz) | **ViSQOL v3 `--audio`** | PQS-DSP |
| Music Understanding | MERT-v1-330M (3,9 GB, lazy) | Harmonicity+Chroma DSP |
| ~~MOS-Schätzung~~ | ~~DNSMOS / NISQA~~ | **⛔ VERBOTEN** für Musik |

**Alle ML-Plugins:** `plugins/` — jedes mit `try/except ImportError` DSP-Fallback.  
**CPU-only:** `providers=["CPUExecutionProvider"]` · Kein CUDA / kein ROCm.  
**Bundled:** Alle primären Modelle lokal gebündelt — kein Download beim ersten Start.

### 🎯 Material-Adaptive Verarbeitung (17 Typen)

**Auto-Detection** via `MediumDetector` (file-ext-aware, DSP + forensische Kettenlogik) — **17 Material-Typen:**

| Material | Hauptdefekte | PQS-Ziel |
| --- | --- | --- |
| `tape` | Dropout, Hiss, Wow/Flutter | MOS ≥ 4.2 |
| `reel_tape` | Print-Through, Hiss, Dropout | MOS ≥ 4.3 |
| `vinyl` | Crackle, Warp, Rille-Distortion | MOS ≥ 4.0 |
| `shellac` | Breites Rauschen, BW ≤ 8 kHz | MOS ≥ 3.8 |
| `wax_cylinder` | Extremrauschen, HF ≤ 5 kHz, Zylinderverzerrung | MOS ≥ 3.5 |
| `wire_recording` | Magnetdraht-Jitter, Frequenz-Dropout | MOS ≥ 3.6 |
| `lacquer_disc` | Riss-Klicken, Rille-Ermüdung, Substrat-Rauschen | MOS ≥ 3.7 |
| `dat` | Jitter, Dropout, ATRAC-Artefakte | MOS ≥ 4.4 |
| `cd_digital` | Clipping, Quantisierungsrauschen | MOS ≥ 4.5 |
| `mp3_low` | Schwere Codec-Artefakte (< 128 kbps) | MOS ≥ 3.9 |
| `mp3_high` | Moderate Codec-Artefakte (≥ 128 kbps) | MOS ≥ 4.2 |
| `aac` | Präsenz-Verlust, Apple-Kompression | MOS ≥ 4.2 |
| `minidisc` | ATRAC-Stufigkeit, HF-Verlust | MOS ≥ 4.0 |
| `streaming` | Variables Bitrate-Profil | MOS ≥ 4.1 |
| `unknown` | Konservative Prior, alle Tier-1 Phasen | MOS ≥ 3.8 |

### 📊 Die 14 Musikalischen Qualitätsziele

Nach jeder Restaurierung werden alle 14 Ziele geprüft (adaptiv via `AdaptiveGoalThresholds`
und `GoalApplicabilityFilter`). Regression in einem Ziel macht das Feature ungültig:

| # | Ziel | Frequenzbereich / Messgröße | Schwellwert |
| --- | --- | --- | --- |
| 1 | **Brillanz** | HF-Klarheit 8–20 kHz | ≥ **0.85** |
| 2 | **Wärme** | Mitten 200–2000 Hz | ≥ **0.80** |
| 3 | **Natürlichkeit** | Artefaktfreiheit | ≥ **0.90** |
| 4 | **Authentizität** | Spektraler Fingerabdruck | ≥ **0.88** |
| 5 | **Emotionalität** | Dynamik, Modulationstiefe | ≥ **0.87** |
| 6 | **Transparenz** | Klangbildtrennung | ≥ **0.89** |
| 7 | **Bass-Kraft** | 20–250 Hz + Virtual Pitch | ≥ **0.85** |
| 8 | **Groove** | Mikro-Timing, DTW ≤ 8 ms RMS | ≥ **0.88** |
| 9 | **Raumtiefe** | Stereobreite, Phantom-Center | ≥ **0.75** |
| 10 | **Timbre-Authentizität** | MFCC-Pearson ≥ 0.95 | ≥ **0.87** |
| 11 | **Tonales Zentrum** | Chroma-Korrelation, kein Key-Shift | ≥ **0.95** |
| 12 | **Mikro-Dynamik** | LUFS-Profil 400 ms, Crest-Faktor | ≥ **0.92** |
| 13 | **Separation-Treue** | SDR ≥ 8 dB / SIR ≥ 12 dB | ≥ **0.82** |
| 14 | **Artikulation** | Attack-Charakter, Transient-Shape | ≥ **0.85** |

**PQS-Metriken** (`PerceptualQualityScorer`):

| Metrik | Minimum | Weltklasse |
| --- | --- | --- |
| MOS | ≥ 3.8 | ≥ 4.5 |
| NSIM | ≥ 0.70 | ≥ 0.90 |
| MCD | ≤ 8.0 dB | ≤ 3.0 dB |
| Spectral Coherence | ≥ 0.60 | ≥ 0.85 |

### ⚡ Verarbeitungs-Modi

**💿 Restoration-Modus** (originalgetreu):

- Chroma-Korrelation ≥ 0.95 · LUFS-Differenz ≤ 1 LU
- Kein Harmonic-Exciter-Material · Authentizität über alles
- `ExcellenceOptimizer(mode="restoration")`: konservative GP-Params
- `MicroDynamicsEnvelopeMorphing` MAX_GAIN = 2.0 LU

**🎯 Studio 2026-Modus** (Highend-Sound):

- PQS MOS ≥ 4.5 · Brillanz ≥ 0.90 · Bass-Kraft ≥ 0.88
- Stem-Separation (MDX23C/BS-RoFormer) → `StemRemixBalancer` → Re-Mix
- `ExcellenceOptimizer(mode="studio2026")`: aggressive Pareto-GP-Params
- 11-stufige Verarbeitungskette bis zum finalen True-Peak-Limiter

**🎵 Genre-Restore-Profile:**

- Schlager: Akkordeon-Charakter erhalten, DeEsser ≤ 45 %, Wärme 0.88
- Klassik: Dereverb deaktiviert, Transienten-Erhalt maximiert
- Jazz: Groove-DTW ≤ 4 ms (Timing heilig), HSI bewahren

---

## 🧪 Testing & Validation

### Test-Suite

```bash
# Alle 7747+ Tests
pytest tests/ --disable-warnings --tb=short

# Unit-Tests (4291+ Tests, schnell)
pytest tests/unit -p no:xdist --timeout=30 --tb=short -q

# Musical Goals
pytest tests/musical_goals tests/unit -q

# Schlager-Klassifikation (≥ 35 Tests)
pytest tests/unit/test_v99_genre_schlager.py -v

# Neue v9.9.9-Module
pytest tests/unit/test_transient_decoupled_processing.py -v
pytest tests/unit/test_harmonic_preservation_guard.py -v
pytest tests/unit/test_per_phase_musical_goals_gate.py -v
pytest tests/unit/test_micro_dynamics_envelope_morphing.py -v
```

**Test-Status:** **7747+ Tests** — alle grün ✅

**Test-Mindestanforderung pro neuem Modul:** ≥ 35 Unit-Tests,
inkl. NaN/Inf-Tests, Bounds-Tests, Mono+Stereo, Edge-Cases, Thread-Safety.

### Ära-Klassifikation & AMRB-Benchmark

```bash
# Vor-Assessment (< 5 s)
python aurik_cli.py --input aufnahme.wav --pre-assess

# AMRB v1.0 (10 Szenarien, OS-Führerschaft ≥ 84.0)
python benchmarks/musical_restoration_benchmark.py

# Kompetitiver Benchmark (vs. iZotope RX 11)
python scripts/competitive_benchmark.py

# Competitive CI-Gate (schneller CI-Run)
pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v

# Competitive Nightly (Spec-robust: n_items ≥ 5 pro Szenario)
AURIK_NIGHTLY_ITEMS=5 pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v
```

---

## 🕰️ Ära-Klassifikation (1890–2025)

| Dekade | Material-Typ | GP-Warmstart NR | Stereo-Invariante |
| --- | --- | --- | --- |
| ≤ 1930 | wax_cylinder / shellac | NR-Stärke ∼ N(0.90, 0.05) | Mono, M/S ≥ 0.97 |
| 1930–1945 | shellac / lacquer_disc | NR-Stärke ∼ N(0.85, 0.07) | Mono |
| 1945–1960 | reel_tape / lacquer_disc | NR-Stärke ∼ N(0.75, 0.08) | Früh-Stereo (Blumlein/Decca) |
| 1960–1970 | tape / vinyl | NR-Stärke ∼ N(0.65, 0.09) | Decca-Wide [0.25, 0.65] |
| ≥ 1970 | tape / vinyl / digital | NR-Stärke ∼ N(0.50, 0.10) | Standard |

`BrillanzMetric` ceiling era-adaptiv: ≤ 1930 → 0.72 · ≤ 1950 → 0.80 · ≥ 1980 → 0.95

---

## 🎶 Genre-Klassifikation (Zero-Shot)

`GermanSchlagerClassifier` erkennt Deutschen Schlager **ohne vortrainiertes Genre-Modell**:

| Schicht | Methode | Schwellwert |
| --- | --- | --- |
| Tier-1 CLAP | 7 gewichtete Prompts (DE+EN) | ≥ 0.26 |
| Tier-2 Akkordeon | Reed-Beating AM 5–15 Hz (Hilbert) | ≥ 0.65 |
| Tier-3 HSI | Chroma Quintenkreis ≤ 2 Schritte | ≥ 0.82 |
| Tier-4 Rhythmus | Oompah/Walzer/Marsch (madmom) | ≥ 0.60 |
| Tier-5 Vokal | SAMPA-Formant-Overlap ä/ö/ü | Tie-Breaker |
| Tier-6 Repetition | SSM MFCC Kosinus ≥ 0.85 | ≥ 0.42 |

Voting: ≥ 3/5 DSP-Schichten + Gesamt ≥ 0.52 → `is_schlager=True`  
Recall ≥ 90 % (mit CLAP) · False-Positive < 5 % · ≤ 4 s/Minute Audio

---

## ⚙️ Technische Details

| Aspekt | Wert |
| --- | --- |
| Interne Sample-Rate | **48 000 Hz** (alle DSP/ML/Metriken) |
| Bit-Tiefe intern | float32, Bereich [−1, 1] |
| Hardware | CPU-only (kein CUDA / kein ROCm) |
| Resampling | Lanczos-4, `scipy.signal.resample_poly`, Kaiser β=14 |
| GP-Gedächtnis | `~/.aurik/gp_memory/<material>.json` (lokal, persistent) |
| Artist-Signaturen | `~/.aurik/artist_signatures/<artist_id>.json` |
| Export-Lautheit | EBU R128: −14 LUFS (Streaming) / −18 LUFS (Archiv) |
| True-Peak-Limit | −1.0 dBTP (ITU-R BS.1770-5) |
| Dithering | POW-r Typ 3 (24→16 bit), Fallback TPDF |
| FeedbackChain | max. 5 Iterationen, Konvergenz ΔMOS < 0.02 |
| PerPhaseGoalsGate | Rollback adaptiv (0.012–0.060), max. 5 Retries |
| Chunk-Verarbeitung | 5/15/60/120 s (defektdichte-adaptiv) |

---

## 📚 Dokumentation

| Dokument | Inhalt |
| --- | --- |
| [docs/INDEX.md](docs/INDEX.md) | Vollständiger Dokumentationsindex |
| [docs/KI-AGENT-INTEGRATION-GUIDE.md](docs/KI-AGENT-INTEGRATION-GUIDE.md) | Richtlinien für KI-Agenten |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Aktueller Projektstand |
| [docs/guides/INSTALLATION.md](docs/guides/INSTALLATION.md) | Installationsanleitung |
| [docs/guides/USER_GUIDE.md](docs/guides/USER_GUIDE.md) | Benutzerhandbuch |
| [CHANGELOG.md](CHANGELOG.md) | Versionshistorie |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | **KI-Programmierrichtlinien (bindend)** |
| [denker/README.md](denker/README.md) | Kognitive Orchestrierungsschicht — Denker-Agenten |

---

## 🙏 Dankeschön

**ML-Modelle & Algorithmen:**

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — Cohen/OMLSA-gestützte Rauschunterdrückung
- [Apollo](https://github.com/Qiuqiu0529/apollo) — Codec-Artefakt-Restaurierung (Zhang 2024)
- [Vocos](https://github.com/hubert-siuzdak/vocos) — Neuronaler Vocoder (MIT, 24 kHz ONNX)
- [WPE / nara_wpe](https://github.com/fgnt/nara_wpe) — Dereverb (Nakatani 2010)
- [MDX23C](https://github.com/ZFTurbo/Music-Source-Separation-Training) — Stem-Separation

**Forschungsgrundlagen:**

- iZotope RX — Kommerzieller Referenz-Standard
- Cohen (2002/2003) — OMLSA/IMCRA
- Févotte & Idier (2011) — NMF-β
- Perraudin et al. (2013) — PGHI
- Fletcher (1964) — Harmonisches Gitter / Inharmonizität

---

## 📜 Lizenz

Aurik 9 steht unter der **Apache-2.0-Lizenz** — siehe [LICENSE](LICENSE).

---

Aurik 9.10.77c — März 2026
