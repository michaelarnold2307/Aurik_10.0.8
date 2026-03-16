# Aurik 9 — Spec 08: Architektur, Code-Standards & Distribution

> Softwareschichten, Code-Konventionen, Frontend-Regeln, Plugin-Policy,
> CLI, Distribution (AppImage/NSIS), Out-of-the-Box-Pflicht.

---

## §11 Softwareschichten-Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend  (frontend/)   PyQt5 · Frameless · Dark Theme     │
├─────────────────────────────────────────────────────────────┤
│  CLI        aurik_cli.py  Kommandozeilen-Adapter            │
├─────────────────────────────────────────────────────────────┤
│  API-Schicht  backend/api/rest/   FastAPI · REST + Progress │
├─────────────────────────────────────────────────────────────┤
│  Backend-Core  backend/core/ · plugins/ · dsp/  DSP + ML   │
└─────────────────────────────────────────────────────────────┘
```

**Verbot:** Frontend darf `backend/core/`, `dsp/` oder `plugins/` **nicht** direkt importieren.
Kommunikation nur über `backend/api/` oder Qt-Signals/Slots.

---

## §3 Code-Standards

### §3.1 Numerische Robustheit (PFLICHT)

```python
# Nach jeder numerischen Operation:
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

# Ausgabe-Audio immer clippen:
audio = np.clip(audio, -1.0, 1.0)

# Float-Guard vor Qualitäts-Updates:
if not math.isfinite(score):
    logger.debug("Score ungültig, überspringe Update")
    return
```

**ABSOLUTES VERBOT:** Jede Audio-Ausgabe und jeder Score muss NaN/Inf-frei sein.

### §3.2 Singleton-Pattern (PFLICHT für jedes neue Kernmodul)

```python
import threading
from typing import Optional

_instance: Optional[MyModule] = None
_lock = threading.Lock()

def get_my_module() -> MyModule:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyModule()
    return _instance

def my_convenience_function(audio: np.ndarray, sr: int) -> MyResult:
    return get_my_module().process(audio, sr)
```

**Pflicht-Invariante:** Jeder globale `_instance`-Cache mit `threading.Lock()`.
`_cache = {}` ohne Lock in Produktionscode ist **verboten**.

### §3.3 Docstring-Standard (PFLICHT)

```python
def score_audio(self, reference: np.ndarray, degraded: np.ndarray, sr: int) -> PQSResult:
    """Berechnet perceptuellen Qualitätsscore (referenz-basiert).

    Algorithmus:
        1. Kreuzkorrelationsbasierte Zeitausrichtung
        2. Gammatone-Filterbank (25 Bänder, 50–8000 Hz)
        3. NSIM = SSIM(Gammatone(ref), Gammatone(deg))
        4. MCD = (10/ln10) · √(2·Σᵢ(cᵢ_ref - cᵢ_deg)²)  [dB]
        5. MOS = 1.0 + 4.0·σ((z-0.5)·8)

    Args:
        reference: Referenz-Audio (1D float32/64, normalisiert [-1,1])
        degraded:  Degradiertes Audio (selbe Länge)
        sr:        Sample-Rate (muss 48000 sein)

    Returns:
        PQSResult mit .mos, .nsim, .mcd_db, .spectral_coherence
    """
```

### §3.4 Lazy Imports & Graceful Degradation

```python
def _load_optional_model(model_path: str):
    try:
        import onnxruntime as ort
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except (ImportError, FileNotFoundError):
        logger.debug("ONNX nicht verfügbar, nutze DSP-Fallback")
        return None
# Pflicht: torch-Imports mit +cpu-Suffix, ONNX mit CPUExecutionProvider
```

### §3.5 Logging-Konventionen

```python
logger = logging.getLogger(__name__)

# Nutzer-Meldungen/UI: DEUTSCH
# Code-Kommentare/Docstrings: ENGLISCH
# Log-Meldungen (technisch): ENGLISCH

logger.info("🧠 CausalReasoner: Ursache=%s Konfidenz=%.2f", cause, conf)
logger.info("📊 PQS-Score: MOS=%.2f NSIM=%.3f MCD=%.1f dB", mos, nsim, mcd)
# KEIN print() in Produktionscode
```

### §3.6 Datenklassen für Ergebnisse

```python
from dataclasses import dataclass, field

@dataclass
class MyResult:
    """Immer als @dataclass — niemals als raw dict zurückgeben."""
    primary_metric: float
    metadata: dict[str, float] = field(default_factory=dict)
```

### §3.7 Type-Annotation-Pflicht (ab v9.8)

```python
# PFLICHT für alle public APIs:
def process(self, audio: np.ndarray, sr: int, *, mode: str = "restoration") -> ProcessResult: ...

# VERBOTEN:
def process(self, audio, sr, mode="restoration"): ...  # ❌ kein Type

# mypy.ini: strict = true, disallow_untyped_defs = true
```

### §3.8 SHA256-Ergebnis-Cache (für teure Operationen)

```python
import hashlib, threading

_result_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

def audio_sha256(audio: np.ndarray, sr: int) -> str:
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    return h.hexdigest()[:16]

# Cache-Regeln:
# - Max. 128 Einträge (FIFO-Trim)
# - Immer mit threading.Lock() (thread-sicher)
# - Kein Disk-Cache (nur RAM, Prozess-Leben)
# - Cache-Keys: Modul-Präfix + SHA256 ("panns:abc123", "scan:def456")
```

---

## §10 Verbotene Praktiken

### Code-Qualität

```python
# VERBOTEN:
print(f"Score: {score}")         # → logger.info()
return {"mos": 3.5}              # → return PQSResult(mos=3.5)
if score > 0:                    # NaN-Falle → if math.isfinite(score) and score > 0:
_cache = {}                      # ohne Lock → threading.Lock() Pflicht
```

### Architektur

- Kein direkter `frontend/`-Import in Core-Modulen
- Keine hardcodierten Pfade → stets `pathlib.Path.home() / ".aurik" / ...`
- Kein `from module import *`
- Keine sync-Datei-I/O in Hot-Paths (GP-Gedächtnis nur am Anfang/Ende)
- Keine realen Audio-Dateien in Tests
- Keine Sprach-Metriken (PESQ, DNSMOS, NISQA, STOI) für Musikbewertung

### Verbotene Legacy-Algorithmen als Primärverarbeitung

```
Ephraim & Malah (1984) Wiener-Filter  → Ersatz: OMLSA/IMCRA
Simple Spectral Subtraction            → Ersatz: MMSE-LSA + OMLSA
YIN Pitch-Tracker                      → Ersatz: pYIN / CREPE
Medianfilter-Declicker (primitiv)      → Ersatz: RBME + iterative Konsistenz
AR-Modell ohne spektrale Konsistenz    → Ersatz: NMF-β + Sinusoidal Modeling
```

---

## §10.5 Eingabedatei-Sicherheit (OWASP-konform)

```python
class AudioFileValidator:
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 ** 3   # 10 GB
    MAX_DURATION_HOURS: float = 8.0

    def validate(self, path: pathlib.Path) -> None:
        # 1. Dateigröße ≤ 10 GB
        # 2. Magic-Bytes-Verifikation (WAV: b'RIFF', FLAC: b'fLaC', ...)
        # 3. Länge ≤ 8 Stunden, SR ∈ [8000, 384000]
        # 4. Kanäle 1–2 (> 2 → Downmix)
        # 5. Kein eval() / subprocess mit Metadaten-Inhalt
        # 6. Pfad-Traversal ausgeschlossen: os.path.realpath() vor Zugriff
        # 7. FFmpeg: Dateiname IMMER als Liste (kein Shell-String → Injection-Schutz)
        ...
```

---

## §11.3 Plugin-Policy — bestehende Plugins nutzen, nicht neu schreiben

```
✅ = lokal gebündelt, out-of-the-box, kein Download

# Vocoder / Synthese
plugins/vocos_plugin.py               ✅ PRIMÄRER Vocoder (Vocos 24 kHz ONNX, 52 MB)
plugins/hifigan_plugin.py             ✅ HiFi-GAN (3,6 MB ONNX, Tertiär-Fallback)

# Stem-Separation
plugins/demucs_v4_plugin.py           ✅ MDX23C Kim_Vocal_2/Kim_Inst (2× 64 MB)
plugins/bs_roformer_plugin.py         ✅ BS-RoFormer / Mel-RoFormer (+0.4–0.8 dB SDR)

# Rauschunterdrückung & Dereverb
plugins/deepfilternet_v3_ii_plugin.py ✅ DeepFilterNet v3.II (37 MB, 3 ONNX) — PRIMÄR NR
plugins/sgmse_plugin.py               ✅ SGMSE+ (120 MB ONNX) — PRIMÄR Dereverb/Enhancement
plugins/mp_senet_plugin.py            ✅ MP-SENet 2023 (35 MB ONNX) — Speech/Music Enhancement
plugins/wpe_plugin.py                 ✅ WPE Dereverb (3-Tier: nara_wpe→NumPy→OMLSA)
# VERBOTEN: dccrn_plugin (deprecated — ersetzt durch mp_senet_plugin)

# Codec-Artefakte
plugins/apollo_plugin.py              ✅ Apollo (65 MB ONNX, PRIMÄR Codec-Artefakte)
plugins/resemble_enhance_plugin.py    ✅ Resemble-Enhance (41 MB ONNX, Fallback)

# Audio-Tagging & MOS
plugins/beats_plugin.py               ✅ BEATs iter3 (90 MB ONNX) — PRIMÄR Audio-Tagging +10.7 % mAP
plugins/panns_plugin.py               ✅ PANNs CNN14 (81 KB ONNX) — Fallback zu BEATs
plugins/versa_plugin.py               ✅ VERSA-MOS (45 MB ONNX) — PRIMÄR MOS-Bewertung 2024
# VERBOTEN: cdpam_plugin (Sprach-Corpus, ersetzt durch versa_plugin §4.4)

# Pitch / Formanten
plugins/rmvpe_plugin.py               ✅ RMVPE (26 MB ONNX) — PRIMÄR Pitch-Tracking 2023
plugins/crepe_plugin.py               ✅ CREPE full (85 MB ONNX) — Fallback zu RMVPE
plugins/fcpe_plugin.py                ✅ FCPE (ONNX) — Fallback zu CREPE
plugins/formant_tracker.py            ✅ LPC-Formanten F1–F4 (DSP)

# Inpainting
plugins/flow_matching_plugin.py       ✅ Flow Matching — PRIMÄR Generatives Inpainting (SOTA)
plugins/cqtdiff_plus_plugin.py        ✅ CQTdiff+ — Dropout-Inpainting ≥ 50 ms
plugins/diffwave_plugin.py            ✅ DiffWave (552 KB ONNX) — Fallback

# Ära / Genre / BW-Erweiterung
plugins/era_classifier_plugin.py      EraClassifier (CLAP + DSP-Rolloff)
core/genre_classifier.py              GermanSchlagerClassifier (6-Schicht)
plugins/audiosr_plugin.py             AudioSR BW-Erweiterung (5,9 GB, lazy load)
plugins/matchering_plugin.py          ✅ Reference Mastering (matchering==2.0.6)
```

**Regel:** Vor jeder Neuimplementierung existierende Plugins prüfen (§9.4 Anti-Parallelwelten).

---

## §11.4 Frontend (PyQt5) — bindende Regeln

```python
# PFLICHT: Frameless Window
window.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

# Magic Buttons: border-image, KEIN Text / setIcon:
_img_r = (Path(__file__).parent.parent / "resources" / "restoration.png").as_posix()
self.btn_magic_restoration = QPushButton()
self.btn_magic_restoration.setStyleSheet(f"""
    QPushButton {{
        border-image: url("{_img_r}") 0 0 0 0 stretch stretch;
        border: 3px solid transparent;
        border-radius: 16px;
    }}
    QPushButton:hover  {{ border: 3px solid rgba(118, 75, 162, 0.75); }}
""")
```

**Keyboard-Shortcuts:**

| Shortcut | Aktion |
|---|---|
| Leertaste | Play / Pause |
| `A` | Original hören |
| `B` | Restauriert hören |
| `L` | Lyrics-Timeline-Overlay an/aus |
| `Ctrl+O` | Datei öffnen |
| `Ctrl+S` | Exportieren |
| `Ctrl+R` | Restaurierung (RESTORATION) |
| `Ctrl+Shift+R` | Restaurierung (STUDIO 2026) |
| `Escape` | Verarbeitung abbrechen |

---

## §11.5 CLI (`aurik_cli.py`)

```bash
# Pflicht-Argumente:
--input FILE   --output FILE   --mode {restoration,studio2026}

# Optionale Argumente:
--material MATERIAL   --verbose   --no-goals-check
--pre-assess          # Nur Restorability-Score ausgeben, dann abbrechen
--no-phase-gate       # PMGG deaktivieren (nur Debugging)
--no-transient-decouple  # TDP deaktivieren (nur Debugging)

# Exit-Codes:
# 0 = Erfolg | 1 = Verarbeitungsfehler | 2 = Musical Goal Regression
```

---

## §13 Distribution & Out-of-the-Box-Pflicht

> Aurik 9 muss auf einem frischen Linux- oder Windows-System **ohne Python,
> ohne Terminal, ohne Vorkenntnisse** sofort lauffähig sein.

### Installer-Ziele

| Plattform | Format | Anforderung |
|---|---|---|
| **Linux** | AppImage (`.AppImage`) | Einzeldatei, keine Root-Rechte |
| **Windows 10/11** | NSIS-Installer (`.exe`) | Signiert, optional ohne Admin |

### ML-Modell-Gewichte — 100 % offline nach Installation

```json
// models/manifest.json (Version 2)
{
  "version": 2,
  "models": [
    {
      "name": "apollo",
      "bundled": true,
      "bundled_path": "models/apollo/apollo_model.onnx",
      "sha256": "440c48b110f66ff6d7b86cf5bb77201a302d1592ea6471a9b5b99791b21762ac",
      "size_bytes": 67713684,
      "required": false,
      "fallback": "resemble_enhance_onnx"
    }
  ]
}
```

**Out-of-the-Box-Garantie:**
- Kein Download beim ersten Start
- Kein Download nach der Installation
- Alle SOTA-Upgrade-Modelle lokal gebündelt in Programmierphase vorab integriert
- `sota_upgrade`-Feld: nur Entwickler-Metadaten (löst **keinen** Laufzeit-Download aus)
- SHA256-Verifikation für jedes gebündelte Modell Pflicht

### Requirements — nur verifizierte PyPI-Pakete

```
# VERBOTEN (existieren nicht auf PyPI oder falsche Version):
dccrn==0.1.0          diffwave==1.0.0
hifi-gan==0.1.1       nisqa==1.0.0

# PyTorch IMMER CPU-only:
torch==2.2.2+cpu  --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
# Vor jedem Release:
python -m pip install --dry-run -r requirements/requirements_aurik.txt
# → 0 Fehler, 0 nicht gefundene Pakete
```

### §9.1 Checkliste neue Kernmodule

```
□ Datei in backend/core/ angelegt mit Singleton-Pattern (§3.2)
□ Thread-safe: threading.Lock() + Double-Checked Locking
□ Vollständige PEP 484 Type-Annotations (§3.7)
□ NaN/Inf-Schutz in JEDER numerischen Ausgabefunktion
□ @dataclass-Ergebnisse (kein raw dict)
□ logger(__name__) — kein print()
□ ≥ 35 Unit-Tests in tests/unit/test_v<x>_<name>.py
□ Musical Goals (alle 14): kein Ziel nach dem Modul schlechter
□ GrooveMetric: kein Timing-Flattening (DTW ≤ 8 ms RMS)
□ SOFT_SATURATION: nicht als CLIPPING fehldetektiert
□ Beide Modi (restoration + studio2026) getestet
□ Out-of-the-Box-Pflicht: DSP-Fallback für alle Plugin-Imports
□ torch-Imports: +cpu-Suffix, CPUExecutionProvider
□ models/manifest.json: neues Modell eingetragen (sha256 + bundled_path + fallback)
□ scripts/verify_requirements.sh fehlerfrei
□ Alle 6312+ bestehenden Tests weiterhin grün
```

### §9.4 Anti-Parallelwelten-Workflow (vor jeder Implementierung)

```
1. grep -r "<Funktionsname>" backend/core/ plugins/ dsp/ → 0 Treffer?
2. Kein Plugin mit gleicher Funktionalität in plugins/?
3. Keine Phase mit gleicher DSP-Logik in backend/core/phases/?
4. Kein Modul in backend/core/ mit ähnlichem Klassenname?
5. Erst wenn alle 4 Checks negativ → Neuentwicklung starten
6. Prüfentscheidung in CHANGELOG.md dokumentieren
```

### §9.7 Performance-Optimierungen (Pflicht)

```python
# §9.7.1: SHA256-Cache für DefectScanner + PANNs (§3.8-Muster, max. 128 Einträge)
# §9.7.2: Parallele Eingangs-Analyse (MediumClassifier + EraClassifier + GenreClassifier)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    fut_mc  = pool.submit(_run_medium_classifier, audio, sr)
    fut_era = pool.submit(_run_era_classifier,    audio, sr)
    fut_sc  = pool.submit(_run_genre_classifier,  audio, sr)

# §9.7.3: Phasen-adaptive PMGG-Sample-Dauer
PHASE_SAMPLE_DURATIONS = {
    "phase_30": 1.5, "phase_05": 1.5, "phase_02": 2.0,
    "phase_15": 1.5, "phase_11": 1.5, "phase_18": 2.0,
}  # alle anderen: 5.0 s Standard

# §9.7.4: Modell-Warmup im Hintergrund (2 s Verzögerung nach App-Start)
threading.Thread(target=_warmup_models_background, daemon=True, name="AurikWarmup").start()
```
