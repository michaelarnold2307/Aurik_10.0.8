# Bandwidth Reconstructor — Komplette Pipeline (Aurik)

Dieser Ordner enthält alles, um ein eigenes BW-Reconstructor-Modell zu
trainieren, nach ONNX zu exportieren und als Aurik-Plugin einzubinden.

## Übersicht

| Datei | Zweck | Wer führt sie aus? |
|---|---|---|
| `generate_defect_training_data.py` | Synthetische BW-Trainingsdaten | Du (einmalig) |
| `train_bw_reconstructor.py` | U-Net-Training (PyTorch) | Du (GPU-Server, ~2h) |
| `export_bw_to_onnx.py` | PyTorch → ONNX | Du (einmalig nach Training) |
| `../plugins/bw_reconstructor_plugin.py` | Aurik-Plugin (ONNX-Inferenz) | Aurik (automatisch) |

## Quickstart

### 1. Trainingsdaten generieren

```bash
# Mit MUSDB18 (empfohlen):
python scripts/generate_defect_training_data.py     --input /data/MUSDB18/train     --output data/bw_defects/train     --duration_hours 20

# Ohne MUSDB18 (rein synthetisch):
python scripts/generate_defect_training_data.py     --output data/bw_defects/synthetic     --duration_hours 10     --synthetic
```

### 2. Modell trainieren

```bash
pip install torch torchaudio torchvision

python scripts/train_bw_reconstructor.py     --data-dir data/bw_defects/train     --epochs 100     --batch-size 32     --lr 1e-3     --output-dir models/bw_reconstructor
```

**Hardware-Empfehlung:** GPU mit ≥8 GB VRAM (RTX 3070/4070 reicht).
Ohne GPU: `--device cpu` aber dann ~10× langsamer.

### 3. Nach ONNX exportieren

```bash
pip install onnx onnxruntime

python scripts/export_bw_to_onnx.py     --checkpoint models/bw_reconstructor/best_model.pt     --output models/bw_reconstructor/bw_reconstructor.onnx
```

### 4. In Aurik nutzen

Das Plugin wird automatisch geladen (lazy, nur bei Bedarf).
Voraussetzung: `bw_reconstructor.onnx` liegt unter `models/bw_reconstructor/`.

```python
from plugins.bw_reconstructor_plugin import BWReconstructorPlugin

plugin = BWReconstructorPlugin(model_dir="models/bw_reconstructor")
plugin.reconstruct(audio, sr=44100)  # → rekonstruiertes Audio
```

Oder direkt aus der Aurik-Pipeline:

```python
from plugins.bw_reconstructor_plugin import get_bw_reconstructor, reconstruct_bandwidth

reconstructor = get_bw_reconstructor()
restored = reconstruct_bandwidth(audio, sr)
```

## Architektur

```
Input Audio (z.B. 8 kHz Bandbreite)
       │
       ▼
┌─────────────────────┐
│   STFT → Mel-Spec   │  256×256 Mel-Bins @ 22.05 kHz
│   (log-Mel, dB)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   U-Net (ONNX)      │  ~50 MB, ~8M Parameter
│   Encoder-Decoder   │  Base channels: 24
│   4-Level Skip-Con  │  Opset 17, CPU-optimiert
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Griffin-Lim       │  Phase-Rekonstruktion (50 Iterationen)
│   (no learned       │
│    vocoder needed)  │
└────────┬────────────┘
         │
         ▼
    Rekonstruiertes Audio (volle Bandbreite)
```

## Modell-Spezifikation

| Eigenschaft | Wert |
|---|---|
| Architektur | U-Net (4-Level Encoder/Decoder) |
| Parameter | ~7.5 Mio. (trainierbar) |
| Input | Mel-Spektrogramm (1×256×256, float32) |
| Output | Mel-Spektrogramm (1×256×256, float32) |
| ONNX-Größe | ~50 MB (FP32) / ~15 MB (INT8 quantisiert) |
| Inferenzzeit (CPU) | ~200 ms pro 6-Sekunden-Segment |
| Opset | 17 |
| Abhängigkeiten | onnxruntime (kein PyTorch zur Laufzeit!) |

## Training-Details

- **Loss:** Kombinierte L1 + Multi-Scale Mel-Spectral-Loss
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing + 5-Epoch-Warmup
- **Augmentation:** Time/Frequency-Masking (SpecAugment)
- **Validation:** Alle 5 Epochen, Early Stopping (Patience=20)
- **Hardware:** ~8 GB VRAM, ~2 Stunden für 100 Epochen auf RTX 4070

## Ohne GPU trainieren

```bash
python scripts/train_bw_reconstructor.py     --data-dir data/bw_defects/synthetic     --epochs 50     --batch-size 8     --device cpu     --output-dir models/bw_reconstructor
```

(CPU-Training dauert ~20 Stunden für 50 Epochen — über Nacht laufen lassen.)

## FAQ

**F: Brauche ich MUSDB18?**
A: Nein. `--synthetic` generiert harmonische + Rausch-Kompositionen,
die für BW-Rekonstruktion ausreichend sind. MUSDB18 verbessert die
Qualität aber deutlich.

**F: Wie groß ist das trainierte Modell?**
A: ~50 MB (FP32 ONNX). Optional INT8-Quantisierung reduziert auf ~15 MB.

**F: Läuft das auf CPU?**
A: Ja. ONNX-Inferenz auf CPU braucht ~200 ms/Segment. Keine GPU nötig.

**F: Kann ich das Modell kommerziell nutzen?**
A: Ja — dein Modell, deine Trainingsdaten, deine Lizenz.
