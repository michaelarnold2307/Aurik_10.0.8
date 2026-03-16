# Quality Metrics Implementation - Aurik 8.0

**Stand:** 14. Februar 2026  
**Status:** ✅ Production-Ready

## Übersicht

Vollständig implementierte Docker-basierte Quality Assessment Plugins für objektive Audio-Qualitätsmessung.

### 🎯 Implementierte Metriken

| Metrik | Type | Docker Image | Größe | Output | Verwendung |
|--------|------|--------------|-------|--------|-----------|
| **CDPAM** | Non-Reference | `cdpam:latest` | 28.3 GB | Single Score (0-100) | Perceptual Quality |
| **DNSMOS** | Non-Reference | `dnsmos:latest` | 22.1 GB | 4 Scores (1-5) | Noise Assessment |
| **NISQA** | Non-Reference | `nisqa:latest` | 9.65 GB | MOS + 4 Dimensions | Broadband Audio |
| **ViSQOL** | Reference-Based | `visqol:latest` | 343 MB | MOS-LQO (1-5) | Speech/Audio Perceptual |

**Total Docker Images:** ~60 GB

---

## 1. CDPAM (Cumulative Distribution of Perceptual Audio Measurements)

### Beschreibung
- **Typ:** Non-reference perceptual quality metric
- **Framework:** PyTorch 2.5.1 (CPU)
- **Output:** Single perceptual score (0-100, higher = better)
- **Use Case:** Allgemeine Audio-Qualitätsbewertung (Musik, Sprache, Effekte)

### Technische Details
```python
# Modell
from cdpam import CDPAM
model = CDPAM(dev='cpu')

# Inference
score = model.forward(audio_tensor)  # [1, 1, samples]
```

### Docker
- **Base:** `python:3.10-slim`
- **Dependencies:** torch 2.5.1+cpu, cdpam, soundfile, numpy
- **Entrypoint:** `/workspace/cdpam_infer.py`
- **Volume Mounts:** `/data/input`, `/data/output`

### Test
```bash
pytest tests/test_cdpam_plugin.py -v
# ✓ Score Range: 0-100
# ✓ Stereo → Mono: Automatic
# ✓ Execution Time: ~8s
```

---

## 2. DNSMOS (Deep Noise Suppression Mean Opinion Score)

### Beschreibung
- **Typ:** Non-reference noise and distortion assessment
- **Framework:** ONNX Runtime 1.23.2
- **Output:** 4 Scores (1-5 MOS scale)
  - `MOS_P808`: Overall quality (P.808)
  - `SIG_P835`: Signal distortion (P.835)
  - `BAK_P835`: Background noise intrusiveness (P.835)
  - `OVRL_P835`: Overall quality (P.835)
- **Use Case:** Noise suppression evaluation, VoIP quality, speech enhancement

### Technische Details
```python
# MEL-Spectrogram Preprocessing
mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, 
                                      hop_length=160, n_mels=120)
mel_db = librosa.power_to_db(mel, ref=np.max) + 80

# ONNX Models
p808_model = onnxruntime.InferenceSession("sig_bak_ovr.onnx")
p835_model = onnxruntime.InferenceSession("model_v8.onnx")
```

### Docker
- **Base:** `python:3.10-slim`
- **Dependencies:** onnxruntime 1.23.2, librosa 0.11.0, soundfile, numpy
- **Pre-trained Models:** 
  - `DNSMOS/sig_bak_ovr.onnx` (P.808)
  - `DNSMOS/model_v8.onnx` (P.835)

### Test
```bash
pytest tests/test_dnsmos_plugin.py -v
# ✓ Scores: MOS_P808, SIG_P835, BAK_P835, OVRL_P835 (all 1-5)
# ✓ Resampling: 16kHz automatic
# ✓ Execution Time: ~4s
```

---

## 3. NISQA (Non-Intrusive Speech Quality Assessment)

### Beschreibung
- **Typ:** Non-reference speech/audio quality metric (TU Berlin)
- **Framework:** PyTorch 2.5.1 (CPU)
- **Output:** MOS + 4 Dimension Scores (1-5)
  - `MOS`: Overall Mean Opinion Score
  - `Noisiness`: Background noise level
  - `Coloration`: Tonal distortions
  - `Discontinuity`: Temporal artifacts
  - `Loudness`: Volume appropriateness
- **Use Case:** Broadband audio quality (music, speech, codecs)

### Technische Details
```python
from nisqa.NISQA_model import nisqaModel

args = {
    'mode': 'predict_file',
    'pretrained_model': '/workspace/weights/nisqa.tar',
    'ms_channel': None,
    'num_workers': 0,
    'bs': 1,
}

model = nisqaModel(args)
scores = model.predict(audio_file)
```

### Docker
- **Base:** `python:3.10-slim`
- **Dependencies:** nisqa (pip), torch 2.5.1+cpu, librosa, soundfile
- **Pre-trained Weights:** `/workspace/weights/nisqa.tar` (1 MB)

### Test
```bash
pytest tests/test_nisqa_plugin.py -v
# ✓ MOS + Dimensions (5 scores total)
# ✓ Score Range: 1.0-5.5
# ✓ Execution Time: ~7s
```

### Hintergrund
NISQA wurde als Ersatz für UTMOSv2 implementiert:
- **UTMOSv2:** Speech-focused (TTS, voice conversion datasets)
- **NISQA:** Broadband audio-optimized (besser für Musik-Restauration)

---

## 4. ViSQOL (Virtual Speech Quality Objective Listener)

### Beschreibung
- **Typ:** Reference-based perceptual quality metric (Google)
- **Framework:** Native C++ Binary (pre-compiled)
- **Output:** MOS-LQO Score (1-5)
  - MOS-LQO: Listening Quality Objective (ITU-T P.862 aligned)
- **Use Case:** ITU-Standard für VoIP, codecs, musik-qualität (reference verfügbar)

### Modi
- **Audio Mode:** Wideband (48 kHz), sensitive to 20 kHz
- **Speech Mode:** Narrowband (8 kHz), optimized for speech

### Technische Details
```bash
# Native ViSQOL Binary
/workspace/visqol \
  --reference_file ref.wav \
  --degraded_file deg.wav \
  --output_debug scores.json \
  --use_speech_mode=false \
  --use_lattice_model=false \
  --similarity_to_quality_model /workspace/model/libsvm_nu_svr_model.txt
```

### Docker
- **Base:** `python:3.10-slim`
- **Dependencies:** libsndfile1, libgomp1 (C++ runtime)
- **Binary:** Pre-compiled ViSQOL v3 (Google)
- **Models:** 
  - `libsvm_nu_svr_model.txt` (classical SVR)
  - `lattice_*.tflite` (deep lattice, optional)

### Test
```bash
pytest tests/test_visqol_plugin.py -v
# ✓ MOS-LQO: 1.0-5.0
# ✓ Reference Required: Yes
# ✓ Execution Time: ~1.5s
```

### Besonderheiten
- **Reference-based:** Benötigt saubere Referenz + degraded Audio
- **Kleinste Image:** 343 MB (nur Binary + Models, kein PyTorch)
- **Schnellste Inferenz:** ~1.5s (native C++ performance)

---

## 📊 Vergleichstabelle

| Metrik | Reference | PyTorch | Inference Zeit | Hauptvorteil |
|--------|-----------|---------|----------------|--------------|
| CDPAM | Nein | Ja | ~8s | Universelle Perzeption |
| DNSMOS | Nein | Nein (ONNX) | ~4s | Noise-spezifisch (4 Scores) |
| NISQA | Nein | Ja | ~7s | Multi-dimensional (5 Scores) |
| ViSQOL | **Ja** | Nein (C++) | ~1.5s | ITU-Standard + Speed |

---

## 🧪 Testing

### Einzelne Metriken
```bash
# CDPAM
pytest tests/test_cdpam_plugin.py -v

# DNSMOS
pytest tests/test_dnsmos_plugin.py -v

# NISQA
pytest tests/test_nisqa_plugin.py -v

# ViSQOL
pytest tests/test_visqol_plugin.py -v
```

### Alle Metriken
```bash
pytest tests/test_*_plugin.py -v
# 4 passed in ~29s
```

### Quick Test Suite
```bash
bash run_tests_quick.sh
# 47 passed (includes all 4 quality metrics)
```

---

## 🔧 Plugin-Architektur

Alle Plugins folgen dem einheitlichen Docker-Pattern:

```python
class QualityMetricPlugin:
    def __init__(self):
        self.docker_image = "metric:latest"
    
    def calculate(self, input_wav: str, output_json: str, **kwargs) -> dict:
        """Run metric in Docker container."""
        # 1. Volume mounts einrichten
        # 2. Docker run mit ENTRYPOINT
        # 3. JSON-Output parsen
        # 4. Dict mit Scores zurückgeben
```

### Volume-Mount-Pattern
```bash
docker run --rm \
  -v /host/input:/data/input \
  -v /host/output:/data/output \
  metric:latest \
  /data/input/audio.wav \
  /data/output/scores.json
```

---

## 📝 Output-Format

Alle Plugins geben standardisierte JSON-Dateien zurück:

### CDPAM
```json
{
  "CDPAM_score": 78.45,
  "reference_file": "test.wav",
  "model_version": "cdpam_v1"
}
```

### DNSMOS
```json
{
  "MOS_P808": 3.82,
  "SIG_P835": 4.12,
  "BAK_P835": 4.45,
  "OVRL_P835": 3.95,
  "reference_file": "test.wav",
  "model_version": "dnsmos_p835"
}
```

### NISQA
```json
{
  "MOS": 3.89,
  "Noisiness": 4.12,
  "Coloration": 3.67,
  "Discontinuity": 4.01,
  "Loudness": 3.95,
  "reference_file": "test.wav",
  "model_version": "nisqa_v1"
}
```

### ViSQOL
```json
{
  "ViSQOL_MOS": 4.23,
  "reference_file": "ref.wav",
  "degraded_file": "deg.wav",
  "model_version": "visqol_v3",
  "mode": "audio"
}
```

---

## 🚀 Verwendung in Aurik

### Backend Integration
```python
from plugins.cdpam_plugin import CDPAMPlugin
from plugins.dnsmos_plugin import DNSMOSPlugin
from plugins.nisqa_plugin import NISQAPlugin
from plugins.visqol_plugin import ViSQOLPlugin

# Non-reference metrics
cdpam = CDPAMPlugin()
scores_cdpam = cdpam.calculate("audio.wav", "cdpam_scores.json")

dnsmos = DNSMOSPlugin()
scores_dnsmos = dnsmos.calculate("audio.wav", "dnsmos_scores.json")

nisqa = NISQAPlugin()
scores_nisqa = nisqa.calculate("audio.wav", "nisqa_scores.json")

# Reference-based metric
visqol = ViSQOLPlugin()
scores_visqol = visqol.calculate(
    "reference.wav", 
    "degraded.wav", 
    "visqol_scores.json",
    mode="audio"
)
```

### Batch Processing
```python
import json
from pathlib import Path

def assess_quality(audio_file: str, output_dir: str):
    """Run all quality metrics on audio file."""
    metrics = {
        "cdpam": CDPAMPlugin(),
        "dnsmos": DNSMOSPlugin(),
        "nisqa": NISQAPlugin()
    }
    
    results = {}
    for name, plugin in metrics.items():
        output = Path(output_dir) / f"{name}_scores.json"
        results[name] = plugin.calculate(audio_file, str(output))
    
    # Aggregate scores
    with open(Path(output_dir) / "aggregate.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results
```

---

## 📦 Docker Image Management

### Build Images
```bash
# CDPAM
docker build -f models/cdpam/Dockerfile.cdpam -t cdpam:latest models/cdpam/

# DNSMOS
docker build -f models/dnsmos/Dockerfile.dnsmos -t dnsmos:latest models/dnsmos/

# NISQA
docker build -f models/nisqa/Dockerfile.nisqa -t nisqa:latest models/nisqa/

# ViSQOL
docker build -f models/visqol/Dockerfile.visqol -t visqol:latest models/visqol/
```

### Check Images
```bash
docker images | grep -E "(cdpam|dnsmos|nisqa|visqol)"
```

### Cleanup Old Images
```bash
docker image prune -f
```

---

## 🔬 Validation & Benchmarks

### CDPAM
- ✅ Validated on LibriSpeech, VCTK, DNS Challenge
- 📊 Correlation with subjective MOS: r=0.91

### DNSMOS
- ✅ Microsoft DNS Challenge 2020/2021 winner
- 📊 P.808 ITU-T compliant

### NISQA
- ✅ TU Berlin research (INTERSPEECH 2021)
- 📊 Broadband audio: r=0.93 with subjective MOS

### ViSQOL
- ✅ Google standard (ITU-T P.863 aligned)
- 📊 Wideband audio: r=0.95 with P.800 subjective tests

---

## 🎓 Referenzen

### CDPAM
- Paper: "CDPAM: Contrastive Learning for Perceptual Audio Metrics"
- Code: https://github.com/pranaymanocha/PerceptualAudio

### DNSMOS
- Paper: "DNSMOS: A non-intrusive perceptual objective speech quality metric" (Microsoft, 2022)
- Code: https://github.com/microsoft/DNS-Challenge

### NISQA
- Paper: "NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction" (TU Berlin, INTERSPEECH 2021)
- Code: https://github.com/gabrielmittag/NISQA

### ViSQOL
- Paper: "ViSQOL v3: An Open Source Production Ready Objective Speech and Audio Metric" (Google, 2020)
- Code: https://github.com/google/visqol
- Standard: ITU-T P.863 (POLQA successor)

---

## ✅ Production Checklist

- [x] CDPAM Docker image built and tested
- [x] DNSMOS Docker image built and tested  
- [x] NISQA Docker image built and tested
- [x] ViSQOL Docker image built and tested
- [x] All plugins passing pytest
- [x] Integrated in `run_tests_quick.sh`
- [x] JSON output format standardized
- [x] Error handling implemented
- [x] Documentation complete

**Status:** 🟢 Production-Ready

---

*Last Updated: 14. Februar 2026*
