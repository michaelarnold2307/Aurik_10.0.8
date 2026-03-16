# Aurik 9.10.51 - Installation Guide

**Version:** 9.10.51  
**Datum:** März 2026  
**Status:** ✅ Production Ready (6312+ Tests grün)

---

## Inhaltsverzeichnis

- [System-Anforderungen](#system-anforderungen)
- [Installation](#installation)
  - [1. Basis-Installation](#1-basis-installation)
  - [2. GPU-Support (Optional)](#2-gpu-support-optional)
  - [3. Entwicklungs-Installation](#3-entwicklungs-installation)
- [Verifizierung](#verifizierung)
- [Troubleshooting](#troubleshooting)
- [Deinstallation](#deinstallation)

---

## System-Anforderungen

### Minimum Requirements

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| **OS** | Linux, Windows 10/11 | Linux (Ubuntu 22.04+) |
| **Python** | 3.10+ | 3.10 |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 Cores (2.5 GHz) | 8+ Cores (3.5 GHz) |
| **Storage** | 10 GB | 20 GB SSD |
| **GPU** | Nicht benötigt (CPU-only) | — |

### Software-Abhängigkeiten

**Python-Bibliotheken** (automatisch installiert):
- PyTorch 2.2.x **+cpu** (CPU-only, kein CUDA erforderlich)
- NumPy, SciPy, Librosa, SoundFile
- onnxruntime (CPUExecutionProvider)
- PyQt5 (GUI)
- tqdm, pyyaml, requests

**System-Bibliotheken** (manuell installieren):
- **Linux:** `libsndfile1`, `ffmpeg`
- **macOS:** `libsndfile`, `ffmpeg` (via Homebrew)
- **Windows:** Keine zusätzlichen Bibliotheken erforderlich

---

## Installation

### 1. Basis-Installation

#### Schritt 1: Repository klonen

```bash
# Clone Repository
git clone https://github.com/yourusername/Aurik_Standalone.git
cd Aurik_Standalone
```

#### Schritt 2: Virtual Environment erstellen

**Linux/macOS:**
```bash
# Virtual Environment erstellen
python3.11 -m venv .venv_aurik

# Aktivieren
source .venv_aurik/bin/activate
```

**Windows:**
```cmd
# Virtual Environment erstellen
python -m venv .venv_aurik

# Aktivieren
.venv_aurik\Scripts\activate
```

#### Schritt 3: Dependencies installieren

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Installiere Aurik Dependencies
pip install -r requirements/requirements.txt

# Optional: Installiere Development-Tools
pip install -r requirements/requirements-dev.txt
```

**requirements.txt Übersicht:**
```txt
torch>=2.10.0
torchaudio>=2.10.0
transformers>=5.1.0
numpy>=2.2.6
scipy>=1.15.3
librosa>=0.11.0
soundfile>=0.13.0
pyyaml>=6.0
tqdm>=4.66.0
pandas>=2.2.0
matplotlib>=3.9.0
```

#### Schritt 4: System-Bibliotheken (Linux)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
```

**Arch Linux:**
```bash
sudo pacman -S libsndfile ffmpeg
```

**Fedora/RHEL:**
```bash
sudo dnf install libsndfile ffmpeg
```

#### Schritt 5: Verifizierung

```bash
# Test Import
python -c "from core.unified_restorer_v2 import UnifiedRestorerV2; print('✅ Aurik installed successfully!')"
```

**Expected Output:**
```
✅ Aurik installed successfully!
```

---

### 2. GPU-Support (Optional)

#### Warum GPU?

**Performance-Gewinn:**
- **CPU:** 3-5x Echtzeit (3min Audio → 10-15min Processing)
- **GPU:** 0.5-1x Echtzeit (3min Audio → 2-3min Processing)

**Empfohlene GPUs:**
- NVIDIA RTX 3090 (24 GB VRAM) - Beste Wahl
- NVIDIA RTX 4090 (24 GB VRAM) - Maximum Performance
- NVIDIA A100 (40/80 GB VRAM) - Professional
- NVIDIA V100 (16/32 GB VRAM) - Server

#### Schritt 1: NVIDIA Driver installieren

**Linux:**
```bash
# Check current driver
nvidia-smi

# Install NVIDIA Driver (if not installed)
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Windows:**
- Download NVIDIA Driver von https://www.nvidia.com/drivers
- Game Ready Driver oder Studio Driver installieren

#### Schritt 2: CUDA Toolkit installieren

**CUDA 12.8 (empfohlen):**

**Linux:**
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 12.8
sudo apt-get install cuda-toolkit-12-8
```

**Windows:**
- Download CUDA Toolkit 12.8 von https://developer.nvidia.com/cuda-downloads
- Installer ausführen (wähle "Custom Installation" → nur CUDA Runtime & Libraries)

#### Schritt 3: PyTorch mit CUDA installieren

```bash
# Aktiviere Virtual Environment
source .venv_aurik/bin/activate  # Linux/macOS
# ODER
.venv_aurik\Scripts\activate     # Windows

# Deinstalliere CPU-Version (falls vorhanden)
pip uninstall torch torchaudio

# Installiere CUDA-Version (CUDA 12.8)
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

#### Schritt 4: Verifizierung

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output (mit GPU):**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

**Expected Output (ohne GPU/CPU-only):**
```
CUDA available: False
GPU: None
```

---

### 3. Entwicklungs-Installation

#### Für Contributors & Entwickler

```bash
# Clone Repository
git clone https://github.com/yourusername/Aurik_Standalone.git
cd Aurik_Standalone

# Virtual Environment
python3.11 -m venv .venv_aurik
source .venv_aurik/bin/activate

# Development Dependencies
pip install -r requirements/requirements-dev.txt

# Install Pre-Commit Hooks
pre-commit install

# Run Tests
pytest tests/ --verbose
```

**requirements-dev.txt:**
```txt
pytest>=8.3.0
pytest-cov>=5.0.0
black>=24.8.0
isort>=5.13.0
flake8>=7.1.0
mypy>=1.11.0
pre-commit>=3.8.0
```

---

## Verifizierung

### Test 1: Import-Test

```bash
python -c "from core.unified_restorer_v2 import UnifiedRestorerV2; print('✅ Import successful')"
```

### Test 2: Quick-Test mit Demo-Audio

```bash
# Generate 3s test audio (1 kHz sine wave)
python -c "
import numpy as np
import soundfile as sf

sr = 48000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
sf.write('test_audio.wav', audio, sr)
print('✅ Test audio created: test_audio.wav')
"

# Restore with Aurik
python -c "
from core.unified_restorer_v2 import UnifiedRestorerV2
import soundfile as sf

audio, sr = sf.read('test_audio.wav')
restorer = UnifiedRestorerV2()
restored = restorer.restore(audio, sr)
sf.write('test_audio_restored.wav', restored, 48000)
print('✅ Restoration successful: test_audio_restored.wav')
"
```

**Expected Output:**
```
📊 Phase 1: Audio-Analyse...
🔊 Phase 0: Subsonic/Ultrasonic Filtering...
🎵 Phase 2: Mechanische Artefakte...
...
✅ Restoration successful: test_audio_restored.wav
```

### Test 3: Vollständige Test-Suite

```bash
# Run all tests (187 tests)
pytest tests/ -v

# Run E2E tests (magic button tests)
pytest tests/test_e2e_magicbutton.py -v -s
```

**Expected:** `187 passed in ~5 minutes`

---

## Troubleshooting

### Problem 1: Import Error - torch

**Symptom:**
```
ImportError: No module named 'torch'
```

**Lösung:**
```bash
pip install torch torchaudio
```

---

### Problem 2: CUDA not available (GPU vorhanden)

**Symptom:**
```python
import torch
print(torch.cuda.is_available())  # False
```

**Ursachen & Lösungen:**

**1. PyTorch CPU-Version installiert:**
```bash
# Check PyTorch version
pip show torch

# Reinstall CUDA version
pip uninstall torch torchaudio
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

**2. NVIDIA Driver nicht installiert:**
```bash
# Check driver
nvidia-smi

# If error: Install driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

**3. CUDA Version Mismatch:**
```bash
# Check CUDA version
nvcc --version

# PyTorch requires CUDA 12.x
# Install CUDA Toolkit 12.8 (siehe oben)
```

---

### Problem 3: libsndfile not found (Linux)

**Symptom:**
```
OSError: cannot load library 'libsndfile.so.1'
```

**Lösung:**
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1

# Arch Linux
sudo pacman -S libsndfile

# Fedora
sudo dnf install libsndfile
```

---

### Problem 4: Out of Memory (CUDA)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Lösungen:**

**1. Reduziere Batch-Size / Chunk-Size:**
```python
# Process in smaller chunks
from core.unified_restorer_v2 import UnifiedRestorerV2
import soundfile as sf

audio, sr = sf.read('large_file.wav')

# Split in chunks (30s)
chunk_size = 30 * sr
audio_chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

restorer = UnifiedRestorerV2()
restored_chunks = [restorer.restore(chunk, sr) for chunk in audio_chunks]
restored = np.concatenate(restored_chunks)

sf.write('output.wav', restored, 48000)
```

**2. Verwende CPU-only:**
```bash
# Force CPU processing
export CUDA_VISIBLE_DEVICES=""

python restore_audio.py
```

**3. Upgrade GPU (mehr VRAM):**
- 8 GB VRAM → 16 GB VRAM (RTX 4080)
- 16 GB VRAM → 24 GB VRAM (RTX 3090/4090)

---

### Problem 5: Slow Processing (CPU-only)

**Symptom:**
```
Processing 3min audio takes 15min (5x realtime)
```

**Lösungen:**

**1. Aktiviere GPU (siehe GPU-Support oben)**

**2. Reduziere Processing Complexity:**
```python
from core.processing_modes import ProcessingConfig, ProcessingMode

# Use FORENSIC mode (minimal processing)
config = ProcessingConfig(
    mode=ProcessingMode.FORENSIC,
    aggressive=0.2,
    denoise_strength=0.1
)

restored = restorer.restore(audio, sr, config=config)
```

**3. Disable Phase 10/11 (3D Enhancement):**
```python
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    enable_phase_10_soundstage=False,  # Skip 3D
    enable_phase_11_binaural=False,    # Skip Binaural
)

restored = restorer.restore(audio, sr, config=config)
```

---

### Problem 6: Test Failures

**Symptom:**
```
pytest tests/ → 15 failed, 172 passed
```

**Lösungen:**

**1. Missing Test Data:**
```bash
# Check test audio files exist
ls audio_examples/  # Should contain test files
```

**2. Model Download Failed:**
```bash
# Models are downloaded on first use
# Check internet connection

# Manual download (if needed)
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('facebook/demucs')
"
```

**3. Environment Issues:**
```bash
# Clean virtual environment
rm -rf .venv_aurik
python3.11 -m venv .venv_aurik
source .venv_aurik/bin/activate
pip install -r requirements/requirements.txt
pytest tests/
```

---

## Deinstallation

### Vollständige Entfernung

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv_aurik

# Remove repository (optional)
cd ..
rm -rf Aurik_Standalone

# Remove CUDA (optional, Linux)
sudo apt-get remove cuda-toolkit-12-8
sudo apt-get autoremove
```

---

## Weitere Informationen

- **Quick Start:** [docs/guides/QUICKSTART_SUPPORT.md](QUICKSTART_SUPPORT.md)
- **User Guide:** [docs/guides/USER_GUIDE.md](USER_GUIDE.md)
- **Python API:** [docs/api/PYTHON_API.md](../api/PYTHON_API.md)
- **Troubleshooting:** [docs/guides/TROUBLESHOOTING.md](TROUBLESHOOTING.md) *(to be created)*

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Installation Guide** | **Status:** Complete
