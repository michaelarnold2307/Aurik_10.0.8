# Aurik 9.x.x — Troubleshooting Guide

**Version:** 9.12.8  
**Datum:** Mai 2026  
**Status:** ✅ Production Ready

---

## Inhaltsverzeichnis

- [Schnellhilfe](#schnellhilfe)
- [Installation & Dependencies](#installation--dependencies)
- [Audio-Verarbeitung](#audio-verarbeitung)
- [GPU & CUDA](#gpu--cuda)
- [Memory & Performance](#memory--performance)
- [Audio-Qualität](#audio-qualität)
- [API & Entwicklung](#api--entwicklung)
- [Testing](#testing)
- [Logs & Debugging](#logs--debugging)

---

## Schnellhilfe

### Häufigste Probleme (Top 5)

| Problem | Lösung | Link |
| --- | --- | --- |
| **ImportError: No module named 'library'** | `pip install -r requirements.txt` | [#1](#1-importerror-no-module-named-library) |
| **CUDA not available** | GPU Driver + CUDA Toolkit installieren | [#6](#6-cuda-not-available-gpu-nicht-erkannt) |
| **RuntimeError: Out of memory (CUDA)** | Batch-Size reduzieren oder CPU-only | [#8](#8-runtimeerror-out-of-memory-cuda) |
| **Audio klingt verzerrt/clipped** | `aggressive` Parameter reduzieren | [#11](#11-audio-klingt-verzerrt-oder-geclippt) |
| **Langsame Verarbeitung** | GPU aktivieren oder CPU-Cores erhöhen | [#9](#9-verarbeitung-ist-sehr-langsam) |

---

## Installation & Dependencies

### #1: ImportError: No module named 'library'

**Symptom:**

```text
ImportError: No module named 'torch'
ImportError: No module named 'librosa'
ImportError: No module named 'soundfile'

```

**Ursache:** Dependencies nicht installiert

**Lösung:**

```bash
# Aktiviere Virtual Environment
source .venv_aurik/bin/activate  # Linux/macOS
.venv_aurik\Scripts\activate     # Windows

# Installiere Dependencies
pip install -r requirements.txt

# Verifiziere Installation
python -c "import torch; print(torch.__version__)"
python -c "import librosa; print(librosa.__version__)"

```

**Alternative (Minimal Install):**

```bash
pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cpu>
pip install numpy librosa soundfile scipy tqdm pyyaml

```

---

### #2: libsndfile not found (Linux)

**Symptom:**

```text
OSError: cannot load library 'libsndfile.so': libsndfile.so: cannot open shared object file

```

**Ursache:** System-Bibliothek `libsndfile` fehlt

**Lösung (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install libsndfile1 libsndfile1-dev

```

**Lösung (Fedora/RHEL):**

```bash
sudo dnf install libsndfile libsndfile-devel

```

**Lösung (Arch):**

```bash
sudo pacman -S libsndfile

```

**Verifizierung:**

```bash
ldconfig -p | grep libsndfile
# Output: libsndfile.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libsndfile.so.1

```

---

### #3: ffmpeg not found

**Symptom:**

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'

```

**Ursache:** ffmpeg nicht installiert (benötigt für MP3/AAC)

**Lösung (Ubuntu/Debian):**

```bash
sudo apt install ffmpeg

```

**Lösung (macOS):**

```bash
brew install ffmpeg

```

**Lösung (Windows):**

1. Download: <https://www.gyan.dev/ffmpeg/builds/>
2. Entpacke nach `C:\ffmpeg`
3. Füge `C:\ffmpeg\bin` zu PATH hinzu

**Verifizierung:**

```bash
ffmpeg -version
# Output: ffmpeg version 4.4.2-0ubuntu0.22.04.1 ...

```

---

### #4: pip install schlägt fehl (SSL-Fehler)

**Symptom:**

```text
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]

```

**Ursache:** Firmen-Proxy oder veraltetes SSL-Zertifikat

**Lösung 1 (Trusted Host):**

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

```

**Lösung 2 (Certificate Update):**

```bash
# macOS
/Applications/Python\ 3.11/Install\ Certificates.command

# Linux
pip install --upgrade certifi

```

**Lösung 3 (Offline Install):**

```bash
# Download Wheels offline, dann:
pip install --no-index --find-links=/path/to/wheels -r requirements.txt

```

---

### #5: Python Version mismatch

**Symptom:**

```text
ERROR: This package requires Python >=3.10

```

**Ursache:** Python < 3.10

**Lösung (Ubuntu/Debian):**

```bash
# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create venv with Python 3.11
python3.11 -m venv .venv_aurik
source .venv_aurik/bin/activate

# Verifiziere Version
python --version
# Output: Python 3.11.5

```

**Lösung (macOS):**

```bash
brew install python@3.11
python3.11 -m venv .venv_aurik
source .venv_aurik/bin/activate

```

---

## GPU & CUDA

### #6: CUDA not available (GPU nicht erkannt)

**Symptom:**

```python
import torch
print(torch.cuda.is_available())  # False

```

**Diagnose:**

```bash
# Check NVIDIA Driver
nvidia-smi

# Check CUDA Version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

```

**Ursache 1:** NVIDIA Driver fehlt

**Lösung:**

```bash
# Ubuntu/Debian
sudo apt install nvidia-driver-535  # or latest

# Reboot
sudo reboot

# Verify
nvidia-smi

```

---

**Ursache 2:** CUDA Toolkit fehlt

**Lösung:**

```bash
# Download CUDA 12.8 Toolkit
# <https://developer.nvidia.com/cuda-downloads>

# Ubuntu 22.04 Example:
wget <https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin>
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget <https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-550.54.15-1_amd64.deb>
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-550.54.15-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version

```

---

**Ursache 3:** PyTorch CPU-only installiert

**Lösung:**

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled PyTorch 2.10.0
pip install torch==2.10.0 torchvision==0.19.0 torchaudio==2.10.0 --index-url <https://download.pytorch.org/whl/cu128>

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # GPU Name

```

---

### #7: CUDA Version Mismatch

**Symptom:**

```text
RuntimeError: CUDA error: incompatible device
RuntimeError: The NVIDIA driver on your system is too old (found version 11020).

```

**Ursache:** CUDA Toolkit (12.8) vs. Driver Version mismatch

**Lösung:**

```bash
# Check Driver Version
nvidia-smi
# Driver Version: 535.xxx (CUDA 12.2+)

# Für CUDA 12.8 benötigt: Driver >= 550.54
# Update Driver:
sudo apt install nvidia-driver-550
sudo reboot

# Verify
nvidia-smi

```

**Alternative:** Downgrade PyTorch zu passendem CUDA

```bash
# Wenn Driver nur CUDA 11.8 unterstützt:
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url <https://download.pytorch.org/whl/cu118>

```

---

## Memory & Performance

### #8: RuntimeError: Out of memory (CUDA)

**Symptom:**

```text
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.70 GiB total capacity)

```

**Ursache:** Audio zu lang, oder zu viele ML-Modelle gleichzeitig

**Lösung 1: Batch-Size reduzieren**

```python
# In backend/denoiser.py (DeepFilterNet)
# Ändere chunk_size:
chunk_size = 30.0  # statt 60.0 Sekunden

```

**Lösung 2: CPU-only Mode**

```bash
# Setze Environment Variable
export CUDA_VISIBLE_DEVICES=""

# Run Aurik
python orchestrator_and_cli.py input.wav output.wav

```

**Lösung 3: GPU Memory freigeben**

```python
import torch
torch.cuda.empty_cache()

```

**Lösung 4: Lazy-Loading prüfen**

```python
# Plugins sollten lazy-loaded sein (nur bei Bedarf)
restorer = UnifiedRestorerV2()  # Lädt NICHT alle Modelle
# Modelle werden erst bei restore() geladen

```

---

### #9: Verarbeitung ist sehr langsam

**Symptom:** 3-Minuten-Audio braucht > 10 Minuten

**Diagnose:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

```

**Ursache 1:** CPU-only (keine GPU)

**Lösung:** GPU aktivieren (siehe [#6](#6-cuda-not-available-gpu-nicht-erkannt))

**Erwartete Performance:**

- **CPU (i7-10700K):** ~3-5x Echtzeit (3min Audio → ~45-60s)
- **GPU (RTX 3090):** ~8-15x Echtzeit (3min Audio → ~12-22s)

---

**Ursache 2:** Zu wenig CPU-Cores

**Lösung:**

```bash
# Check CPU Usage
htop  # Linux
top   # macOS

# Setze Thread-Limit (wenn hyperthreading Probleme macht)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

```

---

**Ursache 3:** I/O Bottleneck (Festplatte langsam)

**Lösung:**

```bash
# Verwende RAMDisk für temp files
sudo mkdir /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk

# Setze Temp Dir
export TMPDIR=/mnt/ramdisk

```

---

### #10: Memory Leak (Speicher wird nicht freigegeben)

**Symptom:** Speicher wächst bei mehreren Files

**Diagnose:**

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / (1024**2):.2f} MB")

```

**Lösung:**

```python
# Explizit GPU Memory freigeben nach jedem File
import torch

for audio_file in audio_files:
    audio, sr = sf.read(audio_file)
    restored = restorer.restore(audio, sr)
    sf.write(output_file, restored, sr)

    # Free Memory
    del audio, restored
    torch.cuda.empty_cache()  # GPU
    gc.collect()               # CPU

```

---

## Audio-Verarbeitung

### #11: Audio klingt verzerrt oder geclippt

**Symptom:** Output Audio hat Clipping oder Distortion

**Diagnose:**

```python
import numpy as np

# Check Clipping
max_amplitude = np.abs(audio).max()
print(f"Max Amplitude: {max_amplitude}")  # Sollte <= 1.0 sein

# Check RMS
rms = np.sqrt(np.mean(audio**2))
print(f"RMS: {20 * np.log10(rms):.2f} dB")

```

**Ursache 1:** `aggressive` zu hoch

**Lösung:**

```python
# Reduziere aggressive Parameter
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.RESTORATION,
    aggressive=0.3  # statt 0.5 (default)
)

```

---

**Ursache 2:** Input Audio bereits geclippt

**Lösung:**

```python
# Normalize Input vor Restoration
audio = audio / np.abs(audio).max()  # Peak-Normalize zu 1.0
audio *= 0.9  # Headroom

restored = restorer.restore(audio, sr)

```

---

**Ursache 3:** Kompression zu aggressiv

**Lösung:**

```python
from core.processing_config import ProcessingConfig

config = ProcessingConfig()
config.compression_ratio = 2.0  # statt 4.0 (sanfter)
config.compression_threshold = -20  # höher (weniger Kompression)

restored = restorer.restore(audio, sr, custom_config=config)

```

---

### #12: Audio klingt zu dumpf (fehlende High-Frequencies)

**Symptom:** Output klingt "gedämpft", keine Brillanz

**Ursache:** Denoise zu aggressiv

**Lösung:**

```python
config = ProcessingConfig()
config.denoise_strength = 0.2  # statt 0.5 (sanfter)

restored = restorer.restore(audio, sr, custom_config=config)

```

**Alternative:** Aktiviere Air & Presence

```python
config = ProcessingConfig()
config.enable_air_presence = True  # Phase 8: +1.5 dB @ 12-20 kHz

restored = restorer.restore(audio, sr, custom_config=config)

```

---

### #13: Audio hat noch Clicks/Crackle (Vinyl)

**Symptom:** Click/Crackle Removal unvollständig

**Lösung 1:** Erhöhe `aggressive`

```python
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.RESTORATION,
    aggressive=0.7  # statt 0.5
)

```

**Lösung 2:** Verwende `STUDIO_2026` Mode (aggressiver)

```python
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.STUDIO_2026  # aggressive=0.8
)

```

**Lösung 3:** Multi-Pass Processing

```python
# Pass 1
restored_pass1 = restorer.restore(audio, sr, mode=ProcessingMode.RESTORATION)

# Pass 2 (auf Output von Pass 1)
restored_pass2 = restorer.restore(restored_pass1, sr, mode=ProcessingMode.RESTORATION)

```

---

### #14: Audio hat DC-Offset

**Symptom:** Waveform ist nicht zentriert um 0

**Diagnose:**

```python
dc_offset = np.mean(audio)
print(f"DC Offset: {dc_offset:.6f}")  # Sollte ~0.0 sein

```

**Lösung:**

```python
# DC Blocker (sollte automatisch in Phase 1.4 sein)
audio = audio - np.mean(audio)

```

**Verify:** Phase 1.4 (DC-Blocker) ist aktiviert

```python
# In core/unified_restorer_v2.py
# Phase 1.4: DC-Blocker sollte ausgeführt werden

```

---

### #15: Audio ist Mono statt Stereo

**Symptom:** Stereo Input → Mono Output

**Ursache:** Stereo-Channels wurden gemergt

**Lösung:**

```python
# Check Input Shape
print(f"Input Shape: {audio.shape}")  # Sollte (samples, 2) sein für Stereo

# Wenn (samples,) → Mono
if audio.ndim == 1:
    # Convert Mono to Stereo (duplicate)
    audio = np.stack([audio, audio], axis=1)

restored = restorer.restore(audio, sr)

```

---

## Audio-Qualität

### #16: Musical Goals Validation schlägt fehl

**Symptom:**

```text
WARNING: Musical Goal 'Clarity' failed: 0.62 < 0.70
WARNING: Musical Goal 'Tonal Balance' failed: 0.58 < 0.70

```

**Ursache:** Input Audio hat schlechte Qualität

**Lösung 1:** Akzeptiere niedrigere Thresholds

```python
# Musical Goals sind nur Warnung, kein Fehler
# Output Audio ist trotzdem verarbeitet

```

**Lösung 2:** Verwende `STUDIO_2026` Mode (aggressivere Enhancement)

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
# Sollte höhere Musical Goals Scores erreichen

```

**Lösung 3:** Disable Musical Goals Validation

```python
# In core/unified_restorer_v2.py
# Kommentiere Phase 9 aus (nicht empfohlen)

```

---

### #17: Transients klingen gedämpft (Drums)

**Symptom:** Drum-Hits haben keinen "Attack"

**Ursache:** Noise Reduction zerstört Transients

**Lösung 1:** Verwende `FORENSIC` Mode (minimal processing)

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.FORENSIC)

```

**Lösung 2:** Aktiviere Transient Sharpening

```python
config = ProcessingConfig()
config.enable_transient_sharpening = True
config.transient_sharpness_factor = 1.5  # Enhance Attack

restored = restorer.restore(audio, sr, custom_config=config)

```

---

### #18: Vocals klingen robotisch

**Symptom:** Stimme hat Artifacts, klingt unnatürlich

**Ursache:** Aggressive Denoise oder Pitch Correction

**Lösung:**

```python
config = ProcessingConfig()
config.denoise_strength = 0.3  # Sanfter
config.enable_vocal_enhancement = False  # Disable Phase 2.2

restored = restorer.restore(audio, sr, custom_config=config)

```

**Alternative:** Verwende `VINTAGE_WARMTH` Mode (erhält Charakter)

```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.VINTAGE_WARMTH)

```

---

## API & Entwicklung

### #19: UnifiedRestorerV2 import schlägt fehl

**Symptom:**

```python
ImportError: cannot import name 'UnifiedRestorerV2' from 'core.unified_restorer_v2'

```

**Lösung:**

```bash
# Check ob File existiert
ls -la core/unified_restorer_v2.py

# Check Python Path
python -c "import sys; print(sys.path)"

# Füge Projekt-Root zu PYTHONPATH hinzu
export PYTHONPATH="${PYTHONPATH}:/path/to/Aurik_Standalone"

```

---

### #20: ProcessingMode Enum nicht gefunden

**Symptom:**

```text
AttributeError: module 'core.unified_restorer_v2' has no attribute 'ProcessingMode'

```

**Lösung:**

```python
# Correct Import
from core.unified_restorer_v2 import UnifiedRestorerV2, ProcessingMode

# NOT:
# from core import unified_restorer_v2
# mode = unified_restorer_v2.ProcessingMode.RESTORATION  # WRONG

```

---

### #21: Custom Config wird nicht angewendet

**Symptom:** Custom Config hat keine Wirkung

**Lösung:**

```python
from core.processing_config import ProcessingConfig

# Create Custom Config
config = ProcessingConfig()
config.aggressive = 0.8
config.denoise_strength = 0.6

# Apply Config (must use custom_config parameter!)
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.RESTORATION,
    custom_config=config  # ← Important!
)

```

---

## Testing

### #22: pytest schlägt fehl (Import Errors)

**Symptom:**

```text
ImportError: No module named 'backend'

```

**Lösung:**

```bash
# Run pytest from project root
cd /path/to/Aurik_Standalone

# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Tests
pytest

```

---

### #23: Test Audio Files nicht gefunden

**Symptom:**

```text
FileNotFoundError: test_audio/vinyl_sample.wav

```

**Lösung:**

```bash
# Erstelle Test-Audio-Directory
mkdir -p test_audio

# Generiere Test-Audio (oder kopiere von samples)
python -c "
import numpy as np
import soundfile as sf

sr = 48000
duration = 3.0
audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
sf.write('test_audio/vinyl_sample.wav', audio, sr)
"

```

---

### #24: Tests laufen zu langsam

**Symptom:** Test Suite braucht > 20 Minuten

**Lösung 1:** Skippe langsame Tests

```bash
pytest -m "not slow"

```

**Lösung 2:** Run parallel

```bash
pip install pytest-xdist
pytest -n 8  # 8 parallel workers

```

**Lösung 3:** Run nur Unit Tests

```bash
pytest -m unit  # Nur schnelle Unit Tests

```

---

## Logs & Debugging

### #25: Keine Logs sichtbar

**Symptom:** Kein Output während Processing

**Lösung:**

```python
# Aktiviere Logging
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.RESTORATION,
    enable_logging=True  # ← Important!
)

```

**Alternative:** Set Logging Level

```python
import logging
logging.basicConfig(level=logging.INFO)

# Oder für mehr Details:
logging.basicConfig(level=logging.DEBUG)

```

---

### #26: Error Messages sind kryptisch

**Symptom:** Keine hilfreichen Fehlermeldungen

**Lösung: Enable Debug Mode**

```python
import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

try:
    restored = restorer.restore(audio, sr)
except Exception as e:
    logging.error(f"Error: {e}")
    traceback.print_exc()  # Vollständiger Stacktrace

```

---

### #27: Processing hängt (keine Fortschrittsanzeige)

**Symptom:** Processing scheint "eingefroren"

**Lösung:**

```python
# Aktiviere Logging für Fortschrittsanzeige
restored = restorer.restore(
    audio, sr,
    enable_logging=True  # Zeigt Phase-Progress
)

```

**Output:**

```text
Phase 0: Subsonic/Ultrasonic Filtering... ✅
Phase 1: Audio Analysis... ✅
Phase 2: Mechanical Artifacts... ✅
Phase 3: ML Noise Reduction... ⏳ (kann länger dauern)

```

---

## Support & Hilfe

### Weitere Hilfe benötigt?

1. **Dokumentation:**
   - [Installation Guide](../guides/INSTALLATION.md)
   - [Configuration Guide](../guides/CONFIGURATION.md)
   - [Python API Reference](../api/PYTHON_API.md)

2. **Community:**
   - GitHub Issues: [github.com/your-repo/aurik/issues]
   - Forum: [community.aurik.audio]

3. **Bug Report:**

    ```bash
    # Sammle System-Info
    python aurik_system_check.py > system_info.txt

    # Attach zu GitHub Issue:
    # - system_info.txt
    # - Input Audio (wenn möglich)
    # - Full Error Log
    # - Steps to reproduce
    ```

4. **Feature Request:**
   - Siehe [Contributing Guide](CONTRIBUTING.md)

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Troubleshooting Guide**
