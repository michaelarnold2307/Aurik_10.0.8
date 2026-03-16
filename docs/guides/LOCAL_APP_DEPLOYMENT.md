# Aurik als lokale Desktop-App ohne Server

## Übersicht

Aurik unterstützt **3 verschiedene Deployment-Modi** - der FastAPI-Server ist **optional**!

---

## Option 1: PyQt5 Desktop-App (EMPFOHLEN) ⭐

### Vorhandene Desktop-App starten

```bash
cd /mnt/1846D15B46D139E8/Aurik_Standalone
source .venv_aurik/bin/activate
python aurik_professional/main.py
```

### Architektur

```
┌─────────────────────────────────────────┐
│   PyQt5 Desktop Window                  │
│  ┌─────────────────────────────────┐    │
│  │  UI Components                   │    │
│  │  - File Browser                  │    │
│  │  - Processing Controls           │    │
│  │  - Real-time Monitoring          │    │
│  └─────────────────────────────────┘    │
│              ↓                           │
│  ┌─────────────────────────────────┐    │
│  │  Direct Python API Calls         │    │
│  │  UnifiedRestorerV2.restore()    │    │
│  │  AdaptiveProcessingPipeline()   │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
           ↓
    ┌──────────────────┐
    │  Audio Files     │
    │  input_audio/    │
    │  output_audio/   │
    └──────────────────┘
```

### Vorteile

✅ **Kein Server nötig** - Direkter Zugriff auf Python-Module  
✅ **Native Performance** - Keine HTTP-Overhead  
✅ **Offline-fähig** - Keine Netzwerkverbindung erforderlich  
✅ **Schnellerer Start** - Keine 2 Prozesse (Backend + Frontend)  
✅ **Einfaches Packaging** - PyInstaller für `.exe` / `.app` / `.deb`

### Nachteile

⚠️ PyQt5-Abhängigkeit (GUI-Framework)  
⚠️ Platform-spezifisches Packaging (Windows/Mac/Linux)

---

## Option 2: CLI / Python-Skript (EINFACHSTE LÖSUNG) ⚡

### Beispiel-Skript

```python
#!/usr/bin/env python3
"""
aurik_cli.py - Kommandozeilen-Interface ohne Server
"""
import sys
import soundfile as sf
from pathlib import Path

# Import Aurik Processing Pipeline
from backend.adaptive_pipeline import AdaptiveProcessingPipeline

def process_file(input_path: str, output_path: str):
    """Verarbeitet Audio-Datei direkt ohne Server"""
    print(f"📂 Lade: {input_path}")
    
    # Audio laden
    with open(input_path, "rb") as f:
        audio_bytes = f.read()
    
    # Pipeline initialisieren
    print("🔧 Starte Processing Pipeline...")
    pipeline = AdaptiveProcessingPipeline()
    
    # Verarbeiten
    result = pipeline.run(
        audio_bytes,
        features={},
        user_profile={},
        reference_audio=None
    )
    
    # Speichern
    audio_out = result["processed_audio"]
    audio_orig, sr = sf.read(input_path)
    sf.write(output_path, audio_out, sr)
    
    print(f"✅ Gespeichert: {output_path}")
    return result

def main():
    if len(sys.argv) < 3:
        print("Usage: python aurik_cli.py <input.wav> <output.wav>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    result = process_file(input_file, output_file)
    
    print(f"\n📊 Verarbeitete Steps: {len(result.get('steps', []))}")
    print(f"📈 Quality Score: {result.get('quality', {})}")

if __name__ == "__main__":
    main()
```

### Verwendung

```bash
# Einzelne Datei
python aurik_cli.py input.wav output.wav

# Batch-Verarbeitung
for file in input_audio/*.wav; do
    python aurik_cli.py "$file" "output_audio/$(basename $file)"
done
```

### Vorteile

✅ **Keine GUI** - Läuft überall (auch headless Server)  
✅ **Scriptable** - Automatisierung / CI/CD-Integration  
✅ **Minimal** - Nur Python + Core-Dependencies  
✅ **Batch-fähig** - Massenverarbeitung einfach

---

## Option 3: Electron Desktop-App (Optional)

### Konzept

Combine React Frontend + Python Backend in einer **nativen Desktop-App**:

```
┌─────────────────────────────────────────────┐
│   Electron Desktop Window                   │
│  ┌─────────────────────────────────────┐    │
│  │   React UI (from frontend/)          │    │
│  │   Läuft in Electron Browser          │    │
│  └─────────────────────────────────────┘    │
│              ↓ IPC                           │
│  ┌─────────────────────────────────────┐    │
│  │   Python Backend (Child Process)     │    │
│  │   AdaptiveProcessingPipeline         │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### Setup (benötigt zusätzliche Arbeit)

```bash
# 1. Electron-App erstellen
mkdir electron_app
cd electron_app
npm init -y
npm install electron electron-builder

# 2. main.js - Electron Entry Point
cat > main.js << 'EOF'
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let backendProcess;

function createWindow() {
  // Starte Python Backend
  backendProcess = spawn('python', [
    '-m', 'uvicorn', 
    'backend.api.rest.api:app',
    '--port', '8000'
  ], {
    cwd: path.join(__dirname, '../')
  });

  // Erstelle Electron Window
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Lade React Frontend
  win.loadURL('http://localhost:8000');
}

app.whenReady().then(createWindow);

app.on('quit', () => {
  if (backendProcess) backendProcess.kill();
});
EOF

# 3. Package als native App
npm run build  # → Erstellt .exe / .app / .AppImage
```

### Vorteile

✅ Native Desktop-App (sieht aus wie normale Software)  
✅ Wiederverwendung des React-Frontends  
✅ Cross-Platform (Windows/Mac/Linux)  
✅ Auto-Update möglich (electron-updater)

### Nachteile

❌ **Größe** - Electron-Apps sind groß (~150-300 MB)  
❌ **Komplexität** - Zusätzliche Packaging-Layer  
❌ **Python-Bundling** - Python muss mit gepackt werden

---

## ⚙️ Cloud-Funktionalität entfernen

Der aktuelle Code verwendet **KEINE Cloud-Services**! Alles läuft lokal:

### Was IST lokal

✅ Audio-Verarbeitung (backend/adaptive_pipeline.py)  
✅ ML-Modelle (in models/ Directory)  
✅ Docker-Container für ML-Plugins (lokal)  
✅ FastAPI Server (localhost:8000)  
✅ React Frontend (localhost:3000)

### Was NICHT existiert

❌ Keine Cloud-API-Calls  
❌ Keine Remote-Model-Loading  
❌ Keine Telemetrie/Analytics  
❌ Keine Lizenz-Server-Checks

### Optionale External Dependencies (alle lokal)

- **Docker** - Für ML-Plugin-Isolation
- **Localhost Ports** - Nur bei Server-Modus (8000, 3000)

---

## 🚀 Deployment-Empfehlungen

### Für Endnutzer (keine Programmierkenntnisse)

**Option 1A: PyQt5 Desktop-App**
```bash
# Einmal installieren
pip install PyQt5

# Jederzeit starten
python aurik_professional/main.py
```

**Oder mit PyInstaller verpacken:**
```bash
pip install pyinstaller
pyinstaller aurik_professional.spec

# Erstellt: dist/AurikProfessional.exe (Windows)
# Doppelklick → App startet ohne Python-Installation
```

### Für Power-User (CLI-Komfort)

Erstelle System-Kommando:

```bash
# Linux/Mac:
echo '#!/bin/bash
cd /path/to/Aurik_Standalone
source .venv_aurik/bin/activate
python -c "
import sys, soundfile as sf
from backend.adaptive_pipeline import AdaptiveProcessingPipeline

with open(sys.argv[1], \"rb\") as f:
    audio_bytes = f.read()
pipeline = AdaptiveProcessingPipeline()
result = pipeline.run(audio_bytes, {}, {}, None)
audio, sr = sf.read(sys.argv[1])
sf.write(sys.argv[2], result[\"processed_audio\"], sr)
print(f\"✓ {sys.argv[2]}\")
" "$1" "$2"
' > /usr/local/bin/aurik
chmod +x /usr/local/bin/aurik

# Verwendung:
aurik input.wav output.wav
```

### Für Entwickler

Behalte den Server-Modus für Development:

```bash
# Terminal 1: Backend API
uvicorn backend.api.rest.api:app --reload --port 8000

# Terminal 2: Frontend Dev Server
cd frontend && npm start

# Browser: http://localhost:3000
```

---

## 📦 Packaging-Optionen

### PyInstaller (PyQt5-App → Native .exe/.app)

```bash
pip install pyinstaller

# Windows
pyinstaller --onefile --windowed --name="Aurik" \
  --add-data="models:models" \
  --add-data="config:config" \
  aurik_professional/main.py

# Mac
pyinstaller --onefile --windowed --name="Aurik" \
  --add-data="models:models" \
  --add-data="config:config" \
  --osx-bundle-identifier=com.aurik.professional \
  aurik_professional/main.py

# Linux
pyinstaller --onefile --name="aurik" \
  --add-data="models:models" \
  aurik_professional/main.py
```

Ergebnis:
- **Windows:** `dist/Aurik.exe` (~50-100 MB)
- **Mac:** `dist/Aurik.app` (~60-120 MB)
- **Linux:** `dist/aurik` (~50-100 MB)

### Nuitka (Kompiliert zu nativer Binary)

```bash
pip install nuitka

python -m nuitka --standalone --follow-imports \
  --enable-plugin=pyqt5 \
  --output-dir=dist \
  aurik_professional/main.py
```

Vorteil: **Schneller** als PyInstaller (echter Compiled Code)

### Electron Builder (React-Frontend mit Python)

```bash
npm install electron-builder --save-dev

# package.json:
{
  "scripts": {
    "build": "electron-builder"
  },
  "build": {
    "extraResources": [
      {
        "from": "../.venv_aurik",
        "to": "python_env"
      },
      {
        "from": "../backend",
        "to": "backend"
      }
    ]
  }
}

npm run build
```

---

## 🎯 Zusammenfassung & Empfehlung

| Methode | Server? | GUI? | Komplexität | Beste für |
|---------|---------|------|-------------|-----------|
| **PyQt5 Desktop** | ❌ Nein | ✅ Ja | Mittel | **Endnutzer (EMPFOHLEN)** |
| **CLI-Skript** | ❌ Nein | ❌ Nein | Niedrig | **Batch / Automation** |
| **FastAPI + React** | ✅ Ja | ✅ Browser | Hoch | **Web-Deployment** |
| **Electron** | ⚠️ Intern | ✅ Ja | Sehr hoch | **Cross-Platform UI** |

### Meine Empfehlung für Sie

**Nutzen Sie die PyQt5 Desktop-App:**

```bash
# 1. Starten:
python aurik_professional/main.py

# 2. Packagen für Distribution:
pyinstaller aurik_professional.spec

# 3. Verteilen:
# → dist/AurikProfessional.exe (Windows)
# → Doppelklick → läuft ohne Installation
```

**Vorteile für Ihre Anforderung:**
- ✅ Keine Server-Prozesse
- ✅ Keine Cloud-Verbindungen
- ✅ Reine Desktop-App
- ✅ Schneller Start
- ✅ Natives Look & Feel

Der FastAPI-Server ist **nur für Web-Deployment** relevant (z.B. wenn Sie Aurik als Web-Service anbieten wollen).

---

## 🔍 Weitere Vereinfachungen

### Backend-only Mode (minimal)

Wenn Sie NUR das Processing benötigen:

```python
# minimal_aurik.py - 10 Zeilen, kein Server
from backend.adaptive_pipeline import AdaptiveProcessingPipeline
import soundfile as sf

audio, sr = sf.read("input.wav")
pipeline = AdaptiveProcessingPipeline()

with open("input.wav", "rb") as f:
    audio_bytes = f.read()

result = pipeline.run(audio_bytes, {}, {}, None)
sf.write("output.wav", result["processed_audio"], sr)
```

### Docker-freier Modus

Falls Sie auch Docker entfernen wollen:

```python
# backend/adaptive_pipeline.py
# Kommentiere Docker-Plugin-Calls aus
# Verwende nur native Python-Plugins
```

Alle ML-Plugins können auch **ohne Docker** laufen (langsamer, aber möglich).

---

## 💡 FAQ

**Q: Warum gibt es überhaupt einen Server?**  
A: Für Web-basierte Deployments / Remote-Zugriff. Für lokale Nutzung ist er optional.

**Q: Kommuniziert Aurik mit der Cloud?**  
A: Nein, 0% Cloud-Abhängigkeit. Alles lokal.

**Q: Kann ich den Server komplett entfernen?**  
A: Ja, löschen Sie einfach `backend/api/` - die Core-Pipeline läuft unabhängig.

**Q: Wie groß wird eine gepackte App?**  
A: PyInstaller: ~80 MB, Electron: ~200 MB (wegen Node.js/Chromium)

**Q: Welche Methode ist am schnellsten?**  
A: CLI-Skript (direkter Python-Call) > PyQt5 > Server-basiert

---

## 📚 Siehe auch

- [Python API Documentation](../api/PYTHON_API.md)
- [PyQt5 Desktop App Guide](DESKTOP_APP.md)
- [CLI Development Guide](../development/CLI.md)
- [Packaging & Distribution](PACKAGING.md)
