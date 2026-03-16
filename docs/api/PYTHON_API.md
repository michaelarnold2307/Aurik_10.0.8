# Aurik 8.0 - Python API Reference

**Version:** 8.0.0  
**Datum:** 13. Februar 2026  
**Status:** ✅ Production Ready

---

## Inhaltsverzeichnis

- [Schnellstart](#schnellstart)
- [Kernklassen](#kernklassen)
  - [UnifiedRestorerV2](#unifiedrestorerv2)
  - [ProcessingMode](#processingmode)
  - [ProcessingConfig](#processingconfig)
- [Processing Modes](#processing-modes)
- [Erweiterte Features](#erweiterte-features)
- [Code-Beispiele](#code-beispiele)
- [Fehlerbehandlung](#fehlerbehandlung)
- [Performance-Optimierung](#performance-optimierung)

---

## Schnellstart

### Einfachste Verwendung

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
import soundfile as sf

# Audio laden
audio, sr = sf.read('input.wav')

# Restorer initialisieren
restorer = UnifiedRestorerV2()

# Audio restaurieren (vollautomatisch)
restored = restorer.restore(audio, sr)

# Speichern
sf.write('output.wav', restored, sr)
```

### Mit Processing Mode

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from core.processing_modes import ProcessingMode
import soundfile as sf

# Audio laden
audio, sr = sf.read('old_recording.wav')

# Restorer mit Mode initialisieren
restorer = UnifiedRestorerV2()

# Verschiedene Modi verfügbar:
# - ProcessingMode.RESTORATION (Default: Authentizität bewahren)
# - ProcessingMode.STUDIO_2026 (Modern: 3D, Air & Presence)

restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
sf.write('output_studio.wav', restored, 48000)
```

---

## Kernklassen

### UnifiedRestorerV2

**Hauptklasse für Audio-Restauration und Enhancement.**

#### Konstruktor

```python
UnifiedRestorerV2()
```

**Parameter:**
- Keine erforderlich (vollständig adaptiv)

**Rückgabewert:**
- `UnifiedRestorerV2` Instanz

**Beispiel:**
```python
restorer = UnifiedRestorerV2()
```

---

#### restore()

**Hauptmethode für Audio-Restauration und Enhancement.**

```python
restore(
    audio: np.ndarray,
    sr: int,
    mode: Union[str, ProcessingMode] = ProcessingMode.RESTORATION,
    config: Optional[ProcessingConfig] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    enable_logging: bool = False,
    **kwargs
) -> np.ndarray
```

**Parameter:**

| Parameter        | Typ                    | Default                                             | Beschreibung                   |
|------------------|------------------------|-----------------------------------------------------|--------------------------------|
| `audio`          | `np.ndarray`           | **Required**                                        | Input Audio (mono oder stereo) |
| `sr`             | `int`                  | **Required**                                        | Sample Rate (Hz)               |
| `mode`           | `str`/`ProcessingMode` | `RESTORATION`  | Processing Mode (siehe [Processing Modes](#processing-modes))        |
| `config`         | `ProcessingConfig`     | `None`                                              | Custom Configuration (überschreibt Mode-Defaults) |
| `input_file`     | `str`                  | `None`         | Input-Dateiname (für Logging/Metadaten)                             |
| `output_file`    | `str`                  | `None`         | Output-Dateiname (für Logging)                                      |
| `enable_logging` | `bool`                 | `False`        | ProcessingLogger aktivieren (detaillierte Metriken)                 |

**Rückgabewert:**
- `np.ndarray`: Restauriertes/Enhanced Audio (48 kHz, float32)

**Audio Format:**
- **Input:** Mono (`(samples,)`) oder Stereo (`(samples, 2)`)
- **Output:** Immer 48 kHz (Professional Audio Standard)
- **Dtype:** `float32`, Range: `[-1.0, 1.0]`

**Beispiel:**
```python
# Einfach
restored = restorer.restore(audio, sr)

# Mit Mode
restored = restorer.restore(audio, sr, mode='studio_2026')

# Mit Logging
restored = restorer.restore(
    audio, sr, 
    mode=ProcessingMode.RESTORATION,
    input_file='old_recording.wav',
    enable_logging=True
)

# Mit Custom Config
from core.processing_modes import ProcessingConfig

config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    denoise_strength=0.4,  # Weniger aggressiv
    aggressive=0.3,        # Konservative Restauration
)
restored = restorer.restore(audio, sr, config=config)
```

---

### ProcessingMode

**Enum für vordefinierte Processing-Modi.**

```python
from core.processing_modes import ProcessingMode

class ProcessingMode(Enum):
    RESTORATION = "restoration"
    STUDIO_2026 = "studio_2026"
    FORENSIC = "forensic"
    VINTAGE_WARMTH = "vintage_warmth"
    ARCHIVAL = "archival"
```

**Modi:**

| Mode             | Zweck                | Aggressive | Denoise | Comp. Ratio | Ideal für                          |
|------------------|----------------------|------------|---------|-------------|------------------------------------|
| `RESTORATION`    | Authentisch bewahren | 0.5        | 0.3     | 3.0         | Vinyl, Tape, historische Aufnahmen |
| `STUDIO_2026`    | Modern & 3D          | 0.8        | 0.5     | 3.5         | Remastering, Commercial Release    |


**Beispiel:**
```python
# String-basiert
restored = restorer.restore(audio, sr, mode='restoration')

# Enum-basiert (empfohlen)
restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
```

---

### ProcessingConfig

**Custom Configuration für erweiterte Kontrolle.**

```python
from core.processing_modes import ProcessingConfig, ProcessingMode

config = ProcessingConfig(
    mode: ProcessingMode,
    aggressive: float = None,
    denoise_strength: float = None,
    compression_ratio: float = None,
    enable_de_esser: bool = None,
    enable_vocaI_enhancement: bool = None,
    enable_instrumental_enhancement: bool = None,
    enable_phase_10_soundstage: bool = None,
    enable_phase_11_binaural: bool = None,
)
```

**Parameter:**

| Parameter                         | Typ              | Default       | Range       | Beschreibung                        |
|-----------------------------------|------------------|---------------|-------------|-------------------------------------|
| `mode`                            | `ProcessingMode` | **Required**  |      -      | Basis-Mode                          |
| `aggressive`                      | `float`          | Mode-Default  | `0.0 - 1.0` | Restaurations-Aggressivität         |
| `denoise_strength`                | `float`          | Mode-Default  | `0.0 - 1.0` | Noise Reduction Stärke              |
| `compression_ratio`               | `float`          | Mode-Default  | `1.0 - 10.0`| Dynamik-Kompression                 |
| `enable_de_esser`                 | `bool`           | `True`        |      -      | De-Esser (Sibilanz-Reduktion)       |
| `enable_vocal_enhancement`        | `bool`           | `auto`        |      -      | Vocal Enhancement (Phase 2.2)       |
| `enable_instrumental_enhancement` | `bool`           | `auto`        |      -      | Instrumental Enhancement (Phase 2.3)|
| `enable_phase_10_soundstage`      | `bool`           | Mode-specific |      -      | 3D Soundstage (nur STUDIO_2026)     |
| `enable_phase_11_binaural`        | `bool`           | Mode-specific |      -      | Binaural & Emotional                |

**Beispiel:**
```python
# Conservative Restoration (für fragile Aufnahmen)
config = ProcessingConfig(
    mode=ProcessingMode.RESTORATION,
    aggressive=0.2,           # Sehr vorsichtig
    denoise_strength=0.15,    # Minimales Denoising
    compression_ratio=1.5,    # Kaum Kompression
)
restored = restorer.restore(audio, sr, config=config)

# Aggressive Studio-Enhancement
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    aggressive=0.9,           # Maximum Enhancement
    denoise_strength=0.6,     # Starkes Denoising
    compression_ratio=4.0,    # Broadcast-Dynamik
    enable_phase_10_soundstage=True,  # 3D aktiviert
    enable_phase_11_binaural=True,    # Binaural aktiviert
)
restored = restorer.restore(audio, sr, config=config)
```

---

## Processing Modes

### 1. RESTORATION (Default)

**Zweck:** Authentische Restauration mit Charakter-Preservation

**Charakteristika:**
- ✅ Erhält Analog-Charakter (Vinyl-Wärme, Tape-Saturation)
- ✅ Konservative Noise Reduction (keine "Digital Sound" Artifacts)
- ✅ Schützt musikalische Nuancen (Breaths, Room Tone)
- ✅ Balance zwischen Clean und Natural

**Parameter:**
```python
aggressive = 0.5
denoise_strength = 0.3
compression_ratio = 3.0
```

**Verwendung:**
```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.RESTORATION)
```

**Ideal für:**
- Vinyl-Transfers
- Tape-Digitalisierung
- Historische Aufnahmen (1920s-1990s)
- Jazz, Blues, Classical

---

### 2. STUDIO_2026 (Modern)

**Zweck:** Modern Mastering mit 3D Immersion & Air

**Charakteristika:**
- ✅ Phase 8: Air & Presence Enhancement (12-20 kHz sparkle)
- ✅ Phase 10: Soundstage Depth (3-Layer Spatial Processing)
- ✅ Phase 11: Binaural Processing (HRTF für Kopfhörer)
- ✅ Emotional Resonance Enhancement (Warmth, Richness, Air)

**Parameter:**
```python
aggressive = 0.8
denoise_strength = 0.5
compression_ratio = 3.5
enable_phase_10_soundstage = True
enable_phase_11_binaural = True
```

**Verwendung:**
```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.STUDIO_2026)
```

**Ideal für:**
- Remastering für Streaming (Spotify, Apple Music)
- Commercial Releases
- Pop, Rock, Electronic
- Immersive Audio (Dolby Atmos Vorbereitung)

---

### 3. FORENSIC (Analyse)

**Zweck:** Minimale Veränderung für forensische Analyse

**Charakteristika:**
- ✅ Nur kritische Defekte entfernen (Clicks, Clipping)
- ✅ Keine künstliche Enhancement
- ✅ Beweis-taugliche Verarbeitung (Chain of Custody)
- ✅ Transparency-Log (Processing-Steps dokumentiert)

**Parameter:**
```python
aggressive = 0.2
denoise_strength = 0.1
compression_ratio = 1.5
```

**Verwendung:**
```python
restored = restorer.restore(
    audio, sr, 
    mode=ProcessingMode.FORENSIC,
    enable_logging=True  # Wichtig für Beweisführung
)
```

**Ideal für:**
- Gerichtsverfahren (Stimmanalyse)
- Police/FBI Aufnahmen
- Wiretap Recordings
- Voice Authentication

---

### 4. VINTAGE_WARMTH (Analog)

**Zweck:** Preserve & Enhance Analog-Charakter

**Charakteristika:**
- ✅ Sanftes Denoising (Rauschen als "Ambience")
- ✅ Harmonic Exciter (Tube-Style Saturation)
- ✅ Vinyl-Crackle-Preservation (wenn musikalisch)
- ✅ Tape-Saturation-Emulation

**Parameter:**
```python
aggressive = 0.6
denoise_strength = 0.25
compression_ratio = 2.5
```

**Verwendung:**
```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.VINTAGE_WARMTH)
```

**Ideal für:**
- Lo-Fi Hip-Hop
- Indie/Alternative Music
- Vintage Sound Preservation
- Analog-Aesthetic

---

### 5. ARCHIVAL (Konservativ)

**Zweck:** Langzeitarchivierung mit minimaler Veränderung

**Charakteristika:**
- ✅ Nur strukturell schädliche Defekte entfernen
- ✅ Original-Dynamik bewahren
- ✅ Keine künstliche Enhancement
- ✅ High Bit-Depth Output (24-bit, 48 kHz)

**Parameter:**
```python
aggressive = 0.3
denoise_strength = 0.2
compression_ratio = 2.0
```

**Verwendung:**
```python
restored = restorer.restore(audio, sr, mode=ProcessingMode.ARCHIVAL)
```

**Ideal für:**
- Museen & Bibliotheken
- Radio-Archives (BBC, ORF, ARD)
- Historical Preservation
- Master-Archive

---

## Erweiterte Features

### 1. Processing mit Logging

**Problem:** Wie wurden Parameter gewählt? Was wurde verändert?

**Lösung:** ProcessingLogger für vollständige Transparency

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
import soundfile as sf

# Restorer mit Logging
restorer = UnifiedRestorerV2()

# Restore mit Logging
audio, sr = sf.read('input.wav')
restored = restorer.restore(
    audio, sr,
    mode='restoration',
    input_file='input.wav',
    output_file='output.wav',
    enable_logging=True  # Aktiviere Logging
)

# Access Logger (wenn verfügbar)
if hasattr(restorer, 'logger') and restorer.logger:
    trace = restorer.logger.trace
    print(f"SNR Improvement: {trace.overall_snr_improvement:.1f} dB")
    print(f"THD Reduction: {trace.overall_thd_reduction:.1f}%")
    print(f"Processing Time: {trace.total_processing_time_sec:.1f}s")
    
    # Save Report
    restorer.logger.save_trace('processing_report.json')

# Ausgabe speichern
sf.write('output.wav', restored, 48000)
```

**Log-Format:**
```json
{
  "session_id": "20260213_143022_abc123",
  "input_file": "input.wav",
  "processing_mode": "restoration",
  "sample_rate": 48000,
  "steps": [
    {
      "phase": "Phase 1F: Declipping",
      "module": "DeclipperSemantic",
      "snr_improvement": 2.3,
      "thd_reduction": 15.2,
      "processing_time_ms": 342
    }
  ],
  "overall_snr_improvement": 12.8,
  "overall_thd_reduction": 58.3,
  "total_processing_time_sec": 8.5
}
```

---

### 2. Batch Processing

**Problem:** Viele Dateien restaurieren (z.B. ganzes Album)

**Lösung:** Batch-API mit Progress-Tracking

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

restorer = UnifiedRestorerV2()

# Liste von Input-Dateien
input_dir = Path('input_album')
output_dir = Path('output_album')
output_dir.mkdir(exist_ok=True)

files = list(input_dir.glob('*.wav'))

for file in tqdm(files, desc="Restoring Album"):
    # Load
    audio, sr = sf.read(file)
    
    # Restore
    restored = restorer.restore(audio, sr, mode='restoration')
    
    # Save
    output_path = output_dir / file.name
    sf.write(output_path, restored, 48000)
    
print(f"✅ {len(files)} Tracks restored!")
```

---

### 3. Musical Goals Validation

**Problem:** Wie gut ist die Restauration?

**Lösung:** Musical Goals Checker (12 Metriken)

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from core.musical_goals_checker_v2 import MusicalGoalsCheckerV2
import soundfile as sf

# Restore
restorer = UnifiedRestorerV2()
audio, sr = sf.read('input.wav')
restored = restorer.restore(audio, sr, mode='studio_2026')

# Validate Musical Goals
checker = MusicalGoalsCheckerV2()
goals = checker.measure_all(restored, 48000)

print("🎯 Musical Goals V2.1 (12 Metriken):")
print(f"1. Tonal Balance: {goals['tonal_balance']:.2f} / 1.00")
print(f"2. Clarity: {goals['clarity']:.2f} / 1.00")
print(f"3. Natural Dynamics: {goals['natural_dynamics']:.2f} / 1.00")
print(f"4. Spatial Presence: {goals['spatial_presence']:.2f} / 1.00")
print(f"5. Defect Absence: {goals['defect_absence']:.2f} / 1.00")
print(f"6. Listening Comfort: {goals['listening_comfort']:.2f} / 1.00")
print(f"7. Authenticity: {goals['authenticity']:.2f} / 1.00")
print(f"8. Transparency: {goals['transparency']:.2f} / 1.00")
print(f"9. Air & Presence: {goals['air_presence']:.2f} / 1.00")
print(f"10. Harmonic Richness: {goals['harmonic_richness']:.2f} / 1.00")
print(f"11. Binaural Quality: {goals['binaural_quality']:.2f} / 1.00")
print(f"12. Emotional Depth: {goals['emotional_depth']:.2f} / 1.00")

# Check if all goals met
all_goals_met = all(v >= 0.70 for v in goals.values())
print(f"\n{'✅' if all_goals_met else '⚠️'} All Goals Met: {all_goals_met}")
```

---

## Fehlerbehandlung

### Typische Fehler & Lösungen

#### 1. Invalid Audio Input

```python
try:
    restored = restorer.restore(audio, sr)
except ValueError as e:
    print(f"❌ Invalid Audio: {e}")
    # Lösung: Prüfe Audio-Format
    print(f"Audio shape: {audio.shape}")
    print(f"Audio dtype: {audio.dtype}")
    print(f"Audio range: [{audio.min():.2f}, {audio.max():.2f}]")
```

**Häufige Ursachen:**
- Audio ist `None` oder leer
- Sample Rate ist 0 oder negativ
- Audio ist integer statt float (use `audio.astype(np.float32) / 32768.0`)
- Audio ist außerhalb [-1, 1] Range (use `np.clip(audio, -1.0, 1.0)`)

---

#### 2. Sample Rate Mismatch

```python
# Problem: Audio ist 22050 Hz, aber sr=44100 angegeben
audio, sr_actual = sf.read('input.wav')  # sr_actual = 22050
restored = restorer.restore(audio, 44100)  # ❌ FALSCH

# Lösung: Verwende tatsächliche Sample Rate
restored = restorer.restore(audio, sr_actual)  # ✅ KORREKT
```

---

#### 3. Out of Memory (große Dateien)

```python
import soundfile as sf
import numpy as np

def restore_large_file(input_file, output_file, chunk_size_sec=30):
    """Restore large file in chunks (streaming)."""
    restorer = UnifiedRestorerV2()
    
    # Read file info (ohne laden)
    info = sf.info(input_file)
    sr = info.samplerate
    
    # Process in chunks
    with sf.SoundFile(input_file) as f_in:
        with sf.SoundFile(output_file, 'w', sr, f_in.channels) as f_out:
            chunk_size = chunk_size_sec * sr
            
            while True:
                # Read chunk
                audio = f_in.read(chunk_size)
                if len(audio) == 0:
                    break
                
                # Restore (adapt internal chunking if needed)
                restored = restorer.restore(audio, sr)
                
                # Write
                f_out.write(restored)
    
    print(f"✅ Large file restored: {output_file}")
```

---

## Performance-Optimierung

### 1. Lazy-Loading von ML-Modellen

**Problem:** Startup-Zeit zu lang (alle 47 Modelle laden)

**Lösung:** Lazy-Loading (bereits implementiert)

```python
# Nur benötigte Modelle werden geladen
restorer = UnifiedRestorerV2()  # Schnell: 0.5s

# Erstes restore() lädt Modelle bei Bedarf
restored = restorer.restore(audio, sr)  # Langsam: 5-10s (einmalig)

# Weitere restore() sind schnell
restored2 = restorer.restore(audio2, sr)  # Schnell: 2-3s
```

---

### 2. GPU-Beschleunigung

**Problem:** CPU-Processing zu langsam (3-5x Echtzeit)

**Lösung:** CUDA-Unterstützung (wenn verfügbar)

```python
import torch

# Check GPU
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ CUDA not available (CPU-only)")

# Aurik nutzt GPU automatisch wenn verfügbar
restorer = UnifiedRestorerV2()
restored = restorer.restore(audio, sr)  # Automatisch GPU wenn CUDA verfügbar
```

**Performance-Vergleich:**
- **CPU (i7-10700K):** 3-5x Echtzeit (3min Audio → 10min Processing)
- **GPU (RTX 3090):** 0.5-1x Echtzeit (3min Audio → 2min Processing)

---

### 3. Multi-Threading für Batch-Processing

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import soundfile as sf

def restore_file(file_path, output_dir, restorer):
    """Restore single file (thread-safe)."""
    audio, sr = sf.read(file_path)
    restored = restorer.restore(audio, sr)
    
    output_path = output_dir / file_path.name
    sf.write(output_path, restored, 48000)
    return file_path.name

# Batch-Restore mit 4 Threads
restorer = UnifiedRestorerV2()
input_files = list(Path('input').glob('*.wav'))
output_dir = Path('output')

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(restore_file, f, output_dir, restorer) 
        for f in input_files
    ]
    
    for future in futures:
        filename = future.result()
        print(f"✅ {filename}")
```

**Speedup:** ~2-3x bei 4 Threads (I/O-bound Tasks profitieren)

---

## Code-Beispiele

### Beispiel 1: Vinyl-Transfer

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from core.processing_modes import ProcessingMode
import soundfile as sf

# Load 16-bit/44.1kHz vinyl rip
audio, sr = sf.read('vinyl_rip.wav')

# Restore with RESTORATION mode (preserve vinyl warmth)
restorer = UnifiedRestorerV2()
restored = restorer.restore(audio, sr, mode=ProcessingMode.RESTORATION)

# Save as 24-bit/48kHz WAV
sf.write('vinyl_restored.wav', restored, 48000, subtype='PCM_24')

print("✅ Vinyl transfer complete!")
print("   - Click/Crackle removed")
print("   - Rumble filtered")
print("   - Vinyl warmth preserved")
```

---

### Beispiel 2: Podcast-Enhancement

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from core.processing_modes import ProcessingConfig, ProcessingMode
import soundfile as sf

# Load podcast recording (noisy room, breath sounds)
audio, sr = sf.read('podcast_recording.wav')

# Custom config for podcasts
config = ProcessingConfig(
    mode=ProcessingMode.STUDIO_2026,
    denoise_strength=0.6,      # Strong noise reduction
    enable_vocal_enhancement=True,  # Voice clarity
    enable_phase_11_binaural=True,  # Immersive listening
)

restorer = UnifiedRestorerV2()
restored = restorer.restore(audio, sr, config=config)

# Save for distribution
sf.write('podcast_enhanced.wav', restored, 48000)

print("✅ Podcast enhanced for Spotify/Apple Podcasts!")
```

---

### Beispiel 3: Archival Preservation

```python
from core.unified_restorer_v2 import UnifiedRestorerV2
from core.processing_modes import ProcessingMode
import soundfile as sf

# Load fragile historical recording (78 RPM shellac)
audio, sr = sf.read('historical_1930s.wav')

# Conservative restoration (preserve authenticity)
restorer = UnifiedRestorerV2()
restored = restorer.restore(
    audio, sr,
    mode=ProcessingMode.ARCHIVAL,
    enable_logging=True  # Document processing for archives
)

# Save as lossless archive master
sf.write('archive_master.wav', restored, 48000, subtype='PCM_24')

# Save processing report
if hasattr(restorer, 'logger'):
    restorer.logger.save_trace('archive_processing_report.json')

print("✅ Archival master created!")
```

---

## Weitere Informationen

- **User Guide:** [docs/guides/USER_GUIDE.md](../guides/USER_GUIDE.md)
- **Architecture:** [docs/architecture/ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
- **Pipeline Flow:** [docs/architecture/PIPELINE_FLOW_ANALYSIS.md](../architecture/PIPELINE_FLOW_ANALYSIS.md)
- **Processing Logger:** [docs/guides/PROCESSING_LOGGER_USAGE.md](../guides/PROCESSING_LOGGER_USAGE.md)

---

**© 2026 Aurik Audio Restoration System**  
**Version:** 8.0.0 | **Status:** Production Ready
