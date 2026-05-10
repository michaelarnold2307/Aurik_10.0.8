# Aurik 9.x.x — Benutzerhandbuch

**Intelligentes Musik-Restaurierungs-, Reparatur- und Rekonstruktionssystem**

---

## 📋 Inhaltsverzeichnis

1. [Einführung](#einführung)
2. [Die zwei Modi](#die-zwei-modi)
3. [Workflow](#workflow)
4. [Störfaktoren & Defekte](#störfaktoren--defekte)
5. [Natürlicher Klang](#natürlicher-klang)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Einführung

AURIK ist ein **vollautomatisches System** zur Restauration, Reparatur und Rekonstruktion von Audio-Material. Das Besondere:

✅ **Keine Nutzereingriffe erforderlich** — Magic Button  
✅ **14 Musikalische Qualitätsziele** — auf Spitzenqualität ausgerichtete Restaurierung  
✅ **100% lokal** — Keine Cloud, volle Datenkontrolle  
✅ **100% offline** — Alle ML-Modelle lokal gebundelt  
✅ **CPU-only** — Kein GPU erforderlich

### Philosophie

**"Primum non nocere"** - Zuerst einmal nicht schaden.

AURIK folgt einem konservativen Ansatz:

- **Bewahrung vor Verbesserung** - Original wird respektiert
- **Stabilität vor Brillanz** - Sicher statt riskant
- **Natürlichkeit vor Perfektion** - Organisch statt künstlich

---

## Die zwei Modi

AURIK bietet zwei grundlegend verschiedene Verarbeitungsmodi:

### **Modus 1: Restoration (Bewahrung)**

**Ziel:** Historische Bewahrung mit maximaler Originalintegrität

**Wann verwenden:**

- Historische Aufnahmen (Vinyl, Shellac, Tape)
- Kulturell bedeutsame Werke
- Archiv-Material
- Wenn Authentizität wichtiger ist als moderne Klangästhetik

**Was passiert:**

- Konservative Störfaktor-Entfernung
- Minimale Klangveränderung
- Erhaltung von zeitlichen Charakteristiken
- Bewahrung von "Patina" (außer bei echten Defekten)

**Restaurations-Aggressivität:** 0.3 (conservative)

**Beispiel-Anwendungen:**

- 1950er Jazz-Vinyl → Sauberes Vinyl (ohne Knistern)
- 1930er Shellac → Hörbares Archiv (ohne Klicks)
- 1970er Tape → Digitales Master (ohne Dropout)

### **Modus 2: Modern Reproduction (Auffrischung)**

**Ziel:** Klingt wie 2026 im Highend-Profi-Studio neu aufgenommen

**Wann verwenden:**

- Moderne Releases/Re-Releases
- Kommerzielle Nutzung
- Streaming-Publikation
- Wenn moderne Klangästhetik gewünscht

**Was passiert:**

- Aggressive Störfaktor-Beseitigung
- Moderne Klangästhetik (2026 Studio-Standard)
- Frequency Extension (wenn möglich)
- Stereo Enhancement (wenn sinnvoll)
- Psychoakustische Optimierung

**Restaurations-Aggressivität:** 0.8 (aggressive, but HIPS-safe)

**Beispiel-Anwendungen:**

- 1960er Beatles-Remaster → 2026 Streaming-Qualität
- 1980er Pop → Moderne Höhen/Dynamik
- Spoken Word → Studio-Mikrofonqualität

### Modus-Vergleich

| Aspekt | Restoration | Modern Reproduction |
| --- | --- | --- |
| **Philosophie** | "Wie Original gemeint" | "Wie heute aufgenommen" |
| **Frequenzgang** | Original-Charakteristik | Moderner Full-Range |
| **Dynamik** | Original-Dynamik | Optimierte Dynamik |
| **Raumklang** | Original-Akustik | Optimierter Raum |
| **Sibilanten** | Konservativ (nur extreme) | Aggressive Behandlung |
| **Hintergrund** | Leise Geräusche OK | Studio-Stille |
| **Transients** | Original-Attack | Optimierte Klarheit |
| **Verzerrung** | Nur echte Defekte | Alle Verzerrungen |

---

## Workflow

### Schritt 1: Audio laden

```bash
# CLI
aurik process input.wav --mode restoration

# API
curl -X POST http://localhost:8000/process \
  -F "audio=@input.wav" \
  -F "mode=modern_reproduction"
```

### Schritt 2: Automatische Analyse

AURIK analysiert automatisch:

- **Medium-Erkennung** - Vinyl, Shellac, Tape, Digital, Cassette
- **Defekt-Erkennung** - Clicks, Crackle, Hiss, Hum, Distortion
- **Content-Analyse** - Music/Speech, Vocal/Instrumental
- **Quality-Assessment** - SNR, THD, Clipping, Bandwidth

### Schritt 3: Adaptive Processing

**Automatische Verarbeitungskette:**

1. **Pre-Processing**
   - DC-Offset Removal
   - Clipping Detection
   - Phase Coherence Check

2. **Defect Removal** (je nach Modus)
   - Click/Pop Removal
   - Crackle Removal  
   - Hum Removal
   - Hiss Reduction

3. **Enhancement** (nur Modus 2)
   - Frequency Extension
   - Stereo Enhancement
   - Psychoacoustic Optimization

4. **Vocal Processing** (falls Vocals erkannt)
   - Sibilant Treatment (De-Esser)
   - Pitch Correction (nur offensichtliche Fehler)
   - Formant Preservation

5. **Quality Control**
   - HIPS Compliance Check
   - Nebenwirkungen Assessment
   - Reversibility Validation

### Schritt 4: Output

**Ausgabe-Formate:**

- WAV (24-bit/96kHz) - Master-Qualität
- FLAC - Verlustfreie Kompression
- MP3/AAC - Streaming-tauglich

**Zusätzliche Outputs:**

- `*_processed.wav` - Verarbeitetes Audio
- `*_audit_report.json` - Vollständiger Audit-Trail
- `*_comparison.html` - Vorher/Nachher-Vergleich
- `*_stems.zip` - Stems (falls Separation verwendet)

---

## Störfaktoren & Defekte

### Vinyl-spezifisch

#### **1. Clicks (Klicks)**

- **Ursache:** Kratzer, Staub auf Oberfläche
- **Erkennung:** Transiente Impulse (< 5ms)
- **Behandlung:**
  - Restoration: Nur extreme Klicks (> -20 dB)
  - Modern: Alle Klicks (> -40 dB)
- **Algorithmus:** Spectral interpolation + temporal smoothing

#### **2. Crackle (Knistern)**

- **Ursache:** Oberflächenschäden, Verschleiß
- **Erkennung:** Hochfrequente Impulsdichte
- **Behandlung:**
  - Restoration: Moderate Reduktion (60%)
  - Modern: Aggressive Reduktion (95%)
- **Algorithmus:** Statistical noise gate + adaptive filtering

#### **3. Hiss (Rauschen)**

- **Ursache:** Tonband-Rauschen, Vinyl-Eigenrauschen
- **Erkennung:** Spektrales Noise Floor
- **Behandlung:**
  - Restoration: Minimal (10-20 dB Reduktion)
  - Modern: Maximal (30+ dB Reduktion)
- **Algorithmus:** Spectral subtraction + psychoacoustic masking

#### **4. Hum (Brummen)**

- **Ursache:** 50/60 Hz Netzbrummen
- **Erkennung:** Tonale Komponenten bei 50/60/100/120 Hz
- **Behandlung:** Beide Modi: Aggressive Entfernung
- **Algorithmus:** Notch filtering + harmonic tracking

### Tape-spezifisch

#### **5. Dropout (Aussetzer)**

- **Ursache:** Bandschäden, Kopfverschmutzung
- **Erkennung:** Plötzliche Pegelverluste
- **Behandlung:** Spectral reconstruction
- **Algorithmus:** LSTM-based inpainting

#### **6. Wow & Flutter**

- **Ursache:** Mechanische Instabilität
- **Erkennung:** Pitch modulation analysis
- **Behandlung:**
  - Restoration: Moderate Korrektur
  - Modern: Aggressive Korrektur
- **Algorithmus:** Phase vocoder + time-stretching

#### **7. Print-Through**

- **Ursache:** Magnetische Überlagerung benachbarter Windungen
- **Erkennung:** Pre-/Post-Echos
- **Behandlung:** Temporal de-reverberation

### Digital-spezifisch

#### **8. Digital Clipping**

- **Ursache:** Übersteuerung (> 0 dBFS)
- **Erkennung:** Sample-Werte at ±1.0
- **Behandlung:**
  - Restoration: Conservative reconstruction
  - Modern: Aggressive reconstruction + dynamics restoration
- **Algorithmus:** Machine learning declipping (Signal-to-Distortion > 20 dB)

#### **9. Aliasing**

- **Ursache:** Zu niedrige Sample-Rate
- **Erkennung:** Aliasing artifacts > Nyquist
- **Behandlung:** Anti-aliasing filtering

#### **10. Quantization Noise**

- **Ursache:** Niedrige Bit-Depth (< 16-bit)
- **Erkennung:** Dithering noise pattern
- **Behandlung:** Noise shaping + dithering

### Vocal-spezifisch

#### **11. Sibilanten (S-Laute)**

- **Ursache:** Hochfrequenz-Energie bei 4-12 kHz
- **Erkennung:** Multi-band energy thresholding
- **Behandlung:**
  - Restoration: Nur extreme Sibilanten (> 12 dB over average)
  - Modern: Alle problematischen Sibilanten (> 6 dB over average)
- **Algorithmus:**

  ```python
  - Sibilant detection (4-12 kHz energy)
  - Formant-aware compression
  - Spectral shaping (preserve voice character)
  - Transient preservation
  ```

- **Best Practice:**
  - Reduction: 3-6 dB (nie mehr!)
  - Frequency: 6-9 kHz (sweet spot)
  - Attack/Release: Ultra-fast (< 1ms)

#### **12. Plosives (P-/B-Laute)**

- **Ursache:** Luftstöße ins Mikrofon
- **Erkennung:** Low-frequency transients
- **Behandlung:** High-pass filtering + transient shaping

#### **13. Harshness (Härte)**

- **Ursache:** Überhöhte Mittenfrequenzen
- **Erkennung:** Spectral tilt analysis
- **Behandlung:** Multi-band compression + EQ

---

## Natürlicher Klang

### Wie AURIK Natürlichkeit bewahrt

#### 1. **Formant Preservation**

Formanten sind die charakteristischen Resonanzen der Stimme/Instrumente.

**Problem:** Viele Algorithmen verschieben Formanten → unnatürlich

**AURIK-Lösung:**

```python
- Formant detection (cepstral analysis)
- Formant-aware processing (separate envelope vs. fine structure)
- Formant restoration (if damaged by processing)
```

#### 2. **Transient Preservation**

Transienten definieren den "Attack" (z.B. Trommel-Schlag, Gitarren-Anschlag).

**Problem:** Smoothing-Algorithmen zerstören Transienten → matschig

**AURIK-Lösung:**

```python
- Transient detection (onset detection)
- Selective processing (smooth only sustain, not attack)
- Transient restoration (if lost)
```

#### 3. **Phase Coherence**

Phase-Beziehungen zwischen Frequenzen sind kritisch für Räumlichkeit.

**Problem:** Spektrale Verarbeitung zerstört Phase → flach

**AURIK-Lösung:**

```python
- Phase-aware STFT (linear phase where possible)
- Phase reconstruction (after magnitude processing)
- Stereo phase validation
```

#### 4. **Harmonic Structure**

Natürliche Klänge haben harmonische Obertonreihen.

**Problem:** Noise Reduction entfernt Harmonics → dünn

**AURIK-Lösung:**

```python
- Harmonic tracking (F0 + overtones)
- Selective filtering (noise vs. harmonics)
- Harmonic restoration (if lost)
```

#### 5. **Micro-Dynamics**

Kleinste Lautstärke-Schwankungen = Lebendigkeit.

**Problem:** Compression zerstört Micro-Dynamics → leblos

**AURIK-Lösung:**

```python
- Adaptive compression (preserve micro-dynamics)
- Transient enhancement (restore lost dynamics)
- Parallel processing (dry/wet blend)
```

### Qualitätsindikatoren für Natürlichkeit

| Metric | Threshold | Bedeutung |
| --- | --- | --- |
| **THD (Total Harmonic Distortion)** | < 0.5% | Keine künstliche Verzerrung |
| **IMD (Intermodulation Distortion)** | < 0.3% | Natürliche Frequenz-Interaktion |
| **Phase Coherence** | > 0.95 | Räumlichkeit erhalten |
| **Transient Preservation** | > 90% | Attack-Charakter erhalten |
| **Formant Shift** | < 5% | Stimm-/Instrumenten-Charakter |
| **Micro-Dynamic Range** | > 20 dB | Lebendigkeit erhalten |

---

## Troubleshooting

### Problem: "Stimme klingt unnatürlich/robotisch"

**Mögliche Ursachen:**

1. Zu aggressive Sibilant-Behandlung
2. Formant-Verschiebung durch Pitch Correction
3. Zu viel Noise Reduction

**Lösung:**

```bash
# Reduziere Sibilant-Aggressivität
aurik process input.wav --sibilant-threshold 9  # statt 6 dB

# Deaktiviere Pitch Correction
aurik process input.wav --no-pitch-correction

# Moderate Noise Reduction
aurik process input.wav --noise-reduction 0.3  # statt 0.8
```

### Problem: "Clicks noch hörbar"

**Mögliche Ursachen:**

1. Modus "restoration" ist zu konservativ
2. Clicks sind sehr laut/frequent
3. Audio ist stark beschädigt

**Lösung:**

```bash
# Verwende "modern_reproduction" Modus
aurik process input.wav --mode modern_reproduction

# Erhöhe Click-Sensitivität
aurik process input.wav --click-threshold -40  # statt -20 dB

# Multi-Pass Processing
aurik process input.wav --click-passes 3
```

### Problem: "Klang zu dünn/flach"

**Mögliche Ursachen:**

1. Zu viel Noise Reduction
2. Zu viel Hum Removal (entfernt low frequencies)
3. Phase-Probleme

**Lösung:**

```bash
# Moderate Processing
aurik process input.wav --conservative

# Prüfe Phase
aurik analyze input.wav --check-phase

# Deaktiviere Hum Removal (falls nicht nötig)
aurik process input.wav --no-hum-removal
```

### Problem: "HIPS Violation"

**Mögliche Ursachen:**

1. Audio ist stark beschädigt (Clipping, extreme Noise)
2. Processing würde zu viel Schaden anrichten
3. Unsicherheit zu hoch

**Lösung:**

```bash
# Prüfe Audit Report
cat output_audit_report.json

# Verwende non-strict Mode (Warnings statt Errors)
aurik process input.wav --hips-mode warning

# Pre-Processing
aurik preprocess input.wav  # Normalisierung, DC-Offset
aurik process input_preprocessed.wav
```

---

## Best Practices

### 1. **Wähle den richtigen Modus**

```
Historisch + Archiv → Restoration
Modern + Kommerziell → Modern Reproduction
```

### 2. **Prüfe Input-Qualität**

```bash
aurik analyze input.wav
```

**Achte auf:**

- Clipping (sollte < 0.1% sein)
- SNR (> 30 dB für gute Ergebnisse)
- Dynamic Range (> 10 dB)

### 3. **Nutze A/B-Vergleich**

```bash
aurik process input.wav --generate-comparison
open output_comparison.html
```

**Bewerte:**

- Natürlichkeit (klingt es echt?)
- Artefakte (neue Störungen?)
- Verbesserung (besser als vorher?)

### 4. **Iteriere bei Bedarf**

```bash
# Pass 1: Conservative
aurik process input.wav --mode restoration

# Bewertung: Noch zu viele Clicks?
# Pass 2: Aggressive Click Removal
aurik process input.wav --click-threshold -40

# Bewertung: Nun zu dünn?
# Pass 3: Formant Restoration
aurik process input_pass2.wav --restore-formants
```

### 5. **Dokumentiere Settings**

```bash
# Speichere verwendete Einstellungen
aurik process input.wav --save-config settings.json

# Reproduziere mit gleichen Settings
aurik process input2.wav --config settings.json
```

### 6. **Nutze Stems bei Vocals**

```bash
# Separiere Vocals für bessere Behandlung
aurik process input.wav --separate-vocals

# Gibt:
# - vocals_processed.wav
# - instrumental_processed.wav
# - mixed_processed.wav
```

### 7. **Prüfe Compliance**

```bash
# Audit Report analysieren
cat output_audit_report.json | jq '.hips_compliance'

# Sollte zeigen:
{
  "status": "pass",
  "violations": 0,
  "compliance_rate": 1.0
}
```

---

## Appendix: Technische Details

### Verfügbare DSP-Module

- **Click Removal:** `automatic_declicker`, `shellac_declicker`
- **Crackle Removal:** `automatic_decrackler`
- **Hum Removal:** `hum_remover`, `harmonic_hum_remover`
- **Hiss Reduction:** `spectral_noise_gate`, `adaptive_noise_reduction`
- **De-Esser:** `aurik_deesser_pro` (formant-aware)
- **Declipping:** `ml_declipping` (LSTM-based)
- **Dynamics:** `intelligent_compressor`, `adaptive_limiter`

### Verfügbare ML-Modelle

- **Vocal Separation:** `HybridVocalSeparator` (MDX-Net + Demucs v5)
- **Noise Reduction:** `DeepFilterNet v3`
- **Enhancement:** `ResembleEnhance`
- **Super-Resolution:** `AudioSR`

### CLI-Referenz

```bash
# Einfachste Verwendung
aurik process input.wav

# Mit Modus
aurik process input.wav --mode restoration
aurik process input.wav --mode modern_reproduction

# Mit spezifischen Optionen
aurik process input.wav \
  --click-threshold -30 \
  --noise-reduction 0.5 \
  --sibilant-threshold 8 \
  --no-pitch-correction

# Batch Processing
aurik batch *.wav --mode restoration --output-dir processed/

# Analyse
aurik analyze input.wav --detailed

# Vergleich
aurik compare original.wav processed.wav
```

---

**Version:** 8.1  
**Datum:** 7. Februar 2026  
**Status:** Produktionsreif  
**Support:** siehe docs/ für weitere Dokumentation
