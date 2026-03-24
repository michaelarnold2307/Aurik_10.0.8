# Aurik 9 — Spec 05: Material-System

> Definiert alle 15 Materialtypen (+ 2 Multichannel → Downmix), defektdichte-adaptive Verarbeitungsregeln,
> GP-Gedächtnis, Export, Sample-Rate-Strategie, Tonträgerketten-Erkennung.

---

## §6.1 Unterstützte Materialien (17 Typen)

```python
# Aurik 9: ausschließlich MONO und STEREO — kein Mehrkanalformat.
# > 2 Kanäle → PANNs-gewichteter Stereo-Downmix (automatisch).
SUPPORTED_MATERIALS = [
    "tape",          # Kassette: Dropout, Hiss, Wow/Flutter
    "reel_tape",     # Profi-Spulenband: Hiss, Print-Through, Dropout
    "vinyl",         # Schallplatte: Crackle, Warp, Rillenverzerrung
    "shellac",       # Schellack-78: Hochpegelrauschen, BW ≤ 8 kHz
    "wax_cylinder",  # Wachswalze (1890–1930): extrem hoher Rauschen, BW ≤ 5 kHz
    "wire_recording",# Drahtband (1940–1955): Jitter, Frequenz-Dropout
    "lacquer_disc",  # Acetat-Lackfolien (1930–1950): Riss-Klicken, Substrat-Rauschen
    "dat",           # Digital Audio Tape: Jitter, Dropout, ATRAC
    "cd_digital",    # CD/WAV: Clipping, Quantisierungsrauschen
    "mp3_low",       # MP3 < 128 kbps: starke Kompressionsartefakte
    "mp3_high",      # MP3 ≥ 128 kbps: moderate Artefakte
    "aac",           # AAC/M4A: moderne Kompression
    "minidisc",      # MiniDisc (ATRAC): 90er-Artefakte
    "streaming",     # Streaming-Kopie: variables Bitrate-Profil
    "unknown",       # Unbekannt: konservative Prior
]
# Hinweis: lacquer_disc, wax_cylinder, wire_recording → historische Materialien (v9.9.5)
```

---

## §6.2 Material-spezifische Verarbeitungsregeln

| Material | Hauptdefekte | Prioritäts-Phasen | PQS-Erwartung |
|---|---|---|---|
| `tape` | Dropout, Hiss, Wow/Flutter | phase_24, phase_29, phase_12 | MOS ≥ 4.2 |
| `reel_tape` | Print-Through, Hiss, Dropout | phase_29, phase_03, phase_24, phase_55 | MOS ≥ 4.3 |
| `vinyl` | Crackle, Warp, DC-Offset | phase_09, phase_12, phase_30 | MOS ≥ 4.0 |
| `shellac` | Breites Rauschen, Bandbegr. | phase_03, phase_06, phase_01 | MOS ≥ 3.8 |
| `dat` | Jitter, Dropout, ATRAC | phase_24, phase_02, phase_23 | MOS ≥ 4.4 |
| `cd_digital` | Clipping, Quantisierung | phase_23, phase_06, phase_40 | MOS ≥ 4.5 |
| `mp3_low` | Schwere Codec-Artefakte | phase_23, phase_03, phase_50 | MOS ≥ 3.9 |
| `mp3_high` | Moderate Codec-Artefakte | phase_23, phase_50 | MOS ≥ 4.2 |
| `aac` | Präsenz-Verlust, Artefakte | phase_23, phase_38, phase_06 | MOS ≥ 4.2 |
| `minidisc` | ATRAC, HF-Verlust | phase_23, phase_06, phase_07 | MOS ≥ 4.0 |
| `wax_cylinder` | Extremrauschen, BW ≤ 5 kHz | phase_03, phase_06, phase_01, phase_29 | MOS ≥ 3.5 |
| `wire_recording` | Jitter, Freq-Dropout | phase_12, phase_24, phase_03, phase_29 | MOS ≥ 3.6 |
| `lacquer_disc` | Riss-Klicken, Substrat-Rauschen | phase_01, phase_09, phase_03, phase_29 | MOS ≥ 3.7 |
| `streaming` | Dropouts, Codec-Artefakte, Bitrate-Varianz | phase_24, phase_23, phase_50 | MOS ≥ 4.1 |
| `unknown` | Alle aktiviert | Alle Tier-1 | MOS ≥ 3.8 |

---

## §6.2a [RELEASE_MUST] Pflicht-Phasen-Aktivierung pro Material (v9.10.73)

Die in §6.2 gelisteten **Prioritäts-Phasen** eines Materials MÜSSEN **unbedingt aktiviert** werden, wenn das Material erkannt wurde — **unabhängig vom DefectScanner-Severity-Score**.

**Begründung**: Der DefectScanner arbeitet mit statistischen Schwellwerten auf begrenztem Audio-Ausschnitt. Einzelne Defekte (z. B. ein kurzer Tape-Dropout im Intro) können unter der Schwelle liegen, obwohl sie für den Hörer klar wahrnehmbar sind. Die Prioritäts-Phasen enthalten eigene, hochauflösende Detektionslogik und entscheiden selbst, ob eine Reparatur notwendig ist.

**Invariante**:

```python
# In _select_phases(): Material-Prioritäts-Phasen immer aktivieren
for phase_id in MATERIAL_PRIORITY_PHASES[material]:
    if phase_id not in selected:
        selected.append(phase_id)
```

**Ausnahme**: Phasen, die explizit durch `GoalApplicabilityFilter` für das Material deaktiviert wurden (z. B. `phase_48_stereo_imaging` bei Mono-Material).

---

## §6.3 DefectType-Vollkatalog (28 Defekte)

```python
# core/defect_scanner.py — DefectType (Enum, 29 Werte)

# Analoge Kerndefekte:
CLICKS, CRACKLE, HUM, LOW_FREQ_RUMBLE, DROPOUTS
WOW          # Pitch-Instabilität < 0.5 Hz (Motorgeschwindigkeit / Capstan) — IEC 60386
FLUTTER      # Pitch-Instabilität 0.5–200 Hz (Antriebsriemen / Bandführung) — IEC 60386
             # Erkennung: WOW = pYIN-Varianz über 500 ms-Fenster; FLUTTER = über 50 ms-Fenster
             # WOW → phase_12 (langsame Pitch-Korrektur); FLUTTER → phase_12 + phase_31

# Klipping, Sättigung & Gleichspannung:
CLIPPING         # Harte Amplitudenbegrenzung → REPARIEREN
SOFT_SATURATION  # Tube-/Tape-Sättigung (gerade Obertöne) → BEWAHREN!
DC_OFFSET

# Spektral:
BANDWIDTH_LOSS, HIGH_FREQ_NOISE

# Kanal/Stereo:
STEREO_IMBALANCE, PHASE_ISSUES

# Pitch:
PITCH_DRIFT

# Groove / Transienten:
TRANSIENT_SMEARING  # Ansatz-Verschmierung durch Kompression → GrooveMetric-relevant

# Hall & Magnetband:
REVERB_EXCESS, PRINT_THROUGH

# Digital/Codec:
DIGITAL_ARTIFACTS, COMPRESSION_ARTIFACTS
PRE_ECHO         # MP3/AAC Temporal-Masking-Artefakt vor Transienten
QUANTIZATION_NOISE, JITTER_ARTIFACTS, DYNAMIC_COMPRESSION_EXCESS

# Kopf-/Azimuth-Fehler:
HEAD_WEAR        # Komplette Frequenzband-Ausblöschung → phase_56
AZIMUTH_ERROR    # Kammfilterung L/R durch Kopf-Fehlausrichtung → phase_14 + phase_25
                 # Signatur: frequenzabhängige L/R-Phasendifferenz, Kreuzkorrelation-Peak ≠ 0 lag
                 # Detektion: PHD(freq) = angle(STFT_L / STFT_R) → monotone HF-Drift > 20°/kHz

# Entzerrungs- & Digitalisierungsfehler (neu v9.10.46):
RIAA_CURVE_ERROR  # Falsche oder historische Disc-Entzerrungskurve → phase_04 + phase_06
                  # Kurvenvarianten (pre-RIAA 1954): NAB, Columbia, AES, Capitol, London, CCIR
                  # Erkennung: Referenzvergleich Spektral-Slope 250–8000 Hz vs. RIAA-Ideal
                  #   Abweichung > ±3 dB → RIAA_CURVE_ERROR mit erkannter Kurve als Subtyp
                  # Klassifikator liefert: curve_type ∈ {"riaa", "nab", "columbia", "aes",
                  #   "capitol", "london", "ccir", "unknown_prestandard"}
                  # phase_04 wendet Inverse-Kurve der erkannten Variante an
ALIASING          # Spiegelfrequenzen durch AA-Filter-Fehler → phase_03 + phase_23
BIAS_ERROR        # Falscher Vormagnetisierungsstrom → phase_04 + phase_03 + phase_29
# --- Spec §6.3 v9.10.57: Sibilanten-Überbetonung (ergibt 28 DefectTypes) ---
SIBILANCE         # Zischlautüberbetonung (> 6 kHz) — De-Esser-Trigger (phase_19 + phase_43)
```

**CLIPPING vs. SOFT_SATURATION — kritische Unterscheidung:**

```python
def classify_clipping(audio: np.ndarray, sr: int) -> ClippingType:
    """Diskriminiert CLIPPING von SOFT_SATURATION.

    CLIPPING:        flat_tops > 0.1 % UND THD_odd > THD_even × 1.5
    SOFT_SATURATION: flat_tops < 0.1 % ODER THD_even > THD_odd
    SOFT_SATURATION → Pipeline überspringt Clipping-Reparatur (BEWAHREN!)
    """
```

---

## §6.4a [RELEASE_MUST] Material-adaptive Erkennungsschwellen im DefectScanner (v9.10.73)

DefectScanner-Erkennungsschwellen MÜSSEN **material-adaptiv** sein. Analoge Medien erfordern empfindlichere Schwellwerte als digitale Quellen.

| Defekttyp | Analog-Medien | Digital-Medien | Begründung |
|---|---|---|---|
| DROPOUTS | 20 % median-RMS | 10 % median-RMS | Tape-Dropouts: graduelle Pegelfades statt hartem Null |
| CLICKS | material-skaliert | Standard | Vinyl-Rillengeräusche vs. digitale Störimpulse |

**Analog-Medien** (empfindlichere Schwellen): `tape`, `reel_tape`, `vinyl`, `shellac`, `wax_cylinder`, `wire_recording`, `lacquer_disc`, `dat`.

**Invariante**: `_detect_dropouts()` greift auf `self.material_type` zu — dieses MUSS vor dem Aufruf aus dem resolved material_type der `scan()`-Methode gesetzt sein.

---

## §6.4 GP-Gedächtnis pro Material & Genre

```
~/.aurik/gp_memory/
    tape.json         vinyl.json      shellac.json
    digital.json      unknown.json
    schlager.json     # Genre-spezifisch (angelegt beim ersten Schlager-Job)
    jazz.json         orchestral.json
    opera.json        rock.json
```

Format:

```json
{
  "observations": [
    {"params": {"noise_reduction_strength": 0.7}, "score": 4.23, "ts": "..."}
  ],
  "version": 1
}
```

**GP-Memory-Recovery:** Korrupte Datei → `.corrupted.json` umbenennen, leer starten. Max. 500 Beobachtungen (LRU-Trim). Atomic-Write via Temp-Datei + `os.replace()`.

---

## §6.5 Export-Formate & Regeln

| Format | Qualität | Anwendungsfall |
|---|---|---|
| FLAC (24-bit) | Archivqualität | Standard-Export |
| WAV (24-bit, 48 kHz) | Produktionsqualität | DAW-Weiterverarbeitung |
| WAV (16-bit, 44.1 kHz) | CD-Qualität | CD-Mastering |
| MP3 CBR / VBR | 128–320 kbps / V0–V5 | Streaming, Kompatibilität |
| OGG Vorbis (q9) | Open Streaming | Plattform-unabhängig |
| AIFF (24-bit, 48 kHz) | Apple-Ökosystem | Logic Pro / Pro Tools |

**Pflicht-Regeln:**

- Bit-Tiefe: 24-bit → 24-bit (kein forced Downgrade ohne Nutzer-Wahl)
- Dithering 24→16 bit: **POW-r Typ 3** (Wannamaker 1992); Fallback: TPDF
- VERBOTEN: Truncation ohne Dithering
- Lautheit: **−14 LUFS** (EBU R128 Streaming) / **−18 LUFS** (Archiv)
- True-Peak: **−1.0 dBTP** (ITU-R BS.1770-5) — immer vor Export
- Metadaten (ID3, Vorbis Comments, BWF): vollständig übertragen + Restaurierungs-Metadaten

**MP3-Export:** LAME über pydub/subprocess. Mono-Quellen bleiben Mono.

---

## §6.6 Sample-Rate- & Bit-Tiefe-Strategie

**Interne SR: immer 48 000 Hz** (vor und nach jedem DSP-Schritt).

```python
# Pflicht-Eingangsprüfung in jeder Phase und jedem Plugin:
assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"

# Resampling: Lanczos-4 (scipy.signal.resample_poly, Kaiser-Filter β=14)
# Bit-Tiefe intern: float32 in [-1, 1]
# Nach Resampling: NaN/Inf-Check Pflicht
```

---

## §6.6 Tonträgerketten-Erkennung (bindend ab v9.10.45)

**Pflicht-Spektralfingerabdruck bei jedem Import (vor allen Klassifikatoren):**

| Merkmal | Messmethode | Schwellwerte |
|---|---|---|
| Rolloff 95 % | `librosa.feature.spectral_rolloff(roll_percent=0.95)` Median | < 4 kHz → Shellac/Wachswalze; < 8 kHz → Kassette |
| WOW-Index | pYIN-Pitch-Varianz über 500 ms-Fenster (IEC 60386) | > 1.0 Hz → Kassette WOW; ≤ 0.1 Hz → digital |
| FLUTTER-Index | pYIN-Pitch-Varianz über 50 ms-Fenster (IEC 60386) | > 0.5 Hz → Bandantrieb-FLUTTER; Drahtband: stochastischer Verlauf (σ > 1.2 Hz) |
| Azimuth-Drift | L/R-Phasendifferenz-Slope STFT (20°/kHz) | > 20°/kHz → AZIMUTH_ERROR |
| RIAA-Slope-Abw. | Spektral-Slope 250–8000 Hz vs. RIAA-Ideal | > ±3 dB → RIAA_CURVE_ERROR + curve_type |
| HF-Energie > 16 kHz | Spektralsumme (STFT), Anteil Gesamtenergie | 0 % → MP3-Kette oder Kassette |
| Rauschpegel (P5 PSD) | 10. Perzentil mittlere PSD | > −30 dBFS² → schweres Bandrauschen |
| Effektive Bandbreite | HF-Rolloff −60 dBFS | < 8 kHz → Material-BW-Limit |

```python
# Kettenerkennung → MaterialType-Ableitung:
if result.is_multi_generation:
    primary_material = result.transfer_chain[0]    # z. B. MaterialType.TAPE
    secondary_chain  = result.transfer_chain[1:]   # z. B. [MediaType.MP3_LOW]
    # → aktiviert kombinierte Phasen beider Materialien

# Kettenergebnis in RestorationResult.genealogy als:
# SampleOperation(operation_type="chain_detection")
```

**Referenz-Fingerabdruck (Elke, Feb 2026):**

| Merkmal | Messwert | Diagnose |
|---|---|---|
| Rolloff 95 % | 1 486 Hz | Kassettenköpfe verschlissen |
| Wow/Flutter | 2,38 Hz | Schwere Pitch-Instabilität → Kassette |
| HF > 16 kHz | 0 % | Kassette + MP3-Kette |
| Rauschen P5 | −31 dBFS² | Schweres Bandrauschen |
| Tonträgerkette | `cassette_tape → mp3_low` | Zwei-stufige Degradation |

---

## §6.7 Importformate (Eingang)

| Format | Erweiterungen |
|---|---|
| WAV / AIFF | `.wav`, `.aiff`, `.aif` |
| FLAC | `.flac` |
| MP3 | `.mp3` |
| AAC / M4A | `.aac`, `.m4a`, `.mp4` |
| OGG Vorbis | `.ogg` |
| WMA | `.wma` |
| Opus | `.opus` |
| CAF | `.caf` |

**Invarianten:**

- Alle Formate → intern float32, 48 000 Hz, Stereo oder Mono
- Maximale Dateigröße: 10 GB (darüber Chunk-Modus)
- > 2 Kanäle → PANNs-gewichteter Stereo-Downmix
- Ungültige Dateien: `AudioLoadError` + Deutsch-Meldung, kein Absturz
