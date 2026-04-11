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
# [RELEASE_MUST] Schlüssel-Mapping MediumDetector → SUPPORTED_MATERIALS (kanonisch):
# MediumDetector (forensics/medium_detector.py) verwendet intern andere Schlüsselnamen.
# MediumDetector.detect() MUSS diese Keys vor Rückgabe auf SUPPORTED_MATERIALS normieren:
#   "cassette"         → "tape"           (Compact Cassette; Typ I/II/IV).
#   "reel_wire"        → "wire_recording" (Drahtbandgerät 1940–1955)
#   "cassette_digital" → "dat"            (Digitalkassette; DAT-Verarbeitungspfad greift)
#   "vhs_audio"        → "tape"           (VHS-Tonspur; Kassetten-Restaurierungspfad)
#   "composite"        → transfer_chain[0] normiert (erster Träger bestimmt primären Pfad)
# Alle anderen MediumDetector-Schlüssel (vinyl, shellac, reel_tape, cd_digital, …)
# entsprechen direkt den SUPPORTED_MATERIALS-Werten — kein Mapping nötig.
# INVARIANTE: material_type in RestorationResult MUSS immer ein SUPPORTED_MATERIALS-Schlüssel sein.
```

---

## §6.2 Material-spezifische Verarbeitungsregeln

| Material | Hauptdefekte | Prioritäts-Phasen | PQS-Erwartung |
| --- | --- | --- | --- |
| `tape` | Dropout, Hiss, Wow/Flutter | phase_24, phase_29, phase_12 | MOS ≥ 4.2 |
| `reel_tape` | Print-Through, Hiss, Dropout | phase_29, phase_03, phase_24, phase_55 | MOS ≥ 4.3 |
| `vinyl` | Crackle, Warp, Low-Freq-Rumble | phase_09, phase_12, phase_05 | MOS ≥ 4.0 |
| | | **Begründung**: Vinyl erzeugt kein DC-Offset (Tape-/ADC-Defekt); typischer LF-Defekt ist Plattenteller-Lagergeräusch (LOW_FREQ_RUMBLE, 5–20 Hz) → phase_05 (Rumble-Filter, Hochpass < 20 Hz). VERBOTEN: phase_30 als Vinyl-Prioritätsphase. | |
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

## §6.3 DefectType-Vollkatalog (46 Defekte)

```python
# core/defect_scanner.py — DefectType (Enum, 46 Werte)

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
                  #
                  # §6.3a PRE-RIAA KURVENPARAMETER (kanonische Zeitkonstanten, bindend):
                  # Alle Werte: (τ_bass_µs, τ_mid_µs, τ_treble_µs) → Pol/Nullstellen-Tripel.
                  # Inverse Korrektur: Shelving-EQ mit diesen Zeitkonstanten gespiegelt.
                  #
                  # PRE_RIAA_EQ_CURVES = {
                  #   # RIAA 1954 (Referenz — Standard ab 1954):
                  #   "riaa":           (3180, 318, 75),     # IEC 60268-4
                  #
                  #   # NAB (National Association of Broadcasters, bis 1953):
                  #   "nab":            (3180, 318, 50),     # Basswendepunkt 500 Hz, HF-Shelf 3180 µs
                  #
                  #   # Columbia 78 rpm (bis 1948):
                  #   "columbia":       (1590, 318, 0),      # Bass turnover 100 Hz, kein HF-Shelf
                  #                                          # → +6 dB Bass vs. RIAA bei 50 Hz
                  #
                  #   # AES (Audio Engineering Society, 1951–1954):
                  #   "aes":            (3180, 500, 0),      # Mittenbetonte Entzerrung
                  #
                  #   # Capitol (US, bis 1953):
                  #   "capitol":        (1590, 400, 0),      # ähnlich Columbia, flacherer HF-Abfall
                  #
                  #   # London / Decca (UK, bis 1954):
                  #   "london":         (3180, 318, 100),    # HF-Boost stärker als RIAA
                  #
                  #   # CCIR (europäischer Rundfunkstandard für Tape, sekundär für lacquers):
                  #   "ccir":           (3180, 318, 120),    # Tape-Entzerrung, 50 µs kurzfristig
                  #
                  #   # Unbekannte Vorstandardkurve — konservative Näherung Columbia:
                  #   "unknown_prestandard": (1590, 318, 0),
                  # }
                  #
                  # Erkennung Algorithmus (MediumClassifier._detect_riaa_curve_error):
                  #   1. Spectral-Slope 250–8000 Hz vs. RIAA-Ideal-LUT (±3 dB Toleranz pro Oktave)
                  #   2. Vergleich Basswendepunkt: Short-time LUFS 50–200 Hz / 200–800 Hz Ratio
                  #      → Ratio > +4 dB → columbia/nab verdächtig
                  #   3. Log-Likelihood über alle Kurven → argmax = curve_type
                  #   4. Konfidenz-Grenzwert ≥ 0.70 → RIAA_CURVE_ERROR setzen, sonst skip
                  #
                  # INVARIANTE: phase_04 MUSS bei curve_type ≠ "riaa" die exakten
                  # Zeitkonstanten aus PRE_RIAA_EQ_CURVES laden und die inverse
                  # Shelving-Kette anwenden. VERBOTEN: generische EQ-Schätzung ohne LUT.
ALIASING          # Spiegelfrequenzen durch AA-Filter-Fehler → phase_03 + phase_23
BIAS_ERROR        # Falscher Vormagnetisierungsstrom → phase_04 + phase_03 + phase_29
# --- Spec §6.3 v9.10.57: Sibilanten-Überbetonung ---
SIBILANCE         # Zischlautüberbetonung (> 6 kHz) — De-Esser-Trigger (phase_19 + phase_43)
# --- v9.10.57b: Transport-Bump ---
TRANSPORT_BUMP    # Impulsartige Mikro-Geschwindigkeitssprünge 50–300 ms (Kassette/Tape-Holpern) → phase_12
# --- v9.10.77: Vocal-Harshness ---
VOCAL_HARSHNESS   # Vokale Härte/Übersteuerung/Kratzigkeit im 2–6 kHz Band → phase_42 + phase_19
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
| --- | --- | --- | --- |
| DROPOUTS | 20 % median-RMS | 10 % median-RMS | Tape-Dropouts: graduelle Pegelfades statt hartem Null |
| CLICKS | material-skaliert | Standard | Vinyl-Rillengeräusche vs. digitale Störimpulse |

**Analog-Medien** (empfindlichere Schwellen): `tape`, `reel_tape`, `vinyl`, `shellac`, `wax_cylinder`, `wire_recording`, `lacquer_disc`, `dat`.

**Invariante**: `_detect_dropouts()` greift auf `self.material_type` zu — dieses MUSS vor dem Aufruf aus dem resolved material_type der `scan()`-Methode gesetzt sein.

---

## §6.4b [RELEASE_MUST] `TAPE_HEAD_LEVEL_DIP` Material-Gate mit Cross-Material-Fallback (v9.11.2)

`TAPE_HEAD_LEVEL_DIP` bleibt primär material-gebunden (`tape`, `reel_tape`, `wire_recording`),
MUSS aber bei starker Morphologie auch bei Fehlklassifikation (z. B. Tape-Transfer als Vinyl markiert)
erkennbar bleiben.

### Primärregel

- Für Tape-Materialien: voller Score aus `_detect_tape_head_level_dips()`.

### Cross-Material-Fallback (nur bei starker Evidenz)

Fallback darf nur aktivieren, wenn **alle** Kriterien erfüllt sind:

- `severity >= 0.12`
- `dip_count >= 2`
- `mean_depth_db >= 6.0`
- `event_rate_per_s >= 0.15`

Bei aktivem Fallback:

- `severity := severity * 0.75`
- `confidence := min(confidence, 0.72)`
- `metadata.cross_material_fallback = True`
- `metadata.fallback_material_gate_bypassed = <material>`

### Periodizitäts-Marker (Capstan-Signatur)

Wenn mindestens 3 Events vorliegen, ist ein Confidence-Bonus zulässig, falls

- Intervall-`cv < 0.35`
- `median_interval_s` in `[0.5, 3.5]`

Dann: `confidence += 0.08` (geclippt), plus

- `metadata.is_periodic_capstan`
- `metadata.median_interval_s`

### Invariante

Sauberes Nicht-Tape-Material darf durch den Fallback nicht flaggen (`severity == 0.0`).

---

## §6.4 GP-Gedächtnis pro Material & Genre

```text
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
| --- | --- | --- |
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

**MP3-Export:** LAME VBR V0 via FFmpeg (`ffmpeg -q:a 0`, adaptiv bis 320 kbps, ≈245 kbps Ø). Mono-Quellen bleiben Mono.
**VERBOTEN:** LAME CBR oder `pydub.export("out.mp3")` ohne explizite VBR-Parameter — CBR erzeugt Pre-Echo auf TDP-restaurierten Transienten (vgl. §4.5 Spec 04).

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

## §6.7 Tonträgerketten-Erkennung (bindend ab v9.10.97)

**Modul**: `forensics/medium_detector.py` — `MediumDetector.detect(audio, sr, file_ext=...)` — einziges autoritatives System ab v9.10.97.

**Architektur**: Zweistufige Fusion aus Bayesian-Gaussian-Scoring + physikalischer Inferenz.

### Phase 1: Bayesian Gaussian-Likelihood-Scoring

16 Materialmodelle (vinyl, shellac, cassette, reel_tape, reel_wire, lacquer_disc, wax_cylinder, cd_digital, dat, minidisc, mp3_low, mp3_high, aac, cassette_digital, vhs_audio, composite) mit je 7 Feature-Dimensionen:

| Feature | Dimension |
| --- | --- |
| `bandwidth_hz` | Effektive Bandbreite (−60 dBFS HF-Rolloff) |
| `snr_db` | Spektrales SNR (Median-PSD vs. Rauschboden P5) |
| `noise_color` | Rauschfarbe-Exponent (pink=2.0, weiß=0.0) |
| `crackle_density` | Anteil Samples > 4σ (Vinyl-Knackser, events/s) |
| `wow_flutter_index` | Pitch-Varianz [Amplituden-Std] über 100-ms-Fenster |
| `infrasonic_rms` | Sub-20 Hz normierter RMS (Vinyl-Rumble, Plattentellerlagerlärm) |
| `codec_type_code` | Codec-Fingerabdruck (0.0=analog, 1.0=digital) |

Log-Likelihood: `log P(m|features) = Σ log N(f; μ_m, σ_m)` → Softmax-Posterior.

**file_ext Prior-Zeroing**: Bei digitalen Dateiendungen (`.mp3`, `.aac`, `.ogg`, `.wma`, `.opus` u. a.) werden Analog-Posteriors auf 0 gezwungen — der Bayesian-Scorer kann keine analoge Primärquelle ausgeben.

### Phase 1b: Physikalische Analogquellen-Inferenz (NEU v9.10.97)

Greift wenn `file_ext ∈ DIGITAL_FILE_EXTS` und Bayesian kein `best_analog` findet. Physikalische Merkmale überleben bei Kassetten/Vinyl auch nach Codec-Encoding:

| Material | Erkennungsbedingung | Kalibrierung |
| --- | --- | --- |
| Vinyl | `infrasonic_rms > 0.030` (Plattentellerlagerlärm) ODER `crackle_density > 0.004 events/s` ODER `rotation_strength > 0.08` | μ_vinyl(infrasonic)=0.08, Schwelle = μ − 1σ |
| Shellac | `crackle_density > 0.015 AND infrasonic_rms > 0.040` (schlägt Vinyl-Erkennung) | |
| Kassette | `wow_flutter_index > 0.30` (Capstan/Pinch-Roller-Transport-Flutter) | Kassette: μ=1.5, σ=1.0; Vinyl-Eigenflutter max ≈ 0.15; Schwelle 0.30 = vinyl-sicher |
| Reel-Tape | `wow_flutter_index > 0.20 AND rotation_strength < 0.05` (kein Disc-Source) | μ_tape_wow=0.3, σ=0.3 |

Rückgabe: sortierte Liste `[(material_key, confidence)]` nach Signalketten-Reihenfolge (Disc vor Band vor Codec). Konfidenzen: [0.20, 0.85].

### Phase 1b: Physikalische Analogquellen-Inferenz (NEU v9.10.97)

Greift wenn `file_ext ∈ DIGITAL_FILE_EXTS` und Bayesian kein `best_analog` findet. Physikalische Merkmale überleben bei Kassetten/Vinyl auch nach Codec-Encoding:

| Material | Erkennungsbedingung | Kalibrierung |
| --- | --- | --- |
| Vinyl | `infrasonic_rms > 0.030` (Plattentellerlagerlärm) ODER `crackle_density > 0.004 events/s` ODER `rotation_strength > 0.08` | μ_vinyl(infrasonic)=0.08, Schwelle = μ − 1σ |
| Shellac | `crackle_density > 0.015 AND infrasonic_rms > 0.040` (schlägt Vinyl-Erkennung) | |
| Kassette | `wow_flutter_index > 0.30` (Capstan/Pinch-Roller-Transport-Flutter) | Kassette: μ=1.5, σ=1.0; Vinyl-Eigenflutter max ≈ 0.15; Schwelle 0.30 = vinyl-sicher |
| Reel-Tape | `wow_flutter_index > 0.20 AND rotation_strength < 0.05` (kein Disc-Source) | μ_tape_wow=0.3, σ=0.3 |

Rückgabe: sortierte Liste `[(material_key, confidence)]` nach Signalketten-Reihenfolge (Disc vor Band vor Codec). Konfidenzen: [0.20, 0.85].

### Phase 1c: Dolby / DBX NR-Erkennung (v9.10.128)

**Modul**: `backend/core/dolby_nr_detector.py` — `DolbyNRDetector.detect(audio, sr, material_type, era_decade)`.

Wird **automatisch** von `MediumDetector.detect()` aufgerufen wenn `primary_material ∈ {tape, reel_tape, wire_recording}`.

**Erkennungsmethode** (Frequenzband-Heuristik):

- `lf_rms` (300–1000 Hz): NR-unabhängige Referenz (Instrumenten-Grundtöne)
- `hf_rms` (800–4000 Hz + 4000–12000 Hz): Bandbereich mit stärkstem NR-Einfluss
- `hf_excess_db` = `20·log10(hf_rms / lf_rms)` − Erwartungswert_für_Material
- Slope-Analyse: DBX hat uniformen Slope (hf2 ≈ hf1), Dolby B/C wächst Richtung HF

| Typ | hf_excess Schwelle | Charakteristik |
| --- | --- | --- |
| Dolby B | ≥ 2.5 dB | +10 dB HF-Anhebung für leise Passagen; ~3-5 dB Durchschnitt |
| Dolby C | ≥ 5.0 dB | Doppelband, bis +20 dB; stark oberhalb 4 kHz |
| Dolby S | ≥ 3.5 dB | Dreifachband; ausgeprägt 2–8 kHz |
| DBX I | ≥ 4.0 dB | Breitbandig; uniformer ~9 dB/Oktave Slope |
| DBX II | ≥ 3.0 dB | Milder: ~6 dB/Oktave |

**Näherungsinversion** (statischer IIR Biquad-Kaskade):

- Dolby B: High-Shelf −4.5 dB @ 3 kHz (Q=0.6) + Peaking −2 dB @ 8 kHz (Q=0.7)
- Dolby C: High-Shelf −9 dB @ 4 kHz (Q=0.7) + Peaking −3 dB @ 10 kHz (Q=1.0)
- DBX I/II: gestaffelte High-Shelf-Kette (annähernde Steigung)

**Integration in `phase_04_eq_correction.py`**: Beide kwargs `dolby_nr_type` und `dolby_nr_confidence` werden nach RIAA/NAB-EQ angewendet. Activation via `kwargs["dolby_nr_type"] = result.dolby_nr_type`.

**`MediumDetectionResult`-Felder** (v9.10.128):

- `dolby_nr_type: str = "none"` — erkannter Typ
- `dolby_nr_confidence: float = 0.0` — Konfidenz [0..1]

**ACHTUNG — physikalische Limitierung**: Dies ist eine statische Näherung des level-abhängigen Dolby-Klangregelungssystems. Für Quellmaterial mit messbarer Klangverfälschung empfiehlt sich Re-Digitalisierung mit korrekt kalibriertem Wiedergabework.

### Phase 2: Transferketten-Aufbau

**Deep-Chain-Pflicht (v9.10.124)**: Kettenaufbau ist nicht auf Primärquelle + 1 Sekundärstufe begrenzt.

- Mehrere analoge Folgeglieder sind zulässig, solange die Reihenfolge kausal bleibt (`_MEDIUM_ORDER` monoton steigend).
- Digitale Zwischenstufen (`cd_digital`, `dat`) müssen eingefügt werden, wenn Posterior-Evidenz vorliegt.
- Lossy-Codec-Layer (`mp3_low`, `mp3_high`, `aac`, `streaming`) kommt am Kettenende und nur einmal.
- Nach Material-Key-Normalisierung (`cassette -> tape`, ...) werden benachbarte Duplikate konsolidiert; Link-Konfidenz bleibt der Maximalwert.

Ergebnis: `MediumDetectionResult.transfer_chain: list[str]` bildet reale 3+/4+/5+-Übertragungsketten vollständig ab.

```python
# Pflicht-Aufruf in allen Analyse-Kontexten:
from forensics.medium_detector import MediumDetector, get_medium_detector
result = get_medium_detector().detect(audio, sr, file_ext=Path(file_path).suffix)

# Kettenerkennung → MaterialType-Ableitung:
if result.transfer_chain:
    primary_material = result.transfer_chain[0]    # z. B. "vinyl"
    secondary_chain  = result.transfer_chain[1:]   # z. B. ["tape", "cd_digital", "mp3_low"]
    # → aktiviert kombinierte Phasen beider Materialien

# Kettenergebnis in RestorationResult.genealogy:
# SampleOperation(operation_type="chain_detection")
```

**VERBOTEN**: `MediumClassifier.classify_medium()` für Tonträgerketten-Erkennung. `MediumClassifier` kennt keinen Dateiendungs-Kontext und kann bei codec-enkodiertem Material "unknown" zurückgeben.

**Referenz-Fingerabdruck (Elke, Feb 2026):**

| Merkmal | Messwert | Diagnose |
| --- | --- | --- |
| infrasonic_rms | 0.065 | Vinyl-Rumble detektiert (> 0.030) |
| wow_flutter_index | 0.82 | Kassette-Flutter detektiert (> 0.30) |
| crackle_density | 0.006 events/s | Vinyl-Knackser detektiert (> 0.004) |
| file_ext | `.mp3` | Digital → Phase 1b physikalische Inferenz |
| Tonträgerkette | `vinyl → cassette → mp3_low` | Drei-stufige Degradation |

**Testpflicht (bindend)**:

- Unit-Test: 4-stufige Kette mit digitaler Zwischenstufe (`vinyl -> tape -> cd_digital -> mp3_low`).
- Unit-Test: `file_ext=.mp3` + physikalische Inferenz muss 4-stufige Kette liefern.
- Integrationstest: `source_fidelity_transfer_chain` muss unverändert bis `metadata["song_calibration"]` durchgereicht werden.

---

## §6.8 Importformate (Eingang)

| Format | Erweiterungen |
| --- | --- |
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
