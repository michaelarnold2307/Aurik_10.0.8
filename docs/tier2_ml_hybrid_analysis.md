# Tier 2 ML-Hybrid Kandidaten - Kosten-Nutzen-Analyse

**Datum:** 16. Februar 2026  
**Status:** Tier 1 Complete (10/10 Phasen) ✅

---

## Aktuelle ML-Hybrid Coverage

**Tier 1 Complete (10 Phasen):**
| Phase | Name | ML-Komponente | Status |
| --- | --- | --- | --- |
| 01 | Click Removal | DeepFilterNet | ✅ |
| 02 | Hum Removal | DeepFilterNet | ✅ |
| **03** | **Denoise** | **OMLSA + Resemble** | ✅ NEW |
| 09 | Crackle Removal | BANQUET Vinyl | ✅ |
| **12** | **Wow/Flutter** | **YIN + CREPE** | ✅ NEW |
| 18 | Noise Gate | Silero VAD | ✅ |
| **20** | **Reverb Reduction** | **DSP + DCCRN** | ✅ NEW |
| 23 | Spectral Repair | AudioSR | ✅ |
| 24 | Dropout Repair | AudioSR | ✅ |
| 29 | Tape Hiss | DeepFilterNet | ✅ |

**Abdeckung:** 10/42 Phasen = **23.8%**

---

## Tier 2 Kandidaten - Prioritäts-Matrix

### 🔥 Hochpriorität (Großer Qualitätsgewinn, moderater Aufwand)

#### 1. Phase 06/07: Frequency & Harmonic Restoration + NVSR ⭐⭐⭐⭐⭐

- **ML-Komponente:** Neural Vocoder Super Resolution (NVSR)
- **Use Case:** Lo-Fi Material (MP3 64kbps, Telefon, alte Aufnahmen)
- **Erwarteter Gewinn:** +0.08-0.12 Quality (Bandwidth Extension 8 kHz → 20 kHz)
- **Aufwand:** ~3-4 Tage (NVSR Plugin + Hybrid Integration)
- **Performance:** ~2-3× RT (MAXIMUM mode)
- **Empfehlung:** ✅ **JA** - Großer Nutzen für Lo-Fi Material

#### 2. Phase 19: De-Esser + ML Phoneme Detection ⭐⭐⭐⭐

- **ML-Komponente:** Phoneme Detection (Whisper/Wav2Vec2)
- **Use Case:** Vocals mit Sibilance (Podcast, Gesang)
- **Erwarteter Gewinn:** +0.05-0.08 Quality (chirurgische De-Essing)
- **Aufwand:** ~2-3 Tage (Phoneme Plugin + Hybrid Integration)
- **Performance:** ~1.5-2× RT (MAXIMUM mode)
- **Empfehlung:** ✅ **JA** - Precision De-Essing für Vocals

#### 3. Phase 31: Speed/Pitch Correction + ML Pitch Tracking ⭐⭐⭐⭐

- **ML-Komponente:** CREPE (already available!) + Phase Vocoder
- **Use Case:** Falsche Abspielgeschwindigkeit (Tape, Vinyl)
- **Erwarteter Gewinn:** +0.06-0.10 Quality (korrekte Tonhöhe)
- **Aufwand:** ~1-2 Tage (CREPE bereits implementiert in Phase 12)
- **Performance:** ~1× RT (MAXIMUM mode)
- **Empfehlung:** ✅ **JA** - Code-Reuse von Phase 12, geringer Aufwand

---

### 🟡 Mittelpriorität (Moderater Gewinn, höherer Aufwand)

#### 4. Phase 25: Azimuth Correction + ML Head Alignment ⭐⭐⭐

- **ML-Komponente:** ML-basierte Kopf-Fehl-Ausrichtung Detection
- **Use Case:** Tape mit Azimuth-Problemen (Phase Cancellation)
- **Erwarteter Gewinn:** +0.04-0.06 Quality (Stereo Imaging)
- **Aufwand:** ~4-5 Tage (neue ML-Komponente notwendig)
- **Performance:** ~1.5× RT (MAXIMUM mode)
- **Empfehlung:** 🟡 **OPTIONAL** - Nischenfall, selten kritisch

#### 5. Phase 42: Vocal Enhancement + ML Vocal Separation ⭐⭐⭐

- **ML-Komponente:** Demucs v5 (already available!)
- **Use Case:** Vocals isolieren + Enhancement
- **Erwarteter Gewinn:** +0.05-0.08 Quality (Vocal Clarity)
- **Aufwand:** ~3-4 Tage (Demucs bereits integriert)
- **Performance:** ~3-4× RT (MAXIMUM mode)
- **Empfehlung:** 🟡 **OPTIONAL** - Nice-to-have, nicht kritisch

#### 6. Phase 28: Surface Noise Profiling + ML Noise Learning ⭐⭐

- **ML-Komponente:** ML-basiertes Noise Profile Learning
- **Use Case:** Adaptives Noise Learning für komplexe Rauschen
- **Erwarteter Gewinn:** +0.03-0.05 Quality
- **Aufwand:** ~5-6 Tage (neue ML-Architektur)
- **Performance:** ~2× RT (MAXIMUM mode)
- **Empfehlung:** 🔴 **NEIN** - Aufwand > Nutzen

---

### 🔴 Niedrigpriorität (Geringer Gewinn oder DSP ausreichend)

#### 7-42. Restliche Phasen ⭐

- **Phasen:** EQ, Compression, Limiting, Stereo Processing, etc.
- **Grund:** DSP bereits exzellent, ML bringt kaum Verbesserung
- **Empfehlung:** 🔴 **NEIN** - DSP ausreichend

---

## Kosten-Nutzen-Zusammenfassung

### Tier 2 Empfehlung: **3 Hochprioritäre Phasen**

| Phase | Aufwand | Gewinn | ROI | Empfehlung |
|-------|---------|--------|-----|------------|
| **06/07 NVSR** | 3-4 Tage | +0.08-0.12 | ⭐⭐⭐⭐⭐ | ✅ **JA** |
| **19 Phoneme** | 2-3 Tage | +0.05-0.08 | ⭐⭐⭐⭐ | ✅ **JA** |
| **31 Pitch** | 1-2 Tage | +0.06-0.10 | ⭐⭐⭐⭐⭐ | ✅ **JA** |
| **GESAMT** | **6-9 Tage** | **+0.19-0.30** | ⭐⭐⭐⭐⭐ | ✅ **JA** |

### Erwartetes Ergebnis nach Tier 2:

- **ML-Hybrid Coverage:** 23.8% → 31.0% (13/42 Phasen)
- **Quality Improvement:** +0.19-0.30 für Lo-Fi & Vocal Material
- **Zeitaufwand:** 6-9 Tage (1-1.5 Wochen)
- **Overall Quality:** 0.88-0.90 → **0.95-1.00** (Near-Perfect) 🚀

---

## Empfehlung

### ✅ **Option 1: Tier 2 Minimal (Phase 31 only)** - 1-2 Tage

- **Nur Phase 31:** Speed/Pitch Correction (CREPE Code-Reuse)
- **Gewinn:** +0.06-0.10 Quality
- **Aufwand:** Minimal (CREPE bereits da)
- **Status:** Quick Win ⚡

### ✅ **Option 2: Tier 2 Optimal (Phase 06/07 + 19 + 31)** - 6-9 Tage

- **3 Hochprioritäre Phasen**
- **Gewinn:** +0.19-0.30 Quality
- **Coverage:** 31% ML-Hybrid
- **Status:** Recommended 🎯

### 🟡 **Option 3: Production Release jetzt, Tier 2 später** - 0 Tage

- **Tier 1 ist bereits exzellent** (10/10 kritische Phasen)
- **User Feedback sammeln** → Priorisierung basierend auf echten Bedürfnissen
- **Status:** Safe Choice 🛡️

---

## Meine Empfehlung: **Option 2 (Tier 2 Optimal)** 🎯

**Begründung:**
1. **Phase 31** ist Quick Win (CREPE bereits da, 1-2 Tage)
2. **Phase 06/07** löst massives Lo-Fi Problem (MP3 64kbps, Telefon)
3. **Phase 19** löst Vocal Sibilance perfekt (Podcasts, Musik)
4. **6-9 Tage** sind überschaubar, **+0.19-0.30 Quality** ist signifikant
5. **Near-Perfect Quality** (0.95-1.00) wäre Alleinstellungsmerkmal

**Danach:** Production Release RC2 mit **13/42 ML-Hybrid Phasen** 🚀
