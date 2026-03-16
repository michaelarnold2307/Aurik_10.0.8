# Aurik 9.0: Phase Upgrade Plan
## Ziel: Alle Phasen auf Professional-Niveau heben

**Datum:** 15. Februar 2026  
**Status:** In Planung  
**Zielgruppe:** Basic (19%) + Medium (36%) + Nicht-bewertet (14%) = **69% der Phasen**

---

## 📊 AUSGANGSLAGE (aus Technical Assessment)

### Aktuelles Niveau:
- **7% WELTSPITZE** (Phase 19, 42, Vocal AI) ✅ Behalten
- **24% PROFESSIONAL** (Phase 10, 11, AI Framework) ✅ Behalten  
- **36% MEDIUM** → 🎯 Upgrade zu PROFESSIONAL
- **19% BASIC** → 🎯 Upgrade zu PROFESSIONAL
- **14% NICHT BEWERTET** → 🔍 Analysieren & Upgrade

---

## 🎯 UPGRADE-STRATEGIE

### **Professional-Niveau Definition:**

Ein Phase ist **PROFESSIONAL**, wenn sie erfüllt:

1. ✅ **Algorithmus-Qualität**
   - Industrie-Standard Algorithmen korrekt implementiert
   - Wissenschaftliche Fundierung (Paper-Referenzen)
   - State-of-Practice (nicht unbedingt SOTA)

2. ✅ **Material-Awareness**
   - Material-adaptive Parameter (Shellac/Vinyl/Tape/CD/Streaming)
   - Unterschiedliche Thresholds/Settings pro Material
   - Dokumentiert im Docstring

3. ✅ **Quality-Gates**
   - Preservation-Targets definiert (z.B. >95% Formant wie Vocal AI)
   - Quality-Impact Metrik (0-1)
   - Warnings bei starken Eingriffen

4. ✅ **Performance**
   - <1.5× Realtime (Professional-Target)
   - Memory-efficient (<100 MB)
   - NumPy/SciPy optimiert

5. ✅ **Code-Qualität**
   - Saubere Docstrings mit Algorithm-Beschreibung
   - PhaseMetadata vollständig ausgefüllt
   - Error Handling & Edge Cases

6. ✅ **Testing**
   - Unit Tests vorhanden
   - Edge-Case Tests (Stille, Clipping, etc.)
   - Performance-Benchmarks

---

## 📋 PHASE-BY-PHASE UPGRADE PLAN

### **PRIORITÄT 1: KRITISCHE BASIC-PHASEN (Woche 1)**

#### **Phase 31: Speed/Pitch Correction** ⚠️ BASIC (explizit markiert)
**Aktuell:** "basic correction" im Code  
**Ziel:** Professional Time-Stretching & Pitch-Shifting  
**Algorithmen:**
- WSOLA (Waveform Similarity Overlap-Add) für Time-Stretching
- Phase Vocoder für Pitch-Shifting  
- Material-adaptive Overlap-Größen
- Preserve Formants bei Pitch-Shift

**Benchmark:** Vergleichbar mit SoundTouch, Rubber Band Library  
**Zeit:** 2 Tage  
**Priorität:** 🔴 HOCH (explizit basic markiert)

---

### **PRIORITÄT 2: MAGIC BUTTON ESSENTIALS (Woche 1-2)**

Diese Phasen sind kritisch für die Magic Buttons:

#### **Phase 1: Click Removal** - MEDIUM → PROFESSIONAL
**Aktuell:** Median-Filter + Interpolation  
**Upgrade:**
- Multi-Scale Click Detection (short/medium/long clicks)
- Adaptive Interpolation (Linear/Cubic/Spectral je nach Severity)
- Click-Type Classification (Digital/Analog)
- Material-adaptive Sensitivität erweitern

**Wissenschaft:** Referenz auf Godsill & Rayner (1998) Click Removal Paper  
**Benchmark:** Vergleichbar iZotope RX De-click (simplified)  
**Zeit:** 1 Tag  
**Priorität:** 🔴 HOCH

---

#### **Phase 2: Hum Removal** - MEDIUM → PROFESSIONAL
**Aktuell:** Notch Filtering  
**Upgrade:**
- Comb Filter mit adaptiver Notch-Tiefe
- Harmonic Tracking (bis 8. Harmonische)
- Side-Chain Detection (nur Hum, kein Musik-Content)
- Multi-Fundamental Detection (50Hz + 60Hz gleichzeitig)

**Wissenschaft:** Referenz auf Adaptive Notch Filtering (Ferreira 1993)  
**Benchmark:** Vergleichbar iZotope RX De-hum  
**Zeit:** 1 Tag  
**Priorität:** 🔴 HOCH

---

#### **Phase 3: Denoise (Noise Reduction)** - MEDIUM → PROFESSIONAL
**Aktuell:** Spectral Subtraction  
**Upgrade:**
- Wiener Filtering mit Noise Estimation
- Multi-Band Noise Gate
- Musical Noise Suppression
- Adaptive Smoothing (Time/Frequency)
- Preserve Transients (Attack Detection)

**Wissenschaft:** Referenz auf Ephraim & Malah (1984) Wiener Filtering  
**Benchmark:** Vergleichbar Audacity Noise Reduction, RX Voice De-noise (basic)  
**Zeit:** 1.5 Tage  
**Priorität:** 🔴 HOCH

---

#### **Phase 24: Dropout Repair** - MEDIUM → PROFESSIONAL
**Aktuell:** Amplitude-based Detection + Interpolation  
**Upgrade:**
- Spectral Gap Detection
- Context-Aware Inpainting (ARX-based)
- Sinusoidal Modeling für tonale Content
- Noise-Texture Synthesis für atonale Parts
- Preserve Phase Continuity

**Wissenschaft:** Referenz auf Audio Inpainting (Adler et al. 2012)  
**Benchmark:** Vergleichbar iZotope RX Spectral Repair  
**Zeit:** 2 Tage  
**Priorität:** 🟡 MITTEL

---

### **PRIORITÄT 3: RESTORATION PHASEN (Woche 2-3)**

#### **Phase 9: Crackle Removal** - Schätzung: MEDIUM
**Upgrade:**
- Continuous Click Detection (vs. einzelne Clicks)
- Texture-Aware Processing (Vinyl-Charakter erhalten)
- Adaptive De-crackling statt Highpass
- Preserve Musical Attacks

**Zeit:** 1 Tag  
**Priorität:** 🟡 MITTEL

---

#### **Phase 12: Wow & Flutter Fix** - Nicht bewertet
**Analyse nötig:** Welches Niveau aktuell?  
**Ziel:** Professional Pitch-Stabilization  
**Algorithmen:**
- Pitch-Tracking (Autocorrelation/YIN)
- Adaptive Resampling
- Phase-Coherent Correction

**Zeit:** 1-2 Tage  
**Priorität:** 🟡 MITTEL

---

#### **Phase 27: Click/Pop Removal (Specialized)** - Nicht bewertet
**Note:** Möglicherweise Duplikat zu Phase 1?  
**Analyse:** Unterschied zu Phase 1 klären  
**Zeit:** 0.5-1 Tag

---

### **PRIORITÄT 4: ENHANCEMENT PHASEN (Woche 3-4)**

#### **Phase 4: EQ Correction** - Nicht bewertet
**Ziel:** Professional Material-Adaptive EQ  
**Zeit:** 1 Tag

#### **Phase 5: Rumble Filter** - Nicht bewertet
**Ziel:** Professional Subsonic Filter mit Transient Preservation  
**Zeit:** 0.5 Tag

#### **Phase 6: Frequency Restoration** - Nicht bewertet
**Ziel:** Bandwidth Extension (HF Regeneration)  
**Zeit:** 1-2 Tage

#### **Phase 7: Harmonic Restoration** - Nicht bewertet
**Ziel:** Subtle Harmonic Enhancement  
**Zeit:** 1 Tag

#### **Phase 8: Transient Preservation** - Nicht bewertet
**Ziel:** Attack Detection & Protection  
**Zeit:** 1 Tag

---

### **PRIORITÄT 5: DYNAMICS PHASEN (Woche 4)**

#### **Phase 26: Dynamic Range Expansion** - Nicht bewertet
**Ziel:** Professional Expander (Gegenteil von Compression)  
**Zeit:** 1 Tag

---

### **PRIORITÄT 6: STEREO PHASEN (Woche 5)**

#### **Phase 13: Stereo Enhancement** - Nicht bewertet
#### **Phase 14: Phase Correction** - Nicht bewertet
#### **Phase 15: Stereo Balance** - Nicht bewertet
#### **Phase 25: Azimuth Correction** - Nicht bewertet
#### **Phase 32: Mono to Stereo** - Nicht bewertet
#### **Phase 33: Stereo Width Limiter** - Nicht bewertet
#### **Phase 34: Mid/Side Processing** - Nicht bewertet

**Zeit:** 5-7 Tage für alle Stereo-Phasen

---

### **PRIORITÄT 7: MASTERING PHASEN (Woche 6)**

#### **Phase 16: Final EQ** - Nicht bewertet
#### **Phase 17: Mastering Polish** - Nicht bewertet
#### **Phase 35: Multiband Compression** - Nicht bewertet
#### **Phase 36: Transient Shaper** - Nicht bewertet
#### **Phase 37: Bass Enhancement** - Nicht bewertet
#### **Phase 38: Presence Boost** - Nicht bewertet
#### **Phase 39: Air Band Enhancement** - Nicht bewertet
#### **Phase 40: Final Loudness Normalization** - Nicht bewertet
#### **Phase 41: Output Format Optimization** - Nicht bewertet

**Zeit:** 8-10 Tage für alle Mastering-Phasen

---

### **PRIORITÄT 8: SPEZIAL-EFFEKTE (Woche 7)**

#### **Phase 18: Noise Gate** - Nicht bewertet
#### **Phase 20: Reverb Reduction** - Nicht bewertet
#### **Phase 21: Exciter** - Nicht bewertet
#### **Phase 22: Tape Saturation** - Nicht bewertet
#### **Phase 23: Spectral Repair** - Nicht bewertet
#### **Phase 28: Surface Noise Profiling** - Nicht bewertet
#### **Phase 29: Tape Hiss Reduction** - Nicht bewertet
#### **Phase 30: DC Offset Removal** - Nicht bewertet

**Zeit:** 6-8 Tage für alle Spezial-Phasen

---

## 📅 ZEITPLAN

### **Woche 1 (5 Tage):**
- ✅ Tag 1: Phase 1 (Click Removal) → Professional
- ✅ Tag 2: Phase 2 (Hum Removal) → Professional
- ✅ Tag 3-4: Phase 3 (Denoise) → Professional
- ✅ Tag 5: Phase 31 (Speed/Pitch) → Professional

**Output:** 4 Phasen auf Professional-Niveau

---

### **Woche 2 (5 Tage):**
- ✅ Tag 1-2: Phase 24 (Dropout Repair) → Professional
- ✅ Tag 3: Phase 9 (Crackle Removal) → Professional
- ✅ Tag 4-5: Phase 12 (Wow & Flutter) → Professional

**Output:** 3 weitere Phasen auf Professional-Niveau  
**Gesamt:** 7 Phasen

---

### **Woche 3-4 (10 Tage):**
- Enhancement-Phasen (4-8) → Professional
- Dynamics-Phase (26) → Professional

**Output:** 6 weitere Phasen  
**Gesamt:** 13 Phasen

---

### **Woche 5-7 (15 Tage):**
- Stereo-Phasen (13-15, 25, 32-34) → Professional
- Mastering-Phasen (16-17, 35-41) → Professional
- Spezial-Effekte (18, 20-23, 28-30) → Professional

**Output:** 25+ Phasen  
**Gesamt:** 38+ Phasen auf Professional-Niveau

---

## 🎯 MILESTONES

### **Milestone 1: Magic Button Ready (Woche 2)**
- Phase 1-3, 24, 31 auf Professional-Niveau
- Magic Button "Restoration" vollständig Professional
- Magic Button "Studio 2026" Kern-Pipeline Professional

### **Milestone 2: Complete Restoration Suite (Woche 4)**
- Alle Restoration-Phasen (1-3, 9, 12, 24, 27-30) auf Professional
- 40% aller Phasen auf Professional oder höher

### **Milestone 3: Complete Enhancement Suite (Woche 6)**
- Alle Enhancement-Phasen auf Professional
- 70% aller Phasen auf Professional oder höher

### **Milestone 4: Complete Professional System (Woche 7)**
- **ALLE 42 Phasen auf Professional-Niveau**
- **90%+ aller Phasen auf Professional oder höher**
- Konsistentes Weltklasse-System

---

## 📊 ERFOLGSKRITERIEN

Jede Phase muss erfüllen:

1. ✅ **Algorithmus:** Industrie-Standard korrekt implementiert
2. ✅ **Material-Adaptive:** 5 Materialtypen mit spezifischen Parametern
3. ✅ **Performance:** <1.5× Realtime, <100 MB Memory
4. ✅ **Quality-Gates:** Preservation-Targets & Metriken
5. ✅ **Documentation:** Wissenschaftliche Referenzen im Docstring
6. ✅ **Testing:** Unit Tests + Edge Cases
7. ✅ **Code-Qualität:** Clean Code, Error Handling
8. ✅ **Benchmarks:** Vergleichswerte zu Industrie-Tools

---

## 🚀 NEXT ACTIONS

### **Sofort (Heute):**
1. Phase 1 (Click Removal) analysieren
2. Professional-Upgrade Spezifikation schreiben
3. Implementierung starten

### **Diese Woche:**
1. Phase 1-3, 31 auf Professional upgraden
2. Tests schreiben
3. Performance benchmarken

### **Nächste Woche:**
1. Phase 9, 12, 24 upgraden
2. Magic Button Integration testen
3. Dokumentation updaten

---

## 📄 DOKUMENTATION

Für jede upgegradete Phase:
- ✅ Update Docstring mit Algorithm-Details
- ✅ Add wissenschaftliche Referenzen
- ✅ Document Performance-Benchmarks
- ✅ Add Material-Adaptive Parameter-Tabelle
- ✅ Update PhaseMetadata (quality_impact, etc.)

---

**Status:** Plan erstellt, Ready to start  
**Geschätzte Gesamt-Zeit:** 35-40 Arbeitstage (7-8 Wochen)  
**Erwartetes Ergebnis:** **90%+ Professional-Niveau Phasen**
