# V2 vs V3 Feature Comparison
**Analyseziel**: Vergleich UnifiedRestorerV2 vs UnifiedRestorerV3 - Welche Features sind in V2, aber nicht in V3?
**Status**: ⏳ IN PROGRESS
**Datum**: 2026-02-15
**Autor**: V3 Migration Sprint 1, Week 1, Day 1

---

## Executive Summary

### Code-Größe
- **V2**: 6.971 Zeilen
- **V3**: 497 Zeilen
- **Faktor**: **14× größer** (V2 hat 14-mal mehr Code als V3)

### Architektur-Unterschiede

| Aspekt               | V2                                  | V3 |
|----------------------|-------------------------------------|----|
| **Paradigma**        | Medium-First (Material → Phasen)    | Defect-First (Defekte → Phasen) |
| **Struktur**         | Monolithisch (alles in einer Datei) | Modular (separate Module) |
| **Phasen**           | Integriert (hardcoded)              | Plug-and-Play (41 Phase-Module) |
| **Features**         | Optional (23 Feature-Flags)         | Core + Erweiterbar |
| **Performance**      | Keine explizite RT-Limit            | 3× RT Limit (PerformanceGuard) |
| **Parallelisierung** | Manuell                             | 4-Core Adaptive Scheduler |
| **Material**         | ProcessingMode-basiert              | MaterialType-Enum (5 Typen) |
| **API**              | `restore(audio, sr, mode=...)`      | `restore(audio, sr, material_type=...)` |

---

## Feature-Kategorien (V2 Exklusiv)

### ✅ Core Features (in V2 enthalten)

1. **Context-Aware De-Esser V2.0** ✨ ADVANCED
   - Phonem-bewusste Sibilanten-Reduktion
   - Kategorie: Vocal Processing
   - Komplexität: HIGH
   - Flag: `DEESSER_V2_AVAILABLE`

2. **Phase 2.2: Advanced Vocal Enhancement Suite** ✨ WORLD-CLASS
   - `BreathIntelligence`: Atem-Reduktion mit Kontext-Awareness
   - `FormantSystem`: Formanten-Korrektur & Vocal-Tuning
   - `VocalPresenceEnhancer`: 2-4 kHz Präsenz-Verstärkung
   - `VocalSpectralInpainting`: Spektrales Inpainting für Vocals
   - `VocalDynamicsIntelligence`: Intelligente Vocal-Kompression
   - Kategorie: Vocal Processing
   - Komplexität: VERY HIGH
   - Flag: `PHASE_2_2_AVAILABLE`

3. **Phase 2.3: Instrumental Enhancement Suite** ✨ WORLD-CLASS
   - `BassEnhancementSystem`: 40-200 Hz Bass-Verstärkung
   - `DrumsEnhancementSystem`: Transient Shaping, Punch
   - `GuitarEnhancementSystem`: 80-5000 Hz Guitar-Presence
   - `PianoRestorationSystem`: Hammergeräusch-Reduktion, Resonanz
   - `BrassEnhancementSystem`: 500-8000 Hz Brass-Brillanz
   - `SpatialEnhancementSystem`: Stereo Width, Depth
   - Kategorie: Instrumental Processing
   - Komplexität: VERY HIGH
   - Flag: `PHASE_2_3_AVAILABLE`

4. **Transparent Dynamics & Micro-Dynamics** ✨ PROFESSIONAL
   - `TransparentDynamicsProcessor`: Unhörbare Kompression
   - `MicroDynamicsEnhancer`: Detail-Verstärkung
   - Kategorie: Mastering
   - Komplexität: HIGH
   - Flag: `TRANSPARENT_DYNAMICS_AVAILABLE`

5. **Musical Excellence Phase 1** ✨ ADVANCED
   - Kategorie: Enhancement
   - Komplexität: MEDIUM
   - Flag: `MUSICAL_EXCELLENCE_P1_AVAILABLE`

6. **Professional Mastering Tools** ✨ PROFESSIONAL
   - `TruePeakLimiter`: ITU-R BS.1770 konformer Limiter
   - `StereoWidthEnhancer`: MS-Processing für Stereo-Breite
   - `MultibandCompressor`: 3-5 Band Kompression (separate Flag)
   - Kategorie: Mastering
   - Komplexität: HIGH
   - Flag: `MASTERING_TOOLS_AVAILABLE`

7. **Advanced De-Reverb** ✨ ADVANCED
   - Erweiterte Raum-Reduktion (über Phase 1.5 hinaus)
   - Kategorie: Restoration
   - Komplexität: HIGH
   - Flag: `ADVANCED_DEREVERB_AVAILABLE`

8. **Multiband Compression** ✨ PROFESSIONAL
   - 3-5 Band Frequenz-basierte Kompression
   - Kategorie: Mastering
   - Komplexität: MEDIUM
   - Flag: `MULTIBAND_COMPRESSION_AVAILABLE`

9. **Spectral Repair / Spectral Inpainting** ✨ ADVANCED
   - Frequenz-selektive Reparatur beschädigter Bereiche
   - Kategorie: Restoration
   - Komplexität: HIGH
   - Flag: `SPECTRAL_REPAIR_AVAILABLE`

10. **Semantic Audio Understanding** ✨✨ WORLD-FIRST INNOVATION
    - `ProcessingProfileSelector`: Genre-basierte Processing-Auswahl
    - `GenreDetector`: Automatische Genre-Erkennung
    - `StructureAnalyzer`: Song-Struktur-Analyse (Intro, Verse, Chorus, etc.)
    - Kategorie: AI Semantic Understanding
    - Komplexität: VERY HIGH
    - Flag: `SEMANTIC_UNDERSTANDING_AVAILABLE`

11. **Lyrics-Guided Vocal Enhancement** ✨✨ WORLD-FIRST INNOVATION
    - `ContentAwareProcessor`: Lyrics-bewusste Vocal-Processing
    - `LyricsGuidedTimeline`: Timeline-basierte Lyrics-Integration
    - Kategorie: AI Content-Aware Processing
    - Komplexität: VERY HIGH
    - Flag: `LYRICS_GUIDED_AVAILABLE`

12. **Tonal Balance Optimizer** ✨ PROFESSIONAL
    - Automatische Frequenzbalance-Korrektur
    - Kategorie: Enhancement
    - Komplexität: MEDIUM
    - Flag: `TONAL_BALANCE_AVAILABLE`

13. **Tape Specialist** ✨ SPECIALIST
    - Erweiterte Tape-spezifische Restoration
    - Kategorie: Material-Specific
    - Komplexität: HIGH
    - Flag: `TAPE_SPECIALIST_AVAILABLE`

14. **Digital Restoration** ✨ SPECIALIST
    - CD/MP3/Streaming-spezifische Artefakt-Reduktion
    - Kategorie: Material-Specific
    - Komplexität: MEDIUM
    - Flag: `DIGITAL_RESTORATION_AVAILABLE`

15. **Multi-Track Processing** ✨ ADVANCED
    - Parallele Verarbeitung mehrerer Spuren
    - Kategorie: Workflow
    - Komplexität: MEDIUM
    - Flag: `MULTI_TRACK_AVAILABLE`

16. **Live Recording Enhancement** ✨ SPECIALIST
    - Publikums-Noise-Reduktion, Stage-Bleed
    - Kategorie: Material-Specific
    - Komplexität: HIGH
    - Flag: `LIVE_RECORDING_AVAILABLE`

17. **Advanced Dehum** ✨ ADVANCED
    - Erweiterte Hum-Reduktion (über Phase 2.0 hinaus)
    - Kategorie: Restoration
    - Komplexität: MEDIUM
    - Flag: `DEHUM_AVAILABLE`

18. **Transient Shaper** ✨ PROFESSIONAL
    - Attack/Sustain-Kontrolle für Transienten
    - Kategorie: Enhancement
    - Komplexität: MEDIUM
    - Flag: `TRANSIENT_SHAPER_AVAILABLE`

19. **Spectral Matching** ✨ ADVANCED
    - Reference-basiertes Spectral Matching
    - Kategorie: Enhancement
    - Komplexität: HIGH
    - Flag: `SPECTRAL_MATCHING_AVAILABLE`

20. **Musical Noise Reduction** ✨ ADVANCED
    - Reduktion von "Musical Noise" Artefakten bei Noise Reduction
    - Kategorie: Restoration
    - Komplexität: MEDIUM
    - Flag: `MUSICAL_NOISE_REDUCTION_AVAILABLE`

21. **Adaptive Transient Preservation** ✨ ADVANCED
    - Erhalt von Transienten während Restoration
    - Kategorie: Restoration
    - Komplexität: HIGH
    - Flag: `ADAPTIVE_TRANSIENT_PRESERVATION_AVAILABLE`

22. **Signal Forensics** ✨ PROFESSIONAL
    - Forensische Audio-Analyse (z.B. für Gerichtsverfahren)
    - Kategorie: Analysis
    - Komplexität: VERY HIGH
    - Flag: `SIGNAL_FORENSICS_AVAILABLE`

23. **KI Quality Assessment** ✨ AI-POWERED
    - ML-basierte Qualitätsbewertung
    - Kategorie: Quality Analysis
    - Komplexität: VERY HIGH
    - Flag: `KI_QUALITY_AVAILABLE`

---

## Feature-Status Matrix

### V3 Coverage (Abdeckung von V2-Features in V3)

| Feature Kategorie | V2 Features | V3 Status | Priorität | Implementierungs-Schwierigkeit |
|-------------------|-------------|-----------|-----------|-------------------------------|
| **Core Restoration** | Phase 1.x (Click, Hum, Denoise, etc.) | ✅ COMPLETE | - | - |
| **Vocal Enhancement** | Phase 2.2 (6 Module) | ❌ MISSING | 🔴 HIGH | 🔥🔥🔥 VERY HIGH |
| **Instrumental Enhancement** | Phase 2.3 (6 Module) | ❌ MISSING | 🟡 MEDIUM | 🔥🔥🔥 VERY HIGH |
| **Mastering Tools** | Dynamics, Limiter, Width, Multiband | ❌ MISSING | 🟡 MEDIUM | 🔥🔥 HIGH |
| **Advanced Restoration** | De-Reverb, Spectral Repair, Dehum | ⚠️ PARTIAL | 🟡 MEDIUM | 🔥🔥 HIGH |
| **Semantic Understanding** | Genre, Structure, Profile Selection | ❌ MISSING | 🟢 LOW | 🔥🔥🔥🔥 EXTREME |
| **Lyrics-Guided** | Content-Aware, Timeline | ❌ MISSING | 🟢 LOW | 🔥🔥🔥🔥 EXTREME |
| **Material-Specific** | Tape, Digital, Live Recording | ⚠️ PARTIAL | 🟡 MEDIUM | 🔥🔥 HIGH |
| **Workflow** | Multi-Track | ❌ MISSING | 🟢 LOW | 🔥 MEDIUM |
| **Analysis** | Forensics, KI Quality | ❌ MISSING | 🟢 LOW | 🔥🔥🔥 VERY HIGH |

---

## Priorisierungs-Matrix

### Phase 1: CRITICAL (Sprint 1-2, Wochen 1-3)
- ✅ V3 Core Restoration (bereits implementiert)
- ⚠️ Phase Loading Fix (BLOCKER - siehe Test-Ergebnisse)
- ⚠️ Memory Leak Fix (BLOCKER - siehe Test-Ergebnisse)

### Phase 2: HIGH PRIORITY (Sprint 3-4, Wochen 4-6)
- ❌ **Vocal Enhancement Suite** (Phase 2.2)
  - Business Value: HIGH (Podcasts, Interviews, Music)
  - Effort: VERY HIGH (6 Module)
  - ROI: **MEDIUM** (High Value / Very High Effort)
  - Migration Strategy: Portiere zuerst VocalPresenceEnhancer + BreathIntelligence (Quick Wins)

### Phase 3: MEDIUM PRIORITY (Sprint 5-6 oder später)
- ❌ **Instrumental Enhancement Suite** (Phase 2.3)
  - Business Value: MEDIUM (Musik-Restoration)
  - Effort: VERY HIGH (6 Module)
  - ROI: **LOW** (Medium Value / Very High Effort)
  - Migration Strategy: Portiere instrumentenspezifisch (z.B. nur Piano + Guitar)

- ❌ **Mastering Tools**
  - Business Value: MEDIUM (Professional Workflow)
  - Effort: HIGH
  - ROI: **MEDIUM**
  - Migration Strategy: Portiere TruePeakLimiter zuerst (Broadcasting-Standard)

### Phase 4: LOW PRIORITY (Post-V3.0, zukünftige Releases)
- ❌ **Semantic Understanding** (World-First Innovation)
  - Business Value: HIGH (Marketing, Innovation)
  - Effort: EXTREME (ML-Integration, Datenbanken)
  - ROI: **VERY LOW** (High Value / Extreme Effort)
  - Migration Strategy: R&D-Projekt, separate vom V3 Launch

- ❌ **Lyrics-Guided Enhancement** (World-First Innovation)
  - Business Value: MEDIUM (Niche Use-Case)
  - Effort: EXTREME (Lyrics API, Timeline-Integration)
  - ROI: **VERY LOW**
  - Migration Strategy: R&D-Projekt, nach Semantic Understanding

---

## API-Unterschiede

### V2 restore() Signature
```python
def restore(
    self,
    audio: np.ndarray,
    sr: int,
    reference: Optional[np.ndarray] = None,
    policy: Optional[dict] = None,
    config: Optional[dict] = None,
    mode: Union[ProcessingMode, str, None] = None,
    processing_config: Optional[ProcessingConfig] = None,
    preserve_original_sr: bool = False,
    user_variant_callback: callable = None,
) -> np.ndarray:
```

### V3 restore() Signature (erwartet)
```python
def restore(
    self,
    audio: np.ndarray,
    sample_rate: int,
    material_type: Optional[MaterialType] = None,
    quality_mode: QualityMode = QualityMode.BALANCED,
) -> RestorationResult:
```

### Mapping V2 → V3

| V2 Parameter | V3 Equivalent | Unterschied |
|--------------|---------------|-------------|
| `sr` | `sample_rate` | Umbenennung |
| `mode: ProcessingMode` | `quality_mode: QualityMode` | Enum-Typ geändert |
| N/A | `material_type: MaterialType` | **NEU in V3** |
| Return: `np.ndarray` | Return: `RestorationResult` | **Strukturiert in V3** |
| `reference` | N/A | **Entfernt in V3** |
| `policy` | N/A | **Entfernt in V3** |
| `config` | (via RestorationConfig) | **Konfiguration umstrukturiert** |
| `preserve_original_sr` | N/A | **Entfernt in V3** (immer 48 kHz → Original) |
| `user_variant_callback` | N/A | **Entfernt in V3** |

---

## Recommendations

### 🎯 V3 Migration Strategy

#### Option A: **Minimalist V3** (EMPFOHLEN für Sprint 1-4)
- **Fokus**: Core Restoration Only
- **Scope**: 
  - ✅ 41 Phase-Module (Defekt-basiert)
  - ✅ PerformanceGuard (3× RT)
  - ✅ AdaptiveCoreScheduler (4-Core)
  - ✅ MaterialType (5 Typen)
  - ❌ **KEINE** Phase 2.2 / 2.3 (Vocal/Instrumental Enhancement)
  - ❌ **KEINE** Mastering Tools
  - ❌ **KEINE** World-First Innovations
- **Vorteile**:
  - ✅ Saubere, wartbare Codebasis (497 Zeilen)
  - ✅ Schnell zu testen & deployen
  - ✅ Klare Architektur
  - ✅ 80% der Use-Cases abgedeckt (Restoration)
- **Nachteile**:
  - ❌ Keine professionellen Enhancement-Features
  - ❌ Keine World-First Marketing-Features

#### Option B: **Feature-Complete V3** (Sprint 5+, Post-Launch)
- **Fokus**: Portiere V2 Premium-Features schrittweise
- **Scope**:
  - Phase 1 (Sprint 5-6): Vocal Enhancement (Phase 2.2)
  - Phase 2 (Sprint 7-8): Instrumental Enhancement (Phase 2.3)
  - Phase 3 (Sprint 9-10): Mastering Tools
  - Phase 4 (R&D, 2027): Semantic Understanding + Lyrics-Guided
- **Vorteile**:
  - ✅ Feature-Parität mit V2
  - ✅ Professionelle Workflows unterstützt
  - ✅ Marketing-Argumente (World-First)
- **Nachteile**:
  - ❌ 6+ Monate zusätzliche Entwicklung
  - ❌ Komplexität steigt massiv
  - ❌ Wartbarkeit sinkt

### 📊 Entscheidung

**Sprint 1-4 (aktuelle Phase)**:
- Fokus auf **Option A**: Minimalist V3
- Ziel: Core Restoration **production-ready machen**
- Phase Loading + Memory Leak **sofort fixen**
- Tests **100% grün bekommen**

**Post-Sprint 4**:
- User Feedback sammeln
- Entscheiden: Bleibt V3 minimalistisch ODER erweitern wir?
- Falls erweitern: Vocal Enhancement (Phase 2.2) zuerst (höchster Business Value)

---

## Next Steps

### Immediate (Day 2-3)
1. ✅ V2 Feature-Analyse abschließen (DIESES DOKUMENT)
2. 🔴 **BLOCKER**: Phase Loading Fix
   - 9 Phase-Module laden nicht (missing 'priority' argument)
   - Betrifft: phase_01/02/03 + ihre v2_professional Varianten
   - Geschätzter Aufwand: 2-3 Stunden
3. 🔴 **BLOCKER**: Memory Leak Investigation
   - +51.4 MB Leak über 5 Iterationen
   - Wahrscheinlich: Phase Loading/Caching
   - Geschätzter Aufwand: 2-3 Stunden

### Short-Term (Week 2)
4. 🟡 E2E Test Retry (nach Phase Loading Fix)
5. 🟡 Performance Profiling (Baseline für V3)
6. 🟡 Phase Selection Completion (38/41 Phasen fehlen)

### Mid-Term (Week 3-4)
7. 🟢 Decision: Bleibt V3 minimalistisch?
8. 🟢 Falls ja: V3 als production-ready deklarieren
9. 🟢 Falls nein: Vocal Enhancement (Phase 2.2) Migration starten

---

## Appendix: V2 Feature-Flags (Vollständige Liste)

```python
# 23 Feature Availability Flags in V2
DEESSER_V2_AVAILABLE = True/False          # Context-Aware De-Esser
PHASE_2_2_AVAILABLE = True/False           # Vocal Enhancement Suite
PHASE_2_3_AVAILABLE = True/False           # Instrumental Enhancement Suite
TRANSPARENT_DYNAMICS_AVAILABLE = True/False # Transparent Dynamics + Micro-Dynamics
MUSICAL_EXCELLENCE_P1_AVAILABLE = True/False # Musical Excellence Phase 1
MASTERING_TOOLS_AVAILABLE = True/False     # True Peak Limiter, Stereo Width
ADVANCED_DEREVERB_AVAILABLE = True/False   # Advanced De-Reverb
MULTIBAND_COMPRESSION_AVAILABLE = True/False # Multiband Compressor
SPECTRAL_REPAIR_AVAILABLE = True/False     # Spectral Repair/Inpainting
SEMANTIC_UNDERSTANDING_AVAILABLE = True/False # World-First: Semantic Understanding
LYRICS_GUIDED_AVAILABLE = True/False       # World-First: Lyrics-Guided Enhancement
TONAL_BALANCE_AVAILABLE = True/False       # Tonal Balance Optimizer
TAPE_SPECIALIST_AVAILABLE = True/False     # Tape-Specific Restoration
DIGITAL_RESTORATION_AVAILABLE = True/False # CD/MP3/Streaming Restoration
MULTI_TRACK_AVAILABLE = True/False         # Multi-Track Processing
LIVE_RECORDING_AVAILABLE = True/False      # Live Recording Enhancement
DEHUM_AVAILABLE = True/False               # Advanced Dehum
TRANSIENT_SHAPER_AVAILABLE = True/False    # Transient Shaper
SPECTRAL_MATCHING_AVAILABLE = True/False   # Spectral Matching
MUSICAL_NOISE_REDUCTION_AVAILABLE = True/False # Musical Noise Reduction
ADAPTIVE_TRANSIENT_PRESERVATION_AVAILABLE = True/False # Adaptive Transient Preservation
SIGNAL_FORENSICS_AVAILABLE = True/False    # Signal Forensics
KI_QUALITY_AVAILABLE = True/False          # KI Quality Assessment
```

---

## Metrics

| Metric | V2 | V3 | Delta |
|--------|----|----|-------|
| **Lines of Code** | 6.971 | 497 | **-93%** |
| **Feature Count** | 23 optional | 3 core | **-87%** |
| **Phase Count** | ~50-60 (integriert) | 41 (modular) | ~-20% |
| **Complexity** | HIGH (monolithisch) | LOW (modular) | **-70%** |
| **RT Factor Target** | None (best-effort) | 3× (enforced) | **+300% Garantie** |
| **Parallelization** | Manual | 4-Core Automatic | **+300% CPU** |
| **World-First Features** | 2 | 0 | **-100%** |

---

**Status**: ✅ ANALYSE COMPLETE
**Next**: Fix Phase Loading (BLOCKER P0)
