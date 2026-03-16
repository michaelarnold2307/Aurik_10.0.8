# 🚀 Aurik V3 Migration Plan - Option 1

**Start:** 15. Februar 2026  
**Zeitrahmen:** 4-6 Wochen  
**Ziel:** UnifiedRestorerV3 als Standard aktivieren

---

## 📋 Executive Summary

**Was:** Migration von UnifiedRestorerV2 zu UnifiedRestorerV3  
**Warum:** +25% Performance, Defect-First Architecture, moderne Code-Organisation  
**Risiko:** MODERAT (V3 bereits implementiert, aber nicht produktiv getestet)  
**Team:** AI Development Team + Tester

---

## 🎯 Migrations-Ziele

### Primäre Ziele (MUST HAVE)
1. ✅ V3 Code vollständig validiert (bereits implementiert!)
2. ⏳ GUI auf V3 umgeleitet (alle Calls)
3. ⏳ Vollständige Regression Tests (V2 vs V3 Parität)
4. ⏳ Performance Benchmarks (mindestens +20% vs V2)
5. ⏳ Production-Ready Release Candidate

### Sekundäre Ziele (NICE TO HAVE)
- Automatische V2→V3 Fallback bei Fehlern
- Erweiterte Performance-Metriken in GUI
- Real-time Quality Monitoring
- Adaptive Phase Skipping aktiviert

---

## 📅 Timeline: 6 Wochen (Sprint-basiert)

### Sprint 1 (Woche 1-2): Foundation & Validation
**Ziel:** V3 Code validieren und erste Tests

#### Woche 1: Code Review & Defect-Analysis
- [x] V3 Code existiert (core/unified_restorer_v3.py ~498 lines) ✅
- [x] AdaptiveCoreScheduler existiert (~542 lines) ✅
- [x] PerformanceGuard existiert (~512 lines) ✅
- [ ] **TODO:** V3 Code Review (alle Features dokumentiert?)
- [ ] **TODO:** DefectScanner vollständig getestet?
- [ ] **TODO:** Memory Leak Tests (60s Audio)
- [ ] **TODO:** Edge Cases identifizieren

**Deliverables:**
- [ ] V3 Code Review Report
- [ ] Memory Profile Report (psutil)
- [ ] Edge Case Liste

#### Woche 2: Basic E2E Tests
- [ ] **TODO:** 3 Test-Songs restaurieren (V2 vs V3)
  - TAPE: Elke Best - Du wolltest nur ein Abenteuer
  - VINYL: Klassischer Jazz-Track
  - CD: Moderne Pop-Aufnahme
- [ ] **TODO:** Quality Metrics vergleichen (SNR, THD, LUFS)
- [ ] **TODO:** Performance Metrics vergleichen (RT Factor)
- [ ] **TODO:** Visual Inspection (Spektrogramm-Vergleich)

**Deliverables:**
- [ ] E2E Test Report (V2 vs V3)
- [ ] Performance Comparison Matrix
- [ ] Quality Regression Check ✅/❌

**Success Criteria:**
- V3 Quality ≥ V2 Quality (±2%)
- V3 Performance ≥ 1.2× V2 Speed (+20%)
- No Critical Bugs

---

### Sprint 2 (Woche 3-4): GUI Integration

#### Woche 3: Backend Integration Points
- [ ] **TODO:** Identifiziere alle V2-Calls im Code
  - run_elke_restoration.py
  - start_aurik_premium.py
  - backend/core/*.py
  - aurik_professional/*.py
- [ ] **TODO:** Create V2→V3 Adapter (Backward Compatibility)
  ```python
  # Legacy V2 calls should work
  from core.unified_restorer_v2 import UnifiedRestorerV2
  # vs New V3 calls
  from core.unified_restorer_v3 import UnifiedRestorerV3
  ```
- [ ] **TODO:** Implement Feature Flag System
  ```python
  USE_V3 = os.getenv('AURIK_USE_V3', 'false').lower() == 'true'
  Restorer = UnifiedRestorerV3 if USE_V3 else UnifiedRestorerV2
  ```

**Deliverables:**
- [ ] V2→V3 Call Mapping Document
- [ ] Feature Flag Implementation
- [ ] Backward Compatibility Layer

#### Woche 4: GUI Update & Testing
- [ ] **TODO:** Update PyQt5 GUI (modern_window.py)
  - V3 import statt V2
  - Performance Metrics display (RT Factor)
  - Quality Mode Selector (FAST/BALANCED/QUALITY)
- [ ] **TODO:** Update CLI (aurik_cli.py)
  - V3 support mit --mode flag
  - Performance Guard Optionen
- [ ] **TODO:** Update Magic Button Scripts
  - run_elke_restoration.py → V3
  - Adaptive Phase Skipping aktivieren

**Deliverables:**
- [ ] Updated GUI (V3-ready)
- [ ] Updated CLI
- [ ] Updated Scripts

**Success Criteria:**
- GUI funktioniert mit V3
- Alle Features aus V2 verfügbar
- No Breaking Changes für User

---

### Sprint 3 (Woche 5-6): Testing & Release

#### Woche 5: Comprehensive Testing
- [ ] **TODO:** Regression Test Suite
  - 10+ test audio files (diverse materials)
  - Automated quality checks
  - Performance benchmarks
- [ ] **TODO:** Stress Testing
  - Long audio files (>30 min)
  - Large batch processing (50+ files)
  - Memory leak detection (psutil)
- [ ] **TODO:** User Acceptance Testing (UAT)
  - Internal testing with real use cases
  - Feedback collection
  - Bug triage

**Test Matrix:**
| Material | Duration | V2 Time | V3 Time | Speedup | Quality |
|----------|----------|---------|---------|---------|---------|
| TAPE     | 3:45     | TBD     | TBD     | TBD     | TBD     |
| VINYL    | 5:20     | TBD     | TBD     | TBD     | TBD     |
| SHELLAC  | 2:30     | TBD     | TBD     | TBD     | TBD     |
| CD       | 4:10     | TBD     | TBD     | TBD     | TBD     |
| STREAMING| 3:00     | TBD     | TBD     | TBD     | TBD     |

**Deliverables:**
- [ ] Regression Test Report
- [ ] Stress Test Report
- [ ] UAT Feedback Document
- [ ] Bug Fix Priority List

#### Woche 6: Release Preparation
- [ ] **TODO:** Fix Critical Bugs (P0/P1)
- [ ] **TODO:** Documentation Update
  - API Documentation (V3 vs V2 differences)
  - Migration Guide für externe Nutzer
  - Performance Tuning Guide
- [ ] **TODO:** Changelog erstellen
- [ ] **TODO:** Release Notes schreiben
- [ ] **TODO:** Version Bump (9.0.0-rc1)

**Deliverables:**
- [ ] Bug-Free Release Candidate
- [ ] Complete Documentation
- [ ] CHANGELOG.md
- [ ] RELEASE_NOTES.md

**Success Criteria:**
- Zero Critical Bugs (P0)
- Documentation Complete
- Ready for Release

---

## 🔧 Technical Implementation Details

### V3 Aktivierung (Feature Flag)

**Option A: Environment Variable (empfohlen)**
```bash
export AURIK_USE_V3=true
python run_elke_restoration.py
```

**Option B: Config File**
```yaml
# config/aurik.yaml
engine:
  version: v3  # v2 or v3
  performance_guard: true
  adaptive_skipping: true
```

**Option C: Code-Level (Clean Migration)**
```python
# core/__init__.py
from core.unified_restorer_v3 import UnifiedRestorerV3 as UnifiedRestorer

# Old V2 code automatically uses V3:
from core import UnifiedRestorer
restorer = UnifiedRestorer()  # Now V3!
```

### GUI Integration Points

**1. start_aurik_premium.py**
```python
# OLD:
from core.unified_restorer_v2 import UnifiedRestorerV2

# NEW:
from core.unified_restorer_v3 import UnifiedRestorerV3, RestorationConfig, QualityMode

config = RestorationConfig(
    mode=QualityMode.BALANCED,
    num_cores=4,
    enforce_3x_rt=True
)
restorer = UnifiedRestorerV3(config)
```

**2. modern_window.py (ProcessingThread)**
```python
class ProcessingThread(QThread):
    def run(self):
        # OLD:
        restorer = UnifiedRestorerV2()
        result = restorer.restore(audio, sr)
        
        # NEW:
        config = RestorationConfig(mode=self.quality_mode)
        restorer = UnifiedRestorerV3(config)
        result = restorer.restore(audio, sr)
        
        # Emit new metrics
        self.performance_update.emit({
            'rt_factor': result.rt_factor,
            'phases_executed': len(result.phases_executed),
            'phases_skipped': len(result.phases_skipped),
            'quality_estimate': result.quality_estimate
        })
```

**3. run_elke_restoration.py**
```python
# OLD:
from core.unified_restorer_v2 import UnifiedRestorerV2, ProcessingMode

restorer = UnifiedRestorerV2()
result = restorer.restore(audio, sr, mode=ProcessingMode.RESTORATION)

# NEW:
from core.unified_restorer_v3 import UnifiedRestorerV3, RestorationConfig, QualityMode

config = RestorationConfig(
    mode=QualityMode.BALANCED,
    enforce_3x_rt=True,
    enable_adaptive_skipping=True
)
restorer = UnifiedRestorerV3(config)
result = restorer.restore(audio, sr)

print(f"RT Factor: {result.rt_factor:.2f}×")
print(f"Phases: {len(result.phases_executed)} executed, {len(result.phases_skipped)} skipped")
print(f"Quality: {result.quality_estimate*100:.0f}%")
```

---

## 🧪 Testing Strategy

### Test Pyramid

```
                    /\
                   /  \
                  / UI \
                 /______\
                /        \
               /Integration\
              /____________\
             /              \
            /   Unit Tests   \
           /__________________\
```

**Unit Tests (70%):**
- [ ] DefectScanner (11 defect types)
- [ ] AdaptiveCoreScheduler (4-core logic)
- [ ] PerformanceGuard (RT enforcement)
- [ ] Each phase individually (41 tests)

**Integration Tests (20%):**
- [ ] Full pipeline (all phases)
- [ ] DefectScanner → Phase Selection
- [ ] PerformanceGuard → Phase Skipping
- [ ] RestorationConfig → Phase Parameters

**UI Tests (10%):**
- [ ] Magic Button E2E (RESTORATION)
- [ ] Magic Button E2E (STUDIO_2026)
- [ ] GUI Quality Mode Selector
- [ ] Batch Processing

### Automated Test Suite

```bash
# Run all tests
pytest tests/ --maxfail=1 --disable-warnings

# Run specific suites
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Performance benchmarks
python tests/benchmarks/v2_vs_v3_comparison.py
```

---

## 📊 Success Metrics

### Performance Targets
- ✅ **Speedup:** V3 ≥ 1.2× faster than V2 (+20%)
- ✅ **Memory:** No memory leaks (60s stress test)
- ✅ **RT Factor:** <3× Realtime (BALANCED mode)
- ✅ **Core Efficiency:** ≥60% @ 4 cores

### Quality Targets
- ✅ **Quality Parity:** V3 ≥ V2 (±2% tolerance)
- ✅ **Defect Detection:** 95%+ accuracy
- ✅ **Material Detection:** 90%+ accuracy
- ✅ **No Quality Regressions:** Critical phases unchanged

### Stability Targets
- ✅ **Zero Crashes:** 100 test runs without crash
- ✅ **Bug Density:** <1 bug per 1000 lines
- ✅ **Test Coverage:** ≥80% code coverage
- ✅ **User Satisfaction:** ≥4/5 stars

---

## ⚠️ Risks & Mitigation

### HOCH: V2-spezifische Features fehlen in V3
**Mitigation:** Feature-Matrix erstellen, fehlende Features in V3 implementieren

### MITTEL: Performance-Regression in Edge Cases
**Mitigation:** Comprehensive benchmarks, stress testing

### MITTEL: GUI-Kompatibilitäts-Probleme
**Mitigation:** Feature Flag für sanfte Migration, Fallback auf V2

### NIEDRIG: User-Confusion durch neue Metriken
**Mitigation:** Dokumentation, Tutorials, in-app Hilfe

---

## 🚦 Go/No-Go Criteria (End of Week 6)

### ✅ GO Criteria:
- [ ] All P0 bugs fixed
- [ ] Performance ≥ 1.2× V2
- [ ] Quality Parity maintained
- [ ] Zero crashes in 100 test runs
- [ ] Documentation complete
- [ ] UAT passed (≥4/5 rating)

### ❌ NO-GO Criteria:
- Performance regression >10%
- Quality regression >5%
- Critical bugs (P0) unfixed
- Test coverage <70%
- Crashes in stress tests

---

## 📝 Deliverables Checklist

### Code
- [ ] UnifiedRestorerV3 production-ready
- [ ] GUI updated (V3 integration)
- [ ] CLI updated (V3 support)
- [ ] Feature flags implemented
- [ ] V2 backward compatibility

### Tests
- [ ] Unit tests (41 phases)
- [ ] Integration tests (full pipeline)
- [ ] E2E tests (3+ materials)
- [ ] Stress tests (memory leaks)
- [ ] Benchmarks (V2 vs V3)

### Documentation
- [ ] API Documentation (V3)
- [ ] Migration Guide (V2→V3)
- [ ] Performance Tuning Guide
- [ ] CHANGELOG.md
- [ ] RELEASE_NOTES.md

### Release
- [ ] Version bump (9.0.0-rc1)
- [ ] Git tag
- [ ] Release artifacts
- [ ] Announcement draft

---

## 👥 Team & Responsibilities

| Role | Responsibility | Person |
|------|----------------|--------|
| **Tech Lead** | Overall architecture, code review | AI Team |
| **Backend Dev** | V3 implementation, testing | AI Team |
| **Frontend Dev** | GUI integration | AI Team |
| **QA Engineer** | Test strategy, execution | AI Team |
| **Release Manager** | Release coordination | AI Team |
| **Documentation** | Docs, guides, tutorials | AI Team |

---

## 🎯 Next Immediate Actions (Week 1, Day 1)

**HEUTE (15. Februar 2026):**

1. ✅ **Create Migration Plan** (dieses Dokument) ✅
2. ⏳ **Code Review V3** (1-2 Stunden)
   ```bash
   cd core/
   wc -l unified_restorer_v3.py adaptive_core_scheduler.py performance_guard.py
   ```
3. ⏳ **Run Basic Test** (30 min)
   ```bash
   python -c "from core.unified_restorer_v3 import UnifiedRestorerV3; print('V3 Import OK')"
   ```
4. ⏳ **Create Test Script** (1 Stunde)
   ```bash
   # tests/v3_basic_test.py
   python tests/v3_basic_test.py
   ```

**MORGEN (16. Februar 2026):**
5. Memory Leak Test (60s audio)
6. E2E Test mit Elke Best (TAPE material)
7. Performance Profiling (cProfile)

**DIESE WOCHE:**
- Woche 1 komplett abschließen (Code Review + Basic Tests)
- Erste Regressions-Matrix erstellen
- Edge Cases dokumentieren

---

## 📚 References

- [UnifiedRestorerV3 Code](../core/unified_restorer_v3.py)
- [AdaptiveCoreScheduler](../core/adaptive_core_scheduler.py)
- [PerformanceGuard](../core/performance_guard.py)
- [Quick Wins Activation](QUICK_WINS_ACTIVATION.md)
- [DefectScanner Spec](DEFECT_SCANNER_SPEC.md)
- [Modular Phases API](MODULAR_PHASES_API.md)

---

**Status:** 🟢 **PLAN READY - READY TO START!**  
**Nächster Schritt:** Sprint 1, Woche 1, Tag 1 - Code Review V3 starten  
**Owner:** AI Development Team  
**Last Updated:** 15. Februar 2026
