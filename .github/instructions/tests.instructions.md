---
applyTo: "tests/**/*.py"
---

# Test-Regeln (normativ, Aurik 9.12.x)

## GC-Konventionen

```python
# VERBOTEN: volles gc.collect() nach jedem Test in großen Suiten
# → zu hoher Overhead bei 11k+ Tests

# RICHTIG: leichter inkrementeller GC
import gc
gc.collect(0)  # nur Generation 0

# Vollständiges gc.collect() nur:
# - an Datei-/Session-Grenzen
# - cadence-gesteuert (z.B. alle 100 Tests)
```

## Langlebige Hintergrund-Manager

```python
# Jeder Monitor-Thread / Background-Manager braucht:
class MyManager:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self):
        """Idempotent — mehrfacher Aufruf = kein Fehler."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

# Cleanup in pytest:
# pytest_sessionfinish oder Finalizer — NICHT daemon=True als einziges Modell
```

## Budget-Tests — Mock is_system_thrashing

```python
# PFLICHT: Budget-Tests MÜSSEN is_system_thrashing mocken
# Sonst: flaky auf Hosts mit hoher Swap-Auslastung

@pytest.fixture(autouse=True)
def mock_no_thrashing(monkeypatch):
    monkeypatch.setattr(
        "backend.core.plugin_lifecycle_manager.is_system_thrashing",
        lambda: False,
    )

# Tests die try_allocate / release prüfen MÜSSEN diese Fixture verwenden
```

## Resampling-Bibliotheken — Warnings

```python
# resampy / librosa können pkg_resources-Warnings unter -W error::Warning auslösen
# IMMER aktuelle Version: resampy >= 0.4.3
# conftest.py global:
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
```

## Teure Transforms — Reihenfolge

```python
# KANONISCH (Kostenpyramide):
# 1. Frame-Energie-Check (günstig) → Gate
# 2. Voiced-Frame-Gate (günstig) → Gate
# 3. dann: filtfilt + Hilbert + STFT (teuer)

# VERBOTEN: Hilbert/STFT vor günstigem Gate
# Beispiel TFS-Guard:
frame_energy = np.sum(frame ** 2)
if frame_energy < _MIN_ENERGY_THRESHOLD:
    continue  # kein Hilbert
if not is_voiced(frame, sr):
    continue  # kein filtfilt
tfs = compute_tfs_hilbert(frame, sr)  # erst jetzt
```

## Guarded Correlation (NaN-safe)

```python
# VERBOTEN: np.corrcoef auf near-constant Signalen → Warning/NaN

# RICHTIG:
def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b + eps))
```

## Sentinel-Pattern für optional-heavy Imports

```python
# RICHTIG: optional imports für ML-Tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_heavy_model():
    ...
```

## AMRB-Update bei Major-Release (9.x.0)

```python
# PFLICHT: benchmarks/update_amrb_history.py ausführen
# benchmarks/amrb_history.json updaten
# OQS-Delta < -2.0 gegenüber vorheriger Baseline = Release-Blocker
```

## Phase-Test-Muster — Pre/Post-Delta

```python
def test_phase_XX_no_regression(synthetic_audio, sr=48000):
    """Stellt sicher dass Phase XX keine Goal-Regression einführt."""
    from backend.core.phases.phase_XX import PhaseXX
    phase = PhaseXX()
    audio_out = phase.process(synthetic_audio, sr, material_type="vinyl", strength=0.8)

    # Längen-Invariante §2.61 (shape[-1] statt len() — len() auf 2D-Stereo gibt 2, nicht N!):
    assert abs(audio_out.shape[-1] - synthetic_audio.shape[-1]) <= 64

    # Clip-Invariante:
    assert np.max(np.abs(audio_out)) <= 1.0

    # NaN-Invariante:
    assert not np.any(np.isnan(audio_out))
```

## PMGG-CIG-Sync-Test (§2.55)

```python
# test_pmgg_cig_sync.py — MUSS nach jeder neuen Phase aktualisiert werden:
# CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2
# ↔ PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2
# bidirektional synchron — CI-Gate
```
