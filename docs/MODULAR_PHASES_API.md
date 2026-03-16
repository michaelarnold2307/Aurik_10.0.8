# Modular Phases API - Aurik 9.0

**Version:** 1.0.0  
**Status:** ✅ Production-Ready  
**Location:** `core/phases/` (41 phases planned, 3 implemented)

---

## 1. Purpose

The **Modular Phases Architecture** provides a standardized interface for all 41 restoration phases in Aurik 9.0. Benefits:
- **Uniform API:** All phases implement same interface (process, validate, estimate_time)
- **Material-Adaptive:** Each phase adjusts behavior based on material type (shellac/vinyl/tape/CD)
- **Dependency Management:** Phases declare dependencies for correct execution order
- **Performance Predictable:** Each phase estimates its own execution time
- **Testable:** Standardized interface enables unit testing every phase independently

---

## 2. Phase Interface

### 2.1 Abstract Base Class

```python
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

class PhaseCategory(Enum):
    """Phase categories for organization."""
    DEFECT_REMOVAL = "defect_removal"      # clicks, crackle, hum
    NOISE_REDUCTION = "noise_reduction"     # denoise, dehiss
    FREQUENCY_RESTORE = "frequency_restore" # EQ, bandwidth extension
    DYNAMIC_RESTORE = "dynamic_restore"     # compression, transients
    STEREO_RESTORE = "stereo_restore"       # balance, phase correction
    FINAL_POLISH = "final_polish"           # limiter, dithering

@dataclass
class PhaseMetadata:
    """Metadata for a restoration phase."""
    phase_id: str                           # e.g., "phase_01_click_removal"
    name: str                               # Human-readable name
    category: PhaseCategory
    priority: int                           # 1 (low) to 9 (critical)
    dependencies: List[str]                 # Phase IDs that must run before this
    estimated_time_factor: float            # Factor of audio duration (e.g., 0.02 = 2%)

@dataclass
class PhaseResult:
    """Result from a phase execution."""
    success: bool
    processed_audio: np.ndarray
    metrics: Dict[str, float]               # Phase-specific metrics
    processing_time: float                  # Actual time taken (seconds)
    metadata: Dict[str, Any]                # Additional info

class PhaseInterface(ABC):
    """
    Abstract base class for all restoration phases.
    
    All 41 phases must implement this interface.
    """
    
    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        """
        Process audio with material-adaptive parameters.
        
        Args:
            audio: Audio samples (mono: [samples], stereo: [samples, 2])
            sample_rate: Sample rate in Hz
            material: Material type for adaptive thresholds
        
        Returns:
            PhaseResult with processed audio and metrics
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> PhaseMetadata:
        """
        Get phase metadata (ID, dependencies, priority, etc.).
        
        Returns:
            PhaseMetadata describing this phase
        """
        pass
    
    def validate_input(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Validate input audio before processing.
        
        Args:
            audio: Audio samples to validate
            sample_rate: Sample rate to validate
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if audio.size == 0:
            raise ValueError("Empty audio")
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        return True
    
    def estimate_time(self, audio_duration: float) -> float:
        """
        Estimate processing time for given audio duration.
        
        Args:
            audio_duration: Audio duration in seconds
        
        Returns:
            Estimated processing time in seconds
        """
        metadata = self.get_metadata()
        return audio_duration * metadata.estimated_time_factor
```

---

## 3. Material-Adaptive Pattern

### 3.1 Threshold Configuration

Every phase maintains **material-specific thresholds:**

```python
class ClickRemovalPhase(PhaseInterface):
    # Material-adaptive thresholds
    THRESHOLDS = {
        MaterialType.SHELLAC: 0.05,   # Very sensitive (78 RPM has many clicks)
        MaterialType.VINYL: 0.10,     # Moderate (33/45 RPM has fewer clicks)
        MaterialType.TAPE: 0.20,      # Less sensitive (tape rarely has clicks)
        MaterialType.CD: 0.30,        # Least sensitive (digital rarely has clicks)
        MaterialType.STREAMING: 0.40, # Only detect very obvious clicks
    }
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        # Get material-specific threshold
        threshold = self.THRESHOLDS.get(material, 0.10)  # Default: vinyl
        
        # Use threshold in algorithm
        clicks_detected = self._detect_clicks(audio, sample_rate, threshold)
        restored = self._repair_clicks(audio, clicks_detected)
        
        return PhaseResult(
            success=True,
            processed_audio=restored,
            metrics={"clicks_removed": len(clicks_detected)},
            processing_time=time.time() - start,
            metadata={"material": material.value, "threshold": threshold}
        )
```

### 3.2 Algorithm Variation by Material

Some phases use **different algorithms** per material:

```python
class FrequencyRestorationPhase(PhaseInterface):
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        if material == MaterialType.SHELLAC:
            # Shellac has severe high-frequency rolloff (>5kHz)
            restored = self._restore_highs_aggressive(audio, cutoff=5000)
        
        elif material == MaterialType.VINYL:
            # Vinyl has moderate rolloff (>12kHz)
            restored = self._restore_highs_moderate(audio, cutoff=12000)
        
        elif material == MaterialType.TAPE:
            # Tape has good high-freq response, minimal restoration
            restored = self._restore_highs_gentle(audio, cutoff=15000)
        
        else:  # CD, Streaming
            # Digital sources don't need frequency restoration
            return PhaseResult(success=True, processed_audio=audio, ...)
        
        return PhaseResult(success=True, processed_audio=restored, ...)
```

---

## 4. Dependency Management

### 4.1 Declaring Dependencies

Phases declare **which phases must run before them:**

```python
class DenoisePhase(PhaseInterface):
    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_03_denoise",
            name="Broadband Noise Reduction",
            category=PhaseCategory.NOISE_REDUCTION,
            priority=6,  # MEDIUM priority
            dependencies=[
                "phase_01_click_removal",  # Remove clicks before denoise
                "phase_02_hum_removal"     # Remove hum before denoise
            ],
            estimated_time_factor=0.05  # 5% of audio duration
        )
```

### 4.2 Dependency Graph Example

```
Phase Execution Order (Dependency Graph):
┌──────────────────────────────────────────┐
│ phase_01_click_removal (no dependencies) │
└─────────────┬────────────────────────────┘
              │
              v
┌──────────────────────────────────────────┐
│ phase_02_hum_removal (depends: 01)       │
└─────────────┬────────────────────────────┘
              │
              v
┌──────────────────────────────────────────┐
│ phase_03_denoise (depends: 01, 02)       │
└─────────────┬────────────────────────────┘
              │
              v
┌──────────────────────────────────────────┐
│ phase_15_stereo_enhancement (depends: 03)│
└──────────────────────────────────────────┘
```

### 4.3 Circular Dependency Detection

The scheduler detects circular dependencies at initialization:

```python
# BAD: Circular dependency (will raise error)
class PhaseA(PhaseInterface):
    def get_metadata(self):
        return PhaseMetadata(..., dependencies=["phase_B"])

class PhaseB(PhaseInterface):
    def get_metadata(self):
        return PhaseMetadata(..., dependencies=["phase_A"])  # ERROR!

# AdaptiveCoreScheduler will detect this and raise:
# DependencyCycleError: Circular dependency detected: phase_A → phase_B → phase_A
```

---

## 5. Priority System

### 5.1 Priority Levels

```python
# Priority scale (used by PerformanceGuard for adaptive skipping)
PRIORITY_CRITICAL = 9    # Never skip (e.g., click removal on shellac)
PRIORITY_HIGH = 8        # Skip only as last resort (e.g., hum removal)
PRIORITY_MEDIUM = 6      # Skip if approaching RT limit (e.g., denoise)
PRIORITY_LOW = 3         # Skip first if time constrained (e.g., stereo enhancement)
```

### 5.2 Material-Adaptive Priorities

Priorities can **change based on material:**

```python
class ClickRemovalPhase(PhaseInterface):
    def get_metadata(self) -> PhaseMetadata:
        # Base priority: HIGH
        priority = 8
        
        # Adjust based on material (if known during init)
        if hasattr(self, 'material'):
            if self.material == MaterialType.SHELLAC:
                priority = 9  # CRITICAL for shellac (many clicks)
            elif self.material == MaterialType.CD:
                priority = 3  # LOW for CD (rare clicks)
        
        return PhaseMetadata(..., priority=priority, ...)
```

---

## 6. Performance Estimation

### 6.1 Time Factor

Each phase estimates its processing time as a **factor of audio duration:**

```python
# Examples:
estimated_time_factor = 0.02  # 2% of audio duration
    # → 300s audio = 6s processing time

estimated_time_factor = 0.10  # 10% of audio duration
    # → 300s audio = 30s processing time

estimated_time_factor = 0.50  # 50% of audio duration (expensive!)
    # → 300s audio = 150s processing time
```

### 6.2 Typical Time Factors

| Phase Category          | Typical Factor | Example Phases                   |
|-------------------------|----------------|----------------------------------|
| Simple filters          | 0.01 - 0.03    | click removal, hum removal       |
| FFT-based processing    | 0.05 - 0.15    | denoise, frequency restoration   |
| Time-domain analysis    | 0.10 - 0.30    | wow/flutter correction           |
| Machine learning models | 0.50 - 2.00    | neural denoise, stem separation  |

### 6.3 Actual vs. Estimated Time

After execution, compare actual vs. estimated:

```python
result = phase.process(audio, sample_rate, material)

estimated = phase.estimate_time(audio_duration=225.0)
actual = result.processing_time

if actual > estimated * 1.5:
    logger.warning(f"Phase {phase_id} took 50% longer than estimated!")
    # Adjust time factors for future runs
```

---

## 7. Example Implementations

### 7.1 Click Removal Phase

```python
class ClickRemovalPhase(PhaseInterface):
    """Remove sharp transient clicks from audio."""
    
    THRESHOLDS = {
        MaterialType.SHELLAC: 0.05,
        MaterialType.VINYL: 0.10,
        MaterialType.TAPE: 0.20,
        MaterialType.CD: 0.30,
    }
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        self.validate_input(audio, sample_rate)
        start_time = time.time()
        
        # Get material threshold
        threshold = self.THRESHOLDS.get(material, 0.10)
        
        # Detect clicks (inter-sample difference > threshold)
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
        diffs = np.abs(np.diff(mono))
        click_indices = np.where(diffs > threshold)[0]
        
        # Repair clicks (median filter interpolation)
        restored = audio.copy()
        for idx in click_indices:
            if 2 <= idx < len(mono) - 2:
                neighbors = [mono[idx-2], mono[idx-1], mono[idx+1], mono[idx+2]]
                restored[idx] = np.median(neighbors)
        
        return PhaseResult(
            success=True,
            processed_audio=restored,
            metrics={
                "clicks_detected": len(click_indices),
                "threshold_used": threshold,
                "material": material.value
            },
            processing_time=time.time() - start_time,
            metadata={"algorithm": "inter_sample_diff"}
        )
    
    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_01_click_removal",
            name="Click Removal",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH (critical for vinyl/shellac)
            dependencies=[],  # No dependencies (first phase)
            estimated_time_factor=0.02  # 2% of audio duration
        )
```

### 7.2 Hum Removal Phase

```python
class HumRemovalPhase(PhaseInterface):
    """Remove 50/60 Hz power line hum and harmonics."""
    
    Q_FACTORS = {
        MaterialType.TAPE: 30,     # Aggressive (tape hum very tonal)
        MaterialType.VINYL: 20,    # Moderate
        MaterialType.SHELLAC: 15,  # Gentle (avoid musical damage)
        MaterialType.CD: 10,       # Very gentle (rare on digital)
    }
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        self.validate_input(audio, sample_rate)
        start_time = time.time()
        
        # Auto-detect hum frequency (50 or 60 Hz)
        hum_freq = self._detect_hum_frequency(audio, sample_rate)
        if hum_freq is None:
            # No hum detected
            return PhaseResult(success=True, processed_audio=audio, ...)
        
        # Get material-specific Q factor
        q = self.Q_FACTORS.get(material, 20)
        
        # Design cascaded notch filters for harmonics
        harmonics = [hum_freq * i for i in range(1, 7)]  # 6 harmonics
        restored = audio.copy()
        for harmonic in harmonics:
            restored = self._apply_notch_filter(restored, sample_rate, harmonic, q)
        
        # Measure reduction
        reduction_db = self._measure_hum_reduction(audio, restored, hum_freq)
        
        return PhaseResult(
            success=True,
            processed_audio=restored,
            metrics={
                "hum_frequency": hum_freq,
                "harmonics_removed": len(harmonics),
                "reduction_db": reduction_db
            },
            processing_time=time.time() - start_time,
            metadata={"q_factor": q, "material": material.value}
        )
    
    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_02_hum_removal",
            name="Hum Removal (50/60 Hz)",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,  # HIGH
            dependencies=["phase_01_click_removal"],  # Run after clicks
            estimated_time_factor=0.03  # 3% of audio duration
        )
```

### 7.3 Denoise Phase

```python
class DenoisePhase(PhaseInterface):
    """Reduce broadband noise using Wiener filtering."""
    
    STRENGTH = {
        MaterialType.TAPE: 0.8,      # Aggressive (tape hiss very audible)
        MaterialType.VINYL: 0.6,     # Moderate (surface noise)
        MaterialType.SHELLAC: 0.5,   # Moderate (surface noise + recording noise)
        MaterialType.CD: 0.3,        # Gentle (minimal noise on digital)
    }
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        self.validate_input(audio, sample_rate)
        start_time = time.time()
        
        # Get material-specific strength
        strength = self.STRENGTH.get(material, 0.6)
        
        # STFT-based Wiener filtering
        from scipy.signal import stft, istft
        
        f, t, Zxx = stft(audio, fs=sample_rate, nperseg=2048)
        
        # Estimate noise floor (first 10% of audio)
        noise_frames = int(0.1 * Zxx.shape[1])
        noise_floor = np.median(np.abs(Zxx[:, :noise_frames]), axis=1)
        
        # Wiener filter
        for i in range(Zxx.shape[1]):
            magnitude = np.abs(Zxx[:, i])
            wiener_gain = np.maximum(0, 1 - strength * (noise_floor / (magnitude + 1e-8)))
            Zxx[:, i] *= wiener_gain
        
        # Reconstruct audio
        _, restored = istft(Zxx, fs=sample_rate)
        
        # Measure noise reduction (>8kHz band)
        reduction_db = self._measure_high_freq_reduction(audio, restored, sample_rate)
        
        return PhaseResult(
            success=True,
            processed_audio=restored[:len(audio)],  # Trim to original length
            metrics={
                "strength": strength,
                "reduction_db": reduction_db
            },
            processing_time=time.time() - start_time,
            metadata={"algorithm": "wiener_stft", "material": material.value}
        )
    
    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_03_denoise",
            name="Broadband Noise Reduction",
            category=PhaseCategory.NOISE_REDUCTION,
            priority=6,  # MEDIUM
            dependencies=["phase_01_click_removal", "phase_02_hum_removal"],
            estimated_time_factor=0.05  # 5% of audio duration
        )
```

---

## 8. Creating New Phases

### 8.1 Template

```python
from core.phases.phase_interface import PhaseInterface, PhaseMetadata, PhaseResult, PhaseCategory
from core.defect_scanner import MaterialType
import numpy as np
import time

class MyNewPhase(PhaseInterface):
    """Brief description of what this phase does."""
    
    # Material-adaptive parameters
    PARAM_A = {
        MaterialType.SHELLAC: value_shellac,
        MaterialType.VINYL: value_vinyl,
        MaterialType.TAPE: value_tape,
        MaterialType.CD: value_cd,
    }
    
    def process(self, audio: np.ndarray, sample_rate: int, 
                material: MaterialType) -> PhaseResult:
        """Process audio with material-adaptive algorithm."""
        self.validate_input(audio, sample_rate)
        start_time = time.time()
        
        # 1. Get material-specific parameters
        param = self.PARAM_A.get(material, default_value)
        
        # 2. Process audio
        restored = self._my_algorithm(audio, sample_rate, param)
        
        # 3. Calculate metrics
        metric_value = self._calculate_metric(audio, restored)
        
        # 4. Return result
        return PhaseResult(
            success=True,
            processed_audio=restored,
            metrics={"my_metric": metric_value},
            processing_time=time.time() - start_time,
            metadata={"param_used": param, "material": material.value}
        )
    
    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_XX_my_new_phase",
            name="My New Phase",
            category=PhaseCategory.DEFECT_REMOVAL,  # Choose appropriate category
            priority=6,  # Choose 1-9 based on importance
            dependencies=["phase_YY_dependency"],  # List dependencies
            estimated_time_factor=0.05  # Estimate as fraction of audio duration
        )
    
    def _my_algorithm(self, audio: np.ndarray, sample_rate: int, 
                      param: float) -> np.ndarray:
        """Implement your restoration algorithm here."""
        # Your code here
        return audio  # Replace with actual processing
```

### 8.2 Testing New Phases

```python
def test_my_new_phase():
    """Unit test for MyNewPhase."""
    # Generate test audio
    audio = generate_sine_wave(duration=10, frequency=440, sr=44100)
    
    # Add synthetic defect
    audio_with_defect = inject_my_defect(audio)
    
    # Run phase
    phase = MyNewPhase()
    result = phase.process(audio_with_defect, 44100, MaterialType.VINYL)
    
    # Validate result
    assert result.success
    assert result.processing_time < 1.0  # Should be fast for 10s audio
    assert result.metrics["my_metric"] > expected_threshold
    
    # Validate metadata
    metadata = phase.get_metadata()
    assert metadata.phase_id == "phase_XX_my_new_phase"
    assert metadata.priority >= 1 and metadata.priority <= 9
```

---

## 9. Current Phase Status (3/41 Implemented)

### 9.1 Implemented Phases ✅

| Phase ID               | Category         | Priority | Status |
|------------------------|------------------|----------|--------|
| phase_01_click_removal | DEFECT_REMOVAL   | 8 (HIGH) | ✅ Tested |
| phase_02_hum_removal   | DEFECT_REMOVAL   | 8 (HIGH) | ✅ Tested |
| phase_03_denoise       | NOISE_REDUCTION  | 6 (MED)  | ✅ Ready  |

### 9.2 Remaining Phases (38 to implement)

**Defect Removal:**
- phase_09_crackle_removal
- phase_05_rumble_filter
- phase_12_wow_flutter_fix
- phase_24_dropout_repair
- phase_22_artifact_removal

**Noise Reduction:**
- phase_11_dehiss
- phase_18_spectral_subtraction

**Frequency Restoration:**
- phase_06_frequency_restoration
- phase_07_bandwidth_extension
- phase_19_eq_correction

**Dynamic Restoration:**
- phase_08_transient_preservation
- phase_13_compression_restore
- phase_20_dynamic_range_expansion

**Stereo Restoration:**
- phase_14_phase_correction
- phase_15_stereo_balance
- phase_16_stereo_enhancement

**Final Polish:**
- phase_40_final_limiter
- phase_41_dithering

(Full list: see PROGRAMMABLAUF_ANALYSE.md)

---

## 10. Integration with UnifiedRestorerV3

### 10.1 Phase Discovery

```python
# UnifiedRestorerV3 auto-discovers all phases in core/phases/
import os
import importlib

phase_files = [f for f in os.listdir("core/phases") if f.startswith("phase_") and f.endswith(".py")]

for phase_file in phase_files:
    module = importlib.import_module(f"core.phases.{phase_file[:-3]}")
    # Find PhaseInterface subclasses
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, PhaseInterface):
            phases.append(obj())
```

### 10.2 Dependency Sorting

```python
def sort_phases_by_dependencies(phases: List[PhaseInterface]) -> List[PhaseInterface]:
    """Topological sort of phases based on dependencies."""
    # Build dependency graph
    graph = {p.get_metadata().phase_id: p for p in phases}
    deps = {p.get_metadata().phase_id: p.get_metadata().dependencies for p in phases}
    
    # Topological sort (Kahn's algorithm)
    sorted_phases = []
    while graph:
        # Find phases with no dependencies
        no_deps = [pid for pid, d in deps.items() if not d]
        if not no_deps:
            raise DependencyCycleError("Circular dependency detected!")
        
        # Add to sorted list
        for pid in no_deps:
            sorted_phases.append(graph.pop(pid))
            deps.pop(pid)
        
        # Remove from other dependencies
        for pid in deps:
            deps[pid] = [d for d in deps[pid] if d not in no_deps]
    
    return sorted_phases
```

---

## 11. Change Log

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0.0   | 2026-02-15 | Production release (3 phases implemented)    |
| 0.9.0   | 2026-02-14 | PhaseInterface finalized                     |

---

## 12. References

- **Interface:** [core/phases/phase_interface.py](../core/phases/phase_interface.py)
- **Examples:** [core/phases/phase_01_click_removal.py](../core/phases/phase_01_click_removal.py)
- **Integration:** [UNIFIED_RESTORER_V3_SPEC.md](UNIFIED_RESTORER_V3_SPEC.md)
- **Full List:** [PROGRAMMABLAUF_ANALYSE.md](../PROGRAMMABLAUF_ANALYSE.md)

---

**Status:** ✅ Production-Ready | **Phases:** 3/41 | **Template:** Available for rapid development
