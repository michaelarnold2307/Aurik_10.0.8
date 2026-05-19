# 🎚️ ~~AURIK DAW Plugin~~ - Architecture & Implementation Plan

> **⚠️ DEPRECATED - NOT BEING PURSUED**  
> **Date:** 9. Februar 2026  
> **Reason:** Licensing independence preferred over third-party dependencies  
> **Alternative:** Professional Desktop Application (Phase 2.4) - fully functional  
>
> This document remains for reference purposes only.

---

**Created:** 9. Februar 2026  
**Target:** ~~VST3 + AU (Audio Units) Plugin~~  
**Framework:** ~~JUCE (Industry Standard)~~  
**Status:** ❌ **Cancelled** - Licensing risk (Steinberg VST3) deemed too high

---

## 📋 Executive Summary

~~Ziel ist es, AURIK's fortschrittliche Audio Restoration Technologie als professionelles DAW-Plugin verfügbar zu machen. Nutzer können dann direkt in Logic Pro, Ableton Live, Pro Tools, etc. arbeiten.~~

**Decision Update (9. Feb 2026):**  
After analysis of VST3 licensing terms (Steinberg) and dependency risks, the decision was made to **NOT pursue plugin development**. Instead, AURIK remains fully independent as a **standalone desktop application** with comprehensive batch processing capabilities.

### ~~Key Benefits~~ Cancelled Approach

- ~~✅ Professional Workflow Integration~~
- ~~✅ Real-time Processing (low latency)~~
- ~~✅ Preset Management & Recall~~
- ~~✅ DAW Automation Support~~
- ~~✅ Cross-Platform (macOS, Windows, Linux)~~

### Actual Implementation: Desktop App ✅

- ✅ **Standalone Application** (no external dependencies)
- ✅ **Batch Processing** (studio workflow support)
- ✅ **Complete Independence** (no licensing risks)
- ✅ **Full Control** (no third-party restrictions)
- ✅ See: [Desktop_App_Final_Status.md](Desktop_App_Final_Status.md)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DAW (Logic, Ableton, etc.)               │
└─────────────────────┬───────────────────────────────────────┘
                      │ VST3/AU Protocol
┌─────────────────────▼───────────────────────────────────────┐
│              AURIK Plugin (C++ / JUCE)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  UI Layer (JUCE Components)                          │   │
│  │  - Parameter Controls                                │   │
│  │  - Real-time Visualization                           │   │
│  │  - Preset Browser                                    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Plugin Processor (C++)                              │   │
│  │  - Audio Buffer Management                           │   │
│  │  - Parameter Smoothing                               │   │
│  │  - State Management                                  │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │  Python Bridge (pybind11 / PyBind)                   │   │
│  │  - Embedded Python Interpreter                       │   │
│  │  - NumPy Array Conversion                            │   │
│  │  - Thread Safety                                     │   │
│  └──────────────────┬───────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          AURIK Core (Python)                                │
│  - UnifiedRestorerV2                                        │
│  - Phase 2.3 Instrumental Enhancement                       │
│  - Musical Goals Metrics                                    │
│  - All existing DSP modules                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Implementation Strategy

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic plugin that loads and processes audio

#### 1.1 JUCE Setup

- [ ] Install JUCE Framework (v7.0+)
- [ ] Create new Audio Plugin project
- [ ] Configure VST3 + AU targets
- [ ] Test basic "hello world" plugin in DAW

#### 1.2 Python Integration

- [ ] Embed Python interpreter (libpython)
- [ ] Add pybind11 for C++ ↔ Python bridge
- [ ] Test NumPy array sharing (zero-copy if possible)
- [ ] Handle Python GIL for thread safety

#### 1.3 Basic Audio Processing

- [ ] Implement `processBlock()` method
- [ ] Convert JUCE AudioBuffer → NumPy array
- [ ] Call AURIK's UnifiedRestorerV2
- [ ] Convert result back to AudioBuffer

**Deliverable:** Working plugin that processes audio through AURIK

---

### Phase 2: Parameters & UI (Week 2-3)

**Goal:** Full parameter control and professional UI

#### 2.1 Parameter System

- [ ] Define plugin parameters (20-30 key controls)
  - Medium Type (Vinyl, Cassette, DAT, CD, MP3, etc.)
  - Processing Mode (Gentle, Balanced, Aggressive, etc.)
  - Musical Goals (Brillanz, Wärme, etc.)
  - Phase 2.3 Controls (Bass, Drums, Guitar, etc.)
- [ ] Map to AURIK's internal settings
- [ ] Implement parameter smoothing (avoid clicks)
- [ ] Support DAW automation

#### 2.2 User Interface

- [ ] Design clean, modern UI (inspired by iZotope RX)
- [ ] Real-time waveform display
- [ ] Musical Goals radar chart
- [ ] Defect detection visualization
- [ ] Preset browser
- [ ] Undo/Redo buttons

**Tools:**
- JUCE GUI Components
- OpenGL for real-time visualization
- Custom LookAndFeel for branding

**Deliverable:** Professional-looking plugin with full control

---

### Phase 3: Optimization & Real-time (Week 3-4)

**Goal:** Low-latency, CPU-efficient processing

#### 3.1 Performance Optimization

- [ ] Implement ring buffer for chunk processing
- [ ] Multi-threading (separate audio thread)
- [ ] Minimize Python GIL contention
- [ ] Cache Python objects between calls
- [ ] Profile with Instruments / VTune

#### 3.2 ONNX Integration

- [ ] Replace Python ML models with ONNX Runtime
- [ ] C++ inference (no Python overhead)
- [ ] GPU acceleration if available

#### 3.3 Latency Reduction

- [ ] Target: < 10ms latency
- [ ] Look-ahead buffer management
- [ ] Report latency to DAW correctly

**Deliverable:** Real-time capable plugin

---

### Phase 4: Presets & Polish (Week 4-5)

**Goal:** Production-ready plugin

#### 4.1 Preset Management

- [ ] Factory presets (10-20 common scenarios)
  - "Vinyl Warmth"
  - "Cassette Rescue"
  - "Digital Cleanup"
  - "Mastering Polish"
- [ ] User preset save/load
- [ ] Preset browser UI
- [ ] Import/Export presets

#### 4.2 Testing & QA

- [ ] Test in Logic Pro X
- [ ] Test in Ableton Live
- [ ] Test in Pro Tools
- [ ] Test in FL Studio
- [ ] Test in Reaper
- [ ] Validate VST3 compliance
- [ ] Validate AU compliance (auval)

#### 4.3 Documentation

- [ ] User manual
- [ ] Parameter reference
- [ ] Tutorial videos
- [ ] Installation guide

**Deliverable:** Shippable plugin

---

## 🔧 Technical Requirements

### Development Environment

#### macOS (Primary)

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install JUCE (via Projucer)
# Download from: https://juce.com/get-juce/download

# Install Python development headers
brew install python@3.10

# Install pybind11
pip install pybind11
```

#### Windows (Secondary)

```powershell
# Install Visual Studio 2022 (Community)
# Install JUCE Projucer
# Install Python 3.10+ (with dev headers)
# Install pybind11
```

#### Linux (Tertiary)

```bash
# Install build essentials
sudo apt-get install build-essential libfreetype6-dev libx11-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libasound2-dev

# Install JUCE dependencies
# Install Python dev headers
sudo apt-get install python3-dev

# Install pybind11
pip install pybind11
```

---

### Plugin Formats

| Format | Platform | Priority | Status |
|--------|----------|----------|--------|
| VST3   | All      | P0       | 🔄 Planning |
| AU     | macOS    | P0       | 🔄 Planning |
| AAX    | All      | P1       | ⏳ Later |
| LV2    | Linux    | P2       | ⏳ Later |

---

## 📊 Parameter Mapping

### Core Parameters (20)

| # | Parameter | Type | Range | Default | Maps To |
| --- | --- | --- | --- | --- | --- |
| 1 | Medium Type | Choice | 10 options | Vinyl | `medium_type` |
| 2 | Processing Mode | Choice | 5 options | Balanced | `processing_mode` |
| 3 | Brillanz Target | Float | 0.0-1.0 | 0.87 | `musical_goals.brillanz` |
| 4 | Wärme Target | Float | 0.0-1.0 | 0.82 | `musical_goals.waerme` |
| 5 | Natürlichkeit | Float | 0.0-1.0 | 0.85 | `musical_goals.natuerlichkeit` |
| 6 | Authentizität | Float | 0.0-1.0 | 0.88 | `musical_goals.authentizitaet` |
| 7 | Emotionalität | Float | 0.0-1.0 | 0.83 | `musical_goals.emotionalitaet` |
| 8 | Transparenz | Float | 0.0-1.0 | 0.89 | `musical_goals.transparenz` |
| 9 | Bass-Kraft | Float | 0.0-1.0 | 0.75 | `musical_goals.bass_kraft` |
| 10 | Bass Enhancement | On/Off | Boolean | Off | `phase_2_3.bass_enable` |
| 11 | Drums Enhancement | On/Off | Boolean | Off | `phase_2_3.drums_enable` |
| 12 | Guitar Enhancement | On/Off | Boolean | Off | `phase_2_3.guitar_enable` |
| 13 | Piano Enhancement | On/Off | Boolean | Off | `phase_2_3.piano_enable` |
| 14 | Brass Enhancement | On/Off | Boolean | Off | `phase_2_3.brass_enable` |
| 15 | Spatial Enhancement | On/Off | Boolean | Off | `phase_2_3.spatial_enable` |
| 16 | Noise Reduction | Float | 0.0-1.0 | 0.5 | `noise_reduction_amount` |
| 17 | Click Removal | Float | 0.0-1.0 | 0.5 | `click_removal_amount` |
| 18 | Dry/Wet Mix | Float | 0.0-1.0 | 1.0 | Mix control |
| 19 | Input Gain | Float | -12 to +12 dB | 0.0 | Pre-gain |
| 20 | Output Gain | Float | -12 to +12 dB | 0.0 | Post-gain |

---

## 🎨 UI Mockup (Text-based)

```
┌─────────────────────────────────────────────────────────────────┐
│ AURIK Audio Restoration                              [?] [≡]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         [ Waveform Display - Real-time ]                  │ │
│  │  ▁▂▃▅▆█▆▅▃▂▁  ▁▂▃▅▆█▆▅▃▂▁  ▁▂▃▅▆█▆▅▃▂▁                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Medium: [Vinyl ▾]    Mode: [Balanced ▾]    Mix: [100%]        │
│                                                                 │
│  ┌─ Musical Goals ────────────┐  ┌─ Enhancement ────────────┐  │
│  │                             │  │                          │  │
│  │     Brillanz  [====·····]  │  │  □ Bass   □ Drums       │  │
│  │     Wärme     [===······]  │  │  □ Guitar □ Piano       │  │
│  │     Natur.    [====·····]  │  │  □ Brass  □ Spatial     │  │
│  │     Authen.   [=====····]  │  │                          │  │
│  │     Emotion.  [===······]  │  │  Noise: [===·······]    │  │
│  │     Transp.   [=====····]  │  │  Click: [===·······]    │  │
│  │     Bass      [===······]  │  │                          │  │
│  │                             │  │  [Apply] [Bypass]       │  │
│  └─────────────────────────────┘  └──────────────────────────┘  │
│                                                                 │
│  Preset: [Custom ▾]  [Save] [Load]                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 File Structure

```
aurik_plugins/
├── CMakeLists.txt                  # Main build config
├── JUCE/                           # JUCE framework (submodule)
├── libs/
│   ├── pybind11/                   # Python bindings (submodule)
│   └── python/                     # Embedded Python runtime
├── Source/
│   ├── PluginProcessor.h/cpp       # Main audio processor
│   ├── PluginEditor.h/cpp          # UI editor
│   ├── PythonBridge.h/cpp          # Python integration
│   ├── Parameters.h/cpp            # Parameter definitions
│   ├── Presets.h/cpp               # Preset management
│   └── Components/
│       ├── WaveformDisplay.h/cpp   # Real-time waveform
│       ├── RadarChart.h/cpp        # Musical goals viz
│       └── PresetBrowser.h/cpp     # Preset UI
├── Resources/
│   ├── Presets/                    # Factory presets
│   ├── Icons/                      # UI icons
│   └── Fonts/                      # Custom fonts
└── Tests/
    ├── ProcessorTest.cpp           # Unit tests
    └── IntegrationTest.cpp         # DAW integration tests
```

---

## 🚀 Build & Deploy

### Build Commands

```bash
# macOS / Linux
cd aurik_plugins
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Windows
cd aurik_plugins
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Installation Paths

**macOS:**
- VST3: `~/Library/Audio/Plug-Ins/VST3/AURIK.vst3`
- AU: `~/Library/Audio/Plug-Ins/Components/AURIK.component`

**Windows:**
- VST3: `C:\Program Files\Common Files\VST3\AURIK.vst3`

**Linux:**
- VST3: `~/.vst3/AURIK.vst3`

---

## ⚠️ Challenges & Mitigations

### Challenge 1: Python Embedding Performance

**Risk:** Python GIL causes latency spikes  
**Mitigation:**
- Use separate processing thread
- Cache Python objects
- Consider ONNX for ML models
- Batch processing where possible

### Challenge 2: Memory Management

**Risk:** Memory leaks in Python/C++ bridge  
**Mitigation:**
- Use smart pointers (std::unique_ptr)
- RAII patterns
- Valgrind / ASAN testing
- Clear Python references properly

### Challenge 3: Cross-platform Compatibility

**Risk:** Different Python versions per OS  
**Mitigation:**
- Bundle Python runtime with plugin
- Use relative paths
- Test on all platforms
- Provide fallback mechanisms

### Challenge 4: Real-time Constraints

**Risk:** Processing too slow for real-time  
**Mitigation:**
- Report latency accurately to DAW
- Use look-ahead buffer
- Implement bypass during heavy processing
- Optimize critical paths

---

## 📈 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency | < 10ms | DAW latency reporting |
| CPU Usage | < 20% (single core) | Activity Monitor |
| Memory | < 500MB RAM | Profiler |
| Load Time | < 2 seconds | User testing |
| Crash Rate | < 0.1% | Telemetry |
| User Rating | > 4.5/5 | App Store reviews |

---

## 🗓️ Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | JUCE Setup + Python Bridge | Basic audio passthrough |
| 2 | Core Processing Integration | AURIK processing works |
| 3 | UI + Parameters | Full control interface |
| 4 | Optimization + Testing | Real-time performance |
| 5 | Presets + Polish | Production release |

**Est. Completion:** 5 weeks (part-time) / 2-3 weeks (full-time)

---

## 📚 Resources

### Documentation

- [JUCE Tutorials](https://juce.com/learn/tutorials)
- [VST3 SDK Documentation](https://steinbergmedia.github.io/vst3_doc/)
- [Audio Unit Programming Guide](https://developer.apple.com/library/archive/documentation/MusicAudio/Conceptual/AudioUnitProgrammingGuide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)

### Example Projects

- [JUCE Audio Plugin Template](https://github.com/McMartin/JUCE-AudioPlugin-Template)
- [Pamplejuce (Modern JUCE Template)](https://github.com/sudara/pamplejuce)
- [Python in JUCE Example](https://github.com/jatinchowdhury18/python-juce-example)

### Tools

- [Pluginval](https://github.com/Tracktion/pluginval) - Plugin validator
- [auval](https://developer.apple.com/library/archive/technotes/tn2276/) - AU validation (macOS)
- [VST3 Plugin Test Host](https://steinbergmedia.github.io/vst3_dev_portal/pages/What+is+the+VST+3+SDK/Plug-in+Test+Host.html)

---

**Next Step:** Install JUCE and create initial project structure
