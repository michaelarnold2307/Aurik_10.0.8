# 📊 Aurik 9.10.51 — Project Status Report

**Datum:** März 2026  
**Version:** 9.10.51  
**Status:** ✅ Produktionsbereit — Weltführendes Musik-Restaurierungssystem

---

## Executive Summary

**Aurik 9.10.51 ist das weltweit erste denkende Musik-Restaurierungssystem.**

| Kennzahl | Wert |
|---|---|
| Tests | **6312** — alle grün ✅ |
| Phasen | **56** (Phase 01–56, Defect-First) |
| Materialien | **15** auto-erkannte Typen (inkl. wax_cylinder, wire_recording, lacquer_disc) |
| Musical Goals | **14** psychoakustisch fundierte Ziele |
| PQS MOS | **>= 4.0** (Minimum) / **>= 4.5** (Weltklasse) |
| DefectTypes | **27** erkennbare Defektarten (inkl. RIAA_CURVE_ERROR, ALIASING, BIAS_ERROR) |
| Hardware | CPU-only, Desktop (Linux & Windows 10/11) |
| Netzwerk | Keine Cloud, keine Serverabhängigkeiten — 100 % offline |

---

## 🧠 Kognitive Architektur — Vollständig implementiert

| Modul | Datei | Status |
|---|---|---|
| `PerceptualEmbedder` | `core/perceptual_embedder.py` | ✅ |
| `CausalDefectReasoner` | `core/causal_defect_reasoner.py` | ✅ |
| `GPParameterOptimizer` (MOO-Pareto) | `core/gp_parameter_optimizer.py` | ✅ |
| `PerceptualQualityScorer` | `core/perceptual_quality_scorer.py` | ✅ |
| `MusicalGoalsChecker` (14 Ziele) | `backend/core/musical_goals/musical_goals_metrics.py` | ✅ |
| `MediumClassifier` | `core/medium_classifier.py` | ✅ |
| `DefectScanner` (27 DefectTypes) | `core/defect_scanner.py` | ✅ |
| `VocalAIEnhancement` | `core/vocal_ai_enhancement.py` | ✅ |
| `ExcellenceOptimizer` | `core/excellence_optimizer.py` | ✅ |
| `FeedbackChain` | `core/feedback_chain.py` | ✅ |
| `UnifiedRestorerV3` | `core/unified_restorer_v3.py` | ✅ |
| `EraClassifier` | `plugins/era_classifier_plugin.py` | ✅ |
| `GermanSchlagerClassifier` | `core/genre_classifier.py` | ✅ |
| `TransientDecoupledProcessing` | `core/transient_decoupled_processor.py` | ✅ |
| `HarmonicPreservationGuard` | `core/harmonic_preservation_guard.py` | ✅ |
| `PerPhaseMusicalGoalsGate` | `core/per_phase_musical_goals_gate.py` | ✅ |
| `MicroDynamicsEnvelopeMorphing` | `core/micro_dynamics_envelope_morphing.py` | ✅ |
| `RestorabilityEstimator` | `core/restorability_estimator.py` | ✅ |
| `StemRemixBalancer` | `core/stem_remix_balancer.py` | ✅ |
| `RemasterDetector` | `core/remaster_detector.py` | ✅ |
| `AdaptiveGoalThresholds` | `backend/core/musical_goals/adaptive_goals_system.py` | ✅ |
| `GoalApplicabilityFilter` | `core/goal_applicability_filter.py` | ✅ |
| `GoalPriorityProtocol` | `core/goal_priority_protocol.py` | ✅ |
| `PhysicalCeilingEstimator` | `core/physical_ceiling_estimator.py` | ✅ |
| `EraAuthenticPerceptualCompletion` | `core/era_authentic_perceptual_completion.py` | ✅ |
| `IntroducedArtifactDetector` | `core/introduced_artifact_detector.py` | ✅ |
| `TemporalQualityCoherenceMetric` | `core/temporal_quality_coherence.py` | ✅ |
| `EmotionalArcPreservationMetric` | `core/emotional_arc_preservation.py` | ✅ |
| `EnsembleProcessor` | `core/ensemble_processor.py` | ✅ |
| `PerceptualAttentionModel` | `core/perceptual_attention_model.py` | ✅ |
| `MusikalischerGlobalplanDienst` | `backend/core/musikalischer_globalplan.py` | ✅ |
| `BatchSessionLearner` | `core/batch_session_learner.py` | ✅ |
| `ReferenceAnchorSynthesizer` | `core/reference_anchor_synthesizer.py` | ✅ |

---

## 🎯 14 Musical Goals — Qualitätsstatus

Alle 14 Ziele werden durch `MusicalGoalsChecker.measure_all()` nach jeder Restaurierung geprüft.
Regression in einem anwendbaren Ziel macht das Feature ungültig.

| Ziel | Klasse | Pflicht-Schwellwert | Studio 2026 |
|---|---|---|---|
| Brillanz | `BrillanzMetric` | >= 0.85 | >= 0.90 |
| Wärme | `WaermeMetric` | >= 0.80 | >= 0.80 |
| Natürlichkeit | `NatuerlichkeitMetric` | >= 0.90 | >= 0.90 |
| Authentizität | `AuthentizitaetMetric` | >= 0.88 | >= 0.88 |
| Emotionalität | `EmotionalitaetMetric` | >= 0.87 | >= 0.87 |
| Transparenz | `TransparenzMetric` | >= 0.89 | >= 0.89 |
| Bass-Kraft | `BassKraftMetric` | >= 0.85 | >= 0.88 |
| Groove | `GrooveMetric` | >= 0.88 | >= 0.88 |
| Raumtiefe | `SpatialDepthMetric` | >= 0.75 | >= 0.75 |
| Timbre-Authentizität | `TimbralAuthenticityMetric` | >= 0.87 | >= 0.87 |
| Tonales Zentrum | `TonalCenterMetric` | >= 0.95 | >= 0.97 |
| Mikro-Dynamik | `MicroDynamicsMetric` | >= 0.92 | >= 0.93 |
| Separation-Treue | `SeparationFidelityMetric` | >= 0.82 | >= 0.82 |
| Artikulation | `ArticulationMetric` | >= 0.85 | >= 0.85 |

`GoalApplicabilityFilter` deaktiviert physikalisch irrelevante Ziele automatisch (z. B. SpatialDepthMetric
bei Mono-Aufnahmen <= 1950). Mindestens 6 Ziele bleiben immer aktiv: Natürlichkeit, Authentizität,
Emotionalität, Transparenz, Timbre-Authentizität, Artikulation.

---

## 📋 56-Phasen-Pipeline (kanonisch)

```
TransientDecoupledProcessing (TDP)
-> RestorabilityEstimator -> EraClassifier -> GermanSchlagerClassifier
-> MediumClassifier -> DefectScanner -> CausalDefectReasoner
-> UncertaintyQuantifier -> GPParameterOptimizer
-> HarmonicPreservationGuard
-> PerPhaseMusicalGoalsGate (umhüllt jede Phase, Rollback bei Regression)
-> Phasen-Ausführung (01–56)
-> EraAuthenticPerceptualCompletion (konditionell, BW < 10 kHz)
-> IntroducedArtifactDetector -> FeedbackChain
-> TemporalQualityCoherenceMetric -> PerceptualQualityScorer
-> ExcellenceOptimizer -> MusicalGoalsChecker (14 Ziele)
-> EmotionalArcPreservationMetric
-> MicroDynamicsEnvelopeMorphing
-> GPParameterOptimizer.update()
-> RestorationResult
```

- Phase 01–30: Defektkorrektur
- Phase 31–46: Enhancement (EQ, Stereo, Gesang, Instrumente)
- Phase 47–55: Mastering (True-Peak, LUFS, DiffWave-Inpainting)
- Phase 56: SpectralBandGapRepair (HEAD_WEAR, confidence >= 0.55)

---

## 📦 15 Materialien

| Material | Prioritäts-Phasen | PQS MOS |
|---|---|---|
| `tape` | 24, 29, 12 | >= 4.2 |
| `reel_tape` | 29, 03, 24, 55 | >= 4.3 |
| `vinyl` | 09, 12, 30 | >= 4.0 |
| `shellac` | 03, 06, 01 | >= 3.8 |
| `wax_cylinder` | 03, 06, 01, 29 | >= 3.5 |
| `wire_recording` | 12, 24, 03, 29 | >= 3.6 |
| `lacquer_disc` | 01, 09, 03, 29 | >= 3.7 |
| `dat` | 24, 02, 23 | >= 4.4 |
| `cd_digital` | 23, 06, 40 | >= 4.5 |
| `mp3_low` | 23, 03, 50 | >= 3.9 |
| `mp3_high` | 23, 50 | >= 4.2 |
| `aac` | 23, 38, 06 | >= 4.2 |
| `minidisc` | 23, 06, 07 | >= 4.0 |
| `streaming` | 03, 23, 50 | >= 4.1 |
| `unknown` | Alle Tier-1 | >= 3.8 |

---

## 🎤 Stimmtyp-Adaptierung (VocalAIEnhancement)

| Typ | F0-Bereich | F1-Bereich | De-Essing-Ziel |
|---|---|---|---|
| MALE | 85–180 Hz | 270–730 Hz | 5–10 kHz |
| FEMALE | 165–255 Hz | 310–860 Hz | 6–12 kHz |
| CHILD | 200–500 Hz | 370–1030 Hz | 7–14 kHz |
| ANDROGYNOUS | auto-detect | auto-detect | adaptiv |
| UNKNOWN | — | — | FEMALE-Fallback |

Invarianten: Formant-Pearson >= 0.95 · Breathiness +/-0.05 · Vibrato +/-0.3 Hz
ConsonantEnhancement: Frikative-SNR >= +3 dB · HF-Anhebung <= +6 dB · Crossfade 5 ms

---

## 🔧 Entwicklungs-Roadmap

### ✅ Abgeschlossen

| Version | Milestone | Tests |
|---|---|---|
| v9.0 | UnifiedRestorerV3, Material-Auto-Detektion | 6 Tests |
| v9.5 | ML-Hybrid, 12 Materialien, 21 DefectTypes, 55 Phasen | 166 Tests |
| v9.7 | Kognitive Architektur (5 Kernmodule), VoiceGender, PANNs | 206 Tests |
| v9.8 | Über-SOTA DSP (OMLSA/IMCRA, pYIN, NMF-b, PGHI) | 222 Tests |
| v9.9.0 | GrooveMetric (#8), MRSA, Psychoakust. Masking, HarmonicLattice | 5169 Tests |
| v9.9.5 | 14 Musical Goals, EraClassifier, TonalCenter, MicroDynamics | 6073 Tests |
| v9.9.7 | StemRemixBalancer, EnsembleProcessor, IAD, BatchSessionLearner | 6180 Tests |
| v9.9.9 | TDP, HPG, PMGG, MDEM | 6312 Tests |
| v9.10.42 | E2E-Tests, TIER-Invarianten, PMGG-Fixes, v2-Cleanup | 6312 Tests |
| v9.10.43 | WPE als kanonisches Dereverb (SGMSE+ entfernt) | 6312 Tests |
| v9.10.45 | RemasterDetector, EraResult.is_remaster_suspected, temporale Defektverortung | 6347 Tests |
| v9.10.46 | Spec-Konsistenz-Audit, JSON-Schema, Genre-Profile, DDSP, UI-Shortcuts | 6312 Tests |
| v9.10.47 | Spec-Konsistenz-Audit: 6 Korrekturen (EraResult, PMGG-Default, MaterialQuality, GP-Genre-Keys) | 6312 Tests |
| v9.10.48 | Infrastruktur: SBOM, GP-Backup, i18n-Tests, Export-Roundtrip | 6312 Tests |
| v9.10.49 | Performance: SHA256-Cache, parallele Eingangs-Analyse, PMGG-Sample-Dauer, Warmup-Thread | 6312 Tests |
| v9.10.50 | §Dach: MusikalischerGlobalplan, 13 Ära-Profile, Genre-Modifikatoren, 17 Phase-Adjustments | 6312 Tests |
| v9.10.51 | §SR-Invariante: assert sample_rate==48000 lückenlos an allen API-Einstiegspunkten | 6312 Tests |

### 🔜 Geplant

| Version | Milestone |
|---|---|
| v10.0 | Multi-Modal-Restaurierung (Audio + Metadaten + Visual) |

---

## 📤 Export & Qualitätsnormen

**Importformate:** WAV, AIFF, FLAC, MP3, AAC/M4A, OGG, WMA, Opus, CAF  
**Exportformate:** FLAC (24-bit), WAV (24-/16-bit), MP3 CBR/VBR (LAME), OGG (q9), AIFF (24-bit)

Lautheit: EBU R128 — -14 LUFS (Streaming) / -18 LUFS (Archiv)  
True-Peak: -1.0 dBTP (ITU-R BS.1770-5)  
Dithering: POW-r Typ 3 bei 24->16-bit; Fallback: TPDF

---

## ⚙️ Technische Konstanten

| Parameter | Wert |
|---|---|
| Interne SR | 48 000 Hz (Pflicht, `assert sample_rate == 48000`) |
| Bit-Tiefe intern | float32, [-1, 1] |
| Hardware | CPU-only (`providers=["CPUExecutionProvider"]`) |
| Resampling | Lanczos-4 (`scipy.signal.resample_poly`, Kaiser b=14) |
| GP-Gedächtnis | `~/.aurik/gp_memory/<material>.json` |
| FeedbackChain | max. 5 Iterationen, D|MOS| < 0.02 |
| PMGG Regression-Threshold | adaptiv: 0.012 / 0.040 / 0.060 |
| PMGG Max-Retries | 5 (strength x 0.65 -> 0.50 -> 0.35 -> 0.20 -> 0.10) |
| Chunk-Verarbeitung | defektdichte-adaptiv: 5 s / 15 s / 60 s / 120 s |

---

## 🔬 Primäre ML-Modelle (lokal gebündelt, 100 % offline)

| Modell | Anwendungsfall | Größe | Fallback |
|---|---|---|---|
| DeepFilterNet v3.II | Breitrauschen (NR) | ~37 MB ONNX | OMLSA/IMCRA DSP |
| MDX23C Kim_Vocal_2/Kim_Inst | Stem-Separation | 2x 64 MB ONNX | NMF-b |
| Apollo | Codec-Artefakte | ~65 MB ONNX | DSP Spectral Repair |
| Vocos 24 kHz | Neuronaler Vocoder | ~52 MB ONNX | HiFi-GAN -> PGHI |
| CREPE full | Pitch-Tracking f0 | ~85 MB ONNX | pYIN DSP |
| PANNs CNN14 | Audio-Tagging | ~81 KB ONNX | DSP Fingerprint |
| DiffWave | Dropout-Inpainting | ~552 KB ONNX | NMF-b + Sinusoidal |
| Resemble-Enhance | Apollo-Fallback | ~41 MB ONNX | DSP Spectral Repair |
| HiFi-GAN | Vocoder-Fallback | ~3.6 MB ONNX | PGHI-ISTFT |

---

## ✅ Universelle Garantien

| Garantie | Prüfung |
|---|---|
| Kein NaN/Inf im Audio-Ausgang | `np.isfinite(audio).all()` |
| Kein Clipping | `np.max(np.abs(audio)) <= 1.0` |
| Chroma-Korrelation | Pearson >= 0.95 |
| Pass-Through (sauberes Material) | PQS-MOS-Verlust <= 0.05, alle 14 Goals stabil +/-0.02 |
| Rauschboden (Studio-2026) | Residual <= -72 dBFS, A-gew. <= -75 dB(A), 0 Musical-Noise |
| Temporale Kohärenz | MOS-Spanne <= 0.30, sigma(MOS) <= 0.15 |
| Stereo-Authentizität | Mono-Ära M/S-Korrelation >= 0.97 |
| HF-Kumulativ-Limit | Presence + Air kumulativ <= +4 dB |

---

*Aurik 9.10.51 — März 2026*
