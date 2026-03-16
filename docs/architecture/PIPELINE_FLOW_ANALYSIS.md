# Aurik 9.10.51 — Pipeline Flow Analysis

**Stand:** März 2026  
**Status:** 56-Phasen-Defect-First-Pipeline, vollständig implementiert

---

## Kernprinzip: Defect-First + Kognitive Steuerung

Im Gegensatz zu früheren Versionen (14-Phasen, fixe Reihenfolge) verarbeitet
Aurik 9.10.51 Audio über eine kognitive Pipeline: erst verstehen (DefectScanner +
CausalReasoner), dann optimiert intervenieren (GP-Optimizer), dann verifizieren
(14 Musical Goals via PMGG).

## Signal-Fluss

```
[Rohes Audio] 
    |
    v
[TDP: HPSS] --> audio_percussive (NUR phase_01+27) 
    |            audio_harmonic (volle Pipeline)
    v
[Restorability 0-100] + [Ära 1890-2025] + [Genre: Schlager?]
    |
    v
[Material: 17 Typen] + [24 DefectTypes] + [11 Kausal-Ursachen]
    |
    v
[GP-Optimizer: 10 Parameter, MOO-Pareto über 14 Ziele]
    |
    v
[HPG: Harmonic-Maske fuer OMLSA/DeepFilterNet G_floor-Override]
    |
    v
[Phasen 01-56, je via PMGG-Gate: Rollback bei Regression]
    |
    v
[EraAuthenticPerceptualCompletion: DDSP bei BW < 10 kHz]
    |
    v
[IAD: ML_HALLUCINATION / NMF_RESIDUAL_CLICK / PHASE_VOCODER_SMEARING]
    |
    v
[FeedbackChain: max. 5 Iter., Konvergenz Delta|MOS| < 0.02]
    |
    v
[TemporalCoherence: MOS-Spanne ueber 10-s-Segmente <= 0.30]
    |
    v
[PQS-MOS: Gammatone-NSIM + MCD + LUFS + MOS]
    |
    v
[ExcellenceOptimizer: GP-Fine-Tuning, Physical Ceiling]
    |
    v
[14 Musical Goals: GoalApplicabilityFilter + GoalPriorityProtocol]
    |
    v
[EmotionalArcPreservationMetric: Arousal/Valence Pearson]
    |
    v
[MicroDynamicsEnvelopeMorphing: 400ms LUFS-Profil-Korrektur]
    |
    v
[TDP-Rekombination: Hanning OLA 10 ms]
    |
    v
[RestorationResult]
```

## Studio 2026-Modus: Zusätzliche Stem-Verarbeitung

```
[MDX23C Stem-Separation] --> Vocals + Instrumente
    |
    v
[VocalAIEnhancement + ConsonantEnhancement (stimmtyp-adaptiv)]
    |
    v
[Genre-adaptive Instrument-Verarbeitung (PANNs)]
    |
    v
[Reference Mastering: Optimal Transport, optional]
    |
    v
[StemRemixBalancer.balance_remix()] -- LUFS-invariant +/-0.3 LU
    |
    v
[LUFS -14 EBU R128 + True-Peak -1.0 dBTP]
    |
    v
[Vocos-Synthese konditionell: nur wenn PQS-MOS < 4.3]
```

## Qualitätsgrenzen

| Metrik | Hard-Fail-Schwelle | Weltklasse-Ziel |
|---|---|---|
| PQS MOS | >= 3.8 | >= 4.5 |
| NSIM | >= 0.70 | >= 0.90 |
| MCD | <= 8.0 dB | <= 3.0 dB |
| Spectral Coherence | >= 0.60 | >= 0.85 |
| quality_estimate | >= 0.55 | > 0.85 |
| Alle 14 Musical Goals | >= Pflicht-Schwellwert | >= Studio-Schwellwert |

---

*Aurik 9.10.51 — März 2026*
