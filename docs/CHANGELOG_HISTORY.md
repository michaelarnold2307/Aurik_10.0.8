# Aurik 9 — Spec-Änderungshistorie (v9.7.0 → v9.10.51)

> Diese Datei enthält die vollständige Changelog-Historie der
> `copilot-instructions.md`-Spezifikation sowie Code-Releases. Sie wird nicht zur
> Pipeline-Laufzeit gelesen — sie dient der Nachvollziehbarkeit
> von Architekturentscheidungen für Entwickler und KI-Agenten.
>
> Stand: März 2026 — Aurik 9.10.51

---

## v9.10.51 (14. März 2026) — §SR-Invariante lückenlos

- `backend/core/genre_classifier.py`: `GermanSchlagerClassifier.classify()` — `assert sr == 48000`
- `backend/core/feedback_chain.py`: `FeedbackChain.run()` — `assert _sr == 48000`
- `backend/core/causal_defect_reasoner.py`: Default `44100` → `48000` korrigiert; bedingter Assert
- `backend/core/perceptual_embedder.py`: `PerceptualEmbedder.embed()` — `assert sample_rate == 48000`
- `backend/core/excellence_optimizer.py`: `assert sample_rate == 48000` als erste Zeile
- `backend/core/unified_restorer_v3.py`: Musical-Goals-Re-Pass-Zweig `logger.info` → `logger.warning`

---

## v9.10.50 (14. März 2026) — §Dach: MusikalischerGlobalplan

- `backend/core/musikalischer_globalplan.py` (neu): `MusikalischerGlobalplanDienst` (Singleton,
  Double-Checked Locking); 13 Ära-Profile (1890–2020); 7 Genre-Modifikatoren; 17 Per-Phase-Adjustments
- `backend/core/unified_restorer_v3.py`: `RestorationConfig.global_plan`-Feld; `_profiled_phase_call()`
- `denker/restaurier_denker.py`: `global_plan`-Parameter-Weitergabe
- `denker/aurik_denker.py`: Stufe 2b (DSP-only Globalplan); `AurikErgebnis.global_plan`-Feld
- 60 neue Tests `test_musikalischer_globalplan.py`

---

## v9.10.49 (12. März 2026) — §9.7 Performance-Optimierungen

- `backend/core/defect_scanner.py`: SHA256-Cache (`_scan_cache`, max. 128, FIFO, Thread-sicher)
- `plugins/panns_plugin.py`: SHA256-Cache (`_tags_cache`, max. 128, FIFO, Thread-sicher)
- `backend/core/unified_restorer_v3.py`: Parallele Eingangs-Analyse via `ThreadPoolExecutor(3)`
- `backend/core/per_phase_musical_goals_gate.py`: `PHASE_SAMPLE_DURATIONS` für 6 triviale Phasen
- `Aurik910/main.py`: Hintergrund-Warmup-Thread (daemon=True, 2 s Verzögerung)

---

## v9.10.48 (9. März 2026) — Infrastruktur: SBOM, GP-Backup, i18n-Tests

- `scripts/generate_sbom.py`: SPDX-SBOM-Generator mit SHA256-Modell-Verifikation
- `scripts/backup_gp_memory.py`: Backup/Restore für GP-Speicher (tar.gz)
- `scripts/verify_requirements.py` + `verify_requirements.sh`: pip dry-run CI-Check
- `tests/unit/test_export_roundtrip.py`: 20 Tests FLAC/WAV, Mono+Stereo
- `tests/unit/test_i18n.py`: 20 Tests DE↔EN, Thread-Sicherheit
- `tests/unit/test_gp_memory_migration.py`: 25 Tests v1→v2-Migration, MAX_OBSERVATIONS

---

## v9.10.47 (7. März 2026) — Spec-Konsistenz-Audit: 6 Korrekturen

- **S-1**: `EraResult.is_remaster_suspected: bool = False` in Spec ergsänzt (war seit v9.10.45 implementiert)
- **S-2**: `wrap_phase(restorability_score)` Default-Kommentar verschsärft (nur Testfallback)
- **S-3**: `MaterialQuality`-Enum + `MaterialQualityAssessment`-Dataclass vollständig in §2.31 definiert
- **S-4**: GP-Gedächtnis-Verzeichnis um Genre-Keys erweitert (schlager.json, jazz.json, etc.)
- **S-5**: Manifest-Beispiel: `"bs_roformer"` → `"mdx23c_kim_vocal_2"`
- **S-6**: README.md: Materialanzahl 17 → **15** (3 Stellen); quadrophony/ambisonic entfernt

---

- §2.1: `CausalDefectReasoner` → **27 DefectTypes → 14 Kausal-Ursachen** (war: 24/11)
- §2.2: DefectScanner-Zeile auf **27 DefectTypes** aktualisiert
- §2.4: Ursachen-Liste um `riaa_curve_error`, `aliasing`, `bias_error` ergänzt
- §6.3: DefectType-Vollkatalog auf **27 Defekte** erweitert; neuer Abschnitt
  „Entzerrungs- & Digitalisierungsfehler" mit `RIAA_CURVE_ERROR`, `ALIASING`, `BIAS_ERROR`

---

## v9.10.46b (März 2026) — §2.36 Lyrics-Guided Enhancement (v10.0-Spec)

- `LyricsTranscriber`: Whisper-Tiny ONNX lokal (39 MB), CPUExecutionProvider, stiller DSP-Fallback
- `ContentAwareProcessor`: Phonem-Typ × Betonung → Salienz-Boost 0.5–2.0,
  G_floor 0.90 an fricative+stressed-Bins, PAM-Integration §2.22
- `LyricsGuidedTimeline`: WaveformWidget-Farboverlay, Shortcut `L`,
  Datenschutz: kein Lyrics-Text geloggt
- Pipeline-Position (v10.0): LyricsTranscriber → ContentAwareProcessor → PAM.apply_to_gain()
- Manifest-Eintrag `whisper_tiny` (bundled:true, 39 MB, Fallback: energy_segmentation_dsp)
- §12: Radford et al. (2022) Whisper aufgenommen
- Roadmap: Tier 2+3 abgeschlossen, Tier 4 als v10.0-Ziel

---

## v9.10.46 (März 2026) — Spec-Konsistenz-Audit (14 Lücken)

- §2.2: RestorationResult JSON-Serialisierungsschema ergänzt (audio nicht in JSON,
  NaN/Inf → null, genealogy als Sidecar)
- §2.20: `JAZZ_RESTORATION_PROFILE`, `KLASSIK_RESTORATION_PROFILE`,
  `OPER_RESTORATION_PROFILE`, `ROCK_RESTORATION_PROFILE` mit allen Parametern
- §4.1: DDSP → NumPy/SciPy-Eigenimplementierung `dsp/ddsp_synth.py` (kein TensorFlow)
- §11.4: A/B-Vergleich (A/B-Shortcuts), vollständige Keyboard-Shortcut-Tabelle,
  Preset-Browser & Queue-Widget spezifiziert
- §13.3: Out-of-the-Box-Garantie verschärft — 100 % offline, SOTA-Upgrades lokal gebündelt,
  `sota_upgrade`-Feld nur Entwickler-Metadaten (kein Laufzeit-Download)
- §13.5: Setup-Wizard: SOTA-Upgrade-Checkbox entfernt
- §13.8 (neu): Manuelles Update-Verfahren dokumentiert

---

## v9.10.45 (Feb 2026) — RemasterDetector + temporale Defektverortung

- `core/remaster_detector.py` (neu): `RemasterDetector`-Singleton; analysiert Rauschboden
  (< −80 dBFS → `_floor_score`) und HF-Rolloff (> 18 kHz → `_bw_score`);
  `confidence = 0.55·floor_score + 0.45·bw_score`; `is_remaster=True` wenn ≥ 0.35
- `plugins/era_classifier_plugin.py`: `EraResult.is_remaster_suspected`-Feld ergänzt
- `core/defect_scanner.py`: `_detect_print_through()` → `locations` mit 20-ms-Dedup,
  50-Einträge-Cap und Zeitstempel
- Tests: `test_remaster_detector.py` (18 Tests), `test_defect_scanner_temporal.py` (17 Tests)

---

## v9.10.43 (Feb 2026) — SGMSE+ entfernt, WPE als kanonisches Dereverb-Plugin

- `plugins/wpe_plugin.py` (neu): `WpePlugin`, 3-Tier-WPE (nara_wpe → NumPy-WPE → OMLSA),
  kein Checkpoint, kein Großmodell-Speicher
- `plugins/sgmse_plugin.py`: Thin-Shim → `wpe_plugin` (Backward-Compat)
- `models/sgmse_plus/` gelöscht
- §4.4: SGMSE+ → WPE (Nakatani 2010), 3-Tier-Kaskade
- §9.5: SGMSE+-Eintrag → WPE-DSP-Hinweis (kein RAM-Budget)
- §11.3: sgmse_plugin → wpe_plugin (kanonisch) + sgmse_plugin (Shim)

---

## v9.10.42 (Feb 2026) — SCHRITTE_ZUR_MUSIKALISCHEN_EXZELLENZ abgeschlossen

Testzahl: 6394 → **6312** (nach v2-Cleanup).

- **K-1**: TIER-1/TIER-6-Assertions in `_validate_restoration_result()`
- **K-2**: `quality_estimate`-Formel: `0.40·(1−sev) + 0.60·(pqs_mos−1)/4`; `× 1.15`-Bonus entfernt
- **M-1**: `_SEQUENTIAL_TIER_PHASES`-Frozenset; TIER-0/TIER-1 immer sequenziell (§2.2.1)
- **M-2**: `self._warnings`-Liste; sicherer `_get_phase()`-except-Block
- **M-3**: `_era_for_stereo`-Fallback auf `SimpleNamespace(decade=1960)`
- **I-3**: `scores["artikulation"] = scores["articulation"]`-Alias
- **W-4a**: `VocalAIEnhancement = UnifiedVocalAIEnhancer`-Alias
- **W-4b**: 11 Vokalketteninvarianten-Tests (`test_vocal_chain_invariants.py`)
- **V-5**: CI-Stub-Guard `tests/normative/test_no_production_stubs.py`
- v2-Cleanup: `unified_restorer_v2.py`, `context_aware_deesser_v2.py`, 17 v2-Tests entfernt

---

## v9.9.9 (Feb 2026) — 4 neue Qualitätsmechanismen

### §2.27 TransientDecoupledProcessing (TDP)

- HPSS-Trennung (Medianfilter-Kernel 31) am allerersten Pipeline-Schritt
- `audio_percussive` → NUR phase_01/phase_27; `audio_harmonic` → volle Pipeline
- OLA-Crossfade Hanning 10 ms; DTW-Sicherheitsnetz (> 8 ms → Original-Percussive)
- Effekt: GrooveMetric +0.03–0.06, Timbre-Authentizität +0.02–0.04
- Referenz: Fitzgerald (2010)

### §2.28 HarmonicPreservationGuard (HPG)

- CREPE (full, Fallback pYIN) → f₀(t), Voicing ≥ 0.6
- Harmonisches Gitter fₙ = n·f₀·√(1+B·n²), Fletcher-B aus INHARMONICITY_PRIORS
- G_floor-Override: 0.85 an protected_bins, 0.10 sonst
- Energie-Korrektur nach NR: gain ∈ [1.0, 2.0] + PGHI
- Effekt: Natürlichkeit +0.03–0.07, Authentizität +0.03–0.06

### §2.29 PerPhaseMusicalGoalsGate (PMGG)

- 5-s-Stichprobe nach jeder Phase → measure_quick() auf 6 Schnell-Ziele (≤ 200 ms)
- Δ < −REGRESSION_THRESHOLD → Retry-1 (×0.65) → … → Retry-5 (×0.10) → Rollback
- Adaptiver Schwellwert: 0.012 (restorability ≥ 70) / 0.040 / 0.060 (< 40)
- Max. Retries: 5 (v9.15-B3)

### §2.30 MicroDynamicsEnvelopeMorphing (MDEM)

- 400-ms-LUFS-Profile; G[k] ±3.0 LU; Savitzky-Golay-Glättung; lineare Interpolation
- Stille-Segmente (< −60 LUFS) → G[k] = 0; True-Peak nach Morphing
- Effekt: MicroDynamicsMetric Pearson 0.88 → 0.93–0.96

---

## v9.9.8 (Feb 2026) — Spec-Konsistenz-Audit (7 Inkonsistenzen)

- §2.1: MusicalGoalsChecker → **14** Ziele; CausalDefectReasoner → **23** DefectTypes
- §2.2: 9 neue Module ins Pipeline-Diagramm: RestorabilityEstimator, EraClassifier,
  GermanSchlagerClassifier, UncertaintyQuantifier, IAD, TemporalQualityCoherenceMetric,
  EmotionalArcPreservationMetric; RestorationResult-Felder vervollständigt
- §7.1: `phase_56_spectral_band_gap_repair.py` ergänzt
- §7.2: `compression_artifacts` und `head_wear` in CAUSE_TO_PHASES ergänzt
- §9.1: Checkliste „Musical Goals (alle 8)" → **14**
- §11.3: `flow_matching_plugin.py`, `era_classifier_plugin.py`, `core/genre_classifier.py` ergänzt

---

## v9.9.7 (Feb 2026) — 11 Architektur-Lücken geschlossen

- §1.4 **StemRemixBalancer**: g_voc/g_inst Gain-Korrektur; |LUFS(mix) − L_orig| ≤ 0.3 LU
- §2.21 **EnsembleProcessor**: 3 Ketten (×0.6/×1.0/×1.4), frame-by-frame Goals-Voting
- §2.22 **PerceptualAttentionModel**: PANNs+MERT Salienz-Karte [n_frames × 24] ∈ [0.3, 2.0]
- §2.23 **IntroducedArtifactDetector**: ML_HALLUCINATION / NMF_RESIDUAL_CLICK / SMEARING / MUSICAL_NOISE
- §2.24 **BatchSessionLearner**: GP-Warm-Start zwischen Dateien (SHA256-Session-ID)
- §2.25 **ReferenceAnchorSynthesizer**: 270 Anker (10 Dek × 9 Genres × 3 Mat), k=3 k-NN
- §2.26 **RestorabilityEstimator**: < 5-s-Assessment, Score 0–100, predicted MOS
- §4.5 **SpectralBandGapRepair**: HEAD_WEAR, 56. Phase; Harmonische Interpolation + NMF-β + PGHI
- §6.1: `quadrophony`/`ambisonic` formal aus SUPPORTED_MATERIALS entfernt (Nur Mono/Stereo)
- §8.2: **EmotionalArcPreservationMetric** (Arousal/Valence Pearson, Klimax-Peak-Abweichung)
- §9.6: **Progressive-Quality-Mode**: Stage-1 (5-s, ≤ 8 s) + Stage-2 (volle Pipeline)

---

## v9.9.6 (Feb 2026) — Zero-Shot-Genre-Klassifikation & Schlager-Erkennung

### §2.19 GermanSchlagerClassifier (6-Schicht-Ensemble)

- Tier-1: LAION-CLAP Zero-Shot (7 positive + 5 negative Prompts, 30 % Gewicht)
- Tier-2: Akkordeon Reed-Beating DSP (Hilbert, [5–15] Hz + [4–8] Hz)
- Tier-3: Harmonischer Simplizitäts-Index (CQT-Chroma, Quintenkreis ≤ 2)
- Tier-4: Rhythmus-Muster (madmom → Schunkel/Walzer/Marsch/Disco)
- Tier-5: Deutsch-Vokal-Formant-Prior (LPC-Burg, SAMPA-Polygone, ±0.08 max.)
- Tier-6: Melodische Wiederholungsrate (MFCC-SSM, Schwelle 0.85, ≥ 8 s Abstand)
- Voting: ≥ 3 von 5 DSP-Schichten + Konfidenz ≥ 0.52 → is_schlager=True
- `SCHLAGER_RESTORATION_PROFILE`: TonalCenter 0.97, Wärme 0.88, Brillanz 0.82

### §2.20 Genre-Klassifikations-Matrix

- 9 Genres mit Erkennungsmethode und Pipeline-Anpassung dokumentiert

---

## v9.9.5 (Feb 2026) — Weltführungsanspruch (14 Spec-Lücken)

- Musical Goals 9 → **14**: TonalCenterMetric (Chroma ≥ 0.95) + MicroDynamicsMetric (LUFS ≥ 0.92)
- §2.14 EraClassifier: 1890–2025, CLAP-Tier-1 + DSP-Tier-2 + Mikrofon-Tier-3
- §2.15 Uncertainty Quantification: Konfidenz-Schwellen 0.80/0.50/0.00
- §2.16 TemporalQualityCoherenceMetric: MOS-Spanne ≤ 0.30, σ ≤ 0.15
- §2.17 MusicalStructureAnalyzer: SSM + Novelty-Kurve (Foote 2000), Chorus-Prior
- §2.18 StereoAuthenticityInvariant: Mono ≥ 0.97, Decca ∈ [0.25, 0.65], Abbey Road ≤ ±3°
- §4.4: Flow Matching, EnCodec/DAC, EraClassifier, MusicalStructureAnalyzer ergänzt
- §6.1/6.2: `wax_cylinder`, `wire_recording`, `lacquer_disc` hinzugefügt (MOS ≥ 3.5/3.6/3.7)
- §8.1: Rauschboden ≤ −72 dBFS / −75 dB(A), HF-Limit ≤ +4 dB kumulativ
- §8.2: EmotionalArcPreservationMetric spezifiziert; Kompetitiver Benchmark (≥ iZotope RX 11)

---

## v9.9.4 (Feb 2026) — SHA256-Größenabgleich (15 Modelle)

- MDX23C: 2× 64 MB; Apollo: 65 MB; CREPE: 85 MB; DeepFilterNet: 37 MB
- Vocos: 52 MB; Banquet Vinyl: 92 MB; Resemble-Enhance: 41 MB
- UVR HQ: 56–64 MB; HTDemucs 6s: 2,5 MB

---

## v9.9.3 (Feb 2026) — Manifest-Verifizierung & Offline-Garantie

- Apollo SHA256 `440c48b1…` / 67,7 MB als primäres Manifest-Modell korrigiert
- SGMSE+, MERT (3,9 GB), AudioSR (5,9 GB) als Lazy-Load dokumentiert
- §11.3: ✅-Markierungen für lokal gebündelte vs. SOTA-Upgrade-Modelle

---

## v9.9.2 (Feb 2026) — Manifest-Abgleich

- §4.4: Apollo primär (bundled) / Resemble-Enhance Fallback (nicht umgekehrt)
- §11.3: 18 Manifest-Modelle und Plugin-Dateien vollständig dokumentiert

---

## v9.9.1 (Feb 2026) — SOTA-Audit & Musik-Ausrichtung

- Musical Goal 9: **SpatialDepthMetric** (≥ 0.75)
- OQS-Evaluator (algorithmische PEAQ-Approximation, kein BS.1534-3-Hörertest)
- AMRB v1.0 (10 Szenarien, OS-Führerschaft ≥ 84.0)
- §4.4: BS-RoFormer, CQTdiff+, Apollo, LAION-CLAP, UTMOS/VERSA als neue Primär-Algorithmen
- BigVGAN-v2 (NVIDIA 2024, Apache-2.0) als primärer Vocoder-Endschritt (MOS < 4.3)
- PESQ/DNSMOS/NISQA/STOI/POLQA explizit verboten (§4.4, §10.2, §11.3)
- CDPAM als primäre Musik-Wahrnehmungsmetrik; ViSQOL v3 --audio Mode erzwungen

---

## v9.9.0 (Feb 2026) — Über-SOTA-DSP-Erweiterungen

- Musical Goal 8: **GrooveMetric** (DTW ≤ 8 ms RMS)
- **Multi-Resolution STFT MRSA**: 128–65536 Samples pro Frequenzzone + PGHI
- **Psychoakustisches Masking-Modell** (ISO 11172-3, OMLSA-Gain-Modifier)
- §2.11 **Harmonic Lattice Coherence** (Fletcher-Modell, B-Koeff., ±3 Cent)
- §2.12 **Musikalische Phrasenkontextfenster** (madmom Beat-Tracking, ≤ 30 s Kontext)
- §2.13 **Künstler-Signaturmodell** (Formant/Vibrato/Breathiness, artist_signatures/)
- `SOFT_SATURATION` als 22. DefectType (Tube-Sättigung BEWAHREN)
- **Noise-Shaped Dithering POW-r Typ 3** beim 24→16 bit Export

---

## v9.8.0 (Feb 2026) — Architektur-Fundament

- Thread-safe Singletons (Double-Checked Locking, §3.2)
- PEP 484 Type-Annotation-Pflicht + mypy strict (§3.7)
- SHA256-Ergebnis-Cache für teure Operationen (§3.8)
- MOO-GP-Optimizer (Pareto-Front, 14 Objectives)
- **SegmentAdaptiveProcessor** (Content-Aware, §2.10)
- Consonant Enhancement in Vocal-Pipeline
- Print-Through: Adaptive Temporal Subtraction + CAUSE_TO_PHASES
- PEAQ/FAD als optionale Parallel-Metriken
- Reference Mastering (Optimal Transport)
- Defektdichte-adaptive Chunk-Größe (5/15/60/120 s)
- Restaurierungs-Genealogie / Sample-Audit-Trail

---

## v9.7.0 (Basis)

- Initiale Systemspezifikation: 14 Musical Goals, 27 DefectTypes
- Pipeline-Grundstruktur: TDP → RestorabilityEstimator → EraClassifier → … → MDEM
- Kernmodule: PerceptualEmbedder, CausalDefectReasoner, GPParameterOptimizer,
  PerceptualQualityScorer, MediumClassifier, DefectScanner, UnifiedRestorerV3
- Out-of-the-Box-Pflicht (bindend): AppImage / NSIS-Installer, alle Modelle gebündelt
- Restaurierungs-Modi: Restoration + Studio 2026
