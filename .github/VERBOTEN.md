# Aurik 9 — Vollständige VERBOTEN-Tabelle

> **Normative Quelle** für alle `[V01–V12]`-Linter-Regeln in `scripts/aurik_verboten_linter.py`.
> Top-10 häufigste Regressions-Ursachen → [`.github/copilot-instructions.md`](copilot-instructions.md)
>
> Inhalt: Grundregeln (Teil A) + Anti-Patterns mit Produktions-Evidenz (Teil B)

---

## Teil A — Grundregeln

| Kategorie | Verboten | Richtig |
| --- | --- | --- |
| Logging | `print(...)` | `logger.info(...)` |
| API-Return | `return dict` | `@dataclass` |
| Cache | `_cache = {}` | `threading.Lock()` + Dict |
| Phase-Rekonstruktion | `griffinlim()` als Endschritt | PGHI / Vocos |
| Normalisierung | RMS / Peak | LUFS ITU-R BS.1770-5 |
| Audio-Import | `sf.read(path)` / `librosa.load(path)` | `load_audio_file(filepath)` |
| Backend-Import | `from Aurik910... import` in `backend/` | Architektur-Trennung |
| GPU | `map_location="cuda"` ohne ml_device_manager | `get_torch_device("PluginName")` via ml_device_manager |
| Musikmetriken | `pesq()`, `dnsmos()`, `nisqa()` | PQS-MOS, VERSA, SingMOS |
| Wiener-Filter | `scipy.signal.wiener()` primär | OMLSA / DeepFilterNet |
| LPC | Ordnung < 16 | Ord. 30–40 @ 48 kHz |
| ML-Budget | `plm.try_allocate()` | `ml_memory_budget.try_allocate()` |
| Tonträgerkette | `MediumClassifier.classify_medium()` | `MediumDetector.detect(audio, sr, file_ext=...)` (§6.7) |
| DC-Offset reel_tape | `np.mean`-Subtraktion / `lfilter` | `scipy.signal.filtfilt([1,-1],[1,-0.9995])` zero-phase |
| SongCal-Bounds | `np.clip(scalar, 0.0, 2.0)` | `global_scalar∈[0.50,1.50]`, `family_scalar∈[0.30,1.80]` |
| MDX23C-Fallback | `HPSS` direkt | NMF-β-Separation (sdB ≥ 5) → HPSS als tertiärer Fallback |
| Pflicht-Phasen | DefectScanner allein entscheidet | Material-Pflicht-Phasen (§6.2a) immer aktivieren |
| Peak-Guard (Gain) | `np.max(np.abs(audio))` | `np.percentile(np.abs(audio), 99.9)` |
| Dolby NR Inversion | Statische globale HF-Absenkung ohne Typ-Erkennung | `DolbyNRDetector.detect()` → `phase_04(dolby_nr_type=..., dolby_nr_confidence=...)` (§6.7 Phase 1c) |
| Head-Bump Tape | Kein LF-Kerbfilter bei Tape-Material | `phase_04(tape_speed_ips=X)` → HEAD_BUMP_PROFILES[nearest_speed] parametrischer Dip |
| Inpainting HF-Halluzination | AR/Diffusion ohne BW-Begrenzung | `_MATERIAL_BW_CAP_HZ` in phase_55 — wax_cylinder ≤ 5kHz, wire_recording ≤ 6kHz (§0) |
| Analogquelle in digitalem Dateicontainer supprimieren | `file_ext=.mp3` → alle Analog-Posteriors auf 0 → vinyl/reel_tape dauerhaft unerkannt | Fallback-Gate Pflicht: `rotation_strength ≥ 0.30 AND conf ≥ 0.20 → vinyl akzeptiert`. `file_ext` bestimmt **ausschließlich die letzte Kettenstufe** (§2.46b) |
| reel_tape vs. cassette ohne Disambiguation | Universelle `wow_flutter`-Schwelle ohne Disc-Kontext | Studio-Pfad: `max(0.010, 0.025×(1−0.55×cc))`; `wow < 0.06 WRMS → reel_tape; wow ≥ 0.06 WRMS → cassette` (IEC 60386:1987, §2.46b) |
| Phase_63 Stereo IMD | Unabhängiges L/R-IMD-Notch | M/S-Domain: Notch-Maske aus Mid, symmetrisch auf Mid+Side (§2.51) |
| Phase-Wetness ohne Feedback | Feste `strength` ohne Mess-Feedback | `PhaseConductor.recommend()` (§2.52) — 4D-State-Vektor → adaptiver Strength-Hint |
| Feste Guard-Schwellwerte | `MAX_DRIFT = -0.05` / `regression > 0.02` als Konstanten | `compute_adaptive_drift_tolerance()` aus Material/Restorability/Defects (§2.54) |
| PhaseSkipper rohe Severity | `defect_score.severity` direkt ohne Salience-Gewichtung | `_salience_adjusted_severity()` (§2.47) — fully-masked (n_masked≥3, n_salient=0) → -50 % |
| Carrier-Formant-Inversion | Phase 42 ohne Material-Kontext in `_enhance_channel` | `_restore_carrier_formant_decay(audio, sr, material_type)` Stage 0.5 (§2.52, Hebel 4) |

---

## Teil B — Anti-Patterns mit Produktions-Evidenz

| Kategorie | Verboten | Richtig |
| --- | --- | --- |
| Loudness-Guard RMS | `np.mean(audio**2)` (globaler RMS in Guards) | `_rms_dbfs_gated()` — Frame-basiert, nur Frames > −50 dBFS, Stille ignoriert (§2.45a-I) |
| Loudness-Guard Gain | `audio *= gain_factor` (uniformer Gain) | `_musical_gain_envelope()` — Gain nur auf Musik-Frames, Stille unverändert (§2.45a-II) |
| Loudness-Guard Limiter | Unbedingter Soft-Limiter nach Makeup-Gain | Soft-Limiter NUR wenn `peak > 0.98` — keine Routine-Dynamik-Kompression (§2.45a-III) |
| `gate_dbfs=-36.0` ohne `reference_for_gate` [V04] | Vinyl/Shellac Rauschboden −33 dBFS > −36 dBFS Gate → Rausch-Frames erhalten Makeup-Gain → Pegelexplosion (bestätigt 2026-04-27, 18 Dateien in 2 Sessions re-introduced) | `compute_signal_relative_gate_dbfs(ref, material_key=...)` via `reference_for_gate=pre_phase_audio` in `apply_musical_gain_envelope` — `reference_for_gate` ist Pflicht-Argument |
| `sosfilt(sos, audio)` addiert zu Signal [V11] | Kausaler Filter → Gruppen-Zeitversatz → destruktive Interferenz → Pegelexplosion | `sosfiltfilt(sos, audio)` (zero-phase) überall wo Bandfilter-Ergebnis auf Originalsignal addiert wird; `sosfilt` nur für Analyse/Sidechain |
| FeedbackChain feste Schwellen | `_prune_threshold = -0.05` / `if history[-1] < history[-2] - 0.05` als Konstanten | `_compute_adaptive_prune_threshold(is_restorative, material, rest, severity)` — shellac 3×, vinyl 2×, clamped [-0.30, base] |
| Carrier-Repair consecutive_rollbacks [V09] | Carrier-Repair-Rollback inkrementiert `consecutive_rollbacks` | `_CARRIER_REPAIR_PHASE_PREFIXES`-Check vor Increment: Carrier-Repair-Rollbacks niemals zählen (§2.48 v9.11.3) |
| Spectral-Tilt-Drift in ADDITIVE-Phasen | HF-Extension (phase_06, phase_39) ohne Tilt-Check | `era_result.spectral_tilt` prüfen; Post-Tilt via `_estimate_spectral_tilt_quick()`; Cap wenn Deviation > material_tolerance (±1.5–±3.0 dB/oct) (Spec 04 §4.7) |
| Roughness/Sharpness Anstieg ungeprüft | DYNAMICS/ADDITIVE-Phasen ohne psychoakustisches Gate | `ArtifactFreedomGate._compute_roughness_zwicker()` + `_compute_sharpness_bismarck()`; Penalty -0.05 / -0.10 (§2.49c) |
| MERT als primäre Qualitätsmetrik | `MertPlugin.score()` als HPI-Haupt-Koeffizient wenn VERSA verfügbar | VERSA primär; MERT als Proxy-Fallback; `metadata["mert_proxy_used"]` setzen (§2.44) |
| VERSA auf RESEARCH-Modus beschränkt | `use_versa_in_loop=deployment_mode == RESEARCH` | `use_versa_in_loop=True` — VERSA ist produktionsstabil (§2.44) |
| DR-Expansion ohne Ceiling | `phase_26` expandiert ungeprüft über Material-Limit | `_MATERIAL_DR_CEILING_DB`: Vinyl ≤ 70 dB, Shellac ≤ 45 dB, Kassette (tape) ≤ 62 dB, Reel-Tape ≤ 72 dB, CD ≤ 96 dB (§6.2b) |
| BW-Extension ohne Ceiling | Phase_06/07/23/39 erzeugen Frequenzinhalt über Material-BW-Limit | `_MATERIAL_BW_CEILING_HZ` Hard-Cap: Shellac ≤ 8 kHz, Vinyl ≤ 16 kHz (§6.2c) |
| Rauschtextur-Check fehlt | Denoising erzeugt weiße Rauschtextur statt Carrier-kohärentem Profil | `NoiseTextureCoherenceGuard` (§4.7): Kohärenz ≥ 0.80 in Restoration; `wet_mult=0.85` für [0.60,0.80) |
| Goals gegen degradierten Input am Pipeline-Ende | `MusicalGoalsChecker.measure_all(restored, sr, original=degraded_input)` | Bei `carrier_chain_recovery_ratio > 0.15`: End-Referenz = `best_carrier_checkpoint` (§0d Ebene 2) |
| carrier_chain_recovery_ratio fehlt | Metadata ohne Carrier-Recovery-Signal | UV3 Pflichtfeld `metadata["carrier_chain_recovery_ratio"]` — berechnet nach letzter Carrier-Phase (§0d) |
| ML-Inferenz ohne PLM-Active-Guard | `session.run()` / `model(input)` ohne `plm.set_active()` | `plm.set_active("model", True)` VOR Inferenz, `plm.set_active("model", False)` in `finally`-Block (§4.6b) |
| `_PHASE_REQUIRED_MODELS` unvollständig | Phase listet nur Primärmodell, nicht Fallback-Modelle | Alle ML-Modelle (primär + Fallback) in `_PHASE_REQUIRED_MODELS`; bidirektionale Sync mit `try_allocate()` (§4.6c) |
| Pitch-Kaskade ohne RMVPE | FCPE → CREPE → PESTO → pYIN (RMVPE übersprungen) | FCPE → RMVPE → PESTO → pYIN: `get_rmvpe_plugin()` als Tier-2 (§4.4) |
| Lautheitsmessung ohne ISO 532-1 | `np.mean(audio**2)` oder LUFS-only nach Rumble/Multiband-Phasen | `compute_specific_loudness_zwicker(audio, sr)` → ΔN > 2.0 sone = FAIL (§4.1b) |
| JND-blinde PMGG-Phase-Akzeptanz | Alle Deltas > 0 und < JND → identisch zu signifikant positiver Phase | `JND_MIN_DELTA` Dict: wenn alle Deltas ≥ 0 UND alle < JND → `sub_threshold` (§2.47b) |
| Uniforme Goal-Gewichtung | Alle 14 Goals gleich gewichtet | `estimate_goal_importance()` → Per-Song-Profil → `goal_weights` in PMGG/CIG/GPP/FC (§2.56) |
| §2.56 nur in Gates nutzen | `goal_weights` ausschließlich für PMGG/CIG | UV3 `_profiled_phase_call`: `_compute_harmonic_adaptation_scalar(...)` advisory-only auf implizite `strength` + wet/dry (§2.56a) |
| Phase-50 Spike-Detection ohne HF-Guard | `_repair_channel()` flaggt restaurierte analoge Harmoniken als Codec-Spikes | `_hf_protected_bin_start = material_rolloff × 0.85 / bin_hz` für analoge Materialtypen (§2.57) |
| PMGG Goal-Scoring bei Passthrough-Audio | Unverändert zurückgegebenes Audio trotzdem 3× CREPE/pYIN-Retries | `np.array_equal(input, output)` → kein Scoring, kein Retry (§2.58) |
| Phase-09 Stub-Interpolation | `_interpolate_hybrid()` ruft `_interpolate_linear()` auf — kein AR-Verhalten | Vollständige LPC/AR-Vorhersage: Vorwärts+Rückwärts-AR, linear übergeblendet (Rabiner & Schafer 1978, §2.57) |
| Phase-50 lineare Zeit-Interpolation | Dropout-Frames einmalig linear interpoliert | Iterative STFT-Konsistenz-Projektion (5 Iter., POCS-Schema, Siedenburg & Dörfler 2013, §2.57) |
| Phase-23 Inpainting ohne POCS | Direkte PGHI aus inkonsistenten Spektren → Aliasing an Defektgrenzen | POCS-Schleife VOR PGHI in `_repair_channel` (material-adaptiv n_iter=2–5, §4.7c) |
| `signal.lfilter` in Vocal Bell-EQ | `lfilter` in `_boost_presence`/`_enhance_chest` (phase_42) → Phasenverschiebung auf Transients | `signal.filtfilt` (zero-phase); Short-Signal-Fallback: `if len(audio) >= 9: filtfilt(...)` |
| Festes `breath_preservation=0.70` für alle Altersgruppen | Senior/Mature-Stimmen erhalten aggressive Atemreduktion → Stimmidentitätsverlust | `_AGE_ADAPTIVE_FACTORS`: Senior=0.90, Mature=0.82, Adult=0.72, YoungAdult=0.70; GenderDetector.detect() → age_group (§VoiceAge) |
| CausalDefectReasoner einseitige Tabellen [V12] | Neue Ursache nur in `CAUSE_TO_PHASES`, nicht in `CAUSES`/`LIKELIHOOD_FNS` | `CAUSES` + `CAUSE_TO_PHASES` bidirektional konsistent — Bayes-Loop iteriert ausschließlich `CAUSES` (§2.59). Linter V12 prüft automatisch. |
| QualityGate SNR/STFT vor Musical-Goal-Check | `_check_audio_array` vor Musical-Goals-Failures | `_check_musical_goals()` zuerst; bei Failure sofort `return` — teure STFT-Analyse nur wenn Goals bestanden |
| TFS-Guard Hilbert vor Voiced-Gate | Hilbert-Phasenextraktion für alle 12 ERB-Bänder vor Voiced-Energy-Gate | Frame-Energie zuerst prüfen; Bänder mit < 3 Voiced-Frames überspringen vor `filtfilt` + Hilbert |
| AudioSR ohne Wall-Time-Budget | AudioSR-Zonen-Schleife zeitlich unbegrenzt | `_AUDIOSR_WALL_BUDGET_S = 900.0`; Zonen jenseits Budget als Passthrough |
| ADMM-Declipping festes `max_iter=200` | 200 Iter. × 12 s/Iter. @ 225 s Vinyl = 2460 s → Wall-Time erschöpft | `clamp(round(200 × min(1.0, 30.0/duration_s)), 30, 200)` + Wall-Time-Guard (Záviška 2021, §4.5a) |
| Stereo-Kanal-Slicing `a[0]` | `a[0]` wenn `a.ndim==2` → gibt erste Zeitzeile zurück, nicht Kanal 0 | `a[:, 0]` für Kanal-0; `_normalize_audio()` für unklare Orientierung |
| ONNX Fixed-Shape-Input ohne Chunking | `session.run()` mit Audio-Länge ≠ `inp.shape[1]` → INVALID_ARGUMENT | `inp.shape[1]` prüfen; Chunking-Loop mit Zero-Padding für letzten Chunk |
| PlateauStop mit festen Konstanten | `_PLATEAU_THRESHOLD = 0.005`, `_PLATEAU_DAMPEN = 0.40` universal | `_compute_plateau_params(material_type)`: Shellac 0.002/0.55, Tape 0.003/0.50, MP3 0.008/0.40 |
| O(n²)-Autokorrelation im DSP-Fallback | `np.correlate(signal, signal, mode="full")` | `np.array([np.dot(s[:n-k], s[k:]) for k in range(AR_ORDER+1)])` — O(n·order) |
| `np.corrcoef` auf nahezu-konstanten Signalen | `np.corrcoef(a, b)` → RuntimeWarning bei near-constant | Guarded: `dot(a,b) / (‖a‖·‖b‖ + ε)` — NaN-safe |
| `scipy.signal.stft` boundary='reflect' | `boundary='reflect'` → ValueError in scipy < 1.12 | `boundary='even'` |
| `load_audio_file()` mit synchroner Carrier-Analyse | Synchroner 6+-Minuten-Block im BatchProcessingThread | `load_audio_file(path, do_carrier_analysis=False)`; Carrier-Analyse in `_carrier_bg`-Thread |
| QualityMode-Vergleich als roher String | `if quality_mode in ("restoration", "balanced")` statt Enum | `if quality_mode in (QualityMode.RESTORATION, QualityMode.QUALITY, ...)` |
| ML-Budget-Größe als MB statt GB | `try_allocate("Plugin", 630)` wenn 630 MB gemeint → `required 630000 MB` | Einheit ist immer GB (float): `630 MB → 0.63` |
| Unit-Tests Budget-Logik ohne `is_system_thrashing`-Mock | Schlagen auf Hosts mit hoher Swap-Auslastung fehl | `monkeypatch(budget.is_system_thrashing, lambda: False)` |
| Tonträgerketten-Mapping inline duplizieren | `_MEDIUM_DATA` lokal in Methoden/Callbacks definieren | `_CARRIER_MEDIUM_DISPLAY` (Modul-Level); `_render_carrier_html()`; `_build_carrier_chain_html()` |
| `chain_info.get("transfer_chain")` als ersten Key | `kette.as_dict()` liefert ausschließlich `"chain"`, nicht `"transfer_chain"` | `_chain_info.get("chain")` direkt |
| `detected_medium_label.setText(...)` ohne `_carrier_bg_label`-Sync | Era-Badge-Update überschreibt Kettenanzeige | Jedes `.setText(html)` MUSS unmittelbar gefolgt von `self._carrier_bg_label = html` sein |
| Kettenanzeige-Update bei len < 2 ohne Logging | Stilles Überspringen; Anzeige bleibt bei Voranalyse-Wert | `logger.debug("Kettenanzeige übersprungen – len=%d < 2 (chain=%s)", len(chain_keys), chain_keys)` |
| Icon-HTML ohne Plaintext-Fallback | `f'<img src="file:///{path}"...'` ohne None-Check → kaputtes Bild-Tag | `_render_carrier_html()`: prüft `_svg` → `_png` → `return label`; `except (OSError, TypeError, ValueError)` |
| MDEM Quiet-Zone mit fester Amplitude | `if _tail_rms < 0.003` / `MIN_LEVEL_LUFS = -60.0` | `if _tail_rms_dbfs < -36.0`; clamp G[k] ≤ 0 (kein positiver Boost im Quiet-Zone) |
| `apply_musical_gain_envelope` mit `gate_dbfs=-50.0` | −50 dBFS Gate → Vinyl/Shellac-Rauschboden erhält Makeup-Gain → Pegelexplosion (bestätigt 2026-04-25) | **Immer `gate_dbfs=-36.0`** — niemals -50.0 als Gain-Gate-Argument |
| Makeup-Gain-Guard in HPF/Notch-Phasen | Per-Phase-RMS-Guard in phase_05/phase_02 → scheinbarer RMS-Drop → Pegelexplosion | Kein per-Phase-Guard in HPF/LPF/Notch/Bandpass. UV3-Cumulative-Guard (§2.45a-IV) übernimmt |
| HPF/Notch-Phase ohne `enable_loudness=False` | `_phase_overrides` fehlt → `enable_loudness=True`-Default → Pegelexplosion | 4-stufige Checkliste: (1) kein Guard in Phase-Datei; (2) `_phase_overrides: enable_loudness=False`; (3) `_HPF_NOTCH_CUM_RESET_PHASES`; (4) `_update_positive_makeup_authority` |
| Single-Gain-Authority fehlt bei neuer HPF-Phase | `_update_positive_makeup_authority` fehlt → positiver Makeup-Gain auf Folge-Phasen | `_update_positive_makeup_authority` MUSS jede HPF/Notch-Phase-ID enthalten. Test: `TestSingleGainAuthorityPolicy` |
| Gain-Morphing ohne Post-Smoothing-Quiet-Zone-Clamp | SG verteilt positiven Gain aus Musik-Segmenten zurück → Pegelexplosion in Fadeout | **Drei-Stufen-Invariante §2.30b**: Pre-SG-Guard → SG → Post-SG-Guard (Pflicht) → `np.interp` → Per-Sample-Guard −36 dBFS |
| PMGG fixer 60/40-Blend canonical/SGT | Shellac `brillanz` 0.71 obwohl physikalisches Ceiling 0.51 | Delta-adaptiver Blend: `delta > 0.10` → SGT direkt; `delta > 0.04` → 40/60; sonst 60/40 (§09.2) |
| Headroom-Scalar mit relativer Normierung | `hr_range = ceil − 0.30` → Over-Dampening | Absoluter Headroom: `hr_ratio = min(1.0, (ceil−curr) / 0.25)` (§2.54) |
| Wall-Time-Referenz Mismatch | `start_phase()` liefert `perf_counter()`, Accumulator nutzt `monotonic()` → Epochen divergieren → Pipeline übersprungen → Pegelexplosion (bestätigt 2026-04-23) | Ausschließlich `time.monotonic()` für beide Seiten des Akkumulators |
| PMGG waerme Proxy-Sättigung | `_e_low_mid / _e_upper_mid / 1.5` → Clip auf 1.0 bei jeder Phase → PMGG blind (bestätigt 2026-04-24) | `/ 4.0` statt `/ 1.5` — Kalibrierung: warmes Musik-Ratio 3.0–4.0 → Proxy 0.75–1.0 |
| MUSHRA-Referenz CCR-kontaminiert | `_mushra_ref_src = original_audio_for_goals` wenn CCR-Shift aktiv → MUSHRA bewertet Enhancement als Degradation (bestätigt 2026-04-24) | Bei aktivem CCR-Shift: `_mushra_ref_src = audio` (degradierter Input) |
| `TonalCenterMetric._KEY_SHIFT_PENALTY_DEFAULT = 0.0` | tonal_center = 0.000 nach phase_23/06 — BW-Extension verschiebt Tonhöhe ohne echten Tonartwechsel (bestätigt 2026-04-25/26) | `_KEY_SHIFT_PENALTY_DEFAULT = 0.20`; Bypass-Guard `corr_score >= 0.60` |
| `GrooveMetric` DTW einseitiger Onset-Ratio-Guard | `n_restored >> n_original` durch Restaurierungs-Artefakte → DTW = 0.000 (bestätigt 2026-04-26) | Bidirektionaler Guard: `_restore_onset_ratio > 1.5 AND dtw < 0.3` → IOI-Fallback |
| `AuthentizitaetMetric._formant_threshold = max(500, ref*0.5)` | BW-Extension hebt Centroid → Formant-Stability = 0.0 → PMGG-Rollback (bestätigt 2026-04-25) | `_formant_threshold = max(1200.0, mean_ref_centroid * 1.5)` |
| `phase_18_noise_gate` ohne PMGG/CIG-Exclusion für `artikulation` | Note-Attack-Unterdrückung senkt artikulation-Proxy um 0.29 → Rollback → Rauschen verbleibt | `{"artikulation", "groove"}` in BEIDEN Tabellen: `PMGG.PHASE_GOAL_EXCLUSIONS` + `CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS` |
| `adaptive_thresholds` ohne Material-Ceiling | PMGG-blended Threshold 0.90 für Vinyl → lehnt physikalisch korrektes Ergebnis ab | `_ADAPTIVE_THR_MATERIAL_CEILING`: Vinyl natuerlichkeit≤0.82, Shellac≤0.68, Tape≤0.78 |
| End-of-Pipeline-Rescue mit integriertem LUFS-Delta | Uniformer Gain auf gesamtes Signal → 12 dB auf Intro/Outro/Fadeout (bestätigt UV3 2026-04-25) | Music-gated: `_adaptive_gate = clip(P5(original)+6.0, -48, -18)` dBFS; `apply_musical_gain_envelope(gate_dbfs=_adaptive_gate)` |
| §2.30c WPG fehlt oder kann positiven Gain auslösen | Keine finale Fangschicht nach MDEM/correct_arc → Pegelexplosion unkontrolliert im Export; `np.interp` erzeugt positive Übergangswerte | `apply_waveform_plausibility_guard()` MUSS nach correct_arc. WPG-Invarianten: (1) `gain_db_interp = np.minimum(gain_db_interp, 0.0)` — NIE Boost |
| MDEM ohne `frisson_zones` | MDEM dämpft Gänsehaut-Momente auf `−frame_max` LU | `get_frisson_detector().detect(original, sr)` → `frisson_zones` VOR MDEM-Aufruf; Zwei-Stufen-Invariante Pre+Post-SG |
| §C10 `SongGoalFeedbackStore` ohne EMA-Blend | Listener-Feedback gespeichert aber nicht in `estimate_goal_importance()` eingeblendet | `_nudges = get_feedback_store().get_nudges()`; 15 %-Blend in Stufe 7 (§C10) |

---

## Linter-Referenz

| Code | Scope | Regel |
| --- | --- | --- |
| V01 | `backend/`, `plugins/` | `print(` → ERROR |
| V02 | `backend/`, `plugins/` | `sf.read(` / `librosa.load(` → ERROR |
| V03 | `plugins/` | `map_location="cuda"` → ERROR |
| V04 | `backend/core/` | `gate_dbfs=-36.0` ohne `reference_for_gate` → ERROR |
| V05 | `backend/core/phases/` | `griffinlim(` als letzter ISTFT → ERROR |
| V08 | `backend/`, `plugins/` | `np.max(np.abs(audio))` in Gain-Pfad → WARNING |
| V09 | `backend/core/` | `consecutive_rollbacks +=` in Carrier-Repair-Phase → ERROR |
| V11 | `backend/core/phases/` | `sosfilt(` + `+=` auf selben Signal → WARNING |
| V12 | `backend/core/causal_defect_reasoner.py` | CAUSE_TO_PHASES-Schlüssel ohne CAUSES-Gegenstück oder CAUSES-Eintrag ohne C2P-Eintrag → ERROR |

> Vollständige Linter-Implementierung: `scripts/aurik_verboten_linter.py`
