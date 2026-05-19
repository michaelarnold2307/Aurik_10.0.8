---
applyTo: "backend/core/unified_restorer_v3.py"
---

# UV3 — Pipeline-Regeln (normativ, Aurik 9.12.x)

## §2.44 Holistic Perceptual Index (HPI) — letztes Export-Gate

```python
# KANONISCH — Restoration (Instrumental):
HPI = MERT_similarity * timbral_fidelity * artifact_freedom * emotional_arc_preservation

# KANONISCH — Restoration (Vokal, panns_singing >= 0.35) [§0p Pflicht]:
HPI = MERT_similarity * timbral_fidelity * VQI * artifact_freedom * emotional_arc_preservation

# KANONISCH — Studio 2026 (Instrumental):
HPI = studio_quality_gain * PQS_improvement * artifact_freedom * emotional_arc_preservation

# KANONISCH — Studio 2026 (Vokal, panns_singing >= 0.35) [§0p Pflicht]:
HPI = studio_quality_gain * PQS_improvement * VQI * artifact_freedom * emotional_arc_preservation

# MERT-Floor PFLICHT (Bug-Fix v9.12.0):
MERT_similarity = max(raw_mert, 0.5)  # verhindert Gesamt-Kollaps auf 0

# carrier_chain_recovery_ratio — KANONISCHE BERECHNUNG (Pflicht vor HPI):
# = Anteil der Phasen 1-4 (Carrier-Stufen), deren phase_score > 0.05:
# n_active_carrier = len([p for p in carrier_phases if metadata["phase_scores"][p] > 0.05])
# n_total_carrier = len(carrier_phases)  # phase_04/09/12/29 etc.
# carrier_chain_recovery_ratio = n_active_carrier / max(n_total_carrier, 1)
# MUSS in metadata["carrier_chain_recovery_ratio"] gesetzt werden (UV3, nach Stufe 4)

# studio_quality_gain — Studio 2026 Qualitätsfaktor:
# = VERSA(composite_score_restored) / VERSA(composite_score_input)  — normiert auf [0, 1]
# > 1.0 → geklippt auf 1.0 (Gain kann nicht > 1 sein im normierten Score)
# studio_quality_gain = min(versa_restored / max(versa_input, 0.01), 1.0)

# PQS_improvement — Perceptual Quality Score-Steigerung (Studio 2026):
# = DNSMOS(restored)["ovr"] / DNSMOS(input)["ovr"] — normiert auf [0, 1]
# PQS_improvement = min(dnsmos_ovr_restored / max(dnsmos_ovr_input, 1.0), 1.0)
# Beide Faktoren nur für Studio 2026 berechnen — nicht in Restoration!

# Primärer Veto-Faktor (artifact_freedom) + Recovery-Trigger (VQI):
if artifact_freedom < 0.95:
    return _recovery_cascade("artifact_freedom < 0.95", audio)  # KEIN Export
# material_vqi_floor: material-adaptiv aus calibration_matrix — VERBOTEN: hardcodierte Konstante
# material_vqi_floor = calibration_matrix.get_material_floor(material, "vqi")
# Shellac: 0.62 | Vinyl: 0.72 | CD/Digital: 0.82 | unknown_analog: 0.72
if panns_singing >= 0.35 and vqi < material_vqi_floor:   # VERBOTEN: vqi < 0.72 hardcoded
    return _recovery_cascade("vqi < material_floor", audio)  # Recovery-Trigger
**VERSA-Primärpflicht**: `use_versa_in_loop=True`. MERT nur Fallback → `metadata["mert_proxy_used"] = True`.

## §0d Vollständiges Referenz-Paradoxon-Handling (Lücke 1 — ALLE HPI-Faktoren)

```python
# Bei Carrier-Chain-Inversion klingt das restaurierte Signal INTENTIONAL anders
# als der degradierte Input → Ähnlichkeit gegen den Input ist kein Qualitätsmaß!

if carrier_chain_recovery_ratio > 0.15:
    # ALLE drei HPI-Referenzwerte auf best_carrier_checkpoint umstellen:
    # 1. timbral_fidelity → spektraler Vergleich gegen best_carrier_checkpoint
    timbral_fidelity = compute_timbral_fidelity(best_carrier_checkpoint, restored)

    # 2. MERT_similarity → MERT(best_carrier_checkpoint, restored)
    #    VERBOTEN: MERT(degraded_input, restored) — würde gute Restaurierung bestrafen
    raw_mert = mert_similarity(best_carrier_checkpoint, restored)
    MERT_similarity = max(raw_mert, 0.5)

    # 3. emotional_arc → Envelope-Korrelation gegen best_carrier_checkpoint
    emo_arc = compute_emotional_arc_preservation(
        best_carrier_checkpoint, restored, sr, frisson_zones=frisson_zones
    )

    metadata["hpi_reference"] = "best_carrier_checkpoint"
else:
    metadata["hpi_reference"] = "degraded_input"  # Standard-Pfad

# VERBOTEN: degraded_input als Referenz wenn carrier_chain_recovery_ratio > 0.15
```

## §2.45b Hochrestorabilität-Gate

```python
if restorability_score > 80 and snr_db > 40:
    metadata["high_restorability_gate"] = True
    # Phasen mit defect_severity < 0.05 → überspringen
    # Strength für _NEVER_SKIP-Phasen auf Restorability-adaptiven Minimalwert senken
```

## §2.47a PreAnalysis-Handover + MediumDetector-Konfidenz-Fallback (Lücke 4)

- `run_pre_analysis()` **genau 1×** nach Import
- `MediumDetector.detect()` **genau 1×** — nie nochmals auf restauriertem Audio
- Neuer File-Import → Cache **HARD** löschen

```python
# MediumDetector Konfidenz-Schwellen (RELEASE_MUST):
detect_result = MediumDetector.detect(audio, sr)
material_confidence = detect_result.confidence  # [0, 1]

if material_confidence >= 0.75:
    material = detect_result.material            # volles material-adaptives Processing
elif material_confidence >= 0.50:
    # Konservative Böden: Vinyl als Fallback-Material
    material = "vinyl"  # weniger aggressiv als Shellac-spezifisch
    metadata["material_fallback"] = f"confidence={material_confidence:.2f} → vinyl"
else:
    # Zu unsicher für material-adaptives Processing
    material = "unknown_analog"  # Universal-Fallback
    # Universal-Fallback-Defaults (sicher für alle analogen Träger):
    #   BW-Ceiling: 16 kHz
    #   DR-Ceiling: 70 dB
    #   VQI-Boden:  0.72 (Vinyl-Niveau)
    metadata["material_fallback"] = f"confidence={material_confidence:.2f} → unknown_analog"
    metadata["material_low_confidence_warning"] = True

# IMMER in metadata schreiben:
metadata["material_detected"] = detect_result.material
metadata["material_confidence"] = material_confidence
metadata["material_used"] = material  # kann von detected abweichen
```

## §2.48 Kumulative-Phasen-Interaktions-Guard

```python
# VERBOTEN: feste Konstante
tolerance = 0.15  # FALSCH

# RICHTIG:
tolerance = compute_adaptive_drift_tolerance(
    restorability, material, severity, n_phases
)
# Carrier-Repair-Phasen (_CARRIER_REPAIR_PHASE_PREFIXES) inkrementieren
# consecutive_rollbacks NICHT

# KANONISCHE DEFINITION _CARRIER_REPAIR_PHASE_PREFIXES (alle Stufen 1–4 aus §2.46):
_CARRIER_REPAIR_PHASE_PREFIXES = {
    "phase_30",   # Stufe 1: DC-Offset
    "phase_31",   # Stufe 1: Quantisierungsrauschen
    "phase_04",   # Stufe 2: RIAA-EQ
    "phase_25",   # Stufe 2: Azimuth-Korrektur
    "phase_12",   # Stufe 2: Wow/Flutter
    "phase_09",   # Stufe 3: Crackle/Knistern
    "phase_24",   # Stufe 3: Dropout-Repair
    "phase_03",   # Stufe 4: Surface-Noise-NR
    "phase_29",   # Stufe 4: Bandrauschen/Tape-NR
}
# VERBOTEN: phase_05/06/07/23 (Stufe 5 — ADDITIV) in dieses Set — sie inkrementieren consecutive_rollbacks
```

## §2.49 Artefakt-Freiheits-Gate

```python
# artifact_freedom = 1.0 - (weighted_sum / _max_tolerance) + penalties — KEIN Pipeline-Delta
# Gewichtete Summe der Komponenten-Scores (alle müssen < Toleranz bleiben):
# Implementierung: artifact_freedom_gate.py (NICHT min()-Formel — weighted-sum ist präziser)
#
# 1. musical_noise_score: Bins wo restored > orig * 1.05 (5 % Überschuss)
#    → STFT-Maskenvergleich; Bereich [0,1], 1.0 = kein Musical Noise
# 2. phase_cancellation_score: Anti-Phasigkeit L+R oder Original vs. Restored
#    → Kreuz-Korrelation < -0.5 über > 10 % der Frames = Fehler
#    → Frames die im Input bereits anti-phasig waren → NICHT flaggen
# 3. ringing_score: Pre/Post-Transient-Energie außerhalb physikalischer Grenzen
#    → STFT Energie 10 ms vor und nach Onset vs. Original-Onset-Energie
# 4. modulation_noise_score: periodische Modulation in Residual (restored - orig)
#    → FFT des Residuals auf Periodizität prüfen (Kamm-Spektrum = Fehler)
# 5. timbre_distortion_score: spektrales Zentroid-Shift > 15 % oder MFCC-Euklid > 0.2
#
# artifact_freedom = 1.0 - (weighted_sum / _max_tolerance) + penalties
# → Gewichtete Summe aller Komponenten-Beiträge; artifact_freedom < 0.95 = Gate-Fail
# Jede Komponente trägt nach Gewicht bei — ein schlechter Einzelwert dominiert die Summe
```

## VERSA — Primäre Qualitäts-Metrik (`use_versa_in_loop=True`)

```python
# VERSA (Versatile Evaluation for Speech and Audio Restoration, 2024):
# Multi-dimensionale Metrik für Audio-Restaurierung:
# - MOS-Prediction (UTMOS-basiert): subjektive Qualität [1,5]
# - Naturalness: DNSMOS-P.835 SIG-Komponente
# - Fidelität: SI-SDR und LSD (Log-Spectral Distance) gegen Referenz
# - Artifact-Freiheit: Kreuz-Spektral-Analyse auf Musical Noise
# Ausgabe: composite_score [0,1] = gewichtetes Mittel aller Komponenten
# VERSA > MERT für Restoration (MERT ist Music-Representation — kein Restaurierungsmaß)!
from backend.core.dsp.quality_predictors import get_versa_predictor
versa_result = get_versa_predictor().evaluate(audio_orig, audio_restored, sr)
versa_score = versa_result["composite_score"]  # [0, 1]

# MERT (MERT-95M Backbone, Li et al. 2023) — nur Fallback wenn VERSA OOM:
# MERT misst semantische Ähnlichkeit im Music-Representation-Space (CQT-Embeddings)
# Cosine-Similarity zwischen Original- und Restaurierungs-Embeddings
# Fallback: metadata["mert_proxy_used"] = True; MERT_floor = max(raw_mert, 0.5)
```

## emotional_arc_preservation — Messung

```python
# Emotional Arc = Energie-/Dynamik-Verlauf über Stück-Zeitachse:
# 1. RMS-Envelope (200 ms Fenster, 50 % Overlap) für Original + Restored
# 2. Pearson-Korrelation der Envelopes r ∈ [-1, 1]
# 3. Normierung: emotional_arc = max(0, r)  (negative Korrelation = 0)
# 4. Frisson-Zonen: innerhalb Frisson-Segmente Gewicht × 2.0 (Klimax-Passagen
#    prägen Emotionswahrnehmung stärker als ruhige Strophen)
# 5. Pegelexplosion-Detektion: wenn max(restored) > max(original) * 1.5 → score = 0
from backend.core.dsp.emotional_arc import compute_emotional_arc_preservation
emo_arc = compute_emotional_arc_preservation(audio_orig, audio_restored, sr,
                                              frisson_zones=frisson_zones)
# VERBOTEN: emotional_arc aus Spektral-Merkmalen approximieren (Envelope ist Ground Truth)
```

## §2.51 Stereo — Hard-Fail-Invariante

```python
# §2.51a — drei Hard-Fails, alle sofort Recovery-Kaskade:
# VERBOTEN: assert (durch python -O deaktivierbar!)
# RICHTIG: if + logger.error + _recovery_cascade()
if interchannel_delay_ms > 1.0:
    logger.error("stereo_hard_fail interchannel_delay=%.2fms", interchannel_delay_ms)
    return _recovery_cascade("stereo_interchannel_delay", audio)
if lr_imbalance_db > 6.0:
    logger.error("stereo_hard_fail lr_imbalance=%.2fdB", lr_imbalance_db)
    return _recovery_cascade("stereo_lr_imbalance", audio)
if true_peak_dbtp > -1.0:
    logger.error("stereo_hard_fail true_peak=%.2fdBTP", true_peak_dbtp)
    return _recovery_cascade("stereo_true_peak", audio)

# VERBOTEN: unabhängiges L/R-Processing
# RICHTIG: M/S-Domain oder Linked-Stereo überall
```

## §2.52 PhaseConductor — _NEVER_SKIP

```python
_NEVER_SKIP = {
    "phase_01", "phase_09", "phase_12",
    "phase_14", "phase_15",
    "phase_30",  # DC-Offset — immer
    "phase_47",  # TruePeak-Limiter — immer
}
# Diese Phasen laufen immer — auch bei hoher Restorability und bei MAS-Early-Stop
```

## §2.53b Determinismus

```python
# precomputed_phase_plan ist Source of Truth
# UV3 überspringt _select_phases() + _optimize_phase_plan_intelligence()
# wenn precomputed_phase_plan vorhanden
```

## §2.60 Rollback-Hierarchie + `_recovery_cascade()` Vollspezifikation (Lücke 2)

```python
# _recovery_cascade() — vollständig spezifiziert, KEINE ad-hoc-Implementierung erlaubt:
#
# Aufruf-Kontext: HolisticPerceptualGate oder Phase-Delta-Guard
# Rückgabe: np.ndarray (bestes verfügbares Audio) — NIEMALS None
# Max-Gesamtzeit: 30 s für den gesamten Kaskaden-Durchlauf (Watchdog)
# Max-Retries pro Stufe: 2 (danach zur nächsten Stufe)

def _recovery_cascade(reason: str, audio_current: np.ndarray) -> np.ndarray:
    # audio_current: Audio zum Aufrufzeitpunkt — NUR für Diagnose/Metadata gespeichert.
    # Rollback-Quellen sind die internen UV3-Checkpoints (s.u.), NICHT audio_current.
    metadata["recovery_audio_snapshot_shape"] = audio_current.shape  # Diagnose, kein Rollback-Ziel
    metadata["recovery_attempts"] = metadata.get("recovery_attempts", 0) + 1
    metadata["recovery_reasons"].append(reason)  # Liste, nie überschreiben

    # Hilfsfunktionen (UV3-private, Implementierung in unified_restorer_v3.py):
    # _validate_audio(a): bool — True wenn a nicht NaN/Inf und shape[-1] > 0
    # _pmgg_check(a): float — PMGG-Score > 0 = Qualität OK; ≤ 0 = weiteres Rollback nötig
    # _can_retry_with_reduced_strength(): bool — True wenn aktuelle Phase retry-fähig ist
    #   (Phase nicht in Blacklist, retry_count < 2, Phase hat einen strength-Parameter)
    # _retry_phase_at_half_strength(): np.ndarray — führt aktuelle Phase mit strength × 0.5 aus

    # Stufe 1: Phase-Rollback + Score negativ markieren
    if _current_phase_audio is not None:
        metadata["phase_scores"][current_phase_id] = -1.0
        audio = _current_phase_input  # auf Phase-Eingang zurück
        if _validate_audio(audio) and _pmgg_check(audio) > 0:
            metadata["recovery_level"] = 1
            metadata["status"] = "recovered"
            return audio

    # Stufe 2: Strength-Reduktion 50 % → erneutes PMGG-Check
    if _can_retry_with_reduced_strength():
        audio = _retry_phase_at_half_strength()
        if _validate_audio(audio) and _pmgg_check(audio) > 0:
            metadata["recovery_level"] = 2
            metadata["status"] = "recovered"
            return audio

    # Stufe 3: best_carrier_checkpoint (nach Carrier-Stufen 1–4, vor Enhancement)
    if _best_carrier_checkpoint is not None:
        metadata["recovery_level"] = 3
        metadata["status"] = "recovered"
        return _best_carrier_checkpoint

    # Stufe 4: Pre-Pipeline-Checkpoint (nach TDP, vor allen Phasen)
    if _pre_pipeline_checkpoint is not None:
        metadata["recovery_level"] = 4
        metadata["status"] = "recovered"
        return _pre_pipeline_checkpoint

    # Stufe 5: Original-Input-Export — IMMER besser als Artefakt
    metadata["recovery_level"] = 5
    metadata["status"] = "degraded"  # KEIN "failed" — degraded = valid output
    metadata["recovery_fail_reason"] = reason
    logger.warning("recovery_cascade: all levels exhausted → degraded export, reason=%s", reason)
    return _original_input  # nie leerer Export, nie None

# VERBOTEN: leerer Export / Abbruch ohne Ausgabe / Export mit bekanntem Artefakt
# VERBOTEN: _recovery_cascade() ohne Watchdog-Timeout (max 30 s)
# VERBOTEN: Stufe überspringen (keine "Shortcut"-Implementierungen)
# VERBOTEN: _recovery_cascade() ohne reason-Parameter aufrufen (Signatur ist fix)
```

## §2.61 Output-Length-Guard

```python
# KANONISCH — nach JEDER Phase in UV3:
# ACHTUNG: shape[-1] statt len() — len() auf 2D-Stereo (2, N) gibt 2 zurück, NICHT N!
_n_in = input_audio.shape[-1]
_n_out = output.shape[-1]
if abs(_n_out - _n_in) > 64:
    logger.error("length_mismatch phase=%s delta=%d", phase_id, _n_out - _n_in)
    output = output[..., :_n_in]  # harter Crop (funktioniert für 1D und 2D)
    metadata["length_corrections"].append(phase_id)
# VERBOTEN: Zero-Padding als primäre Längenkorrektur
```

## §2.64 Per-Phase-Score-Delta (MAS-Konvergenz)

```python
# P1P2_GOALS — KANONISCHE DEFINITION (UV3-Klassenkonstante):
# P1 + P2 Goals (universell, immer aktiv — unabhängig von panns_singing):
P1P2_GOALS = frozenset({
    "natuerlichkeit", "authentizitaet",           # P1
    "tonal_center", "timbre", "artikulation",     # P2
})
# P0-Goals (vocal_quality, formant_fidelity) werden separat via VQI-Gate überwacht (§0p)
# P3–P5-Goals dürfen vorübergehend sinken (kein sofortiger Rollback-Trigger)

# mas_gap — Abstand zum MAS-Ziel (berechnet aus estimate_song_goal_targets()):
# mas_targets = UV3._mas_targets  (gesetzt in SongCalibration, nach estimate_song_goal_targets())
# mas_gap = {g: max(0.0, mas_targets[g] - post[g]) for g in P1P2_GOALS}

# KANONISCH — jede Phase MUSS diesen Rahmen nutzen:
pre_audio = audio.copy()  # PFLICHT: vor phase.process() sichern — für Rollback unten
pre = _fast_goal_snapshot(audio, sr, material)
audio = phase.process(audio, sr)
post = _fast_goal_snapshot(audio, sr, material)
metadata["phase_deltas"][phase_id] = {g: post[g] - pre[g] for g in pre}

# Rollback wenn (AUSNAHME: Carrier-Repair-Phasen dürfen P1/P2 vorübergehend senken):
_is_carrier_repair = phase_id in _CARRIER_REPAIR_PHASE_PREFIXES
if not _is_carrier_repair and any(post[g] - pre[g] < -0.03 for g in P1P2_GOALS):
    audio = pre_audio  # Rollback — pre_audio MUSS oben gesetzt sein (s. oben)

mas_gap = {g: max(0.0, self._mas_targets[g] - post[g]) for g in P1P2_GOALS}
# MAS-Erreicht-Stop:
if all(mas_gap[g] <= 0.02 for g in P1P2_GOALS):
    metadata["mas_achieved_at_phase"] = phase_id
    self._mas_fully_achieved = True  # Flag setzen — KEIN break!
    # VERBOTEN: break hier — _NEVER_SKIP-Phasen müssen trotz MAS weiterlaufen (§0k, §2.52)
    # §2.65 prüft _mas_fully_achieved zu Beginn jeder nächsten Phase und überspringt sie
    # (außer phase_id in _NEVER_SKIP)
```

### §2.64 `_fast_goal_snapshot` — Multi-Segment-Pflicht

```python
# VERBOTEN: Single-Segment-Bias auf Audio-Mitte
spec = fft(mono[N//2: N//2 + frame_size])  # FALSCH

# RICHTIG: 3 Segmente mitteln (25%/50%/75%)
specs = [fft(seg25), fft(seg50), fft(seg75)]
spec = np.mean(specs, axis=0)

# authentizitaet-Proxy: Zentral-Drittel statt Intro
acf_segment = mono[N//3: N//3 + 8192]

# transparenz-Proxy: Vollsignal + SFM-Blend
val = 0.70 * np.log10(p95_full / p05_full + 1e-9) / 4.0 + 0.30 * (1.0 - sfm_avg)
```

## §2.65 MAS-Early-Stop

```python
# VERBOTEN: Pipeline läuft nach _mas_fully_achieved=True weiter
# RICHTIG: UV3-Loop prüft _check_mas_convergence() nach jeder Phase
if self._mas_fully_achieved and phase_id not in _NEVER_SKIP:
    logger.info("MAS erreicht bei Phase %s — Pipeline-Stop", phase_id)
    skipped.append(phase_id)
    continue  # _NEVER_SKIP-Phasen laufen trotz MAS immer durch
```

## §GOAL_BASELINE_CHECK [RELEASE_MUST] (v9.12.7) — Garantierter Goal-Recovery-Pfad

**Problem (CAUSE_TO_PHASES-Lücke)**: Wenn DefectScanner keinen Defekt-Cause für ein Musical Goal findet, trägt `CAUSE_TO_PHASES` keine Recovery-Phasen für dieses Goal ein. Der HolisticPerceptualGate-Blend kann Goals nur durch Rollback in Richtung Original bewegen — er kann ein Goal **nicht über das Original-Niveau heben**. Das bedeutet: Goals, die strukturell unter dem materialadaptiven Floor liegen, aber kein erkennbares Defekt-Signal erzeugen, würden nie erreicht.

**Lösung**: §GOAL_BASELINE_CHECK läuft im UV3 `restore()`-Pfad **nach** GPOptimizer (`selected_phases` ist fertig) und **vor** `_execute_pipeline()`. Er misst alle 14 Goal-Proxies auf dem Eingangssignal via `_fast_goal_snapshot()` (DSP-only, ≤200 ms) und fügt für jedes Goal, das unter `material_floor × 0.95` liegt, die primäre Recovery-Phase in `selected_phases` ein.

```python
# KANONISCH — UV3 restore(), nach SLR, vor _execute_pipeline():
try:
    from backend.core.calibration_matrix import (
        get_goal_recovery_phases, get_material_floor
    )
    _gbc_mat_str = str(material_type.value if material_type else "unknown").lower()
    _gbc_is_studio = self.is_studio_mode()
    _gbc_snapshot = UnifiedRestorerV3._fast_goal_snapshot(audio, sample_rate, _gbc_mat_str)
    if _gbc_snapshot and _applicable_goals:
        _selected_set: set[str] = set(selected_phases)
        for goal, proxy_score in _gbc_snapshot.items():
            if goal not in _applicable_goals:
                continue
            floor = get_material_floor(_gbc_mat_str, goal, is_studio_2026=_gbc_is_studio)
            if proxy_score < floor * 0.95:
                for phase_id in get_goal_recovery_phases(goal, is_studio_2026=_gbc_is_studio):
                    if phase_id not in _selected_set:
                        selected_phases.append(phase_id)
                        _selected_set.add(phase_id)
                        break  # §2.45: nur primäre Recovery-Phase
except Exception as _gbc_exc:
    logger.debug("§GOAL_BASELINE non-blocking: %s", _gbc_exc)
```

**Invarianten** (alle bindend, kein Override):

- **§0a**: `get_goal_recovery_phases(is_studio_2026=False)` gibt niemals `phase_21_exciter`, `phase_35_multiband_compression` oder `phase_42_vocal_enhancement` zurück.
- **§2.45**: Nur die **erste** (primäre) Recovery-Phase wird pro Goal eingefügt — kein Massen-Adding.
- **5 %-Margin**: Auslöse-Schwelle `floor × 0.95`, nicht `floor` — vermeidet False-Positives bei physikalisch limitiertem Material.
- **`_applicable_goals`-Filter**: Nur Goals, die für den aktuellen Kontext gelten (GoalApplicabilityFilter-Ergebnis), werden geprüft.
- **Non-blocking**: `try/except` — jeder Fehler lässt `selected_phases` unverändert.
- **Metadata**: Hinzugefügte Phasen werden in `metadata["goal_baseline_recovery"]` dokumentiert.
- **VERBOTEN**: `_fast_goal_snapshot()` auf restauriertem Audio ausführen — nur auf dem Eingangssignal vor der Pipeline.
- **Reihenfolge** [NORMATIV]: Die Phasen-Liste in `_GOAL_TO_RECOVERY_PHASES_RESTORATION` MUSS §2.46-Carrier-Chain-Hierarchie einhalten — subtraktive/physikalische Korrekturen (Stufen 1–4) stehen VOR additiven Eingriffen (Stufen 5–6). Mechanische Ursachen (Wow/Flutter, EQ-Fehler) MÜSSEN VOR digitalen Korrekturen stehen. **Verboten**: Phasen nach Nummernreihenfolge sortieren.
- **Kausal-Richtung** [NORMATIV]: Primärphase MUSS den Defizit-Vektor invertieren. Für `spatial_depth`-Defizit → Phase muss Raumcues **hinzufügen** (`phase_46_spatial_enhancement`), nie entfernen (`phase_49` ist Kausal-Inversion und verboten als Primärphase für `spatial_depth`). Gleiches Prinzip für alle Goals: niedriger Score = zu wenig dieser Qualität → Enhancement; hoher Score = zu viel → Reduktion.
- **Disk-Validierung** [CI-GUARD]: Alle Phase-IDs in `_GOAL_TO_RECOVERY_PHASES_RESTORATION` und `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS` MÜSSEN gegen existierende `backend/core/phases/phase_*.py`-Dateien validiert sein — Test: `test_get_goal_recovery_phases_all_phase_ids_exist_on_disk()` in `tests/unit/test_calibration_matrix.py`.

**Recovery-Phase-Tabelle** (Quelle: `calibration_matrix.get_goal_recovery_phases()`):

Reihenfolge nach §2.46-Carrier-Chain: subtraktiv vor additiv, mechanisch vor digital, breiteste Wirkung zuerst.

| Goal | Primäre Restoration-Recovery-Phase | Wissenschaftliche Begründung |
|---|---|---|
| timbre_authentizitaet | phase_04_eq_correction | Stufe 2: RIAA/Carrier-EQ-Fehler — physikalische Primärursache |
| natuerlichkeit | phase_03_denoise | Breitband-NR: universell größter Einzelbeitrag zur Natuerlichkeit |
| authentizitaet | phase_09_crackle_removal | Crackle (systematisch) zerstört Authentizität stärker als Einzelklicks |
| tonal_center | phase_12_wow_flutter_fix | Stufe 2: mechanische Rotation **vor** digitalem Pitch — physikalische Ursache zuerst |
| timbre | phase_04_eq_correction | Stufe 2: Carrier-EQ-Korrektur (subtraktiv, kausal) |
| artikulation | phase_08_transient_preservation | Transienten-Hüllkurve ist primärer Träger der Artikulationsklarheit |
| emotionalitaet | phase_26_dynamic_range_expansion | Dynamikkontrast ist der primäre Emotionsträger |
| micro_dynamics | phase_26_dynamic_range_expansion | Überkompression ist Primärursache für Mikrodynamikverlust |
| groove | phase_12_wow_flutter_fix | Unregelemäßige Motorrotation = direkte physikalische Ursache für Timing-Instabilität |
| transparenz | phase_03_denoise | Breitbandrauschen maskiert Details am stärksten |
| waerme | phase_04_eq_correction | 200–600 Hz Tonalbalance (EQ zuerst) ist physikalisches Fundament der Wärme |
| bass_kraft | phase_04_eq_correction | RIAA/EQ-Fehler im Bassbereich als Primärursache |
| separation_fidelity | phase_49_advanced_dereverb | Hall-Bleed zwischen Quellen — WPE am effektivsten |
| brillanz | phase_06_frequency_restoration | BW-Erweiterung (AudioSR): primärer Pfad für verlorenen HF-Inhalt |
| spatial_depth | phase_46_spatial_enhancement | Niedrige Raumtiefe = fehlende Cues → Enhancement; **VERBOTEN phase_49** (entfernt Raumtiefe!) |

> Kanonische Quelle: `backend/core/calibration_matrix.py` → `get_goal_recovery_phases()` + `_GOAL_TO_RECOVERY_PHASES_RESTORATION`. Studio-2026-Extras in `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS`. Tests: `tests/unit/test_calibration_matrix.py` (§09.10-Tests, 8 Testfunktionen).

## §6.2a Material-Pflicht-Phasen

| Material | Pflicht-Phasen (unabhängig von DefectScanner-Score) |
|---|---|
| vinyl | phase_09, phase_12, phase_05 |
| tape / cassette | phase_29, phase_24, phase_06, phase_03 |
| reel_tape | phase_29, phase_24, phase_03, phase_55 |
| shellac | phase_03, phase_06, phase_01 |
| mp3_low | phase_23, phase_03, phase_50 |

> `cassette` → intern immer als `tape` in `_MATERIAL_PRIORITY_PHASES`

## §2.29c Restorative-Baseline-Capping

```python
_RESTORATIVE_PHASES = {
    "phase_02", "phase_03", "phase_09", "phase_18",
    "phase_20", "phase_23", "phase_24", "phase_29", "phase_49"
}
# Für diese Phasen:
effective_before[g] = min(measured_before[g], canonical_threshold[g] + 0.05)
# Enhancement-Phasen: echte scores_before (kein Capping)
```

## §2.29e PMGG Team-Koordination

```python
# UV3 schreibt prior_phase_context nach jeder Phase fort
# Phase_50 nach HF-Restauration (phase_06/07/23):
#   Goal-Exclusion: brillanz, transparenz, timbre
#   Emergency-Retries unterdrückt
# CONFLICT_REGISTRY: get_conflict_phases() aus phase_ontology.py
```

## §2.55 PMGG-CIG-Sync-Invariante

```python
# Bei neuer Phase: BEIDE Tabellen synchron aktualisieren
# CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2
# == PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2
# CI-Test: test_pmgg_cig_sync.py
```

## VQI-Gate (Gesangsmaterial)

```python
# _rollback_last_vocal_phase() — KANONISCHE DEFINITION (UV3-intern):
# UV3 pflegt _vocal_phase_inputs: List[Tuple[str, np.ndarray]] = []
# Jede Phase mit panns_singing >= 0.25: _vocal_phase_inputs.append((phase_id, pre_phase_audio))
# Maximale Stack-Tiefe: 5 (ältere Einträge werden verworfen)

def _rollback_last_vocal_phase(audio_current: np.ndarray) -> np.ndarray:
    if not _vocal_phase_inputs:
        logger.warning("singer_identity_rollback: no vocal phase inputs recorded — no-op")
        return audio_current  # kein Rollback möglich → aktuellen Stand behalten
    phase_id, pre_audio = _vocal_phase_inputs.pop()  # letzten Eintrag entfernen
    logger.info("singer_identity rollback: reverting phase=%s", phase_id)
    metadata["singer_identity_rollback_phase"] = phase_id
    return pre_audio
```

```python
# PFLICHT wenn panns_singing >= 0.35:  (≠ panns_singing_confidence — kanonischer Name: panns_singing)
if panns_singing >= 0.35:
    from backend.core.musical_goals.vocal_quality_index import compute_vqi
    result = compute_vqi(
        audio_orig=original_audio,
        audio_restored=restored_audio,
        sr=sr,
    )
    vqi = result["vqi"]  # float [0, 1]
    metadata["vqi"] = vqi
    metadata["singer_identity_cosine"] = result.get("singer_identity_cosine", 0.85)
    # singer_identity_cosine-Gate (Resemblyzer-Identitäts-Prüfung):
    # NUR aktiv wenn kein Multi-Singer-Track (Duette/Chorprojekte würden falsch-positiv)
    if not metadata.get("multi_singer", False):
        sic = result.get("singer_identity_cosine", 0.85)
        if sic < 0.92:
            logger.warning("singer_identity_cosine=%.3f < 0.92 — rolling back last vocal phase", sic)
            audio_restored = _rollback_last_vocal_phase(audio_restored)
    # Dreistufige Recovery-Kaskade (kein harter Veto, §0p):
    # material_vqi_floor — KANONISCHE BERECHNUNG (VERBOTEN: Konstante 0.72 hardcoden!):
    material_vqi_floor = calibration_matrix.get_material_floor(material, "vqi")
    # Fallback-Werte (falls calibration_matrix nicht verfügbar):
    # {"shellac": 0.62, "vinyl": 0.72, "cd": 0.82, "digital": 0.82, "tape": 0.72, "unknown_analog": 0.72}
    if vqi < material_vqi_floor:        # Shellac: 0.62 | Vinyl: 0.72 | CD: 0.82
        return _recovery_cascade("vqi < material_floor", audio_restored)
    elif mode == "restoration" and vqi < 0.82:
        return _recovery_cascade("vqi < restoration_target", audio_restored)
    elif mode == "studio" and vqi < 0.87:
        return _recovery_cascade("vqi < studio_target", audio_restored)
```

## Per-Phase-Zeitbudgets — Wall-Time-Watchdog (Lücke 5, RELEASE_MUST)

> Gesamtbudget Pipeline: ≤ 240 s. Jede Phase hat ein individuelles Wall-Time-Budget.
> Überschreitung → Rollback auf Phase-Eingang + DSP-Fallback, kein Pipeline-Abbruch.

```python
# Kanonischer Watchdog-Rahmen in UV3 _profiled_phase_call_with_delta():
if elapsed > _PHASE_WALL_TIME_BUDGET[phase_id]:
    logger.error("phase_timeout phase=%s elapsed=%.1fs budget=%.1fs",
                 phase_id, elapsed, _PHASE_WALL_TIME_BUDGET[phase_id])
    audio = pre_phase_audio
    metadata["phase_timeouts"].append(phase_id)
    # _NEVER_SKIP-Phasen: KEIN Blacklisting, Budget 2× (nächster Aufruf gibt mehr Zeit)
    if phase_id in _NEVER_SKIP:
        _PHASE_WALL_TIME_BUDGET[phase_id] *= 2.0  # einmalige Erhöhung, max 3× original
        logger.warning("never_skip_timeout phase=%s — budget doubled, NOT blacklisted", phase_id)
        # Kein continue — _NEVER_SKIP-Phase mit neuem Budget sofort nochmals ausführen:
        audio = phase.process(pre_phase_audio, sr, material_type=material, strength=0.3)
    else:
        # ≥ 3 Timeouts derselben Phase → Session-Blacklist (kein Retry mehr)
        if metadata["phase_timeouts"].count(phase_id) >= 3:
            metadata["phase_blacklist"].add(phase_id)
        continue
```

| Phase | Budget (s) | Begründung |
|---|---|---|
| phase_01 (DC + Clip-Repair) | 4 | leicht, immer aktiv |
| phase_03 (Surface-Noise NR) | 20 | DFN + OMLSA auf vollem Signal |
| phase_04 (RIAA-EQ) | 3 | reines DSP |
| phase_06 (BW-Erweiterung) | 45 | AudioSR-Diffusion — teuerste Phase |
| phase_07 (Harmonik-Extrapolation) | 10 | DSP |
| phase_09 (Crackle/Knistern) | 15 | ML-Transient-Detektion |
| phase_12 (Wow/Flutter) | 12 | PSOLA + Pitch-Track |
| phase_14 (Stereo-Korrektur) | 5 | DSP |
| phase_15 (Loudness-Norm) | 3 | EBUR128, trivial |
| phase_23 (Codec-BW) | 20 | NVSR oder SBR |
| phase_24 (Dropout-Repair) | 8 | ML-Inpainting |
| phase_25 (Azimuth-Korrektur) | 6 | DSP |
| phase_29 (Bandrauschen/Tape-NR) | 25 | SGMSE+ auf Tape-Material |
| phase_30 (DC-Offset) | 2 | trivial |
| phase_31 (Quantisierungs-NR) | 5 | DSP |
| phase_42 (Vokal-Enhancement) | 15 | MIIPHER-Stub + Wiener |
| phase_47 (TruePeak-Limiter) | 2 | trivial |
| phase_49 (Dereverb) | 18 | WPE oder Hybrid-Dereverb |
| phase_50 (MP3-Artefakte) | 12 | Transient-MDCT-Korrektur |
| alle anderen Phasen (Default) | 10 | konservatives Budget |

> `_PHASE_WALL_TIME_BUDGET` als Klassen-Konstante in `backend/core/unified_restorer_v3.py`.
