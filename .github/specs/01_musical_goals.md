# Aurik 9 — Spec 01: 14 Musikalische Ziele

> **Einzige normative Quelle** für alle Goal-Schwellwerte, Prioritäten, Adaptive Thresholds
> und Applicability-Regeln. Alle anderen Dateien **referenzieren** hierher.

---

## §1.2 Die 14 Musikalischen Ziele (Musical Goals) — vollständige Tabelle

Implementiert in `backend/core/musical_goals/musical_goals_metrics.py`,
aufgerufen via `MusicalGoalsChecker.measure_all(audio, sr)`.

| Ziel (Klasse) | Frequenzbereich / Messgröße | Pflicht-Schwellwert | Priorität |
|---|---|---|---|
| **Natürlichkeit** (`NatuerlichkeitMetric`) | Artefaktfreiheit, Rauschen, Klangbild | ≥ **0.90** | **1** |
| **Authentizität** (`AuthentizitaetMetric`) | Voice Identity, spektraler Fingerabdruck | ≥ **0.88** | **1** |
| **Tonales Zentrum** (`TonalCenterMetric`) | Chroma-Korrelation Original↔Restauriert, kein Key-Shift > 0 Cent | ≥ **0.95** | **2** |
| **Timbre-Authentizität** (`TimbralAuthenticityMetric`) | MFCC-Pearson ≥ 0.95, Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 % | ≥ **0.87** | **2** |
| **Artikulation** (`ArticulationMetric`) | Attack-Charakter-Erhalt (Staccato vs. Legato): Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms | ≥ **0.85** | **2** |
| **Emotionalität** (`EmotionalitaetMetric`) | Dynamik, Ausdruck, Modulationstiefe | ≥ **0.87** | **3** |
| **Mikro-Dynamik** (`MicroDynamicsMetric`) | Momentane LUFS-Profil-Korrelation (400 ms-Fenster), Crest-Faktor-Erhalt ≤ 1.5 dB | ≥ **0.92** | **3** |
| **Groove** (`GrooveMetric`) | Mikro-Timing, Swing, Event-Onset-Präzision (DTW ≤ 8 ms RMS) | ≥ **0.88** | **3** |
| **Transparenz** (`TransparenzMetric`) | Klarheit, Trennung der Klangelemente | ≥ **0.89** | **4** |
| **Wärme** (`WaermeMetric`) | Mid-Range-Fülle, 200–2000 Hz | ≥ **0.80** | **4** |
| **Bass-Kraft** (`BassKraftMetric`) | Bassenergie 20–250 Hz + Virtual Pitch (Missing Fundamental, Obertöne 120–500 Hz) | ≥ **0.85** | **4** |
| **Separation-Treue** (`SeparationFidelityMetric`) | SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition | ≥ **0.82** | **4** |
| **Brillanz** (`BrillanzMetric`) | HF-Klarheit, 8–20 kHz — Sparkle & Air | ≥ **0.85** | **5** |
| **Raumtiefe** (`SpatialDepthMetric`) | IACC (Interaural Cross-Correlation, Blauert 1997) + Stereobreite + Phantom-Center-Stabilität; IACC < 0.70 → wahrnehmb. Zusammenbruch | ≥ **0.75** | **5** |

> **Schwellwert-Validierung**: Die Schwellwerte für alle 14 Ziele wurden algorithmisch aus AMRB-Bench­mark­daten (10 Szenarien, Ø OQS-Kalibrierung) abgeleitet. Ein ITU-R BS.1534-3 MUSHRA-Hörertest steht als externe Validierung aus (geplant). Bis zur Validierung gelten die Werte als „best engineering estimate“. Die Schwellwerte dürfen NUR nach dokumentiertem Hörertest geändert werden.

```python
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

checker = MusicalGoalsChecker()
scores = checker.measure_all(audio, sr)  # Dict[str, float]
# Pflicht-Check nach jeder Restaurierung:
assert all(scores[g] >= t for g, t in checker.thresholds.items()), scores
```

**Invariante**: Jede Restaurierungsoperation darf keines dieser 14 Ziele verschlechtern.
Eine Regression in einem Ziel macht das gesamte Feature ungültig.

---

## §2.34 GoalPriorityProtocol — Hierarchie bei Ressourcen-Konflikten

```python
PRIORITY_MAP: dict[str, int] = {
    "natuerlichkeit":        1,   # Rollback bei Verschlechterung
    "authentizitaet":        1,   # Rollback bei Verschlechterung
    "tonal_center":          2,   # Rollback bei Verschlechterung
    "timbre_authentizitaet": 2,   # Rollback bei Verschlechterung
    "artikulation":          2,   # Rollback bei Verschlechterung
    "emotionalitaet":        3,
    "micro_dynamics":        3,
    "groove":                3,
    "transparenz":           4,
    "waerme":                4,
    "bass_kraft":            4,
    "separation_fidelity":   4,
    "brillanz":              5,   # best-effort, kein Misserfolg bei Nichterfüllung
    "spatial_depth":         5,   # best-effort
}
ABORT_PRIORITY_THRESHOLD: int = 2  # Stufe 1+2 verschlechtert → Iteration sofort abbrechen
REGRESSION_EPSILON: float = 0.001
```

**Normative Aufrufstellen**:

```python
# In FeedbackChain.run():
gpp = GoalPriorityProtocol()
abort_result = gpp.should_abort_iteration(scores_before, scores_after)
if abort_result.should_abort:
    best_result = previous_best
    break

# In ExcellenceOptimizer — MOO-Pareto-Konflikt:
conflict_result = gpp.resolve_conflict(goal_a, goal_b, delta_a, delta_b)
# conflict_result.winner = priorisiertes Ziel
```

---

## §2.32 GoalApplicabilityFilter — Physikalisch irrelevante Ziele deaktivieren

```python
ALWAYS_APPLICABLE: frozenset[str] = frozenset({
    "natuerlichkeit", "authentizitaet", "emotionalitaet",
    "transparenz", "timbre_authentizitaet", "artikulation",
})
```

**Deaktivierungs-Regeln:**

| Ziel | Deaktiviert wenn |
|---|---|
| `SpatialDepthMetric` | EraResult.decade ≤ 1950 UND M/S-Korrelation ≥ 0.95 (Mono-Aufnahme) |
| `BrillanzMetric` | Quell-Bandbreite < 8 kHz UND AudioSR nicht geladen |
| `TonalCenterMetric` | Original-SNR < −5 dB ODER MaterialType = WAX_CYLINDER |
| `GrooveMetric` | Dateilänge < 10 s ODER PANNs Percussion confidence < 0.15 |
| `MicroDynamicsMetric` | Dateilänge < 20 s ODER Original-LUFS-Varianz < 0.5 LU |
| `SeparationFidelityMetric` | Mono-Quelle ODER PANNs < 2 Instrumente mit confidence ≥ 0.4 |

Filter läuft EINMAL pro Restaurierung (nach MediumClassifier + EraClassifier).
Inapplicable Goals: im UI grau ausgeblendet, in RestorationResult.goal_applicability gespeichert.

---

## §2.31 AdaptiveGoalThresholds — Material- und ära-adaptive Schwellwerte

**Adaptierungs-Algorithmus (5 Schritte):**

1. **Base-Thresholds** aus MusicalGoalsChecker (Startpunkt)
2. **Material-Prior** (physikalische Bandbreitengrenzen):
   - SHELLAC/WAX_CYLINDER: `brillanz_threshold → min(0.85, bw_hz/20000*0.85+0.20)`, `spatial_depth → 0.30` (Mono)
   - VINYL: `separation_fidelity_threshold → 0.76`
   - DAT/CD_DIGITAL: alle Schwellwerte unverändert
3. **Ära-Prior** (EraClassifier.decade):
   - decade ≤ 1940: `spatial_depth_threshold → 0.30`
   - decade ≤ 1960: `spatial_depth_threshold → 0.55`
   - decade ≥ 1970: alle Spatial-Thresholds Standard
4. **Restorability-Skalierung:**
   - restorability ≥ 70: scale_factor = 1.00
   - restorability 50–69: scale_factor = 0.93
   - restorability 30–49: scale_factor = 0.85
   - restorability < 30: scale_factor = 0.75
5. **Physical Ceiling Clamp**: `adaptive_t = min(adaptive_t, physical_ceiling[goal])`

```python
# MaterialQuality Enum (backend/core/musical_goals/adaptive_goals_system.py):
class MaterialQuality(Enum):
    PRISTINE   = "pristine"    # Studio-Qualität
    EXCELLENT  = "excellent"
    GOOD       = "good"
    FAIR       = "fair"        # MP3 192 kbps
    POOR       = "poor"        # MP3 128 kbps, Cassette
    VERY_POOR  = "very_poor"   # Stark degradiert
    EXTREME    = "extreme"     # Telefon, Walkie-Talkie

# Einstiegspunkt:
from backend.core.musical_goals.adaptive_goals_system import get_adaptive_goals_and_config
thresholds, config, quality_assessment = get_adaptive_goals_and_config(audio, sr)
```

**Invarianten:**
- Adaptierte Schwellwerte NIEMALS höher als Original-Schwellwerte
- Absolute Untergrenze: adaptive_t ≥ 0.50 (unter 0.50 → Goal deaktivieren)
- NaN in restorability_score → alle Schwellwerte auf Original-Werte

**Restorability-Skalierungsfaktoren — Formale Ableitung:**
Die Stufenwerte 1.00 / 0.93 / 0.85 / 0.75 sind aus dem PhysicalCeilingEstimator hergeleitet:
```python
# Formale Herleitung (normativ): scale_factor = ceiling(goal) / baseline_threshold
# Die Stufen approximieren den integralen Ø der Ceiling-Kurven über alle 14 Goals
# pro Restorability-Klasse (gemessen auf 500 AMRB-Testdateien):
# ≥ 70: ceiling_avg = 0.97 → scale = 1.00
# 50–69: ceiling_avg = 0.90 → scale = 0.93
# 30–49: ceiling_avg = 0.82 → scale = 0.85
# <  30: ceiling_avg = 0.73 → scale = 0.75
# Heuristik-Einsatz VERBOTEN — Stufen müssen aus PhysicalCeilingEstimator.ceiling_avg()
# aktualisiert werden, wenn neue AMRB-Szenarien hinzukommen.
```

---

## §2.33 PhysicalCeilingEstimator — Informationstheoretische Qualitätsdecke

**Musical-Goal-Ceiling-Mapping (empirisch aus AMRB-Daten):**

```python
HEADROOM_THRESHOLD: float = 0.03   # Verbesserung < 3 % → keine weiteren Iterationen

# Ceiling-Formeln:
natuerlichkeit_ceiling  = sigmoid((mean(SNR_b) − 5) / 5) × 0.97 + 0.03
brillanz_ceiling        = sigmoid((bw_hz − 8000) / 2000) × 0.95
spatial_depth_ceiling   = sigmoid(stereo_decorrelation × 10) × 0.92
groove_ceiling          = 1 − max(0, wow_flutter_hz − 0.5) × 0.10
tonal_center_ceiling    = sigmoid(snr_tonal_bands × 2) × 0.98
# Alle anderen Goals: 0.98 (konservative Obergrenze)

# FeedbackChain-Terminierung:
# further_optimization_worthwhile = False wenn alle Goals:
#   current_score ≥ ceiling − HEADROOM_THRESHOLD
```

Nutzer-Meldung wenn Decke erreicht (Deutsch):
> „Das Beste aus dieser Aufnahme wurde herausgeholt — die physikalischen Grenzen des Quellmaterials sind erreicht."

---

## §8.2 Perceptuelle Verpflichtungen (vollständig)

1. **Musikalische Natürlichkeit**: MERT-Naturalness-Score ≥ 0.7
   > MERT (Li et al. ICLR 2024) ist ein Music-Understanding-Foundation-Model, kein
   > designierter MOS-Schätzer. `harmonicity` ist ein kalibrierter Proxy-Score.
   > Kalibrierung: Pearson-Korrelation MERT-harmonicity ↔ VERSA-MOS = 0.74 (n=312 Testdateien).
   > Bei VERSA-MOS verfügbar: VERSA hat Vorrang; MERT-Score dient als Schnellprüfung.
2. **Harmonische Kohärenz**: Harmonizitäts-Ratio ≥ 0.85 (via `MertPlugin.analyze().harmonicity`)
3. **Dynamik-Erhalt**: LUFS-Differenz ≤ 1 LU
4. **Transientenerhalt**: Attack-Zeiten ≤ ±2 ms Änderung
5. **Tonale Stabilität**: Chroma-Pearson ≥ 0.95
6. **Groove**: Event-Onset-DTW ≤ 8 ms RMS — kein Begradigen von Swing/Rubato
7. **Pass-Through-Invariante** (SNR > 40 dB): PQS-MOS-Verlust ≤ 0.05, alle 14 Goals ±0.02, LUFS ≤ 0.3 LU, Chroma ≥ 0.99
8. **Rauschboden**: Residual ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events in Stille
9. **Mikro-Dynamik**: Pearson des 400 ms LUFS-Profils ≥ 0.92, Crest-Faktor ≤ 1.5 dB
10. **Vintage Aesthetics** (automatisch via EraClassifier):
    - 1920–1940: Rolloff ≤ 7 kHz nicht künstlich erweitern
    - 1940–1955: Röhren-Kompressions-Fingerabdruck erhalten (H2, H4 ∈ [−30, −20] dBr)
    - 1955–1965: RT60 ∈ [1.2, 2.0] s erhalten (kein aggressives Dereverb)
    - 1965–1975: Tape-Saturation-Signatur nicht entfernen
11. **Kompetitiver Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 AMRB-Szenarien
12. **Emotionaler Dynamik-Bogen**: Arousal-Pearson ≥ 0.85, Valence-Pearson ≥ 0.80, Klimax-Peak-Abw. ≤ 2 Segmente
