# Aurik 9 — Spec 01: 14 Musikalische Ziele

> **Einzige normative Quelle** für alle Goal-Schwellwerte, Prioritäten, Adaptive Thresholds
> und Applicability-Regeln. Alle anderen Dateien **referenzieren** hierher.

## §1.0a [RELEASE_MUST] Universaler Zielraum (All-Import Contract)

Die 14 Musical Goals sind **nicht** auf einen einzelnen Song oder ein Einzelgenre optimiert,
sondern auf den vollständigen Importraum von Aurik (Material × Ära × Genre × Defektschwere).

**Verbindlich:**

1. Goal-Optimierung muss über eine repräsentative Import-Matrix stabil bleiben.
2. Song-spezifische Heuristiken (Dateiname/Artist/Einzelsong-Whitelists) sind verboten.
3. Anpassung von Goal-Schwellwerten oder Retry-Politiken ist nur zulässig, wenn die
    Gesamtqualität über die Matrix nicht regressiert (kein „lokaler Song-Gewinn" bei
    globalem Qualitätsverlust).

Diese Invariante konkretisiert §0 Klangwahrheit für den gesamten Produktbetrieb.

---

## §1.2 Die 14 Musikalischen Ziele (Musical Goals) — vollständige Tabelle

Implementiert in `backend/core/musical_goals/musical_goals_metrics.py`,
aufgerufen via `MusicalGoalsChecker.measure_all(audio, sr)`.

| Ziel (Klasse) | Frequenzbereich / Messgröße | Prio | Restoration | Studio 2026 |
| --- | --- | --- | --- | --- |
| **Natürlichkeit** (`NatuerlichkeitMetric`) | Artefaktfreiheit, Rauschen, Klangbild | **1** | ≥ **0.90** | ≥ **0.90** |
| **Authentizität** (`AuthentizitaetMetric`) | Voice Identity, spektraler Fingerabdruck | **1** | ≥ **0.88** | ≥ **0.88** |
| **Tonales Zentrum** (`TonalCenterMetric`) | Chroma-Korrelation Original↔Restauriert, kein Key-Shift > 0 Cent | **2** | ≥ **0.95** | ≥ **0.97** |
| **Timbre-Authentizität** (`TimbralAuthenticityMetric`) | MFCC-Pearson ≥ 0.95, Spectral-Centroid-Korrelation ≥ 0.93, Rolloff-Abw. ≤ 5 % | **2** | ≥ **0.87** | ≥ **0.87** |
| **Artikulation** (`ArticulationMetric`) | Attack-Charakter-Erhalt (Staccato vs. Legato): Transient-Shape-Korrelation ≥ 0.90, Attack-Time-Abweichung ≤ 10 ms | **2** | ≥ **0.85** | ≥ **0.85** |
| **Emotionalität** (`EmotionalitaetMetric`) | Dynamik, Ausdruck, Modulationstiefe | **3** | ≥ **0.82** | ≥ **0.87** |
| **Mikro-Dynamik** (`MicroDynamicsMetric`) | Momentane LUFS-Profil-Korrelation (400 ms-Fenster), Crest-Faktor-Erhalt ≤ 1.5 dB | **3** | ≥ **0.88** | ≥ **0.92** |
| **Groove** (`GrooveMetric`) | Mikro-Timing, Swing, Event-Onset-Präzision (DTW ≤ 8 ms RMS) | **3** | ≥ **0.83** | ≥ **0.88** |
| **Transparenz** (`TransparenzMetric`) | Klarheit, Trennung der Klangelemente | **4** | ≥ **0.82** | ≥ **0.89** |
| **Wärme** (`WaermeMetric`) | Primär: Even-Harmonic-Ratio (H2/H4 THD_even/THD_total, ISO 226:2023 gewichtet) — misst wahrgenommene Röhren-/Band-Wärme; Sekundär: Warmth Ratio E(200–800)/E(800–3000) als Spektral-Tilt-Proxy (§9.7.14) | **4** | ≥ **0.75** | ≥ **0.80** |
| **Bass-Kraft** (`BassKraftMetric`) | Bassenergie 20–250 Hz + Virtual Pitch (Missing Fundamental, Obertöne 120–500 Hz) | **4** | ≥ **0.78** | ≥ **0.88** |
| **Separation-Treue** (`SeparationFidelityMetric`) | SDR ≥ 8 dB / SIR ≥ 12 dB nach NMF-Dekomposition | **4** | ≥ **0.78** | ≥ **0.85** |
| **Brillanz** (`BrillanzMetric`) | HF-Klarheit, 8–20 kHz — Sparkle & Air | **5** | ≥ **0.78** | ≥ **0.90** |
| **Raumtiefe** (`SpatialDepthMetric`) | IACC (Interaural Cross-Correlation, Blauert 1997) + Stereobreite + Phantom-Center-Stabilität; IACC < 0.70 → wahrnehmb. Zusammenbruch | **5** | ≥ **0.70** | ≥ **0.78** |

> **v9.10.77 Pareto-Differenzierung**: Restoration-Modus senkt P3–P5-Schwellwerte auf physikalisch erreichbare Werte (Pareto-Konflikte: Bass↔Transparenz [0.7], Brillanz↔Wärme [0.6]). P1/P2 bleiben identisch. Studio 2026 behält ambitionierte Ziele.
> **Schwellwert-Validierung**: Die Schwellwerte für alle 14 Ziele wurden algorithmisch aus AMRB-Bench­mark­daten (10 Szenarien, Ø OQS-Kalibrierung) abgeleitet. Ein ITU-R BS.1534-3 MUSHRA-Hörertest steht als externe Validierung aus (geplant). Bis zur Validierung gelten die Werte als „best engineering estimate“. Die Schwellwerte dürfen NUR nach dokumentiertem Hörertest geändert werden.

```python
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

checker = MusicalGoalsChecker(mode="restoration")  # oder "studio_2026"
scores = checker.measure_all(audio, sr)  # Dict[str, float]
# Pflicht-Check nach jeder Restaurierung:
assert all(scores[g] >= t for g, t in checker.thresholds.items()), scores
```

**Invariante (§2.54-konform)**: Am **Pipeline-Ende** müssen alle 14 Ziele ≥ Schwellwert liegen.
Einzelphasen dürfen Proxy-Werte vorübergehend senken, wenn:

- **Carrier-Repair** (§2.44 Referenz-Paradoxon): Tonträgerschaden-Inversion verändert Chroma/Centroid intentional
- **Restorative Defektentfernung** (§2.29c Baseline-Capping): aufgeblähte scores_before durch Defekte
- **Iterative Stärke-Anpassung** (§2.54): Phase wird mit reduzierter Stärke erneut versucht, nicht sofort übersprungen

Eine Regression nach der **gesamten Kette** macht das Feature ungültig.

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
    "brillanz":              5,   # Recovery-Lite: darf Endresultat nicht unkontrolliert regressieren
    "spatial_depth":         5,   # Recovery-Lite: darf Endresultat nicht unkontrolliert regressieren
}
ABORT_PRIORITY_THRESHOLD: int = 2  # Stufe 1+2 verschlechtert → Iteration sofort abbrechen
REGRESSION_EPSILON: float = 0.001

# §2.54: ABORT_PRIORITY_THRESHOLD und REGRESSION_EPSILON sind Notbremsen-Konstanten
# für die FeedbackChain. Sie greifen NUR bei katastrophalen Fällen.
# Die Routine-Steuerung läuft über den iterativen Messen→Handeln→Validieren-Zyklus
# pro Phase (PMGG + PhaseConductor + SongCalibration).
```

**§2.29 Priority-Aware PMGG Retries (v9.10.77)**:

**Ergänzung v9.11.5 (Team-Koordination):**
PMGG-Retry/Strength-Entscheidungen sind kontextbewusst über `prior_phase_context`
zu steuern, damit Folgephasen (insb. `phase_50`) intentionale Vorphasen-Reparaturen
nicht indirekt neutralisieren. Normative Details: Spec 02 §2.29e, Spec 06 §6.9b.

PMGG-Retries werden prioritätsabhängig budgetiert:

| Priorität | Max Retries | Threshold-Faktor | Verhalten |
| --- | --- | --- | --- |
| P1 | 4 | 1.0× | Volle Retry-Kaskade + Emergency |
| P2 | 4 | 1.0× | Volle Retry-Kaskade + Emergency |
| P3 | 2 | 1.5× (mildere Erkennung) | Reduzierte Kaskade, kein Emergency |
| P4 | 1 | 2.0× (Recovery-Lite) | 1 konservativer Retry, kein Emergency |
| P5 | 1 | 2.5× (Recovery-Lite) | 1 konservativer Retry, kein Emergency |

Implementierung: `per_phase_musical_goals_gate.py` — `_PRIORITY_MAX_RETRIES`, `_PRIORITY_THRESHOLD_FACTOR`, `_max_regression_priority_aware()`.

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

# In UnifiedRestorerV3._profiled_phase_call() — §2.56a all-phase coupling:
# goal_weights steuern bounded harmonic_adaptation_scalar (advisory-only),
# der implizite strength/wet-dry für alle Phasen 01-64 harmonisiert.
```

## §2.35 Vocal-Exzellenz-Zusatzmetriken (PFLICHT fuer Gesangsmaterial)

Wenn PANNs/Gender/Vocal-Detektoren Gesang erkennen, werden zusaetzlich zu den 14 Musical Goals folgende Vocal-Zielwerte geprueft:

| Ziel | Messgroesse | Mindestwert |
| --- | --- | --- |
| Formant-Stabilitaet | mittlere Formant-Drift F1/F2 ueber Vokal-Segmente | <= 35 Hz |
| Sibilance-Natuerlichkeit | 5-10 kHz-Energieabweichung in Frikativen | <= 1.5 dB |
| Konsonanten-Klarheit | Plosiv/Frikativ-Onset-Praezision vs. Original | <= 6 ms |

**Invariante:** Vocal-Zusatzmetriken duerfen nie auf Kosten von P1/P2 erzwungen werden.

## §2.35b Vocal-Proximity-Score (PFLICHT bei PANNs Singing ≥ 0.35)

**Psychoakustische Motivation**: Studiogesang klingt, als befände sich der Sänger direkt
vor dem Hörer — durch klare Konsonanten, natürliche Atemgeräusche in Pausen und erhaltene
frühe Reflexionen (C80). Verlust dieser Eigenschaften durch Dereverb oder Over-Enhancement
erzeugt wahrgenommene Distanz, die unmittelbar das Immersionsgefühl (§8.3 Tiefen-Immersion)
zerstört. Dieser Score ist mit bestehenden Vocal-Metrics (§2.35) orthogonal — er misst
räumlich-zeitliche Vertrautheit, nicht spektrale Genauigkeit.

**Formel**:

```python
proximity_score = (
    konsonanten_transient_energy_ratio  # Plosiv/Frikativ-Onsets erhalten
    × breathiness_ratio                 # natürliche Atemgeräusche in Pausen
    × early_reflection_preservation     # Raumcharakter (C80) nicht zerstört
)
```

| Komponente | Messung | Schwelle |
| --- | --- | --- |
| `konsonanten_transient_energy_ratio` | Plosiv/Frikativ-Onset-Energie (Output) / Onset-Energie (Input) | ≥ 0.75 |
| `breathiness_ratio` | RMS in 100 ms-Pausen / Rauschboden (Output) vs. Input-Verhältnis | ≥ 0.60 |
| `early_reflection_preservation` | C80-Verhältnis (Output) vs. (Input): Frühenergie-Anteil ≤ 80 ms | ≥ 0.80 |

**Schwellwerte**:

- `proximity_score ≥ 0.75` → "nah" (Vokalintimität bestanden)
- `proximity_score ∈ [0.70, 0.75)` → WARNING: `metadata["vocal_proximity_warning"]` setzen
- `proximity_score < 0.70` → Dry/Wet-Rescue für Dereverb-Phase empfehlen

**Aktivierungs-Bedingungen**:

- `panns_singing_confidence ≥ 0.35` ODER `vocal_genre = True`
- Nur im finalen Gate-Check (`MusicalGoalsChecker`), nicht im PMGG-Delta

**Invarianten**:

- Vocal-Proximity-Score darf nie auf Kosten von P1/P2 erzwungen werden
- Phase-Rollback nur bei `proximity_score < 0.70` UND gleichzeitig HPI > Rollback-Schwelle
- Protokollierung immer: `RestorationResult.metadata["vocal_proximity_score"]`

> Implementierung: `backend/core/musical_goals/ki_hearing_model.py` — `compute_vocal_proximity_score(audio_orig, audio_restored, sr, vocal_segments)`
> Aufruf: `backend/core/musical_goals/musical_goals_metrics.py` — `MusicalGoalsChecker.measure_all()` wenn Vocal erkannt

## §2.36 Pareto-Tie-Break nach Hoerprioritaet

Bei mehreren Pareto-aequivalenten Kandidaten gilt folgende Tie-Break-Reihenfolge:

1. kleinste P1/P2-Regression,
2. hoechster Vocal-Score (falls Gesang erkannt),
3. geringste Artefaktwahrscheinlichkeit (musical noise, chirps, metallic tails),
4. niedrigere Laufzeit nur, wenn Punkte 1-3 gleichwertig sind.

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
| --- | --- |
| `SpatialDepthMetric` | EraResult.decade ≤ 1950 UND M/S-Korrelation ≥ 0.95 (Mono-Aufnahme) |
| `BrillanzMetric` | Quell-Bandbreite < 8 kHz UND AudioSR nicht geladen |
| `TonalCenterMetric` | MaterialType = WAX_CYLINDER (Fix K, v9.10.100: SNR-Bedingung entfernt — K-S-Key-Detection ist SNR-invariant gemäß §9.7.11; Deaktivierung bei SNR < −5 dB war inkonsistent mit der K-S-Invarianz-Aussage und hätte tonal_center auf stark degradiertem Material blind abgeschaltet) |
| `GrooveMetric` | Dateilänge < 10 s ODER PANNs Percussion confidence < 0.15 |
| `MicroDynamicsMetric` | Dateilänge < 20 s ODER Original-LUFS-Varianz < 0.5 LU |
| `SeparationFidelityMetric` | Mono-Quelle ODER PANNs < 2 Instrumente mit confidence ≥ 0.4 |

Filter läuft EINMAL pro Restaurierung (nach MediumClassifier + EraClassifier).
Inapplicable Goals: im UI grau ausgeblendet, in RestorationResult.goal_applicability gespeichert.

---

## §2.31 AdaptiveGoalThresholds — Material- und ära-adaptive Schwellwerte

> **§2.54 Einordnung**: AdaptiveGoalThresholds definieren die **Ziel-Schwellwerte** für Musical Goals —
> nicht die Guard-Schwellwerte für Phase-Rollback. Die Guard-Drift-Toleranzen werden separat
> über `compute_adaptive_drift_tolerance()` berechnet (§2.48/§2.54). Beide Systeme konsumieren
> Material/Era/Restorability, aber zu unterschiedlichen Zwecken:

- AdaptiveGoalThresholds → „Was ist am Pipeline-Ende für diesen Song erreichbar?"
- Adaptive Drift-Toleranz → „Wie viel darf eine Einzelphase temporär verschlechtern?"

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

### §2.31d Kombinierte Extrembedingungen (v9.10.123)

Wenn **mehrere** erschwerende Faktoren gleichzeitig vorliegen, kaskadieren die Adaptionen:

| Kombination | Zusätzliche Anpassung |
| --- | --- |
| restorability < 20 **+** Material ∈ {SHELLAC, WAX_CYLINDER} | scale_factor = 0.65 statt 0.75; alle P3–P5 Goals → Untergrenze 0.50; Pipeline-Ziel = „Hörbar machen" |
| restorability < 30 **+** Era ≤ 1940 | Vintage-Aesthetics-Schutz verstärken: H2/H4-Preservation-Guard → G_FLOOR_HARMONIC = 0.92; kein Brillanz-Enhancement |
| Material = SHELLAC **+** Era ≤ 1930 **+** BW < 5 kHz | Brillanz-Goal deaktivieren (physikalisch unmöglich); Wärme-Goal als nicht-bindend markieren |
| Dateilänge < 10 s | GrooveMetric + MicroDynamicsMetric + EmotionalArcPreservation deaktivieren; FeedbackChain max 2 Iterationen |
| Dateilänge > 60 min | SegmentAdaptiveProcessor aktivieren; DefectScanner auf 3×60-s-Segmente (Anfang/Mitte/Ende) |

### §2.31e Prior-Konflikt-Auflösung (v9.10.123)

Wenn Ära-Prior und Material-Prior widersprüchliche Anpassungen ergeben:

- **Physikalische Grenzen** (BW, Noise-Floor, Frequenzgang): **Material-Prior hat Vorrang** — das tatsächliche physikalische Medium bestimmt, was maximal erreichbar ist
- **Ästhetische Entscheidungen** (Vintage-Wärme, Raumhall, Soft-Saturation): **Ära-Prior hat Vorrang** — der Zeitgeist der Aufnahme bestimmt, welcher Klangcharakter bewahrt wird
- **Defekt-Schwellen** (Click-Sensitivity, Hum-Threshold): **Material-Prior hat Vorrang** — analoge Medien haben andere Artefakt-Signaturen als digitale
- **Genre-Profil vs. alle anderen Priors**: Genre-Profil-`*_enabled: False`-Keys sind absolute Overrides (§2.20 Spec 03)

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

> **§2.54 Kontext**: Die untenstehenden Werte sind **Pipeline-Ende-Ziele**, nicht per-Phase-Guards.
> Einzelphasen dürfen vorübergehend davon abweichen (Carrier-Repair, Baseline-Capping, iterative Stärke-Anpassung).
> Die Werte werden durch AdaptiveGoalThresholds (§2.31) material-/era-/restorability-adaptiv skaliert.

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
8. **Rauschboden** (modus-differenziert):
   - **Restoration**: Material-adaptiv — Rauschboden des originalen Aufnahmemediums anstreben. Ein Studio-Tape von 1965 hatte ≈ −60 dBFS; erzwungene −72 dBFS entfernt Studio-Ambience und zerstört Raumklang. Richtgrößen: Shellac ≤ −45 dBFS, Vinyl ≤ −55 dBFS, Tape ≤ −60 dBFS, Digital ≤ −72 dBFS.
   - **Studio 2026**: ≤ −72 dBFS, A-gew. ≤ −75 dB(A), 0 Musical-Noise-Events in Stille
   - **Beide Modi**: 0 Musical-Noise-Events in Stille-Segmenten (Musical Noise ist immer ein Artefakt)
9. **Mikro-Dynamik**: Pearson des 400 ms LUFS-Profils ≥ 0.92, Crest-Faktor ≤ 1.5 dB
10. **Vintage Aesthetics** (automatisch via EraClassifier):
    - 1920–1940: Rolloff ≤ 7 kHz nicht künstlich erweitern
    - 1940–1955: Röhren-Kompressions-Fingerabdruck erhalten (H2, H4 ∈ [−30, −20] dBr)
    - 1955–1965: RT60 ∈ [1.2, 2.0] s erhalten (kein aggressives Dereverb)
    - 1965–1975: Tape-Saturation-Signatur nicht entfernen
11. **Kompetitiver Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 AMRB-Szenarien
12. **Emotionaler Dynamik-Bogen**: Arousal-Pearson ≥ 0.85, Valence-Pearson ≥ 0.80, Klimax-Peak-Abw. ≤ 2 Segmente

---

## §2.56 Song-Goal-Importance — Per-Song Goal Weighting (v9.12.0) [RELEASE_MUST]

Die 14 Musical Goals bilden eine Pareto-Front: nicht alle können gleichzeitig vollständig erfüllt werden,
da sie sich physikalisch teilweise ausschließen (z.B. Brillanz vs. Wärme, Transparenz vs. Raumtiefe).
Für jedes Stück muss die richtige Gewichtung aus dem musikalischen Kontext berechnet werden.

**Grundprinzip**: Jeder Song erhält vor der Phase-Pipeline ein individuelles Gewichtungsprofil
`SongGoalImportance` mit 14 Gewichten ∈ [0.3, 2.0]. Diese Gewichte bestimmen, welche Goals
bei diesem konkreten Song Vorrang genießen und welche toleranter behandelt werden.

### §2.56a 5-Stufen-Gewichtungsarchitektur (v9.12.0)

Die Berechnung erfolgt in **5 multiplikativen Stufen** + Soft-Cap + Bounds:

**Stufe 1 — Label-basiert (Genre/Era/Material/Vocal/Restorability):**

| Schritt | Beschreibung | Parameter |
| --- | --- | --- |
| 1a | **Genre-Profil** (16 Profile: Klassik, Oper, Jazz, Rock, Metal, Electronic, Hip-Hop, Pop, Schlager, Soul/R&B, Blues, Country, Folk, Reggae, Latin, Funk, Gospel) | `genre_label` → Alias-Resolution (`_GENRE_ALIASES`) → Basis-Gewichte. Unbekannt → neutral (1.0) |
| 1b | **Ära-Modifikator** (1900er–1980er) multiplikativ auf Genre | `era_decade` → z.B. 1920er: Brillanz ×0.5 (7 kHz Bandbreite) |
| 1c | **Material-Modifikator** (trägertypisch) | `material_type` → z.B. Vinyl: Bass ×1.3, Shellac: Transparenz ×0.7 |
| 1d | **Vokal-Boost** (konfidenzgewichtet) | `vocal_detected`, `vocal_confidence` → Artikulation ×1.3, Emotionalität ×1.2, Authentizität ×1.2, TransparenzVokal ×1.15, TonalCenter ×1.12, TimbreAuth ×1.1, SeparationFidelity ×1.1 (Quellen: Kreiman & Sidtis 2011; Marjieh et al. 2023; Bregman 1990; McDermott 2009 Curr Biol; London 2012; Repp & Su 2013) |
| 1e | **Restorability-Adjustment** | `restorability_score` < 40 → P3–P5-Gewichte ×0.85 |
| 1f | **Studio 2026-Modus** | `is_studio_2026` → Transparenz ×1.2, Brillanz ×1.2, Separation ×1.1 |

**Stufe 2 — Audio-abgeleitet (reale Signalanalyse):**

| Schritt | Feature | Einheit | Schwellwerte | Beispiel-Effekt |
| --- | --- | --- | --- | --- |
| 2a | SNR | dB | < 15: noisy, > 40: clean | noisy → transparenz ↑, waerme ↑ |
| 2b | Effektive Bandbreite | Hz | < 6000: BW-limited, 6k–12k: medium, > 18k: wideband | BW-limited → brillanz ↓ 0.7, waerme ↑ |
| 2c | Dynamikumfang | dB | < 20: compressed, > 50: dynamic | compressed → micro_dynamics ↑ |
| 2d | Stereo-Mono-Kompatibilität | [0, 1] | < 0.4: poor, > 0.9: good | poor → spatial_depth ↑ |
| 2e | BPM | Schläge/min | < 60: slow, > 110: fast | slow → spatial_depth ↑, groove ↓ |
| 2f | Defekt-Schwere | Dict[str, float] | max() über Familien: Rauschen, Knistern, HF-Verlust, Wow | Knistern hoch → transparenz ↑, groove ↑ |
| 2g | Spektraler Tilt | dB/Oct | < −4: dark, > −1: bright | dark → brillanz ↓, waerme ↑ |

**Stufe 2i — Tonträgerketten-Degradation (§2.46/§2.46a):**

| Feature | Einheit | Schwellwerte | Effekt |
| --- | --- | --- | --- |
| `transfer_generation_count` | int | 2: minor, 3: significant, ≥ 4: severe | Mehr Generationen → natuerlichkeit ↑, transparenz ↑; brillanz ↓ bei ≥ 4 |
| `cumulative_hf_loss_db` | dB | > 6: moderate, > 12: severe | > 12 dB → brillanz ↓ 0.85 |
| `source_fidelity_confidence` | [0, 1] | < 0.5: uncertain | Niedrig → alle Carrier-Adjustments ×confidence gedämpft |

**Stufe 3 — Psychoakustisch (Zwicker/Aures/ISO 11172-3):**

| Schritt | Feature | Einheit | Schwellwerte | Effekt |
| --- | --- | --- | --- | --- |
| 3a | Roughness (Zwicker) | asper | > 0.5: rough, < 0.2: smooth | rough → transparenz ↑, waerme ↑ |
| 3b | Sharpness (Bismarck) | acum | > 0.5: sharp, < 0.2: dull | sharp → waerme ↑ |
| 3c | Spectral Flatness | [0, 1] | > 0.3: noisy, < 0.05: tonal | noisy → transparenz ↑; tonal → tonal_center ↑ |
| 3d | Tonality | [0, 1] | > 0.5: tonal, < 0.15: atonal | tonal → tonal_center ↑; atonal → groove ↑ |
| 3e | Frequenzbalance | Dict[bass/mid/treble/air] | bass > 0.6: bass-heavy, treble < 0.1: treble-starved | bass-heavy → bass_kraft ↑; treble-starved → brillanz ↓ |
| 3f | Maskierte Komponenten | [0, 1] | > 0.3: masked | masked → transparenz ↑; **Sanity-Guard**: ratio ≥ 0.95 oder ≤ 0.01 → ignoriert (Messartefakt) |
| 3g | Perzeptueller Schwerpunkt | Bark | < 4.0: dark, > 8.0: bright | dark → waerme ↑, brillanz ↓ |

**Stufe 4 — Vokal/Harmonik/Transient (Musikerhaltung):**

| Schritt | Feature | Messverfahren | Kalibrierter Bereich | Schwellwerte | Effekt |
| --- | --- | --- | --- | --- | --- |
| 4a | HNR | Multi-Frame Pitch-Period AC (50 ms Frames, `find_peaks(height=0.1)` in [sr/2000, sr/50], Median) | Full-Mix: [−10, +20] dB | > 5 dB: clean, < 0 dB: noisy | clean → timbre ↑, artikulation ↑; noisy → transparenz ↑ |
| 4b | Harmonische Kohärenz | STFT Spectral-Peak-Consistency (`PsychoAcousticMetrics.calculate_harmonic_coherence`) | [0, 1] | > 0.7: strong, < 0.3: weak | strong → tonal_center ↑, waerme ↑; weak → groove ↑ |
| 4c | Crest Factor | `20 log10(percentile99.9 / RMS)` — robust gegen Impuls-Artefakte | 99.9-Perzentil: [6, 16] dB | > 12 dB: dynamic, < 7 dB: compressed | dynamic → micro_dynamics ↑; compressed → micro_dynamics ↑ (Repair-Priorität) |
| 4d | Transient Density | STFT Spectral-Flux + adaptive Threshold + 50 ms Min-Gap (`dsp.dtw_groove.detect_onsets`) | [0, ~20]/s | > 8/s: percussive, < 2/s: sustained | percussive → groove ↑, micro_dynamics ↑ |

> **Alle graduiert**: Schwellwerte nutzen `_f = min((val − threshold) / range, 1.0)` statt binärer Flags.

**Stufe 5 — Cross-Feature-Interaktionen (superadditive Effekte):**

| ID | Interaktion | Bedingung | Effekt |
| --- | --- | --- | --- |
| 5a | Roughness × Low SNR | rough > 0.3 AND snr < 20 dB | transparenz ↑↑, artikulation ↑ (Intelligibilitätskrise) |
| 5b | HNR × Vocal | HNR > 5 dB AND vocal_detected | timbre ↑, natuerlichkeit ↑, separation_fidelity ↑ (Bregman 1990 + McDermott 2009: Stimme/Begleittrennung ist Primär-Streamtask bei Vokalmusik) |
| 5c | Low BW × Dark Centroid | BW < 10 kHz AND centroid < 5 Bark | brillanz ↓↓, waerme ↑ (physikalisch unmöglich zu restaurieren) |
| 5d | Coherence × Tonality | coherence > 0.5 AND tonality > 0.4 | tonal_center ↑↑ (definierendes Merkmal des Songs) |
| 5e | Dynamic × Transient | crest > 10 dB AND density > 5/s | groove ↑, micro_dynamics ↑ (perkussiv UND dynamisch) |
| 5f | Multi-Gen Chain × Low SNR | generations ≥ 3 AND snr < 20 dB | natuerlichkeit ↑, authentizitaet ↑ (Maximum erhalten) |

**Stufe 6 — Soft-Cap (rationale Kompression) + Bounds:**

```
Formel:  w > 1.5 → w' = 1.5 + excess/(1 + 3·excess)    // Asymptote: 1.83
         w < 0.5 → w' = 0.5 - deficit/(1 + 3·deficit)   // Asymptote: 0.17
```

- **k = 3.0**: Erhält relatives Ranking, verhindert dass extreme Gewichte PMGG/CIG-Restauration blockieren
- **Danach**: P1/P2-Floor ≥ 0.70, Hard-Bounds [0.30, 2.00] per `np.clip`

### §2.56b Weight-Semantik

- `weight = 1.0` → Neutral (Standard-Verhalten)
- `weight > 1.0` → Goal ist wichtiger: strengere Regressionsprüfung, weniger tolerante Drift
- `weight < 1.0` → Goal ist weniger relevant: tolerantere Regression, mehr Spielraum

### §2.56c Integration in alle Entscheidungspunkte

| Komponente | Gewichtungs-Effekt |
| --- | --- |
| **PMGG** `_max_regression()` | `weighted_reg = raw_reg × weight` → wichtige Goals triggern Retries leichter |
| **PMGG** `_max_regression_priority_aware()` | Gewichtete Regression wird gegen Priority-Threshold geprüft |
| **CIG** Drift-Toleranz | Negative Drift wird per Goal gewichtet: `drift × weight` |
| **GoalPriorityProtocol** `resolve_conflict()` | Bei gleicher Priority entscheidet höheres Gewicht |
| **GoalPriorityProtocol** `should_abort_iteration()` | `effective_epsilon = base_epsilon / weight` |
| **FeedbackChain** | Gewichtete Abort-Prüfung via GPP |
| **RestorationResult.metadata** | `goal_importance` Block mit Weights + Profil-Info |

### §2.56d Invarianten

- §2.56 ist **Steuerungsmechanismus**, nicht Notbremse. Die Gewichte bestimmen die Pareto-Optimierungsrichtung.
- P1/P2-Goals (Natürlichkeit, Authentizität, Tonal, Timbre, Artikulation) haben Mindestgewicht 0.70 (§0 Primum non nocere).
- Das Gewichtungsprofil wird **EINMALIG** nach allen Klassifizierern und Audio-Feature-Extraktion berechnet und gilt für die gesamte Pipeline.
- Fehlende Genre-/Era-/Material-Information → Uniform-Fallback (alle Gewichte = 1.0).
- Fehler in der Gewichtungsberechnung darf die Pipeline nicht blockieren (graceful fallback).
- Alle Audio-Features sind **optional** (None → Schritt wird übersprungen). Rein label-basierte Berechnung bleibt valide.
- Genre-Alias-Resolution: `"soul"` → `"soul/r&b"`, `"deutscher schlager"` → `"schlager"` etc.

### §2.56e Messverfahren-Konventionen (normativ)

| Feature | Korrekt | VERBOTEN |
| --- | --- | --- |
| HNR | Multi-Frame Pitch-Period Autocorrelation, Median über Frames | Lag-1 SNR-Proxy (`_compute_hnr` aus ComprehensiveMetrics) |
| Harmonic Coherence | STFT Spectral-Peak-Consistency (ganzer Song) | 46 ms Single-Frame AC (`_estimate_harmonic_coherence`) |
| Crest Factor | `20 log10(percentile(99.9) / RMS)` | `np.max()` — Impuls-Artefakte dominieren |
| Transient Density | STFT Spectral-Flux + 50 ms Min-Gap | RMS-Flux mit 1.5σ-Threshold (inflationiert Count) |

**Implementierung:**

- Core: `backend/core/song_goal_importance.py` — `SongGoalImportance`, `estimate_goal_importance()`
- Feature-Extraktion: `unified_restorer_v3.py` — SGI-Block in `restore()` nach SongCalibration
- Durchreichung: `phase_kwargs["goal_importance"]` → PMGG `goal_weights` Parameter
