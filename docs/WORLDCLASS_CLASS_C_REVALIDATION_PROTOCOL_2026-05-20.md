# Klasse-C Revalidierungsprotokoll (Weltspitze)

Stand: 2026-05-20

Ziel: Drei aktuell kalibrierte Klasse-C-Schwellen auf belastbare, reproduzierbare Evidenz zu heben, ohne die Release-Sicherheit zu gefaehrden.

## Scope (PR-Serie)

1. `material_vqi_floor` (materialadaptive Vocal-Floors)
2. `MERT_floor = max(raw_mert, 0.5)`
3. `timbral_fidelity` materialadaptiver Mindestboden (Restoration)

Diese drei Schwellwerte sind aktuell Betriebskalibrierung mit praktischer Validierung, aber ohne vollstaendige A/B-klassige Evidenzkette fuer jede Materialklasse.

## Arbeitsmodus

- Modus: `restoration` (primaer), `studio2026` nur als Kontrollarm.
- Materialarme: `shellac`, `vinyl`, `tape`, `cd_digital`, `mp3_low`.
- Fokus: Musik mit Gesang (Vocal-Supremacy), Instrumental als Sekundaerarm.

## WP1: `material_vqi_floor` Revalidierung

### Forschungsfrage

Sind die aktuellen materialadaptiven VQI-Floors (z. B. shellac/vinyl/cd) perzeptuell optimal oder ueber-/unterstrikt?

### Experiment

- Pro Material mindestens 20 realistische Musikfaelle (gesamt >= 100).
- Pro Fall 3 Schwellenvarianten: `baseline`, `baseline - 0.03`, `baseline + 0.03`.
- Blind-Hoertest (ABX + MUSHRA-light) mit Fokus auf:
  - Stimmidentitaet
  - Formantnatuerlichkeit
  - Vibratoerhalt
  - Hoerbare Artefakte

### Akzeptanzkriterien

- Keine Erhoehung der Artefaktquote (`artifact_freedom < 0.95`) gegenueber Baseline.
- Mindestens +0.15 MUSHRA-light Median in Vokalqualitaet ODER signifikant weniger Vocal-Rollbacks.
- Kein signifikanter Verlust bei `singer_identity_cosine`.

### Umsetzung in Code

- Konfiguration in `backend/core/calibration_matrix.py` (nur nach Testabschluss).
- Guard-Tests in `tests/unit/` + materialgruppierte Integrationsfaelle.

## WP2: `MERT_floor` Revalidierung

### Forschungsfrage

Ist der globale MERT-Floor 0.5 ueber alle Material-/Genrekombinationen optimal oder sollte er material-/era-adaptiv werden?

### Experiment

- Grid: `0.45`, `0.50`, `0.55`.
- Stratifikation nach Restorability-Bins: `<40`, `40-70`, `>70`.
- Vergleich gegen VERSA-Primarmetrik: Korrelation zum Hoerurteil und Fehlentscheidungsrate im HPI-Gate.

### Akzeptanzkriterien

- Hoehere Rangkorrelation zwischen HPI-Entscheidung und Blind-Hoertest.
- Keine Erhoehung falsch-positiver Exporte (subjektiv schlechter als Input).
- Keine Erhoehung von `degraded`-Exports in Hochrestorability-Material.

### Umsetzung in Code

- Falls adaptiv: material-/restorability-Map in Kalibrierungsmatrix statt fixer Konstante.
- Regressionstests fuer HPI-Entscheidungen mit eingefrorenen Fixtures.

## WP3: `timbral_fidelity` Floor Revalidierung

### Forschungsfrage

Sind die materialadaptiven Timbral-Floors inkl. restorability-Skalierung passend zur subjektiven Klangtreue?

### Experiment

- Baseline vs. 2 alternative Floor-Kurven (flacher/steiler) pro Materialklasse.
- Kontrolle auf False-Rejects: gute Restaurierungen, die aktuell unnoetig am `TIMBRAL_BELOW_FLOOR` scheitern.
- Kontrolle auf False-Accepts: klanglich verfremdete Exporte, die derzeit noch durchgehen.

### Akzeptanzkriterien

- Weniger False-Rejects bei gleicher oder besserer Artefaktfreiheit.
- Keine Verschlechterung der Vocal-Metriken bei Gesangsmaterial.
- Stabiler oder verbesserter Output-Score in P1/P2 Goals.

### Umsetzung in Code

- `HolisticPerceptualGate` + zugehoerige Kalibrierungsquelle synchron anpassen.
- Neue Tests fuer `TIMBRAL_BELOW_FLOOR` mit materialtypischen Fixtures.

## Test- und Release-Gates (verpflichtend)

- Unit: neue Schwellen-/Mapping-Tests fuer alle drei WPs.
- Integration: Materialarme als deterministische Teil-Suite (`not ml`, falls moeglich mit Stubs).
- Audit: Vorher/Nachher-Report mit Fail-Grund-Verteilung (`artifact_freedom`, `VQI`, `TIMBRAL_BELOW_FLOOR`).
- Rollback-Regel: Jede WP-Aenderung hinter Feature-Flag bis Real-Audio-Gate bestanden.

## PR-Reihenfolge

1. PR-A: Mess-/Audit-Infrastruktur + Fixtures + Statistik-Export.
2. PR-B: WP1 `material_vqi_floor`.
3. PR-C: WP2 `MERT_floor`.
4. PR-D: WP3 `timbral_fidelity` floor.
5. PR-E: Konsolidierung + Source-Tag-Update + Changelog.

### PR-A Sofortstart (operativ)

1. Beispiel-Manifest fuellen: `config/class_c_revalidation_manifest.example.json`
2. Plan generieren:

```bash
./.venv_aurik/bin/python scripts/run_class_c_revalidation_plan.py \
  --manifest config/class_c_revalidation_manifest.example.json \
  --out-dir reports/revalidation
```

3. Ergebnisse in `result_template.csv` je Planzeile eintragen (Batch oder automatisiert).
4. Vorher/Nachher-Auswertung als Audit-Report exportieren.

```bash
./.venv_aurik/bin/python scripts/summarize_class_c_revalidation_results.py \
  --input-csv reports/revalidation/<run_id>/result_template.csv
```

Erzeugt:

- `reports/revalidation/<run_id>/summary.json`
- `reports/revalidation/<run_id>/summary.md`

## Definition of Done (pro WP)

- Reproduzierbare Resultate dokumentiert.
- Schwellwertentscheidung mit Evidenzklasse aktualisiert.
- Source-IDs/Traceability aktualisiert.
- Testgruen in relevanten Suiten.
- Kein Konflikt mit §0h/§0p/§2.49 Hard-Gates.

## PR-B (WP1) Sofortstart

Dry-Run (Plan-/Datei-Validierung, ohne Restaurierung):

```bash
./.venv_aurik/bin/python scripts/run_wp1_material_vqi_revalidation.py \
  --run-dir reports/revalidation/<run_id>
```

Echter Lauf (Restaurierung + VQI-Auswertung):

```bash
./.venv_aurik/bin/python scripts/run_wp1_material_vqi_revalidation.py \
  --run-dir reports/revalidation/<run_id> \
  --execute
```

Schneller technischer Start (empfohlen fuer erste Welle):

```bash
./.venv_aurik/bin/python scripts/run_wp1_material_vqi_revalidation.py \
  --run-dir reports/revalidation/<run_id> \
  --execute \
  --max-cases 1 \
  --max-seconds 8 \
  --ml-runtime-budget-s 20
```

Optional fuer kontrollierte Startwelle:

```bash
./.venv_aurik/bin/python scripts/run_wp1_material_vqi_revalidation.py \
  --run-dir reports/revalidation/<run_id> \
  --execute \
  --max-cases 5
```
