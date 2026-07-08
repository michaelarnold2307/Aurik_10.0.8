# Aurik Session Insights — §2.59 Surgical Repair & Critical Fixes

## Datum: 2026-07-08 | Branch: fix/lint-perf-lpc-v9-12-10

---

## 1. Architektur-Erkenntnis: Chirurgie braucht existierende Phasen, nicht neue Funktionen

**Falscher Ansatz (verworfen):** 15 neue Lightweight-DSP-Funktionen in `surgical_repair.py`.

**Richtiger Ansatz (in Arbeit):** `SurgicalPlan`-Dataclass, die den existierenden,
kampferprobten Phasen Zeitfenster-Instruktionen gibt.
`SURGICAL_DEFECT_TO_PHASE` mappt Defekttypen auf Phase-IDs.

**Warum:** Die globalen Phasen haben jahrelange Validation, PMGG-Integration,
Psychoakustik. Neue Lightweight-Funktionen haben keine dieser Garantien.

**Status:** `surgical_plan.py` definiert die Architektur. Lightweight-Funktionen
mit Safety-Clamps aktiv als Übergangslösung. Finaler Schritt: Phasen im
Zeitfenster-Modus (`time_range`-Parameter).

---

## 2. Safety-Clamps sind Pflicht für jede per-Instance-Reparatur

**`_safety_clamp(repaired, original, max_ratio=2.0)`:**
- Kein Sample darf > 2× Original-Amplitude sein
- Global RMS darf nicht > 2× Original sein

**Ohne Clamp:** Cubic-Spline über 225s = komplettes Signal zerstört.
**Mit Clamp:** Selbst bei fehlerhafter Zone bleibt das Signal hörbar.

**Regel:** JEDE chirurgische Funktion MUSS `_safety_clamp()` vor Return aufrufen.

---

## 3. DefectScanner → Analyzer Datenfluss war gebrochen

**DefectScanner liefert:** `scores[dt].locations = [(start_s, end_s), ...]` (15000+ Einträge)
**Analyzer bekam:** Nur `defect_scores` (Severity-Dict), keine Locations
**Analyzer erstellte:** Placeholder-Zonen `(0.0, duration)` für jeden Defekttyp
**Repair-Funktionen liefen auf:** 225s statt Millisekunden → Signal zerstört

**Fix (in `unified_restorer_v3.py` §2.59.10):**
```python
_defect_locations = {}
for _dt, _ds in defect_result.scores.items():
    if _ds.locations:
        _defect_locations[_dt_key] = list(_ds.locations)

analyzer.analyze(defect_locations=_defect_locations)
```

**Test-Gate:** `test_surgical_repair_gate.py` — Zonen dürfen NIEMALS >1s oder 0–duration sein.

---

## 4. SNR=None → konservativ, nicht aggressiv

**Bug in `phase_03_denoise.py:856`:**
```python
# FALSCH: SNR=None → Gate=True (voller ML-Eingriff)
_bsrof_gate = ... and (_est_snr_db is None or _est_snr_db < 20.0)

# RICHTIG: SNR=None → Gate=False (kein ML, nur DSP)
_bsrof_gate = ... and (_est_snr_db is not None and _est_snr_db < 20.0)
```

**SGMSE+ Sigma-Fallback:** 15dB → 22dB (weniger Diffusion bei unbekanntem SNR).

**Warum kritisch:** Vintage-Aufnahmen haben oft unzuverlässige SNR-Schätzung.
BS-RoFormer + MIIPHER + Resemble bei voller Stärke = Gesangsverzerrung.

---

## 5. Gender-Detection: Formanten > F0 > Confidence

**Fix in `phase_19_de_esser.py`:**
- Confidence-Gate (`confidence < 0.80`) ENTFERNT
- F0-Bereich erweitert: 145–195Hz → 140–220Hz
- Formanten (F1/F2) sind anatomisch fix — F0 variiert mit Tonhöhe

**Regel:** Wenn F1 UND F2 im weiblichen Bereich liegen → FEMALE.
Classifier-Confidence ist nachrangig.

---

## 6. Branch-Management: Nie von Long-Lived-Branch cherry-picken

**`fix/lint-perf-lpc-v9-12-10`:** 254 Commits vor main, funktioniert lokal.
**`fix/surgical-repair-v2-59`:** Cherry-pick-Versuch, fehlten `bridge.py`, `restoration_policy.py`.

**Lektion:** Bei großen Refactorings immer vom ARBEITENDEN Branch aus starten,
nicht von main. Sauberen PR-Branch per `git checkout -b new-branch` vom
funktionierenden Branch, dann ALLE Änderungen in einem Squash-Commit.

---

## 7. Dead Code: §G Feedback-Loop

**`feedback_strength`**: 1× geschrieben (aurik_denker.py:2586), 0× gelesen.
Der Loop rief UV3 mit identischen Parametern erneut auf → kein Effekt,
doppelte Laufzeit. Komplett entfernt.

---

## 8. Pre-Commit-Hooks: Venv-Awareness

Der Surgical-Repair-Check muss außerhalb des venv laufen können.
Lösung: `try/except ImportError` → silent pass mit Hinweis auf manuellen Check.

---

## 9. CI-Pipeline: psutil als optionale Dependency

**Problem:** CI installiert aus `requirements_dev.txt`, nicht `requirements_aurik.txt`.
psutil fehlte → ImportError in `phase_23_spectral_repair.py`.

**Fix:** Import `try/except` mit `psutil = None`, alle Aufrufe über
`_psutil_available_ram_gb()`-Wrapper. Zusätzlich `pip install --no-cache-dir psutil`
im CI-Workflow.

---

## 10. Restaurierungs-Qualität: Soft-Saturation PRESERVE, nicht entfernen

**ClippingDetector** findet `soft_saturation` (analoge Bandsättigung) mit
confidence=0.30. Das ist KEIN Defekt — es ist Aufnahmecharakter.
Die Restaurierung sollte soft_saturation bewahren, nicht "reparieren".
