# Aurik 10 — Spec 11: Entscheidungsintelligenz (PIM, RLP, Artistic Intent)

> **Normative Quelle** für alle v10-Entscheidungsmodule.
> **Invarianten** sind mit `§INV` markiert und MÜSSEN bei allen Änderungen erhalten bleiben.

## §INV-1: PIM-first

Der **Perceptual Intensity Mapper (PIM)** MUSS VOR dem Phasen-Loop ausgeführt werden.
Sein Ergebnis (`IntensityMap`) MUSS in `restoration_context["pim_intensity_map"]`
gespeichert werden. Jede Phase KANN die Map abrufen, MUSS aber mit fehlender Map
umgehen können (Fallback: unity intensity).

```
restore() {
    artistic_intent = get_artistic_intent(genre, era, material)
    restoration_context["artistic_intent"] = artistic_intent

    intensity_map = pim.compute_intensity_map(audio, sr,
        artistic_intent=artistic_intent,
        source_profile=source_profile,
        salience_result=salience)

    for phase in selected_phases:
        phase.process(audio, sr, pim_intensity_map=intensity_map, ...)
}
```

### PIM-Konfiguration

| Parameter | Default | Bereich | Bedeutung |
|-----------|---------|---------|-----------|
| `CRITICAL_BANDS` | 10 Bänder | sub_bass–ultra | Frequenzbänder für per-Band-Intensität |
| `SECTION_MODIFIERS` | 6 Typen | intro–solo | Sektions-Multiplikatoren |
| `MAX_ITERATIONS` | 2 | 1–5 | Max RLP-Iterationen |
| `IMPROVEMENT_THRESHOLD` | 0.001 | 0.0001–0.01 | Mindest-Verbesserung für Akzeptanz |

## §INV-2: RLP-last

Der **Reflective Listening Pass (RLP)** MUSS NACH dem Phasen-Loop ausgeführt werden.
Korrekturen werden NUR bei objektiver Verbesserung übernommen.
Maximal 2 Iterationen („nicht totpolieren").

### RLP-Diagnose-Dimensionen

| # | Dimension | Messung | Korrektur |
|---|-----------|---------|-----------|
| 1 | Spectral Tilt | Polyfit 100Hz–10kHz | High-Shelf ±1.5dB |
| 2 | Sibilanz | 5–10kHz / 1–4kHz Energie-Ratio | De-Essing 0–0.5 Stärke |
| 3 | Bass-Druck | 20–150Hz / 200–2kHz Energie-Ratio | Low-Shelf ±1.5dB |
| 4 | Stereo-Breite | Side/Mid RMS-Ratio | Side-Gain ±5% |
| 5 | Dynamik-Verlust | LRA V1 vs LRA Ref | Keine weitere Kompression |
| 6 | Rausch-Modulation | StdDev der leisen Abschnitte | Sanftes HF-NR |

## §INV-3: Artistic Intent vor Defect-Scan

`get_artistic_intent()` MUSS VOR dem Defect-Scan berechnet werden.
Das Ergebnis beeinflusst die Phasen-Selektion, die PIM-Intensität und
die RLP-Diagnose-Schwellwerte.

### Genre-Profile (12)

| Genre | Dynamik | Wärme | Brillanz | Risiko |
|-------|---------|-------|----------|--------|
| Ballad | Erhalten (12 LU) | 0.85 | 0.35 | 0.20 |
| Classical | Maximal (14 LU) | 0.60 | 0.40 | 0.10 |
| Jazz | Erhalten (9 LU) | 0.80 | 0.45 | 0.25 |
| Schlager | Komprimiert (6 LU) | 0.75 | 0.55 | 0.30 |
| Pop | Komprimiert (6 LU) | 0.60 | 0.65 | 0.35 |
| Electronic | Stark (5 LU) | 0.50 | 0.70 | 0.40 |
| Rock | Komprimiert (5 LU) | 0.65 | 0.60 | 0.35 |
| Metal | Stark (4 LU) | 0.55 | 0.65 | 0.40 |
| Folk | Erhalten (9 LU) | 0.80 | 0.40 | 0.20 |
| Blues | Erhalten (8 LU) | 0.85 | 0.35 | 0.20 |
| HipHop | Komprimiert (5 LU) | 0.50 | 0.60 | 0.35 |
| Unknown | Erhalten (8 LU) | 0.65 | 0.50 | 0.20 |

## §INV-4: Glue Stage immer

Die **Glue Stage** MUSS in ALLEN Modi (Restoration + Studio 2026) als vorletzte
Phase laufen — nach TruePeak, vor Dithering. Parameter:

| Parameter | Default | Bedeutung |
|-----------|---------|-----------|
| Ratio | 1.2:1 | Sanfte Bus-Kompression |
| Attack | 30ms | Lässt Transienten durch |
| Release | 100ms | Folgt dem Groove |
| Max GR | 1.5dB | Kaum hörbare Reduktion |

## §INV-5: Soft-Knee-Gate

`apply_musical_gain_envelope()` in `backend/core/audio_utils.py` MUSS mit
Soft-Knee-Sigmoid arbeiten. Parameter:

| Parameter | Default | Bereich | Bedeutung |
|-----------|---------|---------|-----------|
| knee_width_db | 6.0 | 2.0–12.0 | Sigmoid-Übergangsbreite |
| crossfade_ms | 200.0 | 50–500 | Hanning-Window-Breite |
| small_gain_bypass_db | 2.0 | 0.5–4.0 | Gains ≤ X → uniform |

**VERBOTEN**: Binäres Gate (0/1), Hard-Clamp (§2.30b), crossfade_ms < 50.

## §INV-6: ML-Fallback-Logging

JEDER ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden.
Silent-Failures (nur `logger.debug()` oder gar kein Log) sind VERBOTEN.
Alle 54 ML-Module MÜSSEN einen auditierbaren Fallback-Pfad haben.

## §INV-7: Bridge-Bypass-Verbot

Kein Frontend-Code (`cli/`, `batch_processor.py`) darf `backend/core/`,
`dsp/` oder `plugins/` direkt importieren. Alle Zugriffe MÜSSEN über
`backend/api/bridge.py` laufen.

## §INV-8: Stop-Regel

Die Stop-Regel (`should_stop_pipeline()`) MUSS nach jeder Phase evaluiert werden.
Bei PMGG-Δ < 0.01 über 3 aufeinanderfolgende Phasen MUSS die Pipeline stoppen.
Ausnahmen: `_NEVER_SKIP`-Phasen (Phase 01, 09, 12, 14, 15, 30, 47, 65).

## PIM→Phase-Integration (Referenz)

```python
# In Phase 03 (Denoise):
pim = kwargs.get("pim_intensity_map")
if pim is not None:
    nr_global = pim.global_modifiers.get("nr_global", 1.0)
    nr_presence = pim.get_nr_strength("presence", "verse")  # → 0.20 für Schlager
    nr_ultra = pim.get_nr_strength("ultra", "verse")        # → 0.65 für Kassette
    # Wende per-Band-Stärke an...
```

## RLP→Pipeline-Integration (Referenz)

```python
# Nach dem Phasen-Loop, vor return:
rlp_result = rlp.listen_and_refine(current_audio, sr,
    reference_audio=pre_pipeline_audio,
    artistic_intent=artistic_intent)
if rlp_result.overall_improved:
    current_audio = rlp_result.audio
```
