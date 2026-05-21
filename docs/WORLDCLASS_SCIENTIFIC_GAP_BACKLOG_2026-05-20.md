# Worldclass Scientific Gap Backlog (Stand 2026-05-20)

Ziel: Wissenschaftliche Luecken identifizieren, die den Weltspitzen-Anspruch in Aurik direkt beeinflussen, und fuer jede Luecke ein belastbares Recherche- und Umsetzungsprotokoll definieren.

## Kurzfazit

Es gibt relevante Luecken mit hohem Hebel. Besonders kritisch sind:

- musik-spezifische Validierung der Gate-Schwellen (HPI/AFG/VQI)
- wissenschaftliche Herleitung der derzeit kalibrierten Grenzwerte
- robuste Evaluationsprotokolle fuer Restoration (nicht nur Speech- oder Codec-Kontext)

## Priorisierte Luecken (P1 zuerst)

| Prioritaet | Bereich | Aktueller Zustand | Luecke | Risiko fuer Weltspitze |
| --- | --- | --- | --- | --- |
| P1 | HPI/AFG/VQI Schwellen | Teilweise normativ (BS.1770/R128), teilweise kalibriert | Exakte Grenzwerte nicht durchgaengig primaerquellenbasiert | Falsch-positive Rollbacks oder zu laxe Freigaben |
| P1 | Musik-Restoration-Metrikvaliditaet | Mix aus VERSA/MERT/DNSMOS/Proxy | Domain-Shift: viele Metriken nicht fuer historische Musikrestauration kalibriert | Qualitätssteuerung trifft evtl. falsche Entscheidungen |
| P1 | Vocal Formant/Vibrato Grenzwerte | Starke vokalakustische Basis vorhanden | Exakte toleranzen in der Pipeline brauchen mehr direkte perzeptuelle Evidenz | Vokalnatuerlichkeit kann trotz guter Scores leiden |
| P2 | Transfer-Chain-Oracle (`chain_factor`) | Systemisch implementiert und getestet | Direkte Literaturabdeckung fuer konkrete Faktorformel noch duenn | Under/Over-processing bei komplexen Traegerketten |
| P2 | Artefaktgrenzen (Musical Noise, Pre-Echo, Stereo-Cancellation) | Gute technische Regeln vorhanden | Musik-spezifische, reproduzierbare Grenzvalidierung fehlt teilweise | Instabile Gate-Reaktionen je Material |
| P3 | Hallucination-Guard fuer generative Audio-Pfade | Technisch abgesichert | Einheitliche wissenschaftliche Benchmark fuer Audio-Halluzinationsdetektion fehlt | Versteckte Artefakte oder zu harte Ruecknahmen |

## Bereits verifizierte starke Quellenachsen

- Lautheit/True Peak: ITU-R BS.1770-5, EBU R128 (stark)
- Klassische NR-Theorie: Ephraim/Malah, IMCRA/OMLSA (stark)
- Formant/F0/Vokalphysik: Makhoul, Boersma, Titze, Sundberg (stark)

## Erste Recherche-Resultate (Live-Query, heute)

Die initialen API-Abfragen (Crossref) bestaetigen Kandidaten, zeigen aber auch: fuer mehrere Aurik-spezifische Fragen ist die Treffermenge verrauscht und erfordert kuratierte Nachrecherche.

### Relevante Treffer (Auszug)

- Pre-echo noise reduction in frequency-domain audio codecs (ICASSP 2017)
  DOI: 10.1109/ICASSP.2017.7952243
- Evaluation of short-time spectral attenuation techniques for the restoration of musical recordings (IEEE, 1995)
  DOI: 10.1109/89.365378
- The relationship between measured vibrato characteristics and perception in Western operatic singing (J Voice, 2004)
  DOI: 10.1016/j.jvoice.2003.09.003
- Perception of vibrato rate by professional singing voice teachers (JASA, 2022)
  DOI: 10.1121/10.0015518
- Formant frequency tuning in singing (1992)
  DOI: 10.1016/S0892-1997(05)80150-X

## Konkreter Forschungsplan (naechste 3 Arbeitspakete)

1. P1-Gates wissenschaftlich haerten

- Ziel: Fuer HPI/AFG/VQI-Schwellen eine Evidenzklasse vergeben (A=stark, B=mittel, C=kalibriert)
- Ergebnis: normativer Patch-Vorschlag mit Source-Tag je Schwellwert

2. Musik-Restoration-Evaluation konsolidieren

- Ziel: MUSHRA-/ABX-/Objective-Set fuer historische Musik mit Gesang als kanonisches Testprotokoll
- Ergebnis: neue Test-/Audit-Sektion inkl. Akzeptanzkriterien

3. Vocal-Grenzwerte (Formant/Vibrato) absichern

- Ziel: Perzeptuelle Toleranzbereiche spezifisch fuer Gesang im Restoration-Kontext
- Ergebnis: update-faehige Toleranzmatrix je Material/Era

## Umsetzungsregel fuer Quellenqualitaet

- Nur peer-reviewed, Normen oder AES/IEEE/JASA/J Voice-Quellen als Primaerbeleg
- Preprints nur als sekundaire Evidenz, bis peer-reviewte Bestaetigung vorliegt
- Jede neue Regel braucht: Quelle + Messprotokoll + Regressionstest

## Betroffene Aurik-Dokumente fuer den naechsten Patch

- .github/specs/02_pipeline_architecture.md
- .github/specs/09_global_calibration_matrix.md
- .github/instructions/pipeline.instructions.md
- docs/SCIENTIFIC_INVARIANT_TRACEABILITY_MATRIX.md

## Status

Freigabe fuer Recherche liegt vor. Naechster Schritt ist ein kuratierter, DOI-sauberer Source-Patch pro P1-Luecke mit konkretem Normtext-Delta.

Umsetzungsprotokoll fuer die naechste PR-Serie:

- `docs/WORLDCLASS_CLASS_C_REVALIDATION_PROTOCOL_2026-05-20.md`
- `docs/WORLDCLASS_SOTA_IMPLEMENTATION_MATRIX_2026-05-20.md`

## Aktivierungspaket 2026-05-21 (wissenschaftlich maximal, vokalfokussiert)

Normative Verankerung erfolgt in:

- `.github/specs/07_quality_and_tests.md` via `§8.6 Worldclass Hybrid-Engineer Protocol`

Damit wird der Weltspitzen-Anspruch von einer reinen Zielbeschreibung in ein
release-faehiges Mess- und Gate-System ueberfuehrt.

### AP-1: Human-Talent-Emulation-Vektor produktiv fuehren

- 12-dim Vektor (`hybrid_engineer_vector`) pro Run in Metadata persistieren
- Kontrakt: alle Schluessel vorhanden, normierte Werte, deterministische Berechnung
- Pflichtauswertung je Material/Era auf UAT-Matrix

### AP-2: WCS-Composite in Gates integrieren

- WCS als zusaetzliches End-Gate mit material-/modusbezogenen Minima
- Konfliktauflosung strikt nach Vocal-Supremacy-Hierarchie
- Kein Override fuer `artifact_freedom < 0.95`

### AP-3: Evidenzklassen A/B/C operationalisieren

- Jeder Gate-Schwellwert erhaelt `source_class`, `source_ref`, `validated_on`
- Klasse-C-Werte verpflichtend mit `revalidate_by`
- Build-Blocker fuer fehlende Evidenzmetadaten in neuen Schwellwerten

### AP-4: Weltspitzen-Testmatrix

- Normative Tests fuer HTEV-Contract, WCS-Gate, Evidence-Metadata
- Real-Audio-Gate auf Gesangsmaterial als Pflicht fuer Kernpatches
- Ergebnisaggregation je Materialklasse mit 5/95-Perzentil, nicht nur Mittelwert

### Definition of Ready fuer wissenschaftliche Patches

Ein Patch gilt erst dann als wissenschaftlich freigabefaehig, wenn alle Punkte vorliegen:

1. Quellenklassifikation A/B/C fuer jede neue Schwelle
2. Messprotokoll (Daten, Szenarien, Auswertung) reproduzierbar dokumentiert
3. Mindestens ein Regressionstest pro neue Invariante
4. Kein Konflikt mit Vocal-Supremacy und Artifact-Freedom-Veto
