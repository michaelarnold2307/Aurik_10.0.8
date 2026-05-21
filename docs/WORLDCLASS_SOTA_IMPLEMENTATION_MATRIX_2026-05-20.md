# Worldclass SOTA Implementation Matrix (Stand 2026-05-20)

Ziel: Aus externer Forschung sofort umsetzbare Upgrades fuer Aurik ableiten, so dass die 64-Phasen-Architektur reproduzierbar bessere Restaurierungsergebnisse fuer Musik mit Gesang liefert.

## Scope

- Fokus: Vollautomatische Musikrestaurierung mit Gesang (Restoration zuerst, Studio 2026 als Kontrollarm).
- Leitprinzipien: Vocal-Supremacy, Primum non nocere, materialadaptive Eingriffe, harte Artifact-Gates.
- Einsatzmodus: Keine Parallelwelt. Alle Massnahmen muessen in den bestehenden UV3- und Guard-Vertraegen landen.

## Externe Evidenz (kuratiert)

| Thema | Quelle | Umsetzungsrelevanz |
| --- | --- | --- |
| HT-Demucs / MSS | [arXiv 2211.08553](https://arxiv.org/abs/2211.08553) | Lange Kontexte plus Cross-Domain-Modellierung verbessern Vocal/Accompaniment-Trennung. |
| Demucs Betriebsgrenzen | [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | Segmentierung, Overlap und OOM-Handling als robuste Desktop-Betriebsregeln. |
| Blind-BWE (BABE) | [arXiv 2306.01433](https://arxiv.org/abs/2306.01433) | Historische, unbekannt bandbegrenzte Quellen profitieren von blindem Degradationsmodell. |
| Diffusion-Inpainting | [arXiv 2305.15266](https://arxiv.org/abs/2305.15266) | Mid/Long-Gap-Rekonstruktion (Dropouts) uebertrifft klassische Baselines. |
| Vocos | [arXiv 2306.00814](https://arxiv.org/abs/2306.00814) | Schneller Spektralrekonstruktionspfad bei sauberer Phasenkonsistenz. |
| CREPE | [arXiv 1802.06182](https://arxiv.org/abs/1802.06182) | Robuste F0-Extraktion als Basis fuer Vibrato-/Passaggio-Guards. |
| TorchCREPE Betrieb | [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) | Viterbi plus Periodizitaets-Schwellen reduzieren Halving/Doubling-Fehler. |
| DeepFilterNet | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Praktischer 48-kHz-NR-Baustein fuer sichere Echtzeit-/Desktop-Pfade. |
| ViSQOL Guidelines | [google/visqol](https://github.com/google/visqol) | Korrekte Auswertung (Fensterlaenge, Modus, Referenzannahme) verhindert Fehlinterpretation. |
| DNSMOS Kontext | [microsoft/DNS-Challenge DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | Als Zusatzsignal nuetzlich, fuer Musikrestaurierung aber nur sekundaere Evidenz. |

## Material-zu-Phasen-Matrix

| Materialklasse | Defektcluster | Primarfamilie in Aurik | SOTA-Upgrade | Guard-Bedingung |
| --- | --- | --- | --- | --- |
| shellac, wax, wire | Oberflaechenrauschen, enge BW, impulsive Stoerungen | subtraktive NR + spektrale Reparatur + sanfte BW-Erweiterung | BABE-aehnliche Blind-BWE nur in konservativem Modus | BW-Ceiling, Hallucination-Guard, VQI-Trigger |
| vinyl | knistern, hum, azimuth, moderate BW-Verluste | phase_04, phase_09, phase_12, phase_23 | materialadaptive Rekonstruktion mit striktem HF-Cap | Artifact-Freedom und Mono-Kompatibilitaet |
| tape, cassette | hiss, wow/flutter, dropouts, hf-rolloff | phase_12, phase_24, phase_29, phase_23 | Inpainting bei echten Gaps, Pitch-Guard via CREPE/torchcrepe | Vibrato/Formant-Guard, SSIP, Onset-Guard |
| cd_digital | clipping, jitter-aehnliche Artefakte, leichte Spektraldefekte | phase_30/31/23/50 family | selektive Spektralreparatur statt aggressiver NR | Keine Halluzination, timbral floor halten |
| mp3_low, aac_low | pre-echo, smearing, codec residue | phase_50 + phase_23 | codec-spezifische Residue-Behandlung plus konservative BWE | Pre-Echo-Guard und Spectral-Color-Guard |

## 64-Phasen-Orchestrierung: konkrete Hebel

1. Koalitions-Gating fuer gekoppelte Defekte

Warum: Einzelphase-Entscheidungen fuehren bei gekoppelten Ursachen zu Fehl-Rollbacks.

Umsetzung: Koalitionsweise Bewertung fuer bekannte Defektgruppen, erst danach delta-basierter Entscheid.

2. Per-Phase-Strength-Orakel mit kompletter Zielsicht

Warum: Single-Goal-Optimierung verschlechtert Vocals trotz guter Einzelmetrik.

Umsetzung: Jede adaptive Phase nutzt goal_gaps plus chain_factor plus material_confidence.

3. Kurze-Signal-Robustheit in Analysepfaden

Warum: Fast-Runs erzeugen technische Warnungen und instabile Proxy-Scores.

Umsetzung: safe frame_length, safe hop, fruehe Guards fuer Mini-Segmente.

4. Additive Phase nur unter Hallucination-Kontrolle

Warum: Historisches Material wird sonst unphysikalisch aufgehellt.

Umsetzung: BW-Ceiling vor Guard, Rollback bei Novelty-Ueberschreitung.

5. Vokalzentrierte Schutzkette

Warum: Produktziel ist Stimmwahrheit, nicht maximaler numerischer Score.

Umsetzung: VQI + singer_identity + formant + vibrato als verbindliche Recovery-Trigger.

## Messprotokoll fuer Weltklasse-Claim

1. Pflichtmetriken je Fall

artifact_freedom, vqi, singer_identity_cosine, timbral_fidelity, hpi und goal_violation_count (15-goal).

2. Pflichtauswertung je Material

Median und 5/95-Perzentil je Metrik, Anteil recovered vs degraded, Rollback-Gruende Top-5.

3. Pflichthoertests

ABX fuer Vokalidentitaet, MUSHRA-light fuer natuerlichkeit/artefaktfreiheit/transparenz, mindestens drei Materialklassen pro Release-Kandidat.

## Priorisierte PR-Roadmap

### P1 (sofort)

1. Kurzsignal-Hardening in allen pitch/stft-kritischen Analysepfaden.
2. CSV-Persistenz fuer artifact_freedom und hpi in Klasse-C-Revalidierung vollstaendig schliessen.
3. WP1 auf alle Materialfaelle ausrollen (nicht nur shellac case).

### P2 (kurzfristig)

1. BABE-aehnliche Blind-BWE als streng begrenzter optionaler Rekonstruktionspfad fuer historische Materialien.
2. Diffusion-Inpainting nur fuer echte Gap-Segmente mit SSIP Boundary Guard.
3. Koalitions-Gating fuer gekoppelte Defektfamilien in der Entscheidungslogik absichern.

### P3 (mittelfristig)

1. Vocos-aehnlicher schneller Spektralrekonstruktionspfad fuer geeignete Rekonstruktionsszenarien.
2. Erweiterte no-reference Qualitaetsfusion als Zusatzsignal, nie als primarer Gate-Entscheider.
3. Material-/Era-spezifische Feintuning-Policies fuer Vocal-Recovery.

## Definition of Done fuer Weltklasse-Progress

1. Kein P0-Verstoss in Vocal-Supremacy Guards bei den Referenzfaellen.
2. Artifact-Freedom-Veto bleibt stabil und verhindert ueberprozessierte Exporte.
3. Signifikante Verbesserung in MUSHRA-light ohne Erhoehung der Rollback-Fehlrate.
4. Vollstaendige Reproduzierbarkeit: gleicher Run, gleiche Entscheidung, gleiche Metrik.

## Hinweis zur Evidenz

- Einige angefragte URLs waren fachfremd oder API-seitig verrauscht.
- Dieses Dokument nutzt deshalb nur fachlich passende, audiozentrierte Quellen als operative Basis.
- Neue Schwellenwerte duerfen erst nach klasse-C-Revalidierung und Regressionstest in produktive Defaults uebergehen.
