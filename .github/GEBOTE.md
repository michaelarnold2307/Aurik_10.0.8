# Aurik 10 — GEBOTE & VERBOTE (Normativer Katalog)

> **Status:** Normativ | **Version:** 10.0.7 | **Stand:** 13. Juli 2026 (Update: Lag-Integrität)
>
> Dieser Katalog definiert alle unverhandelbaren GEBOTE (positiv, was Aurik TUN MUSS)
> und VERBOTE (negativ, was Aurik NIEMALS tun darf). Jedes Gebot und Verbot ist mit
> einer eindeutigen ID versehen (§G1, §V1 usw.) und wird im Code per Kommentar
> referenziert. Bei Widerspruch zwischen Specs und diesem Katalog gilt dieser Katalog.

---

## Kategorie I — Individuelle Song-Maximierung (§G1–§G9)

Jeder importierte Song wird individuell maximal für das menschliche Ohr verbessert.

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G1 | **Pro-Song-Kalibrierung** | Jeder Song durchläuft eine vollständige, isolierte SongCalibration (global_scalar, family_scalars, ALLE Guards). Kein Parameter aus einem vorherigen Song darf ungeprüft übernommen werden. |
| §G2 | **Defekt-Vollständigkeit** | Alle 62 DefectTypes werden pro Song gescannt. Defekte werden über die gesamte Songdauer präzise behoben – nicht nur an Stichproben/Checkpoints. |
| §G3 | **Gesangsintegrität** | Gesang darf NIE verzerrt, verschliffen oder mit Artefakten (Ghost-Echo, Phasing) versehen werden. Der Vocal-Safety-Wrapper muss in jeder Phase aktiv sein, die Frequenzen zwischen 80 Hz und 8 kHz bearbeitet. |
| §G4 | **Ghost-Echo-Freiheit** | Kein hörbares Echo oder Pre-Echo durch Phasenverschiebungen, asymmetrische Fensterung oder STFT-Überlappungsartefakte. §2.60 STCG muss in allen Modi laufen. |
| §G5 | **Konsistenz-Mandat** | Alle Maßnahmen müssen über das gesamte Projekt konsistent sein. Kein phasespezifischer Schwellwert ohne zentrale Definition. |
| §G6 | **Null-Toleranz für Phasen-Leckage** | Parameter, Zustände und Circuit-Breaker aus Phase 12, 21, 35, 42 werden pro Song zurückgesetzt (§C3). |
| §G7 | **Interchannel-Lag** | GCC-PHAT-High-Band (§v10.0.4) wird an LAG_PROBE_0B/1/2a/3 gemessen. L/R-Zeitversatz > 50 samples wird vor Phase 1 global korrigiert. Residuale werden von STCG per-Chunk behandelt. |
| §G8 | **CD-Rauschprofil-Pflicht** | Jeder Export (Restoration + Studio 2026) erhält ein CD-charakteristisches Rauschprofil. Das Profil wird NUR dort appliziert, wo es das menschliche Ohr wahrnimmt (psychoakustische Maskierungsschwelle). |
| §G9 | **Quellmaterial-Unabhängigkeit** | Das CD-Rauschprofil wird unabhängig vom Quellmaterial appliziert. Die Charakteristik ist deterministisch und von der CD-Ära (1982–2000) abgeleitet. |

## Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G10 | **ERB-Masking-First** | Jede spektrale Entscheidung muss das ERB-Masking-Modell (Equivalent Rectangular Bandwidth) konsultieren. Kein Gain, kein Filter, kein Dither ohne Masking-Check. |
| §G11 | **Natürlicher Wohlklang** | Das Ziel jedes Processing-Schritts ist der Wohlklang für das menschliche Ohr – nicht mathematische Optimalität. Eine Verschlechterung des PQS-MOS < 3.0 löst Rollback aus. |
| §G12 | **Lautheitskonsistenz** | LUFS-integrated nach EBU R128. Restoration-Ziel: −23 LUFS. Studio-2026-Ziel: −14 LUFS. Kein Hard-Limit ohne ISP-geschützten True-Peak-Limiter. |
| §G13 | **Multi-Point-Lag** | Interchannel-Lag wird an ≥3 Positionen gemessen (Start, Mitte, Ende). Konsistenz-Check: Streuung ≤ 50 samples → globale Korrektur; sonst Median + STCG. |
| §G14 | **Spectral-Tilt-Guard** | Nach jeder Phase wird die spektrale Neigung geprüft. Tilt-Änderung > 1.5 dB/Oktave oder HF-Drop > 3 dB löst Korrektur aus. |
| §G15 | **Rauschprofil-Maskierung** | Das CD-Rauschprofil wird frequenzabhängig und zeitabhängig appliziert. In jedem ERB-Band wird nur dann Rauschen addiert, wenn der Signalpegel unter der simultanen Maskierungsschwelle liegt. |
| §G16 | **Rauschprofil-Charakteristik** | Die Rauschprofil-Charakteristik entspricht einer CD-Neuauflage: −96 dBFS Flat-Noise-Floor (16-bit) mit POW-r-Type-3-Shaping → äquivalente Rauschspannung von −110 dBFS(A) bewertet. |
| §G17 | **Stille-Respekt** | Absolute Stille (digital black) wird NICHT verrauscht. Nur Segmente mit Signalenergie erhalten das Profil. |
| §G18 | **Spektrale Kohärenz** | Frequenzantwort des Rauschprofils folgt dem Langzeit-Leistungsdichtespektrum von CD-Mastern: flach von 20 Hz–16 kHz, −3 dB/Oktave Rolloff ab 16 kHz. |
| §G19 | **Dither-Doppelung-Verbot** | Das CD-Rauschprofil und das Export-Dithering dürfen sich nicht additiv aufschaukeln. Das Rauschprofil wird VOR dem Dithering appliziert; das Dithering berücksichtigt den bereits vorhandenen Rauschpegel. |

## Kategorie III — Architektur & Datenfluss (§G20–§G29)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G20 | **Bridge-Bypass-Verbot** | Kein UI-/Frontend-Code importiert `backend/core/` direkt. Nur über `backend/api/bridge.py`. |
| §G21 | **Denker-Zentralität** | Alle Stärke-Entscheidungen fließen zentral im Denker. Keine dezentralen "Magic Numbers" in Phasen. |
| §G22 | **Determinismus** | Derselbe Input → derselbe Output. Jeder Zufallsgenerator wird mit fixem Seed aus dem Datei-Hash initialisiert. |
| §G23 | **ML-Fallback-Logging** | Jeder ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden. Silent-Failures sind VERBOTEN. |
| §G24 | **NaN/Inf-Schutz** | Jede der 68 Phasen MUSS `np.nan_to_num()` oder `np.isfinite()` auf Ausgabe-Audio anwenden (§0a). |
| §G25 | **Logger-Pflicht** | Jede Python-Datei mit `logger`-Verwendung MUSS `import logging` und `logger = logging.getLogger(__name__)` definieren. |
| §G26 | **Guard-Counter-Lebendigkeit** | Jeder deklarierte Guard-Counter MUSS auch inkrementiert werden. Deklaration ohne `+= 1` ist toter Code. |
| §G27 | **Messschleifen-Plateau** | Jede Messschleife mit ≥3 Kandidaten MUSS Plateau-Erkennung haben. |
| §G28 | **PIM-first, RLP-last** | Vor jedem Phasen-Loop wird PIM berechnet. Nach jedem Loop wird RLP ausgeführt. |
| §G29 | **Artistic Intent vor Defect-Scan** | `get_artistic_intent()` wird VOR dem Defect-Scan aufgerufen. |

## Kategorie IV — CD-Rauschprofil & Export (§G30–§G39)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G30 | **L/R-Unkorreliertheit** | Das Rauschsignal für linken und rechten Kanal MUSS statistisch unabhängig (unkorreliert) sein. Korreliertes Rauschen erzeugt ein hörbares Mono-Rauschzentrum in der Stereomitte — das klingt unnatürlich und ist für CD-Wiedergabe untypisch. |
| §G31 | **Maskierungs-Kanten-Glättung** | An Übergängen zwischen maskierten und unmaskierten Zeit-Frequenz-Regionen MUSS ein 500 ms Cosine-Fade-In/Out erfolgen. Abrupte Rauschpegel-Änderungen sind als "Pumpen" hörbar und verletzen §V1, §V2. |
| §G32 | **ML-Device-Detection** | `next(model.parameters()).device` statt `model.device`. Letzteres ist nach partiellen `.cpu()`/`.to()`-Aufrufen auf Sub-Modulen unzuverlässig und verursacht NaN-Werte auf ROCm. |
| §G33 | **ML-Recovery-API-Äquivalenz** | Recovery-Pfad nach GPU-Fehler MUSS dieselbe API wie der Hauptpfad verwenden (z.B. `model.generate_batch()`), nur mit reduzierten Steps. Niemals komplett andere Funktionssignatur im Retry. |
| §G34 | **Test-Assertion-Konvention** | `np.testing.assert_allclose` nimmt Toleranzen (`rtol`, `atol`). NIEMALS Toleranzen an NumPy-Mathefunktionen übergeben (`np.abs(x, rtol=1e-5)` → `np.abs(x)`). |
| §G35 | **Export-Atomizität** | Jeder Datei-Export MUSS atomar erfolgen: erst in `.tmp`-Datei schreiben, dann `os.replace(tmp, target)`. Bei Abbruch entsteht keine korrupte Datei. |
| §G36 | **True-Peak-Grenze** | Kein Export darf True-Peak > 0 dBTP enthalten. ISP-Interpolation nach ITU-R BS.1770-4 Annex 2 zählt. Oversampling ×4 Minimum. |
| §G37 | **Feedback-Chain-Guards** | Die Feedback-Chain (Phase 12 retry, Phase 35 re-run) MUSS alle Quality-Gates, STCG post-feedbackchain und Spectral-Tilt-Guard durchlaufen. Kein "nackter" Re-Run ohne Guard-Schutz. |
| §G38 | **Modus-Parameter-Isolation** | Parameter eines Modus (Restoration vs. Studio 2026) dürfen nicht in den anderen Modus durchsickern. Die `ProcessingConfig` ist unveränderlich nach Konstruktion; abweichende Parameter werden über `kwargs` nur für den aktuellen Run gesetzt. |
| §G39 | **Rauschprofil-Monitoring** | Jede Rauschprofil-Injektion MUSS im Log vermerken: SNR vorher, SNR nachher, aktive Samples mit Rauschzugabe, maximaler Rauschpegel in dBFS, Onset-Stärke an Übergängen. |

## Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G40 | **Rauschprofil-Zeitpunkt** | Das CD-Rauschprofil wird NACH allen 68 Restaurierungsphasen und VOR dem Dithering appliziert. Dies ist wissenschaftlich der optimale Zeitpunkt: Wird Rauschen früher injiziert, wird es von nachfolgenden Phasen (Denoising, Kompression, EQ) verändert oder verstärkt. Nach der Pipeline ist das Signal stabil und das Rauschen bleibt unverfälscht. |
| §G41 | **Übergangs-Verifikation** | Jeder Übergang zwischen Rauschen und Stille/Musik MUSS verifiziert werden: Die Onset-Stärke (spectral-flux-basiert) darf 0.1 nicht überschreiten. Überschreitung → automatische Verbreiterung des Crossfades auf 500 ms und erneute Prüfung. |
| §G42 | **CD-Produktions-Kohärenz** | Die komplette Export-Kette (Rauschprofil → Dither → Metadaten) MUSS ein Ergebnis liefern, das für einen geschulten Hörer von einer CD-Produktion (1982–2000) nicht unterscheidbar ist. A/B-Blindtest als Validierung. |
| §G43 | **Rauschprofil-Pegel-Anpassung** | Der Rauschpegel passt sich automatisch der Ziel-Bittiefe an: 16-bit → −96 dBFS (CD-Standard), 24-bit → −120 dBFS (Hi-Res-Äquivalent). Kein fester Pegel unabhängig vom Exportformat. |
| §G44 | **Maskierungs-Wissenschaft** | Die Maskierungsschwelle folgt Zwicker & Fastl (1999): −70 dBFS Signalpegel maskiert −96 dBFS breitbandiges Rauschen in ruhiger Umgebung vollständig. Die 50-ms-RMS-Fensterung entspricht der zeitlichen Integration des menschlichen Gehörs. |
| §G45 | **Digital-Black-Integrität** | Exakte Null-Samples (digital black) werden NIE verrauscht — weder durch die Maskierungs-Hüllkurve noch durch Window-Smearing. Sample-genaue Durchsetzung als letzte Verteidigungslinie (§V12). |

---

## VERBOTE — Katalog absoluter Verbote (§V1–§V24)

| ID | Verbot | Beschreibung |
|----|--------|-------------|
| §V1 | **Gesangsverzerrung** | Es ist VERBOTEN, Gesang zu verzerren, zu verschleifen, zu robotisieren oder mit Vocoder-artigen Artefakten zu versehen. |
| §V2 | **Ghost-Echo** | Es ist VERBOTEN, hörbare Echos, Pre-Echos oder Phasing-Artefakte in das restaurierte Signal einzutragen. |
| §V3 | **Hard-Clamp auf Audio** | Es ist VERBOTEN, einen Hard-Clamp (`np.clip(audio, -1, 1)`) ohne Soft-Knee-Übergang (6 dB) auf das finale Audio anzuwenden. |
| §V4 | **Truncation ohne Dither** | Es ist VERBOTEN, Integer-Quantisierung (16-bit, 24-bit) ohne vorheriges Dithering durchzuführen. |
| §V5 | **Dither-Doppelung** | Es ist VERBOTEN, zweimal zu ditheren. Wenn das CD-Rauschprofil bereits appliziert wurde, muss der Dither-Prozess dies berücksichtigen. |
| §V6 | **Silent-Failure** | Es ist VERBOTEN, dass ML→DSP-Fallbacks ohne `logger.warning()` stattfinden. |
| §V7 | **Toter Guard-Code** | Es ist VERBOTEN, einen Guard-Counter zu deklarieren, der nie inkrementiert wird. |
| §V8 | **Globaler Phasen-Zustand** | Es ist VERBOTEN, dass Phasen-Zustände (Circuit-Breaker, Cache, Session-Daten) zwischen verschiedenen Songs persistieren. |
| §V9 | **Workarounds** | Es ist VERBOTEN, Symptome zu umgehen statt Ursachen zu beheben. |
| §V10 | **Phasen-Individuelle Schwellwerte** | Es ist VERBOTEN, Schwellwerte pro Phase zu definieren, die nicht von `global_scalar` oder der zentralen Decision Intelligence abgeleitet sind. |
| §V11 | **Rauschprofil-Flächendeckung** | Es ist VERBOTEN, das CD-Rauschprofil pauschal über den gesamten Song zu legen. Es darf nur dort appliziert werden, wo das menschliche Ohr es wahrnimmt. |
| §V12 | **Stille-Verfälschung** | Es ist VERBOTEN, digital black (absolute Stille) mit Rauschen zu versehen. |
| §V13 | **Spektrale Verfärbung** | Es ist VERBOTEN, das Rauschprofil so zu formen, dass es den spektralen Charakter des Originals verfärbt. Das Profil muss sich unterhalb der Maskierungsschwelle des Signals bewegen. |
| §V14 | **Modus-Ignoranz** | Es ist VERBOTEN, das CD-Rauschprofil nur in einem Modus zu applizieren. Es gilt für Restoration UND Studio 2026. |
| §V15 | **Nicht-deterministisches Rauschen** | Es ist VERBOTEN, nicht-reproduzierbares Rauschen zu verwenden. Der Rauschgenerator wird mit einem deterministischen Seed pro Song initialisiert (SHA256 der ersten 4096 Samples). |
| §V16 | **Übersteuerndes Rauschen** | Es ist VERBOTEN, dass der Rauschpegel −85 dBFS überschreitet. CD-Noise-Floor = −96 dBFS; mit Shaping max. −90 dBFS in den höchsten Bändern. |
| §V17 | **Quellmaterial-Extraktion** | Es ist VERBOTEN, Rauschen aus dem degradierten Quellmaterial zu extrahieren und wieder einzufügen. Das CD-Rauschprofil wird frisch generiert. Quellrauschen ist ein DEFEKT und wird entfernt. |
| §V18 | **Bridge-Bypass** | Es ist VERBOTEN, dass UI-/Frontend-Code `backend/core/` direkt importiert. Nur über `backend/api/bridge.py`. |
| §V19 | **Nicht-atomarer Export** | Es ist VERBOTEN, die Zieldatei direkt zu überschreiben. Export MUSS atomar sein: `.tmp` → `os.replace`. |
| §V20 | **True-Peak-Überschreitung** | Es ist VERBOTEN, dass ein Export True-Peak > 0 dBTP enthält. ISP-Interpolation nach ITU-R BS.1770-4 Annex 2. Oversampling ×4. |
| §V21 | **ML-Device-Fehlgriff** | Es ist VERBOTEN, `model.device` nach `.cpu()`/`.to()` auf Sub-Modulen zu verwenden. Statthaft: `next(model.parameters()).device`. |
| §V22 | **ML-Recovery-Signaturbruch** | Es ist VERBOTEN, im Recovery-Pfad eine komplett andere API-Signatur zu verwenden. Dieselbe Methode, reduzierte Steps. |
| §V23 | **Diffusionsmodell-Rauschen** | Es ist VERBOTEN, dass Diffusionsmodell-Artefakte im Noise Floor unerkannt bleiben. Der Authenticity-Validator MUSS sie als Artefakt markieren. |
| §V24 | **Falsche Test-Toleranzen** | Es ist VERBOTEN, Toleranzen an NumPy-Mathefunktionen zu übergeben (`np.abs(x, rtol=1e-5)` ist FALSCH). Statthaft: `np.testing.assert_allclose(actual, desired, rtol=...)`. |
| §V25 | **Zwischenphasen-Rauschen** | Es ist VERBOTEN, das CD-Rauschprofil VOR Abschluss aller 68 Restaurierungsphasen zu injizieren. Frühe Injektion führt zu unkontrollierbarer Verstärkung/Modifikation durch nachfolgende Phasen (§G40). |
| §V26 | **Hörbare Übergänge** | Es ist VERBOTEN, dass Übergänge an Rauschprofil-Kanten hörbar sind. Die Onset-Stärke (spectral-flux-basiert) muss < 0.1 sein. Überschreitung → Crossfade-Verbreiterung (§G41). |

---

## Referenz-System

Jedes Gebot und Verbot wird im Code als Kommentar referenziert:

```python
# §G8: CD-Rauschprofil-Pflicht — Rauschen nur unterhalb der Maskierungsschwelle
# §V11: Rauschprofil-Flächendeckung verboten
audio = _apply_cd_noise_profile(audio, sr, mask=erb_mask)
```

**ID-Konventionen:**
- `§G1`–`§G99`: GEBOTE (positiv, was getan werden MUSS)
- `§V1`–`§V99`: VERBOTE (negativ, was NIEMALS getan werden DARF)
- `§C1`–`§C99`: Circuit-Breaker / Schutzschaltungen
- `§F1`–`§F99`: Forensische Regeln
- `§D1`–`§D99`: DSP-Regeln

**Prioritäten:**
- Kategorie I (§G1–§G9): Höchste Priorität — Song-Individualität
- Kategorie II (§G10–§G19): Zweithöchste — Psychoakustik
- Kategorie III (§G20–§G29): Architektur-Invarianten
- Kategorie IV (§G30–§G39): CD-Rauschprofil & Export
- Kategorie V (§G40–§G45): Rauschprofil-Zeitpunkt & Übergänge
- Kategorie VI (§G46–§G59): Metriken & Qualitätssicherung
- VERBOTE (§V1–§V26): Absolute Verbote, gelten immer und überall

---

## Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G46 | **Harmonic Preservation Score** | HNR-basierte Metrik. Detektiert Obertonschäden durch Überglättung. |
| §G47 | **Transient Preservation Score** | Crest-Faktor + Onset-Positionsabgleich. Detektiert Transienten-Verschleifung. |
| §G48 | **Formant Preservation Score** | Cepstrale Hüllkurvendistanz. Detektiert Vokalcharakter-Änderungen. |
| §G49 | **ABX Test Harness** | Double-Blind A/B/X mit Binomial-Signifikanztest. |
| §G50 | **MUSHRA Proxy Scorer** | 6-Dimensionen-Ensemble 0–100 Skala. |
| §G51 | **Statistical Report** | Binomialtest für Listening-Panel-Signifikanz. |
| §G52 | **Micro-Dynamics Score** | Crest-Faktor-Verteilung in 200ms-Fenstern. |
| §G53 | **Artifact Detector** | Clicks, Spectral Holes, Pre-Echo, Stereo-Anomalien. |
| §G54 | **Emotional Arc Score** | Lautheitskontur + Sektionskontrast + Spektralbewegung + Stille. |
| §G55 | **Blind Reference-Free Quality** | 6 Single-Ended-Features. Bewertet ohne Originalvergleich. |
| §G56 | **Noise Floor Continuity** | −20 dB Minimum-Floor. Verhindert Noise-Gate-Artefakte. |
| §G57 | **Sliding ERB Gain** | Multi-Segment-ERB-Maske. Adaptiert an spektrale Änderungen. |
| §G58 | **Vocal Repair Module** | Bandbreiten-Erweiterung + Verzerrungs-Reparatur vor Phase 42. |
| §G59 | **Restoration Quality Report** | Integriert alle Metriken in einen Aufruf. Blindtest-Readiness-Verdikt. |

---

## Kategorie VII — Stereo-Lag-Integrität (§G60–§G67)

> **Alle Erkenntnisse aus der Lag-Root-Cause-Analyse vom 2026-07-13.**
> 13 Commits, 8 Root Causes identifiziert und behoben.

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G60 | **STCG Multi-Point-Primär** | STCG MUSS Multi-Point-GCC-PHAT (≥3 Song-Positionen, Median) als PRIMÄRE Messmethode verwenden. Single-Mid-Window nur als Fallback bei Audio < 30s. |
| §G61 | **Chunk-Phasen-STCG-Pflicht** | Jede Chunk-basierte Phase (Phase 12, Phase 24 u.a.) MUSS für Lag-Erkennung und -Korrektur den zentralen STCG verwenden. Eigene Korrelations-Implementierungen (signal.correlate) sind VERBOTEN (§V27). |
| §G62 | **Sub-Sample-Lag-Korrektur** | Lag-Korrektur MUSS `scipy.ndimage.shift` (cubic spline, Sub-Sample-Präzision) oder STCG direkt verwenden. `np.roll` (zirkulär), `np.concatenate` (ganzzahlig), und Audio-Trunkierung sind VERBOTEN (§V32). |
| §G63 | **Lag-Messung-Orientierungsfrei** | Alle Lag-Messfunktionen MÜSSEN sowohl channels-first `(2, N)` als auch channels-last `(N, 2)` korrekt erkennen und messen. `arr.shape[0]` ohne Orientierungs-Check ist VERBOTEN (§V33). |
| §G64 | **STCG-Singleton-Konsistenz** | Alle Lag-Korrekturen MÜSSEN den zentralen STCG-Singleton verwenden. Keine ad-hoc GCC-PHAT-Reimplementierung in einzelnen Phasen. |
| §G65 | **Post-Chunk-Global-STCG** | Nach ABSCHLUSS aller Chunk-basierten Phasen MUSS ein globaler STCG-Check mit Multi-Point-Verifikation erfolgen. Per-Chunk-Korrekturen ohne globalen Abschluss sind VERBOTEN (§V28). |
| §G66 | **Keine konkurrierenden Lag-Fixes** | Nach einer erfolgreichen STCG-Korrektur darf KEINE zweite, unabhängige Lag-"Korrektur" (Onset-Energy-Fallback, manuelle np.concat) durchgeführt werden (§V29). Nur bei STCG-Fehlschlag ist ein Fallback erlaubt. |
| §G67 | **STFT-Input-Length-Guard** | Jeder Aufruf von `scipy.signal.stft` MUSS durch einen zentralen Längen-Guard geschützt sein, der `nperseg > input_length` abfängt. Der Guard ist in `backend/__init__.py` installiert. |

## Kategorie VIII — Neue VERBOTE Stereo-Lag (§V27–§V33)

| ID | Verbot | Beschreibung |
|----|--------|-------------|
| §V27 | **Kein signal.correlate für Lag** | Es ist VERBOTEN, `scipy.signal.correlate` (Standard-Kreuzkorrelation ohne PHAT-Whitening) für Stereo-Lag-Messung zu verwenden. Nur GCC-PHAT (via STCG) ist statthaft. |
| §V28 | **Kein begrenzter Lag-Suchraum** | Es ist VERBOTEN, den Lag-Suchraum für Stereo-Messungen auf < ±200ms (±9600 samples @48kHz) zu begrenzen. Kleinere Limits (z.B. 960 samples = 20ms) verfehlen echte Kanalversätze. |
| §V29 | **Keine konkurrierenden Lag-Korrekturen** | Es ist VERBOTEN, nach erfolgreicher STCG-Korrektur eine zweite Lag-"Korrektur" durchzuführen. Der Onset-Energy-Fallback in `_preserve_phase_loudness` ist NUR bei STCG-Exception aktiv. |
| §V30 | **Kein Single-Window-Lag** | Es ist VERBOTEN, Stereo-Lag nur an EINER Song-Position (z.B. Mid-Window 10s) zu messen, wenn die Song-Dauer > 30s beträgt. Multi-Point (≥3 Positionen) ist Pflicht. |
| §V31 | **Kein np.roll für Lag-Korrektur** | Es ist VERBOTEN, `np.roll` (zirkuläre Verschiebung mit Sample-Wrapping) für Stereo-Lag-Korrektur zu verwenden. Nur `scipy.ndimage.shift` (Zero-Padding, Sub-Sample) oder STCG sind statthaft. |
| §V32 | **Kein Audio-Trunkieren für Lag** | Es ist VERBOTEN, Audio zu trunkieren (`audio[:, :N - lag]`), um Lag zu korrigieren. Die Korrektur MUSS die Originallänge durch Zero-Padding erhalten. |
| §V33 | **Kein shape[0] ohne Orientierungs-Check** | Es ist VERBOTEN, `audio.shape[0]` als Sample-Anzahl zu interpretieren, ohne vorher zu prüfen ob `(2,N)` oder `(N,2)` vorliegt. Die Multi-Point-Funktion MUSS beide Orientierungen unterstützen. |

---

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 10.0.7 | 2026-07-13 | §G60–§G67 + §V27–§V33. Lag-Integritäts-Architektur nach Root-Cause-Analyse (8 Bugs, 13 Commits). Kategorie VII + VIII. |
| 10.0.6 | 2026-07-13 | §G46–§G59 (Metriken & Qualitätssicherung). Kategorie VI. |
| 10.0.5 | 2026-07-13 | §G30–§G39 (CD-Rauschprofil & Export, ML-Device, Test-Assertion). §V16–§V24. |
| 10.0.4 | 2026-07-13 | Initiale Formalisierung. CD-Rauschprofil (§G8, §G15–§G19, §V5, §V11–§V15). Kategorie I–III strukturiert. |
