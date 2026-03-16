# Security Policy — Aurik 9

## Unterstützte Versionen

| Version | Status |
|---------|--------|
| 9.10.x  | ✅ Unterstützt (aktiv) |
| < 9.10  | ❌ Keine Sicherheits-Updates |

## Sicherheitslücken melden

**Bitte melde Sicherheitslücken NICHT als öffentliches GitHub-Issue.**

Sende stattdessen eine E-Mail an das Sicherheitsteam mit:

- Beschreibung der Schwachstelle
- Schritte zur Reproduktion
- Mögliche Auswirkungen
- Vorschlag zur Behebung (falls vorhanden)

Wir bestätigen den Eingang innerhalb von **72 Stunden** und bemühen uns um
eine Behebung innerhalb von **14 Tagen** für kritische Schwachstellen.

## Sicherheitsarchitektur

Aurik 9 verarbeitet lokale Audio-Dateien auf dem Desktop-PC des Nutzers.
Folgende Sicherheitsprinzipien sind implementiert:

### Input-Validierung (OWASP A03 — Injection)
- Alle Eingabedateien werden vor der Verarbeitung validiert (`AudioFileValidator`)
- Dateigröße-Limit: max. 10 GB
- Magic-Bytes-Verifikation (keine Extension-Spoofing-Angriffe)
- FFmpeg-Aufrufe immer als Liste (kein Shell-Injection via Dateinamen)
- Pfad-Traversal ausgeschlossen: `os.path.realpath()` vor jedem Dateizugriff

### Keine Netzwerkabhängigkeiten
- Aurik 9 ist vollständig offline-fähig nach der Installation
- Kein Cloud-Aufruf, kein Telemetrie, keine externe API
- Alle ML-Modelle sind lokal gebündelt (SHA256-verifiziert)

### Lokale Datenspeicherung
- GP-Gedächtnis, Artist-Signaturen und Sessions werden ausschließlich lokal
  unter `~/.aurik/` gespeichert — niemals übertragen
- Genealogie-Logs enthalten keine Audio-Rohdaten

### Abhängigkeiten
- Regelmäßige `pip-audit`-Checks in der CI-Pipeline
- Verwendete ML-Modelle sind SHA256-verifiziert (vgl. `models/manifest.json`)
- CPU-only: kein Treiber-Stack (kein CUDA, kein ROCm)

## Bekannte Einschränkungen

- Aurik 9 vertraut darauf, dass die verarbeiteten Audio-Dateien vom Nutzer
  selbst stammen oder zur Verarbeitung berechtigt sind.
- Das Projekt enthält keine Zugriffskontrolle für Mehrbenutzer-Szenarien
  (Desktop-Einzelplatz-Anwendung).
