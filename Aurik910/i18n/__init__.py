"""
Aurik910/i18n/__init__.py — Internationalisierungs-Framework für Aurik 9 (§3.5).

Aurik 9 zeigt alle UI-Texte standardmäßig auf Deutsch.
Umschaltung auf Englisch via Einstellungen → Sprache (zur Laufzeit).

Verwendung:
    from Aurik910.i18n import t, set_language
    set_language("de")
    print(t("restoration.started"))  # → "Restaurierung gestartet…"
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_language: str = "de"


def set_language(lang: str) -> None:
    """Setzt die aktive UI-Sprache ('de' oder 'en').

    Args:
        lang: ISO-639-1-Sprachcode. Unbekannte Codes → Fallback 'de'.
    """
    global _language
    with _lock:
        _language = lang if lang in _TRANSLATIONS else "de"


def get_language() -> str:
    """Gibt den aktuellen Sprachcode zurück."""
    return _language


def t(key: str, **kwargs: Any) -> str:
    """Gibt die übersetzte Zeichenkette für den gegebenen Schlüssel zurück.

    Unbekannte Schlüssel → Schlüssel selbst (kein Absturz).
    Variablen-Interpolation via {var}-Syntax.

    Args:
        key: Punkt-getrennter Schlüssel (z. B. "restoration.started").
        **kwargs: Optionale Format-Variablen.

    Returns:
        Übersetzte Zeichenkette, ggf. mit eingesetzten Variablen.
    """
    lang_dict = _TRANSLATIONS.get(_language, _TRANSLATIONS["de"])
    text = lang_dict.get(key) or _TRANSLATIONS["de"].get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


# ---------------------------------------------------------------------------
# Übersetzungs-Tabelle
# ---------------------------------------------------------------------------

_TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── Deutsch (Primärsprache) ──────────────────────────────────────────────
    "de": {
        # Allgemein
        "app.name": "Aurik 9",
        "app.tagline": "Intelligentes Musik-Restaurierungssystem",
        "action.open_file": "Datei öffnen",
        "action.export": "Exportieren",
        "action.restore_restoration": "Restaurierung starten",
        "action.restore_studio": "Studio 2026 starten",
        "action.preview": "Vorschau anzeigen",
        "action.cancel": "Abbrechen",
        "action.play": "Wiedergabe",
        "action.pause": "Pause",
        "action.listen_original": "Original hören",
        "action.listen_restored": "Restauriert hören",
        "action.stop": "Stopp",
        # Status
        "status.analyzing": "Analysiere Aufnahme…",
        "status.restoring": "Restaurierung läuft ({percent:.0f} %)…",
        "status.phase_active": "Verarbeitungsschritt: {phase}",
        "status.done": "Fertig — Ergebnis bereit.",
        "status.exporting": "Exportiere Ergebnis…",
        "status.cancelled": "Verarbeitung abgebrochen.",
        "status.ready": "Bereit für Verarbeitung",
        "status.export_finished": "Export abgeschlossen",
        "status.settings_saved": "Einstellungen gespeichert",
        "status.stats": "Queue: {pending} | Verarbeitet: {completed} | Fehler: {failed}",
        "status.processing_cancelled": "⏹ Verarbeitung wurde abgebrochen.",
        "status.no_export_path": "⚠️ Noch kein Export-Pfad vorhanden.",
        "status.import_cancelled": "❌  Import abgebrochen.",
        "status.defects_analyzing": "🔍  Schäden werden analysiert …",
        "status.ready_to_restore": "✅  Bereit zur Restaurierung",
        "status.queue_cleared": "Queue geleert",
        "status.lyrics_overlay_hidden": "🎵 Lyrics-Timeline-Overlay ausgeblendet",
        "status.lyrics_load_file_first": "🎵 Lyrics-Timeline: Bitte zuerst eine Audiodatei laden.",
        "status.lyrics_transcribing": "🎵 Lyrics-Timeline-Overlay: Transkription läuft …",
        "status.lyrics_unavailable": "🎵 Lyrics-Timeline: Transkription nicht verfügbar.",
        # UI / Tabs
        "ui.tab_waveform": "Wellenform",
        "ui.tab_spectrogram": "Spektrogramm",
        "ui.ab_compare": "🎧  Vor / Nachher Vergleich",
        "ui.no_file_loaded": "Keine Datei geladen",
        "ui.no_analysis": "Noch keine Analyse",
        # Dialoge
        "dialog.no_file_title": "❌ Keine Datei",
        "dialog.no_file_body": "Bitte laden Sie zuerst eine Audio-Datei mit dem Button oben!",
        "dialog.processing_error_title": "Verarbeitungsfehler",
        "dialog.processing_error_body": "Die Verarbeitung konnte nicht gestartet werden:\n\n{error}\n\nBitte versuchen Sie es erneut oder laden Sie eine andere Datei.",
        "dialog.invalid_file_title": "Datei ungültig",
        "dialog.invalid_file_body": "Diese Datei kann nicht geladen werden:\n\n{error}",
        "dialog.import_failed_title": "Import fehlgeschlagen",
        "dialog.import_failed_body": "Die Datei »{file}« konnte nicht geladen werden.\n\nUnterstützte Formate: WAV, FLAC, OGG, AIFF, MP3, M4A, WMA, AAC\n\nDetails: {error}",
        "dialog.player_title": "Player",
        "dialog.player_body": "Für die Vorschau bitte 'sounddevice' installieren:\n  pip install sounddevice",
        "dialog.album_import_title": "Album-Import",
        "dialog.album_import_no_files": "Im Ordner '{folder}' wurden keine Audiodateien gefunden.\nUnterstützt: wav, mp3, flac, ogg, aiff, m4a, wma",
        "dialog.no_files_title": "Keine Dateien",
        "dialog.no_files_body": "Bitte fügen Sie zuerst Dateien zur Queue hinzu.",
        "dialog.processing_running_title": "Verarbeitung läuft",
        "dialog.processing_running_body": "Verarbeitung läuft bereits!",
        "dialog.no_pending_title": "Keine ausstehenden Dateien",
        "dialog.no_pending_body": "Alle Dateien wurden bereits verarbeitet.",
        "dialog.low_ram_title": "Zu wenig Arbeitsspeicher",
        "dialog.low_ram_body": "Es stehen nur {avail} GB freier RAM zur Verfügung.\n\nAurik benötigt mindestens 6 GB freien RAM für die Restaurierung.\n\nBitte schließen Sie andere Programme (Browser, VS Code …) und versuchen Sie es erneut.",
        "dialog.timeout_title": "Zeitüberschreitung",
        "dialog.timeout_body": "Die Verarbeitung hat das Zeitlimit überschritten und wurde abgebrochen.\n\nBitte starten Sie Aurik neu und versuchen Sie es erneut.",
        "dialog.queue_busy_title": "Verarbeitung läuft",
        "dialog.queue_busy_body": "Queue kann nicht geleert werden während Verarbeitung läuft.",
        "dialog.no_processed_title": "Keine verarbeiteten Dateien",
        "dialog.no_processed_body": "Es wurden noch keine Dateien verarbeitet.\nBitte zuerst eine Datei restaurieren, dann exportieren.",
        # Materialerkennung
        "material.tape": "Kassette",
        "material.reel_tape": "Spulenband",
        "material.vinyl": "Schallplatte (Vinyl)",
        "material.shellac": "Schellack",
        "material.wax_cylinder": "Wachswalze",
        "material.lacquer_disc": "Lackscheibe",
        "material.wire_recording": "Drahtbandaufnahme",
        "material.dat": "Digital Audio Tape",
        "material.cd_digital": "CD / Digital",
        "material.mp3_low": "MP3 (niedrige Qualität)",
        "material.mp3_high": "MP3 (hohe Qualität)",
        "material.aac": "AAC / M4A",
        "material.minidisc": "MiniDisc",
        "material.streaming": "Streaming-Kopie",
        "material.unknown": "Unbekanntes Material",
        # Fehler
        "error.file_unreadable": "Diese Datei kann nicht geöffnet werden. Möglicherweise ist sie beschädigt oder das Format wird nicht unterstützt.",
        "error.export_failed": "Die Datei konnte nicht gespeichert werden. Bitte prüfen Sie, ob genügend Speicherplatz vorhanden ist.",
        "error.model_unavailable": "Die KI-Unterstützung ist gerade nicht verfügbar. Die klassische Methode wird genutzt — das Ergebnis ist trotzdem sehr gut.",
        "error.ram_low": "Die Datei ist sehr groß. Verarbeitung wird in Abschnitten durchgeführt — das dauert etwas länger.",
        "error.musical_goal_regression": "Die Restaurierung hat die Klangqualität in einem Bereich verschlechtert. Das beste Zwischenergebnis wird verwendet.",
        # Qualitätsanzeigen
        "quality.excellent": "Exzellent restaurierbar — fast wie Neuaufnahme erwartet.",
        "quality.good": "Gut restaurierbar — deutliche Verbesserung erwartet.",
        "quality.fair": "Mäßig restaurierbar — Restdefekte werden bleiben.",
        "quality.poor": "Schwierig restaurierbar — Ergebnis besser als Original, aber begrenzt.",
        "quality.very_poor": "Sehr schwer restaurierbar — das Material ist stark beschädigt.",
        # Genre-Erkennung
        "genre.schlager_detected": "Deutscher Schlager erkannt — Akkordeon-Klangcharakter und Schunkelrhythmus werden sorgfältig bewahrt.",
        # Einstellungen
        "settings.language": "Sprache",
        "settings.language_de": "Deutsch",
        "settings.language_en": "English",
        "settings.title": "⚙️ Einstellungen",
        "settings.default_export_format": "Standard-Export-Format",
        "settings.default_mode_batch_album": "Standard-Modus für Batch / Album",
        "settings.artist_learning": "Künstler-Lernmodus aktivieren",
        "settings.session_learning": "Sitzungs-Lernmodus aktivieren",
        # Physikalische Grenzen
        "ceiling.reached": "Das Beste aus dieser Aufnahme wurde herausgeholt — die physikalischen Grenzen des Quellmaterials sind erreicht.",
        "ceiling.adaptive": "Zielvorgaben wurden an den Zustand der Aufnahme angepasst.",
        # Lyrics
        "lyrics.active": "Lautanalyse aktiv — farbige Markierungen zeigen, wie verschiedene Klangtypen im Gesang individuell behandelt werden.",
        # PMGG
        "pmgg.rollback_warning": "Einige Verarbeitungsschritte wurden angepasst, um den Klang zu schützen.",
    },
    # ── English (Sekundärsprache) ────────────────────────────────────────────
    "en": {
        # General
        "app.name": "Aurik 9",
        "app.tagline": "Intelligent Music Restoration System",
        "action.open_file": "Open File",
        "action.export": "Export",
        "action.restore_restoration": "Start Restoration",
        "action.restore_studio": "Start Studio 2026",
        "action.preview": "Show Preview",
        "action.cancel": "Cancel",
        "action.play": "Play",
        "action.pause": "Pause",
        "action.listen_original": "Listen to Original",
        "action.listen_restored": "Listen to Restored",
        "action.stop": "Stop",
        # Status
        "status.analyzing": "Analyzing recording…",
        "status.restoring": "Restoration in progress ({percent:.0f} %)…",
        "status.phase_active": "Processing step: {phase}",
        "status.done": "Done — result ready.",
        "status.exporting": "Exporting result…",
        "status.cancelled": "Processing cancelled.",
        "status.ready": "Ready for processing",
        "status.export_finished": "Export finished",
        "status.settings_saved": "Settings saved",
        "status.stats": "Queue: {pending} | Processed: {completed} | Errors: {failed}",
        "status.processing_cancelled": "⏹ Processing was cancelled.",
        "status.no_export_path": "⚠️ No export path available yet.",
        "status.import_cancelled": "❌  Import cancelled.",
        "status.defects_analyzing": "🔍  Analyzing defects …",
        "status.ready_to_restore": "✅  Ready to restore",
        "status.queue_cleared": "Queue cleared",
        "status.lyrics_overlay_hidden": "🎵 Lyrics timeline overlay hidden",
        "status.lyrics_load_file_first": "🎵 Lyrics timeline: Please load an audio file first.",
        "status.lyrics_transcribing": "🎵 Lyrics timeline overlay: Transcribing …",
        "status.lyrics_unavailable": "🎵 Lyrics timeline: Transcription unavailable.",
        # UI / Tabs
        "ui.tab_waveform": "Waveform",
        "ui.tab_spectrogram": "Spectrogram",
        "ui.ab_compare": "🎧  Before / After Comparison",
        "ui.no_file_loaded": "No file loaded",
        "ui.no_analysis": "No analysis yet",
        # Dialogs
        "dialog.no_file_title": "❌ No file",
        "dialog.no_file_body": "Please load an audio file first using the button above!",
        "dialog.processing_error_title": "Processing error",
        "dialog.processing_error_body": "Processing could not be started:\n\n{error}\n\nPlease try again or load another file.",
        "dialog.invalid_file_title": "Invalid file",
        "dialog.invalid_file_body": "This file cannot be loaded:\n\n{error}",
        "dialog.import_failed_title": "Import failed",
        "dialog.import_failed_body": "The file '{file}' could not be loaded.\n\nSupported formats: WAV, FLAC, OGG, AIFF, MP3, M4A, WMA, AAC\n\nDetails: {error}",
        "dialog.player_title": "Player",
        "dialog.player_body": "For preview, please install 'sounddevice':\n  pip install sounddevice",
        "dialog.album_import_title": "Album import",
        "dialog.album_import_no_files": "No audio files were found in folder '{folder}'.\nSupported: wav, mp3, flac, ogg, aiff, m4a, wma",
        "dialog.no_files_title": "No files",
        "dialog.no_files_body": "Please add files to the queue first.",
        "dialog.processing_running_title": "Processing in progress",
        "dialog.processing_running_body": "Processing is already running!",
        "dialog.no_pending_title": "No pending files",
        "dialog.no_pending_body": "All files have already been processed.",
        "dialog.low_ram_title": "Insufficient memory",
        "dialog.low_ram_body": "Only {avail} GB of free RAM is available.\n\nAurik requires at least 6 GB of free RAM for restoration.\n\nPlease close other applications (browser, VS Code …) and try again.",
        "dialog.timeout_title": "Timeout",
        "dialog.timeout_body": "Processing exceeded the time limit and was aborted.\n\nPlease restart Aurik and try again.",
        "dialog.queue_busy_title": "Processing in progress",
        "dialog.queue_busy_body": "Queue cannot be cleared while processing is running.",
        "dialog.no_processed_title": "No processed files",
        "dialog.no_processed_body": "No files have been processed yet.\nPlease restore a file first, then export.",
        # Material
        "material.tape": "Cassette Tape",
        "material.reel_tape": "Reel-to-Reel Tape",
        "material.vinyl": "Vinyl Record",
        "material.shellac": "Shellac",
        "material.wax_cylinder": "Wax Cylinder",
        "material.lacquer_disc": "Lacquer Disc",
        "material.wire_recording": "Wire Recording",
        "material.dat": "Digital Audio Tape (DAT)",
        "material.cd_digital": "CD / Digital",
        "material.mp3_low": "MP3 (low quality)",
        "material.mp3_high": "MP3 (high quality)",
        "material.aac": "AAC / M4A",
        "material.minidisc": "MiniDisc",
        "material.streaming": "Streaming Copy",
        "material.unknown": "Unknown Material",
        # Errors
        "error.file_unreadable": "This file cannot be opened. It may be corrupted or the format is not supported.",
        "error.export_failed": "The file could not be saved. Please check if there is enough disk space.",
        "error.model_unavailable": "AI support is currently unavailable. The classical method is used — the result is still very good.",
        "error.ml_model_unavailable": "AI support is currently unavailable. The classical method is used — the result is still very good.",
        "error.musical_goal_regression": "Restoration degraded audio quality in one area. The best intermediate result is used.",
        "error.ram_low": "The file is very large. Processing will be done in sections — this will take a little longer.",
        # Quality
        "quality.excellent": "Excellently restorable — almost like a new recording expected.",
        "quality.good": "Well restorable — significant improvement expected.",
        "quality.fair": "Moderately restorable — some residual defects will remain.",
        "quality.poor": "Difficult to restore — result better than original, but limited.",
        "quality.very_poor": "Very difficult to restore — the material is severely damaged.",
        # Genre
        "genre.schlager_detected": "German Schlager detected — accordion timbre and rhythm will be carefully preserved.",
        # Settings
        "settings.language": "Language",
        "settings.language_de": "Deutsch",
        "settings.language_en": "English",
        "settings.title": "⚙️ Settings",
        "settings.default_export_format": "Default export format",
        "settings.default_mode_batch_album": "Default mode for batch / album",
        "settings.artist_learning": "Enable Artist Learning Mode",
        "settings.session_learning": "Enable Session Learning Mode",
        # Misc
        "ceiling.reached": "The best possible result for this recording has been achieved — the physical limits of the source material have been reached.",
        "ceiling.adaptive": "Quality targets have been adjusted to match the condition of the recording.",
        "pmgg.rollback_warning": "Some processing steps were adjusted to protect the audio quality.",
        "lyrics.active": "Sound analysis active — colour markings show how different sound types in the vocals are treated individually.",
    },
}
