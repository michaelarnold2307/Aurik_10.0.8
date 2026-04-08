import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure repository root is on sys.path when CLI is executed as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.api.bridge import get_aurik_denker_instance, get_load_audio_fn, run_pre_analysis

_TARGET_SR = 48_000
_VALID_MODES = {"Restoration", "Studio 2026"}


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file via canonical bridge import cascade."""
    try:
        load_audio_file = get_load_audio_fn()
        loaded = load_audio_file(path, target_sr=None, mono=False, do_carrier_analysis=False)
        if not isinstance(loaded, dict) or loaded.get("audio") is None or loaded.get("sr") is None:
            raise RuntimeError(str((loaded or {}).get("error") or "Unbekannter Ladefehler"))
        audio = np.asarray(loaded["audio"], dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        return np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), int(loaded["sr"])
    except Exception as exc:
        raise RuntimeError(f"Audio konnte nicht geladen werden: {exc}") from exc


def _normalize_mode(mode: str) -> str:
    raw = str(mode or "Restoration").strip().lower().replace("_", "").replace(" ", "")
    if raw in {"restoration", "quality"}:
        return "Restoration"
    if raw in {"studio2026", "studio"}:
        return "Studio 2026"
    return "Restoration"


def _rms_dbfs(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))


def _resample_to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 48 kHz if necessary (Lanczos via scipy)."""
    if sr == _TARGET_SR:
        return audio
    try:
        import scipy.signal as _sig

        int(round(audio.shape[0] * _TARGET_SR / sr))
        return _sig.resample_poly(audio, _TARGET_SR, sr, axis=0).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Interne 48-kHz-Normierung fehlgeschlagen. Ursache: Resampling konnte nicht ausgefuehrt werden. "
            "Loesung: scipy/librosa im Bundle sicherstellen oder Eingabedatei vorab auf 48 kHz konvertieren."
        ) from exc


def process_audio(input_path: str, output_path: str, verbose: bool = True, mode: str = "Restoration") -> object:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("aurik_cli")

    mode = _normalize_mode(mode)
    if mode not in _VALID_MODES:
        logger.warning("Unbekannter Modus '%s' — verwende 'Restoration'.", mode)
        mode = "Restoration"

    if not os.path.exists(input_path):
        logger.error("Input-Datei nicht gefunden: %s", input_path)
        sys.exit(2)

    # ── 1. Audio laden ────────────────────────────────────────────────────────
    try:
        audio_raw, sr_raw = _load_audio(input_path)
    except RuntimeError as exc:
        logger.error("Fehler beim Laden der Datei: %s", exc)
        sys.exit(3)

    file_mb = os.path.getsize(input_path) / 1024 / 1024
    if verbose:
        logger.info("Datei: %s  (%.2f MB, %d Hz, %d Kanäle)", input_path, file_mb, sr_raw, audio_raw.shape[1])

    # ── 2. Auf 48 kHz resamplen (Aurik-kanonische SR) ─────────────────────────
    try:
        audio_48k = _resample_to_48k(audio_raw, sr_raw)
    except RuntimeError as exc:
        logger.error("Fehler bei der SR-Normierung: %s", exc)
        sys.exit(6)

    try:
        pre = run_pre_analysis(
            audio_native=audio_raw,
            sr_native=sr_raw,
            audio_48k=audio_48k,
            file_path=input_path,
            store_in_bridge_cache=True,
        )
    except Exception as exc:
        logger.error("Fehler in der Voranalyse: %s", exc)
        sys.exit(10)

    if verbose:
        logger.info("🔧 Starte AurikDenker — Modus: %s", mode)

    # ── 3. Kanonischer Einstiegspunkt: AurikDenker.denke() (Spec §2.2) ────────
    try:
        denker = get_aurik_denker_instance()
        # Quality-first policy: prefer full-quality execution over RT budget cuts.
        result = denker.denke(
            audio_48k,
            sr=_TARGET_SR,
            mode=mode,
            no_rt_limit=True,
            input_path=input_path,
            pre_analysis_result=pre,
        )
    except Exception as exc:
        logger.error("Fehler in der Restaurierungspipeline: %s", exc)
        sys.exit(4)

    if verbose:
        logger.info(
            "✅ Verarbeitung abgeschlossen  ·  Material: %s  ·  Qualität: %.3f  ·  RT-Faktor: %.2f×",
            result.material,
            result.quality_estimate,
            result.rt_factor,
        )
        if result.warnings:
            for w in result.warnings:
                logger.warning("⚠ %s", w)
        if result.processing_note:
            logger.info("ℹ %s", result.processing_note)
        logger.info(
            "🎯 Musical Goals: %d/14 bestanden  ·  Phasen: %d",
            result.goals_passed,
            len(result.phases_executed),
        )

    # ── 4. Export-Quality-Gate (Spec §8.1, [RELEASE_MUST]) ─────────────────────
    # quality_estimate < 0.55 → harter Abbruch (normative E2E-Pflicht)
    _qe = getattr(result, "quality_estimate", None)
    if _qe is not None and _qe < 0.55:
        logger.error(
            "Export abgebrochen: quality_estimate=%.3f < 0.55 (Mindestanforderung §8.1). "
            "Ursache: Restaurierungsqualität unzureichend. "
            "Lösung: Eingabedatei prüfen oder anderen Modus verwenden.",
            _qe,
        )
        sys.exit(7)

    # P1/P2 Musical Goals dürfen nicht unter Schwellwert liegen
    _P1_P2_THRESHOLDS = {
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "tonal_center": 0.95,
        "timbre_authentizitaet": 0.87,
        "artikulation": 0.85,
    }
    _goals = getattr(result, "musical_goals_scores", None) or {}
    _failed_goals = [f"{g}={_goals[g]:.3f}<{t}" for g, t in _P1_P2_THRESHOLDS.items() if g in _goals and _goals[g] < t]
    if _failed_goals:
        logger.error(
            "Export abgebrochen: P1/P2-Musical-Goals nicht bestanden — %s. "
            "Ursache: Restaurierung hat Kernqualitätsziele verfehlt. "
            "Lösung: Material prüfen, anderen Modus verwenden oder Eingabedatei verbessern.",
            ", ".join(_failed_goals),
        )
        sys.exit(8)

    # ── 5. Ergebnis speichern ─────────────────────────────────────────────────
    restored = result.audio
    _in_db = _rms_dbfs(audio_48k)
    _out_db = _rms_dbfs(restored)
    _drop_db = _in_db - _out_db
    if _drop_db > 2.5:
        logger.error(
            "Export abgebrochen: Pegelabfall %.2f dB > 2.50 dB. "
            "Ursache: unzulaessiger Loudness-Drift. "
            "Loesung: Material/Defekte pruefen oder konservativeren Lauf starten.",
            _drop_db,
        )
        sys.exit(9)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, restored, _TARGET_SR, subtype="PCM_24")
    except Exception as exc:
        logger.error("Fehler beim Speichern der Audiodatei: %s", exc)
        sys.exit(5)

    if verbose:
        logger.info("📈 Pegel-Drift: in=%.2f dBFS out=%.2f dBFS delta=%.2f dB", _in_db, _out_db, -_drop_db)
        logger.info("💾 Gespeichert: %s", output_path)

    return result


def print_usage():
    print("\nVerwendung: aurik_cli [--input PATH] [--output PATH] [--mode MODUS] [-q] [-h]")
    print("\nOptionen:")
    print("  --input, --input_audio PATH  Eingabe-Audiodatei")
    print("  --output, --output_audio PATH Ausgabe-Audiodatei")
    print("  --mode MODUS                 Restaurierungsmodus: 'Restoration' (Standard) oder 'Studio 2026'")
    print("  -q, --quiet                  Keine Fortschritts-Ausgaben")
    print("  -h, --help                   Diese Hilfe anzeigen")
    print()


def main():
    args = sys.argv[1:]
    verbose = True
    if "-q" in args or "--quiet" in args:
        verbose = False
        args = [a for a in args if a not in ["-q", "--quiet"]]

    if "-h" in args or "--help" in args:
        print_usage()
        sys.exit(0)

    input_file = None
    output_file = None
    mode = "Restoration"
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--input_audio", "--input"):
            if i + 1 < len(args):
                input_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--input_audio", "--input"):
            input_file = arg.split("=", 1)[1]
        elif arg in ("--output_audio", "--output"):
            if i + 1 < len(args):
                output_file = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] in ("--output_audio", "--output"):
            output_file = arg.split("=", 1)[1]
        elif arg == "--mode":
            if i + 1 < len(args):
                mode = args[i + 1]
                skip_next = True
        elif "=" in arg and arg.split("=", 1)[0] == "--mode":
            mode = arg.split("=", 1)[1]

    # Positional Fallback: nur Nicht-Flag-Argumente verwenden
    positional = [a for a in args if not a.startswith("-")]
    if input_file is None and len(positional) >= 1:
        input_file = positional[0]
    if output_file is None and len(positional) >= 2:
        output_file = positional[1]

    if not input_file or not output_file:
        print("❌ Fehler: Zu wenig oder ungültige Argumente\n")
        print_usage()
        sys.exit(1)

    process_audio(input_file, output_file, verbose=verbose, mode=mode)


if __name__ == "__main__":
    main()
