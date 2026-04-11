#!/usr/bin/env python3
"""
_aurik_run_excellence.py — Aurik maximale Exzellenz-Restaurierung
==================================================================
Lädt die MP3 aus dem Hauptverzeichnis, restauriert mit mode="maximum"
(Studio 2026 / maximal mögliche Exzellenz) und exportiert nach output/.

Nutzt kanonischen Einstieg AurikDenker.denke() — kein UV3-Bypass.

Aufruf: .venv_aurik/bin/python _aurik_run_excellence.py [datei.mp3]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# ─── Logging konfigurieren ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("aurik_run")

sys.path.insert(0, str(Path(__file__).parent))

# ─── Konstanten ─────────────────────────────────────────────────────────────
TARGET_SR: int = 48_000
MODE: str = "maximum"  # 8× RT-Budget, maximalste Exzellenz
OUTPUT_DIR: Path = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Dreistufige Audio-Lade-Kaskade (Spec §11.4)."""
    # Stufe 1: soundfile
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.T  # (channels, samples)
        logger.info("Audio geladen via soundfile: %s Hz, %d Kanäle, %.2f s", sr, audio.shape[0], audio.shape[1] / sr)
        return audio, sr
    except Exception as e1:
        logger.debug("soundfile fehlgeschlagen: %s", e1)

    # Stufe 2: pedalboard
    try:
        from pedalboard.io import AudioFile

        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)  # (channels, samples) float32
            sr = f.samplerate
        logger.info("Audio geladen via pedalboard: %s Hz, %d Kanäle, %.2f s", sr, audio.shape[0], audio.shape[1] / sr)
        return audio, sr
    except Exception as e2:
        logger.debug("pedalboard fehlgeschlagen: %s", e2)

    # Stufe 3: librosa
    import librosa

    audio_mono, sr = librosa.load(str(path), sr=None, mono=False, dtype=np.float32)
    if audio_mono.ndim == 1:
        audio_mono = audio_mono[np.newaxis, :]
    logger.info(
        "Audio geladen via librosa: %s Hz, %d Kanäle, %.2f s", sr, audio_mono.shape[0], audio_mono.shape[1] / sr
    )
    return audio_mono, sr


def _resample_to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resampling auf 48 kHz wenn nötig (Lanczos via soxr/librosa)."""
    if sr == TARGET_SR:
        return audio
    try:
        import soxr

        if audio.ndim == 1:
            resampled = soxr.resample(audio, sr, TARGET_SR, quality="VHQ")
        else:
            resampled = np.stack([soxr.resample(ch, sr, TARGET_SR, quality="VHQ") for ch in audio])
        logger.info("Resampling %d → %d Hz via soxr VHQ", sr, TARGET_SR)
        return resampled
    except ImportError:
        pass
    import librosa

    if audio.ndim == 1:
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_best")
    else:
        resampled = np.stack(
            [librosa.resample(ch, orig_sr=sr, target_sr=TARGET_SR, res_type="kaiser_best") for ch in audio]
        )
    logger.info("Resampling %d → %d Hz via librosa", sr, TARGET_SR)
    return resampled


def _to_pipeline_format(audio: np.ndarray) -> np.ndarray:
    """Pipeline-Format: (samples,) für Mono, (samples, 2) für Stereo.
    AurikDenker erwartet last-dim=channels oder 1D."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.shape[0] <= 2:
        # (channels, samples) → (samples, channels)
        return np.ascontiguousarray(audio.T, dtype=np.float32)
    # Bereits (samples, channels)
    return audio.astype(np.float32)


def _progress_cb(pct: int, msg: str, elapsed_s: float = 0.0) -> None:
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    eta = ""
    if pct > 5 and elapsed_s > 0:
        total_est = elapsed_s / (pct / 100)
        remaining = total_est - elapsed_s
        _m, _s = divmod(int(remaining), 60)
        eta = f"  noch ca. {_m}:{_s:02d}" if _m > 0 else f"  noch ca. {_s}s"
    print(f"\r[{bar}] {pct:3d}%{eta}  {msg[:60]}", end="", flush=True)
    if pct >= 100:
        print()


def main() -> int:
    # ── Datei bestimmen ──────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Erste MP3 im Hauptverzeichnis
        mp3s = sorted(Path(".").glob("*.mp3")) + sorted(Path(".").glob("*.MP3"))
        if not mp3s:
            logger.error("Keine MP3-Datei im Hauptverzeichnis gefunden.")
            return 1
        input_path = mp3s[0]

    if not input_path.exists():
        logger.error("Datei nicht gefunden: %s", input_path)
        return 1

    logger.info("═" * 70)
    logger.info("Aurik 9 — Maximale Exzellenz-Restaurierung")
    logger.info("Eingabe : %s", input_path.name)
    logger.info("Modus   : %s  (8× RT-Budget)", MODE)
    logger.info("═" * 70)

    # ── Audio laden und vorbereiten ──────────────────────────────────────────
    audio_raw, sr_orig = _load_audio(input_path)
    audio_48k = _resample_to_48k(audio_raw, sr_orig)
    audio_in = _to_pipeline_format(audio_48k)

    dur_s = audio_in.shape[0] / TARGET_SR
    logger.info("Eingabe-Signal: %.2f s (%.1f min), Shape=%s", dur_s, dur_s / 60, audio_in.shape)

    # NaN/Inf-Guard
    audio_in = np.nan_to_num(audio_in.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio_in = np.clip(audio_in, -1.0, 1.0)

    # ── AurikDenker instantiieren ─────────────────────────────────────────────
    from denker.aurik_denker import get_aurik_denker

    denker = get_aurik_denker()
    logger.info("AurikDenker bereit. Restaurierung startet …")

    # ── Restaurierung durchführen ─────────────────────────────────────────────
    t0 = time.perf_counter()
    ergebnis = denker.denke(
        audio_in,
        sr=TARGET_SR,
        mode=MODE,
        progress_callback=_progress_cb,
        input_path=str(input_path),  # Bug 11: file_ext-Prior für Bayesian-Zeroing
    )
    elapsed = time.perf_counter() - t0
    print()  # Newline nach Progress-Bar

    # ── Ergebnis auswerten ───────────────────────────────────────────────────
    logger.info("═" * 70)
    logger.info("ERGEBNIS — Aurik Restaurierung abgeschlossen")
    logger.info("  Material      : %s", ergebnis.material)
    logger.info("  RT-Faktor     : %.2f×", ergebnis.rt_factor)
    logger.info("  Qualität      : %.3f", ergebnis.quality_estimate)
    logger.info("  Goals passed  : %d/%d", ergebnis.goals_passed, len(ergebnis.musical_goals))
    logger.info("  Phasen        : %d ausgeführt", len(ergebnis.phases_executed))
    logger.info("  Versuche      : %.1f s gesamt", elapsed)

    if ergebnis.musical_goals:
        logger.info("  Musical Goals:")
        for goal, score in sorted(ergebnis.musical_goals.items()):
            status = "✓" if score >= 0.75 else "⚠"
            logger.info("    %s %-30s = %.3f", status, goal, score)

    if ergebnis.warnings:
        logger.info("  Warnungen (%d):", len(ergebnis.warnings))
        for w in ergebnis.warnings[:10]:
            logger.info("    ⚠ %s", w)
        if len(ergebnis.warnings) > 10:
            logger.info("    … und %d weitere", len(ergebnis.warnings) - 10)

    if ergebnis.stage_notes:
        logger.info("  Stufen-Details:")
        for stage, note in ergebnis.stage_notes.items():
            logger.info("    [%s] %s", stage, note)

    # Quality-Gate prüfen
    if ergebnis.quality_estimate < 0.55:
        logger.warning("⚠ quality_estimate=%.3f < 0.55 — Export-Gate NICHT bestanden!", ergebnis.quality_estimate)
        logger.warning(
            "Export wird abgebrochen — Ursache: Qualitäts-Gate verfehlt. Lösung: Pipeline/Gating vor Export korrigieren."
        )
        return 2
    else:
        logger.info("✓ Export-Gate bestanden (quality_estimate=%.3f ≥ 0.55)", ergebnis.quality_estimate)

    # ── Export ───────────────────────────────────────────────────────────────
    stem = input_path.stem
    out_wav = OUTPUT_DIR / f"{stem}_aurik_maximum.wav"
    out_flac = OUTPUT_DIR / f"{stem}_aurik_maximum.flac"

    audio_out = ergebnis.audio
    # NaN/Inf-Guard vor Export
    audio_out = np.nan_to_num(audio_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio_out = np.clip(audio_out, -1.0, 1.0)

    # Exportformat: AudioExporter
    try:
        from backend.core.audio_exporter import AudioExporter

        exp = AudioExporter()
        # WAV 24-bit
        out_wav_final = exp.export(
            audio_out,
            TARGET_SR,
            out_wav,
            bit_depth=24,
            quality="high",
            metadata={
                "title": stem,
                "comment": f"Aurik 9 Maximum Restoration | quality={ergebnis.quality_estimate:.3f}",
                "software": "Aurik 9.10.57",
            },
        )
        logger.info("✓ WAV exportiert: %s", out_wav_final)

        # FLAC 24-bit
        out_flac_final = exp.export(
            audio_out,
            TARGET_SR,
            out_flac,
            bit_depth=24,
            quality="high",
            metadata={
                "title": stem,
                "comment": f"Aurik 9 Maximum Restoration | quality={ergebnis.quality_estimate:.3f}",
                "software": "Aurik 9.10.57",
            },
        )
        logger.info("✓ FLAC exportiert: %s", out_flac_final)

    except Exception as exp_err:
        logger.warning("AudioExporter fehlgeschlagen (%s) — Fallback auf soundfile", exp_err)
        import soundfile as sf

        # (samples, channels) oder (samples,) für soundfile
        sf.write(str(out_wav), audio_out, TARGET_SR, subtype="PCM_24")
        logger.info("✓ WAV exportiert (soundfile-Fallback): %s", out_wav)

    logger.info("═" * 70)
    logger.info("Fertig! Ausgabe liegt in: %s/", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
