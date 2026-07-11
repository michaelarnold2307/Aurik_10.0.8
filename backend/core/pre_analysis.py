"""Pre-analysis module — single authoritative backend entry point for all
pre-restoration analysis tasks.

Replaces the four scattered frontend background threads (_carrier_bg,
_detect_era_genre_bg, _estimate_restorability_bg, _run_defect_scan_bg)
with a single, testable, spec-compliant backend call.

Spec compliance:
  - Analysis modules run at native import SR (no resampling before analysis).
  - Restorability estimator receives the 48 kHz-resampled processing audio
    (its API requires 48 kHz per spec §2.26).
  - DefectScanner.scan() receives native-SR audio with file_ext for
    Bayesian posterior-zeroing (Bug-15 fix).
  - MediumDetector.detect() receives native-SR audio with file_ext.
  - EraClassifier and GermanSchlagerClassifier receive native-SR audio.
  - All four analyses run in parallel (ThreadPoolExecutor max_workers=4).
  - Result is stored in bridge cache so UV3 never re-runs any classifier.

Usage (frontend)::

    from backend.core.pre_analysis import run_pre_analysis, PreAnalysisResult

    result: PreAnalysisResult = run_pre_analysis(
        audio_native=audio_before_resample,
        sr_native=sr_native,
        audio_48k=audio_after_resample,
        file_path="/path/to/song.mp3",
        progress_callback=lambda pct, msg: ...,   # optional
    )
    # Display result.medium, result.era, result.genre, etc. in UI.
    # Pass to UV3 via bridge cache — no kwarg threading required.

Usage (UV3 / CLI — no frontend)::

    result = run_pre_analysis(audio_native=audio, sr_native=sr,
                              audio_48k=audio_48k, file_path=path)
"""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent import futures as _cf
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Observed: DefectScanner needs ~80s for 60s audio (133% overhead on this hardware).
# 150s = 1.87x buffer. concurrent.futures.TimeoutError != builtins.TimeoutError in Python 3.10.
_SUBSTEP_TIMEOUT_S = 150.0

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreAnalysisResult:
    """Aggregated result of all pre-restoration analysis steps.

    All fields use the canonical result types from their respective modules.
    Every field is Optional so callers can handle partial failures gracefully.
    """

    # Carrier / medium chain (forensics.medium_detector)
    medium: object | None = None  # MediumDetectionResult

    # Recording era (backend.core.era_classifier)
    era: object | None = None  # EraResult

    # Genre classification (backend.core.genre_classifier)
    genre: object | None = None  # SchlagerClassificationResult

    # Defect scan (backend.core.defect_scanner)
    defects: object | None = None  # DefectAnalysisResult

    # Restorability estimate (backend.core.restorability_estimator)
    restorability: object | None = None  # RestorabilityResult

    # Metadata
    native_sr: int = 0
    file_path: str = ""
    elapsed_seconds: float = 0.0

    # Per-step error messages (populated on exception, step still gets None above)
    errors: dict[str, str] = field(default_factory=dict)


_run_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_mono_native(audio: np.ndarray) -> np.ndarray:
    """Konvertiert to mono without SR change; clip & nan-guard."""
    if audio.ndim == 2:
        # shape (N, 2) or (2, N)
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T
        mono = audio.mean(axis=1)
    else:
        mono = audio.copy()
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    mono_f32: np.ndarray = np.asarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)
    return mono_f32


def _load_symbol(module_name: str, symbol_name: str) -> object:
    """Lädt optional or heavy symbols lazily without inline import statements."""
    return getattr(import_module(module_name), symbol_name)


def _resample_for_restorability(audio_native: np.ndarray, sr_native: int) -> tuple[np.ndarray, int]:
    """Resample to 48 kHz if not already there (restorability estimator requires 48 kHz)."""
    if sr_native == 48_000:
        return audio_native, sr_native
    try:
        _rp = cast(Callable[..., np.ndarray], _load_symbol("scipy.signal", "resample_poly"))

        gcd = math.gcd(int(sr_native), 48_000)
        audio_48 = _rp(
            audio_native,
            48_000 // gcd,
            int(sr_native) // gcd,
            axis=0 if audio_native.ndim > 1 else -1,
        ).astype(np.float32)
        return audio_48, 48_000
    except Exception as exc:
        logger.warning("pre_analysis: resample for restorability failed (%s) — using native SR", exc)
        return audio_native, sr_native


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pre_analysis(
    audio_native: np.ndarray,
    sr_native: int,
    *,
    audio_48k: np.ndarray | None = None,
    file_path: str = "",
    progress_callback: Callable[[int, str], None] | None = None,
    scan_progress_callback: Callable[[float], None] | None = None,
    store_in_bridge_cache: bool = True,
) -> PreAnalysisResult:
    """Führt aus: all pre-restoration analyses in parallel and return a PreAnalysisResult.

    This is the single authoritative entry point for all pre-restoration analysis.
    It replaces the four scattered frontend background threads.

    Args:
        audio_native:       Audio at native import SR (no resampling applied).
                            Used for medium, era, genre, defect analysis.
        sr_native:          Sample rate of audio_native [Hz].
        audio_48k:          Optional pre-resampled audio at 48 kHz, used for
                            restorability estimation (saves one resample if
                            the caller already has this). If None, computed
                            internally.
        file_path:          Absolute path to source file. Used for file_ext
                            (Bayesian posterior-zeroing) and bridge cache key.
        progress_callback:  Optional (pct: int, msg: str) -> None callback.
                            Reports 0, 25, 50, 75, 100.
        scan_progress_callback:
                    Optional (pct: float) -> None callback forwarded to
                            DefectScanner.scan() for fine-grained progress.
        store_in_bridge_cache:
                            When True, stores each sub-result in the bridge
                            LRU cache so UV3 never re-runs any classifier.

    Returns:
        PreAnalysisResult with all sub-results populated (or None on failure).
    """
    t0 = time.monotonic()

    _cb = progress_callback or (lambda pct, msg: None)
    _cb(0, "Voranalyse gestartet…")

    _cached_parts: dict[str, object | None] = {}
    # Fast-Path: Falls alle Voranalyse-Subresultate bereits im Bridge-Cache liegen,
    # liefern wir deterministisch aus dem Cache statt die Analyzer erneut zu starten.
    if store_in_bridge_cache and file_path:
        _cached_parts = _load_cached_parts(file_path)
        _cached_result = _build_result_from_cached_parts(_cached_parts, sr_native=sr_native, file_path=file_path)
        if _cached_result is not None:
            _cached_result.elapsed_seconds = time.monotonic() - t0
            _cb(100, "Voranalyse aus Cache geladen.")
            logger.info("pre_analysis: cache-hit for %s (%.3fs)", file_path, _cached_result.elapsed_seconds)
            return _cached_result

    file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""

    # Prepare 48 kHz audio for restorability if not supplied
    if audio_48k is None:
        audio_48k, _ = _resample_for_restorability(audio_native, sr_native)

    # Derive material hint from era result later; use "unknown" for initial defect scan
    result = PreAnalysisResult(native_sr=sr_native, file_path=file_path)

    # ------------------------------------------------------------------
    # Step 1 — Medium detection (native SR, with file_ext) — run first
    # so material hint is available for restorability.
    # Steps 2-5 run in parallel after medium finishes.
    # ------------------------------------------------------------------
    _medium_result = _cached_parts.get("medium")
    if _medium_result is not None:
        result.medium = _medium_result
        logger.debug("pre_analysis: medium aus Cache geladen")

    _medium_primary_error: str | None = None
    if _medium_result is None:
        try:
            _get_md = cast(Callable[[], Any], _load_symbol("forensics.medium_detector", "get_medium_detector"))

            _medium_result = _get_md().detect(audio_native, sr_native, file_ext=file_ext)
            result.medium = _medium_result
            _medium_result_any = cast(Any, _medium_result)

            logger.info(
                "pre_analysis: medium=%s conf=%.2f chain=%s",
                _medium_result_any.primary_material,
                _medium_result_any.confidence,
                _medium_result_any.chain_label,
            )
        except Exception as exc:
            _medium_primary_error = str(exc)
            logger.warning("pre_analysis: primary medium detection failed (%s)", exc)

    # Strict detector-only policy:
    # - Primary detector exactly once
    # - No legacy MediumClassifier fallback in production chain detection (§6.7)
    if _medium_result is None:
        if _medium_primary_error is not None:
            result.errors["medium"] = f"primary_failed={_medium_primary_error}; no_legacy_fallback=true"
        else:
            result.errors["medium"] = "medium_detection_failed; no_legacy_fallback=true"
        logger.warning(
            "pre_analysis: medium detection unavailable; continuing without medium result (legacy fallback disabled)"
        )

    _cb(20, "Tonträger erkannt — analysiere Ära, Genre und Defekte…")

    # Material string for downstream modules
    _material_str = "unknown"
    if _medium_result is not None:
        _material_str = str(getattr(_medium_result, "primary_material", None) or "unknown")

    # ------------------------------------------------------------------
    # Steps 2–5 — Era, Genre, DefectScan, Restorability in parallel
    # ------------------------------------------------------------------
    def _run_era() -> object:
        _gec = cast(Callable[[], Any], _load_symbol("backend.core.era_classifier", "get_era_classifier"))

        return _gec().classify(audio_native, sr_native)

    def _run_genre() -> object:
        _ggc = cast(Callable[[], Any], _load_symbol("backend.core.genre_classifier", "get_genre_classifier"))

        return _ggc().classify(audio_native, sr_native)

    def _run_defects() -> object:
        _DS = cast(Callable[..., Any], _load_symbol("backend.core.defect_scanner", "DefectScanner"))

        scanner = _DS(sample_rate=sr_native, material_type=None)
        _kw: dict = {
            "sample_rate": sr_native,
            "file_ext": file_ext,
            "forensic_medium_result": _medium_result,
        }
        # §2.47a: Pass forensically-detected material to scan() so threshold setup
        # uses the MediumDetector result rather than the internal heuristic fallback.
        if _material_str not in ("unknown", ""):
            _kw["material_type"] = _material_str
        if scan_progress_callback is not None:
            _kw["progress_callback"] = scan_progress_callback
        return scanner.scan(audio_native, **_kw)

    def _run_restorability() -> object:
        _er = cast(
            Callable[..., object],
            _load_symbol("backend.core.restorability_estimator", "estimate_restorability"),
        )

        return _er(audio_48k, 48_000, material=_material_str)

    _step_fns: dict[str, Callable[[], object]] = {}
    if _cached_parts.get("era") is not None:
        result.era = _cached_parts["era"]
        logger.debug("pre_analysis: step=era aus Cache geladen")
    else:
        _step_fns["era"] = _run_era

    if _cached_parts.get("genre") is not None:
        result.genre = _cached_parts["genre"]
        logger.debug("pre_analysis: step=genre aus Cache geladen")
    else:
        _step_fns["genre"] = _run_genre

    if _cached_parts.get("defects") is not None:
        result.defects = _cached_parts["defects"]
        logger.debug("pre_analysis: step=defects aus Cache geladen")
    else:
        _step_fns["defects"] = _run_defects

    if _cached_parts.get("restorability") is not None:
        result.restorability = _cached_parts["restorability"]
        logger.debug("pre_analysis: step=restorability aus Cache geladen")
    else:
        _step_fns["restorability"] = _run_restorability

    if _step_fns:
        if len(_step_fns) == 1:
            name, fn = next(iter(_step_fns.items()))
            logger.debug("pre_analysis: single step=%s runs inline (no pool overhead)", name)
            try:
                setattr(result, name, fn())
                logger.debug("pre_analysis: step=%s done", name)
            except Exception as exc:
                result.errors[name] = str(exc)
                logger.warning("pre_analysis: step=%s failed (%s)", name, exc)
        else:
            _worker_count = max(1, min(4, len(_step_fns)))
            logger.debug(
                "pre_analysis: parallel steps=%d cached=%d workers=%d",
                len(_step_fns),
                4 - len(_step_fns),
                _worker_count,
            )
            _pool = _cf.ThreadPoolExecutor(max_workers=_worker_count)
            _had_substep_timeout = False
            try:
                _fut = {name: _pool.submit(fn) for name, fn in _step_fns.items()}

                for name, fut in _fut.items():
                    try:
                        sub = fut.result(timeout=_SUBSTEP_TIMEOUT_S)
                        setattr(result, name, sub)
                        logger.debug("pre_analysis: step=%s done", name)
                    except (TimeoutError, _cf.TimeoutError):  # Python 3.10: cf.TimeoutError != builtins.TimeoutError
                        _had_substep_timeout = True
                        result.errors[name] = f"timeout_after={_SUBSTEP_TIMEOUT_S:.1f}s"
                        fut.cancel()
                        logger.warning(
                            "pre_analysis: step=%s timed out after %.1fs; degrading without this sub-result",
                            name,
                            _SUBSTEP_TIMEOUT_S,
                        )
                    except Exception as exc:
                        result.errors[name] = str(exc)
                        logger.warning("pre_analysis: step=%s failed (%s)", name, exc)
            finally:
                # Die Futures wurden oben bereits konsumiert; hier wollen wir die Worker
                # deterministisch beenden statt sie bis zum Interpreter-Exit offen zu lassen.
                _pool.shutdown(wait=not _had_substep_timeout, cancel_futures=True)
    else:
        logger.debug("pre_analysis: steps 2-5 vollständig aus Cache geladen")

    _cb(90, "Analyse abgeschlossen — Ergebnisse werden gespeichert…")

    # ── §Original-Medium-Chain-Injection ────────────────────────────────
    # Wenn der MediumDetector nur digitale Stufen erkannt hat (z.B. ['mp3_low']),
    # die EraClassifier aber ein analoges Original-Medium für die Ära ermittelt
    # hat (z.B. decade=1980 → 'tape'), wird die Transferkette um den
    # ursprünglichen analogen Tonträger erweitert.
    # Beispiel: ['mp3_low'] → ['tape', 'mp3_low'], is_multi=True
    if result.medium is not None and result.era is not None:
        try:
            _chain = list(getattr(result.medium, "transfer_chain", []) or [])
            _era_material = str(getattr(result.era, "material_prior", "") or "")
            _analog_materials = {"shellac", "wax_cylinder", "vinyl", "tape", "reel_tape", "cassette", "lacquer_disc", "wire_recording"}
            _has_analog = any(m in _analog_materials for m in _chain)

            if not _has_analog and _era_material in _analog_materials and _chain:
                _chain.insert(0, _era_material)
                _md_result = result.medium
                _md_result.transfer_chain = _chain
                _md_result.is_multi_generation = len(_chain) > 1
                _md_result.primary_material = _era_material
                if hasattr(_md_result, "evidence"):
                    _md_result.evidence.append(
                        f"Original-Medium-Inference: EraClassifier → {_era_material} "
                        f"(ursprünglicher Tonträger der Veröffentlichung)"
                    )
                logger.info(
                    "pre_analysis: Original-Medium-Chain-Injection: %s → %s (era=%s)",
                    " → ".join(_chain[1:]) if len(_chain) > 1 else _chain[0],
                    " → ".join(_chain),
                    _era_material,
                )
        except Exception as _inj_exc:
            logger.debug("Original-Medium-Chain-Injection skipped: %s", _inj_exc)

    # ------------------------------------------------------------------
    # Store in bridge cache so UV3 never re-runs classifiers
    # ------------------------------------------------------------------
    if store_in_bridge_cache and file_path:
        _store_in_cache(file_path, result)

    result.elapsed_seconds = time.monotonic() - t0
    logger.info("pre_analysis: complete in %.1fs (errors=%s)", result.elapsed_seconds, list(result.errors))

    # Free DefectScanner STFT/spectral intermediate arrays (30 defect types × full audio).
    # Vollstaendiges GC ist hier sicher; malloc_trim(0) bleibt bewusst deaktiviert,
    # weil der Aufruf im Projekt bereits mehrfach als SIGABRT-Risiko unter
    # konkurrierenden Audio-/NumPy-Threads aufgefallen ist.
    gc.collect()

    _cb(100, "Voranalyse fertig.")
    return result


def _store_in_cache(file_path: str, result: PreAnalysisResult) -> None:
    """Store all sub-results in bridge LRU caches."""
    try:
        cache_defect_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_defect_result"),
        )
        cache_era_genre_result = cast(
            Callable[..., None],
            _load_symbol("backend.api.bridge", "cache_era_genre_result"),
        )
        cache_medium_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_medium_result"),
        )
        cache_restorability_result = cast(
            Callable[[str, object], None],
            _load_symbol("backend.api.bridge", "cache_restorability_result"),
        )

        if result.medium is not None:
            cache_medium_result(file_path, result.medium)

        if result.era is not None or result.genre is not None:
            cache_era_genre_result(
                file_path,
                era_result=result.era,
                genre_result=result.genre,
            )

        if result.defects is not None:
            cache_defect_result(file_path, result.defects)

        if result.restorability is not None:
            cache_restorability_result(file_path, result.restorability)

        logger.debug("pre_analysis: bridge cache updated for %s", file_path)
    except Exception as exc:
        logger.warning("pre_analysis: bridge cache store failed (%s)", exc)


def _load_cached_parts(file_path: str) -> dict[str, object | None]:
    """Lädt verfügbare Bridge-Cache-Teilergebnisse best-effort."""
    try:
        get_cached_defect_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_defect_result"),
        )
        get_cached_era_genre_result = cast(
            Callable[[str], dict[str, object] | None],
            _load_symbol("backend.api.bridge", "get_cached_era_genre_result"),
        )
        get_cached_medium_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_medium_result"),
        )
        get_cached_restorability_result = cast(
            Callable[[str], object | None],
            _load_symbol("backend.api.bridge", "get_cached_restorability_result"),
        )

        era_genre = get_cached_era_genre_result(file_path)
        era_result = era_genre.get("era_result") if isinstance(era_genre, dict) else None
        genre_result = era_genre.get("genre_result") if isinstance(era_genre, dict) else None
        return {
            "medium": get_cached_medium_result(file_path),
            "era": era_result,
            "genre": genre_result,
            "defects": get_cached_defect_result(file_path),
            "restorability": get_cached_restorability_result(file_path),
        }
    except Exception as exc:
        logger.debug("pre_analysis: cache part load non-blocking (%s)", exc)
        return {}


def _load_from_cache(file_path: str, sr_native: int) -> PreAnalysisResult | None:
    """Lädt ein vollständiges PreAnalysisResult aus Bridge-Caches, falls vorhanden.

    Returns ``None``, wenn mindestens ein Pflicht-Subresultat fehlt.
    """
    try:
        _parts = _load_cached_parts(file_path)
        return _build_result_from_cached_parts(_parts, sr_native=sr_native, file_path=file_path)
    except Exception as exc:
        logger.debug("pre_analysis: cache load non-blocking (%s)", exc)
        return None


def _build_result_from_cached_parts(
    parts: dict[str, object | None],
    *,
    sr_native: int,
    file_path: str,
) -> PreAnalysisResult | None:
    """Erzeugt ein vollständiges PreAnalysisResult nur bei vollständig belegten Cache-Parts."""
    medium = parts.get("medium")
    era = parts.get("era")
    genre = parts.get("genre")
    defects = parts.get("defects")
    restorability = parts.get("restorability")
    if medium is None or era is None or genre is None or defects is None or restorability is None:
        return None

    return PreAnalysisResult(
        medium=medium,
        era=era,
        genre=genre,
        defects=defects,
        restorability=restorability,
        native_sr=sr_native,
        file_path=file_path,
    )
