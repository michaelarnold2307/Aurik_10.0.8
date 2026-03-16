"""Global ML Memory Budget — prevents OOM-kills by capping total ML model RAM.

Problem: 35+ GB model files on a 32 GB system. Without a global cap, individual
plugin loaders each check free RAM independently and all see "enough space", then
load their models sequentially until the kernel OOM-killer fires.

Solution: Centralized Thread-safe budget singleton. Every heavy ML model loader
calls ``try_allocate()`` before loading. Once the budget is exhausted, remaining
models fall back to DSP. Budget is set to ``ML_MAX_GB`` (default 16 GB), leaving
~16 GB free for OS + app + audio processing buffers on a 32 GB machine.

Usage in a plugin loader::

    from backend.core.ml_memory_budget import try_allocate, release

    if not try_allocate("AudioSR", size_gb=7.0):
        return None           # → DSP fallback
    try:
        model = load_heavy_model(...)
    except Exception:
        release("AudioSR", size_gb=7.0)   # refund on failure
        raise
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum total RAM allowed for ALL ML models combined.
# On a 32 GB machine with only 2 GB swap: leave ≥18 GB free for OS + Qt + buffers.
# 10 GB cap → AudioSR(7) + UTMOSv2(0.8) = 7.8 GB; MERT/CLAP fall back to DSP.
# Adjust via set_budget() if needed.
ML_MAX_GB: float = 10.0
_SYSTEM_MEMORY_MARGIN: float = 1.35
_MIN_FREE_MB_HARD: float = 1536.0

# ---------------------------------------------------------------------------
# Internal state (thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_allocated: Dict[str, float] = {}   # model_name → GB currently allocated
_total_gb: float = 0.0              # sum of _allocated.values()


def _available_memory_mb() -> float:
    """Return available system memory in MB, or inf if psutil is unavailable."""
    if _psutil is None:
        return float("inf")
    return float(_psutil.virtual_memory().available / (1024 * 1024))


def _preflight_system_memory(required_mb: float) -> bool:
    """Best-effort system RAM preflight before allocating ML budget.

    This complements the logical ML budget with a physical RAM check.
    On pressure, it asks PluginLifecycleManager to evict stale models first.
    """
    if _psutil is None:
        return True

    required_with_margin = max(required_mb * _SYSTEM_MEMORY_MARGIN, _MIN_FREE_MB_HARD)
    available_mb = _available_memory_mb()
    if available_mb >= required_with_margin:
        return True

    try:
        from backend.core.plugin_lifecycle_manager import evict_stale_plugins  # noqa: PLC0415

        evict_stale_plugins(required_mb=required_with_margin)
    except Exception:
        pass

    available_after_evict_mb = _available_memory_mb()
    if available_after_evict_mb >= required_with_margin:
        return True

    logger.warning(
        "Physischer RAM zu knapp: benötigt %.0f MB (inkl. Reserve), verfügbar %.0f MB. "
        "ML-Load wird blockiert, DSP-Fallback aktiv.",
        required_with_margin,
        available_after_evict_mb,
    )
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def try_allocate(model_name: str, size_gb: float) -> bool:
    """Reserve ``size_gb`` GB of ML budget for ``model_name``.

    Returns True if granted (proceed to load the model).
    Returns False if the budget would be exceeded (use DSP fallback instead).
    Idempotent: if ``model_name`` is already allocated, returns True immediately.
    """
    global _total_gb
    with _lock:
        if model_name in _allocated:
            # Already loaded — no additional budget consumed.
            return True

    if not _preflight_system_memory(required_mb=max(size_gb, 0.0) * 1024.0):
        return False

    with _lock:
        if model_name in _allocated:
            return True
        remaining = ML_MAX_GB - _total_gb
        if size_gb > remaining:
            logger.warning(
                "ML-Budget erschöpft: '%s' benötigt %.1f GB, "
                "aber nur %.1f GB von %.1f GB frei — DSP-Fallback aktiv.",
                model_name,
                size_gb,
                remaining,
                ML_MAX_GB,
            )
            return False
        _allocated[model_name] = size_gb
        _total_gb += size_gb
        logger.info(
            "ML-Budget: '%s' reserviert %.1f GB  →  Gesamt %.1f / %.1f GB belegt.",
            model_name,
            size_gb,
            _total_gb,
            ML_MAX_GB,
        )
        return True


def release(model_name: str) -> None:
    """Release the budget slot for ``model_name`` (call when model is unloaded).

    Safe to call even if the model was never allocated.
    """
    global _total_gb
    with _lock:
        freed = _allocated.pop(model_name, 0.0)
        _total_gb = max(0.0, _total_gb - freed)
        if freed:
            logger.info(
                "ML-Budget: '%s' freigegeben (%.1f GB)  →  Gesamt %.1f / %.1f GB belegt.",
                model_name,
                freed,
                _total_gb,
                ML_MAX_GB,
            )


def get_status() -> dict:
    """Return current budget status (for logging/debug)."""
    with _lock:
        return {
            "max_gb": ML_MAX_GB,
            "allocated_gb": round(_total_gb, 2),
            "free_gb": round(ML_MAX_GB - _total_gb, 2),
            "models": dict(_allocated),
        }


def set_budget(max_gb: float) -> None:
    """Override the default 16 GB budget (e.g. on systems with less RAM)."""
    global ML_MAX_GB
    with _lock:
        ML_MAX_GB = float(max_gb)
        logger.info("ML-Budget neu gesetzt: %.1f GB Maximum.", ML_MAX_GB)
