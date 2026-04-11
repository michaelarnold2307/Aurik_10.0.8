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

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _auto_detect_budget() -> float:
    """Derive ML budget from system RAM: total_ram / 3, capped at [4, 12] GB.

    Ensures ~2/3 of RAM stays free for OS, Qt GUI, audio buffers, and numpy
    intermediate arrays.  On 32 GB → 10 GB; on 16 GB → 5 GB; on 64 GB → 12 GB.
    """
    if _psutil is not None:
        total_gb = _psutil.virtual_memory().total / (1024**3)
        budget = max(4.0, min(12.0, total_gb / 3.0))
        return round(budget, 1)
    return 10.0  # conservative default without psutil


# Maximum total RAM allowed for ALL ML models combined.
# Auto-detected from system RAM; override via set_budget() if needed.
ML_MAX_GB: float = _auto_detect_budget()
_SYSTEM_MEMORY_MARGIN_BASE: float = 1.35  # Basis-Margin für kleine Modelle (< 1 GB)
_SYSTEM_MEMORY_MARGIN_MIN: float = 1.10  # Minimale Margin für sehr große Modelle (>= 5 GB)
_MIN_FREE_MB_HARD: float = 3072.0  # 3 GB — angehoben von 1.5 GB (systemd-oomd-Schutz)


def _scaled_margin(size_gb: float) -> float:
    """Skalierte RAM-Margin: Große Modelle brauchen prozentual weniger Reserve.

    Kleine Modelle (<1 GB): 1.35× (35% Reserve für Overhead, Fragmentierung)
    Große Modelle (>=5 GB): 1.10× (10% Reserve — Modellgewichte sind kompakt, wenig Overhead)
    Dazwischen: linear interpoliert.

    Begründung: AudioSR (7 GB) × 1.35 = 9.45 GB — blockiert auf 16-GB-Systemen
    obwohl 8.8 GB frei sind. Mit skalierter Margin: 7 × 1.10 = 7.7 GB — passt.
    """
    if size_gb <= 1.0:
        return _SYSTEM_MEMORY_MARGIN_BASE
    if size_gb >= 5.0:
        return _SYSTEM_MEMORY_MARGIN_MIN
    # Linear interpolation between 1.0 GB and 5.0 GB
    t = (size_gb - 1.0) / 4.0
    return _SYSTEM_MEMORY_MARGIN_BASE + t * (_SYSTEM_MEMORY_MARGIN_MIN - _SYSTEM_MEMORY_MARGIN_BASE)


# ---------------------------------------------------------------------------
# Internal state (thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_allocated: dict[str, float] = {}  # model_name → GB currently allocated
_total_gb: float = 0.0  # sum of _allocated.values()


def _available_memory_mb() -> float:
    """Return available system memory in MB, or inf if psutil is unavailable."""
    if _psutil is None:
        return float("inf")
    return float(_psutil.virtual_memory().available / (1024 * 1024))


def is_system_thrashing() -> bool:
    """Detect swap-thrashing: high swap usage combined with low free RAM.

    Heuristic: swap > 30 % used AND available RAM < 15 % of total.
    On Linux, systemd-oomd kills at ~50 % memory-pressure for > 20 s.
    We detect BEFORE that point to allow graceful degradation.
    """
    if _psutil is None:
        return False
    try:
        swap = _psutil.swap_memory()
        vm = _psutil.virtual_memory()
        swap_used_pct = swap.percent  # 0–100
        avail_ratio = vm.available / max(vm.total, 1)
        thrashing = swap_used_pct > 30.0 and avail_ratio < 0.15
        if thrashing:
            logger.warning(
                "ml_memory_budget: swap thrashing detected — swap %.0f %% used (%.1f GB), "
                "RAM available %.1f %% (%.1f GB) — ML loads will be blocked",
                swap_used_pct,
                swap.used / (1024**3),
                avail_ratio * 100,
                vm.available / (1024**3),
            )
        return thrashing
    except Exception:
        return False


def _preflight_system_memory(required_mb: float) -> bool:
    """Best-effort system RAM preflight before allocating ML budget.

    This complements the logical ML budget with a physical RAM check.
    On pressure, it asks PluginLifecycleManager to evict stale models first.

    For models >= 1 GB we apply a load-peak-aware check:
      PyTorch's torch.load() temporarily uses ~1.6× the model weight size while
      deserializing tensors (the original bytes and the parsed tensors coexist
      briefly in memory).  We must also keep at least 12 % of total system RAM
      free after the peak to stay below the systemd-oomd kill threshold (90 %).
      Without this check a 2 GB model on a system with 5.9 GB free PASSES the
      steady-state margin (1.29 GB needed) but CRASHES during load:
        peak usage ≈ 3.2 GB → RAM reaches 92 % → systemd-oomd fires SIGKILL.
    """
    if _psutil is None:
        return True

    _size_gb = required_mb / 1024.0
    available_mb = _available_memory_mb()

    if _size_gb >= 1.0:
        # Load-peak formula: require enough free RAM to survive the peak
        # (1.6× model size) PLUS a systemd-oomd safety margin (12 % of total).
        _total_ram_mb = float(_psutil.virtual_memory().total) / (1024.0 * 1024.0)
        _oomd_safe_mb = max(2048.0, _total_ram_mb * 0.12)
        _peak_required_mb = required_mb * 1.6 + _oomd_safe_mb
        required_with_margin = max(_peak_required_mb, required_mb * _scaled_margin(_size_gb), _MIN_FREE_MB_HARD)
    else:
        _margin = _scaled_margin(_size_gb)
        required_with_margin = max(required_mb * _margin, _MIN_FREE_MB_HARD)

    if available_mb >= required_with_margin:
        return True

    try:
        from backend.core.plugin_lifecycle_manager import evict_stale_plugins

        evict_stale_plugins(required_mb=required_with_margin)
    except Exception as _exc:
        logger.debug("Operation failed (non-critical): %s", _exc)

    available_after_evict_mb = _available_memory_mb()
    if available_after_evict_mb >= required_with_margin:
        return True

    logger.warning(
        "ml_memory_budget: physical RAM too low — required %.0f MB (incl. safety margin), "
        "available %.0f MB. ML load blocked, DSP fallback active.",
        required_with_margin,
        available_after_evict_mb,
    )
    # §DEBUG: Stack-Trace für suspekt große Requests (> 10000 MB) automatisch loggen
    if required_with_margin > 10_000:
        import traceback as _tb

        logger.warning(
            "ml_memory_budget §DEBUG: suspekt große Anfrage (%.0f MB) — Stack-Trace:\n%s",
            required_with_margin,
            "".join(_tb.format_stack()),
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

    # Swap-Thrashing-Guard: Wenn system bereits thrashing, alle neuen
    # ML-Loads blockieren — DSP-Fallback statt Freeze/OOM.
    if is_system_thrashing():
        logger.warning(
            "ML-Budget: '%s' (%.1f GB) blockiert — System-Thrashing erkannt, DSP-Fallback aktiv.",
            model_name,
            size_gb,
        )
        return False

    if not _preflight_system_memory(required_mb=max(size_gb, 0.0) * 1024.0):
        return False

    with _lock:
        if model_name in _allocated:
            return True
        remaining = ML_MAX_GB - _total_gb
        if size_gb > remaining:
            logger.warning(
                "ml_memory_budget: '%s' needs %.1f GB, only %.1f GB of %.1f GB free — DSP fallback active.",
                model_name,
                size_gb,
                remaining,
                ML_MAX_GB,
            )
            return False
        _allocated[model_name] = size_gb
        _total_gb += size_gb
        logger.info(
            "ml_memory_budget: '%s' allocated %.1f GB  →  total %.1f / %.1f GB used.",
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
                "ml_memory_budget: '%s' released (%.1f GB)  →  total %.1f / %.1f GB used.",
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
        logger.info("ml_memory_budget: max budget set to %.1f GB.", ML_MAX_GB)


# ---------------------------------------------------------------------------
# §3.9.5  Startup reconciliation
# ---------------------------------------------------------------------------


def _reconcile_on_startup() -> None:
    """Reset allocated budget to 0 on fresh process start (§3.9.5).

    Rationale: All allocations from a previous process are gone after OS
    cleanup (SIGKILL / crash). Each module re-registers via try_allocate()
    when it actually loads its model. No stale allocation persists across
    process boundaries.  Called once at module import time.
    """
    global _total_gb
    with _lock:
        _allocated.clear()
        _total_gb = 0.0
    logger.info("ml_memory_budget: startup reconciliation — budget reset to 0.0 GB")


# Run reconciliation exactly once at module import (= new process start).
_reconcile_on_startup()


# ---------------------------------------------------------------------------
# §3.9.5  Thin OO wrapper — returned by get_ml_memory_budget()
# ---------------------------------------------------------------------------


class _MLMemoryBudgetProxy:
    """Thin proxy so callers can use get_ml_memory_budget().try_allocate().

    All work is delegated to the module-level functions; no additional state.
    """

    # Lock-order: Priority 1 (MLMemoryBudget) — see §3.9.8
    def try_allocate(self, model_name: str, size_gb: float) -> bool:
        return try_allocate(model_name, size_gb)

    def release(self, model_name: str) -> None:
        release(model_name)

    def get_status(self) -> dict:
        return get_status()


_proxy_instance = _MLMemoryBudgetProxy()


def get_ml_memory_budget() -> _MLMemoryBudgetProxy:
    """Return the global ML-memory-budget proxy (singleton-safe).

    Provides an OO API (.try_allocate / .release) in addition to the
    module-level functions, so both usage styles work.
    """
    return _proxy_instance
