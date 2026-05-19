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
import time

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
        return round(budget, 1)  # type: ignore[no-any-return]
    return 10.0  # conservative default without psutil


# Maximum total RAM allowed for ALL ML models combined.
# Auto-detected from system RAM; override via set_budget() if needed.
ML_MAX_GB: float = _auto_detect_budget()
_SYSTEM_MEMORY_MARGIN_BASE: float = 1.35  # Basis-Margin für kleine Modelle (< 1 GB)
_SYSTEM_MEMORY_MARGIN_MIN: float = 1.10  # Minimale Margin für sehr große Modelle (>= 5 GB)
_MIN_FREE_MB_HARD: float = 3072.0  # 3 GB — angehoben von 1.5 GB (systemd-oomd-Schutz)
_PRESSURE_LIGHT_MODEL_MAX_GB: float = 0.12
_PRESSURE_LIGHT_MODEL_MIN_AVAIL_RATIO: float = 0.35
_PRESSURE_LIGHT_MODEL_MAX_SWAP_PCT: float = 95.0
_PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP: float = 0.35
_PRESSURE_LIGHT_MODEL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "SileroVAD",
        "SileroVAD_phase18",
        "FCPE",
        "BasicPitch",
    }
)
_HEAVY_MODEL_PREEMPTIVE_MIN_GB: float = 1.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_PCT: float = 70.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT: float = 45.0
_HEAVY_MODEL_PREEMPTIVE_SWAP_IO_MB_S: float = 2.0
_HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX: float = 0.30
# 6 GB free = 2.7× buffer for a 2.2 GB model; 16.0 was triggering on 32 GB systems
# at 15.7 GB free (50% RAM) because 15.7 < 16.0 — far too aggressive.
_HEAVY_MODEL_PREEMPTIVE_AVAIL_GB_MAX: float = 6.0
_PRESSURE_RECOVERY_ATTEMPTS: int = 2
_PRESSURE_RECOVERY_SLEEP_S: float = 0.35

# Cooldown for is_system_thrashing() log-spam guard (BUG G).
# Log WARNING at most once per 60 s; always return the correct bool.
_THRASH_WARN_COOLDOWN_S: float = 60.0
_last_thrash_warn_time: float = 0.0
_last_thrash_warn_lock = threading.Lock()


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
_last_swap_poll_ts: float = 0.0
_last_swap_sin: int = 0
_last_swap_sout: int = 0


def _swap_io_rate_mb_per_s(swap_obj: object) -> float:
    """Schätzt recent swap I/O activity in MB/s.

    High swap usage alone does not always mean active thrashing. This helper uses
    swap sin/sout counters to distinguish stale high swap occupancy from ongoing
    paging pressure that can trigger freezes and OOM-kills.
    """
    global _last_swap_poll_ts, _last_swap_sin, _last_swap_sout  # pylint: disable=global-statement
    if _psutil is None:
        return 0.0
    try:
        now = time.monotonic()
        sin = int(getattr(swap_obj, "sin", 0) or 0)
        sout = int(getattr(swap_obj, "sout", 0) or 0)

        if _last_swap_poll_ts <= 0.0:
            _last_swap_poll_ts = float(now)
            _last_swap_sin = sin
            _last_swap_sout = sout
            return 0.0

        dt = max(1e-3, float(now - _last_swap_poll_ts))
        delta_bytes = max(0, (sin - _last_swap_sin)) + max(0, (sout - _last_swap_sout))
        rate_mb_s = (delta_bytes / (1024.0 * 1024.0)) / dt

        _last_swap_poll_ts = float(now)
        _last_swap_sin = sin
        _last_swap_sout = sout
        return float(rate_mb_s)
    except Exception:
        return 0.0


def _available_memory_mb() -> float:
    """Gibt available system memory in MB, or inf if psutil is unavailable zurück."""
    if _psutil is None:
        return float("inf")
    return float(_psutil.virtual_memory().available / (1024 * 1024))


def is_system_thrashing() -> bool:
    """Erkennt swap-thrashing: high swap usage or combined swap+RAM pressure.

        Conditions (any triggers):
            1. swap > 80 % AND active swap I/O > 8 MB/s (real thrashing)
            2. swap > 95 % AND available RAM < 25 % (critical emergency)
            3. swap > 30 % AND available RAM < 15 % (systemd-oomd warning zone)
            4. available RAM < 8 % (hard emergency regardless of swap)

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
        swap_io_rate_mb_s = _swap_io_rate_mb_per_s(swap)
        # Condition 1: critically high swap plus ongoing swap I/O.
        swap_critical_active = swap_used_pct > 80.0 and swap_io_rate_mb_s > 8.0
        # Condition 2: emergency when swap is nearly full and RAM headroom is shrinking.
        swap_critical_emergency = swap_used_pct > 95.0 and avail_ratio < 0.25
        # Condition 3: combined pressure (legacy heuristic, retained).
        combined_pressure = swap_used_pct > 30.0 and avail_ratio < 0.15
        # Condition 4: hard emergency regardless of swap stats.
        ram_emergency = avail_ratio < 0.08

        thrashing = swap_critical_active or swap_critical_emergency or combined_pressure or ram_emergency
        if thrashing:
            global _last_thrash_warn_time  # pylint: disable=global-statement
            _now = time.monotonic()
            with _last_thrash_warn_lock:
                _emit = (_now - _last_thrash_warn_time) >= _THRASH_WARN_COOLDOWN_S
                if _emit:
                    _last_thrash_warn_time = _now
            if _emit:
                logger.warning(
                    "ml_memory_budget: swap thrashing detected — swap %.0f %% used (%.1f GB), "
                    "swap I/O %.1f MB/s, RAM available %.1f %% (%.1f GB) — ML loads will be blocked",
                    swap_used_pct,
                    swap.used / (1024**3),
                    swap_io_rate_mb_s,
                    avail_ratio * 100,
                    vm.available / (1024**3),
                )
        elif swap_used_pct > 80.0 and swap_io_rate_mb_s <= 8.0:
            logger.debug(
                "ml_memory_budget: high swap occupancy without active paging (swap %.0f %%, I/O %.1f MB/s) "
                "— no thrashing block",
                swap_used_pct,
                swap_io_rate_mb_s,
            )
        return thrashing
    except Exception:
        return False


def _allow_lightweight_under_pressure(model_name: str, size_gb: float) -> bool:
    """Allow tiny models during pressure when memory headroom is still healthy.

    This prevents broad cascade fallbacks (e.g. VAD/pitch helper models) while
    keeping large model loads blocked under thrashing.
    """
    if _psutil is None:
        return False
    if size_gb > _PRESSURE_LIGHT_MODEL_MAX_GB:
        return False
    if model_name not in _PRESSURE_LIGHT_MODEL_ALLOWLIST:
        return False
    try:
        vm = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        avail_ratio = vm.available / max(vm.total, 1)
        swap_pct = float(getattr(swap, "percent", 100.0))
        with _lock:
            _tiny_allocated = float(
                sum(
                    float(_sz)
                    for _name, _sz in _allocated.items()
                    if _name in _PRESSURE_LIGHT_MODEL_ALLOWLIST and float(_sz) <= _PRESSURE_LIGHT_MODEL_MAX_GB
                )
            )
        if (_tiny_allocated + float(size_gb)) > _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP:
            logger.warning(
                "ML-Budget: '%s' (%.2f GB) pressure soft-allow abgelehnt — tiny-budget cap erreicht (%.2f/%.2f GB)",
                model_name,
                size_gb,
                _tiny_allocated,
                _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP,
            )
            return False
        if avail_ratio >= _PRESSURE_LIGHT_MODEL_MIN_AVAIL_RATIO and swap_pct < _PRESSURE_LIGHT_MODEL_MAX_SWAP_PCT:
            logger.warning(
                "ML-Budget: '%s' (%.2f GB) soft-allowed under pressure window "
                "(RAM %.1f %%, swap %.1f %%, tiny-cap %.2f/%.2f GB)",
                model_name,
                size_gb,
                avail_ratio * 100.0,
                swap_pct,
                _tiny_allocated + float(size_gb),
                _PRESSURE_LIGHT_MODEL_TOTAL_GB_CAP,
            )
            return True
    except Exception:
        return False
    return False


def _should_block_heavy_ml_load(size_gb: float) -> bool:
    """Gibt True when heavy model loads should be blocked preemptively zurück.

    Rationale:
    - Crash pattern: swap climbs from ~70 % to >85 % during a single heavy load wave
      even while RAM still looks healthy in absolute GB.
    - Thrashing detection alone can be too late if swap occupancy is already high and
      paging starts shortly after model deserialization begins.

    Policy:
    - Applies only to heavy models (>= 1.0 GB).
    - Blocks when swap is already elevated AND either active paging is visible
      or available RAM ratio is no longer high.
    """
    if _psutil is None:
        return False
    if size_gb < _HEAVY_MODEL_PREEMPTIVE_MIN_GB:
        return False
    try:
        vm = _psutil.virtual_memory()
        swap = _psutil.swap_memory()
        swap_pct = float(getattr(swap, "percent", 0.0) or 0.0)
        avail_bytes = float(vm.available)
        avail_ratio = avail_bytes / max(float(vm.total), 1.0)
        avail_gb = avail_bytes / float(1024**3)
        swap_io_rate_mb_s = _swap_io_rate_mb_per_s(swap)
        elevated_swap = swap_pct >= _HEAVY_MODEL_PREEMPTIVE_SWAP_PCT
        early_swap = swap_pct >= _HEAVY_MODEL_PREEMPTIVE_SWAP_EARLY_PCT
        active_paging = swap_io_rate_mb_s >= _HEAVY_MODEL_PREEMPTIVE_SWAP_IO_MB_S
        low_headroom = (
            avail_ratio <= _HEAVY_MODEL_PREEMPTIVE_AVAIL_RATIO_MAX or avail_gb <= _HEAVY_MODEL_PREEMPTIVE_AVAIL_GB_MAX
        )
        should_block = (elevated_swap and (active_paging or low_headroom)) or (early_swap and low_headroom)
        if should_block:
            logger.warning(
                "ML-Budget: preemptive heavy-load block (%.1f GB) — "
                "swap %.0f %%, swap-I/O %.1f MB/s, RAM available %.1f %% (%.1f GB) "
                "→ DSP fallback before thrashing escalation",
                size_gb,
                swap_pct,
                swap_io_rate_mb_s,
                avail_ratio * 100.0,
                avail_gb,
            )
        return should_block
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
        from backend.core.plugin_lifecycle_manager import evict_stale_plugins  # pylint: disable=import-outside-toplevel

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
        import traceback as _tb  # pylint: disable=import-outside-toplevel

        logger.warning(
            "ml_memory_budget §DEBUG: suspekt große Anfrage (%.0f MB) — Stack-Trace:\n%s",
            required_with_margin,
            "".join(_tb.format_stack()),
        )
    return False


def _attempt_quality_preserving_pressure_recovery(model_name: str, size_gb: float) -> bool:
    """Try short pressure recovery before forcing DSP fallback.

    Strategy: evict stale plugins and wait briefly so the allocator can reclaim
    memory pages. This keeps ML-first behavior under transient pressure and only
    blocks when pressure remains critical after retries.
    """
    if _psutil is None:
        return False

    required_mb = max(float(size_gb), 0.0) * 1024.0
    attempts = max(0, int(_PRESSURE_RECOVERY_ATTEMPTS))
    if attempts <= 0:
        return False

    for attempt in range(1, attempts + 1):
        try:
            from backend.core.plugin_lifecycle_manager import (
                evict_stale_plugins,  # pylint: disable=import-outside-toplevel
            )

            evicted = int(evict_stale_plugins(required_mb=required_mb))
        except Exception as _exc:
            logger.debug("ML-Budget: pressure recovery eviction failed (non-fatal): %s", _exc)
            evicted = 0

        if _PRESSURE_RECOVERY_SLEEP_S > 0.0:
            time.sleep(_PRESSURE_RECOVERY_SLEEP_S)

        still_thrashing = is_system_thrashing()
        still_heavy_block = _should_block_heavy_ml_load(float(max(size_gb, 0.0)))
        if not still_thrashing and not still_heavy_block:
            logger.warning(
                "ML-Budget: pressure recovery succeeded for '%s' after %d/%d attempt(s) "
                "(evicted=%d) — ML load retry statt DSP-fallback.",
                model_name,
                attempt,
                attempts,
                evicted,
            )
            return True

        logger.warning(
            "ML-Budget: pressure recovery %d/%d for '%s' insufficient (evicted=%d, thrashing=%s, heavy_block=%s)",
            attempt,
            attempts,
            model_name,
            evicted,
            still_thrashing,
            still_heavy_block,
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
    global _total_gb  # pylint: disable=global-statement
    with _lock:
        if model_name in _allocated:
            # Already loaded — no additional budget consumed.
            return True

    # Swap-Thrashing-Guard: Wenn system bereits thrashing, alle neuen
    # ML-Loads blockieren — DSP-Fallback statt Freeze/OOM.
    if is_system_thrashing():
        if _allow_lightweight_under_pressure(
            model_name, float(max(size_gb, 0.0))
        ) or _attempt_quality_preserving_pressure_recovery(model_name, size_gb):
            pass
        else:
            logger.warning(
                "ML-Budget: '%s' (%.1f GB) blockiert — System-Thrashing erkannt, DSP-Fallback aktiv.",
                model_name,
                size_gb,
            )
            return False

    # Präventiver Heavy-Load-Guard: große Modell-Ladungen schon vor dem
    # harten Thrashing-Zustand abbrechen, wenn Swap-Druck bereits ansteigt.
    if _should_block_heavy_ml_load(float(max(size_gb, 0.0))):
        if not _attempt_quality_preserving_pressure_recovery(model_name, size_gb):
            logger.warning(
                "ML-Budget: '%s' (%.1f GB) präventiv blockiert — erhöhter Swap-Druck, DSP-Fallback aktiv.",
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
    global _total_gb  # pylint: disable=global-statement
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
    """Gibt current budget status (for logging/debug) zurück."""
    with _lock:
        return {
            "max_gb": ML_MAX_GB,
            "allocated_gb": round(_total_gb, 2),
            "free_gb": round(ML_MAX_GB - _total_gb, 2),
            "models": dict(_allocated),
        }


def set_budget(max_gb: float) -> None:
    """Override the default 16 GB budget (e.g. on systems with less RAM)."""
    global ML_MAX_GB  # pylint: disable=global-statement
    with _lock:
        ML_MAX_GB = float(max_gb)
        logger.info("ml_memory_budget: max budget set to %.1f GB.", ML_MAX_GB)


# ---------------------------------------------------------------------------
# §3.9.5  Startup reconciliation
# ---------------------------------------------------------------------------


def _reconcile_on_startup() -> None:
    """Setzt zurück: allocated budget to 0 on fresh process start (§3.9.5).

    Rationale: All allocations from a previous process are gone after OS
    cleanup (SIGKILL / crash). Each module re-registers via try_allocate()
    when it actually loads its model. No stale allocation persists across
    process boundaries.  Called once at module import time.
    """
    global _total_gb  # pylint: disable=global-statement
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
        """Delegiert an das modulare try_allocate."""
        return try_allocate(model_name, size_gb)

    def release(self, model_name: str) -> None:
        """Delegiert an das modulare release."""
        release(model_name)

    def get_status(self) -> dict:
        """Delegiert an das modulare get_status."""
        return get_status()


_proxy_instance = _MLMemoryBudgetProxy()


def get_ml_memory_budget() -> _MLMemoryBudgetProxy:
    """Gibt the global ML-memory-budget proxy (singleton-safe) zurück.

    Provides an OO API (.try_allocate / .release) in addition to the
    module-level functions, so both usage styles work.
    """
    return _proxy_instance
