"""
Adaptive Resource Manager für Aurik 9.0
Automatische CPU- und Speicher-Auslastungsüberwachung und dynamische Core-Zuteilung für ML-Phasen.
Ermöglicht Fallback auf lightweight-Algorithmen bei Ressourcenknappheit.
Funktioniert out-of-the-box, keine Nutzerinteraktion nötig.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


_instance: Optional["AdaptiveResourceManager"] = None
_lock_singleton = threading.Lock()


def get_resource_manager() -> "AdaptiveResourceManager":
    """Get or create AdaptiveResourceManager singleton.

    Returns:
        AdaptiveResourceManager singleton instance
    """
    global _instance
    if _instance is None:
        with _lock_singleton:
            if _instance is None:
                _instance = AdaptiveResourceManager()
    return _instance


class AdaptiveResourceManager:
    def __init__(self, min_cores: int = 2, max_cores: int | None = None, check_interval: float = 2.0, cpu_threshold: int = 80, memory_threshold: int = 85) -> None:
        self.system_cores = mp.cpu_count()
        self.max_cores = max_cores or self.system_cores
        self.min_cores = min_cores
        self.cpu_threshold = cpu_threshold  # in Prozent
        self.memory_threshold = memory_threshold  # in Prozent
        self.check_interval = check_interval
        self.current_cores = self.max_cores
        self.running = False
        self.lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self.use_lightweight = False  # Fallback-Flag
        logger.info(f"AdaptiveResourceManager initialized: {self.min_cores}-{self.max_cores} cores")

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if psutil:
            # interval=None: non-blocking, gibt Wert seit letztem Aufruf zurück
            return float(psutil.cpu_percent(interval=None))
        else:
            return 0.0  # Fallback: keine Überwachung

    def get_memory_usage(self) -> float:
        """Get current system memory usage percentage."""
        if psutil:
            return psutil.virtual_memory().percent
        else:
            return 0  # Fallback: keine Überwachung

    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        if psutil:
            return psutil.virtual_memory().available / (1024 * 1024)
        else:
            return float("inf")  # Fallback: assume unlimited

    def start_monitoring(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self._stop_event = threading.Event()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="ARM_monitor")
            self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self.running = False
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def _monitor_loop(self) -> None:
        import gc as _gc
        stop_event = getattr(self, "_stop_event", None)
        while self.running:
            cpu_usage = self.get_cpu_usage()
            memory_usage = self.get_memory_usage()

            with self.lock:
                # Determine if lightweight mode is needed
                self.use_lightweight = cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold

                # Adjust cores based on CPU usage
                if cpu_usage > self.cpu_threshold:
                    self.current_cores = max(self.min_cores, self.current_cores - 1)
                elif cpu_usage < self.cpu_threshold * 0.7 and self.current_cores < self.max_cores:
                    if memory_usage < self.memory_threshold * 0.8:
                        self.current_cores = min(self.max_cores, self.current_cores + 1)

            # Active eviction when RAM is critically high — outside lock to avoid deadlock
            if memory_usage > self.memory_threshold:
                try:
                    from backend.core.plugin_lifecycle_manager import evict_stale_plugins
                    n_evicted = evict_stale_plugins()
                    if n_evicted > 0:
                        logger.info("ARM: RAM %.1f%% > %d%% threshold — evicted %d plugin(s)", memory_usage, self.memory_threshold, n_evicted)
                except Exception:
                    pass
                _gc.collect()

            # Interruptible sleep — stoppt sofort bei stop_monitoring()
            if stop_event is not None:
                stop_event.wait(timeout=self.check_interval)
                if stop_event.is_set():
                    break
            else:
                time.sleep(self.check_interval)

    def get_num_cores(self) -> int:
        with self.lock:
            return self.current_cores

    def should_use_lightweight_mode(self) -> bool:
        """
        Check if system resources are constrained and lightweight algorithms should be used.

        Returns:
            bool: True if resources are constrained, False otherwise
        """
        with self.lock:
            return self.use_lightweight

    def check_memory_availability(self, required_mb: float) -> bool:
        """
        Check if required memory is available.

        Args:
            required_mb: Required memory in MB

        Returns:
            bool: True if memory is available, False otherwise
        """
        available_mb = self.get_available_memory_mb()

        # Add 20% safety margin
        safety_margin = 1.2
        required_with_margin = required_mb * safety_margin

        return available_mb >= required_with_margin


# Modul-Level-Singleton — start_monitoring() wird NICHT automatisch
# aufgerufen, damit Test-Imports keinen blockierenden Hintergrund-Thread starten.
# Produktions-Code (UnifiedRestorerV3 etc.) ruft start_monitoring() explizit auf.
adaptive_resource_manager = AdaptiveResourceManager(
    min_cores=2, max_cores=mp.cpu_count(), check_interval=2.0, cpu_threshold=80, memory_threshold=85
)
