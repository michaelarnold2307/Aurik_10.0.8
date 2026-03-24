"""backend/core/plugin_lifecycle_manager.py — Automatisches Plugin-RAM-Management.

Zentraler Manager, der:
  1. Alle registrierten unload_*()-Funktionen kennt (Singleton-Muster).
  2. Bei RAM-Druck (> 85 % oder psutil-Messung) automatisch selten genutzte
     Plugins entlädt (LRU-Strategie — zuletzt weniger genutzt zuerst).
  3. Wird vom AurikDenker nach jeder Datei aufgerufen (Batch-Cleanup).
  4. Kann jederzeit manuell via force_evict_all() geleert werden.

Kein ML-Modell wird dabei ohne Zustimmung der Pipeline beendet —
nur Modelle AUSSERHALB der laufenden Phase werden evicted.

Thread-sicher: alle öffentlichen Methoden sind Lock-geschützt.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
import logging
import threading
import time

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
_RAM_EVICT_THRESHOLD_PCT: float = 82.0  # RAM% ab der Eviction beginnt
_RAM_TARGET_PCT: float = 70.0  # RAM% auf die wir evicten wollen
_MIN_FREE_MB_HARD: float = 1500.0  # immer mind. 1.5 GB frei halten


# ---------------------------------------------------------------------------
# Registry-Eintrag
# ---------------------------------------------------------------------------


class _PluginEntry:
    __slots__ = ("active", "last_used_ts", "name", "size_gb", "unload_fn")

    def __init__(
        self,
        name: str,
        size_gb: float,
        unload_fn: Callable[[], None],
    ) -> None:
        self.name = name
        self.size_gb = size_gb
        self.unload_fn = unload_fn
        self.last_used_ts: float = time.monotonic()
        self.active: bool = False  # True = darf NICHT evicted werden (Phase läuft)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: PluginLifecycleManager | None = None
_singleton_lock = threading.Lock()


def get_plugin_lifecycle_manager() -> PluginLifecycleManager:
    """Gibt den PluginLifecycleManager-Singleton zurück (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _singleton_lock:
            if _instance is None:
                _instance = PluginLifecycleManager()
    return _instance


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class PluginLifecycleManager:
    """LRU-basierter Plugin-Memory-Manager mit automatischer Eviction.

    Registrierte Plugins werden nach LRU (Least-Recently-Used) entladen,
    sobald der Systemspeicher unter den Schwellwert fällt.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _PluginEntry] = {}
        self._auto_evict_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipeline_active: int = 0  # Refcount: >0 suppresses auto-eviction during pipeline
        self._start_auto_evict_monitor()
        logger.info("PluginLifecycleManager: initialisiert (RAM-Threshold %.0f %%)", _RAM_EVICT_THRESHOLD_PCT)

    # ------------------------------------------------------------------
    # Registrierung
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        size_gb: float,
        unload_fn: Callable[[], None],
    ) -> None:
        """Registriert ein Plugin mit seiner Unload-Funktion.

        Idempotent — mehrfacher Aufruf mit demselben `name` aktualisiert
        nur `last_used_ts` und `size_gb`.

        Args:
            name:      Eindeutiger Plugin-Name (z. B. 'MelBandRoformer').
            size_gb:   Geschätzter RAM-Verbrauch des Modells in GB.
            unload_fn: Callable das das Modell aus dem RAM entlädt
                       (ruft typisch `global _instance; _instance = None; gc.collect()` auf).
        """
        with self._lock:
            if name in self._entries:
                self._entries[name].last_used_ts = time.monotonic()
                self._entries[name].size_gb = size_gb
                return
            self._entries[name] = _PluginEntry(name, size_gb, unload_fn)
            logger.debug("PLM: '%s' registriert (%.2f GB).", name, size_gb)

    def touch(self, name: str) -> None:
        """Aktualisiert 'last_used_ts' für `name` (vor jeder Plugin-Nutzung aufrufen)."""
        with self._lock:
            if name in self._entries:
                self._entries[name].last_used_ts = time.monotonic()

    def set_active(self, name: str, active: bool) -> None:
        """Markiert ein Plugin als aktiv (Eviction gesperrt) oder inaktiv."""
        with self._lock:
            if name in self._entries:
                self._entries[name].active = active

    def unregister(self, name: str) -> None:
        """Entfernt ein Plugin aus der Registry (nach manuell erfolgtem Unload)."""
        with self._lock:
            self._entries.pop(name, None)

    # ------------------------------------------------------------------
    # Manuelle Eviction
    # ------------------------------------------------------------------

    def evict_if_needed(self, required_mb: float = 0.0) -> int:
        """Entlädt inaktive Plugins falls RAM-Druck besteht.

        Args:
            required_mb: Mindest-RAM der sofort benötigt wird [MB].
                         Falls 0, wird nur auf Basis des RAM-% entschieden.

        Returns:
            Anzahl der entladenen Plugins.
        """
        ram_pct = self._ram_percent()
        free_mb = self._free_mb()
        # §Safety: Während Pipeline-Ausführung keine automatische Eviction —
        # ONNX-Session-Destruktoren können mit laufender Inferenz kollidieren
        # (double free / heap corruption). Nur force_evict_all() umgeht dies.
        if self._pipeline_active > 0 and required_mb <= 0:
            return 0
        needs_evict = (
            ram_pct > _RAM_EVICT_THRESHOLD_PCT
            or free_mb < _MIN_FREE_MB_HARD
            or (required_mb > 0 and free_mb < required_mb * 1.25)
        )
        if not needs_evict:
            return 0
        return self._do_evict(target_pct=_RAM_TARGET_PCT, required_mb=required_mb)

    def force_evict_all(self) -> int:
        """Entlädt ALLE registrierten, inaktiven Plugins sofort.

        Wird vom AurikDenker nach abgeschlossener Batch-Datei aufgerufen.
        """
        return self._do_evict(target_pct=0.0, force_all=True)

    def _do_evict(
        self,
        target_pct: float = _RAM_TARGET_PCT,
        required_mb: float = 0.0,
        force_all: bool = False,
    ) -> int:
        """Führt die eigentliche Eviction durch (LRU-Reihenfolge)."""
        with self._lock:
            # LRU-Sortierung: ältester Zugriff zuerst; aktive ausgenommen
            candidates: list[_PluginEntry] = sorted(
                [e for e in self._entries.values() if not e.active],
                key=lambda e: e.last_used_ts,
            )

        evicted = 0
        for entry in candidates:
            if not force_all:
                ram_pct = self._ram_percent()
                free_mb = self._free_mb()
                if ram_pct <= target_pct and free_mb >= _MIN_FREE_MB_HARD:
                    break
            try:
                logger.info(
                    "PLM: Entlade '%s' (%.2f GB, last_used=%.0fs ago) …",
                    entry.name,
                    entry.size_gb,
                    time.monotonic() - entry.last_used_ts,
                )
                entry.unload_fn()
                gc.collect()
                # Budget-Freigabe
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release(entry.name)
                except ImportError:
                    pass
                with self._lock:
                    self._entries.pop(entry.name, None)
                evicted += 1
                logger.info("PLM: '%s' erfolgreich entladen.", entry.name)
            except Exception as exc:
                logger.warning("PLM: Fehler beim Entladen von '%s': %s", entry.name, exc)

        if evicted > 0:
            logger.info("PLM: %d Plugin(s) entladen — RAM nach GC: %.0f %%", evicted, self._ram_percent())
        return evicted

    # ------------------------------------------------------------------
    # Automatisches Monitoring
    # ------------------------------------------------------------------

    def _start_auto_evict_monitor(self) -> None:
        """Startet einen Daemon-Thread, der alle 10 s den RAM prüft."""
        self._auto_evict_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="PLM_auto_evict",
        )
        self._auto_evict_thread.start()

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(timeout=10.0):
            try:
                self.evict_if_needed()
            except Exception as exc:
                logger.debug("PLM Monitor-Fehler: %s", exc)

    def shutdown(self) -> None:
        """Stoppt den Monitor-Thread (bei App-Ende)."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # RAM-Messung
    # ------------------------------------------------------------------

    @staticmethod
    def _ram_percent() -> float:
        if _psutil is not None:
            return float(_psutil.virtual_memory().percent)
        return 0.0  # ohne psutil kein automatisches Evict

    @staticmethod
    def _free_mb() -> float:
        if _psutil is not None:
            return float(_psutil.virtual_memory().available / (1024 * 1024))
        return float("inf")

    # ------------------------------------------------------------------
    # Status (Logging/Debug)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "ram_pct": round(self._ram_percent(), 1),
                "free_mb": round(self._free_mb(), 0),
                "registered_plugins": [
                    {
                        "name": e.name,
                        "size_gb": e.size_gb,
                        "active": e.active,
                        "idle_s": round(time.monotonic() - e.last_used_ts, 0),
                    }
                    for e in self._entries.values()
                ],
            }


# ---------------------------------------------------------------------------
# Convenience-Funktionen (Modul-Level)
# ---------------------------------------------------------------------------


def register_plugin(name: str, size_gb: float, unload_fn: Callable[[], None]) -> None:
    """Registriert ein Plugin beim globalen Lifecycle-Manager."""
    get_plugin_lifecycle_manager().register(name, size_gb, unload_fn)


def touch_plugin(name: str) -> None:
    """Aktualisiert 'last_used_ts' für ein Plugin (vor Nutzung)."""
    get_plugin_lifecycle_manager().touch(name)


def evict_stale_plugins(required_mb: float = 0.0) -> int:
    """Entlädt inaktive Plugins falls RAM-Druck besteht. Gibt Anzahl zurück."""
    return get_plugin_lifecycle_manager().evict_if_needed(required_mb)


def set_pipeline_active(active: bool) -> None:
    """Sperrt/entsperrt automatische Plugin-Eviction während Pipeline-Ausführung.

    Uses a refcount so nested enter/leave pairs work correctly:
    AurikDenker._run_rest() → UV3._execute_pipeline() both call this.
    Eviction blocked while count > 0.
    """
    mgr = get_plugin_lifecycle_manager()
    if active:
        mgr._pipeline_active += 1
    else:
        mgr._pipeline_active = max(0, mgr._pipeline_active - 1)


def cleanup_after_file() -> int:
    """Vollständiger Cleanup nach Abschluss einer Batch-Datei.

    Entlädt ALLE inaktiven Plugins. Sollte vom AurikDenker nach jeder
    verarbeiteten Datei im Batch-Modus aufgerufen werden.
    """
    mgr = get_plugin_lifecycle_manager()
    n = mgr.force_evict_all()
    gc.collect()
    logger.info("PLM: Batch-File-Cleanup: %d Plugin(s) entladen.", n)
    return n
