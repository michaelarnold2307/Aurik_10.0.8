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

import gc
import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

try:
    import psutil as _psutil
except ImportError:
    _psutil = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
_RAM_EVICT_THRESHOLD_PCT: float = 75.0  # RAM% ab der Eviction beginnt (gesenkt: früher reagieren)
_RAM_TARGET_PCT: float = 65.0  # RAM% auf die wir evicten wollen
_MIN_FREE_MB_HARD: float = 3000.0  # immer mind. 3 GB frei halten
_PIPELINE_EMERGENCY_PCT: float = 70.0  # RAM% ab der auch WÄHREND Pipeline evicted wird (gesenkt von 78%)
_SWAP_EVICT_THRESHOLD_PCT: float = 80.0  # Swap% ab der Eviction erzwungen wird (unabhängig von RAM-%).
# Rationale: Crash 2026-04-14 — swap=99 %, avail=18.79 GB → OOM-Killer wegen
# Apollo-TorchScript-Allokation. RAM-only-Guards erkannten die Gefahr nicht.
_MONITOR_JOIN_TIMEOUT_S: float = 1.0  # Shutdown darf Tests/App-Ende nicht unbounded blockieren
# Begründung Absenkung: 82% war zu konservativ — AudioSR (7 GB) konnte nicht geladen werden weil
# Eviction erst bei 82% erlaubt war, RAM aber schon bei 78% knapp wurde. Modelle die gerade
# NICHT in Inferenz sind (active=False) können sicher entladen werden auch bei 78%.

# ---------------------------------------------------------------------------
# §2.37 Phase-zu-Modell-Mapping: Welche ML-Modelle braucht welche Phase?
# Nur Phasen mit ML-Modellen sind hier gelistet. DSP-only-Phasen fehlen bewusst.
# Modellnamen müssen EXAKT mit dem Namen in ml_memory_budget.try_allocate() übereinstimmen.
# ---------------------------------------------------------------------------
_PHASE_REQUIRED_MODELS: dict[str, frozenset[str]] = {
    "phase_01_click_removal": frozenset({"DeepFilterNetV3"}),
    "phase_02_hum_removal": frozenset({"DeepFilterNetV3"}),
    "phase_03_denoise": frozenset({"SGMSE+", "ResembleEnhance", "DeepFilterNetV3"}),
    "phase_06_frequency_restoration": frozenset({"AudioSR"}),
    "phase_09_crackle_removal": frozenset({"BanquetVinyl"}),
    "phase_12_wow_flutter_fix": frozenset({"FCPE", "RMVPE", "CREPE"}),
    "phase_18_noise_gate": frozenset({"SileroVAD"}),
    "phase_20_reverb_reduction": frozenset(
        {"SGMSE+", "ResembleEnhance"}
    ),  # §4.6c: HybridDereverb uses SGMSE+ primary + ResembleEnhance fallback
    "phase_23_spectral_repair": frozenset({"Apollo", "AudioSR"}),
    "phase_24_dropout_repair": frozenset(
        {"AudioSR", "GACELA", "AudioLDM2"}
    ),  # §4.6c: cascade DSP→GACELA→AudioSR→AudioLDM2
    "phase_29_tape_hiss_reduction": frozenset({"DeepFilterNetV3"}),
    "phase_31_speed_pitch_correction": frozenset(
        {"BasicPitch", "FCPE", "RMVPE", "CREPE"}
    ),  # §4.6c: HybridSpeedPitch loads FCPE→RMVPE→CREPE cascade
    "phase_42_vocal_enhancement": frozenset(
        {"MelBandRoformer", "MDX23C_vocals", "MDX23C_inst", "DemucsV4"}
    ),  # §4.6c: DemucsV4 is stem-sep fallback
    "phase_43_ml_deesser": frozenset({"MP-SENet"}),
    "phase_49_advanced_dereverb": frozenset({"SGMSE+"}),
    "phase_55_diffusion_inpainting": frozenset(
        {"CQTdiffPlus", "FlowMatching", "DiffWave", "ConsistencyInpaint", "DACInpaint"}
    ),
    "phase_56_spectral_band_gap_repair": frozenset({"FCPE", "RMVPE", "CREPE"}),  # §4.6c: f0-cascade FCPE→RMVPE→CREPE
}


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

    Lock-order: Priority 2 (PLM) — see §3.9.8.
    Never acquire PLM._lock while already holding MLMemoryBudget._lock (Priority 1).
    Always acquire PLM._lock BEFORE AdaptiveResourceManager.lock (Priority 3).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, _PluginEntry] = {}
        self._auto_evict_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pipeline_active: int = 0  # Refcount: >0 suppresses auto-eviction during pipeline
        self._start_auto_evict_monitor()
        logger.info("PluginLifecycleManager: initialized (RAM eviction threshold %.0f %%)", _RAM_EVICT_THRESHOLD_PCT)

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

    def enter_pipeline(self) -> int:
        """Increment pipeline-active refcount in a thread-safe way."""
        with self._lock:
            self._pipeline_active += 1
            return self._pipeline_active

    def leave_pipeline(self) -> int:
        """Decrement pipeline-active refcount in a thread-safe way."""
        with self._lock:
            self._pipeline_active = max(0, self._pipeline_active - 1)
            return self._pipeline_active

    def pipeline_active_count(self) -> int:
        """Return current pipeline-active refcount thread-safely."""
        with self._lock:
            return self._pipeline_active

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
        # §Safety: Während Pipeline-Ausführung normalerweise keine automatische
        # Eviction — ONNX-Session-Destruktoren können mit laufender Inferenz
        # kollidieren. ABER: bei kritischem RAM-Druck (>= 82%) MUSS trotzdem
        # evicted werden, sonst killt systemd-oomd den gesamten Prozess.
        if self._pipeline_active > 0 and required_mb <= 0:
            if ram_pct < _PIPELINE_EMERGENCY_PCT and free_mb >= _MIN_FREE_MB_HARD:
                return 0
            logger.warning(
                "PLM: Pipeline aktiv, aber RAM kritisch (%.0f %%, %.0f MB frei) "
                "— erzwinge Notfall-Eviction inaktiver Plugins",
                ram_pct,
                free_mb,
            )
        # required_mb kommt bereits MIT Margin aus ml_memory_budget._preflight_system_memory.
        # Keine doppelte Margin (war 1.25×) — direkt prüfen ob genug frei ist.
        swap_pct = self._swap_percent()
        needs_evict = (
            ram_pct > _RAM_EVICT_THRESHOLD_PCT
            or free_mb < _MIN_FREE_MB_HARD
            or (required_mb > 0 and free_mb < required_mb)
            or swap_pct > _SWAP_EVICT_THRESHOLD_PCT  # Swap-Druck allein reicht für Eviction
        )
        if swap_pct > _SWAP_EVICT_THRESHOLD_PCT and not (
            ram_pct > _RAM_EVICT_THRESHOLD_PCT or free_mb < _MIN_FREE_MB_HARD
        ):
            logger.warning(
                "PLM: Swap-Druck kritisch (%.0f %%) — erzwinge Eviction inaktiver Plugins (RAM=%.0f %%, frei=%.0f MB)",
                swap_pct,
                ram_pct,
                free_mb,
            )
        if not needs_evict:
            return 0
        return self._do_evict(target_pct=_RAM_TARGET_PCT, required_mb=required_mb)

    def force_evict_all(self) -> int:
        """Entlädt ALLE registrierten, inaktiven Plugins sofort.

        Wird vom AurikDenker nach abgeschlossener Batch-Datei aufgerufen.
        """
        return self._do_evict(target_pct=0.0, force_all=True)

    def evict_for_phase(self, phase_id: str) -> int:
        """Entlädt alle ML-Modelle die für die kommende Phase NICHT benötigt werden.

        §2.37 Automatische RAM-Verwaltung: Vor jeder Phase werden nur die
        tatsächlich benötigten Modelle im RAM behalten. Alle anderen —
        auch während aktiver Pipeline — werden entladen.

        Sicher: Nur inaktive (nicht gerade in Inferenz befindliche) Modelle
        werden entladen. Aktive Modelle (entry.active=True) bleiben geschützt.

        Args:
            phase_id: Die nächste auszuführende Phase (z. B. "phase_06_frequency_restoration").

        Returns:
            Anzahl der entladenen Plugins.
        """
        needed = _PHASE_REQUIRED_MODELS.get(phase_id, frozenset())

        with self._lock:
            candidates = [e for e in self._entries.values() if not e.active and e.name not in needed]
            # LRU: älteste zuerst
            candidates.sort(key=lambda e: e.last_used_ts)

        if not candidates:
            return 0

        evicted = 0
        for entry in candidates:
            try:
                logger.info(
                    "PLM: Entlade '%s' (%.2f GB) vor %s — nicht benötigt",
                    entry.name,
                    entry.size_gb,
                    phase_id,
                )
                entry.unload_fn()
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release(entry.name)
                except ImportError:
                    pass
                with self._lock:
                    self._entries.pop(entry.name, None)
                evicted += 1
            except Exception as exc:
                logger.warning("PLM: Fehler beim Entladen von '%s': %s", entry.name, exc)

        if evicted > 0:
            # GC und malloc_trim EINMAL nach der Schleife statt pro Plugin.
            # Jedes gc.collect() hält die Python-GIL für 50–200 ms →
            # pro-Plugin-Aufruf blockierte den Qt-Hauptthread kumulativ.
            gc.collect()
            time.sleep(0)  # GIL explizit freigeben → Qt-Event-Loop kann X11-Pings beantworten
            # NOTE: malloc_trim(0) entfernt — kann SIGABRT verursachen wenn
            # sbrk() aus diesem Thread gleichzeitig mit numpy-Allokationen
            # im Restaurierungs-Thread läuft (gleiche Root-Cause wie _do_evict).
            logger.info(
                "PLM: %d Plugin(s) entladen vor %s — RAM nach GC: %.0f %% (%.0f MB frei)",
                evicted,
                phase_id,
                self._ram_percent(),
                self._free_mb(),
            )
        return evicted

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
            # GC und malloc_trim EINMAL nach der Schleife statt pro Plugin.
            # Jedes gc.collect() hält die Python-GIL für 50–200 ms →
            # pro-Plugin-Aufruf blockierte den Qt-Hauptthread kumulativ.
            gc.collect()
            time.sleep(0)  # GIL explizit freigeben → Qt-Event-Loop kann X11-Pings beantworten
            # NOTE: malloc_trim(0) wurde entfernt. Es kann SIGABRT verursachen wenn
            # sbrk() im PLM-Thread gleichzeitig mit numpy-Allokationen (z.B.
            # sliding_window_view.copy()) im Restaurierungs-Thread läuft.
            # gc.collect() ist für die RAM-Freigabe ausreichend.
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
        """Stoppt den Monitor-Thread best-effort ohne unbounded block."""
        self._stop_event.set()
        thread = self._auto_evict_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_MONITOR_JOIN_TIMEOUT_S)
        self._auto_evict_thread = None

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

    @staticmethod
    def _swap_percent() -> float:
        """Return swap usage in percent (0–100). 0 if psutil unavailable or no swap."""
        if _psutil is not None:
            try:
                return float(_psutil.swap_memory().percent)
            except Exception:
                return 0.0
        return 0.0

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


def evict_for_phase(phase_id: str) -> int:
    """Entlädt alle ML-Modelle die für ``phase_id`` NICHT benötigt werden.

    §2.37: Vor jeder Phase nur benötigte Modelle im RAM behalten.
    """
    return get_plugin_lifecycle_manager().evict_for_phase(phase_id)


def set_pipeline_active(active: bool) -> None:
    """Sperrt/entsperrt automatische Plugin-Eviction während Pipeline-Ausführung.

    Uses a refcount so nested enter/leave pairs work correctly:
    AurikDenker._run_rest() → UV3._execute_pipeline() both call this.
    Eviction blocked while count > 0.
    """
    mgr = get_plugin_lifecycle_manager()
    if active:
        mgr.enter_pipeline()
    else:
        mgr.leave_pipeline()


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
