# ── Legacy-Testdateien ausschließen ────────────────────────────────────────
# Diese Dateien sind Script-Style-Tests oder testen Module, die nicht mehr
# existieren. Sie werden aus der pytest-Collection ausgeschlossen um
# Collection-Errors zu vermeiden. Neue Unit-Tests gehören nach tests/unit/.
collect_ignore = [
    "tests/test_backend_core.py",
    "tests/test_explainability_engine.py",
    "tests/test_human_in_the_loop.py",
    "tests/test_hybrid_ml_denoiser.py",
    "tests/test_material_detection_debug.py",
    "tests/test_module_communication_bus.py",
    "tests/test_multi_model_ensemble.py",
    "tests/test_multimodal_decision_engine.py",
    "tests/test_optimization_balanced.py",
    "tests/test_phase_29_ml_hybrid.py",
    "tests/test_policy_engine_extended.py",
    "tests/test_realtime_feedback_bus.py",
    "tests/test_restoration_workflow.py",
    "tests/test_transfer_learning.py",
    "tests/test_validate_musical_goals.py",
    "tests/musical_goals/test_musical_goals_monitor_api.py",
    "tests/musical_goals/test_uncertainty_quantification.py",
    # Script-style Dateien (kein pytest-konformes Test-Layout, Modul-Level-Code):
    "tests/test_phase_02_ml_hybrid.py",
    # ONNX-Tests benötigen torch (OSError: libcupti.so.12 in dieser venv):
    "tests/onnx/test_onnx_advanced.py",
    "tests/onnx/test_onnx_runtime.py",
    "tests/onnx/test_plugin_manager.py",
]


"""
Root-Level conftest.py — wird von pytest VOR allen Sub-Verzeichnis-conftest.py-Dateien
geladen. Setzt BLAS-Thread-Limits bevor numpy importiert wird, um unter pytest-xdist
(parallele Worker-Prozesse) OpenBLAS-Thread-Oversubscription zu verhindern.

Problem ohne diesen Fix:
  8 xdist-Worker × OPENBLAS_NUM_THREADS=8 (Default) = 64 BLAS-Threads konkurrieren
  um dieselben CPU-Kerne → np.linalg.solve auf 256×256-Matrizen und np.roots auf
  Polynomen Grad 512 hängen länger als das 30s-pytest-Timeout.

Lösung:
  Jeder Worker nutzt genau 1 BLAS-Thread → keine gegenseitige Blockierung.
  Python-Level-Parallelismus (xdist-Worker-Prozesse) bleibt vollständig erhalten.

VS Code-Volltest-Stabilität (Anti-OOM + Anti-Pipe-Flood):
  URSACHE 1 — Pipe-Flood:
    VS Code vscode_pytest schickt fuer JEDEN der 5 200+ Tests eine eigene JSON-RPC-
    Nachricht ueber TEST_RUN_PIPE. Bei 5 200 Nachrichten laeuft der Extension-Host-
    Puffer voll → VS Code friert ein oder stuerzt ab.
    FIX: settings.json beschraenkt Test Explorer auf tests/unit/ (50 Dateien, ~600
    Tests). Vollsuite laeuft ausschliesslich ueber Terminal-Tasks (tasks.json).

  URSACHE 2 — ONNX-Singleton-OOM:
    Plugin-Singletons halten ONNX-Sessions (37-250 MB/Plugin). Nach ~267 Dateien
    akkumuliert sich >750 MB Singleton-RAM zusaetzlich zu VS Code (~1 GB).
    FIX (hier): _release_heavy_singletons() nach jedem Datei-Wechsel + GC.

  URSACHE 3 — Collection-Phase-Import-Kaskade:
    pytest laedt beim Sammeln ALLE 252 Test-Dateien (auch ohne sie auszufuehren).
    Module-Imports in Testdateien koennen ONNX-Sessions vorzeitig triggern.
    FIX (hier): pytest_collection_finish gibt Singletons direkt nach Collection frei.
"""

import gc as _gc
import os
import sys as _sys
import warnings as _warnings

# Muss VOR dem ersten numpy-Import gesetzt werden.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

_VSCODE_LAST_FILE: str = ""
_VSCODE_TEST_COUNTER: int = 0
_VSCODE_GC_INTERVAL: int = 100


def pytest_configure(config) -> None:  # noqa: ANN001
    """Wird im Haupt-Thread vor allen Tests aufgerufen.

    Löst alle librosa-Lazy-Submodule auf, BEVOR xdist-Worker-Prozesse starten.
    Verhindert den Python Import-Lock-Deadlock (zwei Threads importieren
    librosa.util gleichzeitig via lazy_loader.__getattr__).

    Guard-Logik:
        1. PYTEST_XDIST_WORKER gesetzt → wir SIND BEREITS ein Worker-Prozess.
           Worker importieren librosa lazy, kein paralleler Lock-Konflikt mehr.
           → sofortiger Return, kein Warm-up.
        2. numprocesses == 0 → kein xdist, kein paralleler Import-Lock möglich.
           → sofortiger Return, kein Warm-up (verhindert Numba-JIT-Hang bei
           Einzelprozess-Läufen und beim VS Code Test Explorer).
        3. numprocesses > 0 → Haupt-Prozess mit xdist-Workern. Warm-up here
           löst alle Lazy-Submodule auf, bevor fork() Worker-Prozesse erzeugt.
    """
    # Guard 1: Wir sind bereits in einem xdist-Worker → kein Warm-up nötig.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    # Guard 2: Kein xdist → kein Import-Lock-Risiko → Warm-up überspringen.
    # (Verhindert Numba-JIT-Hang bei pytest -p no:xdist und VS Code Test Explorer)
    numprocs = getattr(getattr(config, "option", None), "numprocesses", 0) or 0
    if not numprocs:
        return

    # Guard 3: xdist aktiv (numprocs > 0) → Warm-up im Haupt-Prozess.
    try:
        import librosa as _librosa
        import numpy as _np

        _d = _np.zeros(4096, dtype=_np.float32)
        _d[::4] = 0.1
        _librosa.stft(_d, n_fft=512, hop_length=128)
        _librosa.feature.mfcc(y=_d, sr=4000, n_mfcc=13)
        _librosa.feature.spectral_centroid(y=_d, sr=4000)
        _librosa.feature.spectral_rolloff(y=_d, sr=4000)
        _librosa.feature.chroma_stft(y=_d, sr=4000)
        _librosa.feature.rms(y=_d)
        _librosa.onset.onset_strength(y=_d, sr=4000)
        _librosa.beat.beat_track(y=_d, sr=4000)
        _ = _librosa.util.MAX_MEM_BLOCK
        _ = _librosa.util.frame
    except Exception:  # noqa: BLE001
        pass  # Kein Absturz — ist nur ein Warm-up


# Schwere ML-Plugin-Module mit ONNX-Sessions (je 37-250 MB RAM)
_HEAVY_PLUGIN_MODULES: tuple = (
    "plugins.deepfilternet_v3_ii_plugin",
    "plugins.apollo_plugin",
    "plugins.vocos_plugin",
    "plugins.crepe_plugin",
    "plugins.banquet_vinyl_plugin",
    "plugins.resemble_enhance_plugin",
    "plugins.panns_plugin",
    "plugins.hifigan_plugin",
    "plugins.uvr_mdxnet_plugin",
    "plugins.mp_senet_plugin",   # §4.4: MP-SENet 2023 (ersetzt DCCRN + FullSubNet+)
    "plugins.versa_plugin",       # §4.4: VERSA 2024 MOS (ersetzt CDPAM)
    "plugins.diffwave_plugin",
    "plugins.bs_roformer_plugin",
    "plugins.mdx23c_plugin",
    "plugins.demucs_v4_plugin",
    "plugins.wpe_plugin",
    "plugins.laion_clap_plugin",
    "plugins.flow_matching_plugin",
    "plugins.cqtdiff_plus_plugin",
)

_VSCODE_SAFE_TEST_LIMIT: int = 1_500


def _release_heavy_singletons() -> None:
    """Setzt _instance in allen schweren Plugin-Modulen auf None."""
    for mod_name in _HEAVY_PLUGIN_MODULES:
        mod = _sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_instance"):
            try:
                mod._instance = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass


def _is_vscode_run() -> bool:
    """True wenn pytest von VS Code Test Explorer gestartet wurde."""
    return "TEST_RUN_PIPE" in os.environ or "VSCODE_PID" in os.environ


def pytest_collection_finish(session) -> None:  # noqa: ANN001
    """Nach vollstaendiger Test-Collection: Singletons freigeben + Warnung.

    Collection importiert alle Test-Module, was ONNX-Sessions vorzeitig
    triggern kann. Einmalig freigeben bevor Ausfuehrung startet.
    Warnt bei VS Code-Laeufen mit > 1 500 Tests.
    """
    n_tests = len(session.items)

    if _is_vscode_run() and n_tests > _VSCODE_SAFE_TEST_LIMIT:
        _warnings.warn(
            f"\n\nAURIK WARNUNG: {n_tests:,} Tests im VS Code Test Explorer.\n"
            f"Bei > {_VSCODE_SAFE_TEST_LIMIT:,} Tests droht VS Code-Absturz (Pipe-Flood).\n"
            f"Loesung: Test Explorer auf tests/unit/ beschraenken (settings.json)\n"
            f"Vollsuite: Terminal → Strg+Shift+B → 'Vollsuite parallel'\n",
            stacklevel=1,
            category=UserWarning,
        )

    _release_heavy_singletons()
    _gc.collect()


def pytest_runtest_teardown(item, nextitem) -> None:  # noqa: ANN001
    """GC und Singleton-Freigabe: nach Datei-Wechsel und alle 100 Tests.

    Verhindert den VS Code-OOM-Crash bei ~97 % der 5236-Test-Suite.
    """
    global _VSCODE_LAST_FILE, _VSCODE_TEST_COUNTER  # noqa: PLW0603

    _VSCODE_TEST_COUNTER += 1
    current_file = str(getattr(item, "fspath", ""))

    if current_file and current_file != _VSCODE_LAST_FILE:
        _VSCODE_LAST_FILE = current_file
        _release_heavy_singletons()
        _gc.collect()
    elif _VSCODE_TEST_COUNTER % _VSCODE_GC_INTERVAL == 0:
        _gc.collect()
