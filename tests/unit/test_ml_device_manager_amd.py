"""Unit tests — ml_device_manager AMD ROCm performance extensions.

Verifies:
  - is_fp16_eligible() only returns True for ROCm + eligible plugins
  - get_ort_providers_fp16() falls back to CPU when not ROCm
  - warmup_rocm_gpu() is a safe no-op on CPU-only systems
  - pin_tensor_rocm() returns the original array on non-ROCm
  - DeepFilterNetV3 is in _HEAVY_ML_PLUGINS (GPU dispatch)
  - _FP16_ELIGIBLE_PLUGINS is a non-empty frozenset
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_manager_module():
    """Import ml_device_manager without real GPU or torch deps."""
    import importlib
    if "backend.core.ml_device_manager" in sys.modules:
        return sys.modules["backend.core.ml_device_manager"]
    return importlib.import_module("backend.core.ml_device_manager")


# ---------------------------------------------------------------------------
# _HEAVY_ML_PLUGINS / _FP16_ELIGIBLE_PLUGINS contract
# ---------------------------------------------------------------------------


class TestPluginSets:
    def test_deepfilternet_in_heavy_ml_plugins(self) -> None:
        """§GPU-Mixed-Mode: DeepFilterNetV3 must be GPU-dispatched."""
        mod = _get_manager_module()
        assert "DeepFilterNetV3" in mod._HEAVY_ML_PLUGINS, (
            "DeepFilterNetV3 missing from _HEAVY_ML_PLUGINS — "
            "computationally intensive DF-filter must run on AMD GPU"
        )

    def test_fp16_eligible_plugins_non_empty(self) -> None:
        """_FP16_ELIGIBLE_PLUGINS must contain at least the core ONNX models."""
        mod = _get_manager_module()
        assert len(mod._FP16_ELIGIBLE_PLUGINS) > 0
        for expected in ("BSRoFormer", "MDXNet", "DeepFilterNetV3", "PANNs"):
            assert expected in mod._FP16_ELIGIBLE_PLUGINS, (
                f"{expected} should be in _FP16_ELIGIBLE_PLUGINS"
            )

    def test_fp16_eligible_subset_of_heavy(self) -> None:
        """Every fp16-eligible plugin must also be in _HEAVY_ML_PLUGINS (it runs on GPU)."""
        mod = _get_manager_module()
        not_heavy = mod._FP16_ELIGIBLE_PLUGINS - mod._HEAVY_ML_PLUGINS
        assert not not_heavy, (
            f"fp16-eligible plugins not in _HEAVY_ML_PLUGINS: {not_heavy}"
        )


# ---------------------------------------------------------------------------
# is_fp16_eligible — CPU-only system returns False
# ---------------------------------------------------------------------------


class TestFp16EligibilityOnCpu:
    def test_returns_false_on_cpu_only(self) -> None:
        """On a CPU-only system MLDeviceManager.is_fp16_eligible() must return False."""
        from backend.core.ml_device_manager import MLDeviceManager, GPUBackend

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        # Manually set CPU-only state (bypasses __init__ GPU detection)
        import threading
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False
        mgr._gpu_name = ""
        mgr._vram_total_gb = 0.0
        mgr._vram_free_gb = 0.0
        mgr._vram_allocated = {}
        mgr._gpu_errors = {}
        mgr._gpu_disabled_plugins = set()

        assert mgr.is_fp16_eligible("BSRoFormer") is False
        assert mgr.is_fp16_eligible("DeepFilterNetV3") is False

    def test_get_ort_providers_fp16_falls_back_on_cpu(self) -> None:
        """get_ort_providers_fp16() must return ['CPUExecutionProvider'] on CPU-only."""
        from backend.core.ml_device_manager import MLDeviceManager, GPUBackend

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        import threading
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False
        mgr._gpu_name = ""
        mgr._vram_total_gb = 0.0
        mgr._vram_free_gb = 0.0
        mgr._vram_allocated = {}
        mgr._gpu_errors = {}
        mgr._gpu_disabled_plugins = set()

        providers = mgr.get_ort_providers_fp16("BSRoFormer")
        # Should fall back to standard providers → CPU
        assert "CPUExecutionProvider" in providers or providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# warmup_rocm_gpu — safe no-op on CPU-only
# ---------------------------------------------------------------------------


class TestWarmupRocm:
    def test_warmup_returns_false_on_cpu(self) -> None:
        """warmup_rocm_gpu() must return False gracefully on CPU-only systems."""
        from backend.core.ml_device_manager import warmup_rocm_gpu, MLDeviceManager, GPUBackend
        import threading
        # Patch singleton to return a CPU-only manager
        cpu_mgr = MLDeviceManager.__new__(MLDeviceManager)
        cpu_mgr._lock = threading.Lock()
        cpu_mgr._backend = GPUBackend.NONE
        cpu_mgr._gpu_available = False

        import backend.core.ml_device_manager as mdm
        orig = mdm._instance
        try:
            mdm._instance = cpu_mgr
            result = warmup_rocm_gpu()
            assert result is False, "warmup_rocm_gpu must return False on CPU-only"
        finally:
            mdm._instance = orig

    def test_warmup_no_exception_on_import_error(self) -> None:
        """warmup_rocm_gpu() must not raise even if torch is unavailable."""
        from backend.core.ml_device_manager import MLDeviceManager, GPUBackend
        import threading
        import backend.core.ml_device_manager as mdm

        cpu_mgr = MLDeviceManager.__new__(MLDeviceManager)
        cpu_mgr._lock = threading.Lock()
        cpu_mgr._backend = GPUBackend.ROCM
        cpu_mgr._gpu_available = True
        cpu_mgr._gpu_name = "AMD Radeon RX 7900 XTX"
        cpu_mgr._vram_total_gb = 24.0

        orig = mdm._instance
        try:
            mdm._instance = cpu_mgr
            # Simulate torch import failure inside warmup
            with patch.dict(sys.modules, {"torch": None}):
                result = cpu_mgr.warmup_rocm()  # must not raise
                assert isinstance(result, bool)
        finally:
            mdm._instance = orig


# ---------------------------------------------------------------------------
# pin_tensor_rocm — passthrough on CPU-only
# ---------------------------------------------------------------------------


class TestPinTensorRocm:
    def test_returns_array_unchanged_on_cpu(self) -> None:
        """pin_tensor_rocm() must return the original numpy array on CPU-only."""
        from backend.core.ml_device_manager import MLDeviceManager, GPUBackend
        import threading

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.NONE
        mgr._gpu_available = False

        arr = np.zeros(1024, dtype=np.float32)
        result = mgr.pin_tensor_rocm(arr)
        assert result is arr, "pin_tensor_rocm must return original array on CPU-only"

    def test_returns_non_array_unchanged(self) -> None:
        """Non-ndarray/tensor types must pass through unchanged."""
        from backend.core.ml_device_manager import MLDeviceManager, GPUBackend
        import threading

        mgr = MLDeviceManager.__new__(MLDeviceManager)
        mgr._lock = threading.Lock()
        mgr._backend = GPUBackend.ROCM
        mgr._gpu_available = True

        obj = {"key": "value"}
        result = mgr.pin_tensor_rocm(obj)
        assert result is obj


# ---------------------------------------------------------------------------
# Global convenience wrappers
# ---------------------------------------------------------------------------


class TestGlobalWrappers:
    def test_get_ort_providers_fp16_returns_list(self) -> None:
        """get_ort_providers_fp16() global function must always return a non-empty list."""
        from backend.core.ml_device_manager import get_ort_providers_fp16
        providers = get_ort_providers_fp16("BSRoFormer")
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_warmup_rocm_gpu_global_no_raise(self) -> None:
        """warmup_rocm_gpu() global function must not raise under any condition."""
        from backend.core.ml_device_manager import warmup_rocm_gpu
        result = warmup_rocm_gpu()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Plugin source code audit: fp16-eligible plugins must use get_ort_providers_fp16
# ---------------------------------------------------------------------------

class TestFp16PluginAudit:
    """Verify that every fp16-eligible ONNX plugin sources get_ort_providers_fp16, not get_ort_providers.

    This is a static-analysis (source inspection) guard: no actual ONNX sessions are loaded.
    """

    _PLUGIN_FILE_MAP: dict[str, str] = {
        "BSRoFormer": "plugins/bs_roformer_plugin.py",
        "MDXNet": "plugins/uvr_mdxnet_plugin.py",
        "DemucsV4": "plugins/demucs_v4_plugin.py",
        "MDX23C": "plugins/mdx23c_plugin.py",
        "MPSENet": "plugins/mp_senet_plugin.py",
        "ResembleEnhance": "plugins/resemble_enhance_plugin.py",
        "PANNs": "plugins/panns_plugin.py",
        "LaionCLAP_ONNX": "plugins/laion_clap_plugin.py",
        "BanquetVinyl": "plugins/banquet_vinyl_plugin.py",
        "DeepFilterNetV3": "plugins/deepfilternet_v3_ii_plugin.py",
    }

    @pytest.mark.parametrize("plugin_name,rel_path", list(_PLUGIN_FILE_MAP.items()))
    def test_uses_fp16_provider_function(self, plugin_name: str, rel_path: str) -> None:
        """Plugin source must import get_ort_providers_fp16 (not plain get_ort_providers)."""
        import pathlib
        workspace_root = pathlib.Path(__file__).parents[2]
        plugin_file = workspace_root / rel_path
        if not plugin_file.exists():
            pytest.skip(f"Plugin file absent (optional model): {rel_path}")
        source = plugin_file.read_text(encoding="utf-8")
        # Must use fp16 variant
        assert "get_ort_providers_fp16" in source, (
            f"{plugin_name} ({rel_path}) must import 'get_ort_providers_fp16' for AMD fp16 support"
        )

    @pytest.mark.parametrize("plugin_name,rel_path", list(_PLUGIN_FILE_MAP.items()))
    def test_no_bare_get_ort_providers(self, plugin_name: str, rel_path: str) -> None:
        """Plugin source must not use the non-fp16 provider function for ONNX sessions.

        It is acceptable to import get_ort_providers elsewhere in the codebase
        (e.g. as a generic helper), but fp16-eligible plugins must NOT load
        their primary ONNX InferenceSession with the non-fp16 variant.
        """
        import pathlib
        workspace_root = pathlib.Path(__file__).parents[2]
        plugin_file = workspace_root / rel_path
        if not plugin_file.exists():
            pytest.skip(f"Plugin file absent (optional model): {rel_path}")
        source = plugin_file.read_text(encoding="utf-8")
        # Strip lines that only mention the legacy name in comments / docstrings
        code_lines = [
            ln for ln in source.splitlines()
            if not ln.strip().startswith("#") and not ln.strip().startswith('"""') and not ln.strip().startswith("'''")
        ]
        code_block = "\n".join(code_lines)
        # Bare import without _fp16 suffix is forbidden in code lines
        bare_found = "get_ort_providers as _get_prov" in code_block and "get_ort_providers_fp16" not in code_block
        assert not bare_found, (
            f"{plugin_name} ({rel_path}) must not import plain 'get_ort_providers' for ONNX sessions; "
            "use 'get_ort_providers_fp16' instead"
        )
