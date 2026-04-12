"""ml_device_manager — Mixed CPU/GPU Device Selection for Aurik 9.

Detects available GPU acceleration (ROCm on Linux, DirectML on Windows) and
provides a unified interface for per-plugin device assignment.

Mixed-mode operation (§GPU-Mixed-Mode):
  - Heavy ML models (large transformers, diffusion models) → GPU when available
  - DSP modules, lightweight/utility models            → always CPU
  - Any GPU failure                                    → transparent CPU fallback
    (§2.47 ML-Failure-Degradationskaskade)

Platform support:
  Linux:   ROCm 6.x via PyTorch ROCm wheel (uses torch.cuda API identically to CUDA)
  Windows: DirectML via onnxruntime-directml + optionally torch-directml
  Both:    CPU-only fallback (preserves §9.5-compatible behaviour when no GPU present)

VRAM budget (§GPU-VRAM-Guard):
  Analog to ml_memory_budget for system RAM — tracks allocated VRAM per plugin and
  falls back to CPU when VRAM headroom is insufficient. On ROCm, free VRAM is queried
  via torch.cuda.mem_get_info(); on DirectML, manually tracked.

Singleton: get_ml_device_manager()
Convenience wrappers: get_torch_device(plugin_name), get_ort_providers(plugin_name)
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
from typing import Any

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU Backend enum
# ---------------------------------------------------------------------------


class GPUBackend(enum.Enum):
    ROCM = "rocm"  # Linux: ROCm 6.x via torch.cuda API (AMD GPU primary path)
    DIRECTML = "directml"  # Windows: DirectML via onnxruntime-directml (AMD Windows)
    NONE = "none"  # CPU-only (no AMD GPU or GPU suppressed)


# ---------------------------------------------------------------------------
# Heavy ML plugins eligible for GPU acceleration
#
# Only plugins in this frozenset are dispatched to the GPU device.
# DSP modules, lightweight utility models (< ~100 MB), and quality-scoring
# plugins stay on CPU — their compute is too small to amortise the
# CPU↔GPU transfer overhead.
# ---------------------------------------------------------------------------

_HEAVY_ML_PLUGINS: frozenset[str] = frozenset(
    {
        # --- Stem Separation (>500 MB) ---
        "SGMSE",  # sgmse_plugin — diffusion score-matching (251 MB)
        "AudioSR",  # audiosr_plugin — bandwidth extension (7 GB)
        "BSRoFormer",  # bs_roformer_plugin — stem separation ONNX (860 MB)
        "MDXNet",  # uvr_mdxnet_plugin — stem separation ONNX (~1.2 GB)
        "DemucsV5",  # demucs_v5_wrapper — stem separation
        "MDX23C",  # mdx23c wrapper — stem separation
        "DemucsV4",  # demucs_v4_plugin — htdemucs_6s ONNX (~500 MB)
        # --- Neural Vocoders / Enhancement ---
        "BigVGAN",  # bigvgan_v2_plugin — neural vocoder (400 MB)
        "ApolloPlugin",  # apollo_plugin — restorative ML
        "CQTDiffPlus",  # cqtdiff_plus_plugin — diffusion inpainting
        "Gacela",  # gacela_plugin — audio inpainting
        "MPSENet",  # mp_senet plugin — speech enhancement
        "ResembleEnhance",  # resemble_enhance_plugin — ONNX (~722 MB)
        "AudioLDM2",  # audioldm2_plugin — diffusion ONNX (~1.3 GB)
        # --- Neural Enhancement (GPU-accelerated on AMD ROCm) ---
        "DeepFilterNetV3",  # deepfilternet_v3_ii_plugin — 3x enc/dec ONNX (~150 MB)
        #                     Computationally intensive DF-filter; ROCm accelerates
        #                     the iterative ERB-inverse + deep filtering significantly.
        # --- Quality / Scoring (heavy Transformer backbones) ---
        "MERT-330M-HF",  # mert_plugin — HuggingFace Transformer (~1.2 GB)
        "MERT-330M-fairseq",  # mert_plugin — fairseq checkpoint (~3.7 GB)
        "VersaSingMOS",  # versa_plugin — wav2vec2 backbone (~800 MB)
        "UTMOSv2",  # utmos_plugin — torch fold models (~800 MB)
        "BanquetVinyl",  # banquet_vinyl_plugin — ONNX (~800 MB)
        # --- Audio Understanding ---
        "PANNs",  # panns_plugin — CNN14 ONNX (~660 MB)
        "LaionCLAP_ONNX",  # laion_clap_plugin — ONNX audio encoder (~300 MB)
    }
)

# ---------------------------------------------------------------------------
# AMD ROCm fp16 ONNX inference — eligible plugins
# ROCm supports fp16 for all ONNX models via ROCMExecutionProvider.
# fp16 halves VRAM usage and doubles inference throughput at negligible quality loss
# for audio enhancement models (activations already clipped to [-1, 1]).
# ---------------------------------------------------------------------------

_FP16_ELIGIBLE_PLUGINS: frozenset[str] = frozenset(
    {
        # ONNX models where fp16 is safe (activations bounded, no integer ops):
        "BSRoFormer",
        "MDXNet",
        "DemucsV4",
        "MPSENet",
        "ResembleEnhance",
        "PANNs",
        "LaionCLAP_ONNX",
        "BanquetVinyl",
        "DeepFilterNetV3",
        # PyTorch models where .half() is safe on AMD ROCm:
        "BigVGAN",
        "MDX23C",
    }
)

# ---------------------------------------------------------------------------
# VRAM budget constants
# ---------------------------------------------------------------------------

# Keep at least this much VRAM free after allocations (hard floor).
_VRAM_MIN_FREE_MB: float = 512.0
# At most this fraction of total VRAM may be used by Aurik plugins combined.
_VRAM_MAX_USAGE_RATIO: float = 0.85

# ---------------------------------------------------------------------------
# AMD ROCm performance constants
# ---------------------------------------------------------------------------

# Warmup dummy tensor dimensions for ROCm runtime initialisation (sec §GPU-Perf)
_ROCM_WARMUP_TENSOR_LEN: int = 4096  # samples — tiny enough to be instant
# ROCm graph capture is not used (Triton compilation overhead > benefit for audio)
_ROCM_TORCH_THREADS: int = 1  # inter-op threads — GPU already parallel

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_instance: MLDeviceManager | None = None
_init_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Public convenience wrappers
# ---------------------------------------------------------------------------


def get_ml_device_manager() -> MLDeviceManager:
    """Return the process-wide MLDeviceManager singleton (thread-safe, lazy-init)."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = MLDeviceManager()
    return _instance


def get_torch_device(plugin_name: str = "") -> str:
    """Return the PyTorch device string for *plugin_name*.

    Returns ``"cpu"``, ``"cuda"`` (ROCm), or ``"dml"`` (DirectML+torch-directml).
    Heavy plugins (in ``_HEAVY_ML_PLUGINS``) get the GPU device when available;
    all other plugins always get ``"cpu"``.
    """
    try:
        return get_ml_device_manager().get_torch_device(plugin_name)
    except Exception as exc:
        logger.debug("get_torch_device fallback to cpu: %s", exc)
        return "cpu"


def get_ort_providers(plugin_name: str = "") -> list[str]:
    """Return the ONNX Runtime provider list for *plugin_name*.

    Heavy plugins get GPU provider + CPU fallback; others get CPU-only.
    """
    try:
        return get_ml_device_manager().get_ort_providers(plugin_name)
    except Exception as exc:
        logger.debug("get_ort_providers fallback to CPU: %s", exc)
        return ["CPUExecutionProvider"]


def get_ort_providers_fp16(plugin_name: str = "") -> list[str]:
    """Return ORT providers with AMD ROCm fp16 hint for *plugin_name*.

    For eligible ONNX plugins on ROCm, returns ROCMExecutionProvider with
    memory-efficient options.  Falls back to standard providers on CPU/DirectML.
    """
    try:
        return get_ml_device_manager().get_ort_providers_fp16(plugin_name)
    except Exception as exc:
        logger.debug("get_ort_providers_fp16 fallback to CPU: %s", exc)
        return ["CPUExecutionProvider"]


def warmup_rocm_gpu() -> bool:
    """Initialise AMD ROCm runtime to eliminate cold-start latency before first inference.

    Call once from the application startup path (e.g. BatchProcessingThread.__init__).
    Safe no-op on CPU-only systems or Windows (DirectML warmup is automatic).
    """
    try:
        return get_ml_device_manager().warmup_rocm()
    except Exception as exc:
        logger.debug("warmup_rocm_gpu failed (non-critical): %s", exc)
        return False


# ---------------------------------------------------------------------------
# MLDeviceManager
# ---------------------------------------------------------------------------


class MLDeviceManager:
    """Detects GPU backend and manages device assignment for ML plugins.

    Responsibilities:
      1. Platform-aware GPU backend detection (ROCm / DirectML / None)
      2. Per-plugin device assignment (heavy plugins → GPU, rest → CPU)
      3. VRAM budget guard (analog to ml_memory_budget for system RAM)
      4. Transparent CPU fallback on any GPU failure or budget exhaustion
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: GPUBackend = GPUBackend.NONE
        self._torch_gpu_device: str = "cpu"  # "cuda" (ROCm) | "dml" (DirectML)
        self._ort_gpu_providers: list[str] = ["CPUExecutionProvider"]
        self._vram_allocated: dict[str, float] = {}  # plugin_name → GB
        self._vram_total_gb: float = 0.0
        self._vram_free_gb: float = 0.0
        self._gpu_available: bool = False
        self._gpu_name: str = "N/A"
        self._gpu_errors: dict[str, int] = {}  # plugin_name → error count
        self._gpu_disabled_plugins: set[str] = set()  # session-disabled plugins
        self._detect_backend()

    # ── Detection ────────────────────────────────────────────────────────

    def _detect_backend(self) -> None:
        """Auto-detect available GPU backend. Silently falls back to CPU on error."""
        try:
            if sys.platform == "win32":
                self._detect_directml()
            else:
                self._detect_rocm()
        except Exception as exc:
            logger.debug("MLDeviceManager: backend detection error (CPU fallback): %s", exc)

        if self._gpu_available:
            logger.info(
                "MLDeviceManager: GPU backend=%s device=%s VRAM=%.1f GB gpu=%s",
                self._backend.value,
                self._torch_gpu_device,
                self._vram_total_gb,
                self._gpu_name,
            )
        else:
            logger.info("MLDeviceManager: no GPU backend — CPU-only mode (ROCm/DirectML not found or not installed)")

    def _detect_rocm(self) -> None:
        """Detect ROCm via torch.cuda (ROCm reuses the CUDA device namespace)."""
        try:
            import torch  # type: ignore[import]

            if not torch.cuda.is_available():
                logger.debug("MLDeviceManager: torch.cuda unavailable — ROCm not present")
                return

            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024**3)

            self._backend = GPUBackend.ROCM
            self._torch_gpu_device = "cuda"  # ROCm exposes "cuda" device string
            # ROCMExecutionProvider falls back to CPUExecutionProvider automatically.
            self._ort_gpu_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            self._vram_total_gb = round(total_vram, 2)
            self._vram_free_gb = self._query_vram_free_rocm()
            self._gpu_available = True
            self._gpu_name = device_name

        except Exception as exc:
            logger.debug("MLDeviceManager: ROCm detection failed: %s", exc)

    def _detect_directml(self) -> None:
        """Detect DirectML on Windows via onnxruntime-directml."""
        try:
            import onnxruntime as _ort  # type: ignore[import]

            available = _ort.get_available_providers()
            if "DmlExecutionProvider" not in available:
                logger.debug(
                    "MLDeviceManager: DmlExecutionProvider not available "
                    "(onnxruntime-directml not installed?). Providers: %s",
                    available,
                )
                return

            self._backend = GPUBackend.DIRECTML
            self._ort_gpu_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self._gpu_available = True
            self._gpu_name = "DirectML (AMD/DX12)"
            self._vram_total_gb = self._query_vram_directml()
            self._vram_free_gb = self._vram_total_gb  # assume clear at startup

            # Optional: torch-directml enables PyTorch models on DirectML.
            # Without it, only ONNX models get DirectML acceleration.
            try:
                import torch_directml  # type: ignore[import]

                self._torch_gpu_device = "dml"
                logger.info("MLDeviceManager: torch-directml available — PyTorch models can use DML device")
            except ImportError:
                self._torch_gpu_device = "cpu"
                logger.info(
                    "MLDeviceManager: torch-directml not installed — ONNX models use DML, PyTorch models stay on CPU"
                )

        except Exception as exc:
            logger.debug("MLDeviceManager: DirectML detection failed: %s", exc)

    # ── VRAM query helpers ────────────────────────────────────────────────

    def _query_vram_free_rocm(self) -> float:
        """Query free VRAM in GB via torch.cuda.mem_get_info(device=0)."""
        try:
            import torch  # type: ignore[import]

            free_bytes, _ = torch.cuda.mem_get_info(0)
            return round(free_bytes / (1024**3), 2)
        except Exception:
            return self._vram_total_gb  # assume empty on query failure

    def _query_vram_directml(self) -> float:
        """Estimate VRAM for DirectML via WMIC (Windows) or conservative default."""
        try:
            import subprocess  # nosec B404 — WMIC read-only hardware query

            result = subprocess.run(  # nosec B603 B607
                ["wmic", "path", "Win32_VideoController", "get", "AdapterRAM", "/value"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                if "AdapterRAM" in line and "=" in line:
                    raw = line.split("=", 1)[1].strip()
                    if raw.isdigit():
                        return round(int(raw) / (1024**3), 2)
        except Exception:
            pass
        return 4.0  # conservative default when WMIC is unavailable

    # ── Public API ────────────────────────────────────────────────────────

    def get_gpu_backend(self) -> GPUBackend:
        """Return the detected GPU backend enum value."""
        return self._backend

    def is_gpu_available(self) -> bool:
        """True when GPU acceleration is available on this system."""
        return self._gpu_available

    def is_heavy_plugin(self, plugin_name: str) -> bool:
        """True when *plugin_name* is eligible for GPU dispatch."""
        return plugin_name in _HEAVY_ML_PLUGINS

    def get_torch_device(self, plugin_name: str = "") -> str:
        """Return PyTorch device string for *plugin_name*.

        Heavy plugins receive the GPU device string; all others receive ``"cpu"``.
        DirectML without torch-directml always returns ``"cpu"`` for PyTorch models.
        """
        if not self._gpu_available:
            return "cpu"
        if plugin_name and plugin_name not in _HEAVY_ML_PLUGINS:
            return "cpu"
        if plugin_name and plugin_name in self._gpu_disabled_plugins:
            return "cpu"
        # DirectML requires separate torch-directml package for PyTorch tensors.
        if self._backend == GPUBackend.DIRECTML and self._torch_gpu_device == "cpu":
            return "cpu"
        return self._torch_gpu_device

    def get_ort_providers(self, plugin_name: str = "") -> list[str]:
        """Return ONNX Runtime provider list for *plugin_name*.

        Heavy plugins get the GPU provider (with CPU fallback); others get CPU-only.
        Returns a defensive copy of the internal list.
        """
        if not self._gpu_available:
            return ["CPUExecutionProvider"]
        if plugin_name and plugin_name not in _HEAVY_ML_PLUGINS:
            return ["CPUExecutionProvider"]
        return list(self._ort_gpu_providers)

    def get_vram_free_gb(self) -> float:
        """Return current estimated free VRAM in GB (refreshed on ROCm via HW query)."""
        with self._lock:
            if self._backend == GPUBackend.ROCM:
                self._vram_free_gb = self._query_vram_free_rocm()
            return self._vram_free_gb

    def try_allocate_vram(self, plugin_name: str, size_gb: float) -> bool:
        """Attempt to reserve *size_gb* GB of VRAM for *plugin_name*.

        Returns True  → proceed with GPU inference.
        Returns False → use CPU fallback (VRAM budget would be exceeded).

        Idempotent: already-allocated plugins return True immediately.
        """
        if not self._gpu_available:
            return False

        with self._lock:
            if self._vram_allocated.get(plugin_name, 0.0) > 0.0:
                return True  # already allocated

            # Refresh VRAM free estimate before deciding.
            if self._backend == GPUBackend.ROCM:
                self._vram_free_gb = self._query_vram_free_rocm()

            already_used = sum(v for k, v in self._vram_allocated.items())
            max_usable = self._vram_total_gb * _VRAM_MAX_USAGE_RATIO

            if already_used + size_gb > max_usable:
                logger.warning(
                    "MLDeviceManager: VRAM budget exceeded for %s (%.2f GB) — used %.2f / %.2f GB → CPU fallback",
                    plugin_name,
                    size_gb,
                    already_used,
                    max_usable,
                )
                return False

            required_with_floor = size_gb + (_VRAM_MIN_FREE_MB / 1024.0)
            effective_free = self._vram_free_gb - already_used
            if effective_free < required_with_floor:
                logger.warning(
                    "MLDeviceManager: insufficient VRAM for %s — "
                    "need %.2f GB (incl. %.0f MB floor), have %.2f GB → CPU fallback",
                    plugin_name,
                    required_with_floor,
                    _VRAM_MIN_FREE_MB,
                    effective_free,
                )
                return False

            self._vram_allocated[plugin_name] = size_gb
            logger.info(
                "MLDeviceManager: VRAM allocated %s=%.2f GB (total used=%.2f / %.2f GB total)",
                plugin_name,
                size_gb,
                sum(self._vram_allocated.values()),
                self._vram_total_gb,
            )
            return True

    def release_vram(self, plugin_name: str) -> None:
        """Release VRAM budget reservation for *plugin_name* (call on model unload)."""
        with self._lock:
            freed = self._vram_allocated.pop(plugin_name, 0.0)
            if freed > 0.0:
                logger.debug("MLDeviceManager: VRAM released %s=%.2f GB", plugin_name, freed)

    def gpu_status_summary(self) -> dict[str, Any]:
        """Return a status snapshot dict for UI display and diagnostics."""
        with self._lock:
            return {
                "backend": self._backend.value,
                "gpu_available": self._gpu_available,
                "gpu_name": self._gpu_name,
                "vram_total_gb": self._vram_total_gb,
                "vram_free_gb": self._vram_free_gb,
                "vram_allocated_gb": sum(self._vram_allocated.values()),
                "allocated_plugins": dict(self._vram_allocated),
                "gpu_errors": dict(self._gpu_errors),
                "gpu_disabled_plugins": list(self._gpu_disabled_plugins),
            }

    def report_gpu_error(self, plugin_name: str, exc: Exception) -> None:
        """Record a runtime GPU inference failure and release VRAM budget.

        Called by plugins when GPU inference fails at runtime so the manager
        can track error frequency and proactively disable GPU for that plugin
        in future sessions (telemetry only — no pipeline impact).
        """
        with self._lock:
            self._gpu_errors[plugin_name] = self._gpu_errors.get(plugin_name, 0) + 1
            count = self._gpu_errors[plugin_name]
            logger.warning(
                "MLDeviceManager: GPU-Inferenz-Fehler #%d für %s (%s)",
                count,
                plugin_name,
                exc,
            )
            # Release VRAM budget so subsequent CPU load fits
            freed = self._vram_allocated.pop(plugin_name, 0.0)
            if freed > 0.0:
                self._vram_free_gb = min(self._vram_total_gb, self._vram_free_gb + freed)
            # After 3 consecutive failures, disable GPU for this plugin this session
            if count >= 3:
                self._gpu_disabled_plugins.add(plugin_name)
                logger.info(
                    "MLDeviceManager: GPU f\u00fcr %s nach %d Fehlern deaktiviert (Session)",
                    plugin_name,
                    count,
                )
    # ── AMD ROCm performance extensions ──────────────────────────────────

    def is_fp16_eligible(self, plugin_name: str) -> bool:
        """True when *plugin_name* supports fp16 inference on AMD ROCm.

        fp16 halves VRAM usage and typically doubles ONNX/PyTorch inference throughput
        on AMD GPUs with no perceptible quality difference for bounded audio signals.
        Returns False on CPU-only or DirectML (DML handles precision internally).
        """
        return (
            self._backend == GPUBackend.ROCM
            and self._gpu_available
            and plugin_name in _FP16_ELIGIBLE_PLUGINS
            and plugin_name not in self._gpu_disabled_plugins
        )

    def get_ort_providers_fp16(self, plugin_name: str = "") -> list[str]:
        """Return ORT providers with fp16 precision hint for AMD ROCm.

        Uses ROCMExecutionProvider with memory-efficient options when eligible;
        falls back to standard providers otherwise (always CPU-safe).
        """
        if not self.is_fp16_eligible(plugin_name):
            return self.get_ort_providers(plugin_name)
        return [
            (
                "ROCMExecutionProvider",
                {
                    "device_id": 0,
                    "gpu_mem_limit": int(self._vram_total_gb * 0.80 * 1024**3),
                    "arena_extend_strategy": "kSameAsRequested",
                    "enable_cuda_graph": 0,
                },
            ),
            "CPUExecutionProvider",
        ]

    def warmup_rocm(self) -> bool:
        """Initialise ROCm runtime with a minimal dummy inference to amortise cold-start.

        Running this once during application startup eliminates the ~500 ms–2 s first-
        inference latency caused by ROCm kernel compilation (HIP JIT). Returns True on
        success, False if warmup failed or ROCm is not active.

        Safe to call from any thread; internally synchronises via the instance lock.
        """
        if self._backend != GPUBackend.ROCM or not self._gpu_available:
            return False
        try:
            import torch  # type: ignore[import]

            with self._lock:
                _t = torch.zeros(_ROCM_WARMUP_TENSOR_LEN, dtype=torch.float32, device="cuda")
                _t_back = _t.cpu()
                del _t, _t_back
                torch.cuda.synchronize()
            torch.set_num_interop_threads(_ROCM_TORCH_THREADS)
            logger.info("MLDeviceManager: ROCm warmup complete — HIP runtime ready (%s)", self._gpu_name)
            return True
        except Exception as exc:
            logger.debug("MLDeviceManager: ROCm warmup failed (non-critical): %s", exc)
            return False

    def pin_tensor_rocm(self, array: "Any") -> "Any":
        """Return a pinned-memory copy of *array* for zero-copy CPU→GPU transfers.

        Only active on ROCm; returns the original array unchanged on all other backends.
        Pinned memory avoids an extra memcopy during `.to('cuda')` and reduces plugin
        inference latency by 10–25 % for large tensors (AudioSR, MERT, etc.).
        """
        if self._backend != GPUBackend.ROCM or not self._gpu_available:
            return array
        try:
            import numpy as np
            import torch  # type: ignore[import]

            if isinstance(array, np.ndarray):
                t = torch.from_numpy(array)
            elif isinstance(array, torch.Tensor):
                t = array
            else:
                return array
            if not t.is_pinned():
                t = t.pin_memory()
            return t
        except Exception as exc:
            logger.debug("pin_tensor_rocm failed (non-critical, using original): %s", exc)
            return array
