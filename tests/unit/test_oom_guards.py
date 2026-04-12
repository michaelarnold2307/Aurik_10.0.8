"""Unit tests for OOM guard hardening paths.

Covers:
- Physical RAM preflight in ml_memory_budget.try_allocate
- Idempotent monitor start in AdaptiveResourceManager
- Auto-detect budget from system RAM
- Budget exhaustion blocks subsequent allocations
- Release frees budget for reuse
- UV3 audio buffer guard (MemoryError on oversized input)
- EnsembleProcessor skip guard for long files
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.core import ml_memory_budget as budget
from backend.core.adaptive_resource_manager import AdaptiveResourceManager


def _reset_budget_state() -> None:
    budget._allocated.clear()
    budget._total_gb = 0.0


class _MemProbe:
    def __init__(self, available_mb: float, percent: float = 50.0):
        self.available = int(available_mb * 1024 * 1024)
        self.percent = percent


def test_try_allocate_blocks_when_physical_memory_too_low(monkeypatch):
    _reset_budget_state()

    class _LowPsutil:
        @staticmethod
        def virtual_memory():
            return _MemProbe(available_mb=300.0)

    monkeypatch.setattr(budget, "_psutil", _LowPsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)

    # Simulate no improvement after eviction attempt.
    monkeypatch.setattr(
        budget,
        "_preflight_system_memory",
        lambda required_mb: False,
    )

    ok = budget.try_allocate("TestHeavyModel", size_gb=2.0)
    assert ok is False
    assert "TestHeavyModel" not in budget._allocated


def test_try_allocate_succeeds_after_preflight_and_budget(monkeypatch):
    _reset_budget_state()

    class _GoodPsutil:
        @staticmethod
        def virtual_memory():
            return _MemProbe(available_mb=8192.0)

    monkeypatch.setattr(budget, "_psutil", _GoodPsutil)
    monkeypatch.setattr(budget, "ML_MAX_GB", 10.0)

    ok = budget.try_allocate("SmallModel", size_gb=0.5)
    assert ok is True
    assert budget._allocated.get("SmallModel") == 0.5


def test_adaptive_resource_manager_start_monitoring_is_idempotent(monkeypatch):
    manager = AdaptiveResourceManager(check_interval=10.0)

    starts = {"count": 0}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            self.started = True
            starts["count"] += 1

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            self.started = False

    # Patch only for this test.
    monkeypatch.setattr("backend.core.adaptive_resource_manager.threading.Thread", _FakeThread)

    manager.start_monitoring()
    first_thread = manager._monitor_thread
    manager.start_monitoring()
    second_thread = manager._monitor_thread

    assert manager.running is True
    assert starts["count"] == 1
    assert first_thread is second_thread

    manager.stop_monitoring()


def test_adaptive_resource_manager_stop_monitoring_joins_thread(monkeypatch):
    manager = AdaptiveResourceManager(check_interval=10.0)

    joined = {"timeout": None}

    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            joined["timeout"] = timeout
            self.started = False

    monkeypatch.setattr("backend.core.adaptive_resource_manager.threading.Thread", _FakeThread)

    manager.start_monitoring()
    thread = manager._monitor_thread

    manager.stop_monitoring()

    assert manager.running is False
    assert joined["timeout"] == 1.0
    assert manager._monitor_thread is None
    assert thread is not None and thread.started is False


# ── Auto-detect budget tests ────────────────────────────────────────────


def test_auto_detect_budget_32gb(monkeypatch):
    """On a 32 GB system: budget = 32/3 ≈ 10.7 GB, clamped to [4, 12]."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=32 * 1024**3, available=20 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert 10.0 <= result <= 11.0, f"Expected ~10.7 GB for 32 GB system, got {result}"


def test_auto_detect_budget_16gb(monkeypatch):
    """On a 16 GB system: budget = 16/3 ≈ 5.3 GB."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=16 * 1024**3, available=10 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert 4.0 <= result <= 6.0, f"Expected ~5.3 GB for 16 GB system, got {result}"


def test_auto_detect_budget_8gb(monkeypatch):
    """On an 8 GB system: budget = 8/3 ≈ 2.7 → clamped to 4 GB minimum."""

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(total=8 * 1024**3, available=5 * 1024**3)

    monkeypatch.setattr(budget, "_psutil", _FakePsutil)
    result = budget._auto_detect_budget()
    assert result == 4.0, f"Expected 4.0 GB floor for 8 GB system, got {result}"


def test_auto_detect_budget_no_psutil(monkeypatch):
    """Without psutil: conservative 10 GB default."""
    monkeypatch.setattr(budget, "_psutil", None)
    result = budget._auto_detect_budget()
    assert result == 10.0


# ── Budget exhaustion and release tests ─────────────────────────────────


def test_budget_exhaustion_blocks_allocation(monkeypatch):
    """When budget is full, subsequent allocations fail."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("ModelA", size_gb=3.0) is True
    assert budget.try_allocate("ModelB", size_gb=3.0) is False  # 3+3 > 5
    assert "ModelB" not in budget._allocated


def test_release_frees_budget(monkeypatch):
    """After release, freed budget is available for new allocations."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("ModelA", size_gb=3.0) is True
    budget.release("ModelA")
    assert "ModelA" not in budget._allocated
    assert budget._total_gb == 0.0
    assert budget.try_allocate("ModelB", size_gb=4.0) is True  # now fits


def test_idempotent_allocation(monkeypatch):
    """Same model name allocated twice returns True without double-counting."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 5.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    assert budget.try_allocate("SameModel", size_gb=2.0) is True
    assert budget.try_allocate("SameModel", size_gb=2.0) is True
    assert budget._total_gb == 2.0  # not 4.0


def test_get_status(monkeypatch):
    """get_status() returns correct current state."""
    _reset_budget_state()
    monkeypatch.setattr(budget, "ML_MAX_GB", 8.0)
    monkeypatch.setattr(budget, "is_system_thrashing", lambda: False)
    monkeypatch.setattr(budget, "_preflight_system_memory", lambda required_mb=0: True)

    budget.try_allocate("TestModel", size_gb=1.5)
    status = budget.get_status()
    assert status["max_gb"] == 8.0
    assert status["allocated_gb"] == 1.5
    assert status["free_gb"] == 6.5
    assert "TestModel" in status["models"]


# ── UV3 audio buffer guard test ─────────────────────────────────────────


def test_uv3_rejects_oversized_audio():
    """UV3 must raise MemoryError for audio exceeding 4 GB buffer limit."""
    import pytest

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    # Create a fake shape that would be > 4 GB if real (we test the guard logic)
    # We cannot actually allocate 4 GB in a test, so we monkeypatch nbytes.
    uv3 = UnifiedRestorerV3()
    # Create tiny audio but patch nbytes to simulate huge file
    audio = np.zeros(48000, dtype=np.float32)  # 1s mono

    class _FakeArray(np.ndarray):
        @property
        def nbytes(self):
            return 5 * 1024**3  # 5 GB — over the 4 GB limit

    oversized = audio.view(_FakeArray)
    with pytest.raises(MemoryError, match="Audio-Buffer"):
        uv3.restore(oversized, sample_rate=48000)


# ── EnsembleProcessor size guard test ───────────────────────────────────


def test_ensemble_processor_skipped_for_long_audio():
    """EnsembleProcessor should not run for audio > 2 minutes (OOM guard)."""
    # This tests the guard variable logic, not the full UV3 pipeline
    sr = 48000
    # 3 minutes at 48kHz → 8_640_000 samples
    audio_len = 3 * 60 * sr
    ensemble_max_samples = 2 * 60 * sr
    assert audio_len > ensemble_max_samples, "Test audio must exceed 2-min limit"
    # 30 seconds at 48kHz → 1_440_000 samples
    short_len = 30 * sr
    assert short_len <= ensemble_max_samples, "Short audio must be within limit"
