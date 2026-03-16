"""Unit tests for OOM guard hardening paths.

Covers:
- Physical RAM preflight in ml_memory_budget.try_allocate
- Idempotent monitor start in AdaptiveResourceManager
"""

from __future__ import annotations

from types import SimpleNamespace

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

        def start(self):
            starts["count"] += 1

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
