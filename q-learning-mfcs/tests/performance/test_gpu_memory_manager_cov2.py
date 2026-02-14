"""Coverage boost tests for gpu_memory_manager.py."""
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Mock torch before import
_mock_torch = MagicMock()
_mock_torch.device.return_value = MagicMock(type="cpu")
_mock_torch.cuda.is_available.return_value = False
_mock_torch.backends.mps.is_available.return_value = False
sys.modules.setdefault("torch", _mock_torch)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from performance.gpu_memory_manager import (
    GPUMemoryManager,
    ManagedGPUContext,
    MemoryStats,
    PerformanceProfile,
)


@pytest.fixture
def manager():
    with patch("performance.gpu_memory_manager.torch") as mock_t:
        mock_t.cuda.is_available.return_value = False
        mock_t.backends.mps.is_available.return_value = False
        mock_t.device.return_value = MagicMock(type="cpu")
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cpu")
        yield mgr


@pytest.mark.coverage_extra
class TestGPUMemoryManager:
    def test_init_cpu(self, manager):
        assert manager.device.type == "cpu"
        assert len(manager.performance_profiles) >= 4

    def test_detect_device_cpu(self):
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.cuda.is_available.return_value = False
            mt.backends.mps.is_available.return_value = False
            mt.device.return_value = MagicMock(type="cpu")
            mgr = GPUMemoryManager()

    def test_detect_device_cuda(self):
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.cuda.is_available.return_value = True
            mt.device.return_value = MagicMock(type="cuda")
            mgr = GPUMemoryManager()

    def test_detect_device_mps(self):
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.cuda.is_available.return_value = False
            mt.backends.mps.is_available.return_value = True
            mt.device.return_value = MagicMock(type="mps")
            mgr = GPUMemoryManager()

    def test_load_custom_profiles(self, tmp_path):
        config = {
            "custom_profile": {
                "name": "custom_profile",
                "max_gpu_memory": 1.0,
                "max_system_memory": 2.0,
                "batch_size": 4,
                "model_complexity": "low",
                "expected_duration": 30,
            }
        }
        cfg_file = tmp_path / "profiles.json"
        cfg_file.write_text(json.dumps(config))
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.cuda.is_available.return_value = False
            mt.backends.mps.is_available.return_value = False
            mt.device.return_value = MagicMock(type="cpu")
            mgr = GPUMemoryManager(config_path=str(cfg_file))
            assert "custom_profile" in mgr.performance_profiles

    def test_load_bad_config(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("{invalid json")
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.cuda.is_available.return_value = False
            mt.backends.mps.is_available.return_value = False
            mt.device.return_value = MagicMock(type="cpu")
            mgr = GPUMemoryManager(config_path=str(cfg_file))
            assert len(mgr.performance_profiles) >= 4

    def test_get_memory_stats_cpu(self, manager):
        stats = manager.get_memory_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.gpu_memory_total == 0.0
        assert len(manager.memory_history) == 1

    def test_get_memory_stats_mps(self, manager):
        manager.device = MagicMock(type="mps")
        with patch("performance.gpu_memory_manager.torch") as mt:
            mt.mps.current_allocated_memory.return_value = 1024**3
            stats = manager.get_memory_stats()
            assert stats.gpu_memory_total == 8.0

    def test_get_gpu_utilization_cpu(self, manager):
        result = manager._get_gpu_utilization()
        assert result == 0.0

    def test_get_gpu_utilization_cuda(self, manager):
        manager.device = MagicMock(type="cuda")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="75\n")
            result = manager._get_gpu_utilization()
            assert result == 75.0

    def test_get_gpu_utilization_timeout(self, manager):
        import subprocess
        manager.device = MagicMock(type="cuda")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = manager._get_gpu_utilization()
            assert result == 0.0

    def test_check_memory_health_healthy(self, manager):
        health = manager.check_memory_health()
        assert health["status"] == "healthy"

    def test_optimize_for_profile(self, manager):
        result = manager.optimize_for_profile("electrode_optimization")
        assert result["profile"] == "electrode_optimization"

    def test_optimize_unknown_profile(self, manager):
        with pytest.raises(ValueError, match="Unknown profile"):
            manager.optimize_for_profile("nonexistent")

    def test_clear_gpu_cache_cpu(self, manager):
        manager.clear_gpu_cache()

    def test_clear_gpu_cache_cuda(self, manager):
        manager.device = MagicMock(type="cuda")
        with patch("performance.gpu_memory_manager.torch") as mt:
            manager.clear_gpu_cache()
            mt.cuda.empty_cache.assert_called_once()

    def test_monitor_allocation(self, manager):
        manager.monitor_allocation("test_tensor", 100.0)
        assert len(manager.allocation_events) == 1
        assert manager.allocation_events[0]["name"] == "test_tensor"

    def test_get_performance_report_empty(self, manager):
        report = manager.get_performance_report()
        assert "current_status" in report
        assert "device_info" in report
        assert "memory_trends" in report

    def test_generate_suggestions_few(self, manager):
        suggestions = manager._generate_optimization_suggestions()
        assert len(suggestions) == 1
        assert "Collect more" in suggestions[0]

    def test_generate_suggestions_many(self, manager):
        for _ in range(10):
            manager.memory_history.append(MemoryStats(
                gpu_memory_total=8.0, gpu_memory_used=0.5,
                gpu_memory_free=7.5, gpu_utilization=10.0,
                system_memory_total=16.0, system_memory_used=8.0,
                system_memory_free=8.0, timestamp=time.time(),
            ))
        suggestions = manager._generate_optimization_suggestions()
        assert len(suggestions) > 0

    def test_save_performance_log(self, tmp_path, manager):
        manager.get_memory_stats()
        filepath = str(tmp_path / "perf.json")
        manager.save_performance_log(filepath)
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert "device" in data


@pytest.mark.coverage_extra
class TestManagedGPUContext:
    def test_context_manager(self, manager):
        ctx = ManagedGPUContext(manager, "gsm_analysis")
        with ctx as mgr:
            assert mgr is manager

    def test_context_memory_increase(self, manager):
        ctx = ManagedGPUContext(manager, "gsm_analysis")
        ctx.__enter__()
        ctx.initial_stats = MemoryStats(
            gpu_memory_total=8.0, gpu_memory_used=1.0,
            gpu_memory_free=7.0, gpu_utilization=50.0,
            system_memory_total=16.0, system_memory_used=8.0,
            system_memory_free=8.0, timestamp=time.time(),
        )
        manager.get_memory_stats = MagicMock(return_value=MemoryStats(
            gpu_memory_total=8.0, gpu_memory_used=1.5,
            gpu_memory_free=6.5, gpu_utilization=50.0,
            system_memory_total=16.0, system_memory_used=8.0,
            system_memory_free=8.0, timestamp=time.time(),
        ))
        manager.clear_gpu_cache = MagicMock()
        ctx.__exit__(None, None, None)
        manager.clear_gpu_cache.assert_called_once()


@pytest.mark.coverage_extra
class TestPerformanceProfile:
    def test_profile_creation(self):
        p = PerformanceProfile(
            name="test", max_gpu_memory=1.0, max_system_memory=2.0,
            batch_size=8, model_complexity="low", expected_duration=60,
        )
        assert p.name == "test"
        assert p.batch_size == 8
