"""Tests for gpu_memory_manager module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock torch before importing
from unittest.mock import MagicMock, patch, PropertyMock
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.backends.mps.is_available.return_value = False
cpu_device = MagicMock()
cpu_device.type = "cpu"
mock_torch.device.return_value = cpu_device
sys.modules["torch"] = mock_torch

import pytest
import json
import time
import numpy as np

from performance.gpu_memory_manager import (
    MemoryStats,
    PerformanceProfile,
    GPUMemoryManager,
    ManagedGPUContext,
    demonstrate_memory_management,
)


def _make_stats(gpu_used=0.0, gpu_total=0.0, gpu_util=0.0, sys_used=8.0, sys_total=16.0):
    return MemoryStats(
        gpu_memory_total=gpu_total,
        gpu_memory_used=gpu_used,
        gpu_memory_free=gpu_total - gpu_used,
        gpu_utilization=gpu_util,
        system_memory_total=sys_total,
        system_memory_used=sys_used,
        system_memory_free=sys_total - sys_used,
        timestamp=time.time(),
    )


class TestMemoryStats:
    def test_dataclass(self):
        ms = _make_stats(gpu_used=4.0, gpu_total=8.0, gpu_util=50.0)
        assert ms.gpu_memory_total == 8.0
        assert ms.gpu_memory_free == 4.0


class TestPerformanceProfile:
    def test_dataclass(self):
        pp = PerformanceProfile(
            name="test", max_gpu_memory=4.0, max_system_memory=8.0,
            batch_size=32, model_complexity="medium", expected_duration=300,
        )
        assert pp.name == "test"
        assert pp.batch_size == 32


class TestGPUMemoryManager:
    def test_init_cpu(self):
        mgr = GPUMemoryManager()
        assert mgr.optimization_enabled is True
        assert len(mgr.performance_profiles) >= 4

    def test_detect_device_cpu(self):
        mgr = GPUMemoryManager.__new__(GPUMemoryManager)
        device = mgr._detect_device()
        assert device is not None

    def test_detect_device_cuda(self):
        mock_torch.cuda.is_available.return_value = True
        try:
            mgr = GPUMemoryManager.__new__(GPUMemoryManager)
            device = mgr._detect_device()
            assert device is not None
        finally:
            mock_torch.cuda.is_available.return_value = False

    def test_detect_device_mps(self):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        try:
            mgr = GPUMemoryManager.__new__(GPUMemoryManager)
            device = mgr._detect_device()
            assert device is not None
        finally:
            mock_torch.backends.mps.is_available.return_value = False

    def test_load_profiles_default(self):
        mgr = GPUMemoryManager()
        assert "electrode_optimization" in mgr.performance_profiles
        assert "physics_simulation" in mgr.performance_profiles

    def test_load_profiles_custom(self, tmp_path):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "custom": {
                "name": "custom", "max_gpu_memory": 1.0,
                "max_system_memory": 2.0, "batch_size": 4,
                "model_complexity": "low", "expected_duration": 30,
            }
        }))
        mgr = GPUMemoryManager(config_path=str(cfg))
        assert "custom" in mgr.performance_profiles

    def test_load_profiles_bad_file(self, tmp_path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("not json")
        mgr = GPUMemoryManager(config_path=str(cfg))
        assert len(mgr.performance_profiles) >= 4

    def test_load_profiles_nonexistent(self):
        mgr = GPUMemoryManager(config_path="/nonexistent.json")
        assert len(mgr.performance_profiles) >= 4

    def test_get_memory_stats_cpu(self):
        mgr = GPUMemoryManager()
        stats = mgr.get_memory_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.gpu_memory_total == 0.0
        assert stats.system_memory_total > 0
        assert len(mgr.memory_history) == 1

    def test_get_memory_stats_cuda(self):
        mock_torch.cuda.is_available.return_value = True
        try:
            mgr = GPUMemoryManager()
            cuda_dev = MagicMock()
            cuda_dev.type = "cuda"
            mgr.device = cuda_dev
            mock_props = MagicMock()
            mock_props.total_memory = 8 * 1024**3
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3
            with patch.object(mgr, '_get_gpu_utilization', return_value=50.0):
                stats = mgr.get_memory_stats()
            assert stats.gpu_memory_total == 8.0
            assert stats.gpu_memory_used == 2.0
        finally:
            mock_torch.cuda.is_available.return_value = False

    def test_get_memory_stats_mps(self):
        mgr = GPUMemoryManager()
        mps_dev = MagicMock()
        mps_dev.type = "mps"
        mgr.device = mps_dev
        mock_torch.mps.current_allocated_memory.return_value = 1 * 1024**3
        stats = mgr.get_memory_stats()
        assert stats.gpu_memory_total == 8.0
        assert stats.gpu_memory_used == 1.0

    def test_get_gpu_utilization_cpu(self):
        mgr = GPUMemoryManager()
        # CPU device falls to else branch (rocm-smi) which will fail
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            import subprocess as real_sub
            ms.TimeoutExpired = real_sub.TimeoutExpired
            ms.CalledProcessError = real_sub.CalledProcessError
            # rocm-smi returns non-zero
            ms.run.return_value = MagicMock(returncode=1)
            assert mgr._get_gpu_utilization() == 0.0

    def test_get_gpu_utilization_cuda_ok(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cuda")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            r = MagicMock(returncode=0, stdout="75\n")
            ms.run.return_value = r
            assert mgr._get_gpu_utilization() == 75.0

    def test_get_gpu_utilization_cuda_fail(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cuda")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            ms.run.return_value = MagicMock(returncode=1)
            assert mgr._get_gpu_utilization() == 0.0

    def test_get_gpu_utilization_timeout(self):
        import subprocess as real_sub
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cuda")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            ms.TimeoutExpired = real_sub.TimeoutExpired
            ms.CalledProcessError = real_sub.CalledProcessError
            ms.run.side_effect = real_sub.TimeoutExpired("nvidia-smi", 5)
            assert mgr._get_gpu_utilization() == 0.0

    def test_get_gpu_utilization_rocm(self):
        mgr = GPUMemoryManager()
        # Use non-cuda device to trigger else branch
        mgr.device = MagicMock(type="rocm")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            import subprocess as real_sub
            ms.TimeoutExpired = real_sub.TimeoutExpired
            ms.CalledProcessError = real_sub.CalledProcessError
            # Format: GPU% must be on the same line as the numeric value
            r = MagicMock(returncode=0, stdout="GPU% utilization: 60% Memory: 40%\n")
            ms.run.return_value = r
            assert mgr._get_gpu_utilization() == 60.0

    def test_get_gpu_utilization_rocm_no_match(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="rocm")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            import subprocess as real_sub
            ms.TimeoutExpired = real_sub.TimeoutExpired
            ms.CalledProcessError = real_sub.CalledProcessError
            ms.run.return_value = MagicMock(returncode=0, stdout="nothing\n")
            assert mgr._get_gpu_utilization() == 0.0

    def test_get_gpu_utilization_rocm_fail(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="rocm")
        with patch("performance.gpu_memory_manager.subprocess") as ms:
            import subprocess as real_sub
            ms.TimeoutExpired = real_sub.TimeoutExpired
            ms.CalledProcessError = real_sub.CalledProcessError
            ms.run.return_value = MagicMock(returncode=1)
            assert mgr._get_gpu_utilization() == 0.0

    def test_check_memory_health_healthy(self):
        mgr = GPUMemoryManager()
        health = mgr.check_memory_health()
        assert health["status"] in ["healthy", "warning"]

    def test_check_memory_health_gpu_critical(self):
        mgr = GPUMemoryManager()
        with patch.object(mgr, 'get_memory_stats', return_value=_make_stats(
            gpu_used=7.5, gpu_total=8.0, sys_used=8.0, sys_total=16.0)):
            health = mgr.check_memory_health()
            assert health["status"] == "critical"

    def test_check_memory_health_gpu_warning(self):
        mgr = GPUMemoryManager()
        with patch.object(mgr, 'get_memory_stats', return_value=_make_stats(
            gpu_used=6.8, gpu_total=8.0, sys_used=8.0, sys_total=16.0)):
            health = mgr.check_memory_health()
            assert health["status"] == "warning"

    def test_check_memory_health_system_warning(self):
        mgr = GPUMemoryManager()
        with patch.object(mgr, 'get_memory_stats', return_value=_make_stats(
            gpu_total=0.0, gpu_used=0.0, sys_used=14.0, sys_total=16.0)):
            health = mgr.check_memory_health()
            assert health["status"] == "warning"

    def test_optimize_for_profile(self):
        mgr = GPUMemoryManager()
        result = mgr.optimize_for_profile("electrode_optimization")
        assert result["profile"] == "electrode_optimization"

    def test_optimize_for_profile_unknown(self):
        mgr = GPUMemoryManager()
        with pytest.raises(ValueError, match="Unknown profile"):
            mgr.optimize_for_profile("nonexistent")

    def test_optimize_for_profile_cuda_high(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cuda")
        call_count = [0]
        def mock_stats():
            call_count[0] += 1
            used = 3.0 if call_count[0] == 1 else 1.0
            s = _make_stats(gpu_used=used, gpu_total=4.0)
            mgr.memory_history.append(s)
            return s
        with patch.object(mgr, 'get_memory_stats', side_effect=mock_stats):
            with patch.object(mgr, 'clear_gpu_cache'):
                result = mgr.optimize_for_profile("electrode_optimization")
                assert result["memory_freed"] >= 0

    def test_optimize_for_profile_low_available(self):
        mgr = GPUMemoryManager()
        with patch.object(mgr, 'get_memory_stats', return_value=_make_stats(
            gpu_used=3.5, gpu_total=4.0)):
            result = mgr.optimize_for_profile("electrode_optimization")
            assert "batch_size" in result["optimized_settings"]

    def test_clear_gpu_cache_cpu(self):
        mgr = GPUMemoryManager()
        mgr.clear_gpu_cache()

    def test_clear_gpu_cache_cuda(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="cuda")
        mgr.clear_gpu_cache()
        mock_torch.cuda.empty_cache.assert_called()

    def test_clear_gpu_cache_mps(self):
        mgr = GPUMemoryManager()
        mgr.device = MagicMock(type="mps")
        mgr.clear_gpu_cache()

    def test_monitor_allocation(self):
        mgr = GPUMemoryManager()
        mgr.monitor_allocation("tensor", 100.0)
        assert len(mgr.allocation_events) == 1

    def test_get_performance_report_empty(self):
        mgr = GPUMemoryManager()
        report = mgr.get_performance_report()
        assert "current_status" in report
        assert "device_info" in report

    def test_get_performance_report_with_history(self):
        mgr = GPUMemoryManager()
        for i in range(5):
            mgr.memory_history.append(_make_stats(gpu_used=float(i), gpu_total=8.0, gpu_util=10.0 * i))
        report = mgr.get_performance_report()
        assert report["memory_trends"]["gpu_memory_peak"] == 4.0

    def test_suggestions_few_data(self):
        mgr = GPUMemoryManager()
        s = mgr._generate_optimization_suggestions()
        assert "more performance data" in s[0]

    def test_suggestions_spike(self):
        mgr = GPUMemoryManager()
        for _ in range(9):
            mgr.memory_history.append(_make_stats(gpu_used=1.0, gpu_total=8.0, gpu_util=10.0))
        mgr.memory_history.append(_make_stats(gpu_used=5.0, gpu_total=8.0, gpu_util=10.0))
        s = mgr._generate_optimization_suggestions()
        assert any("batch" in x.lower() for x in s)

    def test_suggestions_efficient(self):
        mgr = GPUMemoryManager()
        for _ in range(10):
            mgr.memory_history.append(_make_stats(gpu_used=0.5, gpu_total=8.0, gpu_util=10.0))
        s = mgr._generate_optimization_suggestions()
        assert any("efficient" in x.lower() for x in s)

    def test_suggestions_high_usage(self):
        mgr = GPUMemoryManager()
        for _ in range(10):
            mgr.memory_history.append(_make_stats(gpu_used=5.0, gpu_total=8.0, gpu_util=10.0))
        s = mgr._generate_optimization_suggestions()
        assert any("optimization" in x.lower() or "distributed" in x.lower() for x in s)

    def test_suggestions_low_util(self):
        mgr = GPUMemoryManager()
        for _ in range(10):
            mgr.memory_history.append(_make_stats(gpu_used=1.0, gpu_total=8.0, gpu_util=5.0))
        s = mgr._generate_optimization_suggestions()
        assert any("low" in x.lower() for x in s)

    def test_suggestions_high_util(self):
        mgr = GPUMemoryManager()
        for _ in range(10):
            mgr.memory_history.append(_make_stats(gpu_used=1.0, gpu_total=8.0, gpu_util=95.0))
        s = mgr._generate_optimization_suggestions()
        assert any("highly" in x.lower() for x in s)

    def test_save_performance_log(self, tmp_path):
        mgr = GPUMemoryManager()
        mgr.get_memory_stats()
        fp = str(tmp_path / "log.json")
        mgr.save_performance_log(fp)
        assert os.path.exists(fp)
        with open(fp) as f:
            data = json.load(f)
        assert "device" in data


class TestManagedGPUContext:
    def test_context(self):
        mgr = GPUMemoryManager()
        with ManagedGPUContext(mgr, "electrode_optimization") as m:
            assert m is mgr

    def test_context_memory_increase(self):
        mgr = GPUMemoryManager()
        call_count = [0]
        def mock_stats():
            call_count[0] += 1
            used = 1.0 if call_count[0] <= 2 else 2.0
            s = _make_stats(gpu_used=used, gpu_total=8.0)
            mgr.memory_history.append(s)
            return s
        with patch.object(mgr, 'get_memory_stats', side_effect=mock_stats):
            with patch.object(mgr, 'clear_gpu_cache') as mc:
                ctx = ManagedGPUContext(mgr, "electrode_optimization")
                ctx.__enter__()
                ctx.__exit__(None, None, None)

    def test_context_no_memory_increase(self):
        mgr = GPUMemoryManager()
        def mock_stats():
            s = _make_stats(gpu_used=1.0, gpu_total=8.0)
            mgr.memory_history.append(s)
            return s
        with patch.object(mgr, 'get_memory_stats', side_effect=mock_stats):
            with patch.object(mgr, 'clear_gpu_cache') as mc:
                ctx = ManagedGPUContext(mgr, "electrode_optimization")
                ctx.__enter__()
                ctx.__exit__(None, None, None)
                mc.assert_not_called()


class TestDemonstrate:
    def test_demonstrate(self):
        mgr = demonstrate_memory_management()
        assert isinstance(mgr, GPUMemoryManager)
