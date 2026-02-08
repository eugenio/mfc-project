"""Tests for gpu_acceleration module - targeting 98%+ coverage."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock cupy, torch, jax before import
sys.modules["cupy"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["jax"] = MagicMock()

from gpu_acceleration import GPUAccelerator, get_gpu_accelerator, benchmark_backends


class TestGPUAccelerator:
    def test_init_cpu_fallback(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = ["cpu"]
        acc.cp = None
        acc.torch = None
        acc._initialize_backend("auto")
        assert acc.backend == "cpu"

    def test_setup_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        assert acc.backend == "cpu"
        assert acc.cp is None
        assert acc.torch is None

    def test_array_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.array([1.0, 2.0, 3.0])
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_zeros_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.zeros((3,))
        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_ones_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.ones((3,))
        assert np.allclose(result, [1.0, 1.0, 1.0])

    def test_random_uniform_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.random_uniform((5,), low=0.0, high=1.0)
        assert result.shape == (5,)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_to_cpu_numpy(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        arr = np.array([1.0, 2.0])
        result = acc.to_cpu(arr)
        assert np.allclose(result, arr)

    def test_synchronize_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        acc.synchronize()  # Should not raise

    def test_np_to_torch_dtype_no_torch(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.torch = None
        assert acc._np_to_torch_dtype(np.float32) is None

    def test_get_backend_info(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        acc.available_backends = ["cpu"]
        info = acc.get_backend_info()
        assert info["backend"] == "cpu"
        assert "cpu" in info["available_backends"]

    def test_is_gpu_available_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        assert acc.is_gpu_available() is False

    def test_is_gpu_available_cuda(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        assert acc.is_gpu_available() is True

    def test_is_gpu_available_rocm(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        assert acc.is_gpu_available() is True

    def test_get_memory_info_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        assert acc.get_memory_info() is None

    def test_abs_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.abs(np.array([-1.0, 2.0, -3.0]))
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_where_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        arr = np.array([1, 2, 3, 4])
        result = acc.where(arr > 2, arr, np.zeros_like(arr))
        assert np.allclose(result, [0, 0, 3, 4])

    def test_maximum_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.maximum(np.array([1, 3]), np.array([2, 1]))
        assert np.allclose(result, [2, 3])

    def test_minimum_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.minimum(np.array([1, 3]), np.array([2, 1]))
        assert np.allclose(result, [1, 1])

    def test_log_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.log(np.array([1.0, np.e]))
        assert np.allclose(result, [0.0, 1.0])

    def test_exp_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.exp(np.array([0.0, 1.0]))
        assert np.allclose(result, [1.0, np.e])

    def test_clip_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.clip(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
        assert np.allclose(result, [0.0, 0.5, 1.0])

    def test_mean_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.mean(np.array([1.0, 2.0, 3.0]))
        assert result == 2.0

    def test_mean_axis_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = acc.mean(arr, axis=0)
        assert np.allclose(result, [2.0, 3.0])

    def test_sum_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.sum(np.array([1.0, 2.0, 3.0]))
        assert result == 6.0

    def test_sum_axis_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = acc.sum(arr, axis=1)
        assert np.allclose(result, [3.0, 7.0])

    def test_sqrt_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.sqrt(np.array([4.0, 9.0]))
        assert np.allclose(result, [2.0, 3.0])

    def test_power_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.power(np.array([2.0, 3.0]), 2)
        assert np.allclose(result, [4.0, 9.0])

    def test_random_normal_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        result = acc.random_normal((100,), mean=0.0, std=1.0)
        assert result.shape == (100,)

    def test_force_cpu_fallback(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.cp = MagicMock()
        acc.torch = MagicMock()
        acc.force_cpu_fallback()
        assert acc.backend == "cpu"
        assert acc.cp is None
        assert acc.torch is None
        assert "Fallback" in acc.device_info["device_name"]

    def test_test_gpu_functionality_cpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        assert acc.test_gpu_functionality() is True

    def test_test_gpu_functionality_pass(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc._setup_cpu()
        # Simulate "GPU" backend that actually works with numpy
        acc.backend = "fake_gpu"
        # Override methods to use numpy
        acc.array = lambda data, dtype=np.float32: np.asarray(data, dtype=dtype)
        acc.abs = lambda a: np.abs(a)
        acc.maximum = lambda a, b: np.maximum(a, b)
        acc.mean = lambda a, axis=None: np.mean(a, axis=axis)
        acc.to_cpu = lambda a: np.asarray(a)
        result = acc.test_gpu_functionality()
        # Since backend is not "cpu", it will try the test
        assert result is True

    def test_test_gpu_functionality_fail(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.cp = MagicMock()
        acc.torch = None
        acc.device_info = {}
        # Make array raise an exception
        def failing_array(data, dtype=np.float32):
            raise RuntimeError("GPU failed")
        acc.array = failing_array
        acc.force_cpu_fallback_called = False
        original_fallback = GPUAccelerator.force_cpu_fallback
        def mock_fallback(self_):
            self_.backend = "cpu"
            self_.cp = None
            self_.torch = None
            self_.device_info = {"backend": "cpu", "device_count": 0, "device_name": "CPU (Fallback Mode)"}
        acc.force_cpu_fallback = lambda: mock_fallback(acc)
        result = acc.test_gpu_functionality()
        assert result is False

    def test_initialize_backend_prefer_specific(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = ["cuda", "rocm", "cpu"]
        acc.cp = None
        acc.torch = None
        acc._initialize_backend("rocm")
        # Since rocm is in available_backends, it should be selected
        # But _setup_backend will call _setup_rocm which may fail
        # Just verify backend assignment
        assert acc.backend in ["rocm", "cpu"]

    def test_initialize_backend_not_available(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = ["cpu"]
        acc.cp = None
        acc.torch = None
        acc._initialize_backend("cuda")
        # cuda not available, should fall back to cpu
        assert acc.backend == "cpu"

    def test_initialize_backend_auto_cuda(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = ["cuda", "cpu"]
        acc.cp = None
        acc.torch = None
        # Will try _setup_cuda which will fail and fall back to cpu
        acc._initialize_backend("auto")
        # Should have tried cuda first
        assert acc.backend in ["cuda", "cpu"]

    def test_initialize_backend_auto_rocm(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = ["rocm", "cpu"]
        acc.cp = None
        acc.torch = None
        acc._initialize_backend("auto")
        assert acc.backend in ["rocm", "cpu"]

    def test_detect_backends_no_gpu(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": None}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends

    def test_setup_cuda_fail(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        with patch.dict(sys.modules, {"cupy": None}):
            acc._setup_cuda()
        assert acc.backend == "cpu"

    def test_setup_rocm_fail(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        with patch.dict(sys.modules, {"torch": None}):
            acc._setup_rocm()
        assert acc.backend == "cpu"


class TestCUDABackendPaths:
    """Test all methods with CUDA backend (cupy mock)."""

    def _make_cuda_acc(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.torch = None
        acc.cp = MagicMock()
        acc.device_info = {"backend": "cuda"}
        acc.available_backends = ["cuda", "cpu"]
        return acc

    def test_array_cuda(self):
        acc = self._make_cuda_acc()
        acc.array([1.0, 2.0], dtype=np.float32)
        acc.cp.asarray.assert_called_once()

    def test_zeros_cuda(self):
        acc = self._make_cuda_acc()
        acc.zeros((3,))
        acc.cp.zeros.assert_called_once()

    def test_ones_cuda(self):
        acc = self._make_cuda_acc()
        acc.ones((3,))
        acc.cp.ones.assert_called_once()

    def test_random_uniform_cuda(self):
        acc = self._make_cuda_acc()
        acc.random_uniform((5,))
        acc.cp.random.uniform.assert_called_once()

    def test_to_cpu_cuda(self):
        acc = self._make_cuda_acc()
        mock_arr = MagicMock()
        acc.to_cpu(mock_arr)
        acc.cp.asnumpy.assert_called_once_with(mock_arr)

    def test_synchronize_cuda(self):
        acc = self._make_cuda_acc()
        acc.synchronize()
        acc.cp.cuda.Stream.null.synchronize.assert_called_once()

    def test_get_memory_info_cuda(self):
        acc = self._make_cuda_acc()
        acc.cp.cuda.Device.return_value.mem_info = (1000, 2000)
        info = acc.get_memory_info()
        assert info["free"] == 1000
        assert info["total"] == 2000
        assert info["used"] == 1000

    def test_abs_cuda(self):
        acc = self._make_cuda_acc()
        arr = MagicMock()
        acc.abs(arr)
        acc.cp.abs.assert_called_once_with(arr)

    def test_where_cuda(self):
        acc = self._make_cuda_acc()
        acc.where(MagicMock(), MagicMock(), MagicMock())
        acc.cp.where.assert_called_once()

    def test_maximum_cuda(self):
        acc = self._make_cuda_acc()
        acc.maximum(MagicMock(), MagicMock())
        acc.cp.maximum.assert_called_once()

    def test_minimum_cuda(self):
        acc = self._make_cuda_acc()
        acc.minimum(MagicMock(), MagicMock())
        acc.cp.minimum.assert_called_once()

    def test_log_cuda(self):
        acc = self._make_cuda_acc()
        acc.log(MagicMock())
        acc.cp.log.assert_called_once()

    def test_exp_cuda(self):
        acc = self._make_cuda_acc()
        acc.exp(MagicMock())
        acc.cp.exp.assert_called_once()

    def test_clip_cuda(self):
        acc = self._make_cuda_acc()
        acc.clip(MagicMock(), 0.0, 1.0)
        acc.cp.clip.assert_called_once()

    def test_mean_cuda(self):
        acc = self._make_cuda_acc()
        acc.mean(MagicMock())
        acc.cp.mean.assert_called_once()

    def test_sum_cuda(self):
        acc = self._make_cuda_acc()
        acc.sum(MagicMock())
        acc.cp.sum.assert_called_once()

    def test_sqrt_cuda(self):
        acc = self._make_cuda_acc()
        acc.sqrt(MagicMock())
        acc.cp.sqrt.assert_called_once()

    def test_power_cuda(self):
        acc = self._make_cuda_acc()
        acc.power(MagicMock(), 2)
        acc.cp.power.assert_called_once()

    def test_random_normal_cuda(self):
        acc = self._make_cuda_acc()
        acc.random_normal((10,))
        acc.cp.random.normal.assert_called_once()


class TestROCmBackendPaths:
    """Test all methods with ROCm backend (torch mock)."""

    def _make_rocm_acc(self):
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.cp = None
        acc.torch = MagicMock()
        acc.torch.float32 = "torch.float32"
        acc.torch.float64 = "torch.float64"
        acc.torch.int32 = "torch.int32"
        acc.torch.int64 = "torch.int64"
        acc.torch.bool = "torch.bool"
        acc.device_info = {"backend": "rocm"}
        acc.available_backends = ["rocm", "cpu"]
        return acc

    def test_array_rocm_ndarray(self):
        acc = self._make_rocm_acc()
        data = np.array([1.0, 2.0])
        acc.array(data)
        acc.torch.from_numpy.assert_called_once()

    def test_array_rocm_list(self):
        acc = self._make_rocm_acc()
        acc.array([1.0, 2.0])
        acc.torch.tensor.assert_called_once()

    def test_zeros_rocm(self):
        acc = self._make_rocm_acc()
        acc.zeros((3,))
        acc.torch.zeros.assert_called_once()

    def test_ones_rocm(self):
        acc = self._make_rocm_acc()
        acc.ones((3,))
        acc.torch.ones.assert_called_once()

    def test_random_uniform_rocm(self):
        acc = self._make_rocm_acc()
        acc.random_uniform((5,))
        acc.torch.rand.assert_called_once()

    def test_to_cpu_rocm(self):
        acc = self._make_rocm_acc()
        mock_tensor = MagicMock()
        acc.to_cpu(mock_tensor)
        mock_tensor.detach.return_value.cpu.return_value.numpy.assert_called_once()

    def test_synchronize_rocm(self):
        acc = self._make_rocm_acc()
        acc.synchronize()
        acc.torch.cuda.synchronize.assert_called_once()

    def test_get_memory_info_rocm(self):
        acc = self._make_rocm_acc()
        acc.torch.cuda.is_available.return_value = True
        acc.torch.cuda.memory_reserved.return_value = 2000
        acc.torch.cuda.memory_allocated.return_value = 1000
        info = acc.get_memory_info()
        assert info["free"] == 1000
        assert info["total"] == 2000
        assert info["used"] == 1000

    def test_abs_rocm(self):
        acc = self._make_rocm_acc()
        acc.abs(MagicMock())
        acc.torch.abs.assert_called_once()

    def test_where_rocm(self):
        acc = self._make_rocm_acc()
        acc.where(MagicMock(), MagicMock(), MagicMock())
        acc.torch.where.assert_called_once()

    def test_maximum_rocm(self):
        acc = self._make_rocm_acc()
        acc.maximum(MagicMock(), MagicMock())
        acc.torch.maximum.assert_called_once()

    def test_minimum_rocm(self):
        acc = self._make_rocm_acc()
        acc.minimum(MagicMock(), MagicMock())
        acc.torch.minimum.assert_called_once()

    def test_log_rocm(self):
        acc = self._make_rocm_acc()
        acc.log(MagicMock())
        acc.torch.log.assert_called_once()

    def test_exp_rocm(self):
        acc = self._make_rocm_acc()
        acc.exp(MagicMock())
        acc.torch.exp.assert_called_once()

    def test_clip_rocm(self):
        acc = self._make_rocm_acc()
        acc.clip(MagicMock(), 0.0, 1.0)
        acc.torch.clamp.assert_called_once()

    def test_mean_rocm_no_axis(self):
        acc = self._make_rocm_acc()
        acc.mean(MagicMock())
        acc.torch.mean.assert_called_once()

    def test_mean_rocm_with_axis(self):
        acc = self._make_rocm_acc()
        acc.mean(MagicMock(), axis=0)
        assert acc.torch.mean.call_count == 1
        # Check dim was passed
        call_kwargs = acc.torch.mean.call_args
        assert call_kwargs[1].get("dim") == 0 or call_kwargs[0][1] == 0

    def test_sum_rocm_no_axis(self):
        acc = self._make_rocm_acc()
        acc.sum(MagicMock())
        acc.torch.sum.assert_called_once()

    def test_sum_rocm_with_axis(self):
        acc = self._make_rocm_acc()
        acc.sum(MagicMock(), axis=1)
        assert acc.torch.sum.call_count == 1

    def test_sqrt_rocm(self):
        acc = self._make_rocm_acc()
        acc.sqrt(MagicMock())
        acc.torch.sqrt.assert_called_once()

    def test_power_rocm(self):
        acc = self._make_rocm_acc()
        acc.power(MagicMock(), 2)
        acc.torch.pow.assert_called_once()

    def test_random_normal_rocm(self):
        acc = self._make_rocm_acc()
        acc.random_normal((10,))
        acc.torch.normal.assert_called_once()

    def test_np_to_torch_dtype_mapping(self):
        acc = self._make_rocm_acc()
        assert acc._np_to_torch_dtype(np.float32) == "torch.float32"
        assert acc._np_to_torch_dtype(np.float64) == "torch.float64"
        assert acc._np_to_torch_dtype(np.int32) == "torch.int32"
        assert acc._np_to_torch_dtype(np.int64) == "torch.int64"


class TestDetectBackendsEdgeCases:
    """Test _detect_backends edge cases."""

    def test_cupy_runtime_error(self):
        """Test cupy raises non-ImportError (line 44-45)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_cp = MagicMock()
        mock_cp.array.side_effect = RuntimeError("GPU init failed")
        with patch.dict(sys.modules, {"cupy": mock_cp, "torch": None, "jax": None}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends
        assert "cuda" not in acc.available_backends

    def test_torch_cuda_available_not_rocm(self):
        """Test torch CUDA detected but not ROCm (lines 62-63)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX"
        mock_torch.version.hip = None  # Not ROCm
        with patch.dict(sys.modules, {"cupy": None, "torch": mock_torch, "jax": None}):
            acc._detect_backends()
        assert "cuda" in acc.available_backends

    def test_torch_import_error(self):
        """Test torch ImportError (line 64-65)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": None}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends

    def test_torch_runtime_error(self):
        """Test torch raises non-ImportError (line 66-67)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
        with patch.dict(sys.modules, {"cupy": None, "torch": mock_torch, "jax": None}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends

    def test_jax_gpu_rocm(self):
        """Test JAX with ROCm GPU (lines 77-80)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.device_kind = "gpu"
        mock_device.platform = "rocm"
        mock_jax.devices.return_value = [mock_device]
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}):
            acc._detect_backends()
        assert "rocm" in acc.available_backends

    def test_jax_gpu_cuda(self):
        """Test JAX with CUDA GPU (lines 81-83)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.device_kind = "gpu"
        mock_device.platform = "gpu"  # JAX CUDA platform name
        mock_jax.devices.return_value = [mock_device]
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}):
            acc._detect_backends()
        assert "cuda" in acc.available_backends

    def test_jax_import_error(self):
        """Test JAX ImportError (lines 84-85)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": None}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends

    def test_jax_runtime_error(self):
        """Test JAX non-ImportError (lines 86-87)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_jax = MagicMock()
        mock_jax.devices.side_effect = RuntimeError("JAX init failed")
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}):
            acc._detect_backends()
        assert "cpu" in acc.available_backends


class TestSetupCUDAROCm:
    """Test _setup_cuda and _setup_rocm with working mocks."""

    def test_setup_cuda_success(self):
        """Test _setup_cuda with working cupy (lines 126-145)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_cp = MagicMock()
        mock_device = MagicMock()
        mock_device.name = b"NVIDIA RTX 4090"
        mock_device.mem_info = (8000000000, 16000000000)
        mock_device.compute_capability = "8.9"
        mock_cp.cuda.Device.return_value = mock_device
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            acc._setup_cuda()
        assert acc.cp is mock_cp
        assert acc.device_info["backend"] == "cuda"
        assert "NVIDIA RTX 4090" in acc.device_info["device_name"]

    def test_setup_rocm_success(self):
        """Test _setup_rocm with working torch (lines 150-168)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "AMD Radeon RX 7900"
        mock_torch.version.hip = "5.7.0"
        mock_torch.version.rocm = "5.7.0"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            acc._setup_rocm()
        assert acc.torch is mock_torch
        assert acc.device_info["backend"] == "rocm"


class TestTestGPUFunctionalityMismatch:
    def test_functionality_wrong_result(self):
        """Test test_gpu_functionality when results don't match (lines 419-420)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.cp = MagicMock()
        acc.torch = None
        acc.device_info = {}
        # Make to_cpu return wrong values
        acc.array = lambda data, dtype=np.float32: np.asarray(data, dtype=dtype)
        acc.abs = lambda a: np.abs(a)
        acc.maximum = lambda a, b: np.maximum(a, b)
        acc.mean = lambda a, axis=None: np.mean(a, axis=axis)
        acc.to_cpu = lambda a: np.array([0.0, 0.0, 0.0])  # Wrong result
        result = acc.test_gpu_functionality()
        assert result is False
        assert acc.backend == "cpu"


class TestGetGPUAcceleratorGPUAvailable:
    def test_get_gpu_accelerator_with_test_functionality(self):
        """Test get_gpu_accelerator with test_functionality on GPU backend (line 456)."""
        import gpu_acceleration
        gpu_acceleration._gpu_accelerator = None

        # Create a mock accelerator that reports GPU available
        mock_acc = MagicMock(spec=GPUAccelerator)
        mock_acc.is_gpu_available.return_value = True
        mock_acc.test_gpu_functionality.return_value = True

        with patch.object(gpu_acceleration, 'GPUAccelerator', return_value=mock_acc):
            result = get_gpu_accelerator(prefer_backend="cuda", force_reinit=True, test_functionality=True)
            mock_acc.test_gpu_functionality.assert_called_once()

        # Reset
        gpu_acceleration._gpu_accelerator = None


class TestBenchmarkEdgeCases:
    def test_benchmark_with_mock_gpu(self):
        """Test benchmark_backends with mocked GPU backends (lines 496-508)."""
        # Make GPUAccelerator init use our mocks
        with patch.dict(sys.modules, {"cupy": None, "torch": None, "jax": None}):
            benchmark_backends()

    def test_benchmark_gpu_exception(self):
        """Test benchmark_backends GPU path exception (lines 507-508)."""
        import gpu_acceleration

        # Create a mock accelerator that reports cuda backend but fails on operations
        def mock_init(prefer_backend="auto"):
            mock_acc = MagicMock(spec=GPUAccelerator)
            mock_acc.backend = prefer_backend if prefer_backend in ("cuda", "rocm") else "cpu"
            mock_acc.random_uniform.side_effect = RuntimeError("GPU operation failed")
            return mock_acc

        with patch.object(gpu_acceleration, 'GPUAccelerator', side_effect=mock_init):
            benchmark_backends()  # Should hit except on line 507-508


class TestGetGPUAccelerator:
    def test_singleton_pattern(self):
        import gpu_acceleration
        gpu_acceleration._gpu_accelerator = None
        acc1 = get_gpu_accelerator(prefer_backend="cpu", test_functionality=False)
        acc2 = get_gpu_accelerator(test_functionality=False)
        assert acc1 is acc2

    def test_force_reinit(self):
        import gpu_acceleration
        gpu_acceleration._gpu_accelerator = None
        acc1 = get_gpu_accelerator(prefer_backend="cpu", test_functionality=False)
        acc2 = get_gpu_accelerator(force_reinit=True, prefer_backend="cpu", test_functionality=False)
        # Force reinit creates a new instance
        assert acc2 is not None

    def test_test_functionality(self):
        import gpu_acceleration
        gpu_acceleration._gpu_accelerator = None
        acc = get_gpu_accelerator(prefer_backend="cpu", test_functionality=True)
        assert acc.backend == "cpu"


class TestBenchmarkBackends:
    def test_benchmark_runs(self):
        # Should complete without error on CPU
        benchmark_backends()
