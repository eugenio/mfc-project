"""Extended coverage tests for gpu_acceleration module.

Covers edge cases in detect_backends, setup methods, and math operations
not fully exercised by the existing test suite.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gpu_acceleration import GPUAccelerator, get_gpu_accelerator, benchmark_backends


@pytest.mark.coverage_extra
class TestDetectBackendsExtraCoverage:
    """Additional detect_backends edge cases."""

    def test_cupy_import_error(self):
        """Cover cupy ImportError path (line 42-43)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        # cupy raises ImportError, torch and jax also unavailable
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": None}
        ):
            acc._detect_backends()
        assert "cpu" in acc.available_backends

    def test_torch_rocm_detected(self):
        """Cover torch ROCm detection (lines 56-61)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "AMD Radeon"
        mock_torch.version.hip = "5.7.0"  # ROCm indicator
        with patch.dict(
            sys.modules, {"cupy": None, "torch": mock_torch, "jax": None}
        ):
            acc._detect_backends()
        assert "rocm" in acc.available_backends

    def test_torch_cuda_already_in_backends(self):
        """Cover torch CUDA skip when already detected (line 62)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = ["cuda"]  # Already detected via cupy
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX"
        mock_torch.version.hip = None
        with patch.dict(
            sys.modules, {"cupy": None, "torch": mock_torch, "jax": None}
        ):
            acc._detect_backends()
        # cuda should still be there, not duplicated
        assert acc.available_backends.count("cuda") == 1

    def test_jax_rocm_already_detected(self):
        """Cover JAX rocm skip when already detected (line 79)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = ["rocm"]
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.device_kind = "gpu"
        mock_device.platform = "rocm"
        mock_jax.devices.return_value = [mock_device]
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}
        ):
            acc._detect_backends()
        assert acc.available_backends.count("rocm") == 1

    def test_jax_cuda_already_detected(self):
        """Cover JAX CUDA skip when already detected (line 82)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = ["cuda"]
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.device_kind = "gpu"
        mock_device.platform = "gpu"
        mock_jax.devices.return_value = [mock_device]
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}
        ):
            acc._detect_backends()
        assert acc.available_backends.count("cuda") == 1

    def test_jax_no_gpu_devices(self):
        """Cover JAX with no GPU devices (no backends added)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.device_kind = "cpu"  # Not GPU
        mock_jax.devices.return_value = [mock_device]
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": mock_jax}
        ):
            acc._detect_backends()
        assert "cpu" in acc.available_backends
        assert "cuda" not in acc.available_backends
        assert "rocm" not in acc.available_backends

    def test_cpu_already_in_backends(self):
        """Cover cpu already in backends (line 90 skip)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = ["cpu"]
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": None}
        ):
            acc._detect_backends()
        assert acc.available_backends.count("cpu") == 1

    def test_only_cpu_backend_print_line(self):
        """Cover the print when only CPU is available (line 93-94)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.available_backends = []
        with patch.dict(
            sys.modules, {"cupy": None, "torch": None, "jax": None}
        ):
            acc._detect_backends()
        assert acc.available_backends == ["cpu"]


@pytest.mark.coverage_extra
class TestInitializeBackendExtraCoverage:
    """Additional tests for _initialize_backend."""

    def test_empty_available_backends(self):
        """Cover fallback when available_backends is empty (line 110)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = None
        acc.device_info = {}
        acc.available_backends = []
        acc.cp = None
        acc.torch = None
        acc._initialize_backend("cuda")
        assert acc.backend == "cpu"


@pytest.mark.coverage_extra
class TestSetupCUDAExtraCoverage:
    """Additional tests for _setup_cuda."""

    def test_setup_cuda_device_name_string(self):
        """Cover device.name as string (not bytes) (line 139-140)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_cp = MagicMock()
        mock_device = MagicMock()
        mock_device.name = "NVIDIA RTX 4090"  # String, not bytes
        mock_device.mem_info = (8e9, 16e9)
        mock_device.compute_capability = "8.9"
        mock_cp.cuda.Device.return_value = mock_device
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            acc._setup_cuda()
        assert acc.device_info["device_name"] == "NVIDIA RTX 4090"

    def test_setup_cuda_exception_fallback(self):
        """Cover _setup_cuda exception -> _setup_cpu (line 147-148)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_cp = MagicMock()
        mock_cp.cuda.Device.side_effect = RuntimeError("CUDA init failed")
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            acc._setup_cuda()
        assert acc.backend == "cpu"


@pytest.mark.coverage_extra
class TestSetupROCmExtraCoverage:
    """Additional tests for _setup_rocm."""

    def test_setup_rocm_cuda_not_available(self):
        """Cover _setup_rocm when torch.cuda.is_available() is False."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            acc._setup_rocm()
        # Should set torch but device_info won't be set
        assert acc.torch is mock_torch

    def test_setup_rocm_exception(self):
        """Cover _setup_rocm exception -> _setup_cpu (line 170-171)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("ROCm error")
        with patch.dict(sys.modules, {"torch": mock_torch}):
            acc._setup_rocm()
        assert acc.backend == "cpu"


@pytest.mark.coverage_extra
class TestMathOperationsExtraCoverage:
    """Additional math operation edge cases."""

    def test_mean_rocm_axis_kwarg(self):
        """Cover torch.mean with dim= keyword (line 345)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.cp = None
        acc.torch = MagicMock()
        mock_arr = MagicMock()
        acc.mean(mock_arr, axis=1)
        acc.torch.mean.assert_called_once_with(mock_arr, dim=1)

    def test_sum_rocm_axis_kwarg(self):
        """Cover torch.sum with dim= keyword (line 355)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.cp = None
        acc.torch = MagicMock()
        mock_arr = MagicMock()
        acc.sum(mock_arr, axis=0)
        acc.torch.sum.assert_called_once_with(mock_arr, dim=0)

    def test_np_to_torch_dtype_bool(self):
        """Cover bool dtype mapping (line 245)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.torch = MagicMock()
        acc.torch.bool = "torch.bool"
        result = acc._np_to_torch_dtype(np.bool_)
        assert result == "torch.bool"

    def test_np_to_torch_dtype_unknown(self):
        """Cover unknown dtype fallback (line 247)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.torch = MagicMock()
        acc.torch.float32 = "torch.float32"
        result = acc._np_to_torch_dtype(np.complex64)
        assert result == "torch.float32"  # Falls back to float32


@pytest.mark.coverage_extra
class TestGetMemoryInfoExtraCoverage:
    """Additional get_memory_info edge cases."""

    def test_get_memory_info_rocm_not_available(self):
        """Cover rocm memory info when cuda not available (line 272)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.cp = None
        acc.torch = MagicMock()
        acc.torch.cuda.is_available.return_value = False
        result = acc.get_memory_info()
        assert result is None


@pytest.mark.coverage_extra
class TestBenchmarkExtraCoverage:
    """Additional benchmark_backends edge cases."""

    def test_benchmark_rocm_path(self):
        """Cover benchmark rocm backend path (lines 498-504)."""
        import gpu_acceleration

        def mock_gpu_init(prefer_backend="auto"):
            acc = MagicMock(spec=GPUAccelerator)
            if prefer_backend == "rocm":
                acc.backend = "rocm"
                acc.torch = MagicMock()
                acc.random_uniform = lambda shape: np.random.rand(*shape).astype(
                    np.float32
                )
                acc.synchronize = MagicMock()
            else:
                acc.backend = "cpu"
            return acc

        with patch.object(
            gpu_acceleration, "GPUAccelerator", side_effect=mock_gpu_init
        ):
            benchmark_backends()

    def test_benchmark_cuda_path(self):
        """Cover benchmark cuda backend path (lines 496-497)."""
        import gpu_acceleration

        def mock_gpu_init(prefer_backend="auto"):
            acc = MagicMock(spec=GPUAccelerator)
            if prefer_backend == "cuda":
                acc.backend = "cuda"
                acc.cp = MagicMock()
                arr = np.random.rand(100, 100).astype(np.float32)
                acc.random_uniform = lambda shape: arr
                acc.synchronize = MagicMock()
            else:
                acc.backend = "cpu"
            return acc

        with patch.object(
            gpu_acceleration, "GPUAccelerator", side_effect=mock_gpu_init
        ):
            benchmark_backends()


@pytest.mark.coverage_extra
class TestGetGPUAcceleratorExtraCoverage:
    """Additional get_gpu_accelerator tests."""

    def test_no_test_functionality_on_cpu(self):
        """Cover skip test_functionality when not GPU (line 455)."""
        import gpu_acceleration

        gpu_acceleration._gpu_accelerator = None
        acc = get_gpu_accelerator(
            prefer_backend="cpu",
            force_reinit=True,
            test_functionality=True,
        )
        assert acc.backend == "cpu"
        gpu_acceleration._gpu_accelerator = None

    def test_reuse_existing_instance(self):
        """Cover reuse of existing accelerator (line 451 false branch)."""
        import gpu_acceleration

        gpu_acceleration._gpu_accelerator = None
        acc1 = get_gpu_accelerator(
            prefer_backend="cpu", test_functionality=False
        )
        acc2 = get_gpu_accelerator(test_functionality=False)
        assert acc1 is acc2
        gpu_acceleration._gpu_accelerator = None


@pytest.mark.coverage_extra
class TestForCPUFallbackExtraCoverage:
    """Additional force_cpu_fallback tests."""

    def test_force_cpu_fallback_device_info(self):
        """Verify device_info after force_cpu_fallback."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.cp = MagicMock()
        acc.torch = MagicMock()
        acc.device_info = {"backend": "rocm"}
        acc.force_cpu_fallback()
        assert acc.device_info["backend"] == "cpu"
        assert acc.device_info["device_count"] == 0
        assert "Fallback" in acc.device_info["device_name"]


@pytest.mark.coverage_extra
class TestSetupBackendDispatch:
    """Test _setup_backend dispatch."""

    def test_setup_backend_cuda(self):
        """Cover _setup_backend dispatching to _setup_cuda (line 118)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cuda"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        # _setup_cuda will fail without real cupy and fall back to cpu
        acc._setup_backend()
        assert acc.backend in ("cuda", "cpu")

    def test_setup_backend_rocm(self):
        """Cover _setup_backend dispatching to _setup_rocm (line 119)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "rocm"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        acc._setup_backend()
        assert acc.backend in ("rocm", "cpu")

    def test_setup_backend_cpu(self):
        """Cover _setup_backend dispatching to _setup_cpu (line 122)."""
        acc = GPUAccelerator.__new__(GPUAccelerator)
        acc.backend = "cpu"
        acc.device_info = {}
        acc.cp = None
        acc.torch = None
        acc._setup_backend()
        assert acc.backend == "cpu"
