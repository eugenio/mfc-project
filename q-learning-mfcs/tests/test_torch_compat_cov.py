"""Tests for torch_compat module - comprehensive coverage.

Covers all mock classes, fallback behavior, and utility functions
when PyTorch is not available.
"""
import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTorchNotAvailableError:
    """Tests for _TorchNotAvailableError."""

    def test_error_message_contains_feature(self):
        from torch_compat import _TorchNotAvailableError

        err = _TorchNotAvailableError("GPU computation")
        assert "GPU computation" in str(err)
        assert "requires PyTorch" in str(err)

    def test_error_message_default_feature(self):
        from torch_compat import _TorchNotAvailableError

        err = _TorchNotAvailableError()
        assert "This feature" in str(err)

    def test_is_import_error(self):
        from torch_compat import _TorchNotAvailableError

        assert issubclass(_TorchNotAvailableError, ImportError)


class TestMockDevice:
    """Tests for _MockDevice."""

    def test_init_default(self):
        from torch_compat import _MockDevice

        device = _MockDevice()
        assert device.type == "cpu"

    def test_init_custom(self):
        from torch_compat import _MockDevice

        device = _MockDevice("cuda")
        assert device.type == "cuda"

    def test_str(self):
        from torch_compat import _MockDevice

        device = _MockDevice("cpu")
        assert str(device) == "cpu"

    def test_repr(self):
        from torch_compat import _MockDevice

        device = _MockDevice("cpu")
        assert repr(device) == "device(type='cpu')"

    def test_eq_with_string(self):
        from torch_compat import _MockDevice

        device = _MockDevice("cpu")
        assert device == "cpu"
        assert not (device == "cuda")

    def test_eq_with_device(self):
        from torch_compat import _MockDevice

        d1 = _MockDevice("cpu")
        d2 = _MockDevice("cpu")
        d3 = _MockDevice("cuda")
        assert d1 == d2
        assert not (d1 == d3)

    def test_eq_with_other_type(self):
        from torch_compat import _MockDevice

        device = _MockDevice("cpu")
        assert not (device == 42)
        assert not (device == None)

    def test_eq_with_object_having_type(self):
        from torch_compat import _MockDevice

        class FakeDevice:
            type = "cpu"

        device = _MockDevice("cpu")
        assert device == FakeDevice()


class TestMockTensor:
    """Tests for _MockTensor."""

    def test_init_raises(self):
        from torch_compat import _MockTensor, _TorchNotAvailableError

        with pytest.raises(_TorchNotAvailableError, match="Tensor operations"):
            _MockTensor()

    def test_init_with_args_raises(self):
        from torch_compat import _MockTensor, _TorchNotAvailableError

        with pytest.raises(_TorchNotAvailableError):
            _MockTensor(1, 2, 3)


class TestMockModule:
    """Tests for _MockModule."""

    def test_init_raises(self):
        from torch_compat import _MockModule, _TorchNotAvailableError

        with pytest.raises(_TorchNotAvailableError, match="Neural network"):
            _MockModule()

    def test_parameters_raises(self):
        from torch_compat import _MockModule, _TorchNotAvailableError

        # Need to bypass __init__ to test parameters
        obj = object.__new__(_MockModule)
        with pytest.raises(_TorchNotAvailableError, match="Neural network"):
            obj.parameters()


class TestMockCuda:
    """Tests for _MockCuda."""

    def test_is_available(self):
        from torch_compat import _MockCuda

        assert _MockCuda.is_available() is False

    def test_device_count(self):
        from torch_compat import _MockCuda

        assert _MockCuda.device_count() == 0


class TestMockNNUtils:
    """Tests for _MockNNUtils."""

    def test_clip_grad_norm_raises(self):
        from torch_compat import _MockNNUtils, _TorchNotAvailableError

        with pytest.raises(_TorchNotAvailableError, match="Gradient clipping"):
            _MockNNUtils.clip_grad_norm_([], 1.0)


class TestMockNN:
    """Tests for _MockNN."""

    def test_module_attr(self):
        from torch_compat import _MockModule, _MockNN

        assert _MockNN.Module is _MockModule

    def test_utils_attr(self):
        from torch_compat import _MockNN, _MockNNUtils

        assert _MockNN.utils is _MockNNUtils


class TestMockTorch:
    """Tests for _MockTorch."""

    def test_cuda_attr(self):
        from torch_compat import _MockCuda, _MockTorch

        assert _MockTorch.cuda is _MockCuda

    def test_nn_attr(self):
        from torch_compat import _MockNN, _MockTorch

        assert _MockTorch.nn is _MockNN

    def test_tensor_attr(self):
        from torch_compat import _MockTensor, _MockTorch

        assert _MockTorch.Tensor is _MockTensor

    def test_float_tensor_attr(self):
        from torch_compat import _MockTensor, _MockTorch

        assert _MockTorch.FloatTensor is _MockTensor

    def test_device_cpu(self):
        from torch_compat import _MockTorch

        device = _MockTorch.device("cpu")
        assert device.type == "cpu"

    def test_device_non_cpu_falls_back(self):
        from torch_compat import _MockTorch

        device = _MockTorch.device("cuda")
        assert device.type == "cpu"


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_none(self):
        from torch_compat import TORCH_AVAILABLE, get_device

        device = get_device(None)
        if not TORCH_AVAILABLE:
            assert device.type == "cpu"

    def test_get_device_auto(self):
        from torch_compat import TORCH_AVAILABLE, get_device

        device = get_device("auto")
        if not TORCH_AVAILABLE:
            assert device.type == "cpu"

    def test_get_device_cpu(self):
        from torch_compat import TORCH_AVAILABLE, get_device

        device = get_device("cpu")
        if not TORCH_AVAILABLE:
            assert device.type == "cpu"

    def test_get_device_cuda_falls_back(self):
        from torch_compat import TORCH_AVAILABLE, get_device

        device = get_device("cuda")
        if not TORCH_AVAILABLE:
            assert device.type == "cpu"


class TestIsGpuAvailable:
    """Tests for is_gpu_available function."""

    def test_gpu_not_available_without_torch(self):
        from torch_compat import TORCH_AVAILABLE, is_gpu_available

        if not TORCH_AVAILABLE:
            assert is_gpu_available() is False


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self):
        from torch_compat import __all__

        assert "TORCH_AVAILABLE" in __all__
        assert "get_device" in __all__
        assert "is_gpu_available" in __all__
        assert "nn" in __all__
        assert "torch" in __all__

    def test_torch_available_bool(self):
        from torch_compat import TORCH_AVAILABLE

        assert isinstance(TORCH_AVAILABLE, bool)


class TestTorchAvailablePath:
    """Tests for torch available / unavailable code paths via reimport."""

    def test_import_error_path(self):
        """Test the ImportError path by simulating torch not importable."""
        import torch_compat

        # The module is already loaded; verify the error message is set
        if not torch_compat.TORCH_AVAILABLE:
            assert torch_compat._torch_import_error is not None

    def test_mock_torch_instance_when_unavailable(self):
        """When torch is not available, the module-level torch should be _MockTorch."""
        from torch_compat import TORCH_AVAILABLE, _MockTorch, torch

        if not TORCH_AVAILABLE:
            assert isinstance(torch, _MockTorch)

    def test_stub_torch_path(self):
        """Test path where torch is found but is a stub (no __version__/Tensor)."""
        stub_torch = MagicMock(spec=[])  # No __version__ or Tensor
        del stub_torch.__version__
        del stub_torch.Tensor
        stub_torch.__name__ = "torch"

        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch_compat" or key.startswith("torch"):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules["torch"] = stub_torch

        try:
            import torch_compat as tc_reload

            importlib.reload(tc_reload)
            assert tc_reload.TORCH_AVAILABLE is False
            assert "stub" in (tc_reload._torch_import_error or "")
        finally:
            del sys.modules["torch"]
            for key in list(sys.modules.keys()):
                if key == "torch_compat":
                    del sys.modules[key]
            sys.modules.update(saved_modules)
            importlib.reload(sys.modules["torch_compat"])

    def test_general_exception_path(self):
        """Test the generic Exception catch (line 48-49)."""
        class BadModule:
            def __getattr__(self, name):
                if name == "__version__":
                    raise RuntimeError("bad torch")
                return MagicMock()

        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch_compat" or key.startswith("torch"):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules["torch"] = BadModule()

        try:
            import torch_compat as tc_reload

            importlib.reload(tc_reload)
            assert tc_reload.TORCH_AVAILABLE is False
            assert "Error loading" in (tc_reload._torch_import_error or "")
        finally:
            if "torch" in sys.modules:
                del sys.modules["torch"]
            for key in list(sys.modules.keys()):
                if key == "torch_compat":
                    del sys.modules[key]
            sys.modules.update(saved_modules)
            importlib.reload(sys.modules["torch_compat"])

    def test_torch_available_get_device(self):
        """Test get_device when torch IS available (mock real torch)."""
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.Tensor = MagicMock
        mock_torch.cuda.is_available.return_value = False
        mock_device = MagicMock()
        mock_device.type = "cpu"
        mock_torch.device.return_value = mock_device
        mock_torch.nn = MagicMock()

        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch_compat" or key.startswith("torch"):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules["torch"] = mock_torch

        try:
            import torch_compat as tc_reload

            importlib.reload(tc_reload)
            assert tc_reload.TORCH_AVAILABLE is True

            # Test get_device with auto
            device = tc_reload.get_device("auto")
            mock_torch.device.assert_called()

            # Test get_device with explicit
            tc_reload.get_device("cpu")

            # Test is_gpu_available
            result = tc_reload.is_gpu_available()
            assert result is False
        finally:
            del sys.modules["torch"]
            for key in list(sys.modules.keys()):
                if key == "torch_compat":
                    del sys.modules[key]
            sys.modules.update(saved_modules)
            importlib.reload(sys.modules["torch_compat"])

    def test_torch_available_gpu(self):
        """Test is_gpu_available when torch is available with CUDA."""
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.Tensor = MagicMock
        mock_torch.cuda.is_available.return_value = True
        mock_torch.nn = MagicMock()

        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == "torch_compat" or key.startswith("torch"):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules["torch"] = mock_torch

        try:
            import torch_compat as tc_reload

            importlib.reload(tc_reload)
            assert tc_reload.is_gpu_available() is True
        finally:
            del sys.modules["torch"]
            for key in list(sys.modules.keys()):
                if key == "torch_compat":
                    del sys.modules[key]
            sys.modules.update(saved_modules)
            importlib.reload(sys.modules["torch_compat"])
