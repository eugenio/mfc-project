"""Comprehensive tests for torch_compat module with torch mocked.

Covers all classes, functions, branches, and fallback behavior
to achieve 99%+ coverage of torch_compat.py.
"""
import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helper: reload torch_compat under controlled conditions
# ---------------------------------------------------------------------------
def _reload_torch_compat(**overrides):
    """Reload torch_compat after manipulating sys.modules['torch']."""
    # Remove cached module so importlib.reload picks up changes
    mods_to_drop = [k for k in sys.modules if k.startswith("torch_compat")]
    for m in mods_to_drop:
        del sys.modules[m]
    import torch_compat
    return importlib.reload(torch_compat)


# ===================================================================
# Tests for _TorchNotAvailableError
# ===================================================================
class TestTorchNotAvailableError:
    def test_message_includes_feature(self):
        from torch_compat import _TorchNotAvailableError
        err = _TorchNotAvailableError("GPU computation")
        msg = str(err)
        assert "GPU computation" in msg
        assert "requires PyTorch" in msg
        assert "pip install torch" in msg

    def test_message_default_feature(self):
        from torch_compat import _TorchNotAvailableError
        err = _TorchNotAvailableError()
        assert "This feature" in str(err)

    def test_is_import_error(self):
        from torch_compat import _TorchNotAvailableError
        assert issubclass(_TorchNotAvailableError, ImportError)
        with pytest.raises(ImportError):
            raise _TorchNotAvailableError("test")


# ===================================================================
# Tests for _MockDevice
# ===================================================================
class TestMockDevice:
    def test_default_type(self):
        from torch_compat import _MockDevice
        d = _MockDevice()
        assert d.type == "cpu"

    def test_custom_type(self):
        from torch_compat import _MockDevice
        d = _MockDevice("cuda")
        assert d.type == "cuda"

    def test_str(self):
        from torch_compat import _MockDevice
        assert str(_MockDevice("cpu")) == "cpu"

    def test_repr(self):
        from torch_compat import _MockDevice
        assert repr(_MockDevice("cpu")) == "device(type='cpu')"

    def test_eq_string(self):
        from torch_compat import _MockDevice
        d = _MockDevice("cpu")
        assert d == "cpu"
        assert not (d == "cuda")

    def test_eq_other_device(self):
        from torch_compat import _MockDevice
        d1 = _MockDevice("cpu")
        d2 = _MockDevice("cpu")
        d3 = _MockDevice("cuda")
        assert d1 == d2
        assert not (d1 == d3)

    def test_eq_non_comparable(self):
        from torch_compat import _MockDevice
        d = _MockDevice("cpu")
        assert not (d == 42)
        assert not (d == None)  # noqa: E711

    def test_eq_object_with_type_attr(self):
        from torch_compat import _MockDevice
        other = MagicMock()
        other.type = "cpu"
        assert _MockDevice("cpu") == other


# ===================================================================
# Tests for _MockTensor
# ===================================================================
class TestMockTensor:
    def test_instantiation_raises(self):
        from torch_compat import _MockTensor, _TorchNotAvailableError
        with pytest.raises(_TorchNotAvailableError, match="Tensor operations"):
            _MockTensor()

    def test_instantiation_with_args_raises(self):
        from torch_compat import _MockTensor, _TorchNotAvailableError
        with pytest.raises(_TorchNotAvailableError):
            _MockTensor(1, 2, 3)


# ===================================================================
# Tests for _MockModule
# ===================================================================
class TestMockModule:
    def test_init_raises(self):
        from torch_compat import _MockModule, _TorchNotAvailableError
        with pytest.raises(_TorchNotAvailableError, match="Neural network"):
            _MockModule()

    def test_parameters_raises(self):
        from torch_compat import _MockModule, _TorchNotAvailableError
        # bypass __init__ to test parameters
        obj = object.__new__(_MockModule)
        with pytest.raises(_TorchNotAvailableError):
            obj.parameters()


# ===================================================================
# Tests for _MockCuda
# ===================================================================
class TestMockCuda:
    def test_is_available(self):
        from torch_compat import _MockCuda
        assert _MockCuda.is_available() is False

    def test_device_count(self):
        from torch_compat import _MockCuda
        assert _MockCuda.device_count() == 0


# ===================================================================
# Tests for _MockNNUtils
# ===================================================================
class TestMockNNUtils:
    def test_clip_grad_norm_raises(self):
        from torch_compat import _MockNNUtils, _TorchNotAvailableError
        with pytest.raises(_TorchNotAvailableError, match="Gradient clipping"):
            _MockNNUtils.clip_grad_norm_([], 1.0)


# ===================================================================
# Tests for _MockNN
# ===================================================================
class TestMockNN:
    def test_has_module(self):
        from torch_compat import _MockNN, _MockModule
        assert _MockNN.Module is _MockModule

    def test_has_utils(self):
        from torch_compat import _MockNN, _MockNNUtils
        assert _MockNN.utils is _MockNNUtils


# ===================================================================
# Tests for _MockTorch
# ===================================================================
class TestMockTorch:
    def test_device_cpu(self):
        from torch_compat import _MockTorch, _MockDevice
        d = _MockTorch.device("cpu")
        assert isinstance(d, _MockDevice)
        assert d.type == "cpu"

    def test_device_non_cpu_warns(self):
        from torch_compat import _MockTorch, _MockDevice
        with patch("torch_compat.logger") as mock_logger:
            d = _MockTorch.device("cuda")
            mock_logger.warning.assert_called()
        assert isinstance(d, _MockDevice)
        assert d.type == "cpu"

    def test_cuda_attr(self):
        from torch_compat import _MockTorch, _MockCuda
        assert _MockTorch.cuda is _MockCuda

    def test_nn_attr(self):
        from torch_compat import _MockTorch, _MockNN
        assert _MockTorch.nn is _MockNN

    def test_tensor_attr(self):
        from torch_compat import _MockTorch, _MockTensor
        assert _MockTorch.Tensor is _MockTensor
        assert _MockTorch.FloatTensor is _MockTensor


# ===================================================================
# Tests for get_device()
# ===================================================================
class TestGetDevice:
    def test_not_available_cpu(self):
        from torch_compat import _MockDevice
        tc = _reload_torch_compat()
        # torch_compat with real mock, TORCH_AVAILABLE should be False
        # because torch is MagicMock (no __version__)
        if not tc.TORCH_AVAILABLE:
            d = tc.get_device("cpu")
            assert isinstance(d, _MockDevice)
            assert d.type == "cpu"

    def test_not_available_auto(self):
        tc = _reload_torch_compat()
        if not tc.TORCH_AVAILABLE:
            d = tc.get_device("auto")
            assert d.type == "cpu"

    def test_not_available_none(self):
        tc = _reload_torch_compat()
        if not tc.TORCH_AVAILABLE:
            d = tc.get_device(None)
            assert d.type == "cpu"

    def test_not_available_cuda_warns(self):
        tc = _reload_torch_compat()
        if not tc.TORCH_AVAILABLE:
            with patch("torch_compat.logger") as mock_logger:
                d = tc.get_device("cuda")
                mock_logger.warning.assert_called()
            assert d.type == "cpu"


# ===================================================================
# Tests for is_gpu_available()
# ===================================================================
class TestIsGpuAvailable:
    def test_not_available(self):
        tc = _reload_torch_compat()
        if not tc.TORCH_AVAILABLE:
            assert tc.is_gpu_available() is False


# ===================================================================
# Tests for module-level initialization branches
# ===================================================================
class TestModuleInit:
    def test_torch_import_error_branch(self):
        """Test the ImportError branch at module level."""
        orig = sys.modules.pop("torch", None)
        mods_to_drop = [k for k in sys.modules if k.startswith("torch_compat")]
        for m in mods_to_drop:
            del sys.modules[m]
        # Make torch import fail
        import builtins
        _real_import = builtins.__import__
        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return _real_import(name, *args, **kwargs)
        builtins.__import__ = _fake_import
        try:
            import torch_compat
            tc = importlib.reload(torch_compat)
            assert tc.TORCH_AVAILABLE is False
            assert tc._torch_import_error is not None
            assert "not installed" in tc._torch_import_error
        finally:
            builtins.__import__ = _real_import
            if orig is not None:
                sys.modules["torch"] = orig

    def test_torch_generic_exception_branch(self):
        """Test the generic Exception branch at module level."""
        orig = sys.modules.pop("torch", None)
        mods_to_drop = [k for k in sys.modules if k.startswith("torch_compat")]
        for m in mods_to_drop:
            del sys.modules[m]
        import builtins
        _real_import = builtins.__import__
        def _fake_import(name, *args, **kwargs):
            if name == "torch":
                raise RuntimeError("bad torch")
            return _real_import(name, *args, **kwargs)
        builtins.__import__ = _fake_import
        try:
            import torch_compat
            tc = importlib.reload(torch_compat)
            assert tc.TORCH_AVAILABLE is False
            assert "Error loading" in tc._torch_import_error
        finally:
            builtins.__import__ = _real_import
            if orig is not None:
                sys.modules["torch"] = orig

    def test_torch_stub_module_branch(self):
        """Test the branch where torch exists but is a stub."""
        orig = sys.modules.get("torch")
        mods_to_drop = [k for k in sys.modules if k.startswith("torch_compat")]
        for m in mods_to_drop:
            del sys.modules[m]
        # Create a stub torch that has no __version__ or Tensor
        stub = MagicMock(spec=[])
        sys.modules["torch"] = stub
        try:
            import torch_compat
            tc = importlib.reload(torch_compat)
            assert tc.TORCH_AVAILABLE is False
            assert "stub" in tc._torch_import_error
        finally:
            if orig is not None:
                sys.modules["torch"] = orig
            else:
                sys.modules.pop("torch", None)

    def test_exports(self):
        import torch_compat
        assert "TORCH_AVAILABLE" in torch_compat.__all__
        assert "get_device" in torch_compat.__all__
        assert "is_gpu_available" in torch_compat.__all__
        assert "nn" in torch_compat.__all__
        assert "torch" in torch_compat.__all__
