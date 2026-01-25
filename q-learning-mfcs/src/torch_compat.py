"""Torch Compatibility Module for Optional PyTorch Support.

This module provides a compatibility layer for PyTorch, allowing the codebase
to work with or without PyTorch installed. When PyTorch is available, it exports
the real torch module. When PyTorch is not available, it provides minimal stubs
that raise informative errors when GPU/neural network features are used.

Usage:
    from torch_compat import torch, nn, TORCH_AVAILABLE

    if TORCH_AVAILABLE:
        # Use full PyTorch functionality
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Fall back to CPU-only numpy-based operations
        device = "cpu"

Created: 2026-01-25
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import PyTorch
TORCH_AVAILABLE = False
_torch_import_error: str | None = None

try:
    import torch as _torch

    # Verify it's the real PyTorch, not a namespace stub
    if hasattr(_torch, "__version__") and hasattr(_torch, "Tensor"):
        torch = _torch
        nn = _torch.nn
        TORCH_AVAILABLE = True
        logger.debug(f"PyTorch {torch.__version__} loaded successfully")
    else:
        _torch_import_error = (
            "Found 'torch' module but it appears to be a stub or namespace package, "
            "not the real PyTorch library. Please install PyTorch: pip install torch"
        )
except ImportError as e:
    _torch_import_error = f"PyTorch not installed: {e}"
except Exception as e:
    _torch_import_error = f"Error loading PyTorch: {e}"


class _TorchNotAvailableError(ImportError):
    """Raised when PyTorch functionality is used but PyTorch is not available."""

    def __init__(self, feature: str = "This feature"):
        msg = (
            f"{feature} requires PyTorch, but PyTorch is not available. "
            f"Reason: {_torch_import_error or 'Unknown'}. "
            "Install PyTorch with: pip install torch, or use pixi -e nvidia-gpu environment."
        )
        super().__init__(msg)


class _MockDevice:
    """Mock device class for when PyTorch is not available."""

    def __init__(self, device_type: str = "cpu"):
        self.type = device_type

    def __str__(self) -> str:
        return self.type

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.type == other
        if hasattr(other, "type"):
            return bool(self.type == other.type)
        return False


class _MockTensor:
    """Mock tensor class that raises errors when used."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise _TorchNotAvailableError("Tensor operations")


class _MockModule:
    """Mock nn.Module class that raises errors when used."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise _TorchNotAvailableError("Neural network modules")

    def parameters(self, *args: Any, **kwargs: Any) -> None:
        raise _TorchNotAvailableError("Neural network modules")


class _MockCuda:
    """Mock cuda module for GPU availability checks."""

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


class _MockNNUtils:
    """Mock nn.utils module."""

    @staticmethod
    def clip_grad_norm_(parameters: Any, max_norm: float) -> float:
        raise _TorchNotAvailableError("Gradient clipping")


class _MockNN:
    """Mock nn module."""

    Module = _MockModule
    utils = _MockNNUtils


class _MockTorch:
    """Mock torch module for when PyTorch is not available."""

    cuda = _MockCuda
    nn = _MockNN
    Tensor = _MockTensor
    FloatTensor = _MockTensor

    @staticmethod
    def device(device_type: str = "cpu") -> _MockDevice:
        """Create a mock device.

        When PyTorch is not available, only 'cpu' device is valid.
        """
        if device_type != "cpu":
            logger.warning(
                f"Requested device '{device_type}' but PyTorch is not available. "
                "Falling back to CPU.",
            )
        return _MockDevice("cpu")


# Export the appropriate torch module
if not TORCH_AVAILABLE:
    torch = _MockTorch()  # type: ignore[assignment]
    nn = _MockNN  # type: ignore[assignment,misc]
    logger.warning(
        f"PyTorch not available ({_torch_import_error}). "
        "GPU acceleration and neural network features will be disabled.",
    )


def get_device(requested: str | None = None) -> Any:
    """Get the best available device.

    Args:
        requested: Requested device ('cuda', 'cpu', 'auto', or None for auto)

    Returns:
        torch.device if PyTorch is available, else _MockDevice('cpu')

    """
    if not TORCH_AVAILABLE:
        if requested and requested not in ("cpu", "auto", None):
            logger.warning(
                f"Requested device '{requested}' but PyTorch is not available. "
                "Using CPU fallback.",
            )
        return _MockDevice("cpu")

    if requested == "auto" or requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def is_gpu_available() -> bool:
    """Check if GPU is available for computation.

    Returns:
        True if PyTorch is available and CUDA is available, False otherwise

    """
    if not TORCH_AVAILABLE:
        return False
    return bool(torch.cuda.is_available())


__all__ = [
    "TORCH_AVAILABLE",
    "get_device",
    "is_gpu_available",
    "nn",
    "torch",
]
