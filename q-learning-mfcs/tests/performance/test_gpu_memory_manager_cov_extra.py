"""Extra coverage tests for gpu_memory_manager.py - covering lines 432, 434."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock torch before importing
from unittest.mock import MagicMock, patch

mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.backends.mps.is_available.return_value = False
cpu_device = MagicMock()
cpu_device.type = "cpu"
mock_torch.device.return_value = cpu_device

_orig_torch = sys.modules.get("torch")
sys.modules["torch"] = mock_torch

from performance.gpu_memory_manager import (
import pytest
    GPUMemoryManager,
    MemoryStats,
    demonstrate_memory_management,
)

# Restore original torch modules to prevent cross-contamination
if _orig_torch is not None:
    sys.modules["torch"] = _orig_torch
else:
    sys.modules.pop("torch", None)


@pytest.mark.coverage_extra
class TestDemonstrateMemoryManagementBranches:
    """Test the warning/recommendation branches in demonstrate_memory_management."""

    def test_demonstrate_with_warnings_and_recommendations(self):
        """Cover lines 432 and 434: when health has warnings and recommendations."""
        fake_health = {
            "status": "warning",
            "warnings": ["GPU memory high: 85.0%"],
            "critical_issues": [],
            "recommendations": ["Consider reducing batch size"],
        }

        with patch.object(
            GPUMemoryManager, "check_memory_health", return_value=fake_health
        ):
            mgr = demonstrate_memory_management()
            assert isinstance(mgr, GPUMemoryManager)

    def test_demonstrate_with_warnings_only(self):
        """Cover line 432: when health has warnings but no recommendations."""
        fake_health = {
            "status": "warning",
            "warnings": ["System memory high: 88.0%"],
            "critical_issues": [],
            "recommendations": [],
        }

        with patch.object(
            GPUMemoryManager, "check_memory_health", return_value=fake_health
        ):
            mgr = demonstrate_memory_management()
            assert isinstance(mgr, GPUMemoryManager)

    def test_demonstrate_with_recommendations_only(self):
        """Cover line 434: when health has recommendations but no warnings."""
        fake_health = {
            "status": "healthy",
            "warnings": [],
            "critical_issues": [],
            "recommendations": ["Close unnecessary applications"],
        }

        with patch.object(
            GPUMemoryManager, "check_memory_health", return_value=fake_health
        ):
            mgr = demonstrate_memory_management()
            assert isinstance(mgr, GPUMemoryManager)
