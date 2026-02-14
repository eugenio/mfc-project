"""Extra coverage tests for base_controller.py.

Covers the remaining uncovered path: extract_state_features (lines 154-176).
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock torch before importing base_controller
mock_torch = MagicMock()
mock_nn = MagicMock()

class MockModule:
    def __init__(self, *a, **kw):
        pass
    def parameters(self):
        return []
    def to(self, device):
        return self

mock_nn.Module = MockModule
mock_torch.nn = mock_nn
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.device = MagicMock(return_value="cpu")

_orig_torch = sys.modules.get("torch")
_orig_torch_nn = sys.modules.get("torch.nn")
_orig_torch_optim = sys.modules.get("torch.optim")
_orig_torch_nn_functional = sys.modules.get("torch.nn.functional")
_orig_torch_utils = sys.modules.get("torch.utils")
_orig_torch_utils_data = sys.modules.get("torch.utils.data")

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_nn
sys.modules["torch.optim"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()

import numpy as np
import pytest
import torch_compat

torch_compat.TORCH_AVAILABLE = True
torch_compat.torch = mock_torch
torch_compat.nn = mock_nn
torch_compat.get_device = MagicMock(return_value="cpu")

from base_controller import BaseController

# Restore torch modules
for _name, _orig in [
    ("torch", _orig_torch),
    ("torch.nn", _orig_torch_nn),
    ("torch.optim", _orig_torch_optim),
    ("torch.nn.functional", _orig_torch_nn_functional),
    ("torch.utils", _orig_torch_utils),
    ("torch.utils.data", _orig_torch_utils_data),
]:
    if _orig is not None:
        sys.modules[_name] = _orig
    else:
        sys.modules.pop(_name, None)


class ConcreteController(BaseController):
    """Minimal concrete controller for testing."""

    def control_step(self, system_state):
        return 0, {}

    def train_step(self):
        return {"loss": 0.0}

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

    def get_performance_summary(self):
        return {}


@pytest.mark.coverage_extra
class TestExtractStateFeatures:
    """Cover extract_state_features (lines 154-176)."""

    def test_extract_state_features_basic(self):
        ctrl = ConcreteController(state_dim=10, action_dim=5)

        # Create mock system state with required attributes
        system_state = MagicMock()
        system_state.power_output = 0.1
        system_state.current_density = 0.5
        system_state.health_metrics = MagicMock()
        system_state.health_metrics.overall_health_score = 0.85
        system_state.fused_measurement = MagicMock()
        system_state.fused_measurement.fusion_confidence = 0.9
        system_state.anomalies = []

        # Mock the feature engineer
        mock_fe = MagicMock()
        mock_fe.extract_features.return_value = {
            "feature_1": 50.0,
            "feature_2": 100.0,
            "feature_3": -30.0,
        }
        ctrl._feature_engineer = mock_fe

        result = ctrl.extract_state_features(system_state)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64 or result.dtype == np.float32
        # tanh output is always in [-1, 1]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_extract_state_features_with_anomalies(self):
        ctrl = ConcreteController(state_dim=10, action_dim=5)

        system_state = MagicMock()
        system_state.power_output = 0.05
        system_state.current_density = 0.01  # Very low to test max clamping
        system_state.health_metrics = MagicMock()
        system_state.health_metrics.overall_health_score = 0.3
        system_state.fused_measurement = MagicMock()
        system_state.fused_measurement.fusion_confidence = 0.4
        system_state.anomalies = [MagicMock()] * 5  # 5 anomalies

        mock_fe = MagicMock()
        mock_fe.extract_features.return_value = {
            "f1": float("nan"),
            "f2": float("inf"),
            "f3": float("-inf"),
            "f4": 0.0,
        }
        ctrl._feature_engineer = mock_fe

        result = ctrl.extract_state_features(system_state)

        assert isinstance(result, np.ndarray)
        # NaN/inf should be handled
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_extract_state_features_zero_current(self):
        """Test with zero current density to exercise max clamping."""
        ctrl = ConcreteController(state_dim=10, action_dim=5)

        system_state = MagicMock()
        system_state.power_output = 0.5
        system_state.current_density = 0.0  # Zero -> max(0.0, 0.01) = 0.01
        system_state.health_metrics = MagicMock()
        system_state.health_metrics.overall_health_score = 0.9
        system_state.fused_measurement = MagicMock()
        system_state.fused_measurement.fusion_confidence = 0.95
        system_state.anomalies = []

        mock_fe = MagicMock()
        mock_fe.extract_features.return_value = {"a": 1.0, "b": 2.0}
        ctrl._feature_engineer = mock_fe

        result = ctrl.extract_state_features(system_state)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_extract_state_features_lazy_imports_feature_engineer(self):
        """Test that feature_engineer is lazy-loaded correctly."""
        ctrl = ConcreteController(state_dim=10, action_dim=5)
        assert ctrl._feature_engineer is None

        system_state = MagicMock()
        system_state.power_output = 0.1
        system_state.current_density = 0.5
        system_state.health_metrics = MagicMock()
        system_state.health_metrics.overall_health_score = 0.8
        system_state.fused_measurement = MagicMock()
        system_state.fused_measurement.fusion_confidence = 0.9
        system_state.anomalies = []

        # Mock the ml_optimization module for lazy import
        mock_ml = types.ModuleType("ml_optimization")
        mock_fe_instance = MagicMock()
        mock_fe_instance.extract_features.return_value = {"x": 10.0}
        mock_ml.FeatureEngineer = MagicMock(return_value=mock_fe_instance)
        sys.modules["ml_optimization"] = mock_ml

        try:
            result = ctrl.extract_state_features(system_state)
            assert ctrl._feature_engineer is not None
            assert isinstance(result, np.ndarray)
        finally:
            del sys.modules["ml_optimization"]
            ctrl._feature_engineer = None
