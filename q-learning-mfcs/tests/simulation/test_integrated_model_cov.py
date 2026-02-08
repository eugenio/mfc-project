import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json
import pickle
import logging
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

# integrated_model.py uses relative imports, so we mock the package structure
# It imports from .config, .controller_models, .integrated_mfc_model, etc.
# We need to create a fake package and patch sys.modules

# Create mock modules for all relative imports
_mock_modules = {}
for mod_name in [
    "config", "config.config_manager",
    "controller_models", "controller_models.real_time_controller",
    "integrated_mfc_model",
    "ml", "ml.physics_ml_integration",
    "monitoring", "monitoring.dashboard_api",
    "sensing_models", "sensing_models.sensor_fusion",
    "validation", "validation.experimental_validation",
]:
    _mock_modules[mod_name] = MagicMock()

# Set up specific return values
_mock_modules["config.config_manager"].ConfigManager = MagicMock(return_value=MagicMock(
    load_config=MagicMock(return_value={"n_cells": 5, "dt": 0.1}),
    get_default_config=MagicMock(return_value={"n_cells": 5, "dt": 0.1, "simulation_hours": 24})
))
_mock_modules["integrated_mfc_model"].IntegratedMFCState = MagicMock
_mock_modules["ml.physics_ml_integration"].PhysicsMLConfig = MagicMock

# Patch all as submodules of a fake parent package
# The trick: integrated_model.py starts with 'from .xxx import ...'
# We need to make it think it's inside a package
# Instead, let's use importlib and patch

# Approach: create the module with exec after patching
import importlib
import types

# Create a fake parent package
fake_pkg = types.ModuleType("_fake_pkg")
fake_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "src")]
fake_pkg.__package__ = "_fake_pkg"
sys.modules["_fake_pkg"] = fake_pkg

# Set up all sub-modules under the fake package
for name, mock in _mock_modules.items():
    sys.modules[f"_fake_pkg.{name}"] = mock

# Now read the source and modify relative imports to absolute
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "integrated_model.py")
with open(src_path, "r") as f:
    source = f.read()

# Replace relative imports with absolute
source = source.replace("from .config.", "from _fake_pkg.config.")
source = source.replace("from .controller_models.", "from _fake_pkg.controller_models.")
source = source.replace("from .integrated_mfc_model", "from _fake_pkg.integrated_mfc_model")
source = source.replace("from .ml.", "from _fake_pkg.ml.")
source = source.replace("from .monitoring.", "from _fake_pkg.monitoring.")
source = source.replace("from .sensing_models.", "from _fake_pkg.sensing_models.")
source = source.replace("from .validation.", "from _fake_pkg.validation.")

# Execute modified source in a module
integrated_mod = types.ModuleType("integrated_model")
integrated_mod.__file__ = src_path
exec(compile(source, src_path, "exec"), integrated_mod.__dict__)

SystemPerformanceMetrics = integrated_mod.SystemPerformanceMetrics
IntegratedSystemState = integrated_mod.IntegratedSystemState
PhysicsMLBridge = integrated_mod.PhysicsMLBridge
ControlSystemBridge = integrated_mod.ControlSystemBridge
DataIntegrationPipeline = integrated_mod.DataIntegrationPipeline
IntegratedModelManager = integrated_mod.IntegratedModelManager


class TestSystemPerformanceMetrics:
    def test_defaults(self):
        m = SystemPerformanceMetrics()
        assert m.power_output == 0.0
        assert m.efficiency == 0.0
        assert m.hardware_reliability == 1.0

    def test_custom(self):
        m = SystemPerformanceMetrics(power_output=5.0, efficiency=0.8)
        assert m.power_output == 5.0


class TestIntegratedSystemState:
    def test_defaults(self):
        s = IntegratedSystemState()
        assert s.simulation_time == 0.0
        assert s.step_count == 0
        assert s.ph_level == 7.0
        assert s.physics_active is True

    def test_custom(self):
        s = IntegratedSystemState(voltage=0.5, current=0.1)
        assert s.voltage == 0.5


class TestPhysicsMLBridge:
    def test_init(self):
        b = PhysicsMLBridge(MagicMock(), MagicMock())
        assert b.sync_frequency == 10

    def test_sync_no_trigger(self):
        b = PhysicsMLBridge(MagicMock(), MagicMock())
        state = IntegratedSystemState(step_count=5)
        result = b.sync_models(state)
        assert result is state

    def test_sync_triggered(self):
        ml = MagicMock()
        ml.predict_next_state.return_value = {"optimal_flow_rate": 0.5, "substrate_feed_rate": 0.1}
        b = PhysicsMLBridge(MagicMock(), ml)
        state = IntegratedSystemState(step_count=10, voltage=0.5, current=0.1)
        state.metrics = SystemPerformanceMetrics(efficiency=0.5)
        result = b.sync_models(state)
        assert result.flow_rate == 0.5
        assert result.substrate_feed_rate == 0.1

    def test_extract_features(self):
        b = PhysicsMLBridge(MagicMock(), MagicMock())
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        state.metrics = SystemPerformanceMetrics(efficiency=0.8)
        f = b._extract_physics_features(state)
        assert "biofilm_thickness" in f
        assert "power_density" in f

    def test_apply_predictions_empty(self):
        b = PhysicsMLBridge(MagicMock(), MagicMock())
        state = IntegratedSystemState(flow_rate=0.3)
        result = b._apply_ml_predictions(state, {})
        assert result.flow_rate == 0.3


class TestControlSystemBridge:
    def test_init(self):
        b = ControlSystemBridge({"control": {}})
        assert isinstance(b.control_history, list)

    def test_init_failure(self):
        _mock_modules["controller_models.real_time_controller"].RealTimeController = MagicMock(side_effect=Exception("fail"))
        b = ControlSystemBridge({"control": {}})
        _mock_modules["controller_models.real_time_controller"].RealTimeController = MagicMock()

    def test_update_control(self):
        b = ControlSystemBridge({})
        b.controllers = {"test": MagicMock(compute_control_action=MagicMock(return_value={"flow_rate": 0.5}))}
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        result = b.update_control_systems(state)
        assert len(b.control_history) == 1

    def test_update_control_failure(self):
        ctrl = MagicMock()
        ctrl.compute_control_action.side_effect = Exception("fail")
        b = ControlSystemBridge({})
        b.controllers = {"test": ctrl}
        state = IntegratedSystemState()
        result = b.update_control_systems(state)

    def test_merge_outputs_empty(self):
        b = ControlSystemBridge({})
        state = IntegratedSystemState(flow_rate=0.5)
        result = b._merge_control_outputs(state, {})
        assert result.flow_rate == 0.5

    def test_merge_outputs(self):
        b = ControlSystemBridge({})
        state = IntegratedSystemState()
        outputs = {"c1": {"flow_rate": 0.5}, "c2": {"flow_rate": 0.7, "substrate_feed_rate": 0.4}}
        result = b._merge_control_outputs(state, outputs)
        assert 0.4 < result.flow_rate < 0.8


class TestDataIntegrationPipeline:
    def test_init(self):
        p = DataIntegrationPipeline({})
        assert isinstance(p.data_buffer, list)

    def test_process_no_validation_no_dashboard(self):
        p = DataIntegrationPipeline({})
        p.validation_database = None
        p.dashboard_api = None
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        result = p.process_data_flow(state)
        assert len(p.data_buffer) == 1

    def test_process_with_validation(self):
        p = DataIntegrationPipeline({})
        p.validation_database = MagicMock()
        p.validation_database.get_matching_conditions.return_value = {"power_density": 0.05}
        p.dashboard_api = None
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        result = p.process_data_flow(state)
        assert result.metrics.prediction_accuracy >= 0

    def test_process_validation_no_data(self):
        p = DataIntegrationPipeline({})
        p.validation_database = MagicMock()
        p.validation_database.get_matching_conditions.return_value = None
        p.dashboard_api = None
        state = IntegratedSystemState()
        result = p.process_data_flow(state)
        assert result.metrics.prediction_accuracy == 0.5

    def test_process_validation_error(self):
        p = DataIntegrationPipeline({})
        p.validation_database = MagicMock()
        p.validation_database.get_matching_conditions.side_effect = Exception("fail")
        p.dashboard_api = None
        state = IntegratedSystemState()
        result = p.process_data_flow(state)

    def test_process_dashboard_update(self):
        p = DataIntegrationPipeline({})
        p.validation_database = None
        p.dashboard_api = MagicMock()
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        p.process_data_flow(state)

    def test_process_dashboard_error(self):
        p = DataIntegrationPipeline({})
        p.validation_database = None
        p.dashboard_api = MagicMock()
        p.dashboard_api.update_real_time_data.side_effect = Exception("fail")
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        p.process_data_flow(state)

    def test_buffer_cleanup(self):
        p = DataIntegrationPipeline({})
        p.validation_database = None
        p.dashboard_api = None
        p.data_buffer = [{}] * 1001
        state = IntegratedSystemState()
        p.process_data_flow(state)
        assert len(p.data_buffer) <= 501

    def test_serialize_state(self):
        p = DataIntegrationPipeline({})
        state = IntegratedSystemState(voltage=0.5, current=0.1)
        s = p._serialize_state(state)
        assert "voltage" in s
        assert "power" in s


class TestMakeSerializable:
    def setup_method(self):
        self.mgr_cls = IntegratedModelManager

    def test_dict(self):
        m = object.__new__(self.mgr_cls)
        assert m._make_serializable({"k": "v"}) == {"k": "v"}

    def test_list(self):
        m = object.__new__(self.mgr_cls)
        assert m._make_serializable([1, 2]) == [1, 2]

    def test_tuple(self):
        m = object.__new__(self.mgr_cls)
        assert m._make_serializable((1, 2)) == [1, 2]

    def test_datetime(self):
        m = object.__new__(self.mgr_cls)
        obj = MagicMock()
        now = datetime.now()
        obj.__dict__ = {"ts": now, "val": 42}
        result = m._make_serializable(obj)
        assert result["ts"] == now.isoformat()

    def test_ndarray(self):
        m = object.__new__(self.mgr_cls)
        obj = MagicMock()
        obj.__dict__ = {"arr": np.array([1, 2, 3])}
        result = m._make_serializable(obj)
        assert result["arr"] == [1, 2, 3]

    def test_other_obj(self):
        m = object.__new__(self.mgr_cls)
        obj = MagicMock()
        inner = object()
        obj.__dict__ = {"inner": inner}
        result = m._make_serializable(obj)
        assert isinstance(result["inner"], str)

    def test_primitive(self):
        m = object.__new__(self.mgr_cls)
        assert m._make_serializable(42) == 42
        assert m._make_serializable("s") == "s"
        assert m._make_serializable(None) is None
        assert m._make_serializable(True) is True
