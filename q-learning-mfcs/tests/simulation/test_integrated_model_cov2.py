"""Coverage tests for integrated_model.py - lines 158-640."""
import importlib
import os
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Create mock modules for all relative imports
_mock_modules = {}
for mod_name in [
    "config", "config.config_manager",
    "controller_models", "controller_models.real_time_controller",
    "integrated_mfc_model",
    "ml", "ml.physics_ml_integration",
    "monitoring", "monitoring.dashboard_api", "monitoring.ssl_config",
    "sensing_models", "sensing_models.sensor_fusion",
    "validation", "validation.experimental_validation",
]:
    _mock_modules[mod_name] = MagicMock()

_mock_modules["config.config_manager"].ConfigManager = MagicMock(
    return_value=MagicMock(
        load_config=MagicMock(return_value={"n_cells": 5, "dt": 0.1}),
        get_default_config=MagicMock(
            return_value={"n_cells": 5, "dt": 0.1, "simulation_hours": 24},
        ),
    ),
)
_mock_modules["integrated_mfc_model"].IntegratedMFCState = MagicMock
_mock_modules["ml.physics_ml_integration"].PhysicsMLConfig = MagicMock

# Create a fake parent package
fake_pkg = types.ModuleType("_fake_pkg2")
fake_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "src")]
fake_pkg.__package__ = "_fake_pkg2"
sys.modules["_fake_pkg2"] = fake_pkg

for name, mock in _mock_modules.items():
    sys.modules[f"_fake_pkg2.{name}"] = mock

src_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "integrated_model.py",
)
with open(src_path) as f:
    source = f.read()

source = source.replace("from .config.", "from _fake_pkg2.config.")
source = source.replace(
    "from .controller_models.", "from _fake_pkg2.controller_models.",
)
source = source.replace(
    "from .integrated_mfc_model", "from _fake_pkg2.integrated_mfc_model",
)
source = source.replace("from .ml.", "from _fake_pkg2.ml.")
source = source.replace("from .monitoring.", "from _fake_pkg2.monitoring.")
source = source.replace("from .sensing_models.", "from _fake_pkg2.sensing_models.")
source = source.replace("from .validation.", "from _fake_pkg2.validation.")

_mod = types.ModuleType("integrated_model_cov2")
_mod.__file__ = src_path
exec(compile(source, src_path, "exec"), _mod.__dict__)

ControlSystemBridge = _mod.ControlSystemBridge
DataIntegrationPipeline = _mod.DataIntegrationPipeline
IntegratedSystemState = _mod.IntegratedSystemState
IntegratedModelManager = _mod.IntegratedModelManager


@pytest.mark.coverage_extra
class TestControlSystemBridge:
    def test_init_success(self):
        bridge = ControlSystemBridge({'control': {}})
        assert hasattr(bridge, 'controllers')

    def test_init_failure(self):
        # The exec'd module binds RealTimeController at load-time.
        # We need to patch the class in the module namespace itself.
        old = _mod.RealTimeController
        _mod.RealTimeController = MagicMock(side_effect=Exception("fail"))
        try:
            bridge = ControlSystemBridge({'control': {}})
            assert len(bridge.controllers) == 0
        finally:
            _mod.RealTimeController = old

    def test_update_control_systems(self):
        bridge = ControlSystemBridge({'control': {}})
        state = IntegratedSystemState()
        state.voltage = 0.5
        state.current = 0.1
        state.flow_rate = 0.01
        state.substrate_concentration = 20.0
        state.biofilm_thickness = 0.1
        result = bridge.update_control_systems(state)
        assert result.timestamp is not None

    def test_merge_control_outputs(self):
        bridge = ControlSystemBridge({'control': {}})
        state = IntegratedSystemState()
        state.flow_rate = 0.01
        state.substrate_feed_rate = 0.5
        outputs = {
            'ctrl1': {'flow_rate': 0.02, 'substrate_feed_rate': 0.6},
            'ctrl2': {'flow_rate': 0.03},
        }
        result = bridge._merge_control_outputs(state, outputs)
        assert result.flow_rate == pytest.approx(0.025)

    def test_merge_empty_outputs(self):
        bridge = ControlSystemBridge({'control': {}})
        state = IntegratedSystemState()
        state.flow_rate = 0.01
        state.substrate_feed_rate = 0.5
        result = bridge._merge_control_outputs(state, {})
        assert result.flow_rate == 0.01


@pytest.mark.coverage_extra
class TestDataIntegrationPipeline:
    def test_init_with_failures(self):
        old_db = _mod.MFCExperimentalDatabase
        old_api = _mod.DashboardAPI
        _mod.MFCExperimentalDatabase = MagicMock(
            side_effect=Exception("no db"),
        )
        _mod.DashboardAPI = MagicMock(
            side_effect=Exception("no api"),
        )
        try:
            pipeline = DataIntegrationPipeline({})
            assert pipeline.validation_database is None
            assert pipeline.dashboard_api is None
        finally:
            _mod.MFCExperimentalDatabase = old_db
            _mod.DashboardAPI = old_api

    def test_process_data_flow(self):
        pipeline = DataIntegrationPipeline({})
        pipeline.validation_database = None
        pipeline.dashboard_api = None
        state = IntegratedSystemState()
        state.voltage = 0.5
        state.current = 0.1
        pipeline.process_data_flow(state)
        assert len(pipeline.data_buffer) == 1

    def test_buffer_cleanup(self):
        pipeline = DataIntegrationPipeline({})
        pipeline.validation_database = None
        pipeline.dashboard_api = None
        pipeline.data_buffer = [{}] * 1001
        state = IntegratedSystemState()
        pipeline.process_data_flow(state)
        # 1001 + 1 (new) = 1002 > 1000, trimmed to last 500
        assert len(pipeline.data_buffer) == 500

    def test_validate_against_experiments(self):
        pipeline = DataIntegrationPipeline({})
        mock_db = MagicMock()
        mock_db.get_matching_conditions.return_value = {
            'power_density': 0.05,
        }
        pipeline.validation_database = mock_db
        state = IntegratedSystemState()
        state.voltage = 0.5
        state.current = 0.1
        state.substrate_concentration = 20.0
        state.temperature = 298.0
        state.ph_level = 7.0
        score = pipeline._validate_against_experiments(state)
        assert 0.0 <= score <= 1.0

    def test_validate_no_data(self):
        pipeline = DataIntegrationPipeline({})
        mock_db = MagicMock()
        mock_db.get_matching_conditions.return_value = None
        pipeline.validation_database = mock_db
        state = IntegratedSystemState()
        score = pipeline._validate_against_experiments(state)
        assert score == 0.5

    def test_validate_exception(self):
        pipeline = DataIntegrationPipeline({})
        mock_db = MagicMock()
        mock_db.get_matching_conditions.side_effect = Exception("boom")
        pipeline.validation_database = mock_db
        state = IntegratedSystemState()
        score = pipeline._validate_against_experiments(state)
        assert score == 0.5

    def test_update_dashboard(self):
        pipeline = DataIntegrationPipeline({})
        mock_api = MagicMock()
        pipeline.dashboard_api = mock_api
        state = IntegratedSystemState()
        state.voltage = 0.5
        state.current = 0.1
        pipeline._update_dashboard(state)
        mock_api.update_real_time_data.assert_called_once()

    def test_update_dashboard_failure(self):
        pipeline = DataIntegrationPipeline({})
        mock_api = MagicMock()
        mock_api.update_real_time_data.side_effect = Exception("fail")
        pipeline.dashboard_api = mock_api
        state = IntegratedSystemState()
        pipeline._update_dashboard(state)


@pytest.mark.coverage_extra
class TestMakeSerializable:
    def test_basic_types(self):
        mgr = object.__new__(IntegratedModelManager)
        assert mgr._make_serializable(42) == 42
        assert mgr._make_serializable("hello") == "hello"
        assert mgr._make_serializable(None) is None

    def test_datetime(self):
        mgr = object.__new__(IntegratedModelManager)
        dt = datetime(2025, 1, 1)
        obj = MagicMock()
        obj.__dict__ = {'ts': dt}
        result = mgr._make_serializable(obj)
        assert '2025' in result['ts']

    def test_ndarray(self):
        mgr = object.__new__(IntegratedModelManager)
        obj = MagicMock()
        obj.__dict__ = {'arr': np.array([1.0, 2.0])}
        result = mgr._make_serializable(obj)
        assert result['arr'] == [1.0, 2.0]

    def test_nested_dict(self):
        mgr = object.__new__(IntegratedModelManager)
        obj = MagicMock()
        obj.__dict__ = {'d': {'key': 'val'}}
        result = mgr._make_serializable(obj)
        assert result['d'] == {'key': 'val'}

    def test_list_and_tuple(self):
        mgr = object.__new__(IntegratedModelManager)
        result = mgr._make_serializable([1, 2, 3])
        assert result == [1, 2, 3]
        result = mgr._make_serializable((4, 5))
        assert result == [4, 5]

    def test_fallback_str(self):
        mgr = object.__new__(IntegratedModelManager)
        obj = MagicMock()
        obj.__dict__ = {'x': object()}
        result = mgr._make_serializable(obj)
        assert isinstance(result['x'], str)
