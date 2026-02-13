"""Coverage tests for integrated_model.py - IntegratedModelManager and main().

Covers: IntegratedModelManager __init__, step, _step_physics, _step_sensors,
_calculate_integrated_metrics, run_simulation, save_checkpoint, load_checkpoint,
export_results, PhysicsMLBridge sync_models, and main().
"""
import importlib
import os
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
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
_mock_modules["integrated_mfc_model"].IntegratedMFCModel = MagicMock
_mock_modules["ml.physics_ml_integration"].PhysicsMLConfig = MagicMock
_mock_modules["ml.physics_ml_integration"].PhysicsMLIntegrator = MagicMock

# Create a fake parent package
fake_pkg = types.ModuleType("_fake_pkg3")
fake_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "src")]
fake_pkg.__package__ = "_fake_pkg3"
sys.modules["_fake_pkg3"] = fake_pkg

for name, mock in _mock_modules.items():
    sys.modules[f"_fake_pkg3.{name}"] = mock

src_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "integrated_model.py",
)
with open(src_path) as f:
    source = f.read()

source = source.replace("from .config.", "from _fake_pkg3.config.")
source = source.replace(
    "from .controller_models.", "from _fake_pkg3.controller_models.",
)
source = source.replace(
    "from .integrated_mfc_model", "from _fake_pkg3.integrated_mfc_model",
)
source = source.replace("from .ml.", "from _fake_pkg3.ml.")
source = source.replace("from .monitoring.", "from _fake_pkg3.monitoring.")
source = source.replace("from .sensing_models.", "from _fake_pkg3.sensing_models.")
source = source.replace("from .validation.", "from _fake_pkg3.validation.")

_mod = types.ModuleType("integrated_model_cov3")
_mod.__file__ = src_path
exec(compile(source, src_path, "exec"), _mod.__dict__)

IntegratedModelManager = _mod.IntegratedModelManager
IntegratedSystemState = _mod.IntegratedSystemState
SystemPerformanceMetrics = _mod.SystemPerformanceMetrics
PhysicsMLBridge = _mod.PhysicsMLBridge
class TestIntegratedModelManagerInit:
    def test_default_init(self):
        mgr = IntegratedModelManager()
        assert mgr.system_state is not None
        assert mgr.performance_history == []
        assert mgr.simulation_results == {}

    def test_init_with_config_path(self):
        mgr = IntegratedModelManager(config_path="/some/path.yaml")
        mgr.config_manager.load_config.assert_called_once_with("/some/path.yaml")

    def test_init_core_failure(self):
        old_cls = _mod.IntegratedMFCModel
        _mod.IntegratedMFCModel = MagicMock(side_effect=Exception("fail"))
        try:
            with pytest.raises(Exception, match="fail"):
                IntegratedModelManager()
        finally:
            _mod.IntegratedMFCModel = old_cls

class TestIntegratedModelManagerStep:
    def _make_manager(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.system_state.current = 0.1
        mgr.system_state.substrate_concentration = 20.0
        mgr.system_state.biofilm_thickness = 0.1
        mgr.system_state.flow_rate = 0.01
        mgr.system_state.temperature = 298.0
        mgr.system_state.ph_level = 7.0
        return mgr

    def test_step_all_active(self):
        mgr = self._make_manager()
        mgr.system_state.physics_active = True
        mgr.system_state.ml_active = True
        mgr.system_state.control_active = True
        mgr.system_state.sensors_active = True
        result = mgr.step()
        assert result.step_count == 1
        assert result.simulation_time > 0

    def test_step_none_active(self):
        mgr = self._make_manager()
        mgr.system_state.physics_active = False
        mgr.system_state.ml_active = False
        mgr.system_state.control_active = False
        mgr.system_state.sensors_active = False
        result = mgr.step()
        assert result.step_count == 1

    def test_step_physics_only(self):
        mgr = self._make_manager()
        mgr.system_state.physics_active = True
        mgr.system_state.ml_active = False
        mgr.system_state.control_active = False
        mgr.system_state.sensors_active = False
        result = mgr.step()
        assert result.step_count == 1

    def test_step_sensors_only(self):
        mgr = self._make_manager()
        mgr.system_state.physics_active = False
        mgr.system_state.ml_active = False
        mgr.system_state.control_active = False
        mgr.system_state.sensors_active = True
        result = mgr.step()
        assert result.step_count == 1

    def test_step_records_performance(self):
        mgr = self._make_manager()
        mgr.step()
        assert len(mgr.performance_history) == 1
        assert 'step' in mgr.performance_history[0]
        assert 'metrics' in mgr.performance_history[0]

class TestStepPhysics:
    def test_step_physics_success(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.flow_rate = 0.01
        mgr.system_state.substrate_concentration = 20.0
        mock_state = MagicMock()
        mgr.physics_model.step_integrated_dynamics.return_value = (
            mock_state, 1.0, False,
            {'voltage': 0.6, 'current': 0.2, 'biofilm_thickness': 0.15}
        )
        result = mgr._step_physics()
        assert result.voltage == 0.6
        assert result.current == 0.2
        assert result.biofilm_thickness == 0.15

    def test_step_physics_failure(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.physics_model.step_integrated_dynamics.side_effect = Exception("fail")
        result = mgr._step_physics()
        assert result.voltage == 0.5  # unchanged

class TestStepSensors:
    def test_step_sensors_success(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.system_state.current = 0.1
        mgr.sensor_fusion.fuse_sensor_data.return_value = {
            'impedance': 100.0, 'biofilm_mass': 0.05,
        }
        result = mgr._step_sensors()
        assert result.impedance == 100.0
        assert result.biofilm_mass == 0.05

    def test_step_sensors_failure(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.impedance = float('inf')
        mgr.sensor_fusion.fuse_sensor_data.side_effect = Exception("fail")
        result = mgr._step_sensors()
        assert result.impedance == float('inf')  # unchanged

class TestCalculateIntegratedMetrics:
    def _make_manager(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.system_state.current = 0.1
        mgr.system_state.substrate_concentration = 20.0
        mgr.system_state.biofilm_thickness = 0.1
        return mgr

    def test_basic_metrics(self):
        mgr = self._make_manager()
        metrics = mgr._calculate_integrated_metrics()
        assert metrics.power_output == pytest.approx(0.05)
        assert 0.0 <= metrics.efficiency <= 1.0
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_zero_substrate(self):
        mgr = self._make_manager()
        mgr.system_state.substrate_concentration = 0.0
        metrics = mgr._calculate_integrated_metrics()
        assert metrics.efficiency == 0.0

    def test_with_initial_substrate_attr(self):
        mgr = self._make_manager()
        mgr.physics_model.initial_substrate_concentration = 50.0
        mgr.system_state.substrate_concentration = 25.0
        metrics = mgr._calculate_integrated_metrics()
        assert metrics.substrate_utilization == pytest.approx(0.5)

    def test_control_stability_with_history(self):
        mgr = self._make_manager()
        mgr.control_bridge.control_history = [
            {'outputs': {'real_time': {'flow_rate': 0.01 + i * 0.001}}}
            for i in range(15)
        ]
        metrics = mgr._calculate_integrated_metrics()
        assert 0.0 <= metrics.control_stability <= 1.0

    def test_control_stability_no_flow_rate(self):
        mgr = self._make_manager()
        mgr.control_bridge.control_history = [
            {'outputs': {'real_time': {}}}
            for _ in range(15)
        ]
        metrics = mgr._calculate_integrated_metrics()
        assert metrics.control_stability >= 0.0

    def test_control_stability_short_history(self):
        mgr = self._make_manager()
        mgr.control_bridge.control_history = [
            {'outputs': {'real_time': {'flow_rate': 0.01}}}
            for _ in range(5)
        ]
        metrics = mgr._calculate_integrated_metrics()
        assert metrics.control_stability == 0.0

class TestRunSimulation:
    def test_run_short(self):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.system_state.current = 0.1
        mgr.system_state.substrate_concentration = 20.0
        mgr.system_state.biofilm_thickness = 0.1
        with patch.object(mgr, 'save_checkpoint'):
            results = mgr.run_simulation(steps=5, save_interval=2)
        assert results['total_steps'] == 5
        assert 'average_performance' in results
        assert 'peak_performance' in results
        assert 'final_state' in results

    def test_run_with_exception(self):
        mgr = IntegratedModelManager()
        with patch.object(mgr, 'step', side_effect=Exception("fail")):
            with pytest.raises(Exception, match="fail"):
                mgr.run_simulation(steps=5)

class TestSaveLoadCheckpoint:
    def test_save_checkpoint(self, tmp_path):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.performance_history = [{'step': 1}]
        filepath = str(tmp_path / "checkpoint.pkl")
        mgr.save_checkpoint(filepath)
        assert os.path.exists(filepath)

    def test_load_checkpoint(self, tmp_path):
        import pickle
        filepath = str(tmp_path / "checkpoint.pkl")
        state = IntegratedSystemState()
        state.step_count = 42
        data = {
            'system_state': state,
            'performance_history': [{'step': 42}],
            'config': {},
            'timestamp': datetime.now().isoformat(),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        mgr = IntegratedModelManager()
        mgr.load_checkpoint(filepath)
        assert mgr.system_state.step_count == 42
        assert len(mgr.performance_history) == 1

class TestExportResults:
    def test_export_results(self, tmp_path):
        mgr = IntegratedModelManager()
        mgr.system_state = IntegratedSystemState()
        mgr.system_state.voltage = 0.5
        mgr.system_state.current = 0.1
        mgr.simulation_results = {
            'total_steps': 10,
            'average_performance': 0.5,
        }
        mgr.performance_history = [
            {'step': 1, 'timestamp': datetime.now(),
             'metrics': SystemPerformanceMetrics()},
        ]
        output_dir = str(tmp_path / "results")
        mgr.export_results(output_dir)
        assert os.path.exists(os.path.join(output_dir, "simulation_results.json"))
        assert os.path.exists(os.path.join(output_dir, "performance_history.json"))
        assert os.path.exists(os.path.join(output_dir, "final_state.json"))

class TestPhysicsMLBridge:
    def test_sync_models_below_frequency(self):
        mock_phys = MagicMock()
        mock_ml = MagicMock()
        bridge = PhysicsMLBridge(mock_phys, mock_ml)
        state = IntegratedSystemState()
        state.step_count = 5
        bridge.last_sync_step = 0
        bridge.sync_frequency = 10
        result = bridge.sync_models(state)
        mock_ml.update_physics_context.assert_not_called()

    def test_sync_models_at_frequency(self):
        mock_phys = MagicMock()
        mock_ml = MagicMock()
        mock_ml.predict_next_state.return_value = {
            'optimal_flow_rate': 0.02,
            'substrate_feed_rate': 0.5,
        }
        bridge = PhysicsMLBridge(mock_phys, mock_ml)
        state = IntegratedSystemState()
        state.step_count = 10
        state.voltage = 0.5
        state.current = 0.1
        state.biofilm_thickness = 0.1
        state.substrate_concentration = 20.0
        bridge.last_sync_step = 0
        bridge.sync_frequency = 10
        result = bridge.sync_models(state)
        mock_ml.update_physics_context.assert_called_once()
        assert result.flow_rate == 0.02
        assert result.substrate_feed_rate == 0.5

    def test_sync_models_no_predictions(self):
        mock_phys = MagicMock()
        mock_ml = MagicMock()
        mock_ml.predict_next_state.return_value = {}
        bridge = PhysicsMLBridge(mock_phys, mock_ml)
        state = IntegratedSystemState()
        state.step_count = 10
        state.flow_rate = 0.01
        bridge.last_sync_step = 0
        result = bridge.sync_models(state)
        assert result.flow_rate == 0.01

    def test_extract_physics_features(self):
        mock_phys = MagicMock()
        mock_ml = MagicMock()
        bridge = PhysicsMLBridge(mock_phys, mock_ml)
        state = IntegratedSystemState()
        state.voltage = 0.5
        state.current = 0.1
        state.biofilm_thickness = 0.1
        state.substrate_concentration = 20.0
        state.ph_level = 7.0
        state.temperature = 298.0
        features = bridge._extract_physics_features(state)
        assert features['power_density'] == pytest.approx(0.05)
        assert features['biofilm_thickness'] == 0.1

class TestMain:
    def test_main(self):
        with patch.object(IntegratedModelManager, '__init__', return_value=None):
            mgr_mock = MagicMock()
            mgr_mock.run_simulation.return_value = {
                'average_performance': 0.5,
                'peak_performance': 0.8,
                'final_metrics': MagicMock(overall_score=0.7),
            }
            with patch.object(
                _mod, 'IntegratedModelManager', return_value=mgr_mock
            ):
                _mod.main()
            mgr_mock.run_simulation.assert_called_once()
            mgr_mock.export_results.assert_called_once()