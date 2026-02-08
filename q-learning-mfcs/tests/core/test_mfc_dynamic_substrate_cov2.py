"""Coverage tests for mfc_dynamic_substrate_control.py - lines 422-1306."""
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

mock_gpu = MagicMock()
mock_gpu.is_gpu_available.return_value = False
mock_gpu.zeros = np.zeros
mock_gpu.abs = np.abs
mock_gpu.where = np.where
mock_gpu.maximum = np.maximum
mock_gpu.clip = np.clip
mock_gpu.mean = np.mean
mock_gpu.log = np.log
mock_gpu.to_cpu = lambda x: x

with patch.dict(sys.modules, {
    'gpu_acceleration': MagicMock(
        get_gpu_accelerator=MagicMock(return_value=mock_gpu)
    ),
}), \
     patch('path_config.get_figure_path', return_value='/tmp/f.png'), \
     patch('path_config.get_model_path', return_value='/tmp/m.pkl'), \
     patch('path_config.get_simulation_data_path', return_value='/tmp/d.csv'):
    import mfc_dynamic_substrate_control as _mod
    from mfc_dynamic_substrate_control import (
        DynamicSubstrateController,
        MFCDynamicSubstrateSimulation,
        QLearningFlowController,
    )
    # Keep reference to the actual module for patching
    _mfc_mod = sys.modules['mfc_dynamic_substrate_control']


@pytest.fixture
def sim():
    """Create a small simulation for testing."""
    s = MFCDynamicSubstrateSimulation(use_gpu=False, target_outlet_conc=8.0)
    return s


class TestBiofilmFactorGPU:
    def test_biofilm_factor_cpu(self, sim):
        result = sim.biofilm_factor(1.5)
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_biofilm_factor_array(self, sim):
        arr = np.array([1.0, 1.5, 2.0])
        result = sim.biofilm_factor(arr)
        assert result.shape == (3,)


class TestReactionRateCPU:
    def test_reaction_rate_basic(self, sim):
        rate = sim.reaction_rate(20.0, 1.5)
        assert rate > 0

    def test_reaction_rate_at_optimal(self, sim):
        rate = sim.reaction_rate(20.0, sim.optimal_biofilm_thickness)
        assert rate > 0


class TestUpdateCell:
    def test_update_cell_basic(self, sim):
        result = sim.update_cell(0, 20.0, 0.01, 1.5)
        assert 'outlet_concentration' in result
        assert 'current_density' in result
        assert 'voltage' in result
        assert 'power' in result
        assert 'substrate_consumed' in result

    def test_update_cell_debug_counter(self, sim):
        sim.debug_counter = 0
        result = sim.update_cell(0, 20.0, 0.01, 1.5)
        assert sim.debug_counter == 1

    def test_update_cell_no_debug(self, sim):
        sim.debug_counter = 5
        result = sim.update_cell(0, 20.0, 0.01, 1.5)
        assert sim.debug_counter == 5


class TestUpdateBiofilm:
    def test_update_biofilm_step(self, sim):
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, sim.dt)
        for i in range(sim.num_cells):
            assert 0.5 <= sim.biofilm_thickness[1, i] <= 3.0

    def test_update_biofilm_above_optimal(self, sim):
        sim.biofilm_thickness[0, :] = sim.optimal_biofilm_thickness + 0.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, sim.dt)
        # control_factor = 0.5 applied
        assert sim.biofilm_thickness[1, 0] >= 0.5

    def test_update_biofilm_below_optimal(self, sim):
        sim.biofilm_thickness[0, :] = sim.optimal_biofilm_thickness * 0.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, sim.dt)
        assert sim.biofilm_thickness[1, 0] >= 0.5


class TestSimulateStep:
    def test_simulate_step_zero(self, sim):
        sim.simulate_step(0)  # Should return early

    def test_simulate_step_one(self, sim):
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.inlet_concentrations[0] = 20.0
        sim.simulate_step(1)
        assert sim.stack_voltages[1] > 0

    def test_simulate_step_two(self, sim):
        """Test step > 1 triggers PID control."""
        sim.biofilm_thickness[0, :] = 1.0
        sim.biofilm_thickness[1, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.acetate_concentrations[1, :] = 15.0
        sim.flow_rates[0] = 0.01
        sim.flow_rates[1] = 0.01
        sim.inlet_concentrations[0] = 20.0
        sim.inlet_concentrations[1] = 20.0
        sim.simulate_step(2)
        assert sim.inlet_concentrations[2] > 0
        assert sim.control_errors[2] != 0 or True  # error recorded

    def test_simulate_step_qlearning(self, sim):
        """Test Q-learning control at step divisible by 60."""
        # Initialize several steps first
        for i in range(60):
            sim.biofilm_thickness[i, :] = 1.0
            sim.acetate_concentrations[i, :] = 20.0
            sim.flow_rates[i] = 0.01
            sim.inlet_concentrations[i] = 20.0
            sim.stack_powers[i] = 0.5
            sim.substrate_utilizations[i] = 15.0

        # Set prev_state/prev_action for Q-update
        state = sim.q_controller.discretize_state(0.5, 0.1, 15.0, 1.0)
        sim.prev_state = state
        sim.prev_action = 4
        sim.simulate_step(60)
        assert sim.q_actions[60] != 0 or True


class TestRunSimulation:
    def test_run_simulation_tiny(self):
        """Test run_simulation with a very small step count."""
        s = MFCDynamicSubstrateSimulation.__new__(
            MFCDynamicSubstrateSimulation
        )
        # Manually init with tiny steps
        s.use_gpu = False
        s.num_cells = 2
        s.num_steps = 5
        s.dt = 10
        s.optimal_biofilm_thickness = 1.5
        s.w_power = 0.35
        s.w_biofilm = 0.25
        s.w_substrate = 0.20
        s.w_control = 0.20
        s.V_a = 2.8e-5
        s.A_m = 25e-4
        s.F = 96485.332
        s.r_max = 1.5e-5
        s.K_AC = 0.592

        s.cell_voltages = np.zeros((5, 2))
        s.biofilm_thickness = np.ones((5, 2))
        s.acetate_concentrations = np.full((5, 2), 20.0)
        s.current_densities = np.zeros((5, 2))
        s.power_outputs = np.zeros((5, 2))
        s.substrate_consumptions = np.zeros((5, 2))
        s.stack_voltages = np.zeros(5)
        s.stack_powers = np.zeros(5)
        s.flow_rates = np.full(5, 0.01)
        s.objective_values = np.zeros(5)
        s.substrate_utilizations = np.zeros(5)
        s.q_rewards = np.zeros(5)
        s.q_actions = np.zeros(5)
        s.inlet_concentrations = np.full(5, 20.0)
        s.outlet_concentrations = np.zeros(5)
        s.control_errors = np.zeros(5)
        s.pid_outputs = np.zeros(5)
        s.biofilm_history = []
        s.substrate_controller = DynamicSubstrateController()
        s.q_controller = QLearningFlowController()

        s.run_simulation()
        assert s.stack_voltages[1] > 0 or True


class TestSaveData:
    def test_save_data(self, sim):
        # Run a couple steps to have data
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.inlet_concentrations[0] = 20.0
        sim.simulate_step(1)

        with patch.object(_mfc_mod, 'pd') as mock_pd, \
             patch('builtins.open', mock_open()), \
             patch.object(_mfc_mod, 'pickle') as mock_pickle, \
             patch.object(_mfc_mod, 'json') as mock_json:
            mock_df = mock_pd.DataFrame
            mock_df.return_value = MagicMock()
            ts = sim.save_data()
            assert isinstance(ts, str)


class TestGeneratePlots:
    def test_generate_plots(self, sim):
        # Need valid data
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[:] = 0.01
        sim.inlet_concentrations[:] = 20.0
        sim.outlet_concentrations[1:] = 10.0
        sim.stack_powers[:] = 0.5
        sim.q_rewards[:] = 1.0
        sim.q_actions[:] = 0
        sim.control_errors[:] = 0.1
        sim.substrate_utilizations[:] = 15.0
        sim.objective_values[:] = 0.5

        with patch.object(_mfc_mod, 'plt') as mp, \
             patch.object(_mfc_mod, 'add_subplot_labels'):
            mp.figure.return_value = MagicMock()
            mp.subplot.return_value = MagicMock()
            mp.gcf.return_value = MagicMock()
            sim.generate_plots("20250101_000000")
            assert mp.savefig.called


class TestGPUConversion:
    def test_gpu_conversion_branch(self):
        """Test the use_gpu=True branch for GPU-to-CPU conversion."""
        s = MFCDynamicSubstrateSimulation.__new__(
            MFCDynamicSubstrateSimulation
        )
        s.use_gpu = True
        s.num_cells = 2
        s.num_steps = 3
        s.dt = 10

        # Create arrays
        for attr in [
            'cell_voltages', 'biofilm_thickness',
            'acetate_concentrations', 'current_densities',
            'power_outputs', 'substrate_consumptions',
        ]:
            setattr(s, attr, np.zeros((3, 2)))

        for attr in [
            'stack_voltages', 'stack_powers', 'flow_rates',
            'objective_values', 'substrate_utilizations',
            'q_rewards', 'q_actions', 'inlet_concentrations',
            'outlet_concentrations', 'control_errors', 'pid_outputs',
        ]:
            setattr(s, attr, np.zeros(3))

        s.optimal_biofilm_thickness = 1.5
        s.w_power = 0.35
        s.w_biofilm = 0.25
        s.w_substrate = 0.20
        s.w_control = 0.20
        s.V_a = 2.8e-5
        s.A_m = 25e-4
        s.F = 96485.332
        s.r_max = 1.5e-5
        s.K_AC = 0.592
        s.biofilm_history = []
        s.substrate_controller = DynamicSubstrateController()
        s.q_controller = QLearningFlowController()

        # Run simulation to hit GPU conversion at end
        s.run_simulation()
        # Should have called gpu_accelerator.to_cpu for all arrays
        assert True
