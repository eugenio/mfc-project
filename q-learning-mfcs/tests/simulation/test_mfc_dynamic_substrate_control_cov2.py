"""Coverage tests for mfc_dynamic_substrate_control.py."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Mock gpu_acceleration and path_config before importing
mock_gpu = MagicMock()
mock_gpu.get_gpu_accelerator.return_value = MagicMock(
    is_gpu_available=MagicMock(return_value=False),
    zeros=np.zeros,
)
sys.modules["gpu_acceleration"] = mock_gpu

mock_pc = MagicMock()
mock_pc.get_figure_path = MagicMock(return_value="/tmp/fig.png")
mock_pc.get_simulation_data_path = MagicMock(return_value="/tmp/data.csv")
mock_pc.get_model_path = MagicMock(return_value="/tmp/model.pkl")

if "path_config" not in sys.modules:
    sys.modules["path_config"] = mock_pc

for _stale in ["mfc_dynamic_substrate_control"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

from mfc_dynamic_substrate_control import (
    DynamicSubstrateController,
    QLearningFlowController,
    MFCDynamicSubstrateSimulation,
    add_subplot_labels,
)


@pytest.mark.coverage_extra
class TestDynamicSubstrateController:
    def test_init(self):
        c = DynamicSubstrateController()
        assert c.target_outlet_conc == 8.0
        assert c.previous_error == 0.0
        assert c.integral_error == 0.0

    def test_init_custom(self):
        c = DynamicSubstrateController(target_outlet_conc=5.0, kp=1.0, ki=0.01, kd=0.05)
        assert c.target_outlet_conc == 5.0
        assert c.kp == 1.0

    def test_update(self):
        c = DynamicSubstrateController()
        result = c.update(6.0, 10.0)
        assert c.min_substrate <= result <= c.max_substrate
        assert len(c.error_history) == 1

    def test_update_zero_dt(self):
        c = DynamicSubstrateController()
        result = c.update(6.0, 0.0)
        assert isinstance(result, float)

    def test_update_history_trim(self):
        c = DynamicSubstrateController()
        c.error_history = list(range(100))
        c.update(6.0, 1.0)
        assert len(c.error_history) == 100

    def test_update_clips_min(self):
        c = DynamicSubstrateController()
        result = c.update(100.0, 1.0)
        assert result >= c.min_substrate

    def test_update_clips_max(self):
        c = DynamicSubstrateController()
        result = c.update(-100.0, 1.0)
        assert result <= c.max_substrate

    def test_get_control_metrics_empty(self):
        c = DynamicSubstrateController()
        m = c.get_control_metrics()
        assert m["rmse"] == 0
        assert m["mean_error"] == 0

    def test_get_control_metrics_with_data(self):
        c = DynamicSubstrateController()
        c.update(5.0, 1.0)
        c.update(6.0, 1.0)
        c.update(7.0, 1.0)
        m = c.get_control_metrics()
        assert "rmse" in m
        assert "integral_error" in m
        assert m["rmse"] > 0


@pytest.mark.coverage_extra
class TestQLearningFlowController:
    def test_init(self):
        q = QLearningFlowController()
        assert q.learning_rate == 0.1
        assert q.epsilon == 0.3
        assert len(q.actions) == 9

    def test_setup_state_action_spaces(self):
        q = QLearningFlowController()
        assert len(q.power_bins) == 10
        assert len(q.biofilm_bins) == 10

    def test_discretize_state(self):
        q = QLearningFlowController()
        state = q.discretize_state(0.5, 0.3, 10.0, 100)
        assert len(state) == 4

    def test_discretize_state_extremes(self):
        q = QLearningFlowController()
        state = q.discretize_state(0.0, 0.0, 0.0, 0)
        assert all(isinstance(s, (int, np.integer)) for s in state)
        state = q.discretize_state(10.0, 5.0, 100.0, 2000)
        assert all(isinstance(s, (int, np.integer)) for s in state)

    def test_select_action_explore(self):
        q = QLearningFlowController()
        q.epsilon = 1.0
        state = (0, 0, 0, 0)
        action_idx, new_flow = q.select_action(state, 0.02)
        assert 0 <= action_idx < len(q.actions)
        assert 0.005 <= new_flow <= 0.050

    def test_select_action_exploit(self):
        q = QLearningFlowController()
        q.epsilon = 0.0
        state = (0, 0, 0, 0)
        q.q_table[state] = {i: float(i) for i in range(len(q.actions))}
        action_idx, new_flow = q.select_action(state, 0.02)
        assert action_idx == len(q.actions) - 1

    def test_update_q_table(self):
        q = QLearningFlowController()
        state = (1, 2, 3, 0)
        next_state = (1, 2, 3, 1)
        initial_epsilon = q.epsilon
        q.update_q_table(state, 0, 5.0, next_state)
        assert q.epsilon < initial_epsilon

    def test_calculate_reward_power_increase(self):
        q = QLearningFlowController()
        r = q.calculate_reward(1.0, 0.01, 10.0, 0.5, 0.01, 5.0)
        assert isinstance(r, float)

    def test_calculate_reward_power_decrease(self):
        q = QLearningFlowController()
        r = q.calculate_reward(0.5, 0.01, 5.0, 1.0, 0.01, 10.0)
        assert isinstance(r, float)

    def test_calculate_reward_no_change(self):
        q = QLearningFlowController()
        r = q.calculate_reward(1.0, 0.01, 10.0, 1.0, 0.01, 10.0)
        assert isinstance(r, float)

    def test_calculate_reward_deviation_within_threshold(self):
        q = QLearningFlowController()
        threshold = 0.05 * 1.3
        r = q.calculate_reward(1.0, threshold * 0.5, 10.0, 0.8, 0.02, 8.0)
        assert r > 0

    def test_calculate_reward_deviation_outside_threshold(self):
        q = QLearningFlowController()
        r = q.calculate_reward(1.0, 0.5, 10.0, 0.8, 0.02, 8.0)
        assert isinstance(r, float)

    def test_calculate_reward_with_biofilm_history_steady(self):
        q = QLearningFlowController()
        history = [1.3, 1.3, 1.3]
        r = q.calculate_reward(1.0, 0.01, 10.0, 0.8, 0.01, 8.0,
                                biofilm_thickness_history=history)
        assert r > 0

    def test_calculate_reward_with_biofilm_history_growing(self):
        q = QLearningFlowController()
        history = [1.0, 1.1, 1.3]
        r = q.calculate_reward(1.0, 0.01, 10.0, 0.8, 0.01, 8.0,
                                biofilm_thickness_history=history)
        assert isinstance(r, float)

    def test_calculate_reward_combined_penalty(self):
        q = QLearningFlowController()
        threshold = 0.05 * 1.3
        r = q.calculate_reward(0.3, threshold + 0.1, 3.0, 0.5, 0.01, 5.0)
        assert isinstance(r, float)

    def test_calculate_reward_short_history(self):
        q = QLearningFlowController()
        r = q.calculate_reward(1.0, 0.01, 10.0, 0.8, 0.01, 8.0,
                                biofilm_thickness_history=[1.3])
        assert isinstance(r, float)


@pytest.mark.coverage_extra
class TestAddSubplotLabels:
    def test_lowercase(self):
        fig = MagicMock()
        ax1 = MagicMock()
        ax2 = MagicMock()
        fig.get_axes.return_value = [ax1, ax2]
        add_subplot_labels(fig, "a")
        assert ax1.text.called
        assert ax2.text.called

    def test_uppercase(self):
        fig = MagicMock()
        ax1 = MagicMock()
        fig.get_axes.return_value = [ax1]
        add_subplot_labels(fig, "A")
        ax1.text.assert_called_once()


@pytest.mark.coverage_extra
class TestMFCDynamicSubstrateSimulation:
    def test_init(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False, target_outlet_conc=5.0)
        assert sim.num_cells == 5
        assert sim.use_gpu is False
        assert sim.substrate_controller.target_outlet_conc == 5.0

    def test_initialize_arrays(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        assert sim.biofilm_thickness[0, 0] == 1.0
        assert sim.acetate_concentrations[0, 0] == 20.0
        assert sim.flow_rates[0] == 0.010

    def test_biofilm_factor(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        f = sim.biofilm_factor(1.3)
        assert abs(f - 1.0) < 0.01
        f2 = sim.biofilm_factor(0.5)
        assert f2 > 1.0

    def test_biofilm_factor_array(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        t = np.array([0.5, 1.0, 1.3, 2.0])
        f = sim.biofilm_factor(t)
        assert len(f) == 4

    def test_reaction_rate(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        r = sim.reaction_rate(20.0, 1.3)
        assert r > 0

    def test_reaction_rate_array(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        c = np.array([10.0, 20.0])
        b = np.array([1.0, 1.3])
        r = sim.reaction_rate(c, b)
        assert len(r) == 2

    def test_update_cell(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        result = sim.update_cell(0, 20.0, 0.01, 1.3)
        assert "outlet_concentration" in result
        assert "voltage" in result
        assert "power" in result
        assert result["outlet_concentration"] > 0

    def test_update_cell_debug(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.debug_counter = 0
        result = sim.update_cell(0, 20.0, 0.01, 1.3)
        assert sim.debug_counter == 1

    def test_update_biofilm(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)
        for i in range(5):
            assert 0.5 <= sim.biofilm_thickness[1, i] <= 3.0

    def test_update_biofilm_above_optimal(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.biofilm_thickness[0, :] = 1.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)

    def test_update_biofilm_below_optimal(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.biofilm_thickness[0, :] = 0.8
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)

    def test_simulate_step_zero(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.simulate_step(0)

    def test_simulate_step_one(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.simulate_step(1)
        assert sim.stack_voltages[1] > 0

    def test_simulate_step_two(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.simulate_step(1)
        sim.simulate_step(2)

    def test_simulate_step_q_learning(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        for s in range(1, 62):
            sim.simulate_step(s)
        assert sim.flow_rates[60] > 0

    def test_simulate_step_q_update(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        for s in range(1, 122):
            sim.simulate_step(s)

    def test_simulate_step_biofilm_history(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        for s in range(1, 62):
            sim.simulate_step(s)
        assert len(sim.biofilm_history) > 0

    def test_simulate_step_biofilm_history_trim(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.biofilm_history = list(range(10))
        sim.simulate_step(1)
        for s in range(2, 62):
            sim.simulate_step(s)

    def test_run_simulation_very_short(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.num_steps = 5
        sim.initialize_arrays()
        sim.run_simulation()

    def test_save_data(self, tmp_path):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.num_steps = 5
        sim.initialize_arrays()
        sim.run_simulation()
        mock_pc.get_simulation_data_path.return_value = str(tmp_path / "data.csv")
        mock_pc.get_model_path.return_value = str(tmp_path / "model.pkl")
        with patch("mfc_dynamic_substrate_control.get_simulation_data_path",
                    return_value=str(tmp_path / "data.csv")):
            with patch("mfc_dynamic_substrate_control.get_model_path",
                        return_value=str(tmp_path / "model.pkl")):
                ts = sim.save_data()
        assert ts is not None

    def test_generate_plots(self):
        sim = MFCDynamicSubstrateSimulation(use_gpu=False)
        sim.num_steps = 10
        sim.initialize_arrays()
        sim.run_simulation()
        with patch("mfc_dynamic_substrate_control.plt") as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            mock_plt.subplot.return_value = MagicMock()
            mock_plt.gcf.return_value = mock_fig
            mock_fig.get_axes.return_value = [MagicMock() for _ in range(12)]
            with patch("mfc_dynamic_substrate_control.get_figure_path",
                        return_value="/tmp/fig.png"):
                sim.generate_plots("20250101_000000")


@pytest.mark.coverage_extra
class TestMainFunction:
    def test_main(self):
        with patch("mfc_dynamic_substrate_control.MFCDynamicSubstrateSimulation") as mock_cls:
            mock_sim = MagicMock()
            mock_sim.stack_powers = np.array([0.1, 0.2])
            mock_sim.dt = 10.0
            mock_sim.substrate_controller.get_control_metrics.return_value = {"rmse": 0.1}
            mock_sim.save_data.return_value = "ts"
            mock_cls.return_value = mock_sim
            from mfc_dynamic_substrate_control import main
            main()
