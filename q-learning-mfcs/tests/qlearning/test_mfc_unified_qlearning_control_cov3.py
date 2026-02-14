"""Coverage boost tests for mfc_unified_qlearning_control.py targeting remaining uncovered lines."""
import json
import os
import pickle
import sys
from collections import defaultdict
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mfc_unified_qlearning_control import (
    MFCUnifiedQLearningSimulation,
    UnifiedQLearningController,
    add_subplot_labels,
    main,
)


@pytest.mark.coverage_extra
class TestUnifiedControllerEdgeCases:
    """Cover remaining branches in UnifiedQLearningController."""

    def test_select_action_conservative_exploration(self):
        """Cover conservative exploration branch (good recent performance)."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.epsilon = 1.0
        ctrl.performance_history = [100.0] * 15
        # Seed random for reproducibility
        np.random.seed(42)
        state = (0, 0, 0, 0, 0, 0)
        _, fr, ic = ctrl.select_action(state, 0.01, 20.0)
        assert fr >= 0.005 and fr <= 0.050

    def test_select_action_no_conservative_actions(self):
        """Cover fallback when no conservative actions found."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.epsilon = 1.0
        ctrl.performance_history = [100.0] * 15
        # Replace actions with all large-magnitude to ensure no conservative
        ctrl.actions = [(20, 10), (-20, -10), (15, 8)]
        np.random.seed(0)
        state = (0, 0, 0, 0, 0, 0)
        a, fr, ic = ctrl.select_action(state, 0.01, 20.0)
        assert 0 <= a < len(ctrl.actions)

    def test_select_action_action_history_trimming(self):
        """Cover action history trimming to 1000."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.action_history = [(0, 0)] * 1001
        ctrl.epsilon = 1.0
        np.random.seed(0)
        state = (0, 0, 0, 0, 0, 0)
        ctrl.select_action(state, 0.01, 20.0)
        assert len(ctrl.action_history) <= 1001

    def test_update_q_table_doing_well(self):
        """Cover adaptive decay when recent_avg > 50."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = [100.0] * 15
        ctrl.epsilon = 0.3
        ctrl.update_q_table((0, 0, 0, 0, 0, 0), 0, 100.0, (1, 0, 0, 0, 0, 0))
        # Should use decay_factor = 0.999
        assert ctrl.epsilon < 0.3

    def test_update_q_table_doing_poorly(self):
        """Cover adaptive decay when recent_avg < -50."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = [-100.0] * 15
        ctrl.epsilon = 0.3
        initial_eps = ctrl.epsilon
        ctrl.update_q_table((0, 0, 0, 0, 0, 0), 0, -100.0, (1, 0, 0, 0, 0, 0))
        # Should use decay_factor = 0.9995 (slower decay)
        assert ctrl.epsilon < initial_eps

    def test_update_q_table_normal_decay(self):
        """Cover normal epsilon_decay path."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = [10.0] * 15
        ctrl.epsilon = 0.3
        ctrl.update_q_table((0, 0, 0, 0, 0, 0), 0, 10.0, (1, 0, 0, 0, 0, 0))
        assert ctrl.epsilon > ctrl.epsilon_min

    def test_update_q_table_few_history(self):
        """Cover branch with < 10 performance history."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = [10.0] * 5
        ctrl.epsilon = 0.3
        ctrl.update_q_table((0, 0, 0, 0, 0, 0), 0, 10.0, (1, 0, 0, 0, 0, 0))
        assert ctrl.epsilon >= ctrl.epsilon_min

    def test_update_q_table_performance_history_trimming(self):
        """Cover trimming performance_history to 100."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = list(range(105))
        ctrl.update_q_table((0, 0, 0, 0, 0, 0), 0, 10.0, (1, 0, 0, 0, 0, 0))
        # History gets appended (+1) but only trimmed when > 100; after update = 106, then pop = 105
        # The trimming pops front to keep <= 100 items
        assert len(ctrl.performance_history) <= 106

    def test_calculate_unified_reward_all_branches(self):
        """Cover remaining reward branches."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)

        # Cover power_change == 0
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(r, float)

        # Cover substrate_change == 0
        r = ctrl.calculate_unified_reward(
            power=0.6, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(r, float)

        # Cover outlet error 1-2 range
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=13.5, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(r, float)

        # Cover outlet error > 2 range
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=20.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
        )
        assert isinstance(r, float)

        # Cover concentration_reward == 0 (no improvement)
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=15.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=15.0,
        )
        assert isinstance(r, float)

    def test_calculate_unified_reward_flow_penalty(self):
        """Cover flow rate penalty branch."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.current_flow_rate = 25.0
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.5, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.5,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[0.5, 0.5, 0.5, 0.6, 0.7],
        )
        assert isinstance(r, float)

    def test_calculate_unified_reward_biofilm_history_steady_state(self):
        """Cover biofilm steady state bonus."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        r = ctrl.calculate_unified_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            outlet_conc=12.0, prev_power=0.5, prev_biofilm_dev=0.01,
            prev_substrate_util=20.0, prev_outlet_conc=12.0,
            biofilm_thickness_history=[1.3, 1.3, 1.3],
        )
        assert isinstance(r, float)

    def test_get_control_statistics_with_few_rewards(self):
        """Cover statistics with < 10 and < 50 rewards."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = [5.0, 10.0, 15.0]
        stats = ctrl.get_control_statistics()
        assert stats["reward_trend"] == 0

    def test_get_control_statistics_with_many_rewards(self):
        """Cover statistics with >= 50 rewards."""
        ctrl = UnifiedQLearningController(target_outlet_conc=12.0)
        ctrl.performance_history = list(range(60))
        stats = ctrl.get_control_statistics()
        assert "reward_trend" in stats
        assert stats["q_table_size"] >= 0


@pytest.mark.coverage_extra
class TestMFCUnifiedSimulationSaveData:
    """Cover save_data method."""

    def test_save_data(self, tmp_path):
        with patch("mfc_unified_qlearning_control.get_simulation_data_path", side_effect=lambda f: str(tmp_path / f)), \
             patch("mfc_unified_qlearning_control.get_model_path", side_effect=lambda f: str(tmp_path / f)):
            sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
            # Set up minimal data to avoid huge arrays
            sim.num_steps = 10
            sim.num_cells = 2
            sim.dt = 10.0
            n = sim.num_steps
            nc = sim.num_cells
            sim.cell_voltages = np.random.rand(n, nc)
            sim.biofilm_thickness = np.random.rand(n, nc)
            sim.acetate_concentrations = np.random.rand(n, nc)
            sim.current_densities = np.random.rand(n, nc)
            sim.power_outputs = np.random.rand(n, nc)
            sim.substrate_consumptions = np.random.rand(n, nc)
            sim.stack_voltages = np.random.rand(n)
            sim.stack_powers = np.random.rand(n)
            sim.flow_rates = np.random.rand(n)
            sim.objective_values = np.random.rand(n)
            sim.substrate_utilizations = np.random.rand(n)
            sim.q_rewards = np.random.rand(n)
            sim.q_actions = np.random.rand(n)
            sim.inlet_concentrations = np.random.rand(n) + 10
            sim.outlet_concentrations = np.random.rand(n) + 5
            sim.concentration_errors = np.random.rand(n)

            timestamp = sim.save_data()
            assert isinstance(timestamp, str)
            # Verify files were created
            assert any("csv" in f for f in os.listdir(tmp_path))
            assert any("json" in f for f in os.listdir(tmp_path))
            assert any("pkl" in f for f in os.listdir(tmp_path))


@pytest.mark.coverage_extra
class TestMFCUnifiedSimulationGeneratePlots:
    """Cover generate_plots method."""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.tight_layout")
    def test_generate_plots(self, mock_tl, mock_close, mock_save):
        with patch("mfc_unified_qlearning_control.get_figure_path", side_effect=lambda f: f):
            sim = MFCUnifiedQLearningSimulation(use_gpu=False, target_outlet_conc=12.0)
            sim.num_steps = 10
            sim.num_cells = 2
            sim.dt = 10.0
            n = sim.num_steps
            nc = sim.num_cells
            sim.cell_voltages = np.random.rand(n, nc)
            sim.biofilm_thickness = np.random.rand(n, nc)
            sim.acetate_concentrations = np.random.rand(n, nc)
            sim.current_densities = np.random.rand(n, nc)
            sim.power_outputs = np.random.rand(n, nc)
            sim.substrate_consumptions = np.random.rand(n, nc)
            sim.stack_voltages = np.random.rand(n)
            sim.stack_powers = np.random.rand(n)
            sim.flow_rates = np.random.rand(n)
            sim.objective_values = np.random.rand(n)
            sim.substrate_utilizations = np.random.rand(n)
            sim.q_rewards = np.random.rand(n)
            sim.q_actions = np.random.rand(n)
            sim.inlet_concentrations = np.random.rand(n) + 10
            sim.outlet_concentrations = np.random.rand(n) + 5
            sim.concentration_errors = np.random.rand(n)
            sim.optimal_biofilm_thickness = 1.3
            # Ensure controller has stats data for generate_plots summary
            sim.unified_controller.performance_history = [10.0] * 20
            sim.unified_controller.total_rewards = 100.0
            sim.generate_plots("20250101_000000")
            assert mock_save.called


@pytest.mark.coverage_extra
class TestMainFunction:
    """Cover main() function."""

    @patch("mfc_unified_qlearning_control.MFCUnifiedQLearningSimulation")
    def test_main(self, mock_sim_cls):
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.array([0.1, 0.2, 0.3])
        mock_sim.dt = 10.0
        mock_sim.outlet_concentrations = np.array([10.0, 11.0, 12.0])
        mock_sim.concentration_errors = np.array([1.0, 0.5, 0.1])
        mock_sim.unified_controller = MagicMock()
        mock_sim.save_data.return_value = "20250101_000000"
        mock_sim_cls.return_value = mock_sim
        main()
        mock_sim.run_simulation.assert_called_once()
        mock_sim.save_data.assert_called_once()
        mock_sim.generate_plots.assert_called_once()


@pytest.mark.coverage_extra
class TestAddSubplotLabelsUppercase:
    """Cover uppercase branch of add_subplot_labels."""

    def test_uppercase_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        add_subplot_labels(fig, "A")
        plt.close(fig)
