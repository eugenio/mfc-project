"""Coverage boost tests for mfc_qlearning_optimization_parallel.py targeting remaining lines."""
import json
import os
import pickle
import sys
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mfc_qlearning_optimization_parallel import (
    MFCQLearningSimulationParallel,
    QLearningFlowController,
    add_subplot_labels,
    main,
)


@pytest.mark.coverage_extra
class TestQLearningFlowControllerCov3:
    def test_save_q_table(self, tmp_path):
        ctrl = QLearningFlowController()
        ctrl.q_table[(0, 0, 0, 0)][0] = 5.0
        path = str(tmp_path / "q_table.pkl")
        ctrl.save_q_table(path)
        assert os.path.exists(path)

    def test_load_q_table(self, tmp_path):
        ctrl = QLearningFlowController()
        path = str(tmp_path / "q_table.pkl")
        data = {(0, 0, 0, 0): {0: 5.0}}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        ctrl.load_q_table(path)
        assert ctrl.q_table[(0, 0, 0, 0)][0] == 5.0

    def test_load_q_table_missing_file(self, tmp_path):
        ctrl = QLearningFlowController()
        ctrl.load_q_table(str(tmp_path / "nonexistent.pkl"))
        assert len(ctrl.q_table) == 0

    def test_calculate_reward_combined_penalty(self):
        ctrl = QLearningFlowController()
        r = ctrl.calculate_reward(
            power=0.1, biofilm_deviation=0.5, substrate_utilization=10.0,
            prev_power=0.2, prev_biofilm_dev=0.3, prev_substrate_util=15.0,
        )
        assert r < 0

    def test_calculate_reward_power_zero_change(self):
        ctrl = QLearningFlowController()
        r = ctrl.calculate_reward(
            power=0.5, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=0.5, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
        )
        assert isinstance(r, (int, float))

    def test_calculate_reward_substrate_zero_change(self):
        ctrl = QLearningFlowController()
        r = ctrl.calculate_reward(
            power=0.6, biofilm_deviation=0.01, substrate_utilization=20.0,
            prev_power=0.5, prev_biofilm_dev=0.01, prev_substrate_util=20.0,
        )
        assert isinstance(r, (int, float))


@pytest.mark.coverage_extra
class TestSimulationParallelSaveData:
    def test_save_data(self, tmp_path):
        with patch("mfc_qlearning_optimization_parallel.get_simulation_data_path",
                   side_effect=lambda f: str(tmp_path / f)), \
             patch("mfc_qlearning_optimization_parallel.get_model_path",
                   side_effect=lambda f: str(tmp_path / f)):
            sim = MFCQLearningSimulationParallel(use_gpu=False)
            n = 10
            nc = 2
            sim.num_steps = n
            sim.num_cells = nc
            sim.dt = 10.0
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
            timestamp = sim.save_data()
            assert isinstance(timestamp, str)
            assert any("csv" in f for f in os.listdir(tmp_path))


@pytest.mark.coverage_extra
class TestSimulationParallelGeneratePlots:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.tight_layout")
    def test_generate_plots(self, mock_tl, mock_close, mock_save):
        with patch("mfc_qlearning_optimization_parallel.get_figure_path",
                   side_effect=lambda f: f):
            sim = MFCQLearningSimulationParallel(use_gpu=False)
            n = 10
            nc = 2
            sim.num_steps = n
            sim.num_cells = nc
            sim.dt = 10.0
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
            sim.optimal_biofilm_thickness = 1.3
            sim.generate_plots("20250101_000000")
            assert mock_save.called


@pytest.mark.coverage_extra
class TestMainParallel:
    @patch("mfc_qlearning_optimization_parallel.MFCQLearningSimulationParallel")
    def test_main(self, mock_cls):
        mock_sim = MagicMock()
        mock_sim.stack_powers = np.array([0.1, 0.2, 0.3])
        mock_sim.dt = 10.0
        mock_sim.save_data.return_value = "20250101_000000"
        mock_cls.return_value = mock_sim
        main()
        mock_sim.run_simulation.assert_called_once()
        mock_sim.save_data.assert_called_once()
        mock_sim.generate_plots.assert_called_once()


@pytest.mark.coverage_extra
class TestAddSubplotLabelsParallel:
    def test_uppercase_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2)
        add_subplot_labels(fig, "L")
        plt.close(fig)
