"""Tests for mfc_dynamic_substrate_control.py."""
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

mock_gpu = MagicMock()
mock_gpu.is_gpu_available.return_value = False
mock_gpu.zeros = np.zeros

with patch.dict(sys.modules, {'gpu_acceleration': MagicMock(get_gpu_accelerator=MagicMock(return_value=mock_gpu))}), \
     patch('path_config.get_figure_path', return_value='/tmp/f.png'), \
     patch('path_config.get_model_path', return_value='/tmp/m.pkl'), \
     patch('path_config.get_simulation_data_path', return_value='/tmp/d.csv'):
    from mfc_dynamic_substrate_control import (
        DynamicSubstrateController,
        MFCDynamicSubstrateSimulation,
        QLearningFlowController,
        add_subplot_labels,
    )


class TestDynamicSubstrateController:
    @pytest.fixture
    def ctrl(self):
        return DynamicSubstrateController()

    def test_init(self, ctrl):
        assert ctrl.target_outlet_conc == 8.0
        assert ctrl.kp == 2.0

    def test_update(self, ctrl):
        v = ctrl.update(10.0, 1.0)
        assert ctrl.min_substrate <= v <= ctrl.max_substrate

    def test_update_clamps(self, ctrl):
        ctrl.integral_error = 1e6
        v = ctrl.update(0.0, 1.0)
        assert v <= ctrl.max_substrate

    def test_update_zero_dt(self, ctrl):
        v = ctrl.update(8.0, 0.0)
        assert isinstance(v, float)

    def test_update_trims_history(self, ctrl):
        ctrl.error_history = list(range(101))
        ctrl.update(8.0, 1.0)
        assert len(ctrl.error_history) == 101

    def test_get_control_metrics_empty(self, ctrl):
        m = ctrl.get_control_metrics()
        assert m["rmse"] == 0

    def test_get_control_metrics(self, ctrl):
        ctrl.update(10.0, 1.0)
        ctrl.update(6.0, 1.0)
        m = ctrl.get_control_metrics()
        assert "rmse" in m
        assert "integral_error" in m


class TestQLearningFlowController:
    @pytest.fixture
    def ql(self):
        return QLearningFlowController()

    def test_init(self, ql):
        assert ql.epsilon == 0.3
        assert len(ql.actions) == 9

    def test_discretize(self, ql):
        s = ql.discretize_state(0.5, 0.3, 10.0, 200)
        assert len(s) == 4

    def test_select_action_explore(self, ql):
        ql.epsilon = 1.0
        s = (0, 0, 0, 0)
        idx, flow = ql.select_action(s, 0.01)
        assert 0.005 <= flow <= 0.050

    def test_select_action_exploit(self, ql):
        ql.epsilon = 0.0
        s = (0, 0, 0, 0)
        idx, flow = ql.select_action(s, 0.01)
        assert 0.005 <= flow <= 0.050

    def test_update_q_table(self, ql):
        s1 = (0, 0, 0, 0)
        s2 = (1, 1, 1, 1)
        ql.update_q_table(s1, 0, 10.0, s2)
        assert ql.q_table[s1][0] != 0

    def test_calculate_reward_power_increase(self, ql):
        r = ql.calculate_reward(1.0, 0.03, 10.0, 0.5, 0.05, 8.0)
        assert isinstance(r, float)

    def test_calculate_reward_power_decrease(self, ql):
        r = ql.calculate_reward(0.5, 0.03, 8.0, 1.0, 0.05, 10.0)
        assert isinstance(r, float)

    def test_calculate_reward_within_optimal(self, ql):
        r = ql.calculate_reward(1.0, 0.02, 10.0, 1.0, 0.02, 10.0)
        assert r > 0

    def test_calculate_reward_steady_state(self, ql):
        hist = [1.3, 1.3, 1.3]
        r = ql.calculate_reward(
            1.0, 0.02, 10.0, 1.0, 0.02, 10.0,
            biofilm_thickness_history=hist,
        )
        assert r > 0

    def test_calculate_reward_combined_penalty(self, ql):
        r = ql.calculate_reward(0.5, 0.5, 5.0, 1.0, 0.05, 10.0)
        assert isinstance(r, float)

    def test_calculate_reward_no_change(self, ql):
        r = ql.calculate_reward(1.0, 0.03, 10.0, 1.0, 0.03, 10.0)
        assert isinstance(r, float)

    def test_calculate_reward_short_history(self, ql):
        r = ql.calculate_reward(1.0, 0.02, 10.0, 1.0, 0.02, 10.0,
                                biofilm_thickness_history=[1.3])
        assert isinstance(r, float)


class TestAddSubplotLabels:
    @patch('matplotlib.pyplot.subplots')
    def test_lowercase(self, mock_sub):
        fig = MagicMock()
        ax1 = MagicMock()
        ax2 = MagicMock()
        fig.get_axes.return_value = [ax1, ax2]
        add_subplot_labels(fig, "a")
        assert ax1.text.called

    @patch('matplotlib.pyplot.subplots')
    def test_uppercase(self, mock_sub):
        fig = MagicMock()
        ax1 = MagicMock()
        fig.get_axes.return_value = [ax1]
        add_subplot_labels(fig, "A")
        assert ax1.text.called


class TestMFCDynamicSubstrateSimulation:
    @pytest.fixture
    def sim(self, tmp_path):
        with patch('os.makedirs'):
            s = MFCDynamicSubstrateSimulation(use_gpu=False, target_outlet_conc=8.0)
        return s

    def test_init(self, sim):
        assert sim.num_cells == 5
        assert sim.use_gpu is False

    def test_biofilm_factor(self, sim):
        f = sim.biofilm_factor(1.3)
        assert f >= 1.0

    def test_reaction_rate(self, sim):
        r = sim.reaction_rate(10.0, 1.3)
        assert r > 0

    def test_reaction_rate_at_optimal(self, sim):
        r = sim.reaction_rate(10.0, 1.3)
        r2 = sim.reaction_rate(10.0, 2.0)
        assert r != r2

    def test_update_cell(self, sim):
        result = sim.update_cell(0, 20.0, 0.01, 1.3)
        assert "outlet_concentration" in result
        assert "power" in result

    def test_update_biofilm(self, sim):
        sim.biofilm_thickness[0, :] = 1.0
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)
        assert np.all(sim.biofilm_thickness[1, :] >= 0.5)

    def test_update_biofilm_above_optimal(self, sim):
        sim.biofilm_thickness[0, :] = 1.5
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)

    def test_update_biofilm_below_optimal(self, sim):
        sim.biofilm_thickness[0, :] = 0.8
        sim.acetate_concentrations[0, :] = 20.0
        sim.flow_rates[0] = 0.01
        sim.update_biofilm(1, 10.0)

    def test_simulate_step_zero(self, sim):
        sim.simulate_step(0)

    def test_simulate_step_one(self, sim):
        sim.simulate_step(1)

    def test_simulate_few_steps(self, sim):
        for i in range(3):
            sim.simulate_step(i)

    def test_simulate_q_learning_step(self, sim):
        for i in range(62):
            sim.simulate_step(i)

    def test_simulate_biofilm_history(self, sim):
        for i in range(121):
            sim.simulate_step(i)
        assert len(sim.biofilm_history) > 0

    def test_simulate_with_prev_state(self, sim):
        """Cover the Q-table update path with prev_state."""
        for i in range(125):
            sim.simulate_step(i)
        # After 2 Q-learning intervals we have prev_state
        assert hasattr(sim, 'prev_state')

    def test_run_simulation_short(self, tmp_path):
        """Run a very short simulation."""
        with patch('os.makedirs'):
            s = MFCDynamicSubstrateSimulation(use_gpu=False)
        s.num_steps = 5
        s.total_time = 50.0
        s.run_simulation()
        assert s.stack_powers is not None

    def test_save_data(self, tmp_path):
        with patch('os.makedirs'), \
             patch('mfc_dynamic_substrate_control.get_simulation_data_path', return_value=str(tmp_path / 'd.csv')), \
             patch('mfc_dynamic_substrate_control.get_model_path', return_value=str(tmp_path / 'm.pkl')):
            s = MFCDynamicSubstrateSimulation(use_gpu=False)
            s.num_steps = 5
            s.total_time = 50.0
            s.run_simulation()
            ts = s.save_data()
            assert ts is not None

    def test_generate_plots(self, tmp_path):
        with patch('os.makedirs'), \
             patch('mfc_dynamic_substrate_control.get_figure_path', return_value=str(tmp_path / 'f.png')):
            import matplotlib
            matplotlib.use('Agg')
            s = MFCDynamicSubstrateSimulation(use_gpu=False)
            s.num_steps = 5
            s.total_time = 50.0
            s.run_simulation()
            s.generate_plots("20250101_000000")

    def test_progress_reporting(self, tmp_path):
        """Cover the progress reporting code path."""
        with patch('os.makedirs'):
            s = MFCDynamicSubstrateSimulation(use_gpu=False)
        s.num_steps = 36001
        s.total_time = 360010.0
        # Just run step 0 and 36000 for reporting
        s.simulate_step(0)
        s.simulate_step(1)


class TestMain:
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('os.makedirs')
    @patch('path_config.get_figure_path', return_value='/tmp/f.png')
    @patch('path_config.get_simulation_data_path', return_value='/tmp/d.csv')
    @patch('path_config.get_model_path', return_value='/tmp/m.pkl')
    def test_main_runs(self, *_):
        from mfc_dynamic_substrate_control import MFCDynamicSubstrateSimulation
        with patch.object(MFCDynamicSubstrateSimulation, 'run_simulation'), \
             patch.object(MFCDynamicSubstrateSimulation, 'save_data', return_value='ts'), \
             patch.object(MFCDynamicSubstrateSimulation, 'generate_plots'):
            from mfc_dynamic_substrate_control import main
            main()
