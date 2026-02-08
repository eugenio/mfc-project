"""Coverage tests for mfc_recirculation_control.py (98%+ target)."""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mfc_recirculation_control import (
    AdvancedQLearningFlowController,
    AnolytereservoirSystem,
    MFCCellWithMonitoring,
    SubstrateConcentrationController,
    parse_arguments,
    run_mfc_simulation,
    simulate_mfc_with_recirculation,
)


class TestAnolytereservoirSystem:
    def test_init_defaults(self):
        res = AnolytereservoirSystem()
        assert res.volume == 1.0
        assert res.substrate_concentration == 20.0
        assert res.total_substrate_added == 0.0
        assert res.total_volume_circulated == 0.0
        assert res.pump_efficiency == 0.95
        assert res.pipe_dead_volume == 0.05
        assert res.circulation_cycles == 0

    def test_init_custom(self):
        res = AnolytereservoirSystem(initial_substrate_conc=30.0, volume_liters=2.0)
        assert res.volume == 2.0
        assert res.substrate_concentration == 30.0

    def test_add_substrate(self):
        res = AnolytereservoirSystem(initial_substrate_conc=20.0, volume_liters=1.0)
        res.add_substrate(10.0, 0.01)
        assert res.total_substrate_added > 0
        assert res.substrate_concentration > 20.0

    def test_add_substrate_halted(self):
        res = AnolytereservoirSystem()
        res.substrate_halt = True
        old_conc = res.substrate_concentration
        res.add_substrate(10.0, 0.01)
        assert res.substrate_concentration == old_conc
        assert res.total_substrate_added == 0.0

    def test_circulate_anolyte_positive_flow(self):
        res = AnolytereservoirSystem(initial_substrate_conc=20.0)
        res.circulate_anolyte(10.0, 15.0, 0.01)
        assert res.total_volume_circulated > 0
        assert res.circulation_cycles == 1
        assert res.total_pump_time > 0
        assert len(res.substrate_balance_history) == 1
        assert len(res.mixing_efficiency_history) == 1

    def test_circulate_anolyte_zero_flow(self):
        res = AnolytereservoirSystem()
        old_conc = res.substrate_concentration
        res.circulate_anolyte(0.0, 15.0, 0.01)
        assert res.circulation_cycles == 0
        assert res.total_pump_time == 0.0
        assert res.substrate_concentration == old_conc

    def test_circulate_anolyte_concentration_change(self):
        res = AnolytereservoirSystem(initial_substrate_conc=20.0)
        res.circulate_anolyte(10.0, 10.0, 0.01)
        assert res.substrate_concentration < 20.0

    def test_circulate_anolyte_concentration_bounded(self):
        res = AnolytereservoirSystem(initial_substrate_conc=1.0)
        res.circulate_anolyte(1000.0, 0.0, 1.0)
        assert res.substrate_concentration >= 0.0

    def test_get_inlet_concentration(self):
        res = AnolytereservoirSystem(initial_substrate_conc=25.0)
        assert res.get_inlet_concentration() == 25.0

    def test_get_sensor_readings_empty_history(self):
        res = AnolytereservoirSystem()
        readings = res.get_sensor_readings()
        assert readings["substrate_concentration"] == 20.0
        assert readings["total_substrate_added"] == 0.0
        assert readings["total_volume_circulated"] == 0.0
        assert readings["circulation_cycles"] == 0
        assert readings["current_mixing_efficiency"] == 1.0
        assert readings["substrate_addition_active"] is True

    def test_get_sensor_readings_with_history(self):
        res = AnolytereservoirSystem()
        res.circulate_anolyte(10.0, 15.0, 0.01)
        readings = res.get_sensor_readings()
        assert readings["current_mixing_efficiency"] > 0
        assert readings["circulation_cycles"] == 1
        assert readings["pump_operation_time"] > 0

    def test_get_sensor_readings_halted(self):
        res = AnolytereservoirSystem()
        res.substrate_halt = True
        readings = res.get_sensor_readings()
        assert readings["substrate_addition_active"] is False


class TestSubstrateConcentrationController:
    def test_init_defaults(self):
        ctrl = SubstrateConcentrationController()
        assert ctrl.target_outlet_conc == 12.0
        assert ctrl.target_reservoir_conc == 20.0
        assert ctrl.control_mode == "normal"

    def test_init_custom(self):
        ctrl = SubstrateConcentrationController(
            target_outlet_conc=15.0, target_reservoir_conc=25.0
        )
        assert ctrl.target_outlet_conc == 15.0
        assert ctrl.target_reservoir_conc == 25.0

    def _make_sensors(self):
        return {
            "substrate_concentration": 20.0,
            "total_substrate_added": 0.0,
            "total_volume_circulated": 1.0,
            "circulation_cycles": 10,
            "pump_operation_time": 1.0,
            "current_mixing_efficiency": 0.9,
            "substrate_addition_active": True,
        }

    def test_calculate_substrate_addition_normal(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        rate, halt = ctrl.calculate_substrate_addition(
            outlet_conc=12.0,
            reservoir_conc=20.0,
            cell_concentrations=[15.0, 14.0, 13.0, 12.0, 11.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert isinstance(rate, float)
        assert isinstance(halt, bool)
        assert ctrl.control_mode == "normal"
        assert len(ctrl.control_history) == 1

    def test_calculate_substrate_addition_starvation_warning(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        rate, halt = ctrl.calculate_substrate_addition(
            outlet_conc=12.0,
            reservoir_conc=20.0,
            cell_concentrations=[4.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert ctrl.control_mode == "warning"

    def test_calculate_substrate_addition_emergency(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        rate, halt = ctrl.calculate_substrate_addition(
            outlet_conc=12.0,
            reservoir_conc=20.0,
            cell_concentrations=[1.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert ctrl.control_mode == "emergency"

    def test_calculate_substrate_addition_conservation(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        rate, halt = ctrl.calculate_substrate_addition(
            outlet_conc=12.0,
            reservoir_conc=25.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert ctrl.control_mode == "conservation" or halt is True

    def test_calculate_substrate_addition_halt_on_decline(self):
        ctrl = SubstrateConcentrationController()
        ctrl.previous_outlet_conc = 15.0
        sensors = self._make_sensors()
        rate, halt = ctrl.calculate_substrate_addition(
            outlet_conc=14.0,
            reservoir_conc=20.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert halt is True
        assert rate == 0.0

    def test_calculate_substrate_addition_poor_mixing(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        sensors["current_mixing_efficiency"] = 0.5
        rate, _ = ctrl.calculate_substrate_addition(
            outlet_conc=10.0,
            reservoir_conc=15.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert isinstance(rate, float)

    def test_calculate_substrate_addition_excellent_mixing(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        sensors["current_mixing_efficiency"] = 0.98
        rate, _ = ctrl.calculate_substrate_addition(
            outlet_conc=10.0,
            reservoir_conc=15.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert isinstance(rate, float)

    def test_calculate_substrate_addition_no_circulation(self):
        ctrl = SubstrateConcentrationController()
        sensors = self._make_sensors()
        sensors["circulation_cycles"] = 0
        rate, _ = ctrl.calculate_substrate_addition(
            outlet_conc=10.0,
            reservoir_conc=15.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert isinstance(rate, float)

    def test_calculate_substrate_addition_previous_error_none(self):
        ctrl = SubstrateConcentrationController()
        ctrl.previous_outlet_error = None
        ctrl.previous_reservoir_error = None
        sensors = self._make_sensors()
        rate, _ = ctrl.calculate_substrate_addition(
            outlet_conc=12.0,
            reservoir_conc=20.0,
            cell_concentrations=[15.0, 15.0, 15.0, 15.0, 15.0],
            reservoir_sensors=sensors,
            dt_hours=0.01,
        )
        assert isinstance(rate, float)


class TestAdvancedQLearningFlowController:
    def test_init_defaults(self):
        ctrl = AdvancedQLearningFlowController()
        assert ctrl.learning_rate == 0.0987
        assert ctrl.discount_factor == 0.9517
        assert ctrl.epsilon == 0.3702
        assert ctrl.config is not None
        assert ctrl.total_rewards == 0

    def test_init_with_config(self):
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        ctrl = AdvancedQLearningFlowController(config=DEFAULT_QLEARNING_CONFIG)
        assert ctrl.epsilon_decay == DEFAULT_QLEARNING_CONFIG.epsilon_decay

    def test_init_custom_params(self):
        ctrl = AdvancedQLearningFlowController(
            learning_rate=0.2, discount_factor=0.9, epsilon=0.5
        )
        assert ctrl.learning_rate == 0.2
        assert ctrl.discount_factor == 0.9
        assert ctrl.epsilon == 0.5

    def test_setup_enhanced_state_action_spaces(self):
        ctrl = AdvancedQLearningFlowController()
        assert len(ctrl.power_bins) == 8
        assert len(ctrl.biofilm_bins) == 8
        assert len(ctrl.substrate_bins) == 8
        assert len(ctrl.reservoir_conc_bins) == 6
        assert len(ctrl.cell_conc_bins) == 6
        assert len(ctrl.outlet_error_bins) == 6
        assert len(ctrl.time_bins) == 4
        assert ctrl.total_actions == len(ctrl.flow_actions) * len(ctrl.substrate_actions)

    def test_discretize_enhanced_state(self):
        ctrl = AdvancedQLearningFlowController()
        state = ctrl.discretize_enhanced_state(
            power=0.5,
            biofilm_deviation=0.1,
            substrate_utilization=20.0,
            reservoir_conc=20.0,
            min_cell_conc=10.0,
            outlet_error=5.0,
            time_hours=500.0,
        )
        assert isinstance(state, tuple)
        assert len(state) == 7

    def test_discretize_enhanced_state_boundaries(self):
        ctrl = AdvancedQLearningFlowController()
        state_low = ctrl.discretize_enhanced_state(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_high = ctrl.discretize_enhanced_state(
            100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10000.0
        )
        assert all(idx >= 0 for idx in state_low)
        assert all(idx >= 0 for idx in state_high)

    def test_calculate_substrate_reward_reservoir_within_target(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=24.5,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=24.5,
            substrate_addition=0.0,
        )
        assert reward > 0

    def test_calculate_substrate_reward_outlet_within_threshold(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=25.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=25.0,
            substrate_addition=0.0,
        )
        assert isinstance(reward, float)

    def test_calculate_substrate_reward_outlet_penalty_positive_reward(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=25.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=25.0,
            substrate_addition=0.0,
            inlet_conc=25.0,
        )
        assert isinstance(reward, float)

    def test_calculate_substrate_reward_outlet_penalty_negative(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=10.0,
            cell_concentrations=[5.0, 5.0, 5.0, 5.0, 5.0],
            outlet_conc=10.0,
            substrate_addition=0.0,
            inlet_conc=10.0,
        )
        assert isinstance(reward, float)

    def test_calculate_substrate_reward_cell_within_target(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=25.0,
            cell_concentrations=[24.0, 25.0, 26.0, 24.5, 25.5],
            outlet_conc=25.0,
            substrate_addition=0.0,
        )
        assert reward > 0

    def test_calculate_substrate_reward_reservoir_excess(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=35.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=20.0,
            substrate_addition=0.0,
        )
        assert reward < 0

    def test_calculate_substrate_reward_outlet_excess(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=20.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=35.0,
            substrate_addition=0.0,
        )
        assert reward < 0

    def test_calculate_substrate_reward_severe_reservoir(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=55.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=20.0,
            substrate_addition=0.0,
        )
        assert reward < -100

    def test_calculate_substrate_reward_severe_outlet(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=20.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=55.0,
            substrate_addition=0.0,
        )
        assert reward < -100

    def test_calculate_substrate_reward_starvation(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=20.0,
            cell_concentrations=[1.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=20.0,
            substrate_addition=0.0,
        )
        assert reward < 0

    def test_calculate_substrate_reward_addition_penalty(self):
        ctrl = AdvancedQLearningFlowController()
        reward_no_add = ctrl.calculate_substrate_reward(
            reservoir_conc=25.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=25.0,
            substrate_addition=0.0,
        )
        reward_with_add = ctrl.calculate_substrate_reward(
            reservoir_conc=25.0,
            cell_concentrations=[20.0, 20.0, 20.0, 20.0, 20.0],
            outlet_conc=25.0,
            substrate_addition=5.0,
        )
        assert reward_no_add > reward_with_add

    def test_calculate_substrate_reward_empty_cell_concentrations(self):
        ctrl = AdvancedQLearningFlowController()
        reward = ctrl.calculate_substrate_reward(
            reservoir_conc=20.0,
            cell_concentrations=[],
            outlet_conc=20.0,
            substrate_addition=0.0,
        )
        assert isinstance(reward, float)

    def test_choose_combined_action_exploration(self):
        ctrl = AdvancedQLearningFlowController()
        ctrl.epsilon = 1.0
        state = (0, 0, 0, 0, 0, 0, 0)
        np.random.seed(42)
        action_idx, flow_idx, substrate_idx = ctrl.choose_combined_action(state)
        assert 0 <= action_idx < ctrl.total_actions
        assert 0 <= flow_idx < len(ctrl.flow_actions)
        assert 0 <= substrate_idx < len(ctrl.substrate_actions)

    def test_choose_combined_action_exploitation(self):
        ctrl = AdvancedQLearningFlowController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0, 0)
        ctrl.q_table[state][5] = 100.0
        action_idx, _, _ = ctrl.choose_combined_action(state)
        assert action_idx == 5

    def test_choose_action(self):
        ctrl = AdvancedQLearningFlowController()
        ctrl.epsilon = 0.0
        state = (0, 0, 0, 0, 0, 0, 0)
        flow_idx = ctrl.choose_action(state)
        assert 0 <= flow_idx < len(ctrl.flow_actions)

    def test_update_q_value(self):
        ctrl = AdvancedQLearningFlowController()
        state = (0, 0, 0, 0, 0, 0, 0)
        next_state = (1, 1, 1, 1, 1, 1, 1)
        old_eps = ctrl.epsilon
        ctrl.update_q_value(state, 0, 10.0, next_state)
        assert ctrl.q_table[state][0] != 0
        assert ctrl.epsilon < old_eps

    def test_update_q_value_none_next_state(self):
        ctrl = AdvancedQLearningFlowController()
        state = (0, 0, 0, 0, 0, 0, 0)
        ctrl.update_q_value(state, 0, 10.0, None)
        assert ctrl.q_table[state][0] != 0

    def test_update_q_value_epsilon_min(self):
        ctrl = AdvancedQLearningFlowController()
        ctrl.epsilon = ctrl.epsilon_min
        ctrl.update_q_value((0, 0, 0, 0, 0, 0, 0), 0, 1.0, (1, 1, 1, 1, 1, 1, 1))
        assert ctrl.epsilon == ctrl.epsilon_min

    def test_get_state_hash(self):
        ctrl = AdvancedQLearningFlowController()
        h = ctrl.get_state_hash(20.0, 12.0, 1.5)
        assert isinstance(h, str)
        assert "_" in h

    def test_save_checkpoint(self):
        ctrl = AdvancedQLearningFlowController()
        state = (0, 0, 0, 0, 0, 0, 0)
        ctrl.q_table[state][0] = 5.0
        ctrl.total_rewards = 100.0
        ctrl.episode_count = 10
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            checkpoint = ctrl.save_checkpoint(fname)
            assert "model_info" in checkpoint
            assert "hyperparameters" in checkpoint
            assert "q_table" in checkpoint
            assert "training_statistics" in checkpoint
            assert "configuration" in checkpoint
            assert checkpoint["training_statistics"]["q_table_size"] > 0
            with open(fname) as fp:
                loaded = json.load(fp)
            assert "q_table" in loaded
        finally:
            os.unlink(fname)

    def test_load_checkpoint(self):
        ctrl = AdvancedQLearningFlowController()
        state = (0, 0, 0, 0, 0, 0, 0)
        ctrl.q_table[state][0] = 5.0
        ctrl.total_rewards = 100.0
        ctrl.episode_count = 10
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            ctrl.save_checkpoint(fname)
            ctrl2 = AdvancedQLearningFlowController()
            checkpoint = ctrl2.load_checkpoint(fname)
            assert ctrl2.learning_rate == ctrl.learning_rate
            assert ctrl2.total_rewards == 100.0
            assert ctrl2.episode_count == 10
            assert ctrl2.q_table[state][0] == 5.0
        finally:
            os.unlink(fname)


class TestMFCCellWithMonitoring:
    def test_init(self):
        cell = MFCCellWithMonitoring(cell_id=1)
        assert cell.cell_id == 1
        assert cell.biofilm_thickness == 1.0
        assert cell.substrate_concentration == 0.0
        assert cell.optimal_biofilm_thickness == 1.3

    def test_init_custom_biofilm(self):
        cell = MFCCellWithMonitoring(cell_id=2, initial_biofilm=1.5)
        assert cell.biofilm_thickness == 1.5

    def test_update_concentrations(self):
        cell = MFCCellWithMonitoring(cell_id=1)
        outlet = cell.update_concentrations(
            inlet_conc=20.0, flow_rate_ml_h=10.0, dt_hours=0.01
        )
        assert outlet < 20.0
        assert outlet >= 0.0
        assert cell.substrate_concentration > 0
        assert len(cell.concentration_history) == 1
        assert len(cell.consumption_rate_history) == 1

    def test_update_concentrations_zero_flow(self):
        cell = MFCCellWithMonitoring(cell_id=1)
        outlet = cell.update_concentrations(
            inlet_conc=20.0, flow_rate_ml_h=0.0, dt_hours=0.01
        )
        assert outlet >= 0.0

    def test_update_concentrations_high_biofilm(self):
        cell = MFCCellWithMonitoring(cell_id=1, initial_biofilm=2.0)
        outlet = cell.update_concentrations(
            inlet_conc=20.0, flow_rate_ml_h=10.0, dt_hours=0.01
        )
        assert outlet >= 0.0

    def test_update_concentrations_low_thiele(self):
        cell = MFCCellWithMonitoring(cell_id=1, initial_biofilm=0.1)
        outlet = cell.update_concentrations(
            inlet_conc=20.0, flow_rate_ml_h=10.0, dt_hours=0.01
        )
        assert outlet >= 0.0

    def test_update_concentrations_high_substrate(self):
        cell = MFCCellWithMonitoring(cell_id=1)
        outlet = cell.update_concentrations(
            inlet_conc=150.0, flow_rate_ml_h=10.0, dt_hours=0.01
        )
        assert outlet >= 0.0

    def test_update_concentrations_biofilm_bounds(self):
        cell = MFCCellWithMonitoring(cell_id=1, initial_biofilm=1.0)
        for _ in range(50):
            cell.update_concentrations(
                inlet_conc=20.0, flow_rate_ml_h=10.0, dt_hours=0.01
            )
        assert cell.biofilm_thickness > 0

    def test_process_with_monitoring(self):
        cell = MFCCellWithMonitoring(cell_id=1)
        outlet = cell.process_with_monitoring(20.0, 10.0, 0.01)
        assert outlet >= 0.0
        assert outlet < 20.0


class TestSimulateMFCWithRecirculation:
    def test_simulate_short(self):
        results, cells, reservoir, controller, q_controller = (
            simulate_mfc_with_recirculation(duration_hours=0.1)
        )
        assert len(results["time_hours"]) > 0
        assert len(cells) == 5
        assert reservoir is not None
        assert controller is not None
        assert q_controller is not None

    def test_simulate_with_config(self):
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        results, cells, _, _, _ = simulate_mfc_with_recirculation(
            duration_hours=0.05, config=DEFAULT_QLEARNING_CONFIG
        )
        assert len(results["time_hours"]) > 0

    def test_simulate_with_checkpoint(self):
        ctrl = AdvancedQLearningFlowController()
        ctrl.q_table[(0, 0, 0, 0, 0, 0, 0)][0] = 5.0
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            ctrl.save_checkpoint(fname)
            results, _, _, _, q_ctrl = simulate_mfc_with_recirculation(
                duration_hours=0.05, checkpoint_path=fname
            )
            assert len(results["time_hours"]) > 0
        finally:
            os.unlink(fname)

    def test_simulate_nonexistent_checkpoint(self):
        results, _, _, _, _ = simulate_mfc_with_recirculation(
            duration_hours=0.05, checkpoint_path="/nonexistent/path.json"
        )
        assert len(results["time_hours"]) > 0

    def test_simulate_results_keys(self):
        results, _, _, _, _ = simulate_mfc_with_recirculation(duration_hours=0.05)
        expected_keys = [
            "time_hours", "reservoir_concentration", "cell_concentrations",
            "outlet_concentration", "flow_rate", "substrate_addition_rate",
            "total_power", "biofilm_thicknesses", "substrate_halt",
            "q_value", "epsilon", "q_action",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
        for i in range(5):
            assert f"substrate_conc_cell_{i}" in results
            assert f"biofilm_thickness_cell_{i}" in results
            assert f"power_cell_{i}" in results


class TestParseArguments:
    def test_parse_defaults(self):
        with patch("sys.argv", ["prog"]):
            args = parse_arguments()
        assert args.duration == 100
        assert args.prefix == "mfc_simulation"
        assert args.user_suffix == ""
        assert args.no_timestamp is False
        assert args.no_plots is False
        assert args.load_checkpoint is None

    def test_parse_custom(self):
        with patch(
            "sys.argv",
            ["prog", "-d", "200", "-p", "test", "-u", "mysuffix",
             "--no-timestamp", "--no-plots"],
        ):
            args = parse_arguments()
        assert args.duration == 200
        assert args.prefix == "test"
        assert args.user_suffix == "mysuffix"
        assert args.no_timestamp is True
        assert args.no_plots is True

    def test_parse_checkpoint(self):
        with patch("sys.argv", ["prog", "--load-checkpoint", "/path/to/ckpt.json"]):
            args = parse_arguments()
        assert args.load_checkpoint == "/path/to/ckpt.json"


class TestRunMFCSimulation:
    def test_run_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_mfc_simulation(
                duration_hours=0.05,
                output_dir=tmpdir,
                verbose=False,
            )
        assert isinstance(results, dict)
        assert len(results["time_hours"]) > 0

    def test_run_with_config(self):
        from config.qlearning_config import DEFAULT_QLEARNING_CONFIG
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_mfc_simulation(
                duration_hours=0.05,
                output_dir=tmpdir,
                config=DEFAULT_QLEARNING_CONFIG,
                verbose=False,
            )
        assert isinstance(results, dict)

    def test_run_verbose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_mfc_simulation(
                duration_hours=0.05,
                output_dir=tmpdir,
                verbose=True,
                user_suffix="test",
            )
        assert isinstance(results, dict)

    def test_run_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subdir")
            results = run_mfc_simulation(
                duration_hours=0.05,
                output_dir=new_dir,
                verbose=False,
            )
        assert isinstance(results, dict)
