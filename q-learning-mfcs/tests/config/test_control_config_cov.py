"""Tests for control_config module - 98%+ coverage target."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from config.control_config import (
    AdvancedControlConfig,
    ControlMode,
    ControlSystemConfig,
    FlowControlConfig,
    PIDConfig,
    RecirculationConfig,
    SubstrateControlConfig,
    SubstrateControlStrategy,
    get_aggressive_control_config,
    get_conservative_control_config,
    get_precision_control_config,
    validate_control_config,
)


class TestControlMode:
    def test_enum_values(self):
        assert ControlMode.MANUAL.value == "manual"
        assert ControlMode.AUTOMATIC.value == "automatic"
        assert ControlMode.CASCADE.value == "cascade"
        assert ControlMode.FEEDFORWARD.value == "feedforward"
        assert ControlMode.ADAPTIVE.value == "adaptive"


class TestSubstrateControlStrategy:
    def test_enum_values(self):
        assert SubstrateControlStrategy.CONCENTRATION_BASED.value == "concentration_based"
        assert SubstrateControlStrategy.UTILIZATION_BASED.value == "utilization_based"
        assert SubstrateControlStrategy.POWER_BASED.value == "power_based"
        assert SubstrateControlStrategy.MULTI_OBJECTIVE.value == "multi_objective"


class TestPIDConfig:
    def test_default_values(self):
        pid = PIDConfig()
        assert pid.kp == 2.0
        assert pid.ki == 0.1
        assert pid.kd == 0.5
        assert pid.output_min == 0.0
        assert pid.output_max == 100.0
        assert pid.enable_anti_windup is True
        assert pid.enable_bumpless_transfer is True
        assert pid.dead_band == 0.0
        assert pid.enable_gain_scheduling is False

    def test_custom_values(self):
        pid = PIDConfig(kp=3.0, ki=0.5, kd=1.0)
        assert pid.kp == 3.0
        assert pid.ki == 0.5
        assert pid.kd == 1.0

    def test_negative_kp_raises(self):
        with pytest.raises(ValueError, match="Proportional gain"):
            PIDConfig(kp=-1.0)

    def test_negative_ki_raises(self):
        with pytest.raises(ValueError, match="Integral gain"):
            PIDConfig(ki=-0.1)

    def test_negative_kd_raises(self):
        with pytest.raises(ValueError, match="Derivative gain"):
            PIDConfig(kd=-0.5)

    def test_output_min_gte_max_raises(self):
        with pytest.raises(ValueError, match="Output minimum must be less"):
            PIDConfig(output_min=100.0, output_max=100.0)

    def test_output_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="Output minimum must be less"):
            PIDConfig(output_min=200.0, output_max=100.0)

    def test_zero_gains_valid(self):
        pid = PIDConfig(kp=0.0, ki=0.0, kd=0.0)
        assert pid.kp == 0.0

    def test_operating_point_gains(self):
        pid = PIDConfig(
            enable_gain_scheduling=True,
            operating_point_gains={0.5: (1.0, 0.1, 0.5)},
        )
        assert pid.enable_gain_scheduling is True
        assert 0.5 in pid.operating_point_gains


class TestFlowControlConfig:
    def test_default_values(self):
        fc = FlowControlConfig()
        assert fc.min_flow_rate == 5.0
        assert fc.max_flow_rate == 50.0
        assert fc.nominal_flow_rate == 15.0
        assert fc.pump_efficiency == 0.95
        assert fc.control_mode == ControlMode.AUTOMATIC
        assert fc.emergency_shutdown_flow == 0.0

    def test_flow_pid_is_configured(self):
        fc = FlowControlConfig()
        assert fc.flow_pid.kp == 1.5
        assert fc.flow_pid.ki == 0.2
        assert fc.flow_pid.kd == 0.1


class TestSubstrateControlConfig:
    def test_default_values(self):
        sc = SubstrateControlConfig()
        assert sc.target_outlet_concentration == 12.0
        assert sc.target_reservoir_concentration == 20.0
        assert sc.control_strategy == SubstrateControlStrategy.CONCENTRATION_BASED
        assert sc.starvation_threshold_warning == 5.0
        assert sc.starvation_threshold_critical == 2.0
        assert sc.concentration_weight == 0.6
        assert sc.utilization_weight == 0.3
        assert sc.power_weight == 0.1

    def test_outlet_pid(self):
        sc = SubstrateControlConfig()
        assert sc.outlet_pid.kp == 2.0

    def test_reservoir_pid(self):
        sc = SubstrateControlConfig()
        assert sc.reservoir_pid.kp == 1.0


class TestRecirculationConfig:
    def test_default_values(self):
        rc = RecirculationConfig()
        assert rc.reservoir_volume == 1.0
        assert rc.initial_substrate_concentration == 20.0
        assert rc.mixing_time_constant == 0.1
        assert rc.pump_efficiency == 0.95
        assert rc.min_recirculation_rate == 10.0
        assert rc.max_recirculation_rate == 100.0
        assert rc.mixing_efficiency_threshold == 0.7
        assert rc.excellent_mixing_threshold == 0.95
        assert rc.poor_mixing_factor == 0.8
        assert rc.excellent_mixing_factor == 1.1


class TestAdvancedControlConfig:
    def test_default_values(self):
        ac = AdvancedControlConfig()
        assert ac.learning_rate == 0.0987
        assert ac.discount_factor == 0.9517
        assert ac.epsilon == 0.3702
        assert ac.power_bins == 8
        assert len(ac.flow_rate_adjustments) == 10
        assert len(ac.substrate_actions) == 7
        assert len(ac.time_phase_hours) == 4

    def test_weights_sum(self):
        ac = AdvancedControlConfig()
        total = (
            ac.power_objective_weight
            + ac.biofilm_health_weight
            + ac.sensor_agreement_weight
            + ac.stability_weight
        )
        assert abs(total - 1.0) < 0.01


class TestControlSystemConfig:
    def test_default_values(self):
        csc = ControlSystemConfig()
        assert isinstance(csc.flow_control, FlowControlConfig)
        assert isinstance(csc.substrate_control, SubstrateControlConfig)
        assert isinstance(csc.recirculation, RecirculationConfig)
        assert isinstance(csc.advanced_control, AdvancedControlConfig)
        assert csc.control_update_interval == 60.0
        assert csc.data_logging_interval == 10.0
        assert csc.emergency_shutdown_enabled is True
        assert csc.fault_tolerance_enabled is True
        assert csc.configuration_name == "default_control"
        assert csc.configuration_version == "1.0.0"
        assert csc.created_by == "system"


class TestGetConservativeControlConfig:
    def test_returns_config(self):
        config = get_conservative_control_config()
        assert isinstance(config, ControlSystemConfig)
        assert config.configuration_name == "conservative_control"
        assert config.flow_control.flow_pid.kp == 1.0
        assert config.advanced_control.learning_rate == 0.05
        assert config.advanced_control.epsilon == 0.2
        assert config.flow_control.max_flow_rate_change == 5.0


class TestGetAggressiveControlConfig:
    def test_returns_config(self):
        config = get_aggressive_control_config()
        assert isinstance(config, ControlSystemConfig)
        assert config.configuration_name == "aggressive_control"
        assert config.flow_control.flow_pid.kp == 3.0
        assert config.advanced_control.learning_rate == 0.2
        assert config.substrate_control.max_addition_rate == 80.0


class TestGetPrecisionControlConfig:
    def test_returns_config(self):
        config = get_precision_control_config()
        assert isinstance(config, ControlSystemConfig)
        assert config.configuration_name == "precision_control"
        assert config.flow_control.flow_rate_tolerance == 0.1
        assert config.control_update_interval == 30.0
        assert config.data_logging_interval == 5.0


class TestValidateControlConfig:
    def test_valid_default_config(self):
        config = ControlSystemConfig()
        assert validate_control_config(config) is True

    def test_invalid_flow_rates(self):
        config = ControlSystemConfig()
        config.flow_control.min_flow_rate = 100.0
        config.flow_control.max_flow_rate = 10.0
        with pytest.raises(ValueError, match="Minimum flow rate"):
            validate_control_config(config)

    def test_nominal_flow_below_min(self):
        config = ControlSystemConfig()
        config.flow_control.nominal_flow_rate = 1.0
        with pytest.raises(ValueError, match="Nominal flow rate"):
            validate_control_config(config)

    def test_nominal_flow_above_max(self):
        config = ControlSystemConfig()
        config.flow_control.nominal_flow_rate = 999.0
        with pytest.raises(ValueError, match="Nominal flow rate"):
            validate_control_config(config)

    def test_invalid_addition_rates(self):
        config = ControlSystemConfig()
        config.substrate_control.min_addition_rate = 100.0
        config.substrate_control.max_addition_rate = 10.0
        with pytest.raises(ValueError, match="Minimum addition rate"):
            validate_control_config(config)

    def test_invalid_starvation_thresholds(self):
        config = ControlSystemConfig()
        config.substrate_control.starvation_threshold_critical = 10.0
        config.substrate_control.starvation_threshold_warning = 5.0
        with pytest.raises(ValueError, match="Critical starvation threshold"):
            validate_control_config(config)

    def test_invalid_reservoir_volume(self):
        config = ControlSystemConfig()
        config.recirculation.reservoir_volume = 0.0
        with pytest.raises(ValueError, match="Reservoir volume"):
            validate_control_config(config)

    def test_invalid_pump_efficiency_zero(self):
        config = ControlSystemConfig()
        config.recirculation.pump_efficiency = 0.0
        with pytest.raises(ValueError, match="Pump efficiency"):
            validate_control_config(config)

    def test_invalid_pump_efficiency_over_one(self):
        config = ControlSystemConfig()
        config.recirculation.pump_efficiency = 1.5
        with pytest.raises(ValueError, match="Pump efficiency"):
            validate_control_config(config)

    def test_invalid_learning_rate_zero(self):
        config = ControlSystemConfig()
        config.advanced_control.learning_rate = 0.0
        with pytest.raises(ValueError, match="Learning rate"):
            validate_control_config(config)

    def test_invalid_learning_rate_over_one(self):
        config = ControlSystemConfig()
        config.advanced_control.learning_rate = 1.5
        with pytest.raises(ValueError, match="Learning rate"):
            validate_control_config(config)

    def test_invalid_discount_factor(self):
        config = ControlSystemConfig()
        config.advanced_control.discount_factor = 0.0
        with pytest.raises(ValueError, match="Discount factor"):
            validate_control_config(config)

    def test_invalid_weights_sum(self):
        config = ControlSystemConfig()
        config.advanced_control.power_objective_weight = 0.1
        config.advanced_control.biofilm_health_weight = 0.1
        config.advanced_control.sensor_agreement_weight = 0.1
        config.advanced_control.stability_weight = 0.1
        with pytest.raises(ValueError, match="Multi-objective weights"):
            validate_control_config(config)

    def test_valid_conservative_config(self):
        config = get_conservative_control_config()
        assert validate_control_config(config) is True

    def test_valid_aggressive_config(self):
        config = get_aggressive_control_config()
        assert validate_control_config(config) is True

    def test_valid_precision_config(self):
        config = get_precision_control_config()
        assert validate_control_config(config) is True
