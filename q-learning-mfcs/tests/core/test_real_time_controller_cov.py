"""Tests for real_time_controller.py - 98%+ coverage target.

Covers RealTimeController, TimingConstraints, ControlTask, ControlLoop,
TimingAnalyzer, ControllerMode, and helper functions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.real_time_controller import (
    ControllerMeasurement,
    ControllerMode,
    ControlLoop,
    ControlTask,
    RealTimeController,
    TaskPriority,
    TimingAnalyzer,
    TimingConstraints,
    create_standard_real_time_controllers,
)


def _make_timing():
    return TimingConstraints(
        control_loop_period_ms=10.0,
        max_jitter_ms=2.0,
        deadline_violation_limit=5,
        interrupt_response_time_us=100.0,
        context_switch_time_us=50.0,
        worst_case_execution_time_ms=5.0,
        watchdog_timeout_ms=5000.0,
        safety_stop_timeout_ms=100.0,
        sensor_timeout_ms=500.0,
    )


def _make_task(task_id="task1", period=10.0, callback=None):
    if callback is None:
        callback = lambda: None
    return ControlTask(
        task_id=task_id, priority=TaskPriority.MEDIUM,
        period_ms=period, deadline_ms=period * 0.8,
        wcet_ms=period * 0.5, callback=callback,
    )


def _make_loop(loop_id="loop1", algo="PID"):
    return ControlLoop(
        loop_id=loop_id, input_channels=[0],
        output_channels=[0], control_algorithm=algo,
        setpoint=25.0, gains={"kp": 1.0, "ki": 0.1, "kd": 0.05},
        limits={"output": (-10.0, 10.0), "integral": (-5.0, 5.0)},
    )


class TestEnums:
    def test_controller_mode(self):
        assert ControllerMode.MANUAL.value == "manual"
        assert ControllerMode.AUTOMATIC.value == "automatic"
        assert ControllerMode.LEARNING.value == "learning"
        assert ControllerMode.SAFETY.value == "safety"
        assert ControllerMode.MAINTENANCE.value == "maintenance"

    def test_task_priority(self):
        assert TaskPriority.CRITICAL.value == 0
        assert TaskPriority.HIGH.value == 1
        assert TaskPriority.MEDIUM.value == 2
        assert TaskPriority.LOW.value == 3


class TestDataclasses:
    def test_timing_constraints(self):
        tc = _make_timing()
        assert tc.control_loop_period_ms == 10.0

    def test_control_task(self):
        task = _make_task()
        assert task.task_id == "task1"
        assert task.enabled is True
        assert task.deadline_violations == 0

    def test_control_loop(self):
        loop = _make_loop()
        assert loop.loop_id == "loop1"
        assert loop.enabled is True

    def test_controller_measurement(self):
        m = ControllerMeasurement(
            timestamp=1000.0, mode=ControllerMode.AUTOMATIC,
            cpu_utilization_pct=50.0, memory_usage_mb=100.0,
            control_loop_period_actual_ms=10.0, jitter_ms=0.5,
            deadline_violations_recent=0, interrupt_count=0,
            task_execution_times={}, active_control_loops=[],
            safety_state="NORMAL", fault_flags=[],
        )
        assert m.mode == ControllerMode.AUTOMATIC


class TestRealTimeControllerInit:
    def test_init(self):
        tc = _make_timing()
        rtc = RealTimeController(tc)
        assert rtc.mode == ControllerMode.MANUAL
        assert rtc.running is False
        assert rtc.safety_state == "NORMAL"


class TestTaskManagement:
    def test_add_task(self):
        rtc = RealTimeController(_make_timing())
        task = _make_task()
        rtc.add_task(task)
        assert "task1" in rtc.tasks

    def test_remove_task(self):
        rtc = RealTimeController(_make_timing())
        task = _make_task()
        rtc.add_task(task)
        rtc.remove_task("task1")
        assert "task1" not in rtc.tasks

    def test_remove_nonexistent_task(self):
        rtc = RealTimeController(_make_timing())
        rtc.remove_task("nonexistent")


class TestControlLoopManagement:
    def test_add_control_loop(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        assert "loop1" in rtc.control_loops


class TestModeManagement:
    def test_set_mode_automatic(self):
        rtc = RealTimeController(_make_timing())
        rtc.set_mode(ControllerMode.AUTOMATIC)
        assert rtc.mode == ControllerMode.AUTOMATIC

    def test_set_mode_safety(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        rtc.actuator_outputs[0] = 5.0
        rtc.set_mode(ControllerMode.SAFETY)
        assert rtc.mode == ControllerMode.SAFETY
        assert rtc.safety_state == "EMERGENCY_STOP"
        assert rtc.actuator_outputs[0] == 0.0
        assert rtc.control_loops["loop1"].enabled is False

    def test_set_mode_learning(self):
        rtc = RealTimeController(_make_timing())
        rtc.set_mode(ControllerMode.LEARNING)
        assert rtc.safety_state == "LEARNING"


class TestStartStop:
    def test_start_stop(self):
        rtc = RealTimeController(_make_timing())
        rtc.start()
        assert rtc.running is True
        time.sleep(0.05)
        rtc.stop()
        assert rtc.running is False

    def test_start_already_running(self):
        rtc = RealTimeController(_make_timing())
        rtc.start()
        rtc.start()  # Should warn
        rtc.stop()


class TestSensorAndActuator:
    def test_update_sensor_data(self):
        rtc = RealTimeController(_make_timing())
        rtc.update_sensor_data(0, 25.5)
        assert rtc.sensor_data[0] == 25.5

    def test_get_actuator_output(self):
        rtc = RealTimeController(_make_timing())
        assert rtc.get_actuator_output(0) == 0.0
        rtc.actuator_outputs[0] = 3.5
        assert rtc.get_actuator_output(0) == 3.5


class TestControlLoopSetpoint:
    def test_set_setpoint(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        rtc.set_control_loop_setpoint("loop1", 30.0)
        assert rtc.control_loops["loop1"].setpoint == 30.0

    def test_set_setpoint_nonexistent(self):
        rtc = RealTimeController(_make_timing())
        rtc.set_control_loop_setpoint("nonexistent", 30.0)

    def test_enable_loop(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        rtc.enable_control_loop("loop1", False)
        assert rtc.control_loops["loop1"].enabled is False

    def test_enable_loop_nonexistent(self):
        rtc = RealTimeController(_make_timing())
        rtc.enable_control_loop("nonexistent", True)


class TestPIDControl:
    def test_pid_basic(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        inputs = {0: 20.0}
        output = rtc._execute_pid_control(loop, inputs)
        assert output != 0.0

    def test_pid_empty_inputs(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        output = rtc._execute_pid_control(loop, {})
        assert output == 0.0

    def test_pid_integral_windup(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        loop.error_integral = 100.0
        inputs = {0: 0.0}
        rtc._execute_pid_control(loop, inputs)
        assert loop.error_integral <= 5.0

    def test_pid_derivative_filter(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        loop.gains["derivative_filter"] = 0.5
        inputs = {0: 20.0}
        rtc._execute_pid_control(loop, inputs)
        assert hasattr(loop, "_filtered_derivative")


class TestQLearningControl:
    def test_qlearning_basic(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop(algo="Q-learning")
        inputs = {0: 20.0}
        output = rtc._execute_qlearning_control(loop, inputs)
        assert output in [-1.0, 0.0, 1.0]

    def test_qlearning_empty_inputs(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop(algo="Q-learning")
        output = rtc._execute_qlearning_control(loop, {})
        assert output == 0.0

    def test_qlearning_close_to_setpoint(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop(algo="Q-learning")
        loop.setpoint = 25.0
        inputs = {0: 24.95}
        output = rtc._execute_qlearning_control(loop, inputs)
        assert output == 0.0

    def test_qlearning_below_setpoint(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop(algo="Q-learning")
        loop.setpoint = 25.0
        inputs = {0: 20.0}
        output = rtc._execute_qlearning_control(loop, inputs)
        assert output == 1.0


class TestExecuteControlLoops:
    def test_execute_pid_loop(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        rtc.sensor_data[0] = 20.0
        rtc._execute_control_loops()
        assert rtc.actuator_outputs[0] != 0.0

    def test_execute_disabled_loop(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        loop.enabled = False
        rtc.add_control_loop(loop)
        rtc._execute_control_loops()
        assert 0 not in rtc.actuator_outputs

    def test_execute_missing_sensor(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        # No sensor data set
        rtc._execute_control_loops()

    def test_execute_unknown_algorithm(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop(algo="Unknown")
        rtc.add_control_loop(loop)
        rtc.sensor_data[0] = 20.0
        rtc._execute_control_loops()

    def test_execute_with_output_limits(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        loop.limits["output"] = (-1.0, 1.0)
        rtc.add_control_loop(loop)
        rtc.sensor_data[0] = 0.0
        rtc._execute_control_loops()
        assert -1.0 <= rtc.actuator_outputs[0] <= 1.0


class TestTimingViolation:
    def test_handle_timing_violation(self):
        rtc = RealTimeController(_make_timing())
        rtc.deadline_violations = 10
        rtc._handle_timing_violation()
        assert rtc.mode == ControllerMode.SAFETY

    def test_handle_timing_violation_below_limit(self):
        rtc = RealTimeController(_make_timing())
        rtc.deadline_violations = 1
        rtc._handle_timing_violation()
        assert rtc.mode == ControllerMode.MANUAL


class TestFaultHandling:
    def test_handle_fault(self):
        rtc = RealTimeController(_make_timing())
        rtc._handle_controller_fault("SOME_ERROR")
        assert "SOME_ERROR" in rtc.fault_flags

    def test_handle_critical_fault(self):
        rtc = RealTimeController(_make_timing())
        rtc._handle_controller_fault("SCHEDULER_ERROR")
        assert rtc.mode == ControllerMode.SAFETY

    def test_handle_watchdog_fault(self):
        rtc = RealTimeController(_make_timing())
        rtc._handle_controller_fault("WATCHDOG_TIMEOUT")
        assert rtc.mode == ControllerMode.SAFETY


class TestUpdatePerformanceMetrics:
    def test_with_psutil(self):
        rtc = RealTimeController(_make_timing())
        rtc._update_performance_metrics()
        assert rtc.cpu_utilization >= 0.0

    def test_without_psutil(self):
        rtc = RealTimeController(_make_timing())
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                rtc._update_performance_metrics()
        assert rtc.memory_usage == 50.0


class TestExecuteScheduledTasks:
    def test_execute_tasks(self):
        rtc = RealTimeController(_make_timing())
        executed = []
        task = _make_task(callback=lambda: executed.append(True))
        task.next_execution = 0  # Execute immediately
        rtc.add_task(task)
        rtc._execute_scheduled_tasks(time.time())
        assert len(executed) == 1

    def test_task_deadline_violation(self):
        rtc = RealTimeController(_make_timing())

        def slow_task():
            time.sleep(0.02)

        task = _make_task(callback=slow_task)
        task.deadline_ms = 0.001  # Very tight deadline
        task.next_execution = 0
        rtc.add_task(task)
        rtc._execute_scheduled_tasks(time.time())
        assert task.deadline_violations > 0

    def test_task_exception(self):
        rtc = RealTimeController(_make_timing())

        def failing_task():
            raise RuntimeError("fail")

        task = _make_task(callback=failing_task)
        task.next_execution = 0
        rtc.add_task(task)
        rtc._execute_scheduled_tasks(time.time())
        assert task.deadline_violations > 0

    def test_skip_insufficient_time(self):
        rtc = RealTimeController(_make_timing())
        rtc.timing_constraints.control_loop_period_ms = 0.001
        task = _make_task()
        task.next_execution = 0
        task.wcet_ms = 100.0  # Very long WCET
        rtc.add_task(task)
        current = time.time() - 10  # Fake past time
        rtc._execute_scheduled_tasks(current)


class TestGetMeasurement:
    def test_basic_measurement(self):
        rtc = RealTimeController(_make_timing())
        m = rtc.get_measurement()
        assert isinstance(m, ControllerMeasurement)
        assert m.mode == ControllerMode.MANUAL
        assert m.safety_state == "NORMAL"

    def test_measurement_with_tasks(self):
        rtc = RealTimeController(_make_timing())
        task = _make_task()
        task.execution_history = [1.0, 2.0, 3.0]
        task.deadline_violations = 2
        rtc.add_task(task)
        m = rtc.get_measurement()
        assert "task1" in m.task_execution_times

    def test_measurement_with_loops(self):
        rtc = RealTimeController(_make_timing())
        loop = _make_loop()
        rtc.add_control_loop(loop)
        m = rtc.get_measurement()
        assert "loop1" in m.active_control_loops

    def test_measurement_with_jitter(self):
        rtc = RealTimeController(_make_timing())
        rtc.jitter_history.extend([0.1, 0.2, 0.3])
        rtc.loop_execution_times.extend([1.0, 2.0, 3.0])
        m = rtc.get_measurement()
        assert m.jitter_ms > 0


class TestTimingAnalyzer:
    def test_record_execution(self):
        ta = TimingAnalyzer()
        ta.record_task_execution("task1", 1.5)
        ta.record_task_execution("task1", 2.0)
        assert len(ta.task_executions["task1"]) == 2

    def test_get_analysis_empty(self):
        ta = TimingAnalyzer()
        analysis = ta.get_analysis()
        assert analysis["tasks"] == {}

    def test_get_analysis_with_data(self):
        ta = TimingAnalyzer()
        for i in range(10):
            ta.record_task_execution("task1", float(i))
        analysis = ta.get_analysis()
        assert "task1" in analysis["tasks"]
        assert "mean_ms" in analysis["tasks"]["task1"]
        assert "overall" in analysis
        assert analysis["overall"]["total_executions"] == 10


class TestGetTimingAnalysis:
    def test_get_timing_analysis(self):
        rtc = RealTimeController(_make_timing())
        result = rtc.get_timing_analysis()
        assert isinstance(result, dict)


class TestCreateStandard:
    def test_creates_two_configs(self):
        controllers = create_standard_real_time_controllers()
        assert "high_performance" in controllers
        assert "low_power" in controllers

    def test_hp_config(self):
        controllers = create_standard_real_time_controllers()
        hp = controllers["high_performance"]
        assert hp.timing_constraints.control_loop_period_ms == 1.0

    def test_lp_config(self):
        controllers = create_standard_real_time_controllers()
        lp = controllers["low_power"]
        assert lp.timing_constraints.control_loop_period_ms == 10.0
