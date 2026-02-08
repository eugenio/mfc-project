"""Tests for controller_models/real_time_controller.py - coverage target 98%+."""
import sys
import os
import time

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from controller_models.real_time_controller import (
    RealTimeController, TimingConstraints, ControlTask, ControlLoop,
    ControllerMode, TaskPriority, TimingAnalyzer,
)


@pytest.fixture
def timing():
    return TimingConstraints(
        control_loop_period_ms=100.0, max_jitter_ms=50.0, deadline_violation_limit=5,
        interrupt_response_time_us=10.0, context_switch_time_us=5.0,
        worst_case_execution_time_ms=80.0, watchdog_timeout_ms=5000.0,
        safety_stop_timeout_ms=100.0, sensor_timeout_ms=500.0,
    )


@pytest.fixture
def controller(timing):
    ctrl = RealTimeController(timing)
    yield ctrl
    ctrl.running = False


class TestSchedulerControlLoops:
    def test_scheduler_executes_control_loops(self, controller):
        controller.mode = ControllerMode.AUTOMATIC
        controller.running = True
        cc = [0]
        def stop():
            cc[0] += 1; controller.running = False
        with patch.object(controller, "_execute_scheduled_tasks"):
            with patch.object(controller, "_execute_control_loops", side_effect=stop):
                with patch.object(controller, "_update_performance_metrics"):
                    with patch("time.sleep"):
                        controller._scheduler_loop()
        assert cc[0] >= 1


class TestSchedulerLoopException:
    def test_scheduler_exception_triggers_fault(self, controller):
        controller.running = True; controller.mode = ControllerMode.MANUAL
        cc = [0]
        def fail(*a, **kw):
            cc[0] += 1
            if cc[0] == 1: raise RuntimeError("err")
            controller.running = False
        with patch.object(controller, "_execute_scheduled_tasks", side_effect=fail):
            with patch("time.sleep"):
                with patch("time.time", return_value=time.time()):
                    controller._scheduler_loop()
        assert "SCHEDULER_ERROR" in controller.fault_flags


class TestExecuteScheduledTasks:
    def test_task_skipped_insufficient_time(self, controller):
        task = ControlTask(task_id="slow", priority=TaskPriority.LOW, period_ms=10.0,
            deadline_ms=5.0, wcet_ms=9999.0, callback=lambda: None, next_execution=0, enabled=True)
        controller.tasks["slow"] = task
        controller._execute_scheduled_tasks(time.time() - 1)

    def test_task_deadline_violation(self, controller):
        task = ControlTask(task_id="dl", priority=TaskPriority.HIGH, period_ms=100.0,
            deadline_ms=0.0001, wcet_ms=0.001, callback=lambda: time.sleep(0.001),
            next_execution=0, enabled=True)
        controller.tasks["dl"] = task
        now = time.time()
        controller._execute_scheduled_tasks(now)
        assert task.deadline_violations >= 1

    def test_task_execution_exception(self, controller):
        def bad(): raise RuntimeError("fail")
        task = ControlTask(task_id="bad", priority=TaskPriority.HIGH, period_ms=100.0,
            deadline_ms=50.0, wcet_ms=0.001, callback=bad, next_execution=0, enabled=True)
        controller.tasks["bad"] = task
        now = time.time()
        controller._execute_scheduled_tasks(now)
        assert task.deadline_violations >= 1


class TestControlLoopQlearning:
    def test_qlearning_control_loop(self, controller):
        loop = ControlLoop(loop_id="ql", input_channels=[0], output_channels=[0],
            control_algorithm="Q-learning", setpoint=25.0, gains={}, limits={"output":(-10,10)})
        controller.control_loops["ql"] = loop
        controller.sensor_data[0] = 23.0
        controller._execute_control_loops()

    def test_unknown_control_algorithm(self, controller):
        loop = ControlLoop(loop_id="uk", input_channels=[0], output_channels=[0],
            control_algorithm="Unknown", setpoint=25.0, gains={}, limits={})
        controller.control_loops["uk"] = loop
        controller.sensor_data[0] = 23.0
        controller._execute_control_loops()


class TestControlLoopException:
    def test_control_loop_exception_triggers_fault(self, controller):
        loop = ControlLoop(loop_id="el", input_channels=[0], output_channels=[0],
            control_algorithm="PID", setpoint=25.0, gains={"kp":1.0}, limits={})
        controller.control_loops["el"] = loop
        with patch.object(controller, "_execute_pid_control", side_effect=RuntimeError("f")):
            controller._execute_control_loops()
        assert any("CONTROL_LOOP_ERROR" in f for f in controller.fault_flags)


class TestQlearningControl:
    def test_no_action(self, controller):
        loop = ControlLoop(loop_id="q1", input_channels=[0], output_channels=[0],
            control_algorithm="Q-learning", setpoint=25.0, gains={}, limits={})
        assert controller._execute_qlearning_control(loop, {0: 25.0}) == 0.0

    def test_increase(self, controller):
        loop = ControlLoop(loop_id="q2", input_channels=[0], output_channels=[0],
            control_algorithm="Q-learning", setpoint=25.0, gains={}, limits={})
        assert controller._execute_qlearning_control(loop, {0: 20.0}) == 1.0

    def test_decrease(self, controller):
        loop = ControlLoop(loop_id="q3", input_channels=[0], output_channels=[0],
            control_algorithm="Q-learning", setpoint=25.0, gains={}, limits={})
        assert controller._execute_qlearning_control(loop, {0: 30.0}) == -1.0


class TestWatchdogTimeout:
    def test_watchdog_timeout_triggers_fault(self, controller):
        controller.running = True
        controller.watchdog_last_pet = time.time() - 100
        with patch("time.sleep"):
            controller._watchdog_loop()
        assert "WATCHDOG_TIMEOUT" in controller.fault_flags


class TestPerformanceMetrics:
    def test_update_performance_metrics_no_psutil(self, controller):
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                controller._update_performance_metrics()
                assert controller.memory_usage == 50.0
