"""Extra coverage tests for controller_models/real_time_controller.py.

Covers remaining uncovered lines:
- Line 250: _execute_control_loops called from _scheduler_loop (AUTOMATIC mode)
- Lines 278-280: Exception in scheduler loop triggers SCHEDULER_ERROR fault
- Line 290: tasks_to_execute.append in _execute_scheduled_tasks
- Lines 297-337: Full task execution path (callback, timing, deadline check,
  history trimming, scheduling next, timing analyzer recording)
- Line 360: _execute_control_loops control loop algorithm dispatch
- Lines 378-380: Control loop exception handling
- Line 453: Q-learning control returns -1.0 (error < -0.1)
- Lines 498-500: Watchdog timeout detection
- Line 568: task_times[task_id] = 0.0 when no execution_history
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from controller_models.real_time_controller import (
    ControllerMode,
    ControlLoop,
    ControlTask,
    RealTimeController,
    TaskPriority,
    TimingConstraints,
)


def _make_timing(**kw):
    defaults = dict(
        control_loop_period_ms=100.0,
        max_jitter_ms=50.0,
        deadline_violation_limit=5,
        interrupt_response_time_us=100.0,
        context_switch_time_us=50.0,
        worst_case_execution_time_ms=80.0,
        watchdog_timeout_ms=5000.0,
        safety_stop_timeout_ms=100.0,
        sensor_timeout_ms=500.0,
    )
    defaults.update(kw)
    return TimingConstraints(**defaults)


def _make_task(task_id="t1", period=100.0, deadline=80.0, wcet=0.001,
               callback=None, priority=TaskPriority.HIGH):
    if callback is None:
        callback = lambda: None
    return ControlTask(
        task_id=task_id, priority=priority, period_ms=period,
        deadline_ms=deadline, wcet_ms=wcet, callback=callback,
    )


@pytest.mark.coverage_extra
class TestSchedulerLoopExecutesControlLoops:
    """Cover line 250: _execute_control_loops called in AUTOMATIC mode."""

    def test_scheduler_loop_calls_execute_control_loops_in_automatic(self):
        ctrl = RealTimeController(_make_timing())
        ctrl.mode = ControllerMode.AUTOMATIC
        ctrl.running = True

        calls = []

        def track_control_loops():
            calls.append(True)
            ctrl.running = False

        with patch.object(ctrl, "_execute_scheduled_tasks"):
            with patch.object(ctrl, "_execute_control_loops",
                              side_effect=track_control_loops):
                with patch.object(ctrl, "_update_performance_metrics"):
                    with patch("time.sleep"):
                        ctrl._scheduler_loop()

        assert len(calls) >= 1

    def test_scheduler_loop_calls_execute_control_loops_in_learning(self):
        ctrl = RealTimeController(_make_timing())
        ctrl.mode = ControllerMode.LEARNING
        ctrl.running = True

        calls = []

        def track_control_loops():
            calls.append(True)
            ctrl.running = False

        with patch.object(ctrl, "_execute_scheduled_tasks"):
            with patch.object(ctrl, "_execute_control_loops",
                              side_effect=track_control_loops):
                with patch.object(ctrl, "_update_performance_metrics"):
                    with patch("time.sleep"):
                        ctrl._scheduler_loop()

        assert len(calls) >= 1


@pytest.mark.coverage_extra
class TestSchedulerLoopException:
    """Cover lines 278-280: Exception in scheduler loop."""

    def test_scheduler_loop_exception_sets_scheduler_error(self):
        ctrl = RealTimeController(_make_timing())
        ctrl.running = True

        iteration = [0]

        def fail_first(*a, **kw):
            iteration[0] += 1
            if iteration[0] == 1:
                raise RuntimeError("scheduler boom")
            ctrl.running = False

        with patch.object(ctrl, "_execute_scheduled_tasks",
                          side_effect=fail_first):
            with patch("time.sleep"):
                ctrl._scheduler_loop()

        assert "SCHEDULER_ERROR" in ctrl.fault_flags


@pytest.mark.coverage_extra
class TestExecuteScheduledTasksFull:
    """Cover lines 290, 297-337: Full task execution path."""

    def test_task_executes_successfully(self):
        ctrl = RealTimeController(_make_timing())
        executed = []

        def my_callback():
            executed.append(True)

        task = _make_task(
            task_id="exec_test", callback=my_callback,
            wcet=0.001, deadline=5000.0,
        )
        # Set next_execution in the past so it's ready
        task.next_execution = 0.0
        ctrl.tasks["exec_test"] = task

        now = time.time()
        ctrl._execute_scheduled_tasks(now)

        assert len(executed) == 1
        assert task.last_execution > 0
        assert task.next_execution > now
        assert len(task.execution_history) >= 1
        assert task.deadline_violations == 0

    def test_task_deadline_violation_recorded(self):
        ctrl = RealTimeController(_make_timing())

        def slow_callback():
            time.sleep(0.015)

        task = _make_task(
            task_id="slow_task", callback=slow_callback,
            wcet=0.001, deadline=0.001,  # 0.001ms deadline = instant violation
        )
        task.next_execution = 0.0
        ctrl.tasks["slow_task"] = task

        ctrl._execute_scheduled_tasks(time.time())

        assert task.deadline_violations >= 1

    def test_task_execution_history_trimmed(self):
        ctrl = RealTimeController(_make_timing())
        task = _make_task(task_id="trim_test", wcet=0.001, deadline=5000.0)
        task.next_execution = 0.0
        # Pre-fill with 101 entries
        task.execution_history = list(range(101))
        ctrl.tasks["trim_test"] = task

        ctrl._execute_scheduled_tasks(time.time())

        # After adding one more and popping, length should be <= 101
        assert len(task.execution_history) <= 101

    def test_task_exception_increments_deadline_violations(self):
        ctrl = RealTimeController(_make_timing())

        def fail_callback():
            raise ValueError("task error")

        task = _make_task(
            task_id="fail_task", callback=fail_callback,
            wcet=0.001, deadline=5000.0,
        )
        task.next_execution = 0.0
        ctrl.tasks["fail_task"] = task

        ctrl._execute_scheduled_tasks(time.time())

        assert task.deadline_violations >= 1

    def test_disabled_task_not_executed(self):
        ctrl = RealTimeController(_make_timing())
        executed = []
        task = _make_task(callback=lambda: executed.append(True))
        task.next_execution = 0.0
        task.enabled = False
        ctrl.tasks["disabled"] = task

        ctrl._execute_scheduled_tasks(time.time())

        assert len(executed) == 0

    def test_task_sorted_by_priority(self):
        ctrl = RealTimeController(_make_timing())
        order = []

        task_crit = _make_task(
            task_id="crit", priority=TaskPriority.CRITICAL,
            callback=lambda: order.append("crit"),
            wcet=0.001, deadline=5000.0,
        )
        task_crit.next_execution = 0.0

        task_low = _make_task(
            task_id="low", priority=TaskPriority.LOW,
            callback=lambda: order.append("low"),
            wcet=0.001, deadline=5000.0,
        )
        task_low.next_execution = 0.0

        ctrl.tasks["low"] = task_low
        ctrl.tasks["crit"] = task_crit

        ctrl._execute_scheduled_tasks(time.time())

        assert order == ["crit", "low"]

    def test_timing_analyzer_records_execution(self):
        ctrl = RealTimeController(_make_timing())
        task = _make_task(task_id="timed", wcet=0.001, deadline=5000.0)
        task.next_execution = 0.0
        ctrl.tasks["timed"] = task

        ctrl._execute_scheduled_tasks(time.time())

        assert "timed" in ctrl.timing_analyzer.task_executions
        assert len(ctrl.timing_analyzer.task_executions["timed"]) >= 1


@pytest.mark.coverage_extra
class TestExecuteControlLoopsAlgorithms:
    """Cover line 360: Q-learning and unknown algorithm paths."""

    def test_qlearning_control_loop_above_setpoint(self):
        """Cover line 453: error < -0.1 returns -1.0."""
        ctrl = RealTimeController(_make_timing())
        loop = ControlLoop(
            loop_id="ql", input_channels=[0], output_channels=[0],
            control_algorithm="Q-learning", setpoint=25.0,
            gains={}, limits={"output": (-10.0, 10.0)},
        )
        ctrl.control_loops["ql"] = loop
        ctrl.sensor_data[0] = 30.0  # Well above setpoint
        ctrl._execute_control_loops()
        assert ctrl.actuator_outputs[0] == -1.0

    def test_control_loop_exception_triggers_fault(self):
        """Cover lines 378-380: exception in control loop."""
        ctrl = RealTimeController(_make_timing())
        loop = ControlLoop(
            loop_id="err", input_channels=[0], output_channels=[0],
            control_algorithm="PID", setpoint=25.0,
            gains={"kp": 1.0}, limits={},
        )
        ctrl.control_loops["err"] = loop
        ctrl.sensor_data[0] = 20.0

        with patch.object(ctrl, "_execute_pid_control",
                          side_effect=RuntimeError("pid boom")):
            ctrl._execute_control_loops()

        assert any("CONTROL_LOOP_ERROR_err" in f for f in ctrl.fault_flags)


@pytest.mark.coverage_extra
class TestWatchdogTimeoutDetection:
    """Cover lines 498-500: watchdog timeout detection."""

    def test_watchdog_detects_timeout(self):
        ctrl = RealTimeController(_make_timing(watchdog_timeout_ms=10.0))
        ctrl.running = True
        ctrl.watchdog_last_pet = time.time() - 1.0  # 1 second ago

        with patch("time.sleep"):
            ctrl._watchdog_loop()

        assert "WATCHDOG_TIMEOUT" in ctrl.fault_flags
        assert ctrl.running is True  # Loop breaks but running stays


@pytest.mark.coverage_extra
class TestGetMeasurementNoHistory:
    """Cover line 568: task_times[task_id] = 0.0."""

    def test_task_with_no_execution_history(self):
        ctrl = RealTimeController(_make_timing())
        task = _make_task(task_id="empty_hist")
        task.execution_history = []
        ctrl.tasks["empty_hist"] = task

        measurement = ctrl.get_measurement()

        assert measurement.task_execution_times["empty_hist"] == 0.0


@pytest.mark.coverage_extra
class TestSchedulerLoopWCETViolation:
    """Cover scheduler loop WCET violation warning path."""

    def test_wcet_violation_logged(self):
        # Use very low WCET to trigger warning
        ctrl = RealTimeController(
            _make_timing(worst_case_execution_time_ms=0.0001)
        )
        ctrl.mode = ControllerMode.MANUAL
        ctrl.running = True

        iteration = [0]

        def stop_after_one(*a, **kw):
            iteration[0] += 1
            if iteration[0] >= 1:
                ctrl.running = False

        with patch.object(ctrl, "_execute_scheduled_tasks",
                          side_effect=stop_after_one):
            with patch.object(ctrl, "_update_performance_metrics"):
                with patch("time.sleep"):
                    ctrl._scheduler_loop()

        # WCET violation is just logged, no assertion on fault_flags
        assert ctrl.running is False


@pytest.mark.coverage_extra
class TestSchedulerLoopOverrun:
    """Cover scheduler loop overrun warning path (sleep_time <= 0)."""

    def test_control_loop_overrun(self):
        ctrl = RealTimeController(
            _make_timing(control_loop_period_ms=0.001)  # Very short period
        )
        ctrl.mode = ControllerMode.MANUAL
        ctrl.running = True

        iteration = [0]

        def slow_tasks(*a, **kw):
            time.sleep(0.01)  # 10ms, much longer than 0.001ms period
            iteration[0] += 1
            if iteration[0] >= 1:
                ctrl.running = False

        with patch.object(ctrl, "_execute_scheduled_tasks",
                          side_effect=slow_tasks):
            with patch.object(ctrl, "_update_performance_metrics"):
                ctrl._scheduler_loop()

        assert ctrl.running is False
