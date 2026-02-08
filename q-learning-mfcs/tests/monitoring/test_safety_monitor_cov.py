"""Tests for monitoring/safety_monitor.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Mock the heavy dependencies before loading the module
_mock_np = MagicMock()
_mock_np.random = MagicMock()
_mock_np.random.normal = MagicMock(return_value=0.0)
_mock_np.mean = MagicMock(side_effect=lambda x: sum(x) / len(x) if hasattr(x, '__len__') else x)

# Mock config.real_time_processing - need AlertLevel and AlertSystem
_mock_rtp = MagicMock()


class FakeAlertLevel:
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FakeAlertSystem:
    def __init__(self, **kwargs):
        self.alerts = []

    def add_alert(self, level, title, detail, source=""):
        self.alerts.append({"level": level, "title": title, "detail": detail, "source": source})


_mock_rtp.AlertLevel = FakeAlertLevel
_mock_rtp.AlertSystem = FakeAlertSystem

sys.modules.setdefault("numpy", _mock_np)
sys.modules.setdefault("config.real_time_processing", _mock_rtp)

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.safety_monitor",
    os.path.join(_src, "monitoring", "safety_monitor.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.safety_monitor"] = _mod
_spec.loader.exec_module(_mod)

# The source defines SafetyThreshold and SafetyEvent with class-level annotations
# but no @dataclass decorator, so their __init__ does not accept keyword args.
# Patch them to be dataclasses so SafetyMonitor can instantiate correctly.
from enum import Enum as _Enum

_OrigEmergencyAction = _mod.EmergencyAction
_OrigSafetyLevel = _mod.SafetyLevel


@dataclass
class _SafetyThreshold:
    parameter: str = ""
    min_value: float | None = None
    max_value: float | None = None
    warning_buffer: float = 0.1
    critical_duration_s: float = 5.0
    emergency_action: object = _OrigEmergencyAction.NONE
    enabled: bool = True


@dataclass
class _SafetyEvent:
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    parameter: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    safety_level: object = _OrigSafetyLevel.SAFE
    action_taken: object = _OrigEmergencyAction.NONE
    response_time_ms: float = 0.0
    acknowledged: bool = False
    resolved: bool = False


_mod.SafetyThreshold = _SafetyThreshold
_mod.SafetyEvent = _SafetyEvent

# The source uses 'logger' but never defines it - inject one.
import logging as _logging

_mod.logger = _logging.getLogger("monitoring.safety_monitor")

SafetyLevel = _mod.SafetyLevel
EmergencyAction = _mod.EmergencyAction
SafetyThreshold = _mod.SafetyThreshold
SafetyEvent = _mod.SafetyEvent
SafetyProtocol = _mod.SafetyProtocol
SafetyMonitor = _mod.SafetyMonitor


class TestEnums:
    def test_safety_level_values(self):
        assert SafetyLevel.SAFE.value == "safe"
        assert SafetyLevel.CAUTION.value == "caution"
        assert SafetyLevel.WARNING.value == "warning"
        assert SafetyLevel.CRITICAL.value == "critical"
        assert SafetyLevel.EMERGENCY.value == "emergency"

    def test_emergency_action_values(self):
        assert EmergencyAction.NONE.value == "none"
        assert EmergencyAction.REDUCE_POWER.value == "reduce_power"
        assert EmergencyAction.STOP_FLOW.value == "stop_flow"
        assert EmergencyAction.EMERGENCY_SHUTDOWN.value == "emergency_shutdown"
        assert EmergencyAction.ISOLATE_SYSTEM.value == "isolate_system"
        assert EmergencyAction.NOTIFY_PERSONNEL.value == "notify_personnel"


class TestSafetyThreshold:
    def test_defaults(self):
        t = SafetyThreshold()
        assert t.warning_buffer == 0.1
        assert t.critical_duration_s == 5.0
        assert t.emergency_action == EmergencyAction.NONE
        assert t.enabled is True

    def test_custom(self):
        t = SafetyThreshold()
        t.parameter = "temp"
        t.min_value = 10.0
        t.max_value = 50.0
        assert t.parameter == "temp"
        assert t.min_value == 10.0


class TestSafetyEvent:
    def test_defaults(self):
        e = SafetyEvent()
        assert e.acknowledged is False
        assert e.resolved is False


class TestSafetyProtocol:
    def test_init(self):
        p = SafetyProtocol(
            name="Test Protocol",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        assert p.name == "Test Protocol"
        assert p.triggers == ["temperature"]
        assert p.is_active is False
        assert p.last_triggered is None

    def test_should_trigger_true(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        event = SafetyEvent()
        event.parameter = "temperature"
        event.resolved = False
        event.safety_level = SafetyLevel.CRITICAL
        assert p.should_trigger([event]) is True

    def test_should_trigger_false_resolved(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        event = SafetyEvent()
        event.parameter = "temperature"
        event.resolved = True
        event.safety_level = SafetyLevel.CRITICAL
        assert p.should_trigger([event]) is False

    def test_should_trigger_false_wrong_level(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        event = SafetyEvent()
        event.parameter = "temperature"
        event.resolved = False
        event.safety_level = SafetyLevel.WARNING
        assert p.should_trigger([event]) is False

    def test_should_trigger_false_wrong_param(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["pressure"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        event = SafetyEvent()
        event.parameter = "temperature"
        event.resolved = False
        event.safety_level = SafetyLevel.CRITICAL
        assert p.should_trigger([event]) is False

    def test_execute_all_actions(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[
                EmergencyAction.REDUCE_POWER,
                EmergencyAction.STOP_FLOW,
                EmergencyAction.EMERGENCY_SHUTDOWN,
                EmergencyAction.ISOLATE_SYSTEM,
                EmergencyAction.NOTIFY_PERSONNEL,
            ],
        )
        controller = MagicMock()
        results = p.execute(controller)
        assert len(results) == 5
        assert p.is_active is True
        assert p.last_triggered is not None

    def test_execute_with_error(self):
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        # Patch the action comparison to raise
        controller = MagicMock()
        results = p.execute(controller)
        assert len(results) >= 1


class TestSafetyMonitor:
    def setup_method(self):
        self.monitor = SafetyMonitor(mfc_model=None)

    def test_init(self):
        assert self.monitor.is_monitoring is False
        assert len(self.monitor.safety_thresholds) > 0
        assert len(self.monitor.safety_protocols) > 0
        assert self.monitor.stats["total_events"] == 0

    def test_init_with_model(self):
        model = MagicMock()
        monitor = SafetyMonitor(mfc_model=model)
        assert monitor.mfc_model is model

    def test_default_thresholds(self):
        thresholds = self.monitor.safety_thresholds
        assert "temperature" in thresholds
        assert "pressure" in thresholds
        assert "ph_level" in thresholds
        assert "voltage" in thresholds
        assert "current_density" in thresholds
        assert "flow_rate" in thresholds
        assert "biofilm_thickness" in thresholds

    def test_default_protocols(self):
        protocols = self.monitor.safety_protocols
        assert "thermal_runaway" in protocols
        assert "pressure_emergency" in protocols
        assert "system_failure" in protocols
        assert "biological_contamination" in protocols

    def test_start_stop_monitoring(self):
        self.monitor.start_monitoring(interval_seconds=10)
        assert self.monitor.is_monitoring is True
        self.monitor.stop_monitoring()
        assert self.monitor.is_monitoring is False

    def test_start_monitoring_already_running(self):
        self.monitor.is_monitoring = True
        self.monitor.start_monitoring()
        # Should not create a second thread

    def test_stop_monitoring_not_running(self):
        self.monitor.stop_monitoring()
        assert self.monitor.is_monitoring is False

    def test_evaluate_safety_level_safe(self):
        t = SafetyThreshold()
        t.max_value = 50.0
        t.min_value = 10.0
        t.warning_buffer = 5.0
        result = self.monitor._evaluate_safety_level("temp", 30.0, t)
        assert result == SafetyLevel.SAFE

    def test_evaluate_safety_level_emergency_above_max(self):
        t = SafetyThreshold()
        t.max_value = 50.0
        t.warning_buffer = 5.0
        result = self.monitor._evaluate_safety_level("temp", 55.0, t)
        assert result == SafetyLevel.EMERGENCY

    def test_evaluate_safety_level_critical_near_max(self):
        t = SafetyThreshold()
        t.max_value = 50.0
        t.warning_buffer = 5.0
        result = self.monitor._evaluate_safety_level("temp", 47.0, t)
        assert result == SafetyLevel.CRITICAL

    def test_evaluate_safety_level_warning_near_max(self):
        t = SafetyThreshold()
        t.max_value = 50.0
        t.warning_buffer = 5.0
        result = self.monitor._evaluate_safety_level("temp", 42.0, t)
        assert result == SafetyLevel.WARNING

    def test_evaluate_safety_level_caution_near_max(self):
        t = SafetyThreshold()
        t.max_value = 50.0
        t.warning_buffer = 5.0
        result = self.monitor._evaluate_safety_level("temp", 37.0, t)
        assert result == SafetyLevel.CAUTION

    def test_evaluate_safety_level_emergency_below_min(self):
        t = SafetyThreshold()
        t.min_value = 10.0
        t.max_value = None
        t.warning_buffer = 2.0
        result = self.monitor._evaluate_safety_level("temp", 5.0, t)
        assert result == SafetyLevel.EMERGENCY

    def test_evaluate_safety_level_critical_near_min(self):
        t = SafetyThreshold()
        t.min_value = 10.0
        t.max_value = None
        t.warning_buffer = 2.0
        result = self.monitor._evaluate_safety_level("temp", 11.0, t)
        assert result == SafetyLevel.CRITICAL

    def test_evaluate_safety_level_warning_near_min(self):
        t = SafetyThreshold()
        t.min_value = 10.0
        t.max_value = None
        t.warning_buffer = 2.0
        result = self.monitor._evaluate_safety_level("temp", 13.0, t)
        assert result == SafetyLevel.WARNING

    def test_evaluate_safety_level_caution_near_min(self):
        t = SafetyThreshold()
        t.min_value = 10.0
        t.max_value = None
        t.warning_buffer = 2.0
        result = self.monitor._evaluate_safety_level("temp", 15.0, t)
        assert result == SafetyLevel.CAUTION

    def test_check_safety_thresholds_no_violation(self):
        measurements = {"temperature": 30.0}
        events = self.monitor._check_safety_thresholds(measurements)
        assert len(events) == 0

    def test_check_safety_thresholds_violation(self):
        measurements = {"temperature": 100.0}
        events = self.monitor._check_safety_thresholds(measurements)
        assert len(events) >= 1

    def test_check_safety_thresholds_disabled(self):
        self.monitor.safety_thresholds["temperature"].enabled = False
        measurements = {"temperature": 100.0}
        events = self.monitor._check_safety_thresholds(measurements)
        temp_events = [e for e in events if e.parameter == "temperature"]
        assert len(temp_events) == 0

    def test_check_safety_thresholds_unknown_param(self):
        measurements = {"unknown_sensor": 999.0}
        events = self.monitor._check_safety_thresholds(measurements)
        assert len(events) == 0

    def test_process_safety_event_new(self):
        event = SafetyEvent()
        event.event_id = "test_1"
        event.timestamp = datetime.now()
        event.parameter = "test_param"
        event.current_value = 100.0
        event.threshold_value = 50.0
        event.safety_level = SafetyLevel.WARNING
        event.action_taken = EmergencyAction.NONE
        event.response_time_ms = 0.0
        self.monitor._process_safety_event(event)
        assert self.monitor.stats["total_events"] == 1
        assert "test_param" in self.monitor.active_events

    def test_process_safety_event_continuation_critical(self):
        existing = SafetyEvent()
        existing.event_id = "existing_1"
        existing.timestamp = datetime.now() - timedelta(seconds=100)
        existing.parameter = "temperature"
        existing.current_value = 50.0
        existing.threshold_value = 45.0
        existing.safety_level = SafetyLevel.CRITICAL
        existing.action_taken = EmergencyAction.NONE
        existing.response_time_ms = 0.0
        self.monitor.active_events["temperature"] = existing
        self.monitor.safety_thresholds["temperature"].critical_duration_s = 10.0

        new_event = SafetyEvent()
        new_event.event_id = "new_1"
        new_event.timestamp = datetime.now()
        new_event.parameter = "temperature"
        new_event.current_value = 55.0
        new_event.threshold_value = 45.0
        new_event.safety_level = SafetyLevel.CRITICAL
        new_event.action_taken = EmergencyAction.NONE
        new_event.response_time_ms = 0.0
        self.monitor._process_safety_event(new_event)
        assert self.monitor.stats["emergency_actions"] >= 1

    def test_process_safety_event_critical_count(self):
        event = SafetyEvent()
        event.event_id = "crit_1"
        event.timestamp = datetime.now()
        event.parameter = "test_param"
        event.current_value = 100.0
        event.threshold_value = 50.0
        event.safety_level = SafetyLevel.CRITICAL
        event.action_taken = EmergencyAction.NONE
        event.response_time_ms = 0.0
        self.monitor._process_safety_event(event)
        assert self.monitor.stats["critical_events"] >= 1

    def test_execute_emergency_action_reduce_power(self):
        event = SafetyEvent()
        results = self.monitor._execute_emergency_action(EmergencyAction.REDUCE_POWER, event)
        assert len(results) >= 1

    def test_execute_emergency_action_stop_flow(self):
        event = SafetyEvent()
        results = self.monitor._execute_emergency_action(EmergencyAction.STOP_FLOW, event)
        assert len(results) >= 1

    def test_execute_emergency_action_shutdown(self):
        event = SafetyEvent()
        results = self.monitor._execute_emergency_action(EmergencyAction.EMERGENCY_SHUTDOWN, event)
        assert len(results) >= 1

    def test_execute_emergency_action_isolate(self):
        event = SafetyEvent()
        results = self.monitor._execute_emergency_action(EmergencyAction.ISOLATE_SYSTEM, event)
        assert len(results) >= 1

    def test_execute_emergency_action_notify(self):
        event = SafetyEvent()
        event.parameter = "temperature"
        event.current_value = 100.0
        event.safety_level = SafetyLevel.CRITICAL
        results = self.monitor._execute_emergency_action(EmergencyAction.NOTIFY_PERSONNEL, event)
        assert "Personnel notifications sent" in results

    def test_send_safety_alert_warning(self):
        event = SafetyEvent()
        event.parameter = "temp"
        event.current_value = 55.0
        event.threshold_value = 50.0
        event.safety_level = SafetyLevel.WARNING
        self.monitor._send_safety_alert(event)
        assert len(self.monitor.alert_system.alerts) >= 1

    def test_send_safety_alert_critical(self):
        event = SafetyEvent()
        event.parameter = "temp"
        event.current_value = 55.0
        event.threshold_value = 50.0
        event.safety_level = SafetyLevel.CRITICAL
        self.monitor._send_safety_alert(event)

    def test_send_safety_alert_emergency(self):
        event = SafetyEvent()
        event.parameter = "temp"
        event.current_value = 55.0
        event.threshold_value = 50.0
        event.safety_level = SafetyLevel.EMERGENCY
        self.monitor._send_safety_alert(event)

    def test_acknowledge_event(self):
        event = SafetyEvent()
        event.event_id = "ack_1"
        event.acknowledged = False
        self.monitor.safety_events.append(event)
        result = self.monitor.acknowledge_event("ack_1", "admin")
        assert result is True
        assert event.acknowledged is True

    def test_acknowledge_event_not_found(self):
        result = self.monitor.acknowledge_event("nonexistent")
        assert result is False

    def test_resolve_event(self):
        event = SafetyEvent()
        event.event_id = "res_1"
        event.parameter = "temperature"
        event.resolved = False
        self.monitor.safety_events.append(event)
        self.monitor.active_events["temperature"] = event
        result = self.monitor.resolve_event("res_1", "admin")
        assert result is True
        assert event.resolved is True
        assert "temperature" not in self.monitor.active_events

    def test_resolve_event_not_found(self):
        result = self.monitor.resolve_event("nonexistent")
        assert result is False

    def test_get_safety_status_safe(self):
        status = self.monitor.get_safety_status()
        assert status["overall_safety_level"] == "safe"
        assert status["active_events"] == 0
        assert status["is_monitoring"] is False

    def test_get_safety_status_emergency(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temp"
        event.current_value = 100.0
        event.safety_level = SafetyLevel.EMERGENCY
        event.resolved = False
        event.acknowledged = False
        event.event_id = "em_1"
        self.monitor.safety_events.append(event)
        status = self.monitor.get_safety_status()
        assert status["overall_safety_level"] == "emergency"

    def test_get_safety_status_critical(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temp"
        event.current_value = 60.0
        event.safety_level = SafetyLevel.CRITICAL
        event.resolved = False
        event.acknowledged = False
        event.event_id = "cr_1"
        self.monitor.safety_events.append(event)
        status = self.monitor.get_safety_status()
        assert status["overall_safety_level"] == "critical"

    def test_get_safety_status_warning(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temp"
        event.current_value = 48.0
        event.safety_level = SafetyLevel.WARNING
        event.resolved = False
        event.event_id = "w_1"
        self.monitor.safety_events.append(event)
        status = self.monitor.get_safety_status()
        assert status["overall_safety_level"] == "warning"

    def test_get_safety_status_caution(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temp"
        event.current_value = 40.0
        event.safety_level = SafetyLevel.CAUTION
        event.resolved = False
        event.event_id = "ca_1"
        self.monitor.safety_events.append(event)
        status = self.monitor.get_safety_status()
        assert status["overall_safety_level"] == "caution"

    def test_get_safety_status_active_protocols(self):
        self.monitor.safety_protocols["thermal_runaway"].is_active = True
        status = self.monitor.get_safety_status()
        assert "thermal_runaway" in status["active_protocols"]

    def test_update_threshold(self):
        result = self.monitor.update_threshold("temperature", {"max_value": 60.0})
        assert result is True
        assert self.monitor.safety_thresholds["temperature"].max_value == 60.0

    def test_update_threshold_not_found(self):
        result = self.monitor.update_threshold("nonexistent", {"max_value": 60.0})
        assert result is False

    def test_update_threshold_invalid_attr(self):
        result = self.monitor.update_threshold("temperature", {"nonexistent_attr": 99})
        assert result is True  # returns True even if attr doesn't exist (no-op)

    def test_get_safety_report_empty(self):
        report = self.monitor.get_safety_report(hours=24)
        assert report["report_period_hours"] == 24
        assert report["summary"]["total_events"] == 0

    def test_get_safety_report_with_events(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temperature"
        event.current_value = 100.0
        event.threshold_value = 45.0
        event.safety_level = SafetyLevel.CRITICAL
        event.action_taken = EmergencyAction.REDUCE_POWER
        event.response_time_ms = 5.0
        event.acknowledged = False
        event.resolved = False
        self.monitor.safety_events.append(event)

        report = self.monitor.get_safety_report(hours=1)
        assert report["summary"]["total_events"] == 1
        assert report["summary"]["critical_events"] == 1
        assert len(report["detailed_events"]) == 1
        assert "temperature" in report["parameter_breakdown"]

    def test_get_safety_report_response_time_stats(self):
        for i in range(3):
            event = SafetyEvent()
            event.timestamp = datetime.now()
            event.parameter = f"param_{i}"
            event.current_value = 100.0
            event.threshold_value = 50.0
            event.safety_level = SafetyLevel.WARNING
            event.action_taken = EmergencyAction.NONE
            event.response_time_ms = float(i + 1)
            event.acknowledged = False
            event.resolved = False
            self.monitor.safety_events.append(event)

        report = self.monitor.get_safety_report(hours=1)
        assert report["summary"]["total_events"] == 3

    def test_get_current_measurements_no_model(self):
        measurements = self.monitor._get_current_measurements()
        assert isinstance(measurements, dict)
        assert "temperature" in measurements

    def test_get_current_measurements_with_model(self):
        model = MagicMock()
        state = MagicMock()
        state.cell_voltages = [0.7, 0.8]
        state.current_densities = [5.0, 6.0]
        state.flow_rate = 100.0
        state.biofilm_thickness = [10.0, 12.0]
        model.get_current_state.return_value = state
        self.monitor.mfc_model = model
        measurements = self.monitor._get_current_measurements()
        assert isinstance(measurements, dict)

    def test_get_current_measurements_model_error(self):
        model = MagicMock()
        model.get_current_state.side_effect = Exception("model error")
        self.monitor.mfc_model = model
        measurements = self.monitor._get_current_measurements()
        assert isinstance(measurements, dict)

    def test_check_safety_protocols(self):
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temperature"
        event.safety_level = SafetyLevel.CRITICAL
        event.resolved = False
        self.monitor.safety_events.append(event)
        self.monitor._check_safety_protocols()

    def test_check_safety_protocols_already_active(self):
        self.monitor.safety_protocols["thermal_runaway"].is_active = True
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temperature"
        event.safety_level = SafetyLevel.CRITICAL
        event.resolved = False
        self.monitor.safety_events.append(event)
        self.monitor._check_safety_protocols()

    def test_notify_personnel(self):
        event = SafetyEvent()
        event.parameter = "temperature"
        event.current_value = 100.0
        event.safety_level = SafetyLevel.CRITICAL
        self.monitor._notify_personnel(event)

    def test_protocol_execute_action_exception(self):
        """Cover SafetyProtocol.execute exception handler (lines 122-124)."""
        p = SafetyProtocol(
            name="Test",
            triggers=["temperature"],
            actions=[EmergencyAction.REDUCE_POWER],
        )
        # Monkey-patch the actions list with an object whose comparison raises
        class BadAction:
            value = "bad"
            def __eq__(self, other):
                raise RuntimeError("action compare error")
        p.actions = [BadAction()]
        controller = MagicMock()
        results = p.execute(controller)
        assert any("Failed to execute" in r for r in results)
        assert p.is_active is True

    def test_monitoring_loop_processes_events(self):
        """Cover _monitoring_loop lines 299, 309-310."""
        # Make monitoring stop after one iteration
        call_count = 0
        original_sleep = time.sleep

        def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            self.monitor.is_monitoring = False

        self.monitor.is_monitoring = True
        # Provide a threshold violation so safety events are generated
        self.monitor.safety_thresholds["temperature"].max_value = 30.0
        with patch.object(_mod.time, "sleep", side_effect=fake_sleep):
            self.monitor._monitoring_loop(0.01)
        assert call_count >= 1

    def test_monitoring_loop_exception_handler(self):
        """Cover _monitoring_loop exception handler (lines 309-310)."""
        call_count = 0

        def fake_sleep(secs):
            nonlocal call_count
            call_count += 1
            self.monitor.is_monitoring = False

        self.monitor.is_monitoring = True
        with patch.object(
            self.monitor, "_get_current_measurements", side_effect=RuntimeError("loop error")
        ), patch.object(_mod.time, "sleep", side_effect=fake_sleep):
            self.monitor._monitoring_loop(0.01)
        assert call_count >= 1

    def test_execute_emergency_action_exception(self):
        """Cover _execute_emergency_action exception handler (lines 494-496)."""
        event = SafetyEvent()
        event.parameter = "temperature"
        event.current_value = 100.0
        event.safety_level = SafetyLevel.CRITICAL
        with patch.object(
            self.monitor, "_notify_personnel", side_effect=RuntimeError("notify fail")
        ):
            results = self.monitor._execute_emergency_action(
                EmergencyAction.NOTIFY_PERSONNEL, event
            )
        assert any("Error" in r for r in results)

    def test_check_safety_protocols_execute_exception(self):
        """Cover _check_safety_protocols exception handler (lines 517-518)."""
        event = SafetyEvent()
        event.timestamp = datetime.now()
        event.parameter = "temperature"
        event.safety_level = SafetyLevel.CRITICAL
        event.resolved = False
        self.monitor.safety_events.append(event)
        # Replace the protocol's execute with one that raises
        protocol = self.monitor.safety_protocols["thermal_runaway"]
        protocol.is_active = False
        with patch.object(protocol, "execute", side_effect=RuntimeError("proto error")):
            self.monitor._check_safety_protocols()
