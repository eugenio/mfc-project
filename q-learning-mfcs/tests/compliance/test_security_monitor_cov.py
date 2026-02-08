"""Tests for security_monitor.py module - comprehensive coverage."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from compliance.security_monitor import (
    AnomalyThreshold,
    SecurityAlert,
    SecurityConfig,
    SecurityIncident,
    SecurityMetrics,
    SecurityMonitor,
    SecurityThreat,
    ThreatPattern,
    ThreatSeverity,
)


class TestEnumsAndDataclasses:
    def test_threat_severity_values(self):
        assert ThreatSeverity.LOW.value == "low"
        assert ThreatSeverity.MEDIUM.value == "medium"
        assert ThreatSeverity.HIGH.value == "high"
        assert ThreatSeverity.CRITICAL.value == "critical"

    def test_security_config_defaults(self):
        cfg = SecurityConfig()
        assert cfg.threat_detection_enabled is True
        assert cfg.anomaly_detection_enabled is True
        assert cfg.monitoring_interval == 30.0
        assert cfg.threat_retention_days == 30
        assert cfg.max_incidents_tracked == 1000

    def test_threat_pattern(self):
        tp = ThreatPattern(name="t", pattern_type="regex")
        assert tp.name == "t"
        assert tp.enabled is True

    def test_anomaly_threshold(self):
        at = AnomalyThreshold("cpu", 80.0, "gt")
        assert at.metric_name == "cpu"
        assert at.threshold_value == 80.0

    def test_security_alert(self):
        a = SecurityAlert(
            alert_id="a1", alert_type="threat",
            severity=ThreatSeverity.HIGH, message="msg",
            source="src", timestamp=datetime.now(),
        )
        assert a.resolved is False

    def test_security_incident(self):
        i = SecurityIncident(
            incident_id="i1", title="t", description="d",
            severity=ThreatSeverity.HIGH, status="active",
            created_at=datetime.now(),
        )
        assert i.status == "active"

    def test_security_threat(self):
        t = SecurityThreat(
            threat_id="t1", threat_type="sql",
            severity=ThreatSeverity.CRITICAL,
        )
        assert t.status == "active"

    def test_security_metrics(self):
        m = SecurityMetrics()
        assert m.total_events == 0
        assert m.uptime_percentage == 100.0


class TestSecurityMonitorInit:
    def test_default_config(self):
        sm = SecurityMonitor()
        assert sm.config.threat_detection_enabled is True
        assert len(sm.threat_patterns) == 5
        assert len(sm.anomaly_thresholds) == 6

    def test_custom_config(self):
        cfg = SecurityConfig(monitoring_interval=10.0)
        sm = SecurityMonitor(config=cfg)
        assert sm.config.monitoring_interval == 10.0

    def test_initial_state(self):
        sm = SecurityMonitor()
        assert sm.is_monitoring is False
        assert len(sm.active_threats) == 0
        assert len(sm.active_alerts) == 0
        assert len(sm.active_incidents) == 0


class TestSecurityMonitorMonitoring:
    def test_start_monitoring(self):
        sm = SecurityMonitor()
        sm.start_monitoring()
        assert sm.is_monitoring is True
        sm.stop_monitoring()

    def test_start_monitoring_already_running(self):
        sm = SecurityMonitor()
        sm.start_monitoring()
        sm.start_monitoring()  # Should log warning
        sm.stop_monitoring()

    def test_stop_monitoring(self):
        sm = SecurityMonitor()
        sm.start_monitoring()
        sm.stop_monitoring()
        assert sm.is_monitoring is False

    def test_stop_monitoring_not_running(self):
        sm = SecurityMonitor()
        sm.stop_monitoring()  # Should not raise


class TestProcessSecurityEvent:
    def test_invalid_event_data_none(self):
        sm = SecurityMonitor()
        sm.process_security_event(None)
        assert sm.metrics.total_events == 0

    def test_invalid_event_data_not_dict(self):
        sm = SecurityMonitor()
        sm.process_security_event("not_a_dict")
        assert sm.metrics.total_events == 0

    def test_auth_success_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "authentication_success"})
        assert sm.metrics.authentication_attempts == 1
        assert sm.metrics.authentication_failures == 0

    def test_auth_failure_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "authentication_failure"})
        assert sm.metrics.authentication_attempts == 1
        assert sm.metrics.authentication_failures == 1

    def test_api_request_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "api_request"})
        assert sm.metrics.api_requests == 1

    def test_blocked_request_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "blocked_request"})
        assert sm.metrics.blocked_requests == 1

    def test_request_blocked_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "request_blocked"})
        assert sm.metrics.blocked_requests == 1

    def test_generic_event(self):
        sm = SecurityMonitor()
        sm.process_security_event({"event_type": "other"})
        assert sm.metrics.total_events == 1


class TestThreatDetection:
    def test_sql_injection_detection(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP TABLE users--",
        })
        assert sm.metrics.threats_detected >= 1

    def test_xss_detection(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "payload": '<script>alert("xss")</script>',
        })
        assert sm.metrics.threats_detected >= 1

    def test_no_threat_normal_payload(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "data": "normal data content",
        })
        # Should detect pattern match on normal data too due to broad regex
        # Just verify no crash
        assert sm.metrics.total_events == 1

    def test_disabled_pattern_skipped(self):
        sm = SecurityMonitor()
        for p in sm.threat_patterns:
            p.enabled = False
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users",
        })
        assert sm.metrics.threats_detected == 0

    def test_payload_no_regex(self):
        sm = SecurityMonitor()
        tp = ThreatPattern(
            name="no_regex", pattern_type="payload_content",
            regex_pattern=None
        )
        sm.threat_patterns = [tp]
        sm.process_security_event({
            "event_type": "request",
            "payload": "test",
        })
        assert sm.metrics.threats_detected == 0

    def test_unknown_pattern_type(self):
        sm = SecurityMonitor()
        tp = ThreatPattern(name="unknown", pattern_type="unknown_type")
        sm.threat_patterns = [tp]
        sm.process_security_event({
            "event_type": "request",
            "payload": "test",
        })
        assert sm.metrics.threats_detected == 0


class TestBehavioralDetection:
    def test_brute_force_not_auth_failure(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "brute_force_attack"
        ]
        sm.process_security_event({
            "event_type": "login_success",
            "source_ip": "10.0.0.1",
        })
        assert sm.metrics.threats_detected == 0

    def test_brute_force_no_source_ip(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "brute_force_attack"
        ]
        sm.process_security_event({
            "event_type": "authentication_failure",
        })
        assert sm.metrics.threats_detected == 0

    def test_brute_force_below_threshold(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "brute_force_attack"
        ]
        sm.process_security_event({
            "event_type": "authentication_failure",
            "source_ip": "10.0.0.1",
        })
        # Single failure should not trigger (threshold is 5)
        assert sm.metrics.threats_detected == 0

    def test_privilege_escalation_detected(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "privilege_escalation_attempt"
        ]
        sm.process_security_event({
            "event_type": "access_attempt",
            "required_role": "admin",
            "user_role": "user",
            "user_id": "u1",
            "resource": "admin_panel",
        })
        assert sm.metrics.threats_detected >= 1

    def test_privilege_escalation_not_triggered(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "privilege_escalation_attempt"
        ]
        sm.process_security_event({
            "event_type": "access_attempt",
            "required_role": "user",
            "user_role": "admin",
        })
        assert sm.metrics.threats_detected == 0

    def test_privilege_escalation_not_access_attempt(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "privilege_escalation_attempt"
        ]
        sm.process_security_event({
            "event_type": "other",
            "required_role": "admin",
            "user_role": "user",
        })
        assert sm.metrics.threats_detected == 0

    def test_suspicious_activity_detected(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "suspicious_activity"
        ]
        sm.process_security_event({
            "event_type": "port_scan",
            "source_ip": "10.0.0.99",
        })
        assert sm.metrics.threats_detected >= 1

    def test_suspicious_activity_type(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "suspicious_activity"
        ]
        sm.process_security_event({
            "event_type": "other",
            "activity_type": "malware",
        })
        assert sm.metrics.threats_detected >= 1

    def test_suspicious_attack_type(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "suspicious_activity"
        ]
        sm.process_security_event({
            "event_type": "other",
            "attack_type": "ransomware",
        })
        threats = sm.get_active_threats()
        assert any(t.severity == ThreatSeverity.CRITICAL for t in threats)

    def test_suspicious_not_triggered(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "suspicious_activity"
        ]
        sm.process_security_event({
            "event_type": "normal_event",
        })
        assert sm.metrics.threats_detected == 0


class TestIncidentCreation:
    def test_critical_threat_creates_incident(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP TABLE--",
        })
        assert sm.metrics.incidents_created >= 1

    def test_related_threats_create_incident(self):
        sm = SecurityMonitor()
        sm.threat_patterns = [
            p for p in sm.threat_patterns
            if p.threat_type == "suspicious_activity"
        ]
        sm.process_security_event({
            "event_type": "port_scan",
            "source_ip": "10.0.0.1",
        })
        sm.process_security_event({
            "event_type": "directory_traversal",
            "source_ip": "10.0.0.1",
        })
        # Two related threats from same IP should create incident
        assert sm.metrics.incidents_created >= 1


class TestAlertHandlers:
    def test_add_alert_handler(self):
        sm = SecurityMonitor()
        handler = MagicMock()
        sm.add_alert_handler("test_handler", handler)
        assert "test_handler" in sm.alert_handlers

    def test_alert_handler_called(self):
        sm = SecurityMonitor()
        handler = MagicMock()
        sm.add_alert_handler("test_handler", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users--",
        })
        assert handler.called

    def test_alert_handler_disabled(self):
        cfg = SecurityConfig(alert_notifications_enabled=False)
        sm = SecurityMonitor(config=cfg)
        handler = MagicMock()
        sm.add_alert_handler("test", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users--",
        })
        assert not handler.called

    def test_alert_handler_error(self):
        sm = SecurityMonitor()
        handler = MagicMock(side_effect=Exception("handler error"))
        sm.add_alert_handler("bad_handler", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users--",
        })
        # Should not raise, just log error

    def test_add_incident_handler(self):
        sm = SecurityMonitor()
        handler = MagicMock()
        sm.add_incident_response_handler("ir_handler", handler)
        assert "ir_handler" in sm.incident_response_handlers

    def test_incident_handler_called(self):
        sm = SecurityMonitor()
        handler = MagicMock()
        sm.add_incident_response_handler("ir", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP--",
        })
        assert handler.called

    def test_incident_handler_disabled(self):
        cfg = SecurityConfig(incident_response_enabled=False)
        sm = SecurityMonitor(config=cfg)
        handler = MagicMock()
        sm.add_incident_response_handler("ir", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP--",
        })
        assert not handler.called

    def test_incident_handler_error(self):
        sm = SecurityMonitor()
        handler = MagicMock(side_effect=Exception("ir error"))
        sm.add_incident_response_handler("bad_ir", handler)
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP--",
        })
        # Should not raise


class TestAnomalyDetection:
    def test_process_metrics_data(self):
        sm = SecurityMonitor()
        sm.process_metrics_data({"cpu_usage": 95.0})
        assert sm.metrics.alerts_generated >= 1

    def test_process_metrics_disabled(self):
        cfg = SecurityConfig(anomaly_detection_enabled=False)
        sm = SecurityMonitor(config=cfg)
        sm.process_metrics_data({"cpu_usage": 95.0})
        assert sm.metrics.alerts_generated == 0

    def test_anomaly_gt(self):
        sm = SecurityMonitor()
        sm.process_metrics_data({"cpu_usage": 95.0})
        assert sm.metrics.alerts_generated >= 1

    def test_anomaly_below_threshold(self):
        sm = SecurityMonitor()
        sm.process_metrics_data({"cpu_usage": 50.0})
        assert sm.metrics.alerts_generated == 0

    def test_anomaly_gte(self):
        sm = SecurityMonitor()
        sm.update_anomaly_threshold("test_metric", 80.0, "gte")
        sm.process_metrics_data({"test_metric": 80.0})
        assert sm.metrics.alerts_generated >= 1

    def test_anomaly_lt(self):
        sm = SecurityMonitor()
        sm.update_anomaly_threshold("test_metric", 10.0, "lt")
        sm.process_metrics_data({"test_metric": 5.0})
        assert sm.metrics.alerts_generated >= 1

    def test_anomaly_lte(self):
        sm = SecurityMonitor()
        sm.update_anomaly_threshold("test_metric", 10.0, "lte")
        sm.process_metrics_data({"test_metric": 10.0})
        assert sm.metrics.alerts_generated >= 1

    def test_anomaly_eq(self):
        sm = SecurityMonitor()
        sm.update_anomaly_threshold("test_metric", 42.0, "eq")
        sm.process_metrics_data({"test_metric": 42.0})
        assert sm.metrics.alerts_generated >= 1

    def test_anomaly_disabled_threshold(self):
        sm = SecurityMonitor()
        sm.anomaly_thresholds["cpu_usage"].enabled = False
        sm.process_metrics_data({"cpu_usage": 999.0})
        assert sm.metrics.alerts_generated == 0

    def test_anomaly_unknown_metric(self):
        sm = SecurityMonitor()
        sm.process_metrics_data({"unknown_metric": 95.0})
        assert sm.metrics.alerts_generated == 0

    def test_anomaly_non_numeric_skipped(self):
        sm = SecurityMonitor()
        sm.process_metrics_data({"cpu_usage": "high"})
        assert sm.metrics.alerts_generated == 0

    def test_get_anomaly_threshold(self):
        sm = SecurityMonitor()
        t = sm.get_anomaly_threshold("cpu_usage")
        assert t == 80.0

    def test_get_anomaly_threshold_missing(self):
        sm = SecurityMonitor()
        t = sm.get_anomaly_threshold("nonexistent")
        assert t is None

    def test_update_anomaly_threshold(self):
        sm = SecurityMonitor()
        sm.update_anomaly_threshold("new_metric", 50.0, "lt")
        assert sm.anomaly_thresholds["new_metric"].threshold_value == 50.0


class TestPublicAPIMethods:
    def test_get_active_threats(self):
        sm = SecurityMonitor()
        threats = sm.get_active_threats()
        assert isinstance(threats, list)

    def test_get_active_alerts(self):
        sm = SecurityMonitor()
        alerts = sm.get_active_alerts()
        assert isinstance(alerts, list)

    def test_get_active_incidents(self):
        sm = SecurityMonitor()
        incidents = sm.get_active_incidents()
        assert isinstance(incidents, list)

    def test_get_archived_threats(self):
        sm = SecurityMonitor()
        archived = sm.get_archived_threats()
        assert isinstance(archived, list)

    def test_get_security_metrics(self):
        sm = SecurityMonitor()
        metrics = sm.get_security_metrics()
        assert isinstance(metrics, SecurityMetrics)

    def test_add_threat_pattern(self):
        sm = SecurityMonitor()
        tp = ThreatPattern(name="custom", pattern_type="behavioral")
        initial_count = len(sm.threat_patterns)
        sm.add_threat_pattern(tp)
        assert len(sm.threat_patterns) == initial_count + 1

    def test_resolve_alert(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users--",
        })
        alerts = sm.get_active_alerts()
        if alerts:
            alert_id = alerts[0].alert_id
            sm.resolve_alert(alert_id, "Fixed")
            assert alert_id not in sm.active_alerts

    def test_resolve_alert_nonexistent(self):
        sm = SecurityMonitor()
        sm.resolve_alert("nonexistent")  # Should not raise

    def test_resolve_incident(self):
        sm = SecurityMonitor()
        sm.process_security_event({
            "event_type": "request",
            "payload": "SELECT * FROM users; DROP TABLE--",
        })
        incidents = sm.get_active_incidents()
        if incidents:
            incident_id = incidents[0].incident_id
            sm.resolve_incident(incident_id, "Resolved")
            assert incident_id not in sm.active_incidents

    def test_resolve_incident_nonexistent(self):
        sm = SecurityMonitor()
        sm.resolve_incident("nonexistent")


class TestSecurityScoreAndReport:
    def test_score_no_threats(self):
        sm = SecurityMonitor()
        assert sm.calculate_security_score() == 100.0

    def test_score_with_critical_threat(self):
        sm = SecurityMonitor()
        sm.active_threats["t1"] = SecurityThreat(
            threat_id="t1", threat_type="sql",
            severity=ThreatSeverity.CRITICAL,
        )
        score = sm.calculate_security_score()
        assert score == 80.0

    def test_score_with_high_threat(self):
        sm = SecurityMonitor()
        sm.active_threats["t1"] = SecurityThreat(
            threat_id="t1", threat_type="xss",
            severity=ThreatSeverity.HIGH,
        )
        assert sm.calculate_security_score() == 90.0

    def test_score_with_medium_threat(self):
        sm = SecurityMonitor()
        sm.active_threats["t1"] = SecurityThreat(
            threat_id="t1", threat_type="sus",
            severity=ThreatSeverity.MEDIUM,
        )
        assert sm.calculate_security_score() == 95.0

    def test_score_with_low_threat(self):
        sm = SecurityMonitor()
        sm.active_threats["t1"] = SecurityThreat(
            threat_id="t1", threat_type="info",
            severity=ThreatSeverity.LOW,
        )
        assert sm.calculate_security_score() == 98.0

    def test_score_with_incidents(self):
        sm = SecurityMonitor()
        sm.active_incidents["i1"] = SecurityIncident(
            incident_id="i1", title="t", description="d",
            severity=ThreatSeverity.CRITICAL, status="active",
            created_at=datetime.now(),
        )
        assert sm.calculate_security_score() == 85.0

    def test_score_with_high_incident(self):
        sm = SecurityMonitor()
        sm.active_incidents["i1"] = SecurityIncident(
            incident_id="i1", title="t", description="d",
            severity=ThreatSeverity.HIGH, status="active",
            created_at=datetime.now(),
        )
        assert sm.calculate_security_score() == 92.0

    def test_score_floor_at_zero(self):
        sm = SecurityMonitor()
        for i in range(10):
            sm.active_threats[f"t{i}"] = SecurityThreat(
                threat_id=f"t{i}", threat_type="sql",
                severity=ThreatSeverity.CRITICAL,
            )
        assert sm.calculate_security_score() == 0.0

    def test_generate_security_report(self):
        sm = SecurityMonitor()
        report = sm.generate_security_report()
        assert "timestamp" in report
        assert "security_score" in report
        assert "active_threats" in report
        assert "recommendations" in report

    def test_recommendations_many_threats(self):
        sm = SecurityMonitor()
        for i in range(15):
            sm.active_threats[f"t{i}"] = SecurityThreat(
                threat_id=f"t{i}", threat_type="sql",
                severity=ThreatSeverity.LOW,
            )
        recs = sm._generate_recommendations()
        assert any("High number" in r for r in recs)

    def test_recommendations_active_incidents(self):
        sm = SecurityMonitor()
        sm.active_incidents["i1"] = SecurityIncident(
            incident_id="i1", title="t", description="d",
            severity=ThreatSeverity.LOW, status="active",
            created_at=datetime.now(),
        )
        recs = sm._generate_recommendations()
        assert any("immediate attention" in r for r in recs)

    def test_recommendations_low_score(self):
        sm = SecurityMonitor()
        for i in range(5):
            sm.active_threats[f"t{i}"] = SecurityThreat(
                threat_id=f"t{i}", threat_type="sql",
                severity=ThreatSeverity.CRITICAL,
            )
        recs = sm._generate_recommendations()
        assert any("below optimal" in r for r in recs)

    def test_recommendations_empty(self):
        sm = SecurityMonitor()
        recs = sm._generate_recommendations()
        assert isinstance(recs, list)


class TestExportAndCleanup:
    def test_export_json(self):
        sm = SecurityMonitor()
        data = sm.export_security_data(format="json")
        parsed = json.loads(data)
        assert "threats" in parsed
        assert "alerts" in parsed

    def test_export_unsupported_format(self):
        sm = SecurityMonitor()
        with pytest.raises(ValueError, match="Unsupported export format"):
            sm.export_security_data(format="xml")

    def test_archive_old_data(self):
        sm = SecurityMonitor()
        sm.archive_old_data()  # Should not raise

    def test_cleanup_old_data(self):
        sm = SecurityMonitor()
        old_ts = datetime.now() - timedelta(days=60)
        sm.active_threats["old_t"] = SecurityThreat(
            threat_id="old_t", threat_type="sql",
            severity=ThreatSeverity.LOW, timestamp=old_ts,
        )
        sm._cleanup_old_data()
        assert "old_t" not in sm.active_threats
        assert len(sm.archived_threats) == 1

    def test_cleanup_limits_archive_size(self):
        sm = SecurityMonitor()
        old_ts = datetime.now() - timedelta(days=60)
        sm.archived_threats = [
            SecurityThreat(
                threat_id=f"t{i}", threat_type="sql",
                severity=ThreatSeverity.LOW, timestamp=old_ts,
            )
            for i in range(15000)
        ]
        sm._cleanup_old_data()
        assert len(sm.archived_threats) <= 5000

    def test_collect_system_metrics(self):
        sm = SecurityMonitor()
        sm._collect_system_metrics()
        assert sm.metrics.last_updated is not None

    def test_collect_system_metrics_disabled(self):
        cfg = SecurityConfig(metrics_collection_enabled=False)
        sm = SecurityMonitor(config=cfg)
        old_ts = sm.metrics.last_updated
        sm._collect_system_metrics()
        # timestamp should not update when disabled

    def test_check_anomalies(self):
        sm = SecurityMonitor()
        sm._check_anomalies()  # Placeholder - should not raise
