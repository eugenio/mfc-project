"""Tests for monitoring/alert_management.py - targeting 98%+ coverage."""
import importlib.util
import os
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Import the module directly to avoid triggering monitoring/__init__.py
# which has heavy dependencies that cause numpy reimport conflicts.
_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.alert_management",
    os.path.join(_src, "monitoring", "alert_management.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.alert_management"] = _mod
_spec.loader.exec_module(_mod)

Alert = _mod.Alert
AlertDatabase = _mod.AlertDatabase
AlertManager = _mod.AlertManager
AlertThreshold = _mod.AlertThreshold
EmailNotificationService = _mod.EmailNotificationService
EscalationRule = _mod.EscalationRule


class TestAlertThreshold:
    """Tests for AlertThreshold dataclass."""

    def test_check_value_normal(self):
        t = AlertThreshold(parameter="temp", min_value=10.0, max_value=50.0)
        severity, violated = t.check_value(30.0)
        assert severity == "normal"
        assert violated is False

    def test_check_value_disabled(self):
        t = AlertThreshold(parameter="temp", min_value=10.0, max_value=50.0, enabled=False)
        severity, violated = t.check_value(5.0)
        assert severity == "normal"
        assert violated is False

    def test_check_value_warning_below_min(self):
        t = AlertThreshold(parameter="temp", min_value=10.0, max_value=50.0)
        severity, violated = t.check_value(5.0)
        assert severity == "warning"
        assert violated is True

    def test_check_value_warning_above_max(self):
        t = AlertThreshold(parameter="temp", min_value=10.0, max_value=50.0)
        severity, violated = t.check_value(55.0)
        assert severity == "warning"
        assert violated is True

    def test_check_value_critical_below_min(self):
        t = AlertThreshold(parameter="temp", critical_min=5.0)
        severity, violated = t.check_value(3.0)
        assert severity == "critical"
        assert violated is True

    def test_check_value_critical_above_max(self):
        t = AlertThreshold(parameter="temp", critical_max=60.0)
        severity, violated = t.check_value(65.0)
        assert severity == "critical"
        assert violated is True

    def test_check_value_no_thresholds(self):
        t = AlertThreshold(parameter="temp")
        severity, violated = t.check_value(100.0)
        assert severity == "normal"
        assert violated is False


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_default_timestamp(self):
        a = Alert()
        assert a.timestamp is not None
        assert isinstance(a.timestamp, datetime)

    def test_alert_with_timestamp(self):
        ts = datetime(2025, 1, 1)
        a = Alert(timestamp=ts)
        assert a.timestamp == ts

    def test_alert_defaults(self):
        a = Alert()
        assert a.severity == "warning"
        assert a.acknowledged is False
        assert a.escalated is False
        assert a.escalation_level == 0


class TestEscalationRule:
    """Tests for EscalationRule dataclass."""

    def test_defaults(self):
        rule = EscalationRule(
            severity="critical",
            time_window_minutes=5,
            threshold_count=3,
            escalation_action="email_admin",
        )
        assert rule.cooldown_minutes == 60
        assert rule.severity == "critical"


class TestAlertDatabase:
    """Tests for AlertDatabase class."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = AlertDatabase(self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_init_creates_tables(self):
        import sqlite3
        with sqlite3.connect(self.tmp.name) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor}
        assert "alerts" in tables
        assert "thresholds" in tables

    def test_save_and_get_alert(self):
        alert = Alert(
            parameter="temp",
            value=55.0,
            threshold_violated="Above max",
            severity="warning",
            message="Temperature too high",
        )
        alert_id = self.db.save_alert(alert)
        assert alert_id > 0
        alerts = self.db.get_recent_alerts(hours=1)
        assert len(alerts) >= 1
        assert alerts[0].parameter == "temp"

    def test_get_recent_alerts_by_parameter(self):
        a1 = Alert(parameter="temp", value=55.0, severity="warning", message="msg")
        a2 = Alert(parameter="ph", value=3.0, severity="critical", message="msg")
        self.db.save_alert(a1)
        self.db.save_alert(a2)
        alerts = self.db.get_recent_alerts(hours=1, parameter="temp")
        assert all(a.parameter == "temp" for a in alerts)

    def test_get_recent_alerts_unacknowledged(self):
        a1 = Alert(parameter="temp", value=55.0, severity="warning", message="msg")
        alert_id = self.db.save_alert(a1)
        self.db.acknowledge_alert(alert_id, "admin")
        a2 = Alert(parameter="ph", value=3.0, severity="critical", message="msg")
        self.db.save_alert(a2)
        alerts = self.db.get_recent_alerts(hours=1, unacknowledged_only=True)
        assert all(not a.acknowledged for a in alerts)

    def test_acknowledge_alert(self):
        alert = Alert(parameter="temp", value=55.0, severity="warning", message="msg")
        alert_id = self.db.save_alert(alert)
        self.db.acknowledge_alert(alert_id, "admin")
        alerts = self.db.get_recent_alerts(hours=1)
        found = [a for a in alerts if a.id == alert_id]
        assert len(found) == 1
        assert found[0].acknowledged is True

    def test_save_and_get_threshold(self):
        threshold = AlertThreshold(
            parameter="temp",
            min_value=10.0,
            max_value=50.0,
            unit="C",
        )
        self.db.save_threshold(threshold)
        thresholds = self.db.get_thresholds()
        assert "temp" in thresholds
        assert thresholds["temp"].min_value == 10.0


class TestEmailNotificationService:
    """Tests for EmailNotificationService."""

    def test_init(self):
        config = {"server": "smtp.test", "port": 25}
        service = EmailNotificationService(config)
        assert service.smtp_server == "smtp.test"
        assert service.smtp_port == 25

    def test_init_defaults(self):
        service = EmailNotificationService({})
        assert service.smtp_server == "localhost"
        assert service.smtp_port == 587
        assert service.use_tls is True

    def test_send_alert_email_with_tls(self):
        mock_smtp_cls = MagicMock()
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(_mod.smtplib, "SMTP", mock_smtp_cls):
            service = EmailNotificationService({"use_tls": True, "username": "user", "password": "pass"})
            alert = Alert(id=1, parameter="temp", value=55.0, severity="warning", message="test")
            service.send_alert_email(["test@test.com"], alert)
            mock_smtp_cls.assert_called_once()

    def test_send_alert_email_without_tls(self):
        mock_smtp_cls = MagicMock()
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        with patch.object(_mod.smtplib, "SMTP", mock_smtp_cls):
            service = EmailNotificationService({"use_tls": False, "username": ""})
            alert = Alert(id=1, parameter="temp", value=55.0, severity="warning", message="test")
            service.send_alert_email(["test@test.com"], alert)
            mock_smtp_cls.assert_called_once()

    def test_send_alert_email_failure(self):
        with patch.object(_mod.smtplib, "SMTP", side_effect=Exception("fail")):
            service = EmailNotificationService({})
            alert = Alert(id=1, parameter="temp", value=55.0, severity="warning", message="test")
            service.send_alert_email(["test@test.com"], alert)


class TestAlertManager:
    """Tests for AlertManager class."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.manager = AlertManager(db_path=self.tmp.name)

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_init_no_email(self):
        assert self.manager.email_service is None

    def test_init_with_email(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        try:
            mgr = AlertManager(db_path=tmp.name, email_config={"server": "test"})
            assert mgr.email_service is not None
        finally:
            os.unlink(tmp.name)

    def test_set_threshold_new(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0, unit="W/m2")
        assert "power" in self.manager.thresholds
        assert self.manager.thresholds["power"].min_value == 0.5

    def test_set_threshold_update_existing(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        self.manager.set_threshold("power", max_value=3.0)
        assert self.manager.thresholds["power"].max_value == 3.0

    def test_check_value_no_threshold(self):
        result = self.manager.check_value("unknown_param", 10.0)
        assert result is None

    def test_check_value_normal(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        result = self.manager.check_value("power", 1.0)
        assert result is None

    def test_check_value_warning_below_min(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        result = self.manager.check_value("power", 0.3)
        assert result is not None
        assert result.severity == "warning"

    def test_check_value_warning_above_max(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        result = self.manager.check_value("power", 2.5)
        assert result is not None
        assert result.severity == "warning"

    def test_check_value_critical(self):
        self.manager.set_threshold("power", critical_min=0.2, critical_max=2.5)
        result = self.manager.check_value("power", 0.1)
        assert result is not None
        assert result.severity == "critical"

    def test_check_value_critical_above_max(self):
        self.manager.set_threshold("power", critical_max=2.5)
        result = self.manager.check_value("power", 3.0)
        assert result is not None
        assert result.severity == "critical"

    def test_callback_called(self):
        callback = MagicMock()
        self.manager.register_callback(callback)
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        self.manager.check_value("power", 0.3)
        callback.assert_called_once()

    def test_callback_error_handled(self):
        bad_callback = MagicMock(side_effect=Exception("fail"))
        self.manager.register_callback(bad_callback)
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        self.manager.check_value("power", 0.3)

    def test_critical_alert_sends_email(self):
        self.manager.email_service = MagicMock()
        self.manager.admin_emails = ["admin@test.com"]
        self.manager.set_threshold("power", critical_min=0.2)
        self.manager.check_value("power", 0.1)
        self.manager.email_service.send_alert_email.assert_called()

    def test_get_active_alerts(self):
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)
        alerts = self.manager.get_active_alerts()
        assert len(alerts) >= 1

    def test_acknowledge_alert(self):
        self.manager.set_threshold("power", min_value=0.5)
        alert = self.manager.check_value("power", 0.3)
        self.manager.acknowledge_alert(alert.id, "admin")

    def test_get_alert_history(self):
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)
        history = self.manager.get_alert_history(hours=1)
        assert len(history) >= 1

    def test_get_alert_history_with_parameter(self):
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)
        history = self.manager.get_alert_history(hours=1, parameter="power")
        assert all(a.parameter == "power" for a in history)

    def test_export_alert_config(self):
        self.manager.set_threshold("power", min_value=0.5, max_value=2.0)
        config = self.manager.export_alert_config()
        assert "thresholds" in config
        assert "escalation_rules" in config
        assert "email_config" in config
        assert "power" in config["thresholds"]

    def test_import_alert_config(self):
        config = {
            "thresholds": {
                "ph": {"min_value": 6.0, "max_value": 8.0, "unit": ""},
            },
            "escalation_rules": [
                {"severity": "critical", "time_window_minutes": 10, "threshold_count": 5, "escalation_action": "email_admin"},
            ],
            "email_config": {
                "admin_emails": ["admin@test.com"],
                "user_emails": ["user@test.com"],
            },
        }
        self.manager.import_alert_config(config)
        assert "ph" in self.manager.thresholds
        assert len(self.manager.escalation_rules) == 1
        assert self.manager.admin_emails == ["admin@test.com"]

    def test_escalation_email_admin(self):
        self.manager.email_service = MagicMock()
        self.manager.admin_emails = ["admin@test.com"]
        self.manager.escalation_rules = [
            EscalationRule("critical", 60, 1, "email_admin", cooldown_minutes=0),
        ]
        self.manager.set_threshold("power", critical_min=0.2)
        self.manager.check_value("power", 0.1)
        self.manager.email_service.send_alert_email.assert_called()

    def test_escalation_email_all(self):
        self.manager.email_service = MagicMock()
        self.manager.admin_emails = ["admin@test.com"]
        self.manager.user_emails = ["user@test.com"]
        self.manager.escalation_rules = [
            EscalationRule("warning", 60, 1, "email_all", cooldown_minutes=0),
        ]
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)
        self.manager.email_service.send_alert_email.assert_called()

    def test_escalation_dashboard_popup(self):
        def popup_callback(alert):
            pass
        popup_callback.__name__ = "popup_handler"
        self.manager.register_callback(popup_callback)
        self.manager.escalation_rules = [
            EscalationRule("warning", 60, 1, "dashboard_popup", cooldown_minutes=0),
        ]
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)

    def test_escalation_cooldown(self):
        self.manager.email_service = MagicMock()
        self.manager.admin_emails = ["admin@test.com"]
        self.manager.escalation_rules = [
            EscalationRule("warning", 60, 1, "email_admin", cooldown_minutes=60),
        ]
        self.manager.set_threshold("power", min_value=0.5)
        self.manager.check_value("power", 0.3)
        call_count_1 = self.manager.email_service.send_alert_email.call_count
        self.manager.check_value("power", 0.3)
        call_count_2 = self.manager.email_service.send_alert_email.call_count
        assert call_count_2 == call_count_1
