#!/usr/bin/env python3
"""
Configurable Alert Management System
===================================

This module provides a comprehensive alert management system for the MFC monitoring platform.
Supports user-defined thresholds, multiple alert types, and intelligent escalation rules.

Features:
- User-defined threshold settings for key parameters
- Multiple alert types (email, browser notification, dashboard indicator)
- Alert history and acknowledgment system
- Escalation rules for critical alerts
- Integration with email notification system

Created: 2025-08-01
Author: MFC Research Team
"""

import logging
import smtplib
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Defines a threshold for a specific parameter."""
    parameter: str
    min_value: float | None = None
    max_value: float | None = None
    critical_min: float | None = None
    critical_max: float | None = None
    unit: str = ""
    enabled: bool = True

    def check_value(self, value: float) -> tuple[str, bool]:
        """
        Check if value violates threshold.
        Returns: (severity, is_violation)
        """
        if not self.enabled:
            return "normal", False

        # Check critical thresholds first
        if self.critical_min is not None and value < self.critical_min:
            return "critical", True
        if self.critical_max is not None and value > self.critical_max:
            return "critical", True

        # Check warning thresholds
        if self.min_value is not None and value < self.min_value:
            return "warning", True
        if self.max_value is not None and value > self.max_value:
            return "warning", True

        return "normal", False


@dataclass
class Alert:
    """Represents an alert event."""
    id: int | None = None
    timestamp: datetime = None
    parameter: str = ""
    value: float = 0.0
    threshold_violated: str = ""
    severity: str = "warning"  # warning, critical
    message: str = ""
    acknowledged: bool = False
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None
    escalated: bool = False
    escalation_level: int = 0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class EscalationRule:
    """Defines escalation rules for alerts."""
    severity: str
    time_window_minutes: int
    threshold_count: int
    escalation_action: str  # email_admin, email_all, dashboard_popup
    cooldown_minutes: int = 60


class AlertDatabase:
    """Manages alert persistence."""

    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    parameter TEXT,
                    value REAL,
                    threshold_violated TEXT,
                    severity TEXT,
                    message TEXT,
                    acknowledged BOOLEAN,
                    acknowledged_by TEXT,
                    acknowledged_at TIMESTAMP,
                    escalated BOOLEAN,
                    escalation_level INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS thresholds (
                    parameter TEXT PRIMARY KEY,
                    min_value REAL,
                    max_value REAL,
                    critical_min REAL,
                    critical_max REAL,
                    unit TEXT,
                    enabled BOOLEAN
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
                ON alerts(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_parameter
                ON alerts(parameter)
            """)

    def save_alert(self, alert: Alert) -> int:
        """Save an alert to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO alerts (
                    timestamp, parameter, value, threshold_violated,
                    severity, message, acknowledged, acknowledged_by,
                    acknowledged_at, escalated, escalation_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.timestamp, alert.parameter, alert.value,
                alert.threshold_violated, alert.severity, alert.message,
                alert.acknowledged, alert.acknowledged_by,
                alert.acknowledged_at, alert.escalated,
                alert.escalation_level
            ))
            return cursor.lastrowid

    def get_recent_alerts(self, hours: int = 24,
                         parameter: str | None = None,
                         unacknowledged_only: bool = False) -> list[Alert]:
        """Get recent alerts from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = f"""
                SELECT * FROM alerts
                WHERE timestamp > datetime('now', '-{hours} hours')
            """

            if parameter:
                query += f" AND parameter = '{parameter}'"

            if unacknowledged_only:
                query += " AND acknowledged = 0"

            query += " ORDER BY timestamp DESC"

            cursor = conn.execute(query)
            alerts = []

            for row in cursor:
                alert = Alert(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    parameter=row['parameter'],
                    value=row['value'],
                    threshold_violated=row['threshold_violated'],
                    severity=row['severity'],
                    message=row['message'],
                    acknowledged=bool(row['acknowledged']),
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=datetime.fromisoformat(row['acknowledged_at'])
                        if row['acknowledged_at'] else None,
                    escalated=bool(row['escalated']),
                    escalation_level=row['escalation_level']
                )
                alerts.append(alert)

            return alerts

    def acknowledge_alert(self, alert_id: int, user: str):
        """Acknowledge an alert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE alerts
                SET acknowledged = 1,
                    acknowledged_by = ?,
                    acknowledged_at = ?
                WHERE id = ?
            """, (user, datetime.now(), alert_id))

    def save_threshold(self, threshold: AlertThreshold):
        """Save or update a threshold."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO thresholds (
                    parameter, min_value, max_value, critical_min,
                    critical_max, unit, enabled
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                threshold.parameter, threshold.min_value,
                threshold.max_value, threshold.critical_min,
                threshold.critical_max, threshold.unit,
                threshold.enabled
            ))

    def get_thresholds(self) -> dict[str, AlertThreshold]:
        """Get all configured thresholds."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM thresholds")

            thresholds = {}
            for row in cursor:
                threshold = AlertThreshold(
                    parameter=row['parameter'],
                    min_value=row['min_value'],
                    max_value=row['max_value'],
                    critical_min=row['critical_min'],
                    critical_max=row['critical_max'],
                    unit=row['unit'],
                    enabled=bool(row['enabled'])
                )
                thresholds[row['parameter']] = threshold

            return thresholds


class EmailNotificationService:
    """Handles email notifications for alerts."""

    def __init__(self, smtp_config: dict[str, Any]):
        self.smtp_server = smtp_config.get('server', 'localhost')
        self.smtp_port = smtp_config.get('port', 587)
        self.username = smtp_config.get('username', '')
        self.password = smtp_config.get('password', '')
        self.from_email = smtp_config.get('from_email', 'mfc-alerts@system.local')
        self.use_tls = smtp_config.get('use_tls', True)

    def send_alert_email(self, recipients: list[str], alert: Alert):
        """Send an alert notification email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[MFC Alert - {alert.severity.upper()}] {alert.parameter}"

            body = f"""
MFC System Alert Notification
============================

Parameter: {alert.parameter}
Current Value: {alert.value:.3f}
Severity: {alert.severity.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Alert Details:
{alert.message}

Threshold Violated: {alert.threshold_violated}

This is an automated alert from the MFC monitoring system.
Please log in to acknowledge this alert and take appropriate action.

Alert ID: {alert.id}
"""

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Alert email sent to {recipients} for {alert.parameter}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class AlertManager:
    """Main alert management system."""

    def __init__(self, db_path: str = "alerts.db",
                 email_config: dict[str, Any] | None = None):
        self.db = AlertDatabase(db_path)
        self.thresholds = self.db.get_thresholds()
        self.email_service = EmailNotificationService(email_config) if email_config else None

        # Alert callbacks
        self.alert_callbacks: list[Callable[[Alert], None]] = []

        # Escalation tracking
        self.escalation_tracker = defaultdict(list)
        self.last_escalation = {}

        # Default escalation rules
        self.escalation_rules = [
            EscalationRule("critical", 5, 3, "email_admin"),
            EscalationRule("warning", 30, 5, "dashboard_popup"),
            EscalationRule("critical", 15, 5, "email_all")
        ]

        # Default recipients
        self.admin_emails = []
        self.user_emails = []

    def set_threshold(self, parameter: str, **kwargs):
        """Set or update a threshold for a parameter."""
        if parameter in self.thresholds:
            # Update existing threshold
            threshold = self.thresholds[parameter]
            for key, value in kwargs.items():
                if hasattr(threshold, key):
                    setattr(threshold, key, value)
        else:
            # Create new threshold
            threshold = AlertThreshold(parameter=parameter, **kwargs)
            self.thresholds[parameter] = threshold

        self.db.save_threshold(threshold)
        logger.info(f"Threshold updated for {parameter}")

    def check_value(self, parameter: str, value: float) -> Alert | None:
        """Check if a value violates thresholds and create alert if needed."""
        if parameter not in self.thresholds:
            return None

        threshold = self.thresholds[parameter]
        severity, violated = threshold.check_value(value)

        if violated:
            # Determine which threshold was violated
            if severity == "critical":
                if threshold.critical_min and value < threshold.critical_min:
                    threshold_desc = f"Below critical minimum ({threshold.critical_min} {threshold.unit})"
                else:
                    threshold_desc = f"Above critical maximum ({threshold.critical_max} {threshold.unit})"
            else:
                if threshold.min_value and value < threshold.min_value:
                    threshold_desc = f"Below minimum ({threshold.min_value} {threshold.unit})"
                else:
                    threshold_desc = f"Above maximum ({threshold.max_value} {threshold.unit})"

            alert = Alert(
                parameter=parameter,
                value=value,
                threshold_violated=threshold_desc,
                severity=severity,
                message=f"{parameter} value of {value:.3f} {threshold.unit} violates threshold: {threshold_desc}"
            )

            # Save alert
            alert.id = self.db.save_alert(alert)

            # Process alert
            self._process_alert(alert)

            return alert

        return None

    def _process_alert(self, alert: Alert):
        """Process a new alert through the notification system."""
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # Check escalation rules
        self._check_escalation(alert)

        # Send notifications based on severity
        if alert.severity == "critical" and self.email_service:
            # Immediate email for critical alerts
            if self.admin_emails:
                self.email_service.send_alert_email(self.admin_emails, alert)

    def _check_escalation(self, alert: Alert):
        """Check if alert should be escalated based on rules."""
        now = datetime.now()

        # Track alert for escalation
        self.escalation_tracker[alert.parameter].append((now, alert))

        # Clean old entries
        cutoff = now - timedelta(hours=1)
        self.escalation_tracker[alert.parameter] = [
            (t, a) for t, a in self.escalation_tracker[alert.parameter]
            if t > cutoff
        ]

        # Check each escalation rule
        for rule in self.escalation_rules:
            if rule.severity != alert.severity:
                continue

            # Count alerts in time window
            window_start = now - timedelta(minutes=rule.time_window_minutes)
            recent_alerts = [
                a for t, a in self.escalation_tracker[alert.parameter]
                if t > window_start and a.severity == rule.severity
            ]

            if len(recent_alerts) >= rule.threshold_count:
                # Check cooldown
                last_esc = self.last_escalation.get((alert.parameter, rule.escalation_action))
                if last_esc and (now - last_esc) < timedelta(minutes=rule.cooldown_minutes):
                    continue

                # Perform escalation
                self._escalate_alert(alert, rule)
                self.last_escalation[(alert.parameter, rule.escalation_action)] = now

    def _escalate_alert(self, alert: Alert, rule: EscalationRule):
        """Perform escalation action."""
        logger.warning(f"Escalating alert for {alert.parameter}: {rule.escalation_action}")

        alert.escalated = True
        alert.escalation_level += 1

        if rule.escalation_action == "email_admin" and self.email_service:
            if self.admin_emails:
                self.email_service.send_alert_email(self.admin_emails, alert)

        elif rule.escalation_action == "email_all" and self.email_service:
            all_recipients = list(set(self.admin_emails + self.user_emails))
            if all_recipients:
                self.email_service.send_alert_email(all_recipients, alert)

        elif rule.escalation_action == "dashboard_popup":
            # This would be handled by the GUI
            for callback in self.alert_callbacks:
                if hasattr(callback, '__name__') and 'popup' in callback.__name__:
                    callback(alert)

    def register_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def get_active_alerts(self) -> list[Alert]:
        """Get all unacknowledged alerts."""
        return self.db.get_recent_alerts(hours=24*7, unacknowledged_only=True)

    def acknowledge_alert(self, alert_id: int, user: str = "system"):
        """Acknowledge an alert."""
        self.db.acknowledge_alert(alert_id, user)
        logger.info(f"Alert {alert_id} acknowledged by {user}")

    def get_alert_history(self, hours: int = 24,
                         parameter: str | None = None) -> list[Alert]:
        """Get alert history."""
        return self.db.get_recent_alerts(hours, parameter)

    def export_alert_config(self) -> dict[str, Any]:
        """Export current alert configuration."""
        return {
            "thresholds": {
                param: asdict(threshold)
                for param, threshold in self.thresholds.items()
            },
            "escalation_rules": [
                {
                    "severity": rule.severity,
                    "time_window_minutes": rule.time_window_minutes,
                    "threshold_count": rule.threshold_count,
                    "escalation_action": rule.escalation_action,
                    "cooldown_minutes": rule.cooldown_minutes
                }
                for rule in self.escalation_rules
            ],
            "email_config": {
                "admin_emails": self.admin_emails,
                "user_emails": self.user_emails
            }
        }

    def import_alert_config(self, config: dict[str, Any]):
        """Import alert configuration."""
        # Import thresholds
        for param, threshold_data in config.get("thresholds", {}).items():
            self.set_threshold(param, **threshold_data)

        # Import escalation rules
        self.escalation_rules = []
        for rule_data in config.get("escalation_rules", []):
            self.escalation_rules.append(EscalationRule(**rule_data))

        # Import email config
        email_config = config.get("email_config", {})
        self.admin_emails = email_config.get("admin_emails", [])
        self.user_emails = email_config.get("user_emails", [])


# Example usage and testing
if __name__ == "__main__":
    # Example email configuration
    email_config = {
        'server': 'smtp.gmail.com',
        'port': 587,
        'username': 'mfc-alerts@example.com',
        'password': 'your-password',
        'from_email': 'mfc-alerts@example.com',
        'use_tls': True
    }

    # Create alert manager
    manager = AlertManager(email_config=email_config)

    # Set thresholds
    manager.set_threshold(
        "power_density",
        min_value=0.5,
        max_value=2.0,
        critical_min=0.2,
        critical_max=2.5,
        unit="W/m²"
    )

    manager.set_threshold(
        "substrate_concentration",
        min_value=5.0,
        max_value=50.0,
        critical_max=60.0,
        unit="mM"
    )

    manager.set_threshold(
        "pH",
        min_value=6.5,
        max_value=7.5,
        critical_min=6.0,
        critical_max=8.0,
        unit=""
    )

    # Test alert generation
    print("Testing alert system...")

    # Normal value - no alert
    alert = manager.check_value("power_density", 1.0)
    print(f"Power density 1.0: {alert}")

    # Warning alert
    alert = manager.check_value("power_density", 0.4)
    print(f"Power density 0.4: {alert}")

    # Critical alert
    alert = manager.check_value("pH", 5.5)
    print(f"pH 5.5: {alert}")

    # Get active alerts
    active = manager.get_active_alerts()
    print(f"\nActive alerts: {len(active)}")
    for a in active:
        print(f"  - {a.parameter}: {a.message}")
