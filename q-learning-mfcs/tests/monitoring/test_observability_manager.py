#!/usr/bin/env python3
"""
Tests for MFC Observability Manager
==================================

Comprehensive test suite for the MFC observability system including
metrics collection, monitoring, and alerting.

Created: 2025-08-05
Author: TDD Agent 1 - ObservabilityManager Implementation
"""

import unittest
import tempfile
import time
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.monitoring.observability_manager import (
    ObservabilityManager,
    HealthStatus,
    AlertSeverity,
    ServiceHealth,
    AlertCondition,
    Alert,
    MetricsCollector,
    MetricType,
    Metric,
    get_default_manager,
)


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector."""

    def setUp(self):
        self.collector = MetricsCollector(retention_hours=1)

    def test_counter_metrics(self):
        """Test counter metric recording and retrieval."""
        metric = Metric(
            name="requests_total",
            value=1,
            metric_type=MetricType.COUNTER,
            service="mfc_service"
        )
        
        # Record multiple counter increments
        for _ in range(5):
            self.collector.record_metric(metric)
        
        # Check counter value
        value = self.collector.get_counter_value("mfc_service", "requests_total")
        self.assertEqual(value, 5)

    def test_gauge_metrics(self):
        """Test gauge metric recording and retrieval."""
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            metric_type=MetricType.GAUGE,
            service="mfc_service"
        )
        
        self.collector.record_metric(metric)
        
        # Update gauge value
        metric.value = 82.3
        self.collector.record_metric(metric)
        
        # Check latest gauge value
        value = self.collector.get_gauge_value("mfc_service", "cpu_usage")
        self.assertEqual(value, 82.3)

    def test_histogram_metrics(self):
        """Test histogram metric recording and statistics."""
        values = [100, 200, 150, 300, 250]
        
        for val in values:
            metric = Metric(
                name="response_time",
                value=val,
                metric_type=MetricType.HISTOGRAM,
                service="mfc_service"
            )
            self.collector.record_metric(metric)
        
        # Get histogram statistics
        stats = self.collector.get_histogram_stats("mfc_service", "response_time")
        
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["min"], 100)
        self.assertEqual(stats["max"], 300)
        self.assertEqual(stats["mean"], 200)

    def test_timer_metrics(self):
        """Test timer metric recording and statistics."""
        durations = [50, 75, 60, 90, 70]
        
        for duration in durations:
            metric = Metric(
                name="processing_time",
                value=duration,
                metric_type=MetricType.TIMER,
                service="mfc_service"
            )
            self.collector.record_metric(metric)
        
        # Get timer statistics (same as histogram)
        stats = self.collector.get_timer_stats("mfc_service", "processing_time")
        
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], 69.0, places=1)

    def test_metrics_summary(self):
        """Test comprehensive metrics summary."""
        # Add various metrics
        counter_metric = Metric("requests", 10, MetricType.COUNTER, service="test")
        gauge_metric = Metric("memory", 512, MetricType.GAUGE, service="test")
        hist_metric = Metric("latency", 100, MetricType.HISTOGRAM, service="test")
        
        self.collector.record_metric(counter_metric)
        self.collector.record_metric(gauge_metric)
        self.collector.record_metric(hist_metric)
        
        summary = self.collector.get_metrics_summary()
        
        self.assertIn("counters", summary)
        self.assertIn("gauges", summary)
        self.assertIn("histograms", summary)
        self.assertIn("timers", summary)
        
        self.assertEqual(summary["counters"]["test.requests"], 10)
        self.assertEqual(summary["gauges"]["test.memory"], 512)

    def test_metric_history(self):
        """Test metric history retrieval."""
        metric = Metric("test_metric", 100, MetricType.GAUGE, service="test")
        self.collector.record_metric(metric)
        
        history = self.collector.get_metric_history("test", "test_metric", minutes=60)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].value, 100)

    def test_metric_cleanup(self):
        """Test old metrics cleanup."""
        # Create collector with very short retention
        collector = MetricsCollector(retention_hours=0.001)  # ~3.6 seconds
        
        metric = Metric("test", 1, MetricType.COUNTER, service="test")
        collector.record_metric(metric)
        
        # Wait for retention period to pass
        time.sleep(0.1)
        
        # Trigger cleanup
        collector._cleanup_old_metrics()
        
        # Metric should still be there (retention not exceeded)
        self.assertEqual(len(collector.metrics["test.test"]), 1)


class TestObservabilityManager(unittest.TestCase):
    """Test cases for ObservabilityManager."""

    def setUp(self):
        self.manager = ObservabilityManager(check_interval=1.0)

    def tearDown(self):
        if self.manager._running:
            self.manager.stop_monitoring()

    def test_service_registration(self):
        """Test service registration."""
        def mock_health_check():
            return {'status': HealthStatus.HEALTHY, 'response_time_ms': 10.0}
        
        self.manager.register_service("test_service", mock_health_check)
        
        self.assertIn("test_service", self.manager.services)
        self.assertIn("test_service", self.manager.health_callbacks)
        
        service = self.manager.services["test_service"]
        self.assertEqual(service.service_name, "test_service")
        self.assertEqual(service.status, HealthStatus.HEALTHY)

    def test_alert_condition_management(self):
        """Test alert condition management."""
        condition = AlertCondition(
            name="test_condition",
            metric_name="test.metric",
            threshold=80.0,
            comparison="gt",
            duration_seconds=60,
            severity=AlertSeverity.WARNING
        )
        
        self.manager.add_alert_condition(condition)
        
        self.assertIn("test_condition", self.manager.alert_conditions)
        stored_condition = self.manager.alert_conditions["test_condition"]
        self.assertEqual(stored_condition.name, "test_condition")
        self.assertEqual(stored_condition.threshold, 80.0)
        
        # Test removal
        self.manager.remove_alert_condition("test_condition")
        self.assertNotIn("test_condition", self.manager.alert_conditions)

    def test_alert_handler_registration(self):
        """Test alert handler registration."""
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        self.manager.add_alert_handler(alert_handler)
        
        self.assertEqual(len(self.manager.alert_handlers), 1)

    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        self.assertFalse(self.manager._running)
        
        # Start monitoring
        self.manager.start_monitoring()
        self.assertTrue(self.manager._running)
        self.assertIsNotNone(self.manager._monitor_thread)
        
        # Stop monitoring
        self.manager.stop_monitoring()
        self.assertFalse(self.manager._running)

    def test_service_health_retrieval(self):
        """Test service health retrieval."""
        # Register a service
        self.manager.register_service("test_service")
        
        # Get health for specific service
        health = self.manager.get_service_health("test_service")
        self.assertIn("test_service", health)
        self.assertIsInstance(health["test_service"], ServiceHealth)
        
        # Get all service health
        all_health = self.manager.get_service_health()
        self.assertIn("test_service", all_health)

    def test_alert_lifecycle(self):
        """Test alert creation and resolution."""
        alert = Alert(
            alert_id="test_alert_1",
            condition_name="test_condition",
            service_name="test_service",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            triggered_at=datetime.now()
        )
        
        # Manually add alert to test resolution
        with self.manager._lock:
            self.manager.active_alerts[alert.alert_id] = alert
            self.manager.alert_history.append(alert)
        
        # Check active alerts
        active_alerts = self.manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0].alert_id, "test_alert_1")
        
        # Acknowledge alert
        self.manager.acknowledge_alert("test_alert_1")
        self.assertTrue(self.manager.active_alerts["test_alert_1"].acknowledged)
        
        # Resolve alert
        self.manager.resolve_alert("test_alert_1")
        self.assertEqual(len(self.manager.get_active_alerts()), 0)
        
        # Check alert history
        history = self.manager.get_alert_history()
        self.assertGreater(len(history), 0)

    def test_system_status(self):
        """Test system status reporting."""
        # Register a service
        self.manager.register_service("test_service")
        
        status = self.manager.get_system_status()
        
        self.assertIn("overall_status", status)
        self.assertIn("timestamp", status)
        self.assertIn("services_count", status)
        self.assertIn("healthy_services", status)
        self.assertIn("active_alerts", status)
        self.assertIn("services", status)
        self.assertIn("alerts", status)
        
        self.assertEqual(status["services_count"], 1)
        self.assertEqual(status["healthy_services"], 1)
        self.assertEqual(status["overall_status"], "healthy")

    def test_status_export(self):
        """Test status export functionality."""
        # Register a service
        self.manager.register_service("test_service")
        
        exported_status = self.manager.export_status("json")
        
        # Should be valid JSON
        parsed_status = json.loads(exported_status)
        self.assertIn("overall_status", parsed_status)
        self.assertIn("services_count", parsed_status)
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            self.manager.export_status("xml")

    def test_threshold_evaluation(self):
        """Test threshold evaluation logic."""
        # Test different comparison operators
        self.assertTrue(self.manager._evaluate_threshold(90, 80, "gt"))
        self.assertFalse(self.manager._evaluate_threshold(70, 80, "gt"))
        
        self.assertTrue(self.manager._evaluate_threshold(80, 80, "gte"))
        self.assertTrue(self.manager._evaluate_threshold(90, 80, "gte"))
        
        self.assertTrue(self.manager._evaluate_threshold(70, 80, "lt"))
        self.assertFalse(self.manager._evaluate_threshold(90, 80, "lt"))
        
        self.assertTrue(self.manager._evaluate_threshold(80, 80, "lte"))
        self.assertTrue(self.manager._evaluate_threshold(70, 80, "lte"))
        
        self.assertTrue(self.manager._evaluate_threshold(80, 80, "eq"))
        self.assertFalse(self.manager._evaluate_threshold(90, 80, "eq"))
        
        # Test unsupported comparison
        self.assertFalse(self.manager._evaluate_threshold(90, 80, "invalid"))

    def test_default_alert_conditions(self):
        """Test that default alert conditions are set up."""
        # Default conditions should be present
        self.assertGreater(len(self.manager.alert_conditions), 0)
        
        # Check for some expected default conditions
        condition_names = list(self.manager.alert_conditions.keys())
        self.assertIn("high_cpu_usage", condition_names)
        self.assertIn("critical_cpu_usage", condition_names)
        self.assertIn("high_memory_usage", condition_names)
        self.assertIn("mfc_low_power", condition_names)
        self.assertIn("qlearning_poor_convergence", condition_names)

    def test_get_default_manager(self):
        """Test default manager singleton."""
        manager1 = get_default_manager()
        manager2 = get_default_manager()
        
        # Should be the same instance
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ObservabilityManager)


class TestDataClasses(unittest.TestCase):
    """Test the data classes used in observability."""

    def test_service_health_creation(self):
        """Test ServiceHealth data class."""
        health = ServiceHealth(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time_ms=10.5,
            error_rate=0.01,
            availability=99.9,
            details={"version": "1.0"}
        )
        
        self.assertEqual(health.service_name, "test_service")
        self.assertEqual(health.status, HealthStatus.HEALTHY)
        self.assertEqual(health.response_time_ms, 10.5)
        self.assertEqual(health.error_rate, 0.01)
        self.assertEqual(health.availability, 99.9)
        self.assertEqual(health.details["version"], "1.0")

    def test_alert_condition_creation(self):
        """Test AlertCondition data class."""
        condition = AlertCondition(
            name="test_condition",
            metric_name="test.metric",
            threshold=80.0,
            comparison="gt",
            duration_seconds=300,
            severity=AlertSeverity.WARNING,
            enabled=True
        )
        
        self.assertEqual(condition.name, "test_condition")
        self.assertEqual(condition.metric_name, "test.metric")
        self.assertEqual(condition.threshold, 80.0)
        self.assertEqual(condition.comparison, "gt")
        self.assertEqual(condition.duration_seconds, 300)
        self.assertEqual(condition.severity, AlertSeverity.WARNING)
        self.assertTrue(condition.enabled)

    def test_alert_creation(self):
        """Test Alert data class."""
        now = datetime.now()
        alert = Alert(
            alert_id="alert_123",
            condition_name="test_condition",
            service_name="test_service",
            severity=AlertSeverity.ERROR,
            message="Test alert message",
            triggered_at=now,
            resolved_at=None,
            acknowledged=False
        )
        
        self.assertEqual(alert.alert_id, "alert_123")
        self.assertEqual(alert.condition_name, "test_condition")
        self.assertEqual(alert.service_name, "test_service")
        self.assertEqual(alert.severity, AlertSeverity.ERROR)
        self.assertEqual(alert.message, "Test alert message")
        self.assertEqual(alert.triggered_at, now)
        self.assertIsNone(alert.resolved_at)
        self.assertFalse(alert.acknowledged)

    def test_metric_creation(self):
        """Test Metric data class."""
        now = datetime.now()
        metric = Metric(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            service="test_service",
            labels={"env": "test", "region": "us-east-1"},
            timestamp=now
        )
        
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.value, 42.5)
        self.assertEqual(metric.metric_type, MetricType.GAUGE)
        self.assertEqual(metric.service, "test_service")
        self.assertEqual(metric.labels["env"], "test")
        self.assertEqual(metric.labels["region"], "us-east-1")
        self.assertEqual(metric.timestamp, now)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)