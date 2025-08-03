#!/usr/bin/env python3
"""
Tests for TTS Alert Manager
===========================

Test suite for TTS-specific alert management including thresholds,
health checks, performance analysis, and alert processing.

Created: 2025-08-03
Author: Agent Eta - Monitoring and Observability Specialist
"""

import unittest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from monitoring.observability_manager import ObservabilityManager
from monitoring.tts_alert_manager import (
    TTSAlertManager,
    TTSAlert,
    TTSAlertType,
    TTSAlertSeverity,
    TTSThreshold,
    TTSHealthCheck,
    TTSPerformanceAnalyzer
)


class TestTTSPerformanceAnalyzer(unittest.TestCase):
    """Test cases for TTSPerformanceAnalyzer."""

    def setUp(self):
        self.analyzer = TTSPerformanceAnalyzer(window_size=10)

    def test_baseline_calculation(self):
        """Test baseline metrics calculation."""
        # Add enough data to establish baseline
        values = [100, 110, 95, 105, 102, 98, 108, 101, 99, 107]
        
        for value in values:
            self.analyzer.add_metric("pyttsx3", "duration_ms", value)
        
        # Check baseline was established
        baseline = self.analyzer.baseline_metrics.get("pyttsx3.duration_ms")
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline['mean'], 102.5, places=1)
        self.assertGreater(baseline['std'], 0)

    def test_anomaly_detection_z_score(self):
        """Test Z-score based anomaly detection."""
        # Establish baseline with normal values
        normal_values = [100] * 10
        for value in normal_values:
            self.analyzer.add_metric("test_engine", "metric", value)
        
        # Test normal value
        result = self.analyzer.detect_anomalies("test_engine", "metric", 101)
        self.assertFalse(result["is_anomaly"])
        
        # Test anomalous value (high Z-score)
        result = self.analyzer.detect_anomalies("test_engine", "metric", 200)
        self.assertTrue(result["is_anomaly"])
        self.assertIn("z_score", result)

    def test_anomaly_detection_percentile(self):
        """Test percentile-based anomaly detection."""
        # Establish baseline
        values = list(range(90, 110))  # 90-109
        for value in values:
            self.analyzer.add_metric("test_engine", "metric", value)
        
        # Test value significantly above P95
        p95_baseline = self.analyzer.baseline_metrics["test_engine.metric"]["p95"]
        anomalous_value = p95_baseline * 2
        
        result = self.analyzer.detect_anomalies("test_engine", "metric", anomalous_value)
        self.assertTrue(result["is_anomaly"])
        self.assertIn("percentile_ratio", result)

    def test_insufficient_baseline_data(self):
        """Test behavior with insufficient baseline data."""
        # Add only a few data points
        self.analyzer.add_metric("new_engine", "new_metric", 100)
        
        result = self.analyzer.detect_anomalies("new_engine", "new_metric", 200)
        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["reason"], "insufficient_baseline_data")


class TestTTSThreshold(unittest.TestCase):
    """Test cases for TTSThreshold."""

    def test_threshold_creation(self):
        """Test TTSThreshold creation and configuration."""
        threshold = TTSThreshold(
            engine_type="pyttsx3",
            metric_name="duration_ms",
            warning_threshold=2000,
            critical_threshold=5000,
            emergency_threshold=10000
        )
        
        self.assertEqual(threshold.engine_type, "pyttsx3")
        self.assertEqual(threshold.metric_name, "duration_ms")
        self.assertEqual(threshold.warning_threshold, 2000)
        self.assertEqual(threshold.critical_threshold, 5000)
        self.assertEqual(threshold.emergency_threshold, 10000)
        self.assertTrue(threshold.enabled)


class TestTTSHealthCheck(unittest.TestCase):
    """Test cases for TTSHealthCheck."""

    def test_health_check_creation(self):
        """Test TTSHealthCheck creation and configuration."""
        def mock_check():
            return True
        
        health_check = TTSHealthCheck(
            name="engine_availability",
            engine_type="pyttsx3",
            check_function=mock_check,
            interval_seconds=300
        )
        
        self.assertEqual(health_check.name, "engine_availability")
        self.assertEqual(health_check.engine_type, "pyttsx3")
        self.assertEqual(health_check.interval_seconds, 300)
        self.assertTrue(health_check.enabled)
        self.assertEqual(health_check.consecutive_failures, 0)

    def test_health_check_execution(self):
        """Test health check function execution."""
        call_count = 0
        
        def mock_check():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Fail on third call
        
        health_check = TTSHealthCheck(
            name="test_check",
            engine_type="test",
            check_function=mock_check
        )
        
        # First call should succeed
        result1 = health_check.check_function()
        self.assertTrue(result1)
        
        # Second call should succeed  
        result2 = health_check.check_function()
        self.assertTrue(result2)
        
        # Third call should fail
        result3 = health_check.check_function()
        self.assertFalse(result3)


class TestTTSAlert(unittest.TestCase):
    """Test cases for TTSAlert."""

    def test_alert_creation(self):
        """Test TTSAlert creation with all fields."""
        alert = TTSAlert(
            alert_type=TTSAlertType.HIGH_LATENCY,
            severity=TTSAlertSeverity.WARNING,
            engine_type="pyttsx3",
            message="High latency detected",
            metrics={"duration_ms": 3000},
            resolution_steps=["Check system resources", "Restart engine"]
        )
        
        self.assertEqual(alert.alert_type, TTSAlertType.HIGH_LATENCY)
        self.assertEqual(alert.severity, TTSAlertSeverity.WARNING)
        self.assertEqual(alert.engine_type, "pyttsx3")
        self.assertEqual(alert.message, "High latency detected")
        self.assertFalse(alert.acknowledged)
        self.assertFalse(alert.resolved)

    def test_alert_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        before_time = time.time()
        
        alert = TTSAlert(
            alert_type=TTSAlertType.LOW_SUCCESS_RATE,
            severity=TTSAlertSeverity.CRITICAL,
            engine_type="coqui",
            message="Low success rate"
        )
        
        after_time = time.time()
        
        self.assertGreaterEqual(alert.timestamp, before_time)
        self.assertLessEqual(alert.timestamp, after_time)


class TestTTSAlertManager(unittest.TestCase):
    """Test cases for TTSAlertManager."""

    def setUp(self):
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_log_file.close()
        
        self.obs_manager = ObservabilityManager("test_service", {
            "log_file": self.temp_log_file.name
        })
        self.alert_manager = TTSAlertManager(self.obs_manager)

    def tearDown(self):
        self.obs_manager.shutdown()
        os.unlink(self.temp_log_file.name)

    def test_default_thresholds(self):
        """Test that default thresholds are properly set up."""
        # Check that default thresholds exist
        self.assertIn("pyttsx3.synthesis_duration_ms", self.alert_manager.tts_thresholds)
        self.assertIn("coqui.synthesis_duration_ms", self.alert_manager.tts_thresholds)
        self.assertIn("pyttsx3.success_rate", self.alert_manager.tts_thresholds)
        
        # Check threshold values
        pyttsx3_duration_threshold = self.alert_manager.tts_thresholds["pyttsx3.synthesis_duration_ms"]
        self.assertEqual(pyttsx3_duration_threshold.warning_threshold, 2000)
        self.assertEqual(pyttsx3_duration_threshold.critical_threshold, 5000)

    def test_custom_threshold_addition(self):
        """Test adding custom thresholds."""
        custom_threshold = TTSThreshold(
            engine_type="custom_engine",
            metric_name="custom_metric",
            warning_threshold=100,
            critical_threshold=200
        )
        
        self.alert_manager.add_threshold(custom_threshold)
        
        key = "custom_engine.custom_metric"
        self.assertIn(key, self.alert_manager.tts_thresholds)
        self.assertEqual(self.alert_manager.tts_thresholds[key].warning_threshold, 100)

    def test_health_check_addition(self):
        """Test adding custom health checks."""
        def custom_check():
            return True
        
        health_check = TTSHealthCheck(
            name="custom_check",
            engine_type="custom_engine",
            check_function=custom_check
        )
        
        self.alert_manager.add_health_check(health_check)
        
        key = "custom_engine.custom_check"
        self.assertIn(key, self.alert_manager.health_checks)

    def test_high_latency_alert_generation(self):
        """Test high latency alert generation."""
        # Test warning threshold
        metrics = {"synthesis_duration_ms": 3000}  # Above warning (2000) but below critical (5000)
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, TTSAlertType.HIGH_LATENCY)
        self.assertEqual(alert.severity, TTSAlertSeverity.WARNING)
        self.assertEqual(alert.engine_type, "pyttsx3")

    def test_critical_latency_alert_generation(self):
        """Test critical latency alert generation."""
        metrics = {"synthesis_duration_ms": 7000}  # Above critical threshold (5000)
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, TTSAlertType.HIGH_LATENCY)
        self.assertEqual(alert.severity, TTSAlertSeverity.CRITICAL)

    def test_low_success_rate_alert(self):
        """Test low success rate alert generation."""
        metrics = {"success_rate": 0.85}  # Below critical threshold (0.90)
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, TTSAlertType.LOW_SUCCESS_RATE)
        self.assertEqual(alert.severity, TTSAlertSeverity.CRITICAL)

    def test_queue_buildup_alert(self):
        """Test queue buildup alert generation."""
        metrics = {"queue_size": 15}  # Above warning threshold (10)
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, TTSAlertType.QUEUE_BUILDUP)
        self.assertEqual(alert.severity, TTSAlertSeverity.WARNING)

    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        metrics = {"synthesis_duration_ms": 3000}
        
        # Generate first alert
        alerts1 = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        self.assertEqual(len(alerts1), 1)
        
        # Immediately generate second alert - should be suppressed by cooldown
        alerts2 = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        self.assertEqual(len(alerts2), 0)

    def test_multiple_metric_alerts(self):
        """Test multiple alerts from different metrics."""
        metrics = {
            "synthesis_duration_ms": 7000,  # Critical
            "success_rate": 0.85,  # Critical
            "queue_size": 15  # Warning
        }
        
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        # Should generate multiple alerts
        self.assertEqual(len(alerts), 3)
        
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn(TTSAlertType.HIGH_LATENCY, alert_types)
        self.assertIn(TTSAlertType.LOW_SUCCESS_RATE, alert_types)
        self.assertIn(TTSAlertType.QUEUE_BUILDUP, alert_types)

    def test_health_check_execution(self):
        """Test health check execution and alert generation."""
        failure_count = 0
        
        def failing_check():
            nonlocal failure_count
            failure_count += 1
            return failure_count <= 2  # Fail after 2 successes
        
        health_check = TTSHealthCheck(
            name="test_check",
            engine_type="test_engine",
            check_function=failing_check,
            failure_threshold=2,
            interval_seconds=0  # No interval for testing
        )
        
        self.alert_manager.add_health_check(health_check)
        
        # First runs should succeed
        alerts1 = self.alert_manager.run_health_checks()
        self.assertEqual(len(alerts1), 0)
        
        alerts2 = self.alert_manager.run_health_checks()
        self.assertEqual(len(alerts2), 0)
        
        # Third run should start failing
        alerts3 = self.alert_manager.run_health_checks()
        self.assertEqual(len(alerts3), 0)  # Below failure threshold
        
        # Fourth run should trigger alert (reached failure threshold)
        alerts4 = self.alert_manager.run_health_checks()
        self.assertEqual(len(alerts4), 1)
        
        alert = alerts4[0]
        self.assertEqual(alert.alert_type, TTSAlertType.ENGINE_UNAVAILABLE)
        self.assertEqual(alert.engine_type, "test_engine")

    def test_health_check_recovery(self):
        """Test health check recovery alert generation."""
        recovery_count = 0
        
        def recovering_check():
            nonlocal recovery_count
            recovery_count += 1
            return recovery_count > 3  # Start succeeding after 3 failures
        
        health_check = TTSHealthCheck(
            name="recovery_check",
            engine_type="test_engine",
            check_function=recovering_check,
            failure_threshold=1,
            recovery_threshold=1,
            interval_seconds=0
        )
        
        # Start with some failures
        health_check.consecutive_failures = 5
        
        self.alert_manager.add_health_check(health_check)
        
        # Run health check - should succeed and generate recovery alert
        alerts = self.alert_manager.run_health_checks()
        
        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, TTSAlertType.ENGINE_FAILURE)
        self.assertEqual(alert.severity, TTSAlertSeverity.INFO)
        self.assertIn("recovered", alert.message)

    def test_notification_callbacks(self):
        """Test alert notification callbacks."""
        received_alerts = []
        
        def alert_callback(alert):
            received_alerts.append(alert)
        
        self.alert_manager.register_notification_callback(alert_callback)
        
        # Generate an alert
        metrics = {"synthesis_duration_ms": 7000}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        # Check that callback was called
        self.assertEqual(len(received_alerts), 1)
        self.assertEqual(received_alerts[0].alert_type, TTSAlertType.HIGH_LATENCY)

    def test_alert_acknowledgment(self):
        """Test alert acknowledgment functionality."""
        # Generate an alert
        metrics = {"synthesis_duration_ms": 7000}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        alert = alerts[0]
        
        # Check alert is not acknowledged initially
        self.assertFalse(alert.acknowledged)
        
        # Acknowledge the alert
        self.alert_manager.acknowledge_alert(
            alert.engine_type, alert.alert_type, alert.severity, "test_user"
        )
        
        # Check alert is now acknowledged
        self.assertTrue(alert.acknowledged)

    def test_alert_resolution(self):
        """Test alert resolution functionality."""
        # Generate an alert
        metrics = {"synthesis_duration_ms": 7000}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        alert = alerts[0]
        
        # Check alert is in active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        self.assertIn(alert, active_alerts)
        
        # Resolve the alert
        self.alert_manager.resolve_alert(
            alert.engine_type, alert.alert_type, alert.severity, "test_user"
        )
        
        # Check alert is resolved and removed from active alerts
        self.assertTrue(alert.resolved)
        active_alerts_after = self.alert_manager.get_active_alerts()
        self.assertNotIn(alert, active_alerts_after)

    def test_alert_summary(self):
        """Test alert summary generation."""
        # Generate different types of alerts
        self.alert_manager.check_tts_metrics("pyttsx3", {"synthesis_duration_ms": 7000})  # Critical
        self.alert_manager.check_tts_metrics("coqui", {"success_rate": 0.88})  # Critical
        self.alert_manager.check_tts_metrics("hybrid", {"queue_size": 15})  # Warning
        
        summary = self.alert_manager.get_alert_summary()
        
        self.assertEqual(summary["total_active_alerts"], 3)
        self.assertEqual(summary["alerts_by_severity"]["critical"], 2)
        self.assertEqual(summary["alerts_by_severity"]["warning"], 1)
        self.assertEqual(summary["alerts_by_engine"]["pyttsx3"], 1)
        self.assertEqual(summary["alerts_by_engine"]["coqui"], 1)
        self.assertEqual(summary["alerts_by_engine"]["hybrid"], 1)

    def test_resolution_steps_generation(self):
        """Test resolution steps generation for different alert types."""
        # Test high latency resolution steps
        metrics = {"synthesis_duration_ms": 7000}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        alert = alerts[0]
        
        self.assertGreater(len(alert.resolution_steps), 0)
        self.assertTrue(any("resource usage" in step.lower() for step in alert.resolution_steps))
        
        # Test low success rate resolution steps
        metrics = {"success_rate": 0.85}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        alert = alerts[0]
        
        self.assertGreater(len(alert.resolution_steps), 0)
        self.assertTrue(any("error logs" in step.lower() for step in alert.resolution_steps))

    def test_anomaly_detection_integration(self):
        """Test integration with performance analyzer for anomaly detection."""
        # Build baseline
        normal_values = [1000, 1100, 1050, 1080, 1020]
        for value in normal_values:
            self.alert_manager.performance_analyzer.add_metric("pyttsx3", "synthesis_duration_ms", value)
        
        # Test anomalous value
        metrics = {"synthesis_duration_ms": 3000}  # Significantly higher than baseline
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        # Should generate both threshold and anomaly alerts
        self.assertGreaterEqual(len(alerts), 1)
        
        # Check that at least one alert mentions anomaly
        alert_messages = [alert.message for alert in alerts]
        has_anomaly_alert = any("anomalous" in msg.lower() for msg in alert_messages)
        self.assertTrue(has_anomaly_alert)

    def test_disabled_threshold(self):
        """Test that disabled thresholds don't generate alerts."""
        # Disable a threshold
        threshold_key = "pyttsx3.synthesis_duration_ms"
        threshold = self.alert_manager.tts_thresholds[threshold_key]
        threshold.enabled = False
        
        # Try to trigger alert with high value
        metrics = {"synthesis_duration_ms": 10000}
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", metrics)
        
        # Should not generate threshold-based alert (only potential anomaly alert)
        threshold_alerts = [a for a in alerts if a.alert_type == TTSAlertType.HIGH_LATENCY]
        self.assertEqual(len(threshold_alerts), 0)

    def test_alert_export(self):
        """Test alert data export functionality."""
        # Generate some alerts
        self.alert_manager.check_tts_metrics("pyttsx3", {"synthesis_duration_ms": 7000})
        self.alert_manager.check_tts_metrics("coqui", {"success_rate": 0.85})
        
        export_data = self.alert_manager.export_alert_data()
        
        self.assertIn("timestamp", export_data)
        self.assertIn("active_alerts", export_data)
        self.assertIn("recent_alerts", export_data)
        self.assertIn("thresholds", export_data)
        self.assertIn("health_checks", export_data)
        self.assertIn("performance_baselines", export_data)
        
        # Check that alerts are properly serialized
        self.assertGreater(len(export_data["active_alerts"]), 0)
        self.assertGreater(len(export_data["thresholds"]), 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for TTS alert management."""

    def setUp(self):
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_log_file.close()
        
        self.obs_manager = ObservabilityManager("tts_service", {
            "log_file": self.temp_log_file.name
        })
        self.alert_manager = TTSAlertManager(self.obs_manager)

    def tearDown(self):
        self.obs_manager.shutdown()
        os.unlink(self.temp_log_file.name)

    def test_cascading_alert_scenario(self):
        """Test scenario where multiple related alerts are triggered."""
        # Simulate degrading TTS service
        degraded_metrics = {
            "synthesis_duration_ms": 8000,  # High latency
            "success_rate": 0.82,  # Low success rate
            "queue_size": 30,  # Queue buildup
        }
        
        alerts = self.alert_manager.check_tts_metrics("pyttsx3", degraded_metrics)
        
        # Should trigger multiple related alerts
        self.assertEqual(len(alerts), 3)
        
        alert_types = [alert.alert_type for alert in alerts]
        self.assertIn(TTSAlertType.HIGH_LATENCY, alert_types)
        self.assertIn(TTSAlertType.LOW_SUCCESS_RATE, alert_types)
        self.assertIn(TTSAlertType.QUEUE_BUILDUP, alert_types)
        
        # All should be critical or warning severity
        severities = [alert.severity for alert in alerts]
        critical_or_warning = all(
            severity in [TTSAlertSeverity.CRITICAL, TTSAlertSeverity.WARNING]
            for severity in severities
        )
        self.assertTrue(critical_or_warning)

    def test_alert_storm_prevention(self):
        """Test that alert cooldown prevents alert storms."""
        problematic_metrics = {"synthesis_duration_ms": 8000}
        
        # Generate initial alerts
        alerts1 = self.alert_manager.check_tts_metrics("pyttsx3", problematic_metrics)
        self.assertGreater(len(alerts1), 0)
        
        # Immediately check again - should be suppressed
        alerts2 = self.alert_manager.check_tts_metrics("pyttsx3", problematic_metrics)
        self.assertEqual(len(alerts2), 0)
        
        # Check with different engine - should still alert
        alerts3 = self.alert_manager.check_tts_metrics("coqui", problematic_metrics)
        self.assertGreater(len(alerts3), 0)

    def test_multi_engine_monitoring(self):
        """Test monitoring multiple TTS engines simultaneously."""
        # Configure different performance characteristics for different engines
        pyttsx3_metrics = {"synthesis_duration_ms": 1500, "success_rate": 0.98}  # Good
        coqui_metrics = {"synthesis_duration_ms": 8000, "success_rate": 0.85}  # Poor
        hybrid_metrics = {"synthesis_duration_ms": 2500, "success_rate": 0.95}  # Moderate
        
        pyttsx3_alerts = self.alert_manager.check_tts_metrics("pyttsx3", pyttsx3_metrics)
        coqui_alerts = self.alert_manager.check_tts_metrics("coqui", coqui_metrics)
        hybrid_alerts = self.alert_manager.check_tts_metrics("hybrid", hybrid_metrics)
        
        # pyttsx3 should have no alerts (good performance)
        self.assertEqual(len(pyttsx3_alerts), 0)
        
        # coqui should have multiple alerts (poor performance)
        self.assertGreater(len(coqui_alerts), 0)
        
        # hybrid might have some alerts (moderate performance)
        # (depends on specific thresholds)
        
        # Check alert summary reflects multi-engine monitoring
        summary = self.alert_manager.get_alert_summary()
        engines_with_alerts = summary["alerts_by_engine"]
        
        self.assertNotIn("pyttsx3", engines_with_alerts)
        self.assertIn("coqui", engines_with_alerts)

    def test_health_check_failure_recovery_cycle(self):
        """Test complete health check failure and recovery cycle."""
        failure_simulation_counter = 0
        
        def simulated_health_check():
            nonlocal failure_simulation_counter
            failure_simulation_counter += 1
            # Fail for checks 3-5, then recover
            return not (3 <= failure_simulation_counter <= 5)
        
        health_check = TTSHealthCheck(
            name="simulation_check",
            engine_type="test_engine",
            check_function=simulated_health_check,
            failure_threshold=2,
            recovery_threshold=2,
            interval_seconds=0
        )
        
        self.alert_manager.add_health_check(health_check)
        
        # Run through the simulation
        all_alerts = []
        for i in range(8):
            alerts = self.alert_manager.run_health_checks()
            all_alerts.extend(alerts)
        
        # Should have both failure and recovery alerts
        alert_types = [alert.alert_type for alert in all_alerts]
        self.assertIn(TTSAlertType.ENGINE_UNAVAILABLE, alert_types)
        self.assertIn(TTSAlertType.ENGINE_FAILURE, alert_types)  # Recovery alert
        
        # Should have both warning/critical and info severity levels
        severities = [alert.severity for alert in all_alerts]
        self.assertTrue(
            any(sev in [TTSAlertSeverity.WARNING, TTSAlertSeverity.CRITICAL] for sev in severities)
        )
        self.assertIn(TTSAlertSeverity.INFO, severities)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)