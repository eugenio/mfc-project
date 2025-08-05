#!/usr/bin/env python3
"""
Tests for TTS Observability Manager
==================================

Comprehensive test suite for the TTS observability system including
metrics collection, distributed tracing, centralized logging, and monitoring.

Created: 2025-08-03
Author: Agent Eta - Monitoring and Observability Specialist
"""

import unittest
import tempfile
import time
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from src.monitoring.observability_manager import (
    ObservabilityManager,
    HealthStatus,
    AlertSeverity,
    ServiceHealth,
    AlertCondition,
    Alert,
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
            service="tts_service"
        )
        
        # Record multiple counter increments
        for _ in range(5):
            self.collector.record_metric(metric)
        
        # Check counter value
        value = self.collector.get_counter_value("tts_service", "requests_total")
        self.assertEqual(value, 5)

    def test_gauge_metrics(self):
        """Test gauge metric recording and retrieval."""
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            metric_type=MetricType.GAUGE,
            service="tts_service"
        )
        
        self.collector.record_metric(metric)
        
        # Update gauge value
        metric.value = 82.3
        self.collector.record_metric(metric)
        
        # Check latest gauge value
        value = self.collector.get_gauge_value("tts_service", "cpu_usage")
        self.assertEqual(value, 82.3)

    def test_histogram_metrics(self):
        """Test histogram metric recording and statistics."""
        values = [100, 200, 150, 300, 250]
        
        for val in values:
            metric = Metric(
                name="response_time",
                value=val,
                metric_type=MetricType.HISTOGRAM,
                service="tts_service"
            )
            self.collector.record_metric(metric)
        
        # Get histogram statistics
        stats = self.collector.get_histogram_stats("tts_service", "response_time")
        
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
                service="tts_service"
            )
            self.collector.record_metric(metric)
        
        # Get timer statistics (same as histogram)
        stats = self.collector.get_timer_stats("tts_service", "processing_time")
        
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


class TestDistributedTracer(unittest.TestCase):
    """Test cases for DistributedTracer."""

    def setUp(self):
        self.tracer = DistributedTracer("test_service")

    def test_trace_creation(self):
        """Test trace and span creation."""
        span = self.tracer.start_trace("test_operation")
        
        self.assertIsNotNone(span.trace_id)
        self.assertIsNotNone(span.span_id)
        self.assertEqual(span.service_name, "test_service")
        self.assertEqual(span.operation_name, "test_operation")
        self.assertIsNone(span.parent_span_id)

    def test_child_span_creation(self):
        """Test child span creation."""
        parent_span = self.tracer.start_trace("parent_operation")
        child_span = self.tracer.start_child_span(parent_span, "child_operation")
        
        self.assertEqual(child_span.trace_id, parent_span.trace_id)
        self.assertEqual(child_span.parent_span_id, parent_span.span_id)
        self.assertNotEqual(child_span.span_id, parent_span.span_id)

    def test_span_finishing(self):
        """Test span finishing and duration calculation."""
        span = self.tracer.start_trace("test_operation")
        
        # Add some processing time
        time.sleep(0.01)
        
        self.tracer.finish_span(span)
        
        self.assertIsNotNone(span.end_time)
        self.assertIsNotNone(span.duration_ms)
        self.assertGreater(span.duration_ms, 0)
        self.assertEqual(span.status, "ok")

    def test_span_error_handling(self):
        """Test span error handling."""
        span = self.tracer.start_trace("test_operation")
        error_msg = "Test error occurred"
        
        self.tracer.finish_span(span, error=error_msg)
        
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error, error_msg)

    def test_span_tags_and_logs(self):
        """Test span tagging and logging."""
        span = self.tracer.start_trace("test_operation")
        
        span.set_tag("engine", "pyttsx3")
        span.set_tag("text_length", 100)
        span.add_log("Processing started", level="info")
        span.add_log("Processing completed", level="info", result="success")
        
        self.assertEqual(span.tags["engine"], "pyttsx3")
        self.assertEqual(span.tags["text_length"], 100)
        self.assertEqual(len(span.logs), 2)
        self.assertEqual(span.logs[0]["message"], "Processing started")

    def test_trace_retrieval(self):
        """Test trace retrieval by ID."""
        span1 = self.tracer.start_trace("operation1")
        span2 = self.tracer.start_child_span(span1, "operation2")
        
        retrieved_spans = self.tracer.get_trace(span1.trace_id)
        
        self.assertEqual(len(retrieved_spans), 2)
        span_ids = [span.span_id for span in retrieved_spans]
        self.assertIn(span1.span_id, span_ids)
        self.assertIn(span2.span_id, span_ids)

    def test_active_traces(self):
        """Test active traces retrieval."""
        span1 = self.tracer.start_trace("active_operation")
        span2 = self.tracer.start_trace("completed_operation")
        
        # Finish one span
        self.tracer.finish_span(span2)
        
        active_traces = self.tracer.get_active_traces()
        
        # Should have one active trace
        self.assertEqual(len(active_traces), 1)
        self.assertIn(span1.trace_id, active_traces)
        self.assertNotIn(span2.trace_id, active_traces)


class TestCentralizedLogger(unittest.TestCase):
    """Test cases for CentralizedLogger."""

    def setUp(self):
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_log_file.close()
        self.logger = CentralizedLogger("test_service", Path(self.temp_log_file.name))

    def tearDown(self):
        os.unlink(self.temp_log_file.name)

    def test_log_entry_creation(self):
        """Test log entry creation and storage."""
        self.logger.info("Test info message", component="test_component", trace_id="trace123")
        
        recent_logs = self.logger.get_recent_logs(hours=1)
        
        self.assertEqual(len(recent_logs), 1)
        log_entry = recent_logs[0]
        
        self.assertEqual(log_entry.level, "info")
        self.assertEqual(log_entry.message, "Test info message")
        self.assertEqual(log_entry.service, "test_service")
        self.assertEqual(log_entry.component, "test_component")
        self.assertEqual(log_entry.trace_id, "trace123")

    def test_log_levels(self):
        """Test different log levels."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message", error="Exception occurred")
        
        recent_logs = self.logger.get_recent_logs(hours=1)
        
        self.assertEqual(len(recent_logs), 4)
        
        # Check log levels
        levels = [log.level for log in recent_logs]
        self.assertIn("debug", levels)
        self.assertIn("info", levels)
        self.assertIn("warning", levels)
        self.assertIn("error", levels)

    def test_log_filtering(self):
        """Test log filtering by level."""
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        
        error_logs = self.logger.get_recent_logs(hours=1, level="error")
        warning_logs = self.logger.get_recent_logs(hours=1, level="warning")
        
        self.assertEqual(len(error_logs), 1)
        self.assertEqual(len(warning_logs), 1)
        self.assertEqual(error_logs[0].level, "error")
        self.assertEqual(warning_logs[0].level, "warning")

    def test_log_formatting(self):
        """Test log message formatting."""
        test_tags = {"user": "test_user", "action": "synthesis"}
        formatted_msg = self.logger._format_log_entry(LogEntry(
            message="Test message",
            trace_id="trace123",
            component="tts",
            tags=test_tags,
            error="Test error"
        ))
        
        self.assertIn("Test message", formatted_msg)
        self.assertIn("trace_id=trace123", formatted_msg)
        self.assertIn("component=tts", formatted_msg)
        self.assertIn("error=Test error", formatted_msg)
        self.assertIn("user=test_user", formatted_msg)


class TestTTSServiceMonitor(unittest.TestCase):
    """Test cases for TTSServiceMonitor."""

    def setUp(self):
        self.monitor = TTSServiceMonitor()

    def test_tts_request_recording(self):
        """Test TTS request metrics recording."""
        self.monitor.record_tts_request("pyttsx3", 100, 1500, True)
        self.monitor.record_tts_request("pyttsx3", 80, 1200, True)
        self.monitor.record_tts_request("pyttsx3", 120, 2000, False, "Synthesis failed")
        
        summary = self.monitor.get_performance_summary()
        
        self.assertIn("pyttsx3", summary)
        pyttsx3_metrics = summary["pyttsx3"]
        
        self.assertEqual(pyttsx3_metrics["total_requests"], 3)
        self.assertAlmostEqual(pyttsx3_metrics["success_rate"], 2/3, places=2)
        self.assertAlmostEqual(pyttsx3_metrics["avg_duration_ms"], 1566.67, places=1)
        self.assertAlmostEqual(pyttsx3_metrics["avg_text_length"], 100, places=1)

    def test_service_health_updates(self):
        """Test service health status updates."""
        self.monitor.update_service_health("pyttsx3", "healthy", cpu_usage=25.5, memory_mb=128)
        self.monitor.update_service_health("coqui", "degraded", cpu_usage=75.0, memory_mb=512)
        
        health_summary = self.monitor.get_service_health_summary()
        
        self.assertIn("pyttsx3", health_summary)
        self.assertIn("coqui", health_summary)
        
        self.assertEqual(health_summary["pyttsx3"]["status"], "healthy")
        self.assertEqual(health_summary["coqui"]["status"], "degraded")
        self.assertEqual(health_summary["pyttsx3"]["cpu_usage"], 25.5)

    def test_performance_calculation(self):
        """Test performance metrics calculation."""
        # Record requests over time
        base_time = time.time()
        
        with patch('time.time', return_value=base_time):
            self.monitor.record_tts_request("test_engine", 100, 1000, True)
        
        with patch('time.time', return_value=base_time + 30):  # 30 seconds later
            self.monitor.record_tts_request("test_engine", 100, 1000, True)
        
        summary = self.monitor.get_performance_summary()
        test_metrics = summary["test_engine"]
        
        # Should calculate requests per minute
        self.assertGreater(test_metrics["requests_per_minute"], 0)

    def test_multiple_engines(self):
        """Test monitoring multiple TTS engines."""
        # Record requests for different engines
        self.monitor.record_tts_request("pyttsx3", 100, 1000, True)
        self.monitor.record_tts_request("coqui", 100, 3000, True)
        self.monitor.record_tts_request("hybrid", 100, 1500, True)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertEqual(len(summary), 3)
        self.assertIn("pyttsx3", summary)
        self.assertIn("coqui", summary)
        self.assertIn("hybrid", summary)
        
        # Check that different engines have different metrics
        self.assertEqual(summary["pyttsx3"]["avg_duration_ms"], 1000)
        self.assertEqual(summary["coqui"]["avg_duration_ms"], 3000)
        self.assertEqual(summary["hybrid"]["avg_duration_ms"], 1500)


class TestObservabilityManager(unittest.TestCase):
    """Test cases for ObservabilityManager."""

    def setUp(self):
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_log_file.close()
        
        config = {
            "log_file": self.temp_log_file.name,
            "metrics_retention_hours": 1,
            "trace_retention_hours": 1
        }
        
        self.manager = ObservabilityManager("test_service", config)

    def tearDown(self):
        self.manager.shutdown()
        os.unlink(self.temp_log_file.name)

    def test_metric_recording(self):
        """Test metric recording through manager."""
        self.manager.record_metric("test_counter", 5, MetricType.COUNTER, tag1="value1")
        self.manager.record_metric("test_gauge", 42.5, MetricType.GAUGE, tag2="value2")
        
        summary = self.manager.metrics_collector.get_metrics_summary()
        
        self.assertEqual(summary["counters"]["test_service.test_counter"], 5)
        self.assertEqual(summary["gauges"]["test_service.test_gauge"], 42.5)

    def test_trace_operations(self):
        """Test trace operations through manager."""
        span = self.manager.start_trace("test_operation")
        span.set_tag("operation_type", "synthesis")
        
        child_span = self.manager.start_child_span(span, "child_operation")
        child_span.add_log("Child operation started")
        
        self.manager.finish_span(child_span)
        self.manager.finish_span(span)
        
        trace = self.manager.tracer.get_trace(span.trace_id)
        
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0].tags["operation_type"], "synthesis")

    def test_logging_operations(self):
        """Test logging operations through manager."""
        self.manager.log_info("Info message", component="test")
        self.manager.log_warning("Warning message", component="test")
        self.manager.log_error("Error message", component="test", error="Test error")
        
        recent_logs = self.manager.logger.get_recent_logs(hours=1)
        
        self.assertEqual(len(recent_logs), 3)
        
        levels = [log.level for log in recent_logs]
        self.assertIn("info", levels)
        self.assertIn("warning", levels)
        self.assertIn("error", levels)

    def test_tts_monitoring(self):
        """Test TTS-specific monitoring."""
        self.manager.monitor_tts_request("pyttsx3", 100, 1500, True)
        self.manager.monitor_tts_request("pyttsx3", 80, 1200, False, "Synthesis failed")
        
        # Check that metrics were recorded
        counter_value = self.manager.metrics_collector.get_counter_value("test_service", "tts_requests_total")
        error_count = self.manager.metrics_collector.get_counter_value("test_service", "tts_errors_total")
        
        self.assertEqual(counter_value, 2)
        self.assertEqual(error_count, 1)
        
        # Check TTS monitor
        summary = self.manager.tts_monitor.get_performance_summary()
        self.assertIn("pyttsx3", summary)

    def test_service_health_monitoring(self):
        """Test service health monitoring."""
        self.manager.update_service_health("pyttsx3", "healthy", version="1.0")
        self.manager.update_service_health("coqui", "unhealthy", error="Import failed")
        
        health_summary = self.manager.tts_monitor.get_service_health_summary()
        
        self.assertEqual(health_summary["pyttsx3"]["status"], "healthy")
        self.assertEqual(health_summary["coqui"]["status"], "unhealthy")

    def test_monitoring_summary(self):
        """Test comprehensive monitoring summary."""
        # Add some data
        self.manager.record_metric("test_metric", 100, MetricType.COUNTER)
        self.manager.monitor_tts_request("pyttsx3", 100, 1500, True)
        self.manager.log_info("Test log message")
        
        span = self.manager.start_trace("test_trace")
        
        summary = self.manager.get_monitoring_summary()
        
        self.assertIn("service", summary)
        self.assertIn("metrics", summary)
        self.assertIn("tts_performance", summary)
        self.assertIn("service_health", summary)
        self.assertIn("active_traces", summary)
        self.assertIn("recent_logs", summary)
        
        self.assertEqual(summary["service"], "test_service")
        self.assertGreater(summary["active_traces"], 0)

    def test_alert_integration(self):
        """Test alert callback integration."""
        alerts_received = []
        
        def alert_callback(alert_data):
            alerts_received.append(alert_data)
        
        self.manager.register_alert_callback(alert_callback)
        
        # This would typically trigger alerts in a real scenario
        self.manager.check_alerts()
        
        # Alert callback should be registered
        self.assertEqual(len(self.manager.alert_callbacks), 1)

    def test_trace_export(self):
        """Test trace data export."""
        span = self.manager.start_trace("export_test")
        span.set_tag("test", "export")
        self.manager.finish_span(span)
        
        export_data = self.manager.export_traces([span.trace_id])
        
        self.assertIn("traces", export_data)
        self.assertIn("service", export_data)
        self.assertIn(span.trace_id, export_data["traces"])
        
        trace_data = export_data["traces"][span.trace_id]
        self.assertEqual(len(trace_data), 1)
        self.assertEqual(trace_data[0]["operation_name"], "export_test")


class TestTraceOperationContextManager(unittest.TestCase):
    """Test cases for trace_operation context manager."""

    def setUp(self):
        self.manager = ObservabilityManager("test_service")

    def tearDown(self):
        self.manager.shutdown()

    def test_trace_context_manager_success(self):
        """Test trace context manager with successful operation."""
        with trace_operation(self.manager, "test_operation") as span:
            span.set_tag("test_tag", "test_value")
            span.add_log("Operation in progress")
            time.sleep(0.01)  # Simulate work
        
        # Check that span was properly finished
        self.assertIsNotNone(span.end_time)
        self.assertIsNotNone(span.duration_ms)
        self.assertEqual(span.status, "ok")
        self.assertIsNone(span.error)

    def test_trace_context_manager_with_error(self):
        """Test trace context manager with exception."""
        with self.assertRaises(ValueError):
            with trace_operation(self.manager, "error_operation") as span:
                span.set_tag("will_fail", True)
                raise ValueError("Test error")
        
        # Check that span was properly finished with error
        self.assertIsNotNone(span.end_time)
        self.assertEqual(span.status, "error")
        self.assertEqual(span.error, "Test error")

    def test_trace_context_manager_with_parent(self):
        """Test trace context manager with parent span."""
        parent_span = self.manager.start_trace("parent_operation")
        
        with trace_operation(self.manager, "child_operation", parent_span) as child_span:
            child_span.set_tag("is_child", True)
        
        self.assertEqual(child_span.trace_id, parent_span.trace_id)
        self.assertEqual(child_span.parent_span_id, parent_span.span_id)
        
        self.manager.finish_span(parent_span)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios for the complete observability system."""

    def setUp(self):
        self.temp_log_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_log_file.close()
        
        config = {
            "log_file": self.temp_log_file.name,
            "metrics_retention_hours": 1,
            "trace_retention_hours": 1
        }
        
        self.manager = ObservabilityManager("tts_service", config)

    def tearDown(self):
        self.manager.shutdown()
        os.unlink(self.temp_log_file.name)

    def test_complete_tts_request_monitoring(self):
        """Test complete TTS request monitoring scenario."""
        # Start trace for TTS request
        with trace_operation(self.manager, "tts_synthesis_request") as main_span:
            main_span.set_tag("engine", "pyttsx3")
            main_span.set_tag("text_length", 150)
            
            # Log request start
            self.manager.log_info(
                "TTS synthesis request started",
                component="tts_handler",
                trace_id=main_span.trace_id,
                span_id=main_span.span_id,
                engine="pyttsx3",
                text_length=150
            )
            
            # Simulate preprocessing
            with trace_operation(self.manager, "text_preprocessing", main_span) as prep_span:
                prep_span.set_tag("preprocessing_type", "cleanup")
                time.sleep(0.005)  # Simulate preprocessing time
                
                self.manager.log_debug(
                    "Text preprocessing completed",
                    component="text_processor",
                    trace_id=main_span.trace_id,
                    span_id=prep_span.span_id
                )
            
            # Simulate TTS synthesis
            with trace_operation(self.manager, "audio_synthesis", main_span) as synth_span:
                synth_span.set_tag("synthesis_engine", "pyttsx3")
                time.sleep(0.01)  # Simulate synthesis time
                
                self.manager.log_info(
                    "Audio synthesis completed",
                    component="synthesis_engine",
                    trace_id=main_span.trace_id,
                    span_id=synth_span.span_id,
                    audio_duration_ms=3000
                )
            
            # Record metrics
            synthesis_duration = main_span.duration_ms or 15  # Approximate duration
            self.manager.monitor_tts_request("pyttsx3", 150, synthesis_duration, True)
            
            # Update service health
            self.manager.update_service_health("pyttsx3", "healthy", last_request_time=time.time())
            
            # Log completion
            self.manager.log_info(
                "TTS synthesis request completed successfully",
                component="tts_handler",
                trace_id=main_span.trace_id,
                span_id=main_span.span_id,
                total_duration_ms=synthesis_duration
            )
        
        # Verify all components worked together
        # Check traces
        trace = self.manager.tracer.get_trace(main_span.trace_id)
        self.assertEqual(len(trace), 3)  # main + 2 child spans
        
        # Check logs
        recent_logs = self.manager.logger.get_recent_logs(hours=1)
        trace_logs = [log for log in recent_logs if log.trace_id == main_span.trace_id]
        self.assertGreaterEqual(len(trace_logs), 3)
        
        # Check metrics
        counter_value = self.manager.metrics_collector.get_counter_value("tts_service", "tts_requests_total")
        self.assertEqual(counter_value, 1)
        
        # Check TTS monitoring
        performance_summary = self.manager.tts_monitor.get_performance_summary()
        self.assertIn("pyttsx3", performance_summary)
        self.assertEqual(performance_summary["pyttsx3"]["total_requests"], 1)
        
        # Check service health
        health_summary = self.manager.tts_monitor.get_service_health_summary()
        self.assertEqual(health_summary["pyttsx3"]["status"], "healthy")

    def test_error_handling_scenario(self):
        """Test error handling and monitoring scenario."""
        try:
            with trace_operation(self.manager, "failing_tts_request") as span:
                span.set_tag("engine", "coqui")
                span.set_tag("expected_to_fail", True)
                
                # Log error occurrence
                self.manager.log_error(
                    "TTS engine initialization failed",
                    component="coqui_engine",
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                    error="ImportError: Coqui TTS not installed"
                )
                
                # Record failed request
                self.manager.monitor_tts_request("coqui", 100, 0, False, "Engine initialization failed")
                
                # Update service health
                self.manager.update_service_health("coqui", "unhealthy", error="Engine not available")
                
                # Raise exception to test error handling
                raise RuntimeError("Coqui TTS engine initialization failed")
        
        except RuntimeError:
            pass  # Expected error
        
        # Verify error was properly recorded
        self.assertEqual(span.status, "error")
        self.assertIn("Coqui TTS", span.error)
        
        # Check error metrics
        error_count = self.manager.metrics_collector.get_counter_value("tts_service", "tts_errors_total")
        self.assertEqual(error_count, 1)
        
        # Check service health reflects the error
        health_summary = self.manager.tts_monitor.get_service_health_summary()
        self.assertEqual(health_summary["coqui"]["status"], "unhealthy")

    def test_concurrent_operations_monitoring(self):
        """Test monitoring of concurrent TTS operations."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def tts_request_worker(engine_name, request_id):
            try:
                with trace_operation(self.manager, f"concurrent_tts_{engine_name}") as span:
                    span.set_tag("engine", engine_name)
                    span.set_tag("request_id", request_id)
                    
                    # Simulate varying processing times
                    processing_time = 0.01 if engine_name == "pyttsx3" else 0.02
                    time.sleep(processing_time)
                    
                    duration_ms = processing_time * 1000
                    self.manager.monitor_tts_request(engine_name, 100, duration_ms, True)
                    
                    results.put((engine_name, request_id, "success"))
            
            except Exception as e:
                results.put((engine_name, request_id, f"error: {e}"))
        
        # Start concurrent requests
        threads = []
        for i in range(5):
            engine = "pyttsx3" if i % 2 == 0 else "coqui"
            thread = threading.Thread(target=tts_request_worker, args=(engine, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=1.0)
        
        # Collect results
        request_results = []
        while not results.empty():
            request_results.append(results.get())
        
        # Verify all requests completed
        self.assertEqual(len(request_results), 5)
        
        # Check that metrics were recorded for concurrent requests
        total_requests = self.manager.metrics_collector.get_counter_value("tts_service", "tts_requests_total")
        self.assertEqual(total_requests, 5)
        
        # Check performance summary includes both engines
        performance_summary = self.manager.tts_monitor.get_performance_summary()
        self.assertIn("pyttsx3", performance_summary)
        self.assertIn("coqui", performance_summary)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)