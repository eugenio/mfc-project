#!/usr/bin/env python3
"""
Observability Manager for Distributed TTS System
==============================================

This module provides comprehensive monitoring and observability for the distributed TTS system,
including metrics collection, distributed tracing, centralized logging, and alerting.

Features:
- Service health monitoring for TTS engines
- Request tracing across main project and TTS service
- Performance metrics collection and aggregation
- Centralized logging with correlation IDs
- Alert management for TTS service issues
- Real-time monitoring dashboards

Created: 2025-08-03
Author: Agent Eta - Monitoring and Observability Specialist
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class MetricType(Enum):
    """Types of metrics collected by the monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class ServiceType(Enum):
    """Types of services in the distributed system."""
    MAIN_PROJECT = "main_project"
    TTS_SERVICE = "tts_service"
    PYTTSX3_ENGINE = "pyttsx3_engine"
    COQUI_ENGINE = "coqui_engine"
    HYBRID_ENGINE = "hybrid_engine"


@dataclass
class TraceSpan:
    """Represents a span in distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    service_name: str = ""
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None

    def finish(self, error: str | None = None):
        """Finish the span and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if error:
            self.status = "error"
            self.error = error

    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value


@dataclass
class Metric:
    """Represents a metric data point."""
    name: str
    value: int | float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    service: str = ""


@dataclass
class LogEntry:
    """Represents a structured log entry."""
    timestamp: float = field(default_factory=time.time)
    level: str = "info"
    message: str = ""
    service: str = ""
    trace_id: str | None = None
    span_id: str | None = None
    component: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class MetricsCollector:
    """Collects and aggregates metrics from distributed services."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)
        self.timers: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def record_metric(self, metric: Metric):
        """Record a metric data point."""
        with self._lock:
            metric_key = f"{metric.service}.{metric.name}"

            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric_key] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric_key] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histograms[metric_key].append(metric.value)
            elif metric.metric_type == MetricType.TIMER:
                self.timers[metric_key].append(metric.value)

            # Store raw metric for time series
            self.metrics[metric_key].append(metric)

            # Clean old metrics
            self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        for _metric_name, metric_queue in self.metrics.items():
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()

    def get_counter_value(self, service: str, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(f"{service}.{name}", 0)

    def get_gauge_value(self, service: str, name: str) -> float | None:
        """Get current gauge value."""
        with self._lock:
            return self.gauges.get(f"{service}.{name}")

    def get_histogram_stats(self, service: str, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self.histograms.get(f"{service}.{name}", [])
            if not values:
                return {}

            sorted_values = sorted(values)
            count = len(sorted_values)

            return {
                "count": count,
                "min": min(sorted_values),
                "max": max(sorted_values),
                "mean": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)]
            }

    def get_timer_stats(self, service: str, name: str) -> dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(service, name)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timers": {}
            }

            for name in self.histograms:
                service, metric_name = name.split('.', 1)
                summary["histograms"][name] = self.get_histogram_stats(service, metric_name)

            for name in self.timers:
                service, metric_name = name.split('.', 1)
                summary["timers"][name] = self.get_timer_stats(service, metric_name)

            return summary


class DistributedTracer:
    """Manages distributed tracing across TTS services."""

    def __init__(self, service_name: str, retention_hours: int = 24):
        self.service_name = service_name
        self.retention_hours = retention_hours
        self.traces: dict[str, list[TraceSpan]] = defaultdict(list)
        self.active_spans: dict[str, TraceSpan] = {}
        self._lock = threading.RLock()

    def start_trace(self, operation_name: str, trace_id: str | None = None) -> TraceSpan:
        """Start a new trace or add to existing trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span_id = str(uuid.uuid4())
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            service_name=self.service_name,
            operation_name=operation_name
        )

        with self._lock:
            self.traces[trace_id].append(span)
            self.active_spans[span_id] = span

        return span

    def start_child_span(self, parent_span: TraceSpan, operation_name: str) -> TraceSpan:
        """Start a child span."""
        span_id = str(uuid.uuid4())
        child_span = TraceSpan(
            trace_id=parent_span.trace_id,
            span_id=span_id,
            parent_span_id=parent_span.span_id,
            service_name=self.service_name,
            operation_name=operation_name
        )

        with self._lock:
            self.traces[parent_span.trace_id].append(child_span)
            self.active_spans[span_id] = child_span

        return child_span

    def finish_span(self, span: TraceSpan, error: str | None = None):
        """Finish a span."""
        span.finish(error)

        with self._lock:
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return self.traces.get(trace_id, []).copy()

    def get_active_traces(self) -> dict[str, list[TraceSpan]]:
        """Get all active traces."""
        with self._lock:
            return {
                trace_id: spans for trace_id, spans in self.traces.items()
                if any(span.end_time is None for span in spans)
            }

    def cleanup_old_traces(self):
        """Remove old completed traces."""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        with self._lock:
            traces_to_remove = []
            for trace_id, spans in self.traces.items():
                # Remove trace if all spans are completed and old
                if all(span.end_time is not None and span.end_time < cutoff_time for span in spans):
                    traces_to_remove.append(trace_id)

            for trace_id in traces_to_remove:
                del self.traces[trace_id]


class CentralizedLogger:
    """Centralized logging system with correlation IDs."""

    def __init__(self, service_name: str, log_file: Path | None = None):
        self.service_name = service_name
        self.log_file = log_file or Path("/tmp/mfc-monitoring.log")
        self.log_entries: deque = deque(maxlen=10000)
        self._lock = threading.RLock()

        # Setup file logging
        self.logger = logging.getLogger(f"mfc.{service_name}")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: str, message: str, component: str = "",
            trace_id: str | None = None, span_id: str | None = None,
            error: str | None = None, **tags):
        """Log a structured message."""
        log_entry = LogEntry(
            level=level,
            message=message,
            service=self.service_name,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            error=error,
            tags=tags
        )

        with self._lock:
            self.log_entries.append(log_entry)

        # Also log to file
        log_msg = self._format_log_entry(log_entry)
        getattr(self.logger, level.lower(), self.logger.info)(log_msg)

    def _format_log_entry(self, entry: LogEntry) -> str:
        """Format log entry for file output."""
        parts = [entry.message]

        if entry.trace_id:
            parts.append(f"trace_id={entry.trace_id}")
        if entry.span_id:
            parts.append(f"span_id={entry.span_id}")
        if entry.component:
            parts.append(f"component={entry.component}")
        if entry.error:
            parts.append(f"error={entry.error}")
        if entry.tags:
            tag_str = " ".join(f"{k}={v}" for k, v in entry.tags.items())
            parts.append(tag_str)

        return " | ".join(parts)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log("error", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log("debug", message, **kwargs)

    def get_recent_logs(self, hours: int = 1, level: str | None = None) -> list[LogEntry]:
        """Get recent log entries."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            filtered_logs = [
                entry for entry in self.log_entries
                if entry.timestamp > cutoff_time
                and (level is None or entry.level == level)
            ]

        return sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)


class TTSServiceMonitor:
    """Monitors TTS service health and performance."""

    def __init__(self):
        self.service_health: dict[str, dict[str, Any]] = {}
        self.performance_metrics: dict[str, dict[str, float]] = {}
        self._lock = threading.RLock()

    def record_tts_request(self, engine_type: str, text_length: int,
                          duration_ms: float, success: bool, error: str | None = None):
        """Record TTS request metrics."""
        with self._lock:
            if engine_type not in self.performance_metrics:
                self.performance_metrics[engine_type] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_duration_ms": 0,
                    "total_text_length": 0,
                    "min_duration_ms": float('inf'),
                    "max_duration_ms": 0,
                    "last_request_time": time.time()
                }

            metrics = self.performance_metrics[engine_type]
            metrics["total_requests"] += 1
            metrics["total_duration_ms"] += duration_ms
            metrics["total_text_length"] += text_length
            metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)
            metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)
            metrics["last_request_time"] = time.time()

            if success:
                metrics["successful_requests"] += 1
            else:
                metrics["failed_requests"] += 1

    def update_service_health(self, service: str, status: str, **metadata):
        """Update service health status."""
        with self._lock:
            self.service_health[service] = {
                "status": status,
                "timestamp": time.time(),
                **metadata
            }

    def get_service_health_summary(self) -> dict[str, Any]:
        """Get summary of all service health."""
        with self._lock:
            return dict(self.service_health)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get TTS performance summary."""
        with self._lock:
            summary = {}
            for engine, metrics in self.performance_metrics.items():
                if metrics["total_requests"] > 0:
                    success_rate = metrics["successful_requests"] / metrics["total_requests"]
                    avg_duration = metrics["total_duration_ms"] / metrics["total_requests"]
                    avg_text_length = metrics["total_text_length"] / metrics["total_requests"]

                    summary[engine] = {
                        "success_rate": success_rate,
                        "avg_duration_ms": avg_duration,
                        "avg_text_length": avg_text_length,
                        "requests_per_minute": self._calculate_rpm(metrics),
                        "min_duration_ms": metrics["min_duration_ms"],
                        "max_duration_ms": metrics["max_duration_ms"],
                        "total_requests": metrics["total_requests"]
                    }

            return summary

    def _calculate_rpm(self, metrics: dict[str, float]) -> float:
        """Calculate requests per minute."""
        time_window = max(60, time.time() - metrics["last_request_time"])
        return (metrics["total_requests"] / time_window) * 60


class ObservabilityManager:
    """Main observability manager for distributed TTS system."""

    def __init__(self, service_name: str, config: dict[str, Any] | None = None):
        self.service_name = service_name
        self.config = config or {}

        # Initialize components
        self.metrics_collector = MetricsCollector(
            retention_hours=self.config.get("metrics_retention_hours", 24)
        )
        self.tracer = DistributedTracer(
            service_name=service_name,
            retention_hours=self.config.get("trace_retention_hours", 24)
        )
        self.logger = CentralizedLogger(
            service_name=service_name,
            log_file=Path(self.config.get("log_file", "/tmp/mfc-monitoring.log"))
        )
        self.tts_monitor = TTSServiceMonitor()

        # Alert callbacks
        self.alert_callbacks: list[Callable[[dict[str, Any]], None]] = []

        # Background cleanup task
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1)
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background cleanup tasks."""
        def cleanup_task():
            while True:
                try:
                    time.sleep(3600)  # Cleanup every hour
                    self.tracer.cleanup_old_traces()
                    self.metrics_collector._cleanup_old_metrics()
                except Exception as e:
                    self.logger.error(f"Cleanup task error: {e}")

        self.cleanup_executor.submit(cleanup_task)

    def record_metric(self, name: str, value: int | float,
                     metric_type: MetricType, **tags):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            service=self.service_name,
            tags=tags
        )
        self.metrics_collector.record_metric(metric)

    def start_trace(self, operation_name: str, trace_id: str | None = None) -> TraceSpan:
        """Start a new trace."""
        return self.tracer.start_trace(operation_name, trace_id)

    def start_child_span(self, parent_span: TraceSpan, operation_name: str) -> TraceSpan:
        """Start a child span."""
        return self.tracer.start_child_span(parent_span, operation_name)

    def finish_span(self, span: TraceSpan, error: str | None = None):
        """Finish a span."""
        self.tracer.finish_span(span, error)

    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)

    def monitor_tts_request(self, engine_type: str, text_length: int,
                           duration_ms: float, success: bool, error: str | None = None):
        """Monitor TTS request."""
        self.tts_monitor.record_tts_request(engine_type, text_length, duration_ms, success, error)

        # Record metrics
        self.record_metric("tts_requests_total", 1, MetricType.COUNTER, engine=engine_type)
        self.record_metric("tts_duration_ms", duration_ms, MetricType.HISTOGRAM, engine=engine_type)
        self.record_metric("tts_text_length", text_length, MetricType.HISTOGRAM, engine=engine_type)

        if not success:
            self.record_metric("tts_errors_total", 1, MetricType.COUNTER, engine=engine_type)

    def update_service_health(self, service: str, status: str, **metadata):
        """Update service health."""
        self.tts_monitor.update_service_health(service, status, **metadata)
        self.record_metric("service_health", 1 if status == "healthy" else 0,
                          MetricType.GAUGE, service=service)

    def register_alert_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Register alert callback."""
        self.alert_callbacks.append(callback)

    def check_alerts(self):
        """Check for alert conditions."""
        alerts = []

        # Check TTS performance
        tts_summary = self.tts_monitor.get_performance_summary()
        for engine, metrics in tts_summary.items():
            if metrics["success_rate"] < 0.95:
                alerts.append({
                    "type": "tts_low_success_rate",
                    "engine": engine,
                    "success_rate": metrics["success_rate"],
                    "severity": "warning"
                })

            if metrics["avg_duration_ms"] > 5000:  # 5 seconds
                alerts.append({
                    "type": "tts_high_latency",
                    "engine": engine,
                    "avg_duration_ms": metrics["avg_duration_ms"],
                    "severity": "warning"
                })

        # Check service health
        health_summary = self.tts_monitor.get_service_health_summary()
        for service, health in health_summary.items():
            if health["status"] != "healthy":
                alerts.append({
                    "type": "service_unhealthy",
                    "service": service,
                    "status": health["status"],
                    "severity": "critical"
                })

        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "service": self.service_name,
            "timestamp": time.time(),
            "metrics": self.metrics_collector.get_metrics_summary(),
            "tts_performance": self.tts_monitor.get_performance_summary(),
            "service_health": self.tts_monitor.get_service_health_summary(),
            "active_traces": len(self.tracer.get_active_traces()),
            "recent_logs": len(self.logger.get_recent_logs())
        }

    def export_traces(self, trace_ids: list[str] | None = None) -> dict[str, Any]:
        """Export traces for external analysis."""
        if trace_ids is None:
            # Export all traces
            all_traces = {}
            for trace_id in self.tracer.traces:
                all_traces[trace_id] = [asdict(span) for span in self.tracer.get_trace(trace_id)]
        else:
            all_traces = {}
            for trace_id in trace_ids:
                all_traces[trace_id] = [asdict(span) for span in self.tracer.get_trace(trace_id)]

        return {
            "service": self.service_name,
            "export_timestamp": time.time(),
            "traces": all_traces
        }

    def shutdown(self):
        """Shutdown observability manager."""
        self.cleanup_executor.shutdown(wait=True)


# Context manager for tracing
class trace_operation:
    """Context manager for automatic span creation and finishing."""

    def __init__(self, observability_manager: ObservabilityManager,
                 operation_name: str, parent_span: TraceSpan | None = None):
        self.manager = observability_manager
        self.operation_name = operation_name
        self.parent_span = parent_span
        self.span: TraceSpan | None = None

    def __enter__(self) -> TraceSpan:
        if self.parent_span:
            self.span = self.manager.start_child_span(self.parent_span, self.operation_name)
        else:
            self.span = self.manager.start_trace(self.operation_name)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            error = str(exc_val) if exc_val else None
            self.manager.finish_span(self.span, error)


# Example usage and testing
if __name__ == "__main__":
    # Create observability manager
    config = {
        "metrics_retention_hours": 24,
        "trace_retention_hours": 24,
        "log_file": "/tmp/tts-monitoring.log"
    }

    obs_manager = ObservabilityManager("tts_service", config)

    # Example TTS request monitoring
    with trace_operation(obs_manager, "tts_synthesis") as span:
        span.set_tag("engine", "pyttsx3")
        span.set_tag("text_length", 50)

        # Simulate TTS processing
        time.sleep(0.1)

        obs_manager.monitor_tts_request(
            engine_type="pyttsx3",
            text_length=50,
            duration_ms=100,
            success=True
        )

        span.add_log("TTS synthesis completed successfully")

    # Update service health
    obs_manager.update_service_health("pyttsx3_engine", "healthy", cpu_usage=25.5)

    # Get monitoring summary
    summary = obs_manager.get_monitoring_summary()
    print("Monitoring Summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Check for alerts
    obs_manager.check_alerts()

    # Cleanup
    obs_manager.shutdown()
