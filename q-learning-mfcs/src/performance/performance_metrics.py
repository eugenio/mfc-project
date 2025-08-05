"""
Performance Metrics Collection System
====================================

Comprehensive performance monitoring for MFC systems including:
- System resource metrics (CPU, memory, disk)
- Application-specific metrics (simulation performance, Q-learning convergence)
- Network and I/O metrics
- Custom business metrics

Created: 2025-08-05
Author: TDD Agent 44
"""
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import psutil

# Setup logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric measurement"""
    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str] = None
    metric_type: MetricType = MetricType.GAUGE

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class MetricSummary:
    """Summary statistics for a metric over time"""
    name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    percentile_50: float
    percentile_95: float
    percentile_99: float
    last_updated: datetime


class MetricsCollector:
    """Collects and aggregates system and application metrics"""

    def __init__(self, collection_interval: float = 1.0, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval)))
        self.metric_callbacks: dict[str, Callable] = {}
        self.custom_collectors: list[Callable] = []
        self.running = False
        self._collection_thread = None
        self._lock = threading.Lock()

        # Register default system metrics
        self._register_default_collectors()

    def _register_default_collectors(self):
        """Register default system metric collectors"""
        self.register_custom_collector(self._collect_system_metrics)
        self.register_custom_collector(self._collect_process_metrics)

    def register_metric_callback(self, metric_name: str, callback: Callable[[], float]):
        """Register a callback function to collect a specific metric"""
        self.metric_callbacks[metric_name] = callback

    def register_custom_collector(self, collector: Callable):
        """Register a custom collector function that returns Dict[str, float]"""
        self.custom_collectors.append(collector)

    def start_collection(self):
        """Start automatic metric collection"""
        if self.running:
            return

        self.running = True
        self._collection_thread = threading.Thread(target=self._collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop automatic metric collection"""
        self.running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

    def _collect_all_metrics(self):
        """Collect all registered metrics"""
        timestamp = datetime.now()

        # Collect callback metrics
        for metric_name, callback in self.metric_callbacks.items():
            try:
                value = callback()
                if value is not None:
                    self.record_metric(metric_name, value, timestamp)
            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")

        # Collect custom collector metrics
        for collector in self.custom_collectors:
            try:
                metrics = collector()
                if isinstance(metrics, dict):
                    for name, value in metrics.items():
                        if value is not None:
                            self.record_metric(name, value, timestamp)
            except Exception as e:
                logger.error(f"Error in custom collector: {e}")

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect system-level metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'system.cpu.utilization': cpu_percent,
                'system.memory.utilization': memory.percent,
                'system.memory.available_gb': memory.available / (1024**3),
                'system.disk.utilization': disk.percent,
                'system.disk.free_gb': disk.free / (1024**3),
                'system.load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _collect_process_metrics(self) -> dict[str, float]:
        """Collect current process metrics"""
        try:
            process = psutil.Process()

            return {
                'process.cpu.utilization': process.cpu_percent(),
                'process.memory.rss_mb': process.memory_info().rss / (1024**2),
                'process.memory.vms_mb': process.memory_info().vms / (1024**2),
                'process.threads.count': process.num_threads(),
                'process.file_descriptors.count': process.num_fds() if hasattr(process, 'num_fds') else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {}

    def record_metric(self, name: str, value: float, timestamp: datetime | None = None,
                     labels: dict[str, str] | None = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a single metric value"""
        if timestamp is None:
            timestamp = datetime.now()

        metric = MetricValue(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels or {},
            metric_type=metric_type
        )

        with self._lock:
            self.metrics[name].append(metric)

    def get_metric_values(self, name: str, since: datetime | None = None) -> list[MetricValue]:
        """Get metric values for a specific metric"""
        with self._lock:
            values = list(self.metrics.get(name, []))

        if since:
            values = [v for v in values if v.timestamp >= since]

        return values

    def get_metric_summary(self, name: str, since: datetime | None = None) -> MetricSummary | None:
        """Get summary statistics for a metric"""
        values = self.get_metric_values(name, since)

        if not values:
            return None

        numeric_values = [v.value for v in values]

        return MetricSummary(
            name=name,
            count=len(numeric_values),
            min_value=float(np.min(numeric_values)),
            max_value=float(np.max(numeric_values)),
            mean_value=float(np.mean(numeric_values)),
            std_value=float(np.std(numeric_values)),
            percentile_50=float(np.percentile(numeric_values, 50)),
            percentile_95=float(np.percentile(numeric_values, 95)),
            percentile_99=float(np.percentile(numeric_values, 99)),
            last_updated=values[-1].timestamp
        )

    def get_all_metric_names(self) -> list[str]:
        """Get all metric names currently being collected"""
        with self._lock:
            return list(self.metrics.keys())

    def clear_metrics(self, name: str | None = None):
        """Clear metrics data"""
        with self._lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()

    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for name in self.get_all_metric_names():
            summary = self.get_metric_summary(name)
            if summary:
                # Convert dots to underscores for Prometheus compatibility
                prom_name = name.replace('.', '_')
                lines.append(f"# HELP {prom_name} Metric {name}")
                lines.append(f"# TYPE {prom_name} gauge")
                lines.append(f"{prom_name} {summary.mean_value}")

        return '\n'.join(lines)

    def _export_json_format(self) -> str:
        """Export metrics in JSON format"""
        import json

        data = {}
        for name in self.get_all_metric_names():
            summary = self.get_metric_summary(name)
            if summary:
                data[name] = asdict(summary)

        return json.dumps(data, default=str, indent=2)


class MFCMetricsCollector(MetricsCollector):
    """Specialized metrics collector for MFC systems"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._register_mfc_collectors()

    def _register_mfc_collectors(self):
        """Register MFC-specific metric collectors"""
        self.register_custom_collector(self._collect_mfc_simulation_metrics)
        self.register_custom_collector(self._collect_qlearning_metrics)

    def _collect_mfc_simulation_metrics(self) -> dict[str, float]:
        """Collect MFC simulation performance metrics"""
        # This would integrate with actual MFC simulation components
        # For now, return placeholder metrics
        return {
            'mfc.simulation.iterations_per_second': 0.0,
            'mfc.simulation.power_output_mw': 0.0,
            'mfc.simulation.efficiency_percent': 0.0,
            'mfc.biofilm.thickness_mm': 0.0,
            'mfc.substrate.concentration_mg_l': 0.0,
        }

    def _collect_qlearning_metrics(self) -> dict[str, float]:
        """Collect Q-learning performance metrics"""
        # This would integrate with Q-learning components
        # For now, return placeholder metrics
        return {
            'qlearning.episodes_completed': 0.0,
            'qlearning.epsilon_value': 0.0,
            'qlearning.average_reward': 0.0,
            'qlearning.convergence_score': 0.0,
            'qlearning.qtable_size': 0.0,
        }

    def record_simulation_step(self, power_output: float, efficiency: float,
                              biofilm_thickness: float, substrate_conc: float):
        """Record metrics for a single simulation step"""
        timestamp = datetime.now()

        self.record_metric('mfc.simulation.power_output_mw', power_output, timestamp)
        self.record_metric('mfc.simulation.efficiency_percent', efficiency, timestamp)
        self.record_metric('mfc.biofilm.thickness_mm', biofilm_thickness, timestamp)
        self.record_metric('mfc.substrate.concentration_mg_l', substrate_conc, timestamp)

    def record_qlearning_episode(self, episode: int, epsilon: float,
                                avg_reward: float, convergence: float):
        """Record metrics for a Q-learning episode"""
        timestamp = datetime.now()

        self.record_metric('qlearning.episodes_completed', episode, timestamp)
        self.record_metric('qlearning.epsilon_value', epsilon, timestamp)
        self.record_metric('qlearning.average_reward', avg_reward, timestamp)
        self.record_metric('qlearning.convergence_score', convergence, timestamp)


# Singleton instance for easy access
_default_collector = None
class MetricType(Enum):
    """Types of metrics that can be collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Individual metric measurement"""
    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str] = None
    metric_type: MetricType = MetricType.GAUGE

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class MetricSummary:
    """Summary statistics for a metric over time"""
    name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    percentile_50: float
    percentile_95: float
    percentile_99: float
    last_updated: datetime

class MetricsCollector:
    """Collects and aggregates system and application metrics"""

    def __init__(self, collection_interval: float = 1.0, retention_hours: int = 24):
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=int(retention_hours * 3600 / collection_interval)))
        self.metric_callbacks: dict[str, Callable] = {}
        self.custom_collectors: list[Callable] = []
        self.running = False
        self._collection_thread = None
        self._lock = threading.Lock()

        # Register default system metrics
        self._register_default_collectors()

    def _register_default_collectors(self):
        """Register default system metric collectors"""
        self.register_custom_collector(self._collect_system_metrics)
        self.register_custom_collector(self._collect_process_metrics)

    def register_metric_callback(self, metric_name: str, callback: Callable[[], float]):
        """Register a callback function to collect a specific metric"""
        self.metric_callbacks[metric_name] = callback

    def register_custom_collector(self, collector: Callable):
        """Register a custom collector function that returns Dict[str, float]"""
        self.custom_collectors.append(collector)

    def start_collection(self):
        """Start automatic metric collection"""
        if self.running:
            return

        self.running = True
        self._collection_thread = threading.Thread(target=self._collection_loop)
        self._collection_thread.daemon = True
        self._collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop automatic metric collection"""
        self.running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

    def _collect_all_metrics(self):
        """Collect all registered metrics"""
        timestamp = datetime.now()

        # Collect callback metrics
        for metric_name, callback in self.metric_callbacks.items():
            try:
                value = callback()
                if value is not None:
                    self.record_metric(metric_name, value, timestamp)
            except Exception as e:
                logger.error(f"Error collecting metric {metric_name}: {e}")

        # Collect custom collector metrics
        for collector in self.custom_collectors:
            try:
                metrics = collector()
                if isinstance(metrics, dict):
                    for name, value in metrics.items():
                        if value is not None:
                            self.record_metric(name, value, timestamp)
            except Exception as e:
                logger.error(f"Error in custom collector: {e}")

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect system-level metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'system.cpu.utilization': cpu_percent,
                'system.memory.utilization': memory.percent,
                'system.memory.available_gb': memory.available / (1024**3),
                'system.disk.utilization': disk.percent,
                'system.disk.free_gb': disk.free / (1024**3),
                'system.load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _collect_process_metrics(self) -> dict[str, float]:
        """Collect current process metrics"""
        try:
            process = psutil.Process()

            return {
                'process.cpu.utilization': process.cpu_percent(),
                'process.memory.rss_mb': process.memory_info().rss / (1024**2),
                'process.memory.vms_mb': process.memory_info().vms / (1024**2),
                'process.threads.count': process.num_threads(),
                'process.file_descriptors.count': process.num_fds() if hasattr(process, 'num_fds') else 0,
            }
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {}

    def record_metric(self, name: str, value: float, timestamp: datetime | None = None,
                     labels: dict[str, str] | None = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a single metric value"""
        if timestamp is None:
            timestamp = datetime.now()

        metric = MetricValue(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels or {},
            metric_type=metric_type
        )

        with self._lock:
            self.metrics[name].append(metric)

    def get_metric_values(self, name: str, since: datetime | None = None) -> list[MetricValue]:
        """Get metric values for a specific metric"""
        with self._lock:
            values = list(self.metrics.get(name, []))

        if since:
            values = [v for v in values if v.timestamp >= since]

        return values

    def get_metric_summary(self, name: str, since: datetime | None = None) -> MetricSummary | None:
        """Get summary statistics for a metric"""
        values = self.get_metric_values(name, since)

        if not values:
            return None

        numeric_values = [v.value for v in values]

        return MetricSummary(
            name=name,
            count=len(numeric_values),
            min_value=float(np.min(numeric_values)),
            max_value=float(np.max(numeric_values)),
            mean_value=float(np.mean(numeric_values)),
            std_value=float(np.std(numeric_values)),
            percentile_50=float(np.percentile(numeric_values, 50)),
            percentile_95=float(np.percentile(numeric_values, 95)),
            percentile_99=float(np.percentile(numeric_values, 99)),
            last_updated=values[-1].timestamp
        )

    def get_all_metric_names(self) -> list[str]:
        """Get all metric names currently being collected"""
        with self._lock:
            return list(self.metrics.keys())

    def clear_metrics(self, name: str | None = None):
        """Clear metrics data"""
        with self._lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()

    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format_type == "prometheus":
            return self._export_prometheus_format()
        elif format_type == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for name in self.get_all_metric_names():
            summary = self.get_metric_summary(name)
            if summary:
                # Convert dots to underscores for Prometheus compatibility
                prom_name = name.replace('.', '_')
                lines.append(f"# HELP {prom_name} Metric {name}")
                lines.append(f"# TYPE {prom_name} gauge")
                lines.append(f"{prom_name} {summary.mean_value}")

        return '\n'.join(lines)

    def _export_json_format(self) -> str:
        """Export metrics in JSON format"""
        import json

        data = {}
        for name in self.get_all_metric_names():
            summary = self.get_metric_summary(name)
            if summary:
                data[name] = asdict(summary)

        return json.dumps(data, default=str, indent=2)

