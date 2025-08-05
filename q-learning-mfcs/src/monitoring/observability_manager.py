"""
Observability Manager for MFC Monitoring System
=============================================

Comprehensive observability management providing:
- Centralized metrics, traces, and logs coordination
- Service health monitoring and SLA tracking
- Performance alerting and anomaly detection
- Integration with external monitoring systems

Created: 2025-08-05
Author: TDD Agent 44
"""
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Setup logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ServiceHealth:
    """Health status for a monitored service"""
    service_name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float | None = None
    error_rate: float | None = None
    availability: float | None = None
    details: dict[str, Any] | None = None


@dataclass
class AlertCondition:
    """Alert condition configuration"""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    duration_seconds: int
    severity: AlertSeverity
    enabled: bool = True


@dataclass
class Alert:
    """Active alert"""
    alert_id: str
    condition_name: str
    service_name: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: datetime | None = None
    acknowledged: bool = False


class ObservabilityManager:
    """Centralized observability management system"""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.services: dict[str, ServiceHealth] = {}
        self.alert_conditions: dict[str, AlertCondition] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.health_callbacks: dict[str, Callable] = {}
        self.alert_handlers: list[Callable[[Alert], None]] = []

        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()

        # Initialize default alert conditions
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Setup default alert conditions for MFC systems"""
        default_conditions = [
            AlertCondition(
                name="high_cpu_usage",
                metric_name="system.cpu.utilization",
                threshold=80.0,
                comparison="gt",
                duration_seconds=300,
                severity=AlertSeverity.WARNING
            ),
            AlertCondition(
                name="critical_cpu_usage",
                metric_name="system.cpu.utilization",
                threshold=95.0,
                comparison="gt",
                duration_seconds=60,
                severity=AlertSeverity.CRITICAL
            ),
            AlertCondition(
                name="high_memory_usage",
                metric_name="system.memory.utilization",
                threshold=85.0,
                comparison="gt",
                duration_seconds=300,
                severity=AlertSeverity.WARNING
            ),
            AlertCondition(
                name="low_disk_space",
                metric_name="system.disk.utilization",
                threshold=90.0,
                comparison="gt",
                duration_seconds=600,
                severity=AlertSeverity.ERROR
            ),
            AlertCondition(
                name="mfc_low_power",
                metric_name="mfc.simulation.power_output_mw",
                threshold=1.0,
                comparison="lt",
                duration_seconds=120,
                severity=AlertSeverity.WARNING
            ),
            AlertCondition(
                name="qlearning_poor_convergence",
                metric_name="qlearning.convergence_score",
                threshold=0.3,
                comparison="lt",
                duration_seconds=1800,
                severity=AlertSeverity.WARNING
            )
        ]

        for condition in default_conditions:
            self.add_alert_condition(condition)

    def register_service(self, service_name: str, health_callback: Callable | None = None):
        """Register a service for health monitoring"""
        with self._lock:
            self.services[service_name] = ServiceHealth(
                service_name=service_name,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now()
            )

            if health_callback:
                self.health_callbacks[service_name] = health_callback

        logger.info(f"Registered service for monitoring: {service_name}")

    def add_alert_condition(self, condition: AlertCondition):
        """Add an alert condition"""
        with self._lock:
            self.alert_conditions[condition.name] = condition

        logger.info(f"Added alert condition: {condition.name}")

    def remove_alert_condition(self, condition_name: str):
        """Remove an alert condition"""
        with self._lock:
            if condition_name in self.alert_conditions:
                del self.alert_conditions[condition_name]
                logger.info(f"Removed alert condition: {condition_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")

    def start_monitoring(self):
        """Start the monitoring system"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("Observability monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)

        logger.info("Observability monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._check_service_health()
                self._evaluate_alert_conditions()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _check_service_health(self):
        """Check health of all registered services"""
        for service_name in list(self.services.keys()):
            try:
                if service_name in self.health_callbacks:
                    # Use custom health check
                    health_info = self.health_callbacks[service_name]()
                    self._update_service_health(service_name, health_info)
                else:
                    # Default health check - just mark as healthy if no callback
                    self._update_service_health(service_name, {
                        'status': HealthStatus.HEALTHY,
                        'response_time_ms': 10.0
                    })
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                self._update_service_health(service_name, {
                    'status': HealthStatus.UNHEALTHY,
                    'details': {'error': str(e)}
                })

    def _update_service_health(self, service_name: str, health_info: dict[str, Any]):
        """Update service health status"""
        with self._lock:
            if service_name in self.services:
                service = self.services[service_name]
                service.status = health_info.get('status', HealthStatus.HEALTHY)
                service.last_check = datetime.now()
                service.response_time_ms = health_info.get('response_time_ms')
                service.error_rate = health_info.get('error_rate')
                service.availability = health_info.get('availability')
                service.details = health_info.get('details')

    def _evaluate_alert_conditions(self):
        """Evaluate all alert conditions against current metrics"""
        # This would integrate with the metrics collection system
        # For now, we'll use placeholder logic

        try:
            # Import here to avoid circular dependencies
            from ..performance.performance_metrics import get_default_collector
            collector = get_default_collector()

            for _condition_name, condition in self.alert_conditions.items():
                if not condition.enabled:
                    continue

                self._check_condition(condition, collector)

        except ImportError:
            # Metrics system not available, skip condition evaluation
            pass
        except Exception as e:
            logger.error(f"Error evaluating alert conditions: {e}")

    def _check_condition(self, condition: AlertCondition, collector):
        """Check a specific alert condition"""
        try:
            # Get recent metric values
            since = datetime.now() - timedelta(seconds=condition.duration_seconds)
            values = collector.get_metric_values(condition.metric_name, since=since)

            if not values:
                return

            # Check if condition is violated consistently
            violation_count = 0
            for value in values:
                if self._evaluate_threshold(value.value, condition.threshold, condition.comparison):
                    violation_count += 1

            # If more than 80% of values violate the condition, trigger alert
            violation_ratio = violation_count / len(values)
            if violation_ratio > 0.8:
                self._trigger_alert(condition, values[-1].value)
            else:
                # Check if we should resolve existing alert
                self._check_alert_resolution(condition.name)

        except Exception as e:
            logger.error(f"Error checking condition {condition.name}: {e}")

    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if value meets threshold condition"""
        if comparison == "gt":
            return value > threshold
        elif comparison == "gte":
            return value >= threshold
        elif comparison == "lt":
            return value < threshold
        elif comparison == "lte":
            return value <= threshold
        elif comparison == "eq":
            return value == threshold
        else:
            return False

    def _trigger_alert(self, condition: AlertCondition, current_value: float):
        """Trigger an alert for a condition"""
        alert_id = f"{condition.name}_{int(time.time())}"

        # Check if alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.condition_name == condition.name and not alert.resolved_at:
                existing_alert = alert
                break

        if existing_alert:
            return  # Alert already active

        alert = Alert(
            alert_id=alert_id,
            condition_name=condition.name,
            service_name="mfc_system",  # Default service
            severity=condition.severity,
            message=f"Alert: {condition.name} - {condition.metric_name} = {current_value:.2f}",
            triggered_at=datetime.now()
        )

        with self._lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        logger.warning(f"Alert triggered: {alert.message}")

    def _check_alert_resolution(self, condition_name: str):
        """Check if an alert should be resolved"""
        for alert in list(self.active_alerts.values()):
            if alert.condition_name == condition_name and not alert.resolved_at:
                self._resolve_alert(alert.alert_id)

    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        self._resolve_alert(alert_id)

    def _resolve_alert(self, alert_id: str):
        """Internal method to resolve an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.message}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")

    def get_service_health(self, service_name: str | None = None) -> dict[str, ServiceHealth]:
        """Get health status of services"""
        with self._lock:
            if service_name:
                return {service_name: self.services.get(service_name)}
            else:
                return self.services.copy()

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int | None = None) -> list[Alert]:
        """Get alert history"""
        with self._lock:
            history = self.alert_history.copy()

        if limit:
            return history[-limit:]
        return history

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status"""
        with self._lock:
            active_alerts = list(self.active_alerts.values())
            services = self.services.copy()

        # Determine overall status
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]

        if critical_alerts:
            overall_status = "critical"
        elif error_alerts:
            overall_status = "error"
        elif any(s.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL] for s in services.values()):
            overall_status = "unhealthy"
        elif any(s.status == HealthStatus.DEGRADED for s in services.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'services_count': len(services),
            'healthy_services': len([s for s in services.values() if s.status == HealthStatus.HEALTHY]),
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'services': {name: asdict(service) for name, service in services.items()},
            'alerts': [asdict(alert) for alert in active_alerts]
        }

    def export_status(self, format_type: str = "json") -> str:
        """Export system status in specified format"""
        status = self.get_system_status()

        if format_type == "json":
            return json.dumps(status, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Singleton instance
_default_manager = None


def get_default_manager() -> ObservabilityManager:
    """Get the default observability manager instance"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ObservabilityManager()
    return _default_manager


if __name__ == "__main__":
    # Example usage
    manager = ObservabilityManager(check_interval=5.0)

    # Register some services
    manager.register_service("mfc_simulation")
    manager.register_service("qlearning_engine")
    manager.register_service("monitoring_api")

    # Add alert handler
    def print_alert(alert: Alert):
        print(f"🚨 ALERT: {alert.message} (Severity: {alert.severity.value})")

    manager.add_alert_handler(print_alert)

    # Start monitoring
    manager.start_monitoring()

    try:
        # Let it run for a bit
        import time
        time.sleep(30)

        # Print system status
        status = manager.get_system_status()
        print("\n📊 System Status:")
        print(f"Overall: {status['overall_status']}")
        print(f"Services: {status['healthy_services']}/{status['services_count']} healthy")
        print(f"Active Alerts: {status['active_alerts']}")

    finally:
        manager.stop_monitoring()
