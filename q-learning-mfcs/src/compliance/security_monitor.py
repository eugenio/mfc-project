"""
SecurityMonitor implementation for comprehensive security monitoring.

This module provides advanced security monitoring capabilities including:
- Threat detection and analysis
- Anomaly detection with configurable thresholds
- Security alerting and notifications
- Incident response automation
- Security metrics collection and reporting
- Real-time monitoring with threat pattern matching
"""
import logging
import threading
import time
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)
class ThreatSeverity(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityConfig:
    """Configuration for SecurityMonitor."""
    threat_detection_enabled: bool = True
    anomaly_detection_enabled: bool = True
    alert_notifications_enabled: bool = True
    incident_response_enabled: bool = True
    metrics_collection_enabled: bool = True
    monitoring_interval: float = 30.0
    threat_retention_days: int = 30
    max_incidents_tracked: int = 1000


@dataclass
class ThreatPattern:
    """Threat pattern definition for detection."""
    name: str
    pattern_type: str  # 'regex', 'payload_content', 'behavioral', etc.
    regex_pattern: Optional[str] = None
    threat_type: str = ""
    severity: ThreatSeverity = ThreatSeverity.MEDIUM
    description: str = ""
    enabled: bool = True

@dataclass
class AnomalyThreshold:
    """Anomaly detection threshold configuration."""
    metric_name: str
    threshold_value: float
    comparison: str = "gt"  # gt, gte, lt, lte, eq
    enabled: bool = True


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    alert_type: str
    severity: ThreatSeverity
    message: str
    source: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

@dataclass
class SecurityIncident:
    """Security incident data structure."""
    incident_id: str
    title: str
    description: str
    severity: ThreatSeverity
    status: str  # 'active', 'investigating', 'resolved', 'closed'
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    related_threat_ids: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)

@dataclass
class SecurityThreat:
    """Security threat data structure."""
    threat_id: str
    threat_type: str
    severity: ThreatSeverity
    source_ip: Optional[str] = None
    target_system: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # 'active', 'mitigated', 'false_positive'

@dataclass
class SecurityMetrics:
    """Security metrics data structure."""
    total_events: int = 0
    threats_detected: int = 0
    alerts_generated: int = 0
    incidents_created: int = 0
    authentication_attempts: int = 0
    authentication_failures: int = 0
    api_requests: int = 0
    blocked_requests: int = 0
    average_response_time: float = 0.0
    uptime_percentage: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)

class SecurityMonitor:
    """
    Comprehensive security monitoring system for MFC infrastructure.
    
    Provides real-time threat detection, anomaly monitoring, alerting,
    and incident response capabilities with configurable patterns and thresholds.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize SecurityMonitor with configuration.
        
        Args:
            config: SecurityConfig instance, uses defaults if None
        """
        self.config = config or SecurityConfig()
        self.is_monitoring = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Storage for security data
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.archived_threats: List[SecurityThreat] = []
        self.alert_history: List[SecurityAlert] = []
        self.incident_history: List[SecurityIncident] = []
        
        # Metrics tracking
        self.metrics = SecurityMetrics()
        
        # Threat patterns for detection
        self.threat_patterns: List[ThreatPattern] = []
        self._initialize_default_threat_patterns()
        
        # Anomaly thresholds
        self.anomaly_thresholds: Dict[str, AnomalyThreshold] = {}
        self._initialize_default_anomaly_thresholds()
        
        # Event handlers
        self.alert_handlers: Dict[str, Callable] = {}
        self.incident_response_handlers: Dict[str, Callable] = {}
        
        logger.info("SecurityMonitor initialized with configuration")

    def _initialize_default_threat_patterns(self):
        """Initialize default threat detection patterns."""
        default_patterns = [
            ThreatPattern(
                name="sql_injection",
                pattern_type="payload_content",
                regex_pattern=r".*('|\"|;|--|\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b).*",
                threat_type="sql_injection_attempt",
                severity=ThreatSeverity.CRITICAL,
                description="SQL injection attack detection"
            ),
            ThreatPattern(
                name="xss_attack",
                pattern_type="payload_content", 
                regex_pattern=r".*(<script|javascript:|on\w+\s*=).*",
                threat_type="xss_attempt",
                severity=ThreatSeverity.HIGH,
                description="Cross-site scripting attack detection"
            ),
            ThreatPattern(
                name="brute_force",
                pattern_type="behavioral",
                threat_type="brute_force_attack",
                severity=ThreatSeverity.HIGH,
                description="Brute force attack detection based on failed login patterns"
            ),
            ThreatPattern(
                name="privilege_escalation",
                pattern_type="behavioral",
                threat_type="privilege_escalation_attempt",
                severity=ThreatSeverity.HIGH,
                description="Unauthorized privilege escalation detection"
            ),
            ThreatPattern(
                name="suspicious_activity",
                pattern_type="behavioral",
                threat_type="suspicious_activity",
                severity=ThreatSeverity.MEDIUM,
                description="General suspicious activity detection"
            )
        ]
        
        self.threat_patterns.extend(default_patterns)

    def _initialize_default_anomaly_thresholds(self):
        """Initialize default anomaly detection thresholds."""
        default_thresholds = {
            "cpu_usage": AnomalyThreshold("cpu_usage", 80.0, "gt"),
            "memory_usage": AnomalyThreshold("memory_usage", 85.0, "gt"),
            "network_throughput": AnomalyThreshold("network_throughput", 1000.0, "gt"),
            "connection_count": AnomalyThreshold("connection_count", 1000, "gt"),
            "error_rate": AnomalyThreshold("error_rate", 5.0, "gt"),
            "response_time": AnomalyThreshold("response_time", 5000.0, "gt")
        }
        
        self.anomaly_thresholds.update(default_thresholds)

    def start_monitoring(self):
        """Start the security monitoring system."""
        if self.is_monitoring:
            logger.warning("Security monitoring is already running")
            return
        
        self.is_monitoring = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Security monitoring started")

    def stop_monitoring(self):
        """Stop the security monitoring system."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10.0)
        
        logger.info("Security monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._check_anomalies()
                self._cleanup_old_data()
                time.sleep(self.config.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def process_security_event(self, event_data: Dict[str, Any]):
        """
        Process a security event and check for threats.
        
        Args:
            event_data: Dictionary containing event information
        """
        if not event_data or not isinstance(event_data, dict):
            logger.warning("Invalid event data received")
            return
        
        try:
            with self._lock:
                self.metrics.total_events += 1
                
                # Update specific metrics based on event type
                event_type = event_data.get('event_type', '')
                self._update_event_metrics(event_type, event_data)
                
                # Check for threats
                threats = self._detect_threats(event_data)
                for threat in threats:
                    self._handle_threat(threat, event_data)
                
                # Log the event
                logger.info(f"Processed security event: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing security event: {e}")

    def _update_event_metrics(self, event_type: str, event_data: Dict[str, Any]):
        """Update metrics based on event type."""
        if event_type == 'authentication_success':
            self.metrics.authentication_attempts += 1
        elif event_type == 'authentication_failure':
            self.metrics.authentication_attempts += 1
            self.metrics.authentication_failures += 1
        elif event_type == 'api_request':
            self.metrics.api_requests += 1
        elif event_type in ['blocked_request', 'request_blocked']:
            self.metrics.blocked_requests += 1

    def _detect_threats(self, event_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats based on event data and patterns."""
        threats = []
        
        for pattern in self.threat_patterns:
            if not pattern.enabled:
                continue
            
            threat = self._check_pattern(pattern, event_data)
            if threat:
                threats.append(threat)
        
        return threats

    def _check_pattern(self, pattern: ThreatPattern, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check if event matches a threat pattern."""
        try:
            if pattern.pattern_type == "payload_content":
                return self._check_payload_pattern(pattern, event_data)
            elif pattern.pattern_type == "behavioral":
                return self._check_behavioral_pattern(pattern, event_data)
            else:
                return None
        except Exception as e:
            logger.error(f"Error checking pattern {pattern.name}: {e}")
            return None

    def _check_payload_pattern(self, pattern: ThreatPattern, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check payload content against regex patterns."""
        if not pattern.regex_pattern:
            return None
        
        # Check various fields that might contain payload
        content_fields = ['payload', 'query', 'data', 'message', 'content']
        
        for field in content_fields:
            if field in event_data:
                content = str(event_data[field])
                if re.search(pattern.regex_pattern, content, re.IGNORECASE):
                    return SecurityThreat(
                        threat_id=str(uuid.uuid4()),
                        threat_type=pattern.threat_type,
                        severity=pattern.severity,
                        source_ip=event_data.get('source_ip'),
                        timestamp=event_data.get('timestamp', datetime.now()),
                        details={
                            'pattern_matched': pattern.name,
                            'content_field': field,
                            'content_sample': content[:100]
                        }
                    )
        
        return None

    def _check_behavioral_pattern(self, pattern: ThreatPattern, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check behavioral patterns like brute force, privilege escalation."""
        event_type = event_data.get('event_type', '')
        
        if pattern.threat_type == "brute_force_attack":
            return self._check_brute_force(event_data)
        elif pattern.threat_type == "privilege_escalation_attempt":
            return self._check_privilege_escalation(event_data)
        elif pattern.threat_type == "suspicious_activity":
            return self._check_suspicious_activity(event_data)
        
        return None

    def _check_brute_force(self, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check for brute force attack patterns."""
        if event_data.get('event_type') != 'authentication_failure':
            return None
        
        source_ip = event_data.get('source_ip')
        if not source_ip:
            return None
        
        # Count recent failures from same IP
        current_time = datetime.now()
        failure_count = 0
        
        for threat in self.active_threats.values():
            if (threat.threat_type == 'brute_force_attack' and 
                threat.source_ip == source_ip and
                (current_time - threat.timestamp).seconds < 300):  # 5 minutes
                failure_count += 1
        
        # Consider login failure events in last 5 minutes
        failure_count += 1  # Current failure
        
        if failure_count >= 5:  # Threshold for brute force
            return SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="brute_force_attack",
                severity=ThreatSeverity.HIGH,
                source_ip=source_ip,
                timestamp=current_time,
                details={'failure_count': failure_count}
            )
        
        return None

    def _check_privilege_escalation(self, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check for privilege escalation attempts."""
        if event_data.get('event_type') != 'access_attempt':
            return None
        
        required_role = event_data.get('required_role', '')
        user_role = event_data.get('user_role', '')
        
        # Simple role hierarchy check
        role_levels = {'user': 1, 'admin': 2, 'super_admin': 3}
        
        required_level = role_levels.get(required_role, 0)
        user_level = role_levels.get(user_role, 0)
        
        if required_level > user_level:
            return SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="privilege_escalation_attempt",
                severity=ThreatSeverity.HIGH,
                timestamp=event_data.get('timestamp', datetime.now()),
                details={
                    'user_id': event_data.get('user_id'),
                    'required_role': required_role,
                    'user_role': user_role,
                    'resource': event_data.get('resource')
                }
            )
        
        return None

    def _check_suspicious_activity(self, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Check for general suspicious activity."""
        suspicious_indicators = [
            'port_scan', 'directory_traversal', 'file_inclusion',
            'command_injection', 'malware', 'ransomware'
        ]
        
        event_type = event_data.get('event_type', '')
        activity_type = event_data.get('activity_type', '')
        attack_type = event_data.get('attack_type', '')
        
        if (event_type in suspicious_indicators or 
            activity_type in suspicious_indicators or
            attack_type in suspicious_indicators):
            
            severity = ThreatSeverity.CRITICAL if attack_type == 'ransomware' else ThreatSeverity.MEDIUM
            
            return SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="suspicious_activity",
                severity=severity,
                source_ip=event_data.get('source_ip'),
                timestamp=event_data.get('timestamp', datetime.now()),
                details=event_data
            )
        
        return None

    def _handle_threat(self, threat: SecurityThreat, event_data: Dict[str, Any]):
        """Handle detected threat by creating alerts and potentially incidents."""
        # Store the threat
        self.active_threats[threat.threat_id] = threat
        self.metrics.threats_detected += 1
        
        # Create security alert
        alert = self._create_threat_alert(threat)
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.metrics.alerts_generated += 1
        
        # Check if we need to create an incident
        self._evaluate_incident_creation(threat)
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        logger.warning(f"Threat detected: {threat.threat_type} - {threat.severity.value}")

    def _create_threat_alert(self, threat: SecurityThreat) -> SecurityAlert:
        """Create a security alert from a threat."""
        return SecurityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="threat_detected",
            severity=threat.severity,
            message=f"Security threat detected: {threat.threat_type}",
            source="security_monitor",
            timestamp=datetime.now(),
            details={
                'threat_id': threat.threat_id,
                'threat_type': threat.threat_type,
                'source_ip': threat.source_ip,
                'target_system': threat.target_system
            }
        )

    def _evaluate_incident_creation(self, threat: SecurityThreat):
        """Evaluate if threat should trigger incident creation."""
        # Create incident for critical threats or multiple related threats
        if threat.severity == ThreatSeverity.CRITICAL:
            self._create_incident([threat])
        else:
            # Check for related threats from same source
            related_threats = self._find_related_threats(threat)
            if len(related_threats) >= 2:  # Including current threat
                self._create_incident(related_threats)

    def _find_related_threats(self, threat: SecurityThreat) -> List[SecurityThreat]:
        """Find threats related to the given threat."""
        related = [threat]
        current_time = datetime.now()
        
        for existing_threat in self.active_threats.values():
            if (existing_threat.threat_id != threat.threat_id and
                existing_threat.source_ip == threat.source_ip and
                (current_time - existing_threat.timestamp).seconds < 3600):  # 1 hour window
                related.append(existing_threat)
        
        return related

    def _create_incident(self, threats: List[SecurityThreat]):
        """Create a security incident from multiple threats."""
        incident_id = str(uuid.uuid4())
        
        # Determine highest severity
        max_severity = max(threat.severity for threat in threats)
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"Security Incident - Multiple threats detected",
            description=f"Incident created from {len(threats)} related threats",
            severity=max_severity,
            status="active",
            created_at=datetime.now(),
            related_threat_ids=[t.threat_id for t in threats],
            affected_systems=list(set(t.target_system for t in threats if t.target_system))
        )
        
        self.active_incidents[incident_id] = incident
        self.incident_history.append(incident)
        self.metrics.incidents_created += 1
        
        # Trigger incident response
        self._trigger_incident_response(incident)
        
        logger.error(f"Security incident created: {incident_id}")

    def _trigger_alert_handlers(self, alert: SecurityAlert):
        """Trigger registered alert handlers."""
        if not self.config.alert_notifications_enabled:
            return
        
        for handler_name, handler in self.alert_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler {handler_name}: {e}")

    def _trigger_incident_response(self, incident: SecurityIncident):
        """Trigger automated incident response."""
        if not self.config.incident_response_enabled:
            return
        
        for handler_name, handler in self.incident_response_handlers.items():
            try:
                handler(incident)
                incident.response_actions.append(f"Executed {handler_name}")
            except Exception as e:
                logger.error(f"Error in incident response handler {handler_name}: {e}")

    def process_metrics_data(self, metrics_data: Dict[str, Any]):
        """Process system metrics and check for anomalies."""
        if not self.config.anomaly_detection_enabled:
            return
        
        try:
            metric_type = metrics_data.get('metric_type', '')
            timestamp = metrics_data.get('timestamp', datetime.now())
            
            for metric_name, value in metrics_data.items():
                if metric_name in ['metric_type', 'timestamp']:
                    continue
                
                if isinstance(value, (int, float)):
                    anomaly = self._check_anomaly(metric_name, value, timestamp)
                    if anomaly:
                        self._handle_anomaly(anomaly, metrics_data)
                        
        except Exception as e:
            logger.error(f"Error processing metrics data: {e}")

    def _check_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Check if metric value represents an anomaly."""
        if metric_name not in self.anomaly_thresholds:
            return None
        
        threshold = self.anomaly_thresholds[metric_name]
        if not threshold.enabled:
            return None
        
        is_anomaly = False
        
        if threshold.comparison == "gt" and value > threshold.threshold_value:
            is_anomaly = True
        elif threshold.comparison == "gte" and value >= threshold.threshold_value:
            is_anomaly = True
        elif threshold.comparison == "lt" and value < threshold.threshold_value:
            is_anomaly = True
        elif threshold.comparison == "lte" and value <= threshold.threshold_value:
            is_anomaly = True
        elif threshold.comparison == "eq" and value == threshold.threshold_value:
            is_anomaly = True
        
        if is_anomaly:
            return {
                'metric_name': metric_name,
                'value': value,
                'threshold': threshold.threshold_value,
                'comparison': threshold.comparison,
                'timestamp': timestamp
            }
        
        return None

    def _handle_anomaly(self, anomaly: Dict[str, Any], metrics_data: Dict[str, Any]):
        """Handle detected anomaly by creating alert."""
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type="anomaly_detected",
            severity=ThreatSeverity.MEDIUM,
            message=f"Anomaly detected in {anomaly['metric_name']}: {anomaly['value']} > {anomaly['threshold']}",
            source="anomaly_detector",
            timestamp=anomaly['timestamp'],
            details=anomaly
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.metrics.alerts_generated += 1
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        logger.warning(f"Anomaly detected: {anomaly['metric_name']} = {anomaly['value']}")

    def _collect_system_metrics(self):
        """Collect system metrics for monitoring."""
        if not self.config.metrics_collection_enabled:
            return
        
        try:
            # Update metrics timestamp
            self.metrics.last_updated = datetime.now()
            
            # This would integrate with actual system monitoring
            # For now, we'll just update the timestamp
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _check_anomalies(self):
        """Check current system state for anomalies."""
        # This would check current system metrics against thresholds
        # Implementation would depend on actual metrics collection system
        pass

    def _cleanup_old_data(self):
        """Clean up old threats, alerts, and incidents based on retention policy."""
        try:
            current_time = datetime.now()
            retention_period = timedelta(days=self.config.threat_retention_days)
            
            # Archive old threats
            threats_to_archive = []
            for threat_id, threat in list(self.active_threats.items()):
                if current_time - threat.timestamp > retention_period:
                    threats_to_archive.append(threat_id)
            
            for threat_id in threats_to_archive:
                threat = self.active_threats.pop(threat_id)
                self.archived_threats.append(threat)
            
            # Limit archived threats to prevent memory issues
            if len(self.archived_threats) > 10000:
                self.archived_threats = self.archived_threats[-5000:]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    # Public API methods

    def add_threat_pattern(self, pattern: ThreatPattern):
        """Add a custom threat pattern."""
        self.threat_patterns.append(pattern)
        logger.info(f"Added threat pattern: {pattern.name}")

    def update_anomaly_threshold(self, metric_name: str, threshold_value: float, comparison: str = "gt"):
        """Update anomaly threshold for a metric."""
        self.anomaly_thresholds[metric_name] = AnomalyThreshold(
            metric_name, threshold_value, comparison
        )
        logger.info(f"Updated anomaly threshold for {metric_name}: {threshold_value}")

    def get_anomaly_threshold(self, metric_name: str) -> Optional[float]:
        """Get current anomaly threshold for a metric."""
        threshold = self.anomaly_thresholds.get(metric_name)
        return threshold.threshold_value if threshold else None

    def add_alert_handler(self, handler_name: str, handler: Callable[[SecurityAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers[handler_name] = handler
        logger.info(f"Added alert handler: {handler_name}")

    def add_incident_response_handler(self, handler_name: str, handler: Callable[[SecurityIncident], None]):
        """Add an incident response handler function."""
        self.incident_response_handlers[handler_name] = handler
        logger.info(f"Added incident response handler: {handler_name}")

    def get_active_threats(self) -> List[SecurityThreat]:
        """Get list of active security threats."""
        with self._lock:
            return list(self.active_threats.values())

    def get_active_alerts(self) -> List[SecurityAlert]:
        """Get list of active security alerts."""
        with self._lock:
            return list(self.active_alerts.values())

    def get_active_incidents(self) -> List[SecurityIncident]:
        """Get list of active security incidents."""
        with self._lock:
            return list(self.active_incidents.values())

    def get_archived_threats(self) -> List[SecurityThreat]:
        """Get list of archived threats."""
        return self.archived_threats.copy()

    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        return self.metrics

    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Resolve a security alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                logger.info(f"Alert resolved: {alert_id}")

    def resolve_incident(self, incident_id: str, resolution_notes: str = ""):
        """Resolve a security incident."""
        with self._lock:
            if incident_id in self.active_incidents:
                incident = self.active_incidents.pop(incident_id)
                incident.status = "resolved"
                incident.resolved_at = datetime.now()
                logger.info(f"Incident resolved: {incident_id}")

    def calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)."""
        base_score = 100.0
        
        # Deduct points for active threats
        for threat in self.active_threats.values():
            if threat.severity == ThreatSeverity.CRITICAL:
                base_score -= 20.0
            elif threat.severity == ThreatSeverity.HIGH:
                base_score -= 10.0
            elif threat.severity == ThreatSeverity.MEDIUM:
                base_score -= 5.0
            elif threat.severity == ThreatSeverity.LOW:
                base_score -= 2.0
        
        # Deduct points for active incidents
        for incident in self.active_incidents.values():
            if incident.severity == ThreatSeverity.CRITICAL:
                base_score -= 15.0
            elif incident.severity == ThreatSeverity.HIGH:
                base_score -= 8.0
        
        return max(0.0, base_score)

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'security_score': self.calculate_security_score(),
            'active_threats': [asdict(threat) for threat in self.active_threats.values()],
            'active_incidents': [asdict(incident) for incident in self.active_incidents.values()],
            'metrics': asdict(self.metrics),
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        if len(self.active_threats) > 10:
            recommendations.append("High number of active threats detected. Consider reviewing security policies.")
        
        if len(self.active_incidents) > 0:
            recommendations.append("Active security incidents require immediate attention.")
        
        if self.calculate_security_score() < 80:
            recommendations.append("Security score is below optimal. Review and address security issues.")
        
        return recommendations

    def export_security_data(self, format: str = 'json') -> str:
        """Export security data in specified format."""
        data = {
            'threats': [asdict(threat) for threat in self.active_threats.values()],
            'alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'incidents': [asdict(incident) for incident in self.active_incidents.values()],
            'metrics': asdict(self.metrics),
            'timestamp': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def archive_old_data(self):
        """Manually trigger archiving of old data."""
        self._cleanup_old_data()
        logger.info("Manual data archiving completed")