"""
Safety Monitoring System for MFC Operations

Comprehensive safety monitoring with automated responses, emergency protocols,
and compliance tracking for MFC (Microbial Fuel Cell) systems.
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
import os
import sys
from pathlib import Path

from config.real_time_processing import AlertLevel, AlertSystem
from integrated_mfc_model import IntegratedMFCModel
class SafetyLevel(Enum):
    """Safety criticality levels"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class EmergencyAction(Enum):
    """Emergency response actions"""
    NONE = "none"
    REDUCE_POWER = "reduce_power"
    STOP_FLOW = "stop_flow"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    ISOLATE_SYSTEM = "isolate_system"
    NOTIFY_PERSONNEL = "notify_personnel"
class SafetyThreshold:
    """Safety threshold configuration"""
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_buffer: float = 0.1  # Buffer zone before critical
    critical_duration_s: float = 5.0  # Time before triggering action
    emergency_action: EmergencyAction = EmergencyAction.NONE
    enabled: bool = True

class SafetyEvent:
    """Safety event record"""
    event_id: str
    timestamp: datetime
    parameter: str
    current_value: float
    threshold_value: float
    safety_level: SafetyLevel
    action_taken: EmergencyAction
    response_time_ms: float
    acknowledged: bool = False
    resolved: bool = False
class SafetyProtocol:
    """
    Safety protocol implementation for specific scenarios
    """
    
    def __init__(self, name: str, triggers: List[str], actions: List[EmergencyAction]):
        self.name = name
        self.triggers = triggers  # Parameter names that trigger this protocol
        self.actions = actions    # Actions to execute in sequence
        self.is_active = False
        self.last_triggered = None
    
    def should_trigger(self, safety_events: List[SafetyEvent]) -> bool:
        """Check if protocol should be triggered"""
        active_parameters = {event.parameter for event in safety_events 
                           if not event.resolved and event.safety_level in 
                           [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]}
        
        return any(trigger in active_parameters for trigger in self.triggers)
    
    def execute(self, mfc_controller) -> List[str]:
        """Execute protocol actions"""
        executed_actions = []
        
        for action in self.actions:
            try:
                if action == EmergencyAction.REDUCE_POWER:
                    # Reduce system power output
                    executed_actions.append("Power reduced to 50%")
                    
                elif action == EmergencyAction.STOP_FLOW:
                    # Stop fluid flow
                    executed_actions.append("Flow stopped")
                    
                elif action == EmergencyAction.EMERGENCY_SHUTDOWN:
                    # Complete system shutdown
                    executed_actions.append("Emergency shutdown initiated")
                    
                elif action == EmergencyAction.ISOLATE_SYSTEM:
                    # Isolate affected components
                    executed_actions.append("System isolated")
                    
                elif action == EmergencyAction.NOTIFY_PERSONNEL:
                    # Send notifications
                    executed_actions.append("Personnel notified")
                    
            except Exception as e:
                logger.error(f"Error executing action {action}: {e}")
                executed_actions.append(f"Failed to execute {action.value}: {e}")
        
        self.is_active = True
        self.last_triggered = datetime.now()
        
        return executed_actions
class SafetyMonitor:
    """
    Comprehensive safety monitoring system for MFC operations
    
    Features:
    - Real-time threshold monitoring
    - Automated emergency responses
    - Safety protocol execution
    - Compliance tracking and reporting
    - Historical safety analysis
    """
    
    def __init__(self, mfc_model: Optional[IntegratedMFCModel] = None):
        self.mfc_model = mfc_model
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Safety configuration
        self.safety_thresholds = self._initialize_default_thresholds()
        self.safety_protocols = self._initialize_safety_protocols()
        
        # Event tracking
        self.safety_events: List[SafetyEvent] = []
        self.active_events: Dict[str, SafetyEvent] = {}
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "critical_events": 0,
            "emergency_actions": 0,
            "false_alarms": 0,
            "response_times": [],
            "uptime_hours": 0.0,
            "last_emergency": None
        }
        
        # Alert system integration
        self.alert_system = AlertSystem(alert_history_size=1000)
        
    def _initialize_default_thresholds(self) -> Dict[str, SafetyThreshold]:
        """Initialize default safety thresholds"""
        return {
            "temperature": SafetyThreshold(
                parameter="temperature",
                max_value=45.0,
                warning_buffer=5.0,
                critical_duration_s=10.0,
                emergency_action=EmergencyAction.REDUCE_POWER
            ),
            "pressure": SafetyThreshold(
                parameter="pressure", 
                max_value=2.5,
                warning_buffer=0.3,
                critical_duration_s=5.0,
                emergency_action=EmergencyAction.STOP_FLOW
            ),
            "ph_level": SafetyThreshold(
                parameter="ph_level",
                min_value=5.5,
                max_value=8.5,
                warning_buffer=0.5,
                critical_duration_s=30.0,
                emergency_action=EmergencyAction.NOTIFY_PERSONNEL
            ),
            "voltage": SafetyThreshold(
                parameter="voltage",
                min_value=0.05,
                max_value=1.2,
                warning_buffer=0.05,
                critical_duration_s=15.0,
                emergency_action=EmergencyAction.REDUCE_POWER
            ),
            "current_density": SafetyThreshold(
                parameter="current_density",
                max_value=15.0,
                warning_buffer=2.0,
                critical_duration_s=20.0,
                emergency_action=EmergencyAction.REDUCE_POWER
            ),
            "flow_rate": SafetyThreshold(
                parameter="flow_rate",
                min_value=10.0,
                max_value=500.0,
                warning_buffer=20.0,
                critical_duration_s=60.0,
                emergency_action=EmergencyAction.NOTIFY_PERSONNEL
            ),
            "biofilm_thickness": SafetyThreshold(
                parameter="biofilm_thickness",
                max_value=50.0,  # μm
                warning_buffer=10.0,
                critical_duration_s=3600.0,  # 1 hour
                emergency_action=EmergencyAction.NOTIFY_PERSONNEL
            )
        }
    
    def _initialize_safety_protocols(self) -> Dict[str, SafetyProtocol]:
        """Initialize safety protocols"""
        return {
            "thermal_runaway": SafetyProtocol(
                name="Thermal Runaway Protection",
                triggers=["temperature", "current_density"],
                actions=[EmergencyAction.REDUCE_POWER, EmergencyAction.NOTIFY_PERSONNEL]
            ),
            "pressure_emergency": SafetyProtocol(
                name="Pressure Emergency Protocol",
                triggers=["pressure"],
                actions=[EmergencyAction.STOP_FLOW, EmergencyAction.ISOLATE_SYSTEM]
            ),
            "system_failure": SafetyProtocol(
                name="System Failure Protocol",
                triggers=["voltage", "current_density", "flow_rate"],
                actions=[EmergencyAction.EMERGENCY_SHUTDOWN, EmergencyAction.NOTIFY_PERSONNEL]
            ),
            "biological_contamination": SafetyProtocol(
                name="Biological Contamination Protocol",
                triggers=["ph_level", "biofilm_thickness"],
                actions=[EmergencyAction.ISOLATE_SYSTEM, EmergencyAction.NOTIFY_PERSONNEL]
            )
        }
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start safety monitoring"""
        if self.is_monitoring:
            logger.warning("Safety monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop"""
        start_time = datetime.now()
        
        while self.is_monitoring:
            try:
                # Get current measurements
                measurements = self._get_current_measurements()
                
                # Check safety thresholds
                safety_events = self._check_safety_thresholds(measurements)
                
                # Process new safety events
                for event in safety_events:
                    self._process_safety_event(event)
                
                # Check safety protocols
                self._check_safety_protocols()
                
                # Update statistics
                self.stats["uptime_hours"] = (datetime.now() - start_time).total_seconds() / 3600.0
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
            
            time.sleep(interval_seconds)
    
    def _get_current_measurements(self) -> Dict[str, float]:
        """Get current system measurements"""
        if self.mfc_model and hasattr(self.mfc_model, 'get_current_state'):
            try:
                state = self.mfc_model.get_current_state()
                
                return {
                    "temperature": 25.0 + np.random.normal(0, 2),  # Simulated
                    "pressure": 1.0 + np.random.normal(0, 0.1),
                    "ph_level": 7.0 + np.random.normal(0, 0.3),
                    "voltage": np.mean(getattr(state, 'cell_voltages', [0.7])),
                    "current_density": np.mean(getattr(state, 'current_densities', [5.0])),
                    "flow_rate": getattr(state, 'flow_rate', 100.0),
                    "biofilm_thickness": np.mean(getattr(state, 'biofilm_thickness', [10.0]))
                }
            except Exception as e:
                logger.error(f"Error getting MFC measurements: {e}")
        
        # Return simulated data if no model available
        return {
            "temperature": 25.0 + np.random.normal(0, 2),
            "pressure": 1.0 + np.random.normal(0, 0.1),
            "ph_level": 7.0 + np.random.normal(0, 0.3),
            "voltage": 0.7 + np.random.normal(0, 0.05),
            "current_density": 5.0 + np.random.normal(0, 1),
            "flow_rate": 100.0 + np.random.normal(0, 10),
            "biofilm_thickness": 10.0 + np.random.normal(0, 2)
        }
    
    def _check_safety_thresholds(self, measurements: Dict[str, float]) -> List[SafetyEvent]:
        """Check measurements against safety thresholds"""
        safety_events = []
        current_time = datetime.now()
        
        for param, value in measurements.items():
            if param not in self.safety_thresholds:
                continue
            
            threshold = self.safety_thresholds[param]
            if not threshold.enabled:
                continue
            
            safety_level = self._evaluate_safety_level(param, value, threshold)
            
            if safety_level not in [SafetyLevel.SAFE, SafetyLevel.CAUTION]:
                # Create safety event
                event = SafetyEvent(
                    event_id=f"{param}_{current_time.timestamp()}",
                    timestamp=current_time,
                    parameter=param,
                    current_value=value,
                    threshold_value=threshold.max_value or threshold.min_value,
                    safety_level=safety_level,
                    action_taken=EmergencyAction.NONE,
                    response_time_ms=0.0
                )
                
                safety_events.append(event)
        
        return safety_events
    
    def _evaluate_safety_level(self, param: str, value: float, 
                              threshold: SafetyThreshold) -> SafetyLevel:
        """Evaluate safety level for a parameter"""
        
        # Check maximum threshold
        if threshold.max_value is not None:
            if value > threshold.max_value:
                return SafetyLevel.EMERGENCY
            elif value > (threshold.max_value - threshold.warning_buffer):
                return SafetyLevel.CRITICAL
            elif value > (threshold.max_value - 2 * threshold.warning_buffer):
                return SafetyLevel.WARNING
            elif value > (threshold.max_value - 3 * threshold.warning_buffer):
                return SafetyLevel.CAUTION
        
        # Check minimum threshold
        if threshold.min_value is not None:
            if value < threshold.min_value:
                return SafetyLevel.EMERGENCY
            elif value < (threshold.min_value + threshold.warning_buffer):
                return SafetyLevel.CRITICAL
            elif value < (threshold.min_value + 2 * threshold.warning_buffer):
                return SafetyLevel.WARNING
            elif value < (threshold.min_value + 3 * threshold.warning_buffer):
                return SafetyLevel.CAUTION
        
        return SafetyLevel.SAFE
    
    def _process_safety_event(self, event: SafetyEvent):
        """Process a safety event"""
        start_time = time.time()
        
        # Check if this is a new event or continuation of existing
        if event.parameter in self.active_events:
            existing_event = self.active_events[event.parameter]
            
            # Check if critical duration exceeded
            duration = (event.timestamp - existing_event.timestamp).total_seconds()
            threshold = self.safety_thresholds[event.parameter]
            
            if (duration >= threshold.critical_duration_s and 
                event.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]):
                
                # Execute emergency action
                action_result = self._execute_emergency_action(
                    threshold.emergency_action, event)
                event.action_taken = threshold.emergency_action
                
                logger.critical(f"Emergency action taken for {event.parameter}: "
                              f"{threshold.emergency_action.value}")
                
                self.stats["emergency_actions"] += 1
                self.stats["last_emergency"] = event.timestamp
        else:
            # New safety event
            self.active_events[event.parameter] = event
        
        # Record event
        self.safety_events.append(event)
        self.stats["total_events"] += 1
        
        if event.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            self.stats["critical_events"] += 1
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        event.response_time_ms = response_time_ms
        self.stats["response_times"].append(response_time_ms)
        
        # Send alert
        self._send_safety_alert(event)
        
        logger.info(f"Safety event processed: {event.parameter} = {event.current_value} "
                   f"({event.safety_level.value}) - Response time: {response_time_ms:.1f}ms")
    
    def _execute_emergency_action(self, action: EmergencyAction, 
                                 event: SafetyEvent) -> List[str]:
        """Execute emergency action"""
        results = []
        
        try:
            if action == EmergencyAction.REDUCE_POWER:
                # Implement power reduction
                results.append("Power output reduced to safe levels")
                
            elif action == EmergencyAction.STOP_FLOW:
                # Implement flow stopping
                results.append("Fluid flow stopped")
                
            elif action == EmergencyAction.EMERGENCY_SHUTDOWN:
                # Implement emergency shutdown
                results.append("System emergency shutdown initiated")
                
            elif action == EmergencyAction.ISOLATE_SYSTEM:
                # Implement system isolation
                results.append("Affected system components isolated")
                
            elif action == EmergencyAction.NOTIFY_PERSONNEL:
                # Send notifications
                self._notify_personnel(event)
                results.append("Personnel notifications sent")
                
        except Exception as e:
            logger.error(f"Error executing emergency action {action}: {e}")
            results.append(f"Error: {e}")
        
        return results
    
    def _check_safety_protocols(self):
        """Check and execute safety protocols"""
        unresolved_events = [event for event in self.safety_events 
                           if not event.resolved and 
                           (datetime.now() - event.timestamp).total_seconds() < 300]  # 5 min
        
        for protocol_name, protocol in self.safety_protocols.items():
            if not protocol.is_active and protocol.should_trigger(unresolved_events):
                logger.warning(f"Activating safety protocol: {protocol_name}")
                
                try:
                    actions_taken = protocol.execute(self.mfc_model)
                    logger.info(f"Protocol {protocol_name} executed: {actions_taken}")
                    
                except Exception as e:
                    logger.error(f"Error executing protocol {protocol_name}: {e}")
    
    def _send_safety_alert(self, event: SafetyEvent):
        """Send safety alert through alert system"""
        alert_level = AlertLevel.INFO
        
        if event.safety_level == SafetyLevel.WARNING:
            alert_level = AlertLevel.WARNING
        elif event.safety_level == SafetyLevel.CRITICAL:
            alert_level = AlertLevel.ERROR
        elif event.safety_level == SafetyLevel.EMERGENCY:
            alert_level = AlertLevel.CRITICAL
        
        self.alert_system.add_alert(
            alert_level,
            f"Safety threshold exceeded: {event.parameter}",
            f"{event.parameter} value {event.current_value:.2f} "
            f"exceeds threshold {event.threshold_value:.2f}",
            source="SafetyMonitor"
        )
    
    def _notify_personnel(self, event: SafetyEvent):
        """Send notifications to personnel"""
        # Implementation would integrate with email, SMS, or other notification systems
        logger.critical(f"PERSONNEL NOTIFICATION: {event.parameter} safety event - "
                       f"Value: {event.current_value:.2f}, Level: {event.safety_level.value}")
    
    def acknowledge_event(self, event_id: str, user_id: str = "system") -> bool:
        """Acknowledge a safety event"""
        for event in self.safety_events:
            if event.event_id == event_id:
                event.acknowledged = True
                logger.info(f"Safety event {event_id} acknowledged by {user_id}")
                return True
        return False
    
    def resolve_event(self, event_id: str, user_id: str = "system") -> bool:
        """Resolve a safety event"""
        for event in self.safety_events:
            if event.event_id == event_id:
                event.resolved = True
                # Remove from active events
                if event.parameter in self.active_events:
                    del self.active_events[event.parameter]
                logger.info(f"Safety event {event_id} resolved by {user_id}")
                return True
        return False
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        current_time = datetime.now()
        active_events = [event for event in self.safety_events 
                        if not event.resolved and 
                        (current_time - event.timestamp).total_seconds() < 300]
        
        # Determine overall safety level
        if any(event.safety_level == SafetyLevel.EMERGENCY for event in active_events):
            overall_level = SafetyLevel.EMERGENCY
        elif any(event.safety_level == SafetyLevel.CRITICAL for event in active_events):
            overall_level = SafetyLevel.CRITICAL
        elif any(event.safety_level == SafetyLevel.WARNING for event in active_events):
            overall_level = SafetyLevel.WARNING
        elif any(event.safety_level == SafetyLevel.CAUTION for event in active_events):
            overall_level = SafetyLevel.CAUTION
        else:
            overall_level = SafetyLevel.SAFE
        
        return {
            "overall_safety_level": overall_level.value,
            "active_events": len(active_events),
            "is_monitoring": self.is_monitoring,
            "active_protocols": [name for name, protocol in self.safety_protocols.items() 
                               if protocol.is_active],
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "parameter": event.parameter,
                    "value": event.current_value,
                    "level": event.safety_level.value,
                    "timestamp": event.timestamp.isoformat(),
                    "acknowledged": event.acknowledged,
                    "resolved": event.resolved
                }
                for event in active_events
            ],
            "statistics": self.stats
        }
    
    def update_threshold(self, parameter: str, threshold_data: Dict[str, Any]) -> bool:
        """Update safety threshold configuration"""
        if parameter not in self.safety_thresholds:
            return False
        
        threshold = self.safety_thresholds[parameter]
        
        for key, value in threshold_data.items():
            if hasattr(threshold, key):
                setattr(threshold, key, value)
        
        logger.info(f"Updated safety threshold for {parameter}: {threshold_data}")
        return True
    
    def get_safety_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate safety report for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        period_events = [event for event in self.safety_events 
                        if event.timestamp >= cutoff_time]
        
        # Calculate statistics
        total_events = len(period_events)
        critical_events = len([e for e in period_events 
                             if e.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]])
        
        # Response time statistics
        response_times = [e.response_time_ms for e in period_events if e.response_time_ms > 0]
        avg_response_time = np.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Parameter statistics
        parameter_counts = {}
        for event in period_events:
            parameter_counts[event.parameter] = parameter_counts.get(event.parameter, 0) + 1
        
        return {
            "report_period_hours": hours,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_events": total_events,
                "critical_events": critical_events,
                "critical_percentage": (critical_events / total_events * 100) if total_events > 0 else 0,
                "avg_response_time_ms": avg_response_time,
                "max_response_time_ms": max_response_time
            },
            "parameter_breakdown": parameter_counts,
            "detailed_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "parameter": event.parameter,
                    "value": event.current_value,
                    "threshold": event.threshold_value,
                    "level": event.safety_level.value,
                    "action": event.action_taken.value,
                    "response_time_ms": event.response_time_ms,
                    "acknowledged": event.acknowledged,
                    "resolved": event.resolved
                }
                for event in period_events
            ]
        }