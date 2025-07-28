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
