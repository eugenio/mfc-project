"""
Predictive Maintenance Scheduling for MFC Long-term Stability

Intelligent maintenance scheduling system that uses degradation patterns,
reliability analysis, and operational constraints to optimize maintenance
timing and minimize system downtime.

Created: 2025-07-28
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from degradation_detector import DegradationDetector, DegradationPattern, DegradationSeverity
from reliability_analyzer import ReliabilityAnalyzer, ComponentReliability, MaintenanceRecommendation

class MaintenanceType(Enum):
    """Types of maintenance activities."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    ROUTINE = "routine"


class MaintenancePriority(Enum):
    """Priority levels for maintenance tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ComponentStatus(Enum):
    """Component operational status."""
    HEALTHY = "healthy"
    DEGRADING = "degrading"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

