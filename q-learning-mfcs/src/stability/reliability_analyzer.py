"""
Reliability Metrics and Failure Analysis System

Advanced reliability engineering tools for MFC systems including:
- Failure mode and effects analysis (FMEA)
- Reliability block diagrams
- Weibull analysis for component lifetimes
- Fault tree analysis
- Reliability prediction models

Author: MFC Development Team
Date: 2025-07-28
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import scipy.stats as stats
from scipy.optimize import minimize
import warnings

class FailureMode(Enum):
    """Types of failure modes."""
    CATASTROPHIC = "catastrophic"      # Complete system failure
    DEGRADED = "degraded"             # Performance degradation
    INTERMITTENT = "intermittent"     # Sporadic failures
    WEAR_OUT = "wear_out"             # Age-related failure
    RANDOM = "random"                 # Random failures
    INFANT_MORTALITY = "infant_mortality"  # Early life failures
class SeverityLevel(Enum):
    """Failure severity levels."""
    NEGLIGIBLE = 1
    MARGINAL = 2
    CRITICAL = 3
    CATASTROPHIC = 4

class DetectionLevel(Enum):
    """Failure detection levels."""
    VERY_HIGH = 1    # Almost certain detection
    HIGH = 2         # High detection probability
    MODERATE = 3     # Moderate detection
    LOW = 4          # Low detection probability
    VERY_LOW = 5     # Very unlikely to detect
class ComponentReliability:
    """Reliability data for a component."""
    component_id: str
    mtbf_hours: float                    # Mean time between failures
    failure_rate: float                  # Failures per hour (λ)
    availability: float                  # Fraction of uptime
    maintainability: float               # Mean time to repair (hours)
    confidence_interval: Tuple[float, float]  # 95% CI for MTBF
    weibull_shape: float = 1.0          # Weibull shape parameter (β)
    weibull_scale: float = 1000.0       # Weibull scale parameter (η)
    operating_hours: float = 0.0        # Total operating time
    failure_count: int = 0              # Number of observed failures
    last_failure: Optional[datetime] = None
class FailureModeEffect:
    """FMEA entry for a failure mode."""
    component: str
    failure_mode: str
    failure_cause: str
    local_effect: str
    system_effect: str
    detection_method: str
    occurrence_rating: int              # 1-10 scale
    severity_rating: int                # 1-10 scale  
    detection_rating: int               # 1-10 scale
    rpn: int = 0                       # Risk Priority Number
    recommended_actions: List[str] = field(default_factory=list)
    current_controls: List[str] = field(default_factory=list)
class ReliabilityPrediction:
    """Reliability prediction results."""
    system_mtbf_hours: float
    system_failure_rate: float
    mission_reliability: float          # For specific mission duration
    availability: float
    component_contributions: Dict[str, float]  # Contribution to system failure rate
    critical_components: List[str]      # Components with highest failure rates
    maintenance_requirements: Dict[str, float]  # Maintenance intervals
    confidence_bounds: Dict[str, Tuple[float, float]]
