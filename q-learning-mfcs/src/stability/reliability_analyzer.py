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
