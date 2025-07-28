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
