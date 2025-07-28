"""
Long-term Data Storage and Analysis for MFC Stability Studies

Comprehensive data management system for storing, indexing, and analyzing
long-term MFC operational data. Provides efficient storage, querying, and
analysis capabilities for stability studies and degradation tracking.

Created: 2025-07-28
"""
import numpy as np
import pandas as pd
import sqlite3
import h5py
import json
import gzip
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from contextlib import contextmanager
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

from degradation_detector import DegradationPattern, DegradationType
from reliability_analyzer import ComponentReliability


class DataType(Enum):
    """Types of data stored in the system."""
    SENSOR_DATA = "sensor_data"
    SYSTEM_METRICS = "system_metrics"
    PERFORMANCE_DATA = "performance_data"
    DEGRADATION_PATTERNS = "degradation_patterns"
    MAINTENANCE_RECORDS = "maintenance_records"
    RELIABILITY_METRICS = "reliability_metrics"
    ENVIRONMENTAL_DATA = "environmental_data"
    OPERATIONAL_LOGS = "operational_logs"

