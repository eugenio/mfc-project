"""
Comprehensive Tests for MFC Stability Analysis System

Tests for all components of the long-term stability analysis framework:
- StabilityFramework
- ReliabilityAnalyzer  
- DegradationDetector
- MaintenanceScheduler
- LongTermDataManager
- StabilityVisualizer

Created: 2025-07-28
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os
