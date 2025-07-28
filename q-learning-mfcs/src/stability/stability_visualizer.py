"""
Stability Reporting and Visualization Tools for MFC Long-term Analysis

Comprehensive visualization and reporting system for MFC stability studies.
Creates interactive dashboards, reports, and visualizations for degradation
patterns, reliability metrics, and maintenance scheduling.

Created: 2025-07-28
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
