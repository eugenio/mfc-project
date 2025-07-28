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
