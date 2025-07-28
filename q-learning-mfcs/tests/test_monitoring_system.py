"""
Test suite for MFC Real-time Monitoring System

Comprehensive tests for dashboard API, safety monitoring, and real-time streaming.
"""
import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import sys
import os
from pathlib import Path

from monitoring.dashboard_api import app, SystemStatus, SystemMetrics, AlertMessage
from monitoring.safety_monitor import (
from monitoring.realtime_streamer import (