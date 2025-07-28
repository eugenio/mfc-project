"""
Real-time Data Streaming Service for MFC Monitoring

Provides WebSocket-based real-time data streaming, event handling,
and distributed monitoring capabilities for MFC systems.
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
import queue
import numpy as np
from pathlib import Path
import sys
import os

from config.real_time_processing import (
from integrated_mfc_model import IntegratedMFCModel
class StreamEventType(Enum):
    """Types of streaming events"""
    METRICS_UPDATE = "metrics_update"
    ALERT = "alert"
    STATUS_CHANGE = "status_change"
    COMMAND_RESULT = "command_result"
    SENSOR_DATA = "sensor_data"
    CONTROL_UPDATE = "control_update"

class StreamEvent:
    """Stream event data structure"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 1  # 1=highest, 10=lowest
    client_filter: Optional[List[str]] = None  # Specific clients
