"""
MFC Real-time Monitoring System Startup Script

Starts all monitoring system components:
- Dashboard API server
- Real-time data streaming service  
- Safety monitoring system
- Frontend dashboard
"""
import asyncio
import subprocess
import sys
import os
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp

MONITORING_DIR = Path(__file__).parent
SRC_DIR = MONITORING_DIR.parent
PROJECT_DIR = SRC_DIR.parent