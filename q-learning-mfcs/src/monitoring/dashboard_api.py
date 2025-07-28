"""
Real-time Monitoring Dashboard API for MFC Systems

This module provides a FastAPI-based REST API for real-time monitoring and control
of MFC (Microbial Fuel Cell) systems. It integrates with the existing simulation
infrastructure to provide comprehensive system monitoring, safety alerts, and 
operational control.

Features:
- Real-time system metrics and performance data
- Safety monitoring with configurable thresholds
- Historical data access and analytics
- Control system interaction
- Multi-user session management
- WebSocket support for real-time updates
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import threading
from enum import Enum
import os
import sys
from dataclasses import asdict
from pathlib import Path

from config.real_time_processing import (
from controller_models.real_time_controller import (
from integrated_mfc_model import IntegratedMFCModel, IntegratedMFCState
from path_config import get_simulation_data_path, get_model_path

    import uvicorn
class SystemStatus(str, Enum):
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
class SystemMetrics(BaseModel):
    """Current system metrics"""
    timestamp: datetime
    status: SystemStatus
    uptime_hours: float
    
    # Performance metrics
    total_energy_produced_kwh: float
    average_power_w: float
    coulombic_efficiency_pct: float
    current_density_ma_cm2: float
    
    # System health
    temperature_c: float
    ph_level: float
    pressure_bar: float
    flow_rate_ml_min: float
    
    # Individual cell data
    cell_voltages: List[float]
    cell_currents: List[float]
    biofilm_thickness: List[float]
    substrate_concentrations: List[float]
    
    # Control system
    controller_mode: str
    learning_progress_pct: float
    epsilon_value: float
    