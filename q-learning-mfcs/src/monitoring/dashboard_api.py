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
    
class AlertMessage(BaseModel):
    """Safety/operational alert"""
    id: str
    level: AlertLevel
    timestamp: datetime
    message: str
    category: str
    acknowledged: bool = False
    auto_action: Optional[str] = None

class ControlCommand(BaseModel):
    """Control system command"""
    command: str
    parameters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    safety_override: bool = False

class ConfigurationUpdate(BaseModel):
    """System configuration update"""
    section: str
    parameters: Dict[str, Any]
    apply_immediately: bool = True
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if self.active_connections:
            message_str = json.dumps(message, default=str)
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                self.active_connections.remove(conn)
def initialize_monitoring():
    """Initialize the monitoring systems"""
    try:
        # Initialize MFC model (in monitoring mode)
        dashboard_state["mfc_model"] = IntegratedMFCModel(
            n_cells=5,
            species="mixed",
            substrate="lactate",
            use_gpu=True,
            simulation_hours=1000
        )
        
        # Initialize real-time controller
        dashboard_state["controller"] = RealTimeController(
            control_loop_period_ms=100.0,
            max_jitter_ms=5.0
        )
        
        # Initialize data streaming
        dashboard_state["data_stream"] = MFCDataStream(
            buffer_size=10000,
            sampling_rate_hz=10.0
        )
        
        # Initialize alert system
        dashboard_state["alert_system"] = AlertSystem(
            alert_history_size=1000
        )
        
        logger.info("Monitoring systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring systems: {e}")
        return False

# Background task for data collection
async def data_collection_task():
    """Background task for continuous data collection"""
    while True:
        try:
            if dashboard_state["simulation_running"] and dashboard_state["mfc_model"]:
                # Collect current metrics
                metrics = collect_current_metrics()
                dashboard_state["current_metrics"] = metrics
                
                # Check for alerts
                alerts = check_safety_conditions(metrics)
                if alerts:
                    await broadcast_alerts(alerts)
                
                # Broadcast metrics to connected clients
                await manager.broadcast({
                    "type": "metrics_update",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error in data collection task: {e}")
        
        await asyncio.sleep(1.0)  # Update every second
