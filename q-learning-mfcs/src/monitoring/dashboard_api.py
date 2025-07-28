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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MFC Real-time Monitoring API",
    description="REST API for MFC system monitoring and control",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
def collect_current_metrics() -> Dict[str, Any]:
    """Collect current system metrics"""
    try:
        model = dashboard_state["mfc_model"]
        controller = dashboard_state["controller"]
        
        if not model or not hasattr(model, 'get_current_state'):
            return {}
            
        # Get current MFC state
        state = model.get_current_state()
        
        # Get controller measurements
        controller_metrics = controller.get_current_measurement() if controller else None
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": SystemStatus.RUNNING if dashboard_state["simulation_running"] else SystemStatus.OFFLINE,
            "uptime_hours": getattr(model, 'simulation_time', 0.0),
            
            # Performance
            "total_energy_produced_kwh": getattr(state, 'total_energy', 0.0) / 3600.0,  # Convert J to kWh
            "average_power_w": getattr(state, 'average_power', 0.0),
            "coulombic_efficiency_pct": getattr(state, 'coulombic_efficiency', 0.0) * 100,
            "current_density_ma_cm2": np.mean(getattr(state, 'current_densities', [0.0])) * 1000,
            
            # Environmental
            "temperature_c": 25.0 + np.random.normal(0, 1),  # Simulated sensor reading
            "ph_level": 7.0 + np.random.normal(0, 0.2),
            "pressure_bar": 1.0 + np.random.normal(0, 0.05),
            "flow_rate_ml_min": getattr(state, 'flow_rate', 100.0),
            
            # Cell data
            "cell_voltages": getattr(state, 'cell_voltages', []),
            "cell_currents": getattr(state, 'current_densities', []),
            "biofilm_thickness": getattr(state, 'biofilm_thickness', []),
            "substrate_concentrations": getattr(state, 'substrate_concentration', []),
            
            # Control system
            "controller_mode": controller_metrics.mode.value if controller_metrics else "manual",
            "learning_progress_pct": getattr(state, 'learning_progress', 0.0) * 100,
            "epsilon_value": getattr(state, 'epsilon', 0.1),
            
            # System resources
            "cpu_utilization_pct": controller_metrics.cpu_utilization_pct if controller_metrics else 0.0,
            "memory_usage_mb": controller_metrics.memory_usage_mb if controller_metrics else 0.0
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        return {}
def check_safety_conditions(metrics: Dict[str, Any]) -> List[AlertMessage]:
    """Check safety conditions and generate alerts"""
    alerts = []
    thresholds = dashboard_state["safety_thresholds"]
    
    try:
        # Temperature check
        if metrics.get("temperature_c", 0) > thresholds["max_temperature"]:
            alerts.append(AlertMessage(
                id=f"temp_alert_{datetime.now().timestamp()}",
                level=AlertLevel.CRITICAL,
                timestamp=datetime.now(),
                message=f"Temperature {metrics['temperature_c']:.1f}°C exceeds maximum {thresholds['max_temperature']}°C",
                category="temperature",
                auto_action="reduce_power"
            ))
        
        # Current density check
        if metrics.get("current_density_ma_cm2", 0) > thresholds["max_current_density"]:
            alerts.append(AlertMessage(
                id=f"current_alert_{datetime.now().timestamp()}",
                level=AlertLevel.WARNING,
                timestamp=datetime.now(),
                message=f"Current density {metrics['current_density_ma_cm2']:.1f} mA/cm² is high",
                category="electrical"
            ))
        
        # Voltage check (low voltage warning)
        cell_voltages = metrics.get("cell_voltages", [])
        if cell_voltages and min(cell_voltages) < thresholds["min_voltage"]:
            alerts.append(AlertMessage(
                id=f"voltage_alert_{datetime.now().timestamp()}",
                level=AlertLevel.WARNING,
                timestamp=datetime.now(),
                message=f"Cell voltage {min(cell_voltages):.2f}V below minimum {thresholds['min_voltage']}V",
                category="electrical"
            ))
        
        # pH check
        ph = metrics.get("ph_level", 7.0)
        if abs(ph - 7.0) > thresholds["max_ph_deviation"]:
            alerts.append(AlertMessage(
                id=f"ph_alert_{datetime.now().timestamp()}",
                level=AlertLevel.WARNING,
                timestamp=datetime.now(),
                message=f"pH {ph:.1f} deviates significantly from neutral",
                category="chemical"
            ))
        
    except Exception as e:
        logger.error(f"Error checking safety conditions: {e}")
    
    return alerts

async def broadcast_alerts(alerts: List[AlertMessage]):
    """Broadcast alerts to all connected clients"""
    for alert in alerts:
        await manager.broadcast({
            "type": "alert",
            "data": alert.dict(),
            "timestamp": datetime.now().isoformat()
        })

# API Routes
