"""
MFC Real-time Monitoring System

A comprehensive monitoring solution for Microbial Fuel Cell (MFC) systems providing:

- Real-time dashboard with live metrics visualization
- Safety monitoring with automated emergency responses
- WebSocket-based data streaming for real-time updates
- REST API for system control and data access
- Historical data analysis and reporting

Components:
- dashboard_api: FastAPI-based REST API server
- dashboard_frontend: Streamlit-based web dashboard
- realtime_streamer: WebSocket streaming service
- safety_monitor: Safety monitoring and emergency response system
- start_monitoring: System startup and management script

Quick Start:
    >>> from monitoring import start_monitoring
    >>> # Start all monitoring services
    >>> python start_monitoring.py start

Access Points:
- Dashboard UI: http://localhost:8501
- API Documentation: http://localhost:8000/api/docs
- WebSocket Stream: ws://localhost:8001/ws
- Health Check: http://localhost:8000/api/health
"""

from .api_models import AlertMessage, SystemMetrics, SystemStatus
from .dashboard_api import app as dashboard_app
from .realtime_streamer import DataStreamManager
from .safety_monitor import EmergencyAction, SafetyLevel, SafetyMonitor, SafetyThreshold

__version__ = "1.0.0"
__author__ = "MFC Development Team"

__all__ = [
    "dashboard_app",
    "SystemStatus",
    "SystemMetrics",
    "AlertMessage",
    "SafetyMonitor",
    "SafetyLevel",
    "EmergencyAction",
    "SafetyThreshold",
    "RealTimeStreamer",
    "StreamEventType",
    "StreamEvent"
]
