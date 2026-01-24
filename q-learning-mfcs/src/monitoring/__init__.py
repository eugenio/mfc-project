"""MFC Real-time Monitoring System.

A comprehensive monitoring solution for Microbial Fuel Cell (MFC) systems providing:

- Real-time dashboard with live metrics visualization
- Safety monitoring with automated emergency responses
- WebSocket-based data streaming for real-time updates
- REST API for system control and data access
- Historical data analysis and reporting

Dashboard Components:
---------------------
The dashboard system has a clear separation of concerns:

1. **GUI Pages** (gui/pages/dashboard.py):
   - Simple Streamlit page showing system overview metrics
   - Used by navigation_controller.py for main GUI

2. **Live Monitoring Dashboard** (gui/live_monitoring_dashboard.py):
   - Advanced real-time monitoring with Plotly charts
   - Multi-cell comparison, alerts, customizable layouts
   - Use for detailed performance monitoring

3. **Dashboard API** (monitoring/dashboard_api.py):
   - FastAPI REST endpoints for data and control
   - DashboardAPI class for programmatic access
   - Simple/Advanced modes via configuration
   - Models: SimulationConfig, SimulationData, SystemMetrics, ControlCommand

4. **Dashboard Frontend** (monitoring/dashboard_frontend.py):
   - Streamlit client for Dashboard API
   - Secure HTTPS communication with API server

Quick Start:
    # Programmatic access
    >>> from monitoring import DashboardAPI
    >>> api = DashboardAPI(mode='simple')
    >>> metrics = api.get_system_metrics()

    # Start all monitoring services
    >>> python start_monitoring.py start

Access Points:
- Dashboard UI: http://localhost:8501
- API Documentation: http://localhost:8000/api/docs
- WebSocket Stream: ws://localhost:8001/ws
- Health Check: http://localhost:8000/api/health
"""

from .api_models import AlertMessage, SystemStatus
from .dashboard_api import ControlCommand, DashboardAPI, SystemMetrics
from .dashboard_api import app as dashboard_app
from .observability_manager import (
    Alert,
    AlertCondition,
    AlertSeverity,
    HealthStatus,
    ObservabilityManager,
    ServiceHealth,
    get_default_manager,
)
from .realtime_streamer import DataStreamManager
from .safety_monitor import (
    EmergencyAction,
    SafetyLevel,
    SafetyMonitor,
    SafetyThreshold,
)

__version__ = "1.0.0"
__author__ = "MFC Development Team"

__all__ = [
    "Alert",
    "AlertCondition",
    "AlertMessage",
    "AlertSeverity",
    "ControlCommand",
    "DashboardAPI",
    "DataStreamManager",
    "EmergencyAction",
    "HealthStatus",
    "ObservabilityManager",
    "SafetyLevel",
    "SafetyMonitor",
    "SafetyThreshold",
    "ServiceHealth",
    "SystemMetrics",
    "SystemStatus",
    "dashboard_app",
    "get_default_manager",
]
