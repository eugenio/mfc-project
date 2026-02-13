#!/usr/bin/env python3
"""Unified Dashboard API for MFC Monitoring System.

This module consolidates all dashboard API functionality including:
- FastAPI REST endpoints for simulation data, control, and monitoring
- Simple API models (SystemMetrics, ControlCommand) for basic monitoring
- Full simulation models (SimulationConfig, SimulationData) for advanced features
- HTTPS/SSL support for secure communication

Usage Modes:
- Simple Mode: Basic metrics and control (SystemMetrics, ControlCommand endpoints)
- Advanced Mode: Full simulation control with Q-learning parameters

Components:
- DashboardAPI: High-level wrapper class for programmatic access
- FastAPI app: REST API server with full endpoint set
- Pydantic models: Data validation for all API interactions

Quick Start:
    # Simple programmatic access
    >>> from monitoring.dashboard_api import DashboardAPI
    >>> api = DashboardAPI()
    >>> metrics = api.get_system_metrics()

    # Run as API server
    $ python dashboard_api.py --port 8000

For more details see the endpoint documentation at /docs when running the server.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from monitoring.ssl_config import (
    SecurityHeaders,
    SSLConfig,
    SSLContextManager,
    initialize_ssl_infrastructure,
    load_ssl_config,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
ssl_config: SSLConfig | None = None
ssl_context_manager: SSLContextManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for SSL initialization."""
    global ssl_config, ssl_context_manager

    logger.info("Initializing MFC Dashboard API with SSL support...")

    # Load SSL configuration
    ssl_config = load_ssl_config()
    ssl_context_manager = SSLContextManager(ssl_config)

    # Initialize SSL infrastructure if needed
    success, ssl_config = initialize_ssl_infrastructure(ssl_config)
    if not success:
        logger.error("Failed to initialize SSL infrastructure")
        # Continue with HTTP for development

    logger.info(f"Dashboard API starting on port {ssl_config.https_port_api}")
    yield

    logger.info("Shutting down MFC Dashboard API...")


# Initialize FastAPI app with SSL context manager
app = FastAPI(
    title="MFC Monitoring Dashboard API",
    description="REST API for Microbial Fuel Cell monitoring and control with HTTPS security",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models for API
class SimulationStatus(BaseModel):
    """Simulation status response model."""

    is_running: bool
    start_time: datetime | None = None
    duration_hours: float | None = None
    current_time_hours: float | None = None
    progress_percent: float | None = None
    output_directory: str | None = None


class SimulationConfig(BaseModel):
    """Simulation configuration model."""

    duration_hours: float = Field(gt=0, description="Simulation duration in hours")
    n_cells: int = Field(ge=1, le=20, description="Number of MFC cells")
    electrode_area_m2: float = Field(gt=0, description="Electrode area per cell in m²")
    target_concentration: float = Field(
        gt=0,
        description="Target substrate concentration in mM",
    )
    use_pretrained: bool = Field(default=True, description="Use pre-trained Q-table")
    learning_rate: float | None = Field(default=0.1, ge=0.01, le=1.0)
    epsilon_initial: float | None = Field(default=0.4, ge=0.1, le=1.0)
    discount_factor: float | None = Field(default=0.95, ge=0.8, le=0.99)


class SimulationData(BaseModel):
    """Simulation data response model."""

    timestamp: datetime
    time_hours: float
    reservoir_concentration: float
    outlet_concentration: float
    total_power: float
    biofilm_thicknesses: list[float]
    substrate_addition_rate: float
    q_action: int
    epsilon: float
    reward: float


class PerformanceMetrics(BaseModel):
    """Performance metrics model."""

    final_reservoir_concentration: float
    control_effectiveness_2mM: float
    mean_power: float
    total_substrate_added: float
    energy_efficiency: float | None = None
    stability_score: float | None = None


class AlertConfig(BaseModel):
    """Alert configuration model."""

    parameter: str
    threshold_min: float | None = None
    threshold_max: float | None = None
    enabled: bool = True
    email_notify: bool = False


# Simple API models (consolidated from simple_dashboard_api.py)
class SystemMetrics(BaseModel):
    """System metrics response for simple monitoring mode.

    Provides essential MFC system metrics without full simulation detail.
    Use this model for basic monitoring dashboards or quick status checks.
    """

    timestamp: str
    status: str
    uptime_hours: float
    power_output_w: float
    efficiency_pct: float
    temperature_c: float
    ph_level: float
    pressure_bar: float
    flow_rate_ml_min: float
    cell_voltages: list[float]


class ControlCommand(BaseModel):
    """Control command request for system control.

    Simple command interface for sending control commands to MFC systems.
    Use this for basic operations like start/stop, parameter adjustments.
    """

    command: str
    parameters: dict[str, Any] | None = None


class DashboardAPI:
    """High-level wrapper for programmatic dashboard API access.

    This class provides a convenient interface for integrating dashboard
    functionality into other Python modules without running the full
    FastAPI server.

    Modes:
        - simple: Basic metrics and status (default)
        - advanced: Full simulation control with Q-learning parameters

    Example:
        >>> api = DashboardAPI(mode='simple')
        >>> metrics = api.get_system_metrics()
        >>> api.send_control_command('start', {'duration': 24})

        >>> api = DashboardAPI(mode='advanced')
        >>> status = api.get_simulation_status()
        >>> api.start_simulation(duration_hours=100, n_cells=5)
    """

    def __init__(self, config: dict[str, Any] | None = None, mode: str = "simple"):
        """Initialize DashboardAPI.

        Args:
            config: Optional configuration dictionary with keys:
                - data_dir: Path to simulation data directory
                - api_token: Authentication token for API access
                - ssl_enabled: Whether SSL is enabled
            mode: Operation mode - 'simple' for basic metrics, 'advanced' for full control
        """
        self.config = config or {}
        self.mode = mode
        self._data_dir = Path(
            self.config.get("data_dir", "../../../data/simulation_data")
        )
        self._is_running = False
        self._start_time: datetime | None = None
        self._current_config: SimulationConfig | None = None
        logger.info(f"DashboardAPI initialized in {mode} mode")

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics (simple mode).

        Returns:
            SystemMetrics with current MFC system state
        """
        # Return mock metrics for now - in production, connect to actual sensors
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            status="running" if self._is_running else "idle",
            uptime_hours=0.0,
            power_output_w=0.0,
            efficiency_pct=0.0,
            temperature_c=30.0,
            ph_level=7.0,
            pressure_bar=1.0,
            flow_rate_ml_min=0.0,
            cell_voltages=[],
        )

    def send_control_command(
        self, command: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a control command to the system.

        Args:
            command: Command name (e.g., 'start', 'stop', 'adjust')
            parameters: Optional command parameters

        Returns:
            Response dict with status and message
        """
        cmd = ControlCommand(command=command, parameters=parameters)
        logger.info(f"Executing control command: {cmd.command}")

        if cmd.command == "start":
            self._is_running = True
            self._start_time = datetime.now()
            return {"status": "success", "message": "System started"}
        elif cmd.command == "stop":
            self._is_running = False
            self._start_time = None
            return {"status": "success", "message": "System stopped"}
        else:
            return {"status": "success", "message": f"Command '{cmd.command}' executed"}

    def get_simulation_status(self) -> SimulationStatus:
        """Get current simulation status (advanced mode).

        Returns:
            SimulationStatus with detailed simulation state
        """
        return SimulationStatus(
            is_running=self._is_running,
            start_time=self._start_time,
            duration_hours=(
                self._current_config.duration_hours if self._current_config else None
            ),
            current_time_hours=None,
            progress_percent=None,
            output_directory=str(self._data_dir),
        )

    def start_simulation(
        self,
        duration_hours: float,
        n_cells: int = 5,
        electrode_area_m2: float = 0.001,
        target_concentration: float = 25.0,
        use_pretrained: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a new simulation (advanced mode).

        Args:
            duration_hours: Simulation duration in hours
            n_cells: Number of MFC cells
            electrode_area_m2: Electrode area per cell in m²
            target_concentration: Target substrate concentration in mM
            use_pretrained: Whether to use pre-trained Q-table
            **kwargs: Additional parameters (learning_rate, epsilon_initial, etc.)

        Returns:
            Response dict with status and configuration
        """
        self._current_config = SimulationConfig(
            duration_hours=duration_hours,
            n_cells=n_cells,
            electrode_area_m2=electrode_area_m2,
            target_concentration=target_concentration,
            use_pretrained=use_pretrained,
            learning_rate=kwargs.get("learning_rate", 0.1),
            epsilon_initial=kwargs.get("epsilon_initial", 0.4),
            discount_factor=kwargs.get("discount_factor", 0.95),
        )
        self._is_running = True
        self._start_time = datetime.now()

        logger.info(
            f"Starting simulation: {duration_hours}h, {n_cells} cells",
        )
        return {
            "status": "started",
            "config": self._current_config.model_dump(),
            "timestamp": datetime.now().isoformat(),
        }

    def stop_simulation(self) -> dict[str, Any]:
        """Stop the current simulation (advanced mode).

        Returns:
            Response dict with status
        """
        self._is_running = False
        self._start_time = None
        logger.info("Simulation stopped")
        return {"status": "stopped", "timestamp": datetime.now().isoformat()}

    def get_performance_metrics(self) -> PerformanceMetrics | None:
        """Get performance metrics from latest simulation (advanced mode).

        Returns:
            PerformanceMetrics if available, None otherwise
        """
        try:
            results_files = list(self._data_dir.glob("**/gui_simulation_results_*.json"))
            if not results_files:
                return None

            latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file) as f:
                results = json.load(f)

            metrics = results.get("performance_metrics", {})
            return PerformanceMetrics(
                final_reservoir_concentration=metrics.get(
                    "final_reservoir_concentration", 0.0
                ),
                control_effectiveness_2mM=metrics.get("control_effectiveness_2mM", 0.0),
                mean_power=metrics.get("mean_power", 0.0),
                total_substrate_added=metrics.get("total_substrate_added", 0.0),
                energy_efficiency=metrics.get("energy_efficiency"),
                stability_score=metrics.get("stability_score"),
            )
        except Exception as e:
            logger.exception(f"Failed to get performance metrics: {e}")
            return None


# Middleware configuration
def add_security_headers(request: Request, response: Response) -> Response:
    """Add security headers to all responses."""
    if ssl_config:
        security_headers = SecurityHeaders.get_security_headers(ssl_config)
        for header, value in security_headers.items():
            response.headers[header] = value
    return response


# Add CORS middleware with secure settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8444",
        "https://127.0.0.1:8444",
    ],  # Only HTTPS origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.local"],
)


# Add security headers to all responses
@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    return add_security_headers(request, response)


# Authentication (basic bearer token for now)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """Simple bearer token authentication."""
    if credentials is None:
        return None  # Allow unauthenticated access for now

    # In production, verify the token against a database or JWT
    if credentials.credentials == os.getenv("MFC_API_TOKEN", "development-token"):
        return {"username": "api_user"}

    raise HTTPException(status_code=401, detail="Invalid authentication credentials")


# API Routes


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "MFC Monitoring Dashboard API",
        "version": "1.2.0",
        "status": "running",
        "ssl_enabled": ssl_config is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "simulation": "/simulation",
            "data": "/data",
            "metrics": "/metrics",
            "alerts": "/alerts",
            "docs": "/docs",
        },
    }


@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ssl_config": {
            "enabled": ssl_config is not None,
            "port": ssl_config.https_port_api if ssl_config else None,
            "domain": ssl_config.domain if ssl_config else None,
        },
    }


@app.get("/simulation/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Get current simulation status."""
    # This would connect to the actual simulation runner
    # For now, return mock data showing the API structure
    return SimulationStatus(
        is_running=False,
        start_time=None,
        duration_hours=None,
        current_time_hours=None,
        progress_percent=None,
        output_directory=None,
    )


@app.post("/simulation/start")
async def start_simulation(config: SimulationConfig, user=Depends(get_current_user)):
    """Start a new simulation with given configuration."""
    try:
        # Validate configuration
        if config.duration_hours > 8760:  # 1 year max
            raise HTTPException(
                status_code=400,
                detail="Duration exceeds maximum allowed (1 year)",
            )

        # In actual implementation, this would start the simulation process
        logger.info(
            f"Starting simulation: {config.duration_hours}h, {config.n_cells} cells",
        )

        return {
            "status": "started",
            "message": "Simulation started successfully",
            "config": config.dict(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/stop")
async def stop_simulation(user=Depends(get_current_user)):
    """Stop the current simulation."""
    try:
        # In actual implementation, this would stop the simulation process
        logger.info("Stopping simulation")

        return {
            "status": "stopped",
            "message": "Simulation stopped successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to stop simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/latest", response_model=list[SimulationData])
async def get_latest_data(limit: int = 100):
    """Get latest simulation data points."""
    try:
        # Find latest simulation data
        data_dir = Path("../../../data/simulation_data")

        if not data_dir.exists():
            return []

        # Find most recent data file
        data_files = list(data_dir.glob("**/gui_simulation_data_*.csv.gz"))
        if not data_files:
            return []

        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)

        # Load and return data
        with gzip.open(latest_file, "rt") as f:
            df = pd.read_csv(f)

        # Convert to API format
        data_points = []
        for _, row in df.tail(limit).iterrows():
            # Parse biofilm thickness data safely
            try:
                if isinstance(row["biofilm_thicknesses"], str):
                    biofilm_thicknesses = json.loads(
                        row["biofilm_thicknesses"].replace("'", '"'),
                    )
                else:
                    biofilm_thicknesses = [float(row["biofilm_thicknesses"])]
            except Exception:
                biofilm_thicknesses = [10.0]  # Default value

            data_point = SimulationData(
                timestamp=datetime.now(),  # Would use actual timestamp from data
                time_hours=float(row["time_hours"]),
                reservoir_concentration=float(row["reservoir_concentration"]),
                outlet_concentration=float(row["outlet_concentration"]),
                total_power=float(row["total_power"]),
                biofilm_thicknesses=biofilm_thicknesses,
                substrate_addition_rate=float(row["substrate_addition_rate"]),
                q_action=int(row["q_action"]),
                epsilon=float(row["epsilon"]),
                reward=float(row["reward"]),
            )
            data_points.append(data_point)

        return data_points

    except Exception as e:
        logger.exception(f"Failed to get latest data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve simulation data",
        )


@app.get("/data/export/{format}")
async def export_data(format: str, simulation_id: str | None = None):
    """Export simulation data in various formats."""
    if format not in ["csv", "json", "excel", "hdf5"]:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    try:
        # Find and load data
        data_dir = Path("../../../data/simulation_data")

        if simulation_id:
            data_files = list(data_dir.glob(f"**/*{simulation_id}*.csv.gz"))
        else:
            data_files = list(data_dir.glob("**/gui_simulation_data_*.csv.gz"))

        if not data_files:
            raise HTTPException(status_code=404, detail="No simulation data found")

        latest_file = max(data_files, key=lambda f: f.stat().st_mtime)

        with gzip.open(latest_file, "rt") as f:
            df = pd.read_csv(f)

        # Export in requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "csv":
            export_path = f"/tmp/mfc_data_{timestamp}.csv"
            df.to_csv(export_path, index=False)
        elif format == "json":
            export_path = f"/tmp/mfc_data_{timestamp}.json"
            df.to_json(export_path, orient="records", indent=2)
        elif format == "excel":
            export_path = f"/tmp/mfc_data_{timestamp}.xlsx"
            df.to_excel(export_path, index=False)
        elif format == "hdf5":
            export_path = f"/tmp/mfc_data_{timestamp}.h5"
            df.to_hdf(export_path, key="simulation_data", mode="w")

        return FileResponse(
            export_path,
            filename=f"mfc_simulation_data_{timestamp}.{format}",
            media_type="application/octet-stream",
        )

    except Exception as e:
        logger.exception(f"Failed to export data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")


@app.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get current performance metrics."""
    try:
        # Load latest results
        data_dir = Path("../../../data/simulation_data")
        results_files = list(data_dir.glob("**/gui_simulation_results_*.json"))

        if not results_files:
            raise HTTPException(
                status_code=404,
                detail="No performance metrics available",
            )

        latest_file = max(results_files, key=lambda f: f.stat().st_mtime)

        with open(latest_file) as f:
            results = json.load(f)

        metrics = results.get("performance_metrics", {})

        return PerformanceMetrics(
            final_reservoir_concentration=metrics.get(
                "final_reservoir_concentration",
                0.0,
            ),
            control_effectiveness_2mM=metrics.get("control_effectiveness_2mM", 0.0),
            mean_power=metrics.get("mean_power", 0.0),
            total_substrate_added=metrics.get("total_substrate_added", 0.0),
            energy_efficiency=metrics.get("energy_efficiency", None),
            stability_score=metrics.get("stability_score", None),
        )

    except Exception as e:
        logger.exception(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance metrics",
        )


@app.get("/alerts/config")
async def get_alert_config():
    """Get current alert configuration."""
    # Return default alert configuration
    default_alerts = [
        AlertConfig(
            parameter="reservoir_concentration",
            threshold_min=20.0,
            threshold_max=30.0,
            enabled=True,
            email_notify=False,
        ),
        AlertConfig(
            parameter="total_power",
            threshold_min=0.001,
            threshold_max=None,
            enabled=True,
            email_notify=True,
        ),
    ]

    return {"alerts": default_alerts}


@app.post("/alerts/config")
async def update_alert_config(
    alerts: list[AlertConfig],
    user=Depends(get_current_user),
):
    """Update alert configuration."""
    try:
        # In actual implementation, save to database or file
        logger.info(f"Updated alert configuration: {len(alerts)} alerts configured")

        return {
            "status": "updated",
            "message": f"Alert configuration updated with {len(alerts)} alerts",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.exception(f"Failed to update alert config: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update alert configuration",
        )


@app.get("/system/info")
async def get_system_info():
    """Get system information and status."""
    return {
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "api_version": "1.2.0",
            "ssl_enabled": ssl_config is not None,
        },
        "ssl_config": (
            {
                "domain": ssl_config.domain if ssl_config else None,
                "https_port": ssl_config.https_port_api if ssl_config else None,
                "certificate_file": ssl_config.cert_file if ssl_config else None,
                "auto_renewal": ssl_config.auto_renew if ssl_config else None,
            }
            if ssl_config
            else None
        ),
        "data_directories": {
            "simulation_data": str(Path("../../../data/simulation_data").resolve()),
            "q_learning_models": str(Path("../../../q_learning_models").resolve()),
        },
        "timestamp": datetime.now().isoformat(),
    }


def run_dashboard_api(
    host: str = "0.0.0.0",
    port: int | None = None,
    ssl_config_override: SSLConfig | None = None,
    debug: bool = False,
) -> None:
    """Run the FastAPI dashboard with SSL support."""
    # Load SSL configuration
    config = ssl_config_override or load_ssl_config()

    # Use configured port or default
    if port is None:
        port = config.https_port_api

    # SSL context for uvicorn
    if config and Path(config.cert_file).exists() and Path(config.key_file).exists():
        ssl_manager = SSLContextManager(config)
        ssl_params = ssl_manager.get_uvicorn_ssl_config()

        logger.info(f"Starting HTTPS server on {host}:{port}")
        logger.info(f"Certificate: {config.cert_file}")
        logger.info(f"Key: {config.key_file}")

        # Run with SSL
        uvicorn.run(
            "dashboard_api:app",
            host=host,
            port=port,
            reload=debug,
            access_log=True,
            **ssl_params,
        )
    else:
        logger.warning("SSL certificates not found, running HTTP server")
        logger.warning("For production, please setup SSL certificates")

        # Run without SSL (development only)
        uvicorn.run(
            "dashboard_api:app",
            host=host,
            port=8000,  # HTTP port
            reload=debug,
            access_log=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFC Dashboard API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default from SSL config)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--init-ssl",
        action="store_true",
        help="Initialize SSL certificates",
    )

    args = parser.parse_args()

    if args.init_ssl:
        logger.info("Initializing SSL infrastructure...")
        success, config = initialize_ssl_infrastructure()
        if success:
            logger.info("✅ SSL infrastructure initialized")
        else:
            logger.error("❌ SSL initialization failed")
            sys.exit(1)

    run_dashboard_api(host=args.host, port=args.port, debug=args.debug)
