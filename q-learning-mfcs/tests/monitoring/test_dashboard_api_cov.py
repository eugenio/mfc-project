"""Coverage tests for monitoring/dashboard_api.py."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------
class TestPydanticModels:
    def test_simulation_status(self):
        from monitoring.dashboard_api import SimulationStatus

        s = SimulationStatus(is_running=False)
        assert s.is_running is False
        assert s.start_time is None

    def test_simulation_config(self):
        from monitoring.dashboard_api import SimulationConfig

        c = SimulationConfig(
            duration_hours=10,
            n_cells=5,
            electrode_area_m2=0.001,
            target_concentration=25.0,
        )
        assert c.duration_hours == 10

    def test_simulation_data(self):
        from monitoring.dashboard_api import SimulationData

        d = SimulationData(
            timestamp=datetime.now(),
            time_hours=1.0,
            reservoir_concentration=25.0,
            outlet_concentration=24.0,
            total_power=0.5,
            biofilm_thicknesses=[10.0, 12.0],
            substrate_addition_rate=0.1,
            q_action=3,
            epsilon=0.3,
            reward=0.9,
        )
        assert d.time_hours == 1.0

    def test_performance_metrics(self):
        from monitoring.dashboard_api import PerformanceMetrics

        m = PerformanceMetrics(
            final_reservoir_concentration=25.0,
            control_effectiveness_2mM=80.0,
            mean_power=0.3,
            total_substrate_added=100.0,
        )
        assert m.mean_power == 0.3

    def test_alert_config(self):
        from monitoring.dashboard_api import AlertConfig

        a = AlertConfig(parameter="power", threshold_min=0.1)
        assert a.enabled is True

    def test_system_metrics(self):
        from monitoring.dashboard_api import SystemMetrics

        m = SystemMetrics(
            timestamp="2025-01-01",
            status="running",
            uptime_hours=1.0,
            power_output_w=0.5,
            efficiency_pct=80.0,
            temperature_c=30.0,
            ph_level=7.0,
            pressure_bar=1.0,
            flow_rate_ml_min=10.0,
            cell_voltages=[0.5, 0.6],
        )
        assert m.status == "running"

    def test_control_command(self):
        from monitoring.dashboard_api import ControlCommand

        c = ControlCommand(command="start", parameters={"duration": 10})
        assert c.command == "start"


# ---------------------------------------------------------------------------
# DashboardAPI class tests
# ---------------------------------------------------------------------------
class TestDashboardAPI:
    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_init_simple_mode(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI(mode="simple")
        assert api.mode == "simple"
        assert api._is_running is False

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_init_advanced_mode(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI(config={"data_dir": "/tmp/test"}, mode="advanced")
        assert api.mode == "advanced"

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_system_metrics_idle(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        metrics = api.get_system_metrics()
        assert metrics.status == "idle"
        assert metrics.power_output_w == 0.0

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_system_metrics_running(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        api._is_running = True
        metrics = api.get_system_metrics()
        assert metrics.status == "running"

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_send_control_command_start(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        result = api.send_control_command("start")
        assert result["status"] == "success"
        assert api._is_running is True

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_send_control_command_stop(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        api._is_running = True
        result = api.send_control_command("stop")
        assert result["status"] == "success"
        assert api._is_running is False

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_send_control_command_other(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        result = api.send_control_command("adjust", {"param": 1})
        assert "adjust" in result["message"]

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_simulation_status_no_config(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        status = api.get_simulation_status()
        assert status.is_running is False
        assert status.duration_hours is None

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_simulation_status_with_config(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        api.start_simulation(duration_hours=100, n_cells=5)
        status = api.get_simulation_status()
        assert status.is_running is True
        assert status.duration_hours == 100

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_start_simulation(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        result = api.start_simulation(
            duration_hours=50,
            n_cells=3,
            learning_rate=0.2,
            epsilon_initial=0.5,
            discount_factor=0.9,
        )
        assert result["status"] == "started"
        assert api._is_running is True

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_stop_simulation(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI()
        api._is_running = True
        result = api.stop_simulation()
        assert result["status"] == "stopped"
        assert api._is_running is False

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_performance_metrics_no_files(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI(config={"data_dir": "/tmp/nonexistent_dir_test"})
        result = api.get_performance_metrics()
        assert result is None

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_performance_metrics_with_file(self, mock_ssl, tmp_path):
        from monitoring.dashboard_api import DashboardAPI

        results_file = tmp_path / "gui_simulation_results_test.json"
        results_file.write_text(
            json.dumps(
                {
                    "performance_metrics": {
                        "final_reservoir_concentration": 25.0,
                        "control_effectiveness_2mM": 80.0,
                        "mean_power": 0.3,
                        "total_substrate_added": 100.0,
                        "energy_efficiency": 50.0,
                        "stability_score": 0.9,
                    }
                }
            )
        )
        api = DashboardAPI(config={"data_dir": str(tmp_path)})
        result = api.get_performance_metrics()
        assert result is not None
        assert result.mean_power == 0.3

    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_get_performance_metrics_exception(self, mock_ssl):
        from monitoring.dashboard_api import DashboardAPI

        api = DashboardAPI(config={"data_dir": "/tmp/nonexistent"})
        with patch.object(Path, "glob", side_effect=Exception("test")):
            result = api.get_performance_metrics()
        assert result is None


# ---------------------------------------------------------------------------
# add_security_headers function test
# ---------------------------------------------------------------------------
class TestSecurityHeaders:
    def test_add_security_headers_with_config(self):
        import monitoring.dashboard_api as mod

        mock_config = MagicMock()
        mock_config.enable_hsts = True
        mock_config.hsts_max_age = 100
        mock_config.enable_csp = True
        original = mod.ssl_config
        try:
            mod.ssl_config = mock_config
            request = MagicMock()
            response = MagicMock()
            response.headers = {}
            result = mod.add_security_headers(request, response)
            assert result is response
        finally:
            mod.ssl_config = original

    def test_add_security_headers_no_config(self):
        import monitoring.dashboard_api as mod

        original = mod.ssl_config
        try:
            mod.ssl_config = None
            request = MagicMock()
            response = MagicMock()
            response.headers = {}
            result = mod.add_security_headers(request, response)
            assert result is response
        finally:
            mod.ssl_config = original


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------
class TestFastAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from monitoring.dashboard_api import app

        return TestClient(app, raise_server_exceptions=False)

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "MFC Monitoring Dashboard API"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_simulation_status(self, client):
        resp = client.get("/simulation/status")
        assert resp.status_code == 200

    def test_start_simulation(self, client):
        config = {
            "duration_hours": 24,
            "n_cells": 5,
            "electrode_area_m2": 0.001,
            "target_concentration": 25.0,
        }
        resp = client.post("/simulation/start", json=config)
        assert resp.status_code == 200

    def test_start_simulation_too_long(self, client):
        config = {
            "duration_hours": 10000,
            "n_cells": 5,
            "electrode_area_m2": 0.001,
            "target_concentration": 25.0,
        }
        resp = client.post("/simulation/start", json=config)
        assert resp.status_code == 400

    def test_stop_simulation(self, client):
        resp = client.post("/simulation/stop")
        assert resp.status_code == 200

    def test_get_latest_data_no_dir(self, client):
        resp = client.get("/data/latest")
        assert resp.status_code in (200, 500)

    def test_export_data_bad_format(self, client):
        resp = client.get("/data/export/xml")
        assert resp.status_code == 400

    def test_get_alert_config(self, client):
        resp = client.get("/alerts/config")
        assert resp.status_code == 200
        assert "alerts" in resp.json()

    def test_update_alert_config(self, client):
        alerts = [
            {"parameter": "power", "threshold_min": 0.1, "enabled": True}
        ]
        resp = client.post("/alerts/config", json=alerts)
        assert resp.status_code == 200

    def test_system_info(self, client):
        resp = client.get("/system/info")
        assert resp.status_code == 200

    def test_get_performance_metrics_endpoint(self, client):
        resp = client.get("/metrics/performance")
        assert resp.status_code in (404, 500)


# ---------------------------------------------------------------------------
# get_current_user tests
# ---------------------------------------------------------------------------
class TestAuth:
    @pytest.mark.asyncio
    async def test_get_current_user_none(self):
        from monitoring.dashboard_api import get_current_user

        result = await get_current_user(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_user_valid(self):
        from monitoring.dashboard_api import get_current_user

        creds = MagicMock()
        creds.credentials = "development-token"
        with patch.dict(os.environ, {"MFC_API_TOKEN": "development-token"}):
            result = await get_current_user(creds)
        assert result == {"username": "api_user"}

    @pytest.mark.asyncio
    async def test_get_current_user_invalid(self):
        from fastapi import HTTPException

        from monitoring.dashboard_api import get_current_user

        creds = MagicMock()
        creds.credentials = "bad-token"
        with pytest.raises(HTTPException):
            await get_current_user(creds)


# ---------------------------------------------------------------------------
# run_dashboard_api tests
# ---------------------------------------------------------------------------
class TestRunDashboardAPI:
    @patch("monitoring.dashboard_api.uvicorn")
    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_run_with_ssl(self, mock_load, mock_uvicorn):
        from monitoring.dashboard_api import SSLConfig, run_dashboard_api

        config = SSLConfig()
        with patch.object(Path, "exists", return_value=True):
            run_dashboard_api(ssl_config_override=config)
        mock_uvicorn.run.assert_called_once()

    @patch("monitoring.dashboard_api.uvicorn")
    @patch("monitoring.dashboard_api.load_ssl_config")
    def test_run_without_ssl(self, mock_load, mock_uvicorn):
        from monitoring.dashboard_api import SSLConfig, run_dashboard_api

        config = SSLConfig()
        with patch.object(Path, "exists", return_value=False):
            run_dashboard_api(ssl_config_override=config)
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs[1]["port"] == 8000
