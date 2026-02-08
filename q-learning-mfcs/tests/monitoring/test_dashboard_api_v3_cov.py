"""Coverage tests for dashboard_api.py - lines 78-93, 390-807."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'monitoring'),
)

# Mock SSL dependencies before import
mock_ssl = MagicMock()
with patch.dict(sys.modules, {
    'monitoring.ssl_config': mock_ssl,
    'ssl_config': mock_ssl,
}):
    import dashboard_api as _da_mod
    from dashboard_api import (
        DashboardAPI,
        SimulationConfig,
        SimulationStatus,
        AlertConfig,
        add_security_headers,
        app,
    )


class TestDashboardAPIClass:
    def test_init_default(self):
        api = DashboardAPI()
        assert api.mode == "simple"
        assert not api._is_running

    def test_init_advanced(self):
        api = DashboardAPI(mode="advanced")
        assert api.mode == "advanced"

    def test_get_system_metrics(self):
        api = DashboardAPI()
        m = api.get_system_metrics()
        assert m.status == "idle"
        assert m.temperature_c == 30.0

    def test_get_system_metrics_running(self):
        api = DashboardAPI()
        api._is_running = True
        m = api.get_system_metrics()
        assert m.status == "running"

    def test_send_control_start(self):
        api = DashboardAPI()
        r = api.send_control_command("start")
        assert r["status"] == "success"
        assert api._is_running

    def test_send_control_stop(self):
        api = DashboardAPI()
        api._is_running = True
        r = api.send_control_command("stop")
        assert r["status"] == "success"
        assert not api._is_running

    def test_send_control_other(self):
        api = DashboardAPI()
        r = api.send_control_command("adjust", {"param": 1})
        assert r["status"] == "success"

    def test_get_simulation_status(self):
        api = DashboardAPI()
        s = api.get_simulation_status()
        assert isinstance(s, SimulationStatus)
        assert not s.is_running

    def test_start_simulation(self):
        api = DashboardAPI()
        r = api.start_simulation(duration_hours=10.0, n_cells=3)
        assert r["status"] == "started"
        assert api._is_running
        assert api._current_config is not None

    def test_stop_simulation(self):
        api = DashboardAPI()
        api._is_running = True
        r = api.stop_simulation()
        assert r["status"] == "stopped"
        assert not api._is_running

    def test_get_performance_metrics_no_files(self):
        api = DashboardAPI()
        with patch('pathlib.Path.glob', return_value=[]):
            result = api.get_performance_metrics()
        assert result is None

    def test_get_performance_metrics_with_file(self):
        api = DashboardAPI()
        mock_file = MagicMock()
        mock_file.stat.return_value = MagicMock(st_mtime=1000)
        with patch('pathlib.Path.glob', return_value=[mock_file]), \
             patch('builtins.open', MagicMock()), \
             patch('json.load', return_value={
                 'performance_metrics': {
                     'final_reservoir_concentration': 25.0,
                     'control_effectiveness_2mM': 0.8,
                     'mean_power': 0.5,
                     'total_substrate_added': 100.0,
                 },
             }):
            result = api.get_performance_metrics()
        assert result is not None
        assert result.mean_power == 0.5

    def test_get_performance_metrics_exception(self):
        api = DashboardAPI()
        with patch('pathlib.Path.glob', side_effect=Exception("err")):
            result = api.get_performance_metrics()
        assert result is None


class TestSecurityHeaders:
    def test_add_security_headers_no_config(self):
        req = MagicMock()
        resp = MagicMock()
        resp.headers = {}
        old_val = _da_mod.ssl_config
        try:
            _da_mod.ssl_config = None
            result = add_security_headers(req, resp)
            assert result is resp
        finally:
            _da_mod.ssl_config = old_val

    def test_add_security_headers_with_config(self):
        req = MagicMock()
        resp = MagicMock()
        resp.headers = {}
        mock_cfg = MagicMock()
        mock_ssl.SecurityHeaders.get_security_headers.return_value = {
            "X-Test": "value",
        }
        old_val = _da_mod.ssl_config
        try:
            _da_mod.ssl_config = mock_cfg
            result = add_security_headers(req, resp)
            assert resp.headers["X-Test"] == "value"
        finally:
            _da_mod.ssl_config = old_val


@pytest.mark.asyncio
class TestFastAPIEndpoints:
    async def test_root(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "MFC Monitoring Dashboard API"

    async def test_health(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_simulation_status(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/simulation/status")
        assert resp.status_code == 200

    async def test_alert_config_get(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/alerts/config")
        assert resp.status_code == 200

    async def test_system_info(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.get("/system/info")
        assert resp.status_code == 200

    async def test_stop_simulation_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.post("/simulation/stop")
        assert resp.status_code == 200

    async def test_start_simulation_endpoint(self):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost",
        ) as client:
            resp = await client.post(
                "/simulation/start",
                json={
                    "duration_hours": 10.0,
                    "n_cells": 3,
                    "electrode_area_m2": 0.001,
                    "target_concentration": 25.0,
                },
            )
        assert resp.status_code == 200


class TestRunDashboardAPI:
    def test_run_with_ssl(self):
        mock_cfg = MagicMock()
        mock_cfg.https_port_api = 8443
        mock_cfg.cert_file = "/tmp/cert.pem"
        mock_cfg.key_file = "/tmp/key.pem"
        mock_ssl.load_ssl_config.return_value = mock_cfg
        mock_ssl.SSLContextManager.return_value = MagicMock()

        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(_da_mod, 'uvicorn') as mock_uv:
            _da_mod.run_dashboard_api(ssl_config_override=mock_cfg)
            mock_uv.run.assert_called_once()

    def test_run_without_ssl(self):
        mock_cfg = MagicMock()
        mock_cfg.https_port_api = 8443
        mock_cfg.cert_file = "/tmp/cert.pem"
        mock_cfg.key_file = "/tmp/key.pem"
        mock_ssl.load_ssl_config.return_value = mock_cfg

        with patch('pathlib.Path.exists', return_value=False), \
             patch.object(_da_mod, 'uvicorn') as mock_uv:
            _da_mod.run_dashboard_api(ssl_config_override=mock_cfg)
            mock_uv.run.assert_called_once()
