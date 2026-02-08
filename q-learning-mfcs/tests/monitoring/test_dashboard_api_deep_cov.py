"""Coverage tests for monitoring.dashboard_api module."""
import json
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from monitoring.dashboard_api import (
    AlertConfig,
    ControlCommand,
    DashboardAPI,
    PerformanceMetrics,
    SimulationConfig,
    SimulationData,
    SimulationStatus,
    SystemMetrics,
    add_security_headers,
    app,
)


class TestPydanticModels:
    def test_simulation_status(self):
        s = SimulationStatus(is_running=False)
        assert s.is_running is False

    def test_simulation_config(self):
        c = SimulationConfig(
            duration_hours=24.0, n_cells=5,
            electrode_area_m2=0.001, target_concentration=25.0,
        )
        assert c.n_cells == 5

    def test_simulation_data(self):
        d = SimulationData(
            timestamp=datetime.now(), time_hours=1.0,
            reservoir_concentration=25.0, outlet_concentration=20.0,
            total_power=0.5, biofilm_thicknesses=[10.0],
            substrate_addition_rate=0.1, q_action=2, epsilon=0.3, reward=1.5,
        )
        assert d.time_hours == 1.0

    def test_performance_metrics(self):
        m = PerformanceMetrics(
            final_reservoir_concentration=25.0,
            control_effectiveness_2mM=0.8,
            mean_power=0.5, total_substrate_added=100.0,
        )
        assert m.mean_power == 0.5

    def test_alert_config(self):
        a = AlertConfig(parameter="power", threshold_min=0.001)
        assert a.enabled is True

    def test_system_metrics(self):
        m = SystemMetrics(
            timestamp="now", status="idle", uptime_hours=0.0,
            power_output_w=0.0, efficiency_pct=0.0, temperature_c=30.0,
            ph_level=7.0, pressure_bar=1.0, flow_rate_ml_min=0.0,
            cell_voltages=[],
        )
        assert m.status == "idle"

    def test_control_command(self):
        c = ControlCommand(command="start", parameters={"d": 24})
        assert c.command == "start"


class TestDashboardAPI:
    def test_init_simple(self):
        api = DashboardAPI(mode="simple")
        assert api.mode == "simple"

    def test_init_advanced(self):
        api = DashboardAPI(mode="advanced")
        assert api.mode == "advanced"

    def test_get_system_metrics_idle(self):
        api = DashboardAPI()
        m = api.get_system_metrics()
        assert isinstance(m, SystemMetrics)
        assert m.status == "idle"

    def test_get_system_metrics_running(self):
        api = DashboardAPI()
        api._is_running = True
        m = api.get_system_metrics()
        assert m.status == "running"

    def test_send_control_start(self):
        api = DashboardAPI()
        r = api.send_control_command("start")
        assert r["status"] == "success"
        assert api._is_running is True

    def test_send_control_stop(self):
        api = DashboardAPI()
        api._is_running = True
        r = api.send_control_command("stop")
        assert api._is_running is False

    def test_send_control_other(self):
        api = DashboardAPI()
        r = api.send_control_command("adjust", {"p": 1})
        assert r["status"] == "success"

    def test_get_simulation_status(self):
        api = DashboardAPI()
        s = api.get_simulation_status()
        assert isinstance(s, SimulationStatus)

    def test_start_simulation(self):
        api = DashboardAPI()
        r = api.start_simulation(duration_hours=24.0)
        assert r["status"] == "started"
        assert api._is_running is True

    def test_start_simulation_custom(self):
        api = DashboardAPI()
        r = api.start_simulation(
            duration_hours=100, n_cells=10,
            electrode_area_m2=0.002, target_concentration=30.0,
            learning_rate=0.2, epsilon_initial=0.5, discount_factor=0.9,
        )
        assert r["status"] == "started"

    def test_stop_simulation(self):
        api = DashboardAPI()
        api._is_running = True
        r = api.stop_simulation()
        assert r["status"] == "stopped"

    def test_get_performance_metrics_none(self, tmp_path):
        api = DashboardAPI()
        api._data_dir = tmp_path
        assert api.get_performance_metrics() is None

    def test_get_performance_metrics_ok(self, tmp_path):
        api = DashboardAPI()
        api._data_dir = tmp_path
        f = tmp_path / "gui_simulation_results_test.json"
        f.write_text(json.dumps({
            "performance_metrics": {
                "final_reservoir_concentration": 25.0,
                "control_effectiveness_2mM": 0.8,
                "mean_power": 0.5,
                "total_substrate_added": 100.0,
            }
        }))
        r = api.get_performance_metrics()
        assert r is not None
        assert r.mean_power == 0.5

    def test_get_sim_status_with_config(self):
        api = DashboardAPI()
        api.start_simulation(duration_hours=50.0)
        s = api.get_simulation_status()
        assert s.duration_hours == 50.0


class TestSecurityHeaders:
    def test_no_ssl(self):
        import monitoring.dashboard_api as mod
        orig = mod.ssl_config
        mod.ssl_config = None
        req = MagicMock()
        resp = MagicMock()
        resp.headers = {}
        result = add_security_headers(req, resp)
        assert result is resp
        mod.ssl_config = orig


class TestApp:
    def test_app_exists(self):
        assert app is not None
        assert app.title == "MFC Monitoring Dashboard API"
