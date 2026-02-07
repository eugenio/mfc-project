"""Deep unit tests for visualization modules and parameter_input with low coverage.

Targets:
  - gui/qtable_visualization.py (21% -> 40%+)
  - gui/policy_evolution_viz.py (10% -> 40%+)
  - gui/parameter_input.py (23% -> 40%+)
  - gui/live_monitoring_dashboard.py (0% -> 40%+)

All tests use mock-based approach to avoid Streamlit and heavy dependency issues.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_TESTS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

pytestmark = pytest.mark.apptest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Clear cached modules to avoid sys.modules pollution between tests."""
    mods_to_clear = [
        "gui.qtable_visualization",
        "gui.policy_evolution_viz",
        "gui.parameter_input",
        "gui.live_monitoring_dashboard",
    ]
    for mod_name in list(sys.modules):
        if any(mod_name == m or mod_name.startswith(m + ".") for m in mods_to_clear):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if any(mod_name == m or mod_name.startswith(m + ".") for m in mods_to_clear):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_plotly_figure(obj: object) -> bool:
    return isinstance(obj, go.Figure)


def _mock_columns(n_or_spec):
    """Mock for st.columns that returns the right number of MagicMock objects."""
    if isinstance(n_or_spec, int):
        return [MagicMock() for _ in range(n_or_spec)]
    if isinstance(n_or_spec, (list, tuple)):
        return [MagicMock() for _ in n_or_spec]
    return [MagicMock()]


def _mock_tabs(labels):
    """Mock for st.tabs that returns context managers."""
    return [MagicMock() for _ in labels]


# ===================================================================
# 1. LIVE MONITORING DASHBOARD (0% coverage -> target 40%+)
# ===================================================================


@pytest.mark.apptest
class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_level_values(self) -> None:
        from gui.live_monitoring_dashboard import AlertLevel

        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"

    def test_alert_level_members(self) -> None:
        from gui.live_monitoring_dashboard import AlertLevel

        assert len(AlertLevel) == 4


@pytest.mark.apptest
class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""

    def test_performance_metric_creation(self) -> None:
        from gui.live_monitoring_dashboard import PerformanceMetric

        now = datetime.now()
        pm = PerformanceMetric(
            timestamp=now,
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=6.8,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        assert pm.timestamp == now
        assert pm.power_output_mW == 0.5
        assert pm.cell_id == "Cell_01"

    def test_performance_metric_custom_cell_id(self) -> None:
        from gui.live_monitoring_dashboard import PerformanceMetric

        pm = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=6.8,
            temperature_C=30.0,
            conductivity_S_m=0.045,
            cell_id="Cell_03",
        )
        assert pm.cell_id == "Cell_03"


@pytest.mark.apptest
class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_alert_rule_defaults(self) -> None:
        from gui.live_monitoring_dashboard import AlertLevel, AlertRule

        rule = AlertRule(parameter="power_output_mW")
        assert rule.parameter == "power_output_mW"
        assert rule.threshold_min is None
        assert rule.threshold_max is None
        assert rule.level == AlertLevel.WARNING
        assert rule.enabled is True

    def test_alert_rule_custom(self) -> None:
        from gui.live_monitoring_dashboard import AlertLevel, AlertRule

        rule = AlertRule(
            parameter="ph_value",
            threshold_min=6.0,
            threshold_max=8.0,
            level=AlertLevel.CRITICAL,
            message_template="pH is {value:.2f}",
            enabled=False,
        )
        assert rule.threshold_min == 6.0
        assert rule.threshold_max == 8.0
        assert rule.level == AlertLevel.CRITICAL
        assert rule.enabled is False


@pytest.mark.apptest
class TestDashboardLayout:
    """Tests for DashboardLayout dataclass."""

    def test_dashboard_layout_defaults(self) -> None:
        from gui.live_monitoring_dashboard import DashboardLayout

        layout = DashboardLayout(panel_positions={})
        assert layout.refresh_interval == 5
        assert layout.max_data_points == 1000
        assert layout.auto_scroll is True

    def test_dashboard_layout_custom(self) -> None:
        from gui.live_monitoring_dashboard import DashboardLayout

        layout = DashboardLayout(
            panel_positions={"kpi": {"row": 0}},
            refresh_interval=10,
            max_data_points=500,
            auto_scroll=False,
        )
        assert layout.refresh_interval == 10
        assert layout.max_data_points == 500
        assert layout.auto_scroll is False


@pytest.mark.apptest
class TestLiveDataGenerator:
    """Tests for LiveDataGenerator class."""

    def test_init(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({"key": "value"})
        assert gen.base_config == {"key": "value"}
        assert gen.data_history == []
        assert isinstance(gen.start_time, datetime)

    def test_generate_realistic_data(self) -> None:
        from gui.live_monitoring_dashboard import (
            LiveDataGenerator,
            PerformanceMetric,
        )

        gen = LiveDataGenerator({})
        metric = gen.generate_realistic_data()
        assert isinstance(metric, PerformanceMetric)
        assert metric.cell_id == "Cell_01"
        assert isinstance(metric.power_output_mW, float)
        assert isinstance(metric.ph_value, float)

    def test_generate_realistic_data_custom_cell(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({})
        metric = gen.generate_realistic_data(cell_id="Cell_05")
        assert metric.cell_id == "Cell_05"

    def test_get_historical_data(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({})
        data = gen.get_historical_data(hours=1)
        # 1 hour * 12 data points per hour = 12 points
        assert len(data) == 12
        # Check timestamps are ordered
        for i in range(1, len(data)):
            assert data[i].timestamp > data[i - 1].timestamp

    def test_get_historical_data_longer_period(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({})
        data = gen.get_historical_data(hours=2)
        assert len(data) == 24

    def test_power_output_is_nonnegative(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({})
        # Generate multiple data points
        for _ in range(20):
            metric = gen.generate_realistic_data()
            assert metric.power_output_mW >= 0

    def test_substrate_concentration_nonnegative(self) -> None:
        from gui.live_monitoring_dashboard import LiveDataGenerator

        gen = LiveDataGenerator({})
        for _ in range(20):
            metric = gen.generate_realistic_data()
            assert metric.substrate_concentration_mM >= 0


@pytest.mark.apptest
class TestAlertManager:
    """Tests for AlertManager class."""

    def test_init_default_rules(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager

        mgr = AlertManager()
        assert len(mgr.rules) == 5
        assert mgr.active_alerts == []

    def test_default_rules_parameters(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager

        mgr = AlertManager()
        param_names = [r.parameter for r in mgr.rules]
        assert "power_output_mW" in param_names
        assert "substrate_concentration_mM" in param_names
        assert "ph_value" in param_names
        assert "temperature_C" in param_names
        assert "voltage_V" in param_names

    def test_check_alerts_no_trigger(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        # All values within normal range
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        assert alerts == []

    def test_check_alerts_low_power(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.05,  # Below 0.1 threshold
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=0.1,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        assert len(alerts) >= 1
        power_alerts = [a for a in alerts if a["parameter"] == "power_output_mW"]
        assert len(power_alerts) == 1
        assert power_alerts[0]["level"] == "warning"

    def test_check_alerts_low_ph(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=5.5,  # Below 6.0 threshold
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        ph_alerts = [a for a in alerts if a["parameter"] == "ph_value"]
        assert len(ph_alerts) == 1

    def test_check_alerts_high_ph(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=8.5,  # Above 8.0 threshold
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        ph_alerts = [a for a in alerts if a["parameter"] == "ph_value"]
        assert len(ph_alerts) == 1

    def test_check_alerts_high_substrate(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=55.0,  # Above 50.0 threshold
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        sub_alerts = [
            a for a in alerts if a["parameter"] == "substrate_concentration_mM"
        ]
        assert len(sub_alerts) == 1
        assert sub_alerts[0]["level"] == "critical"

    def test_check_alerts_low_substrate(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=1.0,  # Below 2.0 threshold
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        sub_alerts = [
            a for a in alerts if a["parameter"] == "substrate_concentration_mM"
        ]
        assert len(sub_alerts) == 1

    def test_check_alerts_disabled_rule(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        # Disable all rules
        for rule in mgr.rules:
            rule.enabled = False

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.01,  # Would trigger if enabled
            substrate_concentration_mM=1.0,
            current_density_mA_cm2=0.01,
            voltage_V=0.1,
            biofilm_thickness_um=40.0,
            ph_value=5.0,
            temperature_C=50.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        assert alerts == []

    def test_check_alerts_low_temperature(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=20.0,  # Below 25.0
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        temp_alerts = [a for a in alerts if a["parameter"] == "temperature_C"]
        assert len(temp_alerts) == 1

    def test_check_alerts_high_temperature(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.7,
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=45.0,  # Above 40.0
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        temp_alerts = [a for a in alerts if a["parameter"] == "temperature_C"]
        assert len(temp_alerts) == 1

    def test_check_alerts_low_voltage(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager, PerformanceMetric

        mgr = AlertManager()
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            power_output_mW=0.5,
            substrate_concentration_mM=25.0,
            current_density_mA_cm2=1.0,
            voltage_V=0.1,  # Below 0.3
            biofilm_thickness_um=40.0,
            ph_value=7.0,
            temperature_C=30.0,
            conductivity_S_m=0.045,
        )
        alerts = mgr.check_alerts(metric)
        volt_alerts = [a for a in alerts if a["parameter"] == "voltage_V"]
        assert len(volt_alerts) == 1
        assert volt_alerts[0]["level"] == "info"

    def test_add_alerts(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager

        mgr = AlertManager()
        alerts = [
            {"level": "warning", "message": "test1"},
            {"level": "critical", "message": "test2"},
        ]
        mgr.add_alerts(alerts)
        assert len(mgr.active_alerts) == 2

    def test_add_alerts_limit_100(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager

        mgr = AlertManager()
        # Add 150 alerts
        alerts = [{"level": "info", "message": f"test{i}"} for i in range(150)]
        mgr.add_alerts(alerts)
        assert len(mgr.active_alerts) == 100

    def test_get_active_alerts_all(self) -> None:
        from gui.live_monitoring_dashboard import AlertManager

        mgr = AlertManager()
        mgr.active_alerts = [
            {"level": "warning", "message": "w1"},
            {"level": "critical", "message": "c1"},
            {"level": "info", "message": "i1"},
        ]
        result = mgr.get_active_alerts()
        assert len(result) == 3

    def test_get_active_alerts_filtered(self) -> None:
        from gui.live_monitoring_dashboard import AlertLevel, AlertManager

        mgr = AlertManager()
        mgr.active_alerts = [
            {"level": "warning", "message": "w1"},
            {"level": "critical", "message": "c1"},
            {"level": "warning", "message": "w2"},
        ]
        result = mgr.get_active_alerts(level=AlertLevel.WARNING)
        assert len(result) == 2
        result = mgr.get_active_alerts(level=AlertLevel.CRITICAL)
        assert len(result) == 1


@pytest.mark.apptest
class TestLiveMonitoringDashboard:
    """Tests for LiveMonitoringDashboard class."""

    def _make_mock_session_state(self):
        """Create a dict-like mock for st.session_state."""
        state = {
            "simulation_start_time": datetime.now(),
            "monitoring_data": [],
            "monitoring_alerts": [],
            "last_update": datetime.now(),
            "monitoring_n_cells": 3,
        }

        mock_ss = MagicMock()
        mock_ss.__contains__ = lambda self, k: k in state
        mock_ss.__getattr__ = lambda self, k: state.get(k)
        mock_ss.__setattr__ = lambda self, k, v: state.__setitem__(k, v)
        mock_ss.__getitem__ = lambda self, k: state[k]
        mock_ss.__setitem__ = lambda self, k, v: state.__setitem__(k, v)
        mock_ss.get = lambda k, default=None: state.get(k, default)

        # Expose state for inspection
        mock_ss._state = state
        return mock_ss

    @patch("streamlit.session_state")
    def test_init(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 5

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=5)
        assert dashboard.n_cells == 5
        assert dashboard.data_generator is not None
        assert dashboard.alert_manager is not None

    @patch("streamlit.session_state")
    def test_get_default_layout(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        layout = dashboard._get_default_layout()
        assert layout.refresh_interval == 5
        assert layout.max_data_points == 500
        assert "kpi_overview" in layout.panel_positions
        assert "power_trend" in layout.panel_positions

    @patch("streamlit.session_state")
    def test_update_data_force(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 2
        mock_ss.get = MagicMock(return_value=2)

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=2)
        result = dashboard.update_data(force_update=True)
        assert result is True
        # Should have generated data for 2 cells
        assert len(mock_ss.monitoring_data) == 2

    @patch("streamlit.session_state")
    def test_update_data_no_update_needed(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 2
        mock_ss.get = MagicMock(return_value=2)

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=2)
        # Not enough time elapsed, should not update
        result = dashboard.update_data(force_update=False)
        assert result is False

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    def test_render_kpi_overview_no_data(self, mock_info, mock_md, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.render_kpi_overview()
        mock_info.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.metric")
    @patch("streamlit.columns", side_effect=_mock_columns)
    def test_render_kpi_overview_with_data(
        self, mock_cols, mock_metric, mock_md, mock_ss,
    ) -> None:
        from gui.live_monitoring_dashboard import (
            LiveMonitoringDashboard,
            PerformanceMetric,
        )

        now = datetime.now()
        metrics_list = [
            PerformanceMetric(
                timestamp=now,
                power_output_mW=0.5,
                substrate_concentration_mM=25.0,
                current_density_mA_cm2=1.0,
                voltage_V=0.7,
                biofilm_thickness_um=40.0,
                ph_value=7.0,
                temperature_C=30.0,
                conductivity_S_m=0.045,
                cell_id="Cell_01",
            ),
            PerformanceMetric(
                timestamp=now,
                power_output_mW=0.6,
                substrate_concentration_mM=24.0,
                current_density_mA_cm2=1.2,
                voltage_V=0.65,
                biofilm_thickness_um=42.0,
                ph_value=6.9,
                temperature_C=31.0,
                conductivity_S_m=0.044,
                cell_id="Cell_02",
            ),
        ]

        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = now
        mock_ss.monitoring_data = metrics_list
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = now
        mock_ss.monitoring_n_cells = 2

        dashboard = LiveMonitoringDashboard(n_cells=2)
        dashboard.render_kpi_overview()
        mock_metric.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    def test_render_power_trend_no_data(self, mock_info, mock_md, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.render_power_trend_chart()
        mock_info.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    def test_render_power_trend_with_data(
        self, mock_chart, mock_md, mock_ss,
    ) -> None:
        from gui.live_monitoring_dashboard import (
            LiveMonitoringDashboard,
            PerformanceMetric,
        )

        now = datetime.now()
        data_points = []
        for i in range(5):
            data_points.append(
                PerformanceMetric(
                    timestamp=now + timedelta(minutes=i),
                    power_output_mW=0.5 + i * 0.01,
                    substrate_concentration_mM=25.0 - i * 0.1,
                    current_density_mA_cm2=1.0,
                    voltage_V=0.7,
                    biofilm_thickness_um=40.0,
                    ph_value=7.0,
                    temperature_C=30.0,
                    conductivity_S_m=0.045,
                    cell_id="Cell_01",
                ),
            )

        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = now
        mock_ss.monitoring_data = data_points
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = now
        mock_ss.monitoring_n_cells = 1

        dashboard = LiveMonitoringDashboard(n_cells=1)
        dashboard.render_power_trend_chart()
        mock_chart.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    def test_render_substrate_trend_with_data(
        self, mock_chart, mock_md, mock_ss,
    ) -> None:
        from gui.live_monitoring_dashboard import (
            LiveMonitoringDashboard,
            PerformanceMetric,
        )

        now = datetime.now()
        data_points = [
            PerformanceMetric(
                timestamp=now + timedelta(minutes=i),
                power_output_mW=0.5,
                substrate_concentration_mM=25.0 - i,
                current_density_mA_cm2=1.0,
                voltage_V=0.7,
                biofilm_thickness_um=40.0,
                ph_value=7.0,
                temperature_C=30.0,
                conductivity_S_m=0.045,
            )
            for i in range(3)
        ]

        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = now
        mock_ss.monitoring_data = data_points
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = now
        mock_ss.monitoring_n_cells = 1

        dashboard = LiveMonitoringDashboard(n_cells=1)
        dashboard.render_substrate_trend_chart()
        mock_chart.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    def test_render_multicell_no_data(self, mock_info, mock_md, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.render_multicell_comparison()
        mock_info.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.plotly_chart")
    def test_render_multicell_with_data(
        self, mock_chart, mock_md, mock_ss,
    ) -> None:
        from gui.live_monitoring_dashboard import (
            LiveMonitoringDashboard,
            PerformanceMetric,
        )

        now = datetime.now()
        data_points = [
            PerformanceMetric(
                timestamp=now,
                power_output_mW=0.5,
                substrate_concentration_mM=25.0,
                current_density_mA_cm2=1.0,
                voltage_V=0.7,
                biofilm_thickness_um=40.0,
                ph_value=7.0,
                temperature_C=30.0,
                conductivity_S_m=0.045,
                cell_id="Cell_01",
            ),
            PerformanceMetric(
                timestamp=now,
                power_output_mW=0.6,
                substrate_concentration_mM=24.0,
                current_density_mA_cm2=1.2,
                voltage_V=0.65,
                biofilm_thickness_um=42.0,
                ph_value=6.9,
                temperature_C=31.0,
                conductivity_S_m=0.044,
                cell_id="Cell_02",
            ),
        ]

        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = now
        mock_ss.monitoring_data = data_points
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = now
        mock_ss.monitoring_n_cells = 2

        dashboard = LiveMonitoringDashboard(n_cells=2)
        dashboard.render_multicell_comparison()
        mock_chart.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.success")
    def test_render_alerts_panel_no_alerts(
        self, mock_success, mock_md, mock_ss,
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.render_alerts_panel()
        mock_success.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.markdown")
    @patch("streamlit.error")
    @patch("streamlit.warning")
    @patch("streamlit.info")
    def test_render_alerts_panel_with_alerts(
        self, mock_info, mock_warn, mock_err, mock_md, mock_ss,
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = [
            {
                "level": "critical",
                "cell_id": "Cell_01",
                "message": "Substrate low",
            },
            {
                "level": "warning",
                "cell_id": "Cell_02",
                "message": "pH drifting",
            },
            {
                "level": "info",
                "cell_id": "Cell_03",
                "message": "Voltage low",
            },
        ]
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.render_alerts_panel()
        mock_err.assert_called()
        mock_warn.assert_called()
        mock_info.assert_called()

    @patch("streamlit.session_state")
    @patch("streamlit.expander")
    @patch("streamlit.slider")
    @patch("streamlit.checkbox")
    def test_render_settings_panel(
        self, mock_cb, mock_slider, mock_expander, mock_ss,
    ) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now()
        mock_ss.monitoring_data = []
        mock_ss.monitoring_alerts = []
        mock_ss.last_update = datetime.now()
        mock_ss.monitoring_n_cells = 3
        mock_ss.__setitem__ = MagicMock()

        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()
        mock_slider.return_value = 5
        mock_cb.return_value = False

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        with patch("streamlit.columns", side_effect=_mock_columns):
            dashboard = LiveMonitoringDashboard(n_cells=3)
            dashboard.render_settings_panel()

    @patch("streamlit.session_state")
    def test_reset_simulation_time(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.simulation_start_time = datetime.now() - timedelta(hours=1)
        mock_ss.monitoring_data = [MagicMock()]
        mock_ss.monitoring_alerts = [MagicMock()]
        mock_ss.last_update = datetime.now() - timedelta(hours=1)
        mock_ss.monitoring_n_cells = 3

        from gui.live_monitoring_dashboard import LiveMonitoringDashboard

        dashboard = LiveMonitoringDashboard(n_cells=3)
        dashboard.reset_simulation_time()
        assert mock_ss.monitoring_data == []
        assert mock_ss.monitoring_alerts == []


# ===================================================================
# 2. QTABLE VISUALIZATION - deeper tests
# ===================================================================


@pytest.mark.apptest
class TestQTableVisualizationDeep:
    """Deeper tests for gui/qtable_visualization.py methods."""

    def _make_mock_modules(self) -> dict[str, MagicMock]:
        """Create mock modules for qtable_visualization imports."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_available_qtables.return_value = []
        mock_analyzer.load_qtable.return_value = np.random.randn(5, 3)

        # Create a proper ConvergenceStatus enum-like mock
        mock_converged = MagicMock()
        mock_converged.value = "converged"

        mock_converging = MagicMock()
        mock_converging.value = "converging"

        mock_unstable = MagicMock()
        mock_unstable.value = "unstable"

        mock_module = MagicMock()
        mock_module.QTABLE_ANALYZER = mock_analyzer
        mock_module.ConvergenceStatus = MagicMock()
        mock_module.ConvergenceStatus.CONVERGED = mock_converged
        mock_module.ConvergenceStatus.CONVERGING = mock_converging
        mock_module.ConvergenceStatus.UNSTABLE = mock_unstable
        mock_module.QTableMetrics = MagicMock()
        return {"analysis.qtable_analyzer": mock_module}

    @patch("streamlit.session_state")
    def test_create_heatmap_small_qtable(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            q = np.array([[1.0, 2.0], [3.0, 4.0]])
            fig = comp._create_qtable_heatmap(q, "small.pkl")
            assert _is_plotly_figure(fig)
            assert fig.layout.height >= 400

    @patch("streamlit.session_state")
    def test_create_heatmap_large_qtable(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            q = np.random.randn(50, 6)
            fig = comp._create_qtable_heatmap(q, "large.pkl")
            assert _is_plotly_figure(fig)
            assert fig.layout.height <= 800

    @patch("streamlit.session_state")
    def test_create_heatmap_various_colorscales(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            q = np.random.randn(5, 3)
            for scale in ["viridis", "plasma", "rdylbu", "blues", "reds"]:
                fig = comp._create_qtable_heatmap(q, "test.pkl", colorscale=scale)
                assert _is_plotly_figure(fig)

    @patch("streamlit.session_state")
    def test_get_file_size_mb(self, mock_ss) -> None:
        """Test file size formatting for MB range."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            mock_path = MagicMock()
            mock_stat = MagicMock()
            mock_stat.st_size = 2 * 1024 * 1024  # 2 MB
            mock_path.stat.return_value = mock_stat
            result = comp._get_file_size(mock_path)
            assert "MB" in result

    @patch("streamlit.session_state")
    def test_get_file_size_gb(self, mock_ss) -> None:
        """Test file size formatting for GB range."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            mock_path = MagicMock()
            mock_stat = MagicMock()
            mock_stat.st_size = 2 * 1024 * 1024 * 1024  # 2 GB
            mock_path.stat.return_value = mock_stat
            result = comp._get_file_size(mock_path)
            assert "GB" in result

    @patch("streamlit.session_state")
    def test_get_file_size_small_bytes(self, mock_ss) -> None:
        """Test file size formatting for small files (bytes)."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            mock_path = MagicMock()
            mock_stat = MagicMock()
            mock_stat.st_size = 500  # 500 bytes
            mock_path.stat.return_value = mock_stat
            result = comp._get_file_size(mock_path)
            assert "B" in result

    @patch("streamlit.session_state")
    def test_extract_timestamp_various_patterns(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            # With timestamp pattern
            assert comp._extract_timestamp_from_filename(
                "model_20250801_120000.pkl",
            ) == "20250801_120000"
            # Without timestamp
            assert comp._extract_timestamp_from_filename(
                "no_date.pkl",
            ) == "no_date.pkl"
            # Timestamp in the middle
            assert comp._extract_timestamp_from_filename(
                "qtable_20241231_235959_v2.pkl",
            ) == "20241231_235959"

    @patch("streamlit.session_state")
    def test_display_analysis_summary(self, mock_ss) -> None:
        """Test _display_analysis_summary with mock metrics."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        mocks = self._make_mock_modules()
        converged_status = mocks["analysis.qtable_analyzer"].ConvergenceStatus.CONVERGED

        with patch.dict(sys.modules, mocks):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()

            # Create mock metrics
            mock_metrics_1 = MagicMock()
            mock_metrics_1.convergence_status = converged_status
            mock_metrics_1.convergence_score = 0.95
            mock_metrics_1.exploration_coverage = 0.8

            mock_metrics_2 = MagicMock()
            mock_metrics_2.convergence_status = MagicMock()
            mock_metrics_2.convergence_score = 0.7
            mock_metrics_2.exploration_coverage = 0.6

            results = {
                "file1.pkl": mock_metrics_1,
                "file2.pkl": mock_metrics_2,
            }

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"):
                comp._display_analysis_summary(results)

    @patch("streamlit.session_state")
    def test_display_detailed_metrics(self, mock_ss) -> None:
        """Test _display_detailed_metrics table rendering."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        mocks = self._make_mock_modules()
        converged_status = mocks["analysis.qtable_analyzer"].ConvergenceStatus.CONVERGED

        with patch.dict(sys.modules, mocks):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()

            mock_metrics = MagicMock()
            mock_metrics.total_states = 10
            mock_metrics.total_actions = 4
            mock_metrics.convergence_score = 0.85
            mock_metrics.convergence_status = converged_status
            mock_metrics.policy_entropy = 0.5
            mock_metrics.exploration_coverage = 0.75
            mock_metrics.sparsity = 0.3
            mock_metrics.q_value_range = 2.5

            results = {"test.pkl": mock_metrics}

            with patch("streamlit.markdown"), \
                 patch("streamlit.dataframe"):
                comp._display_detailed_metrics(results)

    @patch("streamlit.session_state")
    def test_display_trend_statistics(self, mock_ss) -> None:
        """Test _display_trend_statistics method."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()

            df = pd.DataFrame({
                "convergence_score": [0.5, 0.7, 0.9],
                "stability_measure": [0.3, 0.5, 0.8],
                "exploration_coverage": [0.8, 0.6, 0.4],
                "policy_entropy": [1.2, 0.8, 0.3],
            })

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"):
                comp._display_trend_statistics(df)

    @patch("streamlit.session_state")
    def test_display_policy_insights(self, mock_ss) -> None:
        """Test _display_policy_insights method."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()

            df = pd.DataFrame({
                "File": ["a.pkl", "b.pkl", "c.pkl"],
                "Policy Entropy": [0.5, 0.8, 0.3],
                "Action Diversity": [0.6, 0.9, 0.4],
                "State Value Variance": [0.1, 0.2, 0.3],
                "Convergence Score": [0.9, 0.7, 0.95],
            })

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns):
                comp._display_policy_insights(df)

    @patch("streamlit.session_state")
    def test_display_comparison_results(self, mock_ss) -> None:
        """Test _display_comparison_results method."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}

        mock_comparison = MagicMock()
        mock_comparison.policy_agreement = 0.85
        mock_comparison.convergence_improvement = 0.1
        mock_comparison.learning_progress = 0.2
        mock_comparison.stability_change = 0.05

        mock_ss.comparison_results = {
            "file1.pkl|file2.pkl": mock_comparison,
        }

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"):
                comp._display_comparison_results()

    @patch("streamlit.session_state")
    def test_display_comparison_results_empty(self, mock_ss) -> None:
        """Test _display_comparison_results with no results."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            # Should not raise
            comp._display_comparison_results()

    @patch("streamlit.session_state")
    def test_export_visualizations(self, mock_ss) -> None:
        """Test _export_visualizations shows info."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            with patch("streamlit.info") as mock_info:
                comp._export_visualizations()
                mock_info.assert_called()

    @patch("streamlit.session_state")
    def test_generate_analysis_report(self, mock_ss) -> None:
        """Test _generate_analysis_report shows info."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            with patch("streamlit.info") as mock_info:
                comp._generate_analysis_report()
                mock_info.assert_called()

    @patch("streamlit.session_state")
    def test_export_metrics_csv_no_data(self, mock_ss) -> None:
        """Test _export_metrics_csv when no cache."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {}
        mock_ss.comparison_results = {}

        with patch.dict(sys.modules, self._make_mock_modules()):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            with patch("streamlit.warning") as mock_warn:
                comp._export_metrics_csv()
                mock_warn.assert_called()

    @patch("streamlit.session_state")
    def test_export_metrics_csv_with_data(self, mock_ss) -> None:
        """Test _export_metrics_csv with cached data."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.selected_qtables = []
        mock_ss.qtable_analysis_cache = {"file.pkl": MagicMock()}
        mock_ss.comparison_results = {}

        mocks = self._make_mock_modules()
        with patch.dict(sys.modules, mocks):
            from gui.qtable_visualization import QTableVisualization

            comp = QTableVisualization()
            with patch("streamlit.success"):
                comp._export_metrics_csv()
                comp.analyzer.export_analysis_results.assert_called_once()


# ===================================================================
# 3. POLICY EVOLUTION VIZ - deeper tests
# ===================================================================


@pytest.mark.apptest
class TestPolicyEvolutionVizDeep:
    """Deeper tests for gui/policy_evolution_viz.py methods."""

    def _make_policy_stability(self):
        """Create a proper PolicyStability enum-like mock."""
        mock_stability = MagicMock()
        mock_stability.STABLE = MagicMock()
        mock_stability.STABLE.value = "stable"
        mock_stability.CONVERGING = MagicMock()
        mock_stability.CONVERGING.value = "converging"
        mock_stability.UNSTABLE = MagicMock()
        mock_stability.UNSTABLE.value = "unstable"
        mock_stability.OSCILLATING = MagicMock()
        mock_stability.OSCILLATING.value = "oscillating"
        mock_stability.UNKNOWN = MagicMock()
        mock_stability.UNKNOWN.value = "unknown"
        return mock_stability

    def _make_mock_modules(self) -> tuple[dict[str, MagicMock], MagicMock]:
        """Create mock modules needed by policy_evolution_viz."""
        mock_stability = self._make_policy_stability()

        mock_tracker_instance = MagicMock()
        mock_tracker_instance.policy_snapshots = []

        mock_module = MagicMock()
        mock_module.POLICY_EVOLUTION_TRACKER = mock_tracker_instance
        mock_module.PolicyEvolutionMetrics = MagicMock()
        mock_module.PolicyStability = mock_stability

        return (
            {"analysis.policy_evolution_tracker": mock_module},
            mock_stability,
        )

    def _make_mock_metrics(self, stability):
        """Create a mock PolicyEvolutionMetrics."""
        metrics = MagicMock()
        metrics.total_episodes = 100
        metrics.snapshots_count = 10
        metrics.stability_status = stability.STABLE
        metrics.stability_score = 0.95
        metrics.convergence_episode = 50
        metrics.action_preference_changes = 2
        metrics.dominant_actions = {0: 0.6, 1: 0.3, 2: 0.1}
        metrics.policy_changes = [10, 8, 5, 3, 2, 1, 1, 0, 0]
        metrics.performance_trend = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics.learning_velocity = [0.5, 0.3, 0.2, 0.1]
        metrics.exploration_decay = [0.9, 0.7, 0.5, 0.3]
        return metrics

    @patch("streamlit.session_state")
    def test_render_timeline_insights_early_convergence(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.convergence_episode = 10  # 10% of 100 episodes
            metrics.stability_score = 0.97

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_timeline_insights_good_convergence(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.convergence_episode = 40  # 40% of 100 episodes

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_timeline_insights_late_convergence(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.convergence_episode = 80  # 80% of 100 episodes

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_timeline_insights_no_convergence(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.convergence_episode = None
            metrics.stability_score = 0.5

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_timeline_insights_single_action(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.dominant_actions = {0: 1.0}

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_timeline_insights_many_actions(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.dominant_actions = {0: 0.3, 1: 0.2, 2: 0.2, 3: 0.15, 4: 0.15}

            with patch("streamlit.markdown"):
                comp._render_timeline_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_stability_insights_stable(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_status = stability.STABLE
            metrics.policy_changes = [2, 1, 1, 0, 0]
            metrics.action_preference_changes = 0

            with patch("streamlit.markdown"):
                comp._render_stability_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_stability_insights_unstable(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_status = stability.UNSTABLE
            metrics.policy_changes = [20, 18, 22, 25, 19]
            metrics.action_preference_changes = 5

            with patch("streamlit.markdown"):
                comp._render_stability_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_stability_insights_converging(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_status = stability.CONVERGING
            metrics.policy_changes = [10, 8, 7, 6, 5]
            metrics.action_preference_changes = 2

            with patch("streamlit.markdown"):
                comp._render_stability_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_stability_insights_oscillating(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_status = stability.OSCILLATING
            metrics.policy_changes = []
            metrics.action_preference_changes = 0

            with patch("streamlit.markdown"):
                comp._render_stability_insights(metrics)

    @patch("streamlit.session_state")
    def test_render_stability_insights_unknown(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_status = stability.UNKNOWN
            metrics.policy_changes = [8, 10, 7, 12]
            metrics.action_preference_changes = 3

            with patch("streamlit.markdown"):
                comp._render_stability_insights(metrics)

    @patch("streamlit.session_state")
    def test_generate_policy_evolution_report_converged(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.stability_score = 0.95

            with patch("streamlit.markdown") as mock_md, \
                 patch("streamlit.download_button"):
                comp._generate_policy_evolution_report(metrics)
                # Verify markdown was called with a report
                mock_md.assert_called()

    @patch("streamlit.session_state")
    def test_generate_policy_evolution_report_not_converged(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.convergence_episode = None
            metrics.stability_score = 0.5
            metrics.stability_status = stability.UNSTABLE

            with patch("streamlit.markdown"), \
                 patch("streamlit.download_button"):
                comp._generate_policy_evolution_report(metrics)

    @patch("streamlit.session_state")
    def test_render_learning_curves_no_performance(self, mock_ss) -> None:
        """Test learning curves when no performance trend but velocity exists."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.performance_trend = []
            metrics.learning_velocity = [0.5, 0.3, 0.2, 0.1]
            metrics.exploration_decay = [0.9, 0.7, 0.5]

            with patch("streamlit.markdown"), \
                 patch("streamlit.plotly_chart"), \
                 patch("streamlit.info"):
                comp._render_learning_curves(metrics)

    @patch("streamlit.session_state")
    def test_render_learning_curves_with_performance(self, mock_ss) -> None:
        """Test learning curves with performance trend."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.performance_trend = [1.0, 2.0, 3.0, 4.0, 5.0]
            metrics.learning_velocity = [0.5, 0.3, 0.2, 0.1]
            metrics.exploration_decay = []

            with patch("streamlit.markdown"), \
                 patch("streamlit.plotly_chart"):
                comp._render_learning_curves(metrics)

    @patch("streamlit.session_state")
    def test_render_learning_curves_no_velocity(self, mock_ss) -> None:
        """Test learning curves when no velocity or performance data."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.performance_trend = []
            metrics.learning_velocity = []
            metrics.exploration_decay = []

            with patch("streamlit.markdown"), \
                 patch("streamlit.info"):
                comp._render_learning_curves(metrics)

    @patch("streamlit.session_state")
    def test_render_policy_stability_analysis_with_changes(self, mock_ss) -> None:
        """Test policy stability analysis rendering."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.policy_changes = [10, 8, 5, 3, 2, 1]

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.plotly_chart"):
                comp._render_policy_stability_analysis(metrics)

    @patch("streamlit.session_state")
    def test_render_policy_stability_analysis_no_changes(self, mock_ss) -> None:
        """Test policy stability analysis when no policy changes."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)
            metrics.policy_changes = []

            with patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.plotly_chart"):
                comp._render_policy_stability_analysis(metrics)

    @patch("streamlit.session_state")
    def test_render_episode_comparison_not_enough(self, mock_ss) -> None:
        """Test episode comparison with fewer than 2 snapshots."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()
        mods["analysis.policy_evolution_tracker"].POLICY_EVOLUTION_TRACKER.policy_snapshots = [
            MagicMock(episode=1),
        ]

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()

            with patch("streamlit.markdown"), \
                 patch("streamlit.info") as mock_info:
                comp._render_episode_comparison()
                mock_info.assert_called()

    @patch("streamlit.session_state")
    def test_render_export_section(self, mock_ss) -> None:
        """Test export section rendering."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.policy_snapshots_loaded = False
        mock_ss.policy_evolution_metrics = None
        mock_ss.selected_episodes = []

        mods, stability = self._make_mock_modules()

        with patch.dict(sys.modules, mods):
            from gui.policy_evolution_viz import PolicyEvolutionVisualization

            comp = PolicyEvolutionVisualization()
            metrics = self._make_mock_metrics(stability)

            with patch("streamlit.subheader"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.button", return_value=False):
                comp._render_export_section(metrics)


# ===================================================================
# 4. PARAMETER INPUT - deeper tests
# ===================================================================


@pytest.mark.apptest
class TestParameterInputDeep:
    """Deeper tests for gui/parameter_input.py methods."""

    def _make_mock_config_modules(self) -> dict[str, MagicMock]:
        """Create mock modules for parameter_input imports."""
        # --- literature_database ---
        mock_param_category = MagicMock(spec=Enum)
        mock_param_category.return_value = MagicMock()

        mock_param_info = MagicMock()

        mock_lit_db = MagicMock()
        mock_lit_db.get_all_categories.return_value = []
        mock_lit_db.get_parameter.return_value = None
        mock_lit_db.get_parameters_by_category.return_value = []

        mock_lit_module = MagicMock()
        mock_lit_module.LITERATURE_DB = mock_lit_db
        mock_lit_module.ParameterCategory = mock_param_category
        mock_lit_module.ParameterInfo = mock_param_info

        # --- parameter_bridge ---
        mock_bridge = MagicMock()
        mock_bridge.create_literature_validated_config.return_value = (
            MagicMock(),
            {},
        )
        mock_bridge_module = MagicMock()
        mock_bridge_module.PARAMETER_BRIDGE = mock_bridge

        # --- real_time_validator ---
        mock_validation_level = MagicMock(spec=Enum)
        mock_validation_level.VALID = MagicMock()
        mock_validation_level.VALID.value = "valid"
        mock_validation_level.CAUTION = MagicMock()
        mock_validation_level.CAUTION.value = "caution"
        mock_validation_level.INVALID = MagicMock()
        mock_validation_level.INVALID.value = "invalid"

        mock_validator = MagicMock()
        mock_validator.get_research_objectives.return_value = []
        mock_validator.get_performance_metrics.return_value = {
            "avg_response_time_ms": 50.0,
            "cache_hit_rate": 0.9,
            "fast_validations": 90,
            "instant_validations": 80,
            "total_validations": 100,
        }

        mock_rtv_module = MagicMock()
        mock_rtv_module.REAL_TIME_VALIDATOR = mock_validator
        mock_rtv_module.ValidationLevel = mock_validation_level

        # --- unit_converter ---
        mock_converter = MagicMock()
        mock_converter.get_compatible_units.return_value = ["V"]
        mock_uc_module = MagicMock()
        mock_uc_module.UNIT_CONVERTER = mock_converter

        # --- qlearning_config ---
        mock_ql_config = MagicMock()

        return {
            "config.literature_database": mock_lit_module,
            "config.parameter_bridge": mock_bridge_module,
            "config.real_time_validator": mock_rtv_module,
            "config.unit_converter": mock_uc_module,
            "config.qlearning_config": mock_ql_config,
        }

    def _make_mock_param(self):
        """Create a mock ParameterInfo."""
        mock_param = MagicMock()
        mock_param.name = "test_param"
        mock_param.symbol = "Tp"
        mock_param.unit = "V"
        mock_param.min_value = 0.0
        mock_param.max_value = 1.0
        mock_param.recommended_range = (0.3, 0.7)
        mock_param.typical_value = 0.5
        mock_param.description = "A test parameter"
        mock_param.category = MagicMock()
        mock_param.category.value = "electrochemical"
        mock_param.references = []
        mock_param.notes = "Test notes"
        return mock_param

    @patch("streamlit.session_state")
    def test_render_parameter_references_with_doi(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            param = self._make_mock_param()

            # Add references with DOI
            ref1 = MagicMock()
            ref1.format_citation.return_value = "Author (2020). Title."
            ref1.doi = "10.1234/test"
            ref2 = MagicMock()
            ref2.format_citation.return_value = "Author2 (2021). Title2."
            ref2.doi = None
            param.references = [ref1, ref2]
            param.notes = "Important notes"

            with patch("streamlit.markdown"):
                comp._render_parameter_references(param)

    @patch("streamlit.session_state")
    def test_render_parameter_references_no_notes(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            param = self._make_mock_param()
            param.references = []
            param.notes = ""

            with patch("streamlit.markdown"):
                comp._render_parameter_references(param)

    @patch("streamlit.session_state")
    def test_render_parameter_summary_with_data(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"test_param": 0.5, "another_param": 0.8}
        mock_ss.validation_results = {
            "test_param": {"status": "valid"},
            "another_param": {"status": "caution"},
        }
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        param = self._make_mock_param()
        mocks["config.literature_database"].LITERATURE_DB.get_parameter.return_value = param

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            with patch("streamlit.dataframe"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"):
                comp._render_parameter_summary()

    @patch("streamlit.session_state")
    def test_render_parameter_summary_no_param_found(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"unknown_param": 0.5}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks["config.literature_database"].LITERATURE_DB.get_parameter.return_value = None

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            # Should not raise even if param not found
            comp._render_parameter_summary()

    @patch("streamlit.session_state")
    def test_validate_all_parameters_all_valid(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"p1": 0.5, "p2": 0.7}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks[
            "config.literature_database"
        ].LITERATURE_DB.validate_parameter_value.return_value = {
            "status": "valid",
            "message": "OK",
        }

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.success") as mock_succ:
                comp._validate_all_parameters()
                mock_succ.assert_called()

    @patch("streamlit.session_state")
    def test_validate_all_parameters_some_invalid(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"p1": 0.5, "p2": 99.0}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        call_count = [0]

        def mock_validate(name, value):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"status": "valid", "message": "OK"}
            return {"status": "invalid", "message": "Out of range"}

        mocks[
            "config.literature_database"
        ].LITERATURE_DB.validate_parameter_value.side_effect = mock_validate

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.warning") as mock_warn:
                comp._validate_all_parameters()
                mock_warn.assert_called()

    @patch("streamlit.session_state")
    def test_render_enhanced_validation_valid_high_confidence(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            mock_result = MagicMock()
            mock_result.level = mocks["config.real_time_validator"].ValidationLevel.VALID
            mock_result.scientific_reasoning = "Based on electrochemical literature values for MFC systems"
            mock_result.confidence_score = 0.95
            mock_result.uncertainty_bounds = (0.4, 0.6)
            mock_result.response_time_ms = 50.0
            mock_result.warnings = []
            mock_result.recommendations = []
            mock_result.suggested_ranges = []

            with patch("streamlit.success"), \
                 patch("streamlit.caption"):
                comp._render_enhanced_validation_indicator(mock_result)

    @patch("streamlit.session_state")
    def test_render_enhanced_validation_caution(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            mock_result = MagicMock()
            mock_result.level = mocks["config.real_time_validator"].ValidationLevel.CAUTION
            mock_result.scientific_reasoning = "Value is near the boundary of recommended ranges"
            mock_result.confidence_score = 0.7
            mock_result.uncertainty_bounds = (0.2, 0.8)
            mock_result.response_time_ms = 150.0
            mock_result.warnings = ["Near threshold"]
            mock_result.recommendations = ["Consider adjusting"]
            mock_result.suggested_ranges = [(0.3, 0.7)]

            with patch("streamlit.warning"), \
                 patch("streamlit.caption"), \
                 patch("streamlit.expander") as mock_exp:
                mock_exp.return_value.__enter__ = MagicMock()
                mock_exp.return_value.__exit__ = MagicMock()
                comp._render_enhanced_validation_indicator(mock_result)

    @patch("streamlit.session_state")
    def test_render_enhanced_validation_invalid(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            mock_result = MagicMock()
            mock_result.level = mocks["config.real_time_validator"].ValidationLevel.INVALID
            mock_result.scientific_reasoning = "Value is outside valid range for MFC operation"
            mock_result.confidence_score = 0.2
            mock_result.uncertainty_bounds = (0.0, 1.5)
            mock_result.response_time_ms = 250.0
            mock_result.warnings = ["Critical issue", "Parameter invalid", "Extra"]
            mock_result.recommendations = ["Fix value"]
            mock_result.suggested_ranges = [(0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]

            with patch("streamlit.error"), \
                 patch("streamlit.caption"), \
                 patch("streamlit.expander") as mock_exp:
                mock_exp.return_value.__enter__ = MagicMock()
                mock_exp.return_value.__exit__ = MagicMock()
                with patch("streamlit.markdown"):
                    comp._render_enhanced_validation_indicator(mock_result)

    @patch("streamlit.session_state")
    def test_render_enhanced_validation_unknown(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            mock_result = MagicMock()
            mock_result.level = MagicMock()  # Not valid/caution/invalid
            mock_result.scientific_reasoning = "Unknown"
            mock_result.confidence_score = 0.0
            mock_result.uncertainty_bounds = (0.0, 0.0)
            mock_result.response_time_ms = 100.0
            mock_result.warnings = []
            mock_result.recommendations = []
            mock_result.suggested_ranges = []

            with patch("streamlit.info"), \
                 patch("streamlit.caption"):
                comp._render_enhanced_validation_indicator(mock_result)

    @patch("streamlit.session_state")
    def test_render_enhanced_validation_slow_response(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()

            mock_result = MagicMock()
            mock_result.level = mocks["config.real_time_validator"].ValidationLevel.VALID
            mock_result.scientific_reasoning = "OK value based on literature"
            mock_result.confidence_score = 0.85
            mock_result.uncertainty_bounds = (0.4, 0.6)
            mock_result.response_time_ms = 500.0  # Slow
            mock_result.warnings = []
            mock_result.recommendations = []
            mock_result.suggested_ranges = []

            with patch("streamlit.success"), \
                 patch("streamlit.caption"):
                comp._render_enhanced_validation_indicator(mock_result)

    @patch("streamlit.session_state")
    def test_create_validated_config_no_values(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            result = comp._create_validated_config()
            assert result is None

    @patch("streamlit.session_state")
    def test_create_validated_config_with_values(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"learning_rate": 0.1}
        mock_ss.validation_results = {"learning_rate": {"status": "valid"}}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mock_config = MagicMock()
        mocks[
            "config.parameter_bridge"
        ].PARAMETER_BRIDGE.create_literature_validated_config.return_value = (
            mock_config,
            {"learning_rate": {"adjusted": True}},
        )

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            result = comp._create_validated_config()
            assert result is mock_config

    @patch("streamlit.session_state")
    def test_create_validated_config_error(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"learning_rate": 0.1}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks[
            "config.parameter_bridge"
        ].PARAMETER_BRIDGE.create_literature_validated_config.side_effect = (
            ValueError("Config error")
        )

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.error"):
                result = comp._create_validated_config()
                assert result is None

    @patch("streamlit.session_state")
    def test_show_citations_apa(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {"test_param": 0.5}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        param = self._make_mock_param()
        ref = MagicMock()
        ref.authors = "Smith et al."
        ref.year = 2020
        ref.format_citation.return_value = "Smith et al. (2020). Title."
        param.references = [ref]
        mocks["config.literature_database"].LITERATURE_DB.get_parameter.return_value = (
            param
        )

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.subheader"), \
                 patch("streamlit.selectbox", return_value="APA"), \
                 patch("streamlit.markdown"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"):
                comp._show_citations()

    @patch("streamlit.session_state")
    def test_render_category_parameters_empty(self, mock_ss) -> None:
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            category = MagicMock()
            category.value = "electrochemical"

            with patch("streamlit.info") as mock_info:
                comp._render_category_parameters(category)
                mock_info.assert_called()

    @patch("streamlit.session_state")
    def test_render_performance_metrics_slow(self, mock_ss) -> None:
        """Test performance metrics display when response time is slow."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks[
            "config.real_time_validator"
        ].REAL_TIME_VALIDATOR.get_performance_metrics.return_value = {
            "avg_response_time_ms": 600.0,
            "cache_hit_rate": 0.3,
            "fast_validations": 20,
            "instant_validations": 5,
            "total_validations": 100,
        }

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.subheader"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"), \
                 patch("streamlit.error") as mock_err:
                comp._render_performance_metrics()
                mock_err.assert_called()

    @patch("streamlit.session_state")
    def test_render_performance_metrics_acceptable(self, mock_ss) -> None:
        """Test performance metrics display when response time is acceptable."""
        mock_ss.__contains__ = lambda self, k: True
        mock_ss.parameter_values = {}
        mock_ss.validation_results = {}
        mock_ss.parameter_citations = {}
        mock_ss.research_objective = None
        mock_ss.show_performance_metrics = False

        mocks = self._make_mock_config_modules()
        mocks[
            "config.real_time_validator"
        ].REAL_TIME_VALIDATOR.get_performance_metrics.return_value = {
            "avg_response_time_ms": 300.0,
            "cache_hit_rate": 0.6,
            "fast_validations": 60,
            "instant_validations": 30,
            "total_validations": 100,
        }

        with patch.dict(sys.modules, mocks):
            from gui.parameter_input import ParameterInputComponent

            comp = ParameterInputComponent()
            with patch("streamlit.subheader"), \
                 patch("streamlit.columns", side_effect=_mock_columns), \
                 patch("streamlit.metric"), \
                 patch("streamlit.warning") as mock_warn:
                comp._render_performance_metrics()
                mock_warn.assert_called()
