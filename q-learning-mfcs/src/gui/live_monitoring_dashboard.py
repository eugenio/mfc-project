#!/usr/bin/env python3
"""Live Performance Monitoring Dashboard for MFC Research Platform.

ADVANCED REAL-TIME MONITORING COMPONENT
---------------------------------------
This module provides comprehensive live monitoring capabilities for MFC systems,
distinct from the simple overview dashboard (gui/pages/dashboard.py).

Features:
- Real-time streaming data visualization (<5 second updates)
- Customizable dashboard layout with drag-and-drop panels
- Key performance indicators: power output, substrate concentration, pH
- Historical data integration with zoom and pan functionality
- Alert system for critical parameter thresholds
- Multi-cell stack monitoring support

Dashboard Component Hierarchy:
-----------------------------
1. **gui/pages/dashboard.py**: Simple overview page
   - High-level system status and phase completion
   - Used by NavigationController for main GUI

2. **This file (gui/live_monitoring_dashboard.py)**: ADVANCED MONITORING
   - Detailed performance charts with Plotly
   - AlertManager for configurable alert rules
   - LiveDataGenerator for real-time/simulated data
   - Multi-cell comparison and analysis

3. **monitoring/dashboard_api.py**: REST API backend
   - DashboardAPI class for programmatic access
   - FastAPI endpoints for data and control

4. **monitoring/dashboard_frontend.py**: Streamlit API client
   - Frontend for dashboard_api.py backend

Usage:
    >>> from gui.live_monitoring_dashboard import LiveMonitoringDashboard
    >>> dashboard = LiveMonitoringDashboard(n_cells=5)
    >>> dashboard.render_dashboard()  # In Streamlit context

Created: 2025-07-31
Story Points: 13 (High Priority)
Phase: 3 - Advanced Features

Literature References:
1. Few, S. (2009). "Information Dashboard Design"
2. Logan, B.E. (2008). "Microbial Fuel Cells: Methodology and Technology"
3. Rabaey, K. & Verstraete, W. (2005). "Microbial fuel cells"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


class AlertLevel(Enum):
    """Alert severity levels for monitoring system."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""

    timestamp: datetime
    power_output_mW: float
    substrate_concentration_mM: float
    current_density_mA_cm2: float
    voltage_V: float
    biofilm_thickness_um: float
    ph_value: float
    temperature_C: float
    conductivity_S_m: float
    cell_id: str = "Cell_01"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    parameter: str
    threshold_min: float | None = None
    threshold_max: float | None = None
    level: AlertLevel = AlertLevel.WARNING
    message_template: str = "{parameter} is {value:.2f} (threshold: {threshold})"
    enabled: bool = True


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""

    panel_positions: dict[str, dict[str, Any]]
    refresh_interval: int = 5  # seconds
    max_data_points: int = 1000
    auto_scroll: bool = True


class LiveDataGenerator:
    """Simulates live MFC data stream for demonstration.
    In production, this would connect to actual sensors/controllers.
    """

    def __init__(self, base_config: dict[str, Any]) -> None:
        self.base_config = base_config
        self.start_time = datetime.now()
        self.data_history: list[PerformanceMetric] = []

    def generate_realistic_data(self, cell_id: str = "Cell_01") -> PerformanceMetric:
        """Generate realistic MFC performance data."""
        current_time = datetime.now()
        elapsed_hours = (current_time - self.start_time).total_seconds() / 3600
        # Use simulation elapsed time instead of system time
        simulation_time = self.start_time + timedelta(hours=elapsed_hours)

        # Simulate realistic MFC behavior with trends
        base_power = 0.5 + 0.1 * np.sin(elapsed_hours * 0.1)  # Daily cycle
        power_noise = np.random.normal(0, 0.05)
        power_output = max(0, base_power + power_noise)

        # Substrate follows consumption pattern
        base_substrate = 25.0 - (elapsed_hours * 0.1)  # Gradual consumption
        substrate_noise = np.random.normal(0, 1.5)
        substrate_concentration = max(0, base_substrate + substrate_noise)

        # Current density correlated with power
        current_density = power_output * 2.0 + np.random.normal(0, 0.1)

        # Voltage related to power and current
        voltage = power_output / max(current_density, 0.1) + np.random.normal(0, 0.02)

        # Biofilm growth over time
        biofilm_thickness = 40 + elapsed_hours * 0.5 + np.random.normal(0, 2)

        # pH slightly acidic with variations
        ph_value = 6.8 + np.random.normal(0, 0.2)

        # Temperature around mesophilic range
        temperature = 30 + np.random.normal(0, 1)

        # Conductivity related to substrate concentration
        conductivity = 0.001 * (substrate_concentration + 20) + np.random.normal(
            0,
            0.0002,
        )

        return PerformanceMetric(
            timestamp=simulation_time,
            power_output_mW=power_output,
            substrate_concentration_mM=substrate_concentration,
            current_density_mA_cm2=current_density,
            voltage_V=voltage,
            biofilm_thickness_um=biofilm_thickness,
            ph_value=ph_value,
            temperature_C=temperature,
            conductivity_S_m=conductivity,
            cell_id=cell_id,
        )

    def get_historical_data(self, hours: int = 24) -> list[PerformanceMetric]:
        """Generate historical data for specified time period."""
        historical_data = []
        start_time = datetime.now() - timedelta(hours=hours)

        for i in range(hours * 12):  # 5-minute intervals
            timestamp = start_time + timedelta(minutes=i * 5)

            # Generate data with historical timestamp
            original_start = self.start_time
            self.start_time = start_time
            data_point = self.generate_realistic_data()
            data_point.timestamp = timestamp
            self.start_time = original_start

            historical_data.append(data_point)

        return historical_data


class AlertManager:
    """Manages alert rules and notifications."""

    def __init__(self) -> None:
        self.rules: list[AlertRule] = self._get_default_rules()
        self.active_alerts: list[dict[str, Any]] = []

    def _get_default_rules(self) -> list[AlertRule]:
        """Get default alert rules based on MFC literature."""
        return [
            AlertRule(
                parameter="power_output_mW",
                threshold_min=0.1,
                threshold_max=None,
                level=AlertLevel.WARNING,
                message_template="Power output low: {value:.3f} mW (min: {threshold})",
            ),
            AlertRule(
                parameter="substrate_concentration_mM",
                threshold_min=2.0,
                threshold_max=50.0,
                level=AlertLevel.CRITICAL,
                message_template="Substrate concentration critical: {value:.1f} mM",
            ),
            AlertRule(
                parameter="ph_value",
                threshold_min=6.0,
                threshold_max=8.0,
                level=AlertLevel.WARNING,
                message_template="pH out of optimal range: {value:.2f}",
            ),
            AlertRule(
                parameter="temperature_C",
                threshold_min=25.0,
                threshold_max=40.0,
                level=AlertLevel.WARNING,
                message_template="Temperature suboptimal: {value:.1f}°C",
            ),
            AlertRule(
                parameter="voltage_V",
                threshold_min=0.3,
                threshold_max=None,
                level=AlertLevel.INFO,
                message_template="Voltage low: {value:.3f} V (min: {threshold})",
            ),
        ]

    def check_alerts(self, metric: PerformanceMetric) -> list[dict[str, Any]]:
        """Check current metric against all alert rules."""
        new_alerts = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            value = getattr(metric, rule.parameter, None)
            if value is None:
                continue

            triggered = False
            threshold_str = ""

            if rule.threshold_min is not None and value < rule.threshold_min:
                triggered = True
                threshold_str = f"min {rule.threshold_min}"

            if rule.threshold_max is not None and value > rule.threshold_max:
                triggered = True
                threshold_str = f"max {rule.threshold_max}"

            if triggered:
                alert = {
                    "timestamp": metric.timestamp,
                    "cell_id": metric.cell_id,
                    "parameter": rule.parameter,
                    "value": value,
                    "level": rule.level.value,
                    "message": rule.message_template.format(
                        parameter=rule.parameter.replace("_", " ").title(),
                        value=value,
                        threshold=threshold_str,
                    ),
                }
                new_alerts.append(alert)

        return new_alerts

    def add_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """Add new alerts to active list."""
        self.active_alerts.extend(alerts)
        # Keep only recent alerts (last 100)
        self.active_alerts = self.active_alerts[-100:]

    def get_active_alerts(
        self,
        level: AlertLevel | None = None,
    ) -> list[dict[str, Any]]:
        """Get active alerts, optionally filtered by level."""
        if level is None:
            return self.active_alerts
        return [a for a in self.active_alerts if a["level"] == level.value]


class LiveMonitoringDashboard:
    """Main live monitoring dashboard component.
    Implements real-time visualization with customizable layout.
    """

    def __init__(self, n_cells: int = 5) -> None:
        self.n_cells = n_cells
        self.data_generator = LiveDataGenerator({})
        self.alert_manager = AlertManager()
        self.layout_config = self._get_default_layout()

        # Initialize simulation start time
        if "simulation_start_time" not in st.session_state:
            st.session_state.simulation_start_time = datetime.now()

        # Update data generator start time to match simulation
        self.data_generator.start_time = st.session_state.simulation_start_time

        # Initialize session state
        if "monitoring_data" not in st.session_state:
            st.session_state.monitoring_data = []
        if "monitoring_alerts" not in st.session_state:
            st.session_state.monitoring_alerts = []
        if "last_update" not in st.session_state:
            st.session_state.last_update = datetime.now()
        if "monitoring_n_cells" not in st.session_state:
            st.session_state.monitoring_n_cells = n_cells

    def reset_simulation_time(self) -> None:
        """Reset simulation start time for new simulation."""
        st.session_state.simulation_start_time = datetime.now()
        self.data_generator.start_time = st.session_state.simulation_start_time
        # Clear existing data since we're starting fresh
        st.session_state.monitoring_data = []
        st.session_state.monitoring_alerts = []
        st.session_state.last_update = datetime.now()

    def _get_default_layout(self) -> DashboardLayout:
        """Get default dashboard layout configuration."""
        return DashboardLayout(
            panel_positions={
                "kpi_overview": {"row": 0, "col": 0, "width": 12, "height": 2},
                "power_trend": {"row": 1, "col": 0, "width": 6, "height": 4},
                "substrate_trend": {"row": 1, "col": 6, "width": 6, "height": 4},
                "multicell_comparison": {"row": 2, "col": 0, "width": 8, "height": 4},
                "alerts_panel": {"row": 2, "col": 8, "width": 4, "height": 4},
            },
            refresh_interval=5,
            max_data_points=500,
        )

    def update_data(self, force_update: bool = False) -> bool:
        """Update monitoring data with new measurements."""
        current_time = datetime.now()

        # Initialize last_update if it doesn't exist
        if (
            "last_update" not in st.session_state
            or st.session_state.last_update is None
        ):
            st.session_state.last_update = current_time

        time_since_update = (
            current_time - st.session_state.last_update
        ).total_seconds()

        if force_update or time_since_update >= self.layout_config.refresh_interval:
            # Generate new data points for multiple cells
            # Use the actual number of cells from session state or instance
            n_cells = st.session_state.get("monitoring_n_cells", self.n_cells)
            cell_ids = [f"Cell_{i + 1:02d}" for i in range(n_cells)]
            new_metrics = []

            for cell_id in cell_ids:
                metric = self.data_generator.generate_realistic_data(cell_id)
                new_metrics.append(metric)

                # Check for alerts - store ONLY in session state to avoid duplicates
                alerts = self.alert_manager.check_alerts(metric)
                if alerts:
                    st.session_state.monitoring_alerts.extend(alerts)

            # Add to session state
            st.session_state.monitoring_data.extend(new_metrics)

            # Limit data history
            max_points = self.layout_config.max_data_points
            if len(st.session_state.monitoring_data) > max_points:
                st.session_state.monitoring_data = st.session_state.monitoring_data[
                    -max_points:
                ]

            # Update timestamp
            st.session_state.last_update = current_time

            return True  # Data was updated

        return False  # No update needed

    def render_kpi_overview(self) -> None:
        """Render key performance indicators overview."""
        st.markdown("### 📊 Live Performance Indicators")

        if not st.session_state.monitoring_data:
            st.info("⏳ Waiting for live data...")
            return

        # Get latest data for each cell
        latest_data = {}
        for metric in reversed(st.session_state.monitoring_data):
            if metric.cell_id not in latest_data:
                latest_data[metric.cell_id] = metric

        if not latest_data:
            return

        # Calculate overall metrics
        total_power = sum(m.power_output_mW for m in latest_data.values())
        avg_substrate = np.mean(
            [m.substrate_concentration_mM for m in latest_data.values()],
        )
        avg_ph = np.mean([m.ph_value for m in latest_data.values()])
        avg_temp = np.mean([m.temperature_C for m in latest_data.values()])

        # Display KPIs in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_power = np.random.normal(0, 0.1)  # Mock trend
            st.metric(
                label="🔋 Total Power",
                value=f"{total_power:.3f} mW",
                delta=f"{delta_power:.3f} mW",
            )

        with col2:
            delta_substrate = np.random.normal(0, 0.5)  # Mock trend
            st.metric(
                label="🧪 Avg Substrate",
                value=f"{avg_substrate:.1f} mM",
                delta=f"{delta_substrate:.1f} mM",
            )

        with col3:
            delta_ph = np.random.normal(0, 0.1)  # Mock trend
            st.metric(label="⚗️ Avg pH", value=f"{avg_ph:.2f}", delta=f"{delta_ph:.2f}")

        with col4:
            delta_temp = np.random.normal(0, 0.5)  # Mock trend
            st.metric(
                label="🌡️ Avg Temperature",
                value=f"{avg_temp:.1f}°C",
                delta=f"{delta_temp:.1f}°C",
            )

    def render_power_trend_chart(self) -> None:
        """Render real-time power output trend chart."""
        st.markdown("### ⚡ Power Output Trends")

        if not st.session_state.monitoring_data:
            st.info("No data available for trend analysis")
            return

        # Convert to DataFrame for plotting
        df_data = []
        for metric in st.session_state.monitoring_data:
            df_data.append(
                {
                    "timestamp": metric.timestamp,
                    "power_mW": metric.power_output_mW,
                    "cell_id": metric.cell_id,
                },
            )

        df = pd.DataFrame(df_data)

        # Create interactive time series plot
        fig = px.line(
            df,
            x="timestamp",
            y="power_mW",
            color="cell_id",
            title="Real-Time Power Output by Cell",
            labels={
                "timestamp": "Simulation Time",
                "power_mW": "Power Output (mW)",
                "cell_id": "MFC Cell",
            },
        )

        # Customize layout
        fig.update_layout(
            xaxis_title="Simulation Time",
            yaxis_title="Power Output (mW)",
            hovermode="x unified",
            showlegend=True,
            height=400,
        )

        # Add threshold line
        fig.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="red",
            annotation_text="Min Power Threshold",
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_substrate_trend_chart(self) -> None:
        """Render substrate concentration trend chart."""
        st.markdown("### 🧪 Substrate Concentration Trends")

        if not st.session_state.monitoring_data:
            st.info("No data available for trend analysis")
            return

        # Convert to DataFrame
        df_data = []
        for metric in st.session_state.monitoring_data:
            df_data.append(
                {
                    "timestamp": metric.timestamp,
                    "substrate_mM": metric.substrate_concentration_mM,
                    "cell_id": metric.cell_id,
                },
            )

        df = pd.DataFrame(df_data)

        # Create multi-line chart
        fig = px.line(
            df,
            x="timestamp",
            y="substrate_mM",
            color="cell_id",
            title="Substrate Consumption Over Time",
            labels={
                "timestamp": "Simulation Time",
                "substrate_mM": "Substrate Concentration (mM)",
                "cell_id": "MFC Cell",
            },
        )

        # Add optimal range
        fig.add_hrect(
            y0=15,
            y1=35,
            line_width=0,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Optimal Range",
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def render_multicell_comparison(self) -> None:
        """Render multi-cell performance comparison."""
        st.markdown("### 🔬 Multi-Cell Performance Comparison")

        if not st.session_state.monitoring_data:
            st.info("No data available for comparison")
            return

        # Get latest metrics for each cell
        latest_by_cell = {}
        for metric in reversed(st.session_state.monitoring_data):
            if metric.cell_id not in latest_by_cell:
                latest_by_cell[metric.cell_id] = metric

        if len(latest_by_cell) < 2:
            st.info("Multiple cells needed for comparison")
            return

        # Create comparison chart
        cells = list(latest_by_cell.keys())
        power_values = [latest_by_cell[cell].power_output_mW for cell in cells]
        substrate_values = [
            latest_by_cell[cell].substrate_concentration_mM for cell in cells
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Power Output", "Substrate Concentration"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Power comparison
        fig.add_trace(
            go.Bar(
                x=cells,
                y=power_values,
                name="Power (mW)",
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

        # Substrate comparison
        fig.add_trace(
            go.Bar(
                x=cells,
                y=substrate_values,
                name="Substrate (mM)",
                marker_color="lightgreen",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Current Cell Performance Comparison",
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_alerts_panel(self) -> None:
        """Render active alerts and notifications."""
        st.markdown("### 🚨 Active Alerts")

        # Get recent alerts ONLY from session state to avoid duplicates
        recent_alerts = list(st.session_state.monitoring_alerts[-20:])

        if not recent_alerts:
            st.success("✅ All systems normal - No active alerts")
            return

        # Group alerts by level
        critical_alerts = [a for a in recent_alerts if a["level"] == "critical"]
        warning_alerts = [a for a in recent_alerts if a["level"] == "warning"]
        info_alerts = [a for a in recent_alerts if a["level"] == "info"]

        # Display critical alerts
        if critical_alerts:
            st.error(f"🔴 {len(critical_alerts)} Critical Alerts")
            for alert in critical_alerts[-3:]:  # Show last 3
                st.error(f"**{alert['cell_id']}**: {alert['message']}")

        # Display warnings
        if warning_alerts:
            st.warning(f"🟡 {len(warning_alerts)} Warning Alerts")
            for alert in warning_alerts[-3:]:  # Show last 3
                st.warning(f"**{alert['cell_id']}**: {alert['message']}")

        # Display info alerts
        if info_alerts:
            st.info(f"🔵 {len(info_alerts)} Info Alerts")
            for alert in info_alerts[-2:]:  # Show last 2
                st.info(f"**{alert['cell_id']}**: {alert['message']}")

    def render_settings_panel(self) -> None:
        """Render dashboard configuration settings."""
        with st.expander("⚙️ Dashboard Settings"):
            col1, col2 = st.columns(2)

            with col1:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=1,
                    max_value=30,
                    value=self.layout_config.refresh_interval,
                )

                max_data_points = st.slider(
                    "Max Data Points",
                    min_value=100,
                    max_value=2000,
                    value=self.layout_config.max_data_points,
                )

            with col2:
                show_historical = st.checkbox("📊 Show Historical Data", value=False)

            # Update configuration
            self.layout_config.refresh_interval = refresh_interval
            self.layout_config.max_data_points = max_data_points

            # Store refresh interval in session state for other components
            st.session_state["live_monitoring_refresh"] = refresh_interval

            if show_historical and st.button("Load Historical Data (24h)"):
                historical_data = self.data_generator.get_historical_data(24)
                st.session_state.monitoring_data.extend(historical_data)
                st.success("Historical data loaded!")

    def render_dashboard(self) -> None:
        """Render the complete live monitoring dashboard."""
        st.markdown("## 📡 Live Performance Monitoring Dashboard")
        st.markdown("*Real-time MFC performance tracking with <5 second updates*")

        # Auto-refresh toggle and manual refresh
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh enabled", value=True)
        with col2:
            if st.button("🔄 Refresh Now"):
                self.update_data(force_update=True)
                st.rerun()
        with col3:
            if st.button("🗑️ Clear Data"):
                st.session_state.monitoring_data = []
                st.session_state.monitoring_alerts = []
                st.rerun()

        # Update data based on refresh settings
        if auto_refresh:
            data_updated = self.update_data()
            if data_updated:
                time.sleep(0.1)  # Brief pause for data processing
                st.rerun()

        # Render dashboard panels
        self.render_kpi_overview()

        # Main charts in columns
        col1, col2 = st.columns(2)
        with col1:
            self.render_power_trend_chart()
        with col2:
            self.render_substrate_trend_chart()

        # Lower section
        col1, col2 = st.columns([2, 1])
        with col1:
            self.render_multicell_comparison()
        with col2:
            self.render_alerts_panel()

        # Settings panel
        self.render_settings_panel()

        # Status information
        if st.session_state.monitoring_data:
            latest_update = max(
                m.timestamp for m in st.session_state.monitoring_data
            )
            time_ago = (datetime.now() - latest_update).total_seconds()
            data_count = len(st.session_state.monitoring_data)
            st.caption(
                f"Last updated: {time_ago:.1f}s ago | Data points: {data_count}",
            )


# Export main class for integration
__all__ = [
    "AlertManager",
    "DashboardLayout",
    "LiveMonitoringDashboard",
    "PerformanceMetric",
]
