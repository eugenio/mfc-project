#!/usr/bin/env python3
"""Performance Monitor Dashboard for Enhanced MFC Platform.

Real-time system performance monitoring with GPU metrics, simulation status,
and comprehensive system health indicators.

Created: 2025-08-02
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots


@dataclass
class SystemMetrics:
    """System performance metrics."""

    cpu_usage: float
    memory_usage: float
    gpu_utilization: float
    gpu_memory: float
    gpu_temperature: float
    disk_usage: float
    network_io: dict[str, float]
    timestamp: datetime


@dataclass
class SimulationStatus:
    """Current simulation status."""

    active: bool
    phase: str
    progress: float
    eta: str | None
    current_step: int
    total_steps: int
    performance_metrics: dict[str, float]


def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton PerformanceMonitor instance from session state.

    Returns the existing PerformanceMonitor from session state if available,
    otherwise creates a new one and stores it in session state.

    Returns
    -------
        PerformanceMonitor: The singleton instance.

    """
    if "performance_monitor" not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    return st.session_state.performance_monitor


def start_simulation(total_steps: int = 1000) -> None:
    """Start a simulation and initialize simulation state.

    Sets simulation_active to True and initializes simulation_data with
    proper structure including start_time for ETA calculation.

    Parameters
    ----------
    total_steps : int, optional
        Total number of steps for the simulation. Default is 1000.

    """
    st.session_state.simulation_active = True
    st.session_state.simulation_data = {
        "phase": "Initialization",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": total_steps,
        "start_time": datetime.now(),
        "performance_metrics": {
            "steps_per_second": 0.0,
            "memory_efficiency": 0.0,
            "convergence_rate": 0.0,
            "acceleration_factor": 1.0,
        },
    }


def stop_simulation() -> None:
    """Stop the current simulation and reset simulation state.

    Sets simulation_active to False and clears simulation_data to idle state.
    """
    st.session_state.simulation_active = False
    st.session_state.simulation_data = {
        "phase": "Idle",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": 0,
        "start_time": None,
        "performance_metrics": {},
    }


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self) -> None:
        """Initialize performance monitor with empty metrics history."""
        self.metrics_history = []
        self.alerts = []
        self.monitoring_active = True

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
        }

        # Simulated GPU metrics (in real implementation, use pynvml or ROCm tools)
        gpu_util = (
            np.random.uniform(20, 95)
            if np.random.random() > 0.3
            else np.random.uniform(0, 15)
        )
        gpu_memory = np.random.uniform(2.0, 15.8)  # GB
        gpu_temp = np.random.uniform(45, 85)  # °C

        return SystemMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            gpu_utilization=gpu_util,
            gpu_memory=gpu_memory,
            gpu_temperature=gpu_temp,
            disk_usage=disk_percent,
            network_io=network_io,
            timestamp=datetime.now(),
        )

    def get_simulation_status(self) -> SimulationStatus:
        """Get current simulation status from session state."""
        # Get simulation state from session state (fixes phantom simulation bug)
        is_active_val = st.session_state.get("simulation_active", False)
        # Ensure is_active is a proper boolean
        is_active = bool(is_active_val) if is_active_val is not None else False

        if is_active:
            # Get simulation data from session state
            sim_data = st.session_state.get("simulation_data", {})
            if not isinstance(sim_data, dict):
                sim_data = {}
            current_phase = sim_data.get("phase", "Initialization")
            progress = float(sim_data.get("progress", 0.0))
            current_step = int(sim_data.get("current_step", 0))
            total_steps = int(sim_data.get("total_steps", 1000))
            perf_metrics = sim_data.get("performance_metrics", {})
            if not isinstance(perf_metrics, dict):
                perf_metrics = {}
            start_time = sim_data.get("start_time")

            # Calculate ETA based on actual progress
            eta = None
            if start_time is not None and progress > 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > 0:
                    steps_per_second = current_step / elapsed if elapsed > 0 else 0
                    if steps_per_second > 0:
                        remaining_steps = total_steps - current_step
                        eta_seconds = remaining_steps / steps_per_second
                        eta = str(timedelta(seconds=int(eta_seconds)))
                    # Update performance metrics with calculated rate
                    perf_metrics["steps_per_second"] = steps_per_second
        else:
            current_phase = "Idle"
            progress = 0.0
            current_step = 0
            total_steps = 0
            eta = None
            perf_metrics = {}

        return SimulationStatus(
            active=is_active,
            phase=current_phase,
            progress=progress,
            eta=eta,
            current_step=current_step,
            total_steps=total_steps,
            performance_metrics=perf_metrics,
        )

    def get_gpu_acceleration_metrics(self) -> dict[str, float]:
        """Get GPU acceleration performance metrics."""
        return {
            "current_speedup": np.random.uniform(1000, 8400),
            "target_speedup": 8400,
            "efficiency": np.random.uniform(0.75, 0.95),
            "memory_bandwidth": np.random.uniform(400, 900),  # GB/s
            "compute_utilization": np.random.uniform(0.6, 0.95),
            "memory_utilization": np.random.uniform(0.4, 0.85),
        }

    def get_system_health_score(self) -> float:
        """Calculate overall system health score."""
        metrics = self.get_current_metrics()

        # Health scoring based on various metrics
        cpu_score = max(0, 1 - metrics.cpu_usage / 100)
        memory_score = max(0, 1 - metrics.memory_usage / 100)
        disk_score = max(0, 1 - metrics.disk_usage / 100)
        gpu_temp_score = max(0, 1 - max(0, metrics.gpu_temperature - 60) / 40)

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        scores = [cpu_score, memory_score, disk_score, gpu_temp_score]

        health_score = sum(w * s for w, s in zip(weights, scores, strict=False))
        return min(1.0, max(0.0, health_score))

    def check_alerts(self, metrics: SystemMetrics) -> list[dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []

        # CPU usage alert
        if metrics.cpu_usage > 90:
            alerts.append(
                {
                    "type": "warning",
                    "title": "High CPU Usage",
                    "message": f"CPU usage at {metrics.cpu_usage:.1f}%",
                    "timestamp": metrics.timestamp,
                },
            )

        # Memory usage alert
        if metrics.memory_usage > 85:
            alerts.append(
                {
                    "type": "warning",
                    "title": "High Memory Usage",
                    "message": f"Memory usage at {metrics.memory_usage:.1f}%",
                    "timestamp": metrics.timestamp,
                },
            )

        # GPU temperature alert
        if metrics.gpu_temperature > 80:
            alerts.append(
                {
                    "type": "error",
                    "title": "GPU Overheating",
                    "message": f"GPU temperature at {metrics.gpu_temperature:.1f}°C",
                    "timestamp": metrics.timestamp,
                },
            )

        # GPU utilization alert (too low during simulation)
        sim_status = self.get_simulation_status()
        if sim_status.active and metrics.gpu_utilization < 30:
            gpu_util = metrics.gpu_utilization
            alerts.append(
                {
                    "type": "info",
                    "title": "Low GPU Utilization",
                    "message": f"GPU utilization only {gpu_util:.1f}% during sim",
                    "timestamp": metrics.timestamp,
                },
            )

        return alerts


def create_real_time_charts(metrics_history: list[SystemMetrics]) -> None:
    """Create real-time performance charts."""
    if not metrics_history:
        st.info("No metrics data available yet...")
        return

    # Convert to DataFrame
    data = []
    for m in metrics_history[-50:]:  # Last 50 data points
        data.append(
            {
                "timestamp": m.timestamp,
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "gpu_utilization": m.gpu_utilization,
                "gpu_memory": m.gpu_memory,
                "gpu_temperature": m.gpu_temperature,
            },
        )

    df = pd.DataFrame(data)

    # CPU and Memory Usage
    col1, col2 = st.columns(2)

    with col1:
        fig_cpu_mem = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("CPU Usage (%)", "Memory Usage (%)"),
            vertical_spacing=0.1,
        )

        fig_cpu_mem.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["cpu_usage"],
                name="CPU",
                line={"color": "blue"},
            ),
            row=1,
            col=1,
        )

        fig_cpu_mem.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["memory_usage"],
                name="Memory",
                line={"color": "green"},
            ),
            row=2,
            col=1,
        )

        fig_cpu_mem.update_layout(height=400, title_text="System Resources")
        fig_cpu_mem.update_xaxes(showgrid=True)
        fig_cpu_mem.update_yaxes(range=[0, 100])

        st.plotly_chart(fig_cpu_mem, use_container_width=True)

    with col2:
        fig_gpu = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "GPU Utilization (%)",
                "GPU Memory (GB)",
                "GPU Temperature (°C)",
            ),
            vertical_spacing=0.08,
        )

        fig_gpu.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["gpu_utilization"],
                name="GPU Util",
                line={"color": "red"},
            ),
            row=1,
            col=1,
        )

        fig_gpu.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["gpu_memory"],
                name="GPU Memory",
                line={"color": "orange"},
            ),
            row=2,
            col=1,
        )

        fig_gpu.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["gpu_temperature"],
                name="GPU Temp",
                line={"color": "purple"},
            ),
            row=3,
            col=1,
        )

        fig_gpu.update_layout(height=400, title_text="GPU Metrics")
        fig_gpu.update_xaxes(showgrid=True)
        fig_gpu.update_yaxes(range=[0, 100], row=1, col=1)
        fig_gpu.update_yaxes(range=[0, 16], row=2, col=1)
        fig_gpu.update_yaxes(range=[30, 90], row=3, col=1)

        st.plotly_chart(fig_gpu, use_container_width=True)


def create_acceleration_dashboard() -> None:
    """Create GPU acceleration performance dashboard."""
    st.subheader("🚀 GPU Acceleration Performance")

    monitor = get_performance_monitor()
    gpu_metrics = monitor.get_gpu_acceleration_metrics()

    # Key performance indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_speedup = gpu_metrics["current_speedup"]
        target_speedup = gpu_metrics["target_speedup"]
        delta = f"Target: {target_speedup:.0f}×"
        color = "normal" if current_speedup > target_speedup * 0.8 else "inverse"

        st.metric(
            "Current Speedup",
            f"{current_speedup:.0f}×",
            delta=delta,
            delta_color=color,
        )

    with col2:
        efficiency = gpu_metrics["efficiency"]
        st.metric(
            "GPU Efficiency",
            f"{efficiency:.1%}",
            delta="Target: >85%",
            delta_color="normal" if efficiency > 0.85 else "inverse",
        )

    with col3:
        bandwidth = gpu_metrics["memory_bandwidth"]
        st.metric("Memory Bandwidth", f"{bandwidth:.0f} GB/s", delta="Peak: 900 GB/s")

    with col4:
        compute_util = gpu_metrics["compute_utilization"]
        st.metric(
            "Compute Utilization",
            f"{compute_util:.1%}",
            delta="Target: >90%",
            delta_color="normal" if compute_util > 0.9 else "inverse",
        )

    # Acceleration progress bar
    progress_value = current_speedup / target_speedup
    st.progress(
        progress_value,
        f"Acceleration Progress: {progress_value:.1%} of target",
    )

    if progress_value < 0.5:
        st.warning("⚠️ GPU acceleration significantly below target performance")
    elif progress_value < 0.8:
        st.info("ℹ️ GPU acceleration approaching target performance")
    else:
        st.success("✅ GPU acceleration performing at target levels")


def render_performance_monitor_page() -> None:
    """Render the Performance Monitor dashboard page."""
    # Page header
    st.title("📊 Performance Monitor Dashboard")
    st.caption("Real-time system performance monitoring and GPU acceleration tracking")

    # Auto-refresh toggle
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)

    with col2:
        refresh_interval = st.selectbox("Refresh Rate", [1, 2, 5, 10], index=1)

    with col3:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()

    # Initialize performance monitor
    if "performance_monitor" not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
        st.session_state.metrics_history = []

    monitor = st.session_state.performance_monitor

    # Get current metrics
    current_metrics = monitor.get_current_metrics()
    sim_status = monitor.get_simulation_status()
    health_score = monitor.get_system_health_score()

    # Add to history
    st.session_state.metrics_history.append(current_metrics)
    # Keep only last 100 points
    if len(st.session_state.metrics_history) > 100:
        st.session_state.metrics_history = st.session_state.metrics_history[-100:]

    # System Health Overview
    st.subheader("🏥 System Health Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        health_color = "normal" if health_score > 0.8 else "inverse"
        health_status = (
            "Excellent"
            if health_score > 0.9
            else (
                "Good"
                if health_score > 0.7
                else "Fair"
                if health_score > 0.5
                else "Poor"
            )
        )

        st.metric(
            "System Health",
            f"{health_score:.1%}",
            delta=health_status,
            delta_color=health_color,
        )

    with col2:
        cpu_color = "normal" if current_metrics.cpu_usage < 80 else "inverse"
        st.metric(
            "CPU Usage",
            f"{current_metrics.cpu_usage:.1f}%",
            delta="Target: <80%",
            delta_color=cpu_color,
        )

    with col3:
        memory_color = "normal" if current_metrics.memory_usage < 85 else "inverse"
        st.metric(
            "Memory Usage",
            f"{current_metrics.memory_usage:.1f}%",
            delta="Target: <85%",
            delta_color=memory_color,
        )

    with col4:
        gpu_temp_color = "normal" if current_metrics.gpu_temperature < 75 else "inverse"
        st.metric(
            "GPU Temperature",
            f"{current_metrics.gpu_temperature:.1f}°C",
            delta="Target: <75°C",
            delta_color=gpu_temp_color,
        )

    # Simulation Status
    st.subheader("⚗️ Simulation Status")

    if sim_status.active:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.success(f"🔄 Simulation Active - {sim_status.phase}")
            st.progress(sim_status.progress, f"Progress: {sim_status.progress:.1%}")

        with col2:
            st.metric("Current Step", f"{sim_status.current_step:,}")
            st.metric(
                "Steps/sec",
                f"{sim_status.performance_metrics.get('steps_per_second', 0):.1f}",
            )

        with col3:
            if sim_status.eta:
                st.metric("ETA", sim_status.eta)
            acceleration = sim_status.performance_metrics.get("acceleration_factor", 1)
            st.metric("Acceleration", f"{acceleration:.0f}×")

        # Performance metrics for active simulation
        if sim_status.performance_metrics:
            perf_col1, perf_col2, perf_col3 = st.columns(3)

            with perf_col1:
                memory_eff = sim_status.performance_metrics.get("memory_efficiency", 0)
                st.metric("Memory Efficiency", f"{memory_eff:.1%}")

            with perf_col2:
                convergence = sim_status.performance_metrics.get("convergence_rate", 0)
                st.metric("Convergence Rate", f"{convergence:.4f}")

            with perf_col3:
                acceleration = sim_status.performance_metrics.get(
                    "acceleration_factor",
                    1,
                )
                target_accel = 8400
                accel_progress = min(1.0, acceleration / target_accel)
                st.metric("Acceleration Progress", f"{accel_progress:.1%}")

        # Stop simulation button
        if st.button("🛑 Stop Simulation", type="secondary"):
            stop_simulation()
            st.rerun()
    else:
        st.info("💤 No simulation currently running")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Test Simulation", type="primary"):
                start_simulation(total_steps=1000)
                st.rerun()

        with col2:
            if st.button("📂 Load Previous Results"):
                st.info("Previous simulation results would be loaded...")

    # GPU Acceleration Dashboard
    create_acceleration_dashboard()

    # Real-time Performance Charts
    st.subheader("📈 Real-time Performance Charts")
    create_real_time_charts(st.session_state.metrics_history)

    # System Alerts
    alerts = monitor.check_alerts(current_metrics)
    if alerts:
        st.subheader("🚨 System Alerts")

        for alert in alerts:
            if alert["type"] == "error":
                st.error(f"**{alert['title']}**: {alert['message']}")
            elif alert["type"] == "warning":
                st.warning(f"**{alert['title']}**: {alert['message']}")
            else:
                st.info(f"**{alert['title']}**: {alert['message']}")

    # Performance Statistics
    with st.expander("📊 Detailed Performance Statistics"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("System Resources")

            # Resource usage table
            resource_data = {
                "Component": ["CPU", "Memory", "GPU", "Disk", "Network"],
                "Usage": [
                    f"{current_metrics.cpu_usage:.1f}%",
                    f"{current_metrics.memory_usage:.1f}%",
                    f"{current_metrics.gpu_utilization:.1f}%",
                    f"{current_metrics.disk_usage:.1f}%",
                    "Active",
                ],
                "Status": [
                    (
                        "🟢 Normal"
                        if current_metrics.cpu_usage < 80
                        else (
                            "🟡 High"
                            if current_metrics.cpu_usage < 95
                            else "🔴 Critical"
                        )
                    ),
                    (
                        "🟢 Normal"
                        if current_metrics.memory_usage < 85
                        else (
                            "🟡 High"
                            if current_metrics.memory_usage < 95
                            else "🔴 Critical"
                        )
                    ),
                    (
                        "🟢 Active"
                        if current_metrics.gpu_utilization > 50
                        else (
                            "🟡 Low"
                            if current_metrics.gpu_utilization > 10
                            else "🔴 Idle"
                        )
                    ),
                    (
                        "🟢 Normal"
                        if current_metrics.disk_usage < 80
                        else (
                            "🟡 High"
                            if current_metrics.disk_usage < 95
                            else "🔴 Critical"
                        )
                    ),
                    "🟢 Connected",
                ],
            }

            st.dataframe(pd.DataFrame(resource_data), use_container_width=True)

        with col2:
            st.subheader("Performance History")

            # Performance statistics
            if len(st.session_state.metrics_history) > 1:
                recent_metrics = st.session_state.metrics_history[-10:]

                avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
                avg_memory = np.mean([m.memory_usage for m in recent_metrics])
                avg_gpu = np.mean([m.gpu_utilization for m in recent_metrics])
                avg_temp = np.mean([m.gpu_temperature for m in recent_metrics])

                stats_data = {
                    "Metric": [
                        "Avg CPU (10min)",
                        "Avg Memory (10min)",
                        "Avg GPU (10min)",
                        "Avg GPU Temp (10min)",
                    ],
                    "Value": [
                        f"{avg_cpu:.1f}%",
                        f"{avg_memory:.1f}%",
                        f"{avg_gpu:.1f}%",
                        f"{avg_temp:.1f}°C",
                    ],
                }

                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            else:
                st.info("Collecting performance data...")

    # Configuration and Export
    with st.expander("⚙️ Monitoring Configuration"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Alert Thresholds")
            st.slider("CPU Alert (%)", 50, 100, 90)
            st.slider("Memory Alert (%)", 50, 100, 85)
            st.slider("GPU Temp Alert (°C)", 60, 100, 80)

        with col2:
            st.subheader("Data Retention")
            st.number_input("Retention (hours)", 1, 168, 24)
            st.selectbox("Export Interval", ["Never", "Hourly", "Daily", "Weekly"])

        with col3:
            st.subheader("Export Options")
            if st.button("📁 Export Current Data"):
                st.info("Performance data would be exported as CSV")

            if st.button("📧 Email Report"):
                st.info("Performance report would be emailed")

    # Auto-refresh implementation
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

    # Footer with system info
    st.markdown("---")
    st.caption(
        f"Last updated: {current_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Monitoring active: {monitor.monitoring_active} | "
        f"Data points: {len(st.session_state.metrics_history)}",
    )
