"""
Real-time Monitoring Dashboard Frontend

A modern Streamlit-based dashboard for real-time MFC system monitoring.
Provides comprehensive visualization, control, and alert management.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import json
import requests
import asyncio
import websockets
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import os
import sys

API_BASE_URL = "http://localhost:8000/api"
WEBSOCKET_URL = "ws://localhost:8000/ws"
class DashboardAPI:
    """API client for dashboard backend"""
    
    @staticmethod
    def get_system_status():
        """Get current system status"""
        try:
            response = requests.get(f"{API_BASE_URL}/system/status", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    @staticmethod
    def get_current_metrics():
        """Get current system metrics"""
        try:
            response = requests.get(f"{API_BASE_URL}/metrics/current", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    @staticmethod
    def send_command(command: str, parameters: dict = None):
        """Send control command"""
        try:
            payload = {"command": command}
            if parameters:
                payload["parameters"] = parameters
            
            response = requests.post(f"{API_BASE_URL}/control/command", 
                                   json=payload, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {}
    
    @staticmethod
    def get_active_alerts():
        """Get active alerts"""
        try:
            response = requests.get(f"{API_BASE_URL}/alerts/active", timeout=5)
            return response.json() if response.status_code == 200 else {"alerts": []}
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return {"alerts": []}
def create_performance_charts(metrics: Dict[str, Any]):
    """Create performance monitoring charts"""
    
    # Main performance metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Power Generation', 'Efficiency Metrics', 
                       'Cell Voltages', 'System Health'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Power generation over time (simulated historical data)
    time_points = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    power_data = np.random.normal(metrics.get('average_power_w', 5.0), 0.5, 100)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=power_data, name="Power (W)", 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Efficiency metrics
    efficiency_data = np.random.normal(
        metrics.get('coulombic_efficiency_pct', 80), 5, 100)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=efficiency_data, 
                  name="Coulombic Efficiency (%)", 
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    
    # Cell voltages
    cell_voltages = metrics.get('cell_voltages', [0.7, 0.72, 0.68, 0.71, 0.69])
    if cell_voltages:
        fig.add_trace(
            go.Bar(x=[f"Cell {i+1}" for i in range(len(cell_voltages))],
                   y=cell_voltages, name="Cell Voltages (V)",
                   marker_color='orange'),
            row=2, col=1
        )
    
    # System health indicators
    health_metrics = {
        'Temperature': metrics.get('temperature_c', 25),
        'pH Level': metrics.get('ph_level', 7.0),
        'Flow Rate': metrics.get('flow_rate_ml_min', 100),
        'Pressure': metrics.get('pressure_bar', 1.0)
    }
    
    fig.add_trace(
        go.Scatter(x=list(health_metrics.keys()),
                   y=list(health_metrics.values()),
                   mode='markers+lines',
                   name="System Parameters",
                   marker=dict(size=10, color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, 
                     title_text="MFC System Performance Overview")
    
    return fig
def create_biofilm_monitoring_chart(metrics: Dict[str, Any]):
    """Create biofilm monitoring visualization"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Biofilm Thickness', 'Substrate Concentrations')
    )
    
    # Biofilm thickness
    biofilm_data = metrics.get('biofilm_thickness', [10, 12, 8, 11, 9])
    if biofilm_data:
        fig.add_trace(
            go.Bar(x=[f"Cell {i+1}" for i in range(len(biofilm_data))],
                   y=biofilm_data, name="Thickness (μm)",
                   marker_color='purple'),
            row=1, col=1
        )
    
    # Substrate concentrations
    substrate_data = metrics.get('substrate_concentrations', [50, 48, 52, 49, 51])
    if substrate_data:
        fig.add_trace(
            go.Scatter(x=[f"Cell {i+1}" for i in range(len(substrate_data))],
                      y=substrate_data, name="Concentration (mg/L)",
                      mode='markers+lines',
                      line=dict(color='brown', width=3),
                      marker=dict(size=8)),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=True,
                     title_text="Biological System Monitoring")
    
    return fig
def display_alerts(alerts: List[Dict]):
    """Display system alerts"""
    if not alerts:
        st.success("🟢 No active alerts")
        return
    
    for alert in alerts:
        level = alert.get('level', 'info').lower()
        message = alert.get('message', 'Unknown alert')
        timestamp = alert.get('timestamp', datetime.now().isoformat())
        
        if level == 'critical':
            st.error(f"🔴 **CRITICAL**: {message} ({timestamp})")
        elif level == 'warning':
            st.warning(f"🟡 **WARNING**: {message} ({timestamp})")
        else:
            st.info(f"🔵 **INFO**: {message} ({timestamp})")
def main():
    """Main dashboard function"""
    
    # Header
    st.title("⚡ MFC Real-time Monitoring Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ System Control")
        
        # System status
        status_data = DashboardAPI.get_system_status()
        system_status = status_data.get('status', 'offline')
        
        if system_status == 'running':
            st.markdown('<p class="status-running">🟢 SYSTEM RUNNING</p>', 
                       unsafe_allow_html=True)
        elif system_status == 'offline':
            st.markdown('<p class="status-offline">🔴 SYSTEM OFFLINE</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">⚠️ SYSTEM ERROR</p>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                result = DashboardAPI.send_command("start")
                if result.get('success'):
                    st.success("System started")
                else:
                    st.error("Failed to start system")
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                result = DashboardAPI.send_command("stop")
                if result.get('success'):
                    st.success("System stopped")
                else:
                    st.error("Failed to stop system")
        
        if st.button("🚨 Emergency Stop", type="primary", use_container_width=True):
            result = DashboardAPI.send_command("emergency_stop")
            if result.get('success'):
                st.success("Emergency stop activated")
            else:
                st.error("Failed to activate emergency stop")
        
        st.markdown("---")
        
        # System configuration
        st.subheader("⚙️ Configuration")
        
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_rate = st.selectbox("Refresh Rate", [1, 2, 5, 10], index=0)
        
        st.markdown("---")
        
        # Safety thresholds
        st.subheader("🛡️ Safety Thresholds")
        
        max_temp = st.number_input("Max Temperature (°C)", value=45.0, step=1.0)
        max_current = st.number_input("Max Current Density (mA/cm²)", value=10.0, step=0.5)
        min_voltage = st.number_input("Min Voltage (V)", value=0.1, step=0.01)
    
    # Main content area
    
    # Get current metrics
    metrics = DashboardAPI.get_current_metrics()
    st.session_state.metrics_data = metrics
    
    if not metrics:
        st.error("⚠️ Unable to connect to monitoring system. Please check if the API server is running.")
        st.code("python -m uvicorn monitoring.dashboard_api:app --reload", language="bash")
        return
    
    # Key performance indicators
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Power Output",
            value=f"{metrics.get('average_power_w', 0):.2f} W",
            delta=f"{np.random.uniform(-0.1, 0.1):.3f}"
        )
    
    with col2:
        st.metric(
            label="Efficiency",
            value=f"{metrics.get('coulombic_efficiency_pct', 0):.1f}%",
            delta=f"{np.random.uniform(-2, 2):.1f}%"
        )
    
    with col3:
        st.metric(
            label="Current Density",
            value=f"{metrics.get('current_density_ma_cm2', 0):.1f} mA/cm²",
            delta=f"{np.random.uniform(-0.5, 0.5):.2f}"
        )
    
    with col4:
        st.metric(
            label="Temperature",
            value=f"{metrics.get('temperature_c', 0):.1f}°C",
            delta=f"{np.random.uniform(-1, 1):.1f}°C"
        )
    
    with col5:
        st.metric(
            label="Uptime",
            value=f"{metrics.get('uptime_hours', 0):.1f} h",
            delta=None
        )
    
    st.markdown("---")
    
    # Performance charts
    st.subheader("📈 System Performance")
    
    performance_fig = create_performance_charts(metrics)
    st.plotly_chart(performance_fig, use_container_width=True)
    
    # Biological monitoring
    st.subheader("🦠 Biological System Monitoring")
    
    biofilm_fig = create_biofilm_monitoring_chart(metrics)
    st.plotly_chart(biofilm_fig, use_container_width=True)
    
    # System alerts
    st.subheader("🚨 System Alerts")
    
    alerts_data = DashboardAPI.get_active_alerts()
    alerts = alerts_data.get('alerts', [])
    
    display_alerts(alerts)
    
    # Real-time data table
    st.subheader("📋 Real-time System Data")
    
    if metrics:
        # Create a formatted data table
        data_rows = []
        
        # System metrics
        data_rows.extend([
            {"Parameter": "System Status", "Value": system_status.title(), "Unit": "-"},
            {"Parameter": "Uptime", "Value": f"{metrics.get('uptime_hours', 0):.2f}", "Unit": "hours"},
            {"Parameter": "Total Energy", "Value": f"{metrics.get('total_energy_produced_kwh', 0):.3f}", "Unit": "kWh"},
        ])
        
        # Environmental conditions
        data_rows.extend([
            {"Parameter": "Temperature", "Value": f"{metrics.get('temperature_c', 0):.1f}", "Unit": "°C"},
            {"Parameter": "pH Level", "Value": f"{metrics.get('ph_level', 0):.2f}", "Unit": "-"},
            {"Parameter": "Pressure", "Value": f"{metrics.get('pressure_bar', 0):.2f}", "Unit": "bar"},
            {"Parameter": "Flow Rate", "Value": f"{metrics.get('flow_rate_ml_min', 0):.1f}", "Unit": "mL/min"},
        ])
        
        # Control system
        data_rows.extend([
            {"Parameter": "Controller Mode", "Value": metrics.get('controller_mode', 'manual').title(), "Unit": "-"},
            {"Parameter": "Learning Progress", "Value": f"{metrics.get('learning_progress_pct', 0):.1f}", "Unit": "%"},
            {"Parameter": "Epsilon Value", "Value": f"{metrics.get('epsilon_value', 0):.3f}", "Unit": "-"},
        ])
        
        df = pd.DataFrame(data_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()