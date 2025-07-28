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
