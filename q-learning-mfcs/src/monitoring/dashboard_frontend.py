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
