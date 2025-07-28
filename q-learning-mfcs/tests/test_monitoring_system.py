"""
Test suite for MFC Real-time Monitoring System

Comprehensive tests for dashboard API, safety monitoring, and real-time streaming.
"""
import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import sys
import os
from pathlib import Path

from monitoring.dashboard_api import app, SystemStatus, SystemMetrics, AlertMessage
from monitoring.safety_monitor import (
from monitoring.realtime_streamer import (
class TestDashboardAPI:
    """Test cases for Dashboard API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_system_status(self, client):
        """Test system status endpoint"""
        response = client.get("/api/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "initialized" in data
        assert "connections" in data
    
    def test_current_metrics(self, client):
        """Test current metrics endpoint"""
        response = client.get("/api/metrics/current")
        assert response.status_code == 200
        
        # Should return empty dict if no data available
        data = response.json()
        assert isinstance(data, dict)
    
    def test_control_command_invalid(self, client):
        """Test invalid control command"""
        response = client.post("/api/control/command", json={
            "command": "invalid_command"
        })
        assert response.status_code == 400
    
    def test_control_command_valid(self, client):
        """Test valid control commands"""
        commands = ["start", "stop", "pause", "resume", "reset", "emergency_stop"]
        
        for command in commands:
            response = client.post("/api/control/command", json={
                "command": command
            })
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "message" in data
    
    def test_config_update(self, client):
        """Test configuration update"""
        response = client.post("/api/config/update", json={
            "section": "safety",
            "parameters": {"max_temp": 50.0},
            "apply_immediately": True
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_alerts_endpoints(self, client):
        """Test alert-related endpoints"""
        # Get active alerts
        response = client.get("/api/alerts/active")
        assert response.status_code == 200
        
        data = response.json()
        assert "alerts" in data
        assert "count" in data
        
        # Acknowledge alert (should work even if alert doesn't exist)
        response = client.post("/api/alerts/test_alert_id/acknowledge")
        assert response.status_code == 200
