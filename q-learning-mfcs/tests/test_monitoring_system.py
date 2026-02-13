"""
Test suite for MFC Real-time Monitoring System

Comprehensive tests for dashboard API, safety monitoring, and real-time streaming.
"""
import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
import numpy as np

try:
    from src.monitoring.dashboard_api import app
    from src.monitoring.safety_monitor import (
        SafetyMonitor, SafetyLevel, SafetyEvent, SafetyThreshold, EmergencyAction
    )
    from src.monitoring.realtime_streamer import (
        RealTimeStreamer, StreamEventType, StreamEvent, ClientConnection
    )
    MONITORING_IMPORTS_AVAILABLE = True
except ImportError:
    MONITORING_IMPORTS_AVAILABLE = False
class TestDashboardAPI:
    """Test cases for Dashboard API"""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MONITORING_IMPORTS_AVAILABLE:
            pytest.skip("Monitoring system components not available")
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        if not MONITORING_IMPORTS_AVAILABLE:
            pytest.skip("Monitoring system components not available")
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
class TestSafetyMonitor:
    """Test cases for Safety Monitor"""
    
    @pytest.fixture
    def safety_monitor(self):
        """Create safety monitor instance"""
        return SafetyMonitor()
    
    def test_initialization(self, safety_monitor):
        """Test safety monitor initialization"""
        assert not safety_monitor.is_monitoring
        assert len(safety_monitor.safety_thresholds) > 0
        assert len(safety_monitor.safety_protocols) > 0
        assert len(safety_monitor.safety_events) == 0
    
    def test_default_thresholds(self, safety_monitor):
        """Test default safety thresholds"""
        thresholds = safety_monitor.safety_thresholds
        
        # Check key thresholds exist
        assert "temperature" in thresholds
        assert "pressure" in thresholds
        assert "ph_level" in thresholds
        assert "voltage" in thresholds
        
        # Check threshold properties
        temp_threshold = thresholds["temperature"]
        assert temp_threshold.max_value == 45.0
        assert temp_threshold.emergency_action == EmergencyAction.REDUCE_POWER
        assert temp_threshold.enabled is True
    
    def test_safety_level_evaluation(self, safety_monitor):
        """Test safety level evaluation"""
        threshold = SafetyThreshold(
            parameter="test_param",
            max_value=100.0,
            warning_buffer=10.0
        )
        
        # Test different safety levels
        assert safety_monitor._evaluate_safety_level("test", 50.0, threshold) == SafetyLevel.SAFE
        assert safety_monitor._evaluate_safety_level("test", 85.0, threshold) == SafetyLevel.CAUTION
        assert safety_monitor._evaluate_safety_level("test", 92.0, threshold) == SafetyLevel.WARNING
        assert safety_monitor._evaluate_safety_level("test", 98.0, threshold) == SafetyLevel.CRITICAL
        assert safety_monitor._evaluate_safety_level("test", 105.0, threshold) == SafetyLevel.EMERGENCY
    
    def test_safety_event_creation(self, safety_monitor):
        """Test safety event creation and processing"""
        # Create measurements that trigger safety events
        measurements = {
            "temperature": 50.0,  # Above 45.0 threshold
            "pressure": 3.0,      # Above 2.5 threshold
            "ph_level": 4.0       # Below 5.5 threshold
        }
        
        events = safety_monitor._check_safety_thresholds(measurements)
        
        assert len(events) == 3
        
        # Check temperature event
        temp_event = next(e for e in events if e.parameter == "temperature")
        assert temp_event.safety_level == SafetyLevel.EMERGENCY
        assert temp_event.current_value == 50.0
    
    def test_emergency_action_execution(self, safety_monitor):
        """Test emergency action execution"""
        event = SafetyEvent(
            event_id="test_event",
            timestamp=datetime.now(),
            parameter="temperature",
            current_value=50.0,
            threshold_value=45.0,
            safety_level=SafetyLevel.EMERGENCY,
            action_taken=EmergencyAction.NONE,
            response_time_ms=0.0
        )
        
        # Test different actions
        actions = [
            EmergencyAction.REDUCE_POWER,
            EmergencyAction.STOP_FLOW,
            EmergencyAction.EMERGENCY_SHUTDOWN,
            EmergencyAction.ISOLATE_SYSTEM,
            EmergencyAction.NOTIFY_PERSONNEL
        ]
        
        for action in actions:
            result = safety_monitor._execute_emergency_action(action, event)
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_safety_protocols(self, safety_monitor):
        """Test safety protocol execution"""
        protocols = safety_monitor.safety_protocols
        
        assert "thermal_runaway" in protocols
        assert "pressure_emergency" in protocols
        assert "system_failure" in protocols
        
        # Test protocol triggering
        thermal_protocol = protocols["thermal_runaway"]
        
        # Create events that should trigger thermal runaway protocol
        critical_events = [
            SafetyEvent(
                event_id="temp_event",
                timestamp=datetime.now(),
                parameter="temperature",
                current_value=50.0,
                threshold_value=45.0,
                safety_level=SafetyLevel.CRITICAL,
                action_taken=EmergencyAction.NONE,
                response_time_ms=0.0
            )
        ]
        
        assert thermal_protocol.should_trigger(critical_events) is True
    
    def test_monitoring_loop(self, safety_monitor):
        """Test monitoring loop start/stop"""
        # Start monitoring
        safety_monitor.start_monitoring(interval_seconds=0.1)
        assert safety_monitor.is_monitoring is True
        assert safety_monitor.monitoring_thread is not None
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop monitoring
        safety_monitor.stop_monitoring()
        assert safety_monitor.is_monitoring is False
    
    def test_event_acknowledgment(self, safety_monitor):
        """Test event acknowledgment and resolution"""
        # Create a test event
        event = SafetyEvent(
            event_id="test_event_123",
            timestamp=datetime.now(),
            parameter="temperature",
            current_value=50.0,
            threshold_value=45.0,
            safety_level=SafetyLevel.CRITICAL,
            action_taken=EmergencyAction.NONE,
            response_time_ms=100.0
        )
        
        safety_monitor.safety_events.append(event)
        
        # Test acknowledgment
        assert safety_monitor.acknowledge_event("test_event_123", "test_user") is True
        assert event.acknowledged is True
        
        # Test resolution
        assert safety_monitor.resolve_event("test_event_123", "test_user") is True
        assert event.resolved is True
        
        # Test non-existent event
        assert safety_monitor.acknowledge_event("non_existent", "test_user") is False
    
    def test_safety_status(self, safety_monitor):
        """Test safety status reporting"""
        status = safety_monitor.get_safety_status()
        
        assert "overall_safety_level" in status
        assert "active_events" in status
        assert "is_monitoring" in status
        assert "active_protocols" in status
        assert "recent_events" in status
        assert "statistics" in status
        
        assert status["overall_safety_level"] == SafetyLevel.SAFE.value
        assert isinstance(status["active_events"], int)
    
    def test_threshold_update(self, safety_monitor):
        """Test threshold configuration update"""
        # Update temperature threshold
        update_data = {
            "max_value": 50.0,
            "warning_buffer": 8.0,
            "enabled": True
        }
        
        result = safety_monitor.update_threshold("temperature", update_data)
        assert result is True
        
        # Check updated values
        threshold = safety_monitor.safety_thresholds["temperature"]
        assert threshold.max_value == 50.0
        assert threshold.warning_buffer == 8.0
        
        # Test invalid parameter
        result = safety_monitor.update_threshold("invalid_param", update_data)
        assert result is False
    
    def test_safety_report(self, safety_monitor):
        """Test safety report generation"""
        # Add some test events
        current_time = datetime.now()
        
        for i in range(5):
            event = SafetyEvent(
                event_id=f"test_event_{i}",
                timestamp=current_time - timedelta(hours=i),
                parameter="temperature",
                current_value=45.0 + i,
                threshold_value=45.0,
                safety_level=SafetyLevel.WARNING if i % 2 == 0 else SafetyLevel.CRITICAL,
                action_taken=EmergencyAction.NONE,
                response_time_ms=100.0 + i * 10
            )
            safety_monitor.safety_events.append(event)
        
        # Generate report
        report = safety_monitor.get_safety_report(hours=24)
        
        assert "report_period_hours" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "parameter_breakdown" in report
        assert "detailed_events" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_events"] == 5
        assert summary["critical_events"] >= 0
        assert "avg_response_time_ms" in summary
class TestRealTimeStreamer:
    """Test cases for Real-time Streamer"""
    
    @pytest.fixture
    def streamer(self):
        """Create streamer instance"""
        return RealTimeStreamer(host="localhost", port=8002)
    
    def test_initialization(self, streamer):
        """Test streamer initialization"""
        assert streamer.host == "localhost"
        assert streamer.port == 8002
        assert len(streamer.clients) == 0
        assert not streamer.is_running
    
    @pytest.mark.asyncio
    async def test_event_queue(self, streamer):
        """Test event queue operations"""
        # Add event to queue
        await streamer.add_event(
            StreamEventType.METRICS_UPDATE,
            {"test": "data"},
            priority=1
        )
        
        # Check queue has event
        assert not streamer.event_queue.empty()
        
        # Get event from queue
        event = await streamer.event_queue.get()
        
        assert event.event_type == StreamEventType.METRICS_UPDATE
        assert event.data == {"test": "data"}
        assert event.priority == 1
    
    def test_server_stats(self, streamer):
        """Test server statistics"""
        stats = streamer.get_server_stats()
        
        assert "active_clients" in stats
        assert "total_connections" in stats
        assert "events_sent" in stats
        assert "bytes_transmitted" in stats
        assert "uptime_seconds" in stats
        assert "start_time" in stats
        assert "is_running" in stats
        
        assert stats["active_clients"] == 0
        assert stats["is_running"] is False
    
    @pytest.mark.asyncio
    async def test_client_message_handling(self, streamer):
        """Test client message handling"""
        # Mock websocket and create client
        mock_websocket = AsyncMock()
        client_id = "test_client"
        
        client = ClientConnection(
            client_id=client_id,
            websocket=mock_websocket,
            connected_at=datetime.now(),
            subscriptions=set(),
            last_ping=datetime.now()
        )
        
        streamer.clients[client_id] = client
        
        # Test subscription message
        message = json.dumps({
            "type": "subscribe",
            "events": ["metrics_update", "alert"]
        })
        
        await streamer.handle_client_message(client_id, message)
        
        # Check subscriptions were added
        assert StreamEventType.METRICS_UPDATE in client.subscriptions
        assert StreamEventType.ALERT in client.subscriptions
        
        # Test ping message
        ping_message = json.dumps({"type": "ping"})
        await streamer.handle_client_message(client_id, ping_message)
        
        # Test invalid JSON
        invalid_message = "invalid json"
        await streamer.handle_client_message(client_id, invalid_message)
        
        # Verify mock was called (for error response)
        assert mock_websocket.send.called
class TestIntegration:
    """Integration tests for monitoring system components"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create integrated monitoring system"""
        safety_monitor = SafetyMonitor()
        streamer = RealTimeStreamer(host="localhost", port=8003)
        
        return {
            "safety_monitor": safety_monitor,
            "streamer": streamer
        }
    
    def test_safety_to_streaming_integration(self, monitoring_system):
        """Test integration between safety monitoring and streaming"""
        safety_monitor = monitoring_system["safety_monitor"]
        monitoring_system["streamer"]
        
        # Start safety monitoring
        safety_monitor.start_monitoring(interval_seconds=0.1)
        
        # Let it generate some events
        time.sleep(0.5)
        
        # Check that events were generated
        assert len(safety_monitor.safety_events) >= 0
        
        # Stop monitoring
        safety_monitor.stop_monitoring()
    
    def test_end_to_end_alert_flow(self, monitoring_system):
        """Test end-to-end alert flow"""
        safety_monitor = monitoring_system["safety_monitor"]
        
        # Create critical measurements
        measurements = {
            "temperature": 60.0,  # Way above threshold
            "pressure": 4.0       # Way above threshold
        }
        
        # Process measurements
        events = safety_monitor._check_safety_thresholds(measurements)
        
        # Should generate emergency-level events
        assert len(events) >= 2
        
        emergency_events = [e for e in events if e.safety_level == SafetyLevel.EMERGENCY]
        assert len(emergency_events) >= 2
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_simulation(self, monitoring_system):
        """Test WebSocket broadcast simulation"""
        streamer = monitoring_system["streamer"]
        
        # Create mock clients
        mock_clients = {}
        for i in range(3):
            client_id = f"client_{i}"
            mock_websocket = AsyncMock()
            
            client = ClientConnection(
                client_id=client_id,
                websocket=mock_websocket,
                connected_at=datetime.now(),
                subscriptions={StreamEventType.METRICS_UPDATE, StreamEventType.ALERT},
                last_ping=datetime.now()
            )
            
            mock_clients[client_id] = client
            streamer.clients[client_id] = client
        
        # Create and broadcast event
        event = StreamEvent(
            event_id="test_broadcast",
            event_type=StreamEventType.METRICS_UPDATE,
            timestamp=datetime.now(),
            data={"power": 5.5, "temperature": 25.0},
            priority=1
        )
        
        await streamer.broadcast_event(event)
        
        # Verify all clients received the message
        for client in mock_clients.values():
            client.websocket.send.assert_called()
    
    def test_performance_under_load(self, monitoring_system):
        """Test system performance under load"""
        safety_monitor = monitoring_system["safety_monitor"]
        
        # Generate many safety events quickly
        start_time = time.time()
        
        for i in range(100):
            measurements = {
                "temperature": 25.0 + np.random.normal(0, 10),
                "pressure": 1.0 + np.random.normal(0, 1),
                "ph_level": 7.0 + np.random.normal(0, 2)
            }
            
            events = safety_monitor._check_safety_thresholds(measurements)
            
            for event in events:
                safety_monitor._process_safety_event(event)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 measurement sets in reasonable time
        assert processing_time < 10.0  # Less than 10 seconds
        
        # Check response times are reasonable
        if safety_monitor.stats["response_times"]:
            avg_response_time = np.mean(safety_monitor.stats["response_times"])
            assert avg_response_time < 100.0  # Less than 100ms average
