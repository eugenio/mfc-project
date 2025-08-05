import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import websockets

"""
Real-time Monitoring and Streaming Test Suite
=============================================

Tests for real-time data streaming, WebSocket connectivity,
and monitoring system functionality for digital twin operations.
"""


class TestRealTimeStreaming:
    """Test real-time data streaming and WebSocket connectivity"""

    @pytest.fixture
    def mock_stream_manager(self):
        """Create mock streaming manager"""
        with patch('q_learning_mfcs.src.monitoring.realtime_streamer.load_ssl_config') as mock_ssl:
            mock_ssl.return_value = None

            from q_learning_mfcs.src.monitoring.realtime_streamer import (
                DataStreamManager,
            )
            manager = DataStreamManager()
            return manager

    @pytest.mark.asyncio
    async def test_client_registration(self, mock_stream_manager):
        """Test WebSocket client registration"""
        # Mock WebSocket client
        mock_websocket = Mock()
        mock_websocket.remote_address = ('127.0.0.1', 12345)
        mock_websocket.send = AsyncMock()

        # Register client
        await mock_stream_manager.register_client(mock_websocket)

        assert mock_websocket in mock_stream_manager.clients
        assert len(mock_stream_manager.clients) == 1

    @pytest.mark.asyncio
    async def test_data_broadcasting(self, mock_stream_manager):
        """Test real-time data broadcasting"""
        # Register multiple mock clients
        mock_clients = []
        for i in range(3):
            mock_client = Mock()
            mock_client.remote_address = ('127.0.0.1', 12345 + i)
            mock_client.send = AsyncMock()
            mock_clients.append(mock_client)
            await mock_stream_manager.register_client(mock_client)

        # Broadcast test message
        test_message = {
            'type': 'data_update',
            'timestamp': datetime.now().isoformat(),
            'data': [{'power': 0.5, 'biofilm': [25, 28, 22]}]
        }

        await mock_stream_manager.broadcast_to_clients(test_message)

        # Verify all clients received the message
        for client in mock_clients:
            client.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_latency(self, mock_stream_manager):
        """Test streaming latency requirements"""
        # Mock client
        mock_client = Mock()
        mock_client.remote_address = ('127.0.0.1', 12345)
        mock_client.send = AsyncMock()

        await mock_stream_manager.register_client(mock_client)

        # Measure broadcast latency
        start_time = time.time()

        for _ in range(100):
            message = {
                'type': 'heartbeat',
                'timestamp': datetime.now().isoformat()
            }
            await mock_stream_manager.broadcast_to_clients(message)

        end_time = time.time()
        avg_latency = (end_time - start_time) / 100

        # Should maintain low latency for real-time streaming
        assert avg_latency < 0.001, f"Streaming latency too high: {avg_latency:.4f}s"

    def test_data_cache_management(self, mock_stream_manager):
        """Test data cache size management"""
        # Add data beyond cache limit
        test_data = []
        for i in range(1500):  # Exceed cache_size_limit of 1000
            test_data.append({
                'timestamp': datetime.now().isoformat(),
                'data_point': i,
                'power': 0.5 + 0.1 * np.sin(i/10)
            })

        mock_stream_manager.update_data_cache(test_data)

        # Verify cache size is maintained
        assert len(mock_stream_manager.data_cache) == mock_stream_manager.cache_size_limit

        # Verify most recent data is retained
        assert mock_stream_manager.data_cache[-1]['data_point'] == 1499
        assert mock_stream_manager.data_cache[0]['data_point'] == 500  # Oldest retained

    @pytest.mark.asyncio
    async def test_client_message_handling(self, mock_stream_manager):
        """Test handling of client messages"""
        mock_client = Mock()
        mock_client.remote_address = ('127.0.0.1', 12345)
        mock_client.send = AsyncMock()

        await mock_stream_manager.register_client(mock_client)

        # Test ping message
        ping_message = json.dumps({"type": "ping"})
        await mock_stream_manager.handle_client_message(mock_client, ping_message)

        # Should respond with pong
        mock_client.send.assert_called()
        sent_data = json.loads(mock_client.send.call_args[0][0])
        assert sent_data['type'] == 'pong'

    @pytest.mark.asyncio
    async def test_data_request_handling(self, mock_stream_manager):
        """Test data request message handling"""
        # Add some test data to cache
        test_data = [
            {'timestamp': datetime.now().isoformat(), 'power': 0.5, 'cell': i}
            for i in range(50)
        ]
        mock_stream_manager.update_data_cache(test_data)

        mock_client = Mock()
        mock_client.remote_address = ('127.0.0.1', 12345)
        mock_client.send = AsyncMock()

        await mock_stream_manager.register_client(mock_client)

        # Request data
        request_message = json.dumps({"type": "request_data", "limit": 10})
        await mock_stream_manager.handle_client_message(mock_client, request_message)

        # Should send data response
        mock_client.send.assert_called()
        sent_data = json.loads(mock_client.send.call_args[0][0])
        assert sent_data['type'] == 'data_response'
        assert len(sent_data['data']) == 10


class TestObservabilityManager:
    """Test observability and alert management"""

    @pytest.fixture
    def observability_manager(self):
        """Create observability manager"""
        from q_learning_mfcs.src.monitoring.observability_manager import (
            ObservabilityManager,
        )
        return ObservabilityManager()

    def test_alert_condition_registration(self, observability_manager):
        """Test registration of alert conditions"""
        # Add test alert condition
        observability_manager.add_alert_condition(
            'high_temperature',
            lambda state: state.get('temperature', 0) > 35.0,
            'high'
        )

        # Verify condition is registered
        conditions = observability_manager.get_alert_conditions()
        assert 'high_temperature' in [c.condition_name for c in conditions]

    def test_alert_triggering(self, observability_manager):
        """Test alert triggering mechanism"""
        # Add alert condition
        observability_manager.add_alert_condition(
            'low_power',
            lambda state: state.get('power', 1.0) < 0.1,
            'medium'
        )

        # Test state that should trigger alert
        test_state = {'power': 0.05, 'timestamp': datetime.now()}
        observability_manager.check_conditions(test_state)

        active_alerts = observability_manager.get_active_alerts()
        assert len(active_alerts) > 0
        assert any(alert.condition_name == 'low_power' for alert in active_alerts)

    def test_alert_resolution(self, observability_manager):
        """Test alert resolution when conditions clear"""
        # Add alert condition
        observability_manager.add_alert_condition(
            'biofilm_thickness',
            lambda state: max(state.get('biofilm_thickness', [0])) > 50.0,
            'high'
        )

        # Trigger alert
        alert_state = {'biofilm_thickness': [55.0, 52.0, 48.0]}
        observability_manager.check_conditions(alert_state)

        # Verify alert is active
        active_alerts = observability_manager.get_active_alerts()
        assert len(active_alerts) > 0

        # Clear condition
        normal_state = {'biofilm_thickness': [45.0, 42.0, 40.0]}
        observability_manager.check_conditions(normal_state)

        # Verify alert is resolved
        active_alerts = observability_manager.get_active_alerts()
        biofilm_alerts = [a for a in active_alerts if a.condition_name == 'biofilm_thickness']
        assert len(biofilm_alerts) == 0

    def test_service_health_monitoring(self, observability_manager):
        """Test service health monitoring"""
        # Register a service
        observability_manager.register_service('mfc_controller')

        # Update service health
        observability_manager.update_service_health('mfc_controller', 'healthy', 'Operating normally')

        # Check service status
        service_health = observability_manager.get_service_health('mfc_controller')
        assert service_health.status.value == 'healthy'
        assert service_health.message == 'Operating normally'

    def test_alert_severity_levels(self, observability_manager):
        """Test different alert severity levels"""
        # Add conditions with different severities
        observability_manager.add_alert_condition(
            'critical_failure',
            lambda state: state.get('system_failure', False),
            'critical'
        )

        observability_manager.add_alert_condition(
            'warning_condition',
            lambda state: state.get('warning_flag', False),
            'low'
        )

        # Trigger both alerts
        test_state = {'system_failure': True, 'warning_flag': True}
        observability_manager.check_conditions(test_state)

        active_alerts = observability_manager.get_active_alerts()

        # Should have both alerts with correct severities
        critical_alerts = [a for a in active_alerts if a.severity.value == 'critical']
        warning_alerts = [a for a in active_alerts if a.severity.value == 'low']

        assert len(critical_alerts) > 0
        assert len(warning_alerts) > 0


class TestSensorIntegrationMonitoring:
    """Test sensor integration and monitoring functionality"""

    @pytest.fixture
    def mock_sensor_data(self):
        """Create mock sensor data"""
        return {
            'eis_data': {
                'thickness_measurements': [25.2, 28.5, 21.8, 30.1, 26.7],
                'conductivity': [0.15, 0.18, 0.12, 0.22, 0.16],
                'measurement_quality': [0.9, 0.85, 0.92, 0.88, 0.90]
            },
            'qcm_data': {
                'mass_measurements': [850, 920, 780, 1050, 890],
                'frequency_shifts': [-120, -145, -95, -165, -135],
                'dissipation': [0.001, 0.0012, 0.0008, 0.0015, 0.0011]
            }
        }

    def test_sensor_data_validation(self, mock_sensor_data):
        """Test sensor data validation"""
        # Test EIS data validation
        eis_data = mock_sensor_data['eis_data']

        # Should have consistent array lengths
        assert len(eis_data['thickness_measurements']) == len(eis_data['conductivity'])
        assert len(eis_data['thickness_measurements']) == len(eis_data['measurement_quality'])

        # Should have reasonable values
        assert all(t > 0 for t in eis_data['thickness_measurements'])
        assert all(0 <= q <= 1 for q in eis_data['measurement_quality'])

        # Test QCM data validation
        qcm_data = mock_sensor_data['qcm_data']

        assert len(qcm_data['mass_measurements']) == len(qcm_data['frequency_shifts'])
        assert all(m > 0 for m in qcm_data['mass_measurements'])
        assert all(f < 0 for f in qcm_data['frequency_shifts'])  # Frequency shifts should be negative

    def test_sensor_fusion_accuracy(self, mock_sensor_data):
        """Test accuracy of sensor data fusion"""
        # Simple fusion algorithm for testing
        def fuse_sensor_data(eis_data, qcm_data):
            n_cells = len(eis_data['thickness_measurements'])
            fused_results = {'fused_thickness': [], 'fusion_confidence': []}

            for i in range(n_cells):
                eis_thickness = eis_data['thickness_measurements'][i]
                eis_quality = eis_data['measurement_quality'][i]

                # Convert QCM mass to thickness estimate
                qcm_mass = qcm_data['mass_measurements'][i]
                qcm_thickness = qcm_mass / 30.0  # Simplified conversion
                qcm_quality = 0.8  # Assume good QCM quality

                # Weighted fusion
                total_weight = eis_quality + qcm_quality
                if total_weight > 0:
                    fused_thickness = (eis_thickness * eis_quality + qcm_thickness * qcm_quality) / total_weight
                    fusion_confidence = min(eis_quality, qcm_quality)
                else:
                    fused_thickness = eis_thickness
                    fusion_confidence = 0.1

                fused_results['fused_thickness'].append(fused_thickness)
                fused_results['fusion_confidence'].append(fusion_confidence)

            return fused_results

        # Perform fusion
        fused_results = fuse_sensor_data(
            mock_sensor_data['eis_data'],
            mock_sensor_data['qcm_data']
        )

        assert len(fused_results['fused_thickness']) == 5
        assert all(t > 0 for t in fused_results['fused_thickness'])
        assert all(0 <= c <= 1 for c in fused_results['fusion_confidence'])

    def test_sensor_fault_detection(self, observability_manager, mock_sensor_data):
        """Test detection of sensor faults"""
        # Add sensor fault conditions
        observability_manager.add_alert_condition(
            'eis_sensor_fault',
            lambda state: any(q < 0.3 for q in state.get('eis_quality', [])),
            'high'
        )

        observability_manager.add_alert_condition(
            'qcm_sensor_fault',
            lambda state: any(abs(f) > 200 for f in state.get('qcm_frequency_shifts', [])),
            'high'
        )

        # Create faulty sensor data
        faulty_state = {
            'eis_quality': [0.9, 0.1, 0.8, 0.2, 0.7],  # Some low quality measurements
            'qcm_frequency_shifts': [-120, -250, -95, -165, -135]  # One excessive shift
        }

        observability_manager.check_conditions(faulty_state)
        active_alerts = observability_manager.get_active_alerts()

        # Should detect both sensor faults
        eis_fault_alerts = [a for a in active_alerts if a.condition_name == 'eis_sensor_fault']
        qcm_fault_alerts = [a for a in active_alerts if a.condition_name == 'qcm_sensor_fault']

        assert len(eis_fault_alerts) > 0
        assert len(qcm_fault_alerts) > 0

    def test_sensor_calibration_monitoring(self, observability_manager):
        """Test monitoring of sensor calibration status"""
        # Add calibration drift condition
        observability_manager.add_alert_condition(
            'sensor_calibration_drift',
            lambda state: abs(state.get('calibration_offset', 0)) > 0.1,
            'medium'
        )

        # Simulate calibration drift
        drift_state = {'calibration_offset': 0.15}
        observability_manager.check_conditions(drift_state)

        active_alerts = observability_manager.get_active_alerts()
        calibration_alerts = [a for a in active_alerts if a.condition_name == 'sensor_calibration_drift']

        assert len(calibration_alerts) > 0


class TestDigitalTwinMonitoringIntegration:
    """Test integration between digital twin and monitoring systems"""

    def test_model_validation_monitoring(self, observability_manager):
        """Test monitoring of model validation metrics"""
        # Add model validation conditions
        observability_manager.add_alert_condition(
            'model_accuracy_degradation',
            lambda state: state.get('model_accuracy', 1.0) < 0.8,
            'high'
        )

        observability_manager.add_alert_condition(
            'prediction_uncertainty',
            lambda state: state.get('prediction_uncertainty', 0) > 0.3,
            'medium'
        )

        # Test degraded model performance
        degraded_state = {
            'model_accuracy': 0.75,
            'prediction_uncertainty': 0.35
        }

        observability_manager.check_conditions(degraded_state)
        active_alerts = observability_manager.get_active_alerts()

        # Should trigger both alerts
        accuracy_alerts = [a for a in active_alerts if a.condition_name == 'model_accuracy_degradation']
        uncertainty_alerts = [a for a in active_alerts if a.condition_name == 'prediction_uncertainty']

        assert len(accuracy_alerts) > 0
        assert len(uncertainty_alerts) > 0

    def test_synchronization_monitoring(self, observability_manager):
        """Test monitoring of digital twin synchronization"""
        # Add synchronization conditions
        observability_manager.add_alert_condition(
            'sync_latency_high',
            lambda state: state.get('sync_latency_ms', 0) > 100,
            'medium'
        )

        observability_manager.add_alert_condition(
            'data_loss',
            lambda state: state.get('data_loss_rate', 0) > 0.05,
            'high'
        )

        # Test high latency and data loss
        sync_state = {
            'sync_latency_ms': 150,
            'data_loss_rate': 0.08
        }

        observability_manager.check_conditions(sync_state)
        active_alerts = observability_manager.get_active_alerts()

        latency_alerts = [a for a in active_alerts if a.condition_name == 'sync_latency_high']
        data_loss_alerts = [a for a in active_alerts if a.condition_name == 'data_loss']

        assert len(latency_alerts) > 0
        assert len(data_loss_alerts) > 0

    def test_predictive_maintenance_alerts(self, observability_manager):
        """Test predictive maintenance alert generation"""
        # Add predictive maintenance conditions
        observability_manager.add_alert_condition(
            'biofilm_replacement_due',
            lambda state: state.get('biofilm_health_score', 1.0) < 0.6,
            'high'
        )

        observability_manager.add_alert_condition(
            'sensor_recalibration_due',
            lambda state: state.get('days_since_calibration', 0) > 30,
            'medium'
        )

        # Test maintenance conditions
        maintenance_state = {
            'biofilm_health_score': 0.55,
            'days_since_calibration': 35
        }

        observability_manager.check_conditions(maintenance_state)
        active_alerts = observability_manager.get_active_alerts()

        biofilm_alerts = [a for a in active_alerts if a.condition_name == 'biofilm_replacement_due']
        calibration_alerts = [a for a in active_alerts if a.condition_name == 'sensor_recalibration_due']

        assert len(biofilm_alerts) > 0
        assert len(calibration_alerts) > 0


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics collection"""

    def test_real_time_performance_tracking(self):
        """Test real-time performance metrics tracking"""
        # Create simple performance tracker
        class PerformanceTracker:
            def __init__(self):
                self.metrics = {}
                self.start_times = {}

            def start_timer(self, operation):
                self.start_times[operation] = time.time()

            def end_timer(self, operation):
                if operation in self.start_times:
                    duration = time.time() - self.start_times[operation]
                    if operation not in self.metrics:
                        self.metrics[operation] = []
                    self.metrics[operation].append(duration)
                    del self.start_times[operation]

            def get_average_duration(self, operation):
                if operation in self.metrics and self.metrics[operation]:
                    return sum(self.metrics[operation]) / len(self.metrics[operation])
                return 0

        tracker = PerformanceTracker()

        # Test operation timing
        for _ in range(100):
            tracker.start_timer('simulation_step')
            time.sleep(0.001)  # Simulate work
            tracker.end_timer('simulation_step')

        avg_duration = tracker.get_average_duration('simulation_step')
        assert 0.0008 < avg_duration < 0.002  # Should be around 1ms

    def test_throughput_monitoring(self):
        """Test data throughput monitoring"""
        class ThroughputMonitor:
            def __init__(self, window_size=10):
                self.window_size = window_size
                self.timestamps = []
                self.data_counts = []

            def record_data_batch(self, count):
                current_time = time.time()
                self.timestamps.append(current_time)
                self.data_counts.append(count)

                # Maintain window size
                if len(self.timestamps) > self.window_size:
                    self.timestamps.pop(0)
                    self.data_counts.pop(0)

            def get_throughput(self):
                if len(self.timestamps) < 2:
                    return 0

                time_span = self.timestamps[-1] - self.timestamps[0]
                total_data = sum(self.data_counts)

                return total_data / time_span if time_span > 0 else 0

        monitor = ThroughputMonitor()

        # Simulate data batches
        for i in range(20):
            monitor.record_data_batch(100 + i * 10)  # Increasing batch sizes
            time.sleep(0.01)

        throughput = monitor.get_throughput()
        assert throughput > 0  # Should have positive throughput

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        import psutil

        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.baseline_memory = self.get_memory_usage()

            def get_memory_usage(self):
                return self.process.memory_info().rss / 1024 / 1024  # MB

            def get_memory_increase(self):
                current_memory = self.get_memory_usage()
                return current_memory - self.baseline_memory

        monitor = MemoryMonitor()

        # Simulate memory usage
        large_data = [np.random.rand(1000, 1000) for _ in range(10)]

        memory_increase = monitor.get_memory_increase()
        assert memory_increase > 0  # Should show memory increase

        # Clean up
        del large_data
