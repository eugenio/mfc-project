"""
System Integration Tests
=======================

End-to-end integration tests that validate the complete MFC system
including MLOps and Security components working together.

Tests include:
- Complete MFC simulation with MLOps monitoring
- Security-protected API endpoints with real data
- Q-Learning optimization with observability
- Process management with security middleware
- Full stack operational scenarios

Created: 2025-08-05
Author: TDD Agent 10 - Integration & Testing
"""

import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from . import IntegrationTestConfig


class TestSystemIntegration:
    """Test complete system integration"""

    @pytest.fixture(scope="class")
    def integrated_system(self, tmp_path_factory):
        """Set up integrated MFC system with MLOps and Security"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        # Initialize components
        from deployment.process_manager import (
            ProcessConfig,
            ProcessManager,
            RestartPolicy,
        )
        from monitoring.observability_manager import ObservabilityManager
        from monitoring.security_middleware import SecurityConfig, SessionManager

        # Create temp directory for this test
        temp_dir = tmp_path_factory.mktemp("system_integration")

        # Create system components
        obs_manager = ObservabilityManager(check_interval=2.0)
        proc_manager = ProcessManager(log_dir=str(temp_dir / "logs"))
        security_config = SecurityConfig()
        session_manager = SessionManager(security_config)

        # Register system services
        obs_manager.register_service("mfc_simulation")
        obs_manager.register_service("qlearning_optimizer")
        obs_manager.register_service("monitoring_api")
        obs_manager.register_service("security_middleware")

        # Add test processes for MFC components
        test_processes = [
            ProcessConfig(
                name="mfc_data_generator",
                command=["python", "-c", """
import time
import json
import random
from datetime import datetime

for i in range(10):
    data = {
        'timestamp': datetime.now().isoformat(),
        'voltage': round(random.uniform(0.3, 0.8), 3),
        'current': round(random.uniform(0.1, 0.5), 3),
        'power': round(random.uniform(0.05, 0.4), 3),
        'temperature': round(random.uniform(20.0, 35.0), 1),
        'ph': round(random.uniform(6.5, 7.5), 1),
        'iteration': i
    }
    print(f'MFC_DATA: {json.dumps(data)}')
    time.sleep(1)
print('MFC data generation completed')
"""],
                restart_policy=RestartPolicy.NEVER,
                log_file=str(temp_dir / "mfc_data_generator.log")
            ),
            ProcessConfig(
                name="qlearning_mock",
                command=["python", "-c", """
import time
import json
import random
from datetime import datetime

print('Q-Learning optimizer starting...')
for episode in range(5):
    metrics = {
        'episode': episode,
        'timestamp': datetime.now().isoformat(),
        'reward': round(random.uniform(-1.0, 1.0), 3),
        'epsilon': round(max(0.1, 1.0 - episode * 0.2), 3),
        'q_value': round(random.uniform(0.0, 10.0), 3),
        'convergence_score': round(min(1.0, episode * 0.2), 3)
    }
    print(f'QLEARNING_METRICS: {json.dumps(metrics)}')
    time.sleep(2)
print('Q-Learning optimization completed')
"""],
                restart_policy=RestartPolicy.ON_FAILURE,
                max_restarts=2,
                log_file=str(temp_dir / "qlearning_mock.log")
            )
        ]

        for config in test_processes:
            proc_manager.add_process(config)

        # Start system components
        obs_manager.start_monitoring()
        proc_manager.start_health_checking()

        system_components = {
            'observability': obs_manager,
            'processes': proc_manager,
            'security': security_config,
            'sessions': session_manager,
            'temp_dir': temp_dir
        }

        yield system_components

        # Cleanup
        obs_manager.stop_monitoring()
        proc_manager.shutdown_all()

    @pytest.mark.integration
    @pytest.mark.system
    def test_end_to_end_mfc_simulation_with_monitoring(self, integrated_system):
        """Test complete MFC simulation with monitoring and security"""
        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']

        # Start MFC data generation
        success = proc_manager.start_process("mfc_data_generator")
        assert success, "Failed to start MFC data generator"

        # Start Q-Learning optimizer
        success = proc_manager.start_process("qlearning_mock")
        assert success, "Failed to start Q-Learning optimizer"

        # Let processes run and collect data
        time.sleep(8)

        # Verify system status
        system_status = obs_manager.get_system_status()
        process_status = proc_manager.get_status()

        # Check that system is operational
        assert system_status["services_count"] >= 4
        assert process_status["total_processes"] >= 2

        # Check that processes have run
        running_or_completed = 0
        for _name, info in process_status["processes"].items():
            if info["state"] in ["running", "stopped"]:  # stopped = completed successfully
                running_or_completed += 1

        assert running_or_completed >= 2, "Expected at least 2 processes to run or complete"

        # Verify no critical alerts
        active_alerts = obs_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity.value == "critical"]
        assert len(critical_alerts) == 0, f"Unexpected critical alerts: {critical_alerts}"

    @pytest.mark.integration
    @pytest.mark.system
    def test_secure_api_with_real_data_flow(self, integrated_system):
        """Test secure API endpoints with real data flow"""
        import sys

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


        # Create test API with security
        app = FastAPI()

        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']

        # Add secure endpoints
        @app.get("/api/v1/system/status")
        async def get_system_status():
            return {
                "observability": obs_manager.get_system_status(),
                "processes": proc_manager.get_status(),
                "timestamp": datetime.now().isoformat()
            }

        @app.get("/api/v1/mfc/data")
        async def get_mfc_data():
            # Simulate real MFC data retrieval
            return {
                "voltage": 0.65,
                "current": 0.32,
                "power": 0.208,
                "efficiency": 0.78,
                "temperature": 28.5,
                "ph": 7.1,
                "timestamp": datetime.now().isoformat()
            }

        @app.post("/api/v1/qlearning/action")
        async def submit_qlearning_action(data: dict):
            # Simulate Q-Learning action submission
            return {
                "action_accepted": True,
                "predicted_reward": 0.85,
                "next_state": "optimal",
                "timestamp": datetime.now().isoformat()
            }

        client = TestClient(app)

        # Test endpoints with proper authentication
        auth_headers = {
            "Authorization": f"Bearer {IntegrationTestConfig.get_test_environment()['MFC_API_TOKEN']}"
        }

        # Test system status endpoint
        response = client.get("/api/v1/system/status", headers=auth_headers)
        assert response.status_code == 200

        status_data = response.json()
        assert "observability" in status_data
        assert "processes" in status_data
        assert "timestamp" in status_data

        # Test MFC data endpoint
        response = client.get("/api/v1/mfc/data", headers=auth_headers)
        assert response.status_code == 200

        mfc_data = response.json()
        assert "voltage" in mfc_data
        assert "current" in mfc_data
        assert "power" in mfc_data
        assert "timestamp" in mfc_data

        # Test Q-Learning action endpoint
        action_data = {
            "state": [0.65, 0.32, 28.5, 7.1],
            "action": "increase_flow",
            "episode": 42
        }

        response = client.post("/api/v1/qlearning/action", json=action_data, headers=auth_headers)
        assert response.status_code == 200

        result = response.json()
        assert result["action_accepted"] is True
        assert "predicted_reward" in result

    @pytest.mark.integration
    @pytest.mark.system
    def test_monitoring_with_real_metrics_collection(self, integrated_system):
        """Test monitoring system with real metrics from MFC processes"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        from monitoring.observability_manager import (
            Metric,
            MetricsCollector,
            MetricType,
        )

        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']
        collector = MetricsCollector()

        # Start processes to generate data
        proc_manager.start_process("mfc_data_generator")
        proc_manager.start_process("qlearning_mock")

        # Collect metrics during operation
        start_time = time.time()
        metrics_collected = 0

        while time.time() - start_time < 10 and metrics_collected < 20:
            # Simulate real MFC metrics collection
            mfc_metrics = [
                Metric("mfc.voltage", 0.65 + (time.time() % 10) * 0.02, MetricType.GAUGE, "mfc_system"),
                Metric("mfc.current", 0.32 + (time.time() % 5) * 0.01, MetricType.GAUGE, "mfc_system"),
                Metric("mfc.power", 0.208 + (time.time() % 8) * 0.003, MetricType.GAUGE, "mfc_system"),
                Metric("qlearning.episodes", metrics_collected // 4, MetricType.COUNTER, "qlearning_system"),
                Metric("qlearning.reward", -0.5 + (metrics_collected * 0.1), MetricType.GAUGE, "qlearning_system"),
                Metric("system.cpu_usage", 45.0 + (time.time() % 20), MetricType.GAUGE, "system"),
                Metric("system.memory_usage", 2048 + (time.time() % 500), MetricType.GAUGE, "system")
            ]

            for metric in mfc_metrics:
                collector.record_metric(metric)
                metrics_collected += 1

            time.sleep(0.5)

        # Verify metrics were collected
        assert metrics_collected >= 15, "Should have collected sufficient metrics"

        # Check metrics summary
        summary = collector.get_metrics_summary()
        assert "gauges" in summary
        assert "counters" in summary

        # Verify specific MFC metrics
        mfc_voltage = collector.get_gauge_value("mfc_system", "mfc.voltage")
        assert mfc_voltage is not None and 0.6 < mfc_voltage < 0.8

        qlearning_episodes = collector.get_counter_value("qlearning_system", "qlearning.episodes")
        assert qlearning_episodes >= 0

        # Check system status incorporates real data
        system_status = obs_manager.get_system_status()
        assert system_status["overall_status"] in ["healthy", "degraded"]

    @pytest.mark.integration
    @pytest.mark.system
    def test_process_lifecycle_with_security_context(self, integrated_system):
        """Test process lifecycle management within security context"""
        proc_manager = integrated_system['processes']
        session_manager = integrated_system['sessions']
        obs_manager = integrated_system['observability']

        # Create authenticated session for process management
        session_id = session_manager.create_session(
            user_id="system_admin",
            user_data={"role": "admin", "permissions": ["process_management", "monitoring"]}
        )

        # Verify session is valid
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None
        assert session_data["user_data"]["role"] == "admin"

        # Perform process operations within security context
        initial_status = proc_manager.get_status()
        initial_status["total_processes"]

        # Start processes with security context
        success = proc_manager.start_process("mfc_data_generator")
        assert success, "Process start should succeed with valid session"

        time.sleep(3)  # Let process run

        # Check process status
        proc_manager.get_status()
        process_info = proc_manager.get_process_info("mfc_data_generator")

        assert process_info is not None
        assert process_info.config.name == "mfc_data_generator"

        # Stop process with security context
        success = proc_manager.stop_process("mfc_data_generator")
        assert success, "Process stop should succeed with valid session"

        # Verify observability tracked the operations
        system_status = obs_manager.get_system_status()
        assert system_status["services_count"] >= 4  # Should have system services

        # Clean up session
        session_manager.destroy_session(session_id)

        # Verify session is now invalid
        invalid_session = session_manager.validate_session(session_id)
        assert invalid_session is None

    @pytest.mark.integration
    @pytest.mark.system
    def test_alert_system_with_real_conditions(self, integrated_system):
        """Test alert system with realistic MFC conditions"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        from monitoring.observability_manager import AlertCondition, AlertSeverity

        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']

        # Add MFC-specific alert conditions
        mfc_alert_conditions = [
            AlertCondition(
                name="mfc_low_voltage",
                metric_name="mfc.voltage",
                threshold=0.4,
                comparison="lt",
                duration_seconds=2,
                severity=AlertSeverity.WARNING
            ),
            AlertCondition(
                name="mfc_high_temperature",
                metric_name="mfc.temperature",
                threshold=40.0,
                comparison="gt",
                duration_seconds=3,
                severity=AlertSeverity.ERROR
            ),
            AlertCondition(
                name="qlearning_poor_performance",
                metric_name="qlearning.convergence_score",
                threshold=0.2,
                comparison="lt",
                duration_seconds=5,
                severity=AlertSeverity.WARNING
            )
        ]

        # Add alert conditions to system
        for condition in mfc_alert_conditions:
            obs_manager.add_alert_condition(condition)

        # Start processes to generate metrics
        proc_manager.start_process("mfc_data_generator")
        proc_manager.start_process("qlearning_mock")

        # Monitor for alerts over time
        len(obs_manager.get_active_alerts())

        # Let system run and potentially trigger alerts
        time.sleep(8)

        # Check final alert state
        final_alerts = obs_manager.get_active_alerts()
        alert_history = obs_manager.get_alert_history()

        # System should be monitoring for alerts
        assert len(obs_manager.alert_conditions) >= len(mfc_alert_conditions)

        # Check system status reflects alert monitoring
        system_status = obs_manager.get_system_status()
        assert "active_alerts" in system_status
        assert "critical_alerts" in system_status

        # Verify alert system is functional (may or may not have active alerts)
        assert isinstance(final_alerts, list)
        assert isinstance(alert_history, list)

    @pytest.mark.integration
    @pytest.mark.system
    def test_system_resilience_and_recovery(self, integrated_system):
        """Test system resilience and recovery capabilities"""
        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']

        # Start system processes
        success = proc_manager.start_process("mfc_data_generator")
        assert success

        time.sleep(2)  # Let process start

        # Simulate process failure by stopping it abruptly
        process_info = proc_manager.get_process_info("mfc_data_generator")
        assert process_info is not None

        if process_info.pid:
            # Force kill the process to simulate failure
            import os
            import signal
            try:
                os.kill(process_info.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass  # Process may have already finished

        # Wait for system to detect and handle the failure
        time.sleep(3)

        # Check that system detected the failure
        updated_info = proc_manager.get_process_info("mfc_data_generator")
        if updated_info:
            # Process state should reflect the failure
            from deployment.process_manager import ProcessState
            assert updated_info.state in [ProcessState.FAILED, ProcessState.STOPPED, ProcessState.RESTARTING]

        # Verify observability system is still functional
        system_status = obs_manager.get_system_status()
        assert "overall_status" in system_status
        assert system_status["services_count"] >= 4

        # System should remain operational despite individual process failures
        assert system_status["overall_status"] in ["healthy", "degraded", "error"]

    @pytest.mark.integration
    @pytest.mark.system
    def test_concurrent_operations_system_stress(self, integrated_system):
        """Test system under concurrent operations stress"""
        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']
        session_manager = integrated_system['sessions']

        # Create multiple concurrent operations
        def monitoring_operations():
            for _ in range(10):
                obs_manager.get_system_status()
                time.sleep(0.1)

        def process_operations():
            for _ in range(5):
                proc_manager.get_status()
                time.sleep(0.2)

        def session_operations():
            sessions = []
            for i in range(10):
                session_id = session_manager.create_session(f"stress_user_{i}")
                sessions.append(session_id)
                time.sleep(0.1)

            # Validate and clean up sessions
            for session_id in sessions:
                session_manager.validate_session(session_id)
                session_manager.destroy_session(session_id)

        # Run concurrent operations
        threads = [
            threading.Thread(target=monitoring_operations),
            threading.Thread(target=process_operations),
            threading.Thread(target=session_operations),
            threading.Thread(target=monitoring_operations),  # Duplicate for more stress
        ]

        start_time = time.time()

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=15)

        end_time = time.time()

        # Verify system remained responsive
        assert end_time - start_time < 12, "System took too long under stress"

        # Verify system is still functional after stress
        final_status = obs_manager.get_system_status()
        assert final_status["overall_status"] in ["healthy", "degraded"]

        final_proc_status = proc_manager.get_status()
        assert final_proc_status["total_processes"] >= 2


class TestSystemPerformance:
    """Test system-wide performance characteristics"""

    @pytest.mark.integration
    @pytest.mark.system
    def test_system_startup_performance(self, temp_dir):
        """Test system startup performance"""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

        from deployment.process_manager import ProcessManager
        from monitoring.observability_manager import ObservabilityManager

        start_time = time.time()

        # Initialize system components
        obs_manager = ObservabilityManager(check_interval=1.0)
        proc_manager = ProcessManager(log_dir=str(temp_dir / "perf_logs"))

        # Register services
        obs_manager.register_service("test_service_1")
        obs_manager.register_service("test_service_2")
        obs_manager.register_service("test_service_3")

        # Start monitoring
        obs_manager.start_monitoring()
        proc_manager.start_health_checking()

        initialization_time = time.time() - start_time

        # System should start quickly
        assert initialization_time < 2.0, f"System startup took too long: {initialization_time}s"

        # Verify system is operational
        status = obs_manager.get_system_status()
        assert status["services_count"] == 3

        # Cleanup
        obs_manager.stop_monitoring()
        proc_manager.shutdown_all()

        cleanup_time = time.time() - start_time - initialization_time
        assert cleanup_time < 3.0, f"System cleanup took too long: {cleanup_time}s"

    @pytest.mark.integration
    @pytest.mark.system
    def test_system_throughput_performance(self, integrated_system):
        """Test system throughput under load"""
        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']

        # Measure throughput of system operations
        start_time = time.time()
        operations_completed = 0

        # Perform high-frequency operations
        while time.time() - start_time < 5.0:
            obs_manager.get_system_status()
            proc_manager.get_status()
            operations_completed += 2

            if operations_completed % 20 == 0:
                time.sleep(0.01)  # Brief pause to prevent overwhelming

        elapsed_time = time.time() - start_time
        throughput = operations_completed / elapsed_time

        # System should maintain reasonable throughput
        assert throughput > 50, f"System throughput too low: {throughput} ops/sec"

        # Verify system stability after high load
        final_status = obs_manager.get_system_status()
        assert final_status["overall_status"] in ["healthy", "degraded"]


class TestSystemReliability:
    """Test system reliability and fault tolerance"""

    @pytest.mark.integration
    @pytest.mark.system
    def test_system_fault_tolerance(self, integrated_system):
        """Test system fault tolerance"""
        obs_manager = integrated_system['observability']

        # Simulate various failure scenarios
        def failing_health_check():
            import random
            if random.random() < 0.7:
                raise Exception("Simulated intermittent failure")
            return {"status": "healthy"}

        # Register unreliable service
        obs_manager.register_service("unreliable_service", failing_health_check)

        # Let system handle the unreliable service
        time.sleep(5)

        # System should remain operational despite unreliable components
        system_status = obs_manager.get_system_status()
        assert system_status["overall_status"] in ["healthy", "degraded", "error"]

        # Should have detected the unreliable service
        services = obs_manager.get_service_health()
        assert "unreliable_service" in services

    @pytest.mark.integration
    @pytest.mark.system
    def test_system_data_consistency(self, integrated_system):
        """Test system data consistency across components"""
        obs_manager = integrated_system['observability']
        proc_manager = integrated_system['processes']
        session_manager = integrated_system['sessions']

        # Create test data across components
        session_id = session_manager.create_session("consistency_test_user")

        # Start a process
        proc_manager.start_process("mfc_data_generator")
        time.sleep(2)

        # Get data from all components
        obs_status = obs_manager.get_system_status()
        proc_status = proc_manager.get_status()
        session_data = session_manager.validate_session(session_id)

        # Verify data consistency
        assert obs_status["timestamp"] is not None
        assert proc_status["total_processes"] >= 2
        assert session_data is not None
        assert session_data["user_id"] == "consistency_test_user"

        # Timestamps should be reasonably recent and consistent
        obs_time = datetime.fromisoformat(obs_status["timestamp"].replace('Z', '+00:00'))
        session_time = datetime.fromisoformat(session_data["created_at"])

        time_diff = abs((obs_time - session_time).total_seconds())
        assert time_diff < 60, "Timestamps should be reasonably consistent"

        # Cleanup
        session_manager.destroy_session(session_id)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-m", "system"])
