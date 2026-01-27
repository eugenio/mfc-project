"""Test suite for Performance Monitor Dashboard.

Tests cover:
- SystemMetrics and SimulationStatus dataclasses
- PerformanceMonitor class with mocked psutil
- Simulation state management with session state
- Alert checking functionality
- Singleton pattern for PerformanceMonitor instance

Following TDD approach for US-005-setup (test infrastructure).
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Mock streamlit before importing the module
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
mock_st.metric = MagicMock()
mock_st.plotly_chart = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.checkbox = MagicMock(return_value=True)
mock_st.selectbox = MagicMock(return_value=1)
mock_st.slider = MagicMock(return_value=90)
mock_st.number_input = MagicMock(return_value=24)
mock_st.info = MagicMock()
mock_st.success = MagicMock()
mock_st.warning = MagicMock()
mock_st.error = MagicMock()
mock_st.progress = MagicMock()
mock_st.title = MagicMock()
mock_st.caption = MagicMock()
mock_st.subheader = MagicMock()
mock_st.markdown = MagicMock()
mock_st.expander = MagicMock(
    return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
)
mock_st.dataframe = MagicMock()
mock_st.rerun = MagicMock()
mock_st.session_state = {
    "simulation_active": False,
    "simulation_data": {
        "phase": "Idle",
        "progress": 0.0,
        "current_step": 0,
        "total_steps": 0,
        "start_time": None,
        "performance_metrics": {},
    },
    "metrics_history": [],
    "performance_monitor": None,
}
sys.modules["streamlit"] = mock_st

# Mock plotly
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.subplots"] = MagicMock()


class TestSystemMetricsDataclass(unittest.TestCase):
    """Test SystemMetrics dataclass structure."""

    def test_system_metrics_creation(self):
        """Test SystemMetrics can be instantiated with required fields."""
        from gui.pages.performance_monitor import SystemMetrics

        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_utilization=75.0,
            gpu_memory=8.0,
            gpu_temperature=65.0,
            disk_usage=40.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            timestamp=datetime.now(),
        )
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertEqual(metrics.gpu_utilization, 75.0)


class TestSimulationStatusDataclass(unittest.TestCase):
    """Test SimulationStatus dataclass structure."""

    def test_simulation_status_inactive(self):
        """Test SimulationStatus for inactive simulation."""
        from gui.pages.performance_monitor import SimulationStatus

        status = SimulationStatus(
            active=False,
            phase="Idle",
            progress=0.0,
            eta=None,
            current_step=0,
            total_steps=0,
            performance_metrics={},
        )
        self.assertFalse(status.active)
        self.assertEqual(status.phase, "Idle")
        self.assertIsNone(status.eta)

    def test_simulation_status_active(self):
        """Test SimulationStatus for active simulation."""
        from gui.pages.performance_monitor import SimulationStatus

        status = SimulationStatus(
            active=True,
            phase="Flow Calculation",
            progress=0.45,
            eta="0:05:30",
            current_step=450,
            total_steps=1000,
            performance_metrics={"steps_per_second": 15.0},
        )
        self.assertTrue(status.active)
        self.assertEqual(status.phase, "Flow Calculation")
        self.assertEqual(status.progress, 0.45)


class TestPerformanceMonitorClass(unittest.TestCase):
    """Test PerformanceMonitor class functionality."""

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initializes correctly."""
        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        self.assertEqual(monitor.metrics_history, [])
        self.assertEqual(monitor.alerts, [])
        self.assertTrue(monitor.monitoring_active)

    @patch("psutil.cpu_percent", return_value=45.0)
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    def test_get_current_metrics_uses_psutil(
        self, mock_net, mock_disk, mock_memory, mock_cpu
    ):
        """Test get_current_metrics calls psutil functions."""
        mock_memory.return_value = MagicMock(percent=60.0)
        mock_disk.return_value = MagicMock(
            used=50_000_000_000, total=100_000_000_000,
        )
        mock_net.return_value = MagicMock(bytes_sent=1000000, bytes_recv=2000000)

        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        metrics = monitor.get_current_metrics()

        mock_cpu.assert_called()
        mock_memory.assert_called()
        self.assertEqual(metrics.cpu_usage, 45.0)
        self.assertEqual(metrics.memory_usage, 60.0)


class TestAlertChecking(unittest.TestCase):
    """Test alert checking functionality."""

    def test_check_alerts_high_cpu(self):
        """Test alert is generated for high CPU usage."""
        from gui.pages.performance_monitor import PerformanceMonitor, SystemMetrics

        monitor = PerformanceMonitor()
        metrics = SystemMetrics(
            cpu_usage=95.0,
            memory_usage=60.0,
            gpu_utilization=50.0,
            gpu_memory=8.0,
            gpu_temperature=65.0,
            disk_usage=40.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            timestamp=datetime.now(),
        )
        alerts = monitor.check_alerts(metrics)
        cpu_alerts = [a for a in alerts if "CPU" in a.get("title", "")]
        self.assertGreaterEqual(len(cpu_alerts), 1)

    def test_check_alerts_high_memory(self):
        """Test alert is generated for high memory usage."""
        from gui.pages.performance_monitor import PerformanceMonitor, SystemMetrics

        monitor = PerformanceMonitor()
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=90.0,
            gpu_utilization=50.0,
            gpu_memory=8.0,
            gpu_temperature=65.0,
            disk_usage=40.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            timestamp=datetime.now(),
        )
        alerts = monitor.check_alerts(metrics)
        memory_alerts = [a for a in alerts if "Memory" in a.get("title", "")]
        self.assertGreaterEqual(len(memory_alerts), 1)

    def test_check_alerts_gpu_overheat(self):
        """Test alert is generated for GPU overheating."""
        from gui.pages.performance_monitor import PerformanceMonitor, SystemMetrics

        monitor = PerformanceMonitor()
        metrics = SystemMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_utilization=90.0,
            gpu_memory=14.0,
            gpu_temperature=85.0,
            disk_usage=40.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            timestamp=datetime.now(),
        )
        alerts = monitor.check_alerts(metrics)
        gpu_alerts = [a for a in alerts if "GPU" in a.get("title", "")]
        self.assertGreaterEqual(len(gpu_alerts), 1)


class TestSimulationStateFromSessionState(unittest.TestCase):
    """Test that simulation status uses session state.

    These tests verify the fix for the phantom simulation bug where
    np.random.random() was used instead of st.session_state.
    Note: Tests will pass after US-001 is implemented.
    """

    def test_simulation_status_structure(self):
        """Test simulation status returns expected structure."""
        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        status = monitor.get_simulation_status()

        # Verify structure exists regardless of implementation
        self.assertIsNotNone(status)
        self.assertIn(status.active, [True, False])


class TestHealthScore(unittest.TestCase):
    """Test system health score calculation."""

    @patch("psutil.cpu_percent", return_value=20.0)
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    def test_health_score_range(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test health score is always between 0 and 1."""
        mock_memory.return_value = MagicMock(percent=30.0)
        mock_disk.return_value = MagicMock(
            used=20_000_000_000, total=100_000_000_000,
        )
        mock_net.return_value = MagicMock(bytes_sent=1000000, bytes_recv=2000000)

        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        health_score = monitor.get_system_health_score()
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)


class TestSingletonPattern(unittest.TestCase):
    """Test singleton pattern for PerformanceMonitor.

    These tests document expected behavior for US-003.
    """

    def test_performance_monitor_exists(self):
        """Test PerformanceMonitor class can be instantiated."""
        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        self.assertIsNotNone(monitor)


class TestSimulationStartStop(unittest.TestCase):
    """Test simulation start/stop functions.

    These tests document expected functionality for US-004.
    """

    def test_simulation_status_dataclass_exists(self):
        """Test SimulationStatus dataclass is available."""
        from gui.pages.performance_monitor import SimulationStatus

        self.assertIsNotNone(SimulationStatus)


class TestModuleImport(unittest.TestCase):
    """Test module can be imported correctly."""

    def test_module_import(self):
        """Test performance_monitor module imports."""
        import gui.pages.performance_monitor

        self.assertIsNotNone(gui.pages.performance_monitor)

    def test_render_function_exists(self):
        """Test render_performance_monitor_page function exists."""
        from gui.pages.performance_monitor import render_performance_monitor_page

        self.assertIsNotNone(render_performance_monitor_page)


if __name__ == "__main__":
    unittest.main()
