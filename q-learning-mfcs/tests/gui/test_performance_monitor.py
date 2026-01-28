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


class SessionStateMock(dict):
    """Mock class for st.session_state that supports both dict and attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def create_default_session_state():
    """Create a default session state for testing."""
    return SessionStateMock({
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
    })


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
mock_st.session_state = create_default_session_state()
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
    """

    def setUp(self):
        """Reset session state and patch the module's st reference."""
        mock_st.session_state = create_default_session_state()
        # Patch the module's st reference directly
        import gui.pages.performance_monitor as pm
        pm.st = mock_st

    def test_simulation_status_structure(self):
        """Test simulation status returns expected structure."""
        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        status = monitor.get_simulation_status()

        # Verify structure exists regardless of implementation
        self.assertIsNotNone(status)
        self.assertIn(status.active, [True, False])

    def test_simulation_inactive_by_default(self):
        """Test that simulation is inactive when session state is False."""
        from gui.pages.performance_monitor import PerformanceMonitor

        mock_st.session_state["simulation_active"] = False
        monitor = PerformanceMonitor()
        status = monitor.get_simulation_status()

        self.assertFalse(status.active)
        self.assertEqual(status.phase, "Idle")
        self.assertEqual(status.progress, 0.0)
        self.assertEqual(status.current_step, 0)

    def test_simulation_active_from_session_state(self):
        """Test that simulation is active when session state is True."""
        from gui.pages.performance_monitor import PerformanceMonitor

        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Flow Calculation",
            "progress": 0.45,
            "current_step": 450,
            "total_steps": 1000,
            "start_time": None,
            "performance_metrics": {"memory_efficiency": 0.9},
        }
        monitor = PerformanceMonitor()
        status = monitor.get_simulation_status()

        self.assertTrue(status.active)
        self.assertEqual(status.phase, "Flow Calculation")
        self.assertEqual(status.progress, 0.45)
        self.assertEqual(status.current_step, 450)
        self.assertEqual(status.total_steps, 1000)

    def test_simulation_status_consistent_across_calls(self):
        """Test simulation status is consistent (not random) across calls."""
        from gui.pages.performance_monitor import PerformanceMonitor

        mock_st.session_state["simulation_active"] = False
        monitor = PerformanceMonitor()

        # Get status multiple times - should be consistent since it's from state
        status1 = monitor.get_simulation_status()
        status2 = monitor.get_simulation_status()
        status3 = monitor.get_simulation_status()

        self.assertEqual(status1.active, status2.active)
        self.assertEqual(status2.active, status3.active)
        self.assertEqual(status1.phase, status2.phase)
        self.assertEqual(status2.phase, status3.phase)

    def test_simulation_status_uses_performance_metrics_from_state(self):
        """Test that performance metrics are taken from session state."""
        from gui.pages.performance_monitor import PerformanceMonitor

        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Optimization",
            "progress": 0.8,
            "current_step": 800,
            "total_steps": 1000,
            "start_time": None,
            "performance_metrics": {
                "memory_efficiency": 0.95,
                "convergence_rate": 0.005,
                "acceleration_factor": 7000,
            },
        }
        monitor = PerformanceMonitor()
        status = monitor.get_simulation_status()

        self.assertEqual(
            status.performance_metrics.get("memory_efficiency"),
            0.95,
        )
        self.assertEqual(status.performance_metrics.get("convergence_rate"), 0.005)
        self.assertEqual(status.performance_metrics.get("acceleration_factor"), 7000)


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

    These tests verify the fix for US-003 - Multiple PerformanceMonitor Instances.
    """

    def setUp(self):
        """Reset session state before each test."""
        mock_st.session_state = create_default_session_state()
        # Patch the module's st reference directly
        import gui.pages.performance_monitor as pm
        pm.st = mock_st

    def test_performance_monitor_exists(self):
        """Test PerformanceMonitor class can be instantiated."""
        from gui.pages.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor()
        self.assertIsNotNone(monitor)

    def test_get_performance_monitor_function_exists(self):
        """Test get_performance_monitor helper function exists."""
        from gui.pages.performance_monitor import get_performance_monitor

        self.assertIsNotNone(get_performance_monitor)

    def test_get_performance_monitor_returns_instance(self):
        """Test get_performance_monitor returns a PerformanceMonitor instance."""
        from gui.pages.performance_monitor import (
            PerformanceMonitor,
            get_performance_monitor,
        )

        # Ensure no monitor exists in session state
        if "performance_monitor" in mock_st.session_state:
            del mock_st.session_state["performance_monitor"]

        monitor = get_performance_monitor()
        self.assertIsInstance(monitor, PerformanceMonitor)

    def test_get_performance_monitor_creates_singleton(self):
        """Test get_performance_monitor stores instance in session state."""
        from gui.pages.performance_monitor import get_performance_monitor

        # Initially no monitor in session state - delete key if exists
        if "performance_monitor" in mock_st.session_state:
            del mock_st.session_state["performance_monitor"]

        monitor = get_performance_monitor()

        # Now monitor should be in session state
        self.assertIsNotNone(mock_st.session_state.get("performance_monitor"))
        self.assertIs(mock_st.session_state["performance_monitor"], monitor)

    def test_get_performance_monitor_returns_same_instance(self):
        """Test get_performance_monitor returns same instance on multiple calls."""
        from gui.pages.performance_monitor import get_performance_monitor

        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        monitor3 = get_performance_monitor()

        # All calls should return the exact same instance
        self.assertIs(monitor1, monitor2)
        self.assertIs(monitor2, monitor3)

    def test_get_performance_monitor_reuses_existing(self):
        """Test get_performance_monitor reuses existing instance from state."""
        from gui.pages.performance_monitor import (
            PerformanceMonitor,
            get_performance_monitor,
        )

        # Pre-create a monitor in session state
        existing_monitor = PerformanceMonitor()
        existing_monitor.custom_marker = "test_marker"
        mock_st.session_state["performance_monitor"] = existing_monitor

        # get_performance_monitor should return the existing instance
        retrieved_monitor = get_performance_monitor()

        self.assertIs(retrieved_monitor, existing_monitor)
        self.assertEqual(retrieved_monitor.custom_marker, "test_marker")


class TestSimulationStartStop(unittest.TestCase):
    """Test simulation start/stop functions.

    These tests verify US-004 - Simulation State Management.
    """

    def setUp(self):
        """Reset session state before each test."""
        mock_st.session_state = create_default_session_state()
        # Patch the module's st reference directly
        import gui.pages.performance_monitor as pm
        pm.st = mock_st

    def test_simulation_status_dataclass_exists(self):
        """Test SimulationStatus dataclass is available."""
        from gui.pages.performance_monitor import SimulationStatus

        self.assertIsNotNone(SimulationStatus)

    def test_start_simulation_function_exists(self):
        """Test start_simulation function is available."""
        from gui.pages.performance_monitor import start_simulation

        self.assertIsNotNone(start_simulation)

    def test_stop_simulation_function_exists(self):
        """Test stop_simulation function is available."""
        from gui.pages.performance_monitor import stop_simulation

        self.assertIsNotNone(stop_simulation)

    def test_start_simulation_sets_active_state(self):
        """Test start_simulation sets simulation_active to True."""
        from gui.pages.performance_monitor import start_simulation

        # Initially inactive
        mock_st.session_state["simulation_active"] = False

        start_simulation()

        self.assertTrue(mock_st.session_state["simulation_active"])

    def test_start_simulation_initializes_simulation_data(self):
        """Test start_simulation initializes simulation_data with proper structure."""
        from gui.pages.performance_monitor import start_simulation

        start_simulation(total_steps=500)

        sim_data = mock_st.session_state["simulation_data"]
        self.assertEqual(sim_data["phase"], "Initialization")
        self.assertEqual(sim_data["progress"], 0.0)
        self.assertEqual(sim_data["current_step"], 0)
        self.assertEqual(sim_data["total_steps"], 500)
        self.assertIsNotNone(sim_data["start_time"])
        self.assertIsInstance(sim_data["performance_metrics"], dict)

    def test_start_simulation_default_total_steps(self):
        """Test start_simulation uses default total_steps of 1000."""
        from gui.pages.performance_monitor import start_simulation

        start_simulation()

        sim_data = mock_st.session_state["simulation_data"]
        self.assertEqual(sim_data["total_steps"], 1000)

    def test_start_simulation_records_start_time(self):
        """Test start_simulation records start_time for ETA calculation."""
        from gui.pages.performance_monitor import start_simulation

        before = datetime.now()
        start_simulation()
        after = datetime.now()

        start_time = mock_st.session_state["simulation_data"]["start_time"]
        self.assertIsNotNone(start_time)
        self.assertGreaterEqual(start_time, before)
        self.assertLessEqual(start_time, after)

    def test_stop_simulation_sets_inactive_state(self):
        """Test stop_simulation sets simulation_active to False."""
        from gui.pages.performance_monitor import stop_simulation

        # Initially active
        mock_st.session_state["simulation_active"] = True

        stop_simulation()

        self.assertFalse(mock_st.session_state["simulation_active"])

    def test_stop_simulation_resets_simulation_data(self):
        """Test stop_simulation resets simulation_data to idle state."""
        from gui.pages.performance_monitor import stop_simulation

        # Set up active simulation
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Flow Calculation",
            "progress": 0.5,
            "current_step": 500,
            "total_steps": 1000,
            "start_time": datetime.now(),
            "performance_metrics": {"steps_per_second": 10.0},
        }

        stop_simulation()

        sim_data = mock_st.session_state["simulation_data"]
        self.assertEqual(sim_data["phase"], "Idle")
        self.assertEqual(sim_data["progress"], 0.0)
        self.assertEqual(sim_data["current_step"], 0)
        self.assertEqual(sim_data["total_steps"], 0)
        self.assertIsNone(sim_data["start_time"])
        self.assertEqual(sim_data["performance_metrics"], {})

    def test_start_stop_cycle(self):
        """Test start and stop simulation cycle works correctly."""
        from gui.pages.performance_monitor import start_simulation, stop_simulation

        # Initial state
        self.assertFalse(mock_st.session_state.get("simulation_active", False))

        # Start simulation
        start_simulation()
        self.assertTrue(mock_st.session_state["simulation_active"])
        self.assertEqual(
            mock_st.session_state["simulation_data"]["phase"], "Initialization"
        )

        # Stop simulation
        stop_simulation()
        self.assertFalse(mock_st.session_state["simulation_active"])
        self.assertEqual(mock_st.session_state["simulation_data"]["phase"], "Idle")

    def test_start_simulation_initializes_performance_metrics(self):
        """Test start_simulation initializes performance_metrics with default values."""
        from gui.pages.performance_monitor import start_simulation

        start_simulation()

        perf = mock_st.session_state["simulation_data"]["performance_metrics"]
        self.assertEqual(perf["steps_per_second"], 0.0)
        self.assertEqual(perf["memory_efficiency"], 0.0)
        self.assertEqual(perf["convergence_rate"], 0.0)
        self.assertEqual(perf["acceleration_factor"], 1.0)


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
