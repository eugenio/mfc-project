"""Deep coverage tests for performance_monitor and advanced_physics page modules."""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_DIR = str((Path(__file__).resolve().parent / ".." / ".." / "src").resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_GUI_PREFIX = "gui."


@pytest.fixture(autouse=True)
def _clear_module_cache():
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if mod_name == "gui" or mod_name.startswith(_GUI_PREFIX):
            del sys.modules[mod_name]


class _SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


def _make_mock_st():
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()

    def _smart_columns(n_or_spec):
        n = len(n_or_spec) if isinstance(n_or_spec, list | tuple) else int(n_or_spec)
        cols = []
        for _ in range(n):
            col = MagicMock()
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=False)
            cols.append(col)
        return cols

    mock_st.columns.side_effect = _smart_columns

    def _smart_tabs(labels):
        tabs = []
        for _ in labels:
            tab = MagicMock()
            tab.__enter__ = MagicMock(return_value=tab)
            tab.__exit__ = MagicMock(return_value=False)
            tabs.append(tab)
        return tabs

    mock_st.tabs.side_effect = _smart_tabs

    exp = MagicMock()
    exp.__enter__ = MagicMock(return_value=exp)
    exp.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = exp

    mock_st.sidebar = MagicMock()

    form = MagicMock()
    form.__enter__ = MagicMock(return_value=form)
    form.__exit__ = MagicMock(return_value=False)
    mock_st.form.return_value = form

    spinner = MagicMock()
    spinner.__enter__ = MagicMock(return_value=spinner)
    spinner.__exit__ = MagicMock(return_value=False)
    mock_st.spinner.return_value = spinner

    container = MagicMock()
    container.__enter__ = MagicMock(return_value=container)
    container.__exit__ = MagicMock(return_value=False)
    mock_st.container.return_value = container

    status = MagicMock()
    status.__enter__ = MagicMock(return_value=status)
    status.__exit__ = MagicMock(return_value=False)
    mock_st.status.return_value = status

    progress_bar = MagicMock()
    progress_bar.progress = MagicMock()
    progress_bar.empty = MagicMock()
    mock_st.progress.return_value = progress_bar

    return mock_st


# ────────────────────────────────────────────────────────────────────
# Helpers for importing modules under mock
# ────────────────────────────────────────────────────────────────────


def _build_mock_np():
    """Build a mock numpy that supports basic operations."""
    mock_np = MagicMock()
    mock_np.random.uniform.return_value = 50.0
    mock_np.random.random.return_value = 0.5
    mock_np.mean.return_value = 50.0
    mock_np.max.return_value = 0.001
    mock_np.zeros.side_effect = (
        lambda shape: __import__("numpy").zeros(shape)
    )
    mock_np.linspace.side_effect = (
        lambda *a, **kw: __import__("numpy").linspace(*a, **kw)
    )
    mock_np.exp.side_effect = lambda x: __import__("numpy").exp(x)
    return mock_np


def _build_mock_pd():
    """Build a mock pandas."""
    mock_pd = MagicMock()
    mock_pd.DataFrame.return_value = MagicMock()
    return mock_pd


def _build_mock_go():
    """Build a mock plotly.graph_objects."""
    return MagicMock()


def _build_mock_subplots():
    """Build a mock plotly.subplots."""
    mock_sub = MagicMock()
    mock_sub.make_subplots.return_value = MagicMock()
    return mock_sub


def _build_mock_psutil():
    """Build a mock psutil with proper return values."""
    mock_psutil = MagicMock()
    mock_psutil.cpu_percent.return_value = 45.0

    mem = MagicMock()
    mem.percent = 60.0
    mock_psutil.virtual_memory.return_value = mem

    disk = MagicMock()
    disk.used = 275_000_000_000
    disk.total = 500_000_000_000
    mock_psutil.disk_usage.return_value = disk

    net = MagicMock()
    net.bytes_sent = 1000
    net.bytes_recv = 2000
    mock_psutil.net_io_counters.return_value = net

    return mock_psutil


import importlib  # noqa: E402
import types as _types  # noqa: E402


def _make_gui_package():
    """Create a minimal mock gui package so gui.pages.X can be imported."""
    gui_pkg = _types.ModuleType("gui")
    gui_pkg.__path__ = []
    gui_pkg.__package__ = "gui"
    gui_pages = _types.ModuleType("gui.pages")
    gui_pages.__path__ = [
        str(Path(_SRC_DIR) / "gui" / "pages"),
    ]
    gui_pages.__package__ = "gui.pages"
    gui_pkg.pages = gui_pages
    return gui_pkg, gui_pages


def _safe_exec_module(mod_name, file_path, extra_mods):
    """Import a module with temporary sys.modules entries, preserving others.

    Unlike patch.dict, this does NOT remove entries that were added to
    sys.modules during exec_module (e.g. numpy submodules), which would
    corrupt the import cache.
    """
    saved = {}
    for k, v in extra_mods.items():
        if k in sys.modules:
            saved[k] = sys.modules[k]
        sys.modules[k] = v

    try:
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k in extra_mods:
            if k in saved:
                sys.modules[k] = saved[k]
            elif k in sys.modules and sys.modules[k] is extra_mods[k]:
                del sys.modules[k]

    return mod


def _import_perf_monitor(mock_st, mock_np=None, mock_pd=None,
                         mock_go=None, mock_subplots=None,
                         mock_psutil=None):
    """Import performance_monitor module with mocks."""
    if mock_np is None:
        mock_np = _build_mock_np()
    if mock_pd is None:
        mock_pd = _build_mock_pd()
    if mock_go is None:
        mock_go = _build_mock_go()
    if mock_subplots is None:
        mock_subplots = _build_mock_subplots()
    if mock_psutil is None:
        mock_psutil = _build_mock_psutil()

    gui_pkg, gui_pages = _make_gui_package()

    mods = {
        "streamlit": mock_st,
        "psutil": mock_psutil,
        "plotly": MagicMock(),
        "plotly.graph_objects": mock_go,
        "plotly.subplots": mock_subplots,
        "gui": gui_pkg,
        "gui.pages": gui_pages,
    }

    pm = _safe_exec_module(
        "gui.pages.performance_monitor",
        str(Path(_SRC_DIR) / "gui" / "pages" / "performance_monitor.py"),
        mods,
    )

    pm.st = mock_st
    pm.np = mock_np
    pm.pd = mock_pd
    pm.go = mock_go
    pm.make_subplots = mock_subplots.make_subplots
    pm.psutil = mock_psutil
    return pm


def _import_advanced_physics(mock_st, mock_np=None, mock_go=None):
    """Import advanced_physics module with mocks."""
    if mock_np is None:
        mock_np = _build_mock_np()
    if mock_go is None:
        mock_go = _build_mock_go()

    mock_sw = MagicMock()
    mock_param_spec = MagicMock()
    mock_sw.ParameterSpec = mock_param_spec
    mock_widget_instance = MagicMock()
    mock_widget_instance.render.return_value = 1e-4
    mock_sw.ScientificParameterWidget.return_value = mock_widget_instance

    gui_pkg, gui_pages = _make_gui_package()

    mods = {
        "streamlit": mock_st,
        "plotly": MagicMock(),
        "plotly.graph_objects": mock_go,
        "gui": gui_pkg,
        "gui.pages": gui_pages,
        "gui.scientific_widgets": mock_sw,
    }

    ap = _safe_exec_module(
        "gui.pages.advanced_physics",
        str(Path(_SRC_DIR) / "gui" / "pages" / "advanced_physics.py"),
        mods,
    )

    ap.st = mock_st
    ap.np = mock_np
    ap.go = mock_go
    ap.ParameterSpec = mock_param_spec
    ap.ScientificParameterWidget = mock_sw.ScientificParameterWidget
    return ap


# ====================================================================
# PERFORMANCE MONITOR TESTS
# ====================================================================


@pytest.mark.apptest
class TestPerfMonDataclasses:
    """Test SystemMetrics and SimulationStatus dataclasses."""

    def test_system_metrics_creation(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        m = pm.SystemMetrics(
            cpu_usage=45.0,
            memory_usage=60.0,
            gpu_utilization=70.0,
            gpu_memory=8.0,
            gpu_temperature=65.0,
            disk_usage=55.0,
            network_io={"bytes_sent": 100, "bytes_recv": 200},
            timestamp=datetime.now(),
        )
        assert m.cpu_usage == 45.0
        assert m.gpu_temperature == 65.0

    def test_simulation_status_creation(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        s = pm.SimulationStatus(
            active=True,
            phase="Training",
            progress=0.5,
            eta="0:10:00",
            current_step=500,
            total_steps=1000,
            performance_metrics={"steps_per_second": 50.0},
        )
        assert s.active is True
        assert s.phase == "Training"
        assert s.progress == 0.5


@pytest.mark.apptest
class TestGetPerformanceMonitor:
    """Test get_performance_monitor singleton (lines 63-65)."""

    def test_creates_new_when_missing(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        result = pm.get_performance_monitor()
        assert result is not None
        assert isinstance(result, pm.PerformanceMonitor)
        assert "performance_monitor" in mock_st.session_state

    def test_returns_existing(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        existing = pm.PerformanceMonitor()
        mock_st.session_state["performance_monitor"] = existing
        result = pm.get_performance_monitor()
        assert result is existing


@pytest.mark.apptest
class TestStartSimulation:
    """Test start_simulation (lines 80-81)."""

    def test_sets_state(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        pm.start_simulation(total_steps=500)
        assert mock_st.session_state.simulation_active is True
        d = mock_st.session_state.simulation_data
        assert d["total_steps"] == 500
        assert d["phase"] == "Initialization"
        assert d["progress"] == 0.0
        assert d["current_step"] == 0
        assert d["start_time"] is not None

    def test_default_steps(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        pm.start_simulation()
        assert mock_st.session_state.simulation_data["total_steps"] == 1000


@pytest.mark.apptest
class TestStopSimulation:
    """Test stop_simulation (lines 101-102)."""

    def test_resets_state(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        pm.start_simulation()
        pm.stop_simulation()
        assert mock_st.session_state.simulation_active is False
        d = mock_st.session_state.simulation_data
        assert d["phase"] == "Idle"
        assert d["progress"] == 0.0
        assert d["start_time"] is None
        assert d["performance_metrics"] == {}


@pytest.mark.apptest
class TestPerformanceMonitorInit:
    """Test PerformanceMonitor __init__."""

    def test_init(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        assert m.metrics_history == []
        assert m.alerts == []
        assert m.monitoring_active is True


@pytest.mark.apptest
class TestGetCurrentMetrics:
    """Test get_current_metrics."""

    def test_returns_system_metrics(self):
        mock_st = _make_mock_st()
        mock_psutil = _build_mock_psutil()
        pm = _import_perf_monitor(mock_st, mock_psutil=mock_psutil)
        m = pm.PerformanceMonitor()
        metrics = m.get_current_metrics()
        assert metrics.cpu_usage == 45.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == pytest.approx(55.0, abs=0.01)
        assert metrics.network_io["bytes_sent"] == 1000

    def test_gpu_high_random(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.random.return_value = 0.5
        mock_np.random.uniform.return_value = 75.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        m = pm.PerformanceMonitor()
        metrics = m.get_current_metrics()
        assert metrics.gpu_utilization == 75.0

    def test_gpu_low_random(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.random.return_value = 0.2
        mock_np.random.uniform.return_value = 10.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        m = pm.PerformanceMonitor()
        metrics = m.get_current_metrics()
        assert metrics.gpu_utilization == 10.0


@pytest.mark.apptest
class TestGetSimulationStatus:
    """Test get_simulation_status (lines 166-206)."""

    def test_inactive(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = False
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.active is False
        assert s.phase == "Idle"
        assert s.progress == 0.0
        assert s.eta is None

    def test_active_no_data(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.active is True
        assert s.phase == "Initialization"

    def test_active_with_eta(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Training",
            "progress": 0.5,
            "current_step": 500,
            "total_steps": 1000,
            "start_time": datetime.now() - timedelta(seconds=10),
            "performance_metrics": {"steps_per_second": 0.0},
        }
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.active is True
        assert s.eta is not None

    def test_active_zero_progress(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Init",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 1000,
            "start_time": datetime.now(),
            "performance_metrics": {},
        }
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.eta is None

    def test_active_no_start_time(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "Running",
            "progress": 0.3,
            "current_step": 300,
            "total_steps": 1000,
            "start_time": None,
            "performance_metrics": {},
        }
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.eta is None

    def test_active_non_dict_sim_data(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = "invalid"
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.phase == "Initialization"

    def test_active_non_dict_perf_metrics(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "X",
            "progress": 0.1,
            "current_step": 100,
            "total_steps": 1000,
            "start_time": None,
            "performance_metrics": "bad",
        }
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.performance_metrics == {}

    def test_active_none_value(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        mock_st.session_state["simulation_active"] = None
        m = pm.PerformanceMonitor()
        s = m.get_simulation_status()
        assert s.active is False


@pytest.mark.apptest
class TestGpuAccelerationMetrics:
    """Test get_gpu_acceleration_metrics (line 210)."""

    def test_keys_present(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 5000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        m = pm.PerformanceMonitor()
        r = m.get_gpu_acceleration_metrics()
        for k in ("current_speedup", "target_speedup", "efficiency",
                   "memory_bandwidth", "compute_utilization",
                   "memory_utilization"):
            assert k in r
        assert r["target_speedup"] == 8400


@pytest.mark.apptest
class TestSystemHealthScore:
    """Test get_system_health_score (lines 227-234)."""

    def test_normal(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        score = m.get_system_health_score()
        assert 0.0 <= score <= 1.0

    def test_all_maxed(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 100.0
        mem = MagicMock()
        mem.percent = 100.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 500_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 100.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        m = pm.PerformanceMonitor()
        assert m.get_system_health_score() == 0.0

    def test_all_low(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 0.0
        mem = MagicMock()
        mem.percent = 0.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 0
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 50.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        m = pm.PerformanceMonitor()
        assert m.get_system_health_score() == 1.0


@pytest.mark.apptest
class TestCheckAlerts:
    """Test check_alerts (lines 238-286)."""

    def _metrics(self, pm, cpu=45, mem=60, gpu_util=70,
                 gpu_temp=65, disk=55):
        return pm.SystemMetrics(
            cpu_usage=cpu, memory_usage=mem,
            gpu_utilization=gpu_util, gpu_memory=8.0,
            gpu_temperature=gpu_temp, disk_usage=disk,
            network_io={"bytes_sent": 100, "bytes_recv": 200},
            timestamp=datetime.now(),
        )

    def test_no_alerts(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        assert len(m.check_alerts(self._metrics(pm))) == 0

    def test_cpu_alert(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, cpu=95))
        assert any(a["title"] == "High CPU Usage" for a in alerts)

    def test_memory_alert(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, mem=90))
        assert any(a["title"] == "High Memory Usage" for a in alerts)

    def test_gpu_temp_alert(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, gpu_temp=85))
        t = [a for a in alerts if a["title"] == "GPU Overheating"]
        assert len(t) == 1
        assert t[0]["type"] == "error"

    def test_gpu_util_alert_during_sim(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "R", "progress": 0.5, "current_step": 500,
            "total_steps": 1000, "start_time": None,
            "performance_metrics": {},
        }
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, gpu_util=20))
        assert any(a["title"] == "Low GPU Utilization" for a in alerts)

    def test_no_gpu_util_alert_no_sim(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, gpu_util=10))
        assert not any(a["title"] == "Low GPU Utilization" for a in alerts)

    def test_all_alerts(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "R", "progress": 0.5, "current_step": 500,
            "total_steps": 1000, "start_time": None,
            "performance_metrics": {},
        }
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(
            self._metrics(pm, cpu=95, mem=90, gpu_temp=85, gpu_util=10),
        )
        assert len(alerts) == 4

    def test_boundary_cpu_90(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, cpu=90))
        assert not any(a["title"] == "High CPU Usage" for a in alerts)

    def test_boundary_mem_85(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, mem=85))
        assert not any(a["title"] == "High Memory Usage" for a in alerts)

    def test_boundary_gpu_temp_80(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = False
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, gpu_temp=80))
        assert not any(a["title"] == "GPU Overheating" for a in alerts)

    def test_boundary_gpu_util_30(self):
        mock_st = _make_mock_st()
        mock_st.session_state["simulation_active"] = True
        mock_st.session_state["simulation_data"] = {
            "phase": "R", "progress": 0.5, "current_step": 500,
            "total_steps": 1000, "start_time": None,
            "performance_metrics": {},
        }
        pm = _import_perf_monitor(mock_st)
        m = pm.PerformanceMonitor()
        alerts = m.check_alerts(self._metrics(pm, gpu_util=30))
        assert not any(
            a["title"] == "Low GPU Utilization" for a in alerts
        )


@pytest.mark.apptest
class TestCreateRealTimeCharts:
    """Test create_real_time_charts (lines 291-401)."""

    def _mlist(self, pm, n=5):
        out = []
        for i in range(n):
            out.append(pm.SystemMetrics(
                cpu_usage=40.0 + i, memory_usage=55.0 + i,
                gpu_utilization=60.0 + i, gpu_memory=8.0,
                gpu_temperature=65.0 + i, disk_usage=50.0,
                network_io={"bytes_sent": 100, "bytes_recv": 200},
                timestamp=datetime.now(),
            ))
        return out

    def test_empty(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        pm.create_real_time_charts([])
        mock_st.info.assert_called_once()

    def test_with_data(self):
        mock_st = _make_mock_st()
        mock_pd = _build_mock_pd()
        mock_go = _build_mock_go()
        mock_sub = _build_mock_subplots()
        pm = _import_perf_monitor(mock_st, mock_pd=mock_pd,
                                  mock_go=mock_go, mock_subplots=mock_sub)
        pm.create_real_time_charts(self._mlist(pm, 5))
        mock_pd.DataFrame.assert_called_once()
        assert mock_st.plotly_chart.called

    def test_more_than_50(self):
        mock_st = _make_mock_st()
        mock_pd = _build_mock_pd()
        pm = _import_perf_monitor(mock_st, mock_pd=mock_pd)
        pm.create_real_time_charts(self._mlist(pm, 60))
        args = mock_pd.DataFrame.call_args[0][0]
        assert len(args) == 50


@pytest.mark.apptest
class TestCreateAccelerationDashboard:
    """Test create_acceleration_dashboard (lines 406-461)."""

    def test_high_speedup(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 8000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        pm.create_acceleration_dashboard()
        mock_st.success.assert_called()

    def test_medium_speedup(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 5000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        pm.create_acceleration_dashboard()
        mock_st.info.assert_called()

    def test_low_speedup(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 3000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        pm.create_acceleration_dashboard()
        mock_st.warning.assert_called()

    def test_metrics_displayed(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 7000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        pm.create_acceleration_dashboard()
        assert mock_st.metric.called
        assert mock_st.columns.called

    def test_progress_bar(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 5000.0
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        pm.create_acceleration_dashboard()
        mock_st.progress.assert_called()


@pytest.mark.apptest
class TestRenderPerfPage:
    """Test render_performance_monitor_page (lines 464-756+)."""

    def _setup(self, mock_st, auto_refresh=False,
               sim_active=False, sim_data=None, btns=None):
        mock_st.checkbox.return_value = auto_refresh
        mock_st.selectbox.return_value = 2
        if btns is None:
            btns = [False] * 8
        mock_st.button.side_effect = btns
        if sim_active:
            mock_st.session_state["simulation_active"] = True
            mock_st.session_state["simulation_data"] = sim_data or {
                "phase": "Training", "progress": 0.5,
                "current_step": 500, "total_steps": 1000,
                "start_time": datetime.now() - timedelta(seconds=10),
                "performance_metrics": {
                    "steps_per_second": 50.0,
                    "memory_efficiency": 0.8,
                    "convergence_rate": 0.001,
                    "acceleration_factor": 4200.0,
                },
            }
        else:
            mock_st.session_state["simulation_active"] = False

    def test_renders_title(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        mock_st.title.assert_called()

    def test_no_auto_refresh(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, auto_refresh=False)
        pm.render_performance_monitor_page()
        mock_st.title.assert_called()

    def test_auto_refresh(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, auto_refresh=True)
        with patch("time.sleep"):
            pm.render_performance_monitor_page()
        mock_st.rerun.assert_called()

    def test_refresh_now_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, btns=[True] + [False] * 7)
        pm.render_performance_monitor_page()
        mock_st.rerun.assert_called()

    def test_sim_idle(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=False)
        pm.render_performance_monitor_page()
        mock_st.info.assert_called()

    def test_sim_active(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=True)
        pm.render_performance_monitor_page()
        mock_st.success.assert_called()
        mock_st.progress.assert_called()

    def test_sim_active_with_eta(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=True)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_stop_sim_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=True,
                     btns=[False, True] + [False] * 6)
        pm.render_performance_monitor_page()
        assert mock_st.session_state.get("simulation_active") is False

    def test_start_test_sim_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        # Idle: [0]=RefreshNow, [1]=StartSim, [2]=LoadPrev, [3]=Export, [4]=Email
        self._setup(mock_st, sim_active=False,
                     btns=[False, True] + [False] * 6)
        pm.render_performance_monitor_page()
        assert mock_st.session_state.get("simulation_active") is True

    def test_load_previous_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        # Idle: [0]=RefreshNow, [1]=StartSim, [2]=LoadPrev, [3]=Export, [4]=Email
        self._setup(mock_st, sim_active=False,
                     btns=[False, False, True] + [False] * 5)
        pm.render_performance_monitor_page()
        assert mock_st.info.called

    def test_health_excellent(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 5.0
        mem = MagicMock()
        mem.percent = 5.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 25_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 50.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_health_good(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 20.0
        mem = MagicMock()
        mem.percent = 20.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 100_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 65.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_health_fair(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 50.0
        mem = MagicMock()
        mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 250_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 75.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_health_poor(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 95.0
        mem = MagicMock()
        mem.percent = 95.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 475_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 95.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_cpu_high_color(self):
        mock_st = _make_mock_st()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 85.0
        pm = _import_perf_monitor(mock_st, mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_memory_high_color(self):
        mock_st = _make_mock_st()
        mock_psutil = _build_mock_psutil()
        mem = MagicMock()
        mem.percent = 90.0
        mock_psutil.virtual_memory.return_value = mem
        pm = _import_perf_monitor(mock_st, mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_gpu_temp_high_color(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 80.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_alerts_error(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 85.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.error.called or mock_st.warning.called

    def test_alerts_warning(self):
        mock_st = _make_mock_st()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 95.0
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 65.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.warning.called

    def test_alerts_info(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_np.random.uniform.return_value = 10.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np)
        self._setup(mock_st, sim_active=True)
        pm.render_performance_monitor_page()
        assert mock_st.info.called

    def test_detailed_stats_with_history(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_pd = _build_mock_pd()
        pm = _import_perf_monitor(mock_st, mock_np=mock_np, mock_pd=mock_pd)
        self._setup(mock_st)
        # Must set performance_monitor so render doesn't reset metrics_history
        mock_st.session_state["performance_monitor"] = pm.PerformanceMonitor()
        mock_st.session_state["metrics_history"] = []
        for i in range(5):
            mock_st.session_state["metrics_history"].append(
                pm.SystemMetrics(
                    cpu_usage=40.0 + i, memory_usage=55.0 + i,
                    gpu_utilization=60.0 + i, gpu_memory=8.0,
                    gpu_temperature=65.0 + i, disk_usage=50.0,
                    network_io={"bytes_sent": 100, "bytes_recv": 200},
                    timestamp=datetime.now(),
                ),
            )
        pm.render_performance_monitor_page()
        assert mock_st.dataframe.called

    def test_history_truncation(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st)
        # Must set performance_monitor so render doesn't reset metrics_history
        mock_st.session_state["performance_monitor"] = pm.PerformanceMonitor()
        mock_st.session_state["metrics_history"] = []
        for _ in range(105):
            mock_st.session_state["metrics_history"].append(
                pm.SystemMetrics(
                    cpu_usage=40.0, memory_usage=55.0,
                    gpu_utilization=60.0, gpu_memory=8.0,
                    gpu_temperature=65.0, disk_usage=50.0,
                    network_io={"bytes_sent": 100, "bytes_recv": 200},
                    timestamp=datetime.now(),
                ),
            )
        pm.render_performance_monitor_page()
        assert len(mock_st.session_state["metrics_history"]) <= 101

    def test_export_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        # Idle: [0]=Refresh,[1]=Start,[2]=Load,[3]=Export,[4]=Email
        self._setup(mock_st, btns=[False] * 3 + [True, False] + [False] * 3)
        pm.render_performance_monitor_page()
        assert mock_st.info.called

    def test_email_btn(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        # Idle: [0]=Refresh,[1]=Start,[2]=Load,[3]=Export,[4]=Email
        self._setup(mock_st, btns=[False] * 4 + [True] + [False] * 3)
        pm.render_performance_monitor_page()
        assert mock_st.info.called

    def test_config_sliders(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        mock_st.slider.assert_called()
        mock_st.number_input.assert_called()

    def test_footer(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        mock_st.markdown.assert_called()
        mock_st.caption.assert_called()

    def test_sim_no_perf_metrics(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=True, sim_data={
            "phase": "R", "progress": 0.5, "current_step": 500,
            "total_steps": 1000, "start_time": None,
            "performance_metrics": {},
        })
        pm.render_performance_monitor_page()
        assert mock_st.success.called

    def test_sim_no_eta(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st, sim_active=True, sim_data={
            "phase": "I", "progress": 0.0, "current_step": 0,
            "total_steps": 1000, "start_time": None,
            "performance_metrics": {"acceleration_factor": 1},
        })
        pm.render_performance_monitor_page()
        assert mock_st.metric.called

    def test_resource_table_critical(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 96.0
        mem = MagicMock()
        mem.percent = 96.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 480_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 5.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.dataframe.called

    def test_resource_table_mid(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 85.0
        mem = MagicMock()
        mem.percent = 90.0
        mock_psutil.virtual_memory.return_value = mem
        disk = MagicMock()
        disk.used = 425_000_000_000
        disk.total = 500_000_000_000
        mock_psutil.disk_usage.return_value = disk
        mock_np.random.uniform.return_value = 30.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.dataframe.called

    def test_single_metric_collecting(self):
        mock_st = _make_mock_st()
        pm = _import_perf_monitor(mock_st)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.info.called

    def test_no_alerts_section(self):
        mock_st = _make_mock_st()
        mock_np = _build_mock_np()
        mock_psutil = _build_mock_psutil()
        mock_psutil.cpu_percent.return_value = 30.0
        mem = MagicMock()
        mem.percent = 40.0
        mock_psutil.virtual_memory.return_value = mem
        mock_np.random.uniform.return_value = 60.0
        mock_np.random.random.return_value = 0.5
        pm = _import_perf_monitor(mock_st, mock_np=mock_np,
                                  mock_psutil=mock_psutil)
        self._setup(mock_st)
        pm.render_performance_monitor_page()
        assert mock_st.title.called


# ====================================================================
# ADVANCED PHYSICS TESTS
# ====================================================================


@pytest.mark.apptest
class TestPhysicsResultDataclass:
    """Test PhysicsSimulationResult dataclass."""

    def test_success(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=True, pressure_drop=100.0, execution_time=1.5,
        )
        assert r.success is True
        assert r.pressure_drop == 100.0

    def test_failure(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=False, error_message="err",
        )
        assert r.success is False
        assert r.flow_field is None

    def test_defaults(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(success=True)
        assert r.concentration_profile is None
        assert r.biofilm_thickness is None


@pytest.mark.apptest
class TestPhysicsSimulatorInit:
    """Test AdvancedPhysicsSimulator __init__ (lines 39-40)."""

    def test_init(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        s = ap.AdvancedPhysicsSimulator()
        assert s.simulation_active is False
        assert s.progress == 0.0


@pytest.mark.apptest
class TestPhysicsRunSimulation:
    """Test run_simulation (lines 44-94)."""

    def test_success(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        progress_bar = MagicMock()
        mock_st.progress.return_value = progress_bar
        sim = ap.AdvancedPhysicsSimulator()
        params = {
            "flow_rate": 1e-4, "diffusivity": 1e-9,
            "growth_rate": 1e-5, "substrate_concentration": 1.0,
            "permeability": 1e-10, "yield_coefficient": 0.4,
        }
        with patch("time.sleep"):
            r = sim.run_simulation(params)
        assert r.success is True
        assert r.flow_field is not None
        assert r.concentration_profile is not None
        assert r.biofilm_thickness is not None
        assert r.pressure_drop is not None
        assert r.mass_transfer_coefficient is not None
        assert sim.simulation_active is False

    def test_exception(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        mock_st.progress.side_effect = RuntimeError("boom")
        sim = ap.AdvancedPhysicsSimulator()
        r = sim.run_simulation({"flow_rate": 1e-4})
        assert r.success is False
        assert "boom" in r.error_message
        assert sim.simulation_active is False


@pytest.mark.apptest
class TestCalcFlowField:
    """Test _calculate_flow_field (lines 99-119)."""

    def test_shape(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        r = sim._calculate_flow_field({"flow_rate": 1e-4})
        assert r.shape == (50, 30, 20, 3)

    def test_default_flow(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        r = sim._calculate_flow_field({})
        assert r.shape == (50, 30, 20, 3)


@pytest.mark.apptest
class TestCalcMassTransport:
    """Test _calculate_mass_transport (lines 127-144)."""

    def test_shape(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        ff = real_np.zeros((50, 30, 20, 3))
        r = sim._calculate_mass_transport(
            {"substrate_concentration": 1.0, "diffusivity": 1e-9}, ff,
        )
        assert r.shape == (50, 30, 20)

    def test_defaults(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        ff = real_np.zeros((10, 10, 10, 3))
        r = sim._calculate_mass_transport({}, ff)
        assert r.shape == (10, 10, 10)


@pytest.mark.apptest
class TestCalcBiofilmGrowth:
    """Test _calculate_biofilm_growth (lines 152-172)."""

    def test_shape(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        conc = real_np.ones((50, 30, 20))
        r = sim._calculate_biofilm_growth(
            {"growth_rate": 1e-5, "yield_coefficient": 0.4}, conc,
        )
        assert r.shape == (50, 30)

    def test_defaults(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        sim = ap.AdvancedPhysicsSimulator()
        conc = real_np.ones((10, 10, 10))
        r = sim._calculate_biofilm_growth({}, conc)
        assert r.shape == (10, 10)


@pytest.mark.apptest
class TestCalcPressureDrop:
    """Test _calculate_pressure_drop (lines 180-186)."""

    def test_calc(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        ff = real_np.zeros((10, 10, 10, 3))
        r = sim._calculate_pressure_drop(
            {"flow_rate": 1e-4, "permeability": 1e-10}, ff,
        )
        assert isinstance(r, float)
        assert r > 0

    def test_defaults(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        ff = real_np.zeros((10, 10, 10, 3))
        r = sim._calculate_pressure_drop({}, ff)
        assert isinstance(r, float)


@pytest.mark.apptest
class TestCalcMassTransferCoeff:
    """Test _calculate_mass_transfer_coefficient (lines 190-209)."""

    def test_laminar(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        r = sim._calculate_mass_transfer_coefficient(
            {"flow_rate": 1e-4, "diffusivity": 1e-9},
        )
        assert isinstance(r, float)
        assert r > 0

    def test_turbulent(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        r = sim._calculate_mass_transfer_coefficient(
            {"flow_rate": 0.5, "diffusivity": 1e-9},
        )
        assert isinstance(r, float)
        assert r > 0

    def test_defaults(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        r = sim._calculate_mass_transfer_coefficient({})
        assert isinstance(r, float)


@pytest.mark.apptest
class TestCalcReynoldsNumber:
    """Test _calculate_reynolds_number (lines 217-220)."""

    def test_low_flow(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        re = sim._calculate_reynolds_number(1e-4, {})
        assert abs(re - 1.0) < 1e-10

    def test_high_flow(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        sim = ap.AdvancedPhysicsSimulator()
        re = sim._calculate_reynolds_number(1.0, {})
        assert abs(re - 10000.0) < 1e-6


@pytest.mark.apptest
class TestCreatePhysicsVisualizations:
    """Test create_physics_visualizations (lines 225-296)."""

    def test_failed(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=False, error_message="err",
        )
        ap.create_physics_visualizations(r)
        mock_st.error.assert_called()

    def test_all_data(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_go = _build_mock_go()
        ap = _import_advanced_physics(mock_st, mock_go=mock_go)
        r = ap.PhysicsSimulationResult(
            success=True,
            flow_field=real_np.zeros((50, 30, 20, 3)),
            concentration_profile=real_np.ones((50, 30, 20)),
            biofilm_thickness=real_np.ones((50, 30)) * 10.0,
            pressure_drop=100.0,
            mass_transfer_coefficient=1e-5,
            execution_time=1.5,
        )
        ap.create_physics_visualizations(r)
        assert mock_st.plotly_chart.called

    def test_no_flow_field(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=True, flow_field=None,
            concentration_profile=real_np.ones((50, 30, 20)),
            biofilm_thickness=real_np.ones((50, 30)),
        )
        ap.create_physics_visualizations(r)
        assert mock_st.columns.called

    def test_no_concentration(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=True,
            flow_field=real_np.zeros((50, 30, 20, 3)),
            concentration_profile=None,
            biofilm_thickness=real_np.ones((50, 30)),
        )
        ap.create_physics_visualizations(r)
        assert mock_st.columns.called

    def test_no_biofilm(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=True,
            flow_field=real_np.zeros((50, 30, 20, 3)),
            concentration_profile=real_np.ones((50, 30, 20)),
            biofilm_thickness=None,
        )
        ap.create_physics_visualizations(r)
        assert mock_st.columns.called

    def test_all_none(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        r = ap.PhysicsSimulationResult(
            success=True, flow_field=None,
            concentration_profile=None, biofilm_thickness=None,
        )
        ap.create_physics_visualizations(r)
        assert mock_st.columns.called


@pytest.mark.apptest
class TestRenderAdvancedPhysicsPage:
    """Test render_advanced_physics_page (lines 299-549)."""

    def _setup(self, mock_st, run_sim=False, btns=None):
        if btns is None:
            btns = [run_sim, False, False]
        mock_st.button.side_effect = btns
        mock_st.selectbox.return_value = "1 hour"
        mock_st.number_input.return_value = 1e-4

    def test_title(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.title.assert_called()

    def test_caption(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.caption.assert_called()

    def test_success_banner(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.success.assert_called()

    def test_parameters(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.expander.assert_called()
        mock_st.subheader.assert_called()

    def test_reynolds_laminar(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.info.assert_called()

    def test_reynolds_turbulent(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        widget = MagicMock()
        widget.render.side_effect = [
            1.0, 1e-9, 1.0, 1e-5, 0.4, 1e-10,
        ]
        ap.ScientificParameterWidget.return_value = widget
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.warning.assert_called()

    def test_sim_controls(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.button.assert_called()
        mock_st.selectbox.assert_called()

    def test_run_sim_success(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        mock_np.mean.return_value = 10.0
        mock_np.max.return_value = 0.001
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        self._setup(mock_st, run_sim=True)
        with patch("time.sleep"):
            ap.render_advanced_physics_page()
        mock_st.success.assert_called()
        mock_st.metric.assert_called()

    def test_run_sim_failure(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        pb = MagicMock()
        pb.progress.side_effect = RuntimeError("fail")
        mock_st.progress.return_value = pb
        self._setup(mock_st, run_sim=True)
        ap.render_advanced_physics_page()
        mock_st.error.assert_called()

    def test_export_buttons(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        mock_np.mean.return_value = 10.0
        mock_np.max.return_value = 0.001
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        mock_st.button.side_effect = [True, True, True]
        with patch("time.sleep"):
            ap.render_advanced_physics_page()
        assert mock_st.info.called

    def test_info_panel(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.markdown.assert_called()

    def test_no_run(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st, run_sim=False)
        ap.render_advanced_physics_page()
        mock_st.title.assert_called()

    def test_peclet(self):
        mock_st = _make_mock_st()
        ap = _import_advanced_physics(mock_st)
        self._setup(mock_st)
        ap.render_advanced_physics_page()
        mock_st.metric.assert_called()

    def test_status_ctx(self):
        import numpy as real_np
        mock_st = _make_mock_st()
        mock_np = MagicMock()
        mock_np.zeros.side_effect = lambda s: real_np.zeros(s)
        mock_np.linspace.side_effect = (
            lambda *a, **k: real_np.linspace(*a, **k)
        )
        mock_np.exp.side_effect = lambda x: real_np.exp(x)
        mock_np.mean.return_value = 10.0
        mock_np.max.return_value = 0.001
        ap = _import_advanced_physics(mock_st, mock_np=mock_np)
        status_ctx = MagicMock()
        status_ctx.__enter__ = MagicMock(return_value=status_ctx)
        status_ctx.__exit__ = MagicMock(return_value=False)
        mock_st.status.return_value = status_ctx
        self._setup(mock_st, run_sim=True)
        with patch("time.sleep"):
            ap.render_advanced_physics_page()
        status_ctx.update.assert_called()
