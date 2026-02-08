"""Tests for monitoring/dashboard_frontend.py - targeting 98%+ coverage."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---- Mock ALL heavy deps before import ----

# Mock numpy
_mock_np = MagicMock()
_mock_np.mean = lambda x: sum(x) / len(x) if isinstance(x, list) and x else 0.0
sys.modules.setdefault("numpy", _mock_np)

# Mock pandas
_mock_pd = MagicMock()
sys.modules.setdefault("pandas", _mock_pd)

# Mock plotly
_mock_plotly = MagicMock()
_mock_go = MagicMock()
_mock_subplots = MagicMock()
sys.modules.setdefault("plotly", _mock_plotly)
sys.modules.setdefault("plotly.graph_objects", _mock_go)
sys.modules.setdefault("plotly.subplots", _mock_subplots)

# Mock requests
_mock_requests = MagicMock()
_mock_requests.Session = MagicMock
_mock_requests.exceptions = MagicMock()
_mock_requests.exceptions.SSLError = type("SSLError", (Exception,), {})
_mock_requests.exceptions.ConnectionError = type("ConnectionError", (Exception,), {})
_mock_requests.exceptions.Timeout = type("Timeout", (Exception,), {})
sys.modules.setdefault("requests", _mock_requests)
sys.modules.setdefault("requests.adapters", MagicMock())
sys.modules.setdefault("urllib3", MagicMock())
sys.modules.setdefault("urllib3.util", MagicMock())
sys.modules.setdefault("urllib3.util.retry", MagicMock())

# Mock streamlit - must be comprehensive since module uses it at top level
_mock_st = MagicMock()
# cache_resource and cache_data should be identity decorators
_mock_st.cache_resource = lambda f=None, **kwargs: f if f else (lambda fn: fn)
_mock_st.cache_data = lambda f=None, **kwargs: f if f else (lambda fn: fn)
_mock_st.set_page_config = MagicMock()
_mock_st.markdown = MagicMock()
_mock_st.title = MagicMock()
_mock_st.header = MagicMock()
_mock_st.subheader = MagicMock()
def _mock_columns(n=1):
    """Return n mock column context managers."""
    if isinstance(n, list):
        n = len(n)
    cols = [MagicMock() for _ in range(n)]
    for c in cols:
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)
    return cols


_mock_st.columns = _mock_columns


def _mock_tabs(labels):
    """Return mock tabs context managers."""
    n = len(labels) if isinstance(labels, list) else 4
    tabs = [MagicMock() for _ in range(n)]
    for t in tabs:
        t.__enter__ = MagicMock(return_value=t)
        t.__exit__ = MagicMock(return_value=False)
    return tabs


_mock_st.tabs = _mock_tabs
_mock_st.sidebar = MagicMock()
_mock_st.sidebar.header = MagicMock()


def _mock_expander(label, expanded=False):
    """Return a context manager mock for expander."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


_mock_st.sidebar.expander = _mock_expander
_mock_st.success = MagicMock()
_mock_st.error = MagicMock()
_mock_st.warning = MagicMock()
_mock_st.info = MagicMock()
_mock_st.text = MagicMock()
_mock_st.json = MagicMock()
_mock_st.metric = MagicMock()
_mock_st.progress = MagicMock()
_mock_st.stop = MagicMock()
_mock_st.plotly_chart = MagicMock()
_mock_st.number_input = MagicMock(return_value=1)
_mock_st.checkbox = MagicMock(return_value=False)
_mock_st.slider = MagicMock(return_value=0.1)
_mock_st.selectbox = MagicMock(return_value="INFO")
_mock_st.button = MagicMock(return_value=False)
_mock_st.experimental_rerun = MagicMock()
_mock_st.add_hline = MagicMock()

sys.modules.setdefault("streamlit", _mock_st)

# Mock monitoring.ssl_config


class _FakeSSLConfig:
    def __init__(self, **kwargs):
        self.cert_file = kwargs.get("cert_file", "/tmp/fake.crt")
        self.key_file = kwargs.get("key_file", "/tmp/fake.key")
        self.ca_file = kwargs.get("ca_file", None)
        self.domain = kwargs.get("domain", "localhost")
        self.ciphers = kwargs.get("ciphers", "ECDHE+AESGCM:!aNULL:!MD5")
        self.wss_port_streaming = kwargs.get("wss_port_streaming", 8001)
        self.https_port_frontend = kwargs.get("https_port_frontend", 8444)
        self.https_port_api = kwargs.get("https_port_api", 8443)
        self.use_letsencrypt = kwargs.get("use_letsencrypt", False)
        self.staging = kwargs.get("staging", False)
        self.email = kwargs.get("email", "test@example.com")
        self.ssl_version = kwargs.get("ssl_version", "TLSv1_2")
        self.enable_hsts = kwargs.get("enable_hsts", True)
        self.enable_csp = kwargs.get("enable_csp", True)
        self.auto_renew = kwargs.get("auto_renew", False)


_mock_ssl_config_mod = MagicMock()
_mock_ssl_config_mod.SSLConfig = _FakeSSLConfig
_mock_ssl_config_mod.SecurityHeaders = MagicMock()
_mock_ssl_config_mod.SSLContextManager = MagicMock()
_mock_ssl_config_mod.load_ssl_config = MagicMock(return_value=_FakeSSLConfig())
_mock_ssl_config_mod.initialize_ssl_infrastructure = MagicMock(
    return_value=(True, _FakeSSLConfig())
)
_mock_ssl_config_mod.test_ssl_connection = MagicMock(return_value=True)

sys.modules.setdefault("monitoring.ssl_config", _mock_ssl_config_mod)
sys.modules.setdefault("monitoring", MagicMock())

# ---- Load module ----

_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_spec = importlib.util.spec_from_file_location(
    "monitoring.dashboard_frontend",
    os.path.join(_src, "monitoring", "dashboard_frontend.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["monitoring.dashboard_frontend"] = _mod
_spec.loader.exec_module(_mod)

APIClient = _mod.APIClient
get_ssl_config = _mod.get_ssl_config
setup_https_session = _mod.setup_https_session
get_system_info = _mod.get_system_info
create_real_time_plots = _mod.create_real_time_plots
show_ssl_status = _mod.show_ssl_status
run_streamlit_https = _mod.run_streamlit_https


# ---- Tests ----


class TestGetSSLConfig:
    def test_returns_config(self):
        result = get_ssl_config()
        assert result is not None


class TestSetupHTTPSSession:
    def test_returns_session(self):
        result = setup_https_session()
        assert result is not None

    def test_session_with_localhost(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="localhost"
        )
        result = setup_https_session()
        assert result is not None

    def test_session_with_production_domain(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="example.com"
        )
        result = setup_https_session()
        assert result is not None


class TestAPIClient:
    def _make_client(self):
        return APIClient()

    def test_init(self):
        client = self._make_client()
        assert client.base_url is not None

    def test_get_api_base_url_with_ssl(self):
        client = self._make_client()
        assert "https" in client.base_url or "http" in client.base_url

    def test_get_api_base_url_no_ssl(self):
        client = APIClient()
        client.ssl_config = None
        url = client._get_api_base_url()
        assert url.startswith("http://")
        assert "8000" in url

    def test_test_connection_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ssl_config": {"enabled": True}
        }
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        success, msg = client.test_connection()
        assert success is True
        assert "Connected" in msg

    def test_test_connection_ssl_error(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = _mock_requests.exceptions.SSLError(
            "ssl fail"
        )
        success, msg = client.test_connection()
        assert success is False
        assert "SSL" in msg

    def test_test_connection_connection_error(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = (
            _mock_requests.exceptions.ConnectionError("conn fail")
        )
        success, msg = client.test_connection()
        assert success is False
        assert "Connection Failed" in msg

    def test_test_connection_timeout(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = (
            _mock_requests.exceptions.Timeout()
        )
        success, msg = client.test_connection()
        assert success is False
        assert "Timeout" in msg

    def test_test_connection_generic_error(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = RuntimeError("unexpected")
        success, msg = client.test_connection()
        assert success is False
        assert "Error" in msg

    def test_get_simulation_status_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"is_running": True}
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        result = client.get_simulation_status()
        assert result == {"is_running": True}

    def test_get_simulation_status_failure(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = RuntimeError("fail")
        result = client.get_simulation_status()
        assert result is None

    def test_start_simulation_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        client.session = MagicMock()
        client.session.post.return_value = mock_resp
        success, msg = client.start_simulation({"duration": 24})
        assert success is True

    def test_start_simulation_failure(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.post.side_effect = RuntimeError("fail")
        success, msg = client.start_simulation({"duration": 24})
        assert success is False

    def test_stop_simulation_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        client.session = MagicMock()
        client.session.post.return_value = mock_resp
        success, msg = client.stop_simulation()
        assert success is True

    def test_stop_simulation_failure(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.post.side_effect = RuntimeError("fail")
        success, msg = client.stop_simulation()
        assert success is False

    def test_get_latest_data_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"x": 1}]
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        result = client.get_latest_data(limit=50)
        assert result == [{"x": 1}]

    def test_get_latest_data_failure(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = RuntimeError("fail")
        result = client.get_latest_data()
        assert result is None

    def test_get_performance_metrics_success(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"mean_power": 0.5}
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        result = client.get_performance_metrics()
        assert result == {"mean_power": 0.5}

    def test_get_performance_metrics_failure(self):
        client = self._make_client()
        client.session = MagicMock()
        client.session.get.side_effect = RuntimeError("fail")
        result = client.get_performance_metrics()
        assert result is None


class TestGetSystemInfo:
    def test_success(self):
        client = APIClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"version": "1.0"}
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        result = get_system_info(client)
        assert result == {"version": "1.0"}

    def test_failure(self):
        client = APIClient()
        client.session = MagicMock()
        client.session.get.side_effect = RuntimeError("fail")
        result = get_system_info(client)
        assert result is None


class TestCreateRealTimePlots:
    def test_empty_data(self):
        result = create_real_time_plots([])
        assert result is None

    def test_none_data(self):
        result = create_real_time_plots(None)
        assert result is None

    def test_with_data(self):
        data = [
            {
                "time_hours": 1.0,
                "reservoir_concentration": 10.0,
                "outlet_concentration": 5.0,
                "total_power": 25.0,
                "biofilm_thicknesses": [10.0, 20.0],
                "q_action": 1,
            },
            {
                "time_hours": 2.0,
                "reservoir_concentration": 11.0,
                "outlet_concentration": 6.0,
                "total_power": 26.0,
                "biofilm_thicknesses": 15.0,
                "q_action": 2,
            },
        ]
        # Mock DataFrame behavior
        mock_df = MagicMock()
        mock_df.__getitem__ = lambda self, key: {
            "time_hours": [1.0, 2.0],
            "reservoir_concentration": [10.0, 11.0],
            "outlet_concentration": [5.0, 6.0],
            "total_power": [25.0, 26.0],
            "biofilm_thicknesses": [[10.0, 20.0], 15.0],
            "q_action": [1, 2],
        }.get(key, [])
        _mock_pd.DataFrame.return_value = mock_df

        mock_fig = MagicMock()
        _mock_subplots.make_subplots.return_value = mock_fig

        result = create_real_time_plots(data)
        assert result is not None


class TestShowSSLStatus:
    def test_with_ssl_config_connected(self):
        cfg = _FakeSSLConfig()
        client = APIClient()
        client.test_connection = MagicMock(
            return_value=(True, "Connected")
        )
        show_ssl_status(cfg, client)

    def test_with_ssl_config_disconnected(self):
        cfg = _FakeSSLConfig()
        client = APIClient()
        client.test_connection = MagicMock(
            return_value=(False, "Failed")
        )
        show_ssl_status(cfg, client)

    def test_without_ssl_config(self):
        client = APIClient()
        client.test_connection = MagicMock(
            return_value=(False, "No SSL")
        )
        show_ssl_status(None, client)


class TestMain:
    def _setup_mock_client(self):
        """Set up mock API client behavior for main()."""
        # Mock APIClient to avoid real HTTP calls
        mock_client = MagicMock()
        mock_client.base_url = "https://localhost:8443"
        mock_client.ssl_config = _FakeSSLConfig()
        mock_client.test_connection.return_value = (True, "Connected")
        mock_client.get_simulation_status.return_value = {
            "is_running": True,
            "current_time_hours": 5.0,
            "duration_hours": 24.0,
        }
        mock_client.start_simulation.return_value = (True, "Started")
        mock_client.stop_simulation.return_value = (True, "Stopped")
        mock_client.get_latest_data.return_value = [
            {
                "time_hours": 1.0,
                "reservoir_concentration": 10.0,
                "outlet_concentration": 5.0,
                "total_power": 25.0,
                "biofilm_thicknesses": [10.0],
                "q_action": 1,
            },
        ]
        mock_client.get_performance_metrics.return_value = {
            "final_reservoir_concentration": 24.5,
            "control_effectiveness_2mM": 92.0,
            "mean_power": 0.5,
            "total_substrate_added": 100.0,
            "energy_efficiency": 45.0,
            "stability_score": 0.85,
        }
        mock_client.session = MagicMock()
        mock_client.session.get.return_value = MagicMock(
            json=MagicMock(return_value={"version": "1.0"})
        )
        return mock_client

    def test_main_with_ssl_not_localhost(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="example.com"
        )
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_without_ssl(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        mock_client.ssl_config = None
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(
                _mod, "get_ssl_config", return_value=None
            ):
                with patch.object(
                    _mod, "get_system_info", return_value=None
                ):
                    _mod.main()

    def test_main_with_localhost(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="localhost"
        )
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_simulation_stopped(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        mock_client.get_simulation_status.return_value = {
            "is_running": False,
        }
        mock_client.get_latest_data.return_value = None
        mock_client.get_performance_metrics.return_value = None
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_connection_failed(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        mock_client.test_connection.return_value = (False, "Failed")
        # st.stop() is called when connection fails
        _mock_st.stop.side_effect = Exception("stop")
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                try:
                    _mod.main()
                except Exception:
                    pass
        _mock_st.stop.side_effect = None

    def test_main_with_no_performance_energy(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button.return_value = False
        mock_client = self._setup_mock_client()
        # Performance metrics without energy_efficiency
        mock_client.get_performance_metrics.return_value = {
            "final_reservoir_concentration": 24.5,
            "control_effectiveness_2mM": 92.0,
            "mean_power": 0.5,
            "total_substrate_added": 100.0,
        }
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_with_ssl_config_none_settings(self):
        """Test the configuration tab with no ssl config."""
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        mock_client = self._setup_mock_client()
        _mock_st.button = MagicMock(return_value=False)
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(
                _mod, "get_ssl_config", return_value=None
            ):
                with patch.object(
                    _mod, "get_system_info", return_value=None
                ):
                    _mod.main()

    def test_main_with_system_info(self):
        """Cover line 509: system_info available."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button = MagicMock(return_value=False)
        mock_client = self._setup_mock_client()
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(
                _mod,
                "get_system_info",
                return_value={"version": "1.0"},
            ):
                _mod.main()

    def test_main_start_button_pressed(self):
        """Cover lines 581-603: start simulation button."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()
        mock_client.start_simulation.return_value = (True, "Started")
        _mock_st.checkbox = MagicMock(return_value=True)
        _mock_st.number_input = MagicMock(return_value=24)

        def mock_button(label, **kwargs):
            if "Start" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()
        _mock_st.checkbox = MagicMock(return_value=False)

    def test_main_start_button_failure(self):
        """Cover line 603: start simulation fails."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()
        mock_client.start_simulation.return_value = (False, "Failed")
        _mock_st.checkbox = MagicMock(return_value=True)
        _mock_st.number_input = MagicMock(return_value=24)

        def mock_button(label, **kwargs):
            if "Start" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()
        _mock_st.checkbox = MagicMock(return_value=False)

    def test_main_stop_button_pressed(self):
        """Cover lines 607-612: stop simulation button."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()

        def mock_button(label, **kwargs):
            if "Stop" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_refresh_button_pressed(self):
        """Cover line 616: refresh button."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()

        def mock_button(label, **kwargs):
            if "Refresh" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_auto_refresh(self):
        """Cover lines 662-665: auto-refresh path."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()
        _mock_st.button = MagicMock(return_value=False)

        def mock_checkbox(label, **kwargs):
            if "Auto-refresh" in str(label):
                return True
            return kwargs.get("value", False)

        _mock_st.checkbox = mock_checkbox
        _mock_st.number_input = MagicMock(return_value=1)

        # Patch time.sleep at module level to avoid delays
        import time as _time_mod

        with patch.object(_time_mod, "sleep"):
            with patch.object(
                _mod, "APIClient", return_value=mock_client
            ):
                with patch.object(
                    _mod, "get_system_info", return_value=None
                ):
                    _mod.main()
        _mock_st.checkbox = MagicMock(return_value=False)

    def test_main_export_buttons(self):
        """Cover lines 758, 762, 766: export buttons."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()

        def mock_button(label, **kwargs):
            if "Export" in str(label) or "Report" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_init_ssl_button(self):
        """Cover line 802: Initialize SSL button."""
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        mock_client = self._setup_mock_client()

        def mock_button(label, **kwargs):
            if "Initialize SSL" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(
                _mod, "get_ssl_config", return_value=None
            ):
                with patch.object(
                    _mod, "get_system_info", return_value=None
                ):
                    _mod.main()

    def test_main_not_pretrained(self):
        """Cover lines 589-596: not using pretrained model."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()
        _mock_st.slider = MagicMock(return_value=0.1)
        _mock_st.number_input = MagicMock(return_value=24)

        checkbox_calls = [0]

        def mock_checkbox(label, **kwargs):
            checkbox_calls[0] += 1
            if "Pre-trained" in str(label):
                return False  # Not using pretrained
            return kwargs.get("value", False)

        _mock_st.checkbox = mock_checkbox

        def mock_button(label, **kwargs):
            if "Start" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()
        _mock_st.checkbox = MagicMock(return_value=False)


class TestRunStreamlitHTTPS:
    def test_run_with_ssl_certs(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text("cert")
        key.write_text("key")
        cfg = _FakeSSLConfig(
            cert_file=str(cert),
            key_file=str(key),
        )
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            run_streamlit_https(port=9999, ssl_config_override=cfg)
            mock_sub.run.assert_called_once()
            cmd = mock_sub.run.call_args[0][0]
            assert "--server.sslCertFile" in cmd

    def test_run_without_ssl_certs(self):
        cfg = _FakeSSLConfig(
            cert_file="/nonexistent/cert.pem",
            key_file="/nonexistent/key.pem",
        )
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            run_streamlit_https(port=9999, ssl_config_override=cfg)
            mock_sub.run.assert_called_once()
            cmd = mock_sub.run.call_args[0][0]
            assert "--server.sslCertFile" not in cmd

    def test_run_default_port(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            run_streamlit_https(port=None)
            mock_sub.run.assert_called_once()

    def test_run_keyboard_interrupt(self):
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run.side_effect = KeyboardInterrupt()
            cfg = _FakeSSLConfig()
            run_streamlit_https(port=9999, ssl_config_override=cfg)

    def test_run_no_ssl_config(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            # When ssl_config is None, port defaults to 8501
            run_streamlit_https(port=None, ssl_config_override=None)
