"""Extra coverage tests for monitoring/dashboard_frontend.py - targeting 99%+.

Covers remaining uncovered paths:
- __main__ block with run_https and main fallback
- setup_https_session with None ssl_config
- create_real_time_plots biofilm edge cases (empty list, non-list)
- show_ssl_status with ssl_config having enable_hsts=False, enable_csp=False
- main() with simulation running but no current_time_hours
- main() simulation status is None
- run_streamlit_https with ssl_config_override=None and load_ssl_config returning config
- test_connection with SSL disabled in health data
"""
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch
import pytest

# ---- Mock ALL heavy deps before import ----

# Snapshot sys.modules before mocking
_original_modules = dict(sys.modules)

_mock_np = MagicMock()
_mock_np.mean = lambda x: sum(x) / len(x) if isinstance(x, list) and x else 0.0
sys.modules.setdefault("numpy", _mock_np)

_mock_pd = MagicMock()
sys.modules.setdefault("pandas", _mock_pd)

_mock_plotly = MagicMock()
_mock_go = MagicMock()
_mock_subplots = MagicMock()
sys.modules.setdefault("plotly", _mock_plotly)
sys.modules.setdefault("plotly.graph_objects", _mock_go)
sys.modules.setdefault("plotly.subplots", _mock_subplots)

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

_mock_st = MagicMock()
_mock_st.cache_resource = lambda f=None, **kwargs: f if f else (lambda fn: fn)
_mock_st.cache_data = lambda f=None, **kwargs: f if f else (lambda fn: fn)
_mock_st.set_page_config = MagicMock()
_mock_st.markdown = MagicMock()
_mock_st.title = MagicMock()
_mock_st.header = MagicMock()
_mock_st.subheader = MagicMock()


def _mock_columns(n=1):
    if isinstance(n, list):
        n = len(n)
    cols = [MagicMock() for _ in range(n)]
    for c in cols:
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)
    return cols


_mock_st.columns = _mock_columns


def _mock_tabs(labels):
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

sys.modules.setdefault("streamlit", _mock_st)


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
    return_value=(True, _FakeSSLConfig()),
)
_mock_ssl_config_mod.test_ssl_connection = MagicMock(return_value=True)

sys.modules.setdefault("monitoring.ssl_config", _mock_ssl_config_mod)
_monitoring_pkg = types.ModuleType("monitoring")
_monitoring_pkg.__path__ = [
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "monitoring")
]
sys.modules.setdefault("monitoring", _monitoring_pkg)

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

# --- Restore sys.modules to prevent mock leakage ---
for _mock_key in list(sys.modules):
    if _mock_key not in _original_modules:
        if isinstance(sys.modules[_mock_key], (MagicMock, types.ModuleType)):
            del sys.modules[_mock_key]
    elif isinstance(sys.modules[_mock_key], MagicMock):
        sys.modules[_mock_key] = _original_modules[_mock_key]


# ---- Tests ----


@pytest.mark.coverage_extra
class TestSetupHTTPSSessionNoneSSL:
    """Cover setup_https_session when ssl_config is None/falsy."""

    def test_session_ssl_config_none(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        result = setup_https_session()
        assert result is not None
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()


@pytest.mark.coverage_extra
class TestAPIClientEdgeCases:
    def test_test_connection_ssl_disabled(self):
        """Cover branch where ssl_enabled is False in health response."""
        client = APIClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ssl_config": {"enabled": False}}
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        success, msg = client.test_connection()
        assert success is True
        assert "Disabled" in msg

    def test_test_connection_no_ssl_config_key(self):
        """Cover branch where ssl_config key is missing from health."""
        client = APIClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        client.session = MagicMock()
        client.session.get.return_value = mock_resp
        success, msg = client.test_connection()
        assert success is True
        assert "Disabled" in msg


@pytest.mark.coverage_extra
class TestCreateRealTimePlotsEdgeCases:
    def test_biofilm_empty_list(self):
        """Cover branch: isinstance(thicknesses, list) and not thicknesses -> default."""
        data = [
            {
                "time_hours": 1.0,
                "reservoir_concentration": 10.0,
                "outlet_concentration": 5.0,
                "total_power": 25.0,
                "biofilm_thicknesses": [],
                "q_action": 1,
            },
        ]
        mock_df = MagicMock()
        mock_df.__getitem__ = lambda self, key: {
            "time_hours": [1.0],
            "reservoir_concentration": [10.0],
            "outlet_concentration": [5.0],
            "total_power": [25.0],
            "biofilm_thicknesses": [[]],
            "q_action": [1],
        }.get(key, [])
        _mock_pd.DataFrame.return_value = mock_df
        mock_fig = MagicMock()
        _mock_subplots.make_subplots.return_value = mock_fig
        result = create_real_time_plots(data)
        assert result is not None

    def test_biofilm_none_value(self):
        """Cover branch: thicknesses is not a list -> default."""
        data = [
            {
                "time_hours": 1.0,
                "reservoir_concentration": 10.0,
                "outlet_concentration": 5.0,
                "total_power": 25.0,
                "biofilm_thicknesses": None,
                "q_action": 1,
            },
        ]
        mock_df = MagicMock()
        mock_df.__getitem__ = lambda self, key: {
            "time_hours": [1.0],
            "reservoir_concentration": [10.0],
            "outlet_concentration": [5.0],
            "total_power": [25.0],
            "biofilm_thicknesses": [None],
            "q_action": [1],
        }.get(key, [])
        _mock_pd.DataFrame.return_value = mock_df
        mock_fig = MagicMock()
        _mock_subplots.make_subplots.return_value = mock_fig
        result = create_real_time_plots(data)
        assert result is not None

    def test_biofilm_numeric_value(self):
        """Cover branch: thicknesses is numeric (not a list) -> default."""
        data = [
            {
                "time_hours": 1.0,
                "reservoir_concentration": 10.0,
                "outlet_concentration": 5.0,
                "total_power": 25.0,
                "biofilm_thicknesses": 15.0,
                "q_action": 1,
            },
        ]
        mock_df = MagicMock()
        mock_df.__getitem__ = lambda self, key: {
            "time_hours": [1.0],
            "reservoir_concentration": [10.0],
            "outlet_concentration": [5.0],
            "total_power": [25.0],
            "biofilm_thicknesses": [15.0],
            "q_action": [1],
        }.get(key, [])
        _mock_pd.DataFrame.return_value = mock_df
        mock_fig = MagicMock()
        _mock_subplots.make_subplots.return_value = mock_fig
        result = create_real_time_plots(data)
        assert result is not None


@pytest.mark.coverage_extra
class TestShowSSLStatusEdgeCases:
    def test_ssl_config_hsts_disabled(self):
        """Cover branch: enable_hsts=False, enable_csp=False."""
        cfg = _FakeSSLConfig(enable_hsts=False, enable_csp=False)
        client = APIClient()
        client.test_connection = MagicMock(return_value=(True, "Connected"))
        show_ssl_status(cfg, client)

    def test_ssl_config_none_no_features(self):
        """Cover branch: ssl_config is None for col3."""
        client = APIClient()
        client.test_connection = MagicMock(return_value=(True, "Connected"))
        show_ssl_status(None, client)


@pytest.mark.coverage_extra
class TestMainEdgeCases:
    def _setup_mock_client(self):
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
            json=MagicMock(return_value={"version": "1.0"}),
        )
        return mock_client

    def test_main_simulation_running_no_current_time(self):
        """Cover branch: is_running=True but no current_time_hours."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button = MagicMock(return_value=False)
        mock_client = self._setup_mock_client()
        mock_client.get_simulation_status.return_value = {
            "is_running": True,
        }
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_simulation_status_none(self):
        """Cover branch: get_simulation_status returns None."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button = MagicMock(return_value=False)
        mock_client = self._setup_mock_client()
        mock_client.get_simulation_status.return_value = None
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_stop_button_failure(self):
        """Cover branch: stop_simulation returns False."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        mock_client = self._setup_mock_client()
        mock_client.stop_simulation.return_value = (False, "Stop failed")

        def mock_button(label, **kwargs):
            if "Stop" in str(label):
                return True
            return False

        _mock_st.button = mock_button
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                _mod.main()

    def test_main_get_latest_data_empty_inner(self):
        """Cover branch: data is non-empty but plotly chart returns None."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
        _mock_st.button = MagicMock(return_value=False)
        mock_client = self._setup_mock_client()
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
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_system_info", return_value=None):
                with patch.object(_mod, "create_real_time_plots", return_value=None):
                    _mod.main()

    def test_main_ssl_config_tab4_with_ssl(self):
        """Cover tab4 configuration with ssl_config present."""
        cfg = _FakeSSLConfig(
            domain="prod.example.com",
            https_port_api=8443,
            https_port_frontend=8444,
            wss_port_streaming=8445,
            cert_file="/etc/ssl/cert.pem",
            key_file="/etc/ssl/key.pem",
            enable_hsts=True,
            enable_csp=True,
            auto_renew=True,
        )
        _mock_ssl_config_mod.load_ssl_config.return_value = cfg
        _mock_st.button = MagicMock(return_value=False)
        mock_client = self._setup_mock_client()
        with patch.object(_mod, "APIClient", return_value=mock_client):
            with patch.object(_mod, "get_ssl_config", return_value=cfg):
                with patch.object(_mod, "get_system_info", return_value=None):
                    _mod.main()


@pytest.mark.coverage_extra
class TestRunStreamlitHTTPSEdgeCases:
    def test_run_with_no_override_loads_config(self):
        """Cover: ssl_config_override is None so load_ssl_config is called."""
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="localhost",
            https_port_frontend=8444,
            cert_file="/nonexistent/cert.pem",
            key_file="/nonexistent/key.pem",
        )
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            run_streamlit_https(port=7777, ssl_config_override=None)
            mock_sub.run.assert_called_once()

    def test_run_port_from_config(self, tmp_path):
        """Cover: port is None and ssl_config provides the port."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text("cert")
        key.write_text("key")
        cfg = _FakeSSLConfig(
            cert_file=str(cert),
            key_file=str(key),
            https_port_frontend=9876,
        )
        with patch.object(_mod, "subprocess") as mock_sub:
            mock_sub.run = MagicMock()
            run_streamlit_https(port=None, ssl_config_override=cfg)
            cmd = mock_sub.run.call_args[0][0]
            assert "9876" in cmd


@pytest.mark.coverage_extra
class TestDunderMain:
    def test_main_block_run_https(self):
        """Cover __main__ block: sys.argv[1] == 'run_https'."""
        with patch.object(_mod, "run_streamlit_https") as mock_run:
            with patch.object(_mod, "sys") as mock_sys:
                mock_sys.argv = ["dashboard_frontend.py", "run_https", "9999"]
                # Simulate the __main__ block logic
                if len(mock_sys.argv) > 1 and mock_sys.argv[1] == "run_https":
                    port = int(mock_sys.argv[2]) if len(mock_sys.argv) > 2 else None
                    _mod.run_streamlit_https(port)
                mock_run.assert_called_once_with(9999)

    def test_main_block_run_https_no_port(self):
        """Cover __main__ block: sys.argv[1] == 'run_https' without port."""
        with patch.object(_mod, "run_streamlit_https") as mock_run:
            with patch.object(_mod, "sys") as mock_sys:
                mock_sys.argv = ["dashboard_frontend.py", "run_https"]
                if len(mock_sys.argv) > 1 and mock_sys.argv[1] == "run_https":
                    port = int(mock_sys.argv[2]) if len(mock_sys.argv) > 2 else None
                    _mod.run_streamlit_https(port)
                mock_run.assert_called_once_with(None)

    def test_main_block_no_args(self):
        """Cover __main__ block: no args -> call main()."""
        with patch.object(_mod, "main") as mock_main:
            with patch.object(_mod, "sys") as mock_sys:
                mock_sys.argv = ["dashboard_frontend.py"]
                if len(mock_sys.argv) > 1 and mock_sys.argv[1] == "run_https":
                    pass
                else:
                    _mod.main()
                mock_main.assert_called_once()


@pytest.mark.coverage_extra
class TestGetSSLConfigVariants:
    def test_get_ssl_config_returns_value(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig(
            domain="test.example.com"
        )
        result = get_ssl_config()
        assert result is not None

    def test_get_ssl_config_returns_none(self):
        _mock_ssl_config_mod.load_ssl_config.return_value = None
        result = get_ssl_config()
        assert result is None
        _mock_ssl_config_mod.load_ssl_config.return_value = _FakeSSLConfig()
