"""Unit tests for GUI standalone modules with 0% coverage.

Targets (all at 0% coverage):
- gui/alert_configuration_ui.py (299 stmts)
- gui/browser_download_manager.py (222 stmts)
- gui/electrode_configuration_ui.py (202 stmts)
- gui/membrane_configuration_ui.py (185 stmts)
- gui/simulation_runner.py (270 stmts)

Uses unittest.mock to avoid requiring a live Streamlit runtime.
"""

from __future__ import annotations

import io
import json
import queue
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_SRC_DIR = str(
    (Path(__file__).resolve().parent / ".." / ".." / "src").resolve(),
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Modules that need fresh import when mocking streamlit
_GUI_MODULES_TO_CLEAR = [
    "gui.alert_configuration_ui",
    "gui.browser_download_manager",
    "gui.electrode_configuration_ui",
    "gui.membrane_configuration_ui",
    "gui.simulation_runner",
]


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Remove cached GUI modules so each test gets a fresh import."""
    for mod_name in list(sys.modules):
        if any(
            mod_name == m or mod_name.startswith(m + ".")
            for m in _GUI_MODULES_TO_CLEAR
        ):
            del sys.modules[mod_name]
    yield
    for mod_name in list(sys.modules):
        if any(
            mod_name == m or mod_name.startswith(m + ".")
            for m in _GUI_MODULES_TO_CLEAR
        ):
            del sys.modules[mod_name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_xlsxwriter():
    """Check if xlsxwriter is available."""
    try:
        import xlsxwriter  # noqa: F401
        return True
    except ImportError:
        return False


class _SessionState(dict):
    """Dict subclass that also supports attribute access (like Streamlit)."""

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


def _make_ctx(*_a, **_kw):
    """Return a context manager mock."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _smart_columns(n_or_spec=2, **_kw):
    """Return n context managers, handling both int and list specs."""
    if isinstance(n_or_spec, (list, tuple)):
        n = len(n_or_spec)
    else:
        n = int(n_or_spec)
    return [_make_ctx() for _ in range(n)]


def _smart_tabs(labels):
    """Return context managers matching the number of tab labels."""
    return [_make_ctx() for _ in range(len(labels))]


def _make_mock_st():
    """Create a MagicMock mimicking streamlit with session_state."""
    mock_st = MagicMock()
    mock_st.session_state = _SessionState()
    # Make tabs and columns return proper number of context managers
    mock_st.tabs.side_effect = _smart_tabs
    mock_st.columns.side_effect = _smart_columns
    mock_st.expander.return_value = _make_ctx()
    # checkbox, button etc. return False by default
    mock_st.button.return_value = False
    mock_st.checkbox.return_value = False
    mock_st.selectbox.return_value = "power_density"
    mock_st.number_input.return_value = 1.0
    mock_st.slider.return_value = 0.5
    mock_st.text_input.return_value = "test"
    mock_st.text_area.return_value = ""
    mock_st.file_uploader.return_value = None
    return mock_st


# ===================================================================
# PART 1: alert_configuration_ui.py (299 stmts)
# ===================================================================


@pytest.mark.apptest
class TestAlertConfigurationUIInit:
    """Tests for AlertConfigurationUI constructor and parameter_info."""

    def test_init_stores_alert_manager(self):
        """AlertConfigurationUI stores the alert_manager."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            ui = AlertConfigurationUI(manager)
            assert ui.alert_manager is manager

    def test_parameter_info_keys(self):
        """parameter_info contains expected MFC parameters."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            ui = AlertConfigurationUI(MagicMock())
            expected_params = {
                "power_density",
                "substrate_concentration",
                "pH",
                "temperature",
                "biofilm_thickness",
                "conductivity",
                "dissolved_oxygen",
                "coulombic_efficiency",
            }
            assert set(ui.parameter_info.keys()) == expected_params

    def test_parameter_info_has_required_fields(self):
        """Each parameter has name, unit, typical_range, critical_range."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            ui = AlertConfigurationUI(MagicMock())
            for key, info in ui.parameter_info.items():
                assert "name" in info, f"{key} missing 'name'"
                assert "unit" in info, f"{key} missing 'unit'"
                assert "typical_range" in info, f"{key} missing 'typical_range'"
                assert "critical_range" in info, f"{key} missing 'critical_range'"
                assert "description" in info, f"{key} missing 'description'"


@pytest.mark.apptest
class TestAlertConfigurationUIRender:
    """Tests for AlertConfigurationUI.render and sub-methods."""

    def test_render_calls_st_title_and_tabs(self):
        """render() calls st.title and st.tabs."""
        mock_st = _make_mock_st()
        tabs = [MagicMock() for _ in range(5)]
        mock_st.tabs.return_value = tabs
        for t in tabs:
            t.__enter__ = MagicMock(return_value=t)
            t.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.thresholds = {}
            manager.get_active_alerts.return_value = []
            manager.get_alert_history.return_value = []
            manager.escalation_rules = []
            manager.admin_emails = []
            manager.user_emails = []
            ui = AlertConfigurationUI(manager)
            # Mock _visualize_thresholds to avoid plotly state issues
            ui._visualize_thresholds = MagicMock()
            ui.render()

            mock_st.title.assert_called_once()
            mock_st.tabs.assert_called_once()

    def test_render_threshold_settings_selectbox(self):
        """_render_threshold_settings renders a selectbox for parameters."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "power_density"
        mock_st.number_input.return_value = 1.0
        mock_st.checkbox.return_value = True

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.thresholds = {}
            ui = AlertConfigurationUI(manager)
            # Mock _visualize_thresholds to avoid plotly state issues
            ui._visualize_thresholds = MagicMock()
            ui._render_threshold_settings()

            # selectbox is called for parameter selection
            mock_st.selectbox.assert_called()

    def test_render_alert_dashboard_no_alerts(self):
        """_render_alert_dashboard with no active alerts shows success."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.get_active_alerts.return_value = []
            ui = AlertConfigurationUI(manager)
            ui._render_alert_dashboard()

            mock_st.success.assert_called()

    def test_render_alert_dashboard_with_alerts(self):
        """_render_alert_dashboard renders alert data when alerts exist."""
        mock_st = _make_mock_st()
        mock_st.columns.return_value = [_make_ctx() for _ in range(4)]

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI
            from monitoring.alert_management import Alert

            alert = Alert(
                id=1,
                parameter="pH",
                value=5.0,
                severity="critical",
                message="pH too low",
                threshold_violated="Below critical minimum",
                escalated=False,
            )
            manager = MagicMock()
            manager.get_active_alerts.return_value = [alert]
            ui = AlertConfigurationUI(manager)
            ui._render_alert_dashboard()

            mock_st.metric.assert_called()

    def test_render_alert_history_empty(self):
        """_render_alert_history with no alerts shows metrics only."""
        mock_st = _make_mock_st()
        mock_st.selectbox.side_effect = [24, "All", "All"]
        mock_st.columns.return_value = [_make_ctx() for _ in range(4)]

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.get_alert_history.return_value = []
            ui = AlertConfigurationUI(manager)
            ui._render_alert_history()

            mock_st.header.assert_called()

    def test_render_notification_settings(self):
        """_render_notification_settings renders email and preferences."""
        mock_st = _make_mock_st()
        mock_st.text_area.return_value = ""
        mock_st.columns.return_value = [_make_ctx() for _ in range(3)]

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.admin_emails = []
            manager.user_emails = []
            ui = AlertConfigurationUI(manager)
            ui._render_notification_settings()

            mock_st.header.assert_called()
            mock_st.text_area.assert_called()

    def test_render_escalation_rules_empty(self):
        """_render_escalation_rules with no rules shows add form."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "warning"
        mock_st.number_input.return_value = 30

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI

            manager = MagicMock()
            manager.escalation_rules = []
            ui = AlertConfigurationUI(manager)
            ui._render_escalation_rules()

            mock_st.header.assert_called()


@pytest.mark.apptest
class TestAlertConfigurationUIVisualize:
    """Tests for AlertConfigurationUI threshold visualization."""

    def test_visualize_thresholds(self):
        """_visualize_thresholds creates plotly figure and calls st.plotly_chart."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):

            from gui.alert_configuration_ui import AlertConfigurationUI

            ui = AlertConfigurationUI(MagicMock())
            param_info = {
                "name": "pH",
                "unit": "",
                "typical_range": (6.5, 7.5),
                "critical_range": (5.5, 8.5),
                "description": "test",
            }
            # Mock go.Figure to avoid accumulated plotly state
            with patch("gui.alert_configuration_ui.go") as mock_go:
                mock_fig = MagicMock()
                mock_go.Figure.return_value = mock_fig
                ui._visualize_thresholds("pH", param_info, 6.5, 7.5, 5.5, 8.5)
                mock_st.plotly_chart.assert_called_once()
                mock_fig.add_shape.assert_called()
                mock_fig.add_vline.assert_called()

    def test_render_alert_timeline(self):
        """_render_alert_timeline creates plotly chart for alerts."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI
            from monitoring.alert_management import Alert

            alerts = [
                Alert(
                    id=1,
                    parameter="pH",
                    value=5.0,
                    severity="critical",
                    message="pH too low",
                    threshold_violated="test",
                    escalated=True,
                ),
                Alert(
                    id=2,
                    parameter="temperature",
                    value=45.0,
                    severity="warning",
                    message="temp high",
                    threshold_violated="test",
                    escalated=False,
                ),
            ]
            ui = AlertConfigurationUI(MagicMock())
            ui._render_alert_timeline(alerts)
            mock_st.plotly_chart.assert_called_once()


@pytest.mark.apptest
class TestAlertConfigurationStandaloneFunctions:
    """Tests for standalone functions in alert_configuration_ui."""

    def test_create_alert_rule(self):
        """create_alert_rule returns dict with required keys."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import create_alert_rule

            rule = create_alert_rule("pH", 7.0)
            assert rule["parameter"] == "pH"
            assert rule["threshold"] == 7.0
            assert rule["condition"] == "greater_than"
            assert "created_at" in rule

    def test_create_alert_rule_custom_condition(self):
        """create_alert_rule supports custom conditions."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import create_alert_rule

            rule = create_alert_rule("temperature", 35.0, "less_than")
            assert rule["condition"] == "less_than"

    def test_check_alerts_greater_than(self):
        """check_alerts triggers on greater_than condition."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import check_alerts

            rules = [{"parameter": "pH", "threshold": 7.0, "condition": "greater_than"}]
            result = check_alerts({"pH": 8.0}, rules)
            assert len(result) == 1
            assert result[0]["current_value"] == 8.0

    def test_check_alerts_less_than(self):
        """check_alerts triggers on less_than condition."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import check_alerts

            rules = [{"parameter": "pH", "threshold": 7.0, "condition": "less_than"}]
            result = check_alerts({"pH": 5.0}, rules)
            assert len(result) == 1

    def test_check_alerts_equals(self):
        """check_alerts triggers on equals condition."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import check_alerts

            rules = [{"parameter": "pH", "threshold": 7.0, "condition": "equals"}]
            result = check_alerts({"pH": 7.0}, rules)
            assert len(result) == 1

    def test_check_alerts_no_trigger(self):
        """check_alerts returns empty when value does not trigger."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import check_alerts

            rules = [{"parameter": "pH", "threshold": 7.0, "condition": "greater_than"}]
            result = check_alerts({"pH": 5.0}, rules)
            assert len(result) == 0

    def test_check_alerts_missing_param(self):
        """check_alerts skips params not in current_values."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import check_alerts

            rules = [{"parameter": "pH", "threshold": 7.0, "condition": "greater_than"}]
            result = check_alerts({"temperature": 25.0}, rules)
            assert len(result) == 0

    def test_render_alert_configuration_with_none_manager(self):
        """render_alert_configuration creates default manager when None."""
        mock_st = _make_mock_st()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch(
                "gui.alert_configuration_ui.AlertManager",
            ) as mock_am_cls:
                mock_am_cls.return_value = MagicMock(
                    thresholds={},
                    get_active_alerts=MagicMock(return_value=[]),
                    get_alert_history=MagicMock(return_value=[]),
                    escalation_rules=[],
                    admin_emails=[],
                    user_emails=[],
                )
                from gui.alert_configuration_ui import (
                    AlertConfigurationUI,
                    render_alert_configuration,
                )

                # Mock _visualize_thresholds to avoid plotly state issues
                with patch.object(
                    AlertConfigurationUI, "_visualize_thresholds",
                ):
                    render_alert_configuration(None)
                    mock_am_cls.assert_called_once()

    def test_render_alert_configuration_with_manager(self):
        """render_alert_configuration uses provided manager."""
        mock_st = _make_mock_st()

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import (
                AlertConfigurationUI,
                render_alert_configuration,
            )

            manager = MagicMock()
            manager.thresholds = {}
            manager.get_active_alerts.return_value = []
            manager.get_alert_history.return_value = []
            manager.escalation_rules = []
            manager.admin_emails = []
            manager.user_emails = []
            # Mock _visualize_thresholds to avoid plotly state issues
            with patch.object(AlertConfigurationUI, "_visualize_thresholds"):
                render_alert_configuration(manager)
                mock_st.title.assert_called()


@pytest.mark.apptest
class TestAlertFrequencyChart:
    """Tests for _render_alert_frequency_chart."""

    def test_frequency_chart_renders(self):
        """_render_alert_frequency_chart processes alert list."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.alert_configuration_ui import AlertConfigurationUI
            from monitoring.alert_management import Alert

            alerts = [
                Alert(
                    id=i,
                    parameter="pH",
                    value=5.0 + i * 0.1,
                    severity="warning" if i % 2 == 0 else "critical",
                    message=f"alert {i}",
                    threshold_violated="test",
                )
                for i in range(5)
            ]
            ui = AlertConfigurationUI(MagicMock())
            ui._render_alert_frequency_chart(alerts)
            mock_st.plotly_chart.assert_called_once()


# ===================================================================
# PART 2: browser_download_manager.py (222 stmts)
# ===================================================================


@pytest.mark.apptest
class TestBrowserDownloadManagerInit:
    """Tests for BrowserDownloadManager constructor."""

    def test_init_sets_supported_formats(self):
        """Constructor sets supported_formats list."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            assert "CSV" in mgr.supported_formats
            assert "JSON" in mgr.supported_formats
            assert "ZIP" in mgr.supported_formats

    def test_init_sets_format_handlers(self):
        """Constructor populates format_handlers dict."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            assert "csv" in mgr.format_handlers
            assert "json" in mgr.format_handlers
            assert "xlsx" in mgr.format_handlers
            assert "zip" in mgr.format_handlers


@pytest.mark.apptest
class TestBrowserDownloadManagerCSV:
    """Tests for CSV conversion."""

    def test_to_csv_single_dataframe(self):
        """_to_csv with single DataFrame returns CSV bytes."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            result, mime, ext = mgr._to_csv({"test": df}, False)
            assert ext == "csv"
            assert mime == "text/csv"
            assert b"a,b" in result

    def test_to_csv_multiple_datasets(self):
        """_to_csv with multiple datasets returns ZIP."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            data = {
                "df1": pd.DataFrame({"x": [1]}),
                "df2": pd.DataFrame({"y": [2]}),
            }
            result, mime, ext = mgr._to_csv(data, True)
            assert ext == "zip"
            assert mime == "application/zip"

            # Verify it is a valid ZIP
            zf = zipfile.ZipFile(io.BytesIO(result))
            names = zf.namelist()
            assert "df1.csv" in names
            assert "df2.csv" in names
            assert "metadata.json" in names

    def test_to_csv_non_dataframe(self):
        """_to_csv handles non-DataFrame data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            data = {"list_data": [1, 2, 3]}
            result, mime, ext = mgr._to_csv(data, False)
            assert result is not None


@pytest.mark.apptest
class TestBrowserDownloadManagerJSON:
    """Tests for JSON conversion."""

    def test_to_json_dataframe(self):
        """_to_json converts DataFrame to JSON records."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2]})
            result, mime, ext = mgr._to_json({"data": df}, False)
            assert ext == "json"
            parsed = json.loads(result.decode("utf-8"))
            assert "data" in parsed

    def test_to_json_numpy_array(self):
        """_to_json converts numpy array to list."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            arr = np.array([1.0, 2.0, 3.0])
            result, mime, ext = mgr._to_json({"arr": arr}, False)
            parsed = json.loads(result.decode("utf-8"))
            assert parsed["arr"] == [1.0, 2.0, 3.0]

    def test_to_json_with_metadata(self):
        """_to_json includes metadata when requested."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            result, _, _ = mgr._to_json({"data": {"key": "value"}}, True)
            parsed = json.loads(result.decode("utf-8"))
            assert "_metadata" in parsed

    def test_to_json_plain_dict(self):
        """_to_json passes plain dict through."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            result, _, _ = mgr._to_json({"info": {"k": "v"}}, False)
            parsed = json.loads(result.decode("utf-8"))
            assert parsed["info"]["k"] == "v"


@pytest.mark.apptest
class TestBrowserDownloadManagerZIP:
    """Tests for ZIP archive creation."""

    def test_to_zip_archive(self):
        """_to_zip_archive creates ZIP with csv and json directories."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2]})
            result, mime, ext = mgr._to_zip_archive({"test": df}, True)
            assert ext == "zip"

            zf = zipfile.ZipFile(io.BytesIO(result))
            names = zf.namelist()
            assert "csv/test.csv" in names
            assert "json/test.json" in names
            assert "metadata.json" in names
            assert "README.txt" in names

    def test_compress_data(self):
        """_compress_data wraps bytes in a ZIP."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            result = mgr._compress_data(b"hello world", "test.txt")
            zf = zipfile.ZipFile(io.BytesIO(result))
            assert "test.txt" in zf.namelist()
            assert zf.read("test.txt") == b"hello world"


@pytest.mark.apptest
class TestBrowserDownloadManagerMetadata:
    """Tests for metadata and readme generation."""

    def test_generate_metadata_dataframe(self):
        """_generate_metadata handles DataFrame entries."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            meta = mgr._generate_metadata({"df": df})
            assert meta["datasets"]["df"]["type"] == "DataFrame"
            assert meta["datasets"]["df"]["shape"] == (2, 2)
            assert "x" in meta["datasets"]["df"]["columns"]
            assert meta["platform"] == "MFC Q-Learning Research Platform"

    def test_generate_metadata_non_dataframe(self):
        """_generate_metadata handles non-DataFrame entries."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            meta = mgr._generate_metadata({"list_data": [1, 2, 3]})
            assert meta["datasets"]["list_data"]["type"] == "list"
            assert meta["datasets"]["list_data"]["size"] == 3

    def test_generate_readme(self):
        """_generate_readme returns non-empty string."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            readme = mgr._generate_readme({"data": pd.DataFrame()})
            assert len(readme) > 0


@pytest.mark.apptest
class TestBrowserDownloadManagerPrepare:
    """Tests for _prepare_download."""

    def test_prepare_download_csv(self):
        """_prepare_download dispatches to _to_csv."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            data = {"df": pd.DataFrame({"a": [1]})}
            result, mime, ext = mgr._prepare_download(data, "csv", False)
            assert result is not None

    def test_prepare_download_unknown_format(self):
        """_prepare_download returns None for unknown format."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            result, mime, ext = mgr._prepare_download({}, "unknown", False)
            assert result is None
            assert mime == ""
            assert ext == ""


@pytest.mark.apptest
class TestBrowserDownloadManagerRender:
    """Tests for render_download_interface and related methods."""

    def test_render_download_interface_empty_data(self):
        """render_download_interface with empty dict shows info."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            mgr.render_download_interface({})
            mock_st.info.assert_called()

    def test_render_data_preview_dataframe(self):
        """_render_data_preview handles DataFrame data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2]})
            mgr._render_data_preview({"df": df})
            mock_st.dataframe.assert_called()

    def test_render_data_preview_dict(self):
        """_render_data_preview handles dict data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            mgr._render_data_preview({"d": {"key": "value"}})
            mock_st.json.assert_called()

    def test_render_data_preview_list(self):
        """_render_data_preview handles list data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            mgr._render_data_preview({"lst": [1, 2, 3]})
            mock_st.write.assert_called()

    def test_render_data_preview_string(self):
        """_render_data_preview handles string data."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            mgr._render_data_preview({"s": "hello"})
            mock_st.write.assert_called()

    def test_render_download_button(self):
        """_render_download_button calls st.download_button."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            data = {"df": pd.DataFrame({"a": [1]})}
            mgr._render_download_button(data, "csv", "test", True, False)
            mock_st.download_button.assert_called()

    def test_render_download_button_none_data(self):
        """_render_download_button shows error when data prep fails."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            mgr._render_download_button({}, "unknown_format", "test", False, False)
            mock_st.error.assert_called()

    def test_quick_download(self):
        """_quick_download calls st.download_button when data is valid."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            data = {"df": pd.DataFrame({"a": [1]})}
            mgr._quick_download(data, "csv", "sim")
            mock_st.download_button.assert_called()


@pytest.mark.apptest
class TestBrowserDownloadStandaloneFunction:
    """Tests for render_browser_downloads standalone function."""

    def test_render_browser_downloads_no_data(self):
        """render_browser_downloads with no data shows info."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import render_browser_downloads

            render_browser_downloads()
            mock_st.info.assert_called()

    def test_render_browser_downloads_with_sim_data(self):
        """render_browser_downloads with simulation data."""
        mock_st = _make_mock_st()
        mock_st.checkbox.return_value = True
        mock_st.selectbox.return_value = "csv"
        mock_st.text_input.return_value = "test_file"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import render_browser_downloads

            sim_data = {"results": pd.DataFrame({"a": [1, 2]})}
            render_browser_downloads(simulation_data=sim_data)
            mock_st.markdown.assert_called()

    def test_render_browser_downloads_with_all_data(self):
        """render_browser_downloads with all data sources."""
        mock_st = _make_mock_st()
        mock_st.checkbox.return_value = True
        mock_st.selectbox.return_value = "csv"
        mock_st.text_input.return_value = "test_file"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import render_browser_downloads

            render_browser_downloads(
                simulation_data={"sim": pd.DataFrame({"a": [1]})},
                q_learning_data={"ql": pd.DataFrame({"b": [2]})},
                analysis_results={"res": pd.DataFrame({"c": [3]})},
            )
            mock_st.markdown.assert_called()


@pytest.mark.apptest
class TestBrowserDownloadManagerExcel:
    """Tests for Excel conversion."""

    @pytest.mark.skipif(
        not _has_xlsxwriter(),
        reason="xlsxwriter not installed",
    )
    def test_to_excel_single_dataframe(self):
        """_to_excel with single DataFrame returns XLSX bytes."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            result, mime, ext = mgr._to_excel({"test": df}, False)
            assert ext == "xlsx"
            assert result is not None
            assert len(result) > 0

    @pytest.mark.skipif(
        not _has_xlsxwriter(),
        reason="xlsxwriter not installed",
    )
    def test_to_excel_with_metadata(self):
        """_to_excel includes metadata sheet when requested."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1]})
            result, _, _ = mgr._to_excel({"test": df}, True)
            assert result is not None


@pytest.mark.apptest
class TestBrowserDownloadManagerParquet:
    """Tests for Parquet conversion."""

    def test_to_parquet_dataframe(self):
        """_to_parquet with DataFrame returns ZIP containing parquet."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.browser_download_manager import BrowserDownloadManager

            mgr = BrowserDownloadManager()
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            result, mime, ext = mgr._to_parquet({"test": df}, True)
            if result is not None:
                assert ext == "zip"
                zf = zipfile.ZipFile(io.BytesIO(result))
                assert "test.parquet" in zf.namelist()


# ===================================================================
# PART 3: electrode_configuration_ui.py (202 stmts)
# ===================================================================


@pytest.mark.apptest
class TestElectrodeConfigurationUIInit:
    """Tests for ElectrodeConfigurationUI constructor."""

    def test_init_calls_initialize_session_state(self):
        """Constructor calls initialize_session_state."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            _ui = ElectrodeConfigurationUI()
            assert "anode_config" in mock_st.session_state
            assert "cathode_config" in mock_st.session_state
            assert "electrode_calculations" in mock_st.session_state

    def test_initialize_session_state_idempotent(self):
        """initialize_session_state does not overwrite existing values."""
        mock_st = _make_mock_st()
        mock_st.session_state["anode_config"] = "existing"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            _ui = ElectrodeConfigurationUI()
            assert mock_st.session_state["anode_config"] == "existing"


@pytest.mark.apptest
class TestElectrodeConfigurationUIMaterial:
    """Tests for material selection methods."""

    def test_render_material_selector_returns_material(self):
        """render_material_selector returns an ElectrodeMaterial."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "Graphite Plate"
        mock_st.columns.return_value = [_make_ctx() for _ in range(2)]
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeMaterial
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            material = ui.render_material_selector("anode")
            assert material == ElectrodeMaterial.GRAPHITE_PLATE

    def test_render_material_selector_custom(self):
        """render_material_selector returns CUSTOM for custom choice."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "Custom Material"
        mock_st.columns.return_value = [_make_ctx() for _ in range(2)]
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeMaterial
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            material = ui.render_material_selector("anode")
            assert material == ElectrodeMaterial.CUSTOM

    def test_display_material_properties(self):
        """_display_material_properties shows metrics."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import (
                MATERIAL_PROPERTIES_DATABASE,
                ElectrodeMaterial,
            )
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            props = MATERIAL_PROPERTIES_DATABASE[ElectrodeMaterial.GRAPHITE_PLATE]
            ui._display_material_properties(props, "Graphite Plate")
            mock_st.metric.assert_called()


@pytest.mark.apptest
class TestElectrodeConfigurationUIGeometry:
    """Tests for geometry selection methods."""

    def test_render_geometry_selector_rectangular(self):
        """render_geometry_selector returns RECTANGULAR_PLATE."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "Rectangular Plate"
        mock_st.number_input.return_value = 5.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            geom, dims = ui.render_geometry_selector("anode")
            assert geom == ElectrodeGeometry.RECTANGULAR_PLATE

    def test_render_dimension_inputs_rectangular(self):
        """_render_dimension_inputs for rectangular returns l, w, t."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 5.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            dims = ui._render_dimension_inputs(
                ElectrodeGeometry.RECTANGULAR_PLATE, "anode",
            )
            assert "length" in dims
            assert "width" in dims
            assert "thickness" in dims

    def test_render_dimension_inputs_cylindrical_rod(self):
        """_render_dimension_inputs for cylindrical rod returns diameter, length."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 10.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            dims = ui._render_dimension_inputs(
                ElectrodeGeometry.CYLINDRICAL_ROD, "anode",
            )
            assert "diameter" in dims
            assert "length" in dims

    def test_render_dimension_inputs_cylindrical_tube(self):
        """_render_dimension_inputs for cylindrical tube."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 10.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            dims = ui._render_dimension_inputs(
                ElectrodeGeometry.CYLINDRICAL_TUBE, "anode",
            )
            assert "diameter" in dims
            assert "thickness" in dims
            assert "length" in dims

    def test_render_dimension_inputs_spherical(self):
        """_render_dimension_inputs for spherical returns diameter."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 20.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            dims = ui._render_dimension_inputs(
                ElectrodeGeometry.SPHERICAL, "anode",
            )
            assert "diameter" in dims

    def test_render_dimension_inputs_custom(self):
        """_render_dimension_inputs for custom returns areas."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 25.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import ElectrodeGeometry
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            dims = ui._render_dimension_inputs(
                ElectrodeGeometry.CUSTOM, "anode",
            )
            assert "projected_area" in dims
            assert "total_surface_area" in dims


@pytest.mark.apptest
class TestElectrodeConfigurationUICustomMaterial:
    """Tests for custom material properties."""

    def test_render_custom_material_properties(self):
        """render_custom_material_properties returns MaterialProperties."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 1000.0
        mock_st.slider.return_value = 75
        mock_st.checkbox.return_value = False
        mock_st.text_input.return_value = "Custom user specification"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import MaterialProperties
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            props = ui.render_custom_material_properties("anode")
            assert isinstance(props, MaterialProperties)

    def test_render_custom_material_porous(self):
        """render_custom_material_properties handles porous material."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 1000.0
        mock_st.slider.return_value = 0.8
        mock_st.checkbox.return_value = True  # is_porous
        mock_st.text_input.return_value = "Custom"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.electrode_config import MaterialProperties
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            props = ui.render_custom_material_properties("anode")
            assert isinstance(props, MaterialProperties)


@pytest.mark.apptest
class TestElectrodeConfigurationUIComparison:
    """Tests for electrode comparison."""

    def test_render_electrode_comparison_insufficient_data(self):
        """render_electrode_comparison returns early with < 2 electrodes."""
        mock_st = _make_mock_st()
        mock_st.session_state["electrode_calculations"] = {}
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            # Should return without error
            ui.render_electrode_comparison()

    def test_render_electrode_comparison_with_data(self):
        """render_electrode_comparison renders table and chart."""
        mock_st = _make_mock_st()
        mock_st.session_state["electrode_calculations"] = {
            "anode": {
                "projected_area_cm2": 25.0,
                "geometric_area_cm2": 50.0,
                "effective_area_cm2": 100.0,
                "enhancement_factor": 4.0,
                "biofilm_capacity_ul": 10.0,
                "charge_transfer_coeff": 0.5,
                "volume_cm3": 1.0,
            },
            "cathode": {
                "projected_area_cm2": 25.0,
                "geometric_area_cm2": 50.0,
                "effective_area_cm2": 80.0,
                "enhancement_factor": 3.2,
                "biofilm_capacity_ul": 8.0,
                "charge_transfer_coeff": 0.4,
                "volume_cm3": 1.0,
            },
        }
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            ui.render_electrode_comparison()
            mock_st.dataframe.assert_called()
            mock_st.plotly_chart.assert_called()


@pytest.mark.apptest
class TestElectrodeConfigurationUISummary:
    """Tests for configuration summary."""

    def test_render_configuration_summary(self):
        """render_configuration_summary displays summary info."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.electrode_configuration_ui import ElectrodeConfigurationUI

            ui = ElectrodeConfigurationUI()
            config = MagicMock()
            config.get_configuration_summary.return_value = {
                "material": "graphite_plate",
                "geometry": "rectangular_plate",
                "specific_surface_area_m2_per_g": 0.25,
                "effective_area_cm2": 100.0,
                "biofilm_capacity_ul": 10.0,
                "charge_transfer_coeff": 0.5,
                "specific_conductance_S_per_m": 25000,
                "hydrophobicity_angle_deg": 75.0,
            }
            ui.render_configuration_summary(config, "anode")
            mock_st.write.assert_called()


# ===================================================================
# PART 4: membrane_configuration_ui.py (185 stmts)
# ===================================================================


@pytest.mark.apptest
class TestMembraneConfigurationUIInit:
    """Tests for MembraneConfigurationUI constructor."""

    def test_init_calls_initialize_session_state(self):
        """Constructor initializes session state keys."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            _ui = MembraneConfigurationUI()
            assert "membrane_config" in mock_st.session_state
            assert "membrane_calculations" in mock_st.session_state

    def test_initialize_session_state_preserves_existing(self):
        """initialize_session_state does not overwrite existing values."""
        mock_st = _make_mock_st()
        mock_st.session_state["membrane_config"] = "existing"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            _ui = MembraneConfigurationUI()
            assert mock_st.session_state["membrane_config"] == "existing"


@pytest.mark.apptest
class TestMembraneConfigurationUIMaterial:
    """Tests for membrane material selection."""

    def test_render_material_selector_nafion(self):
        """render_material_selector returns Nafion 117."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "Nafion 117"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.membrane_config import MembraneMaterial
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            material = ui.render_material_selector()
            assert material == MembraneMaterial.NAFION_117

    def test_render_material_selector_custom(self):
        """render_material_selector returns CUSTOM."""
        mock_st = _make_mock_st()
        mock_st.selectbox.return_value = "Custom Material"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.membrane_config import MembraneMaterial
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            material = ui.render_material_selector()
            assert material == MembraneMaterial.CUSTOM

    def test_display_material_properties(self):
        """_display_material_properties renders metrics."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.membrane_config import (
                MEMBRANE_PROPERTIES_DATABASE,
                MembraneMaterial,
            )
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            props = MEMBRANE_PROPERTIES_DATABASE[MembraneMaterial.NAFION_117]
            ui._display_material_properties(props, "Nafion 117")
            mock_st.metric.assert_called()


@pytest.mark.apptest
class TestMembraneConfigurationUIArea:
    """Tests for membrane area input."""

    def test_render_area_input(self):
        """render_area_input returns area in m2."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 25.0
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            area = ui.render_area_input()
            assert area == 25.0 / 10000  # cm2 to m2


@pytest.mark.apptest
class TestMembraneConfigurationUIOperating:
    """Tests for operating conditions."""

    def test_render_operating_conditions(self):
        """render_operating_conditions returns temp, ph_anode, ph_cathode."""
        mock_st = _make_mock_st()
        mock_st.number_input.side_effect = [25.0, 7.0, 7.0]
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            temp, ph_a, ph_c = ui.render_operating_conditions()
            assert temp == 25.0
            assert ph_a == 7.0
            assert ph_c == 7.0

    def test_render_operating_conditions_ph_warning(self):
        """render_operating_conditions warns on large pH gradient."""
        mock_st = _make_mock_st()
        mock_st.number_input.side_effect = [25.0, 3.0, 10.0]
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            ui.render_operating_conditions()
            mock_st.warning.assert_called()


@pytest.mark.apptest
class TestMembraneConfigurationUICustom:
    """Tests for custom membrane properties."""

    def test_render_custom_membrane_properties(self):
        """render_custom_membrane_properties returns MembraneProperties."""
        mock_st = _make_mock_st()
        mock_st.number_input.return_value = 0.05
        mock_st.slider.return_value = 0.95
        mock_st.text_input.return_value = "Custom"
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.membrane_config import MembraneProperties
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            props = ui.render_custom_membrane_properties()
            assert isinstance(props, MembraneProperties)


@pytest.mark.apptest
class TestMembraneConfigurationUIPerformance:
    """Tests for performance display."""

    def test_calculate_and_display_performance(self):
        """calculate_and_display_performance displays metrics and table."""
        mock_st = _make_mock_st()
        mock_st.columns.side_effect = _smart_columns
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            config = MagicMock()
            config.calculate_resistance.return_value = 0.5
            config.properties.proton_conductivity = 0.05
            config.properties.area_resistance = 2.0
            config.properties.expected_lifetime = 2000.0
            config.area = 0.0025
            config.ph_anode = 7.0
            config.ph_cathode = 7.0
            config.calculate_proton_flux.return_value = 0.001
            config.estimate_lifetime_factor.return_value = 0.8

            ui.calculate_and_display_performance(config)
            mock_st.metric.assert_called()
            mock_st.dataframe.assert_called()


@pytest.mark.apptest
class TestMembraneConfigurationUISummary:
    """Tests for membrane configuration summary."""

    def test_render_configuration_summary(self):
        """render_configuration_summary displays config info."""
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            config = MagicMock()
            config.material.value = "nafion_117"
            config.area = 0.0025
            config.properties.thickness = 183.0
            config.operating_temperature = 25.0
            config.ph_anode = 7.0
            config.ph_cathode = 7.0
            config.calculate_resistance.return_value = 0.5
            config.properties.proton_conductivity = 0.05
            config.properties.permselectivity = 0.95
            config.properties.expected_lifetime = 2000.0
            config.properties.cost_per_m2 = 500.0

            ui.render_configuration_summary(config)
            mock_st.write.assert_called()


@pytest.mark.apptest
class TestMembraneConfigurationUIVisualization:
    """Tests for membrane visualization."""

    def test_render_membrane_visualization(self):
        """render_membrane_visualization creates plotly charts."""
        mock_st = _make_mock_st()
        tabs = [MagicMock() for _ in range(3)]
        mock_st.tabs.return_value = tabs
        for t in tabs:
            t.__enter__ = MagicMock(return_value=t)
            t.__exit__ = MagicMock(return_value=False)
        mock_st.session_state["target_current_density"] = 100.0
        mock_st.slider.return_value = 100.0

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            from config.membrane_config import MembraneMaterial
            from gui.membrane_configuration_ui import MembraneConfigurationUI

            ui = MembraneConfigurationUI()
            config = MagicMock()
            config.material = MembraneMaterial.NAFION_117
            config.properties.proton_conductivity = 0.05
            config.properties.permselectivity = 0.95
            config.properties.area_resistance = 2.0
            config.properties.oxygen_permeability = 1e-12
            config.properties.substrate_permeability = 1e-14
            config.properties.expected_lifetime = 2000.0
            config.estimate_lifetime_factor.return_value = 0.8

            ui.render_membrane_visualization(config)
            mock_st.plotly_chart.assert_called()


# ===================================================================
# PART 5: simulation_runner.py (270 stmts)
# ===================================================================


@pytest.mark.apptest
class TestSimulationRunnerInit:
    """Tests for SimulationRunner constructor."""

    def test_init_defaults(self):
        """Constructor initializes default values."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.simulation is None
        assert runner.is_running is False
        assert runner.should_stop is False
        assert runner.thread is None
        assert runner.current_output_dir is None
        assert runner.live_data_buffer == []
        assert runner.gui_refresh_interval == 5.0

    def test_init_queues(self):
        """Constructor creates results and data queues."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert isinstance(runner.results_queue, queue.Queue)
        assert isinstance(runner.data_queue, queue.Queue)
        assert runner.data_queue.maxsize == 100

    def test_init_phase2_flags(self):
        """Constructor initializes Phase 2 change detection flags."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.last_data_count == 0
        assert runner.last_plot_hash is None
        assert runner.last_metrics_hash is None
        assert runner.plot_dirty_flag is True
        assert runner.metrics_dirty_flag is True

    def test_init_phase3_parquet(self):
        """Constructor initializes Phase 3 Parquet config."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.parquet_buffer == []
        assert runner.parquet_batch_size == 100
        assert runner.parquet_writer is None
        assert runner.parquet_schema is None
        assert runner.enable_parquet is True


@pytest.mark.apptest
class TestSimulationRunnerStartStop:
    """Tests for start/stop simulation."""

    def test_start_simulation_when_already_running(self):
        """start_simulation returns False when already running."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.is_running = True
        result = runner.start_simulation(MagicMock(), 10)
        assert result is False

    def test_start_simulation_creates_thread(self):
        """start_simulation creates and starts a thread."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        config = MagicMock()

        with patch.object(
            runner,
            "_run_simulation",
            side_effect=lambda *a, **kw: None,
        ):
            result = runner.start_simulation(config, 10)
            assert result is True
            assert runner.is_running is True
            assert runner.thread is not None
            # Wait for the thread
            runner.thread.join(timeout=2)

    def test_stop_simulation_when_not_running(self):
        """stop_simulation returns False when not running."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.stop_simulation()
        assert result is False

    def test_stop_simulation_when_running(self):
        """stop_simulation sets should_stop and cleans up."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.is_running = True
        runner.live_data_buffer = [{"a": 1}, {"b": 2}]

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False
        runner.thread = mock_thread

        result = runner.stop_simulation()
        assert result is True
        assert runner.is_running is False
        assert runner.should_stop is False
        assert runner.thread is None
        assert len(runner.live_data_buffer) == 0

    def test_stop_simulation_thread_timeout(self):
        """stop_simulation handles thread that does not stop."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.is_running = True

        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Thread refuses to die
        runner.thread = mock_thread

        result = runner.stop_simulation()
        assert result is False


@pytest.mark.apptest
class TestSimulationRunnerCleanup:
    """Tests for cleanup methods."""

    def test_force_cleanup(self):
        """_force_cleanup resets running state."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.is_running = True
        runner.should_stop = True
        runner._force_cleanup()
        assert runner.is_running is False
        assert runner.should_stop is False

    def test_cleanup_resources_handles_import_errors(self):
        """_cleanup_resources handles missing jax/torch gracefully."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        # Should not raise even if jax/torch not installed
        runner._cleanup_resources()


@pytest.mark.apptest
class TestSimulationRunnerDataAccess:
    """Tests for data access methods."""

    def test_get_status_empty_queue(self):
        """get_status returns None when queue is empty."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.get_status() is None

    def test_get_status_with_message(self):
        """get_status returns message from queue."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.results_queue.put(("completed", {"result": "ok"}, "/path"))
        status = runner.get_status()
        assert status[0] == "completed"

    def test_get_live_data_empty(self):
        """get_live_data returns empty list when no data."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        data = runner.get_live_data()
        assert data == []

    def test_get_live_data_with_points(self):
        """get_live_data drains queue and adds to buffer."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.data_queue.put({"time": 1.0, "value": 0.5})
        runner.data_queue.put({"time": 2.0, "value": 0.6})

        data = runner.get_live_data()
        assert len(data) == 2
        assert len(runner.live_data_buffer) == 2

    def test_get_live_data_buffer_limit(self):
        """get_live_data trims buffer to 1000 points."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        # Pre-fill buffer with 999 points
        runner.live_data_buffer = [{"time": i} for i in range(999)]
        # Add 5 more via queue
        for i in range(5):
            runner.data_queue.put({"time": 999 + i})

        runner.get_live_data()
        assert len(runner.live_data_buffer) == 1000

    def test_get_buffered_data_empty(self):
        """get_buffered_data returns None when buffer is empty."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.get_buffered_data() is None

    def test_get_buffered_data_with_data(self):
        """get_buffered_data returns DataFrame."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.live_data_buffer = [
            {"time": 1.0, "value": 0.5},
            {"time": 2.0, "value": 0.6},
        ]
        df = runner.get_buffered_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "time" in df.columns


@pytest.mark.apptest
class TestSimulationRunnerChangeDetection:
    """Tests for Phase 2: Change Detection."""

    def test_has_data_changed_no_change(self):
        """has_data_changed returns False when count unchanged."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.has_data_changed() is False

    def test_has_data_changed_with_change(self):
        """has_data_changed returns True when buffer grows."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.live_data_buffer = [{"a": 1}]
        assert runner.has_data_changed() is True
        assert runner.plot_dirty_flag is True
        assert runner.metrics_dirty_flag is True

    def test_has_data_changed_subsequent_no_change(self):
        """has_data_changed returns False on second call without new data."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.live_data_buffer = [{"a": 1}]
        runner.has_data_changed()
        assert runner.has_data_changed() is False

    def test_should_update_plots_dirty(self):
        """should_update_plots returns True when dirty flag set."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.plot_dirty_flag = True
        assert runner.should_update_plots() is True
        assert runner.plot_dirty_flag is False

    def test_should_update_plots_not_dirty(self):
        """should_update_plots returns False when flag cleared."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.plot_dirty_flag = False
        assert runner.should_update_plots() is False

    def test_should_update_plots_force(self):
        """should_update_plots with force=True always returns True."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.plot_dirty_flag = False
        assert runner.should_update_plots(force=True) is True

    def test_should_update_metrics_dirty(self):
        """should_update_metrics returns True when dirty."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.metrics_dirty_flag = True
        assert runner.should_update_metrics() is True
        assert runner.metrics_dirty_flag is False

    def test_should_update_metrics_force(self):
        """should_update_metrics with force=True always returns True."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.metrics_dirty_flag = False
        assert runner.should_update_metrics(force=True) is True

    def test_get_incremental_update_info(self):
        """get_incremental_update_info returns status dict."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        info = runner.get_incremental_update_info()
        assert "has_new_data" in info
        assert "data_count" in info
        assert "needs_plot_update" in info
        assert "needs_metrics_update" in info


@pytest.mark.apptest
class TestSimulationRunnerParquet:
    """Tests for Phase 3: Parquet Migration."""

    def test_create_parquet_schema_empty(self):
        """create_parquet_schema returns None when data is empty."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.create_parquet_schema(None)
        assert result is None

    def test_create_parquet_schema_disabled(self):
        """create_parquet_schema returns None when disabled."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.enable_parquet = False
        result = runner.create_parquet_schema({"time": 1.0})
        assert result is None

    def test_create_parquet_schema_with_data(self):
        """create_parquet_schema creates schema from sample data."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5, "step": 1}
        schema = runner.create_parquet_schema(sample)
        assert schema is not None
        assert runner.parquet_schema is not None

    def test_init_parquet_writer_disabled(self):
        """init_parquet_writer returns False when disabled."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.enable_parquet = False
        result = runner.init_parquet_writer("/tmp")
        assert result is False

    def test_init_parquet_writer_no_schema(self):
        """init_parquet_writer returns False when no schema."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        result = runner.init_parquet_writer("/tmp")
        assert result is False

    def test_init_parquet_writer_success(self, tmp_path):
        """init_parquet_writer creates writer successfully."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        runner.create_parquet_schema(sample)
        result = runner.init_parquet_writer(str(tmp_path))
        assert result is True
        assert runner.parquet_writer is not None
        # Clean up
        runner.close_parquet_writer()

    def test_write_parquet_batch_not_enough_data(self):
        """write_parquet_batch returns None when buffer too small."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.parquet_buffer = [{"a": 1}]
        result = runner.write_parquet_batch()
        assert result is None

    def test_write_parquet_batch_disabled(self):
        """write_parquet_batch returns None when disabled."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        runner.enable_parquet = False
        result = runner.write_parquet_batch()
        assert result is None

    def test_write_parquet_batch_success(self, tmp_path):
        """write_parquet_batch writes data when buffer is full."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        runner.create_parquet_schema(sample)
        runner.init_parquet_writer(str(tmp_path))

        # Fill buffer to batch size
        runner.parquet_buffer = [
            {"time_hours": float(i), "power": float(i) * 0.1}
            for i in range(100)
        ]
        result = runner.write_parquet_batch()
        assert result is True
        assert len(runner.parquet_buffer) == 0
        runner.close_parquet_writer()

    def test_close_parquet_writer_no_writer(self):
        """close_parquet_writer returns True when no writer."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        assert runner.close_parquet_writer() is True

    def test_close_parquet_writer_with_remaining_buffer(self, tmp_path):
        """close_parquet_writer flushes remaining buffer."""
        from gui.simulation_runner import SimulationRunner

        runner = SimulationRunner()
        sample = {"time_hours": 1.0, "power": 0.5}
        runner.create_parquet_schema(sample)
        runner.init_parquet_writer(str(tmp_path))

        runner.parquet_buffer = [
            {"time_hours": 1.0, "power": 0.5},
            {"time_hours": 2.0, "power": 0.6},
        ]
        result = runner.close_parquet_writer()
        assert result is True
        assert runner.parquet_writer is None
        assert len(runner.parquet_buffer) == 0
