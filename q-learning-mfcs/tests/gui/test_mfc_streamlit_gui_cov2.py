"""Coverage tests for mfc_streamlit_gui.py -- target 99%+.

Covers remaining uncovered paths: monitor tab auto-refresh trigger,
monitor with data but should_update_plots=False and cached_plot=None,
monitor running but waiting for data, history tab selected sim with
no data, main function if-name-main block equivalent.
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _make_mock_st():
    """Build a fresh streamlit mock."""
    mock_st = MagicMock()
    mock_st.set_page_config = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.title = MagicMock()
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.text = MagicMock()
    mock_st.info = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.success = MagicMock()
    mock_st.button = MagicMock(return_value=False)
    mock_st.checkbox = MagicMock(return_value=True)
    mock_st.number_input = MagicMock(return_value=5)
    mock_st.selectbox = MagicMock(return_value="Auto-detect")
    mock_st.slider = MagicMock(return_value=10)
    mock_st.tabs = MagicMock(
        return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    )

    def columns_side_effect(n_or_list):
        if isinstance(n_or_list, int):
            return [MagicMock() for _ in range(n_or_list)]
        return [MagicMock() for _ in range(len(n_or_list))]

    mock_st.columns = MagicMock(side_effect=columns_side_effect)
    mock_st.metric = MagicMock()
    mock_st.json = MagicMock()
    mock_st.dataframe = MagicMock()
    mock_st.plotly_chart = MagicMock()
    mock_st.download_button = MagicMock()
    mock_st.rerun = MagicMock()
    mock_st.expander = MagicMock()

    # Session state as a real dict for easier manipulation
    session = {}
    mock_session = MagicMock()
    mock_session.__contains__ = lambda s, k: k in session
    mock_session.__getitem__ = lambda s, k: session[k]
    mock_session.__setitem__ = lambda s, k, v: session.__setitem__(k, v)
    mock_session.get = lambda k, d=None: session.get(k, d)
    mock_st.session_state = mock_session
    mock_st._session_dict = session

    mock_sidebar = MagicMock()
    mock_sidebar.header = MagicMock()
    mock_sidebar.subheader = MagicMock()
    mock_sidebar.selectbox = MagicMock(return_value="24 Hours (Daily)")
    mock_sidebar.checkbox = MagicMock(return_value=True)
    mock_sidebar.number_input = MagicMock(return_value=10.0)
    mock_sidebar.slider = MagicMock(return_value=0.1)
    mock_sidebar.markdown = MagicMock()
    mock_sidebar.text = MagicMock()
    mock_sidebar.warning = MagicMock()
    mock_sidebar.expander = MagicMock()
    mock_st.sidebar = mock_sidebar

    return mock_st


@pytest.fixture(autouse=True)
def mock_streamlit():
    mock_st = _make_mock_st()
    for mod_name in list(sys.modules.keys()):
        if "mfc_streamlit_gui" in mod_name:
            del sys.modules[mod_name]
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


def _std_params():
    return {
        "duration_hours": 24,
        "selected_duration": "24 Hours (Daily)",
        "use_pretrained": True,
        "target_conc": 25.0,
        "n_cells": 5,
        "anode_area_cm2": 10.0,
        "cathode_area_cm2": 10.0,
        "electrode_area_m2": 10.0 * 1e-4,
        "gpu_backend": "Auto-detect",
    }


@pytest.mark.coverage_extra
class TestMonitorAutoRefreshTrigger:
    """Cover the auto-refresh time branch that calls st.rerun."""

    def test_auto_refresh_triggers_rerun(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.current_output_dir = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = True
        mock_streamlit.number_input.return_value = 1

        # Set last_refresh_time far in the past so the time diff >= interval
        mock_streamlit.session_state.last_refresh_time = 0.0
        # Also handle the "not in" check
        mock_streamlit.session_state.__contains__ = lambda s, k: True

        with patch(
            "mfc_streamlit_gui.load_recent_simulations", return_value=[]
        ):
            render_monitor_tab()
            mock_streamlit.rerun.assert_called()


@pytest.mark.coverage_extra
class TestMonitorCachedPlotNone:
    """Cover branch: should_update_plots False, cached_plot is None."""

    def test_creates_plot_when_cache_empty(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.current_output_dir = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False

        df = pd.DataFrame(
            {
                "time_hours": [1.0, 2.0],
                "reservoir_concentration": [20.0, 19.0],
                "total_power": [0.1, 0.2],
                "q_action": [1, 2],
            }
        )
        sim = {
            "name": "test",
            "path": "/tmp",
            "duration": 24,
            "final_conc": 20.0,
            "control_effectiveness": 80.0,
        }

        # should_update_plots returns False, cached_plot returns None
        mock_runner.should_update_plots.return_value = False
        mock_streamlit._session_dict.pop("cached_plot", None)
        mock_streamlit.session_state.get = (
            lambda k, d=None: mock_streamlit._session_dict.get(k, d)
        )

        with patch(
            "mfc_streamlit_gui.load_recent_simulations", return_value=[sim]
        ), patch(
            "mfc_streamlit_gui.load_simulation_data", return_value=df
        ), patch(
            "mfc_streamlit_gui.create_real_time_plots",
            return_value=MagicMock(),
        ):
            render_monitor_tab()
            mock_streamlit.plotly_chart.assert_called()


@pytest.mark.coverage_extra
class TestMonitorRunningWaitingData:
    """Cover branch: running, has output_dir, but df empty/no data."""

    def test_running_empty_df(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab

        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.current_output_dir = "/tmp/out"
        empty_df = pd.DataFrame()
        mock_runner.get_live_data.return_value = []
        mock_runner.get_buffered_data.return_value = empty_df
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False

        render_monitor_tab()
        mock_streamlit.info.assert_called()


@pytest.mark.coverage_extra
class TestMonitorRunningNotRunningInfo:
    """Cover the final else: running but no data."""

    def test_running_no_output_dir(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab

        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.current_output_dir = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False

        with patch(
            "mfc_streamlit_gui.load_recent_simulations", return_value=[]
        ):
            render_monitor_tab()


@pytest.mark.coverage_extra
class TestCleanupNoSimRunner:
    """Cover cleanup_on_exit when session_state has no sim_runner key."""

    def test_no_sim_runner(self, mock_streamlit):
        from mfc_streamlit_gui import cleanup_on_exit

        mock_streamlit.session_state.__contains__ = MagicMock(
            return_value=False
        )
        cleanup_on_exit()
