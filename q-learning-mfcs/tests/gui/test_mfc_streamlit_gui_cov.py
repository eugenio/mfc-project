"""Tests for mfc_streamlit_gui.py - coverage target 98%+.

This module executes Streamlit calls at import time, so we must mock
streamlit before importing the module.
"""
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit at module level before any imports."""
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
    mock_st.selectbox = MagicMock(return_value="24 Hours (Daily)")
    mock_st.slider = MagicMock(return_value=10)
    mock_st.tabs = MagicMock(return_value=[
        MagicMock(), MagicMock(), MagicMock(), MagicMock()
    ])
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

    # Session state
    mock_session = MagicMock()
    mock_session.__contains__ = MagicMock(return_value=False)
    mock_session.__getitem__ = MagicMock(return_value=None)
    mock_session.__setitem__ = MagicMock()
    mock_session.get = MagicMock(return_value=5.0)
    mock_st.session_state = mock_session

    # Sidebar
    mock_sidebar = MagicMock()
    mock_sidebar.header = MagicMock()
    mock_sidebar.subheader = MagicMock()
    mock_sidebar.selectbox = MagicMock(return_value="24 Hours (Daily)")
    mock_sidebar.checkbox = MagicMock(return_value=True)
    mock_sidebar.number_input = MagicMock(return_value=10.0)
    mock_sidebar.slider = MagicMock(return_value=0.1)
    mock_sidebar.markdown = MagicMock()
    mock_sidebar.text = MagicMock()
    mock_sidebar.multiselect = MagicMock(return_value=[])
    mock_sidebar.warning = MagicMock()
    mock_sidebar.expander = MagicMock()
    mock_st.sidebar = mock_sidebar

    # Invalidate cached module
    for mod_name in list(sys.modules.keys()):
        if "mfc_streamlit_gui" in mod_name:
            del sys.modules[mod_name]

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


class TestRenderSidebar:
    def test_returns_params(self, mock_streamlit):
        from mfc_streamlit_gui import render_sidebar

        mock_streamlit.sidebar.selectbox.return_value = "1 Hour (Quick Test)"
        mock_streamlit.sidebar.number_input.side_effect = [25.0, 5, 10.0, 10.0]
        mock_streamlit.sidebar.checkbox.return_value = True
        mock_streamlit.selectbox.return_value = "Auto-detect"

        params = render_sidebar()
        assert "duration_hours" in params
        assert "target_conc" in params
        assert "n_cells" in params
        assert "electrode_area_m2" in params


class TestRenderSimulationTab:
    def test_no_running(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.return_value = False

        params = {
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
        render_simulation_tab(params)

    def test_completed_status(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = (
            "completed", {"test": True}, "/tmp/out"
        )
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.return_value = False

        params = {
            "duration_hours": 1,
            "selected_duration": "1 Hour",
            "use_pretrained": True,
            "target_conc": 25.0,
            "n_cells": 5,
            "anode_area_cm2": 10.0,
            "cathode_area_cm2": 10.0,
            "electrode_area_m2": 10.0 * 1e-4,
            "gpu_backend": "Auto-detect",
        }
        render_simulation_tab(params)
        mock_streamlit.success.assert_called()


class TestRenderMonitorTab:
    def test_not_running_no_sims(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.current_output_dir = None
        mock_streamlit.session_state.sim_runner = mock_runner

        with patch(
            "mfc_streamlit_gui.load_recent_simulations", return_value=[]
        ):
            render_monitor_tab()


class TestRenderResultsTab:
    def test_no_results(self, mock_streamlit):
        from mfc_streamlit_gui import render_results_tab

        mock_streamlit.session_state.simulation_results = None
        render_results_tab()
        mock_streamlit.info.assert_called()

    def test_with_results(self, mock_streamlit):
        from mfc_streamlit_gui import render_results_tab

        mock_streamlit.session_state.simulation_results = {
            "performance_metrics": {"power": 0.5},
            "simulation_info": {"duration": 24},
        }
        render_results_tab()


class TestRenderHistoryTab:
    def test_no_history(self, mock_streamlit):
        from mfc_streamlit_gui import render_history_tab

        with patch(
            "mfc_streamlit_gui.load_recent_simulations", return_value=[]
        ):
            render_history_tab()
            mock_streamlit.info.assert_called()


class TestCleanupOnExit:
    def test_running(self, mock_streamlit):
        from mfc_streamlit_gui import cleanup_on_exit

        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_streamlit.session_state.__contains__ = MagicMock(
            return_value=True
        )
        mock_streamlit.session_state.sim_runner = mock_runner

        cleanup_on_exit()
        mock_runner.stop_simulation.assert_called_once()

    def test_not_running(self, mock_streamlit):
        from mfc_streamlit_gui import cleanup_on_exit

        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_streamlit.session_state.__contains__ = MagicMock(
            return_value=True
        )
        mock_streamlit.session_state.sim_runner = mock_runner

        cleanup_on_exit()

    def test_exception(self, mock_streamlit):
        from mfc_streamlit_gui import cleanup_on_exit

        mock_streamlit.session_state.__contains__ = MagicMock(
            side_effect=Exception("fail")
        )
        cleanup_on_exit()  # Should not raise
