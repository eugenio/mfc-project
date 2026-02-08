"""Tests for mfc_streamlit_gui.py - deep coverage supplement.

Covers more render paths: status variants, button clicks,
auto-refresh, monitor with data, history with data, main function.
"""
import sys
import os
import time as _time
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture(autouse=True)
def mock_streamlit():
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

    mock_session = MagicMock()
    mock_session.__contains__ = MagicMock(return_value=False)
    mock_session.__getitem__ = MagicMock(return_value=None)
    mock_session.__setitem__ = MagicMock()
    mock_session.get = MagicMock(return_value=5.0)
    mock_st.session_state = mock_session

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

    for mod_name in list(sys.modules.keys()):
        if "mfc_streamlit_gui" in mod_name:
            del sys.modules[mod_name]

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


class TestRenderSidebarNotPretrained:
    def test_no_pretrained(self, mock_streamlit):
        from mfc_streamlit_gui import render_sidebar
        mock_streamlit.sidebar.selectbox.return_value = "1 Hour (Quick Test)"
        mock_streamlit.sidebar.checkbox.return_value = False
        mock_streamlit.sidebar.number_input.side_effect = [25.0, 5, 10.0, 10.0]
        mock_streamlit.selectbox.return_value = "CUDA"
        params = render_sidebar()
        assert params["gpu_backend"] == "CUDA"
        mock_streamlit.sidebar.slider.assert_called()


class TestRenderSimulationTabVariants:
    def _params(self):
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

    def test_stopped_status(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = (
            "stopped", "Stopped by user", "/tmp/out"
        )
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.return_value = False
        render_simulation_tab(self._params())
        mock_streamlit.warning.assert_called()

    def test_error_status(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = ("error", "Crash", None)
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.return_value = False
        render_simulation_tab(self._params())
        mock_streamlit.error.assert_called()

    def test_start_button_clicked(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = None
        mock_runner.start_simulation.return_value = True
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.side_effect = [True, False, False]
        render_simulation_tab(self._params())
        mock_runner.start_simulation.assert_called_once()

    def test_start_button_already_running(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = None
        mock_runner.start_simulation.return_value = False
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.side_effect = [True, False, False]
        render_simulation_tab(self._params())
        mock_streamlit.error.assert_called()

    def test_stop_button_clicked(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.get_status.return_value = None
        mock_runner.stop_simulation.return_value = True
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.side_effect = [False, True, False]
        render_simulation_tab(self._params())
        mock_runner.stop_simulation.assert_called_once()

    def test_stop_button_no_sim(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.get_status.return_value = None
        mock_runner.stop_simulation.return_value = False
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.side_effect = [False, True, False]
        render_simulation_tab(self._params())

    def test_refresh_button(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.get_status.return_value = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.side_effect = [False, False, True]
        render_simulation_tab(self._params())
        mock_streamlit.rerun.assert_called()

    def test_running_status_display(self, mock_streamlit):
        from mfc_streamlit_gui import render_simulation_tab
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.get_status.return_value = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.button.return_value = False
        render_simulation_tab(self._params())


class TestRenderMonitorTabWithData:
    def test_running_with_data(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.current_output_dir = "/tmp/out"
        df = pd.DataFrame({
            "time_hours": [1.0, 2.0],
            "reservoir_concentration": [20.0, 19.0],
            "total_power": [0.1, 0.2],
            "q_action": [1, 2],
        })
        mock_runner.get_live_data.return_value = [{"t": 1}]
        mock_runner.get_buffered_data.return_value = df
        mock_runner.should_update_plots.return_value = True
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False
        mock_streamlit.number_input.return_value = 5
        with patch("mfc_streamlit_gui.create_real_time_plots", return_value=MagicMock()):
            render_monitor_tab()

    def test_running_no_data(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_runner.current_output_dir = "/tmp/out"
        mock_runner.get_live_data.return_value = []
        mock_runner.get_buffered_data.return_value = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False
        render_monitor_tab()

    def test_cached_plot(self, mock_streamlit):
        from mfc_streamlit_gui import render_monitor_tab
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_runner.current_output_dir = None
        mock_streamlit.session_state.sim_runner = mock_runner
        mock_streamlit.checkbox.return_value = False
        df = pd.DataFrame({
            "time_hours": [1.0],
            "reservoir_concentration": [20.0],
            "total_power": [0.1],
            "q_action": [1],
        })
        sim = {"name": "test", "path": "/tmp", "duration": 24,
               "final_conc": 20.0, "control_effectiveness": 80.0}
        with patch("mfc_streamlit_gui.load_recent_simulations",
                   return_value=[sim]), \
             patch("mfc_streamlit_gui.load_simulation_data",
                   return_value=df):
            mock_runner.should_update_plots.return_value = False
            mock_streamlit.session_state.__setitem__ = dict.__setitem__.__get__(
                mock_streamlit.session_state
            ) if False else lambda s, k, v: None
            with patch("mfc_streamlit_gui.create_real_time_plots",
                       return_value=MagicMock()):
                render_monitor_tab()


class TestRenderHistoryWithData:
    def test_with_simulations(self, mock_streamlit):
        from mfc_streamlit_gui import render_history_tab
        sim = {"name": "test", "path": "/tmp", "duration": 24,
               "final_conc": 20.0, "control_effectiveness": 80.0}
        df = pd.DataFrame({
            "time_hours": [1.0],
            "reservoir_concentration": [20.0],
        })
        mock_streamlit.selectbox.return_value = sim
        with patch("mfc_streamlit_gui.load_recent_simulations",
                   return_value=[sim]), \
             patch("mfc_streamlit_gui.load_simulation_data",
                   return_value=df), \
             patch("mfc_streamlit_gui.create_real_time_plots",
                   return_value=MagicMock()):
            render_history_tab()

    def test_history_no_data(self, mock_streamlit):
        from mfc_streamlit_gui import render_history_tab
        sim = {"name": "test", "path": "/tmp", "duration": 24,
               "final_conc": 20.0, "control_effectiveness": 80.0}
        mock_streamlit.selectbox.return_value = sim
        with patch("mfc_streamlit_gui.load_recent_simulations",
                   return_value=[sim]), \
             patch("mfc_streamlit_gui.load_simulation_data",
                   return_value=None):
            render_history_tab()


class TestMainFunction:
    def test_main_not_running(self, mock_streamlit):
        from mfc_streamlit_gui import main
        mock_runner = MagicMock()
        mock_runner.is_running = False
        mock_streamlit.session_state.sim_runner = mock_runner
        with patch("mfc_streamlit_gui.render_sidebar",
                   return_value={"duration_hours": 24}), \
             patch("mfc_streamlit_gui.render_simulation_tab"), \
             patch("mfc_streamlit_gui.render_monitor_tab"), \
             patch("mfc_streamlit_gui.render_results_tab"), \
             patch("mfc_streamlit_gui.render_history_tab"):
            main()

    def test_main_running(self, mock_streamlit):
        from mfc_streamlit_gui import main
        mock_runner = MagicMock()
        mock_runner.is_running = True
        mock_streamlit.session_state.sim_runner = mock_runner
        with patch("mfc_streamlit_gui.render_sidebar",
                   return_value={"duration_hours": 24}), \
             patch("mfc_streamlit_gui.render_simulation_tab"), \
             patch("mfc_streamlit_gui.render_monitor_tab"), \
             patch("mfc_streamlit_gui.render_results_tab"), \
             patch("mfc_streamlit_gui.render_history_tab"):
            main()
            mock_streamlit.sidebar.warning.assert_called()
