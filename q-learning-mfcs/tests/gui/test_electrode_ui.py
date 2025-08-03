#!/usr/bin/env python3
"""Test electrode configuration UI."""

import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.tabs = MagicMock(return_value=[MagicMock() for _ in range(2)])
mock_st.selectbox = MagicMock(return_value="Carbon Cloth")
mock_st.number_input = MagicMock(return_value=100.0)
mock_st.slider = MagicMock(return_value=0.5)
mock_st.checkbox = MagicMock(return_value=False)
mock_st.form = MagicMock()
mock_st.form_submit_button = MagicMock(return_value=False)
mock_st.plotly_chart = MagicMock()
mock_st.dataframe = MagicMock()
mock_st.metric = MagicMock()
mock_st.expander = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.radio = MagicMock(return_value="Anode")
sys.modules['streamlit'] = mock_st

# Mock plotly
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()


class TestElectrodeUI(unittest.TestCase):
    """Test electrode configuration UI."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.electrode_configuration_ui
        self.assertIsNotNone(gui.electrode_configuration_ui)

    def test_render_electrode_config(self):
        """Test render electrode configuration."""
        from gui.electrode_configuration_ui import render_electrode_configuration_ui
        
        # Should not raise
        render_electrode_configuration_ui()
        
        # Check that streamlit methods were called