#!/usr/bin/env python3
"""Test live monitoring dashboard."""

import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
mock_st.metric = MagicMock()
mock_st.plotly_chart = MagicMock()
mock_st.empty = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.selectbox = MagicMock(return_value="1s")
mock_st.checkbox = MagicMock(return_value=False)
mock_st.slider = MagicMock(return_value=100)
mock_st.success = MagicMock()
mock_st.warning = MagicMock()
mock_st.error = MagicMock()
mock_st.sidebar = MagicMock()
mock_st.expander = MagicMock()
mock_st.dataframe = MagicMock()
sys.modules['streamlit'] = mock_st

# Mock plotly
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()


class TestLiveMonitoring(unittest.TestCase):
    """Test live monitoring dashboard."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.live_monitoring_dashboard
        self.assertIsNotNone(gui.live_monitoring_dashboard)

    def test_render_live_monitoring(self):
        """Test render live monitoring dashboard."""
        from gui.live_monitoring_dashboard import render_live_monitoring_dashboard
        
        # Should not raise
        render_live_monitoring_dashboard()
        
        # Check that streamlit methods were called
        self.assertTrue(mock_st.method_calls)