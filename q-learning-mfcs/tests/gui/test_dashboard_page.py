#!/usr/bin/env python3
"""Test dashboard page."""

import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.title = MagicMock()
mock_st.metric = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
mock_st.plotly_chart = MagicMock()
mock_st.success = MagicMock()
mock_st.info = MagicMock()
mock_st.container = MagicMock()
sys.modules['streamlit'] = mock_st

# Mock plotly
sys.modules['plotly.graph_objects'] = MagicMock()


class TestDashboardPage(unittest.TestCase):
    """Test dashboard page."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.pages.dashboard
        self.assertIsNotNone(gui.pages.dashboard)

    def test_render_dashboard_page(self):
        """Test render dashboard page."""
        from gui.pages.dashboard import render_dashboard_page
        
        # Should not raise
        render_dashboard_page()
        
        # Check that streamlit methods were called
        mock_st.title.assert_called()
        self.assertTrue(mock_st.method_calls)


if __name__ == '__main__':
    unittest.main()