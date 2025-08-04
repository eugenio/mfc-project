#!/usr/bin/env python3
"""Test core layout module."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.title = MagicMock()
mock_st.header = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.container = MagicMock()
mock_st.sidebar = MagicMock()
mock_st.empty = MagicMock()
mock_st.markdown = MagicMock()
mock_st.caption = MagicMock()
# Mock sidebar.radio to return a valid navigation option
mock_st.sidebar.radio = MagicMock(return_value="🏠 Dashboard")
mock_st.sidebar.title = MagicMock()
mock_st.sidebar.subheader = MagicMock()
mock_st.sidebar.text = MagicMock()
sys.modules['streamlit'] = mock_st


class TestCoreLayout(unittest.TestCase):
    """Test core layout functions."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.core_layout
        self.assertIsNotNone(gui.core_layout)

    def test_create_page_layout(self):
        """Test create page layout function."""
        from gui.core_layout import create_page_layout

        # Should not raise
        layout = create_page_layout("Test Page")
        self.assertIsNotNone(layout)

    def test_render_header(self):
        """Test render header function."""
        from gui.core_layout import render_header

        # Should not raise
        render_header("Test Title", "Test Subtitle")
        mock_st.title.assert_called()


if __name__ == '__main__':
    unittest.main()
