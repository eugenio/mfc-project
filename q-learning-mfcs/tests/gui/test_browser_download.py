#!/usr/bin/env python3
"""Test browser download manager."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.download_button = MagicMock()
mock_st.selectbox = MagicMock(return_value="CSV")
mock_st.checkbox = MagicMock(return_value=False)
mock_st.expander = MagicMock()
mock_st.markdown = MagicMock()
mock_st.info = MagicMock()
mock_st.warning = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.text_input = MagicMock(return_value="test_filename")

# Mock columns with dynamic return based on argument
def mock_columns(spec):
    if isinstance(spec, list):
        return [MagicMock() for _ in range(len(spec))]
    elif isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    else:
        return [MagicMock(), MagicMock()]  # default 2 columns

mock_st.columns = MagicMock(side_effect=mock_columns)
sys.modules['streamlit'] = mock_st


class TestBrowserDownload(unittest.TestCase):
    """Test browser download manager."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.browser_download_manager
        self.assertIsNotNone(gui.browser_download_manager)

    def test_download_manager_init(self):
        """Test BrowserDownloadManager initialization."""
        from gui.browser_download_manager import BrowserDownloadManager

        manager = BrowserDownloadManager()
        self.assertIsNotNone(manager)
        self.assertIn("CSV", manager.supported_formats)

    def test_render_browser_downloads(self):
        """Test render_browser_downloads function."""
        from gui.browser_download_manager import render_browser_downloads

        # Should not raise
        render_browser_downloads(
            simulation_data={"test": [1, 2, 3]},
            q_learning_data=None,
            analysis_results=None
        )


if __name__ == '__main__':
    unittest.main()
