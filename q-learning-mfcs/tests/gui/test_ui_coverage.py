#!/usr/bin/env python3
"""Comprehensive UI test suite for actual coverage."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit before any imports
mock_st = MagicMock()
mock_st.set_page_config = MagicMock()
mock_st.sidebar = MagicMock()
mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
mock_st.tabs = MagicMock(side_effect=lambda names: [MagicMock() for _ in range(len(names))])
sys.modules['streamlit'] = mock_st


class TestUIModules(unittest.TestCase):
    """Test UI modules for coverage."""

    def test_navigation_controller_import(self):
        """Test NavigationController can be imported."""
        from gui.navigation_controller import NavigationController
        self.assertIsNotNone(NavigationController)

    def test_enhanced_main_app_import(self):
        """Test enhanced_main_app can be imported."""
        import gui.enhanced_main_app
        self.assertTrue(hasattr(gui.enhanced_main_app, 'main'))

    def test_core_layout_functions(self):
        """Test core_layout functions."""
        # Import the module first
        import gui.core_layout
        
        # Configure the mock that the module is using
        gui.core_layout.st.sidebar.radio.return_value = "🏠 Dashboard"
        
        # Test theme application
        gui.core_layout.apply_enhanced_theme()
        gui.core_layout.st.markdown.assert_called()
        
        # Test sidebar creation
        result = gui.core_layout.create_navigation_sidebar()
        self.assertEqual(result, "dashboard")

    def test_scientific_widgets(self):
        """Test scientific widgets module."""
        from gui.scientific_widgets import ParameterSpec, ScientificParameterWidget
        
        # Test ParameterSpec
        spec = ParameterSpec(
            name="Test", unit="m", min_value=0.0, max_value=100.0,
            typical_range=(10.0, 50.0), literature_refs="", description="Test"
        )
        self.assertEqual(spec.name, "Test")
        
        # Test widget
        widget = ScientificParameterWidget(spec, "test_key")
        self.assertIsNotNone(widget)

    def test_browser_download_manager(self):
        """Test browser download manager."""
        from gui.browser_download_manager import BrowserDownloadManager
        
        manager = BrowserDownloadManager()
        self.assertIsNotNone(manager)
        self.assertIn("CSV", manager.supported_formats)


if __name__ == '__main__':
    unittest.main()