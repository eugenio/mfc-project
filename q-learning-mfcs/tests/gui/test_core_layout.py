#!/usr/bin/env python3
"""Comprehensive tests for GUI core layout functionality."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock all dependencies before importing GUI modules
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.title = MagicMock()
mock_st.caption = MagicMock()
mock_st.success = MagicMock()
mock_st.info = MagicMock()
mock_st.warning = MagicMock()
mock_st.metric = MagicMock()
mock_st.markdown = MagicMock()
mock_st.sidebar = MagicMock()
mock_st.sidebar.title = MagicMock()
mock_st.sidebar.radio = MagicMock(return_value="🏠 Dashboard")
mock_st.sidebar.subheader = MagicMock()
mock_st.sidebar.text = MagicMock()
mock_st.header = MagicMock()
mock_st.write = MagicMock()
mock_st.container = MagicMock()
mock_st.empty = MagicMock()
sys.modules['streamlit'] = mock_st

# Mock other dependencies
mock_numpy = MagicMock()
sys.modules['numpy'] = mock_numpy

mock_pandas = MagicMock()
sys.modules['pandas'] = mock_pandas

mock_plotly = MagicMock()
mock_plotly.graph_objects = MagicMock()
mock_plotly.express = MagicMock()
mock_plotly.subplots = MagicMock()
mock_plotly.subplots.make_subplots = MagicMock()
sys.modules['plotly'] = mock_plotly
sys.modules['plotly.graph_objects'] = mock_plotly.graph_objects
sys.modules['plotly.express'] = mock_plotly.express
sys.modules['plotly.subplots'] = mock_plotly.subplots

mock_altair = MagicMock()
sys.modules['altair'] = mock_altair


class TestCoreLayout(unittest.TestCase):
    """Comprehensive test suite for core layout functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset all mocks before each test
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock() for _ in range(3)]
        mock_st.container.return_value = MagicMock()
        mock_st.sidebar.radio.return_value = "🏠 Dashboard"
        
        # Import core_layout module directly to avoid GUI __init__ issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "core_layout", 
            os.path.join(os.path.dirname(__file__), '../../src/gui/core_layout.py')
        )
        self.core_layout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.core_layout)

    def test_module_import(self):
        """Test that the core layout module can be imported."""
        self.assertIsNotNone(self.core_layout)

    def test_phase_status_constant(self):
        """Test that PHASE_STATUS constant exists and has expected structure."""
        self.assertTrue(hasattr(self.core_layout, 'PHASE_STATUS'))
        phase_status = self.core_layout.PHASE_STATUS
        self.assertIsInstance(phase_status, dict)

    def test_apply_enhanced_theme(self):
        """Test apply_enhanced_theme function."""
        self.core_layout.apply_enhanced_theme()
        
        # Verify that st.markdown was called with CSS styling
        mock_st.markdown.assert_called_once()
        args, kwargs = mock_st.markdown.call_args
        
        # Check that CSS is included in the markdown call
        self.assertIn('<style>', args[0])
        self.assertIn('.main-header', args[0])
        self.assertEqual(kwargs.get('unsafe_allow_html'), True)

    def test_create_phase_header_complete_status(self):
        """Test create_phase_header with complete status."""
        # Mock PHASE_STATUS to return complete status
        with patch.object(self.core_layout, 'PHASE_STATUS', {
            'test_phase': {'status': 'complete', 'progress': 100}
        }):
            self.core_layout.create_phase_header(
                "Test Phase", 
                "Test Description", 
                "test_phase"
            )
        
        # Verify columns were created
        mock_st.columns.assert_called_once_with([3, 1, 1])
        
        # Verify title and caption were called
        mock_st.title.assert_called_once_with("Test Phase")
        mock_st.caption.assert_called_once_with("Test Description")
        
        # Verify success status was shown
        mock_st.success.assert_called_once_with("✅ Complete (100%)")

    def test_create_phase_header_ready_status(self):
        """Test create_phase_header with ready status."""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock() for _ in range(3)]
        
        with patch.object(self.core_layout, 'PHASE_STATUS', {
            'test_phase': {'status': 'ready', 'progress': 75}
        }):
            self.core_layout.create_phase_header(
                "Ready Phase", 
                "Ready Description", 
                "test_phase"
            )
        
        # Verify info status was shown
        mock_st.info.assert_called_once_with("🔄 Ready (75%)")

    def test_create_phase_header_pending_status(self):
        """Test create_phase_header with pending status."""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock() for _ in range(3)]
        
        with patch.object(self.core_layout, 'PHASE_STATUS', {
            'test_phase': {'status': 'pending', 'progress': 25}
        }):
            self.core_layout.create_phase_header(
                "Pending Phase", 
                "Pending Description", 
                "test_phase"
            )
        
        # Verify warning status was shown
        mock_st.warning.assert_called_once_with("⏳ Pending (25%)")

    def test_create_navigation_sidebar(self):
        """Test create_navigation_sidebar function."""
        # Mock sidebar radio to return a page
        mock_st.sidebar.radio.return_value = "🏠 Dashboard"
        
        result = self.core_layout.create_navigation_sidebar()
        
        # Verify sidebar methods were called
        mock_st.sidebar.title.assert_called_once_with("🔬 MFC Platform")
        mock_st.sidebar.radio.assert_called()
        mock_st.sidebar.subheader.assert_called_with("Phase Status")
        
        # Should return the mapped page value
        self.assertEqual(result, "dashboard")

    def test_create_page_layout(self):
        """Test create_page_layout function."""
        # Test page layout creation
        container_mock = MagicMock()
        sidebar_mock = MagicMock()
        mock_st.container.return_value = container_mock
        mock_st.sidebar = sidebar_mock
        
        result = self.core_layout.create_page_layout("Test Page")
        
        # Verify apply_enhanced_theme was called (via markdown)
        mock_st.markdown.assert_called()
        
        # Verify title was set
        mock_st.title.assert_called_once_with("Test Page")
        
        # Should return dictionary with layout components
        self.assertIsInstance(result, dict)
        self.assertIn('header', result)
        self.assertIn('content', result)
        self.assertIn('sidebar', result)

    def test_render_header_with_subtitle(self):
        """Test render_header function with subtitle."""
        self.core_layout.render_header("Test Title", "Test Subtitle")
        
        # Verify title was called
        mock_st.title.assert_called_once_with("Test Title")
        
        # Verify subtitle markdown was called
        mock_st.markdown.assert_called_once_with("*Test Subtitle*")

    def test_render_header_without_subtitle(self):
        """Test render_header function without subtitle."""
        mock_st.reset_mock()
        
        self.core_layout.render_header("Test Title")
        
        # Verify title was called
        mock_st.title.assert_called_once_with("Test Title")
        
        # Verify markdown was not called for subtitle
        mock_st.markdown.assert_not_called()

    def test_all_functions_exist(self):
        """Test that all expected functions exist in the module."""
        expected_functions = [
            'apply_enhanced_theme',
            'create_phase_header',
            'create_navigation_sidebar',
            'create_page_layout',
            'render_header'
        ]
        
        for func_name in expected_functions:
            self.assertTrue(
                hasattr(self.core_layout, func_name),
                f"Function {func_name} should exist in core_layout module"
            )
            self.assertTrue(
                callable(getattr(self.core_layout, func_name)),
                f"{func_name} should be callable"
            )


if __name__ == '__main__':
    unittest.main()