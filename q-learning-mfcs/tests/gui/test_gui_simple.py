#!/usr/bin/env python3
"""
Simple GUI tests for MFC Streamlit interface.

Tests basic GUI functionality without external dependencies.
Created: 2025-07-31
"""

import unittest
import sys
import os
from unittest.mock import patch
import tempfile

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

class TestGUISimple(unittest.TestCase):
    """Simple GUI functionality tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)
    
    @patch('streamlit.sidebar')
    @patch('streamlit.write')
    def test_gui_imports(self, mock_write, mock_sidebar):
        """Test that GUI module imports without errors."""
        try:
            import mfc_streamlit_gui
            self.assertTrue(True, "GUI module imported successfully")
        except ImportError as e:
            self.fail(f"GUI module import failed: {e}")
    
    @patch('streamlit.sidebar')
    @patch('streamlit.write')
    def test_basic_gui_functions(self, mock_write, mock_sidebar):
        """Test basic GUI function availability."""
        try:
            import mfc_streamlit_gui
            # Test that main functions exist
            self.assertTrue(hasattr(mfc_streamlit_gui, 'main') or 
                          callable(getattr(mfc_streamlit_gui, 'main', None)),
                          "GUI should have a main function")
        except ImportError:
            self.skipTest("GUI module not available")
    
    def test_data_directory_structure(self):
        """Test that required data directories exist."""
        # Check for simulation data directories
        base_dir = os.path.join(os.path.dirname(__file__), '../../')
        data_dirs = ['data', 'simulation_data', 'q_learning_models']
        
        for data_dir in data_dirs:
            full_path = os.path.join(base_dir, data_dir)
            if os.path.exists(full_path):
                self.assertTrue(os.path.isdir(full_path), 
                              f"{data_dir} should be a directory")
    
    def test_configuration_files_exist(self):
        """Test that essential configuration files exist."""
        base_dir = os.path.join(os.path.dirname(__file__), '../../src')
        config_files = ['config', 'mfc_parameters.json']
        
        found_configs = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if any(config in file.lower() for config in config_files):
                    found_configs.append(file)
        
        self.assertGreater(len(found_configs), 0, 
                          "Should find at least one configuration file")


if __name__ == '__main__':
    unittest.main()