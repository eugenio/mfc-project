#!/usr/bin/env python3
"""
Unit tests for path_config module functionality.
"""

import unittest
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from path_config import (
    get_figure_path, get_simulation_data_path, get_model_path,
    get_report_path, get_log_path, FIGURES_DIR, SIMULATION_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, LOGS_DIR, PROJECT_ROOT
)


class TestPathConfig(unittest.TestCase):
    """Test path configuration functions and directory structure."""

    def test_project_root_exists(self):
        """Test that project root is correctly identified."""
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue((PROJECT_ROOT / "src").exists())

    def test_directories_exist(self):
        """Test that all required directories exist."""
        directories = [FIGURES_DIR, SIMULATION_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]
        for directory in directories:
            with self.subTest(directory=directory):
                self.assertTrue(directory.exists(), f"Directory {directory} should exist")
                self.assertTrue(directory.is_dir(), f"{directory} should be a directory")

    def test_get_figure_path(self):
        """Test figure path generation."""
        test_filename = 'test_plot.png'
        path = get_figure_path(test_filename)

        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(test_filename))
        self.assertIn('data/figures', path)
        self.assertTrue(os.path.isabs(path))

    def test_get_simulation_data_path(self):
        """Test simulation data path generation."""
        test_filename = 'test_data.csv'
        path = get_simulation_data_path(test_filename)

        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(test_filename))
        self.assertIn('data/simulation_data', path)
        self.assertTrue(os.path.isabs(path))

    def test_get_model_path(self):
        """Test model path generation."""
        test_filename = 'test_model.pkl'
        path = get_model_path(test_filename)

        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(test_filename))
        self.assertIn('q_learning_models', path)
        self.assertTrue(os.path.isabs(path))

    def test_get_report_path(self):
        """Test report path generation."""
        test_filename = 'test_report.pdf'
        path = get_report_path(test_filename)

        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(test_filename))
        self.assertIn('reports', path)
        self.assertTrue(os.path.isabs(path))

    def test_get_log_path(self):
        """Test log path generation."""
        test_filename = 'test.log'
        path = get_log_path(test_filename)

        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(test_filename))
        self.assertIn('data/logs', path)
        self.assertTrue(os.path.isabs(path))

    def test_path_consistency(self):
        """Test that paths are consistent across calls."""
        filename = 'consistent_test.png'
        path1 = get_figure_path(filename)
        path2 = get_figure_path(filename)

        self.assertEqual(path1, path2)

    def test_path_functions_with_subdirectories(self):
        """Test path functions with subdirectories in filename."""
        filename = 'subdir/test_file.png'
        path = get_figure_path(filename)

        self.assertTrue(path.endswith(filename))
        self.assertIn('data/figures', path)


if __name__ == '__main__':
    unittest.main()
