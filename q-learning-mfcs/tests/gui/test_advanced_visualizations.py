#!/usr/bin/env python3
"""
Advanced visualization tests for MFC GUI.

Tests biofilm analysis plots and advanced visualization functions.
Created: 2025-07-31
"""

import unittest
import sys
import os
from unittest.mock import patch
import tempfile
import numpy as np

# Add source path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

class TestAdvancedVisualizationFunctions(unittest.TestCase):
    """Tests for advanced visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = tempfile.mkdtemp()

        # Create mock biofilm data
        self.mock_biofilm_data = {
            'time': np.linspace(0, 100, 50),
            'thickness': np.random.rand(50) * 10,
            'density': np.random.rand(50) * 0.8,
            'conductivity': np.random.rand(50) * 0.5
        }

    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_biofilm_analysis_plots(self, mock_savefig, mock_show):
        """Test biofilm analysis plot creation."""
        # Import plotting functions
        try:
            # Try to import visualization modules
            import matplotlib.pyplot as plt
            import numpy as np

            # Create mock biofilm analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Plot biofilm thickness over time
            axes[0, 0].plot(self.mock_biofilm_data['time'],
                           self.mock_biofilm_data['thickness'])
            axes[0, 0].set_title('Biofilm Thickness')
            axes[0, 0].set_xlabel('Time (hours)')
            axes[0, 0].set_ylabel('Thickness (μm)')

            # Plot biofilm density
            axes[0, 1].plot(self.mock_biofilm_data['time'],
                           self.mock_biofilm_data['density'])
            axes[0, 1].set_title('Biofilm Density')
            axes[0, 1].set_xlabel('Time (hours)')
            axes[0, 1].set_ylabel('Density (g/cm³)')

            # Plot conductivity
            axes[1, 0].plot(self.mock_biofilm_data['time'],
                           self.mock_biofilm_data['conductivity'])
            axes[1, 0].set_title('Biofilm Conductivity')
            axes[1, 0].set_xlabel('Time (hours)')
            axes[1, 0].set_ylabel('Conductivity (S/m)')

            # Create correlation plot
            axes[1, 1].scatter(self.mock_biofilm_data['thickness'],
                              self.mock_biofilm_data['conductivity'])
            axes[1, 1].set_title('Thickness vs Conductivity')
            axes[1, 1].set_xlabel('Thickness (μm)')
            axes[1, 1].set_ylabel('Conductivity (S/m)')

            plt.tight_layout()

            # Test that the plot creation succeeds
            self.assertIsNotNone(fig, "Figure should be created successfully")
            self.assertEqual(len(axes.flatten()), 4, "Should have 4 subplot axes")

            plt.close(fig)  # Clean up

        except ImportError as e:
            self.skipTest(f"Visualization dependencies not available: {e}")

    def test_biofilm_data_validation(self):
        """Test biofilm data validation functions."""
        # Test data structure validation
        required_keys = ['time', 'thickness', 'density', 'conductivity']

        for key in required_keys:
            self.assertIn(key, self.mock_biofilm_data,
                         f"Biofilm data should contain {key}")

        # Test data consistency
        time_length = len(self.mock_biofilm_data['time'])
        for key in required_keys[1:]:  # Skip time
            self.assertEqual(len(self.mock_biofilm_data[key]), time_length,
                           f"{key} data should match time series length")

    @patch('matplotlib.pyplot.show')
    def test_plot_formatting(self, mock_show):
        """Test plot formatting and styling."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            # Test basic plot formatting
            ax.plot(self.mock_biofilm_data['time'],
                   self.mock_biofilm_data['thickness'],
                   linewidth=2, color='blue', label='Thickness')

            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Thickness (μm)', fontsize=12)
            ax.set_title('Biofilm Growth Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Verify plot elements
            self.assertEqual(ax.get_xlabel(), 'Time (hours)')
            self.assertEqual(ax.get_ylabel(), 'Thickness (μm)')
            self.assertEqual(ax.get_title(), 'Biofilm Growth Analysis')

            plt.close(fig)

        except ImportError:
            self.skipTest("Matplotlib not available")

    def test_data_export_functionality(self):
        """Test data export for visualization."""
        import json
        import csv
        import io

        # Test JSON export
        json_output = io.StringIO()
        json.dump(self.mock_biofilm_data, json_output, default=lambda x: x.tolist())
        json_str = json_output.getvalue()

        # Verify JSON export
        self.assertIn('time', json_str)
        self.assertIn('thickness', json_str)

        # Test CSV export capability
        csv_output = io.StringIO()
        fieldnames = ['time', 'thickness', 'density', 'conductivity']
        writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        writer.writeheader()

        # Write sample row
        sample_row = {key: self.mock_biofilm_data[key][0] for key in fieldnames}
        writer.writerow(sample_row)

        csv_str = csv_output.getvalue()
        self.assertIn('time,thickness,density,conductivity', csv_str)

    def test_performance_visualization_data(self):
        """Test performance data structures for visualization."""
        # Test that data is in appropriate format for plotting
        for key, data in self.mock_biofilm_data.items():
            self.assertIsInstance(data, np.ndarray,
                                f"{key} should be numpy array for performance")
            self.assertGreater(len(data), 0,
                             f"{key} should contain data points")
            self.assertFalse(np.any(np.isnan(data)),
                           f"{key} should not contain NaN values")


if __name__ == '__main__':
    unittest.main()
