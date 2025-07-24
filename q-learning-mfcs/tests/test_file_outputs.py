#!/usr/bin/env python3
"""
Unit tests for verifying that source files output to correct paths.
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from path_config import (
    get_figure_path, get_simulation_data_path, get_model_path, 
    get_report_path, get_log_path
)


class TestFileOutputIntegration(unittest.TestCase):
    """Test that files can create outputs in correct locations."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_files_created = []
        
    def tearDown(self):
        """Clean up test files."""
        for filepath in self.test_files_created:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
    
    def track_file(self, filepath):
        """Track a file for cleanup."""
        self.test_files_created.append(filepath)
        return filepath
    
    def test_matplotlib_figure_output(self):
        """Test that matplotlib figures can be saved to correct paths."""
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Save using path_config
        filename = 'test_matplotlib_output.png'
        filepath = get_figure_path(filename)
        self.track_file(filepath)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        self.assertIn('data/figures', filepath)
    
    def test_csv_data_output(self):
        """Test CSV data output to correct paths."""
        # Create test data
        data = {
            'time': np.arange(0, 100, 1),
            'power': np.random.uniform(0.1, 0.5, 100),
            'voltage': np.random.uniform(0.3, 0.8, 100),
            'current': np.random.uniform(0.1, 0.6, 100)
        }
        df = pd.DataFrame(data)
        
        # Save CSV
        filename = 'test_csv_output.csv'
        filepath = get_simulation_data_path(filename)
        self.track_file(filepath)
        
        df.to_csv(filepath, index=False)
        
        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        self.assertIn('data/simulation_data', filepath)
        
        # Verify content
        df_loaded = pd.read_csv(filepath)
        self.assertEqual(len(df_loaded), 100)
        self.assertListEqual(list(df_loaded.columns), ['time', 'power', 'voltage', 'current'])
    
    def test_json_data_output(self):
        """Test JSON data output to correct paths."""
        # Create test data
        test_data = {
            'simulation_parameters': {
                'duration_hours': 100,
                'timestep_seconds': 10,
                'num_cells': 5
            },
            'results': {
                'avg_power': 0.25,
                'max_voltage': 0.8,
                'substrate_utilization': 0.85
            },
            'metadata': {
                'timestamp': '2025-01-24T12:00:00',
                'version': '1.0'
            }
        }
        
        # Save JSON
        filename = 'test_json_output.json'
        filepath = get_simulation_data_path(filename)
        self.track_file(filepath)
        
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        self.assertIn('data/simulation_data', filepath)
        
        # Verify content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data['simulation_parameters']['duration_hours'], 100)
        self.assertEqual(loaded_data['results']['avg_power'], 0.25)
    
    def test_pickle_model_output(self):
        """Test pickle model output to correct paths."""
        # Create mock Q-learning model
        from collections import defaultdict
        
        q_table = defaultdict(lambda: defaultdict(float))
        q_table['state_0_0_0']['action_0'] = 0.5
        q_table['state_0_0_0']['action_1'] = 0.3
        q_table['state_1_1_1']['action_0'] = 0.7
        q_table['state_1_1_1']['action_1'] = 0.9
        
        model_data = {
            'q_table': dict(q_table),
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 0.1,
            'total_episodes': 1000
        }
        
        # Save model
        filename = 'test_q_model.pkl'
        filepath = get_model_path(filename)
        self.track_file(filepath)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Verify file was created and has correct content
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
        self.assertIn('q_learning_models', filepath)
        
        # Verify content
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        self.assertEqual(loaded_model['learning_rate'], 0.1)
        self.assertEqual(loaded_model['total_episodes'], 1000)
        self.assertIn('q_table', loaded_model)
    
    def test_multiple_output_types(self):
        """Test creating multiple output types simultaneously."""
        base_name = 'multi_output_test'
        
        # Create figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        fig_path = get_figure_path(f'{base_name}.png')
        self.track_file(fig_path)
        plt.savefig(fig_path)
        plt.close()
        
        # Create CSV data
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 2]})
        csv_path = get_simulation_data_path(f'{base_name}.csv')
        self.track_file(csv_path)
        df.to_csv(csv_path, index=False)
        
        # Create JSON data
        json_data = {'test': True, 'values': [1, 2, 3]}
        json_path = get_simulation_data_path(f'{base_name}.json')
        self.track_file(json_path)
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        
        # Create model
        model = {'weights': [0.1, 0.2, 0.3]}
        model_path = get_model_path(f'{base_name}.pkl')
        self.track_file(model_path)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Verify all files exist
        for path in [fig_path, csv_path, json_path, model_path]:
            self.assertTrue(os.path.exists(path), f"File should exist: {path}")
            self.assertGreater(os.path.getsize(path), 0, f"File should not be empty: {path}")


class TestSpecificFileImports(unittest.TestCase):
    """Test that specific source files can be imported without errors."""
    
    def setUp(self):
        """Set up test environment."""
        # Suppress GPU warnings and other output during testing
        import warnings
        warnings.filterwarnings("ignore")
    
    def test_import_main_simulation_files(self):
        """Test importing main simulation files."""
        main_files = [
            'mfc_unified_qlearning_control',
            'mfc_qlearning_optimization', 
            'mfc_qlearning_optimization_parallel',
            'mfc_dynamic_substrate_control'
        ]
        
        for module_name in main_files:
            with self.subTest(module=module_name):
                try:
                    module = __import__(module_name)
                    self.assertIsNotNone(module)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")
    
    def test_import_analysis_files(self):
        """Test importing analysis and visualization files."""
        analysis_files = [
            'generate_performance_graphs',
            'physics_accurate_biofilm_qcm',
            'eis_qcm_biofilm_correlation',
            'energy_sustainability_analysis',
            'create_summary_plots'
        ]
        
        for module_name in analysis_files:
            with self.subTest(module=module_name):
                try:
                    module = __import__(module_name)
                    self.assertIsNotNone(module)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")
    
    def test_import_utility_files(self):
        """Test importing utility files."""
        utility_files = [
            'stack_physical_specs',
            'path_config'
            # Note: mfc_model excluded due to Mojo dependencies
        ]
        
        for module_name in utility_files:
            with self.subTest(module=module_name):
                try:
                    module = __import__(module_name)
                    self.assertIsNotNone(module)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    unittest.main()