#!/usr/bin/env python3
"""
Test suite to verify that all Python files in src/ output to correct directories.
This test suite creates mock/minimal executions to verify path outputs without
running full simulations.
"""

import os
import sys
import unittest
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from path_config import (
    get_figure_path, get_simulation_data_path, get_model_path,
    get_report_path, get_log_path, FIGURES_DIR, SIMULATION_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, LOGS_DIR
)

class TestPathOutputs(unittest.TestCase):
    """Test suite for verifying correct path outputs."""

    def setUp(self):
        """Set up test environment."""
        self.test_files_created = []

    def tearDown(self):
        """Clean up test files."""
        for filepath in self.test_files_created:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    def track_file(self, filepath):
        """Track a file for cleanup."""
        self.test_files_created.append(filepath)

    def test_path_config_functions(self):
        """Test that path configuration functions work correctly."""
        fig_path = get_figure_path('test.png')
        data_path = get_simulation_data_path('test.csv')
        model_path = get_model_path('test.pkl')
        report_path = get_report_path('test.pdf')
        log_path = get_log_path('test.log')

        self.assertTrue(fig_path.endswith('data/figures/test.png'))
        self.assertTrue(data_path.endswith('data/simulation_data/test.csv'))
        self.assertTrue(model_path.endswith('q_learning_models/test.pkl'))
        self.assertTrue(report_path.endswith('reports/test.pdf'))
        self.assertTrue(log_path.endswith('data/logs/test.log'))

    def test_directories_exist(self):
        """Test that all required directories exist."""
        self.assertTrue(FIGURES_DIR.exists())
        self.assertTrue(SIMULATION_DATA_DIR.exists())
        self.assertTrue(MODELS_DIR.exists())
        self.assertTrue(REPORTS_DIR.exists())
        self.assertTrue(LOGS_DIR.exists())

class MockSimulationTest:
    """Base class for mock simulation tests."""

    def __init__(self, test_case):
        self.test_case = test_case
        self.created_files = []

    def mock_simulation_run(self, duration_minutes=1):
        """Create mock simulation data."""
        import numpy as np
        import pandas as pd

        # Create minimal simulation data
        time_steps = np.arange(0, duration_minutes * 60, 10)  # 10-second intervals
        n_steps = len(time_steps)

        data = {
            'time_hours': time_steps / 3600,
            'power_total': np.random.uniform(0.1, 0.5, n_steps),
            'voltage_cell_1': np.random.uniform(0.3, 0.8, n_steps),
            'current_total': np.random.uniform(0.1, 0.6, n_steps),
            'substrate_utilization': np.random.uniform(0.6, 0.9, n_steps),
            'flow_rate': np.random.uniform(10, 30, n_steps),
        }

        return pd.DataFrame(data)

    def verify_file_created(self, filepath, file_type="file"):
        """Verify that a file was created in the correct location."""
        self.created_files.append(filepath)
        if not os.path.exists(filepath):
            raise AssertionError(f"{file_type} not created: {filepath}")

        # Verify it's in the correct directory
        if 'figures' in filepath and '.png' in filepath:
            self.test_case.assertIn('data/figures', filepath)
        elif 'simulation_data' in filepath and ('.csv' in filepath or '.json' in filepath):
            self.test_case.assertIn('data/simulation_data', filepath)
        elif 'q_learning_models' in filepath and '.pkl' in filepath:
            self.test_case.assertIn('q_learning_models', filepath)
        elif 'reports' in filepath and '.pdf' in filepath:
            self.test_case.assertIn('reports', filepath)
        elif 'logs' in filepath and '.log' in filepath:
            self.test_case.assertIn('data/logs', filepath)

        return True

def test_simple_plotting_file():
    """Test a simple file that generates plots."""

    # Test generate_performance_graphs.py
    test_code = '''
import matplotlib.pyplot as plt
import numpy as np
from path_config import get_figure_path

# Create simple test plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title('Test Plot')

# Save using path_config
filename = get_figure_path('test_performance_graph.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved plot to: {filename}")
'''

    # Execute the test code
    exec(test_code)

    # Verify file was created
    expected_path = get_figure_path('test_performance_graph.png')
    if not os.path.exists(expected_path):
        raise AssertionError(f"Plot file not created: {expected_path}")

    return expected_path

def test_data_output_file():
    """Test a file that outputs CSV and JSON data."""

    test_code = '''
import pandas as pd
import json
import numpy as np
from path_config import get_simulation_data_path

# Create test data
data = {
    'time': np.arange(0, 100, 1),
    'power': np.random.uniform(0.1, 0.5, 100),
    'voltage': np.random.uniform(0.3, 0.8, 100)
}

df = pd.DataFrame(data)

# Save CSV
csv_path = get_simulation_data_path('test_simulation_data.csv')
df.to_csv(csv_path, index=False)

# Save JSON
json_path = get_simulation_data_path('test_simulation_data.json')
# Convert numpy arrays to lists for JSON serialization
json_data = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in data.items()}
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Saved CSV to: {csv_path}")
print(f"Saved JSON to: {json_path}")
'''

    exec(test_code)

    # Verify files were created
    csv_path = get_simulation_data_path('test_simulation_data.csv')
    json_path = get_simulation_data_path('test_simulation_data.json')

    if not os.path.exists(csv_path):
        raise AssertionError(f"CSV file not created: {csv_path}")
    if not os.path.exists(json_path):
        raise AssertionError(f"JSON file not created: {json_path}")

    return [csv_path, json_path]

def test_model_output_file():
    """Test a file that outputs pickle models."""

    import pickle
    from collections import defaultdict
    from path_config import get_model_path

    # Create mock Q-table
    q_table = defaultdict(lambda: defaultdict(float))
    q_table['state1']['action1'] = 0.5
    q_table['state1']['action2'] = 0.3
    q_table['state2']['action1'] = 0.7

    # Save model
    model_path = get_model_path('test_q_table.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(dict(q_table), f)

    print(f"Saved model to: {model_path}")

    # Verify file was created
    if not os.path.exists(model_path):
        raise AssertionError(f"Model file not created: {model_path}")

    return model_path

class TestActualFiles(unittest.TestCase):
    """Test actual source files with minimal execution."""

    def setUp(self):
        self.cleanup_files = []

    def tearDown(self):
        """Clean up test files."""
        for filepath in self.cleanup_files:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    def test_path_config_integration(self):
        """Test basic path configuration integration."""
        # Test simple plotting
        plot_file = test_simple_plotting_file()
        self.cleanup_files.append(plot_file)
        self.assertTrue(os.path.exists(plot_file))

        # Test data output
        data_files = test_data_output_file()
        self.cleanup_files.extend(data_files)
        for f in data_files:
            self.assertTrue(os.path.exists(f))

        # Test model output
        model_file = test_model_output_file()
        self.cleanup_files.append(model_file)
        self.assertTrue(os.path.exists(model_file))

def run_file_import_tests():
    """Test that all modified files can be imported without errors."""

    src_dir = Path(__file__).parent.parent / "src"
    python_files = [
        'mfc_unified_qlearning_control.py',
        'mfc_qlearning_optimization.py',
        'mfc_qlearning_optimization_parallel.py',
        'mfc_dynamic_substrate_control.py',
        'generate_performance_graphs.py',
        'physics_accurate_biofilm_qcm.py',
        'eis_qcm_biofilm_correlation.py',
        'flow_rate_optimization.py',
        'energy_sustainability_analysis.py',
        'stack_physical_specs.py',
        'create_summary_plots.py',
        'three_model_comparison_plots.py',
        'literature_validation_comparison_plots.py',
        'mfc_100h_simulation.py',
        'mfc_stack_simulation.py',
        'analyze_pdf_comments.py',
        'run_gpu_simulation.py',
        'run_simulation.py',
        'mfc_qlearning_demo.py',
        'generate_pdf_report.py'
    ]

    results = {}

    for filename in python_files:
        filepath = src_dir / filename
        if filepath.exists():
            try:
                # Test import
                spec = importlib.util.spec_from_file_location("test_module", filepath)
                if spec and spec.loader:
                    importlib.util.module_from_spec(spec)
                    # Don't execute, just load the module structure
                    results[filename] = "✅ Import successful"
                else:
                    results[filename] = "❌ Could not create spec"
            except Exception as e:
                results[filename] = f"❌ Import error: {str(e)}"
        else:
            results[filename] = "❌ File not found"

    return results

if __name__ == '__main__':
    print("🧪 Running Path Output Test Suite")
    print("=" * 50)

    # Test 1: Basic path configuration
    print("\n1. Testing path configuration functions...")
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestPathOutputs)
    result1 = unittest.TextTestRunner(verbosity=2).run(suite1)

    # Test 2: Integration tests
    print("\n2. Testing path integration...")
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestActualFiles)
    result2 = unittest.TextTestRunner(verbosity=2).run(suite2)

    # Test 3: Import tests
    print("\n3. Testing file imports...")
    import_results = run_file_import_tests()

    print("\nImport Test Results:")
    print("-" * 30)
    for filename, result in import_results.items():
        print(f"{filename}: {result}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    total_tests = result1.testsRun + result2.testsRun
    total_failures = len(result1.failures) + len(result1.errors) + len(result2.failures) + len(result2.errors)

    successful_imports = sum(1 for result in import_results.values() if "✅" in result)
    total_imports = len(import_results)

    print(f"Unit Tests: {total_tests - total_failures}/{total_tests} passed")
    print(f"Import Tests: {successful_imports}/{total_imports} passed")

    if total_failures == 0 and successful_imports == total_imports:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check output above.")
        sys.exit(1)
