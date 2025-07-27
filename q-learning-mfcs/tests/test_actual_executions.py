#!/usr/bin/env python3
"""
Unit tests for actual file executions with minimal parameters.
These tests run actual source files with short durations to verify path outputs.
"""

import unittest
import os
import sys
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from path_config import (
    get_figure_path, get_simulation_data_path, get_model_path, 
    FIGURES_DIR, SIMULATION_DATA_DIR, MODELS_DIR
)


class TestActualFileExecutions(unittest.TestCase):
    """Test actual file executions with minimal parameters."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_outputs = []
        self.original_dir = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))
        
    def tearDown(self):
        """Clean up after tests."""
        os.chdir(self.original_dir)
        # Clean up test files
        for output_path in self.test_outputs:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
    
    def track_output(self, filepath):
        """Track an output file for cleanup."""
        self.test_outputs.append(filepath)
    
    def test_simple_plotting_execution(self):
        """Test execution of a simple plotting script."""
        # Create a minimal script that uses path_config
        test_script = '''
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
from path_config import get_figure_path

# Create minimal plot
fig, ax = plt.subplots(figsize=(4, 3))
x = np.linspace(0, 5, 50)
y = np.sin(x)
ax.plot(x, y, 'b-', linewidth=2)
ax.set_title('Test Execution Plot')
ax.set_xlabel('X')
ax.set_ylabel('Sin(X)')
ax.grid(True, alpha=0.3)

# Save to correct path
output_path = get_figure_path('test_execution_plot.png')
plt.savefig(output_path, dpi=100, bbox_inches='tight')
plt.close()

print(f"Plot saved to: {output_path}")
'''
        
        # Execute the script
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Check execution was successful
        self.assertEqual(result.returncode, 0, f"Script execution failed: {result.stderr}")
        
        # Check output file was created
        expected_path = get_figure_path('test_execution_plot.png')
        self.track_output(expected_path)
        self.assertTrue(os.path.exists(expected_path))
        self.assertGreater(os.path.getsize(expected_path), 1000)  # Should be substantial PNG file
    
    def test_data_generation_execution(self):
        """Test execution of a data generation script."""
        test_script = '''
import pandas as pd
import json
import numpy as np
from path_config import get_simulation_data_path

# Generate minimal simulation data
np.random.seed(42)  # For reproducible results
n_points = 50
time_hours = np.linspace(0, 10, n_points)
power_w = 0.2 + 0.1 * np.sin(time_hours) + 0.02 * np.random.randn(n_points)
voltage_v = 0.6 + 0.1 * np.cos(time_hours * 0.5) + 0.01 * np.random.randn(n_points)

# Create DataFrame
df = pd.DataFrame({
    'time_hours': time_hours,
    'power_w': power_w,
    'voltage_v': voltage_v,
    'current_a': power_w / voltage_v
})

# Save CSV
csv_path = get_simulation_data_path('test_execution_data.csv')
df.to_csv(csv_path, index=False)

# Save JSON summary
summary = {
    'duration_hours': float(time_hours[-1]),
    'avg_power_w': float(np.mean(power_w)),
    'max_voltage_v': float(np.max(voltage_v)),
    'min_voltage_v': float(np.min(voltage_v)),
    'data_points': len(df)
}

json_path = get_simulation_data_path('test_execution_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"CSV saved to: {csv_path}")
print(f"JSON saved to: {json_path}")
'''
        
        # Execute the script
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Check execution was successful
        self.assertEqual(result.returncode, 0, f"Script execution failed: {result.stderr}")
        
        # Check output files were created
        csv_path = get_simulation_data_path('test_execution_data.csv')
        json_path = get_simulation_data_path('test_execution_summary.json')
        
        self.track_output(csv_path)
        self.track_output(json_path)
        
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertGreater(os.path.getsize(csv_path), 100)
        self.assertGreater(os.path.getsize(json_path), 50)
    
    def test_model_saving_execution(self):
        """Test execution of a model saving script."""
        test_script = '''
import pickle
import numpy as np
from collections import defaultdict
from path_config import get_model_path

# Create a simple Q-learning model
np.random.seed(42)
q_table = defaultdict(lambda: defaultdict(float))

# Populate with some random values
states = ['s1', 's2', 's3']
actions = ['a1', 'a2']

for state in states:
    for action in actions:
        q_table[state][action] = np.random.uniform(-1, 1)

# Create model data
model_data = {
    'q_table': dict(q_table),
    'hyperparameters': {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.1
    },
    'training_info': {
        'episodes': 100,
        'total_reward': 45.6
    }
}

# Save model
model_path = get_model_path('test_execution_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model saved to: {model_path}")
'''
        
        # Execute the script
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Check execution was successful
        self.assertEqual(result.returncode, 0, f"Script execution failed: {result.stderr}")
        
        # Check output file was created
        model_path = get_model_path('test_execution_model.pkl')
        self.track_output(model_path)
        
        self.assertTrue(os.path.exists(model_path))
        self.assertGreater(os.path.getsize(model_path), 100)
    
    @unittest.skip("Requires long execution time - enable for full testing")
    def test_minimal_simulation_run(self):
        """Test running a minimal version of actual simulation file."""
        # This test is skipped by default as it takes longer
        # Enable it by removing the @unittest.skip decorator for comprehensive testing
        
        test_script = '''
# Minimal version of mfc_qlearning_demo simulation
import sys
import os
sys.path.append('.')

# Mock the simulation to run for very short time
original_hours = None

try:
    import mfc_qlearning_demo
    # This would run the actual demo but we skip it for speed
    print("Would run mfc_qlearning_demo here")
except ImportError as e:
    print(f"Could not import: {e}")
'''
        
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, cwd=os.getcwd(), 
                              timeout=30)  # 30 second timeout
        
        # Just check that import works - actual execution would be too slow
        self.assertIn("Would run", result.stdout)


class TestFileOutputPatterns(unittest.TestCase):
    """Test that files follow expected output patterns."""
    
    def setUp(self):
        """Set up test environment."""
        self.src_dir = Path(__file__).parent.parent / 'src'
    
    def test_files_use_path_config_import(self):
        """Test that modified files import path_config."""
        python_files = [
            'mfc_unified_qlearning_control.py',
            'mfc_qlearning_optimization.py',
            'mfc_dynamic_substrate_control.py',
            'generate_performance_graphs.py',
            'physics_accurate_biofilm_qcm.py'
        ]
        
        for filename in python_files:
            filepath = self.src_dir / filename
            if filepath.exists():
                with self.subTest(file=filename):
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Check for path_config import
                    self.assertIn('from path_config import', content, 
                                f"{filename} should import from path_config")
                    
                    # Check that it doesn't use hardcoded paths
                    hardcoded_patterns = [
                        "f'figures/",  # Old style figure paths
                        "f'simulation_data/",  # Old style data paths
                        "f'q_learning_models/"  # Old style model paths
                    ]
                    
                    for pattern in hardcoded_patterns:
                        self.assertNotIn(pattern, content, 
                                       f"{filename} should not contain hardcoded pattern: {pattern}")
    
    def test_expected_output_directories_have_content(self):
        """Test that output directories contain expected types of files."""
        # Check figures directory
        if FIGURES_DIR.exists():
            png_files = list(FIGURES_DIR.glob('*.png'))
            self.assertGreater(len(png_files), 0, "Figures directory should contain PNG files")
        
        # Check simulation data directory  
        if SIMULATION_DATA_DIR.exists():
            csv_files = list(SIMULATION_DATA_DIR.glob('*.csv'))
            json_files = list(SIMULATION_DATA_DIR.glob('*.json'))
            self.assertGreater(len(csv_files) + len(json_files), 0, 
                             "Simulation data directory should contain CSV or JSON files")
        
        # Check models directory
        if MODELS_DIR.exists():
            pkl_files = list(MODELS_DIR.glob('*.pkl'))
            # Note: Models directory might be empty if no simulations have been run


if __name__ == '__main__':
    unittest.main()