# Test Suite Documentation

## Overview

The MFC Q-learning project includes a comprehensive test suite that validates functionality across multiple domains including path configuration, file outputs, GPU capabilities, and simulation execution. This document describes the test suite organization, test categories, and usage instructions.

## Test Suite Architecture

The test suite is located in `q-learning-mfcs/tests/` and consists of:

```
tests/
├── run_tests.py                   # Main test runner
├── test_path_config.py           # Path configuration tests
├── test_file_outputs.py          # File output integration tests
├── test_actual_executions.py     # Simulation execution tests
├── test_gpu_capability.py        # GPU hardware detection tests
└── test_gpu_acceleration.py      # GPU functionality tests
```

## Running Tests

### From Main Project Directory

```bash
# Run all tests
python q-learning-mfcs/tests/run_tests.py

# Run with verbose output
python q-learning-mfcs/tests/run_tests.py -v

# Run with quiet output
python q-learning-mfcs/tests/run_tests.py -q

# Run specific test class
python q-learning-mfcs/tests/run_tests.py -c <test_class>
```

### Available Test Classes

- `path_config` - Path configuration and directory structure tests
- `file_outputs` - File output and data saving tests
- `imports` - Module import verification tests
- `executions` - Simulation execution tests
- `patterns` - File output pattern tests
- `gpu_capability` - GPU hardware detection tests
- `gpu_acceleration` - GPU acceleration functionality tests

## Test Categories

### 1. Path Configuration Tests (`test_path_config.py`)

Tests the centralized path configuration system:

- **Directory Existence**: Verifies all required directories exist
- **Path Generation**: Tests path functions for figures, data, models, logs, reports
- **Path Consistency**: Ensures paths are consistent across calls
- **Subdirectory Handling**: Tests path functions with subdirectories

Example test:
```python
def test_get_figure_path(self):
    """Test figure path generation."""
    fig_path = get_figure_path("test_plot.png")
    self.assertTrue(fig_path.endswith("data/figures/test_plot.png"))
    self.assertTrue(os.path.exists(os.path.dirname(fig_path)))
```

### 2. File Output Tests (`test_file_outputs.py`)

Tests file creation and data saving functionality:

- **CSV Output**: Validates CSV data saving to correct paths
- **JSON Output**: Tests JSON data serialization and saving
- **Matplotlib Figures**: Tests plot saving functionality
- **Pickle Models**: Validates model serialization
- **Multiple Outputs**: Tests simultaneous output creation

### 3. Import Tests (`test_file_outputs.py`)

Verifies all modules import correctly:

- **Main Simulations**: Tests importing core simulation files
- **Analysis Files**: Validates analysis module imports
- **Utility Files**: Tests utility module imports

### 4. Execution Tests (`test_actual_executions.py`)

Tests actual code execution:

- **Data Generation**: Tests data generation scripts
- **Model Saving**: Validates model saving execution
- **Plotting**: Tests plotting script execution
- **Minimal Simulations**: Quick simulation runs (skipped by default)

### 5. GPU Capability Tests (`test_gpu_capability.py`)

Comprehensive GPU hardware and software detection:

#### NVIDIA Tests
- **Hardware Detection**: Uses `nvidia-smi` to detect NVIDIA GPUs
- **CUDA Availability**: Checks CUDA toolkit installation
- **CuPy Functionality**: Tests CuPy import and GPU operations
- **PyTorch CUDA**: Validates PyTorch CUDA support

#### AMD Tests
- **Hardware Detection**: Uses `rocm-smi` to detect AMD GPUs
- **ROCm Availability**: Checks ROCm installation
- **PyTorch ROCm**: Tests PyTorch with ROCm backend
- **HIP Detection**: Validates HIP compiler availability

#### Framework Tests
- **TensorFlow GPU**: Tests TensorFlow GPU support
- **JAX GPU**: Validates JAX GPU backend
- **Numba CUDA**: Tests Numba CUDA JIT compilation

Example output:
```
test_amd_gpu_hardware ... ok
test_pytorch_rocm_availability ... ok
  ✅ PyTorch ROCm support detected
  Device: Radeon RX 7900 XTX
  HIP Version: 6.0.32831-4b1a4062a
```

### 6. GPU Acceleration Tests (`test_gpu_acceleration.py`)

Tests the universal GPU acceleration module functionality:

#### Core Tests
- **Initialization**: GPU accelerator initialization
- **Backend Detection**: Automatic backend selection
- **Device Info**: Device information retrieval

#### Operation Tests
- **Array Creation**: Array creation on appropriate device
- **Mathematical Operations**: abs, log, exp, sqrt, power
- **Conditional Operations**: where, maximum, minimum, clip
- **Aggregations**: mean, sum
- **Random Generation**: Normal distribution generation

#### Fallback Tests
- **CPU Fallback**: Tests CPU fallback mode
- **Error Handling**: Validates graceful degradation
- **Memory Management**: Tests memory transfer operations

Example test:
```python
def test_mathematical_operations(self):
    """Test mathematical operations work correctly."""
    a = self.gpu_acc.array([1.0, -2.0, 3.0])
    b = self.gpu_acc.array([4.0, 5.0, -6.0])
    
    # Test abs
    abs_result = self.gpu_acc.abs(a)
    expected_abs = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        self.gpu_acc.to_cpu(abs_result), 
        expected_abs, 
        rtol=1e-6
    )
```

## Test Results Summary

A typical test run produces:

```
🧪 MFC Q-Learning Project - Comprehensive Test Suite
============================================================
Running tests from: /home/user/mfc-project/q-learning-mfcs/tests
Source directory: /home/user/mfc-project/q-learning-mfcs/src
============================================================

[Test execution details...]

============================================================
📊 TEST SUMMARY
============================================================
Tests run: 50
Successful: 40
Failures: 0
Errors: 0
Skipped: 10
🎉 ALL TESTS PASSED!
```

## Test Output Structure

The test suite validates that simulation outputs follow the standardized directory structure:

```
q-learning-mfcs/
├── data/
│   ├── figures/          # Plot outputs (.png, .pdf)
│   ├── simulation_data/  # Data files (.csv, .json)
│   └── logs/            # Log files
├── q_learning_models/    # Trained models (.pkl)
└── reports/             # Generated reports
```

## Continuous Integration

The test suite can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pixi install
    - name: Run tests
      run: python q-learning-mfcs/tests/run_tests.py
```

## Writing New Tests

To add new tests:

1. Create test class inheriting from `unittest.TestCase`
2. Add to appropriate test module or create new module
3. Import in `run_tests.py`
4. Add to `create_test_suite()` function

Example:
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_data()
    
    def test_feature_functionality(self):
        """Test new feature works correctly."""
        result = new_feature(self.test_data)
        self.assertEqual(result, expected_value)
```

## Best Practices

1. **Use Descriptive Names**: Test method names should clearly describe what is being tested
2. **Test Isolation**: Each test should be independent and not rely on other tests
3. **Clean Up**: Use `tearDown()` to clean up test artifacts
4. **Mock External Dependencies**: Use mocking for external services or hardware
5. **Test Edge Cases**: Include tests for boundary conditions and error cases

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the source directory is in Python path
2. **GPU Tests Failing**: May indicate missing GPU drivers or libraries
3. **File Permission Errors**: Check write permissions for output directories
4. **Skipped Tests**: Some tests are skipped based on available hardware/software

### Debug Mode

Run tests with increased verbosity for debugging:
```bash
python q-learning-mfcs/tests/run_tests.py -v
```

## Future Enhancements

1. **Performance Benchmarks**: Add timing tests for critical operations
2. **Integration Tests**: Test complete workflows end-to-end
3. **Parametrized Tests**: Use pytest parametrization for test variations
4. **Coverage Reports**: Add code coverage measurement
5. **Parallel Execution**: Run tests in parallel for faster execution