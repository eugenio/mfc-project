#!/usr/bin/env python3
"""
Test suite for GPU acceleration functionality with CPU fallback testing.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
tests_dir = Path(__file__).parent
src_dir = tests_dir.parent / 'src'
sys.path.insert(0, str(src_dir))

from gpu_acceleration import get_gpu_accelerator, GPUAccelerator


class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration functionality and CPU fallback."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.gpu_acc = get_gpu_accelerator()
        
    def test_gpu_accelerator_initialization(self):
        """Test GPU accelerator initializes correctly."""
        self.assertIsInstance(self.gpu_acc, GPUAccelerator)
        self.assertIn(self.gpu_acc.backend, ['cuda', 'rocm', 'cpu'])
        self.assertIsNotNone(self.gpu_acc.device_info)
        
    def test_array_creation(self):
        """Test array creation on appropriate device."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        array = self.gpu_acc.array(data)
        
        # Convert back to CPU for comparison
        cpu_array = self.gpu_acc.to_cpu(array)
        expected = np.array(data, dtype=np.float32)
        
        np.testing.assert_allclose(cpu_array, expected, rtol=1e-6)
        
    def test_mathematical_operations(self):
        """Test mathematical operations work correctly."""
        a = self.gpu_acc.array([1.0, -2.0, 3.0])
        b = self.gpu_acc.array([4.0, 5.0, -6.0])
        
        # Test abs
        abs_result = self.gpu_acc.abs(a)
        expected_abs = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(abs_result), expected_abs, rtol=1e-6)
        
        # Test maximum
        max_result = self.gpu_acc.maximum(a, b)
        expected_max = np.array([4.0, 5.0, 3.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(max_result), expected_max, rtol=1e-6)
        
        # Test minimum
        min_result = self.gpu_acc.minimum(a, b)
        expected_min = np.array([1.0, -2.0, -6.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(min_result), expected_min, rtol=1e-6)
        
    def test_where_operation(self):
        """Test conditional where operation."""
        condition = self.gpu_acc.array([True, False, True])
        x = self.gpu_acc.array([1.0, 2.0, 3.0])
        y = self.gpu_acc.array([4.0, 5.0, 6.0])
        
        result = self.gpu_acc.where(condition, x, y)
        expected = np.array([1.0, 5.0, 3.0])
        
        np.testing.assert_allclose(self.gpu_acc.to_cpu(result), expected, rtol=1e-6)
        
    def test_logarithmic_operations(self):
        """Test logarithmic and exponential operations."""
        data = self.gpu_acc.array([1.0, 2.0, np.e])
        
        # Test log
        log_result = self.gpu_acc.log(data)
        expected_log = np.array([0.0, np.log(2.0), 1.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(log_result), expected_log, rtol=1e-6)
        
        # Test exp
        exp_data = self.gpu_acc.array([0.0, 1.0, 2.0])
        exp_result = self.gpu_acc.exp(exp_data)
        expected_exp = np.array([1.0, np.e, np.e**2])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(exp_result), expected_exp, rtol=1e-6)
        
    def test_clipping_operation(self):
        """Test array clipping operation."""
        data = self.gpu_acc.array([-2.0, 0.5, 1.5, 3.0])
        clipped = self.gpu_acc.clip(data, 0.0, 2.0)
        expected = np.array([0.0, 0.5, 1.5, 2.0])
        
        np.testing.assert_allclose(self.gpu_acc.to_cpu(clipped), expected, rtol=1e-6)
        
    def test_aggregation_operations(self):
        """Test mean and sum operations."""
        data = self.gpu_acc.array([1.0, 2.0, 3.0, 4.0])
        
        # Test mean
        mean_result = self.gpu_acc.mean(data)
        expected_mean = 2.5
        self.assertAlmostEqual(float(self.gpu_acc.to_cpu(mean_result)), expected_mean, places=5)
        
        # Test sum
        sum_result = self.gpu_acc.sum(data)
        expected_sum = 10.0
        self.assertAlmostEqual(float(self.gpu_acc.to_cpu(sum_result)), expected_sum, places=5)
        
    def test_power_operations(self):
        """Test sqrt and power operations."""
        data = self.gpu_acc.array([1.0, 4.0, 9.0, 16.0])
        
        # Test sqrt
        sqrt_result = self.gpu_acc.sqrt(data)
        expected_sqrt = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(sqrt_result), expected_sqrt, rtol=1e-6)
        
        # Test power
        base = self.gpu_acc.array([2.0, 3.0, 4.0])
        power_result = self.gpu_acc.power(base, 2.0)
        expected_power = np.array([4.0, 9.0, 16.0])
        np.testing.assert_allclose(self.gpu_acc.to_cpu(power_result), expected_power, rtol=1e-6)
        
    def test_cpu_fallback_mode(self):
        """Test CPU fallback functionality."""
        # Create a fresh accelerator to test fallback
        gpu_acc_fallback = get_gpu_accelerator(force_reinit=True)
        
        # Force CPU fallback
        gpu_acc_fallback.force_cpu_fallback()
        self.assertEqual(gpu_acc_fallback.backend, 'cpu')
        
        # Test operations still work in CPU mode
        gpu_acc_fallback.array([1.0, 2.0, 3.0])
        result = gpu_acc_fallback.abs(gpu_acc_fallback.array([-1.0, -2.0, 3.0]))
        expected = np.array([1.0, 2.0, 3.0])
        
        np.testing.assert_allclose(gpu_acc_fallback.to_cpu(result), expected, rtol=1e-6)
        
    def test_random_generation(self):
        """Test random number generation."""
        shape = (10,)
        random_array = self.gpu_acc.random_normal(shape, mean=0.0, std=1.0)
        cpu_array = self.gpu_acc.to_cpu(random_array)
        
        # Basic checks for random array
        self.assertEqual(cpu_array.shape, shape)
        self.assertTrue(np.all(np.isfinite(cpu_array)))
        
        # Check that it's actually random (not all zeros)
        self.assertFalse(np.allclose(cpu_array, 0.0))
        
    def test_device_info(self):
        """Test device information retrieval."""
        device_info = self.gpu_acc.device_info
        self.assertIsInstance(device_info, dict)
        self.assertIn('backend', device_info)
        self.assertIn('device_count', device_info)
        
    def test_memory_info(self):
        """Test memory information retrieval (if available)."""
        memory_info = self.gpu_acc.get_memory_info()
        
        if memory_info is not None:
            self.assertIsInstance(memory_info, dict)
            # Memory info structure depends on backend
            
    def test_conversion_operations(self):
        """Test to_cpu and from_cpu conversions."""
        # Test numpy to GPU and back
        cpu_data = np.array([1.0, 2.0, 3.0, 4.0])
        gpu_array = self.gpu_acc.array(cpu_data)
        converted_back = self.gpu_acc.to_cpu(gpu_array)
        
        np.testing.assert_allclose(converted_back, cpu_data, rtol=1e-6)
        
    def test_dtype_handling(self):
        """Test proper dtype handling across backends."""
        # Test different dtypes
        for dtype in [np.float32, np.float64]:
            data = np.array([1.0, 2.0, 3.0], dtype=dtype)
            gpu_array = self.gpu_acc.array(data, dtype=dtype)
            result = self.gpu_acc.to_cpu(gpu_array)
            
            # Check dtype is preserved (or compatible)
            self.assertTrue(np.issubdtype(result.dtype, np.floating))
            np.testing.assert_allclose(result, data, rtol=1e-6)


if __name__ == '__main__':
    # Set matplotlib backend to avoid Qt issues
    import matplotlib
    matplotlib.use('Agg')
    
    unittest.main(verbosity=2)